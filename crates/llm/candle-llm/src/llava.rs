//! LLaVA vision-language model, served through the engine's multimodal contract (story 7262 — the
//! Candle port of mlx-llm's 7157 JoyCaption VLM, generalized to any SigLIP-based LLaVA checkpoint).
//!
//! A `LlavaForConditionalGeneration` checkpoint is a SigLIP vision tower
//! ([`crate::models::SiglipVisionTower`]) that encodes the image, a two-layer GELU MLP projector
//! that lifts a chosen penultimate-layer hidden state into the language hidden size, and a generic
//! causal decoder ([`CausalLm`], reused as-is — any architecture the config dispatches). The
//! projected patch rows replace the expanded image-token placeholders in the prompt embeddings (the
//! [`CausalLm::decode_logits_from_embeds`] splice hook), then the decoder generates the caption.
//!
//! Geometry is read from the checkpoint's `config.json` (`vision_config`, `text_config`,
//! `image_token_index`, `vision_feature_layer`, `vision_feature_select_strategy`), so the same code
//! loads JoyCaption (SigLIP2-so400m + Llama-3.1) or the smaller llava-* checkpoints.
//!
//! Numerics mirror the reference: the vision tower + projector run in **f32** against the
//! f32-preprocessed pixels (the bf16 weights are promoted on load), then the projected features are
//! cast to the decoder's compute dtype (bf16 on GPU) and spliced into the token embeddings before
//! the decode.

use std::path::Path;

use candle_core::{Device, Tensor};
use serde_json::Value;

use core_llm::{
    Channel, ChatTemplate, Content, Error as CoreError, FinishReason as CoreFinish,
    IncrementalDetok, JinjaChatTemplate, Llama3Template, LoadSpec, Message, Quantize,
    Result as CoreResult, Sampling, StreamEvent as CoreEvent, TextLlm, TextLlmCapabilities,
    TextLlmDescriptor, TextLlmOutput, TextLlmRequest, Tokenizer, Usage,
};

use crate::config::{Architecture, ModelConfig};
use crate::decode::{
    generate_step_from_prefill, CancelFlag, DecodeRecord, FinishReason, GenerationConfig,
    RequestSpan, StreamEvent,
};
use crate::device::select_eager_device;
use crate::error::{Error, Result};
use crate::image::SiglipImageProcessor;
use crate::models::siglip::{select_vision_feature, SiglipVisionConfig, SiglipVisionTower};
use crate::models::CausalLm;
use crate::primitives::nn::{gelu, gelu_erf, linear};
use crate::primitives::projection::QuantSpec;
use crate::primitives::sampler::SamplingParams;
use crate::primitives::{input_ids, AttnFormulation, Weights};

/// The registry id of the LLaVA provider.
pub const PROVIDER_ID: &str = "candle-llava";

/// How the decoder selects rows from the vision tower's chosen hidden state.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum SelectStrategy {
    /// Keep every patch row (SigLIP has no class token — the JoyCaption / SigLIP-LLaVA setting).
    Full,
    /// Drop the leading row (a CLIP class token) — the classic llava-1.5 setting.
    Default,
}

/// Parsed LLaVA wiring: the nested text + vision configs and the image-splice parameters.
#[derive(Clone, Debug)]
pub struct LlavaConfig {
    /// The language decoder config (from `text_config`).
    pub text: ModelConfig,
    /// The vision tower geometry (from `vision_config`).
    pub vision: SiglipVisionConfig,
    /// The placeholder token id expanded to the image rows (`image_token_index`).
    pub image_token_id: i32,
    /// Which vision hidden state the projector reads (`vision_feature_layer`, HF-style; `-2` =
    /// penultimate).
    pub vision_feature_layer: i32,
    /// Row-selection strategy (`vision_feature_select_strategy`).
    pub select_strategy: SelectStrategy,
    /// Whether the projector activation is the tanh GeLU (`gelu_pytorch_tanh`) rather than exact erf.
    pub projector_gelu_tanh: bool,
    /// Number of placeholder rows one image expands to (patch rows after selection).
    pub image_seq_length: usize,
}

impl LlavaConfig {
    /// Parse a `LlavaForConditionalGeneration` `config.json`.
    pub fn from_json(v: &Value) -> Result<Self> {
        let tc = v
            .get("text_config")
            .ok_or_else(|| Error::Config("llava: config.json has no text_config".into()))?;
        let text = ModelConfig::from_json(tc)?;
        let vision = v
            .get("vision_config")
            .map(SiglipVisionConfig::from_json)
            .unwrap_or_default();
        let image_token_id = v
            .get("image_token_index")
            .and_then(|x| x.as_i64())
            .map(|x| x as i32)
            .unwrap_or(128077); // JoyCaption's <|reserved_special_token_69|>
        let vision_feature_layer = v
            .get("vision_feature_layer")
            .and_then(|x| x.as_i64())
            .map(|x| x as i32)
            .unwrap_or(-2);
        let select_strategy = match v
            .get("vision_feature_select_strategy")
            .and_then(|x| x.as_str())
        {
            Some("default") => SelectStrategy::Default,
            _ => SelectStrategy::Full,
        };
        let projector_gelu_tanh = v
            .get("projector_hidden_act")
            .and_then(|x| x.as_str())
            .map(|s| s.contains("tanh"))
            .unwrap_or(false);
        let dropped = matches!(select_strategy, SelectStrategy::Default) as usize;
        let image_seq_length = vision.num_patches() - dropped;
        Ok(Self {
            text,
            vision,
            image_token_id,
            vision_feature_layer,
            select_strategy,
            projector_gelu_tanh,
            image_seq_length,
        })
    }
}

/// The LLaVA multimodal projector: `linear_2(act(linear_1(x)))`, both layers with bias, run in f32.
pub struct LlavaProjector {
    linear1_w: Tensor,
    linear1_b: Tensor,
    linear2_w: Tensor,
    linear2_b: Tensor,
    gelu_tanh: bool,
}

impl LlavaProjector {
    /// Load HF `multi_modal_projector.{linear_1,linear_2}.{weight,bias}` (cast to f32).
    pub fn from_weights(w: &Weights, prefix: &str, gelu_tanh: bool) -> Result<Self> {
        let f32w = |leaf: &str| -> Result<Tensor> {
            Ok(w.require(&format!("{prefix}.{leaf}"))?
                .to_dtype(candle_core::DType::F32)?)
        };
        Ok(Self {
            linear1_w: f32w("linear_1.weight")?,
            linear1_b: f32w("linear_1.bias")?,
            linear2_w: f32w("linear_2.weight")?,
            linear2_b: f32w("linear_2.bias")?,
            gelu_tanh,
        })
    }

    /// Project SigLIP features `[b, seq, vision_hidden]` to language features `[b, seq, hidden]`.
    pub fn forward(&self, features: &Tensor) -> Result<Tensor> {
        let h = linear(features, &self.linear1_w, Some(&self.linear1_b))?;
        let h = if self.gelu_tanh {
            gelu(&h)?
        } else {
            gelu_erf(&h)?
        };
        linear(&h, &self.linear2_w, Some(&self.linear2_b))
    }
}

/// HF LLaVA prompt expansion: each `image_token_id` becomes `image_seq_length` placeholders so the
/// projected image rows replace them one-for-one.
pub fn expand_image_tokens(ids: &[i32], image_token_id: i32, image_seq_length: usize) -> Vec<i32> {
    let mut out = Vec::with_capacity(ids.len() + image_seq_length.saturating_sub(1));
    for &id in ids {
        if id == image_token_id {
            out.extend(std::iter::repeat_n(image_token_id, image_seq_length));
        } else {
            out.push(id);
        }
    }
    out
}

/// Gather index that replaces each image-token row with the next projected image row: text position
/// `p` keeps row `p`; the `k`-th image token maps to row `n_text + k` (the appended features).
fn image_gather_index(
    ids: &[i32],
    image_token_id: i32,
    n_vis: usize,
    n_text: usize,
) -> Result<Vec<u32>> {
    if ids.len() != n_text {
        return Err(Error::Msg(format!(
            "llava splice: ids length {} != embedding rows {n_text}",
            ids.len()
        )));
    }
    let count = ids.iter().filter(|&&id| id == image_token_id).count();
    if count != n_vis {
        return Err(Error::Msg(format!(
            "llava splice: {count} image tokens != {n_vis} projected image rows"
        )));
    }
    let mut out = Vec::with_capacity(n_text);
    let mut vi = 0u32;
    for (p, &id) in ids.iter().enumerate() {
        if id == image_token_id {
            out.push(n_text as u32 + vi);
            vi += 1;
        } else {
            out.push(p as u32);
        }
    }
    Ok(out)
}

/// Replace the image-token rows of `embeds` (`[b, s, h]`) with `features` (`[b, n_vis, h]` or
/// `[n_vis, h]`), keeping all other rows. `expanded_ids` must already be image-token-expanded and
/// `features` must already be the decoder's dtype.
pub fn splice_image_features(
    embeds: &Tensor,
    expanded_ids: &[i32],
    features: &Tensor,
    image_token_id: i32,
) -> Result<Tensor> {
    let (b, s, h) = embeds.dims3()?;
    let n_text = b * s;
    let feat = match features.dims() {
        [fb, fs, fh] if *fb == b && *fh == h => features.reshape((fb * fs, h))?,
        [fs, fh] if *fh == h => features.reshape((*fs, h))?,
        other => {
            return Err(Error::Msg(format!(
                "llava splice: features must be [b, n_vis, {h}] or [n_vis, {h}], got {other:?}"
            )))
        }
    };
    let n_vis = feat.dim(0)?;
    let gather = image_gather_index(expanded_ids, image_token_id, n_vis, n_text)?;
    let embeds_flat = embeds.reshape((n_text, h))?;
    let src = Tensor::cat(&[&embeds_flat, &feat], 0)?;
    let idx = Tensor::from_vec(gather, (n_text,), embeds.device())?;
    Ok(src.index_select(&idx, 0)?.reshape((b, s, h))?)
}

/// The result of a caption generation.
#[derive(Clone, Debug)]
pub struct LlavaGeneration {
    /// Generated token ids (excludes the prompt and any stop token).
    pub tokens: Vec<i32>,
    /// Why generation stopped.
    pub finish_reason: FinishReason,
    /// The measured decode record (sc-24139): the engine's record of the caption decode, with the
    /// host-side counters and the fused / CUDA-graph / NVFP4 tallies of the whole request from
    /// the spliced prefill on — what [`LlavaProvider`] reports on `TextLlmOutput::decode`.
    pub record: DecodeRecord,
}

/// A loaded LLaVA VLM: vision tower, projector, language decoder, and image preprocessor.
pub struct LlavaModel {
    vision: SiglipVisionTower,
    projector: LlavaProjector,
    language: CausalLm,
    processor: SiglipImageProcessor,
    cfg: LlavaConfig,
    device: Device,
}

impl LlavaModel {
    /// Load a `LlavaForConditionalGeneration` snapshot from `dir` onto `device`. Parses the nested
    /// `text_config` for the decoder and loads the LLaVA-prefixed weight tree
    /// (`language_model.*`, `vision_tower.vision_model.*`, `multi_modal_projector.*`).
    pub fn from_dir(dir: impl AsRef<Path>, device: &Device) -> Result<Self> {
        Self::from_dir_with(dir, device, None)
    }

    /// Like [`from_dir`](Self::from_dir) but optionally quantizing the **language decoder**'s
    /// projections (`requested`, else the snapshot's own persisted `quantization` block). The vision
    /// tower and projector always stay dense (they run in f32). This is the VLM resolution of the
    /// former blanket "load-time quantization is not supported" guard: the decoder's quant rides the
    /// same tensor-level path as the text provider (story 7662).
    pub fn from_dir_with(
        dir: impl AsRef<Path>,
        device: &Device,
        requested: Option<QuantSpec>,
    ) -> Result<Self> {
        let dir = dir.as_ref();
        let text = std::fs::read_to_string(dir.join("config.json"))?;
        let v: Value = serde_json::from_str(&text)
            .map_err(|e| Error::Config(format!("llava config.json: {e}")))?;
        let cfg = LlavaConfig::from_json(&v)?;

        let quant = requested.or(cfg.text.quantization);
        let w = Weights::from_dir(dir, device)?;
        let language = CausalLm::from_weights_with(&w, "language_model", cfg.text.clone(), quant)?;
        let vision = SiglipVisionTower::from_weights(&w, "vision_tower.vision_model", cfg.vision)?;
        let projector =
            LlavaProjector::from_weights(&w, "multi_modal_projector", cfg.projector_gelu_tanh)?;
        let processor = SiglipImageProcessor {
            size: cfg.vision.image_size,
            ..SiglipImageProcessor::default()
        };
        Ok(Self {
            vision,
            projector,
            language,
            processor,
            cfg,
            device: device.clone(),
        })
    }

    /// The LLaVA wiring (text + vision configs, splice parameters).
    pub fn config(&self) -> &LlavaConfig {
        &self.cfg
    }

    /// The underlying language decoder (shared with the text path), for callers that drive their own
    /// embed/splice/decode loop.
    pub fn language(&self) -> &CausalLm {
        &self.language
    }

    /// Select how the language decoder attends on its reference paths and growing step cache —
    /// the caption loop's backing ([`CausalLm::set_attn_formulation`]): [`AttnFormulation::Gqa`]
    /// by default, [`AttnFormulation::Expanded`] for the pre-migration arithmetic (a labelled
    /// comparison, e.g. against goldens captured before sc-24138).
    pub fn set_attn_formulation(&mut self, formulation: AttnFormulation) {
        self.language.set_attn_formulation(formulation);
    }

    /// The device the model is loaded on.
    pub fn device(&self) -> &Device {
        &self.device
    }

    /// Encode interleaved RGB8 `pixels` (`width*height*3` bytes) into projected image features
    /// `[1, image_seq_length, hidden]` (f32).
    pub fn image_features(&self, pixels: &[u8], width: usize, height: usize) -> Result<Tensor> {
        let pix = self
            .processor
            .preprocess(pixels, width, height, &self.device)?;
        let out = self.vision.forward(&pix)?;
        let feat = select_vision_feature(&out, self.cfg.vision_feature_layer)?;
        // CLIP-style "default" strategy drops the class token (row 0); SigLIP "full" keeps all rows.
        let feat = match self.cfg.select_strategy {
            SelectStrategy::Full => feat,
            SelectStrategy::Default => {
                let n = feat.dim(1)?;
                feat.narrow(1, 1, n - 1)?
            }
        };
        self.projector.forward(&feat)
    }

    /// Generate a caption from a tokenized prompt (containing a single `image_token_id`) and the
    /// projected image features. Emits each token through `on_token(id, step)`.
    ///
    /// The spliced embeddings are prefilled into the decoder's shared step cache and the caption
    /// decodes through the step seam — the engine's token-at-a-time loop (sc-24138) — on the
    /// cache's growing backing: this provider has no admission surface to price a preallocation of
    /// the whole (unbounded) caption budget. Same sampler, stop tokens and cancellation as before.
    #[allow(clippy::too_many_arguments)]
    pub fn generate(
        &self,
        prompt_ids: &[i32],
        image_features: &Tensor,
        params: &SamplingParams,
        max_new_tokens: usize,
        seed: Option<u64>,
        stop_tokens: &[i32],
        cancel: &CancelFlag,
        on_token: &mut dyn FnMut(i32, usize),
    ) -> Result<LlavaGeneration> {
        if prompt_ids.is_empty() {
            return Err(Error::Msg("llava: empty prompt".into()));
        }
        if cancel.is_cancelled() {
            return Err(Error::Canceled);
        }

        // The whole request from the spliced prefill on: the engine measures only its own loop.
        let span = RequestSpan::begin();
        // Splice the image rows (in the decoder's dtype) into the token embeddings, then decode.
        let expanded = expand_image_tokens(
            prompt_ids,
            self.cfg.image_token_id,
            self.cfg.image_seq_length,
        );
        let ids_arr = input_ids(&expanded, &self.device)?;
        let embeds = self.language.embed(&ids_arr)?;
        let feat = image_features.to_dtype(self.language.compute_dtype())?;
        let spliced = splice_image_features(&embeds, &expanded, &feat, self.cfg.image_token_id)?;

        let mut cache = self.language.new_step_cache();
        let logits = self
            .language
            .step_prefill_from_embeds(&spliced, &mut cache)?;
        let config = GenerationConfig {
            max_new_tokens,
            sampling: *params,
            seed,
            stop_tokens: stop_tokens.to_vec(),
        };
        let (out, record) = generate_step_from_prefill(
            &self.language,
            &mut cache,
            logits,
            &expanded,
            &config,
            cancel,
            &mut |event| {
                if let StreamEvent::Token { id, step } = event {
                    on_token(id, step);
                }
            },
            None,
        )?;
        Ok(LlavaGeneration {
            tokens: out.tokens,
            finish_reason: out.finish_reason,
            record: record.with_request_span(&span),
        })
    }
}

/// The weight format a LLaVA load quantizes its language decoder to, or the typed refusal. The
/// load runs it first, and [`crate::backend::nvfp4_support`] answers a product's per-snapshot
/// NVFP4 question with it (sc-24139): the Llama provider serves NVFP4 (the qwen3_5 hybrid,
/// sc-24135, and the llama family, sc-24140), but LLaVA — whose provider owns the vision-tower
/// load — does not, so it is refused here by name.
pub(crate) fn requested_quantization(spec: &LoadSpec) -> CoreResult<Option<QuantSpec>> {
    spec.quantize
        .map(|q| match q {
            Quantize::Q4 => Ok(QuantSpec::q4()),
            Quantize::Q8 => Ok(QuantSpec::q8()),
            Quantize::Nvfp4 => Err(CoreError::Unsupported(
                "nvfp4: NVFP4 projections are not served for LLaVA".into(),
            )),
        })
        .transpose()
}

/// LLaVA served as a multimodal [`core_llm::TextLlm`] provider.
pub struct LlavaProvider {
    descriptor: TextLlmDescriptor,
    model: LlavaModel,
    tokenizer: Tokenizer,
    template: Box<dyn ChatTemplate>,
    stop_tokens: Vec<i32>,
    /// The speculative option a request that leaves it unset runs with — this backend's row of
    /// the defaults table ([`core_llm::defaults`], E5). The captioner runs no proposer, so a
    /// non-`off` default is reported as the named no-proposer fallback, as an explicit one is.
    speculative_default: core_llm::Speculative,
}

impl LlavaProvider {
    /// Load from a snapshot directory (config.json + tokenizer.json + shards). An explicit
    /// `spec.quantize` (or the snapshot's persisted `quantization` block) quantizes the language
    /// decoder's projections; the vision tower and projector stay dense.
    pub fn load(spec: &LoadSpec) -> CoreResult<Self> {
        let requested = requested_quantization(spec)?;
        let dir = Path::new(&spec.source);
        let device = select_eager_device().map_err(to_core)?;
        let model = LlavaModel::from_dir_with(dir, &device, requested).map_err(to_core)?;
        let tokenizer = Tokenizer::from_file(dir.join("tokenizer.json"))?;
        // `eos_token_ids` always returns a non-empty model-specific set or the Llama-3 fallback.
        let stop_tokens = crate::provider::eos_token_ids(dir);
        Ok(Self {
            descriptor: descriptor(),
            model,
            tokenizer,
            template: load_chat_template(dir),
            stop_tokens,
            speculative_default: crate::device::decode_defaults(&device).speculative,
        })
    }

    /// The loaded model.
    pub fn model(&self) -> &LlavaModel {
        &self.model
    }

    /// Render the request into a chat prompt and the single image. Exactly one image is supported;
    /// the conversation is rendered text-only (a LLaVA chat template inserts the image token itself,
    /// so injecting one here would duplicate it).
    fn build_inputs<'a>(
        &self,
        req: &'a TextLlmRequest,
    ) -> CoreResult<(String, &'a core_llm::ImageRef)> {
        let mut image: Option<&core_llm::ImageRef> = None;
        for msg in &req.messages {
            for c in &msg.content {
                if let Content::Image(img) = c {
                    if image.is_some() {
                        return Err(CoreError::Unsupported(
                            "llava: exactly one image is supported".into(),
                        ));
                    }
                    image = Some(img);
                }
            }
        }
        let image =
            image.ok_or_else(|| CoreError::InvalidRequest("llava: request has no image".into()))?;

        let messages: Vec<Message> = req
            .messages
            .iter()
            .map(|m| Message::text(m.role, m.text_content()))
            .collect();
        let prompt = self.template.render(&messages, true)?;
        Ok((prompt, image))
    }

    /// Tokenize the rendered prompt and guarantee it carries **exactly one** image placeholder token
    /// (one image → one spliced span). LLaVA chat templates place the token themselves; if the
    /// template produced none (e.g. a plain-text fallback), insert one after any leading BOS.
    fn prompt_ids_with_image(&self, chat_text: &str) -> CoreResult<Vec<i32>> {
        let img = self.model.cfg.image_token_id;
        let mut ids: Vec<i32> = self
            .tokenizer
            .encode(chat_text, false)?
            .into_iter()
            .map(|id| id as i32)
            .collect();
        let count = ids.iter().filter(|&&t| t == img).count();
        match count {
            1 => Ok(ids),
            0 => {
                // No image token from the template (e.g. a plain-text fallback template): place one
                // at the front of the prompt, the conventional LLaVA position.
                ids.insert(0, img);
                Ok(ids)
            }
            n => Err(CoreError::InvalidRequest(format!(
                "llava: chat template produced {n} image tokens for one image (expected 1)"
            ))),
        }
    }
}

impl TextLlm for LlavaProvider {
    fn descriptor(&self) -> &TextLlmDescriptor {
        &self.descriptor
    }

    fn validate(&self, req: &TextLlmRequest) -> CoreResult<()> {
        self.descriptor
            .capabilities
            .validate_request(&self.descriptor.id, req)
    }

    fn generate(
        &self,
        req: &TextLlmRequest,
        on_event: &mut dyn FnMut(CoreEvent),
    ) -> CoreResult<TextLlmOutput> {
        self.validate(req)?;
        if req.cancel.is_cancelled() {
            return Err(CoreError::Canceled);
        }

        let (chat_text, image) = self.build_inputs(req)?;
        let prompt_ids = self.prompt_ids_with_image(&chat_text)?;
        // The engine sees the expanded prompt (image token → image_seq_length rows).
        let prompt_len = expand_image_tokens(
            &prompt_ids,
            self.model.cfg.image_token_id,
            self.model.cfg.image_seq_length,
        )
        .len() as u32;

        let features = self
            .model
            .image_features(&image.pixels, image.width as usize, image.height as usize)
            .map_err(to_core)?;

        let params = map_sampling(&req.sampling);
        let max_new = req.max_new_tokens as usize;

        // Stream contract token events via incremental detokenization (re-decode, emit new suffix).
        // The `IncrementalDetok` guard holds back lossy U+FFFD placeholders so a multi-byte
        // character split across BPE tokens streams intact (no mid-char slice panic) — sc-12452.
        let tokenizer = &self.tokenizer;
        let mut acc: Vec<u32> = Vec::new();
        let mut detok = IncrementalDetok::new();
        let mut on_token = |id: i32, step: usize| {
            acc.push(id as u32);
            if let Ok(text) = tokenizer.decode(&acc, true) {
                if let Some(delta) = detok.push(&text) {
                    on_event(CoreEvent::Token {
                        id: id as u32,
                        text: delta.to_string(),
                        index: step,
                        channel: Channel::Content,
                    });
                }
            }
        };
        let gen = self
            .model
            .generate(
                &prompt_ids,
                &features,
                &params,
                max_new,
                req.seed,
                &self.stop_tokens,
                &req.cancel,
                &mut on_token,
            )
            .map_err(to_core)?;

        let gen_u32: Vec<u32> = gen.tokens.iter().map(|&i| i as u32).collect();
        let text = tokenizer.decode(&gen_u32, true)?;
        let finish = map_finish(gen.finish_reason);
        let usage = Usage {
            prompt_tokens: prompt_len,
            generated_tokens: gen.tokens.len() as u32,
        };
        on_event(CoreEvent::Done {
            finish_reason: finish,
            usage,
        });
        Ok(TextLlmOutput {
            timings: None,
            text,
            thinking: None,
            // No tool calling on the vision path (its chat template renders captions, not tools).
            tool_calls: Vec::new(),
            usage,
            mtp: None,
            // The caption decodes through the shared engine (sc-24138), so it reports its path
            // like every engine request. The CUDA-graph switch is not wired into this provider
            // (no graph runner wraps its decoder), so the report says the switch was off here.
            // A captioner advertises no proposer and has no prefix cache: both are named, in the
            // words MLX's JoyCaption uses (E2, E8).
            decode: Some(caption_report(&gen.record, req, self.speculative_default)),
            finish_reason: Some(finish),
        })
    }
}

/// A caption's measured report (sc-24139): the engine record with the CUDA-graph switch off (no
/// graph runner wraps this decoder), and — the captioner advertising no proposer and having no
/// prefix cache — the request's speculative fallback and the prefix-cache reason named in the
/// words MLX's JoyCaption uses (E2, E8). The request's option is resolved against the provider's
/// per-backend `default` (E5), so a table default of `auto` is named too.
fn caption_report(
    record: &DecodeRecord,
    req: &TextLlmRequest,
    default: core_llm::Speculative,
) -> core_llm::DecodeReport {
    record
        .report(false)
        .with_captioner_reasons(req.speculative_or(default))
}

/// The LLaVA provider descriptor (constructible without weights; used for catalog composition).
pub fn descriptor() -> TextLlmDescriptor {
    TextLlmDescriptor {
        id: PROVIDER_ID.to_string(),
        family: "llava".to_string(),
        backend: "candle".to_string(),
        capabilities: TextLlmCapabilities {
            max_context_tokens: 0,
            max_new_tokens: 0,
            supports_system_prompt: true,
            supports_vision: true,
            // Single-image caption path only; no video support.
            supports_video: false,
            // Text+vision captioner; no audio path at all.
            supports_audio: false,
            supports_thinking: false,
            supports_reasoning_effort: false,
            reasoning_efforts: Vec::new(),
            model_sampling_defaults: None,
            supports_preserve_thinking: false,
            // Vision/caption path only; no tool calling (mirrors the mlx JoyCaption provider).
            supports_tools: false,
            mtp: None,
            speculative: Vec::new(),
            supported_constraints: Vec::new(),
        },
    }
}

/// Use the model's own Jinja `chat_template` (from `tokenizer_config.json`) when present; otherwise
/// fall back to the typed Llama-3 template.
fn load_chat_template(dir: &Path) -> Box<dyn ChatTemplate> {
    match JinjaChatTemplate::from_tokenizer_config_file(dir.join("tokenizer_config.json")) {
        Ok(t) => Box::new(t),
        Err(_) => Box::new(Llama3Template),
    }
}

fn map_sampling(s: &Sampling) -> SamplingParams {
    SamplingParams {
        temperature: s.temperature,
        top_p: s.top_p,
        top_k: s.top_k,
        presence_penalty: s.presence_penalty,
        repetition_penalty: s.repetition_penalty,
        repetition_context: s.repetition_context,
    }
}

fn map_finish(f: FinishReason) -> CoreFinish {
    match f {
        FinishReason::StopToken | FinishReason::Stopped => CoreFinish::Stop,
        FinishReason::MaxTokens => CoreFinish::Length,
        FinishReason::Cancelled => CoreFinish::Cancelled,
    }
}

fn to_core(e: Error) -> CoreError {
    match e {
        Error::Canceled => CoreError::Canceled,
        Error::Unsupported(m) => CoreError::Unsupported(m),
        Error::MissingTensor(m) => CoreError::Load(format!("missing tensor: {m}")),
        Error::Config(m) => CoreError::Load(m),
        Error::Io(e) => CoreError::Io(e),
        other => CoreError::backend(other),
    }
}

/// Ordinary registration used by explicit runtime bundles.
pub const REGISTRATION: core_llm::TextLlmRegistration = core_llm::TextLlmRegistration {
    descriptor,
    load: load_registered,
    can_load,
    // The static descriptor already declares `supports_vision=true`; no per-snapshot probe needed.
    weightless_vision: None,
    // LLaVA is a text+vision provider with no audio path; `supports_audio=false` holds for every
    // snapshot its `can_load` claims.
    weightless_audio: None,
};

fn load_registered(spec: &LoadSpec) -> CoreResult<Box<dyn TextLlm>> {
    Ok(Box::new(LlavaProvider::load(spec)?))
}

/// Weightless model-first probe (story 7406): can the `candle-llava` vision provider serve the
/// snapshot directory at `spec.source`? Reads **only** `config.json` and keys on the LLaVA structural
/// signature — a nested `text_config` (the language decoder) plus a `vision_config` (the SigLIP
/// tower) — which [`LlavaConfig::from_json`] requires. Never opens a safetensors shard.
pub fn can_load(spec: &LoadSpec) -> bool {
    can_load_with(Path::new(&spec.source), |path| {
        std::fs::read_to_string(path)
            .ok()
            .and_then(|text| serde_json::from_str(&text).ok())
    })
}

fn can_load_with(source: &Path, read_config: impl FnOnce(&Path) -> Option<Value>) -> bool {
    if !source.is_dir() {
        return false;
    }
    let Some(v) = read_config(&source.join("config.json")) else {
        return false;
    };
    // LLaVA = a SigLIP/CLIP vision tower + a `text_config` decoder. Decline Qwen3.6 (`qwen3_5`): its
    // VLM checkpoint also carries `text_config` + `vision_config`, but it is the hybrid Gated-DeltaNet
    // decoder + a Qwen-VL ViT — served by the `candle-llama` provider (as text), not as a SigLIP LLaVA.
    if matches!(Architecture::from_config(&v), Ok(Architecture::Qwen35)) {
        return false;
    }
    v.get("text_config").is_some() && v.get("vision_config").is_some()
}

#[cfg(test)]
mod tests {
    use super::*;

    /// E2/E8 (sc-24432 feature-end review): the captioner advertises no proposer, so an `auto` or
    /// explicit speculative request decodes plainly with the reason named — the same words MLX's
    /// JoyCaption reports — and its prefix cache's `none` is named too; `off` names no fallback.
    #[test]
    fn the_report_names_the_speculative_fallback_and_the_prefix_cache() {
        use core_llm::{Message, Speculative, SpeculativeProposer};
        let record = crate::decode::DecodeRecord::plain(
            crate::decode::DecodePath::StepModel,
            3,
            2,
            Default::default(),
        );
        let request = |speculative| core_llm::TextLlmRequest {
            messages: vec![Message::user("x")],
            speculative: Some(speculative),
            ..Default::default()
        };
        for mode in [
            Speculative::Auto,
            Speculative::proposer(SpeculativeProposer::PromptLookup, 2),
        ] {
            let report = caption_report(&record, &request(mode), Speculative::Off);
            assert_eq!(
                report.fallbacks,
                core_llm::no_proposer_fallback(mode, core_llm::CAPTIONER_NO_PROPOSER)
                    .into_iter()
                    .collect::<Vec<_>>(),
                "{mode:?}"
            );
            assert_eq!(report.fallbacks.len(), 1, "{mode:?}");
            assert_eq!(report.proposer, core_llm::ProposerKind::None);
        }
        // E5: an unset option takes the provider's per-backend default — a table `auto` is named
        // as the same no-proposer fallback — and an explicit `off` still overrides the default.
        let unset = core_llm::TextLlmRequest {
            messages: vec![Message::user("x")],
            ..Default::default()
        };
        assert_eq!(
            caption_report(&record, &unset, Speculative::Auto).fallbacks,
            core_llm::no_proposer_fallback(Speculative::Auto, core_llm::CAPTIONER_NO_PROPOSER)
                .into_iter()
                .collect::<Vec<_>>()
        );
        assert!(caption_report(&record, &unset, Speculative::Off)
            .fallbacks
            .is_empty());
        assert!(
            caption_report(&record, &request(Speculative::Off), Speculative::Auto)
                .fallbacks
                .is_empty()
        );
        let off = caption_report(&record, &request(Speculative::Off), Speculative::Off);
        assert!(off.fallbacks.is_empty());
        assert_eq!(off.prefix_cache.path, "none");
        assert_eq!(
            off.prefix_cache.reason.as_deref(),
            Some(core_llm::CAPTIONER_NO_PREFIX_CACHE)
        );
    }
    /// A tiny LLaVA snapshot (an 8 × 8 SigLIP tower of 4 patches, a 2-layer Llama of vocab 32,
    /// seeded random weights, a `t0..t31` word-level tokenizer and no chat template — the Llama 3
    /// fallback renders the prompt) that [`LlavaProvider::load`] loads on the CPU.
    fn tiny_llava_snapshot() -> tempfile::TempDir {
        use crate::primitives::{SplitMix64, TokenRng};
        let (v_hidden, v_inter, l_hidden, l_inter, vocab) = (16, 32, 32, 64, 32usize);
        let (q_dim, kv_dim) = (l_hidden, 2 * (l_hidden / 4));
        let dir = tempfile::Builder::new()
            .prefix("candle-llava-fixture-")
            .tempdir()
            .unwrap();
        let config = serde_json::json!({
            "architectures": ["LlavaForConditionalGeneration"], "model_type": "llava",
            "image_token_index": 7, "vision_feature_layer": -1,
            "vision_feature_select_strategy": "full", "projector_hidden_act": "gelu",
            "vision_config": {
                "image_size": 8, "patch_size": 4, "num_channels": 3, "hidden_size": v_hidden,
                "intermediate_size": v_inter, "num_hidden_layers": 1, "num_attention_heads": 2,
                "layer_norm_eps": 1e-6
            },
            "text_config": {
                "architectures": ["LlamaForCausalLM"], "model_type": "llama",
                "hidden_size": l_hidden, "intermediate_size": l_inter, "num_hidden_layers": 2,
                "num_attention_heads": 4, "num_key_value_heads": 2, "vocab_size": vocab,
                "rms_norm_eps": 1e-6, "rope_theta": 10000.0, "tie_word_embeddings": false
            }
        });
        std::fs::write(dir.path().join("config.json"), config.to_string()).unwrap();
        let entries: Vec<String> = (0..vocab).map(|i| format!("\"t{i}\": {i}")).collect();
        let tokenizer = format!(
            r#"{{ "version": "1.0", "added_tokens": [], "normalizer": null,
                 "pre_tokenizer": {{ "type": "Whitespace" }}, "post_processor": null,
                 "decoder": null,
                 "model": {{ "type": "WordLevel", "vocab": {{ {} }}, "unk_token": "t0" }} }}"#,
            entries.join(", ")
        );
        std::fs::write(dir.path().join("tokenizer.json"), tokenizer).unwrap();
        let mut rng = SplitMix64::new(0x11a7_a5ee);
        let mut weights = std::collections::HashMap::new();
        let mut put = |key: String, dims: &[usize], ones: bool| {
            let n: usize = dims.iter().product();
            let data: Vec<f32> = (0..n)
                .map(|_| {
                    if ones {
                        1.0
                    } else {
                        (rng.next_f32() - 0.5) * 0.4
                    }
                })
                .collect();
            let tensor = Tensor::from_vec(data, dims.to_vec(), &Device::Cpu).unwrap();
            weights.insert(key, tensor);
        };
        let vp = |s: &str| format!("vision_tower.vision_model.{s}");
        put(
            vp("embeddings.patch_embedding.weight"),
            &[v_hidden, 3, 4, 4],
            false,
        );
        put(vp("embeddings.patch_embedding.bias"), &[v_hidden], false);
        put(
            vp("embeddings.position_embedding.weight"),
            &[4, v_hidden],
            false,
        );
        let layer = |s: &str| vp(&format!("encoder.layers.0.{s}"));
        for norm in ["layer_norm1", "layer_norm2"] {
            put(layer(&format!("{norm}.weight")), &[v_hidden], true);
            put(layer(&format!("{norm}.bias")), &[v_hidden], false);
        }
        for proj in ["q_proj", "k_proj", "v_proj", "out_proj"] {
            put(
                layer(&format!("self_attn.{proj}.weight")),
                &[v_hidden, v_hidden],
                false,
            );
            put(layer(&format!("self_attn.{proj}.bias")), &[v_hidden], false);
        }
        put(layer("mlp.fc1.weight"), &[v_inter, v_hidden], false);
        put(layer("mlp.fc1.bias"), &[v_inter], false);
        put(layer("mlp.fc2.weight"), &[v_hidden, v_inter], false);
        put(layer("mlp.fc2.bias"), &[v_hidden], false);
        put(vp("post_layernorm.weight"), &[v_hidden], true);
        put(vp("post_layernorm.bias"), &[v_hidden], false);
        for (i, (out, input)) in [(l_hidden, v_hidden), (l_hidden, l_hidden)]
            .iter()
            .enumerate()
        {
            let key = |leaf: &str| format!("multi_modal_projector.linear_{}.{leaf}", i + 1);
            put(key("weight"), &[*out, *input], false);
            put(key("bias"), &[*out], false);
        }
        let lm = |s: &str| format!("language_model.{s}");
        put(lm("model.embed_tokens.weight"), &[vocab, l_hidden], false);
        put(lm("model.norm.weight"), &[l_hidden], true);
        put(lm("lm_head.weight"), &[vocab, l_hidden], false);
        for i in 0..2 {
            let p = |s: &str| lm(&format!("model.layers.{i}.{s}"));
            put(p("input_layernorm.weight"), &[l_hidden], true);
            put(p("post_attention_layernorm.weight"), &[l_hidden], true);
            put(p("self_attn.q_proj.weight"), &[q_dim, l_hidden], false);
            put(p("self_attn.k_proj.weight"), &[kv_dim, l_hidden], false);
            put(p("self_attn.v_proj.weight"), &[kv_dim, l_hidden], false);
            put(p("self_attn.o_proj.weight"), &[l_hidden, q_dim], false);
            put(p("mlp.gate_proj.weight"), &[l_inter, l_hidden], false);
            put(p("mlp.up_proj.weight"), &[l_inter, l_hidden], false);
            put(p("mlp.down_proj.weight"), &[l_hidden, l_inter], false);
        }
        candle_core::safetensors::save(&weights, dir.path().join("model.safetensors")).unwrap();
        dir
    }

    /// E2/E5/E8 (sc-24432), through the provider's own `generate`: a caption's report carries the
    /// captioner reasons — the speculative fallback for `auto` and an explicit proposer, none for
    /// `off`, and the prefix cache's `none` named in every case — and an unset option resolves
    /// against the provider's per-backend default, which an explicit `off` still overrides.
    #[test]
    fn generate_reports_the_captioner_reasons() {
        use core_llm::{ImageRef, Speculative, SpeculativeProposer};
        let dir = tiny_llava_snapshot();
        let mut provider =
            LlavaProvider::load(&LoadSpec::dense(dir.path().display().to_string())).unwrap();
        let pixels: Vec<u8> = (0..8 * 8 * 3).map(|i| (i * 7 % 251) as u8).collect();
        let image = ImageRef::new(8, 8, pixels).unwrap();
        let request = |speculative: Option<Speculative>| TextLlmRequest {
            messages: vec![Message {
                content: vec![Content::Image(image.clone()), Content::text("t3 t5")],
                ..Message::user("")
            }],
            speculative,
            max_new_tokens: 3,
            sampling: Sampling::greedy(),
            ..Default::default()
        };
        let report = |provider: &LlavaProvider, speculative| {
            let out = provider
                .generate(&request(speculative), &mut |_| {})
                .unwrap();
            assert_eq!(out.usage.generated_tokens, 3);
            let report = out.decode.expect("the caption reports its decode");
            assert_eq!(report.path, "step_model");
            assert_eq!(report.proposer, core_llm::ProposerKind::None);
            assert_eq!(report.prefix_cache.path, "none");
            assert_eq!(
                report.prefix_cache.reason.as_deref(),
                Some(core_llm::CAPTIONER_NO_PREFIX_CACHE)
            );
            report.fallbacks
        };
        let named = |mode| {
            core_llm::no_proposer_fallback(mode, core_llm::CAPTIONER_NO_PROPOSER)
                .into_iter()
                .collect::<Vec<_>>()
        };
        let lookup = Speculative::proposer(SpeculativeProposer::PromptLookup, 2);
        for mode in [Speculative::Auto, lookup] {
            let fallbacks = report(&provider, Some(mode));
            assert_eq!(fallbacks.len(), 1, "{mode:?}");
            assert_eq!(fallbacks, named(mode), "{mode:?}");
        }
        assert!(report(&provider, Some(Speculative::Off)).is_empty());
        // E5: the provider's default reaches an unset request (and only an unset one).
        provider.speculative_default = Speculative::Auto;
        assert_eq!(report(&provider, None), named(Speculative::Auto));
        assert!(report(&provider, Some(Speculative::Off)).is_empty());
        provider.speculative_default = Speculative::Off;
        assert!(report(&provider, None).is_empty());
    }

    use std::cell::Cell;

    const IMG: i32 = 128077;

    #[test]
    fn expand_replaces_image_token() {
        let ids = [1, IMG, 2];
        let expanded = expand_image_tokens(&ids, IMG, 729);
        assert_eq!(expanded.len(), 2 + 729);
        assert_eq!(expanded[0], 1);
        assert!(expanded[1..1 + 729].iter().all(|&t| t == IMG));
        assert_eq!(*expanded.last().unwrap(), 2);
    }

    #[test]
    fn gather_index_maps_image_rows_to_appended_features() {
        // ids [10, IMG, IMG, 11], 4 text rows, 2 image rows appended at 4,5.
        let got = image_gather_index(&[10, IMG, IMG, 11], IMG, 2, 4).unwrap();
        assert_eq!(got, vec![0, 4, 5, 3]);
    }

    #[test]
    fn gather_index_rejects_count_mismatch() {
        assert!(image_gather_index(&[IMG, 7], IMG, 2, 2).is_err());
    }

    #[test]
    fn splice_replaces_only_image_rows() {
        use candle_core::Device;
        // rows for ids [5, IMG, IMG, 6]; features [1,2,2] replace the two IMG rows.
        let embeds = Tensor::from_vec(
            vec![1.0f32, 1.0, 10.0, 10.0, 20.0, 20.0, 2.0, 2.0],
            (1, 4, 2),
            &Device::Cpu,
        )
        .unwrap();
        let ids = [5, IMG, IMG, 6];
        let features =
            Tensor::from_vec(vec![100.0f32, 101.0, 200.0, 201.0], (1, 2, 2), &Device::Cpu).unwrap();
        let got = splice_image_features(&embeds, &ids, &features, IMG).unwrap();
        let h = got.flatten_all().unwrap().to_vec1::<f32>().unwrap();
        assert_eq!(h, vec![1.0, 1.0, 100.0, 101.0, 200.0, 201.0, 2.0, 2.0]);
    }

    #[test]
    fn config_from_json_reads_llava_wiring() {
        let v: Value = serde_json::from_str(
            r#"{
                "image_token_index": 32000,
                "vision_feature_layer": -2,
                "vision_feature_select_strategy": "default",
                "vision_config": {"image_size": 336, "patch_size": 14, "hidden_size": 1024,
                    "intermediate_size": 4096, "num_hidden_layers": 24, "num_attention_heads": 16},
                "text_config": {"architectures": ["LlamaForCausalLM"], "model_type": "llama",
                    "hidden_size": 64, "intermediate_size": 128, "num_hidden_layers": 2,
                    "num_attention_heads": 4, "num_key_value_heads": 2, "vocab_size": 100,
                    "rms_norm_eps": 1e-5, "rope_theta": 10000.0, "tie_word_embeddings": false}
            }"#,
        )
        .unwrap();
        let cfg = LlavaConfig::from_json(&v).unwrap();
        assert_eq!(cfg.image_token_id, 32000);
        assert_eq!(cfg.vision_feature_layer, -2);
        assert_eq!(cfg.select_strategy, SelectStrategy::Default);
        // 336/14 = 24 grid -> 576 patches, minus 1 (CLS) for "default".
        assert_eq!(cfg.image_seq_length, 24 * 24 - 1);
    }

    #[test]
    fn descriptor_declares_vision() {
        let d = descriptor();
        assert_eq!(d.id, PROVIDER_ID);
        assert!(d.capabilities.supports_vision);
    }

    #[test]
    fn model_probe_rejects_file_sources_before_reading_them() {
        let file = tempfile::NamedTempFile::new().unwrap();
        std::fs::write(
            file.path(),
            serde_json::json!({
                "text_config": {"model_type": "llama"},
                "vision_config": {"model_type": "clip_vision_model"}
            })
            .to_string(),
        )
        .unwrap();
        let read_attempted = Cell::new(false);
        let result = can_load_with(file.path(), |_| {
            read_attempted.set(true);
            Some(Value::Null)
        });

        assert!(!result);
        assert!(!read_attempted.get(), "file payload must not be read");
        assert!(!can_load(&LoadSpec::dense(
            file.path().display().to_string()
        )));
    }
}
