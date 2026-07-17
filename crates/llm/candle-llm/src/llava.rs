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
use crate::decode::stream::default_seed;
use crate::decode::{CancelFlag, FinishReason};
use crate::device::select_device;
use crate::error::{Error, Result};
use crate::image::SiglipImageProcessor;
use crate::models::siglip::{select_vision_feature, SiglipVisionConfig, SiglipVisionTower};
use crate::models::CausalLm;
use crate::primitives::nn::{gelu, gelu_erf, linear};
use crate::primitives::projection::QuantSpec;
use crate::primitives::sampler::{sample, SamplingParams, SplitMix64};
use crate::primitives::{input_ids, Weights};

/// The registry id of the LLaVA provider.
pub const PROVIDER_ID: &str = "candle-llava";

/// Llama-3's end tokens (`<|end_of_text|>`, `<|eom_id|>`, `<|eot_id|>`) — the JoyCaption stop set,
/// also a safe default for any Llama-3-tokenized LLaVA.
pub const LLAMA3_STOP_TOKENS: &[i32] = &[128001, 128008, 128009];

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

        let mut cache = self.language.new_cache();
        let mut rng = SplitMix64::new(seed.unwrap_or_else(default_seed));
        let mut history = expanded.clone();
        let mut generated: Vec<i32> = Vec::new();
        let prompt_len = expanded.len() as i32;
        let mut logits = self
            .language
            .decode_logits_from_embeds(&spliced, &mut cache, 0)?;
        let mut finish = FinishReason::MaxTokens;

        for step in 0..max_new_tokens {
            if cancel.is_cancelled() {
                finish = FinishReason::Cancelled;
                break;
            }
            let next = sample(&logits, &history, params, &mut rng, None)?;
            if stop_tokens.contains(&next) {
                finish = FinishReason::StopToken;
                break;
            }
            on_token(next, step);
            generated.push(next);
            history.push(next);
            if step + 1 == max_new_tokens {
                break;
            }
            let tok = input_ids(&[next], &self.device)?;
            logits = self
                .language
                .decode_logits(&tok, &mut cache, prompt_len + step as i32)?;
        }

        Ok(LlavaGeneration {
            tokens: generated,
            finish_reason: finish,
        })
    }
}

/// LLaVA served as a multimodal [`core_llm::TextLlm`] provider.
pub struct LlavaProvider {
    descriptor: TextLlmDescriptor,
    model: LlavaModel,
    tokenizer: Tokenizer,
    template: Box<dyn ChatTemplate>,
    stop_tokens: Vec<i32>,
}

impl LlavaProvider {
    /// Load from a snapshot directory (config.json + tokenizer.json + shards). An explicit
    /// `spec.quantize` (or the snapshot's persisted `quantization` block) quantizes the language
    /// decoder's projections; the vision tower and projector stay dense.
    pub fn load(spec: &LoadSpec) -> CoreResult<Self> {
        let requested = spec.quantize.map(|q| match q {
            Quantize::Q4 => QuantSpec::q4(),
            Quantize::Q8 => QuantSpec::q8(),
        });
        let dir = Path::new(&spec.source);
        let device = select_device().map_err(to_core)?;
        let model = LlavaModel::from_dir_with(dir, &device, requested).map_err(to_core)?;
        let tokenizer = Tokenizer::from_file(dir.join("tokenizer.json"))?;
        let stop_tokens = crate::provider::eos_token_ids(dir);
        let stop_tokens = if stop_tokens.is_empty() {
            LLAMA3_STOP_TOKENS.to_vec()
        } else {
            stop_tokens
        };
        Ok(Self {
            descriptor: descriptor(),
            model,
            tokenizer,
            template: load_chat_template(dir),
            stop_tokens,
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
            text,
            thinking: None,
            // No tool calling on the vision path (its chat template renders captions, not tools).
            tool_calls: Vec::new(),
            usage,
            finish_reason: Some(finish),
        })
    }
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
            supports_thinking: false,
            // Vision/caption path only; no tool calling (mirrors the mlx JoyCaption provider).
            supports_tools: false,
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
        repetition_penalty: s.repetition_penalty,
        repetition_context: s.repetition_context,
    }
}

fn map_finish(f: FinishReason) -> CoreFinish {
    match f {
        FinishReason::StopToken => CoreFinish::Stop,
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
};

fn load_registered(spec: &LoadSpec) -> CoreResult<Box<dyn TextLlm>> {
    Ok(Box::new(LlavaProvider::load(spec)?))
}

/// Weightless model-first probe (story 7406): can the `candle-llava` vision provider serve the
/// snapshot at `spec.source`? Reads **only** `config.json` and keys on the LLaVA structural
/// signature — a nested `text_config` (the language decoder) plus a `vision_config` (the SigLIP
/// tower) — which [`LlavaConfig::from_json`] requires. Never opens a safetensors shard.
pub fn can_load(spec: &LoadSpec) -> bool {
    let dir = Path::new(&spec.source);
    let path = if dir.is_dir() {
        dir.join("config.json")
    } else {
        dir.to_path_buf()
    };
    let Ok(text) = std::fs::read_to_string(path) else {
        return false;
    };
    let Ok(v) = serde_json::from_str::<Value>(&text) else {
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
}
