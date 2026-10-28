//! Native Candle provider for the exact `starvector/starvector-8b-im2svg` snapshot.
//!
//! The snapshot is treated as inert local data: config, tokenizer, and safetensors only. No
//! snapshot code, Python, Transformers, network routing, or model download is involved.

use std::cell::{Cell, RefCell};
use std::path::Path;
use std::time::Instant;

use candle_core::{DType, Tensor};
use serde_json::Value;

use core_llm::{
    Error as CoreError, LoadSpec, Result as CoreResult, StarVectorBoundedStream,
    StarVectorDescriptor, StarVectorFinishReason, StarVectorOutput, StarVectorProvider,
    StarVectorRequest, StarVectorStreamEvent, StarVectorStreamStatus, StreamEvent, TextLlm,
    TextLlmDescriptor, TextLlmOutput, TextLlmRequest, Tokenizer,
};

use crate::decode::{
    generate_step_from_prefill, DecodeRecord, FinishReason as DecodeFinish, GenerationConfig,
    RequestSpan, StreamEvent as DecodeEvent,
};
use crate::error::{Error, Result};
use crate::image::{SiglipImageProcessor, SiglipPreprocess};
use crate::models::{SiglipVisionConfig, SiglipVisionTower, StarCoder2, StarCoder2Config};
use crate::primitives::nn::{layer_norm, linear, silu};
use crate::primitives::sampler::SamplingParams;
use crate::primitives::{input_ids, StepKvCache, Weights};
use core_llm::{emit_svg_done, validate_loaded_svg_context_cap};

pub const PROVIDER_ID: &str = "candle-starvector-8b";
// The snapshot's geometry and probe are shared with MLX (sc-27255).
use core_llm::starvector::starvector_8b::{
    exact_config, DECODER_HIDDEN, EOS_TOKEN_ID, IMAGE_SIZE, IMAGE_TOKENS, MAX_SVG_BYTES,
    MAX_WALL_TIME, SNAPSHOT_REPOSITORY, SNAPSHOT_REVISION, SVG_PROMPT, VISION_HIDDEN,
};

/// Snapshot-local observation, rather than a fabricated process-global CUDA allocation number.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct StarVectorCandleMemory {
    pub loaded_tensor_count: usize,
    pub device: String,
}

pub struct StarVector8bModel {
    vision: SiglipVisionTower,
    adapter: StarVector8bAdapter,
    decoder: StarCoder2,
}

impl StarVector8bModel {
    fn from_weights(weights: &Weights) -> Result<Self> {
        Ok(Self {
            vision: SiglipVisionTower::from_weights(
                weights,
                "model.image_encoder.visual_encoder",
                siglip_config(),
            )?,
            adapter: StarVector8bAdapter::from_weights(weights, "model.image_projection")?,
            decoder: StarCoder2::from_weights(
                weights,
                "model.svg_transformer.transformer",
                StarCoder2Config::STARVECTOR_8B,
            )?,
        })
    }

    /// Encode the image, join it to the `<svg` prompt embeddings and prefill them into a fresh
    /// step-seam cache for the decoder (growing backing: the provider has no admission surface to
    /// price a preallocation of the whole 16k-token SVG budget). The continuation then decodes
    /// through the step seam (sc-24138).
    fn prefill(&self, image: &core_llm::ImageRef, prompt: &[i32]) -> Result<(Tensor, StepKvCache)> {
        let pixels = preprocess_image(image, self.decoder.device())?;
        let vision = self.vision.forward(&pixels)?.last_hidden_state;
        let vision = self.adapter.forward(&vision, self.decoder.dtype())?;
        let text = self
            .decoder
            .embed(&input_ids(prompt, self.decoder.device())?)?;
        let embeds = Tensor::cat(&[&vision, &text], 1)?;
        let mut cache = self.decoder.new_step_cache();
        let logits = self.decoder.step_prefill_from_embeds(&embeds, &mut cache)?;
        Ok((logits, cache))
    }
}

/// Exact 8B projection: Linear(1024,2048) -> SiLU -> Linear(2048,4608) -> LayerNorm([576,4608]).
struct StarVector8bAdapter {
    fc_weight: Tensor,
    fc_bias: Tensor,
    proj_weight: Tensor,
    proj_bias: Tensor,
    norm_weight: Tensor,
    norm_bias: Tensor,
}

impl StarVector8bAdapter {
    fn from_weights(weights: &Weights, prefix: &str) -> Result<Self> {
        let key = |suffix: &str| format!("{prefix}.{suffix}");
        Ok(Self {
            fc_weight: weights.require(&key("c_fc.weight"))?.clone(),
            fc_bias: weights.require(&key("c_fc.bias"))?.clone(),
            proj_weight: weights.require(&key("c_proj.weight"))?.clone(),
            proj_bias: weights.require(&key("c_proj.bias"))?.clone(),
            norm_weight: weights.require(&key("norm.weight"))?.clone(),
            norm_bias: weights.require(&key("norm.bias"))?.clone(),
        })
    }

    fn forward(&self, features: &Tensor, dtype: DType) -> Result<Tensor> {
        let (batch, tokens, hidden) = features.dims3()?;
        if tokens != IMAGE_TOKENS || hidden != VISION_HIDDEN {
            return Err(Error::Msg(format!(
                "StarVector-8B SigLIP features must be [batch,{IMAGE_TOKENS},{VISION_HIDDEN}], got {:?}",
                features.dims()
            )));
        }
        let features = features.to_dtype(self.fc_weight.dtype())?;
        let hidden = silu(&__model_proof_output!(linear(
            &features,
            &self.fc_weight,
            Some(&self.fc_bias)
        ))?)?;
        let hidden =
            __model_proof_output!(linear(&hidden, &self.proj_weight, Some(&self.proj_bias)))?;
        let width = IMAGE_TOKENS * DECODER_HIDDEN;
        let flat = hidden.reshape((batch, width))?;
        let weight = self.norm_weight.reshape(width)?;
        let bias = self.norm_bias.reshape(width)?;
        Ok(
            __model_proof_output!(layer_norm(&flat, &weight, &bias, 1e-5))?
                .reshape((batch, IMAGE_TOKENS, DECODER_HIDDEN))?
                .to_dtype(dtype)?,
        )
    }
}

pub struct CandleStarVector8bProvider {
    descriptor: TextLlmDescriptor,
    starvector: StarVectorDescriptor,
    model: StarVector8bModel,
    tokenizer: Tokenizer,
    memory: StarVectorCandleMemory,
    /// The speculative option a request that leaves it unset runs with — this backend's row of
    /// the defaults table ([`core_llm::defaults`], E5). The captioner runs no proposer, so a
    /// non-`off` default is reported as the named no-proposer fallback, as an explicit one is.
    speculative_default: core_llm::Speculative,
}

fn decode_generated_svg_token(
    detok: &mut core_llm::TokenizerDecodeStream,
    id: i32,
) -> CoreResult<Option<String>> {
    detok.step(id as u32)
}

/// The load's quantization refusal: StarVector-8B loads its checkpoint's own encoding only. NVFP4 is
/// refused by name — it is not served for StarVector-8B (the Llama provider serves it for the
/// qwen3_5 hybrid and the llama family, sc-24135 / sc-24140) — and
/// [`crate::backend::nvfp4_support`] answers a product's per-snapshot NVFP4 question with this same
/// gate (sc-24139).
pub(crate) fn quantize_gate(spec: &LoadSpec) -> CoreResult<()> {
    match spec.quantize {
        None => Ok(()),
        Some(core_llm::Quantize::Nvfp4) => Err(CoreError::Unsupported(
            "nvfp4: NVFP4 projections are not served for StarVector-8B (it does not support \
             load-time quantization)"
                .into(),
        )),
        Some(_) => Err(CoreError::Unsupported(
            "StarVector-8B Candle does not support load-time quantization".into(),
        )),
    }
}

impl CandleStarVector8bProvider {
    pub fn load(spec: &LoadSpec) -> CoreResult<Self> {
        quantize_gate(spec)?;
        let dir = Path::new(&spec.source);
        validate_snapshot(dir).map_err(to_core)?;
        let device = crate::device::select_eager_device().map_err(to_core)?;
        let weights = Weights::from_dir(dir, &device).map_err(to_core)?;
        let memory = StarVectorCandleMemory {
            loaded_tensor_count: weights.len(),
            device: format!("{device:?}"),
        };
        let descriptor = descriptor();
        let starvector = starvector_descriptor();
        let tokenizer = Tokenizer::from_hf_byte_level_bpe(
            dir.join("vocab.json"),
            dir.join("merges.txt"),
            dir.join("tokenizer_config.json"),
        )?;
        validate_loaded_svg_context_cap(
            "StarVector-8B",
            &descriptor,
            &starvector,
            tokenizer.encode(SVG_PROMPT, false)?.len(),
        )?;
        Ok(Self {
            descriptor,
            starvector,
            model: StarVector8bModel::from_weights(&weights).map_err(to_core)?,
            tokenizer,
            memory,
            speculative_default: crate::device::decode_defaults(&device).speculative,
        })
    }

    pub fn memory_report(&self) -> StarVectorCandleMemory {
        self.memory.clone()
    }
    /// Consume the loaded provider; tensor ownership is released without touching shared allocator policy.
    pub fn unload(self) {}

    /// One SVG generation plus the measured decode record of its continuation (sc-24139) — `None`
    /// when the bounded stream stopped on the static prefix, before any decode step.
    fn generate_svg_inner(
        &self,
        request: &StarVectorRequest,
        on_event: &mut dyn FnMut(StarVectorStreamEvent),
    ) -> CoreResult<(StarVectorOutput, Option<DecodeRecord>)> {
        self.validate_svg(request)?;
        if request.text_request.cancel.is_cancelled() {
            return Err(CoreError::Canceled);
        }
        let image = image_from_request(&request.text_request)?;
        let prompt: Vec<i32> = self
            .tokenizer
            .encode(SVG_PROMPT, false)?
            .into_iter()
            .map(|id| id as i32)
            .collect();
        let mut stream = StarVectorBoundedStream::new(request);
        match stream.push_static_prefix(SVG_PROMPT)? {
            StarVectorStreamStatus::Continue => on_event(StarVectorStreamEvent::Source {
                text: SVG_PROMPT.into(),
                index: 0,
            }),
            StarVectorStreamStatus::Stop(_) => {
                return Ok((emit_svg_done(stream.output()?, on_event), None))
            }
        }
        let began = Instant::now();
        // The whole request from the conditioning prefill on (the engine's own span starts after
        // it).
        let span = RequestSpan::begin();
        let (first, mut cache) = self.model.prefill(image, &prompt).map_err(to_core)?;
        let config = GenerationConfig {
            max_new_tokens: request.text_request.max_new_tokens as usize,
            sampling: SamplingParams::from(&request.text_request.sampling),
            seed: request.text_request.seed,
            stop_tokens: vec![EOS_TOKEN_ID],
        };
        let mut detok = self.tokenizer.decode_stream(true);
        let stopped = Cell::new(false);
        let failure = RefCell::new(None);
        let mut decode = |event: DecodeEvent| {
            if let DecodeEvent::Token { id, step } = event {
                if stopped.get() {
                    return;
                }
                let delta = match decode_generated_svg_token(&mut detok, id) {
                    Ok(delta) => delta,
                    Err(error) => {
                        *failure.borrow_mut() = Some(error);
                        stopped.set(true);
                        return;
                    }
                };
                let delta = delta.as_deref().unwrap_or("");
                match stream.push(delta, began.elapsed()) {
                    Ok(status @ StarVectorStreamStatus::Continue)
                    | Ok(
                        status @ StarVectorStreamStatus::Stop(StarVectorFinishReason::CompleteRoot),
                    ) => {
                        if !delta.is_empty() {
                            on_event(StarVectorStreamEvent::Source {
                                text: delta.to_owned(),
                                index: step as u32 + 1,
                            });
                        }
                        stopped.set(!matches!(status, StarVectorStreamStatus::Continue));
                    }
                    Ok(StarVectorStreamStatus::Stop(_)) => stopped.set(true),
                    Err(error) => *failure.borrow_mut() = Some(error),
                }
                on_event(StarVectorStreamEvent::Progress {
                    generated_tokens: stream.generated_tokens(),
                });
                if failure.borrow().is_some() {
                    stopped.set(true);
                }
            }
        };
        // A cancel that landed during the prefill is still the typed pre-decode cancellation.
        if request.text_request.cancel.is_cancelled() {
            return Err(CoreError::Canceled);
        }
        let (generated, record) = generate_step_from_prefill(
            &self.model.decoder,
            &mut cache,
            first,
            &prompt,
            &config,
            &request.text_request.cancel,
            &mut decode,
            Some(&|| stopped.get()),
        )
        .map_err(to_core)?;
        if let Some(error) = failure.into_inner() {
            return Err(error);
        }
        if !stopped.get() {
            if let Some(delta) = detok.finish()? {
                let status = stream.push_decoded_suffix(&delta, began.elapsed())?;
                if matches!(
                    status,
                    StarVectorStreamStatus::Continue
                        | StarVectorStreamStatus::Stop(StarVectorFinishReason::CompleteRoot)
                ) {
                    on_event(StarVectorStreamEvent::Source {
                        text: delta,
                        index: generated.tokens.len() as u32,
                    });
                }
                stopped.set(!matches!(status, StarVectorStreamStatus::Continue));
            }
        }
        if !stopped.get() {
            match generated.finish_reason {
                DecodeFinish::StopToken => {
                    stream.finish_eos()?;
                }
                DecodeFinish::MaxTokens | DecodeFinish::Cancelled => {
                    let _ = stream.push("", began.elapsed())?;
                }
                DecodeFinish::Stopped => {}
            }
        }
        Ok((
            emit_svg_done(stream.output()?, on_event),
            Some(record.with_request_span(&span)),
        ))
    }
}

impl TextLlm for CandleStarVector8bProvider {
    fn descriptor(&self) -> &TextLlmDescriptor {
        &self.descriptor
    }
    fn as_starvector_provider(&self) -> Option<&dyn StarVectorProvider> {
        Some(self)
    }
    fn validate(&self, request: &TextLlmRequest) -> CoreResult<()> {
        self.descriptor
            .capabilities
            .validate_request(&self.descriptor.id, request)?;
        image_from_request(request).map(|_| ())
    }
    fn generate(
        &self,
        request: &TextLlmRequest,
        on_event: &mut dyn FnMut(StreamEvent),
    ) -> CoreResult<TextLlmOutput> {
        let prompt_tokens = self.tokenizer.encode(SVG_PROMPT, false)?.len() as u32;
        svg_text_output(
            request,
            IMAGE_TOKENS as u32 + prompt_tokens,
            self.speculative_default,
            on_event,
            |svg_request, events| self.generate_svg_inner(svg_request, events),
        )
    }
}

/// The 8B text answer through the shared SVG scaffolding (sc-27301): the bounded stream's
/// sources reach `on_event` as content tokens and its `Done` as `Done`.
fn svg_text_output(
    request: &TextLlmRequest,
    prompt_tokens: u32,
    speculative_default: core_llm::Speculative,
    on_event: &mut dyn FnMut(StreamEvent),
    decode: impl FnOnce(
        &StarVectorRequest,
        &mut dyn FnMut(StarVectorStreamEvent),
    ) -> CoreResult<(StarVectorOutput, Option<DecodeRecord>)>,
) -> CoreResult<TextLlmOutput> {
    crate::starvector_common::svg_text_output(
        request,
        prompt_tokens,
        MAX_SVG_BYTES,
        MAX_WALL_TIME,
        speculative_default,
        on_event,
        decode,
    )
}
impl StarVectorProvider for CandleStarVector8bProvider {
    fn starvector_descriptor(&self) -> &StarVectorDescriptor {
        &self.starvector
    }
    fn generate_svg(
        &self,
        request: &StarVectorRequest,
        on_event: &mut dyn FnMut(StarVectorStreamEvent),
    ) -> CoreResult<StarVectorOutput> {
        self.generate_svg_inner(request, on_event)
            .map(|(output, _)| output)
    }
}

/// Descriptor used before weight loading by the explicit registry.
pub fn descriptor() -> TextLlmDescriptor {
    core_llm::starvector::starvector_8b::text_descriptor(PROVIDER_ID, "candle")
}

/// Tensor-neutral model facts visible through the shared StarVector contract.
pub fn starvector_descriptor() -> StarVectorDescriptor {
    core_llm::starvector::starvector_8b::starvector_descriptor()
}
pub const REGISTRATION: core_llm::TextLlmRegistration = core_llm::TextLlmRegistration {
    descriptor,
    load: load_registered,
    can_load,
    weightless_vision: None,
    weightless_audio: None,
};
fn load_registered(spec: &LoadSpec) -> CoreResult<Box<dyn TextLlm>> {
    Ok(Box::new(CandleStarVector8bProvider::load(spec)?))
}
pub fn can_load(spec: &LoadSpec) -> bool {
    core_llm::starvector::starvector_8b::snapshot_matches(&spec.source)
}

fn validate_snapshot(dir: &Path) -> Result<()> {
    let config: Value = serde_json::from_str(&crate::__model_input_reader!(
std::fs::read_to_string(dir.join("config.json")),
"crates/llm/candle-llm/src/starvector_8b.rs::validate_snapshot::std::fs::read_to_string@22842",
ArtifactInput,
dir.join("config.json"),
|__model_input_path| std::fs::read_to_string(__model_input_path)
)?)
    .map_err(|error| Error::Config(format!("StarVector-8B config.json: {error}")))?;
    if !exact_config(&config) {
        return Err(Error::Config(format!(
            "expected exact {SNAPSHOT_REPOSITORY}@{SNAPSHOT_REVISION} config"
        )));
    }
    for asset in ["vocab.json", "merges.txt", "tokenizer_config.json"] {
        if !dir.join(asset).is_file() {
            return Err(Error::Config(format!(
                "StarVector-8B snapshot lacks {asset}"
            )));
        }
    }
    Ok(())
}
fn siglip_config() -> SiglipVisionConfig {
    SiglipVisionConfig {
        image_size: IMAGE_SIZE,
        patch_size: 16,
        num_channels: 3,
        hidden_size: VISION_HIDDEN,
        intermediate_size: 4096,
        num_hidden_layers: 24,
        num_attention_heads: 16,
        layer_norm_eps: 1e-6,
    }
}
fn preprocess_image(image: &core_llm::ImageRef, device: &candle_core::Device) -> Result<Tensor> {
    SiglipImageProcessor {
        size: IMAGE_SIZE,
        ..SiglipImageProcessor::default()
    }
    .preprocess(
        &image.pixels,
        image.width as usize,
        image.height as usize,
        device,
    )
}
fn image_from_request(request: &TextLlmRequest) -> CoreResult<&core_llm::ImageRef> {
    core_llm::starvector_conditioning_image("StarVector-8B", request)
}

/// The provider-wide bridge (sc-27300), naming the model in a missing-tensor load failure.
fn to_core(error: Error) -> CoreError {
    match error {
        Error::MissingTensor(key) => {
            CoreError::Load(format!("missing StarVector-8B tensor: {key}"))
        }
        other => crate::error::to_core(other),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::starvector_common::svg_report;
    use core_llm::starvector::starvector_8b::{MAX_CONTEXT_TOKENS, SVG_PROMPT_TOKEN_COUNT};
    use core_llm::{DecoderArchitecture, StarVectorTier};
    use std::time::Duration;

    /// E2/E8 (sc-24432 feature-end review): the captioner advertises no proposer, so an `auto` or
    /// explicit speculative request decodes plainly with the reason named — the same words MLX's
    /// StarVector reports — and its prefix cache's `none` is named too; `off` names no fallback.
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
            let report = svg_report(&record, &request(mode), Speculative::Off);
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
            svg_report(&record, &unset, Speculative::Auto).fallbacks,
            core_llm::no_proposer_fallback(Speculative::Auto, core_llm::CAPTIONER_NO_PROPOSER)
                .into_iter()
                .collect::<Vec<_>>()
        );
        assert!(svg_report(&record, &unset, Speculative::Off)
            .fallbacks
            .is_empty());
        assert!(
            svg_report(&record, &request(Speculative::Off), Speculative::Auto)
                .fallbacks
                .is_empty()
        );
        let off = svg_report(&record, &request(Speculative::Off), Speculative::Off);
        assert!(off.fallbacks.is_empty());
        assert_eq!(off.prefix_cache.path, "none");
        assert_eq!(
            off.prefix_cache.reason.as_deref(),
            Some(core_llm::CAPTIONER_NO_PREFIX_CACHE)
        );
    }
    use core_llm_testkit::{check_starvector_bounded_fixture, StarVectorProfile};
    use serde_json::json;

    const STREAM_TOKENIZER_JSON: &str = r#"{
        "version": "1.0",
        "added_tokens": [],
        "normalizer": null,
        "pre_tokenizer": { "type": "Whitespace" },
        "post_processor": null,
        "decoder": { "type": "Sequence", "decoders": [{ "type": "Fuse" }] },
        "model": {
            "type": "WordLevel",
            "vocab": { "x": 0, "y": 1 },
            "unk_token": "x"
        }
    }"#;
    fn config() -> Value {
        json!({"model_type":"starvector","starcoder_model_name":"bigcode/starcoder2-7b","image_encoder_type":"siglip_384","adapter_norm":"layer_norm","image_size":384,"hidden_size":4608,"num_attention_heads":36,"num_hidden_layers":32,"num_kv_heads":4,"vocab_size":49152})
    }
    /// sc-24139 / E2, E5, E8 (sc-24432), through the seam the provider's `generate` answers with
    /// ([`svg_text_output`] — a real generation needs the published 8B weights, so the bounded
    /// SVG decode is the fixture stream here): the stream's sources and `Done` reach the text
    /// events, and the measured record becomes `TextLlmOutput::decode`, carrying the captioner
    /// reasons — the speculative fallback for `auto` and an explicit proposer, none for `off`, the
    /// prefix cache's `none` named — with an unset option resolved against the provider's
    /// default; a stream that stopped before any decode step reports no record.
    #[test]
    fn generate_reports_the_decode_record_with_the_captioner_reasons() {
        use core_llm::{Message, Speculative, SpeculativeProposer};
        let request = |speculative| TextLlmRequest {
            messages: vec![Message::user("x")],
            speculative,
            max_new_tokens: 64,
            ..Default::default()
        };
        let run = |req: &TextLlmRequest, default, decoded: bool| {
            let (mut tokens, mut done) = (String::new(), Vec::new());
            let out = svg_text_output(
                req,
                IMAGE_TOKENS as u32 + 2,
                default,
                &mut |event| match event {
                    StreamEvent::Token { text, .. } => tokens.push_str(&text),
                    StreamEvent::Done { usage, .. } => done.push(usage),
                },
                |svg, events| {
                    let mut stream = StarVectorBoundedStream::new(svg);
                    let fixture = core_llm_testkit::deterministic_svg_fixture();
                    for (index, fragment) in fixture.fragments.iter().enumerate() {
                        stream.push(fragment, Duration::ZERO)?;
                        events(StarVectorStreamEvent::Source {
                            text: (*fragment).into(),
                            index: index as u32,
                        });
                    }
                    let record = DecodeRecord::plain(
                        crate::decode::DecodePath::StepModel,
                        3,
                        2,
                        Default::default(),
                    );
                    Ok((
                        emit_svg_done(stream.output()?, events),
                        decoded.then_some(record),
                    ))
                },
            )
            .unwrap();
            assert_eq!(tokens, out.text);
            assert_eq!(done, vec![out.usage]);
            assert_eq!(out.usage.prompt_tokens, IMAGE_TOKENS as u32 + 2);
            assert!(out.usage.generated_tokens > 0);
            out.decode
        };
        let named = |mode| {
            core_llm::no_proposer_fallback(mode, core_llm::CAPTIONER_NO_PROPOSER)
                .into_iter()
                .collect::<Vec<_>>()
        };
        let lookup = Speculative::proposer(SpeculativeProposer::PromptLookup, 2);
        for (speculative, default, fallbacks) in [
            (
                Some(Speculative::Auto),
                Speculative::Off,
                named(Speculative::Auto),
            ),
            (Some(lookup), Speculative::Off, named(lookup)),
            (Some(Speculative::Off), Speculative::Off, Vec::new()),
            (None, Speculative::Auto, named(Speculative::Auto)),
            (Some(Speculative::Off), Speculative::Auto, Vec::new()),
            (None, Speculative::Off, Vec::new()),
        ] {
            let case = format!("{speculative:?} over default {default:?}");
            let report = run(&request(speculative), default, true).expect(&case);
            assert_eq!(report.path, "step_model", "{case}");
            assert_eq!(report.fallbacks, fallbacks, "{case}");
            assert_eq!(report.prefix_cache.path, "none", "{case}");
            assert_eq!(
                report.prefix_cache.reason.as_deref(),
                Some(core_llm::CAPTIONER_NO_PREFIX_CACHE),
                "{case}"
            );
        }
        assert!(run(&request(Some(Speculative::Auto)), Speculative::Off, false).is_none());
    }
    /// sc-20683: every SVG text output reports the dense KV cache with the shared reason for its
    /// policy.
    #[test]
    fn svg_text_output_reports_the_dense_kv_cache_for_the_policy() {
        use core_llm::{
            KvCacheFallbackReason as Reason, KvCacheReport, KvCompressionPolicy as Policy,
        };
        for (policy, reason) in [
            (Policy::Off, Reason::PolicyDisabled),
            (Policy::Qualified, Reason::UnqualifiedModel),
        ] {
            let request = TextLlmRequest {
                messages: vec![core_llm::Message::user("x")],
                kv_compression: policy,
                max_new_tokens: 64,
                ..Default::default()
            };
            let output = svg_text_output(
                &request,
                0,
                core_llm::Speculative::Off,
                &mut |_| {},
                |svg, events| {
                    let mut stream = StarVectorBoundedStream::new(svg);
                    for fragment in core_llm_testkit::deterministic_svg_fixture().fragments {
                        stream.push(fragment, Duration::ZERO)?;
                    }
                    Ok((emit_svg_done(stream.output()?, events), None))
                },
            )
            .unwrap();
            assert_eq!(output.kv_cache, Some(KvCacheReport::dense(reason, None)));
        }
    }
    /// sc-24139: NVFP4 is refused by name, any other load-time format with the existing refusal,
    /// and the checkpoint's own encoding passes — the gate the per-snapshot probe asks.
    #[test]
    fn the_quantize_gate_refuses_nvfp4_by_name() {
        let spec = |quantize| LoadSpec {
            quantize,
            ..LoadSpec::dense("/no/such/starvector-8b")
        };
        assert!(quantize_gate(&spec(None)).is_ok());
        match quantize_gate(&spec(Some(core_llm::Quantize::Nvfp4))) {
            Err(CoreError::Unsupported(message)) => {
                assert!(message.starts_with("nvfp4: "), "{message}");
                assert!(message.contains("StarVector-8B"), "{message}");
            }
            other => panic!("expected the NVFP4 refusal, got {other:?}"),
        }
        match quantize_gate(&spec(Some(core_llm::Quantize::Q4))) {
            Err(CoreError::Unsupported(message)) => {
                assert!(message.contains("load-time quantization"), "{message}")
            }
            other => panic!("expected the quantization refusal, got {other:?}"),
        }
        assert!(matches!(
            CandleStarVector8bProvider::load(&spec(Some(core_llm::Quantize::Nvfp4))),
            Err(CoreError::Unsupported(_))
        ));
    }
    #[test]
    fn descriptor_matches_mlx_8b_contract() {
        let d = starvector_descriptor();
        let text = descriptor();
        assert_eq!(text.id, PROVIDER_ID);
        assert_eq!(text.capabilities.max_context_tokens, MAX_CONTEXT_TOKENS);
        assert_eq!(text.capabilities.max_new_tokens, 15_422);
        core_llm::validate_advertised_generated_token_cap(
            text.capabilities.max_new_tokens,
            text.capabilities.max_context_tokens,
            IMAGE_TOKENS + SVG_PROMPT_TOKEN_COUNT,
        )
        .unwrap();
        assert_eq!(d.tier, StarVectorTier::EightB);
        assert_eq!(d.preprocessing.image_size, 384);
        assert_eq!(d.projection.decoder, DecoderArchitecture::StarCoder2);
        assert_eq!(d.projection.image_token_count, 576);
    }
    #[test]
    fn exact_snapshot_admission_rejects_nearby_variants() {
        assert!(exact_config(&config()));
        for (field, replacement) in [
            ("image_size", json!(224)),
            ("hidden_size", json!(2048)),
            ("starcoder_model_name", json!("bigcode/starcoderbase-1b")),
            ("adapter_norm", json!("batch_norm")),
        ] {
            let mut wrong = config();
            wrong[field] = replacement;
            assert!(!exact_config(&wrong), "mutated {field} routed to 8B");
        }
    }
    #[test]
    fn siglip_resize_is_direct_and_nchw() {
        let image = core_llm::ImageRef::new(2, 1, vec![0; 6]).unwrap();
        let pixels = preprocess_image(&image, &candle_core::Device::Cpu).unwrap();
        assert_eq!(pixels.dims(), &[1, 3, 384, 384]);
        assert_eq!(
            pixels.flatten_all().unwrap().to_vec1::<f32>().unwrap()[0],
            -1.0
        );
    }

    #[test]
    fn image_only_mode_rejects_free_form_text_guidance() {
        let image = core_llm::ImageRef::new(1, 1, vec![0; 3]).unwrap();
        let request = TextLlmRequest::new(
            vec![core_llm::Message {
                role: core_llm::Role::User,
                content: vec![
                    core_llm::Content::Text("turn this into an SVG".into()),
                    core_llm::Content::Image(image),
                ],
                thinking: None,
                tool_calls: Vec::new(),
            }],
            16,
        );
        assert!(matches!(
            image_from_request(&request),
            Err(CoreError::Unsupported(_))
        ));
    }
    #[test]
    fn candle_uses_shared_deterministic_svg_fixture() {
        check_starvector_bounded_fixture(
            &StarVectorProfile {
                text: None,
                ..StarVectorProfile::cheap()
            },
            &core_llm_testkit::deterministic_svg_fixture(),
        )
        .unwrap();
    }

    #[test]
    fn continuation_decode_matches_full_decode_for_a_long_sequence() {
        let tokenizer = Tokenizer::from_json(STREAM_TOKENIZER_JSON).unwrap();
        let ids: Vec<u32> = [0, 1].into_iter().cycle().take(4_096).collect();
        let expected = tokenizer.decode(&ids, true).unwrap();
        let mut detok = tokenizer.decode_stream(true);
        let mut actual = String::new();
        for id in ids {
            if let Some(delta) = decode_generated_svg_token(&mut detok, id as i32).unwrap() {
                actual.push_str(&delta);
            }
        }
        if let Some(delta) = detok.finish().unwrap() {
            actual.push_str(&delta);
        }
        assert_eq!(actual, expected);
        assert_eq!(actual.len(), 4_096);
    }
    #[test]
    #[ignore = "sc-22261 terminal real-weight StarVector-8B CUDA campaign only"]
    fn real_weight_provider_satisfies_shared_starvector_conformance() {
        let snapshot = std::env::var("STARVECTOR_8B_SNAPSHOT")
            .expect("sc-22261 must set STARVECTOR_8B_SNAPSHOT to the local exact snapshot");
        let spec = LoadSpec::dense(snapshot);
        let profile = StarVectorProfile {
            image: Some(core_llm::ImageRef::new(2, 2, vec![0x80; 12]).unwrap()),
            text: None,
            max_new_tokens: 4_000,
            max_svg_bytes: 2 * 1024 * 1024,
            max_wall_time: Duration::from_secs(120),
            seed: 7,
        };
        core_llm_testkit::starvector_conformance(
            || Box::new(CandleStarVector8bProvider::load(&spec).unwrap()),
            &profile,
        );

        #[cfg(feature = "model-inference-proof")]
        println!("\nMODEL_INFERENCE_BODY_COMPLETED starvector_8b::tests::real_weight_provider_satisfies_shared_starvector_conformance");
    }
}
