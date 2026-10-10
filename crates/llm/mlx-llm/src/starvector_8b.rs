//! Native MLX provider for the exact `starvector/starvector-8b-im2svg` checkpoint.
//!
//! Loading consumes only local config/tokenizer/safetensors assets. It neither downloads assets nor
//! executes snapshot code, Python, Transformers, or a sidecar.

use std::cell::{Cell, RefCell};
use std::path::Path;
use std::time::{Duration, Instant};

use mlx_rs::ops::concatenate_axis;
use mlx_rs::Array;
use serde_json::Value;

use core_llm::{
    Channel, Content, DecodeReport, DecoderArchitecture, Error as CoreError, FinishReason,
    ImagePreprocessing, LoadSpec, ProjectionMetadata, Result as CoreResult,
    StarVectorBoundedStream, StarVectorDescriptor, StarVectorFinishReason, StarVectorOutput,
    StarVectorProvider, StarVectorRequest, StarVectorStreamEvent, StarVectorStreamStatus,
    StarVectorTier, StreamEvent, TextLlm, TextLlmCapabilities, TextLlmDescriptor, TextLlmOutput,
    TextLlmRequest, Tokenizer, Usage, VisionEncoderArchitecture,
};

use crate::decode::{
    generate_speculative, EngineOptions, FinishReason as DecodeFinish, GenerationConfig,
    NoProposer, SpeculativePrompt, StepTarget, StreamEvent as DecodeEvent,
};
use crate::error::{Error, Result};
use crate::image::SiglipImageProcessor;
use crate::models::{SiglipVisionConfig, SiglipVisionTower, StarCoder2, StarCoder2Config};
use crate::primitives::nn::{layer_norm, linear, silu};
use crate::primitives::sampler::SamplingParams;
use crate::primitives::{input_ids, Weights};

/// Explicit provider id used by the ordinary MLX text registry.
pub const PROVIDER_ID: &str = "mlx-starvector-8b";
const SNAPSHOT_REPOSITORY: &str = "starvector/starvector-8b-im2svg";
const SNAPSHOT_REVISION: &str = "518beea8dcb5f7a37c5911e92d1d62a76beee7f9";
const SVG_PROMPT: &str = "<svg";
const EOS_TOKEN_ID: i32 = 0;
const IMAGE_SIZE: usize = 384;
const IMAGE_TOKENS: i32 = 576;
const VISION_HIDDEN: i32 = 1024;
const DECODER_HIDDEN: i32 = 4608;
const MAX_CONTEXT_TOKENS: usize = 16_000;
// The exact snapshot tokenizer encodes the fixed `<svg` decoder prompt as two IDs; `load`
// verifies this against the local snapshot before provider advertisement is trusted.
const SVG_PROMPT_TOKEN_COUNT: usize = 2;
const MAX_NEW_TOKENS: u32 =
    (MAX_CONTEXT_TOKENS - (IMAGE_TOKENS as usize + SVG_PROMPT_TOKEN_COUNT)) as u32;

/// A fully loaded StarVector-8B MLX model. Dropping it releases all MLX array handles; reload is
/// an ordinary new explicit-registry load.
pub struct StarVector8bModel {
    vision: SiglipVisionTower,
    adapter: StarVector8bAdapter,
    decoder: StarCoder2,
}

impl StarVector8bModel {
    fn from_dir(dir: &Path) -> Result<Self> {
        let weights = Weights::from_dir(dir)?;
        Ok(Self {
            vision: SiglipVisionTower::from_weights(
                &weights,
                "model.image_encoder.visual_encoder",
                siglip_config(),
            )?,
            adapter: StarVector8bAdapter::from_weights(&weights, "model.image_projection")?,
            decoder: StarCoder2::from_weights(
                &weights,
                "model.svg_transformer.transformer",
                StarCoder2Config::STARVECTOR_8B,
            )?,
        })
    }

    fn prefill(
        &self,
        image: &core_llm::ImageRef,
        prompt: &[i32],
    ) -> Result<(Array, Box<dyn crate::primitives::kv_cache::KvCache>)> {
        let pixels = preprocess_image(image)?;
        let vision = self.vision.forward(&pixels)?.last_hidden_state;
        let vision = self.adapter.forward(&vision)?;
        let text = self.decoder.embed(&input_ids(prompt))?;
        let embeds = concatenate_axis(&[&vision, &text], 1)?;
        let mut cache: Box<dyn crate::primitives::kv_cache::KvCache> =
            Box::new(self.decoder.cache());
        let logits = self
            .decoder
            .logits_from_embeds(&embeds, cache.as_mut(), 0)?;
        Ok((logits, cache))
    }
}

/// The exact 8B adapter: `Linear(1024, 2048)` → SiLU → `Linear(2048, 4608)` →
/// `LayerNorm([576, 4608])`.
///
/// The final LayerNorm spans both sequence and hidden dimensions in the upstream module. The MLX
/// layer-norm primitive normalizes its final axis, so the rows are flattened per batch before the
/// call and restored afterward.
struct StarVector8bAdapter {
    /// SigLIP row width ([`VISION_HIDDEN`] for the published snapshot).
    vision_hidden: i32,
    /// Decoder row width ([`DECODER_HIDDEN`] for the published snapshot).
    decoder_hidden: i32,
    fc_weight: Array,
    fc_bias: Array,
    proj_weight: Array,
    proj_bias: Array,
    norm_weight: Array,
    norm_bias: Array,
}

impl StarVector8bAdapter {
    fn from_weights(w: &Weights, prefix: &str) -> Result<Self> {
        let key = |suffix: &str| format!("{prefix}.{suffix}");
        let model = Self {
            vision_hidden: VISION_HIDDEN,
            decoder_hidden: DECODER_HIDDEN,
            fc_weight: w.require(&key("c_fc.weight"))?.clone(),
            fc_bias: w.require(&key("c_fc.bias"))?.clone(),
            proj_weight: w.require(&key("c_proj.weight"))?.clone(),
            proj_bias: w.require(&key("c_proj.bias"))?.clone(),
            norm_weight: w.require(&key("norm.weight"))?.clone(),
            norm_bias: w.require(&key("norm.bias"))?.clone(),
        };
        w.verify_accessed_gpu_view()?;
        Ok(model)
    }

    fn forward(&self, image_features: &Array) -> Result<Array> {
        let shape = image_features.shape();
        let (vision_hidden, decoder_hidden) = (self.vision_hidden, self.decoder_hidden);
        if shape.len() != 3 || shape[1] != IMAGE_TOKENS || shape[2] != vision_hidden {
            return Err(Error::Msg(format!(
                "StarVector-8B SigLIP features must be [batch,{IMAGE_TOKENS},{vision_hidden}], got {shape:?}"
            )));
        }
        let hidden = silu(&linear(
            image_features,
            &self.fc_weight,
            Some(&self.fc_bias),
        )?)?;
        let hidden = linear(&hidden, &self.proj_weight, Some(&self.proj_bias))?;
        let flat_width = IMAGE_TOKENS * decoder_hidden;
        let flat = hidden.reshape(&[shape[0], flat_width])?;
        let weight = self.norm_weight.reshape(&[flat_width])?;
        let bias = self.norm_bias.reshape(&[flat_width])?;
        let normalized = layer_norm(&flat, Some(&weight), Some(&bias), 1e-5)?;
        Ok(normalized.reshape(&[shape[0], IMAGE_TOKENS, decoder_hidden])?)
    }
}

/// MLX-loaded StarVector-8B provider. It remains a `TextLlm`; SVG is the narrow typed extension.
pub struct StarVector8bProvider {
    /// The speculative option a request that leaves it unset runs with — the MLX row of the
    /// defaults table ([`core_llm::defaults::MLX`], E5). The captioner runs no proposer, so a
    /// non-`off` default is reported as the named no-proposer fallback, as an explicit one is.
    speculative_default: core_llm::Speculative,
    descriptor: TextLlmDescriptor,
    starvector: StarVectorDescriptor,
    model: StarVector8bModel,
    tokenizer: Tokenizer,
}

impl StarVector8bProvider {
    /// Load one local, exact StarVector-8B snapshot. This never downloads or executes snapshot code.
    pub fn load(spec: &LoadSpec) -> CoreResult<Self> {
        if spec.quantize.is_some() {
            return Err(CoreError::Unsupported(
                "StarVector-8B MLX does not support load-time quantization".into(),
            ));
        }
        let dir = Path::new(&spec.source);
        validate_snapshot(dir).map_err(to_core)?;
        let descriptor = descriptor();
        let starvector = starvector_descriptor();
        let tokenizer = Tokenizer::from_hf_byte_level_bpe(
            dir.join("vocab.json"),
            dir.join("merges.txt"),
            dir.join("tokenizer_config.json"),
        )?;
        validate_loaded_context_cap(
            &descriptor,
            &starvector,
            tokenizer.encode(SVG_PROMPT, false)?.len(),
        )?;
        Ok(Self {
            speculative_default: core_llm::defaults::MLX.speculative,
            descriptor,
            starvector,
            model: StarVector8bModel::from_dir(dir).map_err(to_core)?,
            tokenizer,
        })
    }

    /// Observe MLX allocator state without changing process-global cache policy.
    pub fn memory_report(&self) -> crate::starvector_1b::StarVectorMlxMemory {
        crate::starvector_1b::StarVectorMlxMemory {
            active_bytes: mlx_rs::memory::get_active_memory(),
            peak_bytes: mlx_rs::memory::get_peak_memory(),
        }
    }

    /// Consume this provider and release its model/tokenizer ownership.
    ///
    /// It deliberately does not clear MLX's process-global allocator cache, which may be shared by
    /// another explicit provider. Re-loading is a new call through the same registry.
    pub fn unload(self) {}

    fn image<'a>(&self, request: &'a TextLlmRequest) -> CoreResult<&'a core_llm::ImageRef> {
        image_from_request(request)
    }

    /// The SVG run and, when the decoder ran, the engine's measured decode report (`None` only
    /// when the stream stopped at the static `<svg` prefix, before any decode).
    fn generate_svg_inner(
        &self,
        request: &StarVectorRequest,
        on_event: &mut dyn FnMut(StarVectorStreamEvent),
    ) -> CoreResult<(StarVectorOutput, Option<DecodeReport>)> {
        self.validate_svg(request)?;
        if request.text_request.cancel.is_cancelled() {
            return Err(CoreError::Canceled);
        }
        let image = self.image(&request.text_request)?;
        let prompt: Vec<i32> = self
            .tokenizer
            .encode(SVG_PROMPT, false)?
            .into_iter()
            .map(|id| id as i32)
            .collect();
        let began = Instant::now();
        let mut guard = StarVectorBoundedStream::new(request);
        match guard.push_static_prefix(SVG_PROMPT)? {
            StarVectorStreamStatus::Continue => on_event(StarVectorStreamEvent::Source {
                text: SVG_PROMPT.into(),
                index: 0,
            }),
            StarVectorStreamStatus::Stop(_) => {
                let output = guard.output()?;
                on_event(StarVectorStreamEvent::Done {
                    finish_reason: output.finish_reason,
                    generated_tokens: output.generated_tokens,
                    generated_bytes: output.generated_bytes,
                });
                return Ok((output, None));
            }
        }
        let (first_logits, mut cache) = self.model.prefill(image, &prompt).map_err(to_core)?;
        let config = GenerationConfig {
            max_new_tokens: request.text_request.max_new_tokens as usize,
            sampling: sampling(&request.text_request.sampling),
            seed: request.text_request.seed,
            stop_tokens: vec![EOS_TOKEN_ID],
        };
        let mut detok = self.tokenizer.decode_stream(true);
        let stopped = Cell::new(false);
        let stream_error = RefCell::new(None);
        let mut decode_event = |event: DecodeEvent| {
            if let DecodeEvent::Token { id, step } = event {
                let delta = match detok.step(id as u32) {
                    Ok(delta) => delta,
                    Err(error) => {
                        *stream_error.borrow_mut() = Some(error);
                        stopped.set(true);
                        return;
                    }
                };
                let delta = delta.as_deref().unwrap_or("");
                let status = match guard.push(delta, began.elapsed()) {
                    Ok(status) => status,
                    Err(error) => {
                        *stream_error.borrow_mut() = Some(error);
                        stopped.set(true);
                        return;
                    }
                };
                on_event(StarVectorStreamEvent::Progress {
                    generated_tokens: guard.generated_tokens(),
                });
                match status {
                    StarVectorStreamStatus::Continue
                    | StarVectorStreamStatus::Stop(StarVectorFinishReason::CompleteRoot) => {
                        if !delta.is_empty() {
                            on_event(StarVectorStreamEvent::Source {
                                text: delta.to_owned(),
                                index: step as u32 + 1,
                            });
                        }
                    }
                    StarVectorStreamStatus::Stop(_) => {}
                }
                stopped.set(!matches!(status, StarVectorStreamStatus::Continue));
            }
        };
        // The decode is the speculative engine's token-at-a-time loop over the step-only decoder,
        // so the SVG run carries a measured `DecodeReport` (epic sc-24432).
        let run = generate_speculative(
            &StepTarget(&self.model.decoder),
            &mut NoProposer,
            SpeculativePrompt::Prefilled {
                cache: &mut cache,
                logits: first_logits,
                hidden: None,
                history: &prompt,
                position_delta: 0,
            },
            &config,
            0,
            &request.text_request.cancel,
            &mut decode_event,
            EngineOptions {
                should_stop: Some(&|| stopped.get()),
                ..EngineOptions::default()
            },
        )
        .map_err(to_core)?;
        let generated = run.output;
        if let Some(error) = stream_error.into_inner() {
            return Err(error);
        }
        if !stopped.get() {
            if let Some(delta) = detok.finish()? {
                let status = guard.push_decoded_suffix(&delta, began.elapsed())?;
                match status {
                    StarVectorStreamStatus::Continue
                    | StarVectorStreamStatus::Stop(StarVectorFinishReason::CompleteRoot) => {
                        on_event(StarVectorStreamEvent::Source {
                            text: delta,
                            index: generated.tokens.len() as u32,
                        });
                    }
                    StarVectorStreamStatus::Stop(_) => {}
                }
                stopped.set(!matches!(status, StarVectorStreamStatus::Continue));
            }
        }
        if !stopped.get() {
            match generated.finish_reason {
                DecodeFinish::StopToken => {
                    guard.finish_eos()?;
                }
                DecodeFinish::MaxTokens | DecodeFinish::Cancelled => {
                    guard.push("", began.elapsed())?;
                }
                DecodeFinish::Stopped => {}
            }
        }
        let output = guard.output()?;
        on_event(StarVectorStreamEvent::Done {
            finish_reason: output.finish_reason,
            generated_tokens: output.generated_tokens,
            generated_bytes: output.generated_bytes,
        });
        Ok((output, Some(run.report)))
    }
}

fn image_from_request(request: &TextLlmRequest) -> CoreResult<&core_llm::ImageRef> {
    if request
        .messages
        .iter()
        .any(|message| !message.text_content().trim().is_empty())
    {
        return Err(CoreError::Unsupported(
            "StarVector-8B im2svg is image-conditioned and does not accept free-form text guidance"
                .into(),
        ));
    }
    let mut image = None;
    for message in &request.messages {
        for content in &message.content {
            if let Content::Image(value) = content {
                if image.replace(value).is_some() {
                    return Err(CoreError::Unsupported(
                        "StarVector-8B accepts exactly one conditioning image".into(),
                    ));
                }
            }
        }
    }
    image.ok_or_else(|| {
        CoreError::InvalidRequest("StarVector-8B requires one conditioning image".into())
    })
}

impl TextLlm for StarVector8bProvider {
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
        self.image(request)?;
        Ok(())
    }

    fn generate(
        &self,
        request: &TextLlmRequest,
        on_event: &mut dyn FnMut(StreamEvent),
    ) -> CoreResult<TextLlmOutput> {
        let svg_request =
            StarVectorRequest::new(request.clone(), 2 * 1024 * 1024, Duration::from_secs(120));
        let prompt_tokens = self.tokenizer.encode(SVG_PROMPT, false)?.len() as u32;
        let (output, report) = self.generate_svg_inner(&svg_request, &mut |event| match event {
            StarVectorStreamEvent::Source { text, index } => {
                on_event(StreamEvent::Token {
                    id: index,
                    text,
                    index: index as usize,
                    channel: Channel::Content,
                });
            }
            StarVectorStreamEvent::Progress { .. } => {}
            StarVectorStreamEvent::Done {
                finish_reason,
                generated_tokens,
                ..
            } => {
                on_event(StreamEvent::Done {
                    finish_reason: map_finish(finish_reason),
                    usage: Usage {
                        prompt_tokens: IMAGE_TOKENS as u32 + prompt_tokens,
                        generated_tokens,
                    },
                });
            }
        })?;
        Ok(svg_text_output(
            output.svg,
            Usage {
                prompt_tokens: IMAGE_TOKENS as u32 + prompt_tokens,
                generated_tokens: output.generated_tokens,
            },
            // StarVector advertises no proposer and has no prefix cache: the request's speculative
            // fallback and the prefix-cache reason join the measured report in the shared words
            // Candle's StarVector uses — never a silent downgrade (E2, E8).
            report.map(|report| {
                report.with_captioner_reasons(request.speculative_or(self.speculative_default))
            }),
            map_finish(output.finish_reason),
            request.kv_compression,
        ))
    }
}

/// The text output of one SVG generation: the SVG source (empty when none closed), its measured
/// `decode` report, and — the StarVector wrapper having no compressed-KV table family — the dense
/// KV-cache report for the request's `policy` (sc-20683).
fn svg_text_output(
    svg: Option<String>,
    usage: Usage,
    decode: Option<core_llm::DecodeReport>,
    finish: FinishReason,
    policy: core_llm::KvCompressionPolicy,
) -> TextLlmOutput {
    TextLlmOutput {
        timings: None,
        text: svg.unwrap_or_default(),
        thinking: None,
        tool_calls: Vec::new(),
        usage,
        mtp: None,
        decode,
        finish_reason: Some(finish),
        kv_cache: Some(core_llm::KvCacheReport::without_table_family(policy)),
    }
}

impl StarVectorProvider for StarVector8bProvider {
    fn starvector_descriptor(&self) -> &StarVectorDescriptor {
        &self.starvector
    }

    fn generate_svg(
        &self,
        request: &StarVectorRequest,
        on_event: &mut dyn FnMut(StarVectorStreamEvent),
    ) -> CoreResult<StarVectorOutput> {
        Ok(self.generate_svg_inner(request, on_event)?.0)
    }
}

/// Descriptor used before weight loading by the existing explicit registry.
pub fn descriptor() -> TextLlmDescriptor {
    TextLlmDescriptor {
        id: PROVIDER_ID.into(),
        family: "starvector".into(),
        backend: "mlx".into(),
        capabilities: TextLlmCapabilities {
            max_context_tokens: MAX_CONTEXT_TOKENS,
            max_new_tokens: MAX_NEW_TOKENS,
            supports_system_prompt: false,
            supports_vision: true,
            supports_video: false,
            supports_audio: false,
            supports_thinking: false,
            supports_reasoning_effort: false,
            reasoning_efforts: Vec::new(),
            model_sampling_defaults: None,
            supports_preserve_thinking: false,
            supports_tools: false,
            mtp: None,
            speculative: Vec::new(),
            supported_constraints: Vec::new(),
        },
    }
}

fn validate_loaded_context_cap(
    descriptor: &TextLlmDescriptor,
    starvector: &StarVectorDescriptor,
    prompt_tokens: usize,
) -> CoreResult<()> {
    let prefill_tokens = usize::try_from(starvector.projection.image_token_count)
        .map_err(|_| {
            CoreError::InvalidRequest("StarVector-8B image prefix does not fit usize".into())
        })?
        .checked_add(prompt_tokens)
        .ok_or_else(|| {
            CoreError::InvalidRequest("StarVector-8B prefill token count overflow".into())
        })?;
    core_llm::validate_advertised_generated_token_cap(
        descriptor.capabilities.max_new_tokens,
        descriptor.capabilities.max_context_tokens,
        prefill_tokens,
    )
}

/// Tensor-neutral model facts visible through the shared StarVector contract.
pub fn starvector_descriptor() -> StarVectorDescriptor {
    StarVectorDescriptor {
        tier: StarVectorTier::EightB,
        preprocessing: ImagePreprocessing {
            image_size: IMAGE_SIZE as u32,
            channels: 3,
            preserve_aspect_ratio: false,
        },
        projection: ProjectionMetadata {
            vision_encoder: VisionEncoderArchitecture::Siglip,
            decoder: DecoderArchitecture::StarCoder2,
            vision_hidden_size: VISION_HIDDEN as u32,
            decoder_hidden_size: DECODER_HIDDEN as u32,
            image_token_count: IMAGE_TOKENS as u32,
        },
        max_svg_bytes: 2 * 1024 * 1024,
        max_wall_time: Some(Duration::from_secs(120)),
    }
}

/// Explicit runtime registration; no global constructors or side effects.
pub const REGISTRATION: core_llm::TextLlmRegistration = core_llm::TextLlmRegistration {
    descriptor,
    load: load_registered,
    can_load,
    weightless_vision: None,
    weightless_audio: None,
};

fn load_registered(spec: &LoadSpec) -> CoreResult<Box<dyn TextLlm>> {
    Ok(Box::new(StarVector8bProvider::load(spec)?))
}

/// Weightless structural probe used by the existing model-first registry selection.
pub fn can_load(spec: &LoadSpec) -> bool {
    let path = Path::new(&spec.source).join("config.json");
    std::fs::read_to_string(path)
        .ok()
        .and_then(|text| serde_json::from_str::<Value>(&text).ok())
        .is_some_and(|config| exact_config(&config))
}

fn validate_snapshot(dir: &Path) -> Result<()> {
    let text = std::fs::read_to_string(dir.join("config.json"))?;
    let config: Value = serde_json::from_str(&text)
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

fn exact_config(config: &Value) -> bool {
    config.get("model_type").and_then(Value::as_str) == Some("starvector")
        && config.get("starcoder_model_name").and_then(Value::as_str)
            == Some("bigcode/starcoder2-7b")
        && config.get("image_encoder_type").and_then(Value::as_str) == Some("siglip_384")
        && config.get("adapter_norm").and_then(Value::as_str) == Some("layer_norm")
        && config.get("image_size").and_then(Value::as_i64) == Some(384)
        && config.get("hidden_size").and_then(Value::as_i64) == Some(4608)
        && config.get("num_attention_heads").and_then(Value::as_i64) == Some(36)
        && config.get("num_hidden_layers").and_then(Value::as_i64) == Some(32)
        && config.get("num_kv_heads").and_then(Value::as_i64) == Some(4)
        && config.get("vocab_size").and_then(Value::as_i64) == Some(49_152)
}

fn siglip_config() -> SiglipVisionConfig {
    SiglipVisionConfig {
        image_size: IMAGE_SIZE as i32,
        patch_size: 16,
        num_channels: 3,
        hidden_size: VISION_HIDDEN,
        intermediate_size: 4096,
        num_hidden_layers: 24,
        num_attention_heads: 16,
        layer_norm_eps: 1e-6,
    }
}

fn preprocess_image(image: &core_llm::ImageRef) -> Result<Array> {
    SiglipImageProcessor {
        size: IMAGE_SIZE,
        ..SiglipImageProcessor::default()
    }
    .preprocess(&image.pixels, image.width as usize, image.height as usize)
}

fn sampling(value: &core_llm::Sampling) -> SamplingParams {
    SamplingParams {
        temperature: value.temperature,
        top_p: value.top_p,
        top_k: value.top_k,
        presence_penalty: value.presence_penalty,
        repetition_penalty: value.repetition_penalty,
        repetition_context: value.repetition_context,
    }
}

fn map_finish(reason: StarVectorFinishReason) -> FinishReason {
    match reason {
        StarVectorFinishReason::CompleteRoot | StarVectorFinishReason::Eos => FinishReason::Stop,
        StarVectorFinishReason::TokenLimit
        | StarVectorFinishReason::ByteLimit
        | StarVectorFinishReason::WallTimeLimit => FinishReason::Length,
        StarVectorFinishReason::Cancelled => FinishReason::Cancelled,
    }
}

fn to_core(error: Error) -> CoreError {
    match error {
        Error::Canceled => CoreError::Canceled,
        Error::Unsupported(message) => CoreError::Unsupported(message),
        Error::MissingTensor(key) => {
            CoreError::Load(format!("missing StarVector-8B tensor: {key}"))
        }
        Error::Config(message) => CoreError::Load(message),
        Error::Io(error) => CoreError::Io(error),
        other => CoreError::backend(other),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;
    use std::env;

    use core_llm_testkit::{
        check_starvector_bounded_fixture, starvector_conformance, StarVectorProfile,
    };

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
            let output = svg_text_output(None, Usage::default(), None, FinishReason::Stop, policy);
            assert_eq!(output.kv_cache, Some(KvCacheReport::dense(reason, None)));
        }
    }

    fn exact_snapshot_config() -> Value {
        json!({
            "model_type": "starvector",
            "starcoder_model_name": "bigcode/starcoder2-7b",
            "image_encoder_type": "siglip_384",
            "adapter_norm": "layer_norm",
            "image_size": 384,
            "hidden_size": 4608,
            "num_attention_heads": 36,
            "num_hidden_layers": 32,
            "num_kv_heads": 4,
            "vocab_size": 49152,
        })
    }

    #[test]
    fn descriptor_binds_the_published_8b_geometry() {
        let text = descriptor();
        let star = starvector_descriptor();
        assert_eq!(text.id, PROVIDER_ID);
        assert!(text.capabilities.supports_vision);
        assert_eq!(text.capabilities.max_context_tokens, MAX_CONTEXT_TOKENS);
        assert_eq!(text.capabilities.max_new_tokens, 15_422);
        core_llm::validate_advertised_generated_token_cap(
            text.capabilities.max_new_tokens,
            text.capabilities.max_context_tokens,
            IMAGE_TOKENS as usize + SVG_PROMPT_TOKEN_COUNT,
        )
        .unwrap();
        assert_eq!(star.tier, StarVectorTier::EightB);
        assert_eq!(star.preprocessing.image_size, 384);
        assert!(!star.preprocessing.preserve_aspect_ratio);
        assert_eq!(star.projection.vision_hidden_size, 1024);
        assert_eq!(star.projection.decoder_hidden_size, 4608);
        assert_eq!(star.projection.image_token_count, 576);
        assert_eq!(star.projection.decoder, DecoderArchitecture::StarCoder2);
    }

    #[test]
    fn exact_snapshot_probe_rejects_nearby_starvector_variants() {
        assert!(exact_config(&exact_snapshot_config()));
        for (field, replacement) in [
            ("image_size", json!(224)),
            ("hidden_size", json!(2048)),
            ("starcoder_model_name", json!("bigcode/starcoderbase-1b")),
            ("adapter_norm", json!("batch_norm")),
        ] {
            let mut wrong = exact_snapshot_config();
            wrong[field] = replacement;
            assert!(
                !exact_config(&wrong),
                "mutated {field} must not route to the 8B provider"
            );
        }
    }

    #[test]
    fn siglip_preprocess_resizes_without_preserving_aspect_ratio() {
        let image = core_llm::ImageRef::new(2, 1, vec![0; 6]).unwrap();
        let pixels = preprocess_image(&image).unwrap();
        assert_eq!(pixels.shape(), &[1, 384, 384, 3]);
        // A direct resize keeps the black source everywhere, unlike the 1B white-padding path.
        assert_eq!(pixels.as_slice::<f32>()[0], -1.0);
        assert_eq!(pixels.as_slice::<f32>()[(192 * 384 + 192) * 3], -1.0);
    }

    #[test]
    fn image_only_8b_rejects_free_form_text_guidance() {
        let image = core_llm::ImageRef::new(1, 1, vec![0; 3]).unwrap();
        let request = TextLlmRequest::new(
            vec![core_llm::Message {
                role: core_llm::Role::User,
                content: vec![
                    Content::Text("turn this into an SVG".into()),
                    Content::Image(image),
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
    fn mlx_uses_the_shared_deterministic_greedy_svg_fixture() {
        let profile = StarVectorProfile {
            text: None,
            ..StarVectorProfile::cheap()
        };
        check_starvector_bounded_fixture(&profile, &core_llm_testkit::deterministic_svg_fixture())
            .unwrap();
    }

    /// Terminal real-weight parity hook. It is deliberately ignored in ordinary story-local checks:
    /// sc-22261 owns the single permitted quality/admission campaign. The hook neither downloads
    /// nor invokes Python; it only opens the explicit local snapshot supplied by that terminal story.
    #[test]
    #[ignore = "sc-22261 terminal real-weight StarVector-8B campaign only"]
    fn real_weight_provider_satisfies_shared_starvector_conformance() {
        let snapshot = env::var("STARVECTOR_8B_SNAPSHOT")
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
        starvector_conformance(
            || Box::new(StarVector8bProvider::load(&spec).unwrap()),
            &profile,
        );
    }

    // ---- The engine decode (epic sc-24432, story sc-24434) on a shape-valid synthetic model. ----

    use crate::decode::{
        generate_speculative, EngineOptions, NoProposer, SpeculativePrompt, StepTarget,
    };
    use crate::synthetic::{word_tokenizer, Synth};

    const VOCAB: i32 = 40;

    /// A shape-valid StarVector-8B: a one-layer, 8-wide SigLIP tower at the published 384 px /
    /// 576-row geometry, the LayerNorm adapter at those widths, and a tiny random StarCoder2 whose
    /// tied head scores EOS exactly zero (so greedy decoding runs to the budget).
    fn tiny_provider() -> StarVector8bProvider {
        let vision_cfg = SiglipVisionConfig {
            hidden_size: 8,
            intermediate_size: 16,
            num_hidden_layers: 1,
            num_attention_heads: 2,
            ..siglip_config()
        };
        let (width, inner, hidden) = (vision_cfg.hidden_size, 16, 16);
        let patch = vision_cfg.patch_size;
        let mut w = Synth::new(0x57A2_008B);
        let v = "model.image_encoder.visual_encoder";
        let e = format!("{v}.encoder.layers.0");
        w.randn(
            format!("{v}.embeddings.patch_embedding.weight"),
            &[width, 3, patch, patch],
        )
        .randn(format!("{v}.embeddings.patch_embedding.bias"), &[width])
        .randn(
            format!("{v}.embeddings.position_embedding.weight"),
            &[IMAGE_TOKENS, width],
        )
        .layer_norm(&format!("{e}.layer_norm1"), width)
        .layer_norm(&format!("{e}.layer_norm2"), width)
        .linear(&format!("{e}.mlp.fc1"), inner, width)
        .linear(&format!("{e}.mlp.fc2"), width, inner)
        .layer_norm(&format!("{v}.post_layernorm"), width);
        for proj in ["q_proj", "k_proj", "v_proj", "out_proj"] {
            w.linear(&format!("{e}.self_attn.{proj}"), width, width);
        }
        let a = "model.image_projection";
        w.linear(&format!("{a}.c_fc"), inner, width)
            .linear(&format!("{a}.c_proj"), hidden, inner)
            .fill(format!("{a}.norm.weight"), &[IMAGE_TOKENS, hidden], 1.0)
            .fill(format!("{a}.norm.bias"), &[IMAGE_TOKENS, hidden], 0.0);
        let d = "model.svg_transformer.transformer";
        let l = format!("{d}.model.layers.0");
        let embed = format!("{d}.model.embed_tokens.weight");
        w.randn(embed.clone(), &[VOCAB, hidden])
            .zero_row(&embed, EOS_TOKEN_ID)
            .layer_norm(&format!("{d}.model.norm"), hidden)
            .layer_norm(&format!("{l}.input_layernorm"), hidden)
            .layer_norm(&format!("{l}.post_attention_layernorm"), hidden)
            .linear(&format!("{l}.self_attn.q_proj"), hidden, hidden)
            .linear(&format!("{l}.self_attn.k_proj"), hidden / 2, hidden)
            .linear(&format!("{l}.self_attn.v_proj"), hidden / 2, hidden)
            .linear(&format!("{l}.self_attn.o_proj"), hidden, hidden)
            .linear(&format!("{l}.mlp.c_fc"), 32, hidden)
            .linear(&format!("{l}.mlp.c_proj"), hidden, 32)
            // Sharp attention, so the decode depends on RoPE positions.
            .scale(&format!("{l}.self_attn.q_proj.weight"), 8.0)
            .scale(&format!("{l}.self_attn.k_proj.weight"), 8.0);
        let weights = w.weights();
        let mut adapter = StarVector8bAdapter::from_weights(&weights, a).unwrap();
        adapter.vision_hidden = width;
        adapter.decoder_hidden = hidden;
        StarVector8bProvider {
            speculative_default: core_llm::defaults::MLX.speculative,
            descriptor: descriptor(),
            starvector: starvector_descriptor(),
            model: StarVector8bModel {
                vision: SiglipVisionTower::from_weights(&weights, v, vision_cfg).unwrap(),
                adapter,
                decoder: StarCoder2::from_weights(
                    &weights,
                    d,
                    StarCoder2Config {
                        vocab_size: VOCAB,
                        hidden_size: hidden,
                        intermediate_size: 32,
                        layers: 1,
                        heads: 2,
                        kv_heads: 1,
                        rope_theta: 10_000.0,
                        layer_norm_eps: 1e-5,
                    },
                )
                .unwrap(),
            },
            tokenizer: word_tokenizer(VOCAB as usize, &[]),
        }
    }

    fn image_request(speculative: Option<core_llm::Speculative>) -> TextLlmRequest {
        TextLlmRequest {
            messages: vec![core_llm::Message {
                role: core_llm::Role::User,
                content: vec![Content::Image(
                    core_llm::ImageRef::new(
                        8,
                        8,
                        (0..8 * 8 * 3).map(|i| (i * 37 % 256) as u8).collect(),
                    )
                    .unwrap(),
                )],
                thinking: None,
                tool_calls: Vec::new(),
            }],
            sampling: core_llm::Sampling::greedy(),
            max_new_tokens: 8,
            seed: Some(3),
            speculative,
            ..Default::default()
        }
    }

    /// The engine port emits exactly the pre-engine `generate_from_prefill` loop's tokens on the
    /// synthetic model's image-conditioned prefill — greedy, and seeded stochastic (whose draws
    /// are sensitive to small logit differences a greedy argmax can absorb).
    #[test]
    fn the_engine_decode_is_the_pre_engine_loop() {
        let provider = tiny_provider();
        let request = image_request(None);
        let image = image_from_request(&request).unwrap();
        let prompt: Vec<i32> = provider
            .tokenizer
            .encode(SVG_PROMPT, false)
            .unwrap()
            .into_iter()
            .map(|id| id as i32)
            .collect();
        let stochastic = SamplingParams {
            temperature: 1.5,
            ..SamplingParams::default()
        };
        for (name, params) in [
            ("greedy", sampling(&request.sampling)),
            ("stochastic", stochastic),
        ] {
            let config = GenerationConfig {
                max_new_tokens: 8,
                sampling: params,
                seed: request.seed,
                stop_tokens: vec![EOS_TOKEN_ID],
            };
            let (logits, mut cache) = provider.model.prefill(image, &prompt).unwrap();
            let expected = crate::decode::generate_from_prefill(
                &provider.model.decoder,
                cache.as_mut(),
                logits,
                prompt.clone(),
                &config,
                &core_llm::CancelFlag::new(),
                &mut |_| {},
                None,
                None,
            )
            .unwrap();
            assert!(
                expected.tokens.len() > 1,
                "{name}: the fixture steps the decoder"
            );
            let (logits, mut cache) = provider.model.prefill(image, &prompt).unwrap();
            let run = generate_speculative(
                &StepTarget(&provider.model.decoder),
                &mut NoProposer,
                SpeculativePrompt::Prefilled {
                    cache: &mut cache,
                    logits,
                    hidden: None,
                    history: &prompt,
                    position_delta: 0,
                },
                &config,
                0,
                &core_llm::CancelFlag::new(),
                &mut |_| {},
                EngineOptions::default(),
            )
            .unwrap();
            assert_eq!(run.output.tokens, expected.tokens, "{name}");
            assert_eq!(run.output.finish_reason, expected.finish_reason, "{name}");
        }
    }

    /// AC2 end to end: an SVG generation carries `decode`, naming the token-at-a-time path, no
    /// proposer, the measured sampler, and the fallback an `auto` request resolves to on a
    /// provider that advertises no proposer.
    #[test]
    fn the_provider_reports_its_decode_and_the_auto_fallback() {
        let mut provider = tiny_provider();
        let out = provider
            .generate(
                &image_request(Some(core_llm::Speculative::Auto)),
                &mut |_| {},
            )
            .unwrap();
        let report = out.decode.expect("an SVG run reports its decode path");
        assert_eq!(report.path, "step_model");
        assert_eq!(report.proposer, core_llm::ProposerKind::None);
        assert_eq!(report.draft_tokens, None);
        assert_eq!(report.sampler, "device");
        assert_eq!(out.usage.generated_tokens, 8);
        assert_eq!(
            report.verify_steps + 1,
            u64::from(out.usage.generated_tokens)
        );
        // The captioner's own reason, in the words Candle's twin reports (E2, E8) — never the
        // generic "no MTP head, no prompt lookup on this backend" (MLX runs prompt lookup).
        assert_eq!(
            report.fallbacks,
            core_llm::no_proposer_fallback(
                core_llm::Speculative::Auto,
                core_llm::CAPTIONER_NO_PROPOSER
            )
            .into_iter()
            .collect::<Vec<_>>()
        );
        assert!(report.fallbacks[0].contains("advertises no proposer"));
        assert_eq!(report.prefix_cache.path, "none");
        assert_eq!(
            report.prefix_cache.reason.as_deref(),
            Some(core_llm::CAPTIONER_NO_PREFIX_CACHE)
        );
        let off = provider
            .generate(&image_request(None), &mut |_| {})
            .unwrap();
        assert_eq!(off.text, out.text, "the fallback decodes plainly");
        assert!(off.decode.unwrap().fallbacks.is_empty());
        // E5: an unset option takes the provider's per-backend default, so a table `auto` decodes
        // plainly with the same named no-proposer fallback as an explicit `auto`.
        provider.speculative_default = core_llm::Speculative::Auto;
        let defaulted = provider
            .generate(&image_request(None), &mut |_| {})
            .unwrap();
        assert_eq!(
            defaulted.text, out.text,
            "the defaulted fallback decodes plainly"
        );
        assert_eq!(defaulted.decode.unwrap().fallbacks, report.fallbacks);
        provider.speculative_default = core_llm::Speculative::Off;
        // An explicit proposer it does not advertise is not refused: it decodes plainly, named.
        let lookup =
            core_llm::Speculative::proposer(core_llm::SpeculativeProposer::PromptLookup, 2);
        let explicit = provider
            .generate(&image_request(Some(lookup)), &mut |_| {})
            .unwrap();
        assert_eq!(
            explicit.text, out.text,
            "the explicit fallback decodes plainly"
        );
        assert_eq!(
            explicit.decode.unwrap().fallbacks,
            core_llm::no_proposer_fallback(lookup, core_llm::CAPTIONER_NO_PROPOSER)
                .into_iter()
                .collect::<Vec<_>>()
        );
    }
}
