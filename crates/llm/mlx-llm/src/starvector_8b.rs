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
    Channel, Content, DecoderArchitecture, Error as CoreError, FinishReason, ImagePreprocessing,
    IncrementalDetok, LoadSpec, ProjectionMetadata, Result as CoreResult, StarVectorBoundedStream,
    StarVectorDescriptor, StarVectorFinishReason, StarVectorOutput, StarVectorProvider,
    StarVectorRequest, StarVectorStreamEvent, StarVectorStreamStatus, StarVectorTier, StreamEvent,
    TextLlm, TextLlmCapabilities, TextLlmDescriptor, TextLlmOutput, TextLlmRequest, Tokenizer,
    Usage, VisionEncoderArchitecture,
};

use crate::decode::{
    generate_from_prefill, FinishReason as DecodeFinish, GenerationConfig,
    StreamEvent as DecodeEvent,
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
        if shape.len() != 3 || shape[1] != IMAGE_TOKENS || shape[2] != VISION_HIDDEN {
            return Err(Error::Msg(format!(
                "StarVector-8B SigLIP features must be [batch,{IMAGE_TOKENS},{VISION_HIDDEN}], got {shape:?}"
            )));
        }
        let hidden = silu(&linear(
            image_features,
            &self.fc_weight,
            Some(&self.fc_bias),
        )?)?;
        let hidden = linear(&hidden, &self.proj_weight, Some(&self.proj_bias))?;
        let flat_width = IMAGE_TOKENS * DECODER_HIDDEN;
        let flat = hidden.reshape(&[shape[0], flat_width])?;
        let weight = self.norm_weight.reshape(&[flat_width])?;
        let bias = self.norm_bias.reshape(&[flat_width])?;
        let normalized = layer_norm(&flat, Some(&weight), Some(&bias), 1e-5)?;
        Ok(normalized.reshape(&[shape[0], IMAGE_TOKENS, DECODER_HIDDEN])?)
    }
}

/// MLX-loaded StarVector-8B provider. It remains a `TextLlm`; SVG is the narrow typed extension.
pub struct StarVector8bProvider {
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
        Ok(Self {
            descriptor: descriptor(),
            starvector: starvector_descriptor(),
            model: StarVector8bModel::from_dir(dir).map_err(to_core)?,
            tokenizer: Tokenizer::from_hf_byte_level_bpe(
                dir.join("vocab.json"),
                dir.join("merges.txt"),
                dir.join("tokenizer_config.json"),
            )?,
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

    fn generate_svg_inner(
        &self,
        request: &StarVectorRequest,
        on_event: &mut dyn FnMut(StarVectorStreamEvent),
    ) -> CoreResult<StarVectorOutput> {
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
                return Ok(output);
            }
        }
        let (first_logits, mut cache) = self.model.prefill(image, &prompt).map_err(to_core)?;
        let config = GenerationConfig {
            max_new_tokens: request.text_request.max_new_tokens as usize,
            sampling: sampling(&request.text_request.sampling),
            seed: request.text_request.seed,
            stop_tokens: vec![EOS_TOKEN_ID],
        };
        let mut tokens = Vec::new();
        let mut detok = IncrementalDetok::new();
        let stopped = Cell::new(false);
        let stream_error = RefCell::new(None);
        let tokenizer = &self.tokenizer;
        let mut decode_event = |event: DecodeEvent| {
            if let DecodeEvent::Token { id, step } = event {
                tokens.push(id as u32);
                let Ok(text) = tokenizer.decode(&tokens, true) else {
                    return;
                };
                let Some(delta) = detok.push(&text) else {
                    return;
                };
                let status = match guard.push(delta, began.elapsed()) {
                    Ok(status) => status,
                    Err(error) => {
                        *stream_error.borrow_mut() = Some(error);
                        stopped.set(true);
                        return;
                    }
                };
                match status {
                    StarVectorStreamStatus::Continue
                    | StarVectorStreamStatus::Stop(StarVectorFinishReason::CompleteRoot) => {
                        on_event(StarVectorStreamEvent::Source {
                            text: delta.to_owned(),
                            index: step as u32 + 1,
                        });
                    }
                    StarVectorStreamStatus::Stop(_) => {}
                }
                stopped.set(!matches!(status, StarVectorStreamStatus::Continue));
            }
        };
        let generated = generate_from_prefill(
            &self.model.decoder,
            cache.as_mut(),
            first_logits,
            prompt,
            &config,
            &request.text_request.cancel,
            &mut decode_event,
            None,
            Some(&|| stopped.get()),
        )
        .map_err(to_core)?;
        if let Some(error) = stream_error.into_inner() {
            return Err(error);
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
        Ok(output)
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
        let output = self.generate_svg_inner(&svg_request, &mut |event| match event {
            StarVectorStreamEvent::Source { text, index } => {
                on_event(StreamEvent::Token {
                    id: index,
                    text,
                    index: index as usize,
                    channel: Channel::Content,
                });
            }
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
        Ok(TextLlmOutput {
            text: output.svg.unwrap_or_default(),
            thinking: None,
            tool_calls: Vec::new(),
            usage: Usage {
                prompt_tokens: IMAGE_TOKENS as u32 + prompt_tokens,
                generated_tokens: output.generated_tokens,
            },
            finish_reason: Some(map_finish(output.finish_reason)),
        })
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
        self.generate_svg_inner(request, on_event)
    }
}

/// Descriptor used before weight loading by the existing explicit registry.
pub fn descriptor() -> TextLlmDescriptor {
    TextLlmDescriptor {
        id: PROVIDER_ID.into(),
        family: "starvector".into(),
        backend: "mlx".into(),
        capabilities: TextLlmCapabilities {
            max_context_tokens: 16_000,
            max_new_tokens: 4_000,
            supports_system_prompt: false,
            supports_vision: true,
            supports_video: false,
            supports_audio: false,
            supports_thinking: false,
            supports_tools: false,
            supported_constraints: Vec::new(),
        },
    }
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
}
