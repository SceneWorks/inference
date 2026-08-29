//! Native MLX provider for the exact `starvector/starvector-1b-im2svg` checkpoint.
//!
//! Loading is wholly native Rust: config/tokenizer/safetensors are consumed locally and no
//! Transformers, Python, remote-code hook, or model sidecar participates at runtime.

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
use crate::image::resize_bicubic_u8;
use crate::models::{
    GptBigCode, GptBigCodeConfig, StarVectorAdapter, StarVectorClipVision, IMAGE_SIZE, IMAGE_TOKENS,
};
use crate::primitives::sampler::SamplingParams;
use crate::primitives::{input_ids, Weights};

/// Explicit provider id used by the ordinary MLX text registry.
pub const PROVIDER_ID: &str = "mlx-starvector-1b";
const SNAPSHOT_REPOSITORY: &str = "starvector/starvector-1b-im2svg";
const SVG_PROMPT: &str = "<svg";
const EOS_TOKEN_ID: i32 = 0;

/// Snapshot of the MLX allocator while this provider is loaded.
///
/// MLX arrays share one process-wide allocator, so these are allocator observations rather than an
/// invented per-model estimate. Callers that need a before/after lifecycle receipt can sample this
/// before loading, after loading, and after [`StarVector1bProvider::unload`].
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct StarVectorMlxMemory {
    /// Bytes currently retained by the MLX allocator.
    pub active_bytes: usize,
    /// High-water bytes observed by the MLX allocator since process start/reset.
    pub peak_bytes: usize,
}

/// A fully loaded StarVector-1B MLX model. Dropping it releases all MLX array handles; reload is
/// an ordinary new registry load, so lifecycle ownership never escapes the provider instance.
pub struct StarVector1bModel {
    vision: StarVectorClipVision,
    adapter: StarVectorAdapter,
    decoder: GptBigCode,
}

impl StarVector1bModel {
    fn from_dir(dir: &Path) -> Result<Self> {
        let weights = Weights::from_dir(dir)?;
        Ok(Self {
            vision: StarVectorClipVision::from_weights(&weights, "model.image_encoder")?,
            adapter: StarVectorAdapter::from_weights(&weights, "model.image_projection")?,
            decoder: GptBigCode::from_weights(
                &weights,
                "model.svg_transformer.transformer",
                GptBigCodeConfig::STARVECTOR_1B,
            )?,
        })
    }

    fn prefill(
        &self,
        image: &core_llm::ImageRef,
        prompt: &[i32],
    ) -> Result<(Array, Box<dyn crate::primitives::kv_cache::KvCache>)> {
        let pixels = preprocess_image(image)?;
        let vision = self.adapter.forward(&self.vision.forward(&pixels)?)?;
        let text = self.decoder.raw_embed(&input_ids(prompt))?;
        let embeds = concatenate_axis(&[&vision, &text], 1)?;
        let embeds = self.decoder.position_embeds(&embeds, 0)?;
        let mut cache: Box<dyn crate::primitives::kv_cache::KvCache> =
            Box::new(self.decoder.cache());
        let logits = self.decoder.logits_from_embeds(&embeds, cache.as_mut())?;
        Ok((logits, cache))
    }
}

/// MLX-loaded StarVector provider. It remains a `TextLlm`; SVG is the narrow typed extension.
pub struct StarVector1bProvider {
    descriptor: TextLlmDescriptor,
    starvector: StarVectorDescriptor,
    model: StarVector1bModel,
    tokenizer: Tokenizer,
}

impl StarVector1bProvider {
    /// Load one local, exact StarVector-1B snapshot. This never downloads or executes snapshot code.
    pub fn load(spec: &LoadSpec) -> CoreResult<Self> {
        if spec.quantize.is_some() {
            return Err(CoreError::Unsupported(
                "StarVector-1B MLX does not support load-time quantization".into(),
            ));
        }
        let dir = Path::new(&spec.source);
        validate_snapshot(dir).map_err(to_core)?;
        Ok(Self {
            descriptor: descriptor(),
            starvector: starvector_descriptor(),
            model: StarVector1bModel::from_dir(dir).map_err(to_core)?,
            tokenizer: Tokenizer::from_file(dir.join("tokenizer.json"))?,
        })
    }

    /// Observe MLX allocator state without mutating global allocator/cache policy.
    pub fn memory_report(&self) -> StarVectorMlxMemory {
        StarVectorMlxMemory {
            active_bytes: mlx_rs::memory::get_active_memory(),
            peak_bytes: mlx_rs::memory::get_peak_memory(),
        }
    }

    /// Consume this provider and release its model/tokenizer ownership.
    ///
    /// This intentionally does not clear MLX's process-global cache, because unrelated explicit
    /// providers may share it. Re-loading is a new call to [`Self::load`] through the same registry.
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
        let params = sampling(&request.text_request.sampling);
        let config = GenerationConfig {
            max_new_tokens: request.text_request.max_new_tokens as usize,
            sampling: params,
            seed: request.text_request.seed,
            stop_tokens: vec![EOS_TOKEN_ID],
        };
        let mut tokens = Vec::new();
        let mut detok = IncrementalDetok::new();
        let stopped = Cell::new(false);
        // Decode callbacks cannot return an error, so retain a malformed-source failure and return
        // it after the generic loop exits. Never turn a guard rejection into a successful stop.
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
                            // The fixed `<svg` prefill is emitted at index zero; sampled
                            // continuation token indices start immediately after it.
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
                DecodeFinish::MaxTokens => {
                    guard.push("", began.elapsed())?;
                }
                DecodeFinish::Cancelled => {
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
            "StarVector-1B is image-conditioned and does not accept free-form text guidance".into(),
        ));
    }
    let mut image = None;
    for message in &request.messages {
        for content in &message.content {
            if let Content::Image(value) = content {
                if image.replace(value).is_some() {
                    return Err(CoreError::Unsupported(
                        "StarVector-1B accepts exactly one conditioning image".into(),
                    ));
                }
            }
        }
    }
    image.ok_or_else(|| {
        CoreError::InvalidRequest("StarVector-1B requires one conditioning image".into())
    })
}

impl TextLlm for StarVector1bProvider {
    fn descriptor(&self) -> &TextLlmDescriptor {
        &self.descriptor
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
                        prompt_tokens: IMAGE_TOKENS as u32 + 1,
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
                prompt_tokens: IMAGE_TOKENS as u32 + 1,
                generated_tokens: output.generated_tokens,
            },
            finish_reason: Some(map_finish(output.finish_reason)),
        })
    }
}

impl StarVectorProvider for StarVector1bProvider {
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
            max_context_tokens: 8_192,
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
        tier: StarVectorTier::OneB,
        preprocessing: ImagePreprocessing {
            image_size: IMAGE_SIZE as u32,
            channels: 3,
            preserve_aspect_ratio: true,
        },
        projection: ProjectionMetadata {
            vision_encoder: VisionEncoderArchitecture::Clip,
            decoder: DecoderArchitecture::GptBigCode,
            vision_hidden_size: 1024,
            decoder_hidden_size: 2048,
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
    Ok(Box::new(StarVector1bProvider::load(spec)?))
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
        .map_err(|error| Error::Config(format!("StarVector-1B config.json: {error}")))?;
    if !exact_config(&config) {
        return Err(Error::Config(format!(
            "expected exact {SNAPSHOT_REPOSITORY} config"
        )));
    }
    if !dir.join("tokenizer.json").is_file() {
        return Err(Error::Config(
            "StarVector-1B snapshot lacks tokenizer.json".into(),
        ));
    }
    Ok(())
}

fn exact_config(config: &Value) -> bool {
    config.get("model_type").and_then(Value::as_str) == Some("starvector")
        && config.get("starcoder_model_name").and_then(Value::as_str)
            == Some("bigcode/starcoderbase-1b")
        && config.get("image_encoder_type").and_then(Value::as_str) == Some("clip")
        && config.get("image_size").and_then(Value::as_i64) == Some(224)
        && config.get("hidden_size").and_then(Value::as_i64) == Some(2048)
        && config.get("num_hidden_layers").and_then(Value::as_i64) == Some(24)
}

#[allow(clippy::excessive_precision)] // Published OpenAI CLIP preprocessing constants.
fn preprocess_image(image: &core_llm::ImageRef) -> Result<Array> {
    let (width, height) = (image.width as usize, image.height as usize);
    let side = width.max(height);
    let mut padded = vec![255u8; side * side * 3];
    let x = (side - width) / 2;
    let y = (side - height) / 2;
    for row in 0..height {
        padded[((y + row) * side + x) * 3..((y + row) * side + x + width) * 3]
            .copy_from_slice(&image.pixels[row * width * 3..(row + 1) * width * 3]);
    }
    let resized = resize_bicubic_u8(&padded, side, side, IMAGE_SIZE, IMAGE_SIZE)?;
    let mean = [0.48145466, 0.4578275, 0.40821073];
    let std = [0.26862954, 0.26130258, 0.27577711];
    let values: Vec<f32> = resized
        .chunks_exact(3)
        .flat_map(|pixel| {
            (0..3).map(move |channel| (pixel[channel] / 255.0 - mean[channel]) / std[channel])
        })
        .collect();
    Ok(Array::from_slice(
        &values,
        &[1, IMAGE_SIZE as i32, IMAGE_SIZE as i32, 3],
    ))
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
            CoreError::Load(format!("missing StarVector-1B tensor: {key}"))
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
            "starcoder_model_name": "bigcode/starcoderbase-1b",
            "image_encoder_type": "clip",
            "image_size": 224,
            "hidden_size": 2048,
            "num_hidden_layers": 24,
        })
    }

    #[test]
    fn descriptor_binds_the_published_1b_geometry() {
        let text = descriptor();
        let star = starvector_descriptor();
        assert_eq!(text.id, PROVIDER_ID);
        assert!(text.capabilities.supports_vision);
        assert_eq!(star.tier, StarVectorTier::OneB);
        assert_eq!(star.preprocessing.image_size, 224);
        assert_eq!(star.projection.vision_hidden_size, 1024);
        assert_eq!(star.projection.decoder_hidden_size, 2048);
        assert_eq!(star.projection.image_token_count, 257);
        assert_eq!(star.projection.decoder, DecoderArchitecture::GptBigCode);
    }

    #[test]
    fn exact_snapshot_probe_rejects_nearby_starvector_variants() {
        assert!(exact_config(&exact_snapshot_config()));
        for (field, replacement) in [
            ("image_size", json!(384)),
            ("hidden_size", json!(4096)),
            ("starcoder_model_name", json!("bigcode/starcoderbase-3b")),
        ] {
            let mut wrong = exact_snapshot_config();
            wrong[field] = replacement;
            assert!(
                !exact_config(&wrong),
                "mutated {field} must not route to the 1B provider"
            );
        }
    }

    #[test]
    fn clip_preprocess_centers_short_side_on_white_padding() {
        let image = core_llm::ImageRef::new(2, 1, vec![0; 6]).unwrap();
        let pixels = preprocess_image(&image).unwrap();
        assert_eq!(pixels.shape(), &[1, 224, 224, 3]);
        // The centred black source and white padding cannot normalize to the same value.
        let values = pixels.as_slice::<f32>();
        assert_ne!(values[0], values[(112 * 224 + 112) * 3]);
    }

    #[test]
    fn image_only_1b_rejects_free_form_text_guidance() {
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

    /// Terminal real-weight hook. It is deliberately ignored in ordinary and story-local CPU
    /// checks: sc-22261 owns the one permitted measurement/capability campaign. The hook neither
    /// downloads nor invokes Python; it only opens the explicit local snapshot when that terminal
    /// story provides `STARVECTOR_1B_SNAPSHOT`.
    #[test]
    #[ignore = "sc-22261 terminal real-weight StarVector-1B campaign only"]
    fn real_weight_provider_satisfies_shared_starvector_conformance() {
        let snapshot = env::var("STARVECTOR_1B_SNAPSHOT")
            .expect("sc-22261 must set STARVECTOR_1B_SNAPSHOT to the local exact snapshot");
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
            || Box::new(StarVector1bProvider::load(&spec).unwrap()),
            &profile,
        );
    }
}
