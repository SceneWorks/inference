//! Exact, tensor-free StarVector-1B snapshot admission and image preparation.
//!
//! StarVector checkpoints use a custom Transformers wrapper.  Candle must never execute that
//! wrapper: this module accepts only the published 1B image-to-SVG shape and records the native
//! preprocessing and weight-tree facts needed by the Candle model implementation.

use std::path::Path;
use std::sync::Mutex;

use candle_core::{Device, Tensor};
use serde_json::Value;

use crate::decode::stream::default_seed;
use crate::error::{Error, Result};
use crate::image::resize_bicubic_u8;
use crate::models::StarVectorModel;
use crate::primitives::sampler::{sample, SamplingParams, SplitMix64};
use crate::primitives::{input_ids, Weights};

/// The only snapshot family this provider admits.
pub const MODEL_TYPE: &str = "starvector";
/// The upstream decoder identity pinned by the published 1B snapshot.
pub const STARCODER_BASE_1B: &str = "bigcode/starcoderbase-1b";
/// The SVG image-to-text model's fixed preprocessing edge.
pub const IMAGE_SIZE: usize = 224;
/// CLIP ViT-L/14 produces a class token and 16 by 16 patch tokens.
pub const IMAGE_TOKEN_COUNT: usize = 257;
/// The fixed CLIP ViT-L/14 projection width.
pub const VISION_HIDDEN_SIZE: usize = 1024;
/// The GPTBigCode/StarCoderBase residual width.
pub const DECODER_HIDDEN_SIZE: usize = 2048;
/// The exact resized StarCoder vocabulary, including the model's three added tokens.
pub const VOCAB_SIZE: usize = 49_156;
/// Literal decoder prompt which is also the beginning of the published SVG source.
const SVG_PROMPT: &str = "<svg";
/// StarCoderBase's `<|endoftext|>` id. It terminates generation and is never decoded as source.
const EOS_TOKEN_ID: i32 = 0;
const MAX_CONTEXT_TOKENS: usize = 8_192;
// The loaded snapshot tokenizer must still encode the fixed decoder prompt as two IDs; `load`
// checks that before the descriptor is trusted by request validation.
const SVG_PROMPT_TOKEN_COUNT: usize = 2;
const MAX_NEW_TOKENS: u32 =
    (MAX_CONTEXT_TOKENS - (IMAGE_TOKEN_COUNT + SVG_PROMPT_TOKEN_COUNT)) as u32;

/// Join the half-precision vision adapter output to the decoder's dense embedding stream.
///
/// The published 1B snapshot deliberately mixes f16 CLIP/adapter tensors with f32 GPTBigCode
/// tensors, so the decoder embeddings are the dtype and device authority at this boundary.
fn concatenate_conditioning_embeddings(vision: &Tensor, text: &Tensor) -> Result<Tensor> {
    let vision = vision.to_dtype(text.dtype())?.to_device(text.device())?;
    Ok(Tensor::cat(&[&vision, text], 1)?)
}

/// Sample one vocabulary row, preserving the provider's single-image shape contract.
fn next_token_id(
    logits: &Tensor,
    history: &[i32],
    sampling: &core_llm::Sampling,
    rng: &mut SplitMix64,
) -> Result<i32> {
    let dims = logits.dims();
    let vocabulary = dims.last().copied().unwrap_or(0);
    if vocabulary == 0
        || dims[..dims.len().saturating_sub(1)]
            .iter()
            .product::<usize>()
            != 1
    {
        return Err(Error::Msg(format!(
            "starvector token selection requires one nonempty vocabulary row; got {dims:?}"
        )));
    }
    sample(
        logits,
        history,
        &SamplingParams {
            temperature: sampling.temperature,
            top_p: sampling.top_p,
            top_k: sampling.top_k,
            presence_penalty: sampling.presence_penalty,
            repetition_penalty: sampling.repetition_penalty,
            repetition_context: sampling.repetition_context,
        },
        rng,
        None,
    )
}

#[derive(Debug, PartialEq, Eq)]
enum DecodedSvgToken {
    Eos,
    Hidden,
    Source(String),
}

/// Decode one sampled continuation token with bounded tokenizer state.
fn decode_generated_svg_token(
    detok: &mut core_llm::TokenizerDecodeStream,
    id: i32,
) -> core_llm::Result<DecodedSvgToken> {
    if id == EOS_TOKEN_ID {
        return Ok(DecodedSvgToken::Eos);
    }
    Ok(match detok.step(id as u32)? {
        Some(delta) => DecodedSvgToken::Source(delta),
        None => DecodedSvgToken::Hidden,
    })
}

fn seed_svg_prompt(
    stream: &mut core_llm::StarVectorBoundedStream<'_>,
    events: &mut dyn FnMut(core_llm::StarVectorStreamEvent),
) -> core_llm::Result<core_llm::StarVectorStreamStatus> {
    let status = stream.push_static_prefix(SVG_PROMPT)?;
    if status == core_llm::StarVectorStreamStatus::Continue {
        events(core_llm::StarVectorStreamEvent::Source {
            text: SVG_PROMPT.into(),
            index: 0,
        });
    }
    Ok(status)
}

fn push_decoded_svg_token(
    stream: &mut core_llm::StarVectorBoundedStream<'_>,
    decoded: DecodedSvgToken,
    elapsed: std::time::Duration,
    index: u32,
    events: &mut dyn FnMut(core_llm::StarVectorStreamEvent),
) -> core_llm::Result<core_llm::StarVectorStreamStatus> {
    let status = match decoded {
        DecodedSvgToken::Eos => stream.finish_eos(),
        DecodedSvgToken::Hidden => stream.push("", elapsed),
        DecodedSvgToken::Source(text) => {
            let status = stream.push(&text, elapsed)?;
            if matches!(
                status,
                core_llm::StarVectorStreamStatus::Continue
                    | core_llm::StarVectorStreamStatus::Stop(
                        core_llm::StarVectorFinishReason::CompleteRoot
                    )
            ) {
                events(core_llm::StarVectorStreamEvent::Source { text, index });
            }
            Ok(status)
        }
    }?;
    events(core_llm::StarVectorStreamEvent::Progress {
        generated_tokens: stream.generated_tokens(),
    });
    Ok(status)
}

fn emit_done(
    output: core_llm::StarVectorOutput,
    events: &mut dyn FnMut(core_llm::StarVectorStreamEvent),
) -> core_llm::StarVectorOutput {
    events(core_llm::StarVectorStreamEvent::Done {
        finish_reason: output.finish_reason,
        generated_tokens: output.generated_tokens,
        generated_bytes: output.generated_bytes,
    });
    output
}

/// Native-only, exact configuration facts for the StarVector-1B image-to-SVG snapshot.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct StarVectorConfig {
    pub image_size: usize,
    pub image_token_count: usize,
    pub vision_hidden_size: usize,
    pub hidden_size: usize,
    pub vocab_size: usize,
    pub max_positions: usize,
    pub num_layers: usize,
    pub num_heads: usize,
    pub multi_query: bool,
}

impl StarVectorConfig {
    /// Parse and require every architecture value which affects native tensor interpretation.
    pub fn from_json(value: &Value) -> Result<Self> {
        let number = |key: &str| -> Result<usize> {
            value
                .get(key)
                .and_then(Value::as_u64)
                .map(|v| v as usize)
                .ok_or_else(|| {
                    Error::Config(format!("starvector: config.json missing integer `{key}`"))
                })
        };
        let text = |key: &str| -> Result<&str> {
            value.get(key).and_then(Value::as_str).ok_or_else(|| {
                Error::Config(format!("starvector: config.json missing string `{key}`"))
            })
        };
        if text("model_type")? != MODEL_TYPE
            || text("starcoder_model_name")? != STARCODER_BASE_1B
            || text("image_encoder_type")? != "clip"
        {
            return Err(Error::Unsupported(
                "starvector: snapshot is not the exact StarVector-1B CLIP/StarCoderBase model"
                    .into(),
            ));
        }
        let cfg = Self {
            image_size: number("image_size")?,
            image_token_count: IMAGE_TOKEN_COUNT,
            vision_hidden_size: VISION_HIDDEN_SIZE,
            hidden_size: number("hidden_size")?,
            vocab_size: number("vocab_size")?,
            max_positions: number("max_position_embeddings")?,
            num_layers: number("num_hidden_layers")?,
            num_heads: number("num_attention_heads")?,
            multi_query: value
                .get("multi_query")
                .and_then(Value::as_bool)
                .ok_or_else(|| {
                    Error::Config("starvector: config.json missing boolean `multi_query`".into())
                })?,
        };
        cfg.validate()?;
        Ok(cfg)
    }

    /// Reject compatible-looking checkpoints whose parameter geometry differs from the pinned 1B.
    pub fn validate(&self) -> Result<()> {
        let exact = self.image_size == IMAGE_SIZE
            && self.hidden_size == DECODER_HIDDEN_SIZE
            && self.vocab_size == VOCAB_SIZE
            && self.max_positions == 8192
            && self.num_layers == 24
            && self.num_heads == 16
            && self.multi_query;
        if !exact {
            return Err(Error::Unsupported(format!(
                "starvector: expected 1B geometry image={IMAGE_SIZE}, hidden={DECODER_HIDDEN_SIZE}, vocab={VOCAB_SIZE}, positions=8192, layers=24, heads=16, multi_query=true; got {self:?}"
            )));
        }
        Ok(())
    }
}

/// Read and validate `config.json` only. This accepts either a snapshot directory or a direct
/// config path and deliberately never opens a safetensors shard.
pub fn read_config(dir: impl AsRef<Path>) -> Result<StarVectorConfig> {
    let source = dir.as_ref();
    let path = if source.is_dir() {
        source.join("config.json")
    } else {
        source.to_path_buf()
    };
    let text = std::fs::read_to_string(&path)?;
    let value: Value = serde_json::from_str(&text)
        .map_err(|error| Error::Config(format!("starvector config {}: {error}", path.display())))?;
    StarVectorConfig::from_json(&value)
}

/// Weightless registry probe for the exact 1B snapshot directory.
pub fn can_load_path(dir: impl AsRef<Path>) -> bool {
    can_load_path_with(dir.as_ref(), |path| read_config(path).is_ok())
}

fn can_load_path_with(dir: &Path, read_config: impl FnOnce(&Path) -> bool) -> bool {
    dir.is_dir() && read_config(dir)
}

/// StarVector's published image processor: pad RGB to a white square, bicubic-resize to 224, and
/// normalize to CLIP's RGB mean/std.  Output is NCHW `[1, 3, 224, 224]` in f32.
#[derive(Clone, Debug)]
pub struct StarVectorImageProcessor {
    pub size: usize,
    pub mean: [f32; 3],
    pub std: [f32; 3],
}

impl Default for StarVectorImageProcessor {
    #[allow(clippy::excessive_precision)] // Published CLIP processor constants; truncating changes conditioning.
    fn default() -> Self {
        Self {
            size: IMAGE_SIZE,
            mean: [0.481_454_66, 0.457_827_5, 0.408_210_73],
            std: [0.268_629_54, 0.261_302_58, 0.275_777_11],
        }
    }
}

impl StarVectorImageProcessor {
    /// Return the exact padded RGB bytes before resizing; public for deterministic fixture tests.
    pub fn pad_square_white(
        &self,
        pixels: &[u8],
        width: usize,
        height: usize,
    ) -> Result<(Vec<u8>, usize)> {
        if width == 0
            || height == 0
            || pixels.len() != width.saturating_mul(height).saturating_mul(3)
        {
            return Err(Error::Msg(format!(
                "starvector preprocess: expected {} RGB bytes for {width}x{height}, got {}",
                width.saturating_mul(height).saturating_mul(3),
                pixels.len()
            )));
        }
        let side = width.max(height);
        let left = (side - width) / 2;
        let top = (side - height) / 2;
        let mut out = vec![255u8; side * side * 3];
        for y in 0..height {
            let src = y * width * 3;
            let dst = ((top + y) * side + left) * 3;
            out[dst..dst + width * 3].copy_from_slice(&pixels[src..src + width * 3]);
        }
        Ok((out, side))
    }

    /// Native Candle preprocessing.  The shared image contract is RGB-only, so alpha conversion is
    /// intentionally not accepted at this boundary.
    pub fn preprocess(
        &self,
        pixels: &[u8],
        width: usize,
        height: usize,
        device: &Device,
    ) -> Result<Tensor> {
        let (padded, side) = self.pad_square_white(pixels, width, height)?;
        let resized = if side == self.size {
            padded.into_iter().map(f32::from).collect()
        } else {
            resize_bicubic_u8(&padded, side, side, self.size, self.size)?
        };
        let plane = self.size * self.size;
        let mut chw = vec![0f32; plane * 3];
        for (pixel_index, rgb) in resized.chunks_exact(3).enumerate() {
            for channel in 0..3 {
                chw[channel * plane + pixel_index] =
                    (rgb[channel] / 255.0 - self.mean[channel]) / self.std[channel];
            }
        }
        Ok(Tensor::from_vec(chw, (1, 3, self.size, self.size), device)?)
    }
}

/// Required checkpoint roots.  The published snapshot is a custom wrapper, so these keys protect
/// us from accidentally accepting a stock CLIP or stock StarCoder directory.
pub const REQUIRED_WEIGHT_KEYS: &[&str] = &[
    "model.image_encoder.visual_encoder.conv1.weight",
    "model.image_encoder.visual_encoder.class_embedding",
    "model.image_encoder.visual_encoder.positional_embedding",
    "model.image_encoder.ln_vision.weight",
    "model.image_projection.c_fc.weight",
    "model.image_projection.c_proj.weight",
    "model.svg_transformer.transformer.transformer.wte.weight",
    "model.svg_transformer.transformer.transformer.wpe.weight",
    "model.svg_transformer.transformer.transformer.ln_f.weight",
];

/// Registered native Candle provider for the exact local StarVector-1B snapshot.
pub const PROVIDER_ID: &str = "candle-starvector-1b";
pub struct CandleStarVectorProvider {
    descriptor: core_llm::TextLlmDescriptor,
    svg: core_llm::StarVectorDescriptor,
    processor: StarVectorImageProcessor,
    /// The device the weights were loaded on; request pixels go here (sc-24134).
    device: Device,
    tokenizer: core_llm::Tokenizer,
    prompt: Vec<i32>,
    model: Mutex<StarVectorModel>,
}
impl CandleStarVectorProvider {
    pub fn load(spec: &core_llm::LoadSpec) -> core_llm::Result<Self> {
        read_config(&spec.source).map_err(to_core)?;
        let device = crate::device::select_device().map_err(to_core)?;
        let weights = Weights::from_dir(&spec.source, &device).map_err(to_core)?;
        for key in REQUIRED_WEIGHT_KEYS {
            if !weights.contains(key) {
                return Err(core_llm::Error::Load(format!(
                    "starvector missing checkpoint tensor `{key}`"
                )));
            }
        }
        let tokenizer =
            core_llm::Tokenizer::from_file(Path::new(&spec.source).join("tokenizer.json"))?;
        let prompt: Vec<i32> = tokenizer
            .encode(SVG_PROMPT, false)?
            .into_iter()
            .map(|id| id as i32)
            .collect();
        let descriptor = descriptor();
        let svg = svg_descriptor();
        validate_loaded_context_cap(&descriptor, &svg, prompt.len())?;
        Ok(Self {
            descriptor,
            svg,
            processor: StarVectorImageProcessor::default(),
            tokenizer,
            prompt,
            model: Mutex::new(StarVectorModel::from_weights(&weights).map_err(to_core)?),
            device,
        })
    }
}
pub fn descriptor() -> core_llm::TextLlmDescriptor {
    core_llm::TextLlmDescriptor {
        id: PROVIDER_ID.into(),
        family: "starvector-1b".into(),
        backend: "candle".into(),
        capabilities: core_llm::TextLlmCapabilities {
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
            supported_constraints: vec![],
        },
    }
}

fn validate_loaded_context_cap(
    descriptor: &core_llm::TextLlmDescriptor,
    starvector: &core_llm::StarVectorDescriptor,
    prompt_tokens: usize,
) -> core_llm::Result<()> {
    let prefill_tokens = usize::try_from(starvector.projection.image_token_count)
        .map_err(|_| {
            core_llm::Error::InvalidRequest("StarVector-1B image prefix does not fit usize".into())
        })?
        .checked_add(prompt_tokens)
        .ok_or_else(|| {
            core_llm::Error::InvalidRequest("StarVector-1B prefill token count overflow".into())
        })?;
    core_llm::validate_advertised_generated_token_cap(
        descriptor.capabilities.max_new_tokens,
        descriptor.capabilities.max_context_tokens,
        prefill_tokens,
    )
}
pub fn svg_descriptor() -> core_llm::StarVectorDescriptor {
    core_llm::StarVectorDescriptor {
        tier: core_llm::StarVectorTier::OneB,
        preprocessing: core_llm::ImagePreprocessing {
            image_size: 224,
            channels: 3,
            preserve_aspect_ratio: true,
        },
        projection: core_llm::ProjectionMetadata {
            vision_encoder: core_llm::VisionEncoderArchitecture::Clip,
            decoder: core_llm::DecoderArchitecture::GptBigCode,
            vision_hidden_size: 1024,
            decoder_hidden_size: 2048,
            image_token_count: 257,
        },
        max_svg_bytes: 2 * 1024 * 1024,
        max_wall_time: Some(std::time::Duration::from_secs(120)),
    }
}
impl core_llm::TextLlm for CandleStarVectorProvider {
    fn descriptor(&self) -> &core_llm::TextLlmDescriptor {
        &self.descriptor
    }
    fn as_starvector_provider(&self) -> Option<&dyn core_llm::StarVectorProvider> {
        Some(self)
    }
    fn validate(&self, req: &core_llm::TextLlmRequest) -> core_llm::Result<()> {
        self.descriptor
            .capabilities
            .validate_request(&self.descriptor.id, req)?;
        if !req.has_image() {
            return Err(core_llm::Error::InvalidRequest(
                "starvector requires one RGB image".into(),
            ));
        }
        Ok(())
    }
    fn generate(
        &self,
        req: &core_llm::TextLlmRequest,
        events: &mut dyn FnMut(core_llm::StreamEvent),
    ) -> core_llm::Result<core_llm::TextLlmOutput> {
        self.validate(req)?;
        if req.cancel.is_cancelled() {
            return Err(core_llm::Error::Canceled);
        };
        let svg = core_llm::StarVectorRequest::new(
            req.clone(),
            1_000_000,
            std::time::Duration::from_secs(120),
        );
        let out = core_llm::StarVectorProvider::generate_svg(self, &svg, &mut |_| {})?;
        let finish = match out.finish_reason {
            core_llm::StarVectorFinishReason::Cancelled => core_llm::FinishReason::Cancelled,
            core_llm::StarVectorFinishReason::TokenLimit => core_llm::FinishReason::Length,
            _ => core_llm::FinishReason::Stop,
        };
        let usage = core_llm::Usage {
            prompt_tokens: (IMAGE_TOKEN_COUNT + self.prompt.len()) as u32,
            generated_tokens: out.generated_tokens,
        };
        events(core_llm::StreamEvent::Done {
            finish_reason: finish,
            usage,
        });
        Ok(core_llm::TextLlmOutput {
            timings: None,
            text: out.svg.unwrap_or_default(),
            thinking: None,
            tool_calls: vec![],
            usage,
            mtp: None,
            finish_reason: Some(finish),
        })
    }
}
impl core_llm::StarVectorProvider for CandleStarVectorProvider {
    fn starvector_descriptor(&self) -> &core_llm::StarVectorDescriptor {
        &self.svg
    }
    fn generate_svg(
        &self,
        req: &core_llm::StarVectorRequest,
        events: &mut dyn FnMut(core_llm::StarVectorStreamEvent),
    ) -> core_llm::Result<core_llm::StarVectorOutput> {
        self.validate_svg(req)?;
        if req.text_request.cancel.is_cancelled() {
            return Err(core_llm::Error::Canceled);
        }
        let image = req
            .text_request
            .messages
            .iter()
            .flat_map(|m| m.content.iter())
            .find_map(|c| {
                if let core_llm::Content::Image(i) = c {
                    Some(i)
                } else {
                    None
                }
            })
            .ok_or_else(|| core_llm::Error::InvalidRequest("starvector requires image".into()))?;
        let started = std::time::Instant::now();
        let pixels = self
            .processor
            .preprocess(
                &image.pixels,
                image.width as usize,
                image.height as usize,
                &self.device,
            )
            .map_err(to_core)?;
        let mut model = self
            .model
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        model.reset();
        let result = (|| {
            let vision = model.image_embeddings(&pixels).map_err(to_core)?;
            let ids = input_ids(&self.prompt, pixels.device())
                .map_err(|e| core_llm::Error::Msg(e.to_string()))?;
            let text = model.decoder.embeddings(&ids).map_err(to_core)?;
            let initial = concatenate_conditioning_embeddings(&vision, &text).map_err(to_core)?;
            let mut logits = model.decoder.forward_embeds(&initial, 0).map_err(to_core)?;
            let mut stream = core_llm::StarVectorBoundedStream::new(req);
            match seed_svg_prompt(&mut stream, events)? {
                core_llm::StarVectorStreamStatus::Continue => {}
                core_llm::StarVectorStreamStatus::Stop(_) => {
                    return Ok(emit_done(stream.output()?, events));
                }
            }
            let mut history = self.prompt.clone();
            let mut rng = SplitMix64::new(req.text_request.seed.unwrap_or_else(default_seed));
            let mut detok = self.tokenizer.decode_stream(true);
            for index in 0..req.text_request.max_new_tokens {
                if req.text_request.cancel.is_cancelled() {
                    break;
                }
                let id = next_token_id(&logits, &history, &req.text_request.sampling, &mut rng)
                    .map_err(to_core)?;
                history.push(id);
                if id == EOS_TOKEN_ID {
                    if let Some(delta) = detok.finish()? {
                        let status = stream.push_decoded_suffix(&delta, started.elapsed())?;
                        if matches!(
                            status,
                            core_llm::StarVectorStreamStatus::Continue
                                | core_llm::StarVectorStreamStatus::Stop(
                                    core_llm::StarVectorFinishReason::CompleteRoot
                                )
                        ) {
                            events(core_llm::StarVectorStreamEvent::Source { text: delta, index });
                        }
                        if matches!(status, core_llm::StarVectorStreamStatus::Stop(_)) {
                            break;
                        }
                    }
                }
                let decoded = decode_generated_svg_token(&mut detok, id)?;
                let status = push_decoded_svg_token(
                    &mut stream,
                    decoded,
                    started.elapsed(),
                    index + 1,
                    events,
                )?;
                if matches!(status, core_llm::StarVectorStreamStatus::Stop(_)) {
                    break;
                }
                let next = input_ids(&[id], pixels.device())
                    .map_err(|e| core_llm::Error::Msg(e.to_string()))?;
                let embed = model.decoder.embeddings(&next).map_err(to_core)?;
                logits = model
                    .decoder
                    .forward_embeds(
                        &embed,
                        IMAGE_TOKEN_COUNT + self.prompt.len() + index as usize,
                    )
                    .map_err(to_core)?;
            }
            if stream.output().is_err() {
                if let Some(delta) = detok.finish()? {
                    let status = stream.push_decoded_suffix(&delta, started.elapsed())?;
                    if matches!(
                        status,
                        core_llm::StarVectorStreamStatus::Continue
                            | core_llm::StarVectorStreamStatus::Stop(
                                core_llm::StarVectorFinishReason::CompleteRoot
                            )
                    ) {
                        events(core_llm::StarVectorStreamEvent::Source {
                            text: delta,
                            index: stream.generated_tokens(),
                        });
                    }
                }
            }
            if stream.output().is_err() {
                // Convert a loop boundary (token budget or mid-stream cancellation) into the
                // bounded stream's existing typed terminal reason without publishing partial SVG.
                let _ = stream.push("", started.elapsed())?;
            }
            Ok(emit_done(stream.output()?, events))
        })();
        model.reset();
        result
    }
}
pub const REGISTRATION: core_llm::TextLlmRegistration = core_llm::TextLlmRegistration {
    descriptor,
    load: |spec| Ok(Box::new(CandleStarVectorProvider::load(spec)?)),
    can_load: |spec| can_load_path(&spec.source),
    weightless_vision: None,
    weightless_audio: None,
};
fn to_core(error: Error) -> core_llm::Error {
    match error {
        Error::Canceled => core_llm::Error::Canceled,
        Error::Unsupported(message) => core_llm::Error::Unsupported(message),
        Error::Config(message) | Error::MissingTensor(message) => core_llm::Error::Load(message),
        Error::Io(error) => core_llm::Error::Io(error),
        other => core_llm::Error::backend(other),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use candle_core::DType;
    use core_llm::{StarVectorBoundedStream, StarVectorProvider, StarVectorStreamEvent, TextLlm};
    use core_llm_testkit::{starvector_conformance, StarVectorProfile};
    use serde_json::json;
    use std::cell::Cell;

    fn exact_config() -> Value {
        json!({
            "model_type": MODEL_TYPE,
            "starcoder_model_name": STARCODER_BASE_1B,
            "image_encoder_type": "clip",
            "image_size": 224,
            "hidden_size": 2048,
            "vocab_size": 49156,
            "max_position_embeddings": 8192,
            "num_hidden_layers": 24,
            "num_attention_heads": 16,
            "multi_query": true
        })
    }

    #[test]
    fn admits_only_the_exact_1b_snapshot_geometry() {
        assert_eq!(
            StarVectorConfig::from_json(&exact_config())
                .unwrap()
                .vocab_size,
            VOCAB_SIZE
        );
        let mut malformed = exact_config();
        malformed["vocab_size"] = json!(49152);
        assert!(StarVectorConfig::from_json(&malformed).is_err());
        malformed = exact_config();
        malformed["starcoder_model_name"] = json!("bigcode/starcoder2-3b");
        assert!(StarVectorConfig::from_json(&malformed).is_err());
    }

    #[test]
    fn descriptor_reserves_the_exact_image_and_svg_prefill() {
        let text = descriptor();
        assert_eq!(text.capabilities.max_context_tokens, MAX_CONTEXT_TOKENS);
        assert_eq!(text.capabilities.max_new_tokens, 7_933);
        core_llm::validate_advertised_generated_token_cap(
            text.capabilities.max_new_tokens,
            text.capabilities.max_context_tokens,
            IMAGE_TOKEN_COUNT + SVG_PROMPT_TOKEN_COUNT,
        )
        .unwrap();
    }

    #[test]
    fn preprocessing_centres_white_padding_and_keeps_rgb_channel_order() {
        let processor = StarVectorImageProcessor::default();
        let pixels = [10u8, 20, 30, 40, 50, 60]; // 2x1
        let (padded, side) = processor.pad_square_white(&pixels, 2, 1).unwrap();
        assert_eq!(side, 2);
        // torchvision's `[left, top, right, bottom]` calculation puts the odd extra pixel on
        // the bottom/right.  Pin that asymmetry: centering with the extra pixel on the leading
        // edge changes every CLIP patch embedding.
        assert_eq!(&padded[..6], &pixels);
        assert_eq!(&padded[6..], &[255, 255, 255, 255, 255, 255]);
        let tensor = processor.preprocess(&pixels, 2, 1, &Device::Cpu).unwrap();
        assert_eq!(tensor.dims(), &[1, 3, IMAGE_SIZE, IMAGE_SIZE]);
        assert_eq!(tensor.dtype(), DType::F32);
    }

    #[test]
    fn preprocessing_rejects_non_rgb_input_without_silently_reinterpreting_bytes() {
        assert!(StarVectorImageProcessor::default()
            .preprocess(&[0; 4], 1, 1, &Device::Cpu)
            .is_err());
    }

    #[test]
    fn conditioning_join_uses_decoder_embedding_dtype() {
        let vision = Tensor::ones((1, 2, 3), DType::F16, &Device::Cpu).unwrap();
        let text = Tensor::zeros((1, 1, 3), DType::F32, &Device::Cpu).unwrap();

        let joined = concatenate_conditioning_embeddings(&vision, &text).unwrap();

        assert_eq!(joined.dtype(), DType::F32);
        assert_eq!(joined.dims(), &[1, 3, 3]);
        assert_eq!(
            joined.flatten_all().unwrap().to_vec1::<f32>().unwrap(),
            vec![1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0]
        );
    }

    #[test]
    fn token_selection_preserves_singleton_value_for_prefill_and_cached_decode() {
        let device = Device::Cpu;
        let prefill = Tensor::from_vec(vec![0.0f32, 1.0, 9.0, 2.0], (1, 4), &device).unwrap();
        let cached_decode = Tensor::from_vec(vec![8.0f32, 3.0, 2.0, 1.0], (1, 4), &device).unwrap();

        assert_eq!(
            (
                next_token_id(
                    &prefill,
                    &[],
                    &core_llm::Sampling::default(),
                    &mut SplitMix64::new(0)
                )
                .unwrap(),
                next_token_id(
                    &cached_decode,
                    &[],
                    &core_llm::Sampling::default(),
                    &mut SplitMix64::new(0)
                )
                .unwrap()
            ),
            (2, 0)
        );
    }

    #[test]
    fn token_selection_rejects_non_singleton_cardinality() {
        let device = Device::Cpu;
        let batched =
            Tensor::from_vec(vec![0.0f32, 5.0, 1.0, 7.0, 0.0, 2.0], (2, 3), &device).unwrap();

        match next_token_id(
            &batched,
            &[],
            &core_llm::Sampling::default(),
            &mut SplitMix64::new(0),
        ) {
            Err(Error::Msg(message)) => assert_eq!(
                message,
                "starvector token selection requires one nonempty vocabulary row; got [2, 3]"
            ),
            other => panic!("expected a classified model-output error, got {other:?}"),
        }
    }

    #[test]
    fn starvector_sampling_honors_seed_temperature_filters_and_repetition_window() {
        let logits = Tensor::from_vec(vec![1.0f32, 1.1, 1.2, 0.9], (1, 4), &Device::Cpu).unwrap();
        let mut sampling = core_llm::Sampling {
            temperature: 1.0,
            ..Default::default()
        };
        let sequence = |seed, params: &core_llm::Sampling| {
            let mut rng = SplitMix64::new(seed);
            (0..32)
                .map(|_| next_token_id(&logits, &[], params, &mut rng).unwrap())
                .collect::<Vec<_>>()
        };
        assert_eq!(sequence(42, &sampling), sequence(42, &sampling));
        assert_ne!(sequence(42, &sampling), sequence(43, &sampling));
        sampling.temperature = 0.0;
        assert!(sequence(42, &sampling).iter().all(|id| *id == 2));
        sampling.temperature = 1.0;
        sampling.top_k = 1;
        assert!(sequence(42, &sampling).iter().all(|id| *id == 2));
        sampling.top_k = 0;
        sampling.top_p = 0.1;
        assert!(sequence(42, &sampling).iter().all(|id| *id == 2));
        sampling.temperature = 0.0;
        sampling.top_p = 1.0;
        sampling.repetition_penalty = 2.0;
        sampling.repetition_context = 1;
        assert_eq!(
            next_token_id(&logits, &[2], &sampling, &mut SplitMix64::new(0)).unwrap(),
            1
        );
        assert_eq!(
            next_token_id(&logits, &[2, 0], &sampling, &mut SplitMix64::new(0)).unwrap(),
            2
        );
        sampling.repetition_context = 2;
        assert_eq!(
            next_token_id(&logits, &[2, 0], &sampling, &mut SplitMix64::new(0)).unwrap(),
            1
        );
    }

    fn source_tokenizer() -> core_llm::Tokenizer {
        core_llm::Tokenizer::from_json(
            r#"{
                "version": "1.0",
                "added_tokens": [
                    { "id": 0, "content": "<|endoftext|>", "single_word": false,
                      "lstrip": false, "rstrip": false, "normalized": false, "special": true },
                    { "id": 5, "content": "<control>", "single_word": false,
                      "lstrip": false, "rstrip": false, "normalized": false, "special": true },
                    { "id": 6, "content": "<svg-start>", "single_word": false,
                      "lstrip": false, "rstrip": false, "normalized": false, "special": false }
                ],
                "normalizer": null,
                "pre_tokenizer": { "type": "Whitespace" },
                "post_processor": null,
                "decoder": {
                    "type": "Sequence",
                    "decoders": [{ "type": "Fuse" }]
                },
                "model": {
                    "type": "WordLevel",
                    "vocab": {
                        "<|endoftext|>": 0,
                        ">": 1,
                        "</svg>": 2,
                        "prose": 3,
                        "```svg": 4
                    },
                    "unk_token": "prose"
                }
            }"#,
        )
        .unwrap()
    }

    #[test]
    fn continuation_decode_skips_special_controls_but_not_added_source_vocabulary() {
        let tokenizer = source_tokenizer();
        let mut detok = tokenizer.decode_stream(true);

        assert_eq!(
            decode_generated_svg_token(&mut detok, 5).unwrap(),
            DecodedSvgToken::Hidden
        );
        assert_eq!(
            decode_generated_svg_token(&mut detok, 1).unwrap(),
            DecodedSvgToken::Source(">".into())
        );
        assert_eq!(
            decode_generated_svg_token(&mut detok, 2).unwrap(),
            DecodedSvgToken::Source("</svg>".into())
        );
        assert_eq!(
            decode_generated_svg_token(&mut detok, 0).unwrap(),
            DecodedSvgToken::Eos
        );
        let mut added_detok = tokenizer.decode_stream(true);
        assert_eq!(
            decode_generated_svg_token(&mut added_detok, 6).unwrap(),
            DecodedSvgToken::Source("<svg-start>".into()),
            "non-special added vocabulary is model output, not a control to silently trim"
        );
    }

    #[test]
    fn continuation_decode_matches_full_decode_for_a_long_sequence() {
        let tokenizer = source_tokenizer();
        let ids: Vec<u32> = [1, 2].into_iter().cycle().take(4_096).collect();
        let expected = tokenizer.decode(&ids, true).unwrap();
        let mut detok = tokenizer.decode_stream(true);
        let mut actual = String::new();
        for id in ids {
            if let DecodedSvgToken::Source(delta) =
                decode_generated_svg_token(&mut detok, id as i32).unwrap()
            {
                actual.push_str(&delta);
            }
        }
        if let Some(delta) = detok.finish().unwrap() {
            actual.push_str(&delta);
        }
        assert_eq!(actual, expected);
        assert_eq!(actual.len(), 14_336);
    }

    #[test]
    fn continuation_decode_resolves_split_utf8_and_drops_an_incomplete_tail() {
        let dir = tempfile::tempdir().unwrap();
        std::fs::write(
            dir.path().join("vocab.json"),
            r#"{"<|endoftext|>":0,"Ã":1,"©":2}"#,
        )
        .unwrap();
        std::fs::write(dir.path().join("merges.txt"), "#version: 0.2\n").unwrap();
        std::fs::write(
            dir.path().join("tokenizer_config.json"),
            r#"{"added_tokens_decoder":{"0":{"content":"<|endoftext|>","lstrip":false,"normalized":false,"rstrip":false,"single_word":false,"special":true}}}"#,
        )
        .unwrap();
        let tokenizer = core_llm::Tokenizer::from_hf_byte_level_bpe(
            dir.path().join("vocab.json"),
            dir.path().join("merges.txt"),
            dir.path().join("tokenizer_config.json"),
        )
        .unwrap();

        let mut complete = tokenizer.decode_stream(true);
        assert_eq!(
            decode_generated_svg_token(&mut complete, 1).unwrap(),
            DecodedSvgToken::Hidden
        );
        assert_eq!(
            decode_generated_svg_token(&mut complete, 2).unwrap(),
            DecodedSvgToken::Source("é".into())
        );
        assert_eq!(complete.finish().unwrap(), None);

        let mut incomplete = tokenizer.decode_stream(true);
        assert_eq!(
            decode_generated_svg_token(&mut incomplete, 1).unwrap(),
            DecodedSvgToken::Hidden
        );
        assert_eq!(incomplete.finish().unwrap(), None);
    }

    #[test]
    fn static_prompt_and_cumulative_decode_publish_exactly_one_svg_root() {
        let request = StarVectorProfile {
            text: None,
            max_new_tokens: 3,
            ..StarVectorProfile::cheap()
        }
        .request();
        let tokenizer = source_tokenizer();
        let mut detok = tokenizer.decode_stream(true);
        let mut stream = StarVectorBoundedStream::new(&request);
        let mut events = Vec::new();
        assert_eq!(
            seed_svg_prompt(&mut stream, &mut |event| events.push(event)).unwrap(),
            core_llm::StarVectorStreamStatus::Continue
        );

        for (index, id) in [5, 1, 2].into_iter().enumerate() {
            let decoded = decode_generated_svg_token(&mut detok, id).unwrap();
            if matches!(
                push_decoded_svg_token(
                    &mut stream,
                    decoded,
                    Default::default(),
                    index as u32 + 1,
                    &mut |event| events.push(event),
                )
                .unwrap(),
                core_llm::StarVectorStreamStatus::Stop(_)
            ) {
                break;
            }
        }

        let output = stream.output().unwrap();
        assert_eq!(output.svg.as_deref(), Some("<svg></svg>"));
        assert_eq!(output.generated_tokens, 3);
        assert_eq!(output.generated_bytes, "<svg></svg>".len());
        assert_eq!(
            events,
            [
                StarVectorStreamEvent::Source {
                    text: SVG_PROMPT.into(),
                    index: 0
                },
                StarVectorStreamEvent::Progress {
                    generated_tokens: 1
                },
                StarVectorStreamEvent::Source {
                    text: ">".into(),
                    index: 2
                },
                StarVectorStreamEvent::Progress {
                    generated_tokens: 2
                },
                StarVectorStreamEvent::Source {
                    text: "</svg>".into(),
                    index: 3
                },
                StarVectorStreamEvent::Progress {
                    generated_tokens: 3
                },
            ]
        );
    }

    #[test]
    fn source_boundary_still_rejects_hostile_or_non_svg_leading_content() {
        let request = StarVectorProfile {
            text: None,
            ..StarVectorProfile::cheap()
        }
        .request();
        for invalid in [
            "prose<svg></svg>",
            "```svg\n<svg></svg>\n```",
            "<script/><svg></svg>",
            "<svg-start><svg></svg>",
            "<svg></svg><svg></svg>",
        ] {
            let mut stream = StarVectorBoundedStream::new(&request);
            assert!(
                stream.push(invalid, Default::default()).is_err(),
                "accepted hostile or non-SVG leading content: {invalid:?}"
            );
        }
    }

    struct FixtureProvider {
        text: core_llm::TextLlmDescriptor,
        svg: core_llm::StarVectorDescriptor,
    }

    impl FixtureProvider {
        fn new() -> Self {
            Self {
                text: descriptor(),
                svg: svg_descriptor(),
            }
        }
    }

    impl TextLlm for FixtureProvider {
        fn descriptor(&self) -> &core_llm::TextLlmDescriptor {
            &self.text
        }
        fn validate(&self, request: &core_llm::TextLlmRequest) -> core_llm::Result<()> {
            self.text
                .capabilities
                .validate_request(&self.text.id, request)?;
            if request.has_image() {
                Ok(())
            } else {
                Err(core_llm::Error::InvalidRequest(
                    "fixture requires image".into(),
                ))
            }
        }
        fn generate(
            &self,
            _request: &core_llm::TextLlmRequest,
            _events: &mut dyn FnMut(core_llm::StreamEvent),
        ) -> core_llm::Result<core_llm::TextLlmOutput> {
            unreachable!("shared StarVector suite drives generate_svg")
        }
    }

    impl StarVectorProvider for FixtureProvider {
        fn starvector_descriptor(&self) -> &core_llm::StarVectorDescriptor {
            &self.svg
        }
        fn generate_svg(
            &self,
            request: &core_llm::StarVectorRequest,
            events: &mut dyn FnMut(StarVectorStreamEvent),
        ) -> core_llm::Result<core_llm::StarVectorOutput> {
            self.validate_svg(request)?;
            if request.text_request.cancel.is_cancelled() {
                return Err(core_llm::Error::Canceled);
            }
            let mut stream = StarVectorBoundedStream::new(request);
            for (index, fragment) in core_llm_testkit::deterministic_svg_fixture()
                .fragments
                .iter()
                .enumerate()
            {
                let status = stream.push(fragment, std::time::Duration::ZERO)?;
                events(StarVectorStreamEvent::Source {
                    text: (*fragment).into(),
                    index: index as u32,
                });
                events(StarVectorStreamEvent::Progress {
                    generated_tokens: stream.generated_tokens(),
                });
                if matches!(status, core_llm::StarVectorStreamStatus::Stop(_)) {
                    break;
                }
            }
            let output = stream.output()?;
            events(StarVectorStreamEvent::Done {
                finish_reason: output.finish_reason,
                generated_tokens: output.generated_tokens,
                generated_bytes: output.generated_bytes,
            });
            Ok(output)
        }
    }

    #[test]
    fn native_provider_metadata_and_bounded_svg_contract_pass_shared_conformance() {
        starvector_conformance(
            || Box::new(FixtureProvider::new()),
            &StarVectorProfile::cheap(),
        );
    }

    fn terminal_profile() -> StarVectorProfile {
        StarVectorProfile {
            image: Some(core_llm::ImageRef::new(2, 2, vec![0x80; 12]).unwrap()),
            text: None,
            max_new_tokens: 4_000,
            max_svg_bytes: 2 * 1024 * 1024,
            max_wall_time: std::time::Duration::from_secs(120),
            seed: 7,
        }
    }

    #[test]
    fn terminal_profile_fits_advertised_svg_limit() {
        FixtureProvider::new()
            .validate_svg(&terminal_profile().request())
            .unwrap();
    }

    /// Terminal real-weight hook. It deliberately opens only the explicit exact snapshot supplied
    /// by SC-22261; ordinary CPU checks neither download weights nor invoke CUDA.
    #[test]
    #[ignore = "sc-22261 terminal real-weight StarVector-1B CUDA campaign only"]
    fn real_weight_provider_satisfies_shared_starvector_conformance() {
        let snapshot = std::env::var("STARVECTOR_1B_SNAPSHOT")
            .expect("sc-22261 must set STARVECTOR_1B_SNAPSHOT to the local exact snapshot");
        let spec = core_llm::LoadSpec::dense(snapshot);
        let profile = terminal_profile();
        starvector_conformance(
            || Box::new(CandleStarVectorProvider::load(&spec).unwrap()),
            &profile,
        );
    }

    #[test]
    fn registry_exposes_only_the_exact_starvector_snapshot_probe() {
        let root = tempfile::tempdir().unwrap();
        std::fs::write(root.path().join("config.json"), exact_config().to_string()).unwrap();
        assert!(can_load_path(root.path()));
        let mut wrong = exact_config();
        wrong["image_size"] = json!(336);
        std::fs::write(root.path().join("config.json"), wrong.to_string()).unwrap();
        assert!(!can_load_path(root.path()));
        assert!(crate::text_registry().unwrap().find(PROVIDER_ID).is_some());
    }

    #[test]
    fn model_probe_rejects_file_sources_before_reading_them() {
        let file = tempfile::NamedTempFile::new().unwrap();
        std::fs::write(file.path(), exact_config().to_string()).unwrap();
        assert!(
            read_config(file.path()).is_ok(),
            "direct config parsing remains supported"
        );

        let read_attempted = Cell::new(false);
        let result = can_load_path_with(file.path(), |_| {
            read_attempted.set(true);
            true
        });

        assert!(!result);
        assert!(!read_attempted.get(), "file payload must not be read");
        assert!(!can_load_path(file.path()));
    }

    /// sc-24134: the provider selects its device once, at load, and a request's pixels go to
    /// that device. A `select_device()` per request would build a second device — on the own
    /// stream a second CUDA stream (and cuBLAS / cuRAND handles) that candle's per-op check,
    /// which compares the GPU ordinal only, cannot tell from the model's (see the CUDA test).
    #[test]
    fn the_provider_selects_its_device_once_at_load() {
        let source = include_str!("starvector.rs");
        let production = &source[..source.find("mod tests {").expect("the test module")];
        // `CandleStarVectorProvider::load` is followed by the free `descriptor()` function.
        let load = production
            .find("pub fn load(")
            .expect("the provider's load");
        let load_end = production
            .find("pub fn descriptor()")
            .expect("descriptor()");
        let calls: Vec<usize> = production
            .match_indices("select_device(")
            .map(|(at, _)| at)
            .collect();
        assert_eq!(calls.len(), 1, "one device selection: {calls:?}");
        assert!(
            (load..load_end).contains(&calls[0]),
            "the device is selected in `load`, not per request"
        );
    }

    /// The load-time device and a request's pixels are one device on one stream; a second
    /// `select_device()` (what a per-request selection did) shares the GPU ordinal — so candle's
    /// per-op check accepts mixing them — but is a different device, and on the own stream a
    /// different CUDA stream.
    #[cfg(feature = "cuda")]
    #[test]
    fn request_pixels_share_the_load_time_device_and_stream() {
        use candle_core::Device;
        // The graph runner on at load: the own stream, where a second device is a second stream.
        let _guard = crate::decode::graph::cuda_graphs_policy_guard(Some(true));
        let Ok(model_device @ Device::Cuda(_)) = crate::device::select_device() else {
            eprintln!("skipping: no CUDA device");
            return;
        };
        let stream = |d: &Device| match d {
            Device::Cuda(c) => (
                c.cuda_stream().cu_stream() as usize,
                c.cuda_stream().context().ordinal(),
            ),
            _ => unreachable!(),
        };
        let pixels = StarVectorImageProcessor::default()
            .preprocess(&[10u8, 20, 30, 40, 50, 60], 2, 1, &model_device)
            .unwrap();
        assert!(pixels.device().same_device(&model_device));
        assert_eq!(stream(pixels.device()), stream(&model_device));

        // Same GPU ordinal (all candle's per-op check compares), yet another device and stream.
        let second = crate::device::select_device().unwrap();
        assert!(!second.same_device(&model_device));
        if !cfg!(feature = "flash-attn") {
            assert_ne!(
                stream(&second).0,
                stream(&model_device).0,
                "a second own stream"
            );
        }
    }
}
