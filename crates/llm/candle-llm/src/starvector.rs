//! Exact, tensor-free StarVector-1B snapshot admission and image preparation.
//!
//! StarVector checkpoints use a custom Transformers wrapper.  Candle must never execute that
//! wrapper: this module accepts only the published 1B image-to-SVG shape and records the native
//! preprocessing and weight-tree facts needed by the Candle model implementation.

use std::path::Path;
use std::sync::Mutex;

use candle_core::{Device, Tensor};
use serde_json::Value;

use crate::error::{Error, Result};
use crate::image::resize_bicubic_u8;
use crate::models::StarVectorModel;
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

/// Read and validate `config.json` only.  This deliberately never opens a safetensors shard.
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

/// Weightless registry probe for the exact 1B snapshot.
pub fn can_load_path(dir: impl AsRef<Path>) -> bool {
    read_config(dir).is_ok()
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
        let prompt = tokenizer
            .encode("<svg", false)?
            .into_iter()
            .map(|id| id as i32)
            .collect();
        Ok(Self {
            descriptor: descriptor(),
            svg: svg_descriptor(),
            processor: StarVectorImageProcessor::default(),
            tokenizer,
            prompt,
            model: Mutex::new(StarVectorModel::from_weights(&weights).map_err(to_core)?),
        })
    }
}
pub fn descriptor() -> core_llm::TextLlmDescriptor {
    core_llm::TextLlmDescriptor {
        id: PROVIDER_ID.into(),
        family: "starvector-1b".into(),
        backend: "candle".into(),
        capabilities: core_llm::TextLlmCapabilities {
            max_context_tokens: 8192,
            max_new_tokens: 4096,
            supports_system_prompt: false,
            supports_vision: true,
            supports_video: false,
            supports_audio: false,
            supports_thinking: false,
            supports_tools: false,
            supported_constraints: vec![],
        },
    }
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
        max_svg_bytes: 1_000_000,
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
            prompt_tokens: (257 + self.prompt.len()) as u32,
            generated_tokens: out.generated_tokens,
        };
        events(core_llm::StreamEvent::Done {
            finish_reason: finish,
            usage,
        });
        Ok(core_llm::TextLlmOutput {
            text: out.svg.unwrap_or_default(),
            thinking: None,
            tool_calls: vec![],
            usage,
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
                &crate::device::select_device().map_err(to_core)?,
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
            let initial = Tensor::cat(&[&vision, &text], 1)
                .map_err(|e| core_llm::Error::Msg(e.to_string()))?;
            let mut logits = model.decoder.forward_embeds(&initial, 0).map_err(to_core)?;
            let mut stream = core_llm::StarVectorBoundedStream::new(req);
            for index in 0..req.text_request.max_new_tokens {
                if req.text_request.cancel.is_cancelled() {
                    break;
                }
                let id = logits
                    .argmax(candle_core::D::Minus1)
                    .map_err(|e| core_llm::Error::Msg(e.to_string()))?
                    .to_scalar::<u32>()
                    .map_err(|e| core_llm::Error::Msg(e.to_string()))?
                    as i32;
                let decoded = self.tokenizer.decode(&[id as u32], true)?;
                if !decoded.is_empty() {
                    match stream.push(&decoded, started.elapsed())? {
                        core_llm::StarVectorStreamStatus::Continue => {
                            events(core_llm::StarVectorStreamEvent::Source {
                                text: decoded,
                                index,
                            })
                        }
                        core_llm::StarVectorStreamStatus::Stop(_) => break,
                    }
                }
                let next = input_ids(&[id], pixels.device())
                    .map_err(|e| core_llm::Error::Msg(e.to_string()))?;
                let embed = model.decoder.embeddings(&next).map_err(to_core)?;
                logits = model
                    .decoder
                    .forward_embeds(&embed, 257 + self.prompt.len() + index as usize)
                    .map_err(to_core)?;
            }
            if stream.output().is_err() {
                let _ = stream.finish_eos();
            }
            stream.output()
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
}
