//! Native Candle provider for the exact `starvector/starvector-8b-im2svg` snapshot.
//!
//! The snapshot is treated as inert local data: config, tokenizer, and safetensors only. No
//! snapshot code, Python, Transformers, network routing, or model download is involved.

use std::cell::{Cell, RefCell};
use std::path::Path;
use std::time::{Duration, Instant};

use candle_core::{DType, Tensor};
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
use crate::primitives::{input_ids, KvCache, Weights};

pub const PROVIDER_ID: &str = "candle-starvector-8b";
const SNAPSHOT_REPOSITORY: &str = "starvector/starvector-8b-im2svg";
const SNAPSHOT_REVISION: &str = "518beea8dcb5f7a37c5911e92d1d62a76beee7f9";
const SVG_PROMPT: &str = "<svg";
const EOS_TOKEN_ID: i32 = 0;
const IMAGE_SIZE: usize = 384;
const IMAGE_TOKENS: usize = 576;
const VISION_HIDDEN: usize = 1024;
const DECODER_HIDDEN: usize = 4608;

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

    fn prefill(
        &self,
        image: &core_llm::ImageRef,
        prompt: &[i32],
    ) -> Result<(Tensor, Box<dyn KvCache>)> {
        let pixels = preprocess_image(image, self.decoder.device())?;
        let vision = self.vision.forward(&pixels)?.last_hidden_state;
        let vision = self.adapter.forward(&vision, self.decoder.dtype())?;
        let text = self
            .decoder
            .embed(&input_ids(prompt, self.decoder.device())?)?;
        let embeds = Tensor::cat(&[&vision, &text], 1)?;
        let mut cache: Box<dyn KvCache> = Box::new(self.decoder.cache());
        let logits = self
            .decoder
            .logits_from_embeds(&embeds, cache.as_mut(), 0)?;
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
        let hidden = silu(&linear(&features, &self.fc_weight, Some(&self.fc_bias))?)?;
        let hidden = linear(&hidden, &self.proj_weight, Some(&self.proj_bias))?;
        let width = IMAGE_TOKENS * DECODER_HIDDEN;
        let flat = hidden.reshape((batch, width))?;
        let weight = self.norm_weight.reshape(width)?;
        let bias = self.norm_bias.reshape(width)?;
        Ok(layer_norm(&flat, &weight, &bias, 1e-5)?
            .reshape((batch, IMAGE_TOKENS, DECODER_HIDDEN))?
            .to_dtype(dtype)?)
    }
}

pub struct CandleStarVector8bProvider {
    descriptor: TextLlmDescriptor,
    starvector: StarVectorDescriptor,
    model: StarVector8bModel,
    tokenizer: Tokenizer,
    memory: StarVectorCandleMemory,
}

impl CandleStarVector8bProvider {
    pub fn load(spec: &LoadSpec) -> CoreResult<Self> {
        if spec.quantize.is_some() {
            return Err(CoreError::Unsupported(
                "StarVector-8B Candle does not support load-time quantization".into(),
            ));
        }
        let dir = Path::new(&spec.source);
        validate_snapshot(dir).map_err(to_core)?;
        let device = crate::device::select_device().map_err(to_core)?;
        let weights = Weights::from_dir(dir, &device).map_err(to_core)?;
        let memory = StarVectorCandleMemory {
            loaded_tensor_count: weights.len(),
            device: format!("{device:?}"),
        };
        Ok(Self {
            descriptor: descriptor(),
            starvector: starvector_descriptor(),
            model: StarVector8bModel::from_weights(&weights).map_err(to_core)?,
            tokenizer: Tokenizer::from_hf_byte_level_bpe(
                dir.join("vocab.json"),
                dir.join("merges.txt"),
                dir.join("tokenizer_config.json"),
            )?,
            memory,
        })
    }

    pub fn memory_report(&self) -> StarVectorCandleMemory {
        self.memory.clone()
    }
    /// Consume the loaded provider; tensor ownership is released without touching shared allocator policy.
    pub fn unload(self) {}

    fn generate_svg_inner(
        &self,
        request: &StarVectorRequest,
        on_event: &mut dyn FnMut(StarVectorStreamEvent),
    ) -> CoreResult<StarVectorOutput> {
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
            StarVectorStreamStatus::Stop(_) => return emit_done(stream.output()?, on_event),
        }
        let began = Instant::now();
        let (first, mut cache) = self.model.prefill(image, &prompt).map_err(to_core)?;
        let config = GenerationConfig {
            max_new_tokens: request.text_request.max_new_tokens as usize,
            sampling: sampling(&request.text_request.sampling),
            seed: request.text_request.seed,
            stop_tokens: vec![EOS_TOKEN_ID],
        };
        let tokens = RefCell::new(Vec::<u32>::new());
        let mut detok = IncrementalDetok::new();
        let stopped = Cell::new(false);
        let failure = RefCell::new(None);
        let tokenizer = &self.tokenizer;
        let mut decode = |event: DecodeEvent| {
            if let DecodeEvent::Token { id, step } = event {
                if stopped.get() {
                    return;
                }
                tokens.borrow_mut().push(id as u32);
                let Ok(text) = tokenizer.decode(&tokens.borrow(), true) else {
                    return;
                };
                let Some(delta) = detok.push(&text) else {
                    return;
                };
                match stream.push(delta, began.elapsed()) {
                    Ok(status @ StarVectorStreamStatus::Continue)
                    | Ok(
                        status @ StarVectorStreamStatus::Stop(StarVectorFinishReason::CompleteRoot),
                    ) => {
                        on_event(StarVectorStreamEvent::Source {
                            text: delta.to_owned(),
                            index: step as u32 + 1,
                        });
                        stopped.set(!matches!(status, StarVectorStreamStatus::Continue));
                    }
                    Ok(StarVectorStreamStatus::Stop(_)) => stopped.set(true),
                    Err(error) => *failure.borrow_mut() = Some(error),
                }
                if failure.borrow().is_some() {
                    stopped.set(true);
                }
            }
        };
        let generated = generate_from_prefill(
            &self.model.decoder,
            cache.as_mut(),
            first,
            prompt,
            &config,
            &request.text_request.cancel,
            &mut decode,
            None,
        )
        .map_err(to_core)?;
        if let Some(error) = failure.into_inner() {
            return Err(error);
        }
        if stream.output().is_err() {
            match generated.finish_reason {
                DecodeFinish::StopToken => {
                    stream.finish_eos()?;
                }
                DecodeFinish::MaxTokens | DecodeFinish::Cancelled => {
                    let _ = stream.push("", began.elapsed())?;
                }
            }
        }
        emit_done(stream.output()?, on_event)
    }
}

fn emit_done(
    output: StarVectorOutput,
    on_event: &mut dyn FnMut(StarVectorStreamEvent),
) -> CoreResult<StarVectorOutput> {
    on_event(StarVectorStreamEvent::Done {
        finish_reason: output.finish_reason,
        generated_tokens: output.generated_tokens,
        generated_bytes: output.generated_bytes,
    });
    Ok(output)
}

impl TextLlm for CandleStarVector8bProvider {
    fn descriptor(&self) -> &TextLlmDescriptor {
        &self.descriptor
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
        let svg_request =
            StarVectorRequest::new(request.clone(), 2 * 1024 * 1024, Duration::from_secs(120));
        let prompt_tokens = self.tokenizer.encode(SVG_PROMPT, false)?.len() as u32;
        let output = self.generate_svg_inner(&svg_request, &mut |event| match event {
            StarVectorStreamEvent::Source { text, index } => on_event(StreamEvent::Token {
                id: index,
                text,
                index: index as usize,
                channel: Channel::Content,
            }),
            StarVectorStreamEvent::Done {
                finish_reason,
                generated_tokens,
                ..
            } => on_event(StreamEvent::Done {
                finish_reason: map_finish(finish_reason),
                usage: Usage {
                    prompt_tokens: IMAGE_TOKENS as u32 + prompt_tokens,
                    generated_tokens,
                },
            }),
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
    }
}

pub fn descriptor() -> TextLlmDescriptor {
    TextLlmDescriptor {
        id: PROVIDER_ID.into(),
        family: "starvector".into(),
        backend: "candle".into(),
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
    std::fs::read_to_string(Path::new(&spec.source).join("config.json"))
        .ok()
        .and_then(|text| serde_json::from_str::<Value>(&text).ok())
        .is_some_and(|config| exact_config(&config))
}

fn validate_snapshot(dir: &Path) -> Result<()> {
    let config: Value = serde_json::from_str(&std::fs::read_to_string(dir.join("config.json"))?)
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
    use core_llm_testkit::{check_starvector_bounded_fixture, StarVectorProfile};
    use serde_json::json;
    fn config() -> Value {
        json!({"model_type":"starvector","starcoder_model_name":"bigcode/starcoder2-7b","image_encoder_type":"siglip_384","adapter_norm":"layer_norm","image_size":384,"hidden_size":4608,"num_attention_heads":36,"num_hidden_layers":32,"num_kv_heads":4,"vocab_size":49152})
    }
    #[test]
    fn descriptor_matches_mlx_8b_contract() {
        let d = starvector_descriptor();
        assert_eq!(descriptor().id, PROVIDER_ID);
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
    }
}
