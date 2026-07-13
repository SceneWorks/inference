//! The `core-llm` provider: a generic Llama model exposed through the backend-neutral contract.
//!
//! This is the candle-llm half of story 7237 — it implements [`core_llm::TextLlm`] by wrapping the
//! [`CausalLm`] decoder, a [`core_llm::Tokenizer`], and a chat template, driving the internal
//! streaming decode loop and translating its token events into contract [`StreamEvent`]s (with
//! incremental detokenization). It registers into [`core_llm::registry`] under the id `candle-llama`.
//! Passing the `core-llm-testkit` conformance suite as a *second, independent* backend is what
//! de-provisionalizes the contract.

use std::cell::OnceCell;
use std::path::Path;

use candle_core::{Device, Tensor};
use core_llm::{
    Channel, ChatTemplate, Constraint, ConstraintDecodeTable, Content, Error as CoreError,
    FinishReason as CoreFinish, ImageRef, JinjaChatTemplate, JsonConstraint, Llama3Template,
    LoadSpec, Message, Quantize, RenderOptions, Result as CoreResult, Sampling,
    StreamEvent as CoreEvent, TextLlm, TextLlmCapabilities, TextLlmDescriptor, TextLlmOutput,
    TextLlmRequest, ThinkingSegmenter, Tokenizer, ToolCallSegmenter, Usage, VideoRef,
};

use crate::config::{Architecture, ModelConfig};
use crate::decode::{
    generate_from_prefill, generate_with, ConstraintMask, Decode, FinishReason, GenerationConfig,
    StreamEvent,
};
use crate::device::select_device;
use crate::gguf::GgufCheckpoint;
use crate::image::Qwen35ImageProcessor;
use crate::models::{
    CausalLm, Qwen35Config, Qwen35Model, Qwen35VisionConfig, Qwen35VisionModel, VlmDecode,
};
use crate::primitives::nn::input_ids;
use crate::primitives::projection::QuantSpec;
use crate::primitives::sampler::SamplingParams;
use crate::primitives::{KvCache, Weights};

/// The registry id of this provider.
pub const PROVIDER_ID: &str = "candle-llama";

/// The loaded decoder behind the provider: the generic Llama-family [`CausalLm`] or the Qwen3.6
/// hybrid [`Qwen35Model`] (DeltaNet linear attention + gated full attention), both driven through the
/// shared [`Decode`] loop. Dispatched by [`Architecture`] at load.
enum Decoder {
    Causal(CausalLm),
    Qwen35(Qwen35Model),
}

impl Decode for Decoder {
    fn make_cache(&self) -> Box<dyn KvCache> {
        match self {
            Decoder::Causal(m) => m.make_cache(),
            Decoder::Qwen35(m) => m.make_cache(),
        }
    }

    fn device(&self) -> &Device {
        match self {
            Decoder::Causal(m) => m.device(),
            Decoder::Qwen35(m) => m.device(),
        }
    }

    fn step(
        &self,
        input_ids: &Tensor,
        cache: &mut dyn KvCache,
        offset: i32,
    ) -> crate::error::Result<Tensor> {
        match self {
            Decoder::Causal(m) => m.step(input_ids, cache, offset),
            Decoder::Qwen35(m) => m.step(input_ids, cache, offset),
        }
    }
}

impl Decoder {
    fn is_quantized(&self) -> bool {
        match self {
            Decoder::Causal(m) => m.is_quantized(),
            Decoder::Qwen35(m) => m.is_quantized(),
        }
    }

    /// The decoder as the backend-neutral multimodal seam. Both backbones implement [`VlmDecode`]
    /// (the Qwen3.6 hybrid and the generic Qwen3-VL causal decoder), so the provider drives the
    /// image prefill + decode through one trait object rather than forking on the concrete type.
    fn as_vlm(&self) -> &dyn VlmDecode {
        match self {
            Decoder::Causal(m) => m,
            Decoder::Qwen35(m) => m,
        }
    }
}

/// The Qwen-VL vision side of the provider: the ViT tower, the image preprocessor, the multimodal
/// token id + merge size needed to expand placeholders and assign M-RoPE positions, and the device
/// the encoder runs on. Present when the loaded `qwen3_5` (Qwen3.6) or `qwen3_vl` (Qwen3-VL)
/// checkpoint carries `model.visual.*`. The two share the identical Qwen3-VL ViT tower
/// ([`Qwen35VisionModel`]); only the decoder prefill differs (hybrid vs generic-causal).
struct Qwen35Vision {
    tower: Qwen35VisionModel,
    processor: Qwen35ImageProcessor,
    image_token_id: i32,
    /// The `<|video_pad|>` placeholder token id (151656 for Qwen3-VL) — the per-frame video
    /// placeholder the processor expands to `frame_seqlen` copies. Only reachable when the loaded
    /// config carried a `video_token_id` (which also flips `supports_video` on).
    video_token_id: i32,
    spatial_merge_size: i32,
    device: Device,
}

impl Qwen35Vision {
    /// Encode one image to its merged patch rows `[n_tokens, hidden]` (the merger output is already
    /// the language hidden size — no separate projector), the per-tap **DeepStack** feature sets
    /// (each `[n_tokens, hidden]`, one per `deepstack_visual_indexes` tap — empty for a Qwen3.6
    /// tower), plus the image's `grid_thw` (`[1, h, w]` in patch units). `n_tokens =
    /// (grid_h/merge)·(grid_w/merge)` is the placeholder expansion count.
    fn encode(&self, img: &ImageRef) -> CoreResult<(Tensor, Vec<Tensor>, [i32; 3])> {
        let (pixels, grid) = self
            .processor
            .preprocess(
                &img.pixels,
                img.width as usize,
                img.height as usize,
                &self.device,
            )
            .map_err(to_core)?;
        let out = self.tower.forward_with_deepstack(&pixels, &grid).map_err(to_core)?;
        Ok((out.pooler_output, out.deepstack_features, grid[0]))
    }

    /// Encode one **video** (sampled frames) to its merged patch rows `[grid_t·n_per_frame, hidden]`,
    /// the per-tap DeepStack features, and the `video_grid_thw` (`[grid_t, h, w]`). The ViT tower is
    /// modality-agnostic — it processes the `grid_t` temporal patches as a block-diagonal-masked frame
    /// sequence exactly like multiple images — so this reuses `forward_with_deepstack`. The per-frame
    /// timestamp tokens are rendered separately (Text–Timestamp Alignment); here we only produce the
    /// visual features and the grid.
    fn encode_video(&self, video: &VideoRef) -> CoreResult<(Tensor, Vec<Tensor>, [i32; 3])> {
        let frames: Vec<(&[u8], usize, usize)> = video
            .frames
            .iter()
            .map(|f| (f.pixels.as_slice(), f.width as usize, f.height as usize))
            .collect();
        let (pixels, grid) = self
            .processor
            .preprocess_video(&frames, &self.device)
            .map_err(to_core)?;
        let out = self.tower.forward_with_deepstack(&pixels, &[grid]).map_err(to_core)?;
        Ok((out.pooler_output, out.deepstack_features, grid))
    }
}

/// The prepared multimodal prefill: the image-token-expanded prompt ids, the decoder input embeds
/// with image features spliced in, the interleaved M-RoPE position rows + delta, the per-position
/// visual mask (`true` at image-token positions), and the per-tap DeepStack feature sets fused into
/// the first decoder layers.
struct MultimodalPrefill {
    expanded_ids: Vec<i32>,
    embeds: Tensor,
    positions: (Vec<i32>, Vec<i32>, Vec<i32>, i32),
    visual_pos_mask: Vec<bool>,
    deepstack: Vec<Tensor>,
}

/// A [`Decode`] wrapper that shifts the RoPE offset by a constant `delta` for the post-prompt
/// continuation of a multimodal decode. Image tokens compress the position cursor, so the text
/// positions that follow the prompt are `cache_len + mrope_delta`, not `cache_len`; the new tokens
/// are text, so a single shifted 1-D position is the correct (interleaved-)M-RoPE position. Drives
/// either backbone through [`VlmDecode`]'s [`Decode`] supertrait — each decoder downcasts its own
/// cache inside `step`, so no concrete-type fork is needed here.
struct Shifted<'a> {
    model: &'a dyn VlmDecode,
    delta: i32,
}

impl Decode for Shifted<'_> {
    fn make_cache(&self) -> Box<dyn KvCache> {
        self.model.make_cache()
    }

    fn device(&self) -> &Device {
        self.model.device()
    }

    fn step(
        &self,
        ids: &Tensor,
        cache: &mut dyn KvCache,
        offset: i32,
    ) -> crate::error::Result<Tensor> {
        self.model.step(ids, cache, offset + self.delta)
    }
}

/// Collect the image blocks of a conversation, in order.
fn collect_images(messages: &[Message]) -> Vec<&ImageRef> {
    messages
        .iter()
        .flat_map(|m| {
            m.content.iter().filter_map(|c| match c {
                Content::Image(img) => Some(img),
                Content::Text(_) | Content::Video(_) => None,
            })
        })
        .collect()
}

/// Collect the video blocks of a conversation, in order.
fn collect_videos(messages: &[Message]) -> Vec<&VideoRef> {
    messages
        .iter()
        .flat_map(|m| {
            m.content.iter().filter_map(|c| match c {
                Content::Video(v) => Some(v),
                Content::Text(_) | Content::Image(_) => None,
            })
        })
        .collect()
}

/// Compute the Qwen3-VL **merged per-frame timestamps** for a sampled video, mirroring
/// `Qwen3VLProcessor._calculate_timestamps`. The vision encoder folds `temporal_patch_size` frames
/// into one temporal patch, so the per-sample timestamps are padded up to a multiple of
/// `temporal_patch_size` (repeating the last) and then **averaged within each temporal patch**,
/// yielding one timestamp per emitted vision frame (`grid_t = padded / temporal_patch_size`). These
/// are the `<{t:.1f} seconds>` values the Text–Timestamp-Alignment placeholder uses.
fn merged_frame_timestamps(timestamps: &[f32], temporal_patch_size: usize) -> Vec<f32> {
    let tps = temporal_patch_size.max(1);
    let mut ts: Vec<f32> = timestamps.to_vec();
    if ts.is_empty() {
        return ts;
    }
    while !ts.len().is_multiple_of(tps) {
        ts.push(*ts.last().unwrap());
    }
    (0..ts.len())
        .step_by(tps)
        .map(|i| (ts[i] + ts[i + tps - 1]) / 2.0)
        .collect()
}

/// The Text–Timestamp-Alignment placeholder text for one video: per merged frame, a
/// `<{t:.1f} seconds>` timestamp tag followed by `<|vision_start|><|video_pad|><|vision_end|>`
/// (exactly `Qwen3VLProcessor.replace_video_token`). The single `<|video_pad|>` per frame is expanded
/// to `frame_seqlen` copies after tokenizing.
fn video_placeholder_text(video: &VideoRef, temporal_patch_size: usize) -> String {
    let merged = merged_frame_timestamps(&video.timestamps, temporal_patch_size);
    let mut out = String::new();
    for t in merged {
        out.push_str(&format!("<{t:.1} seconds>"));
        out.push_str("<|vision_start|><|video_pad|><|vision_end|>");
    }
    out
}

/// Replace each image/video block with its Qwen-VL placeholder text so the (text-only) chat template
/// renders the vision framing verbatim. Images become `<|vision_start|><|image_pad|><|vision_end|>`
/// (one `image_pad`, expanded to the per-image patch count after tokenizing); videos become the
/// per-frame Text–Timestamp-Alignment string
/// `<{t} seconds><|vision_start|><|video_pad|><|vision_end|>` (one `video_pad` per frame, each
/// expanded to `frame_seqlen` after tokenizing). Keeps the core-llm template contract image/video-free.
fn substitute_vision_placeholders(messages: &[Message], temporal_patch_size: usize) -> Vec<Message> {
    const IMAGE_PLACEHOLDER: &str = "<|vision_start|><|image_pad|><|vision_end|>";
    messages
        .iter()
        .map(|m| Message {
            role: m.role,
            content: m
                .content
                .iter()
                .map(|c| match c {
                    Content::Image(_) => Content::text(IMAGE_PLACEHOLDER),
                    Content::Video(v) => Content::text(video_placeholder_text(v, temporal_patch_size)),
                    Content::Text(t) => Content::Text(t.clone()),
                })
                .collect(),
            thinking: m.thinking.clone(),
            tool_calls: m.tool_calls.clone(),
        })
        .collect()
}

/// Expand each `token` placeholder in `ids` into `counts[i]` copies (the i-th occurrence's merged
/// patch / per-frame count), in order. Errors if the placeholder count and the supplied count
/// disagree. Called once per visual token id (`image_token_id`, then `video_token_id`); each call
/// only touches its own token, so order across the two is preserved.
fn expand_vision_placeholders(
    ids: &[i32],
    token: i32,
    counts: &[usize],
) -> crate::error::Result<Vec<i32>> {
    use crate::error::Error;
    let mut out = Vec::with_capacity(ids.len());
    let mut ci = 0usize;
    for &id in ids {
        if id == token {
            let n = *counts.get(ci).ok_or_else(|| {
                Error::Msg(format!(
                    "qwen-vl vision: {} placeholders for token {token} but only {} counts supplied",
                    ci + 1,
                    counts.len()
                ))
            })?;
            ci += 1;
            out.extend(std::iter::repeat_n(token, n));
        } else {
            out.push(id);
        }
    }
    if ci != counts.len() {
        return Err(Error::Msg(format!(
            "qwen-vl vision: {ci} placeholders for token {token} rendered but {} counts supplied",
            counts.len()
        )));
    }
    Ok(out)
}

/// A generic Llama provider implementing [`core_llm::TextLlm`].
pub struct LlamaProvider {
    descriptor: TextLlmDescriptor,
    model: Decoder,
    tokenizer: Tokenizer,
    template: Box<dyn ChatTemplate>,
    stop_tokens: Vec<i32>,
    /// Cached per-vocab decode table for constrained decoding — built once (it decodes the whole
    /// vocabulary) on the first JSON-constrained request, then reused.
    constraint_table: OnceCell<ConstraintDecodeTable>,
    /// The Qwen3.6 vision tower + preprocessor, present iff this is a `qwen3_5` checkpoint carrying
    /// `model.visual.*`. Drives the image path in [`LlamaProvider::generate`].
    vision: Option<Qwen35Vision>,
}

impl LlamaProvider {
    /// Load a provider from `spec.source`: either a `*.gguf` file (loaded directly via Candle's
    /// native GGUF reader, story 7254) or an HF snapshot directory (config.json + tokenizer.json +
    /// shards). Either way the decoder architecture is dispatched (Llama / Mistral / Qwen3) and the
    /// projections are optionally quantized on load per `spec.quantize`.
    pub fn load(spec: &LoadSpec) -> CoreResult<Self> {
        let requested = spec.quantize.map(|q| match q {
            Quantize::Q4 => QuantSpec::q4(),
            Quantize::Q8 => QuantSpec::q8(),
        });
        let device = select_device().map_err(to_core)?;
        if crate::gguf::is_gguf_path(&spec.source) {
            Self::load_gguf(Path::new(&spec.source), &device, requested)
        } else {
            Self::load_dir(Path::new(&spec.source), &device, requested)
        }
    }

    /// Load from an HF snapshot directory (config.json + tokenizer.json + safetensors shards).
    ///
    /// `requested` is an explicit load-time quantization (`spec.quantize`); when it is `None` the
    /// snapshot's own persisted `quantization` block (written by the [`prepare`](crate::prepare)
    /// writer) is honored, so a `LoadSpec::dense` of a prepared Q4/Q8 snapshot loads quantized.
    fn load_dir(dir: &Path, device: &Device, requested: Option<QuantSpec>) -> CoreResult<Self> {
        let cfg_value = read_json(dir, "config.json")
            .ok_or_else(|| CoreError::Load(format!("read config.json in {}", dir.display())))?;
        let arch = Architecture::from_config(&cfg_value).map_err(to_core)?;
        let weights = Weights::from_dir(dir, device).map_err(to_core)?;
        let (model, mut descriptor) = if arch == Architecture::Qwen35 {
            // Qwen3.6 hybrid decoder: its own config, the VLM-nested `model.language_model` prefix, and
            // a top-level untied `lm_head`.
            let qcfg = Qwen35Config::from_json(&cfg_value).map_err(to_core)?;
            let descriptor = descriptor_for_qwen35(&qcfg);
            let m =
                Qwen35Model::from_weights_with(&weights, "model.language_model", qcfg, requested)
                    .map_err(to_core)?;
            (Decoder::Qwen35(m), descriptor)
        } else {
            let cfg = ModelConfig::from_dir(dir).map_err(to_core)?;
            let descriptor = descriptor_for(&cfg);
            let quant = requested.or(cfg.quantization);
            let m = CausalLm::from_weights_with(&weights, "", cfg, quant).map_err(to_core)?;
            (Decoder::Causal(m), descriptor)
        };

        // Qwen-VL vision: load the ViT tower when the checkpoint carries `model.visual.*` (a wrapped
        // VLM) and the config exposes a `vision_config`. Covers Qwen3.6 (`qwen3_5`) and Qwen3-VL
        // (`qwen3_vl`), which share the identical Qwen3-VL ViT tower (Qwen3-VL adds DeepStack taps).
        // Absent → a text-only checkpoint.
        let vision = if (arch == Architecture::Qwen35 || arch == Architecture::Qwen3Vl)
            && cfg_value.get("vision_config").is_some()
            && weights.contains("model.visual.patch_embed.proj.weight")
        {
            let vcfg = Qwen35VisionConfig::from_json(&cfg_value).map_err(to_core)?;
            let tower = Qwen35VisionModel::from_weights(&weights, "model.visual", vcfg.clone())
                .map_err(to_core)?;
            // Both real configs carry `image_token_id`; the fallback is the family's canonical id
            // (Qwen3.6 248056, Qwen3-VL 151655) so a hand-rolled config still resolves.
            let image_token_id = cfg_value
                .get("image_token_id")
                .and_then(|x| x.as_i64())
                .map(|x| x as i32)
                .unwrap_or(if arch == Architecture::Qwen3Vl {
                    151655
                } else {
                    248056
                });
            // Video tokens (Qwen3-VL): `video_token_id` (`<|video_pad|>`, 151656) plus the
            // vision_start/end framing the Text–Timestamp-Alignment substitution emits per frame. A
            // checkpoint without `video_token_id` in its config does not advertise video.
            let video_token_id = cfg_value
                .get("video_token_id")
                .and_then(|x| x.as_i64())
                .map(|x| x as i32);
            descriptor.capabilities.supports_vision = true;
            descriptor.capabilities.supports_video = video_token_id.is_some();
            Some(Qwen35Vision {
                tower,
                processor: Qwen35ImageProcessor::default(),
                image_token_id,
                // Fall back to the canonical Qwen3-VL id when absent so the field is always valid;
                // `supports_video` already gates whether the video path is reachable.
                video_token_id: video_token_id.unwrap_or(151656),
                spatial_merge_size: vcfg.spatial_merge_size,
                device: device.clone(),
            })
        } else {
            None
        };

        let tokenizer = Tokenizer::from_file(dir.join("tokenizer.json"))?;
        let stop_tokens = eos_token_ids(dir);
        let (template, supports_thinking, supports_tools) = load_chat_template(dir);
        descriptor.capabilities.supports_thinking = supports_thinking;
        descriptor.capabilities.supports_tools = supports_tools;
        Ok(Self {
            descriptor,
            model,
            tokenizer,
            template,
            stop_tokens,
            constraint_table: OnceCell::new(),
            vision,
        })
    }

    /// Load a single `*.gguf` checkpoint directly into the decoder. The tokenizer prefers a sibling
    /// `tokenizer.json`, falling back to a reconstruction from the GGUF's embedded tokenizer
    /// metadata; the chat template prefers a sibling `tokenizer_config.json`, then the GGUF's own
    /// `chat_template`, then the typed Llama-3 default.
    fn load_gguf(path: &Path, device: &Device, requested: Option<QuantSpec>) -> CoreResult<Self> {
        let ck = GgufCheckpoint::open(path, device).map_err(to_core)?;
        let mut descriptor = descriptor_for(&ck.config);
        let quant = requested.or(ck.config.quantization);
        // GGUF is the dense Llama-family path only (no hybrid Qwen3.6 GGUF remap).
        let model = Decoder::Causal(
            CausalLm::from_weights_with(&ck.weights, "", ck.config.clone(), quant)
                .map_err(to_core)?,
        );

        let dir = path.parent().unwrap_or_else(|| Path::new("."));
        let sibling_tokenizer = dir.join("tokenizer.json");
        let tokenizer = if sibling_tokenizer.is_file() {
            Tokenizer::from_file(sibling_tokenizer)?
        } else {
            ck.tokenizer_from_metadata().map_err(to_core)?
        };

        let stop_tokens = if ck.stop_tokens.is_empty() {
            eos_token_ids(dir)
        } else {
            ck.stop_tokens.clone()
        };

        let (template, supports_thinking, supports_tools) = gguf_chat_template(dir, &ck);
        descriptor.capabilities.supports_thinking = supports_thinking;
        descriptor.capabilities.supports_tools = supports_tools;
        Ok(Self {
            descriptor,
            model,
            tokenizer,
            template,
            stop_tokens,
            constraint_table: OnceCell::new(),
            vision: None, // GGUF is the dense Llama-family path only — no Qwen3.6 VLM.
        })
    }

    /// Whether the loaded model's projections are quantized.
    pub fn is_quantized(&self) -> bool {
        self.model.is_quantized()
    }

    /// Assemble a provider from already-loaded parts with a default Llama-3 template (used by tests
    /// and converters that don't have a `tokenizer_config.json`).
    pub fn from_parts(model: CausalLm, tokenizer: Tokenizer, stop_tokens: Vec<i32>) -> Self {
        Self {
            descriptor: provider_descriptor(),
            model: Decoder::Causal(model),
            tokenizer,
            template: Box::new(Llama3Template),
            stop_tokens,
            constraint_table: OnceCell::new(),
            vision: None,
        }
    }

    /// Build the multimodal prefill: encode each visual (image or video) in **document order**
    /// (preprocess → ViT → merged rows), expand the rendered `image_pad` / `video_pad` placeholders to
    /// the per-visual / per-frame token counts, splice the features into the token embeds, and compute
    /// the interleaved M-RoPE 3-D positions over the image **and** video grids. `prompt_ids` is the
    /// tokenized prompt (one `image_token_id` per image; one `video_token_id` per frame from the
    /// Text–Timestamp-Alignment placeholders). `messages` is the *original* (un-substituted)
    /// conversation, walked to recover the visual order.
    fn prepare_multimodal(
        &self,
        prompt_ids: &[i32],
        messages: &[Message],
    ) -> CoreResult<MultimodalPrefill> {
        let vision = self.vision.as_ref().ok_or_else(|| {
            CoreError::Load("qwen-vl vision: provider has no vision tower".into())
        })?;
        let model = self.model.as_vlm();
        let merge = vision.spatial_merge_size;

        // Walk the conversation in document order; encode each visual once, in order, so the
        // concatenated feature buffer lines up one-to-one with the visual placeholder spans of the
        // (image+video) prompt. An image placeholder expands to one count; a video expands to `grid_t`
        // per-frame counts (`frame_seqlen` each), in frame order. Each tap's DeepStack features are
        // accumulated separately, then concatenated across visuals per tap.
        let mut feats: Vec<Tensor> = Vec::new();
        let mut image_counts: Vec<usize> = Vec::new();
        let mut video_counts: Vec<usize> = Vec::new();
        let mut image_grids: Vec<[i32; 3]> = Vec::new();
        let mut video_grids: Vec<[i32; 3]> = Vec::new();
        let mut deepstack_by_tap: Vec<Vec<Tensor>> = Vec::new();

        let mut push_deepstack = |deepstack: Vec<Tensor>| -> CoreResult<()> {
            if deepstack_by_tap.is_empty() {
                deepstack_by_tap.resize_with(deepstack.len(), Vec::new);
            }
            if deepstack.len() != deepstack_by_tap.len() {
                return Err(CoreError::Load(format!(
                    "qwen-vl vision: inconsistent DeepStack tap count {} != {}",
                    deepstack.len(),
                    deepstack_by_tap.len()
                )));
            }
            for (tap, feature) in deepstack.into_iter().enumerate() {
                deepstack_by_tap[tap].push(feature);
            }
            Ok(())
        };

        for m in messages {
            for c in &m.content {
                match c {
                    Content::Image(img) => {
                        let (f, deepstack, grid) = vision.encode(img)?;
                        image_counts.push(f.dim(0).map_err(|e| to_core(e.into()))?);
                        image_grids.push(grid);
                        feats.push(f);
                        push_deepstack(deepstack)?;
                    }
                    Content::Video(video) => {
                        let (f, deepstack, grid) = vision.encode_video(video)?;
                        // One placeholder count per frame: `frame_seqlen = (h/merge)·(w/merge)`.
                        let [gt, gh, gw] = grid;
                        let frame_seqlen = ((gh / merge) * (gw / merge)) as usize;
                        for _ in 0..gt {
                            video_counts.push(frame_seqlen);
                        }
                        video_grids.push(grid);
                        feats.push(f);
                        push_deepstack(deepstack)?;
                    }
                    Content::Text(_) => {}
                }
            }
        }

        // Expand both placeholder tokens to their per-occurrence counts. Each call only touches its
        // own token, so order across the two is preserved and the result interleaves correctly.
        let img_id = vision.image_token_id;
        let vid_id = vision.video_token_id;
        let expanded =
            expand_vision_placeholders(prompt_ids, img_id, &image_counts).map_err(to_core)?;
        let expanded =
            expand_vision_placeholders(&expanded, vid_id, &video_counts).map_err(to_core)?;
        let visual_pos_mask: Vec<bool> =
            expanded.iter().map(|&id| id == img_id || id == vid_id).collect();

        let refs: Vec<&Tensor> = feats.iter().collect();
        let all_features = match refs.as_slice() {
            [one] => (*one).clone(),
            many => Tensor::cat(many, 0).map_err(|e| to_core(e.into()))?,
        };
        let mut deepstack = Vec::with_capacity(deepstack_by_tap.len());
        for by_visual in deepstack_by_tap {
            let refs: Vec<&Tensor> = by_visual.iter().collect();
            deepstack.push(match refs.as_slice() {
                [one] => (*one).clone(),
                many => Tensor::cat(many, 0).map_err(|e| to_core(e.into()))?,
            });
        }

        // Embed the expanded ids, splice in the vision features (image+video placeholder rows), and
        // compute interleaved-M-RoPE positions over both image and video grids — through the shared
        // `VlmDecode` seam, identical for whichever decoder powers this VLM.
        let placeholders = [img_id, vid_id];
        let ids = input_ids(&expanded, &vision.device).map_err(to_core)?;
        let embeds = model.embed_input_ids(&ids).map_err(to_core)?;
        let spliced = model
            .splice_vision_features(&embeds, &expanded, &all_features, &placeholders)
            .map_err(to_core)?;
        let positions = model
            .mrope_positions_mm(&expanded, &image_grids, img_id, &video_grids, vid_id, merge)
            .map_err(to_core)?;

        Ok(MultimodalPrefill {
            expanded_ids: expanded,
            embeds: spliced,
            positions,
            visual_pos_mask,
            deepstack,
        })
    }
}

/// Adapts a `core_llm::JsonConstraint` to the engine's [`ConstraintMask`] decode seam.
struct JsonMask<'a>(JsonConstraint<'a>);

impl ConstraintMask for JsonMask<'_> {
    fn allowed(&mut self) -> &[bool] {
        self.0.allowed()
    }
    fn accept(&mut self, token: i32) {
        self.0.accept(token as u32);
    }
}

/// Use the model's own Jinja `chat_template` (from `tokenizer_config.json`) when present; otherwise
/// fall back to the typed Llama-3 template. Also reports two template-gated capabilities, detected
/// from the source (not the family, matching the transformers convention):
/// - **thinking** — the template gates an `enable_thinking` kwarg (the Qwen3, … convention; story
///   7707). The Llama-3 fallback never reasons.
/// - **tools** — the template renders tool calls (its source mentions `tool_call`), so it has a
///   `tools` section and the model emits parseable `<tool_call>` blocks (story 7636). Covers the
///   Qwen3.6 XML and the Qwen2.5/Hermes JSON tool templates alike.
fn load_chat_template(dir: &Path) -> (Box<dyn ChatTemplate>, bool, bool) {
    match JinjaChatTemplate::from_tokenizer_config_file(dir.join("tokenizer_config.json")) {
        Ok(t) => {
            let supports_thinking = t.source().contains("enable_thinking");
            let supports_tools = t.source().contains("tool_call");
            (Box::new(t), supports_thinking, supports_tools)
        }
        Err(_) => (Box::new(Llama3Template), false, false),
    }
}

/// Pick a chat template for a GGUF load: a sibling `tokenizer_config.json` first, then the GGUF's
/// own embedded `chat_template` metadata, then the typed Llama-3 default. Also reports
/// `supports_thinking` (the chosen template's source gates `enable_thinking`) and `supports_tools`
/// (its source renders tool calls — it mentions `tool_call`).
fn gguf_chat_template(dir: &Path, ck: &GgufCheckpoint) -> (Box<dyn ChatTemplate>, bool, bool) {
    if let Ok(t) = JinjaChatTemplate::from_tokenizer_config_file(dir.join("tokenizer_config.json"))
    {
        let supports_thinking = t.source().contains("enable_thinking");
        let supports_tools = t.source().contains("tool_call");
        return (Box::new(t), supports_thinking, supports_tools);
    }
    if let Some(src) = &ck.chat_template {
        let supports_thinking = src.contains("enable_thinking");
        let supports_tools = src.contains("tool_call");
        let bos = ck.bos_token.clone().unwrap_or_default();
        let eos = ck.eos_token.clone().unwrap_or_default();
        return (
            Box::new(JinjaChatTemplate::with_tokens(src.clone(), bos, eos)),
            supports_thinking,
            supports_tools,
        );
    }
    (Box::new(Llama3Template), false, false)
}

/// Whether a rendered prompt ends with an **unclosed** `<think>` block — i.e. the chat template
/// opened reasoning in the prompt (a Qwen3-style thinking/auto generation prompt) so the model
/// generates inside it. True iff the last `<think>` occurs after the last `</think>` (or there is no
/// close), so the segmenter is primed into the Thinking channel.
fn prompt_opens_thinking(prompt: &str) -> bool {
    match prompt.rfind("<think>") {
        None => false,
        Some(open) => prompt.rfind("</think>").is_none_or(|close| open > close),
    }
}

/// Run a piece of answer-channel text through the tool-call segmenter when active, returning the
/// plain-content runs to stream (tool-call blocks lifted out + parsed into [`ToolCallSegmenter`]).
/// With no segmenter the text passes straight through, so the non-tools path is byte-identical to
/// before.
fn tool_pieces(seg: &mut Option<ToolCallSegmenter>, text: &str) -> Vec<String> {
    match seg {
        Some(ts) => ts.push(text),
        None => vec![text.to_string()],
    }
}

/// Emit one answer-channel content `piece` as a [`Channel::Content`] token event with the gap-free
/// `emit_index`, accumulating it into `streamed`. Shared by the streaming loop and the
/// end-of-generation tails; `*emit_index` / `*last_id` advance only when text is actually emitted, so
/// the contract's token index stays gap-free across stripped reasoning markers and lifted-out
/// tool-call blocks.
fn emit_content(
    piece: String,
    id: u32,
    streamed: &mut String,
    emit_index: &mut usize,
    last_id: &mut u32,
    on_event: &mut dyn FnMut(CoreEvent),
) {
    streamed.push_str(&piece);
    *last_id = id;
    on_event(CoreEvent::Token {
        id,
        text: piece,
        index: *emit_index,
        channel: Channel::Content,
    });
    *emit_index += 1;
}

impl TextLlm for LlamaProvider {
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
            return Err(CoreError::Canceled); // typed pre-inference cancel
        }

        // Multimodal (Qwen-VL + image/video content): replace image/video blocks with the Qwen-VL
        // placeholder text so the (vision-free) chat template renders the vision framing
        // (`<|vision_start|><|image_pad|><|vision_end|>` per image; the per-frame Text–Timestamp-
        // Alignment string per video). The visuals are encoded + spliced after tokenizing. Text-only
        // requests are unchanged.
        let temporal_patch = self
            .vision
            .as_ref()
            .map(|v| v.processor.temporal_patch_size)
            .unwrap_or(2);
        let (images, videos): (Vec<&ImageRef>, Vec<&VideoRef>) = match &self.vision {
            Some(_) => (collect_images(&req.messages), collect_videos(&req.messages)),
            None => (Vec::new(), Vec::new()),
        };
        let multimodal = !images.is_empty() || !videos.is_empty();
        let substituted;
        let messages: &[Message] = if multimodal {
            substituted = substitute_vision_placeholders(&req.messages, temporal_patch);
            &substituted
        } else {
            &req.messages
        };

        // Render the conversation and tokenize. The template already includes BOS, so encode without
        // auto special tokens. `enable_thinking` flows into the template kwarg so a no-think
        // (Disabled) request injects the model's closed `<think></think>` generation prompt; Auto
        // omits the kwarg (template default).
        let prompt = self.template.render_with(
            messages,
            &RenderOptions {
                add_generation_prompt: true,
                enable_thinking: req.enable_thinking_kwarg(),
                tools: &req.tools,
            },
        )?;
        let prompt_ids: Vec<i32> = self
            .tokenizer
            .encode(&prompt, false)?
            .into_iter()
            .map(|id| id as i32)
            .collect();

        // Encode + splice the visuals and compute M-RoPE positions (the placeholder-expanded prompt
        // becomes the effective sequence). `None` on the text-only path.
        let mm = if multimodal {
            Some(self.prepare_multimodal(&prompt_ids, &req.messages)?)
        } else {
            None
        };
        let prompt_len = mm
            .as_ref()
            .map(|m| m.expanded_ids.len())
            .unwrap_or(prompt_ids.len());

        let config = GenerationConfig {
            max_new_tokens: req.max_new_tokens as usize,
            sampling: map_sampling(&req.sampling),
            seed: req.seed,
            stop_tokens: self.stop_tokens.clone(),
        };

        // Structured-output constraint: build a JSON mask over the cached decode table.
        let mut json_mask = match req.constraint {
            Some(Constraint::Json) => {
                let table = self
                    .constraint_table
                    .get_or_init(|| self.tokenizer.constraint_decode_table());
                Some(JsonMask(JsonConstraint::new(
                    table,
                    self.stop_tokens.iter().map(|&i| i as u32),
                )))
            }
            None => None,
        };

        // A reasoning segmenter when the model advertises a thinking mode: it splits the decoded
        // stream into `<think>…</think>` reasoning vs answer (markers stripped) across the Thinking /
        // Content channels. `None` otherwise, so a non-thinking provider stays on the original
        // single-channel path (byte-identical streaming).
        let thinking_active = self.descriptor.capabilities.supports_thinking;
        let mut segmenter = thinking_active.then(ThinkingSegmenter::default);
        // Some chat templates open the reasoning block *in the prompt* (e.g. a Qwen3 generation
        // prompt ending `…<|im_start|>assistant\n<think>\n`), so the model generates inside the block
        // and only emits the closing `</think>`. Prime the segmenter into the Thinking channel by
        // feeding it that already-rendered opening marker (stripped, emits nothing); a Disabled
        // request renders a *closed* `<think></think>`, so this correctly does not prime.
        if let Some(seg) = segmenter.as_mut() {
            if prompt_opens_thinking(&prompt) {
                let _ = seg.push("<think>");
            }
        }
        // A tool-call segmenter when the request offers tools and the model's template renders them:
        // it lifts `<tool_call>` blocks out of the answer channel (markup excluded from the streamed
        // text) and parses them into structured calls (story 7636). `None` otherwise, so a no-tools
        // request flows straight through `tool_pieces` unchanged.
        let tools_active = self.descriptor.capabilities.supports_tools && !req.tools.is_empty();
        let mut tool_seg = tools_active.then(|| ToolCallSegmenter::new(&req.tools));
        // Reasoning text (Thinking channel) and the answer (Content channel), accumulated as the
        // segmenter releases each span; the answer becomes the result text when thinking is active.
        let mut thinking_buf = String::new();
        let mut streamed = String::new();
        // Contract token index over *emitted* events, not the raw decode step — stripped
        // `<think>`/`</think>` marker tokens produce no event, so this stays gap-free (and equals the
        // step in the common one-delta-per-token, non-thinking case).
        let mut emit_index = 0usize;
        let mut last_id = 0u32; // id of the last emitted token, for the flushed-tail events

        // Drive the internal loop; translate token-id events to contract text-delta events via
        // incremental detokenization (re-decode the running sequence, emit the new suffix). The
        // segmenter (when active) splits each delta into reasoning vs answer.
        let tokenizer = &self.tokenizer;
        let out = {
            let mut acc: Vec<u32> = Vec::new();
            let mut shown = 0usize;
            let mut sink = |ev: StreamEvent| {
                if let StreamEvent::Token { id, step } = ev {
                    let id = id as u32;
                    acc.push(id);
                    if let Ok(text) = tokenizer.decode(&acc, true) {
                        if text.len() > shown {
                            let delta = text[shown..].to_string();
                            shown = text.len();
                            match segmenter.as_mut() {
                                Some(seg) => {
                                    for span in seg.push(&delta) {
                                        match span.channel {
                                            // Reasoning streams straight out (markers already stripped).
                                            Channel::Thinking => {
                                                thinking_buf.push_str(&span.text);
                                                last_id = id;
                                                on_event(CoreEvent::Token {
                                                    id,
                                                    text: span.text,
                                                    index: emit_index,
                                                    channel: Channel::Thinking,
                                                });
                                                emit_index += 1;
                                            }
                                            // Answer text → tool segmenter (lifts out tool-call
                                            // blocks) → emit.
                                            Channel::Content => {
                                                for piece in tool_pieces(&mut tool_seg, &span.text)
                                                {
                                                    emit_content(
                                                        piece,
                                                        id,
                                                        &mut streamed,
                                                        &mut emit_index,
                                                        &mut last_id,
                                                        &mut *on_event,
                                                    );
                                                }
                                            }
                                        }
                                    }
                                }
                                None if tool_seg.is_some() => {
                                    // No reasoning split, but tools are active: route the whole delta
                                    // through the tool segmenter (the answer is the only channel).
                                    for piece in tool_pieces(&mut tool_seg, &delta) {
                                        emit_content(
                                            piece,
                                            id,
                                            &mut streamed,
                                            &mut emit_index,
                                            &mut last_id,
                                            &mut *on_event,
                                        );
                                    }
                                }
                                None => {
                                    // Neither reasoning nor tools: the original single-channel path,
                                    // byte-identical to before either feature existed (raw `step`
                                    // index, no streamed accumulation).
                                    on_event(CoreEvent::Token {
                                        id,
                                        text: delta,
                                        index: step,
                                        channel: Channel::Content,
                                    });
                                }
                            }
                        }
                    }
                }
            };
            let constraint = json_mask.as_mut().map(|m| m as &mut dyn ConstraintMask);
            match &mm {
                // Multimodal: prefill the spliced embeds with interleaved M-RoPE, then decode the
                // continuation (text positions shifted by `mrope_delta`) through the shared loop.
                Some(m) => {
                    let model = self.model.as_vlm();
                    let mut cache = model.make_cache();
                    let (t, h, w, delta) = &m.positions;
                    let first = model
                        .prefill_with_deepstack(
                            &m.embeds,
                            [t.as_slice(), h.as_slice(), w.as_slice()],
                            &mut *cache,
                            &m.visual_pos_mask,
                            &m.deepstack,
                        )
                        .map_err(to_core)?;
                    let shifted = Shifted {
                        model,
                        delta: *delta,
                    };
                    generate_from_prefill(
                        &shifted,
                        &mut *cache,
                        first,
                        m.expanded_ids.clone(),
                        &config,
                        &req.cancel,
                        &mut sink,
                        constraint,
                    )
                    .map_err(to_core)?
                }
                None => generate_with(
                    &self.model,
                    &prompt_ids,
                    &config,
                    &req.cancel,
                    &mut sink,
                    constraint,
                )
                .map_err(to_core)?,
            }
        };

        // End-of-generation tails, in pipeline order. First the reasoning segmenter's held-back
        // partial marker (it turned out not to begin a marker) as current-channel text — reasoning
        // straight out, answer through the tool segmenter; then the tool segmenter's own tail (a held
        // partial `<tool_call>` / an unterminated block surfaced as content).
        if let Some(seg) = segmenter.as_mut() {
            for span in seg.flush() {
                match span.channel {
                    Channel::Thinking => {
                        thinking_buf.push_str(&span.text);
                        on_event(CoreEvent::Token {
                            id: last_id,
                            text: span.text,
                            index: emit_index,
                            channel: Channel::Thinking,
                        });
                        emit_index += 1;
                    }
                    Channel::Content => {
                        for piece in tool_pieces(&mut tool_seg, &span.text) {
                            emit_content(
                                piece,
                                last_id,
                                &mut streamed,
                                &mut emit_index,
                                &mut last_id,
                                &mut *on_event,
                            );
                        }
                    }
                }
            }
        }
        if let Some(ts) = tool_seg.as_mut() {
            for piece in ts.flush() {
                emit_content(
                    piece,
                    last_id,
                    &mut streamed,
                    &mut emit_index,
                    &mut last_id,
                    &mut *on_event,
                );
            }
        }

        // Result text: the streamed answer when thinking or tools are active (either means the
        // streamed channel is the authoritative answer, with reasoning / tool-call markup removed);
        // otherwise the original decode-all-tokens path (byte-identical to the no-feature case).
        // Reasoning and tool calls, if the model produced any, are reported separately (their markup
        // excluded from `text`).
        let text = if thinking_active || tools_active {
            streamed
        } else {
            let gen_u32: Vec<u32> = out.tokens.iter().map(|&i| i as u32).collect();
            tokenizer.decode(&gen_u32, true)?
        };
        let thinking = (!thinking_buf.is_empty()).then_some(thinking_buf);
        let tool_calls = tool_seg.map(|mut ts| ts.take_calls()).unwrap_or_default();
        let finish = map_finish(out.finish_reason);
        let usage = Usage {
            prompt_tokens: prompt_len as u32,
            generated_tokens: out.tokens.len() as u32,
        };
        on_event(CoreEvent::Done {
            finish_reason: finish,
            usage,
        });
        Ok(TextLlmOutput {
            text,
            thinking,
            tool_calls,
            usage,
            finish_reason: Some(finish),
        })
    }
}

/// The descriptor for the `candle-llama` provider (constructible without loading weights; used for
/// link-time registration and registry discovery).
pub fn provider_descriptor() -> TextLlmDescriptor {
    TextLlmDescriptor {
        id: PROVIDER_ID.to_string(),
        family: "llama".to_string(),
        backend: "candle".to_string(),
        capabilities: TextLlmCapabilities {
            max_context_tokens: 0,
            max_new_tokens: 0,
            supports_system_prompt: true,
            // Text-only today; the VLM path flips this on for a vision provider.
            supports_vision: false,
            // Text-only candle-llama accepts no video content.
            supports_video: false,
            // No controllable reasoning mode yet (a separate story); the contract requires an
            // explicit enable-thinking request to be rejected, which validate_request enforces.
            supports_thinking: false,
            // Weightless default: conservative. The load path flips this on when the loaded model's
            // chat template renders tool calls (story 7636).
            supports_tools: false,
            // JSON-constrained decoding.
            supported_constraints: vec![Constraint::Json],
        },
    }
}

/// A descriptor reflecting a *loaded* model: family from the dispatched architecture and the context
/// length from `config.json`. (Quantization state is reported via [`LlamaProvider::is_quantized`].)
fn descriptor_for(cfg: &ModelConfig) -> TextLlmDescriptor {
    let mut d = provider_descriptor();
    d.family = cfg.architecture.family().to_string();
    d.capabilities.max_context_tokens = cfg.max_position_embeddings.max(0) as usize;
    d
}

/// A descriptor for a loaded Qwen3.6 (`qwen3_5`) hybrid decoder. The context length comes from the
/// [`Qwen35Config`] (which `ModelConfig` does not represent). Text-only here; the vision path is a
/// follow-on story.
fn descriptor_for_qwen35(cfg: &Qwen35Config) -> TextLlmDescriptor {
    let mut d = provider_descriptor();
    d.family = Architecture::Qwen35.family().to_string();
    d.capabilities.max_context_tokens = cfg.max_position_embeddings.max(0) as usize;
    d
}

/// Resolve the stop-token ids for a snapshot directory. Prefers `generation_config.json` (HF's
/// canonical "how to generate" source — where models like Qwen3.6 put the turn-end ids; its
/// top-level `config.json` `eos_token_id` is null), then `config.json` (top-level, then the nested
/// `text_config` of a VLM wrapper), then the Llama-3 defaults. Each `eos_token_id` may be a single
/// int or an array.
pub fn eos_token_ids(dir: &Path) -> Vec<i32> {
    let fallback = vec![128001, 128008, 128009]; // <|end_of_text|>, <|eom_id|>, <|eot_id|>
    if let Some(ids) = read_json(dir, "generation_config.json")
        .as_ref()
        .and_then(|v| parse_token_ids(v.get("eos_token_id")))
    {
        return ids;
    }
    if let Some(v) = read_json(dir, "config.json") {
        if let Some(ids) = parse_token_ids(v.get("eos_token_id"))
            .or_else(|| parse_token_ids(v.get("text_config").and_then(|t| t.get("eos_token_id"))))
        {
            return ids;
        }
    }
    fallback
}

/// Read and parse a JSON file in `dir`, or `None` if absent / malformed.
fn read_json(dir: &Path, name: &str) -> Option<serde_json::Value> {
    let text = std::fs::read_to_string(dir.join(name)).ok()?;
    serde_json::from_str(&text).ok()
}

/// Parse an `eos_token_id`-style field — a single int or an array of ints — into a non-empty id list.
fn parse_token_ids(v: Option<&serde_json::Value>) -> Option<Vec<i32>> {
    match v? {
        serde_json::Value::Number(n) => n.as_i64().map(|x| vec![x as i32]),
        serde_json::Value::Array(a) => {
            let ids: Vec<i32> = a
                .iter()
                .filter_map(|x| x.as_i64().map(|x| x as i32))
                .collect();
            (!ids.is_empty()).then_some(ids)
        }
        _ => None,
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

/// Bridge an engine error into the contract error, preserving the typed cancellation / capability /
/// load variants (do not stringify those).
pub(crate) fn to_core(e: crate::Error) -> CoreError {
    match e {
        crate::Error::Canceled => CoreError::Canceled,
        crate::Error::Unsupported(m) => CoreError::Unsupported(m),
        crate::Error::MissingTensor(m) => CoreError::Load(format!("missing tensor: {m}")),
        crate::Error::Config(m) => CoreError::Load(m),
        crate::Error::Io(e) => CoreError::Io(e),
        other => CoreError::backend(other),
    }
}

/// Ordinary registration used by explicit runtime bundles.
pub const REGISTRATION: core_llm::TextLlmRegistration = core_llm::TextLlmRegistration {
    descriptor: provider_descriptor,
    load: load_registered,
    can_load,
    // Per-snapshot vision probe: the static descriptor reports `supports_vision=false` (most
    // snapshots are text-only), but a Qwen3.6 / Qwen3-VL checkpoint with a `vision_config` IS
    // vision-capable — this provider loads its ViT tower alongside the decoder. The probe lets a
    // vision-required model-first load resolve it without reading weights.
    weightless_vision: Some(can_load_vision),
};

// Compatibility registration for consumers that still use link-time discovery.
inventory::submit! { REGISTRATION }

fn load_registered(spec: &LoadSpec) -> CoreResult<Box<dyn TextLlm>> {
    Ok(Box::new(LlamaProvider::load(spec)?))
}

/// Weightless model-first probe (story 7406): can the `candle-llama` provider serve the model at
/// `spec.source`?
///
/// For a `*.gguf` file this reads **only** the GGUF header/metadata (never a tensor block, via
/// [`gguf_architecture`](crate::gguf::gguf_architecture)) and accepts it iff its `general.architecture`
/// is one the native GGUF loader can reconstruct ([`gguf_arch_to_hf`](crate::gguf::gguf_arch_to_hf):
/// `llama`/`mistral` and `qwen3`); an unsupported or non-LLM GGUF (`bert`, a `clip` mmproj, …) is
/// declined so `load_for_model` returns a clean `Unsupported` rather than routing it here to fail at
/// load (story 7420, replacing the earlier extension-only accept).
///
/// Otherwise this reads **only** `config.json` and runs the same [`Architecture::from_config`] dispatch
/// the loader uses (Llama / Mistral / Qwen2 / Qwen3 / Qwen2-MoE / Gemma2 / GLM-4 / DeepSeek-V2 / Phi-3)
/// — it never opens a safetensors shard, so `core-llm`'s `load_for_model` can resolve a provider by
/// model without loading weights. A multimodal snapshot (a `vision_config` block — including a VLM
/// whose `model_type` substring-matches a text family, e.g. `mllama`) is declined so the vision
/// provider claims it instead.
/// Weightless per-snapshot vision probe (core-llm `weightless_vision`): does this provider serve the
/// snapshot at `spec.source` *with* vision? True for a Qwen3.6 (`qwen3_5`) / Qwen3-VL (`qwen3_vl`)
/// HF checkpoint carrying a `vision_config` (the ViT tower loads alongside the decoder). Reads only
/// `config.json` — never a weight shard. GGUF is declined (candle's GGUF path is dense text-only).
pub fn can_load_vision(spec: &LoadSpec) -> bool {
    if crate::gguf::is_gguf_path(&spec.source) {
        return false;
    }
    let dir = Path::new(&spec.source);
    let path = if dir.is_dir() {
        dir.join("config.json")
    } else {
        dir.to_path_buf()
    };
    let Ok(text) = std::fs::read_to_string(path) else {
        return false;
    };
    let Ok(v) = serde_json::from_str::<serde_json::Value>(&text) else {
        return false;
    };
    v.get("vision_config").is_some()
        && matches!(
            Architecture::from_config(&v),
            Ok(Architecture::Qwen35) | Ok(Architecture::Qwen3Vl)
        )
}

pub fn can_load(spec: &LoadSpec) -> bool {
    if crate::gguf::is_gguf_path(&spec.source) {
        // Confirm the GGUF's architecture from its header alone (weightless) — accept iff the loader
        // can actually reconstruct it. A `.gguf` that is missing/corrupt or names an unsupported arch
        // resolves to `None` and is declined.
        return crate::gguf::gguf_architecture(&spec.source)
            .as_deref()
            .and_then(crate::gguf::gguf_arch_to_hf)
            .is_some();
    }
    let dir = Path::new(&spec.source);
    let path = if dir.is_dir() {
        dir.join("config.json")
    } else {
        dir.to_path_buf()
    };
    let Ok(text) = std::fs::read_to_string(path) else {
        return false;
    };
    let Ok(v) = serde_json::from_str::<serde_json::Value>(&text) else {
        return false;
    };
    // A multimodal snapshot (a `vision_config` block) is normally declined so a vision provider claims
    // it — EXCEPT Qwen3.6 (`qwen3_5`) and Qwen3-VL (`qwen3_vl`), which this provider serves directly:
    // their 27B/35B / 8B checkpoints are VLM-wrapped, but this provider carries the matching ViT tower
    // (the decoder + vision both load here). So a `vision_config` config is declined only when it is
    // neither qwen3_5 nor qwen3_vl.
    let arch = Architecture::from_config(&v);
    let serves_vision = matches!(arch, Ok(Architecture::Qwen35) | Ok(Architecture::Qwen3Vl));
    if v.get("vision_config").is_some() && !serves_vision {
        return false;
    }
    arch.is_ok()
}

#[cfg(test)]
mod tests {
    use super::{
        expand_vision_placeholders, merged_frame_timestamps, prompt_opens_thinking,
        video_placeholder_text,
    };
    use core_llm::{ImageRef, VideoRef};

    #[test]
    fn prompt_opens_thinking_matches_template_modes() {
        // A Qwen3-style thinking/auto generation prompt opens the block and leaves it unclosed.
        assert!(prompt_opens_thinking("<|im_start|>assistant\n<think>\n"));
        // A no-think (Disabled) prompt renders a closed `<think></think>`.
        assert!(!prompt_opens_thinking(
            "<|im_start|>assistant\n<think>\n\n</think>\n\n"
        ));
        // A prior closed reasoning turn followed by a fresh open block still opens.
        assert!(prompt_opens_thinking(
            "<think>\nold\n</think>\n\nq<|im_start|>assistant\n<think>\n"
        ));
        // No reasoning markers at all (a non-thinking template).
        assert!(!prompt_opens_thinking("<|im_start|>assistant\n"));
    }

    fn qwen3vl_video_oracle() -> serde_json::Value {
        serde_json::from_str(include_str!("models/testdata/qwen3vl_video_oracle.json")).unwrap()
    }

    /// **The per-frame placeholder string matches `Qwen3VLProcessor.replace_video_token` (collapsed
    /// form).** The engine emits **one** `<|video_pad|>` per frame and expands it to `frame_seqlen`
    /// copies after tokenizing (exactly the image path's pattern, where the chat template renders a
    /// single `<|image_pad|>`). The reference `replace_video_token` writes the *already-expanded*
    /// string (`frame_seqlen` `<|video_pad|>` per frame). Collapsing each consecutive `<|video_pad|>`
    /// run of the reference string to one token must yield exactly [`video_placeholder_text`]: same
    /// `<{t:.1f} seconds>` Text–Timestamp-Alignment tags, same per-frame vision framing.
    #[test]
    fn video_placeholder_string_matches_hf_reference() {
        let j = qwen3vl_video_oracle();
        let fps = j["fps"].as_f64().unwrap() as f32;
        let temporal = j["temporal_patch_size"].as_u64().unwrap() as usize;
        let indices: Vec<f32> = j["frames_indices"]
            .as_array()
            .unwrap()
            .iter()
            .map(|x| x.as_f64().unwrap() as f32)
            .collect();
        // Build a synthetic VideoRef with one 1x1 frame per sampled index carrying its `idx/fps`
        // timestamp (the frame pixels are irrelevant to the placeholder string).
        let frames: Vec<ImageRef> = indices
            .iter()
            .map(|_| ImageRef::new(1, 1, vec![0, 0, 0]).unwrap())
            .collect();
        let timestamps: Vec<f32> = indices.iter().map(|&i| i / fps).collect();
        let video = VideoRef::new(frames, timestamps).unwrap();
        let got = video_placeholder_text(&video, temporal);

        // Collapse the reference string's `<|video_pad|>` runs to a single token per frame.
        let pad = "<|video_pad|>";
        let mut collapsed = j["placeholder_text"].as_str().unwrap().to_string();
        while collapsed.contains(&format!("{pad}{pad}")) {
            collapsed = collapsed.replace(&format!("{pad}{pad}"), pad);
        }
        assert_eq!(
            got, collapsed,
            "collapsed Text–Timestamp-Alignment placeholder string must byte-match HF replace_video_token"
        );
        // The timestamp tags themselves must appear verbatim (the core of Text–Timestamp Alignment).
        for t in j["merged_timestamps"].as_array().unwrap() {
            let tag = format!("<{:.1} seconds>", t.as_f64().unwrap());
            assert!(
                got.contains(&tag),
                "placeholder must carry the `{tag}` timestamp tag: {got}"
            );
        }
    }

    /// `merged_frame_timestamps` averages within each `temporal_patch_size` group (padding the last)
    /// to one timestamp per emitted vision frame — matching `Qwen3VLProcessor._calculate_timestamps`.
    #[test]
    fn merged_timestamps_match_hf_reference() {
        let j = qwen3vl_video_oracle();
        let fps = j["fps"].as_f64().unwrap() as f32;
        let temporal = j["temporal_patch_size"].as_u64().unwrap() as usize;
        let indices: Vec<f32> = j["frames_indices"]
            .as_array()
            .unwrap()
            .iter()
            .map(|x| x.as_f64().unwrap() as f32)
            .collect();
        let per_sample: Vec<f32> = indices.iter().map(|&i| i / fps).collect();
        let got = merged_frame_timestamps(&per_sample, temporal);
        let want: Vec<f32> = j["merged_timestamps"]
            .as_array()
            .unwrap()
            .iter()
            .map(|x| x.as_f64().unwrap() as f32)
            .collect();
        assert_eq!(got.len(), want.len(), "merged timestamp count vs HF");
        for (g, w) in got.iter().zip(&want) {
            assert!((g - w).abs() < 1e-5, "merged timestamp {g} vs HF {w}");
        }
    }

    /// **The per-frame `<|video_pad|>` expansion matches the HF id stream.** Tokenizing the reference
    /// placeholder string yields one `<|video_pad|>` per frame; expanding each to `frame_seqlen`
    /// copies (the merged patch count the ViT emits per frame) must reproduce the exact id stream the
    /// processor produces — same per-frame vision framing, same timestamp tokens, same counts. This is
    /// the video analogue of `expand_vision_placeholders` for images, but with `grid_t` runs.
    #[test]
    fn video_token_expansion_matches_hf_reference() {
        let j = qwen3vl_video_oracle();
        let expanded_hf: Vec<i32> = j["expanded_ids"]
            .as_array()
            .unwrap()
            .iter()
            .map(|x| x.as_i64().unwrap() as i32)
            .collect();
        let vid = j["video_token_id"].as_i64().unwrap() as i32;
        let grid_t = j["grid_t"].as_u64().unwrap() as usize;
        let frame_seqlen = j["frame_seqlen"].as_u64().unwrap() as usize;
        let expected_video_tokens = j["expected_video_tokens"].as_u64().unwrap() as usize;

        // The merged-token count per frame, and total, must agree with the HF processor.
        assert_eq!(
            grid_t * frame_seqlen,
            expected_video_tokens,
            "total video tokens vs HF"
        );
        assert_eq!(
            expanded_hf.iter().filter(|&&x| x == vid).count(),
            expected_video_tokens,
            "video tokens in HF id stream"
        );

        // Reconstruct the *raw* (pre-expansion) ids: collapse each consecutive `<|video_pad|>` run
        // back to a single placeholder. The HF stream has `grid_t` such runs (one per frame), each of
        // `frame_seqlen` tokens; collapsing recovers one `<|video_pad|>` per frame.
        let mut raw = Vec::new();
        let mut i = 0usize;
        let mut runs = 0usize;
        while i < expanded_hf.len() {
            if expanded_hf[i] == vid {
                raw.push(vid);
                runs += 1;
                while i < expanded_hf.len() && expanded_hf[i] == vid {
                    i += 1;
                }
            } else {
                raw.push(expanded_hf[i]);
                i += 1;
            }
        }
        assert_eq!(runs, grid_t, "one <|video_pad|> run per frame (grid_t)");

        // Expanding each per-frame placeholder to `frame_seqlen` reproduces the HF id stream exactly.
        let counts = vec![frame_seqlen; grid_t];
        let expanded = expand_vision_placeholders(&raw, vid, &counts).unwrap();
        assert_eq!(expanded, expanded_hf, "expanded video ids vs HF processor");
    }

    /// **The video M-RoPE positions over the oracle grid are well-formed and per-frame-reset.** Feed
    /// the expanded video id stream + the `video_grid_thw` through `mrope_positions_mm`: the temporal
    /// row must reset to the frame's cursor at each frame (Qwen3-VL's synthetic time axis splits each
    /// `[t,h,w]` into `t` per-frame `[1,h,w]` blocks). The HF-pinned exact-row check lives in
    /// `deepstack`'s mrope oracle test; here we confirm the provider's video grid drives the same path
    /// consistently.
    #[test]
    fn video_mrope_positions_split_frames() {
        let j = qwen3vl_video_oracle();
        let expanded_hf: Vec<i32> = j["expanded_ids"]
            .as_array()
            .unwrap()
            .iter()
            .map(|x| x.as_i64().unwrap() as i32)
            .collect();
        let vid = j["video_token_id"].as_i64().unwrap() as i32;
        let img = vid - 1; // a distinct unused image id
        let g = j["video_grid_thw"].as_array().unwrap()[0].as_array().unwrap();
        let grid = [
            g[0].as_i64().unwrap() as i32,
            g[1].as_i64().unwrap() as i32,
            g[2].as_i64().unwrap() as i32,
        ];
        let merge = j["merge"].as_i64().unwrap() as i32;

        let (t, h, w, _delta) =
            crate::models::deepstack::mrope_positions_mm(&expanded_hf, &[], img, &[grid], vid, merge)
                .unwrap();
        assert_eq!(t.len(), expanded_hf.len());
        // Each frame's video tokens share one temporal index (gt = 1 per frame after the split), and
        // the two frames sit at *different* temporal positions (the cursor advances between them).
        let frame_temporals: Vec<i32> = expanded_hf
            .iter()
            .zip(&t)
            .filter_map(|(&id, &tt)| (id == vid).then_some(tt))
            .collect();
        let distinct: std::collections::BTreeSet<i32> = frame_temporals.iter().copied().collect();
        assert_eq!(
            distinct.len(),
            grid[0] as usize,
            "one distinct temporal index per frame"
        );
        // h/w spans are bounded by the per-frame grid (h/merge, w/merge).
        let max_w = (grid[2] / merge) - 1;
        let frame_ws: Vec<i32> = expanded_hf
            .iter()
            .zip(&w)
            .zip(&t)
            .filter_map(|((&id, &ww), &tt)| (id == vid).then_some(ww - tt))
            .collect();
        assert!(
            frame_ws.iter().all(|&rel| (0..=max_w).contains(&rel)),
            "w within per-frame grid"
        );
        let _ = h;
    }
}
