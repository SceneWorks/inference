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
use std::sync::Mutex;

use candle_core::{DType, Device, Tensor};
use core_llm::{
    AudioRef, Channel, ChatTemplate, Constraint, ConstraintDecodeTable, ConstraintKind, Content,
    Error as CoreError, FinishReason as CoreFinish, GenerationTimings, ImageRef, IncrementalDetok,
    JinjaChatTemplate, JsonConstraint, Llama3Template, LlmMemoryGeometry, LoadSpec, Message,
    ModelSamplingDefaults, MtpCapabilities, MtpMode, MtpStats, Quantize, ReasoningEffort,
    RenderOptions, Result as CoreResult, Sampling, StopMatcher, StreamEvent as CoreEvent, TextLlm,
    TextLlmCapabilities, TextLlmDescriptor, TextLlmOutput, TextLlmRequest, ThinkingSegmenter,
    Tokenizer, ToolCallSegmenter, Usage, VideoRef,
};
use serde_json::Value;

use crate::config::{Architecture, ModelConfig};
use crate::decode::{
    generate_from_prefill_with_stop, generate_qwen35_mtp_multimodal_with_stop,
    generate_qwen35_mtp_timed_with_stop, ConstraintMask, CountingDecode, Decode, DecodePath,
    DecodeRecord, FinishReason, GenerationConfig, Qwen35MtpMultimodalPrompt, RequestSpan,
    RewindableConstraintMask, StreamEvent,
};
use crate::device::select_device;
use crate::gguf::GgufCheckpoint;
use crate::image::Qwen35ImageProcessor;
use crate::models::gemma4_mm;
use crate::models::{
    CausalLm, Gemma4Layout, Gemma4Mm, Gemma4MmConfig, Qwen35Config, Qwen35Model, Qwen35Mtp,
    Qwen35VisionConfig, Qwen35VisionModel, VlmDecode,
};
use crate::primitives::attention::EAGER_ATTN_QUERY_CHUNK_SIZE;
use crate::primitives::nn::input_ids;
use crate::primitives::projection::{ProjectionFormat, QuantSpec, WeightCensus};
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

/// Upper bound on one Qwen3.5/3.6/3.8 recurrent (Gated DeltaNet) state — every linear layer's SSM
/// state plus its conv tail, priced over all `num_layers` at f32 — times the states a request's
/// cache holds at once: the live state plus `retained_checkpoints` rollback checkpoints (the
/// decoder checkpoints at every forward, and each retained checkpoint pins a distinct state once
/// the live state moves on). The provider's caches retain
/// [`REFERENCE_MAX_CHECKPOINTS`](crate::models::qwen35::REFERENCE_MAX_CHECKPOINTS); a step-seam
/// caller must price [`STEP_MAX_CHECKPOINTS`](crate::models::qwen35::STEP_MAX_CHECKPOINTS) (E6).
fn qwen35_recurrent_admission_bytes(c: &Qwen35Config, retained_checkpoints: usize) -> u64 {
    let one_state = (c.num_layers as u64)
        .saturating_mul(c.linear_num_value_heads as u64)
        .saturating_mul(c.linear_value_head_dim as u64)
        .saturating_mul((c.linear_key_head_dim + c.linear_conv_kernel_dim) as u64)
        .saturating_mul(4);
    one_state.saturating_mul(1 + retained_checkpoints as u64)
}

impl Decoder {
    fn memory_geometry(&self) -> LlmMemoryGeometry {
        let (query_heads, kv_heads, head_dim, layers, hidden, intermediate, vocab, recurrent) =
            match self {
                Decoder::Causal(m) => {
                    let c = m.config();
                    (
                        c.num_heads,
                        c.num_kv_heads,
                        c.head_dim,
                        c.num_layers,
                        c.hidden_size,
                        c.intermediate_size,
                        c.vocab_size,
                        0,
                    )
                }
                Decoder::Qwen35(m) => {
                    let c = m.config();
                    (
                        c.num_heads,
                        c.num_kv_heads,
                        c.head_dim,
                        c.num_layers,
                        c.hidden_size,
                        c.intermediate_size,
                        c.vocab_size,
                        qwen35_recurrent_admission_bytes(
                            c,
                            crate::models::qwen35::REFERENCE_MAX_CHECKPOINTS,
                        ),
                    )
                }
            };
        LlmMemoryGeometry {
            query_heads: query_heads.max(0) as u64,
            kv_heads: kv_heads.max(0) as u64,
            head_dim: head_dim.max(0) as u64,
            layers: layers as u64,
            element_bytes: 4,
            hidden_size: hidden as u64,
            intermediate_size: intermediate as u64,
            vocab_size: vocab as u64,
            recurrent_bytes: recurrent,
        }
    }

    fn is_quantized(&self) -> bool {
        match self {
            Decoder::Causal(m) => m.is_quantized(),
            Decoder::Qwen35(m) => m.is_quantized(),
        }
    }

    /// How the decoder computes grouped-query attention (story sc-24132), for the decode record.
    /// The generic causal family still runs `repeat_kv`-expanded attention over its growing cache
    /// (its migration is S10), so it reports `Expanded`; the Qwen3.5 hybrid reports its selector.
    fn attn_formulation(&self) -> crate::primitives::AttnFormulation {
        match self {
            Decoder::Causal(_) => crate::primitives::AttnFormulation::Expanded,
            Decoder::Qwen35(m) => m.attn_formulation(),
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
    fn estimate_workspace(
        &self,
        images: &[&ImageRef],
        videos: &[&VideoRef],
    ) -> CoreResult<(usize, u64)> {
        let cfg = self.tower.config();
        let patch = cfg.patch_size.max(1) as usize;
        let merge = self.spatial_merge_size.max(1) as usize;
        let temporal = self.processor.temporal_patch_size.max(1);
        let mut tokens = 0usize;
        let mut pixels = 0u64;
        for image in images {
            let (h, w) = self
                .processor
                .smart_resize(image.height as usize, image.width as usize)
                .map_err(to_core)?;
            tokens = tokens
                .checked_add((h / patch) * (w / patch) / (merge * merge))
                .ok_or_else(|| {
                    CoreError::InvalidRequest("visual token geometry overflow".into())
                })?;
            pixels = pixels.saturating_add((h as u64).saturating_mul(w as u64).saturating_mul(12));
        }
        for video in videos {
            let frame = video
                .frames
                .first()
                .ok_or_else(|| CoreError::InvalidRequest("video has no frames".into()))?;
            let t = video.frames.len().div_ceil(temporal);
            let (h, w) = self
                .processor
                .smart_resize_with(
                    frame.height as usize,
                    frame.width as usize,
                    t,
                    self.processor.video_min_pixels,
                    self.processor.video_max_pixels,
                )
                .map_err(to_core)?;
            tokens = tokens
                .checked_add(t * (h / patch) * (w / patch) / (merge * merge))
                .ok_or_else(|| {
                    CoreError::InvalidRequest("visual token geometry overflow".into())
                })?;
            pixels = pixels.saturating_add(
                (t as u64)
                    .saturating_mul(temporal as u64)
                    .saturating_mul(h as u64)
                    .saturating_mul(w as u64)
                    .saturating_mul(12),
            );
        }
        let activations = (tokens as u64)
            .checked_mul(cfg.hidden_size.max(0) as u64)
            .and_then(|v| {
                v.checked_mul((cfg.depth + cfg.deepstack_visual_indexes.len() + 2) as u64)
            })
            .and_then(|v| v.checked_mul(4))
            .ok_or_else(|| CoreError::InvalidRequest("visual workspace overflow".into()))?;
        // The tower attends over unmerged patches, not the merged language tokens.
        // Pricing all frames together is conservative even when attention is frame-local.
        let patches = (tokens as u64)
            .checked_mul(merge as u64)
            .and_then(|v| v.checked_mul(merge as u64))
            .ok_or_else(|| CoreError::InvalidRequest("visual patch geometry overflow".into()))?;
        let attention = patches
            .checked_mul(patches)
            .and_then(|v| v.checked_mul(cfg.num_heads.max(0) as u64))
            .and_then(|v| v.checked_mul(12))
            .ok_or_else(|| {
                CoreError::InvalidRequest("visual attention workspace overflow".into())
            })?;
        let mlp = patches
            .checked_mul(cfg.intermediate_size.max(0) as u64)
            .and_then(|v| v.checked_mul(12))
            .ok_or_else(|| CoreError::InvalidRequest("visual MLP workspace overflow".into()))?;
        let required = pixels
            .checked_add(activations)
            .and_then(|v| v.checked_add(attention))
            .and_then(|v| v.checked_add(mlp))
            .ok_or_else(|| CoreError::InvalidRequest("visual workspace overflow".into()))?;
        Ok((tokens, required))
    }

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
        let out = self
            .tower
            .forward_with_deepstack(&pixels, &grid)
            .map_err(to_core)?;
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
        let out = self
            .tower
            .forward_with_deepstack(&pixels, &[grid])
            .map_err(to_core)?;
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

/// Gemma 4's prepared multimodal prefill: the marker-expanded prompt ids and the decoder input
/// embeds with vision / audio feature rows spliced onto the soft-token positions.
///
/// Deliberately *not* [`MultimodalPrefill`]: Gemma 4 has no M-RoPE, no positional compression, and
/// no DeepStack taps, so there is no `mrope_delta` to shift the continuation by and no per-position
/// visual mask to build. Reusing the Qwen-VL struct would mean carrying four fields that must be
/// filled with values that mean nothing here.
struct Gemma4Prefill {
    expanded_ids: Vec<i32>,
    embeds: Tensor,
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
                Content::Text(_) | Content::Video(_) | Content::Audio(_) => None,
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
                Content::Text(_) | Content::Image(_) | Content::Audio(_) => None,
            })
        })
        .collect()
}

/// Collect the audio blocks of a conversation, in order.
fn collect_audio(messages: &[Message]) -> Vec<&AudioRef> {
    messages
        .iter()
        .flat_map(|m| {
            m.content.iter().filter_map(|c| match c {
                Content::Audio(a) => Some(a),
                Content::Text(_) | Content::Image(_) | Content::Video(_) => None,
            })
        })
        .collect()
}

fn request_media_workspace_bytes(messages: &[Message]) -> CoreResult<u64> {
    messages.iter().try_fold(0u64, |total, message| {
        message.content.iter().try_fold(total, |total, content| {
            let bytes = match content {
                Content::Image(image) => (image.pixels.len() as u64).saturating_mul(4),
                Content::Video(video) => video.frames.iter().fold(0u64, |sum, frame| {
                    sum.saturating_add((frame.pixels.len() as u64).saturating_mul(4))
                }),
                Content::Audio(audio) => (audio.samples.len() as u64).saturating_mul(16),
                Content::Text(_) => 0,
            };
            total
                .checked_add(bytes)
                .ok_or_else(|| CoreError::InvalidRequest("media workspace overflow".into()))
        })
    })
}

/// Gemma 4's rendered image marker — the single token its chat template emits for an image part,
/// which the processor (here, [`LlamaProvider::prepare_gemma4`]) expands into
/// `boi` + N soft tokens + `eoi`.
const GEMMA4_IMAGE_MARKER: &str = "<|image|>";
/// Gemma 4's rendered audio marker, expanded to `boa` + M soft tokens + `eoa`.
const GEMMA4_AUDIO_MARKER: &str = "<|audio|>";

/// Replace each image/audio block with Gemma 4's marker text so the (content-free) core-llm chat
/// template contract renders exactly what the model's own Jinja template renders for a typed image /
/// audio part. Video is refused rather than silently rendered as an image: Gemma 4 declares a
/// `video_token_id`, but this provider ships no frame-sampling path for it, and quietly dropping the
/// frames would answer a question about a video from its text alone.
fn substitute_gemma4_placeholders(messages: &[Message]) -> CoreResult<Vec<Message>> {
    messages
        .iter()
        .map(|m| {
            let content = m
                .content
                .iter()
                .map(|c| match c {
                    Content::Image(_) => Ok(Content::text(GEMMA4_IMAGE_MARKER)),
                    Content::Audio(_) => Ok(Content::text(GEMMA4_AUDIO_MARKER)),
                    Content::Text(t) => Ok(Content::Text(t.clone())),
                    Content::Video(_) => Err(CoreError::Unsupported(
                        "[candle-llama] Gemma 4: video input is not supported by this provider \
                         (no frame-sampling path); send frames as individual images"
                            .to_string(),
                    )),
                })
                .collect::<CoreResult<Vec<_>>>()?;
            Ok(Message {
                role: m.role,
                content,
                thinking: m.thinking.clone(),
                tool_calls: m.tool_calls.clone(),
            })
        })
        .collect()
}

/// The compute dtype of the loaded decoder.
fn model_dtype(model: &Decoder) -> DType {
    match model {
        Decoder::Causal(m) => m.compute_dtype(),
        // Gemma 4 never dispatches to the Qwen3.6 hybrid; this arm exists so the helper is total.
        Decoder::Qwen35(m) => m.compute_dtype(),
    }
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
fn substitute_vision_placeholders(
    messages: &[Message],
    temporal_patch_size: usize,
) -> CoreResult<Vec<Message>> {
    const IMAGE_PLACEHOLDER: &str = "<|vision_start|><|image_pad|><|vision_end|>";
    messages
        .iter()
        .map(|m| {
            let content = m
                .content
                .iter()
                .map(|c| {
                    match c {
                    Content::Image(_) => Ok(Content::text(IMAGE_PLACEHOLDER)),
                    Content::Video(v) => {
                        v.validate().map_err(CoreError::InvalidRequest)?;
                        Ok(Content::text(video_placeholder_text(v, temporal_patch_size)))
                    }
                    Content::Text(t) => Ok(Content::Text(t.clone())),
                    // The Qwen-VL path has no audio projector, and this provider's `supports_audio`
                    // is false for every Qwen checkpoint, so `validate` rejects an audio-carrying
                    // request before substitution. Erroring here rather than dropping the block
                    // means that if that invariant ever breaks, the request fails loudly instead of
                    // being answered from its text alone.
                    Content::Audio(_) => Err(CoreError::Unsupported(
                        "[candle-llama] the Qwen-VL path carries no audio; an audio block reached \
                         placeholder substitution, which the capability gate should have rejected"
                            .to_string(),
                    )),
                }
                })
                .collect::<CoreResult<Vec<_>>>()?;
            Ok(Message {
                role: m.role,
                content,
                thinking: m.thinking.clone(),
                tool_calls: m.tool_calls.clone(),
            })
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
    /// Complete checkpoint-native Qwen3.8 MTP predictor. Its absence never affects ordinary
    /// autoregressive inference and is reflected by an absent MTP capability.
    mtp: Option<Qwen35Mtp>,
    tokenizer: Tokenizer,
    template: Box<dyn ChatTemplate>,
    stop_tokens: Vec<i32>,
    /// Cached per-vocab decode table for constrained decoding — built once (it decodes the whole
    /// vocabulary) on the first JSON-constrained request, then reused.
    constraint_table: OnceCell<ConstraintDecodeTable>,
    /// The Qwen3.6 vision tower + preprocessor, present iff this is a `qwen3_5` checkpoint carrying
    /// `model.visual.*`. Drives the image path in [`LlamaProvider::generate`].
    vision: Option<Qwen35Vision>,
    /// Gemma 4's encoder-free vision embedder and/or audio projector, present iff this is a Gemma 4
    /// checkpoint that actually ships them (sc-18772). Independent of [`vision`](Self::vision):
    /// Gemma 4 does not use the Qwen-VL ViT/M-RoPE/DeepStack machinery at all.
    gemma4: Option<Gemma4Runtime>,
    /// The measured [`DecodeRecord`] of the most recent [`TextLlm::generate`] call — which decode
    /// path ran and its counters (sc-24129). The backend-neutral `TextLlmOutput` carries only the
    /// MTP subset (`MtpStats`); the full record is read back through
    /// [`LlamaProvider::last_decode_record`].
    last_decode: Mutex<Option<DecodeRecord>>,
    /// What the load produced (sc-24135): the requested weight format and, for the qwen3_5 family,
    /// the resident weight census by projection kind. Read through [`LlamaProvider::load_record`].
    load_record: LoadRecord,
}

/// The load telemetry of a [`LlamaProvider`] (sc-24135, epic sc-24128 E2): which weight format was
/// requested and which projection kinds the decoder actually holds, with their resident bytes.
///
/// An NVFP4 request either loads with every requested projection under
/// `census.projections.nvfp4` or fails at load — this record is how a caller (and the evidence
/// harness) sees that, and reads the resident bits/param.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct LoadRecord {
    /// The load-time weight format the caller requested (`None` = the checkpoint's own dtype, or
    /// its persisted `quantization` block).
    pub requested: Option<Quantize>,
    /// The decoder's resident weights by projection kind (target plus native MTP predictor), for
    /// the qwen3_5-family decoders; `None` for the other architectures, which do not report one.
    pub census: Option<WeightCensus>,
}

/// Whether Gemma 4's vision path has been validated end-to-end, and may therefore be advertised.
///
/// **`false`** (sc-18772). The front-end is implemented and its pieces are unit-tested, but its
/// end-to-end behaviour could not be demonstrated on real weights, and two of its steps had to be
/// chosen without a reference implementation to check against (upstream `transformers` with
/// `Gemma4Unified` is not available offline):
///
/// * the patch-grid tiling — how an arbitrary aspect ratio maps onto a patch grid within the
///   280-soft-token budget, and
/// * the resampling of the 1120-entry positional table onto that grid.
///
/// Measured on `google/gemma-4-12B-it` at Q4 on CPU, through this provider:
///
/// * a text-only prompt through the SAME provider instance answers cleanly ("The sun dipped below
///   the horizon, bleeding hues of molten gold ...");
/// * audio conditioning answers cleanly and discriminates content (a 440 Hz tone is described as
///   "a low-frequency, rhythmic thumping or pulsing", silence as "I cannot hear anything");
/// * an image-conditioned prompt degenerates into a `thought` loop and, where it emits anything
///   else, describes the input as **audio**. The span itself is correct — 261 prompt tokens for a
///   17-token question is exactly `boi + 11x22 soft tokens + eoi` — so the framing and expansion
///   are right and the *features* are wrong. Both plausible patch flattening orders (row-major and
///   the channel-major Conv2d equivalent) were tried on real weights and fail the same way.
///
/// Advertising vision on that evidence is exactly the advertised-but-absent failure this story
/// forbids, so an image-carrying request is REJECTED by the capability gate rather than answered
/// from its text. The embedder and its host-side geometry stay in the tree, unadvertised and
/// unreachable, so the next attempt starts from them and from the real-weight leg that must pass
/// before this returns `true` (`gemma4_answers_about_an_image` in the breadth suite).
fn gemma4_vision_is_validated() -> bool {
    false
}

/// Gemma 4's loaded multimodal front-ends plus the tensor context they build in.
struct Gemma4Runtime {
    mm: Gemma4Mm,
    dtype: DType,
    device: Device,
}

/// Qwen3.5/3.8 ships both VLM-wrapped and flat text-only checkpoints. The decoder math is the
/// same; only the root of its HF tensor names differs. Reject mixed roots rather than silently
/// choosing one set of weights from an ambiguous snapshot.
fn qwen35_dense_prefix(has_key: impl Fn(&str) -> bool) -> CoreResult<&'static str> {
    match (
        has_key("model.language_model.embed_tokens.weight"),
        has_key("model.embed_tokens.weight"),
    ) {
        (true, false) => Ok("model.language_model"),
        (false, true) => Ok("model"),
        (false, false) => Err(CoreError::Load(
            "qwen3_5 checkpoint has no wrapped or flat decoder embeddings".into(),
        )),
        (true, true) => Err(CoreError::Load(
            "qwen3_5 checkpoint has both wrapped and flat decoder embeddings".into(),
        )),
    }
}

/// Whether a parsed config is the frozen Qwen3.8 parent served by this release. Keep this
/// deliberately narrower than the generic Qwen3.5 decoder so flat text-only fine-tunes retain
/// their existing CPU support.
fn is_frozen_qwen38_config(config: &Value) -> bool {
    let text = config.get("text_config").and_then(Value::as_object);
    config.get("model_type").and_then(Value::as_str) == Some("qwen3_5")
        && config
            .get("architectures")
            .and_then(Value::as_array)
            .is_some_and(|architectures| {
                architectures.iter().any(|architecture| {
                    architecture.as_str() == Some("Qwen3_5ForConditionalGeneration")
                })
            })
        && text.is_some_and(|text| {
            text.get("model_type").and_then(Value::as_str) == Some("qwen3_5_text")
                && text.get("hidden_size").and_then(Value::as_u64) == Some(5120)
                && text.get("num_hidden_layers").and_then(Value::as_u64) == Some(64)
                && text.get("vocab_size").and_then(Value::as_u64) == Some(248_320)
                && text.get("mtp_num_hidden_layers").and_then(Value::as_u64) == Some(1)
        })
}

fn requires_accelerator(source: &Path) -> CoreResult<bool> {
    if crate::gguf::is_gguf_path(&source.to_string_lossy()) {
        return crate::prism_checkpoint::PrismGgufCheckpoint::is_prism(source).map_err(to_core);
    }
    Ok(read_json(source, "config.json").is_some_and(|config| {
        config.get("model_type").and_then(Value::as_str) == Some("prism_hadamard_qwen35")
            || is_frozen_qwen38_config(&config)
    }))
}

/// The NVFP4 projection format for a load, or `None` when `spec` does not request NVFP4
/// (sc-24135). The capability floor is settled here — before admission or any weight read — so a
/// CPU or sub-sm_120 device is a typed refusal naming the capability (`CoreError::Unsupported`
/// carrying "nvfp4"), never a fallback to another representation. A GGUF source is refused too:
/// it is already block-quantized, and NVFP4 quantizes from a dense snapshot. So is a snapshot
/// whose `config.json` names an architecture outside the qwen3_5 family — also before admission,
/// so a memory refusal can never mask the capability refusal.
fn nvfp4_format(spec: &LoadSpec, device: &Device) -> CoreResult<Option<ProjectionFormat>> {
    nvfp4_format_with(spec, device, ProjectionFormat::nvfp4)
}

/// [`nvfp4_format`] with the device gate injected (`gate` is [`ProjectionFormat::nvfp4`] on the
/// real path), so the gate's refusal can be driven through this helper with a mocked compute
/// capability.
fn nvfp4_format_with(
    spec: &LoadSpec,
    device: &Device,
    gate: impl FnOnce(&Device) -> crate::Result<ProjectionFormat>,
) -> CoreResult<Option<ProjectionFormat>> {
    if spec.quantize != Some(Quantize::Nvfp4) {
        return Ok(None);
    }
    if crate::gguf::is_gguf_path(&spec.source) {
        return Err(CoreError::Unsupported(
            "nvfp4: NVFP4 projections are quantized from a dense safetensors snapshot; a GGUF \
             checkpoint is already block-quantized"
                .into(),
        ));
    }
    // NVFP4's first (and so far only) consumer is the qwen3_5 family (sc-24135); refuse the rest
    // by name. A missing or unreadable config (or a Prism snapshot, which `load_dir` refuses for
    // any repacking) is left to the loader's own error.
    if let Some(config) = read_json(Path::new(&spec.source), "config.json") {
        let is_prism =
            config.get("model_type").and_then(Value::as_str) == Some("prism_hadamard_qwen35");
        if let (false, Ok(arch)) = (is_prism, Architecture::from_config(&config)) {
            if arch != Architecture::Qwen35 {
                return Err(CoreError::Unsupported(format!(
                    "nvfp4: NVFP4 projections are served for the qwen3_5 family \
                     (Qwen3.5/3.6/3.8) only; this checkpoint is {arch:?}"
                )));
            }
        }
    }
    gate(device).map(Some).map_err(to_core)
}

fn ensure_supported_device(source: &Path, device: &Device) -> CoreResult<()> {
    if device.is_cpu() && requires_accelerator(source)? {
        return Err(CoreError::Load(
            "Qwen3.8 and Prism/Bonsai require an accelerator; Candle CPU inference is unsupported"
                .into(),
        ));
    }
    Ok(())
}

impl LlamaProvider {
    /// The measured decode record of the most recent `generate` call on this provider, or `None`
    /// before the first. Says which path ran (`Reference`, `Mtp { drafts }`, …) and its counters;
    /// the MTP subset also reaches callers as `TextLlmOutput::mtp`.
    pub fn last_decode_record(&self) -> Option<DecodeRecord> {
        *self
            .last_decode
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner())
    }

    /// Load a provider from `spec.source`: either a `*.gguf` file (loaded directly via Candle's
    /// native GGUF reader, story 7254) or an HF snapshot directory (config.json + tokenizer.json +
    /// shards). Either way the decoder architecture is dispatched (Llama / Mistral / Qwen3) and the
    /// projections are optionally quantized on load per `spec.quantize`.
    pub fn load(spec: &LoadSpec) -> CoreResult<Self> {
        if spec.projector_source.is_some() && !crate::gguf::is_gguf_path(&spec.source) {
            return Err(CoreError::Load("an external projector is only valid with a GGUF language checkpoint; safetensors vision must be embedded".into()));
        }
        let device = select_device().map_err(to_core)?;
        // NVFP4 (sc-24135): the capability floor is settled first — before the accelerator gate,
        // admission or any weight read — so a CPU or sub-sm_120 device answers an NVFP4 request with
        // the typed refusal naming the capability, never a fallback to another representation.
        let nvfp4 = nvfp4_format(spec, &device)?;
        ensure_supported_device(Path::new(&spec.source), &device)?;
        let payload = core_llm::checkpoint_payload_bytes(Path::new(&spec.source))?;
        let staging = core_llm::checkpoint_staging_bytes(Path::new(&spec.source))?;
        let projector = spec
            .projector_source
            .as_ref()
            .map(|p| core_llm::checkpoint_payload_bytes(Path::new(p)))
            .transpose()?
            .unwrap_or(0);
        let packed = crate::gguf::is_gguf_path(&spec.source)
            || read_json(Path::new(&spec.source), "config.json").is_some_and(|c| {
                c.get("model_type").and_then(Value::as_str) == Some("prism_hadamard_qwen35")
            });
        let (host_required, device_required) = load_memory_requirements(
            payload,
            staging,
            projector,
            packed,
            device.is_cuda(),
            nvfp4.is_some(),
        )
        .ok_or_else(|| CoreError::Load("load memory estimate overflow".into()))?;
        let budget = core_llm::operational_memory_override()?;
        // A discrete CUDA launcher's override describes device headroom. Host staging is a separate
        // allocation domain and must be checked against host capacity rather than GPU capacity.
        let host_available = core_llm::effective_memory_budget(
            core_llm::available_host_memory_bytes(),
            host_load_budget(device.is_cuda(), budget),
        )?;
        core_llm::admit_request_memory(host_required, host_available)?;
        if let Some(device_required) = device_required {
            core_llm::admit_request_memory(device_required, request_available_memory(&device)?)?;
        }

        let requested = match spec.quantize {
            None => None,
            Some(Quantize::Q4) => Some(ProjectionFormat::from(QuantSpec::q4())),
            Some(Quantize::Q8) => Some(ProjectionFormat::from(QuantSpec::q8())),
            Some(Quantize::Nvfp4) => nvfp4,
        };
        let mut provider = if crate::gguf::is_gguf_path(&spec.source) {
            Self::load_gguf(
                Path::new(&spec.source),
                spec.projector_source.as_deref().map(Path::new),
                &device,
                requested.as_ref().and_then(ProjectionFormat::ggml),
            )?
        } else {
            if spec.projector_source.is_some() {
                return Err(CoreError::Load(
                    "an external projector is only valid with a GGUF language checkpoint; safetensors vision must be embedded".into(),
                ));
            }
            Self::load_dir(Path::new(&spec.source), &device, requested.as_ref())?
        };
        provider.load_record.requested = spec.quantize;
        Ok(provider)
    }

    /// Load from an HF snapshot directory (config.json + tokenizer.json + safetensors shards).
    ///
    /// `requested` is an explicit load-time quantization (`spec.quantize`); when it is `None` the
    /// snapshot's own persisted `quantization` block (written by the [`prepare`](crate::prepare)
    /// writer) is honored, so a `LoadSpec::dense` of a prepared Q4/Q8 snapshot loads quantized.
    fn load_dir(
        dir: &Path,
        device: &Device,
        requested: Option<&ProjectionFormat>,
    ) -> CoreResult<Self> {
        let cfg_value = read_json(dir, "config.json")
            .ok_or_else(|| CoreError::Load(format!("read config.json in {}", dir.display())))?;
        let is_prism =
            cfg_value.get("model_type").and_then(Value::as_str) == Some("prism_hadamard_qwen35");
        if is_prism && requested.is_some() {
            return Err(CoreError::Load(
                "Prism/Bonsai is already packed affine-2; Q4/Q8/NVFP4 repacking would expand the model"
                    .into(),
            ));
        }
        let arch = if is_prism {
            Architecture::Qwen35
        } else {
            Architecture::from_config(&cfg_value).map_err(to_core)?
        };
        // A non-qwen3_5 NVFP4 request was already refused by name in `nvfp4_format`, before
        // admission (sc-24135).
        debug_assert!(
            !(requested.is_some_and(ProjectionFormat::is_nvfp4) && arch != Architecture::Qwen35),
            "nvfp4_format refuses non-qwen3_5 NVFP4 before load_dir"
        );
        let (weights, prism) = if is_prism {
            let checkpoint =
                crate::prism_checkpoint::PrismMlxCheckpoint::open(dir, device, &cfg_value)
                    .map_err(to_core)?;
            (checkpoint.weights, Some(checkpoint.registry))
        } else {
            (Weights::from_dir(dir, device).map_err(to_core)?, None)
        };
        let (model, mut descriptor, mtp) = if arch == Architecture::Qwen35 {
            // Qwen3.6 hybrid decoder: its own config, the VLM-nested `model.language_model` prefix, and
            // a top-level untied `lm_head`.
            let qcfg = Qwen35Config::from_json(&cfg_value).map_err(to_core)?;
            let mut descriptor = descriptor_for_qwen35(&qcfg);
            if is_prism {
                descriptor.family = "prism_hadamard_qwen35".into();
                descriptor.capabilities.model_sampling_defaults = Some(bonsai_sampling_defaults());
            }
            let m = if let Some(registry) = prism.as_ref() {
                Qwen35Model::from_prism_weights(
                    &weights,
                    "language_model.model",
                    qcfg,
                    registry,
                    crate::device::compute_dtype(device),
                )
                .map_err(to_core)?
            } else {
                let prefix = qwen35_dense_prefix(|key| weights.contains(key))?;
                Qwen35Model::from_weights_format(&weights, prefix, qcfg, requested)
                    .map_err(to_core)?
            };
            let configured_layers = m.config().mtp_num_hidden_layers;
            let has_mtp_tensors = weights.keys().any(|key| key.starts_with("mtp."));
            let mtp = if configured_layers > 0 {
                Some(Qwen35Mtp::from_weights_format(&weights, &m, requested).map_err(to_core)?)
            } else if has_mtp_tensors {
                return Err(CoreError::Load(
                    "qwen3_5 checkpoint carries MTP tensors while config disables MTP".into(),
                ));
            } else {
                None
            };
            (Decoder::Qwen35(m), descriptor, mtp)
        } else {
            let cfg = ModelConfig::from_dir(dir).map_err(to_core)?;
            let descriptor = descriptor_for(&cfg);
            let quant = requested
                .and_then(ProjectionFormat::ggml)
                .or(cfg.quantization);
            let m = CausalLm::from_weights_with(&weights, "", cfg, quant).map_err(to_core)?;
            (Decoder::Causal(m), descriptor, None)
        };
        if mtp.is_some() {
            descriptor.capabilities.mtp = Some(MtpCapabilities {
                max_draft_tokens: u32::MAX,
                recommended_draft_tokens: 3,
            });
        }

        // Qwen-VL vision: load the ViT tower when the checkpoint carries `model.visual.*` (a wrapped
        // VLM) and the config exposes a `vision_config`. Covers Qwen3.6 (`qwen3_5`) and Qwen3-VL
        // (`qwen3_vl`), which share the identical Qwen3-VL ViT tower (Qwen3-VL adds DeepStack taps).
        // Absent → a text-only checkpoint.
        let dense_vision = weights.contains("model.visual.patch_embed.proj.weight");
        let prism_vision = is_prism && weights.contains("vision_tower.patch_embed.proj.weight");
        let vision = if (arch == Architecture::Qwen35 || arch == Architecture::Qwen3Vl)
            && cfg_value.get("vision_config").is_some()
            && (dense_vision || prism_vision)
        {
            let vcfg = Qwen35VisionConfig::from_json(&cfg_value).map_err(to_core)?;
            let tower = if prism_vision {
                Qwen35VisionModel::from_mlx_weights(&weights, "vision_tower", vcfg.clone())
            } else {
                Qwen35VisionModel::from_weights(&weights, "model.visual", vcfg.clone())
            }
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

        // Gemma 4 multimodal (sc-18772): load whichever front-ends the checkpoint actually ships.
        // The capability flags are set from the LOADED tensors, never from the config alone — a
        // `vision_config` with no `model.vision_embedder.*` yields `None` and stays unadvertised,
        // which is the difference between declaring a capability and having one.
        let gemma4 = if arch.is_gemma4() {
            let layout = Gemma4Layout::detect(&weights).map_err(to_core)?;
            let processor = read_json(dir, "processor_config.json");
            let mm_cfg =
                Gemma4MmConfig::from_json(&cfg_value, processor.as_ref()).map_err(to_core)?;
            let dtype = model_dtype(&model);
            let mm = Gemma4Mm::from_weights(&weights, layout, mm_cfg, dtype).map_err(to_core)?;
            // Audio tracks the loaded tensors; vision does not — see `gemma4_vision_is_validated`.
            descriptor.capabilities.supports_vision =
                mm.vision.is_some() && gemma4_vision_is_validated();
            descriptor.capabilities.supports_audio = mm.audio.is_some();
            mm.audio.is_some().then(|| Gemma4Runtime {
                mm,
                dtype,
                device: device.clone(),
            })
        } else {
            None
        };

        let tokenizer = Tokenizer::from_file(dir.join("tokenizer.json"))?;
        let stop_tokens = eos_token_ids(dir);
        let (
            template,
            supports_thinking,
            supports_reasoning_effort,
            supports_preserve_thinking,
            supports_tools,
        ) = load_chat_template(dir);
        descriptor.capabilities.supports_thinking = supports_thinking;
        descriptor.capabilities.supports_reasoning_effort = supports_reasoning_effort;
        if supports_reasoning_effort && arch == Architecture::Qwen35 {
            descriptor.capabilities.reasoning_efforts = if is_prism {
                vec![ReasoningEffort::XHigh, ReasoningEffort::Medium]
            } else {
                vec![
                    ReasoningEffort::XHigh,
                    ReasoningEffort::Medium,
                    ReasoningEffort::Low,
                ]
            };
        }
        descriptor.capabilities.supports_preserve_thinking = supports_preserve_thinking;
        descriptor.capabilities.supports_tools = supports_tools;
        let census = match &model {
            Decoder::Qwen35(m) => {
                let mut census = m.weight_census();
                if let Some(mtp) = &mtp {
                    census.merge(&mtp.weight_census());
                }
                Some(census)
            }
            Decoder::Causal(_) => None,
        };
        Ok(Self {
            descriptor,
            model,
            mtp,
            tokenizer,
            template,
            stop_tokens,
            constraint_table: OnceCell::new(),
            last_decode: Mutex::new(None),
            vision,
            gemma4,
            load_record: LoadRecord {
                requested: None,
                census,
            },
        })
    }

    /// Load a single `*.gguf` checkpoint directly into the decoder. The tokenizer prefers a sibling
    /// `tokenizer.json`, falling back to a reconstruction from the GGUF's embedded tokenizer
    /// metadata; the chat template prefers a sibling `tokenizer_config.json`, then the GGUF's own
    /// `chat_template`, then the typed Llama-3 default.
    fn load_gguf(
        path: &Path,
        projector_path: Option<&Path>,
        device: &Device,
        requested: Option<QuantSpec>,
    ) -> CoreResult<Self> {
        if crate::prism_checkpoint::PrismGgufCheckpoint::is_prism(path).map_err(to_core)? {
            if requested.is_some() {
                return Err(CoreError::Load(
                    "Prism/Bonsai GGUF is already packed; Q4/Q8 repacking would expand the model"
                        .into(),
                ));
            }
            let ck = crate::prism_checkpoint::PrismGgufCheckpoint::open(path, device)
                .map_err(to_core)?;
            let qcfg = Qwen35Config::from_json(&ck.config_json).map_err(to_core)?;
            let language_hidden_size = qcfg.hidden_size as usize;
            let mut descriptor = descriptor_for_qwen35(&qcfg);
            descriptor.capabilities.model_sampling_defaults = Some(bonsai_sampling_defaults());
            let model = Qwen35Model::from_prism_weights(
                &ck.weights,
                "language_model.model",
                qcfg,
                &ck.registry,
                crate::device::compute_dtype(device),
            )
            .map_err(to_core)?;
            let dir = path.parent().unwrap_or_else(|| Path::new("."));
            let sibling_tokenizer = dir.join("tokenizer.json");
            let tokenizer = if sibling_tokenizer.is_file() {
                Tokenizer::from_file(sibling_tokenizer)?
            } else {
                ck.tokenizer().map_err(to_core)?
            };
            let (template, thinking, effort, preserve, tools): (
                Box<dyn ChatTemplate>,
                bool,
                bool,
                bool,
                bool,
            ) = if let Ok(t) =
                JinjaChatTemplate::from_tokenizer_config_file(dir.join("tokenizer_config.json"))
            {
                let src = t.source();
                let flags = (
                    src.contains("enable_thinking"),
                    src.contains("reasoning_effort"),
                    src.contains("preserve_thinking"),
                    src.contains("tool_call"),
                );
                (Box::new(t), flags.0, flags.1, flags.2, flags.3)
            } else if let Some(src) = ck.chat_template.as_ref() {
                let flags = (
                    src.contains("enable_thinking"),
                    src.contains("reasoning_effort"),
                    src.contains("preserve_thinking"),
                    src.contains("tool_call"),
                );
                (
                    Box::new(JinjaChatTemplate::with_tokens(
                        src.clone(),
                        ck.bos_token.clone().unwrap_or_default(),
                        ck.eos_token.clone().unwrap_or_default(),
                    )),
                    flags.0,
                    flags.1,
                    flags.2,
                    flags.3,
                )
            } else {
                (Box::new(Llama3Template), false, false, false, false)
            };
            descriptor.capabilities.supports_thinking = thinking;
            descriptor.capabilities.supports_reasoning_effort = effort;
            if effort {
                descriptor.capabilities.reasoning_efforts =
                    vec![ReasoningEffort::XHigh, ReasoningEffort::Medium];
            }
            descriptor.capabilities.supports_preserve_thinking = preserve;
            descriptor.capabilities.supports_tools = tools;
            let vision = if let Some(projector_path) = projector_path {
                let loaded = crate::prism_vision_gguf::PrismVisionGguf::open(
                    projector_path,
                    device,
                    language_hidden_size,
                )
                .map_err(to_core)?;
                let one_token = |text: &str| -> CoreResult<i32> {
                    let ids = tokenizer.encode(text, false)?;
                    if ids.len() != 1 {
                        return Err(CoreError::Load(format!(
                            "Bonsai GGUF tokenizer must encode {text:?} as one special token, got {ids:?}"
                        )));
                    }
                    i32::try_from(ids[0]).map_err(|_| {
                        CoreError::Load(format!("Bonsai GGUF token id for {text:?} overflows i32"))
                    })
                };
                let image_token_id = one_token("<|image_pad|>")?;
                let video_token_id = one_token("<|video_pad|>")?;
                descriptor.capabilities.supports_vision = true;
                descriptor.capabilities.supports_video = true;
                Some(Qwen35Vision {
                    tower: loaded.model,
                    processor: Qwen35ImageProcessor::default(),
                    image_token_id,
                    video_token_id,
                    spatial_merge_size: loaded.config.spatial_merge_size,
                    device: device.clone(),
                })
            } else {
                None
            };
            return Ok(Self {
                descriptor,
                model: Decoder::Qwen35(model),
                mtp: None,
                tokenizer,
                template,
                stop_tokens: ck.stop_tokens,
                last_decode: Mutex::new(None),
                constraint_table: OnceCell::new(),
                vision,
                gemma4: None,
                load_record: LoadRecord::default(),
            });
        }
        if projector_path.is_some() {
            return Err(CoreError::Load(
                "external qwen3vl_merger projectors are supported only for Prism/Bonsai GGUF language checkpoints".into(),
            ));
        }
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

        let (
            template,
            supports_thinking,
            supports_reasoning_effort,
            supports_preserve_thinking,
            supports_tools,
        ) = gguf_chat_template(dir, &ck);
        descriptor.capabilities.supports_thinking = supports_thinking;
        descriptor.capabilities.supports_reasoning_effort = supports_reasoning_effort;
        descriptor.capabilities.supports_preserve_thinking = supports_preserve_thinking;
        descriptor.capabilities.supports_tools = supports_tools;
        Ok(Self {
            descriptor,
            model,
            mtp: None,
            tokenizer,
            template,
            stop_tokens,
            last_decode: Mutex::new(None),
            constraint_table: OnceCell::new(),
            vision: None, // GGUF is the dense Llama-family path only — no Qwen3.6 VLM.
            // Likewise no Gemma 4 front-ends: the GGUF path reconstructs a dense text decoder,
            // and llama.cpp GGUFs carry no vision embedder / audio projector tensors.
            gemma4: None,
            load_record: LoadRecord::default(),
        })
    }

    /// Whether the loaded model's projections are quantized.
    pub fn is_quantized(&self) -> bool {
        self.model.is_quantized()
    }

    /// The load telemetry: the requested weight format and the resident weight census by
    /// projection kind (sc-24135).
    pub fn load_record(&self) -> LoadRecord {
        self.load_record
    }

    /// Assemble a provider from already-loaded parts with a default Llama-3 template (used by tests
    /// and converters that don't have a `tokenizer_config.json`).
    pub fn from_parts(model: CausalLm, tokenizer: Tokenizer, stop_tokens: Vec<i32>) -> Self {
        Self {
            descriptor: provider_descriptor(),
            model: Decoder::Causal(model),
            mtp: None,
            tokenizer,
            template: Box::new(Llama3Template),
            stop_tokens,
            last_decode: Mutex::new(None),
            constraint_table: OnceCell::new(),
            vision: None,
            gemma4: None,
            load_record: LoadRecord::default(),
        }
    }

    /// Build the multimodal prefill: encode each visual (image or video) in **document order**
    /// (preprocess → ViT → merged rows), expand the rendered `image_pad` / `video_pad` placeholders to
    /// the per-visual / per-frame token counts, splice the features into the token embeds, and compute
    /// the interleaved M-RoPE 3-D positions over the image **and** video grids. `prompt_ids` is the
    /// tokenized prompt (one `image_token_id` per image; one `video_token_id` per frame from the
    /// Text–Timestamp-Alignment placeholders). `messages` is the *original* (un-substituted)
    /// conversation, walked to recover the visual order.
    /// Build Gemma 4's multimodal prefill: encode every image and audio clip in **document order**,
    /// expand the rendered `<|image|>` / `<|audio|>` markers into their framed soft-token spans, and
    /// splice the feature rows onto the matching placeholder positions.
    ///
    /// Simpler than the Qwen-VL path in every dimension that matters here: no ViT, no M-RoPE, no
    /// DeepStack. The spans are ordinary positions in a causal 1-D sequence, so the continuation
    /// needs no position shift.
    fn prepare_gemma4(
        &self,
        prompt_ids: &[i32],
        messages: &[Message],
    ) -> CoreResult<Gemma4Prefill> {
        let rt = self.gemma4.as_ref().ok_or_else(|| {
            CoreError::Load("gemma 4: provider has no multimodal front-end".into())
        })?;
        let cfg = &rt.mm.cfg;

        // Images: one feature block and one soft-token count per image, in document order.
        let images = collect_images(messages);
        let mut image_feats: Vec<Tensor> = Vec::with_capacity(images.len());
        let mut image_counts: Vec<usize> = Vec::with_capacity(images.len());
        if !images.is_empty() {
            let tower = rt.mm.vision.as_ref().ok_or_else(|| {
                CoreError::Unsupported(
                    "[candle-llama] Gemma 4: this checkpoint ships no vision embedder, so it \
                     cannot be conditioned on an image"
                        .to_string(),
                )
            })?;
            let vcfg = tower.config().clone();
            for img in &images {
                let (gh, gw) = gemma4_mm::soft_token_grid(
                    img.width as usize,
                    img.height as usize,
                    vcfg.max_soft_tokens,
                );
                let (tw, th) = (gw * vcfg.patch_pixels, gh * vcfg.patch_pixels);
                // Resize to an exact patch multiple through the PIL-matching bicubic path, then
                // patchify. `resample: 3` in the shipped processor config is PIL BICUBIC.
                // NOTE the argument order: `resize_bicubic_u8` takes HEIGHT before WIDTH, and
                // returns f32 samples still on the 0..255 scale (the rescale to 0..1 happens in
                // `patchify`). Passing width first would transpose every non-square image.
                let resized = crate::image::resize_bicubic_u8(
                    &img.pixels,
                    img.height as usize,
                    img.width as usize,
                    th,
                    tw,
                )
                .map_err(to_core)?;
                let flat =
                    gemma4_mm::patchify(&resized, tw, th, vcfg.patch_pixels).map_err(to_core)?;
                let patches =
                    gemma4_mm::patch_tensor(&flat, vcfg.patch_elems(), &rt.device, rt.dtype)
                        .map_err(to_core)?;
                let feats = tower.forward(&patches, (gh, gw)).map_err(to_core)?;
                image_counts.push(gh * gw);
                image_feats.push(feats);
            }
        }

        // Audio: one feature block and one soft-token count per clip, in document order.
        let clips = collect_audio(messages);
        let mut audio_feats: Vec<Tensor> = Vec::with_capacity(clips.len());
        let mut audio_counts: Vec<usize> = Vec::with_capacity(clips.len());
        if !clips.is_empty() {
            let proj = rt.mm.audio.as_ref().ok_or_else(|| {
                CoreError::Unsupported(
                    "[candle-llama] Gemma 4: this checkpoint ships no audio projector, so it \
                     cannot be conditioned on audio"
                        .to_string(),
                )
            })?;
            let acfg = proj.config().clone();
            for clip in &clips {
                // The projector's framing is defined in samples at a fixed rate; resampling behind
                // the caller's back would silently change what the model hears.
                if clip.sample_rate != acfg.sample_rate {
                    return Err(CoreError::InvalidRequest(format!(
                        "[candle-llama] Gemma 4 audio expects {} Hz mono PCM, got {} Hz; resample \
                         before sending",
                        acfg.sample_rate, clip.sample_rate
                    )));
                }
                let framed = gemma4_mm::audio_frames(
                    &clip.samples,
                    acfg.samples_per_token,
                    acfg.max_soft_tokens,
                )
                .map_err(to_core)?;
                let frames =
                    gemma4_mm::frame_tensor(&framed, acfg.samples_per_token, &rt.device, rt.dtype)
                        .map_err(to_core)?;
                let feats = proj.forward(&frames).map_err(to_core)?;
                audio_counts.push(framed.len() / acfg.samples_per_token);
                audio_feats.push(feats);
            }
        }

        // Expand each marker into `begin` + count soft tokens + `end`. Images first, then audio;
        // each pass touches only its own marker id, so interleaved order is preserved.
        let expanded = gemma4_mm::expand_framed_placeholders(
            prompt_ids,
            cfg.image_token_id,
            cfg.boi_token_id,
            cfg.eoi_token_id,
            &image_counts,
        )
        .map_err(to_core)?;
        let expanded = gemma4_mm::expand_framed_placeholders(
            &expanded,
            cfg.audio_token_id,
            cfg.boa_token_id,
            cfg.eoa_token_id,
            &audio_counts,
        )
        .map_err(to_core)?;

        // Splice. Each modality is placed on its own token id, so the two calls cannot collide.
        let model = match &self.model {
            Decoder::Causal(m) => m,
            Decoder::Qwen35(_) => {
                return Err(CoreError::Load(
                    "gemma 4: the multimodal path requires the generic causal decoder".into(),
                ))
            }
        };
        let mut embeds = model
            .embed_input_ids(&input_ids(&expanded, &rt.device).map_err(to_core)?)
            .map_err(to_core)?;
        if !image_feats.is_empty() {
            let all = gemma4_mm::concat_features(&image_feats).map_err(to_core)?;
            embeds = model
                .splice_vision_features(&embeds, &expanded, &all, &[cfg.image_token_id])
                .map_err(to_core)?;
        }
        if !audio_feats.is_empty() {
            let all = gemma4_mm::concat_features(&audio_feats).map_err(to_core)?;
            embeds = model
                .splice_vision_features(&embeds, &expanded, &all, &[cfg.audio_token_id])
                .map_err(to_core)?;
        }

        Ok(Gemma4Prefill {
            expanded_ids: expanded,
            embeds,
        })
    }

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
                    // Rejected by `validate` long before here: this provider reports
                    // `supports_audio=false` for every Qwen-VL checkpoint. Refuse rather than skip,
                    // so a broken gate cannot answer an audio question from the text alone.
                    Content::Audio(_) => {
                        return Err(CoreError::Unsupported(
                            "[candle-llama] the Qwen-VL prefill carries no audio; an audio block \
                             reached it, which the capability gate should have rejected"
                                .to_string(),
                        ))
                    }
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
        let visual_pos_mask: Vec<bool> = expanded
            .iter()
            .map(|&id| id == img_id || id == vid_id)
            .collect();

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

/// Conservative load bounds for the two independent allocation domains.
///
/// Candle reads one safetensors shard into a host `Vec`, copies each tensor directly to CUDA at its
/// stored dtype, and drops the shard buffer before reading the next shard. The dense Qwen3.8
/// checkpoint is BF16 and `Tensor::to_dtype(BF16)` shares storage, so CUDA construction does not
/// retain a second expanded language-weight copy. The largest shard is therefore the host staging
/// bound; 25 percent device headroom covers constructor casts such as the BF16-to-F32 vision tower
/// while source tensors are still alive. CPU dense conversion retains source tensors plus
/// constructed F32 tensors. Packed CPU paths keep their existing two-copy upper bound, and external
/// projectors retain the audited four-copy conversion allowance.
///
/// An NVFP4 load (sc-24135) quantizes on the device while the bf16 source tensors are still
/// resident, so the device additionally holds the growing packed copy: 4.5 bits per 16-bit source
/// element, i.e. `payload · 9/32`. The existing 25 percent headroom still covers the transient f32
/// copy of the largest tensor the quantizer reads.
fn load_memory_requirements(
    payload: u64,
    staging: u64,
    projector: u64,
    packed: bool,
    cuda: bool,
    nvfp4: bool,
) -> Option<(u64, Option<u64>)> {
    let projector_bound = projector.checked_mul(4)?;
    let host = if cuda && !packed {
        staging.checked_add(projector_bound)?
    } else if packed {
        payload.checked_mul(2)?.checked_add(projector_bound)?
    } else {
        payload.checked_mul(3)?.checked_add(projector_bound)?
    };
    let device = if cuda {
        let nvfp4_packed = if nvfp4 {
            payload.checked_mul(9)? / 32
        } else {
            0
        };
        Some(
            payload
                .checked_add(payload / 4)?
                .checked_add(nvfp4_packed)?
                .checked_add(projector_bound)?,
        )
    } else {
        None
    };
    Some((host, device))
}

/// The process-wide override caps the execution-memory domain. On a discrete CUDA device, host
/// source staging has its own measured capacity and must not be capped by a VRAM snapshot.
fn host_load_budget(cuda: bool, budget: Option<u64>) -> Option<u64> {
    (!cuda).then_some(budget).flatten()
}

/// Combine CUDA driver-free bytes with idle bytes retained by cudarc's asynchronous allocator.
///
/// `cuMemGetInfo` excludes memory reserved by the current CUDA memory pool, even when part of that
/// reservation is unused and immediately reusable by this process. Admission must include that
/// idle reservation without counting live pool allocations a second time. Invalid or internally
/// inconsistent snapshots fail closed.
#[cfg(any(feature = "cuda", test))]
fn cuda_usable_memory_bytes(
    driver_free: u64,
    device_total: u64,
    pool_usage: Option<(u64, u64)>,
) -> Option<u64> {
    if driver_free > device_total {
        return None;
    }
    let Some((reserved, used)) = pool_usage else {
        return Some(driver_free);
    };
    if used > reserved || reserved > device_total {
        return None;
    }
    let free_plus_reserved = driver_free.checked_add(reserved)?;
    if free_plus_reserved > device_total {
        return None;
    }
    free_plus_reserved.checked_sub(used)
}

#[cfg(feature = "cuda")]
fn cuda_pool_usage(
    context: &candle_core::cuda_backend::cudarc::driver::CudaContext,
) -> Option<(u64, u64)> {
    use candle_core::cuda_backend::cudarc::driver::{result, sys};

    context.bind_to_thread().ok()?;
    // SAFETY: the live context is bound to this thread, `cu_device` remains owned by it,
    // `get_mem_pool` returns that device's live current pool, and both output pointers have the u64
    // type required by these two CUDA attributes.
    unsafe {
        let pool = result::device::get_mem_pool(context.cu_device()).ok()?;
        let mut reserved = 0_u64;
        result::mem_pool::get_attribute(
            pool,
            sys::CUmemPool_attribute::CU_MEMPOOL_ATTR_RESERVED_MEM_CURRENT,
            (&mut reserved as *mut u64).cast(),
        )
        .ok()?;
        let mut used = 0_u64;
        result::mem_pool::get_attribute(
            pool,
            sys::CUmemPool_attribute::CU_MEMPOOL_ATTR_USED_MEM_CURRENT,
            (&mut used as *mut u64).cast(),
        )
        .ok()?;
        Some((reserved, used))
    }
}

fn request_available_memory(device: &Device) -> CoreResult<u64> {
    let capacity = if device.is_cuda() {
        #[cfg(feature = "cuda")]
        {
            // Candle frees asynchronous allocations on this device's stream. Complete those frees
            // before sampling the pool so admission observes the reusable post-load/request state.
            device.synchronize().map_err(|e| to_core(e.into()))?;
            device.as_cuda_device().ok().and_then(|d| {
                let stream = d.cuda_stream();
                let context = stream.context();
                let (free, total) = context.mem_get_info().ok()?;
                let pool_usage = if context.has_async_alloc() {
                    Some(cuda_pool_usage(context)?)
                } else {
                    None
                };
                cuda_usable_memory_bytes(free as u64, total as u64, pool_usage)
            })
        }
        #[cfg(not(feature = "cuda"))]
        {
            None
        }
    } else {
        core_llm::available_host_memory_bytes()
    };
    core_llm::effective_memory_budget(capacity, core_llm::operational_memory_override()?)
}

/// Adapts a `core_llm::JsonConstraint` to the engine's [`ConstraintMask`] decode seam.
struct JsonMask<'a> {
    inner: JsonConstraint<'a>,
    table: &'a ConstraintDecodeTable,
    stop_ids: Vec<u32>,
    accepted: Vec<i32>,
    initial_reasoning: bool,
    reasoning: bool,
    reasoning_tail: String,
    allow: Vec<bool>,
}

impl<'a> JsonMask<'a> {
    fn new(
        table: &'a ConstraintDecodeTable,
        stop_ids: impl IntoIterator<Item = u32>,
        reasoning: bool,
    ) -> Self {
        let stop_ids = stop_ids.into_iter().collect::<Vec<_>>();
        Self {
            inner: JsonConstraint::new(table, stop_ids.iter().copied()),
            table,
            stop_ids,
            accepted: Vec::new(),
            initial_reasoning: reasoning,
            reasoning,
            reasoning_tail: String::new(),
            allow: vec![false; table.pieces.len()],
        }
    }

    fn accept_reasoning_piece(&mut self, piece: &str) {
        self.reasoning_tail.push_str(piece);
        if let Some(end) = self.reasoning_tail.find("</think>") {
            let answer = self.reasoning_tail[end + "</think>".len()..].to_string();
            self.reasoning = false;
            self.reasoning_tail.clear();
            let accepted = self.inner.accept_text(&answer);
            debug_assert!(accepted);
            return;
        }
        let keep = (1.."</think>".len())
            .rev()
            .find(|&n| self.reasoning_tail.ends_with(&"</think>"[..n]))
            .unwrap_or(0);
        if self.reasoning_tail.len() > keep {
            self.reasoning_tail
                .drain(..self.reasoning_tail.len() - keep);
        }
    }
}

impl ConstraintMask for JsonMask<'_> {
    fn allowed(&mut self) -> &[bool] {
        if !self.reasoning {
            return self.inner.allowed();
        }
        for (id, slot) in self.allow.iter_mut().enumerate() {
            if self.stop_ids.contains(&(id as u32)) {
                *slot = false;
                continue;
            }
            let mut candidate = self.reasoning_tail.clone();
            candidate.push_str(&self.table.pieces[id]);
            *slot = candidate
                .find("</think>")
                .is_none_or(|end| self.inner.allows_text(&candidate[end + "</think>".len()..]));
        }
        &self.allow
    }
    fn accept(&mut self, token: i32) {
        if self.reasoning {
            let piece = self
                .table
                .pieces
                .get(token as usize)
                .cloned()
                .unwrap_or_default();
            self.accept_reasoning_piece(&piece);
        } else {
            self.inner.accept(token as u32);
        }
        self.accepted.push(token);
    }
}

impl RewindableConstraintMask for JsonMask<'_> {
    fn checkpoint(&self) -> usize {
        self.accepted.len()
    }

    fn rewind(&mut self, checkpoint: usize) {
        self.accepted.truncate(checkpoint);
        self.inner = JsonConstraint::new(self.table, self.stop_ids.iter().copied());
        self.reasoning_tail.clear();
        self.reasoning = self.initial_reasoning;
        // Replay committed tokens to restore both the reasoning phase and JSON grammar.
        let accepted = self.accepted.clone();
        for token in accepted {
            self.accept(token);
        }
        self.accepted.truncate(checkpoint);
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
fn load_chat_template(dir: &Path) -> (Box<dyn ChatTemplate>, bool, bool, bool, bool) {
    // The sidecar `chat_template.jinja` wins over the embedded key — see `sidecar_chat_template`.
    if let Some(t) = sidecar_chat_template(dir) {
        let supports_thinking = t.source().contains("enable_thinking");
        let supports_tools = t.source().contains("tool_call");
        let supports_reasoning_effort = t.source().contains("reasoning_effort");
        let supports_preserve_thinking = t.source().contains("preserve_thinking");
        return (
            Box::new(t),
            supports_thinking,
            supports_reasoning_effort,
            supports_preserve_thinking,
            supports_tools,
        );
    }
    match JinjaChatTemplate::from_tokenizer_config_file(dir.join("tokenizer_config.json")) {
        Ok(t) => {
            let supports_thinking = t.source().contains("enable_thinking");
            let supports_tools = t.source().contains("tool_call");
            let supports_reasoning_effort = t.source().contains("reasoning_effort");
            let supports_preserve_thinking = t.source().contains("preserve_thinking");
            (
                Box::new(t),
                supports_thinking,
                supports_reasoning_effort,
                supports_preserve_thinking,
                supports_tools,
            )
        }
        Err(_) => (Box::new(Llama3Template), false, false, false, false),
    }
}

/// The modern HF layout ships the chat template as a **separate `chat_template.jinja`** beside
/// `tokenizer_config.json` rather than inside it (transformers writes it that way for newer
/// releases — `google/gemma-4-12B-it` is one, and its `tokenizer_config.json` carries no
/// `chat_template` key at all).
///
/// Reading only the embedded key means such a snapshot silently falls back to the typed Llama-3
/// default: the prompt still renders, the model still generates, and every assertion about shapes
/// and lengths still passes — it just answers badly, because it is being addressed in a chat format
/// it was never trained on. Prefer the sidecar file, then the embedded key, then the default.
///
/// BOS/EOS still come from `tokenizer_config.json`; the sidecar carries only the template body.
fn sidecar_chat_template(dir: &Path) -> Option<JinjaChatTemplate> {
    let source = std::fs::read_to_string(dir.join("chat_template.jinja")).ok()?;
    if source.trim().is_empty() {
        return None;
    }
    let (bos, eos) = tokenizer_special_tokens(dir);
    Some(JinjaChatTemplate::with_tokens(source, bos, eos))
}

/// `(bos_token, eos_token)` strings from `tokenizer_config.json`, empty when absent. Each may be a
/// bare string or an `AddedToken` object carrying `content`.
fn tokenizer_special_tokens(dir: &Path) -> (String, String) {
    let Some(v) = read_json(dir, "tokenizer_config.json") else {
        return (String::new(), String::new());
    };
    let token = |key: &str| -> String {
        match v.get(key) {
            Some(serde_json::Value::String(s)) => s.clone(),
            Some(serde_json::Value::Object(o)) => o
                .get("content")
                .and_then(|c| c.as_str())
                .unwrap_or_default()
                .to_string(),
            _ => String::new(),
        }
    };
    (token("bos_token"), token("eos_token"))
}

/// Pick a chat template for a GGUF load: a sibling `tokenizer_config.json` first, then the GGUF's
/// own embedded `chat_template` metadata, then the typed Llama-3 default. Also reports
/// `supports_thinking` (the chosen template's source gates `enable_thinking`) and `supports_tools`
/// (its source renders tool calls — it mentions `tool_call`).
fn gguf_chat_template(
    dir: &Path,
    ck: &GgufCheckpoint,
) -> (Box<dyn ChatTemplate>, bool, bool, bool, bool) {
    if let Ok(t) = JinjaChatTemplate::from_tokenizer_config_file(dir.join("tokenizer_config.json"))
    {
        let supports_thinking = t.source().contains("enable_thinking");
        let supports_tools = t.source().contains("tool_call");
        let supports_reasoning_effort = t.source().contains("reasoning_effort");
        let supports_preserve_thinking = t.source().contains("preserve_thinking");
        return (
            Box::new(t),
            supports_thinking,
            supports_reasoning_effort,
            supports_preserve_thinking,
            supports_tools,
        );
    }
    if let Some(src) = &ck.chat_template {
        let supports_thinking = src.contains("enable_thinking");
        let supports_tools = src.contains("tool_call");
        let bos = ck.bos_token.clone().unwrap_or_default();
        let eos = ck.eos_token.clone().unwrap_or_default();
        return (
            Box::new(JinjaChatTemplate::with_tokens(src.clone(), bos, eos)),
            supports_thinking,
            src.contains("reasoning_effort"),
            src.contains("preserve_thinking"),
            supports_tools,
        );
    }
    (Box::new(Llama3Template), false, false, false, false)
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
    piece: &str,
    id: u32,
    stop: (&mut StopMatcher, &std::cell::Cell<bool>),
    streamed: &mut String,
    emit_index: &mut usize,
    last_id: &mut u32,
    on_event: &mut dyn FnMut(CoreEvent),
) {
    let (stop_matcher, halt) = stop;
    if halt.get() {
        return;
    }
    let chunk = stop_matcher.push(piece);
    if !chunk.emit.is_empty() {
        streamed.push_str(&chunk.emit);
        *last_id = id;
        on_event(CoreEvent::Token {
            id,
            text: chunk.emit,
            index: *emit_index,
            channel: Channel::Content,
        });
        *emit_index += 1;
    }
    if chunk.stop {
        halt.set(true);
    }
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
        // Request start: a request that fails or is cancelled (anywhere below, before the record
        // is written on success) must not leave the previous request's record readable.
        *self
            .last_decode
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner()) = None;
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
        // Gemma 4 multimodal (sc-18772): its own marker substitution and prefill, taken whenever
        // this is a Gemma 4 checkpoint with front-ends AND the request actually carries a visual or
        // a clip. `validate` has already rejected a modality this checkpoint does not ship.
        let gemma4_mm_request = self.gemma4.is_some()
            && (!collect_images(&req.messages).is_empty()
                || !collect_audio(&req.messages).is_empty());
        let substituted;
        let messages: &[Message] = if gemma4_mm_request {
            substituted = substitute_gemma4_placeholders(&req.messages)?;
            &substituted
        } else if multimodal {
            substituted = substitute_vision_placeholders(&req.messages, temporal_patch)?;
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
                reasoning_effort: req.reasoning_effort,
                preserve_thinking: req.preserve_thinking,
                tools: &req.tools,
            },
        )?;
        let prompt_ids: Vec<i32> = self
            .tokenizer
            .encode(&prompt, false)?
            .into_iter()
            .map(|id| id as i32)
            .collect();

        // Price the request before image/video preprocessing or any model forward allocates native
        // tensors. The visual geometry path is pure integer math over request dimensions.
        let (visual_tokens, vision_workspace) = match &self.vision {
            Some(vision) if multimodal => vision.estimate_workspace(&images, &videos)?,
            _ => (0, 0),
        };
        let vision_workspace = vision_workspace
            .checked_add(request_media_workspace_bytes(&req.messages)?)
            .ok_or_else(|| CoreError::InvalidRequest("media workspace overflow".into()))?;
        let replaced_visual_placeholders =
            self.vision
                .as_ref()
                .filter(|_| multimodal)
                .map_or(0, |vision| {
                    prompt_ids
                        .iter()
                        .filter(|&&id| id == vision.image_token_id || id == vision.video_token_id)
                        .count()
                });
        let admitted_prompt = prompt_ids
            .len()
            .checked_sub(replaced_visual_placeholders)
            .and_then(|n| n.checked_add(visual_tokens))
            .ok_or_else(|| CoreError::InvalidRequest("expanded prompt geometry overflow".into()))?;
        // Pure prompt/visual geometry is known before native preprocessing. Architectural context
        // overflow must win over a transient capacity failure on the current host or CUDA device.
        validate_context_window(
            self.descriptor.capabilities.max_context_tokens,
            admitted_prompt,
            req.max_new_tokens,
        )?;
        // Every portable eager-attention mask variant is query-tiled; CUDA flash attention is
        // bounded more tightly. Use the runtime's exact maximum tile so admission prices the same
        // peak score/mask/weight lifetime the implementation enforces.
        let required = core_llm::estimate_chunked_request_bytes(
            admitted_prompt,
            req.max_new_tokens,
            self.model.memory_geometry(),
            vision_workspace,
            match req.mtp {
                MtpMode::Off => 0,
                MtpMode::Auto => self
                    .descriptor
                    .capabilities
                    .mtp
                    .map_or(0, |c| c.recommended_draft_tokens),
                MtpMode::Enabled { draft_tokens } => draft_tokens,
            },
            EAGER_ATTN_QUERY_CHUNK_SIZE,
        )
        .ok_or_else(|| CoreError::InvalidRequest("request memory estimate overflow".into()))?;
        let available = request_available_memory(self.model.device())?;
        core_llm::admit_request_memory_with_geometry(
            admitted_prompt,
            req.max_new_tokens,
            self.descriptor.capabilities.max_context_tokens,
            required,
            available,
        )?;

        self.model
            .device()
            .synchronize()
            .map_err(|e| to_core(e.into()))?;
        let generation_started = std::time::Instant::now();
        let request_span = RequestSpan::begin();
        // The reference loop's forwards are measured, not inferred (sc-24129).
        let counted = CountingDecode::new(&self.model);
        let mut extra_forwards = 0u64;

        // Encode + splice the visuals and compute M-RoPE positions (the placeholder-expanded prompt
        // becomes the effective sequence). `None` on the text-only path.
        let mm = if multimodal && !gemma4_mm_request {
            Some(self.prepare_multimodal(&prompt_ids, &req.messages)?)
        } else {
            None
        };
        let g4 = if gemma4_mm_request {
            Some(self.prepare_gemma4(&prompt_ids, &req.messages)?)
        } else {
            None
        };
        let prompt_len = mm
            .as_ref()
            .map(|m| m.expanded_ids.len())
            .or_else(|| g4.as_ref().map(|m| m.expanded_ids.len()))
            .unwrap_or(prompt_ids.len());
        validate_context_window(
            self.descriptor.capabilities.max_context_tokens,
            prompt_len,
            req.max_new_tokens,
        )?;

        let config = GenerationConfig {
            max_new_tokens: req.max_new_tokens as usize,
            sampling: map_sampling(&req.sampling),
            seed: req.seed,
            stop_tokens: self.stop_tokens.clone(),
        };
        let mtp_drafts = match req.mtp {
            MtpMode::Off => None,
            MtpMode::Auto => self
                .descriptor
                .capabilities
                .mtp
                .map(|cap| cap.recommended_draft_tokens),
            MtpMode::Enabled { draft_tokens } => Some(draft_tokens),
        };

        // Structured-output constraint: build a JSON mask over the cached decode table.
        let constraint_starts_in_reasoning =
            self.descriptor.capabilities.supports_thinking && prompt_opens_thinking(&prompt);
        let mut json_mask = match req.constraint {
            Some(Constraint::Json) => {
                let table = self
                    .constraint_table
                    .get_or_init(|| self.tokenizer.constraint_decode_table());
                Some(JsonMask::new(
                    table,
                    self.stop_tokens.iter().map(|&i| i as u32),
                    constraint_starts_in_reasoning,
                ))
            }
            None => None,
        };

        let mut stop_matcher = StopMatcher::new(req.stop.iter().cloned());
        let stop_active = !stop_matcher.is_empty();
        let halt = std::cell::Cell::new(false);

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
            if constraint_starts_in_reasoning {
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
        // `IncrementalDetok` guard holds back lossy U+FFFD placeholders so a multi-byte character
        // split across BPE tokens streams intact (and never panics a mid-char slice) — sc-12452.
        // The segmenter (when active) splits each delta into reasoning vs answer.
        let tokenizer = &self.tokenizer;
        let mut mtp_stats = None;
        let phase_prefill;
        let phase_decode_started;
        let out = {
            let mut acc: Vec<u32> = Vec::new();
            let mut detok = IncrementalDetok::new();
            let mut sink = |ev: StreamEvent| {
                if let StreamEvent::Token { id, step } = ev {
                    let id = id as u32;
                    acc.push(id);
                    if let Ok(text) = tokenizer.decode(&acc, true) {
                        if let Some(delta) = detok.push(&text) {
                            let delta = delta.to_string();
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
                                                        &piece,
                                                        id,
                                                        (&mut stop_matcher, &halt),
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
                                None if tool_seg.is_some() || stop_active => {
                                    // No reasoning split, but tools are active: route the whole delta
                                    // through the tool segmenter (the answer is the only channel).
                                    for piece in tool_pieces(&mut tool_seg, &delta) {
                                        emit_content(
                                            &piece,
                                            id,
                                            (&mut stop_matcher, &halt),
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
            let should_stop = || halt.get();
            let should_stop_opt = stop_active.then_some(&should_stop as &dyn Fn() -> bool);
            if let Some(draft_tokens) = mtp_drafts {
                let target = match &self.model {
                    Decoder::Qwen35(model) => model,
                    Decoder::Causal(_) => {
                        return Err(CoreError::Load(
                            "MTP was advertised for a non-Qwen target decoder".into(),
                        ))
                    }
                };
                let mtp = self.mtp.as_ref().ok_or_else(|| {
                    CoreError::Load("MTP was advertised without a loaded predictor".into())
                })?;
                let constraint = json_mask
                    .as_mut()
                    .map(|m| m as &mut dyn RewindableConstraintMask);
                let (generated, stats) = match &mm {
                    Some(m) => {
                        let mut prefill = None;
                        let mut decode_started = None;
                        let mut boundary = || -> crate::error::Result<()> {
                            target.device().synchronize()?;
                            prefill = Some(generation_started.elapsed());
                            decode_started = Some(std::time::Instant::now());
                            Ok(())
                        };
                        let (t, h, w, delta) = &m.positions;
                        let result = generate_qwen35_mtp_multimodal_with_stop(
                            target,
                            mtp,
                            Qwen35MtpMultimodalPrompt {
                                input_ids: &m.expanded_ids,
                                embeddings: &m.embeds,
                                positions: [t.as_slice(), h.as_slice(), w.as_slice()],
                                visual_pos_mask: &m.visual_pos_mask,
                                deepstack: &m.deepstack,
                                continuation_delta: *delta,
                            },
                            &config,
                            draft_tokens,
                            &req.cancel,
                            &mut sink,
                            constraint,
                            should_stop_opt,
                            Some(&mut boundary),
                        )
                        .map_err(to_core)?;
                        target
                            .device()
                            .synchronize()
                            .map_err(|e| to_core(e.into()))?;
                        phase_prefill = prefill.unwrap_or_else(|| generation_started.elapsed());
                        phase_decode_started =
                            decode_started.unwrap_or_else(std::time::Instant::now);
                        result
                    }
                    None => {
                        target
                            .device()
                            .synchronize()
                            .map_err(|e| to_core(e.into()))?;
                        let prefill_started = generation_started;
                        let mut prefill = None;
                        let mut decode_started = None;
                        let mut boundary = || -> crate::error::Result<()> {
                            target.device().synchronize()?;
                            prefill = Some(prefill_started.elapsed());
                            decode_started = Some(std::time::Instant::now());
                            Ok(())
                        };
                        let result = generate_qwen35_mtp_timed_with_stop(
                            target,
                            mtp,
                            &prompt_ids,
                            &config,
                            draft_tokens,
                            &req.cancel,
                            &mut sink,
                            constraint,
                            should_stop_opt,
                            &mut boundary,
                        )
                        .map_err(to_core)?;
                        target
                            .device()
                            .synchronize()
                            .map_err(|e| to_core(e.into()))?;
                        phase_prefill = prefill.unwrap_or_else(|| generation_started.elapsed());
                        phase_decode_started =
                            decode_started.unwrap_or_else(std::time::Instant::now);
                        result
                    }
                };
                mtp_stats = Some(MtpStats {
                    proposed_tokens: u32::try_from(stats.proposed).unwrap_or(u32::MAX),
                    accepted_tokens: u32::try_from(stats.accepted).unwrap_or(u32::MAX),
                    target_forwards: u32::try_from(stats.forwards).unwrap_or(u32::MAX),
                });
                generated
            } else {
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
                        model
                            .device()
                            .synchronize()
                            .map_err(|e| to_core(e.into()))?;
                        let prefill = generation_started.elapsed();
                        let decode_started = std::time::Instant::now();
                        let shifted = Shifted {
                            model,
                            delta: *delta,
                        };
                        let shifted = CountingDecode::new(&shifted);
                        shifted.note_external_forward(); // the DeepStack prefill above
                        let result = generate_from_prefill_with_stop(
                            &shifted,
                            &mut *cache,
                            first,
                            m.expanded_ids.clone(),
                            &config,
                            &req.cancel,
                            &mut sink,
                            constraint,
                            should_stop_opt,
                        )
                        .map_err(to_core)?;
                        extra_forwards = shifted.forwards();
                        model
                            .device()
                            .synchronize()
                            .map_err(|e| to_core(e.into()))?;
                        phase_prefill = prefill;
                        phase_decode_started = decode_started;
                        result
                    }
                    // Gemma 4 multimodal: prefill the spliced embeds on ordinary causal 1-D positions
                    // (no M-RoPE, so no position shift for the continuation), then decode through the
                    // shared loop against the unwrapped decoder.
                    None => match &g4 {
                        Some(m) => {
                            let model =
                                match &self.model {
                                    Decoder::Causal(c) => c,
                                    Decoder::Qwen35(_) => return Err(CoreError::Load(
                                        "gemma 4: the multimodal path requires the generic causal \
                                     decoder"
                                            .into(),
                                    )),
                                };
                            let mut cache = model.make_cache();
                            let first = model
                                .decode_logits_from_embeds(&m.embeds, &mut *cache, 0)
                                .map_err(to_core)?;
                            counted.note_external_forward();
                            self.model
                                .device()
                                .synchronize()
                                .map_err(|e| to_core(e.into()))?;
                            let prefill = generation_started.elapsed();
                            let decode_started = std::time::Instant::now();
                            let result = generate_from_prefill_with_stop(
                                &counted,
                                &mut *cache,
                                first,
                                m.expanded_ids.clone(),
                                &config,
                                &req.cancel,
                                &mut sink,
                                constraint,
                                should_stop_opt,
                            )
                            .map_err(to_core)?;
                            self.model
                                .device()
                                .synchronize()
                                .map_err(|e| to_core(e.into()))?;
                            phase_prefill = prefill;
                            phase_decode_started = decode_started;
                            result
                        }
                        None => match &self.model {
                            Decoder::Qwen35(_) => {
                                self.model
                                    .device()
                                    .synchronize()
                                    .map_err(|e| to_core(e.into()))?;
                                let mut cache = self.model.make_cache();
                                let ids =
                                    input_ids(&prompt_ids, self.model.device()).map_err(to_core)?;
                                let first = counted.step(&ids, &mut *cache, 0).map_err(to_core)?;
                                self.model
                                    .device()
                                    .synchronize()
                                    .map_err(|e| to_core(e.into()))?;
                                let prefill = generation_started.elapsed();
                                let decode_started = std::time::Instant::now();
                                let result = generate_from_prefill_with_stop(
                                    &counted,
                                    &mut *cache,
                                    first,
                                    prompt_ids.clone(),
                                    &config,
                                    &req.cancel,
                                    &mut sink,
                                    constraint,
                                    should_stop_opt,
                                )
                                .map_err(to_core)?;
                                self.model
                                    .device()
                                    .synchronize()
                                    .map_err(|e| to_core(e.into()))?;
                                phase_prefill = prefill;
                                phase_decode_started = decode_started;
                                result
                            }
                            Decoder::Causal(_) => {
                                let mut cache = self.model.make_cache();
                                let ids =
                                    input_ids(&prompt_ids, self.model.device()).map_err(to_core)?;
                                let first = counted.step(&ids, &mut *cache, 0).map_err(to_core)?;
                                self.model
                                    .device()
                                    .synchronize()
                                    .map_err(|e| to_core(e.into()))?;
                                let prefill = generation_started.elapsed();
                                let decode_started = std::time::Instant::now();
                                let result = generate_from_prefill_with_stop(
                                    &counted,
                                    &mut *cache,
                                    first,
                                    prompt_ids.clone(),
                                    &config,
                                    &req.cancel,
                                    &mut sink,
                                    constraint,
                                    should_stop_opt,
                                )
                                .map_err(to_core)?;
                                self.model
                                    .device()
                                    .synchronize()
                                    .map_err(|e| to_core(e.into()))?;
                                phase_prefill = prefill;
                                phase_decode_started = decode_started;
                                result
                            }
                        },
                    },
                }
            }
        };

        let decode_record = match (mtp_stats, mtp_drafts) {
            (Some(stats), Some(drafts)) => DecodeRecord {
                path: DecodePath::Mtp { drafts },
                target_forwards: u64::from(stats.target_forwards),
                proposed_tokens: u64::from(stats.proposed_tokens),
                accepted_tokens: u64::from(stats.accepted_tokens),
                generated_tokens: out.tokens.len() as u64,
                host_syncs: request_span.host_syncs(),
                kv_cache: crate::primitives::KvCacheKind::Growing,
                attn_formulation: self.model.attn_formulation(),
                fused_primitives: request_span.fused_primitives(),
            },
            _ => DecodeRecord::plain(
                DecodePath::Reference,
                counted.forwards() + extra_forwards,
                out.tokens.len(),
                request_span.host_syncs(),
            )
            .with_attn_formulation(self.model.attn_formulation())
            .with_fused_primitives(request_span.fused_primitives()),
        };
        *self
            .last_decode
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner()) = Some(decode_record);

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
                                &piece,
                                last_id,
                                (&mut stop_matcher, &halt),
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
                    &piece,
                    last_id,
                    (&mut stop_matcher, &halt),
                    &mut streamed,
                    &mut emit_index,
                    &mut last_id,
                    &mut *on_event,
                );
            }
        }

        if stop_active && !halt.get() {
            let tail = stop_matcher.flush();
            if !tail.is_empty() {
                streamed.push_str(&tail);
                on_event(CoreEvent::Token {
                    id: last_id,
                    text: tail,
                    index: emit_index,
                    channel: Channel::Content,
                });
            }
        }

        // Result text: the streamed answer when thinking or tools are active (either means the
        // streamed channel is the authoritative answer, with reasoning / tool-call markup removed);
        // otherwise the original decode-all-tokens path (byte-identical to the no-feature case).
        // `streamed` accumulates only `IncrementalDetok`-released deltas, so it carries no
        // transient U+FFFD placeholders; a character truncated by end-of-generation is dropped
        // rather than surfaced as U+FFFD (sc-12452).
        // Reasoning and tool calls, if the model produced any, are reported separately (their markup
        // excluded from `text`).
        let text = if stop_active || thinking_active || tools_active {
            streamed
        } else {
            let gen_u32: Vec<u32> = out.tokens.iter().map(|&i| i as u32).collect();
            tokenizer.decode(&gen_u32, true)?
        };
        let thinking = (!thinking_buf.is_empty()).then_some(thinking_buf);
        let tool_calls = tool_seg.map(|mut ts| ts.take_calls()).unwrap_or_default();
        let finish = if halt.get() {
            core_llm::FinishReason::Stop
        } else {
            map_finish(out.finish_reason)
        };
        let usage = Usage {
            prompt_tokens: prompt_len as u32,
            generated_tokens: out.tokens.len() as u32,
        };
        on_event(CoreEvent::Done {
            finish_reason: finish,
            usage,
        });
        Ok(TextLlmOutput {
            timings: Some(GenerationTimings {
                prefill: phase_prefill,
                decode: phase_decode_started.elapsed(),
            }),
            text,
            thinking,
            tool_calls,
            usage,
            mtp: mtp_stats,
            finish_reason: Some(finish),
        })
    }
}

fn validate_context_window(
    cap: usize,
    prompt_tokens: usize,
    max_new_tokens: u32,
) -> CoreResult<()> {
    if cap == 0 {
        return Ok(());
    }
    let total = prompt_tokens
        .checked_add(max_new_tokens as usize)
        .ok_or_else(|| CoreError::InvalidRequest("prompt + generation length overflow".into()))?;
    if total > cap {
        return Err(CoreError::InvalidRequest(format!(
            "expanded prompt ({prompt_tokens} tokens) + requested generation ({max_new_tokens}) exceeds context window {cap}"
        )));
    }
    Ok(())
}

/// The descriptor for the `candle-llama` provider (constructible without loading weights; used for
/// explicit catalog composition and inspection).
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
            // Weightless default: conservative. The load path flips this on for a Gemma 4
            // checkpoint that actually ships an audio projector (sc-18772).
            supports_audio: false,
            // No controllable reasoning mode yet (a separate story); the contract requires an
            // explicit enable-thinking request to be rejected, which validate_request enforces.
            supports_thinking: false,
            supports_reasoning_effort: false,
            reasoning_efforts: Vec::new(),
            model_sampling_defaults: None,
            supports_preserve_thinking: false,
            // Weightless default: conservative. The load path flips this on when the loaded model's
            // chat template renders tool calls (story 7636).
            supports_tools: false,
            mtp: None,
            // JSON-constrained decoding.
            supported_constraints: vec![ConstraintKind::Json],
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
        presence_penalty: s.presence_penalty,
        repetition_penalty: s.repetition_penalty,
        repetition_context: s.repetition_context,
    }
}

/// Official Bonsai source presets exposed as discovery metadata. The card also explicitly sets
/// `min_p = 0.0` in both modes; that disabled no-op needs no sampler field or runtime operator.
fn bonsai_sampling_defaults() -> ModelSamplingDefaults {
    ModelSamplingDefaults {
        thinking: Sampling {
            temperature: 1.0,
            top_p: 0.95,
            top_k: 20,
            presence_penalty: 0.0,
            repetition_penalty: 1.0,
            repetition_context: 0,
        },
        non_thinking: Sampling {
            temperature: 0.7,
            top_p: 0.8,
            top_k: 20,
            presence_penalty: 1.5,
            repetition_penalty: 1.0,
            repetition_context: 0,
        },
    }
}

fn map_finish(f: FinishReason) -> CoreFinish {
    match f {
        FinishReason::StopToken | FinishReason::Stopped => CoreFinish::Stop,
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
        // The typed capability refusal (sc-24135) is the contract's `Unsupported`, verbatim.
        crate::Error::Nvfp4Refused(r) => CoreError::Unsupported(r.to_string()),
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
    // Per-snapshot audio probe (sc-18772): Gemma 4 unified is the one architecture served here with
    // an audio path, and the same registration also serves text-only checkpoints, so the static
    // descriptor stays `supports_audio=false` and this probe carries the per-snapshot truth.
    weightless_audio: Some(can_load_audio),
};

fn load_registered(spec: &LoadSpec) -> CoreResult<Box<dyn TextLlm>> {
    Ok(Box::new(LlamaProvider::load(spec)?))
}

/// Weightless per-snapshot vision probe (core-llm `weightless_vision`): does this provider serve the
/// snapshot at `spec.source` *with* vision? True for a Qwen3.6 (`qwen3_5`) / Qwen3-VL (`qwen3_vl`)
/// HF checkpoint carrying a `vision_config` (the ViT tower loads alongside the decoder), and for a
/// Gemma 4 unified checkpoint carrying a `vision_config` + `image_token_id` (its encoder-free vision
/// embedder loads alongside the decoder, sc-18772). Reads only `config.json` — never a weight shard.
/// GGUF is declined (candle's GGUF path is dense text-only).
pub fn can_load_vision(spec: &LoadSpec) -> bool {
    if crate::gguf::is_gguf_path(&spec.source) {
        return crate::prism_checkpoint::PrismGgufCheckpoint::is_prism(Path::new(&spec.source))
            .unwrap_or(false)
            && spec.projector_source.as_deref().is_some_and(|path| {
                crate::prism_vision_gguf::PrismVisionGguf::is_qwen3vl_merger(Path::new(path))
            });
    }
    let Some(v) = probe_config(spec) else {
        return false;
    };
    if v.get("model_type").and_then(Value::as_str) == Some("prism_hadamard_qwen35") {
        return spec.projector_source.is_none()
            && v.get("vision_config").is_some()
            && v.pointer("/components/vision").and_then(Value::as_bool) == Some(true);
    }
    let qwen_vl = v.get("vision_config").is_some()
        && matches!(
            Architecture::from_config(&v),
            Ok(Architecture::Qwen35) | Ok(Architecture::Qwen3Vl)
        );
    // Gemma 4 is deliberately absent: its vision path is loaded but unvalidated and therefore
    // unadvertised (see `gemma4_vision_is_validated`). The probe must agree with the loaded
    // descriptor, or a model-first vision-required load would resolve here and then be rejected.
    qwen_vl
}

/// Weightless per-snapshot **audio** probe (core-llm `weightless_audio`, sc-18772): does this
/// provider serve the snapshot at `spec.source` *with* audio? Gemma 4 unified is the only
/// architecture served here that has an audio path; like vision it is per-snapshot, so the static
/// descriptor stays `supports_audio=false` and this probe is what lets a model-first audio-required
/// load resolve here. Reads only `config.json`; GGUF is declined (dense text-only).
pub fn can_load_audio(spec: &LoadSpec) -> bool {
    let Some(v) = probe_config(spec) else {
        return false;
    };
    gemma4_multimodal(&v, "audio_config", "audio_token_id")
}

/// Read and parse a snapshot's `config.json` for a weightless probe, or `None` when the source is a
/// GGUF path (no `config.json`, and candle's GGUF path is dense text-only) or is unreadable.
fn probe_config(spec: &LoadSpec) -> Option<serde_json::Value> {
    if crate::gguf::is_gguf_path(&spec.source) {
        return None;
    }
    let dir = Path::new(&spec.source);
    let path = if dir.is_dir() {
        dir.join("config.json")
    } else {
        dir.to_path_buf()
    };
    let text = std::fs::read_to_string(path).ok()?;
    serde_json::from_str::<serde_json::Value>(&text).ok()
}

/// Whether `v` is a Gemma 4 checkpoint declaring the named front-end block (`vision_config` /
/// `audio_config`) **and** the token id that front-end splices at.
///
/// The token-id check is not redundant with the block: a config carrying a `vision_config` but no
/// `image_token_id` has no row to splice image features into, so advertising vision for it would be
/// exactly the advertised-but-absent case. The load path refuses such a config for the same reason,
/// which keeps probe and loader agreeing.
fn gemma4_multimodal(v: &serde_json::Value, block: &str, token_key: &str) -> bool {
    if v.get(block).is_none() {
        return false;
    }
    if !matches!(Architecture::from_config(v), Ok(a) if a.is_gemma4()) {
        return false;
    }
    v.get(token_key).and_then(|x| x.as_i64()).is_some()
}

/// Weightless model-first probe (story 7406): can the `candle-llama` provider serve the model at
/// `spec.source`?
///
/// For a `*.gguf` file this reads **only** the GGUF header/metadata (never a tensor block, via
/// [`gguf_architecture`](crate::gguf::gguf_architecture)) and accepts it iff its `general.architecture`
/// is one the native GGUF loader can reconstruct ([`gguf_arch_to_hf`](crate::gguf::gguf_arch_to_hf):
/// `llama`/`mistral`, `qwen3`, and `gemma4`); an unsupported or non-LLM GGUF (`bert`, a `clip`
/// mmproj, …) is declined so `load_for_model` returns a clean `Unsupported` rather than routing it
/// here to fail at load (story 7420, replacing the earlier extension-only accept).
///
/// Otherwise this reads **only** `config.json` and runs the same [`Architecture::from_config`] dispatch
/// the loader uses (Llama / Mistral / Qwen2 / Qwen3 / Qwen2-MoE / Gemma2 / Gemma 4 / GLM-4 /
/// DeepSeek-V2 / Phi-3) — it never opens a safetensors shard, so `core-llm`'s `load_for_model` can
/// resolve a provider by model without loading weights. A multimodal snapshot (a `vision_config`
/// block — including a VLM whose `model_type` substring-matches a text family, e.g. `mllama`) is
/// declined so the vision provider claims it instead, **except** the families this provider serves
/// end-to-end: Qwen3.6, Qwen3-VL, and Gemma 4 unified.
pub fn can_load(spec: &LoadSpec) -> bool {
    if crate::gguf::is_gguf_path(&spec.source) {
        let prism = crate::prism_checkpoint::PrismGgufCheckpoint::is_prism(Path::new(&spec.source))
            .unwrap_or(false);
        if prism {
            return spec.projector_source.as_deref().is_none_or(|path| {
                crate::prism_vision_gguf::PrismVisionGguf::is_qwen3vl_merger(Path::new(path))
            });
        }
        if spec.projector_source.is_some() {
            return false;
        }
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
    if v.get("model_type").and_then(Value::as_str) == Some("prism_hadamard_qwen35") {
        return spec.projector_source.is_none();
    }
    // A multimodal snapshot (a `vision_config` block) is normally declined so a vision provider claims
    // it — EXCEPT the families this provider serves directly: Qwen3.6 (`qwen3_5`) and Qwen3-VL
    // (`qwen3_vl`), whose checkpoints are VLM-wrapped but whose matching ViT tower loads here, and
    // Gemma 4 unified, whose encoder-free vision embedder and audio projector likewise load here
    // (sc-18772). Gemma 4 is the reason this is not a Qwen-only list: `google/gemma-4-12B-it` carries
    // a `vision_config`, so the old rule declined the general-purpose Gemma 4 LLM outright.
    let arch = Architecture::from_config(&v);
    let serves_multimodal = matches!(arch, Ok(Architecture::Qwen35) | Ok(Architecture::Qwen3Vl))
        || matches!(arch, Ok(a) if a.is_gemma4());
    if v.get("vision_config").is_some() && !serves_multimodal {
        return false;
    }
    arch.is_ok()
}

#[cfg(test)]
mod tests {
    use super::{
        bonsai_sampling_defaults, can_load, can_load_vision, cuda_usable_memory_bytes,
        emit_content, ensure_supported_device, eos_token_ids, expand_vision_placeholders,
        host_load_budget, is_frozen_qwen38_config, load_memory_requirements,
        merged_frame_timestamps, prompt_opens_thinking, qwen35_dense_prefix,
        substitute_vision_placeholders, validate_context_window, video_placeholder_text, JsonMask,
        EAGER_ATTN_QUERY_CHUNK_SIZE,
    };
    use super::{qwen35_recurrent_admission_bytes, Decode as _, Decoder};

    #[test]
    fn accelerator_only_selector_is_exact_to_qwen38_and_bonsai() {
        let frozen: serde_json::Value = serde_json::from_str(include_str!(concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/../../../docs/reference/qwen38/config.json"
        )))
        .unwrap();
        assert!(is_frozen_qwen38_config(&frozen));

        let mut flat = frozen.clone();
        flat["model_type"] = serde_json::json!("qwen3_5_text");
        flat.as_object_mut().unwrap().remove("text_config");
        assert!(
            !is_frozen_qwen38_config(&flat),
            "generic flat qwen3_5_text checkpoints retain CPU support"
        );

        let mut other_geometry = frozen;
        other_geometry["text_config"]["vocab_size"] = serde_json::json!(32_000);
        assert!(!is_frozen_qwen38_config(&other_geometry));
    }

    #[test]
    fn accelerator_only_snapshots_reject_cpu_before_weight_inventory() {
        let qwen = tempfile::tempdir().unwrap();
        std::fs::write(
            qwen.path().join("config.json"),
            include_bytes!(concat!(
                env!("CARGO_MANIFEST_DIR"),
                "/../../../docs/reference/qwen38/config.json"
            )),
        )
        .unwrap();
        let error = ensure_supported_device(qwen.path(), &candle_core::Device::Cpu)
            .expect_err("Qwen3.8 CPU load must be rejected before missing weights are inspected");
        assert!(error
            .to_string()
            .contains("Candle CPU inference is unsupported"));

        let bonsai = tempfile::tempdir().unwrap();
        std::fs::write(
            bonsai.path().join("config.json"),
            br#"{"model_type":"prism_hadamard_qwen35"}"#,
        )
        .unwrap();
        let error = ensure_supported_device(bonsai.path(), &candle_core::Device::Cpu)
            .expect_err("Bonsai CPU load must be rejected before missing weights are inspected");
        assert!(error
            .to_string()
            .contains("Candle CPU inference is unsupported"));
    }
    use crate::decode::{ConstraintMask, RewindableConstraintMask};
    use core_llm::{
        ConstraintDecodeTable, Content, ImageRef, LlmMemoryGeometry, LoadSpec, Message, Role,
        VideoRef,
    };

    #[test]
    fn qwen35_dense_checkpoint_selects_one_decoder_root() {
        let select = |keys: &[&str]| qwen35_dense_prefix(|key| keys.contains(&key));
        assert_eq!(
            select(&["model.language_model.embed_tokens.weight"]).unwrap(),
            "model.language_model"
        );
        // The pinned Altworld/Hemmingway-1 index uses this flat text-only layout; MTP remains
        // rooted at `mtp.*` and is loaded independently of the decoder prefix.
        assert_eq!(
            select(&["model.embed_tokens.weight", "mtp.fc.weight"]).unwrap(),
            "model"
        );
        assert!(select(&[]).is_err());
        assert!(select(&[
            "model.language_model.embed_tokens.weight",
            "model.embed_tokens.weight"
        ])
        .is_err());
    }

    #[test]
    fn request_estimate_uses_the_eager_attention_runtime_tile() {
        let geometry = LlmMemoryGeometry {
            query_heads: 40,
            kv_heads: 4,
            head_dim: 128,
            layers: 64,
            element_bytes: 4,
            hidden_size: 5120,
            intermediate_size: 17_408,
            vocab_size: 248_320,
            recurrent_bytes: 0,
        };
        let chunked = core_llm::estimate_chunked_request_bytes(
            29_600,
            128,
            geometry,
            0,
            0,
            EAGER_ATTN_QUERY_CHUNK_SIZE,
        )
        .unwrap();
        let eager = core_llm::estimate_request_bytes(29_600, 128, geometry, 0, 0).unwrap();
        assert!(chunked < eager);
        assert!(chunked < 36_000_000_000, "bounded Candle peak: {chunked}");
    }

    #[test]
    fn dense_cuda_load_admission_separates_host_staging_from_current_vram() {
        // Frozen Qwen3.8 parent inventory from release/qwen38-bonsai-artifacts.json. Candle reads
        // the 18 shards serially; this is the total payload and the largest individual shard.
        let payload = 55_563_006_776;
        let largest_shard = 3_988_973_152;
        let (host_required, device_required) =
            load_memory_requirements(payload, largest_shard, 0, false, true, false).unwrap();

        assert_eq!(host_required, largest_shard);
        assert_eq!(device_required, Some(69_453_758_470));
        assert_eq!(host_load_budget(true, Some(102_171_148_288)), None);

        // A launch-time snapshot can cap but never inflate the current post-load CUDA capacity.
        let current_free = 60_000_000_000;
        let available =
            core_llm::effective_memory_budget(Some(current_free), Some(102_171_148_288)).unwrap();
        let error = core_llm::admit_request_memory(device_required.unwrap(), available)
            .expect_err("current CUDA shortfall must fail closed");
        assert!(error
            .to_string()
            .contains("only 60000000000 bytes are available"));
    }

    #[test]
    fn cuda_request_admission_counts_idle_async_pool_memory() {
        // Reproduce the observed zero-driver-free shape with a synthetic pool snapshot. The pool
        // counters were not captured by RC3, so this test deliberately makes no claim about their
        // exact campaign values.
        let total = 100;
        let reserved = total;
        let used = 60;
        let available = cuda_usable_memory_bytes(0, total, Some((reserved, used))).unwrap();
        assert_eq!(available, 40);
        core_llm::admit_request_memory(36, available)
            .expect("reusable pool memory must remain available to the owning allocator");
    }

    #[test]
    fn cuda_request_admission_rejects_real_pool_shortfall() {
        let gib = 1_u64 << 30;
        let available =
            cuda_usable_memory_bytes(gib / 2, 8 * gib, Some((4 * gib, 7 * gib / 2))).unwrap();
        assert_eq!(available, gib);
        let error = core_llm::admit_request_memory(gib + 1, available)
            .expect_err("live allocations and external use must remain unavailable");
        assert!(error
            .to_string()
            .contains("only 1073741824 bytes are available"));
    }

    #[test]
    fn cuda_request_admission_fails_closed_on_invalid_pool_snapshot() {
        assert_eq!(cuda_usable_memory_bytes(9, 8, None), None);
        assert_eq!(cuda_usable_memory_bytes(0, 8, Some((7, 8))), None);
        assert_eq!(cuda_usable_memory_bytes(0, 8, Some((9, 0))), None);
        assert_eq!(cuda_usable_memory_bytes(1, 8, Some((8, 0))), None);
    }

    #[test]
    fn load_memory_requirements_remain_checked() {
        assert!(load_memory_requirements(u64::MAX, 1, 0, false, true, false).is_none());
        assert!(load_memory_requirements(1, 1, u64::MAX, false, true, false).is_none());
        assert!(load_memory_requirements(u64::MAX / 4, 1, 0, false, true, true).is_none());
    }

    #[test]
    fn reasoning_boundary_commits_same_token_json_in_release_builds() {
        let table = core_llm::ConstraintDecodeTable {
            pieces: vec!["</think>{}".into(), String::new(), "{".into()],
            special: [1].into_iter().collect(),
        };
        let mut mask = JsonMask::new(&table, [1], true);
        assert!(mask.allowed()[0]);
        mask.accept(0);
        assert!(
            mask.allowed()[1],
            "same-token answer must commit even with debug assertions off"
        );
        assert!(
            !mask.allowed()[2],
            "completed answer cannot begin a second JSON value"
        );
        let mark = mask.checkpoint();
        mask.rewind(mark);
        assert!(
            mask.allowed()[1],
            "MTP rewind must replay phase and same-token JSON"
        );
    }

    #[test]
    fn json_constraint_waits_for_split_reasoning_close_and_rewinds_phase() {
        let table = ConstraintDecodeTable {
            pieces: vec![
                "reason".into(),
                "</thi".into(),
                "nk>".into(),
                "{".into(),
                "x".into(),
                "</think>x".into(),
                String::new(),
            ],
            special: [6].into_iter().collect(),
        };
        let mut mask = JsonMask::new(&table, [6], true);
        assert!(mask.allowed()[0]);
        assert!(
            !mask.allowed()[5],
            "invalid same-token JSON suffix is masked"
        );
        assert!(!mask.allowed()[6], "EOS cannot end an open reasoning block");
        mask.accept(0);
        let checkpoint = mask.checkpoint();
        mask.accept(1);
        mask.accept(2);
        assert!(mask.allowed()[3], "JSON begins only after split </think>");
        assert!(!mask.allowed()[4]);
        mask.rewind(checkpoint);
        assert!(mask.allowed()[1], "rewind restores the reasoning phase");
        assert!(!mask.allowed()[6]);

        let mut disabled = JsonMask::new(&table, [6], false);
        assert!(disabled.allowed()[3], "disabled thinking starts in JSON");
        assert!(!disabled.allowed()[0]);
    }

    #[test]
    fn stop_matches_utf8_after_incremental_detokenization_and_never_leaks_tail() {
        let mut matcher = core_llm::StopMatcher::new(["βγ".to_string(), "βγmore".to_string()]);
        let mut detok = core_llm::IncrementalDetok::default();
        let halt = std::cell::Cell::new(false);
        let (mut text, mut index, mut last) = (String::new(), 0, 0);
        for decoded in ["α", "α�", "αβ", "αβ�", "αβγleak"] {
            if let Some(delta) = detok.push(decoded) {
                emit_content(
                    delta,
                    1,
                    (&mut matcher, &halt),
                    &mut text,
                    &mut index,
                    &mut last,
                    &mut |_| {},
                );
            }
        }
        emit_content(
            "more leak",
            2,
            (&mut matcher, &halt),
            &mut text,
            &mut index,
            &mut last,
            &mut |_| {},
        );
        assert_eq!(text, "α");
        assert!(halt.get());
    }

    #[test]
    fn content_stop_is_trimmed_across_token_boundaries() {
        let mut matcher = core_llm::StopMatcher::new(["<STOP>".to_string()]);
        let halt = std::cell::Cell::new(false);
        let mut streamed = String::new();
        let mut index = 0;
        let mut last_id = 0;
        let mut events = Vec::new();
        emit_content(
            "answer<ST",
            1,
            (&mut matcher, &halt),
            &mut streamed,
            &mut index,
            &mut last_id,
            &mut |event| events.push(event),
        );
        emit_content(
            "OP>leak",
            2,
            (&mut matcher, &halt),
            &mut streamed,
            &mut index,
            &mut last_id,
            &mut |event| events.push(event),
        );
        assert_eq!(streamed, "answer");
        assert!(halt.get());
        assert_eq!(events.len(), 1);
    }

    #[test]
    fn bonsai_sampling_defaults_match_official_thinking_modes() {
        let defaults = bonsai_sampling_defaults();
        assert_eq!(defaults.thinking.temperature, 1.0);
        assert_eq!(defaults.thinking.top_p, 0.95);
        assert_eq!(defaults.thinking.top_k, 20);
        assert_eq!(defaults.thinking.presence_penalty, 0.0);
        assert_eq!(defaults.thinking.repetition_penalty, 1.0);
        assert_eq!(defaults.thinking.repetition_context, 0);

        assert_eq!(defaults.non_thinking.temperature, 0.7);
        assert_eq!(defaults.non_thinking.top_p, 0.8);
        assert_eq!(defaults.non_thinking.top_k, 20);
        assert_eq!(defaults.non_thinking.presence_penalty, 1.5);
        assert_eq!(defaults.non_thinking.repetition_penalty, 1.0);
        assert_eq!(defaults.non_thinking.repetition_context, 0);
    }

    #[test]
    fn prism_mlx_weightless_probe_requires_declared_embedded_vision() {
        let dir = tempfile::tempdir().unwrap();
        let config = |vision: bool| {
            serde_json::json!({
                "model_type": "prism_hadamard_qwen35",
                "components": {"text": true, "vision": vision, "mtp": false},
                "vision_config": {"depth": 1}
            })
        };
        std::fs::write(
            dir.path().join("config.json"),
            serde_json::to_vec(&config(true)).unwrap(),
        )
        .unwrap();
        let spec = LoadSpec::dense(dir.path().to_string_lossy());
        assert!(can_load(&spec));
        assert!(can_load_vision(&spec));

        std::fs::write(
            dir.path().join("config.json"),
            serde_json::to_vec(&config(false)).unwrap(),
        )
        .unwrap();
        assert!(can_load(&spec), "text Bonsai remains loadable");
        assert!(!can_load_vision(&spec));

        let associated = spec.clone().with_projector("external.gguf");
        assert!(!can_load(&associated));
        assert!(!can_load_vision(&associated));
    }

    #[test]
    fn frozen_qwen38_generation_config_uses_both_official_stop_tokens() {
        let dir = tempfile::tempdir().unwrap();
        std::fs::write(
            dir.path().join("generation_config.json"),
            include_str!(concat!(
                env!("CARGO_MANIFEST_DIR"),
                "/../../../docs/reference/qwen38/generation_config.json"
            )),
        )
        .unwrap();

        assert_eq!(eos_token_ids(dir.path()), vec![248046, 248044]);
    }

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

    #[test]
    fn invalid_video_timestamps_fail_before_prompt_rendering() {
        let frame = || ImageRef::new(1, 1, vec![0, 0, 0]).unwrap();
        for timestamps in [vec![0.0, f32::NAN], vec![0.5, 0.25], vec![-0.1, 0.0]] {
            let video = VideoRef {
                frames: vec![frame(), frame()],
                timestamps,
            };
            let messages = vec![Message {
                role: Role::User,
                content: vec![Content::Video(video)],
                thinking: None,
                tool_calls: Vec::new(),
            }];
            let error = substitute_vision_placeholders(&messages, 2)
                .expect_err("invalid timestamp sequence accepted");
            assert!(error.to_string().contains("timestamp"));
        }
    }

    #[test]
    fn expanded_media_tokens_count_against_context_window() {
        validate_context_window(64, 48, 16).unwrap();
        let error = validate_context_window(64, 49, 16)
            .expect_err("expanded prompt over context was accepted");
        assert!(error.to_string().contains("expanded prompt (49 tokens)"));
        assert!(error.to_string().contains("context window 64"));
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
        let g = j["video_grid_thw"].as_array().unwrap()[0]
            .as_array()
            .unwrap();
        let grid = [
            g[0].as_i64().unwrap() as i32,
            g[1].as_i64().unwrap() as i32,
            g[2].as_i64().unwrap() as i32,
        ];
        let merge = j["merge"].as_i64().unwrap() as i32;

        let (t, h, w, _delta) = crate::models::deepstack::mrope_positions_mm(
            &expanded_hf,
            &[],
            img,
            &[grid],
            vid,
            merge,
        )
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
    /// E6 (sc-24132): the static KV cache preallocates K/V for the whole request bound at its
    /// first step, and the geometry-based admission estimate already covers that preallocation —
    /// `estimate_request_bytes` charges KV for `prompt + max_new_tokens` positions over every layer
    /// at `element_bytes` (4), while the static cache holds only the full-attention layers in the
    /// compute dtype — so a request admitted under the estimate cannot fail its preallocation
    /// (and a request past the model bound fails closed with the typed error before allocating).
    #[test]
    fn qwen35_admission_covers_the_static_kv_preallocation() {
        use crate::decode::StepModel;

        let (_cfg, model) = crate::models::qwen35::tests::text_model();
        let (prompt_tokens, max_new_tokens) = (11usize, 21u32);
        let capacity = prompt_tokens + max_new_tokens as usize;
        let preallocation = model.static_kv_bytes(capacity) as u64;
        assert!(preallocation > 0);
        let decoder = Decoder::Qwen35(model);
        let geometry = decoder.memory_geometry();
        let kv_term = (capacity as u64)
            * geometry.layers
            * geometry.kv_heads
            * geometry.head_dim
            * geometry.element_bytes
            * 2;
        assert!(
            preallocation <= kv_term,
            "static preallocation {preallocation} exceeds the KV term {kv_term} admission charges"
        );
        let estimate =
            core_llm::estimate_request_bytes(prompt_tokens, max_new_tokens, geometry, 0, 0)
                .unwrap();
        assert!(estimate >= preallocation + geometry.recurrent_bytes);

        // What the step seam really allocates for that request is exactly the priced number.
        let Decoder::Qwen35(model) = &decoder else {
            unreachable!()
        };
        let cache = model.new_cache_for(capacity, 0).unwrap();
        assert_eq!(cache.memory().live_bytes as u64, preallocation);
        assert_eq!(cache.kv_kind(), crate::primitives::KvCacheKind::Static);
        // A declared overshoot is part of the bound (and of the priced preallocation).
        let cache = model.new_cache_for(capacity, 3).unwrap();
        assert_eq!(cache.kv_capacity(), Some(capacity + 3));
        assert_eq!(
            cache.memory().live_bytes as u64,
            model.static_kv_bytes(capacity + 3) as u64
        );
    }

    fn nvfp4_spec(source: &str) -> core_llm::LoadSpec {
        core_llm::LoadSpec {
            source: source.into(),
            projector_source: None,
            quantize: Some(core_llm::Quantize::Nvfp4),
        }
    }

    /// sc-24135 AC2: NVFP4 on a CPU device is a typed refusal naming the capability, settled by the
    /// helper `LlamaProvider::load` calls before the accelerator gate, admission or any weight read.
    #[test]
    fn nvfp4_on_cpu_is_a_typed_refusal_naming_the_capability() {
        let device = candle_core::Device::Cpu;
        match super::nvfp4_format(&nvfp4_spec("snapshot-dir"), &device) {
            Err(core_llm::Error::Unsupported(msg)) => {
                assert!(msg.starts_with("nvfp4: "), "names the capability: {msg}");
                assert!(msg.contains("sm_120"), "names the floor: {msg}");
                assert!(msg.contains("Cpu"), "names the device: {msg}");
            }
            other => panic!("expected a typed Unsupported refusal, got {other:?}"),
        }
        // Non-NVFP4 requests never touch the gate.
        for quantize in [
            None,
            Some(core_llm::Quantize::Q4),
            Some(core_llm::Quantize::Q8),
        ] {
            let spec = core_llm::LoadSpec {
                quantize,
                ..nvfp4_spec("snapshot-dir")
            };
            assert!(super::nvfp4_format(&spec, &device).unwrap().is_none());
        }
        // A GGUF source is refused by name on any device (it is already block-quantized).
        match super::nvfp4_format(&nvfp4_spec("model.gguf"), &device) {
            Err(core_llm::Error::Unsupported(msg)) => assert!(msg.starts_with("nvfp4: "), "{msg}"),
            other => panic!("expected a typed Unsupported refusal, got {other:?}"),
        }
    }

    /// sc-24135 AC2 with a mocked sub-sm_120 capability: the device gate's refusal reaches the
    /// backend-neutral contract as `Unsupported`, naming the capability, the floor and the device.
    #[test]
    fn a_sub_sm120_refusal_reaches_the_contract_as_unsupported() {
        for cap in [(8, 9), (9, 0), (10, 0)] {
            let refusal = candle_quant_kernels::nvfp4_refusal_for_compute_cap(cap).unwrap();
            match super::to_core(crate::Error::Nvfp4Refused(refusal)) {
                core_llm::Error::Unsupported(msg) => {
                    assert!(msg.starts_with("nvfp4: "), "{msg}");
                    assert!(msg.contains("sm_120"), "{msg}");
                    assert!(msg.contains(&format!("sm_{}{}", cap.0, cap.1)), "{msg}");
                }
                other => panic!("expected Unsupported, got {other:?}"),
            }
        }
    }

    /// sc-24135 AC2 through the real gate: `nvfp4_format` → `Nvfp4Context::require_with` (a real
    /// cuBLASLt handle, the capability probe mocked to a sub-sm_120 device) → the typed refusal →
    /// the contract's `Unsupported`, naming the capability, the floor and the mocked device.
    #[cfg(feature = "cuda")]
    #[test]
    fn nvfp4_format_refuses_a_mocked_sub_sm120_device_as_unsupported() {
        let Ok(device) = candle_core::Device::new_cuda(0) else {
            eprintln!("skipping: no CUDA device");
            return;
        };
        for cap in [(8, 9), (10, 0)] {
            let gate = |d: &candle_core::Device| {
                crate::primitives::projection::ProjectionFormat::nvfp4_with_cap_probe(d, |_| {
                    Ok(cap)
                })
            };
            match super::nvfp4_format_with(&nvfp4_spec("snapshot-dir"), &device, gate) {
                Err(core_llm::Error::Unsupported(msg)) => {
                    assert!(msg.starts_with("nvfp4: "), "{msg}");
                    assert!(msg.contains("sm_120"), "names the floor: {msg}");
                    assert!(
                        msg.contains(&format!("sm_{}{}", cap.0, cap.1)),
                        "names the device: {msg}"
                    );
                }
                Err(other) => panic!("expected Unsupported for {cap:?}, got {other:?}"),
                Ok(_) => panic!("a mocked sm_{}{} device served NVFP4", cap.0, cap.1),
            }
        }
    }

    /// sc-24135: an NVFP4 request for a snapshot outside the qwen3_5 family is refused by name in
    /// `nvfp4_format` — before admission and the device gate — so no memory or device error can
    /// mask it. The gate here would accept anything; it must not be reached.
    #[test]
    fn nvfp4_for_a_non_qwen35_snapshot_is_refused_before_the_device_gate() {
        let root = tempfile::tempdir().unwrap();
        std::fs::write(
            root.path().join("config.json"),
            r#"{"architectures":["LlamaForCausalLM"],"model_type":"llama"}"#,
        )
        .unwrap();
        let spec = nvfp4_spec(&root.path().to_string_lossy());
        let gate = |_: &candle_core::Device| -> crate::Result<_> {
            panic!("the device gate ran before the architecture refusal")
        };
        match super::nvfp4_format_with(&spec, &candle_core::Device::Cpu, gate) {
            Err(core_llm::Error::Unsupported(msg)) => {
                assert!(msg.starts_with("nvfp4: "), "{msg}");
                assert!(msg.contains("qwen3_5 family"), "{msg}");
                assert!(msg.contains("Llama"), "names the checkpoint: {msg}");
            }
            other => panic!("expected the architecture refusal, got {other:?}"),
        }
        // A qwen3_5 snapshot passes the architecture check and reaches the gate.
        std::fs::write(
            root.path().join("config.json"),
            r#"{"architectures":["Qwen3_5ForConditionalGeneration"],"model_type":"qwen3_5"}"#,
        )
        .unwrap();
        let reached = std::cell::Cell::new(false);
        let gate = |_: &candle_core::Device| -> crate::Result<_> {
            reached.set(true);
            Err(crate::Error::Unsupported("nvfp4: gate reached".into()))
        };
        let _ = super::nvfp4_format_with(&spec, &candle_core::Device::Cpu, gate);
        assert!(
            reached.get(),
            "a qwen3_5 snapshot must reach the device gate"
        );
    }

    /// On a build with no CUDA backend the whole load refuses NVFP4 before reading a byte: the
    /// source here does not exist, so any later step would have failed with a Load/IO error.
    #[cfg(not(feature = "cuda"))]
    #[test]
    fn nvfp4_load_without_cuda_refuses_before_reading_the_snapshot() {
        // A snapshot path that does not exist, inside a self-removing root.
        let root = tempfile::tempdir().unwrap();
        let missing = root.path().join("no-snapshot-here");
        match super::LlamaProvider::load(&nvfp4_spec(&missing.to_string_lossy())) {
            Err(core_llm::Error::Unsupported(msg)) => assert!(msg.starts_with("nvfp4: "), "{msg}"),
            Err(other) => panic!("expected the NVFP4 refusal first, got {other:?}"),
            Ok(_) => panic!("NVFP4 cannot load without CUDA"),
        }
    }

    /// E6: an NVFP4 load prices the packed copy (4.5 of every 16 source bits) on the device on top
    /// of the resident bf16 source and the existing headroom; the host side is unchanged.
    #[test]
    fn nvfp4_admission_prices_the_packed_copy_beside_the_bf16_source() {
        let payload = 55_000_000_000u64; // ~Qwen3.8-27B bf16
        let (host_dense, dense) =
            load_memory_requirements(payload, payload / 10, 0, false, true, false).unwrap();
        let (host_nv, nv) =
            load_memory_requirements(payload, payload / 10, 0, false, true, true).unwrap();
        assert_eq!(
            host_nv, host_dense,
            "NVFP4 quantizes on the device, not the host"
        );
        assert_eq!(nv.unwrap() - dense.unwrap(), payload * 9 / 32);
        assert_eq!(nv.unwrap(), payload + payload / 4 + payload * 9 / 32);
        // No device domain on CPU (NVFP4 never gets this far there, but the pricing is total).
        assert_eq!(
            load_memory_requirements(payload, payload, 0, false, false, true)
                .unwrap()
                .1,
            None
        );
    }

    /// E6: admission prices every recurrent state a Qwen3.5-family request's cache holds. The
    /// provider's cache (the reference/MTP `Decode::make_cache`) retains `REFERENCE_MAX_CHECKPOINTS`
    /// rollback checkpoints, and the geometry charges `1 + REFERENCE_MAX_CHECKPOINTS` states; a
    /// checkpointing step-seam cache holds more than one state and is covered only by the
    /// checkpoint term.
    #[test]
    fn qwen35_admission_prices_every_recurrent_state_the_cache_holds() {
        use crate::decode::{StepModel, StepRequest};
        use crate::models::qwen35::{REFERENCE_MAX_CHECKPOINTS, STEP_MAX_CHECKPOINTS};
        use crate::models::Qwen35Cache;
        use crate::primitives::nn::input_ids;

        let (cfg, model) = crate::models::qwen35::tests::text_model();
        let one_state = qwen35_recurrent_admission_bytes(&cfg, 0);
        assert!(one_state > 0);
        let decoder = Decoder::Qwen35(model);
        let geometry = decoder.memory_geometry();
        assert_eq!(
            geometry.recurrent_bytes,
            one_state * (1 + REFERENCE_MAX_CHECKPOINTS as u64),
            "the geometry charges the live state plus the provider cache's checkpoints"
        );
        // The admission estimate carries the whole recurrent term (charged once with MTP off).
        let without = core_llm::LlmMemoryGeometry {
            recurrent_bytes: 0,
            ..geometry
        };
        let est = |g| core_llm::estimate_chunked_request_bytes(8, 4, g, 0, 0, 64).unwrap();
        assert_eq!(est(geometry) - est(without), geometry.recurrent_bytes);

        // What the provider's cache really holds at steady state (prefill + several steps).
        let device = candle_core::Device::Cpu;
        let mut cache = decoder.make_cache();
        decoder
            .step(&input_ids(&[1, 7, 3], &device).unwrap(), cache.as_mut(), 0)
            .unwrap();
        for (i, t) in [42, 9, 2, 11].into_iter().enumerate() {
            decoder
                .step(
                    &input_ids(&[t], &device).unwrap(),
                    cache.as_mut(),
                    3 + i as i32,
                )
                .unwrap();
        }
        let held = cache.as_any_mut().downcast_mut::<Qwen35Cache>().unwrap();
        assert_eq!(held.max_checkpoints(), REFERENCE_MAX_CHECKPOINTS);
        assert!(
            held.recurrent_bytes() as u64 <= geometry.recurrent_bytes,
            "provider cache holds {} recurrent bytes, admission charges {}",
            held.recurrent_bytes(),
            geometry.recurrent_bytes
        );

        // A step-seam cache retains STEP_MAX_CHECKPOINTS states beyond the live one: one state of
        // admission does not cover it; the checkpoint term does.
        let Decoder::Qwen35(model) = &decoder else {
            unreachable!()
        };
        let mut step = StepModel::new_cache(model);
        model
            .forward_step(&mut step, StepRequest::last(&[1, 7, 3]))
            .unwrap();
        for t in [42, 9, 2, 11] {
            model
                .forward_step(&mut step, StepRequest::last(&[t]))
                .unwrap();
        }
        assert_eq!(step.checkpoint_offsets().len(), STEP_MAX_CHECKPOINTS);
        assert!(
            step.recurrent_bytes() as u64 > one_state,
            "a checkpointing cache holds more than one recurrent state"
        );
        assert!(
            step.recurrent_bytes() as u64
                <= qwen35_recurrent_admission_bytes(&cfg, STEP_MAX_CHECKPOINTS),
            "step cache holds {} recurrent bytes, its admission term is {}",
            step.recurrent_bytes(),
            qwen35_recurrent_admission_bytes(&cfg, STEP_MAX_CHECKPOINTS)
        );
    }
}
