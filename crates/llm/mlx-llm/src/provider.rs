//! The `core-llm` provider: a generic Llama model exposed through the backend-neutral contract.
//!
//! This is the mlx-llm half of story 7154 — it implements [`core_llm::TextLlm`] by wrapping the
//! [`CausalLm`] decoder, a [`core_llm::Tokenizer`], and a chat template, driving the internal
//! streaming decode loop and translating its token events into contract [`StreamEvent`]s (with
//! incremental detokenization). It registers into [`core_llm::registry`] under the id `mlx-llama`.
//!
//! The chat template is the model's own `chat_template` from `tokenizer_config.json` (rendered via
//! `core_llm::JinjaChatTemplate`, story 7164), falling back to the typed [`Llama3Template`] when a
//! snapshot ships no `tokenizer_config.json`.

use std::cell::OnceCell;
use std::path::Path;
use std::time::Instant;

use core_llm::{
    AudioRef, Channel, ChatTemplate, Constraint, ConstraintDecodeTable, ConstraintKind, Content,
    Error as CoreError, FinishReason as CoreFinish, ImageRef, IncrementalDetok, JinjaChatTemplate,
    JsonConstraint, Llama3Template, LlmMemoryGeometry, LoadSpec, Message, ModelSamplingDefaults,
    MtpMode, Quantize, ReasoningEffort, RenderOptions, Result as CoreResult, Sampling, StopMatcher,
    StreamEvent as CoreEvent, TextLlm, TextLlmCapabilities, TextLlmDescriptor, TextLlmOutput,
    TextLlmRequest, ThinkingSegmenter, Tokenizer, ToolCallSegmenter, Usage, VideoRef,
};

use crate::config::{Architecture, ModelConfig};
use crate::decode::{
    generate_from_prefill, generate_from_prefill_with_timings,
    generate_qwen35_mtp_multimodal_with_timings, generate_qwen35_mtp_with_timings,
    generate_with_timings, ConstraintMask, Decode, FinishReason, GenerationConfig,
    Qwen35MtpMultimodalPrompt, RewindableConstraintMask, StreamEvent,
};
use crate::image::Qwen35ImageProcessor;
use crate::models::gemma4_mm;
use crate::models::{
    CausalLm, Gemma4Layout, Gemma4Mm, Gemma4MmConfig, Qwen35Config, Qwen35Model,
    Qwen35VisionConfig, Qwen35VisionModel, VlmDecode,
};
use crate::primitives::attention::SDPA_MAX_FUSED_QLEN;
use crate::primitives::kv_cache::KvCache;
use crate::primitives::projection::QuantSpec;
use crate::primitives::sampler::SamplingParams;
use crate::primitives::{input_ids, Weights};
use crate::prism::PrismMlxPack;
use mlx_rs::ops::concatenate_axis;
use mlx_rs::Array;

/// The registry id of this provider.
pub const PROVIDER_ID: &str = "mlx-llama";

/// The loaded decoder, dispatched by architecture. The generic softmax-attention decoders share
/// [`CausalLm`]; Qwen3.6 (`qwen3_5`) is the hybrid linear-attention/full-attention decoder. Both
/// implement [`Decode`], so the generation loop is identical.
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

    fn step(
        &self,
        input_ids: &Array,
        cache: &mut dyn KvCache,
        offset: i32,
    ) -> crate::error::Result<Array> {
        match self {
            Decoder::Causal(m) => m.step(input_ids, cache, offset),
            Decoder::Qwen35(m) => m.step(input_ids, cache, offset),
        }
    }
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
                        (c.num_layers as u64)
                            .saturating_mul(c.linear_num_value_heads as u64)
                            .saturating_mul(c.linear_value_head_dim as u64)
                            .saturating_mul(
                                (c.linear_key_head_dim + c.linear_conv_kernel_dim) as u64,
                            )
                            .saturating_mul(4),
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

    /// Select the request-memory contract matching the complete decoder implementation.
    fn workspace_contract(&self) -> MlxWorkspaceContract<'_> {
        match self {
            Decoder::Qwen35(m) if m.config().moe.is_none() => MlxWorkspaceContract::Qwen35 {
                config: m.config(),
                prism: m.is_prism(),
            },
            // The MoE expert gather/scatter lifetime differs from the dense path below and has no
            // complete-model allocator calibration yet. Retain the safe eager estimate rather than
            // applying a dense-Qwen envelope to a different execution graph.
            Decoder::Qwen35(_) => MlxWorkspaceContract::Eager,
            Decoder::Causal(m) => {
                let c = m.config();
                if c.attn_logit_softcap.is_none() && !c.is_mla() && c.gemma4.is_none() {
                    MlxWorkspaceContract::Chunked
                } else {
                    MlxWorkspaceContract::Eager
                }
            }
        }
    }

    fn is_quantized(&self) -> bool {
        match self {
            Decoder::Causal(m) => m.is_quantized(),
            Decoder::Qwen35(m) => m.is_quantized(),
        }
    }

    /// The decoder as the backend-neutral multimodal seam. Both backbones implement [`VlmDecode`]
    /// (the Qwen3.6 hybrid and the generic Qwen3-VL causal decoder), so the provider drives the
    /// image/video prefill + decode through this one trait object rather than forking on the concrete
    /// decoder type.
    fn as_vlm(&self) -> &dyn VlmDecode {
        match self {
            Decoder::Causal(m) => m,
            Decoder::Qwen35(m) => m,
        }
    }
}

/// The Qwen-VL vision side of the provider: the ViT tower, the image preprocessor, and the
/// multimodal token ids needed to expand placeholders and assign M-RoPE positions. Present when the
/// loaded `qwen3_5` (Qwen3.6) or `qwen3_vl` (Qwen3-VL) checkpoint carries `model.visual.*`. The two
/// share the identical Qwen3-VL ViT tower (`Qwen3VLVisionModel == Qwen35VisionModel`); only the
/// decoder prefill differs ([`Decoder::Qwen35`] vs [`Decoder::Causal`]).
struct Qwen35Vision {
    tower: Qwen35VisionModel,
    processor: Qwen35ImageProcessor,
    image_token_id: i32,
    /// The `<|video_pad|>` placeholder token id (151656 for Qwen3-VL) — the per-frame video
    /// placeholder the processor expands to `frame_seqlen` copies.
    video_token_id: i32,
    spatial_merge_size: i32,
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
    /// (each `[n_tokens, hidden]`, one per `deepstack_visual_indexes` tap), plus the image's
    /// `grid_thw` (`[1, h, w]` in patch units). `n_tokens = (grid_h/merge)·(grid_w/merge)` is the
    /// placeholder expansion count.
    fn encode(&self, img: &ImageRef) -> CoreResult<(Array, Vec<Array>, [i32; 3])> {
        let (pixels, grid) = self
            .processor
            .preprocess(&img.pixels, img.width as usize, img.height as usize)
            .map_err(to_core)?;
        let out = self
            .tower
            .forward_with_deepstack(&pixels, &grid)
            .map_err(to_core)?;
        Ok((out.pooler_output, out.deepstack_features, grid[0]))
    }

    /// Encode one **video** (sampled frames) to its merged patch rows `[grid_t·n_per_frame, hidden]`,
    /// the per-tap DeepStack features, and the `video_grid_thw` (`[grid_t, h, w]`). The ViT tower is
    /// modality-agnostic — it processes the `grid_t` temporal patches as a block-diagonal-masked
    /// frame sequence exactly like multiple images — so this reuses `forward_with_deepstack`. The
    /// per-frame timestamp tokens are rendered separately (Text–Timestamp Alignment); here we only
    /// produce the visual features and the grid.
    fn encode_video(&self, video: &VideoRef) -> CoreResult<(Array, Vec<Array>, [i32; 3])> {
        let frames: Vec<(&[u8], usize, usize)> = video
            .frames
            .iter()
            .map(|f| (f.pixels.as_slice(), f.width as usize, f.height as usize))
            .collect();
        let (pixels, grid) = self.processor.preprocess_video(&frames).map_err(to_core)?;
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
    embeds: Array,
    positions: (Vec<i32>, Vec<i32>, Vec<i32>, i32),
    visual_pos_mask: Vec<bool>,
    deepstack: Vec<Array>,
}

/// Gemma 4's prepared multimodal prefill: the marker-expanded prompt ids and the decoder input
/// embeds with vision / audio feature rows spliced onto the soft-token positions.
///
/// Deliberately *not* [`MultimodalPrefill`]: Gemma 4 has no M-RoPE, no positional compression, and
/// no DeepStack taps, so there is no `mrope_delta` to shift the continuation by and no per-position
/// visual mask to build. Reusing the Qwen-VL struct would mean carrying four fields that mean
/// nothing here.
struct Gemma4Prefill {
    expanded_ids: Vec<i32>,
    embeds: Array,
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

/// Gemma 4's loaded multimodal front-ends.
struct Gemma4Runtime {
    mm: Gemma4Mm,
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

    fn step(
        &self,
        ids: &Array,
        cache: &mut dyn KvCache,
        offset: i32,
    ) -> crate::error::Result<Array> {
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
                        "[mlx-llama] Gemma 4: video input is not supported by this provider (no \
                         frame-sampling path); send frames as individual images"
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
/// per-frame Text–Timestamp-Alignment string `<{t} seconds><|vision_start|><|video_pad|><|vision_end|>`
/// (one `video_pad` per frame, each expanded to `frame_seqlen` after tokenizing). Keeps the core-llm
/// template contract image/video-free.
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
                .map(|c| -> CoreResult<Content> {
                    match c {
                        Content::Image(_) => Ok(Content::text(IMAGE_PLACEHOLDER)),
                        Content::Video(v) => {
                            v.validate().map_err(CoreError::InvalidRequest)?;
                            Ok(Content::text(video_placeholder_text(v, temporal_patch_size)))
                        }
                        Content::Text(t) => Ok(Content::Text(t.clone())),
                        // The Qwen-VL path has no audio projector, and this provider's
                        // `supports_audio` is false for every Qwen checkpoint, so `validate`
                        // rejects an audio-carrying request before substitution. Erroring here
                        // rather than dropping the block means that if that invariant ever breaks,
                        // the request fails loudly instead of being answered from its text alone.
                        Content::Audio(_) => Err(CoreError::Unsupported(
                            "[mlx-llama] the Qwen-VL path carries no audio; an audio block reached \
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
    /// Gemma 4's encoder-free vision embedder and/or audio projector, present iff this is a Gemma 4
    /// checkpoint that actually ships them (sc-18772). Independent of [`vision`](Self::vision):
    /// Gemma 4 does not use the Qwen-VL ViT/M-RoPE/DeepStack machinery at all.
    gemma4: Option<Gemma4Runtime>,
    /// Dense Prism `vision_tower.*` tensors retained verbatim for the native multimodal adapter.
    /// Text loading must not discard them merely because sc-23937 constructs only the decoder.
    _prism_vision_weights: Option<Weights>,
}

impl LlamaProvider {
    /// Load a provider from a snapshot directory (config.json + tokenizer.json + shards). Dispatches
    /// the decoder architecture from `config.json` (Llama / Mistral / Qwen3) and optionally
    /// quantizes the projections on load per `spec.quantize`.
    pub fn load(spec: &LoadSpec) -> CoreResult<Self> {
        if spec.projector_source.is_some()
            && Path::new(&spec.source).extension().and_then(|v| v.to_str()) != Some("gguf")
        {
            return Err(CoreError::Unsupported(
                "[mlx-llama] projector_source is only valid for a separable Prism GGUF model; \
                 safetensors snapshots carry their vision tower in the snapshot"
                    .into(),
            ));
        }
        let required = crate::load_memory::required_bytes(spec)?;
        let available = core_llm::effective_memory_budget(
            core_llm::available_host_memory_bytes(),
            core_llm::operational_memory_override()?,
        )?;
        core_llm::admit_request_memory(required, available)?;

        let dir = Path::new(&spec.source);
        if dir.extension().and_then(|v| v.to_str()) == Some("gguf") {
            return Self::load_prism_gguf(spec, dir);
        }
        let quant = spec.quantize.map(|q| match q {
            Quantize::Q4 => QuantSpec::q4(),
            Quantize::Q8 => QuantSpec::q8(),
        });
        // Read config.json once to dispatch the architecture: the hybrid Qwen3.6 (`qwen3_5`) decoder
        // has its own config/weights path (and `ModelConfig` deliberately rejects it).
        let cfg_value = read_config_value(dir)?;
        let arch = Architecture::from_config(&cfg_value).map_err(to_core)?;
        let weights = Weights::from_dir(dir).map_err(to_core)?;
        let is_prism =
            cfg_value.get("model_type").and_then(|v| v.as_str()) == Some("prism_hadamard_qwen35");
        if is_prism && quant.is_some() {
            return Err(CoreError::Load(
                "Prism snapshots are already packed 2-bit and reject load-time Q4/Q8".into(),
            ));
        }

        let mut prism_vision_weights = None;
        let (model, mut descriptor) = if arch == Architecture::Qwen35 {
            let qcfg = Qwen35Config::from_json(&cfg_value).map_err(to_core)?;
            let mut descriptor = descriptor_for_qwen35(&qcfg);
            let m = if is_prism {
                let pack = PrismMlxPack::from_dir(dir, &cfg_value, &weights).map_err(to_core)?;
                descriptor.family = "prism_hadamard_qwen35".into();
                let model =
                    Qwen35Model::from_prism_weights(&weights, qcfg, &pack).map_err(to_core)?;
                if pack.has_vision_tower {
                    let keys = weights
                        .keys()
                        .filter(|key| key.starts_with("vision_tower."))
                        .map(str::to_owned)
                        .collect::<Vec<_>>();
                    let retained = keys
                        .into_iter()
                        .filter_map(|key| weights.get(&key).cloned().map(|value| (key, value)))
                        .collect();
                    prism_vision_weights = Some(Weights::from_map(retained));
                }
                model
            } else {
                // The text decoder nests under `model.language_model` in the VLM-wrapped checkpoint.
                Qwen35Model::from_weights_with(&weights, "model.language_model", qcfg, quant)
                    .map_err(to_core)?
            };
            (Decoder::Qwen35(m), descriptor)
        } else {
            let cfg = ModelConfig::from_json(&cfg_value).map_err(to_core)?;
            let descriptor = descriptor_for(&cfg);
            let m = CausalLm::from_weights_with(&weights, "", cfg, quant).map_err(to_core)?;
            (Decoder::Causal(m), descriptor)
        };

        // Qwen-VL vision: load the ViT tower when the checkpoint carries `model.visual.*` (a wrapped
        // VLM) and the config exposes a `vision_config`. Covers Qwen3.6 (`qwen3_5`) and Qwen3-VL
        // (`qwen3_vl`), which share the identical Qwen3-VL ViT tower. Absent → a text-only checkpoint.
        let vision_prefix = if weights.contains("model.visual.patch_embed.proj.weight") {
            Some("model.visual")
        } else if is_prism && weights.contains("vision_tower.patch_embed.proj.weight") {
            Some("vision_tower")
        } else {
            None
        };
        let vision = if let (true, Some(prefix)) = (
            (arch == Architecture::Qwen35 || arch == Architecture::Qwen3Vl)
                && cfg_value.get("vision_config").is_some(),
            vision_prefix,
        ) {
            let vcfg = Qwen35VisionConfig::from_json(&cfg_value).map_err(to_core)?;
            let tower = if is_prism {
                Qwen35VisionModel::from_mlx_weights(&weights, prefix, vcfg.clone())
            } else {
                Qwen35VisionModel::from_weights(&weights, prefix, vcfg.clone())
            }
            .map_err(to_core)?;
            let image_token_id = cfg_value
                .get("image_token_id")
                .and_then(|x| x.as_i64())
                .map(|x| x as i32)
                .unwrap_or(248056);
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
            let mm = Gemma4Mm::from_weights(&weights, layout, mm_cfg).map_err(to_core)?;
            // Audio tracks the loaded tensors; vision does not — see `gemma4_vision_is_validated`.
            descriptor.capabilities.supports_vision =
                mm.vision.is_some() && gemma4_vision_is_validated();
            descriptor.capabilities.supports_audio = mm.audio.is_some();
            mm.audio.is_some().then_some(Gemma4Runtime { mm })
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
        if supports_reasoning_effort {
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
        if is_prism {
            descriptor.capabilities.model_sampling_defaults = Some(bonsai_sampling_defaults());
        }
        descriptor.capabilities.supports_preserve_thinking = supports_preserve_thinking;
        descriptor.capabilities.supports_tools = supports_tools;
        Ok(Self {
            descriptor,
            model,
            tokenizer,
            template,
            stop_tokens,
            constraint_table: OnceCell::new(),
            vision,
            gemma4,
            _prism_vision_weights: prism_vision_weights,
        })
    }

    fn load_prism_gguf(spec: &LoadSpec, path: &Path) -> CoreResult<Self> {
        if spec.quantize.is_some() {
            return Err(CoreError::Load(
                "Prism GGUF is already packed and rejects load-time Q4/Q8".into(),
            ));
        }
        let file = crate::gguf::GgufFile::open(path).map_err(to_core)?;
        let loaded = crate::prism_gguf::load(&file).map_err(to_core)?;
        let qcfg = Qwen35Config::from_json(&loaded.config).map_err(to_core)?;
        let mut descriptor = descriptor_for_qwen35(&qcfg);
        descriptor.family = "prism_hadamard_qwen35".into();
        let (thinking, reasoning, preserve, tools) = loaded.template_capabilities;
        descriptor.capabilities.supports_thinking = thinking;
        descriptor.capabilities.supports_reasoning_effort = reasoning;
        if reasoning {
            descriptor.capabilities.reasoning_efforts =
                vec![ReasoningEffort::XHigh, ReasoningEffort::Medium];
        }
        descriptor.capabilities.model_sampling_defaults = Some(bonsai_sampling_defaults());
        descriptor.capabilities.supports_preserve_thinking = preserve;
        descriptor.capabilities.supports_tools = tools;
        let model = Qwen35Model::from_prism_weights(&loaded.weights, qcfg, &loaded.pack)
            .map_err(to_core)?;
        let vision = match spec.projector_source.as_deref() {
            Some(source) => {
                let projector = crate::gguf::GgufFile::open(source).map_err(to_core)?;
                let loaded_vision = crate::prism_vision_gguf::load(&projector).map_err(to_core)?;
                let tower = Qwen35VisionModel::from_weights(
                    &loaded_vision.weights,
                    "vision_tower",
                    loaded_vision.config.clone(),
                )
                .map_err(to_core)?;
                descriptor.capabilities.supports_vision = true;
                descriptor.capabilities.supports_video = true;
                Some(Qwen35Vision {
                    tower,
                    processor: Qwen35ImageProcessor::default(),
                    image_token_id: 248056,
                    video_token_id: 248057,
                    spatial_merge_size: loaded_vision.config.spatial_merge_size,
                })
            }
            None => None,
        };
        Ok(Self {
            descriptor,
            model: Decoder::Qwen35(model),
            tokenizer: loaded.tokenizer,
            template: loaded.template,
            stop_tokens: loaded.stop_tokens,
            constraint_table: OnceCell::new(),
            vision,
            gemma4: None,
            _prism_vision_weights: None,
        })
    }

    /// Whether the loaded model's projections are quantized.
    pub fn is_quantized(&self) -> bool {
        self.model.is_quantized()
    }

    /// Whether a Prism VLM's dense vision tensors were retained for the multimodal adapter.
    pub fn has_deferred_prism_vision(&self) -> bool {
        self._prism_vision_weights.is_some()
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
            gemma4: None,
            _prism_vision_weights: None,
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
        let mut image_feats: Vec<Array> = Vec::with_capacity(images.len());
        let mut image_counts: Vec<usize> = Vec::with_capacity(images.len());
        if !images.is_empty() {
            let tower = rt.mm.vision.as_ref().ok_or_else(|| {
                CoreError::Unsupported(
                    "[mlx-llama] Gemma 4: this checkpoint ships no vision embedder, so it cannot \
                     be conditioned on an image"
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
                let patches = gemma4_mm::patch_array(&flat, vcfg.patch_elems()).map_err(to_core)?;
                let feats = tower.forward(&patches, (gh, gw)).map_err(to_core)?;
                image_counts.push(gh * gw);
                image_feats.push(feats);
            }
        }

        // Audio: one feature block and one soft-token count per clip, in document order.
        let clips = collect_audio(messages);
        let mut audio_feats: Vec<Array> = Vec::with_capacity(clips.len());
        let mut audio_counts: Vec<usize> = Vec::with_capacity(clips.len());
        if !clips.is_empty() {
            let proj = rt.mm.audio.as_ref().ok_or_else(|| {
                CoreError::Unsupported(
                    "[mlx-llama] Gemma 4: this checkpoint ships no audio projector, so it cannot \
                     be conditioned on audio"
                        .to_string(),
                )
            })?;
            let acfg = proj.config().clone();
            for clip in &clips {
                // The projector's framing is defined in samples at a fixed rate; resampling behind
                // the caller's back would silently change what the model hears.
                if clip.sample_rate != acfg.sample_rate {
                    return Err(CoreError::InvalidRequest(format!(
                        "[mlx-llama] Gemma 4 audio expects {} Hz mono PCM, got {} Hz; resample \
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
                    gemma4_mm::frame_array(&framed, acfg.samples_per_token).map_err(to_core)?;
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
            .embed_input_ids(&input_ids(&expanded))
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

        // Walk the conversation in document order; encode each visual once, in order, so the
        // concatenated feature buffer lines up one-to-one with the visual placeholder spans of the
        // (image+video) prompt. Image placeholders expand to one count; a video expands to `grid_t`
        // per-frame counts (`frame_seqlen` each), in frame order.
        let mut feats: Vec<Array> = Vec::new();
        let mut image_counts: Vec<usize> = Vec::new();
        let mut video_counts: Vec<usize> = Vec::new();
        let mut image_grids: Vec<[i32; 3]> = Vec::new();
        let mut video_grids: Vec<[i32; 3]> = Vec::new();
        let mut deepstack_by_tap: Vec<Vec<Array>> = Vec::new();
        let merge = vision.spatial_merge_size;

        let mut push_deepstack = |deepstack: Vec<Array>| -> CoreResult<()> {
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
                        image_counts.push(f.shape()[0] as usize);
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
                            "[mlx-llama] the Qwen-VL prefill carries no audio; an audio block \
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
            crate::models::qwen35::expand_vision_placeholders(prompt_ids, img_id, &image_counts)
                .map_err(to_core)?;
        let expanded =
            crate::models::qwen35::expand_vision_placeholders(&expanded, vid_id, &video_counts)
                .map_err(to_core)?;
        let visual_pos_mask: Vec<bool> = expanded
            .iter()
            .map(|&id| id == img_id || id == vid_id)
            .collect();

        let refs: Vec<&Array> = feats.iter().collect();
        let all_features = match refs.as_slice() {
            [one] => (*one).clone(),
            many => concatenate_axis(many, 0).map_err(|e| to_core(e.into()))?,
        };
        let mut deepstack = Vec::with_capacity(deepstack_by_tap.len());
        for by_visual in deepstack_by_tap {
            let refs: Vec<&Array> = by_visual.iter().collect();
            deepstack.push(match refs.as_slice() {
                [one] => (*one).clone(),
                many => concatenate_axis(many, 0).map_err(|e| to_core(e.into()))?,
            });
        }

        // Embed the expanded ids, splice in the vision features (image+video placeholder rows), and
        // compute interleaved-M-RoPE positions over both image and video grids — through the shared
        // `VlmDecode` seam, identical for whichever decoder powers this VLM (Qwen3.6 hybrid or
        // Qwen3-VL generic-causal).
        let placeholders = [img_id, vid_id];
        let model = self.model.as_vlm();
        let embeds = model
            .embed_input_ids(&input_ids(&expanded))
            .map_err(to_core)?;
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
        let accepted = self.accepted.clone();
        for token in accepted {
            self.accept(token);
        }
        self.accepted.truncate(checkpoint);
    }
}

/// Use the model's own Jinja `chat_template` (from `tokenizer_config.json`, story 7164) when
/// present; otherwise fall back to the typed Llama-3 template. Also reports two template-gated
/// capabilities, detected from the source (not the family, matching the transformers convention):
/// - **thinking** — the template gates an `enable_thinking` kwarg (sc-7585).
/// - **tools** — the template renders tool calls (it mentions `tool_call`), so it has a `tools`
///   section and the model emits parseable `<tool_call>` blocks (sc-7636). Covers the Qwen3.6 XML and
///   the Qwen2.5/Hermes JSON tool templates alike.
fn load_chat_template(dir: &Path) -> (Box<dyn ChatTemplate>, bool, bool, bool, bool) {
    // The sidecar `chat_template.jinja` wins over the embedded key — see `sidecar_chat_template`.
    if let Some(t) = sidecar_chat_template(dir) {
        let supports_thinking = t.source().contains("enable_thinking");
        let supports_reasoning_effort = t.source().contains("reasoning_effort");
        let supports_preserve_thinking = t.source().contains("preserve_thinking");
        let supports_tools = t.source().contains("tool_call");
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
            let supports_reasoning_effort = t.source().contains("reasoning_effort");
            let supports_preserve_thinking = t.source().contains("preserve_thinking");
            let supports_tools = t.source().contains("tool_call");
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

/// Run a piece of answer-channel text through the tool-call segmenter when active, returning the
/// plain-content runs to stream (tool-call blocks lifted out + parsed). With no segmenter the text
/// passes straight through, so the non-tools path is byte-identical to before.
fn tool_pieces(seg: &mut Option<ToolCallSegmenter>, text: &str) -> Vec<String> {
    match seg {
        Some(ts) => ts.push(text),
        None => vec![text.to_string()],
    }
}

/// Push one content piece through the stop matcher and emit the released text as a Content token
/// event. Shared by the streaming loop and the end-of-generation tails. `*last_id` / `*emit_index`
/// advance only when text is actually emitted, so the contract's token index stays gap-free across
/// stripped markers and lifted-out tool-call blocks.
#[allow(clippy::too_many_arguments)]
fn emit_content(
    piece: &str,
    id: u32,
    stop_matcher: &mut StopMatcher,
    streamed: &mut String,
    emit_index: &mut usize,
    last_id: &mut u32,
    halt: &std::cell::Cell<bool>,
    on_event: &mut dyn FnMut(CoreEvent),
) {
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
        self.validate(req)?;
        if req.cancel.is_cancelled() {
            return Err(CoreError::Canceled); // typed pre-inference cancel
        }

        // Multimodal (Qwen-VL + image/video content): replace image blocks with the Qwen-VL image
        // placeholder (`<|vision_start|><|image_pad|><|vision_end|>`) and video blocks with the
        // per-frame Text–Timestamp-Alignment placeholder string
        // (`<{t} seconds><|vision_start|><|video_pad|><|vision_end|>` ×frames), so the (vision-free)
        // chat template renders the framing verbatim. The visuals are encoded + spliced after
        // tokenizing, in document order. Text-only requests are unchanged.
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

        // Render the conversation and tokenize. The template already includes BOS, so encode
        // without auto special tokens. `enable_thinking` (sc-7585) flows into the template kwarg so
        // a no-think (Disabled) request injects the model's empty `<think></think>` generation
        // prompt; Auto omits the kwarg (template default).
        let prompt = self.template.render_with(
            messages,
            &RenderOptions {
                add_generation_prompt: true,
                enable_thinking: req.enable_thinking_kwarg(),
                // Bonsai's official model card does not recommend `low` as an effective distinct
                // level, so it is omitted from the selectable capability list. The frozen template
                // still accepts and renders `low`; preserve that compatibility input verbatim.
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

        // MLX uses unified memory. Admit from a fresh host availability snapshot before visual
        // preprocessing or model execution, using pure request geometry for visual expansion.
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
        // The tokenized prompt plus pure visual expansion geometry is known before any native
        // preprocessing. Reject architectural overflow before consulting transient capacity so a
        // valid context error cannot be masked by the machine's current memory pressure.
        validate_context_window(
            self.descriptor.capabilities.max_context_tokens,
            admitted_prompt,
            req.max_new_tokens,
        )?;
        let required = estimate_mlx_request_bytes(
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
            self.model.workspace_contract(),
        )
        .ok_or_else(|| CoreError::InvalidRequest("request memory estimate overflow".into()))?;
        let available = core_llm::effective_memory_budget(
            core_llm::available_host_memory_bytes(),
            core_llm::operational_memory_override()?,
        )?;
        core_llm::admit_request_memory(required, available)?;

        // Encode + splice the visuals and compute M-RoPE positions (the placeholder-expanded prompt
        // becomes the effective sequence). `None` on the text-only path.
        let qwen_prefill_started = (multimodal && !gemma4_mm_request).then(Instant::now);
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

        let mut mtp_draft_tokens = match req.mtp {
            MtpMode::Off => None,
            MtpMode::Auto => self
                .descriptor
                .capabilities
                .mtp
                .map(|caps| caps.recommended_draft_tokens as usize),
            MtpMode::Enabled { draft_tokens } => Some(draft_tokens as usize),
        };
        // Gemma 4 has no Qwen predictor. Qwen multimodal prompts use the fused-embedding MTP route
        // below, including their explicit three-axis positions and continuation delta.
        if mtp_draft_tokens.is_some() && gemma4_mm_request {
            if matches!(req.mtp, MtpMode::Enabled { .. }) {
                return Err(CoreError::Unsupported(
                    "[mlx-llama] native Qwen MTP is unavailable for Gemma 4 prompts".into(),
                ));
            }
            mtp_draft_tokens = None;
        }

        // Structured-output constraint (story 7166): build a JSON mask over the cached decode table.
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

        // Request `stop` strings (story 7349): a backend-neutral matcher over the decoded text.
        // Matching is in the detokenization seam below (not the token-id loop) because a stop string
        // need not align to a token boundary. When no stops are requested the matcher is a
        // transparent pass-through, so streaming output stays byte-identical to before.
        let mut stop_matcher = StopMatcher::new(req.stop.iter().cloned());
        let stop_active = !stop_matcher.is_empty();
        // A single-threaded latch the detok sink trips on a stop hit; the decode loop reads it after
        // each token via `should_stop` and halts with `FinishReason::Stopped`.
        let halt = std::cell::Cell::new(false);
        // Content emitted as the matchers release it — the result text when stop strings or thinking
        // are active (the plain path still decodes all tokens, byte-identical to before).
        let mut streamed = String::new();
        // Reasoning text, accumulated from the Thinking channel (sc-7585).
        let mut thinking_buf = String::new();
        // Contract token index: a running counter over *emitted* events, not the raw decode step —
        // detok hold-backs and stripped `<think>`/`</think>` marker tokens produce no event, so this
        // stays gap-free (and equals the step in the common one-delta-per-token case).
        let mut emit_index = 0usize;
        let mut last_id = 0u32; // id of the last emitted token, for flushed-tail events

        // A reasoning segmenter when the model advertises a thinking mode: it splits the decoded
        // stream into `<think>…</think>` reasoning vs answer (markers stripped). `None` otherwise, so
        // a non-thinking provider stays on the original single-channel path.
        let thinking_active = self.descriptor.capabilities.supports_thinking;
        let mut segmenter = thinking_active.then(ThinkingSegmenter::default);
        // Some chat templates open the reasoning block *in the prompt* — e.g. Qwen3.6 ends the
        // generation prompt with `<|im_start|>assistant\n<think>\n`, so the model generates inside
        // the block and only emits the closing `</think>`. Prime the segmenter into the Thinking
        // channel by feeding it that already-rendered opening marker (stripped, so it emits nothing);
        // otherwise the reasoning would be misclassified as answer. Disabled mode renders a *closed*
        // `<think>\n\n</think>`, so this correctly does not prime.
        if let Some(seg) = segmenter.as_mut() {
            if constraint_starts_in_reasoning {
                let _ = seg.push("<think>");
                debug_assert!(seg.in_thinking());
            }
        }

        // A tool-call segmenter when the request offers tools and the model's template renders them:
        // it lifts `<tool_call>` blocks out of the answer channel (markup excluded from the streamed
        // text) and parses them into structured calls (sc-7636). `None` otherwise, so a no-tools
        // request flows straight through `tool_pieces` unchanged.
        let tools_active = self.descriptor.capabilities.supports_tools && !req.tools.is_empty();
        let mut tool_seg = tools_active.then(|| ToolCallSegmenter::new(&req.tools));

        // Drive the internal loop; translate token-id events to contract text-delta events via
        // incremental detokenization (re-decode the running sequence, emit the new suffix). The
        // `IncrementalDetok` guard holds back lossy U+FFFD placeholders so a multi-byte character
        // split across BPE tokens streams intact (and never panics a mid-char slice) — sc-12452.
        // The segmenter (when active) splits each delta into reasoning vs answer; answer text then
        // feeds the stop matcher so a stop string is trimmed and halts generation.
        let tokenizer = &self.tokenizer;
        let (out, mtp_stats, timing) = {
            let mut acc: Vec<u32> = Vec::new();
            let mut detok = IncrementalDetok::new();
            let mut sink = |ev: StreamEvent| {
                if let StreamEvent::Token { id, .. } = ev {
                    let id = id as u32;
                    acc.push(id);
                    if let Ok(text) = tokenizer.decode(&acc, true) {
                        if let Some(delta) = detok.push(&text) {
                            let delta = delta.to_string();
                            match segmenter.as_mut() {
                                Some(seg) => {
                                    for span in seg.push(&delta) {
                                        match span.channel {
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
                                            Channel::Content => {
                                                // Answer text → tool segmenter (lifts out tool-call
                                                // blocks) → stop matcher → emit.
                                                for piece in tool_pieces(&mut tool_seg, &span.text)
                                                {
                                                    emit_content(
                                                        &piece,
                                                        id,
                                                        &mut stop_matcher,
                                                        &mut streamed,
                                                        &mut emit_index,
                                                        &mut last_id,
                                                        &halt,
                                                        &mut *on_event,
                                                    );
                                                }
                                            }
                                        }
                                    }
                                }
                                None => {
                                    for piece in tool_pieces(&mut tool_seg, &delta) {
                                        emit_content(
                                            &piece,
                                            id,
                                            &mut stop_matcher,
                                            &mut streamed,
                                            &mut emit_index,
                                            &mut last_id,
                                            &halt,
                                            &mut *on_event,
                                        );
                                    }
                                }
                            }
                        }
                    }
                }
            };
            let should_stop = || halt.get();
            let should_stop_opt = stop_active.then_some(&should_stop as &dyn Fn() -> bool);
            match &mm {
                // Multimodal: prefill the spliced embeds with interleaved M-RoPE + DeepStack fusion,
                // then decode the continuation (text positions shifted by `mrope_delta`) through the
                // shared loop — one path for either backbone via the `VlmDecode` seam.
                Some(m) => {
                    let (t, h, w, delta) = &m.positions;
                    let pos = [t.as_slice(), h.as_slice(), w.as_slice()];
                    if let (Decoder::Qwen35(model), Some(num_draft)) =
                        (&self.model, mtp_draft_tokens)
                    {
                        let prompt = Qwen35MtpMultimodalPrompt {
                            input_ids: &m.expanded_ids,
                            embeddings: &m.embeds,
                            positions: pos,
                            visual_pos_mask: &m.visual_pos_mask,
                            deepstack: &m.deepstack,
                            continuation_delta: *delta,
                        };
                        let (timed, stats) = generate_qwen35_mtp_multimodal_with_timings(
                            model,
                            &prompt,
                            &config,
                            num_draft,
                            &req.cancel,
                            &mut sink,
                            json_mask
                                .as_mut()
                                .map(|m| m as &mut dyn RewindableConstraintMask),
                            should_stop_opt,
                            qwen_prefill_started
                                .expect("Qwen multimodal preparation starts the prefill clock"),
                        )
                        .map_err(to_core)?;
                        (timed.output, Some(stats), Some(timed.timer))
                    } else {
                        let model = self.model.as_vlm();
                        let mut cache = model.make_cache();
                        let first = model
                            .prefill_with_deepstack(
                                &m.embeds,
                                pos,
                                cache.as_mut(),
                                &m.visual_pos_mask,
                                &m.deepstack,
                            )
                            .map_err(to_core)?;
                        let shifted = Shifted {
                            model,
                            delta: *delta,
                        };
                        let timed = generate_from_prefill_with_timings(
                            &shifted,
                            cache.as_mut(),
                            first,
                            m.expanded_ids.clone(),
                            &config,
                            &req.cancel,
                            &mut sink,
                            json_mask.as_mut().map(|m| m as &mut dyn ConstraintMask),
                            should_stop_opt,
                            qwen_prefill_started
                                .expect("Qwen multimodal preparation starts the prefill clock"),
                        )
                        .map_err(to_core)?;
                        (timed.output, None, Some(timed.timer))
                    }
                }
                // Gemma 4 multimodal: prefill the spliced embeds on ordinary causal 1-D positions
                // (no M-RoPE, so no position shift for the continuation), then decode through the
                // shared loop against the unwrapped decoder.
                None => match &g4 {
                    Some(m) => {
                        let model = match &self.model {
                            Decoder::Causal(c) => c,
                            Decoder::Qwen35(_) => {
                                return Err(CoreError::Load(
                                    "gemma 4: the multimodal path requires the generic causal \
                                     decoder"
                                        .into(),
                                ))
                            }
                        };
                        let mut cache = model.new_cache();
                        let first = model
                            .decode_logits_from_embeds(&m.embeds, &mut cache, 0)
                            .map_err(to_core)?;
                        let out = generate_from_prefill(
                            &self.model,
                            &mut cache,
                            first,
                            m.expanded_ids.clone(),
                            &config,
                            &req.cancel,
                            &mut sink,
                            json_mask.as_mut().map(|m| m as &mut dyn ConstraintMask),
                            should_stop_opt,
                        )
                        .map_err(to_core)?;
                        (out, None, None)
                    }
                    None => match (&self.model, mtp_draft_tokens) {
                        (Decoder::Qwen35(model), Some(num_draft)) => {
                            let (timed, stats) = generate_qwen35_mtp_with_timings(
                                model,
                                &prompt_ids,
                                &config,
                                num_draft,
                                &req.cancel,
                                &mut sink,
                                json_mask
                                    .as_mut()
                                    .map(|m| m as &mut dyn RewindableConstraintMask),
                                should_stop_opt,
                            )
                            .map_err(to_core)?;
                            (timed.output, Some(stats), Some(timed.timer))
                        }
                        _ => {
                            let timed = generate_with_timings(
                                &self.model,
                                &prompt_ids,
                                &config,
                                &req.cancel,
                                &mut sink,
                                json_mask.as_mut().map(|m| m as &mut dyn ConstraintMask),
                                should_stop_opt,
                            )
                            .map_err(to_core)?;
                            (timed.output, None, Some(timed.timer))
                        }
                    },
                },
            }
        };

        // End-of-generation tails, in pipeline order. First the thinking segmenter's held-back
        // partial marker (it turned out not to begin a marker) as current-channel text — reasoning
        // straight out, answer through the tool segmenter; then the tool segmenter's own tail (held
        // partial `<tool_call>` / an unterminated block surfaced as content); then the stop matcher's
        // held-back partial stop.
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
                                &mut stop_matcher,
                                &mut streamed,
                                &mut emit_index,
                                &mut last_id,
                                &halt,
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
                    &mut stop_matcher,
                    &mut streamed,
                    &mut emit_index,
                    &mut last_id,
                    &halt,
                    &mut *on_event,
                );
            }
        }
        // If generation ended for any reason other than a stop string, flush the stop matcher's
        // held-back tail (a partial stop-prefix that never completed) — real output to stream/return.
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

        // Result text: the streamed answer when stop strings, thinking, or tools are active (any of
        // which means the streamed channel is the authoritative answer with markup removed);
        // otherwise the original decode-all-tokens path (byte-identical to before). Reasoning and
        // tool calls, if the model produced any, are reported separately (markup excluded from text).
        // `streamed` accumulates only `IncrementalDetok`-released deltas, so it carries no
        // transient U+FFFD placeholders; a character truncated by end-of-generation is dropped
        // rather than surfaced as U+FFFD (sc-12452).
        let text = if stop_active || thinking_active || tools_active {
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
        let timings = timing.map(|timer| timer.finish());
        Ok(TextLlmOutput {
            timings,
            text,
            thinking,
            tool_calls,
            usage,
            mtp: mtp_stats.map(|stats| core_llm::MtpStats {
                proposed_tokens: u32::try_from(stats.proposed).unwrap_or(u32::MAX),
                accepted_tokens: u32::try_from(stats.accepted).unwrap_or(u32::MAX),
                target_forwards: u32::try_from(stats.forwards).unwrap_or(u32::MAX),
            }),
            finish_reason: Some(finish),
        })
    }
}

/// The descriptor for the `mlx-llama` provider (constructible without loading weights; used for
/// explicit catalog composition and inspection).
pub fn provider_descriptor() -> TextLlmDescriptor {
    TextLlmDescriptor {
        id: PROVIDER_ID.to_string(),
        family: "llama".to_string(),
        backend: "mlx".to_string(),
        capabilities: TextLlmCapabilities {
            max_context_tokens: 0,
            max_new_tokens: 0,
            supports_system_prompt: true,
            // Text-only today; the VLM path (sc-7157) flips this on for a vision provider.
            supports_vision: false,
            // Weightless default: conservative. The load path flips this on for a Qwen3-VL checkpoint
            // whose config carries a `video_token_id` (sc-8081).
            supports_video: false,
            // Weightless default: conservative. The load path flips this on for a Gemma 4 checkpoint
            // that actually ships an audio projector (sc-18772).
            supports_audio: false,
            // Weightless default: conservative. The load path (descriptor_for + load) flips this on
            // when the loaded model's own chat template gates an `enable_thinking` kwarg (sc-7585).
            supports_thinking: false,
            supports_reasoning_effort: false,
            reasoning_efforts: Vec::new(),
            model_sampling_defaults: None,
            supports_preserve_thinking: false,
            // Weightless default: conservative. The load path flips this on when the loaded model's
            // chat template renders tool calls (sc-7636).
            supports_tools: false,
            mtp: None,
            // JSON-constrained decoding (sc-7166).
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

/// The loaded-model descriptor for the hybrid Qwen3.6 (`qwen3_5`) decoder (parsed via
/// [`Qwen35Config`], which `ModelConfig` does not represent).
fn descriptor_for_qwen35(cfg: &Qwen35Config) -> TextLlmDescriptor {
    let mut d = provider_descriptor();
    d.family = Architecture::Qwen35.family().to_string();
    d.capabilities.max_context_tokens = cfg.max_position_embeddings.max(0) as usize;
    if cfg.mtp_num_hidden_layers > 0 {
        d.capabilities.mtp = Some(core_llm::MtpCapabilities {
            max_draft_tokens: u32::MAX,
            recommended_draft_tokens: 3,
        });
    }
    d
}

/// Read and parse `config.json` from a snapshot directory into a JSON value (used to dispatch the
/// architecture before constructing the architecture-specific config).
fn read_config_value(dir: &Path) -> CoreResult<serde_json::Value> {
    let text = std::fs::read_to_string(dir.join("config.json"))
        .map_err(|e| CoreError::Load(format!("read {}: {e}", dir.join("config.json").display())))?;
    serde_json::from_str(&text)
        .map_err(|e| CoreError::Load(format!("parse {}: {e}", dir.join("config.json").display())))
}

/// Read `eos_token_id` (int or array) from `config.json`; falls back to the Llama-3 stop ids.
pub fn eos_token_ids(dir: &Path) -> Vec<i32> {
    let fallback = vec![128001, 128008, 128009]; // <|end_of_text|>, <|eom_id|>, <|eot_id|>

    // Prefer `generation_config.json` — HF's canonical "how to generate" source, where models put
    // the *generation* EOS set (Qwen3.6's `<|im_end|>` turn-end lives only here, not in config.json).
    if let Some(ids) = read_json(dir, "generation_config.json")
        .as_ref()
        .and_then(|v| parse_token_ids(v.get("eos_token_id")))
    {
        return ids;
    }
    // Otherwise fall back to `config.json` — top-level, then the VLM-nested `text_config`.
    if let Some(v) = read_json(dir, "config.json") {
        if let Some(ids) = parse_token_ids(v.get("eos_token_id"))
            .or_else(|| parse_token_ids(v.get("text_config").and_then(|t| t.get("eos_token_id"))))
        {
            return ids;
        }
    }
    fallback
}

/// Whether a rendered prompt ends with an **unclosed** `<think>` block — i.e. the chat template
/// opened reasoning in the prompt (Qwen3.6's thinking/auto generation prompt) so the model generates
/// inside it. True iff the last `<think>` occurs after the last `</think>` (or there is no close).
fn prompt_opens_thinking(prompt: &str) -> bool {
    match prompt.rfind("<think>") {
        None => false,
        Some(open) => prompt.rfind("</think>").is_none_or(|close| open > close),
    }
}

/// Read and parse a JSON file from a snapshot dir, or `None` if missing/invalid.
fn read_json(dir: &Path, name: &str) -> Option<serde_json::Value> {
    let text = std::fs::read_to_string(dir.join(name)).ok()?;
    serde_json::from_str(&text).ok()
}

/// Parse an `eos_token_id`-style field — a single int or an array of ints — into a non-empty id list.
fn parse_token_ids(v: Option<&serde_json::Value>) -> Option<Vec<i32>> {
    let ids = match v? {
        serde_json::Value::Number(n) => vec![n.as_i64()? as i32],
        serde_json::Value::Array(a) => a
            .iter()
            .filter_map(|x| x.as_i64().map(|x| x as i32))
            .collect(),
        _ => return None,
    };
    (!ids.is_empty()).then_some(ids)
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

#[derive(Clone, Copy)]
enum MlxWorkspaceContract<'a> {
    Eager,
    Chunked,
    Qwen35 {
        config: &'a Qwen35Config,
        prism: bool,
    },
}

/// MLX permits ten completed command buffers to remain in flight and can be building the next
/// buffer before applying backpressure. Price that current buffer plus the in-flight set.
const MLX_EVAL_BUFFER_WINDOW: u64 = 11;
/// Apple-Silicon Metal allocations are rounded to 16-KiB VM pages. Gated DeltaNet retains one
/// independently allocated output row per prompt token until its final concatenate.
const MLX_ALLOCATION_PAGE_BYTES: u64 = 16 * 1024;
/// The recurrence creates five one-element index buffers per step and evaluates every 256 steps.
const QWEN35_RECURRENCE_EVAL_CHUNK: u64 = 256;
const QWEN35_RECURRENCE_INDEX_BUFFERS: u64 = 5;
/// The input/decayed state, two state products, outer product, and updated state can coexist across
/// the evaluator window while one recurrence chunk is being materialized.
const QWEN35_RECURRENCE_STATE_BUFFERS: u64 = 6;

fn checked_sum(values: impl IntoIterator<Item = u64>) -> Option<u64> {
    values
        .into_iter()
        .try_fold(0u64, |sum, value| sum.checked_add(value))
}

fn checked_product(values: impl IntoIterator<Item = u64>) -> Option<u64> {
    values
        .into_iter()
        .try_fold(1u64, |product, value| product.checked_mul(value))
}

fn round_up(value: u64, alignment: u64) -> Option<u64> {
    value
        .checked_add(alignment.checked_sub(1)?)?
        .checked_div(alignment)?
        .checked_mul(alignment)
}

/// Extra complete-decoder workspace above the generic chunked-attention estimate for dense
/// Qwen3.5. Every term corresponds to a prompt-sized tensor retained by the Rust graph or by the
/// pinned MLX evaluator; all tensor elements are charged at four bytes even when the runtime value
/// is BF16.
fn estimate_qwen35_workspace_extra_bytes(
    prompt_tokens: usize,
    config: &Qwen35Config,
    prism: bool,
) -> Option<u64> {
    let prompt = u64::try_from(prompt_tokens).ok()?;
    let nonnegative = |value: i32| u64::try_from(value).ok();
    let hidden = nonnegative(config.hidden_size)?;
    let intermediate = nonnegative(config.intermediate_size)?;
    let query_heads = nonnegative(config.num_heads)?;
    let head_dim = nonnegative(config.head_dim)?;
    let key_heads = nonnegative(config.linear_num_key_heads)?;
    let value_heads = nonnegative(config.linear_num_value_heads)?;
    let key_head_dim = nonnegative(config.linear_key_head_dim)?;
    let value_head_dim = nonnegative(config.linear_value_head_dim)?;
    let conv_kernel = nonnegative(config.linear_conv_kernel_dim)?;
    let key_width = key_heads.checked_mul(key_head_dim)?;
    let value_width = value_heads.checked_mul(value_head_dim)?;
    let conv_width = key_width.checked_mul(2)?.checked_add(value_width)?;
    let attention_width = query_heads.checked_mul(head_dim)?;

    // PrismLinear builds cast -> sign multiply -> Hadamard -> cast graphs independently for each
    // projection. The full-attention block has the wider simultaneous input-width sum; the linear
    // block uses two mixer input rotations plus its output rotation. Four buffers per input width
    // safely covers the F32 chain and the BF16 result even when donation is unavailable.
    let packed_rotation_elements = if prism {
        let linear_widths = checked_sum([hidden.checked_mul(4)?, value_width, intermediate])?;
        let attention_widths =
            checked_sum([hidden.checked_mul(5)?, attention_width, intermediate])?;
        linear_widths.max(attention_widths).checked_mul(4)?
    } else {
        0
    };

    // The ops recurrence expands Q/K from key heads to value heads, casts Q/K/V to F32, and keeps
    // every F32 Y row until concatenate. Count two Q/K tensors plus V and Y.
    let recurrence_elements = checked_sum([
        value_heads.checked_mul(key_head_dim)?.checked_mul(2)?,
        value_width.checked_mul(2)?,
    ])?;

    // Bound the lazy linear-mixer graph through the four-term depthwise convolution, normalized
    // Q/K, gates, and projection outputs. `(2*K + 3)` covers K products, K-1 additions, the
    // concatenated input, convolution output, and activation output.
    let mixer_elements = checked_sum([
        conv_width.checked_mul(conv_kernel.checked_mul(2)?.checked_add(3)?)?,
        value_width,
        key_width.checked_mul(2)?,
        value_heads.checked_mul(2)?,
    ])?;

    // Completed Metal command buffers retain primitive inputs until their callbacks run. Bound one
    // largest prompt-sized output for each possible in-flight/current buffer.
    let largest_output = [
        hidden,
        intermediate,
        conv_width,
        attention_width.checked_mul(2)?,
        value_width,
    ]
    .into_iter()
    .max()?;
    let evaluator_elements = largest_output.checked_mul(MLX_EVAL_BUFFER_WINDOW)?;

    let per_token_elements = checked_sum([
        packed_rotation_elements,
        recurrence_elements,
        mixer_elements,
        evaluator_elements,
    ])?;
    let prompt_workspace = checked_product([prompt, per_token_elements, 4])?;
    // The shared tiled estimate prices one score/mask/softmax set. MLX can retain the same three
    // buffers for the rest of its evaluator window, so price those additional tiles here.
    let attention_window = checked_product([
        prompt,
        prompt.min(SDPA_MAX_FUSED_QLEN as u64),
        query_heads,
        3,
        MLX_EVAL_BUFFER_WINDOW.checked_sub(1)?,
        4,
    ])?;

    // Each retained recurrence row owns a separate allocator buffer; price its VM-page rounding.
    let y_row_bytes = value_width.checked_mul(4)?;
    let row_padding = round_up(y_row_bytes, MLX_ALLOCATION_PAGE_BYTES)?
        .checked_sub(y_row_bytes)?
        .checked_mul(prompt)?;
    // Index buffers are created eagerly for all five slices in a 256-step recurrence chunk.
    let index_buffers = checked_product([
        QWEN35_RECURRENCE_EVAL_CHUNK,
        QWEN35_RECURRENCE_INDEX_BUFFERS,
        MLX_ALLOCATION_PAGE_BYTES,
    ])?;
    let recurrent_state_buffers = checked_product([
        value_width,
        key_head_dim,
        4,
        QWEN35_RECURRENCE_STATE_BUFFERS,
        MLX_EVAL_BUFFER_WINDOW,
    ])?;

    checked_sum([
        prompt_workspace,
        attention_window,
        row_padding,
        index_buffers,
        recurrent_state_buffers,
    ])
}

/// Select the estimate matching the complete decoder execution graph. Generic fused attention uses
/// the shared tiled estimate. Dense Qwen3.5 adds its F32 recurrence, packed-Hadamard, allocator, and
/// lazy-evaluator lifetimes; eager/otherwise-unbounded implementations retain the quadratic model.
fn estimate_mlx_request_bytes(
    prompt_tokens: usize,
    max_new_tokens: u32,
    geometry: LlmMemoryGeometry,
    vision_workspace_bytes: u64,
    mtp_width: u32,
    contract: MlxWorkspaceContract<'_>,
) -> Option<u64> {
    match contract {
        MlxWorkspaceContract::Eager => core_llm::estimate_request_bytes(
            prompt_tokens,
            max_new_tokens,
            geometry,
            vision_workspace_bytes,
            mtp_width,
        ),
        MlxWorkspaceContract::Chunked => core_llm::estimate_chunked_request_bytes(
            prompt_tokens,
            max_new_tokens,
            geometry,
            vision_workspace_bytes,
            mtp_width,
            SDPA_MAX_FUSED_QLEN as usize,
        ),
        MlxWorkspaceContract::Qwen35 { config, prism } => {
            let base = core_llm::estimate_chunked_request_bytes(
                prompt_tokens,
                max_new_tokens,
                geometry,
                vision_workspace_bytes,
                mtp_width,
                SDPA_MAX_FUSED_QLEN as usize,
            )?;
            base.checked_add(estimate_qwen35_workspace_extra_bytes(
                prompt_tokens,
                config,
                prism,
            )?)
        }
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
        // An EOS *id* and a host stop condition (a request `stop` string) are both `Stop` per the
        // contract / OpenAI semantics.
        FinishReason::StopToken | FinishReason::Stopped => CoreFinish::Stop,
        FinishReason::MaxTokens => CoreFinish::Length,
        FinishReason::Cancelled => CoreFinish::Cancelled,
    }
}

/// Bridge an engine error into the contract error, preserving the typed cancellation / capability
/// variants (do not stringify those).
fn to_core(e: crate::Error) -> CoreError {
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
    weightless_vision: Some(weightless_vision),
    weightless_audio: Some(weightless_audio),
};

fn load_registered(spec: &LoadSpec) -> CoreResult<Box<dyn TextLlm>> {
    Ok(Box::new(LlamaProvider::load(spec)?))
}

/// Weightless model-first probe (story 7406): can the `mlx-llama` provider serve the snapshot at
/// `spec.source`? Reads **only** `config.json` and runs the same [`Architecture::from_config`]
/// dispatch the loader uses — it never opens a safetensors shard, so `core-llm`'s `load_for_model`
/// can resolve a provider by model without loading weights. A VLM wrapper (one carrying a
/// `vision_config`) is served **text-only** here unless a dedicated vision provider claims it: a
/// LLaVA snapshot is ceded to `mlx-joycaption`, while a Qwen-VL (`qwen3_5`) wrapper is claimed here
/// (its nested text decoder) so it no longer misroutes to JoyCaption (sc-7626).
pub fn can_load(spec: &LoadSpec) -> bool {
    let dir = Path::new(&spec.source);
    if dir.extension().and_then(|v| v.to_str()) == Some("gguf") {
        let language_is_prism = crate::gguf::GgufFile::open(dir).ok().is_some_and(|g| {
            g.meta_str("general.architecture") == Some("qwen35")
                && g.meta("prism.hadamard.version").is_some()
        });
        if !language_is_prism {
            return false;
        }
        return spec.projector_source.as_deref().is_none_or(|source| {
            crate::gguf::GgufFile::open(source)
                .ok()
                .is_some_and(|projector| crate::prism_vision_gguf::validate(&projector).is_ok())
        });
    }
    if spec.projector_source.is_some() {
        return false;
    }
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
    can_load_value(&v)
}

/// Pure architecture/routing decision over a parsed `config.json` (split out for unit testing without
/// a snapshot on disk).
fn can_load_value(v: &serde_json::Value) -> bool {
    // A VLM wrapper carries a `vision_config`. Cede it to the dedicated vision provider when that
    // provider claims it (LLaVA → `mlx-joycaption`); otherwise serve the nested text decoder
    // text-only (a Qwen-VL `qwen3_5` wrapper).
    if v.get("vision_config").is_some() && crate::joycaption::can_load_value(v) {
        return false;
    }
    Architecture::from_config(v).is_ok()
}

/// **Weightless** per-snapshot vision probe (sc-8077): does `mlx-llama` serve the snapshot at
/// `spec.source` *with* vision? Reads only `config.json` for safetensors snapshots, or the GGUF
/// language and explicitly-associated projector headers (never their tensor payloads), mirroring
/// [`can_load`]. It drives core-llm's pre-load capability gate so a *model-first* vision-required
/// load (`load_for_model_with(spec, with_vision())`) resolves a Qwen-VL wrapper here.
///
/// This is necessary because the provider's STATIC [`provider_descriptor`] must report
/// `supports_vision=false` (the one registration also serves plain text-only checkpoints; vision is
/// only flipped on post-load when the loaded checkpoint carries `model.visual.*`). Without a
/// per-snapshot probe a genuine Qwen3-VL snapshot would be rejected at the gate (the story-D gap).
pub fn weightless_vision(spec: &LoadSpec) -> bool {
    let dir = Path::new(&spec.source);
    if dir.extension().and_then(|v| v.to_str()) == Some("gguf") {
        return spec.projector_source.is_some() && can_load(spec);
    }
    if spec.projector_source.is_some() {
        return false;
    }
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
    weightless_vision_value(&v)
}

/// Pure per-snapshot vision decision over a parsed `config.json` (split out for unit testing). True
/// iff this provider both *claims* the snapshot ([`can_load_value`]) AND the snapshot is a Qwen-VL
/// wrapper that loads its vision tower: a `vision_config` is present and the architecture dispatches
/// to a Qwen-VL decoder (`qwen3_vl` or the Qwen3.6 `qwen3_5` hybrid) — exactly the post-load
/// condition that flips `supports_vision` on in [`LlamaProvider::load`]. A plain text checkpoint (no
/// `vision_config`) or a ceded LLaVA snapshot returns false.
fn weightless_vision_value(v: &serde_json::Value) -> bool {
    if !can_load_value(v) || v.get("vision_config").is_none() {
        return false;
    }
    matches!(
        Architecture::from_config(v),
        Ok(Architecture::Qwen3Vl) | Ok(Architecture::Qwen35)
    )
    // Gemma 4 is deliberately absent: its vision path is loaded but unvalidated and therefore
    // unadvertised (see `gemma4_vision_is_validated`). The probe must agree with the loaded
    // descriptor, or a model-first vision-required load would resolve here and then be rejected.
}

/// **Weightless** per-snapshot audio probe (sc-18772): does `mlx-llama` serve the snapshot at
/// `spec.source` *with* audio? The audio analogue of [`weightless_vision`] — reads only
/// `config.json`, never a weight shard.
///
/// Gemma 4 unified is the only architecture this provider serves that has an audio path at all, and
/// like vision it is per-snapshot: the same registration also serves text-only checkpoints, so the
/// static descriptor stays `supports_audio=false` and this probe is what lets a model-first
/// audio-required load (`load_for_model_with(spec, with_audio())`) resolve here.
pub fn weightless_audio(spec: &LoadSpec) -> bool {
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
    weightless_audio_value(&v)
}

/// Pure per-snapshot audio decision over a parsed `config.json` (split out for unit testing).
fn weightless_audio_value(v: &serde_json::Value) -> bool {
    gemma4_multimodal(v, "audio_config", "audio_token_id")
}

/// Whether this provider claims `v` AND it is a Gemma 4 checkpoint declaring the named front-end
/// block (`vision_config` / `audio_config`) plus the token id that front-end splices at.
///
/// The token-id check is not redundant with the block: a config carrying a `vision_config` but no
/// `image_token_id` cannot be conditioned on an image at all (there is no row to splice into), so
/// advertising vision for it would be exactly the advertised-but-absent case. The load path refuses
/// such a config for the same reason, which keeps probe and loader agreeing.
fn gemma4_multimodal(v: &serde_json::Value, block: &str, token_key: &str) -> bool {
    if !can_load_value(v) || v.get(block).is_none() {
        return false;
    }
    if !matches!(Architecture::from_config(v), Ok(a) if a.is_gemma4()) {
        return false;
    }
    v.get(token_key).and_then(|x| x.as_i64()).is_some()
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    fn frozen_dense_qwen35_config() -> Qwen35Config {
        Qwen35Config::from_json(&json!({
            "model_type": "qwen3_5_text",
            "hidden_size": 5120,
            "num_hidden_layers": 64,
            "intermediate_size": 17_408,
            "num_attention_heads": 24,
            "num_key_value_heads": 4,
            "head_dim": 256,
            "vocab_size": 248_320,
            "linear_num_value_heads": 48,
            "linear_num_key_heads": 16,
            "linear_key_head_dim": 128,
            "linear_value_head_dim": 128,
            "linear_conv_kernel_dim": 4,
            "full_attention_interval": 4
        }))
        .unwrap()
    }

    #[test]
    fn fused_request_estimate_tracks_chunked_attention_and_last_row_logits() {
        // Frozen Qwen3.8 parent geometry. This prompt size reproduces the campaign's long-context
        // scale: eager prompt-squared scores dominate hundreds of GB, while MLX runs eight query
        // rows per fused call and projects one final row to the vocabulary.
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
        let fused =
            estimate_mlx_request_bytes(29_600, 128, geometry, 0, 0, MlxWorkspaceContract::Chunked)
                .unwrap();
        let eager =
            estimate_mlx_request_bytes(29_600, 128, geometry, 0, 0, MlxWorkspaceContract::Eager)
                .unwrap();
        assert!(fused < 36_000_000_000, "bounded MLX peak: {fused}");
        assert!(eager > 400_000_000_000, "quadratic eager peak: {eager}");
        assert_eq!(
            eager,
            core_llm::estimate_request_bytes(29_600, 128, geometry, 0, 0).unwrap(),
            "non-fused paths retain the shared fail-closed estimate"
        );
    }

    #[test]
    fn fused_request_estimate_is_checked_and_preserves_mtp_and_media_costs() {
        let geometry = LlmMemoryGeometry {
            query_heads: 8,
            kv_heads: 2,
            head_dim: 64,
            layers: 4,
            element_bytes: 4,
            hidden_size: 256,
            intermediate_size: 512,
            vocab_size: 1024,
            recurrent_bytes: 4096,
        };
        let plain =
            estimate_mlx_request_bytes(128, 16, geometry, 0, 0, MlxWorkspaceContract::Chunked)
                .unwrap();
        let media = estimate_mlx_request_bytes(
            128,
            16,
            geometry,
            123_456,
            0,
            MlxWorkspaceContract::Chunked,
        )
        .unwrap();
        let mtp =
            estimate_mlx_request_bytes(128, 16, geometry, 0, 3, MlxWorkspaceContract::Chunked)
                .unwrap();
        assert_eq!(media - plain, 123_456);
        assert!(mtp > plain);
        assert!(estimate_mlx_request_bytes(
            usize::MAX,
            u32::MAX,
            geometry,
            0,
            3,
            MlxWorkspaceContract::Chunked,
        )
        .is_none());
    }

    #[test]
    fn qwen35_prism_estimate_covers_frozen_allocator_peaks_and_rejects_remote_long_context() {
        let config = frozen_dense_qwen35_config();
        let geometry = LlmMemoryGeometry {
            query_heads: 24,
            kv_heads: 4,
            head_dim: 256,
            layers: 64,
            element_bytes: 4,
            hidden_size: 5120,
            intermediate_size: 17_408,
            vocab_size: 248_320,
            recurrent_bytes: 207_618_048,
        };
        let contract = MlxWorkspaceContract::Qwen35 {
            config: &config,
            prism: true,
        };
        let context_64 = estimate_mlx_request_bytes(1_187, 128, geometry, 0, 0, contract).unwrap();
        let context_512 = estimate_mlx_request_bytes(9_251, 128, geometry, 0, 0, contract).unwrap();
        let context_2048 =
            estimate_mlx_request_bytes(36_899, 128, geometry, 0, 0, contract).unwrap();

        // Frozen campaign 35461924246: subtract the stable loaded-model active residency
        // (8,551,959,848) from each new process-global allocator high-water mark.
        assert!(context_64 >= 4_091_010_468, "estimate: {context_64}");
        assert!(context_512 >= 22_885_373_536, "estimate: {context_512}");
        assert_eq!(context_64, 4_152_141_184);
        assert_eq!(context_512, 28_934_038_912);
        assert_eq!(context_2048, 113_900_545_408);
        assert!(context_64 < context_512 && context_512 < context_2048);
        assert!(
            core_llm::admit_request_memory(context_2048, 47_922_610_176).is_err(),
            "the 36,899-token request must not be admitted on the failed remote's observed budget"
        );
    }

    #[test]
    fn qwen35_workspace_estimate_is_checked_and_keeps_ordinary_requests_available() {
        let config = frozen_dense_qwen35_config();
        let geometry = LlmMemoryGeometry {
            query_heads: 24,
            kv_heads: 4,
            head_dim: 256,
            layers: 64,
            element_bytes: 4,
            hidden_size: 5120,
            intermediate_size: 17_408,
            vocab_size: 248_320,
            recurrent_bytes: 207_618_048,
        };
        let contract = MlxWorkspaceContract::Qwen35 {
            config: &config,
            prism: true,
        };
        let ordinary = estimate_mlx_request_bytes(128, 128, geometry, 0, 0, contract).unwrap();
        assert!(ordinary < 2_000_000_000, "ordinary estimate: {ordinary}");
        assert!(core_llm::admit_request_memory(ordinary, 8_000_000_000).is_ok());
        assert!(
            estimate_mlx_request_bytes(usize::MAX, u32::MAX, geometry, u64::MAX, 3, contract,)
                .is_none()
        );
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
    fn reasoning_controls_are_advertised_only_when_template_names_them() {
        let dir = tempfile::tempdir().unwrap();
        std::fs::write(
            dir.path().join("chat_template.jinja"),
            "{{ enable_thinking }} {{ tool_call }}",
        )
        .unwrap();
        let (_, thinking, effort, preserve, tools) = load_chat_template(dir.path());
        assert!(thinking);
        assert!(!effort);
        assert!(!preserve);
        assert!(tools);

        std::fs::write(
            dir.path().join("chat_template.jinja"),
            "{{ enable_thinking }} {{ reasoning_effort }} {{ preserve_thinking }}",
        )
        .unwrap();
        let (_, thinking, effort, preserve, tools) = load_chat_template(dir.path());
        assert!(thinking);
        assert!(effort);
        assert!(preserve);
        assert!(!tools);
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

    fn qwen36_wrapper() -> serde_json::Value {
        json!({
            "architectures": ["Qwen3_5ForConditionalGeneration"],
            "model_type": "qwen3_5",
            "text_config": { "model_type": "qwen3_5_text" },
            "vision_config": { "model_type": "qwen3_5", "depth": 27 }
        })
    }
    fn llava_wrapper() -> serde_json::Value {
        json!({
            "architectures": ["LlavaForConditionalGeneration"],
            "model_type": "llava",
            "text_config": { "model_type": "llama" },
            "vision_config": { "model_type": "siglip_vision_model" }
        })
    }
    fn qwen3vl_wrapper() -> serde_json::Value {
        json!({
            "architectures": ["Qwen3VLForConditionalGeneration"],
            "model_type": "qwen3_vl",
            "text_config": { "model_type": "qwen3_vl_text" },
            "vision_config": { "model_type": "qwen3_vl", "depth": 27 }
        })
    }

    #[test]
    fn text_provider_claims_qwen36_but_cedes_llava() {
        // Qwen3.6 (Qwen-VL wrapper): claimed by the text provider, served text-only.
        assert!(can_load_value(&qwen36_wrapper()));
        // Qwen3-VL (Qwen-VL wrapper): also claimed by `mlx-llama` (its nested standard Qwen3 decoder).
        assert!(can_load_value(&qwen3vl_wrapper()));
        // LLaVA: ceded to the JoyCaption vision provider.
        assert!(!can_load_value(&llava_wrapper()));
        // Plain text models (no vision_config) are unaffected.
        assert!(can_load_value(
            &json!({ "architectures": ["Qwen3ForCausalLM"], "model_type": "qwen3" })
        ));
        assert!(can_load_value(
            &json!({ "architectures": ["LlamaForCausalLM"], "model_type": "llama" })
        ));
    }

    #[test]
    fn weightless_vision_advertises_qwen_vl_wrappers_only() {
        // The sc-8077 weightless vision gate: `mlx-llama` advertises vision (pre-load, config.json
        // only) for a Qwen-VL wrapper so a model-first vision-required load resolves here.
        // Qwen3-VL: a vision_config + qwen3_vl arch ⇒ vision-capable.
        assert!(weightless_vision_value(&qwen3vl_wrapper()));
        // Qwen3.6 hybrid (qwen3_5) Qwen-VL wrapper: likewise vision-capable.
        assert!(weightless_vision_value(&qwen36_wrapper()));
        // A LLaVA snapshot is ceded to JoyCaption (can_load=false here) ⇒ NOT advertised, so a
        // Qwen3-VL load can never be mistaken for / misrouted via this provider's vision path.
        assert!(!weightless_vision_value(&llava_wrapper()));
        // Plain text checkpoints (no vision_config) ⇒ NOT vision-capable (the static descriptor
        // already reports supports_vision=false; the probe must not flip it on).
        assert!(!weightless_vision_value(
            &json!({ "architectures": ["Qwen3ForCausalLM"], "model_type": "qwen3" })
        ));
        assert!(!weightless_vision_value(
            &json!({ "architectures": ["LlamaForCausalLM"], "model_type": "llama" })
        ));
    }

    /// Locate a staged Qwen3-VL-8B-Instruct snapshot for the chat-template oracle from the explicit
    /// passed-in `QWEN3VL_SNAPSHOT` env path. Inference never self-fetches or derives a cache location
    /// (epic 13657). `None` ⇒ the gated tests self-skip cleanly (CI sets the var for this story).
    fn qwen3vl_snapshot_dir() -> Option<std::path::PathBuf> {
        let path = std::path::PathBuf::from(std::env::var("QWEN3VL_SNAPSHOT").ok()?);
        path.exists().then_some(path)
    }

    /// Build the representative chat-message sets the oracle pins (text-only, system+user, single
    /// image, multi-image/mixed, multi-turn). Image content is a `Content::Image` (a 1×1 black pixel
    /// placeholder); the byte-match exercises the same `substitute_image_placeholders` path the
    /// provider uses, so the rendered single `<|image_pad|>` per image must match HF.
    fn oracle_messages(case: &str) -> Vec<Message> {
        let img = || Content::Image(ImageRef::new(1, 1, vec![0, 0, 0]).unwrap());
        let user = |content: Vec<Content>| Message {
            role: core_llm::Role::User,
            content,
            thinking: None,
            tool_calls: Vec::new(),
        };
        let sys = |t: &str| Message {
            role: core_llm::Role::System,
            content: vec![Content::text(t)],
            thinking: None,
            tool_calls: Vec::new(),
        };
        let asst = |t: &str| Message {
            role: core_llm::Role::Assistant,
            content: vec![Content::text(t)],
            thinking: None,
            tool_calls: Vec::new(),
        };
        match case {
            "text_only" => vec![user(vec![Content::text("What is the capital of France?")])],
            "system_user_text" => {
                vec![
                    sys("You are a helpful assistant."),
                    user(vec![Content::text("Hello!")]),
                ]
            }
            "single_image" => vec![user(vec![img(), Content::text("Describe this image.")])],
            "multi_image_mixed" => vec![user(vec![
                Content::text("Compare:"),
                img(),
                Content::text("and"),
                img(),
                Content::text("please."),
            ])],
            "multi_turn" => vec![
                user(vec![Content::text("Hi")]),
                asst("Hello there!"),
                user(vec![Content::text("How are you?")]),
            ],
            other => panic!("unknown oracle case {other}"),
        }
    }

    /// Byte-match oracle (sc-8075 AC #1): the engine's chat-template + image-placeholder + tokenize
    /// path must reproduce the pinned HF `apply_chat_template` prompt token ids **exactly** for the
    /// representative messages — text-only, single-image, and multi-image/mixed. The single
    /// `<|image_pad|>` (151655) per image is what the processor emits *before* patch-count expansion;
    /// these fixtures pin that pre-expansion prompt. Self-skips cleanly when the snapshot is absent.
    #[test]
    fn chat_template_byte_matches_hf_processor() {
        let Some(dir) = qwen3vl_snapshot_dir() else {
            eprintln!("skipping: Qwen3-VL-8B snapshot not present (set QWEN3VL_SNAPSHOT)");
            return;
        };
        let oracle: serde_json::Value = serde_json::from_str(include_str!(
            "models/testdata/qwen3vl_chat_template_oracle.json"
        ))
        .expect("parse chat-template oracle");

        // Sanity: the dispatch and config see Qwen3-VL (and parse the nested 256K-context text decoder).
        let cfg_value = read_config_value(&dir).expect("read config.json");
        assert_eq!(
            Architecture::from_config(&cfg_value).unwrap(),
            Architecture::Qwen3Vl,
            "snapshot must dispatch to Qwen3-VL"
        );
        let cfg = ModelConfig::from_json(&cfg_value).expect("parse Qwen3-VL text config");
        assert_eq!(cfg.architecture, Architecture::Qwen3Vl);
        assert_eq!(cfg.max_position_embeddings, 262144, "256K context");

        let (template, _, _, _, _) = load_chat_template(&dir);
        let tokenizer = Tokenizer::from_file(dir.join("tokenizer.json")).expect("load tokenizer");

        for (case, expected) in oracle["cases"].as_object().unwrap() {
            let want: Vec<i32> = expected["ids"]
                .as_array()
                .unwrap()
                .iter()
                .map(|x| x.as_i64().unwrap() as i32)
                .collect();
            // Drive the exact provider path: substitute image content with the Qwen-VL placeholder,
            // render the model's own Jinja template (add_generation_prompt), then tokenize without
            // auto special tokens.
            let messages = oracle_messages(case);
            let substituted = substitute_vision_placeholders(&messages, 2)
                .expect("the oracle cases carry no audio, so substitution cannot fail");
            let prompt = template
                .render_with(
                    &substituted,
                    &RenderOptions {
                        add_generation_prompt: true,
                        enable_thinking: None,
                        reasoning_effort: None,
                        preserve_thinking: None,
                        tools: &[],
                    },
                )
                .expect("render");
            assert_eq!(
                prompt,
                expected["text"].as_str().unwrap(),
                "case {case}: rendered prompt string must byte-match HF"
            );
            let got: Vec<i32> = tokenizer
                .encode(&prompt, false)
                .expect("encode")
                .into_iter()
                .map(|id| id as i32)
                .collect();
            assert_eq!(
                got, want,
                "case {case}: tokenized prompt ids must byte-match HF processor"
            );
        }
    }

    #[test]
    fn prompt_opens_thinking_matches_template_modes() {
        // Qwen3.6 thinking/auto generation prompt: opens the block, leaves it unclosed.
        assert!(prompt_opens_thinking("<|im_start|>assistant\n<think>\n"));
        // Disabled mode renders a *closed* empty block: must not prime.
        assert!(!prompt_opens_thinking(
            "<|im_start|>assistant\n<think>\n\n</think>\n\n"
        ));
        // A prior closed reasoning turn followed by a fresh open block still opens.
        assert!(prompt_opens_thinking(
            "<think>\nold\n</think>\n\nq<|im_start|>assistant\n<think>\n"
        ));
        // No reasoning markers at all (non-thinking template).
        assert!(!prompt_opens_thinking("<|im_start|>assistant\n"));
    }

    // --- Qwen3-VL video Text–Timestamp-Alignment oracle (tools/gen_qwen3vl_video_oracle.py) --------

    fn qwen3vl_video_oracle() -> serde_json::Value {
        serde_json::from_str(include_str!("models/testdata/qwen3vl_video_oracle.json")).unwrap()
    }

    /// **Merged per-frame timestamps match `Qwen3VLProcessor._calculate_timestamps`.** Given the
    /// sampled `frames_indices` + `fps`, the per-temporal-patch averaged timestamps must equal the HF
    /// reference exactly — the values that feed the `<{t:.1f} seconds>` Text–Timestamp-Alignment tags.
    #[test]
    fn video_merged_timestamps_match_hf_reference() {
        let j = qwen3vl_video_oracle();
        let fps = j["fps"].as_f64().unwrap() as f32;
        let temporal = j["temporal_patch_size"].as_u64().unwrap() as usize;
        let indices: Vec<f32> = j["frames_indices"]
            .as_array()
            .unwrap()
            .iter()
            .map(|x| x.as_f64().unwrap() as f32)
            .collect();
        // Per-sample timestamps are `idx / fps` (matching the reference, which averages these).
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
    fn invalid_public_video_timestamps_fail_before_prompt_rendering() {
        let frame = || ImageRef::new(1, 1, vec![0, 0, 0]).unwrap();
        for timestamps in [vec![0.0, f32::NAN], vec![0.5, 0.25], vec![-0.1, 0.0]] {
            let video = VideoRef {
                frames: vec![frame(), frame()],
                timestamps,
            };
            let messages = vec![Message {
                role: core_llm::Role::User,
                content: vec![Content::Video(video)],
                thinking: None,
                tool_calls: Vec::new(),
            }];
            let error = substitute_vision_placeholders(&messages, 2)
                .expect_err("invalid timestamp sequence accepted");
            assert!(error.to_string().contains("timestamp"));
        }
        let count_mismatch = vec![Message {
            role: core_llm::Role::User,
            content: vec![Content::Video(VideoRef {
                frames: vec![frame(), frame()],
                timestamps: vec![0.0],
            })],
            thinking: None,
            tool_calls: Vec::new(),
        }];
        assert!(substitute_vision_placeholders(&count_mismatch, 2)
            .unwrap_err()
            .to_string()
            .contains("one timestamp per frame"));
    }

    #[test]
    fn expanded_text_image_video_and_mtp_budgets_share_the_context_gate() {
        for expanded_prompt in [48usize, 52, 60] {
            validate_context_window(64, expanded_prompt, (64 - expanded_prompt) as u32).unwrap();
            let error =
                validate_context_window(64, expanded_prompt, (64 - expanded_prompt + 1) as u32)
                    .expect_err("one-token context overflow accepted");
            assert!(error.to_string().contains("exceeds context window 64"));
        }
        assert!(validate_context_window(usize::MAX, usize::MAX, 1)
            .unwrap_err()
            .to_string()
            .contains("overflow"));
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
        let expanded =
            crate::models::qwen35::expand_vision_placeholders(&raw, vid, &counts).unwrap();
        assert_eq!(expanded, expanded_hf, "expanded video ids vs HF processor");
    }

    /// **The video M-RoPE positions over the oracle grid are well-formed and per-frame-reset.** Feed
    /// the expanded video id stream + the `video_grid_thw` through `mrope_positions_mm`: the temporal
    /// row must reset to the frame's cursor at each frame (Qwen3-VL's synthetic time axis splits each
    /// `[t,h,w]` into `t` per-frame `[1,h,w]` blocks). The HF-pinned exact-row check lives in
    /// `qwen35::qwen3vl_mrope_video_matches_hf_reference`; here we confirm the provider's video grid
    /// drives the same path consistently.
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
        let img = j["video_token_id"].as_i64().unwrap() as i32 - 1; // a distinct unused image id
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
}
