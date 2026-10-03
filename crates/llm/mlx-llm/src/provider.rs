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

use std::cell::{OnceCell, RefCell};
use std::path::Path;
use std::time::Instant;

use core_llm::{
    AudioRef, Channel, ChatTemplate, Constraint, ConstraintDecodeTable, ConstraintKind, Content,
    Error as CoreError, FinishReason as CoreFinish, ImageRef, IncrementalDetok, JinjaChatTemplate,
    JsonConstraint, Llama3Template, LlmMemoryGeometry, LoadSpec, Message, ModelSamplingDefaults,
    ProposerCapabilities, ProposerKind, Quantize, ReasoningEffort, RenderOptions,
    Result as CoreResult, Sampling, SpeculativePlan, SpeculativeProposer, StopMatcher,
    StreamEvent as CoreEvent, TextLlm, TextLlmCapabilities, TextLlmDescriptor, TextLlmOutput,
    TextLlmRequest, ThinkingSegmenter, Tokenizer, ToolCallSegmenter, Usage, VideoRef,
};

use core_llm::DraftReport;

use crate::config::{Architecture, ModelConfig};
use crate::decode::{
    generate_speculative, prefill_restored, Boundary, ConstraintMask, Decode, DraftModelProposer,
    EngineOptions, FinishReason, GenerationConfig, MtpProposer, NgramProposer, NoProposer,
    PrefixCache, PrefixPrefill, PrefixStats, Proposer, Qwen35MtpMultimodalPrompt,
    RewindableConstraintMask, SpeculativePrompt, SpeculativeTarget, StreamEvent,
};
use crate::image::Qwen35ImageProcessor;
use crate::models::gemma4_mm;
use crate::models::{
    CausalLm, Gemma4Layout, Gemma4Mm, Gemma4MmConfig, Qwen35Cache, Qwen35Config, Qwen35Model,
    Qwen35VisionConfig, Qwen35VisionModel, VlmDecode,
};
use crate::primitives::attention::SDPA_SCORE_TILE_QLEN;
use crate::primitives::kv_cache::{ContiguousKvCache, KvCache};
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

    /// Bytes a forward's promoted weight copies hold at once ([`promoted_weight_bytes`]), when
    /// the decoder's activations are wider than its weights ([`CausalLm::activations_promote`]);
    /// `Some(0)` otherwise.
    fn promoted_request_bytes(&self) -> Option<u64> {
        match self {
            Decoder::Causal(m) => match m.promoted_weight_elements() {
                Some((head, largest)) => promoted_weight_bytes(head, largest),
                None => Some(0),
            },
            Decoder::Qwen35(_) => Some(0),
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
    /// The cross-turn prefix cache (sc-24437): one per loaded model, every decoder family, under
    /// the byte budget [`load`](Self::load) admitted.
    prefix: RefCell<PrefixCache>,
    /// The resident draft model for `draft_model` speculation (sc-24436), present iff the load
    /// named a compatible one ([`LoadSpec::draft_source`]); `draft_model` is advertised exactly
    /// then. A second decoder with its own cache per request, driven as a proposer by the engine.
    draft: Option<ResidentDraft>,
    /// What the load settled: the requested weight format, what became of a named draft
    /// (resident, or refused with the reason), the prefix cache's admitted budget, and every other
    /// optional accelerator — the companion MTP head (sc-24444) — requested but not attached,
    /// named in `fallbacks`. Read through [`TextLlm::load_report`].
    load_report: core_llm::LoadReport,
    /// The speculative option a request that leaves it unset runs with (E5, sc-24446): the MLX
    /// defaults-table row ([`core_llm::defaults::MLX`]) unless
    /// [`set_speculative_default`](LlamaProvider::set_speculative_default) overrides it.
    speculative_default: core_llm::Speculative,
}

/// A draft model resident beside its target (sc-24436).
struct ResidentDraft {
    /// The draft decoder.
    model: Decoder,
    /// Leading draft ids it may propose — its tokenizer's tokens
    /// ([`core_llm::draft_compatibility`]); a padding row never becomes a draft.
    proposable: usize,
    /// The target's logits width, which the draft's logits are shaped to.
    width: usize,
    /// The draft's own context window (`0`: unbounded): a request reaching past it runs `auto`
    /// instead ([`core_llm::fit_draft_context`], E2).
    context: usize,
}

/// What a load does with its named draft model (sc-24436), decided before any weight is read.
enum DraftPlan {
    /// No draft named.
    None,
    /// Admitted beside the target: load it from this spec.
    Load(LoadSpec),
    /// Refused before loading (unpriceable, or no room beside the target).
    Refused(DraftReport),
}

impl DraftPlan {
    /// Price a named draft beside the target (E7): its load estimate is admitted together with
    /// the target's. `Err` only when the target alone does not fit — a draft never fails the load
    /// (E2); a draft that cannot be priced or does not fit beside the target is refused by name.
    fn admit(spec: &LoadSpec, target_required: u64, available: u64) -> CoreResult<Self> {
        let Some(source) = spec.draft_source.as_deref() else {
            core_llm::admit_load_memory(target_required, available)?;
            return Ok(DraftPlan::None);
        };
        let draft_spec = draft_load_spec(spec, source);
        let draft_required = match crate::load_memory::required_bytes(&draft_spec) {
            Ok(bytes) => bytes,
            Err(e) => {
                core_llm::admit_load_memory(target_required, available)?;
                return Ok(DraftPlan::Refused(DraftReport::refused(
                    source,
                    core_llm::draft_unpriced_refusal(e),
                )));
            }
        };
        Ok(
            match core_llm::admit_draft_load(target_required, draft_required, available)? {
                None => DraftPlan::Load(draft_spec),
                Some(why) => DraftPlan::Refused(DraftReport::refused(source, why)),
            },
        )
    }
}

/// The load spec of a named draft: its own source, text-only, at the target's load-time tier
/// (a quantization tier covers every model the load makes resident) and naming no draft itself.
fn draft_load_spec(spec: &LoadSpec, source: &str) -> LoadSpec {
    LoadSpec {
        source: source.to_string(),
        projector_source: None,
        quantize: spec.quantize,
        cuda_graphs: spec.cuda_graphs,
        // A draft keeps no cross-turn prefix cache of its own (sc-24437), and no companion head.
        prefix_cache_bytes: Some(0),
        draft_source: None,
        mtp_head_source: None,
    }
}

/// The same Qwen3.5/3.8 decoder appears under `model.language_model` in VLM snapshots and
/// directly under `model` in text-only finetunes. Reject ambiguous or incomplete layouts.
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

/// The MLX projection format of a load-time quantization tier. NVFP4 is a CUDA sm_120
/// capability (sc-24135): refused by name, never substituted.
pub(crate) fn quant_spec(quantize: Quantize) -> CoreResult<QuantSpec> {
    match quantize {
        Quantize::Q4 => Ok(QuantSpec::q4()),
        Quantize::Q8 => Ok(QuantSpec::q8()),
        Quantize::Nvfp4 => Err(CoreError::Unsupported(
            "nvfp4: NVFP4 projections need a CUDA device with compute capability >= sm_120; the \
             MLX backend has no NVFP4 GEMM"
                .into(),
        )),
    }
}

/// Materialize a lazily built decoder group by group ([`Weights::materialize_groups`]) and
/// refuse the load if any source it read went unconsumed (sc-24446): such a source is held
/// resident beside the model, outside what `load_memory::required_bytes` prices — an
/// enumeration gap in the decoder's `param_groups`, never a property of the checkpoint.
fn materialize_decoder(weights: &mut Weights, groups: &[Vec<Array>]) -> CoreResult<()> {
    let report = weights.materialize_groups(groups).map_err(to_core)?;
    if report.leftover > 0 {
        return Err(CoreError::Load(format!(
            "load materialization: {} source tensor(s) the decoder read were consumed by none of \
             its arrays; the load would hold them outside its admitted bound",
            report.leftover
        )));
    }
    Ok(())
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
        // sc-24135: NVFP4 is refused by name before admission, so a memory refusal cannot mask it.
        let quant = spec.quantize.map(quant_spec).transpose()?;
        let required = crate::load_memory::required_bytes(spec)?;
        let available = core_llm::effective_memory_budget(
            core_llm::available_host_memory_bytes(),
            load_memory_budget()?,
        )?;
        let draft = DraftPlan::admit(spec, required, available)?;
        let mut admitted = match &draft {
            DraftPlan::Load(draft_spec) => {
                required.saturating_add(crate::load_memory::required_bytes(draft_spec)?)
            }
            DraftPlan::None | DraftPlan::Refused(_) => required,
        };
        // E7: a companion head is resident weights; it is admitted on top of the target and any
        // admitted draft, and a head that does not fit is refused by name while the target still
        // loads (E2).
        let mut fallbacks = Vec::new();
        let mtp_head = spec.mtp_head_source.as_deref().and_then(|source| {
            admit_companion_head(Path::new(source), admitted, available)
                .map_err(|reason| fallbacks.push(reason))
                .ok()
        });
        if let Some((_, head_bytes)) = mtp_head {
            admitted = admitted.saturating_add(head_bytes);
        }
        // The cross-turn prefix cache's budget (sc-24437, E7): what the load asked for, clamped to
        // the headroom this admission leaves beside the target, any admitted draft and any
        // admitted companion head, so the cache can never push the load past it.
        let prefix_budget = core_llm::prefix_cache_budget(
            core_llm::DecodeBackend::Mlx,
            spec.prefix_cache_bytes,
            admitted,
            available,
        );

        let mut provider = Self::load_admitted(spec, quant)?.with_prefix_budget(prefix_budget);
        provider.load_report.requested = spec.quantize;
        // The load's own fallbacks (a refused companion head) join what the decoder's load named
        // (a configured native MTP head the snapshot does not carry).
        provider.load_report.fallbacks.splice(0..0, fallbacks);
        provider.attach_mtp_head(mtp_head.map(|(head, _)| head));
        provider.load_report.record_prefix_budget(
            core_llm::DecodeBackend::Mlx,
            spec.prefix_cache_bytes,
            prefix_budget,
        );
        match draft {
            DraftPlan::None => {}
            DraftPlan::Refused(report) => provider.load_report.record_draft(report),
            DraftPlan::Load(draft_spec) => provider.attach_draft(&draft_spec),
        }
        Ok(provider)
    }

    /// Load the draft `spec` names beside this target and keep it only if it can propose for it
    /// ([`core_llm::draft_compatibility`]): resident — `draft_model` advertised — or refused
    /// with the reason in the load report, the target unaffected either way (sc-24436, E2).
    fn attach_draft(&mut self, spec: &LoadSpec) {
        let source = spec.source.clone();
        let target_logits = self.model.memory_geometry().vocab_size as usize;
        // The tokenizer is checked before any draft weight is touched when the draft ships one.
        let early = core_llm::draft_tokenizer_refusal(&self.tokenizer, Path::new(&source));
        let outcome = match early {
            Some(why) => Err(why),
            None => match spec.quantize.map(quant_spec).transpose() {
                Err(e) => Err(core_llm::draft_refusal(e)),
                Ok(quant) => match Self::load_admitted(spec, quant) {
                    Err(e) => Err(core_llm::draft_load_refusal(e)),
                    Ok(draft) => core_llm::draft_compatibility(
                        &self.tokenizer,
                        target_logits,
                        &draft.tokenizer,
                        draft.model.memory_geometry().vocab_size as usize,
                    )
                    .map(|proposable| ResidentDraft {
                        proposable,
                        width: target_logits,
                        context: draft.descriptor.capabilities.max_context_tokens,
                        model: draft.model,
                    }),
                },
            },
        };
        let (draft, report) = core_llm::settle_draft(
            source,
            outcome,
            &mut self.descriptor.capabilities,
            &mut self.load_report.fallbacks,
            recommended_depths(),
        );
        self.draft = draft;
        self.load_report.draft = Some(report);
    }

    /// [`load`](Self::load) after admission: the target decoder, its tokenizer, template and
    /// multimodal front-ends, with no draft.
    fn load_admitted(spec: &LoadSpec, quant: Option<QuantSpec>) -> CoreResult<Self> {
        let dir = Path::new(&spec.source);
        if dir.extension().and_then(|v| v.to_str()) == Some("gguf") {
            return Self::load_prism_gguf(spec, dir);
        }
        // Read config.json once to dispatch the architecture: the hybrid Qwen3.6 (`qwen3_5`) decoder
        // has its own config/weights path (and `ModelConfig` deliberately rejects it).
        let cfg_value = read_config_value(dir)?;
        let arch = Architecture::from_config(&cfg_value).map_err(to_core)?;
        let mut weights = Weights::from_dir(dir).map_err(to_core)?;
        let is_prism =
            cfg_value.get("model_type").and_then(|v| v.as_str()) == Some("prism_hadamard_qwen35");
        if is_prism && quant.is_some() {
            return Err(CoreError::Load(
                "Prism snapshots are already packed 2-bit and reject load-time Q4/Q8".into(),
            ));
        }

        let mut prism_vision_weights = None;
        // What the decoder's own load could not attach (E2): a configured native MTP head the
        // snapshot does not carry, or a variant this runtime does not run.
        let mut load_fallbacks = Vec::new();
        let (model, mut descriptor) = if arch == Architecture::Qwen35 {
            let qcfg = Qwen35Config::from_json(&cfg_value).map_err(to_core)?;
            let mut descriptor = descriptor_for_qwen35(&qcfg);
            let m = if is_prism {
                let pack = PrismMlxPack::from_dir(dir, &cfg_value, &weights).map_err(to_core)?;
                descriptor.family = "prism_hadamard_qwen35".into();
                let model =
                    Qwen35Model::build_prism_lazy(&weights, qcfg, &pack).map_err(to_core)?;
                materialize_decoder(&mut weights, &model.param_groups())?;
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
                let prefix = qwen35_dense_prefix(|key| weights.contains(key))?;
                let model =
                    Qwen35Model::build_lazy(&weights, prefix, qcfg, quant).map_err(to_core)?;
                // Group by group: read, verify, convert, release (sc-24446) — the order
                // `load_memory::required_bytes` prices.
                materialize_decoder(&mut weights, &model.param_groups())?;
                model
            };
            // A configured head that was not built is named and not advertised: the loaded
            // config (its MTP layers cleared) settles the advertisement.
            if let Some(why) = m.mtp_fallback() {
                load_fallbacks.push(why.to_string());
                descriptor.capabilities = descriptor_for_qwen35(m.config()).capabilities;
            }
            (Decoder::Qwen35(m), descriptor)
        } else {
            let cfg = ModelConfig::from_json(&cfg_value).map_err(to_core)?;
            let descriptor = descriptor_for(&cfg);
            let m = CausalLm::build_lazy(&weights, "", cfg, quant).map_err(to_core)?;
            materialize_decoder(&mut weights, &m.param_groups())?;
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
            prefix: RefCell::new(PrefixCache::with_budget(0)),
            speculative_default: core_llm::defaults::MLX.speculative,
            draft: None,
            load_report: core_llm::LoadReport {
                fallbacks: load_fallbacks,
                ..core_llm::LoadReport::default()
            },
        })
    }

    /// Attach an admitted companion MTP head (sc-24444). It attaches only to a Qwen3.5/3.8-family
    /// target (dense `qwen3_5` or Prism/Bonsai) without a native head, and only when its geometry
    /// matches; anything else is a named load fallback and the target stays exactly as loaded.
    fn attach_mtp_head(&mut self, head: Option<&Path>) {
        let Some(head) = head else { return };
        let outcome = match &mut self.model {
            Decoder::Qwen35(model) => model
                .attach_companion_mtp(head)
                .map(|()| speculative_max_depth([qwen35_attention_geometry(model.config())]))
                .map_err(|e| format!("mtp_head: {e}")),
            Decoder::Causal(_) => Err(format!(
                "{}; this model's family is `{}`",
                core_llm::COMPANION_MTP_FAMILY_REFUSAL,
                self.descriptor.family
            )),
        };
        match outcome {
            // The same backend-true depth a native head advertises (sc-24438): a companion head
            // runs the identical one-layer predictor through the identical verify path.
            Ok(max_depth) => self
                .descriptor
                .capabilities
                .advertise_mtp(max_depth, recommended_depths().mtp),
            Err(reason) => self
                .load_report
                .fallbacks
                .push(core_llm::companion_head_fallback(&reason, head)),
        }
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
            prefix: RefCell::new(PrefixCache::with_budget(
                core_llm::defaults::MLX.prefix_cache_bytes,
            )),
            speculative_default: core_llm::defaults::MLX.speculative,
            draft: None,
            load_report: core_llm::LoadReport::default(),
        })
    }

    /// Whether the loaded model's projections are quantized.
    pub fn is_quantized(&self) -> bool {
        self.model.is_quantized()
    }

    /// The cross-turn prefix cache's byte budget (sc-24437) — what the load admitted for it.
    pub fn prefix_cache_budget(&self) -> u64 {
        self.prefix.borrow().budget_bytes()
    }

    /// Bytes the prefix cache holds now (never more than [`prefix_cache_budget`](Self::prefix_cache_budget)).
    pub fn prefix_cache_resident_bytes(&self) -> u64 {
        self.prefix.borrow().resident_bytes()
    }

    /// The prefix cache's cumulative reuse accounting.
    pub fn prefix_cache_stats(&self) -> PrefixStats {
        self.prefix.borrow().stats()
    }

    /// Settle the prefix cache budget the load admitted: an empty cache of `budget` bytes, and
    /// the load report naming it.
    fn with_prefix_budget(mut self, budget: u64) -> Self {
        self.prefix.replace(PrefixCache::with_budget(budget));
        self.load_report.prefix_cache_bytes = Some(budget);
        self
    }

    /// The speculative option a request that leaves it unset runs with on this provider: the MLX
    /// defaults-table default ([`core_llm::DecodeDefaults::speculative`], epic sc-24432 E5)
    /// unless [`set_speculative_default`](Self::set_speculative_default) replaced it.
    pub fn speculative_default(&self) -> core_llm::Speculative {
        self.speculative_default
    }

    /// Replace the speculative option an unset request runs with on this provider (a deployment
    /// choice, e.g. a server flag); a request's own `speculative` / legacy `mtp` still wins.
    pub fn set_speculative_default(&mut self, speculative: core_llm::Speculative) {
        self.speculative_default = speculative;
    }

    /// Whether a Prism VLM's dense vision tensors were retained for the multimodal adapter.
    pub fn has_deferred_prism_vision(&self) -> bool {
        self._prism_vision_weights.is_some()
    }

    /// Assemble a provider from already-loaded parts with a default Llama-3 template (used by tests
    /// and converters that don't have a `tokenizer_config.json`).
    pub fn from_parts(model: CausalLm, tokenizer: Tokenizer, stop_tokens: Vec<i32>) -> Self {
        Self {
            descriptor: {
                let mut d = provider_descriptor();
                d.capabilities.speculative = vec![prompt_lookup_capabilities(
                    speculative_max_depth(causal_attention_geometries(model.config())),
                )];
                d
            },
            model: Decoder::Causal(model),
            tokenizer,
            template: Box::new(Llama3Template),
            stop_tokens,
            constraint_table: OnceCell::new(),
            vision: None,
            gemma4: None,
            _prism_vision_weights: None,
            prefix: RefCell::new(PrefixCache::with_budget(
                core_llm::defaults::MLX.prefix_cache_bytes,
            )),
            speculative_default: core_llm::defaults::MLX.speculative,
            draft: None,
            load_report: core_llm::LoadReport::default(),
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
        // Nothing was accepted since the checkpoint: the state is already there, and a rebuild +
        // replay of every accepted token would make each constrained step O(n).
        if checkpoint == self.accepted.len() {
            return;
        }
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

    fn load_report(&self) -> Option<core_llm::LoadReport> {
        Some(self.load_report.clone())
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
        // The fused-primitive routes the whole request builds — the vision tower and a prefill
        // run outside the engine included — are what its report names (E3).
        let fused_start = crate::primitives::fused::fused_tally();

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
        // The backend-neutral speculative resolution (core-llm, sc-24433) against what this
        // provider advertises, then the proposer this request's shape can run on the engine
        // (sc-24434). Anything less than the request asked for is named in the report's
        // `fallbacks` (epic sc-24432 E2) — never a silent downgrade, never a failure. A request
        // that leaves the option unset runs the MLX defaults-table default (E5, sc-24446).
        // The engine also gets the option itself: under `auto` it monitors the proposer it runs
        // against the plain loop it would fall back to, and demotes one measurably slower than
        // plain decoding (sc-24446, E5); an explicit `{proposer, depth}` runs as asked.
        let speculative_mode = req.speculative_or(self.speculative_default);
        let resolution =
            core_llm::resolve_speculative(speculative_mode, &self.descriptor.capabilities);
        let mut fallbacks: Vec<String> = resolution.fallback.into_iter().collect();
        let route = self.speculative_route(resolution.plan, gemma4_mm_request, &mut fallbacks);
        // A `draft_model` request reaching past the draft's own context window runs `auto`
        // instead, by name (sc-24436, E2).
        let route = self.fit_draft_route(
            route,
            admitted_prompt,
            req.max_new_tokens,
            gemma4_mm_request,
            &mut fallbacks,
        );
        // The cross-turn prefix cache (sc-24437) keys on token ids, which cannot tell two images
        // (or clips) behind the same placeholder ids apart — a multimodal prompt neither reads nor
        // feeds it, and the report says so. The hybrid decoder snapshots at the end of the
        // rendered conversation, the prefix the next chat turn extends.
        let (mut prefix_path, mut prefix_reason) = prefix_path_for(
            self.prefix.borrow().budget_bytes() > 0,
            multimodal,
            gemma4_mm_request,
        );
        // Only a request the cache serves reads (restores) or feeds (stores) it — `off` and
        // `bypassed` never touch it.
        let prefix_route = prefix_path == "miss";
        let prefix_boundary = match &self.model {
            Decoder::Qwen35(_) if prefix_route => {
                self.conversation_boundary(messages, req, &prompt_ids)
            }
            _ => None,
        };
        let snapshot_bytes = self
            .prefix_snapshot_bytes(prefix_boundary, route.is_mtp())
            .ok_or_else(|| CoreError::InvalidRequest("prefix snapshot estimate overflow".into()))?;
        let required = self
            .speculative_request_bytes(route, admitted_prompt, req.max_new_tokens, vision_workspace)
            .ok_or_else(|| CoreError::InvalidRequest("request memory estimate overflow".into()))?;
        let available = core_llm::effective_memory_budget(
            core_llm::available_host_memory_bytes(),
            core_llm::operational_memory_override()?,
        )?;
        // Held prefix-cache entries are reclaimable memory (sc-24437, E7): a request that would
        // not fit evicts them least-recently-used first, and the snapshot it would leave behind
        // is admitted with it — or not taken, so caching never makes a request fail and never
        // takes memory admission did not grant.
        let prefix_admission = if prefix_route {
            self.prefix
                .borrow_mut()
                .admit(required, snapshot_bytes, available)
        } else {
            core_llm::PrefixAdmission {
                available: self.prefix.borrow_mut().reclaim_for(required, available),
                snapshot: false,
            }
        };
        let keep_prefix = prefix_admission.snapshot;
        if prefix_route && !keep_prefix {
            prefix_reason = Some(core_llm::PREFIX_NOT_ADMITTED);
        }
        core_llm::admit_request_memory_with_geometry(
            admitted_prompt,
            req.max_new_tokens,
            self.descriptor.capabilities.max_context_tokens,
            if keep_prefix {
                required.saturating_add(snapshot_bytes)
            } else {
                required
            },
            prefix_admission.available,
        )?;
        let prefix_boundary = prefix_boundary.filter(|_| keep_prefix);

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
        // A Gemma 4 prompt's soft-token expansion is known only now: a `draft_model` route checks
        // the draft's context window again against the effective prompt (E2). Nothing has been
        // allocated for the route yet; the priced draft is only an over-estimate.
        let route = self.fit_draft_route(
            route,
            prompt_len,
            req.max_new_tokens,
            gemma4_mm_request,
            &mut fallbacks,
        );

        let config = GenerationConfig {
            max_new_tokens: req.max_new_tokens as usize,
            sampling: map_sampling(&req.sampling),
            seed: req.seed,
            stop_tokens: self.stop_tokens.clone(),
        };

        let mut prefix_hit = 0usize;

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

        // Whether the stream has delivered a token yet. The segmenter, the tool segmenter and the
        // stop matcher can hold the first engine tokens back (a thinking model's opening marker
        // emits nothing), so the engine keeps its look-ahead out of the way until the first
        // delivered token is out — the time to first token never waits behind one (sc-24446).
        let delivered_any = std::cell::Cell::new(false);
        let mut tracked = |event: CoreEvent| {
            if matches!(event, CoreEvent::Token { .. }) {
                delivered_any.set(true);
            }
            on_event(event);
        };
        let on_event: &mut dyn FnMut(CoreEvent) = &mut tracked;
        let delivered = || delivered_any.get();

        // Drive the internal loop; translate token-id events to contract text-delta events via
        // incremental detokenization (re-decode the running sequence, emit the new suffix). The
        // `IncrementalDetok` guard holds back lossy U+FFFD placeholders so a multi-byte character
        // split across BPE tokens streams intact (and never panics a mid-char slice) — sc-12452.
        // The segmenter (when active) splits each delta into reasoning vs answer; answer text then
        // feeds the stop matcher so a stop string is trimmed and halts generation.
        let tokenizer = &self.tokenizer;
        let mut run = {
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
            let width = route.width();
            match (&mm, &g4) {
                // Qwen-VL multimodal: prefill the spliced embeds with interleaved M-RoPE +
                // DeepStack fusion, then decode the continuation (RoPE shifted by `mrope_delta`)
                // through the engine — the MTP head seeds from the fused embeddings.
                (Some(m), _) => {
                    let (t, h, w, delta) = &m.positions;
                    let pos = [t.as_slice(), h.as_slice(), w.as_slice()];
                    let clock = Some(
                        qwen_prefill_started
                            .expect("Qwen multimodal preparation starts the prefill clock"),
                    );
                    match &self.model {
                        Decoder::Qwen35(model) => {
                            let mtp_prompt = Qwen35MtpMultimodalPrompt {
                                input_ids: &m.expanded_ids,
                                embeddings: &m.embeds,
                                positions: pos,
                                visual_pos_mask: &m.visual_pos_mask,
                                deepstack: &m.deepstack,
                                continuation_delta: *delta,
                            };
                            let mut cache = model.new_cache();
                            let (mut proposer, logits, hidden) = if route.is_mtp() {
                                let (hidden, logits) = model
                                    .prefill_hidden_and_last_logits_from_embeds_with_deepstack(
                                        &m.embeds,
                                        pos,
                                        &mut cache,
                                        &m.visual_pos_mask,
                                        &m.deepstack,
                                    )
                                    .map_err(to_core)?;
                                let proposer: Box<dyn Proposer<Qwen35Model>> =
                                    Box::new(MtpProposer::multimodal(&mtp_prompt));
                                (proposer, logits, Some(hidden))
                            } else {
                                let logits = model
                                    .prefill_with_deepstack(
                                        &m.embeds,
                                        pos,
                                        &mut cache,
                                        &m.visual_pos_mask,
                                        &m.deepstack,
                                    )
                                    .map_err(to_core)?;
                                (route.plain_proposer(self.draft.as_ref()), logits, None)
                            };
                            generate_speculative(
                                model,
                                &mut *proposer,
                                SpeculativePrompt::Prefilled {
                                    cache: &mut cache,
                                    logits,
                                    hidden,
                                    history: &m.expanded_ids,
                                    position_delta: *delta,
                                },
                                &config,
                                width,
                                &req.cancel,
                                &mut sink,
                                EngineOptions {
                                    constraint: json_mask
                                        .as_mut()
                                        .map(|m| m as &mut dyn RewindableConstraintMask),
                                    should_stop: should_stop_opt,
                                    delivered: Some(&delivered),
                                    prefill_clock: clock,
                                    speculative_mode,
                                    ..EngineOptions::default()
                                },
                            )
                        }
                        Decoder::Causal(model) => {
                            let mut cache = model.new_cache();
                            let logits = model
                                .prefill_with_deepstack(
                                    &m.embeds,
                                    pos,
                                    &mut cache,
                                    &m.visual_pos_mask,
                                    &m.deepstack,
                                )
                                .map_err(to_core)?;
                            generate_speculative(
                                model,
                                &mut *route.plain_proposer(self.draft.as_ref()),
                                SpeculativePrompt::Prefilled {
                                    cache: &mut cache,
                                    logits,
                                    hidden: None,
                                    history: &m.expanded_ids,
                                    position_delta: *delta,
                                },
                                &config,
                                width,
                                &req.cancel,
                                &mut sink,
                                EngineOptions {
                                    constraint: json_mask
                                        .as_mut()
                                        .map(|m| m as &mut dyn RewindableConstraintMask),
                                    should_stop: should_stop_opt,
                                    delivered: Some(&delivered),
                                    prefill_clock: clock,
                                    speculative_mode,
                                    ..EngineOptions::default()
                                },
                            )
                        }
                    }
                }
                // Gemma 4 multimodal: prefill the spliced embeds on ordinary causal 1-D positions
                // (no M-RoPE, so no position shift for the continuation), then decode through the
                // engine against the unwrapped decoder.
                (None, Some(m)) => {
                    let model =
                        match &self.model {
                            Decoder::Causal(c) => c,
                            Decoder::Qwen35(_) => return Err(CoreError::Load(
                                "gemma 4: the multimodal path requires the generic causal decoder"
                                    .into(),
                            )),
                        };
                    let mut cache = model.new_cache();
                    let logits = model
                        .decode_logits_from_embeds(&m.embeds, &mut cache, 0)
                        .map_err(to_core)?;
                    generate_speculative(
                        model,
                        &mut *route.plain_proposer(self.draft.as_ref()),
                        SpeculativePrompt::Prefilled {
                            cache: &mut cache,
                            logits,
                            hidden: None,
                            history: &m.expanded_ids,
                            position_delta: 0,
                        },
                        &config,
                        width,
                        &req.cancel,
                        &mut sink,
                        EngineOptions {
                            constraint: json_mask
                                .as_mut()
                                .map(|m| m as &mut dyn RewindableConstraintMask),
                            should_stop: should_stop_opt,
                            delivered: Some(&delivered),
                            prefill_clock: None,
                            speculative_mode,
                            ..EngineOptions::default()
                        },
                    )
                }
                // Text: prefill on top of the longest prefix the cross-turn cache restores
                // (sc-24437), decode through the engine, then keep the request's state for the
                // next turn — a softmax KV cache whole, the hybrid's snapshot at the conversation
                // boundary (with the MTP head's state there when it ran).
                (None, None) => {
                    let options = EngineOptions {
                        constraint: json_mask
                            .as_mut()
                            .map(|m| m as &mut dyn RewindableConstraintMask),
                        should_stop: should_stop_opt,
                        delivered: Some(&delivered),
                        prefill_clock: Some(Instant::now()),
                        speculative_mode,
                        ..EngineOptions::default()
                    };
                    match &self.model {
                        Decoder::Causal(model) => {
                            let restored = if prefix_route {
                                self.prefix
                                    .borrow_mut()
                                    .restore::<ContiguousKvCache>(&prompt_ids, false)
                                    .map_err(to_core)?
                            } else {
                                None
                            };
                            let PrefixPrefill {
                                mut cache,
                                logits,
                                fed_tokens,
                                ..
                            } = prefill_restored(
                                model,
                                restored,
                                &prompt_ids,
                                None,
                                false,
                                &req.cancel,
                            )
                            .map_err(to_core)?;
                            // Measured, not looked up: the positions the prefill did not feed.
                            prefix_hit = prompt_ids.len() - fed_tokens;
                            if prefix_hit > 0 {
                                prefix_path = "hit";
                            }
                            let run = generate_speculative(
                                model,
                                &mut *route.plain_proposer(self.draft.as_ref()),
                                SpeculativePrompt::Prefilled {
                                    cache: &mut cache,
                                    logits,
                                    hidden: None,
                                    history: &prompt_ids,
                                    position_delta: 0,
                                },
                                &config,
                                width,
                                &req.cancel,
                                &mut sink,
                                options,
                            );
                            if let (Ok(run), true) = (&run, keep_prefix) {
                                self.prefix
                                    .borrow_mut()
                                    .store_run(&prompt_ids, run, cache, None, None)
                                    .map_err(to_core)?;
                            }
                            run
                        }
                        Decoder::Qwen35(model) => {
                            let mtp = route.is_mtp();
                            let restored = if prefix_route {
                                self.prefix
                                    .borrow_mut()
                                    .restore::<Qwen35Cache>(&prompt_ids, mtp)
                                    .map_err(to_core)?
                            } else {
                                None
                            };
                            let PrefixPrefill {
                                mut cache,
                                logits,
                                hidden,
                                fed_tokens,
                                mtp: resume,
                                boundary,
                                ..
                            } = prefill_restored(
                                model,
                                restored,
                                &prompt_ids,
                                prefix_boundary,
                                mtp,
                                &req.cancel,
                            )
                            .map_err(to_core)?;
                            // Measured, not looked up: the positions the prefill did not feed.
                            prefix_hit = prompt_ids.len() - fed_tokens;
                            if prefix_hit > 0 {
                                prefix_path = "hit";
                            }
                            let mut head = mtp.then(|| {
                                MtpProposer::new()
                                    .resume_from(resume)
                                    .capture_at(boundary.as_ref().map(Boundary::len))
                            });
                            let mut plain = route.plain_proposer(self.draft.as_ref());
                            let proposer: &mut dyn Proposer<Qwen35Model> = match head.as_mut() {
                                Some(head) => head,
                                None => &mut *plain,
                            };
                            let run = generate_speculative(
                                model,
                                proposer,
                                SpeculativePrompt::Prefilled {
                                    cache: &mut cache,
                                    logits,
                                    hidden,
                                    history: &prompt_ids,
                                    position_delta: 0,
                                },
                                &config,
                                width,
                                &req.cancel,
                                &mut sink,
                                options,
                            );
                            if let (Ok(run), true) = (&run, keep_prefix) {
                                let captured = head.as_mut().and_then(MtpProposer::take_captured);
                                self.prefix
                                    .borrow_mut()
                                    .store_run(&prompt_ids, run, cache, boundary, captured)
                                    .map_err(to_core)?;
                            }
                            run
                        }
                    }
                }
            }
            .map_err(to_core)?
        };
        let timing = run.take_timer();
        let out = run.output;
        let stats = run.stats;
        let mut report = run.report;
        // The request's resolution fallbacks join the engine's measured report (E2/E3), and so
        // does what the prefix cache restored. The engine counts a caller's prefill as its one
        // prefill forward: the boundary snapshot is taken inside it (sc-24446).
        report.fallbacks = fallbacks;
        report.prefix_hit_tokens = prefix_hit as u64;
        report.prefix_cache = core_llm::PathReport {
            path: prefix_path.into(),
            reason: prefix_reason.map(str::to_string),
        };
        report.fused_primitives = crate::primitives::fused::fused_tally()
            .since(&fused_start)
            .path_report();

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
            mtp: (report.proposer == ProposerKind::Mtp).then(|| core_llm::MtpStats {
                proposed_tokens: u32::try_from(stats.proposed).unwrap_or(u32::MAX),
                accepted_tokens: u32::try_from(stats.accepted).unwrap_or(u32::MAX),
                target_forwards: u32::try_from(stats.forwards).unwrap_or(u32::MAX),
            }),
            decode: Some(report),
            finish_reason: Some(finish),
        })
    }
}

// Why a multimodal request never reads or feeds the cross-turn prefix cache (sc-24437): the
// shared reason both backends name (E8).
#[cfg(test)]
use core_llm::PREFIX_MULTIMODAL_BYPASS;

/// The cross-turn prefix cache's part in a request before any lookup (sc-24437): `off` when the
/// load settled a zero budget, `bypassed` (with the reason) for a multimodal prompt — Qwen-VL
/// (`multimodal`) or Gemma 4 (`gemma4_mm`) — whose token ids cannot tell two images or clips apart,
/// else `miss` (the lookup may still turn it into a `hit`). Only a `miss` request reads or feeds
/// the cache.
fn prefix_path_for(
    prefix_on: bool,
    multimodal: bool,
    gemma4_mm: bool,
) -> (&'static str, Option<&'static str>) {
    core_llm::prefix_path_before_lookup(prefix_on, multimodal || gemma4_mm)
}

/// The ceiling on any speculative proposal MLX runs (sc-24438, epic sc-24432 E4): drafts per
/// verify step, so the verify forward carries at most `1 + 7 = 8` query rows. A loaded model
/// advertises its own, possibly lower, depth — [`speculative_max_depth`] over its attention
/// geometry — and the weightless [`provider_descriptor`] advertises this ceiling.
///
/// Why 8 rows: the attention primitive's `MLX_SDPA_VECTOR_MAX_QLEN` is the widest query MLX
/// 0.32's single-pass **vector** SDPA kernel takes — its decode kernel, the one every plain
/// `q_len = 1` step runs. A verify it serves in one call costs one fused call per attention layer,
/// no score matrix, the same kernel as the plain step it must agree with token for token. Past it
/// the verify moves to the fused **full** kernel (head dims 64/80/128), whose query blocks are
/// sized for prefill, or — for head dims only the vector kernel serves, like the Qwen35 hybrid's
/// 256 — to a second vector-kernel tile per layer: the per-step verify cost stops being flat in
/// the depth exactly where speculation needs it flat. sc-24442 removed the old 8-row cap on
/// *fused* SDPA; this is the width the decode kernel serves, not that cap. On the Qwen35 hybrid
/// every extra depth also adds a checkpoint-ring slot per linear layer (sc-24435).
pub const SPECULATIVE_MAX_DEPTH: u32 =
    crate::primitives::attention::MLX_SDPA_VECTOR_MAX_QLEN as u32 - 1;

/// The deepest speculative proposal (prompt lookup and MTP alike) a model whose attention layers
/// have these `(q_heads, kv_heads, qk_head_dim, v_head_dim)` geometries runs on MLX (sc-24438,
/// E4): one less than the narrowest `vector_verify_rows` among them, so a max-depth verify is one
/// decode-kernel call in **every** attention layer.
///
/// MLX's vector kernel serves `q_len × gqa <= 32`, so a GQA group wider than 4 narrows it:
/// Qwen3.8-27B (24q/4kv/hd 256, gqa 6) verifies 5 rows in one call → depth 4; Qwen3.5/3.6-35B-A3B
/// (16q/2kv/hd 256, gqa 8) 4 rows → depth 3; a gqa-4 Llama 8 rows → depth 7
/// ([`SPECULATIVE_MAX_DEPTH`]). A geometry the vector kernel does not serve is bounded at 8 rows.
/// Never below 1: a gqa-32 group (one vector row) still speculates one draft, in two calls.
pub fn speculative_max_depth(geometries: impl IntoIterator<Item = (i32, i32, i32, i32)>) -> u32 {
    geometries
        .into_iter()
        .map(|(hq, hkv, qd, vd)| crate::primitives::attention::vector_verify_rows(hq, hkv, qd, vd))
        .min()
        .map_or(SPECULATIVE_MAX_DEPTH, |rows| (rows - 1).max(1) as u32)
}

/// Every attention geometry `(q_heads, kv_heads, qk_head_dim, v_head_dim)` a causal decoder's
/// layers hand `sdpa`: per layer type (Gemma 4's sliding and full layers differ), MLA's
/// head-expanded `qk_nope + qk_rope` / `v_head_dim` keys.
fn causal_attention_geometries(cfg: &ModelConfig) -> Vec<(i32, i32, i32, i32)> {
    if let Some(mla) = &cfg.mla {
        return vec![(
            cfg.num_heads,
            cfg.num_heads,
            mla.q_head_dim(),
            mla.v_head_dim,
        )];
    }
    let mut geometries: Vec<_> = (0..cfg.num_layers.max(1))
        .map(|i| {
            let a = cfg.layer_attention(i);
            (cfg.num_heads, a.num_kv_heads, a.head_dim, a.head_dim)
        })
        .collect();
    geometries.sort_unstable();
    geometries.dedup();
    geometries
}

/// The attention geometry of the Qwen35 hybrid's full-attention layers — the only layers that run
/// `sdpa` (the Gated DeltaNet layers are recurrent) — which its MTP predictor layer shares.
fn qwen35_attention_geometry(cfg: &Qwen35Config) -> (i32, i32, i32, i32) {
    (cfg.num_heads, cfg.num_kv_heads, cfg.head_dim, cfg.head_dim)
}

/// The per-proposer depths [`Speculative::Auto`](core_llm::Speculative::Auto) runs and every MLX
/// advertisement recommends: the MLX row of the decode-defaults table (E5,
/// [`core_llm::defaults`]).
fn recommended_depths() -> &'static core_llm::RecommendedDepths {
    &core_llm::DecodeBackend::Mlx.defaults().recommended_depths
}

/// The prompt-lookup advertisement every `mlx-llama` decoder carries, at the model's
/// [`speculative_max_depth`] (the shared [`core_llm::prompt_lookup_capabilities`], recommending the
/// MLX row's depth).
fn prompt_lookup_capabilities(max_depth: u32) -> ProposerCapabilities {
    core_llm::prompt_lookup_capabilities(max_depth, recommended_depths())
}

/// The proposer a request runs on the engine, after its resolved plan meets the request's shape
/// (sc-24434, sc-24436). `width` is the drafts per verify step (`0` for plain decoding).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum SpeculativeRoute {
    Plain,
    Mtp { width: usize },
    PromptLookup { width: usize },
    DraftModel { width: usize },
}

impl SpeculativeRoute {
    fn width(self) -> usize {
        match self {
            SpeculativeRoute::Plain => 0,
            SpeculativeRoute::Mtp { width }
            | SpeculativeRoute::PromptLookup { width }
            | SpeculativeRoute::DraftModel { width } => width,
        }
    }

    fn is_mtp(self) -> bool {
        matches!(self, SpeculativeRoute::Mtp { .. })
    }

    /// The proposer for any target: prompt lookup, the resident `draft` model, or none. (MTP
    /// needs the Qwen35 target.)
    fn plain_proposer<'a, T: SpeculativeTarget + 'a>(
        self,
        draft: Option<&'a ResidentDraft>,
    ) -> Box<dyn Proposer<T> + 'a> {
        match (self, draft) {
            (SpeculativeRoute::PromptLookup { .. }, _) => Box::new(NgramProposer::default()),
            (SpeculativeRoute::DraftModel { width: drafts }, Some(resident)) => {
                let (proposable, width) = (resident.proposable, resident.width);
                match &resident.model {
                    Decoder::Causal(draft) => Box::new(
                        DraftModelProposer::new(draft, drafts).with_vocab(proposable, width),
                    ),
                    Decoder::Qwen35(draft) => Box::new(
                        DraftModelProposer::new(draft, drafts).with_vocab(proposable, width),
                    ),
                }
            }
            // `speculative_route` routes `draft_model` only with a resident draft.
            (SpeculativeRoute::DraftModel { .. }, None)
            | (SpeculativeRoute::Plain | SpeculativeRoute::Mtp { .. }, _) => Box::new(NoProposer),
        }
    }
}

impl LlamaProvider {
    /// Where the rendered conversation ends and the generation prompt begins, when that is a
    /// strict token prefix of `prompt_ids` (sc-24437): the point the hybrid decoder's prefix-cache
    /// snapshot is taken at, because the next chat turn re-renders these messages and extends
    /// them (whatever the template later does to this turn's reply). `None` when the template
    /// renders the messages differently without a generation prompt.
    fn conversation_boundary(
        &self,
        messages: &[Message],
        req: &TextLlmRequest,
        prompt_ids: &[i32],
    ) -> Option<usize> {
        let rendered = self
            .template
            .render_with(
                messages,
                &RenderOptions {
                    add_generation_prompt: false,
                    enable_thinking: req.enable_thinking_kwarg(),
                    reasoning_effort: req.reasoning_effort,
                    preserve_thinking: req.preserve_thinking,
                    tools: &req.tools,
                },
            )
            .ok()?;
        let ids = self.tokenizer.encode(&rendered, false).ok()?;
        let len = ids.len();
        (len > 0
            && len < prompt_ids.len()
            && ids.iter().zip(prompt_ids).all(|(&a, &b)| a as i32 == b))
        .then_some(len)
    }

    /// An upper bound on the bytes of the prefix-cache snapshot a text request takes during its
    /// run (sc-24437), which its admission adds: the hybrid's whole cache at the conversation
    /// `boundary` (and the MTP head's state there with `mtp`). A softmax decoder keeps the
    /// request's own cache after the run — memory the request already holds — so it takes none.
    fn prefix_snapshot_bytes(&self, boundary: Option<usize>, mtp: bool) -> Option<u64> {
        match (&self.model, boundary) {
            (Decoder::Qwen35(m), Some(b)) => m.config().prefix_snapshot_bytes(b, mtp),
            _ => Some(0),
        }
    }

    /// The request's admission price under its speculative route. Prompt lookup and a draft
    /// model's verify roll back without a forward or a copy on both decoders — a softmax decoder
    /// by truncation, the Qwen35 hybrid through its DeltaNet checkpoint ring (sc-24435) — so they
    /// pay their verify overshoot
    /// (`width` more KV positions and `width` more hidden + logit rows) and, on the hybrid, the
    /// ring ([`Qwen35Model::checkpoint_ring_bytes`]). MTP keeps the predictor-cache pricing
    /// (`width` in the estimate) plus the ring; the ring replaced the step-start snapshot, so the
    /// hybrid's recurrent state is charged once, never as clone + replay copies.
    fn speculative_request_bytes(
        &self,
        route: SpeculativeRoute,
        prompt_tokens: usize,
        max_new_tokens: u32,
        vision_workspace: u64,
    ) -> Option<u64> {
        let mut geometry = self.model.memory_geometry();
        if let (Decoder::Qwen35(m), width @ 1..) = (&self.model, route.width()) {
            geometry.recurrent_bytes = geometry
                .recurrent_bytes
                .checked_add(m.checkpoint_ring_bytes(width)?)?;
        }
        let (priced_width, overshoot) = match route {
            SpeculativeRoute::PromptLookup { width } | SpeculativeRoute::DraftModel { width } => {
                let width = u64::try_from(width).ok()?;
                let kv_position = geometry
                    .layers
                    .checked_mul(geometry.kv_heads)?
                    .checked_mul(geometry.head_dim)?
                    .checked_mul(geometry.element_bytes)?
                    .checked_mul(2)?;
                let rows = geometry
                    .hidden_size
                    .checked_add(geometry.vocab_size)?
                    .checked_mul(geometry.element_bytes)?;
                // The verify step writes `width` positions past the committed ones; the caches
                // round *that* up to a block. What the target estimate already priced: its own
                // padding over prompt + generation, plus the `width` rows below.
                let committed = u64::try_from(prompt_tokens)
                    .ok()?
                    .checked_add(u64::from(max_new_tokens))?;
                let block_extra = kv_overshoot_padding(committed, width)?;
                (
                    0,
                    width
                        .checked_mul(kv_position.checked_add(rows)?)?
                        .checked_add(block_extra.checked_mul(kv_position)?)?,
                )
            }
            other => (u32::try_from(other.width()).ok()?, 0),
        };
        let target = estimate_mlx_request_bytes(
            prompt_tokens,
            max_new_tokens,
            geometry,
            vision_workspace,
            priced_width,
            self.model.workspace_contract(),
        )?
        .checked_add(overshoot)?
        .checked_add(self.model.promoted_request_bytes()?)?;
        // The draft model's own request (E7, sc-24436): its prefill of the same prompt and a
        // cache that grows over the whole run plus the provisional drafts a step writes past the
        // committed tokens — and, when the draft is the hybrid, the checkpoint ring its proposal
        // window holds (sc-24435), armed for the same `width`: no snapshot, no replay.
        let draft = match (route, self.draft.as_ref().map(|d| &d.model)) {
            (SpeculativeRoute::DraftModel { width }, Some(draft)) => {
                let mut geometry = draft.memory_geometry();
                if let Decoder::Qwen35(m) = draft {
                    geometry.recurrent_bytes = geometry
                        .recurrent_bytes
                        .checked_add(m.checkpoint_ring_bytes(width)?)?;
                }
                estimate_mlx_request_bytes(
                    prompt_tokens,
                    max_new_tokens.checked_add(u32::try_from(width).ok()?)?,
                    geometry,
                    0,
                    0,
                    draft.workspace_contract(),
                )?
                .checked_add(draft.promoted_request_bytes()?)?
            }
            _ => 0,
        };
        target.checked_add(draft)
    }

    /// The resident draft's context window (`0`: no draft, or an unbounded one).
    fn draft_context(&self) -> usize {
        self.draft.as_ref().map_or(0, |draft| draft.context)
    }

    /// A `draft_model` route whose request — `prompt_tokens` + `max_new_tokens` and a step's
    /// overshoot — outruns the resident draft's context window runs what `auto` resolves to
    /// instead, the reason named in `fallbacks` ([`core_llm::fit_draft_context`], sc-24436 E2).
    /// Any other route is returned unchanged.
    fn fit_draft_route(
        &self,
        route: SpeculativeRoute,
        prompt_tokens: usize,
        max_new_tokens: u32,
        gemma4_mm_request: bool,
        fallbacks: &mut Vec<String>,
    ) -> SpeculativeRoute {
        let SpeculativeRoute::DraftModel { width } = route else {
            return route;
        };
        let fit = core_llm::fit_draft_context(
            core_llm::SpeculativeResolution {
                plan: SpeculativePlan::Run {
                    proposer: SpeculativeProposer::DraftModel,
                    depth: width as u32,
                },
                fallback: None,
            },
            &self.descriptor.capabilities,
            self.draft_context(),
            prompt_tokens,
            max_new_tokens,
        );
        match fit.fallback {
            None => route,
            Some(why) => {
                fallbacks.push(why);
                self.speculative_route(fit.plan, gemma4_mm_request, fallbacks)
            }
        }
    }

    /// Route a resolved speculative plan onto what this request can run, naming every downgrade
    /// in `fallbacks` (E2): MTP needs a loaded Qwen3.8 predictor and a text or Qwen-VL prompt, and
    /// `draft_model` a resident draft (sc-24436) — it runs on every prompt shape, the draft
    /// proposing over the effective prompt ids.
    fn speculative_route(
        &self,
        plan: SpeculativePlan,
        gemma4_mm_request: bool,
        fallbacks: &mut Vec<String>,
    ) -> SpeculativeRoute {
        match plan {
            SpeculativePlan::Off => SpeculativeRoute::Plain,
            SpeculativePlan::Run {
                proposer: SpeculativeProposer::Mtp,
                depth,
            } => {
                let head = matches!(&self.model, Decoder::Qwen35(m) if m.has_mtp());
                if head && !gemma4_mm_request {
                    SpeculativeRoute::Mtp {
                        width: depth as usize,
                    }
                } else {
                    fallbacks.push(if head {
                        "speculative: `mtp` has no Qwen predictor for a Gemma 4 multimodal \
                         prompt; decoded without a proposer"
                            .into()
                    } else {
                        "speculative: `mtp` has no loaded MTP predictor on this model; decoded \
                         without a proposer"
                            .into()
                    });
                    SpeculativeRoute::Plain
                }
            }
            SpeculativePlan::Run {
                proposer: SpeculativeProposer::PromptLookup,
                depth,
            } => SpeculativeRoute::PromptLookup {
                width: depth as usize,
            },
            SpeculativePlan::Run {
                proposer: SpeculativeProposer::DraftModel,
                depth,
            } if self.draft.is_some() => SpeculativeRoute::DraftModel {
                width: depth as usize,
            },
            SpeculativePlan::Run {
                proposer: SpeculativeProposer::DraftModel,
                ..
            } => {
                fallbacks.push(core_llm::DRAFT_MODEL_NOT_LOADED.into());
                SpeculativeRoute::Plain
            }
        }
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
            // Every decoder this provider loads runs the speculative engine (sc-24434), so prompt
            // lookup needs nothing from the checkpoint. A Qwen3.8 MTP head is advertised on the
            // legacy `mtp` field by the load path.
            speculative: vec![prompt_lookup_capabilities(SPECULATIVE_MAX_DEPTH)],
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
    d.capabilities.speculative = vec![prompt_lookup_capabilities(speculative_max_depth(
        causal_attention_geometries(cfg),
    ))];
    d
}

/// The loaded-model descriptor for the hybrid Qwen3.6 (`qwen3_5`) decoder (parsed via
/// [`Qwen35Config`], which `ModelConfig` does not represent).
fn descriptor_for_qwen35(cfg: &Qwen35Config) -> TextLlmDescriptor {
    let mut d = provider_descriptor();
    d.family = Architecture::Qwen35.family().to_string();
    d.capabilities.max_context_tokens = cfg.max_position_embeddings.max(0) as usize;
    // Both proposers at the full-attention layers' backend-true depth (sc-24438); a configured
    // head — dense or sparse-MoE predictor layer — on both the legacy and the per-proposer field.
    let max_depth = speculative_max_depth([qwen35_attention_geometry(cfg)]);
    d.capabilities.speculative = vec![prompt_lookup_capabilities(max_depth)];
    if cfg.mtp_num_hidden_layers > 0 {
        d.capabilities
            .advertise_mtp(max_depth, recommended_depths().mtp);
    }
    d
}

#[cfg(test)]
thread_local! {
    /// A test's operational load budget on this thread ([`with_load_budget`]); the process-wide
    /// [`core_llm::AVAILABLE_MEMORY_OVERRIDE`] would leak into every concurrently loading test.
    static LOAD_BUDGET: std::cell::Cell<Option<u64>> = const { std::cell::Cell::new(None) };
}

/// Run `f` with load admission capped at `budget` bytes on this thread.
#[cfg(test)]
pub(crate) fn with_load_budget<R>(budget: u64, f: impl FnOnce() -> R) -> R {
    struct Restore(Option<u64>);
    impl Drop for Restore {
        fn drop(&mut self) {
            LOAD_BUDGET.with(|b| b.set(self.0));
        }
    }
    let _restore = Restore(LOAD_BUDGET.with(|b| b.replace(Some(budget))));
    f()
}

/// The operational budget load admission caps measured availability with
/// ([`core_llm::operational_memory_override`]).
fn load_memory_budget() -> CoreResult<Option<u64>> {
    #[cfg(test)]
    if let Some(budget) = LOAD_BUDGET.with(std::cell::Cell::get) {
        return Ok(Some(budget));
    }
    core_llm::operational_memory_override()
}

/// Admit a companion MTP head's resident bytes on top of what the load already admitted (the
/// target and any admitted draft, E7), returning the head and its bytes. `Err` is the named load
/// fallback (E2): the head is unreadable, or those bytes plus the head's exceed the budget.
fn admit_companion_head(
    head: &Path,
    target_bytes: u64,
    available: u64,
) -> Result<(&Path, u64), String> {
    // The shared rule Candle admits by, in MLX's one allocation domain (E8).
    core_llm::admit_companion_head(
        head,
        crate::load_memory::companion_head_bytes(head),
        [("unified memory", target_bytes, available)],
    )
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
/// The largest command buffer MLX 0.32 encodes before committing it, in bytes of its ops'
/// inputs and outputs (`max_mb_per_buffer`: 40 MB on base / Pro / phone GPUs, 50 MB on Max and
/// Ultra; `MLX_MAX_MB_PER_BUFFER` is never set by this runtime). A buffer may exceed it by the one
/// op that crosses it.
const MLX_MAX_BUFFER_BYTES: u64 = 50 * 1024 * 1024;

/// The most ops MLX 0.32 encodes into one command buffer before committing it
/// (`max_ops_per_buffer`: 20–50 by GPU family; `MLX_MAX_OPS_PER_BUFFER` is never set here).
const MLX_MAX_OPS_PER_BUFFER: u64 = 50;

/// What a forward's promoted weight copies can hold at once when the decoder's activations are
/// `f32` against BF16 weights (a Gemma GeGLU whose role the activation-dtype policy keeps on
/// `f32`, [`CausalLm::activations_promote`]; sc-24446).
///
/// Each dense matmul materializes an `f32` copy of its weight (a quantized matmul, of its scales
/// and biases), held until the command buffer that consumes it completes. The LM head's copy —
/// the vocabulary-wide one, 2.4 GB on Gemma 2 and 4.0 GB on the 262K-token Gemma 4 — is priced
/// whole; the decoder's copies are bounded by MLX's evaluation window: at most
/// [`MLX_EVAL_BUFFER_WINDOW`] command buffers are alive (ten committed plus the one encoding),
/// each holding at most [`MLX_MAX_BUFFER_BYTES`] plus the one op that crosses it — at most the
/// largest promoted projection, `largest_elements`.
fn promoted_weight_bytes(head_elements: u64, largest_elements: u64) -> Option<u64> {
    const F32: u64 = 4;
    let per_buffer = largest_elements
        .checked_mul(F32)?
        .checked_add(MLX_MAX_BUFFER_BYTES)?;
    head_elements
        .checked_mul(F32)?
        .checked_add(per_buffer.checked_mul(MLX_EVAL_BUFFER_WINDOW)?)
}

/// Apple-Silicon Metal allocations are rounded to 16-KiB VM pages. Gated DeltaNet retains one
/// independently allocated output row per prompt token until its final concatenate.
const MLX_ALLOCATION_PAGE_BYTES: u64 = 16 * 1024;
/// The recurrence creates five one-element index buffers per step and evaluates every 256 steps.
///
/// These `QWEN35_RECURRENCE_*` terms price the op-by-op recurrence the frozen allocator peaks were
/// measured on. Since sc-24443 the GPU runs the fused Metal kernel instead. Its prompt-sized
/// buffers are its f32 `y` output and the row-contiguous copies MLX makes of non-contiguous `q`,
/// `k` and `v` inputs — `2·Hk·Dk + 2·Hv·Dv` elements per token, never more than the
/// `2·Hv·Dk + 2·Hv·Dv` priced here (`Hv ≥ Hk`). Past `KERNEL_MAX_STEPS` tokens the per-dispatch
/// input gathers replace those copies and the final concatenate of the per-dispatch outputs adds
/// one more `Hv·Dv`, still within the priced width while `Hv·Dv ≤ 2·(Hv − Hk)·Dk` (Qwen3.6-27B:
/// `Hv = 3·Hk`, `Dk = Dv`). Its recurrent state is fixed-size, covered by the 256 × 5 state-array
/// term, and it allocates neither the per-step index buffers nor the page-padded per-token `y`
/// rows priced below. On the GPU these terms are therefore a conservative upper bound for those
/// geometries (the op path remains the CPU-stream route for short runs).
const QWEN35_RECURRENCE_EVAL_CHUNK: u64 = 256;
const QWEN35_RECURRENCE_INDEX_BUFFERS: u64 = 5;
/// Each lazy recurrence step creates five state-shaped arrays: the decayed state, the state-key
/// product, the delta outer product, the updated state, and the updated-state/query product. None
/// can be released before that chunk's explicit evaluation.
const QWEN35_RECURRENCE_STATE_ARRAYS_PER_STEP: u64 = 5;

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

    // PrismLinear rotates each projection input independently. The full-attention block has the
    // wider simultaneous input-width sum; the linear block uses two mixer input rotations plus its
    // output rotation. Four buffers per input width covers the unfused cast -> sign multiply ->
    // Hadamard -> cast chain (F32 intermediates plus the BF16 result, no donation) — the path a
    // block the fused kernel does not cover still takes. The fused kernel (sc-24444) writes only
    // the output buffer, so this stays an upper bound on the Bonsai path.
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
        prompt.min(SDPA_SCORE_TILE_QLEN as u64),
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
    // Index buffers are created eagerly for all five slices in the current recurrence chunk.
    let recurrence_steps = prompt.min(QWEN35_RECURRENCE_EVAL_CHUNK);
    let index_buffers = checked_product([
        recurrence_steps,
        QWEN35_RECURRENCE_INDEX_BUFFERS,
        MLX_ALLOCATION_PAGE_BYTES,
    ])?;
    // `gated_delta_recurrence` builds a complete lazy graph until its 256-step `eval`. Price every
    // state-shaped node in that graph plus the input state. The shared base estimate separately
    // prices the retained recurrent caches for all decoder layers.
    let recurrence_state_arrays = recurrence_steps
        .checked_mul(QWEN35_RECURRENCE_STATE_ARRAYS_PER_STEP)?
        .checked_add(1)?;
    let recurrent_state_buffers =
        checked_product([value_width, key_head_dim, 4, recurrence_state_arrays])?;

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
    let decoder = estimate_mlx_decoder_bytes(
        prompt_tokens,
        max_new_tokens,
        geometry,
        vision_workspace_bytes,
        mtp_width,
        contract,
    )?;
    let prompt = u64::try_from(prompt_tokens).ok()?;
    let runtime = match contract {
        // The hybrid's contract already prices MLX's evaluation window, allocator rounding and
        // block-allocated caches from its own frozen probes; only the driver's wake is added.
        MlxWorkspaceContract::Qwen35 { .. } => MLX_REQUEST_WAKE_BYTES,
        MlxWorkspaceContract::Eager => mlx_runtime_request_bytes(
            prompt_tokens,
            prompt.checked_add(u64::from(max_new_tokens))?,
            u64::from(mtp_width),
            prompt,
            geometry,
        )?,
        MlxWorkspaceContract::Chunked => mlx_runtime_request_bytes(
            prompt_tokens,
            prompt.checked_add(u64::from(max_new_tokens))?,
            u64::from(mtp_width),
            prompt.min(SDPA_SCORE_TILE_QLEN as u64),
            geometry,
        )?,
    };
    decoder.checked_add(runtime)
}

/// The `phys_footprint` a request re-acquires from the Metal driver after the process has idled
/// ([`crate::load_memory::MLX_DRIVER_WAKE_BYTES`]), charged once per request.
const MLX_REQUEST_WAKE_BYTES: u64 = crate::load_memory::MLX_DRIVER_WAKE_BYTES;

/// K/V positions a request's caches allocate beyond `positions`: [`ContiguousKvCache`] grows in
/// whole [`KV_BLOCK_TOKENS`](crate::primitives::kv_cache::KV_BLOCK_TOKENS)-position blocks.
fn kv_block_padding(positions: u64) -> Option<u64> {
    let block = u64::try_from(crate::primitives::kv_cache::KV_BLOCK_TOKENS).ok()?;
    round_up(positions, block)?.checked_sub(positions)
}

/// Block positions a `width`-position verify overshoot past `committed` positions allocates
/// beyond `committed`'s own block padding and the `width` positions themselves.
fn kv_overshoot_padding(committed: u64, width: u64) -> Option<u64> {
    let block = u64::try_from(crate::primitives::kv_cache::KV_BLOCK_TOKENS).ok()?;
    let with = round_up(committed.checked_add(width)?, block)?;
    let without = round_up(committed, block)?.checked_add(width)?;
    Some(with.saturating_sub(without))
}

/// What MLX's runtime holds for a generic decoder's request beyond the decoder's own tensors
/// (sc-24446), which the shared estimates do not see — measured on the one-token requests of the
/// sc-24446 probes (35–87 MB MLX active on Llama 3.2 1B, Qwen3-1.7B and Qwen3-8B, against 6–9 MB
/// estimated before):
///
/// * **KV block padding**: the caches hold whole blocks over every position the request can write
///   — the `committed` prompt and generation plus a `verify_width` step past them (an MTP verify;
///   the prompt-lookup / draft overshoot is added by its route).
/// * **In-flight layer temporaries.** MLX keeps every op's inputs until the command buffer that
///   ran it completes, and runs ahead by up to [`MLX_EVAL_BUFFER_WINDOW`] buffers, each holding at
///   most [`MLX_MAX_BUFFER_BYTES`] plus the one op that crosses it. The shared estimate prices one
///   layer's activations; the rest in flight is every other layer's, capped by that window —
///   whose crossing op can be a whole `prompt × inter` MLP activation or an `attention_rows`-row
///   score block (eager: the full `prompt × prompt` per head).
/// * **Allocation rounding**: a page per op output in the window.
/// * **The driver's wake** ([`MLX_REQUEST_WAKE_BYTES`]).
fn mlx_runtime_request_bytes(
    prompt_tokens: usize,
    committed: u64,
    verify_width: u64,
    attention_rows: u64,
    geometry: LlmMemoryGeometry,
) -> Option<u64> {
    let prompt = u64::try_from(prompt_tokens).ok()?;
    let kv_position = checked_product([
        geometry.layers,
        geometry.kv_heads,
        geometry.head_dim,
        geometry.element_bytes,
        2,
    ])?;
    // Every allocated position past the committed ones: the verify width the step writes beyond
    // them and the rest of their last block.
    let kv_padding = checked_sum([
        verify_width,
        kv_block_padding(committed.checked_add(verify_width)?)?,
    ])?
    .checked_mul(kv_position)?;
    let scores = checked_product([
        prompt,
        attention_rows,
        geometry.query_heads,
        geometry.element_bytes,
    ])?;
    let mlp = checked_product([prompt, geometry.intermediate_size, geometry.element_bytes])?;
    let per_layer = checked_sum([
        checked_product([
            prompt,
            checked_sum([
                geometry.intermediate_size.checked_mul(3)?,
                geometry.hidden_size.checked_mul(8)?,
            ])?,
            geometry.element_bytes,
        ])?,
        scores.checked_mul(3)?,
    ])?;
    let window = MLX_EVAL_BUFFER_WINDOW
        .checked_mul(checked_sum([MLX_MAX_BUFFER_BYTES, mlp.max(scores)])?)?;
    let in_flight = geometry
        .layers
        .saturating_sub(1)
        .checked_mul(per_layer)?
        .min(window);
    // Each in-flight buffer's op outputs are page-rounded allocations, however small.
    let rounding = checked_product([
        MLX_EVAL_BUFFER_WINDOW,
        MLX_MAX_OPS_PER_BUFFER,
        MLX_ALLOCATION_PAGE_BYTES,
    ])?;
    checked_sum([kv_padding, in_flight, rounding, MLX_REQUEST_WAKE_BYTES])
}

/// The decoder's own request tensors under its workspace contract (the shared estimates, plus
/// the Qwen3.5 hybrid's recurrence workspace).
fn estimate_mlx_decoder_bytes(
    prompt_tokens: usize,
    max_new_tokens: u32,
    geometry: LlmMemoryGeometry,
    vision_workspace_bytes: u64,
    mtp_width: u32,
    contract: MlxWorkspaceContract<'_>,
) -> Option<u64> {
    match contract {
        // The recurrent term once: the only MLX decoder with recurrent state (the Qwen35 hybrid,
        // eager for MoE) rolls back through its checkpoint ring, whose bytes the caller folds into
        // `geometry.recurrent_bytes`, and is never cloned; a causal decoder holds none.
        MlxWorkspaceContract::Eager => core_llm::estimate_request_bytes_with_recurrent_copies(
            prompt_tokens,
            max_new_tokens,
            geometry,
            vision_workspace_bytes,
            mtp_width,
            1,
        ),
        MlxWorkspaceContract::Chunked => core_llm::estimate_chunked_request_bytes(
            prompt_tokens,
            max_new_tokens,
            geometry,
            vision_workspace_bytes,
            mtp_width,
            SDPA_SCORE_TILE_QLEN as usize,
        ),
        MlxWorkspaceContract::Qwen35 { config, prism } => {
            // The recurrent term once: the hybrid rolls back through its checkpoint ring, whose
            // bytes the caller folds into `geometry.recurrent_bytes`, and is never cloned.
            let base = core_llm::estimate_chunked_request_bytes_with_recurrent_copies(
                prompt_tokens,
                max_new_tokens,
                geometry,
                vision_workspace_bytes,
                mtp_width,
                SDPA_SCORE_TILE_QLEN as usize,
                1,
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

    /// The MLX row of the decode-defaults table, read directly (not through the provider's own
    /// reader) so a provider that stops following the table is caught.
    const MLX_ROW: &core_llm::DecodeDefaults = core_llm::DecodeBackend::Mlx.defaults();

    #[test]
    fn qwen35_dense_checkpoint_selects_one_decoder_root() {
        let select = |keys: &[&str]| qwen35_dense_prefix(|key| keys.contains(&key));
        assert_eq!(
            select(&["model.language_model.embed_tokens.weight"]).unwrap(),
            "model.language_model"
        );
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
        // scale: eager prompt-squared scores dominate hundreds of GB, while MLX's fused full and
        // vector kernels materialize no score matrix (priced conservatively as one 8-query-row score
        // tile per call) and the decoder projects one final row to the vocabulary.
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
        // sc-24446 prices MLX's in-flight window (eleven command buffers, each up to 50 MB plus
        // the crossing op — here a 2 GB `prompt × inter` MLP activation), so the fused bound grew
        // from ~36 GB; it stays an order of magnitude under the quadratic eager one.
        assert!(
            fused * 8 < eager,
            "bounded MLX peak: {fused} vs eager {eager}"
        );
        assert!(eager > 400_000_000_000, "quadratic eager peak: {eager}");
        assert_eq!(
            eager,
            core_llm::estimate_request_bytes(29_600, 128, geometry, 0, 0).unwrap()
                + mlx_runtime_request_bytes(29_600, 29_728, 0, 29_600, geometry).unwrap(),
            "non-fused paths retain the shared fail-closed estimate, plus MLX's runtime terms"
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
        let arithmetic = estimate_mlx_request_bytes(27, 128, geometry, 0, 0, contract).unwrap();
        let context_64 = estimate_mlx_request_bytes(1_187, 128, geometry, 0, 0, contract).unwrap();
        let context_512 = estimate_mlx_request_bytes(9_251, 128, geometry, 0, 0, contract).unwrap();
        let context_2048 =
            estimate_mlx_request_bytes(36_899, 128, geometry, 0, 0, contract).unwrap();

        // Frozen local probe at 02ed7e958: arithmetic raised the process-global allocator peak to
        // 9,321,652,712 from a captured post-load active 8,602,141,760 (a 719,510,952-byte lower
        // bound). Subtracting the stable post-request residency 8,548,422,952 gives 773,229,760;
        // price that stricter observed bound even though it is not an immediately-pre-request
        // sample. Frozen campaign 35461924246 supplies the two longer-context deltas below.
        assert!(arithmetic >= 773_229_760, "estimate: {arithmetic}");
        assert!(context_64 >= 4_091_010_468, "estimate: {context_64}");
        assert!(context_512 >= 22_885_373_536, "estimate: {context_512}");
        assert_eq!(arithmetic, 788_726_144 + MLX_REQUEST_WAKE_BYTES);
        assert_eq!(context_64, 7_974_200_704 + MLX_REQUEST_WAKE_BYTES);
        assert_eq!(context_512, 32_756_098_432 + MLX_REQUEST_WAKE_BYTES);
        assert_eq!(context_2048, 117_722_604_928 + MLX_REQUEST_WAKE_BYTES);
        assert!(arithmetic < context_64 && context_64 < context_512 && context_512 < context_2048);
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
        // A one-token request exercises the pre-evaluation recurrence branch without applying a
        // 256-step floor. A normal 128-token prompt remains admitted on an 8-GB machine.
        let scalar = estimate_mlx_request_bytes(1, 1, geometry, 0, 0, contract).unwrap();
        let ordinary = estimate_mlx_request_bytes(128, 128, geometry, 0, 0, contract).unwrap();
        assert_eq!(scalar, 231_142_880 + MLX_REQUEST_WAKE_BYTES);
        assert_eq!(ordinary, 2_695_981_056 + MLX_REQUEST_WAKE_BYTES);
        assert!(scalar < ordinary);
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

    // ---- The speculative engine behind the provider (epic sc-24432, story sc-24434). ----

    /// A whitespace WordLevel tokenizer over `t0..t{vocab-1}`, so every id decodes to a distinct
    /// piece and the streamed deltas are the generated ids one-for-one.
    fn word_tokenizer(vocab: usize) -> Tokenizer {
        let entries: Vec<String> = (0..vocab).map(|i| format!("\"t{i}\": {i}")).collect();
        Tokenizer::from_json(&format!(
            r#"{{"version": "1.0", "added_tokens": [], "normalizer": null,
                "pre_tokenizer": {{ "type": "Whitespace" }}, "post_processor": null,
                "decoder": null,
                "model": {{ "type": "WordLevel", "vocab": {{ {} }}, "unk_token": "t0" }} }}"#,
            entries.join(", ")
        ))
        .unwrap()
    }

    fn causal_provider() -> LlamaProvider {
        LlamaProvider::from_parts(
            crate::decode::engine::tests::causal(),
            word_tokenizer(24),
            vec![],
        )
    }

    fn qwen35_mtp_provider() -> LlamaProvider {
        let model = crate::decode::engine::tests::qwen35(true);
        let descriptor = descriptor_for_qwen35(model.config());
        LlamaProvider {
            descriptor,
            model: Decoder::Qwen35(model),
            tokenizer: word_tokenizer(50),
            template: Box::new(Llama3Template),
            stop_tokens: Vec::new(),
            constraint_table: OnceCell::new(),
            vision: None,
            gemma4: None,
            _prism_vision_weights: None,
            prefix: RefCell::new(PrefixCache::with_budget(
                core_llm::defaults::MLX.prefix_cache_bytes,
            )),
            speculative_default: core_llm::defaults::MLX.speculative,
            draft: None,
            load_report: core_llm::LoadReport::default(),
        }
    }

    /// The MoE hybrid (`qwen3_5_moe`), which admission prices on the eager contract.
    fn qwen35_moe_provider() -> LlamaProvider {
        use crate::models::qwen35::tests::{cfg_json_moe, synthetic_weights};
        let cfg = Qwen35Config::from_json(&cfg_json_moe()).unwrap();
        let model =
            Qwen35Model::from_weights(&synthetic_weights(&cfg), "model.language_model", cfg)
                .unwrap();
        LlamaProvider {
            descriptor: descriptor_for_qwen35(model.config()),
            model: Decoder::Qwen35(model),
            tokenizer: word_tokenizer(50),
            template: Box::new(Llama3Template),
            stop_tokens: Vec::new(),
            constraint_table: OnceCell::new(),
            vision: None,
            gemma4: None,
            _prism_vision_weights: None,
            prefix: RefCell::new(PrefixCache::with_budget(
                core_llm::defaults::MLX.prefix_cache_bytes,
            )),
            speculative_default: core_llm::defaults::MLX.speculative,
            draft: None,
            load_report: core_llm::LoadReport::default(),
        }
    }

    fn spec_request(speculative: core_llm::Speculative) -> TextLlmRequest {
        TextLlmRequest {
            messages: vec![Message::user("t3 t9 t4 t11 t3 t9 t4 t11 t3 t9 t4 t11")],
            sampling: Sampling::greedy(),
            max_new_tokens: 16,
            seed: Some(3),
            speculative: Some(speculative),
            ..Default::default()
        }
    }

    /// Generate, returning the output and the token ids the stream carried.
    fn run(provider: &LlamaProvider, req: &TextLlmRequest) -> (TextLlmOutput, Vec<u32>) {
        let mut ids = Vec::new();
        let out = provider
            .generate(req, &mut |ev| {
                if let CoreEvent::Token { id, .. } = ev {
                    ids.push(id);
                }
            })
            .unwrap();
        (out, ids)
    }

    /// The plain, non-engine token-at-a-time loop over the provider's own decoder and its own
    /// rendering of `req` — the reference `off` is held to.
    fn plain_loop(provider: &LlamaProvider, req: &TextLlmRequest) -> Vec<u32> {
        let prompt = provider
            .template
            .render_with(
                &req.messages,
                &RenderOptions {
                    add_generation_prompt: true,
                    enable_thinking: req.enable_thinking_kwarg(),
                    reasoning_effort: req.reasoning_effort,
                    preserve_thinking: req.preserve_thinking,
                    tools: &req.tools,
                },
            )
            .unwrap();
        let ids: Vec<i32> = provider
            .tokenizer
            .encode(&prompt, false)
            .unwrap()
            .into_iter()
            .map(|id| id as i32)
            .collect();
        let config = GenerationConfig {
            max_new_tokens: req.max_new_tokens as usize,
            sampling: map_sampling(&req.sampling),
            seed: req.seed,
            stop_tokens: provider.stop_tokens.clone(),
        };
        crate::decode::generate(
            &provider.model,
            &ids,
            &config,
            &crate::decode::CancelFlag::new(),
            &mut |_| {},
        )
        .unwrap()
        .tokens
        .into_iter()
        .map(|t| t as u32)
        .collect()
    }

    /// AC1 + AC2: `{proposer: prompt_lookup}` on the Causal decoder and `{proposer: mtp}` on the
    /// Qwen35 MTP decoder emit exactly the greedy tokens of `off`; `off` itself is the plain,
    /// non-engine loop over the same decoder and prompt; and every generation carries a populated
    /// `DecodeReport` naming the proposer, its depth, the verify steps behind the mean accepted
    /// length and the sampler path.
    #[test]
    fn speculative_requests_emit_the_off_tokens_and_off_is_the_plain_loop() {
        use core_llm::Speculative;
        // The absolute accounting of a budget-bound run without stop tokens: the first token
        // comes from the prefill and each verify step commits its accepted drafts plus one; every
        // forward is the prefill forward (one, also when the hybrid's prefix cache snapshots its
        // conversation boundary inside it, sc-24446), a verify step or a recovery replay.
        let accounting = |label: &str, ids: &[u32], report: &core_llm::DecodeReport| {
            assert_eq!(
                ids.len() as u64,
                1 + report.verify_steps + report.accepted_tokens,
                "{label}: {report:?}"
            );
            assert!(
                matches!(report.prefill_forwards, 1 | 2),
                "{label}: {report:?}"
            );
            assert_eq!(
                report.target_forwards,
                report.prefill_forwards
                    + report.verify_steps
                    + report.replay_forwards
                    + report.discarded_forwards,
                "{label}: {report:?}"
            );
        };
        for (label, provider, proposer, depth) in [
            (
                "causal prompt lookup",
                causal_provider(),
                SpeculativeProposer::PromptLookup,
                4,
            ),
            (
                "qwen35 prompt lookup",
                qwen35_mtp_provider(),
                SpeculativeProposer::PromptLookup,
                4,
            ),
            (
                "qwen35 mtp",
                qwen35_mtp_provider(),
                SpeculativeProposer::Mtp,
                3,
            ),
        ] {
            let off_req = spec_request(Speculative::Off);
            let (off, off_ids) = run(&provider, &off_req);
            let plain = plain_loop(&provider, &off_req);
            assert_eq!(off_ids, plain, "{label}: off != the plain loop");
            assert_eq!(off.text, provider.tokenizer.decode(&plain, true).unwrap());
            assert_eq!(off_ids.len(), 16, "{label}: runs to the budget");
            assert!(
                provider.stop_tokens.is_empty(),
                "{label}: accounting needs no stop tokens"
            );
            let report = off.decode.expect("off reports its decode path");
            accounting(label, &off_ids, &report);
            assert_eq!(report.path, "step_model", "{label}");
            assert_eq!(report.proposer, ProposerKind::None, "{label}");
            assert_eq!(report.draft_tokens, None, "{label}");
            assert_eq!(report.verify_steps, 15, "{label}");
            assert_eq!(report.sampler, "device", "{label}");
            // E3: no graph runner on MLX — `none`, never `eager`; and the fused primitives that
            // actually ran: the hybrid's Gated DeltaNet recurrence on the GPU stream is the fused
            // Metal kernel, the causal fixture has no fused primitive.
            assert_eq!(report.graph_path, "none", "{label}");
            let fused = if matches!(provider.model, Decoder::Qwen35(_)) {
                "fused"
            } else {
                "none"
            };
            assert_eq!(report.fused_primitives.path, fused, "{label}");
            assert_eq!(report.fused_primitives.reason, None, "{label}");
            assert!(
                report.fallbacks.is_empty(),
                "{label}: {:?}",
                report.fallbacks
            );
            assert!(off.mtp.is_none(), "{label}");
            assert!(off.timings.is_some(), "{label}: text decode is timed");

            let (spec, spec_ids) = run(
                &provider,
                &spec_request(Speculative::proposer(proposer, depth)),
            );
            assert_eq!(spec_ids, off_ids, "{label}: speculative != off");
            assert_eq!(spec.text, off.text, "{label}");
            let report = spec.decode.expect("a speculative run reports");
            accounting(label, &spec_ids, &report);
            // The lookup fixture accepts drafts, so its pin covers full-acceptance steps (the
            // MTP fixture's adversarial head never has one accepted).
            if proposer == SpeculativeProposer::PromptLookup {
                assert!(report.accepted_tokens > 0, "{label}: {report:?}");
            }
            assert_eq!(report.proposer, ProposerKind::from(proposer), "{label}");
            assert_eq!(report.path, ProposerKind::from(proposer).label(), "{label}");
            assert_eq!(report.draft_tokens, Some(depth), "{label}");
            assert!(
                report.proposed_tokens > 0,
                "{label}: the proposer ran: {report:?}"
            );
            assert!(
                report.verify_steps > 0 && report.verify_steps <= 15,
                "{label}"
            );
            assert_eq!(
                report.mean_accepted_length(),
                Some(report.accepted_tokens as f64 / report.verify_steps as f64),
                "{label}"
            );
            assert_eq!(report.sampler, "device", "{label}");
            assert!(
                report.fallbacks.is_empty(),
                "{label}: {:?}",
                report.fallbacks
            );
            assert_eq!(
                spec.mtp.is_some(),
                proposer == SpeculativeProposer::Mtp,
                "{label}"
            );
            if matches!(provider.model, Decoder::Qwen35(_)) {
                // AC3 (sc-24435): the hybrid rejected drafts and recovered every rejection
                // through its checkpoint ring — no replay forward.
                assert!(
                    report.accepted_tokens < report.proposed_tokens,
                    "{label}: a draft was rejected: {report:?}"
                );
                assert_eq!(report.replay_forwards, 0, "{label}: {report:?}");
            }
        }
    }

    /// sc-24446 (E5): `auto` reaches the engine with its acceptance monitor and an explicit
    /// proposer does not. The synthetic hybrid's random MTP head never drafts a token the target
    /// accepts, so `auto` (MTP at the recommended depth) is demoted after its first window and
    /// the report says where, while the same proposer asked for explicitly runs to the end; both
    /// stream exactly `off`'s tokens.
    #[test]
    fn auto_is_demoted_and_an_explicit_proposer_is_not() {
        use core_llm::Speculative;
        let provider = qwen35_mtp_provider();
        let request = |speculative| TextLlmRequest {
            max_new_tokens: 48,
            ..spec_request(speculative)
        };
        let (_, off_ids) = run(&provider, &request(Speculative::Off));
        // Untimed (the static threshold decides, no plain probe): the timed monitor's decisions
        // are pinned on a deterministic clock in the engine's tests.
        let (auto, auto_ids) =
            core_llm::with_decode_clock(None, || run(&provider, &request(Speculative::Auto)));
        let explicit =
            Speculative::proposer(SpeculativeProposer::Mtp, MLX_ROW.recommended_depths.mtp);
        let (asked, asked_ids) = run(&provider, &request(explicit));
        assert_eq!(auto_ids, off_ids, "a demoted run decodes plainly");
        assert_eq!(asked_ids, off_ids);
        let auto = auto.decode.unwrap();
        let asked = asked.decode.unwrap();
        assert_eq!(
            (auto.proposer, asked.proposer),
            (ProposerKind::Mtp, ProposerKind::Mtp)
        );
        assert_eq!(
            (auto.accepted_tokens, asked.accepted_tokens),
            (0, 0),
            "the fixture head never pays"
        );
        assert_eq!(
            auto.speculative_demoted_at,
            Some(1 + u64::from(core_llm::ACCEPTANCE_PROBE_VERIFIES)),
            "{auto:?}"
        );
        assert_eq!(asked.speculative_demoted_at, None, "{asked:?}");
        assert!(
            auto.proposed_tokens < asked.proposed_tokens,
            "auto stopped proposing: {} vs {}",
            auto.proposed_tokens,
            asked.proposed_tokens
        );
    }

    /// AC2: the speculative resolution's fallback reaches `DecodeReport::fallbacks` through
    /// `generate` — `auto` against a descriptor advertising no proposer, and (E2, the sc-24432
    /// feature-end review) an explicit proposer the model does not advertise, which validates and
    /// decodes plainly with the reason named rather than being refused — driven end to end;
    /// dropping the resolution's fallback anywhere between `resolve_speculative` and the output
    /// turns this red.
    #[test]
    fn the_resolution_fallback_reaches_the_decode_report() {
        use core_llm::Speculative;
        let mut provider = causal_provider();
        let (auto, _) = run(&provider, &spec_request(Speculative::Auto));
        let report = auto.decode.unwrap();
        assert_eq!(
            report.proposer,
            ProposerKind::PromptLookup,
            "auto without a head"
        );
        assert_eq!(
            report.draft_tokens,
            Some(MLX_ROW.recommended_depths.prompt_lookup)
        );
        assert!(report.fallbacks.is_empty());

        provider.descriptor.capabilities.speculative.clear();
        let (auto, ids) = run(&provider, &spec_request(Speculative::Auto));
        let (off, off_ids) = run(&provider, &spec_request(Speculative::Off));
        assert_eq!(ids, off_ids, "the fallback decodes plainly");
        assert_eq!(auto.text, off.text);
        let report = auto.decode.unwrap();
        assert_eq!(report.proposer, ProposerKind::None);
        assert_eq!(
            report.fallbacks,
            vec![
                "speculative: auto found no proposer this model can run (it advertises none; \
                 decoded without a proposer)"
                    .to_string()
            ]
        );

        // Unadvertised explicit proposers on the stock causal provider (no head, no draft).
        let provider = causal_provider();
        let (off, off_ids) = run(&provider, &spec_request(Speculative::Off));
        for proposer in [SpeculativeProposer::Mtp, SpeculativeProposer::DraftModel] {
            let req = spec_request(Speculative::proposer(proposer, 2));
            provider
                .validate(&req)
                .unwrap_or_else(|e| panic!("{proposer}: refused, not run plain: {e}"));
            let (out, ids) = run(&provider, &req);
            assert_eq!(ids, off_ids, "{proposer}: the plain path's tokens");
            assert_eq!(out.text, off.text);
            let report = out.decode.unwrap();
            assert_eq!(report.proposer, ProposerKind::None, "{proposer}");
            assert_eq!(
                report.fallbacks,
                vec![format!(
                    "speculative: `{proposer}` is not available for this model (this model \
                     does not advertise it; decoded without a proposer)"
                )]
            );
        }
    }

    /// E2: a plan the request cannot run falls back to plain decoding with the reason named,
    /// never silently and never as a failure.
    #[test]
    fn the_speculative_route_names_every_downgrade() {
        let run = |proposer, depth| SpeculativePlan::Run { proposer, depth };
        let causal = causal_provider();
        let hybrid = qwen35_mtp_provider();
        let route = |p: &LlamaProvider, plan, gemma4| {
            let mut why = Vec::new();
            let route = p.speculative_route(plan, gemma4, &mut why);
            (route, why)
        };

        assert_eq!(
            route(&causal, SpeculativePlan::Off, false),
            (SpeculativeRoute::Plain, vec![])
        );
        assert_eq!(
            route(&causal, run(SpeculativeProposer::PromptLookup, 4), false),
            (SpeculativeRoute::PromptLookup { width: 4 }, vec![])
        );
        assert_eq!(
            route(&hybrid, run(SpeculativeProposer::Mtp, 3), false),
            (SpeculativeRoute::Mtp { width: 3 }, vec![])
        );
        let (r, why) = route(&causal, run(SpeculativeProposer::Mtp, 3), false);
        assert_eq!(r, SpeculativeRoute::Plain);
        assert!(
            why[0].contains("`mtp` has no loaded MTP predictor"),
            "{why:?}"
        );
        let (r, why) = route(&hybrid, run(SpeculativeProposer::Mtp, 3), true);
        assert_eq!(r, SpeculativeRoute::Plain);
        assert!(why[0].contains("Gemma 4 multimodal"), "{why:?}");
        let (r, why) = route(&hybrid, run(SpeculativeProposer::DraftModel, 2), false);
        assert_eq!(r, SpeculativeRoute::Plain);
        assert!(
            why[0].contains("`draft_model` has no draft model"),
            "{why:?}"
        );
    }

    /// The causal fixture with its own first layer resident as the draft model (sc-24436).
    fn causal_provider_with_draft() -> LlamaProvider {
        let mut provider = causal_provider();
        provider.draft = Some(ResidentDraft {
            model: Decoder::Causal(crate::decode::engine::tests::causal_model(24, 1)),
            proposable: 24,
            width: 24,
            context: 0,
        });
        let max_depth = core_llm::verify_depth_bound(&provider.descriptor.capabilities);
        provider
            .descriptor
            .capabilities
            .speculative
            .push(core_llm::draft_model_capabilities(
                max_depth,
                &MLX_ROW.recommended_depths,
            ));
        provider
    }

    /// sc-24436: with a resident draft `draft_model` routes onto the engine for every prompt
    /// shape and decodes exactly `off`'s greedy stream with a report naming it; admission prices
    /// the draft's own prefill and cache on top of the target's (E7).
    #[test]
    fn a_resident_draft_routes_draft_model_and_is_priced() {
        use core_llm::Speculative;
        let provider = causal_provider_with_draft();
        let plan = SpeculativePlan::Run {
            proposer: SpeculativeProposer::DraftModel,
            depth: 3,
        };
        for gemma4 in [false, true] {
            let mut why = Vec::new();
            assert_eq!(
                provider.speculative_route(plan, gemma4, &mut why),
                SpeculativeRoute::DraftModel { width: 3 }
            );
            assert!(why.is_empty(), "{why:?}");
        }
        let (draft, ids) = run(
            &provider,
            &spec_request(Speculative::proposer(SpeculativeProposer::DraftModel, 3)),
        );
        let (off, off_ids) = run(&provider, &spec_request(Speculative::Off));
        assert_eq!(ids, off_ids);
        assert_eq!(draft.text, off.text);
        let report = draft.decode.unwrap();
        assert_eq!(report.proposer, ProposerKind::DraftModel);
        assert_eq!(report.draft_tokens, Some(3));
        assert!(report.proposed_tokens > 0 && report.fallbacks.is_empty());

        let price = |route| {
            provider
                .speculative_request_bytes(route, 64, 32, 0)
                .unwrap()
        };
        let lookup = price(SpeculativeRoute::PromptLookup { width: 4 });
        let with_draft = price(SpeculativeRoute::DraftModel { width: 4 });
        let draft_model = &provider.draft.as_ref().unwrap().model;
        let draft_request = estimate_mlx_request_bytes(
            64,
            32 + 4,
            draft_model.memory_geometry(),
            0,
            0,
            draft_model.workspace_contract(),
        )
        .unwrap();
        assert!(draft_request > 0);
        assert_eq!(
            with_draft,
            lookup + draft_request,
            "the target's verify overshoot plus the draft's own request"
        );
        // Without a resident draft nothing extra is priced (and the route never forms).
        assert_eq!(
            causal_provider().speculative_request_bytes(
                SpeculativeRoute::DraftModel { width: 4 },
                64,
                32,
                0
            ),
            Some(lookup)
        );
    }

    /// sc-24436 E2: a `draft_model` route whose request outruns the draft's context window — the
    /// prompt (the Gemma 4 expansion included, at its second check), the budget and a step's
    /// `K + 1` positions — runs `auto` (prompt lookup on this target) by name; within it the
    /// route stands, and an unbounded draft never falls back.
    #[test]
    fn a_draft_route_past_the_draft_context_runs_auto_by_name() {
        let mut provider = causal_provider_with_draft();
        provider.draft.as_mut().unwrap().context = 20;
        let draft = SpeculativeRoute::DraftModel { width: 3 };
        for gemma4 in [false, true] {
            let mut why = Vec::new();
            assert_eq!(
                provider.fit_draft_route(draft, 10, 6, gemma4, &mut why),
                draft
            );
            assert!(why.is_empty(), "{why:?}");
            let fallen = provider.fit_draft_route(draft, 10, 7, gemma4, &mut why);
            assert_eq!(
                fallen,
                SpeculativeRoute::PromptLookup {
                    width: MLX_ROW.recommended_depths.prompt_lookup as usize
                }
            );
            assert_eq!(why.len(), 1);
            assert!(
                why[0].contains("exceeds the draft model's context window 20"),
                "{why:?}"
            );
            // Other routes are untouched.
            let lookup = SpeculativeRoute::PromptLookup { width: 3 };
            assert_eq!(
                provider.fit_draft_route(lookup, 10, 700, gemma4, &mut why),
                lookup
            );
        }
        provider.draft.as_mut().unwrap().context = 0;
        let mut why = Vec::new();
        assert_eq!(
            provider.fit_draft_route(draft, 10, 7000, false, &mut why),
            draft
        );
    }

    /// sc-24436 E2/E7 through the real load (the sc-24432 feature-end review; the Candle twin
    /// is `a_draft_refused_by_load_admission_is_named_in_the_load_fallbacks`): a budget that fits
    /// the target but not target + draft refuses the draft by name in BOTH the load report's
    /// `draft` and its `fallbacks`, and the target loads alone.
    #[test]
    fn a_draft_refused_by_load_admission_is_named_in_the_load_fallbacks() {
        let root = crate::test_fixture::Fixture::new("mlx-llm-draft-refused-", None);
        let fixture = core_llm_testkit::write_draft_model_fixture(&root).unwrap();
        let spec = fixture.spec_with_draft();
        let target = crate::load_memory::required_bytes(&spec).unwrap();
        let draft =
            crate::load_memory::required_bytes(&LoadSpec::dense(fixture.draft.to_string_lossy()))
                .unwrap();
        let provider = with_load_budget(target + draft - 1, || LlamaProvider::load(&spec))
            .expect("the target fits and loads alone");
        let report = provider.load_report().unwrap();
        let refused = report.draft.clone().expect("the named draft is reported");
        let refusal = refused.refusal.clone().expect("refused by load admission");
        assert!(refusal.starts_with("draft model:"), "{refusal}");
        assert_eq!(refused.source, fixture.draft.to_string_lossy());
        assert!(
            report.fallbacks.contains(&refusal),
            "the refused draft is named in the load fallbacks: {:?}",
            report.fallbacks
        );
        assert!(provider
            .descriptor()
            .capabilities
            .proposer(SpeculativeProposer::DraftModel)
            .is_none());
    }

    /// sc-24436 E7: a named draft is admitted at load only beside the target; without room — or
    /// when it cannot be priced — it is refused by name and the target still loads; a target
    /// that does not fit is refused exactly as before.
    #[test]
    fn a_draft_is_admitted_at_load_only_beside_the_target() {
        let root = crate::test_fixture::Fixture::new("mlx-llm-draft-admission-", None);
        let fixture = core_llm_testkit::write_draft_model_fixture(&root).unwrap();
        let spec = fixture.spec_with_draft();
        let target = crate::load_memory::required_bytes(&spec).unwrap();
        let draft =
            crate::load_memory::required_bytes(&LoadSpec::dense(fixture.draft.to_string_lossy()))
                .unwrap();
        assert!(
            draft > 0 && draft < target,
            "the draft is the smaller model"
        );

        match DraftPlan::admit(&spec, target, target + draft).unwrap() {
            DraftPlan::Load(draft_spec) => {
                assert_eq!(draft_spec.source, fixture.draft.to_string_lossy());
                assert_eq!(draft_spec.draft_source, None);
            }
            _ => panic!("room for both loads the draft"),
        }
        match DraftPlan::admit(&spec, target, target + draft - 1).unwrap() {
            DraftPlan::Refused(report) => {
                assert!(!report.is_resident());
                assert!(report.refusal.unwrap().starts_with("draft model:"));
            }
            _ => panic!("no room beside the target refuses the draft"),
        }
        assert!(
            DraftPlan::admit(&spec, target, target - 1).is_err(),
            "the target alone"
        );
        let missing = LoadSpec::dense(fixture.target.to_string_lossy())
            .with_draft(root.join("no-such-draft").to_string_lossy());
        match DraftPlan::admit(&missing, target, u64::MAX).unwrap() {
            DraftPlan::Refused(report) => {
                assert!(report.refusal.unwrap().contains("cannot be priced"))
            }
            _ => panic!("an unpriceable draft is refused"),
        }
        let unnamed = LoadSpec::dense(fixture.target.to_string_lossy());
        assert!(matches!(
            DraftPlan::admit(&unnamed, target, target).unwrap(),
            DraftPlan::None
        ));

        // End to end, an unloadable draft never fails the target's load.
        let provider = LlamaProvider::load(&missing).unwrap();
        let report = provider.load_report().unwrap().draft.unwrap();
        assert!(!report.is_resident());
        assert!(provider
            .descriptor()
            .capabilities
            .proposer(SpeculativeProposer::DraftModel)
            .is_none());
    }

    /// Both decoder families advertise prompt lookup (the engine runs it on either) at their
    /// geometry's depth; the Qwen3.8 head keeps its MTP advertisement — the same finite depth on
    /// the legacy field and the per-proposer list (sc-24438) — and a depth past the bound is
    /// admitted (it is clamped at generate, by name).
    #[test]
    fn every_decoder_advertises_prompt_lookup() {
        use core_llm::Speculative;
        for provider in [causal_provider(), qwen35_mtp_provider()] {
            let caps = &provider.descriptor().capabilities;
            let lookup = caps.proposer(SpeculativeProposer::PromptLookup).unwrap();
            // The fixtures' head dims (4, 8) are not vector-kernel dims: the 8-row bound.
            assert_eq!(lookup.max_depth, SPECULATIVE_MAX_DEPTH);
            assert_eq!(
                lookup.recommended_depth,
                MLX_ROW.recommended_depths.prompt_lookup
            );
            let too_deep = spec_request(Speculative::proposer(
                SpeculativeProposer::PromptLookup,
                lookup.max_depth + 1,
            ));
            provider.validate(&too_deep).unwrap();
        }
        let hybrid = qwen35_mtp_provider();
        let caps = &hybrid.descriptor().capabilities;
        let mtp = caps.proposer(SpeculativeProposer::Mtp).unwrap();
        assert_eq!(
            (mtp.max_depth, mtp.recommended_depth),
            (SPECULATIVE_MAX_DEPTH, MLX_ROW.recommended_depths.mtp)
        );
        assert_eq!(
            caps.mtp,
            Some(core_llm::MtpCapabilities {
                max_draft_tokens: mtp.max_depth,
                recommended_draft_tokens: mtp.recommended_depth,
            }),
            "the legacy field agrees with the proposer list"
        );
        assert!(caps.speculative.contains(&mtp));
        assert!(causal_provider()
            .descriptor()
            .capabilities
            .proposer(SpeculativeProposer::Mtp)
            .is_none());
    }

    /// sc-24438 AC1 (E4): a loaded model's advertised depth is its attention geometry's — one less
    /// than the rows MLX's decode kernel verifies in one call — for both proposers and on the
    /// legacy MTP field: the shipped Qwen3.8-27B config (24q/4kv/hd 256) → 4, a 35B-A3B-shaped
    /// MoE config (16q/2kv/hd 256) → 3, a gqa-4 hd-128 Llama → 7, a gqa-8 one → 3, Gemma 4 the
    /// narrower of its sliding / full layers, DeepSeek MLA (192/128, head-expanded keys) → 7.
    #[test]
    fn the_advertised_depth_is_the_attention_geometrys() {
        let depths = |caps: &TextLlmCapabilities| {
            let lookup = caps.proposer(SpeculativeProposer::PromptLookup).unwrap();
            assert!(lookup.recommended_depth <= lookup.max_depth);
            let mtp = caps.proposer(SpeculativeProposer::Mtp).map(|m| {
                assert_eq!(
                    caps.mtp,
                    Some(core_llm::MtpCapabilities {
                        max_draft_tokens: m.max_depth,
                        recommended_draft_tokens: m.recommended_depth,
                    })
                );
                m.max_depth
            });
            (lookup.max_depth, mtp)
        };
        let qwen38: serde_json::Value = serde_json::from_str(include_str!(concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/../../../docs/reference/qwen38/config.json"
        )))
        .unwrap();
        let dense = Qwen35Config::from_json(&qwen38).unwrap();
        assert_eq!(
            (dense.num_heads, dense.num_kv_heads, dense.head_dim),
            (24, 4, 256)
        );
        assert_eq!(
            depths(&descriptor_for_qwen35(&dense).capabilities),
            (4, Some(4))
        );
        let mut a3b = qwen38.clone();
        let text = a3b["text_config"].as_object_mut().unwrap();
        text.insert("num_attention_heads".into(), json!(16));
        text.insert("num_key_value_heads".into(), json!(2));
        text.insert("model_type".into(), json!("qwen3_5_moe_text"));
        text.insert("num_experts".into(), json!(256));
        text.insert("num_experts_per_tok".into(), json!(8));
        text.insert("moe_intermediate_size".into(), json!(512));
        text.insert("shared_expert_intermediate_size".into(), json!(512));
        let a3b = Qwen35Config::from_json(&a3b).unwrap();
        assert!(a3b.moe.is_some());
        assert_eq!(
            depths(&descriptor_for_qwen35(&a3b).capabilities),
            (3, Some(3))
        );

        let llama = |heads: i32, kv: i32, extra: serde_json::Value| {
            let mut v = json!({
                "model_type": "llama", "architectures": ["LlamaForCausalLM"],
                "hidden_size": heads * 128, "intermediate_size": 256, "num_hidden_layers": 2,
                "num_attention_heads": heads, "num_key_value_heads": kv,
                "vocab_size": 32, "rms_norm_eps": 1e-5, "max_position_embeddings": 64
            });
            for (k, x) in extra.as_object().unwrap() {
                v[k] = x.clone();
            }
            depths(&descriptor_for(&ModelConfig::from_json(&v).unwrap()).capabilities)
        };
        assert_eq!(llama(32, 8, json!({})), (7, None));
        assert_eq!(llama(64, 8, json!({})), (3, None));
        assert_eq!(
            llama(
                16,
                16,
                json!({"model_type": "deepseek_v2", "architectures": ["DeepseekV2ForCausalLM"],
                       "kv_lora_rank": 64, "qk_nope_head_dim": 128, "qk_rope_head_dim": 64,
                       "v_head_dim": 128})
            ),
            (7, None)
        );

        // Gemma 4: sliding hd 256 over 2 KV heads (gqa 8 → 4 rows) beats the full layers' hd 512.
        let gemma4 = |kv: i32| {
            let cfg = ModelConfig {
                num_heads: 16,
                gemma4: Some(Box::new(crate::config::Gemma4Config {
                    layer_types: vec![
                        crate::config::LayerAttentionType::Sliding,
                        crate::config::LayerAttentionType::Full,
                    ],
                    sliding: crate::config::LayerAttention {
                        head_dim: 256,
                        num_kv_heads: kv,
                        rope_type: crate::config::RopeType::Default,
                        rope_theta: 1e4,
                        partial_rotary_factor: 1.0,
                        rope_factor: 1.0,
                        sliding_window: Some(512),
                        k_eq_v: false,
                    },
                    full: crate::config::LayerAttention {
                        head_dim: 512,
                        num_kv_heads: 1,
                        rope_type: crate::config::RopeType::Default,
                        rope_theta: 1e6,
                        partial_rotary_factor: 1.0,
                        rope_factor: 1.0,
                        sliding_window: None,
                        k_eq_v: true,
                    },
                    bidirectional: None,
                    num_kv_shared_layers: 0,
                    use_double_wide_mlp: false,
                })),
                num_layers: 2,
                ..crate::decode::engine::tests::tiny_llama(8).config().clone()
            };
            depths(&descriptor_for(&cfg).capabilities)
        };
        assert_eq!(gemma4(2), (3, None));
        assert_eq!(gemma4(8), (7, None));
    }

    /// sc-24438 AC1: the advertised max depth is finite and backend-true (on these fixtures'
    /// head dims, the 8-row bound) and a request above it runs at it, with the clamp named in
    /// `DecodeReport::fallbacks`, emitting exactly `off`'s stream: prompt lookup on both
    /// decoders, and MTP (the new option and the legacy field) on the Qwen3.8 head.
    #[test]
    fn a_too_deep_request_runs_at_the_advertised_max_and_names_the_clamp() {
        use core_llm::{MtpMode, Speculative};
        let clamp = |proposer: &str, asked: u32, max: u32| {
            vec![format!(
                "speculative: `{proposer}` depth {asked} clamped to {max} (advertised 1..={max})"
            )]
        };
        let check = |provider: &LlamaProvider, req: TextLlmRequest, kind, max, label: &str| {
            let (_, off_ids) = run(provider, &spec_request(Speculative::Off));
            let (out, ids) = run(provider, &req);
            let report = out.decode.unwrap();
            assert_eq!(report.proposer, kind, "{label}");
            assert_eq!(report.draft_tokens, Some(max), "{label}");
            assert_eq!(ids, off_ids, "{label}: the clamped run is greedy-exact");
            report.fallbacks
        };
        let max = |provider: &LlamaProvider, proposer| {
            provider
                .descriptor()
                .capabilities
                .proposer(proposer)
                .unwrap()
                .max_depth
        };

        for (label, provider) in [
            ("causal", causal_provider()),
            ("qwen35", qwen35_mtp_provider()),
        ] {
            let max = max(&provider, SpeculativeProposer::PromptLookup);
            let asked = max + 5;
            let req = spec_request(Speculative::proposer(
                SpeculativeProposer::PromptLookup,
                asked,
            ));
            let fallbacks = check(&provider, req, ProposerKind::PromptLookup, max, label);
            assert_eq!(fallbacks, clamp("prompt_lookup", asked, max), "{label}");
        }

        let hybrid = qwen35_mtp_provider();
        let mtp_max = max(&hybrid, SpeculativeProposer::Mtp);
        let new = spec_request(Speculative::proposer(SpeculativeProposer::Mtp, 40));
        let fallbacks = check(&hybrid, new, ProposerKind::Mtp, mtp_max, "mtp");
        assert_eq!(fallbacks, clamp("mtp", 40, mtp_max));
        let mut legacy = spec_request(Speculative::Off);
        legacy.speculative = None;
        legacy.mtp = Some(MtpMode::Enabled { draft_tokens: 40 });
        let fallbacks = check(&hybrid, legacy, ProposerKind::Mtp, mtp_max, "legacy");
        assert_eq!(fallbacks, clamp("mtp", 40, mtp_max));
    }

    /// sc-24438 AC2: a sparse-MoE Qwen35 snapshot carrying an MTP head — its predictor layer a
    /// sparse-MoE block, as the 35B-A3B ships it, in the fused (Qwen3.6) and the per-expert
    /// (Qwen3.5) expert layout — loads through the production load path, advertises MTP on both
    /// fields, and **runs** it: `{mtp}` at depths 1, 3 and the advertised max, the legacy
    /// `enabled 3` and `auto` each report the MTP proposer with drafts proposed, and emit exactly
    /// `off`'s stream.
    #[test]
    fn a_moe_snapshot_with_an_mtp_head_runs_mtp() {
        use crate::models::qwen35::tests::{cfg_json_moe_mtp, seeded_moe_mtp_tensors};
        use core_llm::{MtpMode, Speculative};
        for fused in [true, false] {
            let dir = tempfile::tempdir().unwrap();
            let mut config = cfg_json_moe_mtp();
            config["model_type"] = json!("qwen3_5_moe");
            config.as_object_mut().unwrap().remove("vision_config");
            std::fs::write(dir.path().join("config.json"), config.to_string()).unwrap();
            let entries: Vec<String> = (0..50).map(|i| format!("\"t{i}\": {i}")).collect();
            std::fs::write(
                dir.path().join("tokenizer.json"),
                format!(
                    r#"{{"version": "1.0", "added_tokens": [], "normalizer": null,
                    "pre_tokenizer": {{ "type": "Whitespace" }}, "post_processor": null,
                    "decoder": null,
                    "model": {{ "type": "WordLevel", "vocab": {{ {} }}, "unk_token": "t0" }} }}"#,
                    entries.join(", ")
                ),
            )
            .unwrap();
            let cfg = Qwen35Config::from_json(&config).unwrap();
            let tensors = seeded_moe_mtp_tensors(&cfg, fused);
            let refs: Vec<(&str, &Array)> = tensors.iter().map(|(k, v)| (k.as_str(), v)).collect();
            Array::save_safetensors(refs, None, dir.path().join("model.safetensors")).unwrap();

            let label = if fused { "fused" } else { "per-expert" };
            let provider = LlamaProvider::load(&LoadSpec::dense(dir.path().display().to_string()))
                .expect("a MoE checkpoint with an MTP head loads");
            let caps = &provider.descriptor().capabilities;
            let mtp = caps.proposer(SpeculativeProposer::Mtp).expect(label);
            assert_eq!(
                caps.mtp,
                Some(core_llm::MtpCapabilities {
                    max_draft_tokens: mtp.max_depth,
                    recommended_draft_tokens: mtp.recommended_depth,
                }),
                "{label}"
            );

            let (_, off_ids) = run(&provider, &spec_request(Speculative::Off));
            let mut legacy = spec_request(Speculative::Off);
            legacy.speculative = None;
            legacy.mtp = Some(MtpMode::Enabled { draft_tokens: 3 });
            // Epic AT1: the explicit option at depths 1, 3 and the advertised max.
            let explicit = |depth| {
                (
                    depth,
                    spec_request(Speculative::proposer(SpeculativeProposer::Mtp, depth)),
                )
            };
            for (case, (depth, req)) in [
                ("explicit 1", explicit(1)),
                ("explicit 3", explicit(3)),
                ("explicit max", explicit(mtp.max_depth)),
                ("legacy", (3, legacy)),
                // `auto` runs at the advertised recommended depth (sc-24446), not a literal.
                (
                    "auto",
                    (mtp.recommended_depth, spec_request(Speculative::Auto)),
                ),
            ] {
                provider.validate(&req).unwrap();
                let (out, ids) = run(&provider, &req);
                let report = out.decode.unwrap();
                assert_eq!(report.proposer, ProposerKind::Mtp, "{label} {case}");
                assert_eq!(report.draft_tokens, Some(depth), "{label} {case}");
                assert!(
                    report.proposed_tokens > 0,
                    "{label} {case}: the head drafted"
                );
                assert!(report.fallbacks.is_empty(), "{label} {case}");
                assert_eq!(ids, off_ids, "{label} {case}: greedy-exact");
            }
        }
    }

    /// Admission prices what each route holds: prompt lookup on a softmax decoder pays only its
    /// verify overshoot (it rolls back by truncation). On the hybrid every speculative route pays
    /// the DeltaNet checkpoint ring its rollback holds (E7, sc-24435) — prompt lookup the ring plus
    /// the overshoot, MTP the ring plus its predictor cache and verify rows — and the recurrent
    /// state once: no step-start snapshot, no clone/replay copies.
    #[test]
    fn speculative_admission_prices_the_rollback_the_route_holds() {
        let causal = causal_provider();
        let price =
            |p: &LlamaProvider, route| p.speculative_request_bytes(route, 64, 32, 0).unwrap();
        let off = price(&causal, SpeculativeRoute::Plain);
        let lookup = price(&causal, SpeculativeRoute::PromptLookup { width: 4 });
        let g = causal.model.memory_geometry();
        let kv_position = g.layers * g.kv_heads * g.head_dim * g.element_bytes * 2;
        let rows = (g.hidden_size + g.vocab_size) * g.element_bytes;
        assert_eq!(
            lookup,
            off + 4 * (kv_position + rows),
            "truncation: overshoot only"
        );

        let hybrid = qwen35_mtp_provider();
        let Decoder::Qwen35(model) = &hybrid.model else {
            unreachable!("the hybrid fixture")
        };
        let off = price(&hybrid, SpeculativeRoute::Plain);
        let g = hybrid.model.memory_geometry();
        let kv_position = g.layers * g.kv_heads * g.head_dim * g.element_bytes * 2;
        let rows = (g.hidden_size + g.vocab_size) * g.element_bytes;
        let kv = (64 + 32) * kv_position;
        for width in [1usize, 3, 8] {
            let ring = model.checkpoint_ring_bytes(width).unwrap();
            let w = width as u64;
            assert!(ring > 0);
            assert_eq!(
                price(&hybrid, SpeculativeRoute::PromptLookup { width }),
                off + ring + w * (kv_position + rows),
                "lookup {width}: the ring plus the overshoot"
            );
            assert_eq!(
                price(&hybrid, SpeculativeRoute::Mtp { width }),
                off + ring + 2 * kv + w * rows,
                "mtp {width}: the ring plus the predictor cache and verify rows"
            );
        }

        // The MoE hybrid prices on the eager contract, with the same once-charged recurrent term.
        let moe = qwen35_moe_provider();
        let Decoder::Qwen35(model) = &moe.model else {
            unreachable!("the MoE fixture")
        };
        assert!(matches!(
            moe.model.workspace_contract(),
            MlxWorkspaceContract::Eager
        ));
        let off = price(&moe, SpeculativeRoute::Plain);
        let g = moe.model.memory_geometry();
        let kv_position = g.layers * g.kv_heads * g.head_dim * g.element_bytes * 2;
        let rows = (g.hidden_size + g.vocab_size) * g.element_bytes;
        let kv = (64 + 32) * kv_position;
        for width in [1usize, 3, 8] {
            let ring = model.checkpoint_ring_bytes(width).unwrap();
            let w = width as u64;
            assert_eq!(
                price(&moe, SpeculativeRoute::PromptLookup { width }),
                off + ring + w * (kv_position + rows),
                "moe lookup {width}"
            );
            // The generic runtime terms also hold the verify width's K/V positions and their
            // block padding (sc-24446).
            let positions = |extra: u64| w + kv_block_padding(96 + extra).unwrap();
            assert_eq!(
                price(&moe, SpeculativeRoute::Mtp { width }),
                off + ring
                    + 2 * kv
                    + w * rows
                    + (positions(w) - (kv_block_padding(96).unwrap())) * kv_position,
                "moe mtp {width}: the ring and the live state charged once"
            );
        }

        // A draft model (sc-24436) on the hybrid target verifies like prompt lookup: the ring
        // plus the overshoot. A hybrid draft holds its own proposal window in its ring, armed for
        // the same width: its request prices that ring, never a snapshot.
        for width in [1usize, 4] {
            assert_eq!(
                price(&hybrid, SpeculativeRoute::DraftModel { width }),
                price(&hybrid, SpeculativeRoute::PromptLookup { width }),
                "hybrid target, draft {width}: the ring plus the overshoot"
            );
        }
        let mut drafted = causal_provider();
        drafted.draft = Some(ResidentDraft {
            model: Decoder::Qwen35(crate::decode::engine::tests::qwen35(false)),
            proposable: 24,
            width: 24,
            context: 0,
        });
        let Some(Decoder::Qwen35(draft)) = drafted.draft.as_ref().map(|d| &d.model) else {
            unreachable!("the hybrid draft")
        };
        for width in [1usize, 4] {
            let mut geometry = drafted.draft.as_ref().unwrap().model.memory_geometry();
            geometry.recurrent_bytes += draft.checkpoint_ring_bytes(width).unwrap();
            let draft_request = estimate_mlx_request_bytes(
                64,
                32 + width as u32,
                geometry,
                0,
                0,
                drafted.draft.as_ref().unwrap().model.workspace_contract(),
            )
            .unwrap();
            assert_eq!(
                price(&drafted, SpeculativeRoute::DraftModel { width }),
                price(&causal, SpeculativeRoute::PromptLookup { width }) + draft_request,
                "hybrid draft {width}: its request with its ring, no snapshot"
            );
        }
    }

    // ---- Prism/Bonsai companion MTP head (epic sc-24432, story sc-24444). ----

    /// A Prism snapshot directory and, when `head` names a variant, a companion head beside it.
    struct PrismLoad {
        target: crate::test_fixture::Fixture,
        head: crate::test_fixture::Fixture,
    }

    fn prism_load(head_text: Option<serde_json::Value>) -> PrismLoad {
        let target = crate::test_fixture::Fixture::new("mlx-prism-target-", None);
        // Seed 6 decodes a varied greedy run (not a single repeated id) through the chat template.
        crate::synthetic::write_prism_snapshot(&target, 6);
        let head = crate::test_fixture::Fixture::new("mlx-prism-head-", None);
        let text = head_text
            .unwrap_or_else(|| crate::synthetic::prism_qwen35_config()["text_config"].clone());
        crate::synthetic::write_companion_head(&head, &text, 5, None);
        PrismLoad { target, head }
    }

    fn prism_request(speculative: core_llm::Speculative) -> TextLlmRequest {
        TextLlmRequest {
            messages: vec![Message::user("t3 t9 t40 t11 t3 t9 t40 t11 t7")],
            sampling: Sampling::greedy(),
            max_new_tokens: 20,
            seed: Some(3),
            speculative: Some(speculative),
            ..Default::default()
        }
    }

    /// E2 through the real load (sc-24432 feature-end review; the Candle twin is
    /// `a_configured_mtp_head_the_snapshot_cannot_run_is_a_named_load_fallback`): a snapshot whose
    /// config declares a native MTP head it stores no tensor of loads the target plain, names the
    /// `mtp:` fallback in the load report, advertises no MTP and decodes.
    #[test]
    fn a_configured_mtp_head_the_snapshot_does_not_carry_is_a_named_load_fallback() {
        let target = crate::test_fixture::Fixture::new("mlx-mtp-absent-", None);
        crate::synthetic::write_prism_snapshot(&target, 6);
        let path = target.join("config.json");
        let mut config: serde_json::Value =
            serde_json::from_slice(&std::fs::read(&path).unwrap()).unwrap();
        config["text_config"]["mtp_num_hidden_layers"] = serde_json::json!(1);
        std::fs::write(&path, config.to_string()).unwrap();
        let provider = LlamaProvider::load(&LoadSpec::dense(target.to_str().unwrap()))
            .expect("the target still loads (E2)");
        let fallbacks = provider.load_report().unwrap().fallbacks;
        assert_eq!(fallbacks.len(), 1, "{fallbacks:?}");
        assert!(
            fallbacks[0].starts_with("mtp: ") && fallbacks[0].contains("stores no `mtp.*` tensor"),
            "{fallbacks:?}"
        );
        assert!(provider
            .descriptor()
            .capabilities
            .proposer(SpeculativeProposer::Mtp)
            .is_none());
        assert!(provider.descriptor().capabilities.mtp.is_none());
        let (_, ids) = run(&provider, &prism_request(core_llm::Speculative::Off));
        assert_eq!(ids.len(), 20);
    }

    /// Story sc-24444 AC2: a Prism snapshot loaded with a companion head advertises MTP, and
    /// `{proposer: mtp}` at depths 1, 3 and the advertised max — through the real load path and the provider's
    /// engine — emits exactly `off`'s greedy tokens, with the report naming the head as the
    /// proposer that ran.
    #[test]
    fn prism_with_a_companion_head_decodes_mtp_with_the_off_tokens() {
        use core_llm::Speculative;
        let fx = prism_load(None);
        let spec =
            LoadSpec::dense(fx.target.to_str().unwrap()).with_mtp_head(fx.head.to_str().unwrap());
        let provider = LlamaProvider::load(&spec).unwrap();
        assert_eq!(provider.descriptor().family, "prism_hadamard_qwen35");
        let report = provider.load_report().unwrap();
        assert!(report.fallbacks.is_empty(), "{:?}", report.fallbacks);
        // The companion head advertises the same backend-true depth a native head does (sc-24438).
        let advertised = provider
            .descriptor()
            .capabilities
            .proposer(SpeculativeProposer::Mtp)
            .expect("the attached head is advertised");
        assert_eq!(advertised.recommended_depth, MLX_ROW.recommended_depths.mtp);

        let (off, off_ids) = run(&provider, &prism_request(Speculative::Off));
        assert_eq!(off_ids.len(), 20);
        let distinct: std::collections::BTreeSet<_> = off_ids.iter().collect();
        assert!(
            distinct.len() > 2,
            "fixture must not be degenerate: {off_ids:?}"
        );
        assert_eq!(
            off_ids,
            plain_loop(&provider, &prism_request(Speculative::Off))
        );
        assert_eq!(off.decode.unwrap().proposer, ProposerKind::None);
        for depth in [1, 3, advertised.max_depth] {
            let req = prism_request(Speculative::proposer(SpeculativeProposer::Mtp, depth));
            let (out, ids) = run(&provider, &req);
            assert_eq!(ids, off_ids, "depth {depth}");
            let report = out.decode.expect("the engine reports its path");
            assert_eq!(report.proposer, ProposerKind::Mtp, "depth {depth}");
            assert_eq!(report.draft_tokens, Some(depth), "depth {depth}");
            assert!(
                report.proposed_tokens > 0,
                "depth {depth}: the head drafted"
            );
            assert!(report.fallbacks.is_empty(), "{:?}", report.fallbacks);
        }

        // Without the head the same snapshot has no MTP, and loads the same greedy tokens.
        let bare = LlamaProvider::load(&LoadSpec::dense(fx.target.to_str().unwrap())).unwrap();
        assert!(bare.load_report().unwrap().fallbacks.is_empty());
        assert!(bare
            .descriptor()
            .capabilities
            .proposer(SpeculativeProposer::Mtp)
            .is_none());
        assert_eq!(run(&bare, &prism_request(Speculative::Off)).1, off_ids);
    }

    /// Story sc-24444 AC2 / E2: a head whose geometry does not match, a missing head, and a head
    /// named for a non-Qwen3.5 target are each refused with a named load fallback; the model still
    /// loads, advertises no MTP and decodes plainly.
    #[test]
    fn an_unattachable_companion_head_is_a_named_load_fallback_never_a_load_failure() {
        use core_llm::Speculative;
        let mut text = crate::synthetic::prism_qwen35_config()["text_config"].clone();
        text["intermediate_size"] = json!(128);
        let fx = prism_load(Some(text));
        let spec =
            LoadSpec::dense(fx.target.to_str().unwrap()).with_mtp_head(fx.head.to_str().unwrap());
        let provider = LlamaProvider::load(&spec).expect("the target still loads (E2)");
        let fallbacks = provider.load_report().unwrap().fallbacks;
        assert_eq!(fallbacks.len(), 1, "{fallbacks:?}");
        assert!(fallbacks[0].starts_with("mtp_head: "), "{fallbacks:?}");
        assert!(
            fallbacks[0].contains("intermediate_size 128 != target 256"),
            "{fallbacks:?}"
        );
        assert!(
            fallbacks[0].contains("the model loaded without a companion head"),
            "{fallbacks:?}"
        );
        assert!(provider
            .descriptor()
            .capabilities
            .proposer(SpeculativeProposer::Mtp)
            .is_none());
        let (out, ids) = run(&provider, &prism_request(Speculative::Off));
        assert_eq!(ids.len(), 20);
        assert!(out.decode.unwrap().fallbacks.is_empty());
        // An explicit MTP request (not advertised) decodes plainly with the reason named (E2,
        // never refused), and `auto` names why it ran no MTP by choosing prompt lookup.
        let (mtp, mtp_ids) = run(
            &provider,
            &prism_request(Speculative::proposer(SpeculativeProposer::Mtp, 3)),
        );
        assert_eq!(mtp_ids, ids, "the plain path's tokens");
        let mtp = mtp.decode.unwrap();
        assert_eq!(mtp.proposer, ProposerKind::None);
        assert_eq!(mtp.fallbacks.len(), 1, "{:?}", mtp.fallbacks);
        assert!(
            mtp.fallbacks[0].starts_with("speculative: `mtp` is not available"),
            "{:?}",
            mtp.fallbacks
        );
        let (auto, auto_ids) = run(&provider, &prism_request(Speculative::Auto));
        assert_eq!(auto_ids, ids);
        assert_eq!(auto.decode.unwrap().proposer, ProposerKind::PromptLookup);

        let missing = LoadSpec::dense(fx.target.to_str().unwrap())
            .with_mtp_head(fx.target.join("no-such-head").to_str().unwrap());
        let provider = LlamaProvider::load(&missing).expect("a missing head is not fatal");
        let fallbacks = provider.load_report().unwrap().fallbacks;
        assert_eq!(fallbacks.len(), 1, "{fallbacks:?}");
        assert!(fallbacks[0].contains("is not a directory"), "{fallbacks:?}");

        let mut causal = causal_provider();
        causal.attach_mtp_head(Some(fx.head.as_ref()));
        let fallbacks = &causal.load_report.fallbacks;
        assert_eq!(fallbacks.len(), 1, "{fallbacks:?}");
        assert!(
            fallbacks[0].contains("attach to Qwen3.5/3.8-family"),
            "{fallbacks:?}"
        );
        assert!(causal.descriptor.capabilities.mtp.is_none());
    }

    /// Story sc-24444 E7: the head's resident bytes are its safetensors payload plus the header's
    /// norm intermediates, and load admission adds them to the target's: a budget that fits the
    /// target but not target + head refuses the head by name and nothing else. The head's
    /// per-request cache is priced by request admission on the `mtp` route.
    /// Story sc-24444 E7 with sc-24436/sc-24437: the companion head, a draft model and the
    /// prefix cache are admitted together — the head on top of the target and the admitted
    /// draft, the prefix cache in what they leave. A budget that fits target + draft but not the
    /// head too keeps the draft, refuses the head by name and gives the cache the rest; one that
    /// fits all three plus `x` bytes settles the prefix cache at exactly `x`.
    #[test]
    fn companion_head_draft_and_prefix_cache_are_admitted_together() {
        let fx = prism_load(None);
        let head: &Path = fx.head.as_ref();
        let head_bytes = crate::load_memory::companion_head_bytes(head).unwrap();
        let target =
            crate::load_memory::required_bytes(&LoadSpec::dense(fx.target.to_str().unwrap()))
                .unwrap();
        let mut spec = LoadSpec::dense(fx.target.to_str().unwrap())
            .with_mtp_head(fx.head.to_str().unwrap())
            .with_draft(fx.target.to_str().unwrap());
        spec.prefix_cache_bytes = Some(u64::MAX / 4);
        let draft = crate::load_memory::required_bytes(&draft_load_spec(
            &spec,
            fx.target.to_str().unwrap(),
        ))
        .unwrap();

        let short = with_load_budget(target + draft + head_bytes - 1, || {
            LlamaProvider::load(&spec)
        })
        .expect("target + draft fit");
        let report = short.load_report().unwrap();
        assert!(report.draft.as_ref().unwrap().is_resident(), "{report:?}");
        assert_eq!(report.fallbacks.len(), 1, "{:?}", report.fallbacks);
        assert!(
            report.fallbacks[0].starts_with("mtp_head: refused by load admission"),
            "{:?}",
            report.fallbacks
        );
        assert_eq!(
            report.prefix_cache_bytes,
            Some(head_bytes - 1),
            "the refused head's room is the cache's"
        );

        let x = 4096;
        let all = with_load_budget(target + draft + head_bytes + x, || {
            LlamaProvider::load(&spec)
        })
        .expect("all three fit");
        let report = all.load_report().unwrap();
        assert!(report.fallbacks.is_empty(), "{:?}", report.fallbacks);
        assert!(report.draft.as_ref().unwrap().is_resident(), "{report:?}");
        assert_eq!(report.prefix_cache_bytes, Some(x));
        assert!(all
            .descriptor()
            .capabilities
            .proposer(SpeculativeProposer::Mtp)
            .is_some());
    }

    #[test]
    fn companion_head_bytes_are_counted_by_load_and_request_admission() {
        let fx = prism_load(None);
        let head: &Path = fx.head.as_ref();
        let head_bytes = crate::load_memory::companion_head_bytes(head).unwrap();
        let payload = core_llm::checkpoint_payload_bytes(head).unwrap();
        // Seven 1-D norm vectors (4 × 128 + 2 × 64 + 128 wide) at four intermediate bytes each,
        // and one 16 KiB page of rounding per stored tensor and per norm result (sc-24446).
        let norms = (5 * 128 + 2 * 64) * 4;
        let tensors = crate::load_memory::safetensors_headers(head)
            .unwrap()
            .iter()
            .flat_map(|h| h.as_object().unwrap().keys())
            .filter(|k| *k != "__metadata__")
            .count() as u64;
        assert_eq!(head_bytes, payload + norms + (tensors + 7) * 16 * 1024);

        let target =
            crate::load_memory::required_bytes(&LoadSpec::dense(fx.target.to_str().unwrap()))
                .unwrap();
        let with_head =
            LoadSpec::dense(fx.target.to_str().unwrap()).with_mtp_head(fx.head.to_str().unwrap());
        assert_eq!(
            crate::load_memory::required_bytes(&with_head).unwrap(),
            target,
            "the target's own price is unchanged; the head is admitted on top of it"
        );
        assert!(admit_companion_head(head, target, target + head_bytes).is_ok());
        let refused = admit_companion_head(head, target, target + head_bytes - 1).unwrap_err();
        assert!(
            refused.starts_with("mtp_head: refused by load admission"),
            "{refused}"
        );
        assert!(refused.contains(&head_bytes.to_string()), "{refused}");

        // Through the real load: a budget that fits the target but not target + head refuses the
        // head by name and loads the target without MTP; one more byte admits both.
        let short = with_load_budget(target + head_bytes - 1, || LlamaProvider::load(&with_head))
            .expect("the target fits and still loads (E2)");
        let fallbacks = short.load_report().unwrap().fallbacks;
        assert_eq!(fallbacks.len(), 1, "{fallbacks:?}");
        assert!(
            fallbacks[0].starts_with("mtp_head: refused by load admission"),
            "{fallbacks:?}"
        );
        assert!(short
            .descriptor()
            .capabilities
            .proposer(SpeculativeProposer::Mtp)
            .is_none());
        let fits = with_load_budget(target + head_bytes, || LlamaProvider::load(&with_head))
            .expect("target + head fit");
        // Nothing beside target + head is left for the prefix cache: it settles to 0 bytes,
        // named (E2), and the head attaches.
        let load = fits.load_report().unwrap();
        assert_eq!(load.prefix_cache_bytes, Some(0));
        assert_eq!(
            load.fallbacks,
            core_llm::prefix_budget_fallback(core_llm::DecodeBackend::Mlx, None, 0)
                .into_iter()
                .collect::<Vec<_>>()
        );
        assert!(fits
            .descriptor()
            .capabilities
            .proposer(SpeculativeProposer::Mtp)
            .is_some());

        let provider = LlamaProvider::load(&with_head).unwrap();
        // The prefix-cache snapshot of an MTP request carries the head's KV too (sc-24437).
        let snapshot = |mtp| provider.prefix_snapshot_bytes(Some(64), mtp).unwrap();
        assert!(
            snapshot(true) > snapshot(false) + 4 * 128,
            "the head's KV is priced"
        );
        let price = |route| {
            provider
                .speculative_request_bytes(route, 64, 32, 0)
                .unwrap()
        };
        assert!(
            price(SpeculativeRoute::Mtp { width: 3 }) > price(SpeculativeRoute::Plain),
            "the head's cache and rollback are priced on the mtp route"
        );
    }

    /// E1/E8: the backend-neutral greedy parity suite (core-llm-testkit) — the one Candle runs —
    /// holds on both MLX decoders: prompt lookup at depths 1, 3, the recommended 4 and the max,
    /// `auto`, and on the Qwen3.8 head MTP at depths 1 and 3, each emitting exactly `off`'s
    /// stream with a report naming its proposer; and something was actually drafted and accepted.
    #[test]
    fn the_backend_neutral_parity_suite_holds_on_both_decoders() {
        use core_llm::Speculative;
        use core_llm_testkit::{BenchPrompt, ParityCase, PromptClass};
        let prompts = vec![
            BenchPrompt::user(
                "repeat",
                PromptClass::Predictable,
                "t3 t9 t4 t11 t3 t9 t4 t11 t3 t9 t4 t11",
            ),
            BenchPrompt::user("sparse", PromptClass::OpenEnded, "t5 t8 t1 t20 t13"),
        ];
        let lookup = |depth| ParityCase {
            speculative: Speculative::proposer(SpeculativeProposer::PromptLookup, depth),
            expect_proposer: ProposerKind::PromptLookup,
        };
        let mtp = |depth| ParityCase {
            speculative: Speculative::proposer(SpeculativeProposer::Mtp, depth),
            expect_proposer: ProposerKind::Mtp,
        };
        let auto = |expect_proposer| ParityCase {
            speculative: Speculative::Auto,
            expect_proposer,
        };
        let depths = [
            1,
            3,
            MLX_ROW.recommended_depths.prompt_lookup,
            SPECULATIVE_MAX_DEPTH,
        ];
        for (label, provider, cases) in [
            (
                "causal",
                causal_provider(),
                depths
                    .map(lookup)
                    .into_iter()
                    .chain([auto(ProposerKind::PromptLookup)])
                    .collect::<Vec<_>>(),
            ),
            (
                "qwen35 mtp",
                qwen35_mtp_provider(),
                depths
                    .map(lookup)
                    .into_iter()
                    .chain([mtp(1), mtp(3), auto(ProposerKind::Mtp)])
                    .collect(),
            ),
        ] {
            let rows =
                core_llm_testkit::check_speculative_greedy_parity(&provider, &prompts, &cases, 20)
                    .unwrap_or_else(|failures| panic!("{label}: {failures}"));
            assert_eq!(rows.len(), prompts.len() * cases.len(), "{label}");
            assert!(
                rows.iter().all(|r| r.report.fallbacks.is_empty()),
                "{label}: nothing fell back"
            );
            let drafted = |kind| {
                rows.iter()
                    .filter(|r| r.report.proposer == kind)
                    .any(|r| r.report.proposed_tokens > 0)
            };
            assert!(
                drafted(ProposerKind::PromptLookup),
                "{label}: lookup drafted"
            );
            assert!(
                rows.iter().any(|r| r.report.accepted_tokens > 0),
                "{label}: some draft was accepted"
            );
            if label == "qwen35 mtp" {
                assert!(drafted(ProposerKind::Mtp), "{label}: the head drafted");
            }
        }
    }

    /// Epic AT1 at a production SDPA geometry: the shared fixture's hybrid target — 6 query heads
    /// over 1 KV head at head dim 64, a GQA group of 6 — beside its hybrid draft advertises every
    /// proposer at its geometry's depth (the vector kernel takes `32 / 6 = 5` query rows → 4
    /// drafts, below the 8-row ceiling), and every proposer at depths 1, 3 and that max emits
    /// `off`'s greedy stream. Every head-dim-64 attention call ran a fused MLX kernel, never the
    /// fallback: the prefill on the full kernel and each verify on the vector kernel, a max-depth
    /// verify (5 rows) in one call — the kernels `vector_verify_rows` / `speculative_max_depth`
    /// are justified by, which the narrow-head-dim fixtures never reach.
    #[test]
    fn every_proposer_is_greedy_exact_on_the_vector_kernel_geometry() {
        use crate::primitives::attention::kernel_tally::{record, SdpaKernel};
        let root = tempfile::tempdir().unwrap();
        let fixture = core_llm_testkit::write_draft_model_fixture(root.path()).unwrap();
        let provider = LlamaProvider::load(
            &LoadSpec::dense(fixture.hybrid_target.display().to_string())
                .with_draft(fixture.hybrid_draft.display().to_string()),
        )
        .unwrap();
        let Decoder::Qwen35(model) = &provider.model else {
            panic!("the hybrid target loads as the Qwen35 decoder");
        };
        let geometry = qwen35_attention_geometry(model.config());
        assert_eq!(geometry, (6, 1, 64, 64));
        let caps = &provider.descriptor().capabilities;
        let cases = core_llm_testkit::advertised_parity_cases(caps);
        let prompts = core_llm_testkit::draft_model_prompts();
        let (rows, calls) = record(|| {
            core_llm_testkit::check_speculative_greedy_parity(&provider, &prompts, &cases, 24)
        });
        let rows = rows.unwrap_or_else(|failures| panic!("{failures}"));
        assert!(rows.iter().all(|r| r.report.fallbacks.is_empty()));
        for kind in [
            ProposerKind::Mtp,
            ProposerKind::PromptLookup,
            ProposerKind::DraftModel,
        ] {
            assert!(
                rows.iter()
                    .any(|r| r.report.proposer == kind && r.report.proposed_tokens > 0),
                "{kind:?} drafted"
            );
        }

        // The kernels every head-dim-64 call — the target's and its MTP head's — reached.
        let wide: Vec<_> = calls.iter().filter(|c| c.head_dim == 64).collect();
        let fallback: Vec<_> = wide
            .iter()
            .filter(|c| c.kernel == SdpaKernel::Fallback)
            .collect();
        assert!(fallback.is_empty(), "fell back: {fallback:?}");
        assert!(
            wide.iter().any(|c| c.kernel == SdpaKernel::Full),
            "a prefill on the full kernel"
        );
        let deepest = rows
            .iter()
            .filter_map(|r| r.report.draft_tokens)
            .max()
            .unwrap();
        assert!(
            wide.iter()
                .any(|c| c.kernel == SdpaKernel::Vector && c.q_len == 1 + deepest as i32),
            "a max-depth ({deepest}) verify in one vector-kernel call"
        );

        // And that max is the geometry's: every proposer advertises it, the table covers it.
        let max = speculative_max_depth([geometry]);
        assert_eq!(max, 4, "the vector kernel's 5 rows, not the 8-row ceiling");
        assert_eq!(deepest, max);
        for proposer in SpeculativeProposer::ALL {
            assert_eq!(
                caps.proposer(proposer).unwrap().max_depth,
                max,
                "{proposer:?}"
            );
        }
    }

    // ---- The cross-turn prefix cache (sc-24437). ----

    fn chat(messages: Vec<Message>, speculative: core_llm::Speculative) -> TextLlmRequest {
        TextLlmRequest {
            messages,
            sampling: Sampling::greedy(),
            max_new_tokens: 8,
            seed: Some(3),
            speculative: Some(speculative),
            ..Default::default()
        }
    }

    fn rendered_ids(provider: &LlamaProvider, req: &TextLlmRequest, generation: bool) -> Vec<i32> {
        let text = provider
            .template
            .render_with(
                &req.messages,
                &RenderOptions {
                    add_generation_prompt: generation,
                    enable_thinking: req.enable_thinking_kwarg(),
                    reasoning_effort: req.reasoning_effort,
                    preserve_thinking: req.preserve_thinking,
                    tools: &req.tools,
                },
            )
            .unwrap();
        provider
            .tokenizer
            .encode(&text, false)
            .unwrap()
            .into_iter()
            .map(|id| id as i32)
            .collect()
    }

    fn shared(a: &[i32], b: &[i32]) -> usize {
        a.iter().zip(b).take_while(|(x, y)| x == y).count()
    }

    /// AC1 on the softmax family through the provider: turn 2 re-renders turn 1 and its reply, so
    /// it restores the shared run of turn 1's cached `prompt + reply` (N), and decodes exactly
    /// what a cold run (the non-engine plain loop, and a fresh provider) decodes.
    #[test]
    fn a_causal_second_turn_reports_its_prefix_hit_and_matches_a_cold_run() {
        use core_llm::Speculative;
        let provider = causal_provider();
        let user = Message::user("t3 t9 t4 t11 t3 t9");
        let turn1 = chat(vec![user.clone()], Speculative::Off);
        let (out1, ids1) = run(&provider, &turn1);
        assert_eq!(out1.decode.as_ref().unwrap().prefix_hit_tokens, 0, "cold");
        assert_eq!(out1.decode.as_ref().unwrap().prefix_cache.path, "miss");
        assert_eq!(provider.prefix_cache_stats().inserted, 1);

        let turn2 = chat(
            vec![
                user,
                Message::assistant(out1.text.clone()),
                Message::user("t5 t7"),
            ],
            Speculative::Off,
        );
        let p1 = rendered_ids(&provider, &turn1, true);
        let p2 = rendered_ids(&provider, &turn2, true);
        // Turn 1's cache holds its prompt and every reply token but the last (never fed).
        let mut held = p1.clone();
        held.extend(ids1.iter().map(|&t| t as i32));
        held.truncate(p1.len() + ids1.len() - 1);
        let n = shared(&held, &p2).min(p2.len() - 1);
        assert!(n >= p1.len() - 1, "turn 2 extends turn 1's prompt: {n}");

        let (out2, ids2) = run(&provider, &turn2);
        let report = out2.decode.unwrap();
        assert_eq!(report.prefix_hit_tokens, n as u64);
        assert_eq!(
            (
                report.prefix_cache.path.as_str(),
                report.prefix_cache.reason.as_deref()
            ),
            ("hit", None)
        );
        assert!(report.fallbacks.is_empty(), "{:?}", report.fallbacks);
        assert_eq!(ids2, plain_loop(&provider, &turn2));
        assert_eq!(ids2, run(&causal_provider(), &turn2).1);
        assert!(provider.prefix_cache_resident_bytes() <= provider.prefix_cache_budget());
    }

    /// AC2 through the provider: the hybrid snapshots at the end of turn 1's rendered
    /// conversation; turn 2 (and turn 3, re-using the same entry after turn 2 decoded past it)
    /// restores the recurrent state there and matches a cold run — plain and with the MTP head
    /// resuming its warm-up from the stored boundary.
    #[test]
    fn a_qwen35_second_turn_restores_the_boundary_state_and_matches_a_cold_run() {
        use core_llm::Speculative;
        for speculative in [
            Speculative::Off,
            Speculative::proposer(SpeculativeProposer::Mtp, 3),
        ] {
            let provider = qwen35_mtp_provider();
            let user = Message::user("t3 t9 t4 t11 t3 t9 t4 t11");
            let turn1 = chat(vec![user.clone()], speculative);
            let (out1, _) = run(&provider, &turn1);
            assert_eq!(out1.decode.as_ref().unwrap().prefix_hit_tokens, 0);
            let boundary = rendered_ids(&provider, &turn1, false).len();
            assert_eq!(provider.prefix.borrow().keys().len(), 1);
            assert_eq!(provider.prefix.borrow().keys()[0].len(), boundary);
            // The admitted estimate bounds what the snapshot holds (E7).
            let estimate = provider
                .prefix_snapshot_bytes(Some(boundary), speculative != Speculative::Off)
                .unwrap();
            let held = provider.prefix_cache_resident_bytes();
            assert!(held > 0 && held <= estimate, "{held} > {estimate}");

            for next in ["t5 t7", "t8 t2 t6"] {
                let turn = chat(
                    vec![
                        user.clone(),
                        Message::assistant(out1.text.clone()),
                        Message::user(next),
                    ],
                    speculative,
                );
                let (out, ids) = run(&provider, &turn);
                let report = out.decode.unwrap();
                assert_eq!(
                    report.prefix_hit_tokens, boundary as u64,
                    "{speculative:?}: restored at the boundary"
                );
                assert_eq!(report.prefix_cache.path, "hit");
                assert_eq!(report.proposer, speculative_kind(speculative));
                assert_eq!(ids, plain_loop(&provider, &turn), "{speculative:?}");
                assert_eq!(ids, run(&qwen35_mtp_provider(), &turn).1, "{speculative:?}");
            }
        }
    }

    fn speculative_kind(speculative: core_llm::Speculative) -> ProposerKind {
        match speculative {
            core_llm::Speculative::Proposer { proposer, .. } => proposer.into(),
            _ => ProposerKind::None,
        }
    }

    /// A tiny llama snapshot on disk for the load-admission tests.
    fn tiny_snapshot() -> crate::test_fixture::Fixture {
        let dir = crate::test_fixture::Fixture::new("mlx-llm-prefix-", None);
        std::fs::write(
            dir.join("config.json"),
            r#"{"hidden_size": 8, "intermediate_size": 16, "num_hidden_layers": 1,
                "num_attention_heads": 2, "num_key_value_heads": 1, "vocab_size": 16,
                "rms_norm_eps": 1e-5, "rope_theta": 10000.0, "tie_word_embeddings": false,
                "eos_token_id": 999}"#,
        )
        .unwrap();
        let vocab: Vec<String> = (0..16).map(|i| format!("\"t{i}\": {i}")).collect();
        std::fs::write(
            dir.join("tokenizer.json"),
            format!(
                r#"{{"version": "1.0", "added_tokens": [], "normalizer": null,
                    "pre_tokenizer": {{ "type": "Whitespace" }}, "post_processor": null,
                    "decoder": null,
                    "model": {{ "type": "WordLevel", "vocab": {{ {} }}, "unk_token": "t0" }} }}"#,
                vocab.join(", ")
            ),
        )
        .unwrap();
        let w = |shape: &[i32]| Array::ones::<f32>(shape).unwrap();
        let mut arrays = vec![
            ("model.embed_tokens.weight".to_string(), w(&[16, 8])),
            ("model.norm.weight".to_string(), w(&[8])),
            ("lm_head.weight".to_string(), w(&[16, 8])),
        ];
        for (name, shape) in [
            ("input_layernorm.weight", vec![8]),
            ("post_attention_layernorm.weight", vec![8]),
            ("self_attn.q_proj.weight", vec![8, 8]),
            ("self_attn.k_proj.weight", vec![4, 8]),
            ("self_attn.v_proj.weight", vec![4, 8]),
            ("self_attn.o_proj.weight", vec![8, 8]),
            ("mlp.gate_proj.weight", vec![16, 8]),
            ("mlp.up_proj.weight", vec![16, 8]),
            ("mlp.down_proj.weight", vec![8, 16]),
        ] {
            arrays.push((format!("model.layers.0.{name}"), w(&shape)));
        }
        let refs: Vec<(&str, &Array)> = arrays.iter().map(|(k, a)| (k.as_str(), a)).collect();
        Array::save_safetensors(refs, None, dir.join("model.safetensors")).unwrap();
        dir
    }

    /// Sets the operational memory override for one scope (the crate's tests run on one thread,
    /// `.cargo/config.toml`, so no other test observes it).
    struct MemoryOverride;

    impl MemoryOverride {
        fn set(bytes: u64) -> Self {
            std::env::set_var(core_llm::AVAILABLE_MEMORY_OVERRIDE, bytes.to_string());
            Self
        }
    }

    impl Drop for MemoryOverride {
        fn drop(&mut self) {
            std::env::remove_var(core_llm::AVAILABLE_MEMORY_OVERRIDE);
        }
    }

    /// E7 at load: the prefix cache's budget is what the load's admission leaves — never more —
    /// so it cannot push a load past its admission budget, and a load that fits without it still
    /// loads (with no cache) at the exact boundary.
    #[test]
    fn the_load_admits_the_prefix_cache_budget_up_to_its_headroom() {
        let dir = tiny_snapshot();
        let mut spec = LoadSpec::dense(dir.display().to_string());
        let required = crate::load_memory::required_bytes(&spec).unwrap();
        let headroom = 4096;
        spec.prefix_cache_bytes = Some(u64::MAX);
        {
            let _budget = MemoryOverride::set(required + headroom);
            let provider = LlamaProvider::load(&spec).unwrap();
            assert_eq!(provider.prefix_cache_budget(), headroom);
            assert_eq!(
                provider.load_report().unwrap().prefix_cache_bytes,
                Some(headroom),
                "the report names the settled budget"
            );
            spec.prefix_cache_bytes = Some(100);
            assert_eq!(
                LlamaProvider::load(&spec).unwrap().prefix_cache_budget(),
                100
            );
            spec.prefix_cache_bytes = None;
            assert_eq!(
                LlamaProvider::load(&spec).unwrap().prefix_cache_budget(),
                headroom.min(core_llm::defaults::MLX.prefix_cache_bytes)
            );
        }
        spec.prefix_cache_bytes = Some(u64::MAX);
        {
            // At the boundary the budget settles down to nothing rather than refusing.
            let _budget = MemoryOverride::set(required);
            let provider = LlamaProvider::load(&spec).unwrap();
            assert_eq!(provider.prefix_cache_budget(), 0);
            assert_eq!(provider.load_report().unwrap().prefix_cache_bytes, Some(0));
        }
        {
            let _budget = MemoryOverride::set(required - 1);
            assert!(LlamaProvider::load(&spec).is_err(), "past admission");
        }
    }

    /// The cache's part before any lookup: off with a zero budget, bypassed (named) for either
    /// multimodal route, else a miss — the only request that reads or feeds it.
    #[test]
    fn the_prefix_route_is_off_bypassed_or_a_miss() {
        assert_eq!(prefix_path_for(false, false, false), ("off", None));
        assert_eq!(prefix_path_for(false, true, true), ("off", None));
        for (multimodal, gemma4_mm) in [(true, false), (false, true), (true, true)] {
            assert_eq!(
                prefix_path_for(true, multimodal, gemma4_mm),
                ("bypassed", Some(PREFIX_MULTIMODAL_BYPASS)),
                "multimodal {multimodal}, gemma4 {gemma4_mm}"
            );
        }
        assert_eq!(prefix_path_for(true, false, false), ("miss", None));
    }

    /// A load that settled a zero budget never reads or feeds the cache: no lookup, no store,
    /// and every turn reports `off`.
    #[test]
    fn an_off_cache_is_never_read_or_fed() {
        use core_llm::Speculative;
        for provider in [causal_provider(), qwen35_mtp_provider()] {
            let provider = provider.with_prefix_budget(0);
            let user = Message::user("t3 t9 t4 t11");
            let (out1, _) = run(&provider, &chat(vec![user.clone()], Speculative::Off));
            let turn2 = chat(
                vec![user, Message::assistant(out1.text), Message::user("t5 t7")],
                Speculative::Off,
            );
            let report = run(&provider, &turn2).0.decode.unwrap();
            assert_eq!(report.prefix_cache.path, "off");
            assert_eq!(report.prefix_hit_tokens, 0);
            assert_eq!(provider.prefix_cache_stats(), PrefixStats::default());
        }
    }

    /// A tied-embedding decoder snapshot at `vocab × h` (h = 256: every projection a whole number
    /// of Q4/Q8 groups), Gemma 2's soft caps and sandwich norms when `gemma`, with a word-level
    /// tokenizer over the whole vocabulary.
    fn promotion_snapshot(dir: &Path, model_type: &str, arch: &str, vocab: i32, gemma: bool) {
        let (h, softcap) = (256, gemma);
        let (inter, layers, heads, kv, hd) = (512, 2, 2, 1, 128);
        let mut cfg = json!({
            "architectures": [arch], "model_type": model_type, "hidden_size": h,
            "intermediate_size": inter, "num_hidden_layers": layers, "num_attention_heads": heads,
            "num_key_value_heads": kv, "head_dim": hd, "vocab_size": vocab, "rms_norm_eps": 1e-6,
            "rope_theta": 10000.0, "tie_word_embeddings": true, "max_position_embeddings": 8192,
        });
        if softcap {
            cfg["attn_logit_softcapping"] = json!(50.0);
            cfg["final_logit_softcapping"] = json!(30.0);
            cfg["query_pre_attn_scalar"] = json!(hd);
            cfg["sliding_window"] = json!(4096);
        }
        std::fs::write(dir.join("config.json"), cfg.to_string()).unwrap();
        let entries: Vec<String> = (0..vocab).map(|i| format!("\"t{i}\": {i}")).collect();
        std::fs::write(
            dir.join("tokenizer.json"),
            format!(
                r#"{{"version": "1.0", "added_tokens": [], "normalizer": null,
                "pre_tokenizer": {{ "type": "Whitespace" }}, "post_processor": null,
                "decoder": null,
                "model": {{ "type": "WordLevel", "vocab": {{ {} }}, "unk_token": "t0" }} }}"#,
                entries.join(", ")
            ),
        )
        .unwrap();
        let mut seed = 1u64;
        let mut r = |shape: &[i32]| {
            let n: i32 = shape.iter().product();
            let v: Vec<f32> = (0..n)
                .map(|_| {
                    seed = seed.wrapping_mul(6364136223846793005).wrapping_add(1);
                    ((seed >> 40) as f32 / (1u64 << 24) as f32 - 0.5) * 0.1
                })
                .collect();
            Array::from_slice(&v, shape)
                .as_dtype(mlx_rs::Dtype::Bfloat16)
                .unwrap()
        };
        let mut t: Vec<(String, Array)> = vec![
            ("model.embed_tokens.weight".into(), r(&[vocab, h])),
            ("model.norm.weight".into(), r(&[h])),
        ];
        for i in 0..layers {
            let p = |s: &str| format!("model.layers.{i}.{s}");
            for n in ["input_layernorm", "post_attention_layernorm"] {
                t.push((p(&format!("{n}.weight")), r(&[h])));
            }
            if softcap {
                for n in ["pre_feedforward_layernorm", "post_feedforward_layernorm"] {
                    t.push((p(&format!("{n}.weight")), r(&[h])));
                }
            }
            t.push((p("self_attn.q_proj.weight"), r(&[heads * hd, h])));
            t.push((p("self_attn.k_proj.weight"), r(&[kv * hd, h])));
            t.push((p("self_attn.v_proj.weight"), r(&[kv * hd, h])));
            t.push((p("self_attn.o_proj.weight"), r(&[h, heads * hd])));
            t.push((p("mlp.gate_proj.weight"), r(&[inter, h])));
            t.push((p("mlp.up_proj.weight"), r(&[inter, h])));
            t.push((p("mlp.down_proj.weight"), r(&[h, inter])));
        }
        Array::save_safetensors(
            t.iter().map(|(k, v)| (k.as_str(), v)),
            None,
            dir.join("model.safetensors"),
        )
        .unwrap();
    }

    /// sc-24446 (d): a one-token request's MLX working set is priced, under either activation
    /// dtype ([`crate::primitives::activation`]).
    ///
    /// * **LLM decode** (every provider load): a Gemma GeGLU returns BF16, so the forward stays
    ///   BF16 and promotes nothing — `promoted_request_bytes` is 0 — and the measured
    ///   second-request working set on a tied 65,536-token fixture, dense and at load-time Q4,
    ///   stays within the request estimate, as a SwiGLU decoder's does. Measured both as the exact
    ///   settled `phys_footprint` peak growth (driver wake, MLX cache and host heap included) and
    ///   as MLX's active peak growth; the estimate covers the larger.
    /// * **A role pinned to `f32`** (the LTX-2.5 text encoder): the GeGLU leaves the stream `f32`,
    ///   every dense matmul materializes an `f32` copy of its BF16 weight (the LM head's alone
    ///   vocabulary × hidden × 4 bytes), and the promotion term is what covers the measured
    ///   one-token forward — it exceeds the estimate without it.
    ///
    /// The final hidden state is `f32` exactly when the model reports promoting activations.
    ///
    /// MUTATION: return `Some(0)` from `Decoder::promoted_request_bytes` and the pinned cases go
    /// RED; make `CausalLm::activations_promote` ignore the role and the LLM-decode Gemma cases do.
    #[test]
    fn a_one_token_request_working_set_is_priced_under_either_activation_dtype() {
        use crate::primitives::activation::ActivationRole;
        for (model_type, arch, gemma) in [
            ("llama", "LlamaForCausalLM", false),
            ("gemma2", "Gemma2ForCausalLM", true),
        ] {
            for quantize in [None, Some(Quantize::Q4)] {
                let dir = tempfile::tempdir().unwrap();
                promotion_snapshot(dir.path(), model_type, arch, 65_536, gemma);
                let mut spec = LoadSpec::dense(dir.path().display().to_string());
                spec.quantize = quantize;
                spec.prefix_cache_bytes = Some(0);
                let label = format!("{model_type} {quantize:?}");

                // LLM decode, through the provider.
                let provider = LlamaProvider::load(&spec).unwrap();
                let req = TextLlmRequest {
                    messages: vec![Message::user("t3 t9 t4 t11")],
                    sampling: Sampling::greedy(),
                    max_new_tokens: 1,
                    seed: Some(0),
                    speculative: Some(core_llm::Speculative::Off),
                    ..Default::default()
                };
                let prompt = rendered_ids(&provider, &req, true);
                let estimate = provider
                    .speculative_request_bytes(SpeculativeRoute::Plain, prompt.len(), 1, 0)
                    .unwrap();
                let Decoder::Causal(m) = &provider.model else {
                    unreachable!("a causal fixture")
                };
                assert_hidden_dtype_matches_promotion(m, &label);
                assert!(!m.activations_promote(), "{label}: LLM decode promotes");
                assert_eq!(provider.model.promoted_request_bytes(), Some(0), "{label}");
                provider.generate(&req, &mut |_| {}).unwrap();
                let working_set = second_run_peak(|| {
                    provider.generate(&req, &mut |_| {}).unwrap();
                });
                assert!(
                    working_set.max() <= estimate,
                    "{label}: working set {working_set:?} exceeds the estimate {estimate}"
                );

                // A role pinned to f32, on the decoder directly.
                if !gemma {
                    continue;
                }
                let cfg_value = read_config_value(dir.path()).unwrap();
                let mut cfg = ModelConfig::from_json(&cfg_value).unwrap();
                cfg.activation_role = ActivationRole::LtxTextEncoder;
                let w = Weights::from_dir(dir.path()).unwrap();
                let quant = quantize.map(|q| quant_spec(q).unwrap());
                let pinned =
                    Decoder::Causal(CausalLm::from_weights_with(&w, "", cfg, quant).unwrap());
                let Decoder::Causal(m) = &pinned else {
                    unreachable!()
                };
                assert_hidden_dtype_matches_promotion(m, &label);
                assert!(m.activations_promote(), "{label}: the pinned role promotes");
                let promoted = pinned.promoted_request_bytes().unwrap();
                assert!(promoted > 0, "{label}");
                let ids = input_ids(&[3, 9, 4, 11]);
                let decoder_estimate = estimate_mlx_request_bytes(
                    4,
                    1,
                    pinned.memory_geometry(),
                    0,
                    0,
                    pinned.workspace_contract(),
                )
                .unwrap();
                let forward = || {
                    m.decode_logits(&ids, &mut m.new_cache(), 0)
                        .unwrap()
                        .eval()
                        .unwrap();
                };
                forward();
                let working_set = second_run_peak(forward);
                assert!(
                    working_set.max() <= decoder_estimate + promoted,
                    "{label} pinned: working set {working_set:?} exceeds {}",
                    decoder_estimate + promoted
                );
                // The promoted copies are GPU transients: MLX's active peak sees them even where
                // the settled footprint reuses the driver's pooled pages. Without the term, the
                // decoder's own tensors (the estimate less the driver's wake) do not cover them.
                assert!(
                    working_set.active > decoder_estimate - MLX_REQUEST_WAKE_BYTES,
                    "{label} pinned: the promotion term is not load-bearing ({working_set:?} vs \
                     {decoder_estimate})"
                );
            }
        }
    }

    /// The final hidden state is `f32` exactly when `m` reports promoting activations.
    fn assert_hidden_dtype_matches_promotion(m: &CausalLm, label: &str) {
        let hidden = m
            .hidden_states(&input_ids(&[3, 9, 4]), &mut m.new_cache(), 0)
            .unwrap();
        let dtype = hidden.last().unwrap().dtype();
        assert_eq!(
            dtype == mlx_rs::Dtype::Float32,
            m.activations_promote(),
            "{label}: activation dtype {dtype:?}"
        );
    }

    /// A request's working set two ways (the model already materialized by a first run): the
    /// exact `phys_footprint` peak growth after the driver has settled — what macOS charges, MLX's
    /// buffer cache, the host heap and the driver's wake included — and MLX's own peak active
    /// growth, which still sees a GPU transient whose pages the driver recycled from its pool.
    fn second_run_peak(run: impl Fn()) -> WorkingSet {
        let footprint = crate::test_fixture::footprint::peak_growth(&run);
        mlx_rs::memory::clear_cache();
        let before = mlx_rs::memory::get_active_memory();
        mlx_rs::memory::reset_peak_memory();
        run();
        let active = (mlx_rs::memory::get_peak_memory() - before) as u64;
        WorkingSet { footprint, active }
    }

    #[derive(Debug)]
    struct WorkingSet {
        footprint: u64,
        active: u64,
    }

    impl WorkingSet {
        fn max(&self) -> u64 {
            self.footprint.max(self.active)
        }
    }

    /// sc-24446: the request estimate covers every one-token request working set the 2026-10-01
    /// exact-peak re-probes measured on real weights — the kernel's `phys_footprint` maximum over
    /// a second identical one-token "Hi" request after the driver settled (MLX's cache, the host
    /// heap and the driver's wake included), on the current build: BF16 GeGLU for every LLM
    /// decode path, and the `f32` GeGLU (a role pinned to it, with the promotion term) for the
    /// Gemma rows marked so. Recorded evidence, not a machine golden: the relation is pinned.
    ///
    /// MUTATION: drop the driver's wake from the request estimate and the BF16 rows go RED.
    #[test]
    fn the_request_estimate_covers_the_exact_footprint_working_sets() {
        let geometry = |q, kv, hd, layers, hidden, inter, vocab, recurrent| LlmMemoryGeometry {
            query_heads: q,
            kv_heads: kv,
            head_dim: hd,
            layers,
            element_bytes: 4,
            hidden_size: hidden,
            intermediate_size: inter,
            vocab_size: vocab,
            recurrent_bytes: recurrent,
        };
        let llama = geometry(32, 8, 64, 16, 2048, 8192, 128_256, 0);
        let qwen3_small = geometry(16, 8, 128, 28, 2048, 6144, 151_936, 0);
        let qwen3_8b = geometry(32, 8, 128, 36, 4096, 12_288, 151_936, 0);
        let gemma2 = geometry(8, 4, 256, 26, 2304, 9216, 256_000, 0);
        let gemma4 = geometry(16, 8, 256, 48, 3840, 15_360, 262_144, 0);
        let qwen38 = frozen_dense_qwen35_config();
        let qwen38_geometry = geometry(
            24,
            4,
            256,
            64,
            5120,
            17_408,
            248_320,
            64 * 48 * 128 * (128 + 4) * 4,
        );
        let qwen35 = MlxWorkspaceContract::Qwen35 {
            config: &qwen38,
            prism: false,
        };
        let qwen36_inter = Qwen35Config::from_json(
            &serde_json::from_str::<serde_json::Value>(include_str!(
                "../../testdata/load_admission/qwen3.6-35b-a3b.json"
            ))
            .unwrap()["config"],
        )
        .unwrap()
        .intermediate_size as u64;
        let qwen36 = geometry(
            16,
            2,
            256,
            40,
            2048,
            qwen36_inter,
            248_320,
            40 * 32 * 128 * (128 + 4) * 4,
        );
        let (g2_head, g2_mlp) = (256_000 * 2304, 9216 * 2304);
        let (g4_head, g4_mlp) = (262_144 * 3840, 15_360 * 3840);
        let promoted = |head, largest| promoted_weight_bytes(head, largest).unwrap();
        use MlxWorkspaceContract::{Chunked, Eager};
        // (row, geometry, contract, rendered prompt tokens, promoted bytes, measured footprint)
        let cases = [
            ("Llama 3.2 1B BF16", llama, Chunked, 36, 0, 171_000_000u64),
            ("Llama 3.2 1B Q4/Q8", llama, Chunked, 36, 0, 226_000_000),
            ("Qwen3-1.7B", qwen3_small, Chunked, 13, 0, 237_000_000),
            ("Qwen3-8B", qwen3_8b, Chunked, 13, 0, 269_000_000),
            (
                "Gemma 2 2B-it (BF16 GeGLU)",
                gemma2,
                Eager,
                10,
                0,
                195_000_000,
            ),
            (
                "Gemma 4 enhancer (BF16 GeGLU)",
                gemma4,
                Eager,
                14,
                0,
                295_000_000,
            ),
            ("Qwen3.8-27B", qwen38_geometry, qwen35, 13, 0, 382_000_000),
            ("Qwen3.6-35B-A3B", qwen36, Eager, 13, 0, 251_000_000),
            (
                "Gemma 2 2B-it BF16 (f32 GeGLU)",
                gemma2,
                Eager,
                10,
                promoted(g2_head, g2_mlp),
                3_811_000_000,
            ),
            (
                "Gemma 2 2B-it Q4 (f32 GeGLU)",
                gemma2,
                Eager,
                10,
                promoted(g2_head, g2_mlp / 64 * 2),
                2_668_000_000,
            ),
            (
                "Gemma 4 enhancer BF16 (f32 GeGLU)",
                gemma4,
                Eager,
                14,
                promoted(g4_head, g4_mlp),
                6_411_000_000,
            ),
            (
                "Gemma 4 enhancer Q4 (f32 GeGLU)",
                gemma4,
                Eager,
                14,
                promoted(g4_head, g4_mlp / 64 * 2),
                4_659_000_000,
            ),
        ];
        for (row, geometry, contract, prompt, promoted, measured) in cases {
            let estimate =
                estimate_mlx_request_bytes(prompt, 1, geometry, 0, 0, contract).unwrap() + promoted;
            assert!(
                estimate >= measured,
                "{row}: estimate {estimate} < measured footprint {measured}"
            );
        }
    }

    /// sc-24446: the K/V caches hold whole 256-position blocks — over the committed positions and
    /// over any verify width written past them — and a prompt-lookup / draft overshoot that
    /// crosses into a new block charges the rest of that block.
    ///
    /// MUTATION: multiply the runtime K/V padding by zero, or drop `kv_overshoot_padding`, and
    /// this goes RED.
    #[test]
    fn kv_block_padding_covers_committed_and_verify_positions() {
        let g = LlmMemoryGeometry {
            query_heads: 8,
            kv_heads: 4,
            head_dim: 64,
            layers: 6,
            element_bytes: 4,
            hidden_size: 512,
            intermediate_size: 1024,
            vocab_size: 4096,
            recurrent_bytes: 0,
        };
        let kv_position = 6 * 4 * 64 * 4 * 2;
        let runtime =
            |committed, width| mlx_runtime_request_bytes(8, committed, width, 8, g).unwrap();
        // 10 committed positions allocate a whole block: 246 more than 256 committed do.
        assert_eq!(runtime(10, 0) - runtime(256, 0), 246 * kv_position);
        // Three verify positions past 255 committed: the three and the new block's rest.
        assert_eq!(
            runtime(255, 3) - runtime(255, 0),
            (3 + 254 - 1) * kv_position
        );
        assert_eq!(kv_overshoot_padding(255, 3), Some(512 - (256 + 3)));
        assert_eq!(kv_overshoot_padding(100, 3), Some(0));
    }

    /// sc-24446: a prompt-lookup verify step whose overshoot crosses into a new K/V block is
    /// charged that block's rest, beyond the overshoot positions themselves.
    ///
    /// MUTATION: drop `kv_overshoot_padding` from `speculative_request_bytes` and this goes RED.
    #[test]
    fn a_lookup_overshoot_into_a_new_block_charges_the_block() {
        let provider = causal_provider();
        let g = provider.model.memory_geometry();
        let kv_position = g.layers * g.kv_heads * g.head_dim * g.element_bytes * 2;
        let rows = (g.hidden_size + g.vocab_size) * g.element_bytes;
        let price = |route| {
            provider
                .speculative_request_bytes(route, 200, 50, 0)
                .unwrap()
        };
        let off = price(SpeculativeRoute::Plain);
        let lookup = price(SpeculativeRoute::PromptLookup { width: 8 });
        // 250 committed positions + 8 overshoot = 258: a second block, 248 positions beyond.
        assert_eq!(lookup - off, 8 * (kv_position + rows) + 248 * kv_position);
    }

    /// sc-24446: a decoder whose `param_groups` leaves a source it read unconsumed is refused at
    /// load — the source would stay resident outside the admitted bound.
    ///
    /// MUTATION: drop the `leftover > 0` refusal from `materialize_decoder` and this goes RED.
    #[test]
    fn an_unconsumed_source_is_a_load_error() {
        let dir = crate::test_fixture::Fixture::new("mlx-llm-leftover-", None);
        let path = dir.join("model.safetensors");
        let a = Array::from_slice(&[1.0f32, 2.0], &[2]);
        Array::save_safetensors([("used", &a), ("orphan", &a)], None, &path).unwrap();
        let mut w = Weights::from_file(&path).unwrap();
        let used = w
            .require("used")
            .unwrap()
            .as_dtype(mlx_rs::Dtype::Bfloat16)
            .unwrap();
        w.require("orphan").unwrap();
        let err = materialize_decoder(&mut w, &[vec![used]]).unwrap_err();
        assert!(err.to_string().contains("load materialization"), "{err}");
    }

    /// sc-24446: the in-flight window prices each command buffer's crossing op — a long eager
    /// prefill's `prompt × prompt` score block per head, or a `prompt × inter` MLP activation —
    /// not just the 50 MB per-buffer cap.
    ///
    /// MUTATION: drop the crossing op from the window and this goes RED.
    #[test]
    fn the_in_flight_window_prices_each_buffers_crossing_op() {
        let g = LlmMemoryGeometry {
            query_heads: 8,
            kv_heads: 4,
            head_dim: 256,
            layers: 26,
            element_bytes: 4,
            hidden_size: 2304,
            intermediate_size: 9216,
            vocab_size: 256_000,
            recurrent_bytes: 0,
        };
        let prompt = 8192u64;
        let committed = prompt + 1;
        let fixed = (round_up(committed, 256).unwrap() - committed) * (26 * 4 * 256 * 4 * 2)
            + MLX_EVAL_BUFFER_WINDOW * MLX_MAX_OPS_PER_BUFFER * MLX_ALLOCATION_PAGE_BYTES
            + MLX_REQUEST_WAKE_BYTES;
        let in_flight =
            mlx_runtime_request_bytes(prompt as usize, committed, 0, prompt, g).unwrap() - fixed;
        let scores = prompt * prompt * 8 * 4;
        assert!(
            scores > prompt * 9216 * 4,
            "the eager score block is the crossing op"
        );
        assert_eq!(
            in_flight,
            MLX_EVAL_BUFFER_WINDOW * (MLX_MAX_BUFFER_BYTES + scores)
        );
        assert!(in_flight > 10 * MLX_EVAL_BUFFER_WINDOW * MLX_MAX_BUFFER_BYTES);
    }

    /// The request estimate covers every one-token request working set the sc-24446 guarded
    /// probes recorded on real weights (`native-memory-admission.md`: a second identical
    /// one-token "Hi" request on the materialized model), computed from each model's geometry
    /// and rendered prompt length. Recorded evidence, not a machine golden: the relation is
    /// pinned. Two limits, both stated rather than hidden:
    ///
    /// * the probes recorded MLX's **active** peak, not `phys_footprint` (cache retention and
    ///   host heap uncounted); the exact footprint working sets are pinned by
    ///   `the_request_estimate_covers_the_exact_footprint_working_sets`;
    /// * every Gemma row ran the `f32` GeGLU, so it is pinned with the promotion term — today
    ///   that is the path a role pinned to `f32` (the LTX-2.5 text encoder) runs.
    ///
    /// Before sc-24446 the estimate was 6–9 MB for the Llama / Qwen3 cases (measured 35–87 MB)
    /// and 14–31 MB for the Gemma cases (measured 2.4–5.4 GB).
    #[test]
    fn the_request_estimate_covers_the_recorded_one_token_working_sets() {
        let geometry = |q, kv, hd, layers, hidden, inter, vocab, recurrent| LlmMemoryGeometry {
            query_heads: q,
            kv_heads: kv,
            head_dim: hd,
            layers,
            element_bytes: 4,
            hidden_size: hidden,
            intermediate_size: inter,
            vocab_size: vocab,
            recurrent_bytes: recurrent,
        };
        let llama = geometry(32, 8, 64, 16, 2048, 8192, 128_256, 0);
        let qwen3_small = geometry(16, 8, 128, 28, 2048, 6144, 151_936, 0);
        let qwen3_8b = geometry(32, 8, 128, 36, 4096, 12_288, 151_936, 0);
        let gemma2 = geometry(8, 4, 256, 26, 2304, 9216, 256_000, 0);
        let gemma4 = geometry(16, 8, 256, 48, 3840, 15_360, 262_144, 0);
        let qwen38 = frozen_dense_qwen35_config();
        let qwen38_geometry = geometry(
            24,
            4,
            256,
            64,
            5120,
            17_408,
            248_320,
            64 * 48 * 128 * (128 + 4) * 4,
        );
        let qwen35 = MlxWorkspaceContract::Qwen35 {
            config: &qwen38,
            prism: false,
        };
        let (g2_head, g2_mlp) = (256_000 * 2304, 9216 * 2304);
        let (g4_head, g4_mlp) = (262_144 * 3840, 15_360 * 3840);
        let promoted = |head, largest| promoted_weight_bytes(head, largest).unwrap();
        // (config, geometry, contract, rendered prompt tokens, promoted bytes, measured bytes)
        let cases = [
            (
                "Llama 3.2 1B",
                llama,
                MlxWorkspaceContract::Chunked,
                36,
                0,
                41_000_000u64,
            ),
            (
                "Qwen3-1.7B",
                qwen3_small,
                MlxWorkspaceContract::Chunked,
                13,
                0,
                46_000_000,
            ),
            (
                "Qwen3-8B",
                qwen3_8b,
                MlxWorkspaceContract::Chunked,
                13,
                0,
                87_000_000,
            ),
            ("Qwen3.8-27B", qwen38_geometry, qwen35, 13, 0, 236_000_000),
            (
                "Gemma 2 2B-it BF16",
                gemma2,
                MlxWorkspaceContract::Eager,
                10,
                promoted(g2_head, g2_mlp),
                3_528_000_000,
            ),
            (
                "Gemma 2 2B-it Q4",
                gemma2,
                MlxWorkspaceContract::Eager,
                10,
                promoted(g2_head, g2_mlp / 64 * 2),
                2_426_000_000,
            ),
            (
                "Gemma 4 enhancer BF16",
                gemma4,
                MlxWorkspaceContract::Eager,
                14,
                promoted(g4_head, g4_mlp),
                5_416_000_000,
            ),
            (
                "Gemma 4 enhancer Q4",
                gemma4,
                MlxWorkspaceContract::Eager,
                14,
                promoted(g4_head, g4_mlp / 64 * 2),
                4_466_000_000,
            ),
        ];
        for (config, geometry, contract, prompt, promoted, measured) in cases {
            let estimate =
                estimate_mlx_request_bytes(prompt, 1, geometry, 0, 0, contract).unwrap() + promoted;
            assert!(
                estimate >= measured,
                "{config}: estimate {estimate} < measured {measured}"
            );
        }
        assert!(promoted_weight_bytes(u64::MAX, 1).is_none());
    }

    /// E7 per request: held entries are reclaimable. A request short by exactly what the cache
    /// holds is admitted after evicting it; one byte shorter still is refused and leaves the
    /// cache as it was.
    #[test]
    fn a_request_reclaims_the_prefix_cache_up_to_the_admission_boundary() {
        use core_llm::Speculative;
        let provider = causal_provider();
        run(
            &provider,
            &chat(vec![Message::user("t3 t9 t4")], Speculative::Off),
        );
        let held = provider.prefix_cache_resident_bytes();
        assert!(held > 0);
        let req = chat(vec![Message::user("t11 t12 t13 t14")], Speculative::Off);
        let prompt = rendered_ids(&provider, &req, true);
        let required = provider
            .speculative_request_bytes(SpeculativeRoute::Plain, prompt.len(), req.max_new_tokens, 0)
            .unwrap();
        assert!(
            required > held,
            "the test needs a shortfall the cache can cover: {required} vs {held}"
        );

        {
            let _budget = MemoryOverride::set(required - held - 1);
            let err = provider.generate(&req, &mut |_| {}).unwrap_err();
            assert!(
                matches!(err, CoreError::RequestResourceExhausted(_)),
                "{err}"
            );
            assert_eq!(
                provider.prefix_cache_resident_bytes(),
                held,
                "kept on refusal"
            );
        }
        {
            let _budget = MemoryOverride::set(required - held);
            let before = provider.prefix_cache_stats().evicted;
            provider.generate(&req, &mut |_| {}).unwrap();
            assert_eq!(
                provider.prefix_cache_stats().evicted - before,
                1,
                "reclaimed"
            );
            assert_eq!(
                provider.prefix.borrow().len(),
                1,
                "only this request's entry"
            );
        }
    }

    /// E7 for the snapshot a hybrid request leaves behind: admitted with the request when the
    /// memory holds both, and at one byte short the request still runs — without keeping it.
    #[test]
    fn a_qwen35_request_keeps_its_snapshot_only_when_admission_holds_it() {
        use core_llm::Speculative;
        let req = chat(vec![Message::user("t3 t9 t4 t11")], Speculative::Off);
        let probe = qwen35_mtp_provider();
        let prompt = rendered_ids(&probe, &req, true);
        let boundary = rendered_ids(&probe, &req, false).len();
        let required = probe
            .speculative_request_bytes(SpeculativeRoute::Plain, prompt.len(), req.max_new_tokens, 0)
            .unwrap();
        let snapshot = probe.prefix_snapshot_bytes(Some(boundary), false).unwrap();
        {
            let provider = qwen35_mtp_provider();
            let _budget = MemoryOverride::set(required + snapshot - 1);
            let out = provider.generate(&req, &mut |_| {}).unwrap();
            let report = out.decode.unwrap();
            assert_eq!(report.prefix_hit_tokens, 0);
            assert!(provider.prefix.borrow().is_empty(), "no room: nothing kept");
            let reason = report.prefix_cache.reason.unwrap();
            assert!(reason.starts_with("not kept: admission"), "{reason}");
        }
        {
            let provider = qwen35_mtp_provider();
            let _budget = MemoryOverride::set(required + snapshot);
            provider.generate(&req, &mut |_| {}).unwrap();
            assert_eq!(
                provider.prefix.borrow().keys(),
                vec![prompt[..boundary].to_vec()]
            );
        }
    }

    /// sc-24446 (E5): a provider loads with the MLX defaults-table speculative default, and a
    /// request that leaves the option unset runs whatever the provider's default is — here set to
    /// `auto` so the hook is observable — while an explicit off (new or legacy spelling) wins.
    #[test]
    fn an_unset_speculative_option_runs_the_providers_default() {
        let root = tempfile::tempdir().unwrap();
        let fixture = core_llm_testkit::write_draft_model_fixture(root.path()).unwrap();
        let mut provider =
            LlamaProvider::load(&LoadSpec::dense(fixture.target.to_string_lossy())).unwrap();
        assert_eq!(
            provider.speculative_default(),
            core_llm::defaults::MLX.speculative,
            "loaded with the defaults table's default"
        );
        let prompt = &core_llm_testkit::draft_model_prompts()[0];
        core_llm_testkit::check_speculative_default(
            &provider,
            core_llm::defaults::MLX.speculative,
            prompt,
            8,
        );
        provider.set_speculative_default(core_llm::Speculative::Auto);
        core_llm_testkit::check_speculative_default(
            &provider,
            core_llm::Speculative::Auto,
            prompt,
            8,
        );
    }
}
