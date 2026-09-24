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
    ModelSamplingDefaults, MtpCapabilities, MtpStats, Quantize, ReasoningEffort, RenderOptions,
    Result as CoreResult, Sampling, StopMatcher, StreamEvent as CoreEvent, TextLlm,
    TextLlmCapabilities, TextLlmDescriptor, TextLlmOutput, TextLlmRequest, ThinkingSegmenter,
    Tokenizer, ToolCallSegmenter, Usage, VideoRef,
};
use serde_json::Value;

use crate::config::{Architecture, ModelConfig};
use crate::decode::{
    cuda_graphs_enabled, generate_from_prefill_with_stop, generate_speculative_with,
    graph_workspace_admission_bytes, ConstraintMask, CountingDecode, Decode, DecodePath,
    DecodeRecord, FinishReason, GenerationConfig, GraphRunner, GraphTally, MtpProposer, NoProposer,
    Proposer, RequestSpan, RewindableConstraintMask, SpeculativePrompt, StepModel, StreamEvent,
};
use crate::device::select_device;
use crate::gguf::GgufCheckpoint;
use crate::image::Qwen35ImageProcessor;
use crate::models::gemma4_mm;
use crate::models::{
    CausalLm, Gemma4Layout, Gemma4Mm, Gemma4MmConfig, Qwen35Cache, Qwen35Config, Qwen35Model,
    Qwen35Mtp, Qwen35VisionConfig, Qwen35VisionModel, VlmDecode,
};
use crate::primitives::attention::EAGER_ATTN_QUERY_CHUNK_SIZE;
use crate::primitives::nn::input_ids;
use crate::primitives::projection::{
    ggml_device_bytes, nvfp4_device_bytes, ProjectionFormat, ProjectionKind, QuantSpec,
    WeightCensus,
};
use crate::primitives::sampler::SamplingParams;
use crate::primitives::{KvCache, StepKvCache, Weights};

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

/// The CUDA-graph tally of a request decoded on a reference path. The runner wraps the step-seam
/// engine only (sc-24134), so with the switch on such a request's record names why no step went
/// through it — `fallback=reference_path` — instead of a bare `graph: none`.
fn reference_path_graphs(tally: GraphTally, switch_on: bool) -> GraphTally {
    if switch_on && tally.label() == "none" {
        GraphTally {
            fallback_reason: Some(crate::decode::graph::REASON_REFERENCE_PATH),
            ..tally
        }
    } else {
        tally
    }
}

/// Which loop a request decodes on — what admission prices and what [`LlamaProvider::generate`]
/// runs (sc-24140: one decision, so the price and the path cannot disagree).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum DecodeRoute {
    /// The reference `Decode` loop over the growing cache: the parity oracle selected with
    /// [`LlamaProvider::set_decode_path`], and the Qwen3-VL generic-causal multimodal request
    /// (whose DeepStack / M-RoPE prefill has no step-seam form).
    Reference,
    /// The unified engine over the step seam, on the static KV cache, with `drafts` drafts per
    /// verify step: the MTP head's `K`, or `0` — the engine with no proposer, the token-at-a-time
    /// loop — for every request whose speculation is off.
    Engine {
        /// Draft tokens per verify step (`0`: no proposer).
        drafts: u32,
    },
}

impl DecodeRoute {
    /// The verify step's draft width (`0` on the reference loop and the no-proposer engine).
    fn drafts(self) -> u32 {
        match self {
            DecodeRoute::Reference => 0,
            DecodeRoute::Engine { drafts } => drafts,
        }
    }
}

/// The bytes admission prices for a request of `admitted_prompt` + `max_new_tokens` tokens: on
/// the reference geometry for the reference loop, and on the step seam's geometry whenever the
/// engine runs — its cache keeps a per-token checkpoint ring of `K + 2` recurrent states per
/// linear layer the reference cache does not (sc-24131; `2` with no proposer), and its verify
/// step overshoots the budget by `K` positions (E6, sc-24130).
///
/// The recurrent term is charged **once** on both paths: the geometry's `recurrent_bytes` is
/// already the cache's whole recurrent footprint (the ring included), and the engine rolls back
/// by selecting a ring slot — it never clones the cache — so the `x3` clone/replay multiplier
/// [`core_llm::estimate_chunked_request_bytes`] applies with MTP would charge `2 (K + 2)` states
/// the request never holds.
///
/// Whenever the engine runs through the CUDA-graph runner (`cuda_graphs`, sc-24134) — with a
/// proposer or without one — the graphs it may capture are priced too
/// ([`graph_workspace_admission_bytes`] over the `1 ..= K + 1` step token counts), so a request
/// that could not hold them fails closed at admission instead of at instantiation (sc-24140).
fn priced_request_bytes(
    model: &Decoder,
    route: DecodeRoute,
    admitted_prompt: usize,
    max_new_tokens: u32,
    vision_workspace: u64,
    cuda_graphs: bool,
) -> Option<u64> {
    let geometry = match route {
        DecodeRoute::Engine { drafts } => model.step_memory_geometry(drafts as usize),
        DecodeRoute::Reference => model.memory_geometry(),
    };
    let graph_workspace = match route {
        DecodeRoute::Engine { drafts } => engine_graph_workspace(
            model,
            drafts,
            u64::try_from(admitted_prompt)
                .ok()?
                .checked_add(u64::from(max_new_tokens))?,
            cuda_graphs,
        )?,
        DecodeRoute::Reference => 0,
    };
    core_llm::estimate_chunked_request_bytes_with_recurrent_copies(
        admitted_prompt,
        max_new_tokens,
        geometry,
        vision_workspace.checked_add(graph_workspace)?,
        route.drafts(),
        EAGER_ATTN_QUERY_CHUNK_SIZE,
        1,
    )
}

/// The CUDA-graph workspace admission charges a request on the engine with `drafts` drafts per
/// verify step whose prompt and budget reach `positions` positions: every graph the runner may
/// capture ([`graph_workspace_admission_bytes`] over the step geometry, the reach plus the verify
/// overshoot, and the `1 ..= K + 1` step token counts — `1` with no proposer), or `0` when the
/// runner is off (`cuda_graphs`). `None` on overflow (the caller fails closed).
fn engine_graph_workspace(
    model: &Decoder,
    drafts: u32,
    positions: u64,
    cuda_graphs: bool,
) -> Option<u64> {
    if !cuda_graphs {
        return Some(0);
    }
    graph_workspace_admission_bytes(
        &model.step_memory_geometry(drafts as usize),
        positions.checked_add(u64::from(drafts))?,
        drafts.checked_add(1)?,
    )
}

/// What the Gemma 4 soft-token splice re-admits once the expanded prompt (`capacity` = expanded
/// prompt + budget) is known: the static KV preallocation for those positions, plus — when the
/// CUDA-graph runner wraps the engine (`cuda_graphs`) — its graph workspace over the same reach
/// (sc-24140). `None` on overflow.
fn spliced_prompt_bytes(model: &Decoder, capacity: usize, cuda_graphs: bool) -> Option<u64> {
    let graphs = engine_graph_workspace(model, 0, u64::try_from(capacity).ok()?, cuda_graphs)?;
    u64::try_from(model.static_kv_bytes(capacity))
        .ok()?
        .checked_add(graphs)
}

impl Decoder {
    /// Bytes a static KV cache of `capacity` positions preallocates for this decoder (the
    /// model's own `static_kv_bytes`) — what a request that ran past that capacity was refused
    /// for (sc-24140).
    fn static_kv_bytes(&self, capacity: usize) -> usize {
        match self {
            Decoder::Causal(m) => m.static_kv_bytes(capacity),
            Decoder::Qwen35(m) => m.static_kv_bytes(capacity),
        }
    }

    /// The geometry admission prices for a request on the provider's own growing caches (the
    /// reference paths keep [`REFERENCE_MAX_CHECKPOINTS`](crate::models::qwen35::REFERENCE_MAX_CHECKPOINTS)
    /// — no checkpoint ring — so the recurrent term is one live state per linear layer).
    fn memory_geometry(&self) -> LlmMemoryGeometry {
        self.memory_geometry_with_checkpoints(crate::models::qwen35::REFERENCE_MAX_CHECKPOINTS)
    }

    /// The geometry for a request that runs through the step seam — the speculative engine with
    /// `drafts` drafts per verify step — whose cache ([`StepModel::new_cache_for`] with overshoot
    /// `drafts`) holds a per-token checkpoint ring of `drafts + 2` recurrent states per linear
    /// layer (the step start plus the `K + 1` verify positions; sc-24131), and whose verify step
    /// writes `K + 1` positions past the budget (E6). The recurrent term is exactly the ring —
    /// there is no separate start-of-step checkpoint any more.
    ///
    /// [`StepModel::new_cache_for`]: crate::decode::StepModel::new_cache_for
    fn step_memory_geometry(&self, drafts: usize) -> LlmMemoryGeometry {
        self.memory_geometry_with_checkpoints(drafts.saturating_add(1))
    }

    /// The geometry for a cache whose linear layers can roll back `retained_checkpoints`
    /// positions: `1 + retained_checkpoints` recurrent states per linear layer, priced exactly
    /// ([`Qwen35Model::recurrent_state_bytes`]).
    fn memory_geometry_with_checkpoints(&self, retained_checkpoints: usize) -> LlmMemoryGeometry {
        let (query_heads, kv_heads, head_dim, layers, hidden, intermediate, vocab, recurrent) =
            match self {
                Decoder::Causal(m) => {
                    let c = m.config();
                    // The widest layer's KV geometry — the one layer with the largest
                    // `kv_heads × head width` — so `layers × kv_heads × head_dim` covers every
                    // layer's cache whatever its type, without crossing one layer type's head count
                    // with another's width: Gemma 4's full-attention layers can be wider than the
                    // scalar `head_dim` / `num_key_value_heads` (its sliding layers'), and
                    // DeepSeek-V2's materialized MLA caches full-head `qk_nope + qk_rope` keys
                    // (sc-24138, E6 — the step seam's static preallocation is this layout, and the
                    // engine path the provider runs for this family allocates it).
                    let (kv_heads, head_dim) = m.kv_layout().widest_layer();
                    (
                        c.num_heads,
                        kv_heads as i32,
                        head_dim as i32,
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
                        m.recurrent_state_bytes(1 + retained_checkpoints) as u64,
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

    /// How the decoder computes grouped-query attention (story sc-24132), for the decode record of
    /// the provider's reference loop (the engine stamps its own record from the cache it ran on).
    /// The generic causal family reports what its selector really ran over the growing cache
    /// (sc-24138: `Gqa` by default — the static cache's arithmetic — when every layer can express
    /// it; `Expanded` when that pre-migration arithmetic is selected as a comparison, or when a
    /// layer cannot attend un-expanded: a Gemma 2 soft-cap, a Gemma 4 sliding window, MLA); the
    /// Qwen3.5 hybrid reports its selector.
    fn attn_formulation(&self) -> crate::primitives::AttnFormulation {
        match self {
            Decoder::Causal(m) => m.effective_attn_formulation(m.attn_formulation()),
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
    /// What the load produced (sc-24135): the requested weight format and the resident weight
    /// census by projection kind (the qwen3_5 and — sc-24140 — llama families). Read through
    /// [`LlamaProvider::load_record`].
    load_record: LoadRecord,
    /// Which loop a request whose speculation is off decodes on, for both families (sc-24138,
    /// sc-24140): [`DecodePath::StepModel`] — the unified engine over the step seam on the static
    /// KV cache, the default — or [`DecodePath::Reference`], the `Decode` loop kept as the parity
    /// oracle. Selected with [`LlamaProvider::set_decode_path`].
    decode_path: DecodePath,
}

/// The load telemetry of a [`LlamaProvider`] (sc-24135, epic sc-24128 E2): which weight format was
/// requested and which projection kinds the decoder actually holds, with their resident bytes.
///
/// An NVFP4 request never reports a dense projection under an NVFP4 label: the qwen3_5 family
/// loads every requested projection under `census.projections.nvfp4` or fails at load, and the
/// llama family (sc-24140) keeps a projection whose shape the FP4 GEMM cannot serve dense, counted
/// under `census.projections.dense` — this record is how a caller (and the evidence harness)
/// sees which, and reads the resident bits/param.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct LoadRecord {
    /// The load-time weight format the caller requested (`None` = the checkpoint's own dtype, or
    /// its persisted `quantization` block).
    pub requested: Option<Quantize>,
    /// The decoder's resident weights by projection kind (target plus native MTP predictor): the
    /// qwen3_5-family decoders and, since sc-24140, the llama family (`CausalLm`, safetensors or
    /// GGUF); `None` for the Prism/Bonsai GGUF hybrid and a provider assembled from parts.
    pub census: Option<WeightCensus>,
    /// The CUDA-graph switch this provider was loaded under (sc-24139): `LoadSpec::cuda_graphs`,
    /// else the process switch at load time. The CUDA stream is settled at load, so every
    /// generation runs under this switch rather than whatever the process switch says later.
    /// `None` for a provider assembled without a load ([`LlamaProvider::from_parts`]), which
    /// follows the process switch.
    pub cuda_graphs: Option<bool>,
}

impl LoadRecord {
    /// The backend-neutral report a product renders (sc-24139): the requested format, the
    /// projection kinds actually resident (only the kinds present) and the CUDA-graph switch the
    /// load settled ([`cuda_graphs`](Self::cuda_graphs)).
    pub fn report(&self) -> core_llm::LoadReport {
        let projections = self
            .census
            .map(|census| {
                [
                    ProjectionKind::Dense,
                    ProjectionKind::Ggml,
                    ProjectionKind::Prism,
                    ProjectionKind::Nvfp4,
                ]
                .into_iter()
                .filter_map(|kind| {
                    let tally = census.projections.tally(kind);
                    (tally.count > 0).then(|| core_llm::ProjectionReport {
                        kind: kind.label().to_string(),
                        count: tally.count,
                        params: tally.params,
                        resident_bytes: tally.resident_bytes,
                    })
                })
                .collect()
            })
            .unwrap_or_default();
        core_llm::LoadReport {
            requested: self.requested,
            projections,
            cuda_graphs: self.cuda_graphs,
        }
    }
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
/// carrying "nvfp4"), never a fallback to another representation. A source whose family cannot
/// hold NVFP4 projections is refused by name too, also before admission, so a memory refusal can
/// never mask the capability refusal: a GGUF file or a Prism/Bonsai snapshot (both already
/// packed; NVFP4 quantizes from a dense snapshot). The qwen3_5 hybrid (sc-24135) and every
/// llama-family `CausalLm` decoder (sc-24140 — Llama/Mistral, Qwen3 dense, Phi-3, Qwen2-MoE,
/// Gemma 2/4, GLM-4, DeepSeek-V2, the Qwen3-VL decoder) are served; a multimodal tower beside the
/// decoder stays dense, as on the qwen3_5 family, and LLaVA — whose provider owns the tower
/// load — refuses NVFP4 itself.
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
    nvfp4_model_gate(spec)?;
    gate(device).map(Some).map_err(to_core)
}

/// The model half of [`nvfp4_format`]: whether this provider serves NVFP4 for the checkpoint at
/// `spec.source`, from `config.json` and the snapshot's tensor names (never tensor data). The load
/// runs it before the device gate, and [`crate::backend::nvfp4_support`] answers a product's
/// per-snapshot question with it (sc-24139), so the two can never disagree about which
/// checkpoints NVFP4 serves.
///
/// A GGUF source is refused: it is already block-quantized, and NVFP4 quantizes from a dense
/// snapshot. So is a Prism/Bonsai snapshot (already packed affine-2 — `load_dir` refuses any
/// repacking), by name ([`nvfp4_family_refusal`]), and a packed MLX-affine snapshot of any family
/// ([`packed_affine_refusal`] — the loader refuses its triples too, but only after reading the
/// weights). The qwen3_5 hybrid (sc-24135) and every llama-family `CausalLm` architecture
/// (sc-24140) are served. A missing or unreadable config is left to the loader's own error.
pub(crate) fn nvfp4_model_gate(spec: &LoadSpec) -> CoreResult<()> {
    if crate::gguf::is_gguf_path(&spec.source) {
        return Err(CoreError::Unsupported(
            "nvfp4: NVFP4 projections are quantized from a dense safetensors snapshot; a GGUF \
             checkpoint is already block-quantized"
                .into(),
        ));
    }
    // The family rule (sc-24140): the one place it lives, so every caller — the load path and the
    // capability probe alike — answers the same. A missing or unreadable config, or an
    // architecture the dispatch does not recognize, is left to the loader's own error.
    let dir = Path::new(&spec.source);
    if let Some(config) = read_json(dir, "config.json") {
        if let Some(reason) =
            nvfp4_family_refusal(&config).or_else(|| packed_affine_refusal(dir, &config))
        {
            return Err(CoreError::Unsupported(format!("nvfp4: {reason}")));
        }
    }
    Ok(())
}

/// Why a snapshot's projections are already packed MLX-affine, or `None` (the packed half of
/// [`nvfp4_model_gate`], sc-24140 feature-end review). Packed means both: a `quantization` block
/// in `config.json` (top level or `text_config`, where [`ModelConfig`] reads it) **and** a
/// `<stem>.weight` whose `<stem>.scales` sidecar the snapshot names. A prepared Q4/Q8 snapshot
/// ([`crate::prepare`]) whose block says `storage: "ggml"` stores its projections as GGML blocks
/// (sc-19375) and is packed too; an older prepared snapshot carries the block over *dense*
/// weights, so it is not packed: NVFP4 quantizes its projections like any dense snapshot's. The
/// tensor names come from `model.safetensors.index.json` when it exists, else from each shard's
/// header — never from tensor data. An unreadable index or header is left to the loader's own
/// error.
fn packed_affine_refusal(dir: &Path, config: &Value) -> Option<String> {
    let block = config.get("quantization").or_else(|| {
        config
            .get("text_config")
            .and_then(|text| text.get("quantization"))
    });
    let block = block?;
    if block.get("storage").and_then(Value::as_str) == Some(crate::prepare::GGML_STORAGE) {
        return Some(
            "this snapshot's projections are stored as GGML blocks (a prepared Q4 / Q8 tier); \
             NVFP4 projections are quantized from a dense snapshot"
                .into(),
        );
    }
    let names = snapshot_tensor_names(dir)?;
    let mut stems: Vec<&str> = names
        .iter()
        .filter_map(|name| {
            let stem = name.strip_suffix(".scales")?;
            names.contains(&format!("{stem}.weight")).then_some(stem)
        })
        .collect();
    // The first packed projection by name, so the refusal reads the same on every run.
    stems.sort_unstable();
    let stem = stems.first()?;
    Some(format!(
        "this snapshot's projections are packed MLX-affine (a `quantization` block over affine \
         sidecars such as `{stem}.scales`); NVFP4 projections are quantized from a dense snapshot"
    ))
}

/// The tensor names a safetensors snapshot holds: the index's `weight_map` keys, else every
/// `*.safetensors` shard's header keys (the 8-byte length prefix and the JSON header only).
/// `None` when neither can be read.
fn snapshot_tensor_names(dir: &Path) -> Option<std::collections::HashSet<String>> {
    if let Some(index) = read_json(dir, "model.safetensors.index.json") {
        let map = index.get("weight_map")?.as_object()?;
        return Some(map.keys().cloned().collect());
    }
    let mut names = std::collections::HashSet::new();
    for entry in std::fs::read_dir(dir).ok()? {
        let path = entry.ok()?.path();
        if path.extension().and_then(|e| e.to_str()) == Some("safetensors") {
            names.extend(safetensors_header_names(&path)?);
        }
    }
    Some(names)
}

/// The tensor names in one safetensors file's header, without reading its data.
fn safetensors_header_names(path: &Path) -> Option<Vec<String>> {
    Some(
        safetensors_header(path)?
            .into_iter()
            .map(|(key, _)| key)
            .filter(|key| key.as_str() != "__metadata__")
            .collect(),
    )
}

/// One safetensors file's JSON header (every tensor's `dtype`, `shape` and offsets, plus any
/// `__metadata__`): the 8-byte length prefix and the header only, never tensor data.
fn safetensors_header(path: &Path) -> Option<serde_json::Map<String, Value>> {
    use std::io::Read;
    // A real header is kilobytes to a few megabytes; never allocate for a corrupt length.
    const MAX_HEADER_BYTES: u64 = 256 << 20;
    let mut file = std::fs::File::open(path).ok()?;
    let mut len = [0u8; 8];
    file.read_exact(&mut len).ok()?;
    let len = u64::from_le_bytes(len);
    if len > MAX_HEADER_BYTES {
        return None;
    }
    let mut header = vec![0u8; len as usize];
    file.read_exact(&mut header).ok()?;
    match serde_json::from_slice(&header).ok()? {
        Value::Object(map) => Some(map),
        _ => None,
    }
}

/// Test seam for [`packed_affine_refusal`]: write tiny tensors named `names` into `dir` — as a
/// `model.safetensors` shard, or (`index`) only a `model.safetensors.index.json` naming them.
#[cfg(test)]
pub(crate) fn write_test_snapshot_tensors(dir: &Path, names: &[&str], index: bool) {
    if index {
        let map: serde_json::Map<String, Value> = names
            .iter()
            .map(|name| {
                (
                    name.to_string(),
                    Value::from("model-00001-of-00001.safetensors"),
                )
            })
            .collect();
        let index = serde_json::json!({"metadata": {}, "weight_map": map});
        std::fs::write(dir.join("model.safetensors.index.json"), index.to_string()).unwrap();
        return;
    }
    let tensors: std::collections::HashMap<String, Tensor> = names
        .iter()
        .map(|name| {
            let dtype = if name.ends_with(".weight") {
                DType::U32
            } else {
                DType::BF16
            };
            let tensor = Tensor::zeros((2, 2), dtype, &Device::Cpu).unwrap();
            (name.to_string(), tensor)
        })
        .collect();
    candle_core::safetensors::save(&tensors, dir.join("model.safetensors")).unwrap();
}

/// Why a snapshot's family cannot hold NVFP4 projections, or `None` when it can (the family half of
/// [`nvfp4_format_with`]). Prism/Bonsai snapshots are already packed affine-2; the qwen3_5 hybrid
/// and every llama-family (`CausalLm`) architecture quantize their dense projections through the
/// shared loader ([`Projection::load_as`](crate::primitives::Projection::load_as) /
/// [`load_eligible`](crate::primitives::Projection::load_eligible)).
fn nvfp4_family_refusal(config: &Value) -> Option<String> {
    if config.get("model_type").and_then(Value::as_str) == Some("prism_hadamard_qwen35") {
        return Some(
            "a Prism/Bonsai snapshot is already packed affine-2; NVFP4 projections are quantized \
             from a dense snapshot"
                .into(),
        );
    }
    match Architecture::from_config(config) {
        // Every architecture the provider dispatches decodes through `Qwen35Model` or `CausalLm`,
        // and both load NVFP4 projections.
        Ok(
            Architecture::Qwen35
            | Architecture::Llama
            | Architecture::Qwen3
            | Architecture::Phi3
            | Architecture::Qwen2Moe
            | Architecture::Gemma2
            | Architecture::Glm4
            | Architecture::DeepseekV2
            | Architecture::Qwen3Vl
            | Architecture::Gemma4Unified
            | Architecture::Gemma4,
        )
        | Err(_) => None,
    }
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

    /// Select the loop a request whose speculation is off decodes on — one selector for both
    /// families (sc-24138, sc-24140): [`DecodePath::StepModel`] (the default) runs the unified
    /// engine over the step seam with no proposer on the static KV cache (through the CUDA-graph
    /// runner when it is on); [`DecodePath::Reference`] runs the `Decode` loop on the growing
    /// cache — the parity oracle. Any other path is [`CoreError::InvalidRequest`]: speculation is
    /// chosen per request (MTP), and a request with an MTP plan always runs the engine with the
    /// MTP proposer, whatever this says. A Qwen3-VL generic-causal multimodal request (DeepStack /
    /// M-RoPE prefill, no step-seam form) decodes on the reference loop either way, and its
    /// record says so.
    pub fn set_decode_path(&mut self, path: DecodePath) -> CoreResult<()> {
        match path {
            DecodePath::StepModel | DecodePath::Reference => {
                self.decode_path = path;
                Ok(())
            }
            other => Err(CoreError::InvalidRequest(format!(
                "[candle-llama] the decode path is `step_model` or `reference`, not `{}`",
                other.label()
            ))),
        }
    }

    /// The loop a request whose speculation is off decodes on (see
    /// [`set_decode_path`](Self::set_decode_path)).
    pub fn decode_path(&self) -> DecodePath {
        self.decode_path
    }

    /// Which loop this request decodes on: the engine with the MTP proposer for an MTP plan, the
    /// engine with no proposer for a request whose speculation is off (the selector's default —
    /// both families), the reference loop when the selector asks for it or the request is a
    /// Qwen3-VL generic-causal multimodal one. `qwen_vl_multimodal` is whether the request
    /// carries Qwen-VL visuals.
    fn decode_route(&self, mtp_plan: core_llm::MtpPlan, qwen_vl_multimodal: bool) -> DecodeRoute {
        if let Some(drafts) = mtp_plan.draft_tokens() {
            return DecodeRoute::Engine { drafts };
        }
        match (&self.model, self.decode_path) {
            (_, DecodePath::Reference) => DecodeRoute::Reference,
            (Decoder::Causal(_), _) if qwen_vl_multimodal => DecodeRoute::Reference,
            _ => DecodeRoute::Engine { drafts: 0 },
        }
    }

    /// Bridge an error from a request's decode into the contract error. A static KV cache asked
    /// for more positions than it holds ([`crate::Error::KvCapacityExceeded`] — at its
    /// construction for the request's bound plus the verify overshoot, or at a step) is the typed
    /// [`CoreError::RequestResourceExhausted`] with the request's geometry and the figures the
    /// refusal is about: the preallocation the requested positions would take against the one the
    /// cache can hold (sc-24140), never an opaque backend error. Everything else is [`to_core`].
    fn request_error(
        &self,
        error: crate::Error,
        prompt_tokens: usize,
        max_new_tokens: u32,
    ) -> CoreError {
        match error {
            crate::Error::KvCapacityExceeded {
                requested,
                capacity,
            } => CoreError::RequestResourceExhausted(core_llm::RequestResourceExhausted {
                prompt_tokens,
                max_new_tokens,
                max_context_tokens: self.descriptor.capabilities.max_context_tokens,
                required_bytes: self.model.static_kv_bytes(requested) as u64,
                available_bytes: self.model.static_kv_bytes(capacity) as u64,
            }),
            other => to_core(other),
        }
    }

    /// Load a provider from `spec.source`: either a `*.gguf` file (loaded directly via Candle's
    /// native GGUF reader, story 7254) or an HF snapshot directory (config.json + tokenizer.json +
    /// shards). Either way the decoder architecture is dispatched (Llama / Mistral / Qwen3) and the
    /// projections are optionally quantized on load per `spec.quantize`.
    pub fn load(spec: &LoadSpec) -> CoreResult<Self> {
        if spec.projector_source.is_some() && !crate::gguf::is_gguf_path(&spec.source) {
            return Err(CoreError::Load("an external projector is only valid with a GGUF language checkpoint; safetensors vision must be embedded".into()));
        }
        // The load's CUDA-graph policy (sc-24139): settled before the device is opened — the
        // graph runner needs the model on its own CUDA stream, which the device selection picks
        // under this switch — and held for the whole load, then recorded so every generation on
        // this provider runs under the same switch.
        let cuda_graphs = spec.cuda_graphs.unwrap_or_else(cuda_graphs_enabled);
        let _cuda_graphs_scope = crate::decode::cuda_graphs_scope(Some(cuda_graphs));
        let device = select_device().map_err(to_core)?;
        // NVFP4 (sc-24135): the capability floor is settled first — before the accelerator gate,
        // admission or any weight read — so a CPU or sub-sm_120 device answers an NVFP4 request with
        // the typed refusal naming the capability, never a fallback to another representation.
        let nvfp4 = nvfp4_format(spec, &device)?;
        ensure_supported_device(Path::new(&spec.source), &device)?;
        let LoadMemoryEstimate {
            host_required_bytes: host_required,
            device_required_bytes: device_required,
            ..
        } = Self::load_memory_estimate(spec, device.is_cuda())?;
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
        provider.load_record.cuda_graphs = Some(cuda_graphs);
        Ok(provider)
    }

    /// The memory a [`load`](Self::load) of `spec` admits against before it reads a single
    /// weight (E6): the host staging bound and, on CUDA (`cuda`), the device bound — the source
    /// the loader holds resident while it builds the decoder, 25 percent headroom over it, and
    /// the copies it builds beside that source: the quantized projection copy of a Q4 / Q8 /
    /// NVFP4 load ([`LoadMemoryEstimate::quantized_copy_bytes`]) and, for a llama-family GGUF,
    /// the compute-dtype cast of every tensor it keeps dense over the dense f32 map it
    /// dequantizes the file into (sc-24140). `load` admits with exactly this function, so a
    /// caller (or the residency evidence) reads the figure the load was admitted against. Reads
    /// only file sizes, `config.json` and tensor headers — never tensor data; an unreadable
    /// source is [`CoreError::Load`].
    pub fn load_memory_estimate(spec: &LoadSpec, cuda: bool) -> CoreResult<LoadMemoryEstimate> {
        let source = Path::new(&spec.source);
        let payload = core_llm::checkpoint_payload_bytes(source)?;
        let staging = core_llm::checkpoint_staging_bytes(source)?;
        let projector = spec
            .projector_source
            .as_ref()
            .map(|p| core_llm::checkpoint_payload_bytes(Path::new(p)))
            .transpose()?
            .unwrap_or(0);
        let overflow = || CoreError::Load("load memory estimate overflow".into());
        let requested = CopyFormat::requested(spec.quantize);
        // What the load holds on its device while it builds the decoder.
        let working = if crate::gguf::is_gguf_path(&spec.source) {
            if crate::prism_checkpoint::PrismGgufCheckpoint::is_prism(source).map_err(to_core)? {
                // Prism GGUF: packed blocks the loader wraps, never re-quantized.
                LoadWorkingSet::resident_payload(payload, 0, 0, true)
            } else {
                gguf_working_set(source, requested, cuda)?
            }
        } else {
            let config = read_json(source, "config.json");
            let decoder = PricedDecoder::from_config(config.as_ref());
            let format = decoder.format(requested);
            let builds = matches!(decoder, PricedDecoder::Causal(_) | PricedDecoder::Qwen35(_));
            let (cast, copy) = if builds && (cuda || format.is_some()) {
                let tensors = snapshot_tensor_headers(source)?;
                let (copy, consumed) = match format {
                    Some(format) => snapshot_quantized_copy_bytes(&decoder, format, &tensors)
                        .ok_or_else(overflow)?,
                    None => (0, Default::default()),
                };
                // Off CUDA the compute dtype is f32 and the host rule's three payloads cover it.
                let cast = if cuda {
                    snapshot_cast_copy_bytes(&tensors, &consumed).ok_or_else(overflow)?
                } else {
                    0
                };
                (cast, copy)
            } else {
                (0, 0)
            };
            LoadWorkingSet::resident_payload(payload, cast, copy, decoder.is_packed())
        };
        let (host_required, device_required) =
            load_memory_requirements(payload, staging, projector, cuda, working)
                .ok_or_else(overflow)?;
        let on_device = |bytes: u64| if cuda { bytes } else { 0 };
        Ok(LoadMemoryEstimate {
            payload_bytes: payload,
            host_required_bytes: host_required,
            device_required_bytes: device_required,
            source_bytes: on_device(working.source),
            cast_copy_bytes: on_device(working.cast),
            quantized_copy_bytes: on_device(working.copy),
        })
    }

    /// Load from an HF snapshot directory (config.json + tokenizer.json + safetensors shards).
    ///
    /// `requested` is an explicit load-time quantization (`spec.quantize`); when it is `None` a
    /// llama-family snapshot's own persisted `quantization` block (written by the
    /// [`prepare`](crate::prepare) writer) is honored, so a `LoadSpec::dense` of a prepared Q4/Q8
    /// snapshot loads quantized. The qwen3_5 decoder reads only `requested`.
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
            // An explicit request (Q4 / Q8 / NVFP4, sc-24140) wins; otherwise the snapshot's own
            // persisted `quantization` block, as before.
            let format = requested
                .cloned()
                .or_else(|| cfg.quantization.map(ProjectionFormat::from));
            let m = CausalLm::from_weights_format(&weights, "", cfg, format.as_ref())
                .map_err(to_core)?;
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
            // sc-24140: the llama family reports its census too (bits/param by projection kind).
            Decoder::Causal(m) => Some(m.weight_census()),
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
            decode_path: DecodePath::StepModel,
            vision,
            gemma4,
            load_record: LoadRecord {
                requested: None,
                census,
                cuda_graphs: None,
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
                decode_path: DecodePath::StepModel,
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
        let causal = CausalLm::from_weights_with(&ck.weights, "", ck.config.clone(), quant)
            .map_err(to_core)?;
        // sc-24140: the llama family reports its census on every load path.
        let census = Some(causal.weight_census());
        let model = Decoder::Causal(causal);

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
            decode_path: DecodePath::StepModel,
            constraint_table: OnceCell::new(),
            vision: None, // GGUF is the dense Llama-family path only — no Qwen3.6 VLM.
            // Likewise no Gemma 4 front-ends: the GGUF path reconstructs a dense text decoder,
            // and llama.cpp GGUFs carry no vision embedder / audio projector tensors.
            gemma4: None,
            load_record: LoadRecord {
                requested: None,
                census,
                cuda_graphs: None,
            },
        })
    }

    /// Whether the loaded model's projections are quantized.
    pub fn is_quantized(&self) -> bool {
        self.model.is_quantized()
    }

    /// The loaded Llama-family decoder, or `None` for the Qwen3.6 hybrid. An engine that runs its
    /// own decode loop over raw logits (YuE's CFG / allow-range / teacher-forced loops, epic
    /// sc-19373) loads through [`LlamaProvider::load`] — the same `LoadSpec`, admission and
    /// persisted-`quantization` handling as every other caller — and drives this directly.
    pub fn causal_lm(&self) -> Option<&CausalLm> {
        match &self.model {
            Decoder::Causal(m) => Some(m),
            Decoder::Qwen35(_) => None,
        }
    }

    /// [`LlamaProvider::causal_lm`] by value: consume the provider and keep only its Llama-family
    /// decoder (`None` for the Qwen3.6 hybrid). For an engine that must own the decoder in a
    /// `Send` type — the provider itself is not `Send` (its chat template is a non-`Send` trait
    /// object) — after loading through the same [`LlamaProvider::load`] path. The tokenizer, chat
    /// template and decode records are dropped; the decoder's weights are moved, not copied.
    pub fn into_causal_lm(self) -> Option<CausalLm> {
        match self.model {
            Decoder::Causal(m) => Some(m),
            Decoder::Qwen35(_) => None,
        }
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
            decode_path: DecodePath::StepModel,
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
/// The device holds the load's [`LoadWorkingSet`]: its resident source plus the 25 percent
/// headroom over it, and the copies the loader builds beside that source. A quantizing load
/// quantizes on the device while the source tensors are still resident (`Weights::from_dir` holds
/// every source tensor until the decoder is built), so the device additionally holds the growing
/// quantized copy of the projections the loader quantizes ([`snapshot_quantized_copy_bytes`]).
/// The GGML copies were unpriced until sc-24140's terminal measurement showed a Qwen3-8B Q8 load
/// peaking above the dense bound. The 25 percent headroom still covers the transient f32 copy of
/// the largest tensor the quantizer reads, and the CUDA pool's fragmentation around it.
///
/// A llama-family GGUF ([`gguf_working_set`]) holds its dense f32 map as the source and, on a
/// host device, prices the host domain by the same working set: there the f32 map *is* host
/// memory.
fn load_memory_requirements(
    payload: u64,
    staging: u64,
    projector: u64,
    cuda: bool,
    working: LoadWorkingSet,
) -> Option<(u64, Option<u64>)> {
    let projector_bound = projector.checked_mul(4)?;
    let working_bytes = working
        .source
        .checked_add(working.source / 4)?
        .checked_add(working.cast)?
        .checked_add(working.copy)?;
    let host = match (working.host, cuda) {
        (HostRule::Dense, true) => staging,
        (HostRule::Dense, false) => payload.checked_mul(3)?,
        (HostRule::Packed, _) | (HostRule::Gguf, true) => payload.checked_mul(2)?,
        (HostRule::Gguf, false) => working_bytes,
    }
    .checked_add(projector_bound)?;
    let device = if cuda {
        Some(working_bytes.checked_add(projector_bound)?)
    } else {
        None
    };
    Some((host, device))
}

/// What [`LlamaProvider::load`] must find free before it reads a weight (E6), per allocation
/// domain — the figure the load admits against ([`LlamaProvider::load_memory_estimate`]), with
/// the parts of the device bound broken out. Only the estimate builds one (`#[non_exhaustive]`).
#[non_exhaustive]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct LoadMemoryEstimate {
    /// The checkpoint payload on disk: every `*.safetensors` shard of the snapshot, or the file.
    pub payload_bytes: u64,
    /// Host bytes the load needs: one shard of staging on a CUDA load of a dense snapshot; on a
    /// host device a llama-family GGUF's whole working set (the f32 map it dequantizes into plus
    /// its quantized copy).
    pub host_required_bytes: u64,
    /// Device bytes a CUDA load needs (`None` on a host device): `source_bytes`, 25 percent
    /// headroom over it, `cast_copy_bytes`, `quantized_copy_bytes` and any external projector's
    /// bound.
    pub device_required_bytes: Option<u64>,
    /// The source the load holds resident on the device while it builds the decoder: the
    /// payload of a safetensors or Prism checkpoint, the dense f32 map a llama-family GGUF is
    /// dequantized into (`0` off CUDA).
    pub source_bytes: u64,
    /// The compute-dtype (bf16) casts the load builds beside that source for the tensors it
    /// keeps dense: a llama-family GGUF's, cast from its f32 map, and a safetensors snapshot's
    /// tensors stored in a float dtype other than bf16 (f16, f32) (`0` for a bf16 snapshot, a
    /// Prism checkpoint, and off CUDA).
    pub cast_copy_bytes: u64,
    /// The quantized projection copy the load builds beside the source: exactly the projections
    /// the loader stores in the requested (or a snapshot's persisted) format, at the bytes the
    /// quantizer allocates for them (`0` for a dense or Prism load, and off CUDA).
    pub quantized_copy_bytes: u64,
}

/// How a load prices its host domain.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum HostRule {
    /// A dense safetensors snapshot: one shard of staging on CUDA; the source plus the constructed
    /// f32 tensors (three payloads) on a host device.
    Dense,
    /// A packed Prism checkpoint: two payloads.
    Packed,
    /// A llama-family GGUF: tensor-at-a-time staging within two payloads on CUDA; the working set
    /// itself on a host device.
    Gguf,
}

/// What a load holds on the device it loads onto while it builds the decoder (E6, sc-24140).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct LoadWorkingSet {
    /// The source tensors held resident until the decoder is built.
    source: u64,
    /// Compute-dtype casts of the tensors kept dense, built beside the source.
    cast: u64,
    /// The quantized projection copy, built beside the source.
    copy: u64,
    /// How the host domain is priced.
    host: HostRule,
}

impl LoadWorkingSet {
    /// A safetensors or Prism load: the payload itself is the resident source, plus the bf16
    /// `cast` of every tensor stored in another float dtype ([`snapshot_cast_copy_bytes`]) and
    /// the quantized `copy`.
    fn resident_payload(payload: u64, cast: u64, copy: u64, packed: bool) -> Self {
        Self {
            source: payload,
            cast,
            copy,
            host: if packed {
                HostRule::Packed
            } else {
                HostRule::Dense
            },
        }
    }
}

/// The projection format a load stores its quantized copy in, as admission prices it.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum CopyFormat {
    /// GGML Q4_K / Q8_0 (an explicit Q4 / Q8 request, or a snapshot's persisted block).
    Ggml(QuantSpec),
    /// NVFP4.
    Nvfp4,
}

impl CopyFormat {
    /// The format an explicit `spec.quantize` asks for — the specs `load` builds from it.
    fn requested(quantize: Option<Quantize>) -> Option<Self> {
        Some(match quantize? {
            Quantize::Q4 => Self::Ggml(QuantSpec::q4()),
            Quantize::Q8 => Self::Ggml(QuantSpec::q8()),
            Quantize::Nvfp4 => Self::Nvfp4,
        })
    }

    /// Whether the loader keeps a `[rows, cols]` projection dense under this format: an NVFP4
    /// shape the FP4 GEMM cannot serve (the test `load_eligible` applies).
    fn keeps_dense(self, rows: usize, cols: usize) -> bool {
        self == Self::Nvfp4 && nvfp4_device_bytes(rows, cols).is_none()
    }

    /// Device bytes of a dense `[rows, cols]` projection stored in this format: its GGML blocks,
    /// or its NVFP4 nibbles and scales — nothing for an NVFP4 shape `load_eligible` keeps dense
    /// (it shares the source's storage). `None` on overflow.
    fn projection_bytes(self, rows: usize, cols: usize) -> Option<u64> {
        match self {
            Self::Ggml(quant) => {
                ggml_device_bytes((rows as u64).checked_mul(cols as u64)?, quant.dtype)
            }
            Self::Nvfp4 => Some(nvfp4_device_bytes(rows, cols).unwrap_or(0)),
        }
    }
}

/// The decoder a snapshot load builds, as admission prices its quantized copy (sc-24140).
enum PricedDecoder {
    /// A Prism/Bonsai snapshot: packed blocks the loader wraps, never re-quantized.
    Prism,
    /// A llama-family [`CausalLm`].
    Causal(ModelConfig),
    /// The qwen3_5 hybrid ([`Qwen35Model`] and its [`Qwen35Mtp`]).
    Qwen35(Qwen35Config),
    /// No readable config, or one the loader refuses before it builds a projection.
    Refused,
}

impl PricedDecoder {
    /// Dispatch `config` exactly as `load_dir` does.
    fn from_config(config: Option<&Value>) -> Self {
        let Some(config) = config else {
            return Self::Refused;
        };
        if config.get("model_type").and_then(Value::as_str) == Some("prism_hadamard_qwen35") {
            return Self::Prism;
        }
        match Architecture::from_config(config) {
            Ok(Architecture::Qwen35) => {
                Qwen35Config::from_json(config).map_or(Self::Refused, Self::Qwen35)
            }
            Ok(_) => ModelConfig::from_json(config).map_or(Self::Refused, Self::Causal),
            Err(_) => Self::Refused,
        }
    }

    fn is_packed(&self) -> bool {
        matches!(self, Self::Prism)
    }

    /// The format the loader stores the projections in: the request; else, for a `CausalLm`,
    /// the snapshot's persisted `quantization` block (a prepared Q4 / Q8 snapshot, which
    /// `load_dir` re-quantizes on load). The qwen3_5 loader reads only the request
    /// (`Qwen35Model::from_weights_format(.., requested)`), so a persisted block there is not
    /// priced: that load keeps its projections dense.
    fn format(&self, requested: Option<CopyFormat>) -> Option<CopyFormat> {
        match self {
            Self::Causal(cfg) => requested.or(cfg.quantization.map(CopyFormat::Ggml)),
            Self::Qwen35(_) => requested,
            Self::Prism | Self::Refused => None,
        }
    }
}

/// A checkpoint tensor as its safetensors header describes it (never its data).
#[derive(Clone, Debug, PartialEq, Eq)]
struct TensorHeader {
    /// The safetensors dtype tag (`BF16`, `U32`, …).
    dtype: String,
    /// The shape, outermost first.
    shape: Vec<usize>,
}

impl TensorHeader {
    fn dims2(&self) -> Option<(usize, usize)> {
        match self.shape[..] {
            [rows, cols] => Some((rows, cols)),
            _ => None,
        }
    }
}

/// Every tensor header of a snapshot, keyed by name.
type SnapshotTensors = std::collections::HashMap<String, TensorHeader>;

/// The tensor headers of the shards `Weights::from_dir` reads — every `*.safetensors` file in
/// `dir`, in name order, a later shard's tensor replacing an earlier one's — from each shard's
/// JSON header only. An unreadable directory or header is [`CoreError::Load`] (the load would
/// fail reading it too).
fn snapshot_tensor_headers(dir: &Path) -> CoreResult<SnapshotTensors> {
    let io_error =
        |error: std::io::Error| CoreError::Load(format!("checkpoint admission: {error}"));
    let mut shards: Vec<_> = std::fs::read_dir(dir)
        .map_err(io_error)?
        .filter_map(|entry| entry.ok().map(|entry| entry.path()))
        .filter(|path| path.extension().and_then(|e| e.to_str()) == Some("safetensors"))
        .collect();
    shards.sort();
    let mut tensors = SnapshotTensors::new();
    for shard in shards {
        let malformed = || {
            CoreError::Load(format!(
                "checkpoint admission: unreadable safetensors header in {}",
                shard.display()
            ))
        };
        for (name, entry) in safetensors_header(&shard).ok_or_else(malformed)? {
            if name == "__metadata__" {
                continue;
            }
            let dtype = entry.get("dtype").and_then(Value::as_str);
            let shape = entry
                .get("shape")
                .and_then(Value::as_array)
                .and_then(|dims| {
                    dims.iter()
                        .map(|d| d.as_u64().and_then(|d| usize::try_from(d).ok()))
                        .collect::<Option<Vec<usize>>>()
                });
            let (Some(dtype), Some(shape)) = (dtype, shape) else {
                return Err(malformed());
            };
            tensors.insert(
                name,
                TensorHeader {
                    dtype: dtype.to_string(),
                    shape,
                },
            );
        }
    }
    Ok(tensors)
}

/// The quantized projection copy a snapshot load builds on the device beside its resident source
/// tensors (E6, sc-24140): exactly the tensors the loader stores in `format`, chosen by the
/// loader's own rule — for a `CausalLm` the layer projections
/// ([`quantizes_layer_tensor`](crate::models::llama::quantizes_layer_tensor) under
/// [`decoder_root`](crate::models::llama::decoder_root)) and, under NVFP4, the head; for the
/// qwen3_5 hybrid [`quantizes_tensor`](crate::models::qwen35::quantizes_tensor) over the decoder
/// and a configured MTP, and the head under every format — each at the bytes the quantizer
/// allocates for its shape ([`CopyFormat::projection_bytes`]). A tied head is quantized from a
/// copy of the embedding, so the embedding's shape prices it. Embeddings, norms, a GGML-dense
/// head and any vision tower are never priced: the loader never copies them.
///
/// A GGML format over an MLX-affine triple (a `.scales` sidecar) is a Q8_0 repack, priced only
/// where the loader performs one ([`affine_q8_repack_bytes`]). Returns the copy's bytes and the
/// float tensors the loader consumes into it whole — never kept dense, so never cast beside
/// their source ([`snapshot_cast_copy_bytes`]). `None` on overflow.
fn snapshot_quantized_copy_bytes<'a>(
    decoder: &PricedDecoder,
    format: CopyFormat,
    tensors: &'a SnapshotTensors,
) -> Option<(u64, std::collections::HashSet<&'a str>)> {
    use crate::models::{llama, qwen35, split_layer_key};
    let mut total = 0u64;
    let mut consumed = std::collections::HashSet::new();
    let mut add = |bytes: u64| -> Option<()> {
        total = total.checked_add(bytes)?;
        Some(())
    };
    match decoder {
        PricedDecoder::Causal(cfg) => {
            let root = llama::decoder_root(cfg, "", |key| tensors.contains_key(key));
            for (key, tensor) in tensors {
                let Some((key_root, layer, suffix)) = split_layer_key(key) else {
                    continue;
                };
                if key_root != root
                    || layer >= cfg.num_layers
                    || !llama::quantizes_layer_tensor(suffix)
                {
                    continue;
                }
                let fused = matches!(
                    suffix,
                    "self_attn.qkv_proj.weight" | "mlp.gate_up_proj.weight"
                );
                // A stored GGML block tensor (a prepared Q4 / Q8 snapshot, sc-19375) is rebuilt
                // as a `QTensor` beside its source bytes: priced at the blocks the device holds.
                if let Some((stored, rows, cols)) = crate::primitives::quant::ggml_block_storage(
                    tensor.dtype == "U8",
                    &tensor.shape,
                ) {
                    add(ggml_device_bytes(
                        (rows as u64).checked_mul(cols as u64)?,
                        stored,
                    )?)?;
                    continue;
                }
                let stem = key.strip_suffix(".weight").unwrap_or(key);
                if let Some(scales) = tensors.get(&format!("{stem}.scales")) {
                    // A fused triple is refused by the loader; an unfused one is repacked.
                    if !fused {
                        let biases = tensors.get(&format!("{stem}.biases"));
                        add(affine_q8_repack_bytes(format, tensor, scales, biases)?)?;
                    }
                    continue;
                }
                let Some((rows, cols)) = tensor.dims2() else {
                    continue;
                };
                // The loader splits a fused tensor into its q / k / v or gate / up rows and
                // quantizes each part on its own.
                let parts = match suffix {
                    "self_attn.qkv_proj.weight" => {
                        let la = cfg.layer_attention(layer);
                        let head_dim = la.head_dim as usize;
                        let kvd = la.num_kv_heads as usize * head_dim;
                        vec![cfg.num_heads as usize * head_dim, kvd, kvd]
                    }
                    "mlp.gate_up_proj.weight" => {
                        let inter = match cfg.gemma4.as_deref() {
                            Some(g) => g.layer_intermediate_size(layer, cfg.intermediate_size),
                            None => cfg.intermediate_size,
                        } as usize;
                        vec![inter, inter]
                    }
                    _ => vec![rows],
                };
                // An NVFP4 part the FP4 GEMM cannot serve stays dense (as its source's cast).
                if !parts.iter().any(|&part| format.keeps_dense(part, cols)) {
                    consumed.insert(key.as_str());
                }
                for part in parts {
                    add(format.projection_bytes(part, cols)?)?;
                }
            }
            // `CausalLm` quantizes its head only under NVFP4; a tied head from the embedding.
            if format == CopyFormat::Nvfp4 {
                let head = if cfg.tie_word_embeddings {
                    format!("{root}.embed_tokens.weight")
                } else {
                    "lm_head.weight".to_string()
                };
                if let Some((rows, cols)) = tensors.get(&head).and_then(TensorHeader::dims2) {
                    // A tied head is a copy; the embedding itself stays dense.
                    if !cfg.tie_word_embeddings && !format.keeps_dense(rows, cols) {
                        consumed.insert("lm_head.weight");
                    }
                    add(format.projection_bytes(rows, cols)?)?;
                }
            }
        }
        PricedDecoder::Qwen35(cfg) => {
            let Ok(prefix) = qwen35_dense_prefix(|key| tensors.contains_key(key)) else {
                return Some((0, consumed));
            };
            let mtp_layers = cfg.mtp_num_hidden_layers;
            for (key, tensor) in tensors {
                let loaded = match split_layer_key(key) {
                    Some((root, layer, _)) => {
                        (root == prefix && layer < cfg.num_layers)
                            || (root == "mtp" && layer < mtp_layers)
                    }
                    None => mtp_layers > 0,
                };
                if !loaded || !qwen35::quantizes_tensor(key) {
                    continue;
                }
                consumed.insert(key.as_str());
                // The stacked MoE experts are quantized one expert slice at a time: a
                // `[mi, hidden]` gate and up from `gate_up_proj`, a `[hidden, mi]` down.
                let stacked = key.rsplit_once(".mlp.experts.").map(|(_, name)| name);
                match (stacked, cfg.moe, &tensor.shape[..]) {
                    (Some("gate_up_proj"), Some(moe), &[_, _, hidden]) => {
                        let slice =
                            format.projection_bytes(moe.moe_intermediate_size as usize, hidden)?;
                        for _ in 0..2 * moe.num_experts {
                            add(slice)?;
                        }
                    }
                    (Some("down_proj"), Some(moe), &[_, rows, cols]) => {
                        let slice = format.projection_bytes(rows, cols)?;
                        for _ in 0..moe.num_experts {
                            add(slice)?;
                        }
                    }
                    (Some(_), _, _) => {}
                    (None, _, _) => {
                        if let Some((rows, cols)) = tensor.dims2() {
                            add(format.projection_bytes(rows, cols)?)?;
                        }
                    }
                }
            }
            // The qwen3_5 head is quantized under every format; a tied one from the embedding.
            let head = if cfg.tie_word_embeddings {
                format!("{prefix}.embed_tokens.weight")
            } else {
                "lm_head.weight".to_string()
            };
            if let Some((rows, cols)) = tensors.get(&head).and_then(TensorHeader::dims2) {
                if !cfg.tie_word_embeddings {
                    consumed.insert("lm_head.weight");
                }
                add(format.projection_bytes(rows, cols)?)?;
            }
        }
        PricedDecoder::Prism | PricedDecoder::Refused => {}
    }
    Some((total, consumed))
}

/// The compute-dtype casts a CUDA load of a safetensors snapshot builds beside its resident
/// source (E6, sc-24140 review). The decoder takes every tensor it keeps dense through
/// `to_dtype(bf16)`, which shares a bf16 tensor's storage but copies any other float: an f16 or
/// f32 snapshot holds a second, 2-byte copy of each such tensor beside its source until the
/// decoder is built. So every float tensor not stored in bf16 is charged 2 bytes per element,
/// except those the loader consumes whole into its quantized copy (`consumed` — cast only
/// transiently, one projection at a time, inside the headroom) and an MLX-affine triple's
/// `.scales` / `.biases`, which the loader converts on the host. A tensor a multimodal tower casts
/// to f32 instead is charged its bf16 size; the rest of that cast stays in the headroom, as it
/// does for a bf16 snapshot. `None` on overflow.
fn snapshot_cast_copy_bytes(
    tensors: &SnapshotTensors,
    consumed: &std::collections::HashSet<&str>,
) -> Option<u64> {
    let mut total = 0u64;
    for (key, tensor) in tensors {
        // `F16`, `F32`, `F64`, `F8_*`: every safetensors float tag but `BF16`.
        if !tensor.dtype.starts_with('F') || consumed.contains(key.as_str()) {
            continue;
        }
        let sidecar = [".scales", ".biases"].iter().any(|sidecar| {
            key.strip_suffix(sidecar)
                .is_some_and(|stem| tensors.contains_key(&format!("{stem}.weight")))
        });
        if sidecar {
            continue;
        }
        let elems = tensor
            .shape
            .iter()
            .try_fold(1u64, |n, &d| n.checked_mul(d as u64))?;
        total = total.checked_add(elems.checked_mul(2)?)?;
    }
    Some(total)
}

/// The Q8_0 copy the loader repacks an MLX-affine triple into (`Projection::load_mlx_affine_q8`),
/// priced only where it performs the repack: a GGML format of 8 bits over an 8-bit triple — the
/// geometry [`mlx_affine_q8_in_dim`] accepts for the format's group size. Any other triple (a
/// 4-bit pack, a 4-bit format, NVFP4, or no format at all) is the loader's typed refusal, which
/// builds no copy: `Some(0)`. `None` on overflow.
fn affine_q8_repack_bytes(
    format: CopyFormat,
    weight: &TensorHeader,
    scales: &TensorHeader,
    biases: Option<&TensorHeader>,
) -> Option<u64> {
    let CopyFormat::Ggml(quant) = format else {
        return Some(0);
    };
    let geometry = (
        weight.dims2(),
        scales.dims2(),
        biases.and_then(TensorHeader::dims2),
    );
    let (Some(codes), Some(scales), Some(biases)) = geometry else {
        return Some(0);
    };
    let in_dim = (quant.bits() == 8)
        .then(|| {
            crate::primitives::quant::mlx_affine_q8_in_dim(
                weight.dtype == "U32",
                codes,
                scales,
                biases,
                quant.group_size(),
            )
        })
        .flatten();
    match in_dim {
        Some(in_dim) => ggml_device_bytes(
            (codes.0 as u64).checked_mul(in_dim as u64)?,
            candle_core::quantized::GgmlDType::Q8_0,
        ),
        None => Some(0),
    }
}

/// What a llama-family GGUF load holds on its device while it builds the decoder (E6, sc-24140).
///
/// [`GgufCheckpoint::open`] reads each tensor the loader maps ([`crate::gguf::remap_key`]) and
/// dequantizes it to a dense **f32** tensor on the loading device, keeping the whole map until the
/// provider is built — so the source is 4 bytes per element, several times the block-quantized
/// file. `CausalLm::from_weights_with` then builds the decoder beside that map: every tensor it
/// keeps dense is cast to the compute dtype (bf16 on CUDA — a second copy, 2 bytes per element;
/// on a host device the f32 compute dtype shares the source), and under a Q4 / Q8 request the
/// layer projections ([`quantizes_layer_tensor`](crate::models::llama::quantizes_layer_tensor))
/// are quantized to GGML ([`ggml_device_bytes`]). Read from the GGUF header with candle's own
/// reader, never tensor data. A Prism GGUF (packed blocks the loader wraps) is priced by
/// [`LoadWorkingSet::resident_payload`] instead.
fn gguf_working_set(
    path: &Path,
    requested: Option<CopyFormat>,
    cuda: bool,
) -> CoreResult<LoadWorkingSet> {
    use crate::models::{llama, split_layer_key};
    let unreadable = |error: String| {
        CoreError::Load(format!(
            "checkpoint admission: gguf {}: {error}",
            path.display()
        ))
    };
    let mut file = std::fs::File::open(path).map_err(|e| unreadable(e.to_string()))?;
    let content = candle_core::quantized::gguf_file::Content::read(&mut file)
        .map_err(|e| unreadable(e.to_string()))?;
    // A GGUF load re-quantizes only to GGML; NVFP4 refuses a GGUF before admission.
    let quant = match requested {
        Some(CopyFormat::Ggml(quant)) => Some(quant),
        Some(CopyFormat::Nvfp4) | None => None,
    };
    let (mut source, mut cast, mut copy) = (0u64, 0u64, 0u64);
    let mut price = |key: &str, elems: u64| -> Option<()> {
        source = source.checked_add(elems.checked_mul(4)?)?;
        let projection = split_layer_key(key)
            .is_some_and(|(_, _, suffix)| llama::quantizes_layer_tensor(suffix));
        match quant.filter(|_| projection) {
            Some(quant) => copy = copy.checked_add(ggml_device_bytes(elems, quant.dtype)?)?,
            None if cuda => cast = cast.checked_add(elems.checked_mul(2)?)?,
            None => {}
        }
        Some(())
    };
    for (name, info) in &content.tensor_infos {
        if let Some(key) = crate::gguf::remap_key(name) {
            price(&key, info.shape.elem_count() as u64)
                .ok_or_else(|| CoreError::Load("load memory estimate overflow".into()))?;
        }
    }
    Ok(LoadWorkingSet {
        source,
        cast,
        copy,
        host: HostRule::Gguf,
    })
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

    fn load_report(&self) -> Option<core_llm::LoadReport> {
        Some(self.load_record.report())
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
        // The load's CUDA-graph policy (sc-24139) governs everything this request does on this
        // thread — admission's workspace pricing, wrapping the step model, the runner's switch —
        // and the report says which switch it ran under.
        let _cuda_graphs_scope = crate::decode::cuda_graphs_scope(self.load_record.cuda_graphs);
        let cuda_graphs_on = cuda_graphs_enabled();

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
        // The backend-neutral mode resolution (core-llm): Auto on a model without a head decodes
        // normally, and the record says `proposer=none` (E2, sc-24130).
        let mtp_plan = core_llm::resolve_mtp_plan(req.mtp, self.descriptor.capabilities.mtp);
        // Which loop decodes this request (sc-24140): decided once, so admission prices the
        // cache and graphs that loop will actually hold.
        let route = self.decode_route(mtp_plan, multimodal && !gemma4_mm_request);
        let required = priced_request_bytes(
            &self.model,
            route,
            admitted_prompt,
            req.max_new_tokens,
            vision_workspace,
            cuda_graphs_enabled(),
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
        let mut engine_record: Option<DecodeRecord> = None;
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
            if let (DecodeRoute::Engine { drafts }, Decoder::Qwen35(target)) = (route, &self.model)
            {
                // The Qwen3.5 family through the unified engine over the step seam (sc-24130,
                // sc-24140): the MTP proposer for an MTP plan, no proposer — the token-at-a-time
                // loop, on the same static KV cache, checkpoint ring and CUDA-graph runner — for
                // a request whose speculation is off. The reference `Decode` loop stays
                // selectable (`set_decode_path`) as the oracle.
                let drafts = drafts as usize;
                let mut mtp_proposer = match mtp_plan.draft_tokens() {
                    Some(_) => Some(MtpProposer::new(self.mtp.as_ref().ok_or_else(|| {
                        CoreError::Load("MTP was advertised without a loaded predictor".into())
                    })?)),
                    None => None,
                };
                let speculative = mtp_proposer.is_some();
                let mut no_proposer = NoProposer;
                let constraint = json_mask
                    .as_mut()
                    .map(|m| m as &mut dyn RewindableConstraintMask);
                // The CUDA-graph runner (sc-24134) wraps the target when the switch is on: it
                // replays captured decode / verify steps where it can and falls back eager with
                // a named reason otherwise (`decode_record.cuda_graphs`); the engine is the
                // same either way.
                let graphs = GraphRunner::new(target);
                let stepper: &dyn StepModel<Cache = Qwen35Cache> = if cuda_graphs_enabled() {
                    &graphs
                } else {
                    target
                };
                let run = match &mm {
                    Some(m) => {
                        // Multimodal: the fused-embedding / M-RoPE prefill is the caller's, into
                        // the engine's own step cache; an MTP predictor is warmed from the same
                        // fused rows; the continuation positions are shifted by `mrope_delta`
                        // inside the cache.
                        let capacity = m.expanded_ids.len().saturating_add(config.max_new_tokens);
                        let mut cache = target
                            .new_cache_for(capacity, drafts)
                            .map_err(|e| self.request_error(e, prompt_len, req.max_new_tokens))?;
                        let (t, h, w, delta) = &m.positions;
                        let positions = [t.as_slice(), h.as_slice(), w.as_slice()];
                        let (logits, hidden) = target
                            .prefill_from_embeds_deepstack_with_hidden(
                                &m.embeds,
                                positions,
                                &mut cache,
                                &m.visual_pos_mask,
                                &m.deepstack,
                            )
                            .map_err(to_core)?;
                        cache.set_rope_delta(*delta);
                        if let Some(proposer) = mtp_proposer.as_mut() {
                            proposer
                                .warm_multimodal(&m.embeds, &hidden, positions)
                                .map_err(to_core)?;
                        }
                        target
                            .device()
                            .synchronize()
                            .map_err(|e| to_core(e.into()))?;
                        let prefill = generation_started.elapsed();
                        let decode_started = std::time::Instant::now();
                        let proposer: &mut dyn Proposer = match mtp_proposer.as_mut() {
                            Some(proposer) => proposer,
                            None => &mut no_proposer,
                        };
                        let run = generate_speculative_with(
                            stepper,
                            proposer,
                            SpeculativePrompt::Prefilled {
                                cache: &mut cache,
                                logits,
                                hidden: speculative.then_some(hidden),
                                history: &m.expanded_ids,
                                position_delta: *delta,
                                warm_proposer: false,
                            },
                            &config,
                            drafts,
                            &req.cancel,
                            &mut sink,
                            constraint,
                            should_stop_opt,
                            None,
                        )
                        .map_err(|e| self.request_error(e, prompt_len, req.max_new_tokens))?;
                        target
                            .device()
                            .synchronize()
                            .map_err(|e| to_core(e.into()))?;
                        phase_prefill = prefill;
                        phase_decode_started = decode_started;
                        run
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
                        let proposer: &mut dyn Proposer = match mtp_proposer.as_mut() {
                            Some(proposer) => proposer,
                            None => &mut no_proposer,
                        };
                        let run = generate_speculative_with(
                            stepper,
                            proposer,
                            SpeculativePrompt::Tokens(&prompt_ids),
                            &config,
                            drafts,
                            &req.cancel,
                            &mut sink,
                            constraint,
                            should_stop_opt,
                            Some(&mut boundary),
                        )
                        .map_err(|e| self.request_error(e, prompt_len, req.max_new_tokens))?;
                        target
                            .device()
                            .synchronize()
                            .map_err(|e| to_core(e.into()))?;
                        phase_prefill = prefill.unwrap_or_else(|| generation_started.elapsed());
                        phase_decode_started =
                            decode_started.unwrap_or_else(std::time::Instant::now);
                        run
                    }
                };
                if speculative {
                    mtp_stats = Some(MtpStats {
                        proposed_tokens: u32::try_from(run.stats.proposed).unwrap_or(u32::MAX),
                        accepted_tokens: u32::try_from(run.stats.accepted).unwrap_or(u32::MAX),
                        target_forwards: u32::try_from(run.stats.forwards).unwrap_or(u32::MAX),
                    });
                }
                engine_record = Some(run.record);
                run.output
            } else if let (DecodeRoute::Engine { drafts }, Decoder::Causal(model)) =
                (route, &self.model)
            {
                // The llama family (text, and the Gemma 4 soft-token splice) through the unified
                // engine over the step seam, on the static KV cache (sc-24138). The family has no
                // MTP head, so the resolved plan is `Off` and the proposer is none; admission above
                // priced the static preallocation (the widest layer's geometry, E6) and, with the
                // runner on, its graph workspace (sc-24140). The reference `Decode` loop stays
                // selectable (`set_decode_path`) as the oracle.
                if drafts > 0 {
                    return Err(CoreError::Load(
                        "MTP was advertised for a non-Qwen target decoder".into(),
                    ));
                }
                let constraint = json_mask
                    .as_mut()
                    .map(|m| m as &mut dyn RewindableConstraintMask);
                // The CUDA-graph runner (sc-24134) wraps the model when the switch is on, as on
                // the MTP path: the family declares its steps uncapturable
                // (`positions_host_scalar`), so every step runs eager and the record names why.
                let graphs = GraphRunner::new(model);
                let stepper: &dyn StepModel<Cache = StepKvCache> = if cuda_graphs_enabled() {
                    &graphs
                } else {
                    model
                };
                let run = match &g4 {
                    Some(m) => {
                        // Gemma 4 multimodal: the spliced embeddings prefill the step cache on
                        // ordinary causal 1-D positions (no M-RoPE, so no position shift), then the
                        // continuation decodes through the engine. The expanded prompt (soft-token
                        // spans included) is only known here, so its static preallocation is
                        // admitted here, before anything is allocated.
                        let capacity = m.expanded_ids.len().saturating_add(config.max_new_tokens);
                        let required =
                            spliced_prompt_bytes(&self.model, capacity, cuda_graphs_enabled())
                                .ok_or_else(|| {
                                    CoreError::InvalidRequest(
                                        "request memory estimate overflow".into(),
                                    )
                                })?;
                        core_llm::admit_request_memory_with_geometry(
                            m.expanded_ids.len(),
                            req.max_new_tokens,
                            self.descriptor.capabilities.max_context_tokens,
                            required,
                            request_available_memory(model.device())?,
                        )?;
                        let mut cache = model
                            .new_cache_for(capacity, 0)
                            .map_err(|e| self.request_error(e, prompt_len, req.max_new_tokens))?;
                        let logits = model
                            .step_prefill_from_embeds(&m.embeds, &mut cache)
                            .map_err(to_core)?;
                        model
                            .device()
                            .synchronize()
                            .map_err(|e| to_core(e.into()))?;
                        let prefill = generation_started.elapsed();
                        let decode_started = std::time::Instant::now();
                        let run = generate_speculative_with(
                            stepper,
                            &mut NoProposer,
                            SpeculativePrompt::Prefilled {
                                cache: &mut cache,
                                logits,
                                hidden: None,
                                history: &m.expanded_ids,
                                position_delta: 0,
                                warm_proposer: true,
                            },
                            &config,
                            0,
                            &req.cancel,
                            &mut sink,
                            constraint,
                            should_stop_opt,
                            None,
                        )
                        .map_err(|e| self.request_error(e, prompt_len, req.max_new_tokens))?;
                        model
                            .device()
                            .synchronize()
                            .map_err(|e| to_core(e.into()))?;
                        phase_prefill = prefill;
                        phase_decode_started = decode_started;
                        run
                    }
                    None => {
                        model
                            .device()
                            .synchronize()
                            .map_err(|e| to_core(e.into()))?;
                        let mut prefill = None;
                        let mut decode_started = None;
                        let mut boundary = || -> crate::error::Result<()> {
                            model.device().synchronize()?;
                            prefill = Some(generation_started.elapsed());
                            decode_started = Some(std::time::Instant::now());
                            Ok(())
                        };
                        let run = generate_speculative_with(
                            stepper,
                            &mut NoProposer,
                            SpeculativePrompt::Tokens(&prompt_ids),
                            &config,
                            0,
                            &req.cancel,
                            &mut sink,
                            constraint,
                            should_stop_opt,
                            Some(&mut boundary),
                        )
                        .map_err(|e| self.request_error(e, prompt_len, req.max_new_tokens))?;
                        model
                            .device()
                            .synchronize()
                            .map_err(|e| to_core(e.into()))?;
                        phase_prefill = prefill.unwrap_or_else(|| generation_started.elapsed());
                        phase_decode_started =
                            decode_started.unwrap_or_else(std::time::Instant::now);
                        run
                    }
                };
                engine_record = Some(run.record);
                run.output
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

        let span_counters = request_span.counters();
        // The engine's record is measured by the engine itself (the cache it ran on, the
        // proposer, the per-verify-step syncs); the request span — which also covers a caller
        // prefill — supplies the host-side counters and every tally. Both records name the
        // proposer that ran: `none` for a request whose speculation is off, including one whose
        // `MtpMode::Auto` resolved to no proposer (AC3, sc-24130).
        let decode_record = match engine_record {
            Some(record) => record.with_request_span(&request_span),
            None => DecodeRecord::plain(
                DecodePath::Reference,
                counted.forwards() + extra_forwards,
                out.tokens.len(),
                span_counters,
            )
            .with_attn_formulation(self.model.attn_formulation())
            .with_proposer(mtp_plan.proposer())
            .with_span_tallies(&request_span)
            .with_cuda_graphs(reference_path_graphs(
                request_span.cuda_graphs(),
                cuda_graphs_enabled(),
            )),
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
            decode: Some(decode_record.report(cuda_graphs_on)),
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
        // A static KV cache's capacity bound is a request-level refusal, never an opaque backend
        // failure. A request's decode maps it with the request's geometry
        // ([`LlamaProvider::request_error`]: `RequestResourceExhausted`); without one it is still
        // the request's fault.
        e @ crate::Error::KvCapacityExceeded { .. } => CoreError::InvalidRequest(e.to_string()),
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
        substitute_vision_placeholders, validate_context_window, video_placeholder_text, HostRule,
        JsonMask, LoadWorkingSet, EAGER_ATTN_QUERY_CHUNK_SIZE,
    };
    use super::{Decode as _, Decoder};
    use crate::decode::DecodePath;
    use crate::models::CausalLm;
    use candle_core::Tensor;
    use std::collections::HashMap;

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
        let dense = LoadWorkingSet::resident_payload(payload, 0, 0, false);
        let (host_required, device_required) =
            load_memory_requirements(payload, largest_shard, 0, true, dense).unwrap();

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
        let dense = |payload, copy| LoadWorkingSet::resident_payload(payload, 0, copy, false);
        assert!(load_memory_requirements(u64::MAX, 1, 0, true, dense(u64::MAX, 0)).is_none());
        assert!(load_memory_requirements(1, 1, u64::MAX, true, dense(1, 0)).is_none());
        assert!(
            load_memory_requirements(1, 1, 0, true, dense(u64::MAX / 4 * 3, u64::MAX / 4))
                .is_none()
        );
        let gguf = LoadWorkingSet {
            source: u64::MAX / 2,
            cast: u64::MAX / 2,
            copy: 0,
            host: HostRule::Gguf,
        };
        for cuda in [true, false] {
            assert!(
                load_memory_requirements(1, 1, 0, cuda, gguf).is_none(),
                "{cuda}"
            );
        }
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
        // (The live bytes also carry one ring slot per linear layer — the recurrent term, priced
        // separately; sc-24131.)
        let live_recurrent = model.recurrent_state_bytes(1) as u64;
        let cache = model.new_cache_for(capacity, 0).unwrap();
        assert_eq!(
            cache.memory().live_bytes as u64,
            preallocation + live_recurrent
        );
        assert_eq!(cache.kv_kind(), crate::primitives::KvCacheKind::Static);
        // A declared overshoot is part of the bound (and of the priced preallocation).
        let cache = model.new_cache_for(capacity, 3).unwrap();
        assert_eq!(cache.kv_capacity(), Some(capacity + 3));
        assert_eq!(
            cache.memory().live_bytes as u64,
            model.static_kv_bytes(capacity + 3) as u64 + live_recurrent
        );
    }

    /// A tiny dense causal decoder from a JSON config and seeded weights (llama or Gemma 4 keys).
    fn tiny_causal_from(cfg: serde_json::Value, weights: HashMap<String, Tensor>) -> CausalLm {
        let cfg = crate::config::ModelConfig::from_json(&cfg).unwrap();
        CausalLm::from_weights(
            &crate::primitives::Weights::from_map(weights, candle_core::Device::Cpu),
            "",
            cfg,
        )
        .unwrap()
    }

    fn tiny_llama_for_admission() -> CausalLm {
        let (cfg, w) = tiny_llama_parts();
        tiny_causal_from(cfg, w)
    }

    /// The tiny llama decoder's config and seeded weights (3 layers, vocab 40).
    fn tiny_llama_parts() -> (serde_json::Value, HashMap<String, Tensor>) {
        llama_parts(40, 32, 64, 4, 2, 3)
    }

    /// A seeded llama decoder of the given geometry: its config and f32 weights.
    fn llama_parts(
        vocab: usize,
        hidden: usize,
        inter: usize,
        heads: usize,
        kv_heads: usize,
        layers: usize,
    ) -> (serde_json::Value, HashMap<String, Tensor>) {
        use crate::primitives::{SplitMix64, TokenRng};
        let head_dim = hidden / heads;
        let mut rng = SplitMix64::new(0x000A_D417);
        let mut rand = |dims: &[usize]| {
            let n: usize = dims.iter().product();
            let data: Vec<f32> = (0..n).map(|_| rng.next_f32() - 0.5).collect();
            Tensor::from_vec(data, dims.to_vec(), &candle_core::Device::Cpu).unwrap()
        };
        let mut w = HashMap::new();
        w.insert(
            "model.embed_tokens.weight".to_string(),
            rand(&[vocab, hidden]),
        );
        w.insert("model.norm.weight".to_string(), rand(&[hidden]));
        w.insert("lm_head.weight".to_string(), rand(&[vocab, hidden]));
        for i in 0..layers {
            let p = |s: &str| format!("model.layers.{i}.{s}");
            w.insert(p("input_layernorm.weight"), rand(&[hidden]));
            w.insert(p("post_attention_layernorm.weight"), rand(&[hidden]));
            w.insert(
                p("self_attn.q_proj.weight"),
                rand(&[heads * head_dim, hidden]),
            );
            w.insert(
                p("self_attn.k_proj.weight"),
                rand(&[kv_heads * head_dim, hidden]),
            );
            w.insert(
                p("self_attn.v_proj.weight"),
                rand(&[kv_heads * head_dim, hidden]),
            );
            w.insert(
                p("self_attn.o_proj.weight"),
                rand(&[hidden, heads * head_dim]),
            );
            w.insert(p("mlp.gate_proj.weight"), rand(&[inter, hidden]));
            w.insert(p("mlp.up_proj.weight"), rand(&[inter, hidden]));
            w.insert(p("mlp.down_proj.weight"), rand(&[hidden, inter]));
        }
        (
            serde_json::json!({
                "architectures": ["LlamaForCausalLM"], "model_type": "llama",
                "hidden_size": hidden, "intermediate_size": inter, "num_hidden_layers": layers,
                "num_attention_heads": heads, "num_key_value_heads": kv_heads,
                "vocab_size": vocab, "rms_norm_eps": 1e-6, "rope_theta": 10000.0,
                "max_position_embeddings": 256, "tie_word_embeddings": false
            }),
            w,
        )
    }

    /// sc-24140: a llama-family load reports its weight census (the `LoadRecord` said `None` for
    /// every non-qwen3_5 architecture), and the requested format reaches `CausalLm`: dense keeps
    /// all 3 × 7 projections and the head dense; `Quantize::Q8` makes the layer projections GGML
    /// and keeps the head dense.
    #[test]
    fn a_llama_family_load_reports_its_census_for_the_requested_format() {
        use core_llm::{LoadSpec, Quantize};
        let (cfg, weights) = tiny_llama_parts();
        let dir = tempfile::Builder::new()
            .prefix("candle-llama-census-")
            .tempdir()
            .unwrap();
        std::fs::write(dir.path().join("config.json"), cfg.to_string()).unwrap();
        std::fs::write(
            dir.path().join("tokenizer.json"),
            synthetic_tokenizer_json(40),
        )
        .unwrap();
        candle_core::safetensors::save(&weights, dir.path().join("model.safetensors")).unwrap();
        let load = |quantize| {
            super::LlamaProvider::load(&LoadSpec {
                quantize,
                ..LoadSpec::dense(dir.path().display().to_string())
            })
            .expect("load the synthetic llama snapshot")
            .load_record()
        };
        let dense = load(None);
        let census = dense
            .census
            .expect("a llama-family load reports its census");
        assert_eq!(census.projections.dense.count, 3 * 7 + 1);
        assert_eq!(census.projections.ggml.count, 0);
        let q8 = load(Some(Quantize::Q8));
        assert_eq!(q8.requested, Some(Quantize::Q8));
        let census = q8.census.expect("census");
        assert_eq!(
            census.projections.ggml.count,
            3 * 7,
            "every layer projection is Q8"
        );
        assert_eq!(
            census.projections.dense.count, 1,
            "the head stays dense under GGML"
        );
        assert_eq!(census.projections.dense.params, 40 * 32);
    }

    /// The shared Gemma 4 decoder fixture: sliding layers 2 KV heads × 8, full layers
    /// `attention_k_eq_v` with 1 global KV head × 16 — equal `kv_heads × head width` per layer.
    ///
    /// With `k_eq_v` off (`k_eq_v = false`) the full layers keep the ordinary two KV heads at the
    /// global width 16 (upstream gates `num_global_key_value_heads` on the flag) and get their own
    /// `v_proj`: 2 × 16 per full layer against the sliding layers' 2 × 8, so the full layers are
    /// the widest and the scalar `num_key_value_heads × head_dim` (2 × 8) under-prices them.
    fn tiny_gemma4_for_admission_with(k_eq_v: bool) -> CausalLm {
        use crate::primitives::{SplitMix64, TokenRng};
        let g: serde_json::Value = serde_json::from_str(include_str!(
            "../../testdata/gemma4/gemma4_decoder_goldens.json"
        ))
        .unwrap();
        let mut w = HashMap::new();
        for (key, spec) in g["weights"].as_object().unwrap() {
            let shape: Vec<usize> = spec["shape"]
                .as_array()
                .unwrap()
                .iter()
                .map(|x| x.as_u64().unwrap() as usize)
                .collect();
            let data: Vec<f32> = spec["data"]
                .as_array()
                .unwrap()
                .iter()
                .map(|x| x.as_f64().unwrap() as f32)
                .collect();
            w.insert(
                key.clone(),
                Tensor::from_vec(data, shape, &candle_core::Device::Cpu).unwrap(),
            );
        }
        let mut cfg = g["config"].clone();
        if !k_eq_v {
            cfg["text_config"]["attention_k_eq_v"] = serde_json::json!(false);
            let text = &cfg["text_config"];
            let hidden = text["hidden_size"].as_u64().unwrap() as usize;
            let kv_heads = text["num_key_value_heads"].as_u64().unwrap() as usize;
            let global_head_dim = text["global_head_dim"].as_u64().unwrap() as usize;
            let mut rng = SplitMix64::new(0x6E4F_A11D);
            let mut rand = |dims: [usize; 2]| {
                let data: Vec<f32> = (0..dims[0] * dims[1])
                    .map(|_| (rng.next_f32() - 0.5) * 0.2)
                    .collect();
                Tensor::from_vec(data, dims.to_vec(), &candle_core::Device::Cpu).unwrap()
            };
            let layer_types = text["layer_types"].as_array().unwrap().clone();
            for (i, kind) in layer_types.iter().enumerate() {
                if kind.as_str() == Some("full_attention") {
                    let rows = kv_heads * global_head_dim;
                    w.insert(
                        format!("model.layers.{i}.self_attn.k_proj.weight"),
                        rand([rows, hidden]),
                    );
                    w.insert(
                        format!("model.layers.{i}.self_attn.v_proj.weight"),
                        rand([rows, hidden]),
                    );
                }
            }
        }
        tiny_causal_from(cfg, w)
    }

    /// E6 (sc-24138): the provider decodes the llama family through the engine on the step seam's
    /// static KV cache, which preallocates every caching layer's K/V for the request bound at
    /// once, and admission prices it: the causal geometry carries the one widest layer's
    /// `(kv_heads, head width)`, so `layers × kv_heads × head_dim × 2` covers every layer.
    ///
    /// * llama (uniform) and Gemma 4 with `k_eq_v` (sliding 2 × 8, full 1 × 16 — equal widths):
    ///   the KV term is **exactly** the preallocation — no inflation (the old independent maxima,
    ///   2 heads × 16, charged twice what any layer holds);
    /// * Gemma 4 without `k_eq_v` (full 2 × 16 wider than sliding 2 × 8): the full layers set the
    ///   geometry, and the scalar `num_key_value_heads × head_dim` would under-price the cache.
    ///
    /// The request is long enough (9 + 200 positions) that the KV term dominates the estimate, so
    /// `priced >= preallocation` fails if the KV geometry is lost. The provider's causal engine
    /// path runs no proposer, so it declares no verify overshoot and its request is priced on the
    /// `Off` plan for exactly `prompt + max_new_tokens` positions; the seam's own overshoot
    /// contract (`static_kv_bytes(capacity + overshoot)` exactly) is checked here too, but no
    /// provider path prices one for this family.
    #[test]
    fn causal_admission_covers_the_static_kv_preallocation() {
        use crate::decode::StepModel;
        use crate::primitives::DecodeCache;

        for (label, model, exact) in [
            ("llama", tiny_llama_for_admission(), true),
            ("gemma4 k_eq_v", tiny_gemma4_for_admission_with(true), true),
            (
                "gemma4 full wider",
                tiny_gemma4_for_admission_with(false),
                false,
            ),
        ] {
            let (prompt_tokens, max_new_tokens) = (9usize, 200u32);
            let capacity = prompt_tokens + max_new_tokens as usize;
            let preallocation = model.static_kv_bytes(capacity) as u64;
            assert!(preallocation > 0, "{label}");
            let layout = model.kv_layout();
            let widest = layout
                .layers
                .iter()
                .flatten()
                .map(|l| l.kv_heads * l.key_dim.max(l.value_dim))
                .max()
                .unwrap();
            let decoder = Decoder::Causal(model);
            let geometry = decoder.memory_geometry();
            assert_eq!(
                (geometry.kv_heads, geometry.head_dim),
                {
                    let (h, d) = layout.widest_layer();
                    (h as u64, d as u64)
                },
                "{label}: the geometry is the widest layer's"
            );
            assert_eq!(
                geometry.kv_heads * geometry.head_dim,
                widest as u64,
                "{label}: one layer's heads × width, never the maxima of two layer types"
            );
            let kv_term = (capacity as u64)
                * geometry.layers
                * geometry.kv_heads
                * geometry.head_dim
                * geometry.element_bytes
                * 2;
            if exact {
                // CPU f32 cache elements are the geometry's 4 bytes: nothing is over-charged.
                assert_eq!(
                    kv_term, preallocation,
                    "{label}: the KV term is exactly the preallocation (no inflation)"
                );
            } else {
                assert!(
                    preallocation <= kv_term,
                    "{label}: static preallocation {preallocation} exceeds the KV term {kv_term}"
                );
            }
            // What the provider charges this request (the causal engine path: no proposer).
            let engine = super::DecodeRoute::Engine { drafts: 0 };
            let priced = super::priced_request_bytes(
                &decoder,
                engine,
                prompt_tokens,
                max_new_tokens,
                0,
                false,
            )
            .unwrap();
            // With the CUDA-graph runner on, the runner wraps this engine too, so its graph
            // workspace is priced (sc-24140): one step token count (`K + 1 = 1`) over the
            // request's reach.
            let graph_term = crate::decode::graph_workspace_admission_bytes(
                &decoder.step_memory_geometry(0),
                capacity as u64,
                1,
            )
            .unwrap();
            assert!(graph_term > 0, "{label}");
            assert_eq!(
                super::priced_request_bytes(
                    &decoder,
                    engine,
                    prompt_tokens,
                    max_new_tokens,
                    0,
                    true,
                )
                .unwrap()
                    - priced,
                graph_term,
                "{label}"
            );
            let Decoder::Causal(model) = &decoder else {
                unreachable!()
            };
            // The cache the engine builds for that request (`new_cache_for(prompt + budget, 0)`)
            // is exactly the priced preallocation, and the estimate covers it.
            let cache = model.new_cache_for(capacity, 0).unwrap();
            assert_eq!(cache.kv_kind(), crate::primitives::KvCacheKind::Static);
            assert_eq!(cache.memory().live_bytes as u64, preallocation, "{label}");
            assert!(
                priced >= preallocation,
                "{label}: priced {priced} < preallocation {preallocation}"
            );
            // The seam's overshoot contract: a declared overshoot is part of the bound, and the
            // cache is exactly `static_kv_bytes(capacity + overshoot)` (the provider's causal path
            // declares none — see above).
            let cache = model.new_cache_for(capacity, 3).unwrap();
            assert_eq!(cache.kv_capacity(), Some(capacity + 3), "{label}");
            assert_eq!(
                cache.memory().live_bytes as u64,
                model.static_kv_bytes(capacity + 3) as u64,
                "{label}"
            );
        }

        // The non-`k_eq_v` fixture is where the full layers are widest: the scalar geometry the
        // family read before sc-24138 would under-price the static cache.
        let gemma4 = tiny_gemma4_for_admission_with(false);
        let cfg = gemma4.config();
        let scalar = (cfg.num_kv_heads * cfg.head_dim) as u64;
        let (h, d) = gemma4.kv_layout().widest_layer();
        assert_eq!((h, d), (2, 16), "the full layers (2 × 16) are the widest");
        let capacity = 26u64;
        let scalar_term = capacity * cfg.num_layers as u64 * scalar * 4 * 2;
        assert!(
            scalar_term < gemma4.static_kv_bytes(capacity as usize) as u64,
            "the scalar geometry under-prices the full layers"
        );
    }

    /// sc-24140 (E6): the Gemma 4 soft-token splice re-admits its expanded prompt with the
    /// static preallocation and, when the CUDA-graph runner wraps the engine, the runner's graph
    /// workspace over the same reach (one step token count: no proposer).
    #[test]
    fn the_spliced_prompt_readmission_prices_the_graph_workspace_when_the_runner_is_on() {
        let decoder = Decoder::Causal(tiny_gemma4_for_admission_with(false));
        let capacity = 40usize;
        let kv = decoder.static_kv_bytes(capacity) as u64;
        assert!(kv > 0);
        assert_eq!(
            super::spliced_prompt_bytes(&decoder, capacity, false),
            Some(kv)
        );
        let graphs = crate::decode::graph_workspace_admission_bytes(
            &decoder.step_memory_geometry(0),
            capacity as u64,
            1,
        )
        .unwrap();
        assert!(graphs > 0);
        assert_eq!(
            super::spliced_prompt_bytes(&decoder, capacity, true),
            Some(kv + graphs)
        );
    }

    fn nvfp4_spec(source: &str) -> core_llm::LoadSpec {
        core_llm::LoadSpec {
            source: source.into(),
            projector_source: None,
            quantize: Some(core_llm::Quantize::Nvfp4),
            cuda_graphs: None,
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
        let Ok(device) = crate::device::new_cuda_for_test() else {
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

    /// sc-24140: the family rule `nvfp4_format` applies before admission and the device gate. The
    /// qwen3_5 hybrid **and** the llama family (a `CausalLm` — Qwen3-8B's `qwen3`, Llama/Mistral,
    /// Gemma, …) reach the gate; a Prism/Bonsai snapshot (already packed affine-2) is refused by
    /// name before it, so no memory or device error can mask the refusal. The refusing gate below
    /// panics if reached.
    #[test]
    fn nvfp4_family_rule_serves_qwen35_and_the_llama_family_and_refuses_prism() {
        let root = tempfile::tempdir().unwrap();
        let spec = nvfp4_spec(&root.path().to_string_lossy());
        for config in [
            r#"{"architectures":["Qwen3_5ForConditionalGeneration"],"model_type":"qwen3_5"}"#,
            r#"{"architectures":["Qwen3ForCausalLM"],"model_type":"qwen3"}"#,
            r#"{"architectures":["LlamaForCausalLM"],"model_type":"llama"}"#,
            r#"{"architectures":["MistralForCausalLM"],"model_type":"mistral"}"#,
            r#"{"architectures":["Gemma2ForCausalLM"],"model_type":"gemma2"}"#,
            r#"{"architectures":["DeepseekV2ForCausalLM"],"model_type":"deepseek_v2"}"#,
            r#"{"architectures":["Qwen3VLForConditionalGeneration"],"model_type":"qwen3_vl"}"#,
        ] {
            std::fs::write(root.path().join("config.json"), config).unwrap();
            let reached = std::cell::Cell::new(false);
            let gate = |_: &candle_core::Device| -> crate::Result<_> {
                reached.set(true);
                Err(crate::Error::Unsupported("nvfp4: gate reached".into()))
            };
            match super::nvfp4_format_with(&spec, &candle_core::Device::Cpu, gate) {
                Err(core_llm::Error::Unsupported(msg)) => assert_eq!(msg, "nvfp4: gate reached"),
                other => panic!("{config}: expected the gate's refusal, got {other:?}"),
            }
            assert!(reached.get(), "{config} must reach the device gate");
        }

        std::fs::write(
            root.path().join("config.json"),
            r#"{"architectures":["Qwen3_5ForConditionalGeneration"],"model_type":"prism_hadamard_qwen35"}"#,
        )
        .unwrap();
        let gate = |_: &candle_core::Device| -> crate::Result<_> {
            panic!("the device gate ran before the family refusal")
        };
        match super::nvfp4_format_with(&spec, &candle_core::Device::Cpu, gate) {
            Err(core_llm::Error::Unsupported(msg)) => {
                assert!(msg.starts_with("nvfp4: "), "{msg}");
                assert!(msg.contains("Prism/Bonsai"), "names the family: {msg}");
                assert!(msg.contains("dense snapshot"), "names the reason: {msg}");
            }
            other => panic!("expected the family refusal, got {other:?}"),
        }
    }

    /// sc-24140 feature-end review: a packed MLX-affine snapshot — a `quantization` block over
    /// `<stem>.weight` + `<stem>.scales` — is refused by the model gate, from the index's
    /// `weight_map` or the shard headers (never tensor data), on the llama family and the hybrid
    /// alike; the loader would otherwise refuse its triples only after reading every weight. A
    /// prepared Q4/Q8 snapshot (the block over dense weights), a triple without the block and an
    /// orphan `.scales` are not packed. The device gate below panics if the refusal misses it.
    #[test]
    fn nvfp4_gate_refuses_a_packed_mlx_affine_snapshot_before_the_device_gate() {
        let llama =
            serde_json::json!({"architectures": ["LlamaForCausalLM"], "model_type": "llama"});
        let block = serde_json::json!({"group_size": 64, "bits": 4});
        let with_block = |mut config: serde_json::Value| {
            config["quantization"] = block.clone();
            config
        };
        let triple = [
            "model.layers.0.mlp.up_proj.weight",
            "model.layers.0.mlp.up_proj.scales",
            "model.layers.0.mlp.up_proj.biases",
        ];
        let refusal = |config: &serde_json::Value, names: &[&str], index: bool| {
            let root = tempfile::tempdir().unwrap();
            std::fs::write(root.path().join("config.json"), config.to_string()).unwrap();
            super::write_test_snapshot_tensors(root.path(), names, index);
            let gate = |_: &candle_core::Device| -> crate::Result<_> {
                Err(crate::Error::Unsupported("nvfp4: gate reached".into()))
            };
            let spec = nvfp4_spec(&root.path().to_string_lossy());
            match super::nvfp4_format_with(&spec, &candle_core::Device::Cpu, gate) {
                Err(core_llm::Error::Unsupported(msg)) => msg,
                other => panic!("{config}: expected an Unsupported refusal, got {other:?}"),
            }
        };
        let hybrid = serde_json::json!({
            "architectures": ["Qwen3_5ForConditionalGeneration"],
            "model_type": "qwen3_5",
            "text_config": {"quantization": block.clone()}
        });
        for (config, index) in [
            (with_block(llama.clone()), true),
            (with_block(llama.clone()), false),
            (hybrid, false),
        ] {
            let msg = refusal(&config, &triple, index);
            assert!(msg.starts_with("nvfp4: "), "{msg}");
            assert!(
                msg.contains("packed MLX-affine"),
                "{config} (index {index}): {msg}"
            );
            assert!(msg.contains("model.layers.0.mlp.up_proj.scales"), "{msg}");
            assert!(msg.contains("dense snapshot"), "{msg}");
            assert!(!msg.contains("  "), "{msg:?}");
        }
        // Not packed: each reaches the device gate.
        let dense = [
            "model.layers.0.mlp.up_proj.weight",
            "model.layers.0.mlp.down_proj.weight",
        ];
        for (config, names) in [
            (with_block(llama.clone()), &dense[..]),
            (llama.clone(), &triple[..]),
            (
                with_block(llama.clone()),
                &[
                    "model.layers.0.mlp.up_proj.scales",
                    "model.layers.0.mlp.down_proj.weight",
                ][..],
            ),
        ] {
            for index in [true, false] {
                assert_eq!(
                    refusal(&config, names, index),
                    "nvfp4: gate reached",
                    "{config} {names:?} (index {index})"
                );
            }
        }
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

    /// Write `config` and `weights` as a one-shard snapshot — float tensors in bf16, as shipped
    /// snapshots are; packed `U32` codes as they are — beside a synthetic `vocab`-token tokenizer.
    fn write_snapshot(
        config: &serde_json::Value,
        weights: &HashMap<String, Tensor>,
        vocab: usize,
    ) -> tempfile::TempDir {
        write_snapshot_as(config, weights, vocab, |_| candle_core::DType::BF16)
    }

    /// [`write_snapshot`] with each float tensor stored in `dtype(key)`.
    fn write_snapshot_as(
        config: &serde_json::Value,
        weights: &HashMap<String, Tensor>,
        vocab: usize,
        dtype: impl Fn(&str) -> candle_core::DType,
    ) -> tempfile::TempDir {
        use candle_core::DType;
        let dir = tempfile::Builder::new()
            .prefix("candle-admission-")
            .tempdir()
            .unwrap();
        std::fs::write(dir.path().join("config.json"), config.to_string()).unwrap();
        std::fs::write(
            dir.path().join("tokenizer.json"),
            synthetic_tokenizer_json(vocab),
        )
        .unwrap();
        let stored: HashMap<String, Tensor> = weights
            .iter()
            .map(|(key, t)| {
                let t = match t.dtype() {
                    DType::U32 => t.clone(),
                    _ => t.to_dtype(dtype(key)).unwrap(),
                };
                (key.clone(), t)
            })
            .collect();
        candle_core::safetensors::save(&stored, dir.path().join("model.safetensors")).unwrap();
        dir
    }

    fn spec_at(
        source: &std::path::Path,
        quantize: Option<core_llm::Quantize>,
    ) -> core_llm::LoadSpec {
        core_llm::LoadSpec {
            quantize,
            ..core_llm::LoadSpec::dense(source.display().to_string())
        }
    }

    /// The GGML copy a load really built, on the device: the census's GGML blocks plus candle's
    /// CUDA row padding (`MATRIX_ROW_PADDING`, 512 elements) past each tensor.
    fn ggml_copy_built(
        census: &crate::primitives::projection::WeightCensus,
        dtype: candle_core::quantized::GgmlDType,
    ) -> u64 {
        let padding = (512 * dtype.type_size() / dtype.block_size()) as u64;
        census.projections.ggml.resident_bytes + census.projections.ggml.count * padding
    }

    fn loaded_census(spec: &core_llm::LoadSpec) -> crate::primitives::projection::WeightCensus {
        super::LlamaProvider::load(spec)
            .expect("load the synthetic snapshot")
            .load_record()
            .census
            .expect("a census")
    }

    /// E6 (sc-24140): the device bound is the load's resident source, 25 percent headroom over
    /// it, and the copies the loader builds beside it; the host domain is unchanged except for a
    /// llama-family GGUF on a host device, whose dequantized f32 map *is* host memory.
    #[test]
    fn admission_prices_the_working_set_beside_the_resident_source() {
        let payload = 16_400_000_000u64; // ~Qwen3-8B bf16
        let staging = payload / 10;
        let bound =
            |cuda, working| load_memory_requirements(payload, staging, 0, cuda, working).unwrap();
        let copy = 7_400_000_000;
        let dense = LoadWorkingSet::resident_payload(payload, 0, 0, false);
        let quantized = LoadWorkingSet::resident_payload(payload, 0, copy, false);
        assert_eq!(bound(true, dense), (staging, Some(payload + payload / 4)));
        assert_eq!(
            bound(true, quantized),
            (staging, Some(payload + payload / 4 + copy)),
            "the copy is built on the device beside the source, not on the host"
        );
        assert_eq!(bound(false, quantized), (3 * payload, None));
        let prism = LoadWorkingSet::resident_payload(payload, 0, 0, true);
        assert_eq!(
            bound(true, prism),
            (2 * payload, Some(payload + payload / 4))
        );
        // An f16 snapshot: the bf16 casts sit beside the source, on the device only.
        let cast = payload;
        let f16 = LoadWorkingSet::resident_payload(payload, cast, copy, false);
        assert_eq!(
            bound(true, f16),
            (staging, Some(payload + payload / 4 + cast + copy))
        );
        assert_eq!(bound(false, f16), (3 * payload, None));

        // A llama-family GGUF: its f32 map is the source, its bf16 casts and copy sit beside it.
        let (map, cast) = (32_800_000_000u64, 2_500_000_000u64);
        let gguf = LoadWorkingSet {
            source: map,
            cast,
            copy,
            host: HostRule::Gguf,
        };
        let working = map + map / 4 + cast + copy;
        assert_eq!(bound(true, gguf), (2 * payload, Some(working)));
        assert_eq!(bound(false, gguf), (working, None));
    }

    /// E6 (sc-24140 review): a quantizing load is priced at exactly the copy its loader builds —
    /// the layer projections only, never the dense embeddings and head the whole-payload rule
    /// charged for — checked against the census of the real load for Q8_0 and Q4_K, with NVFP4's
    /// eligible shapes priced and its 40-row head (which `load_eligible` keeps dense) not.
    #[test]
    fn quantized_load_estimates_price_exactly_the_copy_the_loader_builds() {
        use crate::primitives::projection::nvfp4_device_bytes;
        use candle_core::quantized::GgmlDType;
        use core_llm::Quantize;
        let (cfg, weights) = tiny_llama_parts();
        let dir = write_snapshot(&cfg, &weights, 40);
        let estimate = |quantize, cuda| {
            super::LlamaProvider::load_memory_estimate(&spec_at(dir.path(), quantize), cuda)
                .unwrap()
        };
        let dense = estimate(None, true);
        let payload = dense.payload_bytes;
        let source_bytes: u64 = weights.values().map(|t| (t.elem_count() * 2) as u64).sum();
        assert!(payload >= source_bytes, "the file holds every bf16 source");
        assert_eq!(dense.device_required_bytes, Some(payload + payload / 4));
        assert_eq!(
            (
                dense.source_bytes,
                dense.cast_copy_bytes,
                dense.quantized_copy_bytes
            ),
            (payload, 0, 0)
        );

        let q8 = estimate(Some(Quantize::Q8), true);
        let census = loaded_census(&spec_at(dir.path(), Some(Quantize::Q8)));
        assert_eq!(
            census.projections.ggml.count, 21,
            "3 layers x 7 projections"
        );
        assert_eq!(
            q8.quantized_copy_bytes,
            ggml_copy_built(&census, GgmlDType::Q8_0),
            "exactly the Q8_0 copy the loader built"
        );
        let (wider, wider_weights) = llama_parts(80, 32, 64, 4, 2, 3);
        let wider = write_snapshot(&wider, &wider_weights, 80);
        assert_eq!(
            super::LlamaProvider::load_memory_estimate(
                &spec_at(wider.path(), Some(Quantize::Q8)),
                true
            )
            .unwrap()
            .quantized_copy_bytes,
            q8.quantized_copy_bytes,
            "the dense embedding and head are never priced: a wider vocabulary leaves the copy as is"
        );
        assert_eq!(
            q8.device_required_bytes,
            Some(payload + payload / 4 + q8.quantized_copy_bytes)
        );
        assert_eq!(q8.host_required_bytes, dense.host_required_bytes);
        assert!(
            source_bytes + q8.quantized_copy_bytes > dense.device_required_bytes.unwrap(),
            "sources plus copy (the load's peak) exceed the dense-only bound, the pricing at R"
        );
        let off_cuda = estimate(Some(Quantize::Q8), false);
        assert_eq!(
            (
                off_cuda.device_required_bytes,
                off_cuda.quantized_copy_bytes
            ),
            (None, 0)
        );

        // NVFP4: every eligible layer projection; the 40-row head fails `N % 16` and stays dense.
        let (hidden, inter, kv) = (32, 64, 16);
        let layer: u64 = [
            (hidden, hidden),
            (kv, hidden),
            (kv, hidden),
            (hidden, hidden),
            (inter, hidden),
            (inter, hidden),
            (hidden, inter),
        ]
        .iter()
        .map(|&(rows, cols)| nvfp4_device_bytes(rows, cols).unwrap())
        .sum();
        assert_eq!(nvfp4_device_bytes(40, hidden), None);
        assert_eq!(
            estimate(Some(Quantize::Nvfp4), true).quantized_copy_bytes,
            3 * layer
        );

        // Q4_K needs 256-wide rows: the same check on a geometry it can quantize.
        let (cfg, weights) = llama_parts(64, 256, 512, 4, 2, 2);
        let dir = write_snapshot(&cfg, &weights, 64);
        let spec = spec_at(dir.path(), Some(Quantize::Q4));
        let q4 = super::LlamaProvider::load_memory_estimate(&spec, true).unwrap();
        let census = loaded_census(&spec);
        assert_eq!(census.projections.ggml.count, 14);
        assert_eq!(
            q4.quantized_copy_bytes,
            ggml_copy_built(&census, GgmlDType::Q4K)
        );
    }

    /// sc-24140 review: the decoder casts every tensor it keeps dense to bf16 (`to_dtype`, which
    /// shares a bf16 tensor's storage but copies any other float), so a CUDA load of an f16 or
    /// f32 snapshot holds a 2-byte cast of each such tensor beside its source. Admission charges
    /// it explicitly — not out of the 25 percent headroom — for every tensor the loader keeps
    /// dense, and not for a projection it consumes whole into its quantized copy (or an NVFP4
    /// shape it keeps dense, which it charges). A bf16 snapshot is charged none.
    #[test]
    fn a_non_bf16_snapshot_is_charged_the_bf16_cast_beside_its_source() {
        use crate::primitives::projection::nvfp4_device_bytes;
        use candle_core::quantized::GgmlDType;
        use candle_core::DType;
        use core_llm::Quantize;
        let (cfg, weights) = tiny_llama_parts();
        let elems = |keep: &dyn Fn(&str) -> bool| -> u64 {
            weights
                .iter()
                .filter(|(key, _)| keep(key))
                .map(|(_, t)| t.elem_count() as u64)
                .sum()
        };
        let all = elems(&|_| true);
        let projections = elems(&|key| key.contains("_proj."));
        let estimate = |dir: &tempfile::TempDir, quantize, cuda| {
            super::LlamaProvider::load_memory_estimate(&spec_at(dir.path(), quantize), cuda)
                .unwrap()
        };

        let bf16 = write_snapshot(&cfg, &weights, 40);
        assert_eq!(estimate(&bf16, None, true).cast_copy_bytes, 0);
        for dtype in [DType::F16, DType::F32] {
            let dir = write_snapshot_as(&cfg, &weights, 40, |_| dtype);
            let dense = estimate(&dir, None, true);
            let payload = dense.payload_bytes;
            assert_eq!(
                dense.cast_copy_bytes,
                2 * all,
                "{dtype:?}: every tensor, cast to bf16"
            );
            assert_eq!(
                dense.device_required_bytes,
                Some(payload + payload / 4 + 2 * all),
                "{dtype:?}"
            );
            let q8 = estimate(&dir, Some(Quantize::Q8), true);
            assert_eq!(
                q8.cast_copy_bytes,
                2 * (all - projections),
                "{dtype:?}: the quantized projections are consumed, not cast"
            );
            assert_eq!(
                q8.quantized_copy_bytes,
                estimate(&bf16, Some(Quantize::Q8), true).quantized_copy_bytes
            );
            // NVFP4 also consumes the layer projections; the 40-row head stays dense, cast.
            assert_eq!(nvfp4_device_bytes(40, 32), None);
            assert_eq!(
                estimate(&dir, Some(Quantize::Nvfp4), true).cast_copy_bytes,
                2 * (all - projections)
            );
            let host = estimate(&dir, None, false);
            assert_eq!(
                (host.cast_copy_bytes, host.host_required_bytes),
                (0, 3 * payload),
                "{dtype:?}: off CUDA the host rule covers the f32 conversion"
            );
            // The loader really keeps its projections quantized from this source.
            let census = loaded_census(&spec_at(dir.path(), Some(Quantize::Q8)));
            assert_eq!(
                q8.quantized_copy_bytes,
                ggml_copy_built(&census, GgmlDType::Q8_0)
            );
        }

        // A bf16 snapshot with f32 norms (a common export): only the norms are cast.
        let mixed = write_snapshot_as(&cfg, &weights, 40, |key| {
            if key.contains("norm") {
                DType::F32
            } else {
                DType::BF16
            }
        });
        assert_eq!(
            estimate(&mixed, None, true).cast_copy_bytes,
            2 * elems(&|key| key.contains("norm"))
        );
    }

    /// sc-24140 review: a persisted `quantization` block is priced only where the loader honours
    /// it — a llama-family snapshot re-quantizes on load, the qwen3_5 loader reads only the
    /// request and keeps the projections dense.
    #[test]
    fn a_persisted_block_is_priced_only_where_the_loader_honours_it() {
        use candle_core::quantized::GgmlDType;
        use core_llm::Quantize;
        let (mut cfg, weights) = tiny_llama_parts();
        cfg["quantization"] = serde_json::json!({ "bits": 8 });
        let dir = write_snapshot(&cfg, &weights, 40);
        let spec = spec_at(dir.path(), None);
        let estimate = super::LlamaProvider::load_memory_estimate(&spec, true).unwrap();
        let census = loaded_census(&spec);
        assert_eq!(
            census.projections.ggml.count, 21,
            "the llama loader re-quantizes"
        );
        assert_eq!(
            estimate.quantized_copy_bytes,
            ggml_copy_built(&census, GgmlDType::Q8_0)
        );

        let dir = synthetic_qwen35_snapshot_without_mtp();
        let path = dir.path().join("config.json");
        let mut config: serde_json::Value =
            serde_json::from_str(&std::fs::read_to_string(&path).unwrap()).unwrap();
        config["quantization"] = serde_json::json!({ "bits": 8 });
        config["text_config"]["quantization"] = serde_json::json!({ "bits": 8 });
        std::fs::write(&path, config.to_string()).unwrap();
        let estimate = |quantize| {
            super::LlamaProvider::load_memory_estimate(&spec_at(dir.path(), quantize), true)
                .unwrap()
        };
        assert_eq!(
            estimate(None).quantized_copy_bytes,
            0,
            "qwen3_5 ignores the persisted block"
        );
        let census = loaded_census(&spec_at(dir.path(), None));
        assert_eq!(
            census.projections.ggml.count, 0,
            "and loads dense, as priced"
        );
        assert!(
            estimate(Some(Quantize::Q8)).quantized_copy_bytes > 0,
            "an explicit request is priced"
        );
    }

    /// sc-24140 review: a GGML format over an MLX-affine triple is priced as the Q8_0 repack only
    /// where the loader performs one — an 8-bit format over an 8-bit triple, at the repacked
    /// weight's Q8_0 bytes (checked against the real load). A 4-bit triple costs nothing: the
    /// loader refuses it with its own typed error, under the persisted 4-bit format and under an
    /// explicit Q8 request alike.
    #[test]
    fn an_affine_triple_is_priced_only_where_the_loader_repacks_it() {
        use candle_core::quantized::GgmlDType;
        use candle_core::{DType, Device};
        use core_llm::Quantize;
        let (cfg, dense_weights) = tiny_llama_parts();
        let stem = "model.layers.0.self_attn.q_proj";
        // The q projection [32, 32] as an affine triple with one 32-wide group per row.
        let with_triple = |bits: usize| {
            let mut weights = dense_weights.clone();
            let codes = Tensor::zeros((32, 32 * bits / 32), DType::U32, &Device::Cpu).unwrap();
            let group = Tensor::ones((32, 1), DType::F32, &Device::Cpu).unwrap();
            weights.insert(format!("{stem}.weight"), codes);
            weights.insert(format!("{stem}.scales"), group.clone());
            weights.insert(format!("{stem}.biases"), group);
            let mut cfg = cfg.clone();
            cfg["quantization"] = serde_json::json!({ "bits": bits, "group_size": 32 });
            write_snapshot(&cfg, &weights, 40)
        };
        let without_q = {
            let mut weights = dense_weights.clone();
            weights.remove(&format!("{stem}.weight"));
            weights
        };
        let copy = |dir: &tempfile::TempDir, quantize| {
            super::LlamaProvider::load_memory_estimate(&spec_at(dir.path(), quantize), true)
                .unwrap()
                .quantized_copy_bytes
        };

        let eight = with_triple(8);
        let census = loaded_census(&spec_at(eight.path(), None));
        assert_eq!(
            census.projections.ggml.count, 21,
            "the triple repacked to Q8_0"
        );
        assert_eq!(
            copy(&eight, None),
            ggml_copy_built(&census, GgmlDType::Q8_0)
        );

        let four = with_triple(4);
        for (quantize, refusal, bits) in [
            (None, "MLX affine projection requires Q8", 4),
            (Some(Quantize::Q8), "invalid MLX affine Q8 triple", 8),
        ] {
            let mut cfg = cfg.clone();
            cfg["quantization"] = serde_json::json!({ "bits": bits, "group_size": 32 });
            let others = write_snapshot(&cfg, &without_q, 40);
            assert_eq!(
                copy(&four, quantize),
                copy(&others, quantize),
                "{quantize:?}: the 4-bit triple is not priced"
            );
            match super::LlamaProvider::load(&spec_at(four.path(), quantize)) {
                Err(error) => assert!(error.to_string().contains(refusal), "{error}"),
                Ok(_) => panic!("a 4-bit affine triple loaded under {quantize:?}"),
            }
        }
    }

    /// The synthetic qwen3_5 decoder (every projection 32 or 64 wide, so Q8_0 can quantize it)
    /// written as a snapshot: with the MTP head, or with the MoE bank instead.
    fn q8_capable_qwen35_snapshot(moe: bool) -> tempfile::TempDir {
        let mut json = crate::models::qwen35::tests::synthetic_cfg_json();
        let text = json["text_config"].as_object_mut().unwrap();
        text.insert("linear_value_head_dim".into(), serde_json::json!(8));
        if moe {
            text.insert("model_type".into(), serde_json::json!("qwen3_5_moe_text"));
            text.insert("num_experts".into(), serde_json::json!(3));
            text.insert("num_experts_per_tok".into(), serde_json::json!(2));
            text.insert("moe_intermediate_size".into(), serde_json::json!(32));
            text.insert(
                "shared_expert_intermediate_size".into(),
                serde_json::json!(32),
            );
            text.insert("mtp_num_hidden_layers".into(), serde_json::json!(0));
        } else {
            text.insert("mtp_num_hidden_layers".into(), serde_json::json!(1));
        }
        let cfg = crate::models::Qwen35Config::from_json(&json).unwrap();
        let weights = crate::models::qwen35::tests::synthetic_snapshot_weights(&cfg);
        let tensors: HashMap<String, Tensor> = weights
            .keys()
            .map(|k| (k.to_string(), weights.get(k).unwrap().clone()))
            .collect();
        let mut config = serde_json::json!({
            "architectures": ["Qwen3_5ForConditionalGeneration"],
            "model_type": "qwen3_5",
        });
        config["text_config"] = json["text_config"].clone();
        write_snapshot(&config, &tensors, cfg.vocab_size as usize)
    }

    /// E6 (sc-24140 review) on the qwen3_5 hybrid: the priced copy is exactly the one the loader
    /// builds — the Gated DeltaNet and attention projections, the MLP (or every MoE expert slice
    /// and the shared expert), the head (quantized under every format here) and the MTP head —
    /// and never the per-head `in_proj_a` / `in_proj_b`, the router or the embeddings.
    #[test]
    fn qwen35_q8_estimate_prices_exactly_the_copy_the_loader_builds() {
        use crate::primitives::projection::ggml_device_bytes;
        use candle_core::quantized::GgmlDType;
        use core_llm::Quantize;
        let estimate = |dir: &tempfile::TempDir| {
            super::LlamaProvider::load_memory_estimate(
                &spec_at(dir.path(), Some(Quantize::Q8)),
                true,
            )
            .unwrap()
            .quantized_copy_bytes
        };
        // Dense: 3 DeltaNet layers x 3 + 1 attention layer x 4 + 4 MLPs x 3 + the head + the MTP's
        // q/k/v/o, MLP and fc — against the census of the real load.
        let dense = q8_capable_qwen35_snapshot(false);
        let census = loaded_census(&spec_at(dense.path(), Some(Quantize::Q8)));
        assert_eq!(census.projections.ggml.count, 34);
        assert_eq!(estimate(&dense), ggml_copy_built(&census, GgmlDType::Q8_0));

        // MoE: the same mixers and head, and per layer every expert's gate / up (sliced from the
        // stacked `gate_up_proj`) and down, plus the shared expert — each a [32, 32] Q8_0 tensor.
        let q8 = |rows: u64, cols: u64| ggml_device_bytes(rows * cols, GgmlDType::Q8_0).unwrap();
        let linear = q8(48, 32) + q8(32, 32) + q8(32, 32);
        let attention = q8(64, 32) + 2 * q8(16, 32) + q8(32, 32);
        let ffn = (3 * 3 + 3) * q8(32, 32);
        let moe = q8_capable_qwen35_snapshot(true);
        assert_eq!(
            estimate(&moe),
            3 * linear + attention + 4 * ffn + q8(50, 32)
        );
        let census = loaded_census(&spec_at(moe.path(), Some(Quantize::Q8)));
        assert_eq!(census.projections.ggml.count, 62);
        assert_eq!(estimate(&moe), ggml_copy_built(&census, GgmlDType::Q8_0));
    }

    /// sc-24140 review: a qwen3_5 MoE snapshot loads under `Quantize::Q8` on every device and
    /// decodes. On a host device the loader hands the GGML quantizer each expert as a view
    /// narrowed out of the stacked tensors, which candle's quantizer read from the storage's
    /// start — a size-check panic at load (debug) or every expert quantized from the first
    /// expert's rows (release) — until `QuantizedLinear::quantize` compacted its source.
    #[test]
    fn a_qwen35_moe_snapshot_loads_q8_and_decodes() {
        use core_llm::{Message, Quantize, Sampling, TextLlm, TextLlmRequest};
        let dir = q8_capable_qwen35_snapshot(true);
        let provider = super::LlamaProvider::load(&spec_at(dir.path(), Some(Quantize::Q8)))
            .expect("load the MoE snapshot Q8");
        assert!(provider.is_quantized());
        let request = TextLlmRequest {
            messages: vec![Message::user("t3 t7 t11 t2")],
            sampling: Sampling::greedy(),
            max_new_tokens: 4,
            seed: Some(0),
            ..Default::default()
        };
        let out = provider.generate(&request, &mut |_| {}).expect("decode");
        assert_eq!(out.usage.generated_tokens, 4);
    }

    /// A tiny `qwen3` GGUF (2 layers, hidden 32, vocab 40): Q8_0 matrices, f32 norms, and a
    /// sibling tokenizer. Returns every tensor's element count and the layer projections'.
    fn write_tiny_gguf(path: &std::path::Path) -> (u64, Vec<u64>) {
        use crate::primitives::{SplitMix64, TokenRng};
        use candle_core::quantized::gguf_file::{self, Value as Meta};
        use candle_core::quantized::{GgmlDType, QTensor};
        let (vocab, hidden, inter, kv, layers) = (40usize, 32usize, 64usize, 16usize, 2usize);
        let mut rng = SplitMix64::new(0x0006_6F0F);
        let mut tensors: Vec<(String, QTensor)> = Vec::new();
        let (mut total, mut projections) = (0u64, Vec::new());
        let mut add = |name: String, dims: &[usize], projection: bool| {
            let n: usize = dims.iter().product();
            let data: Vec<f32> = (0..n).map(|_| rng.next_f32() - 0.5).collect();
            let t = Tensor::from_vec(data, dims.to_vec(), &candle_core::Device::Cpu).unwrap();
            let dtype = if dims.len() == 2 {
                GgmlDType::Q8_0
            } else {
                GgmlDType::F32
            };
            total += n as u64;
            if projection {
                projections.push(n as u64);
            }
            tensors.push((name, QTensor::quantize(&t, dtype).unwrap()));
        };
        add("token_embd.weight".into(), &[vocab, hidden], false);
        add("output_norm.weight".into(), &[hidden], false);
        add("output.weight".into(), &[vocab, hidden], false);
        for i in 0..layers {
            let b = |s: &str| format!("blk.{i}.{s}");
            add(b("attn_norm.weight"), &[hidden], false);
            add(b("ffn_norm.weight"), &[hidden], false);
            add(b("attn_q_norm.weight"), &[8], false);
            add(b("attn_k_norm.weight"), &[8], false);
            add(b("attn_q.weight"), &[hidden, hidden], true);
            add(b("attn_k.weight"), &[kv, hidden], true);
            add(b("attn_v.weight"), &[kv, hidden], true);
            add(b("attn_output.weight"), &[hidden, hidden], true);
            add(b("ffn_gate.weight"), &[inter, hidden], true);
            add(b("ffn_up.weight"), &[inter, hidden], true);
            add(b("ffn_down.weight"), &[hidden, inter], true);
        }
        let metadata = [
            ("general.architecture", Meta::String("qwen3".into())),
            ("qwen3.attention.head_count", Meta::U32(4)),
            ("qwen3.attention.head_count_kv", Meta::U32(2)),
            ("qwen3.attention.key_length", Meta::U32(8)),
            ("qwen3.embedding_length", Meta::U32(hidden as u32)),
            ("qwen3.block_count", Meta::U32(layers as u32)),
            ("qwen3.feed_forward_length", Meta::U32(inter as u32)),
            ("qwen3.context_length", Meta::U32(256)),
            ("qwen3.attention.layer_norm_rms_epsilon", Meta::F32(1e-6)),
            ("qwen3.rope.freq_base", Meta::F32(1e6)),
        ];
        let metadata: Vec<(&str, &Meta)> = metadata.iter().map(|(k, v)| (*k, v)).collect();
        let tensor_refs: Vec<(&str, &QTensor)> =
            tensors.iter().map(|(k, t)| (k.as_str(), t)).collect();
        gguf_file::write(
            &mut std::fs::File::create(path).unwrap(),
            &metadata,
            &tensor_refs,
        )
        .unwrap();
        std::fs::write(
            path.parent().unwrap().join("tokenizer.json"),
            synthetic_tokenizer_json(vocab),
        )
        .unwrap();
        (total, projections)
    }

    /// sc-24140 review: a llama-family GGUF load dequantizes every tensor into a dense f32 map on
    /// its device and keeps it while it builds the decoder beside it — a bf16 cast of every
    /// tensor it keeps dense (on CUDA) and, under Q8, the Q8_0 copy of the layer projections. The
    /// estimate prices that working set (the f32 map as the source, with the 25 percent headroom
    /// over it) instead of the file plus 25 percent, which is several times below it; on a host
    /// device the f32 map is host memory. The map and the copy are checked against the loader's.
    #[test]
    fn gguf_admission_prices_the_dequantized_map_and_the_copies_beside_it() {
        use candle_core::quantized::GgmlDType;
        use core_llm::Quantize;
        let dir = tempfile::Builder::new()
            .prefix("candle-gguf-admission-")
            .tempdir()
            .unwrap();
        let path = dir.path().join("tiny-Q8_0.gguf");
        let (elems, projections) = write_tiny_gguf(&path);
        let projection_elems: u64 = projections.iter().sum();
        let estimate = |quantize, cuda| {
            super::LlamaProvider::load_memory_estimate(&spec_at(&path, quantize), cuda).unwrap()
        };

        let dense = estimate(None, true);
        let payload = std::fs::metadata(&path).unwrap().len();
        assert_eq!(dense.payload_bytes, payload);
        assert_eq!(
            dense.source_bytes,
            4 * elems,
            "every tensor dequantized to f32"
        );
        assert_eq!(
            dense.cast_copy_bytes,
            2 * elems,
            "every tensor cast to bf16 beside it"
        );
        assert_eq!(dense.quantized_copy_bytes, 0);
        assert_eq!(dense.device_required_bytes, Some(5 * elems + 2 * elems));
        assert!(
            dense.device_required_bytes.unwrap() > 4 * (payload + payload / 4),
            "the file plus 25 percent, the pricing at R, is several times below the working set"
        );

        let q8 = estimate(Some(Quantize::Q8), true);
        assert_eq!(q8.source_bytes, 4 * elems);
        assert_eq!(
            q8.cast_copy_bytes,
            2 * (elems - projection_elems),
            "the embedding, head and norms stay dense"
        );
        let copy: u64 = projections.iter().map(|&n| n / 32 * 34 + 544).sum();
        assert_eq!(q8.quantized_copy_bytes, copy);
        assert_eq!(
            q8.device_required_bytes,
            Some(5 * elems + q8.cast_copy_bytes + copy)
        );
        let host = estimate(Some(Quantize::Q8), false);
        assert_eq!(host.device_required_bytes, None);
        assert_eq!(
            host.host_required_bytes,
            5 * elems + copy,
            "on a host device the f32 map and the copy are host memory"
        );

        // The loader holds exactly that f32 map, and quantizes exactly the priced projections.
        let ck = crate::gguf::GgufCheckpoint::open(&path, &candle_core::Device::Cpu).unwrap();
        let map: u64 = ck
            .weights
            .keys()
            .map(|k| {
                let t = ck.weights.get(k).unwrap();
                (t.elem_count() * t.dtype().size_in_bytes()) as u64
            })
            .sum();
        assert_eq!(map, dense.source_bytes);
        let census = loaded_census(&spec_at(&path, Some(Quantize::Q8)));
        assert_eq!(census.projections.ggml.count, projections.len() as u64);
        assert_eq!(copy, ggml_copy_built(&census, GgmlDType::Q8_0));
    }

    /// E6 (sc-24140 review): an NVFP4 load is priced at exactly the packed copy it builds — its
    /// nibbles and swizzled block scales, per eligible projection — checked against the census.
    #[cfg(feature = "cuda")]
    #[test]
    fn nvfp4_load_estimate_prices_exactly_the_copy_the_loader_builds() {
        use core_llm::Quantize;
        let Ok(device) = crate::device::new_cuda_for_test() else {
            candle_quant_kernels::skip_without_sm120("no CUDA device");
            return;
        };
        if crate::primitives::projection::ProjectionFormat::nvfp4(&device).is_err() {
            candle_quant_kernels::skip_without_sm120("CUDA device below the NVFP4 floor");
            return;
        }
        let (cfg, weights) = tiny_llama_parts();
        let dir = write_snapshot(&cfg, &weights, 40);
        let spec = spec_at(dir.path(), Some(Quantize::Nvfp4));
        let estimate = super::LlamaProvider::load_memory_estimate(&spec, true).unwrap();
        let census = loaded_census(&spec);
        assert_eq!(
            census.projections.nvfp4.count, 21,
            "the 40-row head stays dense"
        );
        assert_eq!(
            estimate.quantized_copy_bytes,
            census.projections.nvfp4.resident_bytes
        );
    }

    /// E6: admission prices exactly the recurrent state a Qwen3.5-family request's cache holds.
    /// The provider's reference/MTP cache (`Decode::make_cache`) keeps no checkpoint ring — one
    /// live state per linear layer, `1 + REFERENCE_MAX_CHECKPOINTS` states; the engine's step
    /// cache for `K` drafts keeps a per-token checkpoint ring of `K + 2` states per linear layer,
    /// and the geometry for that request charges exactly the ring's bytes — the old start-of-step
    /// checkpoint term is gone (sc-24131).
    #[test]
    fn qwen35_admission_prices_every_recurrent_state_the_cache_holds() {
        use crate::decode::{StepModel, StepRequest};
        use crate::models::qwen35::{REFERENCE_MAX_CHECKPOINTS, STEP_MAX_CHECKPOINTS};
        use crate::models::Qwen35Cache;
        use crate::primitives::nn::input_ids;

        let (_cfg, model) = crate::models::qwen35::tests::text_model();
        let one_state = model.recurrent_state_bytes(1) as u64;
        assert!(one_state > 0);
        assert_eq!(model.recurrent_state_bytes(4), 4 * one_state as usize);
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
        assert_eq!(
            held.recurrent_bytes() as u64,
            geometry.recurrent_bytes,
            "provider cache holds {} recurrent bytes, admission charges {}",
            held.recurrent_bytes(),
            geometry.recurrent_bytes
        );

        // The engine's step cache for `K` drafts: the ring is the whole recurrent term, priced
        // exactly, from creation (preallocated) through decoding (written in place).
        let Decoder::Qwen35(model) = &decoder else {
            unreachable!()
        };
        for k in 0..=5usize {
            let step_geometry = decoder.step_memory_geometry(k);
            let mut step = model.new_cache_for(16, k).unwrap();
            assert_eq!(
                step.max_checkpoints(),
                k + 1,
                "K={k}: the step start + K + 1 positions"
            );
            assert_eq!(
                step.recurrent_bytes() as u64,
                one_state * (k as u64 + 2),
                "K={k}: a ring of K + 2 states per linear layer"
            );
            assert_eq!(
                step_geometry.recurrent_bytes,
                step.recurrent_bytes() as u64,
                "K={k}: admission charges exactly the ring"
            );
            model
                .forward_step(&mut step, StepRequest::last(&[1, 7, 3]))
                .unwrap();
            for t in [42, 9, 2, 11] {
                model
                    .forward_step(&mut step, StepRequest::last(&[t]))
                    .unwrap();
            }
            assert_eq!(
                step_geometry.recurrent_bytes,
                step.recurrent_bytes() as u64,
                "K={k}: nothing grew while decoding"
            );
        }

        // The unbounded step cache keeps STEP_MAX_CHECKPOINTS positions behind the current one.
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
        assert_eq!(
            step.recurrent_bytes() as u64,
            one_state * (1 + STEP_MAX_CHECKPOINTS as u64)
        );
        assert_eq!(
            decoder
                .memory_geometry_with_checkpoints(STEP_MAX_CHECKPOINTS)
                .recurrent_bytes,
            step.recurrent_bytes() as u64
        );
    }

    /// E6 (sc-24134, sc-24140): with the CUDA-graph runner switched on, every request the runner
    /// wraps is priced for every graph it may capture — exactly [`graph_workspace_admission_bytes`]
    /// over the step geometry, the request's reach (`prompt + budget + K`) and the `K + 1` step
    /// token counts: an MTP request (`K`) and a request whose speculation is off (the engine with
    /// no proposer, `K = 0`) alike. The reference loop, which the runner never wraps, is priced as
    /// before.
    #[test]
    fn admission_includes_the_cuda_graph_workspace_when_the_runner_is_on() {
        use super::DecodeRoute;

        let (_cfg, model) = crate::models::qwen35::tests::text_model();
        let decoder = Decoder::Qwen35(model);
        let (prompt, budget, k) = (11usize, 21u32, 3u32);
        let price = |route, graphs| {
            super::priced_request_bytes(&decoder, route, prompt, budget, 0, graphs).unwrap()
        };
        let graph_term = |drafts: u32| {
            crate::decode::graph_workspace_admission_bytes(
                &decoder.step_memory_geometry(drafts as usize),
                (prompt as u64) + u64::from(budget) + u64::from(drafts),
                drafts + 1,
            )
            .unwrap()
        };
        let mtp = DecodeRoute::Engine { drafts: k };
        let off = DecodeRoute::Engine { drafts: 0 };
        assert!(graph_term(k) > graph_term(0) && graph_term(0) > 0);
        assert_eq!(price(mtp, true) - price(mtp, false), graph_term(k));
        assert_eq!(price(off, true) - price(off, false), graph_term(0));
        assert_eq!(
            price(DecodeRoute::Reference, true),
            price(DecodeRoute::Reference, false)
        );
    }

    #[test]
    fn mtp_admission_prices_the_step_cache_checkpoints_and_the_verify_overshoot() {
        use super::DecodeRoute;

        let (_cfg, model) = crate::models::qwen35::tests::text_model();
        let one_state = model.recurrent_state_bytes(1) as u64;
        let decoder = Decoder::Qwen35(model);
        let (prompt, budget) = (11usize, 21u32);
        let k = 3u32;
        let price =
            |route| super::priced_request_bytes(&decoder, route, prompt, budget, 0, false).unwrap();
        let on_bytes = price(DecodeRoute::Engine { drafts: k });
        let off_bytes = price(DecodeRoute::Engine { drafts: 0 });
        let reference_bytes = price(DecodeRoute::Reference);

        // The same request priced on the reference geometry with the width set: what admission
        // would charge if it forgot the engine's cache. The step geometry's ring term — `K + 2`
        // states per linear layer against the reference cache's one (sc-24131) — is the only
        // difference, charged once: the engine rolls back by slot selection, never by a clone.
        let on_reference_geometry = core_llm::estimate_chunked_request_bytes_with_recurrent_copies(
            prompt,
            budget,
            decoder.memory_geometry(),
            0,
            k,
            EAGER_ATTN_QUERY_CHUNK_SIZE,
            1,
        )
        .unwrap();
        let ring_term = u64::from(k + 1) * one_state;
        assert!(ring_term > 0);
        assert_eq!(
            on_bytes - on_reference_geometry,
            ring_term,
            "an Enabled request is priced on the step cache's per-token checkpoint ring, once"
        );

        // A request whose speculation is off runs the engine with no proposer (sc-24140): its
        // cache's `K = 0` ring holds two states per linear layer against the reference cache's
        // one — the only difference from the reference loop's price.
        assert_eq!(
            decoder.step_memory_geometry(0).recurrent_bytes,
            2 * one_state
        );
        assert_eq!(off_bytes - reference_bytes, one_state);

        // Against the Off request: `K` more ring states plus at least the K positions the verify
        // step writes past the budget (each `layers * kv_heads * head_dim * 2 * element` bytes).
        let geometry = decoder.step_memory_geometry(k as usize);
        assert_eq!(geometry.recurrent_bytes, u64::from(k + 2) * one_state);
        let kv_per_position =
            geometry.layers * geometry.kv_heads * geometry.head_dim * geometry.element_bytes * 2;
        assert!(
            on_bytes >= off_bytes + u64::from(k) * one_state + u64::from(k) * kv_per_position,
            "Enabled {on_bytes} vs Off {off_bytes}: ring {k} x {one_state}, overshoot {k} x {kv_per_position}"
        );
    }

    /// sc-24140: one decision picks the loop — and so the price. An MTP plan always runs the
    /// engine with its proposer; a request whose speculation is off (`Off`, or `Auto` on a
    /// checkpoint without a head) runs the engine with none — text and Qwen-VL multimodal alike on
    /// the hybrid — unless the reference loop is selected.
    #[test]
    fn a_request_decodes_on_the_engine_unless_the_reference_loop_is_selected() {
        use super::DecodeRoute;
        use core_llm::{MtpMode, MtpPlan};

        let (_dir, mut provider) = synthetic_qwen35_provider_without_mtp();
        assert_eq!(provider.decode_path(), DecodePath::StepModel, "the default");
        let auto = core_llm::resolve_mtp_plan(MtpMode::Auto, None);
        assert_eq!(auto, MtpPlan::Off);
        let mtp = MtpPlan::Mtp { draft_tokens: 3 };
        for multimodal in [false, true] {
            assert_eq!(
                provider.decode_route(auto, multimodal),
                DecodeRoute::Engine { drafts: 0 },
                "multimodal={multimodal}"
            );
            assert_eq!(
                provider.decode_route(mtp, multimodal),
                DecodeRoute::Engine { drafts: 3 }
            );
        }
        provider
            .set_decode_path(DecodePath::Reference)
            .expect("the reference loop is selectable");
        assert_eq!(provider.decode_route(auto, false), DecodeRoute::Reference);
        assert_eq!(provider.decode_route(auto, true), DecodeRoute::Reference);
        assert_eq!(
            provider.decode_route(mtp, false),
            DecodeRoute::Engine { drafts: 3 },
            "an MTP plan runs the engine whatever the selector says"
        );
        assert!(provider.set_decode_path(DecodePath::PromptLookup).is_err());
        assert_eq!(provider.decode_path(), DecodePath::Reference);
    }

    /// A tokenizer.json whose vocab is `t0..t{vocab-1}` (whitespace WordLevel), so every id of
    /// the synthetic decoder decodes to a distinct piece.
    fn synthetic_tokenizer_json(vocab: usize) -> String {
        let entries: Vec<String> = (0..vocab).map(|i| format!("\"t{i}\": {i}")).collect();
        format!(
            r#"{{
                "version": "1.0",
                "added_tokens": [],
                "normalizer": null,
                "pre_tokenizer": {{ "type": "Whitespace" }},
                "post_processor": null,
                "decoder": null,
                "model": {{ "type": "WordLevel", "vocab": {{ {} }}, "unk_token": "t0" }}
            }}"#,
            entries.join(", ")
        )
    }

    /// The synthetic Qwen3.5 decoder written as a weights-free snapshot with no `mtp.*` tensors
    /// and `mtp_num_hidden_layers = 0`.
    fn synthetic_qwen35_snapshot_without_mtp() -> tempfile::TempDir {
        let (cfg, weights, cfg_json) = crate::models::qwen35::tests::text_model_snapshot_parts();
        let dir = tempfile::Builder::new()
            .prefix("candle-qwen35-no-mtp-")
            .tempdir()
            .unwrap();
        let mut config = serde_json::json!({
            "architectures": ["Qwen3_5ForConditionalGeneration"],
            "model_type": "qwen3_5",
        });
        config["text_config"] = cfg_json["text_config"].clone();
        assert_eq!(config["text_config"]["mtp_num_hidden_layers"], 0);
        std::fs::write(dir.path().join("config.json"), config.to_string()).unwrap();
        std::fs::write(
            dir.path().join("tokenizer.json"),
            synthetic_tokenizer_json(cfg.vocab_size as usize),
        )
        .unwrap();
        let tensors: std::collections::HashMap<String, candle_core::Tensor> = weights
            .keys()
            .map(|k| (k.to_string(), weights.get(k).unwrap().clone()))
            .collect();
        assert!(tensors.keys().all(|k| !k.starts_with("mtp.")));
        candle_core::safetensors::save(&tensors, dir.path().join("model.safetensors")).unwrap();
        dir
    }

    /// [`synthetic_qwen35_snapshot_without_mtp`] loaded as a provider with the backend's default
    /// graph switch (keep the directory alive with it).
    fn synthetic_qwen35_provider_without_mtp() -> (tempfile::TempDir, super::LlamaProvider) {
        use core_llm::LoadSpec;

        let dir = synthetic_qwen35_snapshot_without_mtp();
        let provider =
            super::LlamaProvider::load(&LoadSpec::dense(dir.path().display().to_string()))
                .expect("load the synthetic Qwen3.5 snapshot");
        (dir, provider)
    }

    /// sc-24139: the load's CUDA-graph policy is settled at load and recorded, every generation
    /// runs under it (not under whatever the process switch says later), and the contract carries
    /// the decode report and the load report a product renders — the same record the evidence
    /// harness reads, never a guess.
    #[test]
    fn the_load_settles_the_graph_policy_and_the_contract_carries_the_reports() {
        use core_llm::{LoadSpec, Message, ProposerKind, Sampling, TextLlm, TextLlmRequest};

        let _process = crate::decode::graph::cuda_graphs_policy_guard(Some(false));
        let dir = synthetic_qwen35_snapshot_without_mtp();
        let source = dir.path().display().to_string();
        let request = TextLlmRequest {
            messages: vec![Message::user("t3 t7 t11 t2 t7 t11")],
            sampling: Sampling::greedy(),
            max_new_tokens: 4,
            seed: Some(0),
            ..Default::default()
        };

        // `Some(true)` wins over a process switch that is off, and the load records it.
        let on = super::LlamaProvider::load(&LoadSpec {
            cuda_graphs: Some(true),
            ..LoadSpec::dense(source.clone())
        })
        .expect("load with graphs requested");
        assert_eq!(on.load_record().cuda_graphs, Some(true));
        // The load report carries the settled switch, not a guess from the request.
        assert_eq!(on.load_report().unwrap().cuda_graphs, Some(true));
        // Flipping the process switch after the load does not change the loaded model's policy.
        crate::decode::set_cuda_graphs(Some(false));
        let out = on.generate(&request, &mut |_| {}).expect("generate");
        let report = out
            .decode
            .clone()
            .expect("the provider reports its decode path");
        assert!(report.cuda_graphs.enabled, "the load's Some(true) governs");
        assert_eq!(report, on.last_decode_record().unwrap().report(true));
        assert_eq!(report.proposer, ProposerKind::None);
        // `Auto` on a snapshot without a head: the engine with no proposer, since sc-24140.
        assert_eq!(report.path, "step_model");
        assert_eq!(
            report.target_forwards,
            u64::from(out.usage.generated_tokens)
        );

        // `None` keeps the process switch at load (off here); the report says so.
        let default = super::LlamaProvider::load(&LoadSpec::dense(source)).expect("load");
        assert_eq!(default.load_record().cuda_graphs, Some(false));
        crate::decode::set_cuda_graphs(Some(true));
        let out = default.generate(&request, &mut |_| {}).expect("generate");
        assert!(!out.decode.expect("reported").cuda_graphs.enabled);

        // The load report names the requested format and the resident projection kinds.
        let load = default
            .load_report()
            .expect("the provider reports its load");
        assert_eq!(load.requested, None);
        // `None` requested, the process switch (off) at load settled it: the report says `false`.
        assert_eq!(load.cuda_graphs, Some(false));
        assert_eq!(load.projections.len(), 1, "{:?}", load.projections);
        assert_eq!(load.projections[0].kind, "dense");
        assert!(load.projections[0].count > 0 && load.projections[0].resident_bytes > 0);
    }

    #[test]
    fn auto_mtp_on_a_qwen35_snapshot_without_a_head_decodes_normally_and_says_proposer_none() {
        // AC3 (sc-24130), weights-free: the synthetic Qwen3.5 decoder written as a snapshot with
        // no `mtp.*` tensors and `mtp_num_hidden_layers = 0`. The provider advertises no MTP,
        // an `Auto` request decodes normally and the record names the proposer that ran — `none`
        // — rather than silently downgrading; `Enabled` is refused. "Normally" is the engine with
        // no proposer on the static KV cache (sc-24140), not the reference loop: the record says
        // `step_model` / `static`, one verify step per token after the first.
        use core_llm::{Message, MtpMode, ProposerKind, Sampling, TextLlm, TextLlmRequest};

        let (_dir, mut provider) = synthetic_qwen35_provider_without_mtp();
        assert!(provider.descriptor().capabilities.mtp.is_none());

        let request = |mtp| TextLlmRequest {
            messages: vec![Message::user("t3 t7 t11 t2 t7 t11")],
            sampling: Sampling::greedy(),
            max_new_tokens: 6,
            seed: Some(0),
            mtp,
            ..Default::default()
        };
        let out = provider
            .generate(&request(MtpMode::Auto), &mut |_| {})
            .expect("Auto decodes normally without a head");
        assert_eq!(out.usage.generated_tokens, 6);
        assert!(out.mtp.is_none());
        let record = provider.last_decode_record().unwrap();
        assert_eq!(record.proposer, ProposerKind::None);
        assert_eq!(record.proposer.label(), "none");
        assert_eq!(record.path, DecodePath::StepModel, "the engine ran");
        assert_eq!(record.kv_cache, crate::primitives::KvCacheKind::Static);
        assert_eq!(record.proposed_tokens, 0);
        assert_eq!(
            (
                record.target_forwards,
                record.prefill_forwards,
                record.verify_steps
            ),
            (6, 1, 5),
            "the prefill plus one single-token verify step per token after the first"
        );
        assert_eq!(record.replay_forwards, 0);
        // The same tokens as an explicit Off request: Auto is not a different decode.
        let off = provider
            .generate(&request(MtpMode::Off), &mut |_| {})
            .unwrap();
        assert_eq!(off.text, out.text);
        assert_eq!(
            provider.last_decode_record().unwrap().path,
            DecodePath::StepModel
        );
        assert!(matches!(
            provider.generate(&request(MtpMode::Enabled { draft_tokens: 2 }), &mut |_| {}),
            Err(core_llm::Error::Unsupported(_))
        ));

        // The reference loop stays selectable as the oracle; it names itself, and on CPU f32
        // (where the two caches are bit-identical) it is the same decode.
        provider
            .set_decode_path(DecodePath::Reference)
            .expect("the reference loop is selectable");
        let reference = provider
            .generate(&request(MtpMode::Auto), &mut |_| {})
            .unwrap();
        let record = provider.last_decode_record().unwrap();
        assert_eq!(record.path, DecodePath::Reference);
        assert_eq!(record.kv_cache, crate::primitives::KvCacheKind::Growing);
        assert_eq!(record.proposer, ProposerKind::None);
        if provider.model.device().is_cpu() {
            assert_eq!(reference.text, out.text, "the same tokens on both loops");
        }
    }

    /// sc-24134, sc-24139, sc-24140: with the CUDA-graph switch on — the one the provider was
    /// loaded under (`LoadSpec::cuda_graphs`; the stream is settled at load) — a request whose
    /// speculation is off runs through the runner (the engine with no proposer) and its record
    /// says what the runner did with each step; a request decoded on the reference loop never runs
    /// through the runner, and its record says why rather than a bare `graph: none`. With the
    /// switch off neither record shows the runner.
    #[test]
    fn the_runner_wraps_the_off_path_and_a_reference_record_names_why_it_did_not_run() {
        use core_llm::{LoadSpec, Message, MtpMode, Sampling, TextLlm, TextLlmRequest};

        let dir = synthetic_qwen35_snapshot_without_mtp();
        let request = TextLlmRequest {
            messages: vec![Message::user("t3 t7 t11 t2")],
            sampling: Sampling::greedy(),
            max_new_tokens: 3,
            seed: Some(0),
            mtp: MtpMode::Off,
            ..Default::default()
        };
        let load = |on: bool| {
            super::LlamaProvider::load(&LoadSpec {
                cuda_graphs: Some(on),
                ..LoadSpec::dense(dir.path().display().to_string())
            })
            .expect("load the synthetic Qwen3.5 snapshot")
        };
        let run = |provider: &super::LlamaProvider| {
            provider.generate(&request, &mut |_| {}).unwrap();
            provider.last_decode_record().unwrap()
        };

        let (mut on, mut off) = (load(true), load(false));
        let engine = run(&on);
        assert_eq!(engine.path, DecodePath::StepModel);
        assert_eq!(
            engine.cuda_graphs.eager + engine.cuda_graphs.replayed,
            engine.target_forwards,
            "every step of the Off request went through the runner: {}",
            engine.cuda_graphs.describe()
        );
        let reason = engine
            .cuda_graphs
            .fallback_reason
            .expect("an eager step names why");
        assert_ne!(reason, crate::decode::graph::REASON_REFERENCE_PATH);
        assert_eq!(run(&off).cuda_graphs.label(), "none");

        let describe = |provider: &mut super::LlamaProvider| {
            provider.set_decode_path(DecodePath::Reference).unwrap();
            let record = run(provider);
            assert_eq!(record.path, DecodePath::Reference);
            record.cuda_graphs.describe()
        };
        assert_eq!(
            describe(&mut on),
            "graph: none replayed=0 eager=0 captured=0 fallback=reference_path"
        );
        assert_eq!(
            describe(&mut off),
            "graph: none replayed=0 eager=0 captured=0"
        );
    }

    /// sc-24140 (E7): a static KV cache's capacity refusal reaches the contract as the typed
    /// `RequestResourceExhausted` — the request's geometry plus the preallocation the requested
    /// positions would take against the one the cache holds — never an opaque backend error; an
    /// unrelated engine error still maps as before, and without a request's geometry the refusal
    /// is still the request's fault (`InvalidRequest`), not the backend's.
    #[test]
    fn a_kv_capacity_refusal_is_a_typed_resource_exhaustion() {
        let (_dir, provider) = synthetic_qwen35_provider_without_mtp();
        let (requested, capacity) = (70usize, 64usize);
        let error = provider.request_error(
            crate::Error::KvCapacityExceeded {
                requested,
                capacity,
            },
            9,
            61,
        );
        let core_llm::Error::RequestResourceExhausted(evidence) = error else {
            panic!("expected RequestResourceExhausted, got {error:?}");
        };
        assert_eq!(
            evidence,
            core_llm::RequestResourceExhausted {
                prompt_tokens: 9,
                max_new_tokens: 61,
                max_context_tokens: provider.descriptor.capabilities.max_context_tokens,
                required_bytes: provider.model.static_kv_bytes(requested) as u64,
                available_bytes: provider.model.static_kv_bytes(capacity) as u64,
            }
        );
        assert!(evidence.required_bytes > evidence.available_bytes);
        assert!(matches!(
            provider.request_error(crate::Error::Canceled, 9, 61),
            core_llm::Error::Canceled
        ));
        assert!(matches!(
            super::to_core(crate::Error::KvCapacityExceeded {
                requested,
                capacity
            }),
            core_llm::Error::InvalidRequest(_)
        ));
    }

    /// sc-24140 (E0 / E2): a temperature + top-p request whose speculation is off — ChatWorks'
    /// default — decodes on the engine with no proposer and draws every token through `sample`:
    /// on CUDA the device sampler (zero logits rows copied to the host), on CPU the host reference
    /// and the record says why; every draw is recorded either way.
    #[test]
    fn a_stochastic_off_request_draws_through_the_device_sampler_where_there_is_one() {
        use core_llm::{Message, MtpMode, Sampling, TextLlm, TextLlmRequest};

        let (_dir, provider) = synthetic_qwen35_provider_without_mtp();
        let request = TextLlmRequest {
            messages: vec![Message::user("t3 t7 t11 t2 t7 t11")],
            sampling: Sampling {
                temperature: 0.8,
                top_p: 0.9,
                ..Sampling::greedy()
            },
            max_new_tokens: 8,
            seed: Some(5),
            mtp: MtpMode::Off,
            ..Default::default()
        };
        let out = provider.generate(&request, &mut |_| {}).unwrap();
        let record = provider.last_decode_record().unwrap();
        assert_eq!(record.path, DecodePath::StepModel);
        // A stop token is drawn but not emitted.
        let stopped = out.finish_reason == Some(core_llm::FinishReason::Stop);
        let draws = u64::from(out.usage.generated_tokens) + u64::from(stopped);
        assert!(draws > 0);
        assert_eq!(
            record.sampler.device_draws + record.sampler.host_draws,
            draws,
            "every draw is recorded: {:?}",
            record.sampler
        );
        if provider.model.device().is_cuda() {
            assert_eq!(record.sampler.label(), "device");
            assert_eq!(record.sampler.device_draws, draws);
            assert_eq!(
                record.sampler.logits_to_host, 0,
                "no logits row reached the host"
            );
        } else {
            assert_eq!(record.sampler.label(), "host:device_unavailable");
            assert_eq!(record.sampler.host_draws, draws);
        }
    }
}
