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
use core_llm::DraftReport;
use core_llm::{
    AudioRef, Channel, ChatTemplate, Constraint, ConstraintDecodeTable, ConstraintKind, Content,
    Error as CoreError, FinishReason as CoreFinish, GenerationTimings, ImageRef, IncrementalDetok,
    JinjaChatTemplate, JsonConstraint, Llama3Template, LlmMemoryGeometry, LoadSpec, Message,
    ModelSamplingDefaults, MtpStats, ProposerCapabilities, ProposerKind, Quantize, ReasoningEffort,
    RenderOptions, Result as CoreResult, Sampling, SpeculativePlan, SpeculativeProposer,
    StopMatcher, StreamEvent as CoreEvent, TextLlm, TextLlmCapabilities, TextLlmDescriptor,
    TextLlmOutput, TextLlmRequest, ThinkingSegmenter, Tokenizer, ToolCallSegmenter, Usage,
    VideoRef,
};
use serde_json::Value;

use crate::config::{Architecture, ModelConfig};
use crate::decode::{
    cuda_graphs_enabled, generate_from_prefill_with_stop, generate_speculative_monitored,
    graph_demoted_step_admission_bytes, graph_workspace_admission_bytes, prefill_restored,
    Boundary, ConstraintMask, CountingDecode, Decode, DecodePath, DecodeRecord, DraftModelProposer,
    FinishReason, GenerationConfig, GraphRunner, GraphTally, MtpProposer, NgramProposer,
    NoProposer, PrefixCache, PrefixStats, Proposer, RequestSpan, RewindableConstraintMask,
    SpeculativePrompt, StepModel, StreamEvent,
};
use crate::device::select_device;
use crate::error::to_core;
use crate::gguf::GgufCheckpoint;
use crate::image::{Qwen35ImageProcessor, Qwen35Preprocess};
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

// The prefix-cache reasons are the shared ones (sc-24437, E8): both backends name a bypass or a
// snapshot that was not kept in the same words.
use core_llm::{PREFIX_COPY_FAILED, PREFIX_NOT_ADMITTED, PREFIX_PAGED_NOT_SNAPSHOTTED};

// The provider split by concern (sc-27302); every item stays reachable from this module, its
// siblings and the tests through the glob imports below.
mod admission;
mod chat_template;
mod constraint;
mod descriptor;
mod generate;
mod snapshot_headers;
pub use admission::LoadMemoryEstimate;
use admission::*;
use chat_template::*;
use constraint::*;
use descriptor::*;
pub use descriptor::{
    can_load, can_load_audio, can_load_vision, eos_token_ids, provider_descriptor, MTP_MAX_DEPTH,
    PROMPT_LOOKUP_MAX_DEPTH, REGISTRATION, SPECULATIVE_MAX_DEPTH,
};
#[cfg(test)]
use generate::*;
#[cfg(test)]
pub(crate) use snapshot_headers::write_test_snapshot_tensors;
use snapshot_headers::*;

/// The prefix-cache reason for a failed [`PrefixCache::store`]: the paged backing's typed refusal
/// (`Unsupported` — the only one a snapshot raises) is named as such; anything else is a failed
/// copy.
fn prefix_store_reason(error: &crate::Error) -> &'static str {
    match error {
        crate::Error::Unsupported(_) => PREFIX_PAGED_NOT_SNAPSHOTTED,
        _ => PREFIX_COPY_FAILED,
    }
}

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
    /// [`LlamaProvider::set_decode_path`].
    Reference,
    /// The unified engine over the step seam, on the static KV cache, running `proposer` with
    /// `drafts` drafts per verify step: the resolved speculative plan's proposer and depth (MTP,
    /// prompt lookup), or no proposer and `0` — the token-at-a-time loop — for every request whose
    /// speculation is off.
    Engine {
        /// The proposer the engine runs ([`ProposerKind::None`]: none).
        proposer: ProposerKind,
        /// Draft tokens per verify step (`0`: no proposer).
        drafts: u32,
    },
}

impl DecodeRoute {
    /// The engine with no proposer — the default route of a request whose speculation is off.
    const PLAIN_ENGINE: DecodeRoute = DecodeRoute::Engine {
        proposer: ProposerKind::None,
        drafts: 0,
    };

    /// The verify step's draft width (`0` on the reference loop and the no-proposer engine).
    fn drafts(self) -> u32 {
        match self {
            DecodeRoute::Reference => 0,
            DecodeRoute::Engine { drafts, .. } => drafts,
        }
    }
}

/// The bytes admission prices for a request of `admitted_prompt` + `max_new_tokens` tokens (the
/// engine's device-positions step buffers included, [`Decoder::device_step_bytes`]): on
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
        DecodeRoute::Engine { drafts, .. } => model.step_memory_geometry(drafts as usize),
        DecodeRoute::Reference => model.memory_geometry(),
    };
    let graph_workspace = match route {
        DecodeRoute::Engine { proposer, drafts } => {
            let positions = u64::try_from(admitted_prompt)
                .ok()?
                .checked_add(u64::from(max_new_tokens))?;
            let hidden_steps = proposer == ProposerKind::Mtp;
            engine_graph_workspace(model, drafts, positions, cuda_graphs, hidden_steps)?
                .checked_add(model.device_step_bytes(drafts, positions)?)?
        }
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
/// runner is off (`cuda_graphs`). A route whose steps ask for hidden rows (`hidden_steps`: MTP)
/// also prices the plain step it takes once `auto` demotes it (sc-24446,
/// [`graph_demoted_step_admission_bytes`]) — whether a request is demoted is decided mid-request,
/// so every MTP route carries it. `None` on overflow (the caller fails closed).
fn engine_graph_workspace(
    model: &Decoder,
    drafts: u32,
    positions: u64,
    cuda_graphs: bool,
    hidden_steps: bool,
) -> Option<u64> {
    if !cuda_graphs {
        return Some(0);
    }
    let geometry = model.step_memory_geometry(drafts as usize);
    let reach = positions.checked_add(u64::from(drafts))?;
    let demoted = if hidden_steps {
        graph_demoted_step_admission_bytes(&geometry, reach)?
    } else {
        0
    };
    graph_workspace_admission_bytes(&geometry, reach, drafts.checked_add(1)?)?.checked_add(demoted)
}

/// What the Gemma 4 soft-token splice re-admits once the expanded prompt (`capacity` = expanded
/// prompt + budget) is known: the static KV preallocation for those positions plus the verify
/// overshoot of `drafts` positions (a prompt-lookup plan, sc-24433; `0` with no proposer), plus —
/// when the CUDA-graph runner wraps the engine (`cuda_graphs`) — its graph workspace over the
/// same reach (sc-24140), and the device-positions step buffers (sc-24441). `None` on overflow.
fn spliced_prompt_bytes(
    model: &Decoder,
    capacity: usize,
    drafts: u32,
    cuda_graphs: bool,
) -> Option<u64> {
    let positions = u64::try_from(capacity).ok()?;
    // The Gemma 4 splice runs prompt lookup or no proposer: no hidden-row steps.
    let graphs = engine_graph_workspace(model, drafts, positions, cuda_graphs, false)?;
    let preallocated = capacity.checked_add(usize::try_from(drafts).ok()?)?;
    u64::try_from(model.static_kv_bytes(preallocated))
        .ok()?
        .checked_add(graphs)?
        .checked_add(model.device_step_bytes(drafts, positions)?)
}

impl Decoder {
    /// Device bytes the engine's device-positions step (sc-24441) holds beyond the static KV, for
    /// a request whose prompt and budget reach `positions` positions with `drafts` drafts per
    /// verify step: the cache's staged position buffers ([`DevicePositions::BYTES`]) and one
    /// step's length-aware attention workspace — a `drafts + 1`-token step over the whole
    /// capacity, the widest head ([`candle_quant_kernels::decode_attention_workspace_bytes`]; it is
    /// freed after each layer, so one is live at a time). `0` when the decoder's caches do not
    /// stage device positions. `None` on overflow.
    ///
    /// [`DevicePositions::BYTES`]: crate::primitives::DevicePositions::BYTES
    fn device_step_bytes(&self, drafts: u32, positions: u64) -> Option<u64> {
        let (active, heads, head_dim) = match self {
            Decoder::Causal(m) => (
                m.device_positions_active(),
                m.config().num_heads,
                m.kv_layout()
                    .layers
                    .iter()
                    .flatten()
                    .map(|l| l.key_dim.max(l.value_dim))
                    .max()
                    .unwrap_or(0),
            ),
            Decoder::Qwen35(m) => (
                m.device_positions_active(),
                m.config().num_heads,
                m.config().head_dim.max(0) as usize,
            ),
        };
        if !active {
            return Some(0);
        }
        let capacity = usize::try_from(positions.checked_add(u64::from(drafts))?).ok()?;
        let queries = usize::try_from(drafts.checked_add(1)?).ok()?;
        let workspace = candle_quant_kernels::decode_attention_workspace_bytes(
            1,
            heads.max(0) as usize,
            queries,
            capacity,
            head_dim,
        );
        u64::try_from(crate::primitives::DevicePositions::BYTES.checked_add(workspace)?).ok()
    }

    /// Whether the decoder's step can be captured as a CUDA graph at all
    /// ([`StepModel::graph_support`](crate::decode::StepModel::graph_support)).
    fn graph_support(&self) -> std::result::Result<(), &'static str> {
        use crate::decode::StepModel;
        match self {
            Decoder::Causal(m) => m.graph_support(),
            Decoder::Qwen35(m) => m.graph_support(),
        }
    }

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
        // A Mixture-of-Experts step's routed dispatch (sc-24440) holds its own per-token working
        // set — `slots` expert projections, the indexed kernels' outputs and Q8_1 activation
        // copies — which the dense `3 · intermediate` term must cover: the estimators charge
        // `3 · intermediate · element_bytes` per token row for the MLP, so the MoE bytes are
        // folded in as an equivalent intermediate width when they are the larger.
        let moe_bytes = match self {
            Decoder::Causal(m) => m.moe_step_bytes_per_token(),
            Decoder::Qwen35(m) => m.moe_step_bytes_per_token(),
        };
        let intermediate = (intermediate.max(0) as u64).max(moe_bytes.div_ceil(3 * 4));
        LlmMemoryGeometry {
            query_heads: query_heads.max(0) as u64,
            kv_heads: kv_heads.max(0) as u64,
            head_dim: head_dim.max(0) as u64,
            layers: layers as u64,
            element_bytes: 4,
            score_element_bytes: 4,
            hidden_size: hidden as u64,
            intermediate_size: intermediate,
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
    /// Qwen3.5 hybrid reports its selector. Either reports `DecodeAttention` when its device
    /// positions are active and it attends un-expanded: the reference loop's cached steps then
    /// run the length-aware decode attention (sc-24441).
    fn attn_formulation(&self) -> crate::primitives::AttnFormulation {
        match self {
            Decoder::Causal(m) => m.decode_attention_formulation(m.attn_formulation()),
            Decoder::Qwen35(m) => m.decode_attention_formulation(m.attn_formulation()),
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
                .map_err(|e| to_core(e.into()))?;
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
                .map_err(|e| to_core(e.into()))?;
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
    /// The cross-turn prefix cache (sc-24437): one per loaded model, every decoder family, under
    /// the byte budget [`load`](Self::load) admitted.
    prefix: Mutex<PrefixCache>,
    /// The resident draft model for `draft_model` speculation (sc-24436), present iff the load
    /// named a compatible one ([`LoadSpec::draft_source`]); `draft_model` is advertised exactly
    /// then. A second decoder on the target's device, driven by the engine as a
    /// [`DraftModelProposer`] with its own static cache per request.
    draft: Option<ResidentDraft>,
    /// What became of the load's named draft (resident, or refused with the reason), reported in
    /// [`TextLlm::load_report`]. `None` when no draft was named.
    draft_report: Option<DraftReport>,
    /// The speculative option a request that leaves it unset runs with (E5, sc-24446): the
    /// device's defaults-table row ([`crate::device::decode_defaults`]) unless
    /// [`set_speculative_default`](LlamaProvider::set_speculative_default) overrides it.
    speculative_default: core_llm::Speculative,
    /// The loaded decoder's compressed-KV table family (sc-20683), named by the same
    /// [`core_llm::qualified_kv_model_family`] rule the MLX backend uses, so a request reaches the same
    /// policy decision on both backends. Candle has no fused compressed-domain reader, so every
    /// generation runs dense and reports why ([`LlamaProvider::kv_cache_plan`]).
    kv_family: Option<core_llm::KvModelFamily>,
}

// The partial model's tensors unwind before this earlier-declared guard. Fence
// the ACTUAL load device/stream on failure, without replacing the primary error.
struct LoadFailureFence<F: FnOnce() -> CoreResult<()>> {
    fence: Option<F>,
}
impl<F: FnOnce() -> CoreResult<()>> LoadFailureFence<F> {
    fn new(fence: F) -> Self {
        Self { fence: Some(fence) }
    }
    fn disarm(&mut self) {
        self.fence = None;
    }
}
impl<F: FnOnce() -> CoreResult<()>> Drop for LoadFailureFence<F> {
    fn drop(&mut self) {
        if let Some(fence) = self.fence.take() {
            if let Err(error) = fence() {
                eprintln!("candle-llama partial-load synchronization failed; original load error preserved, no reclamation claimed: {error}");
            }
        }
    }
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

/// The load spec of a named draft (sc-24436): its own source, text-only, at the target's
/// load-time tier and CUDA-graph policy, naming no draft itself.
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

/// What a request's `draft_model` proposer holds beside the target (E7, sc-24436): the draft's
/// own prefill of the same prompt and a static cache of the request's reach plus the `K + 1`
/// draft positions a step writes, with — for a hybrid draft — the `K + 2` retained recurrent
/// checkpoints [`DraftModelProposer`] keeps; and, when the draft's caches stage device positions
/// (the CUDA default, sc-24441), their position buffers and one step's decode-attention workspace
/// over that reach ([`Decoder::device_step_bytes`]). `None` on overflow.
fn draft_request_bytes(
    draft: &Decoder,
    admitted_prompt: usize,
    max_new_tokens: u32,
    drafts: u32,
) -> Option<u64> {
    let budget = max_new_tokens.checked_add(drafts.checked_add(1)?)?;
    // The draft's cache holds the request's reach plus the `K + 1` positions a step writes
    // (`new_cache_for(capacity, K + 1)`), and its widest step is a `K + 1`-token replay:
    // `device_step_bytes` spans `positions + K` with `K + 1` queries.
    let positions = u64::try_from(admitted_prompt)
        .ok()?
        .checked_add(u64::from(max_new_tokens))?
        .checked_add(1)?;
    core_llm::estimate_chunked_request_bytes_with_recurrent_copies(
        admitted_prompt,
        budget,
        draft.memory_geometry_with_checkpoints(usize::try_from(drafts).ok()?.checked_add(2)?),
        0,
        0,
        EAGER_ATTN_QUERY_CHUNK_SIZE,
        1,
    )?
    .checked_add(draft.device_step_bytes(drafts, positions)?)
}

/// The load telemetry of a [`LlamaProvider`] (sc-24135, epic sc-24128 E2): which weight format was
/// requested and which projection kinds the decoder actually holds, with their resident bytes.
///
/// An NVFP4 request never reports a dense projection under an NVFP4 label: the qwen3_5 family
/// loads every requested projection under `census.projections.nvfp4` or fails at load, and the
/// llama family (sc-24140) keeps a projection whose shape the FP4 GEMM cannot serve dense, counted
/// under `census.projections.dense` — this record is how a caller (and the evidence harness)
/// sees which, and reads the resident bits/param.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
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
    /// Every feature the load asked for but did not deliver, each named by the feature:
    /// `mtp_head: …`, a companion MTP head ([`LoadSpec::mtp_head_source`]) that was refused by
    /// load admission, did not match the target, or named a non-Qwen3.5 target; and, with the
    /// CUDA-graph switch on, `cuda_graphs: <graph_support reason>` for a decoder whose step the
    /// graph runner can never capture (sc-24441), known at load so a product can disable its
    /// graph toggle before the first generation. The target still loaded (epic sc-24432 E2).
    pub fallbacks: Vec<String>,
    /// The cross-turn prefix cache's byte budget the load settled (sc-24437): the requested
    /// budget clamped to the headroom admission left. `None` for a provider assembled without a
    /// load.
    pub prefix_cache_bytes: Option<u64>,
}

/// Why a load whose CUDA-graph switch is on can never capture (sc-24441, E3): the device-and-build
/// refusal (`device`, already worded `cuda_graphs: …`) first, else the decoder's own
/// [`graph_support`](crate::decode::StepModel::graph_support) refusal as `cuda_graphs: <reason>`;
/// `None` when both can capture.
fn graph_load_refusal(
    device: Option<String>,
    decoder: std::result::Result<(), &'static str>,
) -> Option<String> {
    device.or_else(|| decoder.err().map(|reason| format!("cuda_graphs: {reason}")))
}

/// The load fallback Candle names for a companion MTP head requested on a target outside the
/// Qwen3.5/3.8 family, which has no predictor path to attach it to.
pub const CANDLE_MTP_HEAD_FALLBACK: &str = core_llm::COMPANION_MTP_FAMILY_REFUSAL;

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
            fallbacks: self.fallbacks.clone(),
            prefix_cache_bytes: self.prefix_cache_bytes,
            draft: None,
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
pub(crate) fn qwen35_dense_prefix(has_key: impl Fn(&str) -> bool) -> CoreResult<&'static str> {
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

/// The compressed-KV table family of a decoder dispatched as `arch` from `config` (sc-20683): the
/// shared [`core_llm::qualified_kv_model_family`] rule — the (text) decoder config's identity, and
/// only for a measured row's exact architecture (sc-20688 review) — exactly as the MLX backend
/// names it.
fn kv_family_of(arch: Architecture, config: &Value) -> Option<core_llm::KvModelFamily> {
    core_llm::qualified_kv_model_family(arch.family(), config)
}

/// The attention geometry of a causal decoder for the shared compressed-KV plan's geometry stage
/// ([`core_llm::KvAttentionGeometry`]; the MLX backend fills the same fields from its config), so
/// a decoder outside the fused reader's geometry reports `UnsupportedGeometry` on both backends.
fn kv_attention_geometry(cfg: &crate::config::ModelConfig) -> core_llm::KvAttentionGeometry {
    core_llm::KvAttentionGeometry {
        head_dim: u64::try_from(cfg.head_dim).unwrap_or(0),
        attention_softcap: cfg.attn_logit_softcap.is_some(),
        query_pre_attn_scalar: cfg
            .query_pre_attn_scalar
            .map(|scalar| u64::try_from(scalar).unwrap_or(0)),
        latent_or_shared_kv: cfg.mla.is_some() || cfg.gemma4.is_some(),
        mixture_of_experts: cfg.moe.is_some(),
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

    /// The speculative option a request that leaves it unset runs with on this provider: its
    /// device's defaults-table default ([`core_llm::DecodeDefaults::speculative`], epic sc-24432
    /// E5) unless [`set_speculative_default`](Self::set_speculative_default) replaced it.
    pub fn speculative_default(&self) -> core_llm::Speculative {
        self.speculative_default
    }

    /// Replace the speculative option an unset request runs with on this provider (a deployment
    /// choice, e.g. a server flag); a request's own `speculative` / legacy `mtp` still wins.
    pub fn set_speculative_default(&mut self, speculative: core_llm::Speculative) {
        self.speculative_default = speculative;
    }

    /// Select the loop a request whose speculation is off decodes on — one selector for both
    /// families (sc-24138, sc-24140): [`DecodePath::StepModel`] (the default) runs the unified
    /// engine over the step seam with no proposer on the static KV cache (through the CUDA-graph
    /// runner when it is on); [`DecodePath::Reference`] runs the `Decode` loop on the growing
    /// cache — the parity oracle. Any other path is [`CoreError::InvalidRequest`]: speculation is
    /// chosen per request (`TextLlmRequest::speculative`), and a request whose speculative option
    /// resolves to a proposer (MTP, prompt lookup) always runs the engine with it, whatever this
    /// says — multimodal requests included (sc-24446).
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

    /// The cross-turn prefix cache's byte budget (sc-24437) — what the load admitted for it.
    pub fn prefix_cache_budget(&self) -> u64 {
        self.prefix_cache().budget_bytes()
    }

    /// Bytes the prefix cache holds now (never more than
    /// [`prefix_cache_budget`](Self::prefix_cache_budget)).
    pub fn prefix_cache_resident_bytes(&self) -> u64 {
        self.prefix_cache().resident_bytes()
    }

    /// The prefix cache's cumulative reuse accounting.
    pub fn prefix_cache_stats(&self) -> PrefixStats {
        self.prefix_cache().stats()
    }

    fn prefix_cache(&self) -> std::sync::MutexGuard<'_, PrefixCache> {
        self.prefix
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner())
    }

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
                    // The same date the request's own render used (none on Candle), so the
                    // conversation is a prefix of the prompt.
                    date_string: None,
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

    /// An upper bound on the bytes of the prefix-cache snapshot a text request on the engine
    /// takes (sc-24437), which its admission adds: the softmax KV copy of up to `positions`
    /// positions after the run, or the hybrid's whole cache at the conversation `boundary` — its
    /// attention KV, one recurrent state and, with `mtp`, the head's KV and a hidden row (at 4
    /// bytes an element). `None` on overflow.
    fn prefix_snapshot_bytes(
        &self,
        positions: usize,
        boundary: Option<usize>,
        mtp: bool,
    ) -> Option<u64> {
        match (&self.model, boundary) {
            (Decoder::Causal(_), _) => u64::try_from(self.model.static_kv_bytes(positions)).ok(),
            (Decoder::Qwen35(m), Some(b)) => {
                let c = m.config();
                // The resident head's layers — native, or a companion head the target's config
                // does not declare (sc-24444).
                let head_layers = self.mtp.as_ref().map_or(0, Qwen35Mtp::num_layers);
                let head = if mtp {
                    (head_layers as u64)
                        .checked_mul(c.num_kv_heads.max(0) as u64)?
                        .checked_mul(c.head_dim.max(0) as u64)?
                        .checked_mul(2 * 4)?
                        .checked_mul(b as u64)?
                        .checked_add((c.hidden_size.max(0) as u64).checked_mul(4)?)?
                } else {
                    0
                };
                u64::try_from(m.static_kv_bytes(b))
                    .ok()?
                    .checked_add(u64::try_from(m.recurrent_state_bytes(1)).ok()?)?
                    .checked_add(head)
            }
            (Decoder::Qwen35(_), None) => Some(0),
        }
    }

    /// Which loop this request decodes on, and why a resolved proposer could not run when it
    /// could not (sc-24433, epic sc-24432 E2): the engine with the plan's proposer (MTP, prompt
    /// lookup, a draft model) and depth; the engine with no proposer for a request whose
    /// speculation is off (the selector's default — both families, text and multimodal alike: the
    /// Qwen-VL DeepStack / M-RoPE prefill fills the engine's step cache, sc-24446); the reference
    /// loop only when the selector asks for it.
    fn decode_route(&self, plan: SpeculativePlan) -> (DecodeRoute, Option<String>) {
        let plain = match self.decode_path {
            DecodePath::Reference => DecodeRoute::Reference,
            _ => DecodeRoute::PLAIN_ENGINE,
        };
        match plan {
            SpeculativePlan::Off => (plain, None),
            SpeculativePlan::Run {
                proposer: SpeculativeProposer::DraftModel,
                ..
            } if self.draft.is_none() => (plain, Some(core_llm::DRAFT_MODEL_NOT_LOADED.into())),
            SpeculativePlan::Run { proposer, depth } => (
                DecodeRoute::Engine {
                    proposer: proposer.into(),
                    drafts: depth,
                },
                None,
            ),
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
        let fence_device = device.clone();
        let mut failed_load_fence = LoadFailureFence::new(move || {
            fence_device
                .synchronize()
                .map_err(|error| to_core(error.into()))
        });
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
        let budget = load_memory_budget()?;
        // A discrete CUDA launcher's override describes device headroom. Host staging is a separate
        // allocation domain and must be checked against host capacity rather than GPU capacity.
        let host_available = core_llm::effective_memory_budget(
            core_llm::available_host_memory_bytes(),
            host_load_budget(device.is_cuda(), budget),
        )?;
        core_llm::admit_load_memory(host_required, host_available)?;
        let device_available = device_required
            .map(|_| request_available_memory(&device))
            .transpose()?;
        if let (Some(device_required), Some(available)) = (device_required, device_available) {
            core_llm::admit_load_memory(device_required, available)?;
        }
        // A named draft (sc-24436, E7) is priced beside the admitted target in both domains.
        let draft = Self::admit_draft(
            spec,
            device.is_cuda(),
            (host_required, host_available),
            device_required.zip(device_available),
        )?;
        let draft_estimate = match &draft {
            Some((draft_spec, None)) => {
                // As a draft (never graph-wrapped), the price `admit_draft` admitted it at.
                Some(Self::load_memory_estimate_as(
                    draft_spec,
                    device.is_cuda(),
                    false,
                )?)
            }
            _ => None,
        };
        // What the load has admitted so far in each domain: the target plus any admitted draft.
        let mut host_admitted =
            host_required.saturating_add(draft_estimate.map_or(0, |e| e.host_required_bytes));
        let mut device_admitted = device_required.map(|required| {
            required.saturating_add(
                draft_estimate
                    .and_then(|e| e.device_required_bytes)
                    .unwrap_or(0),
            )
        });
        // E7: a companion head is resident weights; it is admitted on top of the target and any
        // admitted draft, and a head that does not fit is refused by name while the target still
        // loads (E2).
        let mut fallbacks = Vec::new();
        let mtp_head = spec.mtp_head_source.as_deref().and_then(|source| {
            admit_companion_head(
                Path::new(source),
                (host_admitted, host_available),
                device_admitted.zip(device_available),
            )
            .map_err(|reason| fallbacks.push(reason))
            .ok()
        });
        if let Some((_, head_bytes)) = mtp_head {
            host_admitted = host_admitted.saturating_add(head_bytes);
            device_admitted = device_admitted.map(|bytes| bytes.saturating_add(head_bytes));
        }
        // The cross-turn prefix cache's budget (sc-24437, E7): what the load asked for, clamped to
        // the headroom this admission leaves beside the target, any admitted draft and any
        // admitted companion head in the domain the cache lives in — the device's on CUDA, the
        // host's otherwise — so the cache can never push the load past it.
        let backend = crate::device::decode_backend(&device);
        let prefix_budget = match device_admitted.zip(device_available) {
            Some((admitted, available)) => {
                core_llm::prefix_cache_budget(backend, spec.prefix_cache_bytes, admitted, available)
            }
            None => core_llm::prefix_cache_budget(
                backend,
                spec.prefix_cache_bytes,
                host_admitted,
                host_available,
            ),
        };

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
        // The load's own fallbacks (a refused companion head) join what the decoder's load named
        // (a configured native MTP head the snapshot does not carry).
        provider.load_record.fallbacks.splice(0..0, fallbacks);
        provider.attach_mtp_head(mtp_head.map(|(head, _)| head));
        provider.load_record.prefix_cache_bytes = Some(prefix_budget);
        provider
            .load_record
            .fallbacks
            .extend(core_llm::prefix_budget_fallback(
                backend,
                spec.prefix_cache_bytes,
                prefix_budget,
            ));
        provider.prefix = Mutex::new(PrefixCache::with_budget(prefix_budget));
        match draft {
            None => {}
            Some((draft_spec, Some(refusal))) => {
                provider.draft_report = Some(
                    DraftReport::refused(draft_spec.source, refusal)
                        .named_in(&mut provider.load_record.fallbacks),
                );
            }
            Some((draft_spec, None)) => {
                provider.attach_draft(&draft_spec, &device, requested.as_ref())
            }
        }
        provider.note_graph_refusal(cuda_graphs, &device);
        failed_load_fence.disarm();
        Ok(provider)
    }

    /// Price a named draft beside the target (sc-24436, E7) in both allocation domains — `host`
    /// and, on CUDA, `device`, each `(target required, available)` — before any weight is read.
    /// `Ok(None)`: no draft named. `Ok(Some((draft spec, refusal)))`: the draft's load spec and,
    /// when it cannot be priced or does not fit beside the target, the named refusal (the target
    /// loads alone, E2). `Err` only when the target alone does not fit — the ordinary refusal.
    fn admit_draft(
        spec: &LoadSpec,
        cuda: bool,
        host: (u64, u64),
        device: Option<(u64, u64)>,
    ) -> CoreResult<Option<(LoadSpec, Option<String>)>> {
        let Some(source) = spec.draft_source.as_deref() else {
            return Ok(None);
        };
        let draft_spec = draft_load_spec(spec, source);
        let refusal = match Self::load_memory_estimate_as(&draft_spec, cuda, false) {
            Err(e) => {
                core_llm::admit_load_memory(host.0, host.1)?;
                if let Some((required, available)) = device {
                    core_llm::admit_load_memory(required, available)?;
                }
                Some(core_llm::draft_unpriced_refusal(e))
            }
            Ok(estimate) => {
                let host =
                    core_llm::admit_draft_load(host.0, estimate.host_required_bytes, host.1)?;
                let device = match (device, estimate.device_required_bytes) {
                    (Some((target, available)), Some(draft)) => {
                        core_llm::admit_draft_load(target, draft, available)?
                    }
                    (Some((target, available)), None) => {
                        core_llm::admit_load_memory(target, available)?;
                        None
                    }
                    (None, _) => None,
                };
                host.or(device)
            }
        };
        Ok(Some((draft_spec, refusal)))
    }

    /// Load the draft `spec` names on the target's `device` and keep it only if it can propose
    /// for this target ([`core_llm::draft_compatibility`]): resident — `draft_model` advertised
    /// — or refused with the reason in the load report, the target unaffected either way
    /// (sc-24436, E2). A draft that ships a `tokenizer.json` is checked before any weight is read.
    fn attach_draft(
        &mut self,
        spec: &LoadSpec,
        device: &Device,
        requested: Option<&ProjectionFormat>,
    ) {
        let source = spec.source.clone();
        let path = Path::new(&source);
        let target_logits = self.model.memory_geometry().vocab_size as usize;
        let early = core_llm::draft_tokenizer_refusal(&self.tokenizer, path).or_else(|| {
            ensure_supported_device(path, device)
                .err()
                .map(core_llm::draft_refusal)
        });
        let outcome = match early {
            Some(why) => Err(why),
            None => {
                let loaded = if crate::gguf::is_gguf_path(&source) {
                    Self::load_gguf(
                        path,
                        None,
                        device,
                        requested.and_then(ProjectionFormat::ggml),
                    )
                } else {
                    Self::load_dir(path, device, requested)
                };
                match loaded {
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
                }
            }
        };
        let (draft, report) = core_llm::settle_draft(
            source,
            outcome,
            &mut self.descriptor.capabilities,
            &mut self.load_record.fallbacks,
            recommended_depths(self.model.device()),
        );
        self.draft = draft;
        self.draft_report = Some(report);
    }

    /// What a request on `route` is admitted against before any preprocessing: the target's
    /// priced request ([`priced_request_bytes`]) and, on a `draft_model` route, the resident
    /// draft's beside it ([`draft_bytes`](Self::draft_bytes), E7). `None` on overflow.
    fn request_bytes(
        &self,
        route: DecodeRoute,
        prompt: usize,
        max_new_tokens: u32,
        vision_workspace: u64,
    ) -> Option<u64> {
        priced_request_bytes(
            &self.model,
            route,
            prompt,
            max_new_tokens,
            vision_workspace,
            cuda_graphs_enabled(),
        )?
        .checked_add(self.draft_bytes(route, prompt, max_new_tokens)?)
    }

    /// What a Gemma 4 spliced-prompt request on `route` is admitted against once its expanded
    /// prompt (`prompt` positions) is known: the target's static preallocation of the prompt, the
    /// budget and the verify overshoot ([`spliced_prompt_bytes`]) and, on a `draft_model` route,
    /// the resident draft's request over the same expanded prompt (E7). `None` on overflow.
    fn spliced_request_bytes(
        &self,
        route: DecodeRoute,
        prompt: usize,
        max_new_tokens: u32,
    ) -> Option<u64> {
        spliced_prompt_bytes(
            &self.model,
            prompt.checked_add(usize::try_from(max_new_tokens).ok()?)?,
            route.drafts(),
            cuda_graphs_enabled(),
        )?
        .checked_add(self.draft_bytes(route, prompt, max_new_tokens)?)
    }

    /// What a request on `route` holds for the resident draft beside the target (E7, sc-24436):
    /// [`draft_request_bytes`] on a `draft_model` route, `0` otherwise. `None` on overflow.
    fn draft_bytes(&self, route: DecodeRoute, prompt: usize, max_new_tokens: u32) -> Option<u64> {
        match (route, self.draft.as_ref()) {
            (
                DecodeRoute::Engine {
                    proposer: ProposerKind::DraftModel,
                    drafts,
                },
                Some(draft),
            ) => draft_request_bytes(&draft.model, prompt, max_new_tokens, drafts),
            _ => Some(0),
        }
    }

    /// The resident draft's context window (`0`: no draft, or an unbounded one).
    fn draft_context(&self) -> usize {
        self.draft.as_ref().map_or(0, |draft| draft.context)
    }

    /// The `draft_model` proposer over the resident draft for a request of at most `capacity`
    /// positions proposing `drafts` per step (sc-24436).
    fn draft_proposer(&self, capacity: usize, drafts: usize) -> CoreResult<Box<dyn Proposer + '_>> {
        match self.draft.as_ref() {
            Some(ResidentDraft {
                model: Decoder::Causal(draft),
                proposable,
                width,
                ..
            }) => Ok(Box::new(
                DraftModelProposer::new(draft, capacity, drafts).with_vocab(*proposable, *width),
            )),
            Some(ResidentDraft {
                model: Decoder::Qwen35(draft),
                proposable,
                width,
                ..
            }) => Ok(Box::new(
                DraftModelProposer::new(draft, capacity, drafts).with_vocab(*proposable, *width),
            )),
            None => Err(CoreError::Load(
                "`draft_model` was routed without a resident draft".into(),
            )),
        }
    }

    /// Attach an admitted companion MTP head (sc-24444, the Candle twin of mlx-llm's). It
    /// attaches only to a Qwen3.5/3.8-family target (dense `qwen3_5` or Prism/Bonsai, safetensors
    /// or GGUF) without a native head, and only when its geometry matches; anything else is a
    /// named load fallback and the target stays exactly as loaded.
    fn attach_mtp_head(&mut self, head: Option<&Path>) {
        let Some(head) = head else { return };
        let outcome = match (&self.model, &self.mtp) {
            (Decoder::Qwen35(_), Some(_)) => Err(format!(
                "mtp_head: {}",
                core_llm::COMPANION_MTP_ALREADY_NATIVE
            )),
            (Decoder::Qwen35(model), None) => {
                Qwen35Mtp::from_companion_dir(head, model).map_err(|e| format!("mtp_head: {e}"))
            }
            (Decoder::Causal(_), _) => Err(format!(
                "{CANDLE_MTP_HEAD_FALLBACK}; this model's family is `{}`",
                self.descriptor.family
            )),
        };
        match outcome {
            Ok(predictor) => {
                if let Some(census) = self.load_record.census.as_mut() {
                    census.merge(&predictor.weight_census());
                }
                self.mtp = Some(predictor);
                // The same backend-true depth a native head advertises (sc-24438): a companion
                // head runs the identical one-layer predictor through the identical verify path.
                let recommended = recommended_depths(self.model.device()).mtp;
                self.descriptor
                    .capabilities
                    .advertise_mtp(MTP_MAX_DEPTH, recommended);
            }
            Err(reason) => self
                .load_record
                .fallbacks
                .push(core_llm::companion_head_fallback(&reason, head)),
        }
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
    ///
    /// A CUDA load whose decoder the CUDA-graph runner wraps (its policy: `spec.cuda_graphs`,
    /// else the process switch, else the CUDA row's default) also prices the parameter cache the runner's captures leave
    /// resident for the process ([`LoadMemoryEstimate::graph_param_cache_bytes`], sc-24441) —
    /// doubled for a load with an MTP head, whose hidden-row steps are captures of their own
    /// (sc-24446).
    pub fn load_memory_estimate(spec: &LoadSpec, cuda: bool) -> CoreResult<LoadMemoryEstimate> {
        Self::load_memory_estimate_as(spec, cuda, true)
    }

    /// [`load_memory_estimate`](Self::load_memory_estimate) of `spec` as a provider's target
    /// (`target`: the graph runner wraps it under its policy) or as a resident draft (never
    /// wrapped: the draft proposer runs its forwards eagerly, so it adds nothing to the
    /// parameter cache).
    fn load_memory_estimate_as(
        spec: &LoadSpec,
        cuda: bool,
        target: bool,
    ) -> CoreResult<LoadMemoryEstimate> {
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
        let mut moe_tables = 0u64;
        // Whether the load can run steps that ask for hidden rows — an MTP head, native (a
        // Qwen3.5 snapshot configuring one) or a companion — which the graph runner captures as
        // shapes of their own (sc-24446).
        let mut hidden_steps = spec.mtp_head_source.is_some();
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
            moe_tables = decoder.moe_indexed_table_bytes();
            hidden_steps |=
                matches!(&decoder, PricedDecoder::Qwen35(cfg, _) if cfg.mtp_num_hidden_layers > 0);
            let format = decoder.format(requested);
            let builds = matches!(
                decoder,
                PricedDecoder::Causal(_) | PricedDecoder::Qwen35(..)
            );
            let (cast, copy, held) = if builds && (cuda || format.is_some()) {
                let tensors = snapshot_tensor_headers(source)?;
                let (copy, consumed) = match format {
                    Some(format) => snapshot_quantized_copy_bytes(&decoder, format, &tensors)
                        .ok_or_else(overflow)?,
                    None => (0, Default::default()),
                };
                // Off CUDA the compute dtype is f32 and the host rule's three payloads cover it.
                let (cast, held) = if cuda {
                    (
                        snapshot_cast_copy_bytes(&tensors, &consumed).ok_or_else(overflow)?,
                        stored_block_bytes(&tensors).ok_or_else(overflow)?,
                    )
                } else {
                    (0, 0)
                };
                (cast, copy, held)
            } else {
                (0, 0, 0)
            };
            LoadWorkingSet::resident_payload(payload, cast, copy, decoder.is_packed())
                .holding_on_host(held)
        };
        let (host_required, device_required) =
            load_memory_requirements(payload, staging, projector, cuda, working)
                .ok_or_else(overflow)?;
        let on_device = |bytes: u64| if cuda { bytes } else { 0 };
        let graphs = target
            && spec
                .cuda_graphs
                .unwrap_or_else(|| crate::decode::graph::cuda_graphs_default_for(cuda));
        let graph_param_cache = if graphs {
            on_device(crate::decode::graph_param_cache_load_bytes(hidden_steps))
        } else {
            0
        };
        // The indexed MoE dispatch's expert-address tables, built at a CUDA load (sc-24440).
        let moe_tables = on_device(moe_tables);
        let device_required = device_required
            .map(|bytes| {
                bytes
                    .checked_add(graph_param_cache)
                    .and_then(|b| b.checked_add(moe_tables))
                    .ok_or_else(overflow)
            })
            .transpose()?;
        Ok(LoadMemoryEstimate {
            payload_bytes: payload,
            host_required_bytes: host_required,
            device_required_bytes: device_required,
            prefix_cache_bytes: core_llm::requested_prefix_cache_bytes(
                crate::device::decode_backend_for(cuda),
                spec.prefix_cache_bytes,
            ),
            source_bytes: on_device(working.source),
            cast_copy_bytes: on_device(working.cast),
            quantized_copy_bytes: on_device(working.copy),
            graph_param_cache_bytes: graph_param_cache,
        })
    }

    /// Load from an HF snapshot directory (config.json + tokenizer.json + safetensors shards).
    ///
    /// `requested` is an explicit load-time quantization (`spec.quantize`); when it is `None` a
    /// llama-family snapshot's own persisted `quantization` block (written by the
    /// [`prepare`](crate::prepare) writer) is honored, so a `LoadSpec::dense` of a prepared Q4/Q8
    /// snapshot loads quantized. The qwen3_5 decoder honours only a stored-block (`storage:
    /// "ggml"`, sc-19375) block, so a qwen3_5 tier prepared before then keeps loading dense.
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
            Architecture::from_config(&cfg_value).map_err(|e| to_core(e.into()))?
        };
        let (weights, prism) = if is_prism {
            let checkpoint =
                crate::prism_checkpoint::PrismMlxCheckpoint::open(dir, device, &cfg_value)
                    .map_err(to_core)?;
            (checkpoint.weights, Some(checkpoint.registry))
        } else {
            (Weights::from_dir(dir, device).map_err(to_core)?, None)
        };
        // What the decoder's own load could not attach (E2): a configured native MTP head the
        // snapshot does not carry, or a variant this runtime does not run.
        let mut load_fallbacks = Vec::new();
        let (model, mut descriptor, mtp) = if arch == Architecture::Qwen35 {
            // Qwen3.6 hybrid decoder: its own config, the VLM-nested `model.language_model` prefix, and
            // a top-level untied `lm_head`.
            let qcfg = Qwen35Config::from_json(&cfg_value).map_err(to_core)?;
            let mut descriptor = descriptor_for_qwen35(&qcfg, device);
            if is_prism {
                descriptor.family = "prism_hadamard_qwen35".into();
                descriptor.capabilities.model_sampling_defaults = Some(bonsai_sampling_defaults());
            }
            // An explicit request wins; otherwise a prepared tier's stored-block format (sc-19375).
            // A bare persisted block (a tier prepared before sc-19375, dense-rounded weights) is
            // not honoured here, so such a tier keeps loading dense exactly as it did.
            let format = match (prism.is_some(), requested) {
                (false, None) => {
                    crate::prepare::persisted_ggml_blocks(&cfg_value).map(ProjectionFormat::from)
                }
                (_, requested) => requested.cloned(),
            };
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
                Qwen35Model::from_weights_format(&weights, prefix, qcfg, format.as_ref())
                    .map_err(to_core)?
            };
            // The configured native head (E2, the rule MLX loads by): built when complete; a
            // declared head the snapshot does not carry, or a variant this runtime does not
            // run, loads the target plain with the reason named; a partial `mtp.*` set still
            // fails the load (integrity).
            let cfg = m.config();
            let mtp = match core_llm::native_mtp_plan(
                cfg.mtp_num_hidden_layers,
                cfg.mtp_use_dedicated_embeddings,
                weights.keys().any(|key| key.starts_with("mtp.")),
            )? {
                core_llm::NativeMtp::Absent => None,
                core_llm::NativeMtp::Build => Some(
                    Qwen35Mtp::from_weights_format(&weights, &m, format.as_ref())
                        .map_err(to_core)?,
                ),
                core_llm::NativeMtp::Fallback(why) => {
                    load_fallbacks.push(why);
                    None
                }
            };
            (Decoder::Qwen35(m), descriptor, mtp)
        } else {
            let cfg = ModelConfig::from_dir(dir).map_err(to_core)?;
            let descriptor = descriptor_for(&cfg, device);
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
            // The backend-true depth on both the legacy and the per-proposer field (sc-24438).
            descriptor
                .capabilities
                .advertise_mtp(MTP_MAX_DEPTH, recommended_depths(device).mtp);
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
            prefix: Mutex::new(PrefixCache::with_budget(
                core_llm::requested_prefix_cache_bytes(crate::device::decode_backend(device), None),
            )),
            speculative_default: crate::device::decode_defaults(device).speculative,
            draft: None,
            draft_report: None,
            kv_family: kv_family_of(arch, &cfg_value),
            vision,
            gemma4,
            load_record: LoadRecord {
                requested: None,
                census,
                cuda_graphs: None,
                fallbacks: load_fallbacks,
                prefix_cache_bytes: None,
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
            let mut descriptor = descriptor_for_qwen35(&qcfg, device);
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
                prefix: Mutex::new(PrefixCache::with_budget(
                    core_llm::requested_prefix_cache_bytes(
                        crate::device::decode_backend(device),
                        None,
                    ),
                )),
                speculative_default: crate::device::decode_defaults(device).speculative,
                draft: None,
                draft_report: None,
                // The Prism/Bonsai hybrid has no table family.
                kv_family: None,
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
        let mut descriptor = descriptor_for(&ck.config, device);
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
            prefix: Mutex::new(PrefixCache::with_budget(
                core_llm::requested_prefix_cache_bytes(crate::device::decode_backend(device), None),
            )),
            speculative_default: crate::device::decode_defaults(device).speculative,
            draft: None,
            draft_report: None,
            // No table family: the HF identity is reconstructed from `general.architecture`, and
            // llama.cpp labels Mistral and other llama-shaped fine-tunes `"llama"` too, so a GGUF
            // cannot be told to be the Llama family the evidence measured (as `from_parts`).
            kv_family: None,
            constraint_table: OnceCell::new(),
            vision: None, // GGUF is the dense Llama-family path only — no Qwen3.6 VLM.
            // Likewise no Gemma 4 front-ends: the GGUF path reconstructs a dense text decoder,
            // and llama.cpp GGUFs carry no vision embedder / audio projector tensors.
            gemma4: None,
            load_record: LoadRecord {
                requested: None,
                census,
                cuda_graphs: None,
                fallbacks: Vec::new(),
                prefix_cache_bytes: None,
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

    /// With the CUDA-graph switch the load settled on (`cuda_graphs`), name why the graph runner
    /// can never capture on this load in its fallbacks — the load `device`'s or the build's
    /// refusal first (`cuda_graphs: not_cuda: …`, `cuda_feature_off`, `flash_attn_stream`,
    /// `legacy_stream`, `no_async_alloc`), else the decoder's own
    /// [`graph_support`](crate::decode::StepModel::graph_support) refusal (`cuda_graphs:
    /// <reason>`) — so a product knows at load, not after the first generation (sc-24441, E3).
    /// Each request still reports its eager steps and their reason. Nothing is named with the
    /// switch off or for a capturable device and decoder.
    fn note_graph_refusal(&mut self, cuda_graphs: bool, device: &Device) {
        if !cuda_graphs {
            return;
        }
        // The device and build first — a CPU / Metal device, a build without CUDA, a legacy or
        // allocator-less stream refuse every capture whatever the decoder declares — in the
        // words `backend_capabilities` uses; then the decoder's own declaration.
        self.load_record.fallbacks.extend(graph_load_refusal(
            crate::backend::cuda_graphs_device_refusal(device),
            self.model.graph_support(),
        ));
    }

    /// The load telemetry: the requested weight format and the resident weight census by
    /// projection kind (sc-24135).
    pub fn load_record(&self) -> LoadRecord {
        self.load_record.clone()
    }

    /// The loaded decoder's compressed-KV table family (sc-20683); `None` for a model the
    /// qualification table did not measure.
    pub fn kv_model_family(&self) -> Option<core_llm::KvModelFamily> {
        self.kv_family
    }

    /// Test seam (sc-20688 review): plan this provider's KV cache as `family`, as if its loaded
    /// checkpoint were that family's measured architecture, so the shared policy can be exercised
    /// on synthetic weights. Production loads name a family only for a measured architecture.
    /// Compiled only with the `test-kv-arm` feature, which this crate's own integration tests
    /// enable; a product build cannot arm a family.
    #[cfg(any(test, feature = "test-kv-arm"))]
    #[doc(hidden)]
    pub fn arm_kv_model_family_for_tests(&mut self, family: Option<core_llm::KvModelFamily>) {
        self.kv_family = family;
    }

    /// The KV cache a generation of `context_tokens` prefilled tokens and up to `max_new_tokens`
    /// more, in a `batch`-sequence decode, runs on (sc-20683) — what [`TextLlm::generate`] reports
    /// on [`TextLlmOutput::kv_cache`]. The same shared policy decision as the MLX backend
    /// ([`core_llm::plan_kv_cache`]: policy, batch, family, prompt minimum, final-context maximum,
    /// request shape, the shared geometry stage), always dense: Candle has no fused
    /// compressed-domain reader on any device, so a request MLX would run compressed reports
    /// [`core_llm::KvCacheFallbackReason::ReaderUnavailable`], and a decoder outside the fused
    /// reader's geometry reports [`core_llm::KvCacheFallbackReason::UnsupportedGeometry`] on both
    /// backends.
    pub fn kv_cache_plan(
        &self,
        policy: core_llm::KvCompressionPolicy,
        context_tokens: u64,
        max_new_tokens: u32,
        batch: u64,
        multimodal: bool,
    ) -> core_llm::KvCacheReport {
        let unsupported_request = if multimodal {
            Some("multimodal prefill splices embeddings outside the compressed cache".to_string())
        } else if matches!(self.model, Decoder::Qwen35(_)) {
            Some("the hybrid recurrent decoder has no compressed cache".to_string())
        } else {
            None
        };
        core_llm::plan_kv_cache_without_reader(
            core_llm::KvCacheRequest {
                policy,
                family: self.kv_family,
                context_tokens,
                max_new_tokens: u64::from(max_new_tokens),
                batch,
                unsupported_request,
                // The hybrid decoder is refused by the request-shape stage before geometry.
                geometry: match &self.model {
                    Decoder::Causal(model) => kv_attention_geometry(model.config()),
                    _ => core_llm::KvAttentionGeometry::default(),
                },
            },
            "the Candle backend",
        )
    }

    /// Assemble a provider from already-loaded parts with a default Llama-3 template (used by tests
    /// and converters that don't have a `tokenizer_config.json`).
    pub fn from_parts(model: CausalLm, tokenizer: Tokenizer, stop_tokens: Vec<i32>) -> Self {
        let backend = crate::device::decode_backend(model.device());
        Self {
            // No config identity, so no table family: a Llama decoder assembled from parts cannot
            // be told from a Mistral or dense Qwen2 one (as MLX's `from_parts`).
            kv_family: None,
            descriptor: provider_descriptor_on(backend),
            model: Decoder::Causal(model),
            mtp: None,
            tokenizer,
            template: Box::new(Llama3Template),
            stop_tokens,
            last_decode: Mutex::new(None),
            decode_path: DecodePath::StepModel,
            prefix: Mutex::new(PrefixCache::with_budget(
                core_llm::requested_prefix_cache_bytes(backend, None),
            )),
            speculative_default: backend.defaults().speculative,
            draft: None,
            draft_report: None,
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
                .map_err(|e| to_core(e.into()))?;
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

#[cfg(test)]
mod tests;
