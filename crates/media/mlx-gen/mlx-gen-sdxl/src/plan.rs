//! The per-request bounded-execution plan threaded through the SDXL U-Net forward (SC-15525).
//!
//! Rungs 3 and 4 of the shared memory ladder are both *request-scoped* levers on the denoise
//! forward, and both have to reach the same place — every [`Transformer2D`](crate::unet) in the
//! U-Net. Threading them as two more positional arguments through `forward` →
//! `forward_core` → `UNetBlock2D::forward_ip` → `Transformer2D::forward_ip` →
//! `TransformerBlock::forward_ip` → `AttentionMHA::forward_ip` would add two arguments at six
//! levels; one `Copy` carrier adds one.
//!
//! [`SdxlForwardPlan::default()`] is the historical unbounded path — [`AttentionPlan::UNBOUNDED`]
//! and no window — so every pre-existing `forward*` entry point keeps its exact previous behaviour
//! and its exact previous signature, and only the new `*_planned` variants can bound anything.
//!
//! ## Why the window is a SIZE and not a `BlockPlan`
//!
//! On a DiT the whole model is one flat block stack, so the plan is one
//! [`BlockPlan`](mlx_gen::block_residency::BlockPlan) built once per forward. SDXL is a U-Net: the
//! windowable blocks live in **eleven** independent `Transformer2D` sub-stacks of two different
//! depths (six of 10, five of 2 — 70 blocks in total). A single `BlockPlan` cannot describe them,
//! because `BlockPlan::new` binds `n_blocks`, so the plan carries the *cadence* and each
//! `Transformer2D` builds its own plan against its own depth. That is SC-16355's
//! "a plan per Transformer2D, not one per model", expressed in the type.

use mlx_gen::attention::AttentionPlan;
use mlx_gen::gen_core::CfgBatching;
use mlx_gen::CancelFlag;
use mlx_rs::Array;

/// Rung 4's request-scoped cadence: how many consecutive `TransformerBlock`s of **each**
/// `Transformer2D` are held materialized at once, plus the cancellation flag the window loop checks
/// at every window boundary.
///
/// The cancel handle is carried here rather than read off the request because the block loop is the
/// only cancellation point inside a denoise *step* — SDXL otherwise checks only between steps
/// (`pipeline::denoise_core`), and a windowed step is materially longer than an unwindowed one.
#[derive(Clone, Copy)]
pub struct SdxlBlockWindow<'a> {
    pub size: usize,
    pub cancel: &'a CancelFlag,
}

/// The bounded-execution levers for one denoise forward.
///
/// `Copy` and cheap (a budget, two `Option`s, a mode and a borrow), so the per-step closures in
/// `pipeline` can capture it without a lifetime fight.
#[derive(Clone, Copy)]
pub struct SdxlForwardPlan<'a> {
    /// Rung 3 — the query-row chunking budget handed to
    /// [`mlx_gen::attention::sdpa_budgeted_bhsd`] at every attention site.
    pub attention: AttentionPlan<'a>,
    /// Rung 4 — the block-window cadence, or `None` for the resident stack.
    pub window: Option<SdxlBlockWindow<'a>>,
    /// How a guided step evaluates its conditional and unconditional halves (sc-26268) — the shared
    /// [`CfgBatching`] execution domain, not a ladder rung. [`CfgBatching::Batched`] is the
    /// historical `[cond, uncond]` doubled-batch forward; [`CfgBatching::Sequential`] runs two
    /// batch-1 forwards, halving the guided denoise's activation high-water. Inert when CFG is off.
    pub cfg_batching: CfgBatching,
    /// Set only on the two halves of a **sequential** guided step (sc-26268): this forward computes
    /// row `k` of the `[cond, uncond]` pair. The latent is one row, but every conditioning-only
    /// input (text memory, pooled text, time ids, IP tokens) arrives at the pair's full batch, and
    /// each projection that reads only conditioning — the time/`text_time` embedding MLPs, every
    /// resnet's `time_emb_proj`, every cross-attention `to_k`/`to_v` (and IP `to_k_ip`/`to_v_ip`)
    /// — runs at that batch before [`select_cfg_row`](Self::select_cfg_row) takes row `k`.
    ///
    /// That is what makes sequential CFG byte-identical to batched CFG rather than merely close:
    /// MLX picks a GEMV/GEMM/`qmv`/`qmm` kernel by row count, and those kernels round differently
    /// (measured on Metal: ~1e-3 of peak for a `[1, K]` vs `[2, K]` q4/f16 product and for the
    /// 77- vs 154-row text-memory q4 `to_k`), which CFG × 7 and the ancestral noise amplified to
    /// 255/255 in the 8-bit q4 image. The latent path runs `tokens` vs `2·tokens` rows; at SDXL's
    /// 512²/1024² attention shapes every q4 projection measured row-exact between the two, so the
    /// conditioning-only sites were the whole q4 deviation. (A dense f16 `ff.net.2` at 1024+
    /// latent tokens is ~4e-6 apart — the GEMM reassociation `CfgBatching` admits.) The extra cost
    /// is those small projections running at two rows; no latent-sized activation is doubled.
    /// `None` everywhere else (batched CFG, CFG off, training) — the historical forward.
    pub cfg_row: Option<i32>,
}

impl Default for SdxlForwardPlan<'_> {
    fn default() -> Self {
        Self::UNBOUNDED
    }
}

impl<'a> SdxlForwardPlan<'a> {
    /// The historical unbounded forward: no attention chunking, no block window, batched CFG.
    pub const UNBOUNDED: Self = Self {
        attention: AttentionPlan::UNBOUNDED,
        window: None,
        cfg_batching: CfgBatching::Batched,
        cfg_row: None,
    };

    /// Rung 3 only, leaving the transformer stack resident.
    pub fn with_attention(attention: AttentionPlan<'a>) -> Self {
        Self {
            attention,
            ..Self::UNBOUNDED
        }
    }

    /// Select the guided step's CFG batching (sc-26268).
    pub fn with_cfg_batching(mut self, cfg_batching: CfgBatching) -> Self {
        self.cfg_batching = cfg_batching;
        self
    }

    /// Mark this forward as row `k` of a sequential guided pair (sc-26268; see [`Self::cfg_row`]).
    pub fn with_cfg_row(mut self, cfg_row: Option<i32>) -> Self {
        self.cfg_row = cfg_row;
        self
    }

    /// The forward's own rows of a conditioning-only tensor `a` computed at the guided pair's
    /// batch: row [`cfg_row`](Self::cfg_row) when `a` carries more rows than the forward's latent
    /// `batch`, `a` itself otherwise (the batched forward, or a single row shared by both halves).
    pub fn select_cfg_row(&self, a: Array, batch: i32) -> mlx_gen::Result<Array> {
        match self.cfg_row {
            Some(k) if a.shape().first().is_some_and(|rows| *rows > batch) => {
                Ok(a.take_axis(Array::from_slice(&[k], &[1]), 0)?)
            }
            _ => Ok(a),
        }
    }

    /// The batch every conditioning-only projection runs at: the guided pair's for a sequential
    /// half, the latent's otherwise.
    pub fn conditioning_batch(&self, latent_batch: i32) -> i32 {
        match self.cfg_row {
            Some(_) => CfgBatching::Batched.guided_batch() as i32,
            None => latent_batch,
        }
    }

    /// Add (or clear) rung 4's cadence.
    pub fn with_window(mut self, window: Option<SdxlBlockWindow<'a>>) -> Self {
        self.window = window;
        self
    }

    /// Whether rung 4 is engaged for this forward.
    pub fn windows_blocks(&self) -> bool {
        self.window.is_some()
    }
}
