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
use mlx_gen::CancelFlag;

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
/// `Copy` and cheap (a budget, two `Option`s and a borrow), so the per-step closures in
/// `pipeline` can capture it without a lifetime fight.
#[derive(Clone, Copy, Default)]
pub struct SdxlForwardPlan<'a> {
    /// Rung 3 — the query-row chunking budget handed to
    /// [`mlx_gen::attention::sdpa_budgeted_bhsd`] at every attention site.
    pub attention: AttentionPlan<'a>,
    /// Rung 4 — the block-window cadence, or `None` for the resident stack.
    pub window: Option<SdxlBlockWindow<'a>>,
}

impl<'a> SdxlForwardPlan<'a> {
    /// The historical unbounded forward: no attention chunking, no block window.
    pub const UNBOUNDED: Self = Self {
        attention: AttentionPlan::UNBOUNDED,
        window: None,
    };

    /// Rung 3 only, leaving the transformer stack resident.
    pub fn with_attention(attention: AttentionPlan<'a>) -> Self {
        Self {
            attention,
            window: None,
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
