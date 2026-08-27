//! Ladder rung 4 — **bounded transformer residency** for the LTX AvDiT block stack, Candle half
//! (sc-18797).
//!
//! The window lifecycle is NOT here: it is [`candle_gen::block_window::run_windowed`], which binds
//! Candle's answers to the shared `gen_core::block_window` driver. This module is only the
//! family-side half — *"how do I rebuild AvBlock `n` from the transformer component"* — which is the
//! part that genuinely differs per family.
//!
//! It is the deliberate twin of `mlx_gen_ltx::block_stream`, and the differences between them are
//! the backends' real differences rather than drift:
//!
//! | | MLX | Candle (here) |
//! |---|---|---|
//! | view | a fresh `Weights` map per window | a cloned [`VarBuilder`] |
//! | freshness | must re-`load_safetensors`: the map RETAINS `Array` handles | structural — the mmap backend caches nothing, so a clone IS fresh |
//! | drain | `remove_accessed` is load-bearing | nothing to drain: `get` produces a new tensor each time |
//! | per-window release | `clear_cache()` | a measured no-op; the synchronize is per-forward and the driver owns it |
//!
//! Carrying MLX's mechanism across rather than its shape is the mistake `gen_core::block_window`'s
//! docs warn about twice, so each row above is Candle's own answer, not a translation.
//!
//! ## What an AV block window bounds — both modalities at once
//!
//! LTX's audio branch is not a second transformer: each `transformer_blocks.{n}` entry carries the
//! video stack, the audio stack and both cross-modal attentions, so one window over the block axis
//! bounds video and audio weights together. See `memory_strategy_2_5`'s
//! `TRANSFORMER_WINDOW_COMPONENT` for why that makes the declared scope `Dit` and not `Both`.

use std::sync::atomic::{AtomicU64, Ordering};

use candle_gen::candle_nn::VarBuilder;
use candle_gen::gen_core::AdapterSpec;
use candle_gen::{CandleError, Result};

use crate::config::AvConfig;
use crate::transformer::AvBlock;

static WINDOW_REOPENS: AtomicU64 = AtomicU64::new(0);
static BLOCK_MATERIALIZATIONS: AtomicU64 = AtomicU64::new(0);

/// Process-wide diagnostics for LTX's bounded transformer-residency path.
///
/// These exist because rung 4's failure mode is **output-invisible**: a load that silently keeps the
/// resident stack produces byte-identical frames while bounding nothing, so no output comparison can
/// see it. They observe *completed* operations rather than requested flags — a reopen increments
/// only after a view was produced, a materialization only after the exact `AvBlock` was assembled.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub struct BlockStreamDiagnostics {
    /// Views opened by [`candle_gen::block_window::run_windowed`], i.e. windows actually walked.
    pub window_reopens: u64,
    /// `AvBlock`s rebuilt out of a window's view.
    pub block_materializations: u64,
}

/// Reset the process-wide block-stream diagnostics before an observation.
pub fn reset_block_stream_diagnostics() {
    WINDOW_REOPENS.store(0, Ordering::Relaxed);
    BLOCK_MATERIALIZATIONS.store(0, Ordering::Relaxed);
}

/// Snapshot completed window operations since the last reset.
pub fn block_stream_diagnostics() -> BlockStreamDiagnostics {
    BlockStreamDiagnostics {
        window_reopens: WINDOW_REOPENS.load(Ordering::Relaxed),
        block_materializations: BLOCK_MATERIALIZATIONS.load(Ordering::Relaxed),
    }
}

/// Everything needed to rebuild one LTX `AvBlock` on demand.
#[derive(Clone)]
pub struct LtxBlockStream {
    /// A `VarBuilder` rooted exactly where [`crate::transformer::AvDiT::new`] roots its own, so a
    /// materialized block reads the same tensors its resident twin would.
    vb: VarBuilder<'static>,
    cfg: AvConfig,
    /// Ladder rung 3, replayed onto each materialized block so a rung-3 + rung-4 composition runs
    /// the SAME attention on the streamed path as on the resident one.
    ///
    /// Without the replay the composition is wrong in the direction that hides: the contract
    /// declares both rungs, the selector engages both (rung 4 engages rung 3 by cost order), and
    /// every window rebuilds a block at the unbounded default — bounded weights, unbounded scores,
    /// identical output.
    attn_budget: usize,
}

impl LtxBlockStream {
    /// Declare a streamable stack over a re-openable `vb`.
    ///
    /// `adapters` is the load spec's adapter set and **must be empty**. LTX installs adapters onto
    /// the *loaded* block objects (`AvDiT::visit_adaptable_mut`), so a block re-read from the base
    /// component per window would silently carry none of them — correct-looking output from the
    /// wrong weights. Refusing to construct is the only honest answer; the contract then declares
    /// rung 4 unavailable for an adapted load rather than bounding an un-adapted stack.
    pub fn new(vb: VarBuilder<'static>, cfg: AvConfig, adapters: &[AdapterSpec]) -> Result<Self> {
        if !adapters.is_empty() {
            return Err(CandleError::Msg(format!(
                "ltx block stream: {} adapter file(s) are installed, and LTX applies adapters to \
                 loaded block objects rather than as forward-time residuals replayable from the \
                 base component. A streamed block re-read per window would carry none of them, so \
                 bounded transformer residency is refused on an adapted load.",
                adapters.len()
            )));
        }
        Ok(Self {
            vb,
            cfg,
            attn_budget: candle_gen::ATTN_SCORES_BUDGET,
        })
    }

    /// The stack depth this stream materializes — read from the model config, never a constant, so a
    /// `num_layers` overlay cannot desynchronize the plan from the checkpoint.
    pub fn n_blocks(&self) -> usize {
        self.cfg.video.num_layers
    }

    /// Record rung 3's attention score budget so every materialized block executes the same
    /// attention its resident twin would.
    pub fn set_attention_budget(&mut self, budget: usize) {
        self.attn_budget = budget;
    }

    /// The budget a materialized block will carry.
    pub fn attention_budget(&self) -> usize {
        self.attn_budget
    }

    /// Open a view for one window.
    ///
    /// A clone, and that is the honest discharge of the driver's freshness obligation on this
    /// backend rather than a shortcut: `VarBuilder` over the mmap backend caches nothing, so every
    /// `get` reads the mapping and produces a new device tensor. There is no retained buffer for a
    /// stale view to pin, and paying a re-`mmap` per window would buy a guarantee the type already
    /// gives. The MLX twin genuinely must re-open, because its view IS a map of live handles.
    pub fn open(&self) -> Result<VarBuilder<'static>> {
        let view = self.vb.clone();
        WINDOW_REOPENS.fetch_add(1, Ordering::Relaxed);
        Ok(view)
    }

    /// Materialize block `index` out of `view` at the resident stack's budget.
    ///
    /// No drain: gen-core's *"`apply` must take its tensors OUT of the view"* rule is an MLX rule
    /// about a `HashMap<String, Array>`. Here `apply` **constructs** its block from the mapping and
    /// drops it at the end of the window; there is nothing to take out. Same obligation, different
    /// discharge.
    pub(crate) fn materialize(&self, view: &VarBuilder<'static>, index: usize) -> Result<AvBlock> {
        if index >= self.n_blocks() {
            return Err(CandleError::Msg(format!(
                "ltx block stream: block {index} is out of range for a {}-block stack",
                self.n_blocks()
            )));
        }
        let mut block = AvBlock::load(view.pp(format!("transformer_blocks.{index}")), &self.cfg)?;
        block.set_attention_budget(self.attn_budget);
        BLOCK_MATERIALIZATIONS.fetch_add(1, Ordering::Relaxed);
        Ok(block)
    }
}
