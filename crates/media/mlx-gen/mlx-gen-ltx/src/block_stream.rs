//! Ladder rung 4 — **bounded transformer residency** for the LTX AvDiT block stack (sc-18797).
//!
//! The window lifecycle is NOT here: it is [`mlx_gen::block_residency::run_windowed`], which binds
//! MLX's two operations to the shared `gen_core::block_window` driver. This module is only the
//! family-side half — *"how do I rebuild AvBlock `n` from the transformer component, at the same
//! precision the resident block carries"* — which is the part that genuinely differs per family.
//!
//! Writing a provider-local window loop here instead would be the sc-15958 duplication trap the
//! epic's R9 forbids: the arithmetic, the loop order, the release discipline and the cancellation
//! contract are identical on every backend and every family, and the one place they live is
//! gen-core.
//!
//! ## Why re-reading per window is nearly free
//!
//! `Array::load_safetensors` is **lazy per tensor**: the handles exist, the bytes do not. Re-opening
//! the transformer component once per window costs a JSON header parse, not the whole 22 B stack.
//! Only the tensors a window's blocks actually read are materialized, and
//! [`Weights::remove_accessed`] then drops the view's own reference to exactly those, so the
//! window's residency is the blocks themselves and nothing else.
//!
//! ## What an AV block window bounds — both modalities at once
//!
//! LTX's audio branch is **not** a second transformer. Each of the 48 `transformer_blocks.{n}`
//! entries carries the video stack, the audio stack *and* the two cross-modal attentions
//! (`AvBlock`), so one window over the block axis bounds video and audio weights together. That is
//! why rung 4 here is declared at `TransformerComponent::Dit`
//! rather than `Both`: `Both` would additionally claim the **Gemma text encoder**, which is a
//! separate component, is not windowed by this module, and has nothing measured for it on this
//! family. Declaring a scope this code does not execute is exactly the unreachable-declaration
//! defect epic 18755's R9 exists to prevent.
//!
//! ## The per-window cost obligation
//!
//! `gen_core::block_window`'s module docs price a window at `n_blocks x steps` materializations,
//! so `open_view` plus the reads `apply` makes must be a transfer of bytes already in the form the
//! accelerator consumes. This module satisfies that structurally: the source is a packed
//! `.safetensors` whose quantized triples are independent file entries, `AvBlock::load` reads them
//! through the same `param`/`Linear::load` path the resident stack uses, and no repack, transpose or
//! dtype round trip happens per window. That is what lets the MLX realization answer
//! `MemoryWindowMaterialization::DeviceFormatTransfer`.

use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicU64, Ordering};

use mlx_gen::attention::AttentionBudget;
use mlx_gen::weights::Weights;
use mlx_gen::{AdapterSpec, Error, Result};

use crate::config::LtxConfig;
use crate::transformer::{AvBlock, Precision};

static WINDOW_REOPENS: AtomicU64 = AtomicU64::new(0);
static BLOCK_MATERIALIZATIONS: AtomicU64 = AtomicU64::new(0);

/// Process-wide diagnostics for LTX's bounded transformer-residency path.
///
/// These counters exist because rung 4's failure mode is **output-invisible**: a load that silently
/// keeps the resident stack produces byte-identical frames while bounding nothing, so no output
/// comparison can see it. They observe *completed* operations rather than requested flags — a reopen
/// increments only after the component file was parsed into a view, and a materialization only after
/// the exact `AvBlock` was assembled — so a test can distinguish a real block-window run from a
/// resident forward that merely accepted the flag.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub struct BlockStreamDiagnostics {
    /// Views opened by [`mlx_gen::block_residency::run_windowed`], i.e. windows actually walked.
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

/// Everything needed to rebuild one LTX `AvBlock` from its component file, on demand.
///
/// Cheap to clone: a path, a config and a precision.
#[derive(Clone, Debug)]
pub struct LtxBlockStream {
    /// The **transformer component** file, resolved by [`crate::bundle::resolve_split_bundle`]. A
    /// single re-openable `.safetensors`, never a directory: the block tensors must all come from
    /// one lazily-mapped file or a window's reopen would fan out across the whole bundle.
    source: PathBuf,
    cfg: LtxConfig,
    /// The precision the resident stack was built at, replayed per materialized block so a streamed
    /// block is bit-identical to its resident twin rather than merely close.
    prec: Precision,
    /// Ladder rung 3, replayed onto each materialized block so a rung-3 + rung-4 composition runs
    /// the SAME attention on the streamed path as on the resident one.
    ///
    /// Without this the composition is silently wrong in the direction that hides: the contract
    /// declares both rungs, the selector engages both (rung 4 engages rung 3 by cost order), and
    /// every window rebuilds a block at `UNBOUNDED` — bounded weights, unbounded scores, identical
    /// output.
    attn_budget: AttentionBudget,
}

impl LtxBlockStream {
    /// Declare a streamable stack over a re-openable transformer component.
    ///
    /// `adapters` is the load spec's adapter set and **must be empty**. LTX installs LoRA through
    /// [`crate::adapters`] against the *loaded* block objects, so a block re-read from the base
    /// component per window would silently carry none of them — correct-looking output from the
    /// wrong weights, which is the exact silent class rung 4 must not introduce. Refusing to
    /// construct is the only honest answer; the contract then declares rung 4 unavailable for an
    /// adapted load rather than bounding an un-adapted stack.
    pub fn new(
        source: impl AsRef<Path>,
        cfg: LtxConfig,
        prec: Precision,
        adapters: &[AdapterSpec],
    ) -> Result<Self> {
        if !adapters.is_empty() {
            return Err(Error::Unsupported(format!(
                "ltx block stream: {} adapter file(s) are installed, and LTX applies adapters to \
                 loaded block objects rather than as forward-time residuals replayable from the \
                 base component. A streamed block re-read per window would carry none of them, so \
                 bounded transformer residency is refused on an adapted load.",
                adapters.len()
            )));
        }
        let source = source.as_ref().to_path_buf();
        if !source.is_file() {
            return Err(Error::Unsupported(format!(
                "ltx block stream: the transformer component must be a single re-openable \
                 safetensors file; {} is not a file",
                source.display()
            )));
        }
        Ok(Self {
            source,
            cfg,
            prec,
            attn_budget: AttentionBudget::UNBOUNDED,
        })
    }

    /// Record rung 3's attention budget so every materialized block executes the same attention its
    /// resident twin would.
    pub fn set_attention_budget(&mut self, budget: AttentionBudget) {
        self.attn_budget = budget;
    }

    /// The budget a materialized block will carry.
    pub fn attention_budget(&self) -> AttentionBudget {
        self.attn_budget
    }

    /// The stack depth this stream materializes — read from the model config, never a constant, so a
    /// `num_layers` overlay in the component's `config` cannot desynchronize the plan from the
    /// checkpoint.
    pub fn n_blocks(&self) -> usize {
        self.cfg.num_layers as usize
    }

    /// The component this stream reopens. Exposed so a caller can pin or fingerprint it.
    pub fn source(&self) -> &Path {
        &self.source
    }

    /// Open a fresh lazy view of the stack's weights. Called once per window by
    /// [`mlx_gen::block_residency::run_windowed`].
    ///
    /// FRESH is load-bearing, not decorative: [`Weights`] is a map of refcounted `Array` handles, so
    /// a view retained across windows keeps every materialized buffer alive through its own map and
    /// the release frees nothing.
    pub fn open(&self) -> Result<Weights> {
        let view = Weights::from_file(&self.source)?;
        WINDOW_REOPENS.fetch_add(1, Ordering::Relaxed);
        Ok(view)
    }

    /// Materialize block `index` out of `view` at the resident stack's precision, then drain the view
    /// of precisely the tensors that block read.
    pub(crate) fn materialize(&self, view: &mut Weights, index: usize) -> Result<AvBlock> {
        if index >= self.n_blocks() {
            return Err(Error::Msg(format!(
                "ltx block stream: block {index} is out of range for a {}-block stack",
                self.n_blocks()
            )));
        }
        let mut block = AvBlock::load(
            view,
            &format!("transformer_blocks.{index}"),
            &self.cfg,
            self.prec,
        )?;
        block.set_attention_budget(self.attn_budget);
        // LOAD-BEARING: the view keeps its own refcounted handle to every tensor the constructor
        // read. Draining exactly the accessed keys is what makes the window's drop a real release
        // rather than a no-op that still produces correct frames.
        view.materialize_accessed()?;
        view.remove_accessed();
        BLOCK_MATERIALIZATIONS.fetch_add(1, Ordering::Relaxed);
        Ok(block)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn cfg() -> LtxConfig {
        LtxConfig::video_only_defaults()
    }

    /// An adapted load must not construct a stream at all. The refusal is the contract seam that
    /// keeps rung 4 from silently serving un-adapted weights, so it is asserted on the constructor
    /// rather than left to a caller's discipline.
    #[test]
    fn an_adapter_install_refuses_to_stream() {
        let spec = AdapterSpec {
            path: "/nonexistent/lora.safetensors".into(),
            scale: 1.0,
            kind: mlx_gen::AdapterKind::Lora,
            pass_scales: None,
            moe_expert: None,
        };
        let error = LtxBlockStream::new(
            "/nonexistent/transformer.safetensors",
            cfg(),
            Precision::quant_f32(4, 32),
            std::slice::from_ref(&spec),
        )
        .expect_err("an adapted load must refuse to stream");
        assert!(
            error.to_string().contains("adapter"),
            "the refusal must name adapters, got: {error}"
        );
    }

    /// A directory (or any non-file) source is refused: a window's reopen must map ONE component,
    /// not walk a bundle.
    #[test]
    fn a_non_file_source_is_refused() {
        let error = LtxBlockStream::new(
            "/nonexistent/sc-18797-not-a-file",
            cfg(),
            Precision::quant_f32(4, 32),
            &[],
        )
        .expect_err("a non-file component must refuse to stream");
        assert!(
            error.to_string().contains("is not a file"),
            "the refusal must name the file requirement, got: {error}"
        );
    }
}
