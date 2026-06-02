//! Z-Image VACE-style control block (sc-2349 / sc-2257). Port of the fork's
//! `ZImageControlTransformerBlock`.
//!
//! A control block mirrors a base [`ZImageTransformerBlock`] (identical attention / SwiGLU FFN /
//! adaLN submodules and weight keys) and adds the two projections that thread the control hidden
//! state through the parallel control stack:
//!
//!   - `before_proj` (block 0 only): projects the incoming control context and adds the base
//!     hidden state once, seeding the control branch.
//!   - `after_proj` (every block): the projection whose output is the per-block *hint* added back
//!     into the base transformer at the matching place.
//!
//! The forward threading itself lives at the transformer level
//! ([`crate::control_transformer::ZImageControlTransformer`]) — a control block only owns the
//! submodules; `_run_control_blocks` runs the base-block forward and applies the projections (this
//! mirrors the fork, where `ZImageControlTransformer._run_control_blocks` calls
//! `ZImageTransformerBlock.__call__(block, …)` explicitly rather than via the subclass).

use crate::transformer_block::{ZImageBlockConfig, ZImageTransformerBlock};
use mlx_gen::adapters::AdaptableLinear;
use mlx_gen::weights::Weights;
use mlx_gen::Result;

pub struct ZImageControlBlock {
    /// The base transformer block (attention + FFN + adaLN) — its `forward` is what
    /// `_run_control_blocks` runs on the threaded control state.
    pub(crate) base: ZImageTransformerBlock,
    /// Block-0-only seed projection (`before_proj(c) + x_base`).
    before_proj: Option<AdaptableLinear>,
    /// Per-block hint projection (`after_proj(c)`), injected into the base stream.
    after_proj: AdaptableLinear,
}

impl ZImageControlBlock {
    /// Load a control block from the Fun-Controlnet-Union checkpoint under `prefix` (e.g.
    /// `"control_layers.0"`). The base-block keys (`attention.*`, `feed_forward.*`,
    /// `attention_norm{1,2}`, `ffn_norm{1,2}`, `adaLN_modulation.0`) map 1:1 onto
    /// [`ZImageTransformerBlock::from_weights`]; `after_proj` is present on every block and
    /// `before_proj` only on block 0 (`has_before_proj`).
    pub fn from_weights(
        w: &Weights,
        prefix: &str,
        cfg: ZImageBlockConfig,
        has_before_proj: bool,
    ) -> Result<Self> {
        let base = ZImageTransformerBlock::from_weights(w, prefix, cfg)?;
        let after_proj = AdaptableLinear::dense(
            w.require(&format!("{prefix}.after_proj.weight"))?.clone(),
            Some(w.require(&format!("{prefix}.after_proj.bias"))?.clone()),
        );
        let before_proj = if has_before_proj {
            Some(AdaptableLinear::dense(
                w.require(&format!("{prefix}.before_proj.weight"))?.clone(),
                Some(w.require(&format!("{prefix}.before_proj.bias"))?.clone()),
            ))
        } else {
            None
        };
        Ok(Self {
            base,
            before_proj,
            after_proj,
        })
    }

    /// Quantize every Linear in the block to Q4/Q8 (group_size 64): the base block's
    /// attention/FFN/adaLN plus the control projections. All have a `% 64 == 0` in-feature
    /// dimension (3840), so all quantize — the only non-divisible control Linear is the patch
    /// embedder, handled in [`crate::control_transformer::ZImageControlTransformer::quantize`].
    pub fn quantize(&mut self, bits: i32) -> Result<()> {
        self.base.quantize(bits)?;
        self.after_proj.quantize(bits, None)?;
        if let Some(bp) = &mut self.before_proj {
            bp.quantize(bits, None)?;
        }
        Ok(())
    }

    /// The block-0 seed projection (`None` for every other block).
    pub(crate) fn before_proj(&self) -> Option<&AdaptableLinear> {
        self.before_proj.as_ref()
    }

    /// The per-block hint projection.
    pub(crate) fn after_proj(&self) -> &AdaptableLinear {
        &self.after_proj
    }
}
