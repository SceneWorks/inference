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
use mlx_gen::attention::AttentionPlan;
use mlx_gen::weights::Weights;
use mlx_gen::Result;
use mlx_rs::Array;

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
        // Packed-detect (sc-8670): the control projections load packed from a pre-quantized control
        // snapshot or dense otherwise; both carry a bias. Their in-features (3840) are group-64
        // divisible, so they pack in a quantized tier.
        let after_proj = crate::quant::lin(w, &format!("{prefix}.after_proj"), true)?;
        let before_proj = if has_before_proj {
            Some(crate::quant::lin(
                w,
                &format!("{prefix}.before_proj"),
                true,
            )?)
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

    /// [`Self::quantize`] as one iOS quantum: the packed copies ([`Self::quantize_bytes`]) are checked
    /// before they exist and materialized before return, so the resident dense source is released
    /// block by block. Exactly `quantize` off device.
    pub(crate) fn quantize_quantum(&mut self, bits: i32) -> Result<()> {
        if !mlx_gen::ios_quantum::enabled() {
            return self.quantize(bits);
        }
        mlx_gen::ios_quantum::before(crate::quantum::CONTROL_QUANTIZE, self.quantize_bytes(bits))?;
        self.quantize(bits)?;
        mlx_gen::ios_quantum::finish(crate::quantum::CONTROL_QUANTIZE, &self.weight_arrays())
    }

    /// The packed bytes [`Self::quantize`] builds from this block's dense projections (0 when packed).
    pub(crate) fn quantize_bytes(&self, bits: i32) -> u64 {
        self.base
            .projections()
            .into_iter()
            .chain(std::iter::once(&self.after_proj))
            .chain(self.before_proj.iter())
            .map(|lin| crate::quantum::quantize_bytes(lin, bits))
            .sum()
    }

    /// The block's iOS quantum price over control state `c` (`[1, L, D]`): the base block's
    /// ([`ZImageTransformerBlock::quantum_scratch`]) plus the `after_proj` hint and, on block 0, the
    /// `before_proj` output and its sum with the base stream (`[1, L, D]` each, f32 — the control
    /// branch runs f32), the projections' stream-width weight copies.
    pub(crate) fn quantum_scratch(
        &self,
        c: &Array,
        x_base: &Array,
        freqs_cis: &Array,
        plan: AttentionPlan<'_>,
        backend: mlx_gen::ios_quantum::Backend,
    ) -> u64 {
        let width = if self.before_proj.is_some() {
            c.item_size().max(x_base.item_size())
        } else {
            c.item_size()
        }
        .max(freqs_cis.item_size()) as u64;
        let sh = c.shape();
        let u = crate::quantum::u;
        let hidden = crate::quantum::tb(width, &[u(sh[0]), u(sh[1]), u(sh[2])]);
        let mut projections = vec![&self.after_proj];
        projections.extend(self.before_proj.iter());
        let units = 1 + 2 * u64::from(self.before_proj.is_some());
        self.base
            .quantum_scratch(c, freqs_cis, plan, backend)
            .saturating_add(hidden.saturating_mul(units))
            .saturating_add(crate::quantum::promoted_weight_bytes(&projections, width))
            .saturating_add(mlx_gen::ios_quantum::page_slack(units))
    }

    /// Every weight array the block's projections hold (packed: weight, scales, biases, bias), for
    /// the iOS load-time packing quantum to materialize.
    pub(crate) fn weight_arrays(&self) -> Vec<&Array> {
        self.base
            .projections()
            .into_iter()
            .chain(std::iter::once(&self.after_proj))
            .chain(self.before_proj.iter())
            .flat_map(|lin| match lin.quantized_params() {
                Some((w, scales, biases, bias, ..)) => {
                    let mut v = vec![w, scales, biases];
                    v.extend(bias);
                    v
                }
                None => lin
                    .dense_weight()
                    .map(|(w, b)| std::iter::once(w).chain(b).collect())
                    .unwrap_or_default(),
            })
            .collect()
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
