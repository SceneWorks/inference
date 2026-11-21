//! ComfyUI **Kitchen** NVFP4 import (sc-20641), split out of the NVFP4 codec in sc-24135.
//!
//! The codec itself ([`Nvfp4Tensor`]) is a backend-neutral pack/dequant container shared by the
//! media and LLM lanes. Importing a Kitchen-serialized checkpoint is a media-checkpoint concern, and
//! it is the one place the codec read [`gen_core::blocked_scale_index`] (the transliterated
//! ComfyUI `to_blocked` swizzle). Keeping it here keeps the codec free of the gen-core contract, so
//! the LLM lane can share the codec without compiling the media contracts.
//!
//! Callers keep the `Nvfp4Tensor::from_kitchen_parts(...)` spelling by bringing
//! [`Nvfp4KitchenExt`] into scope.

use candle_quant_kernels::nvfp4::{non_finite_msg, Nvfp4Tensor, NVFP4_BLOCK, SF_ATOM_COLS, SF_ATOM_ROWS};
use candle_core::Result;

/// The Kitchen constructor for [`Nvfp4Tensor`]. A trait only because the container lives in the
/// shared codec; there is exactly one implementation.
pub trait Nvfp4KitchenExt: Sized {
    /// Build the canonical Candle NVFP4 container from a ComfyUI Kitchen serialization.
    ///
    /// Two lossless representation conversions happen here; no weight or scale is requantized, and
    /// the FP32 second-level scale has the same numeric convention in both.
    ///
    /// 1. **Nibble order.** Kitchen's `hi_first=true` stores the even K element in the high nibble
    ///    (`comfy.float.stochastic_float_to_fp4_e2m1`: `packed = (even << 4) | odd`), while
    ///    [`Nvfp4Tensor`] stores it in the low nibble. Every byte is rotated.
    ///
    /// 2. **Block-scale atom order** (sc-20641). Both layouts are the cuBLAS 128×4 scale-factor
    ///    swizzle with the *same* intra-atom slot, but they walk the atom grid differently:
    ///    ComfyUI's `comfy.float.to_blocked` permutes `(R/128, 128, B/4, 4) → (R/128, B/4, 128, 4)`,
    ///    i.e. **row-major** over atoms, while [`Nvfp4Tensor::scale_offset_for`] walks it **column-major**
    ///    (CuTe `blocked_product` `LayoutLeft`). The two coincide only when the weight has a single
    ///    atom in one dimension — which is why this constructor's original single-atom test could
    ///    not see the difference, and why every real DiT projection was affected (a `[6144, 6144]`
    ///    layer has 48 row atoms × 48 block atoms). Scales are therefore **permuted** into the
    ///    container's own convention — read through [`gen_core::blocked_scale_index`], written
    ///    through [`Nvfp4Tensor::scale_offset_for`] — rather than copied verbatim.
    ///
    ///    Going through both index functions keeps this correct if the atom order the [module
    ///    docs](candle_quant_kernels::nvfp4) flag as "the one degree of freedom" is later flipped to match live cuBLASLt:
    ///    the permutation collapses to the identity on its own, with no edit here.
    ///
    /// `rows` and `cols` are the logical (already 16-aligned) weight dimensions. The constructor is
    /// intentionally strict because imported checkpoints are untrusted input and a malformed scale
    /// surface would otherwise reach cuBLASLt as a device buffer with the wrong layout.
    fn from_kitchen_parts(
        packed_hi_first: &[u8],
        blocked_scales: &[u8],
        global_scale: f32,
        rows: usize,
        cols: usize,
    ) -> Result<Self>;
}

impl Nvfp4KitchenExt for Nvfp4Tensor {
    fn from_kitchen_parts(
        packed_hi_first: &[u8],
        blocked_scales: &[u8],
        global_scale: f32,
        rows: usize,
        cols: usize,
    ) -> Result<Self> {
        if rows == 0 || cols == 0 || !rows.is_multiple_of(16) || !cols.is_multiple_of(NVFP4_BLOCK) {
            return Err(candle_core::Error::Msg(format!(
                "nvfp4 Kitchen weight shape must be non-zero and 16-aligned, got [{rows}, {cols}]"
            )));
        }
        let packed_len = rows
            .checked_mul(cols / 2)
            .ok_or_else(|| candle_core::Error::Msg("nvfp4 Kitchen weight size overflow".into()))?;
        if packed_hi_first.len() != packed_len {
            return Err(candle_core::Error::Msg(format!(
                "nvfp4 Kitchen packed weight has {} bytes, expected {packed_len} for [{rows}, {cols}]",
                packed_hi_first.len()
            )));
        }
        let scale_len = Nvfp4Tensor::scale_tensor_len(rows, cols);
        if blocked_scales.len() != scale_len {
            return Err(candle_core::Error::Msg(format!(
                "nvfp4 Kitchen block scales have {} bytes, expected {scale_len} for [{rows}, {cols}]",
                blocked_scales.len()
            )));
        }
        if !global_scale.is_finite() {
            return Err(candle_core::Error::Msg(non_finite_msg(
                "Kitchen global scale",
                global_scale,
            )));
        }
        if global_scale < 0.0 {
            return Err(candle_core::Error::Msg(format!(
                "nvfp4 Kitchen global scale must be non-negative, got {global_scale}"
            )));
        }

        let packed = packed_hi_first
            .iter()
            .map(|byte| byte.rotate_left(4))
            .collect();
        let cols_padded = cols;
        let blocks = cols / NVFP4_BLOCK;
        let sf_rows = rows.div_ceil(SF_ATOM_ROWS) * SF_ATOM_ROWS;
        let sf_cols = blocks.div_ceil(SF_ATOM_COLS) * SF_ATOM_COLS;
        // Permute the atom grid from Kitchen's `to_blocked` order into this container's — see the
        // doc comment. Padded slots keep the 0x00 they are initialized with; both layouts pad the
        // same `sf_rows * sf_cols` surface, so nothing is lost or invented.
        let mut scales = vec![0u8; sf_rows * sf_cols];
        for r in 0..rows {
            for blk in 0..blocks {
                scales[Nvfp4Tensor::scale_offset_for(r, blk, sf_rows)] =
                    blocked_scales[gen_core::blocked_scale_index(rows, blocks, r, blk)];
            }
        }
        Ok(Nvfp4Tensor {
            rows,
            cols,
            cols_padded,
            packed,
            scales,
            sf_rows,
            sf_cols,
            global_scale,
        })
    }
}

