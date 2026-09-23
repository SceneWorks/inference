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

use super::nvfp4::{non_finite_msg, Nvfp4Tensor, NVFP4_BLOCK, SF_ATOM_COLS, SF_ATOM_ROWS};
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
    ///    docs](super::nvfp4) flag as "the one degree of freedom" is later flipped to match live cuBLASLt:
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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::quant::nvfp4::e4m3_to_f32;

    #[test]
    fn kitchen_parts_swap_only_the_nibble_order() -> Result<()> {
        let (rows, cols) = (128, 64);
        let mut packed = vec![0u8; rows * cols / 2];
        // Kitchen hi-first: even element code 1 in the high nibble, odd code 2 in the low nibble.
        packed[0] = 0x12;
        let mut scales = vec![0u8; Nvfp4Tensor::scale_tensor_len(rows, cols)];
        scales[Nvfp4Tensor::scale_offset_for(0, 0, rows)] = 0x38; // E4M3 1.0

        let tensor = Nvfp4Tensor::from_kitchen_parts(&packed, &scales, 2.0, rows, cols)?;
        assert_eq!(tensor.packed[0], 0x21);
        assert_eq!(
            tensor.scales, scales,
            "a single-atom weight's blocked scales are byte-exact: the atom-order permutation is \
             the identity when there is only one atom"
        );
        let dense = tensor.dequantize_to_vec();
        assert_eq!(dense[0], 1.0); // E2M1(1) 0.5 * scale 1.0 * global 2.0
        assert_eq!(dense[1], 2.0); // E2M1(2) 1.0 * scale 1.0 * global 2.0
        Ok(())
    }

    /// sc-20641. The case the single-atom test above structurally cannot reach: a weight with more
    /// than one scale atom in **both** dimensions, where ComfyUI's row-major `to_blocked` atom walk
    /// and this container's column-major [`Nvfp4Tensor::scale_offset_for`] disagree.
    ///
    /// Every real DiT projection is in this regime (`[6144, 6144]` → 48 × 48 atoms), so a verbatim
    /// copy of Kitchen's scale buffer would hand cuBLASLt — and the dequant fallback — the right
    /// bytes in the wrong slots. The guard is on the decoded value of a specific element whose
    /// block scale moves under the permutation, not on the buffer as a whole.
    #[test]
    fn kitchen_parts_permute_block_scales_across_a_multi_atom_grid() -> Result<()> {
        // 256 rows = 2 row atoms; 128 cols = 8 blocks = 2 block atoms.
        let (rows, cols) = (256, 128);
        let blocks = cols / NVFP4_BLOCK;
        assert!(
            rows / SF_ATOM_ROWS > 1 && blocks / SF_ATOM_COLS > 1,
            "the fixture must have >1 atom in both dimensions or it proves nothing"
        );

        // Kitchen buffer: give each (row, block) a distinct, recoverable E4M3 code so a wrong slot
        // is a wrong *value*, not just a wrong address. Codes 0x30..0x40 are 0.5..1.875.
        let code_for = |r: usize, blk: usize| -> u8 { 0x30 + ((r / 32 + blk * 3) % 16) as u8 };
        let mut kitchen_scales = vec![0u8; Nvfp4Tensor::scale_tensor_len(rows, cols)];
        for r in 0..rows {
            for blk in 0..blocks {
                kitchen_scales[gen_core::blocked_scale_index(rows, blocks, r, blk)] =
                    code_for(r, blk);
            }
        }
        // Every element code 2 (E2M1 1.0), so the decoded value *is* the block scale × global.
        let packed = vec![0x22_u8; rows * cols / 2];
        let global = 4.0_f32;

        let tensor = Nvfp4Tensor::from_kitchen_parts(&packed, &kitchen_scales, global, rows, cols)?;

        // The permutation is NOT the identity here — this is the assertion the old fixture could
        // not make.
        assert_ne!(
            tensor.scales, kitchen_scales,
            "a multi-atom grid must be permuted, not copied"
        );
        // Each scale landed in the container's own slot...
        for r in 0..rows {
            for blk in 0..blocks {
                assert_eq!(
                    tensor.scales[Nvfp4Tensor::scale_offset_for(r, blk, tensor.sf_rows)],
                    code_for(r, blk),
                    "scale for (row {r}, block {blk})"
                );
            }
        }
        // ...and the container's own dequant therefore recovers the value ComfyUI meant, element by
        // element. Column 100 sits in block 6, whose scale moves atoms under the permutation.
        let dense = tensor.dequantize_to_vec();
        for (r, c) in [(0_usize, 0_usize), (5, 100), (200, 100), (255, 127)] {
            let expected = e4m3_to_f32(code_for(r, c / NVFP4_BLOCK)) * global;
            assert_eq!(dense[r * cols + c], expected, "element ({r}, {c})");
        }
        Ok(())
    }

    /// sc-20651 review (9). `scale_swizzle_padding_and_bijection` above proves the swizzle is a
    /// **bijection**; it cannot see the **order**. Neither can any round-trip test in this crate: the
    /// packer writes through [`Nvfp4Tensor::scale_offset_for`] and the dequant reads back through the
    /// same function, so *any* bijective permutation of the offset — a transpose of the intra-atom
    /// slot, a transpose of the atom grid — cancels exactly and is invisible end to end. The order is
    /// only load-bearing at the two seams where a **foreign** producer/consumer is on the other side:
    /// ComfyUI Kitchen's `comfy_kitchen.float_utils.to_blocked` buffer on the way in, and cuBLASLt's
    /// `VEC16` `UE4M3` block-scale descriptor on the way out.
    ///
    /// So this pins the offset against the external producer instead of against itself.
    /// [`gen_core::blocked_scale_index`] *is* `to_blocked`, transliterated from its definition and
    /// itself pinned in `gen-core` against a matrix computed by running `to_blocked` in a real ComfyUI
    /// venv (`mxfp8_swizzle_index_matches_comfy_kitchen_to_blocked`). The two layouts are the same
    /// cuBLAS 128×4 swizzle and differ in exactly one documented way — Kitchen walks the atom grid
    /// row-major, this container walks it column-major (CuTe `LayoutLeft`) — so:
    ///
    /// - the **intra-atom slot** (`offset % 512`) must be *equal* to Kitchen's, and
    /// - the **atom index** (`offset / 512`) must be Kitchen's atom index *transposed*.
    ///
    /// Both halves are stated in terms of `blocked_scale_index`, never by re-spelling
    /// `scale_offset_for`'s own arithmetic, so a permutation applied to the production function has
    /// nothing to cancel against.
    ///
    /// Live cuBLASLt confirmation of the atom order (the second seam) is still the terminal CUDA-box
    /// gate for `nvfp4_native=true`; this is the host-side half.
    #[test]
    fn scale_offset_order_is_pinned_to_the_external_to_blocked_producer() {
        const ATOM: usize = SF_ATOM_ROWS * SF_ATOM_COLS; // 512 bytes

        // A real-DiT-shaped multi-atom grid: 256 rows = 2 row atoms, 8 blocks = 2 block atoms. With a
        // single atom in either dimension the two atom walks coincide and this proves nothing.
        let (rows, blocks) = (256_usize, 8_usize);
        let sf_rows = rows.div_ceil(SF_ATOM_ROWS) * SF_ATOM_ROWS;
        let (num_m_atoms, num_k_atoms) = (sf_rows / SF_ATOM_ROWS, blocks / SF_ATOM_COLS);
        assert!(
            num_m_atoms > 1 && num_k_atoms > 1,
            "the fixture must have >1 atom in both dimensions or the atom-order half is vacuous"
        );

        for r in 0..rows {
            for blk in 0..blocks {
                let ours = Nvfp4Tensor::scale_offset_for(r, blk, sf_rows);
                let kitchen = gen_core::blocked_scale_index(rows, blocks, r, blk);

                // (i) Intra-atom slot: identical to `to_blocked`'s. This is the half the module docs
                // call "the fixed, load-bearing part" — a transpose of `intra` breaks it here.
                assert_eq!(
                    ours % ATOM,
                    kitchen % ATOM,
                    "intra-atom slot for (row {r}, block {blk}) must match comfy `to_blocked`"
                );

                // (ii) Atom index: Kitchen is row-major over the atom grid, this container is
                // column-major, so ours must be the transpose of Kitchen's — not merely different.
                let kitchen_atom = kitchen / ATOM;
                let transposed =
                    (kitchen_atom % num_k_atoms) * num_m_atoms + (kitchen_atom / num_k_atoms);
                assert_eq!(
                    ours / ATOM,
                    transposed,
                    "atom index for (row {r}, block {blk}) must be `to_blocked`'s atom transposed"
                );
            }
        }

        // Two concrete bytes of the documented CUTLASS layout, so a reader can check the claim by
        // hand and so a mutation that happens to preserve the relations above still trips here.
        // Intra-atom: row 1 → slot 16 (`(1 % 32) * 16`); row 32 → slot 4 (`(32 / 32) * 4`). A
        // transpose of the intra formula swaps exactly these two.
        assert_eq!(Nvfp4Tensor::scale_offset_for(1, 0, 128), 16);
        assert_eq!(Nvfp4Tensor::scale_offset_for(32, 0, 128), 4);
        // Atom order, on the 2×2 grid: (row 0, block 4) is atom 2 column-major → byte 1024, where
        // `to_blocked` puts it at 512 (gen-core pins that literal). The disagreement is the whole
        // reason `from_kitchen_parts` permutes rather than copies.
        assert_eq!(Nvfp4Tensor::scale_offset_for(0, 4, 256), 1024);
        assert_eq!(gen_core::blocked_scale_index(256, 8, 0, 4), 512);
    }

    /// The Kitchen constructor's per-tensor-scale gate shares [`non_finite_msg`] with the packer, so
    /// both name a NaN/inf the same way. Non-negativity stays a separate, separately-worded refusal.
    #[test]
    fn kitchen_parts_refuses_a_non_finite_or_negative_global_scale() {
        let (rows, cols) = (128_usize, 64_usize);
        let packed = vec![0u8; rows * cols / 2];
        let scales = vec![0u8; Nvfp4Tensor::scale_tensor_len(rows, cols)];

        for bad in [f32::NAN, f32::INFINITY, f32::NEG_INFINITY] {
            let msg = Nvfp4Tensor::from_kitchen_parts(&packed, &scales, bad, rows, cols)
                .expect_err("non-finite global scale must be refused")
                .to_string();
            assert!(msg.contains("must be finite"), "got {msg}");
        }
        let msg = Nvfp4Tensor::from_kitchen_parts(&packed, &scales, -1.0, rows, cols)
            .expect_err("negative global scale must be refused")
            .to_string();
        assert!(msg.contains("must be non-negative"), "got {msg}");
    }
}
