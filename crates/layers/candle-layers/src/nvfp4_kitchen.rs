//! Kitchen NVFP4 checkpoint constructor.
pub use candle_quant_kernels::nvfp4_kitchen::Nvfp4KitchenExt;

#[cfg(test)]
mod tests {
    use super::*;
    use candle_core::Result;
    use candle_quant_kernels::nvfp4::e4m3_to_f32;
    use candle_quant_kernels::nvfp4::{Nvfp4Tensor, NVFP4_BLOCK, SF_ATOM_COLS, SF_ATOM_ROWS};

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
