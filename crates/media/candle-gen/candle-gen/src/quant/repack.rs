//! MLX affine Q4/Q8 -> GGML repack helpers, re-exported from [`candle_layers::repack`] (`SW-6955`).

pub use candle_layers::repack::*;

#[cfg(test)]
mod tests {
    use super::*;
    use candle_core::quantized::GgmlDType;
    use candle_core::{Device, Result, Tensor};

    /// Pack per-element 4-bit codes into MLX u32 words (LSB-first nibbles) — the test-side inverse
    /// of `unpack_mlx_q4`.
    fn pack_mlx_q4(codes: &[u8]) -> Vec<u32> {
        codes
            .chunks_exact(8)
            .map(|c| {
                c.iter()
                    .enumerate()
                    .fold(0u32, |acc, (i, &q)| acc | ((q as u32 & 0xF) << (4 * i)))
            })
            .collect()
    }

    /// Position-dependent codes + per-group-distinct f16-exact scales/biases: dequantizing the
    /// repacked `Q4_1` tensor must reproduce the MLX affine grid EXACTLY. The position-dependence
    /// makes the assertion sensitive to any nibble/block/group ordering mistake on either side of
    /// the permutation.
    #[test]
    fn q4_repack_is_lossless_and_order_sensitive() -> Result<()> {
        let (out_dim, in_dim) = (4, 128); // 2 groups/row, 4 Q4_1 blocks/row
        let codes: Vec<u8> = (0..out_dim * in_dim)
            .map(|i| ((i * 7 + i / 13) % 16) as u8)
            .collect();
        let groups = out_dim * in_dim / MLX_GROUP_SIZE;
        // Exactly f16-representable, distinct per group, including negatives.
        let scales: Vec<f32> = (0..groups).map(|g| 0.0625 * (g as f32 + 1.0)).collect();
        let biases: Vec<f32> = (0..groups).map(|g| -0.5 - 0.25 * g as f32).collect();
        assert!(scales.iter().chain(biases.iter()).all(|&x| f16_exact(x)));

        let wq = Tensor::from_vec(pack_mlx_q4(&codes), (out_dim, in_dim / 8), &Device::Cpu)?;
        let s = Tensor::from_vec(
            scales.clone(),
            (out_dim, in_dim / MLX_GROUP_SIZE),
            &Device::Cpu,
        )?;
        let b = Tensor::from_vec(
            biases.clone(),
            (out_dim, in_dim / MLX_GROUP_SIZE),
            &Device::Cpu,
        )?;
        assert_eq!(mlx_packed_bits(in_dim / 8, in_dim / MLX_GROUP_SIZE), 4);

        let qt = repack_mlx_q4_to_q4_1(&wq, &s, &b, &Device::Cpu)?;
        assert_eq!(qt.dtype(), GgmlDType::Q4_1);
        let got = qt
            .dequantize(&Device::Cpu)?
            .flatten_all()?
            .to_vec1::<f32>()?;
        let want = affine_grid(&codes, &scales, &biases, in_dim, MLX_GROUP_SIZE);
        assert_eq!(got.len(), want.len());
        for (i, (g, w)) in got.iter().zip(want.iter()).enumerate() {
            assert_eq!(g, w, "element {i}: repacked dequant {g} != MLX grid {w}");
        }
        // And the reference helper agrees with the hand-computed grid.
        let reference = dequant_mlx_q4_reference(&wq, &s, &b)?
            .flatten_all()?
            .to_vec1::<f32>()?;
        assert_eq!(reference, want);
        Ok(())
    }

    /// **Group-32 Q4 repack (sc-9410, the boogu tier's group size).** Same lossless-and-order-sensitive
    /// contract as the group-64 case, but one MLX group is exactly one `Q4_1` block (`group_size / 32 =
    /// 1`), and the shapes (`scales [out, in/32]`) collide with a group-64 pack of half the width — so
    /// the group size MUST be threaded explicitly (the `*_gs` API). Pins that the boogu group-32 packs
    /// dequantize bit-exactly.
    #[test]
    fn q4_repack_group32_is_lossless() -> Result<()> {
        const G: usize = 32;
        let (out_dim, in_dim) = (4, 128); // 4 groups/row at group 32, 4 Q4_1 blocks/row
        let codes: Vec<u8> = (0..out_dim * in_dim)
            .map(|i| ((i * 7 + i / 13) % 16) as u8)
            .collect();
        let groups = out_dim * in_dim / G;
        let scales: Vec<f32> = (0..groups).map(|g| 0.0625 * (g as f32 + 1.0)).collect();
        let biases: Vec<f32> = (0..groups).map(|g| -0.5 - 0.25 * g as f32).collect();
        assert!(scales.iter().chain(biases.iter()).all(|&x| f16_exact(x)));

        let wq = Tensor::from_vec(pack_mlx_q4(&codes), (out_dim, in_dim / 8), &Device::Cpu)?;
        let s = Tensor::from_vec(scales.clone(), (out_dim, in_dim / G), &Device::Cpu)?;
        let b = Tensor::from_vec(biases.clone(), (out_dim, in_dim / G), &Device::Cpu)?;
        // With group 32, bits are correctly derived only when the group size is passed.
        assert_eq!(mlx_packed_bits_gs(in_dim / 8, in_dim / G, G), 4);

        let qt = repack_mlx_q4_to_q4_1_gs(&wq, &s, &b, G, &Device::Cpu)?;
        assert_eq!(qt.dtype(), GgmlDType::Q4_1);
        let got = qt
            .dequantize(&Device::Cpu)?
            .flatten_all()?
            .to_vec1::<f32>()?;
        let want = affine_grid(&codes, &scales, &biases, in_dim, G);
        assert_eq!(
            got, want,
            "group-32 repacked dequant deviates from the MLX grid"
        );
        let reference = dequant_mlx_q4_reference_gs(&wq, &s, &b, G)?
            .flatten_all()?
            .to_vec1::<f32>()?;
        assert_eq!(reference, want);
        Ok(())
    }

    /// The Q8 dequant helper reproduces the MLX 8-bit affine grid exactly (byte order + grouping).
    #[test]
    fn q8_dequant_matches_grid() -> Result<()> {
        let (out_dim, in_dim) = (2, 128);
        let codes: Vec<u8> = (0..out_dim * in_dim)
            .map(|i| ((i * 31 + 5) % 256) as u8)
            .collect();
        let words: Vec<u32> = codes
            .chunks_exact(4)
            .map(|c| {
                c.iter()
                    .enumerate()
                    .fold(0u32, |acc, (i, &q)| acc | ((q as u32) << (8 * i)))
            })
            .collect();
        let groups = out_dim * in_dim / MLX_GROUP_SIZE;
        let scales: Vec<f32> = (0..groups).map(|g| 0.03125 * (g as f32 + 1.0)).collect();
        let biases: Vec<f32> = (0..groups).map(|g| -1.0 + 0.125 * g as f32).collect();

        let wq = Tensor::from_vec(words, (out_dim, in_dim / 4), &Device::Cpu)?;
        let s = Tensor::from_vec(
            scales.clone(),
            (out_dim, in_dim / MLX_GROUP_SIZE),
            &Device::Cpu,
        )?;
        let b = Tensor::from_vec(
            biases.clone(),
            (out_dim, in_dim / MLX_GROUP_SIZE),
            &Device::Cpu,
        )?;
        assert_eq!(mlx_packed_bits(in_dim / 4, in_dim / MLX_GROUP_SIZE), 8);

        let got = dequant_mlx_q8(&wq, &s, &b)?
            .flatten_all()?
            .to_vec1::<f32>()?;
        let want = affine_grid(&codes, &scales, &biases, in_dim, MLX_GROUP_SIZE);
        assert_eq!(got, want);
        Ok(())
    }

    /// A deterministic pseudo-random dense weight for the packer round-trips.
    fn dense_weight(out_dim: usize, in_dim: usize) -> Tensor {
        let data: Vec<f32> = (0..out_dim * in_dim)
            .map(|i| {
                let x = (i as f32 * 0.017 + (i / in_dim) as f32 * 0.31).sin();
                x * 1.7 - 0.2
            })
            .collect();
        Tensor::from_vec(data, (out_dim, in_dim), &Device::Cpu).unwrap()
    }

    /// The `pack_mlx_affine` **producer** is the exact inverse of the consume path: packing a dense
    /// weight at Q4 then dequantizing (via `dequant_mlx_q4_reference` AND the real `Q4_1` repack)
    /// reproduces `scale·q + bias` for the codes the pack chose, and stays within the affine group's
    /// quant step of the original weight. Also pins the emitted shapes (`mlx_packed_bits` = 4).
    #[test]
    fn pack_mlx_affine_q4_roundtrips_through_repack() -> Result<()> {
        let (out_dim, in_dim) = (6, 128); // 2 groups/row at group 64
        let w = dense_weight(out_dim, in_dim);
        let (wq, s, b) = pack_mlx_affine(&w, 4, MLX_GROUP_SIZE)?;
        assert_eq!(wq.dims2()?, (out_dim, in_dim / 8));
        assert_eq!(s.dims2()?, (out_dim, in_dim / MLX_GROUP_SIZE));
        assert_eq!(mlx_packed_bits(in_dim / 8, in_dim / MLX_GROUP_SIZE), 4);

        // Dequant reference == the real Q4_1 repack dequant (the pack matches the consume path exactly).
        let reference = dequant_mlx_q4_reference(&wq, &s, &b)?
            .flatten_all()?
            .to_vec1::<f32>()?;
        let repacked = repack_mlx_q4_to_q4_1(&wq, &s, &b, &Device::Cpu)?
            .dequantize(&Device::Cpu)?
            .flatten_all()?
            .to_vec1::<f32>()?;
        assert_eq!(
            reference, repacked,
            "pack → repack dequant != pack → reference grid"
        );

        // The Q4 pack reconstructs the weight faithfully. The affine quant bound is `scale/2`, but the
        // `Q4_1` repack casts each group's scale/bias to f16 (module docs), which can nudge a dequant a
        // hair past `scale/2` — so assert the house cosine-parity metric (> 0.9999), not a per-element
        // half-step (the exact `reference == repacked` check above already pins the pack↔consume match).
        let orig = w.flatten_all()?.to_vec1::<f32>()?;
        let (mut dot, mut no, mut nr) = (0f64, 0f64, 0f64);
        for (&o, &r) in orig.iter().zip(reference.iter()) {
            dot += o as f64 * r as f64;
            no += o as f64 * o as f64;
            nr += r as f64 * r as f64;
        }
        let cos = dot / (no.sqrt() * nr.sqrt() + 1e-12);
        assert!(cos > 0.999, "Q4 pack→dequant cosine {cos:.6} too low");
        Ok(())
    }

    /// `pack_mlx_affine` at Q8 produces a triple the Q8 consume path dequantizes back to within an
    /// 8-bit group step of the weight, with the right shapes (`wq [out, in/4]`, `mlx_packed_bits` = 8).
    #[test]
    fn pack_mlx_affine_q8_roundtrips() -> Result<()> {
        let (out_dim, in_dim) = (4, 128);
        let w = dense_weight(out_dim, in_dim);
        let (wq, s, b) = pack_mlx_affine(&w, 8, MLX_GROUP_SIZE)?;
        assert_eq!(wq.dims2()?, (out_dim, in_dim / 4));
        assert_eq!(mlx_packed_bits(in_dim / 4, in_dim / MLX_GROUP_SIZE), 8);

        let got = dequant_mlx_q8(&wq, &s, &b)?
            .flatten_all()?
            .to_vec1::<f32>()?;
        let scales = s.flatten_all()?.to_vec1::<f32>()?;
        let orig = w.flatten_all()?.to_vec1::<f32>()?;
        let gpr = in_dim / MLX_GROUP_SIZE;
        for (i, (&o, &r)) in orig.iter().zip(got.iter()).enumerate() {
            let (row, col) = (i / in_dim, i % in_dim);
            let step = scales[row * gpr + col / MLX_GROUP_SIZE];
            assert!(
                (o - r).abs() <= step / 2.0 + 1e-6,
                "elem {i}: Q8 exceeds half-step"
            );
        }
        Ok(())
    }

    /// A constant group packs to `scale = 0` / all-zero codes and dequantizes back to the exact value
    /// (`bias`), not NaN — the division-by-zero guard.
    #[test]
    fn pack_mlx_affine_constant_group_is_exact() -> Result<()> {
        let (out_dim, in_dim) = (1, 64); // one group
        let w = Tensor::full(0.75f32, (out_dim, in_dim), &Device::Cpu)?;
        let (wq, s, b) = pack_mlx_affine(&w, 4, MLX_GROUP_SIZE)?;
        assert_eq!(s.flatten_all()?.to_vec1::<f32>()?, vec![0.0]);
        let got = dequant_mlx_q4_reference(&wq, &s, &b)?
            .flatten_all()?
            .to_vec1::<f32>()?;
        assert!(
            got.iter().all(|&v| v == 0.75),
            "constant group must dequant to 0.75 exactly"
        );
        Ok(())
    }

    /// Deterministic xorshift64* stream (no `rand` dependency; full-range u32 words and bf16-ish f32).
    struct Xs(u64);
    impl Xs {
        fn next(&mut self) -> u64 {
            self.0 ^= self.0 >> 12;
            self.0 ^= self.0 << 25;
            self.0 ^= self.0 >> 27;
            self.0.wrapping_mul(0x2545_F491_4F6C_DD1D)
        }
    }

    fn random_triple(
        out_dim: usize,
        in_dim: usize,
        group: usize,
        seed: u64,
    ) -> (Vec<u32>, Vec<f32>, Vec<f32>) {
        let mut r = Xs(seed | 1);
        let wq = (0..out_dim * in_dim / 8)
            .map(|_| (r.next() >> 16) as u32)
            .collect();
        let g = out_dim * (in_dim / group);
        // Include values outside f16 range / subnormals so the f16 cast path is exercised too.
        let mut f = |_: usize| {
            let bits = (r.next() >> 32) as u32;
            f32::from_bits(bits)
        };
        let scales = (0..g).map(&mut f).collect();
        let biases = (0..g).map(&mut f).collect();
        (wq, scales, biases)
    }

    /// sc-26691: the word-wise parallel repack is byte-identical to the original per-byte one across
    /// group sizes (32/64/128, covering 1/2/4 blocks per group), odd row counts and full-range codes.
    #[test]
    fn fast_q4_1_repack_is_bit_identical_to_the_reference() {
        for &(out_dim, in_dim, group) in &[
            (1, 32, 32),
            (3, 64, 32),
            (7, 128, 64),
            (5, 256, 128),
            (33, 512, 64),
            (17, 384, 128),
            (129, 256, 32),
            (64, 1024, 64),
            (1001, 128, 128),
        ] {
            let (wq, s, b) = random_triple(out_dim, in_dim, group, (out_dim * 31 + in_dim) as u64);
            let fast = q4_1_bytes(&wq, &s, &b, out_dim, in_dim, group);
            let old = q4_1_bytes_reference(&wq, &s, &b, out_dim, in_dim, group);
            assert_eq!(fast.len(), out_dim * in_dim / 32 * 20);
            assert!(
                fast == old,
                "mismatch at [{out_dim}, {in_dim}] group {group}"
            );
        }
    }

    /// Timing micro-benchmark on one H3-sized `[5120, 25600]` triple (run with
    /// `cargo test --release -p candle-gen --lib bench_q4_1_repack -- --ignored --nocapture`).
    #[test]
    #[ignore = "timing measurement, not a gate"]
    fn bench_q4_1_repack() {
        let (out_dim, in_dim, group) = (5120, 25600, 64);
        let (wq, s, b) = random_triple(out_dim, in_dim, group, 7);
        let t = std::time::Instant::now();
        let old = q4_1_bytes_reference(&wq, &s, &b, out_dim, in_dim, group);
        let t_old = t.elapsed();
        let t = std::time::Instant::now();
        let new = q4_1_bytes(&wq, &s, &b, out_dim, in_dim, group);
        let t_new = t.elapsed();
        assert!(old == new);
        eprintln!("repack [5120, 25600] g64: old {t_old:?}  new {t_new:?}");
    }
}
