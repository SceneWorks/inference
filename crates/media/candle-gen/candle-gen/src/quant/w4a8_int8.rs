//! Native packed W4A8 projections, re-exported from [`candle_layers::w4a8_int8`] (`SW-6955`).

pub use candle_layers::w4a8_int8::*;

#[cfg(test)]
mod tests {
    use super::*;
    use crate::Weights;
    use candle_core::{DType, Device, Tensor};
    use std::collections::HashMap;

    fn operands() -> HashMap<String, Tensor> {
        let device = &Device::Cpu;
        let mut map = HashMap::new();
        let payload = br#"{"convrot_groupsize":256,"format":"asym_w4a8_int8","group_size":16}"#;
        map.insert(
            "p.comfy_quant".into(),
            Tensor::from_slice(payload, payload.len(), device).unwrap(),
        );
        map.insert(
            "p.weight".into(),
            Tensor::zeros((16, 128), DType::U8, device).unwrap(),
        );
        map.insert(
            "p.weight_s_rel".into(),
            Tensor::from_raw_buffer(&[0x38; 256], DType::F8E4M3, &[16, 16], device).unwrap(),
        );
        map.insert(
            "p.weight_s_channel".into(),
            Tensor::ones(16, DType::F32, device).unwrap(),
        );
        map.insert(
            "p.weight_codebook".into(),
            Tensor::from_vec((0..16).map(|v| v as f32 - 8.).collect(), 16, device).unwrap(),
        );
        map
    }

    #[test]
    fn refuses_orphaned_and_contradictory_operands() {
        let mut map = operands();
        map.remove("p.comfy_quant");
        let err = W4A8Linear::from_weights(&Weights::from_map(map), "p", false).unwrap_err();
        assert!(err.to_string().contains("authoritative"));
        for suffix in ["weight_scale", "scales", "biases", "weight_correction"] {
            let mut map = operands();
            map.insert(
                format!("p.{suffix}"),
                Tensor::ones(16, DType::F32, &Device::Cpu).unwrap(),
            );
            let err = W4A8Linear::from_weights(&Weights::from_map(map), "p", false).unwrap_err();
            assert!(err.to_string().contains("forbids"));
        }
    }

    #[test]
    fn refuses_noncanonical_descriptor_and_invalid_values_before_cuda() {
        let mut map = operands();
        let payload = br#"{"convrot_groupsize":256,"format":"asym_w4a8_int8","group_size":16,"convrot":true}"#;
        map.insert(
            "p.comfy_quant".into(),
            Tensor::from_slice(payload, payload.len(), &Device::Cpu).unwrap(),
        );
        let err = W4A8Linear::from_weights(&Weights::from_map(map), "p", false).unwrap_err();
        assert!(err.to_string().contains("descriptor"));
        let mut map = operands();
        map.insert(
            "p.weight_s_channel".into(),
            Tensor::zeros(16, DType::F32, &Device::Cpu).unwrap(),
        );
        let err = W4A8Linear::from_weights(&Weights::from_map(map), "p", false).unwrap_err();
        assert!(err.to_string().contains("finite positive F32"));
        let mut map = operands();
        map.insert(
            "p.weight_s_rel".into(),
            Tensor::from_raw_buffer(&[0x7f; 256], DType::F8E4M3, &[16, 16], &Device::Cpu).unwrap(),
        );
        let err = W4A8Linear::from_weights(&Weights::from_map(map), "p", false).unwrap_err();
        assert!(err.to_string().contains("non-negative finite E4M3FN"));
    }

    #[test]
    fn dense_projection_is_not_claimed_as_w4a8() {
        let mut map = HashMap::new();
        map.insert(
            "p.weight".into(),
            Tensor::zeros((16, 256), DType::F32, &Device::Cpu).unwrap(),
        );
        assert!(
            W4A8Linear::from_weights(&Weights::from_map(map), "p", false)
                .unwrap()
                .is_none()
        );
    }

    #[cfg(not(feature = "cuda"))]
    #[test]
    fn validated_w4a8_refuses_a_floating_fallback() {
        let err = W4A8Linear::from_weights(&Weights::from_map(operands()), "p", false).unwrap_err();
        assert!(err
            .to_string()
            .contains("requires native CUDA; no floating fallback"));
    }

    // ---- CUDA compute (SW-6134). The `W4A8Linear::forward` cases are canonical model proofs
    // (`scripts/ci/model-inference-proofs.json`); the raw activation-prep kernel parity case is in
    // the dispatch-only exhaustive suite (`scripts/ci/test_suites.json`). Run with `--release`.

    #[cfg(feature = "cuda")]
    fn gaussian(n: usize, seed: u64) -> Vec<f32> {
        let mut s = seed
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        let mut uniform = move || {
            s = s
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            ((s >> 11) as f64 + 0.5) / (1u64 << 53) as f64
        };
        (0..n)
            .map(|_| {
                let (u, v) = (uniform(), uniform());
                ((-2.0 * u.ln()).sqrt() * (2.0 * std::f64::consts::PI * v).cos()) as f32
            })
            .collect()
    }

    /// Pack a Gaussian `[n, k]` weight (DiT-like scale `1/sqrt(k)`) with the offline packer.
    #[cfg(feature = "cuda")]
    fn packed_weight(n: usize, k: usize, seed: u64) -> gen_core::w4a8_int8::W4A8Packed {
        let scale = (k as f32).sqrt().recip();
        let w: Vec<f32> = gaussian(n * k, seed).iter().map(|v| v * scale).collect();
        gen_core::w4a8_int8::quantize_weight(&w, n, k).expect("pack")
    }

    #[cfg(feature = "cuda")]
    fn run_path(
        lin: &W4A8Linear,
        x: &Tensor,
        fused: bool,
    ) -> (Tensor, candle_layers::fused::FusedTally) {
        use candle_layers::fused::{fused_tally, set_fused_kernels};
        set_fused_kernels(Some(fused));
        let start = fused_tally();
        let y = __model_proof_output!(lin.forward(x)).expect("forward");
        (y, fused_tally().since(&start))
    }

    /// The fused GEMM is an exact kernel relative to the W4A8 reference: integer accumulation is
    /// exact and the epilogue repeats the same F32 ops, so the two paths must agree **bit for bit**
    /// on every output dtype, with and without bias, including partial `M`/`N` edge tiles and
    /// Krea 2 projection widths. Also proves the path telemetry (`fused` vs `disabled`).
    #[cfg(feature = "cuda")]
    #[test]
    #[ignore = "model inference; explicit model-proof workflow"]
    fn w4a8_linear_gpu_fused_gemm_is_bit_identical_to_dequant_igemm() {
        use candle_layers::fused::{fused_policy_guard, REASON_DISABLED};
        let _policy = fused_policy_guard(None);
        let dev = Device::new_cuda(0).unwrap();
        // Both entry points compile through the seam with no per-thread local memory (sc-24164).
        {
            use candle_quant_kernels::w4a8_gemm::{
                W4A8_GEMM_FUNCTION, W4A8_GEMM_SRC, W4A8_QUANTIZE_FUNCTION,
            };
            let Device::Cuda(cuda) = &dev else {
                unreachable!()
            };
            let module = W4A8_GEMM_SRC.compiled(cuda).unwrap();
            for entry in [W4A8_GEMM_FUNCTION, W4A8_QUANTIZE_FUNCTION] {
                let f = module.function(entry).unwrap();
                let (local, regs) = (f.local_size_bytes().unwrap(), f.num_regs().unwrap());
                println!("{entry}: {local} B local, {regs} registers");
                assert_eq!(local, 0, "{entry} spills to local memory");
            }
        }
        let lt = std::sync::Arc::new(crate::quant::cublaslt::CublasLt::new(&dev).unwrap());
        // (m tokens, k in, n out, bias): edge tiles, then Krea 2 Turbo kv / out projections.
        let cases: &[(usize, usize, usize, bool)] = &[
            (1, 256, 16, false),
            (77, 512, 48, true),
            (130, 1024, 272, true),
            (300, 6144, 1536, true),
            (4096, 6144, 6144, false),
        ];
        for (i, &(m, k, n, with_bias)) in cases.iter().enumerate() {
            let p = packed_weight(n, k, 100 + i as u64);
            let bias = with_bias.then(|| {
                Tensor::from_vec(gaussian(n, 7 + i as u64), n, &dev)
                    .unwrap()
                    .to_dtype(DType::BF16)
                    .unwrap()
            });
            let lin = W4A8Linear::from_packed(&p, bias.as_ref(), lt.clone()).unwrap();
            let x32 = Tensor::from_vec(gaussian(m * k, 900 + i as u64), (1, m, k), &dev).unwrap();
            for dtype in [DType::BF16, DType::F16, DType::F32] {
                let x = x32.to_dtype(dtype).unwrap();
                let (fused, ft) = run_path(&lin, &x, true);
                let (reference, rt) = run_path(&lin, &x, false);
                // Two leaves per forward: the activation prep and the GEMM.
                assert_eq!((ft.fused, ft.reference), (2, 0), "fused path ran: {ft:?}");
                assert_eq!(
                    (rt.fused, rt.reference),
                    (0, 2),
                    "reference path ran: {rt:?}"
                );
                assert_eq!(rt.reference_reason, Some(REASON_DISABLED));
                assert_eq!(fused.dims(), &[1, m, n]);
                assert_eq!(fused.dtype(), dtype);
                let a = fused.to_dtype(DType::F32).unwrap().flatten_all().unwrap();
                let b = reference
                    .to_dtype(DType::F32)
                    .unwrap()
                    .flatten_all()
                    .unwrap();
                let (a, b) = (a.to_vec1::<f32>().unwrap(), b.to_vec1::<f32>().unwrap());
                let mismatches = a
                    .iter()
                    .zip(&b)
                    .filter(|(x, y)| x.to_bits() != y.to_bits())
                    .count();
                assert_eq!(
                    mismatches, 0,
                    "(m={m}, k={k}, n={n}, bias={with_bias}, {dtype:?}): fused != reference"
                );
            }
            println!("w4a8 fused == dequant+IGEMM bit-exact: m={m} k={k} n={n} bias={with_bias}");
        }
        #[cfg(feature = "model-inference-proof")]
        println!("\nMODEL_INFERENCE_BODY_COMPLETED quant::w4a8_int8::tests::w4a8_linear_gpu_fused_gemm_is_bit_identical_to_dequant_igemm");
    }

    /// The fused activation prep (ConvRot FHT + Kitchen row quantizer in one launch) is an exact
    /// kernel: its int8 codes are byte-identical and its row scales bit-identical to the eager
    /// reference (`convrot_fht_f32` then `quantize_kitchen_int8`) for every source dtype, on
    /// Gaussian rows, rows with heavy outliers, all-zero rows and rows of exact ties.
    #[cfg(feature = "cuda")]
    #[test]
    #[ignore = "CUDA compute: run by the validation workflow, not CI"]
    fn w4a8_linear_gpu_fused_activation_prep_is_byte_identical() {
        use candle_quant_kernels::w4a8_gemm::w4a8_quantize_activation;
        let dev = Device::new_cuda(0).unwrap();
        let lt = crate::quant::cublaslt::CublasLt::new(&dev).unwrap();
        for &(m, k) in &[(1usize, 256usize), (37, 1024), (300, 6144), (8, 16384)] {
            let mut v = gaussian(m * k, (m * 31 + k) as u64);
            // Row 0: a heavy outlier; last row: all zero; a middle row: exact small integers
            // (many quotient ties after rotation).
            v[3] = 4096.0;
            if m > 2 {
                for (j, e) in v[k..2 * k].iter_mut().enumerate() {
                    *e = ((j % 7) as f32 - 3.0) * 0.25;
                }
            }
            if m > 1 {
                v[(m - 1) * k..].iter_mut().for_each(|e| *e = 0.0);
            }
            let x32 = Tensor::from_vec(v, (m, k), &dev).unwrap();
            for dtype in [DType::BF16, DType::F16, DType::F32] {
                let x = x32.to_dtype(dtype).unwrap();
                let (q, xs) = w4a8_quantize_activation(&x).unwrap();
                let rotated = crate::quant::convrot::convrot_fht_f32(&x, 256).unwrap();
                let (rq, rxs) = lt.quantize_kitchen_int8(&rotated, dtype).unwrap();
                let a = q.flatten_all().unwrap().to_vec1::<u8>().unwrap();
                let b = rq.codes().flatten_all().unwrap().to_vec1::<u8>().unwrap();
                assert_eq!(a, b, "codes m={m} k={k} {dtype:?}");
                let a = xs.flatten_all().unwrap().to_vec1::<f32>().unwrap();
                let b = rxs.flatten_all().unwrap().to_vec1::<f32>().unwrap();
                assert!(
                    a.iter().zip(&b).all(|(x, y)| x.to_bits() == y.to_bits()),
                    "scales m={m} k={k} {dtype:?}"
                );
            }
            println!("w4a8 fused activation prep == FHT + kitchen quantizer: m={m} k={k}");
        }
    }

    /// Upload `bytes` at byte offset `off` of a fresh device allocation and return the contiguous
    /// `shape` view starting there (`narrow` of a flat buffer: `.contiguous()` does not copy it).
    #[cfg(feature = "cuda")]
    fn offset_u8_view(bytes: &[u8], off: usize, shape: (usize, usize), dev: &Device) -> Tensor {
        let mut flat = vec![0xA5u8; off];
        flat.extend_from_slice(bytes);
        let n = flat.len();
        let view = Tensor::from_vec(flat, n, dev)
            .unwrap()
            .narrow(0, off, bytes.len())
            .unwrap()
            .reshape(shape)
            .unwrap();
        assert!(view.is_contiguous());
        assert_eq!(view.layout().start_offset(), off);
        view
    }

    /// The fused GEMM's `uint4` staging needs the activation codes and the packed weight to start
    /// 16-byte aligned. A contiguous `U8` view at an odd byte offset (finding 4717213323: a
    /// `narrow` of a flat allocation from byte 1, reshaped) must be refused before launch with the
    /// typed `misaligned` reason — never a misaligned-address fault — and
    /// `w4a8_gemm_aligned` (what `W4A8Linear` applies to its resident weight at load) must make it
    /// runnable with output bit-identical to the aligned baseline. A nonzero but 16-byte aligned
    /// offset is served as is.
    #[cfg(feature = "cuda")]
    #[test]
    #[ignore = "CUDA compute: run by the validation workflow, not CI"]
    fn w4a8_gemm_gpu_refuses_misaligned_operand_views_and_realigns_them() {
        use candle_quant_kernels::w4a8_gemm::{
            w4a8_gemm, w4a8_gemm_aligned, w4a8_quantize_activation, W4A8GemmError,
            W4A8GemmOperands, W4A8GemmRefusal, W4A8VectorOperand,
        };
        let dev = Device::new_cuda(0).unwrap();
        let (m, k, n) = (77usize, 1024usize, 272usize);
        let p = packed_weight(n, k, 4242);
        let packed = Tensor::from_vec(p.packed.clone(), (n, k / 2), &dev).unwrap();
        let s_rel = Tensor::from_vec(p.s_rel.clone(), (n, k / 16), &dev).unwrap();
        let codebook = Tensor::from_vec(p.codebook.to_vec(), 16, &dev).unwrap();
        let s_channel = Tensor::from_vec(p.s_channel.clone(), n, &dev).unwrap();
        let bias = Tensor::from_vec(gaussian(n, 17), n, &dev).unwrap();
        let x = Tensor::from_vec(gaussian(m * k, 18), (m, k), &dev)
            .unwrap()
            .to_dtype(DType::BF16)
            .unwrap();
        let (qx, xs) = w4a8_quantize_activation(&x).unwrap();
        let gemm = |qx: &Tensor, packed: &Tensor| {
            w4a8_gemm(
                &W4A8GemmOperands {
                    qx,
                    xs: &xs,
                    packed,
                    s_rel: &s_rel,
                    codebook: &codebook,
                    s_channel: &s_channel,
                    bias: Some(&bias),
                },
                DType::BF16,
            )
        };
        let bits = |y: Tensor| {
            y.to_dtype(DType::F32)
                .unwrap()
                .flatten_all()
                .unwrap()
                .to_vec1::<f32>()
                .unwrap()
                .iter()
                .map(|v| v.to_bits())
                .collect::<Vec<u32>>()
        };
        let baseline = bits(gemm(&qx, &packed).unwrap());
        let qx_bytes = qx.flatten_all().unwrap().to_vec1::<u8>().unwrap();

        let mut cases = 0;
        for (operand, off) in [
            (W4A8VectorOperand::Activation, 1usize),
            (W4A8VectorOperand::Activation, 8),
            (W4A8VectorOperand::Activation, 16),
            (W4A8VectorOperand::PackedWeight, 3),
            (W4A8VectorOperand::PackedWeight, 32),
        ] {
            let (qx_v, packed_v) = match operand {
                W4A8VectorOperand::Activation => {
                    (offset_u8_view(&qx_bytes, off, (m, k), &dev), packed.clone())
                }
                W4A8VectorOperand::PackedWeight => {
                    (qx.clone(), offset_u8_view(&p.packed, off, (n, k / 2), &dev))
                }
            };
            if off % 16 == 0 {
                // Aligned at a nonzero offset: admitted, no copy, same bits.
                assert_eq!(
                    bits(gemm(&qx_v, &packed_v).unwrap()),
                    baseline,
                    "{operand:?} +{off}"
                );
                let kept = match operand {
                    W4A8VectorOperand::Activation => w4a8_gemm_aligned(&qx_v).unwrap(),
                    W4A8VectorOperand::PackedWeight => w4a8_gemm_aligned(&packed_v).unwrap(),
                };
                assert_eq!(kept.layout().start_offset(), off, "aligned view was copied");
            } else {
                match gemm(&qx_v, &packed_v) {
                    Err(W4A8GemmError::Refused(W4A8GemmRefusal::Misaligned {
                        operand: got,
                        addr_mod,
                    })) => {
                        assert_eq!(got, operand);
                        assert_eq!(addr_mod, off as u64 % 16);
                    }
                    other => {
                        panic!("{operand:?} +{off}: expected a misaligned refusal, got {other:?}")
                    }
                }
                let (qx_a, packed_a) = match operand {
                    W4A8VectorOperand::Activation => (w4a8_gemm_aligned(&qx_v).unwrap(), packed_v),
                    W4A8VectorOperand::PackedWeight => {
                        (qx_v, w4a8_gemm_aligned(&packed_v).unwrap())
                    }
                };
                assert_eq!(
                    bits(gemm(&qx_a, &packed_a).unwrap()),
                    baseline,
                    "{operand:?} +{off}"
                );
            }
            cases += 1;
            println!(
                "w4a8 gemm {operand:?} view at +{off} bytes: refused/realigned, bit-identical"
            );
        }
        assert_eq!(cases, 5);
    }

    /// `W4A8Linear::forward` (fused path) against the **CPU dequant reference linear**
    /// `x · decode_dense(packed)ᵀ + bias` in F32 on DiT-shaped (Krea 2 Turbo) weights. The gap is
    /// the activation's per-row int8 quantization plus the bf16 output rounding, so cosine must be
    /// `>= 0.999` (story acceptance); the measured values are printed.
    #[cfg(feature = "cuda")]
    #[test]
    #[ignore = "model inference; explicit model-proof workflow"]
    fn w4a8_linear_gpu_matches_cpu_dequant_reference_on_dit_shapes() {
        use candle_layers::fused::{fused_policy_guard, fused_tally};
        let _policy = fused_policy_guard(Some(true));
        let dev = Device::new_cuda(0).unwrap();
        let lt = std::sync::Arc::new(crate::quant::cublaslt::CublasLt::new(&dev).unwrap());
        let m = 64; // tokens: the CPU reference GEMM dominates the runtime
                    // Krea 2 Turbo: hidden 6144, GQA kv 1536, MLP 16384.
        for (i, &(k, n)) in [
            (6144usize, 6144usize),
            (6144, 1536),
            (6144, 16384),
            (16384, 6144),
        ]
        .iter()
        .enumerate()
        {
            let p = packed_weight(n, k, 40 + i as u64);
            let bias: Vec<f32> = gaussian(n, 3 + i as u64).iter().map(|v| v * 0.1).collect();
            let bias_t = Tensor::from_vec(bias.clone(), n, &dev).unwrap();
            let lin = W4A8Linear::from_packed(&p, Some(&bias_t), lt.clone()).unwrap();
            let x = Tensor::from_vec(gaussian(m * k, 70 + i as u64), (m, k), &dev)
                .unwrap()
                .to_dtype(DType::BF16)
                .unwrap();
            let start = fused_tally();
            let y = __model_proof_output!(lin.forward(&x)).unwrap();
            // Two fused leaves: the activation prep and the GEMM.
            assert_eq!(
                fused_tally().since(&start).fused,
                2,
                "the fused path served it"
            );
            // CPU reference linear, F32, on the bf16-rounded activation.
            let w = Tensor::from_vec(gen_core::w4a8_int8::decode_dense(&p), (n, k), &Device::Cpu)
                .unwrap();
            let xr = x
                .to_dtype(DType::F32)
                .unwrap()
                .to_device(&Device::Cpu)
                .unwrap();
            let want = xr
                .matmul(&w.t().unwrap())
                .unwrap()
                .broadcast_add(&Tensor::from_vec(bias, n, &Device::Cpu).unwrap())
                .unwrap();
            let got = y
                .to_dtype(DType::F32)
                .unwrap()
                .to_device(&Device::Cpu)
                .unwrap();
            let cos = crate::testkit::tensor_cosine(&got, &want);
            println!("w4a8 forward vs CPU dequant reference: k={k} n={n} m={m} cosine {cos:.6}");
            assert!(cos >= 0.999, "k={k} n={n}: cosine {cos}");
        }
        #[cfg(feature = "model-inference-proof")]
        println!("\nMODEL_INFERENCE_BODY_COMPLETED quant::w4a8_int8::tests::w4a8_linear_gpu_matches_cpu_dequant_reference_on_dit_shapes");
    }

    /// The offline packer's checkpoint tensor set loads through the production native-file loader
    /// (`Weights::from_files_native_int8`, I8/F8 bytes preserved) into a `W4A8Linear` whose output
    /// is bit-identical to the one built straight from the packer.
    #[cfg(feature = "cuda")]
    #[test]
    #[ignore = "model inference; explicit model-proof workflow"]
    fn w4a8_linear_gpu_checkpoint_round_trip_loads_and_matches() {
        let dev = Device::new_cuda(0).unwrap();
        let (m, k, n) = (33, 1024, 96);
        let p = packed_weight(n, k, 5);
        let tensors = p.checkpoint_tensors("blocks.0.attn.to_q");
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("w4a8.safetensors");
        gen_core::w4a8_int8::write_checkpoint(&tensors, &path).unwrap();
        let w = Weights::from_files_native_int8(&[&path], &dev).unwrap();
        let loaded = W4A8Linear::from_weights(&w, "blocks.0.attn.to_q", false)
            .unwrap()
            .expect("detected as W4A8");
        let lt = std::sync::Arc::new(crate::quant::cublaslt::CublasLt::new(&dev).unwrap());
        let direct = W4A8Linear::from_packed(&p, None, lt).unwrap();
        assert_eq!(loaded.descriptor(), direct.descriptor());
        let x = Tensor::from_vec(gaussian(m * k, 9), (m, k), &dev)
            .unwrap()
            .to_dtype(DType::BF16)
            .unwrap();
        let a = __model_proof_output!(loaded.forward(&x))
            .unwrap()
            .to_dtype(DType::F32)
            .unwrap();
        let b = __model_proof_output!(direct.forward(&x))
            .unwrap()
            .to_dtype(DType::F32)
            .unwrap();
        let diff = (a - b).unwrap().abs().unwrap().max_all().unwrap();
        assert_eq!(diff.to_scalar::<f32>().unwrap(), 0.0);
        #[cfg(feature = "model-inference-proof")]
        println!("\nMODEL_INFERENCE_BODY_COMPLETED quant::w4a8_int8::tests::w4a8_linear_gpu_checkpoint_round_trip_loads_and_matches");
    }
}
