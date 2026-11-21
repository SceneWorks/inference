//! A linear projection that is dense, group-wise quantized, Prism-packed or NVFP4.
//!
//! Moved to [`candle_layers::llm_projection`] (`SW-6957` / `SW-F335`) and re-exported here so all
//! internal decoders and external consumers remain 100% compatible.

pub use candle_layers::llm_projection::{
    ggml_bits, ggml_device_bytes, nvfp4_device_bytes, KvProjection, Projection, ProjectionCensus,
    ProjectionFormat, ProjectionKind, ProjectionTally, QuantSpec, WeightCensus,
};

#[cfg(test)]
mod tests {
    use super::*;
    use candle_core::quantized::GgmlDType;
    #[cfg(feature = "cuda")]
    use candle_core::DType;
    use candle_core::{Device, Tensor};

    fn ramp(rows: usize, cols: usize, seed: u64, device: &Device) -> Tensor {
        let mut x = seed;
        let data: Vec<f32> = (0..rows * cols)
            .map(|_| {
                x ^= x << 13;
                x ^= x >> 7;
                x ^= x << 17;
                ((x >> 40) as f32 / (1u64 << 24) as f32) * 2.0 - 1.0
            })
            .collect();
        Tensor::from_vec(data, (rows, cols), device).unwrap()
    }

    #[cfg(feature = "cuda")]
    fn rel_rms(got: &[f32], want: &[f32]) -> f32 {
        let num: f64 = got
            .iter()
            .zip(want)
            .map(|(g, w)| ((g - w) as f64).powi(2))
            .sum();
        let den: f64 = want.iter().map(|w| (*w as f64).powi(2)).sum();
        (num / den.max(1e-30)).sqrt() as f32
    }

    /// sc-19375: a 4-bit spec serves a 32- but not 256-aligned input dimension (YuE stage-2's
    /// 5504-wide `down_proj`) as Q4_0 instead of refusing the load; aligned dims and Q8 are unchanged.
    #[test]
    fn q4_falls_back_to_q4_0_for_non_256_aligned_inputs() {
        let q4 = QuantSpec::q4();
        assert_eq!(q4.dtype_for_in_dim(4096), GgmlDType::Q4K);
        assert_eq!(q4.dtype_for_in_dim(5504), GgmlDType::Q4_0);
        assert_eq!(q4.dtype_for_in_dim(100), GgmlDType::Q4K); // aligned to neither: quantizer refuses
        assert_eq!(QuantSpec::q8().dtype_for_in_dim(5504), GgmlDType::Q8_0);

        let p = Projection::load(ramp(8, 5504, 7, &Device::Cpu), Some(q4)).unwrap();
        assert_eq!(p.kind(), ProjectionKind::Ggml);
        let Projection::Quantized(q) = &p else {
            panic!("expected a quantized projection")
        };
        assert_eq!(q.ggml_dtype(), Some(GgmlDType::Q4_0));
        assert!(Projection::load(ramp(8, 100, 7, &Device::Cpu), Some(q4)).is_err());
    }

    /// sc-24135 AC2 at the primitive: an NVFP4 format for a CPU device is the typed refusal.
    #[test]
    fn nvfp4_format_on_cpu_is_the_typed_refusal() {
        match ProjectionFormat::nvfp4(&Device::Cpu) {
            Err(crate::Error::Nvfp4Refused(refusal)) => {
                assert!(matches!(
                    refusal,
                    candle_quant_kernels::Nvfp4Refusal::NotCudaDevice { .. }
                ));
                assert_eq!(refusal.capability(), "nvfp4");
            }
            Err(other) => panic!("expected Nvfp4Refused, got {other}"),
            Ok(_) => panic!("CPU cannot serve NVFP4"),
        }
    }

    #[test]
    fn census_counts_kinds_params_and_resident_bytes() {
        let dev = Device::Cpu;
        let dense = Projection::load(ramp(64, 32, 1, &dev), None).unwrap();
        let q8 = Projection::load(ramp(64, 64, 2, &dev), Some(QuantSpec::q8())).unwrap();
        assert_eq!(dense.kind(), ProjectionKind::Dense);
        assert_eq!(q8.kind(), ProjectionKind::Ggml);
        let mut census = ProjectionCensus::default();
        census.record(&dense);
        census.record(&q8);
        assert_eq!(census.dense.count, 1);
        assert_eq!(census.dense.params, 64 * 32);
        assert_eq!(census.dense.resident_bytes, 64 * 32 * 4, "f32 on CPU");
        assert_eq!(census.dense.bits_per_param(), Some(32.0));
        assert_eq!(census.ggml.count, 1);
        assert_eq!(census.ggml.params, 64 * 64);
        // Q8_0: 34 bytes per 32-element block = 8.5 bits/param.
        assert_eq!(census.ggml.bits_per_param(), Some(8.5));
        assert_eq!(census.nvfp4, ProjectionTally::default());
        assert_eq!(census.total().count, 2);
    }

    /// SW-7200: the census reads the resident tier from the GGML block width of every quantized
    /// projection; a dense LM head beside them does not make it mixed, two widths do.
    #[test]
    fn census_resident_quantization_follows_the_block_width() {
        use core_llm::ResidentQuantization as R;
        let dev = Device::Cpu;
        let dense = || Projection::load(ramp(64, 32, 1, &dev), None).unwrap();
        let q4 = || Projection::load(ramp(64, 256, 2, &dev), Some(QuantSpec::q4())).unwrap();
        let q4_0 = || Projection::load(ramp(64, 64, 3, &dev), Some(QuantSpec::q4())).unwrap();
        let q8 = || Projection::load(ramp(64, 64, 4, &dev), Some(QuantSpec::q8())).unwrap();
        let census = |ps: Vec<Projection>| {
            let mut c = ProjectionCensus::default();
            ps.iter().for_each(|p| c.record(p));
            c
        };
        assert_eq!(
            ProjectionCensus::default().resident_quantization(),
            R::Dense
        );
        assert_eq!(
            census(vec![dense(), dense()]).resident_quantization(),
            R::Dense
        );
        assert_eq!(
            census(vec![q4(), q4_0(), dense()]).resident_quantization(),
            R::Bits(4)
        );
        assert_eq!(
            census(vec![q8(), q8(), dense()]).resident_quantization(),
            R::Bits(8)
        );
        assert_eq!(census(vec![q4(), q8()]).resident_quantization(), R::Unknown);
        // Merging (a predictor's census) keeps the widths.
        let mut merged = census(vec![q8()]);
        merged.merge(&census(vec![q8(), dense()]));
        assert_eq!(merged.resident_quantization(), R::Bits(8));
        merged.merge(&census(vec![q4()]));
        assert_eq!(merged.resident_quantization(), R::Unknown);
        // NVFP4 alone, and Prism (no tier), from the tallies.
        let nvfp4 = ProjectionCensus {
            nvfp4: ProjectionTally {
                count: 3,
                ..Default::default()
            },
            ..Default::default()
        };
        assert_eq!(nvfp4.resident_quantization(), R::Nvfp4);
        let prism = ProjectionCensus {
            prism: ProjectionTally {
                count: 1,
                ..Default::default()
            },
            ..Default::default()
        };
        assert_eq!(prism.resident_quantization(), R::Unknown);
        // A GGML block type that is neither 4- nor 8-bit has no tier.
        assert_eq!(ggml_bits(GgmlDType::Q6K), None);
        assert_eq!(ggml_bits(GgmlDType::Q4K), Some(4));
        assert_eq!(ggml_bits(GgmlDType::Q8_0), Some(8));
    }

    /// sc-24140: outside NVFP4, `load_eligible` is exactly `load_as` — whatever the shape.
    #[test]
    fn load_eligible_is_load_as_for_dense_and_ggml() {
        let dev = Device::Cpu;
        let q8 = ProjectionFormat::from(QuantSpec::q8());
        for (rows, cols) in [(64usize, 64usize), (50, 64)] {
            for format in [None, Some(&q8)] {
                let a = Projection::load_eligible(ramp(rows, cols, 3, &dev), None, format).unwrap();
                let b = Projection::load_as(ramp(rows, cols, 3, &dev), None, format).unwrap();
                assert_eq!(a.kind(), b.kind(), "{rows}x{cols} {format:?}");
                assert_eq!(a.resident_bytes(), b.resident_bytes());
            }
        }
    }

    /// A CUDA device that meets the NVFP4 floor, or `None` (the GPU tests then skip loudly).
    #[cfg(feature = "cuda")]
    fn nvfp4_format() -> Option<(Device, ProjectionFormat)> {
        let device = crate::device::new_cuda_for_test().ok()?;
        let format = ProjectionFormat::nvfp4(&device).ok()?;
        Some((device, format))
    }

    /// sc-24135: `Projection::Nvfp4` forward against the dequantize-then-matmul reference over the
    /// *same* quantized operands (the NVFP4 weight read back from the device, the activation packed
    /// by the CPU codec), across the decode (M=1), odd and multi-tile M, a K that needs padding to
    /// the GEMM's 32-alignment, rank-3 input and a bias.
    ///
    /// Declared tolerances: rel-RMS ≤ 1e-2 against that reference (the residual is the bf16 output
    /// rounding and f32 accumulation order), and ≤ 0.2 against the exact dense product (the W4A4
    /// quantization error itself on uniform random operands).
    #[cfg(feature = "cuda")]
    #[test]
    #[ignore = "model inference; explicit model workflow only"]
    fn nvfp4_forward_matches_the_dequantize_then_matmul_reference() {
        let Some((device, format)) = nvfp4_format() else {
            candle_quant_kernels::skip_without_sm120("no sm_120 CUDA device");
            #[cfg(feature = "model-inference-proof")]
            crate::model_proof::missing_required_work("required work omitted at crates/llm/candle-llm/src/primitives/projection.rs:850 (nvfp4_forward_matches_the_dequantize_then_matmul_reference): This source branch exits a selected test/helper or skips a source-declared matrix arm because the prerequisite environment/path/device/format/weight/case is unavailable or deselected. Other successful rows do not satisfy this omitted row. Preserve optional selectors, but report the precise omitted cell before the existing exit.");
            return;
        };
        let (n, k) = (96usize, 80usize); // K=80 pads to 96 at quantization
        let w = ramp(n, k, 0x5c24_1350, &device)
            .to_dtype(DType::BF16)
            .unwrap();
        let bias = ramp(1, n, 0xb1a5, &device)
            .reshape(n)
            .unwrap()
            .to_dtype(DType::BF16)
            .unwrap();
        let p = Projection::load_as(w.clone(), Some(bias.clone()), Some(&format)).unwrap();
        assert_eq!(p.kind(), ProjectionKind::Nvfp4);
        assert!(p.is_quantized());
        let Projection::Nvfp4(nv) = &p else {
            unreachable!()
        };
        let wq = nv.to_host().unwrap(); // [n, 96] incl. zero K padding
        let w_deq = wq
            .dequantize()
            .unwrap()
            .narrow(1, 0, k)
            .unwrap()
            .to_dtype(DType::F32)
            .unwrap();
        let bias_f = bias
            .to_dtype(DType::F32)
            .unwrap()
            .to_device(&Device::Cpu)
            .unwrap();
        let w_exact = w
            .to_dtype(DType::F32)
            .unwrap()
            .to_device(&Device::Cpu)
            .unwrap();

        let _gemv_on = crate::primitives::nvfp4_path::nvfp4_gemv_policy_guard(Some(true));
        let flat_bf16 = |y: Tensor| {
            y.to_dtype(DType::F32)
                .unwrap()
                .flatten_all()
                .unwrap()
                .to_vec1::<f32>()
                .unwrap()
        };
        for m in [1usize, 5, 17, 130] {
            let x = ramp(m, k, 0xacdc + m as u64, &device)
                .to_dtype(DType::BF16)
                .unwrap();
            // The cuBLASLt W4A4 path at every M (the projection's dispatch sends M <= 8 to the
            // fused GEMV, sc-24136; that path is checked below against its own reference).
            let y = nv
                .forward(&x.unsqueeze(0).unwrap())
                .unwrap()
                .squeeze(0)
                .unwrap();
            assert_eq!(y.dims(), &[m, n]);
            assert_eq!(y.dtype(), DType::BF16);
            let got = flat_bf16(y);
            // Through the projection: the GEMV for M <= 8 (activation NOT quantized, so its
            // reference is the unquantized x against the same dequantized weight), cuBLASLt above.
            let y_proj = p
                .forward(&x.unsqueeze(0).unwrap())
                .unwrap()
                .squeeze(0)
                .unwrap();
            assert_eq!(y_proj.dims(), &[m, n]);

            let x_host = x
                .to_dtype(DType::F32)
                .unwrap()
                .to_device(&Device::Cpu)
                .unwrap();
            let xq = candle_quant_kernels::Nvfp4Tensor::pack(&x_host).unwrap();
            let x_deq = xq.dequantize().unwrap();
            let reference = x_deq
                .matmul(&w_deq.t().unwrap())
                .unwrap()
                .broadcast_add(&bias_f)
                .unwrap();
            let exact = x_host
                .matmul(&w_exact.t().unwrap())
                .unwrap()
                .broadcast_add(&bias_f)
                .unwrap();
            let flat = |t: Tensor| t.flatten_all().unwrap().to_vec1::<f32>().unwrap();
            let vs_ref = rel_rms(&got, &flat(reference));
            let vs_exact = rel_rms(&got, &flat(exact.clone()));
            assert!(
                vs_ref <= 1e-2,
                "M={m}: rel-RMS vs quantized reference {vs_ref}"
            );
            assert!(vs_exact <= 0.2, "M={m}: rel-RMS vs dense {vs_exact}");

            let got_proj = flat_bf16(y_proj);
            if m <= candle_quant_kernels::NVFP4_GEMV_MAX_ROWS {
                let unquantized_ref = x_host
                    .matmul(&w_deq.t().unwrap())
                    .unwrap()
                    .broadcast_add(&bias_f)
                    .unwrap();
                let vs_ref = rel_rms(&got_proj, &flat(unquantized_ref));
                assert!(
                    (vs_ref as f64) <= candle_quant_kernels::GEMV_REL_RMS_TOL,
                    "M={m}: GEMV rel-RMS vs unquantized-activation reference {vs_ref}"
                );
                // More accurate than W4A4 against the dense product (no activation quantization).
                assert!(rel_rms(&got_proj, &flat(exact)) < vs_exact, "M={m}");
            } else {
                assert_eq!(got_proj, got, "M={m}: the projection ran cuBLASLt");
            }
        }

        // Residency: packed nibbles + padded UE4M3 scales + the bf16 bias, far below bf16.
        let bytes = p.resident_bytes().unwrap();
        assert_eq!(bytes, (wq.packed.len() + wq.scales.len() + n * 2) as u64);
        assert!(bytes < (n * k * 2) as u64);

        #[cfg(feature = "model-inference-proof")]
        println!("\nMODEL_INFERENCE_BODY_COMPLETED primitives::projection::tests::nvfp4_forward_matches_the_dequantize_then_matmul_reference");
    }

    /// sc-24140: the llama-family loader's shape policy — under NVFP4, an eligible shape becomes
    /// NVFP4 and one the FP4 GEMM cannot serve (`N % 16 != 0`) stays dense, reported as `Dense`
    /// (so the census counts it under `dense`), bias included.
    #[cfg(feature = "cuda")]
    #[test]
    fn load_eligible_keeps_an_ineligible_nvfp4_shape_dense_and_visible() {
        let Some((device, format)) = nvfp4_format() else {
            candle_quant_kernels::skip_without_sm120("no sm_120 CUDA device");
            return;
        };
        let eligible = Projection::load_eligible(
            ramp(64, 32, 7, &device).to_dtype(DType::BF16).unwrap(),
            None,
            Some(&format),
        )
        .unwrap();
        assert_eq!(eligible.kind(), ProjectionKind::Nvfp4);
        let bias = ramp(1, 50, 9, &device).reshape(50).unwrap();
        let odd =
            Projection::load_eligible(ramp(50, 32, 7, &device), Some(bias), Some(&format)).unwrap();
        assert_eq!(odd.kind(), ProjectionKind::Dense);
        assert_eq!(odd.resident_bytes(), Some((50 * 32 + 50) * 4));
        let mut census = ProjectionCensus::default();
        census.record(&eligible);
        census.record(&odd);
        assert_eq!((census.nvfp4.count, census.dense.count), (1, 1));
        assert_eq!(census.dense.params, 50 * 32);
    }

    /// An NVFP4 request never keeps an ineligible projection dense: it is a typed refusal.
    #[cfg(feature = "cuda")]
    #[test]
    fn an_ineligible_shape_is_refused_not_kept_dense() {
        let Some((device, format)) = nvfp4_format() else {
            candle_quant_kernels::skip_without_sm120("no sm_120 CUDA device");
            return;
        };
        let w = ramp(50, 32, 7, &device);
        match Projection::load_as(w, None, Some(&format)) {
            Err(crate::Error::Nvfp4Refused(
                candle_quant_kernels::Nvfp4Refusal::ShapeIneligible { rows: 50, cols: 32 },
            )) => {}
            Err(other) => panic!("expected ShapeIneligible, got {other}"),
            Ok(p) => panic!("an ineligible NVFP4 projection loaded as {:?}", p.kind()),
        }
    }
}
