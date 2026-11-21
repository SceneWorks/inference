//! Shared residual-capable linear projection, re-exported from [`candle_layers::adapt`] (`SW-6955`).

pub use candle_layers::adapt::*;

#[cfg(test)]
mod tests {
    use super::*;
    use crate::quant::{DenseLinear, QLinear, MLX_GROUP_SIZE};
    use crate::testkit::tensor_max_abs;
    use crate::train::lora::reconstruct_lokr_delta;
    use crate::CandleError;
    use candle_core::safetensors::MmapedSafetensors;
    use candle_core::{DType, Device, Tensor};
    use candle_nn::{Linear as CandleNnLinear, Module, VarBuilder};
    use gen_core::Quant;
    use std::collections::HashMap;
    use std::sync::{Arc, Mutex};

    /// Ownership only: seed valid CPU preparation entries directly, without a learned forward.
    /// A window must own its caches even for a same-device copy (CPU is the portable control).
    #[test]
    fn device_copy_owns_frozen_adapter_caches_without_mutating_the_source() {
        let device = Device::Cpu;
        let a = Tensor::from_vec(vec![1f32; 8], (4, 2), &device).unwrap();
        let b = Tensor::from_vec(vec![2f32; 8], (2, 4), &device).unwrap();
        let prepared = PreparedLora::default();
        prepared
            .by_dtype
            .lock()
            .unwrap()
            .push((DType::F32, a.clone(), b.clone()));
        let w1 = Tensor::from_vec(vec![1f32, 2., 3., 4.], (2, 2), &device).unwrap();
        let w2 = Tensor::from_vec(vec![2f32, 3., 5., 7.], (2, 2), &device).unwrap();
        let w2t = Tensor::from_vec(vec![2f32, 5., 3., 7.], (2, 2), &device).unwrap();
        let factors = LokrFactors {
            w1: w1.clone(),
            w2: w2.clone(),
            a: 2,
            b: 2,
            c: 2,
            d: 2,
            output_slice: Some((0, 4)),
            scale: 0.5,
            prepared: Arc::new(Mutex::new(vec![(DType::F32, w1.clone(), w2t.clone())])),
        };
        let source = AdaptLinear {
            base: Base::Dense(CandleNnLinear::new(
                Tensor::zeros((4, 4), DType::F32, &device).unwrap(),
                None,
            )),
            out_features: 4,
            in_features: 4,
            adapters: vec![
                Adapter::Lora {
                    a: a.clone(),
                    b: b.clone(),
                    scale: 0.25,
                    prepared,
                },
                Adapter::LokrStructured { factors },
            ],
        };
        let ordinary = source.clone();
        match (&source.adapters[0], &ordinary.adapters[0]) {
            (Adapter::Lora { prepared: a, .. }, Adapter::Lora { prepared: b, .. }) => {
                assert!(Arc::ptr_eq(&a.by_dtype, &b.by_dtype))
            }
            _ => panic!("frozen LoRA expected"),
        }
        match (&source.adapters[1], &ordinary.adapters[1]) {
            (Adapter::LokrStructured { factors: a }, Adapter::LokrStructured { factors: b }) => {
                assert!(Arc::ptr_eq(&a.prepared, &b.prepared))
            }
            _ => panic!("frozen LoKr expected"),
        }
        drop(ordinary);
        let copy = source.copy_to_device(&device).unwrap();
        assert_eq!(
            source.frozen_adapter_preparation_count(),
            2,
            "device-copy migration must not clear the retained source's prepared views"
        );
        assert_eq!(copy.frozen_adapter_preparation_count(), 0);
        let lora_owner = match (&source.adapters[0], &copy.adapters[0]) {
            (
                Adapter::Lora {
                    a: sa,
                    b: sb,
                    scale: ss,
                    prepared: sp,
                },
                Adapter::Lora {
                    a: ca,
                    b: cb,
                    scale: cs,
                    prepared: cp,
                },
            ) => {
                assert!(!Arc::ptr_eq(&sp.by_dtype, &cp.by_dtype));
                assert_eq!(ss, cs);
                assert_eq!(sa.to_vec2::<f32>().unwrap(), ca.to_vec2::<f32>().unwrap());
                assert_eq!(sb.to_vec2::<f32>().unwrap(), cb.to_vec2::<f32>().unwrap());
                assert!(ca.device().same_device(&device) && cb.device().same_device(&device));
                cp.by_dtype
                    .lock()
                    .unwrap()
                    .push((DType::F32, ca.clone(), cb.clone()));
                Arc::downgrade(&cp.by_dtype)
            }
            _ => panic!("frozen LoRA expected"),
        };
        let lokr_owner = match (&source.adapters[1], &copy.adapters[1]) {
            (Adapter::LokrStructured { factors: s }, Adapter::LokrStructured { factors: c }) => {
                assert!(!Arc::ptr_eq(&s.prepared, &c.prepared));
                assert_eq!(s.scale, c.scale);
                assert_eq!(s.output_slice, c.output_slice);
                assert_eq!(s.shape(), c.shape());
                assert_eq!(
                    s.w1.to_vec2::<f32>().unwrap(),
                    c.w1.to_vec2::<f32>().unwrap()
                );
                assert_eq!(
                    s.w2.to_vec2::<f32>().unwrap(),
                    c.w2.to_vec2::<f32>().unwrap()
                );
                c.prepared
                    .lock()
                    .unwrap()
                    .push((DType::F32, c.w1.clone(), w2t));
                Arc::downgrade(&c.prepared)
            }
            _ => panic!("frozen LoKr expected"),
        };
        assert_eq!(source.frozen_adapter_preparation_count(), 2);
        assert_eq!(copy.frozen_adapter_preparation_count(), 2);
        drop(copy);
        assert!(
            lora_owner.upgrade().is_none() && lokr_owner.upgrade().is_none(),
            "dropping the execution window must release both cache owners"
        );
        assert_eq!(source.frozen_adapter_preparation_count(), 2);
    }

    #[test]
    fn external_base_reuses_checked_residuals_without_a_fake_weight_or_duplicate_cache() {
        let dev = Device::Cpu;
        let mut external = AdaptLinear::for_external_base(3, 2, DType::F32, &dev).unwrap();
        assert!(external.is_quantized());
        assert_eq!(external.external_base_compute_dtype(), Some(DType::F32));
        assert_eq!(external.base_shape(), (2, 3));
        let x = Tensor::new(&[[1f32, 2., 4.], [-1., 0., 2.]], &dev).unwrap();
        let base = Tensor::new(&[[10f32, 20.], [30., 40.]], &dev).unwrap();
        assert!(external.forward(&x).is_err());
        let a = Tensor::new(&[[1f32], [-2.], [0.5]], &dev)
            .unwrap()
            .to_dtype(DType::BF16)
            .unwrap();
        let b = Tensor::new(&[[3f32, -5.]], &dev)
            .unwrap()
            .to_dtype(DType::BF16)
            .unwrap();
        external.push_lora_checked(a, b, 0.25).unwrap();
        assert_eq!(external.frozen_adapter_bytes(), 10);
        let expected = [9.25f32, 21.25, 30., 40.];
        for _ in 0..2 {
            let output = external
                .apply_residuals(&x, base.clone())
                .unwrap()
                .flatten_all()
                .unwrap()
                .to_vec1::<f32>()
                .unwrap();
            let errors: Vec<f32> = output
                .iter()
                .zip(expected)
                .map(|(a, b)| (a - b).abs())
                .collect();
            assert_eq!(errors.iter().copied().fold(0f32, f32::max), 0.);
            assert_eq!(errors.iter().sum::<f32>() / errors.len() as f32, 0.);
            // BF16 source factors + exactly ONE FP32 prepared pair, reused on request 2.
            assert_eq!(external.frozen_adapter_bytes(), 30);
        }
        external.clear_adapters();
        let restored = external.apply_residuals(&x, base.clone()).unwrap();
        assert_eq!(
            restored.flatten_all().unwrap().to_vec1::<f32>().unwrap(),
            base.flatten_all().unwrap().to_vec1::<f32>().unwrap()
        );
        assert_eq!(external.frozen_adapter_bytes(), 0);
    }

    #[test]
    fn external_packed_base_refuses_forward_folding_and_invalid_admission() {
        let dev = Device::Cpu;
        assert!(AdaptLinear::for_external_base(0, 2, DType::F32, &dev).is_err());
        assert!(AdaptLinear::for_external_base(3, 2, DType::U8, &dev).is_err());
        let mut external = AdaptLinear::for_external_base(3, 2, DType::F32, &dev).unwrap();
        let delta = Tensor::zeros((2, 3), DType::F32, &dev).unwrap();
        assert!(external.fold_dense_delta(&delta).is_err());
        assert!(external.quantize(Quant::Q4).is_err());
        assert!(external.quantize_onto(Quant::Q8, &dev).is_err());
        assert!(external.quantize_dequant_onto(Quant::Q4, &dev).is_err());
        let wrong_a = Tensor::zeros((4, 1), DType::F32, &dev).unwrap();
        let b = Tensor::zeros((1, 2), DType::F32, &dev).unwrap();
        assert!(external.push_lora(wrong_a, b, 1.).is_err());
        assert!(!external.is_adapted());
    }

    /// Test-side MLX Q4 packer (group 64): per-element 4-bit codes → u32 words (LSB-first nibbles).
    /// Returns `(wq [out, in/8] u32, scales [out, in/g], biases [out, in/g], affine grid [out, in])`.
    fn q4_packed(out_dim: usize, in_dim: usize) -> (Tensor, Tensor, Tensor, Vec<f32>) {
        let dev = Device::Cpu;
        let g = MLX_GROUP_SIZE;
        let codes: Vec<u8> = (0..out_dim * in_dim)
            .map(|i| ((i * 7 + i / 13) % 16) as u8)
            .collect();
        let gpr = in_dim / g;
        let groups = out_dim * gpr;
        // Small, BOUNDED scales/biases so the dequantized grid stays ~O(1). A large-magnitude grid
        // makes `base.forward` huge, and the residual-isolation test (adapted − base) then recovers a
        // tiny residual from a catastrophic f32 cancellation. Bounded via `% k` so it holds at any
        // group count.
        let scales: Vec<f32> = (0..groups)
            .map(|gi| 0.01 * ((gi % 5) as f32 + 1.0))
            .collect();
        let biases: Vec<f32> = (0..groups).map(|gi| -0.03 * (gi % 7) as f32).collect();
        let grid: Vec<f32> = (0..out_dim * in_dim)
            .map(|i| {
                let (row, col) = (i / in_dim, i % in_dim);
                let gi = row * gpr + col / g;
                scales[gi] * codes[i] as f32 + biases[gi]
            })
            .collect();
        let words: Vec<u32> = codes
            .chunks_exact(8)
            .map(|c| {
                c.iter()
                    .enumerate()
                    .fold(0u32, |acc, (i, &q)| acc | ((q as u32 & 0xF) << (4 * i)))
            })
            .collect();
        let wq = Tensor::from_vec(words, (out_dim, in_dim / 8), &dev).unwrap();
        let s = Tensor::from_vec(scales, (out_dim, gpr), &dev).unwrap();
        let b = Tensor::from_vec(biases, (out_dim, gpr), &dev).unwrap();
        (wq, s, b, grid)
    }

    /// A per-call unique suffix so parallel tests never share a temp file (`cargo test` runs threads in
    /// ONE process, so a pid-only name would collide — one test truncating/deleting the file another is
    /// mid-mmap on → corrupt reads / flaky failures).
    static TMP_COUNTER: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);

    /// Build a packed [`AdaptLinear`] via [`AdaptLinear::detect`] on a written `.weight`/`.scales`/
    /// `.biases` triple (the round-trip the DiT loader takes) plus the affine grid it represents.
    fn packed_adapt(
        tmp: &tempfile::TempDir,
        out_dim: usize,
        in_dim: usize,
    ) -> (AdaptLinear, Tensor) {
        let dev = Device::Cpu;
        let (wq, s, b, grid) = q4_packed(out_dim, in_dim);
        let mut map: HashMap<String, Tensor> = HashMap::new();
        map.insert("p.weight".into(), wq);
        map.insert("p.scales".into(), s);
        map.insert("p.biases".into(), b);
        let uniq = TMP_COUNTER.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        let tmp = tmp.path().join(format!("adapt_core_{}.safetensors", uniq));
        candle_core::safetensors::save(&map, &tmp).unwrap();
        // SAFETY: freshly written, single-reader for the test.
        let st = unsafe { MmapedSafetensors::new(&tmp).unwrap() };
        let vb = VarBuilder::from_backend(Box::new(st), DType::F32, dev.clone());
        let lin = AdaptLinear::detect(&vb, "p").unwrap();
        (
            lin,
            Tensor::from_vec(grid, (out_dim, in_dim), &dev).unwrap(),
        )
    }

    fn deterministic_weight(out_dim: usize, in_dim: usize, dtype: DType) -> Tensor {
        let values: Vec<f32> = (0..out_dim * in_dim)
            .map(|i| ((i % 17) as f32 - 8.0) / 16.0)
            .collect();
        Tensor::from_vec(values, (out_dim, in_dim), &Device::Cpu)
            .unwrap()
            .to_dtype(dtype)
            .unwrap()
    }

    fn deterministic_input(rows: usize, in_dim: usize, dtype: DType) -> Tensor {
        let values: Vec<f32> = (0..rows * in_dim)
            .map(|i| ((i % 11) as f32 + 1.0) / 8.0)
            .collect();
        Tensor::from_vec(values, (rows, in_dim), &Device::Cpu)
            .unwrap()
            .to_dtype(dtype)
            .unwrap()
    }

    fn dense_bf16_adapt(out_dim: usize, in_dim: usize) -> AdaptLinear {
        AdaptLinear::from_dense(
            CandleNnLinear::new(deterministic_weight(out_dim, in_dim, DType::BF16), None),
            in_dim,
            out_dim,
        )
    }

    fn int8_fast_adapt(quant: Quant, out_dim: usize, in_dim: usize) -> AdaptLinear {
        let mut base = QLinear::from_dense(DenseLinear::Linear(CandleNnLinear::new(
            deterministic_weight(out_dim, in_dim, DType::F32),
            None,
        )));
        // `cast_back=false` is the production SAM3-style host regime: a bf16 activation is promoted
        // for QMatMul and the host result stays f32. This is the dtype boundary the residual must
        // follow, rather than assuming the input dtype is also the host-output dtype.
        base.quantize_int8_fast(quant, false, false, false).unwrap();
        AdaptLinear::from_packed(base, in_dim, out_dim)
    }

    fn poison_lora(in_dim: usize, out_dim: usize) -> (Tensor, Tensor) {
        let a =
            Tensor::from_vec(vec![f32::INFINITY; in_dim * 2], (in_dim, 2), &Device::Cpu).unwrap();
        let b = Tensor::ones((2, out_dim), DType::F32, &Device::Cpu).unwrap();
        (a, b)
    }

    fn exact_values(t: &Tensor) -> Vec<Vec<f32>> {
        t.to_dtype(DType::F32).unwrap().to_vec2::<f32>().unwrap()
    }

    fn assert_tensor_exact(actual: &Tensor, expected: &Tensor, context: &str) {
        assert_eq!(actual.dtype(), expected.dtype(), "{context}: dtype differs");
        assert_eq!(
            exact_values(actual),
            exact_values(expected),
            "{context}: element values differ"
        );
    }

    /// sc-15444: a strength-zero LoRA is not merely numerically close to disabled: its factors must
    /// not be evaluated. Poisoned factors make the distinction observable (∞·0 would become NaN).
    /// The ordinary dense forward and the bf16-store/f32-compute `forward_upcast` must return the
    /// exact no-adapter values and dtype. Candle's CPU backend does not implement bf16 matmul, so the
    /// ordinary-forward leg uses its production f32 regime; the bf16-resident leg is exercised through
    /// the public upcast entry point used for that storage dtype.
    #[test]
    #[ignore = "model inference; explicit model workflow only"]
    fn dense_scale_zero_skips_residual_exactly_in_forward_and_bf16_upcast() {
        let (out_dim, in_dim) = (16usize, 32usize);
        let x_f32 = deterministic_input(2, in_dim, DType::F32);
        let bare_forward_host = AdaptLinear::from_dense(
            CandleNnLinear::new(deterministic_weight(out_dim, in_dim, DType::F32), None),
            in_dim,
            out_dim,
        );
        let bare_forward = bare_forward_host.forward(&x_f32).unwrap();

        let mut zero_forward_host = AdaptLinear::from_dense(
            CandleNnLinear::new(deterministic_weight(out_dim, in_dim, DType::F32), None),
            in_dim,
            out_dim,
        );
        let (a, b) = poison_lora(in_dim, out_dim);
        zero_forward_host.push_lora(a, b, 0.0).unwrap();
        let zero_forward = zero_forward_host.forward(&x_f32).unwrap();
        assert_tensor_exact(&zero_forward, &bare_forward, "dense f32 forward scale=0");

        let poison_w1 = Tensor::from_vec(vec![f32::INFINITY; 4 * 4], (4, 4), &Device::Cpu).unwrap();
        let poison_w2 = Tensor::ones((4, 8), DType::F32, &Device::Cpu).unwrap();
        let factors = LokrFactors::build(
            0.0,
            (out_dim, in_dim),
            Some(&poison_w1),
            None,
            None,
            Some(&poison_w2),
            None,
            None,
            None,
        )
        .unwrap()
        .unwrap();
        let mut zero_lokr_host = AdaptLinear::from_dense(
            CandleNnLinear::new(deterministic_weight(out_dim, in_dim, DType::F32), None),
            in_dim,
            out_dim,
        );
        zero_lokr_host.push_lokr_structured(factors).unwrap();
        assert_tensor_exact(
            &zero_lokr_host.forward(&x_f32).unwrap(),
            &bare_forward,
            "dense structured LoKr scale=0",
        );

        let bare_bf16 = dense_bf16_adapt(out_dim, in_dim);
        let bare_upcast = bare_bf16.forward_upcast(&x_f32).unwrap();
        let mut zero_bf16 = dense_bf16_adapt(out_dim, in_dim);
        let (a, b) = poison_lora(in_dim, out_dim);
        zero_bf16.push_lora(a, b, 0.0).unwrap();
        let zero_upcast = zero_bf16.forward_upcast(&x_f32).unwrap();
        assert_tensor_exact(
            &zero_upcast,
            &bare_upcast,
            "dense bf16 forward_upcast scale=0",
        );

        #[cfg(feature = "model-inference-proof")]
        println!("\nMODEL_INFERENCE_BODY_COMPLETED quant::adapt::tests::dense_scale_zero_skips_residual_exactly_in_forward_and_bf16_upcast");
    }

    /// sc-15444: Q8/Q4 int8-fast hosts can intentionally retain an f32 result for non-f32 inputs. A
    /// live residual must be cast to that host-result dtype before addition; a zero residual must be
    /// skipped. The expected output is constructed with the same explicit host-dtype seam, then
    /// compared element-for-element for both public forward entry points. CPU Candle does not implement
    /// bf16 matmul, so f64 is the executable CPU proxy for the same `input dtype != host output dtype`
    /// boundary (the production bf16/CUDA path crosses that identical boundary).
    #[test]
    #[ignore = "model inference; explicit model workflow only"]
    fn quantized_hosts_preserve_output_dtype_and_exact_adapter_semantics() {
        let (out_dim, in_dim, rank) = (32usize, 32usize, 2usize);
        let x = deterministic_input(2, in_dim, DType::F64);
        let a = Tensor::ones((in_dim, rank), DType::F32, &Device::Cpu).unwrap();
        let b = Tensor::ones((rank, out_dim), DType::F32, &Device::Cpu).unwrap();
        let residual = ((x
            .broadcast_matmul(&a.to_dtype(DType::F64).unwrap())
            .unwrap()
            .broadcast_matmul(&b.to_dtype(DType::F64).unwrap())
            .unwrap())
            * 0.5)
            .unwrap()
            .to_dtype(DType::F32)
            .unwrap();
        assert_eq!(
            exact_values(&residual),
            vec![vec![23.375; out_dim], vec![23.5; out_dim]],
            "the deterministic live adapter must inject the measured positive residual"
        );

        for quant in [Quant::Q8, Quant::Q4] {
            let bare = int8_fast_adapt(quant, out_dim, in_dim);
            let bare_forward = bare.forward(&x).unwrap();
            assert_eq!(bare_forward.dtype(), DType::F32);

            let mut zero = int8_fast_adapt(quant, out_dim, in_dim);
            let (poison_a, poison_b) = poison_lora(in_dim, out_dim);
            zero.push_lora(poison_a, poison_b, 0.0).unwrap();
            assert_tensor_exact(
                &zero.forward(&x).unwrap(),
                &bare_forward,
                "quantized forward scale=0",
            );
            assert_tensor_exact(
                &zero.forward_upcast(&x).unwrap(),
                &bare_forward,
                "quantized forward_upcast scale=0",
            );

            let mut live = int8_fast_adapt(quant, out_dim, in_dim);
            live.push_lora(a.clone(), b.clone(), 0.5).unwrap();
            let expected = (&bare_forward + &residual).unwrap();
            let got = live.forward(&x).unwrap();
            assert_tensor_exact(&got, &expected, "quantized forward live adapter");
            assert_tensor_exact(
                &live.forward_upcast(&x).unwrap(),
                &expected,
                "quantized forward_upcast live adapter",
            );
            assert!(
                exact_values(&got)
                    .iter()
                    .flatten()
                    .zip(exact_values(&bare_forward).iter().flatten())
                    .all(|(adapted, base)| adapted > base),
                "positive live adapter must shift every host output upward"
            );
        }

        #[cfg(feature = "model-inference-proof")]
        println!("\nMODEL_INFERENCE_BODY_COMPLETED quant::adapt::tests::quantized_hosts_preserve_output_dtype_and_exact_adapter_semantics");
    }

    /// The hardware counterpart to the CPU dtype-boundary test: Metal supports bf16 dense matmul, so
    /// this runs the production activation dtype directly across dense, Q8, and Q4 hosts. It is gated
    /// with the backend feature and is exercised by the macOS validation lane.
    #[cfg(feature = "metal")]
    #[test]
    #[ignore = "model inference; explicit model workflow only"]
    fn metal_bf16_dense_q8_q4_forwards_are_exact_at_the_adapter_seam() {
        let dev = Device::new_metal(0).unwrap();
        let (out_dim, in_dim, rank) = (32usize, 32usize, 2usize);
        let x = deterministic_input(2, in_dim, DType::BF16)
            .to_device(&dev)
            .unwrap();
        let dense_weight = deterministic_weight(out_dim, in_dim, DType::BF16)
            .to_device(&dev)
            .unwrap();
        let bare_dense = AdaptLinear::from_dense(
            CandleNnLinear::new(dense_weight.clone(), None),
            in_dim,
            out_dim,
        );
        let bare_dense_y = bare_dense.forward(&x).unwrap();
        assert_eq!(bare_dense_y.dtype(), DType::BF16);

        let mut zero_dense =
            AdaptLinear::from_dense(CandleNnLinear::new(dense_weight, None), in_dim, out_dim);
        let poison_a = Tensor::full(f32::INFINITY, (in_dim, rank), &dev).unwrap();
        let poison_b = Tensor::ones((rank, out_dim), DType::F32, &dev).unwrap();
        zero_dense
            .push_lora(poison_a, poison_b, 0.0)
            .expect("scale=0 poison factors are admitted on a dense bf16 host");
        assert_tensor_exact(
            &zero_dense.forward(&x).unwrap(),
            &bare_dense_y,
            "Metal dense bf16 forward scale=0",
        );
        assert_tensor_exact(
            &zero_dense.forward_upcast(&x).unwrap(),
            &bare_dense_y,
            "Metal dense bf16 forward_upcast scale=0",
        );

        let a = Tensor::ones((in_dim, rank), DType::F32, &dev).unwrap();
        let b = Tensor::ones((rank, out_dim), DType::F32, &dev).unwrap();
        let residual_bf16 = ((x
            .broadcast_matmul(&a.to_dtype(DType::BF16).unwrap())
            .unwrap()
            .broadcast_matmul(&b.to_dtype(DType::BF16).unwrap())
            .unwrap())
            * 0.5)
            .unwrap();
        let mut live_dense = AdaptLinear::from_dense(
            CandleNnLinear::new(
                deterministic_weight(out_dim, in_dim, DType::BF16)
                    .to_device(&dev)
                    .unwrap(),
                None,
            ),
            in_dim,
            out_dim,
        );
        live_dense
            .push_lora(a.clone(), b.clone(), 0.5)
            .expect("rank-2 factors are admitted on a dense bf16 host");
        let dense_expected = (&bare_dense_y + &residual_bf16).unwrap();
        assert_tensor_exact(
            &live_dense.forward(&x).unwrap(),
            &dense_expected,
            "Metal dense bf16 live adapter",
        );
        assert_tensor_exact(
            &live_dense.forward_upcast(&x).unwrap(),
            &dense_expected,
            "Metal dense bf16 live adapter upcast entry",
        );

        for quant in [Quant::Q8, Quant::Q4] {
            let build = || {
                let weight = deterministic_weight(out_dim, in_dim, DType::F32)
                    .to_device(&dev)
                    .unwrap();
                let mut base =
                    QLinear::from_dense(DenseLinear::Linear(CandleNnLinear::new(weight, None)));
                base.quantize_int8_fast(quant, false, false, false).unwrap();
                AdaptLinear::from_packed(base, in_dim, out_dim)
            };

            let bare = build();
            let base_y = bare.forward(&x).unwrap();
            assert_eq!(base_y.dtype(), DType::F32);

            let mut zero = build();
            let poison_a = Tensor::full(f32::INFINITY, (in_dim, rank), &dev).unwrap();
            let poison_b = Tensor::ones((rank, out_dim), DType::F32, &dev).unwrap();
            zero.push_lora(poison_a, poison_b, 0.0)
                .expect("scale=0 poison factors are admitted on a packed host");
            assert_tensor_exact(
                &zero.forward(&x).unwrap(),
                &base_y,
                "Metal quantized bf16 forward scale=0",
            );
            assert_tensor_exact(
                &zero.forward_upcast(&x).unwrap(),
                &base_y,
                "Metal quantized bf16 forward_upcast scale=0",
            );

            let mut live = build();
            live.push_lora(a.clone(), b.clone(), 0.5)
                .expect("rank-2 factors are admitted on a packed host");
            let expected = (&base_y + residual_bf16.to_dtype(DType::F32).unwrap()).unwrap();
            assert_tensor_exact(
                &live.forward(&x).unwrap(),
                &expected,
                "Metal quantized bf16 live adapter",
            );
            assert_tensor_exact(
                &live.forward_upcast(&x).unwrap(),
                &expected,
                "Metal quantized bf16 live adapter upcast entry",
            );
        }

        #[cfg(feature = "model-inference-proof")]
        println!("\nMODEL_INFERENCE_BODY_COMPLETED quant::adapt::tests::metal_bf16_dense_q8_q4_forwards_are_exact_at_the_adapter_seam");
    }

    /// `detect` recovers the logical dims from the packed `scales` shape and keeps the base packed.
    #[test]
    fn detect_recovers_dims_and_stays_packed() {
        let tmp = tempfile::tempdir().unwrap();
        let (out_dim, in_dim) = (64usize, 128usize); // in divisible by group 64
        let (lin, _grid) = packed_adapt(&tmp, out_dim, in_dim);
        assert!(lin.is_packed(), "`.scales` ⇒ packed base");
        assert_eq!(
            lin.base_shape(),
            (out_dim, in_dim),
            "dims from scales shape"
        );
        assert_eq!(lin.in_features(), in_dim);
        assert_eq!(lin.out_features(), out_dim);
        assert!(!lin.is_adapted());
    }

    /// The dense arm of `linear_detect` is byte-identical to the legacy `candle_nn::linear` read — a
    /// dense checkpoint (no `.scales`) loads dense, unchanged; a `.scales` sibling fires the packed arm.
    #[test]
    #[ignore = "model inference; explicit model workflow only"]
    fn linear_detect_dense_and_packed_arms() {
        let dev = Device::Cpu;
        let (out_dim, in_dim) = (32usize, 64usize);
        // Dense arm: byte-identical to `Linear::new`.
        let w = Tensor::randn(0f32, 1f32, (out_dim, in_dim), &dev).unwrap();
        let bias = Tensor::randn(0f32, 1f32, (out_dim,), &dev).unwrap();
        let (wq, s, b, _grid) = q4_packed(out_dim, in_dim);
        let mut map: HashMap<String, Tensor> = HashMap::new();
        map.insert("d.weight".into(), w.clone());
        map.insert("d.bias".into(), bias.clone());
        map.insert("p.weight".into(), wq);
        map.insert("p.scales".into(), s);
        map.insert("p.biases".into(), b);
        let uniq = TMP_COUNTER.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        let tmp_guard = tempfile::tempdir().unwrap();
        let tmp = tmp_guard
            .path()
            .join(format!("adapt_core_{}.safetensors", uniq));
        candle_core::safetensors::save(&map, &tmp).unwrap();
        // SAFETY: freshly written, single-reader.
        let st = unsafe { MmapedSafetensors::new(&tmp).unwrap() };
        let vb = VarBuilder::from_backend(Box::new(st), DType::F32, dev.clone());

        let dense = AdaptLinear::linear_detect(in_dim, out_dim, &vb, "d", true).unwrap();
        assert!(!dense.is_packed(), "no `.scales` ⇒ dense");
        let x = Tensor::randn(0f32, 1f32, (4usize, in_dim), &dev).unwrap();
        let want = CandleNnLinear::new(w, Some(bias));
        let dev_max = (dense.forward(&x).unwrap()
            - __model_proof_output!(want.forward(&x)).unwrap())
        .unwrap()
        .abs()
        .unwrap()
        .max_all()
        .unwrap()
        .to_scalar::<f32>()
        .unwrap();
        assert_eq!(
            dev_max, 0.0,
            "dense arm deviates from the legacy linear read"
        );

        let packed = AdaptLinear::linear_detect(in_dim, out_dim, &vb, "p", false).unwrap();
        assert!(
            packed.is_packed(),
            "`.scales` ⇒ packed load, not a silent dense fallback"
        );

        #[cfg(feature = "model-inference-proof")]
        println!("\nMODEL_INFERENCE_BODY_COMPLETED quant::adapt::tests::linear_detect_dense_and_packed_arms");
    }

    #[test]
    fn precomputed_base_residual_seam_preserves_output_and_stack() {
        let dev = Device::Cpu;
        let weight = Tensor::new(&[[1f32, 2.], [-3., 4.]], &dev).unwrap();
        let base = CandleNnLinear::new(weight, None);
        let x = Tensor::new(&[[0.5f32, -2.], [3., 0.25]], &dev).unwrap();
        let bare = base.forward(&x).unwrap();
        let mut linear = AdaptLinear::from_dense(base, 2, 2);
        let bits = |t: Tensor| {
            t.flatten_all()
                .unwrap()
                .to_vec1::<f32>()
                .unwrap()
                .into_iter()
                .map(f32::to_bits)
                .collect::<Vec<_>>()
        };
        assert_eq!(
            bits(linear.apply_residuals(&x, bare.clone()).unwrap()),
            bits(bare.clone())
        );
        let a = Tensor::new(&[[0.25f32], [-0.5]], &dev).unwrap();
        let b = Tensor::new(&[[2f32, -1.]], &dev).unwrap();
        linear.push_lora(a, b, -0.7).unwrap();
        let actual = linear
            .apply_residuals(&x, bare.clone())
            .unwrap()
            .flatten_all()
            .unwrap()
            .to_vec1::<f32>()
            .unwrap();
        for (got, want) in actual.iter().zip([-5.075f32, -8.7125, 2.625, -7.5625]) {
            assert!(
                (got - want).abs() < 1e-6,
                "independent bottleneck arithmetic: {got} != {want}"
            );
        }
        assert_eq!(
            bits(linear.forward(&x).unwrap()),
            bits(linear.apply_residuals(&x, bare.clone()).unwrap())
        );
        let w1 = Tensor::new(&[[0.5f32]], &dev).unwrap();
        let w2 = Tensor::new(&[[1f32, 0.25], [-0.2, 0.7]], &dev).unwrap();
        let factors = LokrFactors::build(
            0.3,
            (2, 2),
            Some(&w1),
            None,
            None,
            Some(&w2),
            None,
            None,
            None,
        )
        .unwrap()
        .unwrap();
        linear.push_lokr_structured(factors).unwrap();
        let actual = linear
            .apply_residuals(&x, bare.clone())
            .unwrap()
            .flatten_all()
            .unwrap()
            .to_vec1::<f32>()
            .unwrap();
        for (got, want) in actual
            .iter()
            .zip([-5.075f32, -8.9375, 3.084_375, -7.626_25])
        {
            assert!(
                (got - want).abs() < 2e-6,
                "independent LoKr arithmetic: {got} != {want}"
            );
        }
        assert_eq!(
            bits(linear.forward(&x).unwrap()),
            bits(linear.apply_residuals(&x, bare).unwrap())
        );
        let wrong = Tensor::zeros((3, 3), DType::F32, &dev).unwrap();
        assert!(linear.apply_residuals(&wrong, wrong.clone()).is_err());
    }

    /// The additive LoRA residual `scale·((x·a)·b)` reproduces the **folded** `x·(W + δ)ᵀ` with
    /// `δ = (alpha/rank)·scale·(up·down)` on a dense f32 base — the additive==folded identity (tight in
    /// f32), the core weight-level property (no GPU needed).
    #[test]
    #[ignore = "model inference; explicit model workflow only"]
    fn additive_lora_matches_folded_dense() {
        let dev = Device::Cpu;
        let (out_dim, in_dim, rank) = (48usize, 64usize, 4usize);
        let w = Tensor::randn(0f32, 1f32, (out_dim, in_dim), &dev).unwrap();
        let down = Tensor::randn(0f32, 1f32, (rank, in_dim), &dev).unwrap(); // A [rank, in]
        let up = Tensor::randn(0f32, 1f32, (out_dim, rank), &dev).unwrap(); // B [out, rank]
        let (alpha, user_scale) = (8.0f64, 0.7f64);
        let ratio = alpha / rank as f64;

        // a = downᵀ [in, rank]; b = (upᵀ·ratio) [rank, out].
        let a = down.t().unwrap().contiguous().unwrap();
        let b = (up.t().unwrap().contiguous().unwrap() * ratio).unwrap();
        let mut lin =
            AdaptLinear::from_dense(CandleNnLinear::new(w.clone(), None), in_dim, out_dim);
        lin.push_lora(a, b, user_scale).unwrap();
        assert!(lin.is_adapted());

        // Folded reference: δ = ratio·user_scale·(up·down); W_merged = W + δ.
        let delta = ((up.matmul(&down).unwrap()) * (ratio * user_scale)).unwrap();
        let folded = CandleNnLinear::new((w + delta).unwrap(), None);

        let x = Tensor::randn(0f32, 1f32, (3usize, in_dim), &dev).unwrap();
        let dev_max = (lin.forward(&x).unwrap()
            - __model_proof_output!(folded.forward(&x)).unwrap())
        .unwrap()
        .abs()
        .unwrap()
        .max_all()
        .unwrap()
        .to_scalar::<f32>()
        .unwrap();
        assert!(dev_max < 1e-4, "additive vs folded deviates by {dev_max}");

        #[cfg(feature = "model-inference-proof")]
        println!("\nMODEL_INFERENCE_BODY_COMPLETED quant::adapt::tests::additive_lora_matches_folded_dense");
    }

    /// A LoRA applied additively onto a **packed** base shifts the output, keeps the base **packed** (no
    /// dense weight materialized), and stays finite — the core acceptance on a quantized tier. A scale-0
    /// residual is an exact no-op (the mutation anchor: break the scale and this equality breaks).
    #[test]
    #[ignore = "model inference; explicit model workflow only"]
    fn additive_lora_on_packed_shifts_stays_packed_and_scale0_is_noop() {
        let tmp = tempfile::tempdir().unwrap();
        let dev = Device::Cpu;
        let (out_dim, in_dim, rank) = (64usize, 128usize, 8usize);
        let (packed_base, _grid) = packed_adapt(&tmp, out_dim, in_dim);
        assert!(packed_base.is_packed());

        let (mut adapted, _) = packed_adapt(&tmp, out_dim, in_dim);
        let a = (Tensor::randn(0f32, 1f32, (in_dim, rank), &dev).unwrap() * 0.1).unwrap();
        let b = (Tensor::randn(0f32, 1f32, (rank, out_dim), &dev).unwrap() * 0.1).unwrap();
        adapted.push_lora(a.clone(), b.clone(), 1.0).unwrap();
        assert!(adapted.is_packed(), "adapter must not un-pack the base");

        let x = Tensor::randn(0f32, 1f32, (4usize, in_dim), &dev).unwrap();
        let base_y = packed_base.forward(&x).unwrap();
        let adapted_y = adapted.forward(&x).unwrap();
        let shift = (adapted_y.sub(&base_y).unwrap())
            .abs()
            .unwrap()
            .max_all()
            .unwrap()
            .to_scalar::<f32>()
            .unwrap();
        assert!(
            shift > 1e-4,
            "additive LoRA on packed did not shift ({shift})"
        );
        assert!(
            adapted_y
                .abs()
                .unwrap()
                .max_all()
                .unwrap()
                .to_scalar::<f32>()
                .unwrap()
                .is_finite(),
            "packed additive output non-finite"
        );

        // scale 0 ⇒ exact no-op vs the un-adapted packed base.
        let (mut zero, _) = packed_adapt(&tmp, out_dim, in_dim);
        zero.push_lora(a, b, 0.0).unwrap();
        let zero_dev = (zero.forward(&x).unwrap().sub(&base_y).unwrap())
            .abs()
            .unwrap()
            .max_all()
            .unwrap()
            .to_scalar::<f32>()
            .unwrap();
        assert_eq!(
            zero_dev, 0.0,
            "scale-0 residual must be an exact no-op on packed"
        );

        #[cfg(feature = "model-inference-proof")]
        println!("\nMODEL_INFERENCE_BODY_COMPLETED quant::adapt::tests::additive_lora_on_packed_shifts_stays_packed_and_scale0_is_noop");
    }

    /// `clear_adapters` reverts an adapted projection to its **bare base** — the per-phase toggle-off a
    /// multi-phase render runs on its job-local DiT between phases. After clearing, the forward is
    /// **byte-identical** to the un-adapted base, and a re-push installs a fresh residual — proving a
    /// phase's adapter set is authoritative regardless of what the prior phase installed, and that the
    /// base weight is never disturbed by the toggle.
    #[test]
    #[ignore = "model inference; explicit model workflow only"]
    fn clear_adapters_reverts_to_bare_base_and_repush_reinstalls() {
        let dev = Device::Cpu;
        let (out_dim, in_dim, rank) = (48usize, 64usize, 4usize);
        let w = Tensor::randn(0f32, 1f32, (out_dim, in_dim), &dev).unwrap();
        let a = (Tensor::randn(0f32, 1f32, (in_dim, rank), &dev).unwrap() * 0.1).unwrap();
        let b = (Tensor::randn(0f32, 1f32, (rank, out_dim), &dev).unwrap() * 0.1).unwrap();
        let x = Tensor::randn(0f32, 1f32, (3usize, in_dim), &dev).unwrap();

        // The bare-base reference (same weight, no residual).
        let base = AdaptLinear::from_dense(CandleNnLinear::new(w.clone(), None), in_dim, out_dim);
        let base_y = base.forward(&x).unwrap();

        let mut lin = AdaptLinear::from_dense(CandleNnLinear::new(w, None), in_dim, out_dim);
        lin.push_lora(a.clone(), b.clone(), 0.7).unwrap();
        assert!(lin.is_adapted());
        let adapted_shift = (lin.forward(&x).unwrap() - &base_y)
            .unwrap()
            .abs()
            .unwrap()
            .max_all()
            .unwrap()
            .to_scalar::<f32>()
            .unwrap();
        assert!(adapted_shift > 1e-4, "the residual must shift the forward");

        // Toggle off: clear ⇒ byte-identical to the bare base (an exact 0 deviation, not a tolerance).
        lin.clear_adapters();
        assert!(!lin.is_adapted(), "clear must drop every residual");
        let cleared_dev = (lin.forward(&x).unwrap() - &base_y)
            .unwrap()
            .abs()
            .unwrap()
            .max_all()
            .unwrap()
            .to_scalar::<f32>()
            .unwrap();
        assert_eq!(
            cleared_dev, 0.0,
            "after clear the forward must equal the bare base exactly"
        );

        // Re-push a fresh (different-scale) residual — the phase's set is authoritative after a clear.
        lin.push_lora(a, b, 0.3).unwrap();
        assert!(lin.is_adapted());
        let repush_shift = (lin.forward(&x).unwrap() - &base_y)
            .unwrap()
            .abs()
            .unwrap()
            .max_all()
            .unwrap()
            .to_scalar::<f32>()
            .unwrap();
        assert!(
            repush_shift > 1e-4 && (repush_shift - adapted_shift).abs() > 1e-6,
            "re-push installs a fresh residual at the new scale"
        );

        #[cfg(feature = "model-inference-proof")]
        println!("\nMODEL_INFERENCE_BODY_COMPLETED quant::adapt::tests::clear_adapters_reverts_to_bare_base_and_repush_reinstalls");
    }

    /// The **acceptance parity** on a packed base, isolated from the base's own quant error: the
    /// residual the adapter contributes equals `scale·((x·a)·b)` **exactly** (f32). `adapted.forward −
    /// base.forward` cancels the (bit-identical) packed base — the dequant-repack quant error included —
    /// leaving only the residual. This proves the LoRA is added **additively over the packed weight**
    /// (never folded into a re-quantized dense weight).
    #[test]
    #[ignore = "model inference; explicit model workflow only"]
    fn additive_on_packed_adds_exact_residual_over_base() {
        let tmp = tempfile::tempdir().unwrap();
        let dev = Device::Cpu;
        let (out_dim, in_dim, rank) = (64usize, 128usize, 4usize);
        let (base, _grid) = packed_adapt(&tmp, out_dim, in_dim);
        let (mut adapted, _grid) = packed_adapt(&tmp, out_dim, in_dim); // bit-identical packed base
        let a = (Tensor::randn(0f32, 1f32, (in_dim, rank), &dev).unwrap() * 0.1).unwrap();
        let b = (Tensor::randn(0f32, 1f32, (rank, out_dim), &dev).unwrap() * 0.1).unwrap();
        let scale = 0.7f64;
        adapted.push_lora(a.clone(), b.clone(), scale).unwrap();
        assert!(adapted.is_packed(), "base stays packed under the residual");

        let x = Tensor::randn(0f32, 1f32, (4usize, in_dim), &dev).unwrap();
        // The residual the adapter contributes = adapted − base (identical packed bases cancel exactly).
        let residual_actual = (adapted.forward(&x).unwrap() - base.forward(&x).unwrap()).unwrap();
        // Expected: scale·((x·a)·b).
        let residual_expected = ((x.matmul(&a).unwrap().matmul(&b).unwrap()) * scale).unwrap();
        let dev_max = (residual_actual - residual_expected)
            .unwrap()
            .abs()
            .unwrap()
            .max_all()
            .unwrap()
            .to_scalar::<f32>()
            .unwrap();
        assert!(
            dev_max < 1e-5,
            "packed residual != scale·(x·a)·b (max diff {dev_max})"
        );

        #[cfg(feature = "model-inference-proof")]
        println!("\nMODEL_INFERENCE_BODY_COMPLETED quant::adapt::tests::additive_on_packed_adds_exact_residual_over_base");
    }

    /// `quantize` folds a **dense** base to packed in place (idempotent no-op if already packed), keeping
    /// any attached residual — the lens/sd3 "quantize after the dense fold" contract (sc-11105). The
    /// folded-packed forward equals the packed base forward plus the same residual (f32 tol), and a
    /// second quantize on the now-packed base is an exact no-op (the additive residuals survive).
    #[test]
    #[ignore = "model inference; explicit model workflow only"]
    fn quantize_folds_dense_base_and_preserves_residual() {
        let dev = Device::Cpu;
        let (out_dim, in_dim, rank) = (64usize, 128usize, 4usize);
        let w = Tensor::randn(0f32, 1f32, (out_dim, in_dim), &dev).unwrap();
        let a = (Tensor::randn(0f32, 1f32, (in_dim, rank), &dev).unwrap() * 0.1).unwrap();
        let b = (Tensor::randn(0f32, 1f32, (rank, out_dim), &dev).unwrap() * 0.1).unwrap();

        let mut lin =
            AdaptLinear::from_dense(CandleNnLinear::new(w.clone(), None), in_dim, out_dim);
        lin.push_lora(a.clone(), b.clone(), 0.7).unwrap();
        assert!(!lin.is_packed());
        lin.quantize(Quant::Q8).unwrap();
        assert!(lin.is_packed(), "dense base must fold to packed");
        assert!(lin.is_adapted(), "residual must survive the fold");

        let x = Tensor::randn(0f32, 1f32, (3usize, in_dim), &dev).unwrap();
        let residual = ((x.matmul(&a).unwrap().matmul(&b).unwrap()) * 0.7).unwrap();
        // The Q8-packed base alone (no residual) — same weight, same fold — plus the residual.
        let mut base_only = AdaptLinear::from_dense(CandleNnLinear::new(w, None), in_dim, out_dim);
        base_only.quantize(Quant::Q8).unwrap();
        let expected = (base_only.forward(&x).unwrap() + residual).unwrap();
        let dev_max = (lin.forward(&x).unwrap() - expected)
            .unwrap()
            .abs()
            .unwrap()
            .max_all()
            .unwrap()
            .to_scalar::<f32>()
            .unwrap();
        assert!(
            dev_max < 1e-4,
            "adapted-packed forward != packed base + residual ({dev_max})"
        );

        // Idempotent: a second quantize is a no-op (packed stays packed; forward unchanged).
        let y0 = lin.forward(&x).unwrap();
        lin.quantize(Quant::Q4).unwrap();
        let y1 = lin.forward(&x).unwrap();
        let noop = (y1 - y0)
            .unwrap()
            .abs()
            .unwrap()
            .max_all()
            .unwrap()
            .to_scalar::<f32>()
            .unwrap();
        assert_eq!(
            noop, 0.0,
            "quantize on an already-packed base must be a no-op"
        );

        #[cfg(feature = "model-inference-proof")]
        println!("\nMODEL_INFERENCE_BODY_COMPLETED quant::adapt::tests::quantize_folds_dense_base_and_preserves_residual");
    }

    // ---- Structured (deferred) LoKr — the Kronecker vec-trick ---------------------------------------

    /// `x·ΔWᵀ` — the materialized-delta residual, the reference the vec-trick must reproduce.
    fn delta_residual(x: &Tensor, delta: &Tensor) -> Tensor {
        x.matmul(&delta.t().unwrap().contiguous().unwrap()).unwrap()
    }

    /// **Full-factor parity.** The structured vec-trick residual `vec(w1·X·w2ᵀ)` equals the
    /// materialized-delta residual `x·(kron(w1,w2))ᵀ` for a full `w1⊗w2` LoKr — proving the row-major
    /// Kronecker identity the whole port rests on, against candle's own `reconstruct_lokr_delta` fold.
    /// The built factors are the SMALL `[a,c]`/`[b,d]` matrices — the `[out,in]` delta is NEVER formed.
    #[test]
    #[ignore = "model inference; explicit model workflow only"]
    fn structured_lokr_full_matches_reconstruct_delta() {
        let dev = Device::Cpu;
        let (a, b, c, d) = (2usize, 3, 4, 5);
        let (out, inp) = (a * b, c * d);
        let w1 = Tensor::from_vec(
            (0..(a * c))
                .map(|i| (i as f32 * 0.11).sin())
                .collect::<Vec<_>>(),
            (a, c),
            &dev,
        )
        .unwrap();
        let w2 = Tensor::from_vec(
            (0..(b * d))
                .map(|i| (i as f32 * 0.07).cos())
                .collect::<Vec<_>>(),
            (b, d),
            &dev,
        )
        .unwrap();
        let scale = 0.9f64;
        let delta = reconstruct_lokr_delta(
            Some(&w1),
            None,
            None,
            Some(&w2),
            None,
            None,
            1.0,
            1.0,
            scale as f32,
            (out, inp),
        )
        .unwrap();
        let factors = LokrFactors::build(
            scale,
            (out, inp),
            Some(&w1),
            None,
            None,
            Some(&w2),
            None,
            None,
            None,
        )
        .unwrap()
        .expect("a plain linear LoKr is deferrable");
        assert_eq!(factors.w1.dims(), &[a, c], "left factor stays [a,c]");
        assert_eq!(factors.w2.dims(), &[b, d], "right factor stays [b,d]");
        assert!(
            factors.w1.elem_count() + factors.w2.elem_count() < out * inp,
            "structured factors must be « the [out,in] delta"
        );

        let x = Tensor::from_vec(
            (0..(2 * inp))
                .map(|i| (i as f32 * 0.013 - 0.5).sin())
                .collect::<Vec<_>>(),
            (2, inp),
            &dev,
        )
        .unwrap();
        let want = delta_residual(&x, &delta);
        let got = factors.residual(&x).unwrap();
        assert_eq!(got.dims(), &[2, out]);
        let dev_max = tensor_max_abs(&(got.clone() - &want).unwrap());
        assert!(
            dev_max < 1e-4,
            "structured LoKr residual != materialized-delta residual ({dev_max})"
        );
        assert!(
            tensor_max_abs(&got) > 1e-2,
            "the LoKr residual must be materially non-zero"
        );

        #[cfg(feature = "model-inference-proof")]
        println!("\nMODEL_INFERENCE_BODY_COMPLETED quant::adapt::tests::structured_lokr_full_matches_reconstruct_delta");
    }

    /// **Decomposed-factor parity.** Same identity for a low-rank LoKr (`w1_a·w1_b`, `w2_a·w2_b`): the
    /// inner products are materialized only as the SMALL `[a,c]`/`[b,d]` factors, never `[out,in]`.
    #[test]
    #[ignore = "model inference; explicit model workflow only"]
    fn structured_lokr_decomposed_matches_reconstruct_delta() {
        let dev = Device::Cpu;
        let (a, b, c, d, r) = (3usize, 2, 5, 4, 2);
        let (out, inp) = (a * b, c * d);
        let mk = |rows: usize, cols: usize, seed: f32| {
            Tensor::from_vec(
                (0..(rows * cols))
                    .map(|i| (i as f32 * 0.09 + seed).sin() * 0.3)
                    .collect::<Vec<_>>(),
                (rows, cols),
                &dev,
            )
            .unwrap()
        };
        let (w1a, w1b) = (mk(a, r, 0.1), mk(r, c, 0.2)); // w1 = [a,c]
        let (w2a, w2b) = (mk(b, r, 0.3), mk(r, d, 0.4)); // w2 = [b,d]
        let scale = 1.3f64;
        let delta = reconstruct_lokr_delta(
            None,
            Some(&w1a),
            Some(&w1b),
            None,
            Some(&w2a),
            Some(&w2b),
            1.0,
            1.0,
            scale as f32,
            (out, inp),
        )
        .unwrap();
        let factors = LokrFactors::build(
            scale,
            (out, inp),
            None,
            Some(&w1a),
            Some(&w1b),
            None,
            None,
            Some(&w2a),
            Some(&w2b),
        )
        .unwrap()
        .expect("a decomposed linear LoKr is deferrable");
        assert_eq!(factors.w1.dims(), &[a, c]);
        assert_eq!(factors.w2.dims(), &[b, d]);

        let x = Tensor::from_vec(
            (0..inp)
                .map(|i| (i as f32 * 0.02).cos())
                .collect::<Vec<_>>(),
            (1, inp),
            &dev,
        )
        .unwrap();
        let dev_max =
            tensor_max_abs(&(factors.residual(&x).unwrap() - delta_residual(&x, &delta)).unwrap());
        assert!(
            dev_max < 1e-4,
            "decomposed structured LoKr != materialized delta ({dev_max})"
        );

        #[cfg(feature = "model-inference-proof")]
        println!("\nMODEL_INFERENCE_BODY_COMPLETED quant::adapt::tests::structured_lokr_decomposed_matches_reconstruct_delta");
    }

    /// Frozen factors prepare one compute-dtype view per installed residual and reuse it across
    /// forwards. The trainable residual deliberately contributes no cached preparation: its live
    /// leaves still transpose inside every eager graph.
    #[test]
    #[ignore = "model inference; explicit model workflow only"]
    fn frozen_adapter_factors_prepare_once_per_compute_dtype() {
        let dev = Device::Cpu;
        let mut linear = AdaptLinear::from_dense(
            CandleNnLinear::new(Tensor::zeros((4, 4), DType::F32, &dev).unwrap(), None),
            4,
            4,
        );
        linear
            .push_lora(
                Tensor::ones((4, 2), DType::F32, &dev).unwrap(),
                Tensor::ones((2, 4), DType::F32, &dev).unwrap(),
                0.5,
            )
            .unwrap();
        let factors = LokrFactors::build(
            0.25,
            (4, 4),
            Some(&Tensor::eye(2, DType::F32, &dev).unwrap()),
            None,
            None,
            Some(&Tensor::eye(2, DType::F32, &dev).unwrap()),
            None,
            None,
            None,
        )
        .unwrap()
        .expect("2x2 Kronecker factors fit the 4x4 base");
        linear.push_lokr_structured(factors).unwrap();
        linear.push_trainable_lora(
            Tensor::ones((2, 4), DType::F32, &dev).unwrap(),
            Tensor::ones((4, 2), DType::F32, &dev).unwrap(),
            0.25,
        );

        let f32 = Tensor::ones((1, 4), DType::F32, &dev).unwrap();
        linear.forward(&f32).unwrap();
        linear.forward(&f32).unwrap();
        assert_eq!(
            linear.frozen_adapter_preparation_count(),
            2,
            "one f32 preparation for each frozen LoRA/LoKr, none for trainable LoRA"
        );

        // CPU matmul only supports f32, but these are the exact frozen prepare helpers used by a
        // bf16/metal or CUDA forward. Probe a second compute dtype without constructing an unsupported
        // CPU matmul, then verify it is cached rather than recreated.
        match &linear.adapters[0] {
            Adapter::Lora { a, b, prepared, .. } => {
                prepared.factors(a, b, DType::BF16).unwrap();
                prepared.factors(a, b, DType::BF16).unwrap();
            }
            _ => panic!("first residual must be frozen LoRA"),
        }
        match &linear.adapters[1] {
            Adapter::LokrStructured { factors } => {
                factors.prepared_factors(DType::BF16).unwrap();
                factors.prepared_factors(DType::BF16).unwrap();
            }
            _ => panic!("second residual must be frozen LoKr"),
        }
        assert_eq!(
            linear.frozen_adapter_preparation_count(),
            4,
            "the bf16 views are added once, then reused on the second preparation"
        );

        linear.to_device(&dev).unwrap();
        assert_eq!(
            linear.frozen_adapter_preparation_count(),
            0,
            "migration must discard stale device-specific frozen views"
        );
        linear.forward(&f32).unwrap();
        assert_eq!(linear.frozen_adapter_preparation_count(), 2);

        #[cfg(feature = "model-inference-proof")]
        println!("\nMODEL_INFERENCE_BODY_COMPLETED quant::adapt::tests::frozen_adapter_factors_prepare_once_per_compute_dtype");
    }

    /// **Acceptance parity on a PACKED base.** The structured LoKr installs on a packed q4 base, the
    /// base stays **packed** (no `[out,in]` weight materialized), and `packed_forward + residual`
    /// reproduces `packed_forward + folded_delta` within quant tolerance. Also the mutation anchor: a
    /// scale-0 LoKr is an exact no-op.
    #[test]
    #[ignore = "model inference; explicit model workflow only"]
    fn structured_lokr_on_packed_matches_folded_and_stays_packed() {
        let tmp = tempfile::tempdir().unwrap();
        let dev = Device::Cpu;
        // in = c·d = 8·16 = 128 (divisible by group 64); out = a·b = 4·16 = 64.
        let (a, b, c, d) = (4usize, 16, 8, 16);
        let (out_dim, in_dim) = (a * b, c * d);
        let (base, _grid) = packed_adapt(&tmp, out_dim, in_dim);
        let (mut adapted, _grid) = packed_adapt(&tmp, out_dim, in_dim); // bit-identical packed base
        assert!(base.is_packed() && adapted.is_packed());

        let w1 = (Tensor::randn(0f32, 1f32, (a, c), &dev).unwrap() * 0.1).unwrap();
        let w2 = (Tensor::randn(0f32, 1f32, (b, d), &dev).unwrap() * 0.1).unwrap();
        let scale = 0.7f64;
        let factors = LokrFactors::build(
            scale,
            (out_dim, in_dim),
            Some(&w1),
            None,
            None,
            Some(&w2),
            None,
            None,
            None,
        )
        .unwrap()
        .expect("deferrable");
        assert!(factors.w1.elem_count() + factors.w2.elem_count() < out_dim * in_dim);
        adapted.push_lokr_structured(factors).unwrap();
        assert!(
            adapted.is_packed(),
            "structured LoKr must not un-pack the base"
        );

        let x = Tensor::randn(0f32, 1f32, (4usize, in_dim), &dev).unwrap();
        let residual_actual = (adapted.forward(&x).unwrap() - base.forward(&x).unwrap()).unwrap();
        let delta = reconstruct_lokr_delta(
            Some(&w1),
            None,
            None,
            Some(&w2),
            None,
            None,
            1.0,
            1.0,
            scale as f32,
            (out_dim, in_dim),
        )
        .unwrap();
        let dev_max =
            tensor_max_abs(&(residual_actual.clone() - delta_residual(&x, &delta)).unwrap());
        assert!(
            dev_max < 1e-4,
            "packed structured LoKr residual != folded delta ({dev_max})"
        );
        assert!(
            tensor_max_abs(&residual_actual) > 1e-3,
            "the LoKr must shift the packed forward"
        );

        // Mutation: a scale-0 structured LoKr is an exact no-op over the packed base.
        let (mut zero, _) = packed_adapt(&tmp, out_dim, in_dim);
        let f0 = LokrFactors::build(
            0.0,
            (out_dim, in_dim),
            Some(&w1),
            None,
            None,
            Some(&w2),
            None,
            None,
            None,
        )
        .unwrap()
        .unwrap();
        zero.push_lokr_structured(f0).unwrap();
        let zero_dev =
            tensor_max_abs(&(zero.forward(&x).unwrap() - base.forward(&x).unwrap()).unwrap());
        assert_eq!(
            zero_dev, 0.0,
            "scale-0 structured LoKr must be an exact no-op"
        );

        #[cfg(feature = "model-inference-proof")]
        println!("\nMODEL_INFERENCE_BODY_COMPLETED quant::adapt::tests::structured_lokr_on_packed_matches_folded_and_stays_packed");
    }

    /// A tucker/CP `w2` (`lokr_t2`, conv-only) and a base that does not factor as `a·b × c·d` are both
    /// NOT deferrable via the 2-D vec-trick → `Ok(None)`, so the installer rejects them on a packed tier
    /// rather than materializing. A missing `w1`/`w2` leg is a typed error, never a panic.
    #[test]
    fn structured_lokr_non_deferrable_and_missing_legs() {
        let dev = Device::Cpu;
        let w1 = Tensor::zeros((3usize, 4), DType::F32, &dev).unwrap();
        let w2a = Tensor::zeros((2usize, 4), DType::F32, &dev).unwrap();
        let w2b = Tensor::zeros((2usize, 5), DType::F32, &dev).unwrap();
        let t2 = Tensor::zeros((2usize, 2, 3, 3), DType::F32, &dev).unwrap();
        // Tucker `w2_t2` present → None (the guard fires before any shape check).
        let got = LokrFactors::build(
            1.0,
            (24, 180),
            Some(&w1),
            None,
            None,
            None,
            Some(&t2),
            Some(&w2a),
            Some(&w2b),
        )
        .unwrap();
        assert!(got.is_none(), "tucker/CP LoKr must be non-deferrable");

        // A base that does not factor as a·b × c·d (here a·b = 3·2 = 6 ≠ out = 7) → None.
        let w2 = Tensor::zeros((2usize, 5), DType::F32, &dev).unwrap();
        let mism = LokrFactors::build(
            1.0,
            (7, 20),
            Some(&w1),
            None,
            None,
            Some(&w2),
            None,
            None,
            None,
        )
        .unwrap();
        assert!(
            mism.is_none(),
            "a base that doesn't factor a·b×c·d is non-deferrable"
        );

        // A missing w2 leg (no full, no a/b) is a typed error, not a panic.
        let err = LokrFactors::build(1.0, (6, 20), Some(&w1), None, None, None, None, None, None);
        assert!(err.is_err(), "missing w2 must be a typed error");
    }

    /// A LoRA residual on an activation with a **non-1 leading (batch) dim** must fold those leading
    /// dims into the GEMM's `M` — never `broadcast_matmul`, which physically copies the 2-D factor once
    /// per batch element. The Krea text-fusion `layerwise_blocks` run `[n_tokens, num_layers, d]`, so at
    /// a 2048² image edit (`n_tokens = 4107`, `d = 2560`, rank 256) the old broadcast path materialized a
    /// 5.4 GB copy per residual leg and the first denoise step never finished. This pins the flattened
    /// path's NUMERICS against a per-batch-slice reference, so the perf fix can't silently change the math.
    #[test]
    #[ignore = "model inference; explicit model workflow only"]
    fn lora_residual_on_batched_activation_matches_per_slice_reference() {
        let dev = Device::Cpu;
        let (out_dim, in_dim, rank, scale) = (12usize, 16usize, 4usize, 0.7f64);
        let w = Tensor::randn(0f32, 1f32, (out_dim, in_dim), &dev).unwrap();
        let a = Tensor::randn(0f32, 1f32, (in_dim, rank), &dev).unwrap(); // [in, rank]
        let b = Tensor::randn(0f32, 1f32, (rank, out_dim), &dev).unwrap(); // [rank, out]
        let mut lin =
            AdaptLinear::from_dense(CandleNnLinear::new(w.clone(), None), in_dim, out_dim);
        lin.push_lora(a.clone(), b.clone(), scale).unwrap();

        // `[N, S, in]`: N is a BATCH of token stacks (the layerwise-block shape), S the stack length.
        let (n, s) = (5usize, 3usize);
        let x = Tensor::randn(0f32, 1f32, (n, s, in_dim), &dev).unwrap();
        let got = lin.forward(&x).unwrap();
        assert_eq!(got.dims(), &[n, s, out_dim]);

        for i in 0..n {
            let xi = x.narrow(0, i, 1).unwrap().reshape((s, in_dim)).unwrap();
            let base =
                __model_proof_output!(CandleNnLinear::new(w.clone(), None).forward(&xi)).unwrap();
            let res = (xi.matmul(&a).unwrap().matmul(&b).unwrap() * scale).unwrap();
            let want = (base + res).unwrap();
            let mine = got.narrow(0, i, 1).unwrap().reshape((s, out_dim)).unwrap();
            let d = (mine - want)
                .unwrap()
                .abs()
                .unwrap()
                .max_all()
                .unwrap()
                .to_vec0::<f32>()
                .unwrap();
            assert!(d < 1e-5, "batch slice {i} diverged from the reference: {d}");
        }

        #[cfg(feature = "model-inference-proof")]
        println!("\nMODEL_INFERENCE_BODY_COMPLETED quant::adapt::tests::lora_residual_on_batched_activation_matches_per_slice_reference");
    }

    // ------------------------------------------------------------------------------------------
    // NVFP4 as an adapter-capable base (sc-21483, epic 11037).
    //
    // These run on the CPU lane: `Nvfp4Linear` retains the packed host container in every regime and
    // serves the transparent dequant→bf16 fallback off `sm_120`, so the ADDITIVE arm — the thing this
    // story adds — is fully exercisable without a Blackwell device. The CUDA lane adds the FP4 GEMM
    // under the same arm; that is what the krea real-weight render covers.
    // ------------------------------------------------------------------------------------------

    /// An NVFP4-based [`AdaptLinear`] plus the dense master it was packed from.
    fn nvfp4_host(out_dim: usize, in_dim: usize) -> AdaptLinear {
        let dev = Device::Cpu;
        let w: Vec<f32> = (0..out_dim * in_dim)
            .map(|i| ((i % 17) as f32 - 8.0) / 11.0)
            .collect();
        let w = Tensor::from_vec(w, (out_dim, in_dim), &dev).unwrap();
        let lin = super::super::Nvfp4Linear::from_dense(
            &w,
            None,
            &dev,
            super::super::ActPrecision::W4A16,
        )
        .unwrap();
        AdaptLinear::from_nvfp4(lin)
    }

    /// A deterministic `a [in, rank]` / `b [rank, out]` LoRA factor pair on the CPU.
    fn lora_pair(in_dim: usize, rank: usize, out_dim: usize) -> (Tensor, Tensor) {
        let dev = Device::Cpu;
        let a: Vec<f32> = (0..in_dim * rank).map(|i| (i % 5) as f32 * 0.03).collect();
        let b: Vec<f32> = (0..rank * out_dim)
            .map(|i| ((i % 7) as f32 - 3.0) * 0.02)
            .collect();
        (
            Tensor::from_vec(a, (in_dim, rank), &dev).unwrap(),
            Tensor::from_vec(b, (rank, out_dim), &dev).unwrap(),
        )
    }

    fn flat(t: &Tensor) -> Vec<f32> {
        t.flatten_all().unwrap().to_vec1::<f32>().unwrap()
    }

    /// AC#1 — identity / delta / removal at the shared seam. A LoRA installed over an NVFP4 base
    /// contributes exactly `scale·((x·a)·b)` on top of the packed forward, and `clear_adapters`
    /// restores the base output **exactly**: the packed weight was never mutated, so this is bit
    /// equality, not a tolerance.
    #[test]
    #[ignore = "model inference; explicit model workflow only"]
    fn nvfp4_base_hosts_an_additive_lora_and_restores_exactly_on_removal() {
        let (out_dim, in_dim, rank) = (32, 64, 4);
        let mut host = nvfp4_host(out_dim, in_dim);
        assert!(host.is_nvfp4(), "the base must be an NVFP4 projection");
        assert!(host.is_quantized(), "NVFP4 is a quantized regime");
        assert!(!host.is_packed(), "NVFP4 is not the MLX-packed regime");
        assert_eq!(host.base_shape(), (out_dim, in_dim));
        assert!(host.base_nvfp4().is_some());

        let x = Tensor::from_vec(
            (0..8 * in_dim)
                .map(|i| (i % 13) as f32 * 0.05)
                .collect::<Vec<_>>(),
            (8, in_dim),
            &Device::Cpu,
        )
        .unwrap();
        let bare = host.forward(&x).unwrap();

        let (a, b) = lora_pair(in_dim, rank, out_dim);
        host.push_lora_checked(a.clone(), b.clone(), 0.75).unwrap();
        assert!(host.is_adapted());
        let adapted = host.forward(&x).unwrap();

        // The delta is the LoRA's own product — the packed base contributed nothing extra.
        let want_delta = (x.matmul(&a).unwrap().matmul(&b).unwrap() * 0.75).unwrap();
        let got_delta = (&adapted - &bare).unwrap();
        assert!(
            tensor_max_abs(&(got_delta - &want_delta).unwrap()) < 1e-5,
            "the additive arm did not contribute the LoRA delta"
        );
        assert!(
            tensor_max_abs(&(adapted - &bare).unwrap()) > 1e-4,
            "the adapter must actually move the output"
        );

        // Removal restores the base output EXACTLY.
        host.clear_adapters();
        assert!(!host.is_adapted());
        assert_eq!(
            flat(&host.forward(&x).unwrap()),
            flat(&bare),
            "clearing the adapter must restore the exact base output"
        );
        assert!(host.is_nvfp4(), "the base regime survived the whole cycle");

        #[cfg(feature = "model-inference-proof")]
        println!("\nMODEL_INFERENCE_BODY_COMPLETED quant::adapt::tests::nvfp4_base_hosts_an_additive_lora_and_restores_exactly_on_removal");
    }

    /// AC#1 — a structured LoKr rides the same NVFP4 base through the Kronecker vec-trick, matching
    /// the reconstructed dense delta, and removal is again exact.
    #[test]
    #[ignore = "model inference; explicit model workflow only"]
    fn nvfp4_base_hosts_a_structured_lokr_residual() {
        let (a, b, c, d) = (4usize, 8usize, 8usize, 8usize);
        let (out_dim, in_dim) = (a * b, c * d);
        let dev = Device::Cpu;
        let mut host = nvfp4_host(out_dim, in_dim);
        let w1 = Tensor::from_vec(
            (0..(a * c))
                .map(|i| (i as f32 * 0.11).sin())
                .collect::<Vec<_>>(),
            (a, c),
            &dev,
        )
        .unwrap();
        let w2 = Tensor::from_vec(
            (0..(b * d))
                .map(|i| (i as f32 * 0.07).cos())
                .collect::<Vec<_>>(),
            (b, d),
            &dev,
        )
        .unwrap();
        let scale = 0.5f64;
        let delta = reconstruct_lokr_delta(
            Some(&w1),
            None,
            None,
            Some(&w2),
            None,
            None,
            1.0,
            1.0,
            scale as f32,
            (out_dim, in_dim),
        )
        .unwrap();
        let factors = LokrFactors::build(
            scale,
            (out_dim, in_dim),
            Some(&w1),
            None,
            None,
            Some(&w2),
            None,
            None,
            None,
        )
        .unwrap()
        .expect("a plain linear LoKr is deferrable");

        let x = Tensor::from_vec(
            (0..(4 * in_dim))
                .map(|i| (i as f32 * 0.013 - 0.5).sin())
                .collect::<Vec<_>>(),
            (4, in_dim),
            &dev,
        )
        .unwrap();
        let bare = host.forward(&x).unwrap();
        host.push_lokr_structured_checked(factors).unwrap();
        let adapted = host.forward(&x).unwrap();

        let want = (&bare + delta_residual(&x, &delta)).unwrap();
        assert!(
            tensor_max_abs(&(adapted - &want).unwrap()) < 1e-4,
            "structured LoKr over NVFP4 diverged from the reconstructed delta"
        );

        host.clear_adapters();
        assert_eq!(flat(&host.forward(&x).unwrap()), flat(&bare));

        #[cfg(feature = "model-inference-proof")]
        println!("\nMODEL_INFERENCE_BODY_COMPLETED quant::adapt::tests::nvfp4_base_hosts_a_structured_lokr_residual");
    }

    /// **sc-11045 fix round (MAJOR 7, minor): the unchecked pushes delegate to the checked forms
    /// on an NVFP4 base**, so the bypass is structurally closed — no caller can attach an
    /// unvalidated factor to a base that has no fallback. Dense/MLX-packed bases keep the
    /// historical unchecked behaviour.
    ///
    /// # Mutation
    ///
    /// Remove the `Base::Nvfp4` delegation from `push_lora`/`push_lokr_structured` (restore the
    /// plain `self.adapters.push(...)`): both `unwrap_err`s below go red.
    #[test]
    fn the_unchecked_pushes_delegate_to_the_checked_forms_on_an_nvfp4_base() {
        let (out_dim, in_dim) = (32, 64);
        let mut host = nvfp4_host(out_dim, in_dim);
        // A mis-shaped LoRA through the UNCHECKED entry point refuses on the NVFP4 base.
        let (bad_a, bad_b) = lora_pair(16, 4, out_dim);
        let error = host.push_lora(bad_a, bad_b, 1.0).unwrap_err().to_string();
        assert!(error.contains("NVFP4 base"), "{error}");
        assert!(!host.is_adapted());
        // A mis-reconstructing LoKr likewise: factors built for a [16, 64] base.
        let dev = Device::Cpu;
        let w1 = Tensor::ones((2, 8), DType::F32, &dev).unwrap();
        let w2 = Tensor::ones((8, 8), DType::F32, &dev).unwrap();
        let wrong = LokrFactors::build(
            1.0,
            (16, 64),
            Some(&w1),
            None,
            None,
            Some(&w2),
            None,
            None,
            None,
        )
        .unwrap()
        .expect("a plain linear LoKr is deferrable");
        let error = host.push_lokr_structured(wrong).unwrap_err().to_string();
        assert!(error.contains("NVFP4 base"), "{error}");
        assert!(!host.is_adapted());
        // Well-shaped factors still install through the same unchecked names.
        let (a, b) = lora_pair(in_dim, 4, out_dim);
        host.push_lora(a, b, 1.0).unwrap();
        assert!(host.is_adapted());
        // And a dense base keeps the historical unchecked semantics: no validation, always Ok.
        let mut dense = AdaptLinear::from_dense(
            CandleNnLinear::new(
                Tensor::zeros((out_dim, in_dim), DType::F32, &dev).unwrap(),
                None,
            ),
            in_dim,
            out_dim,
        );
        let (bad_a, bad_b) = lora_pair(16, 4, out_dim);
        dense.push_lora(bad_a, bad_b, 1.0).unwrap();
    }

    /// AC#2, shape half — a factor that does not compose against the base is refused at ADMISSION,
    /// naming the NVFP4 regime, and nothing is attached. It never reaches a sampler step.
    #[test]
    fn nvfp4_base_refuses_a_mis_shaped_adapter_at_admission() {
        let (out_dim, in_dim) = (32, 64);
        let mut host = nvfp4_host(out_dim, in_dim);

        // `a` contracts against the wrong input width…
        let (a, b) = lora_pair(in_dim / 2, 4, out_dim);
        let error = host.push_lora_checked(a, b, 1.0).unwrap_err().to_string();
        assert!(error.contains("NVFP4 base"), "{error}");
        assert!(error.contains("do not compose"), "{error}");
        assert!(!host.is_adapted(), "a refused factor must not be attached");

        // …and the wrong output width.
        let (a, b) = lora_pair(in_dim, 4, out_dim + 1);
        assert!(host.push_lora_checked(a, b, 1.0).is_err());
        assert!(!host.is_adapted());

        // A LoKr whose Kronecker factors reconstruct a different projection is refused too.
        let w1 = Tensor::from_vec(vec![0.1f32; 4 * 8], (4, 8), &Device::Cpu).unwrap();
        let w2 = Tensor::from_vec(vec![0.1f32; 4 * 8], (4, 8), &Device::Cpu).unwrap();
        let factors = LokrFactors::build(
            0.5,
            (32, 64),
            Some(&w1),
            None,
            None,
            Some(&w2),
            None,
            None,
            None,
        )
        .unwrap();
        if let Some(factors) = factors {
            // Re-host the same factors on a DIFFERENT projection shape: admission must reject.
            let mut other = nvfp4_host(64, 32);
            let error = other
                .push_lokr_structured_checked(factors)
                .unwrap_err()
                .to_string();
            assert!(error.contains("NVFP4 base"), "{error}");
            assert!(error.contains("not the base"), "{error}");
            assert!(!other.is_adapted());
        }
    }

    /// AC#2, dtype/device half — an additive residual is cast to the activation dtype per forward,
    /// so a non-float factor is refused rather than silently reinterpreted.
    #[test]
    fn nvfp4_base_refuses_a_non_float_adapter_factor() {
        let (out_dim, in_dim, rank) = (32, 64, 4);
        let mut host = nvfp4_host(out_dim, in_dim);
        let (a, b) = lora_pair(in_dim, rank, out_dim);
        let a_u32 = a.to_dtype(DType::U32).unwrap();
        let error = host
            .push_lora_checked(a_u32, b, 1.0)
            .unwrap_err()
            .to_string();
        assert!(error.contains("NVFP4 base"), "{error}");
        assert!(error.contains("non-float dtype"), "{error}");
        assert!(!host.is_adapted());
    }

    /// AC#2, regime half — an NVFP4 base is NEVER silently converted to q4/q8 (or dequantized to
    /// BF16) to make something fit. Every fold entry point refuses, and the base stays NVFP4.
    #[test]
    fn nvfp4_base_refuses_every_requantization() {
        let mut host = nvfp4_host(32, 64);
        for quant in [Quant::Q4, Quant::Q8, Quant::Nvfp4] {
            let error = host.quantize(quant).unwrap_err().to_string();
            assert!(error.contains("refusing to re-quantize"), "{error}");
            let error = host
                .quantize_onto(quant, &Device::Cpu)
                .unwrap_err()
                .to_string();
            assert!(error.contains("refusing to re-quantize"), "{error}");
            let error = host
                .quantize_dequant_onto(quant, &Device::Cpu)
                .unwrap_err()
                .to_string();
            assert!(error.contains("refusing to re-quantize"), "{error}");
        }
        assert!(
            host.is_nvfp4(),
            "the base must still be NVFP4 after the refusals"
        );
        assert!(host.base_nvfp4().is_some());
        assert!(host.matmul_strategy().is_none());
        assert!(host.quant_dtype().is_none());
    }

    /// A same-device migration is the ordinary no-op (and still migrates residual factors); the
    /// NVFP4 base is never re-staged behind the caller's back.
    #[test]
    fn nvfp4_base_same_device_migration_is_a_noop() {
        let (out_dim, in_dim, rank) = (32, 64, 4);
        let mut host = nvfp4_host(out_dim, in_dim);
        let (a, b) = lora_pair(in_dim, rank, out_dim);
        host.push_lora_checked(a, b, 0.5).unwrap();
        host.to_device(&Device::Cpu).unwrap();
        assert!(host.is_nvfp4());
        assert!(host.is_adapted());
    }

    /// The checked pushes are not NVFP4-only: they are the shared admission gate, so a dense host
    /// validates identically (and a valid factor still installs).
    #[test]
    fn checked_push_validates_a_dense_host_too() {
        let (out_dim, in_dim, rank) = (16, 8, 2);
        let w = Tensor::from_vec(
            (0..out_dim * in_dim)
                .map(|i| i as f32 * 0.01)
                .collect::<Vec<_>>(),
            (out_dim, in_dim),
            &Device::Cpu,
        )
        .unwrap();
        let mut host = AdaptLinear::from_dense(CandleNnLinear::new(w, None), in_dim, out_dim);
        let (a, b) = lora_pair(in_dim, rank, out_dim);
        host.push_lora_checked(a, b, 1.0).unwrap();
        assert!(host.is_adapted());

        let (a, b) = lora_pair(in_dim + 1, rank, out_dim);
        let error = host.push_lora_checked(a, b, 1.0).unwrap_err().to_string();
        assert!(error.contains("dense base"), "{error}");
    }

    /// `fold_dense_delta` adds the delta to a dense base (output moves by exactly `x·δᵀ`, kept at
    /// the base dtype) and refuses a packed base and a mis-shaped delta with an error, never a
    /// silent dequantize or skip.
    #[test]
    #[ignore = "model inference; explicit model workflow only"]
    fn fold_dense_delta_adds_on_dense_and_refuses_packed() {
        let dev = Device::Cpu;
        let (out_dim, in_dim) = (2usize, 64usize);
        let w = deterministic_weight(out_dim, in_dim, DType::F32);
        let mut dense = AdaptLinear::from_dense(CandleNnLinear::new(w, None), in_dim, out_dim);
        let x = Tensor::ones((1, in_dim), DType::F32, &dev).unwrap();
        let before = dense.forward(&x).unwrap();
        let delta = Tensor::full(0.5f32, (out_dim, in_dim), &dev).unwrap();
        dense.fold_dense_delta(&delta).unwrap();
        let after = dense.forward(&x).unwrap();
        let moved: Vec<f32> = (after - before)
            .unwrap()
            .flatten_all()
            .unwrap()
            .to_vec1()
            .unwrap();
        for value in moved {
            assert!((value - 32.0).abs() < 1e-4, "x·δᵀ = 64·0.5, got {value}");
        }
        assert!(!dense.is_adapted(), "a fold attaches no residual");
        let wrong = Tensor::zeros((out_dim, in_dim + 1), DType::F32, &dev).unwrap();
        assert!(dense.fold_dense_delta(&wrong).is_err());

        let (wq, s, b, _grid) = q4_packed(out_dim, in_dim);
        let packed = QLinear::from_packed(&wq, &s, &b, None, &dev).unwrap();
        let mut packed = AdaptLinear::from_packed(packed, in_dim, out_dim);
        match packed.fold_dense_delta(&delta) {
            Err(CandleError::Unsupported(message)) => assert!(message.contains("packed")),
            other => panic!("a packed base must refuse the fold, got {other:?}"),
        }

        #[cfg(feature = "model-inference-proof")]
        println!("\nMODEL_INFERENCE_BODY_COMPLETED quant::adapt::tests::fold_dense_delta_adds_on_dense_and_refuses_packed");
    }
}
