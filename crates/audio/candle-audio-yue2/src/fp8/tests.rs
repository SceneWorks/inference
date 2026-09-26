//! Tests of the experimental FP8 AR mode (sc-22995).
//!
//! CPU (every CI lane): the upstream-parity of the E4M3 quantizer, and the explicit refusals off
//! CUDA. CUDA (`--features cuda` on a CUDA device, the Windows CUDA lanes): the FP8 GEMM against
//! its dequantized reference, preparation / bit-exact restoration on a loaded model, and the
//! engine's stage lifecycle (FP8 for the AR stages, the exact BF16 originals for the acoustic
//! stage).

use super::*;

use crate::model::{synthetic as mot, MotPaths};

fn fixture() -> Value {
    serde_json::from_str(include_str!("../../tests/fixtures/fp8_quantize.json")).unwrap()
}

fn bf16_tensor(bits: &[i64], shape: &[usize]) -> Tensor {
    let values: Vec<f32> = bits
        .iter()
        .map(|&b| f32::from_bits(((b as i16 as u16) as u32) << 16))
        .collect();
    Tensor::from_vec(values, shape, &Device::Cpu)
        .unwrap()
        .to_dtype(DType::BF16)
        .unwrap()
}

/// `quantize_e4m3` reproduces upstream `quantize_tensor` exactly — the F32 scale bit for bit and
/// every E4M3 code — on weight-like, outlier, subnormal-range and all-zero inputs.
///
/// Mutations run: dropping the `1e-12` amax floor fails the zeros case (scale 0); multiplying by
/// the reciprocal scale instead of dividing fails `activation_outliers` (7 of 512 codes differ in
/// the last bit).
#[test]
fn e4m3_quantization_matches_upstream() {
    let f = fixture();
    for case in f["cases"].as_array().unwrap() {
        let name = case["name"].as_str().unwrap();
        let shape: Vec<usize> = case["shape"]
            .as_array()
            .unwrap()
            .iter()
            .map(|d| d.as_u64().unwrap() as usize)
            .collect();
        let bits: Vec<i64> = case["input_bf16_bits"]
            .as_array()
            .unwrap()
            .iter()
            .map(|b| b.as_i64().unwrap())
            .collect();
        let x = bf16_tensor(&bits, &shape);
        let (q, scale) = quantize_e4m3(&x).unwrap();
        assert_eq!(q.dtype(), DType::F8E4M3, "{name}");
        assert_eq!(
            scale.to_bits(),
            case["scale_f32_bits"].as_u64().unwrap() as u32,
            "{name}: scale {scale:e}"
        );
        let got: Vec<f32> = q
            .to_dtype(DType::F32)
            .unwrap()
            .flatten_all()
            .unwrap()
            .to_vec1()
            .unwrap();
        let want: Vec<f32> = case["q_values"]
            .as_array()
            .unwrap()
            .iter()
            .map(|v| v.as_f64().unwrap() as f32)
            .collect();
        let differ = got
            .iter()
            .zip(&want)
            .filter(|(g, w)| g.to_bits() != w.to_bits() && !(**g == 0.0 && **w == 0.0))
            .count();
        assert_eq!(differ, 0, "{name}: {differ} of {} codes differ", want.len());
    }
}

/// The tensor's extreme maps to exactly ±448 with scale `amax / 448`, and nothing is NaN. (The
/// `±448` clamp is upstream's; for finite input `|x| / scale` cannot exceed 448 by more than an
/// ulp, which rounds back to 448, so the clamp is a guard rather than an observable step.)
#[test]
fn quantize_clamps_at_the_e4m3_limit() {
    let x = Tensor::new(&[1.0f32, -3.0, 2.9999], &Device::Cpu).unwrap();
    let (q, scale) = quantize_e4m3(&x).unwrap();
    assert_eq!(scale, 3.0 / E4M3_MAX);
    let v: Vec<f32> = q.to_dtype(DType::F32).unwrap().to_vec1().unwrap();
    assert!(
        v.iter().all(|x| x.is_finite() && x.abs() <= E4M3_MAX),
        "{v:?}"
    );
    assert_eq!(v[1], -E4M3_MAX);
}

/// Off CUDA — and for a quantized tier or an F32 model anywhere — the mode is refused with
/// `Unsupported` naming the reason; the model is left untouched and nothing falls back.
#[test]
fn fp8_is_refused_explicitly_off_cuda_and_off_the_bf16_tier() {
    let unsupported = |r: gen_core::Result<()>| match r {
        Err(gen_core::Error::Unsupported(m)) => m,
        other => panic!("expected Unsupported, got {other:?}"),
    };
    let m = unsupported(check_fp8_request(Tier::Bf16, DType::BF16, &Device::Cpu));
    assert!(m.contains("CUDA") && m.contains("8.9"), "{m}");
    let m = unsupported(check_fp8_request(Tier::Q8, DType::BF16, &Device::Cpu));
    assert!(m.contains("q8 tier"), "{m}");
    let m = unsupported(check_fp8_request(Tier::Bf16, DType::F32, &Device::Cpu));
    assert!(m.contains("BF16 compute"), "{m}");

    let mut lm = mot::model(MotPaths::ArAndNar);
    let before = lm.weight_residency();
    let err = prepare_fp8_ar(&mut lm).unwrap_err();
    assert!(matches!(err, gen_core::Error::Unsupported(_)), "{err}");
    assert!(lm.fp8.is_none());
    for layer in lm.layers() {
        assert!(layer.ar.projections().iter().all(|p| p.dense().is_some()));
    }
    assert_eq!(lm.weight_residency(), before);
    assert_eq!(
        status(&lm),
        Fp8Status {
            active: false,
            active_ar_linears: 0,
            device_fp8_bytes: 0,
            host_original_bytes: 0,
        }
    );
    restore_ar_bf16(&mut lm).unwrap();
}

/// The engine refuses an FP8 request on the CPU before it reads anything: the error is the
/// capability refusal, not the cache miss of the absent snapshot.
#[test]
fn the_engine_refuses_fp8_before_loading() {
    use crate::engine::{EngineOptions, ModelPrecision, Yue2Engine};
    let err = Yue2Engine::load_with_precision(
        &crate::SnapshotDirs::new(),
        DType::F32,
        &Device::Cpu,
        ModelPrecision {
            tier: None,
            ar: ArPrecision::Fp8,
        },
        crate::GenerationConfig::default(),
        EngineOptions::default(),
    )
    .map(|_| ())
    .unwrap_err();
    assert!(matches!(err, gen_core::Error::Unsupported(_)), "{err}");
}

#[cfg(feature = "cuda")]
mod cuda {
    //! Run on a CUDA device of compute capability >= 8.9 (the Windows CUDA lanes). A missing
    //! device fails these tests rather than skipping them: they only compile under `--features
    //! cuda`, which is only built where the device is expected.

    use candle_audio::candle_core::{DType, Device, Tensor};

    use super::super::*;
    use crate::model::{synthetic as mot, MotPaths};
    use crate::weights::Proj;

    fn cuda() -> Device {
        Device::new_cuda(0).expect("a CUDA device (these tests run on the CUDA lanes)")
    }

    /// The BF16 bit patterns of `t` (read on the host).
    fn host_bits(t: &Tensor) -> Vec<u16> {
        let f: Vec<f32> = t
            .to_device(&Device::Cpu)
            .unwrap()
            .to_dtype(DType::BF16)
            .unwrap()
            .to_dtype(DType::F32)
            .unwrap()
            .flatten_all()
            .unwrap()
            .to_vec1()
            .unwrap();
        // BF16 → F32 is exact, so the upper half of the F32 bits is the BF16 bit pattern.
        f.iter().map(|x| (x.to_bits() >> 16) as u16).collect()
    }

    /// The FP8 GEMM computes `(q_x · s_x) · (q_w · s_w)ᵀ` with FP32 accumulation: compared with
    /// that product evaluated in F32 on the host from the same E4M3 codes, only the BF16 output
    /// rounding remains (relative ≤ 2⁻⁸). A transposed operand, a swapped or dropped scale, or an
    /// unpadded row count fails it by orders of magnitude.
    #[test]
    fn fp8_gemm_matches_its_dequantized_reference() {
        let dev = cuda();
        let (n, k, m) = (48usize, 64usize, 5usize);
        let w: Vec<f32> = (0..n * k)
            .map(|i| ((i * 37 % 101) as f32 - 50.0) / 97.0)
            .collect();
        let x: Vec<f32> = (0..m * k)
            .map(|i| ((i * 13 % 29) as f32 - 14.0) / 11.0)
            .collect();
        let w = Tensor::from_vec(w, (n, k), &Device::Cpu)
            .unwrap()
            .to_dtype(DType::BF16)
            .unwrap();
        let x = Tensor::from_vec(x, (1, m, k), &Device::Cpu)
            .unwrap()
            .to_dtype(DType::BF16)
            .unwrap();
        let lt =
            std::sync::Arc::new(candle_quant_kernels::cublaslt::CublasLt::new(&dev).unwrap());
        assert!(lt.meets_fp8_floor().unwrap(), "the CUDA lane needs sm_89+");
        let (qw, sw) = quantize_e4m3(&w).unwrap();
        let weight = Fp8Weight {
            w: lt.stage_fp8(&qw.to_device(&dev).unwrap()).unwrap(),
            scale: sw,
            shape: (n, k),
            device: dev.clone(),
            lt: std::sync::Arc::clone(&lt),
        };
        let got = weight
            .forward(&x.to_device(&dev).unwrap(), None)
            .unwrap()
            .to_dtype(DType::F32)
            .unwrap()
            .to_device(&Device::Cpu)
            .unwrap();
        assert_eq!(got.dims(), &[1, m, n]);
        // Reference: the same activation codes (quantized over the zero-padded rows, as the GEMM
        // saw them), dequantized, multiplied in F32.
        let padded = Tensor::cat(
            &[
                &x.reshape((m, k)).unwrap(),
                &Tensor::zeros((ROW_ALIGN - m, k), DType::BF16, &Device::Cpu).unwrap(),
            ],
            0,
        )
        .unwrap();
        let (qx, sx) = quantize_e4m3(&padded).unwrap();
        let dq = |q: &Tensor, s: f32| (q.to_dtype(DType::F32).unwrap() * s as f64).unwrap();
        let want = dq(&qx, sx)
            .narrow(0, 0, m)
            .unwrap()
            .matmul(&dq(&qw, sw).t().unwrap())
            .unwrap();
        let (got, want): (Vec<f32>, Vec<f32>) = (
            got.flatten_all().unwrap().to_vec1().unwrap(),
            want.flatten_all().unwrap().to_vec1().unwrap(),
        );
        let scale = want.iter().fold(0f32, |a, b| a.max(b.abs()));
        let worst = got
            .iter()
            .zip(&want)
            .map(|(g, w)| (g - w).abs())
            .fold(0f32, f32::max);
        println!("fp8 GEMM vs dequantized reference: max |Δ| {worst:e} (scale {scale:e})");
        assert!(
            worst <= scale * 2f32.powi(-8),
            "max |Δ| {worst} vs scale {scale}"
        );
    }

    /// Preparing swaps exactly the 7 AR projections of every layer (never `lm_head` or a NAR
    /// twin), counts the host-held originals in the residency, keeps the logits close to BF16, and
    /// restoring puts back tensors **bit-identical** to the originals read before preparation —
    /// the tensors the acoustic stage then runs.
    #[test]
    fn prepare_then_restore_is_bit_exact_and_counted() {
        let dev = cuda();
        let mut lm = mot::model_on(MotPaths::ArAndNar, &dev, DType::BF16);
        let originals: Vec<Vec<Vec<u16>>> = lm
            .layers()
            .iter()
            .map(|l| {
                l.ar.projections()
                    .iter()
                    .map(|p| host_bits(p.dense().unwrap()))
                    .collect()
            })
            .collect();
        let nar_before: Vec<Vec<u16>> = lm
            .layers()
            .iter()
            .flat_map(|l| l.nar.as_ref().unwrap().projections())
            .map(|p| host_bits(p.dense().unwrap()))
            .collect();
        let before = lm.weight_residency();
        let ids = [151643u32, 40, 1234, 99, 151847, 5, 777];
        let reference = {
            let mut cache = lm.new_cache(ids.len()).unwrap();
            lm.prefill(&ids, &mut cache, || Ok(())).unwrap()
        };

        let s = prepare_fp8_ar(&mut lm).unwrap();
        assert!(s.active);
        assert_eq!(
            s.active_ar_linears,
            lm.layers().len() * AR_LINEARS_PER_LAYER
        );
        let ar_params: u64 = originals.iter().flatten().map(|v| v.len() as u64).sum();
        assert_eq!(s.host_original_bytes, ar_params * 2);
        assert_eq!(
            s.device_fp8_bytes,
            ar_params + 4 * s.active_ar_linears as u64
        );
        let during = lm.weight_residency();
        assert_eq!(during.host_bytes, ar_params * 2);
        assert_eq!(
            during.device_bytes,
            before.device_bytes - ar_params * 2 + s.device_fp8_bytes
        );
        assert!(lm.offload_ar().is_err(), "offloading FP8 weights is refused");
        let fp8_logits = {
            let mut cache = lm.new_cache(ids.len()).unwrap();
            lm.prefill(&ids, &mut cache, || Ok(())).unwrap()
        };
        let (a, b): (Vec<f32>, Vec<f32>) = (
            reference.to_dtype(DType::F32).unwrap().to_vec1().unwrap(),
            fp8_logits.to_dtype(DType::F32).unwrap().to_vec1().unwrap(),
        );
        let rel = a
            .iter()
            .zip(&b)
            .map(|(x, y)| (x - y).powi(2))
            .sum::<f32>()
            .sqrt()
            / a.iter().map(|x| x * x).sum::<f32>().sqrt();
        println!("synthetic FP8 AR vs BF16 logits: rel L2 {rel:e}");
        assert!(rel.is_finite() && rel < 0.25, "rel L2 {rel}");

        restore_ar_bf16(&mut lm).unwrap();
        assert!(lm.fp8.is_none());
        for (l, want) in lm.layers().iter().zip(&originals) {
            for (p, w) in l.ar.projections().iter().zip(want) {
                assert_eq!(&host_bits(p.dense().expect("restored dense")), w);
            }
        }
        let nar_after: Vec<Vec<u16>> = lm
            .layers()
            .iter()
            .flat_map(|l| l.nar.as_ref().unwrap().projections())
            .map(|p| host_bits(p.dense().unwrap()))
            .collect();
        assert_eq!(nar_after, nar_before, "the NAR twins are never touched");
        assert_eq!(lm.weight_residency(), before);
        let restored = {
            let mut cache = lm.new_cache(ids.len()).unwrap();
            lm.prefill(&ids, &mut cache, || Ok(())).unwrap()
        };
        assert_eq!(
            host_bits(&restored),
            host_bits(&reference),
            "restored BF16 logits are the original logits"
        );
        // Idempotent in both directions.
        restore_ar_bf16(&mut lm).unwrap();
        prepare_fp8_ar(&mut lm).unwrap();
        prepare_fp8_ar(&mut lm).unwrap();
        assert!(matches!(lm.layers()[0].ar.projections()[0], Proj::Fp8(_)));
        restore_ar_bf16(&mut lm).unwrap();
    }

    /// The engine lifecycle: an FP8 engine runs the AR stages in FP8, and the acoustic stage on the
    /// exact BF16 originals — restored before the NAR prefill — then prepares FP8 again for the
    /// next AR stage. The acoustic stage's latents and identity equal a native engine's.
    #[test]
    fn the_engine_restores_bf16_before_the_acoustic_stage() {
        use crate::engine::{EngineHooks, SongSettings, Yue2Engine};
        let dev = cuda();
        let fp8 = Yue2Engine::synthetic_on(&dev, DType::BF16, ArPrecision::Fp8);
        let native = Yue2Engine::synthetic_on(&dev, DType::BF16, ArPrecision::Native);
        assert!(fp8.fp8_status().unwrap().active, "prepared at load");
        use crate::protocol::{
            GenerationConfig, Sampling, SamplingOverrides, SongRequest, SongRequestSpec,
        };
        let mut spec = SongRequestSpec::new("warm piano pop", "[Verse]\nla la la\n");
        spec.seed = 7;
        let request = SongRequest::new(spec).unwrap();
        // Small token budgets and two midpoint steps (the run tests' settings).
        let o = |min: i64, max: i64| SamplingOverrides {
            min_tokens: Some(min),
            max_tokens: Some(max),
            ..Default::default()
        };
        let settings = SongSettings {
            generation: GenerationConfig::new(
                Sampling::abc_default().with_overrides(&o(2, 6)).unwrap(),
                Sampling::semantic_default()
                    .with_overrides(&o(12, 16))
                    .unwrap(),
                2,
            )
            .unwrap(),
            ..SongSettings::default()
        };
        let mut hooks = EngineHooks {
            cancelled: &|| false,
            observer: &mut (),
        };
        let plan = fp8
            .plan(&request, &settings.generation, &mut hooks)
            .unwrap();
        assert!(fp8.fp8_status().unwrap().active);
        let semantic = fp8
            .generate_semantic(&plan, &settings.generation, &mut hooks)
            .unwrap();
        let synthesis = fp8
            .synthesize(&semantic, &settings.generation, &mut hooks)
            .unwrap();
        assert!(
            !fp8.fp8_status().unwrap().active,
            "the acoustic stage ran on the restored BF16 AR path"
        );
        let reference = native
            .synthesize(&semantic, &settings.generation, &mut hooks)
            .unwrap();
        assert_eq!(
            synthesis.latents.values(),
            reference.latents.values(),
            "the BF16-restored acoustic stage equals the native engine's"
        );
        assert_eq!(
            fp8.synthesis_stage_identity(&semantic, &settings.generation)
                .unwrap(),
            native
                .synthesis_stage_identity(&semantic, &settings.generation)
                .unwrap()
        );
        assert_ne!(
            fp8.semantic_stage_identity(&plan, settings.generation.semantic())
                .unwrap(),
            native
                .semantic_stage_identity(&plan, settings.generation.semantic())
                .unwrap(),
            "FP8 AR tokens are never reused for a native run"
        );
        fp8.generate_semantic(&plan, &settings.generation, &mut hooks)
            .unwrap();
        assert!(
            fp8.fp8_status().unwrap().active,
            "prepared again for the AR stage"
        );
    }
}
