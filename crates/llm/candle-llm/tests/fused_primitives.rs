//! The fused decode primitives behind the shared entry points (sc-24137, epic sc-24128).
//!
//! `primitives::rms_norm` / `rms_norm_residual` / `swiglu` (`nn.rs`) and `rms_norm_rope`
//! (`rope.rs`) each pick the fused CUDA kernel or the op-chain reference, and record which one ran
//! in the thread's fused tally. These tests pin that contract from the caller's side:
//!
//! - on a CPU tensor (every build) the reference runs and says why (`cuda_feature_off` on a
//!   non-CUDA build, `not_cuda` on a CUDA build), with output identical to the reference chain;
//! - on CUDA, fused-on and fused-off outputs are **bit-identical** on the qwen3_5-27B shapes and
//!   the edge shapes, the tally says `fused` / `reference` (`disabled`) accordingly;
//! - an input the kernel does not serve (f16) runs the reference visibly (`dtype`), never a
//!   silent downgrade or an error.
//!
//! The `#[ignore]`d real-weight test is the story's AC2 as a test: Qwen3.8-27B greedy decode of the
//! 256-token fixture, fused on vs off, token-identical.

use candle_core::{DType, Device, Tensor};
use candle_llm::primitives::{
    apply_rope, fused_kernels_enabled, fused_tally, rms_norm, rms_norm_reference,
    rms_norm_residual, rms_norm_rope, set_fused_kernels, silu, swiglu, FusedTally,
};

fn ramp(n: usize, seed: u64, scale: f32) -> Vec<f32> {
    let mut x = seed.max(1);
    (0..n)
        .map(|_| {
            x ^= x << 13;
            x ^= x >> 7;
            x ^= x << 17;
            (((x >> 40) as f32 / (1u64 << 24) as f32) * 2.0 - 1.0) * scale
        })
        .collect()
}

fn tensor(dims: &[usize], seed: u64, scale: f32, dtype: DType, dev: &Device) -> Tensor {
    let n: usize = dims.iter().product();
    Tensor::from_vec(ramp(n, seed, scale), dims, dev)
        .unwrap()
        .to_dtype(dtype)
        .unwrap()
}

fn bits(t: &Tensor) -> Vec<u32> {
    match t.dtype() {
        DType::F32 => t
            .flatten_all()
            .unwrap()
            .to_vec1::<f32>()
            .unwrap()
            .into_iter()
            .map(f32::to_bits)
            .collect(),
        DType::BF16 => t
            .flatten_all()
            .unwrap()
            .to_vec1::<half::bf16>()
            .unwrap()
            .into_iter()
            .map(|v| u32::from(v.to_bits()))
            .collect(),
        DType::F16 => t
            .flatten_all()
            .unwrap()
            .to_vec1::<half::f16>()
            .unwrap()
            .into_iter()
            .map(|v| u32::from(v.to_bits()))
            .collect(),
        other => panic!("{other:?}"),
    }
}

fn assert_identical(what: &str, a: &Tensor, b: &Tensor) {
    assert_eq!(a.dims(), b.dims(), "{what}: shape");
    let (ab, bb) = (bits(a), bits(b));
    let differing = ab.iter().zip(&bb).filter(|(x, y)| x != y).count();
    assert_eq!(
        differing,
        0,
        "{what}: {differing}/{} elements differ",
        ab.len()
    );
}

/// Every entry point once, on one `(x, residual, w, gate, up, q, cos, sin)` set; returns the
/// outputs so two policies can be compared.
fn run_all(dev: &Device, dtype: DType, eps: f64) -> Vec<(&'static str, Tensor)> {
    let x = tensor(&[1, 7, 5120], 1, 3.0, dtype, dev);
    let r = tensor(&[1, 7, 5120], 2, 3.0, dtype, dev);
    let w = tensor(&[5120], 3, 1.5, dtype, dev);
    let gate = tensor(&[1, 7, 17408], 4, 12.0, dtype, dev);
    let up = tensor(&[1, 7, 17408], 5, 4.0, dtype, dev);
    let q = tensor(&[1, 7, 24, 256], 6, 3.0, dtype, dev);
    let qw = tensor(&[256], 7, 1.5, dtype, dev);
    let angles = tensor(&[1, 7, 64], 8, 6.0, DType::F32, dev);
    let cos = angles.cos().unwrap().to_dtype(dtype).unwrap();
    let sin = angles.sin().unwrap().to_dtype(dtype).unwrap();
    // Decode-shaped (seq 1) and an odd hidden size too.
    let x1 = tensor(&[1, 1, 5120], 9, 3.0, dtype, dev);
    let xo = tensor(&[2, 3, 1000], 10, 3.0, dtype, dev);
    let wo = tensor(&[1000], 11, 1.5, dtype, dev);

    let (h, normed) = rms_norm_residual(&x, &r, &w, eps).unwrap();
    vec![
        ("rms_norm", rms_norm(&x, &w, eps).unwrap()),
        ("rms_norm seq1", rms_norm(&x1, &w, eps).unwrap()),
        ("rms_norm odd", rms_norm(&xo, &wo, eps).unwrap()),
        ("rms_norm_residual h", h),
        ("rms_norm_residual normed", normed),
        ("swiglu", swiglu(&gate, &up).unwrap()),
        (
            "rms_norm_rope",
            rms_norm_rope(&q, &qw, eps, &cos, &sin, false).unwrap(),
        ),
    ]
}

fn reference_all(dev: &Device, dtype: DType, eps: f64) -> Vec<(&'static str, Tensor)> {
    let x = tensor(&[1, 7, 5120], 1, 3.0, dtype, dev);
    let r = tensor(&[1, 7, 5120], 2, 3.0, dtype, dev);
    let w = tensor(&[5120], 3, 1.5, dtype, dev);
    let gate = tensor(&[1, 7, 17408], 4, 12.0, dtype, dev);
    let up = tensor(&[1, 7, 17408], 5, 4.0, dtype, dev);
    let q = tensor(&[1, 7, 24, 256], 6, 3.0, dtype, dev);
    let qw = tensor(&[256], 7, 1.5, dtype, dev);
    let angles = tensor(&[1, 7, 64], 8, 6.0, DType::F32, dev);
    let cos = angles.cos().unwrap().to_dtype(dtype).unwrap();
    let sin = angles.sin().unwrap().to_dtype(dtype).unwrap();
    let x1 = tensor(&[1, 1, 5120], 9, 3.0, dtype, dev);
    let xo = tensor(&[2, 3, 1000], 10, 3.0, dtype, dev);
    let wo = tensor(&[1000], 11, 1.5, dtype, dev);

    let h = x.broadcast_add(&r).unwrap();
    vec![
        ("rms_norm", rms_norm_reference(&x, &w, eps).unwrap()),
        ("rms_norm seq1", rms_norm_reference(&x1, &w, eps).unwrap()),
        ("rms_norm odd", rms_norm_reference(&xo, &wo, eps).unwrap()),
        ("rms_norm_residual h", h.clone()),
        (
            "rms_norm_residual normed",
            rms_norm_reference(&h, &w, eps).unwrap(),
        ),
        ("swiglu", silu(&gate).unwrap().broadcast_mul(&up).unwrap()),
        (
            "rms_norm_rope",
            apply_rope(
                &rms_norm_reference(&q, &qw, eps).unwrap(),
                &cos,
                &sin,
                false,
            )
            .unwrap(),
        ),
    ]
}

fn tally_since(start: &FusedTally) -> FusedTally {
    fused_tally().since(start)
}

const EPS: f64 = 1e-6;
/// Leaves `run_all` calls: 3 rms_norm + 1 residual + 1 swiglu + 1 rope.
const LEAVES: u64 = 6;

#[test]
fn a_cpu_tensor_takes_the_reference_path_and_says_why() {
    let dev = Device::Cpu;
    set_fused_kernels(None);
    let start = fused_tally();
    let got = run_all(&dev, DType::F32, EPS);
    let tally = tally_since(&start);
    assert_eq!(tally.fused, 0, "{tally:?}");
    assert_eq!(tally.reference, LEAVES, "{tally:?}");
    assert_eq!(tally.label(), "reference");
    let expected_reason = if cfg!(feature = "cuda") {
        "not_cuda"
    } else {
        "cuda_feature_off"
    };
    assert_eq!(tally.reference_reason, Some(expected_reason));
    for ((name, a), (_, b)) in got.iter().zip(reference_all(&dev, DType::F32, EPS)) {
        assert_identical(name, a, &b);
    }
}

#[test]
fn switching_the_fused_path_off_is_recorded_as_disabled() {
    set_fused_kernels(Some(false));
    assert!(!fused_kernels_enabled());
    let start = fused_tally();
    let _ = rms_norm(
        &tensor(&[1, 4], 1, 1.0, DType::F32, &Device::Cpu),
        &tensor(&[4], 2, 1.0, DType::F32, &Device::Cpu),
        EPS,
    )
    .unwrap();
    let tally = tally_since(&start);
    set_fused_kernels(None);
    assert_eq!(tally.reference, 1);
    // Off beats "not cuda": the switch is checked first, so the reason names the switch on a CUDA
    // build and the build on a CPU one.
    let expected = if cfg!(feature = "cuda") {
        "disabled"
    } else {
        "cuda_feature_off"
    };
    assert_eq!(tally.reference_reason, Some(expected));
}

#[cfg(feature = "cuda")]
mod cuda {
    use super::*;

    fn device() -> Option<Device> {
        Device::new_cuda(0).ok()
    }

    #[test]
    fn fused_on_and_off_are_bit_identical_and_both_visible() {
        let Some(dev) = device() else { return };
        for dtype in [DType::F32, DType::BF16] {
            set_fused_kernels(Some(true));
            let start = fused_tally();
            let on = run_all(&dev, dtype, EPS);
            let t_on = tally_since(&start);
            assert_eq!(t_on.fused, LEAVES, "{dtype:?}: {t_on:?}");
            assert_eq!(t_on.reference, 0, "{dtype:?}: {t_on:?}");
            assert_eq!(t_on.label(), "fused");

            set_fused_kernels(Some(false));
            let start = fused_tally();
            let off = run_all(&dev, dtype, EPS);
            let t_off = tally_since(&start);
            set_fused_kernels(None);
            assert_eq!(t_off.fused, 0, "{dtype:?}: {t_off:?}");
            assert_eq!(t_off.reference, LEAVES, "{dtype:?}: {t_off:?}");
            assert_eq!(t_off.reference_reason, Some("disabled"));
            assert_eq!(t_off.label(), "reference");

            let reference = reference_all(&dev, dtype, EPS);
            for (((name, a), (_, b)), (_, c)) in on.iter().zip(&off).zip(&reference) {
                assert_identical(&format!("{dtype:?} {name} on-vs-off"), a, b);
                assert_identical(&format!("{dtype:?} {name} off-vs-reference"), b, c);
            }
        }
    }

    #[test]
    fn an_unserved_dtype_runs_the_reference_visibly() {
        let Some(dev) = device() else { return };
        set_fused_kernels(Some(true));
        let x = tensor(&[1, 3, 512], 1, 3.0, DType::F16, &dev);
        let w = tensor(&[512], 2, 1.5, DType::F16, &dev);
        let start = fused_tally();
        let got = rms_norm(&x, &w, EPS).unwrap();
        let tally = tally_since(&start);
        set_fused_kernels(None);
        assert_eq!(tally.fused, 0);
        assert_eq!(tally.reference, 1);
        assert_eq!(tally.reference_reason, Some("dtype"));
        assert_identical(
            "f16 rms_norm",
            &got,
            &rms_norm_reference(&x, &w, EPS).unwrap(),
        );
    }

    #[test]
    fn a_shape_the_kernel_refuses_runs_the_reference_visibly() {
        let Some(dev) = device() else { return };
        set_fused_kernels(Some(true));
        // A head wider than the fused RoPE kernel's staging buffer.
        let q = tensor(&[1, 2, 1, 2048], 1, 3.0, DType::BF16, &dev);
        let w = tensor(&[2048], 2, 1.5, DType::BF16, &dev);
        let angles = tensor(&[1, 2, 64], 3, 6.0, DType::F32, &dev);
        let cos = angles.cos().unwrap().to_dtype(DType::BF16).unwrap();
        let sin = angles.sin().unwrap().to_dtype(DType::BF16).unwrap();
        let start = fused_tally();
        let got = rms_norm_rope(&q, &w, EPS, &cos, &sin, false).unwrap();
        let tally = tally_since(&start);
        set_fused_kernels(None);
        assert_eq!(tally.fused, 0);
        assert_eq!(tally.reference, 1);
        assert_eq!(tally.reference_reason, Some("head_dim_too_large"));
        let want =
            apply_rope(&rms_norm_reference(&q, &w, EPS).unwrap(), &cos, &sin, false).unwrap();
        assert_identical("wide-head rms_norm_rope", &got, &want);
    }

    /// AC2: Qwen3.8-27B greedy decode of the 256-token fixture is token-identical with the fused
    /// primitives on vs off, and every leaf of the fused run ran fused. Needs
    /// `FUSED_PARITY_SNAPSHOT` (the snapshot directory) and a GPU; the decode bench records the
    /// same comparison with timings.
    #[test]
    #[ignore = "real weights: needs FUSED_PARITY_SNAPSHOT and a GPU"]
    fn qwen38_27b_greedy_decode_is_token_identical_fused_on_vs_off() {
        use candle_llm::decode::{
            generate_from_prefill, CancelFlag, CountingDecode, GenerationConfig,
        };
        use candle_llm::models::{Qwen35Config, Qwen35Model};
        use candle_llm::primitives::{input_ids, Weights};
        use core_llm::{ChatTemplate, JinjaChatTemplate, Message, RenderOptions, Tokenizer};
        use std::path::PathBuf;

        let snapshot =
            PathBuf::from(std::env::var("FUSED_PARITY_SNAPSHOT").expect("FUSED_PARITY_SNAPSHOT"));
        let new_tokens: usize = std::env::var("FUSED_PARITY_NEW_TOKENS")
            .ok()
            .and_then(|v| v.parse().ok())
            .unwrap_or(256);
        let dev = candle_llm::device::select_device().expect("device");
        assert!(dev.is_cuda(), "the parity run needs the CUDA device");
        let config: serde_json::Value =
            serde_json::from_str(&std::fs::read_to_string(snapshot.join("config.json")).unwrap())
                .unwrap();
        let cfg = Qwen35Config::from_json(&config).expect("qwen3_5 config");
        let weights = Weights::from_dir(&snapshot, &dev).expect("weights");
        let prefix = if weights.contains("model.language_model.embed_tokens.weight") {
            "model.language_model"
        } else {
            "model"
        };
        let model = Qwen35Model::from_weights(&weights, prefix, cfg).expect("model");

        let tokenizer =
            Tokenizer::from_file(snapshot.join("tokenizer.json")).expect("tokenizer.json");
        let template = match std::fs::read_to_string(snapshot.join("chat_template.jinja")) {
            Ok(source) if !source.trim().is_empty() => JinjaChatTemplate::new(source),
            _ => JinjaChatTemplate::from_tokenizer_config_file(
                snapshot.join("tokenizer_config.json"),
            )
            .expect("chat template"),
        };
        // The sc-24129 decode-bench fixture prompt.
        let user = "Write a detailed, multi-paragraph explanation of how transformer \
            language models generate text. Cover tokenization, self-attention, the key/value \
            cache, and greedy versus sampled decoding, and finish with the trade-offs of \
            speculative decoding.";
        let rendered = template
            .render_with(&[Message::user(user)], &RenderOptions::generation())
            .expect("render");
        let prompt: Vec<i32> = tokenizer
            .encode(&rendered, false)
            .expect("encode")
            .into_iter()
            .map(|id| id as i32)
            .collect();
        let mut config = GenerationConfig {
            max_new_tokens: new_tokens,
            seed: Some(0),
            stop_tokens: Vec::new(),
            ..Default::default()
        };
        config.sampling.temperature = 0.0;

        let decode = |enabled: bool| -> (Vec<i32>, FusedTally) {
            set_fused_kernels(Some(enabled));
            let start = fused_tally();
            let counted = CountingDecode::new(&model);
            let mut cache = model.new_cache();
            let first = model
                .decode_logits(&input_ids(&prompt, &dev).unwrap(), &mut cache, 0)
                .expect("prefill");
            let out = generate_from_prefill(
                &counted,
                &mut cache,
                first,
                prompt.clone(),
                &config,
                &CancelFlag::new(),
                &mut |_| {},
                None,
            )
            .expect("generate");
            dev.synchronize().unwrap();
            let tally = fused_tally().since(&start);
            set_fused_kernels(None);
            (out.tokens, tally)
        };

        let (off, t_off) = decode(false);
        let (on, t_on) = decode(true);
        eprintln!("[fused_parity] off: {t_off:?}");
        eprintln!("[fused_parity] on:  {t_on:?}");
        assert_eq!(off.len(), new_tokens);
        assert_eq!(t_off.fused, 0, "{t_off:?}");
        assert_eq!(t_off.reference_reason, Some("disabled"));
        assert_eq!(t_on.reference, 0, "a fused-on decode fell back: {t_on:?}");
        assert!(t_on.fused > 0);
        let first_divergence = off.iter().zip(&on).position(|(a, b)| a != b);
        assert_eq!(
            first_divergence, None,
            "fused-on diverged from fused-off at token {first_divergence:?}"
        );
        assert_eq!(off, on);
    }
}
