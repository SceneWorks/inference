//! Real-weight conformance + numerical parity for the candle MMAudio MM-DiT flow-matching
//! generator (sc-13439).
//!
//! ## What this gates on real weights
//!
//! Loads the pinned `hkchengrex/MMAudio` `weights/mmaudio_small_16k.pth` (~629 MB — the network
//! only), builds the MM-DiT + Euler sampler, and asserts:
//!
//! - [`mmdit_flow_shape_finite_deterministic`] — a fixed `(clip, sync, text, latent, t)` →
//!   `predict_flow` produces `(1, 250, 20)` finite velocities, **byte-identical** run-to-run. A
//!   broken weight mapping (wrong key names / transposed conv / mis-ordered QKV) surfaces here as a
//!   load error, NaN, or shape mismatch.
//! - [`mmdit_sample_shape_finite_deterministic`] — the full CFG-4.5 / Euler-25 sample from a fixed
//!   prior → finite, deterministic, correctly-shaped `(1, 250, 20)` **audio latents** (the shape the
//!   16k VAE decodes), and materially different from the input noise.
//! - [`mmdit_matches_reference`] — **numerical parity** against the PyTorch MMAudio network. Feeds
//!   the exact features/latent from the committed fixture and compares (a) a single `predict_flow`
//!   and (b) the full CFG-25 sample with cosine `> 0.999` and small max-abs-diff. This isolates the
//!   ported MM-DiT + sampler from the feature encoders by feeding the exact reference inputs.
//!
//!   Its fixture is produced by `scripts/reference/mmaudio_reference.py` and **committed**
//!   (sc-17285). Before that this test read a directory path from `MMAUDIO_DIT_PARITY_DIR` and,
//!   when it was unset — which it always was, because no script in this repository produced such a
//!   directory — `return`ed early to a *passing* result. A run-count assertion cannot tell that
//!   apart from real work, so sc-17266 had to exclude this test by name from the real-weight lane
//!   rather than inherit a vacuous green. There is no unset case left: the fixture resolves from
//!   `CARGO_MANIFEST_DIR`, and a missing or malformed one fails.
//!
//! `#[ignore]`d and snapshot-gated like every audio family's real-weight tests:
//! ```text
//! cargo test --locked -p candle-audio-mmaudio --test mmdit_conformance -- --ignored --nocapture
//! ```
//! Set `MMAUDIO_DIT_SNAPSHOT` to a `mmaudio_small_16k.pth` file (or a dir containing it under
//! `weights/` or at its root). It is **required**: `load_net` panics when it is unset, because
//! inference never self-fetches and never derives a hub-cache location (epic 13657). There is no
//! hub fallback.

mod common;

use candle_audio_mmaudio as m;
use candle_audio_mmaudio::candle_audio::candle_core::{Device, Tensor};
use candle_audio_mmaudio::mmdit;

const FIXTURE: &str = "mmaudio_parity_mmdit.safetensors";

fn load_net() -> mmdit::MmAudioDit {
    let dev = Device::Cpu;
    // Required env path — inference never self-fetches or derives a cache location (epic 13657).
    let p = std::env::var("MMAUDIO_DIT_SNAPSHOT")
        .expect("set MMAUDIO_DIT_SNAPSHOT to a mmaudio_small_16k.pth file or its snapshot dir");
    let path = std::path::PathBuf::from(&p);
    if path.is_dir() {
        mmdit::load(&m::gen_core::WeightsSource::Dir(path), &dev)
            .expect("load mmaudio small_16k from MMAUDIO_DIT_SNAPSHOT dir")
    } else {
        mmdit::load_from_pth(&path, &dev)
            .expect("load mmaudio small_16k from MMAUDIO_DIT_SNAPSHOT file")
    }
}

/// A deterministic pseudo-random tensor (a fixed LCG) of the given shape — no rng crate needed.
fn fixed(shape: &[usize], seed: u64) -> Tensor {
    let n: usize = shape.iter().product();
    let mut s = seed.wrapping_add(0x9E37_79B9_7F4A_7C15);
    let mut v = Vec::with_capacity(n);
    for _ in 0..n {
        // xorshift64* -> uniform in [-1,1)
        s ^= s >> 12;
        s ^= s << 25;
        s ^= s >> 27;
        let u = (s.wrapping_mul(0x2545_F491_4F6C_DD1D) >> 11) as f64 / (1u64 << 53) as f64;
        v.push((u * 2.0 - 1.0) as f32);
    }
    Tensor::from_vec(v, shape, &Device::Cpu).unwrap()
}

fn max_abs_diff(a: &[f32], b: &[f32]) -> f32 {
    a.iter()
        .zip(b)
        .map(|(x, y)| (x - y).abs())
        .fold(0f32, f32::max)
}

fn vecof(t: &Tensor) -> Vec<f32> {
    t.flatten_all().unwrap().to_vec1().unwrap()
}

#[test]
#[ignore = "downloads ~629MB mmaudio_small_16k.pth; run explicitly with --ignored"]
fn mmdit_flow_shape_finite_deterministic() {
    let net = load_net();
    let clip = fixed(&[1, 64, 1024], 1);
    let sync = fixed(&[1, 192, 768], 2);
    let text = fixed(&[1, 77, 1024], 3);
    let latent = fixed(&[1, 250, 20], 4);
    let t = Tensor::new(&[0.37f32], &Device::Cpu).unwrap();

    let cond = net.preprocess_conditions(&clip, &sync, &text).unwrap();
    let flow = net.predict_flow(&latent, &t, &cond).unwrap();
    assert_eq!(
        flow.dims(),
        &[1, 250, 20],
        "(B, latent_seq_len, latent_dim)"
    );
    let a = vecof(&flow);
    assert!(a.iter().all(|v| v.is_finite()), "all flow values finite");
    // determinism
    let b = vecof(&net.predict_flow(&latent, &t, &cond).unwrap());
    assert_eq!(a, b, "predict_flow must be deterministic run-to-run");
    let mean = a.iter().sum::<f32>() / a.len() as f32;
    let var = a.iter().map(|v| (v - mean).powi(2)).sum::<f32>() / a.len() as f32;
    assert!(var > 1e-6, "flow must not be a constant (var={var})");
    eprintln!(
        "mmdit flow: shape=(1,250,20) mean={mean:.4} var={var:.4} min={:.3} max={:.3}",
        a.iter().cloned().fold(f32::INFINITY, f32::min),
        a.iter().cloned().fold(f32::NEG_INFINITY, f32::max),
    );
}

#[test]
#[ignore = "downloads ~629MB mmaudio_small_16k.pth; run explicitly with --ignored"]
fn mmdit_sample_shape_finite_deterministic() {
    let net = load_net();
    let clip = fixed(&[1, 64, 1024], 11);
    let sync = fixed(&[1, 192, 768], 12);
    let text = fixed(&[1, 77, 1024], 13);
    let x0 = fixed(&[1, 250, 20], 99);

    let cond = net.preprocess_conditions(&clip, &sync, &text).unwrap();
    let out = net.sample_default(&x0, &cond).unwrap();
    assert_eq!(out.dims(), &[1, 250, 20], "audio latents (B,250,20)");
    let a = vecof(&out);
    assert!(a.iter().all(|v| v.is_finite()), "all latents finite");
    // deterministic
    let b = vecof(&net.sample_default(&x0, &cond).unwrap());
    assert_eq!(a, b, "sample must be deterministic run-to-run");
    // materially different from the input noise
    let diff = max_abs_diff(&a, &vecof(&x0));
    assert!(
        diff > 1e-2,
        "output must differ from the input prior (Δ={diff})"
    );
    let mean = a.iter().sum::<f32>() / a.len() as f32;
    eprintln!(
        "mmdit sample: shape=(1,250,20) mean={mean:.4} min={:.3} max={:.3} Δfrom_x0={diff:.3}",
        a.iter().cloned().fold(f32::INFINITY, f32::min),
        a.iter().cloned().fold(f32::NEG_INFINITY, f32::max),
    );
}

#[test]
#[ignore = "numerical parity vs PyTorch MMAudio; needs mmaudio_small_16k.pth. Run with --ignored"]
fn mmdit_matches_reference() {
    let dev = Device::Cpu;
    let tensors = common::load_fixture(FIXTURE, &dev);
    let load = |name: &str| common::fixture_tensor(&tensors, FIXTURE, name);
    let net = load_net();

    // ---- single predict_flow parity (isolates the MM-DiT from the feature encoders) ----
    let clip = load("clip_f");
    let sync = load("sync_f");
    let text = load("text_f");
    let latent = load("latent");
    let t = load("t");
    let cond = net.preprocess_conditions(&clip, &sync, &text).unwrap();
    let got = vecof(&net.predict_flow(&latent, &t, &cond).unwrap());
    let want = common::flat(&load("flow"));
    assert_eq!(got.len(), want.len(), "flow length");
    let f_cos = common::cosine(&got, &want);
    let f_mad = common::max_abs_diff(&got, &want);
    eprintln!("PARITY predict_flow: cos={f_cos:.6} max|Δ|={f_mad:.6}");
    assert!(f_cos > 0.999, "flow cosine {f_cos} must exceed 0.999");
    assert!(f_mad < 0.05, "flow max-abs-diff {f_mad} too large");

    // ---- full CFG-4.5 / Euler-25 sample parity (uses the SAME cond, the reference x0 noise) ----
    // `sample_default` builds its own DEFAULT empty conditions (no negative text), so the fixture
    // is produced from `get_empty_conditions(1)` rather than the negative-text form the assembled
    // pipeline uses — otherwise the two sides would sample different distributions.
    let x0 = load("x0");
    let got_s = vecof(&net.sample_default(&x0, &cond).unwrap());
    let want_s = common::flat(&load("x1_unnorm"));
    assert_eq!(got_s.len(), want_s.len(), "sample length");
    let s_cos = common::cosine(&got_s, &want_s);
    let s_mad = common::max_abs_diff(&got_s, &want_s);
    // scale the tolerance to the reference magnitude (unnormalized latents are O(100s)).
    let ref_absmax = want_s.iter().fold(0f64, |a, v| a.max(v.abs() as f64));
    eprintln!(
        "PARITY sample(unnorm): cos={s_cos:.6} max|Δ|={s_mad:.4} ref|max|={ref_absmax:.2} rel={:.5}",
        s_mad / ref_absmax
    );
    assert!(s_cos > 0.999, "sample cosine {s_cos} must exceed 0.999");
    assert!(
        s_mad / ref_absmax < 0.02,
        "sample relative max-abs-diff {} too large",
        s_mad / ref_absmax
    );
}
