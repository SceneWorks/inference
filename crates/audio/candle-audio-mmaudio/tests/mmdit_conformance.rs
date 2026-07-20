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
//! - [`mmdit_matches_reference`] — **numerical parity** against the PyTorch MMAudio network, run
//!   only when `MMAUDIO_DIT_PARITY_DIR` points at the `ref_dump.py` dump directory. Feeds the exact
//!   dumped features/latent and compares (a) a single `predict_flow` and (b) the full CFG-25 sample
//!   with cosine `> 0.999` and small max-abs-diff. This isolates the ported MM-DiT + sampler from
//!   the feature encoders by feeding the exact reference inputs.
//!
//! `#[ignore]`d and snapshot-gated like every audio family's real-weight tests:
//! ```text
//! cargo test --locked -p candle-audio-mmaudio --test mmdit_conformance -- --ignored --nocapture
//! ```
//! Set `MMAUDIO_DIT_SNAPSHOT` to a `mmaudio_small_16k.pth` file (or a dir containing it under
//! `weights/` or at its root), or leave unset to resolve the pinned checkpoint via the audio lane's
//! F-029 hub path.

use candle_audio_mmaudio as m;
use candle_audio_mmaudio::candle_audio::candle_core::{Device, Tensor};
use candle_audio_mmaudio::mmdit;

fn load_net() -> mmdit::MmAudioDit {
    let dev = Device::Cpu;
    if let Ok(p) = std::env::var("MMAUDIO_DIT_SNAPSHOT") {
        let path = std::path::PathBuf::from(&p);
        return if path.is_dir() {
            mmdit::load(&m::gen_core::WeightsSource::Dir(path), &dev)
                .expect("load mmaudio small_16k from MMAUDIO_DIT_SNAPSHOT dir")
        } else {
            mmdit::load_from_pth(&path, &dev)
                .expect("load mmaudio small_16k from MMAUDIO_DIT_SNAPSHOT file")
        };
    }
    let src = mmdit::resolve_pinned_weights()
        .expect("resolve the pinned mmaudio_small_16k.pth (network or warm HF cache)");
    mmdit::load(&src, &dev).expect("load the MMAudio MM-DiT generator")
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

fn cosine(a: &[f32], b: &[f32]) -> f32 {
    let dot: f32 = a.iter().zip(b).map(|(x, y)| x * y).sum();
    let na: f32 = a.iter().map(|x| x * x).sum::<f32>().sqrt();
    let nb: f32 = b.iter().map(|x| x * x).sum::<f32>().sqrt();
    dot / (na * nb)
}

fn max_abs_diff(a: &[f32], b: &[f32]) -> f32 {
    a.iter()
        .zip(b)
        .map(|(x, y)| (x - y).abs())
        .fold(0f32, f32::max)
}

fn read_f32(path: &std::path::Path) -> Vec<f32> {
    let bytes = std::fs::read(path).unwrap_or_else(|e| panic!("read {}: {e}", path.display()));
    bytes
        .chunks_exact(4)
        .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
        .collect()
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
#[ignore = "numerical parity vs PyTorch MMAudio; set MMAUDIO_DIT_PARITY_DIR to the ref_dump.py dir"]
fn mmdit_matches_reference() {
    let dir = match std::env::var("MMAUDIO_DIT_PARITY_DIR") {
        Ok(d) => std::path::PathBuf::from(d),
        Err(_) => {
            eprintln!("MMAUDIO_DIT_PARITY_DIR unset; skipping parity");
            return;
        }
    };
    let dev = Device::Cpu;
    let net = load_net();

    let load = |name: &str, shape: &[usize]| -> Tensor {
        Tensor::from_vec(read_f32(&dir.join(format!("{name}.f32"))), shape, &dev).unwrap()
    };

    // ---- single predict_flow parity (isolates the MM-DiT from the feature encoders) ----
    let clip = load("clip_f", &[1, 64, 1024]);
    let sync = load("sync_f", &[1, 192, 768]);
    let text = load("text_f", &[1, 77, 1024]);
    let latent = load("latent", &[1, 250, 20]);
    let t = load("t", &[1]);
    let cond = net.preprocess_conditions(&clip, &sync, &text).unwrap();
    let got = vecof(&net.predict_flow(&latent, &t, &cond).unwrap());
    let want = read_f32(&dir.join("flow.f32"));
    assert_eq!(got.len(), want.len(), "flow length");
    let f_cos = cosine(&got, &want);
    let f_mad = max_abs_diff(&got, &want);
    eprintln!("PARITY predict_flow: cos={f_cos:.6} max|Δ|={f_mad:.6}");
    assert!(f_cos > 0.999, "flow cosine {f_cos} must exceed 0.999");
    assert!(f_mad < 0.05, "flow max-abs-diff {f_mad} too large");

    // ---- full CFG-4.5 / Euler-25 sample parity (uses the SAME cond, the reference x0 noise) ----
    let x0 = load("x0", &[1, 250, 20]);
    let got_s = vecof(&net.sample_default(&x0, &cond).unwrap());
    let want_s = read_f32(&dir.join("x1_unnorm.f32"));
    assert_eq!(got_s.len(), want_s.len(), "sample length");
    let s_cos = cosine(&got_s, &want_s);
    let s_mad = max_abs_diff(&got_s, &want_s);
    // scale the tolerance to the reference magnitude (unnormalized latents are O(100s)).
    let ref_absmax = want_s.iter().cloned().fold(0f32, |a, v| a.max(v.abs()));
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
