//! **Reference parity** for the native xcodec decoder (sc-19377): `get_embed` and the 16 kHz
//! `decode` against the upstream PyTorch `SoundStream` on the same 8-codebook grid.
//!
//! `scripts/reference/yue_xcodec_reference.py` encodes a synthetic (stdlib, licence-free) clip with
//! the upstream codec at 4 kb/s — exactly 8 quantizers — and commits the grid plus the reference
//! outputs (float32, CPU) as `tests/fixtures/xcodec_decode_reference.safetensors`:
//! `codes [8, 25]`, `embed [1024, 25]` (= `get_embed`), `wave [8000]` (= `decode`).
//!
//! Each comparison asserts a **magnitude** bound — `max|Δ| / max|ref|` — never a cosine alone
//! (scale-invariant: a wrong output gain would pass). The bounds sit well above the measured CPU
//! noise floor (recorded beside each constant) and well below what a real defect produces; each
//! golden is then **mutated** and the same check must fail, so the gate provably discriminates.
//!
//! ```text
//! YUE_XCODEC_SNAPSHOT=/path/to/xcodec-mini-infer \
//!   cargo test --locked -p candle-audio-yue --test xcodec_parity -- --ignored --nocapture
//! ```

use std::path::PathBuf;

use candle_audio_yue::candle_audio::candle_core::{DType, Device, Tensor};
use candle_audio_yue::codec::{XcodecDecoder, CHECKPOINT, SAMPLES_PER_FRAME};
use candle_audio_yue::tokens::CodecFrames;

const FIXTURE: &str = "xcodec_decode_reference.safetensors";

/// `get_embed` is a lookup-sum of 8 f32 rows in the reference's own order; measured max relative
/// difference 0 (bit-exact) on CPU. The bound only admits accumulation-order rounding; a 1 % error
/// on one channel measures 1.9e-3.
const EMBED_MAX_REL: f64 = 1e-6;
/// The 16 kHz waveform through `fc_post2` + the ~60-conv DAC decoder; measured max relative
/// difference 1.6e-6 on CPU/f32 (the noise floor), so ~6× headroom — while a 0.01 % output-gain
/// error (1e-4) or a one-sample shift (1.7e-1) lands outside it.
const WAVE_MAX_REL: f64 = 1e-5;

fn fixture() -> std::collections::HashMap<String, Tensor> {
    let path = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("tests")
        .join("fixtures")
        .join(FIXTURE);
    candle_audio_yue::candle_audio::candle_core::safetensors::load(&path, &Device::Cpu)
        .unwrap_or_else(|e| panic!("load {}: {e}", path.display()))
}

fn flat(t: &Tensor) -> Vec<f32> {
    t.to_dtype(DType::F32)
        .unwrap()
        .flatten_all()
        .unwrap()
        .to_vec1()
        .unwrap()
}

/// `max|got − want| / max|want|`.
fn max_rel(got: &[f32], want: &[f32]) -> f64 {
    assert_eq!(got.len(), want.len(), "length mismatch");
    let peak = want.iter().fold(0f64, |m, &v| m.max(v.abs() as f64));
    let diff = got
        .iter()
        .zip(want)
        .fold(0f64, |m, (&a, &b)| m.max((a as f64 - b as f64).abs()));
    diff / peak
}

#[test]
#[ignore = "real weights: set YUE_XCODEC_SNAPSHOT to the staged xcodec-mini-infer snapshot; run with --ignored"]
fn xcodec_get_embed_and_decode_match_the_reference() {
    let snapshot = PathBuf::from(
        std::env::var("YUE_XCODEC_SNAPSHOT")
            .expect("set YUE_XCODEC_SNAPSHOT to the xcodec-mini-infer snapshot dir"),
    );
    // CPU, not `default_device()`: the reference is torch-CPU f32, and a GPU-vs-CPU rounding gate
    // would be a red nobody can reproduce. The Metal/CUDA lanes compile the same code path.
    let device = Device::Cpu;
    let fx = fixture();
    let codes: Vec<Vec<u32>> = fx["codes"]
        .to_dtype(DType::I64)
        .unwrap()
        .to_vec2::<i64>()
        .unwrap()
        .into_iter()
        .map(|row| row.into_iter().map(|c| c as u32).collect())
        .collect();
    let frames = CodecFrames { codebooks: codes };
    let ref_embed = flat(&fx["embed"]);
    let ref_wave = flat(&fx["wave"]);
    assert_eq!(ref_wave.len(), frames.frames() * SAMPLES_PER_FRAME);

    let dec = XcodecDecoder::load(&snapshot.join(CHECKPOINT), &device).expect("load xcodec");

    // AC 2 — get_embed.
    let embed = dec.get_embed(&frames).expect("get_embed");
    assert_eq!(embed.dims(), [1, 1024, frames.frames()]);
    let embed = flat(&embed);
    let embed_rel = max_rel(&embed, &ref_embed);
    println!("get_embed: max|Δ|/max|ref| = {embed_rel:.3e} (bound {EMBED_MAX_REL:.0e})");
    assert!(
        embed_rel <= EMBED_MAX_REL,
        "get_embed diverges: {embed_rel:.3e}"
    );

    // AC 1 — the 16 kHz waveform.
    let embed_t = dec.get_embed(&frames).unwrap();
    let wave = flat(&dec.decode_embedding(&embed_t, &|| false).unwrap().unwrap());
    let wave_rel = max_rel(&wave, &ref_wave);
    println!("decode: max|Δ|/max|ref| = {wave_rel:.3e} (bound {WAVE_MAX_REL:.0e})");
    assert!(wave_rel <= WAVE_MAX_REL, "decode diverges: {wave_rel:.3e}");

    // Mutated goldens must fail the same checks: a 1 % gain error on one embedding channel, and
    // the reference waveform shifted by a single sample.
    let mut bad_embed = ref_embed.clone();
    let t = frames.frames();
    for v in &mut bad_embed[7 * t..8 * t] {
        *v *= 1.01;
    }
    let bad_embed_rel = max_rel(&embed, &bad_embed);
    assert!(
        bad_embed_rel > EMBED_MAX_REL,
        "a mutated embedding golden passed ({bad_embed_rel:.3e})"
    );
    let mut bad_wave = ref_wave.clone();
    bad_wave.rotate_right(1);
    let bad_wave_rel = max_rel(&wave, &bad_wave);
    assert!(
        bad_wave_rel > WAVE_MAX_REL,
        "a mutated waveform golden passed ({bad_wave_rel:.3e})"
    );
    let mut scaled = ref_wave.clone();
    scaled.iter_mut().for_each(|v| *v *= 1.0001);
    let scaled_rel = max_rel(&wave, &scaled);
    assert!(
        scaled_rel > WAVE_MAX_REL,
        "a 0.01 % gain-mutated waveform golden passed ({scaled_rel:.3e})"
    );
    println!(
        "mutations rejected: embed×1.01 {bad_embed_rel:.3e}, wave shift {bad_wave_rel:.3e}, \
         wave×1.0001 {scaled_rel:.3e}"
    );
}
