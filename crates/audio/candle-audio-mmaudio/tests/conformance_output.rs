//! Real-weight conformance for the candle MMAudio **16k output path** (sc-13440): the mel-VAE
//! decoder + BigVGAN vocoder.
//!
//! ## What this gates on real weights
//!
//! Loads the pinned `hkchengrex/MMAudio` `ext_weights/v1-16.pth` (~687 MB mel-VAE) and
//! `ext_weights/best_netG.pt` (~449 MB BigVGAN), builds the two-stage decoder, and drives a fixed
//! synthetic latent through `latent (1,20,L) → mel (1,80,2L) → waveform (1,1,512L)`:
//!
//! - [`output_latent_to_waveform_finite_deterministic`] — the mel and waveform have the exact
//!   expected shapes, every value is finite, the waveform is in `[-1, 1]` (BigVGAN's final `tanh`),
//!   the output is non-degenerate (plausible energy, not a constant), and it is **byte-identical
//!   run-to-run** (deterministic). A broken weight mapping (wrong key, un-removed weight-norm,
//!   transposed conv, mis-ordered AMP block) would surface here as a load error, a NaN, a shape
//!   mismatch, or a clipped/silent waveform.
//!
//! - [`output_16k_matches_reference`] — **numerical parity** against the PyTorch MMAudio reference
//!   over the same two stages: `decode_latent` vs the reference mel, `vocode(ref_mel)` vs the
//!   reference waveform, and the assembled `latent_to_waveform` end to end. Its fixture is produced
//!   by `scripts/reference/mmaudio_reference.py` and **committed** (sc-17285).
//!
//!   This replaces `output_parity_dump`, which asserted nothing at all: it *wrote* the candle side
//!   out as raw f32 when `MMAUDIO_DUMP_DIR` was set — for an external torch harness that existed
//!   nowhere in this repository — and `return`ed early to a *passing* result when it was not. A
//!   run-count assertion cannot tell that apart from real work, so sc-17266 had to exclude it by
//!   name from the real-weight lane. Comparing against a committed reference is what that dump was
//!   always a placeholder for, and it has no unset case.
//!
//! `#[ignore]`d and snapshot-gated like every audio family's real-weight tests:
//! ```text
//! cargo test --locked -p candle-audio-mmaudio --test conformance_output -- --ignored --nocapture
//! ```
//! Set `MMAUDIO_VAE_SNAPSHOT` / `MMAUDIO_BIGVGAN_SNAPSHOT` to the two checkpoint files (or dirs
//! containing them under `ext_weights/` or at the root). Both are **required**: `resolve_source`
//! panics when either is unset, because inference never self-fetches and never derives a hub-cache
//! location (epic 13657). There is no hub fallback.

mod common;

use candle_audio_mmaudio as mm;
use candle_audio_mmaudio::candle_audio::candle_core::{Device, Tensor};
use candle_audio_mmaudio::gen_core::WeightsSource;

const LATENT_LEN: usize = 48;
const FIXTURE: &str = "mmaudio_parity_output_16k.safetensors";

/// Deterministic closed-form latent `(1, 20, L)` — computed identically in the torch parity harness
/// so both sides decode the *same* input without transferring a file.
fn fixed_latent(dev: &Device) -> Tensor {
    let c = mm::vae::EMBED_DIM;
    let l = LATENT_LEN;
    let mut data = vec![0f32; c * l];
    for ci in 0..c {
        for li in 0..l {
            let v = 0.3f64 * (0.11 * ci as f64 + 0.023 * li as f64).sin()
                + 0.2f64 * (0.007 * li as f64 - 0.05 * ci as f64).cos();
            data[ci * l + li] = v as f32;
        }
    }
    Tensor::from_vec(data, (1, c, l), dev).expect("latent tensor")
}

fn resolve_source(env: &str, file: &str, nested: &str) -> WeightsSource {
    // Required env path — inference never self-fetches or derives a cache location (epic 13657).
    let _ = (file, nested);
    let p = std::env::var(env)
        .unwrap_or_else(|_| panic!("set {env} to the {file} weights file or its snapshot dir"));
    let path = std::path::PathBuf::from(&p);
    if path.is_dir() {
        WeightsSource::Dir(path)
    } else {
        WeightsSource::File(path)
    }
}

fn load_decoder() -> mm::AudioDecoder16k {
    let dev = Device::Cpu;
    let vae = resolve_source(
        "MMAUDIO_VAE_SNAPSHOT",
        "v1-16.pth",
        mm::output::VAE_WEIGHTS_PATH,
    );
    let bigvgan = resolve_source(
        "MMAUDIO_BIGVGAN_SNAPSHOT",
        "best_netG.pt",
        mm::output::BIGVGAN_WEIGHTS_PATH,
    );
    mm::AudioDecoder16k::load(&vae, &bigvgan, &dev).expect("load MMAudio 16k output decoder")
}

fn stats(v: &[f32]) -> (f32, f32, f32, f32) {
    let mean = v.iter().sum::<f32>() / v.len() as f32;
    let var = v.iter().map(|x| (x - mean).powi(2)).sum::<f32>() / v.len() as f32;
    let min = v.iter().cloned().fold(f32::INFINITY, f32::min);
    let max = v.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
    (mean, var, min, max)
}

#[test]
#[ignore = "downloads ~1.1GB (v1-16.pth + best_netG.pt); run explicitly with --ignored"]
fn output_latent_to_waveform_finite_deterministic() {
    let dec = load_decoder();
    let dev = dec.device().clone();
    let latent = fixed_latent(&dev);

    let mel = dec.decode_latent(&latent).expect("decode latent -> mel");
    assert_eq!(
        mel.dims(),
        &[1, mm::vae::DATA_DIM, 2 * LATENT_LEN],
        "mel must be (1, 80, 2L)"
    );
    let mel_v = mel.flatten_all().unwrap().to_vec1::<f32>().unwrap();
    assert!(mel_v.iter().all(|x| x.is_finite()), "mel finite");

    let wav = dec.vocode(&mel).expect("vocode mel -> waveform");
    assert_eq!(
        wav.dims(),
        &[1, 1, mm::bigvgan::HOP * 2 * LATENT_LEN],
        "waveform must be (1, 1, 256*mel_len = 512*L)"
    );
    let wav_v = wav.flatten_all().unwrap().to_vec1::<f32>().unwrap();
    assert!(wav_v.iter().all(|x| x.is_finite()), "waveform finite");
    assert!(
        wav_v.iter().all(|x| (-1.0001..=1.0001).contains(x)),
        "waveform in [-1,1] (final tanh)"
    );

    // Non-degenerate: real signal energy, not a constant / silence.
    let (m_mean, m_var, m_min, m_max) = stats(&mel_v);
    let (w_mean, w_var, w_min, w_max) = stats(&wav_v);
    assert!(m_var > 1e-6, "mel must not be constant (var={m_var})");
    assert!(w_var > 1e-8, "waveform must carry energy (var={w_var})");
    assert!(
        w_max - w_min > 1e-3,
        "waveform must not be silent (range={})",
        w_max - w_min
    );

    // Determinism: full path re-run is byte-identical.
    let wav2 = dec
        .latent_to_waveform(&latent)
        .expect("full path")
        .flatten_all()
        .unwrap()
        .to_vec1::<f32>()
        .unwrap();
    assert_eq!(wav_v, wav2, "decoder must be deterministic run-to-run");

    eprintln!(
        "mmaudio-16k output real-weights: mel=(1,80,{}) mean={m_mean:.4} var={m_var:.4} min={m_min:.3} max={m_max:.3}",
        2 * LATENT_LEN
    );
    eprintln!(
        "mmaudio-16k output real-weights: wav=(1,1,{}) mean={w_mean:.5} var={w_var:.6} min={w_min:.4} max={w_max:.4} rms={:.5}",
        mm::bigvgan::HOP * 2 * LATENT_LEN,
        w_var.sqrt()
    );
}

/// Numerical parity for the 16k output path against the committed PyTorch reference (sc-17285).
#[test]
#[ignore = "real weights: needs v1-16.pth + best_netG.pt; run explicitly with --ignored"]
fn output_16k_matches_reference() {
    let dec = load_decoder();
    let dev = dec.device().clone();
    let tensors = common::load_fixture(FIXTURE, &dev);
    let get = |name: &str| common::fixture_tensor(&tensors, FIXTURE, name);

    // The driving latent is a closed form both languages compute independently
    // (`fixed_latent` here, `fixed_latent_16k()` in the producer). Assert they still agree before
    // comparing anything downstream: a cross-language constant that drifts silently would turn
    // this into two implementations decoding *different* inputs and scoring a low cosine for a
    // reason that has nothing to do with the port.
    let latent = fixed_latent(&dev);
    let latent_v = common::flat(&latent);
    let ref_latent_v = common::flat(&get("latent"));
    assert_eq!(
        latent_v, ref_latent_v,
        "the closed-form latent in this test and in scripts/reference/mmaudio_reference.py have \
         diverged — they must stay bit-identical"
    );

    let ref_mel = get("ref_mel");
    let ref_mel_v = common::flat(&ref_mel);
    let ref_wave = common::flat(&get("ref_wave"));

    // Stage 1 — the v1-16 mel-VAE.
    let mel = dec.decode_latent(&latent).expect("decode latent -> mel");
    assert_eq!(
        mel.dims(),
        ref_mel.dims(),
        "candle mel shape differs from reference"
    );
    let mel_v = common::flat(&mel);
    let mel_cos = common::cosine(&mel_v, &ref_mel_v);
    let mel_mad = common::max_abs_diff(&mel_v, &ref_mel_v);
    eprintln!("VAE (v1-16) mel PARITY:  cosine={mel_cos:.6}  max_abs_diff={mel_mad:.6}");

    // Stage 2 — BigVGAN on the reference's own mel, isolating the vocoder from the VAE.
    let wave_from_ref_mel = common::flat(&dec.vocode(&ref_mel).expect("vocode(ref_mel)"));
    let voc_cos = common::cosine(&wave_from_ref_mel, &ref_wave);
    let voc_mad = common::max_abs_diff(&wave_from_ref_mel, &ref_wave);
    eprintln!("BigVGAN 16k wave PARITY: cosine={voc_cos:.6}  max_abs_diff={voc_mad:.6}");

    // Stage 3 — the assembled decoder end to end.
    let wave = common::flat(&dec.latent_to_waveform(&latent).expect("latent_to_waveform"));
    let e2e_cos = common::cosine(&wave, &ref_wave);
    let e2e_mad = common::max_abs_diff(&wave, &ref_wave);
    eprintln!("assembled decoder wave:  cosine={e2e_cos:.6}  max_abs_diff={e2e_mad:.6}");

    assert_eq!(
        wave.len(),
        ref_wave.len(),
        "waveform length differs from reference"
    );
    assert!(
        mel_cos > 0.999,
        "v1-16 mel-VAE decode cosine {mel_cos:.6} vs reference is below 0.999"
    );
    assert!(
        voc_cos > 0.999,
        "16k BigVGAN vocode cosine {voc_cos:.6} vs reference is below 0.999"
    );
    assert!(
        e2e_cos > 0.999,
        "assembled 16k decoder waveform cosine {e2e_cos:.6} vs reference is below 0.999"
    );
}
