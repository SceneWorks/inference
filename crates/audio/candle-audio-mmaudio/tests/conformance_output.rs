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
//! - [`output_parity_dump`] — gated on `MMAUDIO_DUMP_DIR`; writes the fixed latent, decoded mel, and
//!   waveform as raw little-endian f32 so `scripts`/an external torch harness can compare against
//!   the MMAudio reference (cosine / max-abs-diff). Skips silently when the env var is unset.
//!
//! `#[ignore]`d and snapshot-gated like every audio family's real-weight tests:
//! ```text
//! cargo test --locked -p candle-audio-mmaudio --test conformance_output -- --ignored --nocapture
//! ```
//! Set `MMAUDIO_VAE_SNAPSHOT` / `MMAUDIO_BIGVGAN_SNAPSHOT` to the two checkpoint files (or dirs
//! containing them under `ext_weights/` or at the root), or leave unset to resolve the pinned
//! checkpoints via the audio lane's F-029 hub path (downloads ~1.1 GB into the HF cache on first
//! run).

use candle_audio_mmaudio as mm;
use candle_audio_mmaudio::candle_audio::candle_core::{Device, Tensor};
use candle_audio_mmaudio::gen_core::WeightsSource;

const LATENT_LEN: usize = 48;

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

/// Dump the fixed latent / decoded mel / waveform as raw little-endian f32 for the torch parity
/// harness. Gated on `MMAUDIO_DUMP_DIR` so it is a no-op in the normal real-weight run.
#[test]
#[ignore = "parity dump: set MMAUDIO_DUMP_DIR and run with --ignored to emit f32 bins for the torch harness"]
fn output_parity_dump() {
    let Ok(dir) = std::env::var("MMAUDIO_DUMP_DIR") else {
        eprintln!("MMAUDIO_DUMP_DIR unset — skipping parity dump");
        return;
    };
    let dec = load_decoder();
    let dev = dec.device().clone();
    let latent = fixed_latent(&dev);
    let mel = dec.decode_latent(&latent).expect("decode");
    let wav = dec.vocode(&mel).expect("vocode");

    let write = |name: &str, t: &Tensor| {
        let v = t.flatten_all().unwrap().to_vec1::<f32>().unwrap();
        let mut bytes = Vec::with_capacity(v.len() * 4);
        for x in &v {
            bytes.extend_from_slice(&x.to_le_bytes());
        }
        std::fs::write(format!("{dir}/{name}.f32"), bytes).expect("write dump");
        eprintln!("wrote {dir}/{name}.f32 ({} floats)", v.len());
    };
    write("latent", &latent);
    write("mel_rs", &mel);
    write("wav_rs", &wav);
}
