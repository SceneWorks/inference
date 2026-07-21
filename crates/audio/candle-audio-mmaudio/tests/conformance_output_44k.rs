//! Real-weight conformance for the candle MMAudio **44.1 kHz output path** (sc-13441): the 44k
//! mel-VAE decoder + the **NVIDIA BigVGAN v2** vocoder.
//!
//! ## What this gates on real weights
//!
//! Loads the pinned `hkchengrex/MMAudio` `ext_weights/v1-44.pth` (~1.22 GB, 40-d-latent / 128-band
//! mel-VAE) and the **separate** `nvidia/bigvgan_v2_44khz_128band_512x` `bigvgan_generator.pt`
//! (~489 MB), builds the two-stage 44k decoder, and drives a fixed synthetic latent through
//! `latent (1,40,L) → mel (1,128,2L) → waveform (1,1,1024L)`:
//!
//! - [`output_44k_latent_to_waveform_finite_deterministic`] — the mel and waveform have the exact
//!   expected shapes (128-band mel, 512× vocoder upsample), every value is finite, the waveform is
//!   in `[-1, 1]` (BigVGAN v2's `use_tanh_at_final=false` → `clamp`), non-degenerate, and
//!   byte-identical run-to-run. A broken weight mapping (wrong key layout, un-removed weight-norm,
//!   the missing conv_post bias, the tanh-vs-clamp final) surfaces here as a load error, NaN, shape
//!   mismatch, or a silent/clipped waveform. This is the strongest **torch-free** real-weights
//!   evidence for the newly ported 44k output stage.
//!
//! `#[ignore]`d and snapshot-gated like every audio family's real-weight tests:
//! ```text
//! cargo test --locked -p candle-audio-mmaudio --test conformance_output_44k -- --ignored --nocapture
//! ```
//! Set `MMAUDIO_VAE_44K_SNAPSHOT` / `MMAUDIO_BIGVGAN_V2_SNAPSHOT` to the two checkpoint files (or
//! dirs containing them), or leave unset to resolve the pinned checkpoints via the audio lane's F-029
//! hub path (downloads ~1.7 GB into the HF cache on first run).

use candle_audio_mmaudio as mm;
use candle_audio_mmaudio::candle_audio::candle_core::{Device, Tensor};
use candle_audio_mmaudio::gen_core::WeightsSource;

const LATENT_LEN: usize = 48;

/// Deterministic closed-form latent `(1, 40, L)` — computed identically in a torch parity harness so
/// both sides decode the *same* input without transferring a file.
fn fixed_latent(dev: &Device) -> Tensor {
    let c = mm::vae::EMBED_DIM_44K;
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

fn resolve_source(env: &str) -> WeightsSource {
    // Required env path — inference never self-fetches or derives a cache location (epic 13657).
    let p = std::env::var(env)
        .unwrap_or_else(|_| panic!("set {env} to the weights file or its snapshot dir"));
    let path = std::path::PathBuf::from(&p);
    if path.is_dir() {
        WeightsSource::Dir(path)
    } else {
        WeightsSource::File(path)
    }
}

fn load_decoder() -> mm::AudioDecoder44k {
    let dev = Device::Cpu;
    let vae = resolve_source("MMAUDIO_VAE_44K_SNAPSHOT");
    let bigvgan = resolve_source("MMAUDIO_BIGVGAN_V2_SNAPSHOT");
    mm::AudioDecoder44k::load(&vae, &bigvgan, &dev).expect("load MMAudio 44k output decoder")
}

fn stats(v: &[f32]) -> (f32, f32, f32, f32) {
    let mean = v.iter().sum::<f32>() / v.len() as f32;
    let var = v.iter().map(|x| (x - mean).powi(2)).sum::<f32>() / v.len() as f32;
    let min = v.iter().cloned().fold(f32::INFINITY, f32::min);
    let max = v.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
    (mean, var, min, max)
}

#[test]
#[ignore = "downloads ~1.7GB (v1-44.pth + nvidia bigvgan_generator.pt); run explicitly with --ignored"]
fn output_44k_latent_to_waveform_finite_deterministic() {
    let dec = load_decoder();
    let dev = dec.device().clone();
    let latent = fixed_latent(&dev);
    let hop = mm::bigvgan::Config::bigvgan_v2_44khz_128band_512x().hop();

    let mel = dec.decode_latent(&latent).expect("decode latent -> mel");
    assert_eq!(
        mel.dims(),
        &[1, mm::vae::DATA_DIM_44K, 2 * LATENT_LEN],
        "mel must be (1, 128, 2L)"
    );
    let mel_v = mel.flatten_all().unwrap().to_vec1::<f32>().unwrap();
    assert!(mel_v.iter().all(|x| x.is_finite()), "mel finite");

    let wav = dec.vocode(&mel).expect("vocode mel -> waveform");
    assert_eq!(
        wav.dims(),
        &[1, 1, hop * 2 * LATENT_LEN],
        "waveform must be (1, 1, 512*mel_len = 1024*L)"
    );
    let wav_v = wav.flatten_all().unwrap().to_vec1::<f32>().unwrap();
    assert!(wav_v.iter().all(|x| x.is_finite()), "waveform finite");
    assert!(
        wav_v.iter().all(|x| (-1.0001..=1.0001).contains(x)),
        "waveform in [-1,1] (final clamp, use_tanh_at_final=false)"
    );

    let (m_mean, m_var, m_min, m_max) = stats(&mel_v);
    let (w_mean, w_var, w_min, w_max) = stats(&wav_v);
    assert!(m_var > 1e-6, "mel must not be constant (var={m_var})");
    assert!(w_var > 1e-8, "waveform must carry energy (var={w_var})");
    assert!(
        w_max - w_min > 1e-3,
        "waveform must not be silent (range={})",
        w_max - w_min
    );

    let wav2 = dec
        .latent_to_waveform(&latent)
        .expect("full path")
        .flatten_all()
        .unwrap()
        .to_vec1::<f32>()
        .unwrap();
    assert_eq!(wav_v, wav2, "decoder must be deterministic run-to-run");

    eprintln!(
        "mmaudio-44k output real-weights: mel=(1,128,{}) mean={m_mean:.4} var={m_var:.4} min={m_min:.3} max={m_max:.3}",
        2 * LATENT_LEN
    );
    eprintln!(
        "mmaudio-44k output real-weights: wav=(1,1,{}) mean={w_mean:.5} var={w_var:.6} min={w_min:.4} max={w_max:.4} rms={:.5}",
        hop * 2 * LATENT_LEN,
        w_var.sqrt()
    );
}
