//! **Reference-parity** gate for the MMAudio **44.1 kHz output path** (sc-13504) — the v1-44 mel-VAE
//! decode and the NVIDIA **BigVGAN v2** 44 kHz vocoder. These two were the last MMAudio components
//! that had real-weights conformance but had NOT been numerically compared against PyTorch (the
//! large_44k_v2 MM-DiT and the shared CLIP/Synchformer encoders were each parity-verified in their
//! own slices; the assembled 44k *waveform* is gated E2E by `parity_reference_44k`). This closes that
//! one gap directly, stage by stage.
//!
//! `scratchpad/mmaudio_44k_decoder_dump.py` runs the reference `AutoEncoderModule(mode='44k')` in f32
//! on a **fixed deterministic latent** `z (1, 40, 345)` and dumps, as safetensors: `z`, the reference
//! 128-band log-mel `ref_mel (1, 128, 690)`, and the reference 44.1 kHz waveform `ref_wave`. Feeding
//! the same `z` into the candle [`AudioDecoder44k`] isolates:
//!   - `decode_latent(z)` vs `ref_mel`      → proves the v1-44 mel-VAE port,
//!   - `vocode(ref_mel)`  vs `ref_wave`     → proves the NVIDIA BigVGAN v2 port (fixed mel input),
//!   - `latent_to_waveform(z)` vs `ref_wave`→ proves the assembled decoder end to end.
//!
//! ```text
//! MMAUDIO_PARITY_DUMP_44K_DECODER=/path/to/mmaudio_44k_decoder_dump.safetensors \
//!   cargo test --locked -p candle-audio-mmaudio --test parity_reference_44k_decoder -- --ignored --nocapture
//! ```

use candle_audio_mmaudio::candle_audio;
use candle_audio_mmaudio::candle_audio::candle_core::{safetensors, Tensor};
use candle_audio_mmaudio::output::{resolve_pinned_bigvgan_v2, resolve_pinned_vae_44k};
use candle_audio_mmaudio::AudioDecoder44k;

fn cosine(a: &[f32], b: &[f32]) -> f64 {
    let n = a.len().min(b.len());
    let (mut dot, mut na, mut nb) = (0f64, 0f64, 0f64);
    for i in 0..n {
        let (x, y) = (a[i] as f64, b[i] as f64);
        dot += x * y;
        na += x * x;
        nb += y * y;
    }
    dot / (na.sqrt() * nb.sqrt())
}

fn max_abs_diff(a: &[f32], b: &[f32]) -> f64 {
    let n = a.len().min(b.len());
    (0..n)
        .map(|i| (a[i] - b[i]).abs() as f64)
        .fold(0.0, f64::max)
}

fn flat(t: &Tensor) -> Vec<f32> {
    t.flatten_all().unwrap().to_vec1().unwrap()
}

#[test]
#[ignore = "real weights + a reference dump: set MMAUDIO_PARITY_DUMP_44K_DECODER (see \
            scratchpad/mmaudio_44k_decoder_dump.py); run with --ignored"]
fn decoder_44k_matches_reference_mel_and_waveform() {
    let dump = std::env::var("MMAUDIO_PARITY_DUMP_44K_DECODER")
        .expect("set MMAUDIO_PARITY_DUMP_44K_DECODER to the reference safetensors dump");
    let device = candle_audio::default_device().expect("device");
    let t = safetensors::load(&dump, &device).expect("load reference dump");
    let get = |k: &str| -> Tensor {
        t.get(k)
            .unwrap_or_else(|| panic!("dump missing {k}"))
            .clone()
    };
    let z = get("z"); // (1, 40, 345) — the fixed VAE-input latent
    let ref_mel = get("ref_mel"); // (1, 128, 690)
    let ref_mel_v = flat(&ref_mel);
    let ref_wave: Vec<f32> = flat(&get("ref_wave"));
    println!(
        "dump: z{:?} ref_mel{:?} ref_wave={}",
        z.dims(),
        ref_mel.dims(),
        ref_wave.len()
    );

    // Load the real pinned 44k decoder: MMAudio v1-44 mel-VAE + NVIDIA BigVGAN v2.
    let vae_src = resolve_pinned_vae_44k().expect("resolve pinned v1-44 VAE");
    let bigvgan_src = resolve_pinned_bigvgan_v2().expect("resolve pinned NVIDIA BigVGAN v2");
    let decoder = AudioDecoder44k::load(&vae_src, &bigvgan_src, &device).expect("load 44k decoder");

    // Stage 1 — the v1-44 mel-VAE: decode_latent(z) vs ref_mel.
    let mel = decoder.decode_latent(&z).expect("candle 44k VAE decode");
    assert_eq!(
        mel.dims(),
        ref_mel.dims(),
        "candle mel shape differs from reference"
    );
    let mel_v = flat(&mel);
    let mel_cos = cosine(&mel_v, &ref_mel_v);
    let mel_mad = max_abs_diff(&mel_v, &ref_mel_v);
    println!("VAE (v1-44) mel PARITY:      cosine={mel_cos:.6}  max_abs_diff={mel_mad:.6}");

    // Stage 2 — the NVIDIA BigVGAN v2 vocoder on the reference's own mel: vocode(ref_mel) vs ref_wave.
    let wave_from_ref_mel = decoder
        .vocode(&ref_mel)
        .expect("candle BigVGAN v2 vocode(ref_mel)");
    let wrm_v = flat(&wave_from_ref_mel);
    let voc_cos = cosine(&wrm_v, &ref_wave);
    let voc_mad = max_abs_diff(&wrm_v, &ref_wave);
    println!("BigVGAN v2 wave PARITY:      cosine={voc_cos:.6}  max_abs_diff={voc_mad:.6}");

    // Stage 3 — the assembled decoder end to end: latent_to_waveform(z) vs ref_wave.
    let wave = decoder
        .latent_to_waveform(&z)
        .expect("candle 44k latent_to_waveform");
    let wave_v = flat(&wave);
    let e2e_cos = cosine(&wave_v, &ref_wave);
    let e2e_mad = max_abs_diff(&wave_v, &ref_wave);
    println!("assembled decoder wave:      cosine={e2e_cos:.6}  max_abs_diff={e2e_mad:.6}");
    println!(
        "candle wave: {} samples; reference: {} samples",
        wave_v.len(),
        ref_wave.len()
    );

    assert!(
        (wave_v.len() as i64 - ref_wave.len() as i64).abs() <= 1024,
        "waveform length {} differs from reference {} by more than a vocoder frame",
        wave_v.len(),
        ref_wave.len()
    );
    assert!(
        mel_cos > 0.999,
        "v1-44 mel-VAE decode cosine {mel_cos:.6} vs reference is below 0.999"
    );
    assert!(
        voc_cos > 0.999,
        "NVIDIA BigVGAN v2 vocode cosine {voc_cos:.6} vs reference is below 0.999"
    );
    assert!(
        e2e_cos > 0.999,
        "assembled 44k decoder waveform cosine {e2e_cos:.6} vs reference is below 0.999"
    );
}
