//! **Reference-parity** gate for the MMAudio **44.1 kHz output path** (sc-13504, sc-17285) — the
//! v1-44 mel-VAE decode and the NVIDIA **BigVGAN v2** 44 kHz vocoder. These two were the last
//! MMAudio components that had real-weights conformance but had NOT been numerically compared
//! against PyTorch (the large_44k_v2 MM-DiT and the shared CLIP/Synchformer encoders were each
//! parity-verified in their own slices; the assembled 44k *waveform* is gated E2E by
//! `parity_reference_44k`). This closes that one gap directly, stage by stage.
//!
//! `scripts/reference/mmaudio_reference.py` runs the vendored reference `AutoEncoderModule(mode=
//! '44k')` in f32 on the CPU and commits, as safetensors: the latent `z (1, 40, 345)`, the
//! reference 128-band log-mel `ref_mel (1, 128, 690)`, and the reference 44.1 kHz waveform
//! `ref_wave`. Feeding the same `z` into the candle [`AudioDecoder44k`] isolates:
//!   - `decode_latent(z)` vs `ref_mel`      → proves the v1-44 mel-VAE port,
//!   - `vocode(ref_mel)`  vs `ref_wave`     → proves the NVIDIA BigVGAN v2 port (fixed mel input),
//!   - `latent_to_waveform(z)` vs `ref_wave`→ proves the assembled decoder end to end.
//!
//! `z` is the **reference pipeline's own** un-normalized latent, not a synthetic one. That is
//! deliberate: an arbitrary closed-form latent drives the 44.1 kHz VAE far out of distribution and
//! the vocoder's final `tanh` saturates, which compresses exactly the differences this gate exists
//! to detect — a saturated waveform scores a high cosine even against a materially wrong port.
//!
//! The fixture is **committed** (sc-17285). Before that this test read its dump path from
//! `MMAUDIO_PARITY_DUMP_44K_DECODER`, which no workflow set and no committed script produced, so
//! this gate had never run anywhere.
//!
//! ```text
//! cargo test --locked -p candle-audio-mmaudio --test parity_reference_44k_decoder -- --ignored --nocapture
//! ```

mod common;

use candle_audio_mmaudio::candle_audio;
use candle_audio_mmaudio::candle_audio::gen_core::WeightsSource;
use candle_audio_mmaudio::AudioDecoder44k;

const FIXTURE: &str = "mmaudio_parity_reference_44k_decoder.safetensors";

/// Resolve a component from the required `env` path — inference never self-fetches or derives a
/// cache location (epic 13657).
fn env_source(env: &str) -> WeightsSource {
    let p = std::env::var(env)
        .unwrap_or_else(|_| panic!("set {env} to the weights file or snapshot dir"));
    let path = std::path::PathBuf::from(&p);
    if path.is_dir() {
        WeightsSource::Dir(path)
    } else {
        WeightsSource::File(path)
    }
}

#[test]
#[ignore = "real weights: needs the v1-44 mel-VAE and NVIDIA BigVGAN v2 snapshots; run with --ignored"]
fn decoder_44k_matches_reference_mel_and_waveform() {
    let device = candle_audio::default_device().expect("device");
    let tensors = common::load_fixture(FIXTURE, &device);
    let get = |k: &str| common::fixture_tensor(&tensors, FIXTURE, k);
    let z = get("z"); // (1, 40, 345) — the VAE-input latent
    let ref_mel = get("ref_mel"); // (1, 128, 690)
    let ref_mel_v = common::flat(&ref_mel);
    let ref_wave = common::flat(&get("ref_wave"));
    println!(
        "fixture: z{:?} ref_mel{:?} ref_wave={}",
        z.dims(),
        ref_mel.dims(),
        ref_wave.len()
    );

    // Load the real 44k decoder from env-pointed paths: MMAudio v1-44 mel-VAE + NVIDIA BigVGAN v2.
    let vae_src = env_source("MMAUDIO_VAE_44K_SNAPSHOT");
    let bigvgan_src = env_source("MMAUDIO_BIGVGAN_V2_SNAPSHOT");
    let decoder = AudioDecoder44k::load(&vae_src, &bigvgan_src, &device).expect("load 44k decoder");

    // Stage 1 — the v1-44 mel-VAE: decode_latent(z) vs ref_mel.
    let mel = decoder.decode_latent(&z).expect("candle 44k VAE decode");
    assert_eq!(
        mel.dims(),
        ref_mel.dims(),
        "candle mel shape differs from reference"
    );
    let mel_v = common::flat(&mel);
    let mel_cos = common::cosine(&mel_v, &ref_mel_v);
    let mel_mad = common::max_abs_diff(&mel_v, &ref_mel_v);
    println!("VAE (v1-44) mel PARITY:      cosine={mel_cos:.6}  max_abs_diff={mel_mad:.6}");

    // Stage 2 — the NVIDIA BigVGAN v2 vocoder on the reference's own mel: vocode(ref_mel) vs ref_wave.
    let wave_from_ref_mel = decoder
        .vocode(&ref_mel)
        .expect("candle BigVGAN v2 vocode(ref_mel)");
    let wrm_v = common::flat(&wave_from_ref_mel);
    let voc_cos = common::cosine(&wrm_v, &ref_wave);
    let voc_mad = common::max_abs_diff(&wrm_v, &ref_wave);
    println!("BigVGAN v2 wave PARITY:      cosine={voc_cos:.6}  max_abs_diff={voc_mad:.6}");

    // Stage 3 — the assembled decoder end to end: latent_to_waveform(z) vs ref_wave.
    let wave = decoder
        .latent_to_waveform(&z)
        .expect("candle 44k latent_to_waveform");
    let wave_v = common::flat(&wave);
    let e2e_cos = common::cosine(&wave_v, &ref_wave);
    let e2e_mad = common::max_abs_diff(&wave_v, &ref_wave);
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
