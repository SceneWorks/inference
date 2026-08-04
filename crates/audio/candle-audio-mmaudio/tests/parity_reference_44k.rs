//! End-to-end **reference-parity** gate for the shipping MMAudio **44.1 kHz** assembly
//! (sc-13441, sc-17285) — the 44k twin of `parity_reference.rs`.
//!
//! `scripts/reference/mmaudio_reference.py` runs the vendored reference `MMAudio` full pipeline
//! with the `large_44k_v2` variant in f32 on the CPU over a fixed synthetic clip + prompt + seed
//! and commits, as safetensors: the encoded conditioning features
//! (`clip_f`/`sync_f`/`text_f`/`neg_text_f`), the seeded prior `x0` (shape `(1, latent, 40)`), and
//! the reference 44.1 kHz waveform (`ref_wave`). Injecting the reference's own features + prior
//! isolates this crate's 44k assembly (large_44k_v2 MM-DiT + 44k VAE + NVIDIA BigVGAN v2) from the
//! (separately parity-verified) shared encoders AND from torch-vs-Rust RNG, so a high waveform
//! cosine proves the assembled candle 44k pipeline is faithful end to end.
//!
//! The fixture is **committed** (sc-17285). Before that this test read its dump path from
//! `MMAUDIO_PARITY_DUMP_44K`, which no workflow set and no committed script produced, so this gate
//! had never run anywhere.
//!
//! ```text
//! cargo test --locked -p candle-audio-mmaudio --test parity_reference_44k -- --ignored --nocapture
//! ```

mod common;

use candle_audio_mmaudio::candle_audio;
use candle_audio_mmaudio::MmAudio44kPipeline;

const FIXTURE: &str = "mmaudio_parity_reference_44k.safetensors";

#[test]
#[ignore = "real weights: needs the five MMAudio 44k component snapshots; run with --ignored"]
fn assembly_44k_matches_reference_waveform() {
    let device = candle_audio::default_device().expect("device");
    let tensors = common::load_fixture(FIXTURE, &device);
    let get = |k: &str| common::fixture_tensor(&tensors, FIXTURE, k);
    let clip_f = get("clip_f");
    let sync_f = get("sync_f");
    let text_f = get("text_f");
    let neg_f = get("neg_text_f");
    let x0 = get("x0");
    let ref_wave = common::flat(&get("ref_wave"));
    let scalars: Vec<f32> = get("scalars").to_vec1().unwrap(); // [cfg, steps, duration, src_fps]
    let cfg = scalars[0] as f64;
    let steps = scalars[1] as usize;
    println!(
        "fixture: clip_f{:?} sync_f{:?} x0{:?} cfg={cfg} steps={steps} ref_wave={}",
        clip_f.dims(),
        sync_f.dims(),
        x0.dims(),
        ref_wave.len()
    );

    let pipeline = MmAudio44kPipeline::from_components(
        &common::clip_source(),
        &common::synchformer_source(),
        &common::dit_44k_source(),
        &common::vae_44k_source(),
        &common::vocoder_44k_source(),
        &device,
    )
    .expect("load 44k pipeline");

    let wave = pipeline
        .synthesize_from_features(
            &clip_f,
            &sync_f,
            &text_f,
            &neg_f,
            &x0,
            cfg,
            steps,
            &mut |_| {},
            &|| false,
        )
        .expect("candle 44k assembly synthesize_from_features");

    let cos = common::cosine(&wave, &ref_wave);
    let mad = common::max_abs_diff(&wave, &ref_wave);
    println!(
        "candle 44k wave: {} samples; reference: {} samples",
        wave.len(),
        ref_wave.len()
    );
    println!("E2E PARITY (44k): cosine={cos:.6}  max_abs_diff={mad:.6}");
    assert!(
        (wave.len() as i64 - ref_wave.len() as i64).abs() <= 1024,
        "waveform length {} differs from reference {} by more than a vocoder frame",
        wave.len(),
        ref_wave.len()
    );
    assert!(
        cos > 0.99,
        "assembled candle 44k pipeline waveform cosine {cos:.6} vs reference is below 0.99",
    );
}
