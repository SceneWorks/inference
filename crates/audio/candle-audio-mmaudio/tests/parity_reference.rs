//! End-to-end **reference-parity** gate for the shipping MMAudio assembly (sc-12843).
//!
//! The four MMAudio components (CLIP, Synchformer, MM-DiT, 16k VAE+BigVGAN) were each numerically
//! parity-verified against PyTorch in their own slices (cos≈1.0). This test verifies the **assembly
//! this story adds** — the video→audio pipeline wiring: `preprocess_conditions`, the negative-text
//! empty/CFG conditions (`get_empty_conditions(negative_text_features=…)`), variable-duration
//! `update_seq_lengths`, the Euler-25 / CFG-4.5 flow-matching loop, un-normalization, the
//! latent→mel→waveform decode — end to end against the reference's own output.
//!
//! `scripts` (scratchpad `mmaudio_ref_dump.py`) runs the reference `MMAudio` full pipeline in f32 on
//! a fixed video + prompt + seed and dumps, as safetensors: the encoded conditioning features
//! (`clip_f`/`sync_f`/`text_f`/`neg_text_f`), the seeded prior `x0`, and the reference waveform
//! (`ref_wave`). Injecting the reference's own features + prior isolates the assembly from the
//! (already-verified) encoders AND from torch-vs-Rust RNG, so a high waveform cosine proves the
//! assembled candle pipeline is faithful end to end.
//!
//! ```text
//! MMAUDIO_PARITY_DUMP=/path/to/mmaudio_ref_dump.safetensors \
//!   cargo test --locked -p candle-audio-mmaudio --test parity_reference -- --ignored --nocapture
//! ```

mod common;

use candle_audio_mmaudio::candle_audio;
use candle_audio_mmaudio::candle_audio::candle_core::{safetensors, Tensor};
use candle_audio_mmaudio::MmAudioPipeline;

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

#[test]
#[ignore = "real weights + a reference dump: set MMAUDIO_PARITY_DUMP (see scratchpad/mmaudio_ref_dump.py); run with --ignored"]
fn assembly_matches_reference_waveform() {
    let dump = std::env::var("MMAUDIO_PARITY_DUMP")
        .expect("set MMAUDIO_PARITY_DUMP to the reference safetensors dump");
    let device = candle_audio::default_device().expect("device");
    let t = safetensors::load(&dump, &device).expect("load reference dump");
    let get = |k: &str| -> Tensor {
        t.get(k)
            .unwrap_or_else(|| panic!("dump missing {k}"))
            .clone()
    };
    let clip_f = get("clip_f");
    let sync_f = get("sync_f");
    let text_f = get("text_f");
    let neg_f = get("neg_text_f");
    let x0 = get("x0");
    let ref_wave: Vec<f32> = get("ref_wave").flatten_all().unwrap().to_vec1().unwrap();
    let scalars: Vec<f32> = get("scalars").to_vec1().unwrap(); // [cfg, steps, duration, src_fps]
    let cfg = scalars[0] as f64;
    let steps = scalars[1] as usize;
    println!(
        "dump: clip_f{:?} sync_f{:?} x0{:?} cfg={cfg} steps={steps} ref_wave={}",
        clip_f.dims(),
        sync_f.dims(),
        x0.dims(),
        ref_wave.len()
    );

    // Resolve + load the real pinned weights from the five named components (network + VAE + BigVGAN;
    // the CLIP/sync encoders load too but are unused here — we inject the reference's features).
    let pipeline = MmAudioPipeline::from_components(
        &common::clip_source(),
        &common::synchformer_source(),
        &common::dit_16k_source(),
        &common::vae_16k_source(),
        &common::vocoder_16k_source(),
        &device,
    )
    .expect("load pipeline");

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
        .expect("candle assembly synthesize_from_features");

    let cos = cosine(&wave, &ref_wave);
    let mad = max_abs_diff(&wave, &ref_wave);
    println!(
        "candle wave: {} samples; reference: {} samples",
        wave.len(),
        ref_wave.len()
    );
    println!("E2E PARITY: cosine={cos:.6}  max_abs_diff={mad:.6}");
    assert!(
        (wave.len() as i64 - ref_wave.len() as i64).abs() <= 512,
        "waveform length {} differs from reference {} by more than a codec frame",
        wave.len(),
        ref_wave.len()
    );
    assert!(
        cos > 0.99,
        "assembled candle pipeline waveform cosine {cos:.6} vs reference is below 0.99 — the \
         assembly diverges from MMAudio"
    );
}
