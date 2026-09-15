//! End-to-end **reference-parity** gate for the shipping MMAudio assembly (sc-12843, sc-17285).
//!
//! The four MMAudio components (CLIP, Synchformer, MM-DiT, 16k VAE+BigVGAN) were each numerically
//! parity-verified against PyTorch in their own slices (cos≈1.0). This test verifies the **assembly
//! this story adds** — the video→audio pipeline wiring: `preprocess_conditions`, the negative-text
//! empty/CFG conditions (`get_empty_conditions(negative_text_features=…)`), variable-duration
//! `update_seq_lengths`, the Euler-25 / CFG-4.5 flow-matching loop, un-normalization, the
//! latent→mel→waveform decode — end to end against the reference's own output.
//!
//! `scripts/reference/mmaudio_reference.py` runs the vendored reference `MMAudio` full pipeline in
//! f32 on the CPU over a fixed synthetic clip + prompt + seed and commits, as safetensors: the
//! encoded conditioning features (`clip_f`/`sync_f`/`text_f`/`neg_text_f`), the seeded prior `x0`,
//! and the reference waveform (`ref_wave`). Injecting the reference's own features + prior isolates
//! the assembly from the (already-verified) encoders AND from torch-vs-Rust RNG, so a high waveform
//! cosine proves the assembled candle pipeline is faithful end to end.
//!
//! The fixture is **committed** (sc-17285). Before that this test read its dump path from
//! `MMAUDIO_PARITY_DUMP`, which no workflow set and no committed script produced, so this gate had
//! never run anywhere.
//!
//! ```text
//! cargo test --locked -p candle-audio-mmaudio --test parity_reference -- --ignored --nocapture
//! ```

mod common;

use candle_audio_mmaudio::candle_audio::candle_core::Device;
use candle_audio_mmaudio::MmAudioPipeline;

const FIXTURE: &str = "mmaudio_parity_reference.safetensors";

/// Largest tolerated `max_abs_diff / abs_max(reference)`. Cosine is scale-invariant, so it alone
/// would pass a port with a uniform gain error; this is the magnitude half of the gate. Measured
/// 3.0e-4 on CPU/f32, so the bound leaves ~17x headroom.
const MAX_RELATIVE_DIFF: f64 = 5e-3;

#[test]
#[ignore = "real weights: needs the five MMAudio component snapshots; run with --ignored"]
fn assembly_matches_reference_waveform() {
    // Pinned to CPU, not `default_device()`. Under `--features cuda` — how the real-weight lane
    // builds this — `default_device()` returns a CUDA device, and the fixture is a CPU/f32 torch
    // run, so the comparison would silently become candle-CUDA vs torch-CPU and the thresholds
    // below would be gating an accelerator's rounding rather than the port. What this test exists
    // to prove (weight mapping, layer order, sampler schedule) is device-independent; the CUDA
    // path is exercised by `conformance_video_to_audio.rs` on the same lane.
    let device = Device::Cpu;
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

    let cos = common::cosine(&wave, &ref_wave);
    let mad = common::max_abs_diff(&wave, &ref_wave);
    let rel = mad / common::abs_max(&ref_wave);
    println!(
        "candle wave: {} samples; reference: {} samples",
        wave.len(),
        ref_wave.len()
    );
    println!("E2E PARITY: cosine={cos:.6}  max_abs_diff={mad:.6}  relative={rel:.2e}");
    // Exact, not a per-frame tolerance: the sequence lengths are fixed by the fixture's duration,
    // so a shorter output is a defect. `cosine`/`max_abs_diff` compare the overlapping prefix, so
    // without this a truncated waveform would score ~1.0 and pass.
    assert_eq!(
        wave.len(),
        ref_wave.len(),
        "waveform length differs from reference"
    );
    assert!(
        cos > 0.99,
        "assembled candle pipeline waveform cosine {cos:.6} vs reference is below 0.99 — the \
         assembly diverges from MMAudio"
    );
    assert!(
        rel < MAX_RELATIVE_DIFF,
        "assembled candle pipeline waveform relative max-abs-diff {rel:.2e} exceeds \
         {MAX_RELATIVE_DIFF:.0e} — the waveform is mis-scaled even if its shape matches"
    );
}
