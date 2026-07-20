//! End-to-end **reference-parity** gate for the shipping MMAudio **44.1 kHz** assembly (sc-13441) —
//! the 44k twin of `parity_reference.rs`.
//!
//! Runs the reference `MMAudio` full pipeline with `--variant large_44k_v2` in f32 on a fixed video +
//! prompt + seed and dumps, as safetensors: the encoded conditioning features
//! (`clip_f`/`sync_f`/`text_f`/`neg_text_f`), the seeded prior `x0` (shape `(1, latent, 40)`), and the
//! reference 44.1 kHz waveform (`ref_wave`). Injecting the reference's own features + prior isolates
//! this crate's 44k assembly (large_44k_v2 MM-DiT + 44k VAE + NVIDIA BigVGAN v2) from the (separately
//! parity-verified) shared encoders AND from torch-vs-Rust RNG, so a high waveform cosine proves the
//! assembled candle 44k pipeline is faithful end to end.
//!
//! ```text
//! MMAUDIO_PARITY_DUMP_44K=/path/to/mmaudio_44k_ref_dump.safetensors \
//!   cargo test --locked -p candle-audio-mmaudio --test parity_reference_44k -- --ignored --nocapture
//! ```

use candle_audio_mmaudio::candle_audio;
use candle_audio_mmaudio::candle_audio::candle_core::{safetensors, Tensor};
use candle_audio_mmaudio::MmAudio44kPipeline;

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
#[ignore = "real weights + a reference dump: set MMAUDIO_PARITY_DUMP_44K; run with --ignored"]
fn assembly_44k_matches_reference_waveform() {
    let dump = std::env::var("MMAUDIO_PARITY_DUMP_44K")
        .expect("set MMAUDIO_PARITY_DUMP_44K to the reference safetensors dump");
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

    let snap =
        candle_audio_mmaudio::resolve_pinned_snapshot_44k().expect("resolve pinned 44k snapshot");
    let dir = match snap {
        candle_audio::gen_core::WeightsSource::Dir(d) => d,
        candle_audio::gen_core::WeightsSource::File(f) => f,
    };
    let pipeline = MmAudio44kPipeline::from_snapshot(&dir, &device).expect("load 44k pipeline");

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

    let cos = cosine(&wave, &ref_wave);
    let mad = max_abs_diff(&wave, &ref_wave);
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
