//! sc-7911 — Krea 2 **Turbo inference-side LoRA/LoKr apply** real-weight smoke (epic 7565 P3).
//! Weight-gated (`#[ignore]`): trains a tiny adapter on the real `krea/Krea-2-Raw` DiT, then loads
//! the `krea_2_turbo` engine WITH that adapter and renders — proving the Raw-trained adapter is
//! installed (no "not supported" rejection) AND measurably changes the Turbo output while staying a
//! coherent image. This is the apply MECHANISM (sc-7911); the numeric train→infer quality bar is
//! sc-7579. Needs BOTH snapshots:
//!
//! ```sh
//! KREA_RAW_DIR=~/.cache/huggingface/hub/models--krea--Krea-2-Raw/snapshots/<rev> \
//! KREA_TURBO_DIR=~/.cache/huggingface/hub/models--krea--Krea-2-Turbo/snapshots/<rev> \
//!   cargo test -p mlx-gen-krea --release --test apply_real_weights -- --ignored --nocapture
//! ```

use std::path::PathBuf;

use mlx_gen::media::Image;
use mlx_gen::{
    AdapterKind, AdapterSpec, CancelFlag, GenerationOutput, GenerationRequest, LoadSpec,
    NetworkType, Quant, TrainingConfig, TrainingItem, TrainingProgress, TrainingRequest,
    WeightsSource,
};
use mlx_gen_krea::{load, load_trainer};

/// Resolve a cached HF snapshot dir (the `{env}` override, else the newest `models--{repo}` snapshot
/// with a `transformer/` tree).
fn snapshot(env: &str, repo_dir: &str) -> Option<PathBuf> {
    if let Ok(p) = std::env::var(env) {
        return Some(PathBuf::from(p));
    }
    let home = std::env::var("HOME").ok()?;
    let snaps = PathBuf::from(home)
        .join(".cache/huggingface/hub")
        .join(repo_dir)
        .join("snapshots");
    std::fs::read_dir(&snaps)
        .ok()?
        .filter_map(|e| e.ok())
        .map(|e| e.path())
        .find(|p| p.is_dir() && p.join("transformer").is_dir())
}

fn write_synth_image(path: &std::path::Path) {
    let mut img = image::RgbImage::new(320, 256);
    for (x, y, px) in img.enumerate_pixels_mut() {
        *px = image::Rgb([(x % 256) as u8, (y % 256) as u8, ((x + y) % 256) as u8]);
    }
    img.save(path).expect("write synth png");
}

/// (std, distinct-level-count, mean horizontal-adjacent-|Δ|) — a coherent image has a broad histogram
/// AND spatial smoothness; pure noise has a high adjacent Δ and flat std (mirrors `e2e_real_weights`).
fn image_stats(px: &[u8], w: u32) -> (f32, usize, f32) {
    let n = px.len() as f64;
    let mean = px.iter().map(|&v| v as f64).sum::<f64>() / n;
    let var = px.iter().map(|&v| (v as f64 - mean).powi(2)).sum::<f64>() / n;
    let mut seen = [false; 256];
    for &v in px {
        seen[v as usize] = true;
    }
    let distinct = seen.iter().filter(|&&b| b).count();
    let stride = (w * 3) as usize;
    let (mut adj_sum, mut adj_n) = (0f64, 0u64);
    for (i, &v) in px.iter().enumerate() {
        if i >= 3 && i % stride >= 3 {
            adj_sum += (v as i32 - px[i - 3] as i32).unsigned_abs() as f64;
            adj_n += 1;
        }
    }
    (
        var.sqrt() as f32,
        distinct,
        (adj_sum / adj_n.max(1) as f64) as f32,
    )
}

fn is_coherent(img: &Image) -> bool {
    let (std, distinct, adj) = image_stats(&img.pixels, img.width);
    std > 10.0 && distinct > 24 && adj < 60.0
}

/// Mean absolute per-byte difference between two equal-length RGB8 buffers.
fn mean_abs_diff(a: &[u8], b: &[u8]) -> f32 {
    assert_eq!(a.len(), b.len(), "buffers differ in size");
    let sum: u64 = a
        .iter()
        .zip(b)
        .map(|(&x, &y)| (x as i32 - y as i32).unsigned_abs() as u64)
        .sum();
    sum as f32 / a.len() as f32
}

/// Train a tiny adapter (`network_type`) on the real Raw DiT and return its `.safetensors` path.
fn train_tiny_adapter(raw: &std::path::Path, network_type: NetworkType, tag: &str) -> PathBuf {
    let tmp = std::env::temp_dir().join(format!("krea_apply_smoke_{tag}"));
    std::fs::create_dir_all(&tmp).unwrap();
    let img_path = tmp.join("swatch.png");
    write_synth_image(&img_path);

    let mut trainer =
        load_trainer(&LoadSpec::new(WeightsSource::Dir(raw.to_path_buf()))).expect("load_trainer");
    let req = TrainingRequest {
        items: vec![TrainingItem {
            image_path: img_path,
            caption: "a vivid abstract color swatch".into(),
        }],
        config: TrainingConfig {
            rank: 4,
            alpha: 4.0,
            steps: 3,
            resolution: 256,
            save_every: 0,
            learning_rate: 1e-4,
            network_type,
            ..Default::default()
        },
        output_dir: tmp.clone(),
        file_name: format!("krea_apply_{tag}.safetensors"),
        trigger_words: vec!["swatch".into()],
        cancel: CancelFlag::new(),
    };
    let out = trainer
        .train(&req, &mut |p| {
            if let TrainingProgress::Training { step, loss, .. } = p {
                eprintln!("[sc-7911 {tag}] train step {step} loss {loss:.5}");
            }
        })
        .expect("train");
    out.adapter_path
}

/// Load the `krea_2_turbo` engine (optionally with adapters) and render one 512² image.
fn render_turbo(turbo: &std::path::Path, adapters: Vec<AdapterSpec>) -> Image {
    let mut spec = LoadSpec::new(WeightsSource::Dir(turbo.to_path_buf()));
    match std::env::var("KREA_QUANT").ok().as_deref() {
        Some("q8") => spec = spec.with_quant(Quant::Q8),
        Some("q4") => spec = spec.with_quant(Quant::Q4),
        _ => {}
    }
    if !adapters.is_empty() {
        spec = spec.with_adapters(adapters);
    }
    let gen = load(&spec).expect("load krea_2_turbo engine (+adapters)");
    let req = GenerationRequest {
        prompt: "A medium-shot photograph of a red fox in a snowy forest at golden hour.".into(),
        width: 512,
        height: 512,
        count: 1,
        seed: Some(0),
        steps: Some(8),
        ..Default::default()
    };
    let out = gen.generate(&req, &mut |_| {}).expect("generate");
    let GenerationOutput::Images(mut imgs) = out else {
        panic!("expected GenerationOutput::Images");
    };
    imgs.pop().expect("one image")
}

#[test]
#[ignore = "needs real Krea 2 Raw + Turbo snapshots (~45 GB) + a Mac; run as its own process"]
fn raw_trained_lora_applies_at_turbo_inference() {
    let (Some(raw), Some(turbo)) = (
        snapshot("KREA_RAW_DIR", "models--krea--Krea-2-Raw"),
        snapshot("KREA_TURBO_DIR", "models--krea--Krea-2-Turbo"),
    ) else {
        eprintln!("skipping: set KREA_RAW_DIR + KREA_TURBO_DIR");
        return;
    };

    // Train a tiny LoRA on Raw, then render Turbo with and without it (same seed).
    let adapter = train_tiny_adapter(&raw, NetworkType::Lora, "lora");
    // A high scale makes a 3-step adapter's effect visible above sampler noise.
    let spec = AdapterSpec::new(adapter, 4.0, AdapterKind::Lora);

    let base = render_turbo(&turbo, Vec::new());
    let adapted = render_turbo(&turbo, vec![spec]);

    assert_eq!((base.width, base.height), (512, 512));
    assert_eq!((adapted.width, adapted.height), (512, 512));
    assert!(is_coherent(&base), "base Turbo render must be coherent");
    assert!(
        is_coherent(&adapted),
        "adapter-applied Turbo render must stay coherent (not noise)"
    );

    let diff = mean_abs_diff(&base.pixels, &adapted.pixels);
    eprintln!("[sc-7911] base↔adapted mean|Δ| = {diff:.3}");
    assert!(
        diff > 0.5,
        "the Raw-trained LoRA must measurably change the Turbo output (mean|Δ|={diff:.3})"
    );
}

#[test]
#[ignore = "needs real Krea 2 Raw + Turbo snapshots (~45 GB) + a Mac; run as its own process"]
fn raw_trained_lokr_applies_at_turbo_inference() {
    let (Some(raw), Some(turbo)) = (
        snapshot("KREA_RAW_DIR", "models--krea--Krea-2-Raw"),
        snapshot("KREA_TURBO_DIR", "models--krea--Krea-2-Turbo"),
    ) else {
        eprintln!("skipping: set KREA_RAW_DIR + KREA_TURBO_DIR");
        return;
    };

    // The LoKr path rides the same `apply_adapters_strict` seam (Kronecker delta → residual); confirm
    // a Raw-trained LoKr loads + renders coherently at Turbo.
    let adapter = train_tiny_adapter(&raw, NetworkType::Lokr, "lokr");
    let spec = AdapterSpec::new(adapter, 2.0, AdapterKind::Lokr);
    let adapted = render_turbo(&turbo, vec![spec]);

    assert_eq!((adapted.width, adapted.height), (512, 512));
    assert!(
        is_coherent(&adapted),
        "LoKr-applied Turbo render must be a coherent image"
    );
}
