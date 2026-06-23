//! sc-7582 — candle Krea 2 **Turbo** end-to-end real-weight smoke (the Windows/CUDA twin of
//! `mlx-gen-krea`'s `e2e_real_weights.rs`). Loads the full registered engine (`krea_2_turbo`:
//! tokenizer + Qwen3-VL-4B TE + single-stream DiT + Qwen-Image VAE), renders a 1024² image through the
//! `Generator` contract, gates programmatic coherence (a velocity-sign or schedule-direction bug yields
//! pure noise → fails the smoothness gate), and saves the PNG for eyeballing against the mlx render.
//!
//! `#[ignore]` — needs the real snapshot (~12 B params; bf16 ≈ 24 GB resident). Run on the Windows GPU:
//! ```sh
//! KREA_TURBO_DIR=D:\models\Krea-2-Turbo \
//!   cargo test -p candle-gen-krea --release --features cuda --test e2e_real_weights -- --ignored --nocapture
//! ```

use std::path::PathBuf;
use std::time::Instant;

use candle_gen::gen_core::{
    registry, GenerationOutput, GenerationRequest, Image, LoadSpec, WeightsSource,
};

const PROMPT: &str =
    "A medium-shot photograph of a red fox sitting in a snowy forest at golden hour.";

fn snapshot() -> Option<PathBuf> {
    std::env::var("KREA_TURBO_DIR").ok().map(PathBuf::from)
}

/// (std, distinct-level-count, mean horizontal-adjacent-|Δ|) over an RGB8 buffer — a coherent natural
/// image has a broad histogram AND spatial smoothness; pure noise has a high adjacent Δ and flat std.
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

/// A real Turbo render has a broad histogram (`std`/`distinct`) and spatial smoothness (`adjΔ`); pure
/// noise (the failure mode of a flow-sign / schedule-direction bug) fails the `adjΔ` gate.
fn is_coherent(img: &Image) -> bool {
    let (std, distinct, adj) = image_stats(&img.pixels, img.width);
    std > 10.0 && distinct > 24 && adj < 60.0
}

fn save(img: &Image, name: &str) {
    let dir = std::env::temp_dir().join("krea_turbo_smoke");
    std::fs::create_dir_all(&dir).unwrap();
    let path = dir.join(format!("{name}.png"));
    image::save_buffer(
        &path,
        &img.pixels,
        img.width,
        img.height,
        image::ExtendedColorType::Rgb8,
    )
    .unwrap();
    eprintln!("  saved {}", path.display());
}

fn render(width: u32, height: u32) {
    candle_gen_krea::force_link();
    let Some(root) = snapshot() else {
        eprintln!("skipping: set KREA_TURBO_DIR");
        return;
    };

    // The same `load` the `krea_2_turbo` registry entry dispatches to (registration is unit-tested in
    // `tests::registers_krea_2_turbo_as_candle`).
    let spec = LoadSpec::new(WeightsSource::Dir(root));
    let t_load = Instant::now();
    let gen = registry::load("krea_2_turbo", &spec).expect("load krea_2_turbo engine");
    let load_s = t_load.elapsed().as_secs_f32();

    let req = GenerationRequest {
        prompt: PROMPT.into(),
        width,
        height,
        count: 1,
        seed: Some(0),
        steps: Some(8),
        ..Default::default()
    };

    let t_gen = Instant::now();
    let out = gen.generate(&req, &mut |_| {}).expect("generate");
    let gen_s = t_gen.elapsed().as_secs_f32();

    let GenerationOutput::Images(imgs) = out else {
        panic!("expected GenerationOutput::Images");
    };
    assert_eq!(imgs.len(), 1, "count=1 → one image");
    let img = &imgs[0];
    assert_eq!((img.width, img.height), (width, height), "output dims");

    let (std, distinct, adj) = image_stats(&img.pixels, img.width);
    eprintln!(
        "[krea_2_turbo {width}x{height} 8-step] load {load_s:.1}s · render {gen_s:.1}s · \
         std={std:.1} distinct={distinct} adjΔ={adj:.1} coherent={}",
        is_coherent(img)
    );
    save(img, &format!("fox_{width}x{height}_s8"));
    assert!(
        is_coherent(img),
        "Turbo render must be a coherent image, not noise (std={std:.1} distinct={distinct} adjΔ={adj:.1})"
    );
}

#[test]
#[ignore = "needs the real Krea 2 Turbo snapshot (set KREA_TURBO_DIR)"]
fn turbo_engine_renders_coherent_1024() {
    render(1024, 1024);
}

#[test]
#[ignore = "needs the real snapshot (KREA_TURBO_DIR); larger footprint — run if it fits"]
fn turbo_engine_renders_coherent_2048() {
    render(2048, 2048);
}
