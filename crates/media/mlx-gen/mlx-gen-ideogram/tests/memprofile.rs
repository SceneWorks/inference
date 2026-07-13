//! sc-5989 — Ideogram 4 quantization validation + memory profiling. Loads the model dense / Q8 / Q4
//! via the registry, measures the **load transient** (dense bf16 peak before in-place quantize) and
//! the **steady-state weights** (active memory after quantize), then generates and measures the
//! **runtime peak** (weights + activations) at a chosen resolution — the numbers that set
//! `minMemoryGb` and motivate whether to publish pre-quantized-on-disk weights (sc-5990).
//!
//! In-process via `mlx_rs::memory` (Metal wired memory is not in `ps` RSS / `/usr/bin/time` would
//! only catch the process peak). `#[ignore]` — needs the converted snapshot (~53 GB). Run e.g.:
//!   IDEOGRAM4_QUANT=q4 IDEOGRAM4_SMOKE_RES=1024 IDEOGRAM4_SMOKE_STEPS=8 \
//!     cargo test -p mlx-gen-ideogram --test memprofile -- --ignored --nocapture

mod common;

use std::path::PathBuf;

use common::CAPTION_JSON;
use mlx_gen::{GenerationOutput, GenerationRequest, LoadSpec, Quant, WeightsSource};
use mlx_gen_ideogram::MODEL_ID;
use mlx_rs::memory::{clear_cache, get_active_memory, get_peak_memory, reset_peak_memory};

fn snapshot_dir() -> PathBuf {
    std::env::var("IDEOGRAM4_MLX")
        .map(PathBuf::from)
        .unwrap_or_else(|_| {
            PathBuf::from(std::env::var("HOME").expect("HOME")).join(".cache/ideogram4-mlx-convert")
        })
}

fn gb(bytes: usize) -> f64 {
    bytes as f64 / (1024.0 * 1024.0 * 1024.0)
}

#[test]
#[ignore = "needs converted weights (~53 GB)"]
fn profile_footprint() {
    let quant = match std::env::var("IDEOGRAM4_QUANT")
        .unwrap_or_default()
        .to_lowercase()
        .as_str()
    {
        "q4" => Some(Quant::Q4),
        "q8" => Some(Quant::Q8),
        "" | "none" | "bf16" => None,
        other => panic!("IDEOGRAM4_QUANT must be q4|q8|none, got {other:?}"),
    };
    let envn = |k: &str, d: u32| {
        std::env::var(k)
            .ok()
            .and_then(|v| v.parse().ok())
            .unwrap_or(d)
    };
    let res = envn("IDEOGRAM4_SMOKE_RES", 256);
    let steps = envn("IDEOGRAM4_SMOKE_STEPS", 8);
    let label = quant.map(|q| format!("{q:?}")).unwrap_or("bf16".into());

    let mut spec = LoadSpec::new(WeightsSource::Dir(snapshot_dir()));
    if let Some(q) = quant {
        spec = spec.with_quant(q);
    }

    // ── Load. NOTE: MLX lazy-loads weights via mmap and the in-place `quantize` builds a lazy graph
    // — nothing is materialized into Metal until the first forward — so `get_active_memory()` here
    // is ~0. The real dense→quant transient (if any) is captured by the generate peak below, because
    // the quantize evaluates per-weight during the first forward (after the reset). ──
    reset_peak_memory();
    let g = mlx_gen_ideogram::provider_registry()
        .unwrap()
        .load(MODEL_ID, &spec)
        .expect("registry load ideogram_4");
    println!(
        "[{label}] post-load active {:.2} GB (lazy mmap — materializes on first forward)",
        gb(get_active_memory())
    );

    // ── Generate: runtime peak = weights + activations at the target resolution (the number that
    // gates whether a given Mac can run it; it already includes any lazy quantize transient). ──
    clear_cache();
    reset_peak_memory();
    let req = GenerationRequest {
        prompt: CAPTION_JSON.into(),
        width: res,
        height: res,
        steps: Some(steps),
        guidance: Some(7.0),
        seed: Some(0),
        ..Default::default()
    };
    let out = g.generate(&req, &mut |_| {}).expect("generate");
    let gen_peak = get_peak_memory();
    clear_cache();
    let steady_weights = get_active_memory();
    println!(
        "[{label}] generate @{res}²/{steps}step: runtime peak {:.2} GB (weights + activations); \
         steady weights resident {:.2} GB",
        gb(gen_peak),
        gb(steady_weights)
    );

    // The image must be real (quant didn't silently break the forward).
    let imgs = match out {
        GenerationOutput::Images(v) => v,
        other => panic!("expected Images, got {other:?}"),
    };
    let im = &imgs[0];
    assert_eq!((im.width, im.height), (res, res));
    let (mn, mx) = (
        *im.pixels.iter().min().unwrap(),
        *im.pixels.iter().max().unwrap(),
    );
    assert!(
        mx > mn,
        "[{label}] degenerate image — quant broke the forward"
    );

    let out_path = std::env::temp_dir().join(format!("ideogram4_{label}_{res}.png"));
    image::RgbImage::from_raw(res, res, im.pixels.clone())
        .unwrap()
        .save(&out_path)
        .unwrap();
    println!("[{label}] wrote {}", out_path.display());
}
