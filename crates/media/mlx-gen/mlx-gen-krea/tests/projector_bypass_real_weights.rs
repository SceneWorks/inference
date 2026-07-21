//! Krea 2 **ComfyUI diff-patch "filter-bypass"** real-weight harness (sc-13825, MLX parity for candle
//! sc-13726). Weight-gated (`#[ignore]`): the community bypass adapter `krea2filterbypass3.safetensors`
//! is a single diff-patch tensor `diffusion_model.txtfusion.projector.diff` (F32 `[1, num_text_layers]`)
//! — a full-weight delta on the 12→1 `text_fusion.projector` collapse, not a LoRA/LoKr. Two layers:
//!
//! - **fold MECHANISM** (`projector_bypass_folds_exact_delta`, fast, no render) — loads the real DiT via
//!   the public [`load_transformer`], folds the real bypass file through the exact seam
//!   [`KreaHeavy::apply_adapters`] uses ([`apply_adapters_strict_with_diff_patch`]), and asserts the
//!   dense projector base becomes `W + δ` bit-for-bit against the file's own `.diff` tensor (F32, scale
//!   1.0), that the fold counted 1, and that the change is nonzero (candle measured |Δ|≈0.168).
//! - **render** (`turbo_engine_renders_with_projector_bypass`) — the full public `load()` → `generate()`
//!   path with the bypass in `LoadSpec::with_adapters`, same seed: renders coherent and **differs** from
//!   the unadapted baseline, with no "no target modules matched" error.
//!
//! ```sh
//! KREA_TURBO_DIR=/path/to/krea-2-turbo-mlx/snapshots/<rev>[/q8] \
//! KREA_BYPASS_FILE=/path/to/krea2filterbypass3.safetensors \
//!   cargo test -p mlx-gen-krea --release --test projector_bypass_real_weights -- --ignored --nocapture
//! ```

use std::path::PathBuf;

use mlx_gen::adapters::loader::apply_adapters_strict_with_diff_patch;
use mlx_gen::adapters::AdaptableHost;
use mlx_gen::media::Image;
use mlx_gen::weights::Weights;
use mlx_gen::{
    AdapterKind, AdapterSpec, GenerationOutput, GenerationRequest, LoadSpec, Quant, WeightsSource,
};
use mlx_gen_krea::{load, load_transformer};
use mlx_rs::ops::{abs, max, subtract};
use mlx_rs::{Array, Dtype};

/// The bypass file's single diff-patch key.
const PROJECTOR_DIFF_KEY: &str = "diffusion_model.txtfusion.projector.diff";

fn turbo_dir() -> Option<PathBuf> {
    std::env::var("KREA_TURBO_DIR").ok().map(PathBuf::from)
}

fn bypass_file() -> Option<PathBuf> {
    std::env::var("KREA_BYPASS_FILE").ok().map(PathBuf::from)
}

fn max_abs(a: &Array) -> f32 {
    max(abs(a).unwrap(), false).unwrap().item::<f32>()
}

/// The bypass file folds `W += δ` into the dense projector base bit-for-bit — the direct proof the
/// filter-bypass "works": the projector is patched with exactly the file's delta, on whatever tier the
/// snapshot is (the projector stays dense on dense/Q4/Q8). Exercises the SAME seam
/// `KreaHeavy::apply_adapters` calls, via public API only.
#[test]
#[ignore = "needs a Krea-2 turbo snapshot (KREA_TURBO_DIR) + the bypass file (KREA_BYPASS_FILE)"]
fn projector_bypass_folds_exact_delta() {
    let (Some(turbo), Some(bypass)) = (turbo_dir(), bypass_file()) else {
        eprintln!("skipping: set KREA_TURBO_DIR and KREA_BYPASS_FILE");
        return;
    };

    let mut dit = load_transformer(&turbo).expect("load real Krea-2 transformer");
    let projector_path = ["text_fusion", "projector"];

    // Snapshot the dense projector base before the fold.
    let base = dit
        .adaptable_mut(&projector_path)
        .expect("projector must route on the real DiT")
        .dense_weight()
        .expect("projector base is dense on every tier")
        .0
        .clone();

    // The file's own delta (F32 [1, num_text_layers]) — the expected `merged - base`.
    let file = Weights::from_file(&bypass).expect("load bypass file");
    let delta = file
        .require(PROJECTOR_DIFF_KEY)
        .expect("bypass carries the projector .diff")
        .clone();
    assert_eq!(
        delta.shape(),
        base.shape(),
        "the bypass .diff must match the projector base shape"
    );

    let report = apply_adapters_strict_with_diff_patch(
        &mut dit,
        &[AdapterSpec::new(bypass.clone(), 1.0, AdapterKind::Lora)],
        "krea_2",
    )
    .expect("the projector diff-patch must fold, not error with 'no target modules matched'");
    assert_eq!(report.applied, 1, "exactly the projector diff folded");

    let merged = dit
        .adaptable_mut(&projector_path)
        .unwrap()
        .dense_weight()
        .unwrap()
        .0
        .clone();

    let got_delta = subtract(
        merged.as_dtype(Dtype::Float32).unwrap(),
        base.as_dtype(Dtype::Float32).unwrap(),
    )
    .unwrap();
    let fold_err = max_abs(&subtract(&got_delta, delta.as_dtype(Dtype::Float32).unwrap()).unwrap());
    let moved = max_abs(&got_delta);
    eprintln!("[sc-13825] projector fold: |applied delta|={moved:.4}, |fold err|={fold_err:.2e}");
    assert!(
        fold_err < 1e-4,
        "projector base must equal W + δ (err {fold_err:e})"
    );
    assert!(
        moved > 1e-3,
        "the bypass must actually move the projector (|Δ|={moved:e})"
    );
}

// ── render A/B (mirrors candle `turbo_engine_applies_projector_bypass`) ───────────────────────────

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

fn mean_abs_diff(a: &[u8], b: &[u8]) -> f32 {
    assert_eq!(a.len(), b.len());
    let sum: u64 = a
        .iter()
        .zip(b)
        .map(|(&x, &y)| (x as i32 - y as i32).unsigned_abs() as u64)
        .sum();
    sum as f32 / a.len() as f32
}

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
    let gen = load(&spec).expect("load krea_2_turbo engine (+bypass)");
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

/// The full public `load()`→`generate()` path with the bypass applied renders coherent and, same seed,
/// **differs** from the unadapted baseline — no "no target modules matched" error, no crash.
#[test]
#[ignore = "needs a Krea-2 turbo snapshot (KREA_TURBO_DIR) + the bypass file (KREA_BYPASS_FILE); renders on Metal"]
fn turbo_engine_renders_with_projector_bypass() {
    let (Some(turbo), Some(bypass)) = (turbo_dir(), bypass_file()) else {
        eprintln!("skipping: set KREA_TURBO_DIR and KREA_BYPASS_FILE");
        return;
    };

    let base = render_turbo(&turbo, vec![]);
    let bypassed = render_turbo(
        &turbo,
        vec![AdapterSpec::new(bypass, 1.0, AdapterKind::Lora)],
    );

    assert!(is_coherent(&base), "baseline render must be coherent");
    assert!(
        is_coherent(&bypassed),
        "bypass render must be coherent (the projector fold must not break the net)"
    );
    let d = mean_abs_diff(&base.pixels, &bypassed.pixels);
    eprintln!("[sc-13825] render A/B mean|Δ| = {d:.2}");
    assert!(
        d > 0.5,
        "the projector bypass must visibly change the same-seed render (mean|Δ|={d})"
    );
}
