//! Krea Realtime 14B **real** Wan-family style-LoRA validation (sc-8446, S13).
//!
//! `tests/style_lora.rs` proves the install mechanics on synthetic low-rank files. This proves the
//! thing that could not be proven synthetically: that **published** Wan-2.1-14B LoRA files resolve
//! against this DiT's [`AdaptableHost`] surface, install through `LoadSpec::adapters` →
//! `apply_adapters_strict`, and **measurably change the render** — without tripping the strict
//! installer's unmatched-target hard error.
//!
//! ## The two real classes, and why the globals surface was widened (the S13 decision)
//!
//! Surveyed from the published safetensors headers of real Wan-2.1-14B LoRA files:
//!
//! | class | example | targets |
//! |---|---|---|
//! | plain style | `shauray/Origami_WanLora`, `motimalu/wan-flat-color-v2` | 400 per-block stems only |
//! | step-distill | lightx2v `Wan2.1-T2V-14B` cfg-step-distill v2, `FastWan` T2V-14B | the same 400 **plus all 7 whole-model globals** |
//! | I2V-family | `Remade-AI/Squish` | per-block **plus `cross_attn.k_img`/`v_img`** |
//!
//! The step-distill class is what settled the decision: those globals carry genuine `lora_down`/
//! `lora_up` factors, so **soft-skipping** them would have silently installed a step-distill LoRA with
//! its patch/text/time/head projections missing — a wrong render that reports success. Widening the
//! surface applies them instead (see `causal::krea_adaptable_paths`). The I2V-only image cross-attention
//! stays unexposed and still errors loudly: those modules do not exist on a T2V backbone at any surface
//! width.
//!
//! ```text
//! KREA_REALTIME_SNAPSHOT_DIR=~/.cache/krea-realtime-mlx-snapshot/q4 \
//! KREA_STYLE_LORA=~/.cache/wan-loras/origami/origami_000000500.safetensors \
//!   cargo test -p mlx-gen-krea-realtime --test style_lora_real_weights -- --ignored --nocapture
//! ```
//!
//! Env: `KREA_REALTIME_SNAPSHOT_DIR`, `KREA_STYLE_LORA` (a real Wan style LoRA `.safetensors`),
//! `KREA_DISTILL_LORA` (optional — a real Wan step-distill LoRA that targets the globals),
//! `KREA_LORA_W`/`_H`/`_FRAMES` (default 832×480, 13 frames — small: this is an A/B, not a showreel).

use std::path::PathBuf;

use mlx_gen::{
    AdapterKind, AdapterSpec, GenerationOutput, GenerationRequest, Image, LoadSpec, WeightsSource,
};
use mlx_gen_krea_realtime::MODEL_ID;

fn snapshot_dir() -> PathBuf {
    std::env::var("KREA_REALTIME_SNAPSHOT_DIR")
        .map(PathBuf::from)
        .unwrap_or_else(|_| {
            PathBuf::from(std::env::var("HOME").unwrap()).join(".cache/krea-realtime-mlx-convert")
        })
}

fn env_usize(key: &str, default: usize) -> usize {
    std::env::var(key)
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(default)
}

fn require_lora(var: &str) -> PathBuf {
    let p = PathBuf::from(
        std::env::var(var).unwrap_or_else(|_| panic!("set {var} to a real Wan LoRA .safetensors")),
    );
    assert!(
        p.is_file(),
        "{var} does not point at a file: {}",
        p.display()
    );
    p
}

fn request(w: usize, h: usize, frames: usize) -> GenerationRequest {
    GenerationRequest {
        prompt: "a paper crane folding itself on a wooden desk, warm light".into(),
        width: w as u32,
        height: h as u32,
        frames: Some(frames as u32),
        seed: Some(1234),
        fps: Some(24),
        sampler: Some("self_forcing".into()),
        ..Default::default()
    }
}

/// Render `req` with the given adapters and return the decoded frames.
fn render(adapters: Vec<AdapterSpec>, req: &GenerationRequest, label: &str) -> Vec<Image> {
    let root = snapshot_dir();
    assert!(
        root.join("dit.safetensors").exists() || root.join("transformer").is_dir(),
        "no Krea Realtime DiT at {} — set KREA_REALTIME_SNAPSHOT_DIR",
        root.display()
    );
    let mut spec = LoadSpec::new(WeightsSource::Dir(root));
    spec.adapters = adapters;
    let t0 = std::time::Instant::now();
    let gen = mlx_gen_krea_realtime::provider_registry()
        .unwrap()
        .load(MODEL_ID, &spec)
        .unwrap_or_else(|e| panic!("{label}: load must succeed: {e}"));
    let out = gen
        .generate(req, &mut |_| {})
        .unwrap_or_else(|e| panic!("{label}: generate must succeed: {e}"));
    println!("  {label}: rendered in {:.1?}", t0.elapsed());
    match out {
        GenerationOutput::Video { frames, .. } => frames,
        other => panic!("{label}: expected a Video output, got {other:?}"),
    }
}

/// Mean absolute per-byte difference between two clips of the same geometry, in 0..255 units.
fn mean_abs_delta(a: &[Image], b: &[Image]) -> f64 {
    assert_eq!(a.len(), b.len(), "clip lengths differ");
    let mut total = 0.0f64;
    let mut n = 0usize;
    for (x, y) in a.iter().zip(b.iter()) {
        assert_eq!(x.pixels.len(), y.pixels.len(), "frame buffer sizes differ");
        total += x
            .pixels
            .iter()
            .zip(y.pixels.iter())
            .map(|(&p, &q)| (p as f64 - q as f64).abs())
            .sum::<f64>();
        n += x.pixels.len();
    }
    total / n as f64
}

/// **(a) loads, (b) measurably changes the output, (c) no strict-apply hard error** — the three things
/// story sc-8446 asks of a real Wan style LoRA, on one seed so the only difference is the adapter.
///
/// Discriminating in both directions: the same LoRA at **scale 0** must leave the render essentially
/// where the baseline is, so a "the renders differ" pass cannot be produced by nondeterminism alone —
/// if it could, the scale-0 arm would differ by the same amount and the ordering assertion fails.
#[test]
#[ignore = "real snapshot + a real Wan LoRA; run with --ignored on macOS (see module doc)"]
fn real_wan_style_lora_loads_and_changes_the_render() {
    let w = env_usize("KREA_LORA_W", 832);
    let h = env_usize("KREA_LORA_H", 480);
    let frames = env_usize("KREA_LORA_FRAMES", 13);
    let lora = require_lora("KREA_STYLE_LORA");
    let req = request(w, h, frames);

    let base = render(Vec::new(), &req, "baseline");
    let with_lora = render(
        vec![AdapterSpec::new(lora.clone(), 1.0, AdapterKind::Lora)],
        &req,
        "lora@1.0",
    );
    let zeroed = render(
        vec![AdapterSpec::new(lora, 0.0, AdapterKind::Lora)],
        &req,
        "lora@0.0",
    );

    let applied = mean_abs_delta(&base, &with_lora);
    let noop = mean_abs_delta(&base, &zeroed);
    println!("  mean |Δ| vs baseline: lora@1.0 = {applied:.3}, lora@0.0 = {noop:.3} (0..255)");

    assert!(
        applied > 1.0,
        "the LoRA did not measurably change the render (mean |Δ| {applied:.3}) — it loaded but is \
         inert"
    );
    assert!(
        applied > noop * 4.0,
        "the LoRA's effect ({applied:.3}) is not clearly above the scale-0 floor ({noop:.3}) — the \
         'it changed the output' claim would not be attributable to the adapter"
    );
}

/// The **step-distill** class: a real Wan-T2V LoRA that also targets the seven whole-model globals must
/// now install rather than hard-error. This is the S13 globals decision under test on a real file, so a
/// regression that re-narrows the surface fails here with the installer's own unmatched-target message.
///
/// Optional: skipped when `KREA_DISTILL_LORA` is unset, because it needs a ~1.2 GB second LoRA.
#[test]
#[ignore = "real snapshot + a real Wan step-distill LoRA; run with --ignored on macOS"]
fn real_wan_step_distill_lora_installs_over_the_widened_globals() {
    let Ok(path) = std::env::var("KREA_DISTILL_LORA") else {
        eprintln!("skip: set KREA_DISTILL_LORA to a real Wan T2V step-distill LoRA .safetensors");
        return;
    };
    let path = PathBuf::from(path);
    assert!(path.is_file(), "KREA_DISTILL_LORA: {}", path.display());

    let w = env_usize("KREA_LORA_W", 832);
    let h = env_usize("KREA_LORA_H", 480);
    let frames = env_usize("KREA_LORA_FRAMES", 13);
    let req = request(w, h, frames);

    let base = render(Vec::new(), &req, "baseline");
    let distilled = render(
        vec![AdapterSpec::new(path, 1.0, AdapterKind::Lora)],
        &req,
        "step-distill",
    );
    let delta = mean_abs_delta(&base, &distilled);
    println!("  step-distill mean |Δ| vs baseline: {delta:.3} (0..255)");
    assert!(
        delta > 1.0,
        "the step-distill LoRA installed but changed nothing (mean |Δ| {delta:.3})"
    );
}
