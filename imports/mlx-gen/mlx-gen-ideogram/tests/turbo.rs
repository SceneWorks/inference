//! Ideogram 4 **turbo** (issue #488): the CFG-free single-DiT few-step path driven by the ostris
//! TurboTime LoRA. Two `#[ignore]` real-weight tests:
//!
//! 1. `turbo_host_map_covers_turbotime_targets` — the `AdaptableHost` key→module map resolves every
//!    TurboTime target (the 6 per-layer modules × 34 layers) on the REAL transformer tree, rejects
//!    off-surface paths, and `adaptable_paths()` round-trips (every advertised path resolves). Needs
//!    only the base `transformer/` weights — no LoRA file.
//!
//! 2. `turbo_generates_single_dit_8step` — load `ideogram_4_turbo` through the registry (applies the
//!    bundled `turbo_lora.safetensors` onto a single DiT) and render the canonical fox caption at
//!    8 steps, asserting a non-degenerate image. Needs a turbo snapshot (base components + the
//!    bundled LoRA).
//!
//! Run (base transformer for the routing map):
//!   IDEOGRAM4_MLX=~/.cache/ideogram4-mlx-convert \
//!     cargo test -p mlx-gen-ideogram --test turbo turbo_host_map -- --ignored --nocapture
//! Run (turbo render — the snapshot dir must contain `turbo_lora.safetensors`):
//!   IDEOGRAM4_TURBO_MLX=~/.cache/ideogram4-mlx-turbo \
//!     cargo test -p mlx-gen-ideogram --test turbo turbo_generates -- --ignored --nocapture

mod common;

use std::path::PathBuf;

use common::CAPTION_JSON;
use mlx_gen::adapters::AdaptableHost;
use mlx_gen::{GenerationOutput, GenerationRequest, LoadSpec, WeightsSource};
use mlx_gen_ideogram::{load_transformer, MODEL_ID_TURBO};

/// The base converted snapshot (its `transformer/` is the routing-map subject).
fn base_snapshot_dir() -> PathBuf {
    std::env::var("IDEOGRAM4_MLX")
        .map(PathBuf::from)
        .unwrap_or_else(|_| {
            PathBuf::from(std::env::var("HOME").expect("HOME")).join(".cache/ideogram4-mlx-convert")
        })
}

/// A turbo snapshot: base components + the bundled `turbo_lora.safetensors`.
fn turbo_snapshot_dir() -> PathBuf {
    std::env::var("IDEOGRAM4_TURBO_MLX")
        .map(PathBuf::from)
        .unwrap_or_else(|_| {
            PathBuf::from(std::env::var("HOME").expect("HOME")).join(".cache/ideogram4-mlx-turbo")
        })
}

/// The 6 TurboTime per-layer module sub-paths (verified from the safetensors header, issue #488).
const TURBOTIME_LEAVES: [&str; 6] = [
    "attention.qkv",
    "attention.o",
    "feed_forward.w1",
    "feed_forward.w2",
    "feed_forward.w3",
    "adaln_modulation",
];
/// `Ideogram4DitConfig::v4().num_layers`.
const NUM_LAYERS: usize = 34;

#[test]
#[ignore = "needs the base transformer weights (~17 GB)"]
fn turbo_host_map_covers_turbotime_targets() {
    let mut t = load_transformer(&base_snapshot_dir()).expect("load conditional DiT");

    let resolves = |t: &mut _, p: &str| -> bool {
        let segs: Vec<&str> = p.split('.').collect();
        AdaptableHost::adaptable_mut(t, &segs).is_some()
    };

    // Every TurboTime target (6 modules × 34 layers) must resolve — else `apply_adapters_strict`
    // would surface an unmatched target and abort the turbo load.
    for i in 0..NUM_LAYERS {
        for leaf in TURBOTIME_LEAVES {
            let p = format!("layers.{i}.{leaf}");
            assert!(resolves(&mut t, &p), "TurboTime target {p} should resolve");
        }
    }

    // Off-surface paths return None (loud no-silent-drop): a bogus leaf, a bogus layer index, and a
    // non-adaptable norm.
    for p in [
        "layers.0.attention.bogus",
        "layers.34.attention.qkv", // out of range (0..=33)
        "layers.0.attention_norm1",
        "final_layer.bogus",
        "bogus_global",
    ] {
        assert!(!resolves(&mut t, p), "off-surface {p} must NOT resolve");
    }

    // Every advertised `adaptable_paths()` entry resolves (the kohya flattened→dotted contract).
    for p in t.adaptable_paths() {
        let segs: Vec<&str> = p.split('.').collect();
        assert!(
            AdaptableHost::adaptable_mut(&mut t, &segs).is_some(),
            "advertised path {p} must resolve via adaptable_mut"
        );
    }
}

#[test]
#[ignore = "needs a turbo snapshot (base components + turbo_lora.safetensors)"]
fn turbo_generates_single_dit_8step() {
    // Full production path: registry load applies the bundled TurboTime LoRA onto a single DiT.
    let spec = LoadSpec::new(WeightsSource::Dir(turbo_snapshot_dir()));
    let gen = mlx_gen::load(MODEL_ID_TURBO, &spec).expect("ideogram_4_turbo loads via registry");

    let envn = |k: &str, d: u32| {
        std::env::var(k)
            .ok()
            .and_then(|v| v.parse().ok())
            .unwrap_or(d)
    };
    let res = envn("IDEOGRAM4_SMOKE_RES", 1024);
    let steps = envn("IDEOGRAM4_TURBO_STEPS", 8);
    let req = GenerationRequest {
        prompt: CAPTION_JSON.into(),
        width: res,
        height: res,
        count: 1,
        seed: Some(0),
        steps: Some(steps),
        // No `guidance` — turbo is CFG-free (descriptor advertises supports_guidance=false).
        ..Default::default()
    };
    println!("ideogram_4_turbo → {res}x{res} / {steps} steps (CFG-free single DiT) …");

    let out = gen.generate(&req, &mut |_| {}).expect("turbo generate");
    let img = match out {
        GenerationOutput::Images(mut v) => v.pop().expect("one image"),
        other => panic!("expected Images, got {other:?}"),
    };
    assert_eq!((img.width, img.height), (res, res), "image dims");

    let px: Vec<i32> = img.pixels.iter().map(|&v| v as i32).collect();
    let (min, max) = (*px.iter().min().unwrap(), *px.iter().max().unwrap());
    let mean = px.iter().map(|&v| v as f64).sum::<f64>() / px.len() as f64;
    println!(
        "turbo image px range [{min}, {max}], mean {mean:.1}, {} px",
        px.len()
    );
    assert!(max > min, "degenerate (constant) turbo image — no signal");

    let out_path = std::env::temp_dir().join("ideogram4_turbo_8step.png");
    image::RgbImage::from_raw(res, res, img.pixels.clone())
        .unwrap()
        .save(&out_path)
        .unwrap();
    println!("wrote {}", out_path.display());
}
