//! Epic 2123 E1 golden: with every training technique off, a seeded multi-step `train()` on the
//! committed miniature snapshot writes **the adapter the pre-epic trainer wrote**. The golden
//! (`tests/fixtures/pre_epic_golden/adapter.safetensors`) is the adapter THIS file wrote when run
//! unchanged at the pre-epic merge base `8f986217a` (the merge base of `origin/main` and
//! `feature/sc-2123-perceptual-character-lora`), on candle's CPU backend (the default lane's
//! device — no accelerator in the result).
//!
//! Regenerate (only when the off path is *meant* to change, which epic 2123 E1 forbids): check out
//! that merge base in a scratch worktree, copy this file into the same place, add it to
//! `tests/main.rs`, and run
//! `PRE_EPIC_GOLDEN_WRITE=<this crate>/tests/fixtures/pre_epic_golden/adapter.safetensors
//! cargo test -p candle-gen-qwen-image-2-1 --test integration -- pre_epic_golden::` there.
//!
//! The comparison: identical key names and shapes, and every tensor within [`RTOL`] of the
//! golden's peak — loose enough for a SIMD/reduction-order difference between CPU architectures
//! (CI's x86-64 vs the arm64 box that wrote the golden), far below what a technique leaking into
//! the off path does (an extra RNG draw, a changed sample order, a noise term: percent-level moves
//! on every element).

use std::collections::HashMap;
use std::path::{Path, PathBuf};

use candle_core::Device;
use candle_gen::gen_core::{
    LoadSpec, TrainingConfig, TrainingItem, TrainingProgress, TrainingRequest, WeightsSource,
};
use candle_gen_qwen_image_2_1::provider_registry;

use crate::common::{host_f32, tiny_snapshot};

/// Per-tensor tolerance: `max |got − golden| ≤ RTOL · max |golden|`.
const RTOL: f64 = 1e-3;

/// The committed pre-epic adapter.
fn golden_path() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/pre_epic_golden/adapter.safetensors")
}

/// A deterministic 64×64 RGB PNG (a diagonal colour ramp) written under `dir`.
fn write_image(dir: &Path, name: &str, phase: u32) -> PathBuf {
    let img = image::RgbImage::from_fn(64, 64, |x, y| {
        image::Rgb([
            ((x * 4 + phase) % 256) as u8,
            ((y * 4 + 2 * phase) % 256) as u8,
            (((x + y) * 2) % 256) as u8,
        ])
    });
    let path = dir.join(name);
    img.save(&path).expect("write the dataset image");
    path
}

/// An adapter's tensors as `name → (shape, f32 values)`.
fn read_adapter(path: &Path) -> HashMap<String, (Vec<i64>, Vec<f32>)> {
    candle_core::safetensors::load(path, &Device::Cpu)
        .unwrap_or_else(|e| panic!("read {}: {e}", path.display()))
        .into_iter()
        .map(|(k, t)| {
            let shape = t.dims().iter().map(|&d| d as i64).collect();
            (k, (shape, host_f32(&t)))
        })
        .collect()
}

#[test]
fn everything_off_trains_the_pre_epic_adapter() {
    let tmp = tempfile::Builder::new()
        .prefix(&format!("qwen21_pre_epic_golden_{}_", std::process::id()))
        .tempdir()
        .unwrap();
    let data = tmp.path().join("data");
    std::fs::create_dir_all(&data).unwrap();
    let items = vec![
        TrainingItem::captioned(write_image(&data, "a.png", 0), "a red swatch".into()),
        TrainingItem::captioned(write_image(&data, "b.png", 40), "a blue ramp".into()),
    ];
    let req = TrainingRequest {
        items,
        config: TrainingConfig {
            rank: 4,
            alpha: 4.0,
            learning_rate: 1e-2,
            steps: 4,
            resolution: 64,
            save_every: 0,
            seed: 7,
            train_dtype: "f32".into(),
            ..Default::default()
        },
        output_dir: tmp.path().join("out"),
        file_name: "golden.safetensors".into(),
        trigger_words: Vec::new(),
        cancel: Default::default(),
    };
    let mut trainer = provider_registry()
        .unwrap()
        .load_trainer(
            "qwen_image_2_1",
            &LoadSpec::new(WeightsSource::Dir(tiny_snapshot())),
        )
        .expect("the tiny snapshot loads as a trainer");
    let out = trainer
        .train(&req, &mut |_: TrainingProgress| {})
        .expect("the seeded run trains");
    assert_eq!(out.steps, 4);
    if let Some(dst) = std::env::var_os("PRE_EPIC_GOLDEN_WRITE") {
        let dst = PathBuf::from(dst);
        std::fs::create_dir_all(dst.parent().unwrap()).unwrap();
        std::fs::copy(&out.adapter_path, &dst).unwrap();
        eprintln!("[pre_epic_golden] wrote {}", dst.display());
    }
    let got = read_adapter(&out.adapter_path);
    let want = read_adapter(&golden_path());
    let mut names: Vec<&String> = want.keys().collect();
    names.sort();
    let mut got_names: Vec<&String> = got.keys().collect();
    got_names.sort();
    assert_eq!(got_names, names, "adapter key names changed");
    let mut trained_b = false;
    for name in names {
        let ((gs, g), (ws, w)) = (&got[name], &want[name]);
        assert_eq!(gs, ws, "{name}: shape changed");
        let peak = w.iter().fold(0f64, |m, &v| m.max(f64::from(v).abs()));
        let diff = g.iter().zip(w).fold(0f64, |m, (&a, &b)| {
            m.max((f64::from(a) - f64::from(b)).abs())
        });
        assert!(
            diff <= RTOL * peak,
            "{name}: max |Δ| {diff:.3e} vs the pre-epic adapter (peak {peak:.3e}, rtol {RTOL})"
        );
        trained_b |= (name.contains("lora_B") || name.contains("lora_up")) && peak > 0.0;
    }
    assert!(trained_b, "the golden carries no trained up factor");
}
