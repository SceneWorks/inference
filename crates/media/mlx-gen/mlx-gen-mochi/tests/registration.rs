//! Registry wiring + `load` rejection paths (A4, sc-11988).
//!
//! Verifies the Mochi provider registry exposes `mochi_1` with the right descriptor (text-to-video,
//! true CFG, no conditioning, no on-the-fly quant) and that `load` rejects a single-file source, a
//! stray on-the-fly `spec.quantize` (Mochi ships pre-quantized per-tier checkpoints — epic 1788 / A6),
//! and an incomplete snapshot (no `vae/config.json`). The request-validation logic (empty prompt,
//! 16-divisible size, `num_frames = 1 + 6·k`) is unit-tested weight-free in `model.rs`.

use std::path::PathBuf;

use mlx_gen::{LoadSpec, Modality, Quant, WeightsSource};

use mlx_gen_mochi::MODEL_ID;

/// A throwaway empty model dir (an incomplete snapshot — no `vae/config.json`).
fn temp_model_dir(tag: &str) -> PathBuf {
    let dir = std::env::temp_dir().join(format!("mochi_reg_{}_{}", std::process::id(), tag));
    std::fs::create_dir_all(&dir).unwrap();
    dir
}

#[test]
fn mochi_is_registered() {
    let reg = mlx_gen_mochi::provider_registry()
        .unwrap()
        .generators()
        .copied()
        .find(|r| (r.descriptor)().id == MODEL_ID)
        .expect("mochi_1 not registered");
    let d = (reg.descriptor)();
    assert_eq!(d.id, "mochi_1");
    assert_eq!(d.family, "mochi");
    assert_eq!(d.backend, "mlx");
    assert_eq!(d.modality, Modality::Video);
    // Not distilled: true CFG (negative prompt + guidance). Text-to-video only (no conditioning).
    assert!(d.capabilities.supports_negative_prompt);
    assert!(d.capabilities.supports_guidance);
    assert!(d.capabilities.supports_true_cfg);
    assert!(d.capabilities.mac_only);
    assert_eq!(d.capabilities.max_count, 1);
    assert!(d.capabilities.conditioning.is_empty());
    // Quant tiers are pre-quantized per-tier checkpoints, not on-the-fly requant (epic 1788 / A6).
    assert!(d.capabilities.supported_quants.is_empty());
    assert!(!d.capabilities.supports_lora);
    assert!(!d.capabilities.supports_lokr);
}

#[test]
fn load_rejects_single_file_source() {
    let dir = temp_model_dir("single");
    assert!(
        mlx_gen_mochi::provider_registry()
            .unwrap()
            .load(
                MODEL_ID,
                &LoadSpec::new(WeightsSource::File(dir.join("model.safetensors")))
            )
            .is_err(),
        "single-file source must not load (Mochi is a split-weight snapshot dir)"
    );
    std::fs::remove_dir_all(&dir).ok();
}

#[test]
fn load_rejects_on_the_fly_quant() {
    let dir = temp_model_dir("quant");
    assert!(
        mlx_gen_mochi::provider_registry()
            .unwrap()
            .load(
                MODEL_ID,
                &LoadSpec::new(WeightsSource::Dir(dir.clone())).with_quant(Quant::Q4)
            )
            .is_err(),
        "on-the-fly quant is not the Mochi tier mechanism — must reject"
    );
    std::fs::remove_dir_all(&dir).ok();
}

#[test]
fn load_requires_full_snapshot() {
    // An empty dir has no `vae/config.json` (nor T5/DiT shards) — `load` must error rather than
    // return a stub. (The full-model load + generate is exercised by the real-weights `e2e_parity`.)
    let dir = temp_model_dir("incomplete");
    assert!(
        mlx_gen_mochi::provider_registry()
            .unwrap()
            .load(MODEL_ID, &LoadSpec::new(WeightsSource::Dir(dir.clone())))
            .is_err(),
        "incomplete snapshot (no vae/config.json) must not load"
    );
    std::fs::remove_dir_all(&dir).ok();
}
