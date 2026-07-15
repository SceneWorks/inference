//! Registry wiring + `load` rejection paths (A5, sc-11989) — the candle twin of `mlx-gen-mochi`'s
//! `registration.rs`. Non-ignored (no weights): verifies the Mochi provider registry exposes `mochi_1`
//! with the right descriptor (text-to-video, true CFG, no conditioning, no on-the-fly quant, backend
//! `candle`) and that `load` rejects a single-file source, a stray on-the-fly `spec.quantize`, and an
//! incomplete snapshot. The request-validation logic is unit-tested weight-free in `lib.rs`.

use std::path::PathBuf;

use candle_gen::gen_core::{LoadSpec, Modality, Quant, WeightsSource};

use candle_gen_mochi::MODEL_ID;

/// A throwaway empty model dir (an incomplete snapshot — no `vae/config.json`).
fn temp_model_dir(tag: &str) -> PathBuf {
    let dir = std::env::temp_dir().join(format!("mochi_candle_reg_{}_{}", std::process::id(), tag));
    std::fs::create_dir_all(&dir).unwrap();
    dir
}

/// A throwaway **synthetic q4 tier dir** (no weights) — `<root>/q4/` with `transformer/` +
/// `split_model.json` (the tier marker) + `quantize_config.json` (bits 4, group 64), plus a
/// `<root>/text_encoder/` so the shared-root resolves to the parent. Enough for the lazy `load` seam to
/// detect the tier and run the assert-against-manifest, without any model weights. Returns the tier dir.
fn synthetic_q4_tier(tag: &str) -> PathBuf {
    let root =
        std::env::temp_dir().join(format!("mochi_candle_tier_{}_{}", std::process::id(), tag));
    let tier = root.join("q4");
    std::fs::create_dir_all(tier.join("transformer")).unwrap();
    std::fs::create_dir_all(root.join("text_encoder")).unwrap();
    std::fs::write(
        tier.join("split_model.json"),
        r#"{"quantized": true, "quantization_bits": 4, "quantization_group_size": 64}"#,
    )
    .unwrap();
    std::fs::write(
        tier.join("quantize_config.json"),
        r#"{"quantization": {"bits": 4, "group_size": 64}}"#,
    )
    .unwrap();
    tier
}

#[test]
fn mochi_is_registered_as_candle_video() {
    let reg = candle_gen_mochi::provider_registry()
        .unwrap()
        .generators()
        .copied()
        .find(|r| (r.descriptor)().id == MODEL_ID)
        .expect("mochi_1 not registered");
    let d = (reg.descriptor)();
    assert_eq!(d.id, "mochi_1");
    assert_eq!(d.family, "mochi");
    assert_eq!(d.backend, "candle");
    assert_eq!(d.modality, Modality::Video);
    // Not distilled: true CFG (negative prompt + guidance). Text-to-video only (no conditioning).
    assert!(d.capabilities.supports_negative_prompt);
    assert!(d.capabilities.supports_guidance);
    assert!(d.capabilities.supports_true_cfg);
    assert!(!d.capabilities.mac_only);
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
        candle_gen_mochi::provider_registry()
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
        candle_gen_mochi::provider_registry()
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

/// A pre-quantized tier dir IS the quant selection: pointing `WeightsSource` at a q4 tier loads with no
/// `spec.quantize`, and a `spec.quantize` that AGREES with the tier's manifest is honoured (the
/// assert-against-manifest, A6 sc-11990) — the blanket on-the-fly-requant reject no longer blocks a tier.
#[test]
fn load_accepts_tier_dir_and_matching_quant() {
    let tier = synthetic_q4_tier("match");
    let reg = candle_gen_mochi::provider_registry().unwrap();
    // The dir alone (no quant knob) is a valid tier selection.
    assert!(
        reg.load(MODEL_ID, &LoadSpec::new(WeightsSource::Dir(tier.clone())))
            .is_ok(),
        "a q4 tier dir must load with no spec.quantize (the dir is the selection)"
    );
    // A matching `spec.quantize` (Q4) agrees with the manifest → accepted.
    assert!(
        reg.load(
            MODEL_ID,
            &LoadSpec::new(WeightsSource::Dir(tier.clone())).with_quant(Quant::Q4)
        )
        .is_ok(),
        "spec.quantize=Q4 matches the q4 tier manifest → accepted"
    );
    std::fs::remove_dir_all(tier.parent().unwrap()).ok();
}

/// A `spec.quantize` that DISAGREES with the tier's manifest is a hard error (never a silent wrong-tier
/// run) — Q8 requested against a q4 tier fails the assert-against-manifest.
#[test]
fn load_rejects_tier_quant_bits_mismatch() {
    let tier = synthetic_q4_tier("mismatch");
    let reg = candle_gen_mochi::provider_registry().unwrap();
    assert!(
        reg.load(
            MODEL_ID,
            &LoadSpec::new(WeightsSource::Dir(tier.clone())).with_quant(Quant::Q8)
        )
        .is_err(),
        "spec.quantize=Q8 against the q4 tier must error (assert-against-manifest)"
    );
    std::fs::remove_dir_all(tier.parent().unwrap()).ok();
}

/// `load` is lazy (components load on first `generate`), so an incomplete snapshot is caught at
/// generation, not construction. A well-formed request against an empty dir must therefore error
/// (no `text_encoder/`, `transformer/`, `vae/`).
#[test]
fn generate_requires_full_snapshot() {
    use candle_gen::gen_core::GenerationRequest;
    let dir = temp_model_dir("incomplete");
    let g = candle_gen_mochi::provider_registry()
        .unwrap()
        .load(MODEL_ID, &LoadSpec::new(WeightsSource::Dir(dir.clone())))
        .expect("lazy load constructs");
    let req = GenerationRequest {
        prompt: "a calico kitten".into(),
        width: 64,
        height: 64,
        frames: Some(7),
        ..Default::default()
    };
    assert!(
        g.generate(&req, &mut |_| {}).is_err(),
        "incomplete snapshot (no component dirs) must fail at generate"
    );
    std::fs::remove_dir_all(&dir).ok();
}

/// Request validation (mirrors the `mlx-gen-mochi` model tests): empty prompt, misaligned size, and a
/// bad frame count are rejected; a well-formed `1 + 6·k` request passes.
#[test]
fn validate_gates_prompt_size_and_frames() {
    use candle_gen::gen_core::GenerationRequest;
    let g = candle_gen_mochi::provider_registry()
        .unwrap()
        .load(
            MODEL_ID,
            &LoadSpec::new(WeightsSource::Dir("/nonexistent".into())),
        )
        .unwrap();
    let base = GenerationRequest {
        prompt: "a calico kitten".into(),
        width: 64,
        height: 64,
        frames: Some(7), // 1 + 6·1
        ..Default::default()
    };
    assert!(g.validate(&base).is_ok(), "well-formed request validates");
    assert!(g
        .validate(&GenerationRequest {
            prompt: String::new(),
            ..base.clone()
        })
        .is_err());
    assert!(g
        .validate(&GenerationRequest {
            width: 72, // not a multiple of 16
            ..base.clone()
        })
        .is_err());
    assert!(g
        .validate(&GenerationRequest {
            frames: Some(8), // not 1 + 6·k
            ..base.clone()
        })
        .is_err());
    assert!(g
        .validate(&GenerationRequest {
            frames: Some(13), // 1 + 6·2
            ..base
        })
        .is_ok());
}
