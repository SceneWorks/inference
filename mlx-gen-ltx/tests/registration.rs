//! Registry wiring + config-driven `load` (sc-2679 S0).
//!
//! Verifies `ltx_2_3` self-registers into the `mlx-gen` model registry with the right
//! descriptor, and that `load` reads the real `embedded_config.json` shape, returns a stub whose
//! `generate` errors with an explicit "S1–S5 pending" message, and rejects the not-yet-wired
//! sibling features (quant / adapters / single-file source).

use std::path::PathBuf;

use mlx_gen::{
    registry, AdapterKind, AdapterSpec, GenerationRequest, LoadSpec, Modality, Quant, WeightsSource,
};

use mlx_gen_ltx::MODEL_ID;

const EROS_EMBEDDED_CONFIG: &str = r#"{
  "transformer": {
    "_class_name": "AVTransformer3DModel",
    "attention_head_dim": 128,
    "caption_channels": 3840,
    "cross_attention_dim": 4096,
    "in_channels": 128,
    "norm_eps": 1e-06,
    "num_attention_heads": 32,
    "num_layers": 48,
    "out_channels": 128,
    "audio_num_attention_heads": 32,
    "audio_attention_head_dim": 64,
    "audio_cross_attention_dim": 2048,
    "use_embeddings_connector": true,
    "connector_attention_head_dim": 128,
    "connector_num_attention_heads": 32,
    "connector_num_layers": 8,
    "connector_positional_embedding_max_pos": [4096],
    "connector_num_learnable_registers": 128,
    "use_middle_indices_grid": true,
    "apply_gated_attention": true,
    "connector_apply_gated_attention": true,
    "caption_projection_first_linear": false,
    "caption_projection_second_linear": false,
    "audio_connector_attention_head_dim": 64,
    "audio_connector_num_attention_heads": 32,
    "cross_attention_adaln": true,
    "rope_type": "split",
    "frequencies_precision": "float64",
    "positional_embedding_theta": 10000.0,
    "positional_embedding_max_pos": [20, 2048, 2048],
    "timestep_scale_multiplier": 1000
  }
}"#;

/// A throwaway model dir holding just `embedded_config.json` (S0 `load` only reads config).
fn temp_model_dir(tag: &str) -> PathBuf {
    let dir = std::env::temp_dir().join(format!("ltx_s0_{}_{}", std::process::id(), tag));
    std::fs::create_dir_all(&dir).unwrap();
    std::fs::write(dir.join("embedded_config.json"), EROS_EMBEDDED_CONFIG).unwrap();
    dir
}

#[test]
fn ltx_is_registered() {
    let reg = registry::generators()
        .find(|r| (r.descriptor)().id == MODEL_ID)
        .expect("ltx_2_3 not registered");
    let d = (reg.descriptor)();
    assert_eq!(d.id, "ltx_2_3");
    assert_eq!(d.family, "ltx");
    assert_eq!(d.modality, Modality::Video);
    // Distilled core: no guidance / negative prompt; siblings (LoRA/LoKr) off.
    assert!(!d.capabilities.supports_guidance);
    assert!(!d.capabilities.supports_lora);
    assert!(!d.capabilities.requires_sigma_shift);
}

#[test]
fn load_reads_embedded_config_and_stubs_generate() {
    let dir = temp_model_dir("load");
    let g = registry::load(MODEL_ID, &LoadSpec::new(WeightsSource::Dir(dir.clone())))
        .expect("load should succeed (reads embedded_config.json)");
    assert_eq!(g.descriptor().id, MODEL_ID);

    // validate accepts a 64-aligned request; rejects mis-aligned + bad frame counts.
    let ok = GenerationRequest {
        width: 512,
        height: 512,
        frames: Some(33),
        ..Default::default()
    };
    assert!(g.validate(&ok).is_ok());
    let bad_size = GenerationRequest {
        width: 500,
        height: 512,
        ..Default::default()
    };
    assert!(g.validate(&bad_size).is_err());
    let bad_frames = GenerationRequest {
        width: 512,
        height: 512,
        frames: Some(32),
        ..Default::default()
    };
    assert!(g.validate(&bad_frames).is_err());

    // generate is an explicit WIP error until S1–S5.
    let mut noop = |_p| {};
    let err = g.generate(&ok, &mut noop).unwrap_err().to_string();
    assert!(err.contains("S1"), "expected WIP message, got: {err}");

    std::fs::remove_dir_all(&dir).ok();
}

#[test]
fn load_rejects_unwired_features() {
    let dir = temp_model_dir("reject");
    // Single-file source.
    assert!(registry::load(
        MODEL_ID,
        &LoadSpec::new(WeightsSource::File(dir.join("embedded_config.json")))
    )
    .is_err());
    // Quantization (sibling slice).
    assert!(registry::load(
        MODEL_ID,
        &LoadSpec::new(WeightsSource::Dir(dir.clone())).with_quant(Quant::Q8)
    )
    .is_err());
    // Adapters (sibling slice).
    let adapters = vec![AdapterSpec {
        path: dir.join("x.safetensors"),
        scale: 1.0,
        kind: AdapterKind::Lora,
    }];
    assert!(registry::load(
        MODEL_ID,
        &LoadSpec::new(WeightsSource::Dir(dir.clone())).with_adapters(adapters)
    )
    .is_err());

    std::fs::remove_dir_all(&dir).ok();
}
