//! sc-18757 — split-checkpoint loading for LTX-2.5 components (candle side).
//!
//! The mirror of `mlx-gen-ltx/tests/split_checkpoint.rs`, minus the `embedded_config.json` reads
//! candle does not do: this engine pins the LTX-2.3 structure as constants, so the **2.3 regression**
//! here is that feeding the shipped 2.3 config section to the new per-component readers reproduces
//! those constants exactly, and that a 2.3 checkpoint still selects the all-in-one layout so `load`
//! takes the branch it always took.
//!
//! The LTX-2.5 fixtures reproduce the `__metadata__` layout of the shipped `Lightricks/LTX-2.5`
//! folder structure. The weights are gated on Hugging Face, so the fixtures are written on disk —
//! but their shape matches the real headers sc-18756 captured under
//! `docs/reference/sc-18756-headers/`, which `gen_core::ltx_checkpoint`'s own tests parse directly.
//! Reference: `Lightricks/LTX-2` @ `d1511477` — `ltx_core/loader/sft_loader.py`, the per-component
//! `model_configurator.py` files, and `encoder_configurator._check_gemma_version`.
//!
//! No CUDA feature gate: every assertion is a path + JSON read with no device involved.

use std::path::Path;

use candle_gen::gen_core::ltx_checkpoint::{
    CaptionFeatureVersion, GemmaVersionCheck, LtxCheckpointLayout, LtxComponent,
};
use candle_gen::gen_core::{LoadSpec, WeightsSource};
use candle_gen_ltx::bundle::{
    assert_gemma_version, declared_layout, declared_model_version, resolve_split_bundle,
};
use candle_gen_ltx::config::{
    AudioVaeConfig, AvConfig, ConnectorConfig, VideoVaeDeclaration, VocoderConfig,
};

/// A minimal, valid safetensors file: 8-byte header length, header JSON (one tensor plus
/// `__metadata__`), then that tensor's bytes. Only the header is ever read.
fn write_component(path: &Path, metadata: &[(&str, &str)]) {
    let meta_json: String = metadata
        .iter()
        .map(|(k, v)| format!("{}:{}", serde_json::json!(k), serde_json::json!(v)))
        .collect::<Vec<_>>()
        .join(",");
    let header = format!(
        r#"{{"__metadata__":{{{meta_json}}},"w":{{"dtype":"F32","shape":[1],"data_offsets":[0,4]}}}}"#
    );
    let mut bytes = (header.len() as u64).to_le_bytes().to_vec();
    bytes.extend_from_slice(header.as_bytes());
    bytes.extend_from_slice(&0_f32.to_le_bytes());
    std::fs::create_dir_all(path.parent().unwrap()).unwrap();
    std::fs::write(path, bytes).unwrap();
}

const VERSION: (&str, &str) = ("model_version", "2.5.0");

// =================================================================================================
// LTX-2.3 regression.
// =================================================================================================

/// The `transformer` section of the shipped `SceneWorks/ltx-2.3-mlx` `embedded_config.json`, verbatim
/// for the keys these readers consume.
const LTX_2_3_TRANSFORMER_SECTION: &str = r#"{
    "_class_name": "AVTransformer3DModel",
    "attention_head_dim": 128, "num_attention_heads": 32, "num_layers": 48,
    "in_channels": 128, "out_channels": 128, "cross_attention_dim": 4096,
    "caption_channels": 3840, "norm_eps": 1e-06,
    "audio_num_attention_heads": 32, "audio_attention_head_dim": 64,
    "audio_out_channels": 128, "audio_cross_attention_dim": 2048,
    "audio_positional_embedding_max_pos": [20],
    "use_embeddings_connector": true,
    "connector_attention_head_dim": 128, "connector_num_attention_heads": 32,
    "connector_num_layers": 8, "connector_positional_embedding_max_pos": [4096],
    "connector_num_learnable_registers": 128,
    "audio_connector_attention_head_dim": 64, "audio_connector_num_attention_heads": 32,
    "use_middle_indices_grid": true, "apply_gated_attention": true,
    "connector_apply_gated_attention": true,
    "caption_projection_first_linear": false, "caption_projection_second_linear": false,
    "cross_attention_adaln": true, "rope_type": "split", "frequencies_precision": "float64",
    "positional_embedding_theta": 10000.0,
    "positional_embedding_max_pos": [20, 2048, 2048],
    "timestep_scale_multiplier": 1000
}"#;

/// The `audio_vae.model.params.ddconfig` block of the same file.
const LTX_2_3_DDCONFIG: &str = r#"{
    "double_z": true, "mel_bins": 64, "z_channels": 8, "resolution": 256,
    "in_channels": 2, "out_ch": 2, "ch": 128, "ch_mult": [1, 2, 4],
    "num_res_blocks": 2, "dropout": 0.0, "mid_block_add_attention": false,
    "norm_type": "pixel", "causality_axis": "height"
}"#;

/// The `vocoder` block of the same file.
const LTX_2_3_VOCODER: &str = r#"{
    "vocoder": {
        "upsample_initial_channel": 1536, "resblock": "AMP1",
        "upsample_rates": [5, 2, 2, 2, 2, 2],
        "resblock_kernel_sizes": [3, 7, 11],
        "upsample_kernel_sizes": [11, 4, 4, 4, 4, 4],
        "resblock_dilation_sizes": [[1, 3, 5], [1, 3, 5], [1, 3, 5]],
        "stereo": true, "use_tanh_at_final": false, "activation": "snakebeta",
        "use_bias_at_final": false
    },
    "bwe": {
        "upsample_initial_channel": 512, "resblock": "AMP1",
        "upsample_rates": [6, 5, 2, 2, 2],
        "resblock_kernel_sizes": [3, 7, 11],
        "upsample_kernel_sizes": [12, 11, 4, 4, 4],
        "resblock_dilation_sizes": [[1, 3, 5], [1, 3, 5], [1, 3, 5]],
        "stereo": true, "use_tanh_at_final": false, "activation": "snakebeta",
        "use_bias_at_final": false, "apply_final_activation": false,
        "input_sampling_rate": 16000, "output_sampling_rate": 48000,
        "hop_length": 80, "n_fft": 512, "win_size": 512, "num_mels": 64
    }
}"#;

#[test]
fn the_shipped_2_3_sections_reproduce_the_pinned_constants() {
    // The regression that matters for this engine: the new per-component readers, fed the config the
    // shipped LTX-2.3 checkpoint actually carries, must land on exactly the constants the current pin
    // hardcodes. A drift here would silently re-shape the 2.3 DiT.
    let t: serde_json::Value = serde_json::from_str(LTX_2_3_TRANSFORMER_SECTION).unwrap();
    // The shipped 2.3 section declares only the legacy caption pair, so this exercises the
    // measured carve-out: it must resolve, not error.
    let av = AvConfig::from_transformer_config(&t).expect("the shipped 2.3 caption shape resolves");
    assert_eq!(av.caption_feature_version, CaptionFeatureVersion::V2);
    let pinned = AvConfig::ltx_2_3();
    assert_eq!(av.video.num_layers, pinned.video.num_layers);
    assert_eq!(av.video.num_heads, pinned.video.num_heads);
    assert_eq!(av.video.head_dim, pinned.video.head_dim);
    assert_eq!(av.video.inner_dim(), pinned.video.inner_dim());
    assert!((av.video.norm_eps - pinned.video.norm_eps).abs() < f64::EPSILON);
    assert!((av.video.rope_theta - pinned.video.rope_theta).abs() < f64::EPSILON);
    assert_eq!(av.video.rope_max_pos, pinned.video.rope_max_pos);
    assert!(
        (av.video.timestep_scale_multiplier - pinned.video.timestep_scale_multiplier).abs()
            < f64::EPSILON
    );
    assert_eq!(av.audio_heads, pinned.audio_heads);
    assert_eq!(av.audio_head_dim, pinned.audio_head_dim);
    assert_eq!(av.audio_inner(), pinned.audio_inner());
    assert_eq!(av.cross_inner, pinned.cross_inner);
    assert_eq!(av.audio_max_pos, pinned.audio_max_pos);
    assert_eq!(av.cross_max_pos, pinned.cross_max_pos);

    let video_conn = ConnectorConfig::from_transformer_config(&t);
    let pinned_conn = ConnectorConfig::ltx_2_3();
    assert_eq!(video_conn.num_layers, pinned_conn.num_layers);
    assert_eq!(video_conn.num_heads, pinned_conn.num_heads);
    assert_eq!(video_conn.head_dim, pinned_conn.head_dim);
    assert_eq!(video_conn.num_registers, pinned_conn.num_registers);
    assert_eq!(video_conn.max_pos, pinned_conn.max_pos);

    let audio_conn = ConnectorConfig::audio_from_transformer_config(&t);
    let pinned_audio = ConnectorConfig::ltx_2_3_audio();
    assert_eq!(audio_conn.num_layers, pinned_audio.num_layers);
    assert_eq!(audio_conn.num_heads, pinned_audio.num_heads);
    assert_eq!(audio_conn.head_dim, pinned_audio.head_dim);
    assert_eq!(audio_conn.num_registers, pinned_audio.num_registers);
    assert_eq!(audio_conn.max_pos, pinned_audio.max_pos);

    let dd: serde_json::Value = serde_json::from_str(LTX_2_3_DDCONFIG).unwrap();
    assert_eq!(
        AudioVaeConfig::from_ddconfig(&dd),
        AudioVaeConfig::ltx_2_3()
    );

    let voc: serde_json::Value = serde_json::from_str(LTX_2_3_VOCODER).unwrap();
    let parsed = VocoderConfig::from_vocoder_config(&voc);
    let pinned_voc = VocoderConfig::ltx_2_3();
    assert_eq!(parsed.core, pinned_voc.core);
    assert_eq!(parsed.bwe, pinned_voc.bwe);
    assert_eq!(parsed.output_sample_rate, pinned_voc.output_sample_rate);
    assert_eq!(
        parsed.bwe_input_sample_rate,
        pinned_voc.bwe_input_sample_rate
    );
    assert_eq!(
        parsed.bwe_output_sample_rate,
        pinned_voc.bwe_output_sample_rate
    );
    assert_eq!(parsed.bwe_hop_length, pinned_voc.bwe_hop_length);
    assert_eq!(parsed.bwe_win_length, pinned_voc.bwe_win_length);
    assert_eq!(parsed.final_sample_rate(), 48000);
}

#[test]
fn a_2_3_checkpoint_still_selects_the_all_in_one_layout() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("ltx-2.3-22b-distilled.safetensors");
    write_component(
        &path,
        &[
            ("model_version", "2.3.0"),
            (
                "config",
                &format!(
                    r#"{{"transformer":{LTX_2_3_TRANSFORMER_SECTION},"vae":{{"latent_channels":128}}}}"#
                ),
            ),
        ],
    );
    assert_eq!(
        declared_model_version(&path).unwrap().as_deref(),
        Some("2.3.0")
    );
    assert_eq!(
        declared_layout(&path).unwrap(),
        LtxCheckpointLayout::AllInOne
    );
    assert_eq!(
        declared_layout(dir.path()).unwrap(),
        LtxCheckpointLayout::AllInOne
    );
    // A fine-tune that stamps no version at all keeps the historical path too.
    let bare = tempfile::tempdir().unwrap();
    write_component(&bare.path().join("10Eros_v1_bf16.safetensors"), &[]);
    assert_eq!(
        declared_layout(bare.path()).unwrap(),
        LtxCheckpointLayout::AllInOne
    );
}

// =================================================================================================
// LTX-2.5 split resolution.
// =================================================================================================

fn write_2_5_bundle(root: &Path) {
    write_component(
        &root.join("diffusion_models/ltx-2.5-22b-distilled-transformer-bf16.safetensors"),
        &[
            VERSION,
            (
                "gemma_source_checkpoint",
                r#"{"ltx_version":"2.5.0","gemma_version":"gemma4-12b-ltx-v1"}"#,
            ),
            (
                "config",
                r#"{
                    "transformer": {
                        "_class_name": "AVTransformer3DModel",
                        "num_layers": 44, "num_attention_heads": 24, "attention_head_dim": 128,
                        "cross_attention_dim": 3072,
                        "connector_num_attention_heads": 24, "connector_attention_head_dim": 128,
                        "connector_num_layers": 8, "connector_num_learnable_registers": 128,
                        "connector_positional_embedding_max_pos": [4096],
                        "audio_num_attention_heads": 32, "audio_attention_head_dim": 64,
                        "audio_cross_attention_dim": 2048,
                        "audio_connector_num_attention_heads": 32,
                        "audio_connector_attention_head_dim": 64,
                        "audio_positional_embedding_max_pos": [24],
                        "positional_embedding_max_pos": [24, 2048, 2048],
                        "caption_proj_before_connector": true,
                        "caption_projection_first_linear": false,
                        "caption_proj_input_norm": false,
                        "caption_projection_second_linear": false
                    },
                    "scheduler": {"_class_name": "RectifiedFlowScheduler"},
                    "vae": null, "audio_vae": null, "vocoder": null
                }"#,
            ),
        ],
    );
    write_component(
        &root.join("text_encoders/gemma4-12b-with-proj-ltx-2.5-bf16.safetensors"),
        &[
            // Ground truth (sc-18756): a packed TE declares no `model_version`.
            ("format", "pt"),
            (
                "gemma_config",
                r#"{"model_type":"gemma4_unified","gemma_version":"gemma4-12b-ltx-v1"}"#,
            ),
        ],
    );
    write_component(
        &root.join("vae/ltx-2.5-video-vae-conv-bf16.safetensors"),
        &[
            VERSION,
            (
                "config",
                r#"{"vae":{"_class_name":"CausalVideoAutoencoder","latent_channels":128,"patch_size":4}}"#,
            ),
        ],
    );
    write_component(
        &root.join("vae/ltx-2.5-video-vae-bf16.safetensors"),
        &[
            VERSION,
            (
                "config",
                r#"{"vae":{"_class_name":"CausalDiffusionVAE","latent_channels":128,
                    "decoder":{"patch_size":8,"head_dim":64}}}"#,
            ),
        ],
    );
    write_component(
        &root.join("vae/ltx-2.5-audio-vae-bf16.safetensors"),
        &[
            VERSION,
            (
                "config",
                &format!(
                    r#"{{"audio_vae":{{"model":{{"params":{{"ddconfig":{LTX_2_3_DDCONFIG},
                        "sampling_rate":16000}}}}}},"vocoder":{LTX_2_3_VOCODER}}}"#
                ),
            ),
        ],
    );
}

#[test]
fn every_2_5_component_reads_its_own_config_off_its_own_file() {
    let dir = tempfile::tempdir().unwrap();
    write_2_5_bundle(dir.path());
    let spec = LoadSpec::new(WeightsSource::Dir(dir.path().to_path_buf()));
    let bundle = resolve_split_bundle(&spec).expect("resolve the 2.5 bundle");
    assert_eq!(bundle.layout(), LtxCheckpointLayout::Split);

    // The transformer's own section — 2.5 dims, not the 2.3 constants.
    let av = AvConfig::from_bundle(&bundle).unwrap();
    assert_eq!(av.caption_feature_version, CaptionFeatureVersion::V2);
    assert_eq!(av.video.num_layers, 44);
    assert_eq!(av.video.num_heads, 24);
    assert_eq!(av.video.rope_max_pos, [24, 2048, 2048]);
    assert_eq!(av.audio_max_pos, 24);
    assert_eq!(av.cross_max_pos, 24);
    let conn = ConnectorConfig::from_bundle(&bundle).unwrap();
    assert_eq!(conn.num_heads, 24);
    let audio_conn = ConnectorConfig::audio_from_bundle(&bundle).unwrap();
    assert_eq!(audio_conn.head_dim, 64);

    // The two video VAEs are separate components with separate declarations — no silent pick.
    let conv = VideoVaeDeclaration::from_bundle(&bundle, LtxComponent::ConvVideoVae).unwrap();
    assert!(!conv.is_diffusion());
    assert_eq!(conv.patch_size, 4);
    let diff = VideoVaeDeclaration::from_bundle(&bundle, LtxComponent::DiffusionVideoVae).unwrap();
    assert!(diff.is_diffusion());
    assert_eq!(diff.patch_size, 8);

    // The audio VAE file owns BOTH its sections.
    assert_eq!(
        AudioVaeConfig::from_bundle(&bundle).unwrap(),
        AudioVaeConfig::ltx_2_3()
    );
    assert_eq!(
        VocoderConfig::from_bundle(&bundle)
            .unwrap()
            .final_sample_rate(),
        48000
    );

    assert!(matches!(
        assert_gemma_version(&bundle).unwrap(),
        GemmaVersionCheck::Matched(v) if v == "gemma4-12b-ltx-v1"
    ));
}

#[test]
fn the_2_5_transformers_null_vae_never_becomes_a_2_3_shaped_default() {
    let dir = tempfile::tempdir().unwrap();
    write_2_5_bundle(dir.path());
    // Delete the real conv VAE, leaving only the transformer whose `config.vae` is `null`.
    std::fs::remove_file(
        dir.path()
            .join("vae/ltx-2.5-video-vae-conv-bf16.safetensors"),
    )
    .unwrap();
    let spec = LoadSpec::new(WeightsSource::Dir(dir.path().to_path_buf()));
    let bundle = resolve_split_bundle(&spec).unwrap();
    let err = VideoVaeDeclaration::from_bundle(&bundle, LtxComponent::ConvVideoVae)
        .expect_err("no conv VAE component");
    let text = err.to_string();
    assert!(text.contains("conv_video_vae"), "{text}");
    assert!(text.contains("searched:"), "{text}");
}

#[test]
fn a_gemma_3_text_encoder_fails_a_2_5_bundle_with_a_version_mismatch() {
    let dir = tempfile::tempdir().unwrap();
    write_2_5_bundle(dir.path());
    std::fs::remove_file(
        dir.path()
            .join("text_encoders/gemma4-12b-with-proj-ltx-2.5-bf16.safetensors"),
    )
    .unwrap();
    let gemma3 = dir.path().join("gemma-3-12b-it");
    std::fs::create_dir_all(&gemma3).unwrap();
    std::fs::write(
        gemma3.join("config.json"),
        r#"{"model_type":"gemma3","text_config":{"hidden_size":3840}}"#,
    )
    .unwrap();
    let mut spec = LoadSpec::new(WeightsSource::Dir(dir.path().to_path_buf()));
    spec.text_encoder = Some(WeightsSource::Dir(gemma3));
    let bundle = resolve_split_bundle(&spec).unwrap();
    let err = assert_gemma_version(&bundle).expect_err("Gemma 3 cannot serve an LTX-2.5 bundle");
    let text = err.to_string();
    assert!(text.contains("Gemma version mismatch"), "{text}");
    assert!(text.contains("gemma4-12b-ltx-v1"), "{text}");
}

#[test]
fn the_ltx_2_3_engine_refuses_a_2_5_bundle_by_version() {
    let dir = tempfile::tempdir().unwrap();
    write_2_5_bundle(dir.path());
    let spec = LoadSpec::new(WeightsSource::Dir(dir.path().to_path_buf()));
    // `Box<dyn Generator>` is not `Debug`, so unwrap the Result by hand.
    let text = match candle_gen_ltx::load(&spec) {
        Ok(_) => panic!("ltx_2_3_distilled must not load a 2.5 bundle"),
        Err(e) => e.to_string(),
    };
    // Refused on the DECLARED VERSION — the name-keyed `ltx_checkpoint_in` picker is never reached.
    assert!(text.contains("2.5.0"), "{text}");
    assert!(text.contains("split-component bundle"), "{text}");
}
