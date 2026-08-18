//! sc-18757 — split-checkpoint loading for LTX-2.5 components.
//!
//! Two halves:
//!
//! * **LTX-2.3 regression.** `tests/fixtures/ltx_2_3_embedded_config.json` and
//!   `ltx_2_3_split_model.json` are verbatim copies of the shipped `SceneWorks/ltx-2.3-mlx` q4
//!   snapshot. Every config the 2.3 loader reads is asserted against the values the current pin
//!   produces, so the refactor that introduced the split-bundle readers cannot move the 2.3 path.
//!   The 2.3 tree must also still select the all-in-one layout, so `load` takes exactly the branch
//!   it always took.
//!
//! * **LTX-2.5 split resolution.** Bundles laid out like the shipped `Lightricks/LTX-2.5` repo
//!   (`diffusion_models/`, `text_encoders/`, `vae/`, `model_patches/`, `latent_upscale_models/`)
//!   exercise per-component resolution, per-component config isolation, the
//!   `gemma_source_checkpoint` assertion, and the missing-component error.
//!
//!   The 2.5 **weights** are gated on Hugging Face, so these fixtures are written on disk rather
//!   than downloaded — but their `__metadata__` shape is not invented: it matches the real headers
//!   sc-18756 captured under `docs/reference/sc-18756-headers/`, which
//!   `gen_core::ltx_checkpoint`'s own tests parse directly. In particular the latent upsamplers and
//!   the packed text encoder declare **no** `model_version`, exactly as the shipped files do.
//!   Upstream reference: `ltx_core/loader/sft_loader.py`, the per-component
//!   `model_configurator.py` files, and `encoder_configurator._check_gemma_version` at
//!   `Lightricks/LTX-2` @ `d1511477`.

use std::path::Path;

use mlx_gen::gen_core::ltx_checkpoint::{
    GemmaVersionCheck, LtxCheckpointLayout, LtxComponent, SPLIT_MANIFEST_FILE,
};
use mlx_gen::{LoadSpec, WeightsSource};
use mlx_gen_ltx::config::{AudioVaeConfig, LtxConfig, LtxVaeConfig, RopeType, VocoderConfig};
use mlx_gen_ltx::{
    assert_gemma_version, declared_layout, declared_model_version, resolve_split_bundle,
};

// =================================================================================================
// LTX-2.3 regression — the shipped q4 snapshot's own config files.
// =================================================================================================

fn fixture(name: &str) -> String {
    let path = Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("tests/fixtures")
        .join(name);
    std::fs::read_to_string(&path).unwrap_or_else(|e| panic!("read {}: {e}", path.display()))
}

/// Stage the shipped 2.3 snapshot's config + manifest into a temp dir. The weights themselves are
/// tens of gigabytes and irrelevant here: every assertion below is a config read.
fn staged_2_3_tree() -> tempfile::TempDir {
    let dir = tempfile::tempdir().expect("tempdir");
    std::fs::write(
        dir.path().join("embedded_config.json"),
        fixture("ltx_2_3_embedded_config.json"),
    )
    .unwrap();
    std::fs::write(
        dir.path().join(SPLIT_MANIFEST_FILE),
        fixture("ltx_2_3_split_model.json"),
    )
    .unwrap();
    dir
}

#[test]
fn ltx_2_3_transformer_config_is_unchanged() {
    let dir = staged_2_3_tree();
    let cfg = LtxConfig::from_model_dir(dir.path()).expect("2.3 transformer config");
    // Video stack.
    assert_eq!(cfg.num_attention_heads, 32);
    assert_eq!(cfg.attention_head_dim, 128);
    assert_eq!(cfg.inner_dim(), 4096);
    assert_eq!(cfg.num_layers, 48);
    assert_eq!(cfg.in_channels, 128);
    assert_eq!(cfg.out_channels, 128);
    assert_eq!(cfg.cross_attention_dim, 4096);
    assert!((cfg.norm_eps - 1e-6).abs() < f64::EPSILON);
    // Gated family → adaLN coefficient 9, cross-attention adaLN on.
    assert!(cfg.apply_gated_attention);
    assert_eq!(cfg.adaln_embedding_coefficient, 9);
    assert!(cfg.cross_attention_adaln);
    // No caption projection → caption_channels = connector heads × head_dim.
    assert!(!cfg.caption_projection_first_linear);
    assert!(!cfg.caption_projection_second_linear);
    assert_eq!(cfg.caption_channels, 4096);
    assert_eq!(cfg.audio_caption_channels, 2048);
    // RoPE.
    assert_eq!(cfg.rope_type, RopeType::Split);
    assert!(cfg.double_precision_rope);
    assert!((cfg.positional_embedding_theta - 10000.0).abs() < f64::EPSILON);
    assert_eq!(cfg.positional_embedding_max_pos, [20, 2048, 2048]);
    assert!(cfg.use_middle_indices_grid);
    assert_eq!(cfg.timestep_scale_multiplier, 1000);
    // Connector.
    assert!(cfg.use_embeddings_connector);
    assert_eq!(cfg.connector_num_layers, 8);
    assert_eq!(cfg.connector_num_attention_heads, 32);
    assert_eq!(cfg.connector_attention_head_dim, 128);
    assert_eq!(cfg.connector_num_learnable_registers, 128);
    assert_eq!(cfg.connector_positional_embedding_max_pos, 4096);
    assert!(cfg.connector_apply_gated_attention);
    // Audio stack.
    assert_eq!(cfg.audio_num_attention_heads, 32);
    assert_eq!(cfg.audio_attention_head_dim, 64);
    assert_eq!(cfg.audio_inner_dim(), 2048);
    assert_eq!(cfg.audio_cross_attention_dim, 2048);
    assert_eq!(cfg.audio_in_channels, 128);
    assert_eq!(cfg.audio_out_channels, 128);
    assert_eq!(cfg.audio_positional_embedding_max_pos, 20);
    assert_eq!(cfg.cross_pe_max_pos(), 20);
    assert_eq!(cfg.av_ca_timestep_scale_multiplier, 1000);
}

#[test]
fn ltx_2_3_video_vae_config_is_unchanged() {
    let dir = staged_2_3_tree();
    let cfg = LtxVaeConfig::from_model_dir(dir.path()).expect("2.3 vae config");
    assert_eq!(cfg.latent_channels, 128);
    assert_eq!(cfg.patch_size, 4);
    assert!(!cfg.timestep_conditioning);
    assert_eq!(cfg.spatial_padding_mode, "zeros");
    let kinds: Vec<&str> = cfg.decoder_blocks.iter().map(|b| b.kind.as_str()).collect();
    assert_eq!(
        kinds,
        [
            "res_x",
            "compress_space",
            "res_x",
            "compress_time",
            "res_x",
            "compress_all",
            "res_x",
            "compress_all",
            "res_x"
        ]
    );
    let encoder_kinds: Vec<&str> = cfg.encoder_blocks.iter().map(|b| b.kind.as_str()).collect();
    assert_eq!(
        encoder_kinds,
        [
            "res_x",
            "compress_space_res",
            "res_x",
            "compress_time_res",
            "res_x",
            "compress_all_res",
            "res_x",
            "compress_all_res",
            "res_x"
        ]
    );
    // Total compression: ×32 spatial, ×8 temporal.
    let (t, h, w) =
        cfg.decoder_blocks
            .iter()
            .filter(|b| b.is_compress())
            .fold((1, 1, 1), |acc, b| {
                let s = b.stride();
                (acc.0 * s.0, acc.1 * s.1, acc.2 * s.2)
            });
    assert_eq!((t, h * cfg.patch_size, w * cfg.patch_size), (8, 32, 32));
}

#[test]
fn ltx_2_3_audio_vae_and_vocoder_configs_are_unchanged() {
    let dir = staged_2_3_tree();
    let audio = AudioVaeConfig::from_model_dir(dir.path()).expect("2.3 audio vae config");
    assert_eq!(audio.ch, 128);
    assert_eq!(audio.out_ch, 2);
    assert_eq!(audio.ch_mult, vec![1, 2, 4]);
    assert_eq!(audio.num_resolutions(), 3);
    assert_eq!(audio.num_res_blocks, 2);
    assert_eq!(audio.z_channels, 8);
    assert_eq!(audio.mel_bins, 64);
    // The shipped checkpoint ships NO `mid.attn_1` weights; honoring the config skips it.
    assert!(!audio.mid_block_add_attention);

    let vocoder = VocoderConfig::from_model_dir(dir.path()).expect("2.3 vocoder config");
    assert!(vocoder.core.is_bigvgan());
    assert_eq!(vocoder.core.upsample_rates, vec![5, 2, 2, 2, 2, 2]);
    assert_eq!(vocoder.core.upsample_kernel_sizes, vec![11, 4, 4, 4, 4, 4]);
    assert_eq!(vocoder.core.upsample_initial_channel, 1536);
    assert!(!vocoder.core.use_tanh_at_final);
    assert!(!vocoder.core.use_bias_at_final);
    let bwe = vocoder.bwe.as_ref().expect("2.3 ships the BWE stage");
    assert!(bwe.is_bigvgan());
    assert_eq!(bwe.upsample_rates, vec![6, 5, 2, 2, 2]);
    assert_eq!(bwe.upsample_kernel_sizes, vec![12, 11, 4, 4, 4]);
    assert!(!bwe.apply_final_activation);
    assert_eq!(vocoder.bwe_input_sample_rate, 16000);
    assert_eq!(vocoder.bwe_output_sample_rate, 48000);
    assert_eq!(vocoder.bwe_hop_length, 80);
    assert_eq!(vocoder.bwe_win_length, 512);
    assert_eq!(vocoder.final_sample_rate(), 48000);
}

#[test]
fn the_shipped_2_3_snapshot_selects_the_all_in_one_layout() {
    let dir = staged_2_3_tree();
    assert_eq!(
        declared_model_version(dir.path()).unwrap().as_deref(),
        Some("2.3.0")
    );
    assert_eq!(
        declared_layout(dir.path()).unwrap(),
        LtxCheckpointLayout::AllInOne
    );
}

// =================================================================================================
// LTX-2.5 split resolution.
// =================================================================================================

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

/// The documented `Lightricks/LTX-2.5` folder layout, with each component carrying only its own
/// config section and the transformer explicitly nulling the sections it no longer owns.
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
                        "cross_attention_dim": 3072, "in_channels": 128, "out_channels": 128,
                        "apply_gated_attention": true, "cross_attention_adaln": true,
                        "caption_projection_first_linear": false,
                        "caption_projection_second_linear": false,
                        "use_embeddings_connector": true,
                        "connector_num_attention_heads": 24, "connector_attention_head_dim": 128,
                        "connector_num_layers": 8, "connector_num_learnable_registers": 128,
                        "connector_positional_embedding_max_pos": [4096],
                        "audio_num_attention_heads": 32, "audio_attention_head_dim": 64,
                        "audio_connector_num_attention_heads": 32,
                        "audio_connector_attention_head_dim": 64,
                        "rope_type": "split", "frequencies_precision": "float64",
                        "positional_embedding_max_pos": [24, 2048, 2048]
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
                r#"{"vae":{"_class_name":"CausalVideoAutoencoder","latent_channels":128,
                    "patch_size":4,"timestep_conditioning":false,"spatial_padding_mode":"zeros",
                    "decoder_blocks":[["res_x",{"num_layers":4}],["compress_space",{"multiplier":2}],
                        ["res_x",{"num_layers":6}],["compress_time",{"multiplier":2}],
                        ["res_x",{"num_layers":4}],["compress_all",{"multiplier":1}],
                        ["res_x",{"num_layers":2}],["compress_all",{"multiplier":2}],
                        ["res_x",{"num_layers":2}]]}}"#,
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
                    "decoder":{"patch_size":4,"head_dim":64,"stage_channels":[1024,768,512,256,128]}}}"#,
            ),
        ],
    );
    write_component(
        &root.join("vae/ltx-2.5-audio-vae-bf16.safetensors"),
        &[
            VERSION,
            (
                "config",
                r#"{"audio_vae":{"model":{"params":{"ddconfig":{"ch":128,"out_ch":2,
                        "ch_mult":[1,2,4],"num_res_blocks":2,"z_channels":8,"mel_bins":64,
                        "mid_block_add_attention":false},"sampling_rate":16000}}},
                    "vocoder":{"vocoder":{"resblock":"AMP1","activation":"snakebeta",
                        "upsample_rates":[5,2,2,2,2,2],"upsample_initial_channel":1536,
                        "use_tanh_at_final":false,"use_bias_at_final":false},
                        "bwe":{"resblock":"AMP1","activation":"snakebeta",
                        "upsample_rates":[6,5,2,2,2],"input_sampling_rate":16000,
                        "output_sampling_rate":48000,"hop_length":80,"win_size":512}}}"#,
            ),
        ],
    );
    write_component(
        &root.join("model_patches/ltx-2.5-duration-head-bf16.safetensors"),
        &[
            VERSION,
            (
                "config",
                r#"{"transformer":{"cross_attention_dim":3072,"audio_cross_attention_dim":2048},
                    "duration_head":{"pooler_hidden_dim":256,"num_queries":1,"num_pooler_heads":4,
                    "mlp_hidden":256}}"#,
            ),
        ],
    );
    write_component(
        &root.join("latent_upscale_models/ltx-2.5-latent-spatial-upscaler-x2-bf16-1.0.safetensors"),
        &[
            // Ground truth (sc-18756): the latent upsamplers declare no `model_version`.
            (
                "config",
                r#"{"_class_name":"LatentUpsampler","in_channels":128,"mid_channels":512,
                    "num_blocks_per_stage":4,"dims":3,"spatial_upsample":true,
                    "temporal_upsample":false,"spatial_scale":2.0}"#,
            ),
        ],
    );
    write_component(
        &root
            .join("latent_upscale_models/ltx-2.5-latent-temporal-upscaler-x2-bf16-1.0.safetensors"),
        &[
            // Ground truth (sc-18756): the latent upsamplers declare no `model_version`.
            (
                "config",
                r#"{"_class_name":"LatentUpsampler","in_channels":128,"mid_channels":512,
                    "num_blocks_per_stage":4,"dims":3,"spatial_upsample":false,
                    "temporal_upsample":true}"#,
            ),
        ],
    );
}

#[test]
fn every_2_5_component_resolves_independently_and_reads_its_own_config() {
    let dir = tempfile::tempdir().unwrap();
    write_2_5_bundle(dir.path());
    let spec = LoadSpec::new(WeightsSource::Dir(dir.path().to_path_buf()));
    let bundle = resolve_split_bundle(&spec).expect("resolve the 2.5 bundle");
    assert_eq!(bundle.layout(), LtxCheckpointLayout::Split);
    assert_eq!(bundle.model_version(), Some("2.5.0"));
    for component in LtxComponent::ALL {
        bundle
            .require(*component)
            .unwrap_or_else(|e| panic!("{}: {e}", component.id()));
    }

    // The transformer's own section — 2.5 dims, not the 2.3 constants.
    let transformer = LtxConfig::from_bundle(&bundle).expect("transformer config");
    assert_eq!(transformer.num_layers, 44);
    assert_eq!(transformer.num_attention_heads, 24);
    assert_eq!(transformer.inner_dim(), 24 * 128);
    assert_eq!(transformer.cross_attention_dim, 3072);
    assert_eq!(transformer.positional_embedding_max_pos, [24, 2048, 2048]);
    assert_eq!(transformer.caption_channels, 24 * 128);

    // The conv VAE's own section, off its own file.
    let vae = LtxVaeConfig::from_bundle(&bundle, LtxComponent::ConvVideoVae).expect("vae config");
    assert_eq!(vae.latent_channels, 128);
    assert_eq!(vae.patch_size, 4);
    assert_eq!(vae.decoder_blocks.len(), 9);

    // The audio VAE file owns BOTH its sections.
    let audio = AudioVaeConfig::from_bundle(&bundle).expect("audio vae config");
    assert_eq!(audio.ch, 128);
    assert_eq!(audio.z_channels, 8);
    assert!(!audio.mid_block_add_attention);
    let vocoder = VocoderConfig::from_bundle(&bundle).expect("vocoder config");
    assert!(vocoder.core.is_bigvgan());
    assert_eq!(vocoder.final_sample_rate(), 48000);

    // The upsamplers' bare configs, distinguished by their own axis flags.
    let spatial = bundle
        .component_config(LtxComponent::SpatialUpsampler)
        .unwrap();
    assert_eq!(spatial["spatial_upsample"], true);
    let temporal = bundle
        .component_config(LtxComponent::TemporalUpsampler)
        .unwrap();
    assert_eq!(temporal["temporal_upsample"], true);

    // The duration head reads its own hyperparameters AND the transformer dims it projects from.
    let head = bundle.require(LtxComponent::DurationHead).unwrap();
    assert_eq!(head.config().unwrap()["num_queries"], 1);
    assert_eq!(
        head.config_section("transformer").unwrap()["cross_attention_dim"],
        3072
    );
}

#[test]
fn the_2_5_transformer_cannot_satisfy_the_vae_slot() {
    // `config.vae` is explicitly `null` on the 2.5 transformer. Pointing the VAE slot at it must
    // ERROR — silently falling back to the 2.3 block ladder would build the wrong decoder.
    let dir = tempfile::tempdir().unwrap();
    write_2_5_bundle(dir.path());
    let transformer = dir
        .path()
        .join("diffusion_models/ltx-2.5-22b-distilled-transformer-bf16.safetensors");
    let err = mlx_gen::gen_core::ltx_checkpoint::LtxBundleBuilder::new()
        .with_component(LtxComponent::ConvVideoVae, transformer)
        .build()
        .expect_err("the transformer is not the video VAE");
    // Caught at the slot level: the file declares itself the transformer.
    assert!(
        err.to_string()
            .contains("provisioned as the `conv_video_vae`"),
        "{err}"
    );
}

#[test]
fn a_gemma_3_text_encoder_fails_a_2_5_bundle_with_a_version_mismatch() {
    let dir = tempfile::tempdir().unwrap();
    write_2_5_bundle(dir.path());
    // Remove the bundled Gemma 4 encoder and point the typed slot at an LTX-2.3 Gemma-3 snapshot.
    std::fs::remove_file(
        dir.path()
            .join("text_encoders/gemma4-12b-with-proj-ltx-2.5-bf16.safetensors"),
    )
    .unwrap();
    let gemma3 = dir.path().join("gemma-3-12b-it");
    std::fs::create_dir_all(&gemma3).unwrap();
    std::fs::write(
        gemma3.join("config.json"),
        r#"{"model_type":"gemma3","text_config":{"hidden_size":3840,"num_hidden_layers":48}}"#,
    )
    .unwrap();

    let mut spec = LoadSpec::new(WeightsSource::Dir(dir.path().to_path_buf()));
    spec.text_encoder = Some(WeightsSource::Dir(gemma3));
    let bundle = resolve_split_bundle(&spec).unwrap();
    let err = assert_gemma_version(&bundle).expect_err("Gemma 3 cannot serve an LTX-2.5 bundle");
    let text = err.to_string();
    assert!(text.contains("Gemma version mismatch"), "{text}");
    assert!(text.contains("gemma4-12b-ltx-v1"), "{text}");
    assert!(text.contains("gemma-3-12b-it"), "{text}");
}

#[test]
fn the_bundled_gemma_4_encoder_satisfies_the_assertion() {
    let dir = tempfile::tempdir().unwrap();
    write_2_5_bundle(dir.path());
    let spec = LoadSpec::new(WeightsSource::Dir(dir.path().to_path_buf()));
    let bundle = resolve_split_bundle(&spec).unwrap();
    assert!(matches!(
        assert_gemma_version(&bundle).unwrap(),
        GemmaVersionCheck::Matched(v) if v == "gemma4-12b-ltx-v1"
    ));
}

#[test]
fn a_missing_component_names_itself_and_the_paths_searched() {
    let dir = tempfile::tempdir().unwrap();
    write_2_5_bundle(dir.path());
    std::fs::remove_file(dir.path().join("vae/ltx-2.5-audio-vae-bf16.safetensors")).unwrap();
    let spec = LoadSpec::new(WeightsSource::Dir(dir.path().to_path_buf()));
    let bundle = resolve_split_bundle(&spec).unwrap();
    // The bundle is STILL a 2.5 bundle — layout is keyed on `model_version`, not on what is present.
    assert_eq!(bundle.layout(), LtxCheckpointLayout::Split);
    let err = AudioVaeConfig::from_bundle(&bundle).expect_err("the audio VAE is gone");
    let text = err.to_string();
    assert!(text.contains("audio_vae"), "{text}");
    assert!(text.contains("the audio VAE + vocoder"), "{text}");
    assert!(text.contains("searched:"), "{text}");
    assert!(
        text.contains("ltx-2.5-22b-distilled-transformer-bf16.safetensors"),
        "{text}"
    );
    // The vocoder rides the same component, so it reports the same missing component.
    assert!(VocoderConfig::from_bundle(&bundle)
        .unwrap_err()
        .to_string()
        .contains("audio_vae"));
}

#[test]
fn the_ltx_2_3_engine_refuses_a_2_5_bundle_by_version() {
    let dir = tempfile::tempdir().unwrap();
    write_2_5_bundle(dir.path());
    let spec = LoadSpec::new(WeightsSource::Dir(dir.path().to_path_buf()));
    // `Box<dyn Generator>` is not `Debug`, so unwrap the Result by hand.
    let text = match mlx_gen_ltx::load(&spec) {
        Ok(_) => panic!("ltx_2_3 must not load a 2.5 bundle"),
        Err(e) => e.to_string(),
    };
    // Refused on the DECLARED VERSION, not on a missing `transformer.safetensors` file name.
    assert!(text.contains("2.5.0"), "{text}");
    assert!(text.contains("split-component bundle"), "{text}");
    assert!(!text.contains("missing transformer.safetensors"), "{text}");
}
