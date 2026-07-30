//! Real-header and real-config proof for all eight frozen sc-14534 snapshots.
//!
//! The test takes explicit immutable snapshot paths and never scans or derives a cache location.

use std::path::PathBuf;

use candle_audio_stable_audio_3::candle_audio::candle_core::{DType, Device};
use candle_audio_stable_audio_3::config::{
    AutoencoderModuleType, DiffusionObjective, ModelConfig, StableAudioConfig,
};
use candle_audio_stable_audio_3::gen_core::WeightsSource;
use candle_audio_stable_audio_3::prepare;
use candle_audio_stable_audio_3::weights::{
    map_weight_key, safetensors_keys, KeyMapSummary, SnapshotKind, SnapshotLayout, WeightSection,
};
use core_llm::PrepareSpec;

struct Case {
    env: &'static str,
    kind: SnapshotKind,
    expected: KeyMapSummary,
    objective: Option<DiffusionObjective>,
}

const CASES: &[Case] = &[
    Case {
        env: "SA3_SMALL_MUSIC_SNAPSHOT",
        kind: SnapshotKind::Full,
        expected: KeyMapSummary {
            total: 685,
            encoder: 120,
            decoder: 120,
            bottleneck: 4,
            dit: 438,
            conditioner: 3,
        },
        objective: Some(DiffusionObjective::RfDenoiser),
    },
    Case {
        env: "SA3_SMALL_SFX_SNAPSHOT",
        kind: SnapshotKind::Full,
        expected: KeyMapSummary {
            total: 685,
            encoder: 120,
            decoder: 120,
            bottleneck: 4,
            dit: 438,
            conditioner: 3,
        },
        objective: Some(DiffusionObjective::RfDenoiser),
    },
    Case {
        env: "SA3_MEDIUM_SNAPSHOT",
        kind: SnapshotKind::Full,
        expected: KeyMapSummary {
            total: 997,
            encoder: 234,
            decoder: 234,
            bottleneck: 4,
            dit: 522,
            conditioner: 3,
        },
        objective: Some(DiffusionObjective::RfDenoiser),
    },
    Case {
        env: "SA3_SMALL_MUSIC_BASE_SNAPSHOT",
        kind: SnapshotKind::Full,
        expected: KeyMapSummary {
            total: 685,
            encoder: 120,
            decoder: 120,
            bottleneck: 4,
            dit: 438,
            conditioner: 3,
        },
        objective: Some(DiffusionObjective::RectifiedFlow),
    },
    Case {
        env: "SA3_SMALL_SFX_BASE_SNAPSHOT",
        kind: SnapshotKind::Full,
        expected: KeyMapSummary {
            total: 685,
            encoder: 120,
            decoder: 120,
            bottleneck: 4,
            dit: 438,
            conditioner: 3,
        },
        objective: Some(DiffusionObjective::RectifiedFlow),
    },
    Case {
        env: "SA3_MEDIUM_BASE_SNAPSHOT",
        kind: SnapshotKind::Full,
        expected: KeyMapSummary {
            total: 997,
            encoder: 234,
            decoder: 234,
            bottleneck: 4,
            dit: 522,
            conditioner: 3,
        },
        objective: Some(DiffusionObjective::RectifiedFlow),
    },
    Case {
        env: "SA3_SAME_S_SNAPSHOT",
        kind: SnapshotKind::StandaloneAutoencoder,
        expected: KeyMapSummary {
            total: 244,
            encoder: 120,
            decoder: 120,
            bottleneck: 4,
            dit: 0,
            conditioner: 0,
        },
        objective: None,
    },
    Case {
        env: "SA3_SAME_L_SNAPSHOT",
        kind: SnapshotKind::StandaloneAutoencoder,
        expected: KeyMapSummary {
            total: 472,
            encoder: 234,
            decoder: 234,
            bottleneck: 4,
            dit: 0,
            conditioner: 0,
        },
        objective: None,
    },
];

fn path(name: &str) -> PathBuf {
    std::env::var_os(name)
        .map(PathBuf::from)
        .unwrap_or_else(|| panic!("{name} must point to its pinned immutable snapshot"))
}

#[test]
#[ignore = "requires all eight explicitly provisioned sc-14534 snapshots"]
fn all_eight_configs_and_real_headers_match() {
    for case in CASES {
        let layout = SnapshotLayout::from_weights(&WeightsSource::Dir(path(case.env))).unwrap();
        assert_eq!(layout.kind, case.kind, "{}", case.env);
        assert_eq!(layout.keys, case.expected, "{}", case.env);
        assert_eq!(
            layout.tokenizer_model_path.is_some(),
            case.kind == SnapshotKind::Full,
            "{}",
            case.env
        );

        let encoded = serde_json::to_string(&layout.config).unwrap();
        let roundtrip: StableAudioConfig = serde_json::from_str(&encoded).unwrap();
        assert_eq!(roundtrip, layout.config, "{} config roundtrip", case.env);
        roundtrip.validate().unwrap();

        let ae = roundtrip.autoencoder();
        assert!(ae.encoder.config.dyt, "{} encoder must use DyT", case.env);
        assert!(ae.decoder.config.dyt, "{} decoder must use DyT", case.env);
        assert!(ae.encoder.config.differential, "{}", case.env);
        assert!(ae.decoder.config.differential, "{}", case.env);
        assert_eq!(ae.encoder.config.ff_mult, 3.0, "{}", case.env);
        assert_eq!(ae.decoder.config.ff_mult, 3.0, "{}", case.env);
        let expected_ae_type = if case.kind == SnapshotKind::Full {
            AutoencoderModuleType::TaaeV2
        } else {
            AutoencoderModuleType::Same
        };
        assert_eq!(ae.encoder.kind, expected_ae_type, "{}", case.env);
        assert_eq!(ae.decoder.kind, expected_ae_type, "{}", case.env);

        match (&roundtrip.model, case.objective) {
            (ModelConfig::Diffusion(model), Some(objective)) => {
                let dit = &model.diffusion.config;
                let medium = case.env.contains("MEDIUM");
                assert_eq!(
                    dit.embed_dim,
                    if medium { 1536 } else { 1024 },
                    "{}",
                    case.env
                );
                assert_eq!(dit.depth, if medium { 24 } else { 20 }, "{}", case.env);
                assert_eq!(dit.attn_kwargs.differential, medium, "{}", case.env);
                assert_eq!(dit.num_memory_tokens, 64, "{}", case.env);
                assert_eq!(
                    dit.norm_type,
                    candle_audio_stable_audio_3::config::NormType::RmsNorm
                );
                assert!(dit.norm_kwargs.force_fp32, "{}", case.env);
                assert_eq!(dit.norm_kwargs.eps, 1e-5, "{}", case.env);
                assert_eq!(dit.attn_kwargs.qk_norm_eps, 1e-6, "{}", case.env);
                assert_eq!(dit.ff_kwargs.mult, 4.0, "{}", case.env);
                assert_eq!(
                    model.diffusion.diffusion_objective, objective,
                    "{}",
                    case.env
                );
                assert_eq!(
                    model.diffusion.effective_sampling_shift().rate,
                    0.0,
                    "{} default inference LogSNR rate",
                    case.env
                );
            }
            (ModelConfig::Autoencoder(model), None) => {
                let large = case.env == "SA3_SAME_L_SNAPSHOT";
                let expected_dim = if large { 1536 } else { 768 };
                let expected_depth = if large { 12 } else { 6 };
                assert_eq!(
                    model.encoder.config.channels * model.encoder.config.c_mults[0],
                    expected_dim,
                    "{}",
                    case.env
                );
                assert_eq!(
                    model.encoder.config.transformer_depths,
                    vec![expected_depth],
                    "{}",
                    case.env
                );
                assert_eq!(
                    model.decoder.config.transformer_depths,
                    vec![expected_depth],
                    "{}",
                    case.env
                );
                assert_eq!(
                    model.decoder.config.sinusoidal_blocks,
                    vec![if large { 8 } else { 0 }],
                    "{}",
                    case.env
                );
            }
            _ => panic!("{} config family/objective mismatch", case.env),
        }

        // Exercise the actual Candle mmap loader and every real mapped namespace. This reads
        // headers/maps files but does not materialize multi-gigabyte tensors.
        let builders = layout.mmap_builders(DType::F32, &Device::Cpu).unwrap();
        let keys = safetensors_keys(&layout.weights_path).unwrap();
        for section in [
            WeightSection::Encoder,
            WeightSection::Decoder,
            WeightSection::Bottleneck,
            WeightSection::Dit,
            WeightSection::Conditioner,
        ] {
            if case.kind == SnapshotKind::StandaloneAutoencoder
                && matches!(section, WeightSection::Dit | WeightSection::Conditioner)
            {
                continue;
            }
            let mapped = keys
                .iter()
                .filter_map(|key| map_weight_key(case.kind, key))
                .find(|mapped| mapped.section == section)
                .unwrap();
            let present = match section {
                WeightSection::Encoder => builders.encoder.contains_tensor(mapped.local_key),
                WeightSection::Decoder => builders.decoder.contains_tensor(mapped.local_key),
                WeightSection::Bottleneck => builders.bottleneck.contains_tensor(mapped.local_key),
                WeightSection::Dit => builders
                    .dit
                    .as_ref()
                    .unwrap()
                    .contains_tensor(mapped.local_key),
                WeightSection::Conditioner => builders
                    .conditioner
                    .as_ref()
                    .unwrap()
                    .contains_tensor(mapped.local_key),
            };
            assert!(present, "{} failed to load {section:?}", case.env);
        }
        if let Some(text_weights) = &layout.text_weights_path {
            let text_keys = safetensors_keys(text_weights).unwrap();
            assert!(
                builders
                    .text_encoder
                    .as_ref()
                    .unwrap()
                    .contains_tensor(&text_keys[0]),
                "{} failed to load bundled T5Gemma weights",
                case.env
            );
        }

        let report = prepare::prepare(&PrepareSpec::dense(
            &layout.root,
            layout.root.join("unused"),
        ))
        .unwrap();
        assert!(report.passthrough, "{}", case.env);
        assert_eq!(report.out_dir, layout.root, "{}", case.env);
        assert!(report.num_tensors >= layout.keys.total, "{}", case.env);
    }
}

#[test]
#[ignore = "requires the pinned small-music snapshot"]
fn shipped_dit_config_fails_closed_for_every_unsupported_branch() {
    let config_path = path("SA3_SMALL_MUSIC_SNAPSHOT").join("model_config.json");
    let original: serde_json::Value =
        serde_json::from_str(&std::fs::read_to_string(config_path).unwrap()).unwrap();
    let mutations: &[(&str, &str, serde_json::Value)] = &[
        (
            "logSNR timestep",
            "/model/diffusion/config/timestep_features_logsnr",
            true.into(),
        ),
        (
            "prepend global conditioning",
            "/model/diffusion/config/global_cond_type",
            "prepend".into(),
        ),
        (
            "mm transformer",
            "/model/diffusion/config/transformer_type",
            "mm_transformer".into(),
        ),
        (
            "input concat",
            "/model/diffusion/config/input_concat_dim",
            1.into(),
        ),
        (
            "prepend concat",
            "/model/diffusion/config/prepend_cond_dim",
            1.into(),
        ),
        (
            "unprojected prompt",
            "/model/diffusion/config/project_cond_tokens",
            false.into(),
        ),
        (
            "feature scaling",
            "/model/diffusion/config/attn_kwargs/feat_scale",
            true.into(),
        ),
        (
            "conformer",
            "/model/diffusion/config/conformer",
            true.into(),
        ),
        (
            "FF convolution",
            "/model/diffusion/config/ff_kwargs/use_conv",
            true.into(),
        ),
        (
            "absolute position",
            "/model/diffusion/config/use_abs_pos_emb",
            true.into(),
        ),
        (
            "cross RoPE",
            "/model/diffusion/config/cross_attn_rotary_pos_emb",
            true.into(),
        ),
        (
            "sliding attention",
            "/model/diffusion/config/sliding_window",
            serde_json::json!([1, 1]),
        ),
        (
            "layer scale",
            "/model/diffusion/config/layer_scale",
            true.into(),
        ),
        (
            "partial cross attention",
            "/model/diffusion/config/final_cross_attn_ix",
            0.into(),
        ),
    ];
    for (label, pointer, replacement) in mutations {
        let mut value = original.clone();
        let object = pointer.rsplit_once('/').unwrap();
        let parent = value
            .pointer_mut(object.0)
            .unwrap()
            .as_object_mut()
            .unwrap();
        parent.insert(object.1.into(), replacement.clone());
        let parsed: StableAudioConfig = serde_json::from_value(value).unwrap();
        assert!(parsed.validate().is_err(), "{label} must fail closed");
    }

    let mut modular = original;
    modular["model"]["diffusion"]["modular_local_cond_configs"] =
        serde_json::json!([{"id":"future","dim":1}]);
    let parsed: StableAudioConfig = serde_json::from_value(modular).unwrap();
    assert!(
        parsed.validate().is_err(),
        "modular local conditioning must fail closed"
    );
}
