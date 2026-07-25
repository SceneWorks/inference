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
