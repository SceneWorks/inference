//! Cross-backend training artifacts (sc-25686): an Iris artifact trained on one backend loads — and
//! means the same thing — on the other.
//!
//! The committed fixtures `mlx-gen-iris/tests/fixtures/cross-backend/<backend>/` each hold one
//! miniature LoRA, one LoKr and one full-model export produced by that backend's trainer
//! ([`produce_cross_backend_fixtures`] here, its twin in `mlx-gen-iris`), plus `forward.safetensors`:
//! the producing backend's own FP32 forward of each artifact on the `iris_dit_golden` inputs,
//! loaded through its own inference route. The consuming backend loads the artifact through
//! **its** inference route (identity stamps checked) and must reproduce that forward to the
//! cross-backend FP32 bound ([`tolerance`]), and the adapter must visibly move the
//! output — so a file that "loads" but is read differently (a transposed factor, a lost alpha, a
//! dropped target) fails.

use std::collections::HashMap;
use std::path::{Path, PathBuf};

use candle_gen::candle_core::{DType, Tensor};
use candle_gen::gen_core::iris::train::{AdapterMetadata, OPTIONS_KEY};
use candle_gen::gen_core::iris::{IrisConfig, IrisTask, TEXT_ENCODER_COMPONENT};
use candle_gen::gen_core::{
    AdapterKind, AdapterSpec, LoadSpec, NetworkType, TrainingConfig, TrainingItem, TrainingRequest,
    WeightsSource,
};
use candle_gen_iris::{load_backbone, load_backbone_with_adapters, IrisDiT, TextBatch};
use serde_json::json;

use crate::common::{
    assert_close, cpu, errors, fixture, fixture_at, fixtures, host_i32, tiny_backbone, tiny_config,
    tiny_text_encoder,
};

const ID: &str = "iris_3b";
const ARTIFACTS: [(&str, Option<AdapterKind>); 3] = [
    ("lora", Some(AdapterKind::Lora)),
    ("lokr", Some(AdapterKind::Lokr)),
    ("full", None),
];

/// The cross-backend bound: FP32 summation order only (1e-4 of peak, `dit_parity`'s) — except a
/// LoKr, whose `[out, in]` delta the MLX provider reconstructs in **bf16** whatever the compute
/// dtype (`mlx_gen::adapters::loader::apply_lokr`, PARITY-BF16 sc-2609) while the Candle provider
/// holds it in the compute dtype; at this FP32 forward that bf16 rounding of the delta is the whole
/// difference (measured ~1.2e-3 of peak; at the release's bf16 compute both hold a bf16 delta).
fn tolerance(name: &str) -> f32 {
    if name == "lokr" {
        5e-3
    } else {
        1e-4
    }
}

fn artifact_path(dir: &Path, name: &str) -> PathBuf {
    if name == "full" {
        dir.join("full")
    } else {
        dir.join(format!("{name}.safetensors"))
    }
}

/// Load one artifact of `dir` through the Candle inference route at FP32 on the CPU.
fn load(dir: &Path, name: &str, kind: Option<AdapterKind>) -> IrisDiT {
    let path = artifact_path(dir, name);
    match kind {
        Some(kind) => {
            let (dit, reports) = load_backbone_with_adapters(
                &tiny_backbone(),
                &tiny_config(),
                DType::F32,
                &cpu(),
                &[AdapterSpec::new(path, 1.0, kind)],
                IrisTask::Generation,
                ID,
            )
            .unwrap();
            assert_eq!(reports.len(), 1);
            dit
        }
        None => {
            let cfg = IrisConfig::from_dir(&path).unwrap();
            load_backbone(&path, &cfg, DType::F32, &cpu()).unwrap()
        }
    }
}

/// The FP32 forward on the `iris_dit_golden` inputs.
fn forward(dit: &IrisDiT) -> Tensor {
    let golden = fixture("iris_dit_golden.safetensors");
    let flat = host_i32(golden.require("y_mask"));
    let mask: Vec<Vec<i32>> = flat.chunks(flat.len() / 2).map(<[i32]>::to_vec).collect();
    dit.forward(
        golden.require("x"),
        golden.require("t"),
        &TextBatch {
            states: golden.require("y"),
            mask: &mask,
        },
    )
    .unwrap()
}

fn check_trained_elsewhere(backend: &str) {
    let dir = fixtures().join("cross-backend").join(backend);
    let want = fixture_at(&dir.join("forward.safetensors"));
    assert_eq!(want.meta("backend"), backend);
    let bare =
        forward(&load_backbone(&tiny_backbone(), &tiny_config(), DType::F32, &cpu()).unwrap());
    for (name, kind) in ARTIFACTS {
        let tensor_file = match kind {
            Some(_) => artifact_path(&dir, name),
            None => artifact_path(&dir, name).join("model.safetensors"),
        };
        let meta = fixture_at(&tensor_file).metadata;
        assert_eq!(meta["irisTask"], "generation", "{backend}/{name}");
        match kind {
            Some(_) => {
                assert_eq!(meta["family"], "iris", "{backend}/{name}");
                let m = AdapterMetadata::from_map(&meta.into_iter().collect()).unwrap();
                assert_eq!(m.network_type, name);
            }
            None => assert_eq!(meta["irisArtifact"], "full_model"),
        }
        let got = forward(&load(&dir, name, kind));
        let expected = want.require(name);
        assert_close(
            &format!("{backend}-trained {name} on candle"),
            &got,
            expected,
            tolerance(name),
        );
        let (moved, peak, _) = errors(expected, &bare);
        eprintln!("{backend}/{name}: the artifact moves the forward by {moved:.3e}");
        assert!(
            moved > 10.0 * tolerance(name) * peak.max(1.0),
            "{backend}/{name}: the trained artifact barely moves the forward ({moved:.3e})"
        );
    }
}

#[test]
fn mlx_trained_artifacts_load_and_match_on_candle() {
    check_trained_elsewhere("mlx");
}

/// The Candle-produced fixtures still load here (and their recorded forward is this backend's).
#[test]
fn candle_trained_fixtures_match_their_recorded_forward() {
    check_trained_elsewhere("candle");
}

/// Producer of `fixtures/cross-backend/candle/` (run once on purpose, never in CI):
/// `IRIS_CROSS_BACKEND_OUT=<dir> cargo test -p candle-gen-iris --test integration --release \
/// train_cross_backend::produce_cross_backend_fixtures -- --ignored`, then copy `<dir>` over the
/// committed fixture directory.
#[test]
#[ignore = "fixture producer: writes into IRIS_CROSS_BACKEND_OUT"]
fn produce_cross_backend_fixtures() {
    let out =
        PathBuf::from(std::env::var("IRIS_CROSS_BACKEND_OUT").expect("IRIS_CROSS_BACKEND_OUT"));
    std::fs::create_dir_all(&out).unwrap();
    let data = tempfile::tempdir().unwrap();
    let items: Vec<TrainingItem> = ["a red fox in the snow", "golden hour"]
        .iter()
        .enumerate()
        .map(|(i, caption)| {
            let img = image::RgbImage::from_fn(20, 16, |x, y| {
                image::Rgb([(x * 11 + i as u32 * 60) as u8, (y * 13) as u8, 90])
            });
            let path = data.path().join(format!("img{i}.png"));
            img.save(&path).unwrap();
            TrainingItem::captioned(path, caption.to_string())
        })
        .collect();
    let mut spec = LoadSpec::new(WeightsSource::Dir(tiny_backbone()));
    spec.components.insert(
        TEXT_ENCODER_COMPONENT.into(),
        WeightsSource::Dir(tiny_text_encoder()),
    );
    let registry = candle_gen_iris::provider_registry().unwrap();
    let mut forwards: HashMap<String, Tensor> = HashMap::new();
    for (name, kind) in ARTIFACTS {
        let mut cfg = TrainingConfig {
            rank: 2,
            alpha: 4.0,
            learning_rate: 5e-2,
            steps: 2,
            batch_size: 2,
            resolution: 16,
            seed: 11,
            train_dtype: "f32".into(),
            ..Default::default()
        };
        cfg.network_type = match kind {
            Some(AdapterKind::Lokr) => NetworkType::Lokr,
            _ => NetworkType::Lora,
        };
        cfg.full_finetune = kind.is_none();
        cfg.model_options.insert(
            OPTIONS_KEY.into(),
            json!({"ema_enabled": false, "export_dtype": "bf16"}),
        );
        let work = tempfile::tempdir().unwrap();
        let file_name = format!("{name}.safetensors");
        let req = TrainingRequest {
            items: items.clone(),
            config: cfg,
            output_dir: work.path().to_path_buf(),
            file_name: file_name.clone(),
            trigger_words: Vec::new(),
            cancel: Default::default(),
        };
        let res = registry
            .load_trainer(ID, &spec)
            .unwrap()
            .train(&req, &mut |_| {})
            .unwrap();
        match kind {
            Some(_) => {
                std::fs::copy(&res.adapter_path, artifact_path(&out, name)).unwrap();
            }
            None => {
                let dst = artifact_path(&out, name);
                std::fs::create_dir_all(&dst).unwrap();
                let src = res.adapter_path.parent().unwrap();
                for f in ["config.yaml", "model.safetensors"] {
                    std::fs::copy(src.join(f), dst.join(f)).unwrap();
                }
            }
        }
        forwards.insert(name.to_string(), forward(&load(&out, name, kind)));
    }
    let list: Vec<(&str, &Tensor)> = forwards.iter().map(|(k, v)| (k.as_str(), v)).collect();
    let meta: HashMap<String, String> = [("backend".to_string(), "candle".to_string())].into();
    safetensors::serialize_to_file(list, Some(meta), &out.join("forward.safetensors")).unwrap();
}
