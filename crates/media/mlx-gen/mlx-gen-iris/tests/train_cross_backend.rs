//! Cross-backend training artifacts (sc-25686): an Iris artifact trained on one backend loads — and
//! means the same thing — on the other. The MLX half of `candle-gen-iris`'s `train_cross_backend`.
//!
//! `tests/fixtures/cross-backend/<backend>/` each hold one miniature LoRA, one LoKr and one
//! full-model export produced by that backend's trainer ([`produce_cross_backend_fixtures`] here,
//! its twin in `candle-gen-iris`), plus `forward.safetensors`: the producing backend's own FP32
//! forward of each artifact on the `iris_dit_golden` inputs, loaded through its own inference
//! route. This backend loads each artifact through **its** inference route (identity stamps
//! checked) and must reproduce that forward to the cross-backend FP32 bound ([`tolerance`]), and the artifact must visibly move the output — so a file that "loads" but is
//! read differently (a transposed factor, a lost alpha, a dropped target) fails. FP32 on the MLX
//! CPU stream (miniature only).

use std::collections::HashMap;
use std::path::{Path, PathBuf};

use mlx_gen::gen_core::iris::train::{AdapterMetadata, OPTIONS_KEY};
use mlx_gen::gen_core::iris::{IrisConfig, IrisTask, TEXT_ENCODER_COMPONENT};
use mlx_gen::gen_core::{NetworkType, TrainingConfig, TrainingItem, TrainingRequest};
use mlx_gen::weights::Weights;
use mlx_gen::{AdapterKind, AdapterSpec, LoadSpec, WeightsSource};
use mlx_gen_iris::{load_backbone, load_backbone_with_adapters, IrisDiT, TextBatch};
use mlx_rs::{Array, Dtype};
use serde_json::json;

use crate::common::{
    assert_close, errors, fixture, fixtures, host_i32, on_cpu, tiny_backbone, tiny_config,
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

/// Load one artifact of `dir` through the MLX inference route at FP32.
fn load(dir: &Path, name: &str, kind: Option<AdapterKind>) -> IrisDiT {
    let path = artifact_path(dir, name);
    match kind {
        Some(kind) => {
            let (dit, reports) = load_backbone_with_adapters(
                &tiny_backbone(),
                &tiny_config(),
                Dtype::Float32,
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
            load_backbone(&path, &cfg, Dtype::Float32).unwrap()
        }
    }
}

/// The FP32 forward on the `iris_dit_golden` inputs.
fn forward(dit: &IrisDiT) -> Array {
    let golden = fixture("iris_dit_golden.safetensors");
    let flat = host_i32(golden.require("y_mask").unwrap());
    let mask: Vec<Vec<i32>> = flat.chunks(flat.len() / 2).map(<[i32]>::to_vec).collect();
    let out = dit
        .forward(
            golden.require("x").unwrap(),
            golden.require("t").unwrap(),
            &TextBatch {
                states: golden.require("y").unwrap(),
                mask: &mask,
            },
        )
        .unwrap();
    mlx_rs::transforms::eval([&out]).unwrap();
    out
}

fn metadata(path: &Path) -> HashMap<String, String> {
    Array::load_safetensors_with_metadata(path)
        .unwrap()
        .1
        .into_iter()
        .collect()
}

fn check_trained_elsewhere(backend: &str) {
    on_cpu(|| {
        let dir = fixtures().join("cross-backend").join(backend);
        let want = Weights::from_file(dir.join("forward.safetensors")).unwrap();
        assert_eq!(want.metadata("backend"), Some(backend));
        let bare =
            forward(&load_backbone(&tiny_backbone(), &tiny_config(), Dtype::Float32).unwrap());
        for (name, kind) in ARTIFACTS {
            let tensor_file = match kind {
                Some(_) => artifact_path(&dir, name),
                None => artifact_path(&dir, name).join("model.safetensors"),
            };
            let meta = metadata(&tensor_file);
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
            let expected = want.require(name).unwrap();
            assert_close(
                &format!("{backend}-trained {name} on mlx"),
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
    });
}

#[test]
fn candle_trained_artifacts_load_and_match_on_mlx() {
    check_trained_elsewhere("candle");
}

/// The MLX-produced fixtures still load here (and their recorded forward is this backend's).
#[test]
fn mlx_trained_fixtures_match_their_recorded_forward() {
    check_trained_elsewhere("mlx");
}

/// Producer of `fixtures/cross-backend/mlx/` (run once on purpose, never in CI):
/// `IRIS_CROSS_BACKEND_OUT=<dir> cargo test -p mlx-gen-iris --test integration --release \
/// train_cross_backend::produce_cross_backend_fixtures -- --ignored`, then copy `<dir>` over the
/// committed fixture directory. Miniature, MLX CPU stream.
#[test]
#[ignore = "fixture producer: writes into IRIS_CROSS_BACKEND_OUT"]
fn produce_cross_backend_fixtures() {
    on_cpu(|| {
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
        let registry = mlx_gen_iris::provider_registry().unwrap();
        let mut forwards: Vec<(String, Array)> = Vec::new();
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
            let req = TrainingRequest {
                items: items.clone(),
                config: cfg,
                output_dir: work.path().to_path_buf(),
                file_name: format!("{name}.safetensors"),
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
            forwards.push((name.to_string(), forward(&load(&out, name, kind))));
        }
        let meta: HashMap<String, String> = [("backend".to_string(), "mlx".to_string())].into();
        Array::save_safetensors(
            forwards.iter().map(|(k, v)| (k.as_str(), v)),
            Some(&meta),
            out.join("forward.safetensors"),
        )
        .unwrap();
    });
}
