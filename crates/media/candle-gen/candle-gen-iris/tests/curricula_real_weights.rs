//! sc-25689 — the Candle twin of the MLX `curricula_real_weights` (on the build's device: `cuda:0`
//! under `--features cuda`). The official curriculum on real weights, bounded: the `stage1_256` preset (fixed
//! 256², square RoPE, REPA 0.5 against the provisioned `facebook/dinov2-base` teacher, Muon, bf16)
//! for two optimizer steps, then the `stage2_512` preset (`bucket` shapes over the shared21-512
//! table, isotropic RoPE, no REPA, full activation checkpointing) continuing stage 1's checkpoint
//! as a `new_phase` to step 4 — through the catalog trainer on the real Iris-3B generation backbone
//! and Qwen3-VL-4B. The presets are taken as shipped and edited only for the bound: batch 1, the
//! 4-step horizon, no EMA (disk), one rank (the 8-way hybrid FSDP mesh and the 1024 global batch
//! removed; the presets' `perf` block — `lazy_sync` honored, the torch.compile flags inert
//! natively — stays), one retained checkpoint. Writes the metrics, the checkpoint states and a
//! timing line into `IRIS_OUT`.
//!
//! Inputs: `IRIS_WEIGHTS_DIR` (the generation backbone: `config.yaml` + `model.safetensors`),
//! `IRIS_TEXT_ENCODER_DIR`, `IRIS_REPA_TEACHER_DIR` (`facebook/dinov2-base` @ the pinned
//! revision), `IRIS_OUT`; the dataset is crops of the repository photo (squares, a wide and a
//! tall one, so stage 2 fills several buckets).
//!
//! `cargo test -p candle-gen-iris --release --features cuda --test integration -- --ignored
//! curricula_real_weights:: --test-threads 1`

use std::path::{Path, PathBuf};
use std::time::Instant;

use candle_gen::gen_core::{LoadSpec, WeightsSource};
use candle_gen::gen_core::{Trainer, TrainingConfig, TrainingItem, TrainingRequest};
use gen_core_iris::train::repa::TEACHER_COMPONENT;
use gen_core_iris::train::upstream::stage_preset_config;
use gen_core_iris::train::{
    checkpoint_root, latest_checkpoint, read_checkpoint_state, OPTIONS_KEY,
};
use gen_core_iris::TEXT_ENCODER_COMPONENT;
use serde_json::{json, Value as Json};

const CAPTION: &str = "a photo of a dog sitting on grass";
const NAME: &str = "curriculum.safetensors";

fn env_dir(name: &str) -> PathBuf {
    PathBuf::from(std::env::var(name).unwrap_or_else(|_| panic!("{name} is required")))
}

/// Crops of the repository photo: three squares, one 16:9 and one 9:16.
fn dataset(out: &Path) -> Vec<TrainingItem> {
    let photo = Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("../../mlx-gen/_vendor/mage_flow/assets/dog.jpg");
    let rgb = candle_gen::gen_core::__model_input_reader!(
        image::open(photo),
        "crates/media/candle-gen/candle-gen-iris/tests/curricula_real_weights.rs::dataset/image::open",
        Reference,
        photo,
        |__model_input_path| image::open(__model_input_path)
    )
        .expect("the repo photo decodes")
        .to_rgb8();
    let (w, h) = rgb.dimensions();
    let side = w.min(h);
    let dir = out.join("train-data");
    std::fs::create_dir_all(&dir).unwrap();
    let mut rects = vec![
        (0, 0, side, side),
        ((w - side) / 2, (h - side) / 2, side, side),
        (w - side, h - side, side, side),
    ];
    let wide = (w, (w * 9 / 16).min(h));
    rects.push((0, (h - wide.1) / 2, wide.0, wide.1));
    let tall = ((h * 9 / 16).min(w), h);
    rects.push(((w - tall.0) / 2, 0, tall.0, tall.1));
    rects
        .into_iter()
        .enumerate()
        .map(|(i, (x, y, cw, ch))| {
            let crop = image::imageops::crop_imm(&rgb, x, y, cw, ch).to_image();
            let p = dir.join(format!("crop{i}.png"));
            crop.save(&p).unwrap();
            TrainingItem::captioned(p, CAPTION.into())
        })
        .collect()
}

fn trainer(teacher: Option<&Path>) -> Box<dyn Trainer> {
    let mut spec = LoadSpec::new(WeightsSource::Dir(env_dir("IRIS_WEIGHTS_DIR")));
    spec.components.insert(
        TEXT_ENCODER_COMPONENT.into(),
        WeightsSource::Dir(env_dir("IRIS_TEXT_ENCODER_DIR")),
    );
    if let Some(t) = teacher {
        spec.components.insert(
            TEACHER_COMPONENT.into(),
            WeightsSource::Dir(t.to_path_buf()),
        );
    }
    candle_gen_iris::provider_registry()
        .unwrap()
        .load_trainer("iris_3b", &spec)
        .expect("the real backbone loads as a trainer")
}

/// An official preset, edited only for the bound (see the module docs).
fn bounded_preset(id: &str, steps: u32) -> TrainingConfig {
    let mut cfg = stage_preset_config(id, TrainingConfig::default()).unwrap();
    cfg.batch_size = 1;
    cfg.steps = steps;
    let o = cfg
        .model_options
        .get_mut(OPTIONS_KEY)
        .and_then(Json::as_object_mut)
        .unwrap();
    for k in ["dist", "expected_global_batch"] {
        o.remove(k);
    }
    o.insert("ema_enabled".into(), json!(false));
    o.insert("keep_last_checkpoints".into(), json!(1));
    o.insert("log_every".into(), json!(1));
    cfg
}

fn request(items: &[TrainingItem], cfg: TrainingConfig, out: &Path) -> TrainingRequest {
    TrainingRequest {
        items: items.to_vec(),
        config: cfg,
        output_dir: out.to_path_buf(),
        file_name: NAME.into(),
        trigger_words: Vec::new(),
        cancel: Default::default(),
    }
}

fn metrics(out: &Path) -> Vec<Json> {
    std::fs::read_to_string(checkpoint_root(out, NAME).join("metrics.jsonl"))
        .unwrap()
        .lines()
        .map(|l| serde_json::from_str(l).unwrap())
        .collect()
}

#[test]
#[ignore = "needs the real weights (IRIS_WEIGHTS_DIR, IRIS_TEXT_ENCODER_DIR, IRIS_REPA_TEACHER_DIR) and IRIS_OUT"]
fn real_stage_presets_resume_256_to_512_with_repa() {
    let out = env_dir("IRIS_OUT");
    let teacher = env_dir("IRIS_REPA_TEACHER_DIR");
    let work = tempfile::tempdir().unwrap();
    let items = dataset(&out);

    let s1 = bounded_preset("stage1_256", 2);
    let out1 = work.path().join("stage1");
    let t0 = Instant::now();
    let steps = trainer(Some(&teacher))
        .train(&request(&items, s1, &out1), &mut |_| {})
        .expect("stage 1 trains");
    let stage1_secs = t0.elapsed().as_secs_f64();
    assert_eq!(steps.steps, 2);
    let ck1 = latest_checkpoint(&checkpoint_root(&out1, NAME)).unwrap();
    let st1 = read_checkpoint_state(&ck1).unwrap();
    assert_eq!(st1.stage.as_deref(), Some("stage1_256"));
    assert!(st1.repa_projector.is_some());
    let m1 = metrics(&out1);
    let repa: Vec<f64> = m1
        .iter()
        .filter(|r| r["kind"] == "step")
        .map(|r| r["loss/repa_raw"].as_f64().expect("REPA loss logged"))
        .collect();
    assert_eq!(repa.len(), 2);
    assert!(repa.iter().all(|v| v.is_finite()), "{repa:?}");

    let mut s2 = bounded_preset("stage2_512", 4);
    s2.model_options
        .get_mut(OPTIONS_KEY)
        .and_then(Json::as_object_mut)
        .unwrap()
        .extend([
            ("resume_from".to_string(), json!(ck1)),
            ("activation_checkpointing".to_string(), json!("full")),
            // single-process stage 2 drops stage 1's projector: `core`, as upstream's replicated
            // load requires (its FSDP2 cluster load forgets the projector group by itself)
            ("resume_optimizer".to_string(), json!("core")),
        ]);
    let out2 = work.path().join("stage2");
    let t1 = Instant::now();
    let steps = trainer(None)
        .train(&request(&items, s2, &out2), &mut |_| {})
        .expect("stage 2 continues stage 1");
    let stage2_secs = t1.elapsed().as_secs_f64();
    assert_eq!(steps.steps, 4);
    let st2 =
        read_checkpoint_state(&latest_checkpoint(&checkpoint_root(&out2, NAME)).unwrap()).unwrap();
    assert_eq!(
        (st2.stage.as_deref(), st2.parent_stage.as_deref(), st2.step),
        (Some("stage2_512"), Some("stage1_256"), 4)
    );
    assert!(st2.repa_projector.is_none());
    let m2 = metrics(&out2);
    let losses: Vec<f64> = m2
        .iter()
        .filter(|r| r["kind"] == "step")
        .map(|r| r["loss"].as_f64().unwrap())
        .collect();
    assert_eq!(losses.len(), 2, "stage 2 ran steps 3 and 4");
    assert!(losses.iter().all(|v| v.is_finite()), "{losses:?}");
    let run = m2.iter().find(|r| r["kind"] == "run").unwrap();
    assert!(
        run["shape_policy"].as_str().unwrap().starts_with("bucket"),
        "{run}"
    );
    assert!(out2.join("curriculum").join("model.safetensors").is_file());

    std::fs::write(
        out.join("curricula-real.json"),
        serde_json::to_vec_pretty(&json!({
            "stage1": {"secs": stage1_secs, "metrics": m1, "state": st1.to_json()},
            "stage2": {"secs": stage2_secs, "metrics": m2, "state": st2.to_json()},
        }))
        .unwrap(),
    )
    .unwrap();
    eprintln!("[[IRIS_CURRICULA_REAL]] stage1 {stage1_secs:.1}s, stage2 {stage2_secs:.1}s");

    #[cfg(feature = "model-inference-proof")]
    println!("\nMODEL_INFERENCE_BODY_COMPLETED curricula_real_weights::real_stage_presets_resume_256_to_512_with_repa");
}
