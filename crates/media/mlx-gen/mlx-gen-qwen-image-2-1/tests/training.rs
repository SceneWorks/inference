//! The Qwen-Image 2.1 LoRA/LoKr trainer (sc-24159), end to end on the committed miniature
//! snapshot — the real staged lifecycle (caption caching → latent caching → DiT training), with no
//! real weights:
//!
//! * the shared gen-core trainer conformance suite (validate honesty, progress bands, typed
//!   cancellation, registry round-trip);
//! * a trained adapter (LoRA and LoKr) loads back through the sc-24156 adapter host **strictly**,
//!   with no key conversion, and changes the DiT's velocity; its `__metadata__` names the family,
//!   the base model and the Qwen Research License;
//! * checkpoints + resume: a run cancelled after its step-2 checkpoint resumes from that
//!   checkpoint and finishes with the same adapter an uninterrupted run writes;
//! * preview samples render through the crate's own render path during training;
//! * a pre-quantized tier, or a quantize request, is a typed refusal ("install the BF16 tier").

use std::cell::Cell;
use std::path::{Path, PathBuf};

use mlx_gen::gen_core::weightsmeta::safetensors_file_metadata;
use mlx_gen::runtime::{AdapterKind, AdapterSpec};
use mlx_gen::{
    Error, LoadSpec, NetworkType, Quant, Trainer, TrainingConfig, TrainingItem, TrainingProgress,
    TrainingRequest, WeightsSource,
};
use mlx_gen_qwen_image_2_1::convert::prequantize_turnkey;
use mlx_gen_qwen_image_2_1::quant::Tier;
use mlx_gen_qwen_image_2_1::{
    apply_qwen_image_2_1_adapters, load_trainer, load_transformer, provider_registry,
    QwenImage21Transformer, BLOCK_ADAPTER_TARGETS, TRAINER_ID,
};
use mlx_rs::Array;
use gen_core_testkit::trainer::{trainer_conformance, TrainerProfile};

use crate::common::{errors, tiny_snapshot};

/// A deterministic 64×64 RGB PNG (a diagonal colour ramp) written under `dir`.
fn write_image(dir: &Path, name: &str, phase: u32) -> PathBuf {
    let img = image::RgbImage::from_fn(64, 64, |x, y| {
        image::Rgb([
            ((x * 4 + phase) % 256) as u8,
            ((y * 4 + 2 * phase) % 256) as u8,
            (((x + y) * 2) % 256) as u8,
        ])
    });
    let path = dir.join(name);
    img.save(&path).expect("write the dataset image");
    path
}

fn dataset(dir: &Path) -> Vec<TrainingItem> {
    vec![
        TrainingItem::captioned(write_image(dir, "a.png", 0), "a red swatch".into()),
        TrainingItem::captioned(write_image(dir, "b.png", 40), "a blue ramp".into()),
    ]
}

fn dense_spec() -> LoadSpec {
    LoadSpec::new(WeightsSource::Dir(tiny_snapshot()))
}

fn trainer() -> Box<dyn Trainer> {
    // Through the explicit registry, exactly as the worker resolves it.
    provider_registry()
        .unwrap()
        .load_trainer(TRAINER_ID, &dense_spec())
        .expect("the tiny snapshot loads as a trainer")
}

fn config(steps: u32) -> TrainingConfig {
    TrainingConfig {
        rank: 4,
        alpha: 4.0,
        learning_rate: 1e-2,
        steps,
        resolution: 64,
        save_every: 0,
        seed: 7,
        // f32 keeps the run-to-run comparisons tight; the bf16 path is the conformance run's.
        train_dtype: "f32".into(),
        ..Default::default()
    }
}

fn request(items: Vec<TrainingItem>, config: TrainingConfig, out: &Path) -> TrainingRequest {
    TrainingRequest {
        items,
        config,
        output_dir: out.to_path_buf(),
        file_name: "qwen21_lora.safetensors".into(),
        trigger_words: Vec::new(),
        cancel: Default::default(),
    }
}

/// One run's outcome.
type RunResult = mlx_gen::gen_core::Result<mlx_gen::TrainingOutput>;

/// `(steps seen, output)` of one run.
fn run(
    trainer: &mut dyn Trainer,
    req: &TrainingRequest,
    mut on_step: impl FnMut(u32),
) -> (Vec<u32>, RunResult) {
    let mut steps = Vec::new();
    let out = trainer.train(req, &mut |p| {
        if let TrainingProgress::Training { step, .. } = p {
            steps.push(step);
            on_step(step);
        }
    });
    (steps, out)
}

#[test]
fn trainer_conformance_on_the_tiny_snapshot() {
    let tmp = tempfile::tempdir().unwrap();
    let data = tmp.path().join("data");
    std::fs::create_dir_all(&data).unwrap();
    let items = dataset(&data);
    let out = tmp.path().join("out");
    trainer_conformance(trainer, &TrainerProfile::cheap(items, out));
}

/// The velocity of `dit` on a fixed 4×4 latent grid conditioned on 5 fixed text rows.
fn velocity(dit: &QwenImage21Transformer) -> Array {
    let c = dit.config();
    let latents = Array::from_slice(
        &(0..16 * c.in_channels)
            .map(|i| (i as f32 * 0.13).sin())
            .collect::<Vec<_>>(),
        &[1, 16, c.in_channels as i32],
    );
    let text = Array::from_slice(
        &(0..5 * c.context_in_dim)
            .map(|i| (i as f32 * 0.07).cos())
            .collect::<Vec<_>>(),
        &[1, 5, c.context_in_dim as i32],
    );
    let v = dit.forward(&latents, &text, 0.5, 4, 4).unwrap();
    mlx_rs::transforms::eval([&v]).unwrap();
    v
}

/// AC2/AC3: a trained adapter (LoRA and LoKr) is PEFT/LyCORIS safetensors that the sc-24156 host
/// installs **strictly** — every key resolves, nothing is converted — and that changes the
/// velocity; its metadata stamps the family, the base model and the research licence.
#[test]
fn a_trained_adapter_loads_back_strictly_changes_the_velocity_and_is_stamped() {
    for (network, kind) in [
        (NetworkType::Lora, AdapterKind::Lora),
        (NetworkType::Lokr, AdapterKind::Lokr),
    ] {
        let tmp = tempfile::tempdir().unwrap();
        let items = dataset(tmp.path());
        let out = tmp.path().join("out");
        let req = request(
            items,
            TrainingConfig {
                network_type: network,
                ..config(3)
            },
            &out,
        );
        let mut t = trainer();
        let (steps, result) = run(t.as_mut(), &req, |_| {});
        let output = result.unwrap_or_else(|e| panic!("{network:?} training failed: {e}"));
        assert_eq!(steps, [1, 2, 3]);
        assert_eq!(output.steps, 3);
        assert!(output.final_loss.is_finite());

        let meta = safetensors_file_metadata(&output.adapter_path).unwrap();
        assert_eq!(
            meta.get("family").map(String::as_str),
            Some("qwen-image-2-1")
        );
        assert_eq!(
            meta.get("baseModel").map(String::as_str),
            Some("qwen_image_2_1")
        );
        assert_eq!(
            meta.get("ss_base_model_version").map(String::as_str),
            Some("qwen_image_2_1")
        );
        assert!(
            meta.get("license")
                .is_some_and(|l| l.contains("Qwen Research License")),
            "{meta:?}"
        );
        assert!(
            meta.get("licenseNotice")
                .is_some_and(|l| l.contains("Qwen RESEARCH LICENSE AGREEMENT")),
            "{meta:?}"
        );
        let expected_kind = if network == NetworkType::Lora {
            "lora"
        } else {
            "lokr"
        };
        assert_eq!(
            meta.get("networkType").map(String::as_str),
            Some(expected_kind)
        );

        let mut dit = load_transformer(&tiny_snapshot()).unwrap();
        let base = velocity(&dit);
        let report = apply_qwen_image_2_1_adapters(
            &mut dit,
            &[AdapterSpec::new(output.adapter_path.clone(), 1.0, kind)],
        )
        .unwrap_or_else(|e| panic!("{network:?}: the strict host refused the trained file: {e}"));
        assert!(report.unmatched_paths.is_empty(), "{report:?}");
        assert_eq!(
            report.applied,
            dit.num_blocks() * BLOCK_ADAPTER_TARGETS.len(),
            "{network:?}: every default target must land"
        );
        let adapted = velocity(&dit);
        let (max_abs, peak, _) = errors(&adapted, &base);
        eprintln!("[sc-24159] {network:?} adapter Δvelocity max {max_abs:.3e} (peak {peak:.3e})");
        assert!(
            max_abs > 1e-6,
            "{network:?}: the trained adapter must change the velocity"
        );
    }
}

/// AC1/AC3: checkpoints + resume. A run cancelled after its step-2 checkpoint (it stops at step
/// 3) resumes from that checkpoint — steps 3..=4 run again — and writes the same adapter as an
/// uninterrupted run of the same config.
#[test]
fn a_cancelled_run_resumes_from_its_last_checkpoint_and_matches_a_straight_run() {
    let tmp = tempfile::tempdir().unwrap();
    let items = dataset(tmp.path());
    let cfg = TrainingConfig {
        save_every: 2,
        ..config(4)
    };

    // Uninterrupted reference run.
    let straight_dir = tmp.path().join("straight");
    let mut t = trainer();
    let (steps, result) = run(
        t.as_mut(),
        &request(items.clone(), cfg.clone(), &straight_dir),
        |_| {},
    );
    let straight = result.unwrap();
    assert_eq!(steps, [1, 2, 3, 4]);
    assert!(
        straight_dir
            .join("qwen21_lora-step000002.resume.safetensors")
            .is_file(),
        "the step-2 resume snapshot must be written"
    );
    let ckpt_meta =
        safetensors_file_metadata(straight_dir.join("qwen21_lora-step000002.safetensors")).unwrap();
    assert_eq!(
        ckpt_meta.get("family").map(String::as_str),
        Some("qwen-image-2-1"),
        "intermediate checkpoints carry the provenance stamp too"
    );

    // Interrupted run: cancel once step 3 has run → the loop stops before step 4.
    let resumed_dir = tmp.path().join("resumed");
    let req = request(items.clone(), cfg.clone(), &resumed_dir);
    let cancel = req.cancel.clone();
    let tripped = Cell::new(false);
    let mut t = trainer();
    let (steps, result) = run(t.as_mut(), &req, |step| {
        if step == 3 {
            cancel.cancel();
            tripped.set(true);
        }
    });
    assert!(tripped.get());
    assert_eq!(steps, [1, 2, 3]);
    assert_eq!(
        result.unwrap().steps,
        3,
        "a cancel after a step is a partial Ok"
    );

    // Resume: continues from the step-2 snapshot (steps 3 and 4 run again).
    let req = request(
        items,
        TrainingConfig {
            resume: true,
            ..cfg
        },
        &resumed_dir,
    );
    let mut t = trainer();
    let (steps, result) = run(t.as_mut(), &req, |_| {});
    let resumed = result.unwrap();
    assert_eq!(steps, [3, 4], "resume must continue from the saved step");
    assert_eq!(resumed.steps, 4);

    let a = Array::load_safetensors(&straight.adapter_path).unwrap();
    let b = Array::load_safetensors(&resumed.adapter_path).unwrap();
    assert_eq!(a.len(), b.len());
    for (key, want) in &a {
        let got = b
            .get(key)
            .unwrap_or_else(|| panic!("resumed adapter lacks {key}"));
        let (max_abs, peak, _) = errors(got, want);
        assert!(
            max_abs <= 1e-5 * peak.max(1.0),
            "{key}: resumed {max_abs:.3e} away from the straight run (peak {peak:.3e})"
        );
    }
}

/// AC1: preview samples render from the in-progress adapter through the crate's render path.
#[test]
fn preview_samples_render_during_training() {
    let tmp = tempfile::tempdir().unwrap();
    let items = dataset(tmp.path());
    let req = request(
        items,
        TrainingConfig {
            sample_every: 2,
            sample_prompts: vec!["a green square".into()],
            sample_steps: 2,
            sample_guidance_scale: 2.0,
            gradient_checkpointing: true,
            ..config(2)
        },
        &tmp.path().join("out"),
    );
    let mut t = trainer();
    let mut samples = Vec::new();
    t.train(&req, &mut |p| {
        if let TrainingProgress::Sample {
            step,
            index,
            total,
            image,
            ..
        } = p
        {
            samples.push((step, index, total, image.width, image.height));
        }
    })
    .unwrap();
    assert_eq!(samples, [(2, 1, 1, 64, 64)]);
}

/// AC1/AC3: QLoRA is a non-goal — a pre-quantized tier and a quantize request are both typed
/// refusals that say to install the BF16 tier.
#[test]
fn a_quantized_base_is_refused_with_the_bf16_instruction() {
    let tmp = tempfile::tempdir().unwrap();
    let tier = tmp.path().join("q8");
    prequantize_turnkey(&tiny_snapshot(), &tier, Tier::Q8).expect("the tiny snapshot converts");
    for spec in [
        LoadSpec::new(WeightsSource::Dir(tier.clone())),
        LoadSpec::new(WeightsSource::Dir(tier)).with_quant(Quant::Q8),
        dense_spec().with_quant(Quant::Q4),
    ] {
        match load_trainer(&spec) {
            Err(Error::Unsupported(message)) => {
                assert!(
                    message.contains("install the BF16 tier to train"),
                    "{message}"
                )
            }
            Err(other) => panic!("expected a typed Unsupported, got {other}"),
            Ok(_) => panic!("a quantized base must not load as a trainer"),
        }
    }
}
