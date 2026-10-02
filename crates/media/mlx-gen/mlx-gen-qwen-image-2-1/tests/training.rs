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
//! * a pre-quantized tier, or a quantize request, is a typed refusal ("install the BF16 tier");
//! * edit mode (sc-24161): an edit-pair dataset trains an adapter that is marked as an edit
//!   adapter, loads through the same host, and changes a reference/edit render.

use std::cell::Cell;
use std::path::{Path, PathBuf};

use gen_core_testkit::trainer::{trainer_conformance, TrainerProfile};
use mlx_gen::gen_core::weightsmeta::safetensors_file_metadata;
use mlx_gen::gen_core::Conditioning;
use mlx_gen::runtime::{AdapterKind, AdapterSpec};
use mlx_gen::{
    Error, GenerationOutput, GenerationRequest, Image, LoadSpec, NetworkType, Quant, Trainer,
    TrainingConfig, TrainingItem, TrainingProgress, TrainingRequest, WeightsSource,
};
use mlx_gen_qwen_image_2_1::convert::prequantize_turnkey;
use mlx_gen_qwen_image_2_1::quant::Tier;
use mlx_gen_qwen_image_2_1::{
    apply_qwen_image_2_1_adapters, load_trainer, load_transformer, provider_registry,
    QwenImage21Transformer, BLOCK_ADAPTER_TARGETS, TRAINER_ID,
};
use mlx_rs::Array;

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

        let meta = safetensors_file_metadata(output.adapter_path.as_path()).unwrap();
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
        assert_eq!(
            meta.get("trainingMode"),
            None,
            "a text-to-image adapter carries no edit marker"
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

/// One resume scenario: a run of `cfg` cancelled once step `cancel_at` has run resumes from the
/// latest resume snapshot (`snapshot_step`) and finishes with the same adapter an uninterrupted
/// run of `cfg` writes. `no_snapshot_at` are `save_every` multiples that land inside a
/// gradient-accumulation window and must NOT leave a resume snapshot (only an adapter checkpoint).
fn assert_cancelled_run_resumes(
    cfg: TrainingConfig,
    cancel_at: u32,
    snapshot_step: u32,
    no_snapshot_at: &[u32],
) {
    let tmp = tempfile::tempdir().unwrap();
    let items = dataset(tmp.path());
    let all: Vec<u32> = (1..=cfg.steps).collect();
    let resume_file =
        |dir: &Path, step: u32| dir.join(format!("qwen21_lora-step{step:06}.resume.safetensors"));

    // Uninterrupted reference run.
    let straight_dir = tmp.path().join("straight");
    let mut t = trainer();
    let (steps, result) = run(
        t.as_mut(),
        &request(items.clone(), cfg.clone(), &straight_dir),
        |_| {},
    );
    let straight = result.unwrap();
    assert_eq!(steps, all);
    assert!(
        resume_file(&straight_dir, snapshot_step).is_file(),
        "the step-{snapshot_step} resume snapshot must be written"
    );
    for &step in no_snapshot_at {
        assert!(
            !resume_file(&straight_dir, step).exists(),
            "step {step} is inside an accumulation window: no resume snapshot"
        );
        assert!(
            straight_dir
                .join(format!("qwen21_lora-step{step:06}.safetensors"))
                .is_file(),
            "the step-{step} adapter checkpoint is still written"
        );
    }
    let ckpt_meta = safetensors_file_metadata(
        straight_dir.join(format!("qwen21_lora-step{snapshot_step:06}.safetensors")),
    )
    .unwrap();
    assert_eq!(
        ckpt_meta.get("family").map(String::as_str),
        Some("qwen-image-2-1"),
        "intermediate checkpoints carry the provenance stamp too"
    );

    // Interrupted run: cancel once step `cancel_at` has run → the loop stops before the next.
    let resumed_dir = tmp.path().join("resumed");
    let req = request(items.clone(), cfg.clone(), &resumed_dir);
    let cancel = req.cancel.clone();
    let tripped = Cell::new(false);
    let mut t = trainer();
    let (steps, result) = run(t.as_mut(), &req, |step| {
        if step == cancel_at {
            cancel.cancel();
            tripped.set(true);
        }
    });
    assert!(tripped.get());
    assert_eq!(steps, (1..=cancel_at).collect::<Vec<_>>());
    assert_eq!(
        result.unwrap().steps,
        cancel_at,
        "a cancel after a step is a partial Ok"
    );

    // Resume: continues from the snapshot, re-running every step after it.
    let total = cfg.steps;
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
    assert_eq!(
        steps,
        (snapshot_step + 1..=total).collect::<Vec<_>>(),
        "resume must continue from the saved step"
    );
    assert_eq!(resumed.steps, total);

    let a = Array::load_safetensors(straight.adapter_path.as_path()).unwrap();
    let b = Array::load_safetensors(resumed.adapter_path.as_path()).unwrap();
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

/// AC1/AC3: checkpoints + resume. A run cancelled after its step-2 checkpoint (it stops at step
/// 3) resumes from that checkpoint — steps 3..=4 run again — and writes the same adapter as an
/// uninterrupted run of the same config.
#[test]
fn a_cancelled_run_resumes_from_its_last_checkpoint_and_matches_a_straight_run() {
    assert_cancelled_run_resumes(
        TrainingConfig {
            save_every: 2,
            ..config(4)
        },
        3,
        2,
        &[],
    );
}

/// Resume is exact under gradient accumulation too: with `accum = 2` and `save_every = 3`, the
/// step-3 checkpoint falls inside an accumulation window, so it writes the adapter but no resume
/// snapshot (which cannot hold the half-accumulated gradients); the step-6 one is on an update
/// boundary. A run cancelled after step 7 resumes from step 6 and matches the straight run.
///
/// *Mutation that reds this:* writing the resume bundle on every `save_every` again — the step-3
/// snapshot then exists, and (cancelled at step 4) a resume would drop step 3's gradients.
#[test]
fn resume_under_gradient_accumulation_restarts_from_an_update_boundary() {
    assert_cancelled_run_resumes(
        TrainingConfig {
            save_every: 3,
            gradient_accumulation: 2,
            ..config(8)
        },
        7,
        6,
        &[3],
    );
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

// ── edit mode (sc-24161) ─────────────────────────────────────────────────────────────────────

/// A deterministic `width × height` RGB image (a seeded ramp) as both a PNG under `dir` and the
/// in-memory [`Image`] the render path's conditioning carries.
fn reference(dir: &Path, name: &str, width: u32, height: u32, phase: u32) -> (PathBuf, Image) {
    let img = image::RgbImage::from_fn(width, height, |x, y| {
        image::Rgb([
            ((x * 3 + phase) % 256) as u8,
            ((y * 5 + phase) % 256) as u8,
            ((x * y + 3 * phase) % 256) as u8,
        ])
    });
    let path = dir.join(name);
    img.save(&path).expect("write the reference image");
    let image = Image {
        width,
        height,
        pixels: img.into_raw(),
    };
    (path, image)
}

fn render(spec: &LoadSpec, req: &GenerationRequest) -> Image {
    let generator = provider_registry()
        .unwrap()
        .load(TRAINER_ID, spec)
        .expect("the snapshot loads as a generator");
    match generator.generate(req, &mut |_| {}).expect("render") {
        GenerationOutput::Images(mut images) => images.remove(0),
        other => panic!("images expected, got {other:?}"),
    }
}

/// AC: edit training end to end on the miniature snapshot. Two edit pairs (each: a target, an
/// instruction, two ordered references) train a LoRA through the real staged lifecycle; the saved
/// adapter carries the family/base/licence provenance plus the edit marker; and loaded through the
/// sc-24156 host it changes a 2.1 **edit** render (same references, prompt and seed) against the
/// bare base.
#[test]
fn an_edit_adapter_trains_is_marked_and_changes_the_edit_render() {
    let tmp = tempfile::tempdir().unwrap();
    let dir = tmp.path();
    let (ref_a, image_a) = reference(dir, "ref_a.png", 64, 64, 11);
    let (ref_b, image_b) = reference(dir, "ref_b.png", 64, 64, 97);
    let instruction = "paint the first image in the colours of the second";
    let items = vec![
        TrainingItem::edit_pair(
            write_image(dir, "target_1.png", 0),
            instruction.into(),
            vec![ref_a.clone(), ref_b.clone()],
        ),
        TrainingItem::edit_pair(
            write_image(dir, "target_2.png", 40),
            "swap the two images".into(),
            vec![ref_b, ref_a],
        ),
    ];
    let req = request(
        items,
        TrainingConfig {
            learning_rate: 5e-2,
            // Every DiT Linear, globals included, so the 4-step adapter moves a 2-step render.
            lora_target_modules: vec![
                "attn.to_q".into(),
                "attn.to_k".into(),
                "attn.to_v".into(),
                "attn.to_out.0".into(),
                "img_mlp.gate_layer".into(),
                "img_mlp.proj".into(),
                "img_mlp.out".into(),
                "proj_out".into(),
            ],
            ..config(4)
        },
        &dir.join("out"),
    );
    let mut t = trainer();
    t.validate(&req)
        .expect("an edit dataset validates on the edit trainer");
    let mut losses = Vec::new();
    let mut cached = Vec::new();
    let output = t
        .train(&req, &mut |p| match p {
            TrainingProgress::Training { loss, .. } => losses.push(loss),
            TrainingProgress::Caching { current, total } => cached.push((current, total)),
            _ => {}
        })
        .expect("edit training runs");
    assert_eq!(output.steps, 4);
    assert_eq!(cached, [(1, 2), (2, 2)]);
    assert!(losses.iter().all(|l| l.is_finite()), "{losses:?}");
    eprintln!("[sc-24161] edit training losses {losses:?}");

    let meta = safetensors_file_metadata(output.adapter_path.as_path()).unwrap();
    assert_eq!(meta.get("trainingMode").map(String::as_str), Some("edit"));
    assert_eq!(
        meta.get("family").map(String::as_str),
        Some("qwen-image-2-1")
    );
    assert_eq!(
        meta.get("baseModel").map(String::as_str),
        Some("qwen_image_2_1")
    );
    assert!(
        meta.get("license")
            .is_some_and(|l| l.contains("Qwen Research License")),
        "{meta:?}"
    );

    // The same edit render (references, prompt, seed) with and without the trained adapter.
    let edit = GenerationRequest {
        prompt: instruction.to_owned(),
        width: 64,
        height: 64,
        steps: Some(2),
        seed: Some(42),
        conditioning: vec![
            Conditioning::Reference {
                image: image_a,
                strength: None,
            },
            Conditioning::Reference {
                image: image_b,
                strength: None,
            },
        ],
        ..Default::default()
    };
    let plain = render(&dense_spec(), &edit);
    let adapted_spec = dense_spec().with_adapters(vec![AdapterSpec::new(
        output.adapter_path.clone(),
        1.0,
        AdapterKind::Lora,
    )]);
    let adapted = render(&adapted_spec, &edit);
    // The same adapter, prompt and seed WITHOUT the references renders differently: the edit
    // render really is reference-conditioned through the adapted DiT (the T2I route would ignore
    // them). *Mutation that reds this:* the render path dropping `ReferenceConditioning`.
    let adapted_t2i = render(
        &adapted_spec,
        &GenerationRequest {
            conditioning: Vec::new(),
            ..edit.clone()
        },
    );
    assert_ne!(
        adapted.pixels, adapted_t2i.pixels,
        "the adapted edit render must depend on its references"
    );
    let changed = adapted
        .pixels
        .iter()
        .zip(&plain.pixels)
        .filter(|(a, b)| a != b)
        .count();
    eprintln!(
        "[sc-24161] trained edit adapter changed {changed}/{} edit-render bytes",
        plain.pixels.len()
    );
    assert!(
        changed > 0,
        "the trained edit adapter must change the edit render"
    );
}

/// AC: an edit dataset on a snapshot-level edit trainer is held to the render path's reference
/// cap (the descriptor's `max_reference_images`), refused before any weight loads.
#[test]
fn an_edit_item_over_the_reference_cap_is_refused_before_training() {
    let tmp = tempfile::tempdir().unwrap();
    let cap = mlx_gen_qwen_image_2_1::MAX_REFERENCE_IMAGES;
    let refs: Vec<PathBuf> = (0..=cap)
        .map(|i| tmp.path().join(format!("ref{i}.png")))
        .collect();
    let req = request(
        vec![TrainingItem::edit_pair(
            tmp.path().join("target.png"),
            "compose them".into(),
            refs,
        )],
        config(2),
        &tmp.path().join("out"),
    );
    let mut t = trainer();
    assert_eq!(t.descriptor().max_reference_images as usize, cap);
    let err = t.validate(&req).unwrap_err().to_string();
    assert!(err.contains(&format!("at most {cap}")), "{err}");
    let mut events = Vec::new();
    let err = t
        .train(&req, &mut |p| events.push(format!("{p:?}")))
        .unwrap_err()
        .to_string();
    assert!(err.contains(&format!("at most {cap}")), "{err}");
    assert!(
        !events.iter().any(|e| e.starts_with("LoadingModel")),
        "the refusal precedes every load: {events:?}"
    );
}
