//! End-to-end real-weight smoke for the candle Krea 2 **trainer** (sc-7577) — the production
//! `KreaTrainer` driven through the gen-core registry exactly as the SceneWorks worker will, on the
//! real Krea-2-Raw 12B base. The candle twin of `candle-gen-z-image/tests/trainer_e2e.rs`, on the
//! flow-match objective.
//!
//! `#[ignore]`d (not feature-gated, so the bodies are compile-checked in normal CI). Needs the real
//! Krea-2-Raw snapshot (`KREA_RAW_DIR`, or `KREA_TURBO_DIR` — Raw ≡ Turbo architecture, so a Turbo
//! snapshot also exercises the load+train path) and enough VRAM for the bf16 12B DiT (~24 GB resident
//! alongside the f32 VAE encoder / text encoder during caching). On the Windows/Blackwell box
//! (v143 vcvars + CUDA on PATH):
//!
//! ```text
//! set KREA_RAW_DIR=D:\models\Krea-2-Raw
//! cargo test -p candle-gen-krea --features cuda --release --test integration trainer_e2e:: -- --ignored --nocapture
//! ```
//!
//! What it proves (the sc-7577 AC):
//!  - **prepare→cache→train→save** lifecycle: a tiny captioned-PNG dataset is VAE-mean / Qwen3-VL-4B
//!    encoded + cached, the optimizer drives the flow-match velocity loss down (windowed median falls —
//!    convergence on real data, not just finite), and a PEFT/LoKr adapter is written with the expected
//!    bare-dotted DiT keys (`transformer_blocks.{i}.attn.{to_q,to_k,to_v,to_out.0}`) plus the
//!    `networkType`/`rank`/`alpha`/`baseModel`/`family` metadata the Turbo cross-apply policy reads.
//!  - **the 112-target default surface**: a LoRA adapts exactly the 28 single-stream blocks' four
//!    attention projections (28 × 4).
//!  - **gradient checkpointing** (the memory-bounded backward) converges + saves end-to-end (its
//!    bit-exact dense-vs-checkpoint grad parity is the f32 `dense_and_checkpoint_grads_match` unit gate).
//!  - **launch-portable determinism**: the same seed produces the same adapter bytes, run to run.
//!
//! The train→**infer** round-trip (a Raw-trained adapter applied at Turbo inference, sc-7837) is
//! exercised by [`krea_lora_roundtrip_raw_to_turbo`] now that the inference-apply seam has landed
//! (sc-7836): train on Raw → merge across the 112-target surface (nothing skipped) → render at Turbo
//! with the adapter (coherent + visibly adapted; scale-0 ≡ base byte-exact). Cross-framework numeric
//! parity with MLX/torch is explicitly a non-goal (different autograd + RNG); candle-internal
//! determinism + behavioural parity (converges; well-formed adapter; coherent adapted render) is what
//! is guaranteed.

use std::collections::HashMap;
use std::path::{Path, PathBuf};

use candle_gen::candle_core::{DType, Device};
use candle_gen::gen_core::{
    AdapterKind, AdapterSpec, CancelFlag, GenerationOutput, GenerationRequest, Image, LoadSpec,
    NetworkType, TrainingConfig, TrainingItem, TrainingProgress, TrainingRequest, WeightsSource,
};
use candle_gen::testkit::save_image_png;
use candle_gen_krea::loader::Weights;
use candle_gen_krea::{merge_into_weights, Krea2Config};

/// Root for this suite's **deliberately persistent** artifacts. `KREA_TRAINER_ARTIFACT_DIR` points them
/// somewhere durable; the `$TMPDIR` default is intentional and must NOT become a `tempfile`
/// guard — these outputs outlive the test on purpose (trained adapters and sample renders you inspect afterwards), so a guard
/// would delete the very thing the test exists to produce (sc-17791).
fn artifact_root() -> PathBuf {
    std::env::var("KREA_TRAINER_ARTIFACT_DIR")
        .map(PathBuf::from)
        .unwrap_or_else(|_| std::env::temp_dir())
}

/// The Krea-2-Raw base snapshot dir — `KREA_RAW_DIR`, or `KREA_TURBO_DIR` (architecture-identical).
fn snapshot() -> Option<PathBuf> {
    std::env::var("KREA_RAW_DIR")
        .or_else(|_| std::env::var("KREA_TURBO_DIR"))
        .ok()
        .map(PathBuf::from)
}

/// Two solid-colour swatch PNGs + captions in `dir`.
fn make_dataset(dir: &Path) -> Vec<TrainingItem> {
    let _ = std::fs::remove_dir_all(dir);
    std::fs::create_dir_all(dir).unwrap();
    let mut items = Vec::new();
    for (i, color) in [[200u8, 40, 40], [40, 80, 200]].iter().enumerate() {
        let mut img = image::RgbImage::new(96, 96);
        for px in img.pixels_mut() {
            *px = image::Rgb(*color);
        }
        let path = dir.join(format!("img{i}.png"));
        img.save(&path).unwrap();
        items.push(TrainingItem {
            image_path: path,
            caption: format!("a solid colour swatch number {i}"),
            control_image_path: None,
            model_options: Default::default(),
            reference_image_paths: Vec::new(),
            subject_mask_path: None,
        });
    }
    items
}

/// A tiny config: small rank, low resolution (64 → 8×8 latent → 4×4 = 16 image tokens), few steps.
/// **bf16** compute — the Krea DiT's native dtype and what inference loads. The trainable adapter
/// factors / loss / grads / optimizer state stay f32 regardless (master weights).
fn config(network_type: NetworkType, steps: u32, grad_ckpt: bool) -> TrainingConfig {
    TrainingConfig {
        rank: 8,
        alpha: 8.0,
        learning_rate: 1e-3,
        steps,
        resolution: 64,
        save_every: 0,
        seed: 7,
        network_type,
        decompose_factor: -1,
        train_dtype: "bf16".to_string(),
        gradient_checkpointing: grad_ckpt,
        ..Default::default()
    }
}

struct RunOut {
    losses: Vec<f32>,
    adapter_path: PathBuf,
}

/// Train through the registry and collect the per-step losses + the adapter path.
fn run(
    tmp: &Path,
    file_name: &str,
    network_type: NetworkType,
    steps: u32,
    grad_ckpt: bool,
) -> RunOut {
    let items = make_dataset(tmp);
    let mut trainer = candle_gen_krea::provider_registry()
        .unwrap()
        .load_trainer(
            "krea_2_raw",
            &LoadSpec::new(WeightsSource::Dir(
                snapshot().expect("KREA_RAW_DIR / KREA_TURBO_DIR"),
            )),
        )
        .expect("krea_2_raw candle trainer should be registered");

    let req = TrainingRequest {
        items,
        config: config(network_type, steps, grad_ckpt),
        output_dir: tmp.join("out"),
        file_name: file_name.to_string(),
        trigger_words: vec![],
        cancel: CancelFlag::new(),
    };

    let mut losses: Vec<f32> = Vec::new();
    let mut cached = 0u32;
    let out = trainer
        .train(&req, &mut |p| match p {
            TrainingProgress::Caching { current, .. } => cached = current,
            TrainingProgress::Training { loss, .. } => losses.push(loss),
            _ => {}
        })
        .expect("training should succeed");

    assert_eq!(cached, 2, "both dataset items should be cached");
    assert_eq!(out.steps, steps, "all micro-steps should run");
    assert_eq!(losses.len() as u32, steps);
    assert!(
        losses.iter().all(|l| l.is_finite()),
        "no NaN/Inf losses (not diverging)"
    );
    assert!(out.adapter_path.exists(), "adapter file should be written");
    println!("[{file_name}] losses: {losses:?}");
    RunOut {
        losses,
        adapter_path: out.adapter_path,
    }
}

/// Assert the windowed loss **median** falls (robust to the heavy-tailed per-σ flow-match loss — each
/// step samples a fresh σ, and a high-σ draw spikes that step regardless of progress). Compares the
/// first-quarter vs last-quarter median.
fn assert_converged(tag: &str, losses: &[f32]) {
    let q = (losses.len() / 4).max(1);
    let median = |s: &[f32]| {
        let mut v = s.to_vec();
        v.sort_by(|a, b| a.partial_cmp(b).unwrap());
        v[v.len() / 2]
    };
    let (first_q, last_q) = (median(&losses[..q]), median(&losses[losses.len() - q..]));
    println!("[{tag}] loss-median first-quarter {first_q:.5} -> last-quarter {last_q:.5}");
    assert!(
        last_q < first_q * 0.9,
        "[{tag}] windowed loss-median should fall on real data: {first_q:.5} -> {last_q:.5}"
    );
}

/// The adapter file's header metadata.
fn read_meta(path: &Path) -> HashMap<String, String> {
    let bytes = std::fs::read(path).unwrap();
    let (_, md) = safetensors::SafeTensors::read_metadata(&bytes).unwrap();
    md.metadata().clone().unwrap_or_default()
}

/// sc-26850 terminal acceptance: ONE invocation, eighteen total optimizer
/// updates including attribution/warmup. This is smoke/throughput evidence,
/// not a convergence claim. The existing historical tests remain separate.
#[test]
#[ignore = "model inference; explicit model-proof workflow"]
fn krea_bounded_acceleration_profile() {
    #[cfg(not(feature = "model-inference-proof"))]
    panic!("requires explicit model-inference-proof feature and registered CUDA lane");
    #[cfg(feature = "model-inference-proof")]
    {
        let graph_retaining_backward = candle_gen::candle_core::backprop::grad_do_not_detach();
        assert!(
            !graph_retaining_backward,
            "bounded Krea baseline requires the default first-order owners; graph-retaining backward is a separate composable contract"
        );
        let raw = std::env::var("KREA_RAW_DIR")
            .map(PathBuf::from)
            .expect("KREA_RAW_DIR required; Turbo is not a Raw timing baseline");
        let tmp = std::env::var_os("MODEL_PROOF_OUTPUT_DIR")
            .map(PathBuf::from)
            .expect("existing model-proof executor must bind its owned generated output directory")
            .join("krea_bounded_acceleration");
        let items = make_dataset(&tmp.join("data"));
        let mut cfg = config(NetworkType::Lora, 18, false);
        cfg.rank = 16;
        cfg.alpha = 16.;
        cfg.resolution = 1024;
        cfg.gradient_accumulation = 1;
        // Preserve the established production preview policy; the one acceptance
        // sample below is generated after save/reload/apply, outside step timing.
        cfg.sample_every = 500;
        let req = TrainingRequest {
            items,
            config: cfg,
            output_dir: tmp.join("out"),
            file_name: "krea_bounded.safetensors".into(),
            trigger_words: vec![],
            cancel: CancelFlag::new(),
        };
        let sampler = candle_gen::testkit::PeakSampler::start_rendered();
        let mut trainer = candle_gen_krea::provider_registry()
            .unwrap()
            .load_trainer(
                "krea_2_raw",
                &LoadSpec::new(WeightsSource::Dir(raw.clone())),
            )
            .unwrap();
        let mut losses = Vec::new();
        let out = candle_gen::train::profile::with_bounded_krea_profile(|| {
            trainer.train(&req, &mut |p| {
                if let TrainingProgress::Training { loss, .. } = p {
                    losses.push(loss);
                }
            })
        })
        .expect("bounded matched profile succeeds");
        drop(trainer);
        let peak_mib = sampler.stop();
        assert!(
            peak_mib > 0,
            "peak sampler must observe the selected physical GPU"
        );
        assert_eq!(out.steps, 18);
        assert_eq!(losses.len(), 18);
        assert!(losses.iter().all(|l| l.is_finite()));
        assert_eq!(
            read_meta(&out.adapter_path).get("rank").map(String::as_str),
            Some("16")
        );
        assert_eq!(
            adapter_keys(&out.adapter_path)
                .iter()
                .filter(|k| k.ends_with(".lora_A.weight"))
                .count(),
            112
        );
        // Production loader reopens the saved file and applies it to the Raw
        // snapshot. One sample only; no baseline/scale sweep or extra training.
        let gen = candle_gen_krea::provider_registry()
            .unwrap()
            .load(
                "krea_2_raw",
                &LoadSpec::new(WeightsSource::Dir(raw)).with_adapters(vec![AdapterSpec::new(
                    out.adapter_path.clone(),
                    1.,
                    AdapterKind::Lora,
                )]),
            )
            .expect("reload/apply saved adapter");
        let request = GenerationRequest {
            prompt: "a solid colour swatch number 0".into(),
            width: 1024,
            height: 1024,
            count: 1,
            seed: Some(7),
            steps: Some(28),
            ..Default::default()
        };
        let GenerationOutput::Images(images) = gen
            .generate(&request, &mut |_| {})
            .expect("one saved-adapter sample")
        else {
            panic!("expected images")
        };
        assert_eq!(images.len(), 1);
        assert_eq!(images[0].pixels.len(), 1024 * 1024 * 3);
        assert!(images[0].pixels.iter().any(|b| *b != 0));
        save_image_png(&tmp.join("adapter_sample.png"), &images[0]);
        // The existing proof publisher retains only explicitly declared files.
        // Keep the scalar run evidence beside the selected adapter/sample; the
        // complete per-step timings, attribution and go/no-go remain in the
        // independently retained body.stderr.log, where the driver emits them.
        let metrics = serde_json::json!({
            "schema_version": 1,
            "completed_optimizer_updates": out.steps,
            "graph_retaining_backward": graph_retaining_backward,
            "reference_baseline": "Repaired dense standard-softmax Reference with internal query tiles (at most 32Mi score elements, respecting smaller caller budgets), plus RMS/RoPE single autograd owners; common to matched segments. Original divide/reciprocal selection and F32 rounding boundaries retained. Not the original memory-unfit retained-intermediate graph.",
            "configuration": {
                "rank": req.config.rank,
                "alpha": req.config.alpha,
                "learning_rate": req.config.learning_rate,
                "optimizer": &req.config.optimizer,
                "resolution": req.config.resolution,
                "train_dtype": &req.config.train_dtype,
                "gradient_accumulation": req.config.gradient_accumulation,
                "gradient_checkpointing": req.config.gradient_checkpointing,
                "seed": req.config.seed,
                "sample_every": req.config.sample_every,
                "save_every": req.config.save_every,
                "complete_source_configuration_debug": format!("{:?}",req.config)
            },
            "losses_by_physical_update": losses,
            "selected_adapter_final_loss": out.final_loss,
            "peak_device_mib_training_and_preparation": peak_mib,
            "adapter": "out/krea_bounded.safetensors",
            "sample": "adapter_sample.png",
            "profile_log": "body.stderr.log",
            "profile_records": ["KREA_PROFILE_BOUNDARY", "KREA_PROFILE_STEP", "KREA_PROFILE_ATTRIBUTION", "KREA_PROFILE_OPTIMIZER", "KREA_PROFILE_RESULT"],
            "claim_limit": "Bounded smoke and throughput only; four steady updates per segment. Device peak includes preparation and is not a per-process allocation measurement. Provisioned snapshot path/layout does not attest weight integrity; actual input receipts and learned-device witnesses are separate."
        });
        std::fs::write(
            tmp.join("metrics.json"),
            serde_json::to_vec_pretty(&metrics).unwrap(),
        )
        .expect("write declared bounded-profile metrics artifact");
        println!(
            "KREA_PROFILE_ARTIFACT adapter={} peak_device_mib={peak_mib} sample={}",
            out.adapter_path.display(),
            tmp.join("adapter_sample.png").display()
        );
        println!("\nMODEL_INFERENCE_BODY_COMPLETED trainer_e2e::krea_bounded_acceleration_profile");
    }
}

/// Sorted tensor keys of the adapter file (via candle's loader).
fn adapter_keys(path: &Path) -> Vec<String> {
    let map = candle_gen::candle_core::safetensors::load(path, &Device::Cpu).unwrap();
    let mut keys: Vec<String> = map.into_keys().collect();
    keys.sort();
    keys
}

/// LoRA: trains + converges, writes a bare-dotted PEFT adapter over the 112-target attention surface,
/// with the expected metadata; and the same seed reproduces the same adapter bytes.
#[test]
#[ignore = "needs real Krea-2-Raw weights + a GPU; run with --features cuda --release --ignored"]
fn krea_lora_trains_and_is_well_formed() {
    if snapshot().is_none() {
        eprintln!("skipping: set KREA_RAW_DIR (or KREA_TURBO_DIR)");
        #[cfg(feature = "model-inference-proof")]
        crate::model_proof::missing_required_work("required work omitted at crates/media/candle-gen/candle-gen-krea/tests/trainer_e2e.rs:206 (krea_lora_trains_and_is_well_formed): Required selected learned asset/device path returns normally instead of running its neural/reference work; shape/report work in other paths does not satisfy this branch.");
        return;
    }
    let tmp = artifact_root().join("krea_trainer_e2e_lora");
    let out = run(&tmp, "krea_lora.safetensors", NetworkType::Lora, 24, false);
    assert_converged("lora", &out.losses);

    // 28 blocks × 4 attention projections = 112 targets; each writes lora_A/lora_B/alpha (3 tensors).
    let keys = adapter_keys(&out.adapter_path);
    let a = keys
        .iter()
        .filter(|k| k.ends_with(".lora_A.weight"))
        .count();
    let b = keys
        .iter()
        .filter(|k| k.ends_with(".lora_B.weight"))
        .count();
    assert_eq!(
        a, 112,
        "112 LoRA-A factors (28 blocks × 4 attn projs), got {a}"
    );
    assert_eq!(b, 112, "112 LoRA-B factors, got {b}");
    assert!(
        keys.iter().all(|k| k.starts_with("transformer_blocks.")),
        "DiT family adapters use bare-dotted keys (no base_model.model.unet. prefix)"
    );

    let meta = read_meta(&out.adapter_path);
    assert_eq!(meta.get("networkType").map(String::as_str), Some("lora"));
    assert_eq!(meta.get("rank").map(String::as_str), Some("8"));
    assert_eq!(
        meta.get("baseModel").map(String::as_str),
        Some("krea_2_raw")
    );
    assert_eq!(meta.get("family").map(String::as_str), Some("krea_2"));

    // Determinism: a second run at the same seed writes byte-identical factors.
    let out2 = run(
        &tmp,
        "krea_lora_2.safetensors",
        NetworkType::Lora,
        24,
        false,
    );
    let (b1, b2) = (
        std::fs::read(&out.adapter_path).unwrap(),
        std::fs::read(&out2.adapter_path).unwrap(),
    );
    assert_eq!(b1, b2, "same seed must produce a byte-identical adapter");

    #[cfg(feature = "model-inference-proof")]
    println!("\nMODEL_INFERENCE_BODY_COMPLETED trainer_e2e::krea_lora_trains_and_is_well_formed");
}

/// LoKr: trains + converges and writes a bare-keyed LoKr adapter with the LoKr metadata.
#[test]
#[ignore = "needs real Krea-2-Raw weights + a GPU; run with --features cuda --release --ignored"]
fn krea_lokr_trains_and_is_well_formed() {
    if snapshot().is_none() {
        eprintln!("skipping: set KREA_RAW_DIR (or KREA_TURBO_DIR)");
        #[cfg(feature = "model-inference-proof")]
        crate::model_proof::missing_required_work("required work omitted at crates/media/candle-gen/candle-gen-krea/tests/trainer_e2e.rs:262 (krea_lokr_trains_and_is_well_formed): Required selected learned asset/device path returns normally instead of running its neural/reference work; shape/report work in other paths does not satisfy this branch.");
        return;
    }
    let tmp = artifact_root().join("krea_trainer_e2e_lokr");
    let out = run(&tmp, "krea_lokr.safetensors", NetworkType::Lokr, 24, false);
    assert_converged("lokr", &out.losses);

    let keys = adapter_keys(&out.adapter_path);
    let w1 = keys.iter().filter(|k| k.ends_with(".lokr_w1")).count();
    assert_eq!(
        w1, 112,
        "112 LoKr w1 factors (28 blocks × 4 attn projs), got {w1}"
    );
    assert!(
        keys.iter().all(|k| k.starts_with("transformer_blocks.")),
        "bare-dotted LoKr keys"
    );
    let meta = read_meta(&out.adapter_path);
    assert_eq!(meta.get("networkType").map(String::as_str), Some("lokr"));
    assert_eq!(
        meta.get("baseModel").map(String::as_str),
        Some("krea_2_raw")
    );

    #[cfg(feature = "model-inference-proof")]
    println!("\nMODEL_INFERENCE_BODY_COMPLETED trainer_e2e::krea_lokr_trains_and_is_well_formed");
}

/// Gradient checkpointing: the memory-bounded backward converges + saves end-to-end on real weights.
#[test]
#[ignore = "needs real Krea-2-Raw weights + a GPU; run with --features cuda --release --ignored"]
fn krea_lora_gradient_checkpointing() {
    if snapshot().is_none() {
        eprintln!("skipping: set KREA_RAW_DIR (or KREA_TURBO_DIR)");
        #[cfg(feature = "model-inference-proof")]
        crate::model_proof::missing_required_work("required work omitted at crates/media/candle-gen/candle-gen-krea/tests/trainer_e2e.rs:292 (krea_lora_gradient_checkpointing): Required selected learned asset/device path returns normally instead of running its neural/reference work; shape/report work in other paths does not satisfy this branch.");
        return;
    }
    let tmp = artifact_root().join("krea_trainer_e2e_ckpt");
    let out = run(
        &tmp,
        "krea_lora_ckpt.safetensors",
        NetworkType::Lora,
        24,
        true,
    );
    assert_converged("lora-ckpt", &out.losses);
    assert_eq!(
        adapter_keys(&out.adapter_path)
            .iter()
            .filter(|k| k.ends_with(".lora_A.weight"))
            .count(),
        112
    );

    #[cfg(feature = "model-inference-proof")]
    println!("\nMODEL_INFERENCE_BODY_COMPLETED trainer_e2e::krea_lora_gradient_checkpointing");
}

/// sc-8650 regression: preview rendering at **production resolution** (1024², gradient-checkpoint, 28
/// sample-steps) must NOT OOM. The render runs **graph-free** — the driver freezes the adapter `Var`s to
/// detached snapshots for the denoise, so it uses inference-level memory (the same path 24 GB cards run).
/// Before that fix this OOM'd: the live `Var`s made the multi-step denoise retain a whole training-style
/// graph, so every full-res preview was best-effort-skipped (0 emitted). The 64px `*_emits_preview_samples`
/// smoke never accumulated enough graph to surface it — this guards the real production path.
#[test]
#[ignore = "needs real Krea-2-Raw weights + a GPU; run with --features cuda --release --ignored"]
fn krea_preview_full_resolution_no_oom() {
    if snapshot().is_none() {
        eprintln!("skipping: set KREA_RAW_DIR (or KREA_TURBO_DIR)");
        #[cfg(feature = "model-inference-proof")]
        crate::model_proof::missing_required_work("required work omitted at crates/media/candle-gen/candle-gen-krea/tests/trainer_e2e.rs:323 (krea_preview_full_resolution_no_oom): Required selected learned asset/device path returns normally instead of running its neural/reference work; shape/report work in other paths does not satisfy this branch.");
        return;
    }
    let tmp = artifact_root().join("krea_preview_1024");
    let items = make_dataset(&tmp);
    let mut trainer = candle_gen_krea::provider_registry()
        .unwrap()
        .load_trainer(
            "krea_2_raw",
            &LoadSpec::new(WeightsSource::Dir(snapshot().unwrap())),
        )
        .expect("krea_2_raw trainer registered");
    // Production-shaped: 1024² with gradient checkpointing (the 12B DiT forces it), 28 sample-steps.
    let mut cfg = config(NetworkType::Lora, 2, true);
    cfg.resolution = 1024;
    cfg.sample_every = 2;
    cfg.sample_steps = 28;
    cfg.sample_guidance_scale = 3.5;
    cfg.sample_prompts = vec![
        "a portrait photo".to_string(),
        "a landscape photo".to_string(),
    ];
    let req = TrainingRequest {
        items,
        config: cfg,
        output_dir: tmp.join("out"),
        file_name: "krea_1024.safetensors".to_string(),
        trigger_words: vec![],
        cancel: CancelFlag::new(),
    };
    let mut samples = 0u32;
    trainer
        .train(&req, &mut |p| {
            if let TrainingProgress::Sample { image, .. } = p {
                assert_eq!(
                    image.pixels.len(),
                    (image.width * image.height * 3) as usize,
                    "RGB8 row-major"
                );
                samples += 1;
            }
        })
        .expect("training runs");
    assert_eq!(
        samples, 2,
        "expected 2 full-resolution (1024²) previews without OOM (1 cadence × 2 prompts), got {samples}"
    );

    #[cfg(feature = "model-inference-proof")]
    println!("\nMODEL_INFERENCE_BODY_COMPLETED trainer_e2e::krea_preview_full_resolution_no_oom");
}

/// sc-8650: with a sample cadence set, the trainer renders preview images from the **in-progress
/// adapter** and emits them as [`TrainingProgress::Sample`] — each a valid, non-empty RGB8 bitmap. This
/// is the candle twin of the MLX sc-5637 `*_emits_preview_samples` smokes.
#[test]
#[ignore = "needs real Krea-2-Raw weights + a GPU; run with --features cuda --release --ignored"]
fn krea_trainer_emits_preview_samples() {
    if snapshot().is_none() {
        eprintln!("skipping: set KREA_RAW_DIR (or KREA_TURBO_DIR)");
        #[cfg(feature = "model-inference-proof")]
        crate::model_proof::missing_required_work("required work omitted at crates/media/candle-gen/candle-gen-krea/tests/trainer_e2e.rs:379 (krea_trainer_emits_preview_samples): Required selected learned asset/device path returns normally instead of running its neural/reference work; shape/report work in other paths does not satisfy this branch.");
        return;
    }
    let tmp = artifact_root().join("krea_trainer_e2e_samples");
    let items = make_dataset(&tmp);
    let mut trainer = candle_gen_krea::provider_registry()
        .unwrap()
        .load_trainer(
            "krea_2_raw",
            &LoadSpec::new(WeightsSource::Dir(snapshot().unwrap())),
        )
        .expect("krea_2_raw trainer registered");

    // 4 steps, render a preview every 2 steps over 2 prompts → 2 cadences × 2 prompts = 4 Sample events.
    let mut cfg = config(NetworkType::Lora, 4, false);
    cfg.sample_every = 2;
    cfg.sample_steps = 4;
    cfg.sample_guidance_scale = 1.0;
    cfg.sample_prompts = vec![
        "a solid red swatch".to_string(),
        "a solid blue swatch".to_string(),
    ];

    let req = TrainingRequest {
        items,
        config: cfg,
        output_dir: tmp.join("out"),
        file_name: "krea_samples.safetensors".to_string(),
        trigger_words: vec![],
        cancel: CancelFlag::new(),
    };

    let mut samples: Vec<(u32, u32, u32, String, Image)> = Vec::new();
    trainer
        .train(&req, &mut |p| {
            if let TrainingProgress::Sample {
                step,
                index,
                total,
                prompt,
                image,
            } = p
            {
                samples.push((step, index, total, prompt, image));
            }
        })
        .expect("training with sampling succeeds");

    assert_eq!(
        samples.len(),
        4,
        "expected 4 preview samples (2 cadences × 2 prompts), got {}",
        samples.len()
    );
    for (step, index, total, prompt, img) in &samples {
        assert!(
            *step == 2 || *step == 4,
            "preview at a cadence step, got {step}"
        );
        assert_eq!(*total, 2, "two prompts per cadence");
        assert!((1..=2).contains(index), "1-based prompt index, got {index}");
        assert!(img.width > 0 && img.height > 0, "non-empty preview dims");
        assert_eq!(
            img.pixels.len(),
            (img.width * img.height * 3) as usize,
            "RGB8 row-major: pixels.len() == w*h*3"
        );
        assert!(
            img.pixels.iter().any(|&b| b != 0),
            "preview from the in-progress adapter is not all-black (prompt {prompt:?})"
        );
        println!(
            "[krea sample] step {step} {index}/{total} {}x{} prompt {prompt:?}",
            img.width, img.height
        );
    }

    #[cfg(feature = "model-inference-proof")]
    println!("\nMODEL_INFERENCE_BODY_COMPLETED trainer_e2e::krea_trainer_emits_preview_samples");
}

// ── sc-7837: the real Raw→Turbo round trip (train on Raw → apply at Turbo inference) ──────────────

/// The Turbo render snapshot — `KREA_TURBO_DIR`, distinct from the Raw training snapshot ([`snapshot`]).
fn turbo_dir() -> Option<PathBuf> {
    std::env::var("KREA_TURBO_DIR").ok().map(PathBuf::from)
}

/// `n` copies of a vivid solid-colour swatch + a shared caption — a strong, low-frequency concept that
/// survives the train(64²)→render(1024²) resolution jump and the Raw→Turbo cross-apply, so the effect
/// is *visible* rather than a subtle texture shift.
fn make_concept_dataset(dir: &Path, rgb: [u8; 3], caption: &str, n: usize) -> Vec<TrainingItem> {
    let _ = std::fs::remove_dir_all(dir);
    std::fs::create_dir_all(dir).unwrap();
    (0..n)
        .map(|i| {
            let mut img = image::RgbImage::new(96, 96);
            for px in img.pixels_mut() {
                *px = image::Rgb(rgb);
            }
            let path = dir.join(format!("c{i}.png"));
            img.save(&path).unwrap();
            TrainingItem {
                image_path: path,
                caption: caption.to_string(),
                control_image_path: None,
                model_options: Default::default(),
                reference_image_paths: Vec::new(),
                subject_mask_path: None,
            }
        })
        .collect()
}

/// Train a LoRA/LoKr on the Raw snapshot through the registry with a custom dataset + config; returns
/// the written adapter path (asserts finite losses + that the file exists).
fn train_concept(
    items: Vec<TrainingItem>,
    cfg: TrainingConfig,
    out_dir: &Path,
    file_name: &str,
) -> PathBuf {
    let mut trainer = candle_gen_krea::provider_registry()
        .unwrap()
        .load_trainer(
            "krea_2_raw",
            &LoadSpec::new(WeightsSource::Dir(
                snapshot().expect("KREA_RAW_DIR / KREA_TURBO_DIR"),
            )),
        )
        .expect("krea_2_raw trainer registered");
    let req = TrainingRequest {
        items,
        config: cfg,
        output_dir: out_dir.to_path_buf(),
        file_name: file_name.to_string(),
        trigger_words: vec![],
        cancel: CancelFlag::new(),
    };
    let mut losses: Vec<f32> = Vec::new();
    let out = trainer
        .train(&req, &mut |p| {
            if let TrainingProgress::Training { loss, .. } = p {
                losses.push(loss);
            }
        })
        .expect("training succeeds");
    assert!(losses.iter().all(|l| l.is_finite()), "no NaN/Inf losses");
    assert!(out.adapter_path.exists(), "adapter written");
    println!(
        "[train_concept {file_name}] {} steps, last loss {:.5}",
        losses.len(),
        losses.last().copied().unwrap_or(f32::NAN)
    );
    out.adapter_path
}

/// Render one 1024² image at `krea_2_turbo` with `adapters` merged into the DiT (the sc-7836 path).
fn render_turbo(root: &Path, prompt: &str, adapters: Vec<AdapterSpec>) -> Image {
    let spec = LoadSpec::new(WeightsSource::Dir(root.to_path_buf())).with_adapters(adapters);
    let gen = candle_gen_krea::provider_registry()
        .unwrap()
        .load("krea_2_turbo", &spec)
        .expect("load krea_2_turbo");
    let req = GenerationRequest {
        prompt: prompt.into(),
        width: 1024,
        height: 1024,
        count: 1,
        seed: Some(0),
        steps: Some(8),
        ..Default::default()
    };
    let GenerationOutput::Images(mut imgs) = gen.generate(&req, &mut |_| {}).expect("generate")
    else {
        panic!("expected images");
    };
    imgs.remove(0)
}

fn save_png(img: &Image, name: &str) {
    let path = artifact_root()
        .join("krea_roundtrip")
        .join(format!("{name}.png"));
    save_image_png(&path, img);
    println!("  saved {}", path.display());
}

/// Mean of one RGB channel (`off` = 0/1/2 → R/G/B) over an RGB8 buffer.
fn mean_channel(px: &[u8], off: usize) -> f64 {
    let sum: f64 = px.iter().skip(off).step_by(3).map(|&v| v as f64).sum();
    sum / (px.len() as f64 / 3.0)
}

/// The sc-7837 round trip on **real** weights: train a LoRA on Krea-2-Raw (a vivid green concept), then
/// apply it at `krea_2_turbo` inference. Asserts the trained adapter merges across the full 112-target
/// surface (nothing skipped — the `assert_reloads` mirror), scale-0 ≡ base byte-exact, and a non-zero
/// scale yields a coherent, differing render; logs the green-channel shift and saves PNGs for the
/// "visibly adapted" eyeball (the qualitative half of the AC).
#[test]
#[ignore = "needs Krea-2-Raw (train) + Krea-2-Turbo (render) + GPU; --features cuda --release"]
fn krea_lora_roundtrip_raw_to_turbo() {
    let (Some(_raw), Some(turbo)) = (snapshot(), turbo_dir()) else {
        eprintln!("skipping: set KREA_RAW_DIR and KREA_TURBO_DIR");
        #[cfg(feature = "model-inference-proof")]
        crate::model_proof::missing_required_work("required work omitted at crates/media/candle-gen/candle-gen-krea/tests/trainer_e2e.rs:588 (krea_lora_roundtrip_raw_to_turbo): Required selected learned asset/device path returns normally instead of running its neural/reference work; shape/report work in other paths does not satisfy this branch.");
        return;
    };
    let tmp = artifact_root().join("krea_rt");

    // 1) Train a deliberately over-fit LoRA on a vivid emerald-green concept. **Gradient checkpointing
    //    on** — the retained backward OOMs on the 12B DiT (~24 GB resident leaves no VRAM headroom for
    //    the full activation graph); the checkpointed backward is memory-bounded.
    let items = make_concept_dataset(
        &tmp.join("data"),
        [25, 200, 70],
        "a vivid emerald green field",
        4,
    );
    let mut cfg = config(NetworkType::Lora, 150, true);
    cfg.learning_rate = 1.5e-3;
    let adapter = train_concept(items, cfg, &tmp.join("out"), "krea_green.safetensors");

    // 2) The trained adapter merges across the full attention surface of the real Turbo DiT.
    let tcfg = Krea2Config::from_snapshot(&turbo).expect("turbo config");
    let mut w = Weights::from_dir(&turbo.join("transformer"), &Device::Cpu, DType::BF16).unwrap();
    let report = merge_into_weights(
        &mut w,
        &tcfg,
        &[AdapterSpec::new(adapter.clone(), 1.0, AdapterKind::Lora)],
    )
    .expect("merge trained adapter");
    assert_eq!(report.merged, 112, "every trained target merges");
    assert_eq!(report.skipped_keys, 0, "nothing skipped");

    // 3) Render base / scale-0 / scale-1 at Turbo (same prompt + seed isolates the LoRA's effect).
    let prompt = "a photograph of a city street at noon";
    let base = render_turbo(&turbo, prompt, vec![]);
    let zero = render_turbo(
        &turbo,
        prompt,
        vec![AdapterSpec::new(adapter.clone(), 0.0, AdapterKind::Lora)],
    );
    assert_eq!(
        base.pixels, zero.pixels,
        "scale-0 adapter ≡ base render (byte-exact)"
    );

    let adapted = render_turbo(
        &turbo,
        prompt,
        vec![AdapterSpec::new(adapter.clone(), 1.0, AdapterKind::Lora)],
    );
    assert_eq!((adapted.width, adapted.height), (1024, 1024), "output dims");
    let diff = base
        .pixels
        .iter()
        .zip(&adapted.pixels)
        .filter(|(a, b)| a != b)
        .count();
    let (gb, ga) = (
        mean_channel(&base.pixels, 1),
        mean_channel(&adapted.pixels, 1),
    );
    let (rb, ra) = (
        mean_channel(&base.pixels, 0),
        mean_channel(&adapted.pixels, 0),
    );
    println!(
        "[roundtrip] changed_px={diff}/{} green {gb:.1}->{ga:.1} red {rb:.1}->{ra:.1}",
        base.pixels.len()
    );
    save_png(&base, "rt_base");
    save_png(&adapted, "rt_adapter");
    assert!(
        diff > base.pixels.len() / 100,
        "a trained adapter must change the render"
    );

    #[cfg(feature = "model-inference-proof")]
    println!("\nMODEL_INFERENCE_BODY_COMPLETED trainer_e2e::krea_lora_roundtrip_raw_to_turbo");
}

#[cfg(feature = "model-inference-proof")]
fn bounded_root() -> std::path::PathBuf {
    let root = std::path::PathBuf::from(
        std::env::var_os("MODEL_PROOF_OUTPUT_DIR").expect("manual executor output directory"),
    )
    .join("bespoke_acceptance");
    std::fs::create_dir_all(&root).unwrap();
    root
}
#[cfg(feature = "model-inference-proof")]
fn bounded_runtime() -> std::path::PathBuf {
    let root = std::path::PathBuf::from(
        std::env::var_os("BESPOKE_RUNTIME_DIR")
            .expect("source-owned persistent runtime checkpoint directory"),
    );
    assert!(root.is_absolute());
    assert!(
        root.join(".sc-26850-owned-runtime").is_file(),
        "asset setup owns this runtime directory"
    );
    root
}
#[cfg(feature = "model-inference-proof")]
fn bounded_peak(sampler: candle_gen::testkit::PeakSampler, root: &std::path::Path) {
    let peak = sampler.stop();
    assert!(peak > 0);
    std::fs::write(
        root.join("peak_device_bytes.txt"),
        (peak * 1024 * 1024).to_string(),
    )
    .unwrap();
}
#[cfg(feature = "model-inference-proof")]
fn bounded_export_receipt(root: &std::path::Path, path: &std::path::Path) {
    assert!(
        path.starts_with(bounded_runtime()),
        "actual export stays in the owned runtime area"
    );
    let receipt = serde_json::json!({"actual_export":path, "bytes":std::fs::metadata(path).unwrap().len(),
        "sha256":candle_gen::testkit::sha256_file(path), "runtime_directory":bounded_runtime(),
        "publication":"complete model/control checkpoint retained in legitimate runtime storage; model-weight publication ban preserved",
        "reload":"same exported path reopened/applied by production reader before sample; see actual reader receipts"});
    std::fs::write(
        root.join("retained_export.json"),
        serde_json::to_vec_pretty(&receipt).unwrap(),
    )
    .unwrap();
}

#[test]
#[cfg(feature = "model-inference-proof")]
#[ignore = "model inference; explicit model-proof workflow"]
fn krea_control_bounded_acceleration_acceptance() {
    use candle_gen::gen_core::OffloadPolicy;
    use candle_gen_krea::{Krea2Control, Krea2ControlPaths, Krea2ControlRequest};
    let root = bounded_root();
    let snapshot = PathBuf::from(
        std::env::var_os("KREA_CONTROL_BASE_DIR").expect("pinned native Turbo snapshot"),
    );
    let input = root.join("input.png");
    image::RgbImage::from_pixel(64, 64, image::Rgb([80, 140, 210]))
        .save(&input)
        .unwrap();
    let control = root.join("control.png");
    // Nonconstant, aligned control pixels are encoded by the production VAE.
    image::RgbImage::from_fn(64, 64, |x, y| {
        image::Rgb([(x * 4) as u8, (y * 4) as u8, 80])
    })
    .save(&control)
    .unwrap();
    let mut item = TrainingItem::captioned(input, "a blue tile following the control".into());
    item.control_image_path = Some(control);
    let sampler = candle_gen::testkit::PeakSampler::start_rendered();
    let output = candle_gen::train::profile::with_bounded_cohort_for(
        &root.join("metrics.json"),
        &["krea-control:active-item-conditioning-hit"],
        || {
            let mut trainer = candle_gen_krea::provider_registry()
                .unwrap()
                .load_trainer(
                    "krea_2_control",
                    &LoadSpec::new(WeightsSource::Dir(snapshot.clone())),
                )
                .unwrap();
            let output = trainer
                .train(
                    &TrainingRequest {
                        items: vec![item],
                        config: TrainingConfig {
                            control_type: Some("pose".into()),
                            learning_rate: 1e-5,
                            steps: 4,
                            resolution: 64,
                            batch_size: 1,
                            gradient_accumulation: 1,
                            lr_warmup_steps: 0,
                            save_every: 0,
                            seed: 7,
                            train_dtype: "bf16".into(),
                            gradient_checkpointing: true,
                            ..Default::default()
                        },
                        output_dir: bounded_runtime().join("control"),
                        file_name: "control.safetensors".into(),
                        trigger_words: vec![],
                        cancel: CancelFlag::new(),
                    },
                    &mut |_| {},
                )
                .unwrap();
            assert_eq!(output.steps, 4);
            assert!(output.final_loss.is_finite());
            output
        },
    );
    bounded_peak(sampler, &root);
    let generator = Krea2Control::load(&Krea2ControlPaths {
        root: snapshot,
        convrot_dit: None,
        native_dit: None,
        control: output.adapter_path.clone(),
        adapters: vec![],
        branch_tier: None,
        chunk_attention: false,
        offload_policy: OffloadPolicy::default(),
    })
    .unwrap();
    let control_image = Image {
        width: 256,
        height: 256,
        pixels: image::RgbImage::from_fn(256, 256, |x, y| image::Rgb([x as u8, y as u8, 80]))
            .into_raw(),
    };
    let sample = generator
        .generate(
            &Krea2ControlRequest {
                prompt: "a blue tile following the control".into(),
                width: 256,
                height: 256,
                steps: 4,
                seed: 7,
                ..Default::default()
            },
            &control_image,
            &mut |_| {},
        )
        .unwrap();
    assert_eq!(sample.pixels.len(), 256 * 256 * 3);
    assert!(sample.pixels.iter().any(|v| *v != 0));
    candle_gen::testkit::save_image_png(&root.join("adapter_sample.png"), &sample);
    bounded_export_receipt(&root, &output.adapter_path);
    println!("\nMODEL_INFERENCE_BODY_COMPLETED trainer_e2e::krea_control_bounded_acceleration_acceptance");
}
