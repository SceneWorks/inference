//! sc-3044 e2e — the production `ZImageTurboTrainer` (the `Trainer` contract realized on Z-Image),
//! driven through the explicit provider registry.
//!
//! `#[ignore]`d — needs the real `Tongyi-MAI/Z-Image-Turbo` weights in the HF cache (or
//! `ZIMAGE_SNAPSHOT`). Run:
//!   cargo test -p mlx-gen-z-image --release --test integration trainer_e2e:: -- --ignored --nocapture
//!
//! Proves the full prepare→load→cache→train→save lifecycle: a tiny captioned PNG dataset is
//! VAE/text-encoded and cached, AdamW training drives the flow-match loss down, and a PEFT adapter
//! is written that reloads through the real inference path (`apply_z_image_adapters`) onto a fresh
//! transformer — the round-trip contract, for both the LoRA and LoKr adapter kinds.

use std::path::Path;

use mlx_gen::weights::Weights;
use mlx_gen::{
    AdapterKind, AdapterSpec, CancelFlag, LoadSpec, NetworkType, TrainingConfig, TrainingItem,
    TrainingProgress, TrainingRequest, WeightsSource,
};

use crate::common;
use common::snapshot;

/// Two captioned PNGs in `dir` for a run at bucket edge `edge`: item 0 a 2:3 portrait of three
/// distinct horizontal bands (blue / cream / green), item 1 a solid red square swatch, so every run
/// trains a non-square aspect-preserving canvas whose whole frame (all three bands, never a
/// centre-cropped square) is visible in its latents — and the previews, which render at item 0's
/// canvas, are themselves portrait. The square side is `edge` (never below the historical 96 px)
/// and the portrait is that width × 1.5: the canvas never upscales a source (SW-7172), so a fixture
/// smaller than `edge²` would silently train a smaller workload than the run claims.
fn make_dataset(dir: &Path, edge: u32) -> Vec<TrainingItem> {
    let _ = std::fs::remove_dir_all(dir);
    std::fs::create_dir_all(dir).unwrap();
    let side = edge.max(96);
    let fixtures: [(u32, u32, &[[u8; 3]], &str); 2] = [
        (
            side,
            side * 3 / 2,
            &[[40, 80, 200], [235, 225, 200], [40, 160, 70]],
            "a blue, cream and green banded swatch",
        ),
        (side, side, &[[200, 40, 40]], "a solid red colour swatch"),
    ];
    let mut items = Vec::new();
    for (i, (w, h, bands, caption)) in fixtures.into_iter().enumerate() {
        let img = image::RgbImage::from_fn(w, h, |_, y| {
            image::Rgb(bands[(y as usize * bands.len()) / h as usize])
        });
        let path = dir.join(format!("img{i}.png"));
        img.save(&path).unwrap();
        items.push(TrainingItem {
            image_path: path,
            caption: format!("{caption} number {i}"),
            control_image_path: None,
            model_options: Default::default(),
            reference_image_paths: Vec::new(),
            subject_mask_path: None,
        });
    }
    items
}

/// Save preview `n` under the model-proof executor's owned output directory, where the registered
/// row declares it as an `output_artifacts` entry so the proof artifact carries the rendered
/// sample for inspection (SW-7172). A manual run outside the executor writes nothing.
fn save_proof_preview(n: usize, preview: &mlx_gen::Image) {
    if let Some(dir) = std::env::var_os("MODEL_PROOF_OUTPUT_DIR") {
        let path = Path::new(&dir).join(format!("preview_{n}.png"));
        image::save_buffer(
            path,
            &preview.pixels,
            preview.width,
            preview.height,
            image::ColorType::Rgb8,
        )
        .unwrap();
    }
}

#[test]
#[ignore = "model inference; explicit model-proof workflow; needs real Z-Image weights"]
fn z_image_trainer_trains_and_writes_adapter() {
    // Tiny synthetic dataset: two solid-colour swatches + captions, written as PNGs.
    let tmp_guard = tempfile::tempdir().unwrap();
    let tmp = tmp_guard.path().to_path_buf();
    let items = make_dataset(&tmp, 64);

    assert_eq!(mlx_gen_z_image::MODEL_ID, "z_image_turbo");

    // Load the trainer through the explicit provider registry.
    let mut trainer = mlx_gen_z_image::provider_registry()
        .unwrap()
        .load_trainer(
            "z_image_turbo",
            &LoadSpec::new(WeightsSource::Dir(snapshot())),
        )
        .expect("z_image_turbo trainer should be registered");

    let config = TrainingConfig {
        rank: 8,
        alpha: 8.0,
        learning_rate: 1e-3,
        steps: 24,
        resolution: 64, // bucketed to 64 -> 8x8 latent, fast
        save_every: 0,
        seed: 7,
        ..Default::default()
    };
    let req = TrainingRequest {
        items,
        config,
        output_dir: tmp.join("out"),
        file_name: "swatch_lora.safetensors".to_string(),
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

    // --- lifecycle (the integration concern this e2e owns) ---
    assert_eq!(cached, 2, "both dataset items should be cached");
    assert_eq!(out.steps, 24, "all micro-steps should run");
    assert_eq!(losses.len(), 24);
    assert!(
        losses.iter().all(|l| l.is_finite()),
        "no NaN/Inf losses (not diverging)"
    );

    // Each step samples a fresh sigma+noise from the sigmoid distribution, so per-step loss is
    // dominated by sigma variance, not a monotonic curve. As an efficacy signal we compare the mean
    // of the first vs last quarter of the run (sigma noise averages out).
    let q = losses.len() / 4;
    let mean = |s: &[f32]| s.iter().sum::<f32>() / s.len() as f32;
    let (first_q, last_q) = (mean(&losses[..q]), mean(&losses[losses.len() - q..]));
    println!(
        "[trainer] cached {cached}; steps {}; loss-mean first-quarter {first_q:.5} -> last-quarter {last_q:.5}",
        out.steps
    );
    // Deterministic (fixed seed/dataset): the windowed mean must drop — real data does train.
    assert!(
        last_q < first_q * 0.8,
        "windowed loss-mean should fall on real data: {first_q:.5} -> {last_q:.5}"
    );

    // The produced adapter carries the PEFT keys + inference-reload metadata.
    assert!(out.adapter_path.exists(), "adapter file should be written");
    let w = Weights::from_file(&out.adapter_path).unwrap();
    assert_eq!(w.metadata("networkType"), Some("lora"));
    assert!(
        w.keys()
            .any(|k| k == "layers.0.attention.to_q.lora_A.weight"),
        "adapter should contain PEFT-keyed LoRA factors"
    );
    // Suffix-match over the DiT covers attention in the main layers AND the refiner stacks (matching
    // PEFT's target_modules suffix match) — 30 layers + 2 noise + 2 context refiners, ×4 projections.
    let n_targets = w.keys().filter(|k| k.ends_with(".lora_A.weight")).count();
    assert_eq!(
        n_targets, 136,
        "attention across layers + noise/context refiners"
    );

    // Round-trip contract (was the sc-3042 spike's job, now proven on the production adapter): free
    // the trainer's model, then reload the adapter through the REAL inference path
    // (`apply_z_image_adapters`) onto a fresh transformer. The loader applies the same
    // `(transpose, alpha/rank fold, scale=1)` the trainer injected, so the reloaded delta matches
    // what training used by construction; here we confirm every saved target resolves + forwards finite.
    let adapter_path = out.adapter_path.clone();
    drop(trainer);
    let mut t = mlx_gen_z_image::load_transformer(&snapshot()).unwrap();
    let report = mlx_gen_z_image::apply_z_image_adapters(
        &mut t,
        &[AdapterSpec {
            path: adapter_path,
            scale: 1.0,
            kind: AdapterKind::Lora,
            pass_scales: None,
            moe_expert: None,
        }],
    )
    .expect("LoRA adapter should reload through the inference path");
    assert_eq!(report.applied, n_targets, "every saved LoRA target reloads");
    assert!(
        report.unmatched_paths.is_empty(),
        "every LoRA target should resolve"
    );

    // A forward with the reloaded LoRA installed produces a finite velocity.
    let x = mlx_rs::random::normal::<f32>(
        &[16, 1, 16, 16],
        None,
        None,
        Some(&mlx_rs::random::key(1).unwrap()),
    )
    .unwrap();
    let cap = mlx_rs::random::normal::<f32>(
        &[8, 2560],
        None,
        None,
        Some(&mlx_rs::random::key(2).unwrap()),
    )
    .unwrap();
    let v = t.forward(&x, 0.5, &cap).unwrap();
    assert!(
        v.sum(None).unwrap().item::<f32>().is_finite(),
        "reloaded LoRA forward should be finite"
    );
    println!(
        "[trainer] e2e OK — adapter written to {}, reloaded onto {n_targets} targets, forward finite",
        out.adapter_path.display()
    );
    #[cfg(target_os = "macos")]
    crate::common::metal_guard::record_model_inference_device();
    eprintln!(
        "\nMODEL_INFERENCE_BODY_COMPLETED trainer_e2e::z_image_trainer_trains_and_writes_adapter"
    );
}

#[test]
#[ignore = "model inference; explicit model-proof workflow; needs real Z-Image weights"]
fn z_image_trainer_lokr_trains_and_reloads() {
    let tmp_guard = tempfile::tempdir().unwrap();
    let tmp = tmp_guard.path().to_path_buf();
    let items = make_dataset(&tmp, 64);
    assert_eq!(mlx_gen_z_image::MODEL_ID, "z_image_turbo");

    let mut trainer = mlx_gen_z_image::provider_registry()
        .unwrap()
        .load_trainer(
            "z_image_turbo",
            &LoadSpec::new(WeightsSource::Dir(snapshot())),
        )
        .expect("z_image_turbo trainer should be registered");

    let config = TrainingConfig {
        rank: 8,
        alpha: 8.0,
        learning_rate: 1e-3,
        steps: 24,
        resolution: 64,
        save_every: 0,
        seed: 7,
        network_type: NetworkType::Lokr, // <-- LyCORIS Kronecker adapter
        decompose_factor: -1,            // balanced/auto
        ..Default::default()
    };
    let req = TrainingRequest {
        items,
        config,
        output_dir: tmp.join("out"),
        file_name: "swatch_lokr.safetensors".to_string(),
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
        .expect("LoKr training should succeed");

    assert_eq!(out.steps, 24);
    assert!(
        losses.iter().all(|l| l.is_finite()),
        "no NaN/Inf (kron autograd is sane)"
    );
    let q = losses.len() / 4;
    let mean = |s: &[f32]| s.iter().sum::<f32>() / s.len() as f32;
    let (first_q, last_q) = (mean(&losses[..q]), mean(&losses[losses.len() - q..]));
    println!("[trainer-lokr] loss-mean first-quarter {first_q:.5} -> last-quarter {last_q:.5}");
    // Grad flows through the Kronecker reconstruction → the windowed mean must fall.
    assert!(
        last_q < first_q * 0.8,
        "LoKr windowed loss-mean should fall: {first_q:.5} -> {last_q:.5}"
    );

    // Adapter carries LoKr keys + metadata; one `lokr_w1` per trained target.
    let w = Weights::from_file(&out.adapter_path).unwrap();
    assert_eq!(w.metadata("networkType"), Some("lokr"));
    assert!(w.metadata("decomposeFactor").is_some());
    assert!(
        w.keys().any(|k| k == "layers.0.attention.to_q.lokr_w1"),
        "adapter should contain LoKr factor keys"
    );
    // Suffix-match over the DiT covers attention in the main layers AND the refiner stacks (matching
    // PEFT's target_modules suffix match) — 30 layers + 2 noise + 2 context refiners, x4 projections.
    let n_targets = w.keys().filter(|k| k.ends_with(".lokr_w1")).count();
    assert_eq!(
        n_targets, 136,
        "attention across layers + noise/context refiners"
    );

    // Round-trip contract: free the trainer's model, then reload the adapter through the REAL
    // inference path (`apply_z_image_adapters` → parse_lokr → reconstruct_lokr_delta) onto a fresh
    // transformer, and confirm the reconstructed delta installs + forwards finite (the bf16-exact
    // reconstruction matches what training used, by construction).
    let adapter_path = out.adapter_path.clone();
    drop(trainer);
    let mut t = mlx_gen_z_image::load_transformer(&snapshot()).unwrap();
    let report = mlx_gen_z_image::apply_z_image_adapters(
        &mut t,
        &[AdapterSpec {
            path: adapter_path,
            scale: 1.0,
            kind: AdapterKind::Lokr,
            pass_scales: None,
            moe_expert: None,
        }],
    )
    .expect("LoKr adapter should reload through the inference path");
    assert_eq!(report.applied, n_targets, "every saved LoKr target reloads");
    assert!(
        report.unmatched_paths.is_empty(),
        "every LoKr target should resolve"
    );

    // A forward with the reloaded LoKr installed produces a finite velocity.
    let x = mlx_rs::random::normal::<f32>(
        &[16, 1, 16, 16],
        None,
        None,
        Some(&mlx_rs::random::key(1).unwrap()),
    )
    .unwrap();
    let cap = mlx_rs::random::normal::<f32>(
        &[8, 2560],
        None,
        None,
        Some(&mlx_rs::random::key(2).unwrap()),
    )
    .unwrap();
    let v = t.forward(&x, 0.5, &cap).unwrap();
    let s = v.sum(None).unwrap().item::<f32>();
    assert!(s.is_finite(), "reloaded LoKr forward should be finite");
    println!(
        "[trainer-lokr] e2e OK — reloaded LoKr applied to {n_targets} targets, forward finite"
    );
    #[cfg(target_os = "macos")]
    crate::common::metal_guard::record_model_inference_device();
    eprintln!(
        "\nMODEL_INFERENCE_BODY_COMPLETED trainer_e2e::z_image_trainer_lokr_trains_and_reloads"
    );
}

/// sc-5637 — preview samples. A short run with `sample_every`/`sample_prompts` set must emit
/// `TrainingProgress::Sample` events at the cadence, each carrying a real decoded RGB bitmap
/// (`width·height·3` non-empty pixels) rendered from the in-progress adapter. Proves the engine
/// render path (install in-progress adapter → txt2img denoise → VAE decode → `Image`) works on real
/// weights — the part the worker persists as a project asset. A 512 edge keeps the saved previews
/// inspectable while the per-cadence inference stays cheap. Run:
///   cargo test -p mlx-gen-z-image --release --test integration -- --ignored --nocapture trainer_e2e::z_image_trainer_emits_preview_samples
#[test]
#[ignore = "model inference; explicit model-proof workflow; needs real Z-Image weights"]
fn z_image_trainer_emits_preview_samples() {
    let tmp_guard = tempfile::tempdir().unwrap();
    let tmp = tmp_guard.path().to_path_buf();
    // A 512 edge (not the 64 px smoke edge) so the saved previews are inspectable: item 0's
    // 512×768 portrait trains — and previews render at — its whole 2:3 canvas (SW-7172).
    let items = make_dataset(&tmp, 512);
    assert_eq!(mlx_gen_z_image::MODEL_ID, "z_image_turbo");

    let mut trainer = mlx_gen_z_image::provider_registry()
        .unwrap()
        .load_trainer(
            "z_image_turbo",
            &LoadSpec::new(WeightsSource::Dir(snapshot())),
        )
        .expect("z_image_turbo trainer should be registered");

    let config = TrainingConfig {
        rank: 8,
        alpha: 8.0,
        learning_rate: 1e-3,
        steps: 8,
        resolution: 512,
        save_every: 0,
        seed: 7,
        // Sample at steps 4 and 8 (two cadences) over two prompts.
        sample_every: 4,
        sample_steps: 4,
        sample_guidance_scale: 1.0,
        sample_prompts: vec![
            "a blue, cream and green banded swatch".to_string(),
            "a solid red colour swatch".to_string(),
        ],
        ..Default::default()
    };
    let req = TrainingRequest {
        items,
        config,
        output_dir: tmp.join("out"),
        file_name: "swatch_lora.safetensors".to_string(),
        trigger_words: vec![],
        cancel: CancelFlag::new(),
    };

    let mut samples: Vec<(u32, u32, u32, String, usize)> = Vec::new(); // (step, index, total, prompt, pixels)
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
                samples.push((step, index, total, prompt, image.pixels.len()));
                // Every preview must be a real, correctly-sized RGB bitmap.
                assert!(
                    image.width > 0 && image.height > 0,
                    "preview has dimensions"
                );
                assert_eq!(
                    image.pixels.len(),
                    (image.width * image.height * 3) as usize,
                    "preview is a packed RGB8 buffer"
                );
                assert!(
                    image.pixels.iter().any(|&b| b != 0),
                    "preview is not an all-black/empty frame"
                );
                // Previews render at item 0's canvas: the 2:3 portrait, whole frame (SW-7172).
                assert!(
                    image.height > image.width,
                    "preview renders at item 0's portrait canvas, got {}x{}",
                    image.width,
                    image.height
                );
                save_proof_preview(samples.len() - 1, &image);
            }
        })
        .expect("training with sampling should succeed");

    // Two cadences (steps 4 and 8) × two prompts = four previews.
    assert_eq!(
        samples.len(),
        4,
        "expected 4 preview samples, got {samples:?}"
    );
    assert!(
        samples.iter().any(|(step, ..)| *step == 4) && samples.iter().any(|(step, ..)| *step == 8),
        "previews fire at the configured cadence (steps 4 and 8): {samples:?}"
    );
    assert!(
        samples.iter().all(|(_, _, total, ..)| *total == 2),
        "each cadence reports 2 prompts"
    );
    println!(
        "[trainer-samples] e2e OK — {} previews: {samples:?}",
        samples.len()
    );
    #[cfg(target_os = "macos")]
    crate::common::metal_guard::record_model_inference_device();
    eprintln!(
        "\nMODEL_INFERENCE_BODY_COMPLETED trainer_e2e::z_image_trainer_emits_preview_samples"
    );
}
