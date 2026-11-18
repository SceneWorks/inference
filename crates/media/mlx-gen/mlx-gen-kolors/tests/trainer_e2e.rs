//! sc-4568 e2e — the production `KolorsTrainer` (the `Trainer` contract realized on the Kolors
//! U-Net), driven through the explicit provider registry.
//!
//! `#[ignore]`d — needs the real `Kwai-Kolors/Kolors-diffusers` snapshot in the HF cache (or
//! `KOLORS_SNAPSHOT`), with the materialized `tokenizer/tokenizer.json`. Run:
//!   cargo test -p mlx-gen-kolors --release --test integration trainer_e2e:: -- --ignored --nocapture
//!
//! Proves the full prepare→load→cache→train→save lifecycle: a tiny captioned PNG dataset is
//! VAE/ChatGLM3-encoded and cached, AdamW training drives the epsilon flow down, and an adapter is
//! written that reloads through the REAL SDXL inference adapter path (`apply_sdxl_adapters[_with]`)
//! onto a fresh Kolors-loaded U-Net (Kolors' U-Net == the SDXL `UNet2DConditionModel`) — merging into
//! every trained target and forwarding finite under Kolors conditioning shapes.

use crate::common;

use std::path::{Path, PathBuf};

use mlx_gen::weights::Weights;
use mlx_gen::{
    AdapterKind, AdapterSpec, CancelFlag, LoadSpec, NetworkType, TrainingConfig, TrainingItem,
    TrainingProgress, TrainingRequest, WeightsSource,
};
use mlx_gen_sdxl::{
    apply_sdxl_adapters, apply_sdxl_adapters_with, load_unet_kolors_dtype, LoraCoverage,
};
use mlx_rs::{Array, Dtype};

/// The Kolors SDXL-style micro-conditioning `time_ids = (H, W, 0, 0, H, W)` (mirrors the crate's
/// `pub(crate) model::kolors_time_ids`, which an integration test can't reach).
fn kolors_time_ids(batch: i32, height: i32, width: i32) -> Array {
    let (h, w) = (height as f32, width as f32);
    let row = [h, w, 0.0, 0.0, h, w];
    let mut v = Vec::with_capacity(batch as usize * 6);
    for _ in 0..batch {
        v.extend_from_slice(&row);
    }
    Array::from_slice(&v, &[batch, 6])
}

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

/// A tiny config: small rank, low resolution (bucketed to 64 → 8×8 latent), few steps.
fn config(network_type: NetworkType, gradient_checkpointing: bool) -> TrainingConfig {
    TrainingConfig {
        rank: 8,
        alpha: 8.0,
        learning_rate: 1e-3,
        steps: 16,
        resolution: 64,
        save_every: 0,
        seed: 7,
        network_type,
        decompose_factor: -1,
        gradient_checkpointing,
        ..Default::default()
    }
}

fn run(tmp: &Path, file_name: &str, network_type: NetworkType) -> (Vec<f32>, u32, PathBuf) {
    run_cfg(tmp, file_name, network_type, false)
}

fn run_cfg(
    tmp: &Path,
    file_name: &str,
    network_type: NetworkType,
    gradient_checkpointing: bool,
) -> (Vec<f32>, u32, PathBuf) {
    let cfg = config(network_type, gradient_checkpointing);
    let items = make_dataset(tmp, cfg.resolution);
    assert_eq!(mlx_gen_kolors::MODEL_ID, "kolors");

    let mut trainer = mlx_gen_kolors::provider_registry()
        .unwrap()
        .load_trainer("kolors", &LoadSpec::new(WeightsSource::Dir(snapshot())))
        .expect("kolors trainer should be registered");

    let req = TrainingRequest {
        items,
        config: cfg,
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
    assert_eq!(out.steps, 16, "all micro-steps should run");
    assert_eq!(losses.len(), 16);
    assert!(
        losses.iter().all(|l| l.is_finite()),
        "no NaN/Inf losses (not diverging)"
    );
    // Each step samples a fresh integer timestep + noise, so per-step loss is dominated by timestep
    // variance, not a monotonic curve. Compare the first vs last quarter mean (variance averages out).
    let q = losses.len() / 4;
    let mean = |s: &[f32]| s.iter().sum::<f32>() / s.len() as f32;
    let (first_q, last_q) = (mean(&losses[..q]), mean(&losses[losses.len() - q..]));
    println!(
        "[kolors-{network_type:?}] cached {cached}; steps {}; loss-mean first-quarter {first_q:.5} -> last-quarter {last_q:.5}",
        out.steps
    );
    assert!(
        last_q < first_q * 0.9,
        "windowed loss-mean should fall on real data: {first_q:.5} -> {last_q:.5}"
    );
    assert!(out.adapter_path.exists(), "adapter file should be written");
    (losses, out.steps, out.adapter_path)
}

#[test]
#[ignore = "needs real Kolors weights"]
fn kolors_trainer_trains_and_writes_lora_that_reloads() {
    let tmp_guard = tempfile::tempdir().unwrap();
    let tmp = tmp_guard.path().to_path_buf();
    let (_losses, _steps, adapter_path) = run(&tmp, "swatch_lora.safetensors", NetworkType::Lora);

    // The produced adapter carries PEFT keys under the diffusers-UNet prefix + reload metadata.
    let w = Weights::from_file(&adapter_path).unwrap();
    assert_eq!(w.metadata("networkType"), Some("lora"));
    let n_targets = w.keys().filter(|k| k.ends_with(".lora_A.weight")).count();
    assert!(n_targets > 0, "adapter should contain LoRA factors");
    assert!(
        w.keys()
            .any(|k| k.starts_with("base_model.model.unet.") && k.ends_with(".to_q.lora_A.weight")),
        "adapter should carry PEFT-prefixed attention LoRA keys"
    );

    // Round-trip: reload through the REAL SDXL inference adapter path at Complete coverage (which
    // merges the down/mid/up attention the LoRA trained), onto a fresh f32 Kolors U-Net.
    let mut unet = load_unet_kolors_dtype(&snapshot(), Dtype::Float32).unwrap();
    let report = apply_sdxl_adapters_with(
        &mut unet,
        &[AdapterSpec {
            path: adapter_path,
            scale: 1.0,
            kind: AdapterKind::Lora,
            pass_scales: None,
            moe_expert: None,
        }],
        LoraCoverage::Complete,
    )
    .expect("LoRA adapter should reload through the SDXL inference path onto the Kolors U-Net");
    assert_eq!(
        report.merged, n_targets,
        "every trained LoRA target should merge under Complete coverage"
    );
    assert_eq!(report.skipped_keys, 0, "no LoRA key should be skipped");
    forward_finite(&unet);
    println!("[kolors-lora] e2e OK — {n_targets} targets reload + merge, forward finite");
    crate::model_inference_device::record_model_inference_device();
    println!("MODEL_INFERENCE_BODY_COMPLETED trainer_e2e::kolors_trainer_trains_and_writes_lora_that_reloads");
}

#[test]
#[ignore = "needs real Kolors weights"]
fn kolors_trainer_trains_and_writes_lokr_that_reloads() {
    let tmp_guard = tempfile::tempdir().unwrap();
    let tmp = tmp_guard.path().to_path_buf();
    let (_losses, _steps, adapter_path) = run(&tmp, "swatch_lokr.safetensors", NetworkType::Lokr);

    let w = Weights::from_file(&adapter_path).unwrap();
    assert_eq!(w.metadata("networkType"), Some("lokr"));
    assert!(w.metadata("decomposeFactor").is_some());
    let n_targets = w.keys().filter(|k| k.ends_with(".lokr_w1")).count();
    assert!(n_targets > 0, "adapter should contain LoKr factor keys");
    // LoKr targets the vendored attention surface (no mid_block — the SDXL LoKr loader keeps it out,
    // sc-2640), so every saved target reloads cleanly with nothing skipped.
    let mut unet = load_unet_kolors_dtype(&snapshot(), Dtype::Float32).unwrap();
    let report = apply_sdxl_adapters(
        &mut unet,
        &[AdapterSpec {
            path: adapter_path,
            scale: 1.0,
            kind: AdapterKind::Lokr,
            pass_scales: None,
            moe_expert: None,
        }],
    )
    .expect("LoKr adapter should reload through the SDXL inference path onto the Kolors U-Net");
    assert_eq!(
        report.merged, n_targets,
        "every trained LoKr target should merge"
    );
    assert_eq!(report.skipped_keys, 0, "no LoKr key should be skipped");
    forward_finite(&unet);
    println!("[kolors-lokr] e2e OK — {n_targets} targets reload + merge, forward finite");
    crate::model_inference_device::record_model_inference_device();
    println!("MODEL_INFERENCE_BODY_COMPLETED trainer_e2e::kolors_trainer_trains_and_writes_lokr_that_reloads");
}

/// sc-7781 — the `gradient_checkpointing` path (per-block recompute) trains end-to-end on Kolors too,
/// now that it inherits the shared SDXL-family backbone (`mlx_gen_sdxl::train_family`). Convergence
/// matches the dense path (the block-ckpt forward+grads are validated bit-close to dense by the
/// `block_ckpt_grads_match_dense` unit gate, which exercises the Kolors `encoder_hid_proj` U-Net), and
/// the adapter still saves + reloads through the SDXL inference path. Mirrors
/// `sdxl_trainer_gradient_checkpointing_converges`.
#[test]
#[ignore = "needs real Kolors weights"]
fn kolors_trainer_gradient_checkpointing_converges() {
    let tmp_guard = tempfile::tempdir().unwrap();
    let tmp = tmp_guard.path().to_path_buf();
    let (losses, steps, adapter_path) = run_cfg(
        &tmp,
        "swatch_lora_gc.safetensors",
        NetworkType::Lora,
        true, // gradient_checkpointing ON → forward_block_checkpointed
    );
    assert_eq!(steps, 16);
    assert!(
        losses.iter().all(|l| l.is_finite()),
        "no NaN/Inf under checkpointing"
    );
    let q = losses.len() / 4;
    let mean = |s: &[f32]| s.iter().sum::<f32>() / s.len() as f32;
    let (first_q, last_q) = (mean(&losses[..q]), mean(&losses[losses.len() - q..]));
    assert!(
        last_q < first_q * 0.9,
        "checkpointed training should converge like dense: {first_q:.5} -> {last_q:.5}"
    );
    let mut unet = load_unet_kolors_dtype(&snapshot(), Dtype::Float32).unwrap();
    let report = apply_sdxl_adapters_with(
        &mut unet,
        &[AdapterSpec {
            path: adapter_path,
            scale: 1.0,
            kind: AdapterKind::Lora,
            pass_scales: None,
            moe_expert: None,
        }],
        LoraCoverage::Complete,
    )
    .expect("checkpointed LoRA adapter should reload onto the Kolors U-Net");
    assert!(report.merged > 0, "checkpointed adapter should merge");
    forward_finite(&unet);
    println!(
        "[kolors-gc] e2e OK — checkpointed training converged {first_q:.5} -> {last_q:.5}, reloads"
    );
    crate::model_inference_device::record_model_inference_device();
    println!("MODEL_INFERENCE_BODY_COMPLETED trainer_e2e::kolors_trainer_gradient_checkpointing_converges");
}

/// A forward over the adapted Kolors U-Net produces a finite eps (the reloaded adapter installs +
/// runs) — under Kolors conditioning shapes: ChatGLM context `[1, N, 4096]`, pooled `[1, 4096]`, and
/// the real-resolution `time_ids = (H, W, 0, 0, H, W)`.
fn forward_finite(unet: &mlx_gen_sdxl::UNet2DConditionModel) {
    let mk = |shape: &[i32], seed: u64| {
        mlx_rs::random::normal::<f32>(shape, None, None, Some(&mlx_rs::random::key(seed).unwrap()))
            .unwrap()
    };
    let x = mk(&[1, 8, 8, 4], 1);
    let cond = mk(&[1, 8, 4096], 2);
    let pooled = mk(&[1, 4096], 3);
    let time_ids = kolors_time_ids(1, 64, 64);
    let eps = unet.forward(&x, 500.0, &cond, &pooled, &time_ids).unwrap();
    let s = eps.sum(None).unwrap().item::<f32>();
    assert!(s.is_finite(), "reloaded-adapter forward should be finite");
}

/// sc-5637 — preview samples. Proves the Kolors render path (install in-progress adapter → ChatGLM
/// CFG denoise → VAE decode → `Image`) on real weights. Run:
///   cargo test -p mlx-gen-kolors --release --test integration -- --ignored --nocapture trainer_e2e::kolors_trainer_emits_preview_samples
#[test]
#[ignore = "needs real Kolors weights (+ tokenizer.json overlay)"]
fn kolors_trainer_emits_preview_samples() {
    let tmp_guard = tempfile::tempdir().unwrap();
    let tmp = tmp_guard.path().to_path_buf();
    assert_eq!(mlx_gen_kolors::MODEL_ID, "kolors");
    let mut trainer = mlx_gen_kolors::provider_registry()
        .unwrap()
        .load_trainer("kolors", &LoadSpec::new(WeightsSource::Dir(snapshot())))
        .expect("kolors trainer should be registered");

    let mut cfg = config(NetworkType::Lora, false);
    // A 512 edge (not the 64 px smoke edge) so the saved previews are inspectable: item 0's
    // 512×768 portrait trains — and previews render at — its whole 2:3 canvas (SW-7172).
    cfg.resolution = 512;
    let items = make_dataset(&tmp, cfg.resolution);
    cfg.steps = 8;
    cfg.sample_every = 4;
    cfg.sample_steps = 6;
    cfg.sample_guidance_scale = 5.0;
    cfg.sample_prompts = vec![
        "a blue, cream and green banded swatch".to_string(),
        "a solid red colour swatch".to_string(),
    ];
    let req = TrainingRequest {
        items,
        config: cfg,
        output_dir: tmp.join("out"),
        file_name: "swatch_lora.safetensors".to_string(),
        trigger_words: vec![],
        cancel: CancelFlag::new(),
    };

    let mut samples: Vec<(u32, u32, u32, String)> = Vec::new();
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
                    "preview is not an all-black frame"
                );
                // Previews render at item 0's canvas: the 2:3 portrait, whole frame (SW-7172).
                assert!(
                    image.height > image.width,
                    "preview renders at item 0's portrait canvas, got {}x{}",
                    image.width,
                    image.height
                );
                save_proof_preview(samples.len(), &image);
                samples.push((step, index, total, prompt));
            }
        })
        .expect("training with sampling should succeed");

    assert_eq!(samples.len(), 4, "expected 4 previews, got {samples:?}");
    println!(
        "[kolors-samples] e2e OK — {} previews: {samples:?}",
        samples.len()
    );
    crate::model_inference_device::record_model_inference_device();
    println!("MODEL_INFERENCE_BODY_COMPLETED trainer_e2e::kolors_trainer_emits_preview_samples");
}
