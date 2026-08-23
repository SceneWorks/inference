//! sc-14055 e2e — the rectified-flow `MageFlowTrainer` (the `Trainer` contract on Mage-Flow-Base),
//! driven through the explicit provider registry.
//!
//! `#[ignore]`d — needs the real `microsoft/Mage-Flow-Base` snapshot in `MAGE_BASE_SNAPSHOT`. Run
//! (this Mac can run MLX GPU):
//!   MAGE_BASE_SNAPSHOT=/path/to/Mage-Flow-Base \
//!     cargo test -p mlx-gen-mage --release --test integration trainer_e2e:: -- --ignored --nocapture
//!
//! Proves the full prepare→cache→train→save lifecycle: a tiny captioned PNG dataset is Mage-VAE /
//! Qwen3-VL encoded and cached, AdamW training drives the rectified flow-match loss down, and a PEFT
//! adapter is written that (a) carries `alpha`/`rank` in `__metadata__` and (b) reloads through the
//! real inference path (`apply_mage_adapters`) onto a fresh NR-MMDiT and **visibly changes** its
//! velocity output — the DoD round-trip, for both the LoRA and LoKr adapter kinds.

use std::path::Path;

use mlx_gen::weights::Weights;
use mlx_gen::{
    AdapterKind, AdapterSpec, CancelFlag, LoadSpec, NetworkType, TrainingConfig, TrainingItem,
    TrainingProgress, TrainingRequest, WeightsSource,
};
use mlx_gen_mage::{apply_mage_adapters, ImgShape, MageTransformer, PackLayout};

fn base_snapshot() -> String {
    std::env::var("MAGE_BASE_SNAPSHOT")
        .expect("set MAGE_BASE_SNAPSHOT to the full microsoft/Mage-Flow-Base snapshot root")
}

/// Two solid-colour swatch PNGs + captions in `dir`.
fn make_dataset(dir: &Path) -> Vec<TrainingItem> {
    let _ = std::fs::remove_dir_all(dir);
    std::fs::create_dir_all(dir).unwrap();
    let mut items = Vec::new();
    for (i, color) in [[200u8, 40, 40], [40, 80, 200]].iter().enumerate() {
        let mut img = image::RgbImage::new(320, 320);
        for px in img.pixels_mut() {
            *px = image::Rgb(*color);
        }
        let path = dir.join(format!("img{i}.png"));
        img.save(&path).unwrap();
        items.push(TrainingItem::captioned(
            path,
            format!("a solid colour swatch number {i}"),
        ));
    }
    items
}

/// A fixed random packed DiT input `(img, txt, sigma, ctx)` for a `grid`×`grid` latent + `txt_len`
/// caption tokens — the probe for the "reloading the adapter changes the velocity" assertion.
fn probe_forward(
    t: &MageTransformer,
    grid: i32,
    txt_len: i32,
) -> mlx_rs::error::Result<mlx_rs::Array> {
    use mlx_rs::random;
    let layout = PackLayout::generation(vec![ImgShape::latent(grid, grid)], vec![txt_len]).unwrap();
    let ctx = t.pack_context(layout).unwrap();
    let img = random::normal::<f32>(&[1, grid * grid, 128], None, None, Some(&random::key(1)?))?;
    let txt = random::normal::<f32>(&[1, txt_len, 2560], None, None, Some(&random::key(2)?))?;
    let sigma = mlx_rs::Array::from_slice(&[0.5f32], &[1]);
    Ok(t.forward(&img, &txt, &sigma, &ctx).unwrap())
}

fn max_abs_diff(a: &mlx_rs::Array, b: &mlx_rs::Array) -> f32 {
    let n = a.shape().iter().product::<i32>();
    let a = a
        .as_dtype(mlx_rs::Dtype::Float32)
        .unwrap()
        .reshape(&[n])
        .unwrap();
    let b = b
        .as_dtype(mlx_rs::Dtype::Float32)
        .unwrap()
        .reshape(&[n])
        .unwrap();
    a.subtract(&b)
        .unwrap()
        .abs()
        .unwrap()
        .max(None)
        .unwrap()
        .item::<f32>()
}

fn train_and_check(network_type: NetworkType, kind: AdapterKind, tag: &str) {
    let tmp_guard = tempfile::tempdir().unwrap();
    let tmp = tmp_guard.path().to_path_buf();
    let items = make_dataset(&tmp);

    let mut trainer = mlx_gen_mage::provider_registry()
        .unwrap()
        .load_trainer(
            "mage_flow_base",
            &LoadSpec::new(WeightsSource::Dir(base_snapshot().into())),
        )
        .expect("mage_flow_base trainer should be registered");

    let config = TrainingConfig {
        rank: 8,
        alpha: 8.0,
        learning_rate: 1e-4, // 1e-3 AdamW overshoots this landscape; the default converges cleanly
        steps: 120,
        resolution: 256, // → 16×16 latent, 256 tokens — fast
        save_every: 0,
        seed: 7,
        network_type,
        decompose_factor: -1,
        ..Default::default()
    };
    let req = TrainingRequest {
        items,
        config,
        output_dir: tmp.join("out"),
        file_name: format!("swatch_{tag}.safetensors"),
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

    assert_eq!(cached, 2, "both dataset items cached");
    assert_eq!(out.steps, 120, "all micro-steps ran");
    assert_eq!(losses.len(), 120);
    assert!(losses.iter().all(|l| l.is_finite()), "no NaN/Inf losses");

    // Each step samples a fresh sigma+noise, so per-step loss is dominated by sigma variance. Compare
    // the mean of the first vs last third (sigma noise averages out over a wide window) — real data
    // must train. (The stationary-objective monotone convergence is proven decisively by the
    // `overfits_a_fixed_batch` lib test; this is the real random-schedule signal.)
    let third = losses.len() / 3;
    let mean = |s: &[f32]| s.iter().sum::<f32>() / s.len() as f32;
    let (first_t, last_t) = (
        mean(&losses[..third]),
        mean(&losses[losses.len() - third..]),
    );
    println!("[trainer-{tag}] loss-mean first-third {first_t:.5} -> last-third {last_t:.5}");
    assert!(
        last_t < first_t * 0.9,
        "[{tag}] windowed loss-mean should fall on real data: {first_t:.5} -> {last_t:.5}"
    );

    // Adapter carries the reload metadata in __metadata__ (the known past gap) + PEFT/LoKr keys.
    assert!(out.adapter_path.exists());
    let w = Weights::from_file(&out.adapter_path).unwrap();
    let want_net = match network_type {
        NetworkType::Lora => "lora",
        NetworkType::Lokr => "lokr",
    };
    assert_eq!(w.metadata("networkType"), Some(want_net));
    assert_eq!(w.metadata("rank"), Some("8"), "rank in __metadata__");
    assert_eq!(w.metadata("alpha"), Some("8"), "alpha in __metadata__");
    // sc-14057 family provenance. These exact tokens are what SceneWorks' importer reads
    // (`sceneworks-core::lora_family::detect_metadata_family` checks `family` then `baseModel`):
    // without them an exported Mage adapter re-imports family-less, because the NR-MMDiT's tensor
    // names are indistinguishable from Qwen-Image's and the default target set reaches no
    // dual-stream marker at all. Both adapter kinds must carry them.
    assert_eq!(
        w.metadata("family"),
        Some("mage_flow"),
        "family stamp in __metadata__ (sc-14057 import detection)"
    );
    assert_eq!(
        w.metadata("baseModel"),
        Some("mage_flow_base"),
        "baseModel stamp in __metadata__ (sc-14057 import detection)"
    );
    let factor_suffix = match network_type {
        NetworkType::Lora => ".lora_A.weight",
        NetworkType::Lokr => ".lokr_w1",
    };
    let n_targets = w.keys().filter(|k| k.ends_with(factor_suffix)).count();
    // Default targets: image-stream to_q/to_k/to_v/to_out.0 across all 12 dual-stream blocks.
    assert_eq!(n_targets, 48, "[{tag}] to_q/k/v/out over 12 blocks");

    // --- reload changes generation output (the DoD) ---
    let adapter_path = out.adapter_path.clone();
    drop(trainer);
    let mut t = MageTransformer::load(Path::new(&base_snapshot()).join("transformer")).unwrap();
    let v_base = probe_forward(&t, 16, 8).unwrap();
    let report = apply_mage_adapters(
        &mut t,
        &[AdapterSpec {
            path: adapter_path,
            scale: 1.0,
            kind,
            pass_scales: None,
            moe_expert: None,
        }],
    )
    .expect("adapter should reload through the inference path");
    assert_eq!(
        report.applied, n_targets,
        "[{tag}] every saved target reloads"
    );
    assert!(
        report.unmatched_paths.is_empty(),
        "[{tag}] every target resolves"
    );

    let v_adapted = probe_forward(&t, 16, 8).unwrap();
    assert!(
        v_adapted.sum(None).unwrap().item::<f32>().is_finite(),
        "[{tag}] reloaded forward finite"
    );
    let delta = max_abs_diff(&v_base, &v_adapted);
    println!("[trainer-{tag}] reloaded adapter shifts the DiT velocity by max_abs {delta:.5}");
    assert!(
        delta > 1e-4,
        "[{tag}] reloading the trained adapter must visibly change the velocity output (got {delta})"
    );
}

#[test]
#[ignore = "needs real Mage-Flow-Base weights (MAGE_BASE_SNAPSHOT)"]
fn mage_lora_trains_writes_adapter_and_reloads() {
    train_and_check(NetworkType::Lora, AdapterKind::Lora, "lora");
}

#[test]
#[ignore = "needs real Mage-Flow-Base weights (MAGE_BASE_SNAPSHOT)"]
fn mage_lokr_trains_writes_adapter_and_reloads() {
    train_and_check(NetworkType::Lokr, AdapterKind::Lokr, "lokr");
}

/// Preview samples (the `sample_every` / `sample_prompts` controls): a short run must emit
/// `TrainingProgress::Sample` events at the cadence, each a real decoded RGB bitmap rendered from the
/// in-progress adapter (install → native-resolution denoise → VAE decode → `Image`).
#[test]
#[ignore = "needs real Mage-Flow-Base weights (MAGE_BASE_SNAPSHOT); renders previews"]
fn mage_trainer_emits_preview_samples() {
    let tmp_guard = tempfile::tempdir().unwrap();
    let tmp = tmp_guard.path().to_path_buf();
    let items = make_dataset(&tmp);

    let mut trainer = mlx_gen_mage::provider_registry()
        .unwrap()
        .load_trainer(
            "mage_flow_base",
            &LoadSpec::new(WeightsSource::Dir(base_snapshot().into())),
        )
        .unwrap();
    let config = TrainingConfig {
        rank: 8,
        alpha: 8.0,
        learning_rate: 1e-4,
        steps: 8,
        resolution: 512, // preview render uses the training resolution
        save_every: 0,
        seed: 7,
        sample_every: 4,
        sample_steps: 4,
        sample_guidance_scale: 1.0,
        sample_prompts: vec![
            "a solid red colour swatch".to_string(),
            "a solid blue colour swatch".to_string(),
        ],
        ..Default::default()
    };
    let req = TrainingRequest {
        items,
        config,
        output_dir: tmp.join("out"),
        file_name: "swatch.safetensors".to_string(),
        trigger_words: vec![],
        cancel: CancelFlag::new(),
    };

    let mut samples: Vec<(u32, u32, u32, usize)> = Vec::new();
    trainer
        .train(&req, &mut |p| {
            if let TrainingProgress::Sample {
                step,
                index,
                total,
                image,
                ..
            } = p
            {
                samples.push((step, index, total, image.pixels.len()));
                assert!(
                    image.width > 0 && image.height > 0,
                    "preview has dimensions"
                );
                assert_eq!(
                    image.pixels.len(),
                    (image.width * image.height * 3) as usize,
                    "preview is a packed RGB8 buffer"
                );
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
        samples.iter().any(|(s, ..)| *s == 4) && samples.iter().any(|(s, ..)| *s == 8),
        "previews fire at steps 4 and 8: {samples:?}"
    );
    println!(
        "[trainer-samples] e2e OK — {} previews: {samples:?}",
        samples.len()
    );
}

/// The literal-DoD render check: train a LoRA, then decode a full image from the Base pipeline with
/// and without the reloaded adapter (same seed/prompt) and assert the pixels differ. Heavier than the
/// velocity probe (two native-resolution denoises), so it is a separate `#[ignore]`d test.
#[test]
#[ignore = "needs real Mage-Flow-Base weights (MAGE_BASE_SNAPSHOT); renders two images"]
fn mage_lora_reload_changes_generated_image() {
    use mlx_gen_mage::{resolve_gs_key, GenerationSample, MageFlowPipeline};

    let tmp_guard = tempfile::tempdir().unwrap();
    let tmp = tmp_guard.path().to_path_buf();
    let items = make_dataset(&tmp);

    let mut trainer = mlx_gen_mage::provider_registry()
        .unwrap()
        .load_trainer(
            "mage_flow_base",
            &LoadSpec::new(WeightsSource::Dir(base_snapshot().into())),
        )
        .unwrap();
    let config = TrainingConfig {
        rank: 8,
        alpha: 16.0,
        learning_rate: 2e-3,
        steps: 60,
        resolution: 256,
        save_every: 0,
        seed: 7,
        ..Default::default()
    };
    let req = TrainingRequest {
        items,
        config,
        output_dir: tmp.join("out"),
        file_name: "render_lora.safetensors".to_string(),
        trigger_words: vec![],
        cancel: CancelFlag::new(),
    };
    let out = trainer.train(&req, &mut |_| {}).unwrap();
    drop(trainer);

    let mut pipeline = MageFlowPipeline::load(base_snapshot()).unwrap();
    let key = resolve_gs_key(None).unwrap();
    let sample = GenerationSample {
        prompt: "a solid colour swatch number 0",
        negative_prompt: " ",
        height: 512,
        width: 512,
        seed: 123,
    };
    let base_img = pipeline
        .generate_batch(&[sample], 8, 1.0, &key, false)
        .unwrap()
        .swap_remove(0);

    apply_mage_adapters(
        &mut pipeline.transformer,
        &[AdapterSpec {
            path: out.adapter_path.clone(),
            scale: 1.0,
            kind: AdapterKind::Lora,
            pass_scales: None,
            moe_expert: None,
        }],
    )
    .unwrap();
    let adapted_img = pipeline
        .generate_batch(&[sample], 8, 1.0, &key, false)
        .unwrap()
        .swap_remove(0);

    let delta = max_abs_diff(&base_img, &adapted_img);
    let mean = {
        let n = base_img.shape().iter().product::<i32>();
        let a = base_img
            .as_dtype(mlx_rs::Dtype::Float32)
            .unwrap()
            .reshape(&[n])
            .unwrap();
        let b = adapted_img
            .as_dtype(mlx_rs::Dtype::Float32)
            .unwrap()
            .reshape(&[n])
            .unwrap();
        a.subtract(&b)
            .unwrap()
            .abs()
            .unwrap()
            .mean(None)
            .unwrap()
            .item::<f32>()
    };
    println!("[trainer-render] reloaded LoRA changes the decoded 512² image: max_abs {delta:.1}/255, mean {mean:.3}/255");
    assert!(
        delta > 0.5,
        "reloading the trained LoRA must visibly change the generated image (max pixel delta {delta})"
    );
}
