//! sc-3044 e2e — the production `ZImageTurboTrainer` (the `Trainer` contract realized on Z-Image),
//! driven through the core registry exactly as the SceneWorks worker will.
//!
//! `#[ignore]`d — needs the real `Tongyi-MAI/Z-Image-Turbo` weights in the HF cache (or
//! `ZIMAGE_SNAPSHOT`). Run:
//!   cargo test -p mlx-gen-z-image --release --test trainer_e2e -- --ignored --nocapture
//!
//! Proves the full prepare→load→cache→train→save lifecycle: a tiny captioned PNG dataset is
//! VAE/text-encoded and cached, AdamW training drives the flow-match loss down, and a PEFT adapter
//! is written that carries the inference-reload metadata (the spike already proved that adapter
//! round-trips bit-for-bit, so here we assert the loop + the output contract).

use std::path::PathBuf;

use mlx_gen::weights::Weights;
use mlx_gen::{
    CancelFlag, LoadSpec, TrainingConfig, TrainingItem, TrainingProgress, TrainingRequest,
    WeightsSource,
};

fn snapshot() -> PathBuf {
    if let Ok(p) = std::env::var("ZIMAGE_SNAPSHOT") {
        return PathBuf::from(p);
    }
    let home = std::env::var("HOME").unwrap();
    let snaps = PathBuf::from(home)
        .join(".cache/huggingface/hub/models--Tongyi-MAI--Z-Image-Turbo/snapshots");
    std::fs::read_dir(&snaps)
        .expect("HF cache snapshots dir")
        .filter_map(|e| e.ok())
        .map(|e| e.path())
        .find(|p| p.is_dir())
        .expect("a snapshot dir")
}

#[test]
#[ignore = "needs real Z-Image weights"]
fn z_image_trainer_trains_and_writes_adapter() {
    // Tiny synthetic dataset: two solid-colour swatches + captions, written as PNGs.
    let tmp = std::env::temp_dir().join("z_image_trainer_e2e");
    let _ = std::fs::remove_dir_all(&tmp);
    std::fs::create_dir_all(&tmp).unwrap();
    let mut items = Vec::new();
    for (i, color) in [[200u8, 40, 40], [40, 80, 200]].iter().enumerate() {
        let mut img = image::RgbImage::new(96, 96);
        for px in img.pixels_mut() {
            *px = image::Rgb(*color);
        }
        let path = tmp.join(format!("img{i}.png"));
        img.save(&path).unwrap();
        items.push(TrainingItem {
            image_path: path,
            caption: format!("a solid colour swatch number {i}"),
        });
    }

    // Reference the provider crate so its `inventory::submit!` registration is linked into this
    // test binary (a consumer that links the crate — like the worker — gets this for free; an
    // integration test that names nothing from the crate would otherwise have it dead-stripped).
    assert_eq!(mlx_gen_z_image::MODEL_ID, "z_image_turbo");

    // Load the trainer through the registry (validates self-registration), exactly like the worker.
    let mut trainer = mlx_gen::load_trainer(
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

    // Convergence itself is proven deterministically by the spike (lora_train_spike.rs); here each
    // step samples a fresh sigma+noise from the sigmoid distribution, so per-step loss is dominated
    // by sigma variance, not a monotonic curve. As a soft efficacy signal we compare the mean of
    // the first vs last quarter of the run (sigma noise averages out).
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
    println!(
        "[trainer] e2e OK — adapter written to {}",
        out.adapter_path.display()
    );
}
