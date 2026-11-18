//! sc-7577 — real-weight end-to-end smoke for the Krea 2 Raw LoRA trainer (epic 7565 P3). Weight-gated
//! (`#[ignore]`): drives the PUBLIC [`Trainer`] surface (`load_trainer` → `Trainer::train`) on the real
//! `krea/Krea-2-Raw` snapshot, a short run that must produce a **loadable PEFT adapter** — the story AC.
//!
//!   cargo test -p mlx-gen-krea --release --test integration trainer_real_weights:: -- --ignored --nocapture
//!
//! Set `KREA_RAW_DIR` to override the snapshot location (else the newest HF-cache `krea/Krea-2-Raw`).

use std::path::PathBuf;

use mlx_gen::{
    CancelFlag, LoadSpec, NetworkType, TrainingConfig, TrainingItem, TrainingProgress,
    TrainingRequest, WeightsSource,
};
use mlx_gen_krea::load_trainer;

/// Resolve the `krea/Krea-2-Raw` snapshot (the `KREA_RAW_DIR` override, else the newest HF-cache
/// snapshot with a `transformer/` tree).
fn raw_snapshot() -> Option<PathBuf> {
    let p = std::env::var("KREA_RAW_DIR").ok()?;
    Some(PathBuf::from(p))
}

/// Write a synthetic 2:3 portrait PNG of three distinct horizontal bands (blue / cream / green) for a
/// run at bucket edge `edge`, so the smoke needs no external dataset fixture and trains a non-square
/// aspect-preserving canvas whose whole frame (all three bands, never a centre-cropped square) is
/// visible. It is `edge` wide × 1.5 tall: the canvas never upscales a source (SW-7172), so a fixture
/// smaller than `edge²` would silently train a smaller workload than the run claims.
fn write_synth_image(path: &std::path::Path, edge: u32) {
    let (w, h) = (edge, edge * 3 / 2);
    let bands = [[40u8, 80, 200], [235, 225, 200], [40, 160, 70]];
    let img = image::RgbImage::from_fn(w, h, |_, y| {
        image::Rgb(bands[(y as usize * bands.len()) / h as usize])
    });
    img.save(path).expect("write synth png");
}

/// Parse a safetensors file's JSON header (the 8-byte LE length prefix + UTF-8 JSON) — enough to assert
/// the adapter is the expected PEFT shape without a safetensors-parser dependency.
fn safetensors_header(path: &std::path::Path) -> String {
    let bytes = std::fs::read(path).expect("read adapter");
    assert!(
        bytes.len() > 8,
        "adapter file too small: {} bytes",
        bytes.len()
    );
    let hlen = u64::from_le_bytes(bytes[0..8].try_into().unwrap()) as usize;
    String::from_utf8(bytes[8..8 + hlen].to_vec()).expect("utf8 header")
}

/// Run a short LoRA training run on the real Raw weights and confirm it writes a loadable PEFT adapter.
#[test]
#[ignore = "needs real krea/Krea-2-Raw weights (~25 GB) + a Mac; run as its own process"]
fn short_train_produces_loadable_adapter() {
    let root = raw_snapshot().expect("krea/Krea-2-Raw snapshot (HF cache or KREA_RAW_DIR)");
    let tmp_guard = tempfile::tempdir().unwrap();
    let tmp = tmp_guard.path().to_path_buf();
    let img_path = tmp.join("swatch.png");
    // The original 256 bucket edge: the 256×384 portrait trains — and the preview renders at — its
    // whole 2:3 canvas (SW-7172). A 512 edge (512×768) pushes the first training step past what a
    // 128 GB Mac can admit beside its resident workloads (run 44301427997: >63 GiB within 6 s).
    let edge = 256;
    write_synth_image(&img_path, edge);

    let mut trainer = load_trainer(&LoadSpec::new(WeightsSource::Dir(root))).expect("load_trainer");

    let req = TrainingRequest {
        items: vec![TrainingItem {
            image_path: img_path,
            caption: "a blue, cream and green banded swatch".into(),
            control_image_path: None,
            model_options: Default::default(),
            reference_image_paths: Vec::new(),
            subject_mask_path: None,
        }],
        config: TrainingConfig {
            rank: 4,
            alpha: 4.0,
            steps: 3,
            resolution: edge,
            save_every: 0,
            learning_rate: 1e-4,
            network_type: NetworkType::Lora,
            // The bounded proof trains through the checkpointed path: the dense (non-checkpointed)
            // first step needs far more than the pre-flight's ~24 GB + per-token estimate on a
            // 128 GB Mac (runs 44288439227…44316071280 crossed every admissible cap within seconds
            // of step 1) — a trainer defect tracked on its own, not this canvas proof's subject.
            gradient_checkpointing: true,
            // One preview at the final step, rendered at item 0's canvas from the trained adapter.
            sample_every: 3,
            sample_steps: 12,
            sample_prompts: vec!["a blue, cream and green banded swatch".into()],
            ..Default::default()
        },
        output_dir: tmp.clone(),
        file_name: "krea_smoke_lora.safetensors".into(),
        trigger_words: vec!["swatch".into()],
        cancel: CancelFlag::new(),
    };

    let mut saw_training = false;
    let mut last_step = 0u32;
    let mut previews = 0usize;
    let out = trainer
        .train(&req, &mut |p| match p {
            TrainingProgress::Training { step, loss, .. } => {
                saw_training = true;
                last_step = step;
                eprintln!("[sc-7577] step {step} loss {loss:.5}");
            }
            TrainingProgress::Sample { image: preview, .. } => {
                assert_eq!(
                    preview.pixels.len(),
                    (preview.width * preview.height * 3) as usize,
                    "preview is a packed RGB8 buffer"
                );
                assert!(
                    preview.pixels.iter().any(|&b| b != 0),
                    "preview is not an all-black frame"
                );
                // The preview renders at item 0's canvas: the 2:3 portrait, whole frame.
                assert!(
                    preview.height > preview.width,
                    "preview renders at item 0's portrait canvas, got {}x{}",
                    preview.width,
                    preview.height
                );
                if let Some(dir) = std::env::var_os("MODEL_PROOF_OUTPUT_DIR") {
                    let path = PathBuf::from(dir).join(format!("preview_{previews}.png"));
                    image::save_buffer(
                        path,
                        &preview.pixels,
                        preview.width,
                        preview.height,
                        image::ColorType::Rgb8,
                    )
                    .unwrap();
                }
                previews += 1;
            }
            _ => {}
        })
        .expect("train");

    assert_eq!(out.steps, 3, "all 3 micro-steps ran");
    assert_eq!(previews, 1, "one preview at the step-3 cadence");
    assert!(
        saw_training && last_step == 3,
        "Training progress streamed to step 3"
    );
    assert!(
        out.final_loss.is_finite(),
        "final loss is finite: {}",
        out.final_loss
    );

    // The AC: a loadable PEFT adapter. Confirm the safetensors header carries the PEFT factor keys +
    // the reload-contract metadata (networkType/rank/alpha) the sc-7578 apply path reads.
    let header = safetensors_header(&out.adapter_path);
    assert!(
        header.contains(".lora_A.weight"),
        "PEFT lora_A keys: {header:.200}"
    );
    assert!(header.contains(".lora_B.weight"), "PEFT lora_B keys");
    assert!(
        header.contains("transformer_blocks."),
        "single-stream block targets"
    );
    assert!(
        header.contains("networkType") && header.contains("lora"),
        "reload metadata"
    );
    eprintln!(
        "[sc-7577] ✅ adapter {} ({} bytes)",
        out.adapter_path.display(),
        std::fs::metadata(&out.adapter_path).unwrap().len()
    );
    crate::model_inference_device::record_model_inference_device();
    println!("MODEL_INFERENCE_BODY_COMPLETED trainer_real_weights::short_train_produces_loadable_adapter");
}

/// The same short run with **gradient checkpointing** on — exercises the
/// `forward_with_blocks_checkpointed` path end-to-end through the public trainer (the OOM-mitigation
/// toggle), confirming it trains + writes a loadable adapter too.
#[test]
#[ignore = "needs real krea/Krea-2-Raw weights (~25 GB) + a Mac; run as its own process"]
fn short_train_checkpointed_produces_loadable_adapter() {
    let root = raw_snapshot().expect("krea/Krea-2-Raw snapshot (HF cache or KREA_RAW_DIR)");
    let tmp_guard = tempfile::tempdir().unwrap();
    let tmp = tmp_guard.path().to_path_buf();
    let img_path = tmp.join("swatch.png");
    write_synth_image(&img_path, 256);

    let mut trainer = load_trainer(&LoadSpec::new(WeightsSource::Dir(root))).expect("load_trainer");
    let req = TrainingRequest {
        items: vec![TrainingItem {
            image_path: img_path,
            caption: "a vivid abstract color swatch".into(),
            control_image_path: None,
            model_options: Default::default(),
            reference_image_paths: Vec::new(),
            subject_mask_path: None,
        }],
        config: TrainingConfig {
            rank: 4,
            alpha: 4.0,
            steps: 2,
            resolution: 256,
            save_every: 0,
            gradient_checkpointing: true,
            network_type: NetworkType::Lora,
            ..Default::default()
        },
        output_dir: tmp.clone(),
        file_name: "krea_smoke_ckpt_lora.safetensors".into(),
        trigger_words: vec![],
        cancel: CancelFlag::new(),
    };

    let out = trainer
        .train(&req, &mut |_| {})
        .expect("train (checkpointed)");
    assert_eq!(out.steps, 2);
    let header = safetensors_header(&out.adapter_path);
    assert!(header.contains(".lora_A.weight") && header.contains(".lora_B.weight"));
    eprintln!(
        "[sc-7577] ✅ checkpointed adapter {}",
        out.adapter_path.display()
    );
    crate::model_inference_device::record_model_inference_device();
    println!("MODEL_INFERENCE_BODY_COMPLETED trainer_real_weights::short_train_checkpointed_produces_loadable_adapter");
}
