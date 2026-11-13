//! SW-7098 AC2 on MLX: the adapter the **Candle** H3-Image trainer saved loads through this
//! backend's H3-Image LoRA install with no conversion — every trained target resolves on the real
//! image DiT, none is left unmatched, and the trained residual moves the adapted projections. The
//! twin of the Candle `h3_image_train_lora_load_real` gate, in the other direction.
//!
//! `#[ignore]`d: needs `MiniMaxAI/MiniMax-H3-Image` (`MINIMAX_H3_IMAGE_SNAPSHOT`) and the Candle
//! job's trained adapter (`MINIMAX_H3_IMAGE_CANDLE_TRAINED_LORA`, the
//! `minimax-h3-image-candle-trained-lora` artifact). The `minimax-h3-image-train` real-weights
//! profiles run it on `mlx-minimax-h3-image-train-load` after the Candle training job.

use std::path::PathBuf;

use mlx_rs::{Array, Dtype};

use mlx_gen::adapters::AdaptableHost;
use mlx_gen::gen_core::{AdapterKind, AdapterSpec};
use mlx_gen_minimax_h3::adapters::{apply_minimax_h3_adapters, BLOCK_TARGETS};
use mlx_gen_minimax_h3::h3_image::MiniMaxH3ImageDit;

fn env_path(name: &str) -> PathBuf {
    let raw = std::env::var(name).unwrap_or_default();
    assert!(
        !raw.is_empty(),
        "{name} must be set; this test is #[ignore]d and asserts rather than skips"
    );
    PathBuf::from(raw)
}

/// Block 0's six adapted leaves on one seeded input, as float32.
fn block0_leaves(dit: &mut MiniMaxH3ImageDit) -> Vec<Vec<f32>> {
    let b = dit.config().base.clone();
    BLOCK_TARGETS
        .iter()
        .map(|leaf| {
            let inp = match *leaf {
                "attn.to_out.0" => b.inner_dim(),
                "ff.net.2" => b.ffn_dim,
                _ => b.hidden_size,
            };
            let values: Vec<f32> = (0..8 * inp as u32)
                .map(|i| (i.wrapping_mul(2_654_435_761) % 2003) as f32 / 2003.0 - 0.5)
                .collect();
            let x = Array::from_slice(&values, &[1, 8, inp])
                .as_dtype(Dtype::Bfloat16)
                .unwrap();
            let path = format!("transformer_blocks.0.{leaf}");
            let segs: Vec<&str> = path.split('.').collect();
            let y = dit
                .adaptable_mut(&segs)
                .unwrap()
                .forward(&x)
                .unwrap()
                .as_dtype(Dtype::Float32)
                .unwrap()
                .flatten(None, None)
                .unwrap();
            mlx_rs::transforms::eval([&y]).unwrap();
            y.as_slice::<f32>().to_vec()
        })
        .collect()
}

#[test]
#[ignore = "needs MINIMAX_H3_IMAGE_SNAPSHOT and the Candle-trained adapter (MINIMAX_H3_IMAGE_CANDLE_TRAINED_LORA); 66 GB bf16 DiT"]
fn h3_image_candle_trained_lora_installs_on_the_real_dit() {
    let root = env_path("MINIMAX_H3_IMAGE_SNAPSHOT");
    let lora = env_path("MINIMAX_H3_IMAGE_CANDLE_TRAINED_LORA");
    assert!(lora.is_file(), "no trained adapter at {}", lora.display());
    let targets = mlx_gen::gen_core::weightsmeta::safetensors_path_tensor_headers(&lora)
        .unwrap()
        .into_iter()
        .filter(|h| h.name.ends_with(".lora_A.weight"))
        .count();
    assert_eq!(
        targets,
        50 * BLOCK_TARGETS.len(),
        "the trainer's default targets"
    );
    let mut dit = MiniMaxH3ImageDit::load(&root, Dtype::Bfloat16).expect("load the image DiT");
    let bare = block0_leaves(&mut dit);
    let report = apply_minimax_h3_adapters(
        &mut dit,
        &[AdapterSpec::new(lora.clone(), 1.0, AdapterKind::Lora)],
    )
    .expect("the H3-Image install accepts the Candle-trained adapter");
    assert_eq!(report.applied, targets, "every trained target installs");
    assert!(
        report.unmatched_paths.is_empty(),
        "{:?}",
        report.unmatched_paths
    );
    assert_eq!(
        report.converted_from_comfyui + report.converted_from_kohya,
        0
    );
    let adapted = block0_leaves(&mut dit);
    let moved = adapted
        .iter()
        .zip(&bare)
        .map(|(a, b)| a.iter().zip(b).fold(0f32, |m, (x, y)| m.max((x - y).abs())))
        .fold(0f32, f32::max);
    println!("[sw-7098 mlx] Candle-trained LoRA: {targets} targets installed, block-0 leaves moved by max |Δ| {moved:.3e}");
    assert!(
        moved > 0.0,
        "the trained residual must move the adapted leaves"
    );
    crate::model_inference_device::record_model_inference_device();
    println!(
        "MODEL_INFERENCE_BODY_COMPLETED h3_image_candle_lora_load_real::h3_image_candle_trained_lora_installs_on_the_real_dit"
    );
}
