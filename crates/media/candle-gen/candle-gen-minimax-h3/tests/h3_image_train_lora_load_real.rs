//! SW-7097 AC2 on candle/CUDA: the adapter the **MLX** H3-Image trainer saved loads through this
//! backend's H3-Image LoRA install with no conversion — every trained target resolves on the real
//! image DiT, none is left unmatched, and the trained residual moves the adapted projections.
//!
//! `#[ignore]`d: needs `MiniMaxAI/MiniMax-H3-Image` (`MINIMAX_H3_IMAGE_SNAPSHOT`), CUDA, and the
//! MLX job's trained adapter (`MINIMAX_H3_IMAGE_TRAINED_LORA`, the `minimax-h3-image-trained-lora`
//! artifact). The `minimax-h3-image-train` real-weights profile runs it on `candle-minimax-h3-image-train` after
//! the MLX training job.

use std::path::PathBuf;

use candle_gen::candle_core::{DType, Device, Tensor};
use candle_gen::gen_core::{AdapterKind, AdapterSpec};
use candle_gen_minimax_h3::adapters::{apply_minimax_h3_adapters, BLOCK_TARGETS};
use candle_gen_minimax_h3::MiniMaxH3Dit;

fn env_path(name: &str) -> PathBuf {
    let raw = std::env::var(name).unwrap_or_default();
    assert!(
        !raw.is_empty(),
        "{name} must be set; this test is #[ignore]d and asserts rather than skips"
    );
    PathBuf::from(raw)
}

/// Block 0's six adapted leaves on one seeded input, as float32.
fn block0_leaves(dit: &mut MiniMaxH3Dit, device: &Device) -> Vec<Vec<f32>> {
    let cfg = dit.config().clone();
    BLOCK_TARGETS
        .iter()
        .map(|leaf| {
            let inp = match *leaf {
                "attn.to_out.0" => cfg.num_attention_heads * cfg.attention_head_dim,
                "ff.net.2" => cfg.ffn_dim,
                _ => cfg.hidden_size,
            };
            let values: Vec<f32> = (0..8 * inp)
                .map(|i| ((i as u32).wrapping_mul(2_654_435_761) % 2003) as f32 / 2003.0 - 0.5)
                .collect();
            let x = Tensor::from_vec(values, &[1, 8, inp], &Device::Cpu)
                .and_then(|t| t.to_dtype(DType::BF16))
                .and_then(|t| t.to_device(device))
                .unwrap();
            let path = format!("transformer_blocks.0.{leaf}");
            let segs: Vec<&str> = path.split('.').collect();
            dit.adaptable_mut(&segs)
                .unwrap()
                .forward(&x)
                .unwrap()
                .to_dtype(DType::F32)
                .and_then(|y| y.flatten_all())
                .and_then(|y| y.to_vec1::<f32>())
                .unwrap()
        })
        .collect()
}

#[test]
#[ignore = "needs MINIMAX_H3_IMAGE_SNAPSHOT, CUDA and the MLX-trained adapter (MINIMAX_H3_IMAGE_TRAINED_LORA); 66 GB bf16 DiT"]
fn h3_image_mlx_trained_lora_installs_on_the_real_dit_cuda() {
    let root = env_path("MINIMAX_H3_IMAGE_SNAPSHOT");
    let lora = env_path("MINIMAX_H3_IMAGE_TRAINED_LORA");
    assert!(lora.is_file(), "no trained adapter at {}", lora.display());
    let targets = candle_gen::gen_core::weightsmeta::safetensors_path_tensor_headers(&lora)
        .unwrap()
        .into_iter()
        .filter(|h| h.name.ends_with(".lora_A.weight"))
        .count();
    assert_eq!(
        targets,
        50 * BLOCK_TARGETS.len(),
        "the trainer's default targets"
    );
    let device = candle_gen::default_device().expect("a candle device");
    let mut dit =
        MiniMaxH3Dit::load_from_dir(root.join("transformer_diffusers"), &device, DType::BF16)
            .expect("load the image DiT");
    let bare = block0_leaves(&mut dit, &device);
    let report = apply_minimax_h3_adapters(
        &mut dit,
        &[AdapterSpec::new(lora.clone(), 1.0, AdapterKind::Lora)],
    )
    .expect("the H3-Image install accepts the MLX-trained adapter");
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
    let adapted = block0_leaves(&mut dit, &device);
    let moved = adapted
        .iter()
        .zip(&bare)
        .map(|(a, b)| a.iter().zip(b).fold(0f32, |m, (x, y)| m.max((x - y).abs())))
        .fold(0f32, f32::max);
    println!("[sw-7097 candle] MLX-trained LoRA: {targets} targets installed, block-0 leaves moved by max |Δ| {moved:.3e}");
    assert!(
        moved > 0.0,
        "the trained residual must move the adapted leaves"
    );
    #[cfg(feature = "model-inference-proof")]
    println!(
        "\nMODEL_INFERENCE_BODY_COMPLETED h3_image_train_lora_load_real::h3_image_mlx_trained_lora_installs_on_the_real_dit_cuda"
    );
}
