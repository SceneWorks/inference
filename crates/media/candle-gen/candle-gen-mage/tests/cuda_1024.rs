//! Real 1024² CUDA release gate for sc-14051.
//!
//! The expected image is generated independently by the frozen Torch reference in the MLX
//! real-weight job, then transferred as a workflow artifact. The implementation under test never
//! generates or blesses its own oracle.

#![cfg(feature = "cuda")]

use std::path::PathBuf;

use candle_core::{DType, Device};
use candle_gen_mage::MagePipeline;

#[test]
#[ignore = "needs CUDA, MAGE_SNAPSHOT, and the independent 1024² Torch e2e golden"]
fn correct_1024_rl_generation_matches_the_torch_oracle() {
    let root = PathBuf::from(
        std::env::var("MAGE_SNAPSHOT")
            .expect("set MAGE_SNAPSHOT to a caller-provisioned microsoft/Mage-Flow snapshot"),
    );
    let golden_path = PathBuf::from(
        std::env::var("MAGE_GOLDEN_DIR")
            .expect("set MAGE_GOLDEN_DIR to the Torch oracle directory"),
    )
    .join("mage_flow_e2e_golden.safetensors");
    let golden = candle_core::safetensors::load(&golden_path, &Device::Cpu)
        .unwrap_or_else(|e| panic!("load {}: {e}", golden_path.display()));
    let geometry = golden["geometry"].to_vec1::<i32>().expect("geometry");
    assert_eq!(
        geometry,
        [1024, 1024, 20, 4],
        "release oracle geometry and frozen edit-step metadata"
    );
    let device = Device::new_cuda(0).expect("CUDA device");
    let pipeline = MagePipeline::load(&root, &device).expect("load Mage-Flow");
    let mut progress = |_| {};
    let image = pipeline
        .generate(
            "a calico kitten sitting on a wooden windowsill beside a blue ceramic mug",
            " ",
            1024,
            1024,
            20,
            5.0,
            42,
            &mut progress,
        )
        .expect("1024² RL generation");
    assert_eq!((image.width, image.height), (1024, 1024));
    assert_eq!(image.pixels.len(), 1024 * 1024 * 3);
    let want = golden["image_u8"]
        .to_dtype(DType::U8)
        .unwrap()
        .flatten_all()
        .unwrap()
        .to_vec1::<u8>()
        .unwrap();
    assert_eq!(image.pixels.len(), want.len(), "oracle image shape");
    let sum_delta = image
        .pixels
        .iter()
        .zip(&want)
        .map(|(got, want)| got.abs_diff(*want) as f64)
        .sum::<f64>();
    let sum_want = want.iter().map(|value| *value as f64).sum::<f64>();
    let mean_rel = sum_delta / sum_want.max(f64::MIN_POSITIVE);
    assert!(
        mean_rel <= 0.10,
        "1024² Torch image mean_rel {mean_rel:.6} exceeds calibrated 0.10 gate"
    );
}
