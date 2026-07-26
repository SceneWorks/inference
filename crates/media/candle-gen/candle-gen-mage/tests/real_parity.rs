//! Caller-provisioned real-weight acceptance gates for sc-14051.
//!
//! These tests never discover or download model data. Point `MAGE_SNAPSHOT` at the
//! `microsoft/Mage-Flow` snapshot and `MAGE_GOLDEN_DIR` at the CPU Torch bundles produced by
//! `crates/media/mlx-gen/tools/dump_mage_flow_golden.py --stage all`.

use std::collections::HashMap;
use std::path::{Path, PathBuf};

use candle_core::{DType, Device, Tensor};
use candle_gen_mage::rope::{ImgShape, PackLayout};
use candle_gen_mage::{MageConfig, MageTextEncoder, MageTransformer, MageVae};

const PROMPT: &str = "a calico kitten sitting on a wooden windowsill beside a blue ceramic mug";

fn snapshot() -> PathBuf {
    PathBuf::from(
        std::env::var("MAGE_SNAPSHOT").expect(
            "set MAGE_SNAPSHOT to a caller-provisioned microsoft/Mage-Flow snapshot directory",
        ),
    )
}

fn test_device() -> Device {
    #[cfg(feature = "cuda")]
    {
        Device::new_cuda(0).expect("CUDA device")
    }
    #[cfg(not(feature = "cuda"))]
    {
        Device::Cpu
    }
}

fn goldens(name: &str, device: &Device) -> HashMap<String, Tensor> {
    let root = PathBuf::from(
        std::env::var("MAGE_GOLDEN_DIR")
            .expect("set MAGE_GOLDEN_DIR to the CPU Torch Mage golden directory"),
    );
    let path = root.join(name);
    candle_core::safetensors::load(&path, device)
        .unwrap_or_else(|e| panic!("load {}: {e}", path.display()))
}

fn require<'a>(map: &'a HashMap<String, Tensor>, key: &str) -> &'a Tensor {
    map.get(key)
        .unwrap_or_else(|| panic!("golden is missing required key {key}"))
}

fn stats(got: &Tensor, want: &Tensor) -> (f32, f32) {
    assert_eq!(got.dims(), want.dims(), "golden shape mismatch");
    let got = got
        .to_device(&Device::Cpu)
        .unwrap()
        .to_dtype(DType::F32)
        .unwrap()
        .flatten_all()
        .unwrap()
        .to_vec1::<f32>()
        .unwrap();
    let want = want
        .to_device(&Device::Cpu)
        .unwrap()
        .to_dtype(DType::F32)
        .unwrap()
        .flatten_all()
        .unwrap()
        .to_vec1::<f32>()
        .unwrap();
    let max_abs = got
        .iter()
        .zip(&want)
        .fold(0f32, |m, (x, y)| m.max((x - y).abs()));
    let sum_delta = got
        .iter()
        .zip(&want)
        .map(|(x, y)| (x - y).abs() as f64)
        .sum::<f64>();
    let sum_want = want.iter().map(|x| x.abs() as f64).sum::<f64>();
    (
        max_abs,
        (sum_delta / sum_want.max(f64::MIN_POSITIVE)) as f32,
    )
}

fn component_dir(root: &Path, name: &str) -> PathBuf {
    if root.file_name().is_some_and(|n| n == name) {
        root.to_path_buf()
    } else {
        root.join(name)
    }
}

#[test]
#[ignore = "needs MAGE_SNAPSHOT and MAGE_GOLDEN_DIR real Torch artifacts"]
fn final_normalized_qwen_conditioning_matches_torch() {
    let root = snapshot();
    let device = test_device();
    let model = MageTextEncoder::load(&root, &device).expect("load Qwen3-VL-4B");
    let got = model
        .encode(PROMPT)
        .expect("encode prompt")
        .squeeze(0)
        .expect("remove singleton text batch");
    let golden = goldens("mage_flow_te_golden.safetensors", &device);
    let (max_abs, mean_rel) = stats(&got, require(&golden, "gen_txt"));
    assert!(max_abs <= 3.0, "TE max_abs {max_abs} exceeds 3.0");
    assert!(mean_rel <= 3.5e-2, "TE mean_rel {mean_rel} exceeds 3.5e-2");
}

#[test]
#[ignore = "needs MAGE_SNAPSHOT and MAGE_GOLDEN_DIR real Torch artifacts"]
fn twelve_block_nr_mmdit_matches_torch() {
    let root = snapshot();
    let device = test_device();
    let dir = component_dir(&root, "transformer");
    let cfg = MageConfig::from_json(
        &std::fs::read_to_string(dir.join("config.json")).expect("transformer/config.json"),
    )
    .expect("Mage config");
    let model = MageTransformer::load(&dir, &cfg, &device).expect("load NR-MMDiT");
    let golden = goldens("mage_flow_dit_golden.safetensors", &device);
    let ints = |key: &str| {
        let tensor = require(&golden, key)
            .to_device(&Device::Cpu)
            .unwrap()
            .flatten_all()
            .unwrap();
        match tensor.dtype() {
            DType::I32 => tensor.to_vec1::<i32>().unwrap(),
            DType::I64 => tensor
                .to_vec1::<i64>()
                .unwrap()
                .into_iter()
                .map(|value| i32::try_from(value).expect("golden integer fits i32"))
                .collect(),
            dtype => panic!("golden {key} must be I32 or I64, got {dtype:?}"),
        }
    };
    let shapes = ints("img_shapes")
        .chunks_exact(3)
        .map(|s| ImgShape {
            frames: s[0] as usize,
            height: s[1] as usize,
            width: s[2] as usize,
        })
        .collect();
    let lens = |cu: Vec<i32>| {
        cu.windows(2)
            .map(|w| (w[1] - w[0]) as usize)
            .collect::<Vec<_>>()
    };
    let layout = PackLayout::new(
        shapes,
        lens(ints("dit_in.img_cu_seqlens")),
        lens(ints("dit_in.txt_cu_seqlens")),
    )
    .expect("golden packing");
    let got = model
        .forward(
            require(&golden, "dit_in.img"),
            require(&golden, "dit_in.txt"),
            require(&golden, "dit_in.timesteps"),
            &layout,
        )
        .expect("12-block forward");
    let (max_abs, mean_rel) = stats(&got, require(&golden, "dit_out"));
    let peak = require(&golden, "dit_out")
        .to_dtype(DType::F32)
        .unwrap()
        .abs()
        .unwrap()
        .max_all()
        .unwrap()
        .to_scalar::<f32>()
        .unwrap()
        .max(1e-12);
    assert!(
        max_abs / peak <= 4.0e-2,
        "DiT max_rel {} exceeds 4.0e-2",
        max_abs / peak
    );
    assert!(mean_rel <= 4.0e-2, "DiT mean_rel {mean_rel} exceeds 4.0e-2");
}

#[test]
#[ignore = "needs MAGE_SNAPSHOT and MAGE_GOLDEN_DIR real Torch artifacts"]
fn mage_vae_1024_decode_matches_torch() {
    let root = snapshot();
    let device = test_device();
    let model =
        MageVae::load(&component_dir(&root, "vae"), &device).expect("load Mage-VAE decoder");
    let golden = goldens("mage_flow_vae_f32_1024.safetensors", &device);
    let mut failures = Vec::new();
    for (input, output) in [
        ("synth_latent", "dec_from_synth"),
        ("enc_latent", "dec_from_latent"),
    ] {
        let got = model.decode(require(&golden, input)).expect("VAE decode");
        let (max_abs, mean_rel) = stats(&got, require(&golden, output));
        println!("{output}: max_abs={max_abs:.8}, mean_rel={mean_rel:.8}");
        if max_abs > 4.0e-2 {
            failures.push(format!("{output} max_abs {max_abs} exceeds 4.0e-2"));
        }
        if mean_rel > 1.5e-3 {
            failures.push(format!("{output} mean_rel {mean_rel} exceeds 1.5e-3"));
        }
    }
    assert!(
        failures.is_empty(),
        "VAE decode parity failures:\n{}",
        failures.join("\n")
    );
}

#[test]
#[ignore = "needs MAGE_SNAPSHOT and MAGE_GOLDEN_DIR real Torch artifacts"]
fn mage_vae_1024_encoder_moments_match_torch() {
    let root = snapshot();
    let device = test_device();
    let model = MageVae::load_full_dtype(&component_dir(&root, "vae"), &device, DType::F32)
        .expect("load full f32 Mage-VAE");
    let golden = goldens("mage_flow_vae_f32_1024.safetensors", &device);
    let moments = model
        .encode_moments(require(&golden, "pixels"))
        .expect("VAE encode moments");
    for (label, got, want) in [
        ("enc_mean", &moments.mean, require(&golden, "enc_mean")),
        (
            "enc_logvar",
            &moments.logvar,
            require(&golden, "enc_logvar"),
        ),
    ] {
        let (max_abs, mean_rel) = stats(got, want);
        assert!(max_abs <= 4.0e-2, "{label} max_abs {max_abs}");
        assert!(mean_rel <= 2.0e-3, "{label} mean_rel {mean_rel}");
    }
}
