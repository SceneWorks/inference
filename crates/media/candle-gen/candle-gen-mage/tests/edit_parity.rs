//! Cross-backend Mage Edit / Edit-Base / Edit-Turbo acceptance.
//!
//! Each bundle is produced by the frozen Torch reference and separately consumed by the MLX suite.
//! Candle enters through the registered provider and production reference-image path, covering
//! checkpoint identity, resize, Qwen3-VL conditioning, Mage-VAE posterior sampling, packed edit
//! DiT math, target-only Euler integration, and decode. The final-image tolerance absorbs the
//! deliberately different random-normal algorithms used by the three runtimes.

#![cfg(feature = "cuda")]

use std::collections::HashMap;
use std::path::{Path, PathBuf};

use candle_core::{DType, Device, Tensor};
use candle_gen::gen_core::{
    Conditioning, GenerationOutput, GenerationRequest, Image, LoadSpec, WeightsSource,
};

const INSTRUCTION: &str = "Replace the background with a field of sunflowers";

struct Case {
    id: &'static str,
    label: &'static str,
    snapshot_env: &'static str,
    golden: &'static str,
    expected_steps: usize,
    expected_cfg: f32,
}

const CASES: &[Case] = &[
    Case {
        id: "mage_flow_edit",
        label: "Edit",
        snapshot_env: "MAGE_EDIT_SNAPSHOT",
        golden: "mage_flow_edit_golden.safetensors",
        expected_steps: 30,
        expected_cfg: 5.0,
    },
    Case {
        id: "mage_flow_edit_base",
        label: "Edit-Base",
        snapshot_env: "MAGE_EDIT_BASE_SNAPSHOT",
        golden: "mage_flow_edit_base_golden.safetensors",
        expected_steps: 30,
        expected_cfg: 5.0,
    },
    Case {
        id: "mage_flow_edit_turbo",
        label: "Edit-Turbo",
        snapshot_env: "MAGE_EDIT_TURBO_SNAPSHOT",
        golden: "mage_flow_edit_turbo_golden.safetensors",
        expected_steps: 4,
        expected_cfg: 1.0,
    },
];

fn load(path: &Path, device: &Device) -> HashMap<String, Tensor> {
    candle_core::safetensors::load(path, device)
        .unwrap_or_else(|error| panic!("load {}: {error}", path.display()))
}

fn mean_relative_u8(got: &[u8], want: &[u8]) -> f64 {
    let delta = got
        .iter()
        .zip(want)
        .map(|(a, b)| a.abs_diff(*b) as f64)
        .sum::<f64>();
    let scale = want.iter().map(|value| *value as f64).sum::<f64>();
    delta / scale.max(f64::MIN_POSITIVE)
}

#[test]
#[ignore = "needs all three Mage edit snapshots, Torch goldens, and CUDA"]
fn all_registered_edit_variants_match_torch_and_mlx_oracles() {
    let golden_root = PathBuf::from(
        std::env::var("MAGE_GOLDEN_DIR").expect("set MAGE_GOLDEN_DIR to transferred Torch oracles"),
    );
    let device = Device::new_cuda(0).expect("CUDA device");
    for case in CASES {
        let snapshot = PathBuf::from(
            std::env::var(case.snapshot_env)
                .unwrap_or_else(|_| panic!("set {} for {}", case.snapshot_env, case.label)),
        );
        let golden = load(&golden_root.join(case.golden), &device);
        let geometry = golden["geometry"]
            .to_device(&Device::Cpu)
            .unwrap()
            .to_vec1::<i32>()
            .unwrap();
        let (height, width, steps) = (geometry[0] as u32, geometry[1] as u32, geometry[3] as usize);
        let cfg = golden["cfg"].to_vec1::<f32>().unwrap()[0];
        let seed = golden["seed"].to_vec1::<i64>().unwrap()[0] as u64;
        assert_eq!(steps, case.expected_steps, "{} step policy", case.label);
        assert_eq!(cfg, case.expected_cfg, "{} CFG policy", case.label);

        let source = golden["ref_u8"]
            .to_device(&Device::Cpu)
            .unwrap()
            .to_dtype(DType::U8)
            .unwrap()
            .flatten_all()
            .unwrap()
            .to_vec1::<u8>()
            .unwrap();
        let reference = Image {
            width,
            height,
            pixels: source,
        };
        let registry = candle_gen_mage::provider_registry().expect("Mage registry");
        let generator = registry
            .load(
                case.id,
                &LoadSpec::new(WeightsSource::Dir(snapshot.clone())),
            )
            .unwrap_or_else(|error| panic!("load {} provider: {error}", case.label));
        let output = generator
            .generate(
                &GenerationRequest {
                    prompt: INSTRUCTION.into(),
                    width,
                    height,
                    seed: Some(seed),
                    conditioning: vec![Conditioning::Reference {
                        image: reference.clone(),
                        strength: None,
                    }],
                    ..Default::default()
                },
                &mut |_| {},
            )
            .unwrap_or_else(|error| panic!("{} registered edit: {error}", case.label));
        let image = match output {
            GenerationOutput::Images(mut images) if images.len() == 1 => images.remove(0),
            other => panic!("{} returned unexpected output: {other:?}", case.label),
        };
        let want = golden["image_u8"]
            .to_device(&Device::Cpu)
            .unwrap()
            .to_dtype(DType::U8)
            .unwrap()
            .flatten_all()
            .unwrap()
            .to_vec1::<u8>()
            .unwrap();
        let mean_rel = mean_relative_u8(&image.pixels, &want);
        assert!(
            mean_rel <= 0.10,
            "{} Torch/MLX/Candle image mean_rel {mean_rel:.6} exceeds 0.10",
            case.label
        );
        // A constant-white/black oracle would let the tolerance look healthy while carrying no
        // semantic signal. Pin a real dynamic range and reject that false-green class.
        let (min, max) = want
            .iter()
            .fold((u8::MAX, u8::MIN), |(lo, hi), &v| (lo.min(v), hi.max(v)));
        assert!(
            max.saturating_sub(min) >= 64,
            "{} oracle is non-discriminating ({min}..={max})",
            case.label
        );
        if case.id == "mage_flow_edit_turbo" {
            let mut mutated = reference;
            for value in &mut mutated.pixels {
                *value = 255 - *value;
            }
            let changed = generator
                .generate(
                    &GenerationRequest {
                        prompt: INSTRUCTION.into(),
                        width,
                        height,
                        seed: Some(seed),
                        conditioning: vec![Conditioning::Reference {
                            image: mutated,
                            strength: None,
                        }],
                        ..Default::default()
                    },
                    &mut |_| {},
                )
                .expect("Edit-Turbo mutated-reference render");
            let changed = match changed {
                GenerationOutput::Images(mut images) if images.len() == 1 => images.remove(0),
                other => panic!("Edit-Turbo mutation returned unexpected output: {other:?}"),
            };
            let mutation = mean_relative_u8(&changed.pixels, &image.pixels);
            assert!(
                mutation >= 0.01,
                "Candle edit ignored a load-bearing reference mutation ({mutation:.6})"
            );
        }
    }
}
