//! SYNTHETIC source-derived training AE CPU tests. BF16/F16 convolutions here
//! explicitly emulate rounded operands with F32 accumulation, NOT CUDA kernels.
use candle_gen::candle_core::{DType, Device, Tensor};
use candle_gen_flux3::autoencoder::FrozenTrainingEncoder;

use crate::common::{fixture, host_f32, tiny_autoencoder};

fn difference(got: &Tensor, want: &Tensor) -> (f32, f32) {
    assert_eq!(got.dims(), want.dims());
    let a = host_f32(got);
    let b = host_f32(want);
    let diffs: Vec<_> = a.iter().zip(&b).map(|(a, b)| (a - b).abs()).collect();
    assert!(a.iter().all(|x| x.is_finite()));
    (
        diffs.iter().copied().fold(0., f32::max),
        diffs.iter().sum::<f32>() / diffs.len() as f32,
    )
}

#[test]
fn frozen_training_encoder_selected_precision_cpu_emulation() {
    let golden = fixture("flux3_ae_training_precision.safetensors");
    for (dtype, name, max_bound, mean_bound) in [
        (DType::F32, "f32", 0.0002, 0.00004),
        (DType::BF16, "bf16", 0.08, 0.02),
        (DType::F16, "f16", 0.01, 0.002),
    ] {
        let encoder =
            FrozenTrainingEncoder::load(&tiny_autoencoder(), dtype, &Device::Cpu).unwrap();
        assert_eq!(encoder.compute_dtype(), dtype);
        assert_eq!(encoder.latent_channels(), 16);
        let image = golden.require("image");
        let strided = image
            .transpose(2, 3)
            .unwrap()
            .contiguous()
            .unwrap()
            .transpose(2, 3)
            .unwrap();
        assert!(!strided.is_contiguous());
        let (tokens, grid) = encoder.encode(&strided).unwrap();
        assert_eq!(grid, (2, 3));
        assert_eq!(tokens.dims(), [6, 16]);
        assert_eq!(tokens.dtype(), DType::F32);
        let (max, mean) = difference(&tokens, golden.require(&format!("tokens_{name}")));
        println!("{name} CPU emulation |delta| max={max:.9} mean={mean:.9}; bounds max={max_bound} mean={mean_bound}");
        assert!(
            max < max_bound && mean < mean_bound,
            "{name} CPU opmath max={max} mean={mean}"
        );
        if dtype == DType::BF16 {
            let (wrong, _) = difference(&tokens, golden.require("wrong_final_only_bf16"));
            assert!(
                wrong > 0.001,
                "intermediate lowprecision must not be final-cast-only"
            );
        }
    }
}

#[test]
fn frozen_training_encoder_input_layout_and_dtype_refusals() {
    let ae = FrozenTrainingEncoder::load(&tiny_autoencoder(), DType::BF16, &Device::Cpu).unwrap();
    assert!(FrozenTrainingEncoder::load(&tiny_autoencoder(), DType::U32, &Device::Cpu).is_err());
    for shape in [
        (1, 3, 0, 32),
        (1, 3, 31, 32),
        (2, 3, 32, 48),
        (1, 4, 32, 48),
    ] {
        let image = Tensor::zeros(shape, DType::F32, &Device::Cpu).unwrap();
        assert!(ae.encode(&image).is_err(), "invalid shape {shape:?}");
    }
    let image = Tensor::zeros((1, 3, 32, 48), DType::U32, &Device::Cpu).unwrap();
    assert!(ae.encode(&image).is_err());
}
