//! sc-19753 — [`SdxlVaeDecoder`] is a **port** of `candle_transformers`' SDXL VAE decoder, and its
//! bounded route is normalization-correct.
//!
//! The upstream `AutoEncoderKL` keeps its `Decoder`, resnets and `GroupNorm`s private, so the
//! layer-wise bounded decode sc-19753 requires could not be built on top of it and the decode half
//! was ported into this crate. A port is only as good as its equivalence proof, so the first test
//! here builds **both** decoders from the same synthetic checkpoint and asserts their dense decodes
//! agree — that is what catches a key-naming, layer-ordering, group-count, epsilon,
//! `output_scale_factor` or dtype divergence.
//!
//! Everything runs weights-free on CPU at a narrow synthetic config.

use std::collections::HashMap;

use candle_gen::candle_core::{DType, Device, Tensor};
use candle_gen::candle_nn::{VarBuilder, VarMap};
use candle_gen::gen_core::CancelFlag;
use candle_gen::CandleError;
use candle_gen_sdxl::SdxlVaeDecoder;
use candle_transformers::models::stable_diffusion::vae::{AutoEncoderKL, AutoEncoderKLConfig};

/// A narrow but structurally faithful SDXL-family VAE: two `block_out_channels` stages (so the
/// decoder has one upsampling block and one final block), the real `layers_per_block + 1` decoder
/// resnet count, a `conv_shortcut`-bearing stage (16 → 8), and both quant convolutions.
fn config() -> AutoEncoderKLConfig {
    AutoEncoderKLConfig {
        block_out_channels: vec![8, 16],
        layers_per_block: 1,
        latent_channels: 4,
        norm_num_groups: 4,
        use_quant_conv: true,
        use_post_quant_conv: true,
    }
}

/// Deterministic, non-degenerate values for every tensor in the checkpoint.
///
/// The weights are built by letting upstream's own constructor declare the key set and shapes, then
/// overwriting every value. Overwriting matters: candle initializes `GroupNorm` affines to
/// `weight = 1, bias = 0`, under which a mishandled affine would be invisible.
fn checkpoint(dtype: DType, device: &Device) -> HashMap<String, Tensor> {
    let vars = VarMap::new();
    AutoEncoderKL::new(
        VarBuilder::from_varmap(&vars, DType::F32, device),
        3,
        3,
        config(),
    )
    .expect("declare the upstream key set");

    let data = vars.data().lock().unwrap();
    let mut names: Vec<String> = data.keys().cloned().collect();
    names.sort(); // stable phase assignment regardless of hash order
    let mut out = HashMap::new();
    for (phase, name) in names.iter().enumerate() {
        let shape = data[name].shape().clone();
        let count = shape.elem_count();
        let values = (0..count)
            .map(|i| ((i + phase * 7) as f32 * 0.071).sin() * 0.4 + 0.05)
            .collect::<Vec<_>>();
        out.insert(
            name.clone(),
            Tensor::from_vec(values, shape, device)
                .unwrap()
                .to_dtype(dtype)
                .unwrap(),
        );
    }
    out
}

/// Position-dependent latents, so per-crop GroupNorm statistics differ observably from the dense
/// ones and the bounded/dense comparison below discriminates the defect.
fn latent(h: usize, w: usize, dtype: DType, device: &Device) -> Tensor {
    let c = config().latent_channels;
    let values = (0..(c * h * w))
        .map(|i| {
            let y = (i / w % h) as f32;
            let x = (i % w) as f32;
            (i as f32 * 0.037).sin() + y * 0.11 - x * 0.07
        })
        .collect::<Vec<_>>();
    Tensor::from_vec(values, (1, c, h, w), device)
        .unwrap()
        .to_dtype(dtype)
        .unwrap()
}

fn max_abs(a: &Tensor, b: &Tensor) -> f32 {
    (a.to_dtype(DType::F32).unwrap() - b.to_dtype(DType::F32).unwrap())
        .unwrap()
        .abs()
        .unwrap()
        .flatten_all()
        .unwrap()
        .max(0)
        .unwrap()
        .to_scalar::<f32>()
        .unwrap()
}

/// The port must reproduce the stock decoder. Run at f32 *and* f16 because candle's `GroupNorm`
/// computes its statistics in f32 for half-precision inputs and casts back **before** the affine —
/// a port that normalized entirely in the input dtype would pass at f32 and drift at f16, which is
/// the dtype the production fp16-fix VAE actually loads at.
#[test]
fn native_dense_decode_matches_the_stock_autoencoder_kl() {
    let device = Device::Cpu;
    for (dtype, bound) in [(DType::F32, 1e-5_f32), (DType::F16, 6e-3)] {
        let tensors = checkpoint(dtype, &device);
        let stock = AutoEncoderKL::new(
            VarBuilder::from_tensors(tensors.clone(), dtype, &device),
            3,
            3,
            config(),
        )
        .unwrap();
        let native = SdxlVaeDecoder::new(
            VarBuilder::from_tensors(tensors, dtype, &device),
            3,
            &config(),
        )
        .unwrap();

        let z = latent(6, 7, dtype, &device);
        let want = stock.decode(&z).unwrap();
        let got = native.decode(&z).unwrap();
        assert_eq!(got.dims(), want.dims(), "{dtype:?} geometry");
        let delta = max_abs(&got, &want);
        assert!(
            delta < bound,
            "{dtype:?}: the ported decoder diverged from candle's AutoEncoderKL by max|delta|={delta:.3e}"
        );
    }
}

/// The layer-wise bounded decode must track the dense decode, at f32 **and** at the f16 the
/// production fp16-fix VAE actually loads at.
///
/// f16 is the load-bearing case for [`GlobalGroupNorm`]'s dtype dance: candle's `GroupNorm`
/// computes its statistics in f32 for half-precision inputs and casts back before the affine, and
/// only the tiled path exercises the ported reimplementation of that. A port that normalized
/// entirely in the input dtype would pass every f32 row here and drift in production.
///
/// The companion `per_tile_normalization_is_observably_wrong` executes the defect on this same
/// decoder, so the bounds below are known to discriminate rather than assumed to.
#[test]
fn bounded_decode_tracks_the_dense_decode() {
    let device = Device::Cpu;
    for (dtype, bound) in [(DType::F32, 2e-5_f32), (DType::F16, 6e-3)] {
        let tensors = checkpoint(dtype, &device);
        let native = SdxlVaeDecoder::new(
            VarBuilder::from_tensors(tensors, dtype, &device),
            3,
            &config(),
        )
        .unwrap();

        let z = latent(6, 7, dtype, &device);
        let dense = native.decode(&z).unwrap();
        let (_, _, out_h, out_w) = dense.dims4().unwrap();
        // One upsampling block ⇒ ×2. A 6-pixel output edge genuinely splits a 12×14 output.
        let edge = 6usize;
        assert!(
            out_h > edge && out_w > edge,
            "the bounded request must actually split the {out_h}x{out_w} output"
        );

        let bounded = native.decode_tiled(&z, edge, None).unwrap();
        assert_eq!(bounded.dims(), dense.dims(), "{dtype:?} geometry");
        let delta = max_abs(&bounded, &dense);
        assert!(
            delta < bound,
            "{dtype:?}: layer-wise bounded SDXL decode diverged from dense: max|delta|={delta:.3e}"
        );
    }
}

/// The **executed** defect control for the bound above (sc-19753).
///
/// Reproduces the per-tile normalization this story removed — decode each latent crop
/// independently and stitch — and asserts it is observably different from the dense decode. That is
/// what makes `bounded_decode_tracks_the_dense_decode`'s tolerance meaningful: without this, a
/// bound that any implementation satisfies would look like a passing guard.
///
/// It also runs at both dtypes, because a bound loose enough for f16 is the easiest one to satisfy
/// accidentally.
#[test]
fn per_tile_normalization_is_observably_wrong() {
    let device = Device::Cpu;
    for dtype in [DType::F32, DType::F16] {
        let tensors = checkpoint(dtype, &device);
        let native = SdxlVaeDecoder::new(
            VarBuilder::from_tensors(tensors, dtype, &device),
            3,
            &config(),
        )
        .unwrap();

        let z = latent(6, 7, dtype, &device);
        let dense = native.decode(&z).unwrap();

        // The retired shape: decode each latent half on its own and concatenate. Each half's
        // GroupNorms see only their own crop and the mid block attends only to their own tokens.
        let left = native.decode(&z.narrow(3, 0, 3).unwrap()).unwrap();
        let right = native.decode(&z.narrow(3, 3, 4).unwrap()).unwrap();
        let per_tile = candle_gen::candle_core::Tensor::cat(&[left, right], 3).unwrap();
        assert_eq!(per_tile.dims(), dense.dims());

        let delta = max_abs(&per_tile, &dense);
        assert!(
            delta > 1e-2,
            "{dtype:?}: per-crop normalization must be observably different from the dense decode, \
             else the bounded-decode tolerance proves nothing: max|delta|={delta:.3e}"
        );
    }
}

/// An edge wide enough to hold the whole output takes the single-tile fall-through inside every
/// layer, so it is the dense computation up to floating-point reassociation — [`GlobalGroupNorm`]
/// computes `x·(1/σ)` where candle's `GroupNorm` computes `x/σ`. This is the control that separates
/// "bounded and correct" from "never actually bounded": it must land orders of magnitude tighter
/// than a genuinely tiled decode.
#[test]
fn a_wide_edge_falls_through_to_the_dense_computation() {
    let device = Device::Cpu;
    let dtype = DType::F32;
    let tensors = checkpoint(dtype, &device);
    let native = SdxlVaeDecoder::new(
        VarBuilder::from_tensors(tensors, dtype, &device),
        3,
        &config(),
    )
    .unwrap();
    let z = latent(6, 7, dtype, &device);
    let dense = native.decode(&z).unwrap();
    let bounded = native.decode_tiled(&z, 4096, None).unwrap();
    let delta = max_abs(&bounded, &dense);
    assert!(
        delta < 1e-6,
        "the single-tile fall-through must be the dense computation: max|delta|={delta:.3e}"
    );
}

#[test]
fn bounded_decode_honors_a_pretripped_cancel() {
    let device = Device::Cpu;
    let dtype = DType::F32;
    let tensors = checkpoint(dtype, &device);
    let native = SdxlVaeDecoder::new(
        VarBuilder::from_tensors(tensors, dtype, &device),
        3,
        &config(),
    )
    .unwrap();
    let z = latent(6, 7, dtype, &device);
    let cancel = CancelFlag::new();
    cancel.cancel();
    for edge in [6usize, 4096] {
        let result = native.decode_tiled(&z, edge, Some(&cancel));
        assert!(matches!(result, Err(CandleError::Canceled)));
    }
}
