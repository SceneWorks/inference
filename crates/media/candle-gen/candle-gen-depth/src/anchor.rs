//! **Depth anchoring** for the Candle trainers (epic 2123, sc-24830) — Depth-Anything-V2 as a
//! frozen, differentiable perceptor plugged into the shared perceptual-loss path
//! ([`candle_gen::train::perceptual`]); the Candle twin of `mlx_gen_depth::anchor` (sc-2125).

use std::any::Any;
use std::collections::HashMap;
use std::path::Path;

use candle_gen::candle_core::{Device, Tensor};
use candle_gen::gen_core::train::DepthModelSize;
use candle_gen::train::perceptual::{
    depth_consistency_loss, reference_as, AuxModelFootprint, LossReference, PerceptualLoss,
};
use candle_gen::Result;

use crate::common::Weights;
use crate::{DepthAnythingConfig, DepthAnythingV2};

/// Frozen Depth-Anything-V2 + the MiDaS depth-consistency comparison.
pub struct DepthAnchorLoss {
    model: DepthAnythingV2,
}

/// The per-image depth-anchoring reference: the DA2 depth of the image's clean round trip.
pub struct DepthReference {
    /// `[B, h, w]` relative depth at the aspect-preserving model size (detached).
    pub depth: Tensor,
}

impl DepthAnchorLoss {
    /// Wrap a loaded estimator.
    pub fn new(model: DepthAnythingV2) -> Self {
        Self { model }
    }

    /// Load the `size` checkpoint from `dir` (a `Depth-Anything-V2-{Small,Base,Large}-hf`
    /// snapshot) onto `device`.
    pub fn from_dir(dir: impl AsRef<Path>, size: DepthModelSize, device: &Device) -> Result<Self> {
        let w = Weights::from_dir(dir, device)?;
        Ok(Self::new(DepthAnythingV2::from_weights(
            &w,
            DepthAnythingConfig::for_size(size),
            device,
        )?))
    }

    /// The wrapped estimator.
    pub fn model(&self) -> &DepthAnythingV2 {
        &self.model
    }
}

impl PerceptualLoss for DepthAnchorLoss {
    fn name(&self) -> &'static str {
        "depth"
    }

    /// The clean round trip's depth. Every image is usable for depth.
    fn reference(&self, clean: &Tensor) -> Result<Option<LossReference>> {
        let depth = self.model.forward_pixels(clean)?.detach();
        Ok(Some(Box::new(DepthReference { depth })))
    }

    /// Live decoded pixels `[B, H, W, 3]` → DA2 depth → SSI-L1 + multi-scale gradient vs the
    /// reference depth (same shape: same-sized decode, same aspect-preserving resize).
    fn loss(&self, live: &Tensor, reference: &dyn Any) -> Result<Tensor> {
        let r = reference_as::<DepthReference>(self.name(), reference)?;
        depth_consistency_loss(&self.model.forward_pixels(live)?, &r.depth)
    }
}

/// The pre-load memory figures of depth anchoring with a DA2 checkpoint of `size` on
/// `image_h × image_w` training images (epic 2123 E7): resident f32 weights, one differentiable
/// forward/backward at the native size, and one cached depth map per image at the
/// aspect-preserving model size. Identical accounting to the MLX twin.
pub fn depth_anchor_footprint(
    size: DepthModelSize,
    image_h: u32,
    image_w: u32,
) -> AuxModelFootprint {
    let cfg = DepthAnythingConfig::for_size(size);
    let (h, w) = cfg.input_hw(image_h as usize, image_w as usize);
    AuxModelFootprint {
        param_bytes: cfg.param_count() * 4,
        working_set_bytes: cfg.training_working_set_bytes(),
        reference_bytes_per_image: h as u64 * w as u64 * 4,
    }
}

/// A complete random-init checkpoint for `cfg` (every key [`DepthAnythingV2::from_weights`]
/// requires, torch OIHW/IOHW layouts) — for tests of depth anchoring and the trainers that use it.
/// Deterministic in `seed`; scaled so activations stay O(1).
pub fn synthetic_weights(cfg: &DepthAnythingConfig, seed: u64, device: &Device) -> Result<Weights> {
    let mut map: HashMap<String, Tensor> = HashMap::new();
    let mut state = seed.wrapping_mul(0x9E37_79B9_7F4A_7C15).wrapping_add(7_919);
    let mut rnd = |shape: &[usize]| -> Result<Tensor> {
        let fan_in: usize = if shape.len() > 1 {
            shape[1..].iter().product()
        } else {
            shape[0]
        };
        let std = (1.0 / fan_in.max(1) as f32).sqrt();
        let n: usize = shape.iter().product();
        let v: Vec<f32> = (0..n)
            .map(|_| {
                let mut s = 0f32;
                for _ in 0..4 {
                    state = state
                        .wrapping_mul(6364136223846793005)
                        .wrapping_add(1442695040888963407);
                    s += ((state >> 40) as f32) / (1u64 << 24) as f32 - 0.5;
                }
                s * 3f32.sqrt() * std
            })
            .collect();
        Ok(Tensor::from_vec(v, shape, device)?)
    };
    let ones = |shape: &[usize]| -> Result<Tensor> {
        Ok(Tensor::ones(
            shape,
            candle_gen::candle_core::DType::F32,
            device,
        )?)
    };
    let h = cfg.hidden_size;
    let grid = cfg.grid();
    let inter = cfg.intermediate_size();
    let mut put = |k: String, t: Tensor| {
        map.insert(k, t);
    };
    put(
        "backbone.embeddings.patch_embeddings.projection.weight".into(),
        rnd(&[h, cfg.num_channels, cfg.patch_size, cfg.patch_size])?,
    );
    put(
        "backbone.embeddings.patch_embeddings.projection.bias".into(),
        rnd(&[h])?,
    );
    put("backbone.embeddings.cls_token".into(), rnd(&[1, 1, h])?);
    put(
        "backbone.embeddings.position_embeddings".into(),
        rnd(&[1, grid * grid + 1, h])?,
    );
    for i in 0..cfg.num_hidden_layers {
        let p = format!("backbone.encoder.layer.{i}");
        for leaf in ["norm1", "norm2"] {
            put(format!("{p}.{leaf}.weight"), ones(&[h])?);
            put(format!("{p}.{leaf}.bias"), rnd(&[h])?);
        }
        for leaf in ["query", "key", "value"] {
            put(
                format!("{p}.attention.attention.{leaf}.weight"),
                rnd(&[h, h])?,
            );
            put(format!("{p}.attention.attention.{leaf}.bias"), rnd(&[h])?);
        }
        put(format!("{p}.attention.output.dense.weight"), rnd(&[h, h])?);
        put(format!("{p}.attention.output.dense.bias"), rnd(&[h])?);
        put(format!("{p}.layer_scale1.lambda1"), ones(&[h])?);
        put(format!("{p}.layer_scale2.lambda1"), ones(&[h])?);
        put(format!("{p}.mlp.fc1.weight"), rnd(&[inter, h])?);
        put(format!("{p}.mlp.fc1.bias"), rnd(&[inter])?);
        put(format!("{p}.mlp.fc2.weight"), rnd(&[h, inter])?);
        put(format!("{p}.mlp.fc2.bias"), rnd(&[h])?);
    }
    put("backbone.layernorm.weight".into(), ones(&[h])?);
    put("backbone.layernorm.bias".into(), rnd(&[h])?);
    let fh = cfg.fusion_hidden_size;
    for i in 0..4 {
        let nh = cfg.neck_hidden_sizes[i];
        let p = format!("neck.reassemble_stage.layers.{i}");
        put(format!("{p}.projection.weight"), rnd(&[nh, h, 1, 1])?);
        put(format!("{p}.projection.bias"), rnd(&[nh])?);
        let factor = cfg.reassemble_factors[i];
        if factor > 1.0 {
            let k = factor as usize;
            put(format!("{p}.resize.weight"), rnd(&[nh, nh, k, k])?);
            put(format!("{p}.resize.bias"), rnd(&[nh])?);
        } else if factor < 1.0 {
            put(format!("{p}.resize.weight"), rnd(&[nh, nh, 3, 3])?);
            put(format!("{p}.resize.bias"), rnd(&[nh])?);
        }
        put(format!("neck.convs.{i}.weight"), rnd(&[fh, nh, 3, 3])?);
        let fp = format!("neck.fusion_stage.layers.{i}");
        for res in ["residual_layer1", "residual_layer2"] {
            for c in ["convolution1", "convolution2"] {
                put(format!("{fp}.{res}.{c}.weight"), rnd(&[fh, fh, 3, 3])?);
                put(format!("{fp}.{res}.{c}.bias"), rnd(&[fh])?);
            }
        }
        put(format!("{fp}.projection.weight"), rnd(&[fh, fh, 1, 1])?);
        put(format!("{fp}.projection.bias"), rnd(&[fh])?);
    }
    let hh = cfg.head_hidden_size;
    let half = fh / 2;
    put("head.conv1.weight".into(), rnd(&[half, fh, 3, 3])?);
    put("head.conv1.bias".into(), rnd(&[half])?);
    put("head.conv2.weight".into(), rnd(&[hh, half, 3, 3])?);
    put("head.conv2.bias".into(), rnd(&[hh])?);
    put("head.conv3.weight".into(), rnd(&[1, hh, 1, 1])?);
    // A positive bias keeps the final ReLU from zeroing the whole random-init map.
    put(
        "head.conv3.bias".into(),
        Tensor::from_slice(&[1.0f32], 1, device)?,
    );
    Ok(Weights::from_map(map))
}

/// A tiny DA2-shaped config (8-dim embed, grid 4 / image 8) that keeps the real factor ladder —
/// for synthetic tests (same as the MLX twin's).
pub fn tiny_config() -> DepthAnythingConfig {
    DepthAnythingConfig {
        hidden_size: 8,
        num_hidden_layers: 4,
        num_attention_heads: 2,
        mlp_ratio: 2,
        num_channels: 3,
        image_size: 8,
        patch_size: 2,
        layer_norm_eps: 1e-6,
        out_indices: [1, 2, 3, 4],
        neck_hidden_sizes: [3, 4, 5, 6],
        reassemble_factors: [4.0, 2.0, 1.0, 0.5],
        fusion_hidden_size: 6,
        head_hidden_size: 4,
    }
}

/// A tiny random-init [`DepthAnchorLoss`] on `device` (tests of trainers on this path).
pub fn tiny_depth_anchor_loss(seed: u64, device: &Device) -> Result<DepthAnchorLoss> {
    let cfg = tiny_config();
    Ok(DepthAnchorLoss::new(DepthAnythingV2::from_weights(
        &synthetic_weights(&cfg, seed, device)?,
        cfg,
        device,
    )?))
}

#[cfg(test)]
mod tests {
    use super::*;
    use candle_gen::candle_core::Var;

    /// The analytic parameter count matches the published checkpoints' `model.safetensors` sizes.
    /// Mutation: drop the neck `convs` term from `param_count` ⇒ red.
    #[test]
    fn param_count_matches_the_published_checkpoint_sizes() {
        for (size, file_bytes) in [
            (DepthModelSize::Small, 99_173_660u64),
            (DepthModelSize::Base, 389_916_980),
            (DepthModelSize::Large, 1_341_322_868),
        ] {
            let bytes = DepthAnythingConfig::for_size(size).param_count() * 4;
            assert!(
                bytes <= file_bytes && file_bytes - bytes < 200_000,
                "{size:?}: {bytes} analytic vs {file_bytes} on disk"
            );
        }
    }

    fn pixels(seed: u64, h: usize, w: usize) -> Tensor {
        let n = h * w * 3;
        let mut x = seed.wrapping_mul(2654435761).wrapping_add(17);
        let v: Vec<f32> = (0..n)
            .map(|_| {
                x = x
                    .wrapping_mul(6364136223846793005)
                    .wrapping_add(1442695040888963407);
                ((x >> 40) as f32) / (1u64 << 24) as f32
            })
            .collect();
        Tensor::from_vec(v, (1, h, w, 3), &Device::Cpu).unwrap()
    }

    /// The DA2 forward is differentiable end to end in its pixel input, square and non-square — the
    /// gradient reaches the pixels through every backbone/neck/head op. Mutation: route the
    /// backbone LayerNorm through candle's fused `candle_nn::ops::layer_norm` (`apply_op3_no_bwd`)
    /// ⇒ the pixels get no gradient ⇒ red.
    #[test]
    fn depth_anchor_loss_is_differentiable_in_the_pixels() {
        let loss = tiny_depth_anchor_loss(4, &Device::Cpu).unwrap();
        for (h, w, depth_hw) in [(16, 16, (8, 8)), (16, 24, (6, 8))] {
            let reference = loss.reference(&pixels(2, h, w)).unwrap().expect("usable");
            let r = reference.downcast_ref::<DepthReference>().unwrap();
            assert_eq!(r.depth.dims(), &[1, depth_hw.0, depth_hw.1], "{h}x{w}");
            let px = Var::from_tensor(&pixels(1, h, w)).unwrap();
            let l = loss.loss(px.as_tensor(), r).unwrap();
            let grads = l.backward().unwrap();
            let g = grads
                .get(px.as_tensor())
                .unwrap_or_else(|| panic!("{h}x{w}: the pixels get no gradient"));
            let mag = g
                .abs()
                .unwrap()
                .sum_all()
                .unwrap()
                .to_scalar::<f32>()
                .unwrap();
            assert!(
                mag > 0.0 && mag.is_finite(),
                "{h}x{w}: pixel gradient {mag}"
            );
        }
    }

    /// A non-square input is resized aspect-preservingly and a native square input never
    /// resamples the position embedding. Mutation: resize to `(image_size, image_size)` in
    /// `forward_pixels` ⇒ the 16×24 depth is 8×8 ⇒ red.
    #[test]
    fn non_square_inputs_keep_their_aspect() {
        let cfg = tiny_config();
        assert_eq!(cfg.input_hw(16, 24), (6, 8));
        assert_eq!(cfg.input_hw(24, 16), (8, 6));
        assert_eq!(cfg.input_hw(2, 100), (2, 8));
        let loss = tiny_depth_anchor_loss(4, &Device::Cpu).unwrap();
        let d = loss.model().forward_pixels(&pixels(3, 16, 24)).unwrap();
        assert_eq!(d.dims(), &[1, 6, 8]);
        let sq = pixels(4, 8, 8);
        let a = loss.model().forward_pixels(&sq).unwrap();
        let dev = Device::Cpu;
        let mean = Tensor::from_slice(&crate::preprocess::IMAGE_MEAN, (1, 1, 1, 3), &dev).unwrap();
        let std = Tensor::from_slice(&crate::preprocess::IMAGE_STD, (1, 1, 1, 3), &dev).unwrap();
        let norm = sq
            .broadcast_sub(&mean)
            .unwrap()
            .broadcast_div(&std)
            .unwrap();
        let b = loss.model().forward_batch(&norm).unwrap();
        let diff = (a - b)
            .unwrap()
            .abs()
            .unwrap()
            .max_all()
            .unwrap()
            .to_scalar::<f32>()
            .unwrap();
        assert!(diff < 1e-6, "{diff}");
    }

    #[test]
    fn footprint_grows_with_model_size() {
        let f = |s| depth_anchor_footprint(s, 1024, 1024);
        let (s, b, l) = (
            f(DepthModelSize::Small),
            f(DepthModelSize::Base),
            f(DepthModelSize::Large),
        );
        let total = |x: AuxModelFootprint| x.param_bytes + x.working_set_bytes;
        assert!(total(s) < total(b) && total(b) < total(l));
        assert_eq!(
            s.param_bytes,
            DepthAnythingConfig::small().param_count() * 4
        );
        assert_eq!(s.reference_bytes_per_image, 518 * 518 * 4);
        assert_eq!(
            depth_anchor_footprint(DepthModelSize::Small, 768, 1024).reference_bytes_per_image,
            392 * 518 * 4
        );
    }

    /// Base/Large carry the published hyperparameters (`config.json` of the `-hf` checkpoints).
    #[test]
    fn configs_carry_the_published_hyperparameters() {
        let l = DepthAnythingConfig::large();
        assert_eq!(
            (l.hidden_size, l.num_hidden_layers, l.num_attention_heads),
            (1024, 24, 16)
        );
        let b = DepthAnythingConfig::base();
        assert_eq!((b.hidden_size, b.fusion_hidden_size), (768, 128));
    }
}
