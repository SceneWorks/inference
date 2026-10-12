//! **Depth anchoring** (epic 2123, sc-2125) — Depth-Anything-V2 as a frozen, differentiable
//! perceptor plugged into the shared perceptual-loss path
//! ([`mlx_gen::train::perceptual`]): [`DepthAnchorLoss`] implements
//! [`PerceptualLoss`] so every MLX trainer that declares
//! `TrainingTechniques::depth_anchoring` reuses the same decode → DA2 → cached-reference → SSI +
//! multi-scale-gradient comparison.

use std::any::Any;
use std::path::Path;

use mlx_rs::ops::multiply;
use mlx_rs::{random, Array};

use mlx_gen::gen_core::train::DepthModelSize;
use mlx_gen::train::perceptual::{
    depth_consistency_loss, reference_as, AuxModelFootprint, LossReference, PerceptualLoss,
};
use mlx_gen::weights::Weights;
use mlx_gen::Result;

use crate::{DepthAnythingConfig, DepthAnythingV2};

/// Frozen Depth-Anything-V2 + the MiDaS depth-consistency comparison.
pub struct DepthAnchorLoss {
    model: DepthAnythingV2,
}

/// The per-image depth-anchoring reference: the DA2 depth of the image's clean round trip.
pub struct DepthReference {
    /// `[1, h, w]` relative depth at the aspect-preserving model size.
    pub depth: Array,
}

impl DepthAnchorLoss {
    /// Wrap a loaded estimator.
    pub fn new(model: DepthAnythingV2) -> Self {
        Self { model }
    }

    /// Load the `size` checkpoint from `dir` (a `Depth-Anything-V2-{Small,Base,Large}-hf` snapshot).
    pub fn from_dir(dir: impl AsRef<Path>, size: DepthModelSize) -> Result<Self> {
        Ok(Self::new(DepthAnythingV2::from_dir_with(
            dir,
            DepthAnythingConfig::for_size(size),
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
    fn reference(&self, clean: &Array) -> Result<Option<LossReference>> {
        let depth = mlx_rs::stop_gradient(self.model.forward_pixels(clean)?)?;
        depth.eval()?;
        Ok(Some(Box::new(DepthReference { depth })))
    }

    /// Live decoded pixels `[1, H, W, 3]` → DA2 depth → SSI-L1 + multi-scale gradient vs the
    /// reference depth (same shape: same-sized decode, same aspect-preserving resize).
    fn loss(&self, live: &Array, reference: &dyn Any) -> Result<Array> {
        let r = reference_as::<DepthReference>(self.name(), reference)?;
        depth_consistency_loss(&self.model.forward_pixels(live)?, &r.depth)
    }
}

/// The pre-load memory figures of depth anchoring with a DA2 checkpoint of `size` on
/// `image_h × image_w` training images (epic 2123 E7): resident f32 weights, one differentiable
/// forward/backward at the native size (an upper bound for any aspect), and one cached depth map
/// per image at the aspect-preserving model size.
pub fn depth_anchor_footprint(
    size: DepthModelSize,
    image_h: u32,
    image_w: u32,
) -> AuxModelFootprint {
    let cfg = DepthAnythingConfig::for_size(size);
    let (h, w) = cfg.input_hw(image_h as i32, image_w as i32);
    AuxModelFootprint {
        param_bytes: cfg.param_count() * 4,
        working_set_bytes: cfg.training_working_set_bytes(),
        reference_bytes_per_image: h as u64 * w as u64 * 4,
    }
}

/// A complete random-init checkpoint for `cfg` (every key [`DepthAnythingV2::from_weights`]
/// requires, torch OIHW/IOHW conv layouts) — for tests of depth anchoring and the trainers that
/// use it, which must never download real weights. Deterministic in `seed`; scaled so activations
/// stay O(1) through the tiny graph.
pub fn synthetic_weights(cfg: &DepthAnythingConfig, seed: u64) -> Result<Weights> {
    let mut w = Weights::empty();
    let mut n = 0u64;
    let mut rnd = |shape: &[i32]| -> Result<Array> {
        n += 1;
        let fan_in: i32 = if shape.len() > 1 {
            shape[1..].iter().product()
        } else {
            shape[0]
        };
        let std = (1.0 / fan_in.max(1) as f32).sqrt();
        let key = random::key(seed.wrapping_mul(7_919).wrapping_add(n))?;
        Ok(multiply(
            &random::normal::<f32>(shape, None, None, Some(&key))?,
            Array::from_f32(std),
        )?)
    };
    let h = cfg.hidden_size;
    let grid = cfg.grid();
    let inter = cfg.intermediate_size();
    let ones = |shape: &[i32]| -> Array {
        let k: i32 = shape.iter().product();
        Array::from_slice(&vec![1.0f32; k as usize], shape)
    };

    w.insert(
        "backbone.embeddings.patch_embeddings.projection.weight",
        rnd(&[h, cfg.num_channels, cfg.patch_size, cfg.patch_size])?,
    );
    w.insert(
        "backbone.embeddings.patch_embeddings.projection.bias",
        rnd(&[h])?,
    );
    w.insert("backbone.embeddings.cls_token", rnd(&[1, 1, h])?);
    w.insert(
        "backbone.embeddings.position_embeddings",
        rnd(&[1, grid * grid + 1, h])?,
    );
    for i in 0..cfg.num_hidden_layers {
        let p = format!("backbone.encoder.layer.{i}");
        for leaf in ["norm1", "norm2"] {
            w.insert(format!("{p}.{leaf}.weight"), ones(&[h]));
            w.insert(format!("{p}.{leaf}.bias"), rnd(&[h])?);
        }
        for leaf in ["query", "key", "value"] {
            w.insert(
                format!("{p}.attention.attention.{leaf}.weight"),
                rnd(&[h, h])?,
            );
            w.insert(format!("{p}.attention.attention.{leaf}.bias"), rnd(&[h])?);
        }
        w.insert(format!("{p}.attention.output.dense.weight"), rnd(&[h, h])?);
        w.insert(format!("{p}.attention.output.dense.bias"), rnd(&[h])?);
        w.insert(format!("{p}.layer_scale1.lambda1"), ones(&[h]));
        w.insert(format!("{p}.layer_scale2.lambda1"), ones(&[h]));
        w.insert(format!("{p}.mlp.fc1.weight"), rnd(&[inter, h])?);
        w.insert(format!("{p}.mlp.fc1.bias"), rnd(&[inter])?);
        w.insert(format!("{p}.mlp.fc2.weight"), rnd(&[h, inter])?);
        w.insert(format!("{p}.mlp.fc2.bias"), rnd(&[h])?);
    }
    w.insert("backbone.layernorm.weight", ones(&[h]));
    w.insert("backbone.layernorm.bias", rnd(&[h])?);

    let fh = cfg.fusion_hidden_size;
    for i in 0..4 {
        let nh = cfg.neck_hidden_sizes[i];
        let p = format!("neck.reassemble_stage.layers.{i}");
        w.insert(format!("{p}.projection.weight"), rnd(&[nh, h, 1, 1])?);
        w.insert(format!("{p}.projection.bias"), rnd(&[nh])?);
        let factor = cfg.reassemble_factors[i];
        if factor > 1.0 {
            let k = factor as i32;
            w.insert(format!("{p}.resize.weight"), rnd(&[nh, nh, k, k])?);
            w.insert(format!("{p}.resize.bias"), rnd(&[nh])?);
        } else if factor < 1.0 {
            w.insert(format!("{p}.resize.weight"), rnd(&[nh, nh, 3, 3])?);
            w.insert(format!("{p}.resize.bias"), rnd(&[nh])?);
        }
        w.insert(format!("neck.convs.{i}.weight"), rnd(&[fh, nh, 3, 3])?);
        let fp = format!("neck.fusion_stage.layers.{i}");
        for res in ["residual_layer1", "residual_layer2"] {
            for c in ["convolution1", "convolution2"] {
                w.insert(format!("{fp}.{res}.{c}.weight"), rnd(&[fh, fh, 3, 3])?);
                w.insert(format!("{fp}.{res}.{c}.bias"), rnd(&[fh])?);
            }
        }
        w.insert(format!("{fp}.projection.weight"), rnd(&[fh, fh, 1, 1])?);
        w.insert(format!("{fp}.projection.bias"), rnd(&[fh])?);
    }
    let hh = cfg.head_hidden_size;
    let half = fh / 2;
    w.insert("head.conv1.weight", rnd(&[half, fh, 3, 3])?);
    w.insert("head.conv1.bias", rnd(&[half])?);
    w.insert("head.conv2.weight", rnd(&[hh, half, 3, 3])?);
    w.insert("head.conv2.bias", rnd(&[hh])?);
    w.insert("head.conv3.weight", rnd(&[1, hh, 1, 1])?);
    // A positive bias keeps the final ReLU from zeroing the whole random-init map.
    w.insert("head.conv3.bias", Array::from_slice(&[1.0f32], &[1]));
    Ok(w)
}

/// A tiny DA2-shaped config (8-dim embed, grid 4 / image 8) that keeps the real factor ladder —
/// for synthetic tests.
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

#[cfg(test)]
mod tests {
    use super::*;
    use mlx_rs::transforms::{eval, grad};

    /// The analytic parameter count matches the published checkpoints' `model.safetensors` sizes
    /// (f32; the few-KB safetensors header is the only slack). Mutation: drop the neck `convs` term
    /// from `param_count` ⇒ red.
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

    #[test]
    fn synthetic_weights_load_completely() {
        let cfg = tiny_config();
        let w = synthetic_weights(&cfg, 3).unwrap();
        let _ = DepthAnythingV2::from_weights(&w, cfg.clone()).unwrap();
        assert!(w.unused_keys().is_empty(), "unused: {:?}", w.unused_keys());
    }

    fn tiny_loss() -> DepthAnchorLoss {
        let cfg = tiny_config();
        DepthAnchorLoss::new(
            DepthAnythingV2::from_weights(&synthetic_weights(&cfg, 4).unwrap(), cfg).unwrap(),
        )
    }

    fn pixels(seed: u64, h: i32, w: i32) -> Array {
        random::uniform::<_, f32>(
            0.0f32,
            1.0f32,
            &[1, h, w, 3],
            Some(&random::key(seed).unwrap()),
        )
        .unwrap()
    }

    /// The DA2 forward is differentiable end to end in its pixel input (no host round trip,
    /// argmax or stop-gradient), so the depth loss can train the adapter through it — for square
    /// and non-square inputs alike.
    #[test]
    fn depth_anchor_loss_is_differentiable_in_the_pixels() {
        let loss = tiny_loss();
        for (h, w, depth_hw) in [(16, 16, [8, 8]), (16, 24, [6, 8])] {
            let reference = loss
                .reference(&pixels(2, h, w))
                .unwrap()
                .expect("depth usable");
            let r = reference.downcast_ref::<DepthReference>().unwrap();
            assert_eq!(r.depth.shape(), &[1, depth_hw[0], depth_hw[1]], "{h}x{w}");
            let px = pixels(1, h, w);
            let f = |p: &Array| -> mlx_rs::error::Result<Array> {
                loss.loss(p, r)
                    .map_err(|e| mlx_rs::error::Exception::custom(e.to_string()))
            };
            let g = grad(f)(&px).unwrap();
            eval([&g]).unwrap();
            let mag = g.abs().unwrap().sum(None).unwrap().item::<f32>();
            assert!(
                mag > 0.0 && mag.is_finite(),
                "{h}x{w}: pixel gradient {mag}"
            );
        }
    }

    /// Review minor: a non-square input is resized aspect-preservingly (long side `image_size`,
    /// short side a multiple of the patch size), not squashed to a square. Mutation: resize to
    /// `(image_size, image_size)` in `forward_pixels` ⇒ the 16×24 depth is 8×8 ⇒ red.
    #[test]
    fn non_square_inputs_keep_their_aspect() {
        let cfg = tiny_config(); // image_size 8, patch 2
        assert_eq!(cfg.input_hw(16, 16), (8, 8));
        assert_eq!(cfg.input_hw(16, 24), (6, 8));
        assert_eq!(cfg.input_hw(24, 16), (8, 6));
        assert_eq!(cfg.input_hw(2, 100), (2, 8), "at least one patch");
        let real = DepthAnythingConfig::small();
        assert_eq!(real.input_hw(1024, 1024), (518, 518));
        assert_eq!(real.input_hw(768, 1024), (392, 518));
        let d = tiny_loss()
            .model()
            .forward_pixels(&pixels(3, 16, 24))
            .unwrap();
        assert_eq!(d.shape(), &[1, 6, 8]);
        // A square input at the native size never resamples the position embedding: the result
        // is the plain forward of the normalized pixels.
        let sq = pixels(4, 8, 8);
        let a = tiny_loss().model().forward_pixels(&sq).unwrap();
        let mean = Array::from_slice(&crate::preprocess::IMAGE_MEAN, &[1, 1, 1, 3]);
        let std = Array::from_slice(&crate::preprocess::IMAGE_STD, &[1, 1, 1, 3]);
        let norm = mlx_rs::ops::divide(mlx_rs::ops::subtract(&sq, &mean).unwrap(), &std).unwrap();
        let b = tiny_loss().model().forward_batch(&norm).unwrap();
        let diff = a.subtract(&b).unwrap().abs().unwrap().max(None).unwrap();
        eval([&diff]).unwrap();
        assert!(diff.item::<f32>() < 1e-6);
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
}
