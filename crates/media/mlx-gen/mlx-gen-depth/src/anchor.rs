//! **Depth anchoring** (epic 2123, sc-2125) — Depth-Anything-V2 as a frozen, differentiable
//! perceptor plugged into the shared perceptual-loss path
//! ([`mlx_gen::train::perceptual`]): [`DepthAnchorLoss`] implements
//! [`PerceptualLoss`] so every MLX trainer that declares
//! `TrainingTechniques::depth_anchoring` reuses the same decode → DA2 → cached-reference → SSI +
//! multi-scale-gradient comparison.

use std::path::Path;

use mlx_rs::ops::multiply;
use mlx_rs::{random, Array};

use mlx_gen::gen_core::train::DepthModelSize;
use mlx_gen::train::perceptual::{depth_consistency_loss, PerceptualLoss};
use mlx_gen::weights::Weights;
use mlx_gen::Result;

use crate::{DepthAnythingConfig, DepthAnythingV2};

/// Frozen Depth-Anything-V2 + the MiDaS depth-consistency comparison.
pub struct DepthAnchorLoss {
    model: DepthAnythingV2,
    param_bytes: u64,
}

impl DepthAnchorLoss {
    /// Wrap a loaded estimator.
    pub fn new(model: DepthAnythingV2) -> Self {
        let param_bytes = model.config().param_count() * 4;
        Self { model, param_bytes }
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

    /// Decoded x0 pixels `[B, H, W, 3]` in `[0, 1]` → relative depth `[B, S, S]`.
    fn features(&self, pixels: &Array) -> Result<Array> {
        self.model.forward_pixels(pixels)
    }

    fn compare(&self, live: &Array, reference: &Array) -> Result<Array> {
        depth_consistency_loss(live, reference)
    }

    fn param_bytes(&self) -> u64 {
        self.param_bytes
    }

    fn training_working_set_bytes(&self) -> u64 {
        self.model.config().training_working_set_bytes()
    }
}

/// The extra training memory depth anchoring adds for a DA2 checkpoint of `size`, before any model
/// is loaded (resident f32 weights + one differentiable forward/backward) — what a trainer's
/// pre-flight memory estimate adds when the technique is on (epic 2123 E7).
pub fn depth_anchor_footprint_bytes(size: DepthModelSize) -> u64 {
    let cfg = DepthAnythingConfig::for_size(size);
    cfg.param_count() * 4 + cfg.training_working_set_bytes()
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

    /// The DA2 forward is differentiable end to end in its pixel input (no host round trip,
    /// argmax or stop-gradient), so the depth loss can train the adapter through it.
    #[test]
    fn depth_anchor_loss_is_differentiable_in_the_pixels() {
        let cfg = tiny_config();
        let loss = DepthAnchorLoss::new(
            DepthAnythingV2::from_weights(&synthetic_weights(&cfg, 4).unwrap(), cfg).unwrap(),
        );
        let px = random::uniform::<_, f32>(
            0.0f32,
            1.0f32,
            &[1, 16, 16, 3],
            Some(&random::key(1).unwrap()),
        )
        .unwrap();
        let reference = loss
            .features(
                &random::uniform::<_, f32>(
                    0.0f32,
                    1.0f32,
                    &[1, 16, 16, 3],
                    Some(&random::key(2).unwrap()),
                )
                .unwrap(),
            )
            .unwrap();
        assert_eq!(reference.shape(), &[1, 8, 8]);
        let f = |p: &Array| -> mlx_rs::error::Result<Array> {
            let d = loss
                .features(p)
                .map_err(|e| mlx_rs::error::Exception::custom(e.to_string()))?;
            loss.compare(&d, &reference)
                .map_err(|e| mlx_rs::error::Exception::custom(e.to_string()))
        };
        let g = grad(f)(&px).unwrap();
        eval([&g]).unwrap();
        let mag = g.abs().unwrap().sum(None).unwrap().item::<f32>();
        assert!(
            mag > 0.0 && mag.is_finite(),
            "pixel gradient magnitude {mag}"
        );
    }

    #[test]
    fn footprint_grows_with_model_size() {
        let s = depth_anchor_footprint_bytes(DepthModelSize::Small);
        let b = depth_anchor_footprint_bytes(DepthModelSize::Base);
        let l = depth_anchor_footprint_bytes(DepthModelSize::Large);
        assert!(s < b && b < l, "{s} {b} {l}");
        assert!(s > DepthAnythingConfig::small().param_count() * 4);
    }
}
