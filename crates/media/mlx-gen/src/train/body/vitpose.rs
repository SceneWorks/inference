//! **ViTPose+** (HF transformers `VitPoseForPoseEstimation`, `usyd-community/vitpose-plus-base`) —
//! the frozen, differentiable keypoint model of the body-proportion loss and the reference-time
//! person detector of every body loss (epic 2123, sc-24832).
//!
//! Graph (HF `modeling_vitpose_backbone.py` + `modeling_vitpose.py`): a padded patch conv
//! (kernel = stride = 16, padding 2) → `+ pos[1:] + pos[:1]` (no CLS token) → pre-norm ViT layers
//! whose MLP is ViTPose+'s mixture of experts (a shared `fc2` slice concatenated with the selected
//! expert's slice) → final LayerNorm → the classic decoder (2 × ConvTranspose(4, 2, 1) + BatchNorm
//! + ReLU → 1 × 1 conv) or the simple one (ReLU → ×4 bilinear → 3 × 3 conv) → COCO heatmaps.
//! Weight keys are the HF checkpoint's own (`backbone.*`, `head.*`).

use mlx_rs::fast::{layer_norm, scaled_dot_product_attention};
use mlx_rs::ops::indexing::IndexOp;
use mlx_rs::ops::{add, concatenate_axis, conv_transpose2d, maximum, multiply};
use mlx_rs::Array;

use gen_core::train::body::{VitPoseConfig, VitPoseWarp, IMAGENET_MEAN, IMAGENET_STD};

use super::{bn_fold, conv_ohwi, normalize, resample_nhwc, w32, AxisMatrix};
use crate::nn::{conv2d, gelu_exact, linear};
use crate::weights::Weights;
use crate::{Error, Result};

struct Layer {
    ln1: (Array, Array),
    q: (Array, Array),
    k: (Array, Array),
    v: (Array, Array),
    o: (Array, Array),
    ln2: (Array, Array),
    fc1: (Array, Array),
    fc2: (Array, Array),
    /// The selected expert's `[part, inter]` projection (MoE configs only).
    expert: Option<(Array, Array)>,
}

enum Head {
    Classic {
        d1: Array,
        bn1: (Array, Array),
        d2: Array,
        bn2: (Array, Array),
        conv: (Array, Array),
    },
    Simple {
        conv: (Array, Array),
    },
}

/// The loaded ViTPose(+) model.
pub struct VitPose {
    cfg: VitPoseConfig,
    proj_w: Array,
    proj_b: Array,
    /// `pos[:, 1:] + pos[:, :1]` — what HF adds to the patch tokens.
    pos: Array,
    layers: Vec<Layer>,
    ln: (Array, Array),
    head: Head,
}

impl VitPose {
    /// Load from a directory holding the HF checkpoint (`model.safetensors`).
    pub fn from_dir(dir: impl AsRef<std::path::Path>, cfg: VitPoseConfig) -> Result<Self> {
        Self::from_weights(&Weights::from_dir(dir)?, "", cfg)
    }

    /// Load from weights whose HF keys sit under `prefix` (empty = at the root). Any float dtype
    /// is promoted to f32.
    pub fn from_weights(w: &Weights, prefix: &str, cfg: VitPoseConfig) -> Result<Self> {
        let (gh, gw) = cfg.grid();
        let (ih, iw) = cfg.image_size;
        if gh * cfg.patch_size != ih || gw * cfg.patch_size != iw {
            return Err(Error::Msg(format!(
                "vitpose: the padded patch grid {gh}x{gw} must tile the {ih}x{iw} input exactly"
            )));
        }
        let g = |k: &str| w32(w, prefix, k);
        let pair = |k: &str| -> Result<(Array, Array)> {
            Ok((g(&format!("{k}.weight"))?, g(&format!("{k}.bias"))?))
        };
        let pos = g("backbone.embeddings.position_embeddings")?;
        let pos = add(pos.index((.., 1.., ..)), pos.index((.., ..1, ..)))?;
        let mut layers = Vec::with_capacity(cfg.num_layers);
        for i in 0..cfg.num_layers {
            let p = format!("backbone.encoder.layer.{i}");
            layers.push(Layer {
                ln1: pair(&format!("{p}.layernorm_before"))?,
                q: pair(&format!("{p}.attention.attention.query"))?,
                k: pair(&format!("{p}.attention.attention.key"))?,
                v: pair(&format!("{p}.attention.attention.value"))?,
                o: pair(&format!("{p}.attention.output.dense"))?,
                ln2: pair(&format!("{p}.layernorm_after"))?,
                fc1: pair(&format!("{p}.mlp.fc1"))?,
                fc2: pair(&format!("{p}.mlp.fc2"))?,
                expert: if cfg.num_experts > 1 {
                    Some(pair(&format!(
                        "{p}.mlp.experts.{}",
                        cfg.expert_index
                    ))?)
                } else {
                    None
                },
            });
        }
        let head = if cfg.use_simple_decoder {
            let (cw, cb) = pair("head.conv")?;
            Head::Simple {
                conv: (conv_ohwi(&cw)?, cb),
            }
        } else {
            let bn = |k: &str| bn_fold(w, prefix, k, 1e-5);
            let (cw, cb) = pair("head.conv")?;
            Head::Classic {
                d1: g("head.deconv1.weight")?.transpose_axes(&[1, 2, 3, 0])?,
                bn1: bn("head.batchnorm1")?,
                d2: g("head.deconv2.weight")?.transpose_axes(&[1, 2, 3, 0])?,
                bn2: bn("head.batchnorm2")?,
                conv: (conv_ohwi(&cw)?, cb),
            }
        };
        Ok(Self {
            proj_w: conv_ohwi(&g("backbone.embeddings.patch_embeddings.projection.weight")?)?,
            proj_b: g("backbone.embeddings.patch_embeddings.projection.bias")?,
            pos,
            layers,
            ln: pair("backbone.layernorm")?,
            head,
            cfg,
        })
    }

    /// The configuration.
    pub fn config(&self) -> &VitPoseConfig {
        &self.cfg
    }

    fn layer(&self, l: &Layer, x: &Array) -> Result<Array> {
        let c = &self.cfg;
        let sh = x.shape();
        let (b, n) = (sh[0], sh[1]);
        let heads = c.num_heads as i32;
        let hd = (c.hidden_size / c.num_heads) as i32;
        let eps = c.layer_norm_eps;
        let h = layer_norm(x, Some(&l.ln1.0), Some(&l.ln1.1), eps)?;
        let to_heads = |t: Array| -> Result<Array> {
            Ok(t.reshape(&[b, n, heads, hd])?.transpose_axes(&[0, 2, 1, 3])?)
        };
        let q = to_heads(linear(&h, &l.q.0, &l.q.1)?)?;
        let k = to_heads(linear(&h, &l.k.0, &l.k.1)?)?;
        let v = to_heads(linear(&h, &l.v.0, &l.v.1)?)?;
        let a = scaled_dot_product_attention(&q, &k, &v, (hd as f32).powf(-0.5), None, None)?;
        let a = a.transpose_axes(&[0, 2, 1, 3])?.reshape(&[b, n, heads * hd])?;
        let x = add(x, &linear(&a, &l.o.0, &l.o.1)?)?;
        let h = layer_norm(&x, Some(&l.ln2.0), Some(&l.ln2.1), eps)?;
        let h = gelu_exact(&linear(&h, &l.fc1.0, &l.fc1.1)?)?;
        let shared = linear(&h, &l.fc2.0, &l.fc2.1)?;
        let mlp = match &l.expert {
            Some((ew, eb)) => concatenate_axis(&[&shared, &linear(&h, ew, eb)?], -1)?,
            None => shared,
        };
        Ok(add(&x, &mlp)?)
    }

    /// ImageNet-normalized NHWC `[B, H, W, 3]` at the model's input size → heatmaps NHWC
    /// `[B, h, w, K]`.
    pub fn forward(&self, x: &Array) -> Result<Array> {
        let c = &self.cfg;
        let b = x.shape()[0];
        let (gh, gw) = c.grid();
        let y = conv2d(x, &self.proj_w, Some(&self.proj_b), c.patch_size as i32, 2)?;
        let mut t = add(
            &y.reshape(&[b, (gh * gw) as i32, c.hidden_size as i32])?,
            &self.pos,
        )?;
        for l in &self.layers {
            t = self.layer(l, &t)?;
        }
        let t = layer_norm(&t, Some(&self.ln.0), Some(&self.ln.1), c.layer_norm_eps)?;
        let fmap = t.reshape(&[b, gh as i32, gw as i32, c.hidden_size as i32])?;
        let zero = Array::from_f32(0.0);
        match &self.head {
            Head::Classic {
                d1,
                bn1,
                d2,
                bn2,
                conv,
            } => {
                let deconv = |x: &Array, w: &Array, bn: &(Array, Array)| -> Result<Array> {
                    let y = conv_transpose2d(x, w, (2, 2), (1, 1), None, None, None)?;
                    Ok(maximum(&add(&multiply(&y, &bn.0)?, &bn.1)?, &zero)?)
                };
                let y = deconv(&fmap, d1, bn1)?;
                let y = deconv(&y, d2, bn2)?;
                conv2d(&y, &conv.0, Some(&conv.1), 1, 0)
            }
            Head::Simple { conv } => {
                let y = maximum(&fmap, &zero)?;
                let up = |n: usize| AxisMatrix::resize(n, 4 * n, false);
                let y = resample_nhwc(&y, &up(gh), &up(gw))?;
                conv2d(&y, &conv.0, Some(&conv.1), 1, 1)
            }
        }
    }

    /// The **differentiable** pixel entry point: NHWC pixels `[1, H, W, 3]` in `[0, 1]` → the
    /// full-frame affine warp ([`VitPoseWarp`], upstream's `affine_grid` + `grid_sample`) →
    /// ImageNet normalization → heatmaps `[1, h, w, K]`. Also returns the warp so keypoints can be
    /// mapped back to the frame.
    pub fn forward_pixels(&self, pixels: &Array) -> Result<(Array, VitPoseWarp)> {
        let sh = pixels.shape();
        let (ih, iw) = (sh[1] as usize, sh[2] as usize);
        let warp = VitPoseWarp::full_frame(ih, iw, self.cfg.image_size);
        let (oh, ow) = self.cfg.image_size;
        let ay = AxisMatrix::affine(ih, oh, warp.scale_y, warp.offset_y);
        let ax = AxisMatrix::affine(iw, ow, warp.scale_x, warp.offset_x);
        let x = resample_nhwc(pixels, &ay, &ax)?;
        let x = normalize(&x, IMAGENET_MEAN, IMAGENET_STD)?;
        Ok((self.forward(&x)?, warp))
    }
}

/// Upstream `_heatmaps_to_coords` (integral regression: clamp ≥ 0, normalize each heatmap to a
/// distribution, DSNT expectation over pixel-centre coordinates) and the gradient-stopped peak
/// confidence: heatmaps NHWC `[B, h, w, K]` → `(coords [B, K, 2] (x, y) in [-1, 1], confidence
/// [B, K])`.
pub fn heatmaps_to_keypoints(heatmaps: &Array) -> Result<(Array, Array)> {
    let hm = heatmaps.transpose_axes(&[0, 3, 1, 2])?; // [B, K, h, w]
    let sh = hm.shape();
    let (h, w) = (sh[2], sh[3]);
    let confidence = mlx_rs::stop_gradient(hm.max_axes(&[2, 3], false)?)?;
    let p = maximum(&hm, Array::from_f32(0.0))?;
    let total = maximum(&p.sum_axes(&[2, 3], true)?, Array::from_f32(1e-6))?;
    let p = mlx_rs::ops::divide(&p, &total)?;
    let lin = |n: i32| -> Array {
        let v: Vec<f32> = (0..n)
            .map(|i| (i as f32 * 2.0 + 1.0) / n as f32 - 1.0)
            .collect();
        Array::from_slice(&v, &[n])
    };
    let x = multiply(&p.sum_axes(&[2], false)?, lin(w))?.sum_axes(&[2], false)?;
    let y = multiply(&p.sum_axes(&[3], false)?, lin(h))?.sum_axes(&[2], false)?;
    let coords = mlx_rs::ops::stack_axis(&[&x, &y], -1)?;
    Ok((coords, confidence))
}
