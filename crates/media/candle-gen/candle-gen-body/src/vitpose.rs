//! **ViTPose+** (HF transformers `VitPoseForPoseEstimation`, `usyd-community/vitpose-plus-base`) —
//! the candle twin of `mlx_gen_body::vitpose` (epic 2123, sc-24832): the frozen, differentiable
//! keypoint model of the body-proportion loss and every body loss's reference-time person detector.
//!
//! Same graph as the MLX port (padded patch conv → `+ pos[1:] + pos[:1]` → pre-norm ViT layers with
//! the ViTPose+ mixture-of-experts MLP → final LayerNorm → classic or simple decoder → COCO
//! heatmaps). Every op has a candle backward: LayerNorm is composable, the strided patch conv uses
//! the backward-safe pad-and-crop form, ConvTranspose2D backpropagates natively.

use candle_gen::candle_core::{Device, Tensor, D};
use candle_gen::gen_core::train::body::{VitPoseConfig, VitPoseWarp, IMAGENET_MEAN, IMAGENET_STD};
use candle_gen::weights::Weights;
use candle_gen::{CandleError, Result};

use super::{
    bn_fold, conv2d, layer_norm, linear, normalize, resample_nhwc, sdpa, w32, AxisMatrix, BatchNorm,
};

struct Layer {
    ln1: (Tensor, Tensor),
    q: (Tensor, Tensor),
    k: (Tensor, Tensor),
    v: (Tensor, Tensor),
    o: (Tensor, Tensor),
    ln2: (Tensor, Tensor),
    fc1: (Tensor, Tensor),
    fc2: (Tensor, Tensor),
    expert: Option<(Tensor, Tensor)>,
}

enum Head {
    Classic {
        d1: Tensor,
        bn1: BatchNorm,
        d2: Tensor,
        bn2: BatchNorm,
        conv: (Tensor, Tensor),
    },
    Simple {
        conv: (Tensor, Tensor),
    },
}

/// The loaded ViTPose(+) model.
pub struct VitPose {
    cfg: VitPoseConfig,
    proj_w: Tensor,
    proj_b: Tensor,
    pos: Tensor,
    layers: Vec<Layer>,
    ln: (Tensor, Tensor),
    head: Head,
}

impl VitPose {
    /// Load from a directory holding the HF checkpoint (`model.safetensors`) onto `device`.
    pub fn from_dir(
        dir: impl AsRef<std::path::Path>,
        cfg: VitPoseConfig,
        device: &Device,
    ) -> Result<Self> {
        Self::from_weights(&super::load_dir(dir.as_ref(), device)?, "", cfg)
    }

    /// Load from weights whose HF keys sit under `prefix`.
    pub fn from_weights(w: &Weights, prefix: &str, cfg: VitPoseConfig) -> Result<Self> {
        let (gh, gw) = cfg.grid();
        let (ih, iw) = cfg.image_size;
        if gh * cfg.patch_size != ih || gw * cfg.patch_size != iw {
            return Err(CandleError::Msg(format!(
                "vitpose: the padded patch grid {gh}x{gw} must tile the {ih}x{iw} input exactly"
            )));
        }
        let g = |k: &str| w32(w, prefix, k);
        let pair = |k: &str| -> Result<(Tensor, Tensor)> {
            Ok((g(&format!("{k}.weight"))?, g(&format!("{k}.bias"))?))
        };
        let pos = g("backbone.embeddings.position_embeddings")?;
        let n = pos.dim(1)?;
        let pos = pos
            .narrow(1, 1, n - 1)?
            .broadcast_add(&pos.narrow(1, 0, 1)?)?;
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
                    Some(pair(&format!("{p}.mlp.experts.{}", cfg.expert_index))?)
                } else {
                    None
                },
            });
        }
        let head = if cfg.use_simple_decoder {
            Head::Simple {
                conv: pair("head.conv")?,
            }
        } else {
            Head::Classic {
                d1: g("head.deconv1.weight")?,
                bn1: bn_fold(w, prefix, "head.batchnorm1", 1e-5)?,
                d2: g("head.deconv2.weight")?,
                bn2: bn_fold(w, prefix, "head.batchnorm2", 1e-5)?,
                conv: pair("head.conv")?,
            }
        };
        Ok(Self {
            proj_w: g("backbone.embeddings.patch_embeddings.projection.weight")?,
            proj_b: g("backbone.embeddings.patch_embeddings.projection.bias")?,
            pos,
            layers,
            ln: pair("backbone.layernorm")?,
            head,
            cfg,
        })
    }

    /// Whether the weights live on `device`.
    pub(crate) fn device_matches(&self, device: &Device) -> bool {
        self.proj_w.device().same_device(device)
    }

    /// The configuration.
    pub fn config(&self) -> &VitPoseConfig {
        &self.cfg
    }

    fn layer(&self, l: &Layer, x: &Tensor) -> Result<Tensor> {
        let c = &self.cfg;
        let (b, n, _) = x.dims3()?;
        let heads = c.num_heads;
        let hd = c.hidden_size / heads;
        let eps = c.layer_norm_eps as f64;
        let h = layer_norm(x, &l.ln1.0, &l.ln1.1, eps)?;
        let to_heads = |t: Tensor| -> Result<Tensor> {
            Ok(t.reshape((b, n, heads, hd))?
                .transpose(1, 2)?
                .contiguous()?)
        };
        let q = to_heads(linear(&h, &l.q)?)?;
        let k = to_heads(linear(&h, &l.k)?)?;
        let v = to_heads(linear(&h, &l.v)?)?;
        let a = sdpa(&q, &k, &v, (hd as f64).powf(-0.5))?;
        let a = a
            .transpose(1, 2)?
            .contiguous()?
            .reshape((b, n, heads * hd))?;
        let x = (x + linear(&a, &l.o)?)?;
        let h = layer_norm(&x, &l.ln2.0, &l.ln2.1, eps)?;
        let h = linear(&h, &l.fc1)?.gelu_erf()?;
        let shared = linear(&h, &l.fc2)?;
        let mlp = match &l.expert {
            Some(e) => Tensor::cat(&[&shared, &linear(&h, e)?], D::Minus1)?,
            None => shared,
        };
        Ok((x + mlp)?)
    }

    /// ImageNet-normalized NHWC `[B, H, W, 3]` at the model input → heatmaps NHWC `[B, h, w, K]`.
    pub fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let c = &self.cfg;
        let b = x.dim(0)?;
        let (gh, gw) = c.grid();
        let xc = x.permute([0, 3, 1, 2])?.contiguous()?;
        let y = conv2d(&xc, &self.proj_w, Some(&self.proj_b), c.patch_size, 2)?; // [B, C, gh, gw]
        let t = y.flatten_from(2)?.transpose(1, 2)?.contiguous()?;
        let mut t = t.broadcast_add(&self.pos)?;
        for l in &self.layers {
            t = self.layer(l, &t)?;
        }
        let t = layer_norm(&t, &self.ln.0, &self.ln.1, c.layer_norm_eps as f64)?;
        let fmap = t
            .transpose(1, 2)?
            .contiguous()?
            .reshape((b, c.hidden_size, gh, gw))?; // NCHW
        let heat = match &self.head {
            Head::Classic {
                d1,
                bn1,
                d2,
                bn2,
                conv,
            } => {
                let y = bn1.apply(&fmap.conv_transpose2d(d1, 1, 0, 2, 1)?)?.relu()?;
                let y = bn2.apply(&y.conv_transpose2d(d2, 1, 0, 2, 1)?)?.relu()?;
                conv2d(&y, &conv.0, Some(&conv.1), 1, 0)?
            }
            Head::Simple { conv } => {
                let y = fmap.relu()?.permute([0, 2, 3, 1])?.contiguous()?;
                let up = |n: usize| AxisMatrix::resize(n, 4 * n, false, x.device());
                let y = resample_nhwc(&y, &up(gh)?, &up(gw)?)?;
                let y = y.permute([0, 3, 1, 2])?.contiguous()?;
                conv2d(&y, &conv.0, Some(&conv.1), 1, 1)?
            }
        };
        Ok(heat.permute([0, 2, 3, 1])?.contiguous()?)
    }

    /// The differentiable pixel entry point: NHWC `[1, H, W, 3]` in `[0, 1]` → full-frame affine
    /// warp → ImageNet normalization → heatmaps, plus the warp.
    pub fn forward_pixels(&self, pixels: &Tensor) -> Result<(Tensor, VitPoseWarp)> {
        let (_, ih, iw, _) = pixels.dims4()?;
        let warp = VitPoseWarp::full_frame(ih, iw, self.cfg.image_size);
        let (oh, ow) = self.cfg.image_size;
        let dev = pixels.device();
        let ay = AxisMatrix::affine(ih, oh, warp.scale_y, warp.offset_y, dev)?;
        let ax = AxisMatrix::affine(iw, ow, warp.scale_x, warp.offset_x, dev)?;
        let x = resample_nhwc(pixels, &ay, &ax)?;
        let x = normalize(&x, IMAGENET_MEAN, IMAGENET_STD)?;
        Ok((self.forward(&x)?, warp))
    }
}

/// Upstream `_heatmaps_to_coords` + the detached peak confidence: heatmaps NHWC `[B, h, w, K]` →
/// `(coords [B, K, 2] (x, y) in [-1, 1], confidence [B, K])`.
pub fn heatmaps_to_keypoints(heatmaps: &Tensor) -> Result<(Tensor, Tensor)> {
    let hm = heatmaps.permute([0, 3, 1, 2])?.contiguous()?; // [B, K, h, w]
    let (_, _, h, w) = hm.dims4()?;
    let confidence = hm.flatten_from(2)?.max(D::Minus1)?.detach();
    let p = hm.relu()?;
    let total = p.sum_keepdim(2)?.sum_keepdim(3)?.clamp(1e-6f32, f32::MAX)?;
    let p = p.broadcast_div(&total)?;
    let lin = |n: usize| -> Result<Tensor> {
        let v: Vec<f32> = (0..n)
            .map(|i| (i as f32 * 2.0 + 1.0) / n as f32 - 1.0)
            .collect();
        Ok(Tensor::from_vec(v, n, heatmaps.device())?)
    };
    let x = p.sum(2)?.broadcast_mul(&lin(w)?)?.sum(D::Minus1)?;
    let y = p.sum(3)?.broadcast_mul(&lin(h)?)?.sum(D::Minus1)?;
    Ok((Tensor::stack(&[&x, &y], D::Minus1)?, confidence))
}
