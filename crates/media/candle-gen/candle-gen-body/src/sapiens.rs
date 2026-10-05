//! **Sapiens surface-normal estimator** — the candle twin of `mlx_gen_body::sapiens` (epic 2123,
//! sc-24832): ImageNet normalization → padded patch conv → `+` the resampled position grid →
//! pre-norm ViT layers (fused `qkv`) → final LayerNorm → 3 × [ConvTranspose(4, 2, 1) → InstanceNorm →
//! SiLU → 1 × 1 conv → InstanceNorm → SiLU] → `conv_seg`; the training entry letterboxes, resamples
//! to a square map and L2-normalizes. Every op has a candle backward.

use candle_gen::candle_core::{Device, Tensor, D};
use candle_gen::gen_core::train::body::{Letterbox, SapiensConfig, IMAGENET_MEAN, IMAGENET_STD};
use candle_gen::weights::Weights;
use candle_gen::Result;

use super::{conv2d, layer_norm, linear, normalize, resample_nhwc, sdpa, w32, AxisMatrix};

struct Layer {
    ln1: (Tensor, Tensor),
    qkv: (Tensor, Tensor),
    proj: (Tensor, Tensor),
    ln2: (Tensor, Tensor),
    fc1: (Tensor, Tensor),
    fc2: (Tensor, Tensor),
}

struct Stage {
    deconv: Tensor,
    conv: (Tensor, Tensor),
}

/// The loaded Sapiens normal estimator.
pub struct SapiensNormal {
    cfg: SapiensConfig,
    proj_w: Tensor,
    proj_b: Tensor,
    /// `[1, ph, pw, C]` stored position grid (NHWC).
    pos: Tensor,
    layers: Vec<Layer>,
    ln: (Tensor, Tensor),
    stages: Vec<Stage>,
    seg: (Tensor, Tensor),
}

impl SapiensNormal {
    /// Load from a directory holding the checkpoint (`model.safetensors`, Sapiens key layout).
    pub fn from_dir(
        dir: impl AsRef<std::path::Path>,
        cfg: SapiensConfig,
        device: &Device,
    ) -> Result<Self> {
        Self::from_weights(&super::load_dir(dir.as_ref(), device)?, "", cfg)
    }

    /// Load from weights whose Sapiens keys sit under `prefix`.
    pub fn from_weights(w: &Weights, prefix: &str, cfg: SapiensConfig) -> Result<Self> {
        let g = |k: &str| w32(w, prefix, k);
        let pair = |k: &str| -> Result<(Tensor, Tensor)> {
            Ok((g(&format!("{k}.weight"))?, g(&format!("{k}.bias"))?))
        };
        let layers = (0..cfg.num_layers)
            .map(|i| {
                let p = format!("backbone.layers.{i}");
                Ok(Layer {
                    ln1: pair(&format!("{p}.ln1"))?,
                    qkv: pair(&format!("{p}.attn.qkv"))?,
                    proj: pair(&format!("{p}.attn.proj"))?,
                    ln2: pair(&format!("{p}.ln2"))?,
                    fc1: pair(&format!("{p}.ffn.layers.0.0"))?,
                    fc2: pair(&format!("{p}.ffn.layers.1"))?,
                })
            })
            .collect::<Result<Vec<_>>>()?;
        let stages = (0..cfg.decoder_stages)
            .map(|i| {
                Ok(Stage {
                    deconv: g(&format!("decode_head.deconv_layers.{}.weight", 3 * i))?,
                    conv: pair(&format!("decode_head.conv_layers.{}", 3 * i))?,
                })
            })
            .collect::<Result<Vec<_>>>()?;
        let (ph, pw) = cfg.pos_grid;
        Ok(Self {
            proj_w: g("backbone.patch_embed.projection.weight")?,
            proj_b: g("backbone.patch_embed.projection.bias")?,
            pos: g("backbone.pos_embed")?.reshape((1, ph, pw, cfg.embed_dim))?,
            layers,
            ln: pair("backbone.ln1")?,
            stages,
            seg: pair("decode_head.conv_seg")?,
            cfg,
        })
    }

    /// The configuration.
    pub fn config(&self) -> &SapiensConfig {
        &self.cfg
    }

    /// InstanceNorm (no affine) over H, W of an NCHW map.
    fn instance_norm(&self, x: &Tensor) -> Result<Tensor> {
        let mean = x.mean_keepdim(3)?.mean_keepdim(2)?;
        let d = x.broadcast_sub(&mean)?;
        let var = d.sqr()?.mean_keepdim(3)?.mean_keepdim(2)?;
        Ok(d.broadcast_div(&(var + self.cfg.instance_norm_eps as f64)?.sqrt()?)?)
    }

    /// `[0, 1]` NHWC pixels `[B, H, W, 3]` → raw normals NHWC `[B, 8·gh, 8·gw, 3]`.
    pub fn forward(&self, pixels: &Tensor) -> Result<Tensor> {
        let c = &self.cfg;
        let x = normalize(pixels, IMAGENET_MEAN, IMAGENET_STD)?
            .permute([0, 3, 1, 2])?
            .contiguous()?;
        let y = conv2d(&x, &self.proj_w, Some(&self.proj_b), c.patch_size, 2)?; // [B, C, gh, gw]
        let (b, e, gh, gw) = y.dims4()?;
        let (ph, pw) = c.pos_grid;
        let dev = pixels.device();
        let pos = if (gh, gw) == (ph, pw) {
            self.pos.clone()
        } else {
            resample_nhwc(
                &self.pos,
                &AxisMatrix::resize(ph, gh, false, dev)?,
                &AxisMatrix::resize(pw, gw, false, dev)?,
            )?
        };
        let n = gh * gw;
        let mut t = y
            .permute([0, 2, 3, 1])?
            .broadcast_add(&pos)?
            .reshape((b, n, e))?;
        let heads = c.num_heads;
        let hd = e / heads;
        let eps = c.layer_norm_eps as f64;
        for l in &self.layers {
            let h = layer_norm(&t, &l.ln1.0, &l.ln1.1, eps)?;
            let qkv = linear(&h, &l.qkv)?
                .reshape((b, n, 3, heads, hd))?
                .permute([2, 0, 3, 1, 4])?
                .contiguous()?;
            let (q, k, v) = (qkv.get(0)?, qkv.get(1)?, qkv.get(2)?);
            let a = sdpa(&q, &k, &v, (hd as f64).powf(-0.5))?;
            let a = a.transpose(1, 2)?.contiguous()?.reshape((b, n, e))?;
            t = (t + linear(&a, &l.proj)?)?;
            let h = layer_norm(&t, &l.ln2.0, &l.ln2.1, eps)?;
            let h = linear(&h, &l.fc1)?.gelu_erf()?;
            t = (t + linear(&h, &l.fc2)?)?;
        }
        let t = layer_norm(&t, &self.ln.0, &self.ln.1, eps)?;
        let mut m = t.transpose(1, 2)?.contiguous()?.reshape((b, e, gh, gw))?;
        for s in &self.stages {
            let y = m.conv_transpose2d(&s.deconv, 1, 0, 2, 1)?;
            let y = self.instance_norm(&y)?.silu()?;
            let y = conv2d(&y, &s.conv.0, Some(&s.conv.1), 1, 0)?;
            m = self.instance_norm(&y)?.silu()?;
        }
        let out = conv2d(&m, &self.seg.0, Some(&self.seg.1), 1, 0)?;
        Ok(out.permute([0, 2, 3, 1])?.contiguous()?)
    }

    fn letterbox_axes(
        &self,
        h: usize,
        w: usize,
        dev: &Device,
    ) -> Result<(Letterbox, AxisMatrix, AxisMatrix)> {
        let lb = self.cfg.letterbox(h, w);
        let ay = AxisMatrix::from_weights(
            lb.target_h,
            h,
            Letterbox::axis_weights(lb.target_h, lb.new_h, lb.pad_top, h),
            dev,
        )?;
        let ax = AxisMatrix::from_weights(
            lb.target_w,
            w,
            Letterbox::axis_weights(lb.target_w, lb.new_w, lb.pad_left, w),
            dev,
        )?;
        Ok((lb, ay, ax))
    }

    /// The differentiable training entry: NHWC `[1, H, W, 3]` → letterbox →
    /// [`forward`](Self::forward) → bilinear resample to `normal_size²` → L2-normalized (`+ 1e-5`)
    /// normals NHWC `[1, S, S, 3]`.
    pub fn forward_pixels(&self, pixels: &Tensor) -> Result<Tensor> {
        let (_, h, w, _) = pixels.dims4()?;
        let dev = pixels.device();
        let (_, ay, ax) = self.letterbox_axes(h, w, dev)?;
        let raw = self.forward(&resample_nhwc(pixels, &ay, &ax)?)?;
        let (_, rh, rw, _) = raw.dims4()?;
        let s = self.cfg.normal_size;
        let raw = resample_nhwc(
            &raw,
            &AxisMatrix::resize(rh, s, false, dev)?,
            &AxisMatrix::resize(rw, s, false, dev)?,
        )?;
        let norm = raw.sqr()?.sum_keepdim(D::Minus1)?.sqrt()?;
        Ok(raw.broadcast_div(&(norm + 1e-5)?)?)
    }

    /// A subject mask `[H, W]` carried onto the normal grid like the pixels → `[S, S]`.
    pub fn mask_to_normal_grid(&self, mask: &Tensor) -> Result<Tensor> {
        let (h, w) = mask.dims2()?;
        let dev = mask.device();
        let (lb, ay, ax) = self.letterbox_axes(h, w, dev)?;
        let m = resample_nhwc(&mask.reshape((1, h, w, 1))?, &ay, &ax)?;
        let s = self.cfg.normal_size;
        let m = resample_nhwc(
            &m,
            &AxisMatrix::resize(lb.target_h, s, false, dev)?,
            &AxisMatrix::resize(lb.target_w, s, false, dev)?,
        )?;
        Ok(m.reshape((s, s))?)
    }
}

/// Upstream's normal comparison for one image (see the MLX twin): `(1 − cos) + L1`, spatially
/// averaged (mask-weighted when given, the divisor clamped to ≥ 1).
pub fn normal_comparison(
    reference: &Tensor,
    live: &Tensor,
    mask: Option<&Tensor>,
) -> Result<Tensor> {
    let cos = (reference * live)?.sum(D::Minus1)?; // [1, S, S]
    let l1 = (reference - live)?.abs()?.mean(D::Minus1)?;
    let (cos_m, l1_m) = match mask {
        Some(m) => {
            let m = m.unsqueeze(0)?;
            let denom = m.sum_all()?.clamp(1f32, f32::MAX)?;
            (
                (cos * &m)?.sum_all()?.div(&denom)?,
                (l1 * &m)?.sum_all()?.div(&denom)?,
            )
        }
        None => (cos.mean_all()?, l1.mean_all()?),
    };
    Ok(((1.0 - cos_m)? + l1_m)?)
}
