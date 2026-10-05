//! **Sapiens surface-normal estimator** (ai-toolkit-perceptual `SapiensNormal` /
//! `DifferentiableNormalEncoder`, `facebook/sapiens-normal-0.3b`) — epic 2123, sc-24832.
//!
//! Graph: ImageNet normalization → padded patch conv (kernel = stride = 16, padding 2) → `+` the
//! 64 × 48 position table (bilinearly resampled to a non-native grid) → pre-norm ViT layers (fused
//! `qkv`, exact-GELU FFN) → final LayerNorm → 3 × [ConvTranspose(4, 2, 1, no bias) → InstanceNorm →
//! SiLU → 1 × 1 conv → InstanceNorm → SiLU] → 1 × 1 `conv_seg` → raw normals at 8 × the patch
//! grid. The training entry letterboxes the frame to the half-native portrait/landscape size,
//! resamples the normals to a square map and L2-normalizes them. Weight keys are the Sapiens
//! checkpoint's own (`backbone.*` with mmcv FFN naming, `decode_head.*`).

use mlx_rs::fast::{layer_norm, scaled_dot_product_attention};
use mlx_rs::ops::indexing::IndexOp;
use mlx_rs::ops::{add, conv_transpose2d, divide, multiply, subtract};
use mlx_rs::Array;

use mlx_gen::gen_core::train::body::{Letterbox, SapiensConfig, IMAGENET_MEAN, IMAGENET_STD};

use super::{conv_ohwi, normalize, resample_nhwc, w32, AxisMatrix};
use mlx_gen::nn::{conv2d, gelu_exact, linear, silu};
use mlx_gen::weights::Weights;
use mlx_gen::Result;

struct Layer {
    ln1: (Array, Array),
    qkv: (Array, Array),
    proj: (Array, Array),
    ln2: (Array, Array),
    fc1: (Array, Array),
    fc2: (Array, Array),
}

struct Stage {
    deconv: Array,
    conv: (Array, Array),
}

/// The loaded Sapiens normal estimator.
pub struct SapiensNormal {
    cfg: SapiensConfig,
    proj_w: Array,
    proj_b: Array,
    /// `[1, ph, pw, C]` stored position grid.
    pos: Array,
    layers: Vec<Layer>,
    ln: (Array, Array),
    stages: Vec<Stage>,
    seg: (Array, Array),
}

impl SapiensNormal {
    /// Load from a directory holding the checkpoint (`model.safetensors`, Sapiens key layout).
    pub fn from_dir(dir: impl AsRef<std::path::Path>, cfg: SapiensConfig) -> Result<Self> {
        Self::from_weights(&Weights::from_dir(dir)?, "", cfg)
    }

    /// Load from weights whose Sapiens keys sit under `prefix`.
    pub fn from_weights(w: &Weights, prefix: &str, cfg: SapiensConfig) -> Result<Self> {
        let g = |k: &str| w32(w, prefix, k);
        let pair = |k: &str| -> Result<(Array, Array)> {
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
                let (cw, cb) = pair(&format!("decode_head.conv_layers.{}", 3 * i))?;
                Ok(Stage {
                    deconv: g(&format!("decode_head.deconv_layers.{}.weight", 3 * i))?
                        .transpose_axes(&[1, 2, 3, 0])?,
                    conv: (conv_ohwi(&cw)?, cb),
                })
            })
            .collect::<Result<Vec<_>>>()?;
        let (sw, sb) = pair("decode_head.conv_seg")?;
        let (ph, pw) = cfg.pos_grid;
        Ok(Self {
            proj_w: conv_ohwi(&g("backbone.patch_embed.projection.weight")?)?,
            proj_b: g("backbone.patch_embed.projection.bias")?,
            pos: g("backbone.pos_embed")?.reshape(&[
                1,
                ph as i32,
                pw as i32,
                cfg.embed_dim as i32,
            ])?,
            layers,
            ln: pair("backbone.ln1")?,
            stages,
            seg: (conv_ohwi(&sw)?, sb),
            cfg,
        })
    }

    /// The configuration.
    pub fn config(&self) -> &SapiensConfig {
        &self.cfg
    }

    fn instance_norm(&self, x: &Array) -> Result<Array> {
        let mean = x.mean_axes(&[1, 2], true)?;
        let d = subtract(x, &mean)?;
        let var = d.square()?.mean_axes(&[1, 2], true)?;
        Ok(divide(
            &d,
            &add(&var, Array::from_f32(self.cfg.instance_norm_eps))?.sqrt()?,
        )?)
    }

    /// `[0, 1]` NHWC pixels `[B, H, W, 3]` (any size) → raw normals NHWC `[B, 8·gh, 8·gw, 3]`.
    pub fn forward(&self, pixels: &Array) -> Result<Array> {
        let c = &self.cfg;
        let x = normalize(pixels, IMAGENET_MEAN, IMAGENET_STD)?;
        let y = conv2d(&x, &self.proj_w, Some(&self.proj_b), c.patch_size as i32, 2)?;
        let sh = y.shape();
        let (b, gh, gw, e) = (sh[0], sh[1], sh[2], sh[3]);
        let (ph, pw) = c.pos_grid;
        let pos = if (gh as usize, gw as usize) == (ph, pw) {
            self.pos.clone()
        } else {
            resample_nhwc(
                &self.pos,
                &AxisMatrix::resize(ph, gh as usize, false),
                &AxisMatrix::resize(pw, gw as usize, false),
            )?
        };
        let mut t = add(&y, &pos)?.reshape(&[b, gh * gw, e])?;
        let heads = c.num_heads as i32;
        let hd = e / heads;
        let eps = c.layer_norm_eps;
        let n = gh * gw;
        for l in &self.layers {
            let h = layer_norm(&t, Some(&l.ln1.0), Some(&l.ln1.1), eps)?;
            let qkv = linear(&h, &l.qkv.0, &l.qkv.1)?
                .reshape(&[b, n, 3, heads, hd])?
                .transpose_axes(&[2, 0, 3, 1, 4])?;
            let (q, k, v) = (qkv.index(0), qkv.index(1), qkv.index(2));
            let a = scaled_dot_product_attention(&q, &k, &v, (hd as f32).powf(-0.5), None, None)?;
            let a = a.transpose_axes(&[0, 2, 1, 3])?.reshape(&[b, n, e])?;
            t = add(&t, &linear(&a, &l.proj.0, &l.proj.1)?)?;
            let h = layer_norm(&t, Some(&l.ln2.0), Some(&l.ln2.1), eps)?;
            let h = gelu_exact(&linear(&h, &l.fc1.0, &l.fc1.1)?)?;
            t = add(&t, &linear(&h, &l.fc2.0, &l.fc2.1)?)?;
        }
        let t = layer_norm(&t, Some(&self.ln.0), Some(&self.ln.1), eps)?;
        let mut m = t.reshape(&[b, gh, gw, e])?;
        for s in &self.stages {
            let y = conv_transpose2d(&m, &s.deconv, (2, 2), (1, 1), None, None, None)?;
            let y = silu(&self.instance_norm(&y)?)?;
            let y = conv2d(&y, &s.conv.0, Some(&s.conv.1), 1, 0)?;
            m = silu(&self.instance_norm(&y)?)?;
        }
        conv2d(&m, &self.seg.0, Some(&self.seg.1), 1, 0)
    }

    /// The letterbox of an `h × w` frame as two axis matrices `[target, in]`.
    fn letterbox_axes(&self, h: usize, w: usize) -> (Letterbox, AxisMatrix, AxisMatrix) {
        let lb = self.cfg.letterbox(h, w);
        let ay = AxisMatrix::from_weights(
            lb.target_h,
            h,
            Letterbox::axis_weights(lb.target_h, lb.new_h, lb.pad_top, h),
        );
        let ax = AxisMatrix::from_weights(
            lb.target_w,
            w,
            Letterbox::axis_weights(lb.target_w, lb.new_w, lb.pad_left, w),
        );
        (lb, ay, ax)
    }

    /// The **differentiable** training entry (upstream `DifferentiableNormalEncoder.forward`):
    /// NHWC pixels `[1, H, W, 3]` in `[0, 1]` → letterbox → [`forward`](Self::forward) → bilinear
    /// resample to `normal_size²` → L2-normalized (`+ 1e-5`) normals NHWC `[1, S, S, 3]`.
    pub fn forward_pixels(&self, pixels: &Array) -> Result<Array> {
        let sh = pixels.shape();
        let (_, ay, ax) = self.letterbox_axes(sh[1] as usize, sh[2] as usize);
        let raw = self.forward(&resample_nhwc(pixels, &ay, &ax)?)?;
        let rs = raw.shape();
        let s = self.cfg.normal_size;
        let raw = resample_nhwc(
            &raw,
            &AxisMatrix::resize(rs[1] as usize, s, false),
            &AxisMatrix::resize(rs[2] as usize, s, false),
        )?;
        let norm = raw.square()?.sum_axes(&[3], true)?.sqrt()?;
        Ok(divide(&raw, &add(&norm, Array::from_f32(1e-5))?)?)
    }

    /// A subject mask `[H, W]` (frame pixels) carried onto the normal grid exactly like the
    /// pixels: the same letterbox, then the same bilinear resample → `[S, S]`.
    pub fn mask_to_normal_grid(&self, mask: &Array) -> Result<Array> {
        let sh = mask.shape();
        let (h, w) = (sh[0] as usize, sh[1] as usize);
        let (lb, ay, ax) = self.letterbox_axes(h, w);
        let m = resample_nhwc(&mask.reshape(&[1, sh[0], sh[1], 1])?, &ay, &ax)?;
        let s = self.cfg.normal_size;
        let m = resample_nhwc(
            &m,
            &AxisMatrix::resize(lb.target_h, s, false),
            &AxisMatrix::resize(lb.target_w, s, false),
        )?;
        Ok(m.reshape(&[s as i32, s as i32])?)
    }
}

/// Upstream's normal comparison for one image: per-pixel cosine and channel-mean L1 between unit
/// normal maps `[1, S, S, 3]`, averaged spatially (weighted by `mask` `[S, S]` when given, the
/// divisor clamped to ≥ 1), → `(1 − cos) + L1`.
pub fn normal_comparison(reference: &Array, live: &Array, mask: Option<&Array>) -> Result<Array> {
    let cos = multiply(reference, live)?.sum_axes(&[3], false)?; // [1, S, S]
    let l1 = subtract(reference, live)?.abs()?.mean_axes(&[3], false)?;
    let (cos_m, l1_m) = match mask {
        Some(m) => {
            let m = m.reshape(&[1, m.shape()[0], m.shape()[1]])?;
            let denom = mlx_rs::ops::maximum(&m.sum(None)?, Array::from_f32(1.0))?;
            (
                divide(&multiply(&cos, &m)?.sum(None)?, &denom)?,
                divide(&multiply(&l1, &m)?.sum(None)?, &denom)?,
            )
        }
        None => (cos.mean(None)?, l1.mean(None)?),
    };
    Ok(add(&subtract(Array::from_f32(1.0), &cos_m)?, &l1_m)?)
}
