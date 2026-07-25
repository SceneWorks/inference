//! Shared Mage-VAE primitives: PyTorch-shaped `Conv2d`/`Linear` wrappers and the channels-last
//! `LayerNorm2d`.
//!
//! Port of the primitive layers at the top of
//! `_vendor/mage_flow/models/modules/mage_vae.py` (`:25-68`).
//!
//! ## Layout convention
//!
//! Every module in this directory works in **NHWC** internally (mlx convolutions and norms are
//! channels-last) and only the public [`decode`](super::MageVae::decode) boundary speaks NCHW,
//! matching the `mlx-gen-z-image/src/vae/` sibling. The reference's `LayerNorm2d`
//! (`mage_vae.py:40-50`) is a `permute(0,2,3,1) → layer_norm → permute(0,3,1,2)` sandwich purely
//! because PyTorch holds NCHW; in NHWC it collapses to a plain last-axis layer-norm, which is what
//! [`layer_norm_2d`] is.

use mlx_rs::fast::layer_norm;
use mlx_rs::ops::{conv2d as conv2d_op, multiply};
use mlx_rs::Array;

use mlx_gen::weights::Weights;
use mlx_gen::Result;

/// `eps` for every `LayerNorm2d` / `RMSNorm` / `GroupNorm` in the codec (`mage_vae.py:30,41,58`).
pub const NORM_EPS: f32 = 1e-6;

/// `GroupNorm(num_groups=32)` — the CoD decoder's `Normalize` (`mage_vae.py:29-30`).
pub const GN_GROUPS: i32 = 32;

/// A PyTorch `nn.Conv2d` whose weight has been transposed to mlx's `[out, kH, kW, in/groups]`
/// layout at load time.
///
/// `groups` carries the depthwise case: `DiCoBlock.conv2` is `groups=hidden_size`
/// (`mage_vae.py:118`), whose PyTorch weight `[C, 1, 3, 3]` transposes to exactly the `[C, 3, 3, 1]`
/// mlx wants.
pub struct Conv2d {
    weight: Array,
    bias: Option<Array>,
    stride: i32,
    padding: i32,
    groups: i32,
}

impl Conv2d {
    /// Load `{prefix}.weight` (+ `{prefix}.bias` when `bias`), transposing NCHW→NHWC.
    pub fn from_weights(
        w: &Weights,
        prefix: &str,
        stride: i32,
        padding: i32,
        groups: i32,
        bias: bool,
    ) -> Result<Self> {
        let weight = w
            .require(&format!("{prefix}.weight"))?
            .transpose_axes(&[0, 2, 3, 1])?; // [out, in/groups, kH, kW] -> [out, kH, kW, in/groups]
        let bias = if bias {
            Some(w.require(&format!("{prefix}.bias"))?.clone())
        } else {
            None
        };
        Ok(Self {
            weight,
            bias,
            stride,
            padding,
            groups,
        })
    }

    /// 1×1 stride-1 pad-0 dense conv.
    pub fn pointwise(w: &Weights, prefix: &str, bias: bool) -> Result<Self> {
        Self::from_weights(w, prefix, 1, 0, 1, bias)
    }

    /// 3×3 stride-1 pad-1 dense conv.
    pub fn conv3x3(w: &Weights, prefix: &str) -> Result<Self> {
        Self::from_weights(w, prefix, 1, 1, 1, true)
    }

    /// NHWC in, NHWC out.
    pub fn forward(&self, x_nhwc: &Array) -> Result<Array> {
        let y = conv2d_op(
            x_nhwc,
            &self.weight,
            (self.stride, self.stride),
            (self.padding, self.padding),
            (1, 1),
            self.groups,
        )?;
        Ok(match &self.bias {
            Some(b) => mlx_rs::ops::add(&y, b)?,
            None => y,
        })
    }
}

/// A PyTorch `nn.Linear` (weight `[out, in]`, always biased in this codec).
pub struct Linear {
    inner: mlx_gen::adapters::AdaptableLinear,
    in_features: i32,
}

impl Linear {
    /// Load `{prefix}.weight` + `{prefix}.bias`.
    pub fn from_weights(w: &Weights, prefix: &str) -> Result<Self> {
        let weight = w.require(&format!("{prefix}.weight"))?;
        Ok(Self {
            inner: mlx_gen::quant::lin(w, prefix, true, 64)?,
            in_features: weight.shape()[1],
        })
    }

    /// `x @ weightᵀ + bias` over the last axis.
    pub fn forward(&self, x: &Array) -> Result<Array> {
        self.inner.forward(x)
    }

    pub fn quantize(&mut self, bits: i32) -> Result<()> {
        if self.in_features % 64 == 0 {
            self.inner.quantize(bits, None)?;
        }
        Ok(())
    }

    pub(crate) fn quantization_count(&self) -> (usize, usize) {
        if self.in_features % 64 == 0 {
            (1, usize::from(self.inner.is_quantized()))
        } else {
            (0, 0)
        }
    }
}

/// Optional per-channel affine for [`layer_norm_2d`].
///
/// `DiCoBlock` builds its two norms with `affine=False` (`mage_vae.py:131-132`) because adaLN
/// supplies the scale and shift; `_EncoderDiCoBlock` and `_DConvEncoder.norm_out` keep the default
/// `affine=True` (`:168-169`, `:428`).
pub struct LayerNormAffine {
    weight: Array,
    bias: Array,
}

impl LayerNormAffine {
    /// Load `{prefix}.weight` + `{prefix}.bias`.
    pub fn from_weights(w: &Weights, prefix: &str) -> Result<Self> {
        Ok(Self {
            weight: w.require(&format!("{prefix}.weight"))?.clone(),
            bias: w.require(&format!("{prefix}.bias"))?.clone(),
        })
    }

    /// PyTorch `GroupNorm(32, eps=1e-6)` over NHWC `x` — the reference's `Normalize`
    /// (`mage_vae.py:29-30`). The CoD decoder reuses these same affine parameters as a group norm
    /// rather than a layer norm, which is why it hangs off this type.
    pub fn group_norm(&self, x_nhwc: &Array) -> Result<Array> {
        mlx_gen::nn::group_norm(x_nhwc, &self.weight, &self.bias, GN_GROUPS, NORM_EPS)
    }
}

/// Channels-last `LayerNorm2d` (`mage_vae.py:40-50`) — a last-axis layer-norm over NHWC `x`.
pub fn layer_norm_2d(x_nhwc: &Array, affine: Option<&LayerNormAffine>) -> Result<Array> {
    Ok(match affine {
        Some(a) => layer_norm(x_nhwc, &a.weight, &a.bias, NORM_EPS)?,
        None => layer_norm(x_nhwc, None, None, NORM_EPS)?,
    })
}

/// The reference's `RMSNorm` (`mage_vae.py:57-68`) over the last axis.
pub struct RmsNorm {
    weight: Array,
}

impl RmsNorm {
    /// Load `{prefix}.weight` (there is no bias).
    pub fn from_weights(w: &Weights, prefix: &str) -> Result<Self> {
        Ok(Self {
            weight: w.require(&format!("{prefix}.weight"))?.clone(),
        })
    }

    /// `weight * x * rsqrt(mean(x²) + eps)`.
    pub fn forward(&self, x: &Array) -> Result<Array> {
        // Written out rather than delegated to mlx's fused `rms_norm`, which folds the affine
        // weight into the kernel: the reference casts back to the input dtype *before* the affine
        // multiply (`mage_vae.py:63-68`), so the two orders are only interchangeable in f32.
        let var = mlx_rs::ops::mean_axis(&multiply(x, x)?, -1, true)?;
        let scaled = mlx_rs::ops::add(&var, Array::from_f32(NORM_EPS))?;
        let normed = multiply(x, &mlx_rs::ops::rsqrt(&scaled)?)?;
        Ok(multiply(&self.weight, &normed)?)
    }
}
