//! `DiCoBlock` / `_EncoderDiCoBlock` — the depthwise-conv block both halves of the codec are
//! built from (`_vendor/mage_flow/models/modules/mage_vae.py:112-177`).
//!
//! ## Shape
//!
//! ```text
//! spatial: conv1(1×1) → conv2(3×3 depthwise, groups=C) → gelu → × CA → conv3(1×1)
//! channel: conv4(1×1, ×4) → gelu → conv5(1×1)
//! ```
//!
//! `CA` is a squeeze-and-excite gate: global average pool → 1×1 conv → sigmoid
//! (`mage_vae.py:121-125`). `gelu` is `F.gelu`'s **default** (`erf`) form, not the `tanh`
//! approximation — note this differs from the DiT, whose FFN is deliberately
//! [`gelu-approximate`](crate::config::FFN_ACTIVATION).
//!
//! ## adaLN, and why folding it at `t = 0` is sound
//!
//! [`DiCoBlock`]'s modulation is `SiLU → Linear(C → 6C)` applied to the **timestep embedding
//! alone** (`mage_vae.py:134-137,140`) — it has no dependence on the image, the latent or the
//! position. The codec is a *one-step* diffusion model evaluated only ever at `t = 0`
//! (`mage_vae.py:602,632`), so that modulation is a compile-time constant. The reference folds it
//! into a buffer and drops the MLPs (`_replace_adaln_with_const`, `:343-370`, ~37M parameters
//! across both halves; ~18.6M on the decode path this story ships), and [`DiCoBlock::fold_adaln`]
//! does the same.
//!
//! **The fold is deliberately reversible**: [`AdaLn::Mlp`] keeps the runtime path alive so
//! `tests/vae_decode_real_weights.rs` can build the same decoder twice and assert the two produce
//! bit-identical output. A fold that silently mis-ordered the six chunks, or shared one block's
//! modulation across all 21, would otherwise be invisible.
//!
//! `_MLPResBlock`'s adaLN is **not** foldable and is not folded (`mage_vae.py:356-358`): it is
//! conditioned on a per-position latent, not on `t`.

use mlx_rs::ops::{add, multiply, sigmoid, split};
use mlx_rs::Array;

use mlx_gen::nn::{gelu_exact, modulate, silu};
use mlx_gen::weights::Weights;
use mlx_gen::{Error, Result};

use super::layers::{layer_norm_2d, Conv2d, LayerNormAffine, Linear};

/// The six per-channel adaLN vectors, each already shaped `[.., 1, 1, C]` for NHWC broadcast.
#[derive(Clone)]
pub struct Modulation {
    shift_msa: Array,
    scale_msa: Array,
    gate_msa: Array,
    shift_mlp: Array,
    scale_mlp: Array,
    gate_mlp: Array,
}

impl Modulation {
    /// Split a `[B, 6C]` modulation into its six `[B, 1, 1, C]` components, in the reference's
    /// order: `shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp`
    /// (`mage_vae.py:140`).
    fn from_packed(packed: &Array) -> Result<Self> {
        let parts = split(packed, 6, -1)?;
        if parts.len() != 6 {
            return Err(Error::Msg(format!(
                "mage-vae adaLN: expected 6 chunks, got {}",
                parts.len()
            )));
        }
        // [B, C] -> [B, 1, 1, C]: broadcasts over the NHWC spatial axes exactly as the
        // reference's `scale.view(b, c, 1, 1)` does over NCHW (`mage_vae.py:33-37`).
        let nhwc = |a: &Array| -> Result<Array> {
            let b = a.shape()[0];
            Ok(a.reshape(&[b, 1, 1, -1])?)
        };
        Ok(Self {
            shift_msa: nhwc(&parts[0])?,
            scale_msa: nhwc(&parts[1])?,
            gate_msa: nhwc(&parts[2])?,
            shift_mlp: nhwc(&parts[3])?,
            scale_mlp: nhwc(&parts[4])?,
            gate_mlp: nhwc(&parts[5])?,
        })
    }
}

/// Where a [`DiCoBlock`]'s modulation comes from.
pub enum AdaLn {
    /// The live `SiLU → Linear(C → 6C)` MLP (`mage_vae.py:134-137`).
    Mlp(Linear),
    /// The `t = 0` modulation, precomputed; the MLP has been dropped.
    Folded(Modulation),
}

impl AdaLn {
    /// Load `{prefix}.adaLN_modulation.1` (index 0 is the `SiLU`).
    pub fn from_weights(w: &Weights, prefix: &str) -> Result<Self> {
        Ok(Self::Mlp(Linear::from_weights(
            w,
            &format!("{prefix}.adaLN_modulation.1"),
        )?))
    }

    /// Evaluate the modulation for a `[B, C]` conditioning vector.
    fn modulation(&self, c: &Array) -> Result<Modulation> {
        match self {
            Self::Mlp(linear) => Modulation::from_packed(&linear.forward(&silu(c)?)?),
            Self::Folded(m) => Ok(m.clone()),
        }
    }

    /// Replace the MLP with its value at the fixed conditioning vector `c`, freeing the weights.
    ///
    /// Idempotent: folding an already-folded block is a no-op rather than an error, mirroring the
    /// reference's `isinstance(adaln, _ConstAdaLN)` guard (`mage_vae.py:364-365`).
    fn fold(&mut self, c: &Array) -> Result<()> {
        if let Self::Mlp(_) = self {
            *self = Self::Folded(self.modulation(c)?);
        }
        Ok(())
    }

    /// Whether the adaLN MLP has been constant-folded away.
    pub fn is_folded(&self) -> bool {
        matches!(self, Self::Folded(_))
    }
}

/// The convolutional body shared by [`DiCoBlock`] and [`EncoderDiCoBlock`].
pub struct DiCoCore {
    conv1: Conv2d,
    conv2: Conv2d,
    conv3: Conv2d,
    ca: Conv2d,
    conv4: Conv2d,
    conv5: Conv2d,
}

impl DiCoCore {
    /// Load the six convolutions under `{prefix}` (`ca.1` is the only `ca` leaf with weights —
    /// index 0 is the pool and index 2 the sigmoid).
    pub fn from_weights(w: &Weights, prefix: &str, hidden_size: i32) -> Result<Self> {
        Ok(Self {
            conv1: Conv2d::pointwise(w, &format!("{prefix}.conv1"), true)?,
            // groups = hidden_size: the depthwise 3×3 (`mage_vae.py:118`).
            conv2: Conv2d::from_weights(w, &format!("{prefix}.conv2"), 1, 1, hidden_size, true)?,
            conv3: Conv2d::pointwise(w, &format!("{prefix}.conv3"), true)?,
            ca: Conv2d::pointwise(w, &format!("{prefix}.ca.1"), true)?,
            conv4: Conv2d::pointwise(w, &format!("{prefix}.conv4"), true)?,
            conv5: Conv2d::pointwise(w, &format!("{prefix}.conv5"), true)?,
        })
    }

    /// `conv1 → conv2(depthwise) → gelu → × CA → conv3` (`mage_vae.py:142-144`).
    fn spatial(&self, x_nhwc: &Array) -> Result<Array> {
        let h = gelu_exact(&self.conv2.forward(&self.conv1.forward(x_nhwc)?)?)?;
        let h = multiply(&h, &self.channel_attention(&h)?)?;
        self.conv3.forward(&h)
    }

    /// Squeeze-and-excite: `AdaptiveAvgPool2d(1) → Conv2d(1×1) → Sigmoid` (`mage_vae.py:121-125`).
    fn channel_attention(&self, x_nhwc: &Array) -> Result<Array> {
        let pooled = mlx_rs::ops::mean_axes(x_nhwc, &[1, 2], true)?; // NHWC -> [B, 1, 1, C]
        Ok(sigmoid(&self.ca.forward(&pooled)?)?)
    }

    /// `conv4 → gelu → conv5` (`mage_vae.py:146-148`).
    fn channel(&self, x_nhwc: &Array) -> Result<Array> {
        self.conv5
            .forward(&gelu_exact(&self.conv4.forward(x_nhwc)?)?)
    }
}

/// `DiCoBlock` — the adaLN-modulated block (`mage_vae.py:112-149`).
///
/// Both norms are `affine=False`; the scale and shift come from adaLN.
pub struct DiCoBlock {
    core: DiCoCore,
    adaln: AdaLn,
}

impl DiCoBlock {
    /// Load one block under `{prefix}` (e.g. `blocks.7`).
    pub fn from_weights(w: &Weights, prefix: &str, hidden_size: i32) -> Result<Self> {
        Ok(Self {
            core: DiCoCore::from_weights(w, prefix, hidden_size)?,
            adaln: AdaLn::from_weights(w, prefix)?,
        })
    }

    /// Constant-fold the adaLN MLP at the conditioning vector `c` and free it.
    pub fn fold_adaln(&mut self, c: &Array) -> Result<()> {
        self.adaln.fold(c)
    }

    /// Whether this block's adaLN has been folded.
    pub fn is_folded(&self) -> bool {
        self.adaln.is_folded()
    }

    /// The packed `[B, 6C]` modulation this block applies for conditioning `c` — i.e. exactly the
    /// tensor the reference's `_ConstAdaLN` buffer holds (`mage_vae.py:346,368`), reassembled from
    /// the six split components in their declared order.
    ///
    /// Exposed so the parity suite can pin the **chunk order** against the reference. Splitting
    /// `[shift, scale, gate] × [msa, mlp]` in the wrong order still produces a plausible image, so
    /// nothing else in the test suite would catch it.
    pub fn adaln_packed(&self, c: &Array) -> Result<Array> {
        let m = self.adaln.modulation(c)?;
        let flat = |a: &Array| -> Result<Array> { Ok(a.reshape(&[a.shape()[0], -1])?) };
        Ok(mlx_rs::ops::concatenate_axis(
            &[
                flat(&m.shift_msa)?,
                flat(&m.scale_msa)?,
                flat(&m.gate_msa)?,
                flat(&m.shift_mlp)?,
                flat(&m.scale_mlp)?,
                flat(&m.gate_mlp)?,
            ],
            -1,
        )?)
    }

    /// NHWC in, NHWC out. `c` is the `[B, C]` conditioning vector; it is ignored once the block
    /// has been folded.
    pub fn forward(&self, inp_nhwc: &Array, c: &Array) -> Result<Array> {
        let m = self.adaln.modulation(c)?;

        let h = modulate(
            &layer_norm_2d(inp_nhwc, None)?,
            &m.scale_msa,
            &m.shift_msa,
            true,
        )?;
        let h = self.core.spatial(&h)?;
        let x = add(inp_nhwc, &multiply(&m.gate_msa, &h)?)?;

        let h = modulate(&layer_norm_2d(&x, None)?, &m.scale_mlp, &m.shift_mlp, true)?;
        Ok(add(&x, &multiply(&m.gate_mlp, &self.core.channel(&h)?)?)?)
    }
}

/// `_EncoderDiCoBlock` — the same body with no adaLN and ungated residuals
/// (`mage_vae.py:152-177`). Both norms keep the default `affine=True`.
///
/// Only the encoder head uses it, so nothing on the decode path constructs one yet; it lives here
/// because it shares [`DiCoCore`] verbatim, which is what lets sc-14046 add
/// `_DConvEncoder` without touching this file.
pub struct EncoderDiCoBlock {
    core: DiCoCore,
    norm1: LayerNormAffine,
    norm2: LayerNormAffine,
}

impl EncoderDiCoBlock {
    /// Load one block under `{prefix}` (e.g. `head_blocks.0`).
    pub fn from_weights(w: &Weights, prefix: &str, hidden_size: i32) -> Result<Self> {
        Ok(Self {
            core: DiCoCore::from_weights(w, prefix, hidden_size)?,
            norm1: LayerNormAffine::from_weights(w, &format!("{prefix}.norm1"))?,
            norm2: LayerNormAffine::from_weights(w, &format!("{prefix}.norm2"))?,
        })
    }

    /// NHWC in, NHWC out.
    pub fn forward(&self, inp_nhwc: &Array) -> Result<Array> {
        let h = self
            .core
            .spatial(&layer_norm_2d(inp_nhwc, Some(&self.norm1))?)?;
        let x = add(inp_nhwc, &h)?;
        let h = self.core.channel(&layer_norm_2d(&x, Some(&self.norm2))?)?;
        Ok(add(&x, &h)?)
    }
}
