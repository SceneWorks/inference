//! Per-stream feed-forward network — **owned by sc-14040**.
//!
//! `FeedForward(dim, dim_out=dim, activation_fn="gelu-approximate")`
//! (`_vendor/mage_flow/models/modules/mage_layers.py:547`, `:557`) on **both** streams.
//!
//! **This is not SwiGLU.** The epic's original reuse note assumed the `mlx-gen-z-image` sibling's
//! SwiGLU FFN because Mage's own config declares `schedule_mode: "z-image"`; the vendored code says
//! otherwise. [`crate::config::FFN_ACTIVATION`] pins it so the sibling's activation cannot be
//! inherited by accident, and [`crate::config::MLP_RATIO`] pins the 4.0 expansion (hardcoded in
//! the reference's code, *not* read from `transformer/config.json`).
//!
//! diffusers' `FeedForward` is a three-entry `ModuleList` — `GELU(dim → inner, approximate="tanh")`,
//! a dropout, then `Linear(inner → dim)` — so the checkpoint keys are `net.0.proj.*` and `net.2.*`
//! with `net.1` weightless. There is **no gate projection**: `inner_dim = 4 · dim = 12288` and the
//! activation is applied to the whole projection, unlike the sibling's `w2(silu(w1 x) · w3 x)`.

use mlx_rs::{Array, Dtype};

use mlx_gen::weights::Weights;
use mlx_gen::{nn, Result};

use crate::config::FFN_ACTIVATION;
use crate::transformer::Linear;

/// The pointwise activation between the two projections.
///
/// diffusers' `FeedForward` takes `activation_fn` as a constructor parameter and switches on it
/// (`activation.py`), so modelling it as a value rather than baking `gelu-approximate` into the
/// forward mirrors the reference — and gives the parity suite the mutation it needs to prove the
/// gate discriminates. Loading always selects from [`crate::config::FFN_ACTIVATION`]; nothing in
/// production ever calls [`MageFeedForward::set_activation`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FfnActivation {
    /// `"gelu-approximate"` — the tanh approximation, and the one Mage passes
    /// (`mage_layers.py:547`, `:557`).
    GeluApproximate,
    /// `"gelu"` — diffusers' exact/erf GELU. The nearest silent near-miss to the shipped value.
    Gelu,
    /// The SiLU that gates a SwiGLU FFN.
    ///
    /// A **real** SwiGLU cannot load this checkpoint at all: `mlx-gen-z-image`'s spelling wants a
    /// `w1`/`w2`/`w3` triple, and diffusers' `"swiglu"` wants a doubled `net.0.proj`
    /// (`Linear(3072 → 24576)`), while the published tensor is `Linear(3072 → 12288)`. So the part
    /// of "inherit the sibling's SwiGLU FFN" that could survive the load is exactly this
    /// activation swap — which is what the parity suite mutates to prove its tolerance is not
    /// vacuous.
    Silu,
}

impl FfnActivation {
    /// Resolve a diffusers `activation_fn` string, so [`crate::config::FFN_ACTIVATION`] is the
    /// single source of truth rather than a comment next to a hardcoded call.
    pub fn from_name(name: &str) -> Result<Self> {
        match name {
            "gelu-approximate" => Ok(Self::GeluApproximate),
            "gelu" => Ok(Self::Gelu),
            "silu" => Ok(Self::Silu),
            other => Err(mlx_gen::Error::Unsupported(format!(
                "mage_flow: unsupported feed-forward activation {other:?}"
            ))),
        }
    }

    fn apply(self, x: &Array) -> Result<Array> {
        match self {
            Self::GeluApproximate => nn::gelu_tanh(x),
            Self::Gelu => nn::gelu_exact(x),
            Self::Silu => nn::silu(x),
        }
    }
}

/// `net.0.proj` → `gelu(tanh)` → `net.2`.
#[derive(Debug, Clone)]
pub struct MageFeedForward {
    proj: Linear,
    out: Linear,
    activation: FfnActivation,
}

impl MageFeedForward {
    pub fn quantize(&mut self, bits: i32) -> Result<()> {
        self.proj.quantize(bits)?;
        self.out.quantize(bits)
    }

    pub(crate) fn quantized_linear_count(&self) -> usize {
        usize::from(self.proj.is_quantized()) + usize::from(self.out.is_quantized())
    }

    /// Load from `{prefix}.net.0.proj.{weight,bias}` and `{prefix}.net.2.{weight,bias}` — e.g.
    /// `transformer_blocks.0.img_mlp`. Both projections carry a bias (`bias=True` default).
    pub fn from_weights(w: &Weights, prefix: &str) -> Result<Self> {
        Ok(Self {
            proj: Linear::from_weights(w, &format!("{prefix}.net.0.proj"))?,
            out: Linear::from_weights(w, &format!("{prefix}.net.2"))?,
            activation: FfnActivation::from_name(FFN_ACTIVATION)?,
        })
    }

    pub fn activation(&self) -> FfnActivation {
        self.activation
    }

    /// Swap the activation — **a divergence knob for the parity suite only**, so it can show the
    /// tolerance rejects the sibling's SwiGLU gate. Production loads from
    /// [`crate::config::FFN_ACTIVATION`] and never calls this.
    pub fn set_activation(&mut self, activation: FfnActivation) {
        self.activation = activation;
    }

    /// Expansion width (`4 · dim` in production) — read back off the loaded projection so a
    /// checkpoint's real geometry is reported rather than [`crate::config::MLP_RATIO`]'s claim.
    pub fn inner_dim(&self) -> i32 {
        self.proj.out_features()
    }

    pub fn cast_weights(&mut self, dtype: Dtype) -> Result<()> {
        self.proj.cast_weights(dtype)?;
        self.out.cast_weights(dtype)
    }

    /// `net.2(activation(net.0.proj(x)))`.
    ///
    /// [`mlx_gen::nn::gelu_tanh`] is the dtype-preserving tanh approximation matching
    /// `F.gelu(..., approximate="tanh")`; using the exact/erf GELU here would be a silent
    /// per-block divergence.
    pub fn forward(&self, x: &Array) -> Result<Array> {
        let hidden = self.activation.apply(&self.proj.forward(x)?)?;
        self.out.forward(&hidden)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn ffn(activation: FfnActivation) -> MageFeedForward {
        let mut w = Weights::empty();
        // Identity-ish 2 → 2 → 2 so the activation is the only thing under test.
        w.insert(
            "f.net.0.proj.weight",
            Array::from_slice(&[1.0f32, 0.0, 0.0, 1.0], &[2, 2]),
        );
        w.insert("f.net.0.proj.bias", Array::from_slice(&[0.0f32, 0.0], &[2]));
        w.insert(
            "f.net.2.weight",
            Array::from_slice(&[1.0f32, 0.0, 0.0, 1.0], &[2, 2]),
        );
        w.insert("f.net.2.bias", Array::from_slice(&[0.0f32, 0.0], &[2]));
        let mut ffn = MageFeedForward::from_weights(&w, "f").unwrap();
        ffn.set_activation(activation);
        ffn
    }

    #[test]
    fn loading_selects_the_pinned_activation() {
        let mut w = Weights::empty();
        w.insert("f.net.0.proj.weight", Array::from_slice(&[1.0f32], &[1, 1]));
        w.insert("f.net.0.proj.bias", Array::from_slice(&[0.0f32], &[1]));
        w.insert("f.net.2.weight", Array::from_slice(&[1.0f32], &[1, 1]));
        w.insert("f.net.2.bias", Array::from_slice(&[0.0f32], &[1]));
        assert_eq!(
            MageFeedForward::from_weights(&w, "f").unwrap().activation(),
            FfnActivation::GeluApproximate
        );
        assert_eq!(FFN_ACTIVATION, "gelu-approximate");
        assert!(FfnActivation::from_name("swiglu").is_err());
    }

    /// The three activations are numerically distinct on ordinary activations — the premise the
    /// parity suite's SwiGLU probe rests on. At x = 2, `gelu_tanh ≈ 1.9546`, `gelu ≈ 1.9545` and
    /// `silu ≈ 1.7616`: the SiLU swap is a ~10 % error per FFN, the erf swap ~1e-4.
    #[test]
    fn the_activations_are_measurably_different() {
        let x = Array::from_slice(&[2.0f32, -1.0], &[1, 2]);
        let g = ffn(FfnActivation::GeluApproximate).forward(&x).unwrap();
        let e = ffn(FfnActivation::Gelu).forward(&x).unwrap();
        let s = ffn(FfnActivation::Silu).forward(&x).unwrap();
        let (g, e, s) = (
            g.as_slice::<f32>(),
            e.as_slice::<f32>(),
            s.as_slice::<f32>(),
        );
        assert!(
            (g[0] - 1.954_598).abs() < 1e-5,
            "gelu-approximate at 2: {}",
            g[0]
        );
        assert!(
            (g[0] - e[0]).abs() > 1e-6,
            "erf GELU must not be bit-identical"
        );
        assert!(
            (g[0] - s[0]).abs() > 0.15,
            "SiLU must be far from gelu-approximate"
        );
    }
}
