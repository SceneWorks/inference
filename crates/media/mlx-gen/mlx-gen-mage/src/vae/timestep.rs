//! The Mage-VAE `TimestepEmbedder` (`_vendor/mage_flow/models/modules/mage_vae.py:71-97`).
//!
//! **This is not the DiT's timestep embedder.** The NR-MMDiT uses diffusers'
//! `Timesteps(256, flip_sin_to_cos=True, scale=1000)` with a deliberately bf16-rounded frequency
//! table ([`crate::config::TIMESTEP_FREQS_BF16`]); the codec uses this plain DiT-style
//! `cos ‖ sin` sinusoid with no scale and no flip. Keeping it in `vae/` avoids the two being
//! confused for one another.
//!
//! The codec only ever evaluates it at `t = 0` — both encode and decode are a single deterministic
//! forward (`mage_vae.py:602,632`) — which is what makes the adaLN constant-folding in
//! [`super::dico`] sound.

use mlx_rs::ops::{concatenate_axis, cos, multiply, sin};
use mlx_rs::Array;

use mlx_gen::nn::silu;
use mlx_gen::weights::Weights;
use mlx_gen::Result;

use super::layers::Linear;

/// `frequency_embedding_size` (`mage_vae.py:74`).
pub const FREQUENCY_EMBEDDING_SIZE: i32 = 256;

/// `max_period` (`mage_vae.py:84`).
pub const MAX_PERIOD: f32 = 10_000.0;

/// `sinusoid(256, max_period=10000, cos ‖ sin) → Linear → SiLU → Linear`.
pub struct TimestepEmbedder {
    fc1: Linear,
    fc2: Linear,
}

impl TimestepEmbedder {
    pub fn quantize(&mut self, bits: i32) -> Result<()> {
        self.fc1.quantize(bits)?;
        self.fc2.quantize(bits)
    }

    pub(crate) fn quantization_count(&self) -> (usize, usize) {
        let a = self.fc1.quantization_count();
        let b = self.fc2.quantization_count();
        (a.0 + b.0, a.1 + b.1)
    }

    /// Load `{prefix}.mlp.0` and `{prefix}.mlp.2` (index 1 is the `SiLU`).
    pub fn from_weights(w: &Weights, prefix: &str) -> Result<Self> {
        Ok(Self {
            fc1: Linear::from_weights(w, &format!("{prefix}.mlp.0"))?,
            fc2: Linear::from_weights(w, &format!("{prefix}.mlp.2"))?,
        })
    }

    /// The raw sinusoidal features for a `[B]` timestep vector — `cos` first, then `sin`
    /// (`mage_vae.py:90`, the opposite order from the more common `sin ‖ cos`).
    pub fn timestep_embedding(t: &Array, dtype: mlx_rs::Dtype) -> Result<Array> {
        let half = (FREQUENCY_EMBEDDING_SIZE / 2) as usize;
        let ln_max_period = MAX_PERIOD.ln();
        let freqs: Vec<f32> = (0..half)
            .map(|i| (-ln_max_period * i as f32 / half as f32).exp())
            .collect();
        let freqs = Array::from_slice(&freqs, &[1, half as i32]).as_dtype(dtype)?;
        let args = multiply(&t.reshape(&[-1, 1])?.as_dtype(dtype)?, &freqs)?;
        Ok(concatenate_axis(&[cos(&args)?, sin(&args)?], -1)?)
    }

    /// `[B]` timesteps → `[B, hidden_size]`.
    pub fn forward(&self, t: &Array, dtype: mlx_rs::Dtype) -> Result<Array> {
        let emb = Self::timestep_embedding(t, dtype)?;
        self.fc2.forward(&silu(&self.fc1.forward(&emb)?)?)
    }
}
