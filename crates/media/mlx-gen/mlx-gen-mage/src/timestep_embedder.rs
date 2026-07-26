//! Timestep conditioning — **owned by sc-14040**.
//!
//! `MageFlowTimestepProjEmbeddings` (`_vendor/mage_flow/models/modules/mage_layers.py:24-104`):
//! `Timesteps(256, flip_sin_to_cos=True, downscale_freq_shift=0, scale=1000, max_period=10000)`
//! → `TimestepEmbedding(256 → hidden_size)`. Constants live in [`crate::config`]
//! ([`FREQUENCY_EMBEDDING_SIZE`],
//! [`TIMESTEP_SCALE`],
//! [`TIMESTEP_MAX_PERIOD`]).
//!
//! Two traps:
//!
//! 1. The input is the scheduler **sigma ∈ [0, 1]**, fed straight in (`pipeline.py:189`) — not a
//!    0..1000 timestep index. The `scale=1000` inside the embedder is what maps it.
//! 2. The sinusoidal frequency table is **deliberately rounded**, and the rounding follows the
//!    input dtype: `emb = torch.exp(exponent).to(timesteps.dtype)` (`mage_layers.py:45`). Because
//!    the whole transformer is cast to bf16 (`pipeline.py:753`) and the sigma vector is
//!    materialised at the model dtype (`pipeline.py:187`), production rounds to **bf16** — which
//!    is what [`TIMESTEP_FREQS_BF16`] records, and what the
//!    model was trained with. Hardcoding bf16 instead of following the dtype would be wrong for an
//!    f32 run: the reference does no rounding at all there, and the two differ by ~1.8e-1
//!    mean-relative in the resulting conditioning (caught by `tests/mage_flow_small.rs`).
//!
//! The reference also computes a pooled text vector `vec`, then throws it away: the DiT overwrites
//! it with zeros before `temb = temb + txt_vec` (`models/mage_flow.py:116-118`). A port needs no
//! pooled text vector.
//!
//! ## Where each dtype boundary sits
//!
//! The reference's arithmetic is deliberately mixed precision and the order matters:
//! `exponent` is f32; `exp(exponent)` is **rounded to the timestep dtype**; the outer product
//! re-promotes to f32 via `timesteps[:, None].float()`; the `× 1000`, `sin`/`cos` and the flip stay
//! f32; and only then is the whole `[B, 256]` projection cast to the model dtype for the two-layer
//! MLP (`mage_layers.py:98`). `sigma` itself is materialised at the model dtype one level up
//! (`torch.full(..., dtype=x.dtype)`, `pipeline.py:187`), so the bf16 rounding of e.g.
//! `0.94736844 → 0.9453125` happens *before* anything here runs — [`MageTimestepEmbedder::forward`]
//! therefore takes `sigma` already at the model dtype and never re-rounds it.

use mlx_rs::ops::{concatenate_axis, divide, multiply};
use mlx_rs::{Array, Dtype};

use mlx_gen::adapters::{AdaptableHost, AdaptableLinear};
use mlx_gen::array::scalar;
use mlx_gen::weights::Weights;
use mlx_gen::{nn, Error, Result};

use crate::config::{
    FREQUENCY_EMBEDDING_SIZE, TIMESTEP_DOWNSCALE_FREQ_SHIFT, TIMESTEP_FLIP_SIN_TO_COS,
    TIMESTEP_FREQS_BF16, TIMESTEP_MAX_PERIOD, TIMESTEP_SCALE,
};
use crate::transformer::Linear;

/// `Timesteps(256, …)` → `TimestepEmbedding(256 → hidden_size)`.
#[derive(Debug, Clone)]
pub struct MageTimestepEmbedder {
    linear_1: Linear,
    linear_2: Linear,
    frequency_embedding_size: i32,
    scale: f32,
    max_period: f32,
    downscale_freq_shift: f32,
    flip_sin_to_cos: bool,
    round_freqs_to_input_dtype: bool,
}

impl MageTimestepEmbedder {
    pub fn quantize(&mut self, bits: i32) -> Result<()> {
        self.linear_1.quantize(bits)?;
        self.linear_2.quantize(bits)
    }

    pub(crate) fn quantized_linear_count(&self) -> usize {
        usize::from(self.linear_1.is_quantized()) + usize::from(self.linear_2.is_quantized())
    }

    /// Load from `{prefix}.timestep_embedder.linear_{1,2}.{weight,bias}` — the diffusers
    /// `TimestepEmbedding` layout, `prefix = "time_text_embed"` in the published checkpoint. The
    /// sinusoidal `time_proj` half is weightless.
    pub fn from_weights(w: &Weights, prefix: &str) -> Result<Self> {
        Ok(Self {
            linear_1: Linear::from_weights(w, &format!("{prefix}.timestep_embedder.linear_1"))?,
            linear_2: Linear::from_weights(w, &format!("{prefix}.timestep_embedder.linear_2"))?,
            frequency_embedding_size: FREQUENCY_EMBEDDING_SIZE,
            scale: TIMESTEP_SCALE,
            max_period: TIMESTEP_MAX_PERIOD,
            downscale_freq_shift: TIMESTEP_DOWNSCALE_FREQ_SHIFT,
            flip_sin_to_cos: TIMESTEP_FLIP_SIN_TO_COS,
            round_freqs_to_input_dtype: TIMESTEP_FREQS_BF16,
        })
    }

    /// Disable the deliberate rounding of the frequency table — **a divergence knob for the parity
    /// suite only**, so a test can demonstrate that an un-rounded (f32) table really does move the
    /// conditioning. Production never calls this ([`TIMESTEP_FREQS_BF16`] is the shipped setting,
    /// and at bf16 the rounding is the one the model was trained with).
    pub fn set_round_frequency_table(&mut self, on: bool) {
        self.round_freqs_to_input_dtype = on;
    }

    pub fn cast_weights(&mut self, dtype: Dtype) -> Result<()> {
        self.linear_1.cast_weights(dtype)?;
        self.linear_2.cast_weights(dtype)
    }

    /// Conditioning width (`hidden_size`).
    pub fn out_dim(&self) -> i32 {
        self.linear_2.out_features()
    }

    /// `[segments]` sigma → `[segments, hidden_size]` conditioning.
    pub fn forward(&self, sigma: &Array) -> Result<Array> {
        if sigma.ndim() != 1 {
            return Err(Error::Msg(format!(
                "mage_flow: timestep embedder expects a 1-D sigma vector, got shape {:?}",
                sigma.shape()
            )));
        }
        let proj = self.projection(sigma)?;
        // `timesteps_proj.to(dtype=hidden_states.dtype)` (`mage_layers.py:98`).
        let proj = proj.as_dtype(self.linear_1.dtype())?;
        let hidden = nn::silu(&self.linear_1.forward(&proj)?)?;
        self.linear_2.forward(&hidden)
    }

    /// The `Timesteps` half alone: `[B]` → `[B, frequency_embedding_size]`, always f32.
    ///
    /// Exposed so the parity suite can gate the sinusoid independently of the MLP — the bf16
    /// frequency rounding is invisible in the final conditioning unless it is looked at directly.
    pub fn projection(&self, timesteps: &Array) -> Result<Array> {
        let half = self.frequency_embedding_size / 2;
        if half <= 0 || self.frequency_embedding_size % 2 != 0 {
            return Err(Error::Msg(format!(
                "mage_flow: frequency embedding size must be positive and even (got {})",
                self.frequency_embedding_size
            )));
        }
        // `exponent = -log(max_period) * arange(half) / (half - downscale_freq_shift)`, built with
        // MLX ops rather than a host `libm` loop (the F-016 lesson: a ~1e-7 host/device gap flips a
        // bf16 conditioning ULP that a deep joint stack amplifies).
        let arange: Vec<f32> = (0..half).map(|i| i as f32).collect();
        let neg_log = -self.max_period.ln();
        let denom = half as f32 - self.downscale_freq_shift;
        let exponent = divide(
            multiply(Array::from_slice(&arange, &[1, half]), scalar(neg_log))?,
            scalar(denom),
        )?;
        let mut freqs = exponent.exp()?;
        if self.round_freqs_to_input_dtype && timesteps.dtype() != Dtype::Float32 {
            // `torch.exp(exponent).to(timesteps.dtype)` — the table follows the INPUT dtype, which
            // in production is bf16 and is the rounding the model was trained with. Cast straight
            // back to f32: the very next op promotes anyway
            // (`timesteps[:, None].float() * emb[None, :]`) and the rounded values are exact in f32.
            freqs = freqs
                .as_dtype(timesteps.dtype())?
                .as_dtype(Dtype::Float32)?;
        }
        let t = timesteps
            .reshape(&[timesteps.shape()[0], 1])?
            .as_dtype(Dtype::Float32)?;
        let emb = multiply(&t, &freqs)?;
        // `emb = scale * emb` AFTER the outer product (`mage_layers.py:46,49`).
        let emb = multiply(scalar(self.scale), &emb)?;
        let (sin, cos) = (emb.sin()?, emb.cos()?);
        // `cat([sin, cos])` then `flip_sin_to_cos` swaps the halves ⇒ `[cos, sin]`
        // (`mage_layers.py:52-56`).
        Ok(if self.flip_sin_to_cos {
            concatenate_axis(&[&cos, &sin], 1)?
        } else {
            concatenate_axis(&[&sin, &cos], 1)?
        })
    }
}

/// LoRA/LoKr targets on the timestep conditioning (sc-14057). The `Timesteps` half is weightless,
/// so the only adaptable leaves are the two `TimestepEmbedding` projections — named exactly as the
/// published checkpoint spells them under the `time_text_embed` root
/// ([`MageTimestepEmbedder::from_weights`]), which is also how a PEFT `target_modules="all-linear"`
/// community adapter names them.
impl AdaptableHost for MageTimestepEmbedder {
    fn adaptable_mut(&mut self, path: &[&str]) -> Option<&mut AdaptableLinear> {
        match path {
            ["timestep_embedder", "linear_1"] => Some(self.linear_1.adaptable_mut()),
            ["timestep_embedder", "linear_2"] => Some(self.linear_2.adaptable_mut()),
            _ => None,
        }
    }

    fn adaptable_paths(&self) -> Vec<String> {
        vec![
            "timestep_embedder.linear_1".to_string(),
            "timestep_embedder.linear_2".to_string(),
        ]
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A 1 → 1 embedder: enough to exercise the sinusoid, which is where every trap lives.
    fn embedder() -> MageTimestepEmbedder {
        let mut w = Weights::empty();
        for name in ["linear_1", "linear_2"] {
            let (out, inp) = if name == "linear_1" {
                (1, FREQUENCY_EMBEDDING_SIZE)
            } else {
                (1, 1)
            };
            w.insert(
                format!("t.timestep_embedder.{name}.weight"),
                Array::from_slice(&vec![0.01f32; (out * inp) as usize], &[out, inp]),
            );
            w.insert(
                format!("t.timestep_embedder.{name}.bias"),
                Array::from_slice(&[0.0f32], &[out]),
            );
        }
        MageTimestepEmbedder::from_weights(&w, "t").unwrap()
    }

    /// At f32 the reference rounds to f32 — i.e. not at all — so the port must NOT hardcode a bf16
    /// table. Getting this wrong is invisible on the bf16 production path and moves an f32 run's
    /// conditioning by ~1.8e-1 mean-relative (`mage_layers.py:45`: `.to(timesteps.dtype)`).
    #[test]
    fn an_f32_sigma_gets_an_unrounded_table() {
        let sigma = Array::from_slice(&[0.7f32], &[1]);
        let mut unrounded = embedder();
        unrounded.set_round_frequency_table(false);
        let a = embedder().projection(&sigma).unwrap();
        let b = unrounded.projection(&sigma).unwrap();
        assert_eq!(
            a.as_slice::<f32>(),
            b.as_slice::<f32>(),
            "rounding an f32 table to f32 is a no-op; the port must follow the input dtype"
        );
    }

    #[test]
    fn flip_sin_to_cos_puts_cosine_first() {
        // At sigma = 0 every angle is 0 ⇒ the cos half is all ones, the sin half all zeros.
        let proj = embedder()
            .projection(&Array::from_slice(&[0.0f32], &[1]))
            .unwrap();
        let row = proj.as_slice::<f32>();
        assert_eq!(row.len(), FREQUENCY_EMBEDDING_SIZE as usize);
        let half = row.len() / 2;
        assert!(row[..half].iter().all(|&v| v == 1.0), "cos half must lead");
        assert!(row[half..].iter().all(|&v| v == 0.0), "sin half must trail");
    }

    /// The rounding of the frequency table is deliberate (`mage_layers.py:32-46`) — and it is
    /// **observable** at bf16, so a port that "improved" it to f32 would produce a different
    /// conditioning. This is the probe that makes [`TIMESTEP_FREQS_BF16`] non-vacuous.
    #[test]
    fn the_bf16_frequency_table_is_not_the_same_as_an_f32_one() {
        let sigma = Array::from_slice(&[0.94736844f32], &[1])
            .as_dtype(Dtype::Bfloat16)
            .unwrap();
        let rounded = embedder().projection(&sigma).unwrap();
        let mut exact = embedder();
        exact.set_round_frequency_table(false);
        let exact = exact.projection(&sigma).unwrap();
        let (a, b) = (rounded.as_slice::<f32>(), exact.as_slice::<f32>());
        let diff = a.iter().zip(b).fold(0f32, |m, (x, y)| m.max((x - y).abs()));
        assert!(
            diff > 1e-3,
            "bf16 vs f32 frequency table must be distinguishable (max_abs {diff})"
        );
    }

    /// The scale is applied to the whole angle, so it is not absorbable into the frequency table's
    /// rounding: sigma ∈ [0, 1] only reaches the interesting part of the sinusoid because of it.
    #[test]
    fn sigma_is_scaled_by_a_thousand_not_used_as_a_raw_timestep() {
        let e = embedder();
        assert_eq!(e.scale, TIMESTEP_SCALE);
        // ω₀ = 1, so lane `half` (the first sine lane) is sin(1000·σ).
        let proj = e.projection(&Array::from_slice(&[0.001f32], &[1])).unwrap();
        let row = proj.as_slice::<f32>();
        let half = row.len() / 2;
        assert!(
            (row[half] - 1.0f32.sin()).abs() < 1e-5,
            "expected sin(1000·0.001), got {}",
            row[half]
        );
    }

    #[test]
    fn a_non_vector_sigma_is_rejected() {
        let bad = Array::from_slice(&[1.0f32, 1.0], &[2, 1]);
        assert!(embedder().forward(&bad).is_err());
    }

    /// sc-14057: both `TimestepEmbedding` projections are adapter targets, spelled exactly as the
    /// checkpoint (and a PEFT `all-linear` community adapter) names them. The weightless sinusoid
    /// half has no target, and a bogus leaf must not resolve.
    #[test]
    fn the_timestep_projections_are_routable_adapter_targets() {
        let mut e = embedder();
        assert_eq!(
            e.adaptable_paths(),
            ["timestep_embedder.linear_1", "timestep_embedder.linear_2"]
        );
        for path in e.adaptable_paths() {
            let segs: Vec<&str> = path.split('.').collect();
            assert!(
                e.adaptable_mut(&segs).is_some(),
                "{path} is enumerated but does not resolve"
            );
        }
        assert!(e
            .adaptable_mut(&["timestep_embedder", "linear_3"])
            .is_none());
        assert!(e.adaptable_mut(&["time_proj"]).is_none());
    }
}
