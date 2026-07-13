//! The backend tensor-op abstraction for the unified sampler framework (epic 7114, P1).
//!
//! The whole point of this layer: every curated solver (Euler, Heun, DPM++ 2M/SDE, UniPC, ancestral,
//! …) is written ONCE in gen-core, generic over `L: LatentOps`, and each backend supplies a small
//! impl — `mlx-gen` for `mlx_rs::Array` (sc-7118), `candle-gen` for `candle_core::Tensor` (sc-7119).
//! gen-core keeps its zero-tensor-dep invariant: the scalar coefficient math (log-SNR, Vandermonde,
//! `expm1`, …) runs in pure host `f32`/`f64` in the solver modules, and only the final per-step
//! blends touch the backend tensor through this trait.
//!
//! The surface is deliberately minimal. It was sized against the two hardest existing solvers —
//! Wan's flow-mode `dpmpp_2m` and `uni_pc` (`mlx-gen-wan/src/scheduler.rs`), which between them touch
//! latents with only scalar-multiply / add / subtract / clone — plus the stochastic samplers of the
//! curated 7117 set (`euler_ancestral`, `dpmpp_sde`), which add fresh per-step Gaussian noise. Hence
//! exactly: [`LatentOps::scale`], [`LatentOps::add`], [`LatentOps::sub`], the fused
//! [`LatentOps::axpy`], and [`LatentOps::randn_like`]. No element-wise tensor×tensor multiply is
//! required by any curated solver.

use crate::Result;

/// Backend latent-tensor operations the unified samplers are written against.
///
/// Implementors own a concrete tensor type ([`LatentOps::Latent`]); the trait is the small, fixed
/// set of element-wise operations the solver library performs on latents. Every method is fallible
/// because backend ops are (an MLX / candle op can surface an error); gen-core threads the
/// [`crate::Error`] through unchanged.
pub trait LatentOps {
    /// The backend latent tensor (`mlx_rs::Array`, `candle_core::Tensor`, … / `Vec<f32>` in tests).
    type Latent: Clone;

    /// `scale · x` — broadcast scalar multiply.
    fn scale(&self, x: &Self::Latent, scale: f32) -> Result<Self::Latent>;

    /// `a + b` — element-wise add (same shape).
    fn add(&self, a: &Self::Latent, b: &Self::Latent) -> Result<Self::Latent>;

    /// `a - b` — element-wise subtract (same shape).
    fn sub(&self, a: &Self::Latent, b: &Self::Latent) -> Result<Self::Latent>;

    /// `a·x + b·y` — the affine workhorse most solver steps reduce to. The provided default composes
    /// [`Self::scale`] + [`Self::add`]; a backend MAY override to fuse the three ops (a single MLX /
    /// candle graph) for fewer kernel launches. An override MUST stay numerically equal to the
    /// default — the per-engine N1 default-parity gate (epic 7114) depends on it.
    fn axpy(&self, a: f32, x: &Self::Latent, b: f32, y: &Self::Latent) -> Result<Self::Latent> {
        self.add(&self.scale(x, a)?, &self.scale(y, b)?)
    }

    /// Fresh unit-normal noise shaped like `x`, for the stochastic samplers (ancestral / SDE).
    ///
    /// `seed` is the request seed and `step` the 0-based denoise iteration; the impl derives a
    /// per-step subkey so the trajectory is deterministic for a given seed regardless of global RNG
    /// draw order (mirrors `mlx-gen`'s `StepRng`, sc-2769 D6). Deterministic solvers never call this.
    /// Cross-backend bitwise equality of the draw is explicitly NOT a goal (RNG algorithms differ).
    fn randn_like(&self, x: &Self::Latent, seed: u64, step: usize) -> Result<Self::Latent>;
}

/// A host-only [`LatentOps`] over `Vec<f32>` — the gen-core test / reference backend.
///
/// It lets the whole solver library be unit-tested with no tensor library (the role the legacy
/// `policy_drives_a_plain_vec_tensor_backend` test played for `SamplerPolicy`), and documents the
/// exact arithmetic a real backend must reproduce. `randn_like` uses a splitmix64 + Box–Muller draw
/// keyed by `seed`/`step`, so stochastic-sampler tests are deterministic and reproducible.
#[derive(Clone, Copy, Debug, Default)]
pub struct CpuLatentOps;

impl LatentOps for CpuLatentOps {
    type Latent = Vec<f32>;

    fn scale(&self, x: &Vec<f32>, scale: f32) -> Result<Vec<f32>> {
        Ok(x.iter().map(|&v| v * scale).collect())
    }

    fn add(&self, a: &Vec<f32>, b: &Vec<f32>) -> Result<Vec<f32>> {
        debug_assert_eq!(a.len(), b.len(), "LatentOps::add shape mismatch");
        Ok(a.iter().zip(b).map(|(&x, &y)| x + y).collect())
    }

    fn sub(&self, a: &Vec<f32>, b: &Vec<f32>) -> Result<Vec<f32>> {
        debug_assert_eq!(a.len(), b.len(), "LatentOps::sub shape mismatch");
        Ok(a.iter().zip(b).map(|(&x, &y)| x - y).collect())
    }

    fn randn_like(&self, x: &Vec<f32>, seed: u64, step: usize) -> Result<Vec<f32>> {
        // Subkey derivation mirrors mlx-gen's StepRng: de-correlate consecutive steps via the golden-
        // ratio odd constant; the `+1` keeps step 0 off the raw seed used for the init-noise prior.
        let mut state = seed.wrapping_add(0x9E37_79B9_7F4A_7C15_u64.wrapping_mul(step as u64 + 1));
        let mut next_u64 = move || {
            // splitmix64
            state = state.wrapping_add(0x9E37_79B9_7F4A_7C15);
            let mut z = state;
            z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
            z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
            z ^ (z >> 31)
        };
        // u64 -> uniform [0,1) using the high 53 bits (the f64 mantissa width).
        let unit = |u: u64| (u >> 11) as f64 / (1u64 << 53) as f64;
        let mut out = Vec::with_capacity(x.len());
        while out.len() < x.len() {
            // Box–Muller: two uniforms -> two standard normals.
            let u1 = unit(next_u64()).max(1e-12);
            let u2 = unit(next_u64());
            let r = (-2.0 * u1.ln()).sqrt();
            let theta = 2.0 * std::f64::consts::PI * u2;
            out.push((r * theta.cos()) as f32);
            if out.len() < x.len() {
                out.push((r * theta.sin()) as f32);
            }
        }
        Ok(out)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn scale_add_sub_are_elementwise() {
        let ops = CpuLatentOps;
        let a = vec![1.0_f32, 2.0, 3.0];
        let b = vec![0.5_f32, -1.0, 4.0];
        assert_eq!(ops.scale(&a, 2.0).unwrap(), vec![2.0, 4.0, 6.0]);
        assert_eq!(ops.add(&a, &b).unwrap(), vec![1.5, 1.0, 7.0]);
        assert_eq!(ops.sub(&a, &b).unwrap(), vec![0.5, 3.0, -1.0]);
    }

    #[test]
    fn axpy_default_equals_scale_then_add() {
        let ops = CpuLatentOps;
        let x = vec![1.0_f32, -2.0, 3.5];
        let y = vec![0.25_f32, 4.0, -1.0];
        let got = ops.axpy(2.0, &x, -3.0, &y).unwrap();
        let manual = ops
            .add(&ops.scale(&x, 2.0).unwrap(), &ops.scale(&y, -3.0).unwrap())
            .unwrap();
        assert_eq!(got, manual);
        assert_eq!(got, vec![2.0 - 0.75, -4.0 - 12.0, 7.0 + 3.0]);
    }

    #[test]
    fn randn_like_is_deterministic_per_seed_and_step() {
        let ops = CpuLatentOps;
        let x = vec![0.0_f32; 5];
        // Same seed + step => identical draw; different step => different draw.
        assert_eq!(
            ops.randn_like(&x, 42, 0).unwrap(),
            ops.randn_like(&x, 42, 0).unwrap()
        );
        assert_ne!(
            ops.randn_like(&x, 42, 0).unwrap(),
            ops.randn_like(&x, 42, 1).unwrap()
        );
        assert_ne!(
            ops.randn_like(&x, 42, 0).unwrap(),
            ops.randn_like(&x, 43, 0).unwrap()
        );
        assert_eq!(ops.randn_like(&x, 42, 0).unwrap().len(), 5);
    }

    #[test]
    fn randn_like_is_roughly_standard_normal() {
        let ops = CpuLatentOps;
        let x = vec![0.0_f32; 20_000];
        let n = ops.randn_like(&x, 7, 3).unwrap();
        let mean = n.iter().sum::<f32>() / n.len() as f32;
        let var = n.iter().map(|&v| (v - mean).powi(2)).sum::<f32>() / n.len() as f32;
        assert!(mean.abs() < 0.05, "mean {mean}");
        assert!((var.sqrt() - 1.0).abs() < 0.05, "std {}", var.sqrt());
    }
}
