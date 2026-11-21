//! A fixed-geometry video denoise phase. Expert changes occur *inside* its solver
//! trajectory, including at corrector evaluations between the schedule nodes.

use super::RenderPhaseIdentity;
use crate::{sampling::Solver, CancelFlag, Error, MemoryPhase, MemoryRequestScope, Result};

#[derive(Clone, Copy, Debug, PartialEq)]
pub enum VideoModelCoordinate {
    Sigma,
    /// Wan MLX compares the integer-valued model time, including the f32 multiplication.
    TruncatedTimestep {
        num_train_timesteps: usize,
    },
}

impl VideoModelCoordinate {
    pub fn at_sigma(self, sigma: f32) -> f32 {
        match self {
            Self::Sigma => sigma,
            Self::TruncatedTimestep {
                num_train_timesteps,
            } => (sigma * num_train_timesteps as f32).trunc(),
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub enum VideoModelRule<S> {
    Fixed(S),
    AtOrAbove {
        coordinate: VideoModelCoordinate,
        threshold: f32,
        high: S,
        low: S,
    },
}

impl<S: Copy> VideoModelRule<S> {
    pub fn slot(self, sigma: f32) -> S {
        match self {
            Self::Fixed(slot) => slot,
            Self::AtOrAbove {
                coordinate,
                threshold,
                high,
                low,
            } => {
                if coordinate.at_sigma(sigma) >= threshold {
                    high
                } else {
                    low
                }
            }
        }
    }
}

/// A single whole-pass span. One request-owned SolverRunState covers every evaluation,
/// including both sides of an expert crossing. The solver still owns absolute draw indices.
/// The phase uses the same physical identity as RenderRun and MemoryRequestScope. Its
/// separate latent time axis remains explicit for temporally compressed video.
#[derive(Debug)]
pub struct VideoDenoisePhase<S, P = Solver> {
    pub identity: RenderPhaseIdentity,
    pub latent_frames: u32,
    pub solver: P,
    pub sigmas: Vec<f32>,
    pub seed: u64,
    pub models: VideoModelRule<S>,
}

/// Keeps a caller-owned reservation active across preparation, sampling and release.
pub struct ActiveVideoDenoiseScope<'s>(Option<&'s mut dyn MemoryRequestScope>);

impl ActiveVideoDenoiseScope<'_> {
    pub fn finish(self) -> Result<()> {
        if let Some(scope) = self.0 {
            scope.leave_phase(MemoryPhase::Denoise)?;
        }
        Ok(())
    }
}

impl<S, P> VideoDenoisePhase<S, P> {
    /// Scope a complete trajectory under its selected memory rung. As in RenderRun, a
    /// failed phase returns the original error and lets the request owner handle teardown.
    pub fn run<T, E: From<Error>>(
        &self,
        cancel: &CancelFlag,
        scope: Option<&mut dyn MemoryRequestScope>,
        body: impl FnOnce() -> std::result::Result<T, E>,
    ) -> std::result::Result<T, E> {
        let active = self.enter(cancel, scope)?;
        let output = body()?;
        active.finish()?;
        Ok(output)
    }

    pub fn enter<'s>(
        &self,
        cancel: &CancelFlag,
        mut scope: Option<&'s mut dyn MemoryRequestScope>,
    ) -> Result<ActiveVideoDenoiseScope<'s>> {
        if cancel.is_cancelled() {
            return Err(Error::Canceled);
        }
        self.validate()?;
        if let Some(scope) = scope.as_deref_mut() {
            scope.enter_render_phase(MemoryPhase::Denoise, &self.identity)?;
        }
        Ok(ActiveVideoDenoiseScope(scope))
    }

    pub fn validate(&self) -> Result<()> {
        let geometry = self.identity.geometry;
        if geometry.width == 0
            || geometry.height == 0
            || geometry.batch == 0
            || geometry.frames == 0
            || self.latent_frames == 0
            || self.sigmas.len() < 2
            || self.sigmas.iter().any(|s| !s.is_finite() || *s < 0.)
        {
            return Err(Error::Unsupported(
                "invalid video denoise geometry or sigma schedule".into(),
            ));
        }
        if let VideoModelRule::AtOrAbove {
            coordinate,
            threshold,
            ..
        } = self.models
        {
            if !threshold.is_finite()
                || matches!(
                    coordinate,
                    VideoModelCoordinate::TruncatedTimestep {
                        num_train_timesteps: 0
                    }
                )
            {
                return Err(Error::Unsupported("invalid video model threshold".into()));
            }
        }
        Ok(())
    }
}

impl<S> VideoDenoisePhase<S> {
    /// The curated flow solver spans descend to a terminal zero. Other numerical
    /// policies, including an ascending native schedule, own their schedule laws.
    pub fn validate_descending_schedule(&self, first_before_guard: Option<f32>) -> Result<()> {
        self.validate()?;
        // A native log-SNR scheduler can nudge only its first evaluation away from 1.
        // Use the unguarded start for the ordering check, never for integration.
        let initial = first_before_guard.unwrap_or(self.sigmas[0]);
        let tail = &self.sigmas[1..];
        if !initial.is_finite()
            || initial < 0.0
            || self.sigmas.last() != Some(&0.0)
            || tail[0] > initial.max(self.sigmas[0])
            || tail.windows(2).any(|pair| pair[0] < pair[1])
        {
            return Err(Error::Unsupported(
                "curated video sampling needs descending sigma nodes ending in zero".into(),
            ));
        }
        Ok(())
    }
}
