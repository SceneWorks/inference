//! Corrected rectified-flow sampling for Stable Audio 3.
//!
//! The frozen Python implementation is the semantic source for Euler, scalar RK4, RF DPM++,
//! Pingpong, and distribution shifts. Two frozen bugs are deliberately corrected here:
//! per-example RK4 is implemented over the time axis (rather than accidentally zipping the batch
//! axis), and partial-strength schedules warp a normalized `1 -> 0` schedule before scaling every
//! point by the strength. The latter is the ordering already used by the frozen TFLite path and
//! avoids an initial increase in noise for every `0 < strength < 1`.
//!
//! Only the two shipped RF objectives are reachable. k-diffusion and v-diffusion samplers are not
//! exposed by this unregistered component.

use candle_audio::candle_core::{DType, Device, Shape, Tensor};
use candle_audio::{AudioError, Result};
use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};
use rand_distr::StandardNormal;

use crate::config::{
    DiffusionConfig, DiffusionObjective, DistributionShiftConfig, DistributionShiftType,
    StableAudioConfig,
};
use crate::dit::{Guidance, StableAudio3Dit};

macro_rules! bail {
    ($($arg:tt)*) => {
        return Err(AudioError::Msg(format!($($arg)*)))
    };
}

const SAMPLE_RATE: usize = 44_100;
const LATENT_DOWNSAMPLING: usize = 4_096;
const ADAPT_CHUNK_FALLBACK: usize = 32;
const ADAPT_FIRST_STRIDE: usize = 16;
const DEFAULT_DURATION_PADDING: f64 = 6.0;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SamplerKind {
    Euler,
    Rk4,
    Dpmpp,
    Pingpong,
}

impl SamplerKind {
    pub fn recommended(objective: DiffusionObjective) -> Result<Self> {
        match objective {
            DiffusionObjective::RfDenoiser => Ok(Self::Pingpong),
            DiffusionObjective::RectifiedFlow => Ok(Self::Euler),
            DiffusionObjective::V => bail!(
                "Stable Audio 3 v-diffusion is unreachable; only rectified_flow and rf_denoiser are supported"
            ),
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct SamplerResourceEstimate {
    pub model_calls: usize,
    pub full_latent_noise_draws: usize,
    /// One typed schedule materialization per request (zero-copy only on CPU).
    pub schedule_device_elements: usize,
    /// Elements transferred host-to-device when [`SeededNoise`] backs Pingpong.
    ///
    /// This is solver-only accounting after the initial latent already exists. A host-created
    /// initial-noise tensor costs one additional full-latent transfer.
    pub seeded_noise_device_elements: usize,
    /// Total host-created elements transferred to an accelerator.
    pub total_host_to_device_elements: usize,
}

pub fn resource_estimate(
    kind: SamplerKind,
    steps: usize,
    batch: usize,
    latent_elements: usize,
) -> Result<SamplerResourceEstimate> {
    if steps == 0 || batch == 0 {
        bail!("Stable Audio 3 sampling requires steps > 0 and batch > 0")
    }
    let model_calls = match kind {
        SamplerKind::Rk4 => steps.saturating_mul(4),
        _ => steps,
    };
    let full_latent_noise_draws = if kind == SamplerKind::Pingpong {
        steps
    } else {
        0
    };
    let schedule_device_elements = batch.saturating_mul(steps + 1);
    let seeded_noise_device_elements = latent_elements.saturating_mul(full_latent_noise_draws);
    Ok(SamplerResourceEstimate {
        model_calls,
        full_latent_noise_draws,
        schedule_device_elements,
        seeded_noise_device_elements,
        total_host_to_device_elements: schedule_device_elements
            .saturating_add(seeded_noise_device_elements),
    })
}

#[derive(Debug, Clone, PartialEq)]
pub enum DistributionShift {
    Identity,
    Flux {
        min_length: usize,
        max_length: usize,
        alpha_min: f32,
        alpha_max: f32,
    },
    Full {
        min_length: usize,
        max_length: usize,
        base_shift: f32,
        max_shift: f32,
        use_sine: bool,
    },
    LogSnr {
        anchor_length: usize,
        anchor_logsnr: f32,
        rate: f32,
        logsnr_end: f32,
    },
}

impl From<&DistributionShiftConfig> for DistributionShift {
    fn from(value: &DistributionShiftConfig) -> Self {
        match value.kind {
            DistributionShiftType::None => Self::Identity,
            DistributionShiftType::Flux => Self::Flux {
                min_length: value.min_length,
                max_length: value.max_length,
                alpha_min: value.alpha_min as f32,
                alpha_max: value.alpha_max as f32,
            },
            DistributionShiftType::Full => Self::Full {
                min_length: value.min_length,
                max_length: value.max_length,
                base_shift: value.base_shift as f32,
                max_shift: value.max_shift as f32,
                use_sine: value.use_sine,
            },
            DistributionShiftType::Logsnr => Self::LogSnr {
                anchor_length: value.anchor_length,
                anchor_logsnr: value.anchor_logsnr as f32,
                rate: value.rate as f32,
                logsnr_end: value.logsnr_end as f32,
            },
        }
    }
}

impl DistributionShift {
    fn validate(&self) -> Result<()> {
        let positive_range = |min: usize, max: usize| {
            if min == 0 || max <= min {
                Err(AudioError::Msg(format!(
                    "distribution shift length range must satisfy 0 < min < max, got {min}..{max}"
                )))
            } else {
                Ok(())
            }
        };
        match self {
            Self::Identity => Ok(()),
            Self::Flux {
                min_length,
                max_length,
                alpha_min,
                alpha_max,
            } => {
                positive_range(*min_length, *max_length)?;
                if !alpha_min.is_finite()
                    || !alpha_max.is_finite()
                    || *alpha_min <= 0.0
                    || *alpha_max <= 0.0
                {
                    bail!("Flux alpha bounds must be finite and positive")
                }
                Ok(())
            }
            Self::Full {
                min_length,
                max_length,
                base_shift,
                max_shift,
                ..
            } => {
                positive_range(*min_length, *max_length)?;
                if !base_shift.is_finite() || !max_shift.is_finite() {
                    bail!("Full shift bounds must be finite")
                }
                Ok(())
            }
            Self::LogSnr {
                anchor_length,
                anchor_logsnr,
                rate,
                logsnr_end,
            } => {
                if *anchor_length == 0
                    || !anchor_logsnr.is_finite()
                    || !rate.is_finite()
                    || !logsnr_end.is_finite()
                {
                    bail!(
                        "LogSNR shift parameters must be finite and anchor_length must be positive"
                    )
                }
                Ok(())
            }
        }
    }

    fn apply(&self, t: f32, sequence_length: usize) -> f32 {
        if t <= 0.0 {
            return 0.0;
        }
        if t >= 1.0 {
            return 1.0;
        }
        match *self {
            Self::Identity => t,
            Self::Flux {
                min_length,
                max_length,
                alpha_min,
                alpha_max,
            } => {
                let length = sequence_length.clamp(min_length, max_length) as f32;
                let fraction = (length.ln() - (min_length as f32).ln())
                    / ((max_length as f32).ln() - (min_length as f32).ln());
                let alpha = (alpha_min.ln() + fraction * (alpha_max.ln() - alpha_min.ln())).exp();
                alpha * t / (1.0 + (alpha - 1.0) * t)
            }
            Self::Full {
                min_length,
                max_length,
                base_shift,
                max_shift,
                use_sine,
            } => {
                let length = sequence_length.clamp(min_length, max_length) as f32;
                let fraction = (length - min_length as f32) / (max_length - min_length) as f32;
                let mu = -(base_shift + (max_shift - base_shift) * fraction);
                let exponent = mu.exp();
                let shifted = 1.0 - exponent / (exponent + (1.0 / (1.0 - t) - 1.0));
                if use_sine {
                    (shifted * std::f32::consts::FRAC_PI_2).sin()
                } else {
                    shifted
                }
            }
            Self::LogSnr {
                anchor_length,
                anchor_logsnr,
                rate,
                logsnr_end,
            } => {
                let start = anchor_logsnr
                    - rate * ((sequence_length.max(1) as f32 / anchor_length as f32).log2());
                let logsnr = logsnr_end - t * (logsnr_end - start);
                1.0 / (1.0 + logsnr.exp())
            }
        }
    }
}

/// Shape-safe schedule. Shared schedules can never be confused with per-example schedules even
/// when `batch == steps + 1`.
#[derive(Debug, Clone, PartialEq)]
pub enum Schedule {
    Shared(Vec<f32>),
    PerExample(Vec<Vec<f32>>),
}

impl Schedule {
    pub fn steps(&self) -> usize {
        match self {
            Self::Shared(values) => values.len().saturating_sub(1),
            Self::PerExample(values) => values
                .first()
                .map_or(0, |values| values.len().saturating_sub(1)),
        }
    }

    pub fn batch(&self) -> Option<usize> {
        match self {
            Self::Shared(_) => None,
            Self::PerExample(values) => Some(values.len()),
        }
    }

    fn validate_for(&self, batch: usize) -> Result<()> {
        let steps = self.steps();
        if steps == 0 {
            bail!("Stable Audio 3 sampling requires at least one step")
        }
        match self {
            Self::Shared(values) => validate_schedule_row(values),
            Self::PerExample(rows) => {
                if rows.len() != batch {
                    bail!(
                        "per-example schedule batch {} does not match latent batch {batch}",
                        rows.len()
                    )
                }
                for row in rows {
                    if row.len() != steps + 1 {
                        bail!("per-example schedules must have equal lengths")
                    }
                    validate_schedule_row(row)?;
                }
                Ok(())
            }
        }
    }

    fn validate_for_sampling(&self, batch: usize) -> Result<()> {
        self.validate_for(batch)?;
        let strictly_decreasing = |values: &[f32]| {
            values
                .windows(2)
                .all(|pair| pair[0] > 0.0 && pair[1] < pair[0])
        };
        let valid = match self {
            Self::Shared(values) => strictly_decreasing(values),
            Self::PerExample(rows) => rows.iter().all(|values| strictly_decreasing(values)),
        };
        if !valid {
            bail!("solver schedules must be strictly decreasing to terminal zero; strength=0 must use the no-call init path")
        }
        Ok(())
    }

    fn values_at(&self, index: usize, batch: usize) -> Vec<f32> {
        match self {
            Self::Shared(values) => vec![values[index]; batch],
            Self::PerExample(rows) => rows.iter().map(|row| row[index]).collect(),
        }
    }

    fn materialize(&self, batch: usize, device: &Device) -> Result<Tensor> {
        let values = match self {
            Self::Shared(row) => (0..batch)
                .flat_map(|_| row.iter().copied())
                .collect::<Vec<_>>(),
            Self::PerExample(rows) => rows.iter().flatten().copied().collect(),
        };
        Ok(Tensor::from_vec(values, (batch, self.steps() + 1), device)?)
    }
}

fn validate_schedule_row(values: &[f32]) -> Result<()> {
    if values.len() < 2 || values.iter().any(|value| !value.is_finite()) {
        bail!("schedule must contain at least two finite values")
    }
    if values.iter().any(|value| *value < 0.0 || *value > 1.0) {
        bail!("schedule values must be within [0,1]")
    }
    if values.windows(2).any(|pair| pair[1] > pair[0]) {
        bail!("schedule must be monotonically non-increasing")
    }
    if values.last().copied() != Some(0.0) {
        bail!("rectified-flow schedule must terminate at zero")
    }
    Ok(())
}

/// Build the corrected schedule. `strength` is the init-audio noise level and is constrained to
/// the documented public `[0,1]` domain. The normalized schedule is shifted before the entire
/// trajectory is scaled, which preserves monotonicity for partial strength.
pub fn build_schedule(
    steps: usize,
    strength: f32,
    shift: &DistributionShift,
    effective_lengths: Option<&[usize]>,
    fallback_length: usize,
) -> Result<Schedule> {
    if steps == 0 {
        bail!("Stable Audio 3 sampling requires steps > 0")
    }
    if !strength.is_finite() || !(0.0..=1.0).contains(&strength) {
        bail!("init noise strength must be finite and within [0,1]")
    }
    shift.validate()?;
    let normalized = (0..=steps)
        .map(|index| 1.0f32 - index as f32 / steps as f32)
        .collect::<Vec<_>>();
    let row = |length: usize| {
        let mut values = normalized
            .iter()
            .map(|&t| shift.apply(t, length.max(1)) * strength)
            .collect::<Vec<_>>();
        values[0] = strength;
        values[steps] = 0.0;
        values
    };
    let schedule = match effective_lengths {
        Some(lengths) => {
            if lengths.is_empty() {
                bail!("per-example effective lengths cannot be empty")
            }
            Schedule::PerExample(lengths.iter().map(|&length| row(length)).collect())
        }
        None => Schedule::Shared(row(fallback_length)),
    };
    // `strength=0` is a deliberate no-DiT short circuit and is accepted here even though it has
    // no decreasing interval.
    if strength > 0.0 {
        match &schedule {
            Schedule::Shared(values) => validate_schedule_row(values)?,
            Schedule::PerExample(rows) => {
                for values in rows {
                    validate_schedule_row(values)?;
                }
            }
        }
    }
    Ok(schedule)
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SampleGeometry {
    pub sample_size: usize,
    pub latent_length: usize,
    pub effective_lengths: Option<Vec<usize>>,
    pub valid_lengths: Vec<usize>,
}

fn ceil_to(value: usize, multiple: usize) -> usize {
    value.saturating_add(multiple - 1) / multiple * multiple
}

/// Exact frozen `_adapt_sample_size` geometry with its local `chunk_size=32` helper fallback.
pub fn adapt_sample_size(
    config: &StableAudioConfig,
    durations: &[Option<f64>],
    duration_padding_seconds: f64,
) -> Result<SampleGeometry> {
    adapt_sample_size_for_max(config.sample_size, durations, duration_padding_seconds)
}

/// Geometry helper with an explicit selected-config maximum. This is public so callers and tests
/// can prove all six snapshot maxima without constructing or mutating a model configuration.
pub fn adapt_sample_size_for_max(
    max_sample_size: usize,
    durations: &[Option<f64>],
    duration_padding_seconds: f64,
) -> Result<SampleGeometry> {
    if durations.is_empty() {
        bail!("at least one duration entry is required")
    }
    if !duration_padding_seconds.is_finite() || duration_padding_seconds < 0.0 {
        bail!("duration padding must be finite and non-negative")
    }
    for duration in durations.iter().flatten() {
        if !duration.is_finite() || *duration < 0.0 {
            bail!("durations must be finite and non-negative")
        }
    }
    let all_present = durations.iter().all(Option::is_some);
    // Tensor adaptation uses the maximum *present* duration. Missing duration only forces the
    // effective-length schedule to fall back globally.
    let max_duration = durations.iter().flatten().copied().fold(0.0f64, f64::max);
    let alignment = LATENT_DOWNSAMPLING * (ADAPT_CHUNK_FALLBACK / ADAPT_FIRST_STRIDE);
    let sample_size = if max_duration <= 0.0 {
        max_sample_size
    } else {
        let requested = ((max_duration + duration_padding_seconds) * SAMPLE_RATE as f64) as usize;
        ceil_to(ceil_to(requested, LATENT_DOWNSAMPLING), alignment).min(max_sample_size)
    };
    let latent_length = sample_size / LATENT_DOWNSAMPLING;
    let effective_lengths = all_present.then(|| {
        durations
            .iter()
            .map(|duration| {
                let samples = (duration.unwrap_or_default() * SAMPLE_RATE as f64) as usize;
                ceil_to(samples, LATENT_DOWNSAMPLING) / LATENT_DOWNSAMPLING
            })
            .collect::<Vec<_>>()
    });
    let headroom =
        (duration_padding_seconds * SAMPLE_RATE as f64 / LATENT_DOWNSAMPLING as f64) as usize;
    let valid_lengths = effective_lengths
        .as_ref()
        .map(|values| {
            values
                .iter()
                .map(|value| value.saturating_add(headroom).min(latent_length))
                .collect()
        })
        .unwrap_or_else(|| vec![latent_length; durations.len()]);
    Ok(SampleGeometry {
        sample_size,
        latent_length,
        effective_lengths,
        valid_lengths,
    })
}

pub fn default_sample_geometry(
    config: &StableAudioConfig,
    durations: &[Option<f64>],
) -> Result<SampleGeometry> {
    adapt_sample_size(config, durations, DEFAULT_DURATION_PADDING)
}

/// Dense boolean valid-position mask forwarded to DiT attention. The DiT owns the frozen
/// V-zero padding boundary; the sampler deliberately does not duplicate it.
pub fn padding_mask(
    valid_lengths: &[usize],
    latent_length: usize,
    device: &Device,
) -> Result<Tensor> {
    let mut values = Vec::with_capacity(valid_lengths.len() * latent_length);
    for &valid in valid_lengths {
        if valid > latent_length {
            bail!("valid length {valid} exceeds latent length {latent_length}")
        }
        values.extend((0..latent_length).map(|index| if index < valid { 1u8 } else { 0u8 }));
    }
    Ok(
        Tensor::from_vec(values, (valid_lengths.len(), latent_length), device)?
            .to_dtype(DType::U8)?,
    )
}

pub trait NoiseSource {
    fn standard_normal_like(&mut self, tensor: &Tensor) -> Result<Tensor>;
    fn draws(&self) -> usize;
}

/// Request-local deterministic RNG. Noise is generated on the host and then transferred, so
/// concurrent requests cannot perturb one another. Backend RNG algorithms are intentionally not
/// claimed to be byte-identical.
pub struct SeededNoise {
    rng: StdRng,
    draws: usize,
}

impl SeededNoise {
    pub fn new(seed: u64) -> Self {
        Self {
            rng: StdRng::seed_from_u64(seed),
            draws: 0,
        }
    }
}

impl NoiseSource for SeededNoise {
    fn standard_normal_like(&mut self, tensor: &Tensor) -> Result<Tensor> {
        let count = tensor.elem_count();
        let values = (0..count)
            .map(|_| self.rng.sample::<f32, _>(StandardNormal))
            .collect::<Vec<_>>();
        self.draws += 1;
        Ok(
            Tensor::from_vec(values, Shape::from_dims(tensor.dims()), tensor.device())?
                .to_dtype(tensor.dtype())?,
        )
    }

    fn draws(&self) -> usize {
        self.draws
    }
}

/// Explicit oracle noise. Pingpong consumes one full latent draw per step, including the terminal
/// zero-scaled draw.
pub struct InjectedNoise {
    draws: std::vec::IntoIter<Tensor>,
    consumed: usize,
}

impl InjectedNoise {
    pub fn new(draws: Vec<Tensor>) -> Self {
        Self {
            draws: draws.into_iter(),
            consumed: 0,
        }
    }
}

impl NoiseSource for InjectedNoise {
    fn standard_normal_like(&mut self, tensor: &Tensor) -> Result<Tensor> {
        let value = self
            .draws
            .next()
            .ok_or_else(|| AudioError::Msg("injected Pingpong noise exhausted".into()))?;
        if value.dims() != tensor.dims() {
            bail!("injected Pingpong noise shape does not match latent")
        }
        self.consumed += 1;
        Ok(value.to_device(tensor.device())?.to_dtype(tensor.dtype())?)
    }

    fn draws(&self) -> usize {
        self.consumed
    }
}

#[derive(Debug, Clone)]
pub struct SampleStep {
    pub index: usize,
    pub x: Tensor,
    pub timestep: Tensor,
    pub denoised: Tensor,
}

#[derive(Debug, Clone)]
pub struct SampleOutput {
    pub latents: Tensor,
    pub trajectory: Vec<SampleStep>,
    pub model_calls: usize,
    pub noise_draws: usize,
}

/// Live pre-update progress hook. Returning an error cancels sampling.
pub type ProgressCallback<'a> = dyn FnMut(&SampleStep) -> Result<()> + 'a;

fn expand(values: &Tensor) -> Result<Tensor> {
    Ok(values.unsqueeze(1)?.unsqueeze(2)?)
}

fn denoised(x: &Tensor, timestep: &Tensor, velocity: &Tensor) -> Result<Tensor> {
    Ok(x.broadcast_sub(&velocity.broadcast_mul(&expand(timestep)?)?)?)
}

/// Run one corrected RF solver. The closure is the only model seam, allowing callers to reuse
/// [`StableAudio3Dit::forward_guided`] without duplicating guidance math.
#[allow(clippy::too_many_arguments)]
pub fn sample<F, N>(
    kind: SamplerKind,
    initial: &Tensor,
    schedule: &Schedule,
    padding_mask: Option<&Tensor>,
    noise: &mut N,
    record_trajectory: bool,
    model: F,
) -> Result<SampleOutput>
where
    F: FnMut(&Tensor, &Tensor) -> Result<Tensor>,
    N: NoiseSource,
{
    sample_with_callback(
        kind,
        initial,
        schedule,
        padding_mask,
        noise,
        record_trajectory,
        None,
        model,
    )
}

/// Sampling with a live pre-update progress/cancellation callback. The callback observes exactly
/// the frozen callback state (`x`, current timestep, denoised estimate) and runs before the solver
/// update or Pingpong draw. Returning an error cancels immediately and propagates that error.
#[allow(clippy::too_many_arguments)]
pub fn sample_with_callback<F, N>(
    kind: SamplerKind,
    initial: &Tensor,
    schedule: &Schedule,
    padding_mask: Option<&Tensor>,
    noise: &mut N,
    record_trajectory: bool,
    callback: Option<&mut ProgressCallback<'_>>,
    mut model: F,
) -> Result<SampleOutput>
where
    F: FnMut(&Tensor, &Tensor) -> Result<Tensor>,
    N: NoiseSource,
{
    sample_with_host_timestep(
        kind,
        initial,
        schedule,
        padding_mask,
        noise,
        record_trajectory,
        callback,
        |x, timestep, _host_timestep| model(x, timestep),
    )
}

/// Advanced sampler seam retaining typed host timestep values for control decisions without
/// device-to-host synchronization.
#[allow(clippy::too_many_arguments)]
pub fn sample_with_host_timestep<F, N>(
    kind: SamplerKind,
    initial: &Tensor,
    schedule: &Schedule,
    padding_mask: Option<&Tensor>,
    noise: &mut N,
    record_trajectory: bool,
    mut callback: Option<&mut ProgressCallback<'_>>,
    mut model: F,
) -> Result<SampleOutput>
where
    F: FnMut(&Tensor, &Tensor, &[f32]) -> Result<Tensor>,
    N: NoiseSource,
{
    let (batch, _, _) = initial.dims3()?;
    schedule.validate_for_sampling(batch)?;
    if let Some(mask) = padding_mask {
        if mask.dims2()? != (batch, initial.dim(2)?) {
            bail!("padding mask must be [batch,time]")
        }
    }
    let mut x = initial.clone();
    let mut trajectory = Vec::with_capacity(if record_trajectory {
        schedule.steps()
    } else {
        0
    });
    let mut calls = 0usize;
    let mut old_denoised: Option<Tensor> = None;
    let device_schedule = schedule.materialize(batch, initial.device())?;

    for index in 0..schedule.steps() {
        let t_values = schedule.values_at(index, batch);
        let next_values = schedule.values_at(index + 1, batch);
        let t = device_schedule.narrow(1, index, 1)?.squeeze(1)?;
        let next = device_schedule.narrow(1, index + 1, 1)?.squeeze(1)?;
        let dt = (&next - &t)?;
        match kind {
            SamplerKind::Euler => {
                let velocity = model(&x, &t, &t_values)?;
                calls += 1;
                let clean = denoised(&x, &t, &velocity)?;
                let progress = SampleStep {
                    index,
                    x: x.clone(),
                    timestep: t.clone(),
                    denoised: clean,
                };
                if let Some(callback) = callback.as_deref_mut() {
                    callback(&progress)?;
                }
                if record_trajectory {
                    trajectory.push(progress);
                }
                x = (&x + velocity.broadcast_mul(&expand(&dt)?)?)?;
            }
            SamplerKind::Rk4 => {
                let k1 = model(&x, &t, &t_values)?;
                calls += 1;
                let clean = denoised(&x, &t, &k1)?;
                let progress = SampleStep {
                    index,
                    x: x.clone(),
                    timestep: t.clone(),
                    denoised: clean,
                };
                if let Some(callback) = callback.as_deref_mut() {
                    callback(&progress)?;
                }
                if record_trajectory {
                    trajectory.push(progress);
                }
                let half_dt = (&dt / 2f64)?;
                let midpoint = (&t + &half_dt)?;
                let midpoint_values = t_values
                    .iter()
                    .zip(&next_values)
                    .map(|(&current, &next)| current + (next - current) / 2.0)
                    .collect::<Vec<_>>();
                let k2_x = (&x + k1.broadcast_mul(&expand(&half_dt)?)?)?;
                let k2 = model(&k2_x, &midpoint, &midpoint_values)?;
                let k3_x = (&x + k2.broadcast_mul(&expand(&half_dt)?)?)?;
                let k3 = model(&k3_x, &midpoint, &midpoint_values)?;
                let terminal_eval = next.clamp(1e-5, f64::INFINITY)?;
                let terminal_values = next_values
                    .iter()
                    .map(|&value| value.max(1e-5))
                    .collect::<Vec<_>>();
                let k4_x = (&x + k3.broadcast_mul(&expand(&dt)?)?)?;
                let k4 = model(&k4_x, &terminal_eval, &terminal_values)?;
                calls += 3;
                let weighted = ((&k2 * 2f64)? + (&k3 * 2f64)?)?;
                let weighted = ((&weighted + &k1)? + &k4)?;
                x = (&x + weighted.broadcast_mul(&expand(&(&dt / 6f64)?)?)?)?;
            }
            SamplerKind::Dpmpp => {
                let velocity = model(&x, &t, &t_values)?;
                calls += 1;
                let clean = denoised(&x, &t, &velocity)?;
                let progress = SampleStep {
                    index,
                    x: x.clone(),
                    timestep: t.clone(),
                    denoised: clean.clone(),
                };
                if let Some(callback) = callback.as_deref_mut() {
                    callback(&progress)?;
                }
                if record_trajectory {
                    trajectory.push(progress);
                }
                let terminal = index + 1 == schedule.steps();
                let first = old_denoised.is_none();
                let t_broadcast = expand(&t)?;
                let next_broadcast = expand(&next)?;
                let dt_broadcast = expand(&dt)?;
                let alpha = (1f64 - &next_broadcast)?;
                let coefficient = dt_broadcast.broadcast_div(
                    &alpha
                        .clamp(1e-10, f64::INFINITY)?
                        .broadcast_mul(&t_broadcast.clamp(1e-10, f64::INFINITY)?)?,
                )?;
                let corrected = if first || terminal {
                    clean.clone()
                } else {
                    let previous = device_schedule.narrow(1, index - 1, 1)?.squeeze(1)?;
                    let logsnr = |value: &Tensor| -> Result<Tensor> {
                        Ok((1f64 - value)?
                            .clamp(1e-10, f64::INFINITY)?
                            .broadcast_div(&value.clamp(1e-10, f64::INFINITY)?)?
                            .log()?)
                    };
                    let h = (&logsnr(&next)? - &logsnr(&t)?)?;
                    let h_last = (&logsnr(&t)? - &logsnr(&previous)?)?;
                    let ratio = h_last.broadcast_div(&h)?;
                    let inverse_twice_ratio = (ratio * 2f64)?.recip()?;
                    clean
                        .broadcast_mul(&(1f64 + &expand(&inverse_twice_ratio)?)?)?
                        .broadcast_sub(
                            &old_denoised
                                .as_ref()
                                .expect("non-first DPM++ step has history")
                                .broadcast_mul(&expand(&inverse_twice_ratio)?)?,
                        )?
                };
                x = next_broadcast
                    .broadcast_div(&t_broadcast.clamp(1e-10, f64::INFINITY)?)?
                    .broadcast_mul(&x)?
                    .broadcast_sub(
                        &alpha
                            .broadcast_mul(&coefficient)?
                            .broadcast_mul(&corrected)?,
                    )?;
                old_denoised = Some(clean);
            }
            SamplerKind::Pingpong => {
                let velocity = model(&x, &t, &t_values)?;
                calls += 1;
                let clean = denoised(&x, &t, &velocity)?;
                let progress = SampleStep {
                    index,
                    x: x.clone(),
                    timestep: t.clone(),
                    denoised: clean.clone(),
                };
                if let Some(callback) = callback.as_deref_mut() {
                    callback(&progress)?;
                }
                if record_trajectory {
                    trajectory.push(progress);
                }
                // Deliberately eager: frozen Python draws even when next==0.
                let fresh = noise.standard_normal_like(&x)?;
                let next_broadcast = expand(&next)?;
                x = clean
                    .broadcast_mul(&(1f64 - &next_broadcast)?)?
                    .broadcast_add(&fresh.broadcast_mul(&next_broadcast)?)?;
            }
        }
    }
    Ok(SampleOutput {
        latents: x,
        trajectory,
        model_calls: calls,
        noise_draws: noise.draws(),
    })
}

/// Initialize and sample in one operation so `strength=0` has enforceable no-DiT semantics rather
/// than relying on a caller convention.
#[allow(clippy::too_many_arguments)]
pub fn sample_initialized<F, N>(
    kind: SamplerKind,
    noise_latents: &Tensor,
    init_latents: Option<&Tensor>,
    strength: f32,
    schedule: &Schedule,
    padding_mask: Option<&Tensor>,
    noise: &mut N,
    record_trajectory: bool,
    model: F,
) -> Result<SampleOutput>
where
    F: FnMut(&Tensor, &Tensor) -> Result<Tensor>,
    N: NoiseSource,
{
    let batch = noise_latents.dim(0)?;
    schedule.validate_for(batch)?;
    if schedule
        .values_at(0, batch)
        .iter()
        .any(|&first| first != strength)
    {
        bail!("init noise strength must equal every schedule's first sigma")
    }
    let initial = initialize_latents(noise_latents, init_latents, strength)?;
    if init_latents.is_some() && strength == 0.0 {
        return Ok(SampleOutput {
            latents: initial,
            trajectory: Vec::new(),
            model_calls: 0,
            noise_draws: noise.draws(),
        });
    }
    sample(
        kind,
        &initial,
        schedule,
        padding_mask,
        noise,
        record_trajectory,
        model,
    )
}

/// Mix init latents and initial noise. With strength zero the caller must return init latents
/// directly without constructing a schedule or invoking the DiT.
pub fn initialize_latents(noise: &Tensor, init: Option<&Tensor>, strength: f32) -> Result<Tensor> {
    if !strength.is_finite() || !(0.0..=1.0).contains(&strength) {
        bail!("init noise strength must be finite and within [0,1]")
    }
    match init {
        Some(init) => {
            if init.dims() != noise.dims() {
                bail!("init latents and noise must have identical shapes")
            }
            if init.dtype() != noise.dtype() || !init.device().same_device(noise.device()) {
                bail!("init latents and noise must have identical dtype and device")
            }
            Ok(
                ((init * (1.0 - strength) as f64)? + (noise * strength as f64)?)?
                    .to_dtype(noise.dtype())?,
            )
        }
        None if strength == 1.0 => Ok(noise.clone()),
        None => bail!("partial init noise strength requires init latents"),
    }
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct GuidanceInterval {
    pub min_sigma: f32,
    pub max_sigma: f32,
}

impl GuidanceInterval {
    pub const FULL: Self = Self {
        min_sigma: 0.0,
        max_sigma: 1.0,
    };

    fn validate(self) -> Result<()> {
        if !self.min_sigma.is_finite()
            || !self.max_sigma.is_finite()
            || self.min_sigma < 0.0
            || self.max_sigma > 1.0
            || self.min_sigma > self.max_sigma
        {
            bail!("CFG interval must be finite and satisfy 0 <= min <= max <= 1")
        }
        Ok(())
    }

    pub fn guidance_for_values(self, values: &[f32], configured: Guidance) -> Result<Guidance> {
        self.validate()?;
        if self == Self::FULL {
            return Ok(configured);
        }
        let mut decisions = values
            .iter()
            .map(|&sigma| self.min_sigma <= sigma && sigma <= self.max_sigma);
        let first = decisions.next().ok_or_else(|| {
            AudioError::Msg("CFG interval received an empty timestep batch".into())
        })?;
        if decisions.any(|decision| decision != first) {
            bail!(
                "heterogeneous per-example CFG interval decisions are unsupported by mandatory one-pass 2B CFG"
            )
        }
        Ok(if first {
            configured
        } else {
            Guidance {
                cfg_scale: 1.0,
                ..configured
            }
        })
    }
}

pub fn validate_guidance(guidance: Guidance) -> Result<()> {
    if !guidance.cfg_scale.is_finite() || guidance.cfg_scale < 0.0 {
        bail!("CFG scale must be finite and non-negative")
    }
    if !guidance.apg_scale.is_finite() || !(0.0..=1.0).contains(&guidance.apg_scale) {
        bail!("APG scale must be finite and within [0,1]")
    }
    if !guidance.scale_phi.is_finite() || !(0.0..=1.0).contains(&guidance.scale_phi) {
        bail!("CFG scale_phi must be finite and within [0,1]")
    }
    if !guidance.cfg_norm_threshold.is_finite() || guidance.cfg_norm_threshold < 0.0 {
        bail!("CFG norm threshold must be finite and non-negative")
    }
    Ok(())
}

/// Convenience seam binding the sampler to the already-audited DiT guidance implementation.
#[allow(clippy::too_many_arguments)]
pub fn sample_dit<N: NoiseSource>(
    model: &StableAudio3Dit,
    kind: SamplerKind,
    initial: &Tensor,
    schedule: &Schedule,
    positive_prompt: &Tensor,
    negative_prompt: Option<&Tensor>,
    negative_prompt_mask: Option<&Tensor>,
    seconds_total: &Tensor,
    local_conditioning: &Tensor,
    padding: Option<&Tensor>,
    guidance: Guidance,
    noise: &mut N,
    record_trajectory: bool,
) -> Result<SampleOutput> {
    sample_dit_with_interval(
        model,
        kind,
        initial,
        schedule,
        positive_prompt,
        negative_prompt,
        negative_prompt_mask,
        seconds_total,
        local_conditioning,
        padding,
        guidance,
        GuidanceInterval::FULL,
        None,
        noise,
        record_trajectory,
    )
}

/// DiT sampler with explicit CFG interval gating. Nondefault intervals are evaluated for every
/// solver stage, including RK4 sub-stages. A heterogeneous batch decision fails closed because
/// splitting it would violate Stable Audio 3's mandatory single 2B CFG forward.
#[allow(clippy::too_many_arguments)]
pub fn sample_dit_with_interval<N: NoiseSource>(
    model: &StableAudio3Dit,
    kind: SamplerKind,
    initial: &Tensor,
    schedule: &Schedule,
    positive_prompt: &Tensor,
    negative_prompt: Option<&Tensor>,
    negative_prompt_mask: Option<&Tensor>,
    seconds_total: &Tensor,
    local_conditioning: &Tensor,
    padding: Option<&Tensor>,
    guidance: Guidance,
    guidance_interval: GuidanceInterval,
    callback: Option<&mut ProgressCallback<'_>>,
    noise: &mut N,
    record_trajectory: bool,
) -> Result<SampleOutput> {
    guidance_interval.validate()?;
    validate_guidance(guidance)?;
    if model.objective() == DiffusionObjective::V {
        bail!("Stable Audio 3 RF samplers reject the unreachable v-diffusion objective")
    }
    sample_with_host_timestep(
        kind,
        initial,
        schedule,
        padding,
        noise,
        record_trajectory,
        callback,
        |latents, timestep, host_timestep| {
            let step_guidance = guidance_interval.guidance_for_values(host_timestep, guidance)?;
            Ok(model.forward_guided(
                latents,
                timestep,
                positive_prompt,
                negative_prompt,
                negative_prompt_mask,
                seconds_total,
                local_conditioning,
                padding,
                step_guidance,
            )?)
        },
    )
}

/// Effective schedule lengths use the raw duration only. Attention headroom and sample adaptation
/// are deliberately separate.
pub fn effective_schedule_lengths(durations: &[Option<f64>]) -> Result<Option<Vec<usize>>> {
    if durations.iter().any(Option::is_none) {
        return Ok(None);
    }
    durations
        .iter()
        .map(|duration| {
            let duration = duration.unwrap_or_default();
            if !duration.is_finite() || duration < 0.0 {
                bail!("durations must be finite and non-negative")
            }
            let samples = (duration * SAMPLE_RATE as f64) as usize;
            Ok((ceil_to(samples, LATENT_DOWNSAMPLING) / LATENT_DOWNSAMPLING).max(1))
        })
        .collect::<Result<Vec<_>>>()
        .map(Some)
}

pub fn inference_shift(config: &DiffusionConfig) -> DistributionShift {
    DistributionShift::from(&config.effective_sampling_shift())
}
