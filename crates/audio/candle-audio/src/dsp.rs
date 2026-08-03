//! Host-side DSP primitives for the candle audio providers: whole-buffer rational
//! sample-rate conversion (sc-14561), Hann windowing, forward STFT, and inverse-STFT
//! overlap-add reconstruction (sc-12835/sc-12836).
//!
//! Everything here is plain `f32` DSP with no tensor dependency: a provider's model
//! produces magnitude/phase (or mel) tensors on whatever candle device the bundle
//! selected, converts them to host `f32`, and reconstructs audio here — keeping the
//! numerics identical across CPU/Metal/CUDA and unit-testable without weights.
//!
//! Conventions (the librosa/torch `center=True` defaults the reference audio models
//! assume): frames are `n_fft`-long, spaced `hop` apart, over an input reflect-padded by
//! `n_fft / 2` on both ends; spectra carry the `n_fft / 2 + 1` one-sided bins; the
//! inverse normalizes by the summed squared window and trims the centering pad.

use crate::{AudioError, Result};
use std::ops::Range;

#[cfg(any(test, feature = "testkit"))]
use std::sync::atomic::{AtomicUsize, Ordering};

/// A periodic Hann window of length `len` — `0.5 * (1 - cos(2π n / N))`, the analysis and
/// synthesis window the reference STFT stacks (torch.hann_window / librosa) default to.
pub fn hann_window(len: usize) -> Vec<f32> {
    let n = len as f32;
    (0..len)
        .map(|i| 0.5 * (1.0 - (2.0 * std::f32::consts::PI * i as f32 / n).cos()))
        .collect()
}

// An odd tap count makes the zero phase exactly sample-centered. With β=8.6, 197
// unity-rate taps span the 6% guarded transition to approximately the window's
// 85 dB stopband. Stronger downsampling scales this by src/dst so the kernel
// always spans the same number of cutoff-frequency lobes.
const RESAMPLE_BASE_TAPS_PER_PHASE: usize = 197;
const RESAMPLE_MAX_TAPS_PER_PHASE: usize = 16_385;
const RESAMPLE_KAISER_BETA: f64 = 8.6;
const RESAMPLE_CUTOFF_GUARD: f64 = 0.94;
const RESAMPLE_MAX_PRECOMPUTED_COEFFICIENTS: usize = 4_194_304;

#[cfg(any(test, feature = "testkit"))]
static RESAMPLE_OUTPUT_WORK: AtomicUsize = AtomicUsize::new(0);
#[cfg(any(test, feature = "testkit"))]
static RESAMPLE_SOURCE_FRAME_WORK: AtomicUsize = AtomicUsize::new(0);

/// Test-only observability for proving that bounded callers do not process a complete clip.
#[cfg(any(test, feature = "testkit"))]
pub mod resample_test_support {
    use super::{Ordering, RESAMPLE_OUTPUT_WORK, RESAMPLE_SOURCE_FRAME_WORK};

    pub fn reset() {
        RESAMPLE_OUTPUT_WORK.store(0, Ordering::Relaxed);
        RESAMPLE_SOURCE_FRAME_WORK.store(0, Ordering::Relaxed);
    }

    pub fn work() -> (usize, usize) {
        (
            RESAMPLE_OUTPUT_WORK.load(Ordering::Relaxed),
            RESAMPLE_SOURCE_FRAME_WORK.load(Ordering::Relaxed),
        )
    }
}

fn gcd(mut a: u32, mut b: u32) -> u32 {
    while b != 0 {
        (a, b) = (b, a % b);
    }
    a
}

fn bessel_i0(x: f64) -> f64 {
    let y = x * x / 4.0;
    let mut sum = 1.0;
    let mut term = 1.0;
    for k in 1.. {
        term *= y / (k * k) as f64;
        sum += term;
        if term <= sum * f64::EPSILON {
            break;
        }
    }
    sum
}

fn normalized_sinc(x: f64) -> f64 {
    if x.abs() < 1e-12 {
        1.0
    } else {
        let px = std::f64::consts::PI * x;
        px.sin() / px
    }
}

fn fill_phase_kernel(
    kernel: &mut [f64],
    phase: usize,
    phase_count: usize,
    cutoff: f64,
    kaiser_denominator: f64,
) -> f64 {
    let taps_per_phase = kernel.len();
    let fraction = phase as f64 / phase_count as f64;
    let half_taps = (taps_per_phase / 2) as isize;
    let radius = taps_per_phase as f64 / 2.0;
    let mut sum = 0.0;
    for (tap, weight) in kernel.iter_mut().enumerate() {
        let offset = tap as isize - half_taps;
        let distance = offset as f64 - fraction;
        let window_position = (distance / radius).clamp(-1.0, 1.0);
        let window =
            bessel_i0(RESAMPLE_KAISER_BETA * (1.0 - window_position * window_position).sqrt())
                / kaiser_denominator;
        *weight = cutoff * normalized_sinc(cutoff * distance) * window;
        sum += *weight;
    }
    // Phase normalization gives unity DC gain. Each output frame is normalized again
    // after boundary taps are omitted, preserving constants at clip edges as well.
    for weight in kernel.iter_mut() {
        *weight /= sum;
    }
    // This is the divisor the old output loop obtained by adding every normalized
    // coefficient in tap order for an interior frame. Retaining it per phase removes
    // repeated normalization work without changing a single f64 addition.
    let mut normalized_sum = 0.0;
    for &weight in kernel.iter() {
        normalized_sum += weight;
    }
    normalized_sum
}

#[derive(Debug)]
struct ResamplePlan {
    phase_count: usize,
    phase_step: usize,
    taps_per_phase: usize,
    half_taps: usize,
    /// All phases in one allocation: `phase * taps_per_phase .. (phase + 1) * taps_per_phase`.
    coefficients: Vec<f64>,
    /// Sum of each normalized phase, accumulated in original tap order.
    phase_sums: Vec<f64>,
}

impl ResamplePlan {
    fn new(src_rate: u32, dst_rate: u32) -> Result<Self> {
        let divisor = gcd(src_rate, dst_rate);
        let phase_count = (dst_rate / divisor) as usize;
        let phase_step = (src_rate / divisor) as usize;
        let cutoff = (dst_rate as f64 / src_rate as f64).min(1.0) * RESAMPLE_CUTOFF_GUARD;
        let required_taps = RESAMPLE_BASE_TAPS_PER_PHASE as f64 * RESAMPLE_CUTOFF_GUARD / cutoff;
        if required_taps > RESAMPLE_MAX_TAPS_PER_PHASE as f64 {
            return Err(AudioError::Msg(format!(
                "resample ratio {src_rate} -> {dst_rate} needs more than \
                 {RESAMPLE_MAX_TAPS_PER_PHASE} taps per phase"
            )));
        }
        let taps_per_phase = (required_taps.ceil() as usize) | 1;
        let coefficient_count = phase_count.checked_mul(taps_per_phase).ok_or_else(|| {
            AudioError::Msg(format!(
                "resample coefficient table size overflows usize for {src_rate} -> {dst_rate}"
            ))
        })?;
        if coefficient_count > RESAMPLE_MAX_PRECOMPUTED_COEFFICIENTS {
            return Err(AudioError::Msg(format!(
                "resample ratio {src_rate} -> {dst_rate} needs {coefficient_count} precomputed \
                 coefficients, above the limit of {RESAMPLE_MAX_PRECOMPUTED_COEFFICIENTS}"
            )));
        }

        let kaiser_denominator = bessel_i0(RESAMPLE_KAISER_BETA);
        let mut coefficients = Vec::with_capacity(coefficient_count);
        let mut phase_sums = Vec::with_capacity(phase_count);
        for phase in 0..phase_count {
            let start = coefficients.len();
            coefficients.resize(start + taps_per_phase, 0.0);
            let sum = fill_phase_kernel(
                &mut coefficients[start..],
                phase,
                phase_count,
                cutoff,
                kaiser_denominator,
            );
            phase_sums.push(sum);
        }
        Ok(Self {
            phase_count,
            phase_step,
            taps_per_phase,
            half_taps: taps_per_phase / 2,
            coefficients,
            phase_sums,
        })
    }

    fn kernel(&self, phase: usize) -> &[f64] {
        let start = phase * self.taps_per_phase;
        &self.coefficients[start..start + self.taps_per_phase]
    }
}

/// Number of output frames produced by the shared rational resampler.
///
/// Non-empty clips round `input_frames * dst_rate / src_rate` to nearest (ties upward) and are
/// clamped to one frame. Empty clips remain empty. This is the same rule [`resample`] has always
/// used and is exposed so bounded callers can select a range on the global output timeline.
pub fn resample_output_frames(input_frames: usize, src_rate: u32, dst_rate: u32) -> Result<usize> {
    if src_rate == 0 || dst_rate == 0 {
        return Err(AudioError::Msg(format!(
            "resample rates must be non-zero, got {src_rate} -> {dst_rate}"
        )));
    }
    if input_frames == 0 {
        return Ok(0);
    }
    let output_frames_u128 =
        ((input_frames as u128) * (dst_rate as u128) + (src_rate as u128 / 2)) / src_rate as u128;
    usize::try_from(output_frames_u128.max(1)).map_err(|_| {
        AudioError::Msg(format!(
            "resample output length does not fit usize for {input_frames} frames at \
             {src_rate} -> {dst_rate}"
        ))
    })
}

/// Resample one complete interleaved PCM buffer with a rational polyphase
/// Kaiser-windowed-sinc FIR.
///
/// `channels` is the number of interleaved values per frame; each channel is filtered
/// independently, so stereo and multichannel inputs cannot bleed across channel boundaries.
/// The returned frame count is `round(input_frames * dst_rate / src_rate)`, clamped to one
/// frame for every non-empty input. An equal-rate conversion is byte-identical.
///
/// The implementation is in-house rather than dependency-backed: the fixed-rate audio lane
/// needs only rational conversion, and this small kernel keeps the coefficient design and
/// channel/boundary behavior directly auditable. Filter support scales with the downsample
/// ratio so large reductions retain the same stopband quality. All phase coefficients are
/// precomputed in one contiguous table; ratios whose table would exceed the audited bound are
/// rejected before any output work begins.
pub fn resample(samples: &[f32], src_rate: u32, dst_rate: u32, channels: u16) -> Result<Vec<f32>> {
    let channels_usize = validate_resample_input(samples, src_rate, dst_rate, channels)?;
    let input_frames = samples.len() / channels_usize;
    let output_frames = resample_output_frames(input_frames, src_rate, dst_rate)?;
    resample_range(samples, src_rate, dst_rate, channels, 0..output_frames)
}

/// Resample a range of frames on the complete clip's global output timeline.
///
/// This is not chunked/streaming resampling: `samples` is still the full source clip, so output
/// phase and FIR boundary behavior are exactly those of [`resample`]. `output_range` is half-open
/// and must lie within `0..resample_output_frames(input_frames, src_rate, dst_rate)`. Consequently,
/// slicing the whole-buffer result and calling this function with the same range are bit-identical.
/// Empty input admits only `0..0`; equal-rate ranges are byte-identical source-frame slices.
pub fn resample_range(
    samples: &[f32],
    src_rate: u32,
    dst_rate: u32,
    channels: u16,
    output_range: Range<usize>,
) -> Result<Vec<f32>> {
    resample_range_impl(samples, src_rate, dst_rate, channels, output_range, false)
}

/// Downmix interleaved input to mono while resampling a bounded global output range.
///
/// Each source frame is averaged in the same `f32`, channel-order accumulation used by provider
/// mono preparation, but no whole-clip mono buffer is allocated. Range, rounding, empty, and
/// equal-rate semantics are identical to [`resample_range`].
pub fn resample_mono_range(
    samples: &[f32],
    src_rate: u32,
    dst_rate: u32,
    channels: u16,
    output_range: Range<usize>,
) -> Result<Vec<f32>> {
    resample_range_impl(samples, src_rate, dst_rate, channels, output_range, true)
}

fn validate_resample_input(
    samples: &[f32],
    src_rate: u32,
    dst_rate: u32,
    channels: u16,
) -> Result<usize> {
    if src_rate == 0 || dst_rate == 0 {
        return Err(AudioError::Msg(format!(
            "resample rates must be non-zero, got {src_rate} -> {dst_rate}"
        )));
    }
    if channels == 0 {
        return Err(AudioError::Msg("resample channels must be non-zero".into()));
    }
    let channels = channels as usize;
    if !samples.len().is_multiple_of(channels) {
        return Err(AudioError::Msg(format!(
            "resample input has {} samples, not a whole number of {channels}-channel frames",
            samples.len()
        )));
    }
    Ok(channels)
}

fn validate_output_range(range: &Range<usize>, output_frames: usize) -> Result<()> {
    if range.start > range.end || range.end > output_frames {
        return Err(AudioError::Msg(format!(
            "resample output range {}..{} lies outside 0..{output_frames}",
            range.start, range.end
        )));
    }
    Ok(())
}

fn mono_frame(samples: &[f32], frame: usize, channels: usize) -> f32 {
    let start = frame * channels;
    samples[start..start + channels]
        .iter()
        .copied()
        .sum::<f32>()
        / channels as f32
}

fn resample_range_impl(
    samples: &[f32],
    src_rate: u32,
    dst_rate: u32,
    channels: u16,
    output_range: Range<usize>,
    mono: bool,
) -> Result<Vec<f32>> {
    let channels = validate_resample_input(samples, src_rate, dst_rate, channels)?;
    let input_frames = samples.len() / channels;
    let output_frames = resample_output_frames(input_frames, src_rate, dst_rate)?;
    validate_output_range(&output_range, output_frames)?;
    let output_channels = if mono { 1 } else { channels };
    let requested_frames = output_range.end - output_range.start;
    let output_samples = requested_frames
        .checked_mul(output_channels)
        .ok_or_else(|| {
            AudioError::Msg(format!(
                "resample output sample count overflows usize ({requested_frames} frames, \
             {output_channels} channels)"
            ))
        })?;

    if samples.is_empty() {
        return Ok(Vec::new());
    }
    if src_rate == dst_rate {
        if mono {
            return Ok((output_range.start..output_range.end)
                .map(|frame| {
                    #[cfg(any(test, feature = "testkit"))]
                    {
                        RESAMPLE_OUTPUT_WORK.fetch_add(1, Ordering::Relaxed);
                        RESAMPLE_SOURCE_FRAME_WORK.fetch_add(1, Ordering::Relaxed);
                    }
                    mono_frame(samples, frame, channels)
                })
                .collect());
        }
        #[cfg(any(test, feature = "testkit"))]
        {
            RESAMPLE_OUTPUT_WORK.fetch_add(requested_frames, Ordering::Relaxed);
            RESAMPLE_SOURCE_FRAME_WORK.fetch_add(requested_frames, Ordering::Relaxed);
        }
        return Ok(samples[output_range.start * channels..output_range.end * channels].to_vec());
    }

    // Construct and validate the complete coefficient table before allocating or evaluating any
    // output. In particular, pathological relatively-prime rates never fall back to per-output
    // phase construction.
    let plan = ResamplePlan::new(src_rate, dst_rate)?;
    let mut output = vec![0.0f32; output_samples];
    for (local_output_frame, output_frame) in output_range.enumerate() {
        #[cfg(any(test, feature = "testkit"))]
        RESAMPLE_OUTPUT_WORK.fetch_add(1, Ordering::Relaxed);

        let source_numerator = (output_frame as u128) * (plan.phase_step as u128);
        let source_frame =
            usize::try_from(source_numerator / plan.phase_count as u128).map_err(|_| {
                AudioError::Msg("resample source-frame index does not fit usize".into())
            })?;
        let phase = (source_numerator % plan.phase_count as u128) as usize;
        let kernel = plan.kernel(phase);
        let interior_start = source_frame.checked_sub(plan.half_taps);
        let interior = interior_start
            .and_then(|start| {
                start
                    .checked_add(plan.taps_per_phase)
                    .map(|end| (start, end))
            })
            .filter(|&(_, end)| end <= input_frames);

        for output_channel in 0..output_channels {
            let mut value = 0.0f64;
            let included_weight;
            if let Some((first_input_frame, end_input_frame)) = interior {
                // The source window is checked once. This dot product has no per-tap boundary
                // predicate, while preserving the old tap-order f64 accumulation exactly.
                if mono {
                    for (&weight, input_frame) in
                        kernel.iter().zip(first_input_frame..end_input_frame)
                    {
                        value += mono_frame(samples, input_frame, channels) as f64 * weight;
                    }
                } else {
                    let start = first_input_frame * channels + output_channel;
                    let end = end_input_frame * channels;
                    for (&weight, &sample) in kernel
                        .iter()
                        .zip(samples[start..end].iter().step_by(channels))
                    {
                        value += sample as f64 * weight;
                    }
                }
                #[cfg(any(test, feature = "testkit"))]
                RESAMPLE_SOURCE_FRAME_WORK.fetch_add(plan.taps_per_phase, Ordering::Relaxed);
                included_weight = plan.phase_sums[phase];
            } else {
                // Only leading/trailing frames pay checked full-clip boundary evaluation.
                let mut weight_sum = 0.0f64;
                for (tap, &weight) in kernel.iter().enumerate() {
                    let input_frame = source_frame
                        .checked_add(tap)
                        .and_then(|shifted| shifted.checked_sub(plan.half_taps));
                    if let Some(input_frame) = input_frame.filter(|&frame| frame < input_frames) {
                        let sample = if mono {
                            mono_frame(samples, input_frame, channels)
                        } else {
                            samples[input_frame * channels + output_channel]
                        };
                        value += sample as f64 * weight;
                        weight_sum += weight;
                        #[cfg(any(test, feature = "testkit"))]
                        RESAMPLE_SOURCE_FRAME_WORK.fetch_add(1, Ordering::Relaxed);
                    }
                }
                included_weight = weight_sum;
            }
            output[local_output_frame * output_channels + output_channel] =
                if included_weight.abs() > 1e-12 {
                    (value / included_weight) as f32
                } else {
                    let fallback = source_frame.min(input_frames - 1);
                    if mono {
                        mono_frame(samples, fallback, channels)
                    } else {
                        samples[fallback * channels + output_channel]
                    }
                };
        }
    }
    Ok(output)
}

/// A one-sided complex spectrogram: `n_bins = n_fft / 2 + 1` rows by `n_frames` columns,
/// stored bin-major (`index = bin * n_frames + frame`) — the layout of a `[n_bins,
/// n_frames]` model tensor's host copy, so provider code moves data without transposes.
#[derive(Clone, Debug)]
pub struct Spectrogram {
    pub n_bins: usize,
    pub n_frames: usize,
    /// Real parts, bin-major.
    pub re: Vec<f32>,
    /// Imaginary parts, bin-major.
    pub im: Vec<f32>,
}

impl Spectrogram {
    /// Per-bin magnitudes `sqrt(re² + im²)`, bin-major.
    pub fn magnitude(&self) -> Vec<f32> {
        self.re
            .iter()
            .zip(&self.im)
            .map(|(r, i)| (r * r + i * i).sqrt())
            .collect()
    }

    /// Per-bin phases `atan2(im, re)`, bin-major.
    pub fn phase(&self) -> Vec<f32> {
        self.re
            .iter()
            .zip(&self.im)
            .map(|(r, i)| i.atan2(*r))
            .collect()
    }
}

/// In-place iterative radix-2 Cooley–Tukey FFT over `(re, im)` pairs. `invert` runs the
/// inverse transform (without the `1/n` scale — callers apply it). `data.len()` must be a
/// power of two (checked by the public entry points).
fn fft_in_place(data: &mut [(f32, f32)], invert: bool) {
    let n = data.len();
    // Bit-reversal permutation.
    let mut j = 0usize;
    for i in 1..n {
        let mut bit = n >> 1;
        while j & bit != 0 {
            j ^= bit;
            bit >>= 1;
        }
        j |= bit;
        if i < j {
            data.swap(i, j);
        }
    }
    let sign = if invert { 1.0f64 } else { -1.0f64 };
    let mut len = 2;
    while len <= n {
        // f64 twiddles: the recurrence-free per-butterfly angle keeps error flat over
        // long windows (n_fft up to 2048 for the mel front-ends).
        let ang = sign * 2.0 * std::f64::consts::PI / len as f64;
        for start in (0..n).step_by(len) {
            for k in 0..len / 2 {
                let (wr, wi) = ((ang * k as f64).cos() as f32, (ang * k as f64).sin() as f32);
                let (ar, ai) = data[start + k];
                let (br, bi) = data[start + k + len / 2];
                let (tr, ti) = (br * wr - bi * wi, br * wi + bi * wr);
                data[start + k] = (ar + tr, ai + ti);
                data[start + k + len / 2] = (ar - tr, ai - ti);
            }
        }
        len <<= 1;
    }
}

fn require_power_of_two(n_fft: usize) -> Result<()> {
    if n_fft < 2 || !n_fft.is_power_of_two() {
        return Err(AudioError::Msg(format!(
            "n_fft {n_fft} must be a power of two >= 2 (radix-2 STFT)"
        )));
    }
    Ok(())
}

fn require_window(window: &[f32], n_fft: usize) -> Result<()> {
    if window.len() != n_fft {
        return Err(AudioError::Msg(format!(
            "window length {} does not match n_fft {n_fft}",
            window.len()
        )));
    }
    Ok(())
}

/// Real FFT of one `n_fft`-long frame → the `n_fft / 2 + 1` one-sided bins.
fn rfft(frame: &[f32]) -> Vec<(f32, f32)> {
    let n = frame.len();
    let mut data: Vec<(f32, f32)> = frame.iter().map(|&x| (x, 0.0)).collect();
    fft_in_place(&mut data, false);
    data.truncate(n / 2 + 1);
    data
}

/// Inverse real FFT of `n / 2 + 1` one-sided bins → an `n`-long real frame (with the
/// `1/n` scale applied), reconstructing the negative frequencies by conjugate symmetry.
fn irfft(bins: &[(f32, f32)], n: usize) -> Vec<f32> {
    let mut data: Vec<(f32, f32)> = Vec::with_capacity(n);
    data.extend_from_slice(bins);
    for k in (1..n / 2).rev() {
        let (r, i) = bins[k];
        data.push((r, -i));
    }
    fft_in_place(&mut data, true);
    let scale = 1.0 / n as f32;
    data.into_iter().map(|(r, _)| r * scale).collect()
}

/// Forward STFT with `center=True` reflect padding — the analysis half of the pair. The
/// output layout matches a `[n_bins, n_frames]` model tensor (see [`Spectrogram`]).
pub fn stft(samples: &[f32], n_fft: usize, hop: usize, window: &[f32]) -> Result<Spectrogram> {
    require_power_of_two(n_fft)?;
    require_window(window, n_fft)?;
    if hop == 0 {
        return Err(AudioError::Msg("hop must be >= 1".into()));
    }
    if samples.len() < 2 {
        return Err(AudioError::Msg(format!(
            "stft needs at least 2 samples to reflect-pad, got {}",
            samples.len()
        )));
    }
    let pad = n_fft / 2;
    if pad >= samples.len() {
        return Err(AudioError::Msg(format!(
            "stft reflect pad {pad} needs input longer than n_fft/2, got {} samples",
            samples.len()
        )));
    }
    // Reflect-pad by n_fft/2 on both ends (librosa `pad_mode="reflect"`).
    let mut padded = Vec::with_capacity(samples.len() + 2 * pad);
    padded.extend((1..=pad).rev().map(|i| samples[i]));
    padded.extend_from_slice(samples);
    padded.extend((0..pad).map(|i| samples[samples.len() - 2 - i]));

    let n_bins = n_fft / 2 + 1;
    let n_frames = 1 + (padded.len() - n_fft) / hop;
    let mut re = vec![0.0f32; n_bins * n_frames];
    let mut im = vec![0.0f32; n_bins * n_frames];
    let mut frame = vec![0.0f32; n_fft];
    for t in 0..n_frames {
        let start = t * hop;
        for (dst, (x, w)) in frame
            .iter_mut()
            .zip(padded[start..start + n_fft].iter().zip(window))
        {
            *dst = x * w;
        }
        for (bin, (r, i)) in rfft(&frame).into_iter().enumerate() {
            re[bin * n_frames + t] = r;
            im[bin * n_frames + t] = i;
        }
    }
    Ok(Spectrogram {
        n_bins,
        n_frames,
        re,
        im,
    })
}

/// Inverse STFT from bin-major magnitude + phase arrays (`[n_bins, n_frames]`, the host
/// copy of an iSTFT-Net head's output tensors) → time-domain samples. Windowed
/// overlap-add with summed-squared-window normalization, trimming the `center=True` pad —
/// the synthesis half of [`stft`] and the vocoder tail Kokoro's decoder needs (sc-12836).
pub fn istft(
    magnitude: &[f32],
    phase: &[f32],
    n_frames: usize,
    n_fft: usize,
    hop: usize,
    window: &[f32],
) -> Result<Vec<f32>> {
    require_power_of_two(n_fft)?;
    require_window(window, n_fft)?;
    if hop == 0 {
        return Err(AudioError::Msg("hop must be >= 1".into()));
    }
    let n_bins = n_fft / 2 + 1;
    if magnitude.len() != n_bins * n_frames || phase.len() != n_bins * n_frames {
        return Err(AudioError::Msg(format!(
            "istft expects bin-major [{n_bins}, {n_frames}] magnitude and phase \
             ({} values), got {} and {}",
            n_bins * n_frames,
            magnitude.len(),
            phase.len()
        )));
    }
    if n_frames == 0 {
        return Ok(Vec::new());
    }
    let out_len = n_fft + (n_frames - 1) * hop;
    let mut out = vec![0.0f32; out_len];
    let mut wsum = vec![0.0f32; out_len];
    let mut bins = vec![(0.0f32, 0.0f32); n_bins];
    for t in 0..n_frames {
        for (bin, dst) in bins.iter_mut().enumerate() {
            let m = magnitude[bin * n_frames + t];
            let p = phase[bin * n_frames + t];
            *dst = (m * p.cos(), m * p.sin());
        }
        let frame = irfft(&bins, n_fft);
        let start = t * hop;
        for (i, (x, w)) in frame.iter().zip(window).enumerate() {
            out[start + i] += x * w;
            wsum[start + i] += w * w;
        }
    }
    for (x, w) in out.iter_mut().zip(&wsum) {
        // Skip the (near-)zero-coverage edges rather than dividing by ~0.
        if *w > 1e-8 {
            *x /= *w;
        }
    }
    // Trim the center=True analysis pad so a stft→istft round trip aligns with the input.
    let pad = n_fft / 2;
    let end = out_len.saturating_sub(pad);
    Ok(out[pad.min(end)..end].to_vec())
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Test-only copy of the pre-sc-16602 resampling loop. Keep its per-phase allocations,
    /// per-output checked tap walk, and per-channel normalization intact: this is the numerical
    /// compatibility oracle for the flattened/interior-specialized implementation above.
    fn old_resample_oracle(
        samples: &[f32],
        src_rate: u32,
        dst_rate: u32,
        channels: u16,
    ) -> Result<Vec<f32>> {
        let channels = validate_resample_input(samples, src_rate, dst_rate, channels)?;
        if samples.is_empty() || src_rate == dst_rate {
            return Ok(samples.to_vec());
        }
        let input_frames = samples.len() / channels;
        let output_frames = resample_output_frames(input_frames, src_rate, dst_rate)?;
        let divisor = gcd(src_rate, dst_rate);
        let phase_count = (dst_rate / divisor) as usize;
        let phase_step = (src_rate / divisor) as usize;
        let cutoff = (dst_rate as f64 / src_rate as f64).min(1.0) * RESAMPLE_CUTOFF_GUARD;
        let required_taps = RESAMPLE_BASE_TAPS_PER_PHASE as f64 * RESAMPLE_CUTOFF_GUARD / cutoff;
        let taps_per_phase = (required_taps.ceil() as usize) | 1;
        let half_taps = (taps_per_phase / 2) as isize;
        let kaiser_denominator = bessel_i0(RESAMPLE_KAISER_BETA);
        let phase_kernels = (0..phase_count)
            .map(|phase| {
                let mut kernel = vec![0.0; taps_per_phase];
                fill_phase_kernel(&mut kernel, phase, phase_count, cutoff, kaiser_denominator);
                kernel
            })
            .collect::<Vec<_>>();

        let mut output = vec![0.0f32; output_frames * channels];
        for output_frame in 0..output_frames {
            let source_numerator = (output_frame as u128) * (phase_step as u128);
            let source_frame = (source_numerator / phase_count as u128) as isize;
            let phase = (source_numerator % phase_count as u128) as usize;
            let kernel = &phase_kernels[phase];
            let first_input_frame = source_frame - half_taps;
            for channel in 0..channels {
                let mut value = 0.0f64;
                let mut included_weight = 0.0f64;
                for (tap, &weight) in kernel.iter().enumerate() {
                    let input_frame = first_input_frame + tap as isize;
                    if (0..input_frames as isize).contains(&input_frame) {
                        value += samples[input_frame as usize * channels + channel] as f64 * weight;
                        included_weight += weight;
                    }
                }
                output[output_frame * channels + channel] = if included_weight.abs() > 1e-12 {
                    (value / included_weight) as f32
                } else {
                    samples[source_frame.clamp(0, input_frames as isize - 1) as usize * channels
                        + channel]
                };
            }
        }
        Ok(output)
    }

    fn assert_bits_eq(actual: &[f32], expected: &[f32], context: &str) {
        assert_eq!(actual.len(), expected.len(), "{context}: length");
        for (index, (&actual, &expected)) in actual.iter().zip(expected).enumerate() {
            assert_eq!(
                actual.to_bits(),
                expected.to_bits(),
                "{context}: sample {index}: {actual:?} != {expected:?}"
            );
        }
    }

    fn deterministic_signal(frames: usize, channels: usize) -> Vec<f32> {
        let mut state = 0x5eed_cafe_u64;
        (0..frames * channels)
            .map(|index| {
                state = state
                    .wrapping_mul(6_364_136_223_846_793_005)
                    .wrapping_add(1);
                let random = ((state >> 40) as i32 - (1 << 23)) as f32 / (1 << 23) as f32;
                random * 0.7 + (index as f32 * 0.013).sin() * 0.3
            })
            .collect()
    }

    #[test]
    fn flattened_resampler_is_bit_identical_to_old_loop_oracle() {
        for &(src_rate, dst_rate) in &[
            (48_000, 16_000),
            (16_000, 48_000),
            (48_000, 44_100),
            (22_050, 32_000),
            (44_100, 32_000),
        ] {
            for channels in [1usize, 2, 3] {
                for frames in [1usize, 17, 257, 1_024] {
                    let random = deterministic_signal(frames, channels);
                    let constant = vec![0.375; frames * channels];
                    let mut impulse = vec![0.0; frames * channels];
                    impulse[(frames / 2) * channels] = 1.0;
                    for (kind, samples) in [
                        ("random", random),
                        ("constant", constant),
                        ("impulse", impulse),
                    ] {
                        let old =
                            old_resample_oracle(&samples, src_rate, dst_rate, channels as u16)
                                .unwrap();
                        let new = resample(&samples, src_rate, dst_rate, channels as u16).unwrap();
                        assert_bits_eq(
                            &new,
                            &old,
                            &format!(
                                "{kind}, {frames} frames, {channels}ch, {src_rate}->{dst_rate}"
                            ),
                        );
                    }
                }
            }
        }
    }

    #[test]
    fn bounded_ranges_equal_slices_of_full_resample_bit_for_bit() {
        let channels = 2usize;
        let samples = deterministic_signal(4_097, channels);
        let full = resample(&samples, 44_100, 48_000, channels as u16).unwrap();
        let output_frames = full.len() / channels;
        let ranges = [
            0..37,
            41..313,
            output_frames / 2 - 100..output_frames / 2 + 101,
            output_frames - 41..output_frames,
            123..123,
        ];
        for range in ranges {
            let bounded =
                resample_range(&samples, 44_100, 48_000, channels as u16, range.clone()).unwrap();
            assert_bits_eq(
                &bounded,
                &full[range.start * channels..range.end * channels],
                &format!("range {range:?}"),
            );
        }
    }

    #[test]
    fn bounded_mono_matches_downmix_then_full_resample() {
        let channels = 4usize;
        let samples = deterministic_signal(8_000, channels);
        let mono: Vec<f32> = samples
            .chunks_exact(channels)
            .map(|frame| frame.iter().copied().sum::<f32>() / channels as f32)
            .collect();
        let full = resample(&mono, 32_000, 48_000, 1).unwrap();
        let range = 777..2_345;
        let bounded =
            resample_mono_range(&samples, 32_000, 48_000, channels as u16, range.clone()).unwrap();
        assert_bits_eq(&bounded, &full[range], "bounded mono");
    }

    #[test]
    fn bounded_range_defines_empty_equal_rate_and_invalid_semantics() {
        assert!(resample_range(&[], 44_100, 48_000, 2, 0..0)
            .unwrap()
            .is_empty());
        assert!(resample_range(&[], 44_100, 48_000, 2, 0..1).is_err());

        let stereo = [1.0, -1.0, 0.5, -0.5, 0.25, -0.25];
        assert_eq!(
            resample_range(&stereo, 48_000, 48_000, 2, 1..3).unwrap(),
            stereo[2..6]
        );
        assert_eq!(
            resample_mono_range(&stereo, 48_000, 48_000, 2, 1..3).unwrap(),
            [0.0, 0.0]
        );
        let reversed_start = 2;
        let reversed_end = 1;
        assert!(resample_range(&stereo, 48_000, 48_000, 2, reversed_start..reversed_end).is_err());
        assert!(resample_range(&stereo, 48_000, 48_000, 2, 0..4).is_err());
    }

    #[test]
    fn pathological_phase_table_is_rejected_before_output_work() {
        // Relatively-prime rates make 48,000 phases; even the minimum 197-tap support would need
        // 9,456,000 coefficients. The old implementation fell back to phase construction inside
        // the output loop. The bounded contract refuses it up front.
        resample_test_support::reset();
        let error = resample(&[0.25; 100], 47_999, 48_000, 1).unwrap_err();
        assert!(
            error.to_string().contains("precomputed coefficients"),
            "{error}"
        );
        assert_eq!(resample_test_support::work(), (0, 0));
    }

    #[test]
    fn filter_support_keeps_the_197_tap_floor_and_scales_for_downsampling() {
        let equal = ResamplePlan::new(48_000, 48_000).unwrap();
        let down = ResamplePlan::new(48_000, 16_000).unwrap();
        assert_eq!(equal.taps_per_phase, RESAMPLE_BASE_TAPS_PER_PHASE);
        assert!(down.taps_per_phase > RESAMPLE_BASE_TAPS_PER_PHASE);
        assert!(equal.taps_per_phase % 2 == 1 && down.taps_per_phase % 2 == 1);
    }

    #[test]
    fn hann_window_shape_and_symmetry() {
        let w = hann_window(8);
        assert_eq!(w.len(), 8);
        assert!(w[0].abs() < 1e-7, "periodic Hann starts at 0");
        // Periodic symmetry: w[k] == w[N-k].
        for k in 1..8 {
            assert!((w[k] - w[8 - k]).abs() < 1e-6);
        }
        assert!((w[4] - 1.0).abs() < 1e-6, "peak at N/2");
    }

    #[test]
    fn rfft_matches_known_spectrum() {
        // A pure cosine at bin 1 of an 8-point frame: energy lands entirely in bin 1.
        let n = 8;
        let frame: Vec<f32> = (0..n)
            .map(|i| (2.0 * std::f32::consts::PI * i as f32 / n as f32).cos())
            .collect();
        let bins = rfft(&frame);
        assert_eq!(bins.len(), 5);
        assert!((bins[1].0 - 4.0).abs() < 1e-4, "re[1] = n/2");
        for (k, (r, i)) in bins.iter().enumerate() {
            if k != 1 {
                assert!(r.abs() < 1e-4 && i.abs() < 1e-4, "bin {k} must be empty");
            }
        }
    }

    #[test]
    fn irfft_round_trips_rfft() {
        let frame: Vec<f32> = (0..64).map(|i| ((i * 7 % 13) as f32 - 6.0) / 6.0).collect();
        let back = irfft(&rfft(&frame), 64);
        for (a, b) in frame.iter().zip(&back) {
            assert!((a - b).abs() < 1e-5, "{a} vs {b}");
        }
    }

    #[test]
    fn stft_istft_round_trip_reconstructs_signal() {
        // A deterministic multi-tone signal; hop = n_fft/4 gives full window coverage.
        let (n_fft, hop) = (256, 64);
        let signal: Vec<f32> = (0..4096)
            .map(|i| {
                let t = i as f32 / 4096.0;
                (2.0 * std::f32::consts::PI * 40.0 * t).sin()
                    + 0.5 * (2.0 * std::f32::consts::PI * 97.0 * t).cos()
            })
            .collect();
        let window = hann_window(n_fft);
        let spec = stft(&signal, n_fft, hop, &window).unwrap();
        assert_eq!(spec.n_bins, n_fft / 2 + 1);
        let out = istft(
            &spec.magnitude(),
            &spec.phase(),
            spec.n_frames,
            n_fft,
            hop,
            &window,
        )
        .unwrap();
        assert!(
            out.len() >= signal.len(),
            "{} < {}",
            out.len(),
            signal.len()
        );
        // Interior reconstruction error (the outermost hop on each side has partial
        // window coverage by construction).
        let mut worst = 0.0f32;
        for i in n_fft..signal.len() - n_fft {
            worst = worst.max((signal[i] - out[i]).abs());
        }
        assert!(worst < 1e-3, "worst interior error {worst}");
    }

    #[test]
    fn rejects_malformed_configs() {
        let w = hann_window(16);
        assert!(stft(&[0.0; 64], 12, 4, &hann_window(12)).is_err()); // not a power of two
        assert!(stft(&[0.0; 64], 16, 0, &w).is_err()); // zero hop
        assert!(stft(&[0.0; 64], 16, 4, &hann_window(8)).is_err()); // window mismatch
        assert!(stft(&[0.0; 4], 16, 4, &w).is_err()); // too short to reflect-pad
        assert!(istft(&[0.0; 8], &[0.0; 8], 1, 16, 4, &w).is_err()); // 8 != n_fft/2+1 bins
    }

    #[test]
    fn resample_identity_empty_and_single_frame_edges() {
        let stereo = [0.25, -0.5, 0.75, -1.0];
        assert_eq!(resample(&stereo, 48_000, 48_000, 2).unwrap(), stereo);
        assert!(resample(&[], 48_000, 16_000, 1).unwrap().is_empty());
        assert_eq!(
            resample(&stereo[..2], 48_000, 16_000, 2).unwrap(),
            stereo[..2]
        );
        assert_eq!(resample(&[5.0], 8_000, 48_000, 1).unwrap(), [5.0; 6]);
        assert_eq!(resample(&[5.0], 48_000, 8_000, 1).unwrap(), [5.0]);
        assert!(resample(&stereo, 0, 16_000, 2).is_err());
        assert!(resample(&stereo, 48_000, 0, 2).is_err());
        assert!(resample(&stereo, 48_000, 16_000, 0).is_err());
        assert!(resample(&stereo[..3], 48_000, 16_000, 2).is_err());
    }

    #[test]
    fn resample_changes_frame_count_and_preserves_constants() {
        let mono = vec![0.375; 1_000];
        let up = resample(&mono, 16_000, 24_000, 1).unwrap();
        assert_eq!(up.len(), 1_500);
        assert!(up.iter().all(|&x| (x - 0.375).abs() < 1e-5));

        let down = resample(&mono, 48_000, 16_000, 1).unwrap();
        assert_eq!(down.len(), 333);
        assert!(down.iter().all(|&x| (x - 0.375).abs() < 1e-5));
    }

    #[test]
    fn resample_keeps_interleaved_stereo_channels_separate_at_48k_to_44k1() {
        let mut stereo = Vec::with_capacity(4_800 * 2);
        for _ in 0..4_800 {
            stereo.extend_from_slice(&[0.75, -0.25]);
        }
        let out = resample(&stereo, 48_000, 44_100, 2).unwrap();
        assert_eq!(out.len(), 4_410 * 2);
        for frame in out.chunks_exact(2) {
            assert!((frame[0] - 0.75).abs() < 1e-5, "left {}", frame[0]);
            assert!((frame[1] + 0.25).abs() < 1e-5, "right {}", frame[1]);
        }
    }

    #[test]
    fn resample_rejects_aliases_above_destination_nyquist() {
        let src_rate = 48_000u32;
        let input: Vec<f32> = (0..4_800)
            .map(|i| (2.0 * std::f32::consts::PI * 9_000.0 * i as f32 / src_rate as f32).sin())
            .collect();
        let out = resample(&input, src_rate, 16_000, 1).unwrap();
        let interior = &out[64..out.len() - 64];
        let rms = (interior.iter().map(|x| x * x).sum::<f32>() / interior.len() as f32).sqrt();
        assert!(
            rms < 0.01,
            "9 kHz alias leaked into 16 kHz output: RMS {rms}"
        );
    }

    #[test]
    fn resample_preserves_music_band_and_rejects_48k_to_44k1_aliases() {
        fn tone(rate: u32, frequency: f32) -> Vec<f32> {
            (0..rate / 2)
                .map(|i| {
                    (2.0 * std::f64::consts::PI * frequency as f64 * i as f64 / rate as f64).sin()
                        as f32
                })
                .collect()
        }

        let passband = resample(&tone(48_000, 20_000.0), 48_000, 44_100, 1).unwrap();
        let stopband = resample(&tone(48_000, 22_100.0), 48_000, 44_100, 1).unwrap();
        let rms = |samples: &[f32]| {
            (samples.iter().map(|x| x * x).sum::<f32>() / samples.len() as f32).sqrt()
        };
        let passband_rms = rms(&passband[256..passband.len() - 256]);
        let stopband_rms = rms(&stopband[256..stopband.len() - 256]);
        assert!(
            passband_rms > 0.69,
            "20 kHz passband drooped at 48 kHz -> 44.1 kHz: RMS {passband_rms}"
        );
        assert!(
            stopband_rms < 0.00001,
            "22.1 kHz alias leaked into 44.1 kHz output: RMS {stopband_rms}"
        );
    }

    #[test]
    fn resample_rejects_aliases_across_a_large_downsample_ratio() {
        let src_rate = 192_000u32;
        let input: Vec<f32> = (0..19_200)
            .map(|i| (2.0 * std::f32::consts::PI * 6_000.0 * i as f32 / src_rate as f32).sin())
            .collect();
        let out = resample(&input, src_rate, 8_000, 1).unwrap();
        let interior = &out[64..out.len() - 64];
        let rms = (interior.iter().map(|x| x * x).sum::<f32>() / interior.len() as f32).sqrt();
        assert!(
            rms < 0.01,
            "6 kHz alias leaked into 8 kHz output: RMS {rms}"
        );
    }

    #[test]
    fn resample_preserves_impulse_alignment_at_music_and_watermark_rates() {
        let mut input = vec![0.0f32; 2_048];
        input[1_000] = 1.0;

        let music = resample(&input, 48_000, 44_100, 1).unwrap();
        let music_peak = music
            .iter()
            .enumerate()
            .max_by(|a, b| a.1.total_cmp(b.1))
            .map(|(i, _)| i)
            .unwrap();
        let expected_music_peak = (1_000.0f64 * 44_100.0 / 48_000.0).round() as usize;
        assert!(
            music_peak.abs_diff(expected_music_peak) <= 1,
            "48 kHz -> 44.1 kHz peak shifted from {expected_music_peak} to {music_peak}"
        );

        let up = resample(&input, 24_000, 32_000, 1).unwrap();
        let back = resample(&up, 32_000, 24_000, 1).unwrap();
        let peak = back
            .iter()
            .enumerate()
            .max_by(|a, b| a.1.total_cmp(b.1))
            .map(|(i, _)| i)
            .unwrap();
        assert!(
            peak.abs_diff(1_000) <= 1,
            "round-trip peak shifted from 1000 to {peak}"
        );
    }
}
