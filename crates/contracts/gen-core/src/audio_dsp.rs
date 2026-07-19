//! Host-side audio DSP: composition **mixdown** and ITU-R BS.1770-4 **loudness /
//! true-peak** measurement (sc-12837, phase P2 of the audio roadmap).
//!
//! Like `imageops` / `tiling` / `train::LrSchedule`, this is **pure policy math** — plain
//! `f32` slices in, plain values out, zero tensor deps, no I/O, no engine or trait. The math is
//! ported from the retired SoundWorks reference (`composition_mixdown.rs` + `loudness.rs`),
//! adapted to gen-core idioms: typed [`Error`] validation instead of silent
//! clamping, and [`AudioTrack`] as the interleaved-PCM carrier.
//!
//! Two halves:
//!  - [`mixdown`]: sum per-clip PCM ([`MixClip`]: gain / pan / fade / timeline placement /
//!    source window, with nearest-neighbor sample-rate conversion) plus a master gain into one
//!    interleaved, `[-1, 1]`-clamped [`AudioTrack`].
//!  - [`measure_loudness`]: BS.1770-4 gated integrated loudness (LUFS) — K-weighting as two
//!    cascaded biquads recomputed for the actual sample rate (the libebur128 method), 400 ms
//!    blocks at 75 % overlap, −70 LUFS absolute gate then −10 LU relative gate — and 4×
//!    oversampled true peak (dBTP) via a Hann-windowed-sinc polyphase interpolator. Silence and
//!    too-short signals return the finite [`SILENCE_FLOOR_LUFS`], never `-inf`/`NaN`.

use std::f64::consts::PI;

use crate::error::{Error, Result};
use crate::media::AudioTrack;

// ---------------------------------------------------------------------------
// Mixdown
// ---------------------------------------------------------------------------

/// Convert a decibel gain to a linear multiplier (`0 dB → 1.0`, `-6 dB → ~0.501`).
pub fn db_to_linear(db: f32) -> f32 {
    10f32.powf(db / 20.0)
}

/// One source clip placed on the output timeline, with its PCM already resolved to
/// interleaved `f32` samples in `[-1.0, 1.0]`.
#[derive(Debug, Clone, PartialEq)]
pub struct MixClip {
    /// Interleaved source PCM (`source_channels` samples per frame).
    pub samples: Vec<f32>,
    pub source_channels: u16,
    pub source_sample_rate: u32,
    /// Where the clip begins on the output timeline.
    pub timeline_start_ms: u64,
    /// The window of the source used (relative to the source's own start). A window with
    /// `source_end_ms <= source_start_ms` contributes nothing (a zero-length clip is a no-op,
    /// not an error).
    pub source_start_ms: u64,
    pub source_end_ms: u64,
    /// Linear gain (track gain × clip gain already combined by the caller).
    pub gain: f32,
    /// `-1.0` = hard left, `0.0` = center, `+1.0` = hard right (clamped).
    pub pan: f32,
    pub fade_in_ms: u64,
    pub fade_out_ms: u64,
}

/// A full mixdown request: output spec + the clips to sum.
#[derive(Debug, Clone, PartialEq)]
pub struct MixRequest {
    pub sample_rate: u32,
    pub channels: u16,
    pub duration_ms: u64,
    /// Linear master gain applied after summation, before the `[-1, 1]` clamp.
    pub master_gain: f32,
    pub clips: Vec<MixClip>,
}

impl MixRequest {
    /// Typed validation (gen-core idiom — reject instead of silently clamping): zero output
    /// sample rate / channel count, non-finite gains or pan, zero source rates/channels, and
    /// clip PCM whose length is not a whole number of frames all produce [`Error::Msg`].
    fn validate(&self) -> Result<()> {
        if self.sample_rate == 0 {
            return Err(Error::Msg("mixdown: output sample_rate must be > 0".into()));
        }
        if self.channels == 0 {
            return Err(Error::Msg("mixdown: output channels must be > 0".into()));
        }
        if !self.master_gain.is_finite() {
            return Err(Error::Msg(format!(
                "mixdown: master_gain must be finite, got {}",
                self.master_gain
            )));
        }
        for (i, clip) in self.clips.iter().enumerate() {
            if clip.source_sample_rate == 0 {
                return Err(Error::Msg(format!(
                    "mixdown: clip {i}: source_sample_rate must be > 0"
                )));
            }
            if clip.source_channels == 0 {
                return Err(Error::Msg(format!(
                    "mixdown: clip {i}: source_channels must be > 0"
                )));
            }
            if !clip
                .samples
                .len()
                .is_multiple_of(usize::from(clip.source_channels))
            {
                return Err(Error::Msg(format!(
                    "mixdown: clip {i}: {} samples is not a whole number of {}-channel frames",
                    clip.samples.len(),
                    clip.source_channels
                )));
            }
            if !clip.gain.is_finite() || !clip.pan.is_finite() {
                return Err(Error::Msg(format!(
                    "mixdown: clip {i}: gain ({}) and pan ({}) must be finite",
                    clip.gain, clip.pan
                )));
            }
        }
        Ok(())
    }
}

/// Read one source sample for the given source frame, folding channels: a mono source feeds
/// every output channel; a stereo source maps L/R, with any extra output channel reusing
/// channel 0's fold (`min`-clamped).
fn sample_source(clip: &MixClip, frame: usize, channel: usize) -> f32 {
    let src_channels = usize::from(clip.source_channels);
    let total_frames = clip.samples.len() / src_channels;
    if total_frames == 0 {
        return 0.0;
    }
    let src_channel = channel.min(src_channels - 1);
    let index = frame.min(total_frames - 1) * src_channels + src_channel;
    clip.samples.get(index).copied().unwrap_or(0.0)
}

/// Linear fade envelope at `position_ms` into a clip of `clip_len_ms`.
fn fade_scalar(position_ms: f64, clip_len_ms: f64, fade_in_ms: f64, fade_out_ms: f64) -> f32 {
    let mut scale = 1.0f64;
    if fade_in_ms > 0.0 && position_ms < fade_in_ms {
        scale = scale.min(position_ms / fade_in_ms);
    }
    if fade_out_ms > 0.0 {
        let from_end = clip_len_ms - position_ms;
        if from_end < fade_out_ms {
            scale = scale.min((from_end / fade_out_ms).max(0.0));
        }
    }
    scale.clamp(0.0, 1.0) as f32
}

/// Mix the request's clips into one interleaved output buffer, clamped to `[-1.0, 1.0]`.
///
/// Pure: gain, pan (linear taper — center keeps both channels at unity, a hard pan silences
/// the opposite channel), linear fades, timeline placement, nearest-neighbor sample-rate
/// conversion, then master gain and the final clamp. Returns the output as an [`AudioTrack`]
/// carrying the request's `sample_rate`/`channels`.
pub fn mixdown(request: &MixRequest) -> Result<AudioTrack> {
    request.validate()?;
    let channels = usize::from(request.channels);
    let out_sr = request.sample_rate;
    let out_frames = ((u128::from(request.duration_ms) * u128::from(out_sr)) / 1000) as usize;
    let mut out = vec![0.0f32; out_frames * channels];

    for clip in &request.clips {
        if clip.source_end_ms <= clip.source_start_ms || clip.samples.is_empty() {
            continue;
        }
        let clip_len_ms = (clip.source_end_ms - clip.source_start_ms) as f64;
        let clip_out_frames = ((clip_len_ms * f64::from(out_sr)) / 1000.0).round() as usize;
        let start_frame =
            ((clip.timeline_start_ms as f64 * f64::from(out_sr)) / 1000.0).round() as usize;
        // Pan: simple linear taper so center keeps both channels at unity and a hard pan
        // silences the opposite channel.
        let pan = clip.pan.clamp(-1.0, 1.0);
        let left_pan = if pan > 0.0 { 1.0 - pan } else { 1.0 };
        let right_pan = if pan < 0.0 { 1.0 + pan } else { 1.0 };
        let ratio = f64::from(clip.source_sample_rate) / f64::from(out_sr);
        let source_start_frame =
            (clip.source_start_ms as f64 * f64::from(clip.source_sample_rate)) / 1000.0;

        for i in 0..clip_out_frames {
            let out_frame = start_frame + i;
            if out_frame >= out_frames {
                break;
            }
            let position_ms = (i as f64 / f64::from(out_sr)) * 1000.0;
            let env = fade_scalar(
                position_ms,
                clip_len_ms,
                clip.fade_in_ms as f64,
                clip.fade_out_ms as f64,
            );
            let gain = clip.gain * env;
            let src_frame = (source_start_frame + i as f64 * ratio).round() as usize;
            for ch in 0..channels {
                let raw = sample_source(clip, src_frame, ch);
                let pan_gain = if channels >= 2 {
                    if ch == 0 {
                        left_pan
                    } else if ch == 1 {
                        right_pan
                    } else {
                        1.0
                    }
                } else {
                    1.0
                };
                out[out_frame * channels + ch] += raw * gain * pan_gain;
            }
        }
    }

    for sample in out.iter_mut() {
        *sample = (*sample * request.master_gain).clamp(-1.0, 1.0);
    }
    Ok(AudioTrack {
        samples: out,
        sample_rate: request.sample_rate,
        channels: request.channels,
        ..Default::default()
    })
}

// ---------------------------------------------------------------------------
// BS.1770-4 loudness + true peak
// ---------------------------------------------------------------------------

/// Returned when the signal is silent or every block gates out, so downstream metadata stays
/// finite rather than `-inf`/`NaN`.
pub const SILENCE_FLOOR_LUFS: f32 = -120.0;

/// Loudness measurement for an interleaved audio buffer.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct LoudnessStats {
    /// BS.1770-4 gated integrated loudness, LUFS (floored at [`SILENCE_FLOOR_LUFS`]).
    pub integrated_lufs: f32,
    /// 4×-oversampled true peak, dBTP (always finite; never below the sample peak).
    pub true_peak_dbtp: f32,
}

#[derive(Clone, Copy)]
struct Biquad {
    b0: f64,
    b1: f64,
    b2: f64,
    a1: f64,
    a2: f64,
}

impl Biquad {
    /// BS.1770 stage 1: high-shelf "pre-filter" (libebur128 coefficients, recomputed for `fs`
    /// via the bilinear transform).
    fn k_weighting_shelf(fs: f64) -> Self {
        let f0 = 1681.974450955533;
        let q = 0.7071752369554196;
        let gain_db = 3.999843853973347;
        let k = (PI * f0 / fs).tan();
        let vh = 10f64.powf(gain_db / 20.0);
        let vb = vh.powf(0.4996667741545416);
        let a0 = 1.0 + k / q + k * k;
        Self {
            b0: (vh + vb * k / q + k * k) / a0,
            b1: 2.0 * (k * k - vh) / a0,
            b2: (vh - vb * k / q + k * k) / a0,
            a1: 2.0 * (k * k - 1.0) / a0,
            a2: (1.0 - k / q + k * k) / a0,
        }
    }

    /// BS.1770 stage 2: high-pass "RLB" filter.
    fn k_weighting_highpass(fs: f64) -> Self {
        let f0 = 38.13547087602444;
        let q = 0.5003270373238773;
        let k = (PI * f0 / fs).tan();
        let a0 = 1.0 + k / q + k * k;
        Self {
            b0: 1.0,
            b1: -2.0,
            b2: 1.0,
            a1: 2.0 * (k * k - 1.0) / a0,
            a2: (1.0 - k / q + k * k) / a0,
        }
    }

    /// Filter one channel in place (Direct Form I).
    fn process(&self, samples: &mut [f64]) {
        let (mut x1, mut x2, mut y1, mut y2) = (0.0, 0.0, 0.0, 0.0);
        for sample in samples.iter_mut() {
            let x0 = *sample;
            let y0 = self.b0 * x0 + self.b1 * x1 + self.b2 * x2 - self.a1 * y1 - self.a2 * y2;
            x2 = x1;
            x1 = x0;
            y2 = y1;
            y1 = y0;
            *sample = y0;
        }
    }
}

/// Measure BS.1770-4 gated integrated loudness (LUFS) and 4×-oversampled true peak (dBTP) of
/// interleaved `f32` PCM. Per-channel gating weights are 1.0 — correct for the mono and L/R
/// stereo layouts the audio contract produces (BS.1770's surround weights would only differ
/// for >3-channel layouts).
///
/// Silence and signals too short to fill a gating block still return finite values (LUFS
/// floored at [`SILENCE_FLOOR_LUFS`]) — never `-inf`/`NaN`.
pub fn measure_loudness(samples: &[f32], sample_rate: u32, channels: u16) -> Result<LoudnessStats> {
    if sample_rate == 0 {
        return Err(Error::Msg("loudness: sample_rate must be > 0".into()));
    }
    if channels == 0 {
        return Err(Error::Msg("loudness: channels must be > 0".into()));
    }
    let ch = usize::from(channels);
    if !samples.len().is_multiple_of(ch) {
        return Err(Error::Msg(format!(
            "loudness: {} samples is not a whole number of {channels}-channel frames",
            samples.len()
        )));
    }
    let fs = f64::from(sample_rate);
    Ok(LoudnessStats {
        integrated_lufs: integrated_lufs(samples, fs, ch),
        true_peak_dbtp: true_peak_dbtp(samples, ch),
    })
}

/// [`measure_loudness`] over an [`AudioTrack`].
pub fn measure_track_loudness(track: &AudioTrack) -> Result<LoudnessStats> {
    measure_loudness(&track.samples, track.sample_rate, track.channels)
}

/// Split interleaved samples into per-channel planes of `f64`.
fn deinterleave(samples: &[f32], channels: usize) -> Vec<Vec<f64>> {
    let frames = samples.len() / channels;
    let mut planes = vec![Vec::with_capacity(frames); channels];
    for frame in 0..frames {
        for (ch, plane) in planes.iter_mut().enumerate() {
            plane.push(f64::from(samples[frame * channels + ch]));
        }
    }
    planes
}

fn integrated_lufs(samples: &[f32], fs: f64, channels: usize) -> f32 {
    if samples.is_empty() {
        return SILENCE_FLOOR_LUFS;
    }

    let shelf = Biquad::k_weighting_shelf(fs);
    let highpass = Biquad::k_weighting_highpass(fs);
    let mut planes = deinterleave(samples, channels);
    for plane in &mut planes {
        shelf.process(plane);
        highpass.process(plane);
    }
    let frames = planes.first().map_or(0, Vec::len);
    if frames == 0 {
        return SILENCE_FLOOR_LUFS;
    }

    // 400 ms blocks with 75% overlap (100 ms hop).
    let block = ((fs * 0.4).round() as usize).max(1);
    let hop = ((fs * 0.1).round() as usize).max(1);

    // Per-block summed-channel mean square: sum_c G_c * z_c, with G_c = 1.0 for mono and for
    // L/R stereo (the only layouts produced here).
    let mut block_powers: Vec<f64> = Vec::new();
    if frames < block {
        block_powers.push(block_power(&planes, 0, frames));
    } else {
        let mut start = 0;
        while start + block <= frames {
            block_powers.push(block_power(&planes, start, block));
            start += hop;
        }
    }

    // Absolute gate at −70 LUFS (block loudness = −0.691 + 10·log10(power)).
    let abs_gated: Vec<f64> = block_powers
        .into_iter()
        .filter(|p| *p > 0.0 && block_loudness(*p) >= -70.0)
        .collect();
    if abs_gated.is_empty() {
        return SILENCE_FLOOR_LUFS;
    }

    // Relative gate: −10 LU below the mean power of the absolute-gated blocks.
    let mean_abs = abs_gated.iter().sum::<f64>() / abs_gated.len() as f64;
    let rel_threshold = block_loudness(mean_abs) - 10.0;
    let rel_gated: Vec<f64> = abs_gated
        .iter()
        .copied()
        .filter(|p| block_loudness(*p) >= rel_threshold)
        .collect();
    let gated = if rel_gated.is_empty() {
        abs_gated
    } else {
        rel_gated
    };

    let mean_power = gated.iter().sum::<f64>() / gated.len() as f64;
    if mean_power <= 0.0 {
        return SILENCE_FLOOR_LUFS;
    }
    (block_loudness(mean_power) as f32).max(SILENCE_FLOOR_LUFS)
}

fn block_loudness(power: f64) -> f64 {
    -0.691 + 10.0 * power.log10()
}

/// Sum over channels of the mean square over `[start, start + len)`.
fn block_power(planes: &[Vec<f64>], start: usize, len: usize) -> f64 {
    let mut total = 0.0;
    for plane in planes {
        let end = (start + len).min(plane.len());
        if end <= start {
            continue;
        }
        let sum_sq: f64 = plane[start..end].iter().map(|s| s * s).sum();
        total += sum_sq / (end - start) as f64;
    }
    total
}

/// 4× oversampled true peak in dBTP. The raw samples are always included so the result is
/// never below the sample peak; the interpolated inter-sample phases catch overshoot a plain
/// sample-peak meter misses.
fn true_peak_dbtp(samples: &[f32], channels: usize) -> f32 {
    const OVERSAMPLE: usize = 4;
    const TAPS: usize = 12;
    let planes = deinterleave(samples, channels);
    let phases = polyphase_kernel(OVERSAMPLE, TAPS);
    let mut peak = 1e-7f64;
    for plane in &planes {
        let n = plane.len();
        for &s in plane {
            peak = peak.max(s.abs());
        }
        for i in 0..n {
            for phase in &phases[1..] {
                let mut acc = 0.0;
                for (t, coeff) in phase.iter().enumerate() {
                    let idx = i as isize + t as isize - (TAPS as isize / 2 - 1);
                    if idx >= 0 && (idx as usize) < n {
                        acc += plane[idx as usize] * coeff;
                    }
                }
                peak = peak.max(acc.abs());
            }
        }
    }
    (20.0 * peak.log10()) as f32
}

/// `m` polyphase sub-filters (each `taps` long) of a Hann-windowed-sinc low-pass for `m`×
/// interpolation. Each phase is normalized to unit DC gain so a constant input maps to itself.
fn polyphase_kernel(m: usize, taps: usize) -> Vec<Vec<f64>> {
    let length = m * taps;
    let center = (length as f64 - 1.0) / 2.0;
    let mut phases = vec![vec![0.0f64; taps]; m];
    for (phase, sub) in phases.iter_mut().enumerate() {
        for (t, coeff) in sub.iter_mut().enumerate() {
            let n = (phase + t * m) as f64;
            let x = (n - center) / m as f64;
            let sinc = if x.abs() < 1e-9 {
                1.0
            } else {
                (PI * x).sin() / (PI * x)
            };
            let window = 0.5 - 0.5 * (2.0 * PI * n / (length as f64 - 1.0)).cos();
            *coeff = sinc * window;
        }
        let sum: f64 = sub.iter().sum();
        if sum.abs() > 1e-12 {
            for coeff in sub.iter_mut() {
                *coeff /= sum;
            }
        }
    }
    phases
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::f64::consts::TAU;

    fn sine(freq: f64, amp: f64, secs: f64, fs: u32) -> Vec<f32> {
        let n = (secs * f64::from(fs)) as usize;
        (0..n)
            .map(|i| (amp * (TAU * freq * i as f64 / f64::from(fs)).sin()) as f32)
            .collect()
    }

    fn const_clip(value: f32, frames: usize, gain: f32) -> MixClip {
        MixClip {
            samples: vec![value; frames],
            source_channels: 1,
            source_sample_rate: 48_000,
            timeline_start_ms: 0,
            source_start_ms: 0,
            source_end_ms: (frames as u64 * 1000) / 48_000,
            gain,
            pan: 0.0,
            fade_in_ms: 0,
            fade_out_ms: 0,
        }
    }

    fn mono_request(clips: Vec<MixClip>, duration_ms: u64) -> MixRequest {
        MixRequest {
            sample_rate: 48_000,
            channels: 1,
            duration_ms,
            master_gain: 1.0,
            clips,
        }
    }

    // ---- mixdown ----

    #[test]
    fn db_to_linear_unity_and_minus_six() {
        assert!((db_to_linear(0.0) - 1.0).abs() < 1e-6);
        assert!((db_to_linear(-6.0) - 0.5012).abs() < 1e-3);
    }

    #[test]
    fn output_track_carries_request_spec() {
        let out = mixdown(&MixRequest {
            sample_rate: 44_100,
            channels: 2,
            duration_ms: 500,
            master_gain: 1.0,
            clips: vec![],
        })
        .unwrap();
        assert_eq!(out.sample_rate, 44_100);
        assert_eq!(out.channels, 2);
        assert_eq!(out.samples.len(), 22_050 * 2); // 500 ms of stereo frames
        assert!(out.samples.iter().all(|s| *s == 0.0));
    }

    #[test]
    fn applies_clip_and_master_gain() {
        let mut request = mono_request(vec![const_clip(0.25, 4_800, 2.0)], 100);
        request.master_gain = 0.5;
        let out = mixdown(&request).unwrap();
        // 0.25 * clip gain 2.0 * master 0.5 = 0.25.
        assert!(
            (out.samples[10] - 0.25).abs() < 1e-4,
            "got {}",
            out.samples[10]
        );
    }

    #[test]
    fn overlapping_clips_sum_and_clamp() {
        // Two full-scale clips overlap -> sum 2.0 clamped to 1.0.
        let request = mono_request(
            vec![const_clip(1.0, 4_800, 1.0), const_clip(1.0, 4_800, 1.0)],
            100,
        );
        let out = mixdown(&request).unwrap();
        assert!(
            (out.samples[100] - 1.0).abs() < 1e-6,
            "expected clamp to 1.0"
        );
    }

    #[test]
    fn overlapping_clips_sum_linearly_below_full_scale() {
        let request = mono_request(
            vec![const_clip(0.25, 4_800, 1.0), const_clip(0.5, 4_800, 1.0)],
            100,
        );
        let out = mixdown(&request).unwrap();
        assert!(
            (out.samples[100] - 0.75).abs() < 1e-4,
            "got {}",
            out.samples[100]
        );
    }

    #[test]
    fn hard_pan_silences_opposite_channel() {
        let mut clip = const_clip(0.5, 4_800, 1.0);
        clip.pan = 1.0; // hard right
        let request = MixRequest {
            sample_rate: 48_000,
            channels: 2,
            duration_ms: 100,
            master_gain: 1.0,
            clips: vec![clip],
        };
        let out = mixdown(&request).unwrap();
        // frame 50: left (idx 100) silent, right (idx 101) ~0.5
        assert!(
            out.samples[100].abs() < 1e-6,
            "left should be silent: {}",
            out.samples[100]
        );
        assert!(
            (out.samples[101] - 0.5).abs() < 1e-4,
            "right: {}",
            out.samples[101]
        );
    }

    #[test]
    fn center_pan_keeps_both_channels_at_unity() {
        let clip = const_clip(0.5, 4_800, 1.0);
        let request = MixRequest {
            sample_rate: 48_000,
            channels: 2,
            duration_ms: 100,
            master_gain: 1.0,
            clips: vec![clip],
        };
        let out = mixdown(&request).unwrap();
        assert!((out.samples[100] - 0.5).abs() < 1e-4);
        assert!((out.samples[101] - 0.5).abs() < 1e-4);
    }

    #[test]
    fn fade_in_ramps_from_zero() {
        let mut clip = const_clip(1.0, 4_800, 1.0);
        clip.fade_in_ms = 50;
        let out = mixdown(&mono_request(vec![clip], 100)).unwrap();
        // first sample near zero, midpoint of fade ~0.5, after fade full.
        assert!(
            out.samples[0].abs() < 0.05,
            "fade start: {}",
            out.samples[0]
        );
        let mid = out.samples[1_200]; // 25 ms into a 50 ms fade
        assert!((mid - 0.5).abs() < 0.05, "fade midpoint: {mid}");
        assert!(
            out.samples[4_000] > 0.9,
            "post-fade: {}",
            out.samples[4_000]
        );
    }

    #[test]
    fn fade_out_ramps_to_zero() {
        let mut clip = const_clip(1.0, 4_800, 1.0);
        clip.fade_out_ms = 50;
        let out = mixdown(&mono_request(vec![clip], 100)).unwrap();
        assert!(out.samples[100] > 0.99, "pre-fade: {}", out.samples[100]);
        let last = out.samples[4_799];
        assert!(last < 0.05, "fade end should approach zero: {last}");
    }

    #[test]
    fn timeline_offset_leaves_leading_silence() {
        let mut clip = const_clip(1.0, 4_800, 1.0);
        clip.timeline_start_ms = 50;
        let out = mixdown(&mono_request(vec![clip], 200)).unwrap();
        assert!(
            out.samples[10].abs() < 1e-6,
            "leading silence before offset"
        );
        assert!(
            (out.samples[2_500] - 1.0).abs() < 1e-4,
            "clip audible after offset"
        );
    }

    #[test]
    fn source_window_selects_a_slice_of_the_source() {
        // Source: 100 ms of 0.25 then 100 ms of 0.75; the window takes only the second half.
        let mut samples = vec![0.25f32; 4_800];
        samples.extend(std::iter::repeat_n(0.75f32, 4_800));
        let clip = MixClip {
            samples,
            source_channels: 1,
            source_sample_rate: 48_000,
            timeline_start_ms: 0,
            source_start_ms: 100,
            source_end_ms: 200,
            gain: 1.0,
            pan: 0.0,
            fade_in_ms: 0,
            fade_out_ms: 0,
        };
        let out = mixdown(&mono_request(vec![clip], 100)).unwrap();
        assert!(
            (out.samples[100] - 0.75).abs() < 1e-4,
            "got {}",
            out.samples[100]
        );
    }

    #[test]
    fn sample_rate_conversion_upsamples_a_constant_exactly() {
        // 24 kHz source mixed into a 48 kHz output: nearest-neighbor SRC preserves a constant.
        let mut clip = const_clip(0.5, 2_400, 1.0);
        clip.source_sample_rate = 24_000;
        clip.source_end_ms = 100;
        let out = mixdown(&mono_request(vec![clip], 100)).unwrap();
        assert_eq!(out.samples.len(), 4_800);
        assert!(out.samples.iter().all(|s| (*s - 0.5).abs() < 1e-6));
    }

    #[test]
    fn sample_rate_conversion_preserves_loudness_of_a_sine() {
        // Cross-check the two halves: a 440 Hz sine at 44.1 kHz mixed into a 48 kHz output
        // measures the same integrated loudness as the source (nearest-neighbor SRC keeps the
        // fundamental's energy; distortion products are far below it).
        let src = sine(440.0, 0.5, 2.0, 44_100);
        let clip = MixClip {
            samples: src.clone(),
            source_channels: 1,
            source_sample_rate: 44_100,
            timeline_start_ms: 0,
            source_start_ms: 0,
            source_end_ms: 2_000,
            gain: 1.0,
            pan: 0.0,
            fade_in_ms: 0,
            fade_out_ms: 0,
        };
        let out = mixdown(&mono_request(vec![clip], 2_000)).unwrap();
        let src_lufs = measure_loudness(&src, 44_100, 1).unwrap().integrated_lufs;
        let out_lufs = measure_track_loudness(&out).unwrap().integrated_lufs;
        assert!(
            (src_lufs - out_lufs).abs() < 0.5,
            "source {src_lufs} LUFS vs resampled mix {out_lufs} LUFS"
        );
    }

    #[test]
    fn mono_source_feeds_both_stereo_channels() {
        let clip = const_clip(0.5, 4_800, 1.0);
        let request = MixRequest {
            sample_rate: 48_000,
            channels: 2,
            duration_ms: 100,
            master_gain: 1.0,
            clips: vec![clip],
        };
        let out = mixdown(&request).unwrap();
        assert!((out.samples[200] - 0.5).abs() < 1e-4);
        assert!((out.samples[201] - 0.5).abs() < 1e-4);
    }

    #[test]
    fn zero_length_window_is_a_no_op() {
        let mut clip = const_clip(1.0, 4_800, 1.0);
        clip.source_end_ms = clip.source_start_ms;
        let out = mixdown(&mono_request(vec![clip], 100)).unwrap();
        assert!(out.samples.iter().all(|s| *s == 0.0));
    }

    #[test]
    fn mixdown_rejects_invalid_requests_with_typed_errors() {
        let base = mono_request(vec![const_clip(0.5, 4_800, 1.0)], 100);

        let mut zero_sr = base.clone();
        zero_sr.sample_rate = 0;
        assert!(matches!(mixdown(&zero_sr), Err(Error::Msg(_))));

        let mut zero_ch = base.clone();
        zero_ch.channels = 0;
        assert!(matches!(mixdown(&zero_ch), Err(Error::Msg(_))));

        let mut bad_master = base.clone();
        bad_master.master_gain = f32::NAN;
        assert!(matches!(mixdown(&bad_master), Err(Error::Msg(_))));

        let mut bad_clip_sr = base.clone();
        bad_clip_sr.clips[0].source_sample_rate = 0;
        assert!(matches!(mixdown(&bad_clip_sr), Err(Error::Msg(_))));

        let mut bad_clip_ch = base.clone();
        bad_clip_ch.clips[0].source_channels = 0;
        assert!(matches!(mixdown(&bad_clip_ch), Err(Error::Msg(_))));

        let mut ragged = base.clone();
        ragged.clips[0].source_channels = 2; // 4800 samples -> ok; make it ragged:
        ragged.clips[0].samples.push(0.0);
        assert!(matches!(mixdown(&ragged), Err(Error::Msg(_))));

        let mut bad_gain = base.clone();
        bad_gain.clips[0].gain = f32::INFINITY;
        assert!(matches!(mixdown(&bad_gain), Err(Error::Msg(_))));

        let mut bad_pan = base;
        bad_pan.clips[0].pan = f32::NAN;
        assert!(matches!(mixdown(&bad_pan), Err(Error::Msg(_))));
    }

    // ---- BS.1770-4 loudness ----

    #[test]
    fn full_scale_997hz_sine_reads_minus_3_lufs() {
        // The BS.1770 anchor fixture: a 997 Hz sine at full scale (amplitude 1.0) has mean
        // square 0.5 (-3.01 dB) and the K-weighting gain at ~1 kHz is exactly the +0.691 dB
        // the formula's -0.691 offset cancels, so integrated loudness reads -3.01 LUFS.
        let stats = measure_loudness(&sine(997.0, 1.0, 3.0, 48_000), 48_000, 1).unwrap();
        assert!(
            (stats.integrated_lufs - (-3.01)).abs() < 0.2,
            "expected ~-3.01 LUFS, got {}",
            stats.integrated_lufs
        );
    }

    #[test]
    fn minus_20dbfs_997hz_sine_reads_minus_23_lufs() {
        // Same anchor scaled by -20 dB (amplitude 0.1): -23.01 LUFS.
        let stats = measure_loudness(&sine(997.0, 0.1, 3.0, 48_000), 48_000, 1).unwrap();
        assert!(
            (stats.integrated_lufs - (-23.01)).abs() < 0.2,
            "expected ~-23.01 LUFS, got {}",
            stats.integrated_lufs
        );
    }

    #[test]
    fn lufs_anchor_holds_at_44_1khz_too() {
        // The K-weighting biquads are recomputed for the actual sample rate, so the anchor
        // fixture must hold away from 48 kHz as well.
        let stats = measure_loudness(&sine(997.0, 1.0, 3.0, 44_100), 44_100, 1).unwrap();
        assert!(
            (stats.integrated_lufs - (-3.01)).abs() < 0.2,
            "expected ~-3.01 LUFS at 44.1 kHz, got {}",
            stats.integrated_lufs
        );
    }

    #[test]
    fn dual_mono_stereo_reads_3lu_louder_than_mono() {
        // BS.1770 sums channel powers with unity weights, so the same signal in both stereo
        // channels reads +3.01 LU over the mono measurement.
        let mono = sine(997.0, 0.25, 3.0, 48_000);
        let stereo: Vec<f32> = mono.iter().flat_map(|s| [*s, *s]).collect();
        let m = measure_loudness(&mono, 48_000, 1).unwrap().integrated_lufs;
        let s = measure_loudness(&stereo, 48_000, 2)
            .unwrap()
            .integrated_lufs;
        assert!(
            ((s - m) - 3.01).abs() < 0.1,
            "stereo {s} should read ~3.01 LU above mono {m}"
        );
    }

    #[test]
    fn k_weighting_attenuates_low_frequencies() {
        // Equal-amplitude tones: the RLB high-pass makes a 60 Hz tone read clearly quieter
        // than a 1 kHz tone (a plain-RMS meter rates them equal).
        let low = measure_loudness(&sine(60.0, 0.5, 3.0, 48_000), 48_000, 1)
            .unwrap()
            .integrated_lufs;
        let mid = measure_loudness(&sine(1000.0, 0.5, 3.0, 48_000), 48_000, 1)
            .unwrap()
            .integrated_lufs;
        assert!(
            low < mid - 3.0,
            "expected 60 Hz ({low}) >3 LU quieter than 1 kHz ({mid})"
        );
    }

    #[test]
    fn absolute_gate_excludes_silence() {
        // 1 s loud tone then 4 s of silence: the -70 LUFS absolute gate must exclude the
        // silent tail, so the integrated value tracks the loud part.
        let loud = sine(1000.0, 0.5, 1.0, 48_000);
        let mut padded = loud.clone();
        padded.extend(std::iter::repeat_n(0.0f32, 4 * 48_000));
        let loud_only = measure_loudness(&loud, 48_000, 1).unwrap().integrated_lufs;
        let gated = measure_loudness(&padded, 48_000, 1)
            .unwrap()
            .integrated_lufs;
        assert!(
            (gated - loud_only).abs() < 1.5,
            "gated {gated} should track loud-only {loud_only}"
        );
    }

    #[test]
    fn relative_gate_discounts_a_quiet_tail() {
        // Loud passage then a long tail 20 LU down: the tail passes the -70 absolute gate but
        // the -10 LU relative gate must discount it, keeping the integrated value near the
        // loud passage rather than the duration-weighted average.
        let mut signal = sine(997.0, 0.5, 2.0, 48_000);
        signal.extend(sine(997.0, 0.05, 8.0, 48_000)); // -20 dB relative
        let loud_only = measure_loudness(&sine(997.0, 0.5, 2.0, 48_000), 48_000, 1)
            .unwrap()
            .integrated_lufs;
        let gated = measure_loudness(&signal, 48_000, 1)
            .unwrap()
            .integrated_lufs;
        assert!(
            (gated - loud_only).abs() < 1.0,
            "relative gate should keep {gated} near loud-only {loud_only}"
        );
    }

    // ---- true peak ----

    #[test]
    fn true_peak_exceeds_sample_peak_on_intersample_overshoot() {
        // A full-scale tone at fs/4 with a 45 degree phase lands every sample at +-0.707
        // (sample peak ~ -3 dBFS) while the continuous waveform peaks at ~0 dBTP. The 4x
        // oversampled meter must expose most of that overshoot.
        let fs = 48_000u32;
        let n = fs as usize;
        let samples: Vec<f32> = (0..n)
            .map(|i| {
                (TAU * f64::from(fs) / 4.0 * i as f64 / f64::from(fs) + TAU / 8.0).sin() as f32
            })
            .collect();
        let sample_peak = samples.iter().fold(0.0f32, |m, s| m.max(s.abs()));
        let sample_peak_db = 20.0 * sample_peak.log10();
        let tp = measure_loudness(&samples, fs, 1).unwrap().true_peak_dbtp;
        assert!(
            tp > sample_peak_db + 1.0,
            "true peak {tp} should exceed sample peak {sample_peak_db} by >1 dB"
        );
        assert!(
            (-1.0..=0.5).contains(&tp),
            "true peak of a full-scale tone should read near 0 dBTP, got {tp}"
        );
    }

    #[test]
    fn true_peak_is_never_below_sample_peak() {
        // A lone impulse: the raw samples are folded into the measurement, so the reading is
        // at least the sample peak (0 dB here), with only bounded interpolation overshoot.
        let mut samples = vec![0.0f32; 1_000];
        samples[500] = 1.0;
        let tp = measure_loudness(&samples, 48_000, 1)
            .unwrap()
            .true_peak_dbtp;
        assert!(
            tp >= -1e-3,
            "true peak {tp} must not be below the 0 dB sample peak"
        );
        assert!(tp < 1.0, "impulse overshoot should stay bounded, got {tp}");
    }

    // ---- silence / short signals / errors ----

    #[test]
    fn silence_returns_finite_floor() {
        let stats = measure_loudness(&vec![0.0f32; 48_000], 48_000, 1).unwrap();
        assert_eq!(stats.integrated_lufs, SILENCE_FLOOR_LUFS);
        assert!(stats.true_peak_dbtp.is_finite());
    }

    #[test]
    fn empty_signal_returns_finite_floor() {
        let stats = measure_loudness(&[], 48_000, 2).unwrap();
        assert_eq!(stats.integrated_lufs, SILENCE_FLOOR_LUFS);
        assert!(stats.true_peak_dbtp.is_finite());
    }

    #[test]
    fn sub_block_signal_stays_finite() {
        // 100 ms — shorter than one 400 ms gating block — must still yield finite values.
        let stats = measure_loudness(&sine(997.0, 0.5, 0.1, 48_000), 48_000, 1).unwrap();
        assert!(stats.integrated_lufs.is_finite());
        assert!(stats.true_peak_dbtp.is_finite());
        assert!(stats.integrated_lufs >= SILENCE_FLOOR_LUFS);
    }

    #[test]
    fn quiet_below_absolute_gate_returns_floor_not_nan() {
        // Every block below -70 LUFS: the absolute gate empties and the floor is returned.
        let stats = measure_loudness(&sine(997.0, 1e-5, 1.0, 48_000), 48_000, 1).unwrap();
        assert_eq!(stats.integrated_lufs, SILENCE_FLOOR_LUFS);
        assert!(stats.true_peak_dbtp.is_finite());
    }

    #[test]
    fn loudness_rejects_invalid_input_with_typed_errors() {
        assert!(matches!(
            measure_loudness(&[0.0; 4], 0, 1),
            Err(Error::Msg(_))
        ));
        assert!(matches!(
            measure_loudness(&[0.0; 4], 48_000, 0),
            Err(Error::Msg(_))
        ));
        // 5 samples is not a whole number of stereo frames.
        assert!(matches!(
            measure_loudness(&[0.0; 5], 48_000, 2),
            Err(Error::Msg(_))
        ));
    }

    #[test]
    fn measurement_is_deterministic() {
        let s = sine(440.0, 0.4, 2.0, 44_100);
        let a = measure_loudness(&s, 44_100, 1).unwrap();
        let b = measure_loudness(&s, 44_100, 1).unwrap();
        assert_eq!(a, b);
    }
}
