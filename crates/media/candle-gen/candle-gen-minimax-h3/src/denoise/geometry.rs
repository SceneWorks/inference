//! Frame / latent / audio-token geometry, and **the audio-video time alignment** the joint denoise
//! is only correct under.
//!
//! # The alignment, stated once
//!
//! MiniMax-H3 packs audio and video rows into **one** sequence sharing **one** MM-RoPE time axis
//! ([`crate::dit::positions`]). They are quantized on completely different grids — video on latent
//! frames, audio on 25 ms tokens — so "they line up" is a claim that has to be checked, not a
//! property that falls out of the code.
//!
//! It holds because both modalities advance the *shared rotary clock* at the same rate:
//!
//! | modality | grid | rotary units per grid step | grid steps per second | **units per second** |
//! |---|---|---|---|---|
//! | video | frames at [`MINIMAX_H3_FPS`] | [`ROPE_FRAME_RESCALE`] = 5/3 | 24 | **40** |
//! | audio | latent tokens | 1 | [`AUDIO_LATENTS_PER_SECOND`] = 40 | **40** |
//!
//! [`ROPE_UNITS_PER_SECOND`] is that shared rate, and [`rope_clocks_agree`] is the identity as a
//! checked function rather than a comment. It is the reason
//! [`crate::dit::positions::audio_position_ids`] can give an audio latent exactly **one** rotary
//! unit while [`crate::dit::positions::temporal_grid`] gives a latent frame `5/3 ·
//! frames_it_covers`: those are the same clock expressed in two grids.
//!
//! # Where drift would actually hide
//!
//! Not in the rate — that is two constants. It hides in the **latent-frame count**, because
//! `num_latent_frames` is derived (`17n + 5` frames ⇒ `5n + 2` latents) and every plausible wrong
//! derivation is still a positive integer that produces a runnable sequence.
//!
//! A video latent covers `ROPE_FRAMES_PER_LATENT[i % 5]` frames, cyclically `(1, 4, 4, 4, 4)`, so
//! `5k + 2` latents cover exactly `17k + 5` frames — [`JointGeometry::video_rope_span`] sums to
//! `5/3 · num_frames` **only if the two counts are the matching pair**. Substitute the obvious
//! `num_frames / 4` and the sum is wrong by a factor of ~1.18, which is 15 % AV drift by the end of
//! a 15 s clip and *nothing at all* at the first frame. [`JointGeometry::validate`] rejects it up
//! front instead.
//!
//! # The residual is real, and it is bounded
//!
//! `num_audio_latents = round(num_frames / 24 · 40)` is a **round**, so the two tracks are almost
//! never exactly the same length. The residual is `round(x) − x` rotary units, i.e. at most half a
//! unit = **12.5 ms**, and because `num_frames` is `17n + 5` it takes only three values:
//!
//! | frames | video | audio | delta |
//! |---|---|---|---|
//! | 124 | 5.1667 s | 5.1750 s | **+8.33 ms** |
//! | 141 | 5.8750 s | 5.8750 s | 0 |
//! | 158 | 6.5833 s | 6.5750 s | **−8.33 ms** |
//!
//! …cycling with period 3 over the 14 legal durations ([`LEGAL_FRAME_COUNTS`]); 141, 192, 243, 294
//! and 345 are exactly aligned. [`JointGeometry::av_drift_seconds`] reports it and
//! [`MAX_AV_DRIFT_SECONDS`] bounds it. This is a **muxing** concern, not a denoise one — the
//! delivered mp4 needs a pad/trim policy.
//!
//! Which five are exact is **derived, not tabulated**: [`av_grids_align_exactly`] asks whether
//! `num_frames · 40 / 24` needs the `round` at all, which reduces to `3 | num_frames`. A hardcoded
//! `[141, 192, 243, 294, 345]` would pass for any policy that happened to fit those five and says
//! nothing about the other nine — the exact case that makes an AV test a false green (sc-19425).
//!
//! # The mux policy lives here, in the counts, not in either backend's pipeline
//!
//! [`delivered_audio_samples`] is the length the delivered soundtrack **must** be: the picture's,
//! `round(num_frames / 24 · 32000)`. [`decoded_audio_samples`] is what the audio VAE actually emits.
//! They differ at nine of the fourteen, by at most [`MAX_AV_DRIFT_SECONDS`], and **the audio is the
//! side corrected** — the picture's frame count is on the model's own `17n + 5` lattice and its
//! duration is exact, while the audio count is a `round` of it, so correcting the audio corrects the
//! side that carries the error. Trimming or silence-padding at the tail costs at most 400 samples of
//! a 165 000-sample track; moving a frame instead would take `num_frames` off the lattice that
//! [`JointGeometry::validate`] exists to protect.
//!
//! **This crate has no pipeline yet (sc-17156 owns it), so nothing here calls these functions.**
//! They are stated in this module rather than left for that slice precisely so it inherits the MLX
//! lane's decision instead of re-deriving one: `scripts/check-workspace.py::check_cross_backend_geometry`
//! compares every `pub const` in this file against `mlx-gen-minimax-h3`'s, so
//! [`AUDIO_SAMPLES_PER_LATENT`] and [`MAX_DELIVERED_AV_RESIDUAL_SECONDS`] cannot drift apart —
//! which is the failure sc-19419 shipped with `SIZE_MULTIPLE` while both crates' own tests stayed
//! green (sc-19425).
//!
//! # Duration does not buy memory back (sc-17152)
//!
//! The MLX lane measured peak memory **flat to 0.5 % across both duration and canvas**, with
//! wall-clock the binding constraint and canvas dominating duration. Nothing in this module should
//! be read as a memory argument for shorter clips, and the candle lane must not inherit the
//! *conclusion* either: sc-17152's numbers are MLX's, and this backend's attention has a different
//! memory shape (see [`crate::dit::layers`]). Measuring it here is sc-17156's.

use candle_gen::{CandleError, Result};

use crate::audio_config::{AUDIO_SAMPLE_RATE, AUDIO_TOKEN_RATE_HZ};
use crate::dit::positions::{ROPE_FRAMES_PER_LATENT, ROPE_FRAME_RESCALE};

/// Frame rate every MiniMax-H3 duration is expressed at. `MINIMAX_H3_FPS`.
pub const MINIMAX_H3_FPS: f64 = 24.0;

/// Audio latent tokens per second. `MINIMAX_H3_AUDIO_LATENTS_PER_SECOND`, and the same 40 Hz
/// [`AUDIO_TOKEN_RATE_HZ`] the audio VAE's 800-sample hop at 32 kHz implies.
pub const AUDIO_LATENTS_PER_SECOND: u32 = AUDIO_TOKEN_RATE_HZ;

/// Frames one temporal chunk covers. `frames_per_chunk`.
pub const FRAMES_PER_CHUNK: usize = 17;

/// Latent frames one temporal chunk produces. `latents_per_chunk`.
pub const LATENTS_PER_CHUNK: usize = 5;

/// **The shared MM-RoPE clock rate**: rotary units per second, identical for both modalities.
///
/// `24 fps · 5/3` for video and `40 latents/s · 1` for audio. See [`rope_clocks_agree`].
pub const ROPE_UNITS_PER_SECOND: f64 = 40.0;

/// The worst AV length residual the `round` in [`audio_latent_num_frames`] can produce: half a
/// rotary unit at [`ROPE_UNITS_PER_SECOND`], i.e. 12.5 ms.
pub const MAX_AV_DRIFT_SECONDS: f64 = 0.5 / ROPE_UNITS_PER_SECOND;

/// Audio samples one latent token decodes to — the audio VAE's hop, `32000 / 40`.
pub const AUDIO_SAMPLES_PER_LATENT: u32 = AUDIO_SAMPLE_RATE / AUDIO_TOKEN_RATE_HZ;

/// The worst residual the **delivered** soundtrack can carry against the picture, once
/// [`delivered_audio_samples`] has been applied: half a sample at [`AUDIO_SAMPLE_RATE`], i.e.
/// 15.625 µs.
///
/// Three orders of magnitude under [`MAX_AV_DRIFT_SECONDS`], which is the point — the mux policy's
/// job is to move the AV residual off the audio-latent grid (±8.33 ms) and onto the sample grid,
/// where it is smaller than one sample and cannot be expressed in a container at all. A test that
/// admits a millisecond of slack here would pass for a `floor`, a `ceil`, or a delivered length
/// derived from the decoder's output rather than from `num_frames`.
pub const MAX_DELIVERED_AV_RESIDUAL_SECONDS: f64 = 0.5 / AUDIO_SAMPLE_RATE as f64;

/// Every frame count the released model accepts — `17n + 5` clamped to the hardcoded 5–15 s
/// duration range. Nothing else is legal.
///
/// The advertised duration envelope is **derived from this lattice**, not from the 5–15 s prose:
/// the shortest legal render is 124 frames = 5.1667 s and the longest is 345 = 14.375 s, so a
/// request for a flat 15.0 s has no legal frame count and must be refused rather than silently
/// delivered as 14.375 (sc-17152).
pub const LEGAL_FRAME_COUNTS: [usize; 14] = [
    124, 141, 158, 175, 192, 209, 226, 243, 260, 277, 294, 311, 328, 345,
];

/// Whether the video and audio grids advance the shared rotary clock at the same rate.
///
/// **This is the time-alignment contract in one expression.** Video moves `ROPE_FRAME_RESCALE`
/// units per frame at `MINIMAX_H3_FPS` frames per second; audio moves exactly one unit per latent
/// at `AUDIO_LATENTS_PER_SECOND` latents per second. Both are 40 units/s, which is what lets one
/// packed sequence carry both without a resampling step anywhere.
///
/// Exact in f64: `24 · 5/3` and `40 · 1` are both representable.
pub fn rope_clocks_agree() -> bool {
    let video = MINIMAX_H3_FPS * ROPE_FRAME_RESCALE;
    let audio = f64::from(AUDIO_LATENTS_PER_SECOND);
    video == ROPE_UNITS_PER_SECOND && audio == ROPE_UNITS_PER_SECOND
}

/// Round `num_frames` **up** to the next legal `17n + 5`. `align_num_frames`.
pub fn align_num_frames(num_frames: usize) -> usize {
    let mut n = num_frames;
    while n % FRAMES_PER_CHUNK != LATENTS_PER_CHUNK {
        n += 1;
    }
    n
}

/// Video latent frames for a legal frame count: `17n + 5` ⇒ `5n + 2`. `video_latent_num_frames`.
///
/// Rejects a frame count that is not `17n + 5`, because every wrong-but-runnable derivation of this
/// number is exactly the drift [`JointGeometry::validate`] exists to catch.
pub fn video_latent_num_frames(num_frames: usize) -> Result<usize> {
    if num_frames < LATENTS_PER_CHUNK || num_frames % FRAMES_PER_CHUNK != LATENTS_PER_CHUNK {
        return Err(CandleError::Msg(format!(
            "minimax-h3 geometry: num_frames must be 17n + 5 (124, 141, 158, … 345), got \
             {num_frames}"
        )));
    }
    Ok((num_frames - LATENTS_PER_CHUNK) / FRAMES_PER_CHUNK * LATENTS_PER_CHUNK + 2)
}

/// Audio latent tokens for a frame count: `round(num_frames / 24 · 40)`.
/// `audio_latent_num_frames`.
///
/// The `round` is what makes the two tracks differ in length; see [`MAX_AV_DRIFT_SECONDS`].
pub fn audio_latent_num_frames(num_frames: usize) -> usize {
    (num_frames as f64 / MINIMAX_H3_FPS * f64::from(AUDIO_LATENTS_PER_SECOND)).round() as usize
}

/// Whether the two grids land on the **same instant exactly** for `num_frames`, i.e. whether the
/// `round` in [`audio_latent_num_frames`] has anything to do.
///
/// Asked as the gate's own arithmetic — is `num_frames · AUDIO_LATENTS_PER_SECOND` a whole number of
/// [`MINIMAX_H3_FPS`]? — rather than by matching against the five counts it happens to be true at.
/// The two rates are what decide it, so if either ever moves this follows and a tabulated set would
/// not. At today's 40 and 24 it reduces to `3 | num_frames`, which over [`LEGAL_FRAME_COUNTS`] is
/// exactly 141, 192, 243, 294 and 345.
pub fn av_grids_align_exactly(num_frames: usize) -> bool {
    (num_frames * AUDIO_LATENTS_PER_SECOND as usize).is_multiple_of(MINIMAX_H3_FPS as usize)
}

/// Samples **per channel** the audio VAE actually decodes for `num_frames`:
/// `audio_latent_num_frames · AUDIO_SAMPLES_PER_LATENT`.
///
/// This is the length the soundtrack arrives at, **not** the length it is delivered at — see
/// [`delivered_audio_samples`].
pub fn decoded_audio_samples(num_frames: usize) -> usize {
    audio_latent_num_frames(num_frames) * AUDIO_SAMPLES_PER_LATENT as usize
}

/// **The AV mux policy's target length**: samples **per channel** the delivered soundtrack carries,
/// `round(num_frames / MINIMAX_H3_FPS · AUDIO_SAMPLE_RATE)`.
///
/// A function of the frame count alone. Deriving it from the decoded track instead would make the
/// delivered clip's length depend on the audio grid — which is the ±8.33 ms this whole module
/// exists to keep off the delivered file.
pub fn delivered_audio_samples(num_frames: usize) -> usize {
    (num_frames as f64 / MINIMAX_H3_FPS * f64::from(AUDIO_SAMPLE_RATE)).round() as usize
}

/// The frame / latent / audio-token counts one joint request is denoised at.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct JointGeometry {
    /// Output frames — always `17n + 5`.
    pub num_frames: usize,
    /// Video latent frames — always `5n + 2`.
    pub num_latent_frames: usize,
    /// Audio latent tokens **per channel**.
    pub num_audio_latents: usize,
    /// Latent height, a multiple of the patch.
    pub latent_height: usize,
    /// Latent width, a multiple of the patch.
    pub latent_width: usize,
}

impl JointGeometry {
    /// Derive every count from the frame geometry, then [`Self::validate`].
    pub fn new(num_frames: usize, latent_height: usize, latent_width: usize) -> Result<Self> {
        let g = Self {
            num_frames,
            num_latent_frames: video_latent_num_frames(num_frames)?,
            num_audio_latents: audio_latent_num_frames(num_frames),
            latent_height,
            latent_width,
        };
        g.validate()?;
        Ok(g)
    }

    /// Rotary units the video track spans: `Σ 5/3 · ROPE_FRAMES_PER_LATENT[i % 5]` over
    /// [`Self::num_latent_frames`].
    ///
    /// Summed over the actual per-latent frame counts rather than shortcut to
    /// `5/3 · num_frames`, because agreeing with that shortcut is exactly what
    /// [`Self::validate`] checks.
    pub fn video_rope_span(&self) -> f64 {
        (0..self.num_latent_frames)
            .map(|i| ROPE_FRAME_RESCALE * ROPE_FRAMES_PER_LATENT[i % ROPE_FRAMES_PER_LATENT.len()])
            .sum()
    }

    /// Rotary units the audio track spans: one per latent token.
    pub fn audio_rope_span(&self) -> f64 {
        self.num_audio_latents as f64
    }

    /// Signed AV length residual in seconds — positive when the soundtrack outlasts the picture.
    ///
    /// This is `(round(x) − x) / 40` and is bounded by [`MAX_AV_DRIFT_SECONDS`]; it is the
    /// ±8.33 ms / 0 cycle in the module docs, and a muxing policy input, not a defect.
    pub fn av_drift_seconds(&self) -> f64 {
        (self.audio_rope_span() - self.video_rope_span()) / ROPE_UNITS_PER_SECOND
    }

    /// Clip duration in seconds at [`MINIMAX_H3_FPS`].
    pub fn duration_seconds(&self) -> f64 {
        self.num_frames as f64 / MINIMAX_H3_FPS
    }

    /// **The time-alignment assertion.**
    ///
    /// Three separate claims, because they fail in different ways:
    ///
    /// 1. the two grids advance the shared rotary clock at the same rate ([`rope_clocks_agree`]) —
    ///    a constants check, so it catches a future edit to either rate;
    /// 2. the latent-frame count really covers `num_frames` frames, i.e.
    ///    `video_rope_span == 5/3 · num_frames`. **This is the drift gate**: it is the only one of
    ///    the three that a wrong `num_latent_frames` fails, and it fails proportionally to the
    ///    clip length rather than at frame 0;
    /// 3. the residual between the two tracks is within the `round`'s half-unit
    ///    ([`MAX_AV_DRIFT_SECONDS`]) — which a wrong `num_audio_latents` fails.
    pub fn validate(&self) -> Result<()> {
        if self.latent_height == 0 || self.latent_width == 0 {
            return Err(CandleError::Msg(format!(
                "minimax-h3 geometry: latent extents must be positive, got {}x{}",
                self.latent_height, self.latent_width
            )));
        }
        if !rope_clocks_agree() {
            return Err(CandleError::Msg(format!(
                "minimax-h3 geometry: the video rotary clock ({MINIMAX_H3_FPS} fps · \
                 {ROPE_FRAME_RESCALE}) and the audio one ({AUDIO_LATENTS_PER_SECOND} latents/s) \
                 must both be {ROPE_UNITS_PER_SECOND} units/s — audio and video share ONE MM-RoPE \
                 time axis"
            )));
        }
        if self.num_latent_frames != video_latent_num_frames(self.num_frames)? {
            return Err(CandleError::Msg(format!(
                "minimax-h3 geometry: {} latent frames for {} frames; 17n + 5 frames is 5n + 2 \
                 latents",
                self.num_latent_frames, self.num_frames
            )));
        }

        let covered = ROPE_FRAME_RESCALE * self.num_frames as f64;
        if (self.video_rope_span() - covered).abs() > 1e-9 {
            return Err(CandleError::Msg(format!(
                "minimax-h3 geometry: {} latent frames span {} rotary units but {} frames are {} \
                 — the video clock has drifted off the shared audio clock",
                self.num_latent_frames,
                self.video_rope_span(),
                self.num_frames,
                covered
            )));
        }

        let drift = self.av_drift_seconds();
        if drift.abs() > MAX_AV_DRIFT_SECONDS + 1e-12 {
            return Err(CandleError::Msg(format!(
                "minimax-h3 geometry: {} audio latents against {} frames drifts {:.4} s, beyond \
                 the {MAX_AV_DRIFT_SECONDS} s the round(num_frames / 24 · 40) can produce",
                self.num_audio_latents, self.num_frames, drift
            )));
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The shared-clock identity, and that both sides are the numbers they claim to be.
    #[test]
    fn audio_and_video_share_one_forty_hertz_rotary_clock() {
        assert!(rope_clocks_agree());
        assert_eq!(MINIMAX_H3_FPS * ROPE_FRAME_RESCALE, 40.0);
        assert_eq!(AUDIO_LATENTS_PER_SECOND, 40);
        assert_eq!(ROPE_UNITS_PER_SECOND, 40.0);
        // One audio latent and one frame's 5/3 are the same amount of rotary time: a latent lasts
        // 1/40 s and a frame 1/24 s, and both convert at 40 units/s.
        assert_eq!(1.0 / f64::from(AUDIO_LATENTS_PER_SECOND), 1.0 / 40.0);
        assert!(
            (ROPE_FRAME_RESCALE / ROPE_UNITS_PER_SECOND - 1.0 / MINIMAX_H3_FPS).abs() < 1e-15,
            "5/3 rotary units must be exactly one frame of time"
        );
    }

    /// `5k + 2` latents cover exactly `17k + 5` frames, for every legal duration.
    #[test]
    fn every_legal_duration_is_time_aligned() {
        for &frames in &LEGAL_FRAME_COUNTS {
            let g = JointGeometry::new(frames, 20, 36).unwrap();
            assert_eq!(g.num_latent_frames, (frames - 5) / 17 * 5 + 2);
            assert!(
                (g.video_rope_span() - ROPE_FRAME_RESCALE * frames as f64).abs() < 1e-9,
                "{frames}: {} != {}",
                g.video_rope_span(),
                ROPE_FRAME_RESCALE * frames as f64
            );
            assert!(g.av_drift_seconds().abs() <= MAX_AV_DRIFT_SECONDS + 1e-12);
        }
        assert_eq!(LEGAL_FRAME_COUNTS[0], 124);
        assert_eq!(LEGAL_FRAME_COUNTS[13], 345);
    }

    /// The advertised envelope is **derived from the lattice**: 5.1667 s to 14.375 s, 14 points.
    /// A flat 15.0 s has no legal frame count (sc-17152).
    #[test]
    fn the_duration_envelope_is_derived_from_the_lattice() {
        let shortest = JointGeometry::new(LEGAL_FRAME_COUNTS[0], 20, 36).unwrap();
        let longest = JointGeometry::new(LEGAL_FRAME_COUNTS[13], 20, 36).unwrap();
        assert!((shortest.duration_seconds() - 5.166_666).abs() < 1e-4);
        assert!((longest.duration_seconds() - 14.375).abs() < 1e-9);
        assert_eq!(LEGAL_FRAME_COUNTS.len(), 14);
        // 15.0 s would be 360 frames, which is not on the lattice and is past the longest legal
        // count — so it is refused, not silently delivered as 14.375.
        assert!(video_latent_num_frames(360).is_err());
        assert!(!LEGAL_FRAME_COUNTS.contains(&align_num_frames(346)));
    }

    /// The measured ±8.33 ms / 0 cycle, reproduced exactly — and the five exactly-aligned durations
    /// named.
    #[test]
    fn the_av_residual_cycles_through_three_values() {
        let ms = |frames: usize| {
            JointGeometry::new(frames, 20, 36)
                .unwrap()
                .av_drift_seconds()
                * 1e3
        };
        assert!((ms(124) - 8.3333).abs() < 1e-3, "124: {}", ms(124));
        assert!(ms(141).abs() < 1e-9, "141: {}", ms(141));
        assert!((ms(158) + 8.3333).abs() < 1e-3, "158: {}", ms(158));

        let aligned: Vec<usize> = LEGAL_FRAME_COUNTS
            .iter()
            .copied()
            .filter(|&f| ms(f).abs() < 1e-9)
            .collect();
        assert_eq!(aligned, vec![141, 192, 243, 294, 345]);

        // 124 frames is 5.1667 s of picture against 207 audio latents = 5.1750 s of sound.
        let g = JointGeometry::new(124, 20, 36).unwrap();
        assert_eq!(g.num_audio_latents, 207);
        assert_eq!(g.num_latent_frames, 37);
        assert!((g.duration_seconds() - 5.16667).abs() < 1e-4);
        assert!((g.audio_rope_span() / ROPE_UNITS_PER_SECOND - 5.175).abs() < 1e-9);
    }

    /// **The exactly-aligned set is derived, not tabulated.** [`av_grids_align_exactly`] is asked
    /// about all fourteen and must agree, count by count, with the residual actually measured off
    /// [`JointGeometry::av_drift_seconds`] — which is what makes it a derivation of the two grid
    /// rates rather than a restatement of the five counts it is true at.
    #[test]
    fn the_exactly_aligned_durations_are_derived_not_tabulated() {
        let mut exact = Vec::new();
        for &frames in &LEGAL_FRAME_COUNTS {
            // The integer statement of "no rounding happened": the rounded latent count, put back
            // on the frame clock, lands exactly on the frame count. Formulated by MULTIPLYING where
            // `av_grids_align_exactly` divides, so the two are not the same expression twice.
            let round_tripped = audio_latent_num_frames(frames) * MINIMAX_H3_FPS as usize
                == frames * AUDIO_LATENTS_PER_SECOND as usize;
            assert_eq!(
                av_grids_align_exactly(frames),
                round_tripped,
                "{frames}: the predicate and the round-trip disagree"
            );
            // ...and the same claim measured off the rotary spans. `video_rope_span` sums 47 f64
            // terms, so "exact" here means under the summation's own noise (~3.6e-15 s at 141), six
            // orders of magnitude below the ±8.33 ms the inexact counts carry.
            let drift = JointGeometry::new(frames, 20, 36)
                .unwrap()
                .av_drift_seconds();
            assert_eq!(
                av_grids_align_exactly(frames),
                drift.abs() < 1e-9,
                "{frames}: the predicate says {} but the measured residual is {drift} s",
                av_grids_align_exactly(frames)
            );
            if av_grids_align_exactly(frames) {
                exact.push(frames);
            }
        }
        assert_eq!(exact, vec![141, 192, 243, 294, 345]);
        // ...and the other NINE are off by the full ±8.33 ms, not by something negligible. Without
        // this the predicate could be true everywhere and the assertion above would still hold.
        assert_eq!(LEGAL_FRAME_COUNTS.len() - exact.len(), 9);
        for &frames in LEGAL_FRAME_COUNTS
            .iter()
            .filter(|&&f| !av_grids_align_exactly(f))
        {
            let ms = JointGeometry::new(frames, 20, 36)
                .unwrap()
                .av_drift_seconds()
                * 1e3;
            assert!((ms.abs() - 8.3333).abs() < 1e-3, "{frames}: {ms} ms");
        }
        // The predicate is the two RATES, so it must answer off the lattice too — `3 | num_frames`
        // is a consequence, never the definition.
        for frames in [3, 6, 24, 48, 120, 360] {
            assert!(av_grids_align_exactly(frames), "{frames}");
        }
        for frames in [1, 2, 4, 5, 7, 100, 121, 361] {
            assert!(!av_grids_align_exactly(frames), "{frames}");
        }
    }

    /// **The AV mux policy, stated in counts.** The soundtrack is DELIVERED at the picture's length
    /// at every one of the fourteen — including the nine where the decoder hands back a different
    /// one — and the residual that survives is under half a sample.
    ///
    /// Held here even though this crate ships no pipeline: sc-17156's delivery path must trim/pad to
    /// [`delivered_audio_samples`], and this is what stops it inventing a second answer.
    #[test]
    fn the_delivered_soundtrack_length_is_the_pictures_at_every_legal_count() {
        assert_eq!(AUDIO_SAMPLES_PER_LATENT, 800);
        let mut corrected = 0;
        for &frames in &LEGAL_FRAME_COUNTS {
            let picture = frames as f64 / MINIMAX_H3_FPS;
            let delivered = delivered_audio_samples(frames);
            let residual = delivered as f64 / f64::from(AUDIO_SAMPLE_RATE) - picture;
            assert!(
                residual.abs() <= MAX_DELIVERED_AV_RESIDUAL_SECONDS,
                "{frames}: the delivered track is {residual:+.9} s off the picture, over the \
                 half-sample bound"
            );

            // What the decoder emits, and therefore what the correction costs.
            let decoded = decoded_audio_samples(frames);
            let correction = (decoded as f64 - delivered as f64) / f64::from(AUDIO_SAMPLE_RATE);
            assert!(
                correction.abs() <= MAX_AV_DRIFT_SECONDS + 1e-12,
                "{frames}: correction {correction:+.5} s exceeds the audio-latent rounding bound"
            );
            assert_eq!(
                decoded == delivered,
                av_grids_align_exactly(frames),
                "{frames}: whether a correction is needed IS whether the grids align"
            );
            if decoded != delivered {
                corrected += 1;
                // The correction is the audio-latent residual (±8.3333 ms) plus the sub-sample
                // residual `delivered`'s own round leaves — at 124 that is 267 samples = 8.34375 ms.
                assert!(
                    (correction.abs() - 8.3333e-3).abs() <= MAX_DELIVERED_AV_RESIDUAL_SECONDS,
                    "{frames}: {correction} s is not the ±8.3333 ms residual within a half-sample"
                );
            }
        }
        assert_eq!(corrected, 9, "nine of the fourteen need a correction");
        // The measured pair from the spike: 124 frames is 5.1667 s of picture, 207 latents =
        // 165 600 decoded samples = 5.175 s of sound, delivered as 165 333.
        assert_eq!(decoded_audio_samples(124), 165_600);
        assert_eq!(delivered_audio_samples(124), 165_333);
        assert_eq!(decoded_audio_samples(141), delivered_audio_samples(141));
    }

    /// **The drift gate.** A `num_latent_frames` derived the obvious wrong way is still a positive
    /// integer that builds a runnable sequence — and it desynchronizes the two tracks
    /// proportionally to the clip length. `validate` must reject it.
    #[test]
    fn a_wrong_latent_frame_count_is_rejected_as_drift() {
        let good = JointGeometry::new(124, 20, 36).unwrap();
        // The plausible wrong derivations: 4x temporal compression, and ceil of it.
        for wrong in [31, 32, 36, 38, 41] {
            let g = JointGeometry {
                num_latent_frames: wrong,
                ..good
            };
            let e = g.validate().unwrap_err().to_string();
            assert!(
                e.contains("latent"),
                "{wrong} latent frames must be rejected, got: {e}"
            );
        }
        // ...and the magnitude of what it would have cost: 31 latents cover 4/5 of the frames.
        let drifted = JointGeometry {
            num_latent_frames: 31,
            ..good
        };
        let seconds = (drifted.video_rope_span() - good.video_rope_span()) / ROPE_UNITS_PER_SECOND;
        assert!(
            seconds < -0.8,
            "a 4x-compression guess loses {seconds:.3} s of picture against the soundtrack"
        );
    }

    /// A frame count that is not `17n + 5` has no latent count at all.
    #[test]
    fn illegal_frame_counts_are_rejected() {
        for bad in [0, 1, 4, 121, 123, 125, 140, 200] {
            assert!(
                video_latent_num_frames(bad).is_err(),
                "{bad} must not be a legal frame count"
            );
        }
        for &good in &LEGAL_FRAME_COUNTS {
            video_latent_num_frames(good).unwrap();
        }
        assert!(JointGeometry::new(124, 0, 36).is_err(), "zero extent");
        assert!(JointGeometry::new(124, 20, 0).is_err(), "zero extent");
    }

    /// `align_num_frames` walks up to the next legal count and is idempotent on a legal one.
    #[test]
    fn align_walks_up_to_the_next_legal_count() {
        assert_eq!(align_num_frames(124), 124);
        assert_eq!(align_num_frames(125), 141);
        assert_eq!(align_num_frames(141), 141);
        assert_eq!(align_num_frames(142), 158);
        for &f in &LEGAL_FRAME_COUNTS {
            assert_eq!(align_num_frames(f), f);
            video_latent_num_frames(align_num_frames(f - 1)).unwrap();
        }
    }

    /// A wrong audio-token count fails the residual bound rather than passing as a short track.
    #[test]
    fn a_wrong_audio_latent_count_is_rejected() {
        let good = JointGeometry::new(124, 20, 36).unwrap();
        assert_eq!(audio_latent_num_frames(124), 207);
        for wrong in [200, 206, 208, 248] {
            let g = JointGeometry {
                num_audio_latents: wrong,
                ..good
            };
            assert!(
                g.validate().is_err(),
                "{wrong} audio latents must be rejected"
            );
        }
    }
}
