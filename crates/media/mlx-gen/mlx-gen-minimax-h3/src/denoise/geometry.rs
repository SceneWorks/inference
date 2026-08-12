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
//! delivered mp4 needs a pad/trim policy — and is recorded here so the pipeline slice inherits a
//! measured number rather than rediscovering it.

use mlx_gen::{Error, Result};

use crate::audio_config::AUDIO_TOKEN_RATE_HZ;
use crate::dit::positions::{ROPE_FRAMES_PER_LATENT, ROPE_FRAME_RESCALE};

/// Frame rate every MiniMax-H3 duration is expressed at. `MINIMAX_H3_FPS`.
pub const MINIMAX_H3_FPS: f64 = 24.0;

/// Audio latent tokens per second. `MINIMAX_H3_AUDIO_LATENTS_PER_SECOND`, and the same 40 Hz
/// [`AUDIO_TOKEN_RATE_HZ`] the audio VAE's 800-sample hop at 32 kHz implies.
pub const AUDIO_LATENTS_PER_SECOND: i32 = AUDIO_TOKEN_RATE_HZ as i32;

/// Frames one temporal chunk covers. `frames_per_chunk`.
pub const FRAMES_PER_CHUNK: i32 = 17;

/// Latent frames one temporal chunk produces. `latents_per_chunk`.
pub const LATENTS_PER_CHUNK: i32 = 5;

/// **The shared MM-RoPE clock rate**: rotary units per second, identical for both modalities.
///
/// `24 fps · 5/3` for video and `40 latents/s · 1` for audio. See [`rope_clocks_agree`].
pub const ROPE_UNITS_PER_SECOND: f64 = 40.0;

/// The worst AV length residual the `round` in [`audio_latent_num_frames`] can produce: half a
/// rotary unit at [`ROPE_UNITS_PER_SECOND`], i.e. 12.5 ms.
pub const MAX_AV_DRIFT_SECONDS: f64 = 0.5 / ROPE_UNITS_PER_SECOND;

/// Every frame count the released model accepts — `17n + 5` clamped to the hardcoded 5–15 s
/// duration range. Nothing else is legal (sc-17241).
pub const LEGAL_FRAME_COUNTS: [i32; 14] = [
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
pub fn align_num_frames(num_frames: i32) -> i32 {
    let mut n = num_frames;
    while n.rem_euclid(FRAMES_PER_CHUNK) != LATENTS_PER_CHUNK {
        n += 1;
    }
    n
}

/// Video latent frames for a legal frame count: `17n + 5` ⇒ `5n + 2`. `video_latent_num_frames`.
///
/// Rejects a frame count that is not `17n + 5`, because every wrong-but-runnable derivation of this
/// number is exactly the drift [`JointGeometry::validate`] exists to catch.
pub fn video_latent_num_frames(num_frames: i32) -> Result<i32> {
    if num_frames < LATENTS_PER_CHUNK
        || num_frames.rem_euclid(FRAMES_PER_CHUNK) != LATENTS_PER_CHUNK
    {
        return Err(Error::Msg(format!(
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
pub fn audio_latent_num_frames(num_frames: i32) -> i32 {
    (f64::from(num_frames) / MINIMAX_H3_FPS * f64::from(AUDIO_LATENTS_PER_SECOND)).round() as i32
}

/// The frame / latent / audio-token counts one joint request is denoised at.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct JointGeometry {
    /// Output frames — always `17n + 5`.
    pub num_frames: i32,
    /// Video latent frames — always `5n + 2`.
    pub num_latent_frames: i32,
    /// Audio latent tokens **per channel**.
    pub num_audio_latents: i32,
    /// Latent height, a multiple of the patch.
    pub latent_height: i32,
    /// Latent width, a multiple of the patch.
    pub latent_width: i32,
}

impl JointGeometry {
    /// Derive every count from the frame geometry, then [`Self::validate`].
    pub fn new(num_frames: i32, latent_height: i32, latent_width: i32) -> Result<Self> {
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
            .map(|i| {
                ROPE_FRAME_RESCALE
                    * ROPE_FRAMES_PER_LATENT[(i as usize) % ROPE_FRAMES_PER_LATENT.len()]
            })
            .sum()
    }

    /// Rotary units the audio track spans: one per latent token.
    pub fn audio_rope_span(&self) -> f64 {
        f64::from(self.num_audio_latents)
    }

    /// Signed AV length residual in seconds — positive when the soundtrack outlasts the picture.
    ///
    /// This is `(round(x) − x) / 40` and is bounded by [`MAX_AV_DRIFT_SECONDS`]; it is the
    /// ±8.33 ms / 0 cycle in the module docs, and a muxing policy input, not a defect.
    pub fn av_drift_seconds(&self) -> f64 {
        (self.audio_rope_span() - self.video_rope_span()) / ROPE_UNITS_PER_SECOND
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
        if self.latent_height <= 0 || self.latent_width <= 0 {
            return Err(Error::Msg(format!(
                "minimax-h3 geometry: latent extents must be positive, got {}x{}",
                self.latent_height, self.latent_width
            )));
        }
        if !rope_clocks_agree() {
            return Err(Error::Msg(format!(
                "minimax-h3 geometry: the video rotary clock ({} fps · {ROPE_FRAME_RESCALE}) and \
                 the audio one ({AUDIO_LATENTS_PER_SECOND} latents/s) must both be \
                 {ROPE_UNITS_PER_SECOND} units/s — audio and video share ONE MM-RoPE time axis",
                MINIMAX_H3_FPS
            )));
        }
        if self.num_latent_frames != video_latent_num_frames(self.num_frames)? {
            return Err(Error::Msg(format!(
                "minimax-h3 geometry: {} latent frames for {} frames; 17n + 5 frames is 5n + 2 \
                 latents",
                self.num_latent_frames, self.num_frames
            )));
        }

        let covered = ROPE_FRAME_RESCALE * f64::from(self.num_frames);
        if (self.video_rope_span() - covered).abs() > 1e-9 {
            return Err(Error::Msg(format!(
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
            return Err(Error::Msg(format!(
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
                (g.video_rope_span() - ROPE_FRAME_RESCALE * f64::from(frames)).abs() < 1e-9,
                "{frames}: {} != {}",
                g.video_rope_span(),
                ROPE_FRAME_RESCALE * f64::from(frames)
            );
            assert!(g.av_drift_seconds().abs() <= MAX_AV_DRIFT_SECONDS + 1e-12);
        }
        assert_eq!(LEGAL_FRAME_COUNTS[0], 124);
        assert_eq!(LEGAL_FRAME_COUNTS[13], 345);
    }

    /// The measured ±8.33 ms / 0 cycle from the spike's real render, reproduced exactly — and the
    /// five exactly-aligned durations named.
    #[test]
    fn the_av_residual_cycles_through_three_values() {
        let ms = |frames: i32| {
            JointGeometry::new(frames, 20, 36)
                .unwrap()
                .av_drift_seconds()
                * 1e3
        };
        assert!((ms(124) - 8.3333).abs() < 1e-3, "124: {}", ms(124));
        assert!(ms(141).abs() < 1e-9, "141: {}", ms(141));
        assert!((ms(158) + 8.3333).abs() < 1e-3, "158: {}", ms(158));

        let aligned: Vec<i32> = LEGAL_FRAME_COUNTS
            .iter()
            .copied()
            .filter(|&f| ms(f).abs() < 1e-9)
            .collect();
        assert_eq!(aligned, vec![141, 192, 243, 294, 345]);

        // 124 frames is 5.1667 s of picture against 207 audio latents = 5.1750 s of sound.
        let g = JointGeometry::new(124, 20, 36).unwrap();
        assert_eq!(g.num_audio_latents, 207);
        assert_eq!(g.num_latent_frames, 37);
        assert!((f64::from(g.num_frames) / MINIMAX_H3_FPS - 5.16667).abs() < 1e-4);
        assert!((g.audio_rope_span() / ROPE_UNITS_PER_SECOND - 5.175).abs() < 1e-9);
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
