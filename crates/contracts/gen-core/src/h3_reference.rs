//! **MiniMax-H3 `ref2va` reference preprocessing shared by both backends** (sc-26691).
//!
//! The candle and MLX ports, and the gen-core admission pricers ([`crate::veda::workspace`] and
//! [`crate::h3_conditioning_memory`]), all have to agree on two facts about a reference *before* it
//! is encoded: the size an **image** reference is resized to, and how many samples of a reference
//! **soundtrack** survive. Both are pure arithmetic, so they live here once rather than as four
//! copies that drift.
//!
//! # Upstream sources
//!
//! * sglang `multimodal_gen/.../minimax_h3/reference_encoding.py`
//!   (`minimax_h3_resolve_reference_image_shape`, `_nearest_multiple`) and `constants.py`
//!   (`MINIMAX_H3_REFERENCE_RESIZE_MODES`) @ `e0df901d` — the three resize modes.
//! * diffusers `modular_pipelines/minimax_h3/before_encoder.py` @ `cda991a6` — the `diffusers` mode
//!   (force the short edge to 2048, upscaling included) and `_normalize_audio_condition`'s
//!   `waveform[:, : int(max_duration * sample_rate)]` with `max_duration = num_frames / fps`.
//! * ComfyUI `comfy_extras/nodes_minimax_h3.py` @ `79cf64c2` defaults to `match`, and ModelTC
//!   distilled the Ref2VA Turbo LoRA with `match`, which is why [`ReferenceImageResizeMode`]'s
//!   default is [`Match`](ReferenceImageResizeMode::Match) here (sglang keeps `diffusers` as its own
//!   compatibility default).
//!
//! Only `image.reference_preserve` materials are resized by the mode. First/last-frame keyframes
//! (`image.target_canvas` upstream) are stretched onto the target canvas and video references keep
//! the shared canvas rule; neither reads the mode.

use crate::media::AudioTrack;
use crate::{Error, Result};
use serde::{Deserialize, Serialize};

/// The short edge `diffusers` forces and `max` caps an image reference at — upstream's
/// `MINIMAX_H3_REFERENCE_IMAGE_SHORT_EDGE` / `reference_image_short_edge` default.
pub const REFERENCE_IMAGE_BASE_SHORT_EDGE: u32 = 2048;

/// The lattice every resolved reference side is rounded onto (upstream
/// `MINIMAX_H3_REFERENCE_IMAGE_MULTIPLE`): the video VAE's 16x stride times the DiT's 2x2 patch.
pub const REFERENCE_IMAGE_MULTIPLE: u32 = 32;

/// The widest admitted reference aspect, inclusive on both sides (1:4 to 4:1).
pub const REFERENCE_IMAGE_MAX_ASPECT: f64 = 4.0;

/// **How an image reference is resized before it is encoded** (sglang `reference_resize_mode`).
///
/// Every mode preserves the reference's own aspect ratio, never crops, and rounds each side
/// independently to the nearest multiple of [`REFERENCE_IMAGE_MULTIPLE`] (Python `round`, i.e.
/// half-to-even, floored at one multiple):
///
/// | mode | scale |
/// |---|---|
/// | [`Match`](Self::Match) | `min(1, sqrt(target_w · target_h / (w · h)))` |
/// | [`Max`](Self::Max) | `min(1, short_edge / min(w, h))` |
/// | [`Diffusers`](Self::Diffusers) | `short_edge / min(w, h)` (upscales) |
///
/// `short_edge` is the request's effective
/// [`reference_image_short_edge`](crate::GenerationRequest::reference_image_short_edge) —
/// [`REFERENCE_IMAGE_BASE_SHORT_EDGE`] unless the caller lowered it. `Match` does not read it.
///
/// Serde form: lowercase strings `"match"` / `"max"` / `"diffusers"`, upstream's own spelling.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum ReferenceImageResizeMode {
    /// Match the reference's pixel area to the target canvas, never upscaling. The default: the
    /// policy ComfyUI ships and the ModelTC Turbo LoRA was distilled with.
    #[default]
    Match,
    /// Only downsize a reference whose short edge exceeds `short_edge`.
    Max,
    /// Always scale the short edge to `short_edge`, upscaling smaller inputs — the original
    /// diffusers / sglang behaviour.
    Diffusers,
}

impl ReferenceImageResizeMode {
    /// Every mode, in upstream's documentation order.
    pub const ALL: [Self; 3] = [Self::Match, Self::Max, Self::Diffusers];

    /// Upstream's spelling, identical to the serde form.
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Match => "match",
            Self::Max => "max",
            Self::Diffusers => "diffusers",
        }
    }
}

impl std::fmt::Display for ReferenceImageResizeMode {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(self.as_str())
    }
}

impl std::str::FromStr for ReferenceImageResizeMode {
    type Err = Error;

    /// Case-insensitive and whitespace-trimmed, as upstream's `str(mode).strip().lower()`.
    fn from_str(s: &str) -> Result<Self> {
        let wanted = s.trim().to_ascii_lowercase();
        Self::ALL
            .into_iter()
            .find(|m| m.as_str() == wanted)
            .ok_or_else(|| {
                Error::Msg(format!(
                    "reference_image_resize_mode must be one of diffusers, match, max, got {s:?}"
                ))
            })
    }
}

/// The **effective** resize mode of one request: the requested one, or
/// [`ReferenceImageResizeMode::Match`] when it omits one.
pub fn effective_reference_image_resize_mode(
    req: &crate::GenerationRequest,
) -> ReferenceImageResizeMode {
    req.reference_image_resize_mode.unwrap_or_default()
}

/// Upstream `_nearest_multiple`: `max(multiple, int(round(value / multiple)) * multiple)`.
fn nearest_multiple(value: f64, multiple: f64) -> f64 {
    ((value / multiple).round_ties_even() * multiple).max(multiple)
}

/// **The `(width, height)` an image reference is resized to** — upstream
/// `minimax_h3_resolve_reference_image_shape`, bit for bit.
///
/// `target_width` x `target_height` is the request's resolved canvas (only
/// [`Match`](ReferenceImageResizeMode::Match) reads it); `short_edge` is the effective
/// [`reference_image_short_edge`](crate::GenerationRequest::reference_image_short_edge) (only `max`
/// and `diffusers` read it); `multiple` is the lattice, [`REFERENCE_IMAGE_MULTIPLE`] for the
/// released checkpoint.
///
/// Refuses a zero extent, a zero canvas, a zero `short_edge` / `multiple`, and an aspect outside
/// 1:4..=4:1 — the same refusals upstream raises before any pixel is touched.
pub fn resolve_reference_image_size(
    width: u32,
    height: u32,
    target_width: u32,
    target_height: u32,
    mode: ReferenceImageResizeMode,
    short_edge: u32,
    multiple: u32,
) -> Result<(u32, u32)> {
    if width == 0 || height == 0 {
        return Err(Error::Msg(format!(
            "minimax-h3 ref2va: a reference image must have a positive size, got {width}x{height}"
        )));
    }
    if target_width == 0 || target_height == 0 {
        return Err(Error::Msg(format!(
            "minimax-h3 ref2va: the target canvas must have a positive size, got \
             {target_width}x{target_height}"
        )));
    }
    if short_edge == 0 || multiple == 0 {
        return Err(Error::Msg(format!(
            "minimax-h3 ref2va: reference short edge ({short_edge}) and lattice ({multiple}) \
             must be positive"
        )));
    }
    let (w, h) = (f64::from(width), f64::from(height));
    if w > REFERENCE_IMAGE_MAX_ASPECT * h || h > REFERENCE_IMAGE_MAX_ASPECT * w {
        return Err(Error::Msg(format!(
            "minimax-h3 ref2va: reference image ratio must be within the inclusive range 1:4 to \
             4:1, got {width}x{height}"
        )));
    }
    let short = w.min(h);
    let scale = match mode {
        ReferenceImageResizeMode::Match => {
            let (cw, ch) = (f64::from(target_width), f64::from(target_height));
            1.0f64.min(((cw * ch) / (w * h)).sqrt())
        }
        ReferenceImageResizeMode::Max => 1.0f64.min(f64::from(short_edge) / short),
        ReferenceImageResizeMode::Diffusers => f64::from(short_edge) / short,
    };
    let m = f64::from(multiple);
    let side = |v: f64| -> Result<u32> {
        let r = nearest_multiple(v * scale, m);
        if !r.is_finite() || r > f64::from(u32::MAX) {
            return Err(Error::Msg(format!(
                "minimax-h3 ref2va: reference image {width}x{height} resolves out of range"
            )));
        }
        Ok(r as u32)
    };
    Ok((side(w)?, side(h)?))
}

/// **Samples per channel a reference soundtrack keeps** — diffusers'
/// `int(max_duration * sample_rate)` with `max_duration = num_frames / fps`, the same bound sglang
/// hands `ffmpeg -t` (`audio_encoding.py`: `frame_count / fps`).
///
/// Applies to a standalone audio reference *and* a video reference's own soundtrack.
pub fn reference_audio_max_samples_per_channel(
    num_frames: usize,
    fps: f64,
    sample_rate: u32,
) -> usize {
    let max_duration = num_frames as f64 / fps;
    let kept = max_duration * f64::from(sample_rate);
    if kept.is_finite() && kept > 0.0 {
        kept as usize
    } else {
        0
    }
}

/// Truncate a reference soundtrack to the generated duration (see
/// [`reference_audio_max_samples_per_channel`]). Interleaved PCM is cut on a whole sample frame;
/// a track already within the bound is returned unchanged, and stems are cut the same way.
pub fn truncate_reference_audio(track: &AudioTrack, num_frames: usize, fps: f64) -> AudioTrack {
    let per_channel = reference_audio_max_samples_per_channel(num_frames, fps, track.sample_rate);
    let keep = per_channel.saturating_mul(usize::from(track.channels.max(1)));
    let mut out = track.clone();
    out.samples.truncate(keep);
    for stem in &mut out.stems {
        stem.samples.truncate(keep);
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use ReferenceImageResizeMode::{Diffusers, Match, Max};

    fn resolve(w: u32, h: u32, tw: u32, th: u32, mode: ReferenceImageResizeMode) -> (u32, u32) {
        resolve_reference_image_size(
            w,
            h,
            tw,
            th,
            mode,
            REFERENCE_IMAGE_BASE_SHORT_EDGE,
            REFERENCE_IMAGE_MULTIPLE,
        )
        .unwrap()
    }

    #[test]
    fn default_mode_is_match_and_serde_uses_upstream_spelling() {
        assert_eq!(ReferenceImageResizeMode::default(), Match);
        assert_eq!(
            effective_reference_image_resize_mode(&crate::GenerationRequest::default()),
            Match
        );
        for mode in ReferenceImageResizeMode::ALL {
            let json = serde_json::to_string(&mode).unwrap();
            assert_eq!(json, format!("\"{}\"", mode.as_str()));
            assert_eq!(
                serde_json::from_str::<ReferenceImageResizeMode>(&json).unwrap(),
                mode
            );
            assert_eq!(
                format!(" {} ", mode.as_str().to_uppercase())
                    .parse::<ReferenceImageResizeMode>()
                    .unwrap(),
                mode
            );
        }
        assert!("stretch".parse::<ReferenceImageResizeMode>().is_err());
    }

    /// The mode arithmetic table, against values computed from upstream's formula by hand.
    #[test]
    fn mode_arithmetic_table() {
        // (ref w, ref h, target w, target h, mode) -> resolved (w, h)
        type Row = (u32, u32, u32, u32, ReferenceImageResizeMode, (u32, u32));
        let table: &[Row] = &[
            // The film-harness plate against a 1088x544 render: area 184_320 < 591_872, so neither
            // match nor max upscales; diffusers forces 2048 on the short edge (576 * 6.4 = 3686.4
            // -> 115.2 multiples -> 3680).
            (576, 320, 1088, 544, Match, (576, 320)),
            (576, 320, 1088, 544, Max, (576, 320)),
            (576, 320, 1088, 544, Diffusers, (3680, 2048)),
            // A 4K still against the same render: match lands on the target area
            // (sqrt(591_872 / 8_294_400) = 0.26713 -> 1025.8 / 577.0 -> 1024 / 576), max caps the
            // short edge at 2048 (3840 * 0.9481 = 3640.9 -> 3648), diffusers equals max here.
            (3840, 2160, 1088, 544, Match, (1024, 576)),
            (3840, 2160, 1088, 544, Max, (3648, 2048)),
            (3840, 2160, 1088, 544, Diffusers, (3648, 2048)),
            // Portrait 1024x1536 against 1376x768: area ratio 0.672 -> scale 0.8198 -> 839.5 /
            // 1259.2 -> 832 / 1248. Below the 2048 cap so max keeps it; diffusers upscales x2.
            (1024, 1536, 1376, 768, Match, (832, 1248)),
            (1024, 1536, 1376, 768, Max, (1024, 1536)),
            (1024, 1536, 1376, 768, Diffusers, (2048, 3072)),
        ];
        for &(w, h, tw, th, mode, want) in table {
            assert_eq!(
                resolve(w, h, tw, th, mode),
                want,
                "{w}x{h} -> {tw}x{th} {mode}"
            );
        }
    }

    #[test]
    fn rounding_is_nearest_half_to_even_with_a_one_multiple_floor() {
        // 48 / 32 = 1.5 -> 2 (even), 80 / 32 = 2.5 -> 2 (even), 112 / 32 = 3.5 -> 4 (even).
        assert_eq!(resolve(48, 80, 4096, 4096, Max), (64, 64));
        assert_eq!(resolve(112, 112, 4096, 4096, Match), (128, 128));
        // 47 / 32 = 1.47 -> 1; anything under half a multiple still floors at one multiple.
        assert_eq!(resolve(47, 47, 4096, 4096, Match), (32, 32));
        assert_eq!(resolve(10, 10, 4096, 4096, Max), (32, 32));
        // Every resolved side is on the lattice.
        for mode in ReferenceImageResizeMode::ALL {
            let (w, h) = resolve(1001, 777, 1088, 544, mode);
            assert_eq!((w % 32, h % 32), (0, 0), "{mode}");
        }
        // A lowered short edge feeds max and diffusers, never match.
        let lowered =
            |mode| resolve_reference_image_size(3840, 2160, 1088, 544, mode, 1024, 32).unwrap();
        assert_eq!(lowered(Diffusers), (1824, 1024));
        assert_eq!(lowered(Max), (1824, 1024));
        assert_eq!(lowered(Match), resolve(3840, 2160, 1088, 544, Match));
    }

    #[test]
    fn aspect_beyond_four_to_one_is_refused_inclusive_boundary_admitted() {
        for mode in ReferenceImageResizeMode::ALL {
            assert!(resolve_reference_image_size(400, 100, 1088, 544, mode, 2048, 32).is_ok());
            assert!(resolve_reference_image_size(100, 400, 1088, 544, mode, 2048, 32).is_ok());
            for (w, h) in [(401, 100), (100, 401)] {
                let e = resolve_reference_image_size(w, h, 1088, 544, mode, 2048, 32)
                    .unwrap_err()
                    .to_string();
                assert!(e.contains("1:4 to 4:1"), "{e}");
            }
        }
        assert!(resolve_reference_image_size(0, 10, 1088, 544, Match, 2048, 32).is_err());
        assert!(resolve_reference_image_size(10, 10, 0, 544, Match, 2048, 32).is_err());
        assert!(resolve_reference_image_size(10, 10, 1088, 544, Max, 0, 32).is_err());
    }

    #[test]
    fn audio_is_truncated_to_the_generated_duration() {
        // 124 frames at 24 fps = 5.1666.. s; int(5.1666.. * 32000) = 165_333.
        assert_eq!(
            reference_audio_max_samples_per_channel(124, 24.0, 32_000),
            165_333
        );
        // 241 frames = 10.041666.. s -> 321_333.33 -> 321_333.
        assert_eq!(
            reference_audio_max_samples_per_channel(241, 24.0, 32_000),
            321_333
        );
        // Interleaved stereo: a 30 s track keeps 165_333 sample frames = 330_666 samples.
        let long = AudioTrack {
            samples: vec![0.5; 30 * 32_000 * 2],
            sample_rate: 32_000,
            channels: 2,
            stems: Vec::new(),
        };
        let cut = truncate_reference_audio(&long, 124, 24.0);
        assert_eq!(cut.samples.len(), 165_333 * 2);
        assert_eq!((cut.sample_rate, cut.channels), (32_000, 2));
        // Mono, and a track already shorter than the bound, pass through untouched.
        let short = AudioTrack {
            samples: vec![0.25; 32_000],
            sample_rate: 32_000,
            channels: 1,
            stems: Vec::new(),
        };
        assert_eq!(
            truncate_reference_audio(&short, 124, 24.0).samples.len(),
            32_000
        );
        let mono_long = AudioTrack {
            channels: 1,
            samples: vec![0.0; 10 * 32_000],
            ..short
        };
        assert_eq!(
            truncate_reference_audio(&mono_long, 124, 24.0)
                .samples
                .len(),
            165_333
        );
    }
}
