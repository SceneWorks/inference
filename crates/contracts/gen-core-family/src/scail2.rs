//! SCAIL-2 source windows for the registered providers and admission.
use std::ops::Range;

use gen_core::render::video::{VideoDenoisePhase, VideoModelRule};
use gen_core::render::RenderPhaseIdentity;
use gen_core::{Error, MemoryGeometry, Result};

pub const SEGMENT_LEN: usize = 81;
pub const SEGMENT_OVERLAP: usize = 5;
const TEMPORAL_STRIDE: usize = 4;

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum SegmentTail {
    /// MLX emits full windows after the first and drops an incomplete trailing window.
    FullWindows,
    /// Candle also emits a shortened causal window if it adds frames beyond the overlap.
    AlignedRemainder,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct Scail2Segment {
    /// Identical pixel-frame coordinates into the driving clip and its masks.
    pub source: Range<usize>,
    /// Rendered output geometry; driving poses and their masks use half its width and height.
    pub identity: RenderPhaseIdentity,
    pub latent_frames: u32,
    /// Decoded pixel frames retained from the preceding window and skipped when stitching.
    pub history_frames: usize,
}

impl Scail2Segment {
    pub fn denoise_phase<P>(
        &self,
        solver: P,
        sigmas: Vec<f32>,
        request_seed: u64,
    ) -> VideoDenoisePhase<(), P> {
        VideoDenoisePhase {
            identity: self.identity,
            latent_frames: self.latent_frames,
            solver,
            sigmas,
            seed: request_seed.wrapping_add(self.identity.pass as u64),
            models: VideoModelRule::Fixed(()),
        }
    }
}

fn aligned_frames(frames: usize) -> usize {
    if frames == 0 {
        0
    } else {
        1 + (frames - 1) / TEMPORAL_STRIDE * TEMPORAL_STRIDE
    }
}

/// Resolve the existing backend's physical windows without reading pixels or weights.
/// `geometry` is the resolved output lattice, actual source-frame count and number of
/// consumed character images (the paired masks do not add references). The primary image
/// uses the full spatial footprint and each driving frame uses the same source index as its mask.
pub fn resolve_segments(
    tail: SegmentTail,
    geometry: MemoryGeometry,
    window: usize,
    overlap: usize,
) -> Result<Vec<Scail2Segment>> {
    if geometry.width == 0
        || geometry.height == 0
        || !geometry.width.is_multiple_of(32)
        || !geometry.height.is_multiple_of(32)
        || geometry.batch != 1
        || geometry.reference_count == 0
        || geometry.frames == 0
    {
        return Err(Error::Unsupported(
            "scail2: invalid resolved output geometry or empty source".into(),
        ));
    }
    let total = geometry.frames as usize;
    // Short MLX clips never consult the stride or overlap. Candle validates both on every call.
    if total > window || tail == SegmentTail::AlignedRemainder {
        if window == 0 {
            return Err(Error::Unsupported(
                "scail2: segment_len must be >= 1".into(),
            ));
        }
        if overlap >= window {
            return Err(Error::Unsupported(format!(
                "scail2: segment_overlap ({overlap}) must be < segment_len ({window})"
            )));
        }
    }
    if total > window && !(window - 1).is_multiple_of(TEMPORAL_STRIDE) {
        return Err(Error::Unsupported(format!("scail2: segment_len ({window}) must be 1 + {TEMPORAL_STRIDE}·k for windowed driving masks")));
    }

    let mut segments = Vec::new();
    let mut push = |start: usize, frames: usize| {
        segments.push(Scail2Segment {
            source: start..start + frames,
            history_frames: if segments.is_empty() { 0 } else { overlap },
            identity: RenderPhaseIdentity {
                pass: segments.len(),
                phase: 0,
                geometry: MemoryGeometry {
                    frames: frames as u32,
                    ..geometry
                },
            },
            latent_frames: (1 + (frames - 1) / TEMPORAL_STRIDE) as u32,
        });
    };
    if total <= window {
        push(0, aligned_frames(total));
    } else {
        let mut start = 0;
        while total - start >= window {
            push(start, window);
            start += window - overlap;
        }
        let remaining = aligned_frames(total - start);
        if tail == SegmentTail::AlignedRemainder && remaining > overlap {
            push(start, remaining);
        }
    }
    // Candle's causal VAE accepts exactly 1+4k input frames. A single segment never
    // encodes history, and independent (zero-overlap) segments have no history.
    if tail == SegmentTail::AlignedRemainder
        && segments.len() > 1
        && overlap > 0
        && !(overlap - 1).is_multiple_of(TEMPORAL_STRIDE)
    {
        return Err(Error::Unsupported(format!("scail2: Candle clean history requires segment_overlap ({overlap}) to be zero or 1 + {TEMPORAL_STRIDE}·k")));
    }
    Ok(segments)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn clean_history_is_present_only_for_reencodable_overlaps() {
        let plan = |tail, frames, overlap| {
            resolve_segments(
                tail,
                MemoryGeometry {
                    width: 64,
                    height: 32,
                    frames,
                    batch: 1,
                    reference_count: 1,
                },
                9,
                overlap,
            )
        };
        let candle = SegmentTail::AlignedRemainder;
        let mlx = SegmentTail::FullWindows;
        assert!(plan(candle, 17, 4)
            .unwrap_err()
            .to_string()
            .contains("clean history"));
        assert_eq!(plan(candle, 9, 4).unwrap()[0].history_frames, 0);
        assert_eq!(plan(mlx, 17, 4).unwrap()[1].history_frames, 4);
        for tail in [mlx, candle] {
            let independent = plan(tail, 18, 0).unwrap();
            assert_eq!(independent.len(), 2);
            assert_eq!(independent[1].source, 9..18);
            assert!(independent.iter().all(|s| s.history_frames == 0));
            let joined = plan(tail, 17, 5).unwrap();
            assert_eq!(joined[0].history_frames, 0);
            assert_eq!(joined[1].history_frames, 5);
        }
    }

    #[test]
    fn vae_align_rounds_down_to_4k_plus_1() {
        for (input, output) in [
            (0, 0),
            (1, 1),
            (4, 1),
            (5, 5),
            (8, 5),
            (9, 9),
            (81, 81),
            (200, 197),
        ] {
            assert_eq!(aligned_frames(input), output);
        }
    }
}
