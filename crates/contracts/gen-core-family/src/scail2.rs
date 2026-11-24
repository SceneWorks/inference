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

/// Describe the registered render, including the physical frames returned by each z16 decoder.
/// Both providers materialize one f32 noise tensor on the 16-channel, spatially 8x-compressed lattice.
pub fn video_layout(
    tail: SegmentTail,
    domain: gen_core::MemoryBudgetDomain,
    size: (u32, u32),
    character_count: usize,
    driving: gen_core::ControlClipRef<'_>,
) -> Result<gen_core::VideoPlanLayout> {
    if driving.frames.len() != driving.mask.len() {
        return Err(Error::Unsupported(
            "scail2: driving frames need an equal number of masks".into(),
        ));
    }
    let geometry = MemoryGeometry {
        width: size.0,
        height: size.1,
        frames: u32::try_from(driving.frames.len())
            .map_err(|_| Error::Unsupported("scail2: too many driving frames".into()))?,
        batch: 1,
        reference_count: u32::try_from(character_count)
            .map_err(|_| Error::Unsupported("scail2: too many character references".into()))?,
    };
    let segments = resolve_segments(tail, geometry, SEGMENT_LEN, SEGMENT_OVERLAP)?;
    let admission = segments[0].identity.geometry;
    let last = segments.last().unwrap();
    let consumed_frames = last.source.end as u32;
    let final_history_frames = last.history_frames as u64;
    let mut phases = Vec::with_capacity(segments.len());
    let mut output_frames = 0u32;
    let mut retained_rgb_frames = 0u64;
    let spatial_area = u64::from(size.0) * u64::from(size.1);
    let latent_area = spatial_area / (8 * 8);
    let overflow = || Error::Unsupported("scail2: video buffer size overflow".into());
    for segment in segments {
        let causal_frames = segment.identity.geometry.frames;
        let (decoded_frames, decode_chunk_frames) = match tail {
            SegmentTail::FullWindows => {
                // MLX decodes four frames per latent and stitches all of them, including its
                // extra three frames per window. Quality-overlap tiling can fall back to this
                // complete window depending on the live budget.
                let frames = segment.latent_frames * 4;
                (frames, frames)
            }
            SegmentTail::AlignedRemainder => (causal_frames, causal_frames.min(4)),
        };
        output_frames = output_frames
            .checked_add(decoded_frames - segment.history_frames as u32)
            .ok_or_else(|| Error::Unsupported("scail2: stitched frame count overflow".into()))?;
        let frames = u64::from(segment.latent_frames);
        let latent_bytes = latent_area
            .checked_mul(frames * 16 * 4)
            .ok_or_else(overflow)?;
        // Native forward inputs are f32: one 16-channel latent + 28 mask channels per
        // character, 28 full-grid null/noisy channels, and a 16+28 channel pose/mask at
        // half width and height. Subsequent windows also pin a clean latent and mask.
        let mut channels = 44 * u64::from(geometry.reference_count) + 28 * frames + 44 * frames / 4;
        if segment.history_frames > 0 {
            channels += 16 * (1 + (segment.history_frames as u64 - 1) / 4) + 4 * frames;
        }
        let conditioning_bytes = latent_area.checked_mul(channels * 4).ok_or_else(overflow)?;
        // Both providers keep a separate packed copy of the preceding overlap in RGB until
        // the next segment replaces it. Candle's narrow is made contiguous; MLX gathers it.
        let window_rgb_frames = retained_rgb_frames + segment.history_frames as u64;
        let retained_bytes = spatial_area
            .checked_mul(window_rgb_frames * 3 * 4)
            .and_then(|rgb| rgb.checked_add(conditioning_bytes))
            .ok_or_else(overflow)?;
        // MLX materializes the cropped piece; Candle's narrow view keeps the complete
        // decoded allocation alive until the pieces are concatenated.
        let decoded_tensor_bytes = spatial_area
            .checked_mul((window_rgb_frames + u64::from(decoded_frames)) * 3 * 4)
            .ok_or_else(overflow)?;
        retained_rgb_frames += u64::from(match tail {
            SegmentTail::FullWindows => decoded_frames - segment.history_frames as u32,
            SegmentTail::AlignedRemainder => decoded_frames,
        });
        phases.push(gen_core::VideoPhaseLayout {
            identity: segment.identity,
            latent_frames: segment.latent_frames,
            latent_bytes,
            retained_bytes,
            decoded_tensor_bytes,
            decoded_host_bytes: 0,
            decoded_frames,
            decode_chunk_frames,
            source: segment.source,
            history_frames: segment.history_frames,
        });
    }
    // Both cat implementations alias their input for one piece. With multiple pieces the
    // stitched float allocation is distinct from the pieces, which live through RGB export.
    let output = u64::from(output_frames);
    let stitched = if phases.len() > 1 { output } else { 0 };
    let float_frames = retained_rgb_frames + final_history_frames + stitched;
    // MLX borrows the contiguous uint8 tensor while copying returned frames. Candle keeps
    // its scaled f32 tensor, its CPU uint8 clip, and one contiguous HWC frame during to_vec1.
    let (device_planes, host_planes) = match tail {
        SegmentTail::FullWindows => (4 * float_frames + output, 0),
        SegmentTail::AlignedRemainder => (4 * (float_frames + output), output + 1),
    };
    let rgb_bytes = |planes| spatial_area.checked_mul(3 * planes).ok_or_else(overflow);
    let output_conversion = gen_core::VideoOutputConversion {
        device_bytes: rgb_bytes(device_planes)?,
        host_bytes: rgb_bytes(host_planes)?,
    };
    if tail == SegmentTail::AlignedRemainder {
        // Candle rounds the entire stitched tensor before copying the first frame to the
        // returned RGB vectors. Its eager nearest-even intermediates coexist with the
        // decoded pieces, packed overlap and stitched tensor, independently of host export.
        let last = phases.last_mut().unwrap();
        let rounding =
            crate::candle_round_rgb8_live_bytes(rgb_bytes(output)?, rgb_bytes(4 * float_frames)?)?;
        last.decoded_tensor_bytes = last.decoded_tensor_bytes.max(rounding);
    }
    // Both providers resize every authored driving frame and mask before the first window.
    // Candle keeps those two f32 stacks on the CPU even when only the half-size conditioned
    // window crosses to CUDA. MLX owns them on its unified device until output conversion.
    let stacks_domain = match (tail, domain) {
        (SegmentTail::AlignedRemainder, gen_core::MemoryBudgetDomain::DeviceVram) => {
            gen_core::MemoryBudgetDomain::HostRam
        }
        _ => domain,
    };
    let stacks = gen_core::PricedAllocation::new(
        "scail2_driving_rgb_and_masks",
        stacks_domain,
        rgb_bytes(u64::from(geometry.frames) * 2 * 4)?,
    );
    let role = |role, width, height, frames| gen_core::VideoReferenceLayout {
        role,
        width,
        height,
        frames,
    };
    let mut references = Vec::new();
    for _ in 0..character_count {
        references.push(role("reference", size.0, size.1, 1));
        references.push(role("reference-mask", size.0, size.1, 1));
    }
    references.push(role(
        "driving-pose",
        size.0 / 2,
        size.1 / 2,
        consumed_frames,
    ));
    references.push(role(
        "driving-mask",
        size.0 / 2,
        size.1 / 2,
        consumed_frames,
    ));
    Ok(gen_core::VideoPlanLayout {
        packed_components: Vec::new(),
        retained_allocations: vec![stacks],
        output_conversion,
        domain,
        output: MemoryGeometry {
            frames: output_frames,
            ..geometry
        },
        admission,
        phases,
        references,
    })
}

/// Project the catalog mode and content receipt onto the native Resident scope. A caller
/// supplying an already-native scope must present the exact previously sealed revision.
pub fn project_catalog_context(
    backend: &str,
    req: &gen_core::GenerationRequest,
    geometry: MemoryGeometry,
    evidence: &gen_core::MemoryStructuralResidentEvidence,
    native_revision: String,
    context: &mut gen_core::MemoryRunContext,
) -> Result<bool> {
    let crossed = || {
        Error::Unsupported(
            "scail2: selected memory evidence crossed its request, artifact or route".into(),
        )
    };
    if context.geometry != geometry
        || context.selection.tier != evidence.tier
        || context.selection.strategy != gen_core::MemoryStrategy::Resident
        || !context.has_reference
        || context.has_phases
        || context.use_pid
        || geometry.reference_count != 1
    {
        return Err(crossed());
    }
    let mode = req.video_mode.as_deref().ok_or_else(crossed)?;
    let catalog_mode = match mode {
        "animation" => "animate_character",
        "replacement" => "replace_person",
        _ => return Err(crossed()),
    };
    let native_overlay = (evidence.adapter_count > 0).then_some("adapter");
    if context.mode.as_key() == mode {
        if context.overlay.as_deref() != native_overlay
            || context.evidence_revision != native_revision
        {
            return Err(crossed());
        }
    } else {
        let receipt = gen_core::generator::Scail2AnimationConditioningRef::catalog_receipt(
            backend,
            catalog_mode,
            req.width,
            req.height,
            &req.conditioning,
        )?;
        let adapters = if native_overlay.is_some() {
            "adapter+"
        } else {
            ""
        };
        let expected = format!("{adapters}provider_video_mode:{mode}+{receipt}");
        if context.mode.as_key() != catalog_mode
            || context.overlay.as_deref() != Some(&expected)
            || context.evidence_revision.is_empty()
            || context.evidence_revision.starts_with("scail2-resident-v1:")
                && context.evidence_revision != native_revision
        {
            return Err(crossed());
        }
    }
    context.mode = gen_core::MemoryMode::Other(mode.to_owned());
    context.overlay = native_overlay.map(str::to_owned);
    context.evidence_revision = native_revision;
    Ok(true)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn scail2_stacks_price_all_target_pixels_in_their_physical_domain() {
        use gen_core::MemoryBudgetDomain::{DeviceVram, HostRam, UnifiedMemory};
        let images = vec![
            gen_core::Image {
                width: 1,
                height: 1,
                pixels: vec![0; 3]
            };
            65
        ];
        let clip = gen_core::ControlClipRef {
            frames: &images,
            mask: &images,
            masking_strength: 1.0,
            start_frame: 0,
            mode: gen_core::ReplacementMode::default(),
        };
        for (tail, execution, placement) in [
            (SegmentTail::AlignedRemainder, DeviceVram, HostRam),
            (SegmentTail::AlignedRemainder, HostRam, HostRam),
            (SegmentTail::FullWindows, UnifiedMemory, UnifiedMemory),
        ] {
            let layout = video_layout(tail, execution, (512, 512), 1, clip).unwrap();
            assert_eq!(
                layout.retained_allocations,
                [gen_core::PricedAllocation::new(
                    "scail2_driving_rgb_and_masks",
                    placement,
                    408_944_640
                )]
            );
            if execution == DeviceVram {
                // A single piece aliases the stitched backing, and the complete 65-frame
                // sequence rounds before host RGB export. Neither the raw nor resized host
                // carrier contributes to these CUDA bytes.
                let peak = layout.phases[0].decoded_tensor_bytes;
                assert_eq!(peak, 1_482_424_320);
                let mut budget = gen_core::MemoryBudget {
                    total_bytes: peak - 1,
                    committed_bytes: 0,
                    reclaimable_bytes: 0,
                    reserved_headroom_bytes: 0,
                };
                assert!(!budget.fits(peak));
                budget.total_bytes += 1;
                assert!(budget.fits(peak));
            }
        }
    }

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
