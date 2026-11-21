//! Mochi 1's inverted linear-quadratic flow schedule and physical video geometry.

use gen_core::{
    render::video::{run_video_steps, VideoDenoisePhase, VideoStep},
    CancelFlag, Error, MemoryGeometry, Progress,
};

pub const LATENT_CHANNELS: u32 = 12;
pub const MAX_SEQUENCE_LENGTH: usize = 256;
pub const TEXT_EMBED_DIM: usize = 4096;
pub const SIZE_MULTIPLE: u32 = 16;
pub const MOCHI_SHIFT: f32 = 1.0;
const DEFAULT_STEPS: u32 = 64;
const DEFAULT_FRAMES: u32 = 19;

pub fn validate_request(
    caps: &gen_core::Capabilities,
    req: &gen_core::GenerationRequest,
) -> gen_core::Result<()> {
    if req.prompt.is_empty() {
        return Err(Error::Msg("mochi_1: prompt must not be empty".into()));
    }
    caps.validate_request("mochi_1", req)?;
    if !req.width.is_multiple_of(SIZE_MULTIPLE) || !req.height.is_multiple_of(SIZE_MULTIPLE) {
        return Err(Error::Msg(format!(
            "mochi_1: width/height must be divisible by {SIZE_MULTIPLE} (got {}x{})",
            req.width, req.height
        )));
    }
    if let Some(frames) = req.frames {
        if frames % 6 != 1 {
            return Err(Error::Msg(format!(
                "mochi_1: num_frames must be 1 + 6·k (got {frames})"
            )));
        }
    }
    Ok(())
}

/// Compare the geometry compiled from the selected snapshot with the codec actually held by
/// the loaded provider. A warmed generator can retain older metadata from the same directory.
pub fn check_layout(
    plan: &gen_core::VideoRenderPlan,
    loaded: &gen_core::VideoPlanLayout,
) -> gen_core::Result<()> {
    if plan.layout() != loaded {
        return Err(Error::Unsupported(
            "Mochi codec geometry differs from admission".into(),
        ));
    }
    Ok(())
}

/// Candle's first temporal crop is a view, while MLX gathers the cropped frames.
#[derive(Clone, Copy)]
pub enum MochiDecoder {
    Candle,
    Mlx,
}

/// Request-owned Mochi trajectory and the actual spatial grid fed to its transformer.
pub struct MochiVideo {
    pub phase: VideoDenoisePhase<(), MochiEuler>,
    pub latent_width: u32,
    pub latent_height: u32,
    pub temporal_ratio: u32,
    pub spatial_ratio: u32,
}

impl MochiVideo {
    pub fn new(
        req: &gen_core::GenerationRequest,
        temporal_expansions: &[usize],
        spatial_expansions: &[usize],
        seed: u64,
    ) -> gen_core::Result<Self> {
        fn ratio(expansions: &[usize]) -> gen_core::Result<u32> {
            expansions.iter().try_fold(1u32, |n, &factor| {
                u32::try_from(factor)
                    .ok()
                    .filter(|&f| f > 0)
                    .and_then(|f| n.checked_mul(f))
                    .ok_or_else(|| Error::Unsupported("invalid Mochi codec expansion".into()))
            })
        }
        let temporal_ratio = ratio(temporal_expansions)?;
        let spatial_ratio = ratio(spatial_expansions)?;
        let latent_width = req.width / spatial_ratio;
        let latent_height = req.height / spatial_ratio;
        if latent_width == 0 || latent_height == 0 {
            return Err(Error::Unsupported("empty Mochi latent grid".into()));
        }
        let phase = phase(
            req.width,
            req.height,
            req.frames.unwrap_or(DEFAULT_FRAMES),
            temporal_ratio,
            req.steps.unwrap_or(DEFAULT_STEPS) as usize,
            MOCHI_SHIFT,
            seed,
        )?;
        Ok(Self {
            phase,
            latent_width,
            latent_height,
            temporal_ratio,
            spatial_ratio,
        })
    }

    /// The VAE expands every temporal node, then removes the first ratio − 1 physical frames.
    /// Its peak chunk includes those discarded frames even when the requested output is one frame.
    pub fn layout(
        &self,
        chunk_latent_frames: usize,
        decoder: MochiDecoder,
        domain: gen_core::MemoryBudgetDomain,
    ) -> gen_core::Result<gen_core::VideoPlanLayout> {
        let latent_frames = self.phase.latent_frames;
        let latent_bytes = [
            latent_frames,
            LATENT_CHANNELS,
            self.latent_height,
            self.latent_width,
            size_of::<f32>() as u32,
        ]
        .into_iter()
        .try_fold(1u64, |n, dim| n.checked_mul(u64::from(dim)))
        .ok_or_else(|| Error::Unsupported("Mochi latent byte count overflows".into()))?;
        let decoded_frames = latent_frames
            .checked_mul(self.temporal_ratio)
            .ok_or_else(|| Error::Unsupported("Mochi decoded frame count overflows".into()))?;
        let chunk = chunk_latent_frames.clamp(1, latent_frames as usize) as u32;
        let output = MemoryGeometry {
            width: self.latent_width * self.spatial_ratio,
            height: self.latent_height * self.spatial_ratio,
            frames: decoded_frames - (self.temporal_ratio - 1),
            ..self.phase.identity.geometry
        };
        let chunk_frames = chunk * self.temporal_ratio;
        let mut layout = gen_core::VideoPlanLayout::single(
            output,
            &self.phase,
            latent_bytes,
            decoded_frames,
            chunk_frames,
            vec![],
            domain,
        );
        // Both pipelines retain the two original f32 embeddings/masks and their [neg, pos]
        // concatenations. The batched DiT input is distinct from the current latent. From
        // the second step the original seeded latent is distinct from the current state too.
        let conditioning_bytes = 4 * MAX_SEQUENCE_LENGTH * (TEXT_EMBED_DIM + 1) * size_of::<f32>();
        let latent_copies = if self.phase.sigmas.len() > 2 { 3 } else { 2 };
        layout.phases[0].retained_bytes = latent_bytes
            .checked_mul(latent_copies)
            .and_then(|n| n.checked_add(conditioning_bytes as u64))
            .ok_or_else(|| Error::Unsupported("Mochi CFG byte count overflows".into()))?;
        // A returned frame has three channels. The VAE projects into f32 on both backends.
        let overflow = || Error::Unsupported("Mochi decoder byte count overflows".into());
        let rgb_frame = u64::from(output.width)
            .checked_mul(u64::from(output.height))
            .and_then(|pixels| pixels.checked_mul(3))
            .ok_or_else(overflow)?;
        let f32_frame = rgb_frame
            .checked_mul(size_of::<f32>() as u64)
            .ok_or_else(overflow)?;
        let frames = u64::from(output.frames);
        let raw = u64::from(decoded_frames);
        let rgb = rgb_frame.checked_mul(frames).ok_or_else(overflow)?;
        let chunked = chunk < latent_frames;
        let (decoder_bytes, conversion) = match decoder {
            MochiDecoder::Candle => {
                // narrow() retains the entire first chunk, including its discarded prefix.
                // A multi-chunk cat allocates the cropped clip beside all raw chunk backings.
                let (live, returned) = if chunked {
                    (raw + frames, frames)
                } else {
                    (raw, raw)
                };
                let decoded = f32_frame.checked_mul(returned).ok_or_else(overflow)?;
                // to_uint8_frames keeps the source backing and three distinct f32 arrays
                // (the contiguous channel-last copy, half, and clipped) across round_rgb8.
                // The shared rounding calculation already includes scaled.
                let retained = rgb
                    .checked_mul(3 * size_of::<f32>() as u64)
                    .and_then(|n| n.checked_add(decoded))
                    .ok_or_else(overflow)?;
                let rounding = crate::candle_round_rgb8_live_bytes(rgb, retained)?;
                let stitched = f32_frame.checked_mul(live).ok_or_else(overflow)?;
                // render retains the f32 video and u8 frames tensor. to_vec1 creates a flat host
                // Vec that stays live as the individual Image.pixels buffers are copied.
                (
                    stitched.max(rounding),
                    gen_core::VideoOutputConversion {
                        device_bytes: decoded.checked_add(rgb).ok_or_else(overflow)?,
                        host_bytes: rgb,
                    },
                )
            }
            MochiDecoder::Mlx => {
                // take_axis gathers a new buffer while the first raw chunk is live. Later,
                // concatenate reads all evaluated cropped chunks into one distinct f32 clip.
                let kept_first = chunk_frames - (self.temporal_ratio - 1);
                let crop = u64::from(chunk_frames) + u64::from(kept_first);
                let live = if chunked { crop.max(2 * frames) } else { crop };
                // decode_to_frames releases the f32 handle before the final copy. The u8
                // frame tensor is already row-major: the reshape and host slice share its storage.
                (
                    f32_frame.checked_mul(live).ok_or_else(overflow)?,
                    gen_core::VideoOutputConversion {
                        device_bytes: rgb,
                        host_bytes: 0,
                    },
                )
            }
        };
        layout.phases[0].decoded_tensor_bytes = decoder_bytes;
        layout.output_conversion = conversion;
        Ok(layout)
    }
}

/// Genmo's linear-quadratic sigma schedule (`linear_quadratic_schedule`, `linear_steps = num_steps/2`).
///
/// Returns `num_steps` values in `[0, 1]`, descending (`sigma[0] = 1.0`). This is the **pre-inversion**
/// schedule; [`inverted_sigmas`] applies the `invert_sigmas` transform on top.
pub fn linear_quadratic_schedule(num_steps: usize, threshold_noise: f64) -> Vec<f64> {
    // With one interval the evaluation is at the initial node. The two-segment
    // formula has no linear segment here and divides by zero.
    if num_steps == 1 {
        return vec![1.0];
    }
    let n = num_steps as f64;
    let linear_steps = num_steps / 2;
    let ls = linear_steps as f64;

    // Linear segment: sigma_i = i·threshold / linear_steps for i in 0..linear_steps.
    let mut sched: Vec<f64> = (0..linear_steps)
        .map(|i| i as f64 * threshold_noise / ls)
        .collect();

    // Quadratic segment for i in linear_steps..num_steps.
    let threshold_noise_step_diff = ls - threshold_noise * n;
    let quadratic_steps = (num_steps - linear_steps) as f64;
    let quadratic_coef = threshold_noise_step_diff / (ls * quadratic_steps * quadratic_steps);
    let linear_coef = threshold_noise / ls
        - 2.0 * threshold_noise_step_diff / (quadratic_steps * quadratic_steps);
    let konst = quadratic_coef * (ls * ls);
    for i in linear_steps..num_steps {
        let i = i as f64;
        sched.push(quadratic_coef * i * i + linear_coef * i + konst);
    }

    // sigma_schedule = 1 − x.
    sched.iter().map(|x| 1.0 - x).collect()
}

/// Build in f64, shift and invert the rounded f32 nodes, then append the terminal one.
pub fn inverted_sigmas(steps: usize, shift: f32) -> Vec<f32> {
    let mut sigmas: Vec<_> = linear_quadratic_schedule(steps, 0.025)
        .into_iter()
        .map(|value| {
            let mut sigma = value as f32;
            if shift != 1.0 {
                sigma = shift * sigma / (1.0 + (shift - 1.0) * sigma);
            }
            1.0 - sigma
        })
        .collect();
    sigmas.push(1.0);
    sigmas
}

/// Native direct flow Euler. The model predicts velocity even at the first sigma of zero.
#[derive(Debug)]
pub struct MochiEuler;

impl MochiEuler {
    /// The callback predicts, integrates, and materializes its updated latent before
    /// reporting this step. Both backends use the same model coordinates and progress.
    pub fn run<E: From<Error>>(
        sigmas: &[f32],
        cancel: &CancelFlag,
        mut update: impl FnMut(VideoStep, f32) -> Result<(), E>,
        progress: &mut dyn FnMut(Progress),
    ) -> Result<(), E> {
        run_video_steps(sigmas, cancel, |step| {
            update(step, Self::timestep(step.sigma))?;
            progress(Progress::Step {
                current: (step.index + 1) as u32,
                total: (sigmas.len() - 1) as u32,
            });
            Ok(())
        })
    }

    pub fn timestep(sigma: f32) -> f32 {
        sigma * 1000.0
    }

    /// Widen each node before subtraction as the native scheduler does. Candle consumes
    /// this f64 affine coefficient directly. MLX rounds it once for its f32 scalar.
    pub fn delta(step: VideoStep) -> f64 {
        f64::from(step.next_sigma) - f64::from(step.sigma)
    }
}

pub fn phase(
    width: u32,
    height: u32,
    frames: u32,
    temporal_ratio: u32,
    steps: usize,
    shift: f32,
    seed: u64,
) -> gen_core::Result<VideoDenoisePhase<(), MochiEuler>> {
    if frames == 0 || temporal_ratio == 0 {
        return Err(gen_core::Error::Unsupported(
            "invalid Mochi temporal geometry".into(),
        ));
    }
    Ok(VideoDenoisePhase::fixed(
        MemoryGeometry {
            width,
            height,
            batch: 1,
            frames,
            reference_count: 0,
        },
        1 + (frames - 1) / temporal_ratio,
        MochiEuler,
        inverted_sigmas(steps, shift),
        seed,
    ))
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::cell::RefCell;

    #[test]
    fn video_layout_counts_f32_latents_and_raw_decode_before_the_first_crop() {
        let req = gen_core::GenerationRequest {
            width: 80,
            height: 48,
            frames: Some(1),
            ..Default::default()
        };
        for (frames, latent_frames, decoded_frames) in
            [(Some(1), 1, 6), (Some(7), 2, 12), (None, 4, 24)]
        {
            let req = gen_core::GenerationRequest {
                frames,
                ..req.clone()
            };
            let video = MochiVideo::new(&req, &[1, 2, 3], &[2, 2, 2], 37).unwrap();
            let layout = video
                .layout(
                    1,
                    MochiDecoder::Mlx,
                    gen_core::MemoryBudgetDomain::UnifiedMemory,
                )
                .unwrap();
            let phase = &layout.phases[0];
            assert_eq!(layout.output.frames, frames.unwrap_or(19));
            assert_eq!(layout.admission, video.phase.identity.geometry);
            assert_eq!(phase.latent_frames, latent_frames);
            assert_eq!(
                phase.latent_bytes,
                u64::from(latent_frames) * 12 * 10 * 6 * 4
            );
            assert_eq!(phase.decoded_frames, decoded_frames);
            assert_eq!(phase.decode_chunk_frames, 6);
            assert_eq!(phase.source, 0..layout.output.frames as usize);
            assert_eq!(phase.history_frames, 0);
            assert!(layout.references.is_empty());
        }
        assert!(MochiVideo::new(&req, &[0], &[8], 37).is_err());
        assert!(MochiVideo::new(&req, &[6], &[usize::MAX, 2], 37).is_err());
    }

    #[test]
    fn decoder_backings_and_output_staging_follow_the_backend_crop() {
        let frame_bytes = 80 * 48 * 3;
        for (frames, candle_live, candle_staging, mlx_live) in
            [(1, 61, 25, 7), (7, 287, 35, 14), (19, 779, 95, 38)]
        {
            let req = gen_core::GenerationRequest {
                width: 80,
                height: 48,
                frames: Some(frames),
                ..Default::default()
            };
            let video = MochiVideo::new(&req, &[1, 2, 3], &[2, 2, 2], 37).unwrap();
            let candle = video
                .layout(
                    1,
                    MochiDecoder::Candle,
                    gen_core::MemoryBudgetDomain::DeviceVram,
                )
                .unwrap();
            assert_eq!(
                candle.phases[0].decoded_tensor_bytes,
                candle_live * frame_bytes
            );
            assert_eq!(
                candle.output_conversion.device_bytes,
                candle_staging * frame_bytes
            );
            assert_eq!(
                candle.output_conversion.host_bytes,
                u64::from(frames) * frame_bytes
            );
            let mlx = video
                .layout(
                    1,
                    MochiDecoder::Mlx,
                    gen_core::MemoryBudgetDomain::UnifiedMemory,
                )
                .unwrap();
            assert_eq!(
                mlx.phases[0].decoded_tensor_bytes,
                mlx_live * frame_bytes * 4
            );
            assert_eq!(
                mlx.output_conversion.device_bytes,
                u64::from(frames) * frame_bytes
            );
            assert_eq!(mlx.output_conversion.host_bytes, 0);
        }
    }

    #[test]
    fn cfg_counts_both_prompt_backings_and_only_distinct_latents() {
        for (steps, copies) in [(1, 2), (2, 3)] {
            let req = gen_core::GenerationRequest {
                width: 80,
                height: 48,
                frames: Some(7),
                steps: Some(steps),
                ..Default::default()
            };
            let video = MochiVideo::new(&req, &[6], &[8], 4).unwrap();
            for backend in [MochiDecoder::Candle, MochiDecoder::Mlx] {
                let layout = video
                    .layout(1, backend, gen_core::MemoryBudgetDomain::UnifiedMemory)
                    .unwrap();
                let phase = &layout.phases[0];
                assert_eq!(phase.latent_bytes, 5_760);
                assert_eq!(phase.retained_bytes, 16_781_312 + copies * 5_760);
            }
        }
    }

    #[derive(Debug, PartialEq)]
    enum Event {
        Update(usize, u32, u32, u32, u32),
        Progress(u32, u32),
    }

    #[test]
    fn inverted_flow_zero_step_and_post_update_progress_match_native_trace() {
        let phase = phase(64, 48, 19, 6, 2, 1.0, 37).unwrap();
        assert_eq!(phase.identity.geometry.frames, 19);
        assert_eq!(phase.latent_frames, 4);
        assert_eq!(phase.seed, 37);
        let expected = [
            Event::Update(0, 0, 0x3ccc_ccc0, 0, 0x3f8c_cccc),
            Event::Progress(1, 2),
            Event::Update(1, 0x3ccc_ccc0, 0x3f80_0000, 0x41c7_fff4, 0xbf59_999c),
            Event::Progress(2, 2),
        ];
        for cancel_after_first in [false, true] {
            let cancel = CancelFlag::new();
            let mut x = 1.0f32;
            let events = RefCell::new(Vec::new());
            let result = MochiEuler::run(
                &phase.sigmas,
                &cancel,
                |step, t| {
                    let velocity = [4.0, -2.0][step.index];
                    x += velocity * MochiEuler::delta(step) as f32;
                    events.borrow_mut().push(Event::Update(
                        step.index,
                        step.sigma.to_bits(),
                        step.next_sigma.to_bits(),
                        t.to_bits(),
                        x.to_bits(),
                    ));
                    Ok::<_, Error>(())
                },
                &mut |progress| {
                    let Progress::Step { current, total } = progress else {
                        panic!("unexpected progress")
                    };
                    events.borrow_mut().push(Event::Progress(current, total));
                    if cancel_after_first && current == 1 {
                        cancel.cancel();
                    }
                },
            );
            if cancel_after_first {
                assert!(matches!(result, Err(Error::Canceled)));
                assert_eq!(events.into_inner(), expected[..2]);
            } else {
                result.unwrap();
                assert_eq!(events.into_inner(), expected);
            }
        }
    }

    #[test]
    fn single_step_runs_one_full_velocity_update_at_model_time_zero() {
        let phase = phase(64, 48, 7, 6, 1, 1.0, 0).unwrap();
        phase.validate().unwrap();
        assert_eq!(phase.sigmas, [0.0, 1.0]);
        let mut latent = 1.0f32;
        let mut progress = Vec::new();
        MochiEuler::run(
            &phase.sigmas,
            &CancelFlag::new(),
            |step, t| {
                assert_eq!((step.index, t), (0, 0.0));
                latent += 4.5 * MochiEuler::delta(step) as f32;
                Ok::<_, Error>(())
            },
            &mut |p| match p {
                Progress::Step { current, total } => progress.push((current, total)),
                _ => panic!("unexpected progress"),
            },
        )
        .unwrap();
        assert_eq!(latent, 5.5);
        assert_eq!(progress, [(1, 1)]);
    }

    #[test]
    fn shifted_odd_length_schedule_preserves_f32_nodes() {
        let sigmas = inverted_sigmas(5, 1.5);
        assert_eq!(
            sigmas.into_iter().map(f32::to_bits).collect::<Vec<_>>(),
            [
                0,
                0x3c09_1a80,
                0x3c89_ae00,
                0x3dcb_0238,
                0x3ebc_a1ac,
                0x3f80_0000
            ],
        );
    }
}
