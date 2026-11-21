//! SVD image-to-video physical geometry and sealed Resident request projection.
use gen_core::{
    render::video::VideoDenoisePhase, Error, GenerationRequest, Image, MemoryBudgetDomain,
    MemoryGeometry, MemoryNumericTier, MemoryRunContext, MemoryStrategy, Result, VideoPlanLayout,
    VideoReferenceLayout, VideoRenderPlan, SVD_LATENT_SPACE,
};

pub const CLIP_SIZE: usize = 224;
pub const VAE_SCALE: u32 = SVD_LATENT_SPACE.spatial_compression.width as u32;

/// Both temporal VAEs preserve one output frame per latent frame. The backends use
/// different tensor layouts and pass the unpacked B,F,H,W axes here.
pub fn phase(
    [batch, frames, height, width]: [u32; 4],
    sampler: Option<&str>,
    sigmas: Vec<f32>,
    seed: u64,
) -> Result<VideoDenoisePhase<(), Option<&str>>> {
    let pixels = |n: u32| {
        n.checked_mul(VAE_SCALE)
            .ok_or_else(|| Error::Unsupported("SVD pixel grid overflows".into()))
    };
    let geometry = MemoryGeometry {
        width: pixels(width)?,
        height: pixels(height)?,
        frames,
        batch,
        reference_count: 1,
    };
    Ok(VideoDenoisePhase::fixed(
        geometry, frames, sampler, sigmas, seed,
    ))
}

/// The native execution route determines which tensor buffers are materialized.
#[derive(Clone, Copy)]
pub enum Execution {
    Mlx,
    Candle { cfg: bool },
}

fn checked_bytes(n: u128) -> Result<u64> {
    u64::try_from(n).map_err(|_| Error::Unsupported("SVD tensor size overflows".into()))
}

#[derive(Default)]
struct DecodeBuffers {
    device: u128,
    host: u128,
    output: gen_core::VideoOutputConversion,
}

fn candle_decode_buffers(
    frames: u32,
    chunk: u32,
    rgb_frame: u128,
    domain: MemoryBudgetDomain,
) -> Result<DecodeBuffers> {
    let frames = u128::from(frames);
    let chunk = u128::from(chunk);
    let tail = (frames - 1) % chunk + 1;
    let f32_bytes = std::mem::size_of::<f32>() as u128;
    // NCHW -> HWC -> flatten copies the non-contiguous frame before to_vec1 copies
    // its pixels into Image. The chunk's narrow and squeeze are views of the same host u8.
    let flat_frame = if rgb_frame > 3 { rgb_frame } else { 0 };
    let mut buffers = DecodeBuffers {
        output: gen_core::VideoOutputConversion {
            // decoded_chunk_to_images keeps both its borrowed f32 chunk and its scaled
            // f32 chunk until the returned images have been copied from the CPU tensor.
            device_bytes: checked_bytes(2 * f32_bytes * tail * rgb_frame)?,
            host_bytes: checked_bytes(tail * rgb_frame + flat_frame)?,
        },
        ..Default::default()
    };
    // Earlier full chunks have fewer returned frames than the latest full chunk.
    // A final short chunk can have less staging, but more accumulated frame vectors.
    let full_start = (frames / chunk - 1) * chunk;
    for (start, count) in [(full_start, chunk), (frames - tail, tail)] {
        let samples = count * rgb_frame;
        let previous = start * rgb_frame;
        let round = crate::candle_round_rgb8_live_bytes(
            checked_bytes(samples)?,
            checked_bytes(samples * f32_bytes)?,
        )?;
        let device_copy = 2 * f32_bytes * samples;
        let host_copy = previous + samples + samples + flat_frame; // returned RGB + host u8 + HWC
        for (device, host) in [(u128::from(round), previous), (device_copy, host_copy)] {
            if domain == MemoryBudgetDomain::DeviceVram {
                buffers.device = buffers.device.max(device);
                buffers.host = buffers.host.max(host);
            } else if device + host > buffers.device + buffers.host {
                // CPU and unified memory charge both parts of one simultaneously-live stage.
                buffers.device = device;
                buffers.host = host;
            }
        }
    }
    Ok(buffers)
}

pub fn plan_layout(
    req: &GenerationRequest,
    reference: &Image,
    frames: u32,
    decode_chunk: u32,
    sigmas: Vec<f32>,
    domain: MemoryBudgetDomain,
    (execution, projection_dim): (Execution, usize),
) -> Result<VideoPlanLayout> {
    let width = req.width / VAE_SCALE;
    let height = req.height / VAE_SCALE;
    // The layout only consumes phase geometry. An omitted seed is sampled at execution.
    let phase = phase(
        [1, frames, height, width],
        req.sampler.as_deref(),
        sigmas,
        req.seed.unwrap_or_default(),
    )?;
    // Both execution paths integrate f32 latents, including the MLX half-precision UNet.
    let f32_bytes = std::mem::size_of::<f32>() as u128;
    let still_bytes =
        u128::from(width) * u128::from(height) * u128::from(SVD_LATENT_SPACE.channels) * f32_bytes;
    let latent_bytes = still_bytes * u128::from(frames);
    // The VAE, CLIP output, microconditioning IDs and frame guidance are all f32.
    // MLX materializes its two CFG rows together, while the reference latent's original
    // temporal broadcast still aliases a single frame. Candle makes its broadcast contiguous.
    let embed = projection_dim as u128 * f32_bytes;
    let time_ids = 3 * f32_bytes;
    let guidance = u128::from(frames) * f32_bytes;
    let latent_and_embed = latent_bytes + embed;
    let retained = match execution {
        Execution::Mlx => {
            still_bytes + embed + time_ids + 2 * (latent_and_embed + time_ids) + guidance
        }
        Execution::Candle { cfg: true } => 2 * latent_and_embed + time_ids + guidance,
        Execution::Candle { cfg: false } => latent_and_embed + time_ids,
    };
    if frames == 0 {
        return Err(Error::Unsupported(
            "SVD requires at least one output frame".into(),
        ));
    }
    let chunk_frames = decode_chunk.max(1).min(frames);
    let rgb_frame = u128::from(req.width) * u128::from(req.height) * 3;
    let rgb = rgb_frame * u128::from(frames);
    let buffers = match execution {
        Execution::Mlx => {
            // concatenate aliases its sole input. With multiple evaluated chunks it writes
            // a second complete f32 clip while reading the first clip's chunk backings.
            let decoded = rgb * f32_bytes * if chunk_frames < frames { 2 } else { 1 };
            // frames_to_images owns the decoded, rescaled, clipped and rounded f32 arrays.
            // The final contiguous u8 is borrowed for the per-image copies on unified memory.
            DecodeBuffers {
                device: decoded,
                host: 0,
                output: gen_core::VideoOutputConversion {
                    device_bytes: checked_bytes(rgb * (4 * f32_bytes + 1))?,
                    host_bytes: 0,
                },
            }
        }
        Execution::Candle { .. } => candle_decode_buffers(frames, chunk_frames, rgb_frame, domain)?,
    };
    let mut layout = VideoPlanLayout::single(
        phase.identity.geometry,
        &phase,
        checked_bytes(latent_bytes)?,
        frames,
        chunk_frames,
        vec![
            VideoReferenceLayout {
                role: "source",
                width: reference.width,
                height: reference.height,
                frames: 1,
            },
            VideoReferenceLayout {
                role: "image_encoder",
                width: CLIP_SIZE as u32,
                height: CLIP_SIZE as u32,
                frames: 1,
            },
            VideoReferenceLayout {
                role: "vae_encoder",
                width: req.width,
                height: req.height,
                frames: 1,
            },
        ],
        domain,
    );
    layout.phases[0].retained_bytes = checked_bytes(retained)?;
    layout.phases[0].decoded_tensor_bytes = checked_bytes(buffers.device)?;
    layout.phases[0].decoded_host_bytes = checked_bytes(buffers.host)?;
    layout.output_conversion = buffers.output;
    Ok(layout)
}

/// A domain-specific material receipt (including its exact input image and scheduler)
/// may refine the existing selected context. No strategy or byte-budget field is rewritten.
pub fn project_resident_context(
    plan: &VideoRenderPlan,
    context: &mut MemoryRunContext,
    prepared_tier: MemoryNumericTier,
    request_revision: Result<String>,
    validate: impl FnOnce(&MemoryRunContext) -> Result<()>,
) -> Result<bool> {
    let revision = match request_revision {
        Ok(revision) => revision,
        Err(Error::Unsupported(_)) => return Ok(false),
        Err(error) => return Err(error),
    };
    if context.geometry != plan.geometry()
        || context.selection.tier != prepared_tier
        || context.selection.strategy != MemoryStrategy::Resident
        || context.mode.as_key() != "image_to_video"
        || !context.has_reference
        || context.use_pid
        || context.has_phases
    {
        return Err(Error::Unsupported(
            "SVD memory selection crossed its admitted route".into(),
        ));
    }
    if !matches!(
        context.overlay.as_deref(),
        None | Some("provider_video_mode:image_to_video")
    ) {
        return Err(Error::Unsupported(
            "SVD memory selection has an unsupported overlay".into(),
        ));
    }
    let mut projected = context.clone();
    projected.overlay = None;
    projected.evidence_revision = revision;
    validate(&projected)?;
    *context = projected;
    Ok(true)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn candle_accounts_for_the_latest_full_chunk_and_partial_last_chunk_per_budget() {
        let pixel_bytes = 1024 * 576 * 3;
        // (frames, chunk, last, peak f32/rounding storage, accumulated RGB at that peak,
        //  independent host peak during the per-frame HWC conversion)
        for (frames, chunk, last, device, previous, host) in [
            (1, 1, 1, 29, 0, 3),
            (5, 2, 1, 58, 2, 7),
            (9, 4, 1, 116, 4, 13),
            (8, 4, 4, 116, 4, 13),
        ] {
            for domain in MemoryBudgetDomain::ALL {
                let data = candle_decode_buffers(frames, chunk, pixel_bytes, domain).unwrap();
                assert_eq!(data.device, device * pixel_bytes);
                let peak_host = if domain == MemoryBudgetDomain::DeviceVram {
                    host
                } else {
                    previous
                };
                assert_eq!(data.host, peak_host * pixel_bytes);
                assert_eq!(u128::from(data.output.device_bytes), 8 * last * pixel_bytes);
                assert_eq!(u128::from(data.output.host_bytes), (last + 1) * pixel_bytes);
            }
        }
    }
}
