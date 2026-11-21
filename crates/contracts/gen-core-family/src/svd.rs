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
    let checked_bytes =
        |n| u64::try_from(n).map_err(|_| Error::Unsupported("SVD tensor size overflows".into()));
    let chunk_frames = decode_chunk.max(1).min(frames);
    let live_decoded_frames = match execution {
        // MLX concatenates the decoded chunks into a full f32 clip before converting to RGB8.
        Execution::Mlx => frames,
        // Candle materializes packed RGB8 and releases each decoded tensor before the next chunk.
        Execution::Candle { .. } => chunk_frames,
    };
    let decoded_tensor_bytes = u128::from(req.width)
        * u128::from(req.height)
        * u128::from(live_decoded_frames)
        * 3
        * f32_bytes;
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
    layout.phases[0].decoded_tensor_bytes = checked_bytes(decoded_tensor_bytes)?;
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
