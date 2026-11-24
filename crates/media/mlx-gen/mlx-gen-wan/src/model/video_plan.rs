//! The padded shape shared by native execution and pre-load curated video admission.

use super::*;
use mlx_gen::gen_core::{
    self, ProviderRegistryBuilder, VideoPhaseLayout, VideoPlanLayout, VideoPlanRegistration,
    VideoReferenceLayout,
};

pub(super) struct VideoDimensions {
    pub(super) frames: usize,
    pub(super) trim_out: usize,
    pub(super) width: u32,
    pub(super) height: u32,
    pub(super) lat: [i32; 4],
}

pub(super) fn video_dimensions(
    req: &GenerationRequest,
    cfg: &WanModelConfig,
    vae: VaeTiling,
) -> Result<VideoDimensions> {
    let frames = req.frames.map(|f| f as usize).unwrap_or(cfg.frame_num);
    let stride = provider_vae_stride(vae);
    let trim_out = req.trim_first_frames.unwrap_or(0) as usize * stride.0;
    let gen_frames = frames
        .checked_add(trim_out)
        .filter(|f| *f <= i32::MAX as usize)
        .ok_or_else(|| Error::Unsupported("Wan temporal extent overflows".into()))?;
    let (width, height) = resolve_capped_dims(req, cfg, vae);
    let lat = latent_shape(gen_frames, height, width, cfg.vae_z_dim, stride)?;
    Ok(VideoDimensions {
        frames,
        trim_out,
        width,
        height,
        lat,
    })
}

pub(super) fn video_layout(
    desc: &ModelDescriptor,
    cfg: &WanModelConfig,
    req: &GenerationRequest,
    load_quant: Option<Quant>,
) -> Result<VideoPlanLayout> {
    let vae = if desc.id == MODEL_ID {
        Wan::validate_metadata(desc, cfg, req)?;
        Ti2vProviderVae::VAE_TILING
    } else {
        Wan14b::validate_metadata(desc, cfg, req)?;
        A14bProviderVae::VAE_TILING
    };
    let VideoDimensions {
        trim_out,
        width,
        height,
        lat,
        ..
    } = video_dimensions(req, cfg, vae)?;
    let is_i2v = cfg.is_i2v_concat();
    let (identity, latent_frames) =
        crate::pipeline::curated_phase_geometry(&lat, vae, u32::from(is_i2v))?;
    let decoded_frames = if desc.id == MODEL_ID {
        identity.geometry.frames
    } else {
        latent_frames
            .checked_mul(vae.temporal_scale as u32)
            .ok_or_else(|| Error::Unsupported("Wan decoder temporal extent overflows".into()))?
    };
    let delivered = decoded_frames
        .checked_sub(trim_out as u32)
        .ok_or_else(|| Error::Unsupported("invalid Wan trim".into()))?;
    let output = gen_core::MemoryGeometry {
        frames: delivered,
        ..identity.geometry
    };
    let bytes_per_channel = lat[1..]
        .iter()
        .try_fold(4u64, |n, &d| n.checked_mul(u64::try_from(d).ok()?))
        .ok_or_else(|| Error::Unsupported("Wan latent allocation overflows".into()))?;
    let channels = |c: u64| {
        bytes_per_channel
            .checked_mul(c)
            .ok_or_else(|| Error::Unsupported("Wan channel allocation overflows".into()))
    };
    let mut packed_components = vec![];
    let dit = cfg.quantization.or_else(|| {
        load_quant.map(|q| WanQuant {
            bits: q.bits(),
            group_size: mlx_gen::quant::DEFAULT_GROUP_SIZE,
        })
    });
    for (name, packed, stored) in [
        ("transformer", dit, Some(cfg.quantization.is_some())),
        ("text_encoder", effective_te_quant(cfg, load_quant), None),
    ] {
        if let Some(q) = packed {
            packed_components.push(gen_core::VideoPackedComponent {
                name,
                bits: q.bits,
                group_size: q.group_size,
                stored,
            });
        }
    }
    // Every Wan decoder produces the complete RGB8 tensor before copying frames to host images.
    let decoded_rgb_bytes = u64::from(width)
        .checked_mul(u64::from(height))
        .and_then(|n| n.checked_mul(u64::from(decoded_frames)))
        .and_then(|n| n.checked_mul(3))
        .ok_or_else(|| Error::Unsupported("Wan decoded clip overflows".into()))?;
    Ok(VideoPlanLayout {
        retained_allocations: vec![],
        packed_components,
        // frames_u8, its contiguous view and reshape share one allocation. All frames are
        // copied to host before the trimmed prefix is drained.
        output_conversion: gen_core::VideoOutputConversion {
            device_bytes: decoded_rgb_bytes,
            host_bytes: u64::from(width) * u64::from(height) * 3 * trim_out as u64,
        },
        output,
        admission: identity.geometry,
        domain: gen_core::MemoryBudgetDomain::UnifiedMemory,
        references: if is_i2v {
            vec![VideoReferenceLayout {
                role: "first_frame",
                width,
                height,
                frames: 1,
            }]
        } else {
            vec![]
        },
        phases: vec![VideoPhaseLayout {
            identity,
            latent_frames,
            latent_bytes: channels(lat[0] as u64)?,
            // Channel-concat holds the 16-channel encoded video and four-channel mask as y.
            retained_bytes: if is_i2v { channels(20)? } else { 0 },
            decoded_host_bytes: 0,
            decoded_tensor_bytes: decoded_rgb_bytes
                .checked_mul(4)
                .ok_or_else(|| Error::Unsupported("Wan f32 decoder output overflows".into()))?,
            decoded_frames,
            // Live device headroom may further split the temporal tiles. A full pass is supported.
            decode_chunk_frames: decoded_frames,
            source: 0..delivered as usize,
            history_frames: 0,
        }],
    })
}

fn compile(
    desc: ModelDescriptor,
    spec: &LoadSpec,
    req: &GenerationRequest,
) -> gen_core::Result<Option<VideoPlanLayout>> {
    if !is_wan_curated(req.sampler.as_deref()) {
        return Ok(None);
    }
    let WeightsSource::Dir(root) = &spec.weights else {
        return Err(gen_core::Error::Unsupported(
            "MLX Wan video requires a snapshot directory".into(),
        ));
    };
    let cfg = WanModelConfig::from_model_dir(root)?;
    if (desc.id == MODEL_ID && (!cfg.is_ti2v() || cfg.dual_model))
        || (desc.id != MODEL_ID && !cfg.dual_model)
        || (desc.id == MODEL_ID_I2V_14B && !cfg.is_i2v_concat())
    {
        return Err(gen_core::Error::Unsupported(
            "Wan video route and snapshot config differ".into(),
        ));
    }
    let quant = resolve_load_time_quant(desc.id, &cfg, spec.quantize)?;
    Ok(Some(video_layout(&desc, &cfg, req, quant)?))
}

pub(crate) fn register_video_plans(registry: ProviderRegistryBuilder) -> ProviderRegistryBuilder {
    registry
        .register_video_plan(VideoPlanRegistration {
            provider_id: MODEL_ID,
            compile: |s, r| compile(descriptor(), s, r),
        })
        .register_video_plan(VideoPlanRegistration {
            provider_id: MODEL_ID_T2V_14B,
            compile: |s, r| compile(descriptor_t2v_14b(), s, r),
        })
        .register_video_plan(VideoPlanRegistration {
            provider_id: MODEL_ID_I2V_14B,
            compile: |s, r| compile(descriptor_i2v_14b(), s, r),
        })
}
