//! Structural costs only; no native calibration or optimized-fit measurement is claimed.
use crate::loader::Sources;
use candle_gen::gen_core::{self, moss_image::*, *};

pub(crate) fn footprint_sources(s: &Sources) -> gen_core::Result<PerComponentBytes> {
    let mut result = PerComponentBytes::default();
    for a in &s.ordinary {
        if !a.relative.ends_with(".safetensors") {
            continue;
        }
        a.pin.verify_unchanged()?;
        let bytes = moss_image_asset(a.repo, a.revision, &a.relative)
            .ok_or_else(|| {
                gen_core::Error::Unsupported("MOSS missing frozen footprint identity".into())
            })?
            .bytes;
        match a.role {
            MossImageAssetRole::Transformer => result.dit += bytes,
            MossImageAssetRole::TextEncoder => result.text_encoder += bytes,
            MossImageAssetRole::Vae => result.vae += bytes,
            _ => (),
        }
    }
    Ok(result)
}
pub(crate) fn contract_sources(
    variant: MossImageVariant,
    s: &Sources,
) -> gen_core::Result<MemoryProviderContract> {
    let bytes = footprint_sources(s)?;
    let mut c = MemoryProviderContract::compatibility_default(
        variant.provider_id(),
        MemoryBackendRealization::CandleCuda {
            device_residency: true,
            host_backed_weights: false,
            host_to_device_block_materialization: false,
            block_materialization: MemoryWindowMaterialization::DeviceFormatTransfer,
        },
    );
    c.load_shape = s.spec.load_shape;
    c.asset_facts = MemoryAssetFacts {
        base_bytes: bytes.dit + bytes.text_encoder + bytes.vae,
        transformer_bytes: bytes.dit,
        conditioning_bytes: bytes.text_encoder,
        decoder_bytes: bytes.vae,
        overlay_bytes: 0,
    };
    c.architecture_facts = architecture_facts(s);
    c.lifecycle = MemoryLifecycleCapabilities {
        phases: vec![
            MemoryPhase::Conditioning,
            MemoryPhase::Denoise,
            MemoryPhase::Decode,
        ],
        synchronized_phase_release: true,
        ..Default::default()
    };
    c.formula = MemoryFormulaKind::PhaseEnvelope {
        phases: c.lifecycle.phases.clone(),
        variables: vec![
            MemoryFormulaVariable::AssetBytes,
            MemoryFormulaVariable::BatchCount,
            MemoryFormulaVariable::ConditioningTokenCount,
            MemoryFormulaVariable::PixelCount,
        ],
    };
    // The sequential loader is executable; its optimized-fit rung remains Missing until calibrated.
    Ok(c)
}
/// Conservative request-scaled upper bound from authenticated tensor headers, never a measurement.
pub(crate) fn estimate(
    s: &Sources,
    r: &MossImageResolved,
    req: &GenerationRequest,
    images: usize,
    staged: bool,
) -> gen_core::Result<u64> {
    let mut text = 0u64;
    let mut dit = 0u64;
    let mut vae = 0u64;
    let mut extra = 0u64;
    for a in s.assets(r.acceleration)? {
        if !a.relative.ends_with(".safetensors") {
            continue;
        }
        let headers = s.headers.get(a.pin.content_sha256()).ok_or_else(|| {
            gen_core::Error::Unsupported("MOSS missing authenticated tensor header facts".into())
        })?;
        for h in headers {
            let n = h.element_count()?;
            let cost = match a.role {
                MossImageAssetRole::TextEncoder if h.name.starts_with("language_model.") => {
                    text = text.saturating_add(n * 4);
                    0
                }
                MossImageAssetRole::Transformer => {
                    dit = dit.saturating_add(
                        n * if r.acceleration
                            == MossImageAcceleration::Dmd(MossDmdPrecision::FullFp32)
                        {
                            4
                        } else {
                            2
                        },
                    );
                    0
                }
                MossImageAssetRole::Vae
                    if h.name.starts_with("decoder.")
                        || h.name.starts_with("post_quant_conv.")
                        || h.name.starts_with("bn.running_") =>
                {
                    vae = vae.saturating_add(n * 4);
                    0
                }
                MossImageAssetRole::DmdLora
                | MossImageAssetRole::DmdTime
                | MossImageAssetRole::DmdDataspace => n * 4,
                _ => 0,
            };
            extra = extra.saturating_add(cost);
        }
    }
    // FP32 time/modulation islands replace BF16 originals; reserve their entire FP32 envelope.
    if r.acceleration == MossImageAcceleration::Dmd(MossDmdPrecision::UnmergedBf16) {
        extra = extra
            .saturating_add(4 * (256 * 3072 + 3072 * 3072 + 8 * 3072 * 3072 + 16 * 3072) as u64);
    }
    let pixels = u64::from(req.width).saturating_mul(u64::from(req.height));
    let tokens = pixels / 256 + u64::from(r.max_sequence_length);
    let b = images as u64;
    let branches = if matches!(r.acceleration, MossImageAcceleration::Dmd(_)) {
        1
    } else {
        1 + u64::from(r.guidance != 1.0)
            + u64::from(
                r.slg
                    .as_ref()
                    .is_some_and(|s| !s.layers.is_empty() && s.scale > 0.0),
            )
    };
    let attention = b
        .saturating_mul(24)
        .saturating_mul(tokens)
        .saturating_mul(tokens)
        .saturating_mul(8);
    let activations = b
        .saturating_mul(tokens)
        .saturating_mul(3072 * 4 * 24)
        .saturating_mul(branches)
        .saturating_add(attention);
    let text_work = b
        .saturating_mul(u64::from(r.max_sequence_length))
        .saturating_mul(4096 * 4 * 32);
    let decode = b.saturating_mul(pixels).saturating_mul(96 * 4 * 12);
    let overhead = 512 * 1024 * 1024;
    let owned_shard_transient = s
        .assets(r.acceleration)?
        .iter()
        .filter(|a| a.relative.ends_with(".safetensors"))
        .filter_map(|a| moss_image_asset(a.repo, a.revision, &a.relative))
        .map(|a| a.bytes)
        .max()
        .unwrap_or(0);
    // Same-buffer source authentication owns one complete shard until materialization completes.
    Ok(if staged {
        text.saturating_add(text_work)
            .max(dit.saturating_add(extra).saturating_add(activations))
            .max(vae.saturating_add(decode))
    } else {
        text.saturating_add(dit)
            .saturating_add(vae)
            .saturating_add(extra)
            .saturating_add(activations.max(text_work).max(decode))
    }
    .saturating_add(overhead)
    .saturating_add(owned_shard_transient))
}
pub(crate) fn check_request(req: &GenerationRequest) -> gen_core::Result<()> {
    if req.memory.is_some_and(|m| {
        m.tile_vae_decode
            || m.chunk_attention
            || m.stream_transformer_blocks
            || m.decode_tile_edge.is_some()
            || m.decode_overlap.is_some()
            || m.attention_chunk_size.is_some()
            || m.transformer_window_size.is_some()
            || m.transformer_window_component.is_some()
    }) {
        return Err(gen_core::Error::Unsupported(
            "MOSS does not implement VAE tiling, chunked attention or block streaming".into(),
        ));
    }
    Ok(())
}

fn architecture_facts(s: &Sources) -> MemoryArchitectureFacts {
    MemoryArchitectureFacts {
        attention_heads: Some(s.config.num_heads as u32),
        head_dim: Some((s.config.dim / s.config.num_heads) as u32),
        transformer_blocks: Some(s.config.num_layers as u32),
        patch_size: Some(1),
        latent_channels: Some(s.config.latent_channels as u32),
        vae_spatial_scale: Some(16),
        vae_temporal_scale: None,
        activation_dtype_width: Some(4),
    }
}
