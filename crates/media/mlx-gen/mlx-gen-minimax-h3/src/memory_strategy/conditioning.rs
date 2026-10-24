//! Selected native VAE configs and weight readsets, without materializing any array.
use mlx_gen::gen_core::{
    h3_conditioning_memory::H3ConditioningMemoryFacts, h3_memory::NativeAllocationFacts,
    weightsmeta, Error, Result,
};
use std::path::Path;

/// Qwen's native constructor retains source-width embedding/affine triples and the tapped layers,
/// not the unused LM tail/norm/head. Vision weights form a separate resident trunk for grounded
/// requests; keeping that trunk in this route-independent bound covers both presentations.
pub(super) fn text_encoder_stream(
    path: &Path,
) -> Result<Option<mlx_gen::gen_core::StreamedWeightFacts>> {
    if weightsmeta::safetensors_path_bytes(path) == 0 {
        return Ok(None);
    }
    let text = std::fs::read_to_string(path.join("config.json"))?;
    let cfg = crate::text_encoder::MiniMaxH3TeConfig::from_config_json(&text)
        .map_err(|e| Error::Unsupported(e.to_string()))?;
    let count = cfg
        .out_layer()
        .map_err(|e| Error::Unsupported(e.to_string()))?
        + 1;
    let mut blocks = vec![0u64; count];
    let mut root_bytes = 0u64;
    let layers = format!("{}.layers.", crate::text_encoder::LM_PREFIX);
    let embedding = format!("{}.embed_tokens.", crate::text_encoder::LM_PREFIX);
    for h in weightsmeta::safetensors_path_tensor_headers(path)? {
        if let Some(suffix) = h.name.strip_prefix(&layers) {
            let (index, _) = suffix
                .split_once('.')
                .ok_or_else(|| Error::Unsupported(format!("malformed text layer {}", h.name)))?;
            let index = index
                .parse::<usize>()
                .map_err(|_| Error::Unsupported(format!("malformed text layer {}", h.name)))?;
            if let Some(block) = blocks.get_mut(index) {
                *block = block.saturating_add(h.data_bytes);
            }
        } else if h.name.starts_with(&embedding)
            || h.name
                .starts_with(&format!("{}.", crate::text_encoder::VISION_PREFIX))
        {
            root_bytes = root_bytes.saturating_add(h.data_bytes);
        }
    }
    if blocks.contains(&0) || root_bytes == 0 {
        return Err(Error::Unsupported(
            "H3 conditioner source omits a consumed embedding/layer".into(),
        ));
    }
    Ok(Some(mlx_gen::gen_core::StreamedWeightFacts {
        resident_bytes: root_bytes,
        stacks: vec![blocks],
    }))
}

pub(super) fn complete(
    root: &Path,
    precision: mlx_gen::Precision,
    facts: &mut NativeAllocationFacts,
) -> Result<()> {
    let video = root.join("vae");
    let audio = root.join("audio_vae");
    if weightsmeta::safetensors_path_bytes(&video) == 0
        || weightsmeta::safetensors_path_bytes(&audio) == 0
    {
        return Ok(());
    }
    let read = |path: &Path| {
        std::fs::read_to_string(path)
            .map_err(|e| Error::Unsupported(format!("{}: {e}", path.display())))
    };
    let cfg =
        crate::config::MiniMaxH3VaeConfig::from_diffusers_json(&read(&video.join("config.json"))?)
            .map_err(|e| Error::Unsupported(e.to_string()))?;
    let source = root.join("FL2VA/audio_vae");
    // The MLX loader requires the source triple. Never substitute a stock config for an actual
    // component whose constructor would fail or describe another source.
    let a = crate::audio_config::MiniMaxH3AudioVaeConfig::from_source_files(
        &read(&source.join("config.json"))?,
        &read(&source.join("config.yaml"))?,
        &read(&source.join("metadata.json"))?,
    )
    .map_err(|e| Error::Unsupported(e.to_string()))?;
    let width = if precision == mlx_gen::Precision::Fp32 {
        4
    } else {
        2
    };
    let video_headers = weightsmeta::safetensors_path_tensor_headers(&video)?;
    let target = if width == 4 {
        weightsmeta::Dtype::F32
    } else {
        weightsmeta::Dtype::BF16
    };
    let video_allocation =
        mlx_gen::gen_core::h3_memory::video_allocation(&video_headers, target, true)?;
    let stats = cfg.latent_channels as u64 * width.max(4) * 2;
    let stats_cast = if width == 4 {
        0
    } else {
        cfg.latent_channels as u64 * width
    };
    let video_bytes = video_allocation.resident_bytes.saturating_add(stats);
    let audio_bytes = mlx_gen::gen_core::h3_memory::audio_allocation(
        &audio,
        &crate::audio_vae_encoder::MiniMaxH3AudioVaeEncoder::tensor_names(&a),
        &crate::audio_vae::MiniMaxH3AudioVae::tensor_names(&a),
        4,
        true,
        a.latent_channels as u64,
    )?;
    let video_load = video_bytes.saturating_add(video_allocation.load_extra_bytes.max(stats_cast));
    facts.conditioning_resident_bytes = video_bytes.saturating_add(audio_bytes.encoder_resident);
    facts.conditioning_load_bytes =
        video_load.max(video_bytes.saturating_add(audio_bytes.encoder_load));
    facts.video_decode_resident_bytes = video_bytes;
    facts.video_decode_load_bytes = video_load;
    facts.audio_decode_resident_bytes = audio_bytes.decoder_resident;
    facts.audio_decode_load_bytes = audio_bytes.decoder_load;
    facts.conditioning = Some(H3ConditioningMemoryFacts {
        lazy_encode: true,
        unified_memory: true,
        video_forward_extra_bytes: video_allocation.forward_extra_bytes.max(stats_cast),
        audio_encode_forward_extra_bytes: audio_bytes.encoder_forward_extra,
        audio_decode_forward_extra_bytes: audio_bytes.decoder_forward_extra,
        video_element_bytes: width,
        latent_channels: cfg.latent_channels as u64,
        pixel_channels: cfg.out_channels as u64,
        encoder_channels: cfg.block_out_channels.iter().map(|&x| x as u64).collect(),
        encoder_spatial_strides: cfg
            .spatial_downsample_factors
            .iter()
            .map(|&x| x as u64)
            .collect(),
        encoder_temporal_strides: cfg
            .temporal_downsample_factors
            .iter()
            .map(|&x| x as u64)
            .collect(),
        encoder_layers_per_block: cfg.layers_per_block as u64,
        clip_length: cfg.clip_length as u64,
        token_drop: cfg.token_drop as u64,
        spatial_ratio: cfg.patch_size as u64,
        temporal_ratio: cfg.patch_size_t as u64,
        decoder_layers: cfg.num_layers as u64,
        decoder_heads: cfg.num_heads as u64,
        decoder_head_dim: cfg.head_dim as u64,
        decoder_ffn_mult: cfg.ffn_mult as u64,
        decoder_suffix_tokens: cfg.num_register_tokens as u64 + 1,
        tile_edge: super::DECODE_TILE_EDGE as u64,
        tile_overlap: super::DECODE_OVERLAP as u64,
        audio_decode_element_bytes: 4,
        audio_sample_rate: a.sample_rate as u64,
        audio_output_channels: a.output_channels as u64,
        audio_latent_channels: a.latent_channels as u64,
        audio_encoder_dim: a.encoder_dim as u64,
        audio_encoder_rates: a.encoder_rates.iter().map(|&x| x as u64).collect(),
        audio_attention_heads: crate::audio_vae_encoder::ATTN_PROJ_HEADS as u64,
        audio_decoder_dim: a.decoder_dim as u64,
        audio_upsample_rates: a.bigvgan.upsample_rates.iter().map(|&x| x as u64).collect(),
        audio_upsample_kernels: a
            .bigvgan
            .upsample_kernel_sizes
            .iter()
            .map(|&x| x as u64)
            .collect(),
        audio_resblock_kernels: a
            .bigvgan
            .resblock_kernel_sizes
            .iter()
            .map(|&x| x as u64)
            .collect(),
        audio_resblock_dilations: a
            .bigvgan
            .resblock_dilation_sizes
            .iter()
            .map(|v| v.iter().map(|&x| x as u64).collect())
            .collect(),
    });
    Ok(())
}
