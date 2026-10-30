//! Selected native VAE configs and weight readsets, without materializing any array.
use candle_gen::gen_core::{
    h3_conditioning_memory::H3ConditioningMemoryFacts, h3_memory::NativeAllocationFacts,
    weightsmeta, Error, Result,
};
use std::path::Path;

pub(super) fn complete(root: &Path, facts: &mut NativeAllocationFacts) -> Result<()> {
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
    // The Candle loader defaults only when its source config.yaml is absent.
    let a = if source.join("config.yaml").is_file() {
        crate::audio_config::MiniMaxH3AudioVaeConfig::from_source_files(
            &read(&source.join("config.json"))?,
            &read(&source.join("config.yaml"))?,
            &read(&source.join("metadata.json"))?,
        )
        .map_err(|e| Error::Unsupported(e.to_string()))?
    } else {
        crate::audio_config::MiniMaxH3AudioVaeConfig::default()
    };
    let width = 2;
    let video_bytes = weightsmeta::materialized_path_bytes(&video, width)?
        .saturating_add(cfg.latent_channels as u64 * width * 2);
    let audio_bytes = gen_core_h3::h3_memory::audio_allocation(
        &audio,
        &crate::audio_vae_encoder::MiniMaxH3AudioVaeEncoder::tensor_names(&a),
        &crate::audio_vae::MiniMaxH3AudioVae::tensor_names(&a),
        width,
        false,
        a.latent_channels as u64,
    )?;
    facts.conditioning_resident_bytes = video_bytes.saturating_add(audio_bytes.encoder_resident);
    let video_headers = weightsmeta::safetensors_path_tensor_headers(&video)?;
    let video_allocation =
        gen_core_h3::h3_memory::video_allocation(&video_headers, weightsmeta::Dtype::BF16, false)?;
    let stats_cast = cfg.latent_channels as u64 * 4;
    let video_load = video_bytes.saturating_add(video_allocation.load_extra_bytes.max(stats_cast));
    facts.conditioning_load_bytes =
        video_load.max(video_bytes.saturating_add(audio_bytes.encoder_load));
    let headers = weightsmeta::safetensors_path_tensor_headers(&video)?;
    let decode_headers = headers
        .into_iter()
        .filter(|h| h.name.starts_with("decoder.") || h.name.starts_with("post_quant_conv."))
        .collect::<Vec<_>>();
    // load_decode_only never materializes the CNN encoder on the CUDA device.
    facts.video_decode_resident_bytes =
        weightsmeta::materialized_header_bytes(&decode_headers, width, &video)?
            .saturating_add(cfg.latent_channels as u64 * width * 2);
    let decode_allocation =
        gen_core_h3::h3_memory::video_allocation(&decode_headers, weightsmeta::Dtype::BF16, false)?;
    facts.video_decode_load_bytes = facts
        .video_decode_resident_bytes
        .saturating_add(decode_allocation.load_extra_bytes.max(stats_cast));
    facts.audio_decode_resident_bytes = audio_bytes.decoder_resident;
    facts.audio_decode_load_bytes = audio_bytes.decoder_load;
    facts.conditioning = Some(H3ConditioningMemoryFacts {
        lazy_encode: false,
        unified_memory: false,
        video_forward_extra_bytes: 0,
        audio_encode_forward_extra_bytes: 0,
        audio_decode_forward_extra_bytes: 0,
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
        tile_edge: crate::spatial_tiling::TILE_SAMPLE_MIN_SIZE as u64,
        tile_overlap: crate::spatial_tiling::TILE_SAMPLE_MIN_OVERLAP as u64,
        audio_decode_element_bytes: width,
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
