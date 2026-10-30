//! MiniMax-H3 conditioning/VAE memory facts carried by
//! [`NativeAllocationFacts`](crate::h3_memory::NativeAllocationFacts). The workspace pricing lives
//! in `gen_core_h3::h3_conditioning_memory` (sc-27316).

/// Facts copied from the selected native VAE configurations and materialized video dtype.
/// Audio encode is F32; decode follows its selected native materialized dtype (MLX F32,
/// Candle self.dtype). Neither is inferred from the DiT quantization tier.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct H3ConditioningMemoryFacts {
    pub lazy_encode: bool,
    pub unified_memory: bool,
    pub video_element_bytes: u64,
    /// Largest selected native cast/fusion temporary that may first materialize INSIDE forward.
    /// Endpoint component residency is counted elsewhere; this overlaps one active serial op,
    /// not merely a constructor-only maximum and not every layer/reference at once.
    pub video_forward_extra_bytes: u64,
    pub audio_encode_forward_extra_bytes: u64,
    pub audio_decode_forward_extra_bytes: u64,
    pub latent_channels: u64,
    pub pixel_channels: u64,
    pub encoder_channels: Vec<u64>,
    pub encoder_spatial_strides: Vec<u64>,
    pub encoder_temporal_strides: Vec<u64>,
    pub encoder_layers_per_block: u64,
    pub clip_length: u64,
    pub token_drop: u64,
    pub spatial_ratio: u64,
    pub temporal_ratio: u64,
    pub decoder_layers: u64,
    pub decoder_heads: u64,
    pub decoder_head_dim: u64,
    pub decoder_ffn_mult: u64,
    pub decoder_suffix_tokens: u64,
    pub tile_edge: u64,
    pub tile_overlap: u64,
    pub audio_decode_element_bytes: u64,
    pub audio_sample_rate: u64,
    pub audio_output_channels: u64,
    pub audio_latent_channels: u64,
    pub audio_encoder_dim: u64,
    pub audio_encoder_rates: Vec<u64>,
    pub audio_attention_heads: u64,
    pub audio_decoder_dim: u64,
    pub audio_upsample_rates: Vec<u64>,
    pub audio_upsample_kernels: Vec<u64>,
    pub audio_resblock_kernels: Vec<u64>,
    pub audio_resblock_dilations: Vec<Vec<u64>>,
}
impl Default for H3ConditioningMemoryFacts {
    /// Published native configuration, useful for CPU fixtures. Providers must overwrite these
    /// from the selected component configs/dtype, not use this as a missing-profile fallback.
    fn default() -> Self {
        Self {
            lazy_encode: true,
            unified_memory: true,
            video_element_bytes: 4,
            video_forward_extra_bytes: 0,
            audio_encode_forward_extra_bytes: 0,
            audio_decode_forward_extra_bytes: 0,
            latent_channels: 24,
            pixel_channels: 3,
            encoder_channels: vec![128, 256, 256, 512, 512, 1024],
            encoder_spatial_strides: vec![2, 2, 2, 2, 1, 1],
            encoder_temporal_strides: vec![1, 2, 2, 1, 1, 1],
            encoder_layers_per_block: 2,
            clip_length: 17,
            token_drop: 3,
            spatial_ratio: 16,
            temporal_ratio: 4,
            decoder_layers: 36,
            decoder_heads: 32,
            decoder_head_dim: 64,
            decoder_ffn_mult: 4,
            decoder_suffix_tokens: 5,
            tile_edge: 256,
            tile_overlap: 64,
            audio_decode_element_bytes: 4,
            audio_sample_rate: 32_000,
            audio_output_channels: 2,
            audio_latent_channels: 32,
            audio_encoder_dim: 64,
            audio_encoder_rates: vec![2, 4, 4, 5, 5],
            audio_attention_heads: 8,
            audio_decoder_dim: 1024,
            audio_upsample_rates: vec![5, 5, 2, 2, 2, 2, 2],
            audio_upsample_kernels: vec![9, 9, 4, 4, 4, 4, 4],
            audio_resblock_kernels: vec![3, 7, 11],
            audio_resblock_dilations: vec![vec![1, 3, 5]; 3],
        }
    }
}
