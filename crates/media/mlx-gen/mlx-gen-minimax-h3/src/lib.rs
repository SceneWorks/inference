//! # mlx-gen-minimax-h3
//!
//! Native-Rust / MLX inference for **MiniMax-H3 (Hailuo 3.0)** (`MiniMaxAI/MiniMax-H3`,
//! Apache-2.0) — a joint audio+video generation family. This slice (sc-17140) is the crate
//! skeleton plus the **video VAE decode** path:
//!
//! - [`config`] carries the VAE geometry, the temporal knobs and the per-channel latent
//!   statistics, reconciling the two published configs (neither is sufficient alone);
//! - [`chunking`] models the `clip_length` / `token_drop` temporal chunk plan as pure integer
//!   arithmetic;
//! - [`rope`] implements the 3-D **partial** rotary embedding and its normalized token ids;
//! - [`blocks`] and [`decoder`] implement the 36-layer transformer decoder;
//! - [`vae`] assembles the de-normalize → chunk → decode → cross-fade decode path.
//!
//! The VAE is unusual: the encoder is a 3-D causal CNN but the **decoder is a transformer**
//! (36 layers, 2048 dim, ~5.2 B params) that performs all 16× spatial and 4× temporal upsampling
//! in its output projection. Chunk/overlap/drop bookkeeping — not the blocks — is where this port
//! is most easily wrong, so it is isolated and asserted independently of any tensor math.
//!
//! Not in this crate yet: the CNN encoder, the audio VAE (sc-17141), the DiT (sc-17144) and the
//! pipeline (sc-17146/17147). Nothing is registered with `mlx-gen-catalog` — there is no
//! generator to ship until the pipeline lands.

pub mod blocks;
pub mod chunking;
pub mod config;
pub mod decoder;
pub mod rope;
pub mod tensor;
pub mod vae;

pub use chunking::{ChunkSpan, TemporalGeometry, TemporalPlan};
pub use config::{
    MiniMaxH3VaeConfig, CLIP_LENGTH, DECODER_HEAD_DIM, DECODER_NUM_HEADS, DECODER_NUM_LAYERS,
    DECODER_NUM_REGISTER_TOKENS, DECODER_ROPE_DIM_RATIO, DECODER_ROPE_THETA, LATENTS_MEAN,
    LATENTS_STD, LATENT_CHANNELS, TOKEN_DROP, VAE_RATIO, VAE_RATIO_T,
};
pub use decoder::ViT3dDecoder;
pub use rope::{create_token_ids, Rope3d, RopeTables};
pub use vae::{split_fused_qkv, MiniMaxH3VideoVae};

/// The published model id this crate targets.
pub const MODEL_ID: &str = "minimax_h3";

/// Frame/pixel alignment the video decode implies — `VAE_RATIO` spatially.
pub const SIZE_MULTIPLE: u32 = VAE_RATIO as u32;

#[cfg(test)]
mod tests {
    use super::*;

    /// The crate ships no generator yet, so it is deliberately absent from `mlx-gen-catalog`.
    /// This pins the compression contract the later pipeline slices will build on.
    #[test]
    fn published_surface_is_the_vae_decode_contract() {
        assert_eq!(MODEL_ID, "minimax_h3");
        assert_eq!(SIZE_MULTIPLE, 16);
        assert_eq!(VAE_RATIO, 16);
        assert_eq!(VAE_RATIO_T, 4);
        assert_eq!(LATENT_CHANNELS, 24);
        assert_eq!(LATENTS_MEAN.len(), LATENT_CHANNELS);
        assert_eq!(LATENTS_STD.len(), LATENT_CHANNELS);
    }
}
