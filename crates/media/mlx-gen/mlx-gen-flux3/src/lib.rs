//! # mlx-gen-flux3
//!
//! The **FLUX 3 Image Klein** provider crate for [`mlx-gen`](mlx_gen) (epic sc-26233; story
//! sc-26234 is the distilled text-to-image route `flux3_klein`): a native MLX port of the frozen
//! upstream `black-forest-labs/flux-3-image-klein` inference code — the Qwen3-VL-4B text encoder
//! (eight stacked hidden states, a pad-to-80 window), the 16×-downsampling autoencoder (8× VAE +
//! 2×2 patchify, BatchNorm-stat latents), the per-mode + shared single-stream DiT with 4-axis RoPE
//! and target-time (flow-map) conditioning, and the flow-map Euler sampler.
//!
//! The backend-neutral half (identity, resources, `config.json`, text window, schedule, id and
//! RoPE tables, request params) is [`gen_core::flux3`], shared with the Candle twin (sc-26235) and
//! the SceneWorks worker. The tensor work is here.
//!
//! ## Resources
//!
//! `LoadSpec::weights` is the package directory (`config.json` + `model.safetensors`);
//! `components["text_encoder"]` and `components["autoencoder"]` optionally override the bundled
//! `text_encoder/` snapshot and `ae.safetensors`. See `UPSTREAM.md` for the pinned revisions and
//! the coverage table, and `tests/fixtures/` for the committed miniature fixtures dumped by
//! `../tools/dump_flux3_*.py` from the frozen source.
//!
//! [`gen_core::flux3`]: mlx_gen::gen_core::flux3

pub mod autoencoder;

pub use autoencoder::Flux3AutoEncoder;
pub use mlx_gen::gen_core::flux3::{
    AUTOENCODER_COMPONENT, DISTILLED_MODEL_ID, DISTILLED_WEIGHTS_REPO, DISTILLED_WEIGHTS_REVISION,
    FAMILY, TEXT_ENCODER_COMPONENT, TEXT_ENCODER_UPSTREAM_REPO, TEXT_ENCODER_UPSTREAM_REVISION,
    UPSTREAM_CODE_REVISION,
};

#[cfg(test)]
mod tests {
    #[test]
    fn upstream_md_records_the_pinned_revisions() {
        let doc = include_str!("../UPSTREAM.md");
        for sha in [
            super::UPSTREAM_CODE_REVISION,
            super::DISTILLED_WEIGHTS_REVISION,
            super::TEXT_ENCODER_UPSTREAM_REVISION,
        ] {
            assert!(doc.contains(sha), "UPSTREAM.md must record {sha}");
        }
    }
}
