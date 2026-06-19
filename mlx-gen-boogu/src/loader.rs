//! Real-checkpoint loading from a Boogu-Image-0.1 snapshot (standard diffusers multi-component
//! tree): `mllm/` (Qwen3-VL condition encoder), `transformer/` (DiT), `vae/` (FLUX.1 AutoencoderKL).

use std::path::Path;

use mlx_gen::weights::Weights;
use mlx_gen::Result;

use crate::config::BooguConfig;
use crate::text_encoder::{BooguTextEncoder, BooguTextEncoderConfig};
use crate::transformer::BooguTransformer;

/// Load the Qwen3-VL-8B condition encoder from a snapshot's `mllm/` dir. The text tower lives under
/// `model.language_model.*`; the visual tower + `lm_head` are loaded but unused for text-to-image.
pub fn load_text_encoder(root: impl AsRef<Path>) -> Result<BooguTextEncoder> {
    let w = Weights::from_dir(root.as_ref().join("mllm"))?;
    BooguTextEncoder::from_weights(
        &w,
        "model.language_model",
        &BooguTextEncoderConfig::qwen3_vl_8b(),
    )
}

/// Load the DiT from a snapshot's `transformer/` dir: parse the config, load the (identity-keyed)
/// weights, validate the architecture against the config, then assemble the model.
pub fn load_transformer(root: impl AsRef<Path>) -> Result<BooguTransformer> {
    let root = root.as_ref();
    let cfg = BooguConfig::from_snapshot(root)?;
    let w = Weights::from_dir(root.join("transformer"))?;
    crate::convert::validate_transformer(&w, &cfg)?;
    BooguTransformer::from_weights(&w, &cfg)
}
