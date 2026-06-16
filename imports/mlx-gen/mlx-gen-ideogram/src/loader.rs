//! Real-weight loading from a converted Ideogram 4 MLX snapshot (produced by
//! `tools/convert_ideogram4_to_mlx.py`):
//! ```text
//!   <root>/text_encoder/model.safetensors   (Qwen3-VL, `language_model.*` + unused `visual.*`)
//!   <root>/transformer/model.safetensors    (E3)
//!   <root>/unconditional_transformer/...     (E3)
//!   <root>/vae/model.safetensors             (E4)
//! ```
//! The converted `text_encoder` keys map directly onto the encoder under the `"language_model"`
//! prefix — no remap. The `visual.*` vision-tower tensors are present but unused for T2I.

use std::path::Path;

use mlx_gen::weights::Weights;
use mlx_gen::Result;

use crate::config::Ideogram4TextEncoderConfig;
use crate::text_encoder::Ideogram4TextEncoder;

/// Load the Qwen3-VL text encoder from the converted `text_encoder` component.
pub fn load_text_encoder(root: &Path) -> Result<Ideogram4TextEncoder> {
    let w = Weights::from_dir(root.join("text_encoder"))?;
    Ideogram4TextEncoder::from_weights(
        &w,
        "language_model",
        &Ideogram4TextEncoderConfig::qwen3_vl_8b(),
    )
}
