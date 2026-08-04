//! Snapshot-layout loaders for Chroma. The diffusers checkpoint tree is
//! `tokenizer/` (spiece + configs), `text_encoder/` (T5-XXL), `transformer/` (sharded Chroma DiT),
//! `vae/` (AutoencoderKL), `scheduler/`, `model_index.json`.
//!
//! T5 encoder, VAE, and the pack/unpack/sigma helpers are reused from `mlx-gen-flux`. Those shared
//! loaders packed-detect on `{base}.scales`, so a q4/q8 artifact builds quantized modules directly
//! with no dense transient. The Chroma-specific loading concerns are (1) T5 lives in `text_encoder/`
//! not flux's `text_encoder_2/`, (2) the tokenizer ships only `spiece.model`, so we load a vendored,
//! prebuilt `tokenizer.json` (materialized by `tools/build_chroma_t5_tokenizer.py`) — never the
//! network, and (3) Chroma publishes T5 at group 32, so the width is read from the component's own
//! `config.json` rather than assumed (sc-16462).

use std::path::Path;

use mlx_gen::tokenizer::{ChatTemplate, TextTokenizer, TokenizerConfig};
use mlx_gen::weights::Weights;
use mlx_gen::{Error, Result};
use mlx_gen_flux::T5TextEncoder;
use mlx_gen_z_image::vae::Vae;

use crate::config::{ChromaTransformerConfig, MAX_SEQUENCE_LENGTH};
use crate::transformer::ChromaTransformer;

/// The vendored T5-XXL tokenizer (google t5-v1.1-xxl, converted from Chroma's `spiece.model`).
const T5_TOKENIZER_JSON: &str = include_str!("../assets/t5_tokenizer.json");

pub fn load_tokenizer() -> Result<TextTokenizer> {
    load_tokenizer_with_max_len(MAX_SEQUENCE_LENGTH)
}

/// The vendored T5 tokenizer at a given padded length (production uses [`MAX_SEQUENCE_LENGTH`]; the
/// parity tests use a smaller length — the mask logic is length-agnostic).
pub fn load_tokenizer_with_max_len(max_length: usize) -> Result<TextTokenizer> {
    let config = TokenizerConfig {
        max_length,
        // T5 `<pad>`.
        pad_token_id: 0,
        chat_template: ChatTemplate::None,
        pad_to_max_length: true,
    };
    TextTokenizer::from_json_str(T5_TOKENIZER_JSON, config).map_err(Into::into)
}

pub fn load_t5_encoder(root: &Path) -> Result<T5TextEncoder> {
    // Chroma diffusers layout: T5 is `text_encoder/` (FLUX puts it in `text_encoder_2/`).
    let component = root.join("text_encoder");
    let group_size = packed_group_size(&component, "T5")?;
    let w = Weights::from_dir(component)?;
    T5TextEncoder::from_weights_with_group_size(&w, "", group_size)
}

pub fn load_vae(root: &Path) -> Result<Vae> {
    // Identical AutoencoderKL layout to FLUX — reuse the flux loader (decoder/encoder remap included).
    mlx_gen_flux::load_vae(root)
}

/// The affine group size a packed component declares, or the codebase default for a dense one. A
/// packed artifact that omits the marker would be read at the wrong width and dequantize to garbage,
/// so an unreadable/absent marker on a PACKED component is a hard error, not a silent default.
fn packed_group_size(component: &Path, label: &str) -> Result<i32> {
    match mlx_gen::quant::packed_quant_bits_at(component)? {
        // Dense component: nothing is packed, so the group size is unused.
        None => Ok(mlx_gen::quant::DEFAULT_GROUP_SIZE),
        Some(_) => mlx_gen::quant::packed_quant_group_size_at(component)?.ok_or_else(|| {
            Error::Msg(format!(
                "chroma {label}: {} is packed but declares no quantization.group_size; refusing to \
                 guess a width that would dequantize to garbage",
                component.display()
            ))
        }),
    }
}

/// Pack a DENSE T5 source in place at the same `(bits, group_size)` the offline converter writes, so
/// the load-time seam and the shipped artifact are byte-identical. Packed sources no-op.
pub fn quantize_t5_for_dense_source(t5: &mut T5TextEncoder, bits: i32) -> Result<()> {
    t5.quantize_with_group_size(bits, crate::convert::T5_GROUP_SIZE)
}

/// The VAE analogue of [`quantize_t5_for_dense_source`].
pub fn quantize_vae_for_dense_source(vae: &mut Vae, bits: i32) -> Result<()> {
    vae.quantize(bits)
}

pub fn load_transformer(root: &Path, cfg: ChromaTransformerConfig) -> Result<ChromaTransformer> {
    let w = Weights::from_dir(root.join("transformer"))?;
    ChromaTransformer::from_weights(w, cfg)
}
