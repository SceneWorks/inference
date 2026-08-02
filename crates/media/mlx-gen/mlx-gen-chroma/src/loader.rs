//! Snapshot-layout loaders for Chroma. The diffusers checkpoint tree is
//! `tokenizer/` (spiece + configs), `text_encoder/` (T5-XXL), `transformer/` (sharded Chroma DiT),
//! `vae/` (AutoencoderKL), `scheduler/`, `model_index.json`.
//!
//! T5 encoder, VAE, and the pack/unpack/sigma helpers are reused from `mlx-gen-flux`. Those shared
//! loaders packed-detect the T5 embeddings/Linears and the VAE mid-block attention, so q4/q8
//! artifacts construct quantized modules directly without first materializing dense weights. The only
//! Chroma-specific loading concerns are (1) T5 lives in `text_encoder/` not flux's `text_encoder_2/`,
//! and (2) the tokenizer ships only `spiece.model`, so we load a vendored, prebuilt `tokenizer.json`
//! (materialized by `tools/build_chroma_t5_tokenizer.py`) — never the network.

use std::collections::BTreeSet;
use std::path::Path;

use mlx_gen::tokenizer::{ChatTemplate, TextTokenizer, TokenizerConfig};
use mlx_gen::weights::Weights;
use mlx_gen::{Error, Result};
use mlx_gen_flux::T5TextEncoder;
use mlx_gen_z_image::vae::Vae;
use mlx_rs::Dtype;

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
    let primary_bits = mlx_gen::quant::packed_quant_bits_at(&component)?;
    let group_size = mlx_gen::quant::packed_quant_group_size_at(&component)?
        .unwrap_or(mlx_gen::quant::DEFAULT_GROUP_SIZE);
    let residual_bits = t5_residual_bits(&component)?;
    let w = Weights::from_dir(component)?;
    validate_t5_progressive_surface(&w, primary_bits, residual_bits, group_size)?;
    T5TextEncoder::from_weights_with_group_size(&w, "", group_size)
}

/// Apply the same progressive packed-T5 policy used by the offline Chroma converter to a dense
/// source. Keeping this provider-owned seam shared by production and the real-weight identity test
/// prevents the test from validating a hand-written quantization recipe that production never uses.
pub fn quantize_t5_for_dense_source(
    t5: &mut T5TextEncoder,
    bits: i32,
    group_size: i32,
) -> Result<()> {
    t5.quantize_progressive(bits, crate::convert::T5_RESIDUAL_BITS, group_size)
}

fn t5_residual_bits(component: &Path) -> Result<Option<i32>> {
    let config_path = component.join("config.json");
    let bytes = match std::fs::read(&config_path) {
        Ok(bytes) => bytes,
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => return Ok(None),
        Err(error) => return Err(error.into()),
    };
    let config: serde_json::Value = serde_json::from_slice(&bytes).map_err(|error| {
        Error::Msg(format!(
            "chroma T5: parse {}: {error}",
            config_path.display()
        ))
    })?;
    let Some(value) = config
        .get("quantization")
        .and_then(|value| value.get("residual_bits"))
    else {
        return Ok(None);
    };
    let bits = value.as_i64().ok_or_else(|| {
        Error::Msg(format!(
            "chroma T5: {} quantization.residual_bits must be an integer",
            config_path.display()
        ))
    })?;
    let bits = i32::try_from(bits).map_err(|_| {
        Error::Msg(format!(
            "chroma T5: {} quantization.residual_bits is out of range",
            config_path.display()
        ))
    })?;
    if bits != crate::convert::T5_RESIDUAL_BITS {
        return Err(Error::Msg(format!(
            "chroma T5: unsupported packed residual width {bits}; expected Q{}",
            crate::convert::T5_RESIDUAL_BITS
        )));
    }
    Ok(Some(bits))
}

fn validate_t5_progressive_surface(
    w: &Weights,
    primary_bits: Option<i32>,
    residual_bits: Option<i32>,
    group_size: i32,
) -> Result<()> {
    let keys = w.keys().map(str::to_string).collect::<Vec<_>>();
    let primary_bases = keys
        .iter()
        .filter_map(|key| key.strip_suffix(".scales"))
        .filter(|base| !base.ends_with(".residual"))
        .collect::<BTreeSet<_>>();
    let residual_bases = keys
        .iter()
        .filter_map(|key| {
            ["weight", "scales", "biases"]
                .into_iter()
                .find_map(|suffix| key.strip_suffix(&format!(".residual.{suffix}")))
        })
        .collect::<BTreeSet<_>>();
    match primary_bits {
        None if !primary_bases.is_empty() => {
            return Err(Error::Msg(
                "chroma T5: packed primary tensors require quantization.bits provenance".into(),
            ))
        }
        Some(_) if primary_bases.is_empty() => return Err(Error::Msg(
            "chroma T5: quantization marker is present but no primary packed weights were found"
                .into(),
        )),
        _ => {}
    }
    if let Some(primary_bits) = primary_bits {
        for base in &primary_bases {
            let weight = w.require(&format!("{base}.weight"))?;
            let scales = w.require(&format!("{base}.scales"))?;
            let biases = w.require(&format!("{base}.biases"))?;
            validate_packed_companions(base, scales, biases)?;
            let actual_bits = mlx_gen::quant::packed_bits(weight, scales, group_size)?;
            if actual_bits != primary_bits {
                return Err(Error::Msg(format!(
                    "chroma T5: {base} primary is Q{actual_bits}, but config declares Q{primary_bits}"
                )));
            }
        }
        for key in keys
            .iter()
            .filter(|key| key.ends_with(".weight") && !key.ends_with(".residual.weight"))
        {
            let base = key.strip_suffix(".weight").expect("filtered suffix");
            if primary_bases.contains(base) {
                continue;
            }
            let weight = w.require(key)?;
            let shape = weight.shape();
            if weight.dtype() == Dtype::Uint32 {
                return Err(Error::Msg(format!(
                    "chroma T5: {base} has packed codes but no scales"
                )));
            }
            if shape.len() == 2 && shape[1] >= group_size && shape[1] % group_size == 0 {
                return Err(Error::Msg(format!(
                    "chroma T5: progressive packed surface is incomplete; {base}.weight remains dense"
                )));
            }
        }
    }
    if residual_bits.is_none() {
        if !residual_bases.is_empty() {
            return Err(Error::Msg(
                "chroma T5: residual tensors require quantization.residual_bits provenance".into(),
            ));
        }
        return Ok(());
    }
    let residual_bits = residual_bits.ok_or_else(|| {
        Error::Msg("chroma T5: residual bit-width disappeared after validation".into())
    })?;
    if primary_bits.is_none() {
        return Err(Error::Msg(
            "chroma T5: residual packing requires quantization.bits provenance for the primary term"
                .into(),
        ));
    }
    for residual_base in &residual_bases {
        if !primary_bases.contains(residual_base) {
            return Err(Error::Msg(format!(
                "chroma T5: residual {residual_base} has no primary packed term"
            )));
        }
        for suffix in ["weight", "scales", "biases"] {
            let key = format!("{residual_base}.residual.{suffix}");
            if w.get(&key).is_none() {
                return Err(Error::Msg(format!(
                    "chroma T5: progressive packed surface is incomplete; missing {key}"
                )));
            }
        }
    }
    for base in primary_bases {
        for suffix in ["weight", "scales", "biases"] {
            let key = format!("{base}.residual.{suffix}");
            if w.get(&key).is_none() {
                return Err(Error::Msg(format!(
                    "chroma T5: progressive packed surface is incomplete; missing {key}"
                )));
            }
        }
        let residual_weight = w.require(&format!("{base}.residual.weight"))?;
        let residual_scales = w.require(&format!("{base}.residual.scales"))?;
        let residual_biases = w.require(&format!("{base}.residual.biases"))?;
        validate_packed_companions(
            &format!("{base}.residual"),
            residual_scales,
            residual_biases,
        )?;
        let actual_bits =
            mlx_gen::quant::packed_bits(residual_weight, residual_scales, group_size)?;
        if actual_bits != residual_bits {
            return Err(Error::Msg(format!(
                "chroma T5: {base} residual is Q{actual_bits}, but config declares Q{residual_bits}"
            )));
        }
    }
    Ok(())
}

fn validate_packed_companions(
    base: &str,
    scales: &mlx_rs::Array,
    biases: &mlx_rs::Array,
) -> Result<()> {
    if scales.shape() != biases.shape() || scales.dtype() != biases.dtype() {
        return Err(Error::Msg(format!(
            "chroma T5: {base} scales/biases geometry or dtype differs ({:?} {:?} vs {:?} {:?})",
            scales.shape(),
            scales.dtype(),
            biases.shape(),
            biases.dtype()
        )));
    }
    Ok(())
}

pub fn load_vae(root: &Path) -> Result<Vae> {
    // Identical AutoencoderKL layout to FLUX — reuse the flux loader (decoder/encoder remap included).
    mlx_gen_flux::load_vae(root)
}

pub fn load_transformer(root: &Path, cfg: ChromaTransformerConfig) -> Result<ChromaTransformer> {
    let w = Weights::from_dir(root.join("transformer"))?;
    ChromaTransformer::from_weights(w, cfg)
}

#[cfg(test)]
mod tests {
    use super::*;
    use mlx_rs::Array;
    use std::collections::HashMap;

    #[test]
    fn residual_marker_is_optional_but_strict_when_present() {
        let root =
            std::env::temp_dir().join(format!("chroma-t5-residual-marker-{}", std::process::id()));
        std::fs::remove_dir_all(&root).ok();
        std::fs::create_dir_all(&root).unwrap();
        assert_eq!(t5_residual_bits(&root).unwrap(), None);
        std::fs::write(
            root.join("config.json"),
            r#"{"quantization":{"bits":8,"group_size":64,"residual_bits":4}}"#,
        )
        .unwrap();
        assert_eq!(t5_residual_bits(&root).unwrap(), Some(4));
        std::fs::write(
            root.join("config.json"),
            r#"{"quantization":{"bits":8,"group_size":64,"residual_bits":8}}"#,
        )
        .unwrap();
        assert!(t5_residual_bits(&root).is_err());
        std::fs::remove_dir_all(root).ok();
    }

    fn progressive_fixture(
        residual_weight_cols: i32,
        dense_hole: bool,
        primary_bias_cols: i32,
    ) -> Weights {
        let base = "encoder.block.0.layer.0.SelfAttention.q";
        let mut map = HashMap::new();
        map.insert(
            format!("{base}.weight"),
            Array::zeros::<u32>(&[2, 32]).unwrap(),
        );
        map.insert(
            format!("{base}.scales"),
            Array::zeros::<f32>(&[2, 2]).unwrap(),
        );
        map.insert(
            format!("{base}.biases"),
            Array::zeros::<f32>(&[2, primary_bias_cols]).unwrap(),
        );
        map.insert(
            format!("{base}.residual.weight"),
            Array::zeros::<u32>(&[2, residual_weight_cols]).unwrap(),
        );
        map.insert(
            format!("{base}.residual.scales"),
            Array::zeros::<f32>(&[2, 2]).unwrap(),
        );
        map.insert(
            format!("{base}.residual.biases"),
            Array::zeros::<f32>(&[2, 2]).unwrap(),
        );
        if dense_hole {
            map.insert(
                "encoder.block.0.layer.1.DenseReluDense.wi_0.weight".into(),
                Array::zeros::<f32>(&[2, 128]).unwrap(),
            );
        }
        Weights::from_map(map)
    }

    #[test]
    fn progressive_surface_requires_complete_q4_residuals() {
        validate_t5_progressive_surface(&progressive_fixture(16, false, 2), Some(8), Some(4), 64)
            .unwrap();
        assert!(
            validate_t5_progressive_surface(
                &progressive_fixture(32, false, 2),
                Some(8),
                Some(4),
                64,
            )
            .is_err(),
            "a Q8 residual must not pass a Q4 marker"
        );
        assert!(
            validate_t5_progressive_surface(&progressive_fixture(16, false, 2), Some(8), None, 64)
                .is_err(),
            "residual tensors without provenance must fail"
        );
        assert!(
            validate_t5_progressive_surface(
                &progressive_fixture(16, false, 2),
                Some(4),
                Some(4),
                64,
            )
            .is_err(),
            "a Q8 primary must not pass a Q4 marker"
        );
        assert!(
            validate_t5_progressive_surface(&progressive_fixture(16, false, 2), None, None, 64)
                .is_err(),
            "packed primaries without provenance must fail"
        );
        assert!(
            validate_t5_progressive_surface(
                &progressive_fixture(16, true, 2),
                Some(8),
                Some(4),
                64,
            )
            .is_err(),
            "a group-packable dense weight must not pass a progressive artifact marker"
        );
        assert!(
            validate_t5_progressive_surface(
                &progressive_fixture(16, false, 1),
                Some(8),
                Some(4),
                64,
            )
            .is_err(),
            "packed scales and biases must have identical geometry"
        );
    }
}
