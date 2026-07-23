//! Real-checkpoint loading from a Krea 2 snapshot (standard diffusers multi-component tree):
//! `text_encoder/` (Qwen3-VL-4B condition encoder), `transformer/` (single-stream DiT), `vae/`
//! (Qwen-Image `AutoencoderKLQwenImage`, loaded via [`crate::vae::load_vae`]). The transformer +
//! text-encoder checkpoints are identity-keyed (diffusers names = the module tree), so
//! [`Weights::from_dir`] drops straight in; the VAE remap lives in `mlx-gen-qwen-image`.

use std::path::Path;

use mlx_gen::weights::Weights;
use mlx_gen::{Error, Result};
use mlx_gen_boogu::VisionTower;
use mlx_rs::ops::multiply;
use mlx_rs::Dtype;

use crate::config::Krea2Config;
use crate::text_encoder::{krea_vision_config, KreaTeConfig, KreaTextEncoder};
use crate::transformer::Krea2Transformer;

fn prepare_text_weights(mut w: Weights) -> Result<Weights> {
    let packed: std::collections::HashSet<String> = w
        .keys()
        .filter_map(|key| key.strip_suffix(".scales").map(str::to_owned))
        .collect();
    w.cast_matching(mlx_rs::Dtype::Bfloat16, |key| {
        key.starts_with("language_model.")
            && key.ends_with(".weight")
            && !key.contains("norm")
            && !packed.contains(key.strip_suffix(".weight").unwrap_or(key))
    })?;
    w.cast_matching(mlx_rs::Dtype::Float32, |key| {
        key.starts_with("language_model.") && key.ends_with("norm.weight")
    })?;
    Ok(w)
}

/// Load the Qwen3-VL-4B condition encoder from a snapshot's `text_encoder/` dir. The text tower lives
/// under `language_model.*`; the visual tower (`visual.*`) is assembled separately by
/// [`load_vision_tower`] only when image-grounded (edit) encoding is needed.
pub fn load_text_encoder(root: impl AsRef<Path>) -> Result<KreaTextEncoder> {
    let root = root.as_ref();
    let cfg = KreaTeConfig::from_snapshot(root)?;
    let w = prepare_text_weights(Weights::from_dir(root.join("text_encoder"))?)?;
    KreaTextEncoder::from_weights(&w, "language_model", &cfg)
}

/// Load the Qwen3-VL-4B **vision tower** from the same `text_encoder/` dir (epic 10871 P2.1, sc-10879):
/// the `visual.*` subtree that text-to-image never assembles. Casts the (small, parity-grade) vision
/// subtree to f32 before building — mirroring boogu's `load_vision_tower` — and feeds the shared
/// [`mlx_gen_boogu::VisionTower`] the Krea-4B [`krea_vision_config`]. Krea keys are `visual.*` (diffusers
/// naming), unlike boogu's `model.visual.*`.
pub fn load_vision_tower(root: impl AsRef<Path>) -> Result<VisionTower> {
    let root = root.as_ref();
    let mut w = Weights::from_dir(root.join("text_encoder"))?;
    let keys: Vec<String> = w
        .keys()
        .filter(|k| k.starts_with("visual."))
        .map(String::from)
        .collect();
    for k in keys {
        let t = w.require(&k)?.as_dtype(mlx_rs::Dtype::Float32)?;
        w.insert(k, t);
    }
    VisionTower::from_weights(&w, krea_vision_config(), "visual")
}

/// Load the single-stream DiT from a snapshot's `transformer/` dir: parse + validate the config, load
/// the (identity-keyed diffusers) weights, validate the architecture against the config, then assemble
/// the model. A pre-quantized snapshot loads through the same path (`quant::lin` auto-detects packed
/// keys); a dense bf16 build is quantized later via [`crate::pipeline::KreaPipeline::quantize`].
pub fn load_transformer(root: impl AsRef<Path>) -> Result<Krea2Transformer> {
    let root = root.as_ref();
    let cfg = Krea2Config::from_snapshot(root)?;
    let w = Weights::from_dir(root.join("transformer"))?;
    crate::convert::validate_transformer(&w, &cfg)?;
    Krea2Transformer::from_weights(&w, &cfg)
}

/// Validate and dequantize the non-rotated ComfyUI int8-tensorwise convention (sc-14023).
///
/// The app detector only has the safetensors header. Here, before any dequantization, every I8
/// projection must carry a real U8 JSON descriptor with `format=int8_tensorwise`, `per_row=true`, and
/// no `convrot` field, plus an F32 `[out]` or `[out,1]` scale. The consumed companions are removed so
/// the existing strict native-key remap still sees exactly the dense DiT surface.
fn dequant_plain_int8_tensorwise(mut native: Weights) -> Result<Weights> {
    let int8_weights: Vec<String> = native
        .keys()
        .filter(|key| {
            native
                .get(key)
                .is_some_and(|tensor| tensor.dtype() == Dtype::Int8)
        })
        .map(str::to_owned)
        .collect();
    let descriptors: Vec<String> = native
        .keys()
        .filter(|key| key.ends_with(".comfy_quant"))
        .map(str::to_owned)
        .collect();

    if int8_weights.is_empty() {
        if descriptors.is_empty() {
            return Ok(native);
        }
        return Err(Error::Msg(format!(
            "krea plain int8: found {} `.comfy_quant` descriptor(s) but no I8 weight tensors",
            descriptors.len()
        )));
    }

    for weight_key in int8_weights {
        let Some(base) = weight_key.strip_suffix(".weight") else {
            return Err(Error::Msg(format!(
                "krea plain int8: I8 tensor `{weight_key}` is not a projection `.weight`"
            )));
        };
        let weight = native.require(&weight_key)?;
        let [rows, _cols] = weight.shape() else {
            return Err(Error::Msg(format!(
                "krea plain int8: `{weight_key}` must be rank-2 [out,in], got {:?}",
                weight.shape()
            )));
        };
        let rows = *rows;

        let descriptor_key = format!("{base}.comfy_quant");
        let descriptor = native.require(&descriptor_key).map_err(|_| {
            Error::Msg(format!(
                "krea plain int8: `{weight_key}` is missing `{descriptor_key}`"
            ))
        })?;
        if descriptor.dtype() != Dtype::Uint8 || descriptor.shape().len() != 1 {
            return Err(Error::Msg(format!(
                "krea plain int8: `{descriptor_key}` must be a rank-1 U8 JSON blob"
            )));
        }
        let descriptor_bytes = descriptor.try_as_slice::<u8>().map_err(|error| {
            Error::Msg(format!(
                "krea plain int8: could not read `{descriptor_key}`: {error}"
            ))
        })?;
        let json: serde_json::Value =
            serde_json::from_slice(descriptor_bytes).map_err(|error| {
                Error::Msg(format!(
                    "krea plain int8: `{descriptor_key}` is not valid JSON: {error}"
                ))
            })?;
        if json.get("format").and_then(serde_json::Value::as_str) != Some("int8_tensorwise") {
            return Err(Error::Msg(format!(
                "krea plain int8: `{descriptor_key}` must declare format `int8_tensorwise`"
            )));
        }
        if json.get("per_row").and_then(serde_json::Value::as_bool) != Some(true) {
            return Err(Error::Msg(format!(
                "krea plain int8: `{descriptor_key}` must declare `per_row: true`"
            )));
        }
        if json.get("convrot").is_some() {
            return Err(Error::Msg(format!(
                "krea plain int8: `{descriptor_key}` contains `convrot`; rotated checkpoints are not \
                 the plain MLX int8 format"
            )));
        }

        let scale_key = format!("{base}.weight_scale");
        let scale = native.require(&scale_key).map_err(|_| {
            Error::Msg(format!(
                "krea plain int8: `{weight_key}` is missing `{scale_key}`"
            ))
        })?;
        if scale.dtype() != Dtype::Float32 {
            return Err(Error::Msg(format!(
                "krea plain int8: `{scale_key}` must be F32, got {:?}",
                scale.dtype()
            )));
        }
        if scale.shape() != [rows] && scale.shape() != [rows, 1] {
            return Err(Error::Msg(format!(
                "krea plain int8: `{scale_key}` must be [{rows}] or [{rows},1], got {:?}",
                scale.shape()
            )));
        }

        let codes = native
            .remove(&weight_key)
            .ok_or_else(|| Error::MissingTensor(weight_key.clone()))?
            .as_dtype(Dtype::Float32)?;
        let scale = native
            .remove(&scale_key)
            .ok_or_else(|| Error::MissingTensor(scale_key.clone()))?;
        let scale = if scale.shape().len() == 1 {
            scale.reshape(&[rows, 1])?
        } else {
            scale
        };
        let dense = multiply(&codes, &scale)?.as_dtype(Dtype::Bfloat16)?;
        // MLX is lazy: materialize projection-by-projection so the dense model does not retain a
        // graph edge to every removed I8 code/scale buffer (which would keep both the 13.5 GB source
        // and the BF16 reconstruction alive for the whole load).
        dense.eval()?;
        native.insert(weight_key, dense);
        native.remove(&descriptor_key);
    }

    if let Some(orphan) = native.keys().find(|key| key.ends_with(".comfy_quant")) {
        return Err(Error::Msg(format!(
            "krea plain int8: `{orphan}` does not describe an I8 projection weight"
        )));
    }
    Ok(native)
}

/// Load a community single-file Krea 2 DiT through the shared native→diffusers remap.
///
/// Dense bf16 files pass through unchanged. Plain int8-per-row files are descriptor-validated and
/// dequantized first as `codes.i8 * weight_scale` with no rotation. The remapped set then receives the
/// same architecture coverage/shape validation and transformer assembly as the published snapshot.
/// `cfg` comes from the resident base snapshot because the single file has no `config.json`.
pub fn load_transformer_from_native_file(
    dit_file: impl AsRef<Path>,
    cfg: &Krea2Config,
) -> Result<Krea2Transformer> {
    let native = dequant_plain_int8_tensorwise(Weights::from_file(dit_file.as_ref())?)?;
    let mut remapped = crate::native_remap::remap_native_dit_to_diffusers(native)?;
    // Reshape any flat per-block modulation table (`[6·hidden]`) to the diffusers 2-D `[6, hidden]` so
    // the set is shape-identical to a snapshot load and `validate_transformer`'s shape check passes.
    crate::native_remap::normalize_modulation_tables(&mut remapped)?;
    crate::convert::validate_transformer(&remapped, cfg)?;
    Krea2Transformer::from_weights(&remapped, cfg)
}

#[cfg(test)]
mod tests {
    use super::*;
    use mlx_rs::Array;

    fn plain_int8_weights(descriptor: &str, scale: Array) -> Weights {
        let mut weights = Weights::empty();
        weights.insert(
            "model.diffusion_model.blocks.0.attn.wq.weight",
            Array::from_slice(&[1_i8, -2, 3, -4, 5, -6], &[2, 3]),
        );
        weights.insert("model.diffusion_model.blocks.0.attn.wq.weight_scale", scale);
        weights.insert(
            "model.diffusion_model.blocks.0.attn.wq.comfy_quant",
            Array::from_slice(descriptor.as_bytes(), &[descriptor.len() as i32]),
        );
        weights
    }

    #[test]
    fn plain_int8_dequants_per_row_without_rotation() {
        let weights = plain_int8_weights(
            r#"{"format":"int8_tensorwise","per_row":true}"#,
            Array::from_slice(&[0.5_f32, 2.0], &[2, 1]),
        );
        let dequant = dequant_plain_int8_tensorwise(weights).unwrap();
        assert!(dequant
            .get("model.diffusion_model.blocks.0.attn.wq.weight_scale")
            .is_none());
        assert!(dequant
            .get("model.diffusion_model.blocks.0.attn.wq.comfy_quant")
            .is_none());
        let got = dequant
            .require("model.diffusion_model.blocks.0.attn.wq.weight")
            .unwrap()
            .as_dtype(Dtype::Float32)
            .unwrap();
        assert_eq!(got.as_slice::<f32>(), &[0.5, -1.0, 1.5, -8.0, 10.0, -12.0]);
    }

    #[test]
    fn plain_int8_rejects_convrot_or_wrong_descriptor() {
        for (descriptor, expected) in [
            (
                r#"{"format":"int8_tensorwise","per_row":true,"convrot":true}"#,
                "convrot",
            ),
            (r#"{"format":"mxfp4","per_row":true}"#, "int8_tensorwise"),
            (r#"{"format":"int8_tensorwise","per_row":false}"#, "per_row"),
        ] {
            let error = match dequant_plain_int8_tensorwise(plain_int8_weights(
                descriptor,
                Array::from_slice(&[0.5_f32, 2.0], &[2]),
            )) {
                Ok(_) => panic!("invalid descriptor must fail"),
                Err(error) => error.to_string(),
            };
            assert!(error.contains(expected), "{error}");
        }
    }

    #[test]
    fn plain_int8_rejects_non_per_row_scale_shape() {
        let error = match dequant_plain_int8_tensorwise(plain_int8_weights(
            r#"{"format":"int8_tensorwise","per_row":true}"#,
            Array::from_slice(&[0.5_f32], &[1]),
        )) {
            Ok(_) => panic!("wrong scale shape must fail"),
            Err(error) => error.to_string(),
        };
        assert!(
            error.contains("weight_scale") && error.contains("[2]"),
            "{error}"
        );
    }
}
