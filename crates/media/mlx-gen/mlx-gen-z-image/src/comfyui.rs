//! Community Z-Image component and fused-checkpoint loading.

use std::collections::HashMap;
use std::path::Path;

use mlx_gen::weights::Weights;
use mlx_gen::{Error, Result};
use mlx_rs::ops::multiply;
use mlx_rs::{Array, Dtype};

const DIT_DIM: i32 = 3840;
const VAE_LEVELS: usize = 4;

pub(crate) struct ComponentWeights {
    pub transformer: Weights,
    pub text_encoder: Weights,
    pub vae: Weights,
}

fn component_for_key<'a, 'b>(
    key: &'a str,
    transformer: &'b mut HashMap<String, Array>,
    text_encoder: &'b mut HashMap<String, Array>,
    vae: &'b mut HashMap<String, Array>,
) -> Option<(&'a str, &'b mut HashMap<String, Array>)> {
    if let Some(key) = key.strip_prefix("model.diffusion_model.") {
        return Some((key, transformer));
    }
    if let Some(key) = key.strip_prefix("transformer.") {
        return Some((key, transformer));
    }
    if let Some(key) = key.strip_prefix("conditioner.embedders.0.transformer.") {
        return Some((key, text_encoder));
    }
    if let Some(key) = key.strip_prefix("text_encoders.qwen3_4b.transformer.") {
        return Some((key, text_encoder));
    }
    if let Some(key) = key.strip_prefix("text_encoder.") {
        return Some((key, text_encoder));
    }
    if let Some(key) = key.strip_prefix("first_stage_model.") {
        return Some((key, vae));
    }
    key.strip_prefix("vae.").map(|key| (key, vae))
}

pub(crate) fn split_combined_checkpoint(path: &Path) -> Result<ComponentWeights> {
    let mut source = Weights::from_file_with_fp8(path)?;
    let keys: Vec<String> = source.keys().map(str::to_owned).collect();
    let mut transformer = HashMap::new();
    let mut text_encoder = HashMap::new();
    let mut vae = HashMap::new();
    for source_key in keys {
        let Some((target, map)) =
            component_for_key(&source_key, &mut transformer, &mut text_encoder, &mut vae)
        else {
            continue;
        };
        let value = source
            .remove(&source_key)
            .ok_or_else(|| Error::MissingTensor(source_key.clone()))?;
        map.insert(target.to_owned(), value);
    }
    for (name, map) in [
        ("transformer", &transformer),
        ("text encoder", &text_encoder),
        ("VAE", &vae),
    ] {
        if map.is_empty() {
            return Err(Error::Msg(format!(
                "z-image combined checkpoint is missing the {name} component"
            )));
        }
    }
    Ok(ComponentWeights {
        transformer: Weights::from_map(transformer),
        text_encoder: Weights::from_map(text_encoder),
        vae: Weights::from_map(vae),
    })
}

pub(crate) fn normalize_fp8(mut source: Weights, what: &str) -> Result<Weights> {
    let keys: Vec<String> = source.keys().map(str::to_owned).collect();
    let mut out = HashMap::new();
    for key in keys {
        if key == "scaled_fp8"
            || key.ends_with(".scale_weight")
            || key.ends_with(".weight_scale")
            || key.ends_with(".scale_input")
            || key.ends_with(".input_scale")
        {
            continue;
        }
        let value = source
            .remove(&key)
            .ok_or_else(|| Error::MissingTensor(key.clone()))?;
        let base = key.strip_suffix(".weight").unwrap_or(&key);
        let scale = source
            .get(&format!("{base}.scale_weight"))
            .or_else(|| source.get(&format!("{base}.weight_scale")));
        let value = match scale {
            Some(scale) => {
                if scale.size() != 1 {
                    return Err(Error::Msg(format!(
                        "{what}: scale companion for `{key}` must contain one scalar"
                    )));
                }
                let dense = multiply(
                    &value.as_dtype(Dtype::Float32)?,
                    &scale.as_dtype(Dtype::Float32)?,
                )?
                .as_dtype(Dtype::Bfloat16)?;
                dense.eval()?;
                dense
            }
            None => value,
        };
        out.insert(key, value);
    }
    Ok(Weights::from_map(out))
}

pub(crate) fn remap_dit(mut source: Weights) -> Result<Weights> {
    let keys: Vec<String> = source.keys().map(str::to_owned).collect();
    let mut out = HashMap::new();
    for key in keys {
        let value = source
            .remove(&key)
            .ok_or_else(|| Error::MissingTensor(key.clone()))?;
        if let Some(prefix) = key.strip_suffix(".attention.qkv.weight") {
            if value.shape() != [3 * DIT_DIM, DIT_DIM] {
                return Err(Error::Msg(format!(
                    "z-image ComfyUI remap: fused `{key}` has shape {:?}, expected [{}, {DIT_DIM}]",
                    value.shape(),
                    3 * DIT_DIM
                )));
            }
            let parts = mlx_rs::ops::split(&value, 3, 0)?;
            for (leaf, part) in ["to_q", "to_k", "to_v"].into_iter().zip(parts) {
                out.insert(format!("{prefix}.attention.{leaf}.weight"), part);
            }
            continue;
        }
        if key.ends_with(".attention.qkv.bias") {
            return Err(Error::Msg(format!(
                "z-image ComfyUI remap: unexpected fused attention bias `{key}`"
            )));
        }
        let target = if let Some(prefix) = key.strip_suffix(".attention.q_norm.weight") {
            format!("{prefix}.attention.norm_q.weight")
        } else if let Some(prefix) = key.strip_suffix(".attention.k_norm.weight") {
            format!("{prefix}.attention.norm_k.weight")
        } else if let Some((prefix, tail)) = key.split_once(".attention.out.") {
            format!("{prefix}.attention.to_out.0.{tail}")
        } else if let Some(tail) = key.strip_prefix("x_embedder.") {
            format!("all_x_embedder.2-1.{tail}")
        } else if let Some(tail) = key.strip_prefix("final_layer.") {
            format!("all_final_layer.2-1.{tail}")
        } else {
            key
        };
        out.insert(target, value);
    }
    Ok(Weights::from_map(out))
}

fn vae_key(key: &str) -> Result<String> {
    for stem in ["encoder", "decoder"] {
        let Some(rest) = key.strip_prefix(&format!("{stem}.")) else {
            continue;
        };
        let level_word = if stem == "encoder" { "down" } else { "up" };
        if let Some(after) = rest.strip_prefix(&format!("{level_word}.")) {
            let (index, tail) = after
                .split_once('.')
                .ok_or_else(|| Error::Msg(format!("malformed Z-Image VAE key `{key}`")))?;
            let index: usize = index
                .parse()
                .map_err(|_| Error::Msg(format!("bad Z-Image VAE level in `{key}`")))?;
            if index >= VAE_LEVELS {
                return Err(Error::Msg(format!(
                    "Z-Image VAE level {index} is outside 0..{VAE_LEVELS}"
                )));
            }
            let (blocks, samplers, target_index) = if stem == "encoder" {
                ("down_blocks", "downsamplers", index)
            } else {
                ("up_blocks", "upsamplers", VAE_LEVELS - 1 - index)
            };
            if let Some(rest) = tail.strip_prefix("block.") {
                let (block, leaf) = rest
                    .split_once('.')
                    .ok_or_else(|| Error::Msg(format!("malformed Z-Image VAE block `{key}`")))?;
                return Ok(format!(
                    "{stem}.{blocks}.{target_index}.resnets.{block}.{}",
                    leaf.replacen("nin_shortcut.", "conv_shortcut.", 1)
                ));
            }
            if let Some(rest) = tail
                .strip_prefix("downsample.")
                .or_else(|| tail.strip_prefix("upsample."))
            {
                return Ok(format!(
                    "{stem}.{blocks}.{target_index}.{samplers}.0.conv.{}",
                    rest.strip_prefix("conv.").unwrap_or(rest)
                ));
            }
            return Err(Error::Msg(format!(
                "unrecognized Z-Image VAE level key `{key}`"
            )));
        }
        if let Some(after) = rest.strip_prefix("mid.") {
            let target = if let Some(tail) = after.strip_prefix("block_1.") {
                format!("resnets.0.{tail}")
            } else if let Some(tail) = after.strip_prefix("block_2.") {
                format!("resnets.1.{tail}")
            } else if let Some(tail) = after.strip_prefix("attn_1.") {
                let tail = [
                    ("norm.", "group_norm."),
                    ("q.", "to_q."),
                    ("k.", "to_k."),
                    ("v.", "to_v."),
                    ("proj_out.", "to_out.0."),
                ]
                .into_iter()
                .find_map(|(from, to)| tail.strip_prefix(from).map(|tail| format!("{to}{tail}")))
                .unwrap_or_else(|| tail.to_owned());
                format!("attentions.0.{tail}")
            } else {
                after.to_owned()
            };
            return Ok(format!("{stem}.mid_block.{target}"));
        }
        if let Some(tail) = rest.strip_prefix("norm_out.") {
            return Ok(format!("{stem}.conv_norm_out.{tail}"));
        }
        return Ok(key.to_owned());
    }
    Ok(key.to_owned())
}

pub(crate) fn remap_vae(mut source: Weights) -> Result<Weights> {
    let keys: Vec<String> = source.keys().map(str::to_owned).collect();
    let mut out = HashMap::new();
    for key in keys {
        let value = source
            .remove(&key)
            .ok_or_else(|| Error::MissingTensor(key.clone()))?;
        let target = vae_key(&key)?;
        let value = if target.contains(".attentions.0.")
            && (target.ends_with(".to_q.weight")
                || target.ends_with(".to_k.weight")
                || target.ends_with(".to_v.weight")
                || target.ends_with(".to_out.0.weight"))
            && value.shape().len() == 4
        {
            let [out_dim, in_dim, 1, 1] = value.shape() else {
                return Err(Error::Msg(format!(
                    "z-image VAE attention `{key}` must be [C,C,1,1], got {:?}",
                    value.shape()
                )));
            };
            value.reshape(&[*out_dim, *in_dim])?
        } else {
            value
        };
        out.insert(target, value);
    }
    Ok(Weights::from_map(out))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn component_prefixes_cover_diffusers_and_ldm_forms() {
        let mut transformer = HashMap::new();
        let mut text_encoder = HashMap::new();
        let mut vae = HashMap::new();
        assert_eq!(
            component_for_key(
                "text_encoders.qwen3_4b.transformer.model.layers.0.weight",
                &mut transformer,
                &mut text_encoder,
                &mut vae
            )
            .map(|(key, _)| key),
            Some("model.layers.0.weight")
        );
        assert_eq!(
            component_for_key(
                "first_stage_model.decoder.conv_in.weight",
                &mut transformer,
                &mut text_encoder,
                &mut vae
            )
            .map(|(key, _)| key),
            Some("decoder.conv_in.weight")
        );
    }

    #[test]
    fn vae_ldm_keys_map_to_diffusers_schema() {
        assert_eq!(
            vae_key("encoder.down.2.block.1.nin_shortcut.weight").unwrap(),
            "encoder.down_blocks.2.resnets.1.conv_shortcut.weight"
        );
        assert_eq!(
            vae_key("decoder.up.3.upsample.conv.weight").unwrap(),
            "decoder.up_blocks.0.upsamplers.0.conv.weight"
        );
        assert_eq!(
            vae_key("decoder.mid.attn_1.proj_out.weight").unwrap(),
            "decoder.mid_block.attentions.0.to_out.0.weight"
        );
    }
}
