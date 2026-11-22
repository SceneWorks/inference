//! Community Z-Image component and fused-checkpoint loading.

use std::collections::HashMap;
use std::path::Path;

use mlx_gen::weights::Weights;
use mlx_gen::{Error, Result};
use mlx_rs::ops::multiply;
#[cfg(test)]
use mlx_rs::Array;
use mlx_rs::Dtype;

const DIT_DIM: i32 = 3840;
const VAE_LEVELS: usize = 4;

pub(crate) struct ComponentWeights {
    pub transformer: Weights,
    pub text_encoder: Weights,
    pub vae: Weights,
}

#[derive(Clone, Copy, PartialEq, Eq)]
enum Component {
    Transformer,
    TextEncoder,
    Vae,
}

fn component_name(key: &str) -> Option<(Component, &str)> {
    if let Some(key) = key.strip_prefix("model.diffusion_model.") {
        return Some((Component::Transformer, key));
    }
    if let Some(key) = key.strip_prefix("transformer.") {
        return Some((Component::Transformer, key));
    }
    if let Some(key) = key.strip_prefix("conditioner.embedders.0.transformer.") {
        return Some((Component::TextEncoder, key));
    }
    if let Some(key) = key.strip_prefix("text_encoders.qwen3_4b.transformer.") {
        return Some((Component::TextEncoder, key));
    }
    if let Some(key) = key.strip_prefix("text_encoder.") {
        return Some((Component::TextEncoder, key));
    }
    if let Some(key) = key.strip_prefix("first_stage_model.") {
        return Some((Component::Vae, key));
    }
    key.strip_prefix("vae")
        .and_then(|key| key.strip_prefix('.').map(|key| (Component::Vae, key)))
}

fn split_combined_checkpoint_selected(
    path: &Path,
    include_text: bool,
    include_heavy: bool,
) -> Result<ComponentWeights> {
    let source = Weights::from_file_with_fp8_filter(path, |key| {
        component_name(key).is_some_and(|(component, _)| match component {
            Component::TextEncoder => include_text,
            Component::Transformer | Component::Vae => include_heavy,
        })
    })?;
    let component = |which| {
        source.select_renamed(|key| {
            component_name(key).and_then(|(c, name)| (c == which).then(|| name.to_owned()))
        })
    };
    let transformer = component(Component::Transformer);
    let text_encoder = component(Component::TextEncoder);
    let vae = component(Component::Vae);
    for (enabled, name, map) in [
        (include_heavy, "transformer", &transformer),
        (include_text, "text encoder", &text_encoder),
        (include_heavy, "VAE", &vae),
    ] {
        if enabled && map.is_empty() {
            return Err(Error::Msg(format!(
                "z-image combined checkpoint is missing the {name} component"
            )));
        }
    }
    Ok(ComponentWeights {
        transformer,
        text_encoder,
        vae,
    })
}

pub(crate) fn split_combined_text(path: &Path) -> Result<Weights> {
    Ok(split_combined_checkpoint_selected(path, true, false)?.text_encoder)
}

pub(crate) fn split_combined_heavy(path: &Path) -> Result<(Weights, Weights)> {
    let components = split_combined_checkpoint_selected(path, false, true)?;
    Ok((components.transformer, components.vae))
}

pub(crate) fn normalize_fp8(mut source: Weights, what: &str) -> Result<Weights> {
    // Encoded projections and any required float fallback were normalized from source dtypes by
    // the shared reader. Older exporters also attach scalar multipliers to plain floating arrays.
    let keys: Vec<String> = source
        .keys()
        .filter(|key| key.ends_with(".scale_weight") || key.ends_with(".weight_scale"))
        .map(str::to_owned)
        .collect();
    for key in keys {
        let base = key
            .trim_end_matches(".scale_weight")
            .trim_end_matches(".weight_scale");
        let ordinary = format!("{base}.weight");
        let weight_key = if source.get(&ordinary).is_some() {
            ordinary.as_str()
        } else {
            base
        };
        let Some(value) = source.get(weight_key) else {
            continue;
        };
        if !matches!(
            value.dtype(),
            Dtype::Float16 | Dtype::Bfloat16 | Dtype::Float32 | Dtype::Float64
        ) {
            continue;
        }
        let scale = source.require(&key)?;
        if scale.size() != 1 {
            return Err(Error::Msg(format!(
                "{what}: scale companion for {weight_key} must contain one scalar"
            )));
        }
        let dense = multiply(
            &value.as_dtype(Dtype::Float32)?,
            &scale.as_dtype(Dtype::Float32)?,
        )?
        .as_dtype(Dtype::Bfloat16)?;
        dense.eval()?;
        source.insert(weight_key, dense);
        source.remove(&key);
    }
    Ok(source)
}

pub(crate) fn remap_dit(source: Weights) -> Result<Weights> {
    remap_dit_width(source, DIT_DIM)
}

fn remap_dit_width(source: Weights, dim: i32) -> Result<Weights> {
    let mut out = source.select_renamed(|key| {
        if key.contains(".attention.qkv.") {
            return None;
        }
        Some(
            if let Some(prefix) = key.strip_suffix(".attention.q_norm.weight") {
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
                key.to_owned()
            },
        )
    });
    for key in source.keys().filter(|k| k.contains(".attention.qkv.")) {
        if key.ends_with(".attention.qkv.bias") {
            return Err(Error::Msg(format!(
                "z-image ComfyUI remap: unexpected fused attention bias {key}"
            )));
        }
        let Some(prefix) = key.strip_suffix(".attention.qkv.weight") else {
            continue;
        };
        let targets =
            ["to_q", "to_k", "to_v"].map(|leaf| format!("{prefix}.attention.{leaf}.weight"));
        let value = source.require(key)?;
        let native = source.source_projection(key)?;
        let shape = native
            .as_ref()
            .map(|p| p.shape().to_vec())
            .unwrap_or_else(|| value.shape().to_vec());
        if shape != [3 * dim, dim] {
            return Err(Error::Msg(format!(
                "z-image ComfyUI remap: fused {key} has shape {shape:?}, expected [{}, {dim}]",
                3 * dim
            )));
        }
        if let Some(native) = native {
            out.insert_split_projection(key, native, &targets)?;
        } else {
            for (target, part) in targets.into_iter().zip(mlx_rs::ops::split(value, 3, 0)?) {
                out.insert(target, part);
            }
        }
    }
    Ok(out)
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

pub(crate) fn remap_vae(source: Weights) -> Result<Weights> {
    let targets: HashMap<_, _> = source
        .keys()
        .map(|key| Ok((key, vae_key(key)?)))
        .collect::<Result<_>>()?;
    let mut out = source.select_renamed(|key| targets.get(key).cloned());
    for (key, target) in targets {
        if target.contains(".attentions.0.")
            && [
                ".to_q.weight",
                ".to_k.weight",
                ".to_v.weight",
                ".to_out.0.weight",
            ]
            .iter()
            .any(|leaf| target.ends_with(leaf))
        {
            let value = source.require(key)?;
            if value.ndim() == 4 {
                let [out_dim, in_dim, 1, 1] = value.shape() else {
                    return Err(Error::Msg(format!(
                        "z-image VAE attention {key} must be [C,C,1,1], got {:?}",
                        value.shape()
                    )));
                };
                out.insert(target, value.reshape(&[*out_dim, *in_dim])?);
            }
        }
    }
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn component_prefixes_cover_diffusers_and_ldm_forms() {
        assert!(matches!(
            component_name("text_encoders.qwen3_4b.transformer.model.layers.0.weight"),
            Some((Component::TextEncoder, "model.layers.0.weight"))
        ));
        assert!(matches!(
            component_name("first_stage_model.decoder.conv_in.weight"),
            Some((Component::Vae, "decoder.conv_in.weight"))
        ));
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

    #[test]
    fn combined_fp8_qkv_retains_code_order_and_output_row_scales() {
        let mut checkpoint = Weights::empty();
        let base = "model.diffusion_model.layers.0.attention.qkv";
        let codes: Vec<u8> = (0..12).collect();
        checkpoint.insert(format!("{base}.weight"), Array::from_slice(&codes, &[6, 2]));
        checkpoint.insert(
            format!("{base}.weight_scale"),
            Array::from_slice(&[1f32, 2., 3., 4., 5., 6.], &[6]),
        );
        let desc = br#"{"format":"fp8r"}"#;
        checkpoint.insert(
            format!("{base}.comfy_quant"),
            Array::from_slice(desc, &[desc.len() as i32]),
        );
        let component =
            checkpoint.select_renamed(|key| component_name(key).map(|(_, name)| name.to_owned()));
        let mapped = remap_dit_width(normalize_fp8(component, "transformer").unwrap(), 2).unwrap();
        for (i, leaf) in ["to_q", "to_k", "to_v"].into_iter().enumerate() {
            let key = format!("layers.0.attention.{leaf}.weight");
            let Some(mlx_gen::NativeProjection::Fp8(op)) = mapped.source_projection(&key).unwrap()
            else {
                panic!("missing native {key}")
            };
            assert_eq!(op.codes().as_slice::<u8>(), &codes[i * 4..i * 4 + 4]);
            assert_eq!(
                op.scales().as_slice::<f32>(),
                &[(i * 2 + 1) as f32, (i * 2 + 2) as f32]
            );
        }
    }

    #[test]
    fn fused_checkpoint_phase_loaders_filter_before_building_arrays() {
        let dir_tmp = tempfile::tempdir().unwrap();
        let dir = dir_tmp.path().to_path_buf();
        let path = dir.join("combined.safetensors");
        let text = Array::from_slice(&[1.0f32], &[1]);
        let transformer = Array::from_slice(&[2.0f32], &[1]);
        let vae = Array::from_slice(&[3.0f32], &[1]);
        Array::save_safetensors(
            vec![
                ("text_encoder.layer.weight", &text),
                ("transformer.block.weight", &transformer),
                ("vae.decoder.weight", &vae),
            ],
            None,
            &path,
        )
        .unwrap();

        let text_only = split_combined_text(&path).unwrap();
        assert_eq!(text_only.keys().collect::<Vec<_>>(), ["layer.weight"]);
        let (transformer_only, vae_only) = split_combined_heavy(&path).unwrap();
        assert_eq!(
            transformer_only.keys().collect::<Vec<_>>(),
            ["block.weight"]
        );
        assert_eq!(vae_only.keys().collect::<Vec<_>>(), ["decoder.weight"]);
    }
}
