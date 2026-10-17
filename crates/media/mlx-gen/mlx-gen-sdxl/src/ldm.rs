//! In-memory fused SDXL LDM/A1111 checkpoint component split and remap.

use std::collections::HashMap;
use std::path::Path;

use mlx_gen::gen_core::sdxl_ldm::{remap_sdxl_ldm_key, SdxlComponent, TensorRemap};
use mlx_gen::weights::Weights;
use mlx_gen::{Error, Result};
use mlx_rs::ops::split;
use mlx_rs::Array;

#[derive(Clone)]
pub struct LdmComponents {
    pub unet: Weights,
    pub clip_l: Weights,
    pub clip_bigg: Weights,
    pub vae: Weights,
}

fn component_map<'a>(
    component: SdxlComponent,
    unet: &'a mut HashMap<String, Array>,
    clip_l: &'a mut HashMap<String, Array>,
    clip_bigg: &'a mut HashMap<String, Array>,
    vae: &'a mut HashMap<String, Array>,
) -> &'a mut HashMap<String, Array> {
    match component {
        SdxlComponent::Unet => unet,
        SdxlComponent::ClipL => clip_l,
        SdxlComponent::ClipBigG => clip_bigg,
        SdxlComponent::Vae => vae,
    }
}

fn squeeze_linear(value: Array) -> Result<Array> {
    let dims: Vec<i32> = value
        .shape()
        .iter()
        .copied()
        .filter(|&dim| dim != 1)
        .collect();
    if dims.len() == value.shape().len() {
        Ok(value)
    } else {
        Ok(value.reshape(&dims)?)
    }
}

pub fn split_ldm_checkpoint(path: &Path) -> Result<LdmComponents> {
    let mut source = Weights::from_file(path)?;
    let keys: Vec<String> = source.keys().map(str::to_owned).collect();
    let mut unet = HashMap::new();
    let mut clip_l = HashMap::new();
    let mut clip_bigg = HashMap::new();
    let mut vae = HashMap::new();

    for source_key in keys {
        let Some(remap) = remap_sdxl_ldm_key(&source_key) else {
            continue;
        };
        let value = source
            .remove(&source_key)
            .ok_or_else(|| Error::MissingTensor(source_key.clone()))?;
        match remap {
            TensorRemap::Rename(component, target) => {
                component_map(component, &mut unet, &mut clip_l, &mut clip_bigg, &mut vae)
                    .insert(target, value);
            }
            TensorRemap::Transpose(component, target) => {
                component_map(component, &mut unet, &mut clip_l, &mut clip_bigg, &mut vae)
                    .insert(target, value.transpose()?);
            }
            TensorRemap::Squeeze(component, target) => {
                component_map(component, &mut unet, &mut clip_l, &mut clip_bigg, &mut vae)
                    .insert(target, squeeze_linear(value)?);
            }
            TensorRemap::SplitQkv(component, targets) => {
                let parts = split(&value, 3, 0)?;
                if parts.len() != 3 {
                    return Err(Error::Msg(format!(
                        "sdxl LDM {source_key}: fused QKV did not split into three tensors"
                    )));
                }
                let map =
                    component_map(component, &mut unet, &mut clip_l, &mut clip_bigg, &mut vae);
                for (target, part) in targets.into_iter().zip(parts) {
                    map.insert(target, part);
                }
            }
        }
    }
    for (name, map) in [
        ("UNet", &unet),
        ("CLIP-L", &clip_l),
        ("OpenCLIP-bigG", &clip_bigg),
        ("VAE", &vae),
    ] {
        if map.is_empty() {
            return Err(Error::Msg(format!(
                "sdxl LDM checkpoint is missing the {name} component"
            )));
        }
    }
    Ok(LdmComponents {
        unet: Weights::from_map(unet),
        clip_l: Weights::from_map(clip_l),
        clip_bigg: Weights::from_map(clip_bigg),
        vae: Weights::from_map(vae),
    })
}

/// iOS pre-load floor (sc-26270) for constructing `components` out of the fused checkpoint at
/// `path`: the stored bytes of every tensor that remaps into one of them (plus the VAE's f32 cast
/// copies), plus the largest such tensor as read/copy scratch — `mlx_gen::ios_quantum::before_load`'s accounting, restricted to the
/// components this phase actually builds (a fused file also carries the others). Header-only; a
/// no-op (no file read) off device.
pub(crate) fn before_component_load(
    label: &str,
    path: &Path,
    components: &[SdxlComponent],
) -> Result<()> {
    if !mlx_gen::ios_quantum::enabled() {
        return Ok(());
    }
    let mut total = 0u64;
    let mut largest = 0u64;
    for header in mlx_gen::gen_core::weightsmeta::safetensors_path_tensor_headers(path)
        .map_err(|e| Error::Msg(e.to_string()))?
    {
        let component = match remap_sdxl_ldm_key(&header.name) {
            Some(
                TensorRemap::Rename(component, _)
                | TensorRemap::Transpose(component, _)
                | TensorRemap::Squeeze(component, _)
                | TensorRemap::SplitQkv(component, _),
            ) => component,
            None => continue,
        };
        if components.contains(&component) {
            total = total.saturating_add(header.data_bytes);
            largest = largest.max(header.data_bytes);
            // The VAE is upcast to f32 after it is read (`load_vae_from_weights`): its f32 copy
            // is live beside the stored tensor.
            if component == SdxlComponent::Vae {
                let cast = header
                    .shape
                    .iter()
                    .fold(4u64, |a, &d| a.saturating_mul(d as u64));
                total = total.saturating_add(cast);
                largest = largest.max(cast);
            }
        }
    }
    mlx_gen::ios_quantum::before(label, total.saturating_add(largest))
}

#[cfg(test)]
mod tests {
    use super::*;
    use mlx_rs::Dtype;

    #[test]
    #[ignore = "real fused SDXL checkpoint; set SDXL_LDM_CHECKPOINT"]
    fn real_checkpoint_builds_all_mlx_components() {
        let path = std::env::var_os("SDXL_LDM_CHECKPOINT")
            .map(std::path::PathBuf::from)
            .expect("set SDXL_LDM_CHECKPOINT");
        let components = split_ldm_checkpoint(&path).expect("split fused checkpoint");
        crate::loader::load_text_encoder_1_from_weights(components.clip_l, Dtype::Float16)
            .expect("build CLIP-L");
        crate::loader::load_text_encoder_2_from_weights(components.clip_bigg, Dtype::Float16)
            .expect("build OpenCLIP-bigG");
        crate::loader::load_unet_from_weights(components.unet, Dtype::Float16).expect("build UNet");
        crate::loader::load_vae_from_weights(components.vae).expect("build VAE");
    }
}
