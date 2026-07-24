//! In-memory fused SDXL LDM/A1111 checkpoint component split and remap.

use std::collections::HashMap;
use std::path::Path;

use candle_core::{safetensors, Device, Tensor};
use candle_gen::gen_core::sdxl_ldm::{remap_sdxl_ldm_key, SdxlComponent, TensorRemap};

use candle_gen::{CandleError, Result};

#[derive(Clone)]
pub struct LdmComponents {
    pub unet: HashMap<String, Tensor>,
    pub clip_l: HashMap<String, Tensor>,
    pub clip_bigg: HashMap<String, Tensor>,
    pub vae: HashMap<String, Tensor>,
}

fn map_for(
    component: SdxlComponent,
    maps: &mut [HashMap<String, Tensor>; 4],
) -> &mut HashMap<String, Tensor> {
    &mut maps[match component {
        SdxlComponent::Unet => 0,
        SdxlComponent::ClipL => 1,
        SdxlComponent::ClipBigG => 2,
        SdxlComponent::Vae => 3,
    }]
}

fn squeeze_linear(mut value: Tensor) -> Result<Tensor> {
    for dim in (0..value.rank()).rev() {
        if value.dim(dim)? == 1 {
            value = value.squeeze(dim)?;
        }
    }
    Ok(value)
}

pub fn split_ldm_checkpoint(path: &Path) -> Result<LdmComponents> {
    let source = safetensors::load(path, &Device::Cpu)?;
    let mut maps: [HashMap<String, Tensor>; 4] = Default::default();
    for (source_key, value) in source {
        let Some(remap) = remap_sdxl_ldm_key(&source_key) else {
            continue;
        };
        match remap {
            TensorRemap::Rename(component, target) => {
                map_for(component, &mut maps).insert(target, value);
            }
            TensorRemap::Transpose(component, target) => {
                map_for(component, &mut maps).insert(target, value.transpose(0, 1)?);
            }
            TensorRemap::Squeeze(component, target) => {
                map_for(component, &mut maps).insert(target, squeeze_linear(value)?);
            }
            TensorRemap::SplitQkv(component, targets) => {
                let rows = value.dim(0)?;
                if rows % 3 != 0 {
                    return Err(CandleError::Msg(format!(
                        "sdxl LDM {source_key}: fused QKV row count {rows} is not divisible by 3"
                    )));
                }
                let part = rows / 3;
                for (index, target) in targets.into_iter().enumerate() {
                    map_for(component, &mut maps)
                        .insert(target, value.narrow(0, index * part, part)?);
                }
            }
        }
    }
    for (name, map) in ["UNet", "CLIP-L", "OpenCLIP-bigG", "VAE"]
        .into_iter()
        .zip(&maps)
    {
        if map.is_empty() {
            return Err(CandleError::Msg(format!(
                "sdxl LDM checkpoint is missing the {name} component"
            )));
        }
    }
    let [unet, clip_l, clip_bigg, vae] = maps;
    Ok(LdmComponents {
        unet,
        clip_l,
        clip_bigg,
        vae,
    })
}
