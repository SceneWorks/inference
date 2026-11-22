//! Logical component views over one pinned fused SDXL LDM/A1111 checkpoint.

use std::{collections::BTreeMap, path::Path};

use candle_core::{DType, Device};
use candle_gen::loader::{remap_file_var_builder, SourceView};
use candle_gen::{CandleError, Result};
use candle_nn::VarBuilder;
use gen_core_family::sdxl_ldm::{remap_sdxl_ldm_key, SdxlComponent, TensorRemap};

#[derive(Clone)]
pub struct LdmComponents {
    weights: VarBuilder<'static>,
    pub(crate) kohya_targets: BTreeMap<String, String>,
}

fn component_prefix(component: SdxlComponent) -> &'static str {
    match component {
        SdxlComponent::Unet => "unet",
        SdxlComponent::ClipL => "clip_l",
        SdxlComponent::ClipBigG => "clip_bigg",
        SdxlComponent::Vae => "vae",
    }
}

impl LdmComponents {
    pub fn builder(
        &self,
        component: SdxlComponent,
        dtype: DType,
        device: &Device,
    ) -> VarBuilder<'static> {
        self.weights
            .pp(component_prefix(component))
            .set_dtype(dtype)
            .set_device(device.clone())
    }
}

pub fn split_ldm_checkpoint(path: &Path, device: &Device) -> Result<LdmComponents> {
    let mut found = Vec::new();
    let mut kohya_targets = BTreeMap::new();
    let weights = remap_file_var_builder(path, DType::F32, device, |key, shape| {
        let Some(remap) = remap_sdxl_ldm_key(key) else {
            return Ok(Vec::new());
        };
        let component = match &remap {
            TensorRemap::Rename(component, _)
            | TensorRemap::Transpose(component, _)
            | TensorRemap::Squeeze(component, _)
            | TensorRemap::SplitQkv(component, _) => *component,
        };
        if !found.contains(&component) {
            found.push(component);
        }
        let name = |key: &str| format!("{}.{key}", component_prefix(component));
        Ok(match remap {
            TensorRemap::Rename(_, target) => {
                if component == SdxlComponent::Unet {
                    crate::adapters::add_kohya_target(&mut kohya_targets, &target, shape.len());
                }
                vec![SourceView::Whole(name(&target))]
            }
            TensorRemap::Transpose(_, target) => {
                if shape.len() != 2 {
                    return Err(CandleError::Msg(format!(
                        "sdxl LDM {key}: text projection requires a matrix, got {shape:?}"
                    )));
                }
                vec![SourceView::Transposed(name(&target))]
            }
            TensorRemap::Squeeze(_, target) => {
                let shape: Vec<_> = shape.iter().copied().filter(|&dim| dim != 1).collect();
                let name = name(&target);
                vec![match shape.as_slice() {
                    [rows, cols] => SourceView::Matrix {
                        name,
                        rows: *rows,
                        cols: *cols,
                    },
                    _ => SourceView::Reshape { name, shape },
                }]
            }
            TensorRemap::SplitQkv(_, targets) => {
                let rows = *shape
                    .first()
                    .ok_or_else(|| CandleError::Msg(format!("sdxl LDM {key}: scalar fused QKV")))?;
                if rows % 3 != 0 {
                    return Err(CandleError::Msg(format!(
                        "sdxl LDM {key}: fused QKV row count {rows} is not divisible by 3"
                    )));
                }
                let rows = rows / 3;
                targets
                    .into_iter()
                    .enumerate()
                    .map(|(part, target)| SourceView::Rows {
                        name: name(&target),
                        first: part * rows,
                        rows,
                    })
                    .collect()
            }
        })
    })?;
    for (component, label) in [
        (SdxlComponent::Unet, "UNet"),
        (SdxlComponent::ClipL, "CLIP-L"),
        (SdxlComponent::ClipBigG, "OpenCLIP-bigG"),
        (SdxlComponent::Vae, "VAE"),
    ] {
        if !found.contains(&component) {
            return Err(CandleError::Msg(format!(
                "sdxl LDM checkpoint is missing the {label} component"
            )));
        }
    }
    Ok(LdmComponents {
        weights,
        kohya_targets,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use candle_core::{safetensors, Tensor};
    use std::collections::HashMap;

    #[test]
    fn fused_component_views_keep_scales_and_matrix_layouts() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("checkpoint.safetensors");
        let cpu = Device::Cpu;
        let ones = Tensor::from_raw_buffer(&[0x38; 4], DType::F8E4M3, &[2, 2], &cpu).unwrap();
        let text_qkv = Tensor::from_raw_buffer(
            &[0x38, 0x38, 0x40, 0x40, 0x44, 0x44],
            DType::F8E4M3,
            &[3, 2],
            &cpu,
        )
        .unwrap();
        let text_projection = Tensor::from_raw_buffer(
            &[0x38, 0x40, 0x44, 0x48, 0x4a, 0x4c],
            DType::F8E4M3,
            &[2, 3],
            &cpu,
        )
        .unwrap();
        let scalar = Tensor::new(2f32, &cpu).unwrap();
        safetensors::save(
            &HashMap::from([
                ("model.diffusion_model.input_blocks.0.0.weight", ones),
                (
                    "model.diffusion_model.input_blocks.0.0.weight_scale",
                    scalar.clone(),
                ),
                (
                    "conditioner.embedders.0.transformer.text_model.final_layer_norm.weight",
                    scalar,
                ),
                (
                    "conditioner.embedders.1.model.transformer.resblocks.0.attn.in_proj_weight",
                    text_qkv,
                ),
                (
                    "conditioner.embedders.1.model.text_projection",
                    text_projection,
                ),
                (
                    "first_stage_model.decoder.mid.attn_1.q.weight",
                    Tensor::from_raw_buffer(&[0x38; 4], DType::F8E4M3, &[2, 2, 1, 1], &cpu)
                        .unwrap(),
                ),
            ]),
            &path,
        )
        .unwrap();
        let scope = candle_gen::gen_core::CheckpointReceiptScope::default();
        let components = scope.with(|| split_ldm_checkpoint(&path, &cpu)).unwrap();
        assert_eq!(components.kohya_targets.get("conv_in").unwrap(), "conv_in");
        let unet = components.builder(SdxlComponent::Unet, DType::F32, &cpu);
        assert_eq!(
            unet.get((2, 2), "conv_in.weight")
                .unwrap()
                .to_vec2::<f32>()
                .unwrap(),
            [[2., 2.], [2., 2.]]
        );
        let bigg = components.builder(SdxlComponent::ClipBigG, DType::F32, &cpu);
        let k = bigg.pp("text_model.encoder.layers.0.self_attn.k_proj");
        assert_eq!(
            candle_nn::linear_no_bias(2, 1, k)
                .unwrap()
                .try_weight()
                .unwrap()
                .to_vec2::<f32>()
                .unwrap(),
            [[2., 2.]]
        );
        assert_eq!(
            bigg.get((3, 2), "text_projection.weight")
                .unwrap()
                .to_vec2::<f32>()
                .unwrap(),
            [[1., 4.], [2., 5.], [3., 6.]]
        );
        let vae = components.builder(SdxlComponent::Vae, DType::F32, &cpu);
        assert_eq!(
            vae.get((2, 2), "decoder.mid_block.attentions.0.to_q.weight")
                .unwrap()
                .to_vec2::<f32>()
                .unwrap(),
            [[1., 1.], [1., 1.]]
        );
        assert_eq!(scope.try_snapshots().unwrap().len(), 1);
    }
}
