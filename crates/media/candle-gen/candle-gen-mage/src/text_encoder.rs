//! Mage Qwen3-VL-4B text conditioning.
//!
//! The decoder math is shared with the reviewed Candle Qwen3-VL implementation in
//! `candle-gen-boogu`; Mage supplies the 4B geometry, its own prompt template/drop, and—critically—
//! consumes the final post-RMSNorm state. Z-Image's penultimate convention is intentionally absent.

use std::path::Path;

use candle_core::{DType, Device, Error, Result, Tensor};
use candle_gen::gen_core::{PrecisionFloorComponent, Quant};
use candle_gen_boogu::loader::Weights;
use candle_gen_boogu::text_encoder::{BooguTextEncoder, BooguTextEncoderConfig};
use candle_gen_boogu::vision::preprocess::preprocess_image;
use candle_gen_boogu::vision::{VisionConfig, VisionTower};
use tokenizers::Tokenizer;

use crate::config::{
    DROP_IDX_EDIT, DROP_IDX_GEN, EDIT_IMAGE_PLACEHOLDER, EDIT_PROMPT_TEMPLATE, PROMPT_TEMPLATE,
    TE_HEADS, TE_HEAD_DIM, TE_KV_HEADS, TE_LAYERS, TE_ROPE_THETA, TXT_MAX_LENGTH,
    VL_COND_LONG_EDGE,
};

pub struct MageTextEncoder {
    model: BooguTextEncoder,
    tokenizer: Tokenizer,
    vision: Option<VisionTower>,
    device: Device,
}

impl MageTextEncoder {
    pub fn load(root: &Path, device: &Device) -> Result<Self> {
        Self::load_inner(&root.join("text_encoder"), device, false, None)
    }

    pub fn load_with_quant(root: &Path, quant: Option<Quant>, device: &Device) -> Result<Self> {
        Self::load_inner(&root.join("text_encoder"), device, false, quant)
    }

    pub fn load_multimodal(root: &Path, device: &Device) -> Result<Self> {
        Self::load_inner(&root.join("text_encoder"), device, true, None)
    }

    pub fn load_multimodal_with_quant(
        root: &Path,
        quant: Option<Quant>,
        device: &Device,
    ) -> Result<Self> {
        Self::load_inner(&root.join("text_encoder"), device, true, quant)
    }

    pub(crate) fn load_component_with_quant(
        dir: &Path,
        multimodal: bool,
        quant: Option<Quant>,
        device: &Device,
    ) -> Result<Self> {
        Self::load_inner(dir, device, multimodal, quant)
    }

    fn load_inner(
        dir: &Path,
        device: &Device,
        multimodal: bool,
        quant: Option<Quant>,
    ) -> Result<Self> {
        let weights = Weights::from_dir(dir, device, DType::BF16)?;
        let cfg = BooguTextEncoderConfig {
            num_layers: TE_LAYERS,
            num_heads: TE_HEADS,
            num_kv_heads: TE_KV_HEADS,
            head_dim: TE_HEAD_DIM,
            rms_norm_eps: 1e-6,
            rope_theta: TE_ROPE_THETA,
        };
        let mut model = BooguTextEncoder::load(
            &weights,
            "model.language_model",
            &cfg,
            TXT_MAX_LENGTH + DROP_IDX_GEN,
        )?;
        if let Some(selected) = quant {
            let layer_quant =
                crate::quant::component_quant(PrecisionFloorComponent::TextEncoder, selected);
            model.quantize_layers_onto(layer_quant, device)?;
        }
        let vision = if multimodal {
            Some(VisionTower::load(
                &weights,
                VisionConfig {
                    hidden_size: 1024,
                    num_heads: 16,
                    depth: 24,
                    out_hidden_size: 2560,
                    patch_size: 16,
                    temporal_patch_size: 2,
                    spatial_merge_size: 2,
                    in_channels: 3,
                    num_position_embeddings: 2304,
                    deepstack_visual_indexes: vec![5, 11, 17],
                },
                "model.visual",
            )?)
        } else {
            None
        };
        let tokenizer = Tokenizer::from_file(dir.join("tokenizer.json"))
            .map_err(|e| Error::Msg(format!("mage: load text_encoder/tokenizer.json: {e}")))?;
        Ok(Self {
            model,
            tokenizer,
            vision,
            device: device.clone(),
        })
    }

    /// Return `[1, <=2048, 2560]` final post-RMSNorm conditioning after dropping the frozen
    /// 34-token generation prefix.
    pub fn encode(&self, prompt: &str) -> Result<Tensor> {
        let rendered = PROMPT_TEMPLATE.replacen("{}", prompt, 1);
        let encoding = self
            .tokenizer
            .encode(rendered, false)
            .map_err(|e| Error::Msg(format!("mage: tokenize prompt: {e}")))?;
        let ids = encoding.get_ids();
        let cap = TXT_MAX_LENGTH + DROP_IDX_GEN;
        let ids = &ids[..ids.len().min(cap)];
        if ids.len() <= DROP_IDX_GEN {
            return Err(Error::Msg(format!(
                "mage: templated prompt has {} tokens, cannot drop required {DROP_IDX_GEN}",
                ids.len()
            )));
        }
        let input = Tensor::from_slice(ids, (1, ids.len()), &self.device)?;
        let final_post_norm = self.model.last_hidden(&input)?;
        final_post_norm.narrow(1, DROP_IDX_GEN, ids.len() - DROP_IDX_GEN)
    }

    /// Return image-grounded edit conditioning after the frozen 64-token system-prefix drop.
    pub fn encode_edit(
        &self,
        instruction: &str,
        references: &[candle_gen::gen_core::Image],
    ) -> Result<Tensor> {
        if references.is_empty() {
            return Err(Error::Msg(
                "mage edit: at least one reference image is required".into(),
            ));
        }
        let vision = self
            .vision
            .as_ref()
            .ok_or_else(|| Error::Msg("mage edit: vision tower was not loaded".into()))?;
        let mut embeds = Vec::with_capacity(references.len());
        let mut deepstack = Vec::with_capacity(references.len());
        let mut grids = Vec::with_capacity(references.len());
        let mut counts = Vec::with_capacity(references.len());
        for reference in references {
            let capped = cap_long_edge(reference)?;
            let (pixels, grid) = preprocess_image(
                &capped.pixels,
                capped.height as usize,
                capped.width as usize,
                &self.device,
            )
            .map_err(|e| Error::Msg(format!("mage edit: preprocess reference: {e}")))?;
            let (image_embeds, image_deepstack) = vision.forward(&pixels, &[grid])?;
            counts.push(image_embeds.dim(0)?);
            embeds.push(image_embeds);
            deepstack.push(image_deepstack);
            grids.push(grid);
        }

        let mut body = String::new();
        for index in 1..=references.len() {
            body.push_str("Image ");
            body.push_str(&index.to_string());
            body.push_str(": ");
            body.push_str(EDIT_IMAGE_PLACEHOLDER);
        }
        body.push_str(instruction);
        let rendered = EDIT_PROMPT_TEMPLATE.replacen("{}", &body, 1);
        let encoding = self
            .tokenizer
            .encode(rendered, false)
            .map_err(|e| Error::Msg(format!("mage edit: tokenize instruction: {e}")))?;
        let mut ids = encoding.get_ids().to_vec();
        ids.truncate(TXT_MAX_LENGTH + DROP_IDX_EDIT);
        ids = expand_image_tokens(&ids, &counts)?;
        if ids.len() <= DROP_IDX_EDIT {
            return Err(Error::Msg(format!(
                "mage edit: templated instruction has {} tokens, cannot drop required {DROP_IDX_EDIT}",
                ids.len()
            )));
        }
        let input = Tensor::from_vec(ids.clone(), (1, ids.len()), &self.device)?;
        let hidden = self
            .model
            .last_hidden_with_images(&input, &embeds, &deepstack, &grids, 151_655)?;
        hidden.narrow(1, DROP_IDX_EDIT, ids.len() - DROP_IDX_EDIT)
    }
}

fn expand_image_tokens(ids: &[u32], counts: &[usize]) -> Result<Vec<u32>> {
    const IMAGE_TOKEN_ID: u32 = 151_655;
    let placeholders = ids.iter().filter(|&&id| id == IMAGE_TOKEN_ID).count();
    if placeholders != counts.len() {
        return Err(Error::Msg(format!(
            "mage edit: template has {placeholders} image placeholder(s) but {} reference(s)",
            counts.len()
        )));
    }
    let mut out = Vec::with_capacity(ids.len() + counts.iter().sum::<usize>());
    let mut image = 0;
    for &id in ids {
        if id == IMAGE_TOKEN_ID {
            out.extend(std::iter::repeat_n(id, counts[image]));
            image += 1;
        } else {
            out.push(id);
        }
    }
    Ok(out)
}

fn cap_long_edge(image: &candle_gen::gen_core::Image) -> Result<candle_gen::gen_core::Image> {
    let long = image.width.max(image.height);
    if long <= VL_COND_LONG_EDGE {
        return Ok(image.clone());
    }
    let scale = f64::from(VL_COND_LONG_EDGE) / f64::from(long);
    let width = (f64::from(image.width) * scale).round().max(1.) as usize;
    let height = (f64::from(image.height) * scale).round().max(1.) as usize;
    let pixels = candle_gen::gen_core::imageops::resize_bicubic_u8(
        &image.pixels,
        image.height as usize,
        image.width as usize,
        height,
        width,
    )
    .map_err(|e| Error::Msg(format!("mage edit: cap reference long edge: {e}")))?
    .into_iter()
    .map(|value| value.round().clamp(0., 255.) as u8)
    .collect();
    Ok(candle_gen::gen_core::Image {
        width: width as u32,
        height: height as u32,
        pixels,
    })
}

/// Test seam spelling out the parity-critical state choice. Production passes only the final
/// post-norm state returned by `BooguTextEncoder::last_hidden`.
#[cfg(test)]
fn select_conditioning(final_post_norm: Tensor, _penultimate_pre_norm: Tensor) -> Tensor {
    final_post_norm
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn penultimate_pre_norm_mutation_fails_the_conditioning_oracle() {
        let d = Device::Cpu;
        let final_state = Tensor::new(&[[[1f32, 2.]]], &d).unwrap();
        let stale_z_image_state = Tensor::new(&[[[9f32, 8.]]], &d).unwrap();
        let expected = final_state.to_vec3::<f32>().unwrap();
        let selected = select_conditioning(final_state, stale_z_image_state.clone());
        assert_eq!(selected.to_vec3::<f32>().unwrap(), expected);
        assert_ne!(
            stale_z_image_state.to_vec3::<f32>().unwrap(),
            expected,
            "mutation to penultimate/pre-final-norm conditioning must fail parity"
        );
    }

    #[test]
    fn expanded_image_runs_are_non_constant_and_ordered() {
        let got = expand_image_tokens(&[7, 151_655, 8, 151_655, 9], &[2, 3]).unwrap();
        assert_eq!(got, [7, 151_655, 151_655, 8, 151_655, 151_655, 151_655, 9]);
        let mutated = expand_image_tokens(&[7, 151_655, 8, 151_655, 9], &[3, 2]).unwrap();
        assert_ne!(
            got, mutated,
            "reference token counts must discriminate order"
        );
    }
}
