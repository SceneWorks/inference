//! Candle/CUDA Mage-Flow instruction editing.
//!
//! The transformer is unchanged from generation. Each denoise step presents one isolated image
//! window `[noisy target, clean reference 1, …]`; only the target velocity is integrated. Ordered
//! `img_shapes` give target frame 0 and references frames 1..N, making the MSRoPE frame axis
//! load-bearing. Qwen3-VL sees the same references through its vision tower and early-layer
//! deepstack injections.

use std::path::Path;

use candle_core::{DType, Device, IndexOp, Result, Tensor};
use candle_transformers::models::z_image::sampling::postprocess_image;

use crate::config::LATENT_CHANNELS;
use crate::rope::{ImgShape, PackLayout};
use crate::{scheduler, MageConfig, MageTextEncoder, MageTransformer, MageVae};

#[derive(Debug, Clone, Copy, PartialEq)]
pub enum MageEditVariant {
    Edit,
    EditBase,
    EditTurbo,
}

impl MageEditVariant {
    pub const fn id(self) -> &'static str {
        match self {
            Self::Edit => crate::config::EDIT_MODEL_ID,
            Self::EditBase => crate::config::EDIT_BASE_MODEL_ID,
            Self::EditTurbo => crate::config::EDIT_TURBO_MODEL_ID,
        }
    }

    pub const fn repo(self) -> &'static str {
        match self {
            Self::Edit => "microsoft/Mage-Flow-Edit",
            Self::EditBase => "microsoft/Mage-Flow-Edit-Base",
            Self::EditTurbo => "microsoft/Mage-Flow-Edit-Turbo",
        }
    }

    pub const fn defaults(self) -> (usize, f32) {
        match self {
            Self::Edit | Self::EditBase => (30, 5.0),
            Self::EditTurbo => (4, 1.0),
        }
    }
}

pub struct MageEdit {
    text: MageTextEncoder,
    transformer: MageTransformer,
    vae: MageVae,
    device: Device,
}

impl MageEdit {
    pub fn load(root: &Path, device: &Device) -> Result<Self> {
        let cfg_text = std::fs::read_to_string(root.join("transformer/config.json"))?;
        let cfg = MageConfig::from_json(&cfg_text)?;
        Ok(Self {
            text: MageTextEncoder::load_multimodal(root, device)?,
            transformer: MageTransformer::load(&root.join("transformer"), &cfg, device)?,
            vae: MageVae::load_full(&root.join("vae"), device)?,
            device: device.clone(),
        })
    }

    #[allow(clippy::too_many_arguments)]
    pub fn edit(
        &self,
        instruction: &str,
        negative_instruction: &str,
        references: &[candle_gen::gen_core::Image],
        width: u32,
        height: u32,
        steps: usize,
        guidance: f32,
        seed: u64,
        cancel: &candle_gen::gen_core::runtime::CancelFlag,
        on_progress: &mut dyn FnMut(candle_gen::gen_core::Progress),
    ) -> Result<candle_gen::gen_core::Image> {
        if references.is_empty() {
            candle_core::bail!("mage edit: at least one reference image is required");
        }
        let gh = height as usize / 16;
        let gw = width as usize / 16;
        let ref_images = references
            .iter()
            .map(|reference| reference_nchw(reference, width, height, &self.device))
            .collect::<Result<Vec<_>>>()?;
        let ref_batch = Tensor::cat(&ref_images.iter().collect::<Vec<_>>(), 0)?;
        // The reference seeds once, then samples the complete multi-reference posterior in one call.
        let ref_tokens = self
            .vae
            .encode_sample(&ref_batch, seed)?
            .permute((0, 2, 3, 1))?
            .reshape((1, references.len() * gh * gw, LATENT_CHANNELS))?
            .to_dtype(DType::BF16)?;
        self.edit_from_reference_tokens(
            instruction,
            negative_instruction,
            references,
            &ref_tokens,
            width,
            height,
            steps,
            guidance,
            seed,
            cancel,
            on_progress,
        )
    }

    /// Run edit parity from a caller-supplied sampled posterior. The real-weight suite derives this
    /// tensor from Torch's recorded step-zero sequence, separating RNG differences from model math.
    #[allow(clippy::too_many_arguments)]
    pub fn edit_from_reference_tokens(
        &self,
        instruction: &str,
        negative_instruction: &str,
        references: &[candle_gen::gen_core::Image],
        ref_tokens: &Tensor,
        width: u32,
        height: u32,
        steps: usize,
        guidance: f32,
        seed: u64,
        cancel: &candle_gen::gen_core::runtime::CancelFlag,
        on_progress: &mut dyn FnMut(candle_gen::gen_core::Progress),
    ) -> Result<candle_gen::gen_core::Image> {
        if references.is_empty() {
            candle_core::bail!("mage edit: at least one reference image is required");
        }
        let gh = height as usize / 16;
        let gw = width as usize / 16;
        let expected = [1, references.len() * gh * gw, LATENT_CHANNELS];
        if ref_tokens.dims() != expected {
            candle_core::bail!(
                "mage edit: reference tokens are {:?}, expected {expected:?}",
                ref_tokens.dims()
            );
        }
        let ref_tokens = ref_tokens.to_device(&self.device)?.to_dtype(DType::BF16)?;
        let positive = self.text.encode_edit(instruction, references)?;
        let use_cfg = guidance > 1.0;
        let negative = if use_cfg {
            Some(self.text.encode_edit(negative_instruction, references)?)
        } else {
            None
        };
        let shapes =
            std::iter::repeat_n(ImgShape::latent(gh, gw), references.len() + 1).collect::<Vec<_>>();
        let image_tokens = (references.len() + 1) * gh * gw;
        let base = PackLayout::new(shapes, vec![image_tokens], vec![positive.dim(1)?])?;
        let (layout, text) = if let Some(negative) = &negative {
            (
                base.fused_cfg(&[negative.dim(1)?])?,
                Tensor::cat(&[positive.clone(), negative.clone()], 1)?,
            )
        } else {
            (base, positive)
        };

        let noise = crate::latent::watermarked_noise(
            LATENT_CHANNELS,
            gh,
            gw,
            seed,
            DType::BF16,
            &self.device,
        )?;
        let mut target = noise
            .permute((0, 2, 3, 1))?
            .reshape((1, gh * gw, LATENT_CHANNELS))?;
        let ladder = scheduler::sigmas(steps)?;
        for (index, pair) in ladder.windows(2).enumerate() {
            if cancel.is_cancelled() {
                candle_core::bail!("mage edit canceled");
            }
            let sequence = Tensor::cat(&[target.clone(), ref_tokens.clone()], 1)?;
            let (input, sigma) = if use_cfg {
                (
                    Tensor::cat(&[sequence.clone(), sequence], 1)?,
                    Tensor::new(&[pair[0], pair[0]], &self.device)?,
                )
            } else {
                (sequence, Tensor::new(&[pair[0]], &self.device)?)
            };
            let output = self.transformer.forward(&input, &text, &sigma, &layout)?;
            let velocity = if use_cfg {
                let cond = output.narrow(1, 0, image_tokens)?;
                let unc = output.narrow(1, image_tokens, image_tokens)?;
                (&unc + ((&cond - &unc)? * guidance as f64)?)?
            } else {
                output
            };
            let target_velocity = velocity.narrow(1, 0, gh * gw)?;
            target = scheduler::euler_step(&target, &target_velocity, pair[0], pair[1])?
                .to_dtype(DType::BF16)?;
            on_progress(candle_gen::gen_core::Progress::Step {
                current: (index + 1) as u32,
                total: steps as u32,
            });
        }

        on_progress(candle_gen::gen_core::Progress::Decoding);
        let latent = target
            .reshape((1, gh, gw, LATENT_CHANNELS))?
            .permute((0, 3, 1, 2))?;
        let decoded = self.vae.decode(&latent)?.to_dtype(DType::F32)?;
        let image = postprocess_image(&decoded)?.i(0)?.to_device(&Device::Cpu)?;
        let (channels, out_h, out_w) = image.dims3()?;
        if channels != 3 {
            candle_core::bail!("mage edit: VAE returned {channels} channels");
        }
        Ok(candle_gen::gen_core::Image {
            width: out_w as u32,
            height: out_h as u32,
            pixels: image.permute((1, 2, 0))?.flatten_all()?.to_vec1::<u8>()?,
        })
    }
}

fn reference_nchw(
    reference: &candle_gen::gen_core::Image,
    width: u32,
    height: u32,
    device: &Device,
) -> Result<Tensor> {
    if reference.width == 0
        || reference.height == 0
        || reference.pixels.len() != reference.width as usize * reference.height as usize * 3
    {
        candle_core::bail!("mage edit: malformed RGB8 reference image");
    }
    let pixels = candle_gen::gen_core::imageops::resize_bicubic_u8(
        &reference.pixels,
        reference.height as usize,
        reference.width as usize,
        height as usize,
        width as usize,
    )
    .map_err(|e| candle_core::Error::Msg(format!("mage edit: resize reference: {e}")))?;
    Tensor::from_vec(pixels, (height as usize, width as usize, 3), &Device::Cpu)?
        .permute((2, 0, 1))?
        .unsqueeze(0)?
        .affine(1. / 127.5, -1.)?
        .to_device(device)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn variant_defaults_and_repos_are_discriminating() {
        assert_eq!(MageEditVariant::Edit.defaults(), (30, 5.0));
        assert_eq!(MageEditVariant::EditBase.defaults(), (30, 5.0));
        assert_eq!(MageEditVariant::EditTurbo.defaults(), (4, 1.0));
        assert_ne!(
            MageEditVariant::Edit.repo(),
            MageEditVariant::EditBase.repo()
        );
        assert_ne!(
            MageEditVariant::EditBase.repo(),
            MageEditVariant::EditTurbo.repo()
        );
    }

    #[test]
    fn edit_layout_assigns_distinct_reference_frames() {
        let layout = PackLayout::new(
            vec![
                ImgShape::latent(2, 2),
                ImgShape::latent(2, 2),
                ImgShape::latent(2, 2),
            ],
            vec![12],
            vec![5],
        )
        .unwrap();
        assert_eq!(layout.shapes().len(), 3);
        let table = crate::rope::RopeTable::build(&layout, DType::F32, &Device::Cpu).unwrap();
        let cos = table.cos.to_vec2::<f32>().unwrap();
        assert_ne!(
            cos[0][0], cos[4][0],
            "target and reference frame must differ"
        );
        assert_ne!(cos[4][0], cos[8][0], "reference order must affect MSRoPE");
    }
}
