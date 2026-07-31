//! Prompt-to-image Mage-Flow RL pipeline.

use std::path::Path;

use candle_core::{DType, Device, IndexOp, Result, Tensor};
use candle_transformers::models::z_image::sampling::postprocess_image;

use crate::config::LATENT_CHANNELS;
use crate::rope::{ImgShape, PackLayout};
use crate::scheduler;
use crate::{MageComponentDirs, MageConfig, MageTextEncoder, MageTransformer, MageVae};
use candle_gen::gen_core::Quant;

pub struct MagePipeline {
    text: MageTextEncoder,
    transformer: MageTransformer,
    vae: MageVae,
    device: Device,
}

impl MagePipeline {
    pub fn load(root: &Path, device: &Device) -> Result<Self> {
        Self::load_with_quant(root, None, device)
    }

    pub fn load_with_quant(root: &Path, quant: Option<Quant>, device: &Device) -> Result<Self> {
        Self::load_components(&MageComponentDirs::flat(root), quant, device)
    }

    pub(crate) fn load_components(
        dirs: &MageComponentDirs,
        quant: Option<Quant>,
        device: &Device,
    ) -> Result<Self> {
        let cfg_text = std::fs::read_to_string(dirs.transformer.join("config.json"))?;
        let cfg = MageConfig::from_json(&cfg_text)?;
        Ok(Self {
            text: MageTextEncoder::load_component_with_quant(
                &dirs.text_encoder,
                false,
                quant,
                device,
            )?,
            transformer: MageTransformer::load_with_quant(&dirs.transformer, &cfg, quant, device)?,
            vae: MageVae::load(&dirs.vae, device)?,
            device: device.clone(),
        })
    }

    #[allow(clippy::too_many_arguments)]
    pub fn generate(
        &self,
        prompt: &str,
        negative_prompt: &str,
        width: u32,
        height: u32,
        steps: usize,
        guidance: f32,
        seed: u64,
        on_progress: &mut dyn FnMut(candle_gen::gen_core::Progress),
    ) -> Result<candle_gen::gen_core::Image> {
        let gh = height as usize / 16;
        let gw = width as usize / 16;
        let positive = self.text.encode(prompt)?;
        let use_cfg = guidance > 1.0;
        let negative = if use_cfg {
            Some(self.text.encode(negative_prompt)?)
        } else {
            None
        };
        let base = PackLayout::generation(vec![ImgShape::latent(gh, gw)], vec![positive.dim(1)?])?;
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
        let mut tokens = noise
            .permute((0, 2, 3, 1))?
            .reshape((1, gh * gw, LATENT_CHANNELS))?;
        let ladder = scheduler::sigmas(steps)?;
        for (i, pair) in ladder.windows(2).enumerate() {
            let (input, sigma) = if use_cfg {
                (
                    Tensor::cat(&[tokens.clone(), tokens.clone()], 1)?,
                    Tensor::new(&[pair[0], pair[0]], &self.device)?,
                )
            } else {
                (tokens.clone(), Tensor::new(&[pair[0]], &self.device)?)
            };
            let output = self.transformer.forward(&input, &text, &sigma, &layout)?;
            let velocity = if use_cfg {
                let cond = output.narrow(1, 0, gh * gw)?;
                let unc = output.narrow(1, gh * gw, gh * gw)?;
                (&unc + ((&cond - &unc)? * guidance as f64)?)?
            } else {
                output
            };
            tokens = scheduler::euler_step(&tokens, &velocity, pair[0], pair[1])?
                .to_dtype(DType::BF16)?;
            on_progress(candle_gen::gen_core::Progress::Step {
                current: (i + 1) as u32,
                total: steps as u32,
            });
        }
        on_progress(candle_gen::gen_core::Progress::Decoding);
        let latent = tokens
            .reshape((1, gh, gw, LATENT_CHANNELS))?
            .permute((0, 3, 1, 2))?;
        let decoded = self.vae.decode(&latent)?.to_dtype(DType::F32)?;
        let image = postprocess_image(&decoded)?.i(0)?.to_device(&Device::Cpu)?;
        let (c, h, w) = image.dims3()?;
        if c != 3 {
            candle_core::bail!("mage: VAE returned {c} channels");
        }
        Ok(candle_gen::gen_core::Image {
            width: w as u32,
            height: h as u32,
            pixels: image.permute((1, 2, 0))?.flatten_all()?.to_vec1::<u8>()?,
        })
    }
}
