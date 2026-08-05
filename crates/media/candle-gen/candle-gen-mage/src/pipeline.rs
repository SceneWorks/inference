//! Prompt-to-image Mage-Flow RL pipeline.

use std::path::Path;

use candle_core::{DType, Device, IndexOp, Result, Tensor};
use candle_transformers::models::z_image::sampling::postprocess_image;

use crate::config::LATENT_CHANNELS;
use crate::rope::{ImgShape, PackLayout};
use crate::scheduler;
use crate::{MageComponentDirs, MageConfig, MageTextEncoder, MageTransformer, MageVae};
use candle_gen::gen_core::{GenerationMemory, Quant};

pub(crate) struct MageEncoded {
    positive: Tensor,
    negative: Option<Tensor>,
}

pub(crate) struct MageHeavy {
    pub(crate) transformer: MageTransformer,
    pub(crate) vae: MageVae,
}

pub struct MagePipeline {
    text: MageTextEncoder,
    transformer: MageTransformer,
    vae: MageVae,
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
        })
    }

    pub(crate) fn load_text(
        dirs: &MageComponentDirs,
        quant: Option<Quant>,
        device: &Device,
    ) -> Result<MageTextEncoder> {
        MageTextEncoder::load_component_with_quant(&dirs.text_encoder, false, quant, device)
    }

    pub(crate) fn load_heavy(
        dirs: &MageComponentDirs,
        quant: Option<Quant>,
        device: &Device,
        stream_transformer_blocks: bool,
        cancel: &candle_gen::gen_core::CancelFlag,
    ) -> Result<MageHeavy> {
        let cfg_text = std::fs::read_to_string(dirs.transformer.join("config.json"))?;
        let cfg = MageConfig::from_json(&cfg_text)?;
        let transformer = if stream_transformer_blocks {
            MageTransformer::load_block_streamed(&dirs.transformer, &cfg, quant, device, cancel)?
        } else {
            MageTransformer::load_with_quant(&dirs.transformer, &cfg, quant, device)?
        };
        Ok(MageHeavy {
            transformer,
            vae: MageVae::load(&dirs.vae, device)?,
        })
    }

    pub(crate) fn encode_prompt(
        text: &MageTextEncoder,
        prompt: &str,
        negative_prompt: &str,
        guidance: f32,
    ) -> Result<MageEncoded> {
        Ok(MageEncoded {
            positive: text.encode(prompt)?,
            negative: (guidance > 1.0)
                .then(|| text.encode(negative_prompt))
                .transpose()?,
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
        self.generate_with_memory(
            prompt,
            negative_prompt,
            width,
            height,
            steps,
            guidance,
            seed,
            None,
            &candle_gen::gen_core::CancelFlag::default(),
            on_progress,
        )
    }

    #[allow(clippy::too_many_arguments)]
    pub(crate) fn generate_with_memory(
        &self,
        prompt: &str,
        negative_prompt: &str,
        width: u32,
        height: u32,
        steps: usize,
        guidance: f32,
        seed: u64,
        memory: Option<GenerationMemory>,
        cancel: &candle_gen::gen_core::CancelFlag,
        on_progress: &mut dyn FnMut(candle_gen::gen_core::Progress),
    ) -> Result<candle_gen::gen_core::Image> {
        let encoded = Self::encode_prompt(&self.text, prompt, negative_prompt, guidance)?;
        Self::sample(
            &self.transformer,
            &self.vae,
            encoded,
            width,
            height,
            steps,
            guidance,
            seed,
            memory,
            cancel,
            on_progress,
        )
    }

    #[allow(clippy::too_many_arguments)]
    pub(crate) fn sample(
        transformer: &MageTransformer,
        vae: &MageVae,
        encoded: MageEncoded,
        width: u32,
        height: u32,
        steps: usize,
        guidance: f32,
        seed: u64,
        memory: Option<GenerationMemory>,
        cancel: &candle_gen::gen_core::CancelFlag,
        on_progress: &mut dyn FnMut(candle_gen::gen_core::Progress),
    ) -> Result<candle_gen::gen_core::Image> {
        let gh = height as usize / 16;
        let gw = width as usize / 16;
        let MageEncoded { positive, negative } = encoded;
        let device = positive.device().clone();
        let use_cfg = negative.is_some();
        let base = PackLayout::generation(vec![ImgShape::latent(gh, gw)], vec![positive.dim(1)?])?;
        let (layout, text) = if let Some(negative) = &negative {
            (
                base.fused_cfg(&[negative.dim(1)?])?,
                Tensor::cat(&[positive.clone(), negative.clone()], 1)?,
            )
        } else {
            (base, positive)
        };

        let noise =
            crate::latent::watermarked_noise(LATENT_CHANNELS, gh, gw, seed, DType::BF16, &device)?;
        let mut tokens = noise
            .permute((0, 2, 3, 1))?
            .reshape((1, gh * gw, LATENT_CHANNELS))?;
        let ladder = scheduler::sigmas(steps)?;
        let attention_budget = memory
            .filter(|memory| memory.chunk_attention)
            .and_then(|memory| memory.attention_chunk_size)
            .unwrap_or(candle_gen::ATTN_SCORES_BUDGET as u32)
            as usize;
        let transformer_window = memory
            .filter(|memory| memory.stream_transformer_blocks)
            .and_then(|memory| memory.transformer_window_size)
            .map(|window| window as usize)
            .unwrap_or(crate::memory_strategy::DEFAULT_TRANSFORMER_WINDOW);
        for (i, pair) in ladder.windows(2).enumerate() {
            if cancel.is_cancelled() {
                candle_core::bail!("mage canceled");
            }
            let (input, sigma) = if use_cfg {
                (
                    Tensor::cat(&[tokens.clone(), tokens.clone()], 1)?,
                    Tensor::new(&[pair[0], pair[0]], &device)?,
                )
            } else {
                (tokens.clone(), Tensor::new(&[pair[0]], &device)?)
            };
            let output = transformer.forward_with_memory(
                &input,
                &text,
                &sigma,
                &layout,
                attention_budget,
                transformer_window,
                cancel,
            )?;
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
        crate::begin_decode(cancel, "mage", on_progress)?;
        let latent = tokens
            .reshape((1, gh, gw, LATENT_CHANNELS))?
            .permute((0, 3, 1, 2))?;
        let decoded = if memory.is_some_and(|memory| memory.tile_vae_decode) {
            let memory = memory.expect("guarded above");
            vae.decode_bounded(
                &latent,
                memory
                    .decode_tile_edge
                    .unwrap_or(crate::memory_strategy::DECODE_TILE_EDGE),
                memory
                    .decode_overlap
                    .unwrap_or(crate::memory_strategy::DECODE_OVERLAP),
            )?
        } else {
            vae.decode(&latent)?
        }
        .to_dtype(DType::F32)?;
        if cancel.is_cancelled() {
            candle_core::bail!("mage canceled");
        }
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
