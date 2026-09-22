//! PLACEHOLDER — replaced by the real RGBA VAE port. Public API is final.

use candle_core::{DType, Device, Tensor};
use candle_gen::candle_nn::VarBuilder;
use candle_gen::gen_core::tiling::TilingConfig;
use candle_gen::gen_core::{CancelFlag, LatentSpace};
use candle_gen::{CandleError as Error, LatentDecoder, Result};

use crate::config::VaeConfig;

pub struct QwenImage21Vae {
    cfg: VaeConfig,
    device: Device,
}

impl QwenImage21Vae {
    pub fn new(cfg: &VaeConfig, vb: VarBuilder) -> Result<Self> {
        Ok(Self {
            cfg: cfg.clone(),
            device: vb.device().clone(),
        })
    }

    pub fn config(&self) -> &VaeConfig {
        &self.cfg
    }

    pub fn device(&self) -> &Device {
        &self.device
    }

    pub fn compute_dtype(&self) -> DType {
        DType::F32
    }

    pub fn encode_moments(&self, _image: &Tensor) -> Result<Tensor> {
        Err(Error::Msg("unimplemented".into()))
    }

    pub fn encode_moments_traced(&self, _image: &Tensor) -> Result<(Tensor, Vec<(String, Tensor)>)> {
        Err(Error::Msg("unimplemented".into()))
    }

    pub fn encode_mode(&self, _image: &Tensor) -> Result<Tensor> {
        Err(Error::Msg("unimplemented".into()))
    }

    pub fn decode_rgba(&self, _latents: &Tensor) -> Result<Tensor> {
        Err(Error::Msg("unimplemented".into()))
    }

    pub fn decode_rgba_traced(&self, _latents: &Tensor) -> Result<(Tensor, Vec<(String, Tensor)>)> {
        Err(Error::Msg("unimplemented".into()))
    }

    pub fn denormalize(&self, _latents: &Tensor) -> Result<Tensor> {
        Err(Error::Msg("unimplemented".into()))
    }

    pub fn normalize(&self, _latents: &Tensor) -> Result<Tensor> {
        Err(Error::Msg("unimplemented".into()))
    }
}

impl LatentDecoder for QwenImage21Vae {
    fn input_latent_space(&self) -> Option<&LatentSpace> {
        None
    }

    fn decode(&self, latents: &Tensor) -> Result<Tensor> {
        let rgba = self.decode_rgba(&self.denormalize(latents)?)?;
        crate::pipeline::rgba_to_rgb_over_white(&rgba)
    }

    fn decode_tiled(
        &self,
        latents: &Tensor,
        _tiling: &TilingConfig,
        cancel: Option<&CancelFlag>,
    ) -> Result<Tensor> {
        if cancel.is_some_and(CancelFlag::is_cancelled) {
            return Err(Error::Canceled);
        }
        self.decode(latents)
    }
}
