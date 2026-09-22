//! PLACEHOLDER — replaced by the real DiT port. Public API is final.

use candle_core::{DType, Device, Tensor};
use candle_gen::candle_nn::VarBuilder;
use candle_gen::{CandleError as Error, Result};

use crate::config::TransformerConfig;

/// One run of the joint sequence.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Segment {
    /// `len` prompt tokens (strictly causal among themselves).
    Text { len: usize },
    /// One image block of `height × width` latent tokens (row-major), internally bidirectional.
    Image { height: usize, width: usize },
}

/// The joint text/image token layout the transformer attends over.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct JointLayout {
    pub segments: Vec<Segment>,
}

impl JointLayout {
    pub fn text_to_image(text_len: usize, height: usize, width: usize) -> Self {
        Self {
            segments: vec![
                Segment::Text { len: text_len },
                Segment::Image { height, width },
            ],
        }
    }

    fn segment_len(segment: &Segment) -> usize {
        match segment {
            Segment::Text { len } => *len,
            Segment::Image { height, width } => height * width,
        }
    }

    pub fn total_len(&self) -> usize {
        self.segments.iter().map(Self::segment_len).sum()
    }

    pub fn target_tokens(&self) -> usize {
        self.segments.last().map_or(0, Self::segment_len)
    }

    pub fn prefix_len(&self) -> usize {
        self.total_len() - self.target_tokens()
    }

    pub fn prefix_segments(&self) -> Vec<(usize, usize, bool)> {
        Vec::new()
    }

    pub fn position_ids(&self) -> Vec<[i32; 3]> {
        Vec::new()
    }

    pub fn target_mask(&self) -> Vec<bool> {
        Vec::new()
    }
}

pub struct QwenImage21Transformer {
    cfg: TransformerConfig,
    device: Device,
}

impl QwenImage21Transformer {
    pub fn new(cfg: &TransformerConfig, vb: VarBuilder) -> Result<Self> {
        Ok(Self {
            cfg: cfg.clone(),
            device: vb.device().clone(),
        })
    }

    pub fn config(&self) -> &TransformerConfig {
        &self.cfg
    }

    pub fn device(&self) -> &Device {
        &self.device
    }

    pub fn compute_dtype(&self) -> DType {
        DType::F32
    }

    pub fn rope(&self, _layout: &JointLayout) -> Result<(Tensor, Tensor)> {
        Err(Error::Msg("unimplemented".into()))
    }

    pub fn forward(
        &self,
        _latents: &Tensor,
        _text: &Tensor,
        _timestep: f32,
        _height: usize,
        _width: usize,
    ) -> Result<Tensor> {
        Err(Error::Msg("unimplemented".into()))
    }

    pub fn forward_joint(
        &self,
        _text: &Tensor,
        _images: &[&Tensor],
        _timestep: f32,
        _layout: &JointLayout,
    ) -> Result<Tensor> {
        Err(Error::Msg("unimplemented".into()))
    }

    pub fn forward_joint_traced(
        &self,
        _text: &Tensor,
        _images: &[&Tensor],
        _timestep: f32,
        _layout: &JointLayout,
    ) -> Result<(Tensor, Vec<(String, Tensor)>)> {
        Err(Error::Msg("unimplemented".into()))
    }
}
