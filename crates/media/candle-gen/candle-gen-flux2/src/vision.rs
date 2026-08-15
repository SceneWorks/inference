//! Native Candle Pixtral vision tower and Mistral3 multimodal projector for FLUX.2-dev.
//!
//! The implementation mirrors the proven MLX provider's `vision` module while keeping the
//! `vision_tower.*` and `multi_modal_projector.*` checkpoint namespaces byte-for-byte stable. The
//! tower remains dense f32 even when the language model is quantized: Pixtral's patch convolution,
//! RMSNorms, attention projections and SwiGLU layers are intentionally never routed through
//! [`crate::quant::QLinear`]. Inputs are NCHW f32 tensors and multi-image attention is block-diagonal
//! (one independent SDPA window per image).

use candle_gen::candle_core::{DType, Device, Error, Result as CandleResult, Tensor};
use candle_gen::candle_nn::{
    conv2d_no_bias, linear_no_bias, ops::softmax_last_dim, rms_norm, Conv2d, Conv2dConfig, Linear,
    Module, RmsNorm, VarBuilder,
};
use candle_gen::gen_core::CancelFlag;
use candle_gen::{CandleError, Result as GenResult};

const VISION_PREFIX: &str = "vision_tower";
const PROJECTOR_PREFIX: &str = "multi_modal_projector";
const PROJECTOR_TEXT_HIDDEN: usize = 5120;
const PROJECTOR_SPATIAL_MERGE: usize = 2;
const PROJECTOR_RMS_EPS: f64 = 1e-5;

/// Pixtral vision-tower dimensions from FLUX.2-dev's `text_encoder/config.json`.
///
/// The production constructor always uses [`Self::dev`]. Keeping the values together also lets the
/// weights-free tests instantiate a tiny shape-equivalent tower without weakening production keys.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct PixtralVisionConfig {
    pub hidden_size: usize,
    pub num_layers: usize,
    pub num_heads: usize,
    pub head_dim: usize,
    pub intermediate_size: usize,
    pub patch_size: usize,
    pub rope_theta: f32,
    pub rms_norm_eps: f64,
    pub num_channels: usize,
}

impl PixtralVisionConfig {
    /// FLUX.2-dev Pixtral: 24 x 1024, 16 heads x 64, FFN 4096, patch 14.
    pub const fn dev() -> Self {
        Self {
            hidden_size: 1024,
            num_layers: 24,
            num_heads: 16,
            head_dim: 64,
            intermediate_size: 4096,
            patch_size: 14,
            rope_theta: 10_000.0,
            rms_norm_eps: 1e-5,
            num_channels: 3,
        }
    }

    fn validate(self) -> CandleResult<Self> {
        if self.hidden_size == 0
            || self.num_layers == 0
            || self.num_heads == 0
            || self.head_dim == 0
            || self.intermediate_size == 0
            || self.patch_size == 0
            || self.num_channels == 0
        {
            return Err(Error::Msg(
                "flux2 Pixtral config dimensions must all be non-zero".into(),
            ));
        }
        if self.num_heads.checked_mul(self.head_dim) != Some(self.hidden_size) {
            return Err(Error::Msg(format!(
                "flux2 Pixtral config hidden_size {} != num_heads {} * head_dim {}",
                self.hidden_size, self.num_heads, self.head_dim
            )));
        }
        // 2-D RoPE assigns alternating base frequencies to the height/width halves, then applies
        // rotate-half. Each axis therefore needs head_dim / 4 frequency values.
        if !self.head_dim.is_multiple_of(4) {
            return Err(Error::Msg(format!(
                "flux2 Pixtral head_dim {} must be divisible by 4",
                self.head_dim
            )));
        }
        if !self.rope_theta.is_finite() || self.rope_theta <= 0.0 {
            return Err(Error::Msg(format!(
                "flux2 Pixtral rope_theta must be finite and positive, got {}",
                self.rope_theta
            )));
        }
        Ok(self)
    }
}

struct PatchConv {
    conv: Conv2d,
    patch: usize,
}

impl PatchConv {
    fn new(vb: VarBuilder, cfg: &PixtralVisionConfig) -> CandleResult<Self> {
        let conv_cfg = Conv2dConfig {
            stride: cfg.patch_size,
            padding: 0,
            dilation: 1,
            groups: 1,
            ..Default::default()
        };
        Ok(Self {
            conv: conv2d_no_bias(
                cfg.num_channels,
                cfg.hidden_size,
                cfg.patch_size,
                conv_cfg,
                vb,
            )?,
            patch: cfg.patch_size,
        })
    }

    /// NCHW `[1, channels, H, W]` -> `[grid_h * grid_w, hidden]` in row-major order.
    fn forward(&self, image: &Tensor) -> CandleResult<Tensor> {
        let y = self.conv.forward(image)?;
        let (batch, hidden, grid_h, grid_w) = y.dims4()?;
        if batch != 1 {
            return Err(Error::Msg(format!(
                "flux2 Pixtral patch conv expected batch 1, got {batch}"
            )));
        }
        y.reshape((hidden, grid_h * grid_w))?
            .transpose(0, 1)?
            .contiguous()
    }
}

struct PixtralMlp {
    gate: Linear,
    up: Linear,
    down: Linear,
}

impl PixtralMlp {
    fn new(vb: VarBuilder, cfg: &PixtralVisionConfig) -> CandleResult<Self> {
        Ok(Self {
            gate: linear_no_bias(cfg.hidden_size, cfg.intermediate_size, vb.pp("gate_proj"))?,
            up: linear_no_bias(cfg.hidden_size, cfg.intermediate_size, vb.pp("up_proj"))?,
            down: linear_no_bias(cfg.intermediate_size, cfg.hidden_size, vb.pp("down_proj"))?,
        })
    }

    fn forward(&self, x: &Tensor) -> CandleResult<Tensor> {
        let gate = self.gate.forward(x)?.silu()?;
        self.down.forward(&(gate * self.up.forward(x)?)?)
    }
}

struct PixtralAttention {
    q: Linear,
    k: Linear,
    v: Linear,
    out: Linear,
    num_heads: usize,
    head_dim: usize,
    scale: f64,
}

impl PixtralAttention {
    fn new(vb: VarBuilder, cfg: &PixtralVisionConfig) -> CandleResult<Self> {
        Ok(Self {
            q: linear_no_bias(cfg.hidden_size, cfg.hidden_size, vb.pp("q_proj"))?,
            k: linear_no_bias(cfg.hidden_size, cfg.hidden_size, vb.pp("k_proj"))?,
            v: linear_no_bias(cfg.hidden_size, cfg.hidden_size, vb.pp("v_proj"))?,
            out: linear_no_bias(cfg.hidden_size, cfg.hidden_size, vb.pp("o_proj"))?,
            num_heads: cfg.num_heads,
            head_dim: cfg.head_dim,
            scale: (cfg.head_dim as f64).powf(-0.5),
        })
    }

    fn to_heads(&self, x: &Tensor) -> CandleResult<Tensor> {
        let seq = x.dim(0)?;
        x.reshape((seq, self.num_heads, self.head_dim))?
            .transpose(0, 1)?
            .contiguous()
    }

    /// `x`: `[seq, hidden]`; `cos/sin`: `[seq, head_dim]`; `cu`: image boundaries.
    fn forward(
        &self,
        x: &Tensor,
        cos: &Tensor,
        sin: &Tensor,
        cu: &[usize],
    ) -> CandleResult<Tensor> {
        let seq = x.dim(0)?;
        let q = apply_rope(&self.to_heads(&self.q.forward(x)?)?, cos, sin)?;
        let k = apply_rope(&self.to_heads(&self.k.forward(x)?)?, cos, sin)?;
        let v = self.to_heads(&self.v.forward(x)?)?;

        let mut windows = Vec::with_capacity(cu.len().saturating_sub(1));
        for bounds in cu.windows(2) {
            let start = bounds[0];
            let len = bounds[1] - start;
            let q_window = q.narrow(1, start, len)?.unsqueeze(0)?.contiguous()?;
            let k_window = k.narrow(1, start, len)?.unsqueeze(0)?.contiguous()?;
            let v_window = v.narrow(1, start, len)?.unsqueeze(0)?.contiguous()?;
            let attended = candle_gen::sdpa_budgeted_bhsd(
                &q_window,
                &k_window,
                &v_window,
                self.scale,
                None,
                softmax_last_dim,
                candle_gen::ATTN_SCORES_BUDGET,
            )?;
            windows.push(attended.squeeze(0)?); // [heads, image_seq, head_dim]
        }
        if windows.is_empty() {
            return Err(Error::Msg(
                "flux2 Pixtral attention requires at least one image window".into(),
            ));
        }
        let refs: Vec<&Tensor> = windows.iter().collect();
        let attended = Tensor::cat(&refs, 1)?;
        let merged = attended
            .transpose(0, 1)?
            .contiguous()?
            .reshape((seq, self.num_heads * self.head_dim))?;
        self.out.forward(&merged)
    }
}

/// Non-interleaved rotate-half RoPE: `x*cos + [-x2, x1]*sin`, all in f32.
fn apply_rope(x: &Tensor, cos: &Tensor, sin: &Tensor) -> CandleResult<Tensor> {
    let (_heads, seq, head_dim) = x.dims3()?;
    if head_dim % 2 != 0 {
        return Err(Error::Msg(format!(
            "flux2 Pixtral RoPE head_dim {head_dim} must be even"
        )));
    }
    if cos.dims2()? != (seq, head_dim) || sin.dims2()? != (seq, head_dim) {
        return Err(Error::Msg(format!(
            "flux2 Pixtral RoPE table shape must be [{seq}, {head_dim}]"
        )));
    }
    let x = x.to_dtype(DType::F32)?;
    let half = head_dim / 2;
    let first = x.narrow(2, 0, half)?;
    let second = x.narrow(2, half, half)?;
    let rotated = Tensor::cat(&[&second.neg()?, &first], 2)?;
    let cos = cos.unsqueeze(0)?;
    let sin = sin.unsqueeze(0)?;
    x.broadcast_mul(&cos)? + rotated.broadcast_mul(&sin)?
}

struct PixtralBlock {
    attention_norm: RmsNorm,
    ffn_norm: RmsNorm,
    attention: PixtralAttention,
    feed_forward: PixtralMlp,
}

impl PixtralBlock {
    fn new(vb: VarBuilder, cfg: &PixtralVisionConfig) -> CandleResult<Self> {
        Ok(Self {
            attention_norm: rms_norm(cfg.hidden_size, cfg.rms_norm_eps, vb.pp("attention_norm"))?,
            ffn_norm: rms_norm(cfg.hidden_size, cfg.rms_norm_eps, vb.pp("ffn_norm"))?,
            attention: PixtralAttention::new(vb.pp("attention"), cfg)?,
            feed_forward: PixtralMlp::new(vb.pp("feed_forward"), cfg)?,
        })
    }

    fn forward(
        &self,
        x: &Tensor,
        cos: &Tensor,
        sin: &Tensor,
        cu: &[usize],
    ) -> CandleResult<Tensor> {
        let normed = self.attention_norm.forward(x)?;
        let hidden = (x + self.attention.forward(&normed, cos, sin, cu)?)?;
        let normed = self.ffn_norm.forward(&hidden)?;
        &hidden + self.feed_forward.forward(&normed)?
    }
}

/// Dense full-precision FLUX.2-dev Pixtral vision tower.
pub struct PixtralVisionTower {
    patch_conv: PatchConv,
    ln_pre: RmsNorm,
    layers: Vec<PixtralBlock>,
    config: PixtralVisionConfig,
}

impl PixtralVisionTower {
    /// Load the production FLUX.2-dev tower from the exact `vision_tower.*` namespace.
    ///
    /// `vb` must point at the text-encoder shard set and already target the device on which the
    /// tower should run. SceneWorks' Candle FLUX.2 loader uses f32 for this VarBuilder, preserving
    /// the provider's dense-f32 vision contract.
    pub fn new(vb: VarBuilder) -> CandleResult<Self> {
        Self::new_with_config(vb, PixtralVisionConfig::dev())
    }

    fn new_with_config(vb: VarBuilder, config: PixtralVisionConfig) -> CandleResult<Self> {
        let config = config.validate()?;
        if vb.dtype() != DType::F32 {
            return Err(Error::Msg(format!(
                "flux2 Pixtral vision tower requires an f32 VarBuilder, got {:?}",
                vb.dtype()
            )));
        }
        let vb = vb.pp(VISION_PREFIX);
        let patch_conv = PatchConv::new(vb.pp("patch_conv"), &config)?;
        let ln_pre = rms_norm(config.hidden_size, config.rms_norm_eps, vb.pp("ln_pre"))?;
        let mut layers = Vec::with_capacity(config.num_layers);
        let layer_vb = vb.pp("transformer.layers");
        for index in 0..config.num_layers {
            layers.push(PixtralBlock::new(layer_vb.pp(index), &config)?);
        }
        Ok(Self {
            patch_conv,
            ln_pre,
            layers,
            config,
        })
    }

    /// Encode NCHW f32 reference images into `[sum(grid_h * grid_w), 1024]` features.
    ///
    /// The supplied grids must exactly match the pixel geometry (`H = grid_h * 14`, `W =
    /// grid_w * 14`); no silent patch-flooring is permitted. Cancellation is checked before patch
    /// encoding and between every vision block.
    pub fn forward(
        &self,
        images: &[&Tensor],
        grids: &[(usize, usize)],
        cancel: &CancelFlag,
    ) -> GenResult<Tensor> {
        candle_gen::check_cancel(cancel)?;
        self.validate_inputs(images, grids)?;

        let mut patches = Vec::with_capacity(images.len());
        for image in images {
            patches.push(self.patch_conv.forward(image)?);
        }
        let mut x = if patches.len() == 1 {
            patches.remove(0)
        } else {
            let refs: Vec<&Tensor> = patches.iter().collect();
            Tensor::cat(&refs, 0)?
        };
        x = self.ln_pre.forward(&x)?;

        let (cos, sin) = rope_2d(
            grids,
            self.config.head_dim,
            self.config.rope_theta,
            x.device(),
        )?;
        let cu = cu_seqlens(grids)?;
        for layer in &self.layers {
            candle_gen::check_cancel(cancel)?;
            x = layer.forward(&x, &cos, &sin, &cu)?;
        }
        Ok(x)
    }

    fn validate_inputs(&self, images: &[&Tensor], grids: &[(usize, usize)]) -> GenResult<()> {
        if images.is_empty() {
            return Err(CandleError::Msg(
                "flux2 Pixtral vision tower requires at least one image".into(),
            ));
        }
        if images.len() != grids.len() {
            return Err(CandleError::Msg(format!(
                "flux2 Pixtral received {} images but {} grids",
                images.len(),
                grids.len()
            )));
        }
        for (index, (image, &(grid_h, grid_w))) in images.iter().zip(grids).enumerate() {
            if grid_h == 0 || grid_w == 0 {
                return Err(CandleError::Msg(format!(
                    "flux2 Pixtral image {index} has an empty {grid_h}x{grid_w} patch grid"
                )));
            }
            if image.dtype() != DType::F32 {
                return Err(CandleError::Msg(format!(
                    "flux2 Pixtral image {index} must be f32, got {:?}",
                    image.dtype()
                )));
            }
            let (batch, channels, height, width) = image.dims4()?;
            let expected_height = grid_h.checked_mul(self.patch_conv.patch).ok_or_else(|| {
                CandleError::Msg(format!("flux2 Pixtral image {index} height overflow"))
            })?;
            let expected_width = grid_w.checked_mul(self.patch_conv.patch).ok_or_else(|| {
                CandleError::Msg(format!("flux2 Pixtral image {index} width overflow"))
            })?;
            if batch != 1
                || channels != self.config.num_channels
                || height != expected_height
                || width != expected_width
            {
                return Err(CandleError::Msg(format!(
                    "flux2 Pixtral image {index} expected [1, {}, {expected_height}, {expected_width}], got [{batch}, {channels}, {height}, {width}]",
                    self.config.num_channels
                )));
            }
        }
        Ok(())
    }
}

/// Mistral3 projector: RMSNorm -> 2x2 channel-major patch merge -> linear/GELU/linear.
pub struct Mistral3Projector {
    norm: RmsNorm,
    merging_layer: Linear,
    linear_1: Linear,
    linear_2: Linear,
    vision_hidden: usize,
    spatial_merge: usize,
}

impl Mistral3Projector {
    /// Load the production projector from the exact `multi_modal_projector.*` namespace.
    pub fn new(vb: VarBuilder) -> CandleResult<Self> {
        Self::new_with_config(
            vb,
            PixtralVisionConfig::dev().hidden_size,
            PROJECTOR_TEXT_HIDDEN,
            PROJECTOR_SPATIAL_MERGE,
            PROJECTOR_RMS_EPS,
        )
    }

    fn new_with_config(
        vb: VarBuilder,
        vision_hidden: usize,
        text_hidden: usize,
        spatial_merge: usize,
        eps: f64,
    ) -> CandleResult<Self> {
        if vb.dtype() != DType::F32 {
            return Err(Error::Msg(format!(
                "flux2 Mistral3 projector requires an f32 VarBuilder, got {:?}",
                vb.dtype()
            )));
        }
        if vision_hidden == 0 || text_hidden == 0 || spatial_merge == 0 {
            return Err(Error::Msg(
                "flux2 Mistral3 projector dimensions must be non-zero".into(),
            ));
        }
        let merged_width = vision_hidden
            .checked_mul(spatial_merge)
            .and_then(|v| v.checked_mul(spatial_merge))
            .ok_or_else(|| Error::Msg("flux2 Mistral3 projector width overflow".into()))?;
        let vb = vb.pp(PROJECTOR_PREFIX);
        Ok(Self {
            norm: rms_norm(vision_hidden, eps, vb.pp("norm"))?,
            merging_layer: linear_no_bias(
                merged_width,
                vision_hidden,
                vb.pp("patch_merger.merging_layer"),
            )?,
            linear_1: linear_no_bias(vision_hidden, text_hidden, vb.pp("linear_1"))?,
            linear_2: linear_no_bias(text_hidden, text_hidden, vb.pp("linear_2"))?,
            vision_hidden,
            spatial_merge,
        })
    }

    /// Project `[sum(grid_h * grid_w), vision_hidden]` to merged Mistral image tokens.
    pub fn forward(
        &self,
        image_features: &Tensor,
        grids: &[(usize, usize)],
    ) -> CandleResult<Tensor> {
        if image_features.dtype() != DType::F32 {
            return Err(Error::Msg(format!(
                "flux2 Mistral3 projector features must be f32, got {:?}",
                image_features.dtype()
            )));
        }
        let (_, width) = image_features.dims2()?;
        if width != self.vision_hidden {
            return Err(Error::Msg(format!(
                "flux2 Mistral3 projector expected feature width {}, got {width}",
                self.vision_hidden
            )));
        }
        let normed = self.norm.forward(image_features)?;
        let merged = patch_merge(&normed, grids, self.spatial_merge)?;
        let hidden = self.merging_layer.forward(&merged)?;
        let hidden = self.linear_1.forward(&hidden)?.gelu_erf()?;
        self.linear_2.forward(&hidden)
    }
}

/// Cumulative image patch counts (`[0, n0, n0+n1, ...]`) for block-diagonal attention.
pub fn cu_seqlens(grids: &[(usize, usize)]) -> CandleResult<Vec<usize>> {
    if grids.is_empty() {
        return Err(Error::Msg(
            "flux2 Pixtral requires at least one patch grid".into(),
        ));
    }
    let mut cumulative = Vec::with_capacity(grids.len() + 1);
    cumulative.push(0usize);
    let mut offset = 0usize;
    for &(grid_h, grid_w) in grids {
        if grid_h == 0 || grid_w == 0 {
            return Err(Error::Msg(format!(
                "flux2 Pixtral patch grids must be non-zero, got {grid_h}x{grid_w}"
            )));
        }
        let patches = grid_h
            .checked_mul(grid_w)
            .ok_or_else(|| Error::Msg("flux2 Pixtral patch-grid area overflow".into()))?;
        offset = offset
            .checked_add(patches)
            .ok_or_else(|| Error::Msg("flux2 Pixtral cumulative sequence overflow".into()))?;
        cumulative.push(offset);
    }
    Ok(cumulative)
}

/// Build Pixtral's 2-D non-interleaved RoPE tables in f32.
fn rope_2d(
    grids: &[(usize, usize)],
    head_dim: usize,
    theta: f32,
    device: &Device,
) -> CandleResult<(Tensor, Tensor)> {
    if head_dim == 0 || !head_dim.is_multiple_of(4) {
        return Err(Error::Msg(format!(
            "flux2 Pixtral RoPE head_dim must be positive and divisible by 4, got {head_dim}"
        )));
    }
    if !theta.is_finite() || theta <= 0.0 {
        return Err(Error::Msg(format!(
            "flux2 Pixtral RoPE theta must be finite and positive, got {theta}"
        )));
    }
    let cu = cu_seqlens(grids)?;
    let seq = *cu.last().expect("cu_seqlens always has an end boundary");
    let half = head_dim / 2;
    let base: Vec<f32> = (0..half)
        .map(|index| 1.0 / theta.powf((2 * index) as f32 / head_dim as f32))
        .collect();
    let height_freqs: Vec<f32> = base.iter().step_by(2).copied().collect();
    let width_freqs: Vec<f32> = base.iter().skip(1).step_by(2).copied().collect();

    let capacity = seq
        .checked_mul(head_dim)
        .ok_or_else(|| Error::Msg("flux2 Pixtral RoPE allocation overflow".into()))?;
    let mut cos = Vec::with_capacity(capacity);
    let mut sin = Vec::with_capacity(capacity);
    for &(grid_h, grid_w) in grids {
        for row in 0..grid_h {
            for column in 0..grid_w {
                let mut frequencies = Vec::with_capacity(half);
                frequencies.extend(height_freqs.iter().map(|freq| row as f32 * freq));
                frequencies.extend(width_freqs.iter().map(|freq| column as f32 * freq));
                for _ in 0..2 {
                    for &value in &frequencies {
                        cos.push(value.cos());
                        sin.push(value.sin());
                    }
                }
            }
        }
    }
    Ok((
        Tensor::from_vec(cos, (seq, head_dim), device)?,
        Tensor::from_vec(sin, (seq, head_dim), device)?,
    ))
}

/// Per-image unfold with kernel=stride=`spatial_merge`, ordered `(channel, row, column)`.
fn patch_merge(
    features: &Tensor,
    grids: &[(usize, usize)],
    spatial_merge: usize,
) -> CandleResult<Tensor> {
    if spatial_merge == 0 {
        return Err(Error::Msg(
            "flux2 Mistral3 spatial_merge must be non-zero".into(),
        ));
    }
    if grids.is_empty() {
        return Err(Error::Msg(
            "flux2 Mistral3 projector requires at least one patch grid".into(),
        ));
    }
    let (sequence, hidden) = features.dims2()?;
    let mut cursor = 0usize;
    let mut merged_images = Vec::with_capacity(grids.len());
    for &(grid_h, grid_w) in grids {
        if grid_h == 0 || grid_w == 0 {
            return Err(Error::Msg(format!(
                "flux2 Mistral3 patch grids must be non-zero, got {grid_h}x{grid_w}"
            )));
        }
        if grid_h % spatial_merge != 0 || grid_w % spatial_merge != 0 {
            return Err(Error::Msg(format!(
                "flux2 Mistral3 patch grid {grid_h}x{grid_w} is not divisible by spatial_merge {spatial_merge}"
            )));
        }
        let patch_count = grid_h
            .checked_mul(grid_w)
            .ok_or_else(|| Error::Msg("flux2 Mistral3 patch-grid area overflow".into()))?;
        let end = cursor
            .checked_add(patch_count)
            .ok_or_else(|| Error::Msg("flux2 Mistral3 feature offset overflow".into()))?;
        if end > sequence {
            return Err(Error::Msg(format!(
                "flux2 Mistral3 grids require at least {end} features, got {sequence}"
            )));
        }
        let merged_height = grid_h / spatial_merge;
        let merged_width = grid_w / spatial_merge;
        // [gh, gw, d] -> [gh/s, s, gw/s, s, d] -> [gh/s, gw/s, d, s, s].
        let image = features.narrow(0, cursor, patch_count)?;
        let blocks = image
            .reshape((
                merged_height,
                spatial_merge,
                merged_width,
                spatial_merge,
                hidden,
            ))?
            .permute((0, 2, 4, 1, 3))?
            .contiguous()?
            .reshape((
                merged_height * merged_width,
                hidden * spatial_merge * spatial_merge,
            ))?;
        merged_images.push(blocks);
        cursor = end;
    }
    if cursor != sequence {
        return Err(Error::Msg(format!(
            "flux2 Mistral3 grids describe {cursor} features, got {sequence}"
        )));
    }
    if merged_images.len() == 1 {
        Ok(merged_images.remove(0))
    } else {
        let refs: Vec<&Tensor> = merged_images.iter().collect();
        Tensor::cat(&refs, 0)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use candle_gen::candle_core::Shape;
    use std::collections::HashMap;

    fn tiny_config() -> PixtralVisionConfig {
        PixtralVisionConfig {
            hidden_size: 8,
            num_layers: 1,
            num_heads: 2,
            head_dim: 4,
            intermediate_size: 16,
            patch_size: 2,
            rope_theta: 10_000.0,
            rms_norm_eps: 1e-5,
            num_channels: 3,
        }
    }

    fn insert_zeros(
        tensors: &mut HashMap<String, Tensor>,
        name: impl Into<String>,
        shape: impl Into<Shape>,
        device: &Device,
    ) {
        tensors.insert(
            name.into(),
            Tensor::zeros(shape, DType::F32, device).unwrap(),
        );
    }

    fn tiny_vision_vb(device: &Device) -> VarBuilder<'static> {
        let cfg = tiny_config();
        let mut tensors = HashMap::new();
        insert_zeros(
            &mut tensors,
            "vision_tower.patch_conv.weight",
            (
                cfg.hidden_size,
                cfg.num_channels,
                cfg.patch_size,
                cfg.patch_size,
            ),
            device,
        );
        tensors.insert(
            "vision_tower.ln_pre.weight".into(),
            Tensor::ones(cfg.hidden_size, DType::F32, device).unwrap(),
        );
        let prefix = "vision_tower.transformer.layers.0";
        for norm in ["attention_norm", "ffn_norm"] {
            tensors.insert(
                format!("{prefix}.{norm}.weight"),
                Tensor::ones(cfg.hidden_size, DType::F32, device).unwrap(),
            );
        }
        for projection in ["q_proj", "k_proj", "v_proj", "o_proj"] {
            insert_zeros(
                &mut tensors,
                format!("{prefix}.attention.{projection}.weight"),
                (cfg.hidden_size, cfg.hidden_size),
                device,
            );
        }
        for (projection, output, input) in [
            ("gate_proj", cfg.intermediate_size, cfg.hidden_size),
            ("up_proj", cfg.intermediate_size, cfg.hidden_size),
            ("down_proj", cfg.hidden_size, cfg.intermediate_size),
        ] {
            insert_zeros(
                &mut tensors,
                format!("{prefix}.feed_forward.{projection}.weight"),
                (output, input),
                device,
            );
        }
        VarBuilder::from_tensors(tensors, DType::F32, device)
    }

    fn tiny_projector_vb(device: &Device) -> VarBuilder<'static> {
        let (vision_hidden, text_hidden, spatial_merge) = (4usize, 6usize, 2usize);
        let mut tensors = HashMap::new();
        tensors.insert(
            "multi_modal_projector.norm.weight".into(),
            Tensor::ones(vision_hidden, DType::F32, device).unwrap(),
        );
        insert_zeros(
            &mut tensors,
            "multi_modal_projector.patch_merger.merging_layer.weight",
            (vision_hidden, vision_hidden * spatial_merge * spatial_merge),
            device,
        );
        insert_zeros(
            &mut tensors,
            "multi_modal_projector.linear_1.weight",
            (text_hidden, vision_hidden),
            device,
        );
        insert_zeros(
            &mut tensors,
            "multi_modal_projector.linear_2.weight",
            (text_hidden, text_hidden),
            device,
        );
        VarBuilder::from_tensors(tensors, DType::F32, device)
    }

    #[test]
    fn dev_config_matches_pixtral_checkpoint() {
        let cfg = PixtralVisionConfig::dev();
        assert_eq!(cfg.num_layers, 24);
        assert_eq!(cfg.hidden_size, 1024);
        assert_eq!(cfg.num_heads * cfg.head_dim, cfg.hidden_size);
        assert_eq!(cfg.intermediate_size, 4096);
        assert_eq!(cfg.patch_size, 14);
        assert_eq!(cfg.num_channels, 3);
    }

    #[test]
    fn cumulative_seqlens_preserve_image_boundaries() {
        assert_eq!(cu_seqlens(&[(2, 3), (1, 4)]).unwrap(), vec![0, 6, 10]);
        assert!(cu_seqlens(&[]).is_err());
        assert!(cu_seqlens(&[(0, 4)]).is_err());
    }

    #[test]
    fn rope_geometry_splits_height_and_width_frequencies() {
        let device = Device::Cpu;
        let (cos, sin) = rope_2d(&[(1, 2)], 8, 10_000.0, &device).unwrap();
        assert_eq!(cos.dims2().unwrap(), (2, 8));
        assert_eq!(sin.dims2().unwrap(), (2, 8));
        let cos = cos.to_vec2::<f32>().unwrap();
        let sin = sin.to_vec2::<f32>().unwrap();
        // (row=0,col=0) is identity. For (0,1), height frequencies remain zero while the
        // width frequencies are base[1]=0.1 and base[3]=0.001, duplicated for rotate-half.
        assert!(cos[0].iter().all(|value| (*value - 1.0).abs() < 1e-7));
        assert!(sin[0].iter().all(|value| value.abs() < 1e-7));
        let expected = [0.0f32, 0.0, 0.1, 0.001, 0.0, 0.0, 0.1, 0.001];
        for (index, angle) in expected.into_iter().enumerate() {
            assert!((cos[1][index] - angle.cos()).abs() < 1e-6);
            assert!((sin[1][index] - angle.sin()).abs() < 1e-6);
        }
    }

    #[test]
    fn rotate_half_rope_uses_non_interleaved_halves() {
        let device = Device::Cpu;
        let x = Tensor::from_vec(vec![1f32, 2., 3., 4.], (1, 1, 4), &device).unwrap();
        let cos = Tensor::zeros((1, 4), DType::F32, &device).unwrap();
        let sin = Tensor::ones((1, 4), DType::F32, &device).unwrap();
        let rotated = apply_rope(&x, &cos, &sin)
            .unwrap()
            .flatten_all()
            .unwrap()
            .to_vec1::<f32>()
            .unwrap();
        assert_eq!(rotated, vec![-3.0, -4.0, 1.0, 2.0]);
    }

    #[test]
    fn patch_merge_is_channel_major_inside_each_two_by_two_block() {
        let device = Device::Cpu;
        let features =
            Tensor::from_vec((0..16).map(|v| v as f32).collect(), (4, 4), &device).unwrap();
        let merged = patch_merge(&features, &[(2, 2)], 2)
            .unwrap()
            .to_vec2::<f32>()
            .unwrap();
        assert_eq!(
            merged[0],
            vec![0., 4., 8., 12., 1., 5., 9., 13., 2., 6., 10., 14., 3., 7., 11., 15.]
        );
    }

    #[test]
    fn patch_merge_rejects_bad_grid_and_feature_counts() {
        let device = Device::Cpu;
        let features = Tensor::zeros((4, 4), DType::F32, &device).unwrap();
        let odd = patch_merge(&features, &[(1, 4)], 2).unwrap_err();
        assert!(odd.to_string().contains("not divisible"));
        let short = patch_merge(&features, &[(2, 2), (2, 2)], 2).unwrap_err();
        assert!(short.to_string().contains("require at least"));
        let extra = patch_merge(&features, &[(1, 2)], 1).unwrap_err();
        assert!(extra.to_string().contains("describe 2 features"));
    }

    #[test]
    fn tiny_tower_runs_multiple_images_and_checks_geometry() {
        let device = Device::Cpu;
        let tower =
            PixtralVisionTower::new_with_config(tiny_vision_vb(&device), tiny_config()).unwrap();
        let first = Tensor::zeros((1, 3, 4, 4), DType::F32, &device).unwrap();
        let second = Tensor::zeros((1, 3, 2, 4), DType::F32, &device).unwrap();
        let output = tower
            .forward(
                &[&first, &second],
                &[(2, 2), (1, 2)],
                &CancelFlag::default(),
            )
            .unwrap();
        assert_eq!(output.dims2().unwrap(), (6, 8));

        let error = tower
            .forward(&[&first], &[(1, 2)], &CancelFlag::default())
            .unwrap_err();
        assert!(error.to_string().contains("expected [1, 3, 2, 4]"));
    }

    #[test]
    fn tower_honors_pre_cancel_before_tensor_work() {
        let device = Device::Cpu;
        let tower =
            PixtralVisionTower::new_with_config(tiny_vision_vb(&device), tiny_config()).unwrap();
        let image = Tensor::zeros((1, 3, 4, 4), DType::F32, &device).unwrap();
        let cancel = CancelFlag::default();
        cancel.cancel();
        let error = tower.forward(&[&image], &[(2, 2)], &cancel).unwrap_err();
        assert!(matches!(error, CandleError::Canceled));
    }

    #[test]
    fn tiny_projector_loads_exact_prefix_and_returns_expected_shape() {
        let device = Device::Cpu;
        let projector =
            Mistral3Projector::new_with_config(tiny_projector_vb(&device), 4, 6, 2, 1e-5).unwrap();
        let features = Tensor::ones((8, 4), DType::F32, &device).unwrap();
        let projected = projector.forward(&features, &[(2, 2), (2, 2)]).unwrap();
        assert_eq!(projected.dims2().unwrap(), (2, 6));
    }
}
