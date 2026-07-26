//! Mage-VAE one-step decoder.
//!
//! This is not AutoencoderKL: `[B,128,h,w]` is decoded by a CoD conditioning decoder followed by
//! one zero-timestep DConv denoiser and a per-patch pixel MLP.

use std::path::Path;

use candle_core::{DType, Device, Module, Result, Tensor, D};
use candle_gen_boogu::loader::{linear, Weights};
use candle_nn::{Conv2d, Conv2dConfig, Linear};

use crate::config::{LATENT_CHANNELS, NORM_EPS, VAE_DOWNSAMPLE};

fn conv(
    w: &Weights,
    prefix: &str,
    stride: usize,
    padding: usize,
    groups: usize,
    bias: bool,
) -> Result<Conv2d> {
    let weight = w.get(&format!("{prefix}.weight"))?;
    let bias = if bias {
        Some(w.get(&format!("{prefix}.bias"))?)
    } else {
        None
    };
    Ok(Conv2d::new(
        weight,
        bias,
        Conv2dConfig {
            padding,
            stride,
            dilation: 1,
            groups,
            cudnn_fwd_algo: None,
        },
    ))
}

fn layer_norm_channels(
    x: &Tensor,
    weight: Option<&Tensor>,
    bias: Option<&Tensor>,
) -> Result<Tensor> {
    let nhwc = x.permute((0, 2, 3, 1))?;
    let f = nhwc.to_dtype(DType::F32)?;
    let mean = f.mean_keepdim(D::Minus1)?;
    let c = f.broadcast_sub(&mean)?;
    let d = c
        .sqr()?
        .mean_keepdim(D::Minus1)?
        .affine(1., NORM_EPS)?
        .sqrt()?;
    let mut y = c.broadcast_div(&d)?;
    if let Some(weight) = weight {
        y = y.broadcast_mul(&weight.to_dtype(DType::F32)?)?;
    }
    if let Some(bias) = bias {
        y = y.broadcast_add(&bias.to_dtype(DType::F32)?)?;
    }
    y.to_dtype(x.dtype())?.permute((0, 3, 1, 2))
}

fn group_norm_32(x: &Tensor, weight: &Tensor, bias: &Tensor) -> Result<Tensor> {
    let (b, c, h, w) = x.dims4()?;
    let groups = 32usize;
    let f = x
        .to_dtype(DType::F32)?
        .reshape((b, groups, c / groups, h, w))?;
    let mean = f.mean_keepdim((2, 3, 4))?;
    let centered = f.broadcast_sub(&mean)?;
    let denom = centered
        .sqr()?
        .mean_keepdim((2, 3, 4))?
        .affine(1., NORM_EPS)?
        .sqrt()?;
    centered
        .broadcast_div(&denom)?
        .reshape((b, c, h, w))?
        .broadcast_mul(&weight.to_dtype(DType::F32)?.reshape((1, c, 1, 1))?)?
        .broadcast_add(&bias.to_dtype(DType::F32)?.reshape((1, c, 1, 1))?)?
        .to_dtype(x.dtype())
}

struct AffineNorm {
    weight: Tensor,
    bias: Tensor,
}

impl AffineNorm {
    fn load(w: &Weights, prefix: &str) -> Result<Self> {
        Ok(Self {
            weight: w.get(&format!("{prefix}.weight"))?,
            bias: w.get(&format!("{prefix}.bias"))?,
        })
    }
    fn group(&self, x: &Tensor) -> Result<Tensor> {
        group_norm_32(x, &self.weight, &self.bias)
    }
    fn layer(&self, x: &Tensor) -> Result<Tensor> {
        layer_norm_channels(x, Some(&self.weight), Some(&self.bias))
    }
}

struct Resnet {
    n1: AffineNorm,
    c1: Conv2d,
    n2: AffineNorm,
    c2: Conv2d,
}

impl Resnet {
    fn load(w: &Weights, prefix: &str) -> Result<Self> {
        Ok(Self {
            n1: AffineNorm::load(w, &format!("{prefix}.norm1"))?,
            c1: conv(w, &format!("{prefix}.conv1"), 1, 1, 1, true)?,
            n2: AffineNorm::load(w, &format!("{prefix}.norm2"))?,
            c2: conv(w, &format!("{prefix}.conv2"), 1, 1, 1, true)?,
        })
    }
    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let h = self.c1.forward(&self.n1.group(x)?.silu()?)?;
        x + self.c2.forward(&self.n2.group(&h)?.silu()?)?
    }
}

struct LocalAttention {
    norm: AffineNorm,
    q: Conv2d,
    k: Conv2d,
    v: Conv2d,
    out: Conv2d,
}

impl LocalAttention {
    fn load(w: &Weights, prefix: &str) -> Result<Self> {
        Ok(Self {
            norm: AffineNorm::load(w, &format!("{prefix}.norm"))?,
            q: conv(w, &format!("{prefix}.q"), 1, 0, 1, true)?,
            k: conv(w, &format!("{prefix}.k"), 1, 0, 1, true)?,
            v: conv(w, &format!("{prefix}.v"), 1, 0, 1, true)?,
            out: conv(w, &format!("{prefix}.proj_out"), 1, 0, 1, true)?,
        })
    }

    fn padded_index(n: usize, tile: usize, device: &Device) -> Result<Tensor> {
        let padded = n.div_ceil(tile) * tile;
        let ids: Vec<u32> = (0..padded).map(|i| i.min(n - 1) as u32).collect();
        Tensor::from_vec(ids, padded, device)
    }

    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let (_, c, h, w) = x.dims4()?;
        let norm = self.norm.group(x)?;
        let hi = Self::padded_index(h, 32, x.device())?;
        let wi = Self::padded_index(w, 32, x.device())?;
        let tile = |conv: &Conv2d| -> Result<Tensor> {
            let t = conv
                .forward(&norm)?
                .index_select(&hi, 2)?
                .index_select(&wi, 3)?;
            let (_, _, hp, wp) = t.dims4()?;
            t.reshape((1, c, hp / 32, 32, wp / 32, 32))?
                .permute((0, 2, 4, 3, 5, 1))?
                .reshape((hp / 32 * wp / 32, 1, 1024, c))
        };
        let q = tile(&self.q)?;
        let k = tile(&self.k)?;
        let v = tile(&self.v)?;
        let o = candle_gen::sdpa_budgeted_bhsd(
            &q,
            &k,
            &v,
            (c as f64).powf(-0.5),
            None,
            candle_nn::ops::softmax_last_dim,
            candle_gen::ATTN_SCORES_BUDGET,
        )?;
        let hp = h.div_ceil(32) * 32;
        let wp = w.div_ceil(32) * 32;
        let o = o
            .reshape((1, hp / 32, wp / 32, 32, 32, c))?
            .permute((0, 5, 1, 3, 2, 4))?
            .reshape((1, c, hp, wp))?
            .narrow(2, 0, h)?
            .narrow(3, 0, w)?;
        x + self.out.forward(&o)?
    }
}

struct CodDecoder {
    input: Conv2d,
    r0: Resnet,
    a0: LocalAttention,
    r1: Resnet,
    a1: LocalAttention,
    r2: Resnet,
    norm: AffineNorm,
    output: Conv2d,
}

impl CodDecoder {
    fn load(w: &Weights) -> Result<Self> {
        let p = "pipeline.y_embedder.decoder";
        Ok(Self {
            input: conv(w, &format!("{p}.conv_in"), 1, 1, 1, true)?,
            r0: Resnet::load(w, &format!("{p}.block.0"))?,
            a0: LocalAttention::load(w, &format!("{p}.block.1"))?,
            r1: Resnet::load(w, &format!("{p}.block.2"))?,
            a1: LocalAttention::load(w, &format!("{p}.block.3"))?,
            r2: Resnet::load(w, &format!("{p}.block.4"))?,
            norm: AffineNorm::load(w, &format!("{p}.norm_out"))?,
            output: conv(w, &format!("{p}.conv_out"), 1, 1, 1, true)?,
        })
    }
    fn forward(&self, z: &Tensor) -> Result<Tensor> {
        let h = self.r0.forward(&self.input.forward(z)?)?;
        let h = self.a0.forward(&h)?;
        let h = self.r1.forward(&h)?;
        let h = self.a1.forward(&h)?;
        let h = self.r2.forward(&h)?;
        self.output.forward(&self.norm.group(&h)?.silu()?)
    }
}

struct DiCoCore {
    c1: Conv2d,
    c2: Conv2d,
    c3: Conv2d,
    ca: Conv2d,
    c4: Conv2d,
    c5: Conv2d,
    adaln: Linear,
}

impl DiCoCore {
    fn load(w: &Weights, prefix: &str) -> Result<Self> {
        Ok(Self {
            c1: conv(w, &format!("{prefix}.conv1"), 1, 0, 1, true)?,
            c2: conv(w, &format!("{prefix}.conv2"), 1, 1, 384, true)?,
            c3: conv(w, &format!("{prefix}.conv3"), 1, 0, 1, true)?,
            ca: conv(w, &format!("{prefix}.ca.1"), 1, 0, 1, true)?,
            c4: conv(w, &format!("{prefix}.conv4"), 1, 0, 1, true)?,
            c5: conv(w, &format!("{prefix}.conv5"), 1, 0, 1, true)?,
            adaln: linear(w, &format!("{prefix}.adaLN_modulation.1"), true)?,
        })
    }
    fn forward(&self, x: &Tensor, conditioning: &Tensor) -> Result<Tensor> {
        let p = self.adaln.forward(&conditioning.silu()?)?;
        let chunks = p.chunk(6, 1)?;
        let e = |i: usize| chunks[i].reshape((chunks[i].dim(0)?, 384, 1, 1));
        let h = layer_norm_channels(x, None, None)?;
        let h = h.broadcast_mul(&(e(1)? + 1.)?)?.broadcast_add(&e(0)?)?;
        let h = self.c2.forward(&self.c1.forward(&h)?)?.gelu_erf()?;
        let ca = candle_nn::ops::sigmoid(&self.ca.forward(&h.mean_keepdim((2, 3))?)?)?;
        let h = self.c3.forward(&h.broadcast_mul(&ca)?)?;
        let x = (x + h.broadcast_mul(&e(2)?)?)?;
        let h = layer_norm_channels(&x, None, None)?;
        let h = h.broadcast_mul(&(e(4)? + 1.)?)?.broadcast_add(&e(3)?)?;
        let h = self.c5.forward(&self.c4.forward(&h)?.gelu_erf()?)?;
        x + h.broadcast_mul(&e(5)?)?
    }
}

struct VaeTime {
    l1: Linear,
    l2: Linear,
}

impl VaeTime {
    fn load(w: &Weights, prefix: &str) -> Result<Self> {
        Ok(Self {
            l1: linear(w, &format!("{prefix}.t_embedder.mlp.0"), true)?,
            l2: linear(w, &format!("{prefix}.t_embedder.mlp.2"), true)?,
        })
    }
    fn zero(&self, batch: usize, dtype: DType, device: &Device) -> Result<Tensor> {
        let input = Tensor::cat(
            &[
                Tensor::ones((batch, 128), DType::F32, device)?,
                Tensor::zeros((batch, 128), DType::F32, device)?,
            ],
            1,
        )?
        .to_dtype(dtype)?;
        self.l2.forward(&self.l1.forward(&input)?.silu()?)
    }
}

struct EncoderCore {
    c1: Conv2d,
    c2: Conv2d,
    c3: Conv2d,
    ca: Conv2d,
    c4: Conv2d,
    c5: Conv2d,
}

impl EncoderCore {
    fn load(w: &Weights, prefix: &str, hidden: usize) -> Result<Self> {
        Ok(Self {
            c1: conv(w, &format!("{prefix}.conv1"), 1, 0, 1, true)?,
            c2: conv(w, &format!("{prefix}.conv2"), 1, 1, hidden, true)?,
            c3: conv(w, &format!("{prefix}.conv3"), 1, 0, 1, true)?,
            ca: conv(w, &format!("{prefix}.ca.1"), 1, 0, 1, true)?,
            c4: conv(w, &format!("{prefix}.conv4"), 1, 0, 1, true)?,
            c5: conv(w, &format!("{prefix}.conv5"), 1, 0, 1, true)?,
        })
    }

    fn spatial(&self, x: &Tensor) -> Result<Tensor> {
        let h = self.c2.forward(&self.c1.forward(x)?)?.gelu_erf()?;
        let ca = candle_nn::ops::sigmoid(&self.ca.forward(&h.mean_keepdim((2, 3))?)?)?;
        self.c3.forward(&h.broadcast_mul(&ca)?)
    }

    fn channel(&self, x: &Tensor) -> Result<Tensor> {
        self.c5.forward(&self.c4.forward(x)?.gelu_erf()?)
    }
}

struct EncoderHeadBlock {
    core: EncoderCore,
    norm1: AffineNorm,
    norm2: AffineNorm,
}

impl EncoderHeadBlock {
    fn load(w: &Weights, prefix: &str) -> Result<Self> {
        Ok(Self {
            core: EncoderCore::load(w, prefix, 768)?,
            norm1: AffineNorm::load(w, &format!("{prefix}.norm1"))?,
            norm2: AffineNorm::load(w, &format!("{prefix}.norm2"))?,
        })
    }

    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let x = (x + self.core.spatial(&self.norm1.layer(x)?)?)?;
        &x + self.core.channel(&self.norm2.layer(&x)?)?
    }
}

pub struct EncoderMoments {
    pub mean: Tensor,
    pub logvar: Tensor,
}

struct MageVaeEncoder {
    patch_cond: Conv2d,
    head: Vec<EncoderHeadBlock>,
    proj_down: Conv2d,
    z_proj: Conv2d,
    fuse_proj: Conv2d,
    time: VaeTime,
    blocks: Vec<DiCoCore>,
    norm_out: AffineNorm,
    proj_out: Conv2d,
    dtype: DType,
}

impl MageVaeEncoder {
    fn load(w: &Weights, dtype: DType) -> Result<Self> {
        const PREFIX: &str = "student.dconv_encoder";
        let head = (0..2)
            .map(|i| EncoderHeadBlock::load(w, &format!("{PREFIX}.head_blocks.{i}")))
            .collect::<Result<Vec<_>>>()?;
        let blocks = (0..21)
            .map(|i| DiCoCore::load(w, &format!("{PREFIX}.blocks.{i}")))
            .collect::<Result<Vec<_>>>()?;
        Ok(Self {
            patch_cond: conv(w, &format!("{PREFIX}.patch_cond_embed"), 16, 0, 1, true)?,
            head,
            proj_down: conv(w, &format!("{PREFIX}.proj_down"), 1, 0, 1, true)?,
            z_proj: conv(w, &format!("{PREFIX}.z_proj"), 1, 0, 1, true)?,
            fuse_proj: conv(w, &format!("{PREFIX}.fuse_proj"), 1, 0, 1, true)?,
            time: VaeTime::load(w, PREFIX)?,
            blocks,
            norm_out: AffineNorm::load(w, &format!("{PREFIX}.norm_out"))?,
            proj_out: conv(w, &format!("{PREFIX}.proj_out"), 1, 0, 1, true)?,
            dtype,
        })
    }

    fn moments(&self, image: &Tensor) -> Result<EncoderMoments> {
        let (b, c, h, w) = image.dims4()?;
        if c != 3 || !h.is_multiple_of(16) || !w.is_multiple_of(16) {
            candle_core::bail!(
                "mage vae encode: expected [B,3,H,W] with H/W divisible by 16, got {:?}",
                image.dims()
            );
        }
        let mut cond = self.patch_cond.forward(&image.to_dtype(self.dtype)?)?;
        for block in &self.head {
            cond = block.forward(&cond)?;
        }
        cond = self.proj_down.forward(&cond)?;
        let z = Tensor::zeros(
            (b, LATENT_CHANNELS, h / 16, w / 16),
            self.dtype,
            image.device(),
        )?;
        let z = self.z_proj.forward(&z)?;
        let mut state = self.fuse_proj.forward(&Tensor::cat(&[cond, z], 1)?)?;
        let time = self.time.zero(b, self.dtype, image.device())?;
        for block in &self.blocks {
            state = block.forward(&state, &time)?;
        }
        let packed = self.proj_out.forward(&self.norm_out.layer(&state)?)?;
        let parts = packed.chunk(2, 1)?;
        let logvar = parts[1].clamp(-20f64, 10f64)?;
        Ok(EncoderMoments {
            mean: parts[0].clone(),
            logvar,
        })
    }
}

struct MlpBlock {
    norm: AffineNorm,
    l1: Linear,
    l2: Linear,
    adaln: Linear,
}

impl MlpBlock {
    fn load(w: &Weights, prefix: &str) -> Result<Self> {
        Ok(Self {
            norm: AffineNorm::load(w, &format!("{prefix}.in_ln"))?,
            l1: linear(w, &format!("{prefix}.mlp.0"), true)?,
            l2: linear(w, &format!("{prefix}.mlp.2"), true)?,
            adaln: linear(w, &format!("{prefix}.adaLN_modulation.1"), true)?,
        })
    }
    fn forward(&self, x: &Tensor, cond: &Tensor) -> Result<Tensor> {
        let p = self.adaln.forward(&cond.silu()?)?;
        let d = p.dim(D::Minus1)? / 3;
        let shift = p.narrow(D::Minus1, 0, d)?;
        let scale = p.narrow(D::Minus1, d, d)?;
        let gate = p.narrow(D::Minus1, 2 * d, d)?;
        let h = self
            .norm
            .layer(&x.unsqueeze(2)?.permute((0, 3, 1, 2))?)?
            .permute((0, 2, 3, 1))?
            .squeeze(2)?;
        let h = h.broadcast_mul(&(scale + 1.)?)?.broadcast_add(&shift)?;
        x + self
            .l2
            .forward(&self.l1.forward(&h)?.silu()?)?
            .broadcast_mul(&gate)?
    }
}

pub struct MageVae {
    encoder: Option<MageVaeEncoder>,
    cod: CodDecoder,
    time: VaeTime,
    patch_embed: Conv2d,
    patch_fuse: Conv2d,
    blocks: Vec<DiCoCore>,
    y_pixels: Conv2d,
    x_embed: Linear,
    cond_embed: Linear,
    input_proj: Linear,
    mlps: Vec<MlpBlock>,
    final_norm: Tensor,
    final_linear: Linear,
    dtype: DType,
}

impl MageVae {
    pub fn load(dir: &Path, device: &Device) -> Result<Self> {
        Self::load_inner(dir, device, false, DType::BF16)
    }

    pub fn load_full(dir: &Path, device: &Device) -> Result<Self> {
        Self::load_inner(dir, device, true, DType::BF16)
    }

    pub fn load_full_dtype(dir: &Path, device: &Device, dtype: DType) -> Result<Self> {
        Self::load_inner(dir, device, true, dtype)
    }

    fn load_inner(dir: &Path, device: &Device, with_encoder: bool, dtype: DType) -> Result<Self> {
        let w = Weights::from_dir(dir, device, dtype)?;
        let mut blocks = Vec::with_capacity(21);
        for i in 0..21 {
            blocks.push(DiCoCore::load(&w, &format!("pipeline.blocks.{i}"))?);
        }
        let mut mlps = Vec::with_capacity(3);
        for i in 0..3 {
            mlps.push(MlpBlock::load(
                &w,
                &format!("pipeline.dec_net.res_blocks.{i}"),
            )?);
        }
        Ok(Self {
            encoder: with_encoder
                .then(|| MageVaeEncoder::load(&w, dtype))
                .transpose()?,
            cod: CodDecoder::load(&w)?,
            time: VaeTime::load(&w, "pipeline")?,
            patch_embed: conv(&w, "pipeline.s_embedder.proj1", 16, 0, 1, false)?,
            patch_fuse: conv(&w, "pipeline.s_embedder.proj2", 1, 0, 1, true)?,
            blocks,
            y_pixels: conv(&w, "pipeline.y_embedder_x", 1, 0, 1, true)?,
            x_embed: linear(&w, "pipeline.x_embedder.embedder.0", true)?,
            cond_embed: linear(&w, "pipeline.dec_net.cond_embed", true)?,
            input_proj: linear(&w, "pipeline.dec_net.input_proj", true)?,
            mlps,
            final_norm: w.get("pipeline.final_layer.norm.weight")?,
            final_linear: linear(&w, "pipeline.final_layer.linear", true)?,
            dtype,
        })
    }

    pub fn encode_moments(&self, image: &Tensor) -> Result<EncoderMoments> {
        self.encoder
            .as_ref()
            .ok_or_else(|| candle_core::Error::Msg("mage vae encoder was not loaded".into()))?
            .moments(image)
    }

    pub fn encode_sample(&self, image: &Tensor, seed: u64) -> Result<Tensor> {
        let moments = self.encode_moments(image)?;
        let noise = crate::latent::normal_noise(moments.mean.dims(), seed, moments.mean.device())?
            .to_dtype(moments.mean.dtype())?;
        &moments.mean + (moments.logvar.affine(0.5, 0.)?.exp()? * noise)?
    }

    fn dct(device: &Device, dtype: DType) -> Result<Tensor> {
        let p = 16usize;
        let freqs: Vec<f32> = (0..8).map(|i| i as f32 * 8. / 7.).collect();
        let mut out = Vec::with_capacity(256 * 64);
        for y in 0..p {
            for x in 0..p {
                for fx in &freqs {
                    for fy in &freqs {
                        out.push(
                            ((x as f32 / 15.) * fx * std::f32::consts::PI).cos()
                                * ((y as f32 / 15.) * fy * std::f32::consts::PI).cos()
                                / (1. + fx * fy),
                        );
                    }
                }
            }
        }
        Tensor::from_vec(out, (1, 256, 64), device)?.to_dtype(dtype)
    }

    /// `[B,128,h,w]` raw latent -> `[B,3,h*16,w*16]` raw RGB in `[-1,1]`.
    pub fn decode(&self, latent: &Tensor) -> Result<Tensor> {
        let (b, c, h, w) = latent.dims4()?;
        if c != LATENT_CHANNELS {
            candle_core::bail!("mage vae: expected 128 latent channels, got {c}")
        }
        let cond = self.cod.forward(&latent.to_dtype(self.dtype)?)?;
        let zeros = Tensor::zeros(
            (b, 3, h * VAE_DOWNSAMPLE, w * VAE_DOWNSAMPLE),
            self.dtype,
            latent.device(),
        )?;
        let patched = self.patch_embed.forward(&zeros)?;
        let mut state = self
            .patch_fuse
            .forward(&Tensor::cat(&[patched, cond.clone()], 1)?)?;
        let t = self.time.zero(b, self.dtype, latent.device())?;
        for block in &self.blocks {
            state = block.forward(&state, &t)?;
        }
        let state_flat = state.permute((0, 2, 3, 1))?.reshape((b * h * w, 384))?;
        // y_embedder_x is feature-major: [hidden_x, position].
        let y = self
            .y_pixels
            .forward(&cond)?
            .reshape((b, 32, 256, h, w))?
            .permute((0, 3, 4, 2, 1))?
            .reshape((b * h * w, 256, 32))?;
        let rgb = Tensor::zeros((b * h * w, 256, 3), self.dtype, latent.device())?;
        let dct = Self::dct(latent.device(), self.dtype)?.broadcast_as((b * h * w, 256, 64))?;
        // CUDA matmul requires the concatenated feature tensor to be contiguous. `y` is a
        // permuted view and `dct` is broadcast, so Candle can otherwise preserve a strided layout.
        let pixel_features = Tensor::cat(&[rgb, y, dct], 2)?.contiguous()?;
        let mut pixels = self.x_embed.forward(&pixel_features)?;
        pixels = self.input_proj.forward(&pixels)?;
        // cond_embed is position-major.
        let pos_cond = self
            .cond_embed
            .forward(&state_flat)?
            .reshape((b * h * w, 256, 32))?;
        for mlp in &self.mlps {
            pixels = mlp.forward(&pixels, &pos_cond)?;
        }
        let denom = pixels
            .to_dtype(DType::F32)?
            .sqr()?
            .mean_keepdim(D::Minus1)?
            .affine(1., NORM_EPS)?
            .sqrt()?;
        let pixels = pixels
            .to_dtype(DType::F32)?
            .broadcast_div(&denom)?
            .broadcast_mul(&self.final_norm.to_dtype(DType::F32)?)?
            .to_dtype(self.dtype)?;
        self.final_linear
            .forward(&pixels)?
            .reshape((b, h, w, 16, 16, 3))?
            .permute((0, 5, 1, 3, 2, 4))?
            .reshape((b, 3, h * 16, w * 16))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn dct_uses_inclusive_zero_to_eight_frequency_ramp() {
        let got = MageVae::dct(&Device::Cpu, DType::F32)
            .unwrap()
            .to_vec3::<f32>()
            .unwrap();
        assert_eq!(got[0][0][0], 1.0);
        assert!((got[0][0][63] - 1.0 / 65.0).abs() < 1e-7);
        let inclusive = ((1f32 / 15.) * (8.0 / 7.0) * std::f32::consts::PI).cos();
        let integer_ramp = ((1f32 / 15.) * std::f32::consts::PI).cos();
        assert!((got[0][1][8] - inclusive).abs() < 1e-6);
        assert!((got[0][1][8] - integer_ramp).abs() > 1e-3);
    }
}
