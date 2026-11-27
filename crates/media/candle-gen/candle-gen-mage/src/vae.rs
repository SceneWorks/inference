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

/// `layer_norm_channels(x, None, None) · (1 + scale) + shift` for an NCHW `x` and `[B, C, 1, 1]`
/// modulation — the shared fused AdaLN (SW-6129). The channel LayerNorm is the f32
/// `mean_keepdim` / sub / `sqr` / `mean_keepdim` / `affine(1, eps)` / `sqrt` / div chain over the
/// channels-last view, cast back: `AdaLnChain::layer(NORM_EPS)` op for op on the `[B, H·W, C]`
/// tokens (the same elements, the same per-pixel order), modulated per sample in `x`'s dtype. The
/// result is laid back out as NCHW (contiguous, as the elementwise modulate produced it).
fn channel_ln_modulate(x: &Tensor, scale: &Tensor, shift: &Tensor) -> Result<Tensor> {
    let (b, c, h, w) = x.dims4()?;
    let tokens = x.permute((0, 2, 3, 1))?.reshape((b, h * w, c))?;
    let row = |t: &Tensor| t.reshape((b, 1, c));
    let spec = candle_gen::modulate::AdaLnChain::layer(NORM_EPS);
    candle_gen::modulate::adaln_chain(&tokens, &spec, &row(scale)?, &row(shift)?)?
        .reshape((b, h, w, c))?
        .permute((0, 3, 1, 2))?
        .contiguous()
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

    /// This affine channel LayerNorm over a channels-last `[B, N, C]` `x` (what
    /// `layer(x.unsqueeze(2).permute(0, 3, 1, 2)).permute(0, 2, 3, 1).squeeze(2)` computes: the f32
    /// chain, `· weight + bias` in f32, cast back) then `· (1 + scale) + shift` — the shared fused
    /// AdaLN (SW-6129), `AdaLnChain::layer(NORM_EPS)` with the f32 weight and bias, op for op.
    fn ln_modulate_tokens(&self, x: &Tensor, scale: &Tensor, shift: &Tensor) -> Result<Tensor> {
        let (w, b) = (
            self.weight.to_dtype(DType::F32)?,
            self.bias.to_dtype(DType::F32)?,
        );
        let spec = candle_gen::modulate::AdaLnChain::layer(NORM_EPS)
            .with_weight(&w)
            .with_bias(&b);
        candle_gen::modulate::adaln_chain(x, &spec, scale, shift)
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
        let h = __model_proof_output!(self.c1.forward(&self.n1.group(x)?.silu()?))?;
        x + __model_proof_output!(self.c2.forward(&self.n2.group(&h)?.silu()?))?
    }
}

/// The attention softmax: candle's fused `softmax_last_dim` for inference, or the composable
/// `softmax` (which has a backward) when the decode is differentiated (epic 2123 depth anchoring,
/// sc-24830 — the fused op is `apply_op1_no_bwd`, so the gradient would silently stop there).
type SoftmaxFn = fn(&Tensor) -> Result<Tensor>;

fn fused_softmax(t: &Tensor) -> Result<Tensor> {
    candle_nn::ops::softmax_last_dim(t)
}

fn composable_softmax(t: &Tensor) -> Result<Tensor> {
    candle_nn::ops::softmax(t, D::Minus1)
}

/// The softmax [`MageVae::decode_differentiable`] runs.
const DIFFERENTIABLE_SOFTMAX: SoftmaxFn = composable_softmax;

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

    fn forward(&self, x: &Tensor, softmax: SoftmaxFn) -> Result<Tensor> {
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
            softmax,
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
        x + __model_proof_output!(self.out.forward(&o))?
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
    fn forward(&self, z: &Tensor, softmax: SoftmaxFn) -> Result<Tensor> {
        let h = self
            .r0
            .forward(&__model_proof_output!(self.input.forward(z))?)?;
        let h = self.a0.forward(&h, softmax)?;
        let h = self.r1.forward(&h)?;
        let h = self.a1.forward(&h, softmax)?;
        let h = self.r2.forward(&h)?;
        __model_proof_output!(self.output.forward(&self.norm.group(&h)?.silu()?))
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
        let p = __model_proof_output!(self.adaln.forward(&conditioning.silu()?))?;
        let chunks = p.chunk(6, 1)?;
        let e = |i: usize| chunks[i].reshape((chunks[i].dim(0)?, 384, 1, 1));
        let h = channel_ln_modulate(x, &e(1)?, &e(0)?)?;
        let h = __model_proof_output!(self
            .c2
            .forward(&__model_proof_output!(self.c1.forward(&h))?))?
        .gelu_erf()?;
        let ca = candle_nn::ops::sigmoid(&__model_proof_output!(self
            .ca
            .forward(&h.mean_keepdim((2, 3))?))?)?;
        let h = __model_proof_output!(self.c3.forward(&h.broadcast_mul(&ca)?))?;
        let x = (x + h.broadcast_mul(&e(2)?)?)?;
        let h = channel_ln_modulate(&x, &e(4)?, &e(3)?)?;
        let h = __model_proof_output!(self
            .c5
            .forward(&__model_proof_output!(self.c4.forward(&h))?.gelu_erf()?))?;
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
        __model_proof_output!(self
            .l2
            .forward(&__model_proof_output!(self.l1.forward(&input))?.silu()?))
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
        let h =
            __model_proof_output!(self.c2.forward(&__model_proof_output!(self.c1.forward(x))?))?
                .gelu_erf()?;
        let ca = candle_nn::ops::sigmoid(&__model_proof_output!(self
            .ca
            .forward(&h.mean_keepdim((2, 3))?))?)?;
        __model_proof_output!(self.c3.forward(&h.broadcast_mul(&ca)?))
    }

    fn channel(&self, x: &Tensor) -> Result<Tensor> {
        __model_proof_output!(self
            .c5
            .forward(&__model_proof_output!(self.c4.forward(x))?.gelu_erf()?))
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
        let mut cond =
            __model_proof_output!(self.patch_cond.forward(&image.to_dtype(self.dtype)?))?;
        for block in &self.head {
            cond = block.forward(&cond)?;
        }
        cond = __model_proof_output!(self.proj_down.forward(&cond))?;
        let z = Tensor::zeros(
            (b, LATENT_CHANNELS, h / 16, w / 16),
            self.dtype,
            image.device(),
        )?;
        let z = __model_proof_output!(self.z_proj.forward(&z))?;
        let mut state =
            __model_proof_output!(self.fuse_proj.forward(&Tensor::cat(&[cond, z], 1)?))?;
        let time = self.time.zero(b, self.dtype, image.device())?;
        for block in &self.blocks {
            state = block.forward(&state, &time)?;
        }
        let packed = __model_proof_output!(self.proj_out.forward(&self.norm_out.layer(&state)?))?;
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
        let p = __model_proof_output!(self.adaln.forward(&cond.silu()?))?;
        let d = p.dim(D::Minus1)? / 3;
        let shift = p.narrow(D::Minus1, 0, d)?;
        let scale = p.narrow(D::Minus1, d, d)?;
        let gate = p.narrow(D::Minus1, 2 * d, d)?;
        let h = self.norm.ln_modulate_tokens(x, &scale, &shift)?;
        x + __model_proof_output!(self
            .l2
            .forward(&__model_proof_output!(self.l1.forward(&h))?.silu()?))?
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

    /// The decoder half only, at `dtype` (no `student.dconv_encoder.*` encoder) — what the
    /// depth-anchoring x0 decoder needs (sc-24830).
    pub fn load_dtype(dir: &Path, device: &Device, dtype: DType) -> Result<Self> {
        Self::load_inner(dir, device, false, dtype)
    }

    /// Whether the encoder half is loaded.
    pub fn has_encoder(&self) -> bool {
        self.encoder.is_some()
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
            .moments(&image.to_dtype(self.dtype)?)
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
        self.decode_with(latent, fused_softmax)
    }

    /// [`decode`](Self::decode) with every op differentiable (the composable attention softmax), for
    /// the depth-anchoring training loss (epic 2123, sc-24830): the gradient flows from the pixels
    /// back to `latent`. Same values as `decode` up to softmax rounding.
    pub fn decode_differentiable(&self, latent: &Tensor) -> Result<Tensor> {
        // A gradient path: pinned to the materialized attention (SW-6130; the fused kernel has no
        // backward and would not call the composable softmax).
        candle_gen::fused_attention::with_materialized_attention(|| {
            self.decode_with(latent, DIFFERENTIABLE_SOFTMAX)
        })
    }

    fn decode_with(&self, latent: &Tensor, softmax: SoftmaxFn) -> Result<Tensor> {
        let (b, c, h, w) = latent.dims4()?;
        if c != LATENT_CHANNELS {
            candle_core::bail!("mage vae: expected 128 latent channels, got {c}")
        }
        let cond = self.cod.forward(&latent.to_dtype(self.dtype)?, softmax)?;
        let zeros = Tensor::zeros(
            (b, 3, h * VAE_DOWNSAMPLE, w * VAE_DOWNSAMPLE),
            self.dtype,
            latent.device(),
        )?;
        let patched = __model_proof_output!(self.patch_embed.forward(&zeros))?;
        let mut state = __model_proof_output!(self
            .patch_fuse
            .forward(&Tensor::cat(&[patched, cond.clone()], 1)?))?;
        let t = self.time.zero(b, self.dtype, latent.device())?;
        for block in &self.blocks {
            state = block.forward(&state, &t)?;
        }
        let state_flat = state.permute((0, 2, 3, 1))?.reshape((b * h * w, 384))?;
        // y_embedder_x is feature-major: [hidden_x, position].
        let y = __model_proof_output!(self.y_pixels.forward(&cond))?
            .reshape((b, 32, 256, h, w))?
            .permute((0, 3, 4, 2, 1))?
            .reshape((b * h * w, 256, 32))?;
        let rgb = Tensor::zeros((b * h * w, 256, 3), self.dtype, latent.device())?;
        let dct = Self::dct(latent.device(), self.dtype)?.broadcast_as((b * h * w, 256, 64))?;
        // CUDA matmul requires the concatenated feature tensor to be contiguous. `y` is a
        // permuted view and `dct` is broadcast, so Candle can otherwise preserve a strided layout.
        let pixel_features = Tensor::cat(&[rgb, y, dct], 2)?.contiguous()?;
        let mut pixels = __model_proof_output!(self.x_embed.forward(&pixel_features))?;
        pixels = __model_proof_output!(self.input_proj.forward(&pixels))?;
        // cond_embed is position-major.
        let pos_cond = __model_proof_output!(self.cond_embed.forward(&state_flat))?.reshape((
            b * h * w,
            256,
            32,
        ))?;
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
        __model_proof_output!(self.final_linear.forward(&pixels))?
            .reshape((b, h, w, 16, 16, 3))?
            .permute((0, 5, 1, 3, 2, 4))?
            .reshape((b, 3, h * 16, w * 16))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn rand(shape: &[usize], seed: u64) -> Tensor {
        let n: usize = shape.iter().product();
        let mut x = seed.wrapping_mul(0x9E37_79B9_7F4A_7C15) | 1;
        let v: Vec<f32> = (0..n)
            .map(|_| {
                x = x
                    .wrapping_mul(6364136223846793005)
                    .wrapping_add(1442695040888963407);
                ((x >> 40) as f32) / (1u64 << 24) as f32 - 0.5
            })
            .collect();
        Tensor::from_vec(v, shape, &Device::Cpu).unwrap()
    }

    fn tiny_attention(c: usize) -> LocalAttention {
        let conv1 = |seed| {
            Conv2d::new(
                rand(&[c, c, 1, 1], seed),
                Some(rand(&[c], seed + 1)),
                Conv2dConfig::default(),
            )
        };
        LocalAttention {
            norm: AffineNorm {
                weight: Tensor::ones(c, DType::F32, &Device::Cpu).unwrap(),
                bias: rand(&[c], 3),
            },
            q: conv1(10),
            k: conv1(20),
            v: conv1(30),
            out: conv1(40),
        }
    }

    /// sc-24830: the differentiable decode's attention softmax matches the fused forward and its
    /// autograd gradient is the true gradient (a central finite difference along a random
    /// direction agrees). Mutation: set `DIFFERENTIABLE_SOFTMAX = fused_softmax` (no backward) ⇒
    /// the probability path drops out of the gradient ⇒ the directional derivatives disagree ⇒ red.
    #[test]
    #[ignore = "model inference; explicit model workflow only"]
    fn differentiable_attention_matches_fused_and_has_the_true_gradient() {
        let attn = tiny_attention(32);
        let x = rand(&[1, 32, 3, 5], 7);
        let fused = attn.forward(&x, fused_softmax).unwrap();
        let comp = attn.forward(&x, DIFFERENTIABLE_SOFTMAX).unwrap();
        let d = (&fused - &comp)
            .unwrap()
            .abs()
            .unwrap()
            .max_all()
            .unwrap()
            .to_scalar::<f32>()
            .unwrap();
        assert!(d < 1e-5, "{d}");
        let loss = |t: &Tensor| -> Tensor {
            attn.forward(t, DIFFERENTIABLE_SOFTMAX)
                .unwrap()
                .sqr()
                .unwrap()
                .sum_all()
                .unwrap()
        };
        let v = candle_core::Var::from_tensor(&x).unwrap();
        let g = loss(v.as_tensor()).backward().unwrap();
        let g = g.get(v.as_tensor()).unwrap().clone();
        let dir = rand(&[1, 32, 3, 5], 99);
        let analytic = (&g * &dir)
            .unwrap()
            .sum_all()
            .unwrap()
            .to_scalar::<f32>()
            .unwrap() as f64;
        let eps = 1e-2;
        let plus = loss(&(&x + (&dir * eps).unwrap()).unwrap())
            .to_scalar::<f32>()
            .unwrap() as f64;
        let minus = loss(&(&x - (&dir * eps).unwrap()).unwrap())
            .to_scalar::<f32>()
            .unwrap() as f64;
        let numeric = (plus - minus) / (2.0 * eps);
        assert!(
            (analytic - numeric).abs() <= 2e-2 * numeric.abs().max(1e-3),
            "analytic {analytic} vs numeric {numeric}"
        );

        #[cfg(feature = "model-inference-proof")]
        println!("\nMODEL_INFERENCE_BODY_COMPLETED vae::tests::differentiable_attention_matches_fused_and_has_the_true_gradient");
    }

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

/// SW-6129: the Mage-VAE decoder AdaLN sites are the shared fused composable-chain AdaLN.
#[cfg(test)]
mod sw6129_adaln_route_tests {
    use super::*;

    /// Deterministic xorshift values in `[-amp, amp) + bias` (no device RNG).
    fn values(dims: &[usize], seed: u64, amp: f32, bias: f32, device: &Device) -> Tensor {
        let n: usize = dims.iter().product();
        let mut s = seed | 1;
        let v: Vec<f32> = (0..n)
            .map(|_| {
                s ^= s << 13;
                s ^= s >> 7;
                s ^= s << 17;
                ((s >> 11) as f64 / (1u64 << 53) as f64 * 2.0 - 1.0) as f32 * amp + bias
            })
            .collect();
        Tensor::from_vec(v, dims, device).unwrap()
    }

    fn bits(t: &Tensor) -> Vec<u32> {
        t.to_dtype(DType::F32)
            .unwrap()
            .flatten_all()
            .unwrap()
            .to_vec1::<f32>()
            .unwrap()
            .into_iter()
            .map(f32::to_bits)
            .collect()
    }

    /// Runs `run` with the fused AdaLN switch on, then off; asserts `calls` fused calls on, the
    /// same number of `disabled` references off, and bitwise-identical outputs; prints `what`.
    #[cfg(feature = "cuda")]
    fn assert_fused_and_bit_identical(what: &str, calls: u64, run: &dyn Fn() -> Tensor) {
        use candle_gen::fused_path::media_fused_policy_guard;
        use candle_gen::modulate::ADALN_PATH;
        let _media = media_fused_policy_guard(Some(true));
        let _kernel = ADALN_PATH.guard(Some(true));
        ADALN_PATH.set_enabled(Some(true));
        let start = ADALN_PATH.thread_counts();
        let fused = run();
        let on = ADALN_PATH.thread_counts().since(&start);
        assert_eq!((on.fused, on.reference_total()), (calls, 0), "{what}: {on}");
        ADALN_PATH.set_enabled(Some(false));
        let start = ADALN_PATH.thread_counts();
        let reference = run();
        let off = ADALN_PATH.thread_counts().since(&start);
        assert_eq!(off.reference_for("disabled"), calls, "{what}: {off}");
        ADALN_PATH.set_enabled(Some(true));
        let (f, r) = (bits(&fused), bits(&reference));
        let mismatches = f.iter().zip(&r).filter(|(a, b)| a != b).count();
        println!(
            "{what}: {mismatches} of {} elements differ; on {on}, off {off}",
            f.len()
        );
        assert_eq!(mismatches, 0, "{what}");
    }

    /// CPU: the DConv-block channel AdaLN and the pixel-MLP token AdaLN are bit-identical to the
    /// former `layer_norm_channels` + modulate chains, f32 / bf16 / f16, the decoder's 384 channels.
    #[test]
    #[ignore = "model inference; explicit model-proof workflow"]
    fn channel_and_token_adaln_match_the_former_chains() {
        let dev = Device::Cpu;
        let c = 384;
        let x32 = values(&[2, c, 6, 5], 3, 3.0, 0.4, &dev);
        let t32 = values(&[2, 7, c], 4, 3.0, 0.4, &dev);
        let norm = AffineNorm {
            weight: values(&[c], 8, 0.5, 1.0, &dev),
            bias: values(&[c], 9, 0.5, 0.0, &dev),
        };
        for dtype in [DType::F32, DType::BF16, DType::F16] {
            let x = x32.to_dtype(dtype).unwrap();
            let e = |seed| {
                values(&[2, c, 1, 1], seed, 0.5, 0.0, &dev)
                    .to_dtype(dtype)
                    .unwrap()
            };
            let (scale, shift) = (e(5), e(6));
            let want = candle_gen::modulate::modulate(
                &layer_norm_channels(&x, None, None).unwrap(),
                &scale,
                &shift,
            )
            .unwrap();
            let got = channel_ln_modulate(&x, &scale, &shift).unwrap();
            assert_eq!(got.dims(), want.dims());
            assert_eq!(bits(&got), bits(&want), "channel {dtype:?}");

            let t = t32.to_dtype(dtype).unwrap();
            let n = AffineNorm {
                weight: norm.weight.to_dtype(dtype).unwrap(),
                bias: norm.bias.to_dtype(dtype).unwrap(),
            };
            let m = |seed| {
                values(&[2, 7, c], seed, 0.5, 0.0, &dev)
                    .to_dtype(dtype)
                    .unwrap()
            };
            let (scale, shift) = (m(10), m(11));
            let former = n
                .layer(&t.unsqueeze(2).unwrap().permute((0, 3, 1, 2)).unwrap())
                .unwrap()
                .permute((0, 2, 3, 1))
                .unwrap()
                .squeeze(2)
                .unwrap();
            let want = candle_gen::modulate::modulate(&former, &scale, &shift).unwrap();
            let got = n.ln_modulate_tokens(&t, &scale, &shift).unwrap();
            assert_eq!(bits(&got), bits(&want), "tokens {dtype:?}");
        }

        #[cfg(feature = "model-inference-proof")]
        println!("\nMODEL_INFERENCE_BODY_COMPLETED vae::sw6129_adaln_route_tests::channel_and_token_adaln_match_the_former_chains");
    }

    /// SW-6129 CUDA smoke: the Mage-VAE DConv channel AdaLN (bf16 `[1, 384, 64, 64]`) and the
    /// pixel-MLP token AdaLN (bf16 `[1, 4096, 384]`, f32 weight / bias) are fused (2 calls) and
    /// bit-identical to the switch-off chain.
    #[cfg(feature = "cuda")]
    #[test]
    #[ignore = "CUDA compute: run by the validation workflow, not CI"]
    fn cuda_bf16_adaln_route_is_fused_and_bit_identical() {
        let dev = Device::new_cuda(0).unwrap();
        let c = 384;
        let bf = |t: Tensor| t.to_dtype(DType::BF16).unwrap();
        let x = bf(values(&[1, c, 64, 64], 11, 3.0, 0.3, &dev));
        let e = (
            bf(values(&[1, c, 1, 1], 12, 0.5, 0.0, &dev)),
            bf(values(&[1, c, 1, 1], 13, 0.5, 0.0, &dev)),
        );
        let t = bf(values(&[1, 4096, c], 14, 3.0, 0.3, &dev));
        let m = (
            bf(values(&[1, 4096, c], 15, 0.5, 0.0, &dev)),
            bf(values(&[1, 4096, c], 16, 0.5, 0.0, &dev)),
        );
        let norm = AffineNorm {
            weight: bf(values(&[c], 17, 0.5, 1.0, &dev)),
            bias: bf(values(&[c], 18, 0.5, 0.0, &dev)),
        };
        assert_fused_and_bit_identical(
            "mage-vae channel + token AdaLN (bf16, [1,384,64,64] + [1,4096,384])",
            2,
            &|| {
                let a = channel_ln_modulate(&x, &e.0, &e.1)
                    .unwrap()
                    .flatten_all()
                    .unwrap();
                let b = norm
                    .ln_modulate_tokens(&t, &m.0, &m.1)
                    .unwrap()
                    .flatten_all()
                    .unwrap();
                Tensor::cat(&[&a, &b], 0).unwrap()
            },
        );
    }
}
