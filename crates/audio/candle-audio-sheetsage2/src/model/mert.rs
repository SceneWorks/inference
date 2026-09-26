//! The MERT-v2 encoder (`modeling_mert2.py` at `MERT@d8ba1c7`): ConvNeXt-v2 + GRN subsampler to
//! 25 Hz, then Conformer blocks (macaron FFN, rotate-half RoPE self-attention, GLU + depthwise
//! convolution module), with SheetSage2's rank-r LoRA adapters merged into the attention
//! projections in float32 on the CPU **before** anything moves to the model device.
//!
//! Tensors are `[batch, time, channels]` throughout. Convolutions over time are written as shifted
//! products in that layout (depthwise) or as a reshape + matmul (the kernel-2 resampler), so no
//! layout transposes are needed. Attention is computed in query chunks: a materialized
//! `16 × 7500²` score tensor would be 3.6 GB per layer.

use candle_audio::candle_core::{DType, Device, Tensor, D};

use super::config::MertConfig;
use super::frontend::MelFrontend;
use super::weights::Weights;
use crate::Error;

/// Queries per attention chunk.
const ATTENTION_QUERY_CHUNK: usize = 512;

/// A linear layer with its weight stored transposed (`[in, out]`) for row-major matmul.
#[derive(Clone, Debug)]
pub(crate) struct Linear {
    weight_t: Tensor,
    bias: Option<Tensor>,
}

impl Linear {
    pub(crate) fn new(
        weight: &Tensor,
        bias: Option<Tensor>,
        device: &Device,
    ) -> Result<Self, Error> {
        Ok(Self {
            weight_t: weight.t()?.contiguous()?.to_device(device)?,
            bias: bias.map(|b| b.to_device(device)).transpose()?,
        })
    }

    pub(crate) fn forward(&self, x: &Tensor) -> Result<Tensor, Error> {
        let dims = x.dims().to_vec();
        let (rows, features) = (
            dims[..dims.len() - 1].iter().product::<usize>(),
            dims[dims.len() - 1],
        );
        let flat = x.contiguous()?.reshape((rows, features))?;
        let mut out = flat.matmul(&self.weight_t)?;
        if let Some(b) = &self.bias {
            out = out.broadcast_add(b)?;
        }
        let mut shape = dims;
        let last = shape.len() - 1;
        shape[last] = self.weight_t.dim(1)?;
        Ok(out.reshape(shape)?)
    }

    pub(crate) fn bytes(&self) -> usize {
        self.weight_t.elem_count() * 4 + self.bias.as_ref().map_or(0, |b| b.elem_count() * 4)
    }
}

/// LayerNorm over the last dimension.
#[derive(Clone, Debug)]
pub(crate) struct LayerNorm {
    weight: Tensor,
    bias: Tensor,
    eps: f32,
}

impl LayerNorm {
    pub(crate) fn take(
        w: &mut Weights,
        prefix: &str,
        dim: usize,
        eps: f64,
        device: &Device,
    ) -> Result<Self, Error> {
        Ok(Self {
            weight: w
                .take(&format!("{prefix}.weight"), &[dim])?
                .to_device(device)?,
            bias: w
                .take(&format!("{prefix}.bias"), &[dim])?
                .to_device(device)?,
            eps: eps as f32,
        })
    }

    pub(crate) fn forward(&self, x: &Tensor) -> Result<Tensor, Error> {
        Ok(candle_nn::ops::layer_norm(
            &x.contiguous()?,
            &self.weight,
            &self.bias,
            self.eps,
        )?)
    }

    fn bytes(&self) -> usize {
        (self.weight.elem_count() + self.bias.elem_count()) * 4
    }
}

/// A depthwise convolution over time (`padding = (k-1)/2`), as `k` shifted products.
#[derive(Clone, Debug)]
struct DepthwiseConv {
    /// `k` tensors of shape `[channels]`.
    taps: Vec<Tensor>,
    bias: Option<Tensor>,
}

impl DepthwiseConv {
    fn take(
        w: &mut Weights,
        prefix: &str,
        channels: usize,
        kernel: usize,
        bias: bool,
        device: &Device,
    ) -> Result<Self, Error> {
        let weight = w.take(&format!("{prefix}.weight"), &[channels, 1, kernel])?;
        let taps = (0..kernel)
            .map(|j| {
                weight
                    .narrow(2, j, 1)?
                    .reshape(channels)?
                    .contiguous()?
                    .to_device(device)
            })
            .collect::<Result<Vec<_>, _>>()?;
        let bias = if bias {
            Some(
                w.take(&format!("{prefix}.bias"), &[channels])?
                    .to_device(device)?,
            )
        } else {
            None
        };
        Ok(Self { taps, bias })
    }

    fn forward(&self, x: &Tensor) -> Result<Tensor, Error> {
        let (_, t, _) = x.dims3()?;
        let pad = (self.taps.len() - 1) / 2;
        let padded = x.pad_with_zeros(1, pad, pad)?;
        let mut acc: Option<Tensor> = None;
        for (j, tap) in self.taps.iter().enumerate() {
            let term = padded.narrow(1, j, t)?.broadcast_mul(tap)?;
            acc = Some(match acc {
                None => term,
                Some(a) => (a + term)?,
            });
        }
        let mut out = acc.expect("kernel is non-empty");
        if let Some(b) = &self.bias {
            out = out.broadcast_add(b)?;
        }
        Ok(out)
    }

    fn bytes(&self) -> usize {
        self.taps.iter().map(|t| t.elem_count() * 4).sum::<usize>()
            + self.bias.as_ref().map_or(0, |b| b.elem_count() * 4)
    }
}

/// GlobalResponseNorm (ConvNeXt-v2): L2 over **time**, normalized by its mean over channels.
#[derive(Clone, Debug)]
struct Grn {
    weight: Tensor,
    bias: Tensor,
}

impl Grn {
    fn forward(&self, x: &Tensor) -> Result<Tensor, Error> {
        // torch.norm(p=2, dim=1): reduced in float64 so a 30,000-frame sum keeps its precision.
        let magnitude = x
            .to_dtype(DType::F64)?
            .sqr()?
            .sum_keepdim(1)?
            .sqrt()?
            .to_dtype(DType::F32)?;
        let mean = magnitude.mean_keepdim(D::Minus1)?;
        let normalized = magnitude.broadcast_div(&(mean + 1e-6)?)?;
        let scaled = x.broadcast_mul(&normalized)?.broadcast_mul(&self.weight)?;
        Ok(((scaled.broadcast_add(&self.bias))? + x)?)
    }
}

#[derive(Clone, Debug)]
struct ConvNextLayer {
    depthwise: DepthwiseConv,
    norm: LayerNorm,
    up: Linear,
    grn: Grn,
    down: Linear,
}

impl ConvNextLayer {
    fn forward(&self, x: &Tensor) -> Result<Tensor, Error> {
        let h = self.depthwise.forward(x)?;
        let h = self.norm.forward(&h)?;
        let h = self.up.forward(&h)?.gelu_erf()?;
        let h = self.grn.forward(&h)?;
        let h = self.down.forward(&h)?;
        Ok((x + h)?)
    }

    fn bytes(&self) -> usize {
        self.depthwise.bytes()
            + self.norm.bytes()
            + self.up.bytes()
            + self.down.bytes()
            + (self.grn.weight.elem_count() + self.grn.bias.elem_count()) * 4
    }
}

/// A kernel-2 strided resampling convolution (`Conv1d(in, out, 2, stride)`), after a LayerNorm.
#[derive(Clone, Debug)]
struct Resampler {
    norm: LayerNorm,
    /// `[2 * in, out]`: row `k * in + c` is tap `k`, input channel `c`.
    weight: Tensor,
    bias: Tensor,
    stride: usize,
}

impl Resampler {
    fn forward(&self, x: &Tensor) -> Result<Tensor, Error> {
        let x = self.norm.forward(x)?;
        let (b, t, c) = x.dims3()?;
        if t < 2 {
            return Err(Error::Request("too few frames for the resampler".into()));
        }
        let frames = (t - 2) / self.stride + 1;
        let pairs = match self.stride {
            // Frames 2i and 2i+1 are adjacent: fold them into one row.
            2 => x.narrow(1, 0, frames * 2)?.reshape((b, frames, 2 * c))?,
            1 => Tensor::cat(&[x.narrow(1, 0, frames)?, x.narrow(1, 1, frames)?], 2)?,
            s => {
                return Err(Error::Config(format!(
                    "resampler stride {s} is not used by MERT2"
                )))
            }
        };
        let out = pairs
            .reshape((b * frames, 2 * c))?
            .matmul(&self.weight)?
            .broadcast_add(&self.bias)?;
        Ok(out.reshape((b, frames, self.weight.dim(1)?))?)
    }
}

#[derive(Clone, Debug)]
struct Stage {
    resampler: Option<Resampler>,
    layers: Vec<ConvNextLayer>,
}

#[derive(Clone, Debug)]
struct Attention {
    q: Linear,
    k: Linear,
    v: Linear,
    out: Linear,
    heads: usize,
}

#[derive(Clone, Debug)]
struct ConformerBlock {
    ffn1_norm: LayerNorm,
    ffn1: (Linear, Linear),
    attn_norm: LayerNorm,
    attn: Attention,
    conv_norm: LayerNorm,
    conv_pointwise1: Linear,
    conv_depthwise: DepthwiseConv,
    conv_inner_norm: LayerNorm,
    conv_pointwise2: Linear,
    ffn2_norm: LayerNorm,
    ffn2: (Linear, Linear),
    final_norm: LayerNorm,
}

fn rotate_half(x: &Tensor) -> Result<Tensor, Error> {
    let d = x.dim(D::Minus1)?;
    let first = x.narrow(D::Minus1, 0, d / 2)?;
    let second = x.narrow(D::Minus1, d / 2, d / 2)?;
    Ok(Tensor::cat(&[second.neg()?, first], D::Minus1)?)
}

impl Attention {
    fn forward(&self, x: &Tensor, cos: &Tensor, sin: &Tensor) -> Result<Tensor, Error> {
        let (b, t, width) = x.dims3()?;
        let head_dim = width / self.heads;
        let shape = (b, t, self.heads, head_dim);
        let rope = |y: Tensor| -> Result<Tensor, Error> {
            let y = y.reshape(shape)?;
            Ok((y.broadcast_mul(cos)? + rotate_half(&y)?.broadcast_mul(sin)?)?)
        };
        // [b, h, t, d]
        let q = rope(self.q.forward(x)?)?.transpose(1, 2)?.contiguous()?;
        let k = rope(self.k.forward(x)?)?.transpose(1, 2)?.contiguous()?;
        let v = self
            .v
            .forward(x)?
            .reshape(shape)?
            .transpose(1, 2)?
            .contiguous()?;
        let k_t = k.transpose(2, 3)?.contiguous()?;
        let scale = 1.0 / (head_dim as f64).sqrt();
        let mut chunks = Vec::new();
        let mut start = 0;
        while start < t {
            let len = ATTENTION_QUERY_CHUNK.min(t - start);
            let q_chunk = q.narrow(2, start, len)?;
            let scores = (q_chunk.matmul(&k_t)? * scale)?;
            let probs = candle_nn::ops::softmax_last_dim(&scores)?;
            chunks.push(probs.matmul(&v)?);
            start += len;
        }
        let attended = Tensor::cat(&chunks, 2)?
            .transpose(1, 2)?
            .contiguous()?
            .reshape((b, t, width))?;
        self.out.forward(&attended)
    }
}

impl ConformerBlock {
    fn forward(&self, x: &Tensor, cos: &Tensor, sin: &Tensor) -> Result<Tensor, Error> {
        let ffn = |p: &(Linear, Linear), h: &Tensor| -> Result<Tensor, Error> {
            p.1.forward(&p.0.forward(h)?.gelu_erf()?)
        };
        let h = (x + (ffn(&self.ffn1, &self.ffn1_norm.forward(x)?)? * 0.5)?)?;
        let h = (self.attn.forward(&self.attn_norm.forward(&h)?, cos, sin)? + &h)?;
        let c = self.conv_pointwise1.forward(&self.conv_norm.forward(&h)?)?;
        let width = c.dim(D::Minus1)? / 2;
        let glu = c
            .narrow(D::Minus1, 0, width)?
            .mul(&candle_nn::ops::sigmoid(&c.narrow(
                D::Minus1,
                width,
                width,
            )?)?)?;
        let c = self.conv_depthwise.forward(&glu)?;
        let c = self.conv_inner_norm.forward(&c)?.gelu_erf()?;
        let c = self.conv_pointwise2.forward(&c)?;
        let h = (c + &h)?;
        let h = (&h + (ffn(&self.ffn2, &self.ffn2_norm.forward(&h)?)? * 0.5)?)?;
        self.final_norm.forward(&h)
    }
}

/// The merged MERT2 encoder.
#[derive(Clone, Debug)]
pub struct MertEncoder {
    config: MertConfig,
    frontend: MelFrontend,
    stages: Vec<Stage>,
    blocks: Vec<ConformerBlock>,
    device: Device,
    bytes: usize,
}

/// Encoder outputs.
#[derive(Clone, Debug)]
pub struct EncoderStates {
    /// The normalized mel frames `[1, frames, n_mels]`.
    pub mel: Tensor,
    /// The subsampler output (layer-mix input 0) `[1, t, hidden]`.
    pub input_hidden: Tensor,
    /// Every Conformer block's output (only when requested).
    pub blocks: Vec<Tensor>,
}

impl MertEncoder {
    /// Build the encoder from the MERT parent checkpoint and SheetSage2's LoRA adapters (taken from
    /// `adapters`: `adapter.layers.{i}.attn.{query,key,value,out}_proj.lora_{A,B}.weight`). The
    /// merge `W += (B @ A) · alpha / rank` runs in float32 on the CPU before the tensors move to
    /// `device`.
    pub fn load(
        config: &MertConfig,
        parent: &mut Weights,
        adapters: &mut Weights,
        lora_rank: usize,
        lora_alpha: f64,
        device: &Device,
    ) -> Result<Self, Error> {
        let n_bins = config.n_fft / 2 + 1;
        let frontend = MelFrontend::new(
            &parent.take("feature_extractor.spectrogram.window", &[config.win_length])?,
            &parent.take(
                "feature_extractor.mel_scale.fb",
                &[n_bins, config.num_mel_bins],
            )?,
            &parent.take("feature_extractor.mel_mean", &[config.num_mel_bins])?,
            &parent.take("feature_extractor.mel_std", &[config.num_mel_bins])?,
            config.n_fft,
            config.hop_length,
            device,
        )?;
        let eps = config.subsampling_layer_norm_eps;
        let channels = [
            config.num_mel_bins,
            config.subsampling_channels[0],
            config.subsampling_channels[1],
            config.subsampling_channels[2],
        ];
        let mut bytes = 0usize;
        let mut stages = Vec::new();
        for i in 0..3 {
            let (cin, cout, stride) = (channels[i], channels[i + 1], [1, 2, 2][i]);
            let prefix = format!("subsampling_module.{i}");
            let resampler = if cin != cout || stride > 1 {
                let norm = LayerNorm::take(
                    parent,
                    &format!("{prefix}.resampling_layer.0"),
                    cin,
                    eps,
                    device,
                )?;
                let w = parent.take(
                    &format!("{prefix}.resampling_layer.2.weight"),
                    &[cout, cin, 2],
                )?;
                // [out, in, 2] → [2, in, out] → [2*in, out]
                let folded = w
                    .permute((2, 1, 0))?
                    .contiguous()?
                    .reshape((2 * cin, cout))?;
                let bias = parent.take(&format!("{prefix}.resampling_layer.2.bias"), &[cout])?;
                bytes += norm.bytes() + (folded.elem_count() + bias.elem_count()) * 4;
                Some(Resampler {
                    norm,
                    weight: folded.to_device(device)?,
                    bias: bias.to_device(device)?,
                    stride,
                })
            } else {
                None
            };
            let mut layers = Vec::new();
            for j in 0..config.subsampling_depths[i] {
                let p = format!("{prefix}.convnext_layers.{j}");
                let depthwise = DepthwiseConv::take(
                    parent,
                    &format!("{p}.depthwise_block.1"),
                    cout,
                    7,
                    true,
                    device,
                )?;
                let norm =
                    LayerNorm::take(parent, &format!("{p}.pointwise_block.0"), cout, eps, device)?;
                let up = Linear::new(
                    &parent.take(&format!("{p}.pointwise_block.1.weight"), &[4 * cout, cout])?,
                    Some(parent.take(&format!("{p}.pointwise_block.1.bias"), &[4 * cout])?),
                    device,
                )?;
                let grn = Grn {
                    weight: parent
                        .take(&format!("{p}.pointwise_block.3.weight"), &[1, 1, 4 * cout])?
                        .to_device(device)?,
                    bias: parent
                        .take(&format!("{p}.pointwise_block.3.bias"), &[1, 1, 4 * cout])?
                        .to_device(device)?,
                };
                let down = Linear::new(
                    &parent.take(&format!("{p}.pointwise_block.4.weight"), &[cout, 4 * cout])?,
                    Some(parent.take(&format!("{p}.pointwise_block.4.bias"), &[cout])?),
                    device,
                )?;
                let layer = ConvNextLayer {
                    depthwise,
                    norm,
                    up,
                    grn,
                    down,
                };
                bytes += layer.bytes();
                layers.push(layer);
            }
            stages.push(Stage { resampler, layers });
        }

        let (h, f) = (config.hidden_size, config.intermediate_size);
        let leps = config.layer_norm_eps;
        let scale = lora_alpha / lora_rank as f64;
        let mut blocks = Vec::new();
        for i in 0..config.num_hidden_layers {
            let p = format!("layers.{i}");
            let linear = |w: &mut Weights,
                          name: &str,
                          out: usize,
                          inp: usize,
                          bias: bool|
             -> Result<Linear, Error> {
                let weight = w.take(&format!("{p}.{name}.weight"), &[out, inp])?;
                let b = if bias {
                    Some(w.take(&format!("{p}.{name}.bias"), &[out])?)
                } else {
                    None
                };
                Linear::new(&weight, b, device)
            };
            let mut projection = |name: &str| -> Result<Linear, Error> {
                let weight = parent.take(&format!("{p}.attn.{name}.weight"), &[h, h])?;
                let a = adapters.take(
                    &format!("adapter.layers.{i}.attn.{name}.lora_A.weight"),
                    &[lora_rank, h],
                )?;
                let b = adapters.take(
                    &format!("adapter.layers.{i}.attn.{name}.lora_B.weight"),
                    &[h, lora_rank],
                )?;
                // Upstream merge_lora: in float32 on the CPU, before any cast or device move.
                let merged = (weight + (b.matmul(&a)? * scale)?)?;
                let bias = parent.take(&format!("{p}.attn.{name}.bias"), &[h])?;
                Linear::new(&merged, Some(bias), device)
            };
            let attn = Attention {
                q: projection("query_proj")?,
                k: projection("key_proj")?,
                v: projection("value_proj")?,
                out: projection("out_proj")?,
                heads: config.num_attention_heads,
            };
            let pointwise1 = parent.take(
                &format!("{p}.conv_module.conv_block.1.weight"),
                &[2 * h, h, 1],
            )?;
            let pointwise2 =
                parent.take(&format!("{p}.conv_module.conv_block.6.weight"), &[h, h, 1])?;
            let block = ConformerBlock {
                ffn1_norm: LayerNorm::take(
                    parent,
                    &format!("{p}.ffn1_layer_norm"),
                    h,
                    leps,
                    device,
                )?,
                ffn1: (
                    linear(parent, "ffn1.w_1", f, h, true)?,
                    linear(parent, "ffn1.w_2", h, f, true)?,
                ),
                attn_norm: LayerNorm::take(
                    parent,
                    &format!("{p}.attn_layer_norm"),
                    h,
                    leps,
                    device,
                )?,
                attn,
                conv_norm: LayerNorm::take(
                    parent,
                    &format!("{p}.conv_module.layer_norm"),
                    h,
                    leps,
                    device,
                )?,
                conv_pointwise1: Linear::new(&pointwise1.reshape((2 * h, h))?, None, device)?,
                conv_depthwise: DepthwiseConv::take(
                    parent,
                    &format!("{p}.conv_module.conv_block.3"),
                    h,
                    config.conv_depthwise_kernel_size,
                    false,
                    device,
                )?,
                conv_inner_norm: LayerNorm::take(
                    parent,
                    &format!("{p}.conv_module.conv_block.4.1"),
                    h,
                    leps,
                    device,
                )?,
                conv_pointwise2: Linear::new(&pointwise2.reshape((h, h))?, None, device)?,
                ffn2_norm: LayerNorm::take(
                    parent,
                    &format!("{p}.ffn2_layer_norm"),
                    h,
                    leps,
                    device,
                )?,
                ffn2: (
                    linear(parent, "ffn2.w_1", f, h, true)?,
                    linear(parent, "ffn2.w_2", h, f, true)?,
                ),
                final_norm: LayerNorm::take(
                    parent,
                    &format!("{p}.final_layer_norm"),
                    h,
                    leps,
                    device,
                )?,
            };
            bytes += block_bytes(&block);
            blocks.push(block);
        }
        Ok(Self {
            config: config.clone(),
            frontend,
            stages,
            blocks,
            device: device.clone(),
            bytes,
        })
    }

    /// Parameter bytes held on the device.
    pub fn bytes(&self) -> usize {
        self.bytes
    }

    /// The architecture.
    pub fn config(&self) -> &MertConfig {
        &self.config
    }

    /// RoPE tables `[1, t, 1, head_dim]` (float32, as upstream keeps `inv_freq` in float32).
    fn rope(&self, t: usize) -> Result<(Tensor, Tensor), Error> {
        let d = self.config.head_dim();
        let base = self.config.rotary_embedding_base as f32;
        let inv: Vec<f32> = (0..d)
            .step_by(2)
            .map(|i| 1.0 / base.powf(i as f32 / d as f32))
            .collect();
        let mut cos = Vec::with_capacity(t * d);
        let mut sin = Vec::with_capacity(t * d);
        for pos in 0..t {
            // angles = cat(freqs, freqs)
            for _ in 0..2 {
                for &f in &inv {
                    let angle = pos as f32 * f;
                    cos.push(angle.cos());
                    sin.push(angle.sin());
                }
            }
        }
        let shape = (1, t, 1, d);
        Ok((
            Tensor::from_vec(cos, shape, &self.device)?,
            Tensor::from_vec(sin, shape, &self.device)?,
        ))
    }

    /// Encode one window, calling `each_state(index, state)` for the subsampler output (index 0)
    /// and every block output (1..=layers) — the layer mix consumes them one at a time so no
    /// 25-state stack is ever held. Returns the mel frames.
    pub fn forward_states(
        &self,
        samples: &[f32],
        mut each_state: impl FnMut(usize, &Tensor) -> Result<(), Error>,
    ) -> Result<Tensor, Error> {
        let mel = self.frontend.forward(samples)?;
        let mut hidden = mel.clone();
        for stage in &self.stages {
            if let Some(r) = &stage.resampler {
                hidden = r.forward(&hidden)?;
            }
            for layer in &stage.layers {
                hidden = layer.forward(&hidden)?;
            }
        }
        each_state(0, &hidden)?;
        let (cos, sin) = self.rope(hidden.dim(1)?)?;
        for (i, block) in self.blocks.iter().enumerate() {
            hidden = block.forward(&hidden, &cos, &sin)?;
            each_state(i + 1, &hidden)?;
        }
        Ok(mel)
    }

    /// Encode and keep every state (parity tests; ~25 × frames × hidden floats).
    pub fn forward_all(&self, samples: &[f32]) -> Result<EncoderStates, Error> {
        let mut states = Vec::new();
        let mel = self.forward_states(samples, |_, s| {
            states.push(s.clone());
            Ok(())
        })?;
        let input_hidden = states.remove(0);
        Ok(EncoderStates {
            mel,
            input_hidden,
            blocks: states,
        })
    }
}

fn block_bytes(b: &ConformerBlock) -> usize {
    b.ffn1_norm.bytes()
        + b.ffn1.0.bytes()
        + b.ffn1.1.bytes()
        + b.attn_norm.bytes()
        + b.attn.q.bytes()
        + b.attn.k.bytes()
        + b.attn.v.bytes()
        + b.attn.out.bytes()
        + b.conv_norm.bytes()
        + b.conv_pointwise1.bytes()
        + b.conv_depthwise.bytes()
        + b.conv_inner_norm.bytes()
        + b.conv_pointwise2.bytes()
        + b.ffn2_norm.bytes()
        + b.ffn2.0.bytes()
        + b.ffn2.1.bytes()
        + b.final_norm.bytes()
}
