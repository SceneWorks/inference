//! The **XY_Tokenizer** codec decoder (sc-13518) — MOSS-TTSD's 8-codebook RVQ codes → 24 kHz PCM.
//!
//! MOSS-TTSD's AR brain ([`crate::decode`]) emits discrete delay-pattern RVQ frames; turning those
//! into a waveform is the job of a **separate** ~2.1 GB codec, OpenMOSS's **XY_Tokenizer**
//! (`OpenMOSS-Team/XY_Tokenizer_TTSD_V0`, Apache-2.0). Unlike the sibling
//! `candle-audio-moss-tts-realtime`'s RLFQ codec (a CNN-free transformer upsampler), XY_Tokenizer is
//! a dual-channel (semantic + acoustic) codec whose **decode path** is a mel-reconstruction stack
//! ported here natively onto the workspace's pinned candle revision, faithful to the reference
//! `xy_tokenizer/model.py` (`inference_detokenize`) + `nn/modules.py` + `nn/quantizer.py`.
//!
//! ## The decode graph (`inference_detokenize`)
//!
//! ```text
//!   codes (nq=8, 1, T)  ── ResidualVQ.decode_codes ──▶  z [1, 3072, T]          (12.5 Hz)
//!   z ─▶ post_rvq_adapter  Transformer(3072→768→3072, 4L, full attn)  ─▶ [1, 3072, T]
//!     ─▶ upsample          UpConv ConvTranspose1d(3072→768, k4 s4)     ─▶ [1, 768, 4T]   (50 Hz)
//!     ─▶ acoustic_decoder  Transformer(768, 12L) + deconv1(s2)+deconv2 ─▶ [1, 80, 8T]    (100 Hz)
//!     ─▶ enhanced_vocos    ConvNeXt(80→512, 30L) + ISTFTHead(nfft 960) ─▶ [1, 1, 1920·T] (24 kHz)
//! ```
//!
//! `decoder_upsample_rate = 4·2·240 = 1920`, so 24 kHz / 1920 = **12.5 Hz** frame rate — one AR
//! frame becomes [`SAMPLES_PER_FRAME`] waveform samples.
//!
//! ## The 8→8 codebook mapping (resolved against the reference)
//!
//! The AR brain ([`crate::config`]) emits **8** codebooks per frame (`channels = 8`): codebook 0 in
//! `[0, 1024)` (mapped out of the text-vocab speech range by [`crate::decode`]) and codebooks 1..7
//! in `[0, 1025)`. XY_Tokenizer's `ResidualVQ` ships **8** quantizers (`num_quantizers = 8`,
//! `codebook_size = 1024`). So the AR's 8 codes drive the codec's 8 quantizers, in order:
//! `decode_codes` looks up each code in its quantizer's `[1024, 512]` codebook, **sums** the 8
//! per-quantizer embeddings, then a weight-normed `output_proj` (512→3072) lifts to the decoder input
//! dim. Codes are plain codebook indices in `[0, 1024)`; the AR's in-codebook pad id `1024` is loop
//! bookkeeping (an un-shifted frame never carries it) and is clamped defensively.
//!
//! ## Faithfulness notes
//!
//! - **Weights** load from the pinned `xy_tokenizer.ckpt` raw-pickle checkpoint via candle's pickle
//!   reader (section `generator`, the Kokoro idiom), old-style weight-norm (`weight_g`/`weight_v`)
//!   resolved to a plain `weight = g·v/‖v‖` (norm over the non-output dims). Only the **decode-side**
//!   tensors are materialized; the encoder half is never read.
//! - **Transformer layers** (`OmniWhisperTransformerLayer`): pre-norm `LayerNorm` (eps 1e-5), **full
//!   bidirectional** self-attention (the reference's `causal=False`), sinusoidal absolute positional
//!   embedding added once, `gelu` (erf) FFN. Batch = 1 with a fully-valid sequence, so the reference
//!   variable-length pad mask is a no-op and is omitted.
//! - **ISTFTHead** ports the reference's custom `"same"`-padding inverse STFT (n_fft 960, hop 240):
//!   `irfft` as a fixed inverse-DFT basis matmul, Hann-windowed overlap-add with summed-window
//!   envelope normalization, trimming the `(n_fft − hop)/2` edge pad.

use candle_audio::candle_core::pickle::PthTensors;
use candle_audio::candle_core::{DType, Device, IndexOp, Module, Result as CandleResult, Tensor};
use candle_audio::{AudioError, Result};
use candle_nn::{
    Conv1d, Conv1dConfig, ConvTranspose1d, ConvTranspose1dConfig, LayerNorm, Linear, VarBuilder,
};
use std::collections::HashMap;
use std::path::Path;

/// Native codec output sample rate (Hz).
pub const SAMPLE_RATE: u32 = 24_000;
/// Waveform samples produced per RVQ frame (`decoder_upsample_rate`); 24000 / 1920 = 12.5 Hz.
pub const SAMPLES_PER_FRAME: usize = 1920;
/// The number of RVQ quantizers XY_Tokenizer ships — the AR side emits exactly this many codebooks.
pub const NUM_QUANTIZERS: usize = 8;
/// Codebook cardinality of each quantizer (valid code ids are `0..CODEBOOK_SIZE`).
pub const CODEBOOK_SIZE: usize = 1024;
/// Per-code latent dimension (`codebook_dim`, == `rvq_dim`, so each quantizer's in/out proj is I).
pub const CODEBOOK_DIM: usize = 512;
/// The quantizer output / post-RVQ input dimension (`quantizer_kwargs.output_dim`).
pub const CODE_DIM: usize = 3072;
/// Shared transformer hidden width across the adapters and the acoustic decoder.
pub const D_MODEL: usize = 768;
/// Mel-bin count feeding the vocos head (`num_mel_bins`).
pub const NUM_MEL_BINS: usize = 80;

/// Vocos ISTFT transform size and hop (the 24 kHz `vocos_kwargs`).
const VOCOS_N_FFT: usize = 960;
const VOCOS_HOP: usize = 240;
const VOCOS_DIM: usize = 512;
const VOCOS_INTERMEDIATE: usize = 4096;
const VOCOS_LAYERS: usize = 30;

/// Post-RVQ adapter transformer: 4 full-attention layers, 12 heads, ffn 3072.
const POST_RVQ_LAYERS: usize = 4;
/// Acoustic decoder transformer: 12 full-attention layers, 12 heads, ffn 3072.
const ACOUSTIC_LAYERS: usize = 12;
const ATTENTION_HEADS: usize = 12;
const FFN_DIM: usize = 3072;
const LN_EPS: f64 = 1e-5;
const VOCOS_LN_EPS: f64 = 1e-6;

// --------------------------------------------------------------------------------------------
// Weight loading (pickle section `generator`, old-style weight-norm resolved)
// --------------------------------------------------------------------------------------------

/// Load the XY_Tokenizer checkpoint's **decode-side** tensors from the raw-pickle `generator`
/// section into a name→tensor map (f32), resolving old-style weight-norm pairs. The encoder half is
/// skipped so its storage pages are never faulted in.
fn load_decode_tensors(ckpt: &Path) -> Result<HashMap<String, Tensor>> {
    let tensors = PthTensors::new(ckpt, Some("generator"))
        .map_err(|e| AudioError::Msg(format!("codec: open {} [generator]: {e}", ckpt.display())))?;
    // Only the modules `inference_detokenize` traverses.
    const DECODE_PREFIXES: [&str; 5] = [
        "quantizer.",
        "post_rvq_adapter.",
        "upsample.",
        "acoustic_decoder.",
        "enhanced_vocos.",
    ];
    let mut raw: HashMap<String, Tensor> = HashMap::new();
    for name in tensors.tensor_infos().keys() {
        if !DECODE_PREFIXES.iter().any(|p| name.starts_with(p)) {
            continue;
        }
        // The RVQ EMA bookkeeping buffers (`inited`, `cluster_size`, `embed_avg`) are training state,
        // never used at decode; skip them (they are non-f32-friendly / irrelevant).
        if name.ends_with(".inited")
            || name.ends_with(".cluster_size")
            || name.ends_with(".embed_avg")
        {
            continue;
        }
        let t = tensors
            .get(name)
            .map_err(|e| AudioError::Msg(format!("codec: read {name}: {e}")))?
            .ok_or_else(|| AudioError::Msg(format!("codec: tensor {name} vanished")))?
            .to_dtype(DType::F32)?;
        raw.insert(name.clone(), t);
    }
    resolve_weight_norm(raw)
}

/// Fold every `X.weight_g` / `X.weight_v` pair into `X.weight = g · v / ‖v‖` (norm over all dims
/// except 0 — the torch `weight_norm(dim=0)` default). Non-paired tensors pass through unchanged.
fn resolve_weight_norm(raw: HashMap<String, Tensor>) -> Result<HashMap<String, Tensor>> {
    let mut out = HashMap::with_capacity(raw.len());
    for (name, tensor) in &raw {
        if let Some(base) = name.strip_suffix(".weight_g") {
            let v = raw.get(&format!("{base}.weight_v")).ok_or_else(|| {
                AudioError::Msg(format!("codec: {base}.weight_g without weight_v"))
            })?;
            let mut sq = v.sqr()?;
            for d in 1..v.rank() {
                sq = sq.sum_keepdim(d)?;
            }
            let norm = sq.sqrt()?;
            let w = v.broadcast_mul(&tensor.broadcast_div(&norm)?)?;
            out.insert(format!("{base}.weight"), w);
        } else if !name.ends_with(".weight_v") {
            out.insert(name.clone(), tensor.clone());
        }
    }
    Ok(out)
}

// --------------------------------------------------------------------------------------------
// Small load helpers over a VarBuilder
// --------------------------------------------------------------------------------------------

fn linear(vb: &VarBuilder, name: &str, out: usize, inp: usize, bias: bool) -> CandleResult<Linear> {
    let w = vb.get((out, inp), &format!("{name}.weight"))?;
    let b = if bias {
        Some(vb.get(out, &format!("{name}.bias"))?)
    } else {
        None
    };
    Ok(Linear::new(w, b))
}

fn layer_norm(vb: &VarBuilder, name: &str, dim: usize, eps: f64) -> CandleResult<LayerNorm> {
    let w = vb.get(dim, &format!("{name}.weight"))?;
    let b = vb.get(dim, &format!("{name}.bias"))?;
    Ok(LayerNorm::new(w, b, eps))
}

fn conv1d(
    vb: &VarBuilder,
    name: &str,
    out: usize,
    inp: usize,
    k: usize,
    cfg: Conv1dConfig,
) -> CandleResult<Conv1d> {
    let w = vb.get((out, inp, k), &format!("{name}.weight"))?;
    let b = vb.get(out, &format!("{name}.bias"))?;
    Ok(Conv1d::new(w, Some(b), cfg))
}

fn conv_transpose1d(
    vb: &VarBuilder,
    name: &str,
    inp: usize,
    out: usize,
    k: usize,
    cfg: ConvTranspose1dConfig,
    bias: bool,
) -> CandleResult<ConvTranspose1d> {
    let w = vb.get((inp, out, k), &format!("{name}.weight"))?;
    let b = if bias {
        Some(vb.get(out, &format!("{name}.bias"))?)
    } else {
        None
    };
    Ok(ConvTranspose1d::new(w, b, cfg))
}

// --------------------------------------------------------------------------------------------
// ResidualVQ decode (codes -> [1, CODE_DIM, T])
// --------------------------------------------------------------------------------------------

/// The decode side of XY_Tokenizer's `ResidualVQ`: per-quantizer `[1024, 512]` codebook lookups
/// summed over the 8 quantizers, then a weight-normed `output_proj` (512→3072).
struct RvqDecoder {
    codebooks: Vec<Tensor>, // [nq] each [CODEBOOK_SIZE, CODEBOOK_DIM]
    output_proj: Conv1d,    // weight-normed 1x1, CODEBOOK_DIM -> CODE_DIM
}

impl RvqDecoder {
    fn load(vb: &VarBuilder) -> CandleResult<Self> {
        let vb_q = vb.pp("quantizer");
        let mut codebooks = Vec::with_capacity(NUM_QUANTIZERS);
        for i in 0..NUM_QUANTIZERS {
            codebooks.push(
                vb_q.pp("quantizers")
                    .pp(i)
                    .get((CODEBOOK_SIZE, CODEBOOK_DIM), "codebook")?,
            );
        }
        // output_proj is Conv1d(k=1); the resolved weight is [CODE_DIM, CODEBOOK_DIM, 1].
        let output_proj = conv1d(
            &vb_q,
            "output_proj",
            CODE_DIM,
            CODEBOOK_DIM,
            1,
            Conv1dConfig::default(),
        )?;
        Ok(Self {
            codebooks,
            output_proj,
        })
    }

    /// `codes[q]` = quantizer `q`'s code row (length `T`) → `[1, CODE_DIM, T]`.
    fn decode(&self, codes: &[Vec<u32>], device: &Device) -> CandleResult<Tensor> {
        let t = codes.first().map(Vec::len).unwrap_or(0);
        let mut emb = Tensor::zeros((1, CODEBOOK_DIM, t), DType::F32, device)?;
        for (q, code_row) in codes.iter().enumerate().take(self.codebooks.len()) {
            let ids: Vec<u32> = code_row
                .iter()
                .map(|&c| c.min(CODEBOOK_SIZE as u32 - 1))
                .collect();
            let ids_t = Tensor::from_vec(ids, (t,), device)?;
            let looked = self.codebooks[q].index_select(&ids_t, 0)?; // [T, dim]
            let z = looked.t()?.unsqueeze(0)?.contiguous()?; // [1, dim, T]
            emb = (emb + z)?;
        }
        self.output_proj.forward(&emb) // [1, CODE_DIM, T]
    }
}

// --------------------------------------------------------------------------------------------
// OmniWhisperTransformerLayer (full bidirectional attention, batch=1)
// --------------------------------------------------------------------------------------------

/// Sinusoidal absolute positional embedding `[len, channels]` (`sinusoids`, max_timescale 10000).
fn sinusoids(len: usize, channels: usize, device: &Device) -> CandleResult<Tensor> {
    let half = channels / 2;
    let log_inc = (10_000f64).ln() / (half as f64 - 1.0);
    let mut data = vec![0f32; len * channels];
    for pos in 0..len {
        for j in 0..half {
            let inv = (-log_inc * j as f64).exp();
            let angle = pos as f64 * inv;
            data[pos * channels + j] = angle.sin() as f32; // sin block
            data[pos * channels + half + j] = angle.cos() as f32; // cos block
        }
    }
    Tensor::from_vec(data, (len, channels), device)
}

struct OmniLayer {
    self_attn_layer_norm: LayerNorm,
    q_proj: Linear,
    k_proj: Linear,
    v_proj: Linear,
    out_proj: Linear,
    final_layer_norm: LayerNorm,
    fc1: Linear,
    fc2: Linear,
    num_heads: usize,
    head_dim: usize,
}

impl OmniLayer {
    fn load(vb: &VarBuilder, d_model: usize, num_heads: usize, ffn: usize) -> CandleResult<Self> {
        let attn = vb.pp("self_attn");
        Ok(Self {
            self_attn_layer_norm: layer_norm(vb, "self_attn_layer_norm", d_model, LN_EPS)?,
            q_proj: linear(&attn, "q_proj", d_model, d_model, true)?,
            k_proj: linear(&attn, "k_proj", d_model, d_model, false)?,
            v_proj: linear(&attn, "v_proj", d_model, d_model, true)?,
            out_proj: linear(&attn, "out_proj", d_model, d_model, true)?,
            final_layer_norm: layer_norm(vb, "final_layer_norm", d_model, LN_EPS)?,
            fc1: linear(vb, "fc1", ffn, d_model, true)?,
            fc2: linear(vb, "fc2", d_model, ffn, true)?,
            num_heads,
            head_dim: d_model / num_heads,
        })
    }

    /// `x` is `[1, T, d_model]`; full (non-causal) self-attention, no pad mask (batch=1, all valid).
    fn forward(&self, x: &Tensor) -> CandleResult<Tensor> {
        let residual = x;
        let h = self.self_attn_layer_norm.forward(x)?;
        let attn = self.attention(&h)?;
        let x = (residual + attn)?;
        let residual = &x;
        let h = self.final_layer_norm.forward(&x)?;
        let h = self.fc1.forward(&h)?.gelu_erf()?;
        let h = self.fc2.forward(&h)?;
        residual + h
    }

    fn attention(&self, x: &Tensor) -> CandleResult<Tensor> {
        let (b, t, _) = x.dims3()?;
        let h = self.num_heads;
        let d = self.head_dim;
        let scale = (d as f64).powf(-0.5);
        let q = (self.q_proj.forward(x)? * scale)?;
        let k = self.k_proj.forward(x)?;
        let v = self.v_proj.forward(x)?;
        let shape = |t_: Tensor| -> CandleResult<Tensor> {
            t_.reshape((b, t, h, d))?.transpose(1, 2)?.contiguous()
        };
        let q = shape(q)?; // [b, h, t, d]
        let k = shape(k)?;
        let v = shape(v)?;
        let att = q.matmul(&k.transpose(2, 3)?)?; // scaling already folded into q
        let att = candle_nn::ops::softmax_last_dim(&att)?;
        let out = att.matmul(&v)?; // [b, h, t, d]
        let out = out.transpose(1, 2)?.reshape((b, t, h * d))?;
        self.out_proj.forward(&out)
    }
}

/// Add sinusoidal positional embedding to `[1, T, d]` (float32 addition, per the reference).
fn add_positional(x: &Tensor) -> CandleResult<Tensor> {
    let (_, t, d) = x.dims3()?;
    let pe = sinusoids(t, d, x.device())?.unsqueeze(0)?; // [1, T, d]
    x.broadcast_add(&pe)
}

// --------------------------------------------------------------------------------------------
// post_rvq_adapter: Transformer(input_dim 3072 -> d_model 768 -> output_dim 3072)
// --------------------------------------------------------------------------------------------

struct PostRvqAdapter {
    proj: Linear, // 3072 -> 768
    layers: Vec<OmniLayer>,
    layer_norm: LayerNorm,
    out_proj: Linear, // 768 -> 3072
}

impl PostRvqAdapter {
    fn load(vb: &VarBuilder) -> CandleResult<Self> {
        let vb = vb.pp("post_rvq_adapter");
        let mut layers = Vec::with_capacity(POST_RVQ_LAYERS);
        for i in 0..POST_RVQ_LAYERS {
            layers.push(OmniLayer::load(
                &vb.pp("layers").pp(i),
                D_MODEL,
                ATTENTION_HEADS,
                FFN_DIM,
            )?);
        }
        Ok(Self {
            proj: linear(&vb, "proj", D_MODEL, CODE_DIM, true)?,
            layers,
            layer_norm: layer_norm(&vb, "layer_norm", D_MODEL, LN_EPS)?,
            out_proj: linear(&vb, "out_proj", CODE_DIM, D_MODEL, true)?,
        })
    }

    /// `[1, CODE_DIM, T]` -> `[1, CODE_DIM, T]`.
    fn forward(&self, x: &Tensor) -> CandleResult<Tensor> {
        // proj on channels: (1, T, in) -> (1, T, d_model)
        let h = x.transpose(1, 2)?.contiguous()?; // [1, T, CODE_DIM]
        let mut h = self.proj.forward(&h)?; // [1, T, D_MODEL]
        h = add_positional(&h)?;
        for layer in &self.layers {
            h = layer.forward(&h)?;
        }
        h = self.layer_norm.forward(&h)?;
        let h = self.out_proj.forward(&h)?; // [1, T, CODE_DIM]
        h.transpose(1, 2)?.contiguous()
    }
}

// --------------------------------------------------------------------------------------------
// upsample: UpConv ConvTranspose1d(stride*d_model -> d_model, k=stride)
// --------------------------------------------------------------------------------------------

struct Upsample {
    up_conv: ConvTranspose1d,
}

impl Upsample {
    fn load(vb: &VarBuilder) -> CandleResult<Self> {
        // in = stride * d_model = 4 * 768 = 3072; out = 768; k = stride = 4; no bias.
        let up_conv = conv_transpose1d(
            &vb.pp("upsample"),
            "up_conv",
            4 * D_MODEL,
            D_MODEL,
            4,
            ConvTranspose1dConfig {
                stride: 4,
                ..Default::default()
            },
            false,
        )?;
        Ok(Self { up_conv })
    }

    /// `[1, CODE_DIM, T]` -> `[1, D_MODEL, 4T]`.
    fn forward(&self, x: &Tensor) -> CandleResult<Tensor> {
        self.up_conv.forward(x)
    }
}

// --------------------------------------------------------------------------------------------
// acoustic_decoder: Transformer(768, 12L) + deconv1(s2) + deconv2 -> mel (80)
// --------------------------------------------------------------------------------------------

struct AcousticDecoder {
    layers: Vec<OmniLayer>,
    layer_norm: LayerNorm,
    deconv1: ConvTranspose1d, // 768->768, k3, s2
    deconv2: ConvTranspose1d, // 768->80, k3, s1
    stride_size: usize,
}

impl AcousticDecoder {
    fn load(vb: &VarBuilder) -> CandleResult<Self> {
        let vb = vb.pp("acoustic_decoder");
        let mut layers = Vec::with_capacity(ACOUSTIC_LAYERS);
        for i in 0..ACOUSTIC_LAYERS {
            layers.push(OmniLayer::load(
                &vb.pp("layers").pp(i),
                D_MODEL,
                ATTENTION_HEADS,
                FFN_DIM,
            )?);
        }
        let deconv1 = conv_transpose1d(
            &vb,
            "deconv1",
            D_MODEL,
            D_MODEL,
            3,
            ConvTranspose1dConfig {
                stride: 2,
                ..Default::default()
            },
            true,
        )?;
        let deconv2 = conv_transpose1d(
            &vb,
            "deconv2",
            D_MODEL,
            NUM_MEL_BINS,
            3,
            ConvTranspose1dConfig {
                stride: 1,
                ..Default::default()
            },
            true,
        )?;
        Ok(Self {
            layers,
            layer_norm: layer_norm(&vb, "layer_norm", D_MODEL, LN_EPS)?,
            deconv1,
            deconv2,
            stride_size: 2,
        })
    }

    /// `[1, D_MODEL, L]` -> `[1, NUM_MEL_BINS, 2L]` (mel spectrogram at 100 Hz).
    fn forward(&self, x: &Tensor) -> CandleResult<Tensor> {
        let l = x.dim(2)?;
        let mut h = x.transpose(1, 2)?.contiguous()?; // [1, L, D_MODEL]
        h = add_positional(&h)?;
        for layer in &self.layers {
            h = layer.forward(&h)?;
        }
        h = self.layer_norm.forward(&h)?;
        let h = h.transpose(1, 2)?.contiguous()?; // [1, D_MODEL, L]
        let h = self.deconv1.forward(&h)?.gelu_erf()?;
        let h = self.deconv2.forward(&h)?.gelu_erf()?; // [1, 80, ~2L]
                                                       // Trim to exactly L * stride_size (the reference's expected_length guard).
        let expected = l * self.stride_size;
        if h.dim(2)? > expected {
            h.narrow(2, 0, expected)
        } else {
            Ok(h)
        }
    }
}

// --------------------------------------------------------------------------------------------
// enhanced_vocos: ConvNeXt backbone + ISTFTHead (custom "same" ISTFT)
// --------------------------------------------------------------------------------------------

struct ConvNeXtBlock {
    dwconv: Conv1d, // depthwise, groups=dim, k7 p3
    norm: LayerNorm,
    pwconv1: Linear, // dim -> intermediate
    pwconv2: Linear, // intermediate -> dim
    gamma: Tensor,   // [dim]
}

impl ConvNeXtBlock {
    fn load(vb: &VarBuilder) -> CandleResult<Self> {
        let dwconv = conv1d(
            vb,
            "dwconv",
            VOCOS_DIM,
            1,
            7,
            Conv1dConfig {
                padding: 3,
                groups: VOCOS_DIM,
                ..Default::default()
            },
        )?;
        Ok(Self {
            dwconv,
            norm: layer_norm(vb, "norm", VOCOS_DIM, VOCOS_LN_EPS)?,
            pwconv1: linear(vb, "pwconv1", VOCOS_INTERMEDIATE, VOCOS_DIM, true)?,
            pwconv2: linear(vb, "pwconv2", VOCOS_DIM, VOCOS_INTERMEDIATE, true)?,
            gamma: vb.get(VOCOS_DIM, "gamma")?,
        })
    }

    /// `[1, dim, T]` -> `[1, dim, T]`.
    fn forward(&self, x: &Tensor) -> CandleResult<Tensor> {
        let residual = x;
        let h = self.dwconv.forward(x)?; // [1, dim, T]
        let h = h.transpose(1, 2)?.contiguous()?; // [1, T, dim]
        let h = self.norm.forward(&h)?;
        let h = self.pwconv1.forward(&h)?.gelu_erf()?;
        let h = self.pwconv2.forward(&h)?;
        let h = h.broadcast_mul(&self.gamma)?; // per-channel scale
        let h = h.transpose(1, 2)?.contiguous()?; // [1, dim, T]
        residual + h
    }
}

struct Vocos {
    embed: Conv1d, // 80 -> 512, k7 p3
    norm: LayerNorm,
    blocks: Vec<ConvNeXtBlock>,
    final_layer_norm: LayerNorm,
    head_out: Linear, // 512 -> n_fft + 2
    /// Inverse-DFT basis for the custom ISTFT: `[N, n_bins]` cos / sin, backward-normalized.
    idft_cos: Tensor,
    idft_sin: Tensor,
    window: Vec<f32>,
}

impl Vocos {
    fn load(vb: &VarBuilder, device: &Device) -> CandleResult<Self> {
        let vb = vb.pp("enhanced_vocos");
        let bb = vb.pp("backbone");
        let embed = conv1d(
            &bb,
            "embed",
            VOCOS_DIM,
            NUM_MEL_BINS,
            7,
            Conv1dConfig {
                padding: 3,
                ..Default::default()
            },
        )?;
        let mut blocks = Vec::with_capacity(VOCOS_LAYERS);
        for i in 0..VOCOS_LAYERS {
            blocks.push(ConvNeXtBlock::load(&bb.pp("convnext").pp(i))?);
        }
        let head_out = linear(&vb.pp("head"), "out", VOCOS_N_FFT + 2, VOCOS_DIM, true)?;
        let (idft_cos, idft_sin) = idft_basis(VOCOS_N_FFT, device)?;
        Ok(Self {
            embed,
            norm: layer_norm(&bb, "norm", VOCOS_DIM, VOCOS_LN_EPS)?,
            blocks,
            final_layer_norm: layer_norm(&bb, "final_layer_norm", VOCOS_DIM, VOCOS_LN_EPS)?,
            head_out,
            idft_cos,
            idft_sin,
            window: hann_window(VOCOS_N_FFT),
        })
    }

    /// `[1, NUM_MEL_BINS, T_mel]` -> mono waveform `Vec<f32>` of `T_mel * VOCOS_HOP` samples.
    fn forward(&self, mel: &Tensor) -> CandleResult<Vec<f32>> {
        // Backbone.
        let mut x = self.embed.forward(mel)?; // [1, dim, T]
                                              // norm applied on (B, T, dim).
        x = self.norm.forward(&x.transpose(1, 2)?.contiguous()?)?;
        x = x.transpose(1, 2)?.contiguous()?; // [1, dim, T]
        for block in &self.blocks {
            x = block.forward(&x)?;
        }
        let x = self
            .final_layer_norm
            .forward(&x.transpose(1, 2)?.contiguous()?)?; // [1, T, dim]
                                                          // ISTFTHead: out -> (mag, phase).
        let coeffs = self.head_out.forward(&x)?; // [1, T, n_fft+2]
        let coeffs = coeffs.i(0)?; // [T, n_fft+2]
        let n_bins = VOCOS_N_FFT / 2 + 1;
        let mag = coeffs.narrow(1, 0, n_bins)?; // [T, n_bins]
        let phase = coeffs.narrow(1, n_bins, n_bins)?; // [T, n_bins]
        let mag = mag.exp()?.clamp(f32::NEG_INFINITY, 1e2)?;
        let re = (mag.clone() * phase.cos()?)?; // [T, n_bins]
        let im = (mag * phase.sin()?)?;
        // irfft via fixed inverse-DFT basis: frames[T, N] = re @ cos^T - im @ sin^T.
        let frames = (re.matmul(&self.idft_cos.t()?)? - im.matmul(&self.idft_sin.t()?)?)?;
        let frames: Vec<Vec<f32>> = frames.to_vec2::<f32>()?; // [T][N]
        Ok(self.overlap_add(&frames))
    }

    /// Windowed overlap-add with summed-window-square envelope normalization and `"same"` edge
    /// trimming (`pad = (n_fft - hop)/2`), exactly the reference custom ISTFT.
    fn overlap_add(&self, frames: &[Vec<f32>]) -> Vec<f32> {
        let n = VOCOS_N_FFT;
        let hop = VOCOS_HOP;
        let t = frames.len();
        if t == 0 {
            return Vec::new();
        }
        let out_len = (t - 1) * hop + n;
        let mut y = vec![0f32; out_len];
        let mut env = vec![0f32; out_len];
        for (f, frame) in frames.iter().enumerate() {
            let base = f * hop;
            for i in 0..n {
                let w = self.window[i];
                y[base + i] += frame[i] * w;
                env[base + i] += w * w;
            }
        }
        let pad = (n - hop) / 2;
        let mut out = Vec::with_capacity(out_len - 2 * pad);
        for i in pad..out_len - pad {
            let e = env[i];
            out.push(if e > 1e-11 { y[i] / e } else { 0.0 });
        }
        out
    }
}

/// A periodic Hann window of length `n` (`0.5·(1 − cos(2π i / n))`, torch's default).
fn hann_window(n: usize) -> Vec<f32> {
    (0..n)
        .map(|i| 0.5 - 0.5 * (2.0 * std::f64::consts::PI * i as f64 / n as f64).cos() as f32)
        .collect()
}

/// The backward-normalized inverse real-DFT basis `[N, n_bins]`: `cos[n,k] = a_k·cos(2πkn/N)/N`
/// and `sin[n,k] = a_k·sin(2πkn/N)/N`, with the one-sided fold factor `a_k = 1` at DC/Nyquist and
/// `2` in between — so `frames = re @ cosᵀ − im @ sinᵀ` reproduces `torch.fft.irfft(norm="backward")`.
fn idft_basis(n: usize, device: &Device) -> CandleResult<(Tensor, Tensor)> {
    let n_bins = n / 2 + 1;
    let inv_n = 1.0 / n as f64;
    let mut cos = vec![0f32; n * n_bins];
    let mut sin = vec![0f32; n * n_bins];
    for idx in 0..n {
        for k in 0..n_bins {
            let a = if k == 0 || k == n / 2 { 1.0 } else { 2.0 };
            let theta = 2.0 * std::f64::consts::PI * (k as f64) * (idx as f64) * inv_n;
            cos[idx * n_bins + k] = (a * theta.cos() * inv_n) as f32;
            sin[idx * n_bins + k] = (a * theta.sin() * inv_n) as f32;
        }
    }
    Ok((
        Tensor::from_vec(cos, (n, n_bins), device)?,
        Tensor::from_vec(sin, (n, n_bins), device)?,
    ))
}

// --------------------------------------------------------------------------------------------
// The assembled decoder
// --------------------------------------------------------------------------------------------

/// The loaded XY_Tokenizer **decoder** (RVQ decode + post-RVQ adapter + upsample + acoustic decoder
/// + vocos), 8-codebook RVQ frames → 24 kHz mono PCM.
pub struct XyTokenizerCodec {
    rvq: RvqDecoder,
    post_rvq: PostRvqAdapter,
    upsample: Upsample,
    acoustic: AcousticDecoder,
    vocos: Vocos,
    device: Device,
}

impl XyTokenizerCodec {
    /// Load the decoder from a `xy_tokenizer.ckpt` raw-pickle checkpoint.
    pub fn load(ckpt: &Path) -> Result<Self> {
        let device = candle_audio::default_device()?;
        let tensors = load_decode_tensors(ckpt)?
            .into_iter()
            .map(|(k, v)| Ok((k, v.to_device(&device)?)))
            .collect::<Result<HashMap<_, _>>>()?;
        let vb = VarBuilder::from_tensors(tensors, DType::F32, &device);
        Self::from_var_builder(vb, device)
    }

    fn from_var_builder(vb: VarBuilder, device: Device) -> Result<Self> {
        let rvq = RvqDecoder::load(&vb)
            .map_err(|e| AudioError::Msg(format!("codec: RVQ decode: {e}")))?;
        let post_rvq = PostRvqAdapter::load(&vb)
            .map_err(|e| AudioError::Msg(format!("codec: post_rvq_adapter: {e}")))?;
        let upsample =
            Upsample::load(&vb).map_err(|e| AudioError::Msg(format!("codec: upsample: {e}")))?;
        let acoustic = AcousticDecoder::load(&vb)
            .map_err(|e| AudioError::Msg(format!("codec: acoustic_decoder: {e}")))?;
        let vocos = Vocos::load(&vb, &device)
            .map_err(|e| AudioError::Msg(format!("codec: enhanced_vocos: {e}")))?;
        Ok(Self {
            rvq,
            post_rvq,
            upsample,
            acoustic,
            vocos,
            device,
        })
    }

    /// Native sample rate.
    pub fn sample_rate(&self) -> u32 {
        SAMPLE_RATE
    }

    /// Waveform samples produced per RVQ frame.
    pub fn samples_per_frame(&self) -> usize {
        SAMPLES_PER_FRAME
    }

    /// Decode a block of 8-codebook RVQ frames (`frames[f][q]` = codebook `q`'s code at frame `f`)
    /// into an interleaved mono `Vec<f32>` at 24 kHz. `cancel` is polled between the (bounded)
    /// stages — the AR loop is where cancellation primarily lands.
    pub fn decode_frames(
        &self,
        frames: &[Vec<u32>],
        cancel: &dyn Fn() -> bool,
    ) -> CandleResult<Option<Vec<f32>>> {
        if frames.is_empty() {
            return Ok(Some(Vec::new()));
        }
        if cancel() {
            return Ok(None);
        }
        // Transpose [T][nq] -> per-quantizer rows [nq][T].
        let nq = NUM_QUANTIZERS;
        let t = frames.len();
        let mut rows: Vec<Vec<u32>> = (0..nq).map(|_| Vec::with_capacity(t)).collect();
        for frame in frames {
            for (q, row) in rows.iter_mut().enumerate() {
                row.push(frame.get(q).copied().unwrap_or(0));
            }
        }
        let debug = std::env::var_os("MOSS_TTSD_CODEC_DEBUG").is_some();
        let mut x = self.rvq.decode(&rows, &self.device)?; // [1, CODE_DIM, T]
        if debug {
            eprintln!("[xytok] rvq: {:?} rms={:.5}", x.dims(), rms(&x));
        }
        if cancel() {
            return Ok(None);
        }
        x = self.post_rvq.forward(&x)?;
        x = self.upsample.forward(&x)?;
        if debug {
            eprintln!("[xytok] upsample: {:?} rms={:.5}", x.dims(), rms(&x));
        }
        if cancel() {
            return Ok(None);
        }
        let mel = self.acoustic.forward(&x)?; // [1, 80, 8T]
        if debug {
            eprintln!("[xytok] mel: {:?} rms={:.5}", mel.dims(), rms(&mel));
        }
        if cancel() {
            return Ok(None);
        }
        let wav = self.vocos.forward(&mel)?;
        Ok(Some(wav))
    }
}

/// Root-mean-square of all elements (debug instrumentation only).
fn rms(x: &Tensor) -> f32 {
    x.sqr()
        .and_then(|s| s.mean_all())
        .and_then(|m| m.sqrt())
        .and_then(|t| t.to_scalar::<f32>())
        .unwrap_or(f32::NAN)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn hann_window_is_periodic() {
        let w = hann_window(8);
        assert_eq!(w.len(), 8);
        assert!(w[0].abs() < 1e-6, "periodic Hann starts at 0");
        // Symmetric about the middle for a periodic window (w[k] == w[N-k]).
        for k in 1..8 {
            assert!((w[k] - w[8 - k]).abs() < 1e-6, "periodic symmetry at {k}");
        }
    }

    #[test]
    fn sinusoids_shape_and_first_row() {
        let dev = Device::Cpu;
        let s = sinusoids(5, 8, &dev).unwrap();
        assert_eq!(s.dims(), &[5, 8]);
        // Row 0: sin block all 0, cos block all 1.
        let row0 = s.i(0).unwrap().to_vec1::<f32>().unwrap();
        for v in &row0[..4] {
            assert!(v.abs() < 1e-6);
        }
        for v in &row0[4..] {
            assert!((v - 1.0).abs() < 1e-6);
        }
    }

    /// The inverse-DFT basis reproduces `torch.fft.irfft(rfft(x))` for a known real signal: a basis
    /// built for N points, applied to the analytic rfft of a pure cosine, returns that cosine.
    #[test]
    fn idft_basis_reconstructs_a_cosine() {
        let dev = Device::Cpu;
        let n = 16usize;
        let n_bins = n / 2 + 1;
        // x[t] = cos(2π·2·t/N): rfft is a single spike at bin k=2 with value N/2 (real).
        let mut re = vec![0f32; n_bins];
        let im = vec![0f32; n_bins];
        re[2] = (n as f32) / 2.0;
        let (cos, sin) = idft_basis(n, &dev).unwrap();
        let re_t = Tensor::from_vec(re, (1, n_bins), &dev).unwrap();
        let im_t = Tensor::from_vec(im, (1, n_bins), &dev).unwrap();
        let frames = (re_t.matmul(&cos.t().unwrap()).unwrap()
            - im_t.matmul(&sin.t().unwrap()).unwrap())
        .unwrap();
        let got = frames.i(0).unwrap().to_vec1::<f32>().unwrap();
        for (t, g) in got.iter().enumerate() {
            let want = (2.0 * std::f64::consts::PI * 2.0 * t as f64 / n as f64).cos() as f32;
            assert!((g - want).abs() < 1e-4, "sample {t}: {g} vs {want}");
        }
    }

    #[test]
    fn overlap_add_length_matches_downsample_rate() {
        // A synthetic Vocos with the real basis but trivial frames: T frames -> T*hop samples.
        let dev = Device::Cpu;
        let (idft_cos, idft_sin) = idft_basis(VOCOS_N_FFT, &dev).unwrap();
        let vocos = Vocos {
            embed: conv1d(
                &VarBuilder::from_tensors(
                    dummy_conv("e", VOCOS_DIM, NUM_MEL_BINS, 7),
                    DType::F32,
                    &dev,
                ),
                "e",
                VOCOS_DIM,
                NUM_MEL_BINS,
                7,
                Conv1dConfig {
                    padding: 3,
                    ..Default::default()
                },
            )
            .unwrap(),
            norm: dummy_ln(&dev, VOCOS_DIM),
            blocks: Vec::new(),
            final_layer_norm: dummy_ln(&dev, VOCOS_DIM),
            head_out: dummy_linear(&dev, VOCOS_N_FFT + 2, VOCOS_DIM),
            idft_cos,
            idft_sin,
            window: hann_window(VOCOS_N_FFT),
        };
        // 5 frames of zeros -> 5 * hop samples (envelope-normalized zeros).
        let frames = vec![vec![0f32; VOCOS_N_FFT]; 5];
        let wav = vocos.overlap_add(&frames);
        assert_eq!(wav.len(), 5 * VOCOS_HOP);
    }

    fn dummy_conv(name: &str, out: usize, inp: usize, k: usize) -> HashMap<String, Tensor> {
        let dev = Device::Cpu;
        let mut m = HashMap::new();
        m.insert(
            format!("{name}.weight"),
            Tensor::zeros((out, inp, k), DType::F32, &dev).unwrap(),
        );
        m.insert(
            format!("{name}.bias"),
            Tensor::zeros(out, DType::F32, &dev).unwrap(),
        );
        m
    }

    fn dummy_ln(dev: &Device, dim: usize) -> LayerNorm {
        LayerNorm::new(
            Tensor::ones(dim, DType::F32, dev).unwrap(),
            Tensor::zeros(dim, DType::F32, dev).unwrap(),
            VOCOS_LN_EPS,
        )
    }

    fn dummy_linear(dev: &Device, out: usize, inp: usize) -> Linear {
        Linear::new(
            Tensor::zeros((out, inp), DType::F32, dev).unwrap(),
            Some(Tensor::zeros(out, DType::F32, dev).unwrap()),
        )
    }
}
