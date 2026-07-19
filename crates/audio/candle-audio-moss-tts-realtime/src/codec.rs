//! The **MOSS-Audio-Tokenizer** codec decoder (sc-13392) — RVQ speech codes → 24 kHz waveform.
//!
//! MOSS-TTS-Realtime's AR brain ([`crate::decode`]) only emits discrete RVQ codes; turning those
//! into a waveform is the job of a **separate** ~7.1 GB model, `OpenMOSS-Team/MOSS-Audio-Tokenizer`
//! (Apache-2.0), a novel **RLFQ streaming codec** (config `quantizer_type=rlfq`, 32 quantizers,
//! codebook 1024, `codebook_dim=8`, `rvq_dim=512`). It is a Moshi/Mimi-scale, **CNN-free** codec:
//! its decoder is a stack of causal RoPE transformers interleaved with `PatchedPretransform`
//! channel→time upsamplers. This module ports the **decode path only** (the TTS direction) natively
//! onto the pinned candle revision, faithful to the reference `modeling_moss_audio_tokenizer.py`.
//!
//! ## The decode graph (`_decode_frame`)
//!
//! ```text
//!   codes (nq, 1, T)  ── RLFQ.decode_codes ──▶  z [1, 768, T]      (rvq_dim 512 → output 768)
//!   z ─▶ Transformer(768→1280, 32L) ─▶ PatchedPretransform(2) ─▶ [1, 640, 2T]
//!     ─▶ Transformer(640→768, 12L)  ─▶ PatchedPretransform(2) ─▶ [1, 384, 4T]
//!     ─▶ Transformer(384→768, 12L)  ─▶ PatchedPretransform(2) ─▶ [1, 384, 8T]
//!     ─▶ Transformer(384→240, 12L)  ─▶ PatchedPretransform(240) ─▶ [1, 1, 1920·T]  (waveform)
//! ```
//!
//! `∏ patch = 2·2·2·240 = 1920 = downsample_rate`, so 24 kHz / 1920 = **12.5 Hz** frame rate — the
//! AR side's [`crate::model::FRAME_RATE_HZ`].
//!
//! ## The 16-vs-32 quantizer mapping (resolved against the reference)
//!
//! The codec ships **32** quantizers, but `ResidualLFQ.decode_codes` iterates only `quantizers[:nq]`
//! — the codec's documented **variable-bitrate** decode (`model.decode(codes[:8])` in the model
//! card). MOSS-TTS-Realtime emits **16** codebooks per frame (`config.rvq = 16`), a documented
//! operating point (2 kbps). So the AR's 16 codes drive the codec's **first 16** quantizers, in
//! order; the codec sums those 16 residual embeddings and decodes. Codes are plain codebook indices
//! in `[0, 1024)` (the AR's pad/bos/eos ids `≥ 1024` are AR-loop bookkeeping and never reach here;
//! a stray one is clamped defensively).
//!
//! ## Faithfulness notes
//!
//! - **RLFQ decode** takes the raw `Embedding` lookup (codebook `1024×8`) → weight-normed 1×1
//!   `out_proj` (8→512); **no** L2-normalization on the decode side (that lives only in the encode
//!   path). The 16 per-quantizer embeddings are summed, then the RLFQ `output_proj` (512→768,
//!   weight-normed 1×1) lifts to the decoder's input dim.
//! - **Transformer layer**: pre-norm `LayerNorm` (eps 1e-5), interleaved RoPE (`max_period` 10000),
//!   **sliding-window causal** attention (`context = ⌊frame_rate · 10⌋` positions per stage),
//!   `LayerScale` (init 0.01) on both residual branches, `gelu` (erf) MLP (`gating="none"`).
//! - **PatchedPretransform.decode**: `[b, d·h, l] → [b, d, l·h]` (channels→time, the reference's
//!   exact `reshape/permute`).
//! - Every convolution here is `kernel_size=1` (pointwise) with old-new-API weight-norm params
//!   (`parametrizations.weight.original0` = g magnitude, `original1` = v direction; effective
//!   `w = g · v / ‖v‖`, norm over the non-output dims), matching the pinned checkpoint's key layout.
//!
//! The whole decode graph is **causal** (transformers causal, projections/patching pointwise-in-
//! time), so decoding a growing prefix reproduces the earlier samples byte-for-byte — the property
//! [`crate::model`]'s streaming path relies on for the `AudioChunk` reassembly law.

use candle_audio::candle_core::{DType, Device, IndexOp, Module, Result as CandleResult, Tensor};
use candle_audio::{AudioError, Result};
use candle_nn::{Conv1d, Conv1dConfig, LayerNorm, Linear, VarBuilder};
use serde::Deserialize;
use std::path::Path;

/// Native codec output sample rate (Hz).
pub const SAMPLE_RATE: u32 = 24_000;
/// Waveform samples per RVQ frame (`∏ decoder patch sizes`); 24000 / 1920 = 12.5 Hz.
pub const DOWNSAMPLE_RATE: usize = 1920;
/// The codebook cardinality of each RLFQ quantizer (valid code ids are `0..CODEBOOK_SIZE`).
pub const CODEBOOK_SIZE: usize = 1024;
/// Per-code latent dimension (the `Embedding` width).
pub const CODEBOOK_DIM: usize = 8;
/// The residual-quantizer working dimension the per-quantizer embeddings are summed in.
pub const RVQ_DIM: usize = 512;
/// The quantizer output / decoder input dimension.
pub const CODE_DIM: usize = 768;
/// The causal transformers' context window, in seconds (`config.causal_transformer_context_duration`).
pub const CONTEXT_DURATION_SECS: f64 = 10.0;
/// RoPE base period for the codec transformers (`config`'s `max_period`).
pub const ROPE_MAX_PERIOD: f64 = 10_000.0;

/// One decoder Transformer stage's fixed hyperparameters (from `config.json.decoder_kwargs`; the
/// pinned revision is immutable, so these are hardcoded and cross-checked against tensor shapes and
/// the config scalars at load — the moss-sfx VAE idiom).
#[derive(Clone, Copy)]
struct StageSpec {
    /// `decoder.{index}` position in the reference `nn.ModuleList`.
    index: usize,
    input_dim: usize,
    d_model: usize,
    output_dim: usize,
    num_heads: usize,
    num_layers: usize,
    dim_feedforward: usize,
}

/// The four decoder Transformer stages, in order, with the `PatchedPretransform` patch sizes that
/// follow each one (`decoder.{index+1}`). `∏ patch = 1920`.
const STAGES: [StageSpec; 4] = [
    StageSpec {
        index: 0,
        input_dim: 768,
        d_model: 1280,
        output_dim: 1280,
        num_heads: 20,
        num_layers: 32,
        dim_feedforward: 5120,
    },
    StageSpec {
        index: 2,
        input_dim: 640,
        d_model: 768,
        output_dim: 768,
        num_heads: 12,
        num_layers: 12,
        dim_feedforward: 3072,
    },
    StageSpec {
        index: 4,
        input_dim: 384,
        d_model: 768,
        output_dim: 768,
        num_heads: 12,
        num_layers: 12,
        dim_feedforward: 3072,
    },
    StageSpec {
        index: 6,
        input_dim: 384,
        d_model: 768,
        output_dim: 240,
        num_heads: 12,
        num_layers: 12,
        dim_feedforward: 3072,
    },
];

/// Patch sizes of `decoder.{1,3,5,7}` (the upsamplers after each Transformer stage).
const PATCH_SIZES: [usize; 4] = [2, 2, 2, 240];

// --------------------------------------------------------------------------------------------
// config.json (scalar validation only — the module list is hardcoded above)
// --------------------------------------------------------------------------------------------

#[derive(Debug, Deserialize)]
struct QuantizerKwargs {
    codebook_dim: usize,
    codebook_size: usize,
    num_quantizers: usize,
    rvq_dim: usize,
    #[serde(default)]
    output_dim: Option<usize>,
    #[serde(default)]
    quantizer_type: Option<String>,
}

#[derive(Debug, Deserialize)]
struct CodecConfigJson {
    #[serde(default)]
    architectures: Vec<String>,
    sample_rate: Option<u32>,
    downsample_rate: usize,
    #[serde(default = "default_context_duration")]
    causal_transformer_context_duration: f64,
    quantizer_kwargs: QuantizerKwargs,
    #[serde(default)]
    quantizer_type: Option<String>,
}

fn default_context_duration() -> f64 {
    CONTEXT_DURATION_SECS
}

/// The validated codec config: the scalars the decode path depends on, cross-checked against the
/// hardcoded stage table.
#[derive(Debug, Clone)]
pub struct CodecConfig {
    pub sample_rate: u32,
    pub downsample_rate: usize,
    pub context_duration: f64,
    pub num_quantizers: usize,
    pub codebook_size: usize,
    pub codebook_dim: usize,
    pub rvq_dim: usize,
    pub output_dim: usize,
}

impl CodecConfig {
    /// Parse and validate `config.json` from a codec snapshot directory.
    pub fn from_dir(dir: &Path) -> Result<Self> {
        let path = dir.join("config.json");
        let text = std::fs::read_to_string(&path)
            .map_err(|e| AudioError::Msg(format!("read {}: {e}", path.display())))?;
        Self::from_json(&text)
    }

    /// Parse and validate a codec `config.json` document.
    pub fn from_json(text: &str) -> Result<Self> {
        let cfg: CodecConfigJson = serde_json::from_str(text)
            .map_err(|e| AudioError::Msg(format!("parse MOSS-Audio-Tokenizer config.json: {e}")))?;
        if !cfg.architectures.is_empty()
            && !cfg
                .architectures
                .iter()
                .any(|a| a == "MossAudioTokenizerModel")
        {
            return Err(AudioError::Msg(format!(
                "config.json: architectures {:?} is not MossAudioTokenizerModel",
                cfg.architectures
            )));
        }
        let qtype = cfg
            .quantizer_kwargs
            .quantizer_type
            .as_deref()
            .or(cfg.quantizer_type.as_deref())
            .unwrap_or("rvq");
        if !matches!(qtype, "rlfq" | "random_prefix_rlfq") {
            return Err(AudioError::Msg(format!(
                "config.json: quantizer_type {qtype:?} is not the RLFQ this decoder ports"
            )));
        }
        if cfg.downsample_rate != DOWNSAMPLE_RATE {
            return Err(AudioError::Msg(format!(
                "config.json: downsample_rate {} != the pinned {DOWNSAMPLE_RATE}",
                cfg.downsample_rate
            )));
        }
        let q = &cfg.quantizer_kwargs;
        let output_dim = q.output_dim.unwrap_or(CODE_DIM);
        for (name, got, want) in [
            ("codebook_size", q.codebook_size, CODEBOOK_SIZE),
            ("codebook_dim", q.codebook_dim, CODEBOOK_DIM),
            ("rvq_dim", q.rvq_dim, RVQ_DIM),
            ("output_dim", output_dim, CODE_DIM),
        ] {
            if got != want {
                return Err(AudioError::Msg(format!(
                    "config.json: quantizer_kwargs.{name} {got} != the pinned {want}"
                )));
            }
        }
        // The hardcoded patch sizes must reproduce the config's downsample rate.
        let prod: usize = PATCH_SIZES.iter().product();
        if prod != cfg.downsample_rate {
            return Err(AudioError::Msg(format!(
                "internal: hardcoded patch product {prod} != downsample_rate {}",
                cfg.downsample_rate
            )));
        }
        Ok(Self {
            sample_rate: cfg.sample_rate.unwrap_or(SAMPLE_RATE),
            downsample_rate: cfg.downsample_rate,
            context_duration: cfg.causal_transformer_context_duration,
            num_quantizers: q.num_quantizers,
            codebook_size: q.codebook_size,
            codebook_dim: q.codebook_dim,
            rvq_dim: q.rvq_dim,
            output_dim,
        })
    }
}

// --------------------------------------------------------------------------------------------
// Weight-normed 1×1 conv (pointwise) — the checkpoint's `parametrizations.weight.original{0,1}`.
// --------------------------------------------------------------------------------------------

/// Resolve a weight-normed `Conv1d(k=1)` from its `parametrizations.weight.{original0,original1}` +
/// `bias`. `original0` is the per-output-channel magnitude `g` (`[out, 1, 1]`), `original1` the
/// direction `v` (`[out, in, 1]`); the effective weight is `w = g · v / ‖v‖` with the norm over the
/// non-output dims (the `weight_norm(dim=0)` default). Returned as a plain `Conv1d`.
fn wn_conv1d(out_ch: usize, in_ch: usize, vb: &VarBuilder) -> CandleResult<Conv1d> {
    let g = vb.get((out_ch, 1, 1), "parametrizations.weight.original0")?;
    let v = vb.get((out_ch, in_ch, 1), "parametrizations.weight.original1")?;
    let norm = v.sqr()?.sum_keepdim((1, 2))?.sqrt()?;
    let w = v.broadcast_mul(&g)?.broadcast_div(&norm)?;
    let bias = vb.get(out_ch, "bias")?;
    Ok(Conv1d::new(w, Some(bias), Conv1dConfig::default()))
}

// --------------------------------------------------------------------------------------------
// RLFQ quantizer decode
// --------------------------------------------------------------------------------------------

/// The decode side of the Residual-LFQ quantizer: per-quantizer codebook `Embedding` + weight-normed
/// `out_proj` (8→512), summed over the used quantizers, then the shared `output_proj` (512→768).
struct RlfqDecoder {
    /// `[nq]` codebook tables, each `[codebook_size, codebook_dim]`.
    codebooks: Vec<Tensor>,
    /// `[nq]` per-quantizer `out_proj` (weight-normed 1×1, `codebook_dim → rvq_dim`).
    out_projs: Vec<Conv1d>,
    /// The shared `output_proj` (weight-normed 1×1, `rvq_dim → output_dim`).
    output_proj: Conv1d,
    rvq_dim: usize,
    codebook_size: usize,
}

impl RlfqDecoder {
    /// Load the first `nq` quantizers' decode tensors (`quantizer.quantizers.{i}.*` +
    /// `quantizer.output_proj.*`).
    fn load(cfg: &CodecConfig, nq: usize, vb: &VarBuilder) -> CandleResult<Self> {
        let vb_q = vb.pp("quantizer");
        let vb_qs = vb_q.pp("quantizers");
        let mut codebooks = Vec::with_capacity(nq);
        let mut out_projs = Vec::with_capacity(nq);
        for i in 0..nq {
            let vb_i = vb_qs.pp(i);
            codebooks.push(
                vb_i.pp("codebook")
                    .get((cfg.codebook_size, cfg.codebook_dim), "weight")?,
            );
            out_projs.push(wn_conv1d(
                cfg.rvq_dim,
                cfg.codebook_dim,
                &vb_i.pp("out_proj"),
            )?);
        }
        let output_proj = wn_conv1d(cfg.output_dim, cfg.rvq_dim, &vb_q.pp("output_proj"))?;
        Ok(Self {
            codebooks,
            out_projs,
            output_proj,
            rvq_dim: cfg.rvq_dim,
            codebook_size: cfg.codebook_size,
        })
    }

    /// Decode `codes` `[nq, T]` (u32 codebook ids) → `[1, output_dim, T]` — the reference
    /// `ResidualLFQ.decode_codes` with `nq` == `codes.shape[0]`.
    fn decode(&self, codes: &[Vec<u32>], device: &Device) -> CandleResult<Tensor> {
        let nq = codes.len();
        let t = codes.first().map(Vec::len).unwrap_or(0);
        let mut emb = Tensor::zeros((1, self.rvq_dim, t), DType::F32, device)?;
        for (i, code_row) in codes.iter().enumerate().take(self.out_projs.len().min(nq)) {
            // Defensive clamp: AR pad/bos/eos ids (>= codebook_size) never index a real code.
            let ids: Vec<u32> = code_row
                .iter()
                .map(|&c| c.min(self.codebook_size as u32 - 1))
                .collect();
            let ids_t = Tensor::from_vec(ids, (t,), device)?;
            // codebook lookup [T, codebook_dim] → [1, codebook_dim, T].
            let looked = self.codebooks[i].index_select(&ids_t, 0)?; // [T, dim]
            let z = looked.t()?.unsqueeze(0)?.contiguous()?; // [1, dim, T]
            let z_q = self.out_projs[i].forward(&z)?; // [1, rvq_dim, T]
            emb = (emb + z_q)?;
        }
        self.output_proj.forward(&emb) // [1, output_dim, T]
    }
}

// --------------------------------------------------------------------------------------------
// Codec transformer (causal, sliding-window, interleaved RoPE, LayerScale, gelu MLP)
// --------------------------------------------------------------------------------------------

/// Interleaved-RoPE cos/sin tables `[len, head_dim/2]` for `rope_i` (`freqs[j] = max_period^{-2j/D}`).
fn rope_tables(
    device: &Device,
    len: usize,
    head_dim: usize,
    max_period: f64,
) -> CandleResult<(Tensor, Tensor)> {
    let half = head_dim / 2;
    let mut cos = Vec::with_capacity(len * half);
    let mut sin = Vec::with_capacity(len * half);
    for pos in 0..len {
        for j in 0..half {
            let inv = max_period.powf(-(2.0 * j as f64) / head_dim as f64);
            let angle = pos as f64 * inv;
            cos.push(angle.cos() as f32);
            sin.push(angle.sin() as f32);
        }
    }
    Ok((
        Tensor::from_vec(cos, (len, half), device)?,
        Tensor::from_vec(sin, (len, half), device)?,
    ))
}

/// A `[1, 1, len, len]` additive **sliding-window** causal mask: `0` where `0 ≤ i − j < context`,
/// `−inf` elsewhere. `context ≥ len` degenerates to a plain lower-triangular causal mask.
fn banded_causal_mask(device: &Device, len: usize, context: usize) -> CandleResult<Tensor> {
    let data: Vec<f32> = (0..len)
        .flat_map(|i| {
            (0..len).map(move |j| {
                let ok = j <= i && (i - j) < context;
                if ok {
                    0.0
                } else {
                    f32::NEG_INFINITY
                }
            })
        })
        .collect();
    Tensor::from_vec(data, (1, 1, len, len), device)
}

/// One codec transformer layer.
struct CodecLayer {
    norm1: LayerNorm,
    norm2: LayerNorm,
    in_proj: Linear,  // [3*d_model, d_model], no bias — fused qkv
    out_proj: Linear, // [d_model, d_model], no bias
    linear1: Linear,  // [dim_feedforward, d_model]
    linear2: Linear,  // [d_model, dim_feedforward]
    layer_scale_1: Tensor,
    layer_scale_2: Tensor,
    num_heads: usize,
    head_dim: usize,
}

impl CodecLayer {
    fn load(spec: &StageSpec, vb: &VarBuilder) -> CandleResult<Self> {
        let d = spec.d_model;
        let head_dim = d / spec.num_heads;
        let ln = |name: &str| -> CandleResult<LayerNorm> {
            let w = vb.pp(name).get(d, "weight")?;
            let b = vb.pp(name).get(d, "bias")?;
            Ok(LayerNorm::new(w, b, 1e-5))
        };
        let lin = |vbp: VarBuilder, out: usize, inp: usize| -> CandleResult<Linear> {
            Ok(Linear::new(vbp.get((out, inp), "weight")?, None))
        };
        Ok(Self {
            norm1: ln("norm1")?,
            norm2: ln("norm2")?,
            in_proj: lin(vb.pp("self_attn").pp("in_projs").pp(0), 3 * d, d)?,
            out_proj: lin(vb.pp("self_attn").pp("out_projs").pp(0), d, d)?,
            linear1: lin(vb.pp("linear1"), spec.dim_feedforward, d)?,
            linear2: lin(vb.pp("linear2"), d, spec.dim_feedforward)?,
            layer_scale_1: vb.pp("layer_scale_1").get(d, "scale")?,
            layer_scale_2: vb.pp("layer_scale_2").get(d, "scale")?,
            num_heads: spec.num_heads,
            head_dim,
        })
    }

    /// `x` is `[1, T, d_model]`.
    fn forward(
        &self,
        x: &Tensor,
        cos: &Tensor,
        sin: &Tensor,
        mask: &Tensor,
    ) -> CandleResult<Tensor> {
        // Self-attention block: x + layer_scale_1 * attn(norm1(x)).
        let attn = self.attention(&self.norm1.forward(x)?, cos, sin, mask)?;
        let x = (x + attn.broadcast_mul(&self.layer_scale_1)?)?;
        // Feed-forward block: x + layer_scale_2 * linear2(gelu(linear1(norm2(x)))).
        let h = self.linear1.forward(&self.norm2.forward(&x)?)?;
        let h = self.linear2.forward(&h.gelu_erf()?)?;
        x.broadcast_add(&h.broadcast_mul(&self.layer_scale_2)?)
    }

    fn attention(
        &self,
        x: &Tensor,
        cos: &Tensor,
        sin: &Tensor,
        mask: &Tensor,
    ) -> CandleResult<Tensor> {
        let (b, t, _) = x.dims3()?;
        let h = self.num_heads;
        let d = self.head_dim;
        // Fused qkv → [B, T, 3, H, D] → per the reference reshape/permute.
        let qkv = self.in_proj.forward(x)?.reshape((b, t, 3, h, d))?;
        let q = qkv.i((.., .., 0))?.transpose(1, 2)?.contiguous()?; // [B, H, T, D]
        let k = qkv.i((.., .., 1))?.transpose(1, 2)?.contiguous()?;
        let v = qkv.i((.., .., 2))?.transpose(1, 2)?.contiguous()?;
        let q = candle_nn::rotary_emb::rope_i(&q, cos, sin)?;
        let k = candle_nn::rotary_emb::rope_i(&k, cos, sin)?;
        let scale = 1.0 / (d as f64).sqrt();
        let att = (q.matmul(&k.transpose(2, 3)?)? * scale)?;
        let att = att.broadcast_add(mask)?;
        let att = candle_nn::ops::softmax_last_dim(&att)?;
        let out = att.matmul(&v)?; // [B, H, T, D]
        let out = out.transpose(1, 2)?.reshape((b, t, h * d))?;
        self.out_proj.forward(&out)
    }
}

/// One `ProjectedTransformer` decoder stage: input_proj (D→d_model) · layers · output_proj
/// (d_model→output_dim), operating on `[1, D, T]` (channels-first; transposed internally).
struct CodecStage {
    input_proj: Option<Linear>,
    layers: Vec<CodecLayer>,
    output_proj: Option<Linear>,
    head_dim: usize,
    /// Sliding-window context (positions) for this stage's causal attention.
    context: usize,
}

impl CodecStage {
    fn load(spec: &StageSpec, context: usize, vb: &VarBuilder) -> CandleResult<Self> {
        let vb_s = vb.pp(format!("decoder.{}", spec.index));
        let input_proj = if spec.d_model != spec.input_dim {
            Some(Linear::new(
                vb_s.pp("input_proj")
                    .get((spec.d_model, spec.input_dim), "weight")?,
                None,
            ))
        } else {
            None
        };
        let output_proj = if spec.d_model != spec.output_dim {
            Some(Linear::new(
                vb_s.pp("output_proj")
                    .get((spec.output_dim, spec.d_model), "weight")?,
                None,
            ))
        } else {
            None
        };
        let vb_l = vb_s.pp("transformer").pp("layers");
        let mut layers = Vec::with_capacity(spec.num_layers);
        for i in 0..spec.num_layers {
            layers.push(CodecLayer::load(spec, &vb_l.pp(i))?);
        }
        Ok(Self {
            input_proj,
            layers,
            output_proj,
            head_dim: spec.d_model / spec.num_heads,
            context,
        })
    }

    /// `x` is `[1, input_dim, T]` → `[1, output_dim, T]`.
    fn forward(&self, x: &Tensor) -> CandleResult<Tensor> {
        let t = x.dim(2)?;
        // (B, D, T) -> (B, T, D)
        let mut h = x.transpose(1, 2)?.contiguous()?;
        if let Some(p) = &self.input_proj {
            h = p.forward(&h)?;
        }
        let (cos, sin) = rope_tables(h.device(), t, self.head_dim, ROPE_MAX_PERIOD)?;
        let mask = banded_causal_mask(h.device(), t, self.context)?;
        for layer in &self.layers {
            h = layer.forward(&h, &cos, &sin, &mask)?;
        }
        if let Some(p) = &self.output_proj {
            h = p.forward(&h)?;
        }
        // (B, T, D) -> (B, D, T)
        h.transpose(1, 2)?.contiguous()
    }
}

/// `PatchedPretransform.decode`: `[b, d·h, l] → [b, d, l·h]` (channels→time upsample by `h`).
fn patched_unpatch(x: &Tensor, patch: usize) -> CandleResult<Tensor> {
    let (b, dh, l) = x.dims3()?;
    if dh % patch != 0 {
        candle_audio::candle_core::bail!(
            "PatchedPretransform.decode: channels {dh} not divisible by patch {patch}"
        );
    }
    let d = dh / patch;
    // reshape(b, d, h, l).permute(0, 1, 3, 2).reshape(b, d, l*h)
    x.reshape((b, d, patch, l))?
        .permute((0, 1, 3, 2))?
        .contiguous()?
        .reshape((b, d, l * patch))
}

// --------------------------------------------------------------------------------------------
// The assembled decoder
// --------------------------------------------------------------------------------------------

/// The loaded MOSS-Audio-Tokenizer **decoder** (RLFQ decode + the 4 upsampling transformer stages).
pub struct MossAudioCodec {
    config: CodecConfig,
    quantizer: RlfqDecoder,
    stages: Vec<CodecStage>,
    /// The used quantizer count == the AR side's RVQ codebook count.
    num_code_quantizers: usize,
    device: Device,
}

impl MossAudioCodec {
    /// Load the codec decoder for `num_code_quantizers` (the AR side's `rvq`, 16) from a snapshot
    /// directory holding `config.json` + the sharded `model*.safetensors`.
    pub fn load(dir: &Path, num_code_quantizers: usize) -> Result<Self> {
        let config = CodecConfig::from_dir(dir)?;
        if num_code_quantizers == 0 || num_code_quantizers > config.num_quantizers {
            return Err(AudioError::Msg(format!(
                "codec: requested {num_code_quantizers} decode quantizers, but the codec has {}",
                config.num_quantizers
            )));
        }
        let device = candle_audio::default_device()?;
        let shards = safetensors_shards(dir)?;
        // SAFETY: mmap of provider-resolved, pinned-SHA safetensors; F32 checkpoint (config dtype
        // float32) loaded as F32. Only the decoder + quantizer tensors are touched (the encoder half
        // is never requested, so its pages stay unfaulted).
        let vb = unsafe {
            candle_nn::VarBuilder::from_mmaped_safetensors(&shards, DType::F32, &device).map_err(
                |e| AudioError::Msg(format!("codec: mmap safetensors in {}: {e}", dir.display())),
            )?
        };
        Self::from_var_builder(config, num_code_quantizers, vb, device)
    }

    fn from_var_builder(
        config: CodecConfig,
        num_code_quantizers: usize,
        vb: VarBuilder,
        device: Device,
    ) -> Result<Self> {
        let quantizer = RlfqDecoder::load(&config, num_code_quantizers, &vb)
            .map_err(|e| AudioError::Msg(format!("codec: load RLFQ decoder: {e}")))?;
        // Per-stage sliding-window context in the stage's own frame rate: the decoder starts at the
        // codec frame rate (sample_rate / downsample_rate) and each PatchedPretransform multiplies it.
        let mut frame_rate = config.sample_rate as f64 / config.downsample_rate as f64;
        let mut stages = Vec::with_capacity(STAGES.len());
        for (si, spec) in STAGES.iter().enumerate() {
            let context = (frame_rate * config.context_duration).floor() as usize;
            let context = context.max(1);
            stages.push(
                CodecStage::load(spec, context, &vb)
                    .map_err(|e| AudioError::Msg(format!("codec: load stage {si}: {e}")))?,
            );
            frame_rate *= PATCH_SIZES[si] as f64;
        }
        Ok(Self {
            config,
            quantizer,
            stages,
            num_code_quantizers,
            device,
        })
    }

    /// Native sample rate.
    pub fn sample_rate(&self) -> u32 {
        self.config.sample_rate
    }

    /// Waveform samples per RVQ frame.
    pub fn samples_per_frame(&self) -> usize {
        self.config.downsample_rate
    }

    /// The number of quantizer codebooks consumed per frame (the AR side's `rvq`).
    pub fn num_code_quantizers(&self) -> usize {
        self.num_code_quantizers
    }

    /// Decode a block of RVQ frames (`frames[f][q]` = codebook `q`'s code at frame `f`) into an
    /// interleaved mono PCM `Vec<f32>` of `frames.len() * downsample_rate` samples. `cancel` is
    /// polled once up front (each stage is a bounded matmul; the AR loop is where cancellation
    /// primarily lands).
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
        // Transpose frames [T][nq] → per-quantizer rows [nq][T] for the RLFQ decode.
        let nq = self.num_code_quantizers;
        let t = frames.len();
        let mut rows: Vec<Vec<u32>> = vec![Vec::with_capacity(t); nq];
        for frame in frames {
            for (q, row) in rows.iter_mut().enumerate() {
                // A frame shorter than nq (shouldn't happen) is padded with code 0.
                row.push(frame.get(q).copied().unwrap_or(0));
            }
        }
        let mut x = self.quantizer.decode(&rows, &self.device)?; // [1, code_dim, T]
        for (si, stage) in self.stages.iter().enumerate() {
            if cancel() {
                return Ok(None);
            }
            x = stage.forward(&x)?;
            x = patched_unpatch(&x, PATCH_SIZES[si])?;
        }
        // x is [1, 1, T*downsample_rate].
        let wav = x.i((0, 0))?.to_dtype(DType::F32)?.to_vec1::<f32>()?;
        Ok(Some(wav))
    }
}

/// The `model*.safetensors` shards of a codec snapshot (single-file or sharded), sorted for a stable
/// tensor→shard resolution.
fn safetensors_shards(dir: &Path) -> Result<Vec<std::path::PathBuf>> {
    let mut shards: Vec<std::path::PathBuf> = std::fs::read_dir(dir)
        .map_err(|e| AudioError::Msg(format!("codec: read dir {}: {e}", dir.display())))?
        .filter_map(|e| e.ok().map(|e| e.path()))
        .filter(|p| {
            p.extension().and_then(|s| s.to_str()) == Some("safetensors")
                && p.file_name()
                    .and_then(|s| s.to_str())
                    .is_some_and(|n| n.starts_with("model"))
        })
        .collect();
    shards.sort();
    if shards.is_empty() {
        return Err(AudioError::Msg(format!(
            "codec: no model*.safetensors in {}",
            dir.display()
        )));
    }
    Ok(shards)
}

#[cfg(test)]
mod tests {
    use super::*;

    const SAMPLE_CONFIG: &str = r#"{
        "architectures": ["MossAudioTokenizerModel"],
        "sample_rate": 24000,
        "downsample_rate": 1920,
        "causal_transformer_context_duration": 10,
        "quantizer_type": "rlfq",
        "quantizer_kwargs": {
            "codebook_dim": 8, "codebook_size": 1024, "input_dim": 768,
            "num_quantizers": 32, "output_dim": 768, "quantizer_type": "rlfq", "rvq_dim": 512
        }
    }"#;

    #[test]
    fn config_parses_and_validates_the_pinned_scalars() {
        let cfg = CodecConfig::from_json(SAMPLE_CONFIG).unwrap();
        assert_eq!(cfg.sample_rate, 24_000);
        assert_eq!(cfg.downsample_rate, 1920);
        assert_eq!(cfg.num_quantizers, 32);
        assert_eq!(cfg.codebook_size, 1024);
        assert_eq!(cfg.codebook_dim, 8);
        assert_eq!(cfg.rvq_dim, 512);
        assert_eq!(cfg.output_dim, 768);
        // The hardcoded patch product is the downsample rate.
        assert_eq!(PATCH_SIZES.iter().product::<usize>(), 1920);
    }

    #[test]
    fn config_rejects_a_non_rlfq_or_reshaped_codec() {
        let bad_q =
            SAMPLE_CONFIG.replace("\"rlfq\", \"rvq_dim\": 512", "\"rvq\", \"rvq_dim\": 512");
        assert!(CodecConfig::from_json(&bad_q).is_err());
        let bad_ds = SAMPLE_CONFIG.replace("\"downsample_rate\": 1920", "\"downsample_rate\": 960");
        assert!(CodecConfig::from_json(&bad_ds).is_err());
        let bad_cb = SAMPLE_CONFIG.replace("\"codebook_size\": 1024", "\"codebook_size\": 2048");
        assert!(CodecConfig::from_json(&bad_cb).is_err());
    }

    #[test]
    fn patched_unpatch_moves_channels_into_time() {
        let dev = Device::Cpu;
        // [1, 4, 2] with patch 2 → [1, 2, 4]; verify the reference reshape/permute layout.
        // channels ordered [d0h0, d0h1, d1h0, d1h1] over l=2.
        let x = Tensor::from_vec(
            (0..8).map(|i| i as f32).collect::<Vec<_>>(),
            (1, 4, 2),
            &dev,
        )
        .unwrap();
        let y = patched_unpatch(&x, 2).unwrap();
        assert_eq!(y.dims(), &[1, 2, 4]);
        // Reference: reshape(1,2,2,2).permute(0,1,3,2).reshape(1,2,4).
        // x[c,l]: c in 0..4, l in 0..2 → flat 2*c+l.
        // d=c//2, h=c%2. out[d, l*2+h] = x[2d+h, l].
        let y = y.i((0,)).unwrap().to_vec2::<f32>().unwrap();
        let x2 = x.i((0,)).unwrap().to_vec2::<f32>().unwrap();
        for d in 0..2 {
            for l in 0..2 {
                for h in 0..2 {
                    assert_eq!(y[d][l * 2 + h], x2[2 * d + h][l]);
                }
            }
        }
    }

    #[test]
    fn banded_mask_is_causal_and_windowed() {
        let dev = Device::Cpu;
        let m = banded_causal_mask(&dev, 4, 2).unwrap();
        let m = m.i((0, 0)).unwrap().to_vec2::<f32>().unwrap();
        // Row i attends j in (i-2, i]: within window → 0, else -inf.
        for (i, row) in m.iter().enumerate() {
            for (j, &val) in row.iter().enumerate() {
                let allowed = j <= i && (i - j) < 2;
                if allowed {
                    assert_eq!(val, 0.0, "({i},{j}) should be allowed");
                } else {
                    assert!(val.is_infinite() && val < 0.0, "({i},{j}) masked");
                }
            }
        }
    }

    #[test]
    fn wn_conv1d_resolves_weight_norm() {
        use std::collections::HashMap;
        let dev = Device::Cpu;
        // Seed a known g (magnitude 2/3 per output channel) and v so w = g·v/‖v‖.
        let mut ts: HashMap<String, Tensor> = HashMap::new();
        ts.insert(
            "parametrizations.weight.original0".into(),
            Tensor::from_vec(vec![2.0f32, 3.0], (2, 1, 1), &dev).unwrap(),
        );
        // v rows: [3,4] (‖‖=5) and [0,1] (‖‖=1).
        ts.insert(
            "parametrizations.weight.original1".into(),
            Tensor::from_vec(vec![3.0f32, 4.0, 0.0, 1.0], (2, 2, 1), &dev).unwrap(),
        );
        ts.insert(
            "bias".into(),
            Tensor::zeros((2,), DType::F32, &dev).unwrap(),
        );
        let vb = VarBuilder::from_tensors(ts, DType::F32, &dev);
        let conv = wn_conv1d(2, 2, &vb).unwrap();
        let w = conv
            .weight()
            .i((.., .., 0))
            .unwrap()
            .to_vec2::<f32>()
            .unwrap();
        // Row 0: 2 * [3,4]/5 = [1.2, 1.6]; Row 1: 3 * [0,1]/1 = [0, 3].
        assert!((w[0][0] - 1.2).abs() < 1e-6 && (w[0][1] - 1.6).abs() < 1e-6);
        assert!((w[1][0] - 0.0).abs() < 1e-6 && (w[1][1] - 3.0).abs() < 1e-6);
    }
}
