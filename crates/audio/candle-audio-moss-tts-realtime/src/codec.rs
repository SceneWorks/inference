//! The **MOSS-Audio-Tokenizer** codec (sc-13392 decode + sc-14148 encode) — RVQ speech codes ↔
//! 24 kHz waveform.
//!
//! MOSS-TTS-Realtime's AR brain ([`crate::decode`]) only emits discrete RVQ codes; turning those
//! into a waveform is the job of a **separate** ~7.1 GB model, `OpenMOSS-Team/MOSS-Audio-Tokenizer`
//! (Apache-2.0), a novel **RLFQ streaming codec** (config `quantizer_type=rlfq`, 32 quantizers,
//! codebook 1024, `codebook_dim=8`, `rvq_dim=512`). It is a Moshi/Mimi-scale, **CNN-free** codec:
//! its decoder is a stack of causal RoPE transformers interleaved with `PatchedPretransform`
//! channel→time upsamplers, and its encoder the analysis mirror (time→channel downsamplers +
//! transformers → residual-LFQ quantize). This module ports **both directions** natively onto the
//! pinned candle revision, faithful to the reference `modeling_moss_audio_tokenizer.py`: the decode
//! path (sc-13392, the TTS direction) and the encode path (sc-14148, waveform → codes for voice
//! cloning). The encoder half loads lazily, so a pure-TTS decode never faults its weight pages.
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
    /// Load a `ProjectedTransformer` stage from `{module}.{index}` — `module` is `"decoder"` (the
    /// upsampling synthesis stack) or `"encoder"` (the downsampling analysis stack). Both share the
    /// identical layer architecture; only the weight namespace and the per-stage dims differ.
    fn load(spec: &StageSpec, context: usize, module: &str, vb: &VarBuilder) -> CandleResult<Self> {
        let vb_s = vb.pp(format!("{module}.{}", spec.index));
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
// The encoder (waveform → RVQ codes) — sc-14148
// --------------------------------------------------------------------------------------------

/// `PatchedPretransform.encode`: `[b, d, l] → [b, d·patch, l/patch]` (time→channels downsample by
/// `patch`) — the inverse of [`patched_unpatch`], mirroring the reference
/// `MossAudioTokenizerPatchedPretransform.encode`.
fn patched_patch(x: &Tensor, patch: usize) -> CandleResult<Tensor> {
    let (b, d, l) = x.dims3()?;
    if l % patch != 0 {
        candle_audio::candle_core::bail!(
            "PatchedPretransform.encode: time {l} not divisible by patch {patch}"
        );
    }
    // reshape(b, d, l/patch, patch).permute(0, 1, 3, 2).reshape(b, d*patch, l/patch)
    x.reshape((b, d, l / patch, patch))?
        .permute((0, 1, 3, 2))?
        .contiguous()?
        .reshape((b, d * patch, l / patch))
}

/// The four **encoder** Transformer stages (`encoder.{1,3,5,7}`) — the analysis mirror of the decoder
/// [`STAGES`], read from `config.encoder_kwargs`. Each is preceded by an [`ENCODER_PATCH_SIZES`]
/// downsampler; ∏ of the patch sizes = [`DOWNSAMPLE_RATE`] (24 kHz waveform → 12.5 fps code frames).
#[rustfmt::skip]
const ENCODER_STAGES: [StageSpec; 4] = [
    StageSpec { index: 1, input_dim: 240,  d_model: 768,  output_dim: 384, num_heads: 12, num_layers: 12, dim_feedforward: 3072 },
    StageSpec { index: 3, input_dim: 768,  d_model: 768,  output_dim: 384, num_heads: 12, num_layers: 12, dim_feedforward: 3072 },
    StageSpec { index: 5, input_dim: 768,  d_model: 768,  output_dim: 640, num_heads: 12, num_layers: 12, dim_feedforward: 3072 },
    StageSpec { index: 7, input_dim: 1280, d_model: 1280, output_dim: 768, num_heads: 20, num_layers: 32, dim_feedforward: 5120 },
];

/// Patch sizes of `encoder.{0,2,4,6}` — the downsamplers applied **before** each encoder Transformer
/// stage (∏ = [`DOWNSAMPLE_RATE`]).
const ENCODER_PATCH_SIZES: [usize; 4] = [240, 2, 2, 2];

/// L2-normalize each row of a `[n, d]` matrix (the reference `F.normalize`, default eps 1e-12).
fn l2_normalize_rows(x: &Tensor) -> CandleResult<Tensor> {
    let norm = x.sqr()?.sum_keepdim(1)?.sqrt()?;
    x.broadcast_div(&norm.clamp(1e-12f32, f32::INFINITY)?)
}

/// The **encode** side of the Residual-LFQ quantizer: the shared `input_proj` (output_dim → rvq_dim),
/// per-quantizer `in_proj` (rvq_dim → codebook_dim) + `codebook` + `out_proj` (codebook_dim → rvq_dim).
/// Emits `nq` codebook-index rows (the reference `MossAudioTokenizerResidualLFQ.forward` inference
/// path: an L2-normalized argmin over each residual, then subtract the RAW selected codebook vector).
struct RlfqEncoder {
    input_proj: Conv1d,
    codebooks: Vec<Tensor>,
    in_projs: Vec<Conv1d>,
    out_projs: Vec<Conv1d>,
    nq: usize,
}

impl RlfqEncoder {
    fn load(cfg: &CodecConfig, nq: usize, vb: &VarBuilder) -> CandleResult<Self> {
        let vb_q = vb.pp("quantizer");
        // `input_proj` lifts the encoder output into the residual working dim (output_dim → rvq_dim);
        // the reference's quantizer `input_dim` == its `output_dim` (== CODE_DIM, config-validated), so
        // `output_dim` is the correct in-channel count and equals `ENCODER_STAGES`'s last output.
        let input_proj = wn_conv1d(cfg.rvq_dim, cfg.output_dim, &vb_q.pp("input_proj"))?;
        let vb_qs = vb_q.pp("quantizers");
        let mut codebooks = Vec::with_capacity(nq);
        let mut in_projs = Vec::with_capacity(nq);
        let mut out_projs = Vec::with_capacity(nq);
        for i in 0..nq {
            let vb_i = vb_qs.pp(i);
            codebooks.push(
                vb_i.pp("codebook")
                    .get((cfg.codebook_size, cfg.codebook_dim), "weight")?,
            );
            in_projs.push(wn_conv1d(
                cfg.codebook_dim,
                cfg.rvq_dim,
                &vb_i.pp("in_proj"),
            )?);
            out_projs.push(wn_conv1d(
                cfg.rvq_dim,
                cfg.codebook_dim,
                &vb_i.pp("out_proj"),
            )?);
        }
        Ok(Self {
            input_proj,
            codebooks,
            in_projs,
            out_projs,
            nq,
        })
    }

    /// Encode `e` `[1, output_dim, T]` (the encoder's last hidden state) → `nq` code rows `[nq][T]`.
    fn encode(&self, e: &Tensor) -> CandleResult<Vec<Vec<u32>>> {
        let mut residual = self.input_proj.forward(e)?; // [1, rvq_dim, T]
        let mut codes: Vec<Vec<u32>> = Vec::with_capacity(self.nq);
        for i in 0..self.nq {
            // Nearest code by argmin ‖e − c‖² over L2-normalized vectors, which equals argmax cosine:
            // the ‖e‖² (per-position) and ‖c‖² (== 1 per normalized code) terms are constant across
            // codes, so they do not move the argmin — identical to the reference `(-dist).max(1)`.
            let z_e = self.in_projs[i].forward(&residual)?; // [1, codebook_dim, T]
            let enc = z_e.i(0)?.t()?.contiguous()?; // [T, codebook_dim]
            let enc_n = l2_normalize_rows(&enc)?; // [T, codebook_dim] rows
            let cb_n = l2_normalize_rows(&self.codebooks[i])?; // [codebook_size, dim] rows
            let sim = enc_n.matmul(&cb_n.t()?)?; // [T, codebook_size]
            let idx = sim.argmax(1)?; // [T]
            let ids: Vec<u32> = idx.to_dtype(DType::U32)?.to_vec1::<u32>()?;
            // Subtract the RAW (un-normalized) selected codebook vector's out_proj from the residual.
            let looked = self.codebooks[i].index_select(&idx, 0)?; // [T, codebook_dim]
            let z_q = self.out_projs[i].forward(&looked.t()?.unsqueeze(0)?.contiguous()?)?; // [1, rvq_dim, T]
            residual = (residual - z_q)?;
            codes.push(ids);
        }
        Ok(codes)
    }
}

/// The loaded **encoder** half (analysis stages + the RLFQ encode) — built lazily on first
/// [`MossAudioCodec::encode`] so a pure-decode (TTS) load never faults the encoder's weight pages.
struct EncoderHalf {
    stages: Vec<CodecStage>,
    rlfq: RlfqEncoder,
}

impl EncoderHalf {
    fn load(cfg: &CodecConfig, nq: usize, vb: &VarBuilder) -> CandleResult<Self> {
        // Per-stage sliding-window context: the encoder starts at the sample rate and each
        // preceding PatchedPretransform divides the frame rate (the reverse of the decoder), matching
        // the reference `int(current_frame_rate * ctx_dur)` with `frame_rate /= downsample_ratio`.
        let mut frame_rate = cfg.sample_rate as f64;
        let mut stages = Vec::with_capacity(ENCODER_STAGES.len());
        for (si, spec) in ENCODER_STAGES.iter().enumerate() {
            frame_rate /= ENCODER_PATCH_SIZES[si] as f64; // the patch precedes this stage
            let context = ((frame_rate * cfg.context_duration).floor() as usize).max(1);
            stages.push(CodecStage::load(spec, context, "encoder", vb)?);
        }
        Ok(Self {
            stages,
            rlfq: RlfqEncoder::load(cfg, nq, vb)?,
        })
    }
}

/// Minimal linear-interpolation resample to the codec's native rate. Reference-audio preprocessing is
/// provider-owned (candle-audio's `wav` module notes the input path resamples here, not in the codec);
/// exact-rate inputs bypass it. A higher-fidelity resampler can be swapped in without touching callers.
fn resample_linear(samples: &[f32], from: u32, to: u32) -> Vec<f32> {
    if from == to || samples.is_empty() {
        return samples.to_vec();
    }
    let ratio = to as f64 / from as f64;
    let out_len = ((samples.len() as f64) * ratio).round().max(1.0) as usize;
    let last = samples.len() - 1;
    (0..out_len)
        .map(|i| {
            let src = i as f64 / ratio;
            let j = src.floor() as usize;
            let frac = (src - j as f64) as f32;
            let a = samples[j.min(last)];
            let b = samples[(j + 1).min(last)];
            a + (b - a) * frac
        })
        .collect()
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
    /// The snapshot's safetensors shards, retained so the encoder half can be mmapped lazily on the
    /// first [`encode`](Self::encode) without re-resolving the snapshot (voice cloning, sc-14149).
    shards: Vec<std::path::PathBuf>,
    /// The encoder half (analysis stages + RLFQ encode), built lazily on first `encode` — a pure
    /// decode (TTS) load never touches the encoder's weight pages.
    encoder: std::sync::OnceLock<EncoderHalf>,
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
        Self::from_var_builder(config, num_code_quantizers, shards, vb, device)
    }

    fn from_var_builder(
        config: CodecConfig,
        num_code_quantizers: usize,
        shards: Vec<std::path::PathBuf>,
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
                CodecStage::load(spec, context, "decoder", &vb)
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
            shards,
            encoder: std::sync::OnceLock::new(),
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
        let debug = std::env::var_os("MOSS_CODEC_DEBUG").is_some();
        let mut x = self.quantizer.decode(&rows, &self.device)?; // [1, code_dim, T]
        if debug {
            eprintln!("[codec] after quantizer: {:?} rms={:.5}", x.dims(), rms(&x));
        }
        for (si, stage) in self.stages.iter().enumerate() {
            if cancel() {
                return Ok(None);
            }
            x = stage.forward(&x)?;
            if debug {
                eprintln!(
                    "[codec] after stage {si} (before unpatch): {:?} rms={:.5}",
                    x.dims(),
                    rms(&x)
                );
            }
            x = patched_unpatch(&x, PATCH_SIZES[si])?;
        }
        // x is [1, 1, T*downsample_rate].
        let wav = x.i((0, 0))?.to_dtype(DType::F32)?.to_vec1::<f32>()?;
        Ok(Some(wav))
    }

    /// Encode a mono waveform into the codec's `num_code_quantizers` RVQ codes per frame — the
    /// analysis direction (waveform → tokens) for voice cloning (sc-14149). `samples` at
    /// `sample_rate` are resampled to the codec's native rate, right-padded to a multiple of
    /// `downsample_rate`, run through the encoder analysis stages, then residual-LFQ quantized.
    /// Returns `frames[f][q]` (the same layout [`decode_frames`](Self::decode_frames) consumes), so an
    /// encode→decode round-trip reconstructs the input.
    ///
    /// The output is trimmed to the **valid** frame count `⌊samples / downsample_rate⌋` (at the native
    /// rate) — the reference reports exactly this as `audio_codes_lengths` and masks the residual over
    /// the trailing padded frame, so the final (part-padding) frame it would otherwise emit is a
    /// masked artifact, not real audio. Trimming keeps every returned frame faithful to the reference's
    /// valid codes (a clip shorter than one frame yields no codes, as in the reference).
    pub fn encode(&self, samples: &[f32], sample_rate: u32) -> CandleResult<Vec<Vec<u32>>> {
        if samples.is_empty() {
            return Ok(Vec::new());
        }
        let wav = resample_linear(samples, sample_rate, self.config.sample_rate);
        let dr = self.config.downsample_rate;
        let valid_frames = wav.len() / dr; // ⌊T / downsample_rate⌋ — the reference valid length
        if valid_frames == 0 {
            return Ok(Vec::new());
        }
        let enc = self.encoder_half()?;
        let t = wav.len();
        let pad = (dr - t % dr) % dr;
        let mut buf = wav;
        buf.resize(t + pad, 0.0);
        let n = buf.len();
        let mut x = Tensor::from_vec(buf, (1, 1, n), &self.device)?; // [1, 1, n]
        for (i, stage) in enc.stages.iter().enumerate() {
            x = patched_patch(&x, ENCODER_PATCH_SIZES[i])?; // downsample (the patch precedes the stage)
            x = stage.forward(&x)?;
        }
        // x is [1, output_dim, ⌈T/dr⌉]; RLFQ-encode to per-quantizer rows, then take the valid prefix.
        let rows = enc.rlfq.encode(&x)?;
        // Transpose [nq][T] → frames [T][nq] (the decode_frames layout), trimmed to the valid length.
        let frames = (0..valid_frames)
            .map(|f| rows.iter().map(|r| r[f]).collect())
            .collect();
        Ok(frames)
    }

    /// Get-or-build the encoder half, mmapping the retained shards on first use. Thread-safe and
    /// idempotent (a lost init race just discards the redundant build).
    fn encoder_half(&self) -> CandleResult<&EncoderHalf> {
        if let Some(e) = self.encoder.get() {
            return Ok(e);
        }
        // SAFETY: same idiom as `load` — mmap of the provider-resolved, pinned safetensors as F32.
        let vb = unsafe {
            candle_nn::VarBuilder::from_mmaped_safetensors(&self.shards, DType::F32, &self.device)?
        };
        let built = EncoderHalf::load(&self.config, self.num_code_quantizers, &vb)?;
        Ok(self.encoder.get_or_init(|| built))
    }
}

impl crate::chunk::PrefixDecoder for MossAudioCodec {
    fn sample_rate(&self) -> u32 {
        self.config.sample_rate
    }

    fn samples_per_frame(&self) -> usize {
        self.config.downsample_rate
    }

    /// Decode a growing RVQ-frame prefix into its full mono PCM (the causal decode the streaming
    /// chunker relies on). Delegates to [`MossAudioCodec::decode_frames`], so a prefix decode is
    /// byte-identical to the head of any longer decode.
    fn decode_prefix(
        &self,
        frames: &[Vec<u32>],
        cancel: &dyn Fn() -> bool,
    ) -> CandleResult<Option<Vec<f32>>> {
        self.decode_frames(frames, cancel)
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
    fn patched_patch_inverts_unpatch() {
        let dev = Device::Cpu;
        // The encode-direction patch `[1, d, l] → [1, d·h, l/h]` is the exact inverse of the decode
        // `patched_unpatch`, so `unpatch(patch(x)) == x` (the reshape/permute pair cancels).
        let x = Tensor::from_vec(
            (0..24).map(|i| i as f32).collect::<Vec<_>>(),
            (1, 2, 12),
            &dev,
        )
        .unwrap();
        let down = patched_patch(&x, 3).unwrap();
        assert_eq!(down.dims(), &[1, 6, 4], "time→channels downsample by 3");
        let back = patched_unpatch(&down, 3).unwrap();
        assert_eq!(back.dims(), x.dims());
        assert_eq!(
            back.i((0,)).unwrap().to_vec2::<f32>().unwrap(),
            x.i((0,)).unwrap().to_vec2::<f32>().unwrap(),
            "unpatch ∘ patch is the identity"
        );
        // A time length not divisible by the patch is a typed error, not a silent reshape.
        assert!(patched_patch(&x, 5).is_err());
    }

    #[test]
    fn l2_normalize_rows_gives_unit_norm_and_survives_zero() {
        let dev = Device::Cpu;
        // [3,4] → /5 = [0.6,0.8]; [0,0] → stays [0,0] (eps clamp, no NaN); [1,0] already unit.
        let x = Tensor::from_vec(vec![3.0f32, 4.0, 0.0, 0.0, 1.0, 0.0], (3, 2), &dev).unwrap();
        let n = l2_normalize_rows(&x).unwrap().to_vec2::<f32>().unwrap();
        assert!((n[0][0] - 0.6).abs() < 1e-5 && (n[0][1] - 0.8).abs() < 1e-5);
        assert!(n[1].iter().all(|v| v.is_finite()) && n[1] == vec![0.0, 0.0]);
        assert!((n[2][0] - 1.0).abs() < 1e-5 && n[2][1].abs() < 1e-5);
    }

    #[test]
    fn resample_linear_identity_ratio_and_edges() {
        let x = vec![0.0f32, 1.0, 2.0, 3.0];
        assert_eq!(
            resample_linear(&x, 24_000, 24_000),
            x,
            "identity at native rate"
        );
        assert_eq!(resample_linear(&x, 24_000, 48_000).len(), 8, "2× upsample");
        assert_eq!(
            resample_linear(&x, 48_000, 24_000).len(),
            2,
            "2× downsample"
        );
        assert!(
            resample_linear(&[], 48_000, 24_000).is_empty(),
            "empty stays empty"
        );
        assert_eq!(
            resample_linear(&[5.0], 48_000, 24_000),
            vec![5.0],
            "single sample never panics"
        );
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
