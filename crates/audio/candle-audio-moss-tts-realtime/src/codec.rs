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

/// The additive attention mask for one **streaming** chunk: a `[1, 1, c, cache_len + c]` band where
/// column `ki` is the concatenation `[cache (cache_len) | new (c)]` and row `qi` is a new query at
/// absolute position `pos + qi`. The cached columns hold absolute positions `pos - cache_len .. pos`
/// and the new columns `pos .. pos + c`, so the sliding-window-causal admissibility (`0 ≤ i − j <
/// context`) reduces to `qi + cache_len − context < ki ≤ qi + cache_len` — independent of `pos`. This
/// is [`banded_causal_mask`] expressed over the `[cache | new]` key layout, so the per-query key set
/// (hence the attention output) is identical to the single-shot pass.
fn streaming_band_mask(
    device: &Device,
    c: usize,
    cache_len: usize,
    context: usize,
) -> CandleResult<Tensor> {
    let cols = cache_len + c;
    let data: Vec<f32> = (0..c)
        .flat_map(|qi| {
            (0..cols).map(move |ki| {
                // Absolute positions: query = pos+qi, key = (pos-cache_len)+ki; `pos` cancels.
                let causal = ki <= qi + cache_len;
                let in_window = ki + context > qi + cache_len; // qi+cache_len - ki < context
                if causal && in_window {
                    0.0
                } else {
                    f32::NEG_INFINITY
                }
            })
        })
        .collect();
    Tensor::from_vec(data, (1, 1, c, cols), device)
}

/// One transformer layer's **streaming** KV state: the post-RoPE K and V of the most recent
/// (≤ `context − 1`) positions, carried across chunks so a chunked [`CodecStage::forward_chunked`] is
/// byte-identical to the whole-sequence [`CodecStage::forward`]. Bounded to `context − 1` positions —
/// the longest left history any query in the next chunk can attend to — so it never grows with `T`.
#[derive(Default)]
struct KvCache {
    /// `[1, H, cached_len, D]` post-RoPE keys (`None` until the first chunk).
    k: Option<Tensor>,
    /// `[1, H, cached_len, D]` values.
    v: Option<Tensor>,
}

impl KvCache {
    /// Append this chunk's post-RoPE `k`/`v` and return `(k_all, v_all, prev_cached_len)` — the full
    /// `[cache | new]` tensors to attend against, plus the pre-append cache length (the column offset
    /// of the new keys, which [`streaming_band_mask`] needs).
    fn append(&mut self, k: &Tensor, v: &Tensor) -> CandleResult<(Tensor, Tensor, usize)> {
        let (k_all, v_all, prev) = match (&self.k, &self.v) {
            (Some(pk), Some(pv)) => {
                let prev = pk.dim(2)?;
                (Tensor::cat(&[pk, k], 2)?, Tensor::cat(&[pv, v], 2)?, prev)
            }
            _ => (k.clone(), v.clone(), 0),
        };
        self.k = Some(k_all.clone());
        self.v = Some(v_all.clone());
        Ok((k_all, v_all, prev))
    }

    /// Evict all but the last `keep` cached positions, materializing the retained slice contiguously
    /// so the wide `[cache | new]` tensor from [`append`](Self::append) is released (the memory bound).
    fn retain_last(&mut self, keep: usize) -> CandleResult<()> {
        for slot in [&mut self.k, &mut self.v] {
            if let Some(t) = slot.as_mut() {
                let len = t.dim(2)?;
                if len > keep {
                    *t = t.narrow(2, len - keep, keep)?.contiguous()?;
                }
            }
        }
        Ok(())
    }
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

    /// Streaming counterpart of [`forward`](Self::forward) for a chunk of `c` new positions starting
    /// at absolute `pos` (`x` = `[1, c, d_model]`). `cos`/`sin` are the stage's **full** RoPE tables
    /// (`[T, head_dim/2]`, sliced here to `[pos, pos+c)`); `cache` carries this layer's post-RoPE K/V
    /// for the preceding positions. Every op outside attention is pointwise in time, and the attention
    /// attends exactly the sliding window [`banded_causal_mask`] would admit, so the returned chunk is
    /// byte-identical to the corresponding rows of `forward` over the whole sequence.
    fn forward_streaming(
        &self,
        x: &Tensor,
        pos: usize,
        cos: &Tensor,
        sin: &Tensor,
        context: usize,
        cache: &mut KvCache,
    ) -> CandleResult<Tensor> {
        let attn =
            self.attention_streaming(&self.norm1.forward(x)?, pos, cos, sin, context, cache)?;
        let x = (x + attn.broadcast_mul(&self.layer_scale_1)?)?;
        let h = self.linear1.forward(&self.norm2.forward(&x)?)?;
        let h = self.linear2.forward(&h.gelu_erf()?)?;
        x.broadcast_add(&h.broadcast_mul(&self.layer_scale_2)?)
    }

    /// The sliding-window self-attention of one streaming chunk (`x` = normed `[1, c, d_model]` at
    /// absolute `pos`): fused-qkv, interleaved RoPE at the chunk's absolute positions, then attend the
    /// new queries against `[cache | new]` K/V under [`streaming_band_mask`], and retain the last
    /// `context − 1` positions for the next chunk. The finite `q·k` scores are computed per key over
    /// `head_dim` (independent of the matrix width), and the mask admits exactly the single-shot
    /// window, so the result matches [`attention`](Self::attention) row-for-row.
    fn attention_streaming(
        &self,
        x: &Tensor,
        pos: usize,
        cos: &Tensor,
        sin: &Tensor,
        context: usize,
        cache: &mut KvCache,
    ) -> CandleResult<Tensor> {
        let (b, c, _) = x.dims3()?;
        let h = self.num_heads;
        let d = self.head_dim;
        let qkv = self.in_proj.forward(x)?.reshape((b, c, 3, h, d))?;
        let q = qkv.i((.., .., 0))?.transpose(1, 2)?.contiguous()?; // [b, h, c, d]
        let k = qkv.i((.., .., 1))?.transpose(1, 2)?.contiguous()?;
        let v = qkv.i((.., .., 2))?.transpose(1, 2)?.contiguous()?;
        // RoPE at the chunk's absolute positions [pos, pos+c) — the slice of the stage's full tables.
        let cos_c = cos.narrow(0, pos, c)?;
        let sin_c = sin.narrow(0, pos, c)?;
        let q = candle_nn::rotary_emb::rope_i(&q, &cos_c, &sin_c)?;
        let k = candle_nn::rotary_emb::rope_i(&k, &cos_c, &sin_c)?;
        // Attend the new queries against [cached | new] post-RoPE K/V.
        let (k_all, v_all, cache_len) = cache.append(&k, &v)?;
        let scale = 1.0 / (d as f64).sqrt();
        let att = (q.matmul(&k_all.transpose(2, 3)?)? * scale)?; // [b, h, c, cache_len + c]
        let mask = streaming_band_mask(x.device(), c, cache_len, context)?;
        let att = att.broadcast_add(&mask)?;
        let att = candle_nn::ops::softmax_last_dim(&att)?;
        let out = att.matmul(&v_all)?; // [b, h, c, d]
        let out = out.transpose(1, 2)?.reshape((b, c, h * d))?;
        cache.retain_last(context.saturating_sub(1))?;
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

    /// Memory-bounded equivalent of [`forward`](Self::forward): streams each layer over the time axis
    /// in `chunk_len`-position windows with a per-layer sliding KV cache, so the peak attention
    /// allocation is `[1, H, chunk_len, chunk_len + context]` (independent of `T`) instead of the
    /// single-shot `[1, H, T, T]`. Because the attention is sliding-window causal (window `context`)
    /// and every other op is pointwise in time, this is **byte-identical** to `forward`; `chunk_len`
    /// is a pure memory/throughput knob (any value `≥ 1` reproduces the same output). Layer-major (one
    /// KV cache alive at a time) keeps resident state to a single layer's `context − 1` positions.
    fn forward_chunked(&self, x: &Tensor, chunk_len: usize) -> CandleResult<Tensor> {
        let t = x.dim(2)?;
        let chunk_len = chunk_len.max(1);
        // (B, D, T) -> (B, T, D)
        let mut h = x.transpose(1, 2)?.contiguous()?;
        if let Some(p) = &self.input_proj {
            h = p.forward(&h)?;
        }
        // The stage's full RoPE tables; each chunk slices its own absolute-position rows out of them.
        let (cos, sin) = rope_tables(h.device(), t, self.head_dim, ROPE_MAX_PERIOD)?;
        for layer in &self.layers {
            let mut cache = KvCache::default();
            let mut outs: Vec<Tensor> = Vec::with_capacity(t.div_ceil(chunk_len));
            let mut p = 0;
            while p < t {
                let c = chunk_len.min(t - p);
                let chunk = h.narrow(1, p, c)?; // [1, c, d_model]
                outs.push(layer.forward_streaming(
                    &chunk,
                    p,
                    &cos,
                    &sin,
                    self.context,
                    &mut cache,
                )?);
                p += c;
            }
            h = Tensor::cat(&outs, 1)?; // [1, T, d_model]
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

/// How [`MossAudioCodec::encode`] runs the analysis transformer stages. All three emit
/// **byte-identical** codes; they trade peak memory for throughput on the first (~100 fps) stage.
#[derive(Clone, Copy)]
enum Chunking {
    /// Single-shot when the first stage's attention is already bounded (`T ≤ its context window`),
    /// otherwise stream — the shipped default ([`MossAudioCodec::encode`]).
    Auto,
    /// Materialize each stage's full `[1, H, T, T]` attention (fewest kernel launches; quadratic mem).
    SingleShot,
    /// Stream each stage's attention in bounded windows; `chunk_duration` seconds → per-stage chunk
    /// length via the stage frame rate (the reference `encode(chunk_duration=…)` knob).
    Chunked { chunk_duration: f64 },
}

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
    ///
    /// **Memory (sc-14181).** The first analysis stage runs at ~100 fps, so a long reference clip makes
    /// the single-shot `[1, H, T, T]` attention quadratic (a 60 s clip → T ≈ 6000 → multi-GB per layer).
    /// `encode` therefore **auto-selects**: single-shot while that attention is already bounded
    /// (`T ≤ the stage's causal context`, ≈ a 10 s clip), and the memory-bounded chunked/streaming path
    /// beyond it — byte-identical codes either way. [`encode_chunked`](Self::encode_chunked) /
    /// [`encode_single_shot`](Self::encode_single_shot) force a specific path.
    pub fn encode(&self, samples: &[f32], sample_rate: u32) -> CandleResult<Vec<Vec<u32>>> {
        self.encode_stages(samples, sample_rate, Chunking::Auto)
    }

    /// Force the **single-shot** encode (materialize each stage's full `[1, H, T, T]` attention).
    /// Fastest for short clips; quadratic in the reference length. Emits the same codes as
    /// [`encode`](Self::encode) and [`encode_chunked`](Self::encode_chunked) — exposed so callers and
    /// the conformance suite can pin one specific path (the single-shot oracle for the equivalence
    /// gate). For long clips prefer [`encode`](Self::encode) (auto) or [`encode_chunked`](Self::encode_chunked).
    pub fn encode_single_shot(
        &self,
        samples: &[f32],
        sample_rate: u32,
    ) -> CandleResult<Vec<Vec<u32>>> {
        self.encode_stages(samples, sample_rate, Chunking::SingleShot)
    }

    /// Force the **chunked/streaming** encode with a `chunk_duration`-second window (the reference
    /// `encode(chunk_duration=…)` path): each analysis stage streams its sliding-window-causal
    /// attention in bounded chunks with a per-layer KV cache, so peak memory is independent of the
    /// clip length. Byte-identical to [`encode_single_shot`](Self::encode_single_shot) for any
    /// `chunk_duration > 0` — the window is a pure memory/throughput knob, not a fidelity one.
    pub fn encode_chunked(
        &self,
        samples: &[f32],
        sample_rate: u32,
        chunk_duration: f64,
    ) -> CandleResult<Vec<Vec<u32>>> {
        self.encode_stages(samples, sample_rate, Chunking::Chunked { chunk_duration })
    }

    /// The shared encode body: resample → pad → per-stage (patch → transformer) → RLFQ-encode → trim
    /// to the reference valid length. `chunking` selects how each transformer stage runs its attention
    /// (see [`Chunking`]); it never changes the emitted codes, only the peak memory of the first stage.
    fn encode_stages(
        &self,
        samples: &[f32],
        sample_rate: u32,
        chunking: Chunking,
    ) -> CandleResult<Vec<Vec<u32>>> {
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
        // Resolve `Auto` now that the padded first-stage length is known: single-shot while the first
        // stage's attention is already bounded (`T ≤ its context`), stream a longer clip at the codec's
        // causal-context duration (a warmed chunk then spans ≤ `[1, H, context, 2·context]`).
        let chunking = match chunking {
            Chunking::Auto => {
                let stage0_t = n / ENCODER_PATCH_SIZES[0];
                if stage0_t <= enc.stages[0].context {
                    Chunking::SingleShot
                } else {
                    Chunking::Chunked {
                        chunk_duration: self.config.context_duration,
                    }
                }
            }
            explicit => explicit,
        };
        let mut x = Tensor::from_vec(buf, (1, 1, n), &self.device)?; // [1, 1, n]
                                                                     // The encoder starts at the sample rate; each PatchedPretransform (applied before its stage)
                                                                     // divides the frame rate — the same schedule `EncoderHalf::load` uses for the context window.
        let mut frame_rate = self.config.sample_rate as f64;
        for (i, stage) in enc.stages.iter().enumerate() {
            frame_rate /= ENCODER_PATCH_SIZES[i] as f64;
            x = patched_patch(&x, ENCODER_PATCH_SIZES[i])?; // downsample (the patch precedes the stage)
            x = match chunking {
                Chunking::Chunked { chunk_duration } => {
                    let chunk_len = ((frame_rate * chunk_duration).round() as usize).max(1);
                    stage.forward_chunked(&x, chunk_len)?
                }
                _ => stage.forward(&x)?, // SingleShot (Auto is resolved above)
            };
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

    // -----------------------------------------------------------------------------------------
    // sc-14181 — chunked/streaming stage equivalence. CPU, deterministic, no real weights: the
    // exact byte-identity of `CodecStage::forward` vs `forward_chunked` is the core contract of the
    // chunked encode (the codes downstream are a pointwise function of the stage output).
    // -----------------------------------------------------------------------------------------

    /// A deterministic pseudo-random tensor with entries in `[-scale, scale)` (a fixed SplitMix64-style
    /// LCG seeded by `seed`). The exact values are irrelevant — only that both code paths consume the
    /// *identical* synthetic weights, so any output divergence is attributable to the chunking alone.
    fn lcg_tensor(shape: &[usize], seed: u64, scale: f32, dev: &Device) -> Tensor {
        let n: usize = shape.iter().product();
        let mut state = seed
            .wrapping_mul(0x9E3779B97F4A7C15)
            .wrapping_add(0x1234_5678_9ABC_DEF0);
        let mut data = Vec::with_capacity(n);
        for _ in 0..n {
            state = state
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            let u = ((state >> 40) as f32) / (1u64 << 24) as f32; // [0, 2)
            data.push((u - 1.0) * scale);
        }
        Tensor::from_vec(data, shape.to_vec(), dev).unwrap()
    }

    /// Assemble a synthetic [`CodecStage`] with deterministic weights for `spec`/`module`.
    fn synthetic_stage(spec: &StageSpec, module: &str, context: usize, dev: &Device) -> CodecStage {
        use std::collections::HashMap;
        let d = spec.d_model;
        let ff = spec.dim_feedforward;
        let mut ts: HashMap<String, Tensor> = HashMap::new();
        let mut seed = 0u64;
        let mut rnd = |shape: &[usize], scale: f32| {
            seed += 1;
            lcg_tensor(shape, seed, scale, dev)
        };
        let base = format!("{module}.{}", spec.index);
        // Projections only exist when the dims differ (they do for this spec).
        ts.insert(
            format!("{base}.input_proj.weight"),
            rnd(&[d, spec.input_dim], 0.3),
        );
        ts.insert(
            format!("{base}.output_proj.weight"),
            rnd(&[spec.output_dim, d], 0.3),
        );
        for l in 0..spec.num_layers {
            let lb = format!("{base}.transformer.layers.{l}");
            // LayerNorm ~ 1.0±0.1 weight / small bias so activations stay a real, varied signal.
            ts.insert(
                format!("{lb}.norm1.weight"),
                rnd(&[d], 0.1).affine(1.0, 1.0).unwrap(),
            );
            ts.insert(format!("{lb}.norm1.bias"), rnd(&[d], 0.1));
            ts.insert(
                format!("{lb}.norm2.weight"),
                rnd(&[d], 0.1).affine(1.0, 1.0).unwrap(),
            );
            ts.insert(format!("{lb}.norm2.bias"), rnd(&[d], 0.1));
            ts.insert(
                format!("{lb}.self_attn.in_projs.0.weight"),
                rnd(&[3 * d, d], 0.3),
            );
            ts.insert(
                format!("{lb}.self_attn.out_projs.0.weight"),
                rnd(&[d, d], 0.3),
            );
            ts.insert(format!("{lb}.linear1.weight"), rnd(&[ff, d], 0.2));
            ts.insert(format!("{lb}.linear2.weight"), rnd(&[d, ff], 0.2));
            // LayerScale ~ 1.0 (vs the real 0.01 init) so the attention/FFN branches contribute
            // strongly — a wrong window or lost cache then shows up as a large divergence, not noise.
            ts.insert(
                format!("{lb}.layer_scale_1.scale"),
                rnd(&[d], 0.1).affine(1.0, 1.0).unwrap(),
            );
            ts.insert(
                format!("{lb}.layer_scale_2.scale"),
                rnd(&[d], 0.1).affine(1.0, 1.0).unwrap(),
            );
        }
        let vb = VarBuilder::from_tensors(ts, DType::F32, dev);
        CodecStage::load(spec, context, module, &vb).unwrap()
    }

    /// Largest absolute elementwise difference between two same-shaped tensors.
    fn max_abs_diff(a: &Tensor, b: &Tensor) -> f32 {
        let a = a.flatten_all().unwrap().to_vec1::<f32>().unwrap();
        let b = b.flatten_all().unwrap().to_vec1::<f32>().unwrap();
        a.iter()
            .zip(&b)
            .map(|(x, y)| (x - y).abs())
            .fold(0.0f32, f32::max)
    }

    /// The streaming band mask, restricted to a single full chunk (`cache_len = 0`, `c = len`), must
    /// equal the single-shot [`banded_causal_mask`] element-for-element — the invariant that makes the
    /// chunked attention admit exactly the single-shot window.
    #[test]
    fn streaming_band_mask_matches_single_shot_for_a_full_chunk() {
        let dev = Device::Cpu;
        for &context in &[1usize, 2, 3, 7, 40] {
            let len = 12;
            let single = banded_causal_mask(&dev, len, context)
                .unwrap()
                .i((0, 0))
                .unwrap()
                .to_vec2::<f32>()
                .unwrap();
            let stream = streaming_band_mask(&dev, len, 0, context)
                .unwrap()
                .i((0, 0))
                .unwrap()
                .to_vec2::<f32>()
                .unwrap();
            for i in 0..len {
                for j in 0..len {
                    assert_eq!(
                        single[i][j].is_finite(),
                        stream[i][j].is_finite(),
                        "mask mismatch at ({i},{j}) for context {context}"
                    );
                }
            }
        }
    }

    /// With a warmed cache, a query attends exactly its `context` most-recent keys across the
    /// `[cache | new]` boundary. Spot-check the admitted key columns against the absolute-position math
    /// the mask encodes (`qi + cache_len − context < ki ≤ qi + cache_len`).
    #[test]
    fn streaming_band_mask_windows_across_the_cache_boundary() {
        let dev = Device::Cpu;
        let (c, cache_len, context) = (3usize, 4usize, 3usize);
        let m = streaming_band_mask(&dev, c, cache_len, context)
            .unwrap()
            .i((0, 0))
            .unwrap()
            .to_vec2::<f32>()
            .unwrap();
        for (qi, row) in m.iter().enumerate() {
            for (ki, &val) in row.iter().enumerate() {
                let expected = ki <= qi + cache_len && ki + context > qi + cache_len;
                assert_eq!(
                    val.is_finite(),
                    expected,
                    "({qi},{ki}) cache_len={cache_len} context={context}"
                );
            }
        }
        // Each query sees exactly `context` keys once history is available.
        for (qi, row) in m.iter().enumerate() {
            let admitted = row.iter().filter(|v| v.is_finite()).count();
            assert_eq!(
                admitted, context,
                "query {qi} must see exactly {context} keys"
            );
        }
    }

    /// The load-bearing gate: `forward_chunked` reproduces `forward` for a multi-layer stage whose
    /// sequence is many windows long, streamed at several chunk sizes. Multi-layer + a sequence far
    /// exceeding the context exercises the cross-layer KV coupling and cache eviction that a naive
    /// block-with-left-context (receptive-field) port would get wrong. A regression in the window, the
    /// absolute RoPE offset, or the cache diverges by `O(0.1–1)` here (see the discrimination guard).
    ///
    /// The agreement is a **tight tolerance, not bit-exact**: the single-shot `[1, H, T, T]` matmul /
    /// softmax reduce over a different length than the chunked `[1, H, c, cache_len+c]` ones, so
    /// floating-point reduction order differs by `≈ 5e-6` at the stage output (observed, worst case
    /// `chunk = 1`). That is `≈ 10⁴×` smaller than any real bug and argmax-stable — so the emitted
    /// **codes** are identical, which the reference's "identical codes" guarantee and the real-weights
    /// [`moss_audio_codec_chunked_encode_matches_single_shot`](../../tests/conformance.rs) gate assert
    /// end-to-end. Widening `chunk` (the shipped default is a full `context`) shrinks the gap further.
    #[test]
    fn chunked_stage_matches_single_shot() {
        let dev = Device::Cpu;
        // input_dim≠d_model≠output_dim so both projections engage; head_dim=4; 3 layers; context=5.
        let spec = StageSpec {
            index: 0,
            input_dim: 8,
            d_model: 16,
            output_dim: 8,
            num_heads: 4,
            num_layers: 3,
            dim_feedforward: 32,
        };
        let context = 5;
        let stage = synthetic_stage(&spec, "encoder", context, &dev);
        // T = 23 ≫ context: many chunks, warm cache, eviction, a ragged final chunk.
        let t = 23;
        let x = lcg_tensor(&[1, spec.input_dim, t], 0xABCD, 1.0, &dev);
        let full = stage.forward(&x).unwrap();

        // Chunk-size invariance: 1 (max streaming, worst reduction-order gap), a mid size that doesn't
        // divide T, context, and ≥ T (one chunk). All reproduce single-shot within FP reduction noise.
        for &chunk in &[1usize, 4, context, t, t + 5] {
            let chunked = stage.forward_chunked(&x, chunk).unwrap();
            assert_eq!(full.dims(), chunked.dims());
            let diff = max_abs_diff(&full, &chunked);
            assert!(
                diff < 1e-4,
                "forward_chunked(chunk={chunk}) diverged from single-shot by {diff:.3e} — beyond FP \
                 reduction-order noise (≈5e-6); a real window/position/cache regression is O(0.1)"
            );
        }

        // Non-vacuous: the stage output is a genuinely varied, position-dependent signal (not a
        // constant the equivalence would trivially satisfy).
        let vals = full.flatten_all().unwrap().to_vec1::<f32>().unwrap();
        let mean = vals.iter().sum::<f32>() / vals.len() as f32;
        let var = vals.iter().map(|v| (v - mean).powi(2)).sum::<f32>() / vals.len() as f32;
        assert!(
            var > 1e-4,
            "degenerate stage output (var {var:.3e}) — test would be vacuous"
        );
    }

    /// Discrimination guard (mutation check): confirm the equivalence is not a tautology by proving a
    /// *wrong* left context diverges. Streaming with per-chunk state reset (a fresh cache each chunk,
    /// i.e. no cross-chunk history — the classic streaming bug) must NOT match single-shot, so the
    /// passing test above is genuinely constraining the implementation.
    #[test]
    fn chunked_stage_without_cross_chunk_state_diverges() {
        let dev = Device::Cpu;
        let spec = StageSpec {
            index: 0,
            input_dim: 8,
            d_model: 16,
            output_dim: 8,
            num_heads: 4,
            num_layers: 2,
            dim_feedforward: 32,
        };
        let context = 5;
        let stage = synthetic_stage(&spec, "encoder", context, &dev);
        let t = 20;
        let x = lcg_tensor(&[1, spec.input_dim, t], 0x51D, 1.0, &dev);
        let full = stage.forward(&x).unwrap();

        // Emulate the broken variant: reset RoPE position to 0 and drop the cache each chunk (chunks
        // treated as independent sequences). This mirrors what forward_chunked would produce if it
        // failed to carry absolute position + KV state — and must diverge from single-shot.
        let chunk = 4usize;
        let mut h = x.transpose(1, 2).unwrap().contiguous().unwrap();
        h = stage.input_proj.as_ref().unwrap().forward(&h).unwrap();
        for layer in &stage.layers {
            let mut outs: Vec<Tensor> = Vec::new();
            let mut p = 0;
            while p < t {
                let c = chunk.min(t - p);
                let chunk_x = h.narrow(1, p, c).unwrap();
                // Fresh cache + position 0 for every chunk == the bug.
                let mut cache = KvCache::default();
                let (cos, sin) = rope_tables(&dev, c, stage.head_dim, ROPE_MAX_PERIOD).unwrap();
                outs.push(
                    layer
                        .forward_streaming(&chunk_x, 0, &cos, &sin, context, &mut cache)
                        .unwrap(),
                );
                p += c;
            }
            h = Tensor::cat(&outs, 1).unwrap();
        }
        h = stage.output_proj.as_ref().unwrap().forward(&h).unwrap();
        let broken = h.transpose(1, 2).unwrap().contiguous().unwrap();
        let diff = max_abs_diff(&full, &broken);
        assert!(
            diff > 1e-3,
            "the state-reset streaming bug did NOT diverge (diff {diff:.3e}) — the equivalence test \
             would be vacuous"
        );
    }
}
