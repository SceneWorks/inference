//! The ACE-Step 1.5 **FSQ audio tokenizer + detokenizer** (sc-13251) — the cover-task acoustic
//! round-trip, ported from the diffusers v0.39.0 reference (`modeling_ace_step.py`,
//! `AceStepAudioTokenizer` / `AceStepAudioTokenDetokenizer`).
//!
//! These two small modules are the *only* weights the prompted-audio **Cover** mode needs beyond
//! the text-to-music pipeline. They do NOT ship in the pinned turbo checkpoint
//! (`acestep-v15-xl-turbo-diffusers`); they ship in the sibling MIT checkpoint
//! `acestep-v15-xl-sft-diffusers` (same org, same 64-ch/25 Hz acoustic latent space). The Cover
//! path pins those two component dirs and reuses the already-loaded turbo DiT / condition encoder /
//! VAE for the restyle (the reference pipeline's own `is_turbo` cover path — guidance forced to 1,
//! single forward), so the shipped Inpaint/Repaint/Extend fast modes are unchanged.
//!
//! ## The round-trip (reference `prepare_src_latents(task_type="cover")`)
//!
//! ```text
//!   src_lat = vae.encode(source)                       # [1, L@25Hz, 64]  (deterministic mean)
//!   quantized, _ = audio_tokenizer.tokenize(src_lat, silence_latent)   # [1, L/5@5Hz, 2048]
//!   src_latents  = audio_token_detokenizer(quantized)  # [1, (L/5)*5@25Hz, 64], cropped to L
//! ```
//!
//! `audio_tokenizer` pools each `pool_window_size`(5)-frame window of the 25 Hz acoustic latents to
//! one 5 Hz token via a learned-query attention pooler, projects to the FSQ codebook, and
//! Finite-Scalar-Quantizes (levels `[8,8,8,5,5,5]` → a ≈64 K codebook); `audio_token_detokenizer`
//! expands each 5 Hz token back to 5 acoustic frames through a mirror transformer. The quantization
//! bottleneck is what makes Cover a *restyle*: it discards fine timbre while preserving musical
//! structure, and the new prompt supplies the new timbre through the DiT cross-attention context.
//!
//! ## Shared primitives
//!
//! The pooler and detokenizer stack the same `AceStepEncoderLayer` (pre-LN, GQA + per-head q/k
//! RMSNorm, half-split RoPE, SwiGLU MLP) the DiT self-attention and the condition-encoder use — so
//! the layer here mirrors [`crate::dit`]/[`crate::condition`]. Both stacks run over a **tiny**
//! sequence (`pool_window_size`+1 = 6 for the pooler, `pool_window_size` = 5 for the detokenizer),
//! far below the `sliding_window` (128) with `is_causal=False`, so the reference's sliding-window
//! mask is a no-op at these lengths and full bidirectional attention is exact.

use std::path::Path;

use candle_audio::candle_core::{DType, Device, Result as CandleResult, Tensor};
use candle_audio::{AudioError, Result};
use candle_nn::{linear, linear_b, rms_norm, Linear, Module, RmsNorm, VarBuilder};
use serde::Deserialize;

/// Shared transformer hyperparameters for the pooler / detokenizer stacks (the fields both
/// `audio_tokenizer/config.json` and `audio_token_detokenizer/config.json` carry).
#[derive(Debug, Clone, Deserialize)]
pub struct TokenizerConfig {
    /// Transformer width (2048 — equal to the DiT `encoder_hidden_size`).
    pub hidden_size: usize,
    pub intermediate_size: usize,
    pub head_dim: usize,
    pub num_attention_heads: usize,
    pub num_key_value_heads: usize,
    #[serde(default = "default_pooler_layers")]
    pub num_attention_pooler_hidden_layers: usize,
    /// The 25 Hz → 5 Hz pooling factor (5).
    pub pool_window_size: usize,
    /// Acoustic latent channels the tokenizer consumes / the detokenizer emits (64 — the VAE dim).
    pub audio_acoustic_hidden_dim: usize,
    pub rms_norm_eps: f64,
    pub rope_theta: f64,
    /// The FSQ per-dimension levels (`[8,8,8,5,5,5]`) — present only in the tokenizer config.
    #[serde(default)]
    pub fsq_input_levels: Vec<usize>,
    /// FSQ projection width (2048) — present only in the tokenizer config.
    #[serde(default)]
    pub fsq_dim: usize,
    #[serde(default = "default_num_quantizers")]
    pub fsq_input_num_quantizers: usize,
}

fn default_pooler_layers() -> usize {
    2
}
fn default_num_quantizers() -> usize {
    1
}

impl TokenizerConfig {
    pub fn from_json(path: &Path) -> Result<Self> {
        let text = std::fs::read_to_string(path)
            .map_err(|e| AudioError::Msg(format!("read {}: {e}", path.display())))?;
        serde_json::from_str(&text)
            .map_err(|e| AudioError::Msg(format!("parse {}: {e}", path.display())))
    }

    fn validate(&self) -> Result<()> {
        if !self
            .num_attention_heads
            .is_multiple_of(self.num_key_value_heads)
        {
            return Err(AudioError::Msg(format!(
                "acestep tokenizer: heads {} not a multiple of kv heads {}",
                self.num_attention_heads, self.num_key_value_heads
            )));
        }
        if self.pool_window_size == 0 {
            return Err(AudioError::Msg(
                "acestep tokenizer: pool_window_size must be > 0".into(),
            ));
        }
        Ok(())
    }
}

fn to_heads(x: &Tensor, num_heads: usize, head_dim: usize) -> CandleResult<Tensor> {
    let (b, l, _) = x.dims3()?;
    x.reshape((b, l, num_heads, head_dim))?
        .transpose(1, 2)?
        .contiguous()
}

fn from_heads(x: &Tensor) -> CandleResult<Tensor> {
    let (b, h, l, d) = x.dims4()?;
    x.transpose(1, 2)?.reshape((b, l, h * d))
}

fn repeat_kv(x: &Tensor, groups: usize) -> CandleResult<Tensor> {
    if groups == 1 {
        return x.contiguous();
    }
    let (b, h, l, d) = x.dims4()?;
    x.unsqueeze(2)?
        .expand((b, h, groups, l, d))?
        .reshape((b, h * groups, l, d))?
        .contiguous()
}

/// Half-split RoPE tables `[len, head_dim/2]` — the same convention as [`crate::dit`] /
/// [`crate::condition`] (candle `rope`, the `use_real_unbind_dim=-2` branch).
fn rope_tables(
    head_dim: usize,
    len: usize,
    theta: f64,
    device: &Device,
) -> CandleResult<(Tensor, Tensor)> {
    let half = head_dim / 2;
    let mut cos = Vec::with_capacity(len * half);
    let mut sin = Vec::with_capacity(len * half);
    for pos in 0..len {
        for j in 0..half {
            let inv = 1.0 / theta.powf(2.0 * j as f64 / head_dim as f64);
            let a = pos as f64 * inv;
            cos.push(a.cos() as f32);
            sin.push(a.sin() as f32);
        }
    }
    Ok((
        Tensor::from_vec(cos, (len, half), device)?,
        Tensor::from_vec(sin, (len, half), device)?,
    ))
}

/// One `AceStepEncoderLayer` (pre-LN, GQA + per-head q/k RMSNorm, RoPE, SwiGLU MLP) — the same
/// block the DiT self-attention and the condition encoder use, over full bidirectional attention
/// (the pooler/detokenizer sequences are shorter than the sliding window, so no mask applies).
struct EncoderLayer {
    input_layernorm: RmsNorm,
    to_q: Linear,
    to_k: Linear,
    to_v: Linear,
    to_out: Linear,
    norm_q: RmsNorm,
    norm_k: RmsNorm,
    post_attention_layernorm: RmsNorm,
    gate_proj: Linear,
    up_proj: Linear,
    down_proj: Linear,
    num_heads: usize,
    num_kv_heads: usize,
    head_dim: usize,
    theta: f64,
}

impl EncoderLayer {
    fn new(cfg: &TokenizerConfig, vb: VarBuilder) -> CandleResult<Self> {
        let d = cfg.head_dim;
        let hidden = cfg.hidden_size;
        let sa = vb.pp("self_attn");
        Ok(Self {
            input_layernorm: rms_norm(hidden, cfg.rms_norm_eps, vb.pp("input_layernorm"))?,
            to_q: linear_b(hidden, cfg.num_attention_heads * d, false, sa.pp("to_q"))?,
            to_k: linear_b(hidden, cfg.num_key_value_heads * d, false, sa.pp("to_k"))?,
            to_v: linear_b(hidden, cfg.num_key_value_heads * d, false, sa.pp("to_v"))?,
            to_out: linear_b(
                cfg.num_attention_heads * d,
                hidden,
                false,
                sa.pp("to_out.0"),
            )?,
            norm_q: rms_norm(d, cfg.rms_norm_eps, sa.pp("norm_q"))?,
            norm_k: rms_norm(d, cfg.rms_norm_eps, sa.pp("norm_k"))?,
            post_attention_layernorm: rms_norm(
                hidden,
                cfg.rms_norm_eps,
                vb.pp("post_attention_layernorm"),
            )?,
            gate_proj: linear_b(hidden, cfg.intermediate_size, false, vb.pp("mlp.gate_proj"))?,
            up_proj: linear_b(hidden, cfg.intermediate_size, false, vb.pp("mlp.up_proj"))?,
            down_proj: linear_b(cfg.intermediate_size, hidden, false, vb.pp("mlp.down_proj"))?,
            num_heads: cfg.num_attention_heads,
            num_kv_heads: cfg.num_key_value_heads,
            head_dim: d,
            theta: cfg.rope_theta,
        })
    }

    fn forward(&self, x: &Tensor) -> CandleResult<Tensor> {
        let device = x.device();
        let len = x.dim(1)?;
        let h = self.input_layernorm.forward(x)?;
        let q = self.norm_q.forward(&to_heads(
            &self.to_q.forward(&h)?,
            self.num_heads,
            self.head_dim,
        )?)?;
        let k = self.norm_k.forward(&to_heads(
            &self.to_k.forward(&h)?,
            self.num_kv_heads,
            self.head_dim,
        )?)?;
        let v = to_heads(&self.to_v.forward(&h)?, self.num_kv_heads, self.head_dim)?;
        let (cos, sin) = rope_tables(self.head_dim, len, self.theta, device)?;
        let q = candle_nn::rotary_emb::rope(&q.contiguous()?, &cos, &sin)?;
        let k = candle_nn::rotary_emb::rope(&k.contiguous()?, &cos, &sin)?;
        let groups = self.num_heads / self.num_kv_heads;
        let k = repeat_kv(&k, groups)?;
        let v = repeat_kv(&v, groups)?;
        let scale = 1.0 / (self.head_dim as f64).sqrt();
        let att = candle_nn::ops::softmax_last_dim(&(q.matmul(&k.transpose(2, 3)?)? * scale)?)?;
        let attn = self
            .to_out
            .forward(&from_heads(&att.matmul(&v.contiguous()?)?)?)?;
        let x = (x + attn)?;
        let h = self.post_attention_layernorm.forward(&x)?;
        let ff = self
            .down_proj
            .forward(&(self.gate_proj.forward(&h)?.silu()? * self.up_proj.forward(&h)?)?)?;
        x + ff
    }
}

/// The `AceStepAttentionPooler`: prepend a learned query token to each `pool_window_size`-frame
/// window, run the encoder stack, and read out the query token — pooling 5 frames to 1 token.
struct AttentionPooler {
    embed_tokens: Linear,
    special_token: Tensor, // [1, 1, hidden]
    layers: Vec<EncoderLayer>,
    norm: RmsNorm,
    hidden: usize,
}

impl AttentionPooler {
    fn new(cfg: &TokenizerConfig, vb: VarBuilder) -> CandleResult<Self> {
        let h = cfg.hidden_size;
        let embed_tokens = linear(h, h, vb.pp("embed_tokens"))?;
        let special_token = vb.get((1, 1, h), "special_token")?;
        let vb_l = vb.pp("layers");
        let mut layers = Vec::with_capacity(cfg.num_attention_pooler_hidden_layers);
        for i in 0..cfg.num_attention_pooler_hidden_layers {
            layers.push(EncoderLayer::new(cfg, vb_l.pp(i))?);
        }
        let norm = rms_norm(h, cfg.rms_norm_eps, vb.pp("norm"))?;
        Ok(Self {
            embed_tokens,
            special_token,
            layers,
            norm,
            hidden: h,
        })
    }

    /// `[num_patches, window, hidden]` (the windows as a batch) → `[num_patches, hidden]`.
    fn forward(&self, windows: &Tensor) -> CandleResult<Tensor> {
        let (num_patches, _window, _h) = windows.dims3()?;
        let embedded = self.embed_tokens.forward(windows)?;
        let special = self
            .special_token
            .broadcast_as((num_patches, 1, self.hidden))?
            .contiguous()?;
        let mut x = Tensor::cat(&[&special, &embedded], 1)?; // [num_patches, window+1, hidden]
        for l in &self.layers {
            x = l.forward(&x)?;
        }
        let x = self.norm.forward(&x)?;
        x.narrow(1, 0, 1)?.squeeze(1) // the query token: [num_patches, hidden]
    }
}

/// The `_AceStepResidualFSQ` bottleneck (`num_quantizers = 1`): project to the codebook dim,
/// soft-clamp, finite-scalar-quantize (per-dimension `levels`), project back. Only the projected
/// output (the acoustic conditioning) is needed for cover, so indices are not returned.
struct ResidualFsq {
    project_in: Linear,
    project_out: Linear,
    levels: Tensor, // [codebook_dim] f32
}

impl ResidualFsq {
    fn new(cfg: &TokenizerConfig, vb: VarBuilder) -> CandleResult<Self> {
        let levels: Vec<usize> = if cfg.fsq_input_levels.is_empty() {
            vec![8, 8, 8, 5, 5, 5]
        } else {
            cfg.fsq_input_levels.clone()
        };
        let codebook_dim = levels.len();
        let fsq_dim = if cfg.fsq_dim == 0 {
            cfg.hidden_size
        } else {
            cfg.fsq_dim
        };
        let project_in = linear(fsq_dim, codebook_dim, vb.pp("project_in"))?;
        let project_out = linear(codebook_dim, fsq_dim, vb.pp("project_out"))?;
        let levels_f: Vec<f32> = levels.iter().map(|&l| l as f32).collect();
        let levels = Tensor::from_vec(levels_f, codebook_dim, vb.device())?;
        Ok(Self {
            project_in,
            project_out,
            levels,
        })
    }

    /// `x [.., fsq_dim]` → quantized `[.., fsq_dim]`. Mirrors the reference `forward` with
    /// `num_quantizers = 1` (`scales = [1.0]`), returning `project_out(quantize(soft_clamp(...)))`.
    fn forward(&self, x: &Tensor) -> CandleResult<Tensor> {
        let h = self.project_in.forward(x)?; // [.., codebook_dim]
                                             // soft_clamp = 1 + 1/(levels - 1); h = tanh(h/soft_clamp) * soft_clamp.
        let ones = Tensor::ones_like(&self.levels)?;
        let levels_minus_one = (&self.levels - &ones)?;
        let soft_clamp = (&ones + ones.broadcast_div(&levels_minus_one)?)?; // [codebook_dim]
        let h = h
            .broadcast_div(&soft_clamp)?
            .tanh()?
            .broadcast_mul(&soft_clamp)?;
        // quantize (num_quantizers == 1, scale == 1): step*floor(bracket) - 1.
        let step = levels_minus_one.recip()?.affine(2.0, 0.0)?; // 2/(levels-1)
        let clamped = h.clamp(-1.0f32, 1.0f32)?;
        // bracket = (levels-1) * (clamped+1)/2 + 0.5
        let bracket = clamped
            .broadcast_add(&ones)?
            .affine(0.5, 0.0)?
            .broadcast_mul(&levels_minus_one)?
            .affine(1.0, 0.5)?;
        let quantized = bracket.floor()?.broadcast_mul(&step)?.affine(1.0, -1.0)?; // step*floor - 1
        self.project_out.forward(&quantized)
    }
}

/// The `AceStepAudioTokenizer`: `audio_acoustic_proj` (64 → hidden) → `AttentionPooler` (25 Hz →
/// 5 Hz) → `ResidualFsq` bottleneck. `tokenize` pads to a whole pool window with the silence
/// latent (the reference recipe) before pooling.
pub struct AudioTokenizer {
    audio_acoustic_proj: Linear,
    pooler: AttentionPooler,
    fsq: ResidualFsq,
    pool_window_size: usize,
    acoustic_dim: usize,
    hidden: usize,
}

impl AudioTokenizer {
    pub fn load(weights: &Path, cfg: &TokenizerConfig, device: &Device) -> Result<Self> {
        cfg.validate()?;
        let vb = load_vb(weights, device)?;
        let audio_acoustic_proj = linear(
            cfg.audio_acoustic_hidden_dim,
            cfg.hidden_size,
            vb.pp("audio_acoustic_proj"),
        )
        .map_err(|e| AudioError::Msg(format!("acestep audio_tokenizer proj: {e}")))?;
        let pooler = AttentionPooler::new(cfg, vb.pp("attention_pooler"))
            .map_err(|e| AudioError::Msg(format!("acestep audio_tokenizer pooler: {e}")))?;
        let fsq = ResidualFsq::new(cfg, vb.pp("quantizer"))
            .map_err(|e| AudioError::Msg(format!("acestep audio_tokenizer fsq: {e}")))?;
        Ok(Self {
            audio_acoustic_proj,
            pooler,
            fsq,
            pool_window_size: cfg.pool_window_size,
            acoustic_dim: cfg.audio_acoustic_hidden_dim,
            hidden: cfg.hidden_size,
        })
    }

    /// `tokenize(src_lat [1, L, 64], silence [1, T0, 64]) → quantized [1, ceil(L/5), hidden]`.
    /// Pads `src_lat` up to a whole `pool_window_size` with the leading silence latents (falling
    /// back to zeros), reshapes into windows, and runs the pooler + FSQ. Deterministic.
    pub fn tokenize(&self, src_lat: &Tensor, silence: &Tensor) -> Result<Tensor> {
        let (_b, l, c) = src_lat.dims3()?;
        if c != self.acoustic_dim {
            return Err(AudioError::Msg(format!(
                "acestep audio_tokenizer: source acoustic dim {c} != {}",
                self.acoustic_dim
            )));
        }
        let pad_len = (self.pool_window_size - l % self.pool_window_size) % self.pool_window_size;
        let padded = if pad_len == 0 {
            src_lat.clone()
        } else {
            let (_sb, t0, _sc) = silence.dims3()?;
            let pad = if t0 >= pad_len {
                silence.narrow(1, 0, pad_len)?
            } else {
                // Silence buffer shorter than the (tiny) pad — fall back to zeros (reference does
                // the same when the silence latent cannot supply the pad).
                Tensor::zeros((1, pad_len, c), src_lat.dtype(), src_lat.device())?
            };
            Tensor::cat(&[src_lat, &pad], 1)?
        };
        let total = padded.dim(1)?;
        let num_patches = total / self.pool_window_size;
        // [1, total, c] → [num_patches, pool_window, c] (B == 1), then project 64 → hidden.
        let windows = padded
            .reshape((num_patches, self.pool_window_size, c))?
            .contiguous()?;
        let windows = self.audio_acoustic_proj.forward(&windows)?; // [num_patches, pool_window, hidden]
        let pooled = self.pooler.forward(&windows)?; // [num_patches, hidden]
        let quantized = self.fsq.forward(&pooled)?; // [num_patches, hidden]
        Ok(quantized.reshape((1, num_patches, self.hidden))?)
    }
}

/// The `AceStepAudioTokenDetokenizer`: expand each 5 Hz token back to `pool_window_size` acoustic
/// frames (adding a learned per-position offset), run the encoder stack over the 5-frame group, and
/// project to the acoustic dim — 5 Hz codes → 25 Hz acoustic conditioning.
pub struct AudioTokenDetokenizer {
    embed_tokens: Linear,
    special_tokens: Tensor, // [1, pool_window, hidden]
    layers: Vec<EncoderLayer>,
    norm: RmsNorm,
    proj_out: Linear,
    pool_window_size: usize,
    hidden: usize,
    acoustic_dim: usize,
}

impl AudioTokenDetokenizer {
    pub fn load(weights: &Path, cfg: &TokenizerConfig, device: &Device) -> Result<Self> {
        cfg.validate()?;
        let vb = load_vb(weights, device)?;
        let h = cfg.hidden_size;
        let embed_tokens = linear(h, h, vb.pp("embed_tokens"))
            .map_err(|e| AudioError::Msg(format!("acestep detokenizer embed: {e}")))?;
        let special_tokens = vb
            .get((1, cfg.pool_window_size, h), "special_tokens")
            .map_err(|e| AudioError::Msg(format!("acestep detokenizer special_tokens: {e}")))?;
        let vb_l = vb.pp("layers");
        let mut layers = Vec::with_capacity(cfg.num_attention_pooler_hidden_layers);
        for i in 0..cfg.num_attention_pooler_hidden_layers {
            layers.push(
                EncoderLayer::new(cfg, vb_l.pp(i))
                    .map_err(|e| AudioError::Msg(format!("acestep detokenizer layer {i}: {e}")))?,
            );
        }
        let norm = rms_norm(h, cfg.rms_norm_eps, vb.pp("norm"))
            .map_err(|e| AudioError::Msg(format!("acestep detokenizer norm: {e}")))?;
        let proj_out = linear(h, cfg.audio_acoustic_hidden_dim, vb.pp("proj_out"))
            .map_err(|e| AudioError::Msg(format!("acestep detokenizer proj_out: {e}")))?;
        Ok(Self {
            embed_tokens,
            special_tokens,
            layers,
            norm,
            proj_out,
            pool_window_size: cfg.pool_window_size,
            hidden: h,
            acoustic_dim: cfg.audio_acoustic_hidden_dim,
        })
    }

    /// `forward(quantized [1, N, hidden]) → src_latents [1, N*pool_window, 64]`.
    pub fn forward(&self, quantized: &Tensor) -> Result<Tensor> {
        let (_b, num_tokens, h) = quantized.dims3()?;
        if h != self.hidden {
            return Err(AudioError::Msg(format!(
                "acestep detokenizer: token dim {h} != hidden {}",
                self.hidden
            )));
        }
        let embedded = self.embed_tokens.forward(quantized)?; // [1, N, hidden]
                                                              // Expand each token to pool_window frames, add the learned per-position offset.
        let expanded = embedded
            .reshape((num_tokens, 1, self.hidden))?
            .broadcast_as((num_tokens, self.pool_window_size, self.hidden))?
            .contiguous()?;
        let special = self
            .special_tokens
            .reshape((1, self.pool_window_size, self.hidden))?;
        let mut x = expanded.broadcast_add(&special)?; // [N, pool_window, hidden]
        for l in &self.layers {
            x = l.forward(&x)?;
        }
        let x = self.norm.forward(&x)?;
        let x = self.proj_out.forward(&x)?; // [N, pool_window, acoustic]
        Ok(x.reshape((1, num_tokens * self.pool_window_size, self.acoustic_dim))?)
    }
}

/// mmap a single-file component safetensors into an f32 [`VarBuilder`].
fn load_vb(weights: &Path, device: &Device) -> Result<VarBuilder<'static>> {
    // Safety: mmap of a pinned-SHA snapshot file the contract guarantees is not mutated.
    unsafe {
        VarBuilder::from_mmaped_safetensors(&[weights.to_path_buf()], DType::F32, device)
            .map_err(|e| AudioError::Msg(format!("mmap {}: {e}", weights.display())))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn cfg() -> TokenizerConfig {
        serde_json::from_str(
            r#"{"hidden_size": 2048, "intermediate_size": 6144, "head_dim": 128,
                "num_attention_heads": 16, "num_key_value_heads": 8,
                "num_attention_pooler_hidden_layers": 2, "pool_window_size": 5,
                "audio_acoustic_hidden_dim": 64, "rms_norm_eps": 1e-6, "rope_theta": 1000000,
                "fsq_dim": 2048, "fsq_input_levels": [8,8,8,5,5,5], "fsq_input_num_quantizers": 1}"#,
        )
        .unwrap()
    }

    #[test]
    fn config_parses_the_pinned_shape() {
        let c = cfg();
        assert_eq!(c.hidden_size, 2048);
        assert_eq!(c.pool_window_size, 5);
        assert_eq!(c.fsq_input_levels, vec![8, 8, 8, 5, 5, 5]);
        assert_eq!(c.audio_acoustic_hidden_dim, 64);
        c.validate().unwrap();
    }

    /// The FSQ bottleneck is a genuine per-dimension quantizer: after `project_in`/soft-clamp, the
    /// codebook value lands on one of `levels[d]` evenly-spaced points in [-1, 1]. Exercised on the
    /// bare `_quantize` math (identity projections) so it needs no weights.
    #[test]
    fn fsq_quantize_snaps_to_levels() {
        let dev = Device::Cpu;
        let levels = [8usize, 8, 8, 5, 5, 5];
        let levels_t = Tensor::from_vec(
            levels.iter().map(|&l| l as f32).collect::<Vec<_>>(),
            levels.len(),
            &dev,
        )
        .unwrap();
        // Reproduce _quantize(x) for a few inputs and assert the output is on the level grid.
        let quantize = |x: &Tensor| -> Tensor {
            let ones = Tensor::ones_like(&levels_t).unwrap();
            let lm1 = (&levels_t - &ones).unwrap();
            let step = lm1.recip().unwrap().affine(2.0, 0.0).unwrap();
            let clamped = x.clamp(-1.0f32, 1.0f32).unwrap();
            let bracket = clamped
                .broadcast_add(&ones)
                .unwrap()
                .affine(0.5, 0.0)
                .unwrap()
                .broadcast_mul(&lm1)
                .unwrap()
                .affine(1.0, 0.5)
                .unwrap();
            bracket
                .floor()
                .unwrap()
                .broadcast_mul(&step)
                .unwrap()
                .affine(1.0, -1.0)
                .unwrap()
        };
        // A vector at the extremes and middle.
        let x = Tensor::from_vec(vec![-1.0f32, 1.0, 0.0, -1.0, 1.0, 0.0], 6, &dev).unwrap();
        let q = quantize(&x).to_vec1::<f32>().unwrap();
        // Endpoints map to ±1 exactly; the interior lands on a grid point in [-1, 1].
        assert!((q[0] + 1.0).abs() < 1e-5, "min → -1, got {}", q[0]);
        assert!((q[1] - 1.0).abs() < 1e-5, "max → +1, got {}", q[1]);
        for (d, &qd) in q.iter().enumerate() {
            assert!(
                (-1.0..=1.0).contains(&qd),
                "dim {d} value {qd} out of range"
            );
            // On the grid: (qd+1)/step is an integer in [0, levels-1].
            let step = 2.0 / (levels[d] as f32 - 1.0);
            let idx = (qd + 1.0) / step;
            assert!(
                (idx - idx.round()).abs() < 1e-4,
                "dim {d} value {qd} not on the {}-level grid",
                levels[d]
            );
        }
    }
}
