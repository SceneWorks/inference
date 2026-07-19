//! The ACE-Step 1.5 **`AceStepConditionEncoder`** (sc-12842) — assembles the DiT cross-attention
//! context from the prompt, lyric, and timbre streams, and owns the `silence_latent` buffer that
//! seeds the text-to-music source latents.
//!
//! Weight layout (verified against the pinned `condition_encoder/*.safetensors`):
//!
//! ```text
//!   text_projector            Linear(text_hidden 1024 → hidden 2048, no bias)
//!   lyric_encoder.embed_tokens Linear(1024 → 2048, bias)  ← projects the Qwen lyric embeddings
//!   lyric_encoder.layers.N     8× bidirectional block (to_q/to_k/to_v/to_out.0 + norm_q/norm_k,
//!                                 GQA 16/8 head_dim 128, RoPE, SwiGLU MLP, RMSNorms)
//!   lyric_encoder.norm         RMSNorm
//!   timbre_encoder.embed_tokens Linear(64 → 2048, bias)   ← real timbre latents (audio-to-audio)
//!   timbre_encoder.special_token [1, 1, 2048]             ← the text-to-music (no reference) timbre
//!   timbre_encoder.layers.N     4× bidirectional block
//!   timbre_encoder.norm         RMSNorm
//!   silence_latent            [1, 15000, 64]               ← tiled/cropped to the src latents
//! ```
//!
//! The fused context is `cat([text_proj(prompt), lyric_encoder(lyrics), timbre_encoder(timbre)])`
//! along the sequence. For pure text-to-music the timbre stream is the learned `special_token`
//! (no reference audio); an absent lyric stream (instrumental) is simply omitted.

use candle_audio::candle_core::{Device, Result as CandleResult, Tensor};
use candle_nn::{linear, linear_b, rms_norm, Linear, Module, RmsNorm, VarBuilder};

use crate::config::ConditionEncoderConfig;

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

fn rope_tables(head_dim: usize, len: usize, theta: f64, device: &Device) -> CandleResult<(Tensor, Tensor)> {
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
    fn new(cfg: &ConditionEncoderConfig, vb: VarBuilder) -> CandleResult<Self> {
        let d = cfg.head_dim;
        let hidden = cfg.hidden_size;
        let sa = vb.pp("self_attn");
        Ok(Self {
            input_layernorm: rms_norm(hidden, cfg.rms_norm_eps, vb.pp("input_layernorm"))?,
            to_q: linear_b(hidden, cfg.num_attention_heads * d, false, sa.pp("to_q"))?,
            to_k: linear_b(hidden, cfg.num_key_value_heads * d, false, sa.pp("to_k"))?,
            to_v: linear_b(hidden, cfg.num_key_value_heads * d, false, sa.pp("to_v"))?,
            to_out: linear_b(cfg.num_attention_heads * d, hidden, false, sa.pp("to_out.0"))?,
            norm_q: rms_norm(d, cfg.rms_norm_eps, sa.pp("norm_q"))?,
            norm_k: rms_norm(d, cfg.rms_norm_eps, sa.pp("norm_k"))?,
            post_attention_layernorm: rms_norm(hidden, cfg.rms_norm_eps, vb.pp("post_attention_layernorm"))?,
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
        let q = self.norm_q.forward(&to_heads(&self.to_q.forward(&h)?, self.num_heads, self.head_dim)?)?;
        let k = self.norm_k.forward(&to_heads(&self.to_k.forward(&h)?, self.num_kv_heads, self.head_dim)?)?;
        let v = to_heads(&self.to_v.forward(&h)?, self.num_kv_heads, self.head_dim)?;
        let (cos, sin) = rope_tables(self.head_dim, len, self.theta, device)?;
        let q = candle_nn::rotary_emb::rope(&q.contiguous()?, &cos, &sin)?;
        let k = candle_nn::rotary_emb::rope(&k.contiguous()?, &cos, &sin)?;
        let groups = self.num_heads / self.num_kv_heads;
        let k = repeat_kv(&k, groups)?;
        let v = repeat_kv(&v, groups)?;
        let scale = 1.0 / (self.head_dim as f64).sqrt();
        // Bidirectional (encoder) attention — no causal mask.
        let att = candle_nn::ops::softmax_last_dim(&(q.matmul(&k.transpose(2, 3)?)? * scale)?)?;
        let attn = self.to_out.forward(&from_heads(&att.matmul(&v.contiguous()?)?)?)?;
        let x = (x + attn)?;
        let h = self.post_attention_layernorm.forward(&x)?;
        let ff = self.down_proj.forward(&(self.gate_proj.forward(&h)?.silu()? * self.up_proj.forward(&h)?)?)?;
        x + ff
    }
}

/// One `{lyric,timbre}_encoder` sub-stack: an input projection, N bidirectional layers, a norm.
struct Encoder {
    embed_tokens: Linear,
    layers: Vec<EncoderLayer>,
    norm: RmsNorm,
}

impl Encoder {
    fn new(cfg: &ConditionEncoderConfig, in_dim: usize, n: usize, prefix: &str, vb: VarBuilder) -> CandleResult<Self> {
        let root = vb.pp(prefix);
        let embed_tokens = linear(in_dim, cfg.hidden_size, root.pp("embed_tokens"))?;
        let vb_l = root.pp("layers");
        let mut layers = Vec::with_capacity(n);
        for i in 0..n {
            layers.push(EncoderLayer::new(cfg, vb_l.pp(i))?);
        }
        let norm = rms_norm(cfg.hidden_size, cfg.rms_norm_eps, root.pp("norm"))?;
        Ok(Self {
            embed_tokens,
            layers,
            norm,
        })
    }

    /// Run projected inputs through the stack. `project` selects whether to apply `embed_tokens`
    /// (raw inputs) or feed an already-`hidden`-width tensor straight in (the timbre special token).
    fn forward(&self, input: &Tensor, project: bool) -> CandleResult<Tensor> {
        let mut x = if project {
            self.embed_tokens.forward(input)?
        } else {
            input.clone()
        };
        for l in &self.layers {
            x = l.forward(&x)?;
        }
        self.norm.forward(&x)
    }
}

/// The assembled condition encoder.
pub struct ConditionEncoder {
    text_projector: Linear,
    lyric_encoder: Encoder,
    timbre_encoder: Encoder,
    special_token: Tensor, // [1, 1, hidden]
    silence_latent: Tensor, // [1, T0, acoustic]
    cfg: ConditionEncoderConfig,
}

impl ConditionEncoder {
    pub fn new(cfg: &ConditionEncoderConfig, vb: VarBuilder) -> CandleResult<Self> {
        let h = cfg.hidden_size;
        let text_projector = linear_b(cfg.text_hidden_dim, h, false, vb.pp("text_projector"))?;
        let lyric_encoder = Encoder::new(cfg, cfg.text_hidden_dim, cfg.num_lyric_encoder_hidden_layers, "lyric_encoder", vb.clone())?;
        let timbre_encoder = Encoder::new(cfg, cfg.timbre_hidden_dim, cfg.num_timbre_encoder_hidden_layers, "timbre_encoder", vb.clone())?;
        let special_token = vb.get((1, 1, h), "timbre_encoder.special_token")?;
        let silence_latent = vb.get_unchecked("silence_latent")?;
        Ok(Self {
            text_projector,
            lyric_encoder,
            timbre_encoder,
            special_token,
            silence_latent,
            cfg: cfg.clone(),
        })
    }

    /// The source latents `[1, latent_len, acoustic]` for text-to-music: the learned
    /// `silence_latent` tiled/cropped to the requested length.
    pub fn src_latents(&self, latent_len: usize, device: &Device) -> CandleResult<Tensor> {
        let (_, t0, c) = self.silence_latent.dims3()?;
        if latent_len <= t0 {
            self.silence_latent.narrow(1, 0, latent_len)?.to_device(device)
        } else {
            let reps = latent_len.div_ceil(t0);
            let tiled = Tensor::cat(&vec![&self.silence_latent; reps], 1)?;
            tiled.narrow(1, 0, latent_len)?.to_device(device)?.reshape((1, latent_len, c))
        }
    }

    /// Build the DiT cross-attention context `[1, S, hidden]` from the prompt hidden states, the
    /// lyric token embeddings (Qwen embedding lookup), and the text-to-music timbre special token.
    pub fn encode(&self, text_hidden: &Tensor, lyric_embeds: Option<&Tensor>) -> CandleResult<Tensor> {
        let mut parts: Vec<Tensor> = Vec::new();
        parts.push(self.text_projector.forward(text_hidden)?);
        if let Some(lyric) = lyric_embeds {
            parts.push(self.lyric_encoder.forward(lyric, true)?);
        }
        // Text-to-music timbre: the learned special token (already hidden-width), through the
        // timbre encoder stack.
        parts.push(self.timbre_encoder.forward(&self.special_token, false)?);
        let refs: Vec<&Tensor> = parts.iter().collect();
        Tensor::cat(&refs, 1)
    }

    pub fn hidden_size(&self) -> usize {
        self.cfg.hidden_size
    }
}
