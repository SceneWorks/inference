//! The ACE-Step 1.5 **`AceStepConditionEncoder`** (sc-12842) — assembles the DiT cross-attention
//! context from the three conditioning streams: the Qwen prompt hidden states, the lyric token
//! embeddings, and (for audio-to-audio tasks) a timbre stream.
//!
//! ```text
//!   text  [1, Tp, text_hidden]  → text_proj (Linear → hidden)                         ┐
//!   lyric [1, Tl, text_hidden]  → lyric_in_proj → N× encoder layer (bidirectional)    ├─ cat → [1, S, hidden]
//!   timbre[1, Tt, timbre_hidden]→ timbre_in_proj → M× encoder layer (bidirectional)   ┘
//! ```
//!
//! Each encoder layer is a Qwen3-style bidirectional transformer block (GQA, half-split RoPE,
//! RMSNorm, SwiGLU) at the condition encoder's `hidden_size` (2048). For pure text-to-music the
//! timbre stream is absent (no reference audio); the `silence_latent` buffer supplies the DiT's
//! source-latent context, not this encoder's output.
//!
//! ## Fidelity note (sc-12842)
//!
//! The three-stream fusion, the projection module names, and whether the lyric/timbre encoders are
//! causal or bidirectional are reconstructed from the diffusers config plus the pipeline's
//! `encode_prompt` description; the exact submodule key layout of `condition_encoder/*.safetensors`
//! needs validation against the ACE-Step reference before the assembled context is certified
//! bit-faithful. This is one of the components the real-weight conformance test exists to prove out
//! (and why it stays `#[ignore]`d until proven).

use candle_audio::candle_core::{Result as CandleResult, Tensor};
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

fn rope_tables(
    head_dim: usize,
    len: usize,
    theta: f64,
    device: &candle_audio::candle_core::Device,
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

struct EncoderLayer {
    input_layernorm: RmsNorm,
    q_proj: Linear,
    k_proj: Linear,
    v_proj: Linear,
    o_proj: Linear,
    q_norm: RmsNorm,
    k_norm: RmsNorm,
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
        Ok(Self {
            input_layernorm: rms_norm(hidden, cfg.rms_norm_eps, vb.pp("input_layernorm"))?,
            q_proj: linear_b(
                hidden,
                cfg.num_attention_heads * d,
                false,
                vb.pp("self_attn.q_proj"),
            )?,
            k_proj: linear_b(
                hidden,
                cfg.num_key_value_heads * d,
                false,
                vb.pp("self_attn.k_proj"),
            )?,
            v_proj: linear_b(
                hidden,
                cfg.num_key_value_heads * d,
                false,
                vb.pp("self_attn.v_proj"),
            )?,
            o_proj: linear_b(
                cfg.num_attention_heads * d,
                hidden,
                false,
                vb.pp("self_attn.o_proj"),
            )?,
            q_norm: rms_norm(d, cfg.rms_norm_eps, vb.pp("self_attn.q_norm"))?,
            k_norm: rms_norm(d, cfg.rms_norm_eps, vb.pp("self_attn.k_norm"))?,
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
        let q = self.q_norm.forward(&to_heads(
            &self.q_proj.forward(&h)?,
            self.num_heads,
            self.head_dim,
        )?)?;
        let k = self.k_norm.forward(&to_heads(
            &self.k_proj.forward(&h)?,
            self.num_kv_heads,
            self.head_dim,
        )?)?;
        let v = to_heads(&self.v_proj.forward(&h)?, self.num_kv_heads, self.head_dim)?;
        let (cos, sin) = rope_tables(self.head_dim, len, self.theta, device)?;
        let q = candle_nn::rotary_emb::rope(&q.contiguous()?, &cos, &sin)?;
        let k = candle_nn::rotary_emb::rope(&k.contiguous()?, &cos, &sin)?;
        let groups = self.num_heads / self.num_kv_heads;
        let k = repeat_kv(&k, groups)?;
        let v = repeat_kv(&v, groups)?;
        let scale = 1.0 / (self.head_dim as f64).sqrt();
        // Bidirectional (encoder) attention — no causal mask.
        let att = candle_nn::ops::softmax_last_dim(&(q.matmul(&k.transpose(2, 3)?)? * scale)?)?;
        let attn = self
            .o_proj
            .forward(&from_heads(&att.matmul(&v.contiguous()?)?)?)?;
        let x = (x + attn)?;
        let h = self.post_attention_layernorm.forward(&x)?;
        let ff = self
            .down_proj
            .forward(&(self.gate_proj.forward(&h)?.silu()? * self.up_proj.forward(&h)?)?)?;
        x + ff
    }
}

struct Stack {
    layers: Vec<EncoderLayer>,
    norm: RmsNorm,
}

impl Stack {
    fn new(
        cfg: &ConditionEncoderConfig,
        n: usize,
        prefix: &str,
        vb: VarBuilder,
    ) -> CandleResult<Self> {
        let vb_l = vb.pp(format!("{prefix}.layers"));
        let mut layers = Vec::with_capacity(n);
        for i in 0..n {
            layers.push(EncoderLayer::new(cfg, vb_l.pp(i))?);
        }
        let norm = rms_norm(
            cfg.hidden_size,
            cfg.rms_norm_eps,
            vb.pp(format!("{prefix}.norm")),
        )?;
        Ok(Self { layers, norm })
    }

    fn forward(&self, mut x: Tensor) -> CandleResult<Tensor> {
        for l in &self.layers {
            x = l.forward(&x)?;
        }
        self.norm.forward(&x)
    }
}

/// The assembled condition encoder.
pub struct ConditionEncoder {
    text_proj: Linear,
    lyric_in_proj: Linear,
    lyric_stack: Stack,
    timbre_in_proj: Linear,
    timbre_stack: Stack,
    cfg: ConditionEncoderConfig,
}

impl ConditionEncoder {
    pub fn new(cfg: &ConditionEncoderConfig, vb: VarBuilder) -> CandleResult<Self> {
        let h = cfg.hidden_size;
        Ok(Self {
            text_proj: linear(cfg.text_hidden_dim, h, vb.pp("text_proj"))?,
            lyric_in_proj: linear(cfg.text_hidden_dim, h, vb.pp("lyric_in_proj"))?,
            lyric_stack: Stack::new(
                cfg,
                cfg.num_lyric_encoder_hidden_layers,
                "lyric_encoder",
                vb.clone(),
            )?,
            timbre_in_proj: linear(cfg.timbre_hidden_dim, h, vb.pp("timbre_in_proj"))?,
            timbre_stack: Stack::new(
                cfg,
                cfg.num_timbre_encoder_hidden_layers,
                "timbre_encoder",
                vb.clone(),
            )?,
            cfg: cfg.clone(),
        })
    }

    /// Build the DiT cross-attention context `[1, S, hidden]` from the prompt hidden states, the
    /// lyric token embeddings, and an optional timbre stream (absent for text-to-music).
    pub fn encode(
        &self,
        text_hidden: &Tensor,
        lyric_embeds: Option<&Tensor>,
        timbre: Option<&Tensor>,
    ) -> CandleResult<Tensor> {
        let mut parts: Vec<Tensor> = Vec::new();
        parts.push(self.text_proj.forward(text_hidden)?);
        if let Some(lyric) = lyric_embeds {
            let projected = self.lyric_in_proj.forward(lyric)?;
            parts.push(self.lyric_stack.forward(projected)?);
        }
        if let Some(timbre) = timbre {
            let projected = self.timbre_in_proj.forward(timbre)?;
            parts.push(self.timbre_stack.forward(projected)?);
        }
        let refs: Vec<&Tensor> = parts.iter().collect();
        Tensor::cat(&refs, 1)
    }

    pub fn hidden_size(&self) -> usize {
        self.cfg.hidden_size
    }
}
