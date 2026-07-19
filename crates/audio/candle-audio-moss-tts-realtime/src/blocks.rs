//! Shared Qwen3-style transformer blocks (sc-13334).
//!
//! Both the MOSS-TTS-Realtime backbone (`config.json.language_config`) and the local/depth
//! transformer (`config.json.local_config`) are Qwen3 decoder stacks with identical block math —
//! GQA with per-head q/k RMSNorm, half-split (NeoX) RoPE at `rope_theta`, and a SiLU gated MLP.
//! This module factors that block out so the two stacks share one verified implementation. The
//! forward here is a **stateless full-sequence** attention (no KV cache): the AR decode recomputes
//! the growing prefix each frame (see [`crate::decode`]). This is correct and simple; a KV-cache
//! optimization is tracked as a follow-up (it does not change the emitted tokens, only latency).

use candle_audio::candle_core::{DType, Device, Result as CandleResult, Tensor};
use candle_nn::{linear_b, rms_norm, Linear, Module, RmsNorm, VarBuilder};

/// The per-block hyperparameters common to a Qwen3 decoder layer.
#[derive(Debug, Clone, Copy)]
pub struct BlockConfig {
    pub hidden_size: usize,
    pub intermediate_size: usize,
    pub num_attention_heads: usize,
    pub num_key_value_heads: usize,
    pub head_dim: usize,
    pub rms_norm_eps: f64,
    pub rope_theta: f64,
    pub attention_bias: bool,
}

struct Attention {
    q_proj: Linear,
    k_proj: Linear,
    v_proj: Linear,
    o_proj: Linear,
    q_norm: RmsNorm,
    k_norm: RmsNorm,
    num_heads: usize,
    num_kv_heads: usize,
    head_dim: usize,
}

impl Attention {
    fn new(cfg: &BlockConfig, vb: VarBuilder) -> CandleResult<Self> {
        let d = cfg.head_dim;
        Ok(Self {
            q_proj: linear_b(
                cfg.hidden_size,
                cfg.num_attention_heads * d,
                cfg.attention_bias,
                vb.pp("q_proj"),
            )?,
            k_proj: linear_b(
                cfg.hidden_size,
                cfg.num_key_value_heads * d,
                cfg.attention_bias,
                vb.pp("k_proj"),
            )?,
            v_proj: linear_b(
                cfg.hidden_size,
                cfg.num_key_value_heads * d,
                cfg.attention_bias,
                vb.pp("v_proj"),
            )?,
            o_proj: linear_b(
                cfg.num_attention_heads * d,
                cfg.hidden_size,
                cfg.attention_bias,
                vb.pp("o_proj"),
            )?,
            q_norm: rms_norm(d, cfg.rms_norm_eps, vb.pp("q_norm"))?,
            k_norm: rms_norm(d, cfg.rms_norm_eps, vb.pp("k_norm"))?,
            num_heads: cfg.num_attention_heads,
            num_kv_heads: cfg.num_key_value_heads,
            head_dim: d,
        })
    }

    fn forward(
        &self,
        x: &Tensor,
        cos: &Tensor,
        sin: &Tensor,
        mask: Option<&Tensor>,
    ) -> CandleResult<Tensor> {
        let (b, l, _) = x.dims3()?;
        let q = self
            .q_proj
            .forward(x)?
            .reshape((b, l, self.num_heads, self.head_dim))?
            .transpose(1, 2)?;
        let k = self
            .k_proj
            .forward(x)?
            .reshape((b, l, self.num_kv_heads, self.head_dim))?
            .transpose(1, 2)?;
        let v = self
            .v_proj
            .forward(x)?
            .reshape((b, l, self.num_kv_heads, self.head_dim))?
            .transpose(1, 2)?;

        // Per-head q/k RMSNorm (over head_dim), then half-split RoPE — the HF Qwen3 order.
        let q = self.q_norm.forward(&q.contiguous()?)?;
        let k = self.k_norm.forward(&k.contiguous()?)?;
        let q = candle_nn::rotary_emb::rope(&q.contiguous()?, cos, sin)?;
        let k = candle_nn::rotary_emb::rope(&k.contiguous()?, cos, sin)?;

        // GQA: expand kv heads to the query-head count.
        let groups = self.num_heads / self.num_kv_heads;
        let k = repeat_kv(&k, groups)?;
        let v = repeat_kv(&v, groups)?;

        let scale = 1.0 / (self.head_dim as f64).sqrt();
        let mut att = (q.matmul(&k.transpose(2, 3)?)? * scale)?;
        if let Some(m) = mask {
            att = att.broadcast_add(m)?;
        }
        let att = candle_nn::ops::softmax_last_dim(&att)?;
        let out = att.matmul(&v.contiguous()?)?.transpose(1, 2)?.reshape((
            b,
            l,
            self.num_heads * self.head_dim,
        ))?;
        self.o_proj.forward(&out)
    }
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

struct Mlp {
    gate_proj: Linear,
    up_proj: Linear,
    down_proj: Linear,
}

impl Mlp {
    fn new(cfg: &BlockConfig, vb: VarBuilder) -> CandleResult<Self> {
        Ok(Self {
            gate_proj: linear_b(
                cfg.hidden_size,
                cfg.intermediate_size,
                false,
                vb.pp("gate_proj"),
            )?,
            up_proj: linear_b(
                cfg.hidden_size,
                cfg.intermediate_size,
                false,
                vb.pp("up_proj"),
            )?,
            down_proj: linear_b(
                cfg.intermediate_size,
                cfg.hidden_size,
                false,
                vb.pp("down_proj"),
            )?,
        })
    }

    fn forward(&self, x: &Tensor) -> CandleResult<Tensor> {
        let gate = self.gate_proj.forward(x)?.silu()?;
        let up = self.up_proj.forward(x)?;
        self.down_proj.forward(&(gate * up)?)
    }
}

/// One Qwen3 decoder layer (pre-norm attention + pre-norm SiLU MLP, residual).
pub struct Layer {
    input_layernorm: RmsNorm,
    attn: Attention,
    post_attention_layernorm: RmsNorm,
    mlp: Mlp,
}

impl Layer {
    pub fn new(cfg: &BlockConfig, vb: VarBuilder) -> CandleResult<Self> {
        Ok(Self {
            input_layernorm: rms_norm(cfg.hidden_size, cfg.rms_norm_eps, vb.pp("input_layernorm"))?,
            attn: Attention::new(cfg, vb.pp("self_attn"))?,
            post_attention_layernorm: rms_norm(
                cfg.hidden_size,
                cfg.rms_norm_eps,
                vb.pp("post_attention_layernorm"),
            )?,
            mlp: Mlp::new(cfg, vb.pp("mlp"))?,
        })
    }

    pub fn forward(
        &self,
        x: &Tensor,
        cos: &Tensor,
        sin: &Tensor,
        mask: Option<&Tensor>,
    ) -> CandleResult<Tensor> {
        let h = self
            .attn
            .forward(&self.input_layernorm.forward(x)?, cos, sin, mask)?;
        let x = (x + h)?;
        let h = self
            .mlp
            .forward(&self.post_attention_layernorm.forward(&x)?)?;
        x + h
    }
}

/// Half-split (NeoX) cos/sin tables for positions `0..len` at `rope_theta` — `[len, head_dim/2]`,
/// the shape `candle_nn::rotary_emb::rope` consumes.
pub fn rope_tables(
    device: &Device,
    len: usize,
    head_dim: usize,
    rope_theta: f64,
) -> CandleResult<(Tensor, Tensor)> {
    let half = head_dim / 2;
    let mut cos = Vec::with_capacity(len * half);
    let mut sin = Vec::with_capacity(len * half);
    for pos in 0..len {
        for i in 0..half {
            let inv = 1.0 / rope_theta.powf(2.0 * i as f64 / head_dim as f64);
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

/// A `[1, 1, len, len]` additive causal mask (`0` on/below the diagonal, `-inf` above).
pub fn causal_mask(device: &Device, len: usize, dtype: DType) -> CandleResult<Tensor> {
    let data: Vec<f32> = (0..len)
        .flat_map(|i| (0..len).map(move |j| if j <= i { 0.0 } else { f32::NEG_INFINITY }))
        .collect();
    Tensor::from_vec(data, (1, 1, len, len), device)?.to_dtype(dtype)
}
