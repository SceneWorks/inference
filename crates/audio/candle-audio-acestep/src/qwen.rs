//! Qwen3-Embedding-0.6B as ACE-Step's **text encoder** (sc-12842).
//!
//! Two entry points, both stateless full-sequence forwards (no KV-cache accumulation across
//! prompts — the same statelessness argument as the MOSS SFX Qwen3 encoder):
//!
//! - [`Qwen3Encoder::encode`] — the full causal stack + final RMSNorm, returning the
//!   post-norm hidden states the reference feeds as the DiT **prompt** conditioning.
//! - [`Qwen3Encoder::embed`] — the embedding-layer token lookup only, returning the raw token
//!   embeddings the condition encoder's lyric encoder contextualizes (the reference lyric path).
//!
//! The module mirrors the upstream weight layout byte-for-byte (`model.embed_tokens` /
//! `model.layers.N.*` / `model.norm`) and the HF Qwen3 math: GQA with per-head q/k RMSNorm,
//! half-split (NeoX) RoPE at `rope_theta`, SiLU MLP.

use candle_audio::candle_core::{DType, Device, Result as CandleResult, Tensor};
use candle_nn::{linear_b, rms_norm, Embedding, Linear, Module, RmsNorm, VarBuilder};

use crate::config::TextEncoderConfig;

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
    fn new(cfg: &TextEncoderConfig, vb: VarBuilder) -> CandleResult<Self> {
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

        let q = self.q_norm.forward(&q.contiguous()?)?;
        let k = self.k_norm.forward(&k.contiguous()?)?;
        let q = candle_nn::rotary_emb::rope(&q.contiguous()?, cos, sin)?;
        let k = candle_nn::rotary_emb::rope(&k.contiguous()?, cos, sin)?;

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
    fn new(cfg: &TextEncoderConfig, vb: VarBuilder) -> CandleResult<Self> {
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

struct Layer {
    input_layernorm: RmsNorm,
    attn: Attention,
    post_attention_layernorm: RmsNorm,
    mlp: Mlp,
}

impl Layer {
    fn new(cfg: &TextEncoderConfig, vb: VarBuilder) -> CandleResult<Self> {
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

    fn forward(
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

/// The stateless Qwen3 text encoder.
pub struct Qwen3Encoder {
    embed_tokens: Embedding,
    layers: Vec<Layer>,
    norm: RmsNorm,
    rope_theta: f64,
    head_dim: usize,
    hidden_size: usize,
    device: Device,
}

impl Qwen3Encoder {
    pub fn new(cfg: &TextEncoderConfig, vb: VarBuilder) -> CandleResult<Self> {
        // The Qwen3-Embedding-0.6B safetensors store the bare `Qwen3Model` (no `model.` prefix):
        // `embed_tokens.weight` / `layers.N.*` / `norm.weight`.
        let embed_tokens =
            candle_nn::embedding(cfg.vocab_size, cfg.hidden_size, vb.pp("embed_tokens"))?;
        let vb_l = vb.pp("layers");
        let mut layers = Vec::with_capacity(cfg.num_hidden_layers);
        for i in 0..cfg.num_hidden_layers {
            layers.push(Layer::new(cfg, vb_l.pp(i))?);
        }
        let norm = rms_norm(cfg.hidden_size, cfg.rms_norm_eps, vb.pp("norm"))?;
        Ok(Self {
            embed_tokens,
            layers,
            norm,
            rope_theta: cfg.rope_theta,
            head_dim: cfg.head_dim,
            hidden_size: cfg.hidden_size,
            device: vb.device().clone(),
        })
    }

    pub fn hidden_size(&self) -> usize {
        self.hidden_size
    }

    pub fn device(&self) -> &Device {
        &self.device
    }

    fn rope_tables(&self, len: usize) -> CandleResult<(Tensor, Tensor)> {
        let half = self.head_dim / 2;
        let mut cos = Vec::with_capacity(len * half);
        let mut sin = Vec::with_capacity(len * half);
        for pos in 0..len {
            for i in 0..half {
                let inv = 1.0 / self.rope_theta.powf(2.0 * i as f64 / self.head_dim as f64);
                let angle = pos as f64 * inv;
                cos.push(angle.cos() as f32);
                sin.push(angle.sin() as f32);
            }
        }
        Ok((
            Tensor::from_vec(cos, (len, half), &self.device)?,
            Tensor::from_vec(sin, (len, half), &self.device)?,
        ))
    }

    fn causal_mask(&self, len: usize, dtype: DType) -> CandleResult<Tensor> {
        let data: Vec<f32> = (0..len)
            .flat_map(|i| (0..len).map(move |j| if j <= i { 0.0 } else { f32::NEG_INFINITY }))
            .collect();
        Tensor::from_vec(data, (1, 1, len, len), &self.device)?.to_dtype(dtype)
    }

    /// The embedding-layer token lookup only → `[1, len, hidden]` (the lyric path).
    pub fn embed(&self, ids: &[u32]) -> CandleResult<Tensor> {
        let input = Tensor::from_vec(ids.to_vec(), (1, ids.len()), &self.device)?;
        self.embed_tokens.forward(&input)
    }

    /// The full causal stack + final RMSNorm → `[1, len, hidden]` post-norm hidden states (the
    /// prompt path).
    pub fn encode(&self, ids: &[u32]) -> CandleResult<Tensor> {
        let len = ids.len();
        let input = Tensor::from_vec(ids.to_vec(), (1, len), &self.device)?;
        let mut h = self.embed_tokens.forward(&input)?;
        let (cos, sin) = self.rope_tables(len)?;
        let mask = if len > 1 {
            Some(self.causal_mask(len, h.dtype())?)
        } else {
            None
        };
        for layer in &self.layers {
            h = layer.forward(&h, &cos, &sin, mask.as_ref())?;
        }
        self.norm.forward(&h)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use candle_audio::candle_core::IndexOp;
    use candle_nn::VarMap;

    fn tiny_cfg() -> TextEncoderConfig {
        serde_json::from_str(
            r#"{"vocab_size": 32, "hidden_size": 16, "intermediate_size": 32,
                "num_hidden_layers": 2, "num_attention_heads": 4, "num_key_value_heads": 2,
                "head_dim": 4, "attention_bias": false, "rms_norm_eps": 1e-6,
                "rope_theta": 1000000.0, "use_sliding_window": false}"#,
        )
        .unwrap()
    }

    fn tiny_encoder() -> Qwen3Encoder {
        let cfg = tiny_cfg();
        let varmap = VarMap::new();
        let vb = VarBuilder::from_varmap(&varmap, DType::F32, &Device::Cpu);
        let enc = Qwen3Encoder::new(&cfg, vb).unwrap();
        for (i, (_, var)) in varmap.data().lock().unwrap().iter().enumerate() {
            let t = var.as_tensor();
            let n: usize = t.shape().elem_count();
            let vals: Vec<f32> = (0..n)
                .map(|j| (((i * 31 + j * 17) % 13) as f64 * 0.03 - 0.18) as f32)
                .collect();
            let new = Tensor::from_vec(vals, t.shape(), &Device::Cpu).unwrap();
            var.set(&new).unwrap();
        }
        enc
    }

    #[test]
    fn encode_is_causal_and_stateless() {
        let enc = tiny_encoder();
        let full = enc.encode(&[1, 2, 3, 4]).unwrap();
        let prefix = enc.encode(&[1, 2]).unwrap();
        let a = full
            .i((0, 0..2))
            .unwrap()
            .flatten_all()
            .unwrap()
            .to_vec1::<f32>()
            .unwrap();
        let b = prefix.flatten_all().unwrap().to_vec1::<f32>().unwrap();
        for (x, y) in a.iter().zip(&b) {
            assert!((x - y).abs() < 1e-5, "causal prefix mismatch {x} vs {y}");
        }
        let again = enc.encode(&[1, 2, 3, 4]).unwrap();
        assert_eq!(
            full.flatten_all().unwrap().to_vec1::<f32>().unwrap(),
            again.flatten_all().unwrap().to_vec1::<f32>().unwrap()
        );
    }

    #[test]
    fn embed_is_a_pure_lookup() {
        let enc = tiny_encoder();
        let e = enc.embed(&[1, 2, 3]).unwrap();
        assert_eq!(e.dims(), &[1, 3, 16]);
    }
}
