//! Ideogram 4's **Qwen3-VL-8B-Instruct** text encoder (text path only — the vision tower is unused
//! for text-to-image). A 36-layer decoder-only LM whose hidden states at the 13 indices in
//! [`crate::config::EXTRACTED_LAYERS`] (`0,3,…,33,35`) are **interleaved** into the
//! `13·4096 = 53248`-wide features the DiT's `llm_cond_proj` consumes.
//!
//! Adapted from `candle-gen-flux2`'s `Flux2PromptEncoder` (same Qwen3 assembly: GQA 32q/8kv, bias-less
//! q/k/v/o, per-head q/k RMSNorm, HF half-split RoPE, SwiGLU, pre-norm residual blocks, no final
//! norm). Ideogram differs in exactly three ways:
//!   * **θ = 5e6** (klein's Qwen3 is 1e6),
//!   * **13** captured states under the `language_model.*` key prefix (klein concatenates 3 under
//!     `model.*`), and — critically —
//!   * the capture index is the LAYER index whose OUTPUT is taken (`captured[i] = layer_i(hidden)`),
//!     NOT HF `output_hidden_states` (which offsets by one with raw embeddings at index 0); and the
//!     captured states are **interleaved** on the feature axis (`f = h·n + layer`), NOT
//!     block-concatenated — the DiT's `llm_cond_proj` was trained on the interleaved layout; the
//!     wrong order yields a coherent but prompt-agnostic image.
//!
//! The text-only path uses plain 1-D RoPE: Qwen3-VL's MRoPE sections all index the same sequential
//! text position when there are no image tokens, so it reduces to standard RoPE. **Computes in f32**;
//! its weights are **stored bf16** (sc-12828). The Qwen3-VL-8B weights ship bf16 on disk, so an f32
//! store only widens them (~16 GB resident to carry no extra precision). The embedding is upcast to f32
//! and each projection runs [`QLinear::forward_upcast`] (bf16 weight → f32 per matmul), with the
//! RMSNorm weights loaded f32 (`rms_norm_f32`), so the forward is bit-identical to an f32 store at
//! half the resident footprint.

use candle_gen::candle_core::{DType, Device, Result, Tensor, D};
use candle_gen::candle_nn::{ops::softmax_last_dim, rotary_emb::rope, Module, RmsNorm, VarBuilder};
use candle_gen::quant::{embedding_dtype, lin, QEmbedding, QLinear};

use crate::config::Ideogram4TextEncoderConfig;

/// Build a candle_nn [`RmsNorm`] whose weight is forced to **f32** (sc-12828). The encoder stores its
/// bulk weights bf16 but computes in f32 (module docs); candle_nn's RmsNorm applies its weight at the
/// input's dtype — f32 here — so the tiny norm weight must be f32. Byte-identical to the old f32-store
/// build (the disk weight is bf16, so f32 only widens it); replaces the plain `candle_nn::rms_norm`
/// builder, which would read the weight at the VarBuilder's (now bf16) dtype.
fn rms_norm_f32(size: usize, eps: f64, vb: VarBuilder) -> Result<RmsNorm> {
    let weight = vb.get(size, "weight")?.to_dtype(DType::F32)?;
    Ok(RmsNorm::new(weight, eps))
}

/// HF half-split RoPE table (θ over `head_dim`), built once for the max sequence length.
struct Rotary {
    cos: Tensor,
    sin: Tensor,
}

impl Rotary {
    fn new(head_dim: usize, theta: f32, max_seq: usize, device: &Device) -> Result<Self> {
        let inv_freq: Vec<f32> = (0..head_dim)
            .step_by(2)
            .map(|i| 1f32 / theta.powf(i as f32 / head_dim as f32))
            .collect();
        let n = inv_freq.len();
        let inv_freq = Tensor::from_vec(inv_freq, (1, n), device)?;
        let t = Tensor::arange(0u32, max_seq as u32, device)?
            .to_dtype(DType::F32)?
            .reshape((max_seq, 1))?;
        let freqs = t.matmul(&inv_freq)?; // (max_seq, head_dim/2)
        Ok(Self {
            cos: freqs.cos()?,
            sin: freqs.sin()?,
        })
    }

    fn apply(&self, q: &Tensor, k: &Tensor) -> Result<(Tensor, Tensor)> {
        let (_, _, seq, _) = q.dims4()?;
        let cos = self.cos.narrow(0, 0, seq)?;
        let sin = self.sin.narrow(0, 0, seq)?;
        let q = rope(&q.contiguous()?, &cos, &sin)?;
        let k = rope(&k.contiguous()?, &cos, &sin)?;
        Ok((q, k))
    }
}

struct Attention {
    q_proj: QLinear,
    k_proj: QLinear,
    v_proj: QLinear,
    o_proj: QLinear,
    q_norm: RmsNorm,
    k_norm: RmsNorm,
    n_heads: usize,
    n_kv_heads: usize,
    head_dim: usize,
}

impl Attention {
    fn new(cfg: &Ideogram4TextEncoderConfig, vb: VarBuilder) -> Result<Self> {
        let h = cfg.hidden_size;
        let (nh, nkv, hd) = (cfg.num_heads, cfg.num_kv_heads, cfg.head_dim);
        Ok(Self {
            q_proj: lin(&vb, "q_proj", h, nh * hd, false)?,
            k_proj: lin(&vb, "k_proj", h, nkv * hd, false)?,
            v_proj: lin(&vb, "v_proj", h, nkv * hd, false)?,
            o_proj: lin(&vb, "o_proj", nh * hd, h, false)?,
            q_norm: rms_norm_f32(hd, cfg.rms_norm_eps, vb.pp("q_norm"))?,
            k_norm: rms_norm_f32(hd, cfg.rms_norm_eps, vb.pp("k_norm"))?,
            n_heads: nh,
            n_kv_heads: nkv,
            head_dim: hd,
        })
    }

    fn forward(&self, x: &Tensor, rotary: &Rotary, mask: &Tensor) -> Result<Tensor> {
        let (b, s, _) = x.dims3()?;
        let (nh, nkv, hd) = (self.n_heads, self.n_kv_heads, self.head_dim);

        // Project, reshape to [B, H, S, D], apply per-head q/k RMSNorm (over the head_dim axis).
        // `forward_upcast` (sc-12828): bf16-stored projections upcast to the f32 hidden per matmul —
        // bit-identical to an f32 store, inert when `x` already matches the weight dtype.
        let q = self.q_proj.forward_upcast(x)?.reshape((b, s, nh, hd))?;
        let k = self.k_proj.forward_upcast(x)?.reshape((b, s, nkv, hd))?;
        let v = self.v_proj.forward_upcast(x)?.reshape((b, s, nkv, hd))?;
        let q = self.q_norm.forward(&q)?.transpose(1, 2)?; // [B, nh, S, D]
        let k = self.k_norm.forward(&k)?.transpose(1, 2)?; // [B, nkv, S, D]
        let v = v.transpose(1, 2)?.contiguous()?;

        let (q, k) = rotary.apply(&q, &k)?;
        // GQA: repeat kv heads to query-head count.
        let k = repeat_kv(&k, nh / nkv)?;
        let v = repeat_kv(&v, nh / nkv)?;

        let scale = (hd as f64).powf(-0.5);
        let scores = (q.contiguous()?.matmul(&k.transpose(2, 3)?.contiguous()?)? * scale)?;
        let scores = scores.broadcast_add(mask)?; // [B, nh, S, S] + [B, 1, S, S]
        let probs = softmax_last_dim(&scores)?;
        let o = probs.matmul(&v)?; // [B, nh, S, D]
        let o = o.transpose(1, 2)?.reshape((b, s, nh * hd))?;
        self.o_proj.forward_upcast(&o)
    }
}

/// Repeat each kv head `groups` times along the head axis ([B, nkv, S, D] → [B, nkv·groups, S, D]).
fn repeat_kv(x: &Tensor, groups: usize) -> Result<Tensor> {
    if groups == 1 {
        return Ok(x.clone());
    }
    let (b, nkv, s, d) = x.dims4()?;
    x.unsqueeze(2)?
        .expand((b, nkv, groups, s, d))?
        .reshape((b, nkv * groups, s, d))
}

struct Mlp {
    gate: QLinear,
    up: QLinear,
    down: QLinear,
}

impl Mlp {
    fn new(cfg: &Ideogram4TextEncoderConfig, vb: VarBuilder) -> Result<Self> {
        let (h, i) = (cfg.hidden_size, cfg.intermediate_size);
        Ok(Self {
            gate: lin(&vb, "gate_proj", h, i, false)?,
            up: lin(&vb, "up_proj", h, i, false)?,
            down: lin(&vb, "down_proj", i, h, false)?,
        })
    }

    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        // `forward_upcast` (sc-12828): bf16-stored projections, f32 hidden — see `Attention::forward`.
        let g = self.gate.forward_upcast(x)?.silu()?;
        let u = self.up.forward_upcast(x)?;
        self.down.forward_upcast(&(g * u)?)
    }
}

struct DecoderLayer {
    input_ln: RmsNorm,
    post_ln: RmsNorm,
    attn: Attention,
    mlp: Mlp,
}

impl DecoderLayer {
    fn new(cfg: &Ideogram4TextEncoderConfig, vb: VarBuilder) -> Result<Self> {
        Ok(Self {
            input_ln: rms_norm_f32(cfg.hidden_size, cfg.rms_norm_eps, vb.pp("input_layernorm"))?,
            post_ln: rms_norm_f32(
                cfg.hidden_size,
                cfg.rms_norm_eps,
                vb.pp("post_attention_layernorm"),
            )?,
            attn: Attention::new(cfg, vb.pp("self_attn"))?,
            mlp: Mlp::new(cfg, vb.pp("mlp"))?,
        })
    }

    fn forward(&self, x: &Tensor, rotary: &Rotary, mask: &Tensor) -> Result<Tensor> {
        let h = (x + self
            .attn
            .forward(&self.input_ln.forward(x)?, rotary, mask)?)?;
        &h + self.mlp.forward(&self.post_ln.forward(&h)?)?
    }
}

/// The Ideogram 4 Qwen3-VL text-path prompt-embeds encoder.
pub struct Ideogram4TextEncoder {
    embed_tokens: QEmbedding,
    layers: Vec<DecoderLayer>,
    rotary: Rotary,
    /// Layer indices whose OUTPUTS are captured (`captured[i] = layer_i(hidden)`).
    out_layers: Vec<usize>,
}

impl Ideogram4TextEncoder {
    /// Build under the `language_model.*` prefix. The final `language_model.norm` and `lm_head` are
    /// intentionally not loaded — Ideogram uses the raw (pre-final-norm) intermediate states. Only
    /// the first `max(out_layers) + 1` layers are constructed (higher layers cannot affect the kept
    /// states). `max_seq` sizes the RoPE table (use [`crate::config::MAX_TEXT_TOKENS`]).
    pub fn new(
        cfg: &Ideogram4TextEncoderConfig,
        out_layers: &[usize],
        max_seq: usize,
        vb: VarBuilder,
    ) -> Result<Self> {
        let model = vb.pp("language_model");
        // Dense embed rides the bf16 store (widened to f32 in `prompt_embeds`, exact); the packed embed
        // dequantizes to f32 — bit-identical to the old f32 store (sc-12828).
        let embed_tokens = embedding_dtype(
            &model,
            "embed_tokens",
            cfg.vocab_size,
            cfg.hidden_size,
            DType::F32,
        )?;
        let max_layer = *out_layers.iter().max().unwrap_or(&0);
        let mut layers = Vec::with_capacity(max_layer + 1);
        let vb_layers = model.pp("layers");
        for i in 0..=max_layer {
            layers.push(DecoderLayer::new(cfg, vb_layers.pp(i))?);
        }
        let rotary = Rotary::new(cfg.head_dim, cfg.rope_theta, max_seq.max(1), vb.device())?;
        Ok(Self {
            embed_tokens,
            layers,
            rotary,
            out_layers: out_layers.to_vec(),
        })
    }

    /// `input_ids` / `attention_mask`: `[B, S]` (ids u32, mask 1=real/0=pad). Returns the
    /// **interleaved** hidden states `[B, S, n·hidden]` (f32) — Ideogram's `llm` features. The final
    /// norm is never applied; only layers up to `max(out_layers)` are run.
    pub fn prompt_embeds(&self, input_ids: &Tensor, attention_mask: &Tensor) -> Result<Tensor> {
        let (b, s) = input_ids.dims2()?;
        let mask = build_mask(attention_mask, b, s, input_ids.device())?;
        let mut hidden = self.embed_tokens.forward(input_ids)?.to_dtype(DType::F32)?;

        // Capture the OUTPUT of layer `i` (index `i`, NOT `i+1`); run up to the last needed layer.
        let mut saved: Vec<(usize, Tensor)> = Vec::with_capacity(self.out_layers.len());
        for (i, layer) in self.layers.iter().enumerate() {
            hidden = layer.forward(&hidden, &self.rotary, &mask)?;
            if self.out_layers.contains(&i) {
                saved.push((i, hidden.clone()));
            }
        }

        let pick = |idx: usize| -> Result<Tensor> {
            saved
                .iter()
                .find(|(k, _)| *k == idx)
                .map(|(_, v)| v.clone())
                .ok_or_else(|| {
                    candle_gen::candle_core::Error::Msg(format!(
                        "ideogram te: hidden state {idx} not captured"
                    ))
                })
        };
        // INTERLEAVE the layers into the feature axis: each captured `[B,S,H]` → `[B,S,H,1]`, cat on
        // the last axis to `[B,S,H,n]`, reshape to `[B,S,H·n]` so feature `f = h·n + layer`.
        let expanded: Vec<Tensor> = self
            .out_layers
            .iter()
            .map(|&idx| pick(idx)?.unsqueeze(D::Minus1))
            .collect::<Result<_>>()?;
        let stacked = Tensor::cat(&expanded, D::Minus1)?; // [B, S, H, n]
        let (bb, ss, h, n) = stacked.dims4()?;
        stacked.reshape((bb, ss, h * n))
    }
}

/// Additive attention mask `[B, 1, S, S]` (f32): `0` where a query `i` may attend key `j` (causal
/// `j <= i` AND `j` not padding), `-inf` otherwise. Built host-side.
fn build_mask(attention_mask: &Tensor, b: usize, s: usize, device: &Device) -> Result<Tensor> {
    let am: Vec<i64> = attention_mask
        .to_dtype(DType::I64)?
        .flatten_all()?
        .to_vec1::<i64>()?;
    let mut data = vec![0f32; b * s * s];
    for bi in 0..b {
        for i in 0..s {
            for j in 0..s {
                let allowed = j <= i && am[bi * s + j] == 1;
                if !allowed {
                    data[(bi * s + i) * s + j] = f32::NEG_INFINITY;
                }
            }
        }
    }
    Tensor::from_vec(data, (b, 1, s, s), device)
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::HashMap;

    /// A tiny valid Qwen3-VL text-encoder weight map (2 layers, hidden 6, GQA 2/1, head_dim 4) drawn
    /// as **bf16** — modelling the hosted TE, whose weights ship bf16 on disk.
    fn tiny_ideogram_te_map() -> (HashMap<String, Tensor>, Ideogram4TextEncoderConfig) {
        let cfg = Ideogram4TextEncoderConfig {
            hidden_size: 6,
            num_layers: 2,
            num_heads: 2,
            num_kv_heads: 1,
            head_dim: 4,
            intermediate_size: 8,
            rms_norm_eps: 1e-6,
            rope_theta: 5_000_000.0,
            vocab_size: 12,
        };
        let (nh, nkv, hd, h, inter, vocab) = (
            cfg.num_heads,
            cfg.num_kv_heads,
            cfg.head_dim,
            cfg.hidden_size,
            cfg.intermediate_size,
            cfg.vocab_size,
        );
        let bf16 = |shape: &[usize]| {
            Tensor::randn(0f32, 0.5f32, shape, &Device::Cpu)
                .unwrap()
                .to_dtype(DType::BF16)
                .unwrap()
        };
        let mut t = HashMap::new();
        t.insert(
            "language_model.embed_tokens.weight".to_string(),
            bf16(&[vocab, h]),
        );
        for i in 0..cfg.num_layers {
            let p = format!("language_model.layers.{i}");
            t.insert(format!("{p}.input_layernorm.weight"), bf16(&[h]));
            t.insert(format!("{p}.post_attention_layernorm.weight"), bf16(&[h]));
            t.insert(format!("{p}.self_attn.q_proj.weight"), bf16(&[nh * hd, h]));
            t.insert(format!("{p}.self_attn.k_proj.weight"), bf16(&[nkv * hd, h]));
            t.insert(format!("{p}.self_attn.v_proj.weight"), bf16(&[nkv * hd, h]));
            t.insert(format!("{p}.self_attn.o_proj.weight"), bf16(&[h, nh * hd]));
            t.insert(format!("{p}.self_attn.q_norm.weight"), bf16(&[hd]));
            t.insert(format!("{p}.self_attn.k_norm.weight"), bf16(&[hd]));
            t.insert(format!("{p}.mlp.gate_proj.weight"), bf16(&[inter, h]));
            t.insert(format!("{p}.mlp.up_proj.weight"), bf16(&[inter, h]));
            t.insert(format!("{p}.mlp.down_proj.weight"), bf16(&[h, inter]));
        }
        (t, cfg)
    }

    /// The parity gate (sc-12828): a bf16 weight **store** with f32 **compute** is bit-identical to an
    /// f32 store — the disk weights are bf16, so an f32 store only widens them and every matmul still
    /// runs f32 (the projections upcast via `QLinear::forward_upcast`, the RMSNorm weights load f32 via
    /// `rms_norm_f32`, and the embedding is upcast to f32). Reverting any of those makes the bf16 path
    /// a dtype-mismatch error, so this goes RED — it is not a tautology that passes with the win ripped
    /// out. CPU-runnable precisely because the compute never leaves f32.
    #[test]
    fn bf16_store_prompt_embeds_is_bit_identical_to_f32_store() {
        let (map, cfg) = tiny_ideogram_te_map();
        let out_layers = [0usize, 1];
        let dev = Device::Cpu;
        let ids = Tensor::from_vec(vec![1u32, 5, 3, 9], (1, 4), &dev).unwrap();
        let attn = Tensor::ones((1, 4), DType::U32, &dev).unwrap();

        let vb_f32 = VarBuilder::from_tensors(map.clone(), DType::F32, &dev);
        let out_f32 = Ideogram4TextEncoder::new(&cfg, &out_layers, 64, vb_f32)
            .unwrap()
            .prompt_embeds(&ids, &attn)
            .unwrap();

        let vb_bf16 = VarBuilder::from_tensors(map, DType::BF16, &dev);
        let out_bf16 = Ideogram4TextEncoder::new(&cfg, &out_layers, 64, vb_bf16)
            .unwrap()
            .prompt_embeds(&ids, &attn)
            .unwrap();

        // Interleaved features [B, S, hidden·n] (n = out_layers.len()); f32 either way.
        assert_eq!(out_f32.dtype(), DType::F32);
        assert_eq!(out_bf16.dtype(), DType::F32);
        assert_eq!(out_f32.dims(), &[1, 4, 12]);
        let a = out_f32.flatten_all().unwrap().to_vec1::<f32>().unwrap();
        let b = out_bf16.flatten_all().unwrap().to_vec1::<f32>().unwrap();
        assert!(
            a.iter().all(|x| x.is_finite()),
            "prompt embeds must be finite"
        );
        assert_eq!(
            a, b,
            "bf16-store prompt_embeds must be bit-identical to the f32-store forward"
        );
    }
}
