//! ChatGLM3-6B forward — Kolors' text encoder. The candle port of `mlx-gen-kolors`'s `chatglm3.rs`,
//! a faithful reproduction of the diffusers `KolorsPipeline` reference (`ChatGLMModel`, encoder-only:
//! no LM head, no generation, no KV cache), driven by `encode_prompt` with `output_hidden_states`.
//!
//! ChatGLM3-specific pieces:
//!  - **Half-dim interleaved RoPE.** Rotary applies to the **first `rotary_dim` (64)** of the 128-wide
//!    head dim, with **adjacent-pair** interleaving `(x[2i], x[2i+1])` (NOT the HF half-split); the
//!    trailing 64 dims pass through unrotated. θ = 10000, constant across layers.
//!  - **Fused, biased `query_key_value`.** One `[4608, 4096]` Linear (with bias) →
//!    q (32·128) + k (2·128) + v (2·128); **multi-query attention, 2 KV groups** broadcast to 32 query
//!    heads. The output `dense` proj is bias-less.
//!  - **RMSNorm = plain `weight · x̂`** (eps 1e-5) — candle's `RmsNorm` (NOT Gemma's `(1 + weight)`).
//!  - **GLMBlock pre-norm residual**: `h = x + dense(attn(input_ln(x)))`;
//!    `out = h + mlp(post_attn_ln(h))`. MLP = `dense_4h_to_h(silu(g) · u)` where `dense_h_to_4h` fuses
//!    gate+up (out `2·13696`); the activation is SiLU.
//!  - **Standard `1/√head_dim` scaled-dot-product** + plain softmax (the legacy
//!    `apply_query_key_layer_scaling` does not run on the SDPA path the reference takes).
//!
//! ### Output contract (what Kolors consumes)
//! Kolors uses **`hidden_states[-2]`** (penultimate, layer-26 output) as the cross-attention context and
//! **`hidden_states[-1]` at the last sequence position** as the pooled add-embedding — neither is
//! final-normed, so the encoder's `final_layernorm` weight is never consumed and is not loaded.
//!
//! ### Packed q4/q8 tiers (sc-10819, epic 9083)
//! The four GLM projections (`query_key_value`, `self_attention.dense`, `mlp.dense_h_to_4h`,
//! `mlp.dense_4h_to_h`) route through the shared [`candle_gen::quant`] seam
//! ([`QLinear::linear_detect_gs`]) — a **pure superset** of the dense build: absent a `.scales` sibling
//! each takes the plain dense path unchanged (byte-identical to the pre-sc-10819 `candle_nn::Linear`
//! tower, so the txt2img / control / IP-Adapter dense lanes are untouched), and present it builds the
//! quantized projection straight from the packed `{weight u32, scales, biases}` triple. The
//! `SceneWorks/kolors-mlx` q4/q8 tiers pack exactly these four projections (mlx-gen #659 —
//! `prequantize_turnkey`); the embedding + RMSNorms stay dense, matching the MLX packer. `group_size`
//! is threaded from the caller (the packed `text_encoder/config.json`'s `quantization.group_size`, sc-9410).

use candle_gen::candle_core::Result as CandleResult;
use candle_gen::candle_core::{Device, Tensor, D};
use candle_gen::candle_nn::{self as nn, Module, RmsNorm, VarBuilder};
use candle_gen::quant::{QLinear, MLX_GROUP_SIZE};

use crate::config::ChatGlmConfig;
use crate::tokenizer::KolorsTokens;

struct GlmBlock {
    input_ln: RmsNorm,
    post_attn_ln: RmsNorm,
    // The four GLM projections packed-detect (sc-10819): a `.scales`-sibling MLX tier loads each straight
    // from its packed triple; a dense checkpoint takes `QLinear`'s dense arm (byte-identical to the
    // pre-sc-10819 `candle_nn::Linear`).
    qkv: QLinear,     // fused query_key_value, biased
    dense: QLinear,   // output projection, bias-less
    h_to_4h: QLinear, // fused gate+up, bias-less
    h4_to_h: QLinear, // bias-less
}

/// The ChatGLM3-6B backbone used as the Kolors text encoder.
pub struct ChatGlmModel {
    embed: nn::Embedding,
    layers: Vec<GlmBlock>,
    cfg: ChatGlmConfig,
    device: Device,
}

impl ChatGlmModel {
    /// Build from the Kolors `text_encoder/` VarBuilder at the default MLX group size (64) — the dense
    /// entry point every non-quant caller (txt2img dense lane, control, IP-Adapter) uses. Packed-detect
    /// is a no-op on a dense checkpoint (no `.scales` sibling), so this is byte-identical to the
    /// pre-sc-10819 `candle_nn::Linear` tower.
    pub fn new(cfg: ChatGlmConfig, vb: VarBuilder) -> CandleResult<Self> {
        Self::new_gs(cfg, vb, MLX_GROUP_SIZE)
    }

    /// Build from the Kolors `text_encoder/` VarBuilder (the `embedding.word_embeddings.*` /
    /// `encoder.layers.{i}.*` diffusers layout) at an explicit MLX packed `group_size` (sc-10819) —
    /// threaded from the packed `text_encoder/config.json`'s `quantization.group_size` (sc-9410) so a
    /// packed tier repacks on the grid it was packed at. The dense arm ignores `group_size`.
    /// `encoder.final_layernorm.*` is present in the snapshot but unused by Kolors conditioning, so it is
    /// not loaded.
    pub fn new_gs(cfg: ChatGlmConfig, vb: VarBuilder, group_size: usize) -> CandleResult<Self> {
        let device = vb.device().clone();
        let embed = nn::embedding(
            cfg.vocab_size,
            cfg.hidden_size,
            vb.pp("embedding.word_embeddings"),
        )?;
        let attn_out = cfg.num_heads * cfg.head_dim; // 4096
        let mut layers = Vec::with_capacity(cfg.num_layers);
        for i in 0..cfg.num_layers {
            let b = vb.pp(format!("encoder.layers.{i}"));
            let attn = b.pp("self_attention");
            let mlp = b.pp("mlp");
            layers.push(GlmBlock {
                input_ln: nn::rms_norm(cfg.hidden_size, cfg.rms_eps, b.pp("input_layernorm"))?,
                post_attn_ln: nn::rms_norm(
                    cfg.hidden_size,
                    cfg.rms_eps,
                    b.pp("post_attention_layernorm"),
                )?,
                qkv: QLinear::linear_detect_gs(
                    cfg.hidden_size,
                    cfg.qkv_out(),
                    &attn,
                    "query_key_value",
                    true,
                    group_size,
                )?,
                dense: QLinear::linear_detect_gs(
                    attn_out,
                    cfg.hidden_size,
                    &attn,
                    "dense",
                    false,
                    group_size,
                )?,
                h_to_4h: QLinear::linear_detect_gs(
                    cfg.hidden_size,
                    2 * cfg.ffn_hidden,
                    &mlp,
                    "dense_h_to_4h",
                    false,
                    group_size,
                )?,
                h4_to_h: QLinear::linear_detect_gs(
                    cfg.ffn_hidden,
                    cfg.hidden_size,
                    &mlp,
                    "dense_4h_to_h",
                    false,
                    group_size,
                )?,
            });
        }
        Ok(Self {
            embed,
            layers,
            cfg,
            device,
        })
    }

    /// Extract Kolors conditioning for one prompt: `(context [1, S, hidden], pooled [1, hidden])`.
    /// `context` = the penultimate hidden state (layer-26 output); `pooled` = the final hidden state at
    /// the **last sequence position**. Threads the tokenizer's left-padded `position_ids` into RoPE.
    /// Batch size 1 (production encode is always B==1; Kolors CFG-batches the two prompts' results, not
    /// the encode itself).
    pub fn encode_prompt(&self, tokens: &KolorsTokens) -> CandleResult<(Tensor, Tensor)> {
        let s = tokens.input_ids.len();
        let hidden = self.cfg.hidden_size;
        let ids = Tensor::from_vec(tokens.input_ids.clone(), (1, s), &self.device)?;
        let mut h = self.embed.forward(&ids)?; // [1, s, hidden]
        let mask = self.causal_padding_mask(&tokens.attention_mask, s)?;
        let (cos, sin) = self.rope_tables(&tokens.position_ids)?;

        // context = output of layer `num_layers - 2` (hidden_states[-2]); pooled source = final h.
        let context_idx = self.cfg.num_layers - 2;
        let mut context: Option<Tensor> = None;
        for (i, layer) in self.layers.iter().enumerate() {
            h = self.block(layer, &h, &mask, &cos, &sin)?;
            if i == context_idx {
                context = Some(h.clone());
            }
        }
        let context = context.expect("chatglm3 has >= 2 layers");
        // pooled = h[:, s-1, :] (the last sequence position of the final hidden state).
        let pooled = h.narrow(1, s - 1, 1)?.squeeze(1)?.reshape((1, hidden))?;
        Ok((context, pooled))
    }

    /// Test-only: whether every GLM projection (qkv / dense / h_to_4h / h4_to_h) across every layer
    /// loaded packed — i.e. a pre-quantized MLX tier was detected on all four projections (sc-10819).
    #[cfg(test)]
    pub(crate) fn all_projections_packed(&self) -> bool {
        self.layers.iter().all(|l| {
            l.qkv.is_quantized()
                && l.dense.is_quantized()
                && l.h_to_4h.is_quantized()
                && l.h4_to_h.is_quantized()
        })
    }

    /// One GLM block: pre-norm residual attention then pre-norm residual SwiGLU MLP.
    fn block(
        &self,
        layer: &GlmBlock,
        x: &Tensor,
        mask: &Tensor,
        cos: &Tensor,
        sin: &Tensor,
    ) -> CandleResult<Tensor> {
        let r = self.attn(layer, &layer.input_ln.forward(x)?, mask, cos, sin)?;
        let h = (x + r)?;
        let r = self.mlp(layer, &layer.post_attn_ln.forward(&h)?)?;
        &h + r
    }

    /// Fused-QKV GQA attention with GLM half-dim interleaved RoPE and an additive causal+padding mask.
    fn attn(
        &self,
        layer: &GlmBlock,
        x: &Tensor,
        mask: &Tensor,
        cos: &Tensor,
        sin: &Tensor,
    ) -> CandleResult<Tensor> {
        let (b, s, _) = x.dims3()?;
        let (nh, kv, d) = (
            self.cfg.num_heads,
            self.cfg.num_kv_groups,
            self.cfg.head_dim,
        );
        // Fused QKV → [b, s, (nh + 2·kv), d]; head-major: heads 0..nh = q, nh..nh+kv = k, … = v.
        let qkv = layer.qkv.forward(x)?.reshape((b, s, nh + 2 * kv, d))?;
        let q = qkv.narrow(2, 0, nh)?;
        let k = qkv.narrow(2, nh, kv)?;
        let v = qkv.narrow(2, nh + kv, kv)?;

        let q = self.apply_rope(&q, cos, sin)?;
        let k = self.apply_rope(&k, cos, sin)?;

        // [b, s, heads, d] → [b, heads, s, d].
        let q = q.transpose(1, 2)?.contiguous()?;
        let k = k.transpose(1, 2)?.contiguous()?;
        let v = v.transpose(1, 2)?.contiguous()?;
        // GQA: broadcast each KV head to `nh/kv` contiguous query heads.
        let g = nh / kv;
        let k = repeat_kv(&k, g)?;
        let v = repeat_kv(&v, g)?;

        let scale = 1.0 / (d as f64).sqrt();
        let scores = (q.matmul(&k.transpose(2, 3)?.contiguous()?)? * scale)?; // [b, nh, s, s]
        let scores = scores.broadcast_add(mask)?; // mask [b, 1, s, s] broadcasts over heads
        let probs = nn::ops::softmax_last_dim(&scores)?;
        let out = probs.matmul(&v)?; // [b, nh, s, d]
        let out = out.transpose(1, 2)?.contiguous()?.reshape((b, s, nh * d))?;
        layer.dense.forward(&out)
    }

    /// SwiGLU MLP: `dense_h_to_4h` fuses gate+up (out `2·ffn`); `silu(gate)·up → dense_4h_to_h`.
    fn mlp(&self, layer: &GlmBlock, x: &Tensor) -> CandleResult<Tensor> {
        let ffn = self.cfg.ffn_hidden;
        let gu = layer.h_to_4h.forward(x)?;
        let gate = gu.narrow(D::Minus1, 0, ffn)?;
        let up = gu.narrow(D::Minus1, ffn, ffn)?;
        let gated = (nn::ops::silu(&gate)? * up)?;
        layer.h4_to_h.forward(&gated)
    }

    /// Rotary `(cos, sin)`, each `[1, seq, 1, rotary_dim/2]`, for the given absolute `positions` (one
    /// per sequence slot — Kolors' left-padded `position_ids`). θ = 10000, computed once.
    fn rope_tables(&self, positions: &[i64]) -> CandleResult<(Tensor, Tensor)> {
        let half = self.cfg.rotary_dim / 2; // 32
        let rot = self.cfg.rotary_dim as f64; // 64
        let inv_freq: Vec<f32> = (0..half)
            .map(|j| (1.0 / self.cfg.rope_base.powf((2 * j) as f64 / rot)) as f32)
            .collect();
        let seq = positions.len();
        let mut freqs = Vec::with_capacity(seq * half);
        for &p in positions {
            for &f in &inv_freq {
                freqs.push(p as f32 * f);
            }
        }
        let freqs = Tensor::from_vec(freqs, (1, seq, 1, half), &self.device)?;
        Ok((freqs.cos()?, freqs.sin()?))
    }

    /// GLM interleaved half-dim RoPE on `x` `[b, s, heads, d]`: rotate the first `rotary_dim` dims as
    /// adjacent pairs `(x[2i], x[2i+1])` against `(cos, sin)` `[1, s, 1, rotary_dim/2]`; pass the
    /// trailing `d - rotary_dim` dims through.
    fn apply_rope(&self, x: &Tensor, cos: &Tensor, sin: &Tensor) -> CandleResult<Tensor> {
        let (b, s, h, d) = x.dims4()?;
        let rot = self.cfg.rotary_dim;
        let half = rot / 2;
        let x_rot = x.narrow(3, 0, rot)?; // [b, s, h, rot]
        let x_pass = x.narrow(3, rot, d - rot)?; // [b, s, h, d-rot]
        let xr = x_rot.reshape((b, s, h, half, 2))?;
        let x0 = xr.narrow(4, 0, 1)?.squeeze(4)?; // even lane [b, s, h, half]
        let x1 = xr.narrow(4, 1, 1)?.squeeze(4)?; // odd lane
        let out0 = (x0.broadcast_mul(cos)? - x1.broadcast_mul(sin)?)?;
        let out1 = (x1.broadcast_mul(cos)? + x0.broadcast_mul(sin)?)?;
        // Re-interleave: stack on a new trailing axis then fold back to `rot`.
        let rotated = Tensor::stack(&[&out0, &out1], 4)?.reshape((b, s, h, rot))?;
        Tensor::cat(&[&rotated, &x_pass], 3)
    }

    /// Additive `[1, 1, s, s]` mask in f32 (B==1), mirroring the reference `get_masks`: a real query row
    /// `i` attends key `j` iff causal (`j ≤ i`) and the key is not padding; a padding query row attends
    /// everything (so its hidden state is deterministic). Disallowed → a large finite negative.
    fn causal_padding_mask(&self, attention_mask: &[u32], s: usize) -> CandleResult<Tensor> {
        let data = causal_padding_data(attention_mask, 1, s);
        Tensor::from_vec(data, (1, 1, s, s), &self.device)
    }
}

/// Broadcast each of `x`'s `kv` heads to `g` contiguous query heads — `[b, kv, s, d] → [b, kv·g, s, d]`,
/// head `kv_idx·g + j` served by KV head `kv_idx` (so query head `i` uses KV head `i / g`). `g == 1` is
/// a no-op (SDXL/standard MHA).
fn repeat_kv(x: &Tensor, g: usize) -> CandleResult<Tensor> {
    if g == 1 {
        return Ok(x.clone());
    }
    let (b, kv, s, d) = x.dims4()?;
    x.unsqueeze(2)?
        .broadcast_as((b, kv, g, s, d))?
        .contiguous()?
        .reshape((b, kv * g, s, d))
}

/// Flat `[b·s·s]` additive-mask data: for each batch row, query `i` is masked off key `j` (value
/// `-1e30`) unless that row's own padding allows it (causal AND non-pad key, OR a pad query). Pure (no
/// device), so the per-row mask logic is unit-testable. Assumes `m.len() == b·s`.
fn causal_padding_data(m: &[u32], b: usize, s: usize) -> Vec<f32> {
    let neg = -1e30f32;
    let mut data = vec![0f32; b * s * s];
    for bi in 0..b {
        let row = &m[bi * s..(bi + 1) * s];
        for i in 0..s {
            let pad_query = row[i] == 0;
            for j in 0..s {
                let allowed = pad_query || (j <= i && row[j] != 0);
                if !allowed {
                    data[(bi * s + i) * s + j] = neg;
                }
            }
        }
    }
    data
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::tokenizer::KolorsTokens;
    use candle_gen::candle_core::safetensors::MmapedSafetensors;
    use candle_gen::candle_core::DType;
    use std::collections::HashMap;

    const GS: usize = 64;

    /// A tiny ChatGLM3 config exercising every packed path cheaply on CPU: every Linear in-dim is a
    /// multiple of the group 64, `num_layers >= 2` (so the penultimate context read is valid), GQA with
    /// 2 KV groups broadcast to 4 query heads.
    fn tiny_cfg() -> ChatGlmConfig {
        ChatGlmConfig {
            hidden_size: 64,
            num_layers: 2,
            num_heads: 4,
            num_kv_groups: 2,
            head_dim: 16, // num_heads·head_dim = 64 = hidden_size
            ffn_hidden: 64,
            rms_eps: 1e-5,
            rope_base: 10_000.0,
            rotary_dim: 8,
            vocab_size: 64,
        }
    }

    /// The exact affine grid an MLX Q4 pack of `[out, in]` represents (row-major group-64 affine), so a
    /// dense reference can be built from the SAME numbers the packed path repacks. Mirrors the sc-9527
    /// SDXL-CLIP parity test's `grid`.
    fn grid(out_f: usize, in_f: usize) -> Vec<f32> {
        let codes: Vec<u8> = (0..out_f * in_f)
            .map(|i| ((i * 5 + 3) % 16) as u8)
            .collect();
        let gpr = in_f / GS;
        let scale = |g: usize| 0.03125 * (g as f32 + 1.0);
        let bias = |g: usize| -0.25 - 0.1 * g as f32;
        (0..out_f * in_f)
            .map(|i| {
                let (row, col) = (i / in_f, i % in_f);
                let g = row * gpr + col / GS;
                scale(g) * codes[i] as f32 + bias(g)
            })
            .collect()
    }

    /// Insert an MLX Q4 packed triple (`{base}.weight` u32 LSB-first nibbles + per-group `.scales` /
    /// `.biases`) for `[out, in]`, plus an optional dense `.bias`.
    fn pack(map: &mut HashMap<String, Tensor>, base: &str, out_f: usize, in_f: usize, bias: bool) {
        let dev = Device::Cpu;
        let codes: Vec<u8> = (0..out_f * in_f)
            .map(|i| ((i * 5 + 3) % 16) as u8)
            .collect();
        let words: Vec<u32> = codes
            .chunks_exact(8)
            .map(|c| {
                c.iter()
                    .enumerate()
                    .fold(0u32, |acc, (i, &q)| acc | ((q as u32 & 0xF) << (4 * i)))
            })
            .collect();
        let groups = out_f * in_f / GS;
        let scales: Vec<f32> = (0..groups).map(|g| 0.03125 * (g as f32 + 1.0)).collect();
        let biases: Vec<f32> = (0..groups).map(|g| -0.25 - 0.1 * g as f32).collect();
        let gpr = in_f / GS;
        map.insert(
            format!("{base}.weight"),
            Tensor::from_vec(words, (out_f, in_f / 8), &dev).unwrap(),
        );
        map.insert(
            format!("{base}.scales"),
            Tensor::from_vec(scales, (out_f, gpr), &dev).unwrap(),
        );
        map.insert(
            format!("{base}.biases"),
            Tensor::from_vec(biases, (out_f, gpr), &dev).unwrap(),
        );
        if bias {
            map.insert(
                format!("{base}.bias"),
                Tensor::zeros((out_f,), DType::F32, &dev).unwrap(),
            );
        }
    }

    /// Insert a dense `{base}.weight` (+ optional `.bias`) at the SAME affine grid the packed triple
    /// represents — so packed-vs-dense parity is isolated to the quantization.
    fn dense_lin(
        map: &mut HashMap<String, Tensor>,
        base: &str,
        out_f: usize,
        in_f: usize,
        bias: bool,
    ) {
        let dev = Device::Cpu;
        map.insert(
            format!("{base}.weight"),
            Tensor::from_vec(grid(out_f, in_f), (out_f, in_f), &dev).unwrap(),
        );
        if bias {
            map.insert(
                format!("{base}.bias"),
                Tensor::zeros((out_f,), DType::F32, &dev).unwrap(),
            );
        }
    }

    /// Build a full ChatGLM3 checkpoint map (embedding + per-layer RMSNorms dense; the four projections
    /// packed or dense at the SAME grid) for `cfg`.
    fn build_checkpoint(cfg: &ChatGlmConfig, packed: bool) -> HashMap<String, Tensor> {
        let dev = Device::Cpu;
        let mut m = HashMap::new();
        let h = cfg.hidden_size;
        m.insert(
            "embedding.word_embeddings.weight".into(),
            Tensor::from_vec(grid(cfg.vocab_size, h), (cfg.vocab_size, h), &dev).unwrap(),
        );
        let attn_out = cfg.num_heads * cfg.head_dim;
        let put = |m: &mut HashMap<String, Tensor>, base: &str, o: usize, i: usize, b: bool| {
            if packed {
                pack(m, base, o, i, b);
            } else {
                dense_lin(m, base, o, i, b);
            }
        };
        for l in 0..cfg.num_layers {
            let p = format!("encoder.layers.{l}");
            m.insert(
                format!("{p}.input_layernorm.weight"),
                Tensor::ones((h,), DType::F32, &dev).unwrap(),
            );
            m.insert(
                format!("{p}.post_attention_layernorm.weight"),
                Tensor::ones((h,), DType::F32, &dev).unwrap(),
            );
            put(
                &mut m,
                &format!("{p}.self_attention.query_key_value"),
                cfg.qkv_out(),
                h,
                true,
            );
            put(
                &mut m,
                &format!("{p}.self_attention.dense"),
                h,
                attn_out,
                false,
            );
            put(
                &mut m,
                &format!("{p}.mlp.dense_h_to_4h"),
                2 * cfg.ffn_hidden,
                h,
                false,
            );
            put(
                &mut m,
                &format!("{p}.mlp.dense_4h_to_h"),
                h,
                cfg.ffn_hidden,
                false,
            );
        }
        m
    }

    fn vb_from_map(
        map: HashMap<String, Tensor>,
        tag: &str,
    ) -> (VarBuilder<'static>, std::path::PathBuf) {
        let tmp = std::env::temp_dir().join(format!(
            "sc10819_glm_{tag}_{}.safetensors",
            std::process::id()
        ));
        candle_gen::candle_core::safetensors::save(&map, &tmp).unwrap();
        // SAFETY: just-written file, untouched for the test's lifetime.
        let st = unsafe { MmapedSafetensors::new(&tmp).unwrap() };
        let vb = VarBuilder::from_backend(Box::new(st), DType::F32, Device::Cpu);
        (vb, tmp)
    }

    /// A short left-padded token batch (real tokens after one pad slot) — the shape `encode_prompt`
    /// threads through the mask + RoPE.
    fn tiny_tokens(cfg: &ChatGlmConfig) -> KolorsTokens {
        let s = 5usize;
        let input_ids: Vec<u32> = (0..s as u32).map(|i| i % cfg.vocab_size as u32).collect();
        // One leading pad, then real; position_ids restart at 0 for the first real token (left-pad).
        let attention_mask = vec![0u32, 1, 1, 1, 1];
        let position_ids: Vec<i64> = vec![0, 0, 1, 2, 3];
        KolorsTokens {
            input_ids,
            attention_mask,
            position_ids,
        }
    }

    fn cosine(a: &Tensor, b: &Tensor) -> f32 {
        let a = a
            .to_dtype(DType::F32)
            .unwrap()
            .flatten_all()
            .unwrap()
            .to_vec1::<f32>()
            .unwrap();
        let b = b
            .to_dtype(DType::F32)
            .unwrap()
            .flatten_all()
            .unwrap()
            .to_vec1::<f32>()
            .unwrap();
        let (mut dot, mut na, mut nb) = (0f64, 0f64, 0f64);
        for (x, y) in a.iter().zip(b.iter()) {
            dot += (*x as f64) * (*y as f64);
            na += (*x as f64) * (*x as f64);
            nb += (*y as f64) * (*y as f64);
        }
        (dot / (na.sqrt() * nb.sqrt() + 1e-12)) as f32
    }

    /// A `.scales`-sibling checkpoint routes every GLM projection to the packed path; a dense one falls
    /// back — the packed-detect superset (sc-10819).
    #[test]
    fn packed_detect_fires_on_chatglm_layout() {
        let cfg = tiny_cfg();
        let (vb_p, tmp_p) = vb_from_map(build_checkpoint(&cfg, true), "detect_packed");
        let packed = ChatGlmModel::new_gs(cfg, vb_p, GS).unwrap();
        assert!(
            packed.all_projections_packed(),
            "every GLM projection must load packed on a `.scales` checkpoint"
        );
        let (vb_d, tmp_d) = vb_from_map(build_checkpoint(&cfg, false), "detect_dense");
        let dense = ChatGlmModel::new_gs(cfg, vb_d, GS).unwrap();
        assert!(
            !dense.all_projections_packed(),
            "a dense (no `.scales`) checkpoint must fall back to dense Linears"
        );
        std::fs::remove_file(&tmp_p).ok();
        std::fs::remove_file(&tmp_d).ok();
    }

    /// The packed ChatGLM3 encode matches the dense encode built from the SAME affine grid, within the
    /// shared `candle_gen::quant` packed-vs-dense tolerance (cosine > 0.99999) — proving the packed load
    /// path is numerically faithful for BOTH Kolors conditioning outputs (penultimate context + pooled).
    #[test]
    fn packed_vs_dense_encode_parity() {
        let cfg = tiny_cfg();
        let (vb_p, tmp_p) = vb_from_map(build_checkpoint(&cfg, true), "parity_packed");
        let (vb_d, tmp_d) = vb_from_map(build_checkpoint(&cfg, false), "parity_dense");
        let packed = ChatGlmModel::new_gs(cfg, vb_p, GS).unwrap();
        let dense = ChatGlmModel::new_gs(cfg, vb_d, GS).unwrap();
        assert!(packed.all_projections_packed());
        assert!(!dense.all_projections_packed());

        let toks = tiny_tokens(&cfg);
        let (ctx_p, pooled_p) = packed.encode_prompt(&toks).unwrap();
        let (ctx_d, pooled_d) = dense.encode_prompt(&toks).unwrap();
        assert_eq!(ctx_p.dims(), ctx_d.dims());
        let c_ctx = cosine(&ctx_p, &ctx_d);
        let c_pooled = cosine(&pooled_p, &pooled_d);
        assert!(c_ctx > 0.99999, "packed vs dense context cosine {c_ctx:.6}");
        assert!(
            c_pooled > 0.99999,
            "packed vs dense pooled cosine {c_pooled:.6}"
        );

        std::fs::remove_file(&tmp_p).ok();
        std::fs::remove_file(&tmp_d).ok();
    }

    #[test]
    fn causal_padding_data_uses_per_row_padding() {
        // Row 0 all-real ([1,1,1]); row 1 key 0 padded ([0,1,1]).
        let s = 3;
        let m = [1u32, 1, 1, /* row 1 */ 0, 1, 1];
        let data = causal_padding_data(&m, 2, s);
        let neg = -1e30f32;
        let at = |bi: usize, i: usize, j: usize| data[(bi * s + i) * s + j];

        // Row 0 is a plain causal mask.
        assert_eq!(at(0, 0, 1), neg); // future key masked
        assert_eq!(at(0, 1, 0), 0.0); // past real key attended

        // Row 1, query 1 (real): key 0 is padding ⇒ masked even though causal.
        assert_eq!(at(1, 1, 0), neg);
        assert_eq!(at(1, 1, 1), 0.0);
        assert_ne!(at(0, 1, 0), at(1, 1, 0));

        // Row 1, query 0 is itself padding ⇒ attends everything.
        assert_eq!(at(1, 0, 0), 0.0);
        assert_eq!(at(1, 0, 2), 0.0);
        assert_ne!(at(0, 0, 2), at(1, 0, 2));
    }
}
