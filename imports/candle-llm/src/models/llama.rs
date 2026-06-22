//! Generic Llama-family causal decoder (Llama / Mistral / Qwen3).
//!
//! The Candle port of `mlx-llm`'s `LlamaModel`, modelled alongside `candle-gen-sensenova`'s
//! hand-rolled Qwen3 stack. Attention optionally applies per-head q/k RMSNorm (Qwen3); projections
//! are held behind [`Projection`] so a model can be quantized on load. The forward is `&self`; the
//! KV cache is the only mutable state, threaded in as `&mut dyn KvCache`.
//!
//! Shapes are batch-capable (`[batch, seq, …]`). `head_dim` is taken from config and may differ from
//! `hidden_size / num_heads` (e.g. Qwen3-0.6B: hidden 1024, 16 heads, head_dim 128). Compute runs in
//! the device's [`compute_dtype`] (bf16 on GPU, f32 on CPU).

use candle_core::{DType, Device, Tensor};
use candle_nn::{Linear, Module};

use crate::config::{Architecture, LlamaConfig};
use crate::device::compute_dtype;
use crate::error::Result;
use crate::primitives::attention::{sdpa, AttnMask};
use crate::primitives::kv_cache::KvCache;
use crate::primitives::nn::{embed, gelu, rms_norm, silu, soft_cap};
use crate::primitives::projection::{Projection, QuantSpec};
use crate::primitives::rope::{apply_rope, Rope};
use crate::primitives::{repeat_kv, ContiguousKvCache, Weights};

/// A loaded causal decoder.
pub struct LlamaModel {
    embed_tokens: Tensor,
    layers: Vec<LlamaLayer>,
    norm: Tensor,
    lm_head: Linear,
    rope: Rope,
    cfg: LlamaConfig,
    dtype: DType,
    device: Device,
    quantized: bool,
    /// Gemma scales token embeddings by √hidden; `None` ⇒ no scaling.
    embed_scale: Option<f64>,
    /// Gemma-2 final-logit soft-cap; `None` ⇒ no cap.
    final_softcap: Option<f32>,
}

impl LlamaModel {
    /// Build from a loaded checkpoint (dense). `prefix` is the weight-key prefix (`""` for a plain
    /// `*ForCausalLM`, e.g. `"language_model"` for a VLM-nested decoder).
    pub fn from_weights(w: &Weights, prefix: &str, cfg: LlamaConfig) -> Result<Self> {
        Self::from_weights_with(w, prefix, cfg, None)
    }

    /// Build from a loaded checkpoint, optionally quantizing the attention/MLP projections on load.
    /// Embeddings, the LM head, and norms always stay dense.
    pub fn from_weights_with(
        w: &Weights,
        prefix: &str,
        cfg: LlamaConfig,
        quant: Option<QuantSpec>,
    ) -> Result<Self> {
        let device = w.device().clone();
        let dtype = compute_dtype(&device);
        let p = |suffix: &str| join(prefix, suffix);
        let req = |key: String| -> Result<Tensor> { Ok(w.require(&key)?.to_dtype(dtype)?) };
        let proj = |key: String| -> Result<Projection> { Projection::load(req(key)?, quant) };
        // Like `proj`, but also loads a sibling `.bias` when present (Qwen2 attention carries q/k/v
        // bias; Llama / Qwen3 / Phi-3 do not).
        let proj_b = |wkey: String| -> Result<Projection> {
            let stem = wkey.strip_suffix(".weight").unwrap_or(&wkey);
            let bkey = format!("{stem}.bias");
            let bias = if w.contains(&bkey) {
                Some(req(bkey)?)
            } else {
                None
            };
            Projection::load_with_bias(req(wkey)?, bias, quant)
        };
        // Gemma's norms are `(1 + weight)`; fold the +1 into the stored weight so the standard
        // `rms_norm` applies it. (Llama / Qwen3 norm weights are used verbatim.)
        let gemma = cfg.architecture.is_gemma2();
        let norm_w = |key: String| -> Result<Tensor> {
            let t = req(key)?;
            if gemma {
                Ok(t.affine(1.0, 1.0)?)
            } else {
                Ok(t)
            }
        };

        let embed_tokens = req(p("model.embed_tokens.weight"))?;
        let norm = norm_w(p("model.norm.weight"))?;
        let lm_head = if cfg.tie_word_embeddings {
            Linear::new(embed_tokens.clone(), None)
        } else {
            Linear::new(req(p("lm_head.weight"))?, None)
        };

        let qk_norm = cfg.has_qk_norm();
        let groups = cfg.groups() as usize;
        let num_heads = cfg.num_heads as usize;
        let num_kv_heads = cfg.num_kv_heads as usize;
        let head_dim = cfg.head_dim as usize;
        let scale = cfg.attn_scale();
        let eps = cfg.rms_norm_eps as f64;
        // Phi-3 fuses q‖k‖v into one `qkv_proj` and gate‖up into one `gate_up_proj`; the row spans the
        // split slices below carve out (each `[out, hidden]`, so the split is along axis 0).
        let qd = num_heads * head_dim;
        let kvd = num_kv_heads * head_dim;
        let inter = cfg.intermediate_size as usize;

        let mut layers = Vec::with_capacity(cfg.num_layers);
        for i in 0..cfg.num_layers {
            let lp = |suffix: &str| join(prefix, &format!("model.layers.{i}.{suffix}"));
            let (q_norm, k_norm) = if qk_norm {
                (
                    Some(req(lp("self_attn.q_norm.weight"))?),
                    Some(req(lp("self_attn.k_norm.weight"))?),
                )
            } else {
                (None, None)
            };
            // Attention projections: a packed `qkv_proj` (Phi-3, no bias) is split into q/k/v, else
            // the separate `q_proj`/`k_proj`/`v_proj` are loaded directly (with q/k/v bias for Qwen2).
            let (q, k, v) = {
                let packed = lp("self_attn.qkv_proj.weight");
                if w.contains(&packed) {
                    let qkv = req(packed)?; // [qd + 2*kvd, hidden]
                    (
                        Projection::load(qkv.narrow(0, 0, qd)?.contiguous()?, quant)?,
                        Projection::load(qkv.narrow(0, qd, kvd)?.contiguous()?, quant)?,
                        Projection::load(qkv.narrow(0, qd + kvd, kvd)?.contiguous()?, quant)?,
                    )
                } else {
                    (
                        proj_b(lp("self_attn.q_proj.weight"))?,
                        proj_b(lp("self_attn.k_proj.weight"))?,
                        proj_b(lp("self_attn.v_proj.weight"))?,
                    )
                }
            };

            // Feed-forward: a sparse Mixture-of-Experts bank (Qwen2-MoE) or a dense MLP. Gemma uses
            // GeGLU (gelu), everything else SwiGLU (silu).
            let ffn = if let Some(moe) = cfg.moe {
                let mut experts = Vec::with_capacity(moe.num_experts);
                for e in 0..moe.num_experts {
                    let ep = |s: &str| lp(&format!("mlp.experts.{e}.{s}"));
                    experts.push(LlamaMlp {
                        gate: proj(ep("gate_proj.weight"))?,
                        up: proj(ep("up_proj.weight"))?,
                        down: proj(ep("down_proj.weight"))?,
                        gelu: false,
                    });
                }
                Ffn::Moe(MoeMlp {
                    router: req(lp("mlp.gate.weight"))?, // [num_experts, hidden]
                    experts,
                    shared: LlamaMlp {
                        gate: proj(lp("mlp.shared_expert.gate_proj.weight"))?,
                        up: proj(lp("mlp.shared_expert.up_proj.weight"))?,
                        down: proj(lp("mlp.shared_expert.down_proj.weight"))?,
                        gelu: false,
                    },
                    shared_gate: req(lp("mlp.shared_expert_gate.weight"))?, // [1, hidden]
                    experts_per_tok: moe.num_experts_per_tok,
                    norm_topk_prob: moe.norm_topk_prob,
                })
            } else {
                // Dense MLP; Phi-3 fuses gate‖up into one weight, split along axis 0.
                let (gate, up) = {
                    let packed = lp("mlp.gate_up_proj.weight");
                    if w.contains(&packed) {
                        let gu = req(packed)?; // [2*inter, hidden]
                        (
                            Projection::load(gu.narrow(0, 0, inter)?.contiguous()?, quant)?,
                            Projection::load(gu.narrow(0, inter, inter)?.contiguous()?, quant)?,
                        )
                    } else {
                        (
                            proj(lp("mlp.gate_proj.weight"))?,
                            proj(lp("mlp.up_proj.weight"))?,
                        )
                    }
                };
                Ffn::Dense(LlamaMlp {
                    gate,
                    up,
                    down: proj(lp("mlp.down_proj.weight"))?,
                    gelu: gemma,
                })
            };

            // Gemma-2 / GLM-4 wrap the block in a 4-norm "sandwich" (pre+post for both attn and MLP);
            // the Llama shape has only the two pre-norms. The norm key names differ per family.
            let (post_attn_key, pre_ff_key, post_ff_key) = match cfg.architecture {
                Architecture::Glm4 => (
                    "post_self_attn_layernorm",
                    "post_attention_layernorm",
                    "post_mlp_layernorm",
                ),
                // Gemma-2 (and the default fallback for the post-attention norm name).
                _ => (
                    "post_attention_layernorm",
                    "pre_feedforward_layernorm",
                    "post_feedforward_layernorm",
                ),
            };
            let (pre_ff_ln, post_ff_ln) = if cfg.architecture.is_sandwich() {
                (
                    Some(norm_w(lp(&format!("{pre_ff_key}.weight")))?),
                    Some(norm_w(lp(&format!("{post_ff_key}.weight")))?),
                )
            } else {
                (None, None)
            };

            layers.push(LlamaLayer {
                input_ln: norm_w(lp("input_layernorm.weight"))?,
                post_ln: norm_w(lp(&format!("{post_attn_key}.weight")))?,
                pre_ff_ln,
                post_ff_ln,
                attn: LlamaAttention {
                    q,
                    k,
                    v,
                    o: proj_b(lp("self_attn.o_proj.weight"))?,
                    q_norm,
                    k_norm,
                    num_heads,
                    num_kv_heads,
                    head_dim,
                    scale,
                    groups,
                    eps,
                    softcap: cfg.attn_logit_softcap,
                    rope_interleaved: cfg.architecture.rope_interleaved(),
                },
                ffn,
                eps,
            });
        }

        let rope = cfg.build_rope();
        Ok(Self {
            embed_tokens,
            layers,
            norm,
            lm_head,
            rope,
            dtype,
            device,
            quantized: quant.is_some(),
            embed_scale: gemma.then(|| (cfg.hidden_size as f64).sqrt()),
            final_softcap: cfg.final_logit_softcap,
            cfg,
        })
    }

    /// The model config.
    pub fn config(&self) -> &LlamaConfig {
        &self.cfg
    }

    /// Whether the projections were quantized on load.
    pub fn is_quantized(&self) -> bool {
        self.quantized
    }

    /// The device the model lives on.
    pub fn device(&self) -> &Device {
        &self.device
    }

    /// A fresh contiguous KV cache sized for this model.
    pub fn new_cache(&self) -> ContiguousKvCache {
        ContiguousKvCache::new(self.cfg.num_layers)
    }

    /// The engine's compute dtype for this model (bf16 on GPU, f32 on CPU) — the batched decode reads
    /// it to match its additive attention mask to the score dtype.
    pub fn compute_dtype(&self) -> DType {
        self.dtype
    }

    /// Build per-row RoPE `(cos, sin)` tables for a `[rows, cols]` grid of absolute positions
    /// (row-major flat `positions`, length `rows * cols`) — the **per-sequence** position tables the
    /// batched decode (story 7255) feeds [`LlamaModel::decode_logits_masked`]. Each is
    /// `[rows, cols, head_dim]` in the compute dtype.
    pub fn rope_tables(&self, positions: &[i32], rows: i32, cols: i32) -> Result<(Tensor, Tensor)> {
        let (cos, sin) = self.rope.cos_sin_at(positions, self.dtype, &self.device)?; // [1, rows*cols, hd]
        let hd = self.rope.dim();
        Ok((
            cos.reshape((rows as usize, cols as usize, hd))?,
            sin.reshape((rows as usize, cols as usize, hd))?,
        ))
    }

    /// Embed token ids `[batch, seq]` (u32) → `[batch, seq, hidden]`. Gemma scales the embeddings by
    /// √hidden.
    pub fn embed(&self, input_ids: &Tensor) -> Result<Tensor> {
        let e = embed(&self.embed_tokens, input_ids)?;
        match self.embed_scale {
            Some(s) => Ok(e.affine(s, 0.0)?),
            None => Ok(e),
        }
    }

    /// Run a forward step over token ids and return logits for the **last** position only,
    /// `[batch, vocab]`. `offset` is the position of the first input token (number of cached
    /// positions).
    pub fn decode_logits(
        &self,
        input_ids: &Tensor,
        cache: &mut dyn KvCache,
        offset: i32,
    ) -> Result<Tensor> {
        let embeds = self.embed(input_ids)?;
        self.decode_logits_from_embeds(&embeds, cache, offset)
    }

    /// Like [`LlamaModel::decode_logits`] but from pre-computed input embeddings — the hook the VLM
    /// path uses to splice image features before the decoder.
    pub fn decode_logits_from_embeds(
        &self,
        input_embeds: &Tensor,
        cache: &mut dyn KvCache,
        offset: i32,
    ) -> Result<Tensor> {
        let s = input_embeds.dim(1)? as i32;
        // Single-sequence / uniform batch: positions [offset, offset+s) shared across the batch, with
        // an implicit bottom-right causal mask; cos/sin `[1, s, head_dim]` broadcast over the batch.
        let (cos, sin) = self.rope.cos_sin(s, offset, self.dtype, &self.device)?;
        self.forward_to_last_logits(input_embeds, cache, &cos, &sin, AttnMask::Causal)
    }

    /// Batched forward over a **left-padded** `[batch, seq]` step with **per-sequence** RoPE positions
    /// and an explicit additive attention mask — the decode primitive the dynamic-batch scheduler
    /// (story 7255) runs each step.
    ///
    /// `input_ids` is `[batch, seq]` (u32); `cos`/`sin` are `[batch, seq, head_dim]` (per-row
    /// positions, e.g. from [`LlamaModel::rope_tables`]); `mask` is an additive
    /// `[batch, 1, seq, k_total]` score mask (`0` keep, large-negative block) covering left-padding +
    /// causality. Returns logits for the **last column** `[batch, vocab]` — left-padding right-aligns
    /// every row's last real token to that column, so one slice serves the whole batch.
    pub fn decode_logits_masked(
        &self,
        input_ids: &Tensor,
        cache: &mut dyn KvCache,
        cos: &Tensor,
        sin: &Tensor,
        mask: &Tensor,
    ) -> Result<Tensor> {
        let embeds = self.embed(input_ids)?;
        self.forward_to_last_logits(&embeds, cache, cos, sin, AttnMask::Additive(mask))
    }

    /// Run the decoder stack over `input_embeds` with the given RoPE tables and attention mask, and
    /// project the **last column** to logits `[batch, vocab]`. The shared core of the single and
    /// batched forwards: they differ only in how `cos`/`sin` and `mask` are built.
    fn forward_to_last_logits(
        &self,
        input_embeds: &Tensor,
        cache: &mut dyn KvCache,
        cos: &Tensor,
        sin: &Tensor,
        mask: AttnMask<'_>,
    ) -> Result<Tensor> {
        let (b, s, _) = input_embeds.dims3()?;

        let mut h = input_embeds.clone();
        for (i, layer) in self.layers.iter().enumerate() {
            h = layer.forward(&h, cos, sin, mask, cache, i)?;
        }

        let last_h = h.narrow(1, s - 1, 1)?.contiguous()?; // [b, 1, hidden]
        let normed = rms_norm(&last_h, &self.norm, self.cfg.rms_norm_eps as f64)?;
        let logits = self.lm_head.forward(&normed)?; // [b, 1, vocab]
        let logits = logits.reshape((b, self.cfg.vocab_size as usize))?;
        // Gemma-2 soft-caps the final logits.
        match self.final_softcap {
            Some(c) => Ok(soft_cap(&logits, c)?),
            None => Ok(logits),
        }
    }
}

impl crate::decode::Decode for LlamaModel {
    fn make_cache(&self) -> Box<dyn KvCache> {
        Box::new(self.new_cache())
    }

    fn device(&self) -> &Device {
        &self.device
    }

    fn step(&self, input_ids: &Tensor, cache: &mut dyn KvCache, offset: i32) -> Result<Tensor> {
        self.decode_logits(input_ids, cache, offset)
    }
}

/// One transformer block. Pre-norm by default (Llama / Qwen / Phi); Gemma-2 adds the post-attention
/// and post-feedforward norms ([`LlamaLayer::pre_ff_ln`] / [`LlamaLayer::post_ff_ln`] are `Some`) for
/// its 4-norm "sandwich" residual.
struct LlamaLayer {
    /// Pre-attention norm.
    input_ln: Tensor,
    /// Llama: the MLP pre-norm. Gemma-2: the post-attention norm.
    post_ln: Tensor,
    /// Gemma-2 only: the MLP pre-norm (the post-attention residual is normed by `input`/`post_ln`).
    pre_ff_ln: Option<Tensor>,
    /// Gemma-2 only: the post-feedforward norm applied to the MLP output before the residual add.
    post_ff_ln: Option<Tensor>,
    attn: LlamaAttention,
    ffn: Ffn,
    eps: f64,
}

impl LlamaLayer {
    fn forward(
        &self,
        x: &Tensor,
        cos: &Tensor,
        sin: &Tensor,
        mask: AttnMask<'_>,
        cache: &mut dyn KvCache,
        layer_idx: usize,
    ) -> Result<Tensor> {
        let attn = self.attn.forward(
            &rms_norm(x, &self.input_ln, self.eps)?,
            cos,
            sin,
            mask,
            cache,
            layer_idx,
        )?;
        match (&self.pre_ff_ln, &self.post_ff_ln) {
            // Gemma-2 sandwich: post-norm the attention output and the MLP output before each add.
            (Some(pre_ff), Some(post_ff)) => {
                let attn = rms_norm(&attn, &self.post_ln, self.eps)?;
                let h = x.broadcast_add(&attn)?;
                let ffn = self.ffn.forward(&rms_norm(&h, pre_ff, self.eps)?)?;
                let ffn = rms_norm(&ffn, post_ff, self.eps)?;
                Ok(h.broadcast_add(&ffn)?)
            }
            // Llama pre-norm: `post_ln` is the MLP pre-norm.
            _ => {
                let h = x.broadcast_add(&attn)?;
                let ffn = self.ffn.forward(&rms_norm(&h, &self.post_ln, self.eps)?)?;
                Ok(h.broadcast_add(&ffn)?)
            }
        }
    }
}

/// A layer's feed-forward network: a dense SwiGLU MLP, or a sparse Mixture-of-Experts bank.
enum Ffn {
    Dense(LlamaMlp),
    Moe(MoeMlp),
}

impl Ffn {
    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        match self {
            Ffn::Dense(m) => m.forward(x),
            Ffn::Moe(m) => m.forward(x),
        }
    }
}

/// Grouped-query attention with RoPE and optional per-head q/k RMSNorm (Qwen3).
struct LlamaAttention {
    q: Projection,
    k: Projection,
    v: Projection,
    o: Projection,
    q_norm: Option<Tensor>,
    k_norm: Option<Tensor>,
    num_heads: usize,
    num_kv_heads: usize,
    head_dim: usize,
    scale: f32,
    groups: usize,
    eps: f64,
    /// Gemma-2 attention-score soft-cap; `None` ⇒ no cap.
    softcap: Option<f32>,
    /// Whether RoPE uses the interleaved (GPT-J) pairing (GLM-4).
    rope_interleaved: bool,
}

impl LlamaAttention {
    fn forward(
        &self,
        x: &Tensor,
        cos: &Tensor,
        sin: &Tensor,
        mask: AttnMask<'_>,
        cache: &mut dyn KvCache,
        layer_idx: usize,
    ) -> Result<Tensor> {
        let (b, s, _) = x.dims3()?;
        let (nh, nkv, hd) = (self.num_heads, self.num_kv_heads, self.head_dim);

        // Project, then split into heads in [b, s, heads, head_dim] layout.
        let mut q = self.q.forward(x)?.reshape((b, s, nh, hd))?;
        let mut k = self.k.forward(x)?.reshape((b, s, nkv, hd))?;
        let v = self.v.forward(x)?.reshape((b, s, nkv, hd))?;

        // Qwen3 per-head q/k RMSNorm over the head_dim axis, before RoPE.
        if let Some(qn) = &self.q_norm {
            q = rms_norm(&q, qn, self.eps)?;
        }
        if let Some(kn) = &self.k_norm {
            k = rms_norm(&k, kn, self.eps)?;
        }

        // RoPE on q,k (cos/sin broadcast over the head axis), then -> [b, heads, s, head_dim].
        let q = apply_rope(&q, cos, sin, self.rope_interleaved)?
            .transpose(1, 2)?
            .contiguous()?;
        let k = apply_rope(&k, cos, sin, self.rope_interleaved)?
            .transpose(1, 2)?
            .contiguous()?;
        let v = v.transpose(1, 2)?.contiguous()?;

        let (k_all, v_all) = cache.update(layer_idx, &k, &v)?;
        let k_all = repeat_kv(&k_all, self.groups)?;
        let v_all = repeat_kv(&v_all, self.groups)?;

        let out = sdpa(&q, &k_all, &v_all, self.scale, self.softcap, mask)?; // [b, heads, s, head_dim]
        let out = out
            .transpose(1, 2)?
            .contiguous()?
            .reshape((b, s, nh * hd))?;
        self.o.forward(&out)
    }
}

/// A gated MLP: SwiGLU (`silu`) by default, or GeGLU (`gelu`, the Gemma activation) when `gelu`.
struct LlamaMlp {
    gate: Projection,
    up: Projection,
    down: Projection,
    gelu: bool,
}

impl LlamaMlp {
    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let g = self.gate.forward(x)?;
        let g = if self.gelu { gelu(&g)? } else { silu(&g)? };
        let up = self.up.forward(x)?;
        self.down.forward(&(g * up)?)
    }
}

/// A Qwen2-MoE feed-forward: a softmax router over `experts` (top-k per token), plus an always-on
/// `shared` expert gated by a sigmoid. Correctness-first — each expert runs **only on its routed
/// tokens** (gathered, then scatter-added back), so the active compute scales with `experts_per_tok`,
/// not the full bank. Top-k selection is done on the host (Candle has no fused top-k).
struct MoeMlp {
    /// Router weight `[num_experts, hidden]`.
    router: Tensor,
    experts: Vec<LlamaMlp>,
    shared: LlamaMlp,
    /// Shared-expert sigmoid gate `[1, hidden]`.
    shared_gate: Tensor,
    experts_per_tok: usize,
    norm_topk_prob: bool,
}

impl MoeMlp {
    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let (b, s, h) = x.dims3()?;
        let t = b * s;
        let dtype = x.dtype();
        let device = x.device();
        let xf = x.reshape((t, h))?;

        // Router probabilities (computed in f32 for a stable top-k), pulled to host.
        let logits = xf.matmul(&self.router.t()?)?; // [t, E]
        let probs = candle_nn::ops::softmax_last_dim(&logits.to_dtype(DType::F32)?)?;
        let probs = probs.to_vec2::<f32>()?; // [t][E]
        let num_experts = self.experts.len();
        let k = self.experts_per_tok.min(num_experts).max(1);

        // Invert the per-token top-k into per-expert (token, weight) lists.
        let mut routed: Vec<Vec<(u32, f32)>> = vec![Vec::new(); num_experts];
        for (ti, row) in probs.iter().enumerate() {
            let mut idx: Vec<usize> = (0..num_experts).collect();
            idx.sort_unstable_by(|&a, &b| row[b].total_cmp(&row[a]));
            let top = &idx[..k];
            let denom: f32 = if self.norm_topk_prob {
                top.iter()
                    .map(|&e| row[e])
                    .sum::<f32>()
                    .max(f32::MIN_POSITIVE)
            } else {
                1.0
            };
            for &e in top {
                routed[e].push((ti as u32, row[e] / denom));
            }
        }

        // Each expert runs on just its tokens; scatter the weighted outputs back.
        let mut out = Tensor::zeros((t, h), dtype, device)?;
        for (e, toks) in routed.iter().enumerate() {
            if toks.is_empty() {
                continue;
            }
            let n = toks.len();
            let idx = Tensor::from_vec(
                toks.iter().map(|&(ti, _)| ti).collect::<Vec<u32>>(),
                (n,),
                device,
            )?;
            let wts = Tensor::from_vec(
                toks.iter().map(|&(_, w)| w).collect::<Vec<f32>>(),
                (n, 1),
                device,
            )?
            .to_dtype(dtype)?;
            let xe = xf.index_select(&idx, 0)?; // [n, h]
            let ye = self.experts[e].forward(&xe)?.broadcast_mul(&wts)?; // [n, h]
            out = out.index_add(&idx, &ye, 0)?;
        }

        // Always-on shared expert, gated by sigmoid(x · shared_gateᵀ).
        let sg = candle_nn::ops::sigmoid(&xf.matmul(&self.shared_gate.t()?)?)?; // [t, 1]
        let shared = self.shared.forward(&xf)?.broadcast_mul(&sg)?;
        Ok((out + shared)?.reshape((b, s, h))?)
    }
}

/// Join a key prefix and suffix (`""` prefix ⇒ the suffix verbatim).
fn join(prefix: &str, suffix: &str) -> String {
    if prefix.is_empty() {
        suffix.to_string()
    } else {
        format!("{prefix}.{suffix}")
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn join_handles_empty_prefix() {
        assert_eq!(join("", "model.norm.weight"), "model.norm.weight");
        assert_eq!(
            join("language_model", "model.norm.weight"),
            "language_model.model.norm.weight"
        );
    }
}
