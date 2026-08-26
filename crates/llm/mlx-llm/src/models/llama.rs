//! Generic Llama-family causal decoder, config-dispatched across architectures.
//!
//! One block shape covers the family (Llama / Mistral / Qwen2 / Qwen3 / Phi-3 / Qwen2-MoE / Gemma-2 /
//! GLM-4 / DeepSeek-V2 / Gemma 4): self-attention is either grouped-query attention (with optional
//! per-head q/k RMSNorm for Qwen3 and Gemma 4, q/k/v bias for Qwen2 / GLM-4, a packed `qkv_proj` for
//! Phi-3, Gemma-2 score soft-cap) or Multi-head Latent Attention (DeepSeek's low-rank KV path); the
//! FFN is a dense gated MLP (SwiGLU, or GeGLU for Gemma) or a sparse Mixture-of-Experts bank; norms
//! are the Llama pre-norm or the Gemma-2 / GLM-4 / Gemma 4 4-norm "sandwich". Projections are held
//! behind [`Projection`] so a model can be quantized on load. The forward is `&self`; the KV cache is
//! the only mutable state, threaded in as `&mut dyn KvCache`. Ported alongside candle-llm's
//! `models/llama.rs` (the cross-backend blueprint).
//!
//! Shapes are batch-capable (`[batch, seq, …]`). `head_dim` is taken from config and may differ from
//! `hidden_size / num_heads`. Cached decode runs in bf16.
//!
//! # Gemma 4 (sc-18760)
//!
//! Every architecture above this one is **uniform**: one `head_dim`, one KV-head count, one RoPE
//! schedule, one mask, for all layers. Gemma 4 is not. Its `layer_types` table alternates two
//! genuinely different attention shapes (`ModelConfig::layer_attention`), and the decoder resolves
//! per layer rather than per model:
//!
//! | | `sliding_attention` (40 layers) | `full_attention` (8 layers) |
//! |---|---|---|
//! | `head_dim` | 256 | `global_head_dim` 512 |
//! | KV heads | `num_key_value_heads` 8 | `num_global_key_value_heads` 1 |
//! | RoPE | `default`, θ = 10 000 | `proportional`, θ = 1 000 000, `partial_rotary_factor` 0.25 |
//! | mask | causal ∩ `sliding_window` 1024 | plain causal |
//! | K/V projections | separate `k_proj` + `v_proj` | **one** shared `k_proj` (`attention_k_eq_v`) |
//!
//! Because the two head dims differ, one `(cos, sin)` pair cannot serve both — the stack builds one
//! pair per layer type (`RopeTables`), and each layer reads its own. The rest of the Gemma 4
//! block is the Gemma-2 shape it inherits (√hidden embedding scale, GeGLU, 4-norm sandwich, tied
//! embeddings, `final_logit_softcapping`) plus three things no earlier architecture has:
//!
//! * **`attention_k_eq_v`** — the value heads are the **raw** shared key-projection output, taken
//!   *before* `k_norm` and *before* RoPE, then passed through a scale-free `v_norm`. K and V remain
//!   different tensors; only the matmul and the stored weight are shared ([`KvProjection`]).
//! * **`v_norm`** — a per-head RMSNorm with no learned weight (`with_scale=False`), so there is no
//!   `v_norm.weight` in the checkpoint ([`rms_norm_unscaled`]).
//! * **`layer_scalar`** — a per-layer `[1]` buffer multiplying the **whole block output**, after
//!   both residual adds. It is a persistent buffer in the reference, so the shipped checkpoint
//!   carries trained values; assuming `1.0` silently mis-scales every layer.
//!
//! Attention is scaled by a literal `1.0` (not `head_dim^-0.5`) — the learned q/k norms absorb it —
//! and the norms multiply by the stored weight directly, *not* Gemma-2's `(1 + weight)` fold.

use mlx_rs::ops::{
    add, broadcast_to, concatenate_axis, multiply, sigmoid, split_sections, zeros_dtype,
};
use mlx_rs::{Array, Dtype};

use crate::config::{Architecture, BidirectionalAttention, LayerAttentionType, ModelConfig};
use crate::error::{Error, Result};
use crate::models::deepstack::deepstack_fused_decoder_layers;
use crate::primitives::attention::{sdpa_capped, sliding_causal_mask, AttnMask};
use crate::primitives::kv_cache::KvCache;
use crate::primitives::nn::{
    embed, gelu_tanh, linear, rms_norm, rms_norm_unscaled, silu, soft_cap, to_f32_host,
};
use crate::primitives::projection::{KvProjection, Projection, QuantSpec};
use crate::primitives::quant::QuantizedLinear;
use crate::primitives::rope::{apply_rope, Rope};
use crate::primitives::{ContiguousKvCache, PagedKvCache, Weights};

/// Cached decode runs in bf16 (matching the reference engines).
const COMPUTE_DTYPE: Dtype = Dtype::Bfloat16;

/// A loaded causal decoder.
#[derive(Debug)]
pub struct CausalLm {
    embed_tokens: Array,
    layers: Vec<LlamaLayer>,
    norm: Array,
    lm_head: Array,
    /// The model-level RoPE for a uniform architecture; Gemma 4's `sliding_attention` schedule.
    rope: Rope,
    /// Gemma 4's `full_attention` schedule — a different head dim *and* a different frequency
    /// table, so it cannot share [`CausalLm::rope`]'s cos/sin. `None` for every uniform
    /// architecture, which is what keeps their stack pass byte-identical.
    full_rope: Option<Rope>,
    cfg: ModelConfig,
    quantized: bool,
    /// Gemma scales token embeddings by √hidden; `None` ⇒ no scaling.
    embed_scale: Option<f32>,
    /// Gemma-2 final-logit soft-cap; `None` ⇒ no cap.
    final_softcap: Option<f32>,
}

/// The RoPE `(cos, sin)` tables one decoder-stack pass needs — one pair per layer type present in
/// the model.
///
/// A uniform architecture fills only `primary` and every layer reads it, so the pass is exactly the
/// single-table pass it always was. Gemma 4 also fills `full`, because its `full_attention` layers
/// rotate a 512-wide head on a proportional schedule while its `sliding_attention` layers rotate a
/// 256-wide head on the default one.
#[derive(Debug)]
struct RopeTables {
    /// The uniform schedule, or Gemma 4's `sliding_attention` schedule.
    primary: (Array, Array),
    /// Gemma 4's `full_attention` schedule.
    full: Option<(Array, Array)>,
}

impl RopeTables {
    /// A single shared table for every layer — the uniform-architecture pass.
    fn uniform(cos: Array, sin: Array) -> Self {
        Self {
            primary: (cos, sin),
            full: None,
        }
    }

    /// The `(cos, sin)` the given slot rotates with.
    ///
    /// A [`RopeSlot::Full`] layer in a model that never built a `full` table is a construction bug
    /// (the loader assigns the slot exactly when it also builds the table), so this falls back to
    /// `primary` rather than failing.
    fn get(&self, slot: RopeSlot) -> (&Array, &Array) {
        match (slot, &self.full) {
            (RopeSlot::Full, Some((cos, sin))) => (cos, sin),
            _ => (&self.primary.0, &self.primary.1),
        }
    }
}

/// Which of a model's [`RopeTables`] a decoder layer rotates with.
///
/// Every layer of a uniform architecture is [`RopeSlot::Primary`]. Gemma 4 assigns `Full` to its
/// `full_attention` layers and `Primary` to its `sliding_attention` ones.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum RopeSlot {
    Primary,
    Full,
}

/// One K/V slot per Gemma 4 layer type — the `num_kv_shared_layers` scratch.
///
/// The trailing `num_kv_shared_layers` layers project no keys or values of their own; each reuses
/// the K/V of the **last** earlier layer *of its own type*, which is why a single slot is not
/// enough. Upstream keeps this in a `shared_kv_states` dict threaded through the stack, and reads
/// it rather than the KV cache even when a cache is present: a sliding layer's cache is not
/// guaranteed to still hold the full-length keys.
///
/// Left entirely empty — and never read — by every model without KV sharing, which is all of them
/// except a Gemma 4 config that sets `num_kv_shared_layers > 0`.
#[derive(Debug, Default)]
struct SharedKv {
    sliding: Option<(Array, Array)>,
    full: Option<(Array, Array)>,
}

impl SharedKv {
    fn get(&self, kind: LayerAttentionType) -> Option<&(Array, Array)> {
        match kind {
            LayerAttentionType::Sliding => self.sliding.as_ref(),
            LayerAttentionType::Full => self.full.as_ref(),
        }
    }

    fn set(&mut self, kind: LayerAttentionType, kv: (Array, Array)) {
        match kind {
            LayerAttentionType::Sliding => self.sliding = Some(kv),
            LayerAttentionType::Full => self.full = Some(kv),
        }
    }
}

impl CausalLm {
    /// Build from a loaded checkpoint (dense). `prefix` is the weight-key prefix (`""` for a plain
    /// `*ForCausalLM`, e.g. `"language_model"` for a VLM-nested decoder).
    pub fn from_weights(w: &Weights, prefix: &str, cfg: ModelConfig) -> Result<Self> {
        Self::from_weights_with(w, prefix, cfg, None)
    }

    /// Build from a loaded checkpoint, optionally quantizing the attention/MLP projections on load.
    /// Embeddings, the LM head, and norms always stay dense.
    pub fn from_weights_with(
        w: &Weights,
        prefix: &str,
        cfg: ModelConfig,
        quant: Option<QuantSpec>,
    ) -> Result<Self> {
        // Gemma 4's `use_bidirectional_attention: "all"` makes **every** token attend both ways, so
        // its sliding layers want a *symmetric* window rather than the causal one every mask in this
        // crate builds (and the config parse has already halved `sliding_window` to the exclusive
        // bound that mode uses). Refuse it rather than quietly running it causally: this decoder
        // serves a text encoder and a prompt enhancer, both of which ship `"vision"` (vision tokens
        // bidirectional, text causal — and a text-only stack has no vision tokens, so plain causal).
        if let Some(g) = &cfg.gemma4 {
            if g.bidirectional == Some(BidirectionalAttention::All) {
                return Err(Error::Unsupported(
                    "Gemma 4 `use_bidirectional_attention: \"all\"` needs a symmetric (non-causal) \
                     sliding window; this decoder builds causal masks only. Every shipped Gemma 4 \
                     text config carries `\"vision\"`, which is plain causal for a text-only stack."
                        .to_string(),
                ));
            }
        }

        // The Qwen3-VL VLM wrapper nests the decoder under `model.language_model.*` (embeddings,
        // norm, and `layers.{i}.*`) — there is no second `model.` segment — while `lm_head.weight`
        // lives at the checkpoint root (untied). A plain `*ForCausalLM` keeps the historical
        // `[{prefix}.]model.*` / `[{prefix}.]lm_head.weight` layout.
        let vlm_nested = cfg.architecture.is_qwen3_vl();
        let decoder_root = if vlm_nested {
            "model.language_model".to_string()
        } else {
            join(prefix, "model")
        };
        let p = |suffix: &str| join(&decoder_root, suffix);
        let head_key = if vlm_nested {
            "lm_head.weight".to_string()
        } else {
            join(prefix, "lm_head.weight")
        };
        let req_bf16 =
            |key: String| -> Result<Array> { Ok(w.require(&key)?.as_dtype(COMPUTE_DTYPE)?) };

        // A snapshot may store pre-quantized projections (the GGUF converter's MLX-requant output);
        // those are loaded from `weight`/`scales`/`biases` as-is. Otherwise the dense weight is loaded
        // (and quantized on the fly if a load-time `quant` was requested). `bias` is applied dense in
        // both cases (Qwen2 / GLM-4 attention bias).
        let stored_quant = cfg.quantization;
        let load_proj = |key: &str, bias: Option<Array>| -> Result<Projection> {
            let base = key.strip_suffix(".weight").unwrap_or(key);
            let scales_key = format!("{base}.scales");
            if w.contains(&scales_key) {
                let spec = stored_quant.ok_or_else(|| {
                    Error::Config(format!(
                        "snapshot stores quantized tensor `{scales_key}` but config.json has no \
                         `quantization` block"
                    ))
                })?;
                Ok(Projection::Quantized(QuantizedLinear {
                    weight: w.require(key)?.clone(),
                    scales: w.require(&scales_key)?.clone(),
                    biases: w.require(&format!("{base}.biases"))?.clone(),
                    group_size: spec.group_size,
                    bits: spec.bits,
                    bias,
                }))
            } else {
                Projection::load_with_bias(w.require(key)?.as_dtype(COMPUTE_DTYPE)?, bias, quant)
            }
        };
        let proj = |key: String| -> Result<Projection> { load_proj(&key, None) };
        // Like `proj`, but also loads a sibling `.bias` when present (Qwen2 / GLM-4 attention).
        let proj_b = |wkey: String| -> Result<Projection> {
            let base = wkey.strip_suffix(".weight").unwrap_or(&wkey);
            let bkey = format!("{base}.bias");
            let bias = if w.contains(&bkey) {
                Some(req_bf16(bkey)?)
            } else {
                None
            };
            load_proj(&wkey, bias)
        };
        // **Gemma-2's** norms are `(1 + weight)`; fold the +1 into the stored weight so the standard
        // `rms_norm` applies it. (Llama / Qwen3 / Qwen3-VL / GLM-4 norm weights are standard RMSNorm
        // — used verbatim, including Qwen3-VL's `Qwen3VLTextRMSNorm`, which is plain `weight · x`;
        // its small early-layer block-norm weights are genuine, verified by real-weights coherence.
        // **Gemma 4** is also verbatim: `Gemma4UnifiedRMSNorm` multiplies by the stored weight, whose
        // initializer is ones, so folding +1 in would corrupt every norm — hence `norm_unit_offset`
        // rather than the broader `is_gemma`.)
        let norm_offset = cfg.architecture.norm_unit_offset();
        // The two things every Gemma generation shares: the √hidden embedding scale and the GeGLU
        // (`gelu_pytorch_tanh`) MLP.
        let gemma = cfg.architecture.is_gemma();
        let norm_w = |key: String| -> Result<Array> {
            let t = req_bf16(key)?;
            if norm_offset {
                Ok(add(&t, &Array::from_f32(1.0).as_dtype(t.dtype())?)?)
            } else {
                Ok(t)
            }
        };

        let embed_tokens = req_bf16(p("embed_tokens.weight"))?;
        let norm = norm_w(p("norm.weight"))?;
        let lm_head = if cfg.tie_word_embeddings {
            embed_tokens.clone()
        } else {
            req_bf16(head_key)?
        };

        let qk_norm = cfg.has_qk_norm();
        let num_heads = cfg.num_heads;
        let scale = cfg.attn_scale();
        let eps = cfg.rms_norm_eps;
        let inter = cfg.intermediate_size;
        // Gemma 4's scale-free per-head value norm; every earlier architecture leaves V un-normed.
        let v_norm = cfg.architecture.has_v_norm();

        // `num_kv_shared_layers`: the trailing layers project no K/V of their own and instead reuse
        // the K/V of the **last** earlier layer of their own type. Upstream's
        // `first_kv_shared_layer_idx = num_hidden_layers - num_kv_shared_layers`, with the `> 0`
        // guard that keeps a `0` setting from making every layer "shared".
        let kv_shared_from = cfg
            .gemma4
            .as_ref()
            .filter(|g| g.num_kv_shared_layers > 0)
            .map(|g| cfg.num_layers.saturating_sub(g.num_kv_shared_layers));
        // `store_full_length_kv`: the **last** layer of each type before the sharing tail is the one
        // whose K/V the tail reuses. Walking forward and overwriting leaves exactly that layer.
        let mut kv_store_at: [Option<usize>; 2] = [None, None];
        if let (Some(first), Some(g)) = (kv_shared_from, cfg.gemma4.as_ref()) {
            for i in 0..first {
                kv_store_at[kind_slot(g.layer_type(i))] = Some(i);
            }
        }

        let mut layers = Vec::with_capacity(cfg.num_layers);
        for i in 0..cfg.num_layers {
            let lp = |suffix: &str| join(&decoder_root, &format!("layers.{i}.{suffix}"));
            // The per-layer attention shape. Uniform architectures resolve every layer to the same
            // descriptor (the model's scalar `head_dim` / `num_kv_heads`, no window, no `k_eq_v`),
            // so this is the pre-Gemma-4 behaviour verbatim; Gemma 4 resolves two.
            let la = cfg.layer_attention(i);
            // A uniform architecture has no `layer_types` table; the crate's convention is that its
            // whole stack is `full_attention` (one shape, no window).
            let kind = cfg
                .gemma4
                .as_ref()
                .map_or(LayerAttentionType::Full, |g| g.layer_type(i));
            // Uniform architectures never build a `full` table, so every layer reads `Primary`.
            let rope_slot = match (&cfg.gemma4, kind) {
                (Some(_), LayerAttentionType::Full) => RopeSlot::Full,
                _ => RopeSlot::Primary,
            };
            let head_dim = la.head_dim;
            let num_kv_heads = la.num_kv_heads;
            let qd = num_heads * head_dim;
            let kvd = num_kv_heads * head_dim;
            let kv_shared = kv_shared_from.is_some_and(|first| i >= first);
            let stores_kv = kv_store_at[kind_slot(kind)] == Some(i);

            // Attention: Multi-head Latent Attention (DeepSeek-V2) or grouped-query attention.
            let attn = if cfg.architecture.is_mla() {
                Attention::Mla(MlaAttention::load(w, &lp, &cfg, &load_proj, &req_bf16)?)
            } else {
                // A KV-sharing tail layer (Gemma 4's `num_kv_shared_layers`) projects no keys or
                // values of its own — upstream builds no `k_proj` / `v_proj` / `k_norm` / `v_norm`
                // for it at all — and reads the stored K/V of the last earlier layer of its type.
                let q_norm = qk_norm
                    .then(|| req_bf16(lp("self_attn.q_norm.weight")))
                    .transpose()?;
                let k_norm = (qk_norm && !kv_shared)
                    .then(|| req_bf16(lp("self_attn.k_norm.weight")))
                    .transpose()?;
                // A packed `qkv_proj` (Phi-3, no bias) is split into q/k/v along axis 0; otherwise the
                // separate q/k/v projections are loaded (with q/k/v bias for Qwen2 / GLM-4).
                let packed = lp("self_attn.qkv_proj.weight");
                let (q, kv) = if w.contains(&packed) {
                    let qkv = req_bf16(packed)?; // [qd + 2*kvd, hidden]
                    let parts = split_sections(&qkv, &[qd, qd + kvd], 0)?;
                    (
                        Projection::load(parts[0].clone(), quant)?,
                        Some(KvProjection::separate(
                            Projection::load(parts[1].clone(), quant)?,
                            Projection::load(parts[2].clone(), quant)?,
                        )),
                    )
                } else {
                    let q = proj_b(lp("self_attn.q_proj.weight"))?;
                    let kv = if kv_shared {
                        None
                    } else {
                        let k = proj_b(lp("self_attn.k_proj.weight"))?;
                        // `attention_k_eq_v`: there is **no** `v_proj` weight in the checkpoint —
                        // the value heads come from this same key projection's raw output.
                        Some(match la.k_eq_v {
                            true => KvProjection::shared(k),
                            false => {
                                KvProjection::separate(k, proj_b(lp("self_attn.v_proj.weight"))?)
                            }
                        })
                    };
                    (q, kv)
                };
                Attention::Gqa(LlamaAttention {
                    q,
                    kv,
                    o: proj_b(lp("self_attn.o_proj.weight"))?,
                    q_norm,
                    k_norm,
                    v_norm,
                    num_heads,
                    num_kv_heads,
                    head_dim,
                    scale,
                    eps,
                    softcap: cfg.attn_logit_softcap,
                    rope_interleaved: cfg.architecture.rope_interleaved(),
                    sliding_window: la.sliding_window,
                    kind,
                    stores_kv,
                })
            };

            // Feed-forward: a sparse Mixture-of-Experts bank or a dense MLP. DeepSeek keeps its leading
            // `first_k_dense_replace` layers dense even though the model is MoE. Gemma uses GeGLU.
            let moe_layer = cfg.moe.filter(|m| i >= m.first_k_dense_replace);
            let ffn = if let Some(moe) = moe_layer {
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
                // Shared-expert key stem: DeepSeek packs `n_shared_experts` into `mlp.shared_experts`
                // (plural, ungated); Qwen2-MoE has a single `mlp.shared_expert` gated by a sigmoid.
                let shared_stem = if w.contains(&lp("mlp.shared_experts.gate_proj.weight")) {
                    "mlp.shared_experts"
                } else {
                    "mlp.shared_expert"
                };
                let shared_gate_key = lp("mlp.shared_expert_gate.weight");
                Ffn::Moe(MoeMlp {
                    router: req_bf16(lp("mlp.gate.weight"))?, // [num_experts, hidden]
                    experts,
                    shared: LlamaMlp {
                        gate: proj(lp(&format!("{shared_stem}.gate_proj.weight")))?,
                        up: proj(lp(&format!("{shared_stem}.up_proj.weight")))?,
                        down: proj(lp(&format!("{shared_stem}.down_proj.weight")))?,
                        gelu: false,
                    },
                    shared_gate: if w.contains(&shared_gate_key) {
                        Some(req_bf16(shared_gate_key)?) // [1, hidden]
                    } else {
                        None
                    },
                    experts_per_tok: moe.num_experts_per_tok,
                    norm_topk_prob: moe.norm_topk_prob,
                    routed_scaling_factor: moe.routed_scaling_factor,
                })
            } else {
                // Dense MLP; Phi-3 fuses gate‖up into one weight, split along axis 0.
                let (gate, up) = {
                    let packed = lp("mlp.gate_up_proj.weight");
                    if w.contains(&packed) {
                        let gu = req_bf16(packed)?; // [2*inter, hidden]
                        let parts = split_sections(&gu, &[inter], 0)?;
                        (
                            Projection::load(parts[0].clone(), quant)?,
                            Projection::load(parts[1].clone(), quant)?,
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

            // Gemma 4's `layer_scalar`: a `[1]` buffer multiplying the block output after both
            // residual adds. It is a *persistent* buffer upstream — it ships in the checkpoint with
            // trained values, so it is read rather than assumed to be its `ones` initializer. Absent
            // ⇒ `None` (every architecture before Gemma 4, and any Gemma 4 export that omitted it,
            // for which the initializer is an exact identity).
            let scalar_key = lp("layer_scalar");
            let layer_scalar = w
                .contains(&scalar_key)
                .then(|| req_bf16(scalar_key))
                .transpose()?;

            layers.push(LlamaLayer {
                input_ln: norm_w(lp("input_layernorm.weight"))?,
                post_ln: norm_w(lp(&format!("{post_attn_key}.weight")))?,
                pre_ff_ln,
                post_ff_ln,
                attn,
                ffn,
                eps,
                layer_scalar,
                rope_slot,
            });
        }

        // Gemma 4 resolves its RoPE per layer type; every uniform architecture builds one table and
        // leaves `full_rope` `None`, which keeps its stack pass exactly what it was.
        let (rope, full_rope) = match &cfg.gemma4 {
            Some(g) => (
                g.for_type(LayerAttentionType::Sliding).build_rope(),
                Some(g.for_type(LayerAttentionType::Full).build_rope()),
            ),
            None => (cfg.build_rope(), None),
        };
        let quantized = quant.is_some() || cfg.quantization.is_some();
        Ok(Self {
            embed_tokens,
            layers,
            norm,
            lm_head,
            rope,
            full_rope,
            quantized,
            embed_scale: gemma.then(|| (cfg.hidden_size as f32).sqrt()),
            final_softcap: cfg.final_logit_softcap,
            cfg,
        })
    }

    /// The model config.
    pub fn config(&self) -> &ModelConfig {
        &self.cfg
    }

    /// Whether the projections were quantized on load.
    pub fn is_quantized(&self) -> bool {
        self.quantized
    }

    /// A fresh contiguous KV cache sized for this model.
    pub fn new_cache(&self) -> ContiguousKvCache {
        ContiguousKvCache::new(self.cfg.num_layers)
    }

    /// A fresh single-sequence **paged** KV cache (story 7169) sized for this model, with
    /// `block_size`-token blocks.
    pub fn new_paged_cache(&self, block_size: usize) -> PagedKvCache {
        PagedKvCache::new(self.cfg.num_layers, block_size)
    }

    /// The engine's cached-decode compute dtype (bf16).
    pub const fn compute_dtype(&self) -> Dtype {
        COMPUTE_DTYPE
    }

    /// Build per-row RoPE `(cos, sin)` tables for a `[rows, cols]` grid of absolute positions
    /// (row-major flat `positions`, length `rows * cols`). Each is `[rows, cols, rope_dim]` in bf16.
    ///
    /// This is the **primary** schedule only. A Gemma 4 model also has a `full_attention` schedule
    /// at a different head dim, which no single pair can carry — see
    /// [`CausalLm::decode_logits_masked_at`], which builds both from the same positions.
    pub fn rope_tables(&self, positions: &[i32], rows: i32, cols: i32) -> Result<(Array, Array)> {
        Self::grid(&self.rope, positions, rows, cols)
    }

    /// One `rope`'s `(cos, sin)` reshaped to a `[rows, cols, rope_dim]` grid.
    fn grid(rope: &Rope, positions: &[i32], rows: i32, cols: i32) -> Result<(Array, Array)> {
        let (cos, sin) = rope.cos_sin_at(positions, COMPUTE_DTYPE)?; // [1, rows*cols, rope_dim]
        let hd = rope.dim();
        Ok((
            cos.reshape(&[rows, cols, hd])?,
            sin.reshape(&[rows, cols, hd])?,
        ))
    }

    /// The RoPE tables a stack pass over `s` contiguous positions starting at `offset` needs — one
    /// pair per layer type the model has.
    fn stack_rope(&self, s: i32, offset: i32) -> Result<RopeTables> {
        Ok(RopeTables {
            primary: self.rope.cos_sin(s, offset, COMPUTE_DTYPE)?,
            full: match &self.full_rope {
                Some(r) => Some(r.cos_sin(s, offset, COMPUTE_DTYPE)?),
                None => None,
            },
        })
    }

    /// The RoPE tables for an explicit `[rows, cols]` grid of absolute positions — the batched
    /// (per-sequence positions) form of [`CausalLm::stack_rope`].
    fn stack_rope_at(&self, positions: &[i32], rows: i32, cols: i32) -> Result<RopeTables> {
        Ok(RopeTables {
            primary: Self::grid(&self.rope, positions, rows, cols)?,
            full: match &self.full_rope {
                Some(r) => Some(Self::grid(r, positions, rows, cols)?),
                None => None,
            },
        })
    }

    /// Whether this model resolves attention **per layer type** rather than uniformly (Gemma 4).
    /// The entry points that take caller-built `(cos, sin)` cannot serve such a model — one pair
    /// cannot span two head dims — and refuse it rather than rotating half the stack wrongly.
    fn needs_per_type_rope(&self) -> bool {
        self.full_rope.is_some()
    }

    /// Embed token ids `[batch, seq]` → `[batch, seq, hidden]` (bf16). Gemma scales by √hidden.
    pub fn embed(&self, input_ids: &Array) -> Result<Array> {
        let e = embed(&self.embed_tokens, input_ids)?;
        match self.embed_scale {
            Some(s) => Ok(multiply(&e, &Array::from_f32(s).as_dtype(e.dtype())?)?),
            None => Ok(e),
        }
    }

    /// Run a forward step over token ids and return logits for the **last** position only,
    /// `[batch, vocab]`. `offset` is the position of the first input token (number of cached
    /// positions).
    pub fn decode_logits(
        &self,
        input_ids: &Array,
        cache: &mut dyn KvCache,
        offset: i32,
    ) -> Result<Array> {
        let embeds = self.embed(input_ids)?;
        self.decode_logits_from_embeds(&embeds, cache, offset)
    }

    /// Like [`CausalLm::decode_logits`] but from pre-computed input embeddings — the VLM splice hook.
    pub fn decode_logits_from_embeds(
        &self,
        input_embeds: &Array,
        cache: &mut dyn KvCache,
        offset: i32,
    ) -> Result<Array> {
        let s = input_embeds.shape()[1];
        let ropes = self.stack_rope(s, offset)?;
        self.forward_to_last_logits(input_embeds, cache, &ropes, AttnMask::Causal)
    }

    /// Embed token ids `[1, S]` → `[1, S, hidden]` in the compute dtype — the Qwen3-VL multimodal
    /// splice point (image-token rows are overwritten with the vision tower's merged patch features).
    pub fn embed_input_ids(&self, input_ids: &Array) -> Result<Array> {
        Ok(self.embed(input_ids)?.as_dtype(COMPUTE_DTYPE)?)
    }

    /// Replace the `image_token_id` rows of `embeds` `[1, S, hidden]` with the vision tower's merged
    /// patch features `[num_image_tokens, hidden]`, in sequence order (the Qwen3-VL splice).
    pub fn splice_image_features(
        &self,
        embeds: &Array,
        input_ids: &[i32],
        image_features: &Array,
        image_token_id: i32,
    ) -> Result<Array> {
        crate::models::deepstack::splice_image_features(
            embeds,
            input_ids,
            image_features,
            image_token_id,
            self.cfg.hidden_size,
            COMPUTE_DTYPE,
        )
    }

    /// Replace every row whose id is any of `placeholder_tokens` (`<|image_pad|>` and/or
    /// `<|video_pad|>`) with the next vision-feature row, in sequence order — the multimodal splice for
    /// a mixed image+video prompt. Reduces to [`Self::splice_image_features`] for a single token.
    pub fn splice_vision_features(
        &self,
        embeds: &Array,
        input_ids: &[i32],
        vision_features: &Array,
        placeholder_tokens: &[i32],
    ) -> Result<Array> {
        crate::models::deepstack::splice_vision_features(
            embeds,
            input_ids,
            vision_features,
            placeholder_tokens,
            self.cfg.hidden_size,
            COMPUTE_DTYPE,
        )
    }

    /// Compute the interleaved M-RoPE 3-D position rows (`get_rope_index`, B=1) for `input_ids`
    /// containing `image_grid_thw`-described `image_token_id` runs, plus the `mrope_delta`. The
    /// image-only entry point; see the private `deepstack::mrope_positions_mm` helper for
    /// image+video.
    pub fn mrope_positions(
        &self,
        input_ids: &[i32],
        image_grid_thw: &[[i32; 3]],
        image_token_id: i32,
        spatial_merge_size: i32,
    ) -> Result<crate::models::deepstack::MropePositions> {
        crate::models::deepstack::mrope_positions_mm(
            input_ids,
            image_grid_thw,
            image_token_id,
            &[],
            image_token_id,
            spatial_merge_size,
        )
    }

    /// The full image **and** video interleaved-M-RoPE entry: `input_ids` with `image_token_id` runs
    /// (one per `image_grid_thw` entry) and `video_token_id` runs (one per frame; each `[t, h, w]`
    /// video grid is split into `t` per-frame `[1, h, w]` blocks by the synthetic time axis). See
    /// the private `deepstack::mrope_positions_mm` helper.
    #[allow(clippy::too_many_arguments)]
    pub fn mrope_positions_mm(
        &self,
        input_ids: &[i32],
        image_grid_thw: &[[i32; 3]],
        image_token_id: i32,
        video_grid_thw: &[[i32; 3]],
        video_token_id: i32,
        spatial_merge_size: i32,
    ) -> Result<crate::models::deepstack::MropePositions> {
        crate::models::deepstack::mrope_positions_mm(
            input_ids,
            image_grid_thw,
            image_token_id,
            video_grid_thw,
            video_token_id,
            spatial_merge_size,
        )
    }

    /// Run the decoder over precomputed input `embeds` `[1, S, hidden]` (text embeds with image
    /// features spliced in) using **interleaved multimodal RoPE** from explicit 3-D `positions`
    /// (temporal/height/width rows, each length `S`) **and DeepStack feature fusion**: after decoder
    /// layer `i`, for `i < deepstack.len()`, the `i`-th tapped/merged ViT feature set is added to the
    /// visual-token rows (`visual_pos_mask[p]` marks an image-token position). Returns last-position
    /// logits `[1, vocab]`. This is the Qwen3-VL prefill seam (`Qwen3VLTextModel.forward` +
    /// `_deepstack_process`); with all three position rows equal and an empty `deepstack` it is
    /// bit-identical to a plain 1-D-RoPE prefill.
    pub fn decode_logits_from_embeds_mrope_deepstack(
        &self,
        embeds: &Array,
        positions: [&[i32]; 3],
        cache: &mut dyn KvCache,
        visual_pos_mask: &[bool],
        deepstack: &[Array],
    ) -> Result<Array> {
        // Interleaved M-RoPE is a Qwen3-VL schedule; no Gemma 4 model reaches this entry (it has no
        // `mrope_section`), so the single-table pass is the whole model.
        let (cos, sin) = self.rope.mrope_interleaved_cos_sin(
            positions,
            self.cfg.mrope_section_resolved(),
            COMPUTE_DTYPE,
        )?;
        let ropes = RopeTables::uniform(cos, sin);
        let h0 = embeds.as_dtype(COMPUTE_DTYPE)?;
        let s = h0.shape()[1];
        let layers = &self.layers;
        let mut shared = SharedKv::default();
        let h = deepstack_fused_decoder_layers(
            &h0,
            visual_pos_mask,
            deepstack,
            layers.len(),
            |i, h| layers[i].forward(h, &ropes, AttnMask::Causal, cache, i, &mut shared),
        )?;
        let last_h = take_last(&h, s)?;
        let logits = self.project_logits(&last_h)?;
        Ok(logits.reshape(&[logits.shape()[0], self.cfg.vocab_size])?)
    }

    /// Run a forward step over token ids and return logits for **every** position,
    /// `[batch, seq, vocab]` — the all-position output speculative decoding (story 7171) verifies K
    /// proposed tokens in one pass.
    pub fn decode_logits_all(
        &self,
        input_ids: &Array,
        cache: &mut dyn KvCache,
        offset: i32,
    ) -> Result<Array> {
        let embeds = self.embed(input_ids)?;
        let s = embeds.shape()[1];
        let ropes = self.stack_rope(s, offset)?;
        let h = self.run_decoder_stack(&embeds, cache, &ropes, AttnMask::Causal)?;
        self.project_logits(&h) // [batch, seq, vocab]
    }

    /// Batched forward over a **left-padded** `[batch, seq]` step with **per-sequence** RoPE positions
    /// and an explicit additive attention mask — the dynamic-batch scheduler decode primitive.
    ///
    /// The caller supplies one `(cos, sin)` pair, which is every uniform architecture's whole RoPE.
    /// A Gemma 4 model needs two (its layer types rotate different head dims) and is refused here —
    /// use [`CausalLm::decode_logits_masked_at`], which takes the positions and builds both.
    pub fn decode_logits_masked(
        &self,
        input_ids: &Array,
        cache: &mut dyn KvCache,
        cos: &Array,
        sin: &Array,
        mask: &Array,
    ) -> Result<Array> {
        if self.needs_per_type_rope() {
            return Err(Error::Msg(
                "decode_logits_masked takes one caller-built (cos, sin) pair, which cannot cover \
                 Gemma 4's two per-layer-type RoPE schedules; call decode_logits_masked_at with \
                 the positions instead"
                    .into(),
            ));
        }
        let embeds = self.embed(input_ids)?;
        let ropes = RopeTables::uniform(cos.clone(), sin.clone());
        self.forward_to_last_logits(&embeds, cache, &ropes, AttnMask::Additive(mask))
    }

    /// [`CausalLm::decode_logits_masked`] from the **positions** rather than pre-built RoPE tables,
    /// so it can serve a per-layer-type model: `positions` is a row-major `[batch, seq]` grid of
    /// absolute positions, and every layer type's tables are built from it.
    ///
    /// For a uniform architecture this is exactly `rope_tables` + `decode_logits_masked`.
    pub fn decode_logits_masked_at(
        &self,
        input_ids: &Array,
        cache: &mut dyn KvCache,
        positions: &[i32],
        mask: &Array,
    ) -> Result<Array> {
        let sh = input_ids.shape();
        let (b, s) = (sh[0], sh[1]);
        if positions.len() != (b * s) as usize {
            return Err(Error::Msg(format!(
                "decode_logits_masked_at: {} positions for a {b}x{s} step",
                positions.len()
            )));
        }
        let embeds = self.embed(input_ids)?;
        let ropes = self.stack_rope_at(positions, b, s)?;
        self.forward_to_last_logits(&embeds, cache, &ropes, AttnMask::Additive(mask))
    }

    /// Throughput-mode batched decode forward for iteration-level continuous batching (story 7281):
    /// batched embed / projections / MLP / lm_head over a `[batch, seq]` step, attention per-sequence.
    pub fn decode_logits_per_seq(
        &self,
        input_ids: &Array,
        caches: &mut [&mut PagedKvCache],
        positions: &[i32],
    ) -> Result<Array> {
        let sh = input_ids.shape();
        let (b, s) = (sh[0], sh[1]);
        if caches.len() != b as usize {
            return Err(Error::Msg(format!(
                "decode_logits_per_seq: {} caches for a batch of {b}",
                caches.len()
            )));
        }
        if positions.len() != (b * s) as usize {
            return Err(Error::Msg(format!(
                "decode_logits_per_seq: {} positions for a {b}x{s} step",
                positions.len()
            )));
        }
        let ropes = self.stack_rope_at(positions, b, s)?;
        let mut h = self.embed(input_ids)?;
        for (i, layer) in self.layers.iter().enumerate() {
            h = layer.forward_per_seq(&h, &ropes, caches, i)?;
        }
        let last_h = take_last(&h, s)?; // [b, 1, hidden]
        let logits = self.project_logits(&last_h)?; // [b, 1, vocab]
        Ok(logits.reshape(&[b, self.cfg.vocab_size])?)
    }

    /// Run a forward step over token ids and return **every layer's** hidden states rather than
    /// logits — the `output_hidden_states=True` stack a *text encoder* consumes (LTX-2.5 stacks all
    /// of them into its feature extractor; only a language-model head wants logits).
    ///
    /// See [`CausalLm::hidden_states_from_embeds`] for the returned layout.
    ///
    /// Masks purely causally. A caller whose batch is **padded** must use
    /// [`CausalLm::hidden_states_with_mask`] instead — see its docs for why causal-only masking is
    /// silently wrong there.
    pub fn hidden_states(
        &self,
        input_ids: &Array,
        cache: &mut dyn KvCache,
        offset: i32,
    ) -> Result<Vec<Array>> {
        self.hidden_states_with_mask(input_ids, cache, offset, AttnMask::Causal)
    }

    /// [`CausalLm::hidden_states`] under an explicit attention `mask`.
    ///
    /// The reason this exists: a *text encoder* runs a **padded** sequence, and
    /// [`AttnMask::Causal`] alone does not mask padding. Under causal-only masking every valid
    /// token attends the pad run, so every returned hidden state — and every feature built from
    /// them — is wrong, while staying finite, non-zero and correctly shaped. Pass
    /// [`AttnMask::Additive`] carrying `valid(i, j) = j <= i && mask01[j] != 0` (LTX's
    /// `causal_padding_mask`) to mask both at once; a `sliding_attention` layer still narrows it to
    /// its own window on top (see `LlamaLayer::windowed`).
    ///
    /// The mask must be in the model's [`CausalLm::compute_dtype`] — MLX's fused kernel requires
    /// the mask type to promote to the output type, and f32 does not promote to bf16.
    pub fn hidden_states_with_mask(
        &self,
        input_ids: &Array,
        cache: &mut dyn KvCache,
        offset: i32,
        mask: AttnMask<'_>,
    ) -> Result<Vec<Array>> {
        let embeds = self.embed(input_ids)?;
        self.hidden_states_from_embeds_with_mask(&embeds, cache, offset, mask)
    }

    /// Like [`CausalLm::hidden_states`] but from pre-computed input embeddings.
    ///
    /// Returns `num_layers + 1` tensors, each `[batch, seq, hidden]`, in Hugging Face's
    /// `output_hidden_states` layout:
    ///
    /// * `[0]` — the input embeddings (post embedding-scale), i.e. the first layer's input.
    /// * `[i]` for `1 <= i < num_layers` — the output of decoder layer `i - 1`.
    /// * `[num_layers]` — the **final-normed** output of the last layer (HF ties the last entry to
    ///   `last_hidden_state`, which is post-`model.norm`), *not* the raw layer output.
    ///
    /// Getting that last entry wrong is invisible in a decode smoke test — logits go through the
    /// same norm either way — but silently shifts every feature an encoder consumer builds.
    ///
    /// # Evaluate the returned states in order
    ///
    /// MLX is lazy: these are `num_layers + 1` handles onto **one** unevaluated graph. Forcing only
    /// the last (or all of them at once) submits the entire stack — every weight page-in and every
    /// layer's matmuls — as a single Metal command buffer. On a large model that exceeds what the
    /// GPU accepts and comes back as `kIOGPUCommandBufferCallbackErrorSubmissionsIgnored`, which
    /// reads like a driver fault rather than "the batch was too big": measured on the real 48-layer
    /// LTX-2.5 text encoder (26.3 GB), where a 64-token forward fails outright that way.
    ///
    /// Walking the returned slice in order and evaluating each entry splits the work into one
    /// command buffer per layer and runs cleanly (52.8 s, 24.9 GB peak for that same forward). A
    /// consumer that streams the states — which is what a feature extractor does anyway — gets this
    /// for free; one that grabs `states.last()` does not.
    ///
    /// Masks purely causally; see [`CausalLm::hidden_states_from_embeds_with_mask`] for the padded
    /// (text-encoder) case.
    pub fn hidden_states_from_embeds(
        &self,
        input_embeds: &Array,
        cache: &mut dyn KvCache,
        offset: i32,
    ) -> Result<Vec<Array>> {
        self.hidden_states_from_embeds_with_mask(input_embeds, cache, offset, AttnMask::Causal)
    }

    /// [`CausalLm::hidden_states_from_embeds`] under an explicit attention `mask` — the
    /// pre-computed-embeddings twin of [`CausalLm::hidden_states_with_mask`], whose docs carry the
    /// padded-sequence rationale and the dtype requirement.
    pub fn hidden_states_from_embeds_with_mask(
        &self,
        input_embeds: &Array,
        cache: &mut dyn KvCache,
        offset: i32,
        mask: AttnMask<'_>,
    ) -> Result<Vec<Array>> {
        let s = input_embeds.shape()[1];
        let ropes = self.stack_rope(s, offset)?;
        let mut out = Vec::with_capacity(self.layers.len() + 1);
        self.run_decoder_stack_collecting(input_embeds, cache, &ropes, mask, Some(&mut out))?;
        if let Some(last) = out.last_mut() {
            *last = rms_norm(last, &self.norm, self.cfg.rms_norm_eps)?;
        }
        Ok(out)
    }

    /// Run the decoder stack over `input_embeds` with the given RoPE tables and attention mask, and
    /// project the **last column** to logits `[batch, vocab]`.
    fn forward_to_last_logits(
        &self,
        input_embeds: &Array,
        cache: &mut dyn KvCache,
        ropes: &RopeTables,
        mask: AttnMask<'_>,
    ) -> Result<Array> {
        let sh = input_embeds.shape();
        let (b, s) = (sh[0], sh[1]);
        let h = self.run_decoder_stack(input_embeds, cache, ropes, mask)?;
        let last_h = take_last(&h, s)?; // [b, 1, hidden]
        let logits = self.project_logits(&last_h)?; // [b, 1, vocab]
        Ok(logits.reshape(&[b, self.cfg.vocab_size])?)
    }

    /// Run every decoder layer, returning the final hidden states `[batch, seq, hidden]`.
    fn run_decoder_stack(
        &self,
        input_embeds: &Array,
        cache: &mut dyn KvCache,
        ropes: &RopeTables,
        mask: AttnMask<'_>,
    ) -> Result<Array> {
        self.run_decoder_stack_collecting(input_embeds, cache, ropes, mask, None)
    }

    /// [`CausalLm::run_decoder_stack`] with an optional sink for **every** layer's output — one
    /// loop, so the hidden-state-stack forward cannot drift from the logits forward. Gemma 4's
    /// per-layer-type attention (sc-18760) resolves *inside* this loop — each layer picks its RoPE
    /// table, narrows the mask to its own window, and reads or stores shared K/V — so there is
    /// still exactly one stack pass to keep correct. Mirrors candle-llm's
    /// `run_decoder_stack_collecting`.
    fn run_decoder_stack_collecting(
        &self,
        input_embeds: &Array,
        cache: &mut dyn KvCache,
        ropes: &RopeTables,
        mask: AttnMask<'_>,
        mut collect: Option<&mut Vec<Array>>,
    ) -> Result<Array> {
        let mut h = input_embeds.clone();
        if let Some(sink) = collect.as_deref_mut() {
            sink.push(h.clone());
        }
        let mut shared = SharedKv::default();
        for (i, layer) in self.layers.iter().enumerate() {
            h = layer.forward(&h, ropes, mask, cache, i, &mut shared)?;
            if let Some(sink) = collect.as_deref_mut() {
                sink.push(h.clone());
            }
        }
        Ok(h)
    }

    /// Final RMSNorm + `lm_head` (+ Gemma-2 logit soft-cap) over hidden states `[batch, n, hidden]`.
    fn project_logits(&self, h: &Array) -> Result<Array> {
        let normed = rms_norm(h, &self.norm, self.cfg.rms_norm_eps)?;
        let logits = linear(&normed, &self.lm_head, None)?;
        match self.final_softcap {
            // Soft-cap in f32 for precision (the cap denominator matters near the extremes).
            Some(c) => soft_cap(&logits.as_dtype(Dtype::Float32)?, c),
            None => Ok(logits),
        }
    }
}

impl crate::decode::Decode for CausalLm {
    fn make_cache(&self) -> Box<dyn KvCache> {
        Box::new(self.new_cache())
    }

    fn step(&self, input_ids: &Array, cache: &mut dyn KvCache, offset: i32) -> Result<Array> {
        self.decode_logits(input_ids, cache, offset)
    }
}

impl crate::models::VlmDecode for CausalLm {
    fn embed_input_ids(&self, input_ids: &Array) -> Result<Array> {
        CausalLm::embed_input_ids(self, input_ids)
    }

    fn splice_vision_features(
        &self,
        embeds: &Array,
        input_ids: &[i32],
        vision_features: &Array,
        placeholder_tokens: &[i32],
    ) -> Result<Array> {
        CausalLm::splice_vision_features(
            self,
            embeds,
            input_ids,
            vision_features,
            placeholder_tokens,
        )
    }

    fn mrope_positions_mm(
        &self,
        input_ids: &[i32],
        image_grid_thw: &[[i32; 3]],
        image_token_id: i32,
        video_grid_thw: &[[i32; 3]],
        video_token_id: i32,
        spatial_merge_size: i32,
    ) -> Result<crate::models::deepstack::MropePositions> {
        CausalLm::mrope_positions_mm(
            self,
            input_ids,
            image_grid_thw,
            image_token_id,
            video_grid_thw,
            video_token_id,
            spatial_merge_size,
        )
    }

    fn prefill_with_deepstack(
        &self,
        embeds: &Array,
        positions: [&[i32]; 3],
        cache: &mut dyn KvCache,
        visual_pos_mask: &[bool],
        deepstack: &[Array],
    ) -> Result<Array> {
        // The generic decoder's cache is already the trait-object form — no downcast needed.
        self.decode_logits_from_embeds_mrope_deepstack(
            embeds,
            positions,
            cache,
            visual_pos_mask,
            deepstack,
        )
    }
}

/// Slice the last position off the seq axis, keeping the axis (`[b, 1, hidden]`).
fn take_last(h: &Array, s: i32) -> Result<Array> {
    let last_idx = Array::from_slice(&[s - 1], &[1]);
    Ok(h.take_axis(&last_idx, 1)?)
}

/// One transformer block. Pre-norm by default (Llama / Qwen / Phi); Gemma-2 / GLM-4 / Gemma 4 add
/// the post-attention and post-feedforward norms (`pre_ff_ln` / `post_ff_ln` are `Some`) for the
/// 4-norm "sandwich" residual.
#[derive(Debug)]
struct LlamaLayer {
    /// Pre-attention norm.
    input_ln: Array,
    /// Llama: the MLP pre-norm. Sandwich: the post-attention norm.
    post_ln: Array,
    /// Sandwich only: the MLP pre-norm.
    pre_ff_ln: Option<Array>,
    /// Sandwich only: the post-feedforward norm applied to the MLP output before the residual add.
    post_ff_ln: Option<Array>,
    attn: Attention,
    ffn: Ffn,
    eps: f32,
    /// Gemma 4's `layer_scalar` `[1]` buffer, multiplying the block output **after** both residual
    /// adds. `None` ⇒ no scaling (every architecture before Gemma 4).
    layer_scalar: Option<Array>,
    /// Which of the stack's [`RopeTables`] this layer rotates with.
    rope_slot: RopeSlot,
}

impl LlamaLayer {
    fn forward(
        &self,
        x: &Array,
        ropes: &RopeTables,
        mask: AttnMask<'_>,
        cache: &mut dyn KvCache,
        layer_idx: usize,
        shared: &mut SharedKv,
    ) -> Result<Array> {
        let (cos, sin) = ropes.get(self.rope_slot);
        let normed = rms_norm(x, &self.input_ln, self.eps)?;
        let attn = self
            .attn
            .forward(&normed, cos, sin, mask, cache, layer_idx, shared)?;
        self.combine_ffn(x, &attn)
    }

    fn forward_per_seq(
        &self,
        x: &Array,
        ropes: &RopeTables,
        caches: &mut [&mut PagedKvCache],
        layer_idx: usize,
    ) -> Result<Array> {
        let (cos, sin) = ropes.get(self.rope_slot);
        let normed = rms_norm(x, &self.input_ln, self.eps)?;
        let attn = self
            .attn
            .forward_per_seq(&normed, cos, sin, caches, layer_idx)?;
        self.combine_ffn(x, &attn)
    }

    /// The residual + MLP half shared by both forwards: the Llama pre-norm, or the 4-norm sandwich
    /// when `pre_ff_ln`/`post_ff_ln` are set. `x` is the block input, `attn` the attention output.
    ///
    /// Gemma 4's `layer_scalar` multiplies the result of *whichever* shape ran, after the final
    /// residual add — it scales the block's whole contribution, residual stream included.
    fn combine_ffn(&self, x: &Array, attn: &Array) -> Result<Array> {
        let out = match (&self.pre_ff_ln, &self.post_ff_ln) {
            // Sandwich (Gemma-2 / GLM-4 / Gemma 4): post-norm the attention and MLP outputs before
            // each add.
            (Some(pre_ff), Some(post_ff)) => {
                let attn = rms_norm(attn, &self.post_ln, self.eps)?;
                let h = add(x, &attn)?;
                let ffn = self.ffn.forward(&rms_norm(&h, pre_ff, self.eps)?)?;
                let ffn = rms_norm(&ffn, post_ff, self.eps)?;
                add(&h, &ffn)?
            }
            // Llama pre-norm: `post_ln` is the MLP pre-norm.
            _ => {
                let h = add(x, attn)?;
                let ffn = self.ffn.forward(&rms_norm(&h, &self.post_ln, self.eps)?)?;
                add(&h, &ffn)?
            }
        };
        match &self.layer_scalar {
            Some(s) => Ok(multiply(&out, &s.as_dtype(out.dtype())?)?),
            None => Ok(out),
        }
    }
}

/// A layer's self-attention: grouped-query attention (Llama family) or Multi-head Latent Attention
/// (DeepSeek-V2).
#[derive(Debug)]
enum Attention {
    Gqa(LlamaAttention),
    Mla(MlaAttention),
}

impl Attention {
    #[allow(clippy::too_many_arguments)]
    fn forward(
        &self,
        x: &Array,
        cos: &Array,
        sin: &Array,
        mask: AttnMask<'_>,
        cache: &mut dyn KvCache,
        layer_idx: usize,
        shared: &mut SharedKv,
    ) -> Result<Array> {
        match self {
            Attention::Gqa(a) => a.forward(x, cos, sin, mask, cache, layer_idx, shared),
            Attention::Mla(a) => a.forward(x, cos, sin, mask, cache, layer_idx),
        }
    }

    fn forward_per_seq(
        &self,
        x: &Array,
        cos: &Array,
        sin: &Array,
        caches: &mut [&mut PagedKvCache],
        layer_idx: usize,
    ) -> Result<Array> {
        match self {
            Attention::Gqa(a) => a.forward_per_seq(x, cos, sin, caches, layer_idx),
            Attention::Mla(_) => Err(Error::Msg(
                "continuous-batching Throughput mode is not supported for MLA (DeepSeek-V2); use the \
                 Exact mode"
                    .into(),
            )),
        }
    }
}

/// Grouped-query attention with RoPE, optional per-head q/k RMSNorm (Qwen3 / Gemma 4), optional
/// q/k/v bias (Qwen2 / GLM-4), interleaved RoPE (GLM-4), Gemma-2 score soft-cap, and Gemma 4's
/// per-layer-type shape (its own `head_dim` / KV-head count, sliding window, shared K/V projection,
/// and scale-free value norm).
#[derive(Debug)]
struct LlamaAttention {
    q: Projection,
    /// The key **and** value projections, which Gemma 4's `attention_k_eq_v` layers share.
    /// `None` ⇒ a Gemma 4 KV-sharing tail layer, which projects neither and reads [`SharedKv`].
    kv: Option<KvProjection>,
    o: Projection,
    q_norm: Option<Array>,
    /// `None` for an architecture without q/k norm, and for a KV-sharing tail layer (which
    /// normalizes nothing because it projects nothing).
    k_norm: Option<Array>,
    /// Gemma 4's `v_norm`: a per-head RMSNorm with **no learned weight** applied to the value
    /// heads. `false` for every earlier architecture, which leaves V un-normed.
    v_norm: bool,
    num_heads: i32,
    num_kv_heads: i32,
    head_dim: i32,
    scale: f32,
    eps: f32,
    /// Gemma-2 attention-score soft-cap; `None` ⇒ no cap.
    softcap: Option<f32>,
    /// Whether RoPE uses the interleaved (GPT-J) pairing (GLM-4).
    rope_interleaved: bool,
    /// Gemma 4 `sliding_attention` layers: attend only the `w` most recent keys (inclusive of the
    /// query's own position). `None` ⇒ the full causal prefix.
    sliding_window: Option<i32>,
    /// This layer's Gemma 4 layer type — the [`SharedKv`] slot it stores into or reads from.
    kind: LayerAttentionType,
    /// Whether this layer's post-cache K/V is the one the KV-sharing tail reuses
    /// (`store_full_length_kv`). Always `false` without KV sharing.
    stores_kv: bool,
}

impl LlamaAttention {
    /// Project `x` `[b, s, hidden]` into the query heads `[b, heads, s, head_dim]` — the projection,
    /// optional q RMSNorm, RoPE, and the transpose into head-major layout. Always runs, including on
    /// a KV-sharing tail layer (which shares K/V but keeps its own queries).
    fn project_q(&self, x: &Array, cos: &Array, sin: &Array) -> Result<Array> {
        let sh = x.shape();
        let (b, s) = (sh[0], sh[1]);
        let mut q = self
            .q
            .forward(x)?
            .reshape(&[b, s, self.num_heads, self.head_dim])?;
        if let Some(qn) = &self.q_norm {
            q = rms_norm(&q, qn, self.eps)?;
        }
        Ok(apply_rope(&q, cos, sin, self.rope_interleaved)?.transpose_axes(&[0, 2, 1, 3])?)
    }

    /// Project `x` into the key and value heads, both `[b, kv_heads, s, head_dim]`.
    ///
    /// **The K and V paths diverge before the first norm.** V is built from the *raw* projection
    /// output and takes only the scale-free `v_norm`; K takes the learned `k_norm` **and** RoPE.
    /// Under `attention_k_eq_v` that raw output is one shared tensor, so the layer pays a single
    /// matmul — but K and V are still different tensors, because a rotated, learned-scaled key is
    /// not an unrotated, unscaled value. Reading V off the *normed* K instead is the silent
    /// corruption this ordering exists to prevent.
    fn project_kv(
        &self,
        kv: &KvProjection,
        x: &Array,
        cos: &Array,
        sin: &Array,
    ) -> Result<(Array, Array)> {
        let sh = x.shape();
        let (b, s) = (sh[0], sh[1]);
        let shape = [b, s, self.num_kv_heads, self.head_dim];

        let (raw_k, raw_v) = kv.forward(x)?;
        let mut k = raw_k.reshape(&shape)?;
        let mut v = raw_v.reshape(&shape)?;

        if self.v_norm {
            v = rms_norm_unscaled(&v, self.eps)?;
        }
        if let Some(kn) = &self.k_norm {
            k = rms_norm(&k, kn, self.eps)?;
        }

        let k = apply_rope(&k, cos, sin, self.rope_interleaved)?.transpose_axes(&[0, 2, 1, 3])?;
        let v = v.transpose_axes(&[0, 2, 1, 3])?;
        Ok((k, v))
    }

    /// This layer's effective mask: the caller's, narrowed to the layer's sliding window when it
    /// has one. `buf` owns the combined mask when the two must be intersected.
    ///
    /// A window and an explicit per-sequence mask are both "0 keeps, a large negative blocks", so
    /// their intersection is their **sum** — which is how a left-padded batched decode step keeps
    /// its padding masked while a `sliding_attention` layer also bounds how far back it sees.
    fn windowed<'m>(
        &self,
        mask: AttnMask<'m>,
        q_len: i32,
        k_len: i32,
        buf: &'m mut Option<Array>,
    ) -> Result<AttnMask<'m>> {
        let Some(window) = self.sliding_window else {
            return Ok(mask);
        };
        Ok(match mask {
            // The window subsumes causality (it is causal *and* lower-bounded).
            AttnMask::None | AttnMask::Causal => AttnMask::SlidingCausal { window },
            AttnMask::SlidingCausal { window: outer } => AttnMask::SlidingCausal {
                window: outer.min(window),
            },
            AttnMask::Additive(a) => {
                // Back to the caller's dtype: the window is built in f32, and the fused kernel
                // needs a mask that promotes to the (bf16) output type.
                let combined = add(a, &sliding_causal_mask(q_len, k_len, window)?)?;
                *buf = Some(combined.as_dtype(a.dtype())?);
                match buf.as_ref() {
                    Some(m) => AttnMask::Additive(m),
                    // Just assigned above; unreachable.
                    None => AttnMask::SlidingCausal { window },
                }
            }
        })
    }

    /// Project the attended output `[b, heads, s, head_dim]` back to `[b, s, hidden]` through `o`.
    fn output(&self, attn: &Array) -> Result<Array> {
        let sh = attn.shape(); // [b, heads, s, head_dim]
        let (b, s) = (sh[0], sh[2]);
        let merged =
            attn.transpose_axes(&[0, 2, 1, 3])?
                .reshape(&[b, s, self.num_heads * self.head_dim])?;
        self.o.forward(&merged)
    }

    #[allow(clippy::too_many_arguments)]
    fn forward(
        &self,
        x: &Array,
        cos: &Array,
        sin: &Array,
        mask: AttnMask<'_>,
        cache: &mut dyn KvCache,
        layer_idx: usize,
        shared: &mut SharedKv,
    ) -> Result<Array> {
        let q = self.project_q(x, cos, sin)?;
        // A KV-sharing tail layer neither projects nor caches K/V — it reads the full-length K/V the
        // last earlier layer of its type stored. Everything else projects and updates its cache.
        let (k_all, v_all) = match &self.kv {
            Some(kv) => {
                let (k, v) = self.project_kv(kv, x, cos, sin)?;
                let both = cache.update(layer_idx, &k, &v)?;
                if self.stores_kv {
                    shared.set(self.kind, both.clone());
                }
                both
            }
            None => shared.get(self.kind).cloned().ok_or_else(|| {
                Error::Msg(format!(
                    "layer {layer_idx} shares K/V with an earlier {} layer, but none stored any \
                     (internal invariant: `store_full_length_kv` marks one per layer type)",
                    self.kind.as_str()
                ))
            })?,
        };
        let mut buf = None;
        let mask = self.windowed(mask, q.shape()[2], k_all.shape()[2], &mut buf)?;
        // GQA-shaped K/V go straight to the fused kernel (native GQA); Gemma-2's score soft-cap forces
        // the eager path. `sdpa_capped` dispatches on `softcap` + head dims.
        let out = sdpa_capped(&q, &k_all, &v_all, self.scale, self.softcap, mask)?;
        self.output(&out)
    }

    fn forward_per_seq(
        &self,
        x: &Array,
        cos: &Array,
        sin: &Array,
        caches: &mut [&mut PagedKvCache],
        layer_idx: usize,
    ) -> Result<Array> {
        let q = self.project_q(x, cos, sin)?;
        let kv = self.kv.as_ref().ok_or_else(|| {
            Error::Msg(
                "continuous-batching Throughput mode is not supported for Gemma 4's KV-sharing \
                 tail layers (`num_kv_shared_layers`); use the Exact mode"
                    .into(),
            )
        })?;
        let (k, v) = self.project_kv(kv, x, cos, sin)?;
        let mut outs = Vec::with_capacity(caches.len());
        for (i, cache) in caches.iter_mut().enumerate() {
            let i = i as i32;
            let (qi, ki, vi) = (row_axis0(&q, i)?, row_axis0(&k, i)?, row_axis0(&v, i)?);
            let (k_all, v_all) = cache.update(layer_idx, &ki, &vi)?;
            let mut buf = None;
            let mask =
                self.windowed(AttnMask::Causal, qi.shape()[2], k_all.shape()[2], &mut buf)?;
            outs.push(sdpa_capped(
                &qi,
                &k_all,
                &v_all,
                self.scale,
                self.softcap,
                mask,
            )?);
        }
        let refs: Vec<&Array> = outs.iter().collect();
        let out = concatenate_axis(&refs, 0)?; // [b, heads, s, head_dim]
        self.output(&out)
    }
}

/// Multi-head Latent Attention (DeepSeek-V2).
///
/// MLA down-projects the input to a small shared KV latent (`kv_a_proj_with_mqa` → `kv_a_layernorm`,
/// width `kv_lora_rank`) plus a single shared rotary key sub-vector (`k_pe`, MQA-style). The latent is
/// up-projected (`kv_b_proj`) to per-head content keys (`k_nope`) and values. Queries split the same
/// way — a content part (`q_nope`) and a rotary part (`q_pe`), from a full `q_proj` or a low-rank
/// `q_a → norm → q_b`. RoPE rotates only the `qk_rope_head_dim` sub-vectors; the per-head key is
/// `[k_nope ‖ k_pe]` and the query `[q_nope ‖ q_pe]`, attended at `q_head_dim = qk_nope + qk_rope`.
///
/// This is the **correctness-first** materialized form: it reconstructs full per-head K (`q_head_dim`)
/// and V (`v_head_dim`) and caches them like ordinary attention, so the existing [`KvCache`] and
/// [`sdpa_capped`] are reused (the latent-caching "absorbed" optimization is a later throughput
/// concern). Heads are full MHA (no GQA expansion).
#[derive(Debug)]
struct MlaAttention {
    q_proj: Option<Projection>,
    q_a_proj: Option<Projection>,
    q_a_layernorm: Option<Array>,
    q_b_proj: Option<Projection>,
    kv_a_proj: Projection,
    kv_a_layernorm: Array,
    kv_b_proj: Projection,
    o_proj: Projection,
    num_heads: i32,
    qk_nope_head_dim: i32,
    qk_rope_head_dim: i32,
    v_head_dim: i32,
    kv_lora_rank: i32,
    scale: f32,
    eps: f32,
}

impl MlaAttention {
    fn load(
        w: &Weights,
        lp: &dyn Fn(&str) -> String,
        cfg: &ModelConfig,
        load_proj: &dyn Fn(&str, Option<Array>) -> Result<Projection>,
        req_bf16: &dyn Fn(String) -> Result<Array>,
    ) -> Result<Self> {
        let mla = cfg
            .mla
            .expect("MLA config present for a DeepSeek-V2 decoder");
        // Query: a low-rank `q_a → norm → q_b` when the model has a query LoRA, else a full `q_proj`.
        let (q_proj, q_a_proj, q_a_layernorm, q_b_proj) =
            if w.contains(&lp("self_attn.q_a_proj.weight")) {
                (
                    None,
                    Some(load_proj(&lp("self_attn.q_a_proj.weight"), None)?),
                    Some(req_bf16(lp("self_attn.q_a_layernorm.weight"))?),
                    Some(load_proj(&lp("self_attn.q_b_proj.weight"), None)?),
                )
            } else {
                (
                    Some(load_proj(&lp("self_attn.q_proj.weight"), None)?),
                    None,
                    None,
                    None,
                )
            };
        Ok(Self {
            q_proj,
            q_a_proj,
            q_a_layernorm,
            q_b_proj,
            kv_a_proj: load_proj(&lp("self_attn.kv_a_proj_with_mqa.weight"), None)?,
            kv_a_layernorm: req_bf16(lp("self_attn.kv_a_layernorm.weight"))?,
            kv_b_proj: load_proj(&lp("self_attn.kv_b_proj.weight"), None)?,
            o_proj: load_proj(&lp("self_attn.o_proj.weight"), None)?,
            num_heads: cfg.num_heads,
            qk_nope_head_dim: mla.qk_nope_head_dim,
            qk_rope_head_dim: mla.qk_rope_head_dim,
            v_head_dim: mla.v_head_dim,
            kv_lora_rank: mla.kv_lora_rank,
            scale: cfg.attn_scale(),
            eps: cfg.rms_norm_eps,
        })
    }

    fn forward(
        &self,
        x: &Array,
        cos: &Array,
        sin: &Array,
        mask: AttnMask<'_>,
        cache: &mut dyn KvCache,
        layer_idx: usize,
    ) -> Result<Array> {
        let sh = x.shape();
        let (b, s) = (sh[0], sh[1]);
        let nh = self.num_heads;
        let (nope, rope, vhd) = (
            self.qk_nope_head_dim,
            self.qk_rope_head_dim,
            self.v_head_dim,
        );
        let qhd = nope + rope; // per-head q/k dim attended over

        // Query → [b, s, nh, qhd], split into content (nope) and rotary (rope) parts.
        let q = match (&self.q_proj, &self.q_a_proj) {
            (Some(qp), _) => qp.forward(x)?,
            (None, Some(qa)) => {
                let c = qa.forward(x)?;
                let c = rms_norm(&c, self.q_a_layernorm.as_ref().unwrap(), self.eps)?;
                self.q_b_proj.as_ref().unwrap().forward(&c)?
            }
            _ => unreachable!("MLA query has either q_proj or q_a/q_b"),
        };
        let q = q.reshape(&[b, s, nh, qhd])?;
        let q_parts = split_sections(&q, &[nope], 3)?; // [q_nope, q_pe]
        let q_nope = &q_parts[0];
        let q_pe = &q_parts[1];

        // Shared KV latent + the single MQA rotary key.
        let kv = self.kv_a_proj.forward(x)?; // [b, s, kv_lora_rank + rope]
        let kv_parts = split_sections(&kv, &[self.kv_lora_rank], 2)?; // [compressed, k_pe_flat]
        let compressed = rms_norm(&kv_parts[0], &self.kv_a_layernorm, self.eps)?;
        let k_pe = kv_parts[1].reshape(&[b, s, 1, rope])?; // shared across heads

        // Up-project to per-head content keys and values: [b, s, nh, nope + vhd].
        let kv_b = self
            .kv_b_proj
            .forward(&compressed)?
            .reshape(&[b, s, nh, nope + vhd])?;
        let kv_b_parts = split_sections(&kv_b, &[nope], 3)?; // [k_nope, value]
        let k_nope = &kv_b_parts[0];
        let value = &kv_b_parts[1];

        // RoPE the rotary sub-vectors (interleaved); broadcast the shared key over heads.
        let q_pe = apply_rope(q_pe, cos, sin, true)?;
        let k_pe = apply_rope(&k_pe, cos, sin, true)?;
        let k_pe = broadcast_to(&k_pe, &[b, s, nh, rope])?;

        // Assemble full per-head q/k, then [b, nh, s, *] for attention.
        let q = concatenate_axis(&[q_nope, &q_pe], 3)?.transpose_axes(&[0, 2, 1, 3])?;
        let k = concatenate_axis(&[k_nope, &k_pe], 3)?.transpose_axes(&[0, 2, 1, 3])?;
        let v = value.transpose_axes(&[0, 2, 1, 3])?;

        let (k_all, v_all) = cache.update(layer_idx, &k, &v)?;
        // q/k head dim (qhd) ≠ v head dim (vhd) → `sdpa_capped` takes the eager path.
        let out = sdpa_capped(&q, &k_all, &v_all, self.scale, None, mask)?; // [b, nh, s, vhd]
        let out = out
            .transpose_axes(&[0, 2, 1, 3])?
            .reshape(&[b, s, nh * vhd])?;
        self.o_proj.forward(&out)
    }
}

/// Slice row `i` off the batch axis, keeping the axis (`[1, …]`).
fn row_axis0(a: &Array, i: i32) -> Result<Array> {
    let idx = Array::from_slice(&[i], &[1]);
    Ok(a.take_axis(&idx, 0)?)
}

/// A layer's feed-forward network: a dense gated MLP, or a sparse Mixture-of-Experts bank.
#[derive(Debug)]
enum Ffn {
    Dense(LlamaMlp),
    Moe(MoeMlp),
}

impl Ffn {
    fn forward(&self, x: &Array) -> Result<Array> {
        match self {
            Ffn::Dense(m) => m.forward(x),
            Ffn::Moe(m) => m.forward(x),
        }
    }
}

/// A gated MLP: SwiGLU (`silu`) by default, or GeGLU (tanh GELU, the Gemma activation) when `gelu`.
#[derive(Debug)]
struct LlamaMlp {
    gate: Projection,
    up: Projection,
    down: Projection,
    gelu: bool,
}

impl LlamaMlp {
    fn forward(&self, x: &Array) -> Result<Array> {
        let g = self.gate.forward(x)?;
        let g = if self.gelu { gelu_tanh(&g)? } else { silu(&g)? };
        let up = self.up.forward(x)?;
        self.down.forward(&multiply(&g, &up)?)
    }
}

/// A sparse Mixture-of-Experts feed-forward (Qwen2-MoE, DeepSeek-V2): a softmax router over `experts`
/// (top-k per token) plus an always-on `shared` expert. Correctness-first — each expert runs only on
/// its routed tokens (gathered, then scatter-added back), so active compute scales with
/// `experts_per_tok`. Top-k selection is done on the host. `n_group`/`topk_group` group-limited
/// routing (DeepSeek-V2-236B / V3) is not modelled — V2-Lite uses plain greedy top-k.
#[derive(Debug)]
struct MoeMlp {
    /// Router weight `[num_experts, hidden]`.
    router: Array,
    experts: Vec<LlamaMlp>,
    shared: LlamaMlp,
    /// Shared-expert sigmoid gate `[1, hidden]` (Qwen2-MoE); `None` ⇒ added ungated (DeepSeek-V2).
    shared_gate: Option<Array>,
    experts_per_tok: usize,
    norm_topk_prob: bool,
    /// Multiplier on the (un-normalized) routed weights — DeepSeek's `routed_scaling_factor`; `1.0`
    /// for Qwen2-MoE. Ignored when `norm_topk_prob` (the weights are renormalized instead).
    routed_scaling_factor: f32,
}

impl MoeMlp {
    fn forward(&self, x: &Array) -> Result<Array> {
        let sh = x.shape();
        let (b, s, h) = (sh[0], sh[1], sh[2]);
        let t = b * s;
        let dtype = x.dtype();
        let xf = x.reshape(&[t, h])?;
        let num_experts = self.experts.len();
        let k = self.experts_per_tok.min(num_experts).max(1);

        // Router probabilities (f32 softmax on the host, for a stable top-k).
        let logits = linear(&xf, &self.router, None)?; // [t, num_experts]
        let logits = to_f32_host(&logits)?; // row-major [t * num_experts]

        // Invert the per-token top-k into per-expert (token, weight) lists.
        let mut routed: Vec<Vec<(i32, f32)>> = vec![Vec::new(); num_experts];
        for ti in 0..t as usize {
            let row = &logits[ti * num_experts..(ti + 1) * num_experts];
            let m = row.iter().copied().fold(f32::MIN, f32::max);
            let exps: Vec<f32> = row.iter().map(|&x| (x - m).exp()).collect();
            let sum: f32 = exps.iter().sum();
            let probs: Vec<f32> = exps.iter().map(|&e| e / sum).collect();
            let mut idx: Vec<usize> = (0..num_experts).collect();
            idx.sort_unstable_by(|&a, &b| probs[b].total_cmp(&probs[a]));
            let top = &idx[..k];
            // Renormalize the top-k weights to sum to 1, or apply the routed scaling factor.
            let (denom, post_scale) = if self.norm_topk_prob {
                (
                    top.iter()
                        .map(|&e| probs[e])
                        .sum::<f32>()
                        .max(f32::MIN_POSITIVE),
                    1.0,
                )
            } else {
                (1.0, self.routed_scaling_factor)
            };
            for &e in top {
                routed[e].push((ti as i32, probs[e] / denom * post_scale));
            }
        }

        // Each expert runs on just its tokens; scatter the weighted outputs back.
        let mut out = zeros_dtype(&[t, h], dtype)?;
        for (e, toks) in routed.iter().enumerate() {
            if toks.is_empty() {
                continue;
            }
            let n = toks.len() as i32;
            let idx_i: Vec<i32> = toks.iter().map(|&(ti, _)| ti).collect();
            let idx_u: Vec<u32> = toks.iter().map(|&(ti, _)| ti as u32).collect();
            let wts: Vec<f32> = toks.iter().map(|&(_, w)| w).collect();
            let idx = Array::from_slice(&idx_i, &[n]);
            let idx_u = Array::from_slice(&idx_u, &[n]);
            let wts = Array::from_slice(&wts, &[n, 1]).as_dtype(dtype)?;
            let xe = xf.take_axis(&idx, 0)?; // [n, h]
            let ye = multiply(&self.experts[e].forward(&xe)?, &wts)?.reshape(&[n, 1, h])?;
            out = mlx_rs::ops::indexing::scatter_add_single(&out, &idx_u, &ye, 0)?;
        }

        // Always-on shared expert: Qwen2 gates it by sigmoid(x · shared_gateᵀ); DeepSeek adds it
        // ungated.
        let shared = self.shared.forward(&xf)?;
        let shared = match &self.shared_gate {
            Some(g) => {
                let sg = sigmoid(&linear(&xf, g, None)?)?; // [t, 1]
                multiply(&shared, &sg)?
            }
            None => shared,
        };
        Ok(add(&out, &shared)?.reshape(&[b, s, h])?)
    }
}

/// A layer type's index into the loader's two-slot `store_full_length_kv` table.
fn kind_slot(kind: LayerAttentionType) -> usize {
    match kind {
        LayerAttentionType::Sliding => 0,
        LayerAttentionType::Full => 1,
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
