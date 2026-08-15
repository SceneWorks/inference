//! The H3 condition-encoder forward: token embedding → causal Qwen3 decoder layers → the hidden
//! state at [`SELECT_HIDDEN`](super::SELECT_HIDDEN). That `[B, L, 5120]` tensor is the `context`
//! the H3 DiT consumes, **one row per presentation token, none dropped** — the reference applies no
//! chat template, so there is no prefix to slice (see
//! [`APPLIES_CHAT_TEMPLATE`](super::tokenizer::APPLIES_CHAT_TEMPLATE)).
//!
//! # The off-by-one, stated once
//!
//! HF `output_hidden_states` indexing: `hidden_states[k]` is the state **after running `k` decoder
//! layers**, so `hidden_states[0]` is the raw embedding and `hidden_states[50]` — the card's "50th
//! layer" — is the OUTPUT of 0-indexed layer **49**. This encoder therefore:
//!
//! - loads and runs layers `0..=49` only (50 of the checkpoint's 64);
//! - never applies the final `norm` (the selected state is pre-final-norm);
//! - never loads `lm_head`.
//!
//! Layers 50-63 plus `lm_head` are consequently **dead weight for generation**.
//!
//! # Store bf16, compute f32
//!
//! The published shards are bf16 and this loader keeps them that way; every projection then upcasts
//! per matmul through [`crate::nn::linear_nb`], the norms compute in f32
//! ([`crate::nn::rms_weighted`]), and the embedding lookup is explicitly widened. That is the split
//! `candle-gen-boogu` proved bit-identical to an f32 store for this exact tower (sc-12828), at half
//! the resident footprint — which on a 66.7 GB component is 33 GB of difference, not a nicety.
//!
//! # This lane materializes attention scores; MLX's does not
//!
//! MLX calls a fused `scaled_dot_product_attention` that streams the scores. candle has no such
//! kernel, so attention routes through `candle_gen::sdpa_budgeted_bhsd` with the shared
//! `ATTN_SCORES_BUDGET`. At **64 heads** the `[1, 64, S, S]` score tensor crosses `i32::MAX` at
//! S ≈ 5793 — well inside the presentation lengths a long `fl2va` prompt with two keyframe vision
//! blocks can reach — so the budgeted split is load-bearing here, not a precaution.

use candle_gen::candle_core::{DType, IndexOp, Tensor};
use candle_gen::grounding::{
    causal_mask, image_blocks, mrope_cos_sin, mrope_positions, repeat_kv, replace_seq, slice_seq,
    Rotary,
};
use candle_gen::{CandleError, Result, Weights};

use crate::nn::{linear_nb, rms_weighted, silu};

use super::{MiniMaxH3TeConfig, TE_COMPUTE_DTYPE};

/// Longest presentation the pre-built 1-D RoPE table covers.
///
/// The text path reads a prefix of a table built once at load. 16384 is ~4x the longest prompt the
/// reference conditioner has ever been observed to build and costs 4 MB; the grounded path does not
/// use the table at all (it builds MRoPE per call from the actual positions), so this bounds only
/// `t2va`. A presentation past it is a typed error, never a silent truncation.
pub const MAX_TEXT_POSITIONS: usize = 16_384;

/// `y = x · Wᵀ` for a bias-less Qwen3 projection stored `[out, in]`.
///
/// A newtype rather than a bare `Tensor` so the four attention projections and the three MLP ones
/// cannot be transposed past each other at a call site, and so `nbytes` has one home.
#[derive(Debug, Clone)]
struct Proj {
    weight: Tensor,
}

impl Proj {
    fn load(w: &Weights, key: &str, dtype: DType) -> Result<Self> {
        Ok(Self {
            weight: w.require(&format!("{key}.weight"))?.to_dtype(dtype)?,
        })
    }

    /// Upcasting forward: the weight is cast to the activation dtype per matmul, so a bf16 store
    /// computes f32 (see the module docs).
    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        linear_nb(x, &self.weight)
    }

    fn nbytes(&self) -> usize {
        self.weight.elem_count() * self.weight.dtype().size_in_bytes()
    }
}

/// GQA self-attention: 64 query / 8 kv heads, bias-less q/k/v/o, per-head q/k RMSNorm on the head
/// dim **before** RoPE, HF half-split RoPE, masked SDPA.
struct Qwen3Attention {
    q_proj: Proj,
    k_proj: Proj,
    v_proj: Proj,
    o_proj: Proj,
    q_norm: Tensor,
    k_norm: Tensor,
    num_heads: usize,
    num_kv_heads: usize,
    head_dim: usize,
    eps: f64,
}

impl Qwen3Attention {
    fn load(w: &Weights, prefix: &str, cfg: &MiniMaxH3TeConfig, dtype: DType) -> Result<Self> {
        // The q/k norms are `[head_dim]` vectors, ~0.5 KB each. Kept f32 so `rms_weighted` runs
        // f32-on-f32 whatever the projection store is — bit-identical to an all-f32 load.
        let q_norm = w
            .require(&format!("{prefix}.q_norm.weight"))?
            .to_dtype(DType::F32)?;
        let k_norm = w
            .require(&format!("{prefix}.k_norm.weight"))?
            .to_dtype(DType::F32)?;
        // A q/k norm sized to the projection rather than to a head is the plausible wrong shape,
        // and it would broadcast silently against `[B, S, H, D]` only when it is `[D]`.
        for (name, t) in [("q_norm", &q_norm), ("k_norm", &k_norm)] {
            if t.dims() != [cfg.head_dim] {
                return Err(CandleError::Msg(format!(
                    "minimax-h3 te {prefix}.{name}: expected [{}], got {:?}",
                    cfg.head_dim,
                    t.dims()
                )));
            }
        }
        Ok(Self {
            q_proj: Proj::load(w, &format!("{prefix}.q_proj"), dtype)?,
            k_proj: Proj::load(w, &format!("{prefix}.k_proj"), dtype)?,
            v_proj: Proj::load(w, &format!("{prefix}.v_proj"), dtype)?,
            o_proj: Proj::load(w, &format!("{prefix}.o_proj"), dtype)?,
            q_norm,
            k_norm,
            num_heads: cfg.num_heads,
            num_kv_heads: cfg.num_kv_heads,
            head_dim: cfg.head_dim,
            eps: cfg.rms_norm_eps,
        })
    }

    fn nbytes(&self) -> usize {
        self.q_proj.nbytes() + self.k_proj.nbytes() + self.v_proj.nbytes() + self.o_proj.nbytes()
    }

    /// `x`: `[b, s, hidden]`; `cos`/`sin`: `[s, head_dim/2]`; `mask`: additive `[b, 1, s, s]`.
    fn forward(&self, x: &Tensor, cos: &Tensor, sin: &Tensor, mask: &Tensor) -> Result<Tensor> {
        let (b, s, _) = x.dims3()?;
        let (nh, nkv, hd) = (self.num_heads, self.num_kv_heads, self.head_dim);

        let q = self.q_proj.forward(x)?.reshape((b, s, nh, hd))?;
        let k = self.k_proj.forward(x)?.reshape((b, s, nkv, hd))?;
        let v = self.v_proj.forward(x)?.reshape((b, s, nkv, hd))?;

        // Per-head RMSNorm over the head dim, then `[b,s,h,d] → [b,h,s,d]`.
        let q = rms_weighted(&q, &self.q_norm, self.eps)?
            .transpose(1, 2)?
            .contiguous()?;
        let k = rms_weighted(&k, &self.k_norm, self.eps)?
            .transpose(1, 2)?
            .contiguous()?;
        let v = v.transpose(1, 2)?.contiguous()?;

        let q = candle_gen::candle_nn::rotary_emb::rope(&q, cos, sin)?;
        let k = candle_gen::candle_nn::rotary_emb::rope(&k, cos, sin)?;
        let groups = nh / nkv;
        let k = repeat_kv(&k, groups)?;
        let v = repeat_kv(&v, groups)?;

        let scale = (hd as f64).powf(-0.5);
        let o = candle_gen::sdpa_budgeted_bhsd(
            &q,
            &k,
            &v,
            scale,
            Some(mask),
            candle_gen::candle_nn::ops::softmax_last_dim,
            candle_gen::ATTN_SCORES_BUDGET,
        )?;
        let o = o.transpose(1, 2)?.contiguous()?.reshape((b, s, nh * hd))?;
        self.o_proj.forward(&o)
    }
}

/// Qwen3 SwiGLU feed-forward: `down(silu(gate(x)) * up(x))`, bias-less. FFN width 25600 over a
/// 5120 hidden — the widest single tensor in the encoder.
struct Qwen3Mlp {
    gate: Proj,
    up: Proj,
    down: Proj,
}

impl Qwen3Mlp {
    fn load(w: &Weights, prefix: &str, dtype: DType) -> Result<Self> {
        Ok(Self {
            gate: Proj::load(w, &format!("{prefix}.gate_proj"), dtype)?,
            up: Proj::load(w, &format!("{prefix}.up_proj"), dtype)?,
            down: Proj::load(w, &format!("{prefix}.down_proj"), dtype)?,
        })
    }

    fn nbytes(&self) -> usize {
        self.gate.nbytes() + self.up.nbytes() + self.down.nbytes()
    }

    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let g = silu(&self.gate.forward(x)?)?;
        let u = self.up.forward(x)?;
        self.down.forward(&(g * u)?)
    }
}

/// Qwen3 decoder block (pre-norm residual): `h += attn(input_ln(h))`, then `h += mlp(post_ln(h))`.
struct Qwen3DecoderLayer {
    input_ln: Tensor,
    post_ln: Tensor,
    attn: Qwen3Attention,
    mlp: Qwen3Mlp,
    eps: f64,
}

impl Qwen3DecoderLayer {
    fn load(w: &Weights, prefix: &str, cfg: &MiniMaxH3TeConfig, dtype: DType) -> Result<Self> {
        Ok(Self {
            // f32 norm weights, for the same reason the q/k norms are f32.
            input_ln: w
                .require(&format!("{prefix}.input_layernorm.weight"))?
                .to_dtype(DType::F32)?,
            post_ln: w
                .require(&format!("{prefix}.post_attention_layernorm.weight"))?
                .to_dtype(DType::F32)?,
            attn: Qwen3Attention::load(w, &format!("{prefix}.self_attn"), cfg, dtype)?,
            mlp: Qwen3Mlp::load(w, &format!("{prefix}.mlp"), dtype)?,
            eps: cfg.rms_norm_eps,
        })
    }

    fn nbytes(&self) -> usize {
        self.attn.nbytes() + self.mlp.nbytes()
    }

    fn forward(&self, x: &Tensor, cos: &Tensor, sin: &Tensor, mask: &Tensor) -> Result<Tensor> {
        let normed = rms_weighted(x, &self.input_ln, self.eps)?;
        let h = (x + self.attn.forward(&normed, cos, sin, mask)?)?;
        let normed2 = rms_weighted(&h, &self.post_ln, self.eps)?;
        Ok((&h + self.mlp.forward(&normed2)?)?)
    }
}

/// MiniMax-H3's Qwen3-VL-32B condition encoder.
pub struct MiniMaxH3TextEncoder {
    embed_tokens: Tensor,
    layers: Vec<Qwen3DecoderLayer>,
    rotary: Rotary,
    /// 0-indexed decoder layer whose output is the context (`select_hidden - 1`).
    out_layer: usize,
    image_token_id: u32,
    mrope_section: [usize; 3],
    head_dim: usize,
    rope_theta: f32,
}

impl MiniMaxH3TextEncoder {
    /// Load from the `text_encoder` weights under `prefix` (normally
    /// [`LM_PREFIX`](super::LM_PREFIX) = `"model.language_model"`):
    /// `{prefix}.embed_tokens.weight` and `{prefix}.layers.{i}.…` for `i` in `0..select_hidden`.
    ///
    /// Deliberately loads **only** the layers it will run — `{prefix}.norm.weight`, layers
    /// `select_hidden..num_layers` and `lm_head.weight` are never touched. Pair this with
    /// [`lm_prefixes`](super::lm_prefixes) so the untouched tail is never read off disk either.
    pub fn from_weights(
        w: &Weights,
        prefix: &str,
        cfg: &MiniMaxH3TeConfig,
        dtype: DType,
    ) -> Result<Self> {
        let out_layer = cfg.out_layer()?;
        if out_layer >= cfg.num_layers {
            return Err(CandleError::Msg(format!(
                "minimax-h3 te: select_hidden {} needs layer {out_layer} but the encoder has {} \
                 layers",
                cfg.select_hidden, cfg.num_layers
            )));
        }
        // The interleaved MRoPE assignment is the only one this crate implements
        // (`candle_gen::grounding::mrope_cos_sin`). A config declaring the contiguous form would
        // otherwise be silently ignored, and no numeric gate downstream could see it.
        if !cfg.mrope_interleaved {
            return Err(CandleError::Msg(
                "minimax-h3 te: mrope_interleaved is false, but this port implements only the \
                 interleaved Qwen3-VL section assignment"
                    .into(),
            ));
        }

        let mut layers = Vec::with_capacity(out_layer + 1);
        for i in 0..=out_layer {
            layers.push(Qwen3DecoderLayer::load(
                w,
                &format!("{prefix}.layers.{i}"),
                cfg,
                dtype,
            )?);
        }
        let embed_tokens = w.require(&format!("{prefix}.embed_tokens.weight"))?;
        let device = embed_tokens.device().clone();
        Ok(Self {
            embed_tokens,
            layers,
            rotary: Rotary::new(cfg.head_dim, cfg.rope_theta, MAX_TEXT_POSITIONS, &device)?,
            out_layer,
            image_token_id: cfg.image_token_id,
            mrope_section: cfg.mrope_section,
            head_dim: cfg.head_dim,
            rope_theta: cfg.rope_theta,
        })
    }

    /// How many decoder layers were actually loaded — `select_hidden`, not `num_layers`. Exposed so
    /// a caller (and the real-weight smoke) can prove the trim is real.
    pub fn num_loaded_layers(&self) -> usize {
        self.layers.len()
    }

    /// Device bytes the loaded projections hold (the embedding table included).
    ///
    /// The conditioning stage's resident cost, readable without a profiler — which is what makes
    /// the staged-residency tripwire able to assert on the *encoder* rather than on a proxy.
    pub fn nbytes(&self) -> usize {
        self.embed_tokens.elem_count() * self.embed_tokens.dtype().size_in_bytes()
            + self
                .layers
                .iter()
                .map(Qwen3DecoderLayer::nbytes)
                .sum::<usize>()
    }

    /// Text-only (`t2va`) conditioning. `input_ids`: `[b, s]` u32; `attention_mask`: `[b, s]`
    /// (non-zero = attend). Returns the DiT context `[b, s, hidden]` in f32 — one row per
    /// presentation token.
    ///
    /// Uses plain 1-D RoPE: with no vision tokens Qwen3-VL's interleaved MRoPE sections all index
    /// the same sequential position, so it reduces exactly to standard RoPE.
    pub fn forward(&self, input_ids: &Tensor, attention_mask: &Tensor) -> Result<Tensor> {
        let (b, s) = input_ids.dims2()?;
        if s > MAX_TEXT_POSITIONS {
            return Err(CandleError::Msg(format!(
                "minimax-h3 te: a {s}-token presentation exceeds the {MAX_TEXT_POSITIONS}-position \
                 rotary table"
            )));
        }
        let (cos, sin) = self.rotary.text(s)?;
        let mask = self.mask(attention_mask, b, s)?;
        let mut hidden = self.embed(input_ids)?;
        for layer in &self.layers {
            hidden = layer.forward(&hidden, &cos, &sin, &mask)?;
        }
        Ok(hidden)
    }

    /// **Vision-grounded** conditioning (the `fl2va` image path): run the encoder with each
    /// reference's tower features spliced over its `<|image_pad|>` block and 3-D interleaved MRoPE
    /// positions, so the LM "sees" the keyframes while reading the prompt.
    ///
    /// Mirrors [`forward`](Self::forward) but (a) replaces the `<|image_pad|>` embeddings with the
    /// tower's merged `image_embeds` `[nⱼ, hidden]`, (b) additively injects each reference's
    /// `deepstack` feature at those positions for the first `deepstack.len()` layers, and (c) uses
    /// interleaved MRoPE — the image block carries its 2-D merged grid position, text stays
    /// sequential. Returns the same `[b, s, hidden]` context. `b = 1`.
    ///
    /// `<|video_pad|>` runs are **not** scanned: they belong to `ref2va` (sc-17157), which needs a
    /// multi-pad run scan the shared `candle_gen::grounding` helpers do not yet have.
    pub fn forward_with_images(
        &self,
        input_ids: &Tensor,
        attention_mask: &Tensor,
        image_embeds: &[Tensor],
        deepstack: &[Vec<Tensor>],
        grids: &[[i32; 3]],
    ) -> Result<Tensor> {
        let (b, s) = input_ids.dims2()?;
        if b != 1 {
            return Err(CandleError::Msg(format!(
                "minimax-h3 te (grounded): the vision splice is single-batch, got b = {b}"
            )));
        }
        let ids: Vec<u32> = input_ids.i(0)?.to_dtype(DType::U32)?.to_vec1::<u32>()?;

        let blocks = image_blocks(&ids, self.image_token_id);
        if blocks.is_empty() {
            return Err(CandleError::Msg(
                "minimax-h3 te (grounded): presentation has no vision-pad tokens".into(),
            ));
        }
        if blocks.len() != image_embeds.len() || blocks.len() != grids.len() {
            return Err(CandleError::Msg(format!(
                "minimax-h3 te (grounded): {} vision-pad run(s) but {} embeds / {} grids",
                blocks.len(),
                image_embeds.len(),
                grids.len()
            )));
        }

        // Token embeddings, then splice each reference's tower output over its block. Each
        // replacement is the same length as the block it replaces, so earlier splices do not shift
        // later indices.
        let mut hidden = self.embed(input_ids)?;
        for (k, &(start, len)) in blocks.iter().enumerate() {
            if image_embeds[k].dim(0)? != len {
                return Err(CandleError::Msg(format!(
                    "minimax-h3 te (grounded): reference {k} has {} vision tokens but its pad run \
                     is {len}",
                    image_embeds[k].dim(0)?
                )));
            }
            let img = image_embeds[k].unsqueeze(0)?.to_dtype(hidden.dtype())?;
            hidden = replace_seq(&hidden, &img, start, start + len, s)?;
        }

        let (pt, ph, pw) = mrope_positions(&ids, self.image_token_id, grids);
        let (cos, sin) = mrope_cos_sin(
            self.head_dim,
            self.mrope_section,
            self.rope_theta,
            &pt,
            &ph,
            &pw,
            hidden.device(),
        )?;
        let mask = self.mask(attention_mask, b, s)?;

        for (i, layer) in self.layers.iter().enumerate() {
            hidden = layer.forward(&hidden, &cos, &sin, &mask)?;
            for (k, &(start, len)) in blocks.iter().enumerate() {
                if i < deepstack[k].len() {
                    let ds = deepstack[k][i].unsqueeze(0)?.to_dtype(hidden.dtype())?;
                    let mid = (slice_seq(&hidden, start, start + len)? + ds)?;
                    hidden = replace_seq(&hidden, &mid, start, start + len, s)?;
                }
            }
        }
        debug_assert_eq!(self.layers.len(), self.out_layer + 1);
        Ok(hidden)
    }

    /// Token-embedding lookup, widened to the compute dtype.
    ///
    /// Explicit rather than implicit: with a bf16 store the gathered rows are bf16, and every
    /// downstream projection upcasts *the weight* to the activation dtype — so leaving the
    /// activation bf16 would run the whole 50-layer stack in bf16 and silently lose the parity-grade
    /// precision this encoder is specified at.
    fn embed(&self, input_ids: &Tensor) -> Result<Tensor> {
        let (b, s) = input_ids.dims2()?;
        let flat = input_ids.reshape(b * s)?.to_dtype(DType::U32)?;
        let rows = self.embed_tokens.index_select(&flat, 0)?;
        let hidden = self.embed_tokens.dim(1)?;
        Ok(rows.reshape((b, s, hidden))?.to_dtype(TE_COMPUTE_DTYPE)?)
    }

    /// Additive `[b, 1, s, s]` mask: causal, plus `-inf` on any key a row's `attention_mask` marks
    /// as padding.
    ///
    /// The shared `candle_gen::grounding::causal_mask` is causal-only, on the documented grounds
    /// that the candle tokenizers it was written for emit no padding. That holds for this crate's
    /// tokenizer too — but the encoder still *takes* a mask, matching the MLX sibling's signature,
    /// and a mask that were accepted and then ignored would be a silent contract lie the moment a
    /// caller batched two presentations. So the padding term is applied here rather than assumed
    /// away.
    fn mask(&self, attention_mask: &Tensor, b: usize, s: usize) -> Result<Tensor> {
        let dims = attention_mask.dims();
        if dims != [b, s] {
            return Err(CandleError::Msg(format!(
                "minimax-h3 te: attention_mask must be [{b}, {s}], got {dims:?}"
            )));
        }
        let dev = attention_mask.device();
        let causal = causal_mask(b, s, dev)?;
        let keep = attention_mask.to_dtype(DType::F32)?;
        // 0 where kept, -inf where padded, broadcast over the query axis. Built by SELECTION rather
        // than by `(1 - keep) * -inf`, which is `0 * -inf = NaN` on every kept column and would
        // poison the whole row through the softmax.
        let pad = keep
            .eq(0f32)?
            .where_cond(
                &Tensor::full(f32::NEG_INFINITY, (b, s), dev)?,
                &Tensor::zeros((b, s), DType::F32, dev)?,
            )?
            .reshape((b, 1, 1, s))?;
        Ok(causal.broadcast_add(&pad)?)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use candle_gen::candle_core::Device;
    use std::collections::HashMap;

    /// A tiny but structurally faithful Qwen3-VL text tower: 4 layers, hidden 8, GQA 4/2,
    /// head_dim 4, FFN 12 — the same shape relationships as the 32 B (non-square q projection,
    /// GQA repeat, per-head norms) at a size a CPU test can run.
    fn tiny_cfg() -> MiniMaxH3TeConfig {
        MiniMaxH3TeConfig {
            hidden_size: 8,
            num_layers: 4,
            num_heads: 4,
            num_kv_heads: 2,
            head_dim: 4,
            intermediate_size: 12,
            rms_norm_eps: 1e-6,
            rope_theta: 5_000_000.0,
            vocab_size: 16,
            select_hidden: 3,
            image_token_id: 5,
            video_token_id: 6,
            vision_start_token_id: 7,
            vision_end_token_id: 8,
            mrope_section: [1, 1, 0],
            mrope_interleaved: true,
        }
    }

    fn tiny_weights(cfg: &MiniMaxH3TeConfig, store: DType) -> Weights {
        let dev = Device::Cpu;
        let (nh, nkv, hd) = (cfg.num_heads, cfg.num_kv_heads, cfg.head_dim);
        let (hidden, inter) = (cfg.hidden_size, cfg.intermediate_size);
        let t = |shape: &[usize], seed: f32| {
            // Deterministic, non-constant, small — a constant table would make several of the
            // assertions below pass for the wrong reason.
            let n: usize = shape.iter().product();
            let v: Vec<f32> = (0..n)
                .map(|i| ((i as f32 * 0.37 + seed).sin()) * 0.2)
                .collect();
            Tensor::from_vec(v, shape, &dev)
                .unwrap()
                .to_dtype(store)
                .unwrap()
        };
        let mut m: HashMap<String, Tensor> = HashMap::new();
        m.insert(
            "lm.embed_tokens.weight".into(),
            t(&[cfg.vocab_size, hidden], 0.1),
        );
        // Present on purpose: the loader must NOT read it (the tap is pre-final-norm).
        m.insert("lm.norm.weight".into(), t(&[hidden], 0.2));
        m.insert("lm_head.weight".into(), t(&[cfg.vocab_size, hidden], 0.3));
        for i in 0..cfg.num_layers {
            let p = format!("lm.layers.{i}");
            let s = i as f32;
            m.insert(format!("{p}.input_layernorm.weight"), t(&[hidden], s + 0.4));
            m.insert(
                format!("{p}.post_attention_layernorm.weight"),
                t(&[hidden], s + 0.5),
            );
            m.insert(
                format!("{p}.self_attn.q_proj.weight"),
                t(&[nh * hd, hidden], s + 0.6),
            );
            m.insert(
                format!("{p}.self_attn.k_proj.weight"),
                t(&[nkv * hd, hidden], s + 0.7),
            );
            m.insert(
                format!("{p}.self_attn.v_proj.weight"),
                t(&[nkv * hd, hidden], s + 0.8),
            );
            m.insert(
                format!("{p}.self_attn.o_proj.weight"),
                t(&[hidden, nh * hd], s + 0.9),
            );
            m.insert(format!("{p}.self_attn.q_norm.weight"), t(&[hd], s + 1.0));
            m.insert(format!("{p}.self_attn.k_norm.weight"), t(&[hd], s + 1.1));
            m.insert(
                format!("{p}.mlp.gate_proj.weight"),
                t(&[inter, hidden], s + 1.2),
            );
            m.insert(
                format!("{p}.mlp.up_proj.weight"),
                t(&[inter, hidden], s + 1.3),
            );
            m.insert(
                format!("{p}.mlp.down_proj.weight"),
                t(&[hidden, inter], s + 1.4),
            );
        }
        Weights::from_map(m)
    }

    fn ids(v: &[u32]) -> Tensor {
        Tensor::from_vec(v.to_vec(), (1, v.len()), &Device::Cpu).unwrap()
    }

    fn ones_mask(n: usize) -> Tensor {
        Tensor::ones((1, n), DType::F32, &Device::Cpu).unwrap()
    }

    fn flat(t: &Tensor) -> Vec<f32> {
        t.flatten_all().unwrap().to_vec1::<f32>().unwrap()
    }

    /// The tap runs exactly `select_hidden` layers and stops — no final norm, no `lm_head`.
    #[test]
    fn only_the_selected_layers_are_loaded_and_run() {
        let cfg = tiny_cfg();
        let w = tiny_weights(&cfg, DType::F32);
        let te = MiniMaxH3TextEncoder::from_weights(&w, "lm", &cfg, DType::F32).unwrap();
        assert_eq!(te.num_loaded_layers(), cfg.select_hidden);
        assert_eq!(te.num_loaded_layers(), 3);
        assert!(te.num_loaded_layers() < cfg.num_layers);
        let out = te.forward(&ids(&[1, 2, 3, 4]), &ones_mask(4)).unwrap();
        assert_eq!(out.dims(), &[1, 4, cfg.hidden_size]);
        assert_eq!(out.dtype(), DType::F32);
        assert!(flat(&out).iter().all(|x| x.is_finite()));
    }

    /// **The off-by-one is directional.** Taking one layer more or one fewer must change the
    /// context — asserting only that `select_hidden = 50` produces *something* would pass with the
    /// index shifted either way.
    #[test]
    fn the_tap_index_moves_the_context_in_both_directions() {
        let base = tiny_cfg();
        let w = tiny_weights(&base, DType::F32);
        let run = |select: usize| {
            let mut c = base.clone();
            c.select_hidden = select;
            flat(
                &MiniMaxH3TextEncoder::from_weights(&w, "lm", &c, DType::F32)
                    .unwrap()
                    .forward(&ids(&[1, 2, 3, 4]), &ones_mask(4))
                    .unwrap(),
            )
        };
        let at = run(3);
        assert_ne!(at, run(2), "hidden_states[k-1] must differ from [k]");
        assert_ne!(at, run(4), "hidden_states[k+1] must differ from [k]");
    }

    /// A checkpoint whose `norm` and `lm_head` are present must still be read without them: the
    /// context is the raw residual stream. Applying the final norm here would be invisible to any
    /// shape or key-coverage check, so it is pinned by value.
    #[test]
    fn the_context_is_pre_final_norm() {
        let cfg = tiny_cfg();
        let w = tiny_weights(&cfg, DType::F32);
        let te = MiniMaxH3TextEncoder::from_weights(&w, "lm", &cfg, DType::F32).unwrap();
        let out = te.forward(&ids(&[1, 2, 3, 4]), &ones_mask(4)).unwrap();
        let normed = rms_weighted(
            &out,
            &w.require("lm.norm.weight").unwrap(),
            cfg.rms_norm_eps,
        )
        .unwrap();
        assert_ne!(
            flat(&out),
            flat(&normed),
            "the tap must be the pre-final-norm residual stream"
        );
    }

    /// **bf16 store, f32 compute.** The projections are stored bf16 and every matmul still runs
    /// f32, so the forward is bit-identical to an f32 store at half the resident footprint. If the
    /// embedding upcast or the per-matmul weight cast were removed the stack would compute bf16 and
    /// these vectors would separate.
    #[test]
    fn bf16_store_is_bit_identical_to_an_f32_store() {
        let cfg = tiny_cfg();
        let dev = Device::Cpu;
        // One source of bytes, two stores — so any difference is the store, not the draw.
        let w_f32 = tiny_weights(&cfg, DType::F32);
        let w_bf16 = {
            let mut m = HashMap::new();
            for k in w_f32.keys().cloned().collect::<Vec<_>>() {
                let t = w_f32.require(&k).unwrap().to_dtype(DType::BF16).unwrap();
                m.insert(k, t);
            }
            Weights::from_map(m)
        };
        // Round the f32 side through bf16 too: the comparison is store-width, not precision-of-draw.
        let w_ref = {
            let mut m = HashMap::new();
            for k in w_f32.keys().cloned().collect::<Vec<_>>() {
                let t = w_f32
                    .require(&k)
                    .unwrap()
                    .to_dtype(DType::BF16)
                    .unwrap()
                    .to_dtype(DType::F32)
                    .unwrap();
                m.insert(k, t);
            }
            Weights::from_map(m)
        };

        let run = |w: &Weights, store: DType| {
            flat(
                &MiniMaxH3TextEncoder::from_weights(w, "lm", &cfg, store)
                    .unwrap()
                    .forward(&ids(&[1, 2, 3, 4]), &ones_mask(4))
                    .unwrap(),
            )
        };
        let a = run(&w_ref, DType::F32);
        let b = run(&w_bf16, DType::BF16);
        assert!(a.iter().all(|x| x.is_finite()));
        assert_eq!(
            a, b,
            "a bf16 store must compute exactly what an f32 store does"
        );

        // And the win is real: the store dtype reaches the device tensors.
        let te = MiniMaxH3TextEncoder::from_weights(&w_bf16, "lm", &cfg, DType::BF16).unwrap();
        let dense = MiniMaxH3TextEncoder::from_weights(&w_ref, "lm", &cfg, DType::F32).unwrap();
        assert!(te.nbytes() < dense.nbytes());
        let _ = dev;
    }

    /// A `mrope_interleaved: false` config must be REFUSED, not quietly run through the
    /// interleaved kernel. Nothing numeric downstream could tell the two apart.
    #[test]
    fn a_contiguous_mrope_declaration_is_refused() {
        let mut cfg = tiny_cfg();
        cfg.mrope_interleaved = false;
        let w = tiny_weights(&cfg, DType::F32);
        let e = match MiniMaxH3TextEncoder::from_weights(&w, "lm", &cfg, DType::F32) {
            Ok(_) => panic!("a contiguous mrope declaration must be refused"),
            Err(e) => e.to_string(),
        };
        assert!(e.contains("mrope_interleaved"), "unexpected error: {e}");
    }

    /// The attention mask is honored rather than accepted and dropped: masking a key must change
    /// every row that could have attended to it.
    #[test]
    fn the_attention_mask_is_not_ignored() {
        let cfg = tiny_cfg();
        let w = tiny_weights(&cfg, DType::F32);
        let te = MiniMaxH3TextEncoder::from_weights(&w, "lm", &cfg, DType::F32).unwrap();
        let tokens = ids(&[1, 2, 3, 4]);
        let all = te.forward(&tokens, &ones_mask(4)).unwrap();
        let masked = te
            .forward(
                &tokens,
                &Tensor::from_vec(vec![1f32, 0., 1., 1.], (1, 4), &Device::Cpu).unwrap(),
            )
            .unwrap();
        assert_ne!(flat(&all), flat(&masked));
        // Row 0 attends only to itself either way, so it must be UNCHANGED — which is what proves
        // the mask is a key mask and not a blanket perturbation.
        let a = flat(&all.i((0, 0)).unwrap());
        let b = flat(&masked.i((0, 0)).unwrap());
        assert_eq!(a, b, "masking key 1 must not move row 0");
        // And no NaN: `0 * -inf` in the additive build would poison every row.
        assert!(flat(&masked).iter().all(|x| x.is_finite()));
    }

    /// The grounded path splices the tower's embeds over the pad run and injects deepstack at the
    /// first layers. Both must actually reach the output.
    #[test]
    fn the_vision_splice_and_deepstack_both_reach_the_output() {
        let cfg = tiny_cfg();
        let w = tiny_weights(&cfg, DType::F32);
        let te = MiniMaxH3TextEncoder::from_weights(&w, "lm", &cfg, DType::F32).unwrap();
        let dev = Device::Cpu;
        // [txt, pad×4, txt] with a 4×4 patch grid → merged 2×2 = 4 vision tokens.
        let tokens = ids(&[1, 5, 5, 5, 5, 2]);
        let mask = ones_mask(6);
        let embeds =
            vec![(Tensor::ones((4, cfg.hidden_size), DType::F32, &dev).unwrap() * 0.5).unwrap()];
        let zeros = vec![(0..3)
            .map(|_| Tensor::zeros((4, cfg.hidden_size), DType::F32, &dev).unwrap())
            .collect::<Vec<_>>()];
        let nonzero = vec![(0..3)
            .map(|k| {
                (Tensor::ones((4, cfg.hidden_size), DType::F32, &dev).unwrap()
                    * (0.05 * (k + 1) as f64))
                    .unwrap()
            })
            .collect::<Vec<_>>()];
        let grids = [[1i32, 4, 4]];

        let with_zero_ds = te
            .forward_with_images(&tokens, &mask, &embeds, &zeros, &grids)
            .unwrap();
        let with_real_ds = te
            .forward_with_images(&tokens, &mask, &embeds, &nonzero, &grids)
            .unwrap();
        assert_eq!(with_zero_ds.dims(), &[1, 6, cfg.hidden_size]);
        assert!(flat(&with_zero_ds).iter().all(|x| x.is_finite()));
        assert_ne!(
            flat(&with_zero_ds),
            flat(&with_real_ds),
            "deepstack features must reach the output"
        );

        // Different tower embeds must produce a different context — the splice is real.
        let other =
            vec![(Tensor::ones((4, cfg.hidden_size), DType::F32, &dev).unwrap() * -0.5).unwrap()];
        let swapped = te
            .forward_with_images(&tokens, &mask, &other, &zeros, &grids)
            .unwrap();
        assert_ne!(flat(&with_zero_ds), flat(&swapped));

        // And the grounded path is NOT the text path: MRoPE positions differ from the ramp.
        let text_only = te.forward(&tokens, &mask).unwrap();
        assert_ne!(flat(&text_only), flat(&with_zero_ds));
    }

    /// A pad-run count that disagrees with the number of references is a typed error, not a splice
    /// that silently drops a keyframe.
    #[test]
    fn a_reference_count_mismatch_is_a_typed_error() {
        let cfg = tiny_cfg();
        let w = tiny_weights(&cfg, DType::F32);
        let te = MiniMaxH3TextEncoder::from_weights(&w, "lm", &cfg, DType::F32).unwrap();
        let dev = Device::Cpu;
        let tokens = ids(&[1, 5, 5, 5, 5, 2]);
        let embeds = vec![
            Tensor::zeros((4, cfg.hidden_size), DType::F32, &dev).unwrap(),
            Tensor::zeros((4, cfg.hidden_size), DType::F32, &dev).unwrap(),
        ];
        let ds = vec![Vec::new(), Vec::new()];
        let e = te
            .forward_with_images(
                &tokens,
                &ones_mask(6),
                &embeds,
                &ds,
                &[[1, 4, 4], [1, 4, 4]],
            )
            .unwrap_err()
            .to_string();
        assert!(e.contains("vision-pad run"), "unexpected error: {e}");

        // A prompt with no pad run at all is refused rather than silently encoded as text.
        let e = te
            .forward_with_images(
                &ids(&[1, 2]),
                &ones_mask(2),
                &embeds[..1],
                &ds[..1],
                &[[1, 4, 4]],
            )
            .unwrap_err()
            .to_string();
        assert!(e.contains("no vision-pad tokens"), "unexpected error: {e}");
    }

    /// A presentation past the rotary table is a typed error, never a silent truncation to the
    /// table length — which would drop the tail of a long prompt with no diagnostic.
    #[test]
    fn an_overlong_presentation_is_refused_not_truncated() {
        let cfg = tiny_cfg();
        let w = tiny_weights(&cfg, DType::F32);
        let te = MiniMaxH3TextEncoder::from_weights(&w, "lm", &cfg, DType::F32).unwrap();
        let n = MAX_TEXT_POSITIONS + 1;
        let tokens = Tensor::zeros((1, n), DType::U32, &Device::Cpu).unwrap();
        let e = te
            .forward(
                &tokens,
                &Tensor::ones((1, n), DType::F32, &Device::Cpu).unwrap(),
            )
            .unwrap_err()
            .to_string();
        assert!(e.contains("rotary table"), "unexpected error: {e}");
    }
}
