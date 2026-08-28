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
    causal_mask, mrope_cos_sin, repeat_kv, replace_seq, slice_seq, Rotary,
};
use candle_gen::quant::AdaptLinear;
use candle_gen::{CandleError, Result, Weights};

use crate::nn::{rms_weighted, silu};

use super::{MiniMaxH3TeConfig, TE_COMPUTE_DTYPE};

/// Longest presentation the pre-built 1-D RoPE table covers.
///
/// The text path reads a prefix of a table built once at load. 16384 is ~4x the longest prompt the
/// reference conditioner has ever been observed to build and costs 4 MB; the grounded path does not
/// use the table at all (it builds MRoPE per call from the actual positions), so this bounds only
/// `t2va`. A presentation past it is a typed error, never a silent truncation.
pub const MAX_TEXT_POSITIONS: usize = 16_384;

/// `y = x · Wᵀ` for a bias-less Qwen3 projection stored `[out, in]` — **dense or MLX-packed**.
///
/// A newtype rather than a bare `Tensor` so the four attention projections and the three MLP ones
/// cannot be transposed past each other at a call site, and so `nbytes` has one home.
///
/// # Every projection this type loads is a pack target, and that is the whole surface
///
/// The seven bases routed through here — `self_attn.{q,k,v,o}_proj` and `mlp.{gate,up,down}_proj` —
/// are exactly the seven Qwen3 decoder entries in
/// `mlx_gen_minimax_h3::convert::TE_PACK_SUFFIXES`, so this one loader mirrors the converter's
/// entire decoder-side pack set. The tensors the converter leaves dense (`TE_DENSE_BY_POLICY`: the
/// q/k norms and the two layer norms) are read by their own `require` calls, each fronted by
/// [`crate::quant::guard_dense`].
#[derive(Debug, Clone)]
struct Proj {
    inner: AdaptLinear,
    /// Device bytes the frozen base holds, recorded at load — a packed base has no dense weight to
    /// measure afterwards. See [`crate::quant::TieredLinear`].
    base_bytes: usize,
}

impl Proj {
    /// Load `{key}` — **packed** when `{key}.scales` is present, else dense at `dtype`.
    fn load(w: &Weights, key: &str, dtype: DType) -> Result<Self> {
        let loaded = crate::quant::lin(w, crate::quant::TEXT_ENCODER, key, false, dtype)?;
        Ok(Self {
            inner: loaded.linear,
            base_bytes: loaded.base_bytes,
        })
    }

    /// Upcasting forward: the weight is cast to the activation dtype per matmul, so a bf16 store
    /// computes f32 (see the module docs).
    ///
    /// The activation is flattened to 2-D first, exactly as [`crate::dit::layers::LinearNoBias`]
    /// does and for the same reason: this preserves [`crate::nn::linear_nb`]'s rank contract, which
    /// the shared seam does not have — [`AdaptLinear`]'s dense base is a `candle_nn::Linear`, whose
    /// forward special-cases ranks 2-4 and falls through to a same-rank `matmul` otherwise. Folding
    /// every leading dim into the GEMM's `M` is additionally the *same* single large-`M` GEMM
    /// `linear_nb` issued, so the dense forward keeps its previous shape as well as its previous
    /// result.
    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let dims = x.dims().to_vec();
        let in_features = *dims.last().expect("Proj::forward: x has no axes");
        let rows = x.elem_count() / in_features;
        let y = self
            .inner
            .forward_upcast(&x.reshape((rows, in_features))?)?;
        let mut out_dims = dims;
        *out_dims.last_mut().expect("Proj::forward: x has no axes") = self.inner.out_features();
        Ok(y.reshape(out_dims)?)
    }

    fn nbytes(&self) -> usize {
        self.base_bytes
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
        //
        // `.q_norm` / `.k_norm` are named in `mlx_gen_minimax_h3::convert::TE_DENSE_BY_POLICY`, so
        // they are dense in every tier and read here with a raw `require` + `to_dtype`. That cast is
        // the unguarded surface: handed u32 codes it produces floats from a bit pattern and reports
        // nothing (sc-14980). `guard_dense` makes a tier that ever packed them a loud load error.
        crate::quant::guard_dense(w, crate::quant::TEXT_ENCODER, &format!("{prefix}.q_norm"))?;
        crate::quant::guard_dense(w, crate::quant::TEXT_ENCODER, &format!("{prefix}.k_norm"))?;
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
        // Both layer norms are `TE_DENSE_BY_POLICY` entries read through a raw `require` — guarded
        // for the same reason the q/k norms above are.
        crate::quant::guard_dense(
            w,
            crate::quant::TEXT_ENCODER,
            &format!("{prefix}.input_layernorm"),
        )?;
        crate::quant::guard_dense(
            w,
            crate::quant::TEXT_ENCODER,
            &format!("{prefix}.post_attention_layernorm"),
        )?;
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
    /// The token table — **dense or MLX-packed**. `.embed_tokens` is a
    /// `mlx_gen_minimax_h3::convert::TE_PACK_SUFFIXES` entry, so a published packed tier ships it as
    /// u32 codes; [`crate::quant::embed`] detects that on the `.scales` sibling.
    embed_tokens: crate::quant::TieredEmbedding,
    layers: Vec<Qwen3DecoderLayer>,
    rotary: Rotary,
    /// 0-indexed decoder layer whose output is the context (`select_hidden - 1`).
    out_layer: usize,
    image_token_id: u32,
    /// `<|video_pad|>`'s id — read only by [`Self::forward_with_references`]; `fl2va` never emits
    /// this pad, which is what keeps the two grounded paths distinguishable.
    video_token_id: u32,
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
        // The store dtype is passed through so the gathered rows come back exactly as the dense path
        // produced them; `embed` widens to the compute dtype after the gather, not before.
        let embed_tokens = crate::quant::embed(w, &format!("{prefix}.embed_tokens"), dtype)?;
        // A packed table is a `QTensor` with no `Tensor` to ask for a device, so the rotary table is
        // built on the device the *weights* are on — falling back to the `embed_tokens` weight
        // itself, which exists on both paths (a float table, or the u32 code stream) and is what a
        // `Weights::from_map` fixture carries.
        let device = match w.device() {
            Some(d) => d.clone(),
            None => w
                .require(&format!("{prefix}.embed_tokens.weight"))?
                .device()
                .clone(),
        };
        Ok(Self {
            embed_tokens,
            layers,
            rotary: Rotary::new(cfg.head_dim, cfg.rope_theta, MAX_TEXT_POSITIONS, &device)?,
            out_layer,
            image_token_id: cfg.image_token_id,
            video_token_id: cfg.video_token_id,
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

    /// Device bytes the loaded projections hold (the embedding table included when dense).
    ///
    /// The conditioning stage's resident device cost, readable without a profiler. A packed token
    /// table contributes zero here because its compact rows are host-resident and each lookup makes
    /// only a bounded, temporary device table for the selected token IDs.
    pub fn nbytes(&self) -> usize {
        self.embed_tokens.base_bytes
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
    /// `<|video_pad|>` runs are **not** scanned here: they belong to `ref2va`, and mixing them into
    /// the `fl2va` scan would let a stray video pad consume a keyframe's embeds. See
    /// [`Self::forward_with_references`].
    pub fn forward_with_images(
        &self,
        input_ids: &Tensor,
        attention_mask: &Tensor,
        image_embeds: &[Tensor],
        deepstack: &[Vec<Tensor>],
        grids: &[[i32; 3]],
    ) -> Result<Tensor> {
        self.forward_grounded(
            input_ids,
            attention_mask,
            image_embeds,
            deepstack,
            grids,
            &[self.image_token_id],
        )
    }

    /// The **`ref2va`** grounded forward (sc-17157): splice reference features into runs of *both*
    /// `<|image_pad|>` and `<|video_pad|>`.
    ///
    /// `embeds`, `deepstack` and `grids` are in **sequence order** — the order the pad runs appear
    /// in the presentation, which for `ref2va` is request order across modalities.
    ///
    /// # Why sequence order reproduces the reference's per-modality batching
    ///
    /// Qwen3-VL batches vision tensors *per modality* and fills the n-th pad run of a modality with
    /// the n-th entry of that modality's batch. Relative order **within** a modality is preserved by
    /// request order, so walking runs in sequence order while consuming features in request order
    /// yields exactly the same assignment — with one cursor instead of two, and therefore no way
    /// for the two to drift apart on an interleaved request.
    ///
    /// A run never spans two *different* pads, so an image block immediately followed by a video
    /// block stays two runs and two references.
    pub fn forward_with_references(
        &self,
        input_ids: &Tensor,
        attention_mask: &Tensor,
        embeds: &[Tensor],
        deepstack: &[Vec<Tensor>],
        grids: &[[i32; 3]],
    ) -> Result<Tensor> {
        self.forward_grounded(
            input_ids,
            attention_mask,
            embeds,
            deepstack,
            grids,
            &[self.image_token_id, self.video_token_id],
        )
    }

    fn forward_grounded(
        &self,
        input_ids: &Tensor,
        attention_mask: &Tensor,
        image_embeds: &[Tensor],
        deepstack: &[Vec<Tensor>],
        grids: &[[i32; 3]],
        pad_ids: &[u32],
    ) -> Result<Tensor> {
        let (b, s) = input_ids.dims2()?;
        if b != 1 {
            return Err(CandleError::Msg(format!(
                "minimax-h3 te (grounded): the vision splice is single-batch, got b = {b}"
            )));
        }
        let ids: Vec<u32> = input_ids.i(0)?.to_dtype(DType::U32)?.to_vec1::<u32>()?;

        let blocks = vision_pad_runs(&ids, pad_ids);
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

        let (pt, ph, pw) = mrope_positions_multi(&ids, pad_ids, grids);
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
        let rows = self.embed_tokens.embedding.forward(&flat)?;
        let hidden = self.embed_tokens.hidden;
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

/// Vision spatial merge — the LM sees one token per `merge²` patches (Qwen3-VL
/// `spatial_merge_size`). Mirrors the shared `candle_gen::grounding` constant, which is private.
const SPATIAL_MERGE: i32 = 2;

/// Contiguous runs of **any** of `pad_ids`, as `(start, len)` in sequence order.
///
/// The multi-pad generalization of `candle_gen::grounding::image_blocks`, which takes exactly one
/// pad id. `ref2va` interleaves `<|image_pad|>` and `<|video_pad|>` blocks, and a run **never spans
/// two different pads** — an image block immediately followed by a video block is two runs and two
/// references, so the run is broken on the id it started with rather than on "is a pad".
fn vision_pad_runs(ids: &[u32], pad_ids: &[u32]) -> Vec<(usize, usize)> {
    let mut runs = Vec::new();
    let mut i = 0usize;
    while i < ids.len() {
        if pad_ids.contains(&ids[i]) {
            let start = i;
            let first = ids[i];
            while i < ids.len() && ids[i] == first {
                i += 1;
            }
            runs.push((start, i - start));
        } else {
            i += 1;
        }
    }
    runs
}

/// Multi-pad 3-D MRoPE positions (mirrors Qwen3-VL `get_rope_index`): text tokens advance
/// `(i, i, i)`; the `k`-th vision block at offset `cur` gets `t = cur`, `h = cur + row`,
/// `w = cur + col` over its `(h/merge)×(w/merge)` merged grid, then `cur += max(h, w) / merge`.
///
/// The multi-pad generalization of `candle_gen::grounding::mrope_positions`; the single-pad case
/// is byte-identical to it, which `the_multi_pad_scan_reduces_to_the_shared_single_pad_one` pins.
fn mrope_positions_multi(
    ids: &[u32],
    pad_ids: &[u32],
    grids: &[[i32; 3]],
) -> (Vec<i64>, Vec<i64>, Vec<i64>) {
    let (mut pt, mut ph, mut pw) = (Vec::new(), Vec::new(), Vec::new());
    let mut cur = 0i64;
    let mut img_k = 0usize;
    let mut i = 0usize;
    while i < ids.len() {
        if pad_ids.contains(&ids[i]) && img_k < grids.len() {
            let g = grids[img_k];
            let (llm_h, llm_w) = (
                i64::from(g[1] / SPATIAL_MERGE),
                i64::from(g[2] / SPATIAL_MERGE),
            );
            let step = i64::from(g[1].max(g[2]) / SPATIAL_MERGE);
            for idx in 0..(llm_h * llm_w) {
                pt.push(cur);
                ph.push(cur + idx / llm_w);
                pw.push(cur + idx % llm_w);
            }
            cur += step;
            i += (llm_h * llm_w) as usize;
            img_k += 1;
        } else {
            pt.push(cur);
            ph.push(cur);
            pw.push(cur);
            cur += 1;
            i += 1;
        }
    }
    (pt, ph, pw)
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

    /// A **packable** tiny tower: every projection's input width is 64, so the MLX group-64 pack is
    /// expressible. [`tiny_cfg`]'s hidden of 8 cannot be packed at all (the group does not divide
    /// it), which is why the packed tests need their own shape rather than reusing that fixture.
    fn packable_cfg() -> MiniMaxH3TeConfig {
        MiniMaxH3TeConfig {
            hidden_size: 64,
            num_layers: 3,
            num_heads: 4,
            num_kv_heads: 2,
            head_dim: 16,
            intermediate_size: 64,
            select_hidden: 2,
            vocab_size: 64,
            ..tiny_cfg()
        }
    }

    /// The seven per-layer bases [`Proj`] loads — exactly the Qwen3 decoder entries in
    /// `mlx_gen_minimax_h3::convert::TE_PACK_SUFFIXES`, paired with their `[out, in]`.
    fn packable_projections(cfg: &MiniMaxH3TeConfig) -> Vec<(String, usize, usize)> {
        let (nh, nkv, hd) = (cfg.num_heads, cfg.num_kv_heads, cfg.head_dim);
        let (hidden, inter) = (cfg.hidden_size, cfg.intermediate_size);
        vec![
            ("self_attn.q_proj".into(), nh * hd, hidden),
            ("self_attn.k_proj".into(), nkv * hd, hidden),
            ("self_attn.v_proj".into(), nkv * hd, hidden),
            ("self_attn.o_proj".into(), hidden, nh * hd),
            ("mlp.gate_proj".into(), inter, hidden),
            ("mlp.up_proj".into(), inter, hidden),
            ("mlp.down_proj".into(), hidden, inter),
        ]
    }

    /// Build a text-encoder weight map for [`packable_cfg`].
    ///
    /// When `packed`, every [`Proj`] base and the token table are written as MLX packed triples and
    /// the returned map additionally carries the **dequantized affine grid** each one decodes to,
    /// under the plain `{base}.weight` key — that grid is the dense reference the packed forward is
    /// measured against (it is what the tier's producer quantized; the unquantized weight is a
    /// different tensor and would only measure quantization error).
    ///
    /// The norms are dense in both fixtures: `.q_norm` / `.k_norm` / `.input_layernorm` /
    /// `.post_attention_layernorm` are `TE_DENSE_BY_POLICY` entries.
    fn packable_weights(
        cfg: &MiniMaxH3TeConfig,
        packed: bool,
    ) -> (Weights, HashMap<String, Tensor>) {
        let dev = Device::Cpu;
        let hd = cfg.head_dim;
        let t = |shape: &[usize], seed: f32| {
            let n: usize = shape.iter().product();
            let v: Vec<f32> = (0..n)
                .map(|i| ((i as f32 * 0.37 + seed).sin()) * 0.2)
                .collect();
            Tensor::from_vec(v, shape, &dev).unwrap()
        };
        let mut m: HashMap<String, Tensor> = HashMap::new();
        let mut grids: HashMap<String, Tensor> = HashMap::new();

        // The token table — a pack target (`.embed_tokens`).
        if packed {
            let grid = crate::quant::testkit::insert_packed(
                &mut m,
                "lm.embed_tokens",
                cfg.vocab_size,
                cfg.hidden_size,
                1,
            );
            grids.insert("lm.embed_tokens.weight".into(), grid);
        } else {
            m.insert(
                "lm.embed_tokens.weight".into(),
                t(&[cfg.vocab_size, cfg.hidden_size], 0.1),
            );
        }

        for i in 0..cfg.num_layers {
            let p = format!("lm.layers.{i}");
            let s = i as f32;
            // Dense by policy, in every tier.
            m.insert(
                format!("{p}.input_layernorm.weight"),
                t(&[cfg.hidden_size], s + 0.4),
            );
            m.insert(
                format!("{p}.post_attention_layernorm.weight"),
                t(&[cfg.hidden_size], s + 0.5),
            );
            m.insert(format!("{p}.self_attn.q_norm.weight"), t(&[hd], s + 1.0));
            m.insert(format!("{p}.self_attn.k_norm.weight"), t(&[hd], s + 1.1));
            // Pack targets.
            for (n, (suffix, out, in_)) in packable_projections(cfg).into_iter().enumerate() {
                let base = format!("{p}.{suffix}");
                if packed {
                    let grid = crate::quant::testkit::insert_packed(
                        &mut m,
                        &base,
                        out,
                        in_,
                        i * 7 + n + 2,
                    );
                    grids.insert(format!("{base}.weight"), grid);
                } else {
                    m.insert(format!("{base}.weight"), t(&[out, in_], s + n as f32 * 0.1));
                }
            }
        }
        (Weights::from_map(m), grids)
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

    /// **Detection.** A `.scales` sibling packs every one of the seven decoder projections and the
    /// token table; the dense-by-policy norms stay dense. This is the candle mirror of
    /// `mlx_gen_minimax_h3::convert::is_te_pack_target` over the surface this crate loads.
    #[test]
    fn a_packed_tier_packs_every_projection_and_the_token_table() {
        let cfg = packable_cfg();
        let (w, _) = packable_weights(&cfg, true);
        let te = MiniMaxH3TextEncoder::from_weights(&w, "lm", &cfg, DType::F32).unwrap();

        assert!(
            te.embed_tokens.embedding.is_quantized(),
            "`.embed_tokens` is a TE_PACK_SUFFIXES entry — it must load packed"
        );
        assert_eq!(te.layers.len(), cfg.select_hidden);
        for (i, layer) in te.layers.iter().enumerate() {
            for (name, p) in [
                ("q_proj", &layer.attn.q_proj),
                ("k_proj", &layer.attn.k_proj),
                ("v_proj", &layer.attn.v_proj),
                ("o_proj", &layer.attn.o_proj),
                ("gate_proj", &layer.mlp.gate),
                ("up_proj", &layer.mlp.up),
                ("down_proj", &layer.mlp.down),
            ] {
                assert!(
                    p.inner.is_packed(),
                    "layer {i} {name} must load packed from its `.scales` sibling"
                );
                assert_eq!(
                    p.inner.matmul_strategy(),
                    Some(candle_gen::quant::MatmulStrategy::DequantDense),
                    "sc-7702: layer {i} {name} must dequantize the WEIGHT, never quantize the \
                     activation"
                );
            }
            // The norms are dense in every tier, so they are plain tensors, not projections.
            assert_eq!(layer.attn.q_norm.dtype(), DType::F32);
            assert_eq!(layer.input_ln.dtype(), DType::F32);
        }

        // The packed tower is materially smaller than the dense one it decodes to.
        let (dense_w, _) = packable_weights(&cfg, false);
        let dense = MiniMaxH3TextEncoder::from_weights(&dense_w, "lm", &cfg, DType::F32).unwrap();
        assert!(
            te.nbytes() < dense.nbytes(),
            "packed {} must be smaller than dense {}",
            te.nbytes(),
            dense.nbytes()
        );
    }

    /// **Numerics.** The packed tower reproduces a dense tower built from the *same dequantized
    /// grids*, on relative max-abs — never cosine, which is scale-invariant and therefore blind to a
    /// mis-decoded group scale (the defect class the packed path can produce).
    #[test]
    fn the_packed_te_forward_matches_its_dense_grid() {
        let cfg = packable_cfg();
        let (packed_w, grids) = packable_weights(&cfg, true);

        // The dense reference: the packed fixture with each packed triple replaced by the affine
        // grid it decodes to.
        let mut dense_map: HashMap<String, Tensor> = HashMap::new();
        for k in packed_w.keys() {
            if k.ends_with(".scales") || k.ends_with(".biases") {
                continue;
            }
            let t = match grids.get(k) {
                Some(grid) => grid.clone(),
                None => packed_w.require(k).unwrap(),
            };
            dense_map.insert(k.clone(), t);
        }
        let dense_w = Weights::from_map(dense_map);

        let packed = MiniMaxH3TextEncoder::from_weights(&packed_w, "lm", &cfg, DType::F32).unwrap();
        let dense = MiniMaxH3TextEncoder::from_weights(&dense_w, "lm", &cfg, DType::F32).unwrap();
        assert!(packed.embed_tokens.embedding.is_quantized());
        assert!(!dense.embed_tokens.embedding.is_quantized());

        let tokens = ids(&[1, 4, 9, 2, 11]);
        let mask = ones_mask(5);
        let got = packed.forward(&tokens, &mask).unwrap();
        let want = dense.forward(&tokens, &mask).unwrap();
        let drift = crate::quant::testkit::rel_max_abs(&got, &want);
        println!("[te] packed vs dense-grid rel-max-abs = {drift:.3e}");
        assert!(
            drift < 5e-3,
            "the Q4_1 repack is lossless up to the f16 scale/bias cast; got {drift:.3e}"
        );
    }

    /// **The guard.** A packed triple under a dense-by-policy TE key is refused loudly, naming the
    /// key — the sc-14980 class. Asserted on each of the four dense-by-policy suffixes this crate
    /// reads, individually, so a guard dropped from any one of them is caught.
    #[test]
    fn a_packed_tensor_under_a_dense_by_policy_te_key_is_refused() {
        let cfg = packable_cfg();
        for dense_key in [
            "lm.layers.0.self_attn.q_norm",
            "lm.layers.0.self_attn.k_norm",
            "lm.layers.0.input_layernorm",
            "lm.layers.0.post_attention_layernorm",
        ] {
            let (w, _) = packable_weights(&cfg, false);
            let mut map: HashMap<String, Tensor> = w
                .keys()
                .map(|k| (k.clone(), w.require(k).unwrap()))
                .collect();
            // Pack a tensor the policy says is dense in every tier.
            crate::quant::testkit::insert_packed(&mut map, dense_key, 64, 64, 3);
            let msg = match MiniMaxH3TextEncoder::from_weights(
                &Weights::from_map(map),
                "lm",
                &cfg,
                DType::F32,
            ) {
                Ok(_) => panic!("a packed `{dense_key}` must be refused, not loaded"),
                Err(e) => e.to_string(),
            };
            assert!(msg.contains(dense_key), "{msg}");
            assert!(msg.contains("MLX-PACKED"), "{msg}");
            // Attributed to the TEXT ENCODER, citing its policy list — not the DiT's. The two
            // components share `guard_dense`, so this is the assertion that keeps a TE refusal from
            // pointing the reader at `DENSE_BY_POLICY`.
            assert!(msg.contains(&format!("minimax-h3 te {dense_key}")), "{msg}");
            assert!(msg.contains("TE_DENSE_BY_POLICY"), "{msg}");
        }
    }

    /// A packed token table whose `.weight` is not a u32 code stream is a typed error, not a silent
    /// repack of whatever floats happened to be there.
    #[test]
    fn a_packed_marker_over_a_float_token_table_is_refused() {
        let cfg = packable_cfg();
        let (w, _) = packable_weights(&cfg, true);
        let mut map: HashMap<String, Tensor> = w
            .keys()
            .map(|k| (k.clone(), w.require(k).unwrap()))
            .collect();
        map.insert(
            "lm.embed_tokens.weight".into(),
            Tensor::zeros((cfg.vocab_size, 8), DType::F32, &Device::Cpu).unwrap(),
        );
        let msg = match MiniMaxH3TextEncoder::from_weights(
            &Weights::from_map(map),
            "lm",
            &cfg,
            DType::F32,
        ) {
            Ok(_) => panic!("a float table under a packed marker must be refused"),
            Err(e) => e.to_string(),
        };
        assert!(msg.contains("rather than U32"), "{msg}");
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

    const IMG: u32 = 151_655;
    const VID: u32 = 151_656;

    /// **A run never spans two different pads.** An `<|image_pad|>` block immediately followed by a
    /// `<|video_pad|>` block is two references, and merging them would splice one reference's
    /// embeds across both — which is shape-legal exactly when their token counts happen to sum.
    #[test]
    fn a_vision_run_never_spans_two_different_pads() {
        let ids = [9u32, IMG, IMG, VID, VID, VID, 9, VID, 9];
        assert_eq!(
            vision_pad_runs(&ids, &[IMG, VID]),
            vec![(1, 2), (3, 3), (7, 1)],
            "adjacent image and video pads are separate runs"
        );
        // The single-pad scan sees only its own pad — this is what keeps `fl2va` from consuming a
        // video block as a keyframe.
        assert_eq!(vision_pad_runs(&ids, &[IMG]), vec![(1, 2)]);
        assert!(vision_pad_runs(&[1, 2, 3], &[IMG, VID]).is_empty());
    }

    /// The multi-pad scans reduce **exactly** to the shared single-pad helpers on a presentation
    /// that carries only `<|image_pad|>`, so `fl2va` is unchanged by the `ref2va` generalization.
    ///
    /// Asserted against `candle_gen::grounding`'s own functions rather than against a transcribed
    /// expectation, so this cannot pass by agreeing with a copy of itself.
    #[test]
    fn the_multi_pad_scan_reduces_to_the_shared_single_pad_one() {
        use candle_gen::grounding::{image_blocks, mrope_positions};
        // Two image blocks whose merged grids are 2x2 and 1x2 tokens.
        let grids = [[1, 4, 4], [1, 2, 4]];
        let ids = [7u32, IMG, IMG, IMG, IMG, 7, 7, IMG, IMG, 7];
        assert_eq!(vision_pad_runs(&ids, &[IMG]), image_blocks(&ids, IMG));
        assert_eq!(
            mrope_positions_multi(&ids, &[IMG], &grids),
            mrope_positions(&ids, IMG, &grids)
        );
        // …and the shared helper genuinely produced non-trivial positions, so the equality above is
        // not two empty vectors agreeing.
        let (pt, ph, pw) = mrope_positions_multi(&ids, &[IMG], &grids);
        assert_eq!(pt.len(), ids.len());
        assert_ne!(ph, pt, "an image block must move the H axis off the T ramp");
        assert_ne!(pw, pt);
    }

    /// A `ref2va` presentation's video blocks advance the same rotary clock the image blocks do,
    /// and each block consumes exactly one `grids` entry in **sequence** order.
    #[test]
    fn video_pad_blocks_consume_grids_in_sequence_order() {
        // image block (2x2 = 4 tokens), then a video block (1x2 = 2 tokens).
        let grids = [[1, 4, 4], [1, 2, 4]];
        let ids = [7u32, IMG, IMG, IMG, IMG, VID, VID, 7];
        let (pt, ph, pw) = mrope_positions_multi(&ids, &[IMG, VID], &grids);
        assert_eq!(pt.len(), ids.len());
        // token 0 is text at 0; the image block occupies cur = 1 with a 2x2 merged grid and
        // advances the clock by max(4,4)/2 = 2 to cur = 3; the video block's 1x2 merged grid sits
        // there and advances by max(2,4)/2 = 2 to cur = 5, where the trailing text token lands.
        assert_eq!(pt, vec![0, 1, 1, 1, 1, 3, 3, 5]);
        assert_eq!(ph, vec![0, 1, 1, 2, 2, 3, 3, 5]);
        assert_eq!(pw, vec![0, 1, 2, 1, 2, 3, 4, 5]);

        // Scanning with the image pad alone leaves the video pads as TEXT rows — the concrete
        // reason `forward_with_references` exists rather than reusing the `fl2va` path.
        let (image_only, _, _) = mrope_positions_multi(&ids, &[IMG], &grids);
        assert_ne!(image_only, pt);
    }
}
