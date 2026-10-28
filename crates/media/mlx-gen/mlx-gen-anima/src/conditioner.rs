//! `AnimaTextConditioner` — transcribed from diffusers
//! `condition_embedder_anima.py::AnimaTextConditioner`. Maps T5 token ids (learned query tokens via
//! `nn.Embedding(32128, 1024)`) + Qwen3 hidden states → the DiT's `encoder_hidden_states`. 6 blocks of
//! `[self-attn (RoPE θ=10000) → cross-attn into Qwen3 states → GELU MLP]`, then `out_proj` + RMSNorm,
//! mask-multiply, and **right-pad to 512** so the DiT always sees exactly 512 text tokens.
//!
//! Weight keys are the `net.llm_adapter.`-**stripped** names (`blocks.N.*`, `embed.weight`,
//! `out_proj.*`, `norm.weight`) — identical to the diffusers `AnimaTextConditioner` state dict, which
//! the Anima convert script loads with `strict=True` (no rename).

use mlx_rs::error::Result as MlxResult;
use mlx_rs::fast::{rms_norm, scaled_dot_product_attention};
use mlx_rs::ops::{add, concatenate_axis, multiply, zeros_dtype};
use mlx_rs::transforms::checkpoint;
use mlx_rs::{Array, Dtype};

use mlx_gen::adapters::{AdaptableHost, AdaptableLinear, LinearFacts};
use mlx_gen::nn::{gelu_exact, TextRope, TokenEmbedding};
use mlx_gen::qkv::{self, AttnPrepSpec, QkNormSpec, QkvPart, RopeSpec, RopeStyle, RopeTables};

use crate::transformer::Qkv;
use mlx_gen::weights::{join, Weights};
use mlx_gen::Result;

use crate::config::ConditionerConfig;

/// iOS execution-quantum labels (see [`mlx_gen::ios_quantum`]).
pub(crate) const CONDITIONER_EMBED_QUANTUM: &str = "Anima conditioner embed";
pub(crate) const CONDITIONER_BLOCK_QUANTUM: &str = "Anima conditioner block";
pub(crate) const CONDITIONER_OUTPUT_QUANTUM: &str = "Anima conditioner output";

/// Counted live buffers of one conditioner block quantum; see `Block::quantum_geometry`.
pub(crate) const COND_ATTN_HIDDEN_LIVE: u64 = 25;
pub(crate) const COND_CONTEXT_LIVE: u64 = 5;
pub(crate) const COND_MLP_HIDDEN_LIVE: u64 = 6;
pub(crate) const COND_FF_LIVE: u64 = 3;

fn lin(w: &Weights, name: &str, bias: bool) -> Result<AdaptableLinear> {
    mlx_gen::quant::lin(w, name, bias, mlx_gen::quant::DEFAULT_GROUP_SIZE)
}

/// `(cos, sin)` RoPE tables, one pair per sequence length.
type Rope<'a> = (&'a Array, &'a Array);

/// `AnimaTextConditionerAttention`: q/k/v/o projections (no bias), per-head q/k RMSNorm (eps 1e-6),
/// half-split RoPE. Query positions and key positions can differ (cross-attn: target vs source).
struct Attention {
    /// SC-18319 P4 — see [`crate::transformer::Qkv`]. `self_attn` is packed; `cross_attn` is not,
    /// and cannot be: its `k`/`v` read the Qwen3 encoder stream while `q` reads the T5 target
    /// stream. Both are 1024 wide here, so `try_pack` would happily accept the triple — the refusal
    /// has to be the type, which is what `Qkv::Cross` is.
    qkv: Qkv,
    o_proj: AdaptableLinear,
    q_norm: Array,
    k_norm: Array,
    heads: i32,
    head_dim: i32,
    scale: f32,
    eps: f32,
    /// sc-10576 — SDPA-segment gradient checkpointing (see the DiT `Attention::ckpt_sdpa`). The
    /// conditioner is never whole-block checkpointed (it runs in the traced graph so its 60 adapter
    /// factors get gradients), so — like z-image's refiner leg — its attention segment ckpt is left ON
    /// during training to bound the retained `[heads, S, S]` probability matrix. Numerically identical
    /// to the retained backward; inference never sets it.
    ckpt_sdpa: bool,
}

impl Attention {
    fn from_weights(
        w: &Weights,
        prefix: &str,
        cfg: &ConditionerConfig,
        self_attention: bool,
    ) -> Result<Self> {
        let hd = cfg.head_dim() as i32;
        let (q, k, v) = (
            lin(w, &join(prefix, "q_proj"), false)?,
            lin(w, &join(prefix, "k_proj"), false)?,
            lin(w, &join(prefix, "v_proj"), false)?,
        );
        Ok(Self {
            qkv: if self_attention {
                Qkv::self_attn(q, k, v)
            } else {
                Qkv::cross_attn(q, k, v)
            },
            o_proj: lin(w, &join(prefix, "o_proj"), false)?,
            q_norm: w.require(&join(prefix, "q_norm.weight"))?.clone(),
            k_norm: w.require(&join(prefix, "k_norm.weight"))?.clone(),
            heads: cfg.num_attention_heads as i32,
            head_dim: hd,
            scale: (hd as f32).powf(-0.5),
            eps: cfg.norm_eps,
            ckpt_sdpa: false,
        })
    }

    fn set_sdpa_checkpoint(&mut self, on: bool) {
        self.ckpt_sdpa = on;
    }

    /// `hidden`: `[B, Sq, D]` (query source). `encoder`: `Some([B, Sk, Ctx])` (cross) or `None`
    /// (self → `hidden`). `q_rope`/`k_rope` apply half-split RoPE with (possibly) different positions.
    /// Batch is 1 in this pipeline, so all tokens are real ⇒ no attention mask needed.
    fn forward(
        &self,
        hidden: &Array,
        encoder: Option<&Array>,
        q_rope: Rope,
        k_rope: Rope,
    ) -> Result<Array> {
        let kv_src = encoder.unwrap_or(hidden);
        // SC-18319 P4 — see the DiT twin: a fused (self) representation projects from ONE stream.
        if self.qkv.is_self() && encoder.is_some() {
            return Err(mlx_gen::Error::Msg(
                "anima conditioner: a self-attention (fused q/k/v) was given an encoder stream"
                    .into(),
            ));
        }

        // SC-18319 — the shared prologue. The conditioner is **knob 6's reference case**: `q_rope`
        // and `k_rope` are genuinely different position tables (cross-attention positions the query
        // on the hidden stream and the key on the encoder stream), which is why the primitive takes
        // per-stream tables rather than one shared table plus a flag. The rest: separate q/k/v
        // (knob 9), per-head RMSNorm over `head_dim` after the head split (knob 1), half-split
        // `rotate_half` rotation over full-width `[1, S, head_dim]` tables (knob 2 — `apply_text_rope`
        // verbatim), token-major axes, single stream (knob 11 does not apply).
        let spec = AttnPrepSpec::new(self.heads, self.head_dim)
            .with_qk_norm(QkNormSpec::per_head(&self.q_norm, &self.k_norm, self.eps))
            .with_rope(RopeSpec {
                style: RopeStyle::RotateHalf,
                q: Some(RopeTables::new(q_rope.0, q_rope.1)),
                k: Some(RopeTables::new(k_rope.0, k_rope.1)),
                ..RopeSpec::default()
            });
        let projected = self.qkv.project(hidden, kv_src)?;
        let heads = qkv::prepare(projected.source(), &spec)?;
        let (q, k, v) = (heads.q, heads.k, heads.v);
        let o = if self.ckpt_sdpa {
            // sc-10576: checkpoint the SDPA (q/k/v threaded, scale captured) — recompute the seq²
            // attention in the backward instead of retaining it. Numerically identical.
            let scale = self.scale;
            let mut seg = checkpoint(move |inp: &[Array]| -> MlxResult<Vec<Array>> {
                Ok(vec![scaled_dot_product_attention(
                    &inp[0], &inp[1], &inp[2], scale, None, None,
                )?])
            });
            seg(&[q, k, v])?.into_iter().next().ok_or_else(|| {
                mlx_gen::Error::Msg("anima conditioner: checkpoint SDPA produced no output".into())
            })?
        } else {
            scaled_dot_product_attention(&q, &k, &v, self.scale, None, None)?
        };
        self.o_proj.forward(&qkv::merge_heads(&o)?)
    }
}

/// `AnimaTextConditionerBlock`: pre-RMSNorm self-attn → cross-attn → GELU MLP (all residual).
pub(crate) struct Block {
    norm_self_attn: Array,
    self_attn: Attention,
    norm_cross_attn: Array,
    cross_attn: Attention,
    norm_mlp: Array,
    mlp_in: AdaptableLinear,  // mlp.0 (bias)
    mlp_out: AdaptableLinear, // mlp.2 (bias)
    eps: f32,
}

impl Block {
    fn from_weights(w: &Weights, prefix: &str, cfg: &ConditionerConfig) -> Result<Self> {
        Ok(Self {
            norm_self_attn: w.require(&join(prefix, "norm_self_attn.weight"))?.clone(),
            self_attn: Attention::from_weights(w, &join(prefix, "self_attn"), cfg, true)?,
            norm_cross_attn: w.require(&join(prefix, "norm_cross_attn.weight"))?.clone(),
            cross_attn: Attention::from_weights(w, &join(prefix, "cross_attn"), cfg, false)?,
            norm_mlp: w.require(&join(prefix, "norm_mlp.weight"))?.clone(),
            mlp_in: lin(w, &join(prefix, "mlp.0"), true)?,
            mlp_out: lin(w, &join(prefix, "mlp.2"), true)?,
            eps: cfg.norm_eps,
        })
    }

    fn set_sdpa_checkpoint(&mut self, on: bool) {
        self.self_attn.set_sdpa_checkpoint(on);
        self.cross_attn.set_sdpa_checkpoint(on);
    }

    /// The geometry of one forward of THIS block from its loaded tensors: `St` target (T5 query)
    /// tokens attend over themselves and cross-attend over `Ss` source (Qwen3) tokens; the MLP
    /// runs over the `St` tokens. Both attentions call `mlx_rs::fast::scaled_dot_product_attention`
    /// unchunked; their score treatment on `backend` is [`crate::transformer::sdpa_scores`] of the
    /// loaded head dim (64 in the shipped conditioner, which the fused Metal kernel serves at any
    /// `St` above 8), the width and the token counts.
    ///
    /// The width is f32: the f32 RoPE tables promote the rotated q/k and everything after them
    /// (and in production the Qwen3 states arrive f32 already).
    ///
    /// Weight copies ([`Self::promoted_weight_bytes`]): every dense bf16 projection fed the f32
    /// stream is promoted to an f32 copy for its matmul; all of the block's are priced (measured,
    /// 38 of the 64 f32-copy units at the proof's shape are live at once).
    ///
    /// Counted live buffers (MLX holds an op's buffers until its stream catches up; elementwise
    /// outputs it donates into a consumed input are not counted; an RMSNorm is counted with the
    /// temporaries of MLX's composed non-Metal fallback):
    ///
    /// * attention phase, `[B, St, dim]` ([`COND_ATTN_HIDDEN_LIVE`] = 25): the self-attention — its
    ///   RMSNorm (3), q/k/v (3), the per-head q/k RMSNorms (2), the RoPE rotate-half and product
    ///   for q and for k (4), the scaled q, the output, its merge, `o_proj` and the residual sum
    ///   (5) = 17 — and, live with its tail, the cross-attention's RMSNorm (3), q, its norm, its
    ///   rotation (2) and the scaled q = 8;
    /// * cross-attention context, `[B, Ss, dim]` ([`COND_CONTEXT_LIVE`] = 5): k, v, the k RMSNorm
    ///   and the k rotation (2);
    /// * MLP phase: the RMSNorm (3), `mlp.2` and its bias sum, the residual sum = 6 hidden
    ///   ([`COND_MLP_HIDDEN_LIVE`]); `mlp.0`, its bias sum and the compiled exact GELU (3
    ///   `[B, St, ff]`, [`COND_FF_LIVE`]).
    ///
    pub(crate) fn quantum_geometry(
        &self,
        batch: i32,
        target_tokens: i32,
        source_tokens: i32,
        element_bytes: u64,
        backend: crate::transformer::Backend,
    ) -> mlx_gen::ios_quantum::AttentionBlockGeometry {
        let tokens = target_tokens.max(0) as u64;
        let source = source_tokens.max(0) as u64;
        let scores = crate::transformer::sdpa_scores(
            backend,
            self.self_attn.head_dim.max(0) as u64,
            element_bytes,
            &[(tokens, tokens, tokens), (tokens, tokens, source)],
        );
        mlx_gen::ios_quantum::AttentionBlockGeometry {
            batch: batch.max(0) as u64,
            tokens,
            dim: self.norm_self_attn.size() as u64,
            heads: self.self_attn.heads.max(0) as u64,
            context_tokens: source,
            ff_hidden: self.mlp_in.base_shape()[0].max(0) as u64,
            query_rows: tokens,
            element_bytes,
            weight_bytes: self.promoted_weight_bytes(element_bytes),
            attn_hidden_live: COND_ATTN_HIDDEN_LIVE,
            context_live: COND_CONTEXT_LIVE,
            scores,
            mlp_hidden_live: COND_MLP_HIDDEN_LIVE,
            ff_live: COND_FF_LIVE,
        }
    }

    /// The f32 copies a forward makes of this block's dense projection weights when the stream is
    /// wider than their storage (bf16 weights, f32 stream): every q/k/v/o of both attentions and
    /// both MLP projections, with their biases. `0` for packed projections.
    pub(crate) fn promoted_weight_bytes(&self, element_bytes: u64) -> u64 {
        use crate::transformer::{dense_weight_elements, dtype_bytes};
        let narrower = self
            .self_attn
            .o_proj
            .weight_dtype()
            .is_some_and(|dtype| dtype_bytes(dtype) < element_bytes);
        if !narrower {
            return 0;
        }
        let attention: u64 = [&self.self_attn, &self.cross_attn]
            .iter()
            .map(|attn| {
                [QkvPart::Q, QkvPart::K, QkvPart::V]
                    .into_iter()
                    .map(|part| dense_weight_elements(&attn.qkv.part_facts(part)))
                    .sum::<u64>()
                    + dense_weight_elements(&LinearFacts::of(&attn.o_proj))
            })
            .sum();
        let mlp = dense_weight_elements(&LinearFacts::of(&self.mlp_in))
            + dense_weight_elements(&LinearFacts::of(&self.mlp_out));
        attention.saturating_add(mlp).saturating_mul(element_bytes)
    }

    pub(crate) fn quantum_scratch(
        &self,
        batch: i32,
        target_tokens: i32,
        source_tokens: i32,
        element_bytes: u64,
        backend: crate::transformer::Backend,
    ) -> u64 {
        mlx_gen::ios_quantum::attention_block_bytes(self.quantum_geometry(
            batch,
            target_tokens,
            source_tokens,
            element_bytes,
            backend,
        ))
    }

    pub(crate) fn forward(
        &self,
        hidden: &Array,
        encoder: &Array,
        target_rope: Rope,
        source_rope: Rope,
    ) -> Result<Array> {
        // self-attn: q and k both use the target (T5 query-token) positions.
        let normed = rms_norm(hidden, &self.norm_self_attn, self.eps)?;
        let h = add(
            hidden,
            &self
                .self_attn
                .forward(&normed, None, target_rope, target_rope)?,
        )?;
        // cross-attn: q uses target positions, k uses source (Qwen3) positions.
        let normed = rms_norm(&h, &self.norm_cross_attn, self.eps)?;
        let h = add(
            &h,
            &self
                .cross_attn
                .forward(&normed, Some(encoder), target_rope, source_rope)?,
        )?;
        // GELU MLP.
        let normed = rms_norm(&h, &self.norm_mlp, self.eps)?;
        let mlp = self
            .mlp_out
            .forward(&gelu_exact(&self.mlp_in.forward(&normed)?)?)?;
        Ok(add(&h, &mlp)?)
    }
}

/// The full `AnimaTextConditioner`.
pub struct AnimaTextConditioner {
    embed: TokenEmbedding,
    blocks: Vec<Block>,
    out_proj: AdaptableLinear, // bias
    norm: Array,
    rope: TextRope,
    cfg: ConditionerConfig,
}

impl AnimaTextConditioner {
    /// `prefix` is the checkpoint root of the adapter (`"net.llm_adapter"`).
    pub fn from_weights(w: &Weights, prefix: &str, cfg: ConditionerConfig) -> Result<Self> {
        let embed = mlx_gen::quant::embedding(
            w,
            &join(prefix, "embed"),
            mlx_gen::quant::DEFAULT_GROUP_SIZE,
        )?;
        let mut blocks = Vec::with_capacity(cfg.num_layers);
        for i in 0..cfg.num_layers {
            blocks.push(Block::from_weights(
                w,
                &join(prefix, &format!("blocks.{i}")),
                &cfg,
            )?);
        }
        Ok(Self {
            embed,
            blocks,
            out_proj: lin(w, &join(prefix, "out_proj"), true)?,
            norm: w.require(&join(prefix, "norm.weight"))?.clone(),
            rope: TextRope::new(cfg.head_dim() as i32, cfg.rope_theta),
            cfg,
        })
    }

    /// `source_hidden`: `[B, Ss, source_dim]` Qwen3 states (already mask-multiplied), in `dtype`.
    /// `target_ids`: `[B, St]` int32 T5 token ids (`St <= min_sequence_length`). Returns
    /// `[B, min_sequence_length(=512), target_dim]` in `dtype`.
    pub fn forward(
        &self,
        source_hidden: &Array,
        target_ids: &Array,
        dtype: Dtype,
    ) -> Result<Array> {
        self.forward_weighted(source_hidden, target_ids, None, dtype)
    }

    /// [`forward`](Self::forward) with **ComfyUI-style prompt weighting** applied to the T5
    /// query-token path (sc-10566). `target_weights`, when present, is a per-target-token weight
    /// vector aligned 1:1 with `target_ids` (the real, unpadded T5 sequence): each token's full
    /// `target_dim` OUTPUT vector is scaled by its scalar weight **before** the right-pad to 512 —
    /// exactly ComfyUI's `out = self.llm_adapter(...); out = out * t5xxl_weights`
    /// (`comfy/ldm/anima/model.py:198-206`, with `t5xxl_weights` reshaped to `[1, St, 1]` in
    /// `comfy/model_base.py:1470`). The Qwen source path is untouched (its weights are pinned to
    /// `1.0` upstream), and an all-`1.0` weight vector is a strict no-op (identical to
    /// [`Self::forward`]).
    /// Allowance for the lazy Qwen3 mask-multiply feeding `source_hidden`: the states, the cast
    /// mask broadcast and the product (three source-sized f32 buffers).
    pub(crate) fn mask_multiply_bytes(source_hidden: &Array) -> u64 {
        Self::mask_multiply_bytes_for(source_hidden.size() as u64)
    }

    /// [`Self::mask_multiply_bytes`] over `source_elements` Qwen3 state elements.
    pub(crate) fn mask_multiply_bytes_for(source_elements: u64) -> u64 {
        mlx_gen::ios_quantum::elementwise_bytes(source_elements, 3)
    }

    pub fn forward_weighted(
        &self,
        source_hidden: &Array,
        target_ids: &Array,
        target_weights: Option<&[f32]>,
        dtype: Dtype,
    ) -> Result<Array> {
        let ssh = source_hidden.shape();
        let (b, ss) = (ssh[0], ssh[1]);
        let st = target_ids.shape()[1];
        let eb = crate::transformer::dtype_bytes(dtype);
        let _quantum_scope = mlx_gen::ios_quantum::Scope::try_new()?;
        // The entry quantum: its own counted body plus the caller's lazy Qwen3 mask-multiply
        // (states, cast mask, product), which is materialized in this quantum.
        mlx_gen::ios_quantum::before(
            CONDITIONER_EMBED_QUANTUM,
            self.embed_body_bytes(b, st, ss, eb)
                .saturating_add(Self::mask_multiply_bytes(source_hidden)),
        )?;
        let (mut hidden, (t_cos, t_sin), (s_cos, s_sin)) = self.embed(target_ids, ss, dtype)?;
        mlx_gen::ios_quantum::finish(
            CONDITIONER_EMBED_QUANTUM,
            &[&hidden, &t_cos, &t_sin, &s_cos, &s_sin, source_hidden],
        )?;
        let target_rope = (&t_cos, &t_sin);
        let source_rope = (&s_cos, &s_sin);

        for block in &self.blocks {
            // The f32 RoPE tables promote the rotated q/k and everything after them to f32.
            let block_eb = eb.max(t_cos.item_size() as u64);
            hidden = mlx_gen::ios_quantum::run(
                CONDITIONER_BLOCK_QUANTUM,
                block.quantum_scratch(b, st, ss, block_eb, crate::transformer::Backend::current()),
                || block.forward(&hidden, source_hidden, target_rope, source_rope),
            )?;
        }
        mlx_gen::ios_quantum::run(
            CONDITIONER_OUTPUT_QUANTUM,
            self.output_quantum_scratch(b, st, hidden.item_size() as u64),
            || self.output(&hidden, b, st, target_weights, dtype),
        )
    }

    /// The entry quantum's body: the learned T5 query tokens (`in_proj` is Identity: model_dim ==
    /// target_dim) and the target/source RoPE tables. Returns `(hidden, target_rope, source_rope)`.
    #[allow(clippy::type_complexity)]
    pub(crate) fn embed(
        &self,
        target_ids: &Array,
        source_tokens: i32,
        dtype: Dtype,
    ) -> Result<(Array, (Array, Array), (Array, Array))> {
        let hidden = self.embed.forward(target_ids)?.as_dtype(dtype)?; // [B, St, D]
        let target = self.rope.forward(target_ids.shape()[1])?; // target positions
        let source = self.rope.forward(source_tokens)?; // source positions
        Ok((hidden, target, source))
    }

    /// The entry quantum's counted body ([`Self::embed`]): the gathered query rows (priced at f32,
    /// the width a packed embedding dequantizes to), `TokenEmbedding`'s f32 cast and, when the
    /// compute dtype is narrower, the cast to it (`[B, St, model_dim]` each), plus the shared
    /// [`mlx_gen::ios_quantum::rope_table_bytes`] allowance for both position tables.
    /// Proven in `quantum_tests::conditioner_formulas_track_the_measured_peak`.
    pub(crate) fn embed_body_bytes(
        &self,
        batch: i32,
        target_tokens: i32,
        source_tokens: i32,
        element_bytes: u64,
    ) -> u64 {
        let (b, st) = (batch.max(0) as u64, target_tokens.max(0) as u64);
        let d = self.cfg.model_dim as u64;
        let hd = self.cfg.head_dim() as u64;
        let tb = mlx_gen::ios_quantum::tensor_bytes;
        // The compute-dtype cast allocates only when it narrows the f32 rows.
        let cast = if element_bytes < 4 {
            tb(element_bytes, &[b, st, d])
        } else {
            0
        };
        (2 * tb(4, &[b, st, d]))
            .saturating_add(cast)
            .saturating_add(mlx_gen::ios_quantum::rope_table_bytes(st, hd))
            .saturating_add(mlx_gen::ios_quantum::rope_table_bytes(
                source_tokens.max(0) as u64,
                hd,
            ))
    }

    /// The output quantum's counted body ([`Self::output`]) over the blocks' `element_bytes`-wide
    /// output: `out_proj`, the RMSNorm's square and output (3 `[B, St, target_dim]`; the bias sum
    /// and the prompt-weight product are donated), the zero padding (`[B, 512 - St, ..]`), the
    /// padded concat (`[B, 512, ..]`), three f32 per-row vectors (`[B, St]`: the prompt-weight
    /// vector and the RMSNorm's mean of squares and its reciprocal root — measured, they are live
    /// with the rest at the peak), and the f32 copies of `out_proj`'s weight when it is dense and
    /// of the norm weight, each when narrower than the stream. Proven in
    /// `quantum_tests::conditioner_formulas_track_the_measured_peak`.
    pub(crate) fn output_quantum_scratch(
        &self,
        batch: i32,
        target_tokens: i32,
        element_bytes: u64,
    ) -> u64 {
        let (b, st) = (batch.max(0) as u64, target_tokens.max(0) as u64);
        let min = self.cfg.min_sequence_length as u64;
        let d = self.cfg.target_dim as u64;
        let tb = mlx_gen::ios_quantum::tensor_bytes;
        let promoted = if self
            .out_proj
            .weight_dtype()
            .is_some_and(|dtype| crate::transformer::dtype_bytes(dtype) < element_bytes)
        {
            crate::transformer::dense_weight_elements(&LinearFacts::of(&self.out_proj))
                .saturating_mul(element_bytes)
        } else {
            0
        };
        let norm_promoted = if crate::transformer::dtype_bytes(self.norm.dtype()) < element_bytes {
            (self.norm.size() as u64).saturating_mul(element_bytes)
        } else {
            0
        };
        (3 * tb(element_bytes, &[b, st, d]))
            .saturating_add(3 * tb(4, &[b, st]))
            .saturating_add(norm_promoted)
            .saturating_add(tb(element_bytes, &[b, min.saturating_sub(st), d]))
            .saturating_add(tb(element_bytes, &[b, min.max(st), d]))
            .saturating_add(promoted)
    }

    /// `out_proj` + RMSNorm, the optional T5 prompt weighting, and the right-pad to
    /// `min_sequence_length` — the conditioner's post-block quantum.
    pub(crate) fn output(
        &self,
        hidden: &Array,
        b: i32,
        st: i32,
        target_weights: Option<&[f32]>,
        dtype: Dtype,
    ) -> Result<Array> {
        let mut hidden = rms_norm(
            &self.out_proj.forward(hidden)?,
            &self.norm,
            self.cfg.norm_eps,
        )?;

        // Prompt weighting: scale each T5 query-token OUTPUT vector by its weight (T5 path only).
        // Skip entirely when every weight is 1.0 so the unweighted path stays bit-identical.
        if let Some(w) = target_weights {
            if w.iter().any(|&x| x != 1.0) {
                // Align the weight vector to St, padding/truncating with 1.0 (fail gracefully on a
                // length mismatch rather than panic on a shape error).
                let st_usize = st as usize;
                let mut wv = vec![1.0f32; st_usize];
                let take = w.len().min(st_usize);
                wv[..take].copy_from_slice(&w[..take]);
                let warr = Array::from_slice(&wv, &[1, st, 1]).as_dtype(dtype)?; // [1, St, 1]
                hidden = multiply(&hidden, &warr)?; // broadcast over B and target_dim
            }
        }

        // Right-pad to min_sequence_length so the DiT always sees exactly 512 text tokens.
        let min = self.cfg.min_sequence_length as i32;
        if st < min {
            let pad = zeros_dtype(&[b, min - st, self.cfg.target_dim as i32], dtype)?;
            Ok(concatenate_axis(&[&hidden, &pad], 1)?)
        } else {
            Ok(hidden)
        }
    }

    /// The blocks (the CPU proofs run one block's quantum body directly; the iOS pricing prices
    /// each block's quantum).
    pub(crate) fn blocks(&self) -> &[Block] {
        &self.blocks
    }

    pub fn config(&self) -> &ConditionerConfig {
        &self.cfg
    }

    /// Toggle SDPA-segment gradient checkpointing (sc-10576) on every conditioner block. Training-only:
    /// the trainer leaves this ON (the conditioner is never whole-block checkpointed), matching the
    /// z-image refiner leg. Numerically identical to the retained backward; inference never calls it.
    pub fn set_sdpa_checkpoint(&mut self, on: bool) {
        for b in &mut self.blocks {
            b.set_sdpa_checkpoint(on);
        }
    }
}

// ---- LoRA/LoKr adapter surface (sc-10521) --------------------------------------------------------
//
// The **turbo** LoRA (`anima-turbo-lora-v0.2`) addresses 60 = 6×10 `llm_adapter.*` targets that a
// DiT-only injection walk would silently drop — the sc-10274 (eros distill LoRA) failure class. (Those
// 60 `lora_B` are zero-initialized in the turbo file, so `B·A ≡ 0` and dropping them is numerically
// inert for *it*; routing must still be enforced for future non-zero conditioner LoRAs and the shipped
// `anima-rl-v0.1`, which also carries these 60 targets.) The conditioner MUST therefore be a
// first-class injectable target. The trained files address it (after the loader strips the
// `diffusion_model.` prefix and the top-level host strips the `llm_adapter.` segment) by the
// conditioner's own module names: `blocks.N.{self_attn,cross_attn}.{q_proj,k_proj,v_proj,o_proj}` and
// `blocks.N.mlp.{0,2}` (no norm targets).

impl AdaptableHost for Attention {
    fn adaptable_mut(&mut self, path: &[&str]) -> Option<&mut AdaptableLinear> {
        match path {
            ["q_proj"] => self.qkv.part_mut(QkvPart::Q),
            ["k_proj"] => self.qkv.part_mut(QkvPart::K),
            ["v_proj"] => self.qkv.part_mut(QkvPart::V),
            ["o_proj"] => Some(&mut self.o_proj),
            _ => None,
        }
    }

    /// The PROBE half (SC-18319). See the DiT twin.
    fn adaptable_facts(&mut self, path: &[&str]) -> Option<LinearFacts> {
        match path {
            ["q_proj"] => Some(self.qkv.part_facts(QkvPart::Q)),
            ["k_proj"] => Some(self.qkv.part_facts(QkvPart::K)),
            ["v_proj"] => Some(self.qkv.part_facts(QkvPart::V)),
            _ => self.adaptable_mut(path).map(|l| LinearFacts::of(l)),
        }
    }

    fn adaptable_paths(&self) -> Vec<String> {
        ["q_proj", "k_proj", "v_proj", "o_proj"]
            .into_iter()
            .map(String::from)
            .collect()
    }
}

/// Actual conditioner adapter namespace, shared by live enumeration and header-only facts.
pub(crate) fn adapter_block_paths() -> Vec<String> {
    let mut out = Vec::new();
    for attention in ["self_attn", "cross_attn"] {
        for projection in ["q_proj", "k_proj", "v_proj", "o_proj"] {
            out.push(format!("{attention}.{projection}"));
        }
    }
    out.extend(["mlp.0".to_owned(), "mlp.2".to_owned()]);
    out
}
const OUTPUT_ADAPTER_TARGET: &str = "out_proj";
pub(crate) fn adapter_target_paths(blocks: usize, include_globals: bool) -> Vec<String> {
    let mut out = Vec::new();
    for index in 0..blocks {
        out.extend(
            adapter_block_paths()
                .into_iter()
                .map(|path| format!("blocks.{index}.{path}")),
        );
    }
    if include_globals {
        out.push(OUTPUT_ADAPTER_TARGET.into());
    }
    out
}

impl AdaptableHost for Block {
    fn adaptable_mut(&mut self, path: &[&str]) -> Option<&mut AdaptableLinear> {
        match path {
            ["self_attn", rest @ ..] => self.self_attn.adaptable_mut(rest),
            ["cross_attn", rest @ ..] => self.cross_attn.adaptable_mut(rest),
            ["mlp", "0"] => Some(&mut self.mlp_in),
            ["mlp", "2"] => Some(&mut self.mlp_out),
            _ => None,
        }
    }

    fn adaptable_paths(&self) -> Vec<String> {
        adapter_block_paths()
    }
}

/// The `AnimaTextConditioner` adapter host: `blocks.N.*` route into the per-block leaves (the 60 = 6×10
/// adapter targets the turbo LoRA carries); `out_proj` is routable too (the shipped LoRAs never target
/// it). Only block targets are enumerated for the kohya table.
impl AdaptableHost for AnimaTextConditioner {
    fn adaptable_mut(&mut self, path: &[&str]) -> Option<&mut AdaptableLinear> {
        match path {
            ["blocks", n, rest @ ..] => self
                .blocks
                .get_mut(n.parse::<usize>().ok()?)?
                .adaptable_mut(rest),
            [name] if *name == OUTPUT_ADAPTER_TARGET => Some(&mut self.out_proj),
            _ => None,
        }
    }

    fn adaptable_paths(&self) -> Vec<String> {
        adapter_target_paths(self.blocks.len(), false)
    }
}

// -------------------------------------------------------------------------------------------------
// Test-only structural constructor
// -------------------------------------------------------------------------------------------------

/// Build the conditioner module tree with **placeholder** (1×1) weights so the target-enumeration
/// guard (sc-10522) can exercise the real [`AdaptableHost::adaptable_paths`] surface without the
/// licensed checkpoint or any Metal compute — enumeration only walks the tree; nothing is evaluated.
#[cfg(all(test, feature = "training"))]
mod structural {
    use super::*;

    fn ph_lin() -> AdaptableLinear {
        AdaptableLinear::dense(Array::from_slice(&[0.0f32], &[1, 1]), None)
    }
    fn ph_norm() -> Array {
        Array::from_slice(&[1.0f32], &[1])
    }

    impl AnimaTextConditioner {
        /// A weight-free conditioner with `cfg.num_layers` structurally-complete blocks (6 for Anima),
        /// for the path-enumeration guard. Placeholder tensors only; nothing is evaluated.
        pub(crate) fn structural(cfg: ConditionerConfig) -> Self {
            let heads = cfg.num_attention_heads as i32;
            let head_dim = cfg.head_dim() as i32;
            let attn = |self_attention: bool| Attention {
                qkv: if self_attention {
                    Qkv::self_attn(ph_lin(), ph_lin(), ph_lin())
                } else {
                    Qkv::cross_attn(ph_lin(), ph_lin(), ph_lin())
                },
                o_proj: ph_lin(),
                q_norm: ph_norm(),
                k_norm: ph_norm(),
                heads,
                head_dim,
                scale: 1.0,
                eps: cfg.norm_eps,
                ckpt_sdpa: false,
            };
            let blocks = (0..cfg.num_layers)
                .map(|_| Block {
                    norm_self_attn: ph_norm(),
                    self_attn: attn(true),
                    norm_cross_attn: ph_norm(),
                    cross_attn: attn(false),
                    norm_mlp: ph_norm(),
                    mlp_in: ph_lin(),
                    mlp_out: ph_lin(),
                    eps: cfg.norm_eps,
                })
                .collect();
            Self {
                embed: TokenEmbedding::Dense(Array::from_slice(&[0.0f32], &[1, 1])),
                blocks,
                out_proj: ph_lin(),
                norm: ph_norm(),
                rope: TextRope::new(head_dim, cfg.rope_theta),
                cfg,
            }
        }
    }
}

// -------------------------------------------------------------------------------------------------
// Test-only SYNTHETIC constructor — real (small) random weights, evaluable on Metal (sc-10576)
// -------------------------------------------------------------------------------------------------

/// Real-weight conditioner for the grad-parity test (sc-10576): a dimension-parametric build with
/// small random weights so the 6-block conditioner forward+backward runs on Metal without the licensed
/// checkpoint. Mirrors [`CosmosDiT::synthetic`]. `cfg` sets every dim; use a tiny `cfg` for speed.
#[cfg(test)]
pub(crate) mod synthetic {
    use super::*;
    use mlx_rs::random;

    pub(crate) struct Rng(pub u64);

    impl Rng {
        fn lin(&mut self, out: i32, in_f: i32, bias: bool) -> AdaptableLinear {
            self.0 = self.0.wrapping_add(1);
            let key = random::key(self.0).unwrap();
            let w = random::normal::<f32>(&[out, in_f], None, None, Some(&key)).unwrap();
            let w = multiply(&w, Array::from_slice(&[0.05f32], &[1])).unwrap();
            let b = bias.then(|| Array::zeros::<f32>(&[out]).unwrap());
            AdaptableLinear::dense(w, b)
        }

        fn norm(&mut self, d: i32) -> Array {
            Array::ones::<f32>(&[d]).unwrap()
        }

        fn attn(
            &mut self,
            heads: i32,
            head_dim: i32,
            kv_in: i32,
            eps: f32,
            self_attention: bool,
        ) -> Attention {
            let d = heads * head_dim;
            let (q, k, v) = (
                self.lin(d, d, false),
                self.lin(d, kv_in, false),
                self.lin(d, kv_in, false),
            );
            Attention {
                qkv: if self_attention {
                    Qkv::self_attn(q, k, v)
                } else {
                    Qkv::cross_attn(q, k, v)
                },
                o_proj: self.lin(d, d, false),
                q_norm: self.norm(head_dim),
                k_norm: self.norm(head_dim),
                heads,
                head_dim,
                scale: (head_dim as f32).powf(-0.5),
                eps,
                ckpt_sdpa: false,
            }
        }
    }

    impl AnimaTextConditioner {
        /// A synthetic conditioner with real random weights. `cfg` sets every dim (`model_dim`,
        /// `source_dim` for the cross-attn kv, `target_vocab_size` for the T5 embedding).
        pub(crate) fn synthetic(cfg: ConditionerConfig, seed: u64) -> Self {
            let d = cfg.model_dim as i32;
            let heads = cfg.num_attention_heads as i32;
            let head_dim = cfg.head_dim() as i32;
            let src = cfg.source_dim as i32;
            let target = cfg.target_dim as i32;
            let ff = (cfg.mlp_ratio * d as f32).round() as i32;
            let vocab = cfg.target_vocab_size as i32;
            let eps = cfg.norm_eps;

            let mut r = Rng(seed);
            // Learned T5 query-token embedding [vocab, model_dim].
            let embed_rand =
                random::normal::<f32>(&[vocab, d], None, None, Some(&random::key(seed).unwrap()))
                    .unwrap();
            let embed_w = multiply(&embed_rand, Array::from_slice(&[0.05f32], &[1])).unwrap();
            let blocks = (0..cfg.num_layers)
                .map(|_| Block {
                    norm_self_attn: r.norm(d),
                    // self: kv from hidden (model_dim)
                    self_attn: r.attn(heads, head_dim, d, eps, true),
                    norm_cross_attn: r.norm(d),
                    // cross: kv from Qwen source_dim
                    cross_attn: r.attn(heads, head_dim, src, eps, false),
                    norm_mlp: r.norm(d),
                    mlp_in: r.lin(ff, d, true),
                    mlp_out: r.lin(d, ff, true),
                    eps,
                })
                .collect();
            Self {
                embed: TokenEmbedding::Dense(embed_w),
                blocks,
                out_proj: r.lin(target, d, true),
                norm: r.norm(target),
                rope: TextRope::new(head_dim, cfg.rope_theta),
                cfg,
            }
        }
    }
}
