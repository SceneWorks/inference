//! Boogu DiT building blocks: GQA self-attention, the dual-stream joint attention, the SwiGLU FFN,
//! the `LuminaRMSNormZero` modulation, and the three block flavours (plain/context, modulated
//! single-stream, double-stream).
//!
//! All attention is **bidirectional** (no causal mask) and, for the per-sample `B = 1` path, fully
//! unmasked (every token valid) — so SDPA takes no mask. Per-head q/k RMSNorm runs over the head dim
//! before the interleaved RoPE; GQA repeats each kv head to match the query heads (matching the
//! reference's explicit `repeat_interleave`).

use mlx_rs::fast::{rms_norm, scaled_dot_product_attention};
use mlx_rs::ops::{add, concatenate_axis, multiply, tanh};
use mlx_rs::Array;

use mlx_gen::adapters::AdaptableLinear;
use mlx_gen::nn::silu;
use mlx_gen::qkv::{
    self, AttnPrepSpec, FusedQkvProjection, QkNormSpec, QkvSource, RopeDtype, RopeSpec, RopeStyle,
    RopeTables,
};
use mlx_gen::weights::Weights;
use mlx_gen::Result;

use super::{join, slice_axis1};
use crate::quant::lin;

/// diffusers `Attention(eps=1e-5)` — the per-head q/k RMSNorm epsilon (distinct from the block
/// RMSNorm `norm_eps`, which is also 1e-5 here but conceptually separate).
const QK_EPS: f32 = 1e-5;

/// SC-18319 — **Boogu is knob 10's reference case**: 28 query heads / 7 kv heads, with the
/// `repeat_interleave` placed AFTER the rotation, which is the reference's own order and is *not*
/// interchangeable with repeating first (the RoPE table is indexed per kv head). The rest: separate
/// q/k/v projections (knob 9), per-head QK-RMSNorm over `head_dim` (knob 1), adjacent-pair
/// interleaved rotation (knob 2), token-major axes.
///
/// Boogu is also the family whose kv projections the fused-QKV packer refuses: `crate::quant`'s
/// `GROUP_SIZE` is **32** (explicit, non-default, because the DiT hidden 3360 is not divisible by
/// 64), and the kv `out = kv_heads · head_dim = 7 · 120 = 840` is not a multiple of 32. See
/// `mlx_gen::qkv::NoFusion::OutFeaturesNotGroupAligned`.
fn boogu_spec<'a>(
    heads: i32,
    kv_heads: i32,
    head_dim: i32,
    norm_q: &'a Array,
    norm_k: &'a Array,
    cos: &'a Array,
    sin: &'a Array,
) -> AttnPrepSpec<'a> {
    AttnPrepSpec::new(heads, head_dim)
        .with_kv_heads(kv_heads)
        .with_qk_norm(QkNormSpec::per_head(norm_q, norm_k, QK_EPS))
        .with_rope(RopeSpec {
            style: RopeStyle::AdjacentPair,
            q: Some(RopeTables::new(cos, sin)),
            k: Some(RopeTables::new(cos, sin)),
            // Knob 12 — the reference upcasts the stream for the rotation and casts BACK
            // (`rope.rs`: `.as_dtype(dt)`), so a bf16 boogu keeps a bf16 SDPA.
            dtype: RopeDtype::RestoreInput,
            ..RopeSpec::default()
        })
}

/// `1.0 + a`, broadcasting the scalar (used for the `(1 + scale)` modulation factors).
fn plus1(a: &Array) -> Result<Array> {
    Ok(add(a, Array::from_f32(1.0))?)
}

// ── GQA self-attention (standard `BooguImageAttnProcessor`) ─────────────────────────────────
/// Boogu is the tree's **production consumer of [`FusedQkvProjection`]**, and the family that proves
/// both of its arms (SC-18319):
///
/// * **Dense** — `to_q`/`to_k`/`to_v` all read the same `[b, s, hidden]` activation with the same
///   `in_features`, so they pack into one `[q_out + kv_out + kv_out, hidden]` matrix and the block
///   runs one matmul instead of three. Residency is unchanged: the packed matrix *replaces* the
///   three bases rather than shadowing them.
/// * **Quantized** — the pack is **refused**, with boogu's real numbers. `crate::quant::GROUP_SIZE`
///   is 32 (explicit and non-default, because the DiT hidden 3360 is not divisible by 64) and the kv
///   projections are `out = kv_heads · head_dim = 7 · 120 = 840`, with `840 % 32 = 8`. Concatenating
///   a group-misaligned output axis would change effective bits, so
///   [`NoFusion::OutFeaturesNotGroupAligned`](mlx_gen::qkv::NoFusion::OutFeaturesNotGroupAligned) is
///   returned and the block keeps three separate quantized matmuls. Correct beats fast.
///
/// Boogu exposes no `AdaptableHost` routing, so no adapter can be installed behind the pack here;
/// families that do route adapters keep their split projections (see `qkv::FusedQkvProjection`'s
/// adapter contract).
pub struct SelfAttention {
    qkv: FusedQkvProjection,
    o: AdaptableLinear,
    norm_q: Array,
    norm_k: Array,
    heads: i32,
    kv_heads: i32,
    head_dim: i32,
    scale: f32,
}

impl SelfAttention {
    pub fn from_weights(
        w: &Weights,
        prefix: &str,
        heads: i32,
        kv_heads: i32,
        head_dim: i32,
    ) -> Result<Self> {
        Ok(Self {
            qkv: FusedQkvProjection::new(
                lin(w, &join(prefix, "to_q"), false)?,
                lin(w, &join(prefix, "to_k"), false)?,
                lin(w, &join(prefix, "to_v"), false)?,
            ),
            o: lin(w, &join(prefix, "to_out.0"), false)?,
            norm_q: w.require(&join(prefix, "norm_q.weight"))?.clone(),
            norm_k: w.require(&join(prefix, "norm_k.weight"))?.clone(),
            heads,
            kv_heads,
            head_dim,
            scale: (head_dim as f32).powf(-0.5),
        })
    }

    /// `true` when this block's q/k/v are backed by one packed matrix — dense boogu. Q4/Q8 boogu is
    /// refused by the group-32 rule above, which the crate's tests assert at the real widths.
    pub fn fusion_engaged(&self) -> bool {
        self.qkv.fusion_engaged()
    }

    /// Drop back to three separate projections — the P6 matrix's fused-off baseline arm, and what a
    /// caller does before installing adapters. Bit-exact either way (asserted by this module's
    /// tests); [`FusedQkvProjection::repack`] returns to the packed form.
    pub fn unfuse(&mut self) -> Result<()> {
        self.qkv.unfuse()
    }

    /// `x`: `[b, s, hidden]`; `cos`/`sin`: `[1, s, head_dim/2]`. Unmasked (B=1 full sequence).
    pub fn forward(&self, x: &Array, cos: &Array, sin: &Array) -> Result<Array> {
        let (q, k, v) = self.qkv.forward(x)?;
        let heads = qkv::prepare(
            QkvSource::Separate {
                q: &q,
                k: &k,
                v: &v,
            },
            &boogu_spec(
                self.heads,
                self.kv_heads,
                self.head_dim,
                &self.norm_q,
                &self.norm_k,
                cos,
                sin,
            ),
        )?;
        let o = scaled_dot_product_attention(&heads.q, &heads.k, &heads.v, self.scale, None, None)?;
        self.o.forward(&qkv::merge_heads(&o)?)
    }

    pub fn quantize(&mut self, bits: i32) -> Result<()> {
        // Unpacks, quantizes each part at boogu's group 32, then re-attempts the pack — which the
        // safety predicate refuses on the misaligned kv `out`. See this type's doc.
        self.qkv.quantize(bits, Some(crate::quant::GROUP_SIZE))?;
        self.o.quantize(bits, Some(crate::quant::GROUP_SIZE))?;
        Ok(())
    }
}

// ── Dual-stream joint attention (`BooguImageDoubleStreamSelfAttnProcessor`) ──────────────────
/// Separate img/instruct QKV projections; the streams are concatenated **instruct-first**, attended
/// jointly, split back, projected by separate `img_out`/`instruct_out`, re-merged, and run through
/// the shared `to_out.0`. The block then re-splits the result into its instruct/img halves.
///
/// Both streams' QKV go through [`FusedQkvProjection`] on the same terms as [`SelfAttention`] —
/// packed while dense, refused (group 32, kv `out = 840`) once quantized.
pub struct JointAttention {
    img_qkv: FusedQkvProjection,
    instruct_qkv: FusedQkvProjection,
    img_out: AdaptableLinear,
    instruct_out: AdaptableLinear,
    to_out: AdaptableLinear,
    norm_q: Array,
    norm_k: Array,
    heads: i32,
    kv_heads: i32,
    head_dim: i32,
    scale: f32,
}

impl JointAttention {
    pub fn from_weights(
        w: &Weights,
        prefix: &str,
        heads: i32,
        kv_heads: i32,
        head_dim: i32,
    ) -> Result<Self> {
        let p = |s: &str| join(prefix, s);
        Ok(Self {
            img_qkv: FusedQkvProjection::new(
                lin(w, &p("processor.img_to_q"), false)?,
                lin(w, &p("processor.img_to_k"), false)?,
                lin(w, &p("processor.img_to_v"), false)?,
            ),
            instruct_qkv: FusedQkvProjection::new(
                lin(w, &p("processor.instruct_to_q"), false)?,
                lin(w, &p("processor.instruct_to_k"), false)?,
                lin(w, &p("processor.instruct_to_v"), false)?,
            ),
            img_out: lin(w, &p("processor.img_out"), false)?,
            instruct_out: lin(w, &p("processor.instruct_out"), false)?,
            to_out: lin(w, &p("to_out.0"), false)?,
            norm_q: w.require(&p("norm_q.weight"))?.clone(),
            norm_k: w.require(&p("norm_k.weight"))?.clone(),
            heads,
            kv_heads,
            head_dim,
            scale: (head_dim as f32).powf(-0.5),
        })
    }

    /// `img`: `[b, Li, D]`, `instruct`: `[b, Lt, D]`, joint `cos`/`sin`: `[1, Lt+Li, head_dim/2]`.
    /// Returns the joint attention output `[b, Lt+Li, D]` (instruct-first).
    pub fn forward(
        &self,
        img: &Array,
        instruct: &Array,
        cos: &Array,
        sin: &Array,
    ) -> Result<Array> {
        let (li, lt) = (img.shape()[1], instruct.shape()[1]);

        // Knob 8's concat-then-everything arm plus knob 11's `[instruct, img]` order: the two
        // streams' PROJECTIONS are concatenated on the token axis before the shared QK-norm and
        // rotation run over the joint sequence, so there is exactly one prologue here rather than
        // two joined afterwards. (Concatenating `[b, l, n·hd]` on the token axis and then splitting
        // heads is identical to splitting heads first and concatenating on axis 1, which is what
        // this used to do.)
        let (iq, ik, iv) = self.instruct_qkv.forward(instruct)?;
        let (mq, mk, mv) = self.img_qkv.forward(img)?;
        let joint = |instruct_proj: &Array, img_proj: &Array| -> Result<Array> {
            Ok(concatenate_axis(&[instruct_proj, img_proj], 1)?)
        };
        let heads = qkv::prepare(
            QkvSource::Separate {
                q: &joint(&iq, &mq)?,
                k: &joint(&ik, &mk)?,
                v: &joint(&iv, &mv)?,
            },
            &boogu_spec(
                self.heads,
                self.kv_heads,
                self.head_dim,
                &self.norm_q,
                &self.norm_k,
                cos,
                sin,
            ),
        )?;
        let o = scaled_dot_product_attention(&heads.q, &heads.k, &heads.v, self.scale, None, None)?;
        let o = qkv::merge_heads(&o)?;

        // Split → separate output projections → re-merge → shared output projection.
        let instruct_part = slice_axis1(&o, 0, lt)?;
        let img_part = slice_axis1(&o, lt, lt + li)?;
        let merged = concatenate_axis(
            &[
                &self.instruct_out.forward(&instruct_part)?,
                &self.img_out.forward(&img_part)?,
            ],
            1,
        )?;
        self.to_out.forward(&merged)
    }

    /// `true` when both streams' q/k/v are backed by one packed matrix each — dense boogu only.
    pub fn fusion_engaged(&self) -> bool {
        self.img_qkv.fusion_engaged() && self.instruct_qkv.fusion_engaged()
    }

    /// Drop both streams back to separate projections — the P6 fused-off baseline arm.
    pub fn unfuse(&mut self) -> Result<()> {
        self.img_qkv.unfuse()?;
        self.instruct_qkv.unfuse()
    }

    pub fn quantize(&mut self, bits: i32) -> Result<()> {
        self.img_qkv
            .quantize(bits, Some(crate::quant::GROUP_SIZE))?;
        self.instruct_qkv
            .quantize(bits, Some(crate::quant::GROUP_SIZE))?;
        for p in [&mut self.img_out, &mut self.instruct_out, &mut self.to_out] {
            p.quantize(bits, Some(crate::quant::GROUP_SIZE))?;
        }
        Ok(())
    }
}

// ── SwiGLU feed-forward (`LuminaFeedForward`) ───────────────────────────────────────────────
pub struct SwiGlu {
    w1: AdaptableLinear,
    w2: AdaptableLinear,
    w3: AdaptableLinear,
}

impl SwiGlu {
    pub fn from_weights(w: &Weights, prefix: &str) -> Result<Self> {
        Ok(Self {
            w1: lin(w, &join(prefix, "linear_1"), false)?,
            w2: lin(w, &join(prefix, "linear_2"), false)?,
            w3: lin(w, &join(prefix, "linear_3"), false)?,
        })
    }

    pub fn forward(&self, x: &Array) -> Result<Array> {
        let gated = multiply(&silu(&self.w1.forward(x)?)?, &self.w3.forward(x)?)?;
        self.w2.forward(&gated)
    }

    pub fn quantize(&mut self, bits: i32) -> Result<()> {
        self.w1.quantize(bits, Some(crate::quant::GROUP_SIZE))?;
        self.w2.quantize(bits, Some(crate::quant::GROUP_SIZE))?;
        self.w3.quantize(bits, Some(crate::quant::GROUP_SIZE))?;
        Ok(())
    }
}

// ── LuminaRMSNormZero modulation ────────────────────────────────────────────────────────────
/// `emb = linear(silu(temb))` (`1024 → 4·D`), chunked into `(scale_msa, gate_msa, scale_mlp,
/// gate_mlp)`; the returned hidden is `RMSNorm(x)·(1 + scale_msa)`. The caller reuses the other three
/// chunks per its modulation pattern (different blocks read different chunk slots).
pub struct ModNorm {
    linear: AdaptableLinear,
    norm: Array,
    eps: f32,
}

impl ModNorm {
    pub fn from_weights(w: &Weights, prefix: &str, eps: f32) -> Result<Self> {
        Ok(Self {
            linear: lin(w, &join(prefix, "linear"), true)?,
            norm: w.require(&join(prefix, "norm.weight"))?.clone(),
            eps,
        })
    }

    /// `x`: `[b, s, D]`, `temb`: `[b, 1, 1024]`. Returns `(normed, c2, c3, c4)`, each `[b, 1, D]`
    /// except `normed` which is `[b, s, D]`.
    pub fn forward(&self, x: &Array, temb: &Array) -> Result<(Array, Array, Array, Array)> {
        let emb = self.linear.forward(&silu(temb)?)?; // [b, 1, 4D]
        let chunks = mlx_rs::ops::split(&emb, 4, 2)?;
        let normed = multiply(&rms_norm(x, &self.norm, self.eps)?, &plus1(&chunks[0])?)?;
        Ok((
            normed,
            chunks[1].clone(),
            chunks[2].clone(),
            chunks[3].clone(),
        ))
    }

    pub fn quantize(&mut self, bits: i32) -> Result<()> {
        self.linear.quantize(bits, Some(crate::quant::GROUP_SIZE))
    }
}

// ── Plain (non-modulated) block — context refiner ───────────────────────────────────────────
pub struct PlainBlock {
    attn: SelfAttention,
    ff: SwiGlu,
    norm1: Array,
    norm2: Array,
    ffn_norm1: Array,
    ffn_norm2: Array,
    eps: f32,
}

impl PlainBlock {
    pub fn from_weights(
        w: &Weights,
        prefix: &str,
        heads: i32,
        kv_heads: i32,
        head_dim: i32,
        eps: f32,
    ) -> Result<Self> {
        Ok(Self {
            attn: SelfAttention::from_weights(w, &join(prefix, "attn"), heads, kv_heads, head_dim)?,
            ff: SwiGlu::from_weights(w, &join(prefix, "feed_forward"))?,
            norm1: w.require(&join(prefix, "norm1.weight"))?.clone(),
            norm2: w.require(&join(prefix, "norm2.weight"))?.clone(),
            ffn_norm1: w.require(&join(prefix, "ffn_norm1.weight"))?.clone(),
            ffn_norm2: w.require(&join(prefix, "ffn_norm2.weight"))?.clone(),
            eps,
        })
    }

    pub fn forward(&self, x: &Array, cos: &Array, sin: &Array) -> Result<Array> {
        let attn = self
            .attn
            .forward(&rms_norm(x, &self.norm1, self.eps)?, cos, sin)?;
        let x = add(x, &rms_norm(&attn, &self.norm2, self.eps)?)?;
        let mlp = self.ff.forward(&rms_norm(&x, &self.ffn_norm1, self.eps)?)?;
        Ok(add(&x, &rms_norm(&mlp, &self.ffn_norm2, self.eps)?)?)
    }

    pub fn quantize(&mut self, bits: i32) -> Result<()> {
        self.attn.quantize(bits)?;
        self.ff.quantize(bits)
    }

    /// SC-18319's fused-off baseline arm — see `Transformer::unfuse_qkv`.
    pub fn unfuse_qkv(&mut self) -> Result<()> {
        self.attn.unfuse()
    }

    /// SC-18319 — see `Transformer::qkv_fusion_engaged`.
    pub fn qkv_fusion_engaged(&self) -> bool {
        self.attn.fusion_engaged()
    }
}

// ── Modulated single-stream / noise-refiner block ───────────────────────────────────────────
pub struct ModBlock {
    attn: SelfAttention,
    ff: SwiGlu,
    norm1: ModNorm,
    norm2: Array,
    ffn_norm1: Array,
    ffn_norm2: Array,
    eps: f32,
}

impl ModBlock {
    pub fn from_weights(
        w: &Weights,
        prefix: &str,
        heads: i32,
        kv_heads: i32,
        head_dim: i32,
        eps: f32,
    ) -> Result<Self> {
        Ok(Self {
            attn: SelfAttention::from_weights(w, &join(prefix, "attn"), heads, kv_heads, head_dim)?,
            ff: SwiGlu::from_weights(w, &join(prefix, "feed_forward"))?,
            norm1: ModNorm::from_weights(w, &join(prefix, "norm1"), eps)?,
            norm2: w.require(&join(prefix, "norm2.weight"))?.clone(),
            ffn_norm1: w.require(&join(prefix, "ffn_norm1.weight"))?.clone(),
            ffn_norm2: w.require(&join(prefix, "ffn_norm2.weight"))?.clone(),
            eps,
        })
    }

    /// `x`: `[b, s, D]`, `temb`: `[b, 1, 1024]`.
    pub fn forward(&self, x: &Array, cos: &Array, sin: &Array, temb: &Array) -> Result<Array> {
        let (normed, gate_msa, scale_mlp, gate_mlp) = self.norm1.forward(x, temb)?;
        let attn = self.attn.forward(&normed, cos, sin)?;
        let x = add(
            x,
            &multiply(&tanh(&gate_msa)?, &rms_norm(&attn, &self.norm2, self.eps)?)?,
        )?;
        let mlp_in = multiply(
            &rms_norm(&x, &self.ffn_norm1, self.eps)?,
            &plus1(&scale_mlp)?,
        )?;
        let mlp = self.ff.forward(&mlp_in)?;
        Ok(add(
            &x,
            &multiply(
                &tanh(&gate_mlp)?,
                &rms_norm(&mlp, &self.ffn_norm2, self.eps)?,
            )?,
        )?)
    }

    pub fn quantize(&mut self, bits: i32) -> Result<()> {
        self.attn.quantize(bits)?;
        self.ff.quantize(bits)?;
        self.norm1.quantize(bits)
    }

    /// SC-18319's fused-off baseline arm — see `Transformer::unfuse_qkv`.
    pub fn unfuse_qkv(&mut self) -> Result<()> {
        self.attn.unfuse()
    }

    /// SC-18319 — see `Transformer::qkv_fusion_engaged`.
    pub fn qkv_fusion_engaged(&self) -> bool {
        self.attn.fusion_engaged()
    }
}

// ── Double-stream block ─────────────────────────────────────────────────────────────────────
pub struct DoubleBlock {
    joint_attn: JointAttention,
    self_attn: SelfAttention,
    img_ff: SwiGlu,
    instruct_ff: SwiGlu,
    img_norm1: ModNorm,
    img_norm2: ModNorm,
    img_norm3: ModNorm,
    instruct_norm1: ModNorm,
    instruct_norm2: ModNorm,
    img_attn_norm: Array,
    img_self_attn_norm: Array,
    img_ffn_norm1: Array,
    img_ffn_norm2: Array,
    instruct_attn_norm: Array,
    instruct_ffn_norm1: Array,
    instruct_ffn_norm2: Array,
    eps: f32,
}

impl DoubleBlock {
    pub fn from_weights(
        w: &Weights,
        prefix: &str,
        heads: i32,
        kv_heads: i32,
        head_dim: i32,
        eps: f32,
    ) -> Result<Self> {
        let req = |s: &str| -> Result<Array> { Ok(w.require(&join(prefix, s))?.clone()) };
        Ok(Self {
            joint_attn: JointAttention::from_weights(
                w,
                &join(prefix, "img_instruct_attn"),
                heads,
                kv_heads,
                head_dim,
            )?,
            self_attn: SelfAttention::from_weights(
                w,
                &join(prefix, "img_self_attn"),
                heads,
                kv_heads,
                head_dim,
            )?,
            img_ff: SwiGlu::from_weights(w, &join(prefix, "img_feed_forward"))?,
            instruct_ff: SwiGlu::from_weights(w, &join(prefix, "instruct_feed_forward"))?,
            img_norm1: ModNorm::from_weights(w, &join(prefix, "img_norm1"), eps)?,
            img_norm2: ModNorm::from_weights(w, &join(prefix, "img_norm2"), eps)?,
            img_norm3: ModNorm::from_weights(w, &join(prefix, "img_norm3"), eps)?,
            instruct_norm1: ModNorm::from_weights(w, &join(prefix, "instruct_norm1"), eps)?,
            instruct_norm2: ModNorm::from_weights(w, &join(prefix, "instruct_norm2"), eps)?,
            img_attn_norm: req("img_attn_norm.weight")?,
            img_self_attn_norm: req("img_self_attn_norm.weight")?,
            img_ffn_norm1: req("img_ffn_norm1.weight")?,
            img_ffn_norm2: req("img_ffn_norm2.weight")?,
            instruct_attn_norm: req("instruct_attn_norm.weight")?,
            instruct_ffn_norm1: req("instruct_ffn_norm1.weight")?,
            instruct_ffn_norm2: req("instruct_ffn_norm2.weight")?,
            eps,
        })
    }

    /// `img`: `[b, Li, D]`, `instruct`: `[b, Lt, D]`; joint `cos`/`sin`: `[1, Lt+Li, head_dim/2]`;
    /// image `img_cos`/`img_sin`: `[1, Li, head_dim/2]`; `temb`: `[b, 1, 1024]`.
    /// Returns the updated `(img, instruct)`.
    #[allow(clippy::too_many_arguments)]
    pub fn forward(
        &self,
        img: &Array,
        instruct: &Array,
        cos: &Array,
        sin: &Array,
        img_cos: &Array,
        img_sin: &Array,
        temb: &Array,
    ) -> Result<(Array, Array)> {
        let lt = instruct.shape()[1];
        let li = img.shape()[1];

        let (img_n1, img_gate_msa, img_scale_mlp, img_gate_mlp) =
            self.img_norm1.forward(img, temb)?;
        let (img_n2, img_shift_mlp, _, _) = self.img_norm2.forward(img, temb)?;
        let (img_n3, img_gate_self, _, _) = self.img_norm3.forward(img, temb)?;
        let (ins_n1, ins_gate_msa, ins_scale_mlp, ins_gate_mlp) =
            self.instruct_norm1.forward(instruct, temb)?;
        let (ins_n2, ins_shift_mlp, _, _) = self.instruct_norm2.forward(instruct, temb)?;

        // Joint instruct↔img attention, then split back to the two streams.
        let joint = self.joint_attn.forward(&img_n1, &ins_n1, cos, sin)?;
        let instruct_attn_out = slice_axis1(&joint, 0, lt)?;
        let img_attn_out = slice_axis1(&joint, lt, lt + li)?;

        // Image self-attention.
        let img_self_out = self.self_attn.forward(&img_n3, img_cos, img_sin)?;

        // Image residual updates.
        let img = add(
            img,
            &multiply(
                &tanh(&img_gate_msa)?,
                &rms_norm(&img_attn_out, &self.img_attn_norm, self.eps)?,
            )?,
        )?;
        let img = add(
            &img,
            &multiply(
                &tanh(&img_gate_self)?,
                &rms_norm(&img_self_out, &self.img_self_attn_norm, self.eps)?,
            )?,
        )?;
        let img_mlp_in = add(&multiply(&img_n2, &plus1(&img_scale_mlp)?)?, &img_shift_mlp)?;
        let img_mlp =
            self.img_ff
                .forward(&rms_norm(&img_mlp_in, &self.img_ffn_norm1, self.eps)?)?;
        let img = add(
            &img,
            &multiply(
                &tanh(&img_gate_mlp)?,
                &rms_norm(&img_mlp, &self.img_ffn_norm2, self.eps)?,
            )?,
        )?;

        // Instruction residual updates.
        let instruct = add(
            instruct,
            &multiply(
                &tanh(&ins_gate_msa)?,
                &rms_norm(&instruct_attn_out, &self.instruct_attn_norm, self.eps)?,
            )?,
        )?;
        let ins_mlp_in = add(&multiply(&ins_n2, &plus1(&ins_scale_mlp)?)?, &ins_shift_mlp)?;
        let ins_mlp = self.instruct_ff.forward(&rms_norm(
            &ins_mlp_in,
            &self.instruct_ffn_norm1,
            self.eps,
        )?)?;
        let instruct = add(
            &instruct,
            &multiply(
                &tanh(&ins_gate_mlp)?,
                &rms_norm(&ins_mlp, &self.instruct_ffn_norm2, self.eps)?,
            )?,
        )?;

        Ok((img, instruct))
    }

    pub fn quantize(&mut self, bits: i32) -> Result<()> {
        self.joint_attn.quantize(bits)?;
        self.self_attn.quantize(bits)?;
        self.img_ff.quantize(bits)?;
        self.instruct_ff.quantize(bits)?;
        self.img_norm1.quantize(bits)?;
        self.img_norm2.quantize(bits)?;
        self.img_norm3.quantize(bits)?;
        self.instruct_norm1.quantize(bits)?;
        self.instruct_norm2.quantize(bits)?;
        Ok(())
    }

    /// SC-18319's fused-off baseline arm — see `Transformer::unfuse_qkv`.
    pub fn unfuse_qkv(&mut self) -> Result<()> {
        self.joint_attn.unfuse()?;
        self.self_attn.unfuse()
    }

    /// SC-18319 — see `Transformer::qkv_fusion_engaged`.
    pub fn qkv_fusion_engaged(&self) -> bool {
        self.joint_attn.fusion_engaged() && self.self_attn.fusion_engaged()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use mlx_rs::ops::array_eq;
    use std::collections::HashMap;

    /// Boogu's real attention geometry. Every divisibility fact that drives the fusion decision is
    /// preserved exactly: 28 query heads / 7 kv heads at `head_dim = 120`, so the kv projections are
    /// `out = 840`, and `crate::quant::GROUP_SIZE` is 32.
    const HEADS: i32 = 28;
    const KV_HEADS: i32 = 7;
    const HEAD_DIM: i32 = 120;

    fn det(shape: &[i32], scale: f32, offset: f32) -> Array {
        let n: i32 = shape.iter().product();
        let data: Vec<f32> = (0..n)
            .map(|i| ((i as f32) * scale + offset).sin())
            .collect();
        Array::from_slice(&data, shape)
    }

    /// A synthetic `Weights` for one `SelfAttention` at the geometry above. `hidden` is narrowed to
    /// a group-aligned 320 so the test stays cheap, while the kv `out` — the *only* reason the pack
    /// is refused — keeps its real value.
    fn self_attn_weights(hidden: i32) -> Weights {
        let q_out = HEADS * HEAD_DIM;
        let kv_out = KV_HEADS * HEAD_DIM;
        let mut m: HashMap<String, Array> = HashMap::new();
        m.insert("a.to_q.weight".into(), det(&[q_out, hidden], 0.0007, 0.1));
        m.insert("a.to_k.weight".into(), det(&[kv_out, hidden], 0.0009, 0.2));
        m.insert("a.to_v.weight".into(), det(&[kv_out, hidden], 0.0011, 0.3));
        m.insert(
            "a.to_out.0.weight".into(),
            det(&[hidden, q_out], 0.0005, 0.4),
        );
        m.insert("a.norm_q.weight".into(), det(&[HEAD_DIM], 0.31, 1.0));
        m.insert("a.norm_k.weight".into(), det(&[HEAD_DIM], 0.17, 0.5));
        Weights::from_map(m)
    }

    fn build(hidden: i32) -> SelfAttention {
        SelfAttention::from_weights(&self_attn_weights(hidden), "a", HEADS, KV_HEADS, HEAD_DIM)
            .expect("synthetic weights")
    }

    /// SC-18319 — the production wiring, at boogu's own numbers.
    ///
    /// **Dense**: the three projections read the same activation at the same `in_features`, so they
    /// pack into one matrix and the block runs one matmul instead of three. **Quantized**:
    /// `GROUP_SIZE = 32` and the kv `out = 7 · 120 = 840` with `840 % 32 = 8`, so the pack is
    /// refused rather than silently changing effective bits. Fusing must not move a single bit.
    #[test]
    fn self_attention_packs_while_dense_and_is_refused_once_quantized() {
        // The divisibility facts the decision rests on, asserted as arithmetic so a constant drift
        // fails here rather than silently re-enabling an unsafe pack.
        assert_eq!(
            crate::quant::GROUP_SIZE,
            32,
            "boogu's group size is explicit"
        );
        assert_eq!((KV_HEADS * HEAD_DIM) % crate::quant::GROUP_SIZE, 8);
        assert_ne!(HEAD_DIM % crate::quant::GROUP_SIZE, 0);
        assert_eq!(
            (HEADS * HEAD_DIM) % crate::quant::GROUP_SIZE,
            0,
            "the query projection alone would align — a triple packs or it does not"
        );

        let hidden = 320;
        let mut fused = build(hidden);
        assert!(
            fused.fusion_engaged(),
            "a dense boogu self-attention must pack its q/k/v into one matrix"
        );

        let s = 4;
        let x = det(&[1, s, hidden], 0.003, 0.25);
        let cos = det(&[1, s, HEAD_DIM / 2], 0.05, 0.0);
        let sin = det(&[1, s, HEAD_DIM / 2], 0.07, 0.6);
        let packed_out = fused.forward(&x, &cos, &sin).expect("dense forward");

        // Unfusing must not change a single bit — the packed matrix is the row-wise concatenation
        // of the three bases and a matmul's rows are independent.
        let mut split = build(hidden);
        split.unfuse().expect("unfuse");
        assert!(!split.fusion_engaged(), "the baseline arm must be split");
        let split_out = split.forward(&x, &cos, &sin).expect("split forward");
        assert_eq!(packed_out.shape(), split_out.shape());
        assert!(
            array_eq(&packed_out, &split_out, None)
                .unwrap()
                .item::<bool>(),
            "fusing boogu's q/k/v must be bit-exact"
        );

        // Quantizing must REFUSE the pack on the misaligned kv output axis.
        fused.quantize(8).expect("q8");
        assert!(
            !fused.fusion_engaged(),
            "boogu's kv out = {} is not a multiple of GROUP_SIZE {}, so the pack must be refused \
             rather than papered over",
            KV_HEADS * HEAD_DIM,
            crate::quant::GROUP_SIZE
        );
    }
}
