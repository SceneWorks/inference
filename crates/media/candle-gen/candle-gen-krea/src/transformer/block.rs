//! Krea 2 DiT building blocks — port of `mlx-gen-krea`'s `transformer/block.rs` (the reference
//! `mmdit.py` modules): the sigmoid-**gated** GQA attention (`Attention` + `QKNorm`), the `SwiGLU`
//! FFN, the `+1` `RMSNorm`, the un-modulated `TextFusionBlock`, the `DoubleSharedModulation`
//! single-stream block, and the `TextFusionTransformer` layer aggregator.
//!
//! Every `RMSNorm` here computes `weight = scale + 1` in f32 (the reference stores the raw `scale`,
//! centered at 0), distinct from the apply-weight-directly norms of the Qwen3-VL text encoder.
//! Attention adds a `to_gate` projection: the post-attention output is multiplied by
//! `sigmoid(to_gate(x))` before `to_out`. Block gates (`pregate`/`postgate`) are raw (no activation).

use candle_gen::candle_core::{Result, Tensor, D};
use candle_gen::candle_nn::ops::{sigmoid, softmax_last_dim};

use super::rope::apply_interleaved_rope;
use crate::loader::{linear_detect, linear_detect_planned, rms_scale, rms_scale_weight, Weights};
use crate::nvfp4_dit::DitPlan;
use crate::quant::QLinear;
use candle_gen::quant::AdaptLinear;

/// Join a module prefix with a leaf name.
fn join(prefix: &str, name: &str) -> String {
    if prefix.is_empty() {
        name.to_string()
    } else {
        format!("{prefix}.{name}")
    }
}

/// Repeat each kv head `groups` times consecutively ([b,s,hkv,hd] → [b,s,hkv·groups,hd]) —
/// `repeat_interleave` over the head axis, matching the reference `enable_gqa`.
fn repeat_kv(x: &Tensor, groups: usize) -> Result<Tensor> {
    if groups == 1 {
        return Ok(x.clone());
    }
    let (b, s, hkv, hd) = x.dims4()?;
    x.unsqueeze(3)?
        .expand((b, s, hkv, groups, hd))?
        .contiguous()?
        .reshape((b, s, hkv * groups, hd))
}

/// Bidirectional, unmasked scaled-dot-product attention. `q`/`k`/`v`: `[b, h, s, hd]`.
///
/// i32-overflow guard (sc-9116): the image-token scores `[b, h, s, s]` reach `~24·16384² ≈ 6.4e9 >
/// i32::MAX` at a 2048² render, silently corrupting the tail rows on the candle CUDA kernels. The
/// shared budgeted helper chunks over the query rows (byte-identical for the common sizes); the softmax
/// closure preserves the exact fused `softmax_last_dim`.
fn sdpa(q: &Tensor, k: &Tensor, v: &Tensor, scale: f64) -> Result<Tensor> {
    candle_gen::sdpa_budgeted_bhsd(
        q,
        k,
        v,
        scale,
        None,
        softmax_last_dim,
        candle_gen::ATTN_SCORES_BUDGET,
    )
}

// ── `+1` RMSNorm ────────────────────────────────────────────────────────────────────────────
/// Reference `RMSNorm`: `F.rms_norm(x.float(), weight = scale.float() + 1.0)` then cast back. The
/// stored param is the raw `scale` (centered at 0); the `+1` is pre-folded into an f32 weight at load.
pub struct RmsScale {
    weight: Tensor, // f32, = scale + 1
    eps: f64,
}

impl RmsScale {
    pub fn load(w: &Weights, key: &str, eps: f64) -> Result<Self> {
        Ok(Self {
            weight: rms_scale_weight(w, key)?,
            eps,
        })
    }

    pub fn forward(&self, x: &Tensor) -> Result<Tensor> {
        rms_scale(x, &self.weight, self.eps)
    }
}

// ── Sigmoid-gated GQA attention (reference `Attention`) ─────────────────────────────────────
pub struct GatedAttention {
    q: QLinear,
    k: QLinear,
    v: QLinear,
    gate: QLinear,
    o: QLinear,
    norm_q: RmsScale,
    norm_k: RmsScale,
    heads: usize,
    kv_heads: usize,
    head_dim: usize,
    scale: f64,
}

impl GatedAttention {
    pub fn load(
        w: &Weights,
        prefix: &str,
        heads: usize,
        kv_heads: usize,
        head_dim: usize,
        eps: f64,
    ) -> Result<Self> {
        Self::load_planned(
            w,
            prefix,
            heads,
            kv_heads,
            head_dim,
            eps,
            &DitPlan::baseline(),
        )
    }

    /// [`Self::load`] under a [`DitPlan`] (sc-12110) — every one of the five projections is served
    /// through the plan's leg (NVFP4, probed, or the byte-unchanged baseline).
    ///
    /// All five share **one** input activation (`x`), so `to_gate` — the projection with no SANA
    /// analogue — necessarily measures the same outlier sparsity as `to_q`/`to_k`/`to_v`. That is a
    /// property of the topology, not an assumption: the probe records each projection's *input*, and the
    /// gate reads the same `x` the QKV trio does. `to_out.0` is the one distinct site (it reads the
    /// sigmoid-gated attention output).
    #[allow(clippy::too_many_arguments)]
    pub fn load_planned(
        w: &Weights,
        prefix: &str,
        heads: usize,
        kv_heads: usize,
        head_dim: usize,
        eps: f64,
        plan: &DitPlan,
    ) -> Result<Self> {
        let p = |leaf: &str| linear_detect_planned(w, &join(prefix, leaf), false, plan);
        Ok(Self {
            q: p("to_q")?,
            k: p("to_k")?,
            v: p("to_v")?,
            gate: p("to_gate")?,
            o: p("to_out.0")?,
            norm_q: RmsScale::load(w, &join(prefix, "norm_q.weight"), eps)?,
            norm_k: RmsScale::load(w, &join(prefix, "norm_k.weight"), eps)?,
            heads,
            kv_heads,
            head_dim,
            scale: (head_dim as f64).powf(-0.5),
        })
    }

    /// Every projection in this attention module, paired with its dotted key — the walk
    /// [`crate::transformer::Krea2Transformer::nvfp4_report`] uses for SC#6/SC#4 accounting.
    pub(crate) fn projections<'a>(
        &'a self,
        prefix: &str,
    ) -> Vec<(String, &'a crate::quant::QLinear)> {
        [
            ("to_q", &self.q),
            ("to_k", &self.k),
            ("to_v", &self.v),
            ("to_gate", &self.gate),
            ("to_out.0", &self.o),
        ]
        .into_iter()
        .map(|(leaf, p)| (join(prefix, leaf), p))
        .collect()
    }

    /// `x`: `[b, s, hidden]`. `rope`: `Some((cos, sin))` (`[1, s, head_dim/2]`) for the single-stream
    /// blocks; `None` for the text-fusion blocks (no positional encoding). Unmasked (B=1 full sequence).
    pub fn forward(&self, x: &Tensor, rope: Option<(&Tensor, &Tensor)>) -> Result<Tensor> {
        let (b, s, _) = x.dims3()?;
        let (nh, nkv, hd) = (self.heads, self.kv_heads, self.head_dim);

        let q = self.q.forward(x)?.reshape((b, s, nh, hd))?;
        let k = self.k.forward(x)?.reshape((b, s, nkv, hd))?;
        let v = self.v.forward(x)?.reshape((b, s, nkv, hd))?;
        let gate = self.gate.forward(x)?; // [b, s, hidden]

        let q = self.norm_q.forward(&q)?;
        let k = self.norm_k.forward(&k)?;
        let (q, k) = match rope {
            Some((cos, sin)) => (
                apply_interleaved_rope(&q, cos, sin)?,
                apply_interleaved_rope(&k, cos, sin)?,
            ),
            None => (q, k),
        };

        let groups = nh / nkv;
        let k = repeat_kv(&k, groups)?;
        let v = repeat_kv(&v, groups)?;

        let q = q.transpose(1, 2)?;
        let k = k.transpose(1, 2)?;
        let v = v.transpose(1, 2)?;
        let o = sdpa(&q, &k, &v, self.scale)?;
        let o = o.transpose(1, 2)?.contiguous()?.reshape((b, s, nh * hd))?;

        // Sigmoid gate the attention output, then the shared output projection.
        let gated = (o * sigmoid(&gate)?)?;
        self.o.forward(&gated)
    }

    /// Visit the five gated-attention projections under `{prefix}` (`to_q/to_k/to_v/to_gate/to_out.0`) —
    /// the surface a user LoRA/LoKr adapts (sc-11105). The q/k RMS scales are not projections. An
    /// int8-ConvRot projection is skipped (never adaptable; the ConvRot lane rejects adapters).
    pub fn visit_adaptable_mut(
        &mut self,
        prefix: &str,
        f: &mut dyn FnMut(&str, &mut AdaptLinear) -> candle_gen::Result<()>,
    ) -> candle_gen::Result<()> {
        for (leaf, proj) in [
            ("to_q", &mut self.q),
            ("to_k", &mut self.k),
            ("to_v", &mut self.v),
            ("to_gate", &mut self.gate),
            ("to_out.0", &mut self.o),
        ] {
            if let Some(a) = proj.as_adapt_mut() {
                f(&join(prefix, leaf), a)?;
            }
        }
        Ok(())
    }
}

// ── SwiGLU feed-forward (reference `SwiGLU`: `down(silu(gate(x)) * up(x))`) ──────────────────
pub struct SwiGlu {
    gate: QLinear,
    up: QLinear,
    down: QLinear,
}

impl SwiGlu {
    pub fn load(w: &Weights, prefix: &str) -> Result<Self> {
        Self::load_planned(w, prefix, &DitPlan::baseline())
    }

    /// [`Self::load`] under a [`DitPlan`] (sc-12110). On Krea's single-stream blocks these three are the
    /// **largest GEMMs in the trunk** (`[16384, 6144]` ×2 + `[6144, 16384]`) and ~62% of a block's linear
    /// FLOPs — the layers SC#1's ~2× has to come from, and the reason the `N = 16384` amortization
    /// prediction is testable here at all (sc-12110).
    pub fn load_planned(w: &Weights, prefix: &str, plan: &DitPlan) -> Result<Self> {
        let p = |leaf: &str| linear_detect_planned(w, &join(prefix, leaf), false, plan);
        Ok(Self {
            gate: p("gate")?,
            up: p("up")?,
            down: p("down")?,
        })
    }

    /// Every projection in this SwiGLU, paired with its dotted key (SC#6/SC#4 accounting walk).
    pub(crate) fn projections<'a>(
        &'a self,
        prefix: &str,
    ) -> Vec<(String, &'a crate::quant::QLinear)> {
        [("gate", &self.gate), ("up", &self.up), ("down", &self.down)]
            .into_iter()
            .map(|(leaf, p)| (join(prefix, leaf), p))
            .collect()
    }

    pub fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let gated = (self.gate.forward(x)?.silu()? * self.up.forward(x)?)?;
        self.down.forward(&gated)
    }

    /// Visit the three SwiGLU projections under `{prefix}` (`gate/up/down`) — sc-11105.
    pub fn visit_adaptable_mut(
        &mut self,
        prefix: &str,
        f: &mut dyn FnMut(&str, &mut AdaptLinear) -> candle_gen::Result<()>,
    ) -> candle_gen::Result<()> {
        for (leaf, proj) in [
            ("gate", &mut self.gate),
            ("up", &mut self.up),
            ("down", &mut self.down),
        ] {
            if let Some(a) = proj.as_adapt_mut() {
                f(&join(prefix, leaf), a)?;
            }
        }
        Ok(())
    }
}

// ── Un-modulated text-fusion block (reference `TextFusionBlock`) ─────────────────────────────
/// `x = x + attn(prenorm(x)); x = x + mlp(postnorm(x))`. No modulation, no RoPE.
pub struct TextFusionBlock {
    prenorm: RmsScale,
    postnorm: RmsScale,
    attn: GatedAttention,
    mlp: SwiGlu,
}

impl TextFusionBlock {
    pub fn load(
        w: &Weights,
        prefix: &str,
        heads: usize,
        kv_heads: usize,
        head_dim: usize,
        eps: f64,
    ) -> Result<Self> {
        Self::load_planned(
            w,
            prefix,
            heads,
            kv_heads,
            head_dim,
            eps,
            &DitPlan::baseline(),
        )
    }

    /// [`Self::load`] under a [`DitPlan`] (sc-12110). These blocks consume the **raw stacked Qwen3-VL
    /// hidden states** — the massive-activation source — so under the mixed policy they resolve to the
    /// guarded `is_context_read` class (Krea's analogue of SANA's `caption_projection` + `attn2`; see
    /// [`crate::nvfp4_dit`]). The guard is nearly free for SC#1: 4 blocks at width 2560 against 28 at
    /// width 6144.
    #[allow(clippy::too_many_arguments)]
    pub fn load_planned(
        w: &Weights,
        prefix: &str,
        heads: usize,
        kv_heads: usize,
        head_dim: usize,
        eps: f64,
        plan: &DitPlan,
    ) -> Result<Self> {
        Ok(Self {
            prenorm: RmsScale::load(w, &join(prefix, "norm1.weight"), eps)?,
            postnorm: RmsScale::load(w, &join(prefix, "norm2.weight"), eps)?,
            attn: GatedAttention::load_planned(
                w,
                &join(prefix, "attn"),
                heads,
                kv_heads,
                head_dim,
                eps,
                plan,
            )?,
            mlp: SwiGlu::load_planned(w, &join(prefix, "ff"), plan)?,
        })
    }

    /// Every projection in this block, paired with its dotted key (SC#6/SC#4 accounting walk).
    pub(crate) fn projections<'a>(
        &'a self,
        prefix: &str,
    ) -> Vec<(String, &'a crate::quant::QLinear)> {
        let mut v = self.attn.projections(&join(prefix, "attn"));
        v.extend(self.mlp.projections(&join(prefix, "ff")));
        v
    }

    pub fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let x = (x + self.attn.forward(&self.prenorm.forward(x)?, None)?)?;
        &x + self.mlp.forward(&self.postnorm.forward(&x)?)?
    }

    /// Visit the block's attention + SwiGLU projections under `{prefix}.attn` / `{prefix}.ff` — sc-11105.
    pub fn visit_adaptable_mut(
        &mut self,
        prefix: &str,
        f: &mut dyn FnMut(&str, &mut AdaptLinear) -> candle_gen::Result<()>,
    ) -> candle_gen::Result<()> {
        self.attn.visit_adaptable_mut(&join(prefix, "attn"), f)?;
        self.mlp.visit_adaptable_mut(&join(prefix, "ff"), f)?;
        Ok(())
    }
}

// ── DoubleSharedModulation single-stream block (reference `SingleStreamBlock`) ──────────────
/// `mod(tvec) = tvec + scale_shift_table` → 6 chunks `(prescale, preshift, pregate, postscale,
/// postshift, postgate)`; then
/// `x += pregate · attn((1+prescale)·prenorm(x) + preshift)` and
/// `x += postgate · mlp((1+postscale)·postnorm(x) + postshift)`. Gates are raw (no activation).
pub struct SingleStreamBlock {
    scale_shift_table: Tensor, // [1, 1, 6·hidden]
    prenorm: RmsScale,
    postnorm: RmsScale,
    attn: GatedAttention,
    mlp: SwiGlu,
}

impl SingleStreamBlock {
    #[allow(clippy::too_many_arguments)]
    pub fn load(
        w: &Weights,
        prefix: &str,
        heads: usize,
        kv_heads: usize,
        head_dim: usize,
        hidden: usize,
        eps: f64,
    ) -> Result<Self> {
        Self::load_planned(
            w,
            prefix,
            heads,
            kv_heads,
            head_dim,
            hidden,
            eps,
            &DitPlan::baseline(),
        )
    }

    /// [`Self::load`] under a [`DitPlan`] (sc-12110). **This is the compute bulk SC#1 rides on** — 28 of
    /// these at hidden 6144 / intermediate 16384, ~491M params each, 100% linear GEMM and zero `Conv2d`.
    ///
    /// Note the attention here reads the **combined `[ctx ; img]`** sequence: Krea is a single-stream
    /// DiT, so caption-derived tokens flow through the same self-attention as image tokens rather than
    /// through a separate cross-attention block. Whether that puts the caption's massive activations
    /// into the compute bulk's quantization blocks is exactly the question
    /// `nvfp4_krea_dit_real_activation_outlier_sparsity` measures rather than assumes.
    #[allow(clippy::too_many_arguments)]
    pub fn load_planned(
        w: &Weights,
        prefix: &str,
        heads: usize,
        kv_heads: usize,
        head_dim: usize,
        hidden: usize,
        eps: f64,
        plan: &DitPlan,
    ) -> Result<Self> {
        // Stored `[6, hidden]`; flatten row-major to `[1, 1, 6·hidden]` so a single broadcast-add onto
        // `tvec` (`[b, 1, 6·hidden]`) and a 6-way split reproduce the reference's `chunk(6, -1)` order.
        let sst = w
            .get(&join(prefix, "scale_shift_table"))?
            .reshape((1, 1, 6 * hidden))?;
        Ok(Self {
            scale_shift_table: sst,
            prenorm: RmsScale::load(w, &join(prefix, "norm1.weight"), eps)?,
            postnorm: RmsScale::load(w, &join(prefix, "norm2.weight"), eps)?,
            attn: GatedAttention::load_planned(
                w,
                &join(prefix, "attn"),
                heads,
                kv_heads,
                head_dim,
                eps,
                plan,
            )?,
            mlp: SwiGlu::load_planned(w, &join(prefix, "ff"), plan)?,
        })
    }

    /// Every projection in this block, paired with its dotted key (SC#6/SC#4 accounting walk).
    pub(crate) fn projections<'a>(
        &'a self,
        prefix: &str,
    ) -> Vec<(String, &'a crate::quant::QLinear)> {
        let mut v = self.attn.projections(&join(prefix, "attn"));
        v.extend(self.mlp.projections(&join(prefix, "ff")));
        v
    }

    /// `x`: `[b, s, hidden]`, `tvec`: `[b, 1, 6·hidden]` (shared `time_mod_proj` output), `cos`/`sin`:
    /// `[1, s, head_dim/2]`.
    pub fn forward(&self, x: &Tensor, tvec: &Tensor, cos: &Tensor, sin: &Tensor) -> Result<Tensor> {
        let m = tvec.broadcast_add(&self.scale_shift_table)?; // [b, 1, 6·hidden]
        let chunks = m.chunk(6, D::Minus1)?; // 6 × [b, 1, hidden]
        let (prescale, preshift, pregate) = (&chunks[0], &chunks[1], &chunks[2]);
        let (postscale, postshift, postgate) = (&chunks[3], &chunks[4], &chunks[5]);

        let pre = self
            .prenorm
            .forward(x)?
            .broadcast_mul(&(prescale + 1.0)?)?
            .broadcast_add(preshift)?;
        let attn = self.attn.forward(&pre, Some((cos, sin)))?;
        let x = (x + attn.broadcast_mul(pregate)?)?;

        let post = self
            .postnorm
            .forward(&x)?
            .broadcast_mul(&(postscale + 1.0)?)?
            .broadcast_add(postshift)?;
        let mlp = self.mlp.forward(&post)?;
        &x + mlp.broadcast_mul(postgate)?
    }

    /// Visit the block's attention + SwiGLU projections under `{prefix}.attn` / `{prefix}.ff` — sc-11105.
    pub fn visit_adaptable_mut(
        &mut self,
        prefix: &str,
        f: &mut dyn FnMut(&str, &mut AdaptLinear) -> candle_gen::Result<()>,
    ) -> candle_gen::Result<()> {
        self.attn.visit_adaptable_mut(&join(prefix, "attn"), f)?;
        self.mlp.visit_adaptable_mut(&join(prefix, "ff"), f)?;
        Ok(())
    }
}

// ── TextFusionTransformer (reference `TextFusionTransformer`) ────────────────────────────────
/// Aggregates the `num_layers` stacked Qwen3-VL hidden states into one conditioning stream:
/// `layerwise_blocks` attend across the layer axis (per token) → `projector` collapses `num_layers→1`
/// → `refiner_blocks` attend across the token axis.
pub struct TextFusionTransformer {
    layerwise: Vec<TextFusionBlock>,
    projector: QLinear, // Linear(num_layers → 1), no bias — packed-detect for future-proofing (sc-9486)
    refiner: Vec<TextFusionBlock>,
}

impl TextFusionTransformer {
    #[allow(clippy::too_many_arguments)]
    pub fn load(
        w: &Weights,
        num_layerwise: usize,
        num_refiner: usize,
        heads: usize,
        kv_heads: usize,
        head_dim: usize,
        eps: f64,
    ) -> Result<Self> {
        Self::load_planned(
            w,
            num_layerwise,
            num_refiner,
            heads,
            kv_heads,
            head_dim,
            eps,
            &DitPlan::baseline(),
        )
    }

    /// [`Self::load`] under a [`DitPlan`] (sc-12110).
    ///
    /// The `projector` stays on the **baseline** leg in every plan: it is a `[1, num_layers]` collapse,
    /// so `N = 1` is ineligible for the cuBLASLt FP4 path (which needs `N % 16 == 0`) and an
    /// [`crate::quant::QLinear::Nvfp4`] there would silently resolve to the dequant→bf16 fallback and
    /// pad the report's `n_quantized` with a layer that never lights up. Excluding it keeps
    /// [`crate::nvfp4_dit::Nvfp4Report::fp4_lit_fraction`] an honest statement about the lane.
    #[allow(clippy::too_many_arguments)]
    pub fn load_planned(
        w: &Weights,
        num_layerwise: usize,
        num_refiner: usize,
        heads: usize,
        kv_heads: usize,
        head_dim: usize,
        eps: f64,
        plan: &DitPlan,
    ) -> Result<Self> {
        let block = |i: usize, kind: &str| {
            TextFusionBlock::load_planned(
                w,
                &format!("text_fusion.{kind}.{i}"),
                heads,
                kv_heads,
                head_dim,
                eps,
                plan,
            )
        };
        Ok(Self {
            layerwise: (0..num_layerwise)
                .map(|i| block(i, "layerwise_blocks"))
                .collect::<Result<_>>()?,
            projector: linear_detect(w, "text_fusion.projector", false)?,
            refiner: (0..num_refiner)
                .map(|i| block(i, "refiner_blocks"))
                .collect::<Result<_>>()?,
        })
    }

    /// Every quantizable projection in the text-fusion stack, paired with its dotted key (SC#6/SC#4
    /// accounting walk). The `projector` is excluded for the reason [`Self::load_planned`] documents.
    pub(crate) fn projections(&self) -> Vec<(String, &crate::quant::QLinear)> {
        let mut v = Vec::new();
        for (i, b) in self.layerwise.iter().enumerate() {
            v.extend(b.projections(&format!("text_fusion.layerwise_blocks.{i}")));
        }
        for (i, b) in self.refiner.iter().enumerate() {
            v.extend(b.projections(&format!("text_fusion.refiner_blocks.{i}")));
        }
        v
    }

    /// `x`: `[b, n_tokens, num_layers, txt_dim]` (the stacked select-layer hidden states). Returns the
    /// fused conditioning `[b, n_tokens, txt_dim]`.
    pub fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let (b, n_tok, n_layers, d) = x.dims4()?;

        // Layerwise attention: each token's `num_layers` stack is a sequence (batch = b·n_tokens).
        let mut h = x.reshape((b * n_tok, n_layers, d))?;
        for blk in &self.layerwise {
            h = blk.forward(&h)?;
        }

        // `(b n_tok) n_layers d -> b n_tok d n_layers`, project `num_layers → 1`, drop the axis.
        let h = h
            .reshape((b, n_tok, n_layers, d))?
            .permute((0, 1, 3, 2))? // [b, n_tok, d, n_layers]
            .contiguous()?;
        let h = self
            .projector
            .forward(&h.reshape((b * n_tok * d, n_layers))?)?; // [b·n_tok·d, 1]
        let mut h = h.reshape((b, n_tok, d))?;

        // Token-axis refinement.
        for blk in &self.refiner {
            h = blk.forward(&h)?;
        }
        Ok(h)
    }

    /// Visit the text-fusion blocks' adaptable projections under `text_fusion.layerwise_blocks.{i}` /
    /// `text_fusion.refiner_blocks.{i}` (sc-11105) — the adapter surface. The `projector` (num_layers→1)
    /// stays out of the surface (matching `merge_surface_keys`).
    pub fn visit_adaptable_mut(
        &mut self,
        f: &mut dyn FnMut(&str, &mut AdaptLinear) -> candle_gen::Result<()>,
    ) -> candle_gen::Result<()> {
        for (i, blk) in self.layerwise.iter_mut().enumerate() {
            blk.visit_adaptable_mut(&format!("text_fusion.layerwise_blocks.{i}"), f)?;
        }
        for (i, blk) in self.refiner.iter_mut().enumerate() {
            blk.visit_adaptable_mut(&format!("text_fusion.refiner_blocks.{i}"), f)?;
        }
        Ok(())
    }
}
