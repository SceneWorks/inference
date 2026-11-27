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
use candle_gen::modulate::{adaln, AdaLn};

use super::rope::KREA_ROPE;
use crate::adapters::PhaseAdapterSelection;
use crate::loader::{linear_detect, linear_detect_planned, rms_scale, rms_scale_weight, Weights};
use crate::nvfp4_dit::{BlockLeaf, DitPlan};
use crate::quant::QLinear;
use candle_gen::attention::repeat_kv_bshd;
#[cfg(test)]
use candle_gen::gen_core::attention_budget::AttentionBudget;
use candle_gen::gen_core::attention_budget::AttentionPlan;

fn plan_from_budget(budget: usize) -> AttentionPlan<'static> {
    AttentionPlan::budgeted(candle_gen::attention::attention_budget_from_usize(budget))
}

fn into_candle_core(result: candle_gen::Result<Tensor>) -> Result<Tensor> {
    result.map_err(|error| match error {
        candle_gen::CandleError::Candle(error) => error,
        other => candle_gen::candle_core::Error::Msg(other.to_string()),
    })
}

/// Join a module prefix with a leaf name.
fn join(prefix: &str, name: &str) -> String {
    if prefix.is_empty() {
        name.to_string()
    } else {
        format!("{prefix}.{name}")
    }
}

/// Bidirectional, unmasked scaled-dot-product attention. `q`/`k`/`v`: `[b, h, s, hd]`.
///
/// i32-overflow guard (sc-9116): the image-token scores `[b, h, s, s]` reach `48·16384² ≈ 12.9e9 >
/// i32::MAX` at a 2048² render, silently corrupting the tail rows on the candle CUDA kernels. On the
/// constrained-memory rung, complete heads are chunked first. Keeping the full query axis
/// preserves the exact CUDA GEMM shape (and therefore byte identity) at 1024². If one head alone still
/// exceeds the budget (the 2048² overflow case), the shared helper additionally chunks query rows.
#[cfg(test)]
fn sdpa(q: &Tensor, k: &Tensor, v: &Tensor, scale: f64, scores_budget: usize) -> Result<Tensor> {
    into_candle_core(sdpa_planned(
        q,
        k,
        v,
        scale,
        plan_from_budget(scores_budget),
    ))
}

fn sdpa_planned(
    q: &Tensor,
    k: &Tensor,
    v: &Tensor,
    scale: f64,
    plan: AttentionPlan<'_>,
) -> candle_gen::Result<Tensor> {
    let (b, h, sq, _d) = q.dims4()?;
    let sk = k.dim(2)?;
    let head_plan = plan
        .budget
        .head_chunks(b as u64, h as u64, sq as u64, sk as u64);
    // A fused kernel (flash, or the opt-in INT8 one) never materializes the scores the head chunks
    // bound, so a call the shared router sends to it runs as one launch rather than one per head
    // chunk (SW-6130, SW-6131).
    let fused = candle_gen::fused_attention::route_attention_kernel(q, k, v, None).is_fused();
    if fused || !head_plan.chunks_heads() {
        record_head_chunk_count(1);
        return candle_gen::sdpa_planned_bhsd(q, k, v, scale, None, softmax_last_dim, plan);
    }

    // Pure shared-planner delegation above: this site intentionally owns no budget arithmetic. Head
    // chunks preserve the full query GEMM shape and can reconstruct bit-exactly; the shared query-row
    // kernel below is only a fallback when one planned head still exceeds the budget, and is compared
    // with tolerance because narrowing query rows changes the GEMM M dimension.
    let heads_per_chunk = head_plan.heads_per_chunk() as usize;
    let mut chunks = Vec::new();
    let mut start = 0;
    while start < h {
        if plan.is_cancelled() {
            return Err(candle_gen::CandleError::Canceled);
        }
        let len = heads_per_chunk.min(h - start);
        chunks.push(candle_gen::sdpa_planned_bhsd(
            &q.narrow(1, start, len)?,
            &k.narrow(1, start, len)?,
            &v.narrow(1, start, len)?,
            scale,
            None,
            softmax_last_dim,
            plan,
        )?);
        start += len;
    }
    record_head_chunk_count(chunks.len());
    Ok(Tensor::cat(&chunks, 1)?)
}

#[cfg(test)]
mod head_chunk_probe {
    use std::sync::atomic::{AtomicUsize, Ordering};

    static LAST_CHUNK_COUNT: AtomicUsize = AtomicUsize::new(0);

    pub fn last_chunk_count() -> usize {
        LAST_CHUNK_COUNT.load(Ordering::Relaxed)
    }

    pub fn reset() {
        LAST_CHUNK_COUNT.store(0, Ordering::Relaxed);
    }

    pub(super) fn record(n: usize) {
        LAST_CHUNK_COUNT.store(n, Ordering::Relaxed);
    }
}

#[cfg(test)]
fn record_head_chunk_count(n: usize) {
    head_chunk_probe::record(n);
}

#[cfg(not(test))]
#[inline(always)]
fn record_head_chunk_count(_n: usize) {}

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
        __model_proof_output!(rms_scale(x, &self.weight, self.eps))
    }

    /// The block / final-layer AdaLN, `forward(x) · (1 + scale) + shift`, through the shared
    /// fused AdaLN (SW-6127): one launch on CUDA, bit-identical to that op chain (which it runs
    /// everywhere else, and for a gradient-tracked operand). `scale` / `shift` are `[b, 1, hidden]`
    /// in `x`'s dtype.
    pub fn modulate(&self, x: &Tensor, scale: &Tensor, shift: &Tensor) -> Result<Tensor> {
        let spec = AdaLn::rms(self.eps).with_weight(&self.weight).norm_in_f32();
        #[cfg(test)]
        let spec = adaln_mutation::apply(spec);
        __model_proof_output!(adaln(x, &spec, scale, shift))
    }

    /// This norm followed by Krea's interleaved RoPE (`cos`/`sin` `[1, s, head_dim/2]`) on a
    /// per-head `[b, s, heads, head_dim]` q or k — the shared fused QK-RMSNorm + RoPE (SW-6128):
    /// one launch on CUDA that writes the result into `x`'s own buffer, the `rms_scale` → RoPE
    /// chain everywhere else (bit-identical either way). `x` must be a tensor this attention owns
    /// (a fresh projection output): its buffer is overwritten on the fused path.
    pub fn forward_rope_in_place(&self, x: &Tensor, cos: &Tensor, sin: &Tensor) -> Result<Tensor> {
        __model_proof_output!(candle_gen::rope::rms_norm_rope_in_place(
            x,
            &self.weight,
            self.eps,
            cos,
            sin,
            &KREA_ROPE,
        ))
    }

    pub(crate) fn move_to_device(
        &mut self,
        device: &candle_gen::candle_core::Device,
    ) -> Result<()> {
        self.weight = self.weight.to_device(device)?;
        Ok(())
    }
}

/// Test-only mutation switch for the AdaLN parity proof (SW-6127): with it set on this thread,
/// [`RmsScale::modulate`] drops the `+ 1` on the scale, which the block parity check must catch.
#[cfg(test)]
pub(crate) mod adaln_mutation {
    use std::cell::Cell;

    thread_local! {
        static DROP_ONE_PLUS: Cell<bool> = const { Cell::new(false) };
    }

    /// Only the CUDA block proof flips it.
    #[cfg(feature = "cuda")]
    pub(crate) fn set_drop_one_plus(on: bool) {
        DROP_ONE_PLUS.with(|c| c.set(on));
    }

    pub(super) fn apply(spec: candle_gen::modulate::AdaLn<'_>) -> candle_gen::modulate::AdaLn<'_> {
        if DROP_ONE_PLUS.with(Cell::get) {
            spec.direct_scale()
        } else {
            spec
        }
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
        // sc-12121: the leaf strings come from the role table itself (`BlockLeaf::module_leaf`), so
        // the keys this loader creates and the keys `KreaSite::classify` recognises are literally the
        // same literals. Editing one without the other is impossible rather than merely tested.
        let p = |leaf: BlockLeaf| {
            linear_detect_planned(w, &join(prefix, leaf.module_leaf()), false, plan)
        };
        Ok(Self {
            q: p(BlockLeaf::AttnQ)?,
            k: p(BlockLeaf::AttnK)?,
            v: p(BlockLeaf::AttnV)?,
            gate: p(BlockLeaf::AttnGate)?,
            o: p(BlockLeaf::AttnOut)?,
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
            (BlockLeaf::AttnQ, &self.q),
            (BlockLeaf::AttnK, &self.k),
            (BlockLeaf::AttnV, &self.v),
            (BlockLeaf::AttnGate, &self.gate),
            (BlockLeaf::AttnOut, &self.o),
        ]
        .into_iter()
        .map(|(leaf, p)| (join(prefix, leaf.module_leaf()), p))
        .collect()
    }

    /// `x`: `[b, s, hidden]`. `rope`: `Some((cos, sin))` (`[1, s, head_dim/2]`) for the single-stream
    /// blocks; `None` for the text-fusion blocks (no positional encoding). Unmasked (B=1 full sequence).
    pub fn forward(&self, x: &Tensor, rope: Option<(&Tensor, &Tensor)>) -> Result<Tensor> {
        self.forward_with_attention_budget(x, rope, candle_gen::ATTN_SCORES_BUDGET)
    }

    /// [`Self::forward`] with an explicit score element budget. Lowering the budget chunks complete
    /// heads first, preserving the full query/key axes and the exact attention result at 1024².
    pub fn forward_with_attention_budget(
        &self,
        x: &Tensor,
        rope: Option<(&Tensor, &Tensor)>,
        scores_budget: usize,
    ) -> Result<Tensor> {
        into_candle_core(self.forward_with_attention_plan(x, rope, plan_from_budget(scores_budget)))
    }

    /// Request-scoped attention forward. A bounded plan checks cancellation between both the
    /// head-chunks used by Krea at 1024² and any query-row chunks used by the shared kernel.
    pub fn forward_with_attention_plan(
        &self,
        x: &Tensor,
        rope: Option<(&Tensor, &Tensor)>,
        attention_plan: AttentionPlan<'_>,
    ) -> candle_gen::Result<Tensor> {
        self.forward_with_attention_plan_and_pag(x, rope, attention_plan, None)
    }

    /// Request-scoped attention forward with optional Perturbed-Attention Guidance (PAG, SW-7176).
    ///
    /// When `perturb_after_prefix` is `Some(prefix_len)`, prefix queries `0..prefix_len` (`[ctx]`
    /// in T2I/img2img or `[ctx ; ref_latents…]` in Kontext edit) execute normal GQA attention over
    /// the full `[0..s]` key/value sequence, while target image tokens `prefix_len..s` replace
    /// their self-attention output with their own GQA-expanded value vectors `V_target` prior to
    /// `sigmoid(to_gate)` and `to_out`. When `None`, executes the unperturbed baseline unchanged.
    pub fn forward_with_attention_plan_and_pag(
        &self,
        x: &Tensor,
        rope: Option<(&Tensor, &Tensor)>,
        attention_plan: AttentionPlan<'_>,
        perturb_after_prefix: Option<usize>,
    ) -> candle_gen::Result<Tensor> {
        self.forward_with_attention_and_phase(x, rope, attention_plan, perturb_after_prefix, None)
    }

    fn forward_with_attention_and_phase(
        &self,
        x: &Tensor,
        rope: Option<(&Tensor, &Tensor)>,
        attention_plan: AttentionPlan<'_>,
        perturb_after_prefix: Option<usize>,
        phase: Option<&PhaseAdapterSelection<'_>>,
    ) -> candle_gen::Result<Tensor> {
        let (b, s, _) = x.dims3()?;
        let (nh, nkv, hd) = (self.heads, self.kv_heads, self.head_dim);

        let q = self.q.forward_phase(x, phase)?.reshape((b, s, nh, hd))?;
        let k = self.k.forward_phase(x, phase)?.reshape((b, s, nkv, hd))?;
        let v = self.v.forward_phase(x, phase)?.reshape((b, s, nkv, hd))?;
        let gate = self.gate.forward_phase(x, phase)?; // [b, s, hidden]

        // QK-norm (+ RoPE on the single-stream blocks). q and k are this call's own projection
        // outputs, so the fused QK-norm + RoPE may rotate them in place.
        let (q, k) = match rope {
            Some((cos, sin)) => (
                self.norm_q.forward_rope_in_place(&q, cos, sin)?,
                self.norm_k.forward_rope_in_place(&k, cos, sin)?,
            ),
            None => (self.norm_q.forward(&q)?, self.norm_k.forward(&k)?),
        };

        let groups = nh / nkv;
        let k = repeat_kv_bshd(&k, groups)?;
        let v = repeat_kv_bshd(&v, groups)?;

        let q = q.transpose(1, 2)?;
        let k = k.transpose(1, 2)?;
        let v = v.transpose(1, 2)?;
        let o = match perturb_after_prefix {
            Some(prefix_len) => {
                pag_attention_bhsd(&q, &k, &v, self.scale, prefix_len, attention_plan)?
            }
            None => sdpa_planned(&q, &k, &v, self.scale, attention_plan)?,
        };
        let o = o.transpose(1, 2)?.contiguous()?.reshape((b, s, nh * hd))?;

        // Sigmoid gate the attention output, then the shared output projection.
        let gated = (o * sigmoid(&gate)?)?;
        Ok(self.o.forward_phase(&gated, phase)?)
    }

    /// Visit the five gated-attention projections under `{prefix}` (`to_q/to_k/to_v/to_gate/to_out.0`) —
    /// the surface a user LoRA/LoKr adapts (sc-11105). The q/k RMS scales are not projections. Dense,
    /// packed, and int8-ConvRot projections all host additive residuals.
    pub fn visit_adaptable_mut(
        &mut self,
        prefix: &str,
        f: &mut dyn FnMut(&str, &mut QLinear) -> candle_gen::Result<()>,
    ) -> candle_gen::Result<()> {
        for (leaf, proj) in [
            ("to_q", &mut self.q),
            ("to_k", &mut self.k),
            ("to_v", &mut self.v),
            ("to_gate", &mut self.gate),
            ("to_out.0", &mut self.o),
        ] {
            if let Some(a) = proj.as_additive_mut() {
                f(&join(prefix, leaf), a)?;
            }
        }
        Ok(())
    }

    fn quantize_onto(
        &mut self,
        quant: candle_gen::gen_core::Quant,
        device: &candle_gen::candle_core::Device,
    ) -> Result<()> {
        for projection in [
            &mut self.q,
            &mut self.k,
            &mut self.v,
            &mut self.gate,
            &mut self.o,
        ] {
            projection.quantize_onto(quant, device)?;
        }
        self.norm_q.move_to_device(device)?;
        self.norm_k.move_to_device(device)
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
        // sc-12121: leaf strings read off the role table — see `GatedAttention::load_planned`.
        let p = |leaf: BlockLeaf| {
            linear_detect_planned(w, &join(prefix, leaf.module_leaf()), false, plan)
        };
        Ok(Self {
            gate: p(BlockLeaf::FfGate)?,
            up: p(BlockLeaf::FfUp)?,
            down: p(BlockLeaf::FfDown)?,
        })
    }

    /// Every projection in this SwiGLU, paired with its dotted key (SC#6/SC#4 accounting walk).
    pub(crate) fn projections<'a>(
        &'a self,
        prefix: &str,
    ) -> Vec<(String, &'a crate::quant::QLinear)> {
        [
            (BlockLeaf::FfGate, &self.gate),
            (BlockLeaf::FfUp, &self.up),
            (BlockLeaf::FfDown, &self.down),
        ]
        .into_iter()
        .map(|(leaf, p)| (join(prefix, leaf.module_leaf()), p))
        .collect()
    }

    pub fn forward(&self, x: &Tensor) -> Result<Tensor> {
        self.forward_phase(x, None)
    }

    fn forward_phase(
        &self,
        x: &Tensor,
        phase: Option<&PhaseAdapterSelection<'_>>,
    ) -> Result<Tensor> {
        let gated =
            (self.gate.forward_phase(x, phase)?.silu()? * self.up.forward_phase(x, phase)?)?;
        self.down.forward_phase(&gated, phase)
    }

    /// Visit the three SwiGLU projections under `{prefix}` (`gate/up/down`) — sc-11105.
    pub fn visit_adaptable_mut(
        &mut self,
        prefix: &str,
        f: &mut dyn FnMut(&str, &mut QLinear) -> candle_gen::Result<()>,
    ) -> candle_gen::Result<()> {
        for (leaf, proj) in [
            ("gate", &mut self.gate),
            ("up", &mut self.up),
            ("down", &mut self.down),
        ] {
            if let Some(a) = proj.as_additive_mut() {
                f(&join(prefix, leaf), a)?;
            }
        }
        Ok(())
    }

    pub(crate) fn quantize_onto(
        &mut self,
        quant: candle_gen::gen_core::Quant,
        device: &candle_gen::candle_core::Device,
    ) -> Result<()> {
        self.gate.quantize_onto(quant, device)?;
        self.up.quantize_onto(quant, device)?;
        self.down.quantize_onto(quant, device)
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
                &join(prefix, BlockLeaf::AttnQ.module()),
                heads,
                kv_heads,
                head_dim,
                eps,
                plan,
            )?,
            mlp: SwiGlu::load_planned(w, &join(prefix, BlockLeaf::FfGate.module()), plan)?,
        })
    }

    /// Every projection in this block, paired with its dotted key (SC#6/SC#4 accounting walk).
    pub(crate) fn projections<'a>(
        &'a self,
        prefix: &str,
    ) -> Vec<(String, &'a crate::quant::QLinear)> {
        let mut v = self
            .attn
            .projections(&join(prefix, BlockLeaf::AttnQ.module()));
        v.extend(
            self.mlp
                .projections(&join(prefix, BlockLeaf::FfGate.module())),
        );
        v
    }

    pub fn forward(&self, x: &Tensor) -> Result<Tensor> {
        self.forward_phase(x, None)
    }

    fn forward_phase(
        &self,
        x: &Tensor,
        phase: Option<&PhaseAdapterSelection<'_>>,
    ) -> Result<Tensor> {
        let attn = into_candle_core(self.attn.forward_with_attention_and_phase(
            &self.prenorm.forward(x)?,
            None,
            plan_from_budget(candle_gen::ATTN_SCORES_BUDGET),
            None,
            phase,
        ))?;
        let x = (x + attn)?;
        &x + self.mlp.forward_phase(&self.postnorm.forward(&x)?, phase)?
    }

    /// Visit the block's attention + SwiGLU projections under `{prefix}.attn` / `{prefix}.ff` — sc-11105.
    pub fn visit_adaptable_mut(
        &mut self,
        prefix: &str,
        f: &mut dyn FnMut(&str, &mut QLinear) -> candle_gen::Result<()>,
    ) -> candle_gen::Result<()> {
        self.attn
            .visit_adaptable_mut(&join(prefix, BlockLeaf::AttnQ.module()), f)?;
        self.mlp
            .visit_adaptable_mut(&join(prefix, BlockLeaf::FfGate.module()), f)?;
        Ok(())
    }

    fn quantize_onto(
        &mut self,
        quant: candle_gen::gen_core::Quant,
        device: &candle_gen::candle_core::Device,
    ) -> Result<()> {
        self.prenorm.move_to_device(device)?;
        self.postnorm.move_to_device(device)?;
        self.attn.quantize_onto(quant, device)?;
        self.mlp.quantize_onto(quant, device)
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

/// Perturbed-Attention Guidance (PAG) over already GQA-expanded `[B, H, S, D]` tensors (SW-7176).
///
/// Prefix queries `0..prefix_len` (`[ctx]` or `[ctx ; ref_latents…]`) execute normal planned SDPA
/// (preserving head/query chunking and CUDA FlashAttention contiguity) against all keys and values
/// `0..S`; target image tokens `prefix_len..S` bypass `softmax(Q Kᵀ)` and receive identity
/// self-attention `V_target = v[:, :, prefix_len.., :]`.
pub(crate) fn pag_attention_bhsd(
    q: &Tensor,
    k: &Tensor,
    v: &Tensor,
    scale: f64,
    prefix_len: usize,
    attention_plan: AttentionPlan<'_>,
) -> candle_gen::Result<Tensor> {
    let s = q.dim(2)?;
    if prefix_len >= s {
        return Err(candle_gen::CandleError::Msg(format!(
            "krea PAG: prefix_len {prefix_len} out of bounds for sequence length {s} (target slice must be non-empty)"
        )));
    }
    if prefix_len == 0 {
        return Ok(v.clone());
    }
    let q_prefix = q.narrow(2, 0, prefix_len)?.contiguous()?;
    let o_prefix = sdpa_planned(&q_prefix, k, v, scale, attention_plan)?;
    let v_target = v.narrow(2, prefix_len, s - prefix_len)?;
    Ok(Tensor::cat(&[&o_prefix, &v_target], 2)?)
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
                &join(prefix, BlockLeaf::AttnQ.module()),
                heads,
                kv_heads,
                head_dim,
                eps,
                plan,
            )?,
            mlp: SwiGlu::load_planned(w, &join(prefix, BlockLeaf::FfGate.module()), plan)?,
        })
    }

    /// Every projection in this block, paired with its dotted key (SC#6/SC#4 accounting walk).
    pub(crate) fn projections<'a>(
        &'a self,
        prefix: &str,
    ) -> Vec<(String, &'a crate::quant::QLinear)> {
        let mut v = self
            .attn
            .projections(&join(prefix, BlockLeaf::AttnQ.module()));
        v.extend(
            self.mlp
                .projections(&join(prefix, BlockLeaf::FfGate.module())),
        );
        v
    }

    /// `x`: `[b, s, hidden]`, `tvec`: `[b, 1, 6·hidden]` (shared `time_mod_proj` output), `cos`/`sin`:
    /// `[1, s, head_dim/2]`.
    pub fn forward(&self, x: &Tensor, tvec: &Tensor, cos: &Tensor, sin: &Tensor) -> Result<Tensor> {
        self.forward_with_attention_budget(x, tvec, cos, sin, candle_gen::ATTN_SCORES_BUDGET)
    }

    /// [`Self::forward`] with an explicit attention-score budget for the joint attention. All
    /// modulation, residual, and MLP math remains on the existing path.
    pub fn forward_with_attention_budget(
        &self,
        x: &Tensor,
        tvec: &Tensor,
        cos: &Tensor,
        sin: &Tensor,
        scores_budget: usize,
    ) -> Result<Tensor> {
        into_candle_core(self.forward_with_attention_plan(
            x,
            tvec,
            cos,
            sin,
            plan_from_budget(scores_budget),
        ))
    }

    /// Request-scoped variant of [`Self::forward_with_attention_budget`].
    pub fn forward_with_attention_plan(
        &self,
        x: &Tensor,
        tvec: &Tensor,
        cos: &Tensor,
        sin: &Tensor,
        attention_plan: AttentionPlan<'_>,
    ) -> candle_gen::Result<Tensor> {
        self.forward_with_attention_plan_and_pag(x, tvec, cos, sin, attention_plan, None)
    }

    /// Request-scoped variant with optional PAG target-token perturbation (SW-7176).
    pub fn forward_with_attention_plan_and_pag(
        &self,
        x: &Tensor,
        tvec: &Tensor,
        cos: &Tensor,
        sin: &Tensor,
        attention_plan: AttentionPlan<'_>,
        perturb_after_prefix: Option<usize>,
    ) -> candle_gen::Result<Tensor> {
        self.forward_with_attention_and_phase(
            x,
            tvec,
            cos,
            sin,
            attention_plan,
            perturb_after_prefix,
            None,
        )
    }

    #[allow(clippy::too_many_arguments)]
    pub(crate) fn forward_with_attention_and_phase(
        &self,
        x: &Tensor,
        tvec: &Tensor,
        cos: &Tensor,
        sin: &Tensor,
        attention_plan: AttentionPlan<'_>,
        perturb_after_prefix: Option<usize>,
        phase: Option<&PhaseAdapterSelection<'_>>,
    ) -> candle_gen::Result<Tensor> {
        let m = tvec.broadcast_add(&self.scale_shift_table)?; // [b, 1, 6·hidden]
        let chunks = m.chunk(6, D::Minus1)?; // 6 × [b, 1, hidden]
        let (prescale, preshift, pregate) = (&chunks[0], &chunks[1], &chunks[2]);
        let (postscale, postshift, postgate) = (&chunks[3], &chunks[4], &chunks[5]);

        let pre = self.prenorm.modulate(x, prescale, preshift)?;
        let attn = self.attn.forward_with_attention_and_phase(
            &pre,
            Some((cos, sin)),
            attention_plan,
            perturb_after_prefix,
            phase,
        )?;
        let x = (x + attn.broadcast_mul(pregate)?)?;

        let post = self.postnorm.modulate(&x, postscale, postshift)?;
        let mlp = self.mlp.forward_phase(&post, phase)?;
        Ok((&x + mlp.broadcast_mul(postgate)?)?)
    }

    /// Visit the block's attention + SwiGLU projections under `{prefix}.attn` / `{prefix}.ff` — sc-11105.
    pub fn visit_adaptable_mut(
        &mut self,
        prefix: &str,
        f: &mut dyn FnMut(&str, &mut QLinear) -> candle_gen::Result<()>,
    ) -> candle_gen::Result<()> {
        self.attn
            .visit_adaptable_mut(&join(prefix, BlockLeaf::AttnQ.module()), f)?;
        self.mlp
            .visit_adaptable_mut(&join(prefix, BlockLeaf::FfGate.module()), f)?;
        Ok(())
    }

    pub(crate) fn quantize_onto(
        &mut self,
        quant: candle_gen::gen_core::Quant,
        device: &candle_gen::candle_core::Device,
    ) -> Result<()> {
        self.scale_shift_table = self.scale_shift_table.to_device(device)?;
        self.prenorm.move_to_device(device)?;
        self.postnorm.move_to_device(device)?;
        self.attn.quantize_onto(quant, device)?;
        self.mlp.quantize_onto(quant, device)
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

    pub(crate) fn quantize_onto(
        &mut self,
        quant: candle_gen::gen_core::Quant,
        device: &candle_gen::candle_core::Device,
    ) -> Result<()> {
        for block in self.layerwise.iter_mut().chain(self.refiner.iter_mut()) {
            block.quantize_onto(quant, device)?;
        }
        self.projector.to_device(device)
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
        self.forward_phase(x, None)
    }

    pub(crate) fn forward_phase(
        &self,
        x: &Tensor,
        phase: Option<&PhaseAdapterSelection<'_>>,
    ) -> Result<Tensor> {
        let (b, n_tok, n_layers, d) = x.dims4()?;

        // Layerwise attention: each token's `num_layers` stack is a sequence (batch = b·n_tokens).
        let mut h = x.reshape((b * n_tok, n_layers, d))?;
        for blk in &self.layerwise {
            h = blk.forward_phase(&h, phase)?;
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
            h = blk.forward_phase(&h, phase)?;
        }
        Ok(h)
    }

    /// Visit the text-fusion blocks' adaptable projections under `text_fusion.layerwise_blocks.{i}` /
    /// `text_fusion.refiner_blocks.{i}` (sc-11105) — the adapter surface. The `projector` (num_layers→1)
    /// stays out of the surface (matching `merge_surface_keys`).
    pub fn visit_adaptable_mut(
        &mut self,
        f: &mut dyn FnMut(&str, &mut QLinear) -> candle_gen::Result<()>,
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

#[cfg(test)]
mod tests {
    use super::*;
    use candle_gen::candle_core::Device;

    fn reset_probes() {
        head_chunk_probe::reset();
        candle_gen::attention::chunk_probe::reset();
    }

    fn assert_chunks(heads: usize, queries: usize) {
        assert_eq!(head_chunk_probe::last_chunk_count(), heads);
        assert_eq!(
            candle_gen::attention::chunk_probe::last_chunk_count(),
            queries
        );
    }

    fn approx_eq(a: &Tensor, b: &Tensor) {
        let a = a.flatten_all().unwrap().to_vec1::<f32>().unwrap();
        let b = b.flatten_all().unwrap().to_vec1::<f32>().unwrap();
        assert_eq!(a.len(), b.len());
        for (x, y) in a.iter().zip(&b) {
            assert!((x - y).abs() < 1e-5, "{x} != {y} within 1e-5");
        }
    }

    /// The production 128 Mi boundary is literal and delegated to gen-core. Krea's native 1024² grid
    /// has 4096 image tokens and 48 DiT heads; the real-caption harness records about 4118 combined
    /// tokens. Both keep the full query axis. At the 2048² image-token floor, one head already exceeds
    /// the budget and the query fallback is required.
    #[test]
    fn production_head_boundaries_preserve_the_axis_distinction() {
        let budget = candle_gen::attention::AttentionBudget::from_score_elements(
            crate::pipeline::CONSTRAINED_ATTN_SCORES_BUDGET as u64,
            false,
        );
        let at_1024 = budget.head_chunks(1, 48, 4096, 4096);
        assert!(at_1024.chunks_heads());
        assert_eq!(at_1024.heads_per_chunk(), 8);
        assert_eq!(
            budget.query_block_rows(8 * 4096, 4096),
            4096,
            "the 1024² head-chunked path must leave the full query axis intact"
        );

        let with_caption = budget.head_chunks(1, 48, 4118, 4118);
        assert!(with_caption.chunks_heads());
        assert_eq!(with_caption.heads_per_chunk(), 7);
        assert_eq!(budget.query_block_rows(7 * 4118, 4118), 4118);

        let at_2048 = budget.head_chunks(1, 48, 16384, 16384);
        assert!(at_2048.chunks_heads());
        assert_eq!(at_2048.heads_per_chunk(), 1);
        assert_eq!(budget.query_block_rows(16384, 16384), 8192);
    }

    /// A scaled execution of the 1024² plan: two complete-head calls, each with one full-query call.
    /// The executed probes make this fail if either boundary is computed and ignored.
    #[test]
    fn head_chunking_executes_and_is_bit_exact_with_full_queries() {
        let device = Device::Cpu;
        let q = Tensor::randn(0f32, 1f32, (1, 4, 7, 4), &device).unwrap();
        let k = Tensor::randn(0f32, 1f32, (1, 4, 7, 4), &device).unwrap();
        let v = Tensor::randn(0f32, 1f32, (1, 4, 7, 4), &device).unwrap();
        reset_probes();
        let full = sdpa(&q, &k, &v, 0.5, usize::MAX).unwrap();
        assert_chunks(1, 1);
        reset_probes();
        let chunked = sdpa(&q, &k, &v, 0.5, 2 * 7 * 7).unwrap();
        assert_chunks(2, 1);
        let full = full.flatten_all().unwrap().to_vec1::<f32>().unwrap();
        let chunked = chunked.flatten_all().unwrap().to_vec1::<f32>().unwrap();
        assert_eq!(full, chunked);
    }

    #[test]
    fn planned_head_chunks_honor_typed_cancel_but_unbounded_ignores_it() {
        let device = Device::Cpu;
        let q = Tensor::randn(0f32, 1f32, (1, 2, 7, 4), &device).unwrap();
        let k = Tensor::randn(0f32, 1f32, (1, 2, 7, 4), &device).unwrap();
        let v = Tensor::randn(0f32, 1f32, (1, 2, 7, 4), &device).unwrap();
        let cancel = candle_gen::gen_core::CancelFlag::default();
        cancel.cancel();

        let bounded = AttentionPlan::budgeted(AttentionBudget::from_score_elements(7 * 7, false))
            .with_cancel(&cancel);
        let error = sdpa_planned(&q, &k, &v, 0.5, bounded).unwrap_err();
        assert!(matches!(error, candle_gen::CandleError::Canceled));

        let unbounded = AttentionPlan::UNBOUNDED.with_cancel(&cancel);
        assert!(sdpa_planned(&q, &k, &v, 0.5, unbounded).is_ok());
    }

    /// When one head exceeds the budget, the head path stays at one head per outer chunk and the
    /// shared query kernel runs four row chunks. Changing the query GEMM M dimension is only
    /// tolerance-equivalent, never asserted byte-exact (SC-15943).
    #[test]
    fn single_head_over_budget_executes_tolerant_query_fallback() {
        let device = Device::Cpu;
        let q = Tensor::randn(0f32, 1f32, (1, 2, 7, 4), &device).unwrap();
        let k = Tensor::randn(0f32, 1f32, (1, 2, 7, 4), &device).unwrap();
        let v = Tensor::randn(0f32, 1f32, (1, 2, 7, 4), &device).unwrap();
        reset_probes();
        let full = sdpa(&q, &k, &v, 0.5, usize::MAX).unwrap();
        assert_chunks(1, 1);
        reset_probes();
        let fallback = sdpa(&q, &k, &v, 0.5, 20).unwrap();
        assert_chunks(2, 4);
        approx_eq(&full, &fallback);
    }

    /// Both shared fast-path decisions execute one outer delegation. The single-head case remains
    /// distinguishable from a clamped head loop through `HeadChunkPlan::chunks_heads` and still
    /// exercises query fallback when its one head is over budget.
    #[test]
    fn in_budget_and_single_head_early_paths_are_executed() {
        let device = Device::Cpu;
        let q2 = Tensor::randn(0f32, 1f32, (1, 2, 7, 4), &device).unwrap();
        reset_probes();
        sdpa(&q2, &q2, &q2, 0.5, 2 * 7 * 7).unwrap();
        assert_chunks(1, 1);
        let in_budget =
            candle_gen::attention::AttentionBudget::from_score_elements((2 * 7 * 7) as u64, false)
                .head_chunks(1, 2, 7, 7);
        assert!(!in_budget.chunks_heads());
        assert_eq!(in_budget.heads_per_chunk(), 2);

        let q1 = Tensor::randn(0f32, 1f32, (1, 1, 7, 4), &device).unwrap();
        reset_probes();
        sdpa(&q1, &q1, &q1, 0.5, 20).unwrap();
        assert_chunks(1, 4);

        let budget = candle_gen::attention::AttentionBudget::from_score_elements(20, false);
        let one_head = budget.head_chunks(1, 1, 7, 7);
        assert!(!one_head.chunks_heads());
        assert_eq!(one_head.heads_per_chunk(), 1);
    }

    #[test]
    fn pag_attention_preserves_prefix_and_refs_while_replacing_target_with_expanded_v() {
        let device = Device::Cpu;
        // Krea 2's exact 48Q / 12KV GQA ratio (groups = 4), across T2I (0 refs) and 1–2 ref
        // Kontext edit.
        let (b, nh, nkv, hd) = (1usize, 48usize, 12usize, 4usize);
        let groups = nh / nkv;
        let cap_len = 3usize;
        let img_len = 4usize;

        for n_refs in [0usize, 1, 2] {
            let prefix_len = cap_len + n_refs * img_len;
            let s = prefix_len + img_len;
            let q = Tensor::randn(0f32, 1f32, (b, s, nh, hd), &device)
                .unwrap()
                .transpose(1, 2)
                .unwrap();
            let k_raw = Tensor::randn(0f32, 1f32, (b, s, nkv, hd), &device).unwrap();
            let mut v_vals = Vec::with_capacity(b * s * nkv * hd);
            for t in 0..s {
                for g in 0..nkv {
                    for d in 0..hd {
                        v_vals.push(100.0 * (t as f32) + 1.0 * (g as f32) + 0.1 * (d as f32));
                    }
                }
            }
            let v_raw = Tensor::from_vec(v_vals, (b, s, nkv, hd), &device).unwrap();
            let k = repeat_kv_bshd(&k_raw, groups)
                .unwrap()
                .transpose(1, 2)
                .unwrap();
            let v = repeat_kv_bshd(&v_raw, groups)
                .unwrap()
                .transpose(1, 2)
                .unwrap();

            let full = sdpa(&q, &k, &v, 0.5, usize::MAX).unwrap();
            let perturbed =
                pag_attention_bhsd(&q, &k, &v, 0.5, prefix_len, AttentionPlan::UNBOUNDED).unwrap();
            // Also exercise under a bounded head-chunking plan.
            let bounded = AttentionPlan::budgeted(AttentionBudget::from_score_elements(
                (8 * prefix_len * s) as u64,
                false,
            ));
            let perturbed_chunked =
                pag_attention_bhsd(&q, &k, &v, 0.5, prefix_len, bounded).unwrap();
            approx_eq(&perturbed, &perturbed_chunked);

            // Scramble Q_target and confirm output is invariant to Q_target.
            let q_prefix = q.narrow(2, 0, prefix_len).unwrap();
            let q_target_alt = Tensor::from_vec(
                vec![77.0f32; b * nh * img_len * hd],
                (b, nh, img_len, hd),
                &device,
            )
            .unwrap();
            let q_alt = Tensor::cat(&[&q_prefix, &q_target_alt], 2).unwrap();
            let perturbed_alt =
                pag_attention_bhsd(&q_alt, &k, &v, 0.5, prefix_len, AttentionPlan::UNBOUNDED)
                    .unwrap();
            assert_eq!(
                perturbed.flatten_all().unwrap().to_vec1::<f32>().unwrap(),
                perturbed_alt
                    .flatten_all()
                    .unwrap()
                    .to_vec1::<f32>()
                    .unwrap()
            );

            // 1) Prefix ([ctx] and [ref_latents...]) matches unperturbed SDPA.
            approx_eq(
                &perturbed.narrow(2, 0, prefix_len).unwrap(),
                &full.narrow(2, 0, prefix_len).unwrap(),
            );

            // 2) Target tokens equal expanded V_target with head h reading kv head h / 4.
            let pert_target = perturbed.narrow(2, prefix_len, img_len).unwrap();
            let expected_v_target = v.narrow(2, prefix_len, img_len).unwrap();
            assert_eq!(
                pert_target.flatten_all().unwrap().to_vec1::<f32>().unwrap(),
                expected_v_target
                    .flatten_all()
                    .unwrap()
                    .to_vec1::<f32>()
                    .unwrap()
            );
            let target_bshd = pert_target
                .transpose(1, 2)
                .unwrap()
                .contiguous()
                .unwrap()
                .flatten_all()
                .unwrap()
                .to_vec1::<f32>()
                .unwrap();
            for t_rel in 0..img_len {
                let t_abs = prefix_len + t_rel;
                for h in 0..nh {
                    let expected_kv_head = h / groups;
                    for d in 0..hd {
                        let idx = (t_rel * nh + h) * hd + d;
                        let want = 100.0 * (t_abs as f32)
                            + 1.0 * (expected_kv_head as f32)
                            + 0.1 * (d as f32);
                        assert!(
                            (target_bshd[idx] - want).abs() < 1e-5,
                            "head {h} must read kv head {expected_kv_head}: got {}, want {want}",
                            target_bshd[idx]
                        );
                    }
                }
            }

            // Out-of-bounds prefix_len >= s must fail closed.
            assert!(pag_attention_bhsd(&q, &k, &v, 0.5, s, AttentionPlan::UNBOUNDED).is_err());
        }
    }

    /// SW-7284 discriminator: one production-geometry Krea 2 single-stream DiT block (hidden 6144,
    /// GQA 48Q/12KV x 128, the 1024^2 joint sequence of 512 caption + 64x64 image tokens, real RoPE
    /// table) on synthetic weights, on CUDA. The f32 block on the materialized path is the reference;
    /// the bf16 block runs once pinned to the materialized path (the pre-SW-6130 route) and once
    /// through the shared router (FlashAttention-2). Compared on the block's residual update
    /// (`out - x`), which is what a denoise step integrates. Synthetic q/k norm scales near 2 make the
    /// logits sharp (std ~4), the regime where a softmax / accumulation defect would show.
    ///
    /// Declared criterion: flash is no worse than the old bf16 path, within a small margin —
    /// `flash_rel <= 1.1 * materialized_rel + 1e-3` and the same for max-abs (+1e-2).
    #[cfg(feature = "cuda")]
    #[test]
    #[ignore = "model inference; explicit model-proof workflow"]
    fn cuda_production_geometry_block_flash_matches_materialized() -> candle_gen::Result<()> {
        use crate::loader::Weights;
        use crate::testfix::attn_ffn;
        use crate::transformer::rope::RopeTables;
        use candle_gen::candle_core::DType;
        use candle_gen::fused_attention::{attention_path_counts, with_materialized_attention};
        use std::collections::HashMap;

        let dev = Device::new_cuda(0)?;
        let cfg = crate::config::Krea2Config::turbo();
        let (hidden, heads, kv, hd) = (
            cfg.hidden_size,
            cfg.num_attention_heads,
            cfg.num_kv_heads,
            cfg.attention_head_dim,
        );
        // The attention is the subject; a narrow FFN keeps the synthetic block cheap.
        let intermediate = 1024;
        let mut draw =
            |shape: &[usize]| Tensor::randn(0f32, 0.02f32, shape, &Device::Cpu).expect("cpu draw");
        let mut map: HashMap<String, Tensor> = HashMap::new();
        attn_ffn(
            &mut draw,
            &mut map,
            "b",
            hidden,
            heads,
            kv,
            hd,
            intermediate,
        );
        map.insert("b.scale_shift_table".into(), draw(&[6, hidden]));
        // Raw (centered) q/k RMS scales near 1, i.e. an effective scale near 2 after the folded +1.
        for key in ["b.attn.norm_q.weight", "b.attn.norm_k.weight"] {
            map.insert(key.into(), Tensor::randn(1f32, 0.3f32, hd, &Device::Cpu)?);
        }
        let tmp = tempfile::tempdir().map_err(|e| candle_gen::CandleError::Msg(e.to_string()))?;
        let path = tmp.path().join("block.safetensors");
        candle_gen::candle_core::safetensors::save(&map, &path)?;
        let load = |dtype: DType| -> candle_gen::Result<SingleStreamBlock> {
            let w = Weights::from_file(&path, &dev, dtype)?;
            Ok(SingleStreamBlock::load(
                &w,
                "b",
                heads,
                kv,
                hd,
                hidden,
                cfg.norm_eps,
            )?)
        };
        let (f32_block, bf16_block) = (load(DType::F32)?, load(DType::BF16)?);

        let (cap_len, side) = (512usize, 64usize);
        let s = cap_len + side * side;
        let rope = RopeTables::build_t2i(
            cap_len,
            side,
            side,
            cfg.axes_dims_rope,
            cfg.rope_theta as f64,
            &dev,
        )?;
        let (cos, sin) = rope.joint();
        let x = Tensor::randn(0f32, 1.0, (1, s, hidden), &dev)?;
        let tvec = Tensor::randn(0f32, 0.1, (1, 1, 6 * hidden), &dev)?;
        let plan = AttentionPlan::budgeted(AttentionBudget::from_score_elements(
            candle_gen::ATTN_SCORES_BUDGET as u64,
            false,
        ));

        let update =
            |block: &SingleStreamBlock, x: &Tensor, tvec: &Tensor| -> candle_gen::Result<Tensor> {
                let out = block.forward_with_attention_plan(x, tvec, &cos, &sin, plan)?;
                Ok((out.to_dtype(DType::F32)? - x.to_dtype(DType::F32)?)?)
            };
        let reference = with_materialized_attention(|| update(&f32_block, &x, &tvec))?;
        let (xb, tb) = (x.to_dtype(DType::BF16)?, tvec.to_dtype(DType::BF16)?);
        let materialized = with_materialized_attention(|| update(&bf16_block, &xb, &tb))?;
        let before = attention_path_counts();
        let fused = __model_proof_output!(update(&bf16_block, &xb, &tb))?;
        let routed = attention_path_counts().since(&before);
        assert!(
            routed.flash >= 1,
            "the bf16 block must take the fused kernel (needs candle-gen/flash-attn): {routed:?}"
        );

        let errors = |a: &Tensor| -> candle_gen::Result<(f32, f32)> {
            let d = (a - &reference)?;
            let max = d.abs()?.max_all()?.to_scalar::<f32>()?;
            let num = d.sqr()?.sum_all()?.sqrt()?.to_scalar::<f32>()?;
            let den = reference.sqr()?.sum_all()?.sqrt()?.to_scalar::<f32>()?;
            Ok((num / den.max(f32::MIN_POSITIVE), max))
        };
        let (fr, fm) = errors(&fused)?;
        let (mr, mm) = errors(&materialized)?;
        println!(
            "krea block (S={s}, {heads}Q/{kv}KV x {hd}, bf16) vs f32 materialized: flash rel-L2 \
             {fr:.4e} max|d| {fm:.4e}; bf16 materialized rel-L2 {mr:.4e} max|d| {mm:.4e}; \
             routed {routed:?}"
        );
        assert!(
            fr <= 1.1 * mr + 1e-3,
            "flash rel-L2 {fr} vs materialized {mr}"
        );
        assert!(
            fm <= 1.1 * mm + 1e-2,
            "flash max|d| {fm} vs materialized {mm}"
        );

        #[cfg(feature = "model-inference-proof")]
        println!("\nMODEL_INFERENCE_BODY_COMPLETED transformer::block::tests::cuda_production_geometry_block_flash_matches_materialized");
        Ok(())
    }

    /// SW-6127 block parity: one production-geometry Krea 2 single-stream DiT block (hidden 6144,
    /// 48Q/12KV x 128, the 1024^2 joint sequence of 512 caption + 64x64 image tokens, real RoPE
    /// table) on synthetic weights, bf16 on CUDA, with the fused AdaLN on and off (the op chain).
    /// Both block outputs must be **bit-identical** (the declared tolerance: the kernel is
    /// bit-identical to the chain per call, and everything else in the block is the same
    /// deterministic launch sequence), and the path marker must report `fused` for both
    /// modulations with the switch on and `disabled` with it off, at B = 1 and the B = 2 CFG batch.
    ///
    /// Mutation: the same forward with the kernel's `+ 1` on the scale dropped
    /// ([`adaln_mutation`]) must fail that parity.
    #[cfg(feature = "cuda")]
    #[test]
    #[ignore = "model inference; explicit model-proof workflow"]
    fn cuda_production_geometry_block_fused_adaln_matches_reference() -> candle_gen::Result<()> {
        use crate::loader::Weights;
        use crate::testfix::attn_ffn;
        use crate::transformer::rope::RopeTables;
        use candle_gen::candle_core::DType;
        use candle_gen::fused_path::media_fused_policy_guard;
        use candle_gen::modulate::ADALN_PATH;
        use std::collections::HashMap;

        let dev = Device::new_cuda(0)?;
        let cfg = crate::config::Krea2Config::turbo();
        let (hidden, heads, kv, hd) = (
            cfg.hidden_size,
            cfg.num_attention_heads,
            cfg.num_kv_heads,
            cfg.attention_head_dim,
        );
        // The modulation is the subject; a narrow FFN keeps the synthetic block cheap.
        let intermediate = 1024;
        let mut draw =
            |shape: &[usize]| Tensor::randn(0f32, 0.02f32, shape, &Device::Cpu).expect("cpu draw");
        let mut map: HashMap<String, Tensor> = HashMap::new();
        attn_ffn(
            &mut draw,
            &mut map,
            "b",
            hidden,
            heads,
            kv,
            hd,
            intermediate,
        );
        // A modulation table and raw (centered) norm scales large enough that a dropped `+ 1`
        // or a wrong norm weight moves every element.
        map.insert(
            "b.scale_shift_table".into(),
            Tensor::randn(0f32, 0.3f32, (6, hidden), &Device::Cpu)?,
        );
        for key in ["b.norm1.weight", "b.norm2.weight"] {
            map.insert(
                key.into(),
                Tensor::randn(0f32, 0.3f32, hidden, &Device::Cpu)?,
            );
        }
        let tmp = tempfile::tempdir().map_err(|e| candle_gen::CandleError::Msg(e.to_string()))?;
        let path = tmp.path().join("block.safetensors");
        candle_gen::candle_core::safetensors::save(&map, &path)?;
        let w = Weights::from_file(&path, &dev, DType::BF16)?;
        let block = SingleStreamBlock::load(&w, "b", heads, kv, hd, hidden, cfg.norm_eps)?;

        let (cap_len, side) = (512usize, 64usize);
        let s = cap_len + side * side;
        let rope = RopeTables::build_t2i(
            cap_len,
            side,
            side,
            cfg.axes_dims_rope,
            cfg.rope_theta as f64,
            &dev,
        )?;
        let (cos, sin) = rope.joint();
        let plan = AttentionPlan::budgeted(AttentionBudget::from_score_elements(
            candle_gen::ATTN_SCORES_BUDGET as u64,
            false,
        ));
        let bits = |t: &Tensor| -> candle_gen::Result<Vec<u16>> {
            Ok(t.to_dtype(DType::F32)?
                .flatten_all()?
                .to_vec1::<f32>()?
                .into_iter()
                .map(|v| (v.to_bits() >> 16) as u16)
                .collect())
        };
        let _media = media_fused_policy_guard(Some(true));
        let _kernel = ADALN_PATH.guard(Some(true));
        for b in [1usize, 2] {
            let x = Tensor::randn(0f32, 1.0, (b, s, hidden), &dev)?.to_dtype(DType::BF16)?;
            let tvec = Tensor::randn(0f32, 0.3, (b, 1, 6 * hidden), &dev)?.to_dtype(DType::BF16)?;
            let forward = || block.forward_with_attention_plan(&x, &tvec, &cos, &sin, plan);

            ADALN_PATH.set_enabled(Some(false));
            let start = ADALN_PATH.thread_counts();
            let reference = forward()?;
            let off = ADALN_PATH.thread_counts().since(&start);
            assert_eq!(
                (off.fused, off.reference_for("disabled")),
                (0, 2),
                "B={b}: {off}"
            );

            ADALN_PATH.set_enabled(Some(true));
            let start = ADALN_PATH.thread_counts();
            let fused = __model_proof_output!(forward())?;
            let on = ADALN_PATH.thread_counts().since(&start);
            assert_eq!((on.label(), on.fused), ("fused", 2), "B={b}: {on}");
            let (want, got) = (bits(&reference)?, bits(&fused)?);
            let mismatches = want.iter().zip(&got).filter(|(a, b)| a != b).count();
            assert_eq!(
                mismatches,
                0,
                "B={b}: the fused-AdaLN block differs from the reference block in {mismatches} \
                 of {} elements",
                want.len()
            );

            adaln_mutation::set_drop_one_plus(true);
            let mutated = forward();
            adaln_mutation::set_drop_one_plus(false);
            let mutated = bits(&mutated?)?;
            let caught = want.iter().zip(&mutated).filter(|(a, b)| a != b).count();
            assert!(
                caught * 2 > want.len(),
                "B={b}: dropping the +1 moved only {caught} of {} elements",
                want.len()
            );
            println!(
                "krea block (B={b}, S={s}, D={hidden}, bf16): fused AdaLN bit-identical to the \
                 reference ({} elements, path {on}); the dropped-+1 mutation differs in {caught}",
                want.len()
            );
        }

        #[cfg(feature = "model-inference-proof")]
        println!("\nMODEL_INFERENCE_BODY_COMPLETED transformer::block::tests::cuda_production_geometry_block_fused_adaln_matches_reference");
        Ok(())
    }

    /// SW-6128: the Krea 2 gated attention at production geometry (hidden 6144, GQA 48Q/12KV x
    /// 128, the 1024^2 joint sequence of 512 caption + 64x64 image tokens, the real t2i RoPE table)
    /// on synthetic weights, on CUDA, with the fused in-place QK-RMSNorm + RoPE against the same
    /// attention on the reference `rms_scale` -> RoPE chain (the kernel's switch off).
    ///
    /// Declared criterion: **bit identity** of the attention output in bf16 and f32 (the fused
    /// kernel reproduces the chain operation for operation, see `media_rope.cu`; everything after
    /// it is the same code on the same inputs). The path counters show both q and k took the
    /// fused kernel with the switch on, and the `disabled` reference with it off; the text-fusion
    /// form (no RoPE) never reaches the kernel.
    #[cfg(feature = "cuda")]
    #[test]
    #[ignore = "model inference; explicit model-proof workflow"]
    fn cuda_production_geometry_attention_fused_qk_norm_rope_matches_reference(
    ) -> candle_gen::Result<()> {
        use crate::loader::Weights;
        use crate::testfix::attn_ffn;
        use crate::transformer::rope::RopeTables;
        use candle_gen::candle_core::DType;
        use candle_gen::fused_path::media_fused_policy_guard;
        use candle_gen::rope::RMS_NORM_ROPE_PATH;
        use std::collections::HashMap;

        let dev = Device::new_cuda(0)?;
        let cfg = crate::config::Krea2Config::turbo();
        let (hidden, heads, kv, hd) = (
            cfg.hidden_size,
            cfg.num_attention_heads,
            cfg.num_kv_heads,
            cfg.attention_head_dim,
        );
        let mut draw =
            |shape: &[usize]| Tensor::randn(0f32, 0.02f32, shape, &Device::Cpu).expect("cpu draw");
        let mut map: HashMap<String, Tensor> = HashMap::new();
        attn_ffn(&mut draw, &mut map, "b", hidden, heads, kv, hd, 64);
        for key in ["b.attn.norm_q.weight", "b.attn.norm_k.weight"] {
            map.insert(key.into(), Tensor::randn(1f32, 0.3f32, hd, &Device::Cpu)?);
        }
        let tmp = tempfile::tempdir().map_err(|e| candle_gen::CandleError::Msg(e.to_string()))?;
        let path = tmp.path().join("attn.safetensors");
        candle_gen::candle_core::safetensors::save(&map, &path)?;

        let (cap_len, side) = (512usize, 64usize);
        let s = cap_len + side * side;
        let rope = RopeTables::build_t2i(
            cap_len,
            side,
            side,
            cfg.axes_dims_rope,
            cfg.rope_theta as f64,
            &dev,
        )?;
        let (cos, sin) = rope.joint();
        let plan = AttentionPlan::budgeted(AttentionBudget::from_score_elements(
            candle_gen::ATTN_SCORES_BUDGET as u64,
            false,
        ));
        let _media = media_fused_policy_guard(Some(true));
        let _kernel = RMS_NORM_ROPE_PATH.guard(Some(true));
        let x32 = Tensor::randn(0f32, 1.0, (1, s, hidden), &dev)?;
        for dtype in [DType::BF16, DType::F32] {
            let w = Weights::from_file(&path, &dev, dtype)?;
            let attn = GatedAttention::load(&w, "b.attn", heads, kv, hd, cfg.norm_eps)?;
            let x = x32.to_dtype(dtype)?;
            let run =
                |rope: Option<(&Tensor, &Tensor)>| attn.forward_with_attention_plan(&x, rope, plan);

            RMS_NORM_ROPE_PATH.set_enabled(Some(true));
            let start = RMS_NORM_ROPE_PATH.thread_counts();
            let fused = __model_proof_output!(run(Some((&cos, &sin))))?;
            let on = RMS_NORM_ROPE_PATH.thread_counts().since(&start);
            assert_eq!((on.fused, on.reference_total()), (2, 0), "{dtype:?}: {on}");

            RMS_NORM_ROPE_PATH.set_enabled(Some(false));
            let start = RMS_NORM_ROPE_PATH.thread_counts();
            let reference = run(Some((&cos, &sin)))?;
            let off = RMS_NORM_ROPE_PATH.thread_counts().since(&start);
            assert_eq!(off.reference_for("disabled"), 2, "{dtype:?}: {off}");
            RMS_NORM_ROPE_PATH.set_enabled(Some(true));

            let (f, r) = (
                fused
                    .to_dtype(DType::F32)?
                    .flatten_all()?
                    .to_vec1::<f32>()?,
                reference
                    .to_dtype(DType::F32)?
                    .flatten_all()?
                    .to_vec1::<f32>()?,
            );
            let mismatches = f
                .iter()
                .zip(&r)
                .filter(|(a, b)| a.to_bits() != b.to_bits())
                .count();
            let max_abs = f
                .iter()
                .zip(&r)
                .map(|(a, b)| (a - b).abs())
                .fold(0f32, f32::max);
            println!(
                "krea attention (S={s}, {heads}Q/{kv}KV x {hd}, {dtype:?}): fused qk-norm+rope vs \
                 reference: {mismatches} of {} elements differ, max|d| {max_abs:e}; \
                 on {on}, off {off}",
                f.len()
            );
            assert_eq!(
                mismatches, 0,
                "{dtype:?}: fused attention differs from the reference"
            );

            // The text-fusion form (no RoPE) keeps the plain norm.
            let start = RMS_NORM_ROPE_PATH.thread_counts();
            run(None)?;
            assert_eq!(RMS_NORM_ROPE_PATH.thread_counts().since(&start).total(), 0);
        }

        #[cfg(feature = "model-inference-proof")]
        println!("\nMODEL_INFERENCE_BODY_COMPLETED transformer::block::tests::cuda_production_geometry_attention_fused_qk_norm_rope_matches_reference");
        Ok(())
    }
}
