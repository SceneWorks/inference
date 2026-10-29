//! The LTX-local `Linear` (dense / quantized base) and its per-pass LoRA / LoKr adapter stack,
//! plus the `LtxAdaptable` seam both DiTs expose.

use super::*;

/// A Linear's base weight — dense or Q8-quantized, selected by [`Precision`] at load. `b` is `None`
/// only for the LTX-2.5 `ff_bias:false` FFN Linears (sc-18758) — every other Linear in the checkpoint
/// (attention, patchify/proj_out, adaLN) always carries a bias in both 2.3 and 2.5 (the reference
/// `attention_bias` check is hardcoded `True`), so `None` never appears there.
#[derive(Clone)]
pub(super) enum LinearKind {
    Dense {
        w: Array,         // [out, in]
        b: Option<Array>, // [out]
    },
    Quant {
        q: Array,      // [out, in_packed] U32
        scales: Array, // [out, in/group]
        biases: Array,
        b: Option<Array>,
        group: i32,
        bits: i32,
    },
}

impl LinearKind {
    pub(super) fn forward(&self, x: &Array) -> Result<Array> {
        match self {
            LinearKind::Dense { w, b: Some(b) } => linear(x, w, b),
            LinearKind::Dense { w, b: None } => Ok(matmul(x, w.t())?),
            LinearKind::Quant {
                q,
                scales,
                biases,
                b,
                group,
                bits,
            } => quantized_matmul_with_bias(x, q, scales, biases, b.as_ref(), *group, *bits),
        }
    }

    /// The base weight's logical `[out, in]` — what a LoKr delta reshapes to. The quantized packed
    /// weight is opaque, so recover `in` from the scales grid (`[out, in/group]`) × the group size
    /// (mirrors the core `AdaptableLinear::base_shape`).
    pub(super) fn base_shape(&self) -> Vec<i32> {
        match self {
            LinearKind::Dense { w, .. } => w.shape().to_vec(),
            LinearKind::Quant { scales, group, .. } => {
                let s = scales.shape();
                vec![s[0], s[1] * group]
            }
        }
    }

    /// The base weight's current compute dtype (sc-4942) — the dense weight, or the quantized
    /// scales (the packed `q` carries no float dtype). Lets the trainer detect a prior bf16 cast.
    pub(super) fn weight_dtype(&self) -> Dtype {
        match self {
            LinearKind::Dense { w, .. } => w.dtype(),
            LinearKind::Quant { scales, .. } => scales.dtype(),
        }
    }

    /// Cast the base to the training compute `dtype` in place (sc-4942 bf16 lever). A dense weight
    /// casts whole; a quantized weight keeps its packed `q` (the quant geometry is dtype-free) and
    /// casts only the float-bearing scales / biases / bias so `quantized_matmul` matches a bf16
    /// activation. Bias casts in both arms.
    pub(super) fn cast(&mut self, dtype: Dtype) -> Result<()> {
        match self {
            LinearKind::Dense { w, b } => {
                *w = to_dtype(w, dtype)?;
                if let Some(b) = b {
                    *b = to_dtype(b, dtype)?;
                }
            }
            LinearKind::Quant {
                scales, biases, b, ..
            } => {
                *scales = to_dtype(scales, dtype)?;
                *biases = to_dtype(biases, dtype)?;
                if let Some(b) = b {
                    *b = to_dtype(b, dtype)?;
                }
            }
        }
        Ok(())
    }
}

/// One adapter as a forward-time residual — the reference `lora/apply.py::LoRALinear`
/// (`out + scale·strength·(x·Aᵀ·Bᵀ)`), NOT a merged weight. Residual (vs the reference's
/// alternative `apply_loras_to_model` merge→bf16) keeps the shipped Q8 base intact — a full
/// attn+ff adapter over this 22B Q8 transformer would dequantize ~15 GB to bf16 if merged, and
/// per-pass strength would double it — and leaves the bit-exact base forward (sc-2842) untouched.
///
/// **LoRA** keeps its factors at their loaded (bf16) dtype so a bf16 `x` (Bf16Q8) runs bf16 and an
/// f32 `x` (F32Q8) promotes to f32, exactly as the reference does (the sc-2772 toolchain fix means
/// the low-rank bf16 GEMM is correct, so no f32 forcing is needed). **LoKr** (sc-2393 — net-new, the
/// reference `lora/` has no LoKr) carries a precomputed `[out,in]` delta (`alpha/rank` already folded
/// in by `reconstruct_lokr_delta`); the residual is `x·ΔWᵀ` with `ΔW` cast to the activation dtype,
/// mirroring the fork's `LoKrLinear` / the core `Adapter::Lokr`. Both variants apply a per-pass scale.
#[derive(Clone)]
pub(super) enum LtxAdapter {
    /// `residual = pass_scale[pass] · (x·Aᵀ)·Bᵀ`. `pass_scale` already folds `(alpha/rank)·strength`.
    Lora {
        a: Array, // Aᵀ : [in, rank] (residual form; `lora_A` transposed)
        b: Array, // Bᵀ : [rank, out] (`lora_B` transposed)
        pass_scale: Vec<f32>,
    },
    /// `residual = pass_scale[pass] · x·ΔWᵀ`. `ΔW` ([out,in], bf16) has `alpha/rank` baked in, so
    /// `pass_scale` is the user `strength[pass]` alone (no further alpha/rank fold).
    Lokr {
        delta: Array, // ΔW : [out, in] (bf16; cast to the activation dtype at forward)
        pass_scale: Vec<f32>,
    },
}

impl LtxAdapter {
    /// Per-pass effective strengths (one entry per distilled stage, or a length-1 uniform vec).
    /// `Linear::forward` clamps the active pass index into this.
    pub(super) fn pass_scale(&self) -> &[f32] {
        match self {
            LtxAdapter::Lora { pass_scale, .. } | LtxAdapter::Lokr { pass_scale, .. } => pass_scale,
        }
    }

    /// The unscaled residual `x·Aᵀ·Bᵀ` (LoRA) or `x·ΔWᵀ` (LoKr), before the per-pass scale.
    pub(super) fn residual(&self, x: &Array) -> Result<Array> {
        Ok(match self {
            LtxAdapter::Lora { a, b, .. } => matmul(&matmul(x, a)?, b)?,
            LtxAdapter::Lokr { delta, .. } => matmul(x, delta.as_dtype(x.dtype())?.t())?,
        })
    }
}

/// The adapter overlay on a [`Linear`]: the stacked residuals plus the active denoise-pass index,
/// set by [`LtxDiT::set_lora_pass`] before each stage. The pass lives in a `Cell` so a shared
/// `&self` forward reads it and the pipeline switches passes without `&mut` (the crate runs
/// single-device, one job per thread — see the runtime docs; `Cell<usize>` is `Send`).
#[derive(Clone)]
pub(super) struct LoraStack {
    pub(super) adapters: Vec<LtxAdapter>,
    pub(super) pass: Cell<usize>,
}

/// A Linear: a base weight (`LinearKind`) plus an optional forward-time LoRA overlay. With no
/// adapters the forward is byte-identical to the pre-sc-2687 path (the Q8/dense base only).
#[derive(Clone)]
pub struct Linear {
    pub(super) kind: LinearKind,
    pub(super) lora: Option<LoraStack>,
}

impl Linear {
    /// Every Linear except the LTX-2.5 `ff_bias:false` FFN Linears carries a bias — see
    /// [`load_with_bias`](Self::load_with_bias).
    pub(crate) fn load(w: &Weights, prefix: &str, prec: Precision) -> Result<Self> {
        Self::load_with_bias(w, prefix, prec, true)
    }

    /// Load with the bias tensor optionally absent (sc-18758: LTX-2.5's `ff_bias:false` FFN Linears
    /// carry no `{prefix}.bias`). `has_bias:false` must not `require` a bias key that legitimately
    /// doesn't exist in the checkpoint — and must not silently zero-fill a *missing* bias it should
    /// have found, so this only ever skips the read when the caller (the FFN construction, gated on
    /// `LtxConfig::ff_bias`/`audio_ff_bias`/`connector_ff_bias`) says the tensor was never shipped.
    pub(crate) fn load_with_bias(
        w: &Weights,
        prefix: &str,
        prec: Precision,
        has_bias: bool,
    ) -> Result<Self> {
        let dt = prec.dtype();
        let b = has_bias
            .then(|| to_dtype(w.require(&format!("{prefix}.bias"))?, dt))
            .transpose()?;
        let kind = match w.get(&format!("{prefix}.scales")) {
            Some(scales) => {
                let q = w.require(&format!("{prefix}.weight"))?;
                let biases = w.require(&format!("{prefix}.biases"))?;
                if prec.keep_quant() {
                    // Keep the weights packed; `quantized_matmul` dequantizes on the fly with fp32
                    // accumulation and is correct for f32 *or* bf16 activations at either bit-width
                    // (the Z-Image/Qwen Q8 path). Scales / biases are cast to the compute dtype so the
                    // on-the-fly dequant matches the reference's (f32 for quant_f32 — a lossless upcast
                    // of the bf16 file scales). bits/group come from the checkpoint's split_model.json
                    // via `prec` (sc-2686), so Q4 and Q8 both load unchanged.
                    LinearKind::Quant {
                        q: q.clone(),
                        scales: to_dtype(scales, dt)?,
                        biases: to_dtype(biases, dt)?,
                        b,
                        group: prec.group,
                        bits: prec.bits,
                    }
                } else {
                    // Dequantize to dense f32 (bit-identical to the reference's mx.dequantize).
                    let dense =
                        dequantize(q, scales, Some(biases), Some(prec.group), Some(prec.bits))?;
                    LinearKind::Dense {
                        w: to_dtype(&dense, Dtype::Float32)?,
                        b,
                    }
                }
            }
            None => LinearKind::Dense {
                w: to_dtype(w.require(&format!("{prefix}.weight"))?, dt)?,
                b,
            },
        };
        Ok(Linear { kind, lora: None })
    }

    pub(crate) fn forward(&self, x: &Array) -> Result<Array> {
        let mut out = self.kind.forward(x)?;
        if let Some(stack) = &self.lora {
            let pass = stack.pass.get();
            for ad in &stack.adapters {
                // residual = (LoRA: (x·Aᵀ)·Bᵀ | LoKr: x·ΔWᵀ) · scale[pass], factors/delta at the
                // activation dtype (MLX promotes a bf16 factor against an f32 activation), the scale
                // through a dtype-matched scalar so the multiply preserves the residual dtype (the
                // reference's weak Python-float `scale·…`).
                let ps = ad.pass_scale();
                // A malformed adapter with an empty `pass_scale` would underflow `ps.len() - 1`
                // (usize) into an out-of-bounds index → panic; use a checked access and surface a
                // clear error instead (F-009). The loader rejects this earlier in `pass_strengths`,
                // but keep the access itself safe regardless of how the stack was built.
                let s = *ps
                    .get(pass.min(ps.len().saturating_sub(1)))
                    .ok_or_else(|| {
                        Error::Msg(
                            "ltx adapter: pass_scale must have at least one entry".to_string(),
                        )
                    })?;
                // sc-15265: LTX carries its own adapter stack rather than `AdaptableLinear`, so it
                // needs the same two rules the shared seam now enforces (see
                // `mlx_gen::adapters::AdaptableLinear::apply_adapters`):
                //   * a `scale == 0` pass contributes nothing — skip it, so a disabled adapter (or a
                //     pass the schedule turns off) leaves `out` byte-identical to the unadapted base;
                //   * the residual is narrowed to `out`'s dtype BEFORE the add, so f32 factors over a
                //     bf16 host cannot widen this Linear — and every op downstream of it — to f32.
                // Unlike the shared seam, this loop carries NO training exemption (rule 0 there),
                // even though the training factors installed by `set_train_lora` (`training.rs`'s
                // `install_train_lora`, and the checkpointed block closure below) flow through here
                // too. That is safe only because LTX training is **f32-only today**: `train_impl`
                // hardcodes `let compute_dtype = Dtype::Float32` (sc-4942 — a bf16 activation cast
                // was measured to decorrelate the gradient, so the worker's `train_dtype=bf16` is
                // deliberately ignored), both install paths derive their factor dtype from it, and
                // the only `cast_weights(Bfloat16)` callers are `#[cfg(test)]` harnesses. Host and
                // factors are therefore both f32 and the narrowing cast is a no-op. Enabling the
                // sc-4942 bf16 compute lever REQUIRES revisiting this: f32 factors would then meet
                // a bf16 host here and the training residual would be narrowed — exactly what
                // `AdaptableLinear::apply_adapters` exempts training from.
                if s == 0.0 {
                    continue;
                }
                let r = ad.residual(x)?;
                let r = multiply(&r, &scalar(s).as_dtype(r.dtype())?)?;
                out = add(&out, &r.as_dtype(out.dtype())?)?;
            }
        }
        Ok(out)
    }

    /// Stack a LoRA residual (the loader installs one per resolved target). `a`/`b` are the raw
    /// `lora_A [rank,in]` / `lora_B [out,rank]` transposed to residual form by the caller.
    pub(crate) fn push_lora(&mut self, a: Array, b: Array, pass_scale: Vec<f32>) {
        self.lora_stack()
            .adapters
            .push(LtxAdapter::Lora { a, b, pass_scale });
    }

    /// Stack a LoKr residual (sc-2393). `delta` is the precomputed `[out,in]` bf16 weight delta
    /// (`alpha/rank` already folded in by `reconstruct_lokr_delta`); `pass_scale` is the per-pass
    /// user strength alone. Net-new — the reference `lora/` is LoRA-only.
    pub(crate) fn push_lokr(&mut self, delta: Array, pass_scale: Vec<f32>) {
        self.lora_stack()
            .adapters
            .push(LtxAdapter::Lokr { delta, pass_scale });
    }

    /// **Training seam** (sc-3047): replace the adapter stack with a single trainable LoRA residual.
    /// `a`/`b` are the residual-form factors (`lora_A.t()` `[in,rank]` and the `alpha/rank`-scaled
    /// `lora_B.t()` `[rank,out]`), with a single unit pass-scale, so the forward adds
    /// `(x·a)·b` = `(x·Aᵀ·Bᵀ)·(alpha/rank)` — the same residual the reference `_MlxLoRALinear` applies.
    /// The trainer re-injects the freshly-traced factors each optimizer step (functional autograd),
    /// the LTX analog of the core `AdaptableLinear::set_adapters`. Unlike [`push_lora`](Self::push_lora)
    /// this *replaces* the stack (one residual at a time), not append.
    pub(crate) fn set_train_lora(&mut self, a: Array, b: Array) {
        self.lora = Some(LoraStack {
            adapters: vec![LtxAdapter::Lora {
                a,
                b,
                pass_scale: vec![1.0],
            }],
            pass: Cell::new(0),
        });
    }

    /// The adapter stack, created empty (pass 0) on first install.
    pub(super) fn lora_stack(&mut self) -> &mut LoraStack {
        self.lora.get_or_insert_with(|| LoraStack {
            adapters: Vec::new(),
            pass: Cell::new(0),
        })
    }

    /// Select the active denoise pass for this linear's LoRA residuals (no-op without adapters).
    pub(super) fn set_lora_pass(&self, pass: usize) {
        if let Some(stack) = &self.lora {
            stack.pass.set(pass);
        }
    }

    /// The base weight's logical `[out, in]` — the shape a LoKr delta reshapes to (sc-2393).
    pub(crate) fn base_shape(&self) -> Vec<i32> {
        self.kind.base_shape()
    }

    /// The base weight's current compute dtype (sc-4942) — used to detect a prior bf16 cast.
    pub(crate) fn weight_dtype(&self) -> Dtype {
        self.kind.weight_dtype()
    }

    /// Cast the base weight to the training compute `dtype` in place (sc-4942 bf16 lever). Only the
    /// frozen base is touched; a LoRA overlay is installed AFTER the cast (at the compute dtype) by
    /// the trainer, so `lora` is `None` here in practice.
    pub(crate) fn cast_weights(&mut self, dtype: Dtype) -> Result<()> {
        self.kind.cast(dtype)
    }
}

/// A model tree that routes a LoRA target's dotted path (post LTX key normalization) to its
/// [`Linear`] and selects the active denoise pass for all installed residuals. Implemented by the
/// video-only [`LtxDiT`] building block and the production dual-modality [`AvDiT`], so the adapter
/// loader ([`crate::adapters::apply_ltx_adapters`]) is generic over both (sc-2687).
pub trait LtxAdaptable {
    /// Resolve a target path to its [`Linear`], or `None` if it names no adaptable module (reported
    /// skipped by the loader, never silently dropped).
    fn adaptable_mut(&mut self, path: &[&str]) -> Option<&mut Linear>;
    /// Select the active distilled denoise pass for every installed residual (no-op without adapters).
    fn set_lora_pass(&self, pass: usize);
}

impl LtxAdaptable for LtxDiT {
    fn adaptable_mut(&mut self, path: &[&str]) -> Option<&mut Linear> {
        LtxDiT::adaptable_mut(self, path)
    }
    fn set_lora_pass(&self, pass: usize) {
        LtxDiT::set_lora_pass(self, pass)
    }
}

impl LtxAdaptable for AvDiT {
    fn adaptable_mut(&mut self, path: &[&str]) -> Option<&mut Linear> {
        AvDiT::adaptable_mut(self, path)
    }
    fn set_lora_pass(&self, pass: usize) {
        AvDiT::set_lora_pass(self, pass)
    }
}
