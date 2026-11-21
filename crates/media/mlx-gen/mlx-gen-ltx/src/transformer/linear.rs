//! The LTX `Linear` (dense / quantized base) and its per-pass LoRA / LoKr adapter stack on the
//! shared [`mlx_gen::adapters::AdaptableLinear`] (`mlx_layers::Linear`) spine (`SW-6956`), plus
//! the [`LtxAdaptable`] seam both DiTs expose.

use super::*;
use mlx_gen::adapters::{AdaptableLinear, Adapter};

/// A Linear projection backed by the shared [`AdaptableLinear`] (`mlx_layers::Linear`) spine
/// (`SW-6956`, `BACK-1`, `BACK-4`, `PIPE-14`): dense (`addmm` / `matmul`) or group-quantized
/// (`quantized_matmul_with_bias`) base weight plus an optional forward-time per-pass LoRA / LoKr
/// residual stack. With no adapters the forward is byte-identical to the bare Q4/Q8/dense base.
#[derive(Clone)]
pub struct Linear {
    pub(super) inner: AdaptableLinear,
}

impl Linear {
    /// Wrap a raw dense `[out, in]` weight and optional `[out]` bias on the shared spine.
    pub(crate) fn from_dense(weight: Array, bias: Option<Array>) -> Self {
        Self {
            inner: AdaptableLinear::dense(weight, bias),
        }
    }

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
        let inner = match w.get(&format!("{prefix}.scales")) {
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
                    AdaptableLinear::from_quantized_parts(
                        q.clone(),
                        to_dtype(scales, dt)?,
                        to_dtype(biases, dt)?,
                        b,
                        prec.group,
                        prec.bits,
                    )
                } else {
                    // Dequantize to dense f32 (bit-identical to the reference's mx.dequantize).
                    let dense =
                        dequantize(q, scales, Some(biases), Some(prec.group), Some(prec.bits))?;
                    AdaptableLinear::dense(to_dtype(&dense, Dtype::Float32)?, b)
                }
            }
            None => AdaptableLinear::dense(to_dtype(w.require(&format!("{prefix}.weight"))?, dt)?, b),
        };
        Ok(Self { inner })
    }

    pub(crate) fn forward(&self, x: &Array) -> Result<Array> {
        self.inner.forward(x)
    }

    /// Stack a LoRA residual (the loader installs one per resolved target). `a`/`b` are the raw
    /// `lora_A [rank,in]` / `lora_B [out,rank]` transposed to residual form by the caller.
    pub(crate) fn push_lora(&mut self, a: Array, b: Array, pass_scale: Vec<f32>) {
        self.inner.push_lora_per_pass(a, b, pass_scale);
    }

    /// Stack a LoKr residual (sc-2393). `delta` is the precomputed `[out,in]` bf16 weight delta
    /// (`alpha/rank` already folded in by `reconstruct_lokr_delta`); `pass_scale` is the per-pass
    /// user strength alone. Net-new — the reference `lora/` is LoRA-only.
    pub(crate) fn push_lokr(&mut self, delta: Array, pass_scale: Vec<f32>) {
        self.inner.push_lokr_per_pass(delta, pass_scale);
    }

    /// **Training seam** (sc-3047): replace the adapter stack with a single trainable LoRA residual.
    /// `a`/`b` are the residual-form factors (`lora_A.t()` `[in,rank]` and the `alpha/rank`-scaled
    /// `lora_B.t()` `[rank,out]`), with a single unit pass-scale, so the forward adds
    /// `(x·a)·b` = `(x·Aᵀ·Bᵀ)·(alpha/rank)` — the same residual the reference `_MlxLoRALinear` applies.
    /// The trainer re-injects the freshly-traced factors each optimizer step (functional autograd),
    /// delegating to [`AdaptableLinear::set_adapters`]. Unlike [`push_lora`](Self::push_lora)
    /// this *replaces* the stack (one residual at a time), not append.
    pub(crate) fn set_train_lora(&mut self, a: Array, b: Array) {
        self.inner.set_adapters(vec![Adapter::Lora {
            a,
            b,
            scale: 1.0,
        }]);
    }

    /// Select the active denoise pass for this linear's LoRA residuals (no-op without adapters).
    pub(super) fn set_lora_pass(&self, pass: usize) {
        self.inner.set_active_pass(pass);
    }

    /// The base weight's logical `[out, in]` — the shape a LoKr delta reshapes to (sc-2393).
    pub(crate) fn base_shape(&self) -> Vec<i32> {
        self.inner.base_shape()
    }

    /// The optional dense bias vector attached to the base projection.
    #[cfg(test)]
    pub(crate) fn bias(&self) -> Option<&Array> {
        self.inner.bias()
    }

    /// The base weight's current compute dtype (sc-4942) — used to detect a prior bf16 cast.
    pub(crate) fn weight_dtype(&self) -> Dtype {
        self.inner.compute_dtype().unwrap_or(Dtype::Float32)
    }

    /// Cast the base weight to the training compute `dtype` in place (sc-4942 bf16 lever). Only the
    /// frozen base is touched; a LoRA overlay is installed AFTER the cast (at the compute dtype) by
    /// the trainer.
    pub(crate) fn cast_weights(&mut self, dtype: Dtype) -> Result<()> {
        self.inner.cast_compute_dtype(dtype)
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
