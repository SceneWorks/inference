//! The shared **AdaLN modulation** for the candle DiT providers (SW-6127, feature SW-F287).
//!
//! Two layers, one home:
//!
//! - [`modulate`] / [`modulate_with`] — the composable `normed · (1 + scale) + shift` (or
//!   `· scale`) every DiT crate used to carry as a private `fn modulate` copy. Pure candle ops, so
//!   gradient-carrying training code can use it as is.
//! - [`adaln`] (with the [`rms_adaln`] / [`layer_adaln`] shorthands) — the whole
//!   `norm(x) · (1 + scale) + shift`, RMSNorm or LayerNorm with an optional learned weight (and
//!   LayerNorm bias), modulation per sample `[B, 1, D]` or per token `[B, S, D]`. On CUDA it is one
//!   fused launch, **bit-identical** to [`adaln_reference`] (candle's `ops::rms_norm` /
//!   `ops::layer_norm` then [`modulate_with`]) for f32, bf16 and f16; everywhere else — and for a
//!   gradient-tracked operand, an unserved shape or the switch off — it *is* that reference. Each
//!   call is recorded in [`ADALN_PATH`] (`candle_gen::fused_path`), switch `CANDLE_FUSED_ADALN`.
//!
//! [`AdaLn`] describes one call: [`AdaLn::rms`] / [`AdaLn::layer`] plus `with_weight`,
//! `with_bias`, `norm_in_f32` (normalize an f32 copy and cast back — the
//! `F.rms_norm(x.float(), w).to(dtype)` convention, weight and bias then f32) and `direct_scale`.
//!
//! ```ignore
//! use candle_gen::modulate::{adaln, AdaLn};
//! // Krea's `+1` RMSNorm (weight pre-folded to `scale + 1`, f32), then `(1 + prescale)`, `+ preshift`.
//! let pre = adaln(&x, &AdaLn::rms(eps).with_weight(&w_f32).norm_in_f32(), &prescale, &preshift)?;
//! ```
//!
//! [`adaln_chain`] / [`AdaLnChain`] (SW-6129) are the same AdaLN for the providers' **composable**
//! norm chains (`mean_keepdim` / `sqr` / `sqrt` / `broadcast_div` LayerNorm and RMS in f32 or the
//! activation dtype, candle_nn's LayerNorm without a bias), bit-identical to each chain.
//!
//! The implementation (kernel, plan, reference, parity tests) lives in
//! `candle_quant_kernels::adaln`, beside the nvrtc seam it compiles through.

pub use candle_quant_kernels::adaln::{
    adaln, adaln_block_dim, adaln_norm_reference, adaln_reference, adaln_scale_only,
    adaln_scale_only_reference, check_adaln, layer_adaln, modulate, modulate_with, plan_adaln,
    rms_adaln, AdaLn, AdaLnPlan, AdaLnRefusal, AdaNorm, OperandMeta, ParamKind, ScaleForm,
    ADALN_MAX_ELEMS, ADALN_PATH, ADALN_SRC,
};

// SW-6129: the same AdaLN for the providers' composable norm chains (`mean_keepdim` / `sqr` /
// `sqrt` / `broadcast_div` LayerNorm and RMS, f32 or native precision, any modulation / output
// dtype) — bit-identical to each chain on CUDA, the chain itself everywhere else, counted on the
// same `ADALN_PATH`.
pub use candle_quant_kernels::adaln_chain::{
    adaln_chain, adaln_chain_norm_reference, adaln_chain_reference, adaln_chain_scale_only,
    adaln_chain_scale_only_reference, chain_block_dim, check_adaln_chain, plan_adaln_chain,
    AdaLnChain, ChainNorm, ChainPlan, ChainPrecision, ADALN_CHAIN_SRC,
};
