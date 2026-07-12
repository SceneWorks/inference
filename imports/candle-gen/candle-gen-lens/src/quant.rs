//! Lens DiT quantization seam — **two routes to a quantized DiT**, both built on the ONE shared
//! [`candle_gen::quant`] seam (F-025 / sc-9005: this crate's `QLinear` was one of four drifted copies,
//! now unified into `candle_gen::quant::QLinear`) and candle-core's first-class GGUF quant:
//!
//! - **Packed tier (sc-9413, the fast path).** The hosted `SceneWorks/lens-mlx` / `lens-turbo-mlx`
//!   q4/q8 tiers store each quantized DiT `Linear` as the MLX packed triple `{base}.weight` (u32
//!   codes) + `{base}.scales` + `{base}.biases` (group size 64, the default — the pipeline asserts the
//!   parsed `quantization.group_size == 64` at load so a future group-32 tier fails LOUD rather than
//!   silently repacking to garbage through the group-64 shared loaders, sc-9474). [`QLinear::linear_detect`]
//!   packed-**detects** the `.scales` sibling and builds the quantized weight **straight from the
//!   packed parts** on the DiT device (Q4 → `Q4_1` lossless repack, Q8 → `Q8_0` requant). **No dense
//!   bf16 weight is ever materialized** — the q4 DiT lands directly from the packed parts, with no
//!   dense staging *and* no load-then-quantize pass.
//!
//! - **Dense → quantize (the legacy path, unchanged; sc-5117).** When the snapshot is a dense bf16
//!   tier (the stock `SceneWorks/Lens` diffusers snapshot; `.scales` absent), each DiT projection loads
//!   dense and [`crate::transformer::LensTransformer::quantize`] folds it to `Q4_0`/`Q8_0` in place
//!   **after** the (dense) weights — and any adapter merge — have loaded. [`QLinear::quantize`] is a
//!   **no-op** on an already-packed projection (idempotent), so a packed-detect load and the
//!   unconditional post-load `quantize` pass compose: an MLX-packed weight is never double-quantized.
//!
//! **The quantized matmul dequantizes the weight and runs a *dense* matmul — it does NOT take candle's
//! int8 `QMatMul` fast path (sc-7702).** That fast path (CUDA `fast_mmvq`/`fast_mmq`) quantizes the
//! *activation* to `q8_1` per 32-element block; gpt-oss's massive outlier text activations (±10⁴) blow
//! out a block's int8 scale and zero the co-located channels, so the Q4 DiT denoise diverges to NaN
//! within a few steps — a solid-black render (Q8 only masks it with more weight bits). Dequantizing the
//! weight to a dense matmul keeps the activation full-precision, so **uniform Q4 renders coherently** —
//! GPU-verified on Blackwell. Both Lens routes therefore use the shared seam's
//! [`candle_gen::quant::MatmulStrategy::DequantDense`] arm — [`QLinear::linear_detect`] and
//! [`QLinear::quantize`] both build/fold to it. (The FLUX.2 / SAM3 / SeedVR2 sites use the int8-fast
//! arm; the strategy is now an explicit per-site knob, not four silently-diverged types — F-025.)
//!
//! **Text encoder & VAE.** This seam is the **DiT** only. The gpt-oss text encoder
//! ([`crate::text_encoder`]) has its own expert quant seam (also on the shared
//! [`candle_gen::quant::repack_packed_weight`], sc-9457), and the Flux.2 VAE stays f32.

// The whole `Dense | Quantized` seam now lives once in `candle-gen` (F-025 / sc-9005). Lens's DiT uses
// the dequant-dense (sc-7702-safe) forward: `QLinear::linear_detect` builds a packed `DequantDense`
// projection and `QLinear::quantize` folds a dense one to the same, exactly as this crate's former
// local copy did. Re-export under the crate-local names the transformer/lib already reference.
//
// The projection type is now the shared residual-capable [`candle_gen::quant::AdaptLinear`] (sc-11105),
// aliased to `QLinear` so every `linear_detect` / `quantize` call site in the DiT stays unchanged. It
// carries an optional **forward-time additive LoRA/LoKr residual** (`AdaptLinear::push_lora` /
// `push_lokr_structured`), so a user LoRA applies on a **packed q4/q8** tier with the base kept packed
// ([`crate::adapters::install_additive`]) — the deltas ride unmerged, never folded into u32 codes. The
// dense tier keeps folding (bit-exact) via [`crate::adapters::merge_adapters`] + `LensTransformer::
// quantize` (which no-ops on a packed base, so the residuals survive). With no adapter attached the
// forward is byte-identical to the bare base.
pub use candle_gen::quant::{ggml_dtype, AdaptLinear as QLinear};
