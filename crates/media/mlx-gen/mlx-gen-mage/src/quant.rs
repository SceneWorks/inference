//! Packed Q4/Q8 loading for Mage-Flow — the Group-B per-crate template (sc-8669).
//!
//! Pre-quantized snapshots carry, for every packed base, the triple `{base}.weight` (u32 codes) +
//! `{base}.scales` + `{base}.biases`. The `lin`/`embedding` loaders **auto-detect** the pack by the
//! presence of `{base}.scales` — there is no `quantization` manifest to read on the load path, so
//! one loader serves both a dense bf16 snapshot and a pre-quantized tier (sc-14980).
//!
//! `GROUP_SIZE` is the workspace default 64 and must match what [`crate::convert`] writes; the two
//! sides are pinned together by `convert`'s byte-equality test against the load-time
//! `AdaptableLinear::quantize`.

use mlx_gen::adapters::AdaptableLinear;
use mlx_gen::weights::Weights;
use mlx_gen::Result;

/// The quantization group size every Mage component packs at.
///
/// Shared by the DiT ([`crate::transformer`]), the Qwen3-VL text encoder
/// ([`crate::text_encoder`]), and the offline converter ([`crate::convert`]) so a tier written by
/// one is loadable by the other. The text encoder's own `QUANT_GROUP_SIZE` is this value.
pub(crate) const GROUP_SIZE: i32 = 64;

/// Load `{base}` as an [`AdaptableLinear`], packed when `{base}.scales` is present, else dense.
///
/// `bias` additionally loads the dense `{base}.bias`. Note the two distinct tensors: `{base}.bias`
/// is the Linear's own additive bias (always dense, never quantized) while `{base}.biases` is the
/// quantization zero-point that rides with a packed weight.
pub(crate) fn lin(w: &Weights, base: &str, bias: bool) -> Result<AdaptableLinear> {
    mlx_gen::quant::lin(w, base, bias, GROUP_SIZE)
}
