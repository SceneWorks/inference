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

/// The on-disk base of the output head's `AdaLayerNormContinuous` modulation projection —
/// `norm_out.linear`, `[6144, 3072]` (`scale, shift = chunk(linear(silu(temb)), 2)`).
///
/// Named here rather than inlined because the **load-time** packer ([`crate::final_layer`]) and the
/// **offline** converter ([`crate::convert`]) must apply [`FINAL_MOD_MIN_BITS`] to the same tensor or
/// a pre-quantized tier stops matching load-time quantization byte for byte.
pub(crate) const FINAL_MOD_BASE: &str = "norm_out.linear";

/// The bit-width floor for [`FINAL_MOD_BASE`] — **8, never 4** (sc-15071).
///
/// This one projection is the whole reason the Q4 tier used to render a repeating tiled texture
/// instead of the prompt. It is not a quality nicety; below 8 bits the sampler diverges.
///
/// **Why this tensor and no other.** The output head is
/// `x = layer_norm(img, affine=false) * (1 + scale) + shift` followed by `proj_out`, with
/// `scale, shift` produced by this projection from `temb` — **one vector per pack segment**, so its
/// error is not per-token noise that averages out but a coherent per-channel distortion applied
/// identically to every token. The non-affine LayerNorm immediately upstream has just erased each
/// token's own scale, so this modulation *is* the channel statistics `proj_out` decodes into the
/// predicted velocity, and nothing downstream (no residual, no later block) can dilute it. Feeding
/// that distortion back through 30 Euler steps drives the latent to a degenerate periodic
/// attractor, which the VAE decodes as tiling.
///
/// **Measured** (512², 30 steps, cfg 5.0, seed 42, `"a red cube on a white table, studio lighting"`,
/// luma correlation against the bf16 render — `tests/quant_real_weights.rs`). Quantizing exactly one
/// group to Q4 with the entire rest of the DiT left dense:
///
/// | group held at Q4 | mae vs bf16 | luma corr |
/// |---|---|---|
/// | `img_in` | 1.9 | 0.999 |
/// | `txt_in` | 1.2 | 1.000 |
/// | `time_text_embed.*` | 6.2 | 0.998 |
/// | `proj_out` | 1.3 | 1.000 |
/// | block `attn` / `mlp` / `*_mod.1` | 8.9 / 10.1 / 13.6 | 0.984 / 0.979 / 0.929 |
/// | **`norm_out.linear`** | **52.7** | **0.553** |
///
/// Four times the damage of the next-worst group, on 0.5% of the DiT's parameters. The converse
/// holds too: the whole DiT at Q4 *except* this tensor renders correctly (0.853), and at this
/// 8-bit floor with everything else at Q4 it renders correctly (0.835) — against 0.329 for the
/// uniform-Q4 DiT that shipped. (Those three figures hold the text encoder dense, to isolate the
/// DiT; the end-to-end Q4 tier additionally needs [`LM_LAYER_MIN_BITS`].)
///
/// **Why a floor and not an exclusion.** Keeping it packed at 8 bits costs 18.9 MB against 10.6 MB
/// at Q4 (a dense bf16 exclusion would cost 37.7 MB), leaves the Q4 tier's quantized-projection
/// count at the same 174, and leaves the Q8 and bf16 tiers byte-identical to what they already were
/// — only the Q4 artifact changes. Bit-width is derived per tensor from the packed shapes
/// ([`mlx_gen::quant::packed_bits`]), so a mixed-width artifact needs no loader change and no side
/// manifest.
pub(crate) const FINAL_MOD_MIN_BITS: i32 = 8;

/// On-disk key prefix of the Qwen3-VL LM's 36 decoder layers — everything under
/// `model.language_model.layers.` (`self_attn.{q,k,v,o}_proj`, `mlp.{gate,up,down}_proj`).
///
/// The **token embedding** (`model.language_model.embed_tokens`) and the **vision tower**
/// (`model.visual.*`) sit outside this prefix on purpose; see [`LM_LAYER_MIN_BITS`].
pub(crate) const LM_LAYER_PREFIX: &str = "model.language_model.layers.";

/// The bit-width floor for the LM decoder-layer projections — **8, never 4** (sc-15071).
///
/// The second half of the same defect. `norm_out.linear` alone does not fix the Q4 tier: with the
/// DiT head repaired but the text encoder still uniformly Q4, the end-to-end tier still renders a
/// tiled texture. This is the Wan/UMT5 text-encoder precedent, and the open question sc-14046
/// explicitly parked ("the 36-layer 2560-hidden LM may need a higher floor than q4 … validate text
/// fidelity at q4 vs q8") — answered here by rendering rather than by comparing prompt vectors.
///
/// **Why the vector-space check was not enough.** The existing
/// `qwen_q8_beats_q4_and_both_preserve_prompt_fidelity` gate reports Q4 pooled-vector cosine ≈ 0.96
/// against dense, with prompt ranking intact, and concluded Q4 was "explicitly low-fidelity but
/// usable". It is — *against a dense DiT*. Q4 text and Q4 image weights are not independently
/// tolerable errors: each alone is recoverable, together they are not.
///
/// **Measured** (same scene/seed as [`FINAL_MOD_MIN_BITS`], luma correlation vs the bf16 render,
/// always with the **repaired** Q4 DiT so this table isolates the text encoder):
///
/// | embed_tokens | layer attn | layer MLP | luma corr |
/// |---|---|---|---|
/// | Q4 | Q4 | Q4 | **0.163** |
/// | Q8 | Q4 | Q4 | 0.163 |
/// | Q8 | Q8 | Q4 | **0.143** |
/// | Q4 | Q4 | Q8 | 0.761 |
/// | Q4 | Q8 | Q8 | **0.877** |
/// | Q8 | Q8 | Q8 | 0.835 |
///
/// Two things fall out. The **SwiGLU MLP** is what breaks: holding attention at Q8 while the MLP
/// stays Q4 is no better than uniform Q4 (0.143). And the **token embedding is fine at Q4** — the
/// Q4/Q8/Q8 and Q8/Q8/Q8 rows differ by less than the run-to-run spread, so the embedding keeps the
/// tier's width and its 195 MB saving on the single largest tensor in the model. The floor covers
/// attention as well as the MLP because attention at Q4 costs a further 0.877 → 0.761 for ~0.25 GB,
/// which is not a trade worth taking on the tier that has to render correctly.
pub(crate) const LM_LAYER_MIN_BITS: i32 = 8;

/// The width `base` is actually packed at when tier `requested` was asked for.
///
/// The **single seam** every packing path calls — the offline converter
/// ([`crate::convert::quant_floor_bits`] re-exports it) and both load-time `quantize` methods — so
/// a pre-quantized artifact and load-time quantization cannot disagree about a floor. That equality
/// is asserted end-to-end by `a_packed_tier_renders_identically_to_load_time_quantization`
/// (max_abs 0).
pub(crate) fn floor_bits(base: &str, requested: i32) -> i32 {
    if base == FINAL_MOD_BASE {
        requested.max(FINAL_MOD_MIN_BITS)
    } else if base.starts_with(LM_LAYER_PREFIX) {
        requested.max(LM_LAYER_MIN_BITS)
    } else {
        requested
    }
}

/// Load `{base}` as an [`AdaptableLinear`], packed when `{base}.scales` is present, else dense.
///
/// `bias` additionally loads the dense `{base}.bias`. Note the two distinct tensors: `{base}.bias`
/// is the Linear's own additive bias (always dense, never quantized) while `{base}.biases` is the
/// quantization zero-point that rides with a packed weight.
pub(crate) fn lin(w: &Weights, base: &str, bias: bool) -> Result<AdaptableLinear> {
    mlx_gen::quant::lin(w, base, bias, GROUP_SIZE)
}
