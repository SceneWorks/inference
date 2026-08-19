//! Krea 2 pose-control **memory estimators** — the provider-owned Krea/Qwen-VAE cost model used by
//! the real-weight calibration harness and defensive selected-shape validation. The promoted integer-byte evidence
//! produced from that harness is submitted to SceneWorks' worker-owned `select_strategy`; this module
//! deliberately contains no budget-to-strategy decision.
//!
//! Two shape-derived peaks are exposed (both **excluding** the phase-A Qwen3-VL text encoder, whose
//! footprint is accounted for separately when a caller models co-residency):
//!   * the **control-denoise** peak — the resident heavy weights (base DiT + the pose branch, at the tier
//!     [`mlx_gen::gen_core::tier_integrity::control_branch_tier`] assigns it, + the VAE, all held through
//!     the heavy phase) plus the per-step activation working set of the concatenated single-stream forward
//!     with the N-block branch injected;
//!   * the **Qwen-VAE decode** peak — the same resident heavy weights plus the full-output decode spike
//!     through the `AutoencoderKLQwenImage` decoder stack (the sc-11747 target); its tiled floor is the
//!     resident weights + the assembled output buffers + one minimal tile.
//!
//! The weight terms are **first-principles** param counts (validated against the published Krea shapes:
//! ~11.1 B base @ 28×6144, ~3.0 B branch @ N=7 → ~6.1 GB bf16, matching the candle #480 profile's
//! ~6.6 GB), padded by a measured `RESIDENT_OVERHEAD_GIB` for the terms the block count omits (the
//! VAE, the DiT's non-block params, and the MLX Metal-allocator resident floor).
//!
//! **Measured-MLX calibration (sc-11847).** The activation + decode-spike coefficients and the resident
//! overhead were RE-FIT on real weights on a 128 GB M-series Metal Mac — the story's e2e gate — via
//! `tests/control_memory_calibration_real_weights.rs`, which measures the isolated denoise and decode
//! `mlx_rs::memory::get_peak_memory` high-water of a real `krea_2_turbo_control` render (base tier ∈
//! {bf16, q4}, resolution ∈ {512², 768², 1024²}, pose branch bf16, `Sequential` residency so the peaks
//! are ex-text). The candle #480 CUDA priors it replaced were **wrong for MLX in both directions**: the
//! coefficients ~8–10× (denoise) / ~4× (decode) too high, yet the estimate still UNDER-shot the real
//! peak at 512² because MLX's materialized-weight + framework resident floor (~33.4 GB bf16 / ~15.9 GB
//! q4) is ~4 GB above the bare param count — the CUDA activation coefficient had merely masked that gap
//! at 1024². The measured slopes are ~44 (bf16) / ~61 (q4) B/(token·hidden) for denoise and ~5211 B/px
//! for decode (tier-independent — the VAE decode is the same at every base tier). The constants below
//! keep the **over-predict / never-under-shoot** convention (an under-shoot is an OOM; an over-shoot
//! only tiles/adapts slightly sooner — the Wan sc-4998 / PiD sc-10087 guard): each is rounded up so the
//! estimate stays ≥ the measured peak (within ≤ ~1.16× at every tested point).

use mlx_gen::gen_core::tier_integrity::control_branch_tier;
use mlx_gen::tiling::TilingConfig;
use mlx_gen::{Error, Quant, Result};

use crate::config::Krea2Config;

/// 1 GiB in bytes (`1024³`, matching MLX's `metal::malloc` GiB reporting / the core `memory` module).
const GIB: f64 = 1024.0 * 1024.0 * 1024.0;

/// Effective bytes-per-parameter for each residency tier, including the group-wise affine quant
/// overhead (a per-64 group `scale` + `bias`, f16 each ⇒ `2·2/64 = 0.0625` B/param). bf16 is the dense
/// default the pose overlay ships as.
fn bytes_per_param(tier: Option<Quant>) -> f64 {
    match tier {
        None => 2.0,                     // bf16 dense
        Some(Quant::Q8) => 1.0 + 0.0625, // 8-bit + group scale/bias
        Some(Quant::Q4) => 0.5 + 0.0625, // 4-bit + group scale/bias
        // NVFP4 (epic 11037, sc-11042) — ~4.5 effective bits/weight (E2M1 4-bit + FP8 block scale).
        // NVFP4 is a candle/Blackwell tier, NOT served by this MLX pose overlay; the arm exists only to
        // keep the match total over the shared `gen_core::Quant`. The MLX/macOS runtime has no FP4
        // hardware and does not surface NVFP4, so `tier` is never `Some(Nvfp4)` on this path.
        Some(Quant::Nvfp4) => 4.5 / 8.0,
    }
}

/// A small safety multiplier on the counted parameters to cover the terms this first-principles count
/// omits (RMSNorm scales, the shared modulation projections, `img_in`/`time_embed`/final layers) —
/// deliberately over-counting so the resident-weight estimate never under-shoots the real footprint.
const PARAM_MARGIN: f64 = 1.1;

/// Dominant parameter count of ONE single-stream block: GQA attention (`q`,`k`,`v`,`out`) + the SwiGLU
/// FFN's three projections. `q_dim == hidden`; `kv_dim == num_kv_heads·head_dim`.
fn single_stream_block_params(cfg: &Krea2Config) -> f64 {
    let h = cfg.hidden_size as f64;
    let q = cfg.q_dim() as f64;
    let kv = cfg.kv_dim() as f64;
    let inter = cfg.intermediate_size as f64;
    // attn: q(h·q) + k(h·kv) + v(h·kv) + out(q·h); ffn SwiGLU: gate + up + down = 3·h·inter.
    2.0 * h * q + 2.0 * h * kv + 3.0 * h * inter
}

/// Resident weight bytes of the **base DiT** (`num_layers` single-stream blocks) at `tier`.
fn base_dit_bytes(cfg: &Krea2Config, tier: Option<Quant>) -> f64 {
    let params = cfg.num_layers as f64 * single_stream_block_params(cfg) * PARAM_MARGIN;
    params * bytes_per_param(tier)
}

/// Resident weight bytes of the **pose control branch**: `n` copied single-stream blocks, each plus a
/// `proj_out` (`hidden·hidden`) zero-init output projection.
fn branch_bytes(cfg: &Krea2Config, n: usize, tier: Option<Quant>) -> f64 {
    let h = cfg.hidden_size as f64;
    let per_block = single_stream_block_params(cfg) + h * h;
    let params = n as f64 * per_block * PARAM_MARGIN;
    params * bytes_per_param(tier)
}

/// Fixed resident **overhead** (GiB) the first-principles block count does NOT capture, added to both
/// stage peaks: the Qwen-Image VAE (`AutoencoderKLQwenImage`, f32), the DiT's non-block params
/// (`img_in`/`time_embed`/text-fusion aggregator/`final_layer`/modulation), AND the MLX Metal-allocator
/// resident floor (materialized-weight buffers + the retained working set). **Measured-MLX (sc-11847),
/// not the old ~0.4 GiB VAE-only guess:** the real ex-text resident floor is ~33.4 GiB (bf16) / ~15.9
/// GiB (q4), ~4 GiB above `base_dit + branch` alone; `5.5` covers that residual at every tested point
/// (the bf16 decode-512² point binds it) with a small over-predict margin. Tier-independent to first
/// order (the VAE stays f32; the uncounted DiT params + allocator floor barely move with the base tier —
/// bf16 needs ~4.5, q4 ~4.1), so a single constant is both simpler and conservative.
const RESIDENT_OVERHEAD_GIB: f64 = 5.5;

/// Text/vision-encoder (Qwen3-VL-4B, ~4 B params) resident bytes at `tier` — the phase-A footprint the
/// residency lever frees. Packs with the base tier (sc-11727 `load_krea_text`).
fn text_resident_gib(tier: Option<Quant>) -> f64 {
    const TEXT_PARAMS: f64 = 4.0e9;
    TEXT_PARAMS * bytes_per_param(tier) / GIB
}

/// Per-step denoise **activation** bytes per (token · hidden) element — the concatenated-stream
/// activations + the (fused-SDPA) attention working set + the N-block branch forward, on top of the
/// resident weights. **Measured-MLX (sc-11847):** the real slope is ~44 (bf16) / ~61 (q4)
/// B/(token·hidden) — MLX's fused SDPA + the CFG-free single forward keep the denoise peak
/// resident-weight-dominated, far below the candle #480 CUDA prior (470, from ~11 GB @ 1024²). `80`
/// rounds up over the larger (q4) measured slope with headroom for higher resolutions.
const DENOISE_ACT_BYTES_PER_TOKEN_HIDDEN: f64 = 80.0;

/// Decode **spike** bytes per output pixel through the Qwen-VAE decoder conv stack — the transient that
/// tiling (sc-11747) shrinks. **Measured-MLX (sc-11847):** the real slope is ~5211 B/px, tier-independent
/// (the VAE decode is the same at every base tier), far below the candle #480 CUDA prior (22500, from
/// ~22 GB @ 1024²). `6500` rounds up over the measured slope; the fixed VAE-materialization part of the
/// decode peak lives in `RESIDENT_OVERHEAD_GIB`, so this term is the pure per-pixel conv growth.
const DECODE_SPIKE_BYTES_PER_PIXEL: f64 = 6_500.0;

/// The assembled full-resolution RGB output buffer held across a tiled decode (`output [1,3,H,W]`),
/// f32 — the term tiling can NOT shrink. sc-18320's separable fold retired the full-size 1-channel
/// blend-weight accumulator this constant originally budgeted alongside the output (the normalizer
/// is now three 1-D host vectors, materialized once at the end); `16` deliberately keeps the old
/// 12 + 4 B/px envelope — this is a shipped, calibrated memory model, and over-predicting the
/// unshrinkable term is the safe direction (resolution reduction slightly sooner, never OOM).
const DECODE_ACCUM_BYTES_PER_PIXEL: f64 = 16.0;

/// The working set of ONE minimal decode tile — the least the tiled decode can spike to, on top of the
/// resident weights + output buffers. Resolution-independent (a fixed tile), so it is the decode floor's
/// only non-buffer term. At the measured [`DECODE_SPIKE_BYTES_PER_PIXEL`] a ~256²-tile conv forward is
/// only ~0.4 GiB, so `1.5` is deliberately conservative; the tiled floor is NOT exercised until the
/// decode-tiling lever lands (sc-11747), so a precise re-fit of this floor is deferred to that story
/// (which drives a real tiled decode). Over-predicting the floor is the safe direction (it only makes
/// the policy prefer resolution reduction slightly sooner, never OOM).
const DECODE_MIN_TILE_GIB: f64 = 1.5;

/// Denoise-forward token count for a `width × height` render: the latent is `[16, H/8, W/8]`,
/// patchified 2×2 → `(H/16)·(W/16)` image tokens (the text tokens are a negligible add).
fn denoise_tokens(width: u32, height: u32) -> f64 {
    (width as f64 / 16.0).floor() * (height as f64 / 16.0).floor()
}

/// The **control-denoise** stage peak (GiB, ex-text): resident heavy weights + the activation working
/// set at `width × height`. Pure (shape + config only) → unit-testable.
pub fn control_denoise_peak_ex_text_gib(
    cfg: &Krea2Config,
    branch_blocks: usize,
    base_tier: Option<Quant>,
    branch_tier: Option<Quant>,
    width: u32,
    height: u32,
) -> f64 {
    let heavy = base_dit_bytes(cfg, base_tier) + branch_bytes(cfg, branch_blocks, branch_tier);
    let act =
        DENOISE_ACT_BYTES_PER_TOKEN_HIDDEN * denoise_tokens(width, height) * cfg.hidden_size as f64;
    (heavy / GIB) + RESIDENT_OVERHEAD_GIB + act / GIB
}

/// The single-pass **Qwen-VAE decode** stage peak (GiB, ex-text): resident heavy weights + the
/// full-output decode spike. Pure.
pub fn qwen_vae_decode_peak_ex_text_gib(
    cfg: &Krea2Config,
    branch_blocks: usize,
    base_tier: Option<Quant>,
    branch_tier: Option<Quant>,
    width: u32,
    height: u32,
) -> f64 {
    let heavy = base_dit_bytes(cfg, base_tier) + branch_bytes(cfg, branch_blocks, branch_tier);
    let px = width as f64 * height as f64;
    (heavy / GIB) + RESIDENT_OVERHEAD_GIB + (DECODE_SPIKE_BYTES_PER_PIXEL * px) / GIB
}

/// The **tiled** Qwen-VAE decode floor (GiB, ex-text): resident heavy weights + the un-shrinkable
/// full-output buffers + one minimal tile. This is the lower envelope against which calibrated,
/// explicitly parameterized bounded-decode candidates are measured. Pure.
pub fn qwen_vae_decode_tiled_floor_ex_text_gib(
    cfg: &Krea2Config,
    branch_blocks: usize,
    base_tier: Option<Quant>,
    branch_tier: Option<Quant>,
    width: u32,
    height: u32,
) -> f64 {
    let heavy = base_dit_bytes(cfg, base_tier) + branch_bytes(cfg, branch_blocks, branch_tier);
    let px = width as f64 * height as f64;
    (heavy / GIB)
        + RESIDENT_OVERHEAD_GIB
        + (DECODE_ACCUM_BYTES_PER_PIXEL * px) / GIB
        + DECODE_MIN_TILE_GIB
}

/// Map a base-DiT quant width to the [`Quant`] tier it names (`4 → Q4`, `8 → Q8`); any other width is
/// dense (`None`). NVFP4 has no honest bit width (sc-11042) and never reaches this lane, so it is not
/// expressible here — the branch tier is derived from the resulting [`Quant`] by
/// [`control_branch_tier`], which does handle it.
pub(crate) fn tier_from_bits(bits: i32) -> Option<Quant> {
    match bits {
        4 => Some(Quant::Q4),
        8 => Some(Quant::Q8),
        _ => None,
    }
}

/// The quant width the pose control branch is packed to at LOAD, given the base DiT's width
/// (sc-15799 tier integrity — this REPLACES the sc-11748 budget gate `should_quantize_control_branch`).
///
/// A pure function of the base width. The device budget is deliberately **not** an input: packing the
/// branch to the selected tier is not a memory lever the policy spends, it is what the user's tier
/// choice already means. See [`mlx_gen::gen_core::tier_integrity`] for the rule and for the one
/// declared, measured exception (a q4 base floors its branch at q8, because a q4 control residual
/// measures "pose-locked; non-pose details drift").
///
/// - `None` (dense bf16 base) ⇒ `None`: the branch is already at the selected tier.
/// - `Some(8)` ⇒ `Some(8)`: follows exactly.
/// - `Some(4)` ⇒ `Some(8)`: the declared floor.
pub fn control_branch_quant_bits(base_bits: Option<i32>) -> Option<i32> {
    control_branch_tier(base_bits.and_then(tier_from_bits)).map(Quant::bits)
}

/// Test whether the requested render geometry fits after the non-geometry strategies have engaged.
/// This is deliberately a predicate: it cannot return a substitute width or height.
///   * the **un-tileable DENOISE** activation peak — the sc-11749 target (candle #480's ~11 GiB @ 1024²),
///     and
///   * the Qwen-VAE decode peak for the strategy already selected upstream: the selected tile edge for
///     bounded decode, single-pass peak otherwise.
///
/// Both peaks are computed at the ACTUAL `base_tier`/`branch_tier` (a resident weight can't be re-packed
/// mid-render), so this never assumes a quant saving the loaded model can't realize — unlike the
/// deleted load-time planner, which estimated the branch bf16 and treated packing as a future lever.
/// `text_co_resident` adds the phase-A Qwen3-VL encoder footprint when it stays resident (the
/// `Resident` path); under `Sequential` it was dropped before the heavy phase, matching the
/// `*_ex_text_gib` estimators.
///
/// `safe_gib` is injected ([`mlx_gen::memory::safe_budget_gib`] at the call site) so the decision is
/// unit-testable without a device.
#[allow(clippy::too_many_arguments)]
pub fn control_geometry_fits(
    safe_gib: f64,
    cfg: &Krea2Config,
    branch_blocks: usize,
    base_tier: Option<Quant>,
    branch_tier: Option<Quant>,
    width: u32,
    height: u32,
    text_co_resident: bool,
    decode_tile_edge: Option<u32>,
) -> bool {
    let text = if text_co_resident {
        text_resident_gib(base_tier)
    } else {
        0.0
    };
    // A resolution fits when both the un-tileable denoise peak and the SELECTED decode shape are within
    // budget — the provider validates that selection but never changes it. The text encoder is counted
    // only when it stays co-resident (`Resident`; `Sequential` dropped it before the heavy phase).
    let denoise =
        control_denoise_peak_ex_text_gib(cfg, branch_blocks, base_tier, branch_tier, width, height)
            + text;
    let decode = if let Some(tile_edge) = decode_tile_edge {
        let heavy = base_dit_bytes(cfg, base_tier) + branch_bytes(cfg, branch_blocks, branch_tier);
        let output_px = width as f64 * height as f64;
        let tile_px = tile_edge as f64 * tile_edge as f64;
        (heavy / GIB)
            + RESIDENT_OVERHEAD_GIB
            + (DECODE_ACCUM_BYTES_PER_PIXEL * output_px) / GIB
            + (DECODE_SPIKE_BYTES_PER_PIXEL * tile_px) / GIB
    } else {
        qwen_vae_decode_peak_ex_text_gib(cfg, branch_blocks, base_tier, branch_tier, width, height)
    } + text;
    denoise <= safe_gib && decode <= safe_gib
}

/// Admit the requested geometry or return a typed refusal. This seam intentionally returns no
/// replacement dimensions: after admission, the render path must use the immutable request fields.
pub fn require_control_geometry(width: u32, height: u32, feasible: bool) -> Result<()> {
    if feasible {
        Ok(())
    } else {
        Err(Error::GeometryRefused {
            reason: "krea_2_turbo_control requested geometry exceeds the measured unified-memory \
                     feasibility bound; no current verified alternative is available"
                .to_owned(),
            requested_width: width,
            requested_height: height,
            alternative: None,
        })
    }
}

/// Resolve the exact bounded-decode parameters selected through the shared calibration contract.
pub fn requested_control_decode_tiling(
    memory: mlx_gen::gen_core::GenerationMemory,
) -> Result<TilingConfig> {
    let edge = memory.decode_tile_edge.ok_or_else(|| {
        Error::Msg("krea_2_turbo_control bounded decode requires decode_tile_edge".to_owned())
    })?;
    let overlap = memory.decode_overlap.ok_or_else(|| {
        Error::Msg("krea_2_turbo_control bounded decode requires decode_overlap".to_owned())
    })?;
    if !crate::memory_strategy::DECODE_TILE_EDGES.contains(&edge)
        || overlap != crate::memory_strategy::DECODE_OVERLAP
    {
        return Err(Error::Msg(format!(
            "krea_2_turbo_control unsupported bounded decode {edge}/{overlap}"
        )));
    }
    Ok(TilingConfig {
        spatial: Some(mlx_gen::tiling::SpatialTiling {
            tile_px: edge as i32,
            overlap_px: overlap as i32,
        }),
        temporal: None,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The first-principles weight counts land on the published anchors: ~11 B base params, and a bf16
    /// branch of ~6 GB (candle #480 ~6.6 GB) — proof the cost model is grounded, not arbitrary.
    #[test]
    fn weight_estimates_match_published_anchors() {
        let cfg = Krea2Config::turbo();
        let base_params = cfg.num_layers as f64 * single_stream_block_params(&cfg);
        assert!(
            (10.5e9..12.5e9).contains(&base_params),
            "base ≈ 11 B params, got {base_params:.3e}"
        );
        let branch_gib = branch_bytes(&cfg, 7, None) / GIB;
        assert!(
            (5.5..7.5).contains(&branch_gib),
            "bf16 branch ≈ 6–7 GiB (candle #480 ~6.6), got {branch_gib:.2}"
        );
        // Packing the branch to q4 saves the bulk of that.
        let saved = (branch_bytes(&cfg, 7, None) - branch_bytes(&cfg, 7, Some(Quant::Q4))) / GIB;
        assert!(saved > 4.0, "q4 branch saving ≈ 4–5 GiB, got {saved:.2}");
    }

    /// Peaks grow with resolution (more tokens, more pixels) — a sanity guard on the shape scaling.
    #[test]
    fn peaks_grow_with_resolution() {
        let cfg = Krea2Config::turbo();
        let dn_512 = control_denoise_peak_ex_text_gib(&cfg, 7, Some(Quant::Q4), None, 512, 512);
        let dn_1024 = control_denoise_peak_ex_text_gib(&cfg, 7, Some(Quant::Q4), None, 1024, 1024);
        assert!(dn_1024 > dn_512);
        let dc_512 = qwen_vae_decode_peak_ex_text_gib(&cfg, 7, Some(Quant::Q4), None, 512, 512);
        let dc_1024 = qwen_vae_decode_peak_ex_text_gib(&cfg, 7, Some(Quant::Q4), None, 1024, 1024);
        assert!(dc_1024 > dc_512);
        // The single-pass decode peak exceeds its tiled floor (tiling's whole point). Note: on measured
        // MLX (sc-11847) the decode spike is only ~6 GiB @ 1024² — far smaller than the candle #480 CUDA
        // ~22 GiB — so tiling shaves ~5 GiB (~20%) off the peak, NOT the ~50% the CUDA prior implied. The
        // resident weights + overhead dominate the decode peak on Metal, so tiling is a modest lever here.
        let floor =
            qwen_vae_decode_tiled_floor_ex_text_gib(&cfg, 7, Some(Quant::Q4), None, 1024, 1024);
        assert!(
            floor < dc_1024 && (dc_1024 - floor) > 3.0,
            "tiled floor {floor:.1} must be a real reduction below single-pass {dc_1024:.1}"
        );
    }

    // ── sc-15799: the load-time branch TIER (`control_branch_quant_bits`) — not a lever. ───────────

    /// A dense bf16 base carries a dense branch: it is already at the selected tier, so there is nothing
    /// to pack. Same for a width that names no tier.
    #[test]
    fn branch_tier_dense_base_stays_dense() {
        assert_eq!(control_branch_quant_bits(None), None);
        assert_eq!(control_branch_quant_bits(Some(16)), None);
    }

    /// The invariant this story exists for: a PACKED base never carries a bf16 branch, and the tier is a
    /// function of the base alone — q8 follows exactly, q4 floors at q8 (the declared, measured
    /// exception). A mutation restoring the sc-11748 budget gate fails here, because there is no budget
    /// to pass.
    #[test]
    fn branch_tier_follows_the_base_with_the_declared_q4_floor() {
        assert_eq!(control_branch_quant_bits(Some(8)), Some(8));
        assert_eq!(control_branch_quant_bits(Some(4)), Some(8));
    }

    /// The provider's peaks are estimated with the branch at its integrity tier, so a q4 base projects a
    /// LOWER peak than it did when the branch was left bf16 — the ~3.3 GB of unrequested precision the
    /// old ladder carried until its last rung (6.6 GB bf16 → ~3.3 GB at the q8 floor; NOT the retracted
    /// 8.4 GB, which exceeds the whole branch). Guards against a regression that re-estimates the branch
    /// dense while the loader packs it (an under- or over-prediction of the real resident set).
    #[test]
    fn provider_peaks_use_the_integrity_branch_tier_not_bf16() {
        let cfg = Krea2Config::turbo();
        let branch_tier = control_branch_tier(Some(Quant::Q4));
        let integrity =
            control_denoise_peak_ex_text_gib(&cfg, 7, Some(Quant::Q4), branch_tier, 1024, 1024)
                .max(qwen_vae_decode_peak_ex_text_gib(
                    &cfg,
                    7,
                    Some(Quant::Q4),
                    branch_tier,
                    1024,
                    1024,
                ))
                + text_resident_gib(Some(Quant::Q4));
        let dense_branch =
            control_denoise_peak_ex_text_gib(&cfg, 7, Some(Quant::Q4), None, 1024, 1024).max(
                qwen_vae_decode_peak_ex_text_gib(&cfg, 7, Some(Quant::Q4), None, 1024, 1024),
            ) + text_resident_gib(Some(Quant::Q4));
        assert!(
            integrity < dense_branch,
            "the provider estimate must price the q8 branch it will load ({integrity:.2}), not a bf16 one \
             ({dense_branch:.2})"
        );
    }

    /// Fast, recorded-data companion to the ignored real-weight calibration harness. The sc-11847
    /// Metal measurements are the expensive source of truth; keeping their six published points in a
    /// normal unit test makes lowering any calibrated coefficient or resident floor fail without
    /// requiring weights. The live harness remains authoritative when the execution shape changes.
    fn assert_never_under_shoot(estimate_gib: f64, measured_gib: f64, label: &str) {
        assert!(
            estimate_gib >= measured_gib,
            "{label} estimate {estimate_gib:.3} under-shot recorded {measured_gib:.3} GiB"
        );
    }

    #[test]
    fn sc_11847_recorded_points_remain_never_under_shoot() {
        struct Point {
            tier: Option<Quant>,
            size: u32,
            measured_denoise_gib: f64,
            measured_decode_gib: f64,
        }

        let points = [
            Point {
                tier: None,
                size: 512,
                measured_denoise_gib: 33.63,
                measured_decode_gib: 34.81,
            },
            Point {
                tier: None,
                size: 768,
                measured_denoise_gib: 33.65,
                measured_decode_gib: 35.46,
            },
            Point {
                tier: None,
                size: 1024,
                measured_denoise_gib: 34.41,
                measured_decode_gib: 38.85,
            },
            Point {
                tier: Some(Quant::Q4),
                size: 512,
                measured_denoise_gib: 16.29,
                measured_decode_gib: 18.30,
            },
            Point {
                tier: Some(Quant::Q4),
                size: 768,
                measured_denoise_gib: 16.58,
                measured_decode_gib: 18.73,
            },
            Point {
                tier: Some(Quant::Q4),
                size: 1024,
                measured_denoise_gib: 17.36,
                measured_decode_gib: 22.12,
            },
        ];
        let cfg = Krea2Config::turbo();
        for point in points {
            // sc-11847 measured before tier integrity repacked the pose branch, so `None` recreates
            // that exact dense-branch calibration shape for both base tiers.
            let denoise =
                control_denoise_peak_ex_text_gib(&cfg, 7, point.tier, None, point.size, point.size);
            let decode =
                qwen_vae_decode_peak_ex_text_gib(&cfg, 7, point.tier, None, point.size, point.size);
            assert_never_under_shoot(
                denoise,
                point.measured_denoise_gib,
                &format!("{:?} {}² denoise", point.tier, point.size),
            );
            assert_never_under_shoot(
                decode,
                point.measured_decode_gib,
                &format!("{:?} {}² decode", point.tier, point.size),
            );
        }
    }

    #[test]
    #[should_panic(expected = "under-shot recorded")]
    fn sc_11847_guard_kills_removed_resident_overhead_mutant() {
        let cfg = Krea2Config::turbo();
        let mutant = qwen_vae_decode_peak_ex_text_gib(&cfg, 7, Some(Quant::Q4), None, 1024, 1024)
            - RESIDENT_OVERHEAD_GIB;
        assert_never_under_shoot(mutant, 22.12, "q4 1024² decode mutant");
    }

    #[test]
    fn geometry_feasibility_checks_only_the_requested_shape() {
        let cfg = Krea2Config::turbo();
        let selected_decode = qwen_vae_decode_tiled_floor_ex_text_gib(
            &cfg,
            7,
            Some(Quant::Q4),
            Some(Quant::Q4),
            1024,
            1024,
        ) - DECODE_MIN_TILE_GIB
            + (DECODE_SPIKE_BYTES_PER_PIXEL
                * crate::memory_strategy::DECODE_TILE_EDGE.pow(2) as f64)
                / GIB;
        let full_peak =
            control_denoise_peak_ex_text_gib(&cfg, 7, Some(Quant::Q4), Some(Quant::Q4), 1024, 1024)
                .max(selected_decode);
        assert!(control_geometry_fits(
            full_peak,
            &cfg,
            7,
            Some(Quant::Q4),
            Some(Quant::Q4),
            1024,
            1024,
            false,
            Some(crate::memory_strategy::DECODE_TILE_EDGE),
        ));
        assert!(!control_geometry_fits(
            full_peak - 0.01,
            &cfg,
            7,
            Some(Quant::Q4),
            Some(Quant::Q4),
            1024,
            1024,
            false,
            Some(crate::memory_strategy::DECODE_TILE_EDGE),
        ));
        assert!(
            !control_geometry_fits(
                full_peak,
                &cfg,
                7,
                Some(Quant::Q4),
                Some(Quant::Q4),
                1024,
                1024,
                false,
                None,
            ),
            "a resident or staged request must be checked against single-pass decode, not silently tiled"
        );
    }

    #[test]
    fn admitted_geometry_is_exactly_the_requested_geometry_or_typed_refusal() {
        require_control_geometry(1024, 768, true).unwrap();
        let error = require_control_geometry(1024, 768, false).unwrap_err();
        assert!(matches!(
            error,
            Error::GeometryRefused {
                requested_width: 1024,
                requested_height: 768,
                alternative: None,
                ..
            }
        ));
    }
}
