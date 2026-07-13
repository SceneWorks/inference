//! Krea 2 pose-control **memory-adaptation estimators** (sc-11750) — the per-lane peak estimators that
//! plug the Krea control lane into the shared, backend-neutral escalation policy
//! ([`mlx_gen::gen_core::mempolicy`]). This module holds the Krea/Qwen-VAE cost model; the escalation
//! order + budget comparison live in gen-core.
//!
//! Two shape-derived peaks drive the plan (both **excluding** the phase-A Qwen3-VL text encoder, whose
//! footprint the policy carries once as the residency lever):
//!   * the **control-denoise** peak — the resident heavy weights (base DiT + the bf16 pose branch + the
//!     VAE, all held through the heavy phase) plus the per-step activation working set of the
//!     concatenated single-stream forward with the N-block branch injected;
//!   * the **Qwen-VAE decode** peak — the same resident heavy weights plus the full-output decode spike
//!     through the `AutoencoderKLQwenImage` decoder stack (the sc-11747 target); its tiled floor is the
//!     resident weights + the assembled output buffers + one minimal tile.
//!
//! The weight terms are **first-principles** param counts (validated against the published Krea shapes:
//! ~11.1 B base @ 28×6144, ~3.0 B branch @ N=7 → ~6.1 GB bf16, matching the candle #480 profile's
//! ~6.6 GB). The activation/decode-spike coefficients are seeded from that candle #480 CUDA profile
//! (~11 GB denoise activations, ~30 GB decode peak, both @ q4/1024²) and rounded **up** so the estimate
//! over-predicts rather than under-shoots (an under-shoot is an OOM; an over-shoot only tiles/adapts
//! slightly sooner — the Wan/PiD guard convention). The **MLX on-device recalibration of these
//! coefficients is the story's true e2e gate** (a real 32 GB-Mac Krea-control render); until then they
//! are conservative CUDA-seeded priors, not measured MLX constants.

use mlx_gen::gen_core::mempolicy::{plan_memory_adaptation, LaneLevers, MemoryPlan, StagePeaks};
use mlx_gen::Quant;

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

/// Resident weight bytes of the Qwen-Image VAE (`AutoencoderKLQwenImage`, f32, ~few-hundred-MB conv
/// stack). A fixed conservative estimate — the VAE stays dense (the published `vae/` is f32) and is tiny
/// beside the DiT, so a coarse constant is sufficient. Rounded up.
const VAE_RESIDENT_GIB: f64 = 0.4;

/// Text/vision-encoder (Qwen3-VL-4B, ~4 B params) resident bytes at `tier` — the phase-A footprint the
/// residency lever frees. Packs with the base tier (sc-11727 `load_krea_text`).
fn text_resident_gib(tier: Option<Quant>) -> f64 {
    const TEXT_PARAMS: f64 = 4.0e9;
    TEXT_PARAMS * bytes_per_param(tier) / GIB
}

/// Per-step denoise **activation** bytes per (token · hidden) element — a conservative fold of the
/// concatenated-stream activations, the attention working set, and the N-block branch forward. Seeded
/// from the candle #480 ~11 GB @ 1024² anchor (tokens = (1024/16)² = 4096, hidden = 6144): 11·1024³ /
/// (4096·6144) ≈ 470 B/elem. **MLX recalibration pending** (the e2e gate).
const DENOISE_ACT_BYTES_PER_TOKEN_HIDDEN: f64 = 470.0;

/// Decode **spike** bytes per output pixel through the Qwen-VAE decoder conv stack — the transient that
/// tiling (sc-11747) shrinks. Seeded from the candle #480 ~30 GB @ 1024² decode peak minus the ~8 GB
/// resident heavy weights ⇒ ~22 GB spike: 22·1024³ / 1024² ≈ 22500 B/px. Rounded up; **MLX
/// recalibration pending**.
const DECODE_SPIKE_BYTES_PER_PIXEL: f64 = 22_500.0;

/// The assembled full-resolution RGB output buffers held across a tiled decode (`output [1,3,H,W]` +
/// blend `weights`), f32 — the term tiling can NOT shrink. ~16 B/px (12 for the 3-channel output + a
/// 1-channel weight accumulator + margin).
const DECODE_ACCUM_BYTES_PER_PIXEL: f64 = 16.0;

/// The working set of ONE minimal decode tile — the least the tiled decode can spike to, on top of the
/// resident weights + output buffers. Resolution-independent (a fixed tile), so it is the decode floor's
/// only non-buffer term. Seeded from a ~256²-tile forward at [`DECODE_SPIKE_BYTES_PER_PIXEL`]
/// (22500·256² ≈ 1.4 GiB); rounded up.
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
    (heavy / GIB) + VAE_RESIDENT_GIB + act / GIB
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
    (heavy / GIB) + VAE_RESIDENT_GIB + (DECODE_SPIKE_BYTES_PER_PIXEL * px) / GIB
}

/// The **tiled** Qwen-VAE decode floor (GiB, ex-text): resident heavy weights + the un-shrinkable
/// full-output buffers + one minimal tile. The least [`plan_memory_adaptation`]'s decode-tiling lever
/// can drive the decode peak toward (`budgeted_plan` then sizes the actual tile). Pure.
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
        + VAE_RESIDENT_GIB
        + (DECODE_ACCUM_BYTES_PER_PIXEL * px) / GIB
        + DECODE_MIN_TILE_GIB
}

/// Everything the Krea control lane needs to decide its memory-adaptation plan against a device budget.
/// The caller fills this from the load spec + request; [`plan_control_adaptation`] turns it into a
/// [`MemoryPlan`] via the shared policy.
#[derive(Clone, Copy, Debug)]
pub struct ControlLaneInputs<'a> {
    /// Architecture config (block count, hidden width, FFN width).
    pub cfg: &'a Krea2Config,
    /// Copied branch blocks `N` (`Krea2ControlBranch::num_blocks`; the S0 recipe is 7).
    pub branch_blocks: usize,
    /// The base DiT / text-encoder quant tier — `None` = dense bf16, `Some(Q4/Q8)` = packed (sc-11727).
    pub base_tier: Option<Quant>,
    /// Requested render size.
    pub width: u32,
    pub height: u32,
    /// Whether the lane can drop to `OffloadPolicy::Sequential` (the Krea control lane wires it).
    pub supports_sequential: bool,
    /// Whether the pose branch may be packed to the base tier at load (sc-11748). `true` ⇒ the
    /// branch-quant lever is offered (bf16 → `base_tier`); `false` ⇒ the branch stays bf16.
    pub allow_branch_quant: bool,
    /// Candidate resolution scales (largest-first, in `(0,1)`) offered as the last resort (sc-11749).
    pub resolution_scales: &'a [f64],
}

/// Decide the Krea control lane's memory-adaptation plan (sc-11750): assemble the per-lane
/// [`StagePeaks`] estimator + [`LaneLevers`] and run them through the shared
/// [`plan_memory_adaptation`]. `safe_gib` is the device budget ([`mlx_gen::memory::safe_budget_gib`]);
/// the returned [`MemoryPlan`] tells the caller which levers to apply (residency / decode tiling /
/// branch quant / resolution) and whether the render fits at all.
pub fn plan_control_adaptation(safe_gib: f64, inputs: ControlLaneInputs<'_>) -> MemoryPlan {
    // The branch packs to the base tier when the lever is offered (and the base is itself packed);
    // otherwise it stays bf16. `branch_quant_saves` is the bf16 → base-tier delta on the branch weights.
    let branch_quant_saves_gib = if inputs.allow_branch_quant && inputs.base_tier.is_some() {
        (branch_bytes(inputs.cfg, inputs.branch_blocks, None)
            - branch_bytes(inputs.cfg, inputs.branch_blocks, inputs.base_tier))
            / GIB
    } else {
        0.0
    };

    let levers = LaneLevers {
        text_resident_gib: text_resident_gib(inputs.base_tier),
        supports_sequential: inputs.supports_sequential,
        branch_quant_saves_gib,
        resolution_scales: inputs.resolution_scales,
    };

    plan_memory_adaptation(safe_gib, levers, |scale| {
        // Scale the render dimensions; keep them multiple-of-16 legal (the DiT/VAE alignment) by
        // flooring to the nearest 16 — the same alignment the pipeline validates.
        let w = (((inputs.width as f64 * scale) as u32) / 16 * 16).max(16);
        let h = (((inputs.height as f64 * scale) as u32) / 16 * 16).max(16);
        // The branch is estimated at bf16 here; the branch-quant *saving* is applied by the policy via
        // `branch_quant_saves_gib`, so the peaks must NOT also pre-apply it (double counting).
        StagePeaks {
            denoise_ex_text_gib: control_denoise_peak_ex_text_gib(
                inputs.cfg,
                inputs.branch_blocks,
                inputs.base_tier,
                None,
                w,
                h,
            ),
            decode_ex_text_gib: qwen_vae_decode_peak_ex_text_gib(
                inputs.cfg,
                inputs.branch_blocks,
                inputs.base_tier,
                None,
                w,
                h,
            ),
            decode_tiled_floor_ex_text_gib: Some(qwen_vae_decode_tiled_floor_ex_text_gib(
                inputs.cfg,
                inputs.branch_blocks,
                inputs.base_tier,
                None,
                w,
                h,
            )),
        }
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use mlx_gen::gen_core::mempolicy::Lever;
    use mlx_gen::OffloadPolicy;

    const SCALES: [f64; 2] = [0.75, 0.5];

    fn inputs(base_tier: Option<Quant>) -> ControlLaneInputs<'static> {
        // `Krea2Config` is not `'static`; the tests build one and leak it once (test-only) so the
        // `'a` borrow lives long enough for the `'static` return. Simpler than threading a lifetime
        // through every case.
        static CFG: std::sync::OnceLock<Krea2Config> = std::sync::OnceLock::new();
        let cfg = CFG.get_or_init(Krea2Config::turbo);
        ControlLaneInputs {
            cfg,
            branch_blocks: 7,
            base_tier,
            width: 1024,
            height: 1024,
            supports_sequential: true,
            allow_branch_quant: true,
            resolution_scales: &SCALES,
        }
    }

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
        // The single-pass decode peak dwarfs its tiled floor (tiling's whole point).
        let floor =
            qwen_vae_decode_tiled_floor_ex_text_gib(&cfg, 7, Some(Quant::Q4), None, 1024, 1024);
        assert!(
            floor < dc_1024 * 0.5,
            "floor {floor:.1} vs single {dc_1024:.1}"
        );
    }

    /// Large-memory Mac (128 GB → ~100 GiB safe): the fast path — NOTHING engages, full res, Resident,
    /// bf16 branch, single-pass decode. The sc-11750 regression guard, end to end through the Krea
    /// estimators (not just the synthetic gen-core model).
    #[test]
    fn large_memory_mac_engages_nothing() {
        let plan = plan_control_adaptation(100.0, inputs(Some(Quant::Q4)));
        assert!(plan.engaged.is_empty(), "no lever on a big Mac: {plan:?}");
        assert_eq!(plan.residency, OffloadPolicy::Resident);
        assert!(!plan.tile_decode && !plan.quantize_branch);
        assert_eq!(plan.resolution_scale, 1.0);
        assert!(plan.feasible);
    }

    /// A 32 GB Mac (~24 GiB usable) with a q4 base: levers engage in cost order and the render fits.
    /// The decode spike (~30 GB single-pass) is the first thing over budget → tiling must engage.
    #[test]
    fn constrained_mac_engages_levers_in_cost_order_and_fits() {
        let plan = plan_control_adaptation(24.0, inputs(Some(Quant::Q4)));
        assert!(
            plan.feasible,
            "a 32GB Mac must fit a q4 control render: {plan:?}"
        );
        assert!(
            !plan.engaged.is_empty(),
            "something must engage under 24 GiB"
        );
        // Cost order: whatever engaged is ascending (never quant before residency, etc.).
        let mut sorted = plan.engaged.clone();
        sorted.sort();
        assert_eq!(
            plan.engaged, sorted,
            "levers must be cost-ordered: {plan:?}"
        );
        // The decode spike is the dominant peak → decode tiling is engaged.
        assert!(
            plan.tile_decode,
            "decode tiling must engage on a 32GB Mac: {plan:?}"
        );
        assert!(
            plan.projected_peak_gib <= 24.0,
            "projected peak over budget: {plan:?}"
        );
    }

    /// Residency is tried before the quality-costing resolution lever: a mid budget engages the cheap
    /// levers (residency/tiling/quant) but keeps full resolution.
    #[test]
    fn keeps_full_resolution_until_forced() {
        // 24 GiB fits with the cheap levers (proven above) → resolution must stay 1.0.
        let plan = plan_control_adaptation(24.0, inputs(Some(Quant::Q4)));
        assert_eq!(
            plan.resolution_scale, 1.0,
            "must not drop res prematurely: {plan:?}"
        );
        assert!(!plan.engaged.contains(&Lever::ResolutionReduction));
    }
}
