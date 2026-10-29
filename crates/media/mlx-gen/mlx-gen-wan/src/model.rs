//! `mlx-gen-wan` model entries: the Wan2.2 **TI2V-5B** (`wan2_2_ti2v_5b`, dense, z48 VAE — fully
//! wired, sc-2680), the Wan2.2 **T2V-A14B** (`wan2_2_t2v_14b`, dual-expert MoE, z16 VAE — fully wired
//! here on the S1–S5 core), and the Wan2.2 **I2V-A14B** (`wan2_2_i2v_14b`, dual-expert MoE,
//! channel-concat image conditioning, in_dim 36 — sc-2681), plus their registry self-registration.
//!
//! The 5B [`Wan`] struct runs the complete dense pipeline (sc-2680) — [`Wan::generate`]: UMT5-XXL
//! encode → the dense [`crate::pipeline::denoise`] (T2V) or the [`denoise_ti2v`] per-token mask-blend (TI2V, single- or
//! multi-keyframe) → z48 VAE decode → RGB8 frames, with Q4/Q8 + LoRA. The shared [`Wan14b`] struct
//! serves both A14B variants — [`Wan14b::generate`] runs the
//! complete pipeline: UMT5-XXL encode → (I2V only) build the channel-concat conditioning `y` →
//! per-step dual-expert MoE denoise (boundary-switched high/low experts, [`denoise_moe`]) → z16 VAE
//! decode → RGB8 frames, **staging** each heavy component (T5, the two 27 GB experts, the VAE) in and
//! out to bound peak memory (mirrors `generate_wan.py`). The I2V variant differs only by the `y`
//! conditioning (the image's first-frame VAE latent + temporal mask, channel-concatenated to in_dim
//! 36) and the max-area resolution cap.

use std::path::PathBuf;

use mlx_gen::gen_core::{
    adapter_stack_resident_bytes, AdapterResidencyMode, ApproximationPlan, ApproximationSurface,
};
use mlx_gen::tiling::VaeTiling;
use mlx_gen::weights::Weights;
use mlx_gen::{
    AdapterApplyReport, AdapterSpec, CancelFlag, Capabilities, Conditioning, ConditioningKind,
    Error, GenerationOutput, GenerationRequest, Generator, Image, LoadPhase, LoadSpec, Modality,
    ModelDescriptor, MoeExpert, OffloadPolicy, Precision, Progress, Quant, Result, WeightsSource,
};
use mlx_rs::random;
use mlx_rs::Array;

use crate::adapters::{
    adapter_apply_reports, apply_wan_adapters_additive, merge_wan_adapters, reject_loha_on_packed,
    warn_skipped_adapters, WanLoraReport,
};
use crate::config::{WanModelConfig, WanQuant, MIN_SIZE};
use crate::feature_cache::TrunkCache;
use crate::pipeline::{
    align_dim, auto_tiling_budgeted_z16, build_i2v_y, build_ti2v_keyframe_z, build_ti2v_mask,
    crossing_index, decode_to_frames, decode_to_frames_22, denoise_approx, denoise_curated,
    denoise_moe, denoise_moe_curated, denoise_moe_curated_swapped, denoise_range, denoise_ti2v,
    frames_to_images, latent_shape, preflight_denoise_memory_guard, preprocess_ti2v_image,
    refuse_unwired_approximation, reject_off_grid, reject_over_area, resolve_sampler_knobs,
    seq_len, staged_expert_swap, ti2v_blend_init, Expert,
};
use crate::scheduler::{make_scheduler, SolverKind, WanScheduler};
use crate::text_encoder::encode_text_staged_for_tier;
use crate::token_pruning::TokenPruner;
use crate::transformer::WanTransformer;

// sc-27342: the descriptors, approximation bridges, TI2V-5B and A14B providers, denoise
// residency facts and request resolution live in sibling modules; the shared VAE / sampler surface
// and the registrations stay here.
mod a14b;
mod a14b_generate;
mod approximation;
mod denoise_memory;
mod descriptor;
mod request;
mod ti2v;
mod ti2v_generate;
pub use a14b::*;
pub(crate) use approximation::*;
pub(crate) use denoise_memory::*;
pub use descriptor::*;
use request::*;
pub use ti2v::*;

/// Concrete z48 VAE assigned to the TI2V-5B route.
pub type Ti2vProviderVae = crate::vae22::Wan22Vae;
/// Concrete z16 VAE assigned to both A14B routes.
pub type A14bProviderVae = crate::vae::WanVae;

const fn provider_vae_stride(vae: VaeTiling) -> (usize, usize, usize) {
    (
        vae.temporal_scale as usize,
        vae.spatial_scale as usize,
        vae.spatial_scale as usize,
    )
}

/// Resolve the TI2V-5B route's load-bearing VAE geometry.
pub fn ti2v_vae_tiling(provider_id: &str) -> Option<mlx_gen::tiling::VaeTiling> {
    (provider_id == MODEL_ID).then_some(Ti2vProviderVae::VAE_TILING)
}

/// Resolve either A14B route's load-bearing VAE geometry.
pub fn a14b_vae_tiling(provider_id: &str) -> Option<mlx_gen::tiling::VaeTiling> {
    matches!(provider_id, MODEL_ID_T2V_14B | MODEL_ID_I2V_14B)
        .then_some(A14bProviderVae::VAE_TILING)
}

/// The curated unified solvers (epic 7114, sc-7121) every Wan generator exposes ADDITIVELY beyond its
/// native solvers — the gen-core-only solvers, routed through `run_flow_sampler` over Wan's own flow-σ
/// schedule ([`crate::pipeline::denoise_curated`] / [`denoise_moe_curated`]).
///
/// A14B residency note (sc-12795): under `Sequential`, these route the expert **per eval** inside the
/// solver callback. This preserves boundary-straddling sub-evaluations while making one high→low
/// evict-then-load swap, so only one expert is resident.
const WAN_CURATED_SAMPLERS: [&str; 4] = ["euler_ancestral", "heun", "dpmpp_sde", "ddim"];

/// Wan's native flow-SNR solvers ([`crate::scheduler::SolverKind`]), advertised under the curated
/// gen-core vocabulary (epic 7114 sc-7296). `uni_pc`/`dpmpp_2m`/`euler` route to the NATIVE flow-SNR
/// solver — NOT the gen-core VE-space `uni_pc`/`dpmpp_2m` (`λ = −ln σ`), which would not reproduce Wan's
/// diffusers FLOW-SNR (`λ = log((1−σ)/σ)`) parity. Sampler names are family labels in this framework
/// (cf. `euler` across prediction types), so Wan's native UniPC IS the `uni_pc` for Wan — the native
/// math is unchanged (the byte-exact N1 default).
const WAN_NATIVE_SAMPLERS: [&str; 3] = ["uni_pc", "euler", "dpmpp_2m"];

/// Legacy sampler spellings (pre-sc-7296) kept advertised so old recipes still validate + reproduce;
/// they map to the same native solvers via [`crate::scheduler::SolverKind::from_name`]. The SceneWorks
/// manifest surfaces only the curated names.
const WAN_LEGACY_SAMPLERS: [&str; 2] = ["unipc", "dpmpp2m"];

/// The dense-5B decode plan. A benchmark scope's fixed-tile control still wins outright — it exists to
/// hold tiles constant across a measurement campaign — and otherwise this defers to the carrier-aware
/// planner, which honors a worker-selected bounded-decode rung and falls back to
/// `auto_tiling_budgeted` when the request carries no memory strategy.
fn dense_decode_tiling(
    request: &GenerationRequest,
    width: u32,
    height: u32,
    out_frames: u32,
) -> Result<Option<mlx_gen::TilingConfig>> {
    match mlx_gen::diagnostics::benchmark_decode_control() {
        Some(control) => Ok(Some(control.tiling_config())),
        None if request.video_mode.as_deref() == Some("image_to_video")
            && request.memory.is_some() =>
        {
            crate::i2v_memory_strategy::decode_tiling(request, width, height, out_frames)
        }
        None => crate::memory_strategy::decode_tiling(request, width, height, out_frames),
    }
}

/// Wan's full per-generation sampler menu: native solvers (curated vocabulary) + the curated gen-core
/// fold-ins + the legacy aliases.
fn wan_samplers() -> Vec<&'static str> {
    let mut s = WAN_NATIVE_SAMPLERS.to_vec();
    s.extend(WAN_CURATED_SAMPLERS);
    s.extend(WAN_LEGACY_SAMPLERS);
    s
}

/// The native-only menu (curated vocabulary + legacy aliases, NO gen-core fold-ins) — for the VACE
/// path, which advertises its native solvers without the `run_flow_sampler` fold-ins.
pub(crate) fn wan_native_samplers() -> Vec<&'static str> {
    let mut s = WAN_NATIVE_SAMPLERS.to_vec();
    s.extend(WAN_LEGACY_SAMPLERS);
    s
}

/// Whether `req.sampler` selects a curated gen-core solver (routed through `run_flow_sampler`) rather
/// than a native Wan solver (handled by `scheduler.rs`). Used to branch the denoise dispatch.
pub(crate) fn is_wan_curated(name: Option<&str>) -> bool {
    matches!(name, Some(n) if WAN_CURATED_SAMPLERS.contains(&n))
}

/// Public provider id: `"wan2_2_ti2v_5b"`.
pub const MODEL_ID: &str = "wan2_2_ti2v_5b";

// The registration constants bridge the crate's rich `Result` into backend-neutral
// `gen_core::Result`.
mlx_gen::register_generators! {
    pub(crate) const TI2V_REGISTRATION = descriptor => load
}

// ===========================================================================================
// Wan2.2 T2V-A14B — dual-expert MoE text→video (the S1–S5 core, fully wired)
// ===========================================================================================

/// Public provider id for the dual-expert MoE T2V model: `"wan2_2_t2v_14b"`.
pub const MODEL_ID_T2V_14B: &str = "wan2_2_t2v_14b";

mlx_gen::register_generators! {
    pub(crate) const T2V_14B_REGISTRATION = descriptor_t2v_14b => load_t2v_14b
}

// ===========================================================================================
// Wan2.2 I2V-A14B — dual-expert MoE image→video (channel-concat conditioning, in_dim 36)
// ===========================================================================================

/// Public provider id for the channel-concat I2V model: `"wan2_2_i2v_14b"`.
pub const MODEL_ID_I2V_14B: &str = "wan2_2_i2v_14b";

mlx_gen::register_generators! {
    pub(crate) const I2V_14B_REGISTRATION = descriptor_i2v_14b => load_i2v_14b
}

#[cfg(test)]
mod tests;
