//! S4/S5 — the **T2V generation pipeline**: the denoise loop + CFG + VAE decode + frame assembly
//! that turns latents into video. Port of `generate_wan.py`'s `generate_video` — both the
//! single-model dense path ([`denoise`], S4) and the dual-expert MoE path ([`denoise_moe`], S5).
//!
//! This is **reusable machinery**, not a model: the dense loop is exactly what each Wan2.2-A14B MoE
//! expert runs (the MoE adds only the per-step boundary swap) and what the 5B runs (sc-2680, with
//! its z48 VAE). The concrete `Generator::generate` wiring lands in `model.rs`.
//!
//! Shapes are channels-first **`[C, F, H, W]`** (no batch dim) for the latents + scheduler. CFG runs
//! cond + uncond as a **single batched B=2 forward** ([`WanTransformer::forward_cached`]) — the shared
//! latent is patchified once and broadcast across the batch, so each per-step GPU kernel launches once
//! instead of twice (the small-seq win, sc-2853); it stays bit-identical to two B=1 forwards since
//! attention never mixes batch elements. The per-block cross-attention K/V and the RoPE cos/sin are
//! **precomputed once per expert** before the loop (the reference's `prepare_cross_kv` / `prepare_rope`)
//! and reused across all steps, instead of recomputed every forward.

use mlx_rs::memory::{get_active_memory, get_memory_limit};
use mlx_rs::ops::{add, concatenate_axis, maximum, minimum, multiply, subtract};
use mlx_rs::Array;

use mlx_gen::array::scalar;
use mlx_gen::image::resize_lanczos_u8;
use mlx_gen::tiling::{
    budgeted_plan, SpatialTiling, TemporalOverlapPolicy, TileCandidates, TilingBudgetError,
    TilingConfig, VaeTiling,
};
use mlx_gen::{default_seed, CancelFlag, Error, GenerationRequest, Image, Progress, Result};

use crate::scheduler::{compute_sigmas, make_scheduler, SolverKind, WanScheduler};
use crate::transformer::WanTransformer;
use crate::vae::WanVae;
use crate::vae22::Wan22Vae;

// sc-27342: geometry, the memory guard, VAE budgets and tiling plans, the denoise loops,
// the MoE route, decode and image conditioning live in sibling modules.
mod conditioning;
mod decode;
mod denoise;
mod geometry;
mod memory_guard;
mod moe;
mod vae16_tiling;
mod vae22_tiling;
mod vae_budget;
pub use conditioning::*;
pub use decode::*;
pub use denoise::*;
pub use geometry::*;
pub use memory_guard::*;
pub use moe::*;
pub use vae16_tiling::*;
pub use vae22_tiling::*;
use vae_budget::*;

#[cfg(test)]
mod planned_decode_tests;

#[cfg(test)]
#[path = "sc20686_hooks_tests.rs"]
mod sc20686_hooks_tests;

#[cfg(test)]
mod tests;

#[cfg(test)]
mod denoise_activation_tests;
