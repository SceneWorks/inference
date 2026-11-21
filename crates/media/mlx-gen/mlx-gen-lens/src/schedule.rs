//! Lens sampling schedule + CFG (sc-3170).
//!
//! Shared schedule construction lives in [`mlx_gen::gen_core::sampling::{lens_mu, lens_sigmas, lens_timesteps}`],
//! and norm-rescaled CFG lives in [`mlx_gen::gen_core::guidance::cfg_rescale`].

use mlx_gen::FlowMatchEuler;

/// Per-variant sampling defaults (`num_steps`, `guidance_scale`).
#[derive(Clone, Copy, Debug)]
pub struct LensSamplingDefaults {
    pub num_steps: usize,
    pub guidance_scale: f32,
}

/// `microsoft/Lens-Turbo`: distilled **4 steps, guidance 1.0** (≈ no CFG).
pub const TURBO: LensSamplingDefaults = LensSamplingDefaults {
    num_steps: 4,
    guidance_scale: 1.0,
};
/// `microsoft/Lens` (base): **20 steps, guidance 5.0**.
pub const BASE: LensSamplingDefaults = LensSamplingDefaults {
    num_steps: 20,
    guidance_scale: 5.0,
};

/// Build the Lens flow-match schedule for `num_steps` at the given latent grid.
pub fn lens_schedule(num_steps: usize, latent_h: usize, latent_w: usize) -> FlowMatchEuler {
    FlowMatchEuler {
        sigmas: mlx_gen::gen_core::sampling::lens_sigmas(num_steps, latent_h, latent_w),
    }
}

/// The per-step transformer timesteps: the **shifted sigmas** `sigmas[0..num_steps]`.
pub fn timesteps(schedule: &FlowMatchEuler) -> Vec<f32> {
    mlx_gen::gen_core::sampling::lens_timesteps(&schedule.sigmas)
}
