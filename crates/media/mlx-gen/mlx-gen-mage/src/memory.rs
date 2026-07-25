//! Measured Mage-Flow generation fit model.
//!
//! The 512² anchors are MLX `get_peak_memory()` measurements of the complete generation plus
//! mean-latent encode/decode acceptance path. The VAE term is the independently measured linear
//! decode fit from the five-geometry sweep. Keeping the estimator pure makes the worker gate
//! testable without provoking MLX's process-terminating default OOM handler.

use mlx_gen::{Error, Quant, Result};
use mlx_rs::memory::get_memory_limit;

const GIB_TO_DECIMAL_GB: f64 = 1.073_741_824;
const SAFE_FRAC: f64 = 0.85;

/// Fixed + linear VAE decode high-water in decimal GB.
pub const VAE_FIXED_GB: f64 = 0.267;
pub const VAE_GB_PER_MEGAPIXEL: f64 = 2.039;

/// Complete measured 512² peaks in decimal GB after source-map draining.
fn calibration_peak_gb(tier: Option<Quant>) -> f64 {
    match tier {
        None => 17.660,
        Some(Quant::Q8) => 10.012,
        Some(Quant::Q4) => 5.940,
        Some(Quant::Nvfp4) => f64::INFINITY,
    }
}

pub fn vae_peak_gb(width: u32, height: u32) -> f64 {
    VAE_FIXED_GB + VAE_GB_PER_MEGAPIXEL * (width as f64 * height as f64 / 1e6)
}

/// Conservative complete-pipeline peak. Weight/DiT/TE residency is the tier's measured 512²
/// anchor minus that anchor's VAE term; only the empirically resolution-linear VAE spike grows.
pub fn generation_peak_gb(tier: Option<Quant>, width: u32, height: u32, count: u32) -> f64 {
    let resident = calibration_peak_gb(tier) - vae_peak_gb(512, 512);
    resident + vae_peak_gb(width, height) * count.max(1) as f64
}

fn resolve_live_budget_gb(limit_bytes: usize, max_buffer_gib: Option<f64>) -> Result<f64> {
    let limit_gib = limit_bytes as f64 / (1024.0 * 1024.0 * 1024.0);
    let Some(max_buffer_gib) = max_buffer_gib.filter(|value| value.is_finite() && *value > 0.0)
    else {
        return Err(Error::Msg(
            "mage_flow: Metal max-buffer probe is unavailable; refusing to use a guessed \
             unified-memory budget"
                .into(),
        ));
    };
    if !limit_gib.is_finite() || limit_gib < 1.0 {
        return Err(Error::Msg(
            "mage_flow: MLX unified-memory limit is unavailable; refusing to use the shared 8 GiB \
             fallback with MLX's process-terminating default OOM handler"
                .into(),
        ));
    }
    Ok((limit_gib * SAFE_FRAC).min(max_buffer_gib) * GIB_TO_DECIMAL_GB)
}

/// Query the actual MLX/Metal limits used by the allocator. Unlike the generic tiling helper, this
/// provider cannot safely substitute an 8 GiB guess: its smallest measured complete tier is already
/// ~5.94 GB at 512² and MLX terminates the process rather than returning a catchable allocation error.
pub fn production_safe_budget_gb() -> Result<f64> {
    resolve_live_budget_gb(get_memory_limit(), mlx_gen::memory::max_buffer_length_gib())
}

/// Fail before generation when the measured peak does not fit the injected safe budget.
///
/// A non-finite/non-positive budget is treated as an unavailable wired-memory probe, not as
/// permission to continue. This is deliberate: MLX's default allocation error handler exits the
/// process, so optimistic execution cannot produce a catchable application error.
pub fn ensure_generation_fits(
    tier: Option<Quant>,
    width: u32,
    height: u32,
    count: u32,
    safe_gb: f64,
) -> Result<()> {
    if !safe_gb.is_finite() || safe_gb <= 0.0 {
        return Err(Error::Msg(
            "mage_flow: unified-memory budget is unavailable; refusing to enter MLX's \
             process-terminating default OOM handler"
                .into(),
        ));
    }
    let required = generation_peak_gb(tier, width, height, count);
    if !required.is_finite() || required > safe_gb {
        return Err(Error::Msg(format!(
            "mage_flow: requested {width}x{height} count {count} at {tier:?} requires an estimated \
             {required:.2} GB, exceeding the safe wired-memory budget {safe_gb:.2} GB"
        )));
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn measured_tier_order_and_vae_curve_are_pinned() {
        assert!(
            generation_peak_gb(Some(Quant::Q4), 512, 512, 1)
                < generation_peak_gb(Some(Quant::Q8), 512, 512, 1)
        );
        assert!(
            generation_peak_gb(Some(Quant::Q8), 512, 512, 1)
                < generation_peak_gb(None, 512, 512, 1)
        );
        assert!((vae_peak_gb(2048, 2048) - 8.818).abs() < 0.002);
    }

    #[test]
    fn gate_fails_closed_for_unknown_or_insufficient_budget() {
        assert!(ensure_generation_fits(Some(Quant::Q4), 512, 512, 1, f64::NAN).is_err());
        assert!(ensure_generation_fits(Some(Quant::Q4), 512, 512, 1, 5.0).is_err());
        assert!(ensure_generation_fits(Some(Quant::Q4), 512, 512, 1, 6.0).is_ok());
    }

    #[test]
    fn production_budget_resolver_has_no_guess_fallback() {
        assert!(resolve_live_budget_gb(0, Some(32.0)).is_err());
        assert!(resolve_live_budget_gb(64 << 30, None).is_err());
        assert!(resolve_live_budget_gb(64 << 30, Some(f64::NAN)).is_err());
        let got = resolve_live_budget_gb(64 << 30, Some(32.0)).unwrap();
        assert!((got - 32.0 * GIB_TO_DECIMAL_GB).abs() < 1e-9);
    }
}
