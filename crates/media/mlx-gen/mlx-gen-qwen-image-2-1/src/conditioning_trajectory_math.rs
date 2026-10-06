//! Binary64 accounting of autonomous trajectories, never a native arithmetic bound.
pub const STEPS: usize = 8;
pub const TRAJECTORIES: usize = 4;
pub const CPU_SCRATCH: u64 = 8 * 1024 * 1024;

pub fn validate_schedule(sigmas: &[f32]) -> Result<(), &'static str> {
    if sigmas.len() != STEPS + 1
        || sigmas[0] != 1.0
        || sigmas[STEPS] != 0.0
        || !sigmas.iter().all(|s| s.is_finite())
        || !sigmas.windows(2).all(|s| s[0] > s[1])
    {
        return Err("fixed descending eight-step schedule required");
    }
    Ok(())
}

#[derive(Debug)]
pub struct StepMetrics {
    pub target_path_error: f64,
    pub estimate_error: f64,
    pub update_projection: f64,
    pub update_norm2: f64,
}
pub fn step_metrics(
    x: &[f32],
    velocity: &[f32],
    x0: &[f32],
    noise: &[f32],
    sigma: f32,
    next_sigma: f32,
) -> Result<StepMetrics, &'static str> {
    if x.is_empty()
        || [velocity.len(), x0.len(), noise.len()]
            .iter()
            .any(|&n| n != x.len())
        || !sigma.is_finite()
        || !next_sigma.is_finite()
        || sigma <= next_sigma
        || next_sigma < 0.0
        || sigma > 1.0
    {
        return Err("invalid step vectors or schedule");
    }
    let (s, dt) = (f64::from(sigma), f64::from(next_sigma - sigma));
    let mut result = StepMetrics {
        target_path_error: 0.0,
        estimate_error: 0.0,
        update_projection: 0.0,
        update_norm2: 0.0,
    };
    for (((&x, &v), &target), &noise) in x.iter().zip(velocity).zip(x0).zip(noise) {
        if ![x, v, target, noise].iter().all(|n| n.is_finite()) {
            return Err("nonfinite observed trajectory");
        }
        let (x, v, target, noise) = (
            f64::from(x),
            f64::from(v),
            f64::from(target),
            f64::from(noise),
        );
        let path = (1.0 - s) * target + s * noise;
        let update = dt * v;
        result.target_path_error += (x - path).powi(2);
        result.estimate_error += (x - s * v - target).powi(2);
        // Negative means this actual solver update points toward x0 from its own input.
        result.update_projection += (x - target) * update;
        result.update_norm2 += update * update;
    }
    let n = x.len() as f64;
    result.target_path_error /= n;
    result.estimate_error /= n;
    result.update_projection /= n;
    result.update_norm2 /= n;
    Ok(result)
}

pub fn mean_squared_difference(a: &[f32], b: &[f32]) -> Result<f64, &'static str> {
    if a.is_empty() || a.len() != b.len() {
        return Err("trajectory vector sizes differ");
    }
    let mut sum = 0.0;
    for (&a, &b) in a.iter().zip(b) {
        if !a.is_finite() || !b.is_finite() {
            return Err("nonfinite trajectory comparison");
        }
        sum += (f64::from(a) - f64::from(b)).powi(2);
    }
    Ok(sum / a.len() as f64)
}

pub fn update_residual(x: &[f32], v: &[f32], next: &[f32], dt: f32) -> Result<f64, &'static str> {
    if x.is_empty() || x.len() != v.len() || x.len() != next.len() || !dt.is_finite() || dt >= 0.0 {
        return Err("invalid update comparison");
    }
    let mut max: f64 = 0.0;
    for ((&x, &v), &next) in x.iter().zip(v).zip(next) {
        if ![x, v, next].iter().all(|n| n.is_finite()) {
            return Err("nonfinite update comparison");
        }
        max = max.max((f64::from(next) - f64::from(x) - f64::from(dt) * f64::from(v)).abs());
    }
    // Report only: this is not an asserted exact F32 evaluation order or native bound.
    Ok(max)
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn exact_target_path_and_velocity_have_zero_estimate_error() {
        let m = step_metrics(
            &[3.0, 3.0],
            &[4.0, -4.0],
            &[1.0, 5.0],
            &[5.0, 1.0],
            0.5,
            0.25,
        )
        .unwrap();
        assert_eq!(m.target_path_error, 0.0);
        assert_eq!(m.estimate_error, 0.0);
        assert_eq!(m.update_projection, -2.0);
        assert_eq!(m.update_norm2, 1.0);
        let reversed = step_metrics(
            &[3.0, 3.0],
            &[-4.0, 4.0],
            &[1.0, 5.0],
            &[5.0, 1.0],
            0.5,
            0.25,
        )
        .unwrap();
        assert!(reversed.estimate_error > 0.0 && reversed.update_projection > 0.0);
    }
    #[test]
    fn sigma_zero_is_only_a_final_latent_comparison() {
        assert!(step_metrics(&[1.0], &[1.0], &[1.0], &[2.0], 0.0, 0.0).is_err());
        assert_eq!(mean_squared_difference(&[1.0], &[1.0]).unwrap(), 0.0);
        assert!(mean_squared_difference(&[f32::NAN], &[1.0]).is_err());
        assert!(step_metrics(&[1.0], &[], &[1.0], &[2.0], 0.5, 0.25).is_err());
        assert!(update_residual(&[1.0], &[f32::INFINITY], &[0.0], -0.5).is_err());
    }
    #[test]
    fn update_sign_and_shape_are_discriminated() {
        assert_eq!(update_residual(&[3.0], &[4.0], &[2.0], -0.25).unwrap(), 0.0);
        assert!(update_residual(&[3.0], &[-4.0], &[2.0], -0.25).unwrap() > 0.0);
        assert!(update_residual(&[3.0], &[4.0], &[2.0], 0.25).is_err());
        assert!(update_residual(&[3.0], &[4.0], &[], -0.25).is_err());
    }
    #[test]
    fn fixed_inventory_and_schedule_refuse_shortened_or_nonfinite() {
        let sigmas = [1.0, 0.9, 0.8, 0.7, 0.6, 0.5, 0.3, 0.02, 0.0];
        assert!(validate_schedule(&sigmas).is_ok());
        assert!(validate_schedule(&sigmas[..8]).is_err());
        let mut bad = sigmas;
        bad[3] = f32::NAN;
        assert!(validate_schedule(&bad).is_err());
        assert_eq!(STEPS * TRAJECTORIES, 32);
    }
}
