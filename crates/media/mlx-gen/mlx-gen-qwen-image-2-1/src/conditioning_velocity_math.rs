//! CPU-only accounting of observed vectors. No MLX arithmetic bound is asserted.
pub const CACHE_CAP: u64 = 67_108_864;
pub const ACTIVE_FLOOR: u64 = 69_147_475_387;
pub const FREE_CACHE: u64 = 11_142_168_576;
pub const ARITHMETIC: &str = "UNPROVEN_RELAXED_NAX_PRECISION";

#[derive(Default, Debug)]
pub struct Cache {
    pub live: u64,
    pub peak: u64,
}
impl Cache {
    pub fn reserve(&mut self, bytes: u64) -> Result<(), &'static str> {
        let next = self.live.checked_add(bytes).ok_or("CPU cache overflow")?;
        if next > CACHE_CAP {
            return Err("CPU cache cap exceeded before copy");
        }
        self.live = next;
        self.peak = self.peak.max(next);
        Ok(())
    }
    pub fn release(&mut self, bytes: u64) {
        self.live = self
            .live
            .checked_sub(bytes)
            .expect("CPU cache ownership accounting");
    }
}

pub fn closure_active(phases: &[u64]) -> Result<u64, &'static str> {
    if phases.len() != 5 || phases.contains(&0) {
        return Err("unsealed phase closure");
    }
    Ok(ACTIVE_FLOOR.max(*phases.iter().max().unwrap()))
}

#[derive(Debug)]
pub struct Metrics {
    pub base_error: f64,
    pub adapted_error: f64,
    pub gain: f64,
    pub projection: f64,
    pub delta_norm2: f64,
    pub identity_residual: f64,
    pub identity_bound: f64,
}
pub fn metrics(base: &[f32], adapted: &[f32], target: &[f32]) -> Result<Metrics, &'static str> {
    if base.is_empty() || base.len() != adapted.len() || base.len() != target.len() {
        return Err("full vector sizes differ");
    }
    let mut eb = 0.0_f64;
    let mut ea = 0.0_f64;
    let mut p = 0.0_f64;
    let mut dd = 0.0_f64;
    let mut absolute = 0.0_f64;
    for ((&b, &a), &t) in base.iter().zip(adapted).zip(target) {
        if !b.is_finite() || !a.is_finite() || !t.is_finite() {
            return Err("nonfinite observed vector");
        }
        let b = f64::from(b) - f64::from(t);
        let a = f64::from(a) - f64::from(t);
        let d = a - b;
        eb += b * b;
        ea += a * a;
        p += b * d;
        dd += d * d;
        absolute += b * b + a * a + 2.0 * (b * d).abs() + d * d;
    }
    let n = base.len() as f64;
    let gain = (eb - ea) / n;
    let projection = p / n;
    let delta_norm2 = dd / n;
    let residual = gain + 2.0 * projection + delta_norm2;
    // Independent CPU scalar binary64 reductions/products only. Conservative
    // gamma covers each ordinary sum and the final identity arithmetic. This
    // is expressly not an IEEE-F32/NAX GEMM or inference-error guarantee.
    let neps = (n + 16.0) * f64::EPSILON;
    if neps >= 0.5 {
        return Err("CPU reduction is too large to bound");
    }
    let bound = 16.0 * (neps / (1.0 - neps)) * (absolute / n).max(f64::MIN_POSITIVE);
    if !residual.is_finite() || residual.abs() > bound {
        return Err("CPU projection identity failed");
    }
    Ok(Metrics {
        base_error: eb / n,
        adapted_error: ea / n,
        gain,
        projection,
        delta_norm2,
        identity_residual: residual,
        identity_bound: bound,
    })
}
pub fn variability(a: &[f32], b: &[f32]) -> Result<(f64, f64), &'static str> {
    if a.is_empty() || a.len() != b.len() {
        return Err("repeat vector size mismatch");
    }
    let mut max = 0.0_f64;
    let mut squared = 0.0;
    for (&a, &b) in a.iter().zip(b) {
        if !a.is_finite() || !b.is_finite() {
            return Err("repeat nonfinite");
        }
        let d = f64::from(a) - f64::from(b);
        max = max.max(d.abs());
        squared += d * d;
    }
    Ok((max, (squared / a.len() as f64).sqrt()))
}

pub fn difference(a: &[f32], b: &[f32]) -> Result<(f64, f64, f64), &'static str> {
    if a.is_empty() || a.len() != b.len() {
        return Err("difference vector size mismatch");
    }
    let mut absolute = 0.0_f64;
    let mut squared = 0.0_f64;
    let mut maximum = 0.0_f64;
    for (&a, &b) in a.iter().zip(b) {
        if !a.is_finite() || !b.is_finite() {
            return Err("difference vector nonfinite");
        }
        let delta = f64::from(a) - f64::from(b);
        absolute += delta.abs();
        squared += delta * delta;
        maximum = maximum.max(delta.abs());
    }
    let n = a.len() as f64;
    Ok((absolute / n, (squared / n).sqrt(), maximum))
}
pub fn localization(gains: &[[f64; 2]; 4]) -> &'static str {
    let signs = gains.map(|g| {
        if !g[0].is_finite()
            || !g[1].is_finite()
            || g[0] == 0.0
            || g[1] == 0.0
            || g[0].is_sign_negative() != g[1].is_sign_negative()
        {
            0
        } else if g[0] < 0.0 {
            -1
        } else {
            1
        }
    });
    if signs.contains(&0) {
        return "UNRESOLVED";
    }
    if signs[0] == signs[2] && signs[1] == signs[3] && signs[0] != signs[1] {
        "OBSERVATIONAL_LANGUAGE_PATTERN"
    } else if signs[0] == signs[3] && signs[1] == signs[2] && signs[0] != signs[1] {
        "OBSERVATIONAL_DIT_REPRESENTATION_DTYPE_KERNEL_PATTERN"
    } else {
        "UNRESOLVED"
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn projection_direction_is_not_movement() {
        let base = [2.0, 4.0, 6.0];
        let target = [1.0, 1.0, 1.0];
        let toward = metrics(&base, &[1.5, 2.0, 3.0], &target).unwrap();
        let away = metrics(&base, &[2.5, 6.0, 9.0], &target).unwrap();
        assert!(toward.gain > 0.0 && toward.projection < 0.0);
        assert!(away.gain < 0.0 && away.projection > 0.0);
        assert!(toward.identity_residual.abs() <= toward.identity_bound);
        assert!(metrics(&base, &[f32::NAN, 1.0, 1.0], &target).is_err());
        assert!(metrics(&base, &[1.0], &target).is_err());
        assert!(metrics(&base, &base, &target).unwrap().gain == 0.0);
        assert!(
            metrics(&base, &[1.5, 2.0, 3.0], &[8.0, 8.0, 8.0])
                .unwrap()
                .gain
                < 0.0
        );
    }
    #[test]
    fn repeat_patterns_require_both_repetitions_and_backgrounds() {
        assert_eq!(
            localization(&[[1.0; 2], [-1.0; 2], [1.0; 2], [-1.0; 2]]),
            "OBSERVATIONAL_LANGUAGE_PATTERN"
        );
        assert_eq!(
            localization(&[[1.0; 2], [-1.0; 2], [-1.0; 2], [1.0; 2]]),
            "OBSERVATIONAL_DIT_REPRESENTATION_DTYPE_KERNEL_PATTERN"
        );
        assert_eq!(
            localization(&[[1.0, -1.0], [-1.0; 2], [1.0; 2], [-1.0; 2]]),
            "UNRESOLVED"
        );
        assert_eq!(
            localization(&[[0.0; 2], [-1.0; 2], [1.0; 2], [-1.0; 2]]),
            "UNRESOLVED"
        );
        assert_eq!(
            variability(&[1.0, 2.0], &[1.0, 4.0]).unwrap(),
            (2.0, 2.0_f64.sqrt())
        );
        assert_eq!(
            difference(&[1.0, 4.0], &[3.0, 2.0]).unwrap(),
            (2.0, 2.0, 2.0)
        );
        assert!(difference(&[1.0], &[]).is_err());
        assert!(difference(&[f32::NAN], &[0.0]).is_err());
    }
    #[test]
    fn cache_refuses_before_copy_and_closure_never_underprices() {
        let mut c = Cache::default();
        c.reserve(CACHE_CAP).unwrap();
        assert!(c.reserve(1).is_err());
        assert_eq!(c.live, CACHE_CAP);
        c.release(CACHE_CAP);
        c.reserve(8).unwrap();
        assert_eq!(c.peak, CACHE_CAP);
        assert_eq!(closure_active(&[1; 5]).unwrap(), ACTIVE_FLOOR);
        assert_eq!(
            closure_active(&[ACTIVE_FLOOR + 1; 5]).unwrap(),
            ACTIVE_FLOOR + 1
        );
        assert!(closure_active(&[1; 4]).is_err());
        assert!(closure_active(&[0; 5]).is_err());
        assert_eq!(ACTIVE_FLOOR + FREE_CACHE, 80_289_643_963);
        assert_eq!(ARITHMETIC, "UNPROVEN_RELAXED_NAX_PRECISION");
    }
}
