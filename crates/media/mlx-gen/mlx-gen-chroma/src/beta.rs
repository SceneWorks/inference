//! Beta-distribution sigma schedule for `chroma1_base` (sc-3840).
//!
//! Shared implementation lives in [`mlx_gen::gen_core::sampling::chroma_base_sigmas`].

#[cfg(test)]
fn beta_ppf(p: f64) -> f64 {
    mlx_gen::gen_core::sampling::chroma_beta_ppf(p)
}

/// Base's beta sigma schedule for `steps` denoise steps: length `steps + 1` with a trailing `0.0`.
pub fn base_sigmas(steps: usize) -> Vec<f32> {
    mlx_gen::gen_core::sampling::chroma_base_sigmas(steps)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn beta_ppf_matches_scipy() {
        // scipy.stats.beta.ppf(p, 0.6, 0.6)
        let cases = [
            (0.05, 0.01255446),
            (0.25, 0.17568038),
            (0.5, 0.5),
            (0.75, 0.82431962),
            (0.95, 0.98744554),
        ];
        for (p, want) in cases {
            let got = beta_ppf(p);
            assert!((got - want).abs() < 1e-6, "ppf({p}) = {got}, want {want}");
        }
    }

    #[test]
    fn base_sigmas_match_diffusers() {
        // diffusers FlowMatchEulerDiscreteScheduler(use_beta_sigmas=true), sigmas=linspace(1,1/N,N).
        let s4 = base_sigmas(4);
        let want4 = [1.0, 0.79344, 0.45656, 0.25, 0.0];
        for (g, w) in s4.iter().zip(want4) {
            assert!((g - w).abs() < 1e-4, "N=4: {g} vs {w}");
        }
        let s8 = base_sigmas(8);
        let want8 = [
            1.0, 0.937751, 0.810255, 0.648749, 0.476251, 0.314745, 0.187249, 0.125, 0.0,
        ];
        for (g, w) in s8.iter().zip(want8) {
            assert!((g - w).abs() < 1e-4, "N=8: {g} vs {w}");
        }
    }
}
