//! Hadamard rotation helpers for ConvRot / W4A8, re-exported from [`candle_layers::convrot`] (`SW-6955`).

pub use candle_layers::convrot::*;

#[cfg(test)]
use candle_core::{DType, Device, Result, Tensor};

#[cfg(test)]
mod tests {
    use super::*;

    fn to_vec2(t: &Tensor) -> Vec<Vec<f32>> {
        t.to_dtype(DType::F32).unwrap().to_vec2::<f32>().unwrap()
    }

    #[test]
    fn power_of_four_predicate() {
        for n in [1usize, 4, 16, 64, 256, 1024, 4096] {
            assert!(is_power_of_four(n), "{n} is 4^k");
        }
        for n in [0usize, 2, 8, 32, 128, 512, 3, 6] {
            assert!(!is_power_of_four(n), "{n} is not 4^k");
        }
    }

    /// `R = H/√N₀` is a symmetric orthogonal involution for every supported order.
    #[test]
    fn regular_hadamard_is_orthogonal_involution() -> Result<()> {
        let dev = Device::Cpu;
        for &g in &[4usize, 16, 64, 256] {
            let r = regular_hadamard(g, &dev)?;
            assert_eq!(r.dims(), &[g, g]);
            let rrt = r.matmul(&r.t()?)?;
            let eye = Tensor::from_vec(
                (0..g * g)
                    .map(|i| if i / g == i % g { 1f32 } else { 0f32 })
                    .collect::<Vec<_>>(),
                (g, g),
                &dev,
            )?;
            let max_off = rrt.sub(&eye)?.abs()?.max_all()?.to_scalar::<f32>()?;
            assert!(max_off < 1e-5, "R·Rᵀ ≠ I for order {g}: max|.| = {max_off}");
            // Symmetric ⇒ its own inverse.
            let asym = r.sub(&r.t()?)?.abs()?.max_all()?.to_scalar::<f32>()?;
            assert!(asym < 1e-6, "R not symmetric for order {g}: {asym}");
        }
        Ok(())
    }

    /// The order-4 matrix is exactly the regular `H₄/2` (not the Sylvester `[[1,1],[1,-1]]⊗…`).
    #[test]
    fn order4_is_regular_not_sylvester() -> Result<()> {
        let r = regular_hadamard(4, &Device::Cpu)?;
        let got = to_vec2(&r);
        let want = [
            [0.5, 0.5, 0.5, -0.5],
            [0.5, 0.5, -0.5, 0.5],
            [0.5, -0.5, 0.5, 0.5],
            [-0.5, 0.5, 0.5, 0.5],
        ];
        for (gr, wr) in got.iter().zip(&want) {
            for (g, w) in gr.iter().zip(wr) {
                assert!((g - w).abs() < 1e-6, "regular H₄/2 mismatch: {g} vs {w}");
            }
        }
        Ok(())
    }

    /// Rotating twice with the involution `R` is the identity (`RHT(RHT(x)) = x`), and the round trip
    /// `RHT(x)·(W·R)ᵀ == x·Wᵀ` holds block-diagonally over a `K` that spans multiple groups.
    #[test]
    fn rotate_round_trip_reconstructs_linear() -> Result<()> {
        let dev = Device::Cpu;
        let (g, groups) = (16usize, 3usize);
        let k = g * groups; // 48, spans 3 blocks
        let r = regular_hadamard(g, &dev)?;

        // Double rotation is identity.
        let x = Tensor::randn(0f32, 1f32, (5, k), &dev)?;
        let twice = convrot_rotate(&convrot_rotate(&x, &r)?, &r)?;
        let back = twice.sub(&x)?.abs()?.max_all()?.to_scalar::<f32>()?;
        assert!(back < 1e-4, "RHT∘RHT ≠ identity: max|.| = {back}");

        // RHT(x)·(W·R)ᵀ == x·Wᵀ, where W·R is the per-block right rotation of W (the stored weight).
        let out = 7usize;
        let w = Tensor::randn(0f32, 1f32, (out, k), &dev)?;
        let wr = convrot_rotate(&w, &r)?; // stored rotated weight
        let xr = convrot_rotate(&x, &r)?;
        let got = xr.matmul(&wr.t()?)?;
        let want = x.matmul(&w.t()?)?;
        let err = got.sub(&want)?.abs()?.max_all()?.to_scalar::<f32>()?;
        assert!(err < 1e-3, "ConvRot round trip max abs err {err}");
        Ok(())
    }
}
