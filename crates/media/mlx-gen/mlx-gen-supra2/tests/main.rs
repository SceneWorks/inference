//! Supra2-IMG MLX integration tests (sc-24213): weights-free DiT parity against the frozen upstream
//! module plus `#[ignore]`d real-weight component, trajectory and registry-path renders.

mod real_weights;
mod tiny_dit_parity;

/// Row-major f32 host copy (a transposed view must be made contiguous before `as_slice`).
pub fn host(a: &mlx_rs::Array) -> Vec<f32> {
    let a = mlx_gen::array::contiguous(&a.as_dtype(mlx_rs::Dtype::Float32).unwrap()).unwrap();
    mlx_rs::transforms::eval([&a]).unwrap();
    a.as_slice::<f32>().to_vec()
}

/// `(max_abs, mean_abs, ref_absmax)` between two equal-length f32 slices.
pub fn diff_stats(got: &[f32], want: &[f32]) -> (f32, f32, f32) {
    assert_eq!(got.len(), want.len(), "length mismatch");
    let mut max_abs = 0f32;
    let mut sum = 0f64;
    let mut ref_max = 0f32;
    for (&g, &w) in got.iter().zip(want) {
        let d = (g - w).abs();
        max_abs = max_abs.max(d);
        sum += d as f64;
        ref_max = ref_max.max(w.abs());
    }
    (max_abs, (sum / got.len() as f64) as f32, ref_max)
}
