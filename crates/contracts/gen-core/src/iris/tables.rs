//! The host-side `Vec<f32>` tables both Iris backends build before uploading to a device: the
//! RoPE angle grids, the text refiner's additive key mask, the timestep sinusoid frequencies and
//! the `PixelEmbedder` 2D sincos table. One definition, so the MLX and Candle twins cannot drift;
//! each backend only wraps the result in its own tensor type.

/// Width of `TimestepEmbedder`'s sinusoid bank (`cos ‖ sin`, so half as many frequencies).
pub const TIMESTEP_FREQUENCY_DIM: usize = 256;

/// `rope_2d(head_dim, height, width, theta, scale, aspect="isotropic", frame_pairs=0)` angles,
/// `[height·width, 2·(head_dim/4)]` row-major: `head_dim/4` frequencies, interleaved `(x, y)` per
/// pair along the last axis, both axes at one step `scale / (max(h, w) − 1)`.
pub fn rope_2d_angles(
    head_dim: usize,
    height: usize,
    width: usize,
    theta: f32,
    scale: f32,
) -> Vec<f32> {
    let n_pairs = head_dim / 4;
    let freqs: Vec<f32> = (0..n_pairs)
        .map(|j| 1.0 / theta.powf((4 * j) as f32 / head_dim as f32))
        .collect();
    let step = scale / ((height.max(width) as f32) - 1.0).max(1.0);
    let mut angles = Vec::with_capacity(height * width * 2 * n_pairs);
    for r in 0..height {
        let y = r as f32 * step;
        for c in 0..width {
            let x = c as f32 * step;
            for f in &freqs {
                angles.push(x * f);
                angles.push(y * f);
            }
        }
    }
    angles
}

/// `rope_1d(head_dim, length, theta)` angles, `[length, head_dim/2]`: standard 1-D RoPE over
/// integer positions.
pub fn rope_1d_angles(head_dim: usize, length: usize, theta: f32) -> Vec<f32> {
    let pairs = head_dim / 2;
    let freqs: Vec<f32> = (0..pairs)
        .map(|j| 1.0 / theta.powf((2 * j) as f32 / head_dim as f32))
        .collect();
    let mut angles = Vec::with_capacity(length * pairs);
    for p in 0..length {
        for f in &freqs {
            angles.push(p as f32 * f);
        }
    }
    angles
}

/// `TransformerTextEmbedder`'s key mask, additive `[B, 1, T, T]` (row-major): a key is visible
/// when it is real (`mask = 1`) OR on the query's own diagonal, else `−∞`.
pub fn key_padding_additive(mask: &[Vec<i32>], tokens: usize) -> Vec<f32> {
    let b = mask.len();
    let mut data = vec![0f32; b * tokens * tokens];
    for (bi, row) in mask.iter().enumerate() {
        for i in 0..tokens {
            for j in 0..tokens {
                if row[j] == 0 && i != j {
                    data[(bi * tokens + i) * tokens + j] = f32::NEG_INFINITY;
                }
            }
        }
    }
    data
}

/// `TimestepEmbedder`'s [`TIMESTEP_FREQUENCY_DIM`]` / 2` frequencies `exp(−ln(max_period)·k/n)`.
pub fn timestep_freqs(max_period: f64) -> Vec<f32> {
    let n = TIMESTEP_FREQUENCY_DIM / 2;
    let neg_log = -(max_period.ln()) as f32;
    (0..n)
        .map(|k| (neg_log * k as f32 / n as f32).exp())
        .collect()
}

/// `PixelEmbedder`'s fixed full-resolution 2D sincos table, `[height, width, dim]` row-major:
/// per pixel `sin(c·ω) ‖ cos(c·ω) ‖ sin(r·ω) ‖ cos(r·ω)` with `dim/4` frequencies
/// `ω_k = 10000^(−k / (dim/4))`, computed in f64.
pub fn pixel_sincos_table(height: usize, width: usize, dim: usize) -> Vec<f32> {
    let half = dim / 2; // per axis
    let quarter = half / 2; // frequencies per axis
    let omega: Vec<f64> = (0..quarter)
        .map(|k| 1.0 / 10_000f64.powf(k as f64 / (half as f64 / 2.0)))
        .collect();
    let mut data = vec![0f32; height * width * dim];
    for r in 0..height {
        for c in 0..width {
            let row = &mut data[(r * width + c) * dim..(r * width + c + 1) * dim];
            for (k, w) in omega.iter().enumerate() {
                let (xc, yr) = (c as f64 * w, r as f64 * w);
                row[k] = xc.sin() as f32;
                row[quarter + k] = xc.cos() as f32;
                row[half + k] = yr.sin() as f32;
                row[half + quarter + k] = yr.cos() as f32;
            }
        }
    }
    data
}

#[cfg(test)]
mod tests {
    use super::*;

    fn close(a: f32, b: f32) -> bool {
        (a - b).abs() <= 1e-6 * b.abs().max(1.0)
    }

    #[test]
    fn rope_2d_interleaves_x_then_y_on_one_isotropic_step() {
        // 3 rows × 2 cols, head_dim 8 → 2 frequencies {1, 1/θ^(1/2)}, step 1/(3−1).
        let (head_dim, h, w, theta, scale) = (8, 3, 2, 100.0f32, 1.0f32);
        let a = rope_2d_angles(head_dim, h, w, theta, scale);
        let pairs = 2 * (head_dim / 4);
        assert_eq!(a.len(), h * w * pairs);
        let step = 0.5f32;
        let f1 = 1.0 / theta.powf(0.5);
        // position (r=2, c=1): x = 1·step, y = 2·step for both frequencies.
        let at = &a[(2 * w + 1) * pairs..(2 * w + 2) * pairs];
        let want = [step, 2.0 * step, step * f1, 2.0 * step * f1];
        assert!(
            at.iter().zip(&want).all(|(g, w)| close(*g, *w)),
            "{at:?} vs {want:?}"
        );
    }

    #[test]
    fn rope_1d_rotates_integer_positions_at_the_standard_frequencies() {
        let (head_dim, len, theta) = (6, 4, 10_000.0f32);
        let a = rope_1d_angles(head_dim, len, theta);
        assert_eq!(a.len(), len * head_dim / 2);
        let want: Vec<f32> = (0..3)
            .map(|j| 3.0 / theta.powf((2 * j) as f32 / head_dim as f32))
            .collect();
        assert!(
            a[9..12].iter().zip(&want).all(|(g, w)| close(*g, *w)),
            "{:?} vs {want:?}",
            &a[9..12]
        );
    }

    #[test]
    fn key_padding_hides_pad_keys_except_each_querys_own_diagonal() {
        let mask = vec![vec![1, 1, 0], vec![1, 0, 0]];
        let m = key_padding_additive(&mask, 3);
        let at = |b: usize, i: usize, j: usize| m[(b * 3 + i) * 3 + j];
        assert_eq!(m.len(), 2 * 9);
        for (b, row) in mask.iter().enumerate() {
            for i in 0..3 {
                for (j, real) in row.iter().enumerate() {
                    let visible = *real == 1 || i == j;
                    assert_eq!(
                        at(b, i, j),
                        if visible { 0.0 } else { f32::NEG_INFINITY },
                        "batch {b} query {i} key {j}"
                    );
                }
            }
        }
    }

    #[test]
    fn timestep_freqs_decay_geometrically_from_one_to_the_period() {
        let f = timestep_freqs(10.0);
        assert_eq!(f.len(), TIMESTEP_FREQUENCY_DIM / 2);
        assert_eq!(f[0], 1.0);
        assert!(close(f[64], 10f32.powf(-0.5)), "{}", f[64]);
        assert!(close(f[127], 10f32.powf(-127.0 / 128.0)), "{}", f[127]);
    }

    #[test]
    fn pixel_table_lays_out_sin_cos_of_column_then_row() {
        let (h, w, dim) = (2, 3, 8); // 2 frequencies per axis: ω = {1, 1/100}
        let t = pixel_sincos_table(h, w, dim);
        assert_eq!(t.len(), h * w * dim);
        let px = &t[(w + 2) * dim..(w + 3) * dim]; // r = 1, c = 2
        let want = [
            2f64.sin(),
            0.02f64.sin(),
            2f64.cos(),
            0.02f64.cos(),
            1f64.sin(),
            0.01f64.sin(),
            1f64.cos(),
            0.01f64.cos(),
        ]
        .map(|v| v as f32);
        assert!(
            px.iter().zip(&want).all(|(g, w)| close(*g, *w)),
            "{px:?} vs {want:?}"
        );
    }
}
