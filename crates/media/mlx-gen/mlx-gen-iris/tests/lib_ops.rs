//! Op-level checks of library helpers (formerly `src` unit tests; sc-25690 moved them here so they
//! run in a CPU child — the lib test binary has no CPU-pinned process and would run them on the
//! default GPU device).

use mlx_gen_iris::nn::modulate;
use mlx_gen_iris::pipeline::preview_image;
use mlx_gen_iris::train::optim::newton_schulz;
use mlx_rs::ops::matmul;
use mlx_rs::transforms::eval;
use mlx_rs::{Array, Dtype};

use crate::common::cpu_process;

/// Upstream's `x * (1 + scale) + shift` keeps a bf16 stream bf16 (the Python `1` is a weak
/// scalar); the `1` here must not promote the PiT stream to f32 either.
#[test]
fn modulate_keeps_a_bf16_stream_in_bf16() {
    cpu_process("lib_ops::modulate_keeps_a_bf16_stream_in_bf16", || {
        let bf16 = |v: &[f32]| {
            Array::from_slice(v, &[1, v.len() as i32])
                .as_dtype(Dtype::Bfloat16)
                .unwrap()
        };
        let out = modulate(&bf16(&[1.0, 2.0]), &bf16(&[0.5, 0.5]), &bf16(&[1.0, -1.0])).unwrap();
        assert_eq!(out.dtype(), Dtype::Bfloat16);
        let got: Vec<f32> = out.as_dtype(Dtype::Float32).unwrap().as_slice().to_vec();
        assert_eq!(got, vec![2.5, 0.5]);
    });
}

#[test]
fn the_preview_is_the_exact_decode_of_the_patch_mean_of_the_first_row() {
    cpu_process(
        "lib_ops::the_preview_is_the_exact_decode_of_the_patch_mean_of_the_first_row",
        || {
            // [2, 3, 8, 8] with values spanning past [−1, 1], patch 4 → a 2×2 frame of row 0.
            let n = 2 * 3 * 8 * 8;
            let data: Vec<f32> = (0..n)
                .map(|i| ((i * 37 % 97) as f32 / 32.0) - 1.5)
                .collect();
            let x0 = Array::from_slice(&data, &[2, 3, 8, 8]);
            let frame = preview_image(&x0, 4).unwrap();
            assert_eq!((frame.width, frame.height), (2, 2));
            for py in 0..2 {
                for px in 0..2 {
                    for c in 0..3 {
                        let mut sum = 0f32;
                        for y in 0..4 {
                            for x in 0..4 {
                                sum += data[(c * 8 + py * 4 + y) * 8 + px * 4 + x];
                            }
                        }
                        let unit = ((sum / 16.0 + 1.0) / 2.0).clamp(0.0, 1.0);
                        let want = (unit * 255.0).round_ties_even() as u8;
                        assert_eq!(frame.pixels[(py * 2 + px) * 3 + c], want, "({px},{py},{c})");
                    }
                }
            }
            assert!(preview_image(&x0, 3).is_err(), "off the patch grid");
        },
    );
}

/// Every factor shape the miniature exercises, on the CPU: near-orthogonal output (the quintic
/// leaves singular values in roughly [0.7, 1.2]). MLX's CPU bf16 GEMM faults on some transposed
/// narrow shapes such as 17x2 — `mm_bf16` keeps the orthogonalization off that path; this test is
/// the regression. (Values against upstream Dion: `train_parity::newton_schulz_matches_dion`.)
#[test]
fn newton_schulz_orthogonalizes_every_factor_shape() {
    cpu_process(
        "lib_ops::newton_schulz_orthogonalizes_every_factor_shape",
        || {
            let key = mlx_rs::random::key(3).unwrap();
            for (r, c) in [
                (17, 2),
                (2, 8),
                (10, 8),
                (4, 8),
                (8, 8),
                (2, 17),
                (10, 17),
                (17, 10),
                (1, 4),
            ] {
                let g = mlx_rs::random::normal::<f32>(&[r, c][..], None, None, Some(&key)).unwrap();
                let o = newton_schulz(&g, 1e-8)
                    .unwrap()
                    .as_dtype(Dtype::Float32)
                    .unwrap();
                eval([&o]).unwrap();
                let wide = if r > c { o.t() } else { o.clone() };
                let gram = matmul(&wide, wide.t()).unwrap();
                let k = r.min(c);
                let diag: Vec<f32> = gram
                    .reshape(&[-1])
                    .unwrap()
                    .as_slice::<f32>()
                    .iter()
                    .step_by(k as usize + 1)
                    .copied()
                    .collect();
                assert!(
                    diag.iter().all(|d| (0.5..1.4).contains(d)),
                    "{r}x{c}: gram diagonal {diag:?}"
                );
            }
        },
    );
}
