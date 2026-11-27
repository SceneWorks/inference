//! SW-6129 GPU check of `candle_gen::rope::apply_interleaved_rope` — the shared interleaved RoPE of the
//! Lens, Qwen-Image, Wan (and SCAIL-2), FLUX.2 and Chroma DiTs and of Qwen-Image-2.1's fused-ops
//! mode — now the fused strided `media_rope` kernel (SW-6128), against the candle `rope_i` it used to
//! call, on CUDA device 0.
//!
//! Declared tolerance (epic E2): candle's CUDA `rope_i` is compiled by nvcc with fma contraction
//! (`a·c − b·s` becomes `fma(a, c, −b·s)`), while the fused kernel reproduces the shared reference
//! chain's separately rounded products (bit-identical to that chain and to candle's CPU `rope_i`, see
//! `fused_rope_gpu` and `rope::tests::cpu_rope_i_equals_the_shared_interleaved_chain`). The two GPU
//! results therefore differ by at most one rounding of one product: in f32 `|Δ| ≤ 2⁻²³·(|a·c| +
//! |b·s|)` per element (asserted with a 2x margin), max rel err to the operand scale far below the
//! E2 1e-5; in bf16 / f16 that f32 bound plus at most one output ulp (the single rounding to the
//! output dtype can straddle a boundary; near a cancellation the f32 term dominates).
//! The test prints the mismatch count and the worst errors it observed.
//!
//! `#[ignore]`d CUDA compute (registered in `scripts/ci/test_suites.json` for the dispatch-only
//! exhaustive suite):
//! `cargo test -p candle-gen --features cuda --test integration -- interleaved_rope_gpu:: --ignored`.

#![cfg(feature = "cuda")]

use candle_gen::candle_core::{DType, Device, Result, Tensor};
use candle_gen::candle_nn::rotary_emb::rope_i;
use candle_gen::fused_path::media_fused_policy_guard;
use candle_gen::rope::ROPE_PATH;

fn values(n: usize, seed: u64, scale: f32) -> Vec<f32> {
    let mut s = seed | 1;
    (0..n)
        .map(|_| {
            s ^= s << 13;
            s ^= s >> 7;
            s ^= s << 17;
            ((s >> 11) as f64 / (1u64 << 53) as f64 * 2.0 - 1.0) as f32 * scale
        })
        .collect()
}

fn host(t: &Tensor) -> Result<Vec<f32>> {
    t.to_dtype(DType::F32)?.flatten_all()?.to_vec1::<f32>()
}

/// One output ulp of `v` in `dtype` (subnormals: the fixed spacing below the smallest normal).
fn ulp(v: f32, dtype: DType) -> f32 {
    let (mantissa, min_exp) = match dtype {
        DType::BF16 => (7, -126),
        DType::F16 => (10, -14),
        _ => (23, -126),
    };
    let e = v
        .abs()
        .max(f32::MIN_POSITIVE)
        .log2()
        .floor()
        .max(min_exp as f32);
    2f32.powf(e - mantissa as f32)
}

#[test]
#[ignore = "CUDA compute: run by the validation workflow, not CI"]
fn apply_interleaved_rope_is_within_one_product_rounding_of_rope_i() -> Result<()> {
    let dev = Device::new_cuda(0)?;
    let _media = media_fused_policy_guard(Some(true));
    let _kernel = ROPE_PATH.guard(Some(true));
    // Qwen-Image 1024^2 (24 heads x 128, 4096 image + 256 text tokens), a CFG batch of 2 with
    // per-batch tables, and an odd small shape.
    let shapes: [(usize, usize, usize, usize); 3] =
        [(1, 24, 4352, 128), (2, 24, 1024, 128), (1, 3, 7, 10)];
    for (b, h, s, d) in shapes {
        let per_batch = b > 1;
        let tb = if per_batch { b } else { 1 };
        let angles = Tensor::from_vec(values(tb * s * d / 2, 7, 40.0), (tb, s, d / 2), &dev)?;
        let (cos, sin) = (angles.cos()?, angles.sin()?);
        let (cos, sin) = if per_batch {
            (cos, sin)
        } else {
            (cos.squeeze(0)?, sin.squeeze(0)?)
        };
        let x32 = Tensor::from_vec(values(b * h * s * d, 11, 3.0), (b, h, s, d), &dev)?;
        // The tables on the host, for the per-element f32 bound `|a·c| + |b·s|`.
        let (cf, sf) = (host(&cos.contiguous()?)?, host(&sin.contiguous()?)?);
        for dtype in [DType::F32, DType::BF16, DType::F16] {
            let x = x32.to_dtype(dtype)?;
            let want =
                rope_i(&x.to_dtype(DType::F32)?.contiguous()?, &cos, &sin)?.to_dtype(dtype)?;
            let start = ROPE_PATH.thread_counts();
            let got = candle_gen::rope::apply_interleaved_rope(&x, &cos, &sin)?;
            let counts = ROPE_PATH.thread_counts().since(&start);
            assert_eq!((counts.fused, counts.reference_total()), (1, 0), "{counts}");
            assert_eq!((got.dims(), got.dtype()), (want.dims(), dtype));
            let (g, w) = (host(&got)?, host(&want)?);
            let xin = host(&x)?;
            let (mut mismatches, mut worst_ratio, mut max_abs) = (0usize, 0f32, 0f32);
            for (i, (a, r)) in g.iter().zip(&w).enumerate() {
                if a.to_bits() == r.to_bits() {
                    continue;
                }
                mismatches += 1;
                let delta = (a - r).abs();
                max_abs = max_abs.max(delta);
                // Pair (even, odd) and its table entry.
                let lane = i % d;
                let row = i / d; // (b, h, s) flattened
                let (bi, si) = (row / (h * s), row % s);
                let t = (if per_batch { bi * s } else { 0 } + si) * (d / 2) + lane / 2;
                let (e, o) = (xin[i - lane % 2], xin[i - lane % 2 + 1]);
                // even lane `e·c − o·s`, odd lane `e·s + o·c`
                let magnitude = if lane % 2 == 0 {
                    (e * cf[t]).abs() + (o * sf[t]).abs()
                } else {
                    (e * sf[t]).abs() + (o * cf[t]).abs()
                };
                // The f32 results differ by at most one product rounding; a half-precision output
                // adds at most one output ulp where the two straddle a rounding boundary.
                let f32_bound = 2.0 * 2f32.powi(-23) * magnitude;
                let bound = if dtype == DType::F32 {
                    f32_bound
                } else {
                    f32_bound + ulp(a.abs().max(r.abs()), dtype)
                };
                worst_ratio = worst_ratio.max(delta / bound.max(f32::MIN_POSITIVE));
            }
            println!(
                "apply_interleaved_rope [{b},{h},{s},{d}] {dtype:?} (tables {:?}) vs rope_i: {mismatches} \
                 of {} elements differ, max|d| {max_abs:e}, worst |d|/bound {worst_ratio:.3}",
                cos.dims(),
                g.len()
            );
            assert!(
                worst_ratio <= 1.0,
                "{dtype:?} [{b},{h},{s},{d}]: a difference exceeds the declared bound ({worst_ratio})"
            );
        }
    }
    Ok(())
}
