//! SW-6140 — the shared RoPE / modulate helpers' route tests.
//!
//! Two kinds, kept apart on purpose:
//!
//! * **Pure plan tests** (ordinary CI): [`plan_fused_rope`]'s admission logic and the opt-in
//!   toggle. They touch `Dtype` values and shapes only and execute no MLX kernel.
//! * **MLX compute tests** (`#[ignore]`, registered in `scripts/ci/test_suites.json` for the
//!   dispatch-only exhaustive suite): the fused `fast::rope` route against the shared cos/sin route
//!   for both pairing conventions (with a swapped-pairing mutation control that must fail), and the
//!   shared cos/sin route + modulate helpers against verbatim copies of every private formulation
//!   SW-6140 removed from the family crates, **bit-for-bit** at f32 and bf16, including a
//!   multi-axis (3-axis id) table.

use super::*;
use crate::nn::{max_rel_diff, modulate_eager, rotate_pairwise, scale_shift, set_compile_glue};
use mlx_rs::ops::{array_eq, negative};

const STYLES_1D: [RopeStyle; 4] = [
    RopeStyle::AdjacentPair,
    RopeStyle::InterleavedFull,
    RopeStyle::RotateHalf,
    RopeStyle::HalvesPaired,
];

fn seq(shape: &[i32], scale: f32, offset: f32) -> Array {
    let n: i32 = shape.iter().product();
    let data: Vec<f32> = (0..n)
        .map(|i| ((i as f32) * scale + offset).sin())
        .collect();
    Array::from_slice(&data, shape)
}

/// Bit-identical: same dtype, same shape, every element equal.
fn bits(a: &Array, b: &Array) -> bool {
    a.dtype() == b.dtype() && a.shape() == b.shape() && array_eq(a, b, None).unwrap().item::<bool>()
}

/// Host 1-D tables for positions `offset..offset+len` over `rot` rotated channels, base `base`:
/// `θ[t][i] = (offset + t) · base^(−2i/rot)` in f64, rounded to f32, laid out for `style` —
/// half-width for the pair-rotation arms, full-width (`cat(f, f)` or `repeat_interleave(2)`) for
/// the rest. This is the schedule `Rope1d::standard(base, offset)` describes.
fn tables_1d(style: RopeStyle, len: i32, offset: i32, rot: i32, base: f64) -> (Array, Array) {
    let half = rot / 2;
    let mut cos = Vec::new();
    let mut sin = Vec::new();
    for t in 0..len {
        let angles: Vec<f64> = (0..half)
            .map(|i| (offset + t) as f64 * base.powf(-2.0 * i as f64 / rot as f64))
            .collect();
        let row: Vec<f64> = match style {
            RopeStyle::AdjacentPair | RopeStyle::HalvesPaired => angles,
            RopeStyle::RotateHalf => angles.iter().chain(angles.iter()).copied().collect(),
            RopeStyle::InterleavedFull => angles.iter().flat_map(|&a| [a, a]).collect(),
            RopeStyle::None => unreachable!(),
        };
        cos.extend(row.iter().map(|a| a.cos() as f32));
        sin.extend(row.iter().map(|a| a.sin() as f32));
    }
    let width = cos.len() as i32 / len;
    (
        Array::from_slice(&cos, &[len, width]),
        Array::from_slice(&sin, &[len, width]),
    )
}

// ── pure plan tests (no MLX kernel executes) ───────────────────────────────────────────────────

/// The pairing convention is derived from the style, never passed in — a wrong `traditional` flag
/// renders garbage without an error, so this mapping is pinned.
#[test]
fn fused_rope_plan_derives_the_pairing_from_the_style() {
    let f32_ = Dtype::Float32;
    let plan = |style, axes| {
        plan_fused_rope(
            &[1, 8, 4, 64],
            f32_,
            f32_,
            style,
            axes,
            None,
            RopeDtype::RestoreInput,
        )
        .unwrap()
    };
    for (style, traditional) in [
        (RopeStyle::AdjacentPair, true),
        (RopeStyle::InterleavedFull, true),
        (RopeStyle::RotateHalf, false),
        (RopeStyle::HalvesPaired, false),
    ] {
        let p = plan(style, RotationAxes::HeadMajor);
        assert_eq!(p.traditional, traditional, "{style:?}");
        assert_eq!(p.dims, 64);
        assert!(
            !p.transpose,
            "head-major streams rotate along axis -2 as they are"
        );
        assert!(plan(style, RotationAxes::TokenMajor).transpose);
    }
    let partial = plan_fused_rope(
        &[1, 8, 4, 128],
        f32_,
        f32_,
        RopeStyle::RotateHalf,
        RotationAxes::HeadMajor,
        Some(96),
        RopeDtype::Stream,
    )
    .unwrap();
    assert_eq!(partial.dims, 96, "rot_dims is fast::rope's `dims`");
}

/// Every layout `fast::rope` cannot reproduce is a typed refusal, decided before any launch.
#[test]
fn fused_rope_plan_refuses_what_fast_rope_cannot_express() {
    use FusedRopeRefusal as R;
    let (f32_, bf16) = (Dtype::Float32, Dtype::Bfloat16);
    let shape = [1, 8, 4, 64];
    let go = |shape: &[i32], stream, table, style, rot, dtype| {
        plan_fused_rope(
            shape,
            stream,
            table,
            style,
            RotationAxes::HeadMajor,
            rot,
            dtype,
        )
    };
    let adj = RopeStyle::AdjacentPair;
    let restore = RopeDtype::RestoreInput;
    assert_eq!(
        go(&shape, f32_, f32_, RopeStyle::None, None, restore),
        Err(R::NoRotation)
    );
    assert_eq!(
        go(&shape, f32_, f32_, adj, Some(0), restore),
        Err(R::NoRotation)
    );
    assert_eq!(
        go(&[8, 4, 64], f32_, f32_, adj, None, restore),
        Err(R::Rank(3))
    );
    assert_eq!(
        go(&shape, f32_, f32_, adj, Some(33), restore),
        Err(R::RotatedDims {
            rot: 33,
            head_dim: 64
        })
    );
    assert_eq!(
        go(&shape, f32_, f32_, adj, Some(66), restore),
        Err(R::RotatedDims {
            rot: 66,
            head_dim: 64
        })
    );
    assert_eq!(
        go(&shape, Dtype::Int32, f32_, adj, None, restore),
        Err(R::StreamDtype(Dtype::Int32))
    );
    // `Promoted` over f32 tables would hand SDPA an f32 stream; fast::rope returns bf16.
    assert_eq!(
        go(&shape, bf16, f32_, adj, None, RopeDtype::Promoted),
        Err(R::PromotionWouldBeLost {
            stream: bf16,
            table: f32_
        })
    );
    // ...while matching dtypes, an f32 stream, `RestoreInput` and `Stream` are all admitted.
    for (stream, table, dtype) in [
        (bf16, bf16, RopeDtype::Promoted),
        (f32_, bf16, RopeDtype::Promoted),
        (bf16, f32_, RopeDtype::RestoreInput),
        (bf16, f32_, RopeDtype::Stream),
    ] {
        assert!(
            go(&shape, stream, table, adj, None, dtype).is_ok(),
            "{stream:?}/{table:?}/{dtype:?}"
        );
    }
    let labels: std::collections::HashSet<&str> = [
        R::Disabled,
        R::NoRotation,
        R::Rank(3),
        R::RotatedDims {
            rot: 1,
            head_dim: 2,
        },
        R::StreamDtype(Dtype::Int32),
        R::PromotionWouldBeLost {
            stream: bf16,
            table: f32_,
        },
    ]
    .iter()
    .map(|r| r.label())
    .collect();
    assert_eq!(labels.len(), 6, "every refusal has its own stable reason");
}

/// The fused route is opt-in (the default keeps every adopter on its exact cos/sin numerics) and
/// the guard restores the prior value.
#[test]
fn the_fused_route_is_opt_in_and_the_guard_restores() {
    set_fused_rope(false);
    assert!(!fused_rope(), "the fused route is off by default");
    {
        let _on = FusedRopeGuard::set(true);
        assert!(fused_rope());
        {
            let _off = FusedRopeGuard::set(false);
            assert!(!fused_rope());
        }
        assert!(fused_rope(), "the inner guard restores the outer value");
    }
    assert!(!fused_rope(), "the outer guard restores the default");
}

// ── MLX compute: fused route vs cos/sin route ─────────────────────────────────────────────────

/// f32 bound for fused-vs-table, relative to the reference peak. `fast::rope` evaluates its angles
/// on the device (`exp2` inverse frequencies, `metal::fast::cos/sin`) where the reference reads
/// precise host trig rounded to f32, so the two differ by the fast-trig error at the test's
/// largest angle (|θ| ≤ 55 rad here) — not by a rounding of the rotation itself. This is a
/// **declared** bound to be tightened from the first measured Mac number (the test prints it); a
/// pairing, frequency or offset bug is O(1e-1) and is caught by the mutation controls below.
const FUSED_F32_TOL: f32 = 5e-4;
/// bf16 bound under `RestoreInput`: both routes compute in f32 and round once to bf16, so they
/// agree to one bf16 rounding of the peak plus the trig difference — 2 bf16 ULPs (2⁻⁷).
const FUSED_BF16_TOL: f32 = 1.0 / 128.0;
/// bf16 bound under `Stream`: the reference narrows the tables to bf16 and rounds each of its
/// three ops in bf16, the fused kernel rounds once — 4 bf16 ULPs (2⁻⁶).
const FUSED_BF16_STREAM_TOL: f32 = 1.0 / 64.0;

#[test]
#[ignore = "MLX compute: SW-6140 fused fast::rope vs shared cos/sin route parity; exhaustive suite, not ordinary CI"]
fn the_fused_route_matches_the_cos_sin_route_for_both_pairings() {
    set_compile_glue(false);
    let base = 10_000.0f32;
    let mut worst = [0.0f32; 3];
    for style in STYLES_1D {
        for axes in [RotationAxes::HeadMajor, RotationAxes::TokenMajor] {
            for (rot, hd) in [(64, 64), (32, 64)] {
                for offset in [0, 7] {
                    for (stream, dtype, tol, slot) in [
                        (Dtype::Float32, RopeDtype::RestoreInput, FUSED_F32_TOL, 0),
                        (Dtype::Bfloat16, RopeDtype::RestoreInput, FUSED_BF16_TOL, 1),
                        (Dtype::Bfloat16, RopeDtype::Stream, FUSED_BF16_STREAM_TOL, 2),
                    ] {
                        let (b, h, l) = (2, 3, 48);
                        let shape = match axes {
                            RotationAxes::HeadMajor => [b, h, l, hd],
                            RotationAxes::TokenMajor => [b, l, h, hd],
                        };
                        let x = seq(&shape, 0.37, 0.1).as_dtype(stream).unwrap();
                        let (cos, sin) = tables_1d(style, l, offset, rot, base as f64);
                        let tables = RopeTables::new(&cos, &sin);
                        let layout = Rope1d::standard(base, offset);
                        let run = |on: bool| {
                            let _g = FusedRopeGuard::set(on);
                            let out =
                                apply_rope_1d(&x, tables, layout, style, axes, Some(rot), dtype)
                                    .unwrap();
                            out.eval().unwrap();
                            out
                        };
                        let reference = run(false);
                        let fused = run(true);
                        let case = format!(
                            "{style:?}/{axes:?}/rot{rot}of{hd}/off{offset}/{stream:?}/{dtype:?}"
                        );
                        assert_eq!(fused.dtype(), reference.dtype(), "{case}: dtype");
                        assert_eq!(fused.shape(), reference.shape(), "{case}: shape");
                        let err = max_rel_diff(&fused, &reference);
                        worst[slot] = worst[slot].max(err);
                        assert!(err <= tol, "{case}: fused vs cos/sin rel {err:e} > {tol:e}");
                        // The tail beyond `rot` passes through both routes untouched.
                        if rot < hd {
                            let axis = 3;
                            let ft = split_sections(&fused, &[rot], axis).unwrap();
                            let rt = split_sections(&reference, &[rot], axis).unwrap();
                            assert!(bits(&ft[1], &rt[1]), "{case}: tail must pass through");
                        }
                    }
                }
            }
        }
    }
    println!(
        "SW-6140 fused-vs-cos/sin max rel: f32 {:e} (tol {FUSED_F32_TOL:e}), bf16 RestoreInput {:e} \
         (tol {FUSED_BF16_TOL:e}), bf16 Stream {:e} (tol {FUSED_BF16_STREAM_TOL:e})",
        worst[0], worst[1], worst[2]
    );
}

/// Mutation controls: the parity above is not a false green. Swapping the pairing convention, or
/// shifting the positions by one, must move the result by orders of magnitude more than the
/// declared tolerance — the failure mode this story was told to investigate first ("a wrong
/// pairing renders garbage without erroring").
#[test]
#[ignore = "MLX compute: SW-6140 fused fast::rope mutation controls (swapped pairing / offset); exhaustive suite, not ordinary CI"]
fn a_swapped_pairing_or_offset_fails_the_fused_parity() {
    set_compile_glue(false);
    let (b, h, l, hd) = (1, 2, 32, 64);
    let x = seq(&[b, h, l, hd], 0.29, 0.3);
    for style in STYLES_1D {
        let (cos, sin) = tables_1d(style, l, 0, hd, 10_000.0);
        let reference = apply_rope_eager(
            &x,
            RopeTables::new(&cos, &sin),
            style,
            RotationAxes::HeadMajor,
            None,
            RopeDtype::RestoreInput,
        )
        .unwrap();
        let plan = plan_fused_rope(
            x.shape(),
            x.dtype(),
            cos.dtype(),
            style,
            RotationAxes::HeadMajor,
            None,
            RopeDtype::RestoreInput,
        )
        .unwrap();
        let right =
            mlx_rs::fast::rope(&x, hd, plan.traditional, Some(10_000.0f32), 1.0, 0, None).unwrap();
        let swapped =
            mlx_rs::fast::rope(&x, hd, !plan.traditional, Some(10_000.0f32), 1.0, 0, None).unwrap();
        let shifted =
            mlx_rs::fast::rope(&x, hd, plan.traditional, Some(10_000.0f32), 1.0, 1, None).unwrap();
        let ok = max_rel_diff(&right, &reference);
        assert!(
            ok <= FUSED_F32_TOL,
            "{style:?}: the right pairing must pass ({ok:e})"
        );
        for (what, wrong) in [("swapped pairing", &swapped), ("offset by one", &shifted)] {
            let err = max_rel_diff(wrong, &reference);
            assert!(
                err > 100.0 * FUSED_F32_TOL,
                "{style:?}: a {what} must fail the parity, got rel {err:e}"
            );
        }
    }
}

// ── MLX compute: the shared cos/sin route + modulate vs the removed private copies ─────────────

/// Verbatim (modulo names) the private rotations SW-6140 removed, as bit-exactness oracles.
mod removed {
    use mlx_rs::ops::{
        add, concatenate_axis, multiply, negative, split, split_sections, stack_axis, subtract,
    };
    use mlx_rs::Array;

    /// qwen-image `apply_rope_qwen` / z-image DiT / wan: reshape-split + `rope_rotate` eager,
    /// token-major `[B, S, H, D]` with `[S, D/2]` tables.
    pub fn qwen(x: &Array, cos: &Array, sin: &Array) -> Array {
        let sh = x.shape();
        let (b, seq, heads, hd) = (sh[0], sh[1], sh[2], sh[3]);
        let half = hd / 2;
        let parts = split(x.reshape(&[b, seq, heads, half, 2]).unwrap(), 2, 4).unwrap();
        let xr = parts[0].reshape(&[b, seq, heads, half]).unwrap();
        let xi = parts[1].reshape(&[b, seq, heads, half]).unwrap();
        let cos = cos.reshape(&[1, seq, 1, half]).unwrap();
        let sin = sin.reshape(&[1, seq, 1, half]).unwrap();
        let out_r = subtract(multiply(&xr, &cos).unwrap(), multiply(&xi, &sin).unwrap()).unwrap();
        let out_i = add(multiply(&xr, &sin).unwrap(), multiply(&xi, &cos).unwrap()).unwrap();
        concatenate_axis(
            &[
                &out_r.expand_dims(4).unwrap(),
                &out_i.expand_dims(4).unwrap(),
            ],
            4,
        )
        .unwrap()
        .reshape(&[b, seq, heads, hd])
        .unwrap()
    }

    /// sam2 `memory::apply_rope`: head-major, `take_axis` lanes, `stack_axis` back.
    pub fn sam2(x: &Array, cos: &Array, sin: &Array) -> Array {
        let sh = x.shape();
        let (b, h, n, d) = (sh[0], sh[1], sh[2], sh[3]);
        let pair = x.reshape(&[b, h, n, d / 2, 2]).unwrap();
        let even = pair.take_axis(Array::from_int(0), 4).unwrap();
        let odd = pair.take_axis(Array::from_int(1), 4).unwrap();
        let cos = cos.reshape(&[1, 1, n, d / 2]).unwrap();
        let sin = sin.reshape(&[1, 1, n, d / 2]).unwrap();
        let oe = subtract(
            multiply(&even, &cos).unwrap(),
            multiply(&odd, &sin).unwrap(),
        )
        .unwrap();
        let oo = add(
            multiply(&even, &sin).unwrap(),
            multiply(&odd, &cos).unwrap(),
        )
        .unwrap();
        stack_axis(&[&oe, &oo], 4)
            .unwrap()
            .reshape(&[b, h, n, d])
            .unwrap()
    }

    /// minimax-h3 `MmRope::apply`: tables narrowed to the stream, rotate-half over the leading
    /// `rot` channels, tail untouched.
    pub fn minimax(x: &Array, cos: &Array, sin: &Array) -> Array {
        let sh = x.shape();
        let (seq, d) = (sh[1], sh[3]);
        let rot = cos.shape()[1];
        let cos_t = cos
            .as_dtype(x.dtype())
            .unwrap()
            .reshape(&[1, seq, 1, rot])
            .unwrap();
        let sin_t = sin
            .as_dtype(x.dtype())
            .unwrap()
            .reshape(&[1, seq, 1, rot])
            .unwrap();
        let parts = split_sections(x, &[rot], 3).unwrap();
        let (head, tail) = (&parts[0], &parts[1]);
        let halves = split(head, 2, 3).unwrap();
        let rotated = concatenate_axis(&[&negative(&halves[1]).unwrap(), &halves[0]], -1).unwrap();
        let out = add(
            multiply(head, &cos_t).unwrap(),
            multiply(&rotated, &sin_t).unwrap(),
        )
        .unwrap();
        if rot == d {
            out
        } else {
            concatenate_axis(&[&out, tail], -1).unwrap()
        }
    }

    /// sam3 `vision::apply_rope` (`negative`) and seedvr2 (`× −1`, f32, cast back): `x·cos +
    /// rotate_pairwise(x)·sin` with full-width repeat-interleaved tables.
    pub fn pairwise(x: &Array, cos: &Array, sin: &Array, negate_by_multiply: bool) -> Array {
        let sh = x.shape().to_vec();
        let mut paired = sh[..sh.len() - 1].to_vec();
        paired.push(sh[sh.len() - 1] / 2);
        paired.push(2);
        let xr = x.reshape(&paired).unwrap();
        let last = paired.len() as i32 - 1;
        let x1 = xr.take_axis(Array::from_int(0), last).unwrap();
        let x2 = xr.take_axis(Array::from_int(1), last).unwrap();
        let neg = if negate_by_multiply {
            multiply(&x2, Array::from_f32(-1.0)).unwrap()
        } else {
            negative(&x2).unwrap()
        };
        let rot = stack_axis(&[&neg, &x1], last)
            .unwrap()
            .reshape(&sh)
            .unwrap();
        add(multiply(x, cos).unwrap(), multiply(&rot, sin).unwrap()).unwrap()
    }

    /// lens gpt-oss `apply_half_rope`: halves rotated against `[1, 1, L, d/2]` tables.
    pub fn halves(x: &Array, cos: &Array, sin: &Array) -> Array {
        let parts = split(x, 2, -1).unwrap();
        let (first, second) = (&parts[0], &parts[1]);
        let a = subtract(
            multiply(first, cos).unwrap(),
            multiply(second, sin).unwrap(),
        )
        .unwrap();
        let b = add(
            multiply(second, cos).unwrap(),
            multiply(first, sin).unwrap(),
        )
        .unwrap();
        concatenate_axis(&[a, b], -1).unwrap()
    }

    /// The private eager modulates: `x·(1 + scale) + shift`, `1` in scale's dtype or strong f32.
    pub fn modulate(x: &Array, scale: &Array, shift: &Array, matching: bool) -> Array {
        let one = if matching {
            Array::from_f32(1.0).as_dtype(scale.dtype()).unwrap()
        } else {
            Array::from_f32(1.0)
        };
        add(multiply(x, add(scale, &one).unwrap()).unwrap(), shift).unwrap()
    }
}

#[test]
#[ignore = "MLX compute: SW-6140 shared cos/sin route vs the removed private RoPE copies, bit-for-bit; exhaustive suite, not ordinary CI"]
fn the_shared_cos_sin_route_reproduces_every_removed_private_rope_bitwise() {
    set_compile_glue(false);
    for dtype in [Dtype::Float32, Dtype::Bfloat16] {
        let (b, s, h, d) = (2, 12, 3, 16);
        let tm = seq(&[b, s, h, d], 0.31, 0.2).as_dtype(dtype).unwrap(); // token-major
        let hm = seq(&[b, h, s, d], 0.23, 0.5).as_dtype(dtype).unwrap(); // head-major
        let (c_half, s_half) = tables_1d(RopeStyle::AdjacentPair, s, 3, d, 10_000.0);
        let (c_full, s_full) = tables_1d(RopeStyle::RotateHalf, s, 3, d, 10_000.0);
        let (c_int, s_int) = tables_1d(RopeStyle::InterleavedFull, s, 3, d, 10_000.0);
        let half_tables = RopeTables::new(&c_half, &s_half);
        let shared = |x: &Array, t, style, axes, rot, policy| {
            apply_rope_eager(x, t, style, axes, rot, policy).unwrap()
        };

        // AdjacentPair token-major, no casts (qwen-image / z-image DiT / wan / kolors / krea).
        let want = removed::qwen(&tm, &c_half, &s_half);
        let got = shared(
            &tm,
            half_tables,
            RopeStyle::AdjacentPair,
            RotationAxes::TokenMajor,
            None,
            RopeDtype::Promoted,
        );
        assert!(bits(&got, &want), "{dtype:?}: AdjacentPair token-major");
        // ...and the glue-routed `apply_rope` with the glue off is the same bits.
        let glue = apply_rope(
            &tm,
            half_tables,
            RopeStyle::AdjacentPair,
            RotationAxes::TokenMajor,
            None,
            RopeDtype::Promoted,
        )
        .unwrap();
        assert!(
            bits(&glue, &want),
            "{dtype:?}: apply_rope (glue off) == apply_rope_eager"
        );

        // AdjacentPair head-major (sam2, flux, flux2, flux3, iris).
        let want = removed::sam2(&hm, &c_half, &s_half);
        let got = shared(
            &hm,
            half_tables,
            RopeStyle::AdjacentPair,
            RotationAxes::HeadMajor,
            None,
            RopeDtype::Promoted,
        );
        assert!(bits(&got, &want), "{dtype:?}: AdjacentPair head-major");

        // RotateHalf partial under `Stream` (minimax-h3 DiT + VAE, pid gemma2 at full width).
        let (c_part, s_part) = tables_1d(RopeStyle::RotateHalf, s, 0, 12, 10_000.0);
        let want = removed::minimax(&tm, &c_part, &s_part);
        let got = shared(
            &tm,
            RopeTables::new(&c_part, &s_part),
            RopeStyle::RotateHalf,
            RotationAxes::TokenMajor,
            Some(12),
            RopeDtype::Stream,
        );
        assert!(bits(&got, &want), "{dtype:?}: RotateHalf partial Stream");
        assert_eq!(got.dtype(), dtype, "Stream never promotes");
        // Full-width rotate-half is the shared `nn::apply_text_rope` op sequence too.
        let text = crate::nn::apply_text_rope(
            &tm,
            &c_full.expand_dims(0).unwrap(),
            &s_full.expand_dims(0).unwrap(),
        )
        .unwrap();
        let got = shared(
            &tm,
            RopeTables::new(&c_full, &s_full),
            RopeStyle::RotateHalf,
            RotationAxes::TokenMajor,
            None,
            RopeDtype::Promoted,
        );
        assert!(
            bits(&got, &text),
            "{dtype:?}: RotateHalf == nn::apply_text_rope"
        );

        // InterleavedFull head-major, `negative` lane (sam3 vision/tracker, pulid).
        let want = removed::pairwise(
            &hm,
            &c_int.reshape(&[1, 1, s, d]).unwrap(),
            &s_int.reshape(&[1, 1, s, d]).unwrap(),
            false,
        );
        let got = shared(
            &hm,
            RopeTables::new(&c_int, &s_int),
            RopeStyle::InterleavedFull,
            RotationAxes::HeadMajor,
            None,
            RopeDtype::Promoted,
        );
        assert!(bits(&got, &want), "{dtype:?}: InterleavedFull head-major");
        // ...and seedvr2's `× −1` lane in f32, cast back, over a rank-3 `[N, heads, d]` view.
        let n3 = seq(&[s, h, d], 0.41, 0.7).as_dtype(dtype).unwrap();
        let f = n3.as_dtype(Dtype::Float32).unwrap();
        let want = removed::pairwise(
            &f,
            &c_int.reshape(&[s, 1, d]).unwrap(),
            &s_int.reshape(&[s, 1, d]).unwrap(),
            true,
        )
        .as_dtype(dtype)
        .unwrap();
        let got = shared(
            &n3.reshape(&[1, s, h, d]).unwrap(),
            RopeTables::new(&c_int, &s_int),
            RopeStyle::InterleavedFull,
            RotationAxes::TokenMajor,
            None,
            RopeDtype::RestoreInput,
        )
        .reshape(&[s, h, d])
        .unwrap();
        assert!(
            bits(&got, &want),
            "{dtype:?}: InterleavedFull RestoreInput (seedvr2)"
        );

        // HalvesPaired head-major over pre-broadcast half tables (lens gpt-oss).
        let c4 = c_half
            .reshape(&[1, 1, s, d / 2])
            .unwrap()
            .as_dtype(dtype)
            .unwrap();
        let s4 = s_half
            .reshape(&[1, 1, s, d / 2])
            .unwrap()
            .as_dtype(dtype)
            .unwrap();
        let want = removed::halves(&hm, &c4, &s4);
        let got = shared(
            &hm,
            RopeTables::new(&c4, &s4),
            RopeStyle::HalvesPaired,
            RotationAxes::HeadMajor,
            None,
            RopeDtype::Promoted,
        );
        assert!(bits(&got, &want), "{dtype:?}: HalvesPaired pre-broadcast");
    }
}

/// Multi-axis layouts — FLUX-style 3-axis position ids through the shared table builder — take
/// the same shared cos/sin route and reproduce the removed private rotation bit-for-bit. These
/// are the layouts `fast::rope` cannot express, so they never reach the fused route.
#[test]
#[ignore = "MLX compute: SW-6140 shared cos/sin route on a multi-axis (3-axis id) table; exhaustive suite, not ordinary CI"]
fn a_multi_axis_table_takes_the_shared_cos_sin_route_bitwise() {
    set_compile_glue(false);
    let (txt, gh, gw) = (3, 2, 3);
    let n = txt + gh * gw;
    let mut ids = vec![0f32; (n * 3) as usize];
    for r in 0..gh {
        for c in 0..gw {
            let row = (txt + r * gw + c) as usize;
            ids[row * 3 + 1] = r as f32;
            ids[row * 3 + 2] = c as f32;
        }
    }
    let ids = Array::from_slice(&ids, &[n, 3]);
    let (cos, sin) = crate::nn::rope_sincos_from_ids(&ids, &[4, 6, 6], 10_000.0).unwrap();
    for dtype in [Dtype::Float32, Dtype::Bfloat16] {
        let x = seq(&[1, n, 2, 16], 0.19, 0.9).as_dtype(dtype).unwrap();
        let want = removed::qwen(&x, &cos, &sin);
        let got = apply_rope_eager(
            &x,
            RopeTables::new(&cos, &sin),
            RopeStyle::AdjacentPair,
            RotationAxes::TokenMajor,
            None,
            RopeDtype::Promoted,
        )
        .unwrap();
        assert!(bits(&got, &want), "{dtype:?}: 3-axis table");
        // The fused route refuses nothing here by shape — the *caller* keeps a multi-axis layout
        // off `apply_rope_1d`; prove the tables really are not a 1-D schedule.
        let (c1d, _) = tables_1d(RopeStyle::AdjacentPair, n, 0, 16, 10_000.0);
        assert!(
            max_rel_diff(&cos, &c1d) > 1e-2,
            "a 3-axis table is not a 1-D layout"
        );
    }
}

#[test]
#[ignore = "MLX compute: SW-6140 shared modulate helpers vs the removed private copies, bit-for-bit; exhaustive suite, not ordinary CI"]
fn the_shared_modulate_helpers_reproduce_the_removed_private_copies_bitwise() {
    set_compile_glue(false);
    for dtype in [Dtype::Float32, Dtype::Bfloat16] {
        let x = seq(&[2, 7, 32], 0.17, 0.3).as_dtype(dtype).unwrap();
        let scale = seq(&[2, 1, 32], 0.71, 0.2).as_dtype(dtype).unwrap();
        let shift = seq(&[2, 1, 32], 0.53, 0.6).as_dtype(dtype).unwrap();
        for matching in [true, false] {
            let want = removed::modulate(&x, &scale, &shift, matching);
            let got = modulate_eager(&x, &scale, &shift, matching).unwrap();
            assert!(
                bits(&got, &want),
                "{dtype:?}/matching={matching}: modulate_eager"
            );
            // The glue-gated `modulate` with the glue off runs the same body.
            let glue = crate::nn::modulate(&x, &scale, &shift, matching).unwrap();
            assert!(
                bits(&glue, &want),
                "{dtype:?}/matching={matching}: modulate (glue off)"
            );
            let body = if matching {
                crate::nn::modulate_matching_one_impl((&x, &scale, &shift)).unwrap()
            } else {
                crate::nn::modulate_f32_one_impl((&x, &scale, &shift)).unwrap()
            };
            assert!(
                bits(&body, &want),
                "{dtype:?}/matching={matching}: provider glue body"
            );
        }
        // The dtype policy is real: on a bf16 stream the matching `1` stays bf16 and the strong
        // f32 `1` promotes — the two must not collapse.
        if dtype == Dtype::Bfloat16 {
            assert_eq!(
                modulate_eager(&x, &scale, &shift, true).unwrap().dtype(),
                Dtype::Bfloat16
            );
            assert_eq!(
                modulate_eager(&x, &scale, &shift, false).unwrap().dtype(),
                Dtype::Float32
            );
        }
        let want = add(multiply(&x, &scale).unwrap(), &shift).unwrap();
        assert!(
            bits(&scale_shift(&x, &scale, &shift).unwrap(), &want),
            "{dtype:?}: scale_shift"
        );
    }
    // `rotate_pairwise` is an exact quarter turn on adjacent lanes: applied twice it negates.
    let x = Array::from_slice(&[1.0f32, 2.0, 3.0, 4.0], &[1, 4]);
    let twice = rotate_pairwise(&rotate_pairwise(&x).unwrap()).unwrap();
    assert!(bits(&twice, &negative(&x).unwrap()));
    assert_eq!(
        rotate_pairwise(&x).unwrap().as_slice::<f32>(),
        &[-2.0, 1.0, -4.0, 3.0]
    );
}
