//! SW-6142 route tests.
//!
//! * **Pure tests** (ordinary CI): [`plan_norm_silu_pad`]'s admission and refusals, the route
//!   switch/guard, the counters and the proofs' host-side PSNR gate. They touch shapes, dtypes,
//!   thread-locals and host `f32` slices only.
//! * **MLX compute tests** (`#[ignore]`, registered in `scripts/ci/test_suites.json` for the
//!   dispatch-only exhaustive suite): the Metal and compiled routes against the reference op chain
//!   over every Wan/LTX geometry of [`parity`], gated on E2's declared tolerance
//!   ([`TOLERANCE_F32`]: every route returns f32, so it is the one per-dtype bound that applies)
//!   through the same NaN-rejecting [`parity::Comparison::within_tolerance`] the real-weight proofs
//!   use, with bit identity, mismatch counts and max ULP printed as diagnostics only; the
//!   wrong-temporal-side mutation control; and the dispatch through [`try_fused`].

use super::parity::{self, Report};
use super::*;

const WAN: TemporalPad = TemporalPad::Zeros { front: 2, back: 0 };

fn plan(
    shape: &[i32],
    temporal: TemporalPad,
    cache: Option<(&[i32], Dtype)>,
    tail: i32,
) -> std::result::Result<NormSiluPadPlan, NormSiluPadRefusal> {
    plan_norm_silu_pad(
        shape,
        Dtype::Float32,
        Some((96, Dtype::Float32)),
        temporal,
        cache,
        (1, 1),
        tail,
        true,
    )
}

#[test]
fn the_plan_pads_time_and_space_around_the_cached_prefix() {
    let p = plan(&[1, 96, 4, 9, 11], WAN, None, 0).unwrap();
    assert_eq!(p.out, [1, 6, 11, 13, 96]);
    assert!(p.affine && !p.replicate && p.cache_frames == 0);

    let cache = [1, 96, 2, 9, 11];
    let p = plan(
        &[1, 96, 4, 9, 11],
        TemporalPad::Zeros { front: 0, back: 0 },
        Some((&cache, Dtype::Float32)),
        2,
    )
    .unwrap();
    assert_eq!((p.cache_frames, p.tail, p.out[1]), (2, 2, 6));

    let p = plan_norm_silu_pad(
        &[2, 128, 3, 4, 5],
        Dtype::Bfloat16,
        None,
        TemporalPad::Replicate { front: 1, back: 1 },
        None,
        (1, 1),
        0,
        true,
    )
    .unwrap();
    assert_eq!(p.out, [2, 5, 6, 7, 128]);
    assert!(p.replicate && !p.affine);
}

#[test]
fn the_plan_refuses_what_the_kernel_does_not_serve() {
    use NormSiluPadRefusal as R;
    let x = [1, 96, 4, 9, 11];
    let refused = |r: std::result::Result<NormSiluPadPlan, NormSiluPadRefusal>| r.unwrap_err();
    assert_eq!(
        refused(plan_norm_silu_pad(
            &x,
            Dtype::Float32,
            Some((96, Dtype::Float32)),
            WAN,
            None,
            (1, 1),
            0,
            false
        )),
        R::CpuStream
    );
    assert_eq!(refused(plan(&[1, 96, 9, 11], WAN, None, 0)), R::Rank(4));
    assert_eq!(
        refused(plan_norm_silu_pad(
            &x,
            Dtype::Int32,
            None,
            WAN,
            None,
            (1, 1),
            0,
            true
        )),
        R::Dtype(Dtype::Int32)
    );
    assert_eq!(refused(plan(&[1, 96, 0, 9, 11], WAN, None, 0)), R::Empty);
    assert_eq!(
        refused(plan_norm_silu_pad(
            &x,
            Dtype::Float32,
            Some((95, Dtype::Float32)),
            WAN,
            None,
            (1, 1),
            0,
            true
        )),
        R::GammaShape
    );
    let bad_cache = [1, 96, 2, 9, 10];
    assert_eq!(
        refused(plan(&x, WAN, Some((&bad_cache, Dtype::Float32)), 0)),
        R::CacheShape
    );
    let cache = [1, 96, 2, 9, 11];
    assert_eq!(
        refused(plan(&x, WAN, Some((&cache, Dtype::Bfloat16)), 0)),
        R::CacheShape
    );
    assert_eq!(
        refused(plan(
            &x,
            TemporalPad::Replicate { front: 1, back: 1 },
            Some((&cache, Dtype::Float32)),
            0
        )),
        R::CacheWithReplicate
    );
    assert_eq!(refused(plan(&x, WAN, None, 5)), R::Tail);
    assert_eq!(
        refused(plan(
            &x,
            TemporalPad::Replicate { front: 1, back: 1 },
            None,
            1
        )),
        R::Tail
    );
    assert_eq!(
        refused(plan(&x, TemporalPad::Zeros { front: -1, back: 0 }, None, 0)),
        R::Pad
    );
    // E6: 96 · 2 · 4096 · 4096 · 2 frames is past i32::MAX elements.
    assert_eq!(
        refused(plan(&[1, 96, 2, 4096, 4096], WAN, None, 0)),
        R::IndexOverflow
    );
    for refusal in [
        R::CpuStream,
        R::Rank(4),
        R::Dtype(Dtype::Int32),
        R::Empty,
        R::GammaShape,
        R::CacheShape,
        R::CacheWithReplicate,
        R::Tail,
        R::Pad,
        R::IndexOverflow,
        R::KernelCompile,
    ] {
        assert!(!refusal.label().is_empty());
    }
}

#[test]
fn the_route_switch_parses_and_the_guard_restores() {
    for route in [
        NormSiluPadRoute::Reference,
        NormSiluPadRoute::Compiled,
        NormSiluPadRoute::Metal,
    ] {
        assert_eq!(NormSiluPadRoute::parse(route.label()), Some(route));
    }
    assert_eq!(
        NormSiluPadRoute::parse(" OFF "),
        Some(NormSiluPadRoute::Reference)
    );
    assert_eq!(NormSiluPadRoute::parse("fast"), None);

    set_route(None);
    let ambient = route();
    {
        let _outer = RouteGuard::set(NormSiluPadRoute::Reference);
        assert_eq!(route(), NormSiluPadRoute::Reference);
        {
            let _inner = RouteGuard::set(NormSiluPadRoute::Compiled);
            assert_eq!(route(), NormSiluPadRoute::Compiled);
        }
        assert_eq!(route(), NormSiluPadRoute::Reference);
    }
    assert_eq!(route(), ambient);
}

#[test]
fn route_counts_subtract_a_snapshot() {
    let start = RouteCounts {
        metal: 2,
        compiled: 1,
        reference: 3,
        refused: 0,
        last_refusal: None,
    };
    let now = RouteCounts {
        metal: 5,
        compiled: 1,
        reference: 4,
        refused: 1,
        last_refusal: Some("rank"),
    };
    let delta = now.since(&start);
    assert_eq!(
        (delta.metal, delta.compiled, delta.reference, delta.refused),
        (3, 0, 1, 1)
    );
    assert_eq!(delta.last_refusal, Some("rank"));
}

#[test]
fn the_kernel_source_splits_into_header_and_body() {
    let (header, body) = kernel_source();
    assert!(header.contains("#pragma clang fp contract(off)"));
    assert!(header.contains("inline float sceneworks_vae_silu"));
    // MLX adds a strided-input parameter for every `<name>_strides` the body mentions.
    for name in ["inp", "denom", "gamma", "cache"] {
        assert!(body.contains(&format!("{name}_strides")), "{name}");
    }
    assert!(body.contains("thread_position_in_grid"));
    assert!(!body.contains("_shape") && !body.contains("_ndim"));
    // The JIT's floating-point functions default to `Fast`; the metallib reference is built with
    // `-fno-fast-math`. Every exp / divide names the precise function explicitly.
    assert!(header.contains("metal::precise::exp(") && header.contains("metal::precise::divide("));
    assert!(body.contains("metal::precise::divide("));
    for source in [header, body] {
        assert!(!source.contains("metal::exp(") && !source.contains("fast::"));
    }
}

/// The real-weight proofs' PSNR gate on host values (no MLX): a NaN output element makes the
/// score NaN and the gate rejects it (a `db < 50` failure check would have accepted it), while
/// bit-identical outputs score `+inf` and pass.
#[test]
fn the_psnr_gate_rejects_a_nan_output_and_accepts_identical_outputs() {
    const FLOOR_DB: f64 = 50.0;
    let want = [0.25f32, -0.5, 0.75, 1.0];

    let identical = parity::psnr_db(&want, &want, 2.0);
    assert_eq!(identical, f64::INFINITY);
    assert!(parity::psnr_clears(identical, FLOOR_DB));

    let mut nan_output = want;
    nan_output[2] = f32::NAN;
    let nan_db = parity::psnr_db(&nan_output, &want, 2.0);
    assert!(nan_db.is_nan(), "{nan_db}");
    assert!(!parity::psnr_clears(nan_db, FLOOR_DB));
    // The NaN-blind failure check this gate replaces does not fire on it.
    let blind_check_fires = nan_db < FLOOR_DB;
    assert!(!blind_check_fires);
    // A NaN on both sides (a corrupted reference route) is rejected too.
    let both = parity::psnr_db(&nan_output, &nan_output, 2.0);
    assert!(!parity::psnr_clears(both, FLOOR_DB), "{both}");

    let mut close = want;
    close[0] += 1e-4;
    let close_db = parity::psnr_db(&close, &want, 2.0);
    assert!(parity::psnr_clears(close_db, FLOOR_DB), "{close_db}");
    // A NaN range is rejected even for a close output.
    let nan_range = parity::psnr_db(&close, &want, f64::NAN);
    assert!(!parity::psnr_clears(nan_range, FLOOR_DB), "{nan_range}");
    let mut far = want;
    far[0] += 0.5;
    let far_db = parity::psnr_db(&far, &want, 2.0);
    assert!(!parity::psnr_clears(far_db, FLOOR_DB), "{far_db}");
}

fn summarize(reports: &[Report]) -> (usize, usize) {
    let bitwise = reports.iter().filter(|r| r.bitwise()).count();
    let within = reports.iter().filter(|r| r.within_tolerance()).count();
    println!(
        "SW6142_PARITY_SUMMARY cases={} bitwise={bitwise} within_tolerance={within}",
        reports.len()
    );
    (bitwise, within)
}

/// The Metal route over every geometry, gated on E2's declared tolerance only. Bit identity is
/// the kernel's design target (precise exp/divide, contraction off) but is device- and
/// compiler-dependent, so the bitwise count, mismatches and max ULP are printed, never asserted.
#[test]
#[ignore = "MLX compute: SW-6142 fused Metal norm+SiLU+pad vs the reference op chain (E2 tolerance); exhaustive suite, not ordinary CI"]
fn metal_route_matches_the_reference_chain() {
    let reports = parity::run_all(&parity::all_cases(), NormSiluPadRoute::Metal).unwrap();
    let (_, within) = summarize(&reports);
    assert_eq!(
        within,
        reports.len(),
        "a Metal case exceeds max rel {TOLERANCE_F32:e}"
    );
}

#[test]
#[ignore = "MLX compute: SW-6142 compiled norm+SiLU+pad route vs the reference op chain (E2 tolerance); exhaustive suite, not ordinary CI"]
fn compiled_route_matches_the_reference_chain() {
    let reports = parity::run_all(&parity::all_cases(), NormSiluPadRoute::Compiled).unwrap();
    let (_, within) = summarize(&reports);
    assert_eq!(
        within,
        reports.len(),
        "a compiled case exceeds max rel {TOLERANCE_F32:e}"
    );
}

#[test]
#[ignore = "MLX compute: SW-6142 mutation control (temporal pad on the wrong side must fail parity); exhaustive suite, not ordinary CI"]
fn a_wrong_side_temporal_pad_fails_parity() {
    let cases = parity::mutation_cases();
    assert!(cases.len() >= 4, "the mutation needs asymmetric-pad cases");
    for case in &cases {
        let c = parity::run_mutation(case).unwrap();
        println!(
            "SW6142_MUTATION case={} mismatches={} max_abs={:.3e}",
            case.name, c.mismatches, c.max_abs
        );
        assert!(
            !c.bitwise() && c.max_abs > 1e-2,
            "{}: swapping the temporal pad side must break parity",
            case.name
        );
    }
}

#[test]
#[ignore = "MLX compute: SW-6142 try_fused dispatch by route switch with route counters; exhaustive suite, not ordinary CI"]
fn try_fused_follows_the_switch_and_counts_its_route() {
    let case = parity::wan_cases()
        .into_iter()
        .find(|c| c.cache_frames == 2 && !c.cache_view)
        .unwrap();
    let inputs = parity::inputs(&case).unwrap();
    let norm = inputs.norm(&case);
    let spec = inputs.spec(&case);
    let want = reference_conv_input(&inputs.x, &norm, &spec, case.tail).unwrap();

    for route in [NormSiluPadRoute::Metal, NormSiluPadRoute::Compiled] {
        let _guard = RouteGuard::set(route);
        let start = thread_counts();
        let got = try_fused(&inputs.x, &norm, &spec, case.tail)
            .unwrap()
            .expect("the fused route serves the Wan cached chunk");
        let d = thread_counts().since(&start);
        assert_eq!(
            (d.metal, d.compiled, d.reference, d.refused),
            if route == NormSiluPadRoute::Metal {
                (1, 0, 0, 0)
            } else {
                (0, 1, 0, 0)
            },
            "{route:?}"
        );
        let out = parity::compare(&got.ndhwc, &want.ndhwc).unwrap();
        let tail =
            parity::compare(got.tail.as_ref().unwrap(), want.tail.as_ref().unwrap()).unwrap();
        assert!(
            out.within_tolerance() && tail.within_tolerance(),
            "{route:?}"
        );
    }

    let _guard = RouteGuard::set(NormSiluPadRoute::Reference);
    let start = thread_counts();
    assert!(try_fused(&inputs.x, &norm, &spec, case.tail)
        .unwrap()
        .is_none());
    assert_eq!(thread_counts().since(&start).reference, 1);

    // A refused input falls back and says why.
    let _guard = RouteGuard::set(NormSiluPadRoute::Metal);
    let start = thread_counts();
    let rank4 = inputs.x.reshape(&[1, 192, 4, 48]).unwrap();
    assert!(try_fused(&rank4, &norm, &spec, 0).unwrap().is_none());
    let d = thread_counts().since(&start);
    assert_eq!((d.refused, d.last_refusal), (1, Some("rank")));
}
