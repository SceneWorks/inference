//! FlowDPM-Solver++ vs upstream `FlowDPMSolver` (order 2, CFG 2.5 on the raw output, shift 4,
//! 7 steps — order ramps 1→2→…→2→1 and the last step is the exact x0 projection) on an analytic
//! model function both sides implement identically, so the comparison isolates the solver. Same
//! fixture as the MLX twin's `solver_parity`.
//!
//! Tolerance: FP32 integration state on both sides with identical f32 coefficients; 1e-5 of peak
//! covers Candle vs torch elementwise rounding. The 100-step default grid is compared in f64.

use candle_gen::candle_core::Tensor;
use candle_gen::gen_core::iris::{dpm_solver_plan, time_grid};
use candle_gen::gen_core::CancelFlag;
use candle_gen_iris::solver::{cfg_combine, sample};

use crate::common::{assert_close, fixture, Fixture};

/// `tanh(x)·(0.5 + t) + y·(1 − t)` with `t = t_model / 1000` — the oracle's `analytic_model`.
fn analytic(x: &Tensor, t_model: f32, y: f32) -> Tensor {
    let t = t_model / 1000.0;
    let a = Tensor::new(0.5 + t, x.device()).unwrap();
    let b = Tensor::new(y * (1.0 - t), x.device()).unwrap();
    x.tanh()
        .unwrap()
        .broadcast_mul(&a)
        .unwrap()
        .broadcast_add(&b)
        .unwrap()
}

/// The f64 grids travel as exact JSON in the fixture metadata.
fn grid_meta(golden: &Fixture, key: &str) -> Vec<f64> {
    serde_json::from_str(golden.meta(key)).expect("grid json")
}

fn model(x: &Tensor, t: f32) -> candle_gen::Result<Tensor> {
    let (cond, uncond, scale) = (0.7f32, -0.4f32, 2.5f32);
    cfg_combine(&analytic(x, t, uncond), &analytic(x, t, cond), scale)
}

#[test]
fn default_grid_matches_upstream_time_grid() {
    let golden = fixture("iris_solver_golden.safetensors");
    for (key, steps) in [("default_grid", 100), ("grid", 7)] {
        let want = grid_meta(&golden, key);
        let got = time_grid(steps, 4.0);
        assert_eq!(got.len(), want.len());
        for (i, (g, w)) in got.iter().zip(&want).enumerate() {
            assert!((g - w).abs() <= 1e-15, "{key}[{i}]: {g} vs {w}");
        }
    }
}

#[test]
fn multistep_trajectory_matches_upstream() {
    let golden = fixture("iris_solver_golden.safetensors");
    let z = golden.require("z");
    let plan = dpm_solver_plan(7, 2, 4.0).unwrap();
    let mut seen = Vec::new();
    let final_x = sample(
        z,
        &plan,
        &CancelFlag::new(),
        |x, step| model(x, step.model_time(1000)),
        |i| seen.push(i),
    )
    .unwrap();
    // re-run recording each state
    let mut states = Vec::new();
    for n in 1..=plan.len() {
        let x = sample(
            z,
            &plan[..n],
            &CancelFlag::new(),
            |x, step| model(x, step.model_time(1000)),
            |_| {},
        )
        .unwrap();
        states.push(x);
    }
    assert_eq!(seen, (1..=7).collect::<Vec<_>>());
    let got = Tensor::stack(&states, 0).unwrap();
    assert_close("trajectory", &got, golden.require("states"), 1e-5);
    assert_close("final", &final_x, states.last().unwrap(), 0.0);
}

#[test]
fn cancellation_returns_the_typed_error_not_a_partial_state() {
    let golden = fixture("iris_solver_golden.safetensors");
    let plan = dpm_solver_plan(7, 2, 4.0).unwrap();
    let cancel = CancelFlag::new();
    let mut calls = 0;
    let result = sample(
        golden.require("z"),
        &plan,
        &cancel,
        |x, step| {
            calls += 1;
            Ok(analytic(x, step.model_time(1000), 0.0))
        },
        |i| {
            if i == 2 {
                cancel.cancel();
            }
        },
    );
    assert!(matches!(result, Err(candle_gen::CandleError::Canceled)));
    assert_eq!(calls, 2, "no network evaluation after the flag trips");
}
