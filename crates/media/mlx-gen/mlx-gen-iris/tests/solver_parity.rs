//! FlowDPM-Solver++ vs upstream `FlowDPMSolver` (order 2, CFG 2.5 on the raw output, shift 4,
//! 7 steps — order ramps 1→2→…→2→1 and the last step is the exact x0 projection) on an analytic
//! model function both sides implement identically, so the comparison isolates the solver.
//!
//! Tolerance: FP32 integration state on both sides with identical f32 coefficients; 1e-5 of peak
//! covers MLX vs torch elementwise rounding. The 100-step default grid is compared in f64.

use mlx_gen::gen_core::iris::{dpm_solver_plan, time_grid, Prediction};
use mlx_gen::CancelFlag;
use mlx_gen_iris::solver::{cfg_combine, sample};
use mlx_rs::ops::{stack_axis, tanh};
use mlx_rs::Array;

use crate::common::{assert_close, fixture};

/// `tanh(x)·(0.5 + t) + y·(1 − t)` with `t = t_model / 1000` — the oracle's `analytic_model`.
fn analytic(x: &Array, t_model: f32, y: f32) -> Array {
    let t = t_model / 1000.0;
    tanh(x)
        .unwrap()
        .multiply(Array::from_f32(0.5 + t))
        .unwrap()
        .add(Array::from_f32(y * (1.0 - t)))
        .unwrap()
}

/// The f64 grids travel as exact JSON in the fixture metadata (MLX cannot load F64 tensors).
fn grid_meta(golden: &mlx_gen::weights::Weights, key: &str) -> Vec<f64> {
    serde_json::from_str(golden.metadata(key).expect("grid metadata")).expect("grid json")
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
    let z = golden.require("z").unwrap();
    let plan = dpm_solver_plan(7, 2, 4.0).unwrap();
    let (cond, uncond, scale) = (0.7f32, -0.4f32, 2.5f32);
    let mut states = Vec::new();
    let mut seen = Vec::new();
    let final_x = sample(
        z,
        &plan,
        Prediction::Velocity,
        &CancelFlag::new(),
        |x, step| {
            let t = step.model_time(1000);
            let out = cfg_combine(&analytic(x, t, uncond), &analytic(x, t, cond), scale)?;
            Ok(out)
        },
        |i, _| seen.push(i),
    )
    .unwrap();
    // re-run recording each state
    let mut x = z.clone();
    for n in 1..=plan.len() {
        x = sample(
            z,
            &plan[..n],
            Prediction::Velocity,
            &CancelFlag::new(),
            |x, step| {
                let t = step.model_time(1000);
                cfg_combine(&analytic(x, t, uncond), &analytic(x, t, cond), scale)
            },
            |_, _| {},
        )
        .unwrap();
        states.push(x.clone());
    }
    assert_eq!(seen, (1..=7).collect::<Vec<_>>());
    let refs: Vec<&Array> = states.iter().collect();
    let got = stack_axis(&refs, 0).unwrap();
    assert_close("trajectory", &got, golden.require("states").unwrap(), 1e-5);
    assert_close("final", &final_x, &x, 0.0);
}

#[test]
fn cancellation_returns_the_typed_error_not_a_partial_state() {
    let golden = fixture("iris_solver_golden.safetensors");
    let plan = dpm_solver_plan(7, 2, 4.0).unwrap();
    let cancel = CancelFlag::new();
    let mut calls = 0;
    let result = sample(
        golden.require("z").unwrap(),
        &plan,
        Prediction::Velocity,
        &cancel,
        |x, step| {
            calls += 1;
            Ok(analytic(x, step.model_time(1000), 0.0))
        },
        |i, _| {
            if i == 2 {
                cancel.cancel();
            }
        },
    );
    assert!(matches!(result, Err(mlx_gen::Error::Canceled)));
    assert_eq!(calls, 2, "no network evaluation after the flag trips");
}

/// sc-25681: `on_step(i, x0)` hands out step `i`'s predicted clean image (the preview source) under
/// both readings of the network output — `v` (`x0 = x − s·out`) and `x` (`x0 = out`) — and the two
/// readings integrate to different trajectories.
#[test]
fn the_step_callback_receives_each_steps_predicted_clean_image() {
    let golden = fixture("iris_solver_golden.safetensors");
    let z = golden.require("z").unwrap();
    let plan = dpm_solver_plan(7, 2, 4.0).unwrap();
    let (cond, uncond, scale) = (0.7f32, -0.4f32, 2.5f32);
    let model =
        |x: &Array, t: f32| cfg_combine(&analytic(x, t, uncond), &analytic(x, t, cond), scale);
    let mut finals = Vec::new();
    for prediction in [Prediction::Velocity, Prediction::Clean] {
        let mut x0s = Vec::new();
        let mut states = vec![z.clone()];
        for n in 1..=plan.len() {
            let x = sample(
                z,
                &plan[..n],
                prediction,
                &CancelFlag::new(),
                |x, step| model(x, step.model_time(1000)),
                |i, x0| {
                    if i == n {
                        x0s.push(x0.clone())
                    }
                },
            )
            .unwrap();
            states.push(x);
        }
        for (i, step) in plan.iter().enumerate() {
            let out = model(&states[i], step.model_time(1000)).unwrap();
            let want = match prediction {
                Prediction::Velocity => states[i]
                    .subtract(out.multiply(Array::from_f32(step.s_f32())).unwrap())
                    .unwrap(),
                Prediction::Clean => out,
            };
            assert_close(&format!("{prediction:?} x0[{i}]"), &x0s[i], &want, 1e-6);
        }
        // The terminal update is the exact x0 projection: the last x0 IS the result.
        assert_close(
            "terminal x0",
            states.last().unwrap(),
            x0s.last().unwrap(),
            0.0,
        );
        finals.push(states.pop().unwrap());
    }
    let (moved, _, _) = crate::common::errors(&finals[0], &finals[1]);
    assert!(
        moved > 0.1,
        "the prediction reading must change the trajectory ({moved})"
    );
}
