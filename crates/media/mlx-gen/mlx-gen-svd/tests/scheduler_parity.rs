//! SVD EDM scheduler parity vs the real diffusers `EulerDiscreteScheduler` (epic 3040 / sc-3371).
//! Gates `EdmSchedule::karras` (sigmas + continuous timesteps + init_noise_sigma), `scale_model_input`
//! (c_in), `v_pred_denoised` (v-prediction → x̂0), and `euler_step` against a committed golden
//! (`tools/dump_svd_scheduler_golden.py`, scheduler configured exactly like the SVD checkpoint).

use mlx_gen::testing::max_abs_diff;
use mlx_gen::weights::Weights;
use mlx_gen_svd::scheduler::{euler_step, scale_model_input, v_pred_denoised};
use mlx_gen_svd::{EdmSchedule, SchedulerConfig};

const GOLDEN: &str = concat!(
    env!("CARGO_MANIFEST_DIR"),
    "/tests/fixtures/svd_scheduler_golden.safetensors"
);

#[test]
fn svd_scheduler_matches_diffusers() {
    let g = Weights::from_file(GOLDEN).expect("svd scheduler golden");
    let sched = EdmSchedule::karras(25, &SchedulerConfig::default());

    // sigmas (26) + timesteps (25): relative to the 700-scale, compare via max|Δ|.
    let g_sigmas = g.require("sigmas").unwrap().as_slice::<f32>().to_vec();
    let g_ts = g.require("timesteps").unwrap().as_slice::<f32>().to_vec();
    assert_eq!(sched.sigmas.len(), g_sigmas.len());
    assert_eq!(sched.timesteps.len(), g_ts.len());
    for (i, (a, b)) in sched.sigmas.iter().zip(&g_sigmas).enumerate() {
        let rel = (a - b).abs() / b.abs().max(1.0);
        assert!(rel < 1e-4, "sigma[{i}] {a} vs {b} (rel {rel})");
    }
    for (i, (a, b)) in sched.timesteps.iter().zip(&g_ts).enumerate() {
        assert!((a - b).abs() < 1e-4, "timestep[{i}] {a} vs {b}");
    }
    let g_ins = g.require("init_noise_sigma").unwrap().as_slice::<f32>()[0];
    assert!(
        (sched.init_noise_sigma() - g_ins).abs() < 1e-2,
        "init_noise_sigma"
    );

    // One step at index 5: scale_model_input, v_pred, euler.
    let sigma = sched.sigmas[5];
    let sigma_next = sched.sigmas[6];
    let x = g.require("x").unwrap();
    let v = g.require("v").unwrap();

    let scaled = scale_model_input(x, sigma).unwrap();
    assert!(
        max_abs_diff(&scaled, g.require("scaled").unwrap()) < 1e-4,
        "scale_model_input"
    );

    let pred = v_pred_denoised(v, x, sigma).unwrap();
    let d_pred = max_abs_diff(&pred, g.require("pred_x0").unwrap());
    // x̂0 magnitude is O(|v|); allow a small f32 tolerance.
    assert!(d_pred < 1e-3, "v_pred_denoised max|Δ| {d_pred}");

    let prev = euler_step(x, &pred, sigma, sigma_next).unwrap();
    let d_prev = max_abs_diff(&prev, g.require("prev").unwrap());
    assert!(d_prev < 1e-2, "euler_step max|Δ| {d_prev}");

    println!("scheduler parity: sigmas/timesteps OK, pred Δ {d_pred}, prev Δ {d_prev}");
}
