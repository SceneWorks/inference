//! SCM (sCM / TrigFlow continuous-time consistency) scheduler — the few-step sampler behind
//! **SANA-Sprint** (epic 8485, story sc-8490). Faithful port of diffusers `SCMScheduler`
//! (`schedulers/scheduling_scm.py`) + the Sprint pipeline's per-step trigflow recombination
//! (`pipelines/sana/pipeline_sana_sprint.py`).
//!
//! ## Why this is NOT the unified flow-match sampler
//!
//! The epic-7114 unified framework ([`mlx_gen::run_flow_sampler`] / the `Solver` menu) integrates in
//! **sigma space** with an `x0 = k_x·x + k_out·output` recombination. SCM is a different
//! parameterization: it works in **angle / atan space** (`t ∈ [0, π/2]`, the TrigFlow continuous-time
//! consistency map), predicts `x0 = cos(s)·x − sin(s)·output`, and **renoises** to the next angle
//! `x_{t} = cos(t)·x0 + sin(t)·noise·σ_data`. The model output itself is recombined trigonometrically
//! *before* the scheduler step (the Sprint pipeline's `noise_pred = …` block). None of that maps onto
//! the flow-match `ModelSampling`/`Solver` contract, so SCM is a dedicated scheduler here — the
//! consistency analog of the crate-local [`mlx_gen::FlowMatchEuler`] (which is itself a small
//! scheduler the engines drive directly, not a `Solver`). The unified framework's **seam** we DO reuse
//! is the run-loop contract (per-step `eval` boundary + cooperative cancel + monotone progress),
//! mirrored by [`crate::pipeline::denoise_sprint`].
//!
//! ## Guidance axis (epic 7434)
//!
//! Sprint is **CFG-free**: a single trunk forward per step with the guidance scale fed as an
//! *embedded scalar* (the trunk's guidance embedder, [`crate::transformer::SanaTransformer::forward_with_guidance`]).
//! There is no conditional/unconditional pair to combine, so the epic-7434 guidance library (`cfg` /
//! `cfg_rescale` / `apg` / `cfg_pp` — all *combine* operators) does not apply; Sprint slots into the
//! guidance axis as the embedded / CFG-free case (the descriptor advertises NO
//! `supported_guidance_methods`, [`crate::model`]).

pub use mlx_gen::gen_core::sampling::{
    ScmScheduler, SCM_DEFAULT_INTERMEDIATE_TIMESTEP as DEFAULT_INTERMEDIATE_TIMESTEP,
    SCM_DEFAULT_MAX_TIMESTEP as DEFAULT_MAX_TIMESTEP, SCM_SIGMA_DATA as SIGMA_DATA,
};

#[cfg(test)]
mod tests {
    use super::*;

    fn approx(a: f32, b: f32) {
        assert!((a - b).abs() < 1e-5, "got {a} want {b}");
    }

    #[test]
    fn two_step_schedule_is_diffusers_intermediate() {
        let s = ScmScheduler::new(2);
        assert_eq!(s.timesteps.len(), 3);
        approx(s.timesteps[0], DEFAULT_MAX_TIMESTEP);
        approx(s.timesteps[1], DEFAULT_INTERMEDIATE_TIMESTEP);
        approx(s.timesteps[2], 0.0);
        assert_eq!(s.num_steps(), 2);
    }

    #[test]
    fn four_step_schedule_is_linspace() {
        let s = ScmScheduler::new(4);
        assert_eq!(s.timesteps.len(), 5);
        // linspace(pi/2, 0, 5).
        for (i, &t) in s.timesteps.iter().enumerate() {
            approx(t, DEFAULT_MAX_TIMESTEP * (1.0 - i as f32 / 4.0));
        }
        assert!(s.timesteps.windows(2).all(|w| w[0] > w[1]));
    }

    #[test]
    fn single_step_flag() {
        assert!(ScmScheduler::new(1).is_single_step());
        assert!(!ScmScheduler::new(2).is_single_step());
    }

    #[test]
    fn scm_timestep_endpoints() {
        let s = ScmScheduler::new(4);
        // t = pi/2 -> sin=1, cos=0 -> scm = 1.
        approx(s.scm_timestep(0), 1.0);
        // t = 0 -> sin=0 -> scm = 0.
        approx(s.scm_timestep(4), 0.0);
        // input_scale at scm=1 is sqrt(1+0)=1; at scm=0 is sqrt(0+1)=1.
        approx(s.input_scale(0), 1.0);
        approx(s.input_scale(4), 1.0);
    }
}
