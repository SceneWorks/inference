//! Current-phase source-prepared proofs; CPU/native execution and individual RED mutations are
//! held until explicit CPU admission. The native test isolates real solver allocation, not a model.
use super::*;
use crate::gen_core::sampling::solvers::{
    SolverEvaluation, SolverObservation, SolverObservationKind,
};
use crate::gen_core::sampling::{CpuLatentOps, FlowModelSampling, LatentOps};

#[derive(Clone, Debug)]
struct RecordedExecution {
    kind: SolverObservationKind,
    history: u8,
    outer: u8,
    stage: u8,
    next_bits: u32,
    alias: bool,
    scalar_history: bool,
    cfg_pp: bool,
}
#[derive(Default)]
struct ObservedCpuOps {
    events: std::cell::RefCell<Vec<RecordedExecution>>,
}
impl LatentOps for ObservedCpuOps {
    type Latent = Vec<f32>;
    fn scale(&self, x: &Vec<f32>, s: f32) -> crate::gen_core::Result<Vec<f32>> {
        CpuLatentOps.scale(x, s)
    }
    fn add(&self, a: &Vec<f32>, b: &Vec<f32>) -> crate::gen_core::Result<Vec<f32>> {
        CpuLatentOps.add(a, b)
    }
    fn sub(&self, a: &Vec<f32>, b: &Vec<f32>) -> crate::gen_core::Result<Vec<f32>> {
        CpuLatentOps.sub(a, b)
    }
    fn randn_like(
        &self,
        x: &Vec<f32>,
        seed: u64,
        step: usize,
    ) -> crate::gen_core::Result<Vec<f32>> {
        CpuLatentOps.randn_like(x, seed, step)
    }
    fn observe_solver(&self, e: SolverObservation<'_, Vec<f32>>) {
        self.events.borrow_mut().push(RecordedExecution {
            kind: e.kind,
            history: e.history.iter().filter(|x| x.is_some()).count() as u8,
            outer: e.outer.iter().filter(|x| x.is_some()).count() as u8,
            stage: e.correction_order,
            next_bits: e.next_sigma.to_bits(),
            alias: e.terminal_alias,
            scalar_history: e.scalar_history,
            cfg_pp: e.cfg_pp,
        });
    }
}

#[test]
fn real_factory_observations_come_from_executed_branches_and_preserve_output() {
    let ms = FlowModelSampling::new(TimestepConvention::Sigma);
    let sigmas = [1.0, 0.8, 0.5, 0.25, 0.0];
    for solver in Solver::ALL {
        let ops = ObservedCpuOps::default();
        let mut callbacks = 0;
        let observed = solver
            .boxed::<ObservedCpuOps>()
            .sample(
                &ops,
                &ms,
                &mut |_: &Vec<f32>, _| {
                    callbacks += 1;
                    Ok(vec![0.25])
                },
                vec![0.5],
                &sigmas,
                42,
            )
            .unwrap();
        let direct = solver
            .boxed::<CpuLatentOps>()
            .sample(
                &CpuLatentOps,
                &ms,
                &mut |_: &Vec<f32>, _| Ok(vec![0.25]),
                vec![0.5],
                &sigmas,
                42,
            )
            .unwrap();
        assert_eq!(observed, direct, "observer must not alter solver output");
        let events = ops.events.borrow();
        assert_eq!(
            events
                .iter()
                .filter(|e| matches!(e.kind, SolverObservationKind::Callback(_)))
                .count(),
            callbacks
        );
        assert_eq!(
            events
                .iter()
                .filter(|e| e.kind == SolverObservationKind::Update)
                .count(),
            sigmas.len() - 1
        );
        assert!(events.iter().all(|e| !e.cfg_pp));
        match solver {
            Solver::ErSde => assert!(events
                .iter()
                .any(|e| e.kind == SolverObservationKind::Correction
                    && e.stage == 3
                    && e.history == 2)),
            Solver::Dpmpp2mSde => assert!(events
                .iter()
                .any(|e| e.kind == SolverObservationKind::Correction
                    && e.history == 1
                    && e.scalar_history)),
            Solver::Heun => assert!(events.iter().any(|e| e.kind
                == SolverObservationKind::Callback(SolverEvaluation::HeunCorrector)
                && e.outer == 3)),
            Solver::DpmppSde => assert!(events.iter().any(|e| e.kind
                == SolverObservationKind::Callback(SolverEvaluation::SdeMidpoint)
                && e.outer == 2)),
            Solver::UniPc => assert!(events
                .iter()
                .any(|e| e.kind == SolverObservationKind::Correction
                    && e.next_bits == 0
                    && e.history == 2)),
            _ => {}
        }
        if matches!(
            solver,
            Solver::Dpmpp2m | Solver::Dpmpp2mSde | Solver::ErSde | Solver::UniPc | Solver::Lcm
        ) {
            assert!(events.last().unwrap().alias);
        }
    }
}

#[test]
fn cfgpp_actual_factory_observations_report_unconditional_history() {
    for base in [Solver::Euler, Solver::Ddim, Solver::Dpmpp2m] {
        let ops = ObservedCpuOps::default();
        let sigmas = [1.0, 0.75, 0.5, 0.0];
        let mut callbacks = 0;
        let observed = crate::gen_core::sampling::cfgpp_sampler_for::<ObservedCpuOps>(base)
            .unwrap()
            .sample(
                &ops,
                &mut |_: &Vec<f32>, _| {
                    callbacks += 1;
                    Ok((vec![0.25], vec![0.125]))
                },
                vec![0.5],
                &sigmas,
            )
            .unwrap();
        let direct = crate::gen_core::sampling::cfgpp_sampler_for::<CpuLatentOps>(base)
            .unwrap()
            .sample(
                &CpuLatentOps,
                &mut |_: &Vec<f32>, _| Ok((vec![0.25], vec![0.125])),
                vec![0.5],
                &sigmas,
            )
            .unwrap();
        assert_eq!(observed, direct);
        let events = ops.events.borrow();
        assert!(events.iter().all(|e| e.cfg_pp));
        assert_eq!(
            events
                .iter()
                .filter(|e| matches!(e.kind, SolverObservationKind::Callback(_)))
                .count(),
            callbacks
        );
        assert_eq!(
            events
                .iter()
                .any(|e| e.kind == SolverObservationKind::Correction && e.history == 1),
            base == Solver::Dpmpp2m
        );
    }
}

fn defaults() -> FlowSamplingDefaults {
    FlowSamplingDefaults {
        solver: Solver::ErSde,
        steps: 10,
        guidance: None,
        true_cfg_overrides: false,
        default_alias: false,
    }
}

#[test]
fn actual_factories_match_epoch_cadence_including_er_history_and_zero_windows() {
    let ms = FlowModelSampling::new(TimestepConvention::Sigma);
    for solver in Solver::ALL {
        for sigmas in [
            &[1.0, 0.8, 0.5, 0.25, 0.0][..],
            &[1.0, 0.8, 0.5][..],
            &[0.0][..],
            &[0.0, 0.0][..],
            &[1.0, 1.0, 0.0][..],
        ] {
            let expected = solver.logical_epochs(sigmas).unwrap();
            let mut seen = Vec::new();
            solver
                .boxed::<CpuLatentOps>()
                .sample(
                    &CpuLatentOps,
                    &ms,
                    &mut |_: &Vec<f32>, sigma| {
                        seen.push(sigma);
                        Ok(vec![0.25])
                    },
                    vec![0.5],
                    sigmas,
                    0,
                )
                .unwrap();
            assert_eq!(
                seen,
                expected.iter().map(|e| e.sigma).collect::<Vec<_>>(),
                "{} {sigmas:?}",
                solver.name()
            );
        }
    }
    let epochs = Solver::ErSde
        .logical_epochs(&[1.0, 0.8, 0.5, 0.25, 0.0])
        .unwrap();
    assert_eq!(
        epochs
            .iter()
            .map(|e| e.history_owners())
            .collect::<Vec<_>>(),
        vec![0, 1, 2, 2]
    );
    assert_eq!(
        epochs
            .iter()
            .map(|e| e.correction_order)
            .collect::<Vec<_>>(),
        vec![1, 2, 3, 0]
    );
    assert!(epochs.last().unwrap().state_aliases_denoised_after_step);
    let epochs = Solver::Dpmpp2mSde.logical_epochs(&[1.0, 0.5, 0.0]).unwrap();
    assert_eq!(
        epochs
            .iter()
            .map(|e| e.history_owners())
            .collect::<Vec<_>>(),
        vec![0, 1]
    );
    assert!(epochs.last().unwrap().state_aliases_denoised_after_step);
}

#[test]
fn recipe_replays_normalized_defaults_cfg_precedence_and_aliases() {
    let native = crate::FlowMatchEuler::for_static_shift(10, 3.0);
    let req = GenerationRequest::default();
    let plan = ResolvedFlowSampling::resolve(
        &req,
        defaults(),
        &native.sigmas,
        3.0_f32.ln(),
        TimestepConvention::Sigma,
    )
    .unwrap();
    assert_eq!(plan.solver(), Solver::ErSde);
    assert_eq!(plan.sigmas(), native.sigmas);
    let recipe = plan.recipe();
    let roundtrip = FlowSamplingRecipe::from_token(&recipe.canonical_token()).unwrap();
    assert_eq!(&roundtrip, recipe);
    assert_eq!(roundtrip.to_request().steps, Some(10));
    assert!(require_generation(None).is_err());
    let mut cfg_defaults = defaults();
    cfg_defaults.guidance = Some(4.5);
    cfg_defaults.true_cfg_overrides = true;
    cfg_defaults.default_alias = true;
    let req = GenerationRequest {
        sampler: Some("default".into()),
        scheduler: Some("default".into()),
        guidance: Some(1.0),
        true_cfg: Some(7.0),
        ..Default::default()
    };
    let plan = ResolvedFlowSampling::resolve(
        &req,
        cfg_defaults,
        &native.sigmas,
        3.0_f32.ln(),
        TimestepConvention::Sigma,
    )
    .unwrap();
    assert_eq!(plan.effective_guidance(), Some(7.0));
    assert_eq!(plan.recipe().to_request().guidance, Some(7.0));
    assert_eq!(plan.recipe().to_request().true_cfg, None);
    assert!(plan.recipe().validate_defaults(defaults()).is_err());
    // Binding a missing guidance to the actual effective default prevents old-None exact-key alias.
    let default_cfg = ResolvedFlowSampling::resolve(
        &GenerationRequest::default(),
        cfg_defaults,
        &native.sigmas,
        3.0_f32.ln(),
        TimestepConvention::Sigma,
    )
    .unwrap();
    cfg_defaults.guidance = Some(1.0);
    let changed_cfg = ResolvedFlowSampling::resolve(
        &GenerationRequest::default(),
        cfg_defaults,
        &native.sigmas,
        3.0_f32.ln(),
        TimestepConvention::Sigma,
    )
    .unwrap();
    assert_ne!(
        default_cfg.recipe().canonical_token(),
        changed_cfg.recipe().canonical_token()
    );
    let replay = ResolvedFlowSampling::resolve(
        &default_cfg.recipe().to_request(),
        cfg_defaults,
        &native.sigmas,
        3.0_f32.ln(),
        TimestepConvention::Sigma,
    )
    .unwrap();
    assert_eq!(replay.effective_guidance(), Some(4.5));
    let mut value = serde_json::to_value(recipe).unwrap();
    for field in [
        "version",
        "guidance",
        "scheduler",
        "strength",
        "reference_strength",
        "pid_capture_sigma",
    ] {
        let mut missing = value.clone();
        missing.as_object_mut().unwrap().remove(field);
        assert!(
            serde_json::from_value::<FlowSamplingRecipe>(missing).is_err(),
            "{field}"
        );
    }
    let mut captured = serde_json::to_value(default_cfg.recipe()).unwrap();
    captured["strength"] = serde_json::json!(0.2);
    captured["reference_strength"] = serde_json::json!(0.75);
    captured["steps"] = serde_json::json!(u32::MAX);
    let captured: FlowSamplingRecipe = serde_json::from_value(captured).unwrap();
    let direct =
        ResolvedFlowControls::from_recipe(&captured, cfg_defaults, TimestepConvention::Sigma)
            .unwrap();
    assert_eq!(direct.recipe(), &captured); // retain BOTH raw controls, no fake reference carrier
    assert_eq!(direct.recipe().effective_strength(), Some(0.75));
    assert_eq!(direct.steps(), u32::MAX as usize);
    assert_eq!(direct.effective_guidance(), Some(4.5));
    assert!(
        ResolvedFlowControls::from_recipe(&captured, defaults(), TimestepConvention::Sigma)
            .is_err()
    );
    value["class"] = serde_json::json!("caller-claimed");
    assert!(serde_json::from_value::<FlowSamplingRecipe>(value).is_err());
}

#[test]
fn logical_memory_uses_factory_history_state_promotion_and_aliases_not_multipliers() {
    let sigmas = [1.0, 0.8, 0.5, 0.25, 0.0];
    for solver in Solver::ALL {
        let native = crate::FlowMatchEuler::from_sigmas(sigmas.to_vec()).unwrap();
        let req = GenerationRequest {
            sampler: Some(solver.name().into()),
            steps: Some(4),
            ..Default::default()
        };
        let plan = ResolvedFlowSampling::resolve(
            &req,
            defaults(),
            &native.sigmas,
            3.0_f32.ln(),
            TimestepConvention::OneMinusSigma,
        )
        .unwrap();
        let law = plan.memory(16, 2, 4).unwrap();
        assert_eq!(law.callbacks, plan.epochs().unwrap().len());
        assert!(law.state_bytes >= 16 * 2);
        assert!(law.step_peak_bytes >= law.final_bytes);
    }
    let native = crate::FlowMatchEuler::from_sigmas(vec![1.0, 0.5, 0.0]).unwrap();
    let euler = ResolvedFlowSampling::resolve(
        &GenerationRequest {
            sampler: Some("euler".into()),
            ..Default::default()
        },
        defaults(),
        &native.sigmas,
        0.0,
        TimestepConvention::Sigma,
    )
    .unwrap();
    let law = euler.memory(16, 2, 4).unwrap();
    assert_eq!(law.state_bytes, 16 * 4); // actual axpy promotion follows F32 prediction, not initial BF16
    assert_eq!(law.callback_outer_bytes, 0); // no history: no fictitious all-run retained buffer set
}

#[test]
fn model_sampling_and_cfgpp_windows_use_the_actual_selected_factories() {
    use super::super::solver_memory::solver_callback_windows_for_model;
    let alpha = crate::AlphaSchedule::scaled_linear(1000, 0.00085, 0.012);
    let ms = crate::DiscreteModelSampling::sdxl(&alpha);
    for solver in [Solver::Euler, Solver::Ddim, Solver::Dpmpp2m] {
        let windows = solver_callback_windows_for_model(solver, true, &ms, 16, 4, 2).unwrap();
        assert!(!windows.is_empty());
        assert!(windows.iter().all(|window| window.cfg_pp));
        assert!(windows.iter().any(|window| !window.scaled_input_alias));
        assert!(windows
            .iter()
            .any(|window| window.scaled_input_alias && window.epoch.sigma == 0.0));
        assert!(windows
            .iter()
            .all(|window| window.scaled_input_bytes == window.state_bytes));
        if solver == Solver::Dpmpp2m {
            assert!(windows.iter().any(|window| window.epoch.previous_denoised));
            assert!(windows
                .iter()
                .filter(|window| window.epoch.sigma > 0.0 && window.epoch.next_sigma == 0.0)
                .all(|window| !window.epoch.state_aliases_denoised_after_step));
        }
    }
    let ordinary =
        solver_callback_windows_for_model(Solver::Dpmpp2m, false, &ms, 16, 4, 2).unwrap();
    assert!(ordinary.iter().any(|window| window.epoch.sigma > 0.0
        && window.epoch.next_sigma == 0.0
        && window.epoch.state_aliases_denoised_after_step
        && !window.cfg_pp));
    assert!(solver_callback_windows_for_model(Solver::Heun, true, &ms, 16, 4, 2).is_err());
}

#[test]
fn finite_callback_windows_keep_coincident_history_correctors_and_zero_families() {
    use super::super::solver_memory::solver_callback_windows;
    for solver in Solver::ALL {
        let windows = solver_callback_windows(solver, TimestepConvention::Sigma, 16, 2, 4).unwrap();
        assert!(!windows.is_empty());
        assert!(windows.iter().any(|w| w.epoch.sigma == 0.0));
        assert!(windows.iter().any(|w| w.epoch.next_sigma == 0.0));
        assert!(windows
            .iter()
            .all(|w| w.state_bytes == 32 || w.state_bytes == 64));
        assert!(windows.iter().all(|w| !w.family().is_empty()));
        if solver == Solver::Heun {
            assert!(windows
                .iter()
                .any(|w| w.epoch.evaluation == SolverEvaluation::HeunCorrector
                    && w.epoch.outer_owners == 3
                    && w.callback_outer_bytes > 0));
        }
        if solver == Solver::DpmppSde {
            assert!(windows
                .iter()
                .any(|w| w.epoch.evaluation == SolverEvaluation::SdeMidpoint
                    && w.epoch.outer_owners == 2
                    && w.callback_outer_bytes > 0));
        }
        if solver == Solver::ErSde {
            assert!(windows.iter().any(|w| w.epoch.correction_order == 3
                && w.epoch.history_owners() == 2
                && w.callback_outer_bytes > 0));
        }
        if solver == Solver::UniPc {
            assert!(windows
                .iter()
                .any(|w| w.epoch.corrector_after_callback && w.epoch.next_sigma == 0.0));
        }
    }
}

#[test]
fn controls_extreme_steps_and_serial_count_stay_bounded_metadata_only() {
    // Source-checked path: Controls never builds a requested-N schedule. The symbolic envelope
    // runs only minimum_capture_transitions (2..=4), not requested steps or serial output count.
    for solver in Solver::ALL {
        let small_request = GenerationRequest {
            sampler: Some(solver.name().into()),
            steps: Some(4),
            count: 1,
            ..Default::default()
        };
        let mut huge_request = small_request.clone();
        huge_request.steps = Some(u32::MAX);
        huge_request.count = u32::MAX;
        let small =
            ResolvedFlowControls::resolve(&small_request, defaults(), TimestepConvention::Sigma)
                .unwrap();
        let huge =
            ResolvedFlowControls::resolve(&huge_request, defaults(), TimestepConvention::Sigma)
                .unwrap();
        assert_eq!(huge.steps(), u32::MAX as usize);
        let restored = FlowSamplingRecipe::from_token(&huge.recipe().canonical_token()).unwrap();
        restored.validate_controls(&huge).unwrap();
        assert_eq!(restored.to_request().steps, Some(u32::MAX));
        assert!(huge.host_schedule_bytes().unwrap() > small.host_schedule_bytes().unwrap());
        assert!(huge.check_capture_schedule_bytes(1 << 20).is_err());
        let small_memory = small.memory_envelope(16, 2, 4).unwrap();
        let huge_memory = huge.memory_envelope(16, 2, 4).unwrap();
        assert_eq!(huge_memory.callbacks, small_memory.callbacks);
        assert!(huge_memory.callbacks > 0 && huge_memory.callbacks <= 8);
        assert_eq!(
            huge_memory.callback_outer_bytes,
            small_memory.callback_outer_bytes
        );
        assert_eq!(huge_memory.step_peak_bytes, small_memory.step_peak_bytes);
        assert_eq!(huge_memory.state_bytes, small_memory.state_bytes);
        assert_eq!(huge_memory.final_bytes, small_memory.final_bytes);
        huge_request.count = 1;
        let single =
            ResolvedFlowControls::resolve(&huge_request, defaults(), TimestepConvention::Sigma)
                .unwrap();
        assert_eq!(huge.recipe(), single.recipe());
        huge_request.steps = Some(0);
        assert!(ResolvedFlowControls::resolve(
            &huge_request,
            defaults(),
            TimestepConvention::Sigma,
        )
        .is_err());
    }
}

#[cfg(target_os = "macos")]
#[test]
fn small_real_native_solver_allocation_checks_logical_adapter_assumptions() {
    const CHILD: &str = "FLOW_SOLVER_NATIVE_PROOF_CHILD";
    if std::env::var_os(CHILD).is_none() {
        let status=std::process::Command::new(std::env::current_exe().unwrap())
            .args(["--exact","sampler::flow_sampling::tests::small_real_native_solver_allocation_checks_logical_adapter_assumptions","--nocapture"])
            .env(CHILD,"1").status().unwrap();
        assert!(status.success());
        return;
    }
    assert!(crate::ios_quantum::quantum_metal_guard::forbid_metal());
    mlx_rs::Device::set_default(&mlx_rs::Device::cpu());
    crate::model_inference_device::record_model_inference_device();
    assert_eq!(
        unsafe { mlx_sys::mlx_set_default_stream(mlx_rs::Stream::cpu().as_ptr()) },
        0
    );
    assert_eq!(
        mlx_rs::Stream::new().get_index().unwrap(),
        mlx_rs::Stream::cpu().get_index().unwrap()
    );
    let ms = FlowModelSampling::new(TimestepConvention::Sigma);
    let sigmas = [1.0, 0.8, 0.5, 0.25, 0.0];
    // Four dtype mechanisms, not a scheduler/steps/guidance measurement grid or machine golden.
    for (initial_width, prediction_width) in [(2, 2), (2, 4), (4, 2), (4, 4)] {
        for solver in Solver::ALL {
            let dtype = |width| {
                if width == 2 {
                    mlx_rs::Dtype::Bfloat16
                } else {
                    mlx_rs::Dtype::Float32
                }
            };
            let initial = mlx_rs::Array::from_slice(&[0.5f32; 16], &[16])
                .as_dtype(dtype(initial_width))
                .unwrap();
            mlx_rs::transforms::eval([&initial]).unwrap();
            mlx_rs::memory::clear_cache();
            let before = mlx_rs::memory::get_active_memory();
            mlx_rs::memory::reset_peak_memory();
            let mut callbacks = 0;
            let output = crate::run_curated_sampler(
                crate::SamplerRun {
                    sampler_name: Some(solver.name()),
                    sigmas: &sigmas,
                    seed: 0,
                    cancel: &crate::CancelFlag::new(),
                    on_progress: &mut |_| {},
                },
                &ms,
                initial,
                |_, _| {
                    callbacks += 1;
                    let prediction = mlx_rs::Array::from_slice(&[0.25f32; 16], &[16])
                        .as_dtype(dtype(prediction_width))?;
                    // Provider output quantum materializes the actual prediction before returning it.
                    mlx_rs::transforms::eval([&prediction])?;
                    Ok(prediction)
                },
            )
            .unwrap();
            mlx_rs::transforms::eval([&output]).unwrap();
            let measured = mlx_rs::memory::get_peak_memory().saturating_sub(before) as u64;
            let law = super::super::solver_memory::solver_memory(
                solver,
                &ms,
                &sigmas,
                16,
                initial_width,
                prediction_width,
            )
            .unwrap();
            assert_eq!(callbacks, law.callbacks);
            assert_eq!(
                crate::ios_quantum::bytes(&output),
                law.final_bytes,
                "actual solver state promotion"
            );
            assert!(
                measured > 0,
                "native allocation observer must not silently report an empty proof"
            );
            assert!(law.step_peak_bytes>=measured,"{} widths{initial_width}/{prediction_width}: raw logical upper{} < observed{measured}",solver.name(),law.step_peak_bytes);
            if measured != 0 && law.step_peak_bytes > measured.saturating_mul(3) / 2 {
                eprintln!(
                    "REPORT ONLY {} logical{} measured{measured}",
                    solver.name(),
                    law.step_peak_bytes
                );
            }
            drop(output);
        }
    }
}
