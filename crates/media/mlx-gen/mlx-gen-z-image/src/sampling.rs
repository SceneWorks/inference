//! Z-Image Turbo/control's shared actual FLOW factory projection; no base-model policy changes.

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn largest_legal_metadata_request_resolves_without_materializing_steps() {
        let req = GenerationRequest {
            steps: Some(u32::MAX),
            count: u32::MAX,
            ..Default::default()
        };
        let controls = resolve(&req).unwrap();
        assert_eq!(controls.steps(), u32::MAX as usize);
        assert!(controls.host_schedule_bytes().unwrap() >= (u64::from(u32::MAX) + 1) * 4);
        assert!(controls.check_capture_schedule_bytes(1 << 20).is_err());
        assert_eq!(initial_state_width(&req, false).unwrap(), 2);
    }

    #[test]
    fn turbo_and_control_descriptors_resolve_every_actual_factory() {
        assert_eq!(
            resolve(&GenerationRequest::default()).unwrap().solver(),
            Solver::Euler
        );
        for descriptor in [
            crate::model::descriptor(),
            crate::model_control::descriptor(),
        ] {
            for name in descriptor.capabilities.samplers {
                let req = GenerationRequest {
                    sampler: Some(name.into()),
                    steps: Some(4),
                    ..Default::default()
                };
                let plan = resolve_capture(&req, false, 1 << 20).unwrap();
                assert_eq!(plan.solver().name(), name);
                assert_eq!(
                    FlowSamplingRecipe::from_token(&plan.recipe().canonical_token()).unwrap(),
                    *plan.recipe()
                );
                assert_eq!(
                    plan.memory(16, 2, 4).unwrap().callbacks,
                    plan.epochs().unwrap().len()
                );
            }
        }
        let mut req = GenerationRequest::default();
        assert_eq!(initial_state_width(&req, false).unwrap(), 2);
        assert_eq!(initial_state_width(&req, true).unwrap(), 2); // actual absent strength => start0
        req.strength = Some(0.5);
        assert_eq!(initial_state_width(&req, true).unwrap(), 4);
        let full = resolve_capture(&req, false, 1 << 20).unwrap();
        let tail = resolve_capture(&req, true, 1 << 20).unwrap();
        let start = crate::pipeline::init_time_step(full.steps(), req.strength);
        assert_eq!(tail.sigmas(), &full.sigmas()[start..]);
        req.strength = Some(1.0);
        assert!(resolve_capture(&req, true, 1 << 20)
            .unwrap()
            .epochs()
            .unwrap()
            .is_empty());
    }
}
use mlx_gen::sampler::flow_sampling::{
    FlowSamplingDefaults, FlowSamplingRecipe, ResolvedFlowControls, ResolvedFlowSampling,
};
use mlx_gen::{FlowMatchEuler, GenerationRequest, Result, Solver, TimestepConvention};

pub(crate) fn defaults() -> FlowSamplingDefaults {
    FlowSamplingDefaults {
        solver: Solver::Euler,
        steps: crate::model::DEFAULT_STEPS,
        guidance: None,
        true_cfg_overrides: false,
        default_alias: false,
    }
}

pub fn resolve_full(req: &GenerationRequest) -> Result<ResolvedFlowSampling> {
    let steps = req.steps.unwrap_or(crate::model::DEFAULT_STEPS) as usize;
    let native = FlowMatchEuler::for_static_shift(steps, crate::model::SCHEDULE_SHIFT);
    ResolvedFlowSampling::resolve(
        req,
        defaults(),
        &native.sigmas,
        crate::model::SCHEDULE_SHIFT.ln(),
        TimestepConvention::OneMinusSigma,
    )
}
/// O(1) admission/key controls. Exact effective N is not fabricated from requested steps.
pub fn resolve(req: &GenerationRequest) -> Result<ResolvedFlowControls> {
    ResolvedFlowControls::resolve(req, defaults(), TimestepConvention::OneMinusSigma)
}
pub fn resolve_with_init(req: &GenerationRequest, _has_init: bool) -> Result<ResolvedFlowControls> {
    resolve(req)
}
pub fn resolve_capture(
    req: &GenerationRequest,
    has_init: bool,
    host_budget_bytes: u64,
) -> Result<ResolvedFlowSampling> {
    resolve(req)?.check_capture_schedule_bytes(host_budget_bytes)?;
    resolve_actual_with_init(req, has_init)
}
/// Post-admission/bounded-capture exact strength/PiD window using unchanged numeric builders.
pub fn resolve_actual_with_init(
    req: &GenerationRequest,
    has_init: bool,
) -> Result<ResolvedFlowSampling> {
    let mut plan = resolve_full(req)?;
    let reference = crate::pipeline::resolve_reference(req, crate::model::MODEL_ID)?;
    let strength = reference
        .and_then(|(_, strength)| strength)
        .or(req.strength);
    let start = if has_init {
        crate::pipeline::init_time_step(plan.steps(), strength)
    } else {
        0
    };
    let (_, keep) = mlx_gen_pid::flow_capture_for_request(req, plan.sigmas(), start);
    // Do not clamp an init beyond an effective re-strided/deduplicated schedule: actual
    // denoise_batch's seed blend uses get(start_step) and refuses that request.
    plan.window(start, keep)?;
    Ok(plan)
}
/// Actual Turbo prior is BF16; an active clean/noise blend uses F32 scalar coefficients and
/// promotes that initial state. No per-step narrowing is imposed on later solver states.
pub fn initial_state_width(req: &GenerationRequest, has_init: bool) -> Result<u64> {
    let reference = crate::pipeline::resolve_reference(req, crate::model::MODEL_ID)?;
    let strength = reference.and_then(|(_, s)| s).or(req.strength);
    let steps = req.steps.unwrap_or(crate::model::DEFAULT_STEPS) as usize;
    Ok(
        if has_init && crate::pipeline::init_time_step(steps, strength) > 0 {
            4
        } else {
            2
        },
    )
}
pub fn representative_recipes() -> Vec<FlowSamplingRecipe> {
    crate::model::descriptor()
        .capabilities
        .samplers
        .into_iter()
        .map(|sampler| {
            resolve(&GenerationRequest {
                sampler: Some(sampler.into()),
                steps: Some(
                    Solver::from_name(sampler)
                        .expect("advertised solver")
                        .minimum_capture_transitions() as u32,
                ),
                ..Default::default()
            })
            .expect("advertised Turbo factory")
            .recipe()
            .clone()
        })
        .collect()
}
