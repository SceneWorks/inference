//! Anima-Turbo's actual FLOW projection. Base/Aesthetic remain on their existing CFG path.

use mlx_gen::gen_core::sampling::TimestepConvention;
use mlx_gen::sampler::flow_sampling::{
    FlowSamplingDefaults, FlowSamplingRecipe, ResolvedFlowControls, ResolvedFlowSampling,
};
use mlx_gen::{GenerationRequest, Result, Solver};

/// O(1) admission/key controls; no schedule or epoch vector is constructed.
pub fn resolve(req: &GenerationRequest) -> Result<ResolvedFlowControls> {
    ResolvedFlowControls::resolve(req, defaults(), TimestepConvention::Sigma)
}
pub fn resolve_capture(
    req: &GenerationRequest,
    host_budget_bytes: u64,
) -> Result<ResolvedFlowSampling> {
    resolve(req)?.check_capture_schedule_bytes(host_budget_bytes)?;
    resolve_actual(req)
}
/// Post-admission exact schedule, preserving the existing numeric builder.
pub fn resolve_actual(req: &GenerationRequest) -> Result<ResolvedFlowSampling> {
    let steps = req
        .steps
        .unwrap_or(crate::config::Variant::Turbo.default_steps()) as usize;
    let native = crate::pipeline::anima_sigmas(steps);
    ResolvedFlowSampling::resolve(
        req,
        defaults(),
        &native,
        crate::config::SIGMA_SHIFT.ln(),
        TimestepConvention::Sigma,
    )
}
pub(crate) fn defaults() -> FlowSamplingDefaults {
    FlowSamplingDefaults {
        solver: Solver::ErSde,
        steps: crate::config::Variant::Turbo.default_steps(),
        guidance: None,
        true_cfg_overrides: false,
        default_alias: false,
    }
}
pub fn representative_recipes() -> Vec<FlowSamplingRecipe> {
    crate::model::descriptor_turbo()
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
        let ordinary = resolve(&GenerationRequest::default()).unwrap();
        assert_eq!(
            controls.memory_envelope(16, 4, 4).unwrap().callbacks,
            ordinary.memory_envelope(16, 4, 4).unwrap().callbacks
        );
    }

    #[test]
    fn turbo_descriptor_drives_all_real_factory_recipes() {
        let default = resolve(&GenerationRequest::default()).unwrap();
        assert_eq!(default.solver(), Solver::ErSde);
        assert_eq!(
            default.steps(),
            crate::config::Variant::Turbo.default_steps() as usize
        );
        assert_eq!(default.effective_guidance(), None);
        for name in crate::model::descriptor_turbo().capabilities.samplers {
            let req = GenerationRequest {
                sampler: Some(name.into()),
                steps: Some(4),
                ..Default::default()
            };
            let plan = resolve_capture(&req, 1 << 20).unwrap();
            assert_eq!(plan.solver().name(), name);
            let recipe = FlowSamplingRecipe::from_token(&plan.recipe().canonical_token()).unwrap();
            assert_eq!(
                resolve(&recipe.to_request()).unwrap().allocation_class(),
                plan.allocation_class()
            );
            assert_eq!(
                plan.memory(16, 4, 4).unwrap().callbacks,
                plan.epochs().unwrap().len()
            );
        }
        for name in crate::model::descriptor_turbo().capabilities.schedulers {
            let req = GenerationRequest {
                scheduler: Some(name.into()),
                steps: Some(4),
                ..Default::default()
            };
            assert_eq!(
                resolve_capture(&req, 1 << 20).unwrap().sigmas(),
                crate::pipeline::anima_schedule(Some(name), 4)
            );
        }
    }
}
