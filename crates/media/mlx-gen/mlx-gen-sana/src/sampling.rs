//! Provider-owned SANA FLOW projection and separate SCM/TrigFlow replay. No SCM->Solver mapping.

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn largest_legal_metadata_requests_do_not_materialize_flow_or_scm_steps() {
        let req = GenerationRequest {
            steps: Some(u32::MAX),
            count: u32::MAX,
            ..Default::default()
        };
        let base = resolve_base(&req).unwrap();
        assert_eq!(base.steps(), u32::MAX as usize);
        assert!(base.check_capture_schedule_bytes(1 << 20).is_err());
        let sprint = resolve_sprint(&req).unwrap();
        assert_eq!(sprint.steps(), u32::MAX as usize);
        assert_eq!(sprint.effective_transitions(), u32::MAX as usize);
        assert!(sprint.host_schedule_bytes().unwrap() >= (u64::from(u32::MAX) + 1) * 4);
    }

    #[test]
    fn scm_captured_recipe_uses_typed_init_and_raw_strength_precedence_without_media() {
        let recipe = ScmSamplingRecipe::try_from(SprintWire {
            version: 1,
            steps: u32::MAX,
            guidance: 4.5,
            strength: Some(0.2),
            reference_strength: Some(0.75),
        })
        .unwrap();
        let text = ResolvedSprintControls::from_recipe(&recipe, false).unwrap();
        let reference = ResolvedSprintControls::from_recipe(&recipe, true).unwrap();
        assert_eq!(text.start_step(), 0);
        assert_eq!(reference.recipe(), &recipe);
        assert_eq!(
            reference.start_step(),
            mlx_gen::img2img::init_time_step(u32::MAX as usize, Some(0.75))
        );
        assert_eq!(reference.steps(), u32::MAX as usize);
        assert!(reference.effective_transitions() < text.effective_transitions());
    }

    #[test]
    fn base_descriptor_and_true_cfg_use_the_shared_actual_factory() {
        let mut req = GenerationRequest::default();
        req.guidance = Some(1.0);
        req.true_cfg = Some(7.0);
        assert_eq!(resolve_base(&req).unwrap().effective_guidance(), Some(7.0));
        for name in crate::model::descriptor().capabilities.samplers {
            req.sampler = Some(name.into());
            let plan = resolve_base_capture(&req, false, 1 << 20).unwrap();
            assert_eq!(
                plan.solver().name(),
                if name == "default" { "euler" } else { name }
            );
            assert_eq!(
                FlowSamplingRecipe::from_token(&plan.recipe().canonical_token()).unwrap(),
                *plan.recipe()
            );
            assert_eq!(
                plan.memory(16, 4, 4).unwrap().callbacks,
                plan.epochs().unwrap().len()
            );
        }
        let req = GenerationRequest::default();
        let full = resolve_base_capture(&req, false, 1 << 20).unwrap();
        let default_init = resolve_base_capture(&req, true, 1 << 20).unwrap();
        let start = mlx_gen::img2img::init_time_step(
            full.steps(),
            Some(mlx_gen::img2img::DEFAULT_IMG2IMG_STRENGTH),
        );
        assert_eq!(default_init.sigmas(), &full.sigmas()[start..]);
    }
    #[test]
    fn sprint_descriptor_replay_is_dedicated_scm_with_real_transition_predicates() {
        for sampler in crate::model::sprint_descriptor().capabilities.samplers {
            for scheduler in crate::model::sprint_descriptor().capabilities.schedulers {
                let req = GenerationRequest {
                    sampler: Some(sampler.into()),
                    scheduler: Some(scheduler.into()),
                    ..Default::default()
                };
                let plan = resolve_sprint_capture(&req, false, 1 << 20).unwrap();
                let epochs = plan.epochs();
                assert_eq!(
                    plan.scheduler().timesteps,
                    ScmScheduler::new(crate::pipeline::SPRINT_DEFAULT_STEPS).timesteps
                );
                assert_eq!(plan.guidance(), crate::pipeline::SPRINT_DEFAULT_GUIDANCE);
                assert!(epochs.first().unwrap().renoise);
                assert!(!epochs.last().unwrap().renoise);
                assert!(!epochs.first().unwrap().previous_denoised);
                assert!(epochs.last().unwrap().previous_denoised);
                assert_eq!(
                    ScmSamplingRecipe::from_token(&plan.recipe().canonical_token()).unwrap(),
                    *plan.recipe()
                );
            }
        }
        let req = GenerationRequest {
            sampler: Some("er_sde".into()),
            ..Default::default()
        };
        assert!(resolve_sprint(&req).is_err());
        let req = GenerationRequest {
            steps: Some(1),
            ..Default::default()
        };
        assert!(
            !resolve_sprint_capture(&req, false, 1 << 20)
                .unwrap()
                .epochs()[0]
                .renoise
        );
    }
}
use crate::scm::ScmScheduler;
use mlx_gen::sampler::flow_sampling::{
    decode_recipe_token, encode_recipe_token, FlowSamplingDefaults, FlowSamplingRecipe,
    ResolvedFlowControls, ResolvedFlowSampling,
};
use mlx_gen::{
    Conditioning, Error, FlowMatchEuler, GenerationRequest, Result, Solver, TimestepConvention,
};
use serde::{Deserialize, Serialize};

pub fn resolve_base_full(req: &GenerationRequest) -> Result<ResolvedFlowSampling> {
    let steps = req.steps.unwrap_or(crate::pipeline::DEFAULT_STEPS as u32) as usize;
    let native = FlowMatchEuler::for_static_shift(steps, crate::pipeline::SCHEDULE_SHIFT);
    ResolvedFlowSampling::resolve(
        req,
        base_defaults(),
        &native.sigmas,
        crate::pipeline::SCHEDULE_SHIFT.ln(),
        TimestepConvention::Sigma,
    )
}
/// O(1) admission/key controls; no user-N scheduler/epoch allocation.
pub(crate) fn base_defaults() -> FlowSamplingDefaults {
    FlowSamplingDefaults {
        solver: Solver::Euler,
        steps: crate::pipeline::DEFAULT_STEPS as u32,
        guidance: Some(crate::pipeline::DEFAULT_GUIDANCE),
        true_cfg_overrides: true,
        default_alias: true,
    }
}

pub fn resolve_base(req: &GenerationRequest) -> Result<ResolvedFlowControls> {
    ResolvedFlowControls::resolve(req, base_defaults(), TimestepConvention::Sigma)
}
pub fn resolve_base_with_init(
    req: &GenerationRequest,
    _has_init: bool,
) -> Result<ResolvedFlowControls> {
    resolve_base(req)
}
pub fn resolve_base_capture(
    req: &GenerationRequest,
    has_init: bool,
    host_budget_bytes: u64,
) -> Result<ResolvedFlowSampling> {
    resolve_base(req)?.check_capture_schedule_bytes(host_budget_bytes)?;
    resolve_base_actual_with_init(req, has_init)
}
/// Post-admission/bounded-capture exact schedule and strength tail.
pub fn resolve_base_actual_with_init(
    req: &GenerationRequest,
    has_init: bool,
) -> Result<ResolvedFlowSampling> {
    let mut plan = resolve_base_full(req)?;
    let reference = crate::model::resolve_reference(req, crate::model::MODEL_ID)?;
    let strength = reference
        .map(|(_, strength)| strength)
        .unwrap_or_else(|| mlx_gen::img2img::resolve_strength(None, req.strength));
    let start = if has_init {
        mlx_gen::img2img::init_time_step(plan.steps(), Some(strength))
    } else {
        0
    };
    let end = plan.sigmas().len();
    // Actual seed blending refuses an init index beyond the effective schedule; no metadata clamp.
    plan.window(start, end)?;
    Ok(plan)
}
pub fn representative_recipes() -> Vec<FlowSamplingRecipe> {
    crate::model::descriptor()
        .capabilities
        .samplers
        .into_iter()
        .map(|sampler| {
            resolve_base(&GenerationRequest {
                sampler: Some(sampler.into()),
                steps: Some(
                    mlx_gen::sampler::canonical_solver(Some(sampler), Solver::Euler)
                        .minimum_capture_transitions() as u32,
                ),
                ..Default::default()
            })
            .expect("advertised SANA factory")
            .recipe()
            .clone()
        })
        .collect()
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct SprintWire {
    version: u32,
    steps: u32,
    guidance: f32,
    #[serde(deserialize_with = "required_option")]
    strength: Option<f32>,
    #[serde(deserialize_with = "required_option")]
    reference_strength: Option<f32>,
}
fn required_option<'de, D: serde::Deserializer<'de>>(
    d: D,
) -> std::result::Result<Option<f32>, D::Error> {
    Option::deserialize(d)
}
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(try_from = "SprintWire", into = "SprintWire")]
pub struct ScmSamplingRecipe(SprintWire);
impl From<ScmSamplingRecipe> for SprintWire {
    fn from(r: ScmSamplingRecipe) -> Self {
        r.0
    }
}
impl TryFrom<SprintWire> for ScmSamplingRecipe {
    type Error = String;
    fn try_from(w: SprintWire) -> std::result::Result<Self, String> {
        if w.version != 1
            || w.steps == 0
            || !w.guidance.is_finite()
            || [w.strength, w.reference_strength]
                .into_iter()
                .flatten()
                .any(|v| !v.is_finite())
        {
            return Err("invalid SCM replay version/controls".into());
        }
        Ok(Self(w))
    }
}
impl ScmSamplingRecipe {
    pub fn canonical_token(&self) -> String {
        encode_recipe_token("scm1-", self)
    }
    pub fn from_token(token: &str) -> Result<Self> {
        decode_recipe_token("scm1-", token)
    }
    pub fn apply_to(&self, req: &mut GenerationRequest) -> Result<()> {
        req.sampler = Some("default".into());
        req.scheduler = Some("default".into());
        req.steps = Some(self.0.steps);
        req.guidance = Some(self.0.guidance);
        req.true_cfg = None;
        req.guidance_method = None;
        req.strength = self.0.strength;
        for c in &mut req.conditioning {
            if let Conditioning::Reference { strength, .. } = c {
                *strength = self.0.reference_strength;
            }
        }
        Ok(())
    }
    pub fn to_request(&self) -> GenerationRequest {
        let mut req = GenerationRequest::default();
        self.apply_to(&mut req).expect("validated SCM recipe");
        req.strength = self.0.reference_strength.or(self.0.strength);
        req
    }
}
/// O(1) SCM controls/window; no schedule allocation before admission.
#[derive(Clone, Debug)]
pub struct ResolvedSprintControls {
    recipe: ScmSamplingRecipe,
    start: usize,
}
impl ResolvedSprintControls {
    /// Metadata-only SCM replay under an explicit init role; no request/media/schedule allocation.
    pub fn from_recipe(recipe: &ScmSamplingRecipe, has_init: bool) -> Result<Self> {
        let n = recipe.0.steps as usize;
        let strength =
            mlx_gen::img2img::resolve_strength(recipe.0.reference_strength, recipe.0.strength);
        let start = if has_init {
            mlx_gen::img2img::init_time_step(n, Some(strength))
        } else {
            0
        };
        if start > n {
            return Err(Error::Unsupported(
                "SCM init index exceeds actual schedule length".into(),
            ));
        }
        Ok(Self {
            recipe: recipe.clone(),
            start,
        })
    }

    pub fn recipe(&self) -> &ScmSamplingRecipe {
        &self.recipe
    }
    pub fn steps(&self) -> usize {
        self.recipe.0.steps as usize
    }
    pub fn start_step(&self) -> usize {
        self.start
    }
    pub fn guidance(&self) -> f32 {
        self.recipe.0.guidance
    }
    pub fn allocation_class(&self) -> &'static str {
        "scm1-trigflow"
    }
    pub fn effective_transitions(&self) -> usize {
        self.steps().saturating_sub(self.start)
    }
    pub fn host_schedule_bytes(&self) -> Result<u64> {
        (self.steps() as u64)
            .checked_add(1)
            .and_then(|n| n.checked_mul(4))
            .ok_or_else(|| Error::Unsupported("SCM schedule bytes overflow".into()))
    }
}

#[derive(Clone, Debug)]
pub struct ResolvedSprintSampling {
    recipe: ScmSamplingRecipe,
    scheduler: ScmScheduler,
    start: usize,
}
#[derive(Clone, Debug, PartialEq)]
pub struct ScmEpoch {
    pub step: usize,
    pub angle: f32,
    pub next_angle: f32,
    pub timestep: f32,
    pub input_scale: f32,
    pub renoise: bool,
    /// Previous pred_x0 remains in `denoised` across this callback after a nonterminal renoise.
    pub previous_denoised: bool,
}
impl ResolvedSprintSampling {
    pub fn recipe(&self) -> &ScmSamplingRecipe {
        &self.recipe
    }
    pub fn scheduler(&self) -> &ScmScheduler {
        &self.scheduler
    }
    pub fn start_step(&self) -> usize {
        self.start
    }
    pub fn guidance(&self) -> f32 {
        self.recipe.0.guidance
    }
    pub fn allocation_class(&self) -> &'static str {
        "scm1-trigflow"
    }
    pub fn epochs(&self) -> Vec<ScmEpoch> {
        let n = self.scheduler.num_steps();
        (self.start..n)
            .map(|step| ScmEpoch {
                step,
                angle: self.scheduler.timesteps[step],
                next_angle: self.scheduler.timesteps[step + 1],
                timestep: self.scheduler.scm_timestep(step),
                input_scale: self.scheduler.input_scale(step),
                renoise: !self.scheduler.is_single_step() && step + 1 < n,
                previous_denoised: step > self.start,
            })
            .collect()
    }
}
pub(crate) fn sprint_scheduler(steps: usize) -> ScmScheduler {
    ScmScheduler::new(steps)
}

pub fn resolve_sprint(req: &GenerationRequest) -> Result<ResolvedSprintControls> {
    let has_init = crate::model::resolve_reference(req, crate::model::SPRINT_MODEL_ID)?.is_some();
    resolve_sprint_with_init(req, has_init)
}
pub fn resolve_sprint_with_init(
    req: &GenerationRequest,
    has_init: bool,
) -> Result<ResolvedSprintControls> {
    if req.sampler.as_deref().is_some_and(|s| s != "default")
        || req.scheduler.as_deref().is_some_and(|s| s != "default")
        || req.guidance_method.is_some()
        || req.true_cfg.is_some()
        || req.use_pid
    {
        return Err(Error::Unsupported(
            "SANA Sprint serves only default SCM with embedded guidance".into(),
        ));
    }
    crate::model::resolve_reference(req, crate::model::SPRINT_MODEL_ID)?;
    let reference_strength = req.conditioning.iter().find_map(|c| match c {
        Conditioning::Reference { strength, .. } => *strength,
        _ => None,
    });
    let recipe = ScmSamplingRecipe::try_from(SprintWire {
        version: 1,
        steps: req
            .steps
            .unwrap_or(crate::pipeline::SPRINT_DEFAULT_STEPS as u32),
        guidance: req
            .guidance
            .unwrap_or(crate::pipeline::SPRINT_DEFAULT_GUIDANCE),
        strength: req.strength,
        reference_strength,
    })
    .map_err(Error::Unsupported)?;
    ResolvedSprintControls::from_recipe(&recipe, has_init)
}
pub fn resolve_sprint_actual_with_init(
    req: &GenerationRequest,
    has_init: bool,
) -> Result<ResolvedSprintSampling> {
    let controls = resolve_sprint_with_init(req, has_init)?;
    let scheduler = sprint_scheduler(controls.steps());
    Ok(ResolvedSprintSampling {
        recipe: controls.recipe,
        scheduler,
        start: controls.start,
    })
}
pub fn resolve_sprint_capture(
    req: &GenerationRequest,
    has_init: bool,
    host_budget_bytes: u64,
) -> Result<ResolvedSprintSampling> {
    let controls = resolve_sprint_with_init(req, has_init)?;
    let schedule_bytes = controls.host_schedule_bytes()?;
    let epoch_bytes = (controls.steps() as u64)
        .checked_mul(std::mem::size_of::<ScmEpoch>() as u64)
        .and_then(|n| n.checked_add(schedule_bytes))
        .ok_or_else(|| Error::Unsupported("SCM capture bytes overflow".into()))?;
    if epoch_bytes > host_budget_bytes {
        return Err(Error::Unsupported(
            "SCM capture exceeds explicit host byte budget".into(),
        ));
    }
    resolve_sprint_actual_with_init(req, has_init)
}
