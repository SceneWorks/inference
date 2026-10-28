//! SDXL's request-to-factory boundary. Resolution is host-only: no MLX arrays, weights or RNG.
//!
//! Recipes replay sampling controls, not images/load policy. Schedule/lifetime facts describe
//! logical owners, NOT physical storage/donation or equivalence of whole-phase anchor residues.
use mlx_gen::gen_core::sampling::{
    base_supports_cfgpp, lcm_style_origin_timesteps, lcm_style_timestep_index,
    lcm_transition_coeffs, tcd_transition_coeffs, vp_capture_plan, LcmPolicy, LightningPolicy,
    SamplerPolicy, StepCoeffs, TcdPolicy, LCM_TIMESTEP_SCALING,
};
use mlx_gen::gen_core::CfgBatching;
use mlx_gen::{
    schedule_sigmas, AlphaSchedule, Conditioning, DiscreteModelSampling, Error, GenerationRequest,
    Result, Scheduler, Solver,
};
use serde::{Deserialize, Serialize};
use serde_json::Value;

/// Existing diffusers LCM/TCD original-inference-step selection, shared by facts and factory.
const ORIGINAL_STEPS: usize = 50;
/// The existing SDXL Hyper factory is deterministic regardless of the generic request eta.
const HYPER_ETA: f32 = 0.0;

/// The actual SDXL factory, not the larger shared solver registry.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum SamplingRoute {
    Ancestral,
    Lcm,
    Lightning,
    Hyper,
    Curated(Solver),
    CfgPp(Solver),
}

impl SamplingRoute {
    pub fn is_acceleration(self) -> bool {
        matches!(self, Self::Lcm | Self::Lightning | Self::Hyper)
    }

    pub fn is_curated(self) -> bool {
        matches!(self, Self::Curated(_) | Self::CfgPp(_))
    }
}

mod source_context;
pub use source_context::{
    ControlSources, IndexedImageSourceFacts, SamplingSourceFacts, SamplingSourceFactsSeed,
    SamplingSourceView, SamplingTextSourceFacts, SourceRequestBudget,
};

/// Validated, versioned replay of the controls SDXL actually consumes. Private fields prevent
/// accepting caller-claimed factory/class identities. JSON is exposed through `to_value` /
/// `from_value` so consumers can embed this in serde evidence without a generic-crate SDXL edge.
/// Geometry, count, seed, conditioning payloads and loaded components belong to the enclosing case.
#[derive(Clone, Debug, PartialEq, Serialize)]
pub struct SamplingRecipe {
    version: u32,
    sampler: String,
    scheduler: String,
    steps: u32,
    guidance: f32,
    guidance_method: String,
    #[serde(with = "batching_serde")]
    cfg_batching: CfgBatching,
    strength: Option<f32>,
    reference_strength: Option<f32>,
    use_pid: bool,
    pid_capture_sigma: Option<f32>,
}

/// Payload-free conditioning roles supplied by a capture case or an actual request projection.
#[derive(
    Clone, Copy, Debug, Default, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize,
)]
#[serde(deny_unknown_fields)]
pub struct SamplingWorkFacts {
    /// TRUE Conditioning::Reference roles, NOT MemoryGeometry's aggregate image-carrier count.
    pub reference_count: usize,
    pub mask_count: usize,
    pub control_count: usize,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub reference_image: Option<ReferenceImageFacts>,
}

/// Exact unvalidated source facts from the ACTUAL single Reference. A digest binds content but
/// cannot recover these fields; never substitute output/CLIP/crop dimensions or width*height*3.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ReferenceImageFacts {
    pub width: u32,
    pub height: u32,
    pub buffer_bytes: u64,
}
impl ReferenceImageFacts {
    pub fn from_image(image: &mlx_gen::media::Image) -> Self {
        Self {
            width: image.width,
            height: image.height,
            buffer_bytes: image.pixels.len() as u64,
        }
    }
}

impl SamplingWorkFacts {
    /// Bounded scalar projection only. Keep all existing native validation chronology: copying
    /// malformed dimensions/span here does not validate, resize, allocate media or discard errors.
    pub fn from_request(request: &GenerationRequest) -> Self {
        Self {
            reference_count: request
                .conditioning
                .iter()
                .filter(|c| matches!(c, Conditioning::Reference { .. }))
                .count(),
            mask_count: request
                .conditioning
                .iter()
                .filter(|c| matches!(c, Conditioning::Mask { .. }))
                .count(),
            control_count: request
                .conditioning
                .iter()
                .filter(|c| matches!(c, Conditioning::Control { .. }))
                .count(),
            reference_image: request.conditioning.iter().find_map(|c| match c {
                Conditioning::Reference { image, .. } => {
                    Some(ReferenceImageFacts::from_image(image))
                }
                _ => None,
            }),
        }
    }
    pub fn to_value(&self) -> Value {
        serde_json::to_value(self).expect("scalar work facts serialize")
    }
    /// Separate capture work sidecar, not a change to the sampling recipe/version or generic DTO.
    pub fn from_value(value: Value) -> Result<Self> {
        serde_json::from_value(value)
            .map_err(|error| invalid(format!("invalid conditioning work facts: {error}")))
    }
    /// Used by replay at its existing source-binding boundary; never replace the full content SHA.
    pub fn validate_actual(&self, request: &GenerationRequest) -> Result<()> {
        if *self != Self::from_request(request) {
            return Err(invalid(
                "captured conditioning work facts differ from actual request",
            ));
        }
        Ok(())
    }
}

#[derive(Clone, Debug)]
pub struct ResolvedSampling {
    recipe: SamplingRecipe,
    route: SamplingRoute,
    scheduler: Option<Scheduler>,
}

fn invalid(message: impl Into<String>) -> Error {
    Error::Msg(format!("sdxl sampling: {}", message.into()))
}

impl ResolvedSampling {
    /// Normalize the actual post-caller-policy request. This is NOT checkpoint/UI policy and does
    /// not replace provider validation of images, load compatibility or other unsupported controls.
    pub fn resolve(req: &GenerationRequest) -> Result<Self> {
        let mut references = req.conditioning.iter().filter_map(|c| match c {
            Conditioning::Reference { strength, .. } => Some(*strength),
            _ => None,
        });
        let reference_strength = references.next().flatten();
        let work = SamplingWorkFacts::from_request(req);
        Self::resolve_controls(
            req.sampler.as_deref(),
            req.scheduler.as_deref(),
            req.steps,
            req.guidance,
            req.guidance_method.as_deref(),
            crate::model::resolve_cfg_batching(req),
            req.strength,
            reference_strength,
            req.use_pid,
            req.pid_capture_sigma,
            work,
        )
    }

    /// Metadata-only captured recipe resolution. The work facts are authoritative typed roles,
    /// not fabricated Image/GenerationRequest carriers. Load-dependent init/IP policy stays with
    /// the provider; use effective_init_strength only after that policy identifies a true VAE init.
    pub fn resolve_recipe(recipe: &SamplingRecipe, work: SamplingWorkFacts) -> Result<Self> {
        let resolved = Self::resolve_controls(
            Some(&recipe.sampler),
            Some(&recipe.scheduler),
            Some(recipe.steps),
            Some(recipe.guidance),
            Some(&recipe.guidance_method),
            recipe.cfg_batching,
            recipe.strength,
            recipe.reference_strength,
            recipe.use_pid,
            recipe.pid_capture_sigma,
            work,
        )?;
        if &resolved.recipe != recipe {
            return Err(invalid("noncanonical recipe controls"));
        }
        Ok(resolved)
    }

    #[allow(clippy::too_many_arguments)]
    fn resolve_controls(
        sampler_name: Option<&str>,
        schedule_name: Option<&str>,
        requested_steps: Option<u32>,
        requested_guidance: Option<f32>,
        guidance_method: Option<&str>,
        cfg_batching: CfgBatching,
        strength: Option<f32>,
        reference_strength: Option<f32>,
        use_pid: bool,
        pid_capture_sigma: Option<f32>,
        work: SamplingWorkFacts,
    ) -> Result<Self> {
        let sampler = sampler_name.unwrap_or("euler_ancestral");
        let scheduler_name = schedule_name.unwrap_or("discrete");
        let scheduler = if scheduler_name == "discrete" {
            None
        } else {
            Some(Scheduler::from_name(scheduler_name).ok_or_else(|| invalid("unknown scheduler"))?)
        };
        let ordinary = (crate::model::DEFAULT_STEPS, crate::model::DEFAULT_GUIDANCE);
        let (base, defaults) = match sampler {
            "euler_ancestral" => (SamplingRoute::Ancestral, ordinary),
            "lcm" => (SamplingRoute::Lcm, (8, 1.0)),
            "lightning" => (SamplingRoute::Lightning, (4, 1.0)),
            "hyper" => (SamplingRoute::Hyper, (4, 1.0)),
            "euler" | "heun" | "dpmpp_2m" | "dpmpp_sde" | "uni_pc" | "ddim" => (
                SamplingRoute::Curated(Solver::from_name(sampler).expect("served solver")),
                ordinary,
            ),
            _ => return Err(invalid(format!("unsupported sampler {sampler:?}"))),
        };
        if base.is_acceleration() && scheduler.is_some() {
            return Err(invalid(format!(
                "the {sampler:?} acceleration sampler uses its own distilled schedule and cannot honor the {scheduler_name:?} scheduler"
            )));
        }
        let steps = requested_steps.unwrap_or(defaults.0);
        let guidance = requested_guidance.unwrap_or(defaults.1);
        if steps == 0 || !guidance.is_finite() {
            return Err(invalid("positive steps and finite guidance required"));
        }
        let method = guidance_method.unwrap_or("cfg");
        if !matches!(method, "cfg" | "cfg_pp") {
            return Err(invalid("unsupported guidance method"));
        }
        let mut route = if base == SamplingRoute::Ancestral && scheduler.is_some() {
            SamplingRoute::Curated(Solver::EulerAncestral)
        } else {
            base
        };
        if let SamplingRoute::Curated(solver) = route {
            if method == "cfg_pp" && guidance > 1.0 && base_supports_cfgpp(solver) {
                route = SamplingRoute::CfgPp(solver);
            }
        }
        let has_reference = work.reference_count != 0;
        let masks = work.mask_count;
        let has_control = work.control_count != 0;
        if base.is_acceleration() && has_reference {
            return Err(invalid(format!("the {sampler:?} acceleration sampler is txt2img-only (no img2img reference) in this build")));
        }
        if masks > 1 {
            return Err(invalid("multiple inpaint masks are not supported"));
        }
        if masks != 0 {
            if base.is_acceleration() {
                return Err(invalid(
                    "inpaint masks are not supported with the acceleration samplers",
                ));
            }
            if route.is_curated() {
                return Err(invalid(
                    "curated samplers/schedulers are not supported with an inpaint Mask",
                ));
            }
            if !has_reference {
                return Err(invalid(
                    "inpaint requires an init image (a Reference) alongside the Mask",
                ));
            }
            if has_control {
                return Err(invalid(
                    "combining a ControlNet (Control) with an inpaint Mask is not supported",
                ));
            }
        }
        if has_control {
            crate::model::validate_control_acceleration(sampler, requested_guidance)?;
        }
        if use_pid && pid_capture_sigma.is_some() && (base.is_acceleration() || masks != 0) {
            return Err(invalid("pid_capture_sigma is not supported on the SDXL few-step accel or inpaint mask-blend path"));
        }
        if work.reference_count > 1 {
            return Err(invalid("multiple reference images are not supported"));
        }
        for v in [strength, reference_strength, pid_capture_sigma]
            .into_iter()
            .flatten()
        {
            if !v.is_finite() {
                return Err(invalid("nonfinite replay control"));
            }
        }
        let scheduler = if route.is_curated() {
            Some(scheduler.unwrap_or(Scheduler::Normal))
        } else {
            None
        };
        Ok(Self {
            route,
            scheduler,
            recipe: SamplingRecipe {
                version: 1,
                sampler: sampler.into(),
                scheduler: scheduler.map(Scheduler::name).unwrap_or("discrete").into(),
                steps,
                guidance,
                // Preserve an incompatible/off CFG++ request as a reproducible fallback.
                guidance_method: method.into(),
                cfg_batching,
                strength,
                reference_strength,
                use_pid,
                pid_capture_sigma,
            },
        })
    }

    /// Fail closed at a dynamic-pricing boundary rather than inventing a default production run.
    pub fn for_pricing(generation: Option<&GenerationRequest>) -> Result<Self> {
        Self::resolve(generation.ok_or_else(|| invalid("missing generation context"))?)
    }

    pub fn recipe(&self) -> &SamplingRecipe {
        &self.recipe
    }
    pub fn route(&self) -> SamplingRoute {
        self.route
    }
    pub fn sampler_name(&self) -> &str {
        &self.recipe.sampler
    }
    pub fn steps(&self) -> usize {
        self.recipe.steps as usize
    }
    pub fn guidance(&self) -> f32 {
        self.recipe.guidance
    }
    pub fn cfg_batching(&self) -> CfgBatching {
        self.recipe.cfg_batching
    }
    /// Actual U-Net rows in ONE serial output iteration, never multiplied by request.count.
    pub fn denoise_rows(&self) -> u32 {
        if self.guidance() > 1.0 {
            self.cfg_batching().guided_batch()
        } else {
            1
        }
    }
    /// Logical enclosing owner, outside the callback solver: `noise` remains in the curated
    /// count-loop scope. Physical backing aliases are a separate runtime proof.
    pub fn retains_initial_noise(&self) -> bool {
        self.route.is_curated()
    }

    /// The curated denoise adapter owns its F32 x_in across the SDXL callback, alongside the
    /// callback's F16 cast. Lightning's scale_input temporary instead dies before returning its
    /// F16 input. This is logical retention, not a promise of distinct backing when scale==1.
    pub fn retains_scaled_input_in_callback(&self) -> bool {
        self.route.is_curated()
    }

    /// Actual unclamped IP scale; distinct from img2img init strength clamping.
    pub(crate) fn effective_ip_scale(&self) -> f32 {
        crate::model::ip_reference_scale(self.recipe.reference_strength.or(self.recipe.strength))
    }

    /// Effective strength ONLY for an actual VAE init. IP image prompts/text pass false and must
    /// not acquire an img2img strength. Reuses production's exact override/default/clamp utility.
    pub fn effective_init_strength(&self, true_init: bool, mask: bool) -> Option<f32> {
        true_init.then(|| {
            crate::model::resolve_strength(
                self.recipe.reference_strength.or(self.recipe.strength),
                if mask {
                    crate::model::INPAINT_DEFAULT_STRENGTH
                } else {
                    crate::model::DEFAULT_STRENGTH
                },
            )
        })
    }

    /// Graph family, not a measured whole-phase equivalence assertion. Ordinary Euler/DDIM stay
    /// separate; CFG++ Euler/DDIM share the SAME `cfgpp_sampler_for` factory (`EulerCfgPp`).
    /// Load form/rung/precision/geometry/step count/float values are deliberately not class axes.
    pub fn memory_class(&self) -> &'static str {
        match self.route {
            SamplingRoute::Ancestral => "sdxl1-ancestral",
            SamplingRoute::Lcm => "sdxl1-lcm",
            SamplingRoute::Lightning => "sdxl1-lightning",
            SamplingRoute::Hyper => "sdxl1-hyper",
            SamplingRoute::CfgPp(Solver::Euler | Solver::Ddim) => "sdxl1-euler-cfgpp",
            SamplingRoute::CfgPp(Solver::Dpmpp2m) => "sdxl1-dpmpp2m-cfgpp",
            SamplingRoute::Curated(Solver::Euler) => "sdxl1-euler",
            SamplingRoute::Curated(Solver::EulerAncestral) => "sdxl1-curated-ancestral",
            SamplingRoute::Curated(Solver::Heun) => "sdxl1-heun",
            SamplingRoute::Curated(Solver::Dpmpp2m) => "sdxl1-dpmpp2m",
            SamplingRoute::Curated(Solver::DpmppSde) => "sdxl1-dpmppsde",
            SamplingRoute::Curated(Solver::UniPc) => "sdxl1-unipc",
            SamplingRoute::Curated(Solver::Ddim) => "sdxl1-ddim",
            _ => unreachable!("private resolved route admits only SDXL factories"),
        }
    }

    /// CFG execution is an allocation-graph distinction independent of solver family. This ID
    /// does not assert that ref/init/control inputs or phase residues share an anchor.
    pub fn allocation_class(&self) -> String {
        let cfg = if self.guidance() <= 1.0 {
            "off"
        } else if self.cfg_batching().is_batched() {
            "batched"
        } else {
            "sequential"
        };
        format!("{}-{cfg}", self.memory_class())
    }

    /// Existing curated builder, unchanged. `init_strength` is Some ONLY for a true img2img init,
    /// not an IP image prompt. The caller resolves that from its loaded components/mode.
    /// Includes the same strength-tail and supported PiD truncation as actual dispatch.
    pub fn curated_sigmas(
        &self,
        alpha: &AlphaSchedule,
        init_strength: Option<f32>,
    ) -> Result<Vec<f32>> {
        let full = self.curated_full_sigmas(alpha)?;
        let mut run = match init_strength {
            Some(strength) => {
                if !strength.is_finite() {
                    return Err(invalid("nonfinite init strength"));
                }
                strength_tail(&full, self.steps(), strength.clamp(0.0, 1.0))
            }
            None => full,
        };
        if self.recipe.use_pid {
            if let Some(plan) = vp_capture_plan(&run, self.recipe.pid_capture_sigma) {
                run.truncate(plan.keep);
            }
        }
        Ok(run)
    }

    pub(crate) fn curated_full_sigmas(&self, alpha: &AlphaSchedule) -> Result<Vec<f32>> {
        let scheduler = self
            .scheduler
            .ok_or_else(|| invalid("not a curated route"))?;
        let ms = DiscreteModelSampling::sdxl(alpha);
        Ok(schedule_sigmas(scheduler, &ms, self.steps()))
    }

    /// Bespoke schedule structure only. Exact f16-rounded timesteps/sigmas remain owned by
    /// `AncestralEuler` AFTER admission; do not replace its MLX numerics with a host approximation.
    /// `init_strength` is the already-resolved effective img2img/inpaint strength, not an IP scale.
    pub fn ancestral_window(
        &self,
        max_time: f32,
        init_strength: Option<f32>,
    ) -> Result<(usize, f32)> {
        if self.route != SamplingRoute::Ancestral {
            return Err(invalid("not a bespoke ancestral route"));
        }
        if !max_time.is_finite() || max_time < 0.0 {
            return Err(invalid("invalid ancestral max time"));
        }
        match init_strength {
            Some(strength) if strength.is_finite() => {
                Ok(crate::model::ancestral_strength_schedule(
                    self.steps(),
                    max_time,
                    strength.clamp(0.0, 1.0),
                ))
            }
            Some(_) => Err(invalid("nonfinite init strength")),
            None => Ok((self.steps(), max_time)),
        }
    }

    /// Host policy used by the existing acceleration factory. Hyper's eta is fixed ZERO, not
    /// the generic request eta. No MLX wrapper is instantiated here.
    pub fn acceleration_coefficients(&self, alpha: &AlphaSchedule) -> Result<Vec<StepCoeffs>> {
        let n = alpha.alphas_cumprod.len();
        if n == 0 {
            return Err(invalid("empty alpha schedule"));
        }
        let policy: Box<dyn SamplerPolicy> = match self.route {
            SamplingRoute::Lcm => Box::new(LcmPolicy::new(
                alpha.clone(),
                n,
                ORIGINAL_STEPS,
                self.steps(),
            )),
            SamplingRoute::Lightning => Box::new(LightningPolicy::new(alpha, n, self.steps())),
            SamplingRoute::Hyper => Box::new(TcdPolicy::new(
                alpha.clone(),
                n,
                ORIGINAL_STEPS,
                self.steps(),
                HYPER_ETA,
            )),
            _ => return Err(invalid("not an acceleration route")),
        };
        Ok((0..policy.num_steps()).map(|i| policy.coeffs(i)).collect())
    }

    /// Actual production acceleration factory. Kept separate from metadata resolution: callers
    /// invoke this only at the existing post-admission generation boundary. Numeric wrappers,
    /// model dtype, LCM original-step count and deterministic Hyper eta are unchanged.
    pub fn build_acceleration_sampler(
        &self,
        alpha: &AlphaSchedule,
        seed: u64,
    ) -> Result<Box<dyn mlx_gen::DiffusionSampler>> {
        let n = alpha.alphas_cumprod.len();
        if n == 0 {
            return Err(invalid("empty alpha schedule"));
        }
        let dtype = crate::model::DTYPE;
        Ok(match self.route {
            SamplingRoute::Lcm => Box::new(mlx_gen::LcmSampler::new(
                alpha.clone(),
                n,
                ORIGINAL_STEPS,
                self.steps(),
                dtype,
                seed,
            )),
            SamplingRoute::Lightning => Box::new(mlx_gen::LightningSampler::new(
                alpha,
                n,
                self.steps(),
                dtype,
            )),
            SamplingRoute::Hyper => Box::new(mlx_gen::TcdSampler::new(
                alpha.clone(),
                n,
                ORIGINAL_STEPS,
                self.steps(),
                HYPER_ETA,
                dtype,
                seed,
            )),
            _ => return Err(invalid("not an acceleration route")),
        })
    }

    /// Complete mechanism families without a requested-N schedule. LCM/TCD visit fixed origin
    /// intervals using the actual f64 index producer: at most 32 probes per boundary in the actual
    /// GenerationRequest u32 count domain, never an idealized integer ceil/floor substitute.
    pub fn bounded_acceleration_families(
        &self,
        alpha: &AlphaSchedule,
    ) -> Result<Vec<AccelerationFamily>> {
        use std::collections::BTreeSet;
        let n = self.steps();
        if n == 0 || u32::try_from(n).is_err() {
            return Err(invalid(
                "step metadata is not representable by the actual GenerationRequest",
            ));
        }
        let train = alpha.alphas_cumprod.len();
        if train == 0 {
            return Err(invalid("empty alpha schedule"));
        }
        let mut families = BTreeSet::new();
        if self.route == SamplingRoute::Lightning {
            // Trailing timesteps are round(v)-1, hence integers in [-1,train-1]. Clamped -1
            // uses exactly the same sigma as 0. This fixed TRAIN table covers every input-scale
            // branch without constructing the requested schedule. Update a_x=1/noise=0 is the
            // actual LightningPolicy invariant; a_out never selects a legacy update branch.
            let table = LightningPolicy::new(alpha, train, train);
            families.insert(AccelerationFamily::at(0, n, table.coeffs(0)));
            if n > 1 {
                for i in 0..table.num_steps() {
                    let coefficients = table.coeffs(i);
                    families.insert(AccelerationFamily::at(n - 1, n, coefficients));
                    if n > 2 {
                        families.insert(AccelerationFamily::at(1, n, coefficients));
                    }
                }
            }
            return Ok(families.into_iter().collect());
        }
        if !matches!(self.route, SamplingRoute::Lcm | SamplingRoute::Hyper) {
            return Err(invalid("not an acceleration route"));
        }
        let origin = lcm_style_origin_timesteps(train, ORIGINAL_STEPS);
        let mut candidates = BTreeSet::from([0, n - 1]);
        for index in 0..origin.len() {
            let start = lcm_index_lower_bound(origin.len(), n, index)?;
            let end = lcm_index_lower_bound(origin.len(), n, index + 1)?;
            if start < end {
                candidates.insert(start);
                if start + 1 < end {
                    candidates.insert(start + 1);
                }
                candidates.insert(end - 1);
            }
        }
        for step in candidates {
            let index = lcm_style_timestep_index(origin.len(), n, step)
                .ok_or_else(|| invalid("actual origin index missing"))?;
            let t = origin[index] as usize;
            let next = if step + 1 < n {
                let index = lcm_style_timestep_index(origin.len(), n, step + 1)
                    .ok_or_else(|| invalid("actual next origin index missing"))?;
                Some(origin[index] as usize)
            } else {
                None
            };
            let coefficients = match self.route {
                SamplingRoute::Lcm => lcm_transition_coeffs(alpha, t, next, LCM_TIMESTEP_SCALING),
                SamplingRoute::Hyper => tcd_transition_coeffs(alpha, t, next, HYPER_ETA),
                _ => unreachable!("resolved bounded acceleration route"),
            };
            families.insert(AccelerationFamily::at(step, n, coefficients));
        }
        Ok(families.into_iter().collect())
    }

    /// Actual acceleration transitions AFTER admission: this constructs the requested-N vector.
    /// Metadata pricing must use `bounded_acceleration_families` instead. LCM/TCD do NOT cap N.
    pub fn acceleration_epochs(&self, alpha: &AlphaSchedule) -> Result<Vec<AccelerationEpoch>> {
        let coefficients = self.acceleration_coefficients(alpha)?;
        let n = coefficients.len();
        Ok(coefficients
            .into_iter()
            .enumerate()
            .map(|(step, coefficients)| AccelerationEpoch {
                step,
                final_step: step + 1 == n,
                input_scale_alias: coefficients.c_in == 1.0,
                draws_noise: coefficients.has_noise(),
                additive_update: coefficients.uses_unity_noiseless_update(),
                coefficients,
            })
            .collect())
    }

    /// Exact logical callback epochs for an ACTUAL curated schedule (including repeated nodes and
    /// nonzero-terminal truncation). Callers must not substitute requested N for `sigmas.len()-1`.
    pub fn logical_epochs(&self, sigmas: &[f32]) -> Result<Vec<SamplingEpoch>> {
        if !self.route.is_curated() {
            return Err(invalid("curated epochs require a curated route"));
        }
        if sigmas.is_empty() || sigmas.iter().any(|s| !s.is_finite() || *s < 0.0) {
            return Err(invalid("invalid sigma schedule"));
        }
        let mut epochs = Vec::new();
        let mut history = 0;
        for (i, pair) in sigmas.windows(2).enumerate() {
            let (sigma, next) = (pair[0], pair[1]);
            let unipc = self.route == SamplingRoute::Curated(Solver::UniPc);
            let leading_terminal = sigma == 0.0; // unified::is_terminal
            epochs.push(SamplingEpoch {
                step: i,
                evaluation: Evaluation::Primary,
                sigma,
                next_sigma: next,
                history_owners: history,
                outer_owners: 0,
                denoised_outputs: if matches!(self.route, SamplingRoute::CfgPp(_)) {
                    2
                } else {
                    1
                },
                // The actual ordinary/CFG++ DPM2M factories also emit a stage-2 correction,
                // but skip it on a leading zero or zero landing. UniPC corrects even at its
                // terminal landing. This is a logical event, not a physical allocation credit.
                corrector_after_callback: (unipc && history == 2)
                    || (matches!(
                        self.route,
                        SamplingRoute::Curated(Solver::Dpmpp2m)
                            | SamplingRoute::CfgPp(Solver::Dpmpp2m)
                    ) && history == 1
                        && !leading_terminal
                        && next != 0.0),
                zero_landing: next == 0.0,
            });
            if !leading_terminal && next != 0.0 {
                let secondary = match self.route {
                    SamplingRoute::Curated(Solver::Heun) => {
                        Some((Evaluation::HeunCorrector, next, 3))
                    }
                    SamplingRoute::Curated(Solver::DpmppSde) => {
                        let t = -(sigma.max(1e-12) as f64).ln();
                        let t_next = -(next.max(1e-12) as f64).ln();
                        Some((
                            Evaluation::SdeMidpoint,
                            (-(t + (t_next - t) * 0.5)).exp() as f32,
                            2,
                        ))
                    }
                    _ => None,
                };
                if let Some((evaluation, sigma, outer_owners)) = secondary {
                    epochs.push(SamplingEpoch {
                        step: i,
                        evaluation,
                        sigma,
                        next_sigma: next,
                        history_owners: 0,
                        outer_owners,
                        denoised_outputs: 1,
                        corrector_after_callback: false,
                        zero_landing: false,
                    });
                }
            }
            match self.route {
                SamplingRoute::Curated(Solver::Dpmpp2m) | SamplingRoute::CfgPp(Solver::Dpmpp2m) => {
                    history = 1
                }
                SamplingRoute::Curated(Solver::UniPc) if next != 0.0 => history = 2,
                _ => {}
            }
            // UniPC breaks at a zero landing; other solvers continue for all supplied transitions.
            if unipc && next == 0.0 {
                break;
            }
        }
        Ok(epochs)
    }

    pub fn input_ownership(&self) -> InputOwnership {
        match self.route {
            SamplingRoute::Ancestral => InputOwnership::AncestralAliasThenCfgThenCast,
            SamplingRoute::Lcm | SamplingRoute::Hyper => InputOwnership::F32AliasThenCastThenCfg,
            _ => InputOwnership::F32ScaleThenCastThenCfg,
        }
    }
}

/// Order matters: ancestral batches BEFORE its F16 boundary; curated/accel cast BEFORE batching.
/// Bespoke initial text state is F16, reference state F32; subsequent state promotion follows the
/// ACTUAL epsilon dtype, not a blanket `model::DTYPE`. No enum variant promises physical donation.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum InputOwnership {
    AncestralAliasThenCfgThenCast,
    F32AliasThenCastThenCfg,
    F32ScaleThenCastThenCfg,
}

fn lcm_index_lower_bound(origin_len: usize, n: usize, threshold: usize) -> Result<usize> {
    let (mut lo, mut hi) = (0, n);
    for _ in 0..u32::BITS {
        if lo == hi {
            return Ok(lo);
        }
        let mid = lo + (hi - lo) / 2;
        let index = lcm_style_timestep_index(origin_len, n, mid)
            .ok_or_else(|| invalid("actual interval index missing"))?;
        if index < threshold {
            lo = mid + 1;
        } else {
            hi = mid;
        }
    }
    if lo == hi {
        Ok(lo)
    } else {
        Err(invalid("actual u32 interval search incomplete"))
    }
}

/// A bounded actual legacy-update mechanism, not an N-sized schedule, fitted cost or physical
/// certificate. Branches come from the same coefficients/predicates the executable sampler uses.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord)]
pub struct AccelerationFamily {
    pub first_step: bool,
    pub final_step: bool,
    pub input_scale_alias: bool,
    pub draws_noise: bool,
    pub additive_update: bool,
}

impl AccelerationFamily {
    fn at(step: usize, n: usize, coefficients: StepCoeffs) -> Self {
        Self {
            first_step: step == 0,
            final_step: step + 1 == n,
            input_scale_alias: coefficients.c_in == 1.0,
            draws_noise: coefficients.has_noise(),
            additive_update: coefficients.uses_unity_noiseless_update(),
        }
    }

    pub fn family(self) -> String {
        format!(
            "{}-{}-input{}-update{}-noise{}",
            if self.first_step { "first" } else { "steady" },
            if self.final_step {
                "terminal"
            } else {
                "nonterminal"
            },
            u8::from(self.input_scale_alias),
            u8::from(self.additive_update),
            u8::from(self.draws_noise)
        )
    }
}

#[derive(Clone, Debug, PartialEq)]
pub struct AccelerationEpoch {
    pub step: usize,
    pub final_step: bool,
    pub input_scale_alias: bool,
    pub draws_noise: bool,
    pub additive_update: bool,
    pub coefficients: StepCoeffs,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Evaluation {
    Primary,
    HeunCorrector,
    SdeMidpoint,
}

/// Owners in addition to the callback's current state: history is old_x0 / old_uncond (DPM2M),
/// prev_x0 + last_sample (UniPC); secondary outer owners are x+x0+d (Heun) or x+x0 (SDE).
/// These are logical handles, not counts of disjoint physical allocations. Curated enclosing
/// initialization noise and completed serial-image finals are owned OUTSIDE this solver scope and
/// must be added by pricing at each matched epoch. `count` never multiplies U-Net denoise rows.
#[derive(Clone, Debug, PartialEq)]
pub struct SamplingEpoch {
    pub step: usize,
    pub evaluation: Evaluation,
    pub sigma: f32,
    pub next_sigma: f32,
    pub history_owners: u8,
    pub outer_owners: u8,
    pub denoised_outputs: u8,
    pub corrector_after_callback: bool,
    pub zero_landing: bool,
}

pub(crate) fn strength_tail(full: &[f32], steps: usize, strength: f32) -> Vec<f32> {
    let effective = (steps as f32 * strength) as usize;
    let start = full.len().saturating_sub(1).saturating_sub(effective);
    full[start..].to_vec()
}

impl SamplingRecipe {
    pub fn from_request(req: &GenerationRequest) -> Result<Self> {
        Ok(ResolvedSampling::resolve(req)?.recipe)
    }

    /// Sampling-only request. For an actual replay first reconstruct the mode/conditioning and
    /// then call `apply_to`; this method does not fabricate image payloads or checkpoint policy.
    pub fn to_request(&self) -> GenerationRequest {
        let mut req = GenerationRequest::default();
        self.write_controls(&mut req);
        // With no media carrier there is nowhere to keep the per-Reference override. Project its
        // EFFECTIVE value onto the global strength, while the recipe/token retain both raw fields.
        req.strength = self.reference_strength.or(self.strength);
        req
    }

    fn write_controls(&self, req: &mut GenerationRequest) {
        req.sampler = Some(self.sampler.clone());
        req.scheduler = Some(self.scheduler.clone());
        req.steps = Some(self.steps);
        req.guidance = Some(self.guidance);
        req.guidance_method = Some(self.guidance_method.clone());
        req.memory.get_or_insert_with(Default::default).cfg_batching = Some(self.cfg_batching);
        req.strength = self.strength;
        req.use_pid = self.use_pid;
        req.pid_capture_sigma = self.pid_capture_sigma;
        for c in &mut req.conditioning {
            if let Conditioning::Reference { strength, .. } = c {
                *strength = self.reference_strength;
            }
        }
    }

    pub fn apply_to(&self, req: &mut GenerationRequest) -> Result<()> {
        self.write_controls(req);
        ResolvedSampling::resolve(req)?;
        Ok(())
    }

    pub fn to_value(&self) -> Value {
        serde_json::to_value(self).expect("validated finite recipe")
    }

    pub fn from_value(value: Value) -> Result<Self> {
        serde_json::from_value(value).map_err(|e| invalid(e.to_string()))
    }

    /// Safe single case-parser token; JSON bytes encoded as lowercase hex (no delimiters/quotes).
    pub fn canonical_token(&self) -> String {
        let json = self.to_value().to_string();
        let mut token = String::from("sdxl1-");
        for byte in json.bytes() {
            use std::fmt::Write;
            write!(&mut token, "{byte:02x}").expect("String write");
        }
        token
    }

    pub fn from_token(token: &str) -> Result<Self> {
        let hex = token
            .strip_prefix("sdxl1-")
            .ok_or_else(|| invalid("invalid token prefix"))?;
        if hex.len() > 16384
            || hex.len() % 2 != 0
            || !hex
                .bytes()
                .all(|b| b.is_ascii_digit() || (b'a'..=b'f').contains(&b))
        {
            return Err(invalid("invalid token encoding"));
        }
        let bytes: Result<Vec<u8>> = (0..hex.len())
            .step_by(2)
            .map(|i| {
                u8::from_str_radix(&hex[i..i + 2], 16).map_err(|_| invalid("invalid token byte"))
            })
            .collect();
        let value = serde_json::from_slice(&bytes?).map_err(|e| invalid(e.to_string()))?;
        let recipe = Self::from_value(value)?;
        if recipe.canonical_token() != token {
            return Err(invalid("noncanonical token"));
        }
        Ok(recipe)
    }
}

// The wire helper is deliberately separate from the validated public type: direct serde
// deserialization has the SAME validation as token parsing, not a back door around the resolver.
impl<'de> Deserialize<'de> for SamplingRecipe {
    fn deserialize<D: serde::Deserializer<'de>>(
        deserializer: D,
    ) -> std::result::Result<Self, D::Error> {
        #[derive(Deserialize)]
        #[serde(deny_unknown_fields)]
        struct Wire {
            version: u32,
            sampler: String,
            scheduler: String,
            steps: u32,
            guidance: f32,
            guidance_method: String,
            #[serde(with = "batching_serde")]
            cfg_batching: CfgBatching,
            strength: Option<f32>,
            reference_strength: Option<f32>,
            use_pid: bool,
            pid_capture_sigma: Option<f32>,
        }
        // Option fields must be present too: absence is not an implicit null/default in evidence.
        let value = Value::deserialize(deserializer)?;
        let object = value
            .as_object()
            .ok_or_else(|| <D::Error as serde::de::Error>::custom("recipe object required"))?;
        for field in ["strength", "reference_strength", "pid_capture_sigma"] {
            if !object.contains_key(field) {
                return Err(<D::Error as serde::de::Error>::custom(format!(
                    "missing {field}"
                )));
            }
        }
        let wire: Wire =
            serde_json::from_value(value).map_err(<D::Error as serde::de::Error>::custom)?;
        let recipe = Self {
            version: wire.version,
            sampler: wire.sampler,
            scheduler: wire.scheduler,
            steps: wire.steps,
            guidance: wire.guidance,
            guidance_method: wire.guidance_method,
            cfg_batching: wire.cfg_batching,
            strength: wire.strength,
            reference_strength: wire.reference_strength,
            use_pid: wire.use_pid,
            pid_capture_sigma: wire.pid_capture_sigma,
        };
        if recipe.version != 1
            || !recipe.guidance.is_finite()
            || [
                recipe.strength,
                recipe.reference_strength,
                recipe.pid_capture_sigma,
            ]
            .into_iter()
            .flatten()
            .any(|v| !v.is_finite())
        {
            return Err(<D::Error as serde::de::Error>::custom(
                "invalid recipe version/finite controls",
            ));
        }
        let resolved = ResolvedSampling::resolve_recipe(&recipe, SamplingWorkFacts::default())
            .map_err(<D::Error as serde::de::Error>::custom)?;
        if resolved.recipe.sampler != recipe.sampler
            || resolved.recipe.scheduler != recipe.scheduler
        {
            return Err(<D::Error as serde::de::Error>::custom(
                "noncanonical recipe controls",
            ));
        }
        Ok(recipe)
    }
}

mod batching_serde {
    use super::CfgBatching;
    use serde::{Deserialize, Deserializer, Serializer};
    pub fn serialize<S: Serializer>(value: &CfgBatching, serializer: S) -> Result<S::Ok, S::Error> {
        serializer.serialize_str(if value.is_batched() {
            "batched"
        } else {
            "sequential"
        })
    }
    pub fn deserialize<'de, D: Deserializer<'de>>(
        deserializer: D,
    ) -> Result<CfgBatching, D::Error> {
        match String::deserialize(deserializer)?.as_str() {
            "batched" => Ok(CfgBatching::Batched),
            "sequential" => Ok(CfgBatching::Sequential),
            _ => Err(<D::Error as serde::de::Error>::custom(
                "invalid CFG batching",
            )),
        }
    }
}

/// One representative per actual solver/CFG graph family, not a step×scheduler×float grid.
/// Endpoint load forms/rungs are supplied independently by the capture planner. Guidance off,
/// batched and sequential are distinct callback input graphs and stay explicit here.
pub fn representative_recipes() -> Vec<SamplingRecipe> {
    let mut recipes = Vec::new();
    for sampler in crate::model::descriptor().capabilities.samplers {
        for (guidance, batching) in [
            (1.0, CfgBatching::Batched),
            (7.0, CfgBatching::Batched),
            (7.0, CfgBatching::Sequential),
        ] {
            let req = GenerationRequest {
                sampler: Some(sampler.into()),
                steps: Some(3),
                guidance: Some(guidance),
                ..Default::default()
            };
            let mut req = req;
            req.memory.get_or_insert_with(Default::default).cfg_batching = Some(batching);
            recipes.push(SamplingRecipe::from_request(&req).expect("advertised route"));
            if sampler == "euler_ancestral" {
                req.scheduler = Some("normal".into());
                recipes.push(SamplingRecipe::from_request(&req).expect("curated ancestral"));
            }
            if matches!(sampler, "euler" | "dpmpp_2m") && guidance > 1.0 {
                req.guidance_method = Some("cfg_pp".into());
                recipes.push(SamplingRecipe::from_request(&req).expect("CFG++ route"));
            }
        }
    }
    recipes
}

#[cfg(test)]
#[path = "sampling_tests.rs"]
mod tests;
