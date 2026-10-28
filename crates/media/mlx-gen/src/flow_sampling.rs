//! Host-only canonical FLOW request replay and the plan consumed by actual curated factories.
//! Native schedules/defaults/aliases are supplied by the provider; no model policy is inferred here.
use super::{canonical_solver, resolve_flow_schedule};
use crate::gen_core::sampling::solvers::SolverEpoch;
use crate::gen_core::sampling::TimestepConvention;
use crate::{Conditioning, Error, GenerationRequest, Result, Scheduler, Solver};
use serde::{Deserialize, Serialize};

#[derive(Clone, Copy, Debug)]
pub struct FlowSamplingDefaults {
    pub solver: Solver,
    pub steps: u32,
    /// None means a genuinely CFG-free provider (not guidance=1 with an uncond branch).
    pub guidance: Option<f32>,
    pub true_cfg_overrides: bool,
    pub default_alias: bool,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct FlowSamplingWire {
    version: u32,
    solver: String,
    #[serde(deserialize_with = "required_option")]
    scheduler: Option<String>,
    steps: u32,
    #[serde(deserialize_with = "required_option")]
    guidance: Option<f32>,
    #[serde(deserialize_with = "required_option")]
    strength: Option<f32>,
    #[serde(deserialize_with = "required_option")]
    reference_strength: Option<f32>,
    use_pid: bool,
    #[serde(deserialize_with = "required_option")]
    pid_capture_sigma: Option<f32>,
}

fn required_option<'de, D: serde::Deserializer<'de>, T: Deserialize<'de>>(
    deserializer: D,
) -> std::result::Result<Option<T>, D::Error> {
    Option::<T>::deserialize(deserializer)
}

/// Validated controls, not a caller-claimed memory class. The enclosing case owns conditioning,
/// component paths, geometry/count and seed. Direct serde deserialization validates too.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(try_from = "FlowSamplingWire", into = "FlowSamplingWire")]
pub struct FlowSamplingRecipe(FlowSamplingWire);
impl From<FlowSamplingRecipe> for FlowSamplingWire {
    fn from(recipe: FlowSamplingRecipe) -> Self {
        recipe.0
    }
}
impl TryFrom<FlowSamplingWire> for FlowSamplingRecipe {
    type Error = String;
    fn try_from(wire: FlowSamplingWire) -> std::result::Result<Self, Self::Error> {
        if wire.version != 1
            || wire.steps == 0
            || Solver::from_name(&wire.solver).is_none()
            || wire
                .scheduler
                .as_deref()
                .is_some_and(|s| Scheduler::from_name(s).is_none())
            || [
                wire.guidance,
                wire.strength,
                wire.reference_strength,
                wire.pid_capture_sigma,
            ]
            .into_iter()
            .flatten()
            .any(|v| !v.is_finite())
        {
            return Err("invalid flow sampling version/controls".into());
        }
        Ok(Self(wire))
    }
}

/// O(1) in requested steps/count: admission/key resolution NEVER materializes a schedule.
#[derive(Clone, Debug)]
pub struct ResolvedFlowControls {
    recipe: FlowSamplingRecipe,
    solver: Solver,
    convention: TimestepConvention,
    effective_guidance: Option<f32>,
}

/// Exact schedule plan: only post-admission runtime or an explicitly byte-bounded capture analysis.
#[derive(Clone, Debug)]
pub struct ResolvedFlowSampling {
    recipe: FlowSamplingRecipe,
    solver: Solver,
    sigmas: Vec<f32>,
    convention: TimestepConvention,
    effective_guidance: Option<f32>,
}
fn invalid(message: impl Into<String>) -> Error {
    Error::Unsupported(format!("flow sampling: {}", message.into()))
}

/// Actual dynamic pricing must not reinterpret missing production context as a default run.
pub fn require_generation(generation: Option<&GenerationRequest>) -> Result<&GenerationRequest> {
    generation.ok_or_else(|| invalid("missing generation sampling context"))
}

impl ResolvedFlowControls {
    /// Resolve an already validated canonical recipe under the REAL provider's semantics, without
    /// creating a GenerationRequest/media carrier or materializing any schedule.
    pub fn from_recipe(
        recipe: &FlowSamplingRecipe,
        defaults: FlowSamplingDefaults,
        convention: TimestepConvention,
    ) -> Result<Self> {
        recipe.validate_defaults(defaults)?;
        let solver =
            Solver::from_name(&recipe.0.solver).ok_or_else(|| invalid("unserved sampler"))?;
        if solver.name() != recipe.0.solver
            || recipe.0.scheduler.as_deref().is_some_and(|name| {
                Scheduler::from_name(name).is_none_or(|scheduler| scheduler.name() != name)
            })
        {
            return Err(invalid("noncanonical recipe sampler/scheduler"));
        }
        Ok(Self {
            recipe: recipe.clone(),
            solver,
            convention,
            effective_guidance: recipe.0.guidance,
        })
    }

    pub fn resolve(
        req: &GenerationRequest,
        defaults: FlowSamplingDefaults,
        convention: TimestepConvention,
    ) -> Result<Self> {
        let sampler = req
            .sampler
            .as_deref()
            .filter(|s| !(defaults.default_alias && *s == "default"));
        if sampler.is_some_and(|s| Solver::from_name(s).is_none()) {
            return Err(invalid("unserved sampler"));
        }
        let scheduler = req
            .scheduler
            .as_deref()
            .filter(|s| !(defaults.default_alias && *s == "default"));
        if scheduler.is_some_and(|s| Scheduler::from_name(s).is_none()) {
            return Err(invalid("unserved scheduler"));
        }
        // These providers advertise no combine-method axis. Do not silently invent CFG++/APG.
        if req.guidance_method.is_some() {
            return Err(invalid("guidance methods are not served"));
        }
        let solver = canonical_solver(sampler, defaults.solver);
        let steps = req.steps.unwrap_or(defaults.steps);
        let mut references = req.conditioning.iter().filter_map(|c| match c {
            Conditioning::Reference { strength, .. } => Some(*strength),
            _ => None,
        });
        let reference_strength = references.next().flatten();
        if references.next().is_some() {
            return Err(invalid("multiple reference images"));
        }
        let effective_guidance = defaults.guidance.map(|default| {
            if defaults.true_cfg_overrides {
                req.true_cfg.or(req.guidance).unwrap_or(default)
            } else {
                req.guidance.unwrap_or(default)
            }
        });
        let recipe = FlowSamplingRecipe::try_from(FlowSamplingWire {
            version: 1,
            solver: solver.name().into(),
            scheduler: scheduler.map(str::to_owned),
            steps,
            guidance: effective_guidance,
            strength: req.strength,
            reference_strength,
            use_pid: req.use_pid,
            pid_capture_sigma: req.pid_capture_sigma,
        })
        .map_err(invalid)?;
        Ok(Self {
            recipe,
            solver,
            convention,
            effective_guidance,
        })
    }
    pub fn recipe(&self) -> &FlowSamplingRecipe {
        &self.recipe
    }
    pub fn solver(&self) -> Solver {
        self.solver
    }
    pub fn steps(&self) -> usize {
        self.recipe.0.steps as usize
    }
    /// The provider's unchanged native schedule, not a curated schedule whose stride/dedup may
    /// alter the effective transition count. This is a scalar control fact; no schedule is built.
    pub fn uses_native_schedule(&self) -> bool {
        self.recipe.0.scheduler.is_none()
    }
    pub fn effective_guidance(&self) -> Option<f32> {
        self.effective_guidance
    }
    pub fn convention(&self) -> TimestepConvention {
        self.convention
    }
    pub fn allocation_class(&self) -> String {
        allocation_class(self.solver, self.effective_guidance)
    }
    /// Checked source host-buffer envelope, not a step cap. Named terms follow native schedule,
    /// normal's timestep buffer, sigma Vec growth, resolved/window/model copies and virtual table.
    /// This raw allocation demand is NOT subtractive resident C / a measured physical law.
    pub fn host_schedule_bytes(&self) -> Result<u64> {
        use crate::gen_core::sampling::{FlowModelSampling, ModelSampling};
        let n = self.steps() as u64;
        let nodes = n
            .checked_add(1)
            .ok_or_else(|| invalid("schedule size overflow"))?;
        let table = FlowModelSampling::new(self.convention).num_timesteps() as u64;
        let sigma_nodes = nodes.max(
            table
                .checked_add(1)
                .ok_or_else(|| invalid("table size overflow"))?,
        );
        let grown = sigma_nodes
            .checked_mul(2)
            .ok_or_else(|| invalid("schedule growth overflow"))?;
        let terms = [
            ("native", nodes),
            ("normal_timesteps", n),
            ("sigma_growth_old", sigma_nodes),
            ("sigma_growth_new", grown),
            ("resolved_copy", sigma_nodes),
            ("window_copy", sigma_nodes),
            ("model_schedule_copy", sigma_nodes),
            ("virtual_table", table),
        ];
        terms
            .into_iter()
            .try_fold(0u64, |bytes, (_name, elements)| {
                elements
                    .checked_mul(4)
                    .and_then(|term| bytes.checked_add(term))
                    .ok_or_else(|| invalid("host schedule bytes overflow"))
            })
    }
    /// Byte budget belongs to the explicit bounded capture analysis, not product sampler policy.
    /// It is checked BEFORE any provider-native schedule builder may allocate.
    pub fn check_capture_schedule_bytes(&self, budget: u64) -> Result<()> {
        use crate::gen_core::sampling::{FlowModelSampling, ModelSampling};
        let nodes = (self.steps() as u64)
            .checked_add(1)
            .ok_or_else(|| invalid("capture nodes overflow"))?
            .max(FlowModelSampling::new(self.convention).num_timesteps() as u64 + 1);
        // Exact analysis may collect two callback epochs per transition plus a primary-ref view.
        let schedule_bytes = self.host_schedule_bytes()?;
        let epoch_bytes = nodes
            .checked_mul(
                (2 * std::mem::size_of::<SolverEpoch>() + std::mem::size_of::<&SolverEpoch>())
                    as u64,
            )
            .and_then(|n| n.checked_add(schedule_bytes))
            .ok_or_else(|| invalid("capture analysis bytes overflow"))?;
        if epoch_bytes > budget {
            return Err(invalid(
                "capture schedule exceeds explicit host byte budget",
            ));
        }
        Ok(())
    }
    /// Materialize the exact canonical schedule from already resolved controls,
    /// without manufacturing a GenerationRequest or media carrier. The caller
    /// must first pass `check_capture_schedule_bytes` (bounded analysis) or own
    /// runtime admission. `native` and `mu` are the provider's unchanged builder
    /// inputs; this is the SAME path as `ResolvedFlowSampling::resolve`.
    pub fn materialize_schedule(self, native: &[f32], mu: f32) -> Result<ResolvedFlowSampling> {
        let sigmas =
            resolve_flow_schedule(self.recipe.0.scheduler.as_deref(), mu, self.steps(), native);
        Ok(ResolvedFlowSampling {
            recipe: self.recipe,
            solver: self.solver,
            sigmas,
            convention: self.convention,
            effective_guidance: self.effective_guidance,
        })
    }

    /// Bounded symbolic factory envelope, NEVER replay user N here. FLOW input scaling is identity;
    /// the representative reaches every fixed-order history/secondary/update branch. No actual
    /// execution/physical donation/equivalence is claimed by this raw no-donation upper.
    pub fn memory_envelope(
        &self,
        elements: u64,
        initial_width: u64,
        prediction_width: u64,
    ) -> Result<super::solver_memory::SolverMemory> {
        let n = self.solver.minimum_capture_transitions();
        let sigmas: Vec<f32> = (0..=n).map(|i| 1.0 - i as f32 / n as f32).collect();
        let ms = crate::gen_core::sampling::FlowModelSampling::new(self.convention);
        super::solver_memory::solver_memory(
            self.solver,
            &ms,
            &sigmas,
            elements,
            initial_width,
            prediction_width,
        )
    }
}
fn allocation_class(solver: Solver, guidance: Option<f32>) -> String {
    format!(
        "flow1-{}-{}",
        solver.name(),
        if guidance.is_some_and(|g| g > 1.0) {
            "sequential-cfg"
        } else {
            "cfgfree"
        }
    )
}

impl ResolvedFlowSampling {
    /// Exact unchanged numeric builder. Call ONLY after admission or an explicit capture byte check.
    pub fn resolve(
        req: &GenerationRequest,
        defaults: FlowSamplingDefaults,
        native: &[f32],
        mu: f32,
        convention: TimestepConvention,
    ) -> Result<Self> {
        ResolvedFlowControls::resolve(req, defaults, convention)?.materialize_schedule(native, mu)
    }
    pub fn recipe(&self) -> &FlowSamplingRecipe {
        &self.recipe
    }
    pub fn solver(&self) -> Solver {
        self.solver
    }
    pub fn sigmas(&self) -> &[f32] {
        &self.sigmas
    }
    pub fn convention(&self) -> TimestepConvention {
        self.convention
    }
    pub fn steps(&self) -> usize {
        self.recipe.0.steps as usize
    }
    pub fn effective_guidance(&self) -> Option<f32> {
        self.effective_guidance
    }
    pub fn epochs(&self) -> Result<Vec<SolverEpoch>> {
        self.solver.logical_epochs(&self.sigmas).map_err(Into::into)
    }
    /// Restrict the actual resolved schedule to the same provider-selected tail/early-stop nodes.
    pub fn window(&mut self, start: usize, end: usize) -> Result<()> {
        if start >= end || end > self.sigmas.len() {
            return Err(invalid("invalid schedule window"));
        }
        self.sigmas = self.sigmas[start..end].to_vec();
        Ok(())
    }
    pub fn memory(
        &self,
        elements: u64,
        initial_width: u64,
        prediction_width: u64,
    ) -> Result<super::solver_memory::SolverMemory> {
        let ms = crate::gen_core::sampling::FlowModelSampling::new(self.convention);
        super::solver_memory::solver_memory(
            self.solver,
            &ms,
            &self.sigmas,
            elements,
            initial_width,
            prediction_width,
        )
    }
    /// Bounded capture PLANNING only. A real run must independently emit matching actual source
    /// observations; this method cannot establish exercised coverage.
    pub fn planned_capture_coverage(&self) -> Result<bool> {
        use crate::gen_core::sampling::solvers::SolverEvaluation;
        let epochs = self.epochs()?;
        let primary: Vec<_> = epochs
            .iter()
            .filter(|e| e.evaluation == SolverEvaluation::Primary)
            .collect();
        if primary.len() < self.solver.minimum_capture_transitions()
            || primary.last().is_none_or(|e| e.next_sigma != 0.0)
        {
            return Ok(false);
        }
        Ok(match self.solver {
            Solver::ErSde => epochs.iter().any(|e| e.correction_order == 3),
            Solver::Dpmpp2m | Solver::Dpmpp2mSde => primary
                .iter()
                .any(|e| e.previous_denoised && e.next_sigma != 0.0),
            Solver::UniPc => {
                primary
                    .iter()
                    .any(|e| e.previous_sample && e.next_sigma != 0.0)
                    && primary
                        .iter()
                        .any(|e| e.corrector_after_callback && e.next_sigma == 0.0)
            }
            Solver::Heun => epochs
                .iter()
                .any(|e| e.evaluation == SolverEvaluation::HeunCorrector),
            Solver::DpmppSde => epochs
                .iter()
                .any(|e| e.evaluation == SolverEvaluation::SdeMidpoint),
            _ => true,
        })
    }
    pub fn allocation_class(&self) -> String {
        format!(
            "flow1-{}-{}",
            self.solver.name(),
            if self.effective_guidance.is_some_and(|g| g > 1.0) {
                "sequential-cfg"
            } else {
                "cfgfree"
            }
        )
    }
}

impl FlowSamplingRecipe {
    /// Raw reference override wins over global strength; a provider supplies its own init default
    /// only after typed work facts establish an actual VAE init.
    pub fn effective_strength(&self) -> Option<f32> {
        self.0.reference_strength.or(self.0.strength)
    }

    /// Provider-owned capability semantics, not a default re-interpretation of an old None token.
    pub fn validate_defaults(&self, defaults: FlowSamplingDefaults) -> Result<()> {
        if self.0.guidance.is_some() != defaults.guidance.is_some() {
            return Err(invalid("recipe CFG-free/CFG-capable semantic mismatch"));
        }
        Ok(())
    }
    /// Reprice/capture replays must resolve through the REAL provider projection and verify this
    /// before accepting the token. This rejects cross-provider CFG semantics/default drift without
    /// copying provider policy into the shared recipe parser.
    pub fn validate_resolved(&self, resolved: &ResolvedFlowSampling) -> Result<()> {
        if self != resolved.recipe() {
            return Err(invalid("recipe differs from effective provider projection"));
        }
        Ok(())
    }
    pub fn validate_controls(&self, controls: &ResolvedFlowControls) -> Result<()> {
        if self != controls.recipe() {
            return Err(invalid("recipe differs from effective provider controls"));
        }
        Ok(())
    }
    pub fn to_request(&self) -> GenerationRequest {
        let mut req = GenerationRequest::default();
        self.apply_to(&mut req).expect("validated recipe");
        req.strength = self.0.reference_strength.or(self.0.strength);
        req
    }
    pub fn apply_to(&self, req: &mut GenerationRequest) -> Result<()> {
        req.sampler = Some(self.0.solver.clone());
        req.scheduler = self.0.scheduler.clone();
        req.steps = Some(self.0.steps);
        req.guidance = self.0.guidance;
        req.true_cfg = None;
        req.guidance_method = None;
        req.strength = self.0.strength;
        req.use_pid = self.0.use_pid;
        req.pid_capture_sigma = self.0.pid_capture_sigma;
        for c in &mut req.conditioning {
            if let Conditioning::Reference { strength, .. } = c {
                *strength = self.0.reference_strength;
            }
        }
        Ok(())
    }
    pub fn canonical_token(&self) -> String {
        encode_recipe_token("flow1-", self)
    }
    pub fn from_token(token: &str) -> Result<Self> {
        decode_recipe_token("flow1-", token)
    }
}

/// Shared SAFE wire encoding, not an evidence-class registry. Callers use digest-derived artifact
/// filenames: a replay token is deliberately complete and may exceed a filesystem component limit.
pub fn encode_recipe_token<T: Serialize>(prefix: &str, recipe: &T) -> String {
    use std::fmt::Write;
    let bytes = serde_json::to_vec(recipe).expect("validated finite replay controls");
    let mut token = prefix.to_owned();
    for byte in bytes {
        write!(&mut token, "{byte:02x}").expect("String write");
    }
    token
}
pub fn decode_recipe_token<T: serde::de::DeserializeOwned + Serialize>(
    prefix: &str,
    token: &str,
) -> Result<T> {
    let hex = token
        .strip_prefix(prefix)
        .ok_or_else(|| invalid("invalid token prefix"))?;
    if hex.len() > 16384
        || hex.len() % 2 != 0
        || !hex
            .bytes()
            .all(|b| b.is_ascii_digit() || (b'a'..=b'f').contains(&b))
    {
        return Err(invalid("invalid token encoding"));
    }
    let bytes: std::result::Result<Vec<_>, _> = (0..hex.len())
        .step_by(2)
        .map(|i| u8::from_str_radix(&hex[i..i + 2], 16))
        .collect();
    let recipe = serde_json::from_slice(&bytes.map_err(|e| invalid(e.to_string()))?)
        .map_err(|e| invalid(e.to_string()))?;
    if encode_recipe_token(prefix, &recipe) != token {
        return Err(invalid("noncanonical recipe token"));
    }
    Ok(recipe)
}

#[cfg(test)]
#[path = "flow_sampling_tests.rs"]
mod tests;
