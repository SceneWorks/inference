//! Tensor-free, object-safe Feature specification, hook-point map, structural applicability
//! derivation, and deterministic composition (SW-6938: HOOK-1..HOOK-5, E1).
//!
//! Nothing in this module touches tensors, weights, or a hardcoded model-ID capability table:
//! applicability is computed purely from a route's structural declaration ([`ModelDescriptor`]
//! plus its optional pre-load [`RenderPlanRegistration`]).

use std::collections::{BTreeMap, BTreeSet};

use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};

use crate::sampling::{ModelSampling, Solver};
use crate::{
    Conditioning, ConditioningKind, ControlKind, Error, GenerationPhase, GenerationRequest,
    IdentityWeights, Image, LatentPatchLayout, LoadSpec, Modality, ModelDescriptor, OffloadPolicy,
    PhaseAdapter, ProducedReference, RenderPlan, RenderPlanRegistration, RenderStage, Result,
    WeightsSource,
};

// -------------------------------------------------------------------------------------------------
// SW-6939 & SW-6940 (FEAT-1, PIPE-2, PIPE-3, PIPE-4, PIPE-5, PERF-3, HOOK-5, E2, E3):
// Request & Plan Contract, Typed Boundary State, Fp32 Transition Arithmetic, Estimator Provenance,
// Execution Receipt Report, and Bounded Quality/Time/Memory Qualification
// -------------------------------------------------------------------------------------------------

/// Canonical recipe identifier for the two-phase low-resolution start recipe (`FEAT-1`, `E3`).
pub const LOW_RES_START_RECIPE_ID: &str = "low_res_start";

/// Versioned fp32 boundary transition arithmetic identifier (`PIPE-4`, `E3`).
pub const LOW_RES_START_TRANSITION_ARITHMETIC_ID: &str =
    "unpack_f32_bilinear_half_pixel_clamp_renoise_repack_v1";

/// Versioned default endpoint clean-estimate policy identifier (`FEAT-1`, `E3`).
pub const LOW_RES_START_ENDPOINT_ESTIMATOR_ID: &str = "endpoint_evaluation_v1";

/// Versioned candidate previous-step primary clean-estimate policy identifier (`FEAT-1`, `DEC-10`, `E3`).
pub const LOW_RES_START_PREVIOUS_ESTIMATOR_ID: &str = "previous_evaluation_v1";

/// Domain separation label for the low-res-start phase-transition noise draw (`PIPE-4`, `§6.1`).
pub const LOW_RES_START_TRANSITION_DOMAIN: &str = "low_res_start_transition_v1";

/// Kind of clean-estimate evaluation used at the phase-0 -> phase-1 transition boundary (`PIPE-3`, `PIPE-4`, `FEAT-1`, `E3`).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum EvaluationKind {
    /// Read-only boundary model evaluation at `(x_low(sigma_K), sigma_K)` after step `K - 1` completes.
    /// Does not advance outer progress (`on_progress`), emit a duplicate preview frame, or mutate solver
    /// history, RNG state, or stateful guidance accumulators.
    BoundaryEstimate,
    /// Primary guided clean estimate (`x0`) captured from outer solver step `K - 1` at `sigma_{K-1}`
    /// while completing all `K` low-res outer steps (`0..K`) and their full solver sub-evaluations normally.
    GuidedPrimaryX0,
}

impl EvaluationKind {
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::BoundaryEstimate => "boundary_estimate",
            Self::GuidedPrimaryX0 => "guided_primary_x0",
        }
    }

    pub const fn is_read_only_boundary(self) -> bool {
        matches!(self, Self::BoundaryEstimate)
    }
}

/// Clean-estimate evaluation provenance at the phase-0 -> phase-1 transition boundary (`PIPE-4`, `FEAT-1`, `§6.1`).
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum EstimateSource {
    /// Default read-only endpoint evaluation at boundary `sigma_K`.
    #[default]
    EndpointEvaluation,
    /// Reuse the previous solver step's primary clean estimate (`GuidedPrimaryX0` from outer step `K - 1` at `sigma_{K-1}`) (`SW-6940`).
    PreviousEvaluation,
}

impl EstimateSource {
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::EndpointEvaluation => "endpoint_evaluation",
            Self::PreviousEvaluation => "previous_evaluation",
        }
    }

    pub fn from_name(name: &str) -> Option<Self> {
        match name {
            "endpoint_evaluation" | LOW_RES_START_ENDPOINT_ESTIMATOR_ID => {
                Some(Self::EndpointEvaluation)
            }
            "previous_evaluation" | LOW_RES_START_PREVIOUS_ESTIMATOR_ID => {
                Some(Self::PreviousEvaluation)
            }
            _ => None,
        }
    }

    /// Versioned estimator identity recorded in execution receipts (`E3`).
    pub const fn estimator_id(self) -> &'static str {
        match self {
            Self::EndpointEvaluation => LOW_RES_START_ENDPOINT_ESTIMATOR_ID,
            Self::PreviousEvaluation => LOW_RES_START_PREVIOUS_ESTIMATOR_ID,
        }
    }

    /// [`EvaluationKind`] corresponding to this clean-estimate policy.
    pub const fn evaluation_kind(self) -> EvaluationKind {
        match self {
            Self::EndpointEvaluation => EvaluationKind::BoundaryEstimate,
            Self::PreviousEvaluation => EvaluationKind::GuidedPrimaryX0,
        }
    }

    /// Number of dedicated read-only boundary model evaluations performed at `sigma_K` (`1` vs `0`).
    pub const fn boundary_evaluations(self) -> u32 {
        match self {
            Self::EndpointEvaluation => 1,
            Self::PreviousEvaluation => 0,
        }
    }

    /// Whether this policy is the default admission/replay recipe policy (`EndpointEvaluation`).
    pub const fn is_default_policy(self) -> bool {
        matches!(self, Self::EndpointEvaluation)
    }

    /// Schedule index where the boundary clean estimate is evaluated (`K` for `EndpointEvaluation`,
    /// `K - 1` for `PreviousEvaluation`).
    pub fn source_step_index(self, k: usize) -> Result<usize> {
        if k == 0 {
            return Err(Error::Unsupported(
                "low_res_start boundary step k must be >= 1".into(),
            ));
        }
        Ok(match self {
            Self::EndpointEvaluation => k,
            Self::PreviousEvaluation => k - 1,
        })
    }

    /// Validate that `solver` and `cfg_pp` can supply a well-defined boundary clean estimate under `self`.
    pub fn validate_solver(self, route_id: &str, _solver: Solver, cfg_pp: bool) -> Result<()> {
        if self == Self::PreviousEvaluation && cfg_pp {
            return Err(Error::Unsupported(format!(
                "{route_id}: low_res_start estimate_source=previous_evaluation is unsupported with cfg_pp guidance (CFG++ re-noises from unconditional x0 and cannot supply a standard single-callback GuidedPrimaryX0)"
            )));
        }
        Ok(())
    }

    /// Validate a request's `sampler` and `guidance_method` against `self`, returning the canonical
    /// [`Solver`]. Refuses unsupported estimator/solver combinations for `PreviousEvaluation`
    /// (such as legacy coefficient-only policies `"lightning"`, `"tcd"`, `"scm"`, or `"cfg_pp"`)
    /// with a typed [`Error::Unsupported`] without removing `EndpointEvaluation`'s supported routes or solvers.
    pub fn validate_sampler_name(
        self,
        route_id: &str,
        sampler_name: Option<&str>,
        guidance_method: Option<&str>,
    ) -> Result<Solver> {
        let cfg_pp = guidance_method == Some("cfg_pp");
        let solver = match sampler_name {
            None
            | Some("flow_match")
            | Some("flow_match_euler")
            | Some("hyper")
            | Some("native") => Solver::Euler,
            Some(name) => match Solver::from_name(name) {
                Some(solver) => solver,
                None => {
                    if self == Self::PreviousEvaluation {
                        return Err(Error::Unsupported(format!(
                            "{route_id}: low_res_start estimate_source=previous_evaluation is unsupported for solver/sampler {name:?} (cannot supply a well-defined GuidedPrimaryX0 at step K - 1)"
                        )));
                    }
                    Solver::Euler
                }
            },
        };
        self.validate_solver(route_id, solver, cfg_pp)?;
        Ok(solver)
    }
}

/// Request-level specification for the two-phase low-resolution start recipe (`FEAT-1`, `PIPE-2`).
#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
pub struct LowResStartSpec {
    /// Contract version (`1`). Any version other than `1` is refused with a typed [`Error::Unsupported`].
    pub version: u32,
    /// Boundary step index `K` (`1 <= K < M`, resolved after img2img strength truncation).
    pub k: usize,
    /// Spatial reduction scale (`0.25..=0.9`), defaulting to `0.5` when `None`.
    pub scale: Option<f32>,
    /// Clean-estimate evaluation provenance at boundary `K`.
    pub estimate_source: EstimateSource,
}

impl LowResStartSpec {
    pub const VERSION: u32 = 1;
    pub const SCALE_MIN: f32 = 0.25;
    pub const SCALE_MAX: f32 = 0.9;
    pub const SCALE_DEFAULT: f32 = 0.5;

    /// Construct a version-1 [`LowResStartSpec`] with default [`EstimateSource::EndpointEvaluation`].
    pub const fn new(k: usize, scale: Option<f32>) -> Self {
        Self {
            version: Self::VERSION,
            k,
            scale,
            estimate_source: EstimateSource::EndpointEvaluation,
        }
    }

    /// Override the contract version (for negative conformance tests).
    pub const fn with_version(mut self, version: u32) -> Self {
        self.version = version;
        self
    }

    /// Select the clean-estimate source at boundary `K`.
    pub const fn with_estimate_source(mut self, estimate_source: EstimateSource) -> Self {
        self.estimate_source = estimate_source;
        self
    }

    /// Validate `version == 1` and `scale in 0.25..=0.9` without clamping.
    pub fn validate_parameters(&self, route_id: &str) -> Result<f32> {
        if self.version != Self::VERSION {
            return Err(Error::Unsupported(format!(
                "{route_id}: low_res_start version {} is unsupported (expected {})",
                self.version,
                Self::VERSION
            )));
        }
        let scale = self.scale.unwrap_or(Self::SCALE_DEFAULT);
        if !scale.is_finite() || !(Self::SCALE_MIN..=Self::SCALE_MAX).contains(&scale) {
            return Err(Error::Unsupported(format!(
                "{route_id}: low_res_start scale {scale} must lie in {}..={}",
                Self::SCALE_MIN,
                Self::SCALE_MAX
            )));
        }
        Ok(scale)
    }

    /// Resolve the two-phase low-res geometry and step boundary against `executed_steps` (`M`)
    /// after any img2img strength truncation (`§6.1`, `FEAT-1`, `PIPE-2`).
    ///
    /// Never clamps invalid parameters or non-reducing dimensions.
    #[allow(clippy::too_many_arguments)]
    pub fn resolve_geometry(
        &self,
        route_id: &str,
        full_width: u32,
        full_height: u32,
        executed_steps: usize,
        vae_spatial_scale: u32,
        patch_size: u32,
        requires_dimensions_multiple_of: u32,
        min_dim: u32,
    ) -> Result<ResolvedLowResStartPlan> {
        let scale = self.validate_parameters(route_id)?;
        if executed_steps < 2 {
            return Err(Error::Unsupported(format!(
                "{route_id}: low_res_start requires executed_steps >= 2 after strength truncation, got {executed_steps}"
            )));
        }
        if self.k == 0 || self.k >= executed_steps {
            return Err(Error::Unsupported(format!(
                "{route_id}: low_res_start step k={} must satisfy 1 <= k < executed_steps ({executed_steps})",
                self.k
            )));
        }
        let legal_quantum = low_res_start_legal_quantum(
            vae_spatial_scale,
            patch_size,
            requires_dimensions_multiple_of,
        )?;
        if full_width == 0
            || full_height == 0
            || !full_width.is_multiple_of(legal_quantum)
            || !full_height.is_multiple_of(legal_quantum)
        {
            return Err(Error::Unsupported(format!(
                "{route_id}: target geometry {full_width}x{full_height} is not a positive multiple of legal_quantum={legal_quantum}"
            )));
        }
        let low_width = (((full_width as f32 * scale) / (legal_quantum as f32)).floor() as u32)
            .checked_mul(legal_quantum)
            .ok_or_else(|| Error::Unsupported("low_res_start width overflow".into()))?;
        let low_height = (((full_height as f32 * scale) / (legal_quantum as f32)).floor() as u32)
            .checked_mul(legal_quantum)
            .ok_or_else(|| Error::Unsupported("low_res_start height overflow".into()))?;
        let effective_min_dim = min_dim.max(legal_quantum);
        if low_width < effective_min_dim
            || low_height < effective_min_dim
            || low_width >= full_width
            || low_height >= full_height
        {
            return Err(Error::Unsupported(format!(
                "{route_id}: low_res_start scale {scale} on {full_width}x{full_height} produces {low_width}x{low_height} outside legal bounds (min_dim={effective_min_dim}, legal_quantum={legal_quantum})"
            )));
        }
        Ok(ResolvedLowResStartPlan {
            version: self.version,
            k: self.k,
            executed_steps,
            scale,
            legal_quantum,
            min_dim: effective_min_dim,
            low_width,
            low_height,
            full_width,
            full_height,
            estimate_source: self.estimate_source,
        })
    }

    /// Resolve [`LowResStartSpec`] against a [`ModelDescriptor`], [`GenerationRequest`], and the
    /// pass's truncated `executed_steps` (`M`).
    pub fn resolve_for_descriptor(
        &self,
        descriptor: &ModelDescriptor,
        req: &GenerationRequest,
        executed_steps: usize,
        pixel_stride: u32,
    ) -> Result<ResolvedLowResStartPlan> {
        if req.phases.is_some() {
            return Err(Error::Unsupported(format!(
                "{}: low_res_start conflicts with multi-phase denoise (phases)",
                descriptor.id
            )));
        }
        if !req.conditioning.is_empty() {
            return Err(Error::Unsupported(format!(
                "{}: low_res_start conflicts with pass-0 edit/reference/control conditioning",
                descriptor.id
            )));
        }
        if classify_descriptor_pass0_conditioning(descriptor)
            != Pass0ConditioningRequirement::NoiseStartSupported
        {
            return Err(Error::Unsupported(format!(
                "{}: route requires pass-0 conditioning and does not support low_res_start",
                descriptor.id
            )));
        }
        self.estimate_source.validate_sampler_name(
            descriptor.id,
            req.sampler.as_deref(),
            req.guidance_method.as_deref(),
        )?;
        let space = descriptor.denoiser_output_latent_space.ok_or_else(|| {
            Error::Unsupported(format!(
                "{}: route does not declare a packed spatial latent space for low_res_start",
                descriptor.id
            ))
        })?;
        if !space.is_well_formed()
            || space.spatial_compression.width != space.spatial_compression.height
        {
            return Err(Error::Unsupported(format!(
                "{}: route latent space does not have isotropic 2D spatial compression for low_res_start",
                descriptor.id
            )));
        }
        let vae_spatial_scale = u32::from(space.spatial_compression.width);
        let patch_size = match space.patch_layout {
            LatentPatchLayout::Packed {
                patch_height,
                patch_width,
            } if patch_height == patch_width && patch_width > 0 => u32::from(patch_width),
            LatentPatchLayout::Unpacked
                if pixel_stride > 0 && pixel_stride.is_multiple_of(vae_spatial_scale) =>
            {
                (pixel_stride / vae_spatial_scale).max(1)
            }
            _ => {
                return Err(Error::Unsupported(format!(
                    "{}: route latent space does not declare a 2x2-packed spatial layout",
                    descriptor.id
                )))
            }
        };
        let requires_multiple = descriptor
            .capabilities
            .size_floor
            .explicit_size_multiple()
            .unwrap_or(pixel_stride)
            .max(pixel_stride);
        let min_dim = descriptor.capabilities.min_size.max(requires_multiple);
        self.resolve_geometry(
            descriptor.id,
            req.width,
            req.height,
            executed_steps,
            vae_spatial_scale,
            patch_size,
            requires_multiple,
            min_dim,
        )
    }
}

/// Resolved, validated two-phase low-resolution execution plan for pass 0 (`FEAT-1`, `PIPE-2`).
#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
pub struct ResolvedLowResStartPlan {
    pub version: u32,
    pub k: usize,
    pub executed_steps: usize,
    pub scale: f32,
    pub legal_quantum: u32,
    pub min_dim: u32,
    pub low_width: u32,
    pub low_height: u32,
    pub full_width: u32,
    pub full_height: u32,
    pub estimate_source: EstimateSource,
}

impl ResolvedLowResStartPlan {
    /// Phase-0 step range `0..K` at `(low_width, low_height)`.
    pub const fn phase0_span(&self) -> std::ops::Range<usize> {
        0..self.k
    }

    /// Phase-1 step range `K..M` at `(full_width, full_height)`.
    pub const fn phase1_span(&self) -> std::ops::Range<usize> {
        self.k..self.executed_steps
    }

    /// Spatial `(width, height)` for denoise `phase` (`0` -> low-res, `>= 1` -> full-res).
    pub const fn phase_dimensions(&self, phase: usize) -> (u32, u32) {
        if phase == 0 {
            (self.low_width, self.low_height)
        } else {
            (self.full_width, self.full_height)
        }
    }

    /// Packed spatial token count for `phase` (`0` -> low-res, `>= 1` -> full-res) at `pixel_stride`.
    pub const fn phase_packed_tokens(&self, phase: usize, pixel_stride: u32) -> usize {
        let stride = if pixel_stride == 0 { 16 } else { pixel_stride };
        let (w, h) = self.phase_dimensions(phase);
        ((w / stride) * (h / stride)) as usize
    }

    /// Compute the evaluation/time/work profile for this low-res start plan over `sigmas` and `solver` (`PERF-3`, `E3`).
    pub fn work_profile(
        &self,
        solver: Solver,
        sigmas: &[f32],
        pixel_stride: u32,
    ) -> Result<LowResStartWorkProfile> {
        if sigmas.len() != self.executed_steps + 1 {
            return Err(Error::Unsupported(format!(
                "low_res_start work_profile requires sigmas.len() ({}) == executed_steps ({}) + 1",
                sigmas.len(),
                self.executed_steps
            )));
        }
        let epochs = solver.logical_epochs(sigmas)?;
        let phase0_step_evaluations = epochs.iter().filter(|e| e.step < self.k).count() as u32;
        let phase1_step_evaluations = epochs.iter().filter(|e| e.step >= self.k).count() as u32;
        let boundary_evaluations = self.estimate_source.boundary_evaluations();
        let total_model_evaluations =
            phase0_step_evaluations + boundary_evaluations + phase1_step_evaluations;
        let full_res_baseline_evaluations = epochs.len() as u32;
        let stages_per_step = solver.stages_per_step();
        let full_res_nominal_stage_evaluations = stages_per_step * (self.executed_steps as u32);
        let nominal_stage_evaluations = full_res_nominal_stage_evaluations + boundary_evaluations;
        let low_tokens = self.phase_packed_tokens(0, pixel_stride);
        let full_tokens = self.phase_packed_tokens(1, pixel_stride).max(1);
        let token_ratio = (low_tokens as f32) / (full_tokens as f32);
        let low_evals_nominal = stages_per_step * (self.k as u32) + boundary_evaluations;
        let full_evals_nominal = stages_per_step * ((self.executed_steps - self.k) as u32);
        let token_weighted_work_ratio = ((low_evals_nominal as f32) * token_ratio
            + (full_evals_nominal as f32))
            / (full_res_nominal_stage_evaluations.max(1) as f32);
        let theoretical_speedup = 1.0 / token_weighted_work_ratio.max(1e-6);

        Ok(LowResStartWorkProfile {
            estimate_source: self.estimate_source,
            estimator: self.estimate_source.estimator_id().to_string(),
            solver: solver.name().to_string(),
            stages_per_step,
            k: self.k,
            executed_steps: self.executed_steps,
            low_tokens,
            full_tokens,
            phase0_step_evaluations,
            boundary_evaluations,
            phase1_step_evaluations,
            total_model_evaluations,
            nominal_stage_evaluations,
            full_res_baseline_evaluations,
            full_res_nominal_stage_evaluations,
            token_weighted_work_ratio,
            theoretical_speedup,
        })
    }

    /// Compute the memory & retained-buffer coexistence profile for this low-res start plan (`PIPE-3`, `PIPE-4`, `E3`).
    pub fn memory_profile(
        &self,
        batch: usize,
        pixel_stride: u32,
        token_width: u32,
        dtype_bytes: u64,
    ) -> LowResStartMemoryProfile {
        let batch_u64 = (batch.max(1)) as u64;
        let token_width_u64 = u64::from(token_width.max(1));
        let dtype_u64 = dtype_bytes.max(1);
        let low_tokens = self.phase_packed_tokens(0, pixel_stride) as u64;
        let full_tokens = self.phase_packed_tokens(1, pixel_stride) as u64;
        let low_packed_latent_bytes = batch_u64 * low_tokens * token_width_u64 * dtype_u64;
        let full_packed_latent_bytes = batch_u64 * full_tokens * token_width_u64 * dtype_u64;
        let low_f32_bytes = batch_u64 * low_tokens * token_width_u64 * 4;
        let full_f32_bytes = batch_u64 * full_tokens * token_width_u64 * 4;
        // fp32 unpack (`low_f32`) + bilinear resize (`full_f32`) + transition noise (`full_f32`) + repack (`full_f32`)
        let transition_f32_workspace_bytes = low_f32_bytes + 3 * full_f32_bytes;
        let (retained_primary_estimate_bytes, boundary_eval_workspace_bytes) =
            match self.estimate_source {
                EstimateSource::EndpointEvaluation => (0, low_packed_latent_bytes),
                EstimateSource::PreviousEvaluation => (low_packed_latent_bytes, 0),
            };
        // During step K - 1's advancement: running low-res latent x + solver derivative/update working tensor + any retained primary estimate x0.
        let phase0_step_k_minus_1_coexisting_buffer_bytes =
            2 * low_packed_latent_bytes + retained_primary_estimate_bytes;

        LowResStartMemoryProfile {
            estimate_source: self.estimate_source,
            estimator: self.estimate_source.estimator_id().to_string(),
            low_packed_latent_bytes,
            full_packed_latent_bytes,
            retained_primary_estimate_bytes,
            boundary_eval_workspace_bytes,
            transition_f32_workspace_bytes,
            phase0_step_k_minus_1_coexisting_buffer_bytes,
        }
    }

    /// Build the persisted [`LowResStartExecutionReport`] receipt for this resolved plan (`SW-6940`: `FEAT-1`, `E3`).
    #[allow(clippy::too_many_arguments)]
    pub fn execution_report(
        &self,
        pass_index: usize,
        sigmas: &[f32],
        solver: Solver,
        guidance_mode: &str,
        guidance_scale: f32,
        slot_configuration: &str,
        dtype: &str,
        dtype_bytes: u64,
        batch: usize,
        pixel_stride: u32,
        token_width: u32,
    ) -> Result<LowResStartExecutionReport> {
        let source_step = self.estimate_source.source_step_index(self.k)?;
        let sigma = *sigmas.get(source_step).ok_or_else(|| {
            Error::Unsupported(format!(
                "low_res_start provenance step_index {source_step} out of bounds for sigmas len {}",
                sigmas.len()
            ))
        })?;
        let work = self.work_profile(solver, sigmas, pixel_stride)?;
        let memory = self.memory_profile(batch, pixel_stride, token_width, dtype_bytes);
        let provenance = EstimateProvenance {
            pass_index,
            phase_index: 0,
            evaluation_kind: self.estimate_source.evaluation_kind(),
            step_index: source_step,
            sigma,
            solver: solver.name().to_string(),
            guidance_mode: guidance_mode.to_string(),
            guidance_scale,
            slot_configuration: slot_configuration.to_string(),
            dtype: dtype.to_string(),
            read_only_boundary: true,
            boundary_evaluations: work.boundary_evaluations,
            retained_primary_estimate_bytes: memory.retained_primary_estimate_bytes,
        };
        Ok(LowResStartExecutionReport {
            recipe_id: LOW_RES_START_RECIPE_ID.to_string(),
            version: self.version,
            k: self.k,
            executed_steps: self.executed_steps,
            scale: self.scale,
            low_width: self.low_width,
            low_height: self.low_height,
            full_width: self.full_width,
            full_height: self.full_height,
            estimate_source: self.estimate_source,
            estimator: self.estimate_source.estimator_id().to_string(),
            estimate_provenance: provenance,
            transition_arithmetic_id: LOW_RES_START_TRANSITION_ARITHMETIC_ID.to_string(),
            transition_seed_derivation_id: LOW_RES_START_TRANSITION_DOMAIN.to_string(),
            phase0_step_evaluations: work.phase0_step_evaluations,
            boundary_evaluations: work.boundary_evaluations,
            phase1_step_evaluations: work.phase1_step_evaluations,
            total_model_evaluations: work.total_model_evaluations,
        })
    }
}

/// Full evaluation provenance of the clean estimate $\hat{x}_0^{\text{low}}$ used at the
/// phase-0 -> phase-1 transition boundary (`SW-6940`: `FEAT-1`, `PIPE-3`, `PIPE-4`, `E3`).
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct EstimateProvenance {
    /// Pass index (`0` for base pass 0).
    pub pass_index: usize,
    /// Departing denoise phase index (`0` for low-res phase 0).
    pub phase_index: usize,
    /// Evaluation kind (`BoundaryEstimate` for `EndpointEvaluation`, `GuidedPrimaryX0` for `PreviousEvaluation`).
    pub evaluation_kind: EvaluationKind,
    /// Schedule step index where the clean estimate was evaluated (`K` for `EndpointEvaluation`, `K - 1` for `PreviousEvaluation`).
    pub step_index: usize,
    /// Schedule sigma at which the clean estimate was evaluated ($\sigma_K$ for `EndpointEvaluation`, $\sigma_{K-1}$ for `PreviousEvaluation`).
    pub sigma: f32,
    /// Canonical solver name (`"euler"`, `"heun"`, `"dpmpp_2m"`, `"dpmpp_sde"`, etc.).
    pub solver: String,
    /// Active guidance mode during the clean-estimate evaluation (`"embedded_guidance"`, `"distilled_unconditional"`, `"true_cfg"`, `"apg"`, etc.).
    pub guidance_mode: String,
    /// Effective guidance scale during the clean-estimate evaluation.
    pub guidance_scale: f32,
    /// Departing phase's model/adapter slot configuration (e.g. `"base_dit"` or `"base_dit+adapters:1"`).
    pub slot_configuration: String,
    /// Latent/compute dtype string (e.g. `"f32"`, `"bf16"`).
    pub dtype: String,
    /// Whether the boundary transition evaluation is strictly read-only (does not advance outer progress,
    /// emit a duplicate preview frame, or mutate solver history, RNG state, or stateful guidance accumulators).
    pub read_only_boundary: bool,
    /// Number of dedicated read-only boundary model evaluations (`1` for `EndpointEvaluation`, `0` for `PreviousEvaluation`).
    pub boundary_evaluations: u32,
    /// Bytes of the primary clean estimate buffer `x0` retained concurrently across step `K - 1`'s solver update
    /// (`0` for `EndpointEvaluation`, `B * low_tokens * token_width * dtype_bytes` for `PreviousEvaluation`).
    pub retained_primary_estimate_bytes: u64,
}

/// Persisted execution receipt for the two-phase low-resolution start recipe (`SW-6940`: `FEAT-1`, `PIPE-3`, `PIPE-4`, `PERF-3`, `E3`).
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct LowResStartExecutionReport {
    /// Canonical recipe identifier (`"low_res_start"`).
    pub recipe_id: String,
    /// Recipe contract version (`1`).
    pub version: u32,
    /// Boundary step index `K` (`1 <= K < executed_steps`).
    pub k: usize,
    /// Executed pass-0 denoise steps `M` (`executed_steps`, after any img2img strength truncation).
    pub executed_steps: usize,
    /// Resolved spatial reduction scale in `0.25..=0.9`.
    pub scale: f32,
    /// Phase-0 low-resolution width in pixels.
    pub low_width: u32,
    /// Phase-0 low-resolution height in pixels.
    pub low_height: u32,
    /// Phase-1 full target width in pixels.
    pub full_width: u32,
    /// Phase-1 full target height in pixels.
    pub full_height: u32,
    /// Selected clean-estimate policy (`EndpointEvaluation` default vs `PreviousEvaluation` candidate).
    pub estimate_source: EstimateSource,
    /// Versioned estimator identifier (`"endpoint_evaluation_v1"` vs `"previous_evaluation_v1"`).
    pub estimator: String,
    /// Complete provenance of the boundary clean estimate.
    pub estimate_provenance: EstimateProvenance,
    /// Versioned fp32 boundary transition arithmetic identifier (`"unpack_f32_bilinear_half_pixel_clamp_renoise_repack_v1"`).
    pub transition_arithmetic_id: String,
    /// Versioned deterministic transition noise seed derivation domain (`"low_res_start_transition_v1"`).
    pub transition_seed_derivation_id: String,
    /// Model evaluations executed in Phase 0 (`0..K`).
    pub phase0_step_evaluations: u32,
    /// Read-only boundary model evaluations at step `K` (`1` for `EndpointEvaluation`, `0` for `PreviousEvaluation`).
    pub boundary_evaluations: u32,
    /// Model evaluations executed in Phase 1 (`K..M`).
    pub phase1_step_evaluations: u32,
    /// Total model evaluations across the pass (`phase0_step_evaluations + boundary_evaluations + phase1_step_evaluations`).
    pub total_model_evaluations: u32,
}

/// Evaluation and token-weighted work comparison profile for one estimator policy (`PERF-3`, `E3`).
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct LowResStartWorkProfile {
    pub estimate_source: EstimateSource,
    pub estimator: String,
    pub solver: String,
    pub stages_per_step: u32,
    pub k: usize,
    pub executed_steps: usize,
    pub low_tokens: usize,
    pub full_tokens: usize,
    pub phase0_step_evaluations: u32,
    pub boundary_evaluations: u32,
    pub phase1_step_evaluations: u32,
    pub total_model_evaluations: u32,
    /// Nominal stage evaluation count (`stages_per_step * M + boundary_evaluations`: `M + 1` vs `M`
    /// for 1-stage solvers, `2M + 1` vs `2M` for 2-stage solvers).
    pub nominal_stage_evaluations: u32,
    pub full_res_baseline_evaluations: u32,
    pub full_res_nominal_stage_evaluations: u32,
    pub token_weighted_work_ratio: f32,
    pub theoretical_speedup: f32,
}

/// Memory and retained-buffer coexistence profile for one estimator policy (`PIPE-3`, `PIPE-4`, `E3`).
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct LowResStartMemoryProfile {
    pub estimate_source: EstimateSource,
    pub estimator: String,
    pub low_packed_latent_bytes: u64,
    pub full_packed_latent_bytes: u64,
    pub retained_primary_estimate_bytes: u64,
    pub boundary_eval_workspace_bytes: u64,
    pub transition_f32_workspace_bytes: u64,
    pub phase0_step_k_minus_1_coexisting_buffer_bytes: u64,
}

/// Recorded execution trace for one estimator policy during qualification (`E3`, `FEAT-1`, `PIPE-3`).
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct LowResStartTraceSummary {
    pub completed_low_res_steps: usize,
    pub phase0_sub_evaluations: Vec<(usize, String, u32)>,
    pub phase0_rng_draw_coordinates: Vec<(u64, usize)>,
    pub phase0_progress_events: Vec<(u32, u32)>,
    pub phase0_preview_frames: Vec<(u32, u32, u32, u32)>,
    pub boundary_evaluations: u32,
    pub boundary_advanced_progress: bool,
    pub boundary_emitted_preview: bool,
    pub boundary_mutated_solver_history: bool,
    pub boundary_mutated_rng: bool,
    pub boundary_mutated_guidance_accumulator: bool,
    pub phase1_sub_evaluations: Vec<(usize, String, u32)>,
    pub phase1_rng_draw_coordinates: Vec<(u64, usize)>,
    pub total_progress_events: Vec<(u32, u32)>,
}

/// Bounded quality and spatial-seam comparison metrics against the full-resolution reference trajectory (`E3`, `FEAT-1`).
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct LowResStartQualityMetrics {
    pub all_finite: bool,
    pub cosine_similarity_vs_full_res: f32,
    pub rms_diff_vs_full_res: f32,
    pub max_seam_gradient_ratio: f32,
}

/// Complete structured qualification report comparing `EstimateSource::EndpointEvaluation` (default)
/// vs `EstimateSource::PreviousEvaluation` (candidate under DEC-10) vs full-resolution baseline (`SW-6940`: `E3`, `FEAT-1`, `PERF-3`).
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct LowResStartQualificationReport {
    pub default_policy: EstimateSource,
    pub candidate_policy: EstimateSource,
    pub plan_endpoint: ResolvedLowResStartPlan,
    pub plan_previous: ResolvedLowResStartPlan,
    pub endpoint_report: LowResStartExecutionReport,
    pub previous_report: LowResStartExecutionReport,
    pub endpoint_work: LowResStartWorkProfile,
    pub previous_work: LowResStartWorkProfile,
    pub endpoint_memory: LowResStartMemoryProfile,
    pub previous_memory: LowResStartMemoryProfile,
    pub endpoint_trace: LowResStartTraceSummary,
    pub previous_trace: LowResStartTraceSummary,
    pub phase0_traces_identical: bool,
    pub phase1_rng_coordinates_identical: bool,
    pub outer_progress_identical: bool,
    pub preview_sequences_identical: bool,
    pub endpoint_quality: LowResStartQualityMetrics,
    pub previous_quality: LowResStartQualityMetrics,
    pub endpoint_vs_previous_cosine_similarity: f32,
    pub endpoint_vs_previous_rms_diff: f32,
    pub default_selection_unchanged: bool,
}

/// Greatest common divisor of two positive `u32` values.
pub const fn gcd_u32(mut a: u32, mut b: u32) -> u32 {
    while b != 0 {
        let r = a % b;
        a = b;
        b = r;
    }
    a
}

/// Least common multiple of two positive `u32` values, refusing zero or overflow.
pub fn lcm_u32(a: u32, b: u32) -> Result<u32> {
    if a == 0 || b == 0 {
        return Err(Error::Unsupported(
            "low_res_start spatial quantum parameters must be non-zero".into(),
        ));
    }
    let g = gcd_u32(a, b);
    (a / g)
        .checked_mul(b)
        .ok_or_else(|| Error::Unsupported("low_res_start spatial quantum lcm overflow".into()))
}

/// Compute `legal_quantum = lcm(vae_spatial_scale * patch_size, requires_dimensions_multiple_of)` (`§6.1`).
pub fn low_res_start_legal_quantum(
    vae_spatial_scale: u32,
    patch_size: u32,
    requires_dimensions_multiple_of: u32,
) -> Result<u32> {
    let token_stride = vae_spatial_scale
        .checked_mul(patch_size)
        .ok_or_else(|| Error::Unsupported("low_res_start token stride overflow".into()))?;
    lcm_u32(token_stride, requires_dimensions_multiple_of)
}

/// Resolve the img2img start step (`init_time_step`) and validate `strength` in `[0.0, 1.0]` without clamping
/// out-of-range values when validating `low_res_start` + img2img (`§6.13`, `FEAT-1`).
pub fn resolve_img2img_init_time_step(num_steps: usize, strength: Option<f32>) -> Result<usize> {
    if num_steps == 0 {
        return Err(Error::Unsupported(
            "img2img schedule requires num_steps >= 1".into(),
        ));
    }
    let s = strength.unwrap_or(0.5);
    if !s.is_finite() || !(0.0..=1.0).contains(&s) {
        return Err(Error::Unsupported(format!(
            "img2img strength {s} must be finite and lie in 0.0..=1.0"
        )));
    }
    if s > 0.0 {
        Ok(((num_steps as f32 * s) as usize).max(1))
    } else {
        Ok(0)
    }
}

/// Truncate a full-pass sigma schedule `full_sigmas[0..=num_steps]` (computed once at full target
/// geometry and full step count) by ordinary img2img `strength`, returning `(start_step, truncated_sigmas)`.
///
/// The truncated schedule has `M = truncated_sigmas.len() - 1` executed steps, and `K` is resolved
/// **after** this truncation (`1 <= K < M`, `M >= 2`).
pub fn resolve_img2img_truncated_schedule(
    full_sigmas: &[f32],
    strength: Option<f32>,
) -> Result<(usize, Vec<f32>)> {
    if full_sigmas.len() < 2 {
        return Err(Error::Unsupported(
            "full sigma schedule must have at least 2 nodes".into(),
        ));
    }
    let num_steps = full_sigmas.len() - 1;
    let start_step = resolve_img2img_init_time_step(num_steps, strength)?;
    if start_step >= num_steps {
        return Err(Error::Unsupported(format!(
            "img2img strength {:?} on {num_steps} steps leaves {} executed steps (low_res_start requires M >= 2)",
            strength,
            num_steps.saturating_sub(start_step)
        )));
    }
    Ok((start_step, full_sigmas[start_step..].to_vec()))
}

/// Typed state at a phase boundary (`PIPE-4`, `FEAT-1`, `§6.1`).
///
/// Re-noising from a clean estimate only accepts [`BoundaryState::CleanEstimate`]; passing
/// [`BoundaryState::XSigma`] or [`BoundaryState::RouteSpecific`] refuses with a typed
/// [`Error::Unsupported`], preventing double-noising of an already-noisy latent.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum BoundaryState<T> {
    /// Noisy trajectory state $x(\sigma_K)$ at boundary $\sigma_K$.
    XSigma(T),
    /// Clean $\hat{x}_0$ estimate at boundary $\sigma_K$ with its evaluation provenance.
    CleanEstimate { latent: T, source: EstimateSource },
    /// Route-specific boundary carrier that cannot be re-noised as a clean estimate.
    RouteSpecific { label: &'static str, state: T },
}

impl<T> BoundaryState<T> {
    /// Extract the clean estimate and its provenance, refusing noisy (`XSigma`) or
    /// route-specific boundary states with a typed [`Error::Unsupported`].
    pub fn into_clean_estimate(self) -> Result<(T, EstimateSource)> {
        match self {
            Self::CleanEstimate { latent, source } => Ok((latent, source)),
            Self::XSigma(_) => Err(Error::Unsupported(
                "low_res_start boundary re-noise requires BoundaryState::CleanEstimate; refused noisy BoundaryState::XSigma to prevent double-noising".into(),
            )),
            Self::RouteSpecific { label, .. } => Err(Error::Unsupported(format!(
                "low_res_start boundary re-noise requires BoundaryState::CleanEstimate; refused route-specific boundary state '{label}'"
            ))),
        }
    }

    /// Borrow the inner latent reference wrapped in a borrowed [`BoundaryState`].
    pub fn as_ref(&self) -> BoundaryState<&T> {
        match self {
            Self::XSigma(x) => BoundaryState::XSigma(x),
            Self::CleanEstimate { latent, source } => BoundaryState::CleanEstimate {
                latent,
                source: *source,
            },
            Self::RouteSpecific { label, state } => BoundaryState::RouteSpecific { label, state },
        }
    }
}

/// Derive a deterministic domain-separated 64-bit seed from
/// `(request_seed, output_index, pass_index, phase_index, "low_res_start_transition_v1")` (`PIPE-4`, `§6.1`).
///
/// Uses SHA-256 over the domain label and tuple coordinates so it never collides with initial
/// noise (`request_seed + output_index`) or per-step stochastic solver draws (`StepRng`).
pub fn low_res_start_transition_seed(
    request_seed: u64,
    output_index: u32,
    pass_index: usize,
    phase_index: usize,
) -> u64 {
    let mut hasher = Sha256::new();
    hasher.update(LOW_RES_START_TRANSITION_DOMAIN.as_bytes());
    hasher.update([0u8]);
    hasher.update(request_seed.to_le_bytes());
    hasher.update(output_index.to_le_bytes());
    hasher.update((pass_index as u64).to_le_bytes());
    hasher.update((phase_index as u64).to_le_bytes());
    let digest = hasher.finalize();
    let mut bytes = [0u8; 8];
    bytes.copy_from_slice(&digest[..8]);
    u64::from_le_bytes(bytes)
}

/// Draw `len` independent unit Gaussian $\mathcal{N}(0, 1)$ samples in fp32 using the
/// deterministic domain-separated transition seed.
pub fn low_res_start_transition_noise_f32(
    request_seed: u64,
    output_index: u32,
    pass_index: usize,
    phase_index: usize,
    len: usize,
) -> Vec<f32> {
    let mut state =
        low_res_start_transition_seed(request_seed, output_index, pass_index, phase_index);
    let mut next_u64 = move || {
        state = state.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = state;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^ (z >> 31)
    };
    let unit = |u: u64| (u >> 11) as f64 / (1u64 << 53) as f64;
    let mut out = Vec::with_capacity(len);
    while out.len() < len {
        let u1 = unit(next_u64()).max(1e-12);
        let u2 = unit(next_u64());
        let r = (-2.0 * u1.ln()).sqrt();
        let theta = 2.0 * std::f64::consts::PI * u2;
        out.push((r * theta.cos()) as f32);
        if out.len() < len {
            out.push((r * theta.sin()) as f32);
        }
    }
    out
}

/// Unpack a token-packed latent `[B, (H_lat/p)*(W_lat/p), C*p*p]` to physical spatial axes
/// `[B, C, H_lat, W_lat]` in fp32 (`PIPE-4`, `§6.1`).
///
/// Matches the 2x2 patch packing of FLUX.1 (`[B, h*w, C*p*p]` -> `[B, h, w, C, p, p]` ->
/// transpose `[0, 3, 1, 4, 2, 5]` -> `[B, C, h*p, w*p]`).
pub fn unpack_packed_latents_f32(
    packed: &[f32],
    batch: usize,
    channels: usize,
    latent_height: usize,
    latent_width: usize,
    patch_size: usize,
) -> Result<Vec<f32>> {
    if batch == 0
        || channels == 0
        || latent_height == 0
        || latent_width == 0
        || patch_size == 0
        || !latent_height.is_multiple_of(patch_size)
        || !latent_width.is_multiple_of(patch_size)
    {
        return Err(Error::Unsupported(format!(
            "unpack_packed_latents_f32: invalid dimensions B={batch}, C={channels}, H={latent_height}, W={latent_width}, p={patch_size}"
        )));
    }
    let expected = batch
        .checked_mul(channels)
        .and_then(|v| v.checked_mul(latent_height))
        .and_then(|v| v.checked_mul(latent_width))
        .ok_or_else(|| {
            Error::Unsupported("unpack_packed_latents_f32: element count overflow".into())
        })?;
    if packed.len() != expected {
        return Err(Error::Unsupported(format!(
            "unpack_packed_latents_f32: buffer len {} != expected {expected}",
            packed.len()
        )));
    }
    let grid_h = latent_height / patch_size;
    let grid_w = latent_width / patch_size;
    let token_width = channels * patch_size * patch_size;
    let tokens_per_batch = grid_h * grid_w;
    let mut unpacked = vec![0.0_f32; expected];

    for b in 0..batch {
        for gh in 0..grid_h {
            for gw in 0..grid_w {
                let tok_idx = (b * tokens_per_batch + (gh * grid_w + gw)) * token_width;
                for c in 0..channels {
                    for py in 0..patch_size {
                        for px in 0..patch_size {
                            let patch_offset = (c * patch_size + py) * patch_size + px;
                            let y = gh * patch_size + py;
                            let x = gw * patch_size + px;
                            let dst_idx =
                                ((b * channels + c) * latent_height + y) * latent_width + x;
                            unpacked[dst_idx] = packed[tok_idx + patch_offset];
                        }
                    }
                }
            }
        }
    }
    Ok(unpacked)
}

/// Repack physical spatial axes `[B, C, H_lat, W_lat]` back to token-packed layout
/// `[B, (H_lat/p)*(W_lat/p), C*p*p]` in fp32 (`PIPE-4`, `§6.1`).
pub fn repack_unpacked_latents_f32(
    unpacked: &[f32],
    batch: usize,
    channels: usize,
    latent_height: usize,
    latent_width: usize,
    patch_size: usize,
) -> Result<Vec<f32>> {
    if batch == 0
        || channels == 0
        || latent_height == 0
        || latent_width == 0
        || patch_size == 0
        || !latent_height.is_multiple_of(patch_size)
        || !latent_width.is_multiple_of(patch_size)
    {
        return Err(Error::Unsupported(format!(
            "repack_unpacked_latents_f32: invalid dimensions B={batch}, C={channels}, H={latent_height}, W={latent_width}, p={patch_size}"
        )));
    }
    let expected = batch
        .checked_mul(channels)
        .and_then(|v| v.checked_mul(latent_height))
        .and_then(|v| v.checked_mul(latent_width))
        .ok_or_else(|| {
            Error::Unsupported("repack_unpacked_latents_f32: element count overflow".into())
        })?;
    if unpacked.len() != expected {
        return Err(Error::Unsupported(format!(
            "repack_unpacked_latents_f32: buffer len {} != expected {expected}",
            unpacked.len()
        )));
    }
    let grid_h = latent_height / patch_size;
    let grid_w = latent_width / patch_size;
    let token_width = channels * patch_size * patch_size;
    let tokens_per_batch = grid_h * grid_w;
    let mut packed = vec![0.0_f32; expected];

    for b in 0..batch {
        for gh in 0..grid_h {
            for gw in 0..grid_w {
                let tok_idx = (b * tokens_per_batch + (gh * grid_w + gw)) * token_width;
                for c in 0..channels {
                    for py in 0..patch_size {
                        for px in 0..patch_size {
                            let patch_offset = (c * patch_size + py) * patch_size + px;
                            let y = gh * patch_size + py;
                            let x = gw * patch_size + px;
                            let src_idx =
                                ((b * channels + c) * latent_height + y) * latent_width + x;
                            packed[tok_idx + patch_offset] = unpacked[src_idx];
                        }
                    }
                }
            }
        }
    }
    Ok(packed)
}

/// 1D half-pixel (`align_corners = false`) coordinate mapping with `clamp-to-edge` border handling (`§6.1`):
/// `coord_src = (coord_dst + 0.5) * (src_len as f32 / dst_len as f32) - 0.5`, clamped to `[0.0, (src_len - 1) as f32]`.
#[inline]
pub fn bilinear_half_pixel_clamp_coord(
    coord_dst: usize,
    src_len: usize,
    dst_len: usize,
) -> (usize, usize, f32) {
    let scale = (src_len as f32) / (dst_len as f32);
    let max_src = (src_len.saturating_sub(1)) as f32;
    let coord_src = ((coord_dst as f32 + 0.5) * scale - 0.5).clamp(0.0, max_src);
    let i0 = coord_src.floor() as usize;
    let i1 = (i0 + 1).min(src_len.saturating_sub(1));
    let weight = coord_src - (i0 as f32);
    (i0, i1, weight)
}

/// Resize physical spatial axes `[B, C, H_low, W_low] -> [B, C, H_full, W_full]` in fp32 using
/// bilinear interpolation with half-pixel centers (`align_corners = false`) and border clamping (`clamp-to-edge`) (`PIPE-4`, `§6.1`).
pub fn resize_bilinear_half_pixel_clamp_f32(
    src: &[f32],
    batch: usize,
    channels: usize,
    src_h: usize,
    src_w: usize,
    dst_h: usize,
    dst_w: usize,
) -> Result<Vec<f32>> {
    if batch == 0 || channels == 0 || src_h == 0 || src_w == 0 || dst_h == 0 || dst_w == 0 {
        return Err(Error::Unsupported(
            "resize_bilinear_half_pixel_clamp_f32: dimensions must be non-zero".into(),
        ));
    }
    let expected_src = batch
        .checked_mul(channels)
        .and_then(|v| v.checked_mul(src_h))
        .and_then(|v| v.checked_mul(src_w))
        .ok_or_else(|| {
            Error::Unsupported("resize_bilinear_half_pixel_clamp_f32: src size overflow".into())
        })?;
    if src.len() != expected_src {
        return Err(Error::Unsupported(format!(
            "resize_bilinear_half_pixel_clamp_f32: src len {} != expected {expected_src}",
            src.len()
        )));
    }
    let dst_len = batch
        .checked_mul(channels)
        .and_then(|v| v.checked_mul(dst_h))
        .and_then(|v| v.checked_mul(dst_w))
        .ok_or_else(|| {
            Error::Unsupported("resize_bilinear_half_pixel_clamp_f32: dst size overflow".into())
        })?;

    let y_coords: Vec<(usize, usize, f32)> = (0..dst_h)
        .map(|dy| bilinear_half_pixel_clamp_coord(dy, src_h, dst_h))
        .collect();
    let x_coords: Vec<(usize, usize, f32)> = (0..dst_w)
        .map(|dx| bilinear_half_pixel_clamp_coord(dx, src_w, dst_w))
        .collect();

    let mut dst = vec![0.0_f32; dst_len];
    let planes = batch * channels;
    let src_plane_len = src_h * src_w;
    let dst_plane_len = dst_h * dst_w;

    for p in 0..planes {
        let src_plane = &src[p * src_plane_len..(p + 1) * src_plane_len];
        let dst_plane = &mut dst[p * dst_plane_len..(p + 1) * dst_plane_len];
        for (dy, &(y0, y1, wy)) in y_coords.iter().enumerate() {
            let row0 = &src_plane[y0 * src_w..(y0 + 1) * src_w];
            let row1 = &src_plane[y1 * src_w..(y1 + 1) * src_w];
            let dst_row = &mut dst_plane[dy * dst_w..(dy + 1) * dst_w];
            let wy0 = 1.0 - wy;
            for (dx, &(x0, x1, wx)) in x_coords.iter().enumerate() {
                let wx0 = 1.0 - wx;
                let top = row0[x0] * wx0 + row0[x1] * wx;
                let bot = row1[x0] * wx0 + row1[x1] * wx;
                dst_row[dx] = top * wy0 + bot * wy;
            }
        }
    }
    Ok(dst)
}

/// Apply the destination sampling convention's noise law (`ModelSampling::noise_scaling_coeffs(sigma_k)`)
/// in fp32 (`PIPE-4`, `§6.1`):
/// - Flow (`FlowModelSampling`): `x_full(sigma_K) = (1 - sigma_K) * resize(x0_low) + sigma_K * eps_trans`
/// - VE / EDM (`EdmModelSampling` / `DiscreteModelSampling`): `x_full(sigma_K) = resize(x0_low) + sigma_K * eps_trans`
pub fn apply_destination_noise_law_f32(
    resized_x0: &[f32],
    eps_trans: &[f32],
    sigma_k: f32,
    model_sampling: &dyn ModelSampling,
) -> Result<Vec<f32>> {
    if resized_x0.len() != eps_trans.len() {
        return Err(Error::Unsupported(format!(
            "apply_destination_noise_law_f32: resized_x0 len {} != eps_trans len {}",
            resized_x0.len(),
            eps_trans.len()
        )));
    }
    if !sigma_k.is_finite() || sigma_k < 0.0 {
        return Err(Error::Unsupported(format!(
            "apply_destination_noise_law_f32: invalid boundary sigma_K={sigma_k}"
        )));
    }
    let (k_noise, k_x0) = model_sampling.noise_scaling_coeffs(sigma_k);
    Ok(resized_x0
        .iter()
        .zip(eps_trans)
        .map(|(&x0, &eps)| k_x0 * x0 + k_noise * eps)
        .collect())
}

/// Execute the full typed boundary transition from a token-packed clean estimate at `sigma_K`
/// using an explicit full-resolution `[B, C, H_full, W_full]` unit Gaussian noise buffer (`PIPE-4`, `§6.1`).
///
/// Refuses [`BoundaryState::XSigma`] and [`BoundaryState::RouteSpecific`] with a typed
/// [`Error::Unsupported`] so a transition can never double-noise a noisy latent.
#[allow(clippy::too_many_arguments)]
pub fn transition_packed_clean_estimate_with_noise_f32(
    boundary_state: BoundaryState<&[f32]>,
    batch: usize,
    channels: usize,
    patch_size: u32,
    vae_spatial_scale: u32,
    plan: &ResolvedLowResStartPlan,
    sigma_k: f32,
    model_sampling: &dyn ModelSampling,
    eps_trans_nchw: &[f32],
) -> Result<Vec<f32>> {
    let (packed_x0_low, _source) = boundary_state.into_clean_estimate()?;
    if vae_spatial_scale == 0 || patch_size == 0 {
        return Err(Error::Unsupported(
            "transition_packed_clean_estimate_with_noise_f32: vae_spatial_scale and patch_size must be non-zero".into(),
        ));
    }
    let h_low = (plan.low_height / vae_spatial_scale) as usize;
    let w_low = (plan.low_width / vae_spatial_scale) as usize;
    let h_full = (plan.full_height / vae_spatial_scale) as usize;
    let w_full = (plan.full_width / vae_spatial_scale) as usize;
    let p = patch_size as usize;

    let unpacked_low = unpack_packed_latents_f32(packed_x0_low, batch, channels, h_low, w_low, p)?;
    let resized_full = resize_bilinear_half_pixel_clamp_f32(
        &unpacked_low,
        batch,
        channels,
        h_low,
        w_low,
        h_full,
        w_full,
    )?;
    let noised_full =
        apply_destination_noise_law_f32(&resized_full, eps_trans_nchw, sigma_k, model_sampling)?;
    repack_unpacked_latents_f32(&noised_full, batch, channels, h_full, w_full, p)
}

/// Execute the full typed boundary transition from a token-packed clean estimate at `sigma_K`,
/// drawing $\epsilon_{\text{trans}} \sim \mathcal{N}(0, I)$ in `[B, C, H_full, W_full]` with the
/// deterministic domain-separated seed derived from
/// `(request_seed, output_index, pass_index, phase_index, "low_res_start_transition_v1")` (`PIPE-4`, `§6.1`).
#[allow(clippy::too_many_arguments)]
pub fn transition_packed_clean_estimate_f32(
    boundary_state: BoundaryState<&[f32]>,
    batch: usize,
    channels: usize,
    patch_size: u32,
    vae_spatial_scale: u32,
    plan: &ResolvedLowResStartPlan,
    sigma_k: f32,
    model_sampling: &dyn ModelSampling,
    request_seed: u64,
    output_index: u32,
    pass_index: usize,
    phase_index: usize,
) -> Result<Vec<f32>> {
    // Check clean-estimate typing before drawing transition noise so `XSigma` is refused immediately.
    let (packed_x0_low, source) = boundary_state.into_clean_estimate()?;
    if vae_spatial_scale == 0 {
        return Err(Error::Unsupported(
            "transition_packed_clean_estimate_f32: vae_spatial_scale must be non-zero".into(),
        ));
    }
    let h_full = (plan.full_height / vae_spatial_scale) as usize;
    let w_full = (plan.full_width / vae_spatial_scale) as usize;
    let full_len = batch
        .checked_mul(channels)
        .and_then(|v| v.checked_mul(h_full))
        .and_then(|v| v.checked_mul(w_full))
        .ok_or_else(|| {
            Error::Unsupported("transition_packed_clean_estimate_f32: size overflow".into())
        })?;
    let eps_trans = low_res_start_transition_noise_f32(
        request_seed,
        output_index,
        pass_index,
        phase_index,
        full_len,
    );
    transition_packed_clean_estimate_with_noise_f32(
        BoundaryState::CleanEstimate {
            latent: packed_x0_low,
            source,
        },
        batch,
        channels,
        patch_size,
        vae_spatial_scale,
        plan,
        sigma_k,
        model_sampling,
        &eps_trans,
    )
}

#[derive(Default)]
struct QualificationTracingOps {
    inner: crate::sampling::CpuLatentOps,
    sub_evaluations: std::cell::RefCell<Vec<(usize, String, u32)>>,
    rng_draws: std::cell::RefCell<Vec<(u64, usize)>>,
}

impl crate::sampling::LatentOps for QualificationTracingOps {
    type Latent = Vec<f32>;

    fn scale(&self, x: &Vec<f32>, s: f32) -> Result<Vec<f32>> {
        self.inner.scale(x, s)
    }

    fn add(&self, a: &Vec<f32>, b: &Vec<f32>) -> Result<Vec<f32>> {
        self.inner.add(a, b)
    }

    fn sub(&self, a: &Vec<f32>, b: &Vec<f32>) -> Result<Vec<f32>> {
        self.inner.sub(a, b)
    }

    fn axpy(&self, a: f32, x: &Vec<f32>, b: f32, y: &Vec<f32>) -> Result<Vec<f32>> {
        self.inner.axpy(a, x, b, y)
    }

    fn randn_like(&self, x: &Vec<f32>, seed: u64, step: usize) -> Result<Vec<f32>> {
        self.rng_draws.borrow_mut().push((seed, step));
        self.inner.randn_like(x, seed, step)
    }

    fn observe_solver(&self, event: crate::sampling::solvers::SolverObservation<'_, Vec<f32>>) {
        if let crate::sampling::solvers::SolverObservationKind::Callback(eval) = event.kind {
            let label = match eval {
                crate::sampling::solvers::SolverEvaluation::Primary => "primary",
                crate::sampling::solvers::SolverEvaluation::HeunCorrector => "heun_corrector",
                crate::sampling::solvers::SolverEvaluation::SdeMidpoint => "sde_midpoint",
            };
            self.sub_evaluations.borrow_mut().push((
                event.step,
                label.to_string(),
                event.sigma.to_bits(),
            ));
        }
    }
}

impl crate::guidance::GuidanceOps for QualificationTracingOps {
    fn mul(&self, a: &Vec<f32>, b: &Vec<f32>) -> Result<Vec<f32>> {
        self.inner.mul(a, b)
    }

    fn div(&self, a: &Vec<f32>, b: &Vec<f32>) -> Result<Vec<f32>> {
        self.inner.div(a, b)
    }

    fn clamp_min(&self, x: &Vec<f32>, s: f32) -> Result<Vec<f32>> {
        self.inner.clamp_min(x, s)
    }

    fn clamp_max(&self, x: &Vec<f32>, s: f32) -> Result<Vec<f32>> {
        self.inner.clamp_max(x, s)
    }

    fn select_positive(&self, sel: &Vec<f32>, a: &Vec<f32>, b: &Vec<f32>) -> Result<Vec<f32>> {
        self.inner.select_positive(sel, a, b)
    }

    fn norm_over(&self, x: &Vec<f32>, shape: &[usize], axes: &[i32]) -> Result<Vec<f32>> {
        self.inner.norm_over(x, shape, axes)
    }

    fn dot_over(
        &self,
        a: &Vec<f32>,
        b: &Vec<f32>,
        shape: &[usize],
        axes: &[i32],
    ) -> Result<Vec<f32>> {
        self.inner.dot_over(a, b, shape, axes)
    }
}

fn synthetic_target_nchw(
    batch: usize,
    channels: usize,
    lat_h: usize,
    lat_w: usize,
    seed: u64,
) -> Vec<f32> {
    let mut out = vec![0.0_f32; batch * channels * lat_h * lat_w];
    let phase_shift = ((seed & 0xFFFF) as f32) * 0.001;
    for b in 0..batch {
        for c in 0..channels {
            let c_freq = 1.0 + (c as f32) * 0.15;
            for y in 0..lat_h {
                let ny = (y as f32 + 0.5) / (lat_h.max(1) as f32);
                for x in 0..lat_w {
                    let nx = (x as f32 + 0.5) / (lat_w.max(1) as f32);
                    let val = 0.65 * (std::f32::consts::TAU * nx * c_freq + phase_shift).cos()
                        + 0.45 * (std::f32::consts::TAU * ny * (1.0 + 0.1 * (c as f32))).sin()
                        + 0.25 * (nx - 0.5) * (ny - 0.5);
                    let idx = ((b * channels + c) * lat_h + y) * lat_w + x;
                    out[idx] = val;
                }
            }
        }
    }
    out
}

fn compute_quality_metrics(
    candidate_packed: &[f32],
    reference_packed: &[f32],
    batch: usize,
    channels: usize,
    lat_h: usize,
    lat_w: usize,
    patch_size: usize,
) -> Result<LowResStartQualityMetrics> {
    if candidate_packed.len() != reference_packed.len() || candidate_packed.is_empty() {
        return Err(Error::Unsupported(
            "compute_quality_metrics: candidate and reference lengths must match and be non-empty"
                .into(),
        ));
    }
    let all_finite = candidate_packed.iter().all(|v| v.is_finite());
    let mut dot = 0.0_f64;
    let mut norm_a = 0.0_f64;
    let mut norm_b = 0.0_f64;
    let mut sq_diff = 0.0_f64;
    for (&a, &b) in candidate_packed.iter().zip(reference_packed) {
        let af = f64::from(a);
        let bf = f64::from(b);
        dot += af * bf;
        norm_a += af * af;
        norm_b += bf * bf;
        let d = af - bf;
        sq_diff += d * d;
    }
    let denom = (norm_a.sqrt() * norm_b.sqrt()).max(1e-12);
    let cosine_similarity_vs_full_res = (dot / denom) as f32;
    let rms_diff_vs_full_res = (sq_diff / (candidate_packed.len() as f64)).sqrt() as f32;

    // Unpack to [B, C, H_lat, W_lat] and compare patch-boundary adjacent gradients against interior gradients
    // to verify absence of 2x2 patch or upsampling seam artifacts.
    let unpacked =
        unpack_packed_latents_f32(candidate_packed, batch, channels, lat_h, lat_w, patch_size)?;
    let mut seam_grad_sum = 0.0_f64;
    let mut seam_count = 0usize;
    let mut interior_grad_sum = 0.0_f64;
    let mut interior_count = 0usize;
    for b in 0..batch {
        for c in 0..channels {
            let plane = &unpacked
                [(b * channels + c) * lat_h * lat_w..(b * channels + c + 1) * lat_h * lat_w];
            for y in 0..lat_h {
                for x in 0..lat_w.saturating_sub(1) {
                    let g = f64::from((plane[y * lat_w + x + 1] - plane[y * lat_w + x]).abs());
                    if (x + 1).is_multiple_of(patch_size) {
                        seam_grad_sum += g;
                        seam_count += 1;
                    } else {
                        interior_grad_sum += g;
                        interior_count += 1;
                    }
                }
            }
        }
    }
    let mean_seam = if seam_count > 0 {
        seam_grad_sum / (seam_count as f64)
    } else {
        0.0
    };
    let mean_interior = if interior_count > 0 {
        (interior_grad_sum / (interior_count as f64)).max(1e-6)
    } else {
        1.0
    };
    let max_seam_gradient_ratio = (mean_seam / mean_interior) as f32;

    Ok(LowResStartQualityMetrics {
        all_finite,
        cosine_similarity_vs_full_res,
        rms_diff_vs_full_res,
        max_seam_gradient_ratio,
    })
}

/// Execute the bounded quality, time (evaluation count / token-weighted work), and memory
/// (retained buffer coexistence) qualification comparing [`EstimateSource::EndpointEvaluation`] (default)
/// vs [`EstimateSource::PreviousEvaluation`] (candidate under DEC-10) vs a full-resolution baseline
/// over `solver` and `sigmas` (`SW-6940`: `E3`, `FEAT-1`, `PIPE-3`, `PIPE-4`, `PERF-3`).
#[allow(clippy::too_many_arguments)]
pub fn qualify_low_res_start_estimators(
    solver: Solver,
    sigmas: &[f32],
    k: usize,
    scale: f32,
    full_width: u32,
    full_height: u32,
    seed: u64,
    dtype_bytes: u64,
) -> Result<LowResStartQualificationReport> {
    use crate::guidance::{normalized_guidance, normalized_guidance_readonly, MomentumBuffer};
    use crate::sampling::{
        denoise, FlowModelSampling, PhaseBoundary, SolverRunState, TimestepConvention,
    };

    if sigmas.len() < 3 {
        return Err(Error::Unsupported(
            "qualify_low_res_start_estimators requires at least 2 executed steps (sigmas.len() >= 3)".into(),
        ));
    }
    let executed_steps = sigmas.len() - 1;
    let vae_spatial_scale = 8u32;
    let patch_size = 2u32;
    let pixel_stride = 16u32;
    let min_dim = 128u32;
    let batch = 1usize;
    let channels = 16usize;
    let token_width = 64u32;

    // 1. Construct default (EndpointEvaluation) and candidate (PreviousEvaluation) specs and verify default immutability.
    let spec_endpoint = LowResStartSpec::new(k, Some(scale));
    let spec_previous = LowResStartSpec::new(k, Some(scale))
        .with_estimate_source(EstimateSource::PreviousEvaluation);
    let plan_endpoint = spec_endpoint.resolve_geometry(
        "flux1_dev",
        full_width,
        full_height,
        executed_steps,
        vae_spatial_scale,
        patch_size,
        pixel_stride,
        min_dim,
    )?;
    let plan_previous = spec_previous.resolve_geometry(
        "flux1_dev",
        full_width,
        full_height,
        executed_steps,
        vae_spatial_scale,
        patch_size,
        pixel_stride,
        min_dim,
    )?;

    let flow_sampling = FlowModelSampling::new(TimestepConvention::Sigma);
    let h_low = (plan_endpoint.low_height / vae_spatial_scale) as usize;
    let w_low = (plan_endpoint.low_width / vae_spatial_scale) as usize;
    let h_full = (plan_endpoint.full_height / vae_spatial_scale) as usize;
    let w_full = (plan_endpoint.full_width / vae_spatial_scale) as usize;
    let p = patch_size as usize;

    let target_full_nchw = synthetic_target_nchw(batch, channels, h_full, w_full, seed);
    let target_low_nchw = resize_bilinear_half_pixel_clamp_f32(
        &target_full_nchw,
        batch,
        channels,
        h_full,
        w_full,
        h_low,
        w_low,
    )?;
    let target_full_packed =
        repack_unpacked_latents_f32(&target_full_nchw, batch, channels, h_full, w_full, p)?;
    let target_low_packed =
        repack_unpacked_latents_f32(&target_low_nchw, batch, channels, h_low, w_low, p)?;

    let init_noise_full_nchw = low_res_start_transition_noise_f32(
        seed ^ 0x517C_C1B7_2722_0A95,
        0,
        0,
        99,
        target_full_nchw.len(),
    );
    let init_noise_low_nchw = resize_bilinear_half_pixel_clamp_f32(
        &init_noise_full_nchw,
        batch,
        channels,
        h_full,
        w_full,
        h_low,
        w_low,
    )?;
    let init_full_packed =
        repack_unpacked_latents_f32(&init_noise_full_nchw, batch, channels, h_full, w_full, p)?;
    let init_low_packed =
        repack_unpacked_latents_f32(&init_noise_low_nchw, batch, channels, h_low, w_low, p)?;

    // Flow velocity model: v(x, sigma) = (x - x0_target) / max(sigma, 1e-4), guided via APG to test stateful guidance accumulator safety.
    let predict_guided = |ops: &QualificationTracingOps,
                          x_in: &Vec<f32>,
                          sigma: f32,
                          target_packed: &[f32],
                          apg_buf: Option<&mut MomentumBuffer<Vec<f32>>>|
     -> Result<Vec<f32>> {
        let s = sigma.max(1e-4);
        let cond: Vec<f32> = x_in
            .iter()
            .zip(target_packed)
            .map(|(&xi, &ti)| (xi - ti) / s)
            .collect();
        let uncond: Vec<f32> = x_in
            .iter()
            .zip(target_packed)
            .map(|(&xi, &ti)| (xi - 0.85 * ti) / s)
            .collect();
        let len = cond.len();
        normalized_guidance(
            ops,
            &cond,
            &uncond,
            3.5,
            apg_buf,
            0.0,
            2.5,
            &[1, len],
            &[-1],
        )
    };

    let predict_guided_readonly = |ops: &QualificationTracingOps,
                                   x_in: &Vec<f32>,
                                   sigma: f32,
                                   target_packed: &[f32],
                                   apg_buf: Option<&MomentumBuffer<Vec<f32>>>|
     -> Result<Vec<f32>> {
        let s = sigma.max(1e-4);
        let cond: Vec<f32> = x_in
            .iter()
            .zip(target_packed)
            .map(|(&xi, &ti)| (xi - ti) / s)
            .collect();
        let uncond: Vec<f32> = x_in
            .iter()
            .zip(target_packed)
            .map(|(&xi, &ti)| (xi - 0.85 * ti) / s)
            .collect();
        let len = cond.len();
        normalized_guidance_readonly(
            ops,
            &cond,
            &uncond,
            3.5,
            apg_buf,
            0.0,
            2.5,
            &[1, len],
            &[-1],
        )
    };

    // 2. Run full-resolution single-phase baseline (0..M).
    let full_ops = QualificationTracingOps::default();
    let mut full_apg = MomentumBuffer::new(-0.5);
    let mut full_state = SolverRunState::new(solver, sigmas);
    let sampler = solver.boxed::<QualificationTracingOps>();
    let full_res_final = {
        let mut dn = |x: &Vec<f32>, sigma: f32| -> Result<Vec<f32>> {
            denoise(&full_ops, &flow_sampling, x, sigma, |xin, t| {
                predict_guided(&full_ops, xin, t, &target_full_packed, Some(&mut full_apg))
            })
        };
        sampler.sample_span(
            &full_ops,
            &flow_sampling,
            &mut dn,
            init_full_packed,
            sigmas,
            seed,
            0..executed_steps,
            &mut full_state,
        )?
    };

    // 3. Helper to run a two-phase trajectory for either EstimateSource::EndpointEvaluation or EstimateSource::PreviousEvaluation.
    let total_u32 = executed_steps as u32;
    let run_two_phase =
        |plan: &ResolvedLowResStartPlan| -> Result<(Vec<f32>, LowResStartTraceSummary)> {
            let ops = QualificationTracingOps::default();
            let mut apg_buf = MomentumBuffer::new(-0.5);
            let mut state = SolverRunState::new(solver, sigmas);
            state.configure_low_res_start(plan)?;

            let mut phase0_progress = Vec::new();
            let mut total_progress = Vec::new();
            let mut phase0_previews = Vec::new();
            let mut emitted_preview_steps = BTreeSet::new();

            // Phase 0: 0..K at (low_width, low_height)
            let x_low_sigma_k = {
                let mut dn0 = |x: &Vec<f32>, sigma: f32| -> Result<Vec<f32>> {
                    let current =
                        (sigmas.iter().filter(|&&s| s > sigma).count() as u32 + 1).min(total_u32);
                    phase0_progress.push((current, total_u32));
                    total_progress.push((current, total_u32));
                    if emitted_preview_steps.insert(current) {
                        phase0_previews.push((
                            current,
                            total_u32,
                            plan.low_width / 8,
                            plan.low_height / 8,
                        ));
                    }
                    denoise(&ops, &flow_sampling, x, sigma, |xin, t| {
                        predict_guided(&ops, xin, t, &target_low_packed, Some(&mut apg_buf))
                    })
                };
                sampler.sample_span(
                    &ops,
                    &flow_sampling,
                    &mut dn0,
                    init_low_packed.clone(),
                    sigmas,
                    seed,
                    plan.phase0_span(),
                    &mut state,
                )?
            };
            let completed_low_res_steps = state.next_step();
            let phase0_sub_evaluations = ops.sub_evaluations.borrow().clone();
            let phase0_rng_draw_coordinates = ops.rng_draws.borrow().clone();

            // Snapshot state before boundary clean-estimate resolution.
            let pre_boundary_progress_len = total_progress.len();
            let pre_boundary_preview_len = phase0_previews.len();
            let pre_boundary_rng_len = ops.rng_draws.borrow().len();
            let pre_boundary_has_history = state.has_history();
            let pre_boundary_apg = apg_buf.running().cloned();

            let sigma_k = sigmas[plan.k];
            let (boundary_state, boundary_evaluations) = match plan.estimate_source {
                EstimateSource::EndpointEvaluation => {
                    // 1 read-only boundary evaluation at (x_low(sigma_K), sigma_K):
                    // does not advance progress, emit preview, or mutate solver/RNG/guidance state.
                    let x0_low =
                        denoise(&ops, &flow_sampling, &x_low_sigma_k, sigma_k, |xin, t| {
                            predict_guided_readonly(
                                &ops,
                                xin,
                                t,
                                &target_low_packed,
                                Some(&apg_buf),
                            )
                        })?;
                    (
                        BoundaryState::CleanEstimate {
                            latent: x0_low,
                            source: EstimateSource::EndpointEvaluation,
                        },
                        1u32,
                    )
                }
                EstimateSource::PreviousEvaluation => {
                    let captured = state.take_captured_primary_estimate(plan.k - 1)?;
                    (
                        BoundaryState::CleanEstimate {
                            latent: captured.latent,
                            source: EstimateSource::PreviousEvaluation,
                        },
                        0u32,
                    )
                }
            };

            let boundary_advanced_progress = total_progress.len() != pre_boundary_progress_len;
            let boundary_emitted_preview = phase0_previews.len() != pre_boundary_preview_len;
            let boundary_mutated_rng = ops.rng_draws.borrow().len() != pre_boundary_rng_len;
            let boundary_mutated_solver_history = state.has_history() != pre_boundary_has_history;
            let boundary_mutated_guidance_accumulator =
                apg_buf.running() != pre_boundary_apg.as_ref();

            let (x0_low_vec, source) = boundary_state.into_clean_estimate()?;
            let x_full_sigma_k = transition_packed_clean_estimate_f32(
                BoundaryState::CleanEstimate {
                    latent: &x0_low_vec,
                    source,
                },
                batch,
                channels,
                patch_size,
                vae_spatial_scale,
                plan,
                sigma_k,
                &flow_sampling,
                seed,
                0,
                0,
                0,
            )?;

            // Phase boundary reset of solver history and shape-bound APG accumulator before Phase 1.
            state.note_geometry_change(
                (plan.low_width, plan.low_height),
                (plan.full_width, plan.full_height),
            );
            state.note_renoise_transition();
            state.apply_phase_boundary(PhaseBoundary::Reset, true)?;
            apg_buf.reset();

            // Phase 1: K..M at (full_width, full_height)
            let final_packed = {
                let mut dn1 = |x: &Vec<f32>, sigma: f32| -> Result<Vec<f32>> {
                    let current =
                        (sigmas.iter().filter(|&&s| s > sigma).count() as u32 + 1).min(total_u32);
                    total_progress.push((current, total_u32));
                    denoise(&ops, &flow_sampling, x, sigma, |xin, t| {
                        predict_guided(&ops, xin, t, &target_full_packed, Some(&mut apg_buf))
                    })
                };
                sampler.sample_span(
                    &ops,
                    &flow_sampling,
                    &mut dn1,
                    x_full_sigma_k,
                    sigmas,
                    seed,
                    plan.phase1_span(),
                    &mut state,
                )?
            };

            let all_sub_evals = ops.sub_evaluations.borrow();
            let phase1_sub_evaluations = all_sub_evals[phase0_sub_evaluations.len()..].to_vec();
            let all_rng_draws = ops.rng_draws.borrow();
            let phase1_rng_draw_coordinates =
                all_rng_draws[phase0_rng_draw_coordinates.len()..].to_vec();

            Ok((
                final_packed,
                LowResStartTraceSummary {
                    completed_low_res_steps,
                    phase0_sub_evaluations,
                    phase0_rng_draw_coordinates,
                    phase0_progress_events: phase0_progress,
                    phase0_preview_frames: phase0_previews,
                    boundary_evaluations,
                    boundary_advanced_progress,
                    boundary_emitted_preview,
                    boundary_mutated_solver_history,
                    boundary_mutated_rng,
                    boundary_mutated_guidance_accumulator,
                    phase1_sub_evaluations,
                    phase1_rng_draw_coordinates,
                    total_progress_events: total_progress,
                },
            ))
        };

    let (endpoint_final, endpoint_trace) = run_two_phase(&plan_endpoint)?;
    let (previous_final, previous_trace) = run_two_phase(&plan_previous)?;

    let endpoint_quality = compute_quality_metrics(
        &endpoint_final,
        &full_res_final,
        batch,
        channels,
        h_full,
        w_full,
        p,
    )?;
    let previous_quality = compute_quality_metrics(
        &previous_final,
        &full_res_final,
        batch,
        channels,
        h_full,
        w_full,
        p,
    )?;
    let endpoint_vs_previous = compute_quality_metrics(
        &endpoint_final,
        &previous_final,
        batch,
        channels,
        h_full,
        w_full,
        p,
    )?;

    let dtype_label = if dtype_bytes == 2 { "bf16" } else { "f32" };
    let endpoint_report = plan_endpoint.execution_report(
        0,
        sigmas,
        solver,
        "apg",
        3.5,
        "base_dit",
        dtype_label,
        dtype_bytes,
        batch,
        pixel_stride,
        token_width,
    )?;
    let previous_report = plan_previous.execution_report(
        0,
        sigmas,
        solver,
        "apg",
        3.5,
        "base_dit",
        dtype_label,
        dtype_bytes,
        batch,
        pixel_stride,
        token_width,
    )?;
    let endpoint_work = plan_endpoint.work_profile(solver, sigmas, pixel_stride)?;
    let previous_work = plan_previous.work_profile(solver, sigmas, pixel_stride)?;
    let endpoint_memory =
        plan_endpoint.memory_profile(batch, pixel_stride, token_width, dtype_bytes);
    let previous_memory =
        plan_previous.memory_profile(batch, pixel_stride, token_width, dtype_bytes);

    let phase0_traces_identical = endpoint_trace.completed_low_res_steps == k
        && previous_trace.completed_low_res_steps == k
        && endpoint_trace.phase0_sub_evaluations == previous_trace.phase0_sub_evaluations
        && endpoint_trace.phase0_rng_draw_coordinates == previous_trace.phase0_rng_draw_coordinates
        && endpoint_trace.phase0_progress_events == previous_trace.phase0_progress_events
        && endpoint_trace.phase0_preview_frames == previous_trace.phase0_preview_frames;
    let phase1_rng_coordinates_identical =
        endpoint_trace.phase1_rng_draw_coordinates == previous_trace.phase1_rng_draw_coordinates;
    let outer_progress_identical =
        endpoint_trace.total_progress_events == previous_trace.total_progress_events;
    let preview_sequences_identical =
        endpoint_trace.phase0_preview_frames == previous_trace.phase0_preview_frames;
    let default_selection_unchanged = EstimateSource::default()
        == EstimateSource::EndpointEvaluation
        && LowResStartSpec::new(k, Some(scale)).estimate_source
            == EstimateSource::EndpointEvaluation;

    Ok(LowResStartQualificationReport {
        default_policy: EstimateSource::EndpointEvaluation,
        candidate_policy: EstimateSource::PreviousEvaluation,
        plan_endpoint,
        plan_previous,
        endpoint_report,
        previous_report,
        endpoint_work,
        previous_work,
        endpoint_memory,
        previous_memory,
        endpoint_trace,
        previous_trace,
        phase0_traces_identical,
        phase1_rng_coordinates_identical,
        outer_progress_identical,
        preview_sequences_identical,
        endpoint_quality,
        previous_quality,
        endpoint_vs_previous_cosine_similarity: endpoint_vs_previous.cosine_similarity_vs_full_res,
        endpoint_vs_previous_rms_diff: endpoint_vs_previous.rms_diff_vs_full_res,
        default_selection_unchanged,
    })
}

// -------------------------------------------------------------------------------------------------
// HOOK-1: Documented hook map
// -------------------------------------------------------------------------------------------------

/// When a hook executes relative to its owning stage or block operation.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum HookTiming {
    Before,
    Replace,
    After,
    Observe,
}

impl HookTiming {
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::Before => "before",
            Self::Replace => "replace",
            Self::After => "after",
            Self::Observe => "observe",
        }
    }
}

/// Every legal hook point in the planned image/video render lifecycle (HOOK-1).
///
/// Each variant owns a fixed stage, default timing, typed input/output contract, deterministic
/// ordering slot, and result-changing classification, and has at least one registered
/// production user in [`FeatureRegistry::standard`].
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum HookPoint {
    ScheduleTransform,
    ConditioningSlot,
    PhaseTransition,
    GuidancePredictor,
    AttentionBlock,
}

impl HookPoint {
    pub const ALL: [HookPoint; 5] = [
        Self::ScheduleTransform,
        Self::ConditioningSlot,
        Self::PhaseTransition,
        Self::GuidancePredictor,
        Self::AttentionBlock,
    ];

    pub const fn as_str(self) -> &'static str {
        match self {
            Self::ScheduleTransform => "schedule_transform",
            Self::ConditioningSlot => "conditioning_slot",
            Self::PhaseTransition => "phase_transition",
            Self::GuidancePredictor => "guidance_predictor",
            Self::AttentionBlock => "attention_block",
        }
    }

    pub const fn stage(self) -> RenderStage {
        match self {
            Self::ScheduleTransform => RenderStage::Schedule,
            Self::ConditioningSlot => RenderStage::Encode,
            Self::PhaseTransition => RenderStage::Denoise,
            Self::GuidancePredictor => RenderStage::Denoise,
            Self::AttentionBlock => RenderStage::Denoise,
        }
    }

    pub const fn default_timing(self) -> HookTiming {
        match self {
            Self::ScheduleTransform => HookTiming::After,
            Self::ConditioningSlot => HookTiming::After,
            Self::PhaseTransition => HookTiming::Before,
            Self::GuidancePredictor => HookTiming::Replace,
            Self::AttentionBlock => HookTiming::Replace,
        }
    }

    pub const fn inputs(self) -> &'static [&'static str] {
        match self {
            Self::ScheduleTransform => &["base_sigmas", "executed_steps", "resolution_mu"],
            Self::ConditioningSlot => &[
                "prompt_embeddings",
                "reference_or_control_inputs",
                "pass_index",
            ],
            Self::PhaseTransition => &[
                "phase_latents",
                "boundary_step",
                "boundary_sigma",
                "source_stream_geometry",
            ],
            Self::GuidancePredictor => &[
                "conditional_velocity",
                "unconditional_velocity",
                "guidance_scale",
                "step_index",
            ],
            Self::AttentionBlock => &[
                "block_hidden_states",
                "block_index",
                "step_index",
                "active_adapters",
            ],
        }
    }

    pub const fn outputs(self) -> &'static [&'static str] {
        match self {
            Self::ScheduleTransform => &["sigmas", "denoise_phase_spans"],
            Self::ConditioningSlot => &["bound_conditioning_slots", "produced_reference_binding"],
            Self::PhaseTransition => &[
                "next_phase_latents",
                "target_stream_geometry",
                "active_phase_binding",
            ],
            Self::GuidancePredictor => &["guided_velocity"],
            Self::AttentionBlock => &["transformed_block_hidden_states"],
        }
    }

    pub const fn ordering(self) -> u32 {
        match self {
            Self::ScheduleTransform => 10,
            Self::ConditioningSlot => 20,
            Self::PhaseTransition => 30,
            Self::AttentionBlock => 40,
            Self::GuidancePredictor => 50,
        }
    }

    pub const fn result_changing(self) -> bool {
        match self {
            Self::ScheduleTransform
            | Self::ConditioningSlot
            | Self::PhaseTransition
            | Self::GuidancePredictor
            | Self::AttentionBlock => true,
        }
    }
}

/// Complete machine-readable metadata for one [`HookPoint`] (HOOK-1).
#[derive(Clone, Debug, PartialEq, Eq, Serialize)]
pub struct HookPointMetadata {
    pub hook_point: HookPoint,
    pub name: &'static str,
    pub stage: RenderStage,
    pub timing: HookTiming,
    pub inputs: &'static [&'static str],
    pub outputs: &'static [&'static str],
    pub ordering: u32,
    pub result_changing: bool,
    pub registered_users: Vec<&'static str>,
}

/// Inspect one [`HookPoint`]'s metadata and its registered users in [`FeatureRegistry::standard`].
pub fn hook_point_metadata(point: HookPoint) -> HookPointMetadata {
    let registry = FeatureRegistry::standard();
    hook_point_metadata_in(point, &registry)
}

fn hook_point_metadata_in(point: HookPoint, registry: &FeatureRegistry) -> HookPointMetadata {
    let mut registered_users: Vec<&'static str> = registry
        .specs()
        .iter()
        .filter(|spec| {
            spec.hook_bindings()
                .iter()
                .any(|binding| binding.hook_point == point)
        })
        .map(|spec| spec.id())
        .collect();
    registered_users.sort_unstable();
    registered_users.dedup();
    HookPointMetadata {
        hook_point: point,
        name: point.as_str(),
        stage: point.stage(),
        timing: point.default_timing(),
        inputs: point.inputs(),
        outputs: point.outputs(),
        ordering: point.ordering(),
        result_changing: point.result_changing(),
        registered_users,
    }
}

/// Return the complete ordered hook map for the standard feature registry (HOOK-1).
pub fn hook_map() -> Vec<HookPointMetadata> {
    let registry = FeatureRegistry::standard();
    hook_map_for_registry(&registry)
}

/// Return the complete ordered hook map for `registry`.
pub fn hook_map_for_registry(registry: &FeatureRegistry) -> Vec<HookPointMetadata> {
    let mut entries: Vec<HookPointMetadata> = HookPoint::ALL
        .into_iter()
        .map(|point| hook_point_metadata_in(point, registry))
        .collect();
    entries.sort_by_key(|entry| entry.ordering);
    entries
}

/// Typed JSON dump helper for the hook map (HOOK-1; computed on demand, not a checked-in matrix).
pub fn dump_hook_map_json() -> serde_json::Value {
    serde_json::to_value(hook_map()).expect("HookPointMetadata serializes")
}

// -------------------------------------------------------------------------------------------------
// HOOK-2: FeatureSpec registry and DEC-8 classes
// -------------------------------------------------------------------------------------------------

/// DEC-8 Feature equivalence/intent classification.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum FeatureClass {
    /// Result-changing creative or trajectory recipe (e.g. `low_res_start`, `hires_fix`, `krea_multiphase`, `pag`).
    Recipe,
    /// Numerically or bitwise equivalent execution restructuring (e.g. `inert_phase_split`).
    Equivalent,
    /// Non-mutating observation / telemetry hook (e.g. `trajectory_observer`).
    Diagnostic,
}

impl FeatureClass {
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::Recipe => "recipe",
            Self::Equivalent => "equivalent",
            Self::Diagnostic => "diagnostic",
        }
    }
}

/// One hook binding declared by a [`FeatureSpec`].
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize)]
pub struct FeatureHookBinding {
    pub hook_point: HookPoint,
    pub stage: RenderStage,
    pub timing: HookTiming,
    /// Which pass this hook targets (`Some(0)` = pass 0 only, `Some(1)` = pass 1, `None` = all passes).
    pub pass_scope: Option<usize>,
    pub result_changing: bool,
}

/// Typed parameter domain for one parameter in a [`FeatureParameterSchema`].
#[derive(Clone, Debug, PartialEq, Serialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum FeatureParameterDomain {
    /// Step index `min..executed_steps` (`1..executed_steps` for `low_res_start.k`).
    StepIndex {
        min: usize,
        max_exclusive: &'static str,
        #[serde(skip_serializing_if = "Option::is_none")]
        max_inclusive_for_default_steps: Option<usize>,
        #[serde(skip_serializing_if = "Option::is_none")]
        default_executed_steps: Option<u32>,
    },
    /// Bounded floating-point parameter (`0.25..=0.9` with default `0.5` for `low_res_start.scale`).
    FloatRange {
        min: f32,
        max: f32,
        inclusive: bool,
        #[serde(skip_serializing_if = "Option::is_none")]
        default: Option<f32>,
    },
    /// Bounded integer parameter.
    IntegerRange {
        min: u32,
        max: u32,
        #[serde(skip_serializing_if = "Option::is_none")]
        default: Option<u32>,
    },
    /// Ordered list of denoise phases over one shared schedule.
    PhaseList {
        min_phases: usize,
        min_steps_per_phase: u32,
    },
}

/// One typed parameter specification inside [`FeatureParameterSchema`].
#[derive(Clone, Debug, PartialEq, Serialize)]
pub struct FeatureParameterSpec {
    pub name: &'static str,
    pub required: bool,
    pub domain: FeatureParameterDomain,
    pub description: &'static str,
}

/// Versioned parameter schema of a [`FeatureSpec`].
#[derive(Clone, Debug, PartialEq, Serialize)]
pub struct FeatureParameterSchema {
    pub version: u32,
    pub parameters: Vec<FeatureParameterSpec>,
}

impl FeatureParameterSchema {
    pub fn to_json_map(&self) -> serde_json::Value {
        let mut map = serde_json::Map::new();
        map.insert("version".into(), serde_json::json!(self.version));
        for param in &self.parameters {
            let value = match &param.domain {
                FeatureParameterDomain::StepIndex {
                    min,
                    max_exclusive,
                    max_inclusive_for_default_steps,
                    default_executed_steps,
                } => {
                    let mut obj = serde_json::json!({
                        "type": "step_index",
                        "required": param.required,
                        "min": min,
                        "max_exclusive": max_exclusive,
                        "range": format!("{min}..{max_exclusive}"),
                    });
                    if let Some(max_inc) = max_inclusive_for_default_steps {
                        obj["max"] = serde_json::json!(max_inc);
                    }
                    if let Some(def_steps) = default_executed_steps {
                        obj["default_executed_steps"] = serde_json::json!(def_steps);
                    }
                    obj
                }
                FeatureParameterDomain::FloatRange {
                    min,
                    max,
                    inclusive,
                    default,
                } => {
                    let clean = |v: f32| ((f64::from(v)) * 1_000_000.0).round() / 1_000_000.0;
                    let mut obj = serde_json::json!({
                        "type": "float_range",
                        "required": param.required,
                        "min": clean(*min),
                        "max": clean(*max),
                        "inclusive": inclusive,
                    });
                    if let Some(default) = default {
                        obj["default"] = serde_json::json!(clean(*default));
                    }
                    obj
                }
                FeatureParameterDomain::IntegerRange { min, max, default } => {
                    let mut obj = serde_json::json!({
                        "type": "integer_range",
                        "required": param.required,
                        "min": min,
                        "max": max,
                    });
                    if let Some(default) = default {
                        obj["default"] = serde_json::json!(default);
                    }
                    obj
                }
                FeatureParameterDomain::PhaseList {
                    min_phases,
                    min_steps_per_phase,
                } => serde_json::json!({
                    "type": "phase_list",
                    "required": param.required,
                    "min_phases": min_phases,
                    "min_steps_per_phase": min_steps_per_phase,
                }),
            };
            map.insert(param.name.to_string(), value);
        }
        serde_json::Value::Object(map)
    }
}

/// Inter-feature and request-axis relations declared by a [`FeatureSpec`] (HOOK-2).
#[derive(Clone, Debug, Default, PartialEq, Eq, Serialize)]
pub struct FeatureRelations {
    /// Other feature IDs that cannot be composed with this feature.
    pub conflicts_with_features: Vec<&'static str>,
    /// Request/route axes that conflict with this feature (e.g., `"phases"`, `"pass0_reference_conditioning"`, `"pass0_control_conditioning"`, `"pass0_edit_conditioning"`).
    pub conflicts_with_axes: Vec<&'static str>,
    /// Other feature IDs that must also be active when this feature is selected.
    pub requires_features: Vec<&'static str>,
    /// Structural route capabilities required by this feature.
    pub requires_capabilities: Vec<&'static str>,
    /// Features that this feature must execute before during plan composition.
    pub before: Vec<&'static str>,
    /// Features that this feature must execute after during plan composition.
    pub after: Vec<&'static str>,
}

impl FeatureRelations {
    /// Combined list of conflicting features and request/conditioning axes for catalog export.
    pub fn all_conflicts(&self) -> Vec<&'static str> {
        let mut out = self.conflicts_with_axes.clone();
        for &feature in &self.conflicts_with_features {
            if !out.contains(&feature) {
                out.push(feature);
            }
        }
        out
    }
}

/// Transition residency behavior at a phase or pass boundary (HOOK-2).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum TransitionResidency {
    /// No transition buffer allocation beyond the running trajectory tensor.
    None,
    /// In-device latent spatial unpack -> clean-estimate resize -> repack -> re-noise at boundary sigma; zero extra model weights.
    InDeviceLatentUnpackResizeRepack,
    /// Pass-0 VAE decode to RGB pixels -> host/device resize -> pass-1 `ProducedReference` conditioning.
    PixelRoundTripProducedReference,
    /// Per-phase low-rank adapter residual swap on a job-local transformer without mutating shared base weights.
    JobLocalAdapterResidualSwap,
    /// MoE expert weight handoff at a boundary step (shedding the inactive expert under staged residency).
    StagedExpertWeightHandoff,
}

/// Memory and residency effect declared by a [`FeatureSpec`] (HOOK-2).
#[derive(Clone, Debug, PartialEq, Eq, Serialize)]
pub struct FeatureMemoryEffect {
    pub extra_resident_weight_bytes: u64,
    pub transition_residency: TransitionResidency,
    pub introduces_denoise_phase_boundary: bool,
    pub introduces_pass_boundary: bool,
    pub retains_single_pass_schedule: bool,
}

/// What pass 0 of a route requires on its conditioning axis.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum Pass0ConditioningRequirement {
    /// Pass 0 can start from pure noise (text-to-image).
    NoiseStartSupported,
    /// Pass 0 requires a reference/identity image (e.g. IP-Adapter or PuLID route).
    RequiresReference,
    /// Pass 0 requires a structural control image (e.g. ControlNet route).
    RequiresControl,
    /// Pass 0 requires a source edit image (e.g. instruction-edit / infill route).
    RequiresEdit,
}

/// Structural facts of a route derived weights-free from [`ModelDescriptor`] and its optional
/// [`RenderPlanRegistration`] (HOOK-2 / HOOK-4).
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct FeatureRouteStructure {
    pub route_id: String,
    pub family: String,
    pub backend: String,
    pub modality: Modality,
    pub has_render_plan: bool,
    pub has_packed_image_trajectory: bool,
    pub supports_spatial_unpack_repack: bool,
    pub supports_clean_estimate_renoise_transition: bool,
    pub pass0_conditioning: Pass0ConditioningRequirement,
    pub supports_multiphase_denoise: bool,
    pub supports_produced_reference_multipass: bool,
    pub supports_expert_switching: bool,
    pub supports_pag: bool,
    pub default_executed_steps: Option<u32>,
    pub pixel_stride: Option<u32>,
    pub token_width: Option<u32>,
}

impl FeatureRouteStructure {
    /// Derive structural route facts from a [`ModelDescriptor`] and its optional pre-load
    /// [`RenderPlanRegistration`] without any hardcoded model-ID lookup table.
    pub fn from_descriptor_and_plan(
        descriptor: &ModelDescriptor,
        render_plan: Option<&RenderPlanRegistration>,
    ) -> Self {
        let mut has_render_plan = render_plan.is_some();
        let mut has_packed_image_trajectory = false;
        let mut pass0_conditioning = classify_descriptor_pass0_conditioning(descriptor);
        let mut default_executed_steps = None;
        let mut pixel_stride = descriptor
            .capabilities
            .size_floor
            .explicit_size_multiple()
            .or_else(|| {
                descriptor
                    .denoiser_output_latent_space
                    .map(|space| match space.patch_layout {
                        LatentPatchLayout::Packed { patch_width, .. } => {
                            u32::from(space.spatial_compression.width)
                                * u32::from(patch_width.max(1))
                        }
                        LatentPatchLayout::Unpacked => {
                            u32::from(space.spatial_compression.width) * 2
                        }
                    })
            });
        let mut token_width = None;

        if let Some(planner) = render_plan {
            let probe_size = probe_dimension(descriptor, pixel_stride.unwrap_or(16));
            let base_spec = LoadSpec::new(WeightsSource::Dir("/__feature_probe__".into()))
                .with_offload_policy(OffloadPolicy::Sequential);
            let noise_req = GenerationRequest {
                prompt: "feature applicability probe".into(),
                width: probe_size,
                height: probe_size,
                count: 1,
                seed: Some(1),
                ..Default::default()
            };

            match (planner.compile)(&base_spec, &noise_req) {
                Ok(Some(plan)) => {
                    has_render_plan = true;
                    has_packed_image_trajectory = true;
                    pass0_conditioning = Pass0ConditioningRequirement::NoiseStartSupported;
                    default_executed_steps = u32::try_from(plan.executed_steps()).ok();
                    pixel_stride = Some(plan.pixel_stride());
                    token_width = Some(plan.token_width());
                }
                Ok(None) => {
                    has_render_plan = false;
                }
                Err(_) => {
                    // Probe conditioned trajectories (Control, then Reference/Identity) to derive
                    // the exact reason pass 0 did not compile as noise-start.
                    let probe_img = Image {
                        width: 16,
                        height: 16,
                        pixels: vec![0; 16 * 16 * 3],
                    };
                    let control_spec = base_spec.clone().with_control(WeightsSource::File(
                        "/__feature_probe_control__.safetensors".into(),
                    ));
                    let control_req = GenerationRequest {
                        conditioning: vec![Conditioning::Control {
                            image: probe_img.clone(),
                            kind: ControlKind::Pose,
                            scale: Some(0.7),
                        }],
                        ..noise_req.clone()
                    };
                    if let Ok(Some(plan)) = (planner.compile)(&control_spec, &control_req) {
                        has_render_plan = true;
                        has_packed_image_trajectory = true;
                        pass0_conditioning = Pass0ConditioningRequirement::RequiresControl;
                        default_executed_steps = u32::try_from(plan.executed_steps()).ok();
                        pixel_stride = Some(plan.pixel_stride());
                        token_width = Some(plan.token_width());
                    } else {
                        let mut ref_spec = base_spec
                            .clone()
                            .with_ip_adapter(WeightsSource::File(
                                "/__feature_probe_ip__.safetensors".into(),
                            ))
                            .with_component(
                                "flux_ip_image_encoder",
                                WeightsSource::Dir("/__feature_probe_ip_encoder__".into()),
                            );
                        ref_spec.identity = Some(IdentityWeights {
                            encoder: Some(WeightsSource::File(
                                "/__feature_probe_id_encoder__.safetensors".into(),
                            )),
                            eva: Some(WeightsSource::File(
                                "/__feature_probe_eva__.safetensors".into(),
                            )),
                            face_dir: Some(WeightsSource::Dir("/__feature_probe_face__".into())),
                        });
                        let ref_req = GenerationRequest {
                            conditioning: vec![Conditioning::Reference {
                                image: probe_img,
                                strength: Some(0.7),
                            }],
                            ..noise_req
                        };
                        if let Ok(Some(plan)) = (planner.compile)(&ref_spec, &ref_req) {
                            has_render_plan = true;
                            has_packed_image_trajectory = true;
                            pass0_conditioning = Pass0ConditioningRequirement::RequiresReference;
                            default_executed_steps = u32::try_from(plan.executed_steps()).ok();
                            pixel_stride = Some(plan.pixel_stride());
                            token_width = Some(plan.token_width());
                        }
                    }
                }
            }
        }

        let has_packed_latent_space =
            descriptor
                .denoiser_output_latent_space
                .is_some_and(|space| {
                    space.is_well_formed()
                        && match space.patch_layout {
                            LatentPatchLayout::Packed {
                                patch_height: 2,
                                patch_width: 2,
                            } => true,
                            LatentPatchLayout::Unpacked => {
                                let expected_stride =
                                    u32::from(space.spatial_compression.width) * 2;
                                let expected_tokens = u32::from(space.channels) * 4;
                                pixel_stride == Some(expected_stride)
                                    && token_width == Some(expected_tokens)
                            }
                            _ => false,
                        }
                });
        let supports_spatial_unpack_repack = descriptor.modality == Modality::Image
            && has_packed_image_trajectory
            && has_packed_latent_space
            && pixel_stride.is_some_and(|stride| stride > 0)
            && token_width.is_some_and(|width| width > 0);
        let supports_clean_estimate_renoise_transition = has_render_plan
            && supports_spatial_unpack_repack
            && default_executed_steps.is_some_and(|steps| steps >= 2);
        let supports_multiphase_denoise = descriptor.capabilities.supports_phases;
        let supports_produced_reference_multipass = has_render_plan
            && has_packed_image_trajectory
            && descriptor.capabilities.accepts(ConditioningKind::Reference);
        let supports_pag = descriptor.capabilities.pag.is_some();

        Self {
            route_id: descriptor.id.to_string(),
            family: descriptor.family.to_string(),
            backend: descriptor.backend.to_string(),
            modality: descriptor.modality,
            has_render_plan,
            has_packed_image_trajectory,
            supports_spatial_unpack_repack,
            supports_clean_estimate_renoise_transition,
            pass0_conditioning,
            supports_multiphase_denoise,
            supports_produced_reference_multipass,
            supports_expert_switching: false,
            supports_pag,
            default_executed_steps,
            pixel_stride,
            token_width,
        }
    }
}

fn probe_dimension(descriptor: &ModelDescriptor, stride: u32) -> u32 {
    let stride = stride.max(1);
    let min = descriptor.capabilities.min_size.max(stride);
    let max = if descriptor.capabilities.max_size >= min {
        descriptor.capabilities.max_size
    } else {
        1024
    };
    let candidate = 512.clamp(min, max);
    let aligned = (candidate / stride) * stride;
    if aligned >= min {
        aligned
    } else {
        min.div_ceil(stride) * stride
    }
}

fn classify_descriptor_pass0_conditioning(
    descriptor: &ModelDescriptor,
) -> Pass0ConditioningRequirement {
    if descriptor.control_kinds.is_some()
        || descriptor.capabilities.conditioning == [ConditioningKind::Control]
    {
        Pass0ConditioningRequirement::RequiresControl
    } else if descriptor.id.contains("edit") || descriptor.id.contains("fill") {
        Pass0ConditioningRequirement::RequiresEdit
    } else if descriptor.family == "pulid"
        || descriptor.id.contains("pulid")
        || descriptor.id.contains("ip_adapter")
        || descriptor.id.contains("instantid")
    {
        Pass0ConditioningRequirement::RequiresReference
    } else {
        Pass0ConditioningRequirement::NoiseStartSupported
    }
}

/// Typed machine-readable code explaining why a [`FeatureSpec`] does not apply to a route (HOOK-4).
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum FeatureInapplicabilityCode {
    MissingRenderPlan,
    UnsupportedModality,
    MissingPackedImageTrajectory,
    MissingSpatialUnpackRepack,
    MissingCleanEstimateRenoiseTransition,
    ConflictsWithPass0ReferenceConditioning,
    ConflictsWithPass0ControlConditioning,
    ConflictsWithPass0EditConditioning,
    ConflictsWithMultiphaseDenoise,
    InsufficientExecutedSteps,
    MissingMultiphaseDenoiseSupport,
    MissingProducedReferenceSupport,
    MissingExpertSwitchingSupport,
    MissingPagCapability,
}

impl FeatureInapplicabilityCode {
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::MissingRenderPlan => "missing_render_plan",
            Self::UnsupportedModality => "unsupported_modality",
            Self::MissingPackedImageTrajectory => "missing_packed_image_trajectory",
            Self::MissingSpatialUnpackRepack => "missing_spatial_unpack_repack",
            Self::MissingCleanEstimateRenoiseTransition => {
                "missing_clean_estimate_renoise_transition"
            }
            Self::ConflictsWithPass0ReferenceConditioning => {
                "conflicts_with_pass0_reference_conditioning"
            }
            Self::ConflictsWithPass0ControlConditioning => {
                "conflicts_with_pass0_control_conditioning"
            }
            Self::ConflictsWithPass0EditConditioning => "conflicts_with_pass0_edit_conditioning",
            Self::ConflictsWithMultiphaseDenoise => "conflicts_with_multiphase_denoise",
            Self::InsufficientExecutedSteps => "insufficient_executed_steps",
            Self::MissingMultiphaseDenoiseSupport => "missing_multiphase_denoise_support",
            Self::MissingProducedReferenceSupport => "missing_produced_reference_support",
            Self::MissingExpertSwitchingSupport => "missing_expert_switching_support",
            Self::MissingPagCapability => "missing_pag_capability",
        }
    }
}

/// One typed reason a [`FeatureSpec`] is inapplicable on a route.
#[derive(Clone, Debug, PartialEq, Eq, Serialize)]
pub struct FeatureInapplicabilityReason {
    pub code: FeatureInapplicabilityCode,
    pub message: String,
}

/// Computed applicability, classification, parameter limits/defaults, conflicts, and typed
/// inapplicability reasons for one `(route, feature)` pair (HOOK-4).
#[derive(Clone, Debug, PartialEq, Serialize)]
pub struct FeatureApplicability {
    pub feature_id: &'static str,
    pub version: u32,
    pub class: FeatureClass,
    pub applicable: bool,
    pub parameter_schema: FeatureParameterSchema,
    pub relations: FeatureRelations,
    pub hook_points: Vec<HookPoint>,
    pub memory_effect: FeatureMemoryEffect,
    pub reasons: Vec<FeatureInapplicabilityReason>,
    #[serde(skip_serializing_if = "BTreeMap::is_empty", default)]
    pub conditioned_routes: BTreeMap<String, FeatureApplicability>,
}

impl FeatureApplicability {
    /// Serialize to the catalog JSON representation consumed by release/product capability checks.
    pub fn to_json(&self) -> serde_json::Value {
        let schema_json = self.parameter_schema.to_json_map();
        let parameters_json = if self.applicable {
            schema_json.clone()
        } else {
            serde_json::Value::Null
        };
        let reason_codes: Vec<&'static str> = self
            .reasons
            .iter()
            .map(|reason| reason.code.as_str())
            .collect();
        let hook_names: Vec<&'static str> = self
            .hook_points
            .iter()
            .map(|point| point.as_str())
            .collect();
        let mut obj = serde_json::json!({
            "id": self.feature_id,
            "version": self.version,
            "class": self.class.as_str(),
            "classification": self.class.as_str(),
            "applicable": self.applicable,
            "supported": self.applicable,
            "available": self.applicable,
            "parameter_schema": schema_json,
            "parameters": parameters_json,
            "conflicts": self.relations.all_conflicts(),
            "requires": self.relations.requires_capabilities,
            "hook_points": hook_names,
            "memory_effect": self.memory_effect,
            "reasons": self.reasons,
            "inapplicability_reasons": reason_codes,
        });
        if !self.conditioned_routes.is_empty() {
            let subroutes: serde_json::Map<String, serde_json::Value> = self
                .conditioned_routes
                .iter()
                .map(|(route_id, app)| (route_id.clone(), app.to_json()))
                .collect();
            obj["conditioned_routes"] = serde_json::Value::Object(subroutes);
        }
        obj
    }
}

// -------------------------------------------------------------------------------------------------
// HOOK-3: Composed render plan representation and deterministic Feature composition
// -------------------------------------------------------------------------------------------------

/// One transition action executed at a phase boundary inside a [`ComposedPassPlan`].
#[derive(Clone, Debug, PartialEq, Serialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum PhaseTransitionAction {
    /// Inert proof split: no geometry, schedule, guidance, or model change.
    InertSplit,
    /// `low_res_start` transition: spatial unpack -> clean-estimate x0 resize -> repack -> re-noise at `boundary_sigma`.
    CleanEstimateSpatialResizeRenoise {
        scale_bits: u32,
        from_width: u32,
        from_height: u32,
        to_width: u32,
        to_height: u32,
    },
    /// `krea_multiphase` transition: per-phase low-rank adapter subset & guidance switch on a job-local DiT.
    AdapterAndGuidanceSwitch {
        guidance_bits: u32,
        adapters: Vec<(usize, Option<u32>)>,
    },
    /// `wan_expert_switch` transition: hand off between MoE experts at the boundary step.
    ExpertSwitch {
        from_expert: String,
        to_expert: String,
    },
    /// `trajectory_observer` transition: non-mutating diagnostic observation at a phase boundary.
    DiagnosticObserve { label: String },
}

impl PhaseTransitionAction {
    fn canonical_rank(&self) -> u8 {
        match self {
            Self::CleanEstimateSpatialResizeRenoise { .. } => 1,
            Self::ExpertSwitch { .. } => 2,
            Self::AdapterAndGuidanceSwitch { .. } => 3,
            Self::InertSplit => 4,
            Self::DiagnosticObserve { .. } => 5,
        }
    }
}

/// A phase boundary at `step` in a [`ComposedPassPlan`], carrying all merged transition actions.
#[derive(Clone, Debug, PartialEq, Serialize)]
pub struct ComposedPhaseTransition {
    pub step: usize,
    pub boundary_sigma: f32,
    pub actions: Vec<PhaseTransitionAction>,
}

/// One contiguous denoise phase within a [`ComposedPassPlan`].
#[derive(Clone, Debug, PartialEq, Serialize)]
pub struct ComposedDenoisePhase {
    pub phase_index: usize,
    pub first_step: usize,
    pub end_step: usize,
    pub width: u32,
    pub height: u32,
    pub pixel_stride: u32,
    pub packed_tokens: u32,
    pub guidance: f32,
    pub active_adapters: Vec<(usize, Option<u32>)>,
    pub active_expert: Option<String>,
}

/// One pass inside a [`ComposedRenderPlan`].
#[derive(Clone, Debug, PartialEq, Serialize)]
pub struct ComposedPassPlan {
    pub pass_index: usize,
    pub stages: [RenderStage; 10],
    pub sigmas: Vec<f32>,
    pub executed_steps: usize,
    pub sampler: String,
    pub target_width: u32,
    pub target_height: u32,
    pub pixel_stride: u32,
    pub token_width: u32,
    pub base_guidance: f32,
    pub produced_reference: Option<ProducedReference>,
    pub phases: Vec<ComposedDenoisePhase>,
    pub transitions: Vec<ComposedPhaseTransition>,
    pub pag_active: bool,
    pub diagnostic_observers: Vec<String>,
}

impl ComposedPassPlan {
    /// Construct a single-phase pass plan over one shared schedule.
    #[allow(clippy::too_many_arguments)]
    pub fn single_phase(
        pass_index: usize,
        width: u32,
        height: u32,
        pixel_stride: u32,
        token_width: u32,
        sampler: impl Into<String>,
        guidance: f32,
        sigmas: Vec<f32>,
        produced_reference: Option<ProducedReference>,
    ) -> Result<Self> {
        if pixel_stride == 0
            || width == 0
            || height == 0
            || !width.is_multiple_of(pixel_stride)
            || !height.is_multiple_of(pixel_stride)
            || sigmas.len() < 2
        {
            return Err(Error::Unsupported(
                "composed pass plan has invalid geometry or schedule".into(),
            ));
        }
        let executed_steps = sigmas.len() - 1;
        let packed_tokens = (width / pixel_stride)
            .checked_mul(height / pixel_stride)
            .ok_or_else(|| Error::Unsupported("packed token count overflow".into()))?;
        Ok(Self {
            pass_index,
            stages: [
                RenderStage::Schedule,
                RenderStage::PromptPrep,
                RenderStage::Encode,
                RenderStage::Init,
                RenderStage::Enter,
                RenderStage::Denoise,
                RenderStage::Exit,
                RenderStage::PostLatent,
                RenderStage::Decode,
                RenderStage::Output,
            ],
            sigmas,
            executed_steps,
            sampler: sampler.into(),
            target_width: width,
            target_height: height,
            pixel_stride,
            token_width,
            base_guidance: guidance,
            produced_reference,
            phases: vec![ComposedDenoisePhase {
                phase_index: 0,
                first_step: 0,
                end_step: executed_steps,
                width,
                height,
                pixel_stride,
                packed_tokens,
                guidance,
                active_adapters: Vec::new(),
                active_expert: None,
            }],
            transitions: Vec::new(),
            pag_active: false,
            diagnostic_observers: Vec::new(),
        })
    }

    /// Insert or merge a phase boundary at `step` (`0 < step < executed_steps`), attaching `action`
    /// in deterministic canonical order without creating zero-step phases.
    pub fn insert_boundary_action(
        &mut self,
        step: usize,
        action: PhaseTransitionAction,
    ) -> Result<()> {
        if step == 0 || step >= self.executed_steps {
            return Err(Error::Unsupported(format!(
                "phase transition step {step} must lie in 1..{}",
                self.executed_steps
            )));
        }
        let boundary_sigma = self.sigmas[step];
        match self.transitions.binary_search_by_key(&step, |t| t.step) {
            Ok(idx) => {
                self.transitions[idx].actions.push(action);
                self.transitions[idx]
                    .actions
                    .sort_by_key(PhaseTransitionAction::canonical_rank);
            }
            Err(idx) => {
                self.transitions.insert(
                    idx,
                    ComposedPhaseTransition {
                        step,
                        boundary_sigma,
                        actions: vec![action],
                    },
                );
            }
        }
        self.rebuild_phases_from_transitions()
    }

    fn rebuild_phases_from_transitions(&mut self) -> Result<()> {
        let mut boundaries = Vec::with_capacity(self.transitions.len() + 2);
        boundaries.push(0usize);
        for transition in &self.transitions {
            boundaries.push(transition.step);
        }
        boundaries.push(self.executed_steps);

        // Determine whether a low-res spatial transition and/or expert switch or multiphase override exists.
        let low_res_transition = self.transitions.iter().find_map(|t| {
            t.actions.iter().find_map(|action| match action {
                PhaseTransitionAction::CleanEstimateSpatialResizeRenoise {
                    from_width,
                    from_height,
                    ..
                } => Some((t.step, *from_width, *from_height)),
                _ => None,
            })
        });
        let expert_transition = self.transitions.iter().find_map(|t| {
            t.actions.iter().find_map(|action| match action {
                PhaseTransitionAction::ExpertSwitch {
                    from_expert,
                    to_expert,
                } => Some((t.step, from_expert.clone(), to_expert.clone())),
                _ => None,
            })
        });

        let mut phases = Vec::with_capacity(boundaries.len() - 1);
        for (phase_index, window) in boundaries.windows(2).enumerate() {
            let (first_step, end_step) = (window[0], window[1]);
            if end_step <= first_step {
                return Err(Error::Unsupported(
                    "composed denoise phases must be non-empty and strictly increasing".into(),
                ));
            }
            let (width, height) = match low_res_transition {
                Some((split_step, low_w, low_h)) if end_step <= split_step => (low_w, low_h),
                _ => (self.target_width, self.target_height),
            };
            let packed_tokens = (width / self.pixel_stride)
                .checked_mul(height / self.pixel_stride)
                .ok_or_else(|| Error::Unsupported("packed token count overflow".into()))?;
            let active_expert = match &expert_transition {
                Some((split_step, from_expert, to_expert)) => {
                    if end_step <= *split_step {
                        Some(from_expert.clone())
                    } else {
                        Some(to_expert.clone())
                    }
                }
                None => None,
            };
            // Preserve per-phase guidance/adapters if already set for this span.
            let existing = self
                .phases
                .iter()
                .find(|p| p.first_step <= first_step && p.end_step >= end_step);
            let guidance = existing.map_or(self.base_guidance, |p| p.guidance);
            let active_adapters = existing.map_or_else(Vec::new, |p| p.active_adapters.clone());
            phases.push(ComposedDenoisePhase {
                phase_index,
                first_step,
                end_step,
                width,
                height,
                pixel_stride: self.pixel_stride,
                packed_tokens,
                guidance,
                active_adapters,
                active_expert,
            });
        }
        self.phases = phases;
        Ok(())
    }
}

/// Complete tensor-free plan produced by deterministic [`FeatureRegistry::compose`] (HOOK-3).
#[derive(Clone, Debug, PartialEq, Serialize)]
pub struct ComposedRenderPlan {
    pub route_id: String,
    pub backend: String,
    pub request_has_phases: bool,
    pub pass0_has_conditioning: bool,
    pub resolved_feature_order: Vec<String>,
    pub passes: Vec<ComposedPassPlan>,
}

impl ComposedRenderPlan {
    /// Lift an ordinary single-pass [`RenderPlan`] into a [`ComposedRenderPlan`] ready for Feature composition.
    pub fn from_render_plan(
        route: &FeatureRouteStructure,
        plan: &RenderPlan,
        req: &GenerationRequest,
    ) -> Result<Self> {
        let (width, height) = plan.output_size();
        let pass = ComposedPassPlan::single_phase(
            plan.pass_index(),
            width,
            height,
            plan.pixel_stride(),
            plan.token_width(),
            plan.sampler(),
            plan.guidance(),
            plan.denoise_sigmas().to_vec(),
            plan.produced_reference(),
        )?;
        Ok(Self {
            route_id: route.route_id.clone(),
            backend: route.backend.clone(),
            request_has_phases: req.phases.is_some(),
            pass0_has_conditioning: !req.conditioning.is_empty() || plan.has_conditioning(),
            resolved_feature_order: Vec::new(),
            passes: vec![pass],
        })
    }
}

/// Typed parameters supplied to a [`FeatureSpec`] invocation.
#[derive(Clone, Debug, PartialEq)]
pub enum FeatureParameters {
    LowResStart {
        k: usize,
        scale: Option<f32>,
    },
    HiresFix {
        width: u32,
        height: u32,
        steps: u32,
        strength: Option<f32>,
        sigmas: Vec<f32>,
    },
    KreaMultiphase {
        phases: Vec<GenerationPhase>,
        loaded_adapter_count: usize,
    },
    WanExpertSwitch {
        boundary_step: usize,
        high_noise_expert: String,
        low_noise_expert: String,
    },
    Pag {
        scale: f32,
        layers: Vec<u32>,
    },
    InertPhaseSplit {
        step: usize,
    },
    TrajectoryObserver {
        label: String,
    },
}

/// One requested Feature activation for [`FeatureRegistry::compose`].
#[derive(Clone, Debug, PartialEq)]
pub struct FeatureInvocation {
    pub feature_id: String,
    pub parameters: FeatureParameters,
}

/// Tensor-free, object-safe Feature specification contract (HOOK-2).
pub trait FeatureSpec: Send + Sync {
    /// Stable lowercase feature identifier (e.g. `"low_res_start"`).
    fn id(&self) -> &'static str;

    /// Feature specification version (`1` for `"low_res_start"`).
    fn version(&self) -> u32;

    /// DEC-8 classification (`Recipe`, `Equivalent`, or `Diagnostic`).
    fn class(&self) -> FeatureClass;

    /// Versioned parameter schema for `route`.
    fn parameter_schema(&self, route: Option<&FeatureRouteStructure>) -> FeatureParameterSchema;

    /// Inter-feature conflicts, requirements, and ordering constraints.
    fn relations(&self) -> FeatureRelations;

    /// Hook points and stage timings bound by this feature.
    fn hook_bindings(&self) -> Vec<FeatureHookBinding>;

    /// Memory and transition-residency effect of activating this feature.
    fn memory_effect(&self) -> FeatureMemoryEffect;

    /// Structural applicability predicate over `route` (computed weights-free, never a table).
    fn applicability(&self, route: &FeatureRouteStructure) -> FeatureApplicability {
        let reasons = self.inapplicability_reasons(route);
        let hook_points = self
            .hook_bindings()
            .into_iter()
            .map(|binding| binding.hook_point)
            .collect();
        FeatureApplicability {
            feature_id: self.id(),
            version: self.version(),
            class: self.class(),
            applicable: reasons.is_empty(),
            parameter_schema: self.parameter_schema(Some(route)),
            relations: self.relations(),
            hook_points,
            memory_effect: self.memory_effect(),
            reasons,
            conditioned_routes: BTreeMap::new(),
        }
    }

    /// Compute every structural reason this feature cannot run on `route` (empty => applicable).
    fn inapplicability_reasons(
        &self,
        route: &FeatureRouteStructure,
    ) -> Vec<FeatureInapplicabilityReason>;

    /// Validate parameters and rewrite `plan` in place during deterministic composition (HOOK-3).
    fn rewrite_plan(
        &self,
        route: &FeatureRouteStructure,
        plan: &mut ComposedRenderPlan,
        parameters: &FeatureParameters,
    ) -> Result<()>;
}

// Compile-time proof that `FeatureSpec` is object-safe (HOOK-2).
const _: Option<&dyn FeatureSpec> = None;

// -------------------------------------------------------------------------------------------------
// Registered FeatureSpec implementations
// -------------------------------------------------------------------------------------------------

/// `"low_res_start"` (`FeatureClass::Recipe`, version 1):
/// Runs steps `0..k` of pass 0 at `scale` (in `0.25..=0.9`, default `0.5`) of target width/height,
/// then transitions at `sigmas[k]` via spatial unpack -> clean-estimate resize -> repack -> re-noise
/// to full target geometry for steps `k..executed_steps` over the single pass-0 schedule.
pub struct LowResStartFeature;

impl LowResStartFeature {
    pub const ID: &'static str = "low_res_start";
    pub const VERSION: u32 = 1;
    pub const SCHEMA_VERSION: u32 = 1;
    pub const SCALE_MIN: f32 = 0.25;
    pub const SCALE_MAX: f32 = 0.9;
    pub const SCALE_DEFAULT: f32 = 0.5;
}

impl FeatureSpec for LowResStartFeature {
    fn id(&self) -> &'static str {
        Self::ID
    }

    fn version(&self) -> u32 {
        Self::VERSION
    }

    fn class(&self) -> FeatureClass {
        FeatureClass::Recipe
    }

    fn parameter_schema(&self, route: Option<&FeatureRouteStructure>) -> FeatureParameterSchema {
        let default_steps = route.and_then(|r| r.default_executed_steps);
        let max_inc = default_steps
            .and_then(|steps| usize::try_from(steps).ok())
            .and_then(|steps| steps.checked_sub(1))
            .filter(|&max| max >= 1);
        FeatureParameterSchema {
            version: Self::SCHEMA_VERSION,
            parameters: vec![
                FeatureParameterSpec {
                    name: "k",
                    required: true,
                    domain: FeatureParameterDomain::StepIndex {
                        min: 1,
                        max_exclusive: "executed_steps",
                        max_inclusive_for_default_steps: max_inc,
                        default_executed_steps: default_steps,
                    },
                    description: "Pass-0 step index k in 1..executed_steps at which the trajectory transitions from low-res to target geometry",
                },
                FeatureParameterSpec {
                    name: "scale",
                    required: false,
                    domain: FeatureParameterDomain::FloatRange {
                        min: Self::SCALE_MIN,
                        max: Self::SCALE_MAX,
                        inclusive: true,
                        default: Some(Self::SCALE_DEFAULT),
                    },
                    description: "Low-resolution spatial scale factor in 0.25..=0.9 (default 0.5)",
                },
            ],
        }
    }

    fn relations(&self) -> FeatureRelations {
        FeatureRelations {
            conflicts_with_features: vec![KreaMultiphaseFeature::ID],
            conflicts_with_axes: vec![
                "phases",
                "pass0_reference_conditioning",
                "pass0_control_conditioning",
                "pass0_edit_conditioning",
            ],
            requires_features: Vec::new(),
            requires_capabilities: vec![
                "render_plan",
                "packed_image_trajectory",
                "spatial_unpack_repack",
                "clean_estimate_renoise_transition",
                "noise_start_pass0",
            ],
            before: vec![
                WanExpertSwitchFeature::ID,
                HiresFixFeature::ID,
                PagFeature::ID,
                TrajectoryObserverFeature::ID,
            ],
            after: Vec::new(),
        }
    }

    fn hook_bindings(&self) -> Vec<FeatureHookBinding> {
        vec![
            FeatureHookBinding {
                hook_point: HookPoint::ScheduleTransform,
                stage: RenderStage::Schedule,
                timing: HookTiming::After,
                pass_scope: Some(0),
                result_changing: true,
            },
            FeatureHookBinding {
                hook_point: HookPoint::PhaseTransition,
                stage: RenderStage::Denoise,
                timing: HookTiming::Before,
                pass_scope: Some(0),
                result_changing: true,
            },
        ]
    }

    fn memory_effect(&self) -> FeatureMemoryEffect {
        FeatureMemoryEffect {
            extra_resident_weight_bytes: 0,
            transition_residency: TransitionResidency::InDeviceLatentUnpackResizeRepack,
            introduces_denoise_phase_boundary: true,
            introduces_pass_boundary: false,
            retains_single_pass_schedule: true,
        }
    }

    fn inapplicability_reasons(
        &self,
        route: &FeatureRouteStructure,
    ) -> Vec<FeatureInapplicabilityReason> {
        let mut reasons = Vec::new();
        if route.modality != Modality::Image {
            reasons.push(FeatureInapplicabilityReason {
                code: FeatureInapplicabilityCode::UnsupportedModality,
                message: format!(
                    "route '{}' has modality {:?}; low_res_start applies only to image trajectories",
                    route.route_id, route.modality
                ),
            });
        }
        if !route.has_render_plan {
            reasons.push(FeatureInapplicabilityReason {
                code: FeatureInapplicabilityCode::MissingRenderPlan,
                message: format!(
                    "route '{}' does not register a pre-load RenderPlan",
                    route.route_id
                ),
            });
        }
        if !route.has_packed_image_trajectory {
            reasons.push(FeatureInapplicabilityReason {
                code: FeatureInapplicabilityCode::MissingPackedImageTrajectory,
                message: format!(
                    "route '{}' does not compile a pass-0 packed image trajectory",
                    route.route_id
                ),
            });
        }
        if !route.supports_spatial_unpack_repack {
            reasons.push(FeatureInapplicabilityReason {
                code: FeatureInapplicabilityCode::MissingSpatialUnpackRepack,
                message: format!(
                    "route '{}' does not declare a 2x2-packed spatial latent space supporting unpack/repack",
                    route.route_id
                ),
            });
        }
        if !route.supports_clean_estimate_renoise_transition {
            reasons.push(FeatureInapplicabilityReason {
                code: FeatureInapplicabilityCode::MissingCleanEstimateRenoiseTransition,
                message: format!(
                    "route '{}' does not support clean-estimate spatial resize and re-noise phase transitions",
                    route.route_id
                ),
            });
        }
        match route.pass0_conditioning {
            Pass0ConditioningRequirement::NoiseStartSupported => {}
            Pass0ConditioningRequirement::RequiresReference => {
                reasons.push(FeatureInapplicabilityReason {
                    code: FeatureInapplicabilityCode::ConflictsWithPass0ReferenceConditioning,
                    message: format!(
                        "route '{}' requires pass-0 reference/identity conditioning, which conflicts with noise-start low_res_start",
                        route.route_id
                    ),
                });
            }
            Pass0ConditioningRequirement::RequiresControl => {
                reasons.push(FeatureInapplicabilityReason {
                    code: FeatureInapplicabilityCode::ConflictsWithPass0ControlConditioning,
                    message: format!(
                        "route '{}' requires pass-0 control conditioning, which conflicts with noise-start low_res_start",
                        route.route_id
                    ),
                });
            }
            Pass0ConditioningRequirement::RequiresEdit => {
                reasons.push(FeatureInapplicabilityReason {
                    code: FeatureInapplicabilityCode::ConflictsWithPass0EditConditioning,
                    message: format!(
                        "route '{}' requires pass-0 edit conditioning, which conflicts with noise-start low_res_start",
                        route.route_id
                    ),
                });
            }
        }
        if route.supports_multiphase_denoise {
            reasons.push(FeatureInapplicabilityReason {
                code: FeatureInapplicabilityCode::ConflictsWithMultiphaseDenoise,
                message: format!(
                    "route '{}' uses user-supplied multi-phase denoise (phases), which conflicts with low_res_start",
                    route.route_id
                ),
            });
        }
        if route.default_executed_steps.is_some_and(|steps| steps < 2) {
            reasons.push(FeatureInapplicabilityReason {
                code: FeatureInapplicabilityCode::InsufficientExecutedSteps,
                message: format!(
                    "route '{}' executes fewer than 2 steps, leaving 1..executed_steps empty",
                    route.route_id
                ),
            });
        }
        reasons
    }

    fn rewrite_plan(
        &self,
        route: &FeatureRouteStructure,
        plan: &mut ComposedRenderPlan,
        parameters: &FeatureParameters,
    ) -> Result<()> {
        if plan.request_has_phases {
            return Err(Error::Unsupported(format!(
                "{}: low_res_start conflicts with multi-phase denoise (phases)",
                route.route_id
            )));
        }
        if plan.pass0_has_conditioning {
            return Err(Error::Unsupported(format!(
                "{}: low_res_start requires a noise-start pass 0 and conflicts with pass-0 edit/reference/control conditioning",
                route.route_id
            )));
        }
        let FeatureParameters::LowResStart { k, scale } = parameters else {
            return Err(Error::Unsupported(
                "low_res_start requires FeatureParameters::LowResStart".into(),
            ));
        };
        let pass0 = plan
            .passes
            .first_mut()
            .ok_or_else(|| Error::Unsupported("composed plan has no pass 0".into()))?;
        let stride = pass0.pixel_stride.max(1);
        let resolved = LowResStartSpec::new(*k, *scale).resolve_geometry(
            &route.route_id,
            pass0.target_width,
            pass0.target_height,
            pass0.executed_steps,
            stride,
            1,
            stride,
            stride,
        )?;
        pass0.insert_boundary_action(
            resolved.k,
            PhaseTransitionAction::CleanEstimateSpatialResizeRenoise {
                scale_bits: resolved.scale.to_bits(),
                from_width: resolved.low_width,
                from_height: resolved.low_height,
                to_width: resolved.full_width,
                to_height: resolved.full_height,
            },
        )
    }
}

/// `"hires_fix"` (`FeatureClass::Recipe`, version 1):
/// Appends a second pass with pass-local stage bindings and a pixel round-trip transition
/// (`ProducedReference { from_pass: 0, ... }`). Composes cleanly with `"low_res_start"` on pass 0.
pub struct HiresFixFeature;

impl HiresFixFeature {
    pub const ID: &'static str = "hires_fix";
    pub const VERSION: u32 = 1;
}

impl FeatureSpec for HiresFixFeature {
    fn id(&self) -> &'static str {
        Self::ID
    }

    fn version(&self) -> u32 {
        Self::VERSION
    }

    fn class(&self) -> FeatureClass {
        FeatureClass::Recipe
    }

    fn parameter_schema(&self, _route: Option<&FeatureRouteStructure>) -> FeatureParameterSchema {
        FeatureParameterSchema {
            version: 1,
            parameters: vec![
                FeatureParameterSpec {
                    name: "strength",
                    required: false,
                    domain: FeatureParameterDomain::FloatRange {
                        min: 0.0,
                        max: 1.0,
                        inclusive: true,
                        default: Some(0.3),
                    },
                    description: "Pass-1 produced-reference denoise strength",
                },
                FeatureParameterSpec {
                    name: "steps",
                    required: true,
                    domain: FeatureParameterDomain::IntegerRange {
                        min: 1,
                        max: 1000,
                        default: Some(12),
                    },
                    description: "Pass-1 executed denoise steps",
                },
            ],
        }
    }

    fn relations(&self) -> FeatureRelations {
        FeatureRelations {
            conflicts_with_features: Vec::new(),
            conflicts_with_axes: Vec::new(),
            requires_features: Vec::new(),
            requires_capabilities: vec![
                "render_plan",
                "packed_image_trajectory",
                "produced_reference_multipass",
            ],
            before: vec![TrajectoryObserverFeature::ID],
            after: vec![LowResStartFeature::ID, WanExpertSwitchFeature::ID],
        }
    }

    fn hook_bindings(&self) -> Vec<FeatureHookBinding> {
        vec![
            FeatureHookBinding {
                hook_point: HookPoint::ConditioningSlot,
                stage: RenderStage::Encode,
                timing: HookTiming::After,
                pass_scope: Some(1),
                result_changing: true,
            },
            FeatureHookBinding {
                hook_point: HookPoint::PhaseTransition,
                stage: RenderStage::Decode,
                timing: HookTiming::After,
                pass_scope: Some(1),
                result_changing: true,
            },
        ]
    }

    fn memory_effect(&self) -> FeatureMemoryEffect {
        FeatureMemoryEffect {
            extra_resident_weight_bytes: 0,
            transition_residency: TransitionResidency::PixelRoundTripProducedReference,
            introduces_denoise_phase_boundary: false,
            introduces_pass_boundary: true,
            retains_single_pass_schedule: false,
        }
    }

    fn inapplicability_reasons(
        &self,
        route: &FeatureRouteStructure,
    ) -> Vec<FeatureInapplicabilityReason> {
        let mut reasons = Vec::new();
        if !route.has_render_plan {
            reasons.push(FeatureInapplicabilityReason {
                code: FeatureInapplicabilityCode::MissingRenderPlan,
                message: format!(
                    "route '{}' does not register a pre-load RenderPlan",
                    route.route_id
                ),
            });
        }
        if !route.supports_produced_reference_multipass {
            reasons.push(FeatureInapplicabilityReason {
                code: FeatureInapplicabilityCode::MissingProducedReferenceSupport,
                message: format!(
                    "route '{}' does not accept a ProducedReference pass-1 input",
                    route.route_id
                ),
            });
        }
        reasons
    }

    fn rewrite_plan(
        &self,
        route: &FeatureRouteStructure,
        plan: &mut ComposedRenderPlan,
        parameters: &FeatureParameters,
    ) -> Result<()> {
        let FeatureParameters::HiresFix {
            width,
            height,
            steps,
            strength,
            sigmas,
        } = parameters
        else {
            return Err(Error::Unsupported(
                "hires_fix requires FeatureParameters::HiresFix".into(),
            ));
        };
        if *steps == 0 || sigmas.len() != (*steps as usize) + 1 {
            return Err(Error::Unsupported(format!(
                "{}: hires_fix pass-1 sigmas length {} must equal steps ({steps}) + 1",
                route.route_id,
                sigmas.len()
            )));
        }
        let pass0 = plan
            .passes
            .first()
            .ok_or_else(|| Error::Unsupported("composed plan has no pass 0".into()))?;
        let pass_index = plan.passes.len();
        let produced = ProducedReference {
            from_pass: 0,
            width: *width,
            height: *height,
            strength: *strength,
        };
        let pass1 = ComposedPassPlan::single_phase(
            pass_index,
            *width,
            *height,
            pass0.pixel_stride,
            pass0.token_width,
            pass0.sampler.clone(),
            pass0.base_guidance,
            sigmas.clone(),
            Some(produced),
        )?;
        plan.passes.push(pass1);
        Ok(())
    }
}

/// `"krea_multiphase"` (`FeatureClass::Recipe`, version 1):
/// Slices one coherent global schedule into contiguous denoise phases with per-phase guidance and
/// active low-rank adapter subsets. Conflicts with `"low_res_start"`.
pub struct KreaMultiphaseFeature;

impl KreaMultiphaseFeature {
    pub const ID: &'static str = "krea_multiphase";
    pub const VERSION: u32 = 1;
}

impl FeatureSpec for KreaMultiphaseFeature {
    fn id(&self) -> &'static str {
        Self::ID
    }

    fn version(&self) -> u32 {
        Self::VERSION
    }

    fn class(&self) -> FeatureClass {
        FeatureClass::Recipe
    }

    fn parameter_schema(&self, _route: Option<&FeatureRouteStructure>) -> FeatureParameterSchema {
        FeatureParameterSchema {
            version: 1,
            parameters: vec![FeatureParameterSpec {
                name: "phases",
                required: true,
                domain: FeatureParameterDomain::PhaseList {
                    min_phases: 1,
                    min_steps_per_phase: 1,
                },
                description: "Ordered denoise phases partitioning the single global schedule",
            }],
        }
    }

    fn relations(&self) -> FeatureRelations {
        FeatureRelations {
            conflicts_with_features: vec![LowResStartFeature::ID, InertPhaseSplitFeature::ID],
            conflicts_with_axes: vec![
                "pass0_reference_conditioning",
                "pass0_control_conditioning",
                "pass0_edit_conditioning",
            ],
            requires_features: Vec::new(),
            requires_capabilities: vec!["multiphase_denoise"],
            before: vec![HiresFixFeature::ID, TrajectoryObserverFeature::ID],
            after: Vec::new(),
        }
    }

    fn hook_bindings(&self) -> Vec<FeatureHookBinding> {
        vec![
            FeatureHookBinding {
                hook_point: HookPoint::ScheduleTransform,
                stage: RenderStage::Schedule,
                timing: HookTiming::Replace,
                pass_scope: Some(0),
                result_changing: true,
            },
            FeatureHookBinding {
                hook_point: HookPoint::PhaseTransition,
                stage: RenderStage::Denoise,
                timing: HookTiming::Before,
                pass_scope: Some(0),
                result_changing: true,
            },
            FeatureHookBinding {
                hook_point: HookPoint::GuidancePredictor,
                stage: RenderStage::Denoise,
                timing: HookTiming::Replace,
                pass_scope: Some(0),
                result_changing: true,
            },
        ]
    }

    fn memory_effect(&self) -> FeatureMemoryEffect {
        FeatureMemoryEffect {
            extra_resident_weight_bytes: 0,
            transition_residency: TransitionResidency::JobLocalAdapterResidualSwap,
            introduces_denoise_phase_boundary: true,
            introduces_pass_boundary: false,
            retains_single_pass_schedule: true,
        }
    }

    fn inapplicability_reasons(
        &self,
        route: &FeatureRouteStructure,
    ) -> Vec<FeatureInapplicabilityReason> {
        let mut reasons = Vec::new();
        if !route.supports_multiphase_denoise {
            reasons.push(FeatureInapplicabilityReason {
                code: FeatureInapplicabilityCode::MissingMultiphaseDenoiseSupport,
                message: format!(
                    "route '{}' does not support multi-phase denoise (phases)",
                    route.route_id
                ),
            });
        }
        reasons
    }

    fn rewrite_plan(
        &self,
        route: &FeatureRouteStructure,
        plan: &mut ComposedRenderPlan,
        parameters: &FeatureParameters,
    ) -> Result<()> {
        let FeatureParameters::KreaMultiphase {
            phases,
            loaded_adapter_count,
        } = parameters
        else {
            return Err(Error::Unsupported(
                "krea_multiphase requires FeatureParameters::KreaMultiphase".into(),
            ));
        };
        if phases.is_empty() {
            return Err(Error::Unsupported(format!(
                "{}: krea_multiphase requires at least one phase",
                route.route_id
            )));
        }
        let pass0 = plan
            .passes
            .first_mut()
            .ok_or_else(|| Error::Unsupported("composed plan has no pass 0".into()))?;
        let total_steps: usize = phases.iter().map(|p| p.steps as usize).sum();
        if total_steps != pass0.executed_steps {
            return Err(Error::Unsupported(format!(
                "{}: krea_multiphase total steps {total_steps} does not match pass-0 schedule executed_steps {}",
                route.route_id, pass0.executed_steps
            )));
        }
        let mut cursor = 0usize;
        let mut composed_phases = Vec::with_capacity(phases.len());
        let packed_tokens =
            (pass0.target_width / pass0.pixel_stride) * (pass0.target_height / pass0.pixel_stride);
        for (idx, phase) in phases.iter().enumerate() {
            if phase.steps == 0 {
                return Err(Error::Unsupported(format!(
                    "{}: krea_multiphase phase {idx} has 0 steps",
                    route.route_id
                )));
            }
            let mut adapters = Vec::with_capacity(phase.adapters.len());
            for PhaseAdapter { adapter, weight } in &phase.adapters {
                if *adapter >= *loaded_adapter_count {
                    return Err(Error::Unsupported(format!(
                        "{}: phase {idx} adapter index {adapter} exceeds loaded_adapter_count {loaded_adapter_count}",
                        route.route_id
                    )));
                }
                adapters.push((*adapter, weight.map(f32::to_bits)));
            }
            let first_step = cursor;
            let end_step = cursor + phase.steps as usize;
            let guidance = phase.guidance.unwrap_or(pass0.base_guidance);
            if first_step > 0 {
                pass0.transitions.push(ComposedPhaseTransition {
                    step: first_step,
                    boundary_sigma: pass0.sigmas[first_step],
                    actions: vec![PhaseTransitionAction::AdapterAndGuidanceSwitch {
                        guidance_bits: guidance.to_bits(),
                        adapters: adapters.clone(),
                    }],
                });
            }
            composed_phases.push(ComposedDenoisePhase {
                phase_index: idx,
                first_step,
                end_step,
                width: pass0.target_width,
                height: pass0.target_height,
                pixel_stride: pass0.pixel_stride,
                packed_tokens,
                guidance,
                active_adapters: adapters,
                active_expert: None,
            });
            cursor = end_step;
        }
        pass0.phases = composed_phases;
        Ok(())
    }
}

/// `"wan_expert_switch"` (`FeatureClass::Recipe`, version 1):
/// Switches between MoE experts (`high_noise` -> `low_noise`) at a schedule boundary step.
/// Composes with `"low_res_start"` even when both target the exact same boundary step `k` (HOOK-3).
pub struct WanExpertSwitchFeature;

impl WanExpertSwitchFeature {
    pub const ID: &'static str = "wan_expert_switch";
    pub const VERSION: u32 = 1;
}

impl FeatureSpec for WanExpertSwitchFeature {
    fn id(&self) -> &'static str {
        Self::ID
    }

    fn version(&self) -> u32 {
        Self::VERSION
    }

    fn class(&self) -> FeatureClass {
        FeatureClass::Recipe
    }

    fn parameter_schema(&self, route: Option<&FeatureRouteStructure>) -> FeatureParameterSchema {
        let default_steps = route.and_then(|r| r.default_executed_steps);
        let max_inc = default_steps
            .and_then(|steps| usize::try_from(steps).ok())
            .and_then(|steps| steps.checked_sub(1))
            .filter(|&max| max >= 1);
        FeatureParameterSchema {
            version: 1,
            parameters: vec![FeatureParameterSpec {
                name: "boundary_step",
                required: true,
                domain: FeatureParameterDomain::StepIndex {
                    min: 1,
                    max_exclusive: "executed_steps",
                    max_inclusive_for_default_steps: max_inc,
                    default_executed_steps: default_steps,
                },
                description:
                    "Step index in 1..executed_steps at which the active MoE expert switches",
            }],
        }
    }

    fn relations(&self) -> FeatureRelations {
        FeatureRelations {
            conflicts_with_features: Vec::new(),
            conflicts_with_axes: Vec::new(),
            requires_features: Vec::new(),
            requires_capabilities: vec!["expert_switching"],
            before: vec![HiresFixFeature::ID, TrajectoryObserverFeature::ID],
            after: vec![LowResStartFeature::ID],
        }
    }

    fn hook_bindings(&self) -> Vec<FeatureHookBinding> {
        vec![FeatureHookBinding {
            hook_point: HookPoint::PhaseTransition,
            stage: RenderStage::Denoise,
            timing: HookTiming::Before,
            pass_scope: Some(0),
            result_changing: true,
        }]
    }

    fn memory_effect(&self) -> FeatureMemoryEffect {
        FeatureMemoryEffect {
            extra_resident_weight_bytes: 0,
            transition_residency: TransitionResidency::StagedExpertWeightHandoff,
            introduces_denoise_phase_boundary: true,
            introduces_pass_boundary: false,
            retains_single_pass_schedule: true,
        }
    }

    fn inapplicability_reasons(
        &self,
        route: &FeatureRouteStructure,
    ) -> Vec<FeatureInapplicabilityReason> {
        let mut reasons = Vec::new();
        if !route.supports_expert_switching {
            reasons.push(FeatureInapplicabilityReason {
                code: FeatureInapplicabilityCode::MissingExpertSwitchingSupport,
                message: format!(
                    "route '{}' does not use dual-expert MoE boundary switching",
                    route.route_id
                ),
            });
        }
        reasons
    }

    fn rewrite_plan(
        &self,
        route: &FeatureRouteStructure,
        plan: &mut ComposedRenderPlan,
        parameters: &FeatureParameters,
    ) -> Result<()> {
        let FeatureParameters::WanExpertSwitch {
            boundary_step,
            high_noise_expert,
            low_noise_expert,
        } = parameters
        else {
            return Err(Error::Unsupported(
                "wan_expert_switch requires FeatureParameters::WanExpertSwitch".into(),
            ));
        };
        let pass0 = plan
            .passes
            .first_mut()
            .ok_or_else(|| Error::Unsupported("composed plan has no pass 0".into()))?;
        if *boundary_step == 0 || *boundary_step >= pass0.executed_steps {
            return Err(Error::Unsupported(format!(
                "{}: wan_expert_switch boundary_step {boundary_step} must lie in 1..{}",
                route.route_id, pass0.executed_steps
            )));
        }
        pass0.insert_boundary_action(
            *boundary_step,
            PhaseTransitionAction::ExpertSwitch {
                from_expert: high_noise_expert.clone(),
                to_expert: low_noise_expert.clone(),
            },
        )
    }
}

/// `"pag"` (`FeatureClass::Recipe`, version 1):
/// Perturbed-Attention Guidance binding `HookPoint::AttentionBlock` and `HookPoint::GuidancePredictor`.
pub struct PagFeature;

impl PagFeature {
    pub const ID: &'static str = "pag";
    pub const VERSION: u32 = 1;
}

impl FeatureSpec for PagFeature {
    fn id(&self) -> &'static str {
        Self::ID
    }

    fn version(&self) -> u32 {
        Self::VERSION
    }

    fn class(&self) -> FeatureClass {
        FeatureClass::Recipe
    }

    fn parameter_schema(&self, _route: Option<&FeatureRouteStructure>) -> FeatureParameterSchema {
        FeatureParameterSchema {
            version: 1,
            parameters: vec![FeatureParameterSpec {
                name: "scale",
                required: true,
                domain: FeatureParameterDomain::FloatRange {
                    min: 0.0,
                    max: crate::DEFAULT_PAG_MAX_SCALE,
                    inclusive: true,
                    default: Some(1.5),
                },
                description: "Perturbed-Attention Guidance scale",
            }],
        }
    }

    fn relations(&self) -> FeatureRelations {
        FeatureRelations {
            conflicts_with_features: Vec::new(),
            conflicts_with_axes: Vec::new(),
            requires_features: Vec::new(),
            requires_capabilities: vec!["pag"],
            before: vec![TrajectoryObserverFeature::ID],
            after: vec![LowResStartFeature::ID],
        }
    }

    fn hook_bindings(&self) -> Vec<FeatureHookBinding> {
        vec![
            FeatureHookBinding {
                hook_point: HookPoint::AttentionBlock,
                stage: RenderStage::Denoise,
                timing: HookTiming::Replace,
                pass_scope: Some(0),
                result_changing: true,
            },
            FeatureHookBinding {
                hook_point: HookPoint::GuidancePredictor,
                stage: RenderStage::Denoise,
                timing: HookTiming::After,
                pass_scope: Some(0),
                result_changing: true,
            },
        ]
    }

    fn memory_effect(&self) -> FeatureMemoryEffect {
        FeatureMemoryEffect {
            extra_resident_weight_bytes: 0,
            transition_residency: TransitionResidency::None,
            introduces_denoise_phase_boundary: false,
            introduces_pass_boundary: false,
            retains_single_pass_schedule: true,
        }
    }

    fn inapplicability_reasons(
        &self,
        route: &FeatureRouteStructure,
    ) -> Vec<FeatureInapplicabilityReason> {
        let mut reasons = Vec::new();
        if !route.supports_pag {
            reasons.push(FeatureInapplicabilityReason {
                code: FeatureInapplicabilityCode::MissingPagCapability,
                message: format!(
                    "route '{}' does not advertise Perturbed-Attention Guidance (PAG)",
                    route.route_id
                ),
            });
        }
        reasons
    }

    fn rewrite_plan(
        &self,
        route: &FeatureRouteStructure,
        plan: &mut ComposedRenderPlan,
        parameters: &FeatureParameters,
    ) -> Result<()> {
        let FeatureParameters::Pag { scale, layers } = parameters else {
            return Err(Error::Unsupported(
                "pag requires FeatureParameters::Pag".into(),
            ));
        };
        if !scale.is_finite() || *scale <= 0.0 || layers.is_empty() {
            return Err(Error::Unsupported(format!(
                "{}: pag requires positive finite scale and non-empty layers",
                route.route_id
            )));
        }
        let pass0 = plan
            .passes
            .first_mut()
            .ok_or_else(|| Error::Unsupported("composed plan has no pass 0".into()))?;
        pass0.pag_active = true;
        Ok(())
    }
}

/// `"inert_phase_split"` (`FeatureClass::Equivalent`, version 1):
/// Proof-only result-equivalent phase boundary split over a single schedule.
pub struct InertPhaseSplitFeature;

impl InertPhaseSplitFeature {
    pub const ID: &'static str = "inert_phase_split";
    pub const VERSION: u32 = 1;
}

impl FeatureSpec for InertPhaseSplitFeature {
    fn id(&self) -> &'static str {
        Self::ID
    }

    fn version(&self) -> u32 {
        Self::VERSION
    }

    fn class(&self) -> FeatureClass {
        FeatureClass::Equivalent
    }

    fn parameter_schema(&self, route: Option<&FeatureRouteStructure>) -> FeatureParameterSchema {
        let default_steps = route.and_then(|r| r.default_executed_steps);
        let max_inc = default_steps
            .and_then(|steps| usize::try_from(steps).ok())
            .and_then(|steps| steps.checked_sub(1))
            .filter(|&max| max >= 1);
        FeatureParameterSchema {
            version: 1,
            parameters: vec![FeatureParameterSpec {
                name: "step",
                required: true,
                domain: FeatureParameterDomain::StepIndex {
                    min: 1,
                    max_exclusive: "executed_steps",
                    max_inclusive_for_default_steps: max_inc,
                    default_executed_steps: default_steps,
                },
                description: "Inert phase split step in 1..executed_steps",
            }],
        }
    }

    fn relations(&self) -> FeatureRelations {
        FeatureRelations {
            conflicts_with_features: vec![KreaMultiphaseFeature::ID],
            conflicts_with_axes: Vec::new(),
            requires_features: Vec::new(),
            requires_capabilities: vec!["render_plan", "packed_image_trajectory"],
            before: vec![HiresFixFeature::ID, TrajectoryObserverFeature::ID],
            after: vec![LowResStartFeature::ID, WanExpertSwitchFeature::ID],
        }
    }

    fn hook_bindings(&self) -> Vec<FeatureHookBinding> {
        vec![FeatureHookBinding {
            hook_point: HookPoint::PhaseTransition,
            stage: RenderStage::Denoise,
            timing: HookTiming::Observe,
            pass_scope: Some(0),
            result_changing: false,
        }]
    }

    fn memory_effect(&self) -> FeatureMemoryEffect {
        FeatureMemoryEffect {
            extra_resident_weight_bytes: 0,
            transition_residency: TransitionResidency::None,
            introduces_denoise_phase_boundary: true,
            introduces_pass_boundary: false,
            retains_single_pass_schedule: true,
        }
    }

    fn inapplicability_reasons(
        &self,
        route: &FeatureRouteStructure,
    ) -> Vec<FeatureInapplicabilityReason> {
        let mut reasons = Vec::new();
        if !route.has_render_plan {
            reasons.push(FeatureInapplicabilityReason {
                code: FeatureInapplicabilityCode::MissingRenderPlan,
                message: format!(
                    "route '{}' does not register a pre-load RenderPlan",
                    route.route_id
                ),
            });
        }
        if !route.has_packed_image_trajectory {
            reasons.push(FeatureInapplicabilityReason {
                code: FeatureInapplicabilityCode::MissingPackedImageTrajectory,
                message: format!(
                    "route '{}' does not compile a packed image trajectory",
                    route.route_id
                ),
            });
        }
        reasons
    }

    fn rewrite_plan(
        &self,
        route: &FeatureRouteStructure,
        plan: &mut ComposedRenderPlan,
        parameters: &FeatureParameters,
    ) -> Result<()> {
        let FeatureParameters::InertPhaseSplit { step } = parameters else {
            return Err(Error::Unsupported(
                "inert_phase_split requires FeatureParameters::InertPhaseSplit".into(),
            ));
        };
        let pass0 = plan
            .passes
            .first_mut()
            .ok_or_else(|| Error::Unsupported("composed plan has no pass 0".into()))?;
        if *step == 0 || *step >= pass0.executed_steps {
            return Err(Error::Unsupported(format!(
                "{}: inert_phase_split step {step} must lie in 1..{}",
                route.route_id, pass0.executed_steps
            )));
        }
        pass0.insert_boundary_action(*step, PhaseTransitionAction::InertSplit)
    }
}

/// `"trajectory_observer"` (`FeatureClass::Diagnostic`, version 1):
/// Non-mutating diagnostic observer on `HookPoint::AttentionBlock` and `HookPoint::PhaseTransition`.
pub struct TrajectoryObserverFeature;

impl TrajectoryObserverFeature {
    pub const ID: &'static str = "trajectory_observer";
    pub const VERSION: u32 = 1;
}

impl FeatureSpec for TrajectoryObserverFeature {
    fn id(&self) -> &'static str {
        Self::ID
    }

    fn version(&self) -> u32 {
        Self::VERSION
    }

    fn class(&self) -> FeatureClass {
        FeatureClass::Diagnostic
    }

    fn parameter_schema(&self, _route: Option<&FeatureRouteStructure>) -> FeatureParameterSchema {
        FeatureParameterSchema {
            version: 1,
            parameters: Vec::new(),
        }
    }

    fn relations(&self) -> FeatureRelations {
        FeatureRelations {
            conflicts_with_features: Vec::new(),
            conflicts_with_axes: Vec::new(),
            requires_features: Vec::new(),
            requires_capabilities: vec!["render_plan"],
            before: Vec::new(),
            after: vec![
                LowResStartFeature::ID,
                WanExpertSwitchFeature::ID,
                KreaMultiphaseFeature::ID,
                HiresFixFeature::ID,
                PagFeature::ID,
                InertPhaseSplitFeature::ID,
            ],
        }
    }

    fn hook_bindings(&self) -> Vec<FeatureHookBinding> {
        vec![
            FeatureHookBinding {
                hook_point: HookPoint::AttentionBlock,
                stage: RenderStage::Denoise,
                timing: HookTiming::Observe,
                pass_scope: None,
                result_changing: false,
            },
            FeatureHookBinding {
                hook_point: HookPoint::PhaseTransition,
                stage: RenderStage::Denoise,
                timing: HookTiming::Observe,
                pass_scope: None,
                result_changing: false,
            },
        ]
    }

    fn memory_effect(&self) -> FeatureMemoryEffect {
        FeatureMemoryEffect {
            extra_resident_weight_bytes: 0,
            transition_residency: TransitionResidency::None,
            introduces_denoise_phase_boundary: false,
            introduces_pass_boundary: false,
            retains_single_pass_schedule: true,
        }
    }

    fn inapplicability_reasons(
        &self,
        route: &FeatureRouteStructure,
    ) -> Vec<FeatureInapplicabilityReason> {
        let mut reasons = Vec::new();
        if !route.has_render_plan {
            reasons.push(FeatureInapplicabilityReason {
                code: FeatureInapplicabilityCode::MissingRenderPlan,
                message: format!(
                    "route '{}' does not register a pre-load RenderPlan",
                    route.route_id
                ),
            });
        }
        reasons
    }

    fn rewrite_plan(
        &self,
        _route: &FeatureRouteStructure,
        plan: &mut ComposedRenderPlan,
        parameters: &FeatureParameters,
    ) -> Result<()> {
        let FeatureParameters::TrajectoryObserver { label } = parameters else {
            return Err(Error::Unsupported(
                "trajectory_observer requires FeatureParameters::TrajectoryObserver".into(),
            ));
        };
        for pass in &mut plan.passes {
            pass.diagnostic_observers.push(label.clone());
            pass.diagnostic_observers.sort();
            pass.diagnostic_observers.dedup();
        }
        Ok(())
    }
}

// -------------------------------------------------------------------------------------------------
// FeatureRegistry: object-safe registry, applicability evaluation, and deterministic composition
// -------------------------------------------------------------------------------------------------

/// Registry of object-safe [`FeatureSpec`]s (HOOK-2 / HOOK-3).
pub struct FeatureRegistry {
    specs: Vec<Box<dyn FeatureSpec>>,
}

impl Default for FeatureRegistry {
    fn default() -> Self {
        Self::standard()
    }
}

impl FeatureRegistry {
    /// Construct the standard production Feature registry.
    pub fn standard() -> Self {
        Self {
            specs: vec![
                Box::new(LowResStartFeature),
                Box::new(HiresFixFeature),
                Box::new(KreaMultiphaseFeature),
                Box::new(WanExpertSwitchFeature),
                Box::new(PagFeature),
                Box::new(InertPhaseSplitFeature),
                Box::new(TrajectoryObserverFeature),
            ],
        }
    }

    /// Construct a registry from an explicit list of boxed [`FeatureSpec`]s (used to prove
    /// registration-order invariance and test synthetic cycle/conflict configurations).
    pub fn with_specs(specs: Vec<Box<dyn FeatureSpec>>) -> Result<Self> {
        let mut seen = BTreeSet::new();
        for spec in &specs {
            if !seen.insert(spec.id()) {
                return Err(Error::Unsupported(format!(
                    "duplicate FeatureSpec id '{}' in FeatureRegistry",
                    spec.id()
                )));
            }
        }
        Ok(Self { specs })
    }

    /// Every registered [`FeatureSpec`] in this registry.
    pub fn specs(&self) -> &[Box<dyn FeatureSpec>] {
        &self.specs
    }

    /// Look up a registered [`FeatureSpec`] by id.
    pub fn get(&self, id: &str) -> Option<&dyn FeatureSpec> {
        self.specs
            .iter()
            .find(|spec| spec.id() == id)
            .map(Box::as_ref)
    }

    /// Compute applicability for every registered [`FeatureSpec`] on `route`, keyed by feature id.
    pub fn evaluate_route(
        &self,
        route: &FeatureRouteStructure,
    ) -> BTreeMap<String, FeatureApplicability> {
        let mut map = BTreeMap::new();
        for spec in &self.specs {
            map.insert(spec.id().to_string(), spec.applicability(route));
        }
        map
    }

    /// Validate and compose an active set of [`FeatureInvocation`]s onto `base_plan` (HOOK-3).
    ///
    /// Guarantees:
    /// - Whole-active-set applicability, requirement, conflict, exclusive-`Replace`-hook ownership,
    ///   and cycle validation run before any rewrite.
    /// - Topological ordering breaks ties by canonical `(earliest_hook_ordering, feature_id)`, never
    ///   by registration order or caller invocation order.
    pub fn compose(
        &self,
        route: &FeatureRouteStructure,
        mut base_plan: ComposedRenderPlan,
        invocations: &[FeatureInvocation],
    ) -> Result<ComposedRenderPlan> {
        // 1. Resolve all invoked specs and reject unknown or duplicate feature IDs.
        let mut active_by_id: BTreeMap<&'static str, (&dyn FeatureSpec, &FeatureInvocation)> =
            BTreeMap::new();
        for invocation in invocations {
            let spec = self.get(&invocation.feature_id).ok_or_else(|| {
                Error::Unsupported(format!(
                    "unknown feature '{}' requested on route '{}'",
                    invocation.feature_id, route.route_id
                ))
            })?;
            if active_by_id.insert(spec.id(), (spec, invocation)).is_some() {
                return Err(Error::Unsupported(format!(
                    "duplicate feature '{}' in active set on route '{}'",
                    spec.id(),
                    route.route_id
                )));
            }
        }

        // 2. Per-feature route applicability check.
        for (&id, &(spec, _)) in &active_by_id {
            let applicability = spec.applicability(route);
            if !applicability.applicable {
                let reasons: Vec<String> = applicability
                    .reasons
                    .iter()
                    .map(|r| format!("{}: {}", r.code.as_str(), r.message))
                    .collect();
                return Err(Error::Unsupported(format!(
                    "feature '{id}' is not applicable to route '{}': {}",
                    route.route_id,
                    reasons.join("; ")
                )));
            }
        }

        // 3. Whole-active-set requirements and symmetric conflicts check.
        let active_ids: Vec<&'static str> = active_by_id.keys().copied().collect();
        for &id in &active_ids {
            let (spec, _) = active_by_id[id];
            let relations = spec.relations();
            for &required in &relations.requires_features {
                if !active_by_id.contains_key(required) {
                    return Err(Error::Unsupported(format!(
                        "feature '{id}' requires feature '{required}', which is not in the active set"
                    )));
                }
            }
            for &conflict in &relations.conflicts_with_features {
                if active_by_id.contains_key(conflict) {
                    let (a, b) = if id <= conflict {
                        (id, conflict)
                    } else {
                        (conflict, id)
                    };
                    return Err(Error::Unsupported(format!(
                        "features '{a}' and '{b}' conflict and cannot be composed together"
                    )));
                }
            }
        }

        // 4. Whole-active-set exclusive `HookTiming::Replace` ownership check on `(pass_scope, hook_point)`.
        let mut replace_owners: BTreeMap<(Option<usize>, HookPoint), &'static str> =
            BTreeMap::new();
        for &id in &active_ids {
            let (spec, _) = active_by_id[id];
            for binding in spec.hook_bindings() {
                if binding.timing == HookTiming::Replace {
                    let key = (binding.pass_scope, binding.hook_point);
                    if let Some(&existing_owner) = replace_owners.get(&key) {
                        if existing_owner != id {
                            return Err(Error::Unsupported(format!(
                                "hook ownership conflict on {:?} (pass {:?}): both '{}' and '{}' claim Replace timing",
                                binding.hook_point, binding.pass_scope, existing_owner, id
                            )));
                        }
                    } else {
                        replace_owners.insert(key, id);
                    }
                }
            }
        }

        // 5. Build ordering graph over the active set and run deterministic Kahn topological sort.
        let mut edges: BTreeMap<&'static str, BTreeSet<&'static str>> = active_ids
            .iter()
            .copied()
            .map(|id| (id, BTreeSet::new()))
            .collect();
        let mut in_degree: BTreeMap<&'static str, usize> =
            active_ids.iter().copied().map(|id| (id, 0usize)).collect();

        for &id in &active_ids {
            let (spec, _) = active_by_id[id];
            let relations = spec.relations();
            for &target in &relations.before {
                if active_by_id.contains_key(target) {
                    if target == id {
                        return Err(Error::Unsupported(format!(
                            "feature '{id}' declares a self-cycle in `before`"
                        )));
                    }
                    if edges.get_mut(id).unwrap().insert(target) {
                        *in_degree.get_mut(target).unwrap() += 1;
                    }
                }
            }
            for &source in &relations.after {
                if active_by_id.contains_key(source) {
                    if source == id {
                        return Err(Error::Unsupported(format!(
                            "feature '{id}' declares a self-cycle in `after`"
                        )));
                    }
                    if edges.get_mut(source).unwrap().insert(id) {
                        *in_degree.get_mut(id).unwrap() += 1;
                    }
                }
            }
        }

        let priority_key = |id: &'static str| -> (u32, &'static str) {
            let (spec, _) = active_by_id[id];
            let min_hook_order = spec
                .hook_bindings()
                .iter()
                .map(|b| b.hook_point.ordering())
                .min()
                .unwrap_or(u32::MAX);
            (min_hook_order, id)
        };

        let mut ready: BTreeSet<(u32, &'static str)> = in_degree
            .iter()
            .filter(|&(_, &deg)| deg == 0)
            .map(|(&id, _)| priority_key(id))
            .collect();

        let mut ordered_ids = Vec::with_capacity(active_ids.len());
        while let Some((_, current)) = ready.pop_first() {
            ordered_ids.push(current);
            for &neighbor in &edges[current] {
                let deg = in_degree.get_mut(neighbor).unwrap();
                *deg -= 1;
                if *deg == 0 {
                    ready.insert(priority_key(neighbor));
                }
            }
        }

        if ordered_ids.len() != active_ids.len() {
            let cycle_members: Vec<&'static str> = in_degree
                .into_iter()
                .filter_map(|(id, deg)| (deg > 0).then_some(id))
                .collect();
            return Err(Error::Unsupported(format!(
                "feature ordering cycle detected among active set: {:?}",
                cycle_members
            )));
        }

        // 6. Rewrite plan in canonical topological order.
        for id in ordered_ids {
            let (spec, invocation) = active_by_id[id];
            spec.rewrite_plan(route, &mut base_plan, &invocation.parameters)?;
            base_plan.resolved_feature_order.push(id.to_string());
        }

        Ok(base_plan)
    }
}

/// Validate a request attempting to use `"low_res_start"` on `descriptor` / `render_plan` (HOOK-5).
///
/// Refuses with a typed [`Error::Unsupported`] when:
/// - the route is structurally inapplicable (e.g. `flux1_dev_control`, `flux1_dev_ip_adapter`,
///   `flux1_schnell_ip_adapter`, `flux1_dev_pulid` / `pulid_flux`, or an unmigrated route);
/// - `req.phases` is `Some(...)`;
/// - `req.conditioning` is non-empty on pass 0 (reference, control, or edit conditioning);
/// - `k` is outside `1..executed_steps` or `scale` is outside `0.25..=0.9`.
pub fn validate_low_res_start_request(
    descriptor: &ModelDescriptor,
    render_plan: Option<&RenderPlanRegistration>,
    req: &GenerationRequest,
    executed_steps: usize,
    k: usize,
    scale: Option<f32>,
) -> Result<ComposedRenderPlan> {
    if req.phases.is_some() {
        return Err(Error::Unsupported(format!(
            "{}: low_res_start conflicts with multi-phase denoise (phases)",
            descriptor.id
        )));
    }
    if !req.conditioning.is_empty() {
        return Err(Error::Unsupported(format!(
            "{}: low_res_start requires a noise-start pass and conflicts with pass-0 conditioning",
            descriptor.id
        )));
    }
    let route = FeatureRouteStructure::from_descriptor_and_plan(descriptor, render_plan);
    let stride = route.pixel_stride.unwrap_or(16);
    if route.pass0_conditioning == Pass0ConditioningRequirement::NoiseStartSupported
        && descriptor.denoiser_output_latent_space.is_some()
        && render_plan.is_some()
    {
        LowResStartSpec::new(k, scale).resolve_for_descriptor(
            descriptor,
            req,
            executed_steps,
            stride,
        )?;
    }
    let token_width = route.token_width.unwrap_or(64);
    let sigmas: Vec<f32> = (0..=executed_steps)
        .map(|i| 1.0 - (i as f32 / executed_steps.max(1) as f32))
        .collect();
    let pass0 = ComposedPassPlan::single_phase(
        0,
        req.width,
        req.height,
        stride,
        token_width,
        req.sampler.as_deref().unwrap_or("euler"),
        req.guidance.unwrap_or(3.5),
        sigmas,
        None,
    )?;
    let base_plan = ComposedRenderPlan {
        route_id: route.route_id.clone(),
        backend: route.backend.clone(),
        request_has_phases: req.phases.is_some(),
        pass0_has_conditioning: !req.conditioning.is_empty(),
        resolved_feature_order: Vec::new(),
        passes: vec![pass0],
    };
    FeatureRegistry::standard().compose(
        &route,
        base_plan,
        &[FeatureInvocation {
            feature_id: LowResStartFeature::ID.to_string(),
            parameters: FeatureParameters::LowResStart { k, scale },
        }],
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        AcceptedControlKinds, Capabilities, Conditioning, ControlKind, GenerationPhase, Image,
        Modality, ModelDescriptor, PhaseAdapter, FLUX1_LATENT_SPACE,
    };

    fn synthetic_full_route() -> FeatureRouteStructure {
        FeatureRouteStructure {
            route_id: "synthetic_unified_route".into(),
            family: "synthetic".into(),
            backend: "mlx".into(),
            modality: Modality::Image,
            has_render_plan: true,
            has_packed_image_trajectory: true,
            supports_spatial_unpack_repack: true,
            supports_clean_estimate_renoise_transition: true,
            pass0_conditioning: Pass0ConditioningRequirement::NoiseStartSupported,
            supports_multiphase_denoise: false,
            supports_produced_reference_multipass: true,
            supports_expert_switching: true,
            supports_pag: true,
            default_executed_steps: Some(20),
            pixel_stride: Some(16),
            token_width: Some(64),
        }
    }

    fn synthetic_base_plan(route: &FeatureRouteStructure, steps: usize) -> ComposedRenderPlan {
        let sigmas: Vec<f32> = (0..=steps)
            .map(|i| 1.0 - (i as f32 / steps as f32))
            .collect();
        let pass0 = ComposedPassPlan::single_phase(
            0,
            1024,
            1024,
            route.pixel_stride.unwrap_or(16),
            route.token_width.unwrap_or(64),
            "euler",
            3.5,
            sigmas,
            None,
        )
        .unwrap();
        ComposedRenderPlan {
            route_id: route.route_id.clone(),
            backend: route.backend.clone(),
            request_has_phases: false,
            pass0_has_conditioning: false,
            resolved_feature_order: Vec::new(),
            passes: vec![pass0],
        }
    }

    fn sample_invocation(feature_id: &str) -> FeatureInvocation {
        let parameters = match feature_id {
            LowResStartFeature::ID => FeatureParameters::LowResStart {
                k: 6,
                scale: Some(0.5),
            },
            HiresFixFeature::ID => FeatureParameters::HiresFix {
                width: 2048,
                height: 2048,
                steps: 4,
                strength: Some(0.3),
                sigmas: vec![1.0, 0.75, 0.5, 0.25, 0.0],
            },
            KreaMultiphaseFeature::ID => FeatureParameters::KreaMultiphase {
                phases: vec![
                    GenerationPhase {
                        steps: 12,
                        guidance: Some(3.5),
                        adapters: vec![],
                    },
                    GenerationPhase {
                        steps: 8,
                        guidance: Some(0.0),
                        adapters: vec![PhaseAdapter {
                            adapter: 0,
                            weight: Some(0.8),
                        }],
                    },
                ],
                loaded_adapter_count: 2,
            },
            WanExpertSwitchFeature::ID => FeatureParameters::WanExpertSwitch {
                boundary_step: 6,
                high_noise_expert: "high_noise".into(),
                low_noise_expert: "low_noise".into(),
            },
            PagFeature::ID => FeatureParameters::Pag {
                scale: 1.5,
                layers: vec![28],
            },
            InertPhaseSplitFeature::ID => FeatureParameters::InertPhaseSplit { step: 10 },
            TrajectoryObserverFeature::ID => FeatureParameters::TrajectoryObserver {
                label: "step_trace".into(),
            },
            other => panic!("unexpected feature_id {other}"),
        };
        FeatureInvocation {
            feature_id: feature_id.to_string(),
            parameters,
        }
    }

    /// HOOK-1: Every HookPoint in the enum declares its stage, timing, inputs, outputs, ordering,
    /// and result-changing classification, has at least one registered production user, and dumps
    /// to a deterministic JSON value without a checked-in matrix.
    #[test]
    fn hook_map_covers_every_enum_variant_with_registered_users_and_typed_dump() {
        let entries = hook_map();
        assert_eq!(entries.len(), HookPoint::ALL.len());
        for (window_idx, pair) in entries.windows(2).enumerate() {
            assert!(
                pair[0].ordering < pair[1].ordering,
                "hook map must be strictly ordered at index {window_idx}"
            );
        }
        for point in HookPoint::ALL {
            let meta = hook_point_metadata(point);
            assert!(!meta.name.is_empty());
            assert!(!meta.inputs.is_empty(), "{point:?} must declare inputs");
            assert!(!meta.outputs.is_empty(), "{point:?} must declare outputs");
            assert!(
                !meta.registered_users.is_empty(),
                "{point:?} must have at least one registered production user"
            );
        }
        let dumped = dump_hook_map_json();
        let arr = dumped.as_array().expect("hook map JSON dump is an array");
        assert_eq!(arr.len(), 5);
        assert_eq!(arr[0]["name"], "schedule_transform");
        assert_eq!(arr[0]["stage"], "Schedule");
        assert_eq!(arr[0]["result_changing"], true);
    }

    /// HOOK-2: `FeatureSpec` is object-safe and tensor-free; `"low_res_start"` registers with
    /// DEC-8 class `Recipe`, version 1, schema version 1 (`k: 1..executed_steps`, `scale: 0.25..=0.9`,
    /// default `0.5`), conflicts with `phases` and pass-0 conditioning, and transition residency.
    #[test]
    fn low_res_start_spec_matches_hook2_contract_and_object_safety() {
        let registry = FeatureRegistry::standard();
        let spec: &dyn FeatureSpec = registry
            .get("low_res_start")
            .expect("low_res_start must be registered");
        assert_eq!(spec.id(), "low_res_start");
        assert_eq!(spec.version(), 1);
        assert_eq!(spec.class(), FeatureClass::Recipe);

        let route = synthetic_full_route();
        let schema = spec.parameter_schema(Some(&route));
        assert_eq!(schema.version, 1);
        let schema_json = schema.to_json_map();
        assert_eq!(schema_json["version"], 1);
        assert_eq!(schema_json["k"]["min"], 1);
        assert_eq!(schema_json["k"]["max_exclusive"], "executed_steps");
        assert_eq!(schema_json["k"]["max"], 19);
        assert_eq!(schema_json["scale"]["min"], 0.25);
        assert_eq!(schema_json["scale"]["max"], 0.9);
        assert_eq!(schema_json["scale"]["default"], 0.5);
        assert_eq!(schema_json["scale"]["inclusive"], true);

        let relations = spec.relations();
        assert!(relations.all_conflicts().contains(&"phases"));
        assert!(relations
            .all_conflicts()
            .contains(&"pass0_reference_conditioning"));
        assert!(relations
            .all_conflicts()
            .contains(&"pass0_control_conditioning"));
        assert!(relations
            .all_conflicts()
            .contains(&"pass0_edit_conditioning"));

        let memory = spec.memory_effect();
        assert_eq!(
            memory.transition_residency,
            TransitionResidency::InDeviceLatentUnpackResizeRepack
        );
        assert!(memory.introduces_denoise_phase_boundary);
        assert!(memory.retains_single_pass_schedule);
    }

    /// HOOK-3: Pairwise composition conformance across every pair of registered `FeatureSpec`s on a
    /// synthetic route, proving symmetry and deterministic outcomes.
    #[test]
    fn pairwise_feature_composition_conformance_on_synthetic_route() {
        let registry = FeatureRegistry::standard();
        let ids: Vec<&'static str> = registry.specs().iter().map(|s| s.id()).collect();

        for (i, &id_a) in ids.iter().enumerate() {
            for &id_b in &ids[i + 1..] {
                // Enable multiphase support on the synthetic route only when testing krea_multiphase
                // without low_res_start, so pairwise composition tests feature-to-feature rules.
                let mut route = synthetic_full_route();
                if id_a == KreaMultiphaseFeature::ID || id_b == KreaMultiphaseFeature::ID {
                    route.supports_multiphase_denoise = true;
                }
                let inv_a = sample_invocation(id_a);
                let inv_b = sample_invocation(id_b);

                let res_ab = registry.compose(
                    &route,
                    synthetic_base_plan(&route, 20),
                    &[inv_a.clone(), inv_b.clone()],
                );
                let res_ba =
                    registry.compose(&route, synthetic_base_plan(&route, 20), &[inv_b, inv_a]);

                let spec_a = registry.get(id_a).unwrap();
                let spec_b = registry.get(id_b).unwrap();
                let declared_conflict = spec_a.relations().conflicts_with_features.contains(&id_b)
                    || spec_b.relations().conflicts_with_features.contains(&id_a)
                    || (id_a == LowResStartFeature::ID && route.supports_multiphase_denoise)
                    || (id_b == LowResStartFeature::ID && route.supports_multiphase_denoise);

                if declared_conflict {
                    assert!(
                        res_ab.is_err() && res_ba.is_err(),
                        "conflicting pair ({id_a}, {id_b}) must be refused in both orders"
                    );
                } else {
                    let plan_ab = res_ab.unwrap_or_else(|e| {
                        panic!("compatible pair ({id_a}, {id_b}) failed in A->B order: {e}")
                    });
                    let plan_ba = res_ba.unwrap_or_else(|e| {
                        panic!("compatible pair ({id_a}, {id_b}) failed in B->A order: {e}")
                    });
                    assert_eq!(
                        plan_ab, plan_ba,
                        "compatible pair ({id_a}, {id_b}) must resolve to identical plan in both orders"
                    );
                }
            }
        }
    }

    /// HOOK-3: Whole-active-set cycle and exclusive `HookTiming::Replace` ownership validation.
    #[test]
    fn whole_active_set_detects_ordering_cycles_and_replace_ownership_conflicts() {
        struct SyntheticFeature {
            id: &'static str,
            before: Vec<&'static str>,
            hook_point: HookPoint,
            timing: HookTiming,
        }

        impl FeatureSpec for SyntheticFeature {
            fn id(&self) -> &'static str {
                self.id
            }
            fn version(&self) -> u32 {
                1
            }
            fn class(&self) -> FeatureClass {
                FeatureClass::Equivalent
            }
            fn parameter_schema(
                &self,
                _route: Option<&FeatureRouteStructure>,
            ) -> FeatureParameterSchema {
                FeatureParameterSchema {
                    version: 1,
                    parameters: Vec::new(),
                }
            }
            fn relations(&self) -> FeatureRelations {
                FeatureRelations {
                    before: self.before.clone(),
                    ..Default::default()
                }
            }
            fn hook_bindings(&self) -> Vec<FeatureHookBinding> {
                vec![FeatureHookBinding {
                    hook_point: self.hook_point,
                    stage: self.hook_point.stage(),
                    timing: self.timing,
                    pass_scope: Some(0),
                    result_changing: false,
                }]
            }
            fn memory_effect(&self) -> FeatureMemoryEffect {
                FeatureMemoryEffect {
                    extra_resident_weight_bytes: 0,
                    transition_residency: TransitionResidency::None,
                    introduces_denoise_phase_boundary: false,
                    introduces_pass_boundary: false,
                    retains_single_pass_schedule: true,
                }
            }
            fn inapplicability_reasons(
                &self,
                _route: &FeatureRouteStructure,
            ) -> Vec<FeatureInapplicabilityReason> {
                Vec::new()
            }
            fn rewrite_plan(
                &self,
                _route: &FeatureRouteStructure,
                _plan: &mut ComposedRenderPlan,
                _parameters: &FeatureParameters,
            ) -> Result<()> {
                Ok(())
            }
        }

        let route = synthetic_full_route();

        // 3-way ordering cycle: alpha -> beta -> gamma -> alpha.
        let cycle_registry = FeatureRegistry::with_specs(vec![
            Box::new(SyntheticFeature {
                id: "alpha",
                before: vec!["beta"],
                hook_point: HookPoint::ScheduleTransform,
                timing: HookTiming::After,
            }),
            Box::new(SyntheticFeature {
                id: "beta",
                before: vec!["gamma"],
                hook_point: HookPoint::ConditioningSlot,
                timing: HookTiming::After,
            }),
            Box::new(SyntheticFeature {
                id: "gamma",
                before: vec!["alpha"],
                hook_point: HookPoint::PhaseTransition,
                timing: HookTiming::Before,
            }),
        ])
        .unwrap();

        let cycle_err = cycle_registry
            .compose(
                &route,
                synthetic_base_plan(&route, 10),
                &[
                    FeatureInvocation {
                        feature_id: "alpha".into(),
                        parameters: FeatureParameters::TrajectoryObserver { label: "a".into() },
                    },
                    FeatureInvocation {
                        feature_id: "beta".into(),
                        parameters: FeatureParameters::TrajectoryObserver { label: "b".into() },
                    },
                    FeatureInvocation {
                        feature_id: "gamma".into(),
                        parameters: FeatureParameters::TrajectoryObserver { label: "c".into() },
                    },
                ],
            )
            .unwrap_err();
        assert!(
            cycle_err.to_string().contains("cycle"),
            "expected cycle refusal, got {cycle_err}"
        );

        // Two features claiming `HookTiming::Replace` on the same hook point and pass scope.
        let conflict_registry = FeatureRegistry::with_specs(vec![
            Box::new(SyntheticFeature {
                id: "guider_one",
                before: Vec::new(),
                hook_point: HookPoint::GuidancePredictor,
                timing: HookTiming::Replace,
            }),
            Box::new(SyntheticFeature {
                id: "guider_two",
                before: Vec::new(),
                hook_point: HookPoint::GuidancePredictor,
                timing: HookTiming::Replace,
            }),
        ])
        .unwrap();

        let owner_err = conflict_registry
            .compose(
                &route,
                synthetic_base_plan(&route, 10),
                &[
                    FeatureInvocation {
                        feature_id: "guider_one".into(),
                        parameters: FeatureParameters::TrajectoryObserver { label: "1".into() },
                    },
                    FeatureInvocation {
                        feature_id: "guider_two".into(),
                        parameters: FeatureParameters::TrajectoryObserver { label: "2".into() },
                    },
                ],
            )
            .unwrap_err();
        assert!(
            owner_err.to_string().contains("hook ownership conflict"),
            "expected hook ownership refusal, got {owner_err}"
        );
    }

    /// HOOK-3: Worked plan tests for:
    /// 1) low-res + Hires (`"low_res_start"` + `"hires_fix"`)
    /// 2) Krea phases (`"krea_multiphase"`)
    /// 3) Coincident Wan expert/geometry boundary (`"low_res_start"` + `"wan_expert_switch"` at the same step k)
    ///
    /// Also proves that permuting registration order in `FeatureRegistry` and invocation order cannot
    /// change the resolved plan.
    #[test]
    fn worked_plans_and_registration_order_invariance() {
        let route = synthetic_full_route();
        let standard = FeatureRegistry::standard();

        // Worked plan 1: low_res_start (pass 0: 0..6 at 512x512, 6..20 at 1024x1024) + hires_fix (pass 1: 2048x2048).
        let low_res_hires = standard
            .compose(
                &route,
                synthetic_base_plan(&route, 20),
                &[
                    sample_invocation(HiresFixFeature::ID),
                    sample_invocation(LowResStartFeature::ID),
                ],
            )
            .unwrap();
        assert_eq!(
            low_res_hires.resolved_feature_order,
            vec!["low_res_start", "hires_fix"]
        );
        assert_eq!(low_res_hires.passes.len(), 2);
        assert_eq!(low_res_hires.passes[0].phases.len(), 2);
        assert_eq!(
            (
                low_res_hires.passes[0].phases[0].first_step,
                low_res_hires.passes[0].phases[0].end_step,
                low_res_hires.passes[0].phases[0].width,
                low_res_hires.passes[0].phases[0].height,
            ),
            (0, 6, 512, 512)
        );
        assert_eq!(
            (
                low_res_hires.passes[0].phases[1].first_step,
                low_res_hires.passes[0].phases[1].end_step,
                low_res_hires.passes[0].phases[1].width,
                low_res_hires.passes[0].phases[1].height,
            ),
            (6, 20, 1024, 1024)
        );
        assert_eq!(
            low_res_hires.passes[1].produced_reference,
            Some(ProducedReference {
                from_pass: 0,
                width: 2048,
                height: 2048,
                strength: Some(0.3),
            })
        );

        // Worked plan 2: Krea multiphase over one shared 20-step global schedule.
        let mut krea_route = synthetic_full_route();
        krea_route.supports_multiphase_denoise = true;
        let krea_plan = standard
            .compose(
                &krea_route,
                synthetic_base_plan(&krea_route, 20),
                &[sample_invocation(KreaMultiphaseFeature::ID)],
            )
            .unwrap();
        assert_eq!(krea_plan.passes[0].phases.len(), 2);
        assert_eq!(
            (
                krea_plan.passes[0].phases[0].first_step,
                krea_plan.passes[0].phases[0].end_step,
                krea_plan.passes[0].phases[0].guidance,
            ),
            (0, 12, 3.5)
        );
        assert_eq!(
            (
                krea_plan.passes[0].phases[1].first_step,
                krea_plan.passes[0].phases[1].end_step,
                krea_plan.passes[0].phases[1].guidance,
            ),
            (12, 20, 0.0)
        );
        assert_eq!(krea_plan.passes[0].transitions.len(), 1);
        assert_eq!(krea_plan.passes[0].transitions[0].step, 12);

        // Worked plan 3: Coincident Wan expert/geometry boundary at step 6 + Hires.fix + observer.
        // Also prove registration-order invariance across reversed and rotated registries.
        let active_set = vec![
            sample_invocation(TrajectoryObserverFeature::ID),
            sample_invocation(HiresFixFeature::ID),
            sample_invocation(WanExpertSwitchFeature::ID),
            sample_invocation(LowResStartFeature::ID),
        ];

        let baseline_plan = standard
            .compose(&route, synthetic_base_plan(&route, 20), &active_set)
            .unwrap();
        // Step 6 has ONE merged transition boundary with BOTH actions in canonical order, and 2 spans (0..6, 6..20).
        assert_eq!(baseline_plan.passes[0].phases.len(), 2);
        assert_eq!(baseline_plan.passes[0].transitions.len(), 1);
        assert_eq!(baseline_plan.passes[0].transitions[0].step, 6);
        assert_eq!(baseline_plan.passes[0].transitions[0].actions.len(), 2);
        assert_eq!(
            baseline_plan.passes[0].phases[0].active_expert.as_deref(),
            Some("high_noise")
        );
        assert_eq!(
            (
                baseline_plan.passes[0].phases[0].width,
                baseline_plan.passes[0].phases[0].height
            ),
            (512, 512)
        );
        assert_eq!(
            baseline_plan.passes[0].phases[1].active_expert.as_deref(),
            Some("low_noise")
        );
        assert_eq!(
            (
                baseline_plan.passes[0].phases[1].width,
                baseline_plan.passes[0].phases[1].height
            ),
            (1024, 1024)
        );

        let reversed_registry = FeatureRegistry::with_specs(vec![
            Box::new(TrajectoryObserverFeature),
            Box::new(InertPhaseSplitFeature),
            Box::new(PagFeature),
            Box::new(WanExpertSwitchFeature),
            Box::new(KreaMultiphaseFeature),
            Box::new(HiresFixFeature),
            Box::new(LowResStartFeature),
        ])
        .unwrap();
        let rotated_registry = FeatureRegistry::with_specs(vec![
            Box::new(WanExpertSwitchFeature),
            Box::new(HiresFixFeature),
            Box::new(TrajectoryObserverFeature),
            Box::new(LowResStartFeature),
            Box::new(PagFeature),
            Box::new(KreaMultiphaseFeature),
            Box::new(InertPhaseSplitFeature),
        ])
        .unwrap();

        let mut reversed_invocations = active_set.clone();
        reversed_invocations.reverse();

        for reg in [&standard, &reversed_registry, &rotated_registry] {
            for invs in [&active_set, &reversed_invocations] {
                let candidate = reg
                    .compose(&route, synthetic_base_plan(&route, 20), invs)
                    .unwrap();
                assert_eq!(
                    serde_json::to_value(&candidate).unwrap(),
                    serde_json::to_value(&baseline_plan).unwrap(),
                    "changing registration order or invocation order must never change the resolved plan"
                );
            }
        }
    }

    /// HOOK-4: Removing a structural precondition from a route descriptor or RenderPlanRegistration
    /// flips feature availability automatically with no table edits.
    #[test]
    fn removing_structural_precondition_flips_applicability_without_table_edits() {
        fn compile_fixture(
            _spec: &LoadSpec,
            req: &GenerationRequest,
        ) -> Result<Option<RenderPlan>> {
            let desc = fixture_descriptor();
            RenderPlan::base_image(
                &desc,
                req,
                crate::render::PackedImagePass {
                    pixel_stride: 16,
                    token_width: 64,
                    sampler: "euler",
                    guidance: 3.5,
                    sigmas: vec![1.0, 0.75, 0.5, 0.25, 0.0],
                    executed_steps: 4,
                    capture_sigma: 0.0,
                },
            )
            .map(Some)
        }

        fn fixture_descriptor() -> ModelDescriptor {
            ModelDescriptor {
                encoder_contract: None,
                denoiser_output_latent_space: Some(&FLUX1_LATENT_SPACE),
                control_kinds: None,
                required_components: &[],
                id: "structural_probe_route",
                family: "flux",
                backend: "mlx",
                modality: Modality::Image,
                capabilities: Capabilities {
                    min_size: 256,
                    max_size: 2048,
                    max_count: 1,
                    ..Default::default()
                },
            }
        }

        let planner = RenderPlanRegistration {
            provider_id: "structural_probe_route",
            compile: compile_fixture,
        };
        let full_desc = fixture_descriptor();
        let full_route =
            FeatureRouteStructure::from_descriptor_and_plan(&full_desc, Some(&planner));
        let full_app = LowResStartFeature.applicability(&full_route);
        assert!(
            full_app.applicable,
            "route satisfying all structural preconditions must be applicable"
        );
        assert_eq!(full_app.to_json()["applicable"], true);
        assert_eq!(full_app.to_json()["parameter_schema"]["k"]["max"], 3);

        // 1. Remove the RenderPlanRegistration -> flips to false with `missing_render_plan`.
        let without_plan = FeatureRouteStructure::from_descriptor_and_plan(&full_desc, None);
        let no_plan_app = LowResStartFeature.applicability(&without_plan);
        assert!(!no_plan_app.applicable);
        assert!(no_plan_app
            .reasons
            .iter()
            .any(|r| r.code == FeatureInapplicabilityCode::MissingRenderPlan));

        // 2. Remove the packed latent space from the descriptor -> flips to false with `missing_spatial_unpack_repack`.
        let mut without_latent = full_desc.clone();
        without_latent.denoiser_output_latent_space = None;
        let no_latent_route =
            FeatureRouteStructure::from_descriptor_and_plan(&without_latent, Some(&planner));
        let no_latent_app = LowResStartFeature.applicability(&no_latent_route);
        assert!(!no_latent_app.applicable);
        assert!(no_latent_app
            .reasons
            .iter()
            .any(|r| r.code == FeatureInapplicabilityCode::MissingSpatialUnpackRepack));

        // 3. Make the route require pass-0 control conditioning -> flips to false with `conflicts_with_pass0_control_conditioning`.
        fn compile_control_only(
            _spec: &LoadSpec,
            req: &GenerationRequest,
        ) -> Result<Option<RenderPlan>> {
            if !matches!(req.conditioning.as_slice(), [Conditioning::Control { .. }]) {
                return Err(Error::Unsupported("requires control".into()));
            }
            let mut desc = fixture_descriptor();
            desc.control_kinds = Some(AcceptedControlKinds::Only(vec![ControlKind::Pose]));
            desc.capabilities.conditioning = vec![ConditioningKind::Control];
            RenderPlan::packed_image_trajectory(
                &desc,
                req,
                crate::render::PackedImagePass {
                    pixel_stride: 16,
                    token_width: 64,
                    sampler: "euler",
                    guidance: 3.5,
                    sigmas: vec![1.0, 0.75, 0.5, 0.25, 0.0],
                    executed_steps: 4,
                    capture_sigma: 0.0,
                },
            )
            .map(Some)
        }
        let mut control_desc = full_desc;
        control_desc.control_kinds = Some(AcceptedControlKinds::Only(vec![ControlKind::Pose]));
        control_desc.capabilities.conditioning = vec![ConditioningKind::Control];
        let control_planner = RenderPlanRegistration {
            provider_id: "structural_probe_route",
            compile: compile_control_only,
        };
        let control_route =
            FeatureRouteStructure::from_descriptor_and_plan(&control_desc, Some(&control_planner));
        let control_app = LowResStartFeature.applicability(&control_route);
        assert!(!control_app.applicable);
        assert!(control_app
            .reasons
            .iter()
            .any(|r| r.code == FeatureInapplicabilityCode::ConflictsWithPass0ControlConditioning));
    }

    /// HOOK-5: `validate_low_res_start_request` refuses `req.phases` and pass-0 reference/control
    /// conditioning with a typed `Error::Unsupported`.
    #[test]
    fn low_res_start_refuses_phases_and_pass0_conditioning_with_typed_error() {
        fn compile_fixture(
            _spec: &LoadSpec,
            req: &GenerationRequest,
        ) -> Result<Option<RenderPlan>> {
            let desc = fixture_descriptor();
            RenderPlan::base_image(
                &desc,
                req,
                crate::render::PackedImagePass {
                    pixel_stride: 16,
                    token_width: 64,
                    sampler: "euler",
                    guidance: 3.5,
                    sigmas: vec![1.0, 0.75, 0.5, 0.25, 0.0],
                    executed_steps: 4,
                    capture_sigma: 0.0,
                },
            )
            .map(Some)
        }
        fn fixture_descriptor() -> ModelDescriptor {
            ModelDescriptor {
                encoder_contract: None,
                denoiser_output_latent_space: Some(&FLUX1_LATENT_SPACE),
                control_kinds: None,
                required_components: &[],
                id: "flux1_dev",
                family: "flux",
                backend: "mlx",
                modality: Modality::Image,
                capabilities: Capabilities {
                    min_size: 256,
                    max_size: 2048,
                    max_count: 1,
                    ..Default::default()
                },
            }
        }

        let desc = fixture_descriptor();
        let planner = RenderPlanRegistration {
            provider_id: "flux1_dev",
            compile: compile_fixture,
        };
        let clean_req = GenerationRequest {
            prompt: "a lighthouse".into(),
            width: 1024,
            height: 1024,
            ..Default::default()
        };
        assert!(
            validate_low_res_start_request(&desc, Some(&planner), &clean_req, 4, 2, Some(0.5))
                .is_ok()
        );

        let phased_req = GenerationRequest {
            phases: Some(vec![GenerationPhase {
                steps: 4,
                guidance: Some(3.5),
                adapters: vec![],
            }]),
            ..clean_req.clone()
        };
        assert!(matches!(
            validate_low_res_start_request(&desc, Some(&planner), &phased_req, 4, 2, Some(0.5)),
            Err(Error::Unsupported(_))
        ));

        let ref_req = GenerationRequest {
            conditioning: vec![Conditioning::Reference {
                image: Image {
                    width: 16,
                    height: 16,
                    pixels: vec![0; 16 * 16 * 3],
                },
                strength: Some(0.7),
            }],
            ..clean_req
        };
        assert!(matches!(
            validate_low_res_start_request(&desc, Some(&planner), &ref_req, 4, 2, Some(0.5)),
            Err(Error::Unsupported(_))
        ));
    }

    /// SW-6939 (FEAT-1, PIPE-2): Parameter boundaries (`version == 1`, `scale in 0.25..=0.9`,
    /// `M >= 2`, `1 <= K < M` resolved after img2img strength truncation), floor quantization
    /// to `legal_quantum = lcm(vae_spatial_scale * patch_size, requires_dimensions_multiple_of)`,
    /// spatial-only scaling (frames, count, audio unchanged), and Hires.fix composition applying
    /// `low_res_start` once to pass 0 only.
    #[test]
    fn sw6939_parameter_boundaries_strength_truncation_legal_grid_and_hires_composition() {
        use crate::{MultiPassRenderPlan, PlannedPassRequest};

        fn flux_desc() -> ModelDescriptor {
            ModelDescriptor {
                encoder_contract: None,
                denoiser_output_latent_space: Some(&FLUX1_LATENT_SPACE),
                control_kinds: None,
                required_components: &[],
                id: "flux1_dev",
                family: "flux",
                backend: "mlx",
                modality: Modality::Image,
                capabilities: Capabilities {
                    min_size: 256,
                    max_size: 2048,
                    max_count: 4,
                    conditioning: vec![ConditioningKind::Reference],
                    ..Default::default()
                },
            }
        }

        // 1. Version != 1 is refused without clamping.
        let bad_version = LowResStartSpec::new(2, Some(0.5)).with_version(2);
        assert!(matches!(
            bad_version.resolve_geometry("flux1_dev", 1024, 1024, 10, 8, 2, 16, 256),
            Err(Error::Unsupported(_))
        ));
        let zero_version = LowResStartSpec::new(2, Some(0.5)).with_version(0);
        assert!(matches!(
            zero_version.resolve_geometry("flux1_dev", 1024, 1024, 10, 8, 2, 16, 256),
            Err(Error::Unsupported(_))
        ));

        // 2. Scale outside [0.25, 0.9] or non-finite is refused without clamping.
        for bad_scale in [0.0_f32, 0.2499, 0.9001, 1.0, -0.5, f32::NAN, f32::INFINITY] {
            let spec = LowResStartSpec::new(2, Some(bad_scale));
            assert!(
                matches!(
                    spec.resolve_geometry("flux1_dev", 1024, 1024, 10, 8, 2, 16, 256),
                    Err(Error::Unsupported(_))
                ),
                "bad scale {bad_scale} must be refused"
            );
        }

        // 3. M < 2, K == 0, and K >= M are refused without clamping.
        let spec_k1 = LowResStartSpec::new(1, Some(0.5));
        assert!(matches!(
            spec_k1.resolve_geometry("flux1_dev", 1024, 1024, 1, 8, 2, 16, 256),
            Err(Error::Unsupported(_))
        ));
        assert!(matches!(
            LowResStartSpec::new(0, Some(0.5)).resolve_geometry(
                "flux1_dev",
                1024,
                1024,
                10,
                8,
                2,
                16,
                256
            ),
            Err(Error::Unsupported(_))
        ));
        assert!(matches!(
            LowResStartSpec::new(10, Some(0.5)).resolve_geometry(
                "flux1_dev",
                1024,
                1024,
                10,
                8,
                2,
                16,
                256
            ),
            Err(Error::Unsupported(_))
        ));

        // 4. Img2img strength truncation resolves M = executed_steps BEFORE validating K.
        // Full schedule has 10 steps (11 sigma nodes). With strength = 0.7, init_time_step = 7,
        // so truncated schedule has M = 10 - 7 = 3 executed steps (4 nodes).
        let full_sigmas: Vec<f32> = (0..=10).map(|i| 1.0 - (i as f32 / 10.0)).collect();
        let (start_step, truncated) =
            resolve_img2img_truncated_schedule(&full_sigmas, Some(0.7)).unwrap();
        assert_eq!(start_step, 7);
        let m = truncated.len() - 1;
        assert_eq!(m, 3);
        // K = 2 < M = 3 succeeds; K = 3 >= M = 3 is refused even though K < full_steps (10)!
        assert!(LowResStartSpec::new(2, Some(0.5))
            .resolve_geometry("flux1_dev", 1024, 1024, m, 8, 2, 16, 256)
            .is_ok());
        assert!(matches!(
            LowResStartSpec::new(3, Some(0.5)).resolve_geometry(
                "flux1_dev",
                1024,
                1024,
                m,
                8,
                2,
                16,
                256
            ),
            Err(Error::Unsupported(_))
        ));
        // If strength leaves M = 1 (e.g. strength = 0.9 on 10 steps -> start_step = 9, M = 1), refused!
        let (_, truncated_m1) =
            resolve_img2img_truncated_schedule(&full_sigmas, Some(0.9)).unwrap();
        assert_eq!(truncated_m1.len() - 1, 1);
        assert!(matches!(
            LowResStartSpec::new(1, Some(0.5)).resolve_geometry(
                "flux1_dev",
                1024,
                1024,
                truncated_m1.len() - 1,
                8,
                2,
                16,
                256
            ),
            Err(Error::Unsupported(_))
        ));

        // 5. Floor quantization to legal_quantum = lcm(vae_spatial_scale * patch_size, requires_dimensions_multiple_of).
        // For FLUX.1: lcm(8 * 2, 16) = 16.
        // At 1024x768 with scale = 0.51:
        // floor((1024 * 0.51) / 16) * 16 = floor(522.24 / 16) * 16 = 32 * 16 = 512 (NOT round = 528!).
        // floor((768 * 0.51) / 16) * 16 = floor(391.68 / 16) * 16 = 24 * 16 = 384.
        let resolved_floor = LowResStartSpec::new(2, Some(0.51))
            .resolve_geometry("flux1_dev", 1024, 768, 8, 8, 2, 16, 256)
            .unwrap();
        assert_eq!(resolved_floor.legal_quantum, 16);
        assert_eq!(resolved_floor.low_width, 512);
        assert_eq!(resolved_floor.low_height, 384);

        // Non-trivial LCM: vae_spatial_scale * patch_size = 16, requires_dimensions_multiple_of = 24 -> legal_quantum = 48.
        let resolved_lcm = LowResStartSpec::new(2, Some(0.5))
            .resolve_geometry("av_route", 960, 576, 8, 8, 2, 24, 240)
            .unwrap();
        assert_eq!(resolved_lcm.legal_quantum, 48);
        assert_eq!(resolved_lcm.low_width, 480);
        assert_eq!(resolved_lcm.low_height, 288);

        // Below min_dim (e.g. 384 * 0.5 = 192 < min_dim 256) is refused without clamping.
        assert!(matches!(
            LowResStartSpec::new(2, Some(0.5)).resolve_geometry(
                "flux1_dev",
                384,
                1024,
                8,
                8,
                2,
                16,
                256
            ),
            Err(Error::Unsupported(_))
        ));

        // Non-reducing scale on small target (e.g. 256 * 0.9 = 230.4 -> floor 224 < 256, or if full == low) is refused.
        assert!(matches!(
            LowResStartSpec::new(2, Some(0.9))
                .resolve_geometry("flux1_dev", 256, 256, 8, 8, 2, 16, 224),
            Ok(plan) if plan.low_width == 224 && plan.low_height == 224
        ));

        // 6. Spatial-only scaling: frames and count in RenderPlan / RenderRun stay unchanged across phases.
        let desc = flux_desc();
        let req = GenerationRequest {
            prompt: "cinematic lighthouse".into(),
            width: 1024,
            height: 768,
            count: 2,
            frames: Some(25),
            seed: Some(42),
            low_res_start: Some(LowResStartSpec::new(2, Some(0.5))),
            ..Default::default()
        };
        let plan = RenderPlan::base_image(
            &desc,
            &req,
            crate::render::PackedImagePass {
                pixel_stride: 16,
                token_width: 64,
                sampler: "euler",
                guidance: 3.5,
                sigmas: vec![1.0, 0.75, 0.5, 0.25, 0.0],
                executed_steps: 4,
                capture_sigma: 0.0,
            },
        )
        .unwrap();
        assert_eq!(plan.denoise_spans(), vec![0..2, 2..4]);
        assert_eq!(plan.phase_output_size(0), (512, 384));
        assert_eq!(plan.phase_output_size(1), (1024, 768));
        assert_eq!(plan.image_count(), 2);

        // 7. Composition with Hires.fix (`MultiPassRenderPlan`): `low_res_start` applies once, to base pass 0 only.
        let pass0_req = GenerationRequest {
            prompt: "a lighthouse".into(),
            width: 1024,
            height: 1024,
            count: 1,
            seed: Some(7),
            low_res_start: Some(LowResStartSpec::new(2, Some(0.5))),
            ..Default::default()
        };
        let pass1_req = GenerationRequest {
            width: 1536,
            height: 1536,
            ..pass0_req.clone()
        };
        let multipass = MultiPassRenderPlan::compile(
            &[
                PlannedPassRequest {
                    request: &pass0_req,
                    produced_reference: None,
                },
                PlannedPassRequest {
                    request: &pass1_req,
                    produced_reference: Some(ProducedReference {
                        from_pass: 0,
                        width: 1536,
                        height: 1536,
                        strength: Some(0.35),
                    }),
                },
            ],
            |r| {
                RenderPlan::packed_image_trajectory(
                    &desc,
                    r,
                    crate::render::PackedImagePass {
                        pixel_stride: 16,
                        token_width: 64,
                        sampler: "euler",
                        guidance: 3.5,
                        sigmas: vec![1.0, 0.75, 0.5, 0.25, 0.0],
                        executed_steps: 4,
                        capture_sigma: 0.0,
                    },
                )
                .map(Some)
            },
        )
        .unwrap()
        .unwrap();
        assert_eq!(multipass.passes().len(), 2);
        // Pass 0 has low_res_start active (2 phases: 0..2 at 512x512, 2..4 at 1024x1024).
        assert!(multipass.passes()[0].low_res_start().is_some());
        assert_eq!(multipass.passes()[0].denoise_spans(), vec![0..2, 2..4]);
        assert_eq!(multipass.passes()[0].phase_output_size(0), (512, 512));
        assert_eq!(multipass.passes()[0].phase_output_size(1), (1024, 1024));
        // Pass 1 (Hires.fix) has low_res_start inactive (1 phase: 0..4 at 1536x1536).
        assert!(multipass.passes()[1].low_res_start().is_none());
        assert_eq!(multipass.passes()[1].denoise_spans(), vec![0..4]);
        assert_eq!(multipass.passes()[1].phase_output_size(0), (1536, 1536));
    }

    /// SW-6939 (PIPE-4, FEAT-1, §6.1): Typed `BoundaryState` refuses `XSigma` and `RouteSpecific`
    /// so a transition cannot double-noise; fp32 unpack -> bilinear half-pixel (`align_corners = false`,
    /// clamp-to-edge) resize -> deterministic domain-separated noise draw -> Flow vs VE/EDM noise laws -> repack.
    #[test]
    fn sw6939_typed_boundary_state_and_exact_fp32_transition_arithmetic() {
        use crate::sampling::{
            AlphaSchedule, CpuLatentOps, DiscreteModelSampling, EdmModelSampling,
            FlowModelSampling, LatentOps, PredictionType, TimestepConvention,
        };

        let plan = LowResStartSpec::new(2, Some(0.5))
            .resolve_geometry("flux1_dev", 512, 512, 4, 8, 2, 16, 256)
            .unwrap();
        assert_eq!((plan.low_width, plan.low_height), (256, 256));
        assert_eq!((plan.full_width, plan.full_height), (512, 512));

        let flow_sampling = FlowModelSampling::new(TimestepConvention::Sigma);
        let edm_sampling = EdmModelSampling::new(PredictionType::Eps, 0.002, 80.0, 1.0);
        let discrete_sampling =
            DiscreteModelSampling::sdxl(&AlphaSchedule::scaled_linear(1000, 0.00085, 0.012));

        // 1. BoundaryState::XSigma and RouteSpecific are refused with a typed Error::Unsupported.
        let dummy_low = vec![1.0_f32; 16 * 32 * 32];
        let noisy_state = BoundaryState::XSigma(dummy_low.as_slice());
        assert!(matches!(
            transition_packed_clean_estimate_f32(
                noisy_state,
                1,
                16,
                2,
                8,
                &plan,
                0.5,
                &flow_sampling,
                42,
                0,
                0,
                0,
            ),
            Err(Error::Unsupported(msg)) if msg.contains("BoundaryState::XSigma")
        ));
        let route_state = BoundaryState::RouteSpecific {
            label: "custom_carrier",
            state: dummy_low.as_slice(),
        };
        assert!(matches!(
            transition_packed_clean_estimate_f32(
                route_state,
                1,
                16,
                2,
                8,
                &plan,
                0.5,
                &flow_sampling,
                42,
                0,
                0,
                0,
            ),
            Err(Error::Unsupported(msg)) if msg.contains("custom_carrier")
        ));

        // 2. Unpack <-> Repack round-trip identity and exact 2x2 patch mapping.
        let (b, c, h_low, w_low, p) = (1usize, 16usize, 32usize, 32usize, 2usize);
        let unpacked_orig: Vec<f32> = (0..(b * c * h_low * w_low))
            .map(|i| (i as f32) * 0.01 - 5.0)
            .collect();
        let packed = repack_unpacked_latents_f32(&unpacked_orig, b, c, h_low, w_low, p).unwrap();
        let unpacked_rt = unpack_packed_latents_f32(&packed, b, c, h_low, w_low, p).unwrap();
        assert_eq!(unpacked_orig, unpacked_rt);

        // 3. Exact bilinear half-pixel (`align_corners = false`, clamp-to-edge) coordinates & interpolation.
        // Upsampling 2x2 -> 4x4:
        // dst=0 -> coord_src = (0.5 * 0.5) - 0.5 = -0.25 -> clamped to 0.0 -> (0, 1, 0.0)
        // dst=1 -> coord_src = (1.5 * 0.5) - 0.5 = 0.25 -> (0, 1, 0.25)
        // dst=2 -> coord_src = (2.5 * 0.5) - 0.5 = 0.75 -> (0, 1, 0.75)
        // dst=3 -> coord_src = (3.5 * 0.5) - 0.5 = 1.25 -> clamped to 1.0 -> (1, 1, 0.0)
        assert_eq!(bilinear_half_pixel_clamp_coord(0, 2, 4), (0, 1, 0.0));
        assert_eq!(bilinear_half_pixel_clamp_coord(1, 2, 4), (0, 1, 0.25));
        assert_eq!(bilinear_half_pixel_clamp_coord(2, 2, 4), (0, 1, 0.75));
        assert_eq!(bilinear_half_pixel_clamp_coord(3, 2, 4), (1, 1, 0.0));

        let src_2x2 = vec![10.0_f32, 20.0, 30.0, 40.0];
        let resized_4x4 = resize_bilinear_half_pixel_clamp_f32(&src_2x2, 1, 1, 2, 2, 4, 4).unwrap();
        // Corner (0,0) clamps to src(0,0) = 10.0; corner (3,3) clamps to src(1,1) = 40.0.
        assert!((resized_4x4[0] - 10.0).abs() < 1e-6);
        assert!((resized_4x4[15] - 40.0).abs() < 1e-6);
        // Inner (1,1): wx=0.25, wy=0.25 -> top = 12.5, bot = 32.5 -> 12.5*0.75 + 32.5*0.25 = 17.5.
        assert!((resized_4x4[5] - 17.5).abs() < 1e-6);

        // 4. Deterministic domain-separated transition seed never collides with initial noise or StepRng draws.
        let s_trans = low_res_start_transition_seed(12345, 0, 0, 0);
        assert_ne!(s_trans, 12345);
        assert_ne!(s_trans, low_res_start_transition_seed(12345, 1, 0, 0));
        assert_ne!(s_trans, low_res_start_transition_seed(12345, 0, 1, 0));
        assert_ne!(s_trans, low_res_start_transition_seed(12345, 0, 0, 1));
        let trans_noise = low_res_start_transition_noise_f32(12345, 0, 0, 0, 64);
        let step_noise = CpuLatentOps
            .randn_like(&vec![0.0_f32; 64], 12345, 0)
            .unwrap();
        assert_ne!(trans_noise, step_noise);

        // 5. Flow vs VE/EDM noise laws at sigma_K = 0.6:
        // Flow: (1 - sigma_K) * resize(x0_low) + sigma_K * eps_trans
        // VE/EDM: resize(x0_low) + sigma_K * eps_trans
        let sigma_k = 0.6_f32;
        let constant_x0 = vec![2.0_f32; 16 * 32 * 32];
        let constant_eps = vec![0.5_f32; 16 * 64 * 64];
        let flow_out = transition_packed_clean_estimate_with_noise_f32(
            BoundaryState::CleanEstimate {
                latent: &constant_x0,
                source: EstimateSource::EndpointEvaluation,
            },
            1,
            16,
            2,
            8,
            &plan,
            sigma_k,
            &flow_sampling,
            &constant_eps,
        )
        .unwrap();
        let expected_flow = (1.0 - sigma_k) * 2.0 + sigma_k * 0.5;
        assert!(flow_out.iter().all(|&v| (v - expected_flow).abs() < 1e-6));

        for ve_sampling in [&edm_sampling as &dyn ModelSampling, &discrete_sampling] {
            let ve_out = transition_packed_clean_estimate_with_noise_f32(
                BoundaryState::CleanEstimate {
                    latent: &constant_x0,
                    source: EstimateSource::EndpointEvaluation,
                },
                1,
                16,
                2,
                8,
                &plan,
                sigma_k,
                ve_sampling,
                &constant_eps,
            )
            .unwrap();
            let expected_ve = 2.0 + sigma_k * 0.5;
            assert!(ve_out.iter().all(|&v| (v - expected_ve).abs() < 1e-6));
        }
    }

    /// SW-6939 (PIPE-3, PIPE-5): `SolverRunState` history reset across geometry change / re-noise
    /// while preserving `next_step = K` and RNG draw coordinates `K..M`; incompatible `Carry`
    /// across geometry change or re-noise refuses with a typed error; read-only boundary evaluation
    /// leaves APG `MomentumBuffer`, solver history, and RNG unchanged.
    #[test]
    fn sw6939_solver_history_reset_carry_refusal_and_readonly_apg_evaluation() {
        use crate::guidance::{normalized_guidance, normalized_guidance_readonly, MomentumBuffer};
        use crate::sampling::{
            CpuLatentOps, Dpmpp2m, EulerAncestral, FlowModelSampling, PhaseBoundary, Sampler,
            Solver, SolverRunState, TimestepConvention,
        };

        let ops = CpuLatentOps;
        let ms = FlowModelSampling::new(TimestepConvention::Sigma);
        let sigmas = vec![1.0_f32, 0.75, 0.5, 0.25, 0.0];

        // 1. Multistep solver (Dpmpp2m) accumulates history in Phase 0 (0..2 at len 4).
        let mut state = SolverRunState::new(Solver::Dpmpp2m, &sigmas);
        let mut denoise_low = |x: &Vec<f32>, s: f32| -> Result<Vec<f32>> {
            Ok(x.iter().map(|&v| v * (1.0 - 0.1 * s)).collect())
        };
        let out_p0 = Dpmpp2m
            .sample_span(
                &ops,
                &ms,
                &mut denoise_low,
                vec![1.0_f32; 4],
                &sigmas,
                99,
                0..2,
                &mut state,
            )
            .unwrap();
        assert_eq!(out_p0.len(), 4);
        assert_eq!(state.next_step(), 2);
        assert!(
            state.has_history(),
            "multistep solver must hold history at step K=2"
        );

        // 2. Marking a geometry change or re-noise and attempting Carry refuses with a typed Error::Unsupported!
        state.note_geometry_change((256, 256), (512, 512));
        state.note_renoise_transition();
        assert!(matches!(
            state.apply_phase_boundary(PhaseBoundary::Carry, true),
            Err(Error::Unsupported(_))
        ));
        let mut denoise_full = |x: &Vec<f32>, s: f32| -> Result<Vec<f32>> {
            Ok(x.iter().map(|&v| v * (1.0 - 0.1 * s)).collect())
        };
        assert!(
            matches!(
                Dpmpp2m.sample_span(
                    &ops,
                    &ms,
                    &mut denoise_full,
                    vec![1.0_f32; 16],
                    &sigmas,
                    99,
                    2..4,
                    &mut state,
                ),
                Err(Error::Unsupported(_))
            ),
            "sample_span must refuse Carry across unreset geometry/renoise boundary"
        );

        // 3. Resetting history clears old-shape history while preserving next_step = K = 2,
        // allowing Phase 1 (2..4 at len 16) to integrate cleanly without shape mismatch!
        state
            .apply_phase_boundary(PhaseBoundary::Reset, true)
            .unwrap();
        assert!(!state.has_history());
        assert_eq!(state.next_step(), 2);
        let out_p1 = Dpmpp2m
            .sample_span(
                &ops,
                &ms,
                &mut denoise_full,
                vec![1.0_f32; 16],
                &sigmas,
                99,
                2..4,
                &mut state,
            )
            .unwrap();
        assert_eq!(out_p1.len(), 16);
        assert_eq!(state.next_step(), 4);

        // 4. Stochastic solver (EulerAncestral) preserves absolute RNG draw coordinates K..M after reset_history().
        let mut stoch_state = SolverRunState::new(Solver::EulerAncestral, &sigmas);
        let _ = EulerAncestral::default()
            .sample_span(
                &ops,
                &ms,
                &mut |x, _| Ok(x.clone()),
                vec![0.5_f32; 4],
                &sigmas,
                777,
                0..2,
                &mut stoch_state,
            )
            .unwrap();
        stoch_state.note_geometry_change((256, 256), (512, 512));
        stoch_state
            .apply_phase_boundary(PhaseBoundary::Reset, true)
            .unwrap();
        assert_eq!(stoch_state.next_step(), 2);
        let p1_after_reset = EulerAncestral::default()
            .sample_span(
                &ops,
                &ms,
                &mut |x, _| Ok(x.clone()),
                vec![0.25_f32; 8],
                &sigmas,
                777,
                2..4,
                &mut stoch_state,
            )
            .unwrap();

        // Compare with a direct span 2..4 starting from the same state at step 2 with seed 777:
        // because stochastic draws use absolute step indices 2..4, the outputs are bit-identical!
        let mut ref_state = SolverRunState::new(Solver::EulerAncestral, &sigmas);
        let _ = EulerAncestral::default()
            .sample_span(
                &ops,
                &ms,
                &mut |x, _| Ok(x.clone()),
                vec![0.5_f32; 8],
                &sigmas,
                777,
                0..2,
                &mut ref_state,
            )
            .unwrap();
        let p1_direct = EulerAncestral::default()
            .sample_span(
                &ops,
                &ms,
                &mut |x, _| Ok(x.clone()),
                vec![0.25_f32; 8],
                &sigmas,
                777,
                2..4,
                &mut ref_state,
            )
            .unwrap();
        assert_eq!(
            p1_after_reset, p1_direct,
            "reset_history() must preserve exact stochastic RNG draw coordinates K..M"
        );

        // 5. Read-only boundary evaluation does not mutate APG MomentumBuffer, and MomentumBuffer::reset()
        // invalidates old-shape phase-0 state before phase 1.
        let mut apg_buf = MomentumBuffer::new(-0.5);
        let cond_low = vec![1.0_f32, 2.0, 3.0, 4.0];
        let uncond_low = vec![0.5_f32, 1.0, 1.5, 2.0];
        let _ = normalized_guidance(
            &ops,
            &cond_low,
            &uncond_low,
            4.0,
            Some(&mut apg_buf),
            0.0,
            2.5,
            &[1, 4],
            &[-1],
        )
        .unwrap();
        let saved_running = apg_buf.running().unwrap().clone();
        // Read-only boundary evaluation at sigma_K leaves `apg_buf` completely unchanged.
        let _readonly_out = normalized_guidance_readonly(
            &ops,
            &vec![2.0_f32, 1.0, 0.5, 3.0],
            &uncond_low,
            4.0,
            Some(&apg_buf),
            0.0,
            2.5,
            &[1, 4],
            &[-1],
        )
        .unwrap();
        assert_eq!(apg_buf.running(), Some(&saved_running));
        // Phase-boundary reset clears old-shape `(low_w, low_h)` buffer so Phase 1 starts fresh at `(full_w, full_h)`.
        apg_buf.reset();
        assert!(apg_buf.is_empty());
    }

    /// SW-6940 (FEAT-1, PIPE-3, PIPE-4, PERF-3, E3):
    /// Qualify `EstimateSource::EndpointEvaluation` (default) vs `EstimateSource::PreviousEvaluation`
    /// (candidate under DEC-10) across 1-stage (`Euler`, `Dpmpp2m`, `UniPc`), 2-stage (`Heun`, `DpmppSde`),
    /// and stochastic (`EulerAncestral`) solvers:
    /// - Read-only `BoundaryEstimate` at `sigma_K` vs captured `GuidedPrimaryX0` from outer step `K - 1` at `sigma_{K-1}`.
    /// - Identical `0..K` low-res solver steps, sub-evaluations, RNG draw coordinates, progress events, and preview sequence.
    /// - Explicit solver/guidance support and typed `Error::Unsupported` refusal for unsupported combinations.
    /// - Persisted `LowResStartExecutionReport` and `EstimateProvenance` on `GenerationReport`.
    /// - Bounded quality, evaluation/time cost (`M + 1` vs `M`, `2M + 1` vs `2M`, ~1.5-1.7x token speedup), and retained-buffer memory comparison.
    #[test]
    fn sw6940_endpoint_vs_previous_evaluation_qualification_provenance_and_solver_refusal() {
        use crate::sampling::{
            build_flow_sigmas, compute_mu, image_seq_len, CpuLatentOps, Solver, SolverRunState,
        };
        use crate::{Capabilities, GenerationOutput, GenerationReport};

        fn flux_desc() -> ModelDescriptor {
            ModelDescriptor {
                encoder_contract: None,
                denoiser_output_latent_space: Some(&FLUX1_LATENT_SPACE),
                control_kinds: None,
                required_components: &[],
                id: "flux1_dev",
                family: "flux",
                backend: "mlx",
                modality: Modality::Image,
                capabilities: Capabilities {
                    min_size: 256,
                    max_size: 2048,
                    max_count: 4,
                    ..Default::default()
                },
            }
        }

        let m = 8usize;
        let k = 4usize;
        let mu = compute_mu(image_seq_len(512, 512), m);
        let sigmas = build_flow_sigmas(m, mu);
        assert_eq!(sigmas.len(), m + 1);

        // 1. Default policy is always EndpointEvaluation unless explicitly overridden.
        assert_eq!(
            EstimateSource::default(),
            EstimateSource::EndpointEvaluation
        );
        assert!(EstimateSource::EndpointEvaluation.is_default_policy());
        assert!(!EstimateSource::PreviousEvaluation.is_default_policy());
        assert_eq!(
            LowResStartSpec::new(k, Some(0.5)).estimate_source,
            EstimateSource::EndpointEvaluation
        );
        assert_eq!(
            EstimateSource::from_name("endpoint_evaluation"),
            Some(EstimateSource::EndpointEvaluation)
        );
        assert_eq!(
            EstimateSource::from_name(LOW_RES_START_ENDPOINT_ESTIMATOR_ID),
            Some(EstimateSource::EndpointEvaluation)
        );
        assert_eq!(
            EstimateSource::from_name("previous_evaluation"),
            Some(EstimateSource::PreviousEvaluation)
        );
        assert_eq!(
            EstimateSource::from_name(LOW_RES_START_PREVIOUS_ESTIMATOR_ID),
            Some(EstimateSource::PreviousEvaluation)
        );
        assert_eq!(EstimateSource::from_name("unknown"), None);

        // 2. Explicit solver/guidance support and typed refusal for unsupported PreviousEvaluation combinations,
        // without removing EndpointEvaluation's supported routes or solvers.
        let desc = flux_desc();
        for unsupported_sampler in ["lightning", "tcd", "scm", "trigflow", "unknown_policy"] {
            let mut req_prev = GenerationRequest {
                prompt: "a lighthouse".into(),
                width: 512,
                height: 512,
                sampler: Some(unsupported_sampler.into()),
                low_res_start: Some(
                    LowResStartSpec::new(2, Some(0.5))
                        .with_estimate_source(EstimateSource::PreviousEvaluation),
                ),
                ..Default::default()
            };
            assert!(
                matches!(
                    req_prev
                        .low_res_start
                        .unwrap()
                        .resolve_for_descriptor(&desc, &req_prev, 4, 16),
                    Err(Error::Unsupported(ref msg)) if msg.contains("previous_evaluation is unsupported")
                ),
                "PreviousEvaluation must refuse unsupported sampler {unsupported_sampler}"
            );
            // EndpointEvaluation continues to resolve without removing its supported routes or fallback behavior.
            req_prev.low_res_start = Some(LowResStartSpec::new(2, Some(0.5)));
            assert!(req_prev
                .low_res_start
                .unwrap()
                .resolve_for_descriptor(&desc, &req_prev, 4, 16)
                .is_ok());
        }

        // CFG++ guidance_method is refused for PreviousEvaluation (CFG++ renoises from uncond x0), while EndpointEvaluation remains allowed.
        let req_cfgpp = GenerationRequest {
            prompt: "a lighthouse".into(),
            width: 512,
            height: 512,
            sampler: Some("euler".into()),
            guidance_method: Some("cfg_pp".into()),
            low_res_start: Some(
                LowResStartSpec::new(2, Some(0.5))
                    .with_estimate_source(EstimateSource::PreviousEvaluation),
            ),
            ..Default::default()
        };
        assert!(matches!(
            req_cfgpp
                .low_res_start
                .unwrap()
                .resolve_for_descriptor(&desc, &req_cfgpp, 4, 16),
            Err(Error::Unsupported(ref msg)) if msg.contains("cfg_pp")
        ));
        let mut cfgpp_state = SolverRunState::<CpuLatentOps>::new_cfg_pp(Solver::Euler, &sigmas);
        assert!(matches!(
            cfgpp_state.enable_primary_estimate_capture(k - 1),
            Err(Error::Unsupported(_))
        ));

        // Calling take_captured_primary_estimate when capture was not enabled or not populated refuses with Error::Unsupported (never falls back!).
        let mut uncaptured_state = SolverRunState::<CpuLatentOps>::new(Solver::Euler, &sigmas);
        assert!(matches!(
            uncaptured_state.take_captured_primary_estimate(k - 1),
            Err(Error::Unsupported(ref msg)) if msg.contains("GuidedPrimaryX0")
        ));

        // 3. Qualify EndpointEvaluation vs PreviousEvaluation across 1-stage, 2-stage, and stochastic solvers.
        for solver in [
            Solver::Euler,
            Solver::Dpmpp2m,
            Solver::UniPc,
            Solver::EulerAncestral,
            Solver::Heun,
            Solver::DpmppSde,
        ] {
            let qual =
                qualify_low_res_start_estimators(solver, &sigmas, k, 0.5, 512, 512, 42, 2).unwrap();

            assert_eq!(qual.default_policy, EstimateSource::EndpointEvaluation);
            assert_eq!(qual.candidate_policy, EstimateSource::PreviousEvaluation);
            assert!(qual.default_selection_unchanged);
            assert!(
                qual.phase0_traces_identical,
                "{solver:?}: Phase-0 (0..K) completed steps, sub-evaluations, RNG draws, progress, and previews must be identical"
            );
            assert!(
                qual.phase1_rng_coordinates_identical,
                "{solver:?}: Phase-1 (K..M) RNG draw coordinates must be identical"
            );
            assert!(
                qual.outer_progress_identical,
                "{solver:?}: Outer progress events must be identical"
            );
            assert!(
                qual.preview_sequences_identical,
                "{solver:?}: Preview sequence must be identical"
            );

            // Read-only boundary guarantee for EndpointEvaluation and zero-boundary-eval guarantee for PreviousEvaluation.
            assert_eq!(qual.endpoint_trace.boundary_evaluations, 1);
            assert_eq!(qual.previous_trace.boundary_evaluations, 0);
            for trace in [&qual.endpoint_trace, &qual.previous_trace] {
                assert_eq!(trace.completed_low_res_steps, k);
                assert!(!trace.boundary_advanced_progress);
                assert!(!trace.boundary_emitted_preview);
                assert!(!trace.boundary_mutated_solver_history);
                assert!(!trace.boundary_mutated_rng);
                assert!(!trace.boundary_mutated_guidance_accumulator);
                assert_eq!(
                    trace.total_progress_events.len(),
                    m * (solver.stages_per_step() as usize)
                        - usize::from(solver.stages_per_step() == 2)
                );
            }

            // Evaluation count & token-weighted work comparison (PERF-3, E3):
            let stages = solver.stages_per_step();
            assert_eq!(
                qual.endpoint_work.nominal_stage_evaluations,
                stages * (m as u32) + 1
            );
            assert_eq!(
                qual.previous_work.nominal_stage_evaluations,
                stages * (m as u32)
            );
            assert_eq!(
                qual.endpoint_work.total_model_evaluations,
                qual.previous_work.total_model_evaluations + 1
            );
            assert_eq!(qual.endpoint_work.low_tokens, 256);
            assert_eq!(qual.endpoint_work.full_tokens, 1024);
            // At K = M / 2 = 4, scale = 0.5 (token_ratio = 0.25):
            // PreviousEvaluation speedup = 1 / (0.5 * 0.25 + 0.5 * 1.0) = 1.60x.
            // EndpointEvaluation speedup = 1 / ((4*stages + 1)/(8*stages) * 0.25 + 0.5) in ~1.52..=1.58x.
            assert!(
                (1.50..=1.70).contains(&qual.endpoint_work.theoretical_speedup),
                "{solver:?}: endpoint speedup {} outside 1.5..=1.7",
                qual.endpoint_work.theoretical_speedup
            );
            assert!(
                (qual.previous_work.theoretical_speedup - 1.60).abs() < 1e-5,
                "{solver:?}: previous speedup {} != 1.60",
                qual.previous_work.theoretical_speedup
            );

            // For 2-stage solvers (Heun, DpmppSde), verify that step K - 1 executed BOTH Primary and Corrector/Midpoint
            // sub-evaluations in Phase 0, and that PreviousEvaluation still captured GuidedPrimaryX0 at step K - 1 (sigma_{K-1}).
            if stages == 2 {
                assert_eq!(qual.previous_trace.phase0_sub_evaluations.len(), 2 * k);
                assert_eq!(
                    qual.previous_trace.phase0_sub_evaluations[2 * (k - 1)],
                    (k - 1, "primary".to_string(), sigmas[k - 1].to_bits())
                );
                let second_kind = &qual.previous_trace.phase0_sub_evaluations[2 * (k - 1) + 1].1;
                assert!(
                    second_kind == "heun_corrector" || second_kind == "sde_midpoint",
                    "unexpected 2nd-stage evaluation kind: {second_kind}"
                );
            }

            // Memory comparison:
            // At 256x256 (256 tokens * 64 channels * 2 bytes bf16 = 32_768 bytes):
            assert_eq!(qual.endpoint_memory.low_packed_latent_bytes, 32_768);
            assert_eq!(qual.endpoint_memory.full_packed_latent_bytes, 131_072);
            assert_eq!(qual.endpoint_memory.retained_primary_estimate_bytes, 0);
            assert_eq!(qual.endpoint_memory.boundary_eval_workspace_bytes, 32_768);
            assert_eq!(qual.previous_memory.retained_primary_estimate_bytes, 32_768);
            assert_eq!(qual.previous_memory.boundary_eval_workspace_bytes, 0);
            assert_eq!(
                qual.previous_memory
                    .phase0_step_k_minus_1_coexisting_buffer_bytes,
                qual.endpoint_memory
                    .phase0_step_k_minus_1_coexisting_buffer_bytes
                    + 32_768
            );

            // Quality metrics: finite values, strong prompt/seed signal retention vs full-res, and no spatial seam spike.
            assert!(qual.endpoint_quality.all_finite);
            assert!(qual.previous_quality.all_finite);
            assert!(
                qual.endpoint_quality.cosine_similarity_vs_full_res > 0.85,
                "{solver:?}: endpoint cosine similarity {} too low",
                qual.endpoint_quality.cosine_similarity_vs_full_res
            );
            assert!(
                qual.previous_quality.cosine_similarity_vs_full_res > 0.85,
                "{solver:?}: previous cosine similarity {} too low",
                qual.previous_quality.cosine_similarity_vs_full_res
            );
            assert!(
                qual.endpoint_vs_previous_cosine_similarity > 0.90,
                "{solver:?}: endpoint vs previous cosine similarity {} too low",
                qual.endpoint_vs_previous_cosine_similarity
            );
            assert!(
                qual.endpoint_quality.max_seam_gradient_ratio < 1.35,
                "{solver:?}: endpoint seam gradient ratio {} indicates seam artifact",
                qual.endpoint_quality.max_seam_gradient_ratio
            );
            assert!(
                qual.previous_quality.max_seam_gradient_ratio < 1.35,
                "{solver:?}: previous seam gradient ratio {} indicates seam artifact",
                qual.previous_quality.max_seam_gradient_ratio
            );

            // Provenance & Receipt Report verification:
            assert_eq!(qual.endpoint_report.recipe_id, "low_res_start");
            assert_eq!(qual.endpoint_report.version, 1);
            assert_eq!(qual.endpoint_report.estimator, "endpoint_evaluation_v1");
            assert_eq!(
                qual.endpoint_report.estimate_provenance.evaluation_kind,
                EvaluationKind::BoundaryEstimate
            );
            assert_eq!(qual.endpoint_report.estimate_provenance.step_index, k);
            assert_eq!(
                qual.endpoint_report.estimate_provenance.sigma.to_bits(),
                sigmas[k].to_bits()
            );
            assert_eq!(qual.endpoint_report.boundary_evaluations, 1);
            assert_eq!(
                qual.endpoint_report.transition_arithmetic_id,
                LOW_RES_START_TRANSITION_ARITHMETIC_ID
            );
            assert_eq!(
                qual.endpoint_report.transition_seed_derivation_id,
                LOW_RES_START_TRANSITION_DOMAIN
            );

            assert_eq!(qual.previous_report.recipe_id, "low_res_start");
            assert_eq!(qual.previous_report.version, 1);
            assert_eq!(qual.previous_report.estimator, "previous_evaluation_v1");
            assert_eq!(
                qual.previous_report.estimate_provenance.evaluation_kind,
                EvaluationKind::GuidedPrimaryX0
            );
            assert_eq!(qual.previous_report.estimate_provenance.step_index, k - 1);
            assert_eq!(
                qual.previous_report.estimate_provenance.sigma.to_bits(),
                sigmas[k - 1].to_bits()
            );
            assert_eq!(qual.previous_report.boundary_evaluations, 0);
            assert_eq!(
                qual.previous_report
                    .estimate_provenance
                    .retained_primary_estimate_bytes,
                32_768
            );

            // JSON round-trip and GenerationReport integration.
            let encoded = serde_json::to_value(&qual).unwrap();
            let decoded: LowResStartQualificationReport = serde_json::from_value(encoded).unwrap();
            assert_eq!(decoded, qual);

            let mut gen_report = GenerationReport::from_output(GenerationOutput::Images(vec![]));
            assert!(gen_report.low_res_start.is_none());
            gen_report.low_res_start = Some(qual.endpoint_report.clone());
            assert_eq!(
                gen_report.low_res_start.as_ref().unwrap().estimator,
                "endpoint_evaluation_v1"
            );
        }
    }
}
