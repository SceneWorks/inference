//! Weight-free execution contract and owner-local lifecycle for fixed image passes.

pub mod conditional;

use crate::{
    default_seed, CancelFlag, Error, GenerationRequest, LoadSpec, MemoryPhase, MemoryRequestScope,
    ModelDescriptor, Progress, Result,
};
use serde::Serialize;
use sha2::{Digest, Sha256};
use std::cell::{Cell, RefCell};
use std::ops::Range;

pub struct RenderPlanRegistration {
    pub provider_id: &'static str,
    pub compile: fn(&LoadSpec, &GenerationRequest) -> Result<Option<RenderPlan>>,
}

/// A serialized plan exposes the actual execution order, including once-per-request encoding
/// and the subset repeated for each sequential image. This noise-start pass has no reference stage.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize)]
enum Stage {
    PromptPrep,
    Encode,
    Schedule,
    Init,
    Enter,
    Denoise,
    Exit,
    PostLatent,
    Decode,
    Output,
}

#[derive(Clone, Debug, PartialEq, Serialize)]
struct PackedImageStream {
    width: u32,
    height: u32,
    patch_stride: u32,
    tokens: u32,
    token_width: u32,
}

#[derive(Clone, Debug, PartialEq, Serialize)]
struct ImagePhase {
    first_step: usize,
    end_step: usize,
    stream: PackedImageStream,
    sampler: String,
    guidance: f32,
}

#[derive(Clone, Debug, PartialEq, Serialize)]
struct ImagePass {
    stages: [Stage; 10],
    sigmas: Vec<f32>,
    phase: ImagePhase,
    #[serde(skip_serializing_if = "Vec::is_empty")]
    inert_splits: Vec<usize>,
    capture_sigma: f32,
}

#[derive(Clone, Debug, PartialEq, Serialize)]
struct ImageInputs {
    prompt_sha256: [u8; 32],
    negative_prompt: Option<String>,
    width: u32,
    height: u32,
    count: u32,
    requested_seed: Option<u64>,
    frames: u32,
    references: u32,
    steps: Option<u32>,
    sampler: Option<String>,
    scheduler: Option<String>,
    scheduler_shift: Option<f32>,
    guidance: Option<f32>,
    use_pid: bool,
    pid_capture_sigma: Option<f32>,
}

impl From<&GenerationRequest> for ImageInputs {
    fn from(req: &GenerationRequest) -> Self {
        Self {
            prompt_sha256: Sha256::digest(req.prompt.as_bytes()).into(),
            negative_prompt: req.negative_prompt.clone(),
            width: req.width,
            height: req.height,
            count: req.count,
            requested_seed: req.seed,
            frames: req.frames.unwrap_or(1),
            references: req.memory_reference_count(),
            steps: req.steps,
            sampler: req.sampler.clone(),
            scheduler: req.scheduler.clone(),
            scheduler_shift: req.scheduler_shift,
            guidance: req.guidance,
            use_pid: req.use_pid,
            pid_capture_sigma: req.pid_capture_sigma,
        }
    }
}

/// Dependencies of base FLUX preparation at load. The cache/loader independently enforces
/// its prepared source pins and admitted offload policy. Neither pins nor tensors are cloned.
#[derive(Clone, Debug, PartialEq, Serialize)]
pub struct RenderLoadBinding {
    weights: crate::WeightsSource,
    resolved_route: Option<String>,
    quant: Option<crate::Quant>,
    precision: crate::Precision,
    adapters: Vec<crate::AdapterSpec>,
    ip_adapter: Option<crate::WeightsSource>,
    pid: Option<crate::PidWeights>,
}

impl RenderLoadBinding {
    pub fn from_spec(spec: &LoadSpec) -> Self {
        Self {
            weights: spec.weights.clone(),
            resolved_route: spec.resolved_route.clone(),
            quant: spec.quantize,
            precision: spec.precision,
            adapters: spec.adapters.clone(),
            ip_adapter: spec.ip_adapter.clone(),
            pid: spec.pid.clone(),
        }
    }
}

/// Count and seed order belong to the plan. Each image gets its own preview sequence
/// and is decoded before the next image's initial noise is drawn.
#[derive(Clone, Debug, PartialEq, Serialize)]
enum CountPlan {
    SequentialWrappingSeeds { count: u32, base_seed: u64 },
}

impl CountPlan {
    fn seeds(&self) -> impl Iterator<Item = u64> {
        let Self::SequentialWrappingSeeds { count, base_seed } = *self;
        (0..count).map(move |index| base_seed.wrapping_add(u64::from(index)))
    }

    fn base_seed(&self) -> u64 {
        let Self::SequentialWrappingSeeds { base_seed, .. } = *self;
        base_seed
    }
}
/// Family-resolved packing, solver and schedule for one image pass. The compiler
/// validates these values before the provider can materialize its components.
pub struct PackedImagePass<'a> {
    pub pixel_stride: u32,
    pub token_width: u32,
    pub sampler: &'a str,
    pub guidance: f32,
    pub sigmas: Vec<f32>,
    pub executed_steps: usize,
    pub capture_sigma: f32,
}

/// Complete resolved base image pass. Tensor operations, callbacks and mutable trajectory
/// state stay with the executor. Only the plan can travel across the caller/owner boundary.
#[derive(Clone, Debug, PartialEq, Serialize)]
pub struct RenderPlan {
    backend: String,
    provider: String,
    inputs: ImageInputs,
    preparation: Option<RenderLoadBinding>,
    count: CountPlan,
    pass: ImagePass,
}

impl RenderPlan {
    pub fn base_image(
        descriptor: &ModelDescriptor,
        req: &GenerationRequest,
        pass: PackedImagePass<'_>,
    ) -> Result<Self> {
        if !req.conditioning.is_empty() {
            return Err(Error::Unsupported(
                "base image plan needs noise-start inputs".into(),
            ));
        }
        Self::packed_image_trajectory(descriptor, req, pass)
    }

    /// Inner image trajectory for a provider that already owns its conditioning/injector.
    /// The enclosing reference route still owns preparation and admission.
    pub fn packed_image_trajectory(
        descriptor: &ModelDescriptor,
        req: &GenerationRequest,
        pass: PackedImagePass<'_>,
    ) -> Result<Self> {
        let PackedImagePass {
            pixel_stride,
            token_width,
            sampler,
            guidance,
            sigmas,
            executed_steps,
            capture_sigma,
        } = pass;
        if pixel_stride == 0
            || token_width == 0
            || req.count == 0
            || req.width == 0
            || req.height == 0
            || !req.width.is_multiple_of(pixel_stride)
            || !req.height.is_multiple_of(pixel_stride)
            || !guidance.is_finite()
            || !capture_sigma.is_finite()
            || sigmas.len() < 2
            || executed_steps == 0
            || executed_steps >= sigmas.len()
            || sigmas
                .iter()
                .any(|sigma| !sigma.is_finite() || *sigma < 0.0)
            || sigmas.windows(2).any(|nodes| nodes[0] < nodes[1])
            || sigmas.last() != Some(&0.0)
            || sigmas[executed_steps] != capture_sigma
        {
            return Err(Error::Unsupported(format!(
                "{} has an invalid base image pass",
                descriptor.id
            )));
        }
        let tokens = (req.width / pixel_stride)
            .checked_mul(req.height / pixel_stride)
            .ok_or_else(|| Error::Unsupported("packed image token count overflow".into()))?;
        Ok(Self {
            backend: descriptor.backend.to_owned(),
            provider: descriptor.id.to_owned(),
            inputs: ImageInputs::from(req),
            preparation: None,
            count: CountPlan::SequentialWrappingSeeds {
                count: req.count,
                base_seed: req.seed.unwrap_or_else(default_seed),
            },
            pass: ImagePass {
                stages: [
                    Stage::Schedule,
                    Stage::PromptPrep,
                    Stage::Encode,
                    Stage::Init,
                    Stage::Enter,
                    Stage::Denoise,
                    Stage::Exit,
                    Stage::PostLatent,
                    Stage::Decode,
                    Stage::Output,
                ],
                sigmas,
                capture_sigma,
                phase: ImagePhase {
                    first_step: 0,
                    end_step: executed_steps,
                    sampler: sampler.to_owned(),
                    guidance,
                    stream: PackedImageStream {
                        width: req.width,
                        height: req.height,
                        patch_stride: pixel_stride,
                        tokens,
                        token_width,
                    },
                },
                inert_splits: Vec::new(),
            },
        })
    }

    /// Admission may configure memory controls, preview and cancellation after compilation.
    /// Output-affecting inputs of the base route must still describe this exact trajectory.
    pub fn check_request(
        &self,
        descriptor: &ModelDescriptor,
        req: &GenerationRequest,
    ) -> Result<()> {
        if descriptor.id != self.provider
            || descriptor.backend != self.backend
            || !req.conditioning.is_empty()
            || self.inputs != ImageInputs::from(req)
        {
            return Err(Error::Unsupported(
                "compiled render plan does not match this request/provider".into(),
            ));
        }
        Ok(())
    }
    pub(crate) fn bind_load(mut self, spec: &LoadSpec) -> Self {
        self.preparation = Some(RenderLoadBinding::from_spec(spec));
        self
    }

    pub fn check_load(&self, loaded: &RenderLoadBinding) -> Result<()> {
        if self
            .preparation
            .as_ref()
            .is_some_and(|expected| expected != loaded)
        {
            return Err(Error::Unsupported(
                "render plan belongs to a different model or adapter stack".into(),
            ));
        }
        Ok(())
    }

    pub fn base_seed(&self) -> u64 {
        self.count.base_seed()
    }
    pub fn guidance(&self) -> f32 {
        self.pass.phase.guidance
    }
    pub fn sampler(&self) -> &str {
        &self.pass.phase.sampler
    }
    pub fn capture_sigma(&self) -> f32 {
        self.pass.capture_sigma
    }
    pub fn sigmas(&self) -> &[f32] {
        &self.pass.sigmas
    }
    pub fn packed_tokens(&self) -> usize {
        self.pass.phase.stream.tokens as usize
    }

    pub fn denoise_sigmas(&self) -> &[f32] {
        &self.pass.sigmas[self.pass.phase.first_step..=self.pass.phase.end_step]
    }

    /// A proof-only phase boundary with no changes to geometry, model, guidance or schedule.
    /// It is compiled before load; product Feature controls introduce their own validated phases.
    #[doc(hidden)]
    pub fn with_inert_split_at(mut self, step: usize) -> Result<Self> {
        if step <= self.pass.phase.first_step || step >= self.pass.phase.end_step {
            return Err(Error::Unsupported(
                "inert split must lie inside the executed pass steps".into(),
            ));
        }
        if self.pass.inert_splits.binary_search(&step).is_err() {
            self.pass.inert_splits.push(step);
            self.pass.inert_splits.sort_unstable();
        }
        Ok(self)
    }

    /// Half-open absolute steps in the executed pass-local schedule. Keeping a single full
    /// schedule and one request-owned solver state makes inert boundaries numerically invisible.
    pub fn denoise_spans(&self) -> Vec<Range<usize>> {
        let mut first = self.pass.phase.first_step;
        let mut spans = Vec::with_capacity(self.pass.inert_splits.len() + 1);
        for &split in &self.pass.inert_splits {
            spans.push(first..split);
            first = split;
        }
        spans.push(first..self.pass.phase.end_step);
        spans
    }
}

/// Physical geometry of the stage about to operate under the request's already selected
/// memory rung. Sequential images process one trajectory at a time, including when the
/// admission reservation covers the whole count.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct RenderPhaseIdentity {
    pub pass: usize,
    pub phase: usize,
    pub geometry: crate::MemoryGeometry,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum RunState {
    BeforeEncode,
    Encoding,
    Encoded,
    Rendering,
    Finished,
}

/// Owner-local state for an invocation. A provider's existing residency driver temporarily
/// owns its text and heavy bundles. The two callbacks share one request scope without holding
/// a dynamic borrow while running provider code or progress callbacks.
pub struct RenderRun<'p, 's> {
    plan: &'p RenderPlan,
    cancel: &'p CancelFlag,
    state: Cell<RunState>,
    scope: RefCell<Option<&'s mut dyn MemoryRequestScope>>,
    next_denoise_phase: Cell<usize>,
    active_denoise_phase: Cell<Option<usize>>,
}

impl<'p, 's> RenderRun<'p, 's> {
    pub fn new(
        plan: &'p RenderPlan,
        cancel: &'p CancelFlag,
        scope: Option<&'s mut dyn MemoryRequestScope>,
    ) -> Self {
        Self {
            plan,
            cancel,
            scope: RefCell::new(scope),
            state: Cell::new(RunState::BeforeEncode),
            next_denoise_phase: Cell::new(0),
            active_denoise_phase: Cell::new(None),
        }
    }

    fn phase_identity(&self, phase: usize) -> RenderPhaseIdentity {
        let stream = &self.plan.pass.phase.stream;
        RenderPhaseIdentity {
            pass: 0,
            phase,
            geometry: crate::MemoryGeometry {
                width: stream.width,
                height: stream.height,
                batch: 1,
                frames: self.plan.inputs.frames,
                reference_count: self.plan.inputs.references,
            },
        }
    }

    fn phase<T, E: From<Error>>(
        &self,
        phase: MemoryPhase,
        body: impl FnOnce() -> std::result::Result<T, E>,
    ) -> std::result::Result<T, E> {
        if self.cancel.is_cancelled() {
            return Err(Error::Canceled.into());
        }
        if let Some(scope) = self.scope.borrow_mut().as_deref_mut() {
            let identity = self.phase_identity(0);
            scope.enter_render_phase(phase, &identity)?;
        }
        // A failed stage leaves teardown to the scope's owner, preserving the original error.
        let result = body()?;
        if let Some(scope) = self.scope.borrow_mut().as_deref_mut() {
            scope.leave_phase(phase)?;
        }
        Ok(result)
    }

    fn begin(&self, expected: RunState, next: RunState) -> Result<()> {
        if self.state.get() != expected {
            return Err(Error::Unsupported(format!(
                "render stage {next:?} cannot start from {:?}",
                self.state.get()
            )));
        }
        self.state.set(next);
        Ok(())
    }

    pub fn plan(&self) -> &RenderPlan {
        self.plan
    }

    /// Enter the next declared denoise span on the owner thread. A failed body leaves scope
    /// teardown to the owner, as in `phase`; the next image cannot start with an active span.
    pub fn begin_denoise_phase(&self, phase: usize) -> Result<()> {
        if self.state.get() != RunState::Rendering
            || self.active_denoise_phase.get().is_some()
            || phase != self.next_denoise_phase.get()
            || phase > self.plan.pass.inert_splits.len()
        {
            return Err(Error::Unsupported(
                "denoise phase is out of order or outside the compiled plan".into(),
            ));
        }
        if self.cancel.is_cancelled() {
            return Err(Error::Canceled);
        }
        if let Some(scope) = self.scope.borrow_mut().as_deref_mut() {
            scope.enter_render_phase(MemoryPhase::Denoise, &self.phase_identity(phase))?;
        }
        self.active_denoise_phase.set(Some(phase));
        Ok(())
    }

    pub fn end_denoise_phase(&self, phase: usize) -> Result<()> {
        if self.active_denoise_phase.get() != Some(phase) {
            return Err(Error::Unsupported(
                "denoise phase end does not match its active phase".into(),
            ));
        }
        if let Some(scope) = self.scope.borrow_mut().as_deref_mut() {
            scope.leave_phase(MemoryPhase::Denoise)?;
        }
        self.active_denoise_phase.set(None);
        self.next_denoise_phase.set(phase + 1);
        Ok(())
    }

    pub fn encode<T, E: From<Error>>(
        &self,
        body: impl FnOnce() -> std::result::Result<T, E>,
    ) -> std::result::Result<T, E> {
        self.begin(RunState::BeforeEncode, RunState::Encoding)?;
        let result = self.phase(MemoryPhase::Conditioning, body);
        self.state.set(if result.is_ok() {
            RunState::Encoded
        } else {
            RunState::Finished
        });
        result
    }

    pub fn images<I, D, O, E: From<Error>>(
        &self,
        progress: &mut dyn FnMut(Progress),
        mut init: impl FnMut(u64) -> std::result::Result<I, E>,
        mut denoise: impl FnMut(u64, I, &mut dyn FnMut(Progress)) -> std::result::Result<D, E>,
        mut decode: impl FnMut(u64, D, &mut dyn FnMut(Progress)) -> std::result::Result<O, E>,
    ) -> std::result::Result<Vec<O>, E> {
        self.begin(RunState::Encoded, RunState::Rendering)?;
        let images = self
            .plan
            .count
            .seeds()
            .map(|seed| {
                if self.cancel.is_cancelled() {
                    return Err(Error::Canceled.into());
                }
                let initial = init(seed)?;
                let denoised =
                    self.phase(MemoryPhase::Denoise, || denoise(seed, initial, progress))?;
                self.phase(MemoryPhase::Decode, || decode(seed, denoised, progress))
            })
            .collect();
        self.state.set(RunState::Finished);
        images
    }

    /// The phase-aware FLUX path keeps one image trajectory while each declared denoise span
    /// enters/leaves its own memory scope through `begin_denoise_phase`/`end_denoise_phase`.
    /// Preparation and decode remain once per image, and the existing one-span `images` path
    /// remains available to callers without explicit phase driving.
    pub fn images_with_denoise_phases<I, D, O, E: From<Error>>(
        &self,
        progress: &mut dyn FnMut(Progress),
        mut init: impl FnMut(u64) -> std::result::Result<I, E>,
        mut denoise: impl FnMut(u64, I, &mut dyn FnMut(Progress)) -> std::result::Result<D, E>,
        mut decode: impl FnMut(u64, D, &mut dyn FnMut(Progress)) -> std::result::Result<O, E>,
    ) -> std::result::Result<Vec<O>, E> {
        self.begin(RunState::Encoded, RunState::Rendering)?;
        let expected_phases = self.plan.pass.inert_splits.len() + 1;
        let images = self
            .plan
            .count
            .seeds()
            .map(|seed| {
                if self.cancel.is_cancelled() {
                    return Err(Error::Canceled.into());
                }
                self.next_denoise_phase.set(0);
                self.active_denoise_phase.set(None);
                let initial = init(seed)?;
                let denoised = denoise(seed, initial, progress)?;
                if self.active_denoise_phase.get().is_some()
                    || self.next_denoise_phase.get() != expected_phases
                {
                    return Err(Error::Unsupported(
                        "denoise did not finish every compiled phase".into(),
                    )
                    .into());
                }
                self.phase(MemoryPhase::Decode, || decode(seed, denoised, progress))
            })
            .collect();
        self.state.set(RunState::Finished);
        images
    }
}

#[cfg(test)]
mod tests;
