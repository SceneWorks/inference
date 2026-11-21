//! Metadata-only physical video layout shared by cache admission and the loaded provider.

use super::{video::VideoDenoisePhase, RenderPhaseIdentity};
use crate::{
    Conditioning, Error, GenerationMemory, GenerationRequest, LoadSpec, MemoryBudgetDomain,
    MemoryGeometry, MemoryPhase, MemoryRequestScope, MemoryRunContext, MemoryRunOutcome,
    ModelDescriptor, RenderLoadBinding, Result,
};
use sha2::{Digest, Sha256};
use std::ops::Range;

#[derive(Clone, Copy, Debug)]
pub struct VideoPlanRegistration {
    pub provider_id: &'static str,
    pub compile: fn(&LoadSpec, &GenerationRequest) -> Result<Option<VideoPlanLayout>>,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct VideoReferenceLayout {
    pub role: &'static str,
    pub width: u32,
    pub height: u32,
    pub frames: u32,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct VideoPhaseLayout {
    pub identity: RenderPhaseIdentity,
    pub latent_frames: u32,
    pub latent_bytes: u64,
    /// Distinct prepared tensor storage kept alongside the latent for this whole denoise.
    pub retained_bytes: u64,
    pub decoded_frames: u32,
    pub decode_chunk_frames: u32,
    pub source: Range<usize>,
    pub history_frames: usize,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct VideoPlanLayout {
    pub output: MemoryGeometry,
    pub admission: MemoryGeometry,
    pub phases: Vec<VideoPhaseLayout>,
    pub references: Vec<VideoReferenceLayout>,
    pub domain: MemoryBudgetDomain,
}

impl VideoPlanLayout {
    pub fn single<S, P>(
        output: MemoryGeometry,
        phase: &VideoDenoisePhase<S, P>,
        latent_bytes: u64,
        decoded_frames: u32,
        decode_chunk_frames: u32,
        references: Vec<VideoReferenceLayout>,
        domain: MemoryBudgetDomain,
    ) -> Self {
        Self {
            output,
            admission: phase.identity.geometry,
            phases: vec![VideoPhaseLayout {
                identity: phase.identity,
                latent_frames: phase.latent_frames,
                latent_bytes,
                retained_bytes: 0,
                decoded_frames,
                decode_chunk_frames,
                source: 0..output.frames as usize,
                history_frames: 0,
            }],
            references,
            domain,
        }
    }

    fn validate(&self) -> Result<()> {
        let valid =
            |g: MemoryGeometry| g.width != 0 && g.height != 0 && g.frames != 0 && g.batch != 0;
        if !valid(self.output)
            || !valid(self.admission)
            || self.phases.is_empty()
            || self.phases.iter().any(|p| {
                !valid(p.identity.geometry)
                    || p.latent_frames == 0
                    || p.latent_bytes == 0
                    || p.decoded_frames == 0
                    || p.decode_chunk_frames == 0
                    || p.decode_chunk_frames > p.decoded_frames
                    || p.source.is_empty()
            })
            || self
                .references
                .iter()
                .any(|r| r.width == 0 || r.height == 0 || r.frames == 0)
        {
            return Err(Error::Unsupported("invalid physical video layout".into()));
        }
        Ok(())
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
struct VideoInputs {
    width: u32,
    height: u32,
    frames: Option<u32>,
    trim: Option<u32>,
    count: u32,
    fps: Option<u32>,
    mode: Option<String>,
    sampler: Option<String>,
    steps: Option<u32>,
    seed: Option<u64>,
    guidance: Option<u32>,
    shift: Option<u32>,
    decode_chunk: Option<u32>,
    carrier_bytes: u64,
    content: [u8; 32],
    strength: Option<u32>,
    scheduler: Option<String>,
    conditioning_fps: Option<u32>,
    motion_bucket: Option<u32>,
    noise_aug: Option<u32>,
}

impl VideoInputs {
    fn from_request(req: &GenerationRequest) -> Result<Self> {
        // The request keeps these original, owned RGB buffers live through decode. Prepared
        // (resized or padded) reference roles are separately resolved by the family compiler.
        let mut carrier_bytes = 0u64;
        let mut content = Sha256::new();
        for prompt in [Some(req.prompt.as_str()), req.negative_prompt.as_deref()] {
            content.update([u8::from(prompt.is_some())]);
            let prompt = prompt.unwrap_or_default().as_bytes();
            content.update((prompt.len() as u64).to_le_bytes());
            content.update(prompt);
        }
        let mut count_image = |content: &mut Sha256, im: &crate::Image| -> Result<()> {
            content.update(im.width.to_le_bytes());
            content.update(im.height.to_le_bytes());
            content.update(super::image_digest(im)?);
            carrier_bytes = carrier_bytes
                .checked_add(im.pixels.len() as u64)
                .ok_or_else(|| Error::Unsupported("video input size overflow".into()))?;
            Ok(())
        };
        for c in &req.conditioning {
            match c {
                Conditioning::Reference { image, strength } => {
                    content.update([0, u8::from(strength.is_some())]);
                    content.update(strength.unwrap_or_default().to_bits().to_le_bytes());
                    count_image(&mut content, image)?;
                }
                Conditioning::Mask { image } => {
                    content.update([1]);
                    count_image(&mut content, image)?;
                }
                Conditioning::ControlClip {
                    frames,
                    mask,
                    mode,
                    start_frame,
                    masking_strength,
                } => {
                    content.update([2, *mode as u8]);
                    content.update(start_frame.to_le_bytes());
                    content.update(masking_strength.to_bits().to_le_bytes());
                    for images in [frames, mask] {
                        content.update((images.len() as u64).to_le_bytes());
                        for image in images {
                            count_image(&mut content, image)?;
                        }
                    }
                }
                _ => {
                    return Err(Error::Unsupported(
                        "unbound video conditioning carrier".into(),
                    ))
                }
            }
        }
        Ok(Self {
            width: req.width,
            height: req.height,
            frames: req.frames,
            trim: req.trim_first_frames,
            count: req.count,
            fps: req.fps,
            mode: req.video_mode.clone(),
            sampler: req.sampler.clone(),
            steps: req.steps,
            seed: req.seed,
            guidance: req.guidance.map(f32::to_bits),
            shift: req.scheduler_shift.map(f32::to_bits),
            decode_chunk: req.decode_chunk_size,
            carrier_bytes,
            content: content.finalize().into(),
            strength: req.strength.map(f32::to_bits),
            scheduler: req.scheduler.clone(),
            conditioning_fps: req.conditioning_fps,
            motion_bucket: req.motion_bucket_id.map(f32::to_bits),
            noise_aug: req.noise_aug_strength.map(f32::to_bits),
        })
    }
}

#[derive(Clone, Debug)]
pub struct VideoRenderPlan {
    registration: VideoPlanRegistration,
    layout: VideoPlanLayout,
    load: RenderLoadBinding,
    inputs: VideoInputs,
    bound: bool,
    context: Option<MemoryRunContext>,
    memory: Option<GenerationMemory>,
}

impl VideoRenderPlan {
    pub(crate) fn compile(
        registration: &VideoPlanRegistration,
        spec: &LoadSpec,
        req: &GenerationRequest,
    ) -> Result<Option<Self>> {
        let Some(layout) =
            spec.read_prepared_files_unchanged(|| (registration.compile)(spec, req))?
        else {
            return Ok(None);
        };
        layout.validate()?;
        Ok(Some(Self {
            registration: *registration,
            layout,
            load: RenderLoadBinding::from_video_spec(spec),
            inputs: VideoInputs::from_request(req)?,
            bound: false,
            context: None,
            memory: req.memory,
        }))
    }
    pub fn layout(&self) -> &VideoPlanLayout {
        &self.layout
    }
    pub fn geometry(&self) -> MemoryGeometry {
        self.layout.admission
    }

    /// Check the public request boundary before constructing a source-bound evidence receipt.
    pub fn check_inputs(&self, req: &GenerationRequest) -> Result<()> {
        if self.inputs != VideoInputs::from_request(req)? {
            return Err(Error::Unsupported(
                "video inputs changed after admission".into(),
            ));
        }
        Ok(())
    }

    pub fn check_provider(
        &self,
        descriptor: &ModelDescriptor,
        load: Option<&RenderLoadBinding>,
    ) -> Result<()> {
        let matches = load.is_some_and(|actual| {
            if self.bound {
                return actual == &self.load;
            }
            let mut expected = self.load.clone();
            expected.sequential_offload = actual.sequential_offload;
            actual == &expected
        });
        if descriptor.id != self.registration.provider_id || !matches {
            return Err(Error::Unsupported(
                "video plan and loaded source or placement differ".into(),
            ));
        }
        Ok(())
    }

    /// Bind a cache-selected placement and the existing ladder's configured memory controls.
    /// The ladder retains sole ownership of the budget, tier, evidence and resident credit.
    pub fn bind_admission(
        &self,
        spec: &LoadSpec,
        context: Option<&MemoryRunContext>,
        req: &GenerationRequest,
    ) -> Result<Self> {
        let mut load = self.load.clone();
        load.sequential_offload = spec.offload_policy == crate::OffloadPolicy::Sequential;
        if self.bound
            || (context.is_none() && req.memory != self.memory)
            || load != RenderLoadBinding::from_video_spec(spec)
            || self.inputs != VideoInputs::from_request(req)?
            || context.is_some_and(|c| c.geometry != self.geometry())
        {
            return Err(Error::Unsupported(
                "video admission crossed its load, input or geometry binding".into(),
            ));
        }
        let Some(layout) =
            spec.read_prepared_files_unchanged(|| (self.registration.compile)(spec, req))?
        else {
            return Err(Error::Unsupported("admitted video route changed".into()));
        };
        layout.validate()?;
        // Only an equivalent selected decoder policy may refine the physical chunk bound.
        let mut expected = self.layout.clone();
        for (p, selected) in expected.phases.iter_mut().zip(&layout.phases) {
            if selected.decode_chunk_frames != p.decode_chunk_frames {
                if selected.decode_chunk_frames > p.decode_chunk_frames
                    || !context.is_some_and(|c| c.selection.strategy.is_optimized())
                {
                    return Err(Error::Unsupported(
                        "decoder exceeds its admitted policy".into(),
                    ));
                }
                p.decode_chunk_frames = selected.decode_chunk_frames;
            }
        }
        if expected != layout {
            return Err(Error::Unsupported(
                "admitted video physical layout changed".into(),
            ));
        }
        Ok(Self {
            load,
            layout,
            bound: true,
            context: context.cloned(),
            memory: req.memory,
            ..self.clone()
        })
    }

    pub fn minimum_denoise_live_bytes(&self, domain: MemoryBudgetDomain) -> Result<u64> {
        let mut peak = 0;
        for phase in &self.layout.phases {
            peak = peak.max(crate::sum_deduplicated_live_bytes(
                domain,
                &[
                    self.source_allocation(),
                    crate::PricedAllocation::new(
                        "video_latent",
                        self.layout.domain,
                        phase.latent_bytes,
                    ),
                    crate::PricedAllocation::new(
                        "denoise_conditioning",
                        self.layout.domain,
                        phase.retained_bytes,
                    ),
                ],
            )?);
        }
        Ok(peak)
    }

    fn host_domain(&self) -> MemoryBudgetDomain {
        if self.layout.domain == MemoryBudgetDomain::UnifiedMemory {
            MemoryBudgetDomain::UnifiedMemory
        } else {
            MemoryBudgetDomain::HostRam
        }
    }

    fn source_allocation(&self) -> crate::PricedAllocation {
        crate::PricedAllocation::new("source_rgb", self.host_domain(), self.inputs.carrier_bytes)
    }

    /// Minimum live buffers on a single physical budget. Source pixels survive until the
    /// buffered result is complete. Existing calibrated costs and resident credits remain
    /// with the application's priced operation.
    pub fn minimum_live_bytes(&self, domain: MemoryBudgetDomain) -> Result<u64> {
        let output = self.layout.output;
        let bytes = [output.width, output.height, output.frames, output.batch, 3]
            .into_iter()
            .try_fold(1u64, |n, d| n.checked_mul(u64::from(d)))
            .ok_or_else(|| Error::Unsupported("video output size overflow".into()))?;
        let decode = crate::sum_deduplicated_live_bytes(
            domain,
            &[
                self.source_allocation(),
                crate::PricedAllocation::new("output_rgb", self.host_domain(), bytes),
            ],
        )?;
        Ok(decode.max(self.minimum_denoise_live_bytes(domain)?))
    }

    pub(crate) fn run<T>(
        &self,
        req: &GenerationRequest,
        contract: Option<&crate::MemoryProviderContract>,
        scope: Option<&mut dyn MemoryRequestScope>,
        body: impl FnOnce(&mut dyn MemoryRequestScope) -> Result<T>,
    ) -> Result<T> {
        if !self.bound
            || req.memory != self.memory
            || self.inputs != VideoInputs::from_request(req)?
            || self.context.as_ref().is_some_and(|c| {
                scope.is_none()
                    || !contract.is_some_and(|k| k.generation_memory(&c.selection) == req.memory)
            })
        {
            return Err(Error::Unsupported(
                "video execution differs from its admitted request".into(),
            ));
        }
        let mut trace = VideoScope {
            plan: &self.layout,
            scope,
            index: 0,
            active: false,
        };
        let output = body(&mut trace)?;
        if trace.active || trace.index != self.layout.phases.len() {
            return Err(Error::Unsupported(
                "provider did not complete its admitted video phases".into(),
            ));
        }
        Ok(output)
    }
}

struct VideoScope<'p, 's> {
    plan: &'p VideoPlanLayout,
    scope: Option<&'s mut dyn MemoryRequestScope>,
    index: usize,
    active: bool,
}

impl MemoryRequestScope for VideoScope<'_, '_> {
    fn enter_video_phase(
        &mut self,
        identity: &RenderPhaseIdentity,
        latent_frames: u32,
    ) -> Result<()> {
        if self.active
            || !self
                .plan
                .phases
                .get(self.index)
                .is_some_and(|p| p.identity == *identity && p.latent_frames == latent_frames)
        {
            return Err(Error::Unsupported(
                "provider requested an unbound video phase".into(),
            ));
        }
        if let Some(s) = self.scope.as_deref_mut() {
            s.enter_video_phase(identity, latent_frames)?;
        }
        self.active = true;
        Ok(())
    }
    fn enter_phase(&mut self, phase: MemoryPhase) -> Result<()> {
        if phase == MemoryPhase::Denoise {
            return Err(Error::Unsupported(
                "video denoise must declare its physical geometry".into(),
            ));
        }
        if let Some(s) = self.scope.as_deref_mut() {
            s.enter_phase(phase)?;
        }
        Ok(())
    }
    fn leave_phase(&mut self, phase: MemoryPhase) -> Result<()> {
        if phase == MemoryPhase::Denoise && !self.active {
            return Err(Error::Unsupported("video denoise was not active".into()));
        }
        if let Some(s) = self.scope.as_deref_mut() {
            s.leave_phase(phase)?;
        }
        if phase == MemoryPhase::Denoise {
            self.index += 1;
            self.active = false;
        }
        Ok(())
    }
    fn configure_decode(
        &mut self,
        edge: u32,
        overlap: u32,
        geometry: MemoryGeometry,
    ) -> Result<()> {
        if let Some(s) = self.scope.as_deref_mut() {
            s.configure_decode(edge, overlap, geometry)?;
        }
        Ok(())
    }
    fn configure_attention(&mut self, chunk: u32) -> Result<()> {
        if let Some(s) = self.scope.as_deref_mut() {
            s.configure_attention(chunk)?;
        }
        Ok(())
    }
    fn materialize_transformer_window(&mut self, first: u32, count: u32) -> Result<()> {
        if let Some(s) = self.scope.as_deref_mut() {
            s.materialize_transformer_window(first, count)?;
        }
        Ok(())
    }
    fn configure_request(&mut self, _: &mut GenerationRequest) -> Result<()> {
        Err(Error::Unsupported(
            "the admission owner configures the video request".into(),
        ))
    }
    fn finish(&mut self, _: MemoryRunOutcome) -> Result<()> {
        Err(Error::Unsupported(
            "the admission owner finishes the video request".into(),
        ))
    }
}
