//! Admission-bound iOS execution for every registered model (generators, the Iris restoration
//! transform and the Iris depth task). The wrapped provider is never returned to callers: every
//! entry point rechecks the exact admitted work (geometry, tile, mode, conditioning, prompt,
//! execution — everything but seed and cancellation), live capacity and the sealed source pins
//! BEFORE reaching the provider, and preserves the caller's real cancellation handle.
use crate::ios_admission::{work_binding, IosWork};
use mlx_gen::gen_core::iris::depth::{DepthOutput, DepthRequest, IrisDepthEstimator};
use mlx_gen::gen_core::*;

fn generation_binding(req: &GenerationRequest) -> [u8; 32] {
    crate::binding::generation_binding(req)
}

pub struct BoundGenerator {
    inner: Box<dyn Generator>,
    spec: LoadSpec,
    context: MemoryRunContext,
    request: [u8; 32],
    /// The admission's `demand + margin` (0 for a capture permit).
    footprint: u64,
    capacity: Box<dyn Fn(u64) -> Result<()>>,
    /// The admission's measured envelope, open from before the load for this object's lifetime
    /// (`None` for a capture permit, which runs on the formula guards alone).
    _envelope: Option<mlx_gen::ios_quantum::AdmittedEnvelope>,
}

impl BoundGenerator {
    pub(crate) fn new(
        inner: Box<dyn Generator>,
        spec: LoadSpec,
        context: MemoryRunContext,
        request: [u8; 32],
        footprint: u64,
        capacity: Box<dyn Fn(u64) -> Result<()>>,
        envelope: Option<mlx_gen::ios_quantum::AdmittedEnvelope>,
    ) -> Self {
        Self {
            _envelope: envelope,
            inner,
            spec,
            context,
            request,
            footprint,
            capacity,
        }
    }
    pub fn admitted_context(&self) -> &MemoryRunContext {
        &self.context
    }
    fn check_request(&self, req: &GenerationRequest) -> Result<()> {
        if req.cancel.is_cancelled() {
            return Err(Error::Canceled);
        }
        if generation_binding(req) != self.request {
            return Err(Error::Unsupported(
                "iOS request differs from the admitted request \
                 (geometry/tile/mode/conditioning/prompt/execution)"
                    .into(),
            ));
        }
        Ok(())
    }
    fn check_context(&self, context: &MemoryRunContext) -> Result<()> {
        if context != &self.context {
            return Err(Error::Unsupported(
                "iOS lifecycle context differs from admission".into(),
            ));
        }
        self.spec.validate_prepared_file_pins()
    }
    fn run<T>(
        &self,
        req: &GenerationRequest,
        call: impl FnOnce(&dyn Generator) -> Result<T>,
    ) -> Result<T> {
        self.check_request(req)?;
        (self.capacity)(self.footprint)?;
        self.spec
            .read_prepared_files_unchanged(|| call(self.inner.as_ref()))
    }
}

impl Generator for BoundGenerator {
    fn descriptor(&self) -> &ModelDescriptor {
        self.inner.descriptor()
    }
    fn veda_execution_report(&self) -> Option<VedaExecutionReport> {
        self.inner.veda_execution_report()
    }
    fn adapter_apply_reports(&self) -> Vec<AdapterApplyReport> {
        self.inner.adapter_apply_reports()
    }
    fn checkpoint_weight_facts(&self) -> Option<CheckpointWeightFacts> {
        self.inner.checkpoint_weight_facts()
    }
    fn memory_strategy_contract(&self) -> Option<&MemoryProviderContract> {
        self.inner.memory_strategy_contract()
    }
    fn structural_resident_evidence(&self) -> Option<&MemoryStructuralResidentEvidence> {
        self.inner.structural_resident_evidence()
    }
    fn predicted_memory_peak_from_base(&self, bytes: u64) -> MemoryPeakBreakdown {
        self.inner.predicted_memory_peak_from_base(bytes)
    }
    fn veda_workspace_bound(
        &self,
        req: veda::workspace::Request<'_>,
    ) -> Result<Option<veda::workspace::WorkspaceBound>> {
        self.spec
            .read_prepared_files_unchanged(|| self.inner.veda_workspace_bound(req))
    }
    fn memory_strategy_safety_check(&self, context: &MemoryRunContext) -> MemorySafetyDecision {
        let decision = self.check_context(context).and_then(|()| {
            self.spec.read_prepared_files_unchanged(|| {
                Ok::<_, Error>(self.inner.memory_strategy_safety_check(context))
            })
        });
        match decision {
            Ok(decision) => decision,
            Err(error) => MemorySafetyDecision::Reject {
                reason: error.to_string(),
            },
        }
    }
    fn begin_memory_strategy_request(
        &self,
        context: &MemoryRunContext,
    ) -> Result<Option<Box<dyn MemoryRequestScope + '_>>> {
        self.check_context(context)?;
        self.spec
            .read_prepared_files_unchanged(|| self.inner.begin_memory_strategy_request(context))
            .map(|scope| {
                scope.map(|inner| {
                    Box::new(BoundScope { inner, owner: self }) as Box<dyn MemoryRequestScope>
                })
            })
    }
    fn validate(&self, req: &GenerationRequest) -> Result<()> {
        self.run(req, |inner| inner.validate(req))
    }
    fn generate(
        &self,
        req: &GenerationRequest,
        progress: &mut dyn FnMut(Progress),
    ) -> Result<GenerationOutput> {
        self.run(req, |inner| inner.generate(req, progress))
    }
    fn generate_streaming(
        &self,
        req: &GenerationRequest,
        chunk: &mut dyn FnMut(AudioChunk),
        progress: &mut dyn FnMut(Progress),
    ) -> Result<GenerationOutput> {
        self.run(req, |inner| inner.generate_streaming(req, chunk, progress))
    }
    fn generate_with_report(
        &self,
        req: &GenerationRequest,
        progress: &mut dyn FnMut(Progress),
    ) -> Result<GenerationReport> {
        self.run(req, |inner| inner.generate_with_report(req, progress))
    }
    fn open_conversation(
        &self,
        req: &GenerationRequest,
    ) -> Result<Box<dyn ConversationSession + '_>> {
        self.check_request(req)?;
        (self.capacity)(self.footprint)?;
        self.spec
            .read_prepared_files_unchanged(|| self.inner.open_conversation(req))
    }
}

struct BoundScope<'a> {
    inner: Box<dyn MemoryRequestScope + 'a>,
    owner: &'a BoundGenerator,
}
impl MemoryRequestScope for BoundScope<'_> {
    fn configure_request(&mut self, req: &mut GenerationRequest) -> Result<()> {
        self.owner.check_request(req)?;
        let admitted_memory = req.memory;
        let ladder_memory = self
            .owner
            .inner
            .memory_strategy_contract()
            .and_then(|contract| contract.generation_memory(&self.owner.context.selection));
        self.owner
            .spec
            .read_prepared_files_unchanged(|| self.inner.configure_request(req))?;
        // Shared scopes configure LADDER controls and can replace request.memory, omitting
        // independent execution knobs such as sequential CFG. Accept only the exact provider
        // ladder configuration (or an already preserved admitted memory), then restore the
        // evidence-bound full execution. Never let configure_request silently demote CFG.
        if req.memory != admitted_memory && req.memory != ladder_memory {
            return Err(Error::Unsupported(
                "scope configured execution outside admission".into(),
            ));
        }
        req.memory = admitted_memory;
        self.owner.check_request(req)
    }
    fn enter_phase(&mut self, phase: MemoryPhase) -> Result<()> {
        self.owner
            .spec
            .read_prepared_files_unchanged(|| self.inner.enter_phase(phase))
    }
    fn leave_phase(&mut self, phase: MemoryPhase) -> Result<()> {
        self.owner
            .spec
            .read_prepared_files_unchanged(|| self.inner.leave_phase(phase))
    }
    fn configure_decode(
        &mut self,
        edge: u32,
        overlap: u32,
        geometry: MemoryGeometry,
    ) -> Result<()> {
        let p = &self.owner.context.selection.parameters;
        if p.decode_tile_edge != Some(edge)
            || p.decode_overlap != Some(overlap)
            || geometry != self.owner.context.geometry
        {
            return Err(Error::Unsupported(
                "decode geometry differs from admission".into(),
            ));
        }
        self.owner
            .spec
            .read_prepared_files_unchanged(|| self.inner.configure_decode(edge, overlap, geometry))
    }
    fn configure_attention(&mut self, chunk: u32) -> Result<()> {
        if self.owner.context.selection.parameters.attention_chunk_size != Some(chunk) {
            return Err(Error::Unsupported(
                "attention chunk differs from admission".into(),
            ));
        }
        self.owner
            .spec
            .read_prepared_files_unchanged(|| self.inner.configure_attention(chunk))
    }
    fn materialize_transformer_window(&mut self, first: u32, count: u32) -> Result<()> {
        if self
            .owner
            .context
            .selection
            .parameters
            .transformer_window_size
            .is_none_or(|bound| count > bound)
        {
            return Err(Error::Unsupported(
                "transformer window differs from admission".into(),
            ));
        }
        self.owner.spec.read_prepared_files_unchanged(|| {
            self.inner.materialize_transformer_window(first, count)
        })
    }
    fn finish(&mut self, outcome: MemoryRunOutcome) -> Result<()> {
        // Cleanup must run even when a source was replaced or the request cancelled.
        let result = self.inner.finish(outcome);
        self.owner.spec.validate_prepared_file_pins()?;
        result
    }
}

/// Admission-bound restoration transform.
pub struct BoundTransform {
    inner: Box<dyn Transform>,
    spec: LoadSpec,
    binding: [u8; 32],
    /// The admission's `demand + margin` (0 for a capture permit).
    footprint: u64,
    capacity: Box<dyn Fn(u64) -> Result<()>>,
    /// The admission's measured envelope, open from before the load for this object's lifetime
    /// (`None` for a capture permit, which runs on the formula guards alone).
    _envelope: Option<mlx_gen::ios_quantum::AdmittedEnvelope>,
}

impl BoundTransform {
    pub(crate) fn new(
        inner: Box<dyn Transform>,
        spec: LoadSpec,
        binding: [u8; 32],
        footprint: u64,
        capacity: Box<dyn Fn(u64) -> Result<()>>,
        envelope: Option<mlx_gen::ios_quantum::AdmittedEnvelope>,
    ) -> Self {
        Self {
            _envelope: envelope,
            inner,
            spec,
            binding,
            footprint,
            capacity,
        }
    }
    fn run<T>(
        &self,
        req: &TransformRequest,
        call: impl FnOnce(&dyn Transform) -> Result<T>,
    ) -> Result<T> {
        if req.cancel.is_cancelled() {
            return Err(Error::Canceled);
        }
        if work_binding(&IosWork::Restore(req.clone())) != self.binding {
            return Err(Error::Unsupported(
                "iOS restoration request differs from the admitted request".into(),
            ));
        }
        (self.capacity)(self.footprint)?;
        self.spec
            .read_prepared_files_unchanged(|| call(self.inner.as_ref()))
    }
}

impl Transform for BoundTransform {
    fn descriptor(&self) -> &TransformDescriptor {
        self.inner.descriptor()
    }
    fn validate(&self, req: &TransformRequest) -> Result<()> {
        self.run(req, |inner| inner.validate(req))
    }
    fn apply(
        &self,
        req: &TransformRequest,
        on_progress: &mut dyn FnMut(Progress),
    ) -> Result<Image> {
        self.run(req, |inner| inner.apply(req, on_progress))
    }
}

/// Admission-bound Iris depth task.
pub struct BoundDepth {
    inner: Box<dyn IrisDepthEstimator>,
    spec: LoadSpec,
    binding: [u8; 32],
    /// The admission's `demand + margin` (0 for a capture permit).
    footprint: u64,
    capacity: Box<dyn Fn(u64) -> Result<()>>,
    /// The admission's measured envelope, open from before the load for this object's lifetime
    /// (`None` for a capture permit, which runs on the formula guards alone).
    _envelope: Option<mlx_gen::ios_quantum::AdmittedEnvelope>,
}

impl BoundDepth {
    pub(crate) fn new(
        inner: Box<dyn IrisDepthEstimator>,
        spec: LoadSpec,
        binding: [u8; 32],
        footprint: u64,
        capacity: Box<dyn Fn(u64) -> Result<()>>,
        envelope: Option<mlx_gen::ios_quantum::AdmittedEnvelope>,
    ) -> Self {
        Self {
            _envelope: envelope,
            inner,
            spec,
            binding,
            footprint,
            capacity,
        }
    }
}

impl IrisDepthEstimator for BoundDepth {
    fn backend(&self) -> &'static str {
        self.inner.backend()
    }
    fn estimate(
        &self,
        req: &DepthRequest,
        on_progress: &mut dyn FnMut(Progress),
    ) -> Result<DepthOutput> {
        if req.cancel.is_cancelled() {
            return Err(Error::Canceled);
        }
        if work_binding(&IosWork::Depth(req.clone())) != self.binding {
            return Err(Error::Unsupported(
                "iOS depth request differs from the admitted request".into(),
            ));
        }
        (self.capacity)(self.footprint)?;
        self.spec
            .read_prepared_files_unchanged(|| self.inner.estimate(req, on_progress))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::binding::tests::allocations_during;
    use mlx_gen::gen_core::{Conditioning, Image};

    #[test]
    fn borrowed_generation_binding_matches_owned_role_scalar_matrix_without_request_clone() {
        use mlx_gen::gen_core::{ControlKind, GenerationMemory};

        fn image(tag: u8) -> Image {
            Image {
                width: 3,
                height: 2,
                pixels: (0..18).map(|i| tag.wrapping_add(i)).collect(),
            }
        }
        fn buffers(request: &GenerationRequest) -> Vec<(u32, u32, *const u8, usize, usize)> {
            request
                .conditioning
                .iter()
                .map(|role| {
                    let image = match role {
                        Conditioning::Reference { image, .. }
                        | Conditioning::Mask { image }
                        | Conditioning::Control { image, .. } => image,
                        _ => unreachable!("the matrix only constructs SDXL image roles"),
                    };
                    (
                        image.width,
                        image.height,
                        image.pixels.as_ptr(),
                        image.pixels.len(),
                        image.pixels.capacity(),
                    )
                })
                .collect()
        }

        // Initialize fixed neutral-cancellation state before the allocation observation.
        work_binding(&IosWork::Generate(GenerationRequest::default()));
        for scalar in [None, Some(0.0), Some(-0.0), Some(0.625)] {
            for sampler in [None, Some("euler_ancestral"), Some("heun")] {
                for memory_case in 0..3 {
                    for role_case in 0..5 {
                        let reference = || Conditioning::Reference {
                            image: image(11),
                            strength: scalar,
                        };
                        let control = |tag| Conditioning::Control {
                            image: image(tag),
                            kind: ControlKind::Canny,
                            scale: scalar,
                        };
                        let conditioning = match role_case {
                            0 => vec![],
                            1 => vec![reference()],
                            2 => vec![reference(), Conditioning::Mask { image: image(29) }],
                            3 => vec![control(47), control(83)],
                            4 => vec![reference(), control(47), control(83)],
                            _ => unreachable!(),
                        };
                        let memory = match memory_case {
                            0 => None,
                            1 => Some(GenerationMemory::default()),
                            2 => Some(GenerationMemory {
                                stage_residency: true,
                                chunk_attention: true,
                                attention_chunk_size: Some(128),
                                stream_transformer_blocks: true,
                                transformer_window_size: Some(2),
                                ..Default::default()
                            }),
                            _ => unreachable!(),
                        };
                        let mut request = GenerationRequest {
                            prompt: "borrowed λ\n\"quoted\"\\path\t雪".into(),
                            negative_prompt: Some("negative Ω\n\"quoted\"\\tail".into()),
                            sampler: sampler.map(str::to_owned),
                            memory,
                            strength: scalar,
                            conditioning,
                            ..Default::default()
                        };
                        // The legacy clone is TEST-ONLY and outside the counted region.
                        let expected = work_binding(&IosWork::Generate(request.clone()));
                        let before_buffers = buffers(&request);
                        let before_prompt = (request.prompt.as_ptr(), request.prompt.capacity());
                        let before_negative = request
                            .negative_prompt
                            .as_ref()
                            .map(|text| (text.as_ptr(), text.capacity()));
                        let (actual, allocated) =
                            allocations_during(|| generation_binding(&request));
                        assert_eq!(actual, expected, "role/scalar/sampler/memory projection");
                        assert_eq!(
                            allocated, 0,
                            "the warmed borrowed Generate hash must not allocate Rust heap storage"
                        );
                        assert_eq!(buffers(&request), before_buffers);
                        assert_eq!(
                            (request.prompt.as_ptr(), request.prompt.capacity()),
                            before_prompt
                        );
                        assert_eq!(
                            request
                                .negative_prompt
                                .as_ref()
                                .map(|text| (text.as_ptr(), text.capacity())),
                            before_negative
                        );

                        if role_case >= 3 {
                            let end = request.conditioning.len();
                            request.conditioning.swap(end - 2, end - 1);
                            let reordered = generation_binding(&request);
                            assert_ne!(reordered, expected, "Control occurrence order is bound");
                            assert_eq!(
                                reordered,
                                work_binding(&IosWork::Generate(request.clone()))
                            );
                            request.conditioning.swap(end - 2, end - 1);
                        }
                        if let Some(role) = request.conditioning.first_mut() {
                            let image = match role {
                                Conditioning::Reference { image, .. }
                                | Conditioning::Mask { image }
                                | Conditioning::Control { image, .. } => image,
                                _ => unreachable!(),
                            };
                            image.pixels[0] ^= 1;
                            assert_ne!(
                                generation_binding(&request),
                                expected,
                                "the helper retains full pixel-content binding"
                            );
                            match request.conditioning.first_mut().unwrap() {
                                Conditioning::Reference { image, .. }
                                | Conditioning::Mask { image }
                                | Conditioning::Control { image, .. } => image.pixels[0] ^= 1,
                                _ => unreachable!(),
                            }
                        }
                        request.seed = Some(19);
                        request.cancel.cancel();
                        assert_eq!(generation_binding(&request), expected);
                        assert_eq!(
                            generation_binding(&request),
                            work_binding(&IosWork::Generate(request.clone()))
                        );
                        request.prompt.push('!');
                        assert_ne!(generation_binding(&request), expected);
                    }
                }
            }
        }
    }
}
