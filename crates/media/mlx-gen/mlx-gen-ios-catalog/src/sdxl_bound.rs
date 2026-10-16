//! Admission-bound SDXL execution. The wrapped provider is never returned to callers.
use crate::sdxl::actual_request_key;
use mlx_gen::gen_core::*;

pub struct BoundSdxlGenerator {
    inner: Box<dyn Generator>,
    spec: LoadSpec,
    context: MemoryRunContext,
    request: [u8; 32],
    footprint: u64,
    capacity: Box<dyn Fn(u64) -> Result<()>>,
}

impl BoundSdxlGenerator {
    pub(crate) fn new(
        inner: Box<dyn Generator>,
        spec: LoadSpec,
        context: MemoryRunContext,
        request: [u8; 32],
        footprint: u64,
        capacity: Box<dyn Fn(u64) -> Result<()>>,
    ) -> Self {
        Self {
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
        if actual_request_key(req) != self.request {
            return Err(Error::Unsupported("SDXL request differs from the admitted actual request (geometry/mode/conditioning/prompt/execution)".into()));
        }
        Ok(())
    }
    fn check_context(&self, context: &MemoryRunContext) -> Result<()> {
        if context != &self.context {
            return Err(Error::Unsupported(
                "SDXL lifecycle context differs from admission".into(),
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

impl Generator for BoundSdxlGenerator {
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
    owner: &'a BoundSdxlGenerator,
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
