//! Admission-bound iOS execution for every registered model (generators, the Iris restoration
//! transform and the Iris depth task). The wrapped provider is never returned to callers: every
//! entry point rechecks the exact admitted work (geometry, tile, mode, conditioning, prompt,
//! execution — everything but seed and cancellation), live capacity and the sealed source pins
//! BEFORE reaching the provider, and preserves the caller's real cancellation handle.
use crate::ios_admission::{work_binding, IosWork};
use gen_core_iris::depth::{DepthOutput, DepthRequest, IrisDepthEstimator};
use mlx_gen::gen_core::*;

fn generation_binding(req: &GenerationRequest) -> [u8; 32] {
    crate::binding::generation_binding(req)
}

/// One actual configured invocation. Preserve the original typed body error;
/// drain on success/error/cancel and after provider cleanup. A successful
/// endpoint is published only after ALL body/fence/provider outcomes succeeded.
fn invoke_native_retirement<T>(
    lease: &mlx_gen::ios_quantum::NativeRetirementLease,
    streams: &std::rc::Rc<mlx_gen::ios_quantum::NativeRetirementStreams>,
    actual: &GenerationRequest,
    body: impl FnOnce() -> Result<T>,
    finish: impl FnOnce(MemoryRunOutcome) -> Result<()>,
) -> Result<T> {
    mlx_gen::ios_quantum::invoke_native_retirement(lease, streams, actual, body, finish, |error| {
        matches!(error, Error::Canceled)
    })
}

/// Read-only description of the SAME realised selection that created this
/// generator. No load/permit methods, public constructor, mutation or authority.
pub struct GeneratorAdmissionInfo {
    pub(crate) index: usize,
    pub(crate) key: crate::evidence::EvidenceKey,
    pub(crate) demand_bytes: u64,
    pub(crate) margin_bytes: u64,
    pub(crate) required_bytes: u64,
    pub(crate) basis: crate::ios_admission::AdmissionBasis,
}
impl GeneratorAdmissionInfo {
    pub fn index(&self) -> usize {
        self.index
    }
    pub fn evidence_key(&self) -> &crate::evidence::EvidenceKey {
        &self.key
    }
    pub fn demand_bytes(&self) -> u64 {
        self.demand_bytes
    }
    pub fn margin_bytes(&self) -> u64 {
        self.margin_bytes
    }
    pub fn required_bytes(&self) -> u64 {
        self.required_bytes
    }
    pub fn basis(&self) -> crate::ios_admission::AdmissionBasis {
        self.basis
    }
}

pub(crate) struct GeneratorExecution {
    pub info: GeneratorAdmissionInfo,
    pub envelope: Option<mlx_gen::ios_quantum::AdmittedEnvelope>,
    pub native_retirement: mlx_gen::ios_pricing::IosNativeRetirementExecution,
    pub streams: Option<std::rc::Rc<mlx_gen::ios_quantum::NativeRetirementStreams>>,
}
pub struct BoundGenerator {
    inner: Box<dyn Generator>,
    admission_info: GeneratorAdmissionInfo,
    native_retirement: mlx_gen::ios_pricing::IosNativeRetirementExecution,
    retirement_streams: Option<std::rc::Rc<mlx_gen::ios_quantum::NativeRetirementStreams>>,
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
        execution: GeneratorExecution,
    ) -> Self {
        Self {
            _envelope: execution.envelope,
            inner,
            admission_info: execution.info,
            native_retirement: execution.native_retirement,
            retirement_streams: execution.streams,
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
    pub fn admitted_selection(&self) -> &GeneratorAdmissionInfo {
        &self.admission_info
    }
    /// ACTUAL selected receipt requirement (ordinary-first may choose either).
    /// Metadata identity only; not device success or physical/process permission.
    pub fn native_retirement_execution(
        &self,
    ) -> mlx_gen::ios_pricing::IosNativeRetirementExecution {
        self.native_retirement
    }
    fn requires_retired_lifecycle(&self) -> bool {
        self.native_retirement
            == mlx_gen::ios_pricing::IosNativeRetirementExecution::ProviderLifecycle
    }
    fn retirement_binding(&self) -> mlx_gen::ios_quantum::NativeRetirementBinding {
        mlx_gen::ios_quantum::NativeRetirementBinding {
            owner_token: self as *const Self as usize,
            work: self.request,
            provider: self.inner.descriptor().id,
        }
    }
    fn run_generation<T>(
        &self,
        req: &GenerationRequest,
        call: impl FnOnce(&dyn Generator, &GenerationRequest) -> Result<T>,
    ) -> Result<T> {
        if !self.requires_retired_lifecycle() {
            let _ordinary = mlx_gen::ios_quantum::OrdinaryNativeRetirementScope::enter();
            return self.run(req, |inner| call(inner, req));
        }
        let outcome=self.run(req, |inner| {
            let binding = self.retirement_binding();
            let streams=self.retirement_streams.as_ref().ok_or_else(||Error::Unsupported("required native lifecycle has no pre-load stream cache".into()))?;
            streams.validate_current()?;
            mlx_gen::ios_quantum::validate_native_retirement_scope()?;
            if mlx_gen::ios_quantum::native_retirement_binding().is_some() {
                return Err(Error::Unsupported("generation cannot reenter an active admitted retirement invocation".into()));
            }
            // A matching external scope has ALREADY run the exact provider
            // configure hook. Holding it alone does not activate retirement.
            if let Some(lease) = configured_bound_retirement(binding) {
                return invoke_native_retirement(&lease, streams, req, || call(inner, req), |_| Ok(()));
            }
            // Caller omitted optional begin: establish the real provider scope
            // internally. Borrow the ORIGINAL carrier; never deep-clone pixels
            // or manufacture a metadata-only GenerationRequest.
            let mut scope = self.begin_memory_strategy_request(&self.context)?
                .ok_or_else(|| Error::Unsupported("admitted retirement requires a provider lifecycle scope".into()))?;
            if let Err(primary) = scope.configure_admitted_request(req) {
                if let Err(cleanup)=streams.drain(){eprintln!("configuration native drain failed: {cleanup}");}
                if let Err(cleanup) = scope.finish(MemoryRunOutcome::Error { message: primary.to_string() }) {
                    eprintln!("admitted lifecycle configuration cleanup failed: {cleanup}");
                }
                if let Err(cleanup)=streams.drain(){eprintln!("configuration final native drain failed: {cleanup}");}
                return Err(primary);
            }
            let lease=match configured_bound_retirement(binding) {
                Some(lease)=>lease,
                None=>{
                    let primary=Error::Unsupported("successful provider configuration did not bind source retirement lifecycle".into());
                    if let Err(cleanup)=streams.drain(){eprintln!("unbound configuration drain failed: {cleanup}");}
                    if let Err(cleanup)=scope.finish(MemoryRunOutcome::Error{message:primary.to_string()}){eprintln!("unbound provider cleanup failed: {cleanup}");}
                    if let Err(cleanup)=streams.drain(){eprintln!("unbound configuration final drain failed: {cleanup}");}
                    return Err(primary);
                }
            };
            invoke_native_retirement(&lease, streams, req, || call(inner, req), |outcome| scope.finish(outcome))
        });
        // Includes cancellation/capacity/source-pin refusal BEFORE the body: an
        // already-loaded generator may still have load-check captures to drain.
        if outcome.is_err() {
            let cleanup = self
                .retirement_streams
                .as_ref()
                .map_or(Ok(()), |streams| streams.drain());
            if let Err(cleanup) = cleanup {
                eprintln!("admitted generation final error-path drain failed: {cleanup}");
            }
        }
        outcome
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
            Err(error) => MemorySafetyDecision::rejected(error),
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
                    Box::new(BoundScope {
                        inner,
                        owner: self,
                        retirement_scope: None,
                    }) as Box<dyn MemoryRequestScope>
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
        self.run_generation(req, |inner, exact| inner.generate(exact, progress))
    }
    fn generate_streaming(
        &self,
        req: &GenerationRequest,
        chunk: &mut dyn FnMut(AudioChunk),
        progress: &mut dyn FnMut(Progress),
    ) -> Result<GenerationOutput> {
        self.run_generation(req, |inner, exact| {
            inner.generate_streaming(exact, chunk, progress)
        })
    }
    fn generate_with_report(
        &self,
        req: &GenerationRequest,
        progress: &mut dyn FnMut(Progress),
    ) -> Result<GenerationReport> {
        self.run_generation(req, |inner, exact| {
            inner.generate_with_report(exact, progress)
        })
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

// Only successful configuration by this PRIVATE wrapper enters the reuse
// registry. A public shared selector/registration cannot bypass provider setup.
struct BoundConfiguredRetirement {
    binding: mlx_gen::ios_quantum::NativeRetirementBinding,
    registration: mlx_gen::ios_quantum::ConfiguredNativeRetirement,
}
thread_local! {
    static BOUND_RETIREMENT: std::cell::RefCell<Vec<std::rc::Weak<BoundConfiguredRetirement>>> = const { std::cell::RefCell::new(Vec::new()) };
}
fn configured_bound_retirement(
    binding: mlx_gen::ios_quantum::NativeRetirementBinding,
) -> Option<mlx_gen::ios_quantum::NativeRetirementLease> {
    BOUND_RETIREMENT.with(|slot| {
        let mut entries = slot.borrow_mut();
        entries.retain(|entry| entry.strong_count() != 0);
        entries
            .iter()
            .rev()
            .filter_map(std::rc::Weak::upgrade)
            .find(|entry| entry.binding == binding)
            .map(|entry| entry.registration.lease())
    })
}
impl BoundConfiguredRetirement {
    fn register(binding: mlx_gen::ios_quantum::NativeRetirementBinding) -> std::rc::Rc<Self> {
        let record = std::rc::Rc::new(Self {
            binding,
            registration: mlx_gen::ios_quantum::ConfiguredNativeRetirement::register(binding),
        });
        BOUND_RETIREMENT.with(|slot| {
            let mut entries = slot.borrow_mut();
            entries.retain(|entry| entry.strong_count() != 0);
            entries.push(std::rc::Rc::downgrade(&record));
        });
        record
    }
}
struct BoundScope<'a> {
    inner: Box<dyn MemoryRequestScope + 'a>,
    owner: &'a BoundGenerator,
    retirement_scope: Option<std::rc::Rc<BoundConfiguredRetirement>>,
}
impl BoundScope<'_> {
    fn register_configured_retirement(&mut self, req: &GenerationRequest) -> Result<()> {
        self.owner.check_request(req)?;
        if self.owner.requires_retired_lifecycle() {
            if req.memory.is_none() {
                return Err(Error::Unsupported("retired lifecycle requires the exact admitted Some(execution), not legacy None".into()));
            }
            if self.retirement_scope.is_none() {
                self.retirement_scope = Some(BoundConfiguredRetirement::register(
                    self.owner.retirement_binding(),
                ));
            }
        }
        Ok(())
    }
}
impl MemoryRequestScope for BoundScope<'_> {
    fn configure_request(&mut self, req: &mut GenerationRequest) -> Result<()> {
        self.owner.check_request(req)?;
        if self.owner.requires_retired_lifecycle() && self.retirement_scope.is_some() {
            return Ok(());
        }
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
        self.register_configured_retirement(req)
    }
    fn configure_admitted_request(&mut self, req: &GenerationRequest) -> Result<()> {
        self.owner.check_request(req)?;
        if self.owner.requires_retired_lifecycle() && self.retirement_scope.is_some() {
            return Ok(());
        }
        self.owner
            .spec
            .read_prepared_files_unchanged(|| self.inner.configure_admitted_request(req))?;
        // Unlike mutable ladder configuration, the read-only delegate must
        // accept the ORIGINAL full canonical request without rewriting it.
        self.register_configured_retirement(req)
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
        if let Some(record) = &self.retirement_scope {
            record.registration.invalidate();
        }
        self.retirement_scope.take();
        let pins = self.owner.spec.validate_prepared_file_pins();
        match (result, pins) {
            (Err(primary), Err(secondary)) => {
                eprintln!("provider cleanup source-pin recheck also failed: {secondary}");
                Err(primary)
            }
            (Err(primary), _) => Err(primary),
            (Ok(()), pins) => pins,
        }
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
#[path = "bound_lifecycle_tests.rs"]
mod lifecycle_tests;

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
