//! Keep reader publication attached to the loaded provider, including request-time materialization.
use super::*;
use crate::CheckpointReceiptScope;

pub(crate) struct ReceiptGenerator {
    pub(crate) inner: Box<dyn Generator>,
    pub(crate) scope: CheckpointReceiptScope,
}

impl Generator for ReceiptGenerator {
    fn descriptor(&self) -> &ModelDescriptor {
        self.inner.descriptor()
    }
    fn generate_moss_image_edit(
        &self,
        request: &GenerationRequest,
        inputs: crate::moss_image_edit::MossImageEditInputs<'_>,
        progress: &mut dyn FnMut(Progress),
    ) -> Result<crate::moss_image::MossImageAdvancedOutput> {
        self.scope.with(|| {
            self.inner
                .generate_moss_image_edit(request, inputs, progress)
        })
    }
    fn generate_moss_image_edit_with_execution_report(
        &self,
        request: &GenerationRequest,
        inputs: crate::moss_image_edit::MossImageEditInputs<'_>,
        progress: &mut dyn FnMut(Progress),
    ) -> Result<CompletedMossImageEdit> {
        self.scope.with(|| {
            self.inner
                .generate_moss_image_edit_with_execution_report(request, inputs, progress)
        })
    }
    fn moss_image_edit_request_peak(
        &self,
        request: &GenerationRequest,
        inputs: crate::moss_image_edit::MossImageEditInputs<'_>,
    ) -> Result<u64> {
        self.inner.moss_image_edit_request_peak(request, inputs)
    }
    fn moss_image_edit_execution_report(
        &self,
    ) -> Option<crate::moss_image_edit::MossImageEditExecutionReport> {
        self.inner.moss_image_edit_execution_report()
    }
    fn veda_execution_report(&self) -> Option<crate::VedaExecutionReport> {
        self.inner.veda_execution_report()
    }
    fn moss_image_execution_report(&self) -> Option<crate::moss_image::MossImageExecutionReport> {
        self.inner.moss_image_execution_report()
    }
    fn moss_image_owned_weight_bytes(&self) -> Option<u64> {
        self.inner.moss_image_owned_weight_bytes()
    }
    fn retained_weight_bytes(&self) -> Option<u64> {
        self.inner.retained_weight_bytes()
    }
    fn adapter_apply_reports(&self) -> Vec<crate::AdapterApplyReport> {
        self.inner.adapter_apply_reports()
    }
    fn checkpoint_weight_facts(&self) -> Option<crate::CheckpointWeightFacts> {
        self.inner
            .checkpoint_weight_facts()
            .or_else(|| self.scope.facts())
    }
    fn memory_strategy_contract(&self) -> Option<&MemoryProviderContract> {
        self.inner.memory_strategy_contract()
    }
    fn structural_resident_evidence(&self) -> Option<&MemoryStructuralResidentEvidence> {
        self.inner.structural_resident_evidence()
    }
    fn predicted_memory_peak_from_base(
        &self,
        base_predicted_peak_bytes: u64,
    ) -> MemoryPeakBreakdown {
        self.inner
            .predicted_memory_peak_from_base(base_predicted_peak_bytes)
    }
    fn veda_workspace_bound(
        &self,
        req: crate::veda::workspace::Request<'_>,
    ) -> Result<Option<crate::veda::workspace::WorkspaceBound>> {
        self.inner.veda_workspace_bound(req)
    }
    fn memory_strategy_safety_check(&self, context: &MemoryRunContext) -> MemorySafetyDecision {
        self.inner.memory_strategy_safety_check(context)
    }
    fn begin_memory_strategy_request(
        &self,
        context: &MemoryRunContext,
    ) -> Result<Option<Box<dyn MemoryRequestScope + '_>>> {
        self.scope
            .with(|| self.inner.begin_memory_strategy_request(context))
    }
    fn validate(&self, req: &GenerationRequest) -> Result<()> {
        self.inner.validate(req)
    }
    fn plan_execution(&self, req: &GenerationRequest) -> Result<ExecutionIdentity> {
        self.scope.with(|| self.inner.plan_execution(req))
    }
    fn generate(
        &self,
        req: &GenerationRequest,
        on_progress: &mut dyn FnMut(Progress),
    ) -> Result<GenerationOutput> {
        self.scope.with(|| self.inner.generate(req, on_progress))
    }
    fn generate_planned(
        &self,
        request: &GenerationRequest,
        plan: &crate::RenderPlan,
        scope: Option<&mut dyn MemoryRequestScope>,
        on_progress: &mut dyn FnMut(Progress),
    ) -> Result<GenerationReport> {
        self.scope.with(|| {
            self.inner
                .generate_planned(request, plan, scope, on_progress)
        })
    }
    fn generate_planned_with_execution_report(
        &self,
        request: &GenerationRequest,
        plan: &crate::RenderPlan,
        scope: Option<&mut dyn MemoryRequestScope>,
        on_progress: &mut dyn FnMut(Progress),
    ) -> Result<CompletedGeneration> {
        self.scope.with(|| {
            self.inner
                .generate_planned_with_execution_report(request, plan, scope, on_progress)
        })
    }
    fn generate_streaming(
        &self,
        req: &GenerationRequest,
        on_chunk: &mut dyn FnMut(AudioChunk),
        on_progress: &mut dyn FnMut(Progress),
    ) -> Result<GenerationOutput> {
        self.scope
            .with(|| self.inner.generate_streaming(req, on_chunk, on_progress))
    }
    fn generate_streaming_with_execution_report(
        &self,
        req: &GenerationRequest,
        on_chunk: &mut dyn FnMut(AudioChunk),
        on_progress: &mut dyn FnMut(Progress),
    ) -> Result<CompletedGeneration> {
        self.scope.with(|| {
            self.inner
                .generate_streaming_with_execution_report(req, on_chunk, on_progress)
        })
    }
    fn generate_with_report(
        &self,
        req: &GenerationRequest,
        on_progress: &mut dyn FnMut(Progress),
    ) -> Result<GenerationReport> {
        self.scope
            .with(|| self.inner.generate_with_report(req, on_progress))
    }
    fn generate_with_execution_report(
        &self,
        req: &GenerationRequest,
        on_progress: &mut dyn FnMut(Progress),
    ) -> Result<CompletedGeneration> {
        self.scope
            .with(|| self.inner.generate_with_execution_report(req, on_progress))
    }
    fn open_conversation(
        &self,
        req: &GenerationRequest,
    ) -> Result<Box<dyn ConversationSession + '_>> {
        let inner = self.scope.with(|| self.inner.open_conversation(req))?;
        Ok(Box::new(ReceiptConversation {
            inner,
            scope: self.scope.clone(),
        }))
    }
    fn checkpoint_weight_facts_by_source(&self) -> Vec<crate::CheckpointWeightFacts> {
        let mut facts = self.scope.snapshots();
        if let Some(primary) = self.inner.checkpoint_weight_facts() {
            if !facts.contains(&primary) {
                facts.push(primary);
            }
        }
        facts
    }
}
struct ReceiptConversation<'a> {
    inner: Box<dyn ConversationSession + 'a>,
    scope: CheckpointReceiptScope,
}
impl ConversationSession for ReceiptConversation<'_> {
    fn step(
        &mut self,
        turn: &ConversationTurn,
        on_chunk: &mut dyn FnMut(AudioChunk),
        on_progress: &mut dyn FnMut(Progress),
    ) -> Result<AudioTrack> {
        self.scope
            .with(|| self.inner.step(turn, on_chunk, on_progress))
    }
    fn step_with_execution_report(
        &mut self,
        turn: &ConversationTurn,
        on_chunk: &mut dyn FnMut(AudioChunk),
        on_progress: &mut dyn FnMut(Progress),
    ) -> Result<CompletedAudioTurn> {
        self.scope.with(|| {
            self.inner
                .step_with_execution_report(turn, on_chunk, on_progress)
        })
    }
    fn finish(&mut self) -> Result<()> {
        self.scope.with(|| self.inner.finish())
    }
}
