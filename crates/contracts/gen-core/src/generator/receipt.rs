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
    fn retained_weight_bytes(&self) -> Option<u64> {
        self.inner.retained_weight_bytes()
    }
    fn video_plan_resident_weight_bytes(&self, plan: &crate::VideoRenderPlan) -> Option<u64> {
        self.inner.video_plan_resident_weight_bytes(plan)
    }
    fn video_load_binding(&self) -> Option<&crate::RenderLoadBinding> {
        self.inner.video_load_binding()
    }
    fn video_memory_evidence_tier(
        &self,
        plan: &crate::VideoRenderPlan,
        req: &GenerationRequest,
    ) -> Result<Option<crate::MemoryNumericTier>> {
        self.scope
            .with_prepared(|| self.inner.video_memory_evidence_tier(plan, req))
    }
    fn project_video_memory_context(
        &self,
        plan: &crate::VideoRenderPlan,
        req: &GenerationRequest,
        context: &mut MemoryRunContext,
    ) -> Result<bool> {
        self.scope
            .with_prepared(|| self.inner.project_video_memory_context(plan, req, context))
    }
    fn validate_video_plan(
        &self,
        plan: &crate::VideoRenderPlan,
        req: &GenerationRequest,
    ) -> Result<()> {
        self.scope
            .with_prepared(|| self.inner.validate_video_plan(plan, req))
    }
    fn retained_component_state(&self) -> Result<Option<GeneratorComponentResidency>> {
        self.inner.retained_component_state()
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
    fn request_memory_peak(&self, req: &GenerationRequest) -> Result<Option<u64>> {
        self.inner.request_memory_peak(req)
    }
    fn request_workspace_bound(
        &self,
        req: &GenerationRequest,
    ) -> Result<Option<RequestWorkspaceBound>> {
        self.inner.request_workspace_bound(req)
    }
    fn memory_strategy_safety_check(&self, context: &MemoryRunContext) -> MemorySafetyDecision {
        self.inner.memory_strategy_safety_check(context)
    }
    fn begin_memory_strategy_request(
        &self,
        context: &MemoryRunContext,
    ) -> Result<Option<Box<dyn MemoryRequestScope + '_>>> {
        self.scope
            .with_prepared(|| self.inner.begin_memory_strategy_request(context))
    }
    fn begin_planned_memory_strategy_request(
        &self,
        request: &GenerationRequest,
        plan: &crate::RenderPlan,
        context: &MemoryRunContext,
    ) -> Result<Option<Box<dyn MemoryRequestScope + '_>>> {
        self.scope.with_prepared(|| {
            self.inner
                .begin_planned_memory_strategy_request(request, plan, context)
        })
    }
    fn validate(&self, req: &GenerationRequest) -> Result<()> {
        self.inner.validate(req)
    }
    fn plan_execution(&self, req: &GenerationRequest) -> Result<ExecutionIdentity> {
        self.scope.with_prepared(|| self.inner.plan_execution(req))
    }
    fn generate(
        &self,
        req: &GenerationRequest,
        on_progress: &mut dyn FnMut(Progress),
    ) -> Result<GenerationOutput> {
        self.scope
            .with_prepared(|| self.inner.generate(req, on_progress))
    }
    fn generate_planned(
        &self,
        request: &GenerationRequest,
        plan: &crate::RenderPlan,
        scope: Option<&mut dyn MemoryRequestScope>,
        on_progress: &mut dyn FnMut(Progress),
    ) -> Result<GenerationReport> {
        self.scope.with_prepared(|| {
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
        self.scope.with_prepared(|| {
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
            .with_prepared(|| self.inner.generate_streaming(req, on_chunk, on_progress))
    }
    fn generate_streaming_with_execution_report(
        &self,
        req: &GenerationRequest,
        on_chunk: &mut dyn FnMut(AudioChunk),
        on_progress: &mut dyn FnMut(Progress),
    ) -> Result<CompletedGeneration> {
        self.scope.with_prepared(|| {
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
            .with_prepared(|| self.inner.generate_with_report(req, on_progress))
    }
    fn generate_with_execution_report(
        &self,
        req: &GenerationRequest,
        on_progress: &mut dyn FnMut(Progress),
    ) -> Result<CompletedGeneration> {
        self.scope
            .with_prepared(|| self.inner.generate_with_execution_report(req, on_progress))
    }
    fn generate_with_scoped_execution_report(
        &self,
        req: &GenerationRequest,
        scope: Option<&mut dyn MemoryRequestScope>,
        on_progress: &mut dyn FnMut(Progress),
    ) -> Result<CompletedGeneration> {
        self.scope.with_prepared(|| {
            self.inner
                .generate_with_scoped_execution_report(req, scope, on_progress)
        })
    }
    fn generate_video_planned_with_execution_report(
        &self,
        req: &GenerationRequest,
        plan: &crate::VideoRenderPlan,
        scope: Option<&mut dyn MemoryRequestScope>,
        on_progress: &mut dyn FnMut(Progress),
    ) -> Result<CompletedGeneration> {
        self.scope.with_prepared(|| {
            self.inner
                .generate_video_planned_with_execution_report(req, plan, scope, on_progress)
        })
    }
    fn open_conversation(
        &self,
        req: &GenerationRequest,
    ) -> Result<Box<dyn ConversationSession + '_>> {
        let inner = self
            .scope
            .with_prepared(|| self.inner.open_conversation(req))?;
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
            .with_prepared(|| self.inner.step(turn, on_chunk, on_progress))
    }
    fn step_with_execution_report(
        &mut self,
        turn: &ConversationTurn,
        on_chunk: &mut dyn FnMut(AudioChunk),
        on_progress: &mut dyn FnMut(Progress),
    ) -> Result<CompletedAudioTurn> {
        self.scope.with_prepared(|| {
            self.inner
                .step_with_execution_report(turn, on_chunk, on_progress)
        })
    }
    fn finish(&mut self) -> Result<()> {
        self.scope.with_prepared(|| self.inner.finish())
    }
}
