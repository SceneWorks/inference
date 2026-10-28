//! The `Generator` and `ConversationSession` traits.

use super::*;

/// A prompt-conditioned media generator. `generate` is **synchronous** (long/blocking; the
/// worker runs each job on its own thread); the request carries a cancel flag and
/// `on_progress` streams step/decode progress.
pub trait Generator {
    /// Identity + capabilities + modality (drives `validate` and consumer UI introspection).
    fn descriptor(&self) -> &ModelDescriptor;

    /// Native MOSS edit entry point available through catalog-loaded trait objects.
    fn generate_moss_image_edit(
        &self,
        _request: &GenerationRequest,
        _inputs: crate::moss_image_edit::MossImageEditInputs<'_>,
        _progress: &mut dyn FnMut(Progress),
    ) -> Result<crate::moss_image::MossImageAdvancedOutput> {
        Err(Error::Unsupported(
            "provider has no native MOSS edit path".into(),
        ))
    }
    /// Complete structural edit peak, including reference encoding and request-local K/V.
    fn moss_image_edit_request_peak(
        &self,
        _request: &GenerationRequest,
        _inputs: crate::moss_image_edit::MossImageEditInputs<'_>,
    ) -> Result<u64> {
        Err(Error::Unsupported(
            "provider has no native MOSS edit admission".into(),
        ))
    }
    /// Last completed edit; cleared on every attempt, including failures and cancellation.
    fn moss_image_edit_execution_report(
        &self,
    ) -> Option<crate::moss_image_edit::MossImageEditExecutionReport> {
        None
    }

    /// Observed sparse-attention outcome of the most recent request, when adopted.
    fn veda_execution_report(&self) -> Option<crate::VedaExecutionReport> {
        None
    }

    /// Actual last successful MOSS execution. Providers clear this under their lifecycle lock
    /// before every attempt and publish only after all outputs and a final cancellation check.
    /// Callers must serialize generation and receipt retrieval together; concurrent attempts may
    /// clear/replace the last-completed receipt. None never implies successful MOSS execution.
    fn moss_image_execution_report(&self) -> Option<crate::moss_image::MossImageExecutionReport> {
        None
    }

    /// MOSS-owned materialized weight buffers retained by this provider, excluding workspace,
    /// allocator slack, source payloads and other providers. This metadata-only getter never
    /// evaluates tensors. Call within the serialized cache transaction; warm admission may credit
    /// at most this value bounded by the current committed snapshot. None grants no extra credit.
    fn moss_image_owned_weight_bytes(&self) -> Option<u64> {
        None
    }

    /// Actual per-adapter install outcomes from the most recent successful generation.
    ///
    /// Most providers either reject an adapter atomically or have no partial-install surface, so the
    /// compatibility default is empty. Providers that can accept part of a file override this and
    /// publish their engine-owned result after [`generate`](Self::generate); consumers must not
    /// predict the outcome by inspecting an adapter header.
    fn adapter_apply_reports(&self) -> Vec<crate::AdapterApplyReport> {
        Vec::new()
    }

    /// The **three correlated facts** about the checkpoint this generator loaded (sc-21484): what
    /// the source stores per codec, what this host can execute natively, and what actually
    /// materialized — split per [`crate::ExecutionRepresentation`], so a source stored `nvfp4-v1`
    /// that ran the packed W4A4 operand is distinguishable from the same source decoded to dense
    /// BF16.
    ///
    /// **This is the surface a consumer across the worker boundary holds.** The producing accessors
    /// (`candle_gen_krea::loader::Weights::checkpoint_weight_facts`, the shared
    /// `LogicalWeightReader`) live on loader types a worker never sees; it hands the runtime
    /// registry a [`crate::LoadSpec`] and receives a `Box<dyn Generator>`. Providers riding the
    /// shared logical-weight reader propagate their facts here through a
    /// [`crate::CheckpointFactsSink`].
    ///
    /// `None` — the default every provider inherits — means **this load produced no compiled
    /// plan**: a directory-sourced import, a packed-tier variant resolved to a folder, or a
    /// provider that has not adopted the seam. It never means "the source stores nothing
    /// quantized", and it is *not* a claim that the run was dense. A provider whose components load
    /// lazily also reports `None` until its first materialization, because until then there is no
    /// measured receipt to report.
    fn checkpoint_weight_facts(&self) -> Option<crate::CheckpointWeightFacts> {
        None
    }

    /// The loaded provider's memory-strategy contract, when adopted. Existing providers inherit
    /// `None`, which is the compatibility-safe resident-only/unverified state.
    fn memory_strategy_contract(&self) -> Option<&MemoryProviderContract> {
        None
    }

    /// Load-exact structural evidence for a deliberately narrow estimate-backed Resident route.
    /// This never substitutes for measured calibration and is absent by default.
    fn structural_resident_evidence(&self) -> Option<&MemoryStructuralResidentEvidence> {
        None
    }

    /// Canonical construction of a full provider peak from a caller's base-model estimate.
    /// Non-adopting providers and contracts without typed auxiliary components preserve the input
    /// scalar byte-for-byte.
    fn predicted_memory_peak_from_base(
        &self,
        base_predicted_peak_bytes: u64,
    ) -> MemoryPeakBreakdown {
        self.memory_strategy_contract().map_or(
            MemoryPeakBreakdown::from_unattributed(base_predicted_peak_bytes),
            |contract| contract.predicted_peak_from_base(base_predicted_peak_bytes),
        )
    }

    /// Optional problem-bounded sparse residency. Disabled/absent settings return None without
    /// loading predictors or allocating device state. Worker admission calls this per queued job.
    fn veda_workspace_bound(
        &self,
        _req: crate::veda::workspace::Request<'_>,
    ) -> Result<Option<crate::veda::workspace::WorkspaceBound>> {
        Ok(None)
    }

    /// Provider safety defense in depth. This can reject a shared worker selection but cannot
    /// replace its strategy, parameters, or numeric tier. Non-adopting providers accept only the
    /// resident baseline.
    fn memory_strategy_safety_check(&self, context: &MemoryRunContext) -> MemorySafetyDecision {
        match self.memory_strategy_contract() {
            Some(contract) => default_memory_strategy_safety_check(contract, context),
            None if context.selection.strategy == MemoryStrategy::Resident => {
                MemorySafetyDecision::Accept
            }
            None => MemorySafetyDecision::Reject {
                reason: format!(
                    "{} has not adopted the shared memory-strategy contract",
                    self.descriptor().id
                ),
            },
        }
    }

    /// Open request-scoped lifecycle state after the shared selection passes the provider safety
    /// check. Existing providers return `Ok(None)` and therefore cannot execute optimized rungs.
    fn begin_memory_strategy_request(
        &self,
        context: &MemoryRunContext,
    ) -> Result<Option<Box<dyn MemoryRequestScope + '_>>> {
        if self
            .memory_strategy_contract()
            .is_some_and(requires_memory_request_scope)
        {
            return Err(Error::Unsupported(format!(
                "{} advertises an implemented optimized memory strategy but does not open a request scope",
                self.descriptor().id
            )));
        }
        let _ = context;
        Ok(None)
    }

    /// Reject a request this model cannot serve (unsupported conditioning, guidance on a
    /// distilled model, out-of-range size/count, …) before doing expensive work.
    fn validate(&self, req: &GenerationRequest) -> Result<()>;

    /// Run generation to completion (or until `req.cancel` trips).
    fn generate(
        &self,
        req: &GenerationRequest,
        on_progress: &mut dyn FnMut(Progress),
    ) -> Result<GenerationOutput>;

    /// **Incremental / low-latency audio synthesis** (sc-12846) — the streaming counterpart of
    /// [`generate`](Self::generate), the audio analog of `core_llm`'s token-streaming
    /// [`TextLlm::generate`](crate::core_llm::TextLlm::generate). A realtime/streaming provider
    /// (`Modality::Audio` with [`Capabilities::supports_streaming`]) emits an [`AudioChunk`] through
    /// `on_chunk` as each block of PCM becomes available — so a consumer can start playback well
    /// before the full track finishes — and returns the **same** [`GenerationOutput`] as
    /// [`generate`](Self::generate) for the identical request. `on_progress` carries the usual
    /// step/decode [`Progress`] alongside the audio chunks, and cancellation rides
    /// [`GenerationRequest::cancel`] exactly as in [`generate`](Self::generate) (a mid-stream cancel
    /// must stop promptly, returning the typed [`Error::Canceled`]).
    ///
    /// ## Why a separate entry point (and not a `Progress` payload)
    ///
    /// [`Progress`] is `Copy + Eq` and is matched exhaustively across the workspace; widening it to
    /// carry a `Vec<f32>` of PCM would strip those derives and ripple a breaking change through every
    /// consumer. A dedicated method with a **default implementation** keeps the streaming surface
    /// strictly *additive* and *tensor-free*: every existing [`Generator`] — image, video, and the
    /// one-shot audio families — inherits the default unchanged and is byte-for-byte unaffected.
    ///
    /// ## The default implementation (one-shot as "collect all chunks", inverted)
    ///
    /// The default runs the one-shot [`generate`](Self::generate) and, when it produced audio, emits
    /// the whole track as a single terminal [`AudioChunk`] (`index 0`). This satisfies the
    /// [`AudioChunk`] reassembly law trivially (one chunk == the whole track) and means **every**
    /// provider — streaming or not — can be driven through this entry point. A model whose
    /// [`Capabilities::supports_streaming`] is `false` (the default) is not expected to be incremental
    /// here; the flag is the opt-in signal a consumer reads to know whether it will get genuine
    /// low-latency chunks or one terminal chunk.
    ///
    /// A streaming provider **overrides** this to emit chunks incrementally, and drives its own
    /// one-shot [`generate`](Self::generate) by collecting all chunks into the returned
    /// [`GenerationOutput::Audio`] — the streaming path is the primary implementation, `generate` its
    /// aggregate, so the two never diverge.
    fn generate_streaming(
        &self,
        req: &GenerationRequest,
        on_chunk: &mut dyn FnMut(AudioChunk),
        on_progress: &mut dyn FnMut(Progress),
    ) -> Result<GenerationOutput> {
        let out = self.generate(req, on_progress)?;
        if let GenerationOutput::Audio(track) = &out {
            on_chunk(AudioChunk {
                samples: track.samples.clone(),
                sample_rate: track.sample_rate,
                channels: track.channels,
                index: 0,
            });
        }
        Ok(out)
    }

    /// **Generate and report** (sc-22988): the render's [`GenerationReport`] — its output, the
    /// reproducibility record it published ([`AudioParams::artifacts`]) and its non-fatal
    /// warnings (e.g. a truncated autoregressive phase).
    ///
    /// Additive for the reason [`generate_streaming`](Self::generate_streaming) is: a dedicated
    /// default-implemented method instead of new fields on [`GenerationOutput`] / [`AudioTrack`],
    /// whose literals and exhaustive matches span the workspace and its consumers. The default
    /// wraps [`generate`](Self::generate) in [`GenerationReport::from_output`] (no record, no
    /// warnings), so every provider can be driven through it and is byte-for-byte unaffected.
    ///
    /// A provider that publishes records or reports warnings overrides this as its primary
    /// implementation and derives [`generate`](Self::generate) from it. That is the only way to
    /// receive an artifacts-only render ([`SongParams::plan_only`], `output: None`), which
    /// [`generate`](Self::generate) refuses.
    fn generate_with_report(
        &self,
        req: &GenerationRequest,
        on_progress: &mut dyn FnMut(Progress),
    ) -> Result<GenerationReport> {
        self.generate(req, on_progress)
            .map(GenerationReport::from_output)
    }

    /// **Open a stateful multi-turn conversational session** (sc-14150) — the stateful counterpart
    /// (path **B**) of the stateless [`Conditioning::ConversationHistory`] carrier (path **A**). A
    /// context-aware conversational TTS model (e.g. MOSS-TTS-Realtime, a voice-agent foundation model)
    /// synthesizes turn *N* conditioned on turns *1..N-1*. The stateless path rebuilds the whole
    /// conversation prefix on every [`generate`](Self::generate) call; a session instead keeps the
    /// model's live cross-turn state (the warm KV cache) **hot across `step`s**, so a turn does not
    /// recompute the prefix — the low-latency real-time voice-agent path, where an upstream LLM feeds
    /// assistant turns incrementally.
    ///
    /// ## Why a session opens from the loaded generator (and not a weight-reloading registration kind)
    ///
    /// The session shares this **already-loaded** model's weights through `&self` — it is *not* a
    /// second registry kind whose `load` would re-read the checkpoint (doubling residency and fighting
    /// the single-backend-per-bundle invariant). Discovery is the additive
    /// [`Capabilities::supports_conversation_session`] flag on the already-registered descriptor,
    /// exactly as [`Capabilities::supports_streaming`] advertises the streaming path. This mirrors how
    /// [`generate_streaming`](Self::generate_streaming) is an **additive, default-implemented** method
    /// on this same trait: every existing [`Generator`] inherits the default below and is byte-for-byte
    /// unaffected.
    ///
    /// `req` carries the conversation-level constants read once at open — the seed base, the audio
    /// sub-block (target sample rate / language), and any [`Conditioning::ReferenceAudio`] voice-clone
    /// clip that is held constant across the whole conversation. Per-turn text + audio arrive through
    /// [`ConversationSession::step`]. The default returns the typed [`Error::Unsupported`]; a provider
    /// advertising [`Capabilities::supports_conversation_session`] overrides it.
    fn open_conversation(
        &self,
        req: &GenerationRequest,
    ) -> Result<Box<dyn ConversationSession + '_>> {
        let _ = req;
        Err(Error::Unsupported(format!(
            "{}: stateful conversational sessions are not supported",
            self.descriptor().id
        )))
    }
}

pub(super) fn requires_memory_request_scope(contract: &MemoryProviderContract) -> bool {
    contract.strategies.iter().any(|capability| {
        capability.strategy.is_optimized()
            && matches!(
                capability.support,
                crate::MemoryStrategySupport::Implemented
            )
    })
}

/// A **stateful multi-turn conversational TTS session** (sc-14150, path **B**) — opened from a loaded
/// [`Generator`] via [`Generator::open_conversation`], it holds the model's live cross-turn state (the
/// warm KV cache) so each [`step`](Self::step) synthesizes the next turn conditioned on every prior
/// turn **without** recomputing the conversation prefix. This is the low-latency real-time voice-agent
/// path; the stateless [`Conditioning::ConversationHistory`] carrier (path **A**) is the equivalent
/// batch render.
///
/// **The A≡B equivalence law:** for the same conversation + seed, driving the turns one-per-`step`
/// through a session must emit **byte-identical** audio to rendering the same conversation in one
/// stateless [`generate`](Generator::generate) call carrying the whole
/// [`Conditioning::ConversationHistory`] — the session is a warm-cache *optimization* of the batch
/// path, not a different computation (the multi-turn analogue of the
/// `generate`≡`generate_streaming` law the streaming testkit enforces). The `gen-core-testkit`
/// `check_multi_turn` conformance check enforces this, so a session that drifts from the batch path
/// is a CI failure rather than a field report.
///
/// The trait is object-safe and tensor-free: turns cross the boundary as [`ConversationTurn`] (PCM
/// [`AudioTrack`], never model tokens). A session borrows the loaded model (`+ '_` on the boxed
/// handle) and is dropped to release its state; [`finish`](Self::finish) is an explicit,
/// idempotent close for symmetry with the reference `open → step → finish` handshake.
pub trait ConversationSession {
    /// Advance the conversation by one `turn`, returning that turn's audio.
    ///
    /// - A **synthesis** turn ([`ConversationRole::Assistant`] with `audio: None`) is generated
    ///   conditioned on every prior turn folded into this session, and its generated audio is
    ///   retained as context for later turns; the returned [`AudioTrack`] is the synthesized speech,
    ///   streamed incrementally through `on_chunk` (the [`AudioChunk`] reassembly law holds, as in
    ///   [`generate_streaming`](Generator::generate_streaming)).
    /// - A **context** turn (any turn carrying `audio: Some`) is folded into the session as prior
    ///   context (the user's speech, or a previously-generated assistant turn resumed from another
    ///   session); no synthesis happens and the provided track is returned unchanged (echoed).
    ///
    /// `on_progress` carries the usual step/decode [`Progress`]; cancellation rides
    /// [`GenerationRequest::cancel`] on the request passed to [`Generator::open_conversation`],
    /// returning the typed [`Error::Canceled`] on a mid-turn cancel.
    fn step(
        &mut self,
        turn: &ConversationTurn,
        on_chunk: &mut dyn FnMut(AudioChunk),
        on_progress: &mut dyn FnMut(Progress),
    ) -> Result<AudioTrack>;

    /// Explicitly close the session, releasing any held state. Idempotent; the default is a no-op
    /// (state is also released on drop). A provider overrides this only if closing can surface an
    /// error worth propagating.
    fn finish(&mut self) -> Result<()> {
        Ok(())
    }
}
