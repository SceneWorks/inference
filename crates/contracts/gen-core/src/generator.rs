//! The `Generator` contract — prompt-conditioned synthesis of image, video, **or** audio
//! (or a mix), including multi-modal models. See `docs/MODEL_ARCHITECTURE.md` §3.1.
//!
//! One trait covers everything text→media: T2I, T2V, edit (image+text→image), LTX
//! (text→video+audio), and pure audio synthesis (TTS / music). Modality is a
//! [`ModelDescriptor`] property plus a [`GenerationOutput`] variant — *not* a per-modality
//! trait split (which breaks on multi-modal models).

use crate::approximation::{ApproximationPlan, ApproximationRequest, ApproximationSurface};
use crate::execution_domains::{CfgBatching, ExecutionSurface, FfnChunk, GraphEvalCadence};
use crate::media::{AudioChunk, AudioTrack, Image};
use crate::runtime::{CancelFlag, PreviewSink, Progress, PromptEnhancementSink, Quant};
use crate::voice_embed::VoiceEmbedding;
use crate::{
    default_memory_strategy_safety_check, Error, MemoryPeakBreakdown, MemoryPhase,
    MemoryProviderContract, MemoryRequestScope, MemoryRunContext, MemorySafetyDecision,
    MemoryStrategy, MemoryStructuralResidentEvidence, Result,
};

/// A prompt-conditioned media generator. `generate` is **synchronous** (long/blocking; the
/// worker runs each job on its own thread); the request carries a cancel flag and
/// `on_progress` streams step/decode progress.
pub trait Generator {
    /// Identity + capabilities + modality (drives `validate` and consumer UI introspection).
    fn descriptor(&self) -> &ModelDescriptor;

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

fn requires_memory_request_scope(contract: &MemoryProviderContract) -> bool {
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

/// What a [`Generator`] produced. The `Video` variant's `audio` is `Some` for LTX (always
/// audio) and `None` for Wan — no contract change needed across the two. The `Audio` variant
/// is a **pure** audio synthesis (TTS / music — `Modality::Audio`); audio attached to a video
/// stays on the `Video` variant.
#[derive(Clone, Debug)]
pub enum GenerationOutput {
    Images(Vec<Image>),
    Video {
        frames: Vec<Image>,
        fps: u32,
        audio: Option<AudioTrack>,
    },
    Audio(AudioTrack),
}

/// The request union (lifted from the SceneWorks worker's `ImageRequest`/`VideoRequest`). Most
/// fields are optional; a model reads what it supports and `validate()` rejects the rest. A
/// single `Default`-able struct (no builder): `GenerationRequest { prompt, ..Default::default() }`.
#[derive(Clone, Debug)]
pub struct GenerationRequest {
    // --- Core ---
    pub prompt: String,
    pub negative_prompt: Option<String>,
    pub width: u32,
    pub height: u32,
    /// Number of images to produce (1..=8 for image models).
    pub count: u32,

    // --- Sampling (all optional; model/descriptor supply defaults) ---
    pub seed: Option<u64>,
    pub steps: Option<u32>,
    pub guidance: Option<f32>,
    pub true_cfg: Option<f32>,
    /// CFG-scheduling start step — the companion to [`true_cfg`](Self::true_cfg): real classifier-free
    /// guidance (and any per-branch conditioning gated with it) engages only once the denoise reaches
    /// this step, leaving earlier steps single-forward. `None` ⇒ each model's own default. Today only
    /// PuLID-FLUX honors it (default 1; its photoreal preset uses 4 to delay CFG a few steps); models
    /// without CFG scheduling ignore it.
    pub timestep_to_start_cfg: Option<u32>,
    pub sampler: Option<String>,
    pub scheduler: Option<String>,
    pub scheduler_shift: Option<f32>,

    /// Guidance method — how the conditional and unconditional model predictions are combined (epic
    /// 7434, the fourth orthogonal sampling layer). `"cfg"` (plain) | `"cfg_rescale"` (Lin et al.
    /// per-token norm-rescale) | `"apg"` (adaptive-projected guidance) | `"cfg_pp"` (CFG++ — renoise
    /// from the unconditional branch). `None` ⇒ the engine's default guidance path (the N1 no-op).
    /// Gated per-model-per-backend by [`Capabilities::supported_guidance_methods`]; an unadvertised
    /// value is rejected here at the contract boundary, and dropped-to-default with a worker event by
    /// the N3 fallback layer (P5).
    pub guidance_method: Option<String>,
    /// APG projection mix η (`apg` only): recombine as `orthogonal + η·parallel` against the
    /// conditional base. `η = 1` with no momentum and `norm_threshold = 0` reduces APG to plain CFG.
    /// `None` ⇒ the engine default. Ignored by non-APG methods.
    pub guidance_eta: Option<f32>,
    /// APG momentum (`apg` only): `running = diff + momentum·running`, the buffer persisting across
    /// denoise steps (0 ⇒ no momentum). `None` ⇒ the engine default. Ignored by non-APG methods.
    pub guidance_momentum: Option<f32>,
    /// APG norm-threshold (`apg` only): clamp the guidance delta to `‖diff‖ ≤ norm_threshold`
    /// (`0` disables the clamp). `None` ⇒ the engine default. Ignored by non-APG methods.
    pub guidance_norm_threshold: Option<f32>,

    // --- Conditioning ---
    pub conditioning: Vec<Conditioning>,
    /// img2img strength when a single `Reference` is supplied without its own strength.
    pub strength: Option<f32>,
    /// Wan-VACE control strength — the diffusers `conditioning_scale` / per-vace-layer
    /// `control_hidden_states_scale` (`hidden += proj_out(control)·scale`), broadcast to every
    /// `vace_layers` entry. `None` ⇒ the diffusers default `1.0`. Only the `wan_vace` model reads it;
    /// other models ignore it. (sc-3441)
    pub control_scale: Option<f32>,
    /// Krea "text style" gain — reweights the 12 stacked Qwen3-VL select-layer taps before the DiT's
    /// `TextFusionTransformer` aggregates them (the ComfyUI-Conditioning-Rebalance mechanism, sc-8596/
    /// sc-11878). A single scalar `g` maps to the per-layer ramp `w[i] = g + (2−2g)·i/(n−1)`: `g = 1`
    /// (or `None`) is a byte-exact no-op, `g > 1` emphasizes the early (low-level) taps for a
    /// warmer/richer/moodier look, `g < 1` biases toward the late (semantic) taps. GPU-validated safe
    /// over `[0.25, 1.75]` (the engine clamps to that range). **Krea / Qwen-Image-family only** (depends
    /// on the multi-tap text encoder); other models ignore it. It does NOT transfer subject/identity —
    /// it is a stylistic nudge, distinct from the reference-image [`strength`](Self::strength) lever.
    pub text_style_gain: Option<f32>,
    /// Image-guidance (true CFG on the **reference/image** condition) for reference-conditioned edit
    /// models — the identity-strength lever (sc-8273/sc-8278). When `Some(s)` with `s > 1`, the
    /// denoise extrapolates the with-reference velocity against the reference-dropped
    /// (image-unconditional) velocity: `v = v_img0 + s·(v_ref − v_img0)`, pulling output toward the
    /// reference identity *without* pinning composition. `None`/`≤1` ⇒ off (the shipped behavior;
    /// the reference is plain edit conditioning). Today only the FLUX.2 klein/dev **edit** path reads
    /// it (non-kv); other models ignore it. The `FLUX2_IMG_GUIDANCE` env var overrides this (debug).
    pub image_guidance: Option<f32>,

    // --- Video (Option; consumed by video models at the follow-on port) ---
    pub frames: Option<u32>,
    pub fps: Option<u32>,
    pub duration: Option<f32>,
    pub video_mode: Option<String>,
    /// Generate this many extra leading temporal chunks (each = `vae_stride_t` latent frames) and
    /// discard them after decode, so the first *kept* frame has a full temporal receptive field of
    /// real (non-zero-padded) data — mitigates first-frame VAE/causal-conv artifacts. `None`/0 = off
    /// (the default). Consumed by Wan video models (`generate_wan.py`'s `trim_first_frames`); video
    /// models that don't support it ignore it.
    pub trim_first_frames: Option<u32>,

    // --- LTX-2.5 DFR (sc-18789; the `--num-generated-keyframes` / `--temporal-upsample-rounds`
    //     CLI equivalents) ---
    /// Number of extra **generated keyframe slots** placed at evenly spaced interior frame
    /// positions (reference `--num-generated-keyframes`; `ltx_dfr::evenly_spaced_keyframe_positions`).
    /// Each slot relaxes the effective temporal compression at its position at the cost of one
    /// latent frame's worth of tokens for one pixel frame. Requires a checkpoint whose transformer
    /// sets `use_keyframes_abs_pos_embedding` (LTX ≥ 2.5) — an engine whose checkpoint lacks the
    /// learned marker (`ltx_2_3`) **refuses** the request with a typed `Unsupported` rather than
    /// denoising unmarked slots as wasted compute (the reference validates the same way, up
    /// front). `None`/0 = off. Non-LTX models ignore it.
    pub num_generated_keyframes: Option<u32>,
    /// Number of DFR temporal ×2 refine rounds, `0..=2` (reference `--temporal-upsample-rounds`:
    /// each round doubles the frame rate, splits the canvas into `2^round` keyframe-seam tiles and
    /// re-denoises them ancestrally). Requires the temporal latent upsampler component and a
    /// generated-keyframe-capable checkpoint (LTX ≥ 2.5); `ltx_2_3` refuses a non-zero value with
    /// a typed `Unsupported`. `None`/0 = off. Non-LTX models ignore it.
    pub temporal_upsample_rounds: Option<u32>,

    // --- SVD image→video micro-conditioning (sc-3523; ignored by other models) ---
    /// SVD `motion_bucket_id` — the motion-strength bucket baked into the `added_time_ids`
    /// micro-conditioning (higher = more motion). `None` ⇒ the model default (127). Only the
    /// `svd_xt` model reads it; other models ignore it.
    pub motion_bucket_id: Option<f32>,
    /// SVD `noise_aug_strength` — Gaussian noise added to the VAE-encoded conditioning image (and
    /// surfaced in `added_time_ids`); higher = less fidelity to the source / more motion. `None` ⇒
    /// the model default (0.02). Only `svd_xt` reads it.
    pub noise_aug_strength: Option<f32>,
    /// Frames decoded per temporal-VAE pass (diffusers `decode_chunk_size`) — a memory/quality knob
    /// for chunked video VAE decode (smaller = less peak memory, changes temporal-boundary
    /// behavior). `None` ⇒ the model default. Only `svd_xt` reads it today.
    pub decode_chunk_size: Option<u32>,
    /// SVD motion **conditioning** fps — the cadence the model was trained on, baked into the
    /// `added_time_ids` micro-conditioning (`fps − 1`); lower ⇒ smoother/slower motion. This is
    /// distinct from [`fps`](Self::fps), which is the output/playback cadence used when muxing the
    /// clip: SVD decouples them (diffusers `StableVideoDiffusionPipeline(fps=…)` vs
    /// `export_to_video(fps=…)`). `None` ⇒ the model default (7). Only `svd_xt` reads it (sc-3764).
    pub conditioning_fps: Option<u32>,

    // --- SeedVR2 super-resolution (sc-4816; ignored by other models) ---
    /// SeedVR2 input **softness** — a pre-blur applied to the bicubic-upscaled low-resolution input
    /// before VAE encode (reference `SeedVR2.generate_image(softness=…)`). Higher = more smoothing of
    /// source compression/noise artifacts before the one-step restoration (trades fine detail for
    /// fewer amplified artifacts on degraded footage). `None`/0.0 ⇒ no pre-blur (the reference
    /// default). Only the `seedvr2` upscaler reads it; other models ignore it.
    pub softness: Option<f32>,

    // --- Prompt enhancement (LTX-2.3 sc-2845 + FLUX.2-dev caption upsampling sc-6030; ignored by
    //     other models) ---
    /// Rewrite `prompt` with an autoregressive LLM before encoding. Default `false` — the diffusion
    /// path is unchanged. On any enhancer failure the model falls back to the original prompt
    /// (reference-faithful). Consumed by: LTX-2.3 (the Gemma-3 `--enhance-prompt`) and FLUX.2-**dev**
    /// (the Mistral3 multimodal `upsample_prompt`, sc-6030 — text-only for T2I, image-conditioned on
    /// the request's reference images for edit; gated like the reference `caption_upsample_temperature`).
    pub enhance_prompt: bool,
    /// Use the separate uncensored 4-bit Gemma enhancer (`--use-uncensored-enhancer`) instead of the
    /// loaded text-encoder backbone. Only consulted when `enhance_prompt` is set. LTX-2.3 only;
    /// FLUX.2-dev ignores it (its upsampler is the loaded Mistral3 tower).
    pub use_uncensored_enhancer: bool,
    /// Max tokens for prompt enhancement (LTX default 512, FLUX.2-dev caption-upsample default 512,
    /// each model's own default when `None`).
    pub enhance_max_tokens: Option<u32>,
    /// Sampling temperature for prompt enhancement (model default when `None`: LTX 0.7, FLUX.2-dev
    /// caption-upsample 0.15 — the reference `caption_upsample_temperature`).
    pub enhance_temperature: Option<f32>,
    /// Optional consumer sink for the engine-owned effective-prompt fact. Supporting providers emit
    /// exactly one enhanced/fallback/absent report before encoding their effective prompt.
    pub prompt_enhancement: PromptEnhancementSink,

    // --- Decoder (epic 7840; ignored by models without a PiD backbone) ---
    /// Route this generation's decode through the optional **PiD** super-resolving decoder instead of
    /// the native VAE. Default `false` — the VAE-decode path is unchanged. Only honored when the model
    /// was loaded with [`LoadSpec::pid`](crate::LoadSpec::pid) weights (the PiD-eligible providers,
    /// Qwen-Image / Krea today — sc-7845); a provider with no PiD loaded errors rather than silently
    /// ignoring the request, and PiD-ineligible models ignore the flag. Turning PiD on also changes the
    /// output resolution (native → 4×), so it is not a transparent decoder swap. PiD output is
    /// research/evaluation-only (NSCLv1), surfaced/labeled at the worker/web layer (Phase 3).
    pub use_pid: bool,
    /// PiD **`from_ldm` early-stop** capture σ (epic 7840, sc-7993). Only consulted when
    /// [`use_pid`](Self::use_pid) is set. When `Some(σ)`, stop the denoise as soon as the schedule's
    /// noise level first drops to `≤ σ`, then hand that *partially-denoised* `x_k` to PiD with the
    /// **achieved** degrade σ (`= sigmas[k]`) — the speed optimization that lets the (expensive)
    /// backbone denoise exit early and the 4-step pixel decoder finish the rest. `None`/`≤0` (the
    /// default) = the clean σ=0 path (full denoise, then decode the clean latent). The value is a
    /// noise *ceiling*, schedule-agnostic, so the same σ maps to the right step on an 8-step Turbo and
    /// a 50-step trajectory alike (the policy is [`crate::sampling::flow_capture_plan`]).
    ///
    /// **Frame:** σ is interpreted in the **flow-matching** frame `x_t = (1−σ)x0 + σε` — the path wired
    /// today is the qwenimage latent space (Qwen-Image / Krea / Lightning-Qwen). A latent space whose
    /// PiD student is variance-preserving (SDXL) or whose `from_ldm` wiring is a follow-on errors rather
    /// than silently ignoring the request (see `mlx_gen_pid::resolve_pid_decoder`).
    pub pid_capture_sigma: Option<f32>,

    // --- Memory adaptation (provider-specific, quality-preserving levers) ---
    /// Optional quality-preserving memory adaptations selected by the consumer from live device
    /// budget plus measured, stage-specific peaks. `None` is the historical fast path. Providers
    /// ignore levers they do not advertise or implement.
    ///
    /// These are execution choices, not creative settings: they must preserve the requested
    /// precision, dimensions, seed, and sampling recipe. SceneWorks currently uses the complete
    /// surface for ordinary Krea 2 Turbo on constrained CUDA cards.
    pub memory: Option<GenerationMemory>,

    // --- Approximate capabilities (sc-18322, epic 18304 P7) ---
    /// Optional **result-changing** cost reductions — the deliberate opposite of
    /// [`memory`](Self::memory) and of the typed execution domains, both of which promise
    /// equivalence. `None` (the `Default`) is the provider's exact path, byte-for-byte the
    /// pre-sc-18322 render.
    ///
    /// This is a separate sub-block rather than more [`GenerationMemory`] fields precisely because
    /// `GenerationMemory` is documented as *quality-preserving*: a lever that changes the output has
    /// no business sharing a struct whose whole contract is that it does not. It also keeps
    /// `GenerationMemory` `Copy`, which a characterization reference (an owned string pair) would
    /// break.
    ///
    /// Every selection here is gated at the shared request floor against
    /// [`Capabilities::approximation`], and — until epic 18304's terminal measurement campaign
    /// defines a quality-characterization artifact — **every** selection is refused, because an
    /// approximate mechanism may only be selected alongside a characterization of what it costs in
    /// quality. See [`crate::approximation`].
    pub approximation: Option<ApproximationRequest>,

    // --- Audio (Option; consumed by audio models — `Modality::Audio`) ---
    /// The typed audio sub-block (sc-12834). `None` for every image/video request — the top-level
    /// request stays un-bloated, mirroring the planned typed video guider block (§9 known additive
    /// extensions). Audio models read what they support; the shared floor gates the values against
    /// the [`Capabilities`] audio surface. See [`AudioParams`].
    pub audio: Option<AudioParams>,

    // --- Multi-phase denoise (epic 13879, sc-13884; consumed by Krea MLX today) ---
    /// An ordered list of denoise **phases** run within ONE trajectory over ONE coherent global
    /// sigma schedule (sc-13884). Each [`GenerationPhase`] owns a contiguous slice of the shared
    /// schedule (its [`steps`](GenerationPhase::steps)) plus its own guidance and active adapter
    /// stack, so a request can e.g. run *N* steps of Raw with true-CFG on, then *M* steps of
    /// Raw+turbo-LoRA with CFG off, all sharing the latent and sigma trajectory across the boundary
    /// (no per-phase schedule reset). The total step budget is the **sum** of the phases' steps —
    /// the flat [`steps`](Self::steps) is ignored when `phases` is present.
    ///
    /// **Additive and single-phase-preserving.** `None` (the default) is the ordinary single-phase
    /// render, byte-for-byte unaffected: a model with no multi-phase support behaves exactly as
    /// before sc-13884, and a model that reads `phases` falls back to its single-phase path when this
    /// is `None`. Only the Krea MLX family reads it today; other models ignore it. Per-phase
    /// *scheduler* selection is a deliberate follow-on — every phase shares the one global schedule.
    pub phases: Option<Vec<GenerationPhase>>,

    // --- Control ---
    pub cancel: CancelFlag,
    /// Per-step latent-preview sink: a supporting engine emits a small linear latent→RGB
    /// approximation of the developing image on each denoise evaluation, so a consumer UI can
    /// render the image developing instead of a bare progress bar. The inert
    /// [`PreviewSink::default`] costs a supporting engine one
    /// [`is_active`](PreviewSink::is_active) check per evaluation and nothing else — the
    /// projection is skipped entirely, so a request that does not ask for previews is
    /// byte-for-byte unaffected.
    ///
    /// A request **field** (the [`CancelFlag`] pattern), deliberately not a [`Progress`] variant:
    /// `Progress` stays `Copy` and no exhaustive match downstream changes. Support is per-engine
    /// and opt-in — an engine that never emits simply never calls it. Consumers distinguish an
    /// unsupported engine from a supporting engine that has not emitted yet through
    /// [`Capabilities::supports_preview`]; the sink callback alone cannot distinguish those states.
    /// See [`PreviewSink`] for the frame contract.
    pub preview: PreviewSink,
}

/// Quality-preserving execution levers for a single generation.
///
/// The consumer selects these in increasing cost order from measured stage peaks. Keeping them on
/// the per-generation request (rather than [`LoadSpec`](crate::LoadSpec)) lets one cached provider
/// serve different resolutions and live budgets truthfully without a process-global toggle.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct GenerationMemory {
    /// Release whole model components between conditioning, denoise, and decode for this request.
    ///
    /// Unlike [`LoadSpec::offload_policy`](crate::LoadSpec::offload_policy), this is an execution
    /// decision on one generation. A cached generator may therefore keep a warm component pair for
    /// one request, stage the next request, and rebuild the warm pair for a later request.
    pub stage_residency: bool,
    /// Force the provider's bounded/tiled native VAE decode even below its automatic tiling threshold.
    pub tile_vae_decode: bool,
    /// Bound attention scratch by chunking independent query rows. This must preserve the provider's
    /// existing byte-identity contract.
    pub chunk_attention: bool,
    /// Keep transformer trunk blocks in host-backed storage and materialize only the current block on
    /// the accelerator. This is the last quality-preserving rung when complete weights plus bounded
    /// activations still exceed the live device budget.
    pub stream_transformer_blocks: bool,

    // --- Selected strategy parameters (SC-15510) -------------------------------------------------
    //
    // The three booleans above say WHICH rungs run; these say with WHAT. Before SC-15510 a provider
    // could only advertise the single hardcoded value its pipeline happened to use, which made
    // `MemoryParameterRanges` a one-element list and left SC-15508's "a single-point pass cannot
    // mark untested production parameters Verified" unsatisfiable. Each is `Option`, and `None` (the
    // `Default`) means "the provider's own historical constant", so every provider that does not read
    // them — and every request that does not set them — is byte-for-byte unaffected.
    //
    // The values a provider will accept are exactly the candidates it publishes in its
    // `MemoryProviderContract`; its request scope re-validates them, so an out-of-domain value is
    // a typed rejection rather than a silently different execution than the selector chose.
    /// Decode tile edge in **output pixels** for the bounded (tiled) decode. `None` ⇒ the provider's
    /// default tile edge.
    pub decode_tile_edge: Option<u32>,
    /// Feather/blend overlap in output pixels paired with [`Self::decode_tile_edge`]. `None` ⇒ the
    /// provider's default overlap.
    pub decode_overlap: Option<u32>,
    /// Maximum attention-score elements materialized by one bounded attention chunk. `None` uses the
    /// provider's historical chunk size. Meaningful only when [`Self::chunk_attention`] is enabled.
    pub attention_chunk_size: Option<u32>,
    /// Number of consecutive transformer trunk blocks held materialized at once when
    /// [`Self::stream_transformer_blocks`] is set. `None` ⇒ the provider's default window.
    pub transformer_window_size: Option<u32>,
    /// Which transformer(s) [`Self::stream_transformer_blocks`] applies to (SC-15794). `None` ⇒
    /// [`TransformerComponent::Dit`](crate::memory_strategy::TransformerComponent::Dit), the
    /// by-convention scope every provider had before the component scope existed — so an untouched
    /// request is byte-for-byte unaffected.
    ///
    /// The text-encoder scope exists because rung 4 cut the denoise far enough that **conditioning
    /// became the binding phase** on the larger tiers. Measured on z_image_turbo at 1024²
    /// (Apple/Metal, real weights, SC-15794): conditioning binds bf16 at 8.344 GiB against a 4.365
    /// GiB decode floor, and the encoder's weights are 7.440 GiB of that — so the window has real
    /// work to do there, while q4 is already decode-bound and gains nothing.
    pub transformer_window_component: Option<crate::memory_strategy::TransformerComponent>,

    // --- Typed execution domains (sc-18317, epic 18304 P2) --------------------------------------
    //
    // The parameters above say how the five-rung memory LADDER runs. These three say how one forward
    // pass is SCHEDULED, which is a different axis: none of them sheds or bounds a component's
    // residency, none has a place in the ladder's cost order, and none is engaged by a rung. Each
    // existed before this story as a per-provider ad hoc knob with no shared vocabulary, so a planner
    // could not discover it and a caller could not be told it had been ignored.
    //
    // Same `Option` convention as the SC-15510 parameters: `None` (the `Default`) is the provider's
    // own historical constant, so a request that sets none of them is byte-for-byte the pre-sc-18317
    // render on every provider. Unlike the ladder parameters these are NOT gated on a rung boolean —
    // they are independent of the ladder — so each is validated on its own against the provider's
    // declared `Capabilities::execution` surface, and a value a provider cannot honour is a typed
    // refusal at the shared floor rather than a silently different schedule.
    // See [`crate::execution_domains`] for the domains, the refusal semantics, and the per-domain
    // equivalence classes (cadence and CFG batching are bit-identical; FFN chunking is numerically
    // equivalent).
    /// Blocks per forced lazy-graph evaluation inside the denoise forward. `None` ⇒ the provider's
    /// own evaluation schedule (for every provider today, its historical whole-forward or
    /// per-block constant). See [`GraphEvalCadence`] — in particular that
    /// this is the *within-forward block* cadence and never the output-identity-bearing per-step
    /// latent evaluation.
    pub graph_eval_cadence: Option<GraphEvalCadence>,
    /// Sequence rows per chunk of a chunked FFN intermediate. `None` ⇒ the provider's whole-sequence
    /// FFN. The one domain here that is *numerically* equivalent rather than bit-identical
    /// ([`FfnChunk`]), so a selector that must not perturb pixels leaves it unset.
    pub ffn_chunk: Option<FfnChunk>,
    /// Whether classifier-free guidance runs as one doubled-batch forward or two batch-1 forwards.
    /// `None` ⇒ the provider's own convention (batched, for every CFG provider in this workspace).
    /// See [`CfgBatching`]: batched CFG doubles exactly the transients rungs 2-4
    /// exist to bound, which is why it is a planner-selectable axis.
    pub cfg_batching: Option<CfgBatching>,

    /// Calibration-only request-local fault injection. Adopting providers may return a deterministic
    /// error at the named physical phase boundary so a conformance harness can verify cleanup and a
    /// warm follow-up request. The shared request floor accepts this only when paired with
    /// [`Self::calibration_fault_harness_authorized`]. Production selectors must leave both controls
    /// at their defaults.
    #[doc(hidden)]
    pub calibration_error_phase: Option<MemoryPhase>,
    /// Explicit authorization paired with [`Self::calibration_error_phase`] by calibration harnesses.
    /// A phase without authorization, or authorization without a phase, is rejected at the shared
    /// request floor rather than reaching a provider-internal failure seam by convention alone.
    #[doc(hidden)]
    pub calibration_fault_harness_authorized: bool,
}

impl GenerationMemory {
    /// Authorize one deterministic phase failure for a calibration/conformance harness.
    #[doc(hidden)]
    pub fn authorize_calibration_fault(&mut self, phase: MemoryPhase) {
        self.calibration_error_phase = Some(phase);
        self.calibration_fault_harness_authorized = true;
    }
}

/// The typed audio request sub-block carried by [`GenerationRequest::audio`] (sc-12834). A single
/// `Default`-able struct (no builder), like the request itself: every field is optional so the
/// struct stays **additively extensible** — a later story adds e.g. a multi-speaker script field
/// without breaking `AudioParams { voice: Some(..), ..Default::default() }` construction.
///
/// A model reads what it supports and ignores the rest; the shared validation floor
/// ([`Capabilities::validate_request`] and its size-skipping siblings) rejects values outside the
/// advertised audio surface ([`Capabilities::audio_voices`] / [`audio_languages`](Capabilities::audio_languages)
/// / [`audio_sample_rates`](Capabilities::audio_sample_rates) /
/// [`max_audio_duration_secs`](Capabilities::max_audio_duration_secs)).
#[derive(Clone, Debug, Default, PartialEq)]
pub struct AudioParams {
    /// Voice / speaker id (TTS). Gated by [`Capabilities::audio_voices`] when supplied.
    pub voice: Option<String>,
    /// Language code (e.g. `"en"`). Gated by [`Capabilities::audio_languages`] when supplied.
    pub language: Option<String>,
    /// Requested output duration in seconds. Range-checked against
    /// [`Capabilities::max_audio_duration_secs`] (and the shared duration sanity cap).
    pub target_duration: Option<f32>,
    /// Requested output sample rate in Hz. Gated by [`Capabilities::audio_sample_rates`] when
    /// supplied; `None` ⇒ the model's native rate.
    pub sample_rate: Option<u32>,
    /// Musical tempo in beats per minute (music models). Must be finite and positive.
    pub bpm: Option<f32>,
    /// Musical key (e.g. `"C minor"`; music models). Free-form — each model documents what it
    /// accepts and rejects the rest in its own `validate`.
    pub musical_key: Option<String>,
    /// Lyrics to sing / condition on (music models). Free-form text, distinct from `prompt`.
    pub lyrics: Option<String>,
    /// A **multi-speaker dialogue script** (sc-12848) — an ordered sequence of spoken
    /// [`SpeechSegment`]s, each carrying its own text plus an optional speaker/voice assignment and
    /// per-segment style. This is the long-form / conversational-TTS carrier: a narration or a
    /// two-person dialogue is one request whose segments are rendered in their assigned voices into a
    /// single [`AudioTrack`], rather than a single voice reading everything.
    ///
    /// **Additive and single-voice-preserving.** `None` (the default) is a plain single-voice
    /// request, byte-for-byte unaffected by this field: a provider with no script support behaves
    /// exactly as before sc-12848. A provider opts in through
    /// [`Capabilities::supports_multi_speaker`] (and optionally advertises a
    /// [`Capabilities::max_speakers`] cap); the shared floor rejects a script sent to a
    /// non-multi-speaker model as the typed [`Error::Unsupported`], the same convention
    /// [`supports_streaming`](Capabilities::supports_streaming) uses. Tensor-free, like the rest of
    /// [`AudioParams`]. `prompt` still carries any single-voice / global text; a model that reads the
    /// script renders it in preference to `prompt`.
    pub script: Option<Vec<SpeechSegment>>,
}

/// One segment of a multi-speaker dialogue [`script`](AudioParams::script) (sc-12848) — the text a
/// single speaker utters, plus which voice utters it. Tensor-free and additive: new per-segment
/// controls arrive as further `Option` fields without breaking
/// `SpeechSegment { text: .., ..Default::default() }` construction.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct SpeechSegment {
    /// The text this segment speaks. A segment with empty/whitespace-only text is a malformed
    /// request each script-capable model rejects in its own `validate`.
    pub text: String,
    /// The speaker / voice this segment is spoken in — a dialogue label (`"S1"` / `"S2"`) or a
    /// concrete voice id, at the model's discretion. Gated against
    /// [`Capabilities::audio_voices`] exactly like [`AudioParams::voice`] **only when the model
    /// advertises a closed voice surface** (a non-empty `audio_voices`); a dialogue model with
    /// opaque speaker labels advertises an empty voice surface and maps the labels itself. `None`
    /// ⇒ the model's default / first speaker.
    pub speaker: Option<String>,
    /// Optional free-form per-segment style / emotion hint (e.g. `"cheerful"`, `"whisper"`).
    /// Advisory and not gated: each model documents what it honors and ignores the rest.
    pub style: Option<String>,
}

/// Who speaks a [`ConversationTurn`] in a multi-turn conversation (sc-14150). The distinction is
/// semantic to a voice-agent TTS model: a [`User`](Self::User) turn is *provided* context (the
/// user's speech), an [`Assistant`](Self::Assistant) turn is the model's synthesized reply — and only
/// an assistant turn is generated (a turn with `audio: None`).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ConversationRole {
    /// A user turn — provided context (the user's speech + its transcript). Always carries `audio`.
    User,
    /// An assistant turn — the model's spoken reply. `audio: None` marks it as the turn to
    /// **synthesize**; `audio: Some` is a previously-generated assistant turn resumed as context.
    Assistant,
}

/// One turn of a multi-turn conversation (sc-14150) — the unit both the stateless
/// [`Conditioning::ConversationHistory`] carrier (path **A**) and the stateful
/// [`ConversationSession`] (path **B**) consume. Tensor-free: audio crosses as a PCM
/// [`AudioTrack`], never model tokens — the provider encodes it to its own codec representation.
///
/// A turn carries a [`role`](Self::role), the turn's `text`, and its [`audio`](Self::audio):
/// - `audio: Some(track)` ⇒ a **context** turn (the user's speech, or a prior assistant turn
///   resumed from elsewhere); the model conditions on it.
/// - `audio: None` ⇒ a **synthesis** turn — the assistant reply to generate (`role` must be
///   [`ConversationRole::Assistant`]); the model synthesizes `text` in the conversation's voice,
///   conditioned on all prior turns.
///
/// Additive and single-turn-preserving, like the rest of the request surface: a provider with no
/// multi-turn support is byte-for-byte unaffected (it advertises neither
/// [`Capabilities::supports_conversation_history`] nor
/// [`Capabilities::supports_conversation_session`], and the shared floor rejects a conversation as
/// the typed [`Error::Unsupported`]). New per-turn controls arrive as further `Option` fields
/// without breaking `ConversationTurn { role, text, ..Default::default() }` construction.
#[derive(Clone, Debug, PartialEq)]
pub struct ConversationTurn {
    /// Who speaks this turn.
    pub role: ConversationRole,
    /// The turn's text — the user's transcript (context turn) or the assistant text to speak
    /// (synthesis turn). A turn with empty/whitespace-only text is a malformed request each
    /// multi-turn model rejects in its own `validate`.
    pub text: String,
    /// The turn's PCM audio: `Some` for a context turn, `None` for the assistant reply to
    /// synthesize. See the type docs.
    pub audio: Option<AudioTrack>,
}

impl Default for ConversationTurn {
    fn default() -> Self {
        Self {
            role: ConversationRole::Assistant,
            text: String::new(),
            audio: None,
        }
    }
}

/// One phase of a [multi-phase denoise](GenerationRequest::phases) (epic 13879, sc-13884): a
/// contiguous slice of the ONE shared global sigma schedule, run with this phase's own guidance and
/// active adapter stack. The latent flows continuously from the previous phase — a phase does **not**
/// restart the schedule, it resumes at the sigma the prior phase reached (that shared boundary is the
/// whole point: one coherent trajectory, no seam/reset artifact).
///
/// Tensor-free and additive, like the rest of the request: new per-phase controls (e.g. a per-phase
/// scheduler, the deliberate follow-on) arrive as further `Option` fields without breaking
/// `GenerationPhase { steps, ..Default::default() }` construction.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct GenerationPhase {
    /// The number of contiguous denoise steps this phase runs, as a slice of the shared global
    /// schedule. The sum of every phase's `steps` is the request's total step budget (the flat
    /// [`GenerationRequest::steps`] is ignored when `phases` is present). A phase with `steps == 0`
    /// is a malformed request each multi-phase model rejects in its own `validate`.
    pub steps: u32,
    /// This phase's guidance. `Some(g)` with `g > 0` runs the true classifier-free-guidance path
    /// (two model forwards per step: conditional + unconditional, combined by the model's CFG rule);
    /// `Some(0.0)` runs the single-forward CFG-off path. `None` inherits the request/model default
    /// guidance. This is what lets the "N steps CFG-on, then M steps CFG-off" split vary freely. Joins
    /// the request finiteness floor.
    pub guidance: Option<f32>,
    /// The adapters active during this phase, referencing the load-time adapter stack
    /// ([`crate::LoadSpec::adapters`], in load order) by index. An **empty** vector means this phase
    /// runs the bare base model (no adapters) — the common phase-1 case of the Raw→Raw+turbo-LoRA
    /// workflow. A phase that names an adapter index out of range of the loaded stack is a malformed
    /// request the model rejects in its own `validate`.
    pub adapters: Vec<PhaseAdapter>,
}

/// One adapter activated by a [`GenerationPhase`] (sc-13884): which load-time adapter it enables and,
/// optionally, at what per-phase weight. The adapters are provisioned ONCE at model-load time (via
/// [`crate::LoadSpec::adapters`]); a phase selects which of them are active and at what weight — so a
/// two-phase job can run base-only, then base+adapter, without reloading the model.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct PhaseAdapter {
    /// Index into the load-time adapter stack ([`crate::LoadSpec::adapters`], in the order the loader
    /// received them) this phase activates. Referencing by index keeps the request contract
    /// tensor-neutral and free of load paths — the consumer that provisioned the adapters knows their
    /// order. An out-of-range index is rejected when the model resolves the phase list at generate.
    pub adapter: usize,
    /// Per-phase weight override for this adapter. `None` uses the adapter's load-time
    /// [`scale`](crate::AdapterSpec::scale); `Some(w)` scales its contribution to `w` for this phase
    /// only (e.g. ramping a turbo LoRA in over the later phases). Joins the request finiteness floor.
    pub weight: Option<f32>,
}

impl Default for GenerationRequest {
    fn default() -> Self {
        Self {
            prompt: String::new(),
            negative_prompt: None,
            width: 1024,
            height: 1024,
            count: 1,
            seed: None,
            steps: None,
            guidance: None,
            true_cfg: None,
            timestep_to_start_cfg: None,
            sampler: None,
            scheduler: None,
            scheduler_shift: None,
            guidance_method: None,
            guidance_eta: None,
            guidance_momentum: None,
            guidance_norm_threshold: None,
            conditioning: Vec::new(),
            strength: None,
            control_scale: None,
            text_style_gain: None,
            image_guidance: None,
            frames: None,
            fps: None,
            duration: None,
            video_mode: None,
            trim_first_frames: None,
            num_generated_keyframes: None,
            temporal_upsample_rounds: None,
            motion_bucket_id: None,
            noise_aug_strength: None,
            decode_chunk_size: None,
            conditioning_fps: None,
            softness: None,
            enhance_prompt: false,
            use_uncensored_enhancer: false,
            enhance_max_tokens: None,
            enhance_temperature: None,
            prompt_enhancement: PromptEnhancementSink::default(),
            use_pid: false,
            pid_capture_sigma: None,
            memory: None,
            approximation: None,
            audio: None,
            phases: None,
            cancel: CancelFlag::default(),
            preview: PreviewSink::default(),
        }
    }
}

/// A first_last_frame / multi-keyframe input — a borrowed, normalized view of a
/// [`Conditioning::Keyframe`]. Returned by [`GenerationRequest::keyframes`].
#[derive(Clone, Copy, Debug)]
pub struct KeyframeRef<'a> {
    pub image: &'a Image,
    pub frame_idx: i32,
    pub strength: f32,
}

/// An in-context conditioning clip — a borrowed view of a [`Conditioning::VideoClip`]. Returned by
/// [`GenerationRequest::video_clips`].
#[derive(Clone, Copy, Debug)]
pub struct VideoClipRef<'a> {
    pub frames: &'a [Image],
    pub frame_idx: i32,
    pub strength: f32,
}

/// Which edit operation a [`Conditioning::AudioEdit`] requests of a prompted audio editor (sc-12847),
/// mapped by the provider onto ACE-Step 1.5's native audio-to-audio task modes.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum AudioEditMode {
    /// Regenerate a bounded interior span fresh from the prompt while keeping the rest of the clip
    /// (ACE-Step `repaint`, the edit window silence-substituted so the model fills it anew).
    /// Requires a [`TimeRegion`].
    Inpaint,
    /// Regenerate a bounded span, the model conditioning on the surrounding source for continuity
    /// (ACE-Step `repaint`). Requires a [`TimeRegion`]. Distinct from [`Inpaint`](Self::Inpaint) at
    /// the contract level; the shared ACE-Step machinery differs only in whether the window is
    /// seeded from silence.
    Repaint,
    /// Continue the clip past its end: the appended tail is generated from the prompt while the
    /// original audio is preserved (ACE-Step `repaint` with the generate window at the tail). The
    /// [`TimeRegion`]'s `start_secs` is where generation begins (defaults to the source length) and
    /// `end_secs` names the new total length.
    Extend,
    /// Restyle the whole clip from a new prompt (ACE-Step `cover`). Whole-clip; any [`TimeRegion`]
    /// is ignored.
    Cover,
}

/// A half-open time span `[start_secs, end_secs)` in seconds — the edit region of a
/// [`Conditioning::AudioEdit`] (sc-12847). `end_secs = None` means "to the end of the clip" (and for
/// [`AudioEditMode::Extend`] names the new total length). Both bounds join the finiteness floor.
///
/// Expressed in **seconds** (not latent frames) so the contract stays VAE-stride-agnostic: the
/// provider converts to latent-frame indices via its own `latents_per_second`. This is the audio
/// analogue of the image lane's masked-region conditioning (the pixel [`Conditioning::Mask`] / the
/// video [`Conditioning::ControlClip`]'s `start_frame`).
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct TimeRegion {
    /// Region start (seconds from the clip start). Must be finite and `>= 0`.
    pub start_secs: f32,
    /// Region end (seconds). `None` ⇒ the end of the clip. When present must be finite and
    /// `> start_secs`.
    pub end_secs: Option<f32>,
}

/// A prompted source-audio edit — a borrowed, normalized view of a [`Conditioning::AudioEdit`].
/// Returned by [`GenerationRequest::audio_edit`] (sc-12847).
#[derive(Clone, Copy, Debug)]
pub struct AudioEditRef<'a> {
    pub audio: &'a AudioTrack,
    pub mode: AudioEditMode,
    pub region: Option<TimeRegion>,
    pub strength: Option<f32>,
}

/// A **multi-region** prompted source-audio edit — a borrowed, normalized view of a
/// [`Conditioning::AudioEditRegions`]. Returned by [`GenerationRequest::audio_edit_regions`]
/// (sc-14549).
///
/// Deliberately a separate accessor from [`AudioEditRef`] rather than a widened one: the two
/// carriers are different [`ConditioningKind`]s, so a provider that serves only the single-region
/// shape keeps refusing multi-region through the shared allowlist without writing any code.
#[derive(Clone, Copy, Debug)]
pub struct AudioEditRegionsRef<'a> {
    pub audio: &'a AudioTrack,
    pub mode: AudioEditMode,
    /// One or more spans to regenerate in a **single** pass. Non-empty; every `end_secs` is
    /// `Some` (see [`Conditioning::AudioEditRegions`] for why). Caller order is not significant.
    pub regions: &'a [TimeRegion],
    pub strength: Option<f32>,
}

/// A replace_person masked control clip — a borrowed view of a [`Conditioning::ControlClip`].
/// Returned by [`GenerationRequest::control_clip`].
#[derive(Clone, Copy, Debug)]
pub struct ControlClipRef<'a> {
    pub frames: &'a [Image],
    pub mask: &'a [Image],
    pub masking_strength: f32,
    pub start_frame: i32,
    pub mode: ReplacementMode,
}

/// Exact SCAIL-2 public animation/replacement carrier: one character image, its paired color mask, and one
/// driving clip with a one-to-one frame/mask sequence. The three physical conditioning entries are
/// one conceptual character reference for memory-evidence geometry.
#[derive(Clone, Copy, Debug)]
pub struct Scail2AnimationConditioningRef<'a> {
    pub character: &'a Image,
    pub character_mask: &'a Image,
    pub driving_frames: &'a [Image],
    pub driving_masks: &'a [Image],
}

impl Scail2AnimationConditioningRef<'_> {
    pub const fn reference_count(&self) -> u32 {
        1
    }

    pub fn identity_shape(&self, mode: &str) -> crate::Result<String> {
        let control = self
            .driving_frames
            .first()
            .ok_or_else(|| Error::Unsupported("scail2 carrier has no driving frames".to_owned()))?;
        Ok(format!(
            "{mode}:reference:{}x{}:control:{}x{}x{}",
            self.character.width,
            self.character.height,
            control.width,
            control.height,
            self.driving_frames.len()
        ))
    }
}

impl GenerationRequest {
    /// Number of image-conditioning inputs represented by this request for memory-evidence
    /// geometry. Multi-image carriers contribute their flattened image count; control/depth/mask
    /// carriers each contribute one. A keyframe is a distinct image input even though its placement
    /// is temporal; clips remain represented by the frame axis.
    pub fn image_reference_count(&self) -> u32 {
        self.conditioning.iter().fold(0_u32, |count, conditioning| {
            let increment = match conditioning {
                Conditioning::Reference { .. }
                | Conditioning::Keyframe { .. }
                | Conditioning::Control { .. }
                | Conditioning::Depth { .. }
                | Conditioning::Mask { .. } => 1,
                Conditioning::MultiReference { images } => {
                    u32::try_from(images.len()).unwrap_or(u32::MAX)
                }
                Conditioning::ReduxRefs { refs } => u32::try_from(refs.len()).unwrap_or(u32::MAX),
                _ => 0,
            };
            count.saturating_add(increment)
        })
    }

    /// Conceptual reference count used by memory evidence and request scopes. SCAIL-2's physical
    /// `Reference + Mask + ControlClip` list describes one character identity, not two images (or
    /// three generic conditioning entries); every other carrier retains the generic image count.
    ///
    /// This is the **execution-side** half of a pair: the admission side computes the same quantity
    /// from the request via
    /// [`wan_i2v_memory::geometry_from_request`](crate::wan_i2v_memory::geometry_from_request), and
    /// both request-scope cores reject a request whose `memory_reference_count()` differs from the
    /// admitted `MemoryGeometry::reference_count`. The two therefore have to agree carrier for
    /// carrier or the mode is unreachable by construction — admission and execution can never both
    /// pass. The temporal-carrier arms below exist to close exactly that gap:
    ///
    /// * `video_bridge` owns two immutable source boundaries — two `Keyframe`s (Wan2.2 Ti2V-5B) or
    ///   one masked `ControlClip` (Wan-VACE) — and generates only the middle. Those are execution
    ///   inputs, not conceptual image references, so a bridge cannot borrow a still-image /
    ///   first-last memory row. Admission already declares this as zero.
    /// * `extend_clip` carries exactly one source: a boundary `Keyframe` (Ti2V-5B, already counted
    ///   as one image) or the whole source clip as one masked `ControlClip` (VACE). The generic
    ///   image count scores the VACE carrier as zero while admission scores it as one, so the VACE
    ///   arm is spelled out here.
    pub fn memory_reference_count(&self) -> u32 {
        match self.video_mode.as_deref() {
            Some("animation" | "replacement") if self.scail2_animation_conditioning().is_ok() => 1,
            Some("video_bridge") => 0,
            Some("extend_clip")
                if matches!(
                    self.conditioning.as_slice(),
                    [Conditioning::ControlClip { .. }]
                ) =>
            {
                1
            }
            _ => self.image_reference_count(),
        }
    }

    /// The first request-supplied `Option<f32>` knob that is **non-finite** (NaN / ±Inf), returned as
    /// `(field, value)` — or `None` when every present float is finite. This is the single home of the
    /// finiteness floor (F-053 / F-001): a NaN/Inf on *any* float knob flows into the guidance /
    /// scheduler / conditioning math and silently poisons the whole denoise (garbage-as-success, no
    /// error), so it must be rejected at the contract boundary.
    ///
    /// The exhaustive destructuring below (no `..`) is deliberate and load-bearing: adding a field to
    /// [`GenerationRequest`] fails to compile *here*, forcing the author to decide whether the new knob
    /// is a float that must join the guard. New `Option<f32>` fields therefore inherit the finiteness
    /// check **by construction** instead of silently slipping past it — the recurring "the floor lags
    /// the request surface" regression this method exists to close.
    pub fn first_nonfinite_float(&self) -> Option<(&'static str, f32)> {
        let Self {
            // Non-float fields: explicitly ignored, but named (no `..`) so a newly-added field breaks
            // the build here and the author must classify it.
            prompt: _,
            negative_prompt: _,
            width: _,
            height: _,
            count: _,
            seed: _,
            steps: _,
            timestep_to_start_cfg: _,
            sampler: _,
            scheduler: _,
            guidance_method: _,
            conditioning,
            frames: _,
            fps: _,
            video_mode: _,
            trim_first_frames: _,
            // Integer DFR knobs (sc-18789): slot count + round count carry no floats.
            num_generated_keyframes: _,
            temporal_upsample_rounds: _,
            decode_chunk_size: _,
            conditioning_fps: _,
            enhance_prompt: _,
            use_uncensored_enhancer: _,
            enhance_max_tokens: _,
            prompt_enhancement: _,
            use_pid: _,
            memory: _,
            // The approximation sub-block is float-free by construction (sc-18322): its policy is
            // step counts and its characterization reference is opaque strings. A float-bearing
            // approximate parameter — a quality/cost threshold, say — must join the floor, and this
            // named-not-`..` binding is what forces that decision when one arrives.
            approximation: _,
            cancel: _,
            preview: _,
            // The audio sub-block carries its own floats — destructured below the flat knobs.
            audio,
            // The multi-phase list carries per-phase floats (guidance + adapter weights), checked
            // below the flat knobs (sc-13884). Named (no `..`) so a future float-bearing per-phase
            // control fails to compile here until it is classified into the floor.
            phases,
            // Every `Option<f32>` knob the floor owns.
            guidance,
            true_cfg,
            scheduler_shift,
            guidance_eta,
            guidance_momentum,
            guidance_norm_threshold,
            strength,
            control_scale,
            image_guidance,
            duration,
            motion_bucket_id,
            noise_aug_strength,
            softness,
            enhance_temperature,
            pid_capture_sigma,
            text_style_gain,
        } = self;
        let floats: [(&'static str, Option<f32>); 16] = [
            ("guidance", *guidance),
            ("true_cfg", *true_cfg),
            ("scheduler_shift", *scheduler_shift),
            ("guidance_eta", *guidance_eta),
            ("guidance_momentum", *guidance_momentum),
            ("guidance_norm_threshold", *guidance_norm_threshold),
            ("strength", *strength),
            ("control_scale", *control_scale),
            ("image_guidance", *image_guidance),
            ("duration", *duration),
            ("motion_bucket_id", *motion_bucket_id),
            ("noise_aug_strength", *noise_aug_strength),
            ("softness", *softness),
            ("enhance_temperature", *enhance_temperature),
            ("pid_capture_sigma", *pid_capture_sigma),
            ("text_style_gain", *text_style_gain),
        ];
        for (name, v) in floats {
            if let Some(x) = v {
                if !x.is_finite() {
                    return Some((name, x));
                }
            }
        }
        // Audio sub-block floats (sc-12834): destructured exhaustively (no `..`) for the same
        // reason as the request itself — a new `AudioParams` float field fails to compile here
        // until it is classified into the floor.
        if let Some(AudioParams {
            voice: _,
            language: _,
            sample_rate: _,
            musical_key: _,
            lyrics: _,
            // The script carries no floats (text + opaque labels); named (no `..`) so a future
            // float-bearing per-segment control fails to compile here until it is classified.
            script: _,
            target_duration,
            bpm,
        }) = audio
        {
            let audio_floats: [(&'static str, Option<f32>); 2] = [
                ("audio.target_duration", *target_duration),
                ("audio.bpm", *bpm),
            ];
            for (name, v) in audio_floats {
                if let Some(x) = v {
                    if !x.is_finite() {
                        return Some((name, x));
                    }
                }
            }
        }
        // Conditioning-carried floats the floor also owns (F-001): every numeric a `Conditioning`
        // variant carries flows into the same denoise / scheduler / mask math as the flat knobs.
        //
        // **This match is deliberately wildcard-free and guard-free** (sc-19571). It used to end in
        // `_ => {}`, and that one arm is the whole reason the floor lagged the request surface: the
        // exhaustive `Self { .. }` destructure above makes a new *field* break the build, but a new
        // float-bearing `Conditioning` **variant** compiled clean and slipped straight past. Four
        // did — `Keyframe.strength`, `VideoClip.strength`, `ControlClip.masking_strength` and every
        // `ReduxRefs` per-ref strength — each of them a `1 − strength` denoise mask or a blend
        // weight, i.e. exactly the math this method exists to protect. Guards are avoided for the
        // same reason: `match` exhaustiveness ignores them, so a guarded arm re-opens the hole.
        // Adding a variant now fails to compile here until its numerics are classified.
        for c in conditioning {
            match c {
                Conditioning::Control { scale, .. } => {
                    if let Some(s) = scale {
                        if !s.is_finite() {
                            return Some(("conditioning.control.scale", *s));
                        }
                    }
                }
                Conditioning::Reference { strength, .. } => {
                    if let Some(s) = strength {
                        if !s.is_finite() {
                            return Some(("conditioning.reference.strength", *s));
                        }
                    }
                }
                Conditioning::ReferenceAudio { strength, .. } => {
                    if let Some(s) = strength {
                        if !s.is_finite() {
                            return Some(("conditioning.reference_audio.strength", *s));
                        }
                    }
                }
                // A reference clip's declared rate (sc-17149). Not a cosmetic label: it is the
                // divisor of the resample stride that picks which frames the model actually reads,
                // so a NaN propagates into a frame-index computation rather than into denoise math
                // — the same class of silent poisoning, one step earlier.
                Conditioning::ReferenceVideo { fps, .. } => {
                    if !fps.is_finite() {
                        return Some(("conditioning.reference_video.fps", *fps));
                    }
                }
                // The audio-edit strength and its region bounds all flow into the edit-window /
                // blend math; a NaN would silently poison the region conversion or the strength
                // gate (sc-12847).
                Conditioning::AudioEdit {
                    strength, region, ..
                } => {
                    if let Some(s) = strength {
                        if !s.is_finite() {
                            return Some(("conditioning.audio_edit.strength", *s));
                        }
                    }
                    if let Some(r) = region {
                        if !r.start_secs.is_finite() {
                            return Some((
                                "conditioning.audio_edit.region.start_secs",
                                r.start_secs,
                            ));
                        }
                        if let Some(end) = r.end_secs {
                            if !end.is_finite() {
                                return Some(("conditioning.audio_edit.region.end_secs", end));
                            }
                        }
                    }
                }
                // The multi-region carrier (sc-14549). Written as a **loop over every** region
                // rather than as more `if`-guarded arms, and that difference is the whole point:
                // the exhaustive destructuring above makes a new *field* break the build, but a
                // `Vec` defeats that mechanism entirely — a guard that reached only `regions[0]`
                // would compile, pass every pre-existing test, and let a NaN in region two flow
                // into the provider's mask rasterisation and poison it silently. That is exactly
                // the "the floor lags the request surface" regression this method exists to close,
                // so the testkit gate puts its bad value in region **two**.
                Conditioning::AudioEditRegions {
                    regions, strength, ..
                } => {
                    if let Some(s) = strength {
                        if !s.is_finite() {
                            return Some(("conditioning.audio_edit_regions.strength", *s));
                        }
                    }
                    for region in regions {
                        if !region.start_secs.is_finite() {
                            return Some((
                                "conditioning.audio_edit_regions.regions.start_secs",
                                region.start_secs,
                            ));
                        }
                        if let Some(end) = region.end_secs {
                            if !end.is_finite() {
                                return Some((
                                    "conditioning.audio_edit_regions.regions.end_secs",
                                    end,
                                ));
                            }
                        }
                    }
                }
                Conditioning::VoiceEmbedding { strength, .. } => {
                    if let Some(s) = strength {
                        if !s.is_finite() {
                            return Some(("conditioning.voice_embedding.strength", *s));
                        }
                    }
                }
                // sc-19571 — the four that the old `_ => {}` swallowed.
                //
                // A keyframe's `strength` is a `1 − strength` denoise mask on the pinned latent
                // frame AND the per-token diffusion timestep for that frame's tokens (Wan TI2V,
                // LTX). A NaN there does not merely mis-weight a pin: it makes the whole masked
                // frame's timestep NaN, which the denoise then multiplies into every step.
                Conditioning::Keyframe { strength, .. } => {
                    if !strength.is_finite() {
                        return Some(("conditioning.keyframe.strength", *strength));
                    }
                }
                // An in-context clip's `strength` is the same `1 − strength` mask on appended
                // conditioning tokens (LTX IC-LoRA, krea-realtime v2v).
                Conditioning::VideoClip { strength, .. } => {
                    if !strength.is_finite() {
                        return Some(("conditioning.video_clip.strength", *strength));
                    }
                }
                // `masking_strength` gates BOTH the pixel-space neutralization of the person region
                // and the mask-injection step count (`ceil(steps · masking_strength)`), so a NaN
                // reaches an integer step count as well as a blend weight.
                Conditioning::ControlClip {
                    masking_strength, ..
                } => {
                    if !masking_strength.is_finite() {
                        return Some((
                            "conditioning.control_clip.masking_strength",
                            *masking_strength,
                        ));
                    }
                }
                // Per-reference Redux weights. A **loop**, for the `AudioEditRegions` reason above:
                // checking only `refs[0]` would compile and pass, and a NaN in ref two would flow
                // into the conditioning blend unseen.
                Conditioning::ReduxRefs { refs } => {
                    for (_, strength) in refs {
                        if !strength.is_finite() {
                            return Some(("conditioning.redux_refs.strength", *strength));
                        }
                    }
                }
                // Float-free carriers: media and opaque labels only. Named rather than swallowed by
                // a wildcard so that adding a numeric to any of them breaks this match.
                Conditioning::MultiReference { images: _ } => {}
                Conditioning::Depth { image: _ } => {}
                Conditioning::Mask { image: _ } => {}
                Conditioning::VideoSync { frames: _ } => {}
                Conditioning::ConversationHistory { turns: _ } => {}
            }
        }
        // Multi-phase denoise floats (sc-13884): each phase's guidance and each phase-adapter weight
        // flow into the same guidance / adapter-scale math as the flat knobs, so a NaN/Inf must be
        // rejected here too rather than silently poisoning the phase's forward.
        if let Some(phases) = phases {
            for ph in phases {
                if let Some(g) = ph.guidance {
                    if !g.is_finite() {
                        return Some(("phases.guidance", g));
                    }
                }
                for pa in &ph.adapters {
                    if let Some(w) = pa.weight {
                        if !w.is_finite() {
                            return Some(("phases.adapter.weight", w));
                        }
                    }
                }
            }
        }
        None
    }

    /// Reject the request when any `Option<f32>` knob is non-finite (see
    /// [`first_nonfinite_float`](Self::first_nonfinite_float)). The shared home of the F-053 / F-001
    /// finiteness floor: [`Capabilities::validate_request`] calls it, and providers with a bespoke
    /// `validate` (e.g. flux1's IP-Adapter carve-out) call it directly so they inherit the guard too.
    pub fn ensure_finite_floats(&self) -> Result<()> {
        if let Some((field, value)) = self.first_nonfinite_float() {
            return Err(Error::Msg(format!("{field} must be finite (got {value})")));
        }
        Ok(())
    }

    /// All [`Conditioning::Keyframe`] inputs (first_last_frame / multi-keyframe), in request order.
    pub fn keyframes(&self) -> Vec<KeyframeRef<'_>> {
        self.conditioning
            .iter()
            .filter_map(|c| match c {
                Conditioning::Keyframe {
                    image,
                    frame_idx,
                    strength,
                } => Some(KeyframeRef {
                    image,
                    frame_idx: *frame_idx,
                    strength: *strength,
                }),
                _ => None,
            })
            .collect()
    }

    /// All [`Conditioning::VideoClip`] in-context clips (extend_clip / video_bridge), in request order.
    pub fn video_clips(&self) -> Vec<VideoClipRef<'_>> {
        self.conditioning
            .iter()
            .filter_map(|c| match c {
                Conditioning::VideoClip {
                    frames,
                    frame_idx,
                    strength,
                } => Some(VideoClipRef {
                    frames,
                    frame_idx: *frame_idx,
                    strength: *strength,
                }),
                _ => None,
            })
            .collect()
    }

    /// The replace_person masked control clip ([`Conditioning::ControlClip`]), if present. The first
    /// one wins (a request carries at most one person edit per generation).
    pub fn control_clip(&self) -> Option<ControlClipRef<'_>> {
        self.conditioning.iter().find_map(|c| match c {
            Conditioning::ControlClip {
                frames,
                mask,
                masking_strength,
                start_frame,
                mode,
            } => Some(ControlClipRef {
                frames,
                mask,
                masking_strength: *masking_strength,
                start_frame: *start_frame,
                mode: *mode,
            }),
            _ => None,
        })
    }

    /// Parse the exact ordered SCAIL-2 animation/replacement carrier. This deliberately does not infer the
    /// route from `conditioning.len() == 3`: every position and field is typed and crossed shapes
    /// fail closed before a provider configures or loads tensors.
    pub fn scail2_animation_conditioning(
        &self,
    ) -> crate::Result<Scail2AnimationConditioningRef<'_>> {
        let [Conditioning::Reference {
            image: character,
            strength: None,
        }, Conditioning::Mask {
            image: character_mask,
        }, Conditioning::ControlClip {
            frames,
            mask: driving_masks,
            masking_strength,
            start_frame,
            mode,
        }] = self.conditioning.as_slice()
        else {
            return Err(Error::Unsupported(
                "scail2 requires exactly ordered Reference(strength unset), Mask, and ControlClip conditioning"
                    .to_owned(),
            ));
        };
        if frames.is_empty() || driving_masks.len() != frames.len() {
            return Err(Error::Unsupported(format!(
                "scail2 requires one driving mask per non-empty driving frame ({} frames, {} masks)",
                frames.len(),
                driving_masks.len()
            )));
        }
        if *masking_strength != 1.0 || *start_frame != 0 || *mode != ReplacementMode::default() {
            return Err(Error::Unsupported(
                "scail2 ControlClip requires masking_strength=1, start_frame=0, and full-person mode"
                    .to_owned(),
            ));
        }
        if character.width == 0
            || character.height == 0
            || (character_mask.width, character_mask.height) != (character.width, character.height)
        {
            return Err(Error::Unsupported(
                "scail2 requires a non-empty character image and an exact-shape character mask"
                    .to_owned(),
            ));
        }
        let control_shape = (frames[0].width, frames[0].height);
        if control_shape.0 == 0
            || control_shape.1 == 0
            || frames.iter().zip(driving_masks).any(|(frame, mask)| {
                (frame.width, frame.height) != control_shape
                    || (mask.width, mask.height) != control_shape
            })
        {
            return Err(Error::Unsupported(
                "scail2 requires uniform non-empty driving frames with one exact-shape mask each"
                    .to_owned(),
            ));
        }
        Ok(Scail2AnimationConditioningRef {
            character,
            character_mask,
            driving_frames: frames,
            driving_masks,
        })
    }

    /// The prompted audio-edit conditioning ([`Conditioning::AudioEdit`]), if present. The first one
    /// wins (a request carries at most one source-audio edit per generation, mirroring
    /// [`control_clip`](Self::control_clip)). sc-12847.
    pub fn audio_edit(&self) -> Option<AudioEditRef<'_>> {
        self.conditioning.iter().find_map(|c| match c {
            Conditioning::AudioEdit {
                audio,
                mode,
                region,
                strength,
            } => Some(AudioEditRef {
                audio,
                mode: *mode,
                region: *region,
                strength: *strength,
            }),
            _ => None,
        })
    }

    /// The **multi-region** audio-edit conditioning ([`Conditioning::AudioEditRegions`]), if present
    /// (sc-14549). The first one wins, exactly as [`audio_edit`](Self::audio_edit) does — a request
    /// carries at most one source-audio edit per generation, and a provider that wants to refuse a
    /// second carrier rather than silently take the first inspects `conditioning` directly (which is
    /// what Stable Audio 3 does).
    ///
    /// This is a **separate accessor** from [`audio_edit`](Self::audio_edit), which is left
    /// byte-identical: a single-region caller keeps its exact shape, and a provider that has not
    /// opted into multi-region never sees one because
    /// [`ConditioningKind::AudioEditRegions`] is a distinct kind the shared allowlist refuses.
    pub fn audio_edit_regions(&self) -> Option<AudioEditRegionsRef<'_>> {
        self.conditioning.iter().find_map(|c| match c {
            Conditioning::AudioEditRegions {
                audio,
                mode,
                regions,
                strength,
            } => Some(AudioEditRegionsRef {
                audio,
                mode: *mode,
                regions,
                strength: *strength,
            }),
            _ => None,
        })
    }
}

/// Seed when a [`GenerationRequest`] omits one: nanos since the epoch (any nonzero value works —
/// this only sets which sample is drawn; a caller wanting reproducibility passes `req.seed`).
/// Shared by every generator (F-006).
pub fn default_seed() -> u64 {
    use std::time::{SystemTime, UNIX_EPOCH};
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_nanos() as u64)
        // Fall back to a nonzero value: 0 is the "no seed" sentinel a caller would pass to mean
        // "pick one", so the default must never itself be 0 (F-089).
        .unwrap_or(1)
}

/// Typed conditioning inputs. Each image family uses the subset its `Capabilities` advertises.
///
/// The video families ([`Conditioning::Keyframe`] / [`Conditioning::VideoClip`] /
/// [`Conditioning::ControlClip`]) are the epic-3040 advanced-mode inputs and map onto the two LTX
/// conditioning mechanisms (see `docs/SPIKE_ADVANCED_VIDEO_3040.md`): a [`Keyframe`](Conditioning::Keyframe)
/// is **replace-latent** (overwrite the target latent at a frame index — first_last_frame); a
/// [`VideoClip`](Conditioning::VideoClip) / [`ControlClip`](Conditioning::ControlClip) is
/// **keyframe-append** (append the clip's VAE latents as extra in-context tokens — extend_clip /
/// video_bridge / replace_person, the IC-LoRA path).
///
/// [`Conditioning::ReferenceVideo`] (sc-17149) is a **third** video mechanism and not part of that
/// pair: both epic-3040 mechanisms place their clip at an index in the generated timeline, and a
/// reference has no such index at all — it conditions the render without appearing in it.
#[derive(Clone, Debug)]
pub enum Conditioning {
    /// img2img / IP-Adapter / identity reference.
    Reference { image: Image, strength: Option<f32> },
    /// A reference **audio** clip — voice cloning / style reference for audio models
    /// (sc-12834; the audio analogue of [`Conditioning::Reference`]). `strength` mirrors the
    /// per-reference img2img strength: `None` ⇒ the model default. Video→audio (Foley)
    /// conditioning uses the dedicated [`Conditioning::VideoSync`] variant (sc-13436), not this
    /// one — this variant is audio-in only.
    ReferenceAudio {
        audio: AudioTrack,
        strength: Option<f32>,
    },
    /// A reference **video** clip — the video analogue of [`Conditioning::Reference`] /
    /// [`Conditioning::ReferenceAudio`], completing the reference triple (sc-17149). A motion and
    /// camera reference the model conditions on, optionally carrying **its own soundtrack**.
    ///
    /// A reference has **no position in the generated timeline** — that is what separates it from
    /// every other video-frame variant. It is not spliced into the output at an index, it does not
    /// bind the output geometry, and it is not softened by a caller-supplied denoise mask; it is
    /// encoded at its own resolution and packed as extra fully-pinned rows the denoise loop never
    /// writes. Its placement comes from its **ordinal in the request's `conditioning` list**, which
    /// is why that list is an ordered `Vec` and why reordering references is a different request.
    ///
    /// This is a **distinct variant**, deliberately not an overload of the two existing carriers:
    ///
    /// - It is **not** [`Conditioning::VideoClip`]. That is the LTX in-context *latent-append* path,
    ///   whose whole contract is the two fields a reference cannot use: `frame_idx` (a position in
    ///   the generated timeline) and `strength` (a `1 − strength` denoise mask). A reference has
    ///   neither, so riding `VideoClip` means shipping a vocabulary in which two of three fields are
    ///   constants a provider must reject — and it still cannot carry the two things a reference
    ///   genuinely needs, `fps` and `audio`.
    /// - It is **not** [`Conditioning::VideoSync`]. That is the Foley condition (sc-13436): frames
    ///   an audio decoder attends to in order to score a *supplied silent clip*, explicitly "not
    ///   spliced into a video latent". A reference video *is* VAE-encoded into latent rows and
    ///   conditions a *generated* clip. Advertising `VideoSync` would advertise a Foley capability,
    ///   and the kind is what routing reads.
    ///
    /// # `fps` is required data, not a hint
    ///
    /// A model resamples a reference onto its own frame rate by dropping and duplicating whole
    /// frames, so a clip whose real rate was lost is conditioned on **at the wrong speed with
    /// nothing to raise about it**. That is why the rate is a bare `f32` here rather than an
    /// `Option` with a plausible default.
    ///
    /// This does **not** contradict the single-source-of-truth argument [`Conditioning::VideoSync`]
    /// makes for reading the request-level [`GenerationRequest::fps`]: that field is the rate of the
    /// *generated output*, whereas this one is the rate of *supplied input media*. For a variant
    /// whose defining property is that it does not bind the output timeline, the two are genuinely
    /// different quantities and a model may legally reject an output rate it happily accepts as an
    /// input rate. Reusing `req.fps` for both would pin every reference to the output rate.
    ///
    /// # `audio` is the reference's own soundtrack
    ///
    /// `Some(track)` conditions on the clip's own soundtrack, aligned with its video rows and
    /// sharing their origin on the model's rotary clock. This is **not** the same request as sending
    /// the same waveform as a separate [`Conditioning::ReferenceAudio`], which is a *standalone*
    /// reference occupying its own slot — a distinction models with per-modality reference caps also
    /// count differently. `None` conditions on motion alone.
    ///
    /// A model opts in by advertising [`ConditioningKind::ReferenceVideo`] in
    /// [`Capabilities::conditioning`]; the shared floor rejects the variant on a non-advertising
    /// model as the typed [`Error::Unsupported`] (F-008), an empty `frames` as [`Error::Msg`], and a
    /// non-finite `fps` through the same finiteness floor that owns every other conditioning float.
    /// Per-model rate, resolution and cap bounds are layered by the provider's own `validate`.
    ReferenceVideo {
        frames: Vec<Image>,
        /// **The rate `frames` actually carry** — see the variant docs.
        fps: f32,
        /// This clip's own soundtrack, conditioned on as the reference's own rather than as a
        /// reference of its own. `None` conditions on motion alone.
        audio: Option<AudioTrack>,
    },
    /// **Prompted source-audio editing** (sc-12847) — the audio analogue of the image lane's masked
    /// edit / inpaint conditioning ([`Conditioning::Mask`] and the region-carrying
    /// [`Conditioning::ControlClip`]): a source clip plus an edit *mode* and an optional *region*,
    /// so the prompt (+ lyrics/metadata) restyles or regenerates part or all of the clip.
    ///
    /// This is a **distinct variant**, not an extension of [`Conditioning::ReferenceAudio`] — that
    /// variant is deliberately scoped to a whole-clip voice/style reference (audio-in only), and an
    /// edit carries a fundamentally different shape (a task mode + a bounded region). Bundling the
    /// clip, mode, region, and strength in one self-contained variant mirrors how
    /// [`Conditioning::ControlClip`] carries `frames` + `mask` + `mode` + `start_frame` +
    /// `masking_strength` together for the video replace_person edit, and keeps
    /// `ReferenceAudio`'s serialized/semantic contract stable (CONTRIBUTING.md compatibility).
    ///
    /// - `audio` — the source clip to edit.
    /// - `mode` — which edit operation ([`AudioEditMode`]); the provider maps it onto ACE-Step's
    ///   native task modes.
    /// - `region` — the span to edit ([`TimeRegion`], seconds); `None` = whole clip. Region modes
    ///   ([`AudioEditMode::Inpaint`] / [`AudioEditMode::Repaint`] / [`AudioEditMode::Extend`])
    ///   require it; [`AudioEditMode::Cover`] ignores it.
    /// - `strength` — edit strength; `None` ⇒ the model default. Joins the finiteness floor.
    AudioEdit {
        audio: AudioTrack,
        mode: AudioEditMode,
        region: Option<TimeRegion>,
        strength: Option<f32>,
    },
    /// **Multi-region** prompted source-audio editing (sc-14549) — several non-contiguous spans
    /// regenerated in a *single* pass, sharing one denoising trajectory.
    ///
    /// This is not equivalent to N sequential [`Conditioning::AudioEdit`]s: each sequential pass
    /// re-noises and re-decodes, so the regions do not share a trajectory and the seams differ.
    /// Providers that support it build **one** mask carrying every span and run **one** sampler.
    ///
    /// # Why a separate variant rather than a field on [`AudioEdit`](Conditioning::AudioEdit)
    ///
    /// Adding `regions` to the existing variant is not source-compatible in Rust — every
    /// constructor and every exact destructuring pattern breaks — and it would make every
    /// single-region caller carry a field it must ignore. A separate variant with its **own**
    /// [`ConditioningKind::AudioEditRegions`] is additive in the only sense that matters: the
    /// legacy variant, [`AudioEditRef`] and [`GenerationRequest::audio_edit`] are untouched, and
    /// default-deny comes for free. [`Capabilities::accepts`] refuses any unadvertised kind as a
    /// typed [`Error::Unsupported`], so every provider that has not opted in — ACE-Step included —
    /// rejects multi-region cleanly with no code change and no capability flag.
    ///
    /// # Semantics (this repository's, deliberately stated rather than inherited)
    ///
    /// These are **not** claimed as parity with any upstream implementation; no upstream
    /// multi-region reference has been verified here (see sc-15431).
    ///
    /// - **`regions` is non-empty.** An empty list is a malformed request, not a whole-clip edit —
    ///   whole-clip restyle is [`Conditioning::ReferenceAudio`].
    /// - **Order is not significant.** Regions may arrive in any order; a provider normalizes them
    ///   into a canonical union, so `[a, b]` and `[b, a]` describe the same edit.
    /// - **Overlapping, touching and duplicate regions are accepted, not rejected**, and merged
    ///   into that union. Refusing them would make the contract order-sensitive in practice and
    ///   would reject requests with an unambiguous meaning.
    /// - **`end_secs` must be `Some` on every region.** `None` means "to the end of the clip",
    ///   which is only well-defined for a *final* region — and since order is not significant,
    ///   "final" is not well-defined here. Rather than leave that ambiguity, `None` is refused
    ///   outright; a caller wanting a span that runs to the clip end states the end explicitly, and
    ///   the single-region [`Conditioning::AudioEdit`] keeps the `None` shorthand unchanged. No
    ///   capability is lost: [`AudioEditMode::Extend`] stays a single-tail operation on the legacy
    ///   variant.
    /// - Each region is otherwise gated exactly as [`TimeRegion`] already is: finite bounds
    ///   (**every** region's, not just the first), `start_secs >= 0`, `end_secs > start_secs`.
    /// - Clip-bound checks (a region inside the source's duration) and latent-collapse checks stay
    ///   with the provider, which knows the clip length and its own VAE stride — the same division
    ///   of labour the single-region path uses.
    AudioEditRegions {
        audio: AudioTrack,
        mode: AudioEditMode,
        /// The spans to regenerate. Non-empty; every `end_secs` is `Some`; any order.
        regions: Vec<TimeRegion>,
        strength: Option<f32>,
    },
    /// A precomputed **voice-identity embedding** — a cloned voice driving TTS (sc-12838; the audio
    /// analogue of how a [`FaceEmbedder`](crate::face::FaceEmbedder) identity vector conditions
    /// InstantID / PuLID). Unlike [`Conditioning::Reference`] / [`Conditioning::ReferenceAudio`],
    /// which carry raw media the generator re-embeds, this carries the
    /// [`VoiceEmbedder`](crate::voice_embed::VoiceEmbedder) output directly, because the embedder is
    /// a standalone registry provider composed separately from the TTS generator (sc-12844).
    /// `strength` mirrors the img2img/reference strength: `None` ⇒ the model default identity
    /// weight; it joins the same finiteness floor.
    VoiceEmbedding {
        embedding: VoiceEmbedding,
        strength: Option<f32>,
    },
    /// Multiple references with no per-image strength (Qwen-Image-Edit).
    MultiReference { images: Vec<Image> },
    /// FLUX.1-Redux references, each with its own strength.
    ReduxRefs { refs: Vec<(Image, f32)> },
    /// ControlNet / pose conditioning. `scale` mirrors the strength on
    /// [`Conditioning::Reference`]: `None` means "use the
    /// per-model default control scale" and `Some(x)` is an explicit override — including `Some(0.0)`,
    /// a deliberately inert control branch. The `Option` is what distinguishes explicit-inert from
    /// unset (the old bare `f32` could not; F-085).
    Control {
        image: Image,
        kind: ControlKind,
        scale: Option<f32>,
    },
    /// FLUX.1-Depth.
    Depth { image: Image },
    /// FIBO-Edit / inpaint mask.
    Mask { image: Image },
    /// A keyframe pinned at a specific output **latent** frame index (first_last_frame / general
    /// multi-keyframe). VAE-encoded and its tokens **overwrite** the target latent at `frame_idx`
    /// with denoise mask `1 − strength` (the replace-latent mechanism — reference
    /// `VideoConditionByLatentIndex`). `strength = 1.0` fully pins the frame. first_last_frame is two
    /// of these (at `0` and the last latent frame).
    Keyframe {
        image: Image,
        frame_idx: i32,
        strength: f32,
    },
    /// An in-context conditioning **clip** (extend_clip / video_bridge — the LTX IC-LoRA path). The
    /// frames are VAE-encoded and **appended** as extra tokens at `frame_idx` (RoPE-offset on the
    /// frame axis) with denoise mask `1 − strength` (reference `VideoConditionByKeyframeIndex`).
    /// extend_clip = one clip at `frame_idx 0`; video_bridge = a left clip at `0` and a right clip at
    /// the tail.
    VideoClip {
        frames: Vec<Image>,
        frame_idx: i32,
        strength: f32,
    },
    /// A masked control clip for replace_person. `frames` is the (host-built, person-region
    /// neutralized) control clip; `mask` is the per-frame binary person mask (white = regenerate).
    /// Drives the keyframe-append in-context conditioning **plus** mask injection (force the masked
    /// region toward the re-noised source for the first `ceil(steps · masking_strength)` steps —
    /// reference `prepare_mask_injection`). Person detect/track stays in onnx and supplies these.
    ControlClip {
        frames: Vec<Image>,
        mask: Vec<Image>,
        masking_strength: f32,
        /// Output latent-frame the control clip aligns to (reference `masking_source.start_frame`).
        start_frame: i32,
        /// Replacement granularity (reference `replacement_mode`); the LTX mask path is region-driven
        /// so it is carried for the worker contract / WanVACE parity rather than changing the mask math.
        mode: ReplacementMode,
    },
    /// A **video clip whose RGB frames drive a video→audio (Foley) generator** (sc-13436) — the
    /// visual condition an MMAudio-style model reads to synthesize a synchronized soundtrack for a
    /// silent clip.
    ///
    /// This is a **distinct variant**, deliberately *not* an overload of the two existing video
    /// mechanisms, exactly as [`Conditioning::AudioEdit`] is kept distinct from
    /// [`Conditioning::ReferenceAudio`]:
    ///
    /// - It is **not** [`Conditioning::VideoClip`]. That variant is the LTX in-context *latent-append*
    ///   path — the clip is VAE-encoded and appended as extra denoise tokens at a specific `frame_idx`
    ///   with a `strength` denoise mask (extend_clip / video_bridge). `VideoSync` carries no
    ///   `frame_idx` and no `strength`: the frames are not spliced into a video latent, they are the
    ///   whole-clip visual condition an audio decoder attends to. Reusing `VideoClip` would force a
    ///   Foley model to invent a meaningless frame index and pin it against the video denoise contract.
    /// - It is **not** [`Conditioning::ControlClip`]. That is the masked replace_person edit (`frames`
    ///   **+** a per-frame binary `mask` + `masking_strength` + `start_frame`), a fundamentally
    ///   different shape carrying a mask this variant has no notion of.
    ///
    /// The clip is just its ordered RGB `frames`. The frame **rate** is *not* carried here — it rides
    /// the existing request-level [`GenerationRequest::fps`], exactly as the LTX
    /// (`mlx-gen-ltx`) and Wan-VACE (`candle-gen-wan`) video paths already read `req.fps`; duplicating
    /// it on the variant would create a second source of truth the two could disagree on. A model opts
    /// in by advertising [`ConditioningKind::VideoSync`] in
    /// [`Capabilities::conditioning`]; the shared floor rejects the variant on a non-advertising model
    /// as the typed [`Error::Unsupported`] (F-008) and an empty `frames` as [`Error::Msg`].
    VideoSync { frames: Vec<Image> },
    /// A **multi-turn conversation history** driving context-aware conversational TTS (sc-14150,
    /// path **A**) — an ordered list of [`ConversationTurn`]s a voice-agent model reads to synthesize
    /// the trailing assistant reply conditioned on every prior turn (their text **and** audio). This
    /// is the *stateless* carrier: the whole conversation rides in the request, so the model rebuilds
    /// the conversation prefix on each [`Generator::generate`] call (batch conversational render). Its
    /// stateful counterpart is the warm-cache [`ConversationSession`] (path **B**), which
    /// [`Generator::open_conversation`] opens — the same per-turn computation kept hot across turns.
    ///
    /// This is a **distinct variant**, deliberately not an overload of the single-request multi-speaker
    /// [`AudioParams::script`] (sc-12848): a script is one utterance rendered in assigned voices into a
    /// single track with **no** cross-utterance conditioning, whereas a conversation is a sequence of
    /// turns where turn *N* is *conditioned on* turns *1..N-1* (their generated audio carried forward).
    /// Tensor-free: each turn's audio is a PCM [`AudioTrack`], the provider encodes it. A model opts in
    /// through [`Capabilities::supports_conversation_history`] **and** by advertising
    /// [`ConditioningKind::ConversationHistory`] in [`Capabilities::conditioning`]; the shared floor
    /// rejects a conversation on a non-advertising model as the typed [`Error::Unsupported`] (F-008)
    /// and an empty `turns` as [`Error::Msg`].
    ConversationHistory { turns: Vec<ConversationTurn> },
}

impl Conditioning {
    /// The [`ConditioningKind`] discriminant — for capability checks / `validate()`. Centralized here
    /// so adding a [`Conditioning`] variant updates every model's validation in one place.
    pub fn kind(&self) -> ConditioningKind {
        match self {
            Conditioning::Reference { .. } => ConditioningKind::Reference,
            Conditioning::ReferenceAudio { .. } => ConditioningKind::ReferenceAudio,
            Conditioning::ReferenceVideo { .. } => ConditioningKind::ReferenceVideo,
            Conditioning::AudioEdit { .. } => ConditioningKind::AudioEdit,
            Conditioning::AudioEditRegions { .. } => ConditioningKind::AudioEditRegions,
            Conditioning::VoiceEmbedding { .. } => ConditioningKind::VoiceEmbedding,
            Conditioning::MultiReference { .. } => ConditioningKind::MultiReference,
            Conditioning::ReduxRefs { .. } => ConditioningKind::ReduxRefs,
            Conditioning::Control { .. } => ConditioningKind::Control,
            Conditioning::Depth { .. } => ConditioningKind::Depth,
            Conditioning::Mask { .. } => ConditioningKind::Mask,
            Conditioning::Keyframe { .. } => ConditioningKind::Keyframe,
            Conditioning::VideoClip { .. } => ConditioningKind::VideoClip,
            Conditioning::ControlClip { .. } => ConditioningKind::ControlClip,
            Conditioning::VideoSync { .. } => ConditioningKind::VideoSync,
            Conditioning::ConversationHistory { .. } => ConditioningKind::ConversationHistory,
        }
    }
}

/// Granularity of a replace_person edit (reference `replacement_mode`).
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum ReplacementMode {
    /// Replace the face region only.
    #[default]
    FaceOnly,
    /// Replace the full person but keep the original outfit.
    FullPersonKeepOutfit,
    /// Replace the full person including the outfit.
    FullPersonReplaceOutfit,
}

/// The control signal carried by [`Conditioning::Control`].
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum ControlKind {
    Pose,
    Canny,
    Depth,
    Other(String),
}

/// Which [`Conditioning`] variants a model accepts — for capability introspection + validation.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ConditioningKind {
    Reference,
    /// Voice/style reference audio ([`Conditioning::ReferenceAudio`]).
    ReferenceAudio,
    /// Motion/camera reference video, optionally with its own soundtrack
    /// ([`Conditioning::ReferenceVideo`], sc-17149). A **distinct** kind from
    /// [`VideoClip`](Self::VideoClip): a reference has no position in the generated timeline and no
    /// denoise mask, so the two carry disjoint payloads and default-deny keeps every existing
    /// in-context-clip provider from being handed a reference it would splice into the output.
    ReferenceVideo,
    /// Prompted source-audio editing ([`Conditioning::AudioEdit`]).
    AudioEdit,
    /// **Multi-region** prompted source-audio editing ([`Conditioning::AudioEditRegions`],
    /// sc-14549). A **distinct** kind from [`AudioEdit`](Self::AudioEdit), deliberately: that is
    /// what makes default-deny free. A provider serving only single-region editing advertises only
    /// `AudioEdit`, and [`Capabilities::accepts`] then refuses a multi-region request as a typed
    /// [`Error::Unsupported`] with no flag and no provider code. Reusing `AudioEdit` for both would
    /// let every existing audio-edit provider through the allowlist and would require a
    /// default-false capability flag purely to re-close the hole that created.
    AudioEditRegions,
    /// A precomputed cloned-voice identity embedding ([`Conditioning::VoiceEmbedding`]).
    VoiceEmbedding,
    MultiReference,
    ReduxRefs,
    Control,
    Depth,
    Mask,
    /// first_last_frame / multi-keyframe ([`Conditioning::Keyframe`]).
    Keyframe,
    /// extend_clip / video_bridge ([`Conditioning::VideoClip`]).
    VideoClip,
    /// replace_person ([`Conditioning::ControlClip`]).
    ControlClip,
    /// video→audio (Foley) sync ([`Conditioning::VideoSync`]).
    VideoSync,
    /// multi-turn conversation history ([`Conditioning::ConversationHistory`]).
    ConversationHistory,
}

/// What kind of media a model emits.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Modality {
    Image,
    Video,
    /// Emits both image and video (e.g. the SeedVR2 upscaler over either).
    Both,
    /// Pure audio synthesis — TTS / music (sc-12834). Emits [`GenerationOutput::Audio`];
    /// `width`/`height` are unused, so audio models validate through the size-skipping floor
    /// ([`Capabilities::validate_request_audio`]).
    Audio,
}

/// A model's stable identity + advertised capabilities. Returned by `descriptor()` and also
/// constructible without loading weights (registry introspection).
#[derive(Clone, Debug)]
pub struct ModelDescriptor {
    /// Exact text-encoder architecture/output contract for safe [`crate::LoadSpec::text_encoder`]
    /// substitution. `None` means the provider has not advertised a substitutable encoder.
    pub encoder_contract: Option<crate::EncoderContract>,
    /// The exact latent tensor emitted by the denoiser at its decoder boundary. `None` means the
    /// provider has not advertised enough information; consumers must treat it as incompatible with
    /// every decoder rather than infer compatibility from family names or channel counts.
    pub denoiser_output_latent_space: Option<&'static crate::latent::LatentSpace>,
    pub id: &'static str,
    pub family: &'static str,
    /// `"mlx"` | `"candle"` — the tensor backend whose provider crate registered this engine.
    /// Drives the worker's registry-derived capability advertisement (sc-3723); MLX families set
    /// `"mlx"`.
    pub backend: &'static str,
    pub modality: Modality,
    pub capabilities: Capabilities,
    /// **Named model components this engine requires** at load (epic 13657) — a weights-free
    /// advertisement of the extra artifacts a consumer must provision, beyond the base `weights` and
    /// the typed [`LoadSpec`](crate::LoadSpec) overlays, before calling `load`. The complement of
    /// [`LoadSpec::components`](crate::LoadSpec::components): the model declares its required ids here
    /// so SceneWorks knows what to stage (and the load fails fast if it doesn't — see
    /// [`require_component`](crate::control::require_component)), and the caller supplies each id's
    /// resolved local path in the load spec's `components` map.
    ///
    /// `Default` / the shipped value for every image/video provider and every single-file audio model
    /// is `&[]` — no extra components; the field is strictly additive. Each id is a lowercase
    /// `snake_case` registry identifier; the descriptor conformance sweep
    /// ([`model_descriptor_errors`](crate::registry::model_descriptor_errors)) requires the declared
    /// ids to be non-empty and unique. The concrete ids per model are the registry documented on
    /// [`LoadSpec::components`](crate::LoadSpec::components) (e.g. chatterbox `["perth",
    /// "voice_embedding"]`).
    pub required_components: &'static [&'static str],
    /// **Which control signals this model admits**, weights-free — the descriptor-level twin of
    /// [`ControlBranch::accepted_control_kinds`](crate::control::ControlBranch::accepted_control_kinds).
    ///
    /// [`capabilities.conditioning`](Capabilities::conditioning) says *whether* a model takes
    /// [`Conditioning::Control`] but never *which kind*, and the kind policy lived only on the
    /// loaded control struct (`&self` on a `ControlBranch`). A consumer planning a render therefore
    /// could not tell a pose-only branch from a pose/canny/depth union without loading multi-GB
    /// weights, and a depth request aimed at a pose-only branch failed inside `generate` — after
    /// residency — instead of before it.
    ///
    /// The distinction between the two `None`-ish answers is the point, because the permissive
    /// answer is the dangerous one:
    ///
    /// - `None` — **not advertised**. The model may still reject kinds at render time; a consumer
    ///   must treat control kind as unchecked. This is the `Default` for a reason: a model that
    ///   forgot to declare must not read as "accepts anything".
    /// - `Some(`[`Any`](crate::control::AcceptedControlKinds::Any)`)` — deliberately
    ///   input-agnostic, the Fun-Controlnet-Union position (pose/canny/depth share one VAE-encoded
    ///   path and differ only by the host-side preprocessor).
    /// - `Some(`[`Only`](crate::control::AcceptedControlKinds::Only)`(..))` — exactly these kinds;
    ///   anything else is rejected rather than silently coerced.
    ///
    /// A [`ControlBranch`](crate::control::ControlBranch) implementor should declare it here rather
    /// than override the trait method: the trait's default **reads this field**, so the descriptor
    /// is the single source of truth and the advertised policy cannot drift from the enforced one.
    ///
    /// [`Conditioning::Control`]: Conditioning::Control
    pub control_kinds: Option<crate::control::AcceptedControlKinds>,
}

impl ModelDescriptor {
    /// Alternate decoders this exact provider has wired and whose input space is compatible with its
    /// advertised denoiser output. Missing/learned normalization evidence therefore fails closed.
    pub fn compatible_decoder_options(&self) -> Vec<crate::latent::DecoderOption> {
        crate::latent::DECODER_OPTIONS
            .iter()
            .copied()
            .filter(|option| option.eligible_backends.contains(&self.backend))
            .filter(|option| option.eligible_provider_ids.contains(&self.id))
            .filter(|option| {
                crate::latent::latent_spaces_compatible(
                    self.denoiser_output_latent_space,
                    Some(option.input_latent_space),
                )
            })
            .collect()
    }
}

/// How a model's advertised size range is enforced.
///
/// The distinction already existed as two entry points —
/// [`Capabilities::validate_request`] and
/// [`Capabilities::validate_request_skip_size`] — with the choice made by the
/// provider and advertised nowhere. This makes it readable.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum SizeFloor {
    /// `width`/`height` must lie within `min_size..=max_size`. The common case.
    #[default]
    RangeChecked,
    /// The ordinary range check plus a required grid for every explicit `width`/`height`.
    ///
    /// A request is accepted only when both dimensions are multiples of `multiple`. This is
    /// advertised here, rather than hidden in provider code, so a weights-free consumer can predict
    /// whether the engine will render the exact geometry it was asked for.
    RangeCheckedOnGrid { multiple: u32 },
    /// `width`/`height == 0` is a "resolve from the driving media" sentinel, and the shared range
    /// check does not apply **to that pair of values** (F-158). Any other size — including a
    /// half-sentinel like `0x512` — is still range-checked normally.
    ///
    /// # What this does not promise
    ///
    /// It says nothing about the size the provider *resolves* to. A consumer must not read this
    /// variant as "the resolved geometry is bounded by `min_size..=max_size`": whether a provider
    /// re-checks after resolving is per-provider and is **not advertised here**.
    ///
    /// SCAIL-2, the only model using this resolved-downstream policy family, sets
    /// [`SizeFloor::ResolvedDownstreamExplicitGrid`] on both backends and does re-check — it refuses a
    /// resolved geometry outside `min_size..=max_size` or over its area cap before the render, naming
    /// the largest in-envelope geometry at the source aspect. But that is SCAIL-2's own guarantee,
    /// made by each SCAIL-2 provider, **not** something this variant asserts on behalf of a provider that
    /// sets it later. A consumer needing the bound must still read the provider, or treat the
    /// resolved size as unbounded.
    ///
    /// The Candle provider adopted the same safe sentinel policy in sc-16199; before that change it
    /// declared [`SizeFloor::RangeCheckedOnGrid`] and its resolve-from-the-clip branch was unreachable.
    ResolvedDownstream,
    /// [`ResolvedDownstream`](Self::ResolvedDownstream), with an additional grid requirement for
    /// **explicit** dimensions.
    ///
    /// The `0x0` sentinel remains exempt because the caller did not choose the source-media
    /// geometry; the provider may align that resolved geometry as part of its documented downstream
    /// policy. Any non-sentinel request must already land on `multiple`, so it is either rendered
    /// exactly or rejected before generation. SCAIL-2 uses this to preserve ordinary `640x360`
    /// driving clips while refusing an explicit `1280x730` instead of silently rendering
    /// `1280x704` (sc-16198).
    ResolvedDownstreamExplicitGrid { multiple: u32 },
}

impl SizeFloor {
    /// The required multiple for explicit dimensions, when this floor advertises one.
    ///
    /// `None` means the shared floor has no grid opinion; a provider may still layer a model-local
    /// rule. Consumers can call this without loading weights.
    pub fn explicit_size_multiple(self) -> Option<u32> {
        match self {
            Self::RangeChecked | Self::ResolvedDownstream => None,
            Self::RangeCheckedOnGrid { multiple }
            | Self::ResolvedDownstreamExplicitGrid { multiple } => Some(multiple),
        }
    }
}

/// A model component whose resident numeric tier may deliberately differ from the tier selected for
/// the model as a whole.
///
/// This vocabulary is intentionally shared with SceneWorks' tier-integrity ledger. Providers use it
/// to expose precision floors to callers before weights are loaded, so tier selection, telemetry,
/// and memory-evidence identity do not have to infer provider-local packing exceptions.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum PrecisionFloorComponent {
    TextEncoder,
    TransformerHead,
}

impl PrecisionFloorComponent {
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::TextEncoder => "textEncoder",
            Self::TransformerHead => "transformerHead",
        }
    }
}

/// One worker-visible component precision floor.
///
/// When `selected_tier` is requested, the named component is resident at no less than
/// `resident_tier`. A provider that raises a component above the selected tier must declare that
/// substitution here; callers include the declaration in labels and memory evidence.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct ComponentPrecisionFloor {
    pub component: PrecisionFloorComponent,
    pub selected_tier: Quant,
    pub resident_tier: Quant,
}

impl ComponentPrecisionFloor {
    pub const fn applies_to(self, selected: Quant) -> bool {
        matches!(
            (self.selected_tier, selected),
            (Quant::Q4, Quant::Q4) | (Quant::Q8, Quant::Q8) | (Quant::Nvfp4, Quant::Nvfp4)
        )
    }
}

/// Resolve the quant tier a provider must use for one component from its advertised floor table.
/// Providers and callers share this function so the load path cannot apply a different substitution
/// from the one visible in descriptor introspection.
pub fn effective_component_quant(
    floors: &[ComponentPrecisionFloor],
    component: PrecisionFloorComponent,
    selected: Quant,
) -> Quant {
    floors
        .iter()
        .copied()
        .find(|floor| floor.component == component && floor.applies_to(selected))
        .map_or(selected, |floor| floor.resident_tier)
}

/// Provider-owned warm activation transient measured at 1024×1024, in bytes.
///
/// `bytes_1024` is the bare engine allocation (`peak − resident`) for one warm image. It excludes
/// model weights and OS/application reserve; consumers add those independently and may scale this
/// anchor for request geometry. A route-wide anchor is valid only when measurements establish that
/// its activation high-water is tier-independent. A provider with storage- or tier-dependent
/// activation memory must omit this route-only carrier until a spec-aware contract exists. Distinct
/// edit/control routes retain their own provider ids and must register separately. Providers publish
/// only real on-device measurements at or above the observed high-water mark; no registration means
/// "unmeasured".
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct ActivationMemoryAnchor {
    pub bytes_1024: u64,
}

/// Static descriptor classification for the provider's staged-residency behavior. This describes
/// physical execution independent of request-selected memory-strategy evidence.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum StagedResidencyAvailability {
    /// The provider neither stages components unconditionally nor offers the shared selectable
    /// sequential-residency control.
    #[default]
    Absent,
    /// The provider honors the shared selectable sequential-residency control, but does not perform
    /// the staged load/use/drop lifecycle on every request by default.
    Selectable,
    /// The provider performs a staged component load/use/drop lifecycle on every request, regardless
    /// of whether it also offers a stronger selectable sequential-residency control.
    UnconditionallyEngaged,
}

/// The denoise step counts a model can render — the ONE representation for every step-count
/// shape the catalog has needed (sc-19559).
///
/// Three shapes accreted before this type existed: a **minimum** (SceneWorks'
/// `limits.hardMinSteps`, sc-19426), an **exact set** (`supported_steps: Vec<u32>`, sc-19502),
/// and a **ceiling**, which nothing could express at all — SVD's `MAX_STEPS = 200` lived only
/// inside `mlx-gen-svd`'s and `candle-gen-svd`'s `validate`, so a consumer could not learn the
/// bound without dispatching a job. Rather than adding a fourth key, the shapes collapse here:
/// a model is unconstrained, pinned to an exact menu, or bounded by an inclusive range.
///
/// **Rejected alternatives.** A separate `max_steps: Option<u32>` beside the existing `Vec<u32>`
/// was cheaper but lets a descriptor declare `supported_steps: [8]` *and* `max_steps: 4` — a
/// contradiction only a cross-check could catch, and it leaves the minimum still unexpressible,
/// so the next model needing a floor adds a fourth key. Extending `Vec<u32>` to enumerate
/// `1..=200` as 200 elements is a set pretending to be a range: it makes the ceiling
/// undiscoverable as a *bound*, and Kolors' `1..=1100` would be an 1100-element vector in every
/// descriptor snapshot.
///
/// **Polarity.** [`Unconstrained`](Self::Unconstrained) is the `Default` and means **no
/// constraint** — deliberately the opposite of [`Capabilities::samplers`], where empty means
/// "reject any explicit value". A model opts *in* to being constrained. Inverting it would make
/// a bare `Default::default()` refuse every step count in the repo.
///
/// Only an EXPLICIT `req.steps` is judged; `None` means "the model picks its baked default" and
/// always passes.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub enum StepSupport {
    /// No advertised constraint: any count the shared sanity caps admit.
    #[default]
    Unconstrained,
    /// The **only** legal counts. A distilled model bakes its σ waypoints into training, so the
    /// step count is not a knob at all — LTX-2.3 runs a fixed 8-step stage-1 schedule
    /// (`STAGE1_SIGMAS`, 9 waypoints) and cannot honor any other count without going
    /// out-of-distribution.
    ///
    /// An empty set would refuse every count including the model's own default, which no model
    /// wants; [`model_descriptor_errors`](crate::registry::model_descriptor_errors) rejects it as
    /// a descriptor mistake.
    Exact(Vec<u32>),
    /// Any count in `min..=max`, inclusive on both ends. This is the shape a model with a real
    /// engine bound has: SVD-XT's 200-step ceiling, ACE-Step's 200, MMAudio's 500, Kolors'
    /// 1100 train timesteps.
    ///
    /// `min > max` is an unsatisfiable declaration and is rejected by
    /// [`model_descriptor_errors`](crate::registry::model_descriptor_errors).
    Range { min: u32, max: u32 },
}

impl StepSupport {
    /// Whether this model can render exactly `steps` denoise steps.
    pub fn admits(&self, steps: u32) -> bool {
        match self {
            Self::Unconstrained => true,
            Self::Exact(counts) => counts.contains(&steps),
            Self::Range { min, max } => (*min..=*max).contains(&steps),
        }
    }

    /// The largest step count this model renders, or `None` when it advertises no ceiling.
    ///
    /// This is the weights-free ceiling read: a consumer sizing a Steps control asks the
    /// descriptor rather than dispatching a job and reading the failure. An
    /// [`Exact`](Self::Exact) menu's ceiling is its largest member; an empty menu has none.
    pub fn ceiling(&self) -> Option<u32> {
        match self {
            Self::Unconstrained => None,
            Self::Exact(counts) => counts.iter().copied().max(),
            Self::Range { max, .. } => Some(*max),
        }
    }

    /// The smallest step count this model renders, or `None` when it advertises no floor.
    ///
    /// ⚠️ This is a **bound, not a default**. A consumer must not seed a Steps control with it:
    /// omitting `steps` selects the model's own baked default, which is generally not the floor.
    pub fn floor(&self) -> Option<u32> {
        match self {
            Self::Unconstrained => None,
            Self::Exact(counts) => counts.iter().copied().min(),
            Self::Range { min, .. } => Some(*min),
        }
    }

    /// Whether this advertises no step constraint at all (the `Default`).
    pub fn is_unconstrained(&self) -> bool {
        matches!(self, Self::Unconstrained)
    }
}

/// What a model supports — drives `validate()` and consumer UI.
///
/// `Default` is "supports nothing"; a model **turns on only what it offers** and defers every
/// other field, which is what makes adding a capability additive instead of a repo-wide compile
/// break (sc-19561):
///
/// ```
/// # use gen_core::Capabilities;
/// Capabilities {
///     supports_guidance: true,
///     max_count: 1,
///     ..Default::default()
/// }
/// # ;
/// ```
///
/// This is not a style preference. Every descriptor in the workspace constructs `Capabilities`
/// from another crate, so a new field lands in ~70 files at once unless each construction ends in
/// `..Default::default()` (or another base). `#[non_exhaustive]` cannot enforce it — that
/// attribute makes cross-crate construction impossible outright (E0639), and so does a private
/// field (E0451) — so the invariant is enforced by the
/// `capabilities_are_constructed_additively` integration test instead, which reads every `.rs`
/// file in the workspace and fails on any `Capabilities { .. }` literal with no base expression.
///
/// # Measured cost of adding a field (sc-19561 AC1)
///
/// Demonstrated rather than asserted, on 2026-08-15: a scratch `pub scratch_ac1_demonstration:
/// bool` was added here, both macOS lane sets were run, and it was removed again.
///
/// | | files touched | lines |
/// |---|---|---|
/// | at this revision | **1** (this one) | **1** |
/// | at the pre-conversion parent `2225b5026` | 70 + this one | ≥ 75 |
///
/// `cargo check --locked --all-targets -p sceneworks-gen-core -p sceneworks-gen-core-testkit
/// -p mlx-gen -p 'mlx-gen-*'` and `cargo check --locked --all-targets --features metal
/// -p 'candle-gen*' -p 'candle-audio*'` both exited **0 with zero errors and zero warnings** —
/// nothing outside this file needed to change. The parent row is the counterfactual: a
/// brace-matching parse of `2225b5026` finds **126 `Capabilities` literals, 74 of them
/// exhaustive, across 70 files**, and each would have raised its own `E0063`.
///
/// Reproduce the parent count with the same parser the guard uses — it is the guard, run over a
/// different revision.
#[derive(Clone, Debug, Default)]
pub struct Capabilities {
    pub supports_negative_prompt: bool,
    pub supports_guidance: bool,
    pub supports_true_cfg: bool,
    pub conditioning: Vec<ConditioningKind>,
    pub supports_lora: bool,
    pub supports_lokr: bool,
    pub samplers: Vec<&'static str>,
    pub schedulers: Vec<&'static str>,
    /// The guidance methods this model+backend honors (epic 7434), e.g. `["cfg", "cfg_rescale"]`.
    /// Empty ⇒ only the engine's implicit default path (no selectable guidance axis). Per-model-
    /// per-backend, like [`samplers`](Self::samplers) / [`schedulers`](Self::schedulers).
    pub supported_guidance_methods: Vec<&'static str>,
    pub min_size: u32,
    pub max_size: u32,
    pub max_count: u32,
    /// The denoise step counts this model can render (sc-19502 for the exact menu, sc-19559 for
    /// the range) — see [`StepSupport`] for the representation and the rejected alternatives.
    ///
    /// [`StepSupport::Unconstrained`] (the `Default`) means **no constraint**, deliberately the
    /// opposite polarity to [`samplers`](Self::samplers) / [`audio_voices`](Self::audio_voices),
    /// where empty means "no selectable surface, reject any explicit value". A model opts *in* to
    /// being constrained.
    ///
    /// Constrained ⇒ an explicit `req.steps` outside the menu/range is rejected by the shared
    /// floor. `None` on the request always passes: that is "use the model's baked default", not a
    /// step count anyone chose. A declared bound is therefore **never a default** — read
    /// [`StepSupport::floor`] as a bound only.
    ///
    /// # Why this is on `Capabilities` rather than in each provider's `validate`
    ///
    /// Distilled models bake their σ waypoints into training, so the step count is not a knob at
    /// all — LTX-2.3 runs a fixed 8-step stage-1 schedule (`STAGE1_SIGMAS`, 9 waypoints) and
    /// cannot honor any other count without going out-of-distribution. That constraint was
    /// previously written as an ad-hoc `if` inside ONE lane's `validate`, which produced the exact
    /// failure this field exists to prevent: `candle-gen-ltx` rejected `steps: 30` while
    /// `mlx-gen-ltx` never read `req.steps` at all and silently rendered its baked 8-step schedule
    /// anyway. Same model, same manifest entry, two behaviours, and the silent lane is the worse
    /// one — a user-facing control that quietly does nothing (the sc-11993 silent-coercion class).
    ///
    /// Hanging it on the advertised surface fixes both halves at once: the two lanes now share ONE
    /// enforcement site instead of two hand-maintained copies that already drifted, and a
    /// weights-free consumer holding a `Capabilities` can finally discover the constraint by
    /// inspection rather than by dispatching a job and reading the failure — the same
    /// discoverability argument [`size_floor`](Self::size_floor) makes for the size axis.
    ///
    /// The same argument carries the ceiling (sc-19559): SVD-XT refuses `steps > 200` in both
    /// lanes' `validate`, but nothing said so on the advertised surface, so the only way to learn
    /// the bound was to dispatch a job. A [`StepSupport::Range`] declaration makes the shared
    /// floor the enforcing site and the descriptor the discoverable one.
    ///
    /// # Conformance coupling
    ///
    /// `gen-core-testkit`'s `check_validate_honesty` validates at `profile.steps`, so a model
    /// declaring a constraint must align its conformance `Profile` — LTX pins 8. A declaration
    /// whose model default falls outside its own menu is a descriptor mistake
    /// caught by [`model_descriptor_errors`](crate::registry::model_descriptor_errors).
    pub supported_steps: StepSupport,
    pub mac_only: bool,
    /// How this model's size range is enforced — and therefore whether
    /// `min_size`/`max_size` are a bound a caller may rely on, plus any explicit-size grid the
    /// provider advertises.
    ///
    /// `Default` is [`SizeFloor::RangeChecked`], so every existing descriptor keeps
    /// today's behaviour with no edit. A provider whose `width`/`height == 0` means
    /// "resolve from the driving media" sets [`SizeFloor::ResolvedDownstream`]
    /// instead — SCAIL-2 sizes from its driving-video frames and uses
    /// [`SizeFloor::ResolvedDownstreamExplicitGrid`] to advertise that only explicit sizes must
    /// already sit on its tile grid.
    ///
    /// # Why this is on `Capabilities` rather than only in the provider
    ///
    /// The policy already exists, as
    /// [`validate_request_skip_size`](Self::validate_request_skip_size), and a
    /// provider selects it by *calling* that variant. Nothing **advertises** the
    /// choice, so a consumer holding a `Capabilities` cannot tell which floor a
    /// provider will apply, and a consumer that mirrors the shared floor to
    /// type-check requests before a load will reject a legal `0×0` on SCAIL-2.
    /// SceneWorks' Aether Studio hit exactly that and now carries a hand-built
    /// table of size-sentinel families; this field deletes it.
    pub size_floor: SizeFloor,
    // Audio surface (sc-12834) — read by the floor when a request carries
    // [`GenerationRequest::audio`]; all `Default` to the empty/no-audio surface so image/video
    // descriptors are untouched.
    /// Output sample rates (Hz) this model can synthesize. Empty ⇒ no selectable sample-rate
    /// surface: an explicit `audio.sample_rate` is rejected as [`Error::Unsupported`] (the
    /// same convention as [`samplers`](Self::samplers)); `None` on the request always passes
    /// (the model's native rate).
    pub audio_sample_rates: Vec<u32>,
    /// Longest audio clip (seconds) this model synthesizes. `None` ⇒ no advertised cap — an
    /// `audio.target_duration` is then bounded only by the shared duration sanity cap.
    pub max_audio_duration_secs: Option<f32>,
    /// Voice / speaker ids this model offers (TTS). Empty ⇒ no selectable voice surface: an
    /// explicit `audio.voice` is rejected as [`Error::Unsupported`].
    pub audio_voices: Vec<&'static str>,
    /// Language codes this model supports. Empty ⇒ no selectable language surface: an explicit
    /// `audio.language` is rejected as [`Error::Unsupported`].
    pub audio_languages: Vec<&'static str>,
    /// The prompted-audio-edit modes this model serves ([`AudioEditMode`]) — advertised so a
    /// consumer knows which edits the (edit-capable) generator accepts, and the shared floor
    /// rejects an [`Conditioning::AudioEdit`] whose `mode` is not listed as
    /// [`Error::Unsupported`] (sc-12847). Empty ⇒ the model is not an audio editor; combined with
    /// admitting [`ConditioningKind::AudioEdit`] in [`conditioning`](Self::conditioning) it names
    /// exactly the editable surface.
    pub audio_edit_modes: Vec<AudioEditMode>,
    /// Whether this model can emit **intermediate denoise previews** through
    /// [`GenerationRequest::preview`]. `Default` is `false`: existing and unwired providers are
    /// unsupported, so a consumer must not wait for preview frames from them. A provider sets this
    /// to `true` only when every generation route behind the descriptor forwards the request's
    /// [`PreviewSink`] into its denoise loop. That lets a weights-free consumer distinguish
    /// `false` (unsupported) from `true` with no frame received yet (supported, but not emitted yet,
    /// including before the first denoise step). This is advisory discoverability: it does not gate
    /// [`GenerationRequest::preview`] and [`PreviewSink`] itself does not report support.
    pub supports_preview: bool,
    /// Whether [`GenerationRequest::enhance_prompt`] changes the prompt consumed by this provider.
    ///
    /// This is weights-free discoverability for an optional semantic path, not a routing promise:
    /// consumers must still validate the complete request against the registered provider. `false`
    /// means the field is ignored or rejected, so a UI must not describe the toggle as effective.
    /// The flag is deliberately separate from text-LLM registration because prompt enhancement can
    /// be an internal model component rather than a standalone catalog provider.
    pub supports_prompt_enhancement: bool,
    /// Whether this model synthesizes audio **incrementally** through
    /// [`Generator::generate_streaming`] (sc-12846) — the opt-in signal for the realtime/streaming
    /// TTS path. `Default` is `false`: a non-streaming generator (every image/video model and the
    /// one-shot audio families) leaves it unset and its `generate_streaming` uses the default
    /// passthrough (a single terminal [`AudioChunk`]). A provider sets it
    /// `true` only when it genuinely emits multiple chunks before completion, so a consumer can read
    /// it to decide whether to drive the low-latency path and expect first-audio well before the full
    /// track. Advisory to the *shape* of the stream, not to correctness: the [`AudioChunk`] reassembly
    /// law (chunks concatenate to the returned track) holds for streaming and non-streaming providers
    /// alike.
    pub supports_streaming: bool,
    /// Whether this model renders a **multi-speaker dialogue script**
    /// ([`AudioParams::script`], sc-12848) — the opt-in signal for long-form / conversational
    /// multi-speaker TTS, mirroring [`supports_streaming`](Self::supports_streaming). `Default` is
    /// `false`: every non-dialogue model (image/video and the single-voice TTS / SFX / music audio
    /// families) leaves it unset, and the shared floor rejects a request carrying a
    /// [`script`](AudioParams::script) as the typed [`Error::Unsupported`]. A provider sets it `true`
    /// only when it genuinely assigns per-segment voices from the script's speaker labels; a consumer
    /// reads it to know whether a segmented script will be honored or must be rejected.
    pub supports_multi_speaker: bool,
    /// The largest number of **distinct** speaker labels a multi-speaker
    /// [`script`](AudioParams::script) may name (sc-12848). `None` ⇒ no advertised cap (bounded only
    /// by the model). Consulted only when [`supports_multi_speaker`](Self::supports_multi_speaker) is
    /// set; a script naming more than `max_speakers` distinct speakers is a range error
    /// ([`Error::Msg`], not a capability gap). `Default` is `None`.
    pub max_speakers: Option<u32>,
    /// Whether this model renders a **stateless multi-turn conversation history**
    /// ([`Conditioning::ConversationHistory`], sc-14150, path **A**) — the opt-in signal for
    /// context-aware conversational TTS carried entirely in the request, mirroring
    /// [`supports_multi_speaker`](Self::supports_multi_speaker) /
    /// [`supports_streaming`](Self::supports_streaming). `Default` is `false`: every non-conversational
    /// model leaves it unset and the shared floor rejects a request carrying a
    /// [`Conditioning::ConversationHistory`] as the typed [`Error::Unsupported`]. A provider that sets
    /// it `true` also advertises [`ConditioningKind::ConversationHistory`] in
    /// [`conditioning`](Self::conditioning) (the two are cross-checked by the descriptor conformance
    /// sweep). A consumer reads it to know whether a conversation will be honored or must be rejected.
    pub supports_conversation_history: bool,
    /// Whether this model can open a **stateful multi-turn conversational session**
    /// ([`Generator::open_conversation`] → [`ConversationSession`], sc-14150, path **B**) — the opt-in
    /// signal for the warm-KV real-time voice-agent path, mirroring
    /// [`supports_streaming`](Self::supports_streaming). `Default` is `false`: every model without
    /// cross-turn state leaves it unset, and [`Generator::open_conversation`]'s default returns the
    /// typed [`Error::Unsupported`]. A provider sets it `true` only when it genuinely keeps the model's
    /// live cross-turn state hot across `step`s (so a turn does not recompute the prefix). The session
    /// path must satisfy the A≡B equivalence law against the stateless
    /// [`supports_conversation_history`](Self::supports_conversation_history) render for the same
    /// conversation+seed; the `gen-core-testkit` `check_multi_turn` check enforces it. A model may
    /// advertise either path independently.
    pub supports_conversation_session: bool,
    /// On-the-fly quantization levels this engine offers (empty slice = none). Read by the worker's
    /// capability advertisement (sc-3723) instead of a hardcoded per-row flag. `Default` is `&[]`.
    pub supported_quants: &'static [Quant],
    /// Component-local numeric floors applied above the selected model tier. Empty means that a q4
    /// load is uniformly q4 across every packable component. This is a binding provider contract:
    /// callers use it for effective-tier labels and memory-evidence identity.
    pub component_precision_floors: &'static [ComponentPrecisionFloor],
    // Loader hints.
    pub supports_kv_cache: bool,
    pub requires_sigma_shift: bool,
    /// Whether this engine honors [`OffloadPolicy::Sequential`](crate::runtime::OffloadPolicy)
    /// (epic 10765, sc-11126). [`crate::OffloadPolicy::Sequential`] is *advisory* — a provider that
    /// has not
    /// wired the load→use→drop residency lifecycle silently treats it as `Resident` (never an error),
    /// which makes the fallback undiscoverable from the outside. This bit is the discovery signal: a
    /// consumer (worker / UI) reads it to know whether requesting `Sequential` will actually bound peak
    /// **footprint** on this engine or be a no-op. "Bound peak footprint" covers both shapes: an engine
    /// that holds several components co-resident (e.g. the Wan A14B MoE) bounds the peak **active** set
    /// by dropping the inactive ones, while an engine that already stages its active set (e.g. the dense
    /// Wan TI2V-5B) bounds the peak **retained cache / RSS** by `clear_cache`-flushing each dead
    /// component off-GPU instead of leaving it warm — the fit-gate models both via the staged
    /// max-single-component estimate. `Default` is `false` so an unwired engine does not over-advertise;
    /// a provider that drives the shared [`crate::runtime`] residency seam sets it `true`.
    pub supports_sequential_offload: bool,
    /// Whether every generation physically stages eligible heavyweight components through a
    /// load/use/drop lifecycle even when no selectable [`OffloadPolicy::Sequential`](crate::runtime::OffloadPolicy)
    /// control is requested. This is independent of [`supports_sequential_offload`](Self::supports_sequential_offload):
    /// MLX Wan, for example, stages phases unconditionally and also exposes a stronger selectable
    /// Sequential mode that flushes dead allocator cache or narrows expert residency. `Default` is
    /// `false`; this is static descriptor truth and must not be copied into request evidence
    /// composition.
    pub unconditionally_engages_staged_residency: bool,
    /// The typed **execution domains** this provider honours on
    /// [`GenerationRequest::memory`] (sc-18317): graph-evaluation cadence, FFN chunking, and CFG
    /// batching. See [`crate::execution_domains`].
    ///
    /// `Default` declares every domain [`ExecutionValueDomain::Unsupported`](crate::ExecutionValueDomain::Unsupported),
    /// which is the fail-closed state — every existing descriptor keeps today's behaviour with no
    /// edit, and a request that names a knob the provider does not implement is a typed
    /// [`Error::Unsupported`] at the shared floor instead of a silently different execution schedule.
    ///
    /// # Why it lives on `Capabilities` rather than on `MemoryProviderContract`
    ///
    /// These knobs are not ladder rungs, and the two surfaces have different coverage: the memory
    /// contract is adopted per *memory-strategy* provider (the Candle Kolors lane, for one, has no
    /// contract at all yet still owns a CFG-batching convention), whereas every generator has a
    /// `Capabilities` and every generator's `validate` runs the shared floor. Declaring here is
    /// therefore the only placement where the refusal cannot be forgotten by a provider — the same
    /// reason [`supports_multi_speaker`](Self::supports_multi_speaker) and the audio surface live
    /// here. A planner reading the descriptor sees the domains beside the rest of the surface.
    pub execution: ExecutionSurface,
    /// The typed **approximate capabilities** this provider implements on
    /// [`GenerationRequest::approximation`] (sc-18322): mechanisms that make a render cheaper by
    /// changing its result. See [`crate::approximation`].
    ///
    /// `Default` declares the mechanism absent *and* binds no quality-characterization artifact
    /// family, so every existing descriptor keeps today's behaviour with no edit and a request that
    /// selects an approximation is a typed [`Error::Unsupported`] at the shared floor.
    ///
    /// # Why this is not part of [`execution`](Self::execution)
    ///
    /// [`ExecutionSurface`]'s contract is that every domain it carries is bit-identical or
    /// numerically equivalent, which is what lets a planner select one freely. An approximation is
    /// the negation of that promise. Folding the two surfaces together would leave a consumer unable
    /// to tell, from the descriptor, whether selecting a declared knob can move the pixels — so the
    /// two live side by side, and the equivalence class stays readable off the field name.
    ///
    /// A mechanism declared here is **implemented but not selectable**: see
    /// [`ApproximationSurface::is_selectable`](crate::ApproximationSurface::is_selectable), which is
    /// `false` for every provider until the terminal measurement campaign defines a
    /// characterization artifact family.
    pub approximation: ApproximationSurface,
}

/// The one typed refusal for an **adapter-bearing** request against a descriptor whose advertised
/// surface does not support adapters (sc-21483, epic 11037 E6).
///
/// The motivating case is an imported-model route whose binding declares `inherit_adapters = false`:
/// [`crate::registry::ProviderRegistry::imported_model_descriptor`] withdraws
/// [`Capabilities::supports_lora`] / [`Capabilities::supports_lokr`] from that route's descriptor,
/// and this is what makes the withdrawal *observable*. Without it a withdrawn capability and an
/// ignored adapter look identical from the outside: the load succeeds, the adapter is dropped on the
/// floor, and the user gets an un-adapted render with no error — the sc-11993 silent-coercion class.
///
/// [`Error::Unsupported`] (never [`Error::Msg`]) so a consumer can tell a capability gap apart from a
/// bad adapter file.
pub fn reject_unsupported_adapters(
    id: &str,
    capabilities: &Capabilities,
    adapter_count: usize,
) -> Result<()> {
    if adapter_count == 0 || capabilities.supports_lora || capabilities.supports_lokr {
        return Ok(());
    }
    Err(Error::Unsupported(format!(
        "{id}: this model route does not inherit adapters, so the {adapter_count} selected \
         LoRA/LoKr adapter(s) cannot be applied; it is refused rather than silently rendered \
         un-adapted"
    )))
}

/// Generous upper sanity caps for the unbounded counter knobs (F-004). Not model limits — each model
/// layers a tighter, better-messaged bound in its own `validate` (e.g. kolors caps `steps` at its
/// train-timestep count); these sit ABOVE any real model bound so they only reject a pathological
/// value (`u32::MAX` steps/frames) that would otherwise launch an effectively-unbounded, cancel-only
/// run — never preempting a model's own check.
const MAX_STEPS: u32 = 100_000;
const MAX_FRAMES: u32 = 1_000_000;
const MAX_FPS: u32 = 100_000;
const MAX_DURATION_SECS: f32 = 1_000_000.0;
/// Shared hard ceiling for autoregressive prompt enhancement. Providers may choose a lower cap.
pub const MAX_ENHANCE_TOKENS: u32 = 2_048;
/// Shared temperature range for prompt enhancement sampling.
pub const MAX_ENHANCE_TEMPERATURE: f32 = 2.0;

impl Capabilities {
    /// Static descriptor view of staged-residency availability. Unconditional physical staging wins
    /// the derived classification when both independent bits are true; callers that need to know
    /// whether the stronger selectable control also exists must inspect
    /// [`supports_sequential_offload`](Self::supports_sequential_offload) separately.
    pub const fn staged_residency_availability(&self) -> StagedResidencyAvailability {
        if self.unconditionally_engages_staged_residency {
            StagedResidencyAvailability::UnconditionallyEngaged
        } else if self.supports_sequential_offload {
            StagedResidencyAvailability::Selectable
        } else {
            StagedResidencyAvailability::Absent
        }
    }

    /// Whether this model accepts the given conditioning kind.
    pub fn accepts(&self, kind: ConditioningKind) -> bool {
        self.conditioning.contains(&kind)
    }

    /// Reject a request that violates the **advertised** capability surface — the model-agnostic
    /// checks every `Generator::validate` shares, so a descriptor cannot promise something
    /// `validate` then silently ignores at runtime:
    ///
    /// - `count` within `1..=max_count`,
    /// - `steps` (when supplied) must be `>= 1` — an explicit `0` would run a 0-step denoise and
    ///   VAE-decode pure noise (F-007); the schedule builders' `.max(1)` clamps document this as the
    ///   real floor,
    /// - `width`/`height` within `min_size..=max_size` and, when advertised by [`SizeFloor`], on the
    ///   required explicit-size grid,
    /// - `negative_prompt` / `guidance` / `true_cfg` only when the matching `supports_*` flag is set,
    ///   and `guidance` / `true_cfg` must be finite (a NaN would poison the guidance math, F-053),
    /// - `sampler` / `scheduler` / `guidance_method` (when supplied) must name an advertised entry,
    /// - every `conditioning` entry must be an [`accepts`](Self::accepts)ed kind,
    /// - the [`audio`](GenerationRequest::audio) sub-block's supplied values must sit inside the
    ///   advertised audio surface (voice / language / sample-rate membership,
    ///   `target_duration` within `(0, `[`max_audio_duration_secs`](Self::max_audio_duration_secs)`]`,
    ///   positive `bpm` — sc-12834); and a multi-speaker [`script`](AudioParams::script) only when
    ///   [`supports_multi_speaker`](Self::supports_multi_speaker) is set, within any advertised
    ///   [`max_speakers`](Self::max_speakers) cap (sc-12848),
    ///
    /// Capability-gap rejections (unsupported negative_prompt / guidance / true_cfg / sampler /
    /// scheduler / guidance_method / conditioning) return the typed [`Error::Unsupported`] so a
    /// consumer (SceneWorks worker / candle gating) can distinguish "this backend can't do that"
    /// from a range violation or generic failure (F-008); malformed-value rejections (count/size/
    /// steps out of range, non-finite guidance) return [`Error::Msg`].
    ///
    /// `id` is the model's descriptor id, used in error messages. Model-specific constraints — an
    /// empty-prompt rejection, a size-alignment rule not advertised by [`SizeFloor`], frame-count
    /// divisibility, sampler→solver mapping — are layered on top by each model's own `validate`;
    /// this is the shared floor, not a replacement for them.
    pub fn validate_request(&self, id: &str, req: &GenerationRequest) -> Result<()> {
        // The size check is gated on the ADVERTISED floor rather than on the caller picking the
        // right entry point — that is what lets a weights-free consumer hold a `Capabilities` and
        // get the right answer without knowing which variant the provider calls.
        //
        // `ResolvedDownstream` exempts THE SENTINEL, not every size. Its doc says the range check
        // "does not apply to it", meaning `0x0` specifically, and the scoping is load-bearing: an
        // earlier revision read the variant as a blanket opt-out and disabled the range check
        // outright. That silently deleted SCAIL-2's explicit-size rejection — on main its
        // `pipeline.rs` range-checks whenever dimensions are given and only skips for the sentinel,
        // with a comment saying exactly that — so an explicit 16x16 against declared bounds of
        // 32..=1280 stopped being rejected, with nothing downstream to catch it.
        //
        // Both dimensions must be zero. A half-sentinel (`0x512`) is not the "resolve from the
        // driving media" convention, it is a malformed request, and it must still be rejected.
        let is_sentinel = req.width == 0 && req.height == 0;
        let check_size = match self.size_floor {
            SizeFloor::RangeChecked | SizeFloor::RangeCheckedOnGrid { .. } => true,
            SizeFloor::ResolvedDownstream | SizeFloor::ResolvedDownstreamExplicitGrid { .. } => {
                !is_sentinel
            }
        };
        self.validate_request_inner(id, req, check_size)?;

        if !is_sentinel {
            if let Some(multiple) = self.size_floor.explicit_size_multiple() {
                if multiple == 0 {
                    return Err(Error::Msg(format!(
                        "{id}: descriptor advertises an invalid explicit-size multiple of 0"
                    )));
                }
                if !req.width.is_multiple_of(multiple) || !req.height.is_multiple_of(multiple) {
                    return Err(Error::Msg(format!(
                        "{id}: width/height must be multiples of {multiple} (got {}×{})",
                        req.width, req.height
                    )));
                }
            }
        }
        Ok(())
    }

    /// The shared floor **minus the advertised spatial checks** (range and any explicit-size grid) —
    /// for providers with a "match the driving-media size" convention (`width`/`height == 0` is a
    /// resolve-downstream sentinel, e.g. SCAIL-2 sizing from the driving-video frames), where those
    /// checks would wrongly reject the sentinel. Every non-spatial floor check still runs
    /// unconditionally: count / steps / frame / fps / duration caps, negative-prompt / guidance /
    /// true_cfg support gating, finiteness (F-053), sampler / scheduler / guidance_method membership,
    /// and the conditioning allowlist. Prefer [`validate_request`](Self::validate_request) with a
    /// [`SizeFloor::ResolvedDownstream`] variant so non-sentinel requests retain advertised spatial
    /// validation. A provider that calls this lower-level escape hatch must range-check its resolved
    /// size itself (F-158).
    pub fn validate_request_skip_size(&self, id: &str, req: &GenerationRequest) -> Result<()> {
        self.validate_request_inner(id, req, false)
    }

    /// The audio-aware floor (sc-12834): the shared floor **minus the width/height range check**,
    /// for pure-audio models (`Modality::Audio`) where the request's `width`/`height` are unused
    /// and the visual size range would wrongly reject every request. Parallel to
    /// [`validate_request_skip_size`](Self::validate_request_skip_size) — every other floor check
    /// still runs unconditionally: count / steps / frame / fps / duration caps, capability gating,
    /// finiteness (F-053, including the [`AudioParams`] floats), sampler / scheduler /
    /// guidance_method membership, the conditioning allowlist, and the audio-surface checks
    /// (voice / language / sample-rate membership, `target_duration` vs
    /// [`max_audio_duration_secs`](Self::max_audio_duration_secs)).
    pub fn validate_request_audio(&self, id: &str, req: &GenerationRequest) -> Result<()> {
        self.validate_request_inner(id, req, false)
    }

    /// The **approximate-capability plan** for one request — what a provider's denoise will actually
    /// run (sc-18322).
    ///
    /// The same call the shared floor already made, exposed so a provider consumes the resolved
    /// [`ApproximationPlan`] instead of re-reading
    /// [`GenerationRequest::approximation`] and re-deriving the policy. There is therefore exactly one
    /// place in the workspace that turns a request into a plan, and it is the place that refuses.
    ///
    /// Returns [`ApproximationPlan::Exact`] for every request today — see
    /// [`crate::approximation`] for why that is the designed state, not a gap.
    pub fn approximation_plan(
        &self,
        id: &str,
        req: &GenerationRequest,
    ) -> Result<ApproximationPlan> {
        self.approximation.resolve(id, req.approximation.as_ref())
    }

    /// Shared implementation of the floor. `check_size` gates only the size-range check so the
    /// auto-size path ([`validate_request_skip_size`](Self::validate_request_skip_size)) still runs
    /// every other check; the public [`validate_request`](Self::validate_request) passes `true`.
    fn validate_request_inner(
        &self,
        id: &str,
        req: &GenerationRequest,
        check_size: bool,
    ) -> Result<()> {
        // Footgun guard (F-084): a descriptor that enables a capability but leaves max_count/max_size
        // at the `Default` 0 would reject EVERY request with a confusing "out of range 0..=0". A real
        // model always sets non-zero bounds, so catch the descriptor mistake in debug/test builds.
        // `max_size` is only asserted when the size check runs: on the size-skipping floors the size
        // bounds are legitimately unused (a pure-audio descriptor leaves them at 0 — sc-12834).
        debug_assert!(
            self.max_count > 0 && (!check_size || self.max_size > 0),
            "{id}: Capabilities max_count={} max_size={} left at Default 0 — descriptor forgot its bounds",
            self.max_count,
            self.max_size
        );
        if let Some(memory) = req.memory {
            match (
                memory.calibration_fault_harness_authorized,
                memory.calibration_error_phase,
            ) {
                (false, None) | (true, Some(_)) => {}
                (false, Some(_)) => {
                    return Err(Error::Unsupported(format!(
                        "{id}: calibration fault injection requires explicit harness authorization"
                    )));
                }
                (true, None) => {
                    return Err(Error::Unsupported(format!(
                        "{id}: calibration fault harness authorization requires an error phase"
                    )));
                }
            }
        }
        // Typed execution domains (sc-18317). Gated here, on the shared floor, for the same reason
        // the audio surface is: a per-provider check is a check a provider can forget, and a
        // forgotten one means the knob is silently ignored — the exact defect the typed domains
        // exist to remove. Unset fields validate vacuously, so this is inert for every request that
        // does not select an execution schedule.
        self.execution.validate(id, req.memory.as_ref())?;
        // Approximate capabilities (sc-18322). Gated on the same shared floor and for the same
        // reason, but with a stronger conclusion: `resolve` refuses EVERY approximate selection
        // today, because no provider can bind a quality-characterization artifact family (the
        // binding's payload type is uninhabited). Resolving here — and discarding the plan — is what
        // makes the refusal unforgettable; a provider that wants to *execute* an approximation calls
        // `Capabilities::approximation_plan` and gets the same answer from the same code.
        // An absent or empty selection resolves to `Exact` vacuously, so this is inert for every
        // request that does not ask for an approximation.
        self.approximation
            .resolve(id, req.approximation.as_ref())
            .map(|_plan| ())?;
        if req.count == 0 || req.count > self.max_count {
            return Err(Error::Msg(format!(
                "{id}: count {} out of range 1..={}",
                req.count, self.max_count
            )));
        }
        // An explicit `steps: Some(0)` runs a 0-step denoise and VAE-decodes pure scaled noise; the
        // schedule builders' `.max(1)` clamps (sampling.rs) cite this as the real floor, so enforce it
        // here rather than letting it fall through to ad-hoc per-provider guards (F-007). `None` falls
        // back to each model's default; a *derived* 0 from img2img `int(steps·strength)` is a separate,
        // legitimate no-op handled downstream.
        if req.steps == Some(0) {
            return Err(Error::Msg(format!(
                "{id}: steps must be >= 1 (an explicit 0 renders undenoised noise)"
            )));
        }
        // Upper sanity caps (F-004): the floor enforced `steps >= 1` but no ceiling, so
        // `steps: Some(u32::MAX)` (and the video frame/counter fields) validated and launched an
        // effectively-unbounded, cancel-only-recoverable run. These are deliberately generous — far
        // above any real request (LTX's frame ceiling is 1025, the priciest image trajectories ~50–100
        // steps) — a footgun guard against a pathological/garbage value, not a model-specific limit
        // (each model layers its own tighter bound in its `validate`).
        if let Some(steps) = req.steps {
            if steps > MAX_STEPS {
                return Err(Error::Msg(format!(
                    "{id}: steps {steps} exceeds the sanity cap {MAX_STEPS}"
                )));
            }
        }
        // The advertised step surface — an exact menu (sc-19502) or an inclusive range (sc-19559).
        // Distinct from the sanity cap above: that one is a footgun guard every model shares, this
        // one is the model's own declaration, either that the step count is not a knob at all or
        // that its engine bound is `min..=max`. `Unconstrained` is the `Default`, so this is inert
        // for every model that does not opt in (see `Capabilities::supported_steps` for why the
        // polarity is inverted relative to `samplers`).
        //
        // **Reject, never snap to the nearest legal count.** Quietly rewriting `steps: 30` to 8
        // would deliver a render the caller did not ask for with no error and no signal — the
        // silent-coercion class — and it is the precise defect this replaces: `mlx-gen-ltx` used to
        // ignore `req.steps` outright and render its baked schedule regardless.
        //
        // Only an EXPLICIT count is judged. `None` means "the model picks", so there is nothing to
        // refuse; that is what keeps the common path (omit `steps`, get the baked schedule) working
        // and is why a distilled model is still usable without the caller knowing its magic number.
        if let Some(steps) = req.steps {
            if !self.supported_steps.admits(steps) {
                return Err(Error::Msg(match &self.supported_steps {
                    // Unreachable: `admits` is always true here, but spelling the arm keeps the
                    // match exhaustive without an `unreachable!` on a request path.
                    StepSupport::Unconstrained => format!("{id}: steps {steps} is not supported"),
                    // Singular reads as the original per-provider message it replaces ("a fixed
                    // 8-step schedule"), because one legal count is the case that actually ships;
                    // the plural arm keeps the set readable rather than emitting "count(s)".
                    StepSupport::Exact(counts) => {
                        let schedule = match counts.as_slice() {
                            [only] => format!("a fixed {only}-step schedule"),
                            many => format!(
                                "a fixed schedule ({} steps only)",
                                many.iter()
                                    .map(u32::to_string)
                                    .collect::<Vec<_>>()
                                    .join(" or ")
                            ),
                        };
                        format!(
                            "{id}: this distilled model runs {schedule} and cannot honor \
                             steps={steps}; omit `steps` to use the baked schedule."
                        )
                    }
                    // The advertised RANGE (sc-19559). Distinct from the shared sanity cap above:
                    // that one is a footgun guard every model shares, this is the model's own
                    // engine bound — SVD-XT's 200, Kolors' 1100 train timesteps — which used to
                    // live only inside a provider `validate` and so was undiscoverable
                    // weights-free.
                    StepSupport::Range { min, max } => format!(
                        "{id}: steps {steps} is outside this model's supported range \
                         {min}..={max}; omit `steps` to use the model's default."
                    ),
                }));
            }
        }
        if let Some(frames) = req.frames {
            if frames > MAX_FRAMES {
                return Err(Error::Msg(format!(
                    "{id}: frames {frames} exceeds the sanity cap {MAX_FRAMES}"
                )));
            }
        }
        if let Some(fps) = req.fps {
            if fps > MAX_FPS {
                return Err(Error::Msg(format!(
                    "{id}: fps {fps} exceeds the sanity cap {MAX_FPS}"
                )));
            }
        }
        if let Some(d) = req.duration {
            // `d` is finiteness-checked below; here only the upper bound (a NaN compares false and is
            // caught by `ensure_finite_floats`).
            if d > MAX_DURATION_SECS {
                return Err(Error::Msg(format!(
                    "{id}: duration {d}s exceeds the sanity cap {MAX_DURATION_SECS}s"
                )));
            }
        }
        // Audio sub-block (sc-12834): gate the supplied values against the advertised audio
        // surface. Membership gaps (voice / language / sample rate) are capability gaps →
        // typed `Error::Unsupported` (F-008); malformed values (non-positive / over-cap
        // duration, non-positive bpm) are range errors → `Error::Msg`. Finiteness of the audio
        // floats is enforced by `ensure_finite_floats` below (a NaN compares false here and
        // falls through to that guard, like `duration`).
        if let Some(audio) = &req.audio {
            if let Some(d) = audio.target_duration {
                if d <= 0.0 {
                    return Err(Error::Msg(format!(
                        "{id}: audio.target_duration must be > 0 (got {d}s)"
                    )));
                }
                if d > MAX_DURATION_SECS {
                    return Err(Error::Msg(format!(
                        "{id}: audio.target_duration {d}s exceeds the sanity cap {MAX_DURATION_SECS}s"
                    )));
                }
                if let Some(cap) = self.max_audio_duration_secs {
                    if d > cap {
                        return Err(Error::Msg(format!(
                            "{id}: audio.target_duration {d}s exceeds the supported maximum {cap}s"
                        )));
                    }
                }
            }
            if let Some(bpm) = audio.bpm {
                if bpm <= 0.0 {
                    return Err(Error::Msg(format!(
                        "{id}: audio.bpm must be > 0 (got {bpm})"
                    )));
                }
            }
            if let Some(sr) = audio.sample_rate {
                if !self.audio_sample_rates.contains(&sr) {
                    return Err(Error::Unsupported(format!(
                        "{id}: unsupported audio.sample_rate {sr} (supported: {:?})",
                        self.audio_sample_rates
                    )));
                }
            }
            if let Some(v) = &audio.voice {
                if !self.audio_voices.contains(&v.as_str()) {
                    return Err(Error::Unsupported(format!(
                        "{id}: unsupported audio.voice {v:?} (supported: {:?})",
                        self.audio_voices
                    )));
                }
            }
            if let Some(l) = &audio.language {
                if !self.audio_languages.contains(&l.as_str()) {
                    return Err(Error::Unsupported(format!(
                        "{id}: unsupported audio.language {l:?} (supported: {:?})",
                        self.audio_languages
                    )));
                }
            }
            // Multi-speaker dialogue script (sc-12848): a script sent to a model that does not
            // advertise `supports_multi_speaker` is a capability gap → typed `Error::Unsupported`
            // (the same convention `audio.voice` / streaming use), so a single-voice model can never
            // silently read only the first segment. When supported, the script must be non-empty (an
            // empty script is a malformed request → `Error::Msg`), stay within any advertised
            // `max_speakers` cap (range → `Error::Msg`), and — for a model with a **closed** voice
            // surface (a non-empty `audio_voices`) — name only advertised voices (gap →
            // `Error::Unsupported`, exactly like `audio.voice`). A dialogue model with opaque speaker
            // labels advertises an empty voice surface and is not per-label gated here; each model
            // still layers per-segment text checks (empty text) in its own `validate`.
            if let Some(script) = &audio.script {
                if !self.supports_multi_speaker {
                    return Err(Error::Unsupported(format!(
                        "{id}: a multi-speaker audio.script is not supported"
                    )));
                }
                if script.is_empty() {
                    return Err(Error::Msg(format!(
                        "{id}: audio.script is empty — a multi-speaker script must carry at least \
                         one segment"
                    )));
                }
                if let Some(max) = self.max_speakers {
                    let mut labels: Vec<&str> =
                        script.iter().filter_map(|s| s.speaker.as_deref()).collect();
                    labels.sort_unstable();
                    labels.dedup();
                    if labels.len() as u32 > max {
                        return Err(Error::Msg(format!(
                            "{id}: audio.script names {} distinct speakers, above the supported \
                             maximum {max}",
                            labels.len()
                        )));
                    }
                }
                if !self.audio_voices.is_empty() {
                    for seg in script {
                        if let Some(sp) = &seg.speaker {
                            if !self.audio_voices.contains(&sp.as_str()) {
                                return Err(Error::Unsupported(format!(
                                    "{id}: unsupported audio.script speaker {sp:?} (supported \
                                     voices: {:?})",
                                    self.audio_voices
                                )));
                            }
                        }
                    }
                }
            }
        }
        if check_size
            && (req.width < self.min_size
                || req.height < self.min_size
                || req.width > self.max_size
                || req.height > self.max_size)
        {
            return Err(Error::Msg(format!(
                "{id}: size {}x{} outside supported range {}..={}",
                req.width, req.height, self.min_size, self.max_size
            )));
        }
        if req.negative_prompt.is_some() && !self.supports_negative_prompt {
            return Err(Error::Unsupported(format!(
                "{id}: negative prompts are not supported"
            )));
        }
        if req.guidance.is_some() && !self.supports_guidance {
            return Err(Error::Unsupported(format!(
                "{id}: guidance is not supported"
            )));
        }
        if req.true_cfg.is_some() && !self.supports_true_cfg {
            return Err(Error::Unsupported(format!(
                "{id}: true_cfg is not supported"
            )));
        }
        if req.enhance_prompt && !self.supports_prompt_enhancement {
            return Err(Error::Unsupported(format!(
                "{id}: prompt enhancement is not supported"
            )));
        }
        if let Some(tokens) = req.enhance_max_tokens {
            if !req.enhance_prompt {
                return Err(Error::Msg(format!(
                    "{id}: enhance_max_tokens requires enhance_prompt=true"
                )));
            }
            if tokens == 0 || tokens > MAX_ENHANCE_TOKENS {
                return Err(Error::Msg(format!(
                    "{id}: enhance_max_tokens {tokens} outside supported range 1..={MAX_ENHANCE_TOKENS}"
                )));
            }
        }
        if let Some(temperature) = req.enhance_temperature {
            if !temperature.is_finite() {
                return Err(Error::Msg(format!(
                    "enhance_temperature must be finite (got {temperature})"
                )));
            }
            if !req.enhance_prompt {
                return Err(Error::Msg(format!(
                    "{id}: enhance_temperature requires enhance_prompt=true"
                )));
            }
            if !(0.0..=MAX_ENHANCE_TEMPERATURE).contains(&temperature) {
                return Err(Error::Msg(format!(
                    "{id}: enhance_temperature {temperature} outside supported range 0..={MAX_ENHANCE_TEMPERATURE}"
                )));
            }
        }
        // A non-finite guidance / true_cfg / eta / momentum / strength / control_scale / … would flow
        // into the CFG combine, scheduler shift, or conditioning math and NaN-poison the run (a NaN
        // passes `x > 1.0`-style checks silently). The finiteness guard is centralized on the request
        // so every `Option<f32>` knob — including ones added after F-053 — inherits it by construction
        // (F-053 / F-001). `id`-prefixing is dropped from the message here; the field name is enough.
        //
        // Multi-region bounds get an **indexed** pass first (sc-14549). `first_nonfinite_float`
        // returns a `&'static str` key, so the multi-region arm can only say
        // `conditioning.audio_edit_regions.regions.start_secs` — it cannot say *which* region. That
        // is precisely the failure mode the `Vec`-defeats-the-destructure gate exists to close, and
        // the caller of a ten-region repaint needs the index to act on the message. The two
        // range guards further down (`start < 0`, `end <= start`) do name the index but never see a
        // NaN: both comparisons evaluate `false` for NaN, so without this pass a non-finite bound is
        // caught only by the index-free key below.
        //
        // It runs **before** `ensure_finite_floats` because that call returns on the first
        // non-finite float anywhere in the request; placed after it, this loop would be
        // unreachable. `ensure_finite_floats` stays intact as the backstop — it is what providers
        // with a bespoke `validate` (flux1's IP-Adapter carve-out, mlx-gen-flux) call directly
        // without ever entering `validate_request`, so both layers are load-bearing.
        for c in &req.conditioning {
            if let Conditioning::AudioEditRegions { regions, .. } = c {
                for (i, r) in regions.iter().enumerate() {
                    if !r.start_secs.is_finite() {
                        return Err(Error::Msg(format!(
                            "{id}: multi-region audio edit region {i} start must be finite (got {})",
                            r.start_secs
                        )));
                    }
                    if let Some(end) = r.end_secs {
                        if !end.is_finite() {
                            return Err(Error::Msg(format!(
                                "{id}: multi-region audio edit region {i} end must be finite (got \
                                 {end})"
                            )));
                        }
                    }
                }
            }
        }
        req.ensure_finite_floats()?;
        if let Some(s) = &req.sampler {
            if !self.samplers.contains(&s.as_str()) {
                return Err(Error::Unsupported(format!(
                    "{id}: unsupported sampler {s:?} (supported: {:?})",
                    self.samplers
                )));
            }
        }
        if let Some(s) = &req.scheduler {
            if !self.schedulers.contains(&s.as_str()) {
                return Err(Error::Unsupported(format!(
                    "{id}: unsupported scheduler {s:?} (supported: {:?})",
                    self.schedulers
                )));
            }
        }
        if let Some(m) = &req.guidance_method {
            if !self.supported_guidance_methods.contains(&m.as_str()) {
                return Err(Error::Unsupported(format!(
                    "{id}: unsupported guidance_method {m:?} (supported: {:?})",
                    self.supported_guidance_methods
                )));
            }
        }
        for c in &req.conditioning {
            let kind = c.kind();
            if !self.accepts(kind) {
                return Err(Error::Unsupported(format!(
                    "{id}: {kind:?} conditioning is not supported"
                )));
            }
        }
        // Audio-edit sub-surface (sc-12847): once the `AudioEdit` kind is admitted above, the
        // requested *mode* must sit inside the advertised [`audio_edit_modes`] (an unlisted mode is
        // a capability gap → typed `Error::Unsupported`, like an unadvertised sampler), and the
        // region — when present — must be well-formed (`start >= 0`, `end > start`). Region
        // finiteness is already enforced by `ensure_finite_floats` above; clip-bound checks (region
        // inside the source duration) belong to the provider, which knows the clip length.
        for c in &req.conditioning {
            if let Conditioning::AudioEdit { mode, region, .. } = c {
                if !self.audio_edit_modes.contains(mode) {
                    return Err(Error::Unsupported(format!(
                        "{id}: unsupported audio edit mode {mode:?} (supported: {:?})",
                        self.audio_edit_modes
                    )));
                }
                if let Some(r) = region {
                    if r.start_secs < 0.0 {
                        return Err(Error::Msg(format!(
                            "{id}: audio edit region start {}s must be >= 0",
                            r.start_secs
                        )));
                    }
                    if let Some(end) = r.end_secs {
                        if end <= r.start_secs {
                            return Err(Error::Msg(format!(
                                "{id}: audio edit region end {end}s must be > start {}s",
                                r.start_secs
                            )));
                        }
                    }
                }
            }
        }
        // Multi-region audio edit (sc-14549). The `AudioEditRegions` **kind** is already gated by
        // the allowlist above — a provider that has not advertised it never reaches here, which is
        // the whole reason this variant carries its own kind instead of a capability flag. What is
        // left is the shape of the list itself, and the mode, which deliberately reuses the same
        // `audio_edit_modes` surface the single-region path uses rather than introducing a second
        // one to drift from it.
        //
        // Region-bound finiteness is enforced above — for **every** region, not just the first — by
        // the indexed pass that precedes `ensure_finite_floats`, with `ensure_finite_floats` itself
        // as the backstop for callers that bypass `validate_request`. The two range guards below
        // therefore only ever see finite bounds, which is what makes `start < 0` and `end <= start`
        // safe to write as plain comparisons. Clip-bound and latent-collapse checks stay with the
        // provider.
        for c in &req.conditioning {
            if let Conditioning::AudioEditRegions { mode, regions, .. } = c {
                if !self.audio_edit_modes.contains(mode) {
                    return Err(Error::Unsupported(format!(
                        "{id}: unsupported audio edit mode {mode:?} (supported: {:?})",
                        self.audio_edit_modes
                    )));
                }
                if regions.is_empty() {
                    return Err(Error::Msg(format!(
                        "{id}: multi-region audio edit carries no regions — it must name at least \
                         one span to regenerate (a whole-clip restyle is ReferenceAudio)"
                    )));
                }
                for (i, r) in regions.iter().enumerate() {
                    if r.start_secs < 0.0 {
                        return Err(Error::Msg(format!(
                            "{id}: multi-region audio edit region {i} start {}s must be >= 0",
                            r.start_secs
                        )));
                    }
                    // `end_secs: None` means "to the end of the clip", which is only well-defined
                    // for a *final* region — and region order is not significant here, so "final"
                    // is not well-defined. Refused outright rather than left ambiguous; the
                    // single-region `AudioEdit` keeps the `None` shorthand.
                    let Some(end) = r.end_secs else {
                        return Err(Error::Msg(format!(
                            "{id}: multi-region audio edit region {i} has no end — every region \
                             must state an explicit end_secs, because region order is not \
                             significant so \"to the end of the clip\" has no unambiguous meaning \
                             here (use a single-region AudioEdit for that)"
                        )));
                    };
                    if end <= r.start_secs {
                        return Err(Error::Msg(format!(
                            "{id}: multi-region audio edit region {i} end {end}s must be > start \
                             {}s",
                            r.start_secs
                        )));
                    }
                }
            }
        }
        // Video→audio (Foley) sync conditioning (sc-13436): once the `VideoSync` kind is admitted by
        // the allowlist above (the un-admitted case is already the typed `Error::Unsupported`, F-008),
        // the clip must actually carry frames — an empty `frames` leaves the audio decoder nothing to
        // condition on, a malformed request → `Error::Msg`. The frame rate rides `req.fps`, so there
        // is nothing further to gate on the variant here; per-model frame-count / resolution bounds are
        // layered by the provider's own `validate`.
        for c in &req.conditioning {
            if let Conditioning::VideoSync { frames } = c {
                if frames.is_empty() {
                    return Err(Error::Msg(format!(
                        "{id}: VideoSync conditioning carries no frames — a video→audio clip must \
                         have at least one frame"
                    )));
                }
            }
        }
        // A reference clip (sc-17149) must carry frames, and must carry the rate they were shot at.
        // The rate is checked for **positivity** here and not only finiteness, unlike every other
        // conditioning float the floor owns: those feed denoise math where 0.0 is a meaningful
        // (inert) value, whereas a rate of 0 or below has no reading at all — it makes the resample
        // stride undefined or negative, and the frames a model would then read are arbitrary rather
        // than merely unweighted. Per-model rate *bounds* stay with the provider, which knows what
        // it resamples onto.
        for c in &req.conditioning {
            if let Conditioning::ReferenceVideo { frames, fps, .. } = c {
                if frames.is_empty() {
                    return Err(Error::Msg(format!(
                        "{id}: ReferenceVideo conditioning carries no frames — a reference clip \
                         must have at least one frame"
                    )));
                }
                if !fps.is_finite() || *fps <= 0.0 {
                    return Err(Error::Msg(format!(
                        "{id}: ReferenceVideo conditioning declares a frame rate of {fps} — a \
                         reference clip is resampled from the rate it carries, so the rate must be \
                         a positive finite number"
                    )));
                }
            }
        }
        // Multi-turn conversation history (sc-14150, path A): a conversation sent to a model that
        // does not advertise `supports_conversation_history` is a capability gap → typed
        // `Error::Unsupported` (the same convention `supports_multi_speaker` / streaming use), so a
        // single-turn model can never silently render only the last turn. The allowlist above already
        // rejects the kind when it is not admitted; this keyed check gives the specific message and is
        // authoritative when a descriptor advertises the kind but leaves the flag unset. When
        // supported the conversation must be well-formed: non-empty, every turn carries non-blank
        // text, a `User` turn must carry its audio (it is provided context, never synthesized), and
        // there must be at least one assistant turn to synthesize (`audio: None`) — all malformed
        // requests → `Error::Msg`. Per-model turn-ordering / count bounds are layered by the
        // provider's own `validate`.
        for c in &req.conditioning {
            if let Conditioning::ConversationHistory { turns } = c {
                if !self.supports_conversation_history {
                    return Err(Error::Unsupported(format!(
                        "{id}: a multi-turn conversation history is not supported"
                    )));
                }
                if turns.is_empty() {
                    return Err(Error::Msg(format!(
                        "{id}: conversation history is empty — a conversation must carry at least \
                         one turn"
                    )));
                }
                let mut has_synthesis = false;
                for (i, turn) in turns.iter().enumerate() {
                    if turn.text.trim().is_empty() {
                        return Err(Error::Msg(format!(
                            "{id}: conversation turn {i} has empty text"
                        )));
                    }
                    match (turn.role, turn.audio.is_none()) {
                        (ConversationRole::User, true) => {
                            return Err(Error::Msg(format!(
                                "{id}: conversation turn {i} is a User turn with no audio — a user \
                                 turn is provided context and must carry its audio"
                            )));
                        }
                        (ConversationRole::Assistant, true) => has_synthesis = true,
                        _ => {}
                    }
                }
                if !has_synthesis {
                    return Err(Error::Msg(format!(
                        "{id}: conversation history has no assistant turn to synthesize (a turn with \
                         audio: None)"
                    )));
                }
            }
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::execution_domains::{CfgBatchingDomain, ExecutionValueDomain};

    #[test]
    fn staged_residency_availability_preserves_the_two_independent_capabilities() {
        let classify = |unconditional, selectable| {
            Capabilities {
                unconditionally_engages_staged_residency: unconditional,
                supports_sequential_offload: selectable,
                ..Default::default()
            }
            .staged_residency_availability()
        };

        assert_eq!(classify(false, false), StagedResidencyAvailability::Absent);
        assert_eq!(
            classify(false, true),
            StagedResidencyAvailability::Selectable
        );
        assert_eq!(
            classify(true, false),
            StagedResidencyAvailability::UnconditionallyEngaged
        );
        assert_eq!(
            classify(true, true),
            StagedResidencyAvailability::UnconditionallyEngaged,
            "unconditional physical staging wins the derived tri-state without erasing that the \
             separate supports_sequential_offload bit is also true"
        );
    }

    #[test]
    fn default_scope_is_allowed_only_for_resident_only_contracts() {
        let mut contract = MemoryProviderContract::compatibility_default(
            "fixture",
            crate::MemoryBackendRealization::MlxMetal {
                bounded_wired_residency: true,
                lazy_or_mmap_materialization: true,
                explicit_evaluation_and_synchronization: true,
                cache_eviction: true,
            },
        );
        assert!(!requires_memory_request_scope(&contract));
        contract
            .strategies
            .iter_mut()
            .find(|capability| capability.strategy == MemoryStrategy::StagedResidency)
            .unwrap()
            .support = crate::MemoryStrategySupport::Implemented;
        assert!(requires_memory_request_scope(&contract));
    }

    #[test]
    fn a_component_cannot_be_raised_without_a_visible_floor_declaration() {
        assert_eq!(
            effective_component_quant(&[], PrecisionFloorComponent::TextEncoder, Quant::Q4),
            Quant::Q4
        );
        let declared = [ComponentPrecisionFloor {
            component: PrecisionFloorComponent::TextEncoder,
            selected_tier: Quant::Q4,
            resident_tier: Quant::Q8,
        }];
        assert_eq!(
            effective_component_quant(&declared, PrecisionFloorComponent::TextEncoder, Quant::Q4),
            Quant::Q8
        );
        assert_eq!(
            effective_component_quant(
                &declared,
                PrecisionFloorComponent::TransformerHead,
                Quant::Q4
            ),
            Quant::Q4,
            "an unrelated component cannot inherit another component's floor"
        );
    }

    #[test]
    fn generation_memory_is_opt_in_and_quality_preserving_levers_default_off() {
        let request = GenerationRequest::default();
        assert_eq!(request.memory, None);
        assert_eq!(
            GenerationMemory::default(),
            GenerationMemory {
                stage_residency: false,
                tile_vae_decode: false,
                chunk_attention: false,
                stream_transformer_blocks: false,
                decode_tile_edge: None,
                decode_overlap: None,
                attention_chunk_size: None,
                transformer_window_size: None,
                transformer_window_component: None,
                // sc-18317's typed execution domains. `None` is the provider's own historical
                // schedule for each, which is what keeps an untouched request byte-for-byte the
                // pre-sc-18317 render.
                graph_eval_cadence: None,
                ffn_chunk: None,
                cfg_batching: None,
                calibration_error_phase: None,
                calibration_fault_harness_authorized: false,
            }
        );
    }

    /// SC-15510: the strategy-parameter carriers are additive and default-inert. This is deliberately
    /// an exhaustive literal — adding a lever without deciding its default here fails to compile,
    /// which is the point: a new field that defaults to anything but "the provider's own historical
    /// constant" would silently change every existing render.
    #[test]
    fn strategy_parameters_default_to_the_providers_own_constants() {
        let memory = GenerationMemory::default();
        assert!(!memory.stage_residency);
        assert_eq!(memory.decode_tile_edge, None);
        assert_eq!(memory.decode_overlap, None);
        assert_eq!(memory.attention_chunk_size, None);
        assert_eq!(memory.transformer_window_size, None);
        // Setting a parameter does NOT turn its rung on: the boolean is the switch, the parameter is
        // only the value. A selector that set an edge but not `tile_vae_decode` gets no tiling.
        let parameterized = GenerationMemory {
            decode_tile_edge: Some(384),
            decode_overlap: Some(64),
            attention_chunk_size: Some(128),
            transformer_window_size: Some(2),
            ..Default::default()
        };
        assert!(!parameterized.tile_vae_decode);
        assert!(!parameterized.stream_transformer_blocks);
    }

    fn img(w: u32, h: u32) -> Image {
        Image {
            width: w,
            height: h,
            pixels: vec![0u8; (w * h * 3) as usize],
        }
    }

    #[test]
    fn scail2_typed_carrier_is_one_reference_and_rejects_crossed_shapes() {
        let exact = GenerationRequest {
            video_mode: Some("animation".to_owned()),
            conditioning: vec![
                Conditioning::Reference {
                    image: img(2, 2),
                    strength: None,
                },
                Conditioning::Mask { image: img(2, 2) },
                Conditioning::ControlClip {
                    frames: vec![img(2, 2), img(2, 2)],
                    mask: vec![img(2, 2), img(2, 2)],
                    masking_strength: 1.0,
                    start_frame: 0,
                    mode: ReplacementMode::default(),
                },
            ],
            ..Default::default()
        };
        let carrier = exact.scail2_animation_conditioning().unwrap();
        assert_eq!(carrier.reference_count(), 1);
        assert_eq!(exact.memory_reference_count(), 1);
        assert_eq!(
            exact.image_reference_count(),
            2,
            "physical Mask remains visible generically"
        );

        let mut extra = exact.clone();
        extra.conditioning.push(Conditioning::Reference {
            image: img(2, 2),
            strength: None,
        });
        assert!(extra.scail2_animation_conditioning().is_err());

        let mut mismatched = exact.clone();
        let Conditioning::ControlClip { mask, .. } = &mut mismatched.conditioning[2] else {
            unreachable!()
        };
        mask.pop();
        assert!(mismatched.scail2_animation_conditioning().is_err());

        let mut replacement = exact;
        replacement.video_mode = Some("replacement".to_owned());
        assert_eq!(replacement.memory_reference_count(), 1);
        let carrier = replacement.scail2_animation_conditioning().unwrap();
        assert_eq!(
            carrier.identity_shape("replacement").unwrap(),
            "replacement:reference:2x2:control:2x2x2"
        );
    }

    #[test]
    fn keyframes_accessor_collects_in_order() {
        // first_last_frame: two keyframes at 0 and the last latent frame.
        let req = GenerationRequest {
            conditioning: vec![
                Conditioning::Keyframe {
                    image: img(2, 2),
                    frame_idx: 0,
                    strength: 1.0,
                },
                Conditioning::Reference {
                    image: img(2, 2),
                    strength: None,
                },
                Conditioning::Keyframe {
                    image: img(4, 4),
                    frame_idx: 8,
                    strength: 0.75,
                },
            ],
            ..Default::default()
        };
        let kf = req.keyframes();
        assert_eq!(kf.len(), 2);
        assert_eq!((kf[0].frame_idx, kf[0].strength), (0, 1.0));
        assert_eq!((kf[1].frame_idx, kf[1].strength), (8, 0.75));
        assert_eq!((kf[1].image.width, kf[1].image.height), (4, 4));
        // Reference is not a keyframe and is not a video clip / control clip.
        assert!(req.video_clips().is_empty());
        assert!(req.control_clip().is_none());
    }

    #[test]
    fn video_clips_accessor_collects_clips() {
        // video_bridge: left clip @0, right clip @tail.
        let req = GenerationRequest {
            conditioning: vec![
                Conditioning::VideoClip {
                    frames: vec![img(2, 2), img(2, 2)],
                    frame_idx: 0,
                    strength: 1.0,
                },
                Conditioning::VideoClip {
                    frames: vec![img(2, 2)],
                    frame_idx: 24,
                    strength: 0.9,
                },
            ],
            ..Default::default()
        };
        let clips = req.video_clips();
        assert_eq!(clips.len(), 2);
        assert_eq!((clips[0].frames.len(), clips[0].frame_idx), (2, 0));
        assert_eq!((clips[1].frames.len(), clips[1].frame_idx), (1, 24));
        assert!(req.keyframes().is_empty());
    }

    #[test]
    fn control_clip_accessor_returns_first() {
        let req = GenerationRequest {
            conditioning: vec![Conditioning::ControlClip {
                frames: vec![img(2, 2), img(2, 2)],
                mask: vec![img(2, 2), img(2, 2)],
                masking_strength: 0.8,
                start_frame: 0,
                mode: ReplacementMode::FaceOnly,
            }],
            ..Default::default()
        };
        let cc = req.control_clip().expect("control clip present");
        assert_eq!((cc.frames.len(), cc.mask.len()), (2, 2));
        assert_eq!(cc.masking_strength, 0.8);
        assert_eq!(cc.mode, ReplacementMode::FaceOnly);
    }

    #[test]
    fn accessors_empty_by_default() {
        let req = GenerationRequest::default();
        assert!(req.keyframes().is_empty());
        assert!(req.video_clips().is_empty());
        assert!(req.control_clip().is_none());
    }

    /// A capability surface that turns nothing extra on: a single 256..=1024 image, no
    /// negative/guidance/true_cfg, no samplers/schedulers, only `Reference` conditioning.
    fn caps() -> Capabilities {
        Capabilities {
            conditioning: vec![ConditioningKind::Reference],
            samplers: vec!["euler"],
            min_size: 256,
            max_size: 1024,
            max_count: 1,
            ..Default::default()
        }
    }

    /// **sc-18317: the typed execution domains are fail-closed at the shared floor.**
    ///
    /// The defect this closes is the pre-story state of all three knobs: a per-provider ad hoc
    /// parameter a caller could not set, could not discover, and — had it been threaded through the
    /// request without a declaration — would have been silently dropped by every provider that does
    /// not implement it. So the floor's contract is exactly two things: an unset field is inert, and
    /// a set field on a non-declaring descriptor is a typed `Unsupported` naming the field and the
    /// remedy.
    #[test]
    fn execution_domains_are_refused_by_name_on_a_non_declaring_descriptor() {
        let unsupported = caps();
        assert!(
            unsupported.execution.is_inert(),
            "the default capability surface must declare no execution domain"
        );

        for (label, memory) in [
            (
                "graph_eval_cadence",
                GenerationMemory {
                    graph_eval_cadence: Some(GraphEvalCadence::EVERY_BLOCK),
                    ..Default::default()
                },
            ),
            (
                "ffn_chunk",
                GenerationMemory {
                    ffn_chunk: Some(FfnChunk::new(4096).unwrap()),
                    ..Default::default()
                },
            ),
            (
                "cfg_batching",
                GenerationMemory {
                    cfg_batching: Some(CfgBatching::Sequential),
                    ..Default::default()
                },
            ),
        ] {
            let request = GenerationRequest {
                memory: Some(memory),
                ..base_req()
            };
            let error = unsupported
                .validate_request("m", &request)
                .expect_err("a non-declaring descriptor must refuse a selected execution domain");
            assert!(
                matches!(error, Error::Unsupported(_)),
                "{label} must be a capability gap, not a range error: {error:?}"
            );
            let message = error.to_string();
            assert!(message.contains(label), "{label}: {message}");
            assert!(
                message.contains("unset"),
                "{label} refusal must name the remedy: {message}"
            );
            // The size-skipping floors run every non-spatial check, so the gate must be reached
            // through them too — a provider on the auto-size convention must not lose the refusal.
            assert!(
                unsupported
                    .validate_request_skip_size("m", &request)
                    .is_err(),
                "{label} must also be refused on the size-skipping floor"
            );
            assert!(
                unsupported.validate_request_audio("m", &request).is_err(),
                "{label} must also be refused on the audio floor"
            );
        }
    }

    /// The other half: a declaring descriptor admits exactly its declared values, and the ladder
    /// parameters are untouched by the execution gate.
    #[test]
    fn execution_domains_admit_declared_values_and_leave_the_ladder_alone() {
        let declaring = Capabilities {
            execution: ExecutionSurface {
                graph_eval_cadence_blocks: ExecutionValueDomain::ANY_POSITIVE,
                ffn_chunk_rows: ExecutionValueDomain::Candidates(vec![2048]),
                cfg_batching: CfgBatchingDomain::Modes(vec![CfgBatching::Batched]),
            },
            ..caps()
        };
        assert!(declaring.execution.declaration_errors().is_empty());

        let admitted = GenerationRequest {
            memory: Some(GenerationMemory {
                graph_eval_cadence: Some(GraphEvalCadence::new(4).unwrap()),
                ffn_chunk: Some(FfnChunk::new(2048).unwrap()),
                cfg_batching: Some(CfgBatching::Batched),
                ..Default::default()
            }),
            ..base_req()
        };
        declaring
            .validate_request("m", &admitted)
            .expect("every declared value must reach the provider");

        let off_grid = GenerationRequest {
            memory: Some(GenerationMemory {
                ffn_chunk: Some(FfnChunk::new(2047).unwrap()),
                ..Default::default()
            }),
            ..base_req()
        };
        let message = declaring
            .validate_request("m", &off_grid)
            .expect_err("an off-domain chunk must be refused")
            .to_string();
        assert!(
            message.contains("[2048]"),
            "domain must be named: {message}"
        );

        let unimplemented_mode = GenerationRequest {
            memory: Some(GenerationMemory {
                cfg_batching: Some(CfgBatching::Sequential),
                ..Default::default()
            }),
            ..base_req()
        };
        assert!(
            declaring
                .validate_request("m", &unimplemented_mode)
                .is_err(),
            "a mode the provider does not implement must be refused even though CFG batching is \
             declared"
        );

        // An unset execution selection carrying a ladder rung still validates on the inert surface:
        // the execution gate is independent of the memory ladder.
        let ladder_only = GenerationRequest {
            memory: Some(GenerationMemory {
                stage_residency: true,
                chunk_attention: true,
                attention_chunk_size: Some(128),
                ..Default::default()
            }),
            ..base_req()
        };
        caps()
            .validate_request("m", &ladder_only)
            .expect("the execution gate must not touch ladder parameters");
    }

    #[test]
    fn preview_support_is_weights_free_and_defaults_to_unsupported() {
        let unsupported = Capabilities::default();
        let supported_waiting_for_first_frame = Capabilities {
            supports_preview: true,
            ..Default::default()
        };

        assert!(!unsupported.supports_preview);
        assert!(supported_waiting_for_first_frame.supports_preview);
    }

    #[test]
    fn prompt_enhancement_is_fail_closed_and_tuning_is_bounded() {
        let unsupported = caps();
        let requested = GenerationRequest {
            enhance_prompt: true,
            ..base_req()
        };
        assert!(matches!(
            unsupported.validate_request("m", &requested),
            Err(Error::Unsupported(message)) if message.contains("prompt enhancement")
        ));

        let supported = Capabilities {
            supports_prompt_enhancement: true,
            ..caps()
        };
        for request in [
            GenerationRequest {
                enhance_max_tokens: Some(32),
                ..base_req()
            },
            GenerationRequest {
                enhance_temperature: Some(0.15),
                ..base_req()
            },
        ] {
            assert!(supported
                .validate_request("m", &request)
                .unwrap_err()
                .to_string()
                .contains("requires enhance_prompt=true"));
        }
        for tokens in [0, MAX_ENHANCE_TOKENS + 1] {
            let request = GenerationRequest {
                enhance_prompt: true,
                enhance_max_tokens: Some(tokens),
                ..base_req()
            };
            assert!(supported.validate_request("m", &request).is_err());
        }
        for temperature in [-0.1, MAX_ENHANCE_TEMPERATURE + 0.1] {
            let request = GenerationRequest {
                enhance_prompt: true,
                enhance_temperature: Some(temperature),
                ..base_req()
            };
            assert!(supported.validate_request("m", &request).is_err());
        }
        assert!(supported
            .validate_request(
                "m",
                &GenerationRequest {
                    enhance_prompt: true,
                    enhance_max_tokens: Some(MAX_ENHANCE_TOKENS),
                    enhance_temperature: Some(MAX_ENHANCE_TEMPERATURE),
                    ..base_req()
                }
            )
            .is_ok());
    }

    fn base_req() -> GenerationRequest {
        GenerationRequest {
            prompt: "x".into(),
            width: 512,
            height: 512,
            ..Default::default()
        }
    }

    #[test]
    fn validate_request_accepts_in_surface() {
        let c = caps();
        assert!(c.validate_request("m", &base_req()).is_ok());
        // An advertised sampler + an accepted conditioning kind are fine.
        assert!(c
            .validate_request(
                "m",
                &GenerationRequest {
                    sampler: Some("euler".into()),
                    conditioning: vec![Conditioning::Reference {
                        image: img(8, 8),
                        strength: None,
                    }],
                    ..base_req()
                }
            )
            .is_ok());
    }

    #[test]
    fn calibration_fault_injection_requires_a_complete_harness_authorization_pair() {
        let capabilities = caps();

        let mut phase_without_authorization = base_req();
        phase_without_authorization.memory = Some(GenerationMemory {
            calibration_error_phase: Some(MemoryPhase::Denoise),
            ..Default::default()
        });
        let error = capabilities
            .validate_request("m", &phase_without_authorization)
            .unwrap_err();
        assert!(
            matches!(error, Error::Unsupported(message) if message.contains("explicit harness authorization"))
        );

        let mut authorization_without_phase = base_req();
        authorization_without_phase.memory = Some(GenerationMemory {
            calibration_fault_harness_authorized: true,
            ..Default::default()
        });
        let error = capabilities
            .validate_request_skip_size("m", &authorization_without_phase)
            .unwrap_err();
        assert!(
            matches!(error, Error::Unsupported(message) if message.contains("requires an error phase"))
        );

        let mut authorized = base_req();
        let mut memory = GenerationMemory::default();
        memory.authorize_calibration_fault(MemoryPhase::Decode);
        authorized.memory = Some(memory);
        capabilities
            .validate_request("m", &authorized)
            .expect("an explicitly authorized calibration harness request reaches the provider");
    }

    /// `ResolvedDownstream` exempts **the sentinel**, not every size.
    ///
    /// The variant's own doc says "`width`/`height == 0` is a resolve-from-driving-media sentinel and
    /// the range check does not apply *to it*". Scoping matters: a provider advertising this is
    /// saying one specific pair of values is special, not that it has opted out of the shared floor.
    ///
    /// Written as a regression gate because the first implementation of `size_floor` read the
    /// variant as a blanket opt-out and disabled the range check entirely. On main, SCAIL-2 got the
    /// full check whenever dimensions were explicit — its `pipeline.rs` hand-rolls exactly this
    /// branch, with a comment saying so — and routing that branch through a blanket
    /// `ResolvedDownstream` silently stopped rejecting an explicit 16x16 or 4096x4096 against
    /// declared bounds of 32..=1280. Nothing downstream catches it: `min_size`/`max_size` appear
    /// nowhere in `mlx-gen-scail2` outside the descriptor.
    #[test]
    fn resolved_downstream_exempts_the_sentinel_but_still_range_checks_explicit_sizes() {
        let c = Capabilities {
            size_floor: SizeFloor::ResolvedDownstream,
            ..caps()
        };

        // The sentinel is the whole point of the variant: accepted.
        assert!(
            c.validate_request(
                "m",
                &GenerationRequest {
                    width: 0,
                    height: 0,
                    ..base_req()
                }
            )
            .is_ok(),
            "the 0x0 resolve-downstream sentinel must be accepted"
        );

        // Explicit dimensions are NOT the sentinel and must still meet the advertised range.
        for (w, h, why) in [
            (16, 16, "below min_size"),
            (4096, 4096, "above max_size"),
            (
                0,
                512,
                "half-sentinel: only width is 0, so this is not the sentinel",
            ),
            (512, 0, "half-sentinel: only height is 0"),
        ] {
            assert!(
                c.validate_request(
                    "m",
                    &GenerationRequest {
                        width: w,
                        height: h,
                        ..base_req()
                    }
                )
                .is_err(),
                "{w}x{h} ({why}) must still be rejected under ResolvedDownstream"
            );
        }
    }

    #[test]
    fn advertised_explicit_grid_rejects_only_off_grid_explicit_sizes() {
        let ranged = Capabilities {
            size_floor: SizeFloor::RangeCheckedOnGrid { multiple: 32 },
            ..caps()
        };
        let resolved = Capabilities {
            size_floor: SizeFloor::ResolvedDownstreamExplicitGrid { multiple: 32 },
            ..caps()
        };

        for c in [&ranged, &resolved] {
            assert!(c
                .validate_request(
                    "m",
                    &GenerationRequest {
                        width: 512,
                        height: 480,
                        ..base_req()
                    }
                )
                .is_ok());
            for (width, height) in [(512, 481), (513, 480)] {
                let err = c
                    .validate_request(
                        "m",
                        &GenerationRequest {
                            width,
                            height,
                            ..base_req()
                        },
                    )
                    .expect_err("an explicit off-grid size must be refused")
                    .to_string();
                assert!(
                    err.contains("multiples of 32") && err.contains(&format!("{width}×{height}")),
                    "got: {err}"
                );
            }
        }

        assert!(ranged
            .validate_request(
                "m",
                &GenerationRequest {
                    width: 0,
                    height: 0,
                    ..base_req()
                }
            )
            .is_err());
        assert!(resolved
            .validate_request(
                "m",
                &GenerationRequest {
                    width: 0,
                    height: 0,
                    ..base_req()
                }
            )
            .is_ok());
        assert_eq!(ranged.size_floor.explicit_size_multiple(), Some(32));
        assert_eq!(resolved.size_floor.explicit_size_multiple(), Some(32));
        assert_eq!(SizeFloor::RangeChecked.explicit_size_multiple(), None);
    }

    /// The default variant is unaffected — a descriptor that says nothing behaves exactly as before.
    #[test]
    fn range_checked_is_the_default_and_rejects_out_of_range_sizes() {
        let c = caps();
        assert_eq!(c.size_floor, SizeFloor::RangeChecked);
        for (w, h) in [(16, 16), (4096, 4096), (0, 0)] {
            assert!(
                c.validate_request(
                    "m",
                    &GenerationRequest {
                        width: w,
                        height: h,
                        ..base_req()
                    }
                )
                .is_err(),
                "{w}x{h} must be rejected when the floor is RangeChecked"
            );
        }
    }

    /// `ResolvedDownstream` exempts **size**, and nothing else.
    ///
    /// The sibling above pins one half of the scoping — an *explicit* size is still range-checked.
    /// This pins the half a mis-scoping would break in the other direction: routing the sentinel
    /// around the **whole** floor rather than around one check. That is not a hypothetical failure
    /// mode for the provider this variant exists for. SCAIL-2's `generate` re-runs the floor itself
    /// precisely so a `guidance: NaN` cannot NaN-poison a multi-minute video render into
    /// garbage-as-success, and `0x0` — the auto-size path — is its *normal* request shape, not an
    /// edge case. A sentinel that skipped the floor would therefore be un-validated in the common
    /// case and validated only in the rare one.
    ///
    /// Every case is a single-field mutation of a request the test first proves is accepted, so a
    /// rejection can only have come from the field named: no case can pass vacuously.
    #[test]
    fn resolved_downstream_sentinel_still_runs_every_non_size_check() {
        let c = Capabilities {
            size_floor: SizeFloor::ResolvedDownstream,
            ..caps()
        };
        let sentinel = GenerationRequest {
            width: 0,
            height: 0,
            ..base_req()
        };
        assert!(
            c.validate_request("m", &sentinel).is_ok(),
            "the 0x0 sentinel must be accepted, or every case below would pass vacuously"
        );

        let cases: Vec<(&str, &str, GenerationRequest)> = vec![
            (
                "count above max_count",
                "count",
                GenerationRequest {
                    count: 2,
                    ..sentinel.clone()
                },
            ),
            (
                "an explicit steps: 0 (F-007)",
                "steps must be >= 1",
                GenerationRequest {
                    steps: Some(0),
                    ..sentinel.clone()
                },
            ),
            (
                "an unadvertised sampler",
                "unsupported sampler",
                GenerationRequest {
                    sampler: Some("unipc".into()),
                    ..sentinel.clone()
                },
            ),
            (
                "an unadvertised scheduler",
                "unsupported scheduler",
                GenerationRequest {
                    scheduler: Some("linear".into()),
                    ..sentinel.clone()
                },
            ),
            (
                "a conditioning kind outside the allowlist",
                "conditioning is not supported",
                GenerationRequest {
                    conditioning: vec![Conditioning::Depth { image: img(8, 8) }],
                    ..sentinel.clone()
                },
            ),
            (
                "a non-finite float knob (F-053)",
                "must be finite",
                GenerationRequest {
                    strength: Some(f32::NAN),
                    ..sentinel.clone()
                },
            ),
            (
                "a negative prompt on a model that does not advertise one",
                "negative prompts are not supported",
                GenerationRequest {
                    negative_prompt: Some("n".into()),
                    ..sentinel.clone()
                },
            ),
            (
                "guidance on a distilled/CFG-free model",
                "guidance is not supported",
                GenerationRequest {
                    guidance: Some(3.5),
                    ..sentinel.clone()
                },
            ),
        ];
        for (why, expected, req) in cases {
            let Err(err) = c.validate_request("m", &req) else {
                panic!(
                    "{why} must still be rejected on the 0x0 sentinel path — \
                     ResolvedDownstream exempts SIZE only, not the rest of the floor"
                );
            };
            // Asserting the *message* and not merely `is_err` is what makes a case non-vacuous: a
            // future reordering that made some earlier check fire first would otherwise look green
            // while the check this row is actually about had stopped running.
            assert!(
                err.to_string().contains(expected),
                "{why}: expected a rejection naming {expected:?}, got: {err}"
            );
        }
    }

    #[test]
    fn validate_request_enforces_advertised_surface() {
        let c = caps();
        let cases: Vec<GenerationRequest> = vec![
            // count out of range
            GenerationRequest {
                count: 0,
                ..base_req()
            },
            GenerationRequest {
                count: 2,
                ..base_req()
            },
            // size out of range (below min, above max)
            GenerationRequest {
                width: 128,
                ..base_req()
            },
            GenerationRequest {
                height: 2048,
                ..base_req()
            },
            // capability flags not advertised
            GenerationRequest {
                negative_prompt: Some("n".into()),
                ..base_req()
            },
            GenerationRequest {
                guidance: Some(3.5),
                ..base_req()
            },
            GenerationRequest {
                true_cfg: Some(4.0),
                ..base_req()
            },
            // sampler / scheduler not advertised
            GenerationRequest {
                sampler: Some("unipc".into()),
                ..base_req()
            },
            GenerationRequest {
                scheduler: Some("linear".into()),
                ..base_req()
            },
            // conditioning kind not accepted
            GenerationRequest {
                conditioning: vec![Conditioning::Depth { image: img(8, 8) }],
                ..base_req()
            },
        ];
        for (i, req) in cases.iter().enumerate() {
            assert!(
                c.validate_request("m", req).is_err(),
                "case {i} should have been rejected"
            );
        }
    }

    #[test]
    fn validate_request_skip_size_runs_every_non_size_check() {
        // F-158: the auto-size floor. `validate_request_skip_size` must still enforce the whole floor
        // *except* the size range — so a sentinel `0x0` size passes, but an out-of-surface count /
        // sampler / conditioning / non-finite knob is still rejected. Used by providers that resolve
        // size downstream (SCAIL-2 sizing from the driving video).
        let c = caps();
        // Auto-size sentinel (0x0) is accepted where the full floor would reject it for being < min.
        let auto = GenerationRequest {
            width: 0,
            height: 0,
            ..base_req()
        };
        assert!(
            c.validate_request("m", &auto).is_err(),
            "size 0x0 is below min for the full floor"
        );
        assert!(
            c.validate_request_skip_size("m", &auto).is_ok(),
            "skip_size must accept the 0x0 auto-size sentinel"
        );
        // Every non-size violation must still fire on the auto-size path. Labelled rather than
        // indexed: "skip_size case 3 failed" does not tell you which guarantee stopped holding.
        //
        // The last three rows close a real gap. This method's doc promises "negative-prompt /
        // guidance / true_cfg support gating" and "scheduler … membership" on this path, and
        // neither had a single case here — the *capability-gap* family (typed `Error::Unsupported`,
        // F-008) was represented only by `sampler`. Support gating is the family a consumer relies
        // on to tell "this backend can't do that" from "that value is out of range", so a
        // size-skipping floor that quietly dropped it would let a negative prompt reach a
        // CFG-free distilled model and be silently ignored.
        let rejected: Vec<(&str, GenerationRequest)> = vec![
            (
                "oversized count",
                GenerationRequest {
                    count: 2,
                    ..auto.clone()
                },
            ),
            (
                "explicit zero steps",
                GenerationRequest {
                    steps: Some(0),
                    ..auto.clone()
                },
            ),
            (
                "unadvertised sampler",
                GenerationRequest {
                    sampler: Some("unipc".into()),
                    ..auto.clone()
                },
            ),
            (
                "disallowed conditioning kind",
                GenerationRequest {
                    conditioning: vec![Conditioning::Depth { image: img(8, 8) }],
                    ..auto.clone()
                },
            ),
            (
                // A non-finite knob is not support-gated and would NaN-poison the run.
                "non-finite strength (F-053)",
                GenerationRequest {
                    strength: Some(f32::NAN),
                    ..auto.clone()
                },
            ),
            (
                "negative prompt on a model that does not advertise one",
                GenerationRequest {
                    negative_prompt: Some("n".into()),
                    ..auto.clone()
                },
            ),
            (
                "guidance on a model that does not advertise it",
                GenerationRequest {
                    guidance: Some(3.5),
                    ..auto.clone()
                },
            ),
            (
                "true_cfg on a model that does not advertise it",
                GenerationRequest {
                    true_cfg: Some(4.0),
                    ..auto.clone()
                },
            ),
            (
                "unadvertised scheduler",
                GenerationRequest {
                    scheduler: Some("linear".into()),
                    ..auto.clone()
                },
            ),
        ];
        for (why, req) in &rejected {
            assert!(
                c.validate_request_skip_size("m", req).is_err(),
                "{why} must still be rejected on the auto-size path"
            );
        }
    }

    #[test]
    fn validate_request_rejects_explicit_zero_steps() {
        // F-007: the floor now enforces the steps>=1 claim the schedule builders rely on.
        let c = caps();
        let bad = GenerationRequest {
            steps: Some(0),
            ..base_req()
        };
        let err = c.validate_request("m", &bad).unwrap_err();
        assert!(matches!(err, Error::Msg(_)), "steps=0 is a range error");
        assert!(err.to_string().contains("steps must be >= 1"));
        // `None` and a positive count still pass.
        assert!(c.validate_request("m", &base_req()).is_ok());
        assert!(c
            .validate_request(
                "m",
                &GenerationRequest {
                    steps: Some(1),
                    ..base_req()
                }
            )
            .is_ok());
    }

    /// sc-19502 — the exact-step surface refuses an off-schedule count, admits every count it
    /// advertises, and admits `None`.
    ///
    /// The `None` half is the load-bearing one: it is what keeps "omit `steps`, get the baked
    /// schedule" working, and inverting it would break every caller of a distilled model who does
    /// not know its magic number.
    #[test]
    fn advertised_supported_steps_reject_every_off_schedule_count() {
        let distilled = Capabilities {
            supported_steps: StepSupport::Exact(vec![8]),
            ..caps()
        };
        let at = |steps: Option<u32>| {
            distilled.validate_request(
                "ltx_2_3",
                &GenerationRequest {
                    steps,
                    ..base_req()
                },
            )
        };

        // Admitted: the advertised count, and "the model picks".
        assert!(at(Some(8)).is_ok(), "the advertised count must be admitted");
        assert!(
            at(None).is_ok(),
            "an omitted count must use the baked schedule, not be refused"
        );

        // Refused on BOTH sides of the advertised value — a floor-shaped guard would let 30 through,
        // which is the precise half `limits.hardMinSteps` could not express (sc-19426).
        for steps in [1, 4, 7, 9, 30] {
            let err = at(Some(steps)).unwrap_err().to_string();
            assert!(
                err.contains("ltx_2_3")
                    && err.contains(&format!("steps={steps}"))
                    && err.contains("8"),
                "the refusal must name the model, the request and the legal value: {err}"
            );
        }
    }

    /// sc-19502 — the polarity guard. Empty `supported_steps` is NO constraint, so every descriptor
    /// that never opts in is byte-identical to before the field existed.
    ///
    /// Deliberately its own test rather than an assertion inside the one above: this is the
    /// property that makes the field safe to add to a 128-descriptor repo, and if it regressed, a
    /// bare `Default::default()` would refuse every step count in the workspace.
    #[test]
    fn an_unconstrained_model_admits_every_step_count_the_sanity_caps_admit() {
        let c = caps();
        assert!(
            c.supported_steps.is_unconstrained(),
            "the default must be unconstrained"
        );
        for steps in [1, 2, 8, 30, MAX_STEPS] {
            assert!(
                c.validate_request(
                    "m",
                    &GenerationRequest {
                        steps: Some(steps),
                        ..base_req()
                    }
                )
                .is_ok(),
                "an undeclared model must still admit {steps} steps"
            );
        }
    }

    /// sc-19502 — a multi-value schedule is a SET, not a range: the gap between two advertised
    /// counts is refused.
    ///
    /// Pins the shape choice. A future distilled model with two baked schedules can declare both
    /// without the guard degrading into "anything between the smallest and largest".
    #[test]
    fn supported_steps_is_a_set_not_a_range() {
        let two_schedules = Capabilities {
            supported_steps: StepSupport::Exact(vec![4, 8]),
            ..caps()
        };
        let at = |steps: u32| {
            two_schedules.validate_request(
                "m",
                &GenerationRequest {
                    steps: Some(steps),
                    ..base_req()
                },
            )
        };
        assert!(at(4).is_ok());
        assert!(at(8).is_ok());
        let gap = at(6)
            .expect_err("6 sits between the two schedules and belongs to neither")
            .to_string();
        // The plural arm names every legal count, so the caller can pick one rather than guess.
        assert!(
            gap.contains("4 or 8") && gap.contains("steps=6"),
            "the multi-schedule refusal must list the legal counts: {gap}"
        );
    }

    /// sc-19559 — the ceiling. SVD-XT's `MAX_STEPS = 200` used to live only inside the two SVD
    /// providers' `validate`; the shared floor now enforces the advertised range, and refuses on
    /// BOTH ends of it.
    ///
    /// The `200` boundary pair is the load-bearing part: an off-by-one that made the range
    /// exclusive would refuse the model's own top count, and one that never fired would admit
    /// `201` — asserting only the far-outside `10_000` would catch neither.
    #[test]
    fn an_advertised_step_range_refuses_both_ends_and_admits_the_interior() {
        let bounded = Capabilities {
            supported_steps: StepSupport::Range { min: 2, max: 200 },
            ..caps()
        };
        let at = |steps: Option<u32>| {
            bounded.validate_request(
                "svd_xt",
                &GenerationRequest {
                    steps,
                    ..base_req()
                },
            )
        };

        for ok in [2, 3, 25, 199, 200] {
            assert!(at(Some(ok)).is_ok(), "{ok} is inside 2..=200 and must pass");
        }
        assert!(
            at(None).is_ok(),
            "an omitted count must use the model's default, not be refused"
        );

        for bad in [1, 201, 10_000] {
            let err = at(Some(bad)).unwrap_err().to_string();
            assert!(
                err.contains("svd_xt") && err.contains(&bad.to_string()) && err.contains("2..=200"),
                "the refusal must name the model, the request and the legal range: {err}"
            );
        }
    }

    /// sc-19559 — the ceiling is readable from the descriptor, which is the whole point: a
    /// consumer must not have to dispatch a job to learn the bound.
    ///
    /// Also pins that a bound is NOT a default: `floor()` is the smallest legal count, and the
    /// step count a request omitting `steps` actually gets is the model's own baked default,
    /// which the capability surface deliberately does not carry.
    #[test]
    fn a_declared_bound_is_readable_and_is_not_a_default() {
        assert_eq!(StepSupport::Unconstrained.ceiling(), None);
        assert_eq!(StepSupport::Unconstrained.floor(), None);

        let range = StepSupport::Range { min: 2, max: 200 };
        assert_eq!(range.ceiling(), Some(200));
        assert_eq!(range.floor(), Some(2));

        // An exact menu's ceiling is its largest member, not its declaration order.
        let menu = StepSupport::Exact(vec![8, 4]);
        assert_eq!(menu.ceiling(), Some(8));
        assert_eq!(menu.floor(), Some(4));

        // A model that declares a bound still renders its own default when `steps` is omitted —
        // the floor never substitutes the bound for the missing value.
        let bounded = Capabilities {
            supported_steps: range,
            ..caps()
        };
        let req = GenerationRequest {
            steps: None,
            ..base_req()
        };
        assert!(bounded.validate_request("m", &req).is_ok());
        assert_eq!(
            req.steps, None,
            "validation must not seed `steps` from a bound"
        );
    }

    #[test]
    fn non_finite_extended_float_knobs_are_rejected() {
        // F-001: the finiteness floor now covers every `Option<f32>` knob added after F-053, not just
        // guidance/true_cfg. A guidance-capable caps so the support gate never fires; each knob is
        // exercised with NaN/±Inf and must produce a typed `Msg` naming the field.
        let c = Capabilities {
            supports_guidance: true,
            supports_true_cfg: true,
            conditioning: vec![ConditioningKind::Reference, ConditioningKind::Control],
            ..caps()
        };
        type Build = fn(f32) -> GenerationRequest;
        let mk: [(&str, Build); 12] = [
            ("guidance_eta", |v| GenerationRequest {
                guidance_eta: Some(v),
                ..base_req()
            }),
            ("guidance_momentum", |v| GenerationRequest {
                guidance_momentum: Some(v),
                ..base_req()
            }),
            ("guidance_norm_threshold", |v| GenerationRequest {
                guidance_norm_threshold: Some(v),
                ..base_req()
            }),
            ("strength", |v| GenerationRequest {
                strength: Some(v),
                ..base_req()
            }),
            ("control_scale", |v| GenerationRequest {
                control_scale: Some(v),
                ..base_req()
            }),
            ("image_guidance", |v| GenerationRequest {
                image_guidance: Some(v),
                ..base_req()
            }),
            ("scheduler_shift", |v| GenerationRequest {
                scheduler_shift: Some(v),
                ..base_req()
            }),
            ("motion_bucket_id", |v| GenerationRequest {
                motion_bucket_id: Some(v),
                ..base_req()
            }),
            ("noise_aug_strength", |v| GenerationRequest {
                noise_aug_strength: Some(v),
                ..base_req()
            }),
            ("softness", |v| GenerationRequest {
                softness: Some(v),
                ..base_req()
            }),
            ("enhance_temperature", |v| GenerationRequest {
                enhance_temperature: Some(v),
                ..base_req()
            }),
            ("pid_capture_sigma", |v| GenerationRequest {
                pid_capture_sigma: Some(v),
                ..base_req()
            }),
        ];
        for (field, build) in mk {
            for bad in [f32::NAN, f32::INFINITY, f32::NEG_INFINITY] {
                let req = build(bad);
                let err = c.validate_request("m", &req).unwrap_err();
                assert!(matches!(err, Error::Msg(_)), "{field} {bad} → Msg");
                assert!(
                    err.to_string().contains(field) && err.to_string().contains("must be finite"),
                    "{field} {bad}: got {err}"
                );
            }
        }
        // The Control-branch scale carried inside a conditioning entry is guarded too (F-001).
        let ctrl = GenerationRequest {
            conditioning: vec![Conditioning::Control {
                image: img(8, 8),
                kind: ControlKind::Pose,
                scale: Some(f32::NAN),
            }],
            ..base_req()
        };
        let err = c.validate_request("m", &ctrl).unwrap_err();
        assert!(
            err.to_string().contains("conditioning.control.scale"),
            "got {err}"
        );
        // A fully-finite request across the extended knobs still passes.
        assert!(c
            .validate_request(
                "m",
                &GenerationRequest {
                    guidance_eta: Some(1.0),
                    guidance_momentum: Some(0.0),
                    strength: Some(0.6),
                    control_scale: Some(1.0),
                    ..base_req()
                }
            )
            .is_ok());
    }

    #[test]
    fn oversized_counters_hit_the_sanity_cap() {
        // F-004: the floor now rejects a pathological `steps` / video-counter value that would launch
        // an effectively-unbounded run. Model-realistic values still pass.
        let c = Capabilities {
            max_size: 4096,
            ..caps()
        };
        let steps = GenerationRequest {
            steps: Some(u32::MAX),
            ..base_req()
        };
        assert!(
            c.validate_request("m", &steps)
                .unwrap_err()
                .to_string()
                .contains("steps"),
            "u32::MAX steps must be capped"
        );
        let frames = GenerationRequest {
            frames: Some(u32::MAX),
            ..base_req()
        };
        assert!(
            c.validate_request("m", &frames).is_err(),
            "u32::MAX frames capped"
        );
        let fps = GenerationRequest {
            fps: Some(u32::MAX),
            ..base_req()
        };
        assert!(
            c.validate_request("m", &fps).is_err(),
            "u32::MAX fps capped"
        );
        // Realistic values pass.
        assert!(c
            .validate_request(
                "m",
                &GenerationRequest {
                    steps: Some(50),
                    frames: Some(121),
                    fps: Some(24),
                    ..base_req()
                }
            )
            .is_ok());
    }

    #[test]
    fn capability_gaps_return_typed_unsupported() {
        // F-008: capability-gap branches must be `Error::Unsupported`, not `Msg`, so candle gating /
        // the worker can distinguish them. Malformed-value branches (range/finiteness) stay `Msg`.
        let c = caps();
        let gap_cases: Vec<GenerationRequest> = vec![
            GenerationRequest {
                negative_prompt: Some("n".into()),
                ..base_req()
            },
            GenerationRequest {
                guidance: Some(3.5),
                ..base_req()
            },
            GenerationRequest {
                true_cfg: Some(4.0),
                ..base_req()
            },
            GenerationRequest {
                sampler: Some("unipc".into()),
                ..base_req()
            },
            GenerationRequest {
                scheduler: Some("linear".into()),
                ..base_req()
            },
            GenerationRequest {
                guidance_method: Some("apg".into()),
                ..base_req()
            },
            GenerationRequest {
                conditioning: vec![Conditioning::Depth { image: img(8, 8) }],
                ..base_req()
            },
        ];
        for (i, req) in gap_cases.iter().enumerate() {
            let err = c.validate_request("m", req).unwrap_err();
            assert!(
                matches!(err, Error::Unsupported(_)),
                "gap case {i} should be typed Unsupported, got {err:?}"
            );
        }
    }

    /// A pure-audio capability surface (sc-12834): no visual size bounds (unused for
    /// `Modality::Audio`), an advertised voice/language/sample-rate surface, a 60 s cap, and
    /// `ReferenceAudio` conditioning.
    fn audio_caps() -> Capabilities {
        Capabilities {
            conditioning: vec![ConditioningKind::ReferenceAudio],
            audio_sample_rates: vec![24_000, 48_000],
            max_audio_duration_secs: Some(60.0),
            audio_voices: vec!["nova"],
            audio_languages: vec!["en"],
            max_count: 1,
            ..Default::default()
        }
    }

    fn track() -> AudioTrack {
        AudioTrack {
            samples: vec![0.0; 16],
            sample_rate: 24_000,
            channels: 1,
            ..Default::default()
        }
    }

    /// A TTS-shaped request: prompt + typed audio sub-block, size left at the unused 0x0.
    fn audio_req() -> GenerationRequest {
        GenerationRequest {
            prompt: "read this aloud".into(),
            width: 0,
            height: 0,
            audio: Some(AudioParams {
                voice: Some("nova".into()),
                language: Some("en".into()),
                target_duration: Some(12.5),
                sample_rate: Some(24_000),
                ..Default::default()
            }),
            ..Default::default()
        }
    }

    #[test]
    fn audio_request_validates_against_audio_descriptor() {
        // sc-12834 acceptance: a `GenerationRequest { prompt, audio: Some(..) }` validates against
        // an audio Capabilities descriptor through the size-skipping audio floor.
        let c = audio_caps();
        assert!(c.validate_request_audio("tts", &audio_req()).is_ok());
        // A reference-audio conditioned request (voice cloning) passes when advertised.
        let cloned = GenerationRequest {
            conditioning: vec![Conditioning::ReferenceAudio {
                audio: track(),
                strength: Some(0.8),
            }],
            ..audio_req()
        };
        assert!(c.validate_request_audio("tts", &cloned).is_ok());
        // The music-shaped knobs are free-form/positive-gated, not membership-gated.
        let music = GenerationRequest {
            audio: Some(AudioParams {
                target_duration: Some(30.0),
                bpm: Some(120.0),
                musical_key: Some("C minor".into()),
                lyrics: Some("la la la".into()),
                ..Default::default()
            }),
            ..audio_req()
        };
        assert!(c.validate_request_audio("music", &music).is_ok());
    }

    #[test]
    fn audio_floor_rejects_visual_only_mismatches_with_typed_errors() {
        // sc-12834 acceptance: the audio floor still runs every non-size check — capability gaps
        // are typed `Unsupported`, malformed values are `Msg`.
        let c = audio_caps();
        // Capability gaps (visual-only surface the audio descriptor does not advertise).
        let gap_cases: Vec<GenerationRequest> = vec![
            GenerationRequest {
                negative_prompt: Some("n".into()),
                ..audio_req()
            },
            GenerationRequest {
                guidance: Some(3.5),
                ..audio_req()
            },
            GenerationRequest {
                sampler: Some("euler".into()),
                ..audio_req()
            },
            GenerationRequest {
                conditioning: vec![Conditioning::Depth { image: img(8, 8) }],
                ..audio_req()
            },
        ];
        for (i, req) in gap_cases.iter().enumerate() {
            let err = c.validate_request_audio("tts", req).unwrap_err();
            assert!(
                matches!(err, Error::Unsupported(_)),
                "gap case {i} should be typed Unsupported, got {err:?}"
            );
        }
        // Malformed values stay `Msg`.
        let msg_cases: Vec<GenerationRequest> = vec![
            GenerationRequest {
                count: 0,
                ..audio_req()
            },
            GenerationRequest {
                count: 2,
                ..audio_req()
            },
            GenerationRequest {
                steps: Some(0),
                ..audio_req()
            },
        ];
        for (i, req) in msg_cases.iter().enumerate() {
            let err = c.validate_request_audio("tts", req).unwrap_err();
            assert!(
                matches!(err, Error::Msg(_)),
                "msg case {i} should be a Msg range error, got {err:?}"
            );
        }
    }

    #[test]
    fn multi_speaker_script_gating_is_additive_and_typed() {
        // sc-12848: a script is a capability gap on a non-multi-speaker model (the default), and
        // gated by supports_multi_speaker / max_speakers / the closed voice surface when advertised.
        let seg = |sp: &str| SpeechSegment {
            text: "hello".into(),
            speaker: Some(sp.into()),
            ..Default::default()
        };
        let script_req = |caps_voices: Vec<&'static str>,
                          ms: bool,
                          max: Option<u32>,
                          segs: Vec<SpeechSegment>| {
            let c = Capabilities {
                audio_voices: caps_voices,
                supports_multi_speaker: ms,
                max_speakers: max,
                max_count: 1,
                ..Default::default()
            };
            let req = GenerationRequest {
                audio: Some(AudioParams {
                    script: Some(segs),
                    ..Default::default()
                }),
                ..audio_req()
            };
            c.validate_request_audio("tts", &req)
        };

        // A single-voice model with no script capability: a script is a typed Unsupported.
        assert!(matches!(
            script_req(vec![], false, None, vec![seg("S1"), seg("S2")]),
            Err(Error::Unsupported(_))
        ));

        // Advertised multi-speaker, opaque labels (empty voice surface): a valid script passes.
        assert!(script_req(vec![], true, Some(2), vec![seg("S1"), seg("S2")]).is_ok());

        // An empty script is a malformed request (Msg), not a capability gap.
        assert!(matches!(
            script_req(vec![], true, None, vec![]),
            Err(Error::Msg(_))
        ));

        // Over the max_speakers cap → range error (Msg).
        assert!(matches!(
            script_req(vec![], true, Some(2), vec![seg("S1"), seg("S2"), seg("S3")]),
            Err(Error::Msg(_))
        ));
        // At the cap, distinct-count dedups repeated labels → OK.
        assert!(script_req(vec![], true, Some(2), vec![seg("S1"), seg("S1"), seg("S2")]).is_ok());

        // A closed voice surface gates script speakers exactly like `audio.voice` (typed Unsupported).
        assert!(script_req(
            vec!["nova", "onyx"],
            true,
            None,
            vec![seg("nova"), seg("onyx")]
        )
        .is_ok());
        assert!(matches!(
            script_req(vec!["nova"], true, None, vec![seg("nova"), seg("ghost")]),
            Err(Error::Unsupported(_))
        ));

        // The additive floor: a request with NO script behaves exactly as before (single-voice).
        let c = Capabilities {
            audio_voices: vec!["nova"],
            max_count: 1,
            ..Default::default()
        };
        let single = GenerationRequest {
            audio: Some(AudioParams {
                voice: Some("nova".into()),
                ..Default::default()
            }),
            ..audio_req()
        };
        assert!(c.validate_request_audio("tts", &single).is_ok());
    }

    #[test]
    fn conversation_history_gating_is_additive_and_typed() {
        // sc-14150: a ConversationHistory is a capability gap on a non-conversational model, gated by
        // supports_conversation_history (+ the conditioning allowlist); when supported the shape must
        // be well-formed; a request with no conversation is byte-for-byte unaffected.
        let user = |t: &str| ConversationTurn {
            role: ConversationRole::User,
            text: t.into(),
            audio: Some(track()),
        };
        let asst = |t: &str, audio: Option<AudioTrack>| ConversationTurn {
            role: ConversationRole::Assistant,
            text: t.into(),
            audio,
        };
        // A bare audio request (no voice/language sub-block, so only the conversation is exercised).
        let conv_req = |caps: Capabilities, turns: Vec<ConversationTurn>| {
            let req = GenerationRequest {
                prompt: "read this aloud".into(),
                width: 0,
                height: 0,
                conditioning: vec![Conditioning::ConversationHistory { turns }],
                ..Default::default()
            };
            caps.validate_request_audio("tts", &req)
        };
        let conv_caps = || Capabilities {
            conditioning: vec![ConditioningKind::ConversationHistory],
            supports_conversation_history: true,
            max_count: 1,
            ..Default::default()
        };

        // A non-conversational model (neither the flag nor the kind): a conversation is a typed gap.
        let plain = Capabilities {
            max_count: 1,
            ..Default::default()
        };
        assert!(matches!(
            conv_req(plain, vec![user("hi"), asst("hello", None)]),
            Err(Error::Unsupported(_))
        ));

        // Advertised: a well-formed conversation (context user turn + an assistant reply to
        // synthesize) passes.
        assert!(conv_req(conv_caps(), vec![user("hi"), asst("hello", None)]).is_ok());

        // Empty conversation → malformed (Msg), not a capability gap.
        assert!(matches!(conv_req(conv_caps(), vec![]), Err(Error::Msg(_))));

        // A blank-text turn → Msg.
        assert!(matches!(
            conv_req(conv_caps(), vec![user("   "), asst("hi", None)]),
            Err(Error::Msg(_))
        ));

        // A User turn with no audio (provided context must carry its audio) → Msg.
        assert!(matches!(
            conv_req(
                conv_caps(),
                vec![
                    ConversationTurn {
                        role: ConversationRole::User,
                        text: "hi".into(),
                        audio: None,
                    },
                    asst("hello", None),
                ]
            ),
            Err(Error::Msg(_))
        ));

        // No assistant turn to synthesize (every turn already carries audio) → Msg.
        assert!(matches!(
            conv_req(
                conv_caps(),
                vec![user("hi"), asst("was said", Some(track()))]
            ),
            Err(Error::Msg(_))
        ));

        // Keyed gate is authoritative: a descriptor listing the kind but leaving the flag unset
        // (inconsistent — the registry conformance sweep also flags it) still rejects a conversation
        // as the typed Unsupported.
        let flag_off = Capabilities {
            conditioning: vec![ConditioningKind::ConversationHistory],
            supports_conversation_history: false,
            max_count: 1,
            ..Default::default()
        };
        assert!(matches!(
            conv_req(flag_off, vec![user("hi"), asst("hello", None)]),
            Err(Error::Unsupported(_))
        ));

        // Additive: a plain request with no conversation validates exactly as before.
        let bare = GenerationRequest {
            prompt: "hi".into(),
            width: 0,
            height: 0,
            ..Default::default()
        };
        let plain2 = Capabilities {
            max_count: 1,
            ..Default::default()
        };
        assert!(plain2.validate_request_audio("tts", &bare).is_ok());
    }

    #[test]
    fn audio_surface_membership_and_ranges_are_enforced() {
        let c = audio_caps();
        let with_audio = |a: AudioParams| GenerationRequest {
            audio: Some(a),
            ..audio_req()
        };
        // Membership gaps → typed Unsupported naming the field.
        let gaps: [(&str, AudioParams); 3] = [
            (
                "audio.voice",
                AudioParams {
                    voice: Some("santa".into()),
                    ..Default::default()
                },
            ),
            (
                "audio.language",
                AudioParams {
                    language: Some("xx".into()),
                    ..Default::default()
                },
            ),
            (
                "audio.sample_rate",
                AudioParams {
                    sample_rate: Some(44_100),
                    ..Default::default()
                },
            ),
        ];
        for (field, params) in gaps {
            let err = c
                .validate_request_audio("tts", &with_audio(params))
                .unwrap_err();
            assert!(matches!(err, Error::Unsupported(_)), "{field}: got {err:?}");
            assert!(err.to_string().contains(field), "{field}: got {err}");
        }
        // Range violations → Msg.
        let over_cap = with_audio(AudioParams {
            target_duration: Some(61.0),
            ..Default::default()
        });
        let err = c.validate_request_audio("tts", &over_cap).unwrap_err();
        assert!(matches!(err, Error::Msg(_)), "got {err:?}");
        assert!(
            err.to_string().contains("audio.target_duration"),
            "got {err}"
        );
        for bad in [0.0, -3.0] {
            let req = with_audio(AudioParams {
                target_duration: Some(bad),
                ..Default::default()
            });
            assert!(
                matches!(
                    c.validate_request_audio("tts", &req).unwrap_err(),
                    Error::Msg(_)
                ),
                "target_duration {bad} must be rejected"
            );
        }
        let bad_bpm = with_audio(AudioParams {
            bpm: Some(0.0),
            ..Default::default()
        });
        assert!(matches!(
            c.validate_request_audio("tts", &bad_bpm).unwrap_err(),
            Error::Msg(_)
        ));
        // No advertised duration cap ⇒ only the sanity cap applies.
        let uncapped = Capabilities {
            max_audio_duration_secs: None,
            ..audio_caps()
        };
        assert!(uncapped
            .validate_request_audio(
                "tts",
                &with_audio(AudioParams {
                    target_duration: Some(3600.0),
                    ..Default::default()
                })
            )
            .is_ok());
        assert!(uncapped
            .validate_request_audio(
                "tts",
                &with_audio(AudioParams {
                    target_duration: Some(MAX_DURATION_SECS + 1.0),
                    ..Default::default()
                })
            )
            .is_err());
    }

    #[test]
    fn audio_floats_join_the_finiteness_floor() {
        // The `AudioParams` floats inherit the F-053/F-001 finiteness floor by construction.
        let c = audio_caps();
        for bad in [f32::NAN, f32::INFINITY, f32::NEG_INFINITY] {
            for (field, params) in [
                (
                    "audio.target_duration",
                    AudioParams {
                        target_duration: Some(bad),
                        ..Default::default()
                    },
                ),
                (
                    "audio.bpm",
                    AudioParams {
                        bpm: Some(bad),
                        ..Default::default()
                    },
                ),
            ] {
                let req = GenerationRequest {
                    audio: Some(params),
                    ..audio_req()
                };
                let err = c.validate_request_audio("tts", &req).unwrap_err();
                assert!(matches!(err, Error::Msg(_)), "{field} {bad} → Msg");
                // ±Inf may trip the range/sanity-cap branch first (same convention as the
                // request-level `duration`); NaN falls through every comparison and must be
                // caught by the finiteness floor naming the field.
                assert!(err.to_string().contains(field), "{field} {bad}: got {err}");
                if bad.is_nan() {
                    assert!(
                        err.to_string().contains("must be finite"),
                        "{field} NaN: got {err}"
                    );
                }
            }
        }
        // The ReferenceAudio conditioning strength is guarded too.
        let cloned = GenerationRequest {
            conditioning: vec![Conditioning::ReferenceAudio {
                audio: track(),
                strength: Some(f32::NAN),
            }],
            ..audio_req()
        };
        let err = c.validate_request_audio("tts", &cloned).unwrap_err();
        assert!(
            err.to_string()
                .contains("conditioning.reference_audio.strength"),
            "got {err}"
        );
    }

    #[test]
    fn reference_audio_is_gated_by_the_conditioning_allowlist() {
        // A visual descriptor that does not advertise ReferenceAudio rejects it, typed.
        let visual = caps();
        let req = GenerationRequest {
            conditioning: vec![Conditioning::ReferenceAudio {
                audio: track(),
                strength: None,
            }],
            ..base_req()
        };
        let err = visual.validate_request("m", &req).unwrap_err();
        assert!(matches!(err, Error::Unsupported(_)), "got {err:?}");
        assert_eq!(
            Conditioning::ReferenceAudio {
                audio: track(),
                strength: None
            }
            .kind(),
            ConditioningKind::ReferenceAudio
        );
    }

    #[test]
    fn audio_output_and_modality_variants_carry_the_track() {
        // The additive output variant round-trips the host-type track (tensor-free invariant).
        let out = GenerationOutput::Audio(track());
        match out {
            GenerationOutput::Audio(t) => {
                assert_eq!(t.sample_rate, 24_000);
                assert_eq!(t.channels, 1);
                assert_eq!(t.samples.len(), 16);
            }
            other => panic!("expected Audio output, got {other:?}"),
        }
        assert_ne!(Modality::Audio, Modality::Both);
        // A visual request is untouched by the audio block: `Default` carries `audio: None`.
        assert!(GenerationRequest::default().audio.is_none());
    }

    #[test]
    fn non_finite_guidance_and_true_cfg_are_rejected() {
        // F-053: a NaN passes `x > 1.0`-style checks; the floor rejects non-finite explicitly. Uses
        // a caps that advertises guidance/true_cfg so the finiteness branch (not the support gate) runs.
        let c = Capabilities {
            supports_guidance: true,
            supports_true_cfg: true,
            ..caps()
        };
        for bad in [f32::NAN, f32::INFINITY, f32::NEG_INFINITY] {
            let g = GenerationRequest {
                guidance: Some(bad),
                ..base_req()
            };
            let err = c.validate_request("m", &g).unwrap_err();
            assert!(
                matches!(err, Error::Msg(_)),
                "guidance {bad} → Msg range error"
            );
            assert!(err.to_string().contains("guidance must be finite"));
            let t = GenerationRequest {
                true_cfg: Some(bad),
                ..base_req()
            };
            assert!(matches!(
                c.validate_request("m", &t).unwrap_err(),
                Error::Msg(_)
            ));
        }
        // Finite guidance/true_cfg still pass.
        assert!(c
            .validate_request(
                "m",
                &GenerationRequest {
                    guidance: Some(3.5),
                    true_cfg: Some(2.0),
                    ..base_req()
                }
            )
            .is_ok());
    }

    // ---- Prompted audio editing (sc-12847) -------------------------------------------------

    /// An audio-edit capability surface: admits the `AudioEdit` kind and advertises two modes.
    fn edit_caps() -> Capabilities {
        Capabilities {
            conditioning: vec![ConditioningKind::AudioEdit],
            audio_edit_modes: vec![AudioEditMode::Repaint, AudioEditMode::Extend],
            min_size: 1,
            max_size: 4096,
            max_count: 1,
            ..Default::default()
        }
    }

    fn edit_req(
        mode: AudioEditMode,
        region: Option<TimeRegion>,
        strength: Option<f32>,
    ) -> GenerationRequest {
        GenerationRequest {
            prompt: "x".into(),
            width: 512,
            height: 512,
            conditioning: vec![Conditioning::AudioEdit {
                audio: track(),
                mode,
                region,
                strength,
            }],
            ..Default::default()
        }
    }

    #[test]
    fn audio_edit_kind_and_accessor_round_trip() {
        let region = TimeRegion {
            start_secs: 4.0,
            end_secs: Some(8.0),
        };
        let req = edit_req(AudioEditMode::Repaint, Some(region), Some(0.7));
        assert_eq!(
            req.conditioning[0].kind(),
            ConditioningKind::AudioEdit,
            "AudioEdit maps to its own kind"
        );
        let e = req.audio_edit().expect("audio_edit present");
        assert_eq!(e.mode, AudioEditMode::Repaint);
        assert_eq!(e.region, Some(region));
        assert_eq!(e.strength, Some(0.7));
        assert_eq!(e.audio.sample_rate, 24_000);
        // A request without an AudioEdit yields None.
        assert!(GenerationRequest::default().audio_edit().is_none());
    }

    #[test]
    fn audio_edit_mode_is_gated_by_the_advertised_surface() {
        let c = edit_caps();
        // Advertised modes pass; the region is well-formed.
        assert!(c
            .validate_request(
                "m",
                &edit_req(
                    AudioEditMode::Repaint,
                    Some(TimeRegion {
                        start_secs: 4.0,
                        end_secs: Some(8.0),
                    }),
                    None,
                ),
            )
            .is_ok());
        // An unadvertised mode is a typed capability gap.
        let err = c
            .validate_request("m", &edit_req(AudioEditMode::Cover, None, None))
            .unwrap_err();
        assert!(
            matches!(err, Error::Unsupported(_)),
            "unlisted mode → Unsupported"
        );
        assert!(err.to_string().contains("unsupported audio edit mode"));
        // The whole kind is rejected when not admitted at all.
        let no_edit = Capabilities {
            conditioning: vec![ConditioningKind::Reference],
            ..edit_caps()
        };
        assert!(matches!(
            no_edit
                .validate_request("m", &edit_req(AudioEditMode::Repaint, None, None))
                .unwrap_err(),
            Error::Unsupported(_)
        ));
    }

    #[test]
    fn audio_edit_region_and_strength_are_floored() {
        let c = edit_caps();
        // start < 0 and end <= start are malformed ranges → Msg.
        for region in [
            TimeRegion {
                start_secs: -1.0,
                end_secs: Some(4.0),
            },
            TimeRegion {
                start_secs: 8.0,
                end_secs: Some(4.0),
            },
            TimeRegion {
                start_secs: 4.0,
                end_secs: Some(4.0),
            },
        ] {
            let err = c
                .validate_request("m", &edit_req(AudioEditMode::Repaint, Some(region), None))
                .unwrap_err();
            assert!(matches!(err, Error::Msg(_)), "{region:?} → Msg range error");
        }
        // Non-finite strength / region bounds are caught by the finiteness floor.
        for bad in [f32::NAN, f32::INFINITY] {
            assert!(c
                .validate_request("m", &edit_req(AudioEditMode::Repaint, None, Some(bad)))
                .is_err());
            assert!(c
                .validate_request(
                    "m",
                    &edit_req(
                        AudioEditMode::Repaint,
                        Some(TimeRegion {
                            start_secs: bad,
                            end_secs: None,
                        }),
                        None,
                    ),
                )
                .is_err());
            assert!(c
                .validate_request(
                    "m",
                    &edit_req(
                        AudioEditMode::Repaint,
                        Some(TimeRegion {
                            start_secs: 1.0,
                            end_secs: Some(bad),
                        }),
                        None,
                    ),
                )
                .is_err());
        }
        // A well-formed open-ended region (end None) passes.
        assert!(c
            .validate_request(
                "m",
                &edit_req(
                    AudioEditMode::Extend,
                    Some(TimeRegion {
                        start_secs: 2.0,
                        end_secs: None,
                    }),
                    Some(0.5),
                ),
            )
            .is_ok());
    }

    // ---- Multi-region prompted audio editing (sc-14549) ------------------------------------

    /// A **multi-region** capability surface: admits the distinct `AudioEditRegions` kind.
    fn multi_edit_caps() -> Capabilities {
        Capabilities {
            conditioning: vec![ConditioningKind::AudioEditRegions],
            ..edit_caps()
        }
    }

    fn multi_edit_req(mode: AudioEditMode, regions: Vec<TimeRegion>) -> GenerationRequest {
        GenerationRequest {
            prompt: "x".into(),
            width: 512,
            height: 512,
            conditioning: vec![Conditioning::AudioEditRegions {
                audio: track(),
                mode,
                regions,
                strength: None,
            }],
            ..Default::default()
        }
    }

    fn span(start_secs: f32, end_secs: f32) -> TimeRegion {
        TimeRegion {
            start_secs,
            end_secs: Some(end_secs),
        }
    }

    #[test]
    fn multi_region_audio_edit_kind_and_accessor_round_trip() {
        let regions = vec![span(2.0, 6.0), span(14.0, 18.0)];
        let req = multi_edit_req(AudioEditMode::Repaint, regions.clone());
        assert_eq!(
            req.conditioning[0].kind(),
            ConditioningKind::AudioEditRegions,
            "AudioEditRegions maps to its OWN kind — that is what makes default-deny free"
        );
        let e = req
            .audio_edit_regions()
            .expect("audio_edit_regions present");
        assert_eq!(e.mode, AudioEditMode::Repaint);
        assert_eq!(e.regions, regions.as_slice());
        assert_eq!(e.strength, None);
        assert_eq!(e.audio.sample_rate, 24_000);
        assert!(GenerationRequest::default().audio_edit_regions().is_none());
        // The two carriers do not alias: a multi-region request is invisible to the single-region
        // accessor, and vice versa. This is what keeps the legacy path byte-identical.
        assert!(
            req.audio_edit().is_none(),
            "a multi-region carrier must not be visible through the single-region accessor"
        );
        let single = edit_req(AudioEditMode::Repaint, Some(span(2.0, 6.0)), None);
        assert!(single.audio_edit_regions().is_none());
    }

    /// Default-deny, with **no capability flag and no provider code**: a provider advertising only
    /// the single-region `AudioEdit` kind refuses a multi-region request as typed `Unsupported`.
    ///
    /// This is the acceptance criterion "every non-multi-region audio provider rejects multi-region
    /// cleanly", proven against the shared allowlist rather than against any one provider.
    #[test]
    fn a_single_region_provider_refuses_multi_region_as_typed_unsupported() {
        let single_only = edit_caps();
        assert!(
            single_only.accepts(ConditioningKind::AudioEdit)
                && !single_only.accepts(ConditioningKind::AudioEditRegions),
            "precondition: the surface advertises single-region editing only"
        );
        let err = single_only
            .validate_request(
                "m",
                &multi_edit_req(
                    AudioEditMode::Repaint,
                    vec![span(2.0, 6.0), span(14.0, 18.0)],
                ),
            )
            .unwrap_err();
        assert!(
            matches!(err, Error::Unsupported(_)),
            "multi-region on a single-region surface → Unsupported, got {err:?}"
        );
        assert!(err.to_string().contains("AudioEditRegions"));
        // And the same surface still accepts the single-region request it always did — so the
        // refusal is a real discrimination, not a blanket rejection.
        assert!(single_only
            .validate_request(
                "m",
                &edit_req(AudioEditMode::Repaint, Some(span(2.0, 6.0)), None)
            )
            .is_ok());
    }

    #[test]
    fn multi_region_reuses_the_advertised_audio_edit_mode_surface() {
        let c = multi_edit_caps();
        assert!(c
            .validate_request(
                "m",
                &multi_edit_req(
                    AudioEditMode::Repaint,
                    vec![span(2.0, 6.0), span(14.0, 18.0)]
                )
            )
            .is_ok());
        // Order is not significant: the reversed list is equally valid.
        assert!(c
            .validate_request(
                "m",
                &multi_edit_req(
                    AudioEditMode::Repaint,
                    vec![span(14.0, 18.0), span(2.0, 6.0)]
                )
            )
            .is_ok());
        // Overlapping / touching / duplicate spans are ACCEPTED — the provider merges them.
        for regions in [
            vec![span(2.0, 6.0), span(4.0, 8.0)], // overlapping
            vec![span(2.0, 6.0), span(6.0, 8.0)], // touching
            vec![span(2.0, 6.0), span(2.0, 6.0)], // duplicate
            vec![span(2.0, 6.0), span(3.0, 4.0)], // contained
        ] {
            assert!(
                c.validate_request(
                    "m",
                    &multi_edit_req(AudioEditMode::Repaint, regions.clone())
                )
                .is_ok(),
                "{regions:?} must be accepted and normalized by the provider, not rejected"
            );
        }
        // An unadvertised mode is the same typed capability gap the single-region path gives.
        let err = c
            .validate_request(
                "m",
                &multi_edit_req(AudioEditMode::Cover, vec![span(2.0, 6.0)]),
            )
            .unwrap_err();
        assert!(matches!(err, Error::Unsupported(_)));
        assert!(err.to_string().contains("unsupported audio edit mode"));
    }

    #[test]
    fn multi_region_list_shape_is_gated() {
        let c = multi_edit_caps();
        // Empty list → malformed request, not a whole-clip edit.
        let err = c
            .validate_request("m", &multi_edit_req(AudioEditMode::Repaint, vec![]))
            .unwrap_err();
        assert!(matches!(err, Error::Msg(_)));
        assert!(err.to_string().contains("carries no regions"));
        // `end_secs: None` is refused OUTRIGHT, on any position — "to the end of the clip" has no
        // unambiguous meaning when order is not significant.
        for regions in [
            vec![TimeRegion {
                start_secs: 2.0,
                end_secs: None,
            }],
            vec![
                span(2.0, 6.0),
                TimeRegion {
                    start_secs: 14.0,
                    end_secs: None,
                },
            ],
            vec![
                TimeRegion {
                    start_secs: 2.0,
                    end_secs: None,
                },
                span(14.0, 18.0),
            ],
        ] {
            let err = c
                .validate_request(
                    "m",
                    &multi_edit_req(AudioEditMode::Repaint, regions.clone()),
                )
                .unwrap_err();
            assert!(
                matches!(err, Error::Msg(_)) && err.to_string().contains("must state an explicit"),
                "{regions:?} → open-ended refusal, got {err:?}"
            );
        }
    }

    /// **The list defeats the exhaustive-destructuring mechanism, so this is the gate that replaces
    /// it.** Every malformed value is placed in region **two**, never region one.
    ///
    /// `first_nonfinite_float` destructures `GenerationRequest` without `..` so a new *field* breaks
    /// the build and must be classified. A `Vec` field satisfies that once and then hides an
    /// unbounded number of floats behind it: a guard checking only `regions[0]` compiles, passes
    /// every pre-existing test, and lets a NaN in region two reach the provider's mask.
    #[test]
    fn every_region_is_floored_not_just_the_first() {
        let c = multi_edit_caps();
        let good = span(2.0, 6.0);
        for bad in [f32::NAN, f32::INFINITY, f32::NEG_INFINITY] {
            // Non-finite START in region TWO.
            let req = multi_edit_req(
                AudioEditMode::Repaint,
                vec![
                    good,
                    TimeRegion {
                        start_secs: bad,
                        end_secs: Some(18.0),
                    },
                ],
            );
            assert_eq!(
                req.first_nonfinite_float().map(|(f, _)| f),
                Some("conditioning.audio_edit_regions.regions.start_secs"),
                "a non-finite start in region TWO must be found ({bad})"
            );
            assert!(c.validate_request("m", &req).is_err());

            // Non-finite END in region TWO.
            let req = multi_edit_req(
                AudioEditMode::Repaint,
                vec![
                    good,
                    TimeRegion {
                        start_secs: 14.0,
                        end_secs: Some(bad),
                    },
                ],
            );
            assert_eq!(
                req.first_nonfinite_float().map(|(f, _)| f),
                Some("conditioning.audio_edit_regions.regions.end_secs"),
                "a non-finite end in region TWO must be found ({bad})"
            );
            assert!(c.validate_request("m", &req).is_err());

            // And in region THREE, so the gate does not merely check "the last one".
            let req = multi_edit_req(
                AudioEditMode::Repaint,
                vec![
                    good,
                    span(8.0, 10.0),
                    TimeRegion {
                        start_secs: bad,
                        end_secs: Some(18.0),
                    },
                ],
            );
            assert!(req.first_nonfinite_float().is_some());
            assert!(c.validate_request("m", &req).is_err());

            // Strength still joins the floor.
            let mut req = multi_edit_req(AudioEditMode::Repaint, vec![good]);
            let Conditioning::AudioEditRegions { strength, .. } = &mut req.conditioning[0] else {
                unreachable!()
            };
            *strength = Some(bad);
            assert_eq!(
                req.first_nonfinite_float().map(|(f, _)| f),
                Some("conditioning.audio_edit_regions.strength")
            );
        }
        // Control: a wholly well-formed multi-region request has no non-finite float at all, so the
        // assertions above discriminate rather than firing on everything.
        let ok = multi_edit_req(
            AudioEditMode::Repaint,
            vec![good, span(8.0, 10.0), span(14.0, 18.0)],
        );
        assert_eq!(ok.first_nonfinite_float(), None);
        assert!(c.validate_request("m", &ok).is_ok());
    }

    /// The non-finite refusal must name **which** region is malformed.
    ///
    /// `first_nonfinite_float` keys are `&'static str`, so its multi-region arm can only report
    /// `…regions.start_secs` with no index — and the two range guards that *do* carry an index
    /// (`start < 0`, `end <= start`) never fire on a NaN, because both comparisons evaluate `false`
    /// for NaN. Without the indexed pass in `validate_request` the caller of a ten-region repaint is
    /// told a region is malformed but not which one. The bad value is placed in region **two** so a
    /// guard that reached only `regions[0]` fails here, and in region **one** so a pass that skips
    /// `regions[0]` fails too.
    #[test]
    fn a_non_finite_region_bound_is_reported_with_its_index() {
        let c = multi_edit_caps();
        let good = span(2.0, 6.0);
        for bad in [f32::NAN, f32::INFINITY, f32::NEG_INFINITY] {
            // Non-finite START in region TWO (index 1).
            let err = c
                .validate_request(
                    "m",
                    &multi_edit_req(
                        AudioEditMode::Repaint,
                        vec![
                            good,
                            TimeRegion {
                                start_secs: bad,
                                end_secs: Some(18.0),
                            },
                        ],
                    ),
                )
                .unwrap_err();
            let msg = err.to_string();
            assert!(
                matches!(err, Error::Msg(_)) && msg.contains("region 1") && msg.contains("start"),
                "a non-finite start in region TWO must name `region 1`, got {msg:?} ({bad})"
            );

            // Non-finite END in region THREE (index 2), so the message is not merely reporting a
            // constant index — and so the pass is not checking only "the second one".
            let err = c
                .validate_request(
                    "m",
                    &multi_edit_req(
                        AudioEditMode::Repaint,
                        vec![
                            good,
                            span(8.0, 10.0),
                            TimeRegion {
                                start_secs: 14.0,
                                end_secs: Some(bad),
                            },
                        ],
                    ),
                )
                .unwrap_err();
            let msg = err.to_string();
            assert!(
                matches!(err, Error::Msg(_)) && msg.contains("region 2") && msg.contains("end"),
                "a non-finite end in region THREE must name `region 2`, got {msg:?} ({bad})"
            );

            // Non-finite START in region ONE (index 0), the symmetric twin of the two cases above:
            // a pass that *skips* `regions[0]` drops the index and falls through to the index-free
            // backstop, which the region-1 and region-2 cases alone do not catch.
            let err = c
                .validate_request(
                    "m",
                    &multi_edit_req(
                        AudioEditMode::Repaint,
                        vec![
                            TimeRegion {
                                start_secs: bad,
                                end_secs: Some(6.0),
                            },
                            span(8.0, 10.0),
                        ],
                    ),
                )
                .unwrap_err();
            let msg = err.to_string();
            assert!(
                matches!(err, Error::Msg(_)) && msg.contains("region 0") && msg.contains("start"),
                "a non-finite start in region ONE must name `region 0`, got {msg:?} ({bad})"
            );
        }
        // Discrimination: a well-formed multi-region request produces no error at all, so the
        // assertions above are not firing on everything. And the *index-free* backstop is still
        // wired — a bespoke provider that calls `ensure_finite_floats` directly (never entering
        // `validate_request`) still rejects the same request.
        let ok = multi_edit_req(AudioEditMode::Repaint, vec![good, span(8.0, 10.0)]);
        assert!(c.validate_request("m", &ok).is_ok());
        let nan_req = multi_edit_req(
            AudioEditMode::Repaint,
            vec![
                good,
                TimeRegion {
                    start_secs: f32::NAN,
                    end_secs: Some(18.0),
                },
            ],
        );
        assert!(
            nan_req.ensure_finite_floats().is_err(),
            "the index-free floor stays the backstop for callers that bypass validate_request"
        );
    }

    /// Malformed ranges are gated on **every** region, again with the bad value in region two.
    #[test]
    fn every_region_range_is_gated_not_just_the_first() {
        let c = multi_edit_caps();
        for bad in [span(-1.0, 4.0), span(8.0, 4.0), span(4.0, 4.0)] {
            let err = c
                .validate_request(
                    "m",
                    &multi_edit_req(AudioEditMode::Repaint, vec![span(20.0, 24.0), bad]),
                )
                .unwrap_err();
            assert!(
                matches!(err, Error::Msg(_)) && err.to_string().contains("region 1"),
                "{bad:?} in region two → Msg naming index 1, got {err:?}"
            );
        }
    }

    // ---- Video→audio (Foley) sync conditioning (sc-13436) ----------------------------------

    /// A video→audio (Foley) capability surface: a `Modality::Audio` model that admits the
    /// `VideoSync` kind and advertises a duration cap. No visual size bounds (audio floor).
    fn foley_caps() -> Capabilities {
        Capabilities {
            conditioning: vec![ConditioningKind::VideoSync],
            max_audio_duration_secs: Some(30.0),
            max_count: 1,
            ..Default::default()
        }
    }

    /// A video→audio request: a silent clip's frames plus a prompt, size left at the unused 0x0 and
    /// the frame rate on `fps` (never on the variant).
    fn foley_req(frame_count: usize) -> GenerationRequest {
        GenerationRequest {
            prompt: "footsteps on gravel".into(),
            width: 0,
            height: 0,
            fps: Some(24),
            conditioning: vec![Conditioning::VideoSync {
                frames: (0..frame_count).map(|_| img(8, 8)).collect(),
            }],
            ..Default::default()
        }
    }

    #[test]
    fn video_sync_maps_to_its_own_kind() {
        // The variant is distinct from the LTX clip kinds — its discriminant is VideoSync, not
        // VideoClip / ControlClip.
        let req = foley_req(3);
        assert_eq!(req.conditioning[0].kind(), ConditioningKind::VideoSync);
        // It is not collected by the LTX in-context clip / control accessors (a Foley clip is not an
        // extend_clip / replace_person input).
        assert!(req.video_clips().is_empty());
        assert!(req.control_clip().is_none());
        assert!(req.keyframes().is_empty());
    }

    #[test]
    fn video_sync_accepted_when_advertised() {
        let c = foley_caps();
        assert!(c.validate_request_audio("foley", &foley_req(4)).is_ok());
    }

    #[test]
    fn video_sync_unsupported_on_a_non_advertising_model() {
        // F-008: a model whose `conditioning` does not list `VideoSync` rejects it as the typed
        // Error::Unsupported (a capability gap), not a stringified Msg.
        let c = Capabilities {
            conditioning: vec![ConditioningKind::ReferenceAudio],
            max_count: 1,
            ..Default::default()
        };
        let err = c.validate_request_audio("tts", &foley_req(2)).unwrap_err();
        assert!(
            matches!(err, Error::Unsupported(_)),
            "un-advertised VideoSync → typed Unsupported, got {err:?}"
        );
    }

    #[test]
    fn video_sync_empty_frames_is_a_msg_range_error() {
        // An empty clip is a malformed request (nothing to condition on) → Error::Msg, even on a model
        // that admits the kind.
        let c = foley_caps();
        let err = c
            .validate_request_audio("foley", &foley_req(0))
            .unwrap_err();
        assert!(
            matches!(err, Error::Msg(_)),
            "empty VideoSync frames → Msg, got {err:?}"
        );
        assert!(err.to_string().contains("carries no frames"));
    }

    // ---- Reference video conditioning (sc-17149) -------------------------------------------

    /// A video model that admits the `ReferenceVideo` kind.
    fn ref_video_caps() -> Capabilities {
        Capabilities {
            conditioning: vec![ConditioningKind::ReferenceVideo],
            max_count: 1,
            min_size: 8,
            max_size: 2048,
            ..Default::default()
        }
    }

    /// A reference-video request. `fps` is on the **variant** (the clip's own rate), and the
    /// request's own `fps` is deliberately left unset — the two are different quantities.
    fn ref_video_req(frame_count: usize, fps: f32, audio: Option<AudioTrack>) -> GenerationRequest {
        GenerationRequest {
            prompt: "the same subject, new scene".into(),
            width: 64,
            height: 64,
            conditioning: vec![Conditioning::ReferenceVideo {
                frames: (0..frame_count).map(|_| img(8, 8)).collect(),
                fps,
                audio,
            }],
            ..Default::default()
        }
    }

    #[test]
    fn reference_video_maps_to_its_own_kind() {
        // Distinct from every other video-frame carrier. If this ever collapsed onto VideoClip, an
        // in-context-clip provider would start receiving references it would splice into the output.
        let req = ref_video_req(3, 30.0, None);
        assert_eq!(req.conditioning[0].kind(), ConditioningKind::ReferenceVideo);
        // And it is not collected by the epic-3040 in-context accessors: a reference has no position
        // in the generated timeline, so it is not an extend_clip / keyframe / replace_person input.
        assert!(req.video_clips().is_empty());
        assert!(req.control_clip().is_none());
        assert!(req.keyframes().is_empty());
    }

    #[test]
    fn reference_video_accepted_when_advertised() {
        let c = ref_video_caps();
        c.validate_request("refvid", &ref_video_req(4, 30.0, None))
            .expect("an advertised reference clip is legal");
        // Carrying its own soundtrack changes nothing at the floor — the pairing rules that do
        // exist are per-model and layered by the provider's own validate.
        c.validate_request("refvid", &ref_video_req(4, 24.0, Some(track())))
            .expect("a reference clip with its own soundtrack is legal");
    }

    #[test]
    fn reference_video_unsupported_on_a_non_advertising_model() {
        // F-008: default-deny. A provider that advertises the in-context clip kind but not the
        // reference kind must still reject a reference — this is the arm that makes the two kinds
        // worth separating in the first place.
        let c = Capabilities {
            conditioning: vec![ConditioningKind::VideoClip],
            max_count: 1,
            min_size: 8,
            max_size: 2048,
            ..Default::default()
        };
        let err = c
            .validate_request("ltx", &ref_video_req(2, 24.0, None))
            .unwrap_err();
        assert!(
            matches!(err, Error::Unsupported(_)),
            "un-advertised ReferenceVideo → typed Unsupported, got {err:?}"
        );
        // The capability verdict outranks a payload-*shape* problem: an empty reference on a model
        // that does not admit the kind is still `Unsupported`, because the caller's first problem is
        // that this model cannot do this at all.
        let err = c
            .validate_request("ltx", &ref_video_req(0, 24.0, None))
            .unwrap_err();
        assert!(
            matches!(err, Error::Unsupported(_)),
            "capability gap outranks payload shape, got {err:?}"
        );
        // The finiteness floor is the one exception, and it is deliberate and pre-existing: it runs
        // ahead of the conditioning allowlist for *every* float in the request, so a NaN rate is
        // reported as the non-finite float it is even on a model that would have refused the kind.
        // Pinned here so a future reordering of the floor is a visible decision rather than a
        // silent change of which error a caller sees.
        let err = c
            .validate_request("ltx", &ref_video_req(2, f32::NAN, None))
            .unwrap_err();
        assert!(
            matches!(&err, Error::Msg(m) if m.contains("conditioning.reference_video.fps")),
            "the finiteness floor precedes the allowlist, got {err:?}"
        );
    }

    #[test]
    fn reference_video_empty_frames_is_a_msg_range_error() {
        let c = ref_video_caps();
        let err = c
            .validate_request("refvid", &ref_video_req(0, 24.0, None))
            .unwrap_err();
        assert!(
            matches!(err, Error::Msg(_)),
            "empty ReferenceVideo frames → Msg, got {err:?}"
        );
        assert!(err.to_string().contains("carries no frames"));
    }

    #[test]
    fn reference_video_rejects_a_rate_that_has_no_reading() {
        // Zero and negative are refused as well as non-finite, and that is deliberate: a rate is the
        // divisor of a resample stride, so unlike every other conditioning float 0.0 is not a
        // meaningful inert value here — it makes the set of frames the model reads arbitrary.
        let c = ref_video_caps();
        for bad in [0.0f32, -24.0, f32::NAN, f32::INFINITY] {
            let err = c
                .validate_request("refvid", &ref_video_req(2, bad, None))
                .unwrap_err();
            assert!(matches!(err, Error::Msg(_)), "fps {bad} → Msg, got {err:?}");
        }
        // A legal rate on the same shape passes, so the loop above is not passing because the
        // request is malformed for some unrelated reason.
        c.validate_request("refvid", &ref_video_req(2, 30.0, None))
            .expect("30 fps is a legal rate");
    }

    #[test]
    fn reference_video_rate_joins_the_finiteness_floor() {
        // The floor is a separate mechanism from validate()'s range check above, and it names the
        // offending field. Both must cover the rate: the floor is what providers reading the
        // conditioning directly rely on.
        let (field, value) = ref_video_req(2, f32::NAN, None)
            .first_nonfinite_float()
            .expect("NaN fps must be caught");
        assert_eq!(field, "conditioning.reference_video.fps");
        assert!(value.is_nan());
        assert!(ref_video_req(2, 30.0, None)
            .first_nonfinite_float()
            .is_none());
    }

    /// sc-19571 — the four conditioning floats the old `_ => {}` arm swallowed.
    ///
    /// Each of these is a `1 − strength` denoise mask or a blend weight, i.e. the same math the
    /// floor already protected on `Reference` — they were missed only because the conditioning
    /// `match` ended in a wildcard, so a new float-bearing variant compiled clean. The wildcard is
    /// gone; this test is what proves the four arms that replaced it actually fire.
    ///
    /// Mutation guard: delete any ONE of the four arms and exactly one sub-assertion below goes
    /// red — they are checked individually, not as a batch.
    #[test]
    fn keyframe_video_clip_control_clip_and_redux_floats_join_the_finiteness_floor() {
        let with = |c: Conditioning| GenerationRequest {
            prompt: "x".into(),
            conditioning: vec![c],
            ..Default::default()
        };

        let (field, value) = with(Conditioning::Keyframe {
            image: img(8, 8),
            frame_idx: 0,
            strength: f32::NAN,
        })
        .first_nonfinite_float()
        .expect("NaN keyframe strength must be caught");
        assert_eq!(field, "conditioning.keyframe.strength");
        assert!(value.is_nan());

        let (field, value) = with(Conditioning::VideoClip {
            frames: vec![img(8, 8)],
            frame_idx: 0,
            strength: f32::INFINITY,
        })
        .first_nonfinite_float()
        .expect("infinite clip strength must be caught");
        assert_eq!(field, "conditioning.video_clip.strength");
        assert!(value.is_infinite());

        let (field, value) = with(Conditioning::ControlClip {
            frames: vec![img(8, 8)],
            mask: vec![img(8, 8)],
            masking_strength: f32::NAN,
            start_frame: 0,
            mode: ReplacementMode::FaceOnly,
        })
        .first_nonfinite_float()
        .expect("NaN masking_strength must be caught");
        assert_eq!(field, "conditioning.control_clip.masking_strength");
        assert!(value.is_nan());

        // The bad value sits in ref **two** — a guard that only reached `refs[0]` would compile and
        // pass every other assertion here (the `AudioEditRegions` lesson, applied).
        let (field, value) = with(Conditioning::ReduxRefs {
            refs: vec![(img(8, 8), 0.5), (img(8, 8), f32::NAN)],
        })
        .first_nonfinite_float()
        .expect("NaN in the SECOND redux ref must be caught");
        assert_eq!(field, "conditioning.redux_refs.strength");
        assert!(value.is_nan());

        // Finite values on all four pass cleanly — the floor rejects non-finiteness, not the knob.
        let clean = GenerationRequest {
            prompt: "x".into(),
            conditioning: vec![
                Conditioning::Keyframe {
                    image: img(8, 8),
                    frame_idx: -1,
                    strength: 0.25,
                },
                Conditioning::VideoClip {
                    frames: vec![img(8, 8)],
                    frame_idx: 0,
                    strength: 0.75,
                },
                Conditioning::ControlClip {
                    frames: vec![img(8, 8)],
                    mask: vec![img(8, 8)],
                    masking_strength: 0.5,
                    start_frame: 0,
                    mode: ReplacementMode::FaceOnly,
                },
                Conditioning::ReduxRefs {
                    refs: vec![(img(8, 8), 0.5), (img(8, 8), 0.25)],
                },
            ],
            ..Default::default()
        };
        assert!(clean.first_nonfinite_float().is_none());
        assert!(clean.ensure_finite_floats().is_ok());
    }

    /// sc-13884: the default request carries no phases (single-phase, byte-for-byte the pre-13884
    /// behavior), and a `phases: Some([...])` request round-trips its typed phase list through a clone.
    #[test]
    fn phases_default_none_and_round_trip() {
        // Default = single-phase, unaffected.
        assert_eq!(GenerationRequest::default().phases, None);

        // A two-phase Raw→Raw+turbo-LoRA split: phase 1 = 20 steps, CFG on, base-only; phase 2 = 8
        // steps, CFG off, turbo LoRA (load-time adapter 0) at weight 0.8.
        let phases = vec![
            GenerationPhase {
                steps: 20,
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
        ];
        let req = GenerationRequest {
            prompt: "a phased render".into(),
            phases: Some(phases.clone()),
            ..Default::default()
        };
        // The typed phases survive a clone unchanged (no serde in this contract — Clone is the
        // transport the worker uses to hand a request across the thread boundary).
        assert_eq!(req.clone().phases, Some(phases));
        // The flat steps knob is left at its default None — the total budget is the phases' sum.
        assert_eq!(req.steps, None);
    }

    /// sc-13884: a NaN/Inf on a phase guidance OR a phase-adapter weight is caught by the shared
    /// finiteness floor, exactly like the flat float knobs (a NaN in the phase forward would silently
    /// poison the guidance / adapter-scale math otherwise).
    #[test]
    fn phase_floats_join_the_finiteness_floor() {
        let bad_guidance = GenerationRequest {
            phases: Some(vec![GenerationPhase {
                steps: 4,
                guidance: Some(f32::NAN),
                adapters: vec![],
            }]),
            ..Default::default()
        };
        assert_eq!(
            bad_guidance.first_nonfinite_float().map(|(f, _)| f),
            Some("phases.guidance")
        );
        assert!(bad_guidance.ensure_finite_floats().is_err());

        let bad_weight = GenerationRequest {
            phases: Some(vec![GenerationPhase {
                steps: 4,
                guidance: Some(0.0),
                adapters: vec![PhaseAdapter {
                    adapter: 0,
                    weight: Some(f32::INFINITY),
                }],
            }]),
            ..Default::default()
        };
        assert_eq!(
            bad_weight.first_nonfinite_float().map(|(f, _)| f),
            Some("phases.adapter.weight")
        );

        // A finite two-phase request passes the floor untouched.
        let ok = GenerationRequest {
            phases: Some(vec![
                GenerationPhase {
                    steps: 4,
                    guidance: Some(3.5),
                    adapters: vec![],
                },
                GenerationPhase {
                    steps: 4,
                    guidance: None,
                    adapters: vec![PhaseAdapter {
                        adapter: 1,
                        weight: None,
                    }],
                },
            ]),
            ..Default::default()
        };
        assert_eq!(ok.first_nonfinite_float(), None);
    }
}
