//! The `Generator` contract — prompt-conditioned synthesis of image, video, **or** audio
//! (or a mix), including multi-modal models. See `docs/MODEL_ARCHITECTURE.md` §3.1.
//!
//! One trait covers everything text→media: T2I, T2V, edit (image+text→image), LTX
//! (text→video+audio), and pure audio synthesis (TTS / music). Modality is a
//! [`ModelDescriptor`] property plus a [`GenerationOutput`] variant — *not* a per-modality
//! trait split (which breaks on multi-modal models).

use crate::media::{AudioChunk, AudioTrack, Image};
use crate::runtime::{CancelFlag, Progress, Quant};
use crate::voice_embed::VoiceEmbedding;
use crate::{Error, Result};

/// A prompt-conditioned media generator. `generate` is **synchronous** (long/blocking; the
/// worker runs each job on its own thread); the request carries a cancel flag and
/// `on_progress` streams step/decode progress.
pub trait Generator {
    /// Identity + capabilities + modality (drives `validate` and consumer UI introspection).
    fn descriptor(&self) -> &ModelDescriptor;

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
            motion_bucket_id: None,
            noise_aug_strength: None,
            decode_chunk_size: None,
            conditioning_fps: None,
            softness: None,
            enhance_prompt: false,
            use_uncensored_enhancer: false,
            enhance_max_tokens: None,
            enhance_temperature: None,
            use_pid: false,
            pid_capture_sigma: None,
            audio: None,
            phases: None,
            cancel: CancelFlag::default(),
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

impl GenerationRequest {
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
            decode_chunk_size: _,
            conditioning_fps: _,
            enhance_prompt: _,
            use_uncensored_enhancer: _,
            enhance_max_tokens: _,
            use_pid: _,
            cancel: _,
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
        // Conditioning-carried floats the floor also owns (F-001): the Control-branch scale and the
        // per-Reference img2img strength both flow into the same denoise/scheduler math.
        for c in conditioning {
            match c {
                Conditioning::Control { scale: Some(s), .. } if !s.is_finite() => {
                    return Some(("conditioning.control.scale", *s));
                }
                Conditioning::Reference {
                    strength: Some(s), ..
                } if !s.is_finite() => {
                    return Some(("conditioning.reference.strength", *s));
                }
                Conditioning::ReferenceAudio {
                    strength: Some(s), ..
                } if !s.is_finite() => {
                    return Some(("conditioning.reference_audio.strength", *s));
                }
                // The audio-edit strength and its region bounds all flow into the edit-window /
                // blend math; a NaN would silently poison the region conversion or the strength
                // gate (sc-12847).
                Conditioning::AudioEdit {
                    strength: Some(s), ..
                } if !s.is_finite() => {
                    return Some(("conditioning.audio_edit.strength", *s));
                }
                Conditioning::AudioEdit {
                    region: Some(r), ..
                } if !r.start_secs.is_finite() => {
                    return Some(("conditioning.audio_edit.region.start_secs", r.start_secs));
                }
                Conditioning::AudioEdit {
                    region:
                        Some(TimeRegion {
                            end_secs: Some(end),
                            ..
                        }),
                    ..
                } if !end.is_finite() => {
                    return Some(("conditioning.audio_edit.region.end_secs", *end));
                }
                Conditioning::VoiceEmbedding {
                    strength: Some(s), ..
                } if !s.is_finite() => {
                    return Some(("conditioning.voice_embedding.strength", *s));
                }
                _ => {}
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
            Conditioning::AudioEdit { .. } => ConditioningKind::AudioEdit,
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
    /// Prompted source-audio editing ([`Conditioning::AudioEdit`]).
    AudioEdit,
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
}

/// What a model supports — drives `validate()` and consumer UI. `Default` is "supports
/// nothing"; a model turns on what it offers (`Capabilities { supports_guidance: true,
/// ..Default::default() }`).
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
    pub mac_only: bool,
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

impl Capabilities {
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
    /// - `width`/`height` within `min_size..=max_size`,
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
    /// empty-prompt rejection, size-alignment (multiple-of-N), frame-count divisibility,
    /// sampler→solver mapping — are layered on top by each model's own `validate`; this is the shared
    /// floor, not a replacement for them.
    pub fn validate_request(&self, id: &str, req: &GenerationRequest) -> Result<()> {
        self.validate_request_inner(id, req, true)
    }

    /// The shared floor **minus the size-range check** — for providers with a "match the driving-media
    /// size" convention (`width`/`height == 0` is a resolve-downstream sentinel, e.g. SCAIL-2 sizing
    /// from the driving-video frames), where the size range would wrongly reject the sentinel. Every
    /// other floor check still runs unconditionally: count / steps / frame / fps / duration caps,
    /// negative-prompt / guidance / true_cfg support gating, finiteness (F-053), sampler / scheduler /
    /// guidance_method membership, and the conditioning allowlist. A provider that calls this must
    /// range-check its resolved size itself (F-158).
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
        // A non-finite guidance / true_cfg / eta / momentum / strength / control_scale / … would flow
        // into the CFG combine, scheduler shift, or conditioning math and NaN-poison the run (a NaN
        // passes `x > 1.0`-style checks silently). The finiteness guard is centralized on the request
        // so every `Option<f32>` knob — including ones added after F-053 — inherits it by construction
        // (F-053 / F-001). `id`-prefixing is dropped from the message here; the field name is enough.
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

    fn img(w: u32, h: u32) -> Image {
        Image {
            width: w,
            height: h,
            pixels: vec![0u8; (w * h * 3) as usize],
        }
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
        // Every non-size violation must still fire on the auto-size path.
        let rejected: Vec<GenerationRequest> = vec![
            // oversized count
            GenerationRequest {
                count: 2,
                ..auto.clone()
            },
            // explicit zero steps
            GenerationRequest {
                steps: Some(0),
                ..auto.clone()
            },
            // unadvertised sampler
            GenerationRequest {
                sampler: Some("unipc".into()),
                ..auto.clone()
            },
            // disallowed conditioning kind
            GenerationRequest {
                conditioning: vec![Conditioning::Depth { image: img(8, 8) }],
                ..auto.clone()
            },
            // non-finite knob (not support-gated) would NaN-poison the run — finiteness still fires
            GenerationRequest {
                strength: Some(f32::NAN),
                ..auto.clone()
            },
        ];
        for (i, req) in rejected.iter().enumerate() {
            assert!(
                c.validate_request_skip_size("m", req).is_err(),
                "skip_size case {i} should have been rejected on the auto-size path"
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
