//! Per-family request parameter groups (memory, audio, song, speech, conversation, multi-phase).

use super::*;

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
    /// Whether a block-causal denoiser keeps its prefix key/value cache for this request
    /// (sc-25905). `None` ⇒ the provider's own default, which for this domain is
    /// [`PrefixKvCacheMode::Prefill`] — the path that holds nothing, so a request admitted against a
    /// budget that never priced the cache cannot run above it. A caller that has priced the cache
    /// and found it fits selects [`PrefixKvCacheMode::Cached`]
    /// ([`PrefixKvCacheMode::select`]). See [`PrefixKvCacheMode`]: numerically equivalent, not
    /// bit-identical, and independent of every rung of the ladder.
    pub prefix_kv_cache: Option<PrefixKvCacheMode>,

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
    /// How many **lyric segments** to render (segment-by-segment autoregressive song models, e.g.
    /// YuE — sc-19382). `None` ⇒ the model default. A model that renders the whole text in one pass
    /// ignores it; one that reads it documents how it resolves a count larger than the lyrics hold.
    pub segments: Option<u32>,
    /// Per-segment autoregressive **token budget** (segment-by-segment song models, sc-19382).
    /// `None` ⇒ the model default. Must be `>= 1` where read.
    pub max_new_tokens_per_segment: Option<u32>,
    /// Autoregressive **repetition penalty** (token-sampled audio models, sc-19382). `None` ⇒ the
    /// model default. Must be finite and `> 0` where read.
    pub repetition_penalty: Option<f32>,
    /// The span of the [`Conditioning::ReferenceAudio`] clip a model conditions on (in-context
    /// audio prompting, e.g. YuE ICL — sc-19382), in seconds. `None` ⇒ the model default window.
    pub reference_region: Option<TimeRegion>,
    /// How the rendered output is kept inside full scale (song models whose reference writes its
    /// output through a limiter, e.g. YuE's `save_audio` — sc-19378). `None` ⇒ the model default
    /// (YuE: [`OutputLimiter::Clamp`], the reference default).
    pub output_limiter: Option<OutputLimiter>,
    /// **Symbolic-plan song controls** (sc-22994 — YuE2): the planning mode, an externally supplied
    /// score, an exact saved plan, cached-latent decoding, the decoder and per-phase token sampling.
    /// `None` ⇒ every model default. Gated by [`Capabilities::supports_symbolic_song`]: a model
    /// that does not plan a score refuses it as the typed [`Error::Unsupported`] instead of
    /// rendering from a plan the caller never asked for.
    pub song: Option<SongParams>,
    /// **Persist the render's reproducibility record** (sc-22994): the model writes its complete
    /// artifact set (score / plan / tokens / latents / effective settings / identities / timings /
    /// integrity records) into [`AudioArtifacts::dir`], publishing it only once the render
    /// succeeds. Gated by [`Capabilities::supports_audio_artifacts`]: a model that keeps no record
    /// refuses it rather than returning audio with nothing written where the caller looks.
    pub artifacts: Option<AudioArtifacts>,
}

/// Symbolic-plan song controls ([`AudioParams::song`], sc-22994 — YuE2). Every field is optional
/// so the block stays additively extensible; each model documents how it resolves defaults and
/// refuses combinations it cannot honour in its own `validate`.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct SongParams {
    /// How the model plans the score before rendering. `None` ⇒ the model default.
    pub planning: Option<SongPlanning>,
    /// An externally supplied score (YuE2: ABC notation) used as the plan instead of sampling one.
    /// Requires a planning mode that has a score.
    pub score: Option<String>,
    /// Restore an exact saved plan instead of planning — its token ids are reused as saved, never
    /// re-tokenized.
    pub plan: Option<SavedPlan>,
    /// Decode the verified cached latents of this completed run directory instead of generating.
    pub cached_latents: Option<std::path::PathBuf>,
    /// Which published decoder renders the latents. `None` ⇒ the model default.
    pub decoder: Option<SongDecoder>,
    /// Token sampling of the score-planning phase. `None` fields ⇒ the model defaults.
    pub score_sampling: Option<TokenSampling>,
    /// Token sampling of the semantic-token phase. `None` fields ⇒ the model defaults.
    pub semantic_sampling: Option<TokenSampling>,
    /// **Plan only** (sc-22988): plan the score and publish the exact plan as an artifacts-only
    /// record (kind `plan`) into [`AudioParams::artifacts`], rendering no audio. Requires
    /// `artifacts` (a plan that is never published is lost — the shared floor refuses the pair
    /// otherwise) and is gated by [`Capabilities::supports_song_plan_only`]. Its result is
    /// [`GenerationReport::artifacts`] with [`GenerationReport::output`] `None`, read through
    /// [`Generator::generate_with_report`]; [`Generator::generate`] has no audio to return and
    /// refuses.
    pub plan_only: bool,
    /// **Zero-shot cover** (sc-22988): plan from a reviewed score instead of sampling one. The
    /// target style is the request `prompt`; the sung lyrics are [`AudioParams::lyrics`]. Gated
    /// by [`Capabilities::supports_song_cover`]. The cover fixes the planning mode and the score
    /// itself, so a model refuses it together with [`Self::planning`], [`Self::score`],
    /// [`Self::plan`] or [`Self::cached_latents`].
    pub cover: Option<SongCover>,
}

/// A zero-shot cover from a reviewed score ([`SongParams::cover`], sc-22988 — YuE2). Symbolic
/// only: the melody travels as the score; no source audio reaches the generator. Transcribing a
/// recording into such a score is a separate step outside this contract.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct SongCover {
    /// Melody-only (harmony removed, accompaniment free) or the full score with its harmony.
    pub mode: SongCoverMode,
    /// The reviewed score (YuE2: the native two-voice ABC dialect).
    pub score: String,
    /// Which melodies a melody-only cover keeps. `None` ⇒ both.
    pub keep: Option<SongCoverVoice>,
    /// When [`AudioParams::lyrics`] is a translation: the source lyrics it translates,
    /// section-aligned. `None` ⇒ the lyrics are the source lyrics.
    pub translated_from: Option<String>,
}

/// How much of a [`SongCover`] score is kept.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum SongCoverMode {
    /// The melody only: every chord symbol removed, so the accompaniment is free.
    Melody,
    /// The full score with its supplied harmony.
    Full,
}

/// Which melodies a melody-only [`SongCover`] keeps.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum SongCoverVoice {
    /// Both melodies.
    Both,
    /// The vocal melody only (the instrumental voice becomes rests on the same grid).
    Vocal,
    /// The instrumental melody only (the vocal voice becomes rests on the same grid).
    Instrumental,
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
