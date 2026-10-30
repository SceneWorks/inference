//! `Capabilities` and request validation against them.

use super::*;

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
    /// Whether this model can render the **HDR** lane through [`GenerationRequest::hdr`] —
    /// scene-linear EXR conditioning in, scene-linear EXR frames plus a BT.2020/HLG master out
    /// (sc-18790).
    ///
    /// `Default` is `false`, so every existing provider is unsupported and the shared floor
    /// rejects an `hdr` request against it as [`Error::Unsupported`]. That rejection is the
    /// point: HDR is not a knob a model can partially honor. A model that quietly ignored it
    /// would return SDR pixels tagged as an HDR render, which is precisely the washed-out-
    /// playback failure the opt-in exists to prevent.
    pub supports_hdr: bool,
    /// Whether this model honours [`GenerationRequest::cfg_interval`] (epic sc-25678). `Default` is
    /// `false`, and the shared floor refuses a `Some` interval against such a model rather than
    /// letting it be accepted and ignored.
    pub supports_cfg_interval: bool,
    /// Whether this model renders a [`GenerationRequest::prompt_batch`] as one batched denoise (epic
    /// sc-25678). `Default` is `false`; the shared floor refuses a non-empty batch otherwise.
    pub supports_prompt_batch: bool,
    /// Whether this model honours [`GenerationRequest::caption_overflow`] (epic sc-25678). `Default`
    /// is `false`; the shared floor refuses a `Some` policy otherwise.
    pub supports_caption_overflow_policy: bool,
    /// The longest prompt, in the model's own text-encoder tokens, that this model conditions on
    /// (sc-24217). `None` (the `Default`) advertises no limit.
    ///
    /// Advisory, like [`supports_preview`](Self::supports_preview): the shared floor does not read
    /// it and a longer prompt is not refused. The provider truncates it to this many tokens and
    /// reports that through [`Generator::generate_with_report`] as a [`GenerationWarning`]
    /// (Supra2-IMG: code `prompt_truncated`, 128 Flan-T5 tokens). The field exists so a consumer can
    /// size a prompt control or warn before dispatch without a per-model table of its own.
    pub max_prompt_tokens: Option<u32>,
    /// Whether this model can emit **RGBA** images with a straight alpha channel through
    /// [`GenerationRequest::output_channels`] = [`OutputChannels::Rgba`] (sc-24111).
    ///
    /// `Default` is `false`, so every existing provider is unsupported and the shared floor
    /// rejects an RGBA request against it as [`Error::Unsupported`]. As with
    /// [`supports_hdr`](Self::supports_hdr), the rejection is the point rather than a nuisance: a
    /// model that quietly returned RGB would answer "extract this subject onto a transparent
    /// background" with the subject flattened over whatever its decoder painted, and the caller
    /// has no channel left to notice it with.
    ///
    /// A provider sets this only when **every** generation route behind the descriptor can emit
    /// the four-channel image — i.e. its decoder is natively RGBA. It is not a request to
    /// synthesise a matte: there is no separate matting model behind this flag, and a provider
    /// must never fabricate an alpha channel (a constant `A = 255` widening of an RGB decode) to
    /// advertise it.
    pub supports_alpha_output: bool,
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
    /// Whether this model renders lyrics **segment by segment** and reads
    /// [`AudioParams::segments`] and [`AudioParams::max_new_tokens_per_segment`] (sc-19382, YuE).
    /// `Default` is `false`, and the shared floor then rejects a request carrying either field as
    /// the typed [`Error::Unsupported`] — a one-pass model would otherwise silently ignore the
    /// requested segment count or budget.
    pub supports_segmented_lyrics: bool,
    /// Whether this model's autoregressive sampler reads [`AudioParams::repetition_penalty`]
    /// (sc-19382). `Default` is `false`; the shared floor then rejects the field as the typed
    /// [`Error::Unsupported`] instead of letting a model that never samples with it drop it.
    pub supports_repetition_penalty: bool,
    /// Whether this model conditions on a caller-chosen **window** of its
    /// [`Conditioning::ReferenceAudio`] clip ([`AudioParams::reference_region`], sc-19382 — YuE
    /// ICL). `Default` is `false`; the shared floor then rejects the field as the typed
    /// [`Error::Unsupported`], since a model that reads the whole clip (or no reference at all)
    /// would render from a different span than the caller asked for.
    pub supports_reference_region: bool,
    /// Whether this model reads [`AudioParams::output_limiter`] (sc-19378 — YuE's `save_audio`
    /// clamp vs. rescale). `Default` is `false`; the shared floor then rejects the field as the
    /// typed [`Error::Unsupported`] instead of letting a model with no such limiter drop it.
    pub supports_output_limiter: bool,
    /// Whether this model plans a symbolic score and reads [`AudioParams::song`] (sc-22994 —
    /// YuE2). `Default` is `false`; the shared floor then rejects the block as the typed
    /// [`Error::Unsupported`].
    pub supports_symbolic_song: bool,
    /// Whether this model publishes a reproducibility record into [`AudioParams::artifacts`]
    /// (sc-22994). `Default` is `false`; the shared floor then rejects the block as the typed
    /// [`Error::Unsupported`].
    pub supports_audio_artifacts: bool,
    /// Whether this model serves an artifacts-only plan ([`SongParams::plan_only`], sc-22988).
    /// `Default` is `false`; the shared floor then rejects `plan_only` as the typed
    /// [`Error::Unsupported`], so a symbolic-song model that cannot stop after planning never
    /// renders audio the caller did not ask for.
    pub supports_song_plan_only: bool,
    /// Whether this model builds a zero-shot cover from a reviewed score ([`SongParams::cover`],
    /// sc-22988). `Default` is `false`; the shared floor then rejects the cover as the typed
    /// [`Error::Unsupported`].
    pub supports_song_cover: bool,
    /// The per-request [`GenerationMemory`] rungs this model honours **without** a
    /// [`MemoryProviderContract`] (sc-22988 — the audio lane, which the image memory ladder does
    /// not cover): [`MemoryStrategy::StagedResidency`] ↔ `stage_residency`,
    /// [`MemoryStrategy::BoundedDecode`] ↔ `tile_vae_decode` / `decode_tile_edge`,
    /// [`MemoryStrategy::BoundedAttention`] ↔ `chunk_attention` / `attention_chunk_size`.
    ///
    /// Weights-free discoverability for a consumer's admission (each provider documents the units
    /// of its parameters). `Default` is empty, which says nothing either way — an image provider
    /// declares its rungs through its [`MemoryProviderContract`] instead — so the shared floor does
    /// not read it; a provider that lists a rung refuses every [`GenerationMemory`] field outside
    /// its list in its own `validate` rather than ignoring it.
    pub request_memory_strategies: &'static [MemoryStrategy],

    // --- LTX-2.5 generation axes (sc-18778) ------------------------------------------------------
    //
    // Each of the three is a REQUEST axis with a real consumer, gated here on the shared floor for
    // the reason the audio surface is: a per-provider check is a check a provider can forget, and a
    // forgotten one means the knob is silently ignored. Every flag below is `false`/`0` by
    // `Default`, so every pre-2.5 provider keeps refusing these requests without editing its
    // descriptor — declaration and enforcement move together.
    /// Whether this model predicts clip length from a **duration head** when the request opts in
    /// via [`GenerationRequest::auto_duration`] (reference `--auto-duration`).
    ///
    /// `Default` is `false`, and the shared floor then rejects an `auto_duration` request as the
    /// typed [`Error::Unsupported`] rather than silently rendering the model's default length — an
    /// opt-in that is quietly dropped is worse than a refusal, because the caller cannot tell the
    /// prediction never ran. A provider sets it `true` only when its load actually binds a duration
    /// head and its generate routes the opt-in through
    /// `gen_core_ltx::duration_head::resolve_request_num_frames`.
    pub supports_auto_duration: bool,
    /// Whether this model places **generated keyframe slots** at interior positions when the
    /// request sets [`GenerationRequest::num_generated_keyframes`] (reference
    /// `--num-generated-keyframes`).
    ///
    /// Requires a transformer with the learned `use_keyframes_abs_pos_embedding` slot marker
    /// (LTX ≥ 2.5). `Default` is `false`: a checkpoint without the marker would denoise the slots
    /// as unmarked tokens — wasted compute carrying no conditioning — so the floor refuses a
    /// positive count as the typed [`Error::Unsupported`].
    pub supports_generated_keyframes: bool,
    /// The largest [`GenerationRequest::temporal_upsample_rounds`] this model honours (reference
    /// `--temporal-upsample-rounds`). `0` (the `Default`) means the DFR temporal path is not
    /// implemented and the floor refuses any positive round count as the typed
    /// [`Error::Unsupported`].
    ///
    /// A count within the advertised bound must do real per-round work: each round temporally
    /// upsamples the latent and re-denoises the keyframe-seam tiles, so `rounds = 2` runs strictly
    /// more forward passes and yields strictly more frames than `rounds = 1`. Advertising a bound
    /// above what the pipeline actually loops over would recreate the inert-knob defect this flag
    /// exists to prevent.
    pub max_temporal_upsample_rounds: u32,
    /// Whether this model can decode through the **diffusion VAE decoder** (DiffVAE) when the
    /// caller stages that component in [`crate::LoadSpec::components`].
    ///
    /// Unlike the two knobs above this is a **load-time** selection, not a request field — it
    /// follows the alternate-decoder contract already in
    /// [`crate::latent::DECODER_OPTIONS`], where staging the option's `component_id` is what
    /// chooses it (the same seam the Wan 2.1 VAE substitution uses). The flag exists so a consumer
    /// can discover the choice weights-free, and so a provider that has NOT wired the branch
    /// refuses the staged component instead of accepting it and silently decoding through the plain
    /// feed-forward decoder — frames that do not match what was asked for, with nothing in the
    /// result saying so.
    ///
    /// `Default` is `false`. A provider sets it `true` only when its decode genuinely branches on
    /// the staged component.
    pub supports_diffusion_decoder: bool,

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
    /// Exact native quantization formats this engine can load (empty slice = none),
    /// whether from a pre-converted/publisher artifact or an explicitly supported
    /// on-the-fly conversion. This is NOT a list of interchangeable bit widths:
    /// FP8R does not inherit Q8 support. A loader must reject incompatible artifacts
    /// and must not claim on-the-fly conversion where only a packaged tier exists.
    /// Read by the worker's capability advertisement instead of a hardcoded per-row flag.
    /// `Default` is `&[]`.
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
pub(super) const MAX_STEPS: u32 = 100_000;
pub(super) const MAX_FRAMES: u32 = 1_000_000;
pub(super) const MAX_FPS: u32 = 100_000;
pub(super) const MAX_DURATION_SECS: f32 = 1_000_000.0;
/// Legacy prompt-enhancement token ceiling. Unrelated providers must not inherit FLUX 3's cap.
pub const MAX_ENHANCE_TOKENS: u32 = 2_048;

/// Resolve the provider's token ceiling, distinct from the effective-caption character bound.
/// Only the two canonical FLUX 3 ids opt into the frozen PUP source's 3584-token budget.
/// Unknown ids and every legacy enhancer retain their existing 2048-token limit.
pub fn max_enhance_tokens_for_model(model_id: &str) -> u32 {
    match model_id {
        "flux3_klein" | "flux3_klein_base" => crate::flux3::pup::MAX_TOKENS,
        _ => MAX_ENHANCE_TOKENS,
    }
}
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
    ///   [`max_speakers`](Self::max_speakers) cap (sc-12848); and `segments` /
    ///   `max_new_tokens_per_segment` / `repetition_penalty` / `reference_region` only when the
    ///   matching [`supports_segmented_lyrics`](Self::supports_segmented_lyrics) /
    ///   [`supports_repetition_penalty`](Self::supports_repetition_penalty) /
    ///   [`supports_reference_region`](Self::supports_reference_region) flag is set (sc-19382), and
    ///   `output_limiter` only when [`supports_output_limiter`](Self::supports_output_limiter) is
    ///   set (sc-19378), and `song` / `artifacts` only when
    ///   [`supports_symbolic_song`](Self::supports_symbolic_song) /
    ///   [`supports_audio_artifacts`](Self::supports_audio_artifacts) is set (sc-22994), and
    ///   `song.plan_only` / `song.cover` only when
    ///   [`supports_song_plan_only`](Self::supports_song_plan_only) /
    ///   [`supports_song_cover`](Self::supports_song_cover) is set, `plan_only` also only together
    ///   with `artifacts` (sc-22988),
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
    pub(super) fn validate_request_inner(
        &self,
        id: &str,
        req: &GenerationRequest,
        check_size: bool,
    ) -> Result<()> {
        if (req.flux3_layout.is_some() || !req.flux3_layout_bindings.is_empty())
            && !matches!(
                id,
                crate::flux3::DISTILLED_MODEL_ID | crate::flux3::BASE_MODEL_ID
            )
        {
            return Err(Error::Unsupported(format!(
                "{id}: flux3_layout is supported only by FLUX 3; remove the layout controls"
            )));
        }
        if req.flux3_reference_output.is_some()
            && !matches!(
                id,
                crate::flux3::DISTILLED_MODEL_ID | crate::flux3::BASE_MODEL_ID
            )
        {
            return Err(Error::Unsupported(format!(
                "{id}: flux3_reference_output is supported only by FLUX 3 reference editing; remove it"
            )));
        }
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
        if req.moss_image.is_some()
            && crate::moss_image::MossImageVariant::from_provider_id(id).is_none()
        {
            return Err(Error::Unsupported(format!(
                "{id}: MOSS controls require a MOSS-Image provider"
            )));
        }
        if req.veda.as_ref().is_some_and(|v| v.enabled) && id != "minimax_h3" {
            return Err(Error::Unsupported(format!(
                "{id}: Veda attention requires MiniMax-H3"
            )));
        }
        // HDR opt-in (sc-18790). Gated on the shared floor, not per-provider, for the reason the
        // audio surface is: a per-provider check is a check a provider can forget, and a forgotten
        // one here means the request renders SDR while the caller believes it asked for HDR — a
        // silently wrong render rather than a loud rejection. `None` (SDR) validates vacuously, so
        // this is inert for every request that has not opted in.
        if req.hdr.is_some() && !self.supports_hdr {
            return Err(Error::Unsupported(format!(
                "{id}: HDR output is not supported by this model"
            )));
        }
        // Guidance interval, prompt batch and caption-overflow policy (epic sc-25678). On the shared
        // floor for the HDR reason: each is a control a provider that does not read it would accept
        // and silently ignore. Unset values validate vacuously.
        if let Some((lo, hi)) = req.cfg_interval {
            if !self.supports_cfg_interval {
                return Err(Error::Unsupported(format!(
                    "{id}: `cfg_interval` is not supported by this model"
                )));
            }
            if !(lo.is_finite() && hi.is_finite() && lo < hi) {
                return Err(Error::Msg(format!(
                    "{id}: cfg_interval ({lo}, {hi}) must be finite with lo < hi"
                )));
            }
        }
        if !req.prompt_batch.is_empty() {
            if !self.supports_prompt_batch {
                return Err(Error::Unsupported(format!(
                    "{id}: a `prompt_batch` is not supported by this model; send one request per \
                     prompt"
                )));
            }
            if !req.prompt.is_empty() {
                return Err(Error::Msg(format!(
                    "{id}: a request carries either `prompt` or a non-empty `prompt_batch`, not both"
                )));
            }
            let images = req.prompt_batch.len() as u64 * req.count as u64;
            if images > self.max_count as u64 {
                return Err(Error::Msg(format!(
                    "{id}: prompt_batch of {} × count {} = {images} images exceeds the {} images \
                     per request this model renders",
                    req.prompt_batch.len(),
                    req.count,
                    self.max_count
                )));
            }
        }
        if req.caption_overflow.is_some() && !self.supports_caption_overflow_policy {
            return Err(Error::Unsupported(format!(
                "{id}: `caption_overflow` is not supported by this model"
            )));
        }
        // RGBA opt-in (sc-24111). On the shared floor for the same reason HDR is: a per-provider
        // check is a check a provider can forget, and a forgotten one returns an opaque RGB image
        // to a caller who asked for transparency, with nothing in the reply to say the alpha was
        // dropped. `Rgb` (the `Default`) validates vacuously, so this is inert for every request
        // that has not opted in.
        if req.output_channels == OutputChannels::Rgba && !self.supports_alpha_output {
            return Err(Error::Unsupported(format!(
                "{id}: RGBA (alpha-channel) output is not supported by this model; it emits RGB \
                 only. Drop `output_channels: Rgba` from the request, or route to a provider whose \
                 capabilities advertise `supports_alpha_output`."
            )));
        }
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
        // --- LTX-2.5 generation axes (sc-18778) --------------------------------------------------
        //
        // Fail-closed, on the shared floor, for the reason the audio surface is: a per-provider
        // check is a check a provider can forget, and a forgotten one means the knob is silently
        // ignored. Every gate below is inert for a request that leaves the axis unset, and every
        // flag defaults to the refusing position, so adding these does not change any existing
        // provider's answer to any existing request.
        //
        // A provider that wants a MORE actionable message (naming the checkpoint generation that
        // does support the axis, say) runs its own check BEFORE calling the floor — that is what
        // `ltx_2_3` does, and why its typed refusals still name 2.5.
        if req.num_generated_keyframes.is_some_and(|n| n > 0) && !self.supports_generated_keyframes
        {
            return Err(Error::Unsupported(format!(
                "{id}: num_generated_keyframes is not supported by this engine (it requires a \
                 checkpoint whose transformer carries the learned generated-keyframe slot marker)"
            )));
        }
        if let Some(rounds) = req.temporal_upsample_rounds {
            if rounds > self.max_temporal_upsample_rounds {
                return Err(if self.max_temporal_upsample_rounds == 0 {
                    // Distinguish "this engine has no DFR temporal path at all" from "you asked for
                    // more rounds than it runs" — the first is a capability gap the caller cannot
                    // fix by lowering the number, the second is a range error they can.
                    Error::Unsupported(format!(
                        "{id}: temporal_upsample_rounds is not supported by this engine (it \
                         requires the LTX-2.5 DFR pipeline: generated keyframe slots + the \
                         temporal latent upsampler)"
                    ))
                } else {
                    Error::Msg(format!(
                        "{id}: temporal_upsample_rounds {rounds} exceeds this engine's maximum {}",
                        self.max_temporal_upsample_rounds
                    ))
                });
            }
        }
        if req.auto_duration.is_some() && !self.supports_auto_duration {
            return Err(Error::Unsupported(format!(
                "{id}: auto_duration is not supported by this engine (it requires a checkpoint \
                 that ships a duration head)"
            )));
        }
        if let Some(fps) = req.fps {
            if fps == 0 {
                return Err(Error::Msg(format!("{id}: fps must be greater than zero")));
            }
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
            // Segmented-song / autoregressive / reference-window controls (sc-19382): each is a
            // capability gap on a model that does not advertise reading it → typed
            // `Error::Unsupported`, so no provider can silently ignore one.
            let gated = [
                (
                    "audio.segments",
                    audio.segments.is_some(),
                    self.supports_segmented_lyrics,
                ),
                (
                    "audio.max_new_tokens_per_segment",
                    audio.max_new_tokens_per_segment.is_some(),
                    self.supports_segmented_lyrics,
                ),
                (
                    "audio.repetition_penalty",
                    audio.repetition_penalty.is_some(),
                    self.supports_repetition_penalty,
                ),
                (
                    "audio.reference_region",
                    audio.reference_region.is_some(),
                    self.supports_reference_region,
                ),
                (
                    "audio.output_limiter",
                    audio.output_limiter.is_some(),
                    self.supports_output_limiter,
                ),
                (
                    "audio.song",
                    audio.song.is_some(),
                    self.supports_symbolic_song,
                ),
                (
                    "audio.artifacts",
                    audio.artifacts.is_some(),
                    self.supports_audio_artifacts,
                ),
                (
                    "audio.song.plan_only",
                    audio.song.as_ref().is_some_and(|s| s.plan_only),
                    self.supports_song_plan_only,
                ),
                (
                    "audio.song.cover",
                    audio.song.as_ref().is_some_and(|s| s.cover.is_some()),
                    self.supports_song_cover,
                ),
            ];
            for (field, present, supported) in gated {
                if present && !supported {
                    return Err(Error::Unsupported(format!(
                        "{id}: {field} is not supported"
                    )));
                }
            }
            // An artifacts-only render publishes its result and returns nothing else: without a
            // record to publish into, the plan would be computed and lost (sc-22988).
            if audio.song.as_ref().is_some_and(|s| s.plan_only) && audio.artifacts.is_none() {
                return Err(Error::Msg(format!(
                    "{id}: audio.song.plan_only requires audio.artifacts (the plan is published \
                     there; nothing else is returned)"
                )));
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
            let max_tokens = max_enhance_tokens_for_model(id);
            if tokens == 0 || tokens > max_tokens {
                return Err(Error::Msg(format!(
                    "{id}: enhance_max_tokens {tokens} outside supported range 1..={max_tokens}"
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
