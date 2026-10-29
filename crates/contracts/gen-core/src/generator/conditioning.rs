//! `Conditioning` and its kind/mode enums.

use super::*;

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
    /// A reference image that **carries its own alpha channel** — a transparent layer being
    /// edited, or a previously extracted subject being re-composed (sc-24111). The RGBA sibling of
    /// [`Reference`](Self::Reference), with the same ordering semantics: it takes its place in the
    /// request's ordered `conditioning` list and may be freely interleaved with RGB
    /// [`Reference`](Self::Reference) / [`MultiReference`](Self::MultiReference) entries.
    ///
    /// A **distinct variant**, not an alpha field on `Reference`, for the reason
    /// [`RgbaImage`] is a distinct type: a provider that has never thought about alpha cannot be
    /// handed a transparent reference and quietly encode whatever colour sits behind `A = 0` as if
    /// it were content. Default-deny on [`ConditioningKind::ReferenceRgba`] makes that free.
    ///
    /// **A transparent reference is not the same request as its flattened self.** The alpha is
    /// consumed, not decoration: on Qwen-Image 2.1 the VAE encodes all four channels while the
    /// vision tower reads the reference composited over white, which is exactly how upstream feeds
    /// it (`img.convert("RGBA")`, `white.paste(img, mask=A)` for the processor copy,
    /// `image_processor.preprocess(img)` for the VAE copy). Sending the flattened RGB instead
    /// would hand the VAE white pixels where the layer is transparent.
    ///
    /// `strength` mirrors [`Reference`](Self::Reference)'s; a provider with no img2img strength
    /// (Qwen-Image 2.1) requires it unset or exactly `1.0`.
    ReferenceRgba {
        image: RgbaImage,
        strength: Option<f32>,
    },
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
            Conditioning::ReferenceRgba { .. } => ConditioningKind::ReferenceRgba,
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
    /// A reference image carrying its own alpha channel ([`Conditioning::ReferenceRgba`],
    /// sc-24111). A **distinct** kind from [`Reference`](Self::Reference), deliberately: that is
    /// what makes default-deny free. Every provider that predates transparent references keeps
    /// advertising `Reference` alone, and [`Capabilities::accepts`] then refuses a transparent
    /// reference as a typed [`Error::Unsupported`] with no flag and no provider code — rather than
    /// letting it through to a preprocessor that would flatten or misread the fourth channel.
    ReferenceRgba,
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
