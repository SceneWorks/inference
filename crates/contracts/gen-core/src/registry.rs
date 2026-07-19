//! Explicit model + transform discovery: provider crates publish registration constants, family
//! crates add them to a [`ProviderRegistryBuilder`], and platform catalogs select the families they
//! ship. This is the Rust equivalent of an ordinary DI composition root with resolve-by-id.

use crate::audio_embed::{AudioEmbedder, AudioEmbedderDescriptor};
use crate::audio_transform::{AudioTransform, AudioTransformDescriptor, AudioTransformKind};
use crate::caption::{Captioner, CaptionerDescriptor};
use crate::generator::{ConditioningKind, Generator, Modality, ModelDescriptor};
use crate::image_embed::{ImageEmbedder, ImageEmbedderDescriptor};
use crate::runtime::{LoadSpec, Quant, WeightsSource};
use crate::text_embed::{TextEmbedder, TextEmbedderDescriptor};
use crate::train::{Trainer, TrainerDescriptor};
use crate::transcribe::{Transcriber, TranscriberDescriptor};
use crate::transform::{Transform, TransformDescriptor};
use crate::voice_embed::{VoiceEmbedder, VoiceEmbedderDescriptor};
use crate::weightsmeta::safetensors_path_bytes;
use crate::{Error, Result};

use std::path::Path;

/// The provider-owned per-component resident-weight estimate (bytes), used by pre-load fit gates for
/// staged residency (sc-10894/sc-11924). Each component defaults to its summed on-disk
/// `.safetensors` size; a provider whose load materializes a larger representation must replace that
/// component with the conservative in-memory size:
///
/// - `text_encoder` — the phase-A prompt encoder(s) that [`OffloadPolicy::Sequential`](crate::runtime::OffloadPolicy)
///   drops *before* the heavy render bundle loads (one or more, e.g. SDXL's two CLIPs, SD3's three);
/// - `dit` — the heavy transformer / U-Net (the "DiT"), the dominant render-phase component;
/// - `vae` — the autoencoder, co-resident with the DiT through the render.
///
/// Why a provider owns this rather than the consumer inferring it: the Sequential/staged peak is
/// `max(text_encoder, dit + vae)` (the encoder is freed before the renderer materializes — see
/// [`LoadPhase::Renderer`](crate::runtime::LoadPhase::Renderer)), not the resident sum. A consumer that
/// guesses the text-encoder size from `text_encoder*` subdir NAMING reads **zero** for any family whose
/// encoder is not under such a subdir — or has no separable encoder at all (a flat unified checkpoint) —
/// collapsing the staged peak back to the resident peak so no saving is ever selected. Each provider,
/// by contrast, computes the split from the exact subdir paths its own loader resolves.
///
/// The default constructors are tensor-free on-disk sums ([`crate::safetensors_dir_bytes`]) — **zero**
/// MLX allocation and no whole-file reads — so this remains safe in a pre-load gate. A component a
/// model does not have (or cannot separate) is `0`.
///
/// **On-disk byte SUMS, not load-exact.** Each field totals *every* `.safetensors` under the named
/// path(s), which can exceed what a single load materializes: one component dir may ship multiple
/// interchangeable variant files (anima's `diffusion_models/` holds the base/aesthetic/turbo DiTs, but
/// a run loads exactly one — so `dit` over-counts by the unused variants), or side-by-side dtype shards
/// (an SD3 `text_encoder_3/` carrying both f32 and fp16 double-counts). Today the worker consumes only
/// `text_encoder` plus the true whole-model total; `dit` / `vae` are **informational** for a future
/// consumer, which must treat them as an upper-bound on-disk footprint, not the resident size of one load.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct PerComponentBytes {
    pub text_encoder: u64,
    pub dit: u64,
    pub vae: u64,
}

impl PerComponentBytes {
    /// Best-effort footprint for a diffusers-style snapshot: sum the `.safetensors` bytes under each
    /// named component subdir of the spec's weights DIRECTORY. Each list is the exact subdir(s) the
    /// caller's own loader resolves — `["text_encoder", "text_encoder_2"]` for the two SDXL CLIPs,
    /// `["unet"]` / `["transformer"]` for the DiT, `["vae"]` — so the paths are always correct per
    /// engine. A subdir that is absent contributes `0` ([`crate::safetensors_dir_bytes`]).
    ///
    /// Each name may be a component *subdir* OR a flat component *file* ([`safetensors_path_bytes`]),
    /// so this also covers the bernini / anima flat-file layouts. Errors only when `spec.weights` is a
    /// single [`WeightsSource::File`]: a one-file checkpoint has no component tree to split (the consumer
    /// then falls back to whole-file / resident accounting).
    pub fn from_spec_subdirs(
        spec: &LoadSpec,
        text_encoder: &[&str],
        dit: &[&str],
        vae: &[&str],
    ) -> Result<Self> {
        let root = match &spec.weights {
            WeightsSource::Dir(p) => p.as_path(),
            WeightsSource::File(_) => return Err(Error::Msg(
                "per-component footprint requires a snapshot directory, not a single .safetensors \
                     file"
                    .to_owned(),
            )),
        };
        Ok(Self::from_root_subdirs(root, text_encoder, dit, vae))
    }

    /// Sum each component's `.safetensors` bytes under an already-resolved `root` — for a provider whose
    /// component tree is NOT directly under `spec.weights` (e.g. anima's `split_files/` nesting resolves
    /// the root itself, then names `text_encoders` / `diffusion_models` / `vae` under it). Each name is a
    /// subdir or a flat file ([`safetensors_path_bytes`]); a missing one contributes `0`. Infallible —
    /// the root is the caller's to validate.
    pub fn from_root_subdirs(
        root: &Path,
        text_encoder: &[&str],
        dit: &[&str],
        vae: &[&str],
    ) -> Self {
        let sum = |names: &[&str]| -> u64 {
            names
                .iter()
                .map(|n| safetensors_path_bytes(root.join(n)))
                .sum()
        };
        Self {
            text_encoder: sum(text_encoder),
            dit: sum(dit),
            vae: sum(vae),
        }
    }
}

/// A generator provider's registration — `descriptor` for introspection (no weights loaded),
/// `load` to construct the model, and the optional [`footprint`](Self::footprint) size seam.
/// ≈ `services.AddKeyedSingleton<IGenerator>("id", factory)`.
#[derive(Clone, Copy)]
pub struct ModelRegistration {
    pub descriptor: fn() -> ModelDescriptor,
    pub load: fn(&LoadSpec) -> Result<Box<dyn Generator>>,
    /// Optional per-component on-disk footprint (sc-10894) — `Some` for a provider that has declared its
    /// [`PerComponentBytes`] split (via `register_generators! { … ; footprint = … }`), `None` otherwise.
    /// `None` is the default so **every** provider that does not set it registers unchanged; a consumer
    /// reaching [`footprint`](Self::footprint) then gets `Ok(None)` and falls back to its own
    /// accounting. Mirrors the [`load`](Self::load) fn-pointer shape (a spec in, a `Result` out).
    pub footprint: Option<fn(&LoadSpec) -> Result<PerComponentBytes>>,
}

/// A transform provider's registration (parallel to [`ModelRegistration`]).
#[derive(Clone, Copy)]
pub struct TransformRegistration {
    pub descriptor: fn() -> TransformDescriptor,
    pub load: fn(&LoadSpec) -> Result<Box<dyn Transform>>,
}

/// An audio-transform provider's registration (parallel to [`TransformRegistration`]; sc-12839). The
/// audio sibling of the (image) transform: a non-prompt audio→audio / audio→stems transform (voice
/// conversion, stem separation, super-resolution) resolved and loaded exactly like every other kind.
#[derive(Clone, Copy)]
pub struct AudioTransformRegistration {
    pub descriptor: fn() -> AudioTransformDescriptor,
    pub load: fn(&LoadSpec) -> Result<Box<dyn AudioTransform>>,
}

/// A trainer provider's registration (parallel to [`ModelRegistration`]) — `descriptor` for
/// introspection, `load` to construct the trainer with its (frozen) base model from a [`LoadSpec`].
#[derive(Clone, Copy)]
pub struct TrainerRegistration {
    pub descriptor: fn() -> TrainerDescriptor,
    pub load: fn(&LoadSpec) -> Result<Box<dyn Trainer>>,
}

/// A captioner provider's registration (parallel to [`ModelRegistration`]).
#[derive(Clone, Copy)]
pub struct CaptionerRegistration {
    pub descriptor: fn() -> CaptionerDescriptor,
    pub load: fn(&LoadSpec) -> Result<Box<dyn Captioner>>,
}

/// A transcriber provider's registration (parallel to [`CaptionerRegistration`]; sc-12850). The
/// audio-to-text sibling of the captioner: an ASR provider resolved and loaded exactly like every
/// other kind.
#[derive(Clone, Copy)]
pub struct TranscriberRegistration {
    pub descriptor: fn() -> TranscriberDescriptor,
    pub load: fn(&LoadSpec) -> Result<Box<dyn Transcriber>>,
}

/// An image-embedder provider's registration (parallel to [`ModelRegistration`]).
#[derive(Clone, Copy)]
pub struct ImageEmbedderRegistration {
    pub descriptor: fn() -> ImageEmbedderDescriptor,
    pub load: fn(&LoadSpec) -> Result<Box<dyn ImageEmbedder>>,
}

/// A text-embedder provider's registration (parallel to [`ImageEmbedderRegistration`]). Used by the
/// worker's `dataset_analysis` job for caption/image alignment in CLIP's joint space.
#[derive(Clone, Copy)]
pub struct TextEmbedderRegistration {
    pub descriptor: fn() -> TextEmbedderDescriptor,
    pub load: fn(&LoadSpec) -> Result<Box<dyn TextEmbedder>>,
}

/// A voice-embedder provider's registration (parallel to [`ImageEmbedderRegistration`]; sc-12838).
/// The audio-identity sibling of the (unregistered) face embedder: a cloned-voice speaker vector
/// that conditions TTS via [`Conditioning::VoiceEmbedding`](crate::generator::Conditioning::VoiceEmbedding).
#[derive(Clone, Copy)]
pub struct VoiceEmbedderRegistration {
    pub descriptor: fn() -> VoiceEmbedderDescriptor,
    pub load: fn(&LoadSpec) -> Result<Box<dyn VoiceEmbedder>>,
}

/// An audio-embedder provider's registration (parallel to [`ImageEmbedderRegistration`]; sc-12851).
/// The semantic audio-text (CLAP-style) sibling of the image embedder: a whole-clip vector in a
/// joint audio-text space, for retrieval / search / auto-tagging.
#[derive(Clone, Copy)]
pub struct AudioEmbedderRegistration {
    pub descriptor: fn() -> AudioEmbedderDescriptor,
    pub load: fn(&LoadSpec) -> Result<Box<dyn AudioEmbedder>>,
}

/// Builder for an ordinary, explicit generative-media provider registry.
///
/// Platform bundles add exactly the registrations they ship.
#[derive(Default)]
pub struct ProviderRegistryBuilder {
    generators: Vec<ModelRegistration>,
    transforms: Vec<TransformRegistration>,
    audio_transforms: Vec<AudioTransformRegistration>,
    trainers: Vec<TrainerRegistration>,
    captioners: Vec<CaptionerRegistration>,
    transcribers: Vec<TranscriberRegistration>,
    image_embedders: Vec<ImageEmbedderRegistration>,
    text_embedders: Vec<TextEmbedderRegistration>,
    voice_embedders: Vec<VoiceEmbedderRegistration>,
    audio_embedders: Vec<AudioEmbedderRegistration>,
    rejected_quants: Vec<(Quant, &'static str)>,
}

macro_rules! builder_registration_method {
    ($name:ident, $field:ident, $registration:ty) => {
        pub fn $name(mut self, registration: $registration) -> Self {
            self.$field.push(registration);
            self
        }
    };
}

impl ProviderRegistryBuilder {
    /// Start an empty explicit registry.
    pub fn new() -> Self {
        Self::default()
    }

    builder_registration_method!(register_generator, generators, ModelRegistration);
    builder_registration_method!(register_transform, transforms, TransformRegistration);
    builder_registration_method!(
        register_audio_transform,
        audio_transforms,
        AudioTransformRegistration
    );
    builder_registration_method!(register_trainer, trainers, TrainerRegistration);
    builder_registration_method!(register_captioner, captioners, CaptionerRegistration);
    builder_registration_method!(register_transcriber, transcribers, TranscriberRegistration);
    builder_registration_method!(
        register_image_embedder,
        image_embedders,
        ImageEmbedderRegistration
    );
    builder_registration_method!(
        register_text_embedder,
        text_embedders,
        TextEmbedderRegistration
    );
    builder_registration_method!(
        register_voice_embedder,
        voice_embedders,
        VoiceEmbedderRegistration
    );
    builder_registration_method!(
        register_audio_embedder,
        audio_embedders,
        AudioEmbedderRegistration
    );

    /// Declare that this platform's backend has **no implementation** of quant tier `quant`, so every
    /// `load*` through the built registry rejects a [`LoadSpec`] requesting it with `reason`.
    ///
    /// A defense-in-depth guard for the rule that **a quant tier is a creative choice** (epic 11037
    /// SC#5): a tier a backend cannot actually serve must fail loudly at the composition boundary, and
    /// must never be quietly coerced into whatever the backend *can* do. That coercion is a live hazard
    /// wherever the tier's element width collides with a tier the backend does implement — e.g.
    /// [`Quant::Nvfp4`] reports 4 bits, so a backend that keys its
    /// quantizer off [`Quant::bits`](crate::runtime::Quant::bits) alone would silently int4-affine
    /// quantize an NVFP4 request and hand back different numerics under the tier the caller picked.
    ///
    /// This is a *platform capability* statement, not a tensor concern: the mechanism stays
    /// backend-neutral and each catalog names the tiers its own backend leaves unimplemented (the MLX
    /// catalog rejects `Nvfp4`; the CUDA candle catalog, which implements it, does not).
    pub fn reject_quant(mut self, quant: Quant, reason: &'static str) -> Self {
        self.rejected_quants.push((quant, reason));
        self
    }

    /// Validate per-kind id uniqueness and produce the immutable registry.
    pub fn build(self) -> Result<ProviderRegistry> {
        macro_rules! ensure_unique {
            ($field:ident, $kind:literal) => {{
                let mut ids = std::collections::BTreeSet::new();
                for registration in &self.$field {
                    let id = (registration.descriptor)().id;
                    if !ids.insert(id) {
                        return Err(Error::Msg(format!(
                            concat!("duplicate ", $kind, " id '{id}' in explicit registry"),
                            id = id
                        )));
                    }
                }
            }};
        }
        ensure_unique!(generators, "generator");
        ensure_unique!(transforms, "transform");
        ensure_unique!(audio_transforms, "audio transform");
        ensure_unique!(trainers, "trainer");
        ensure_unique!(captioners, "captioner");
        ensure_unique!(transcribers, "transcriber");
        ensure_unique!(image_embedders, "image embedder");
        ensure_unique!(text_embedders, "text embedder");
        ensure_unique!(voice_embedders, "voice embedder");
        ensure_unique!(audio_embedders, "audio embedder");

        Ok(ProviderRegistry {
            generators: self.generators.into_boxed_slice(),
            transforms: self.transforms.into_boxed_slice(),
            audio_transforms: self.audio_transforms.into_boxed_slice(),
            trainers: self.trainers.into_boxed_slice(),
            captioners: self.captioners.into_boxed_slice(),
            transcribers: self.transcribers.into_boxed_slice(),
            image_embedders: self.image_embedders.into_boxed_slice(),
            text_embedders: self.text_embedders.into_boxed_slice(),
            voice_embedders: self.voice_embedders.into_boxed_slice(),
            audio_embedders: self.audio_embedders.into_boxed_slice(),
            rejected_quants: self.rejected_quants.into_boxed_slice(),
        })
    }
}

/// An immutable, explicit catalog of generative-media providers.
pub struct ProviderRegistry {
    generators: Box<[ModelRegistration]>,
    transforms: Box<[TransformRegistration]>,
    audio_transforms: Box<[AudioTransformRegistration]>,
    trainers: Box<[TrainerRegistration]>,
    captioners: Box<[CaptionerRegistration]>,
    transcribers: Box<[TranscriberRegistration]>,
    image_embedders: Box<[ImageEmbedderRegistration]>,
    text_embedders: Box<[TextEmbedderRegistration]>,
    voice_embedders: Box<[VoiceEmbedderRegistration]>,
    audio_embedders: Box<[AudioEmbedderRegistration]>,
    rejected_quants: Box<[(Quant, &'static str)]>,
}

macro_rules! explicit_registry_kind {
    (
        $iter:ident, $load:ident, $field:ident, $registration:ty,
        $kind:literal, $trait:ty
    ) => {
        pub fn $iter(&self) -> impl ExactSizeIterator<Item = &$registration> {
            self.$field.iter()
        }

        pub fn $load(&self, id: &str, spec: &LoadSpec) -> Result<Box<$trait>> {
            let registration = self
                .$iter()
                .find(|registration| (registration.descriptor)().id == id)
                .ok_or_else(|| {
                    Error::Msg(format!(
                        concat!("no ", $kind, " registered for id '{id}'"),
                        id = id
                    ))
                })?;
            self.ensure_quant_supported(id, spec)?;
            (registration.load)(spec)
        }
    };
}

impl ProviderRegistry {
    /// Reject a [`LoadSpec`] whose requested quant tier this platform's backend does not implement,
    /// as declared by [`ProviderRegistryBuilder::reject_quant`].
    ///
    /// The single boundary every registry-routed load of every provider kind passes through, so one
    /// check covers the whole catalog — the composition root states the platform's tier support once
    /// instead of each provider re-deriving it. Runs *after* id resolution so an unknown id still
    /// reports as an unknown id.
    fn ensure_quant_supported(&self, id: &str, spec: &LoadSpec) -> Result<()> {
        let Some(quant) = spec.quantize else {
            return Ok(());
        };
        match self.rejected_quants.iter().find(|(q, _)| *q == quant) {
            Some((_, reason)) => Err(Error::Unsupported(format!(
                "quant tier {quant:?} is not implemented by this runtime's backend \
                 (requested for '{id}'): {reason}. Refusing to load rather than silently \
                 serving a different tier's numerics."
            ))),
            None => Ok(()),
        }
    }

    explicit_registry_kind!(
        generators,
        load,
        generators,
        ModelRegistration,
        "generator",
        dyn Generator
    );
    explicit_registry_kind!(
        transforms,
        load_transform,
        transforms,
        TransformRegistration,
        "transform",
        dyn Transform
    );
    explicit_registry_kind!(
        audio_transforms,
        load_audio_transform,
        audio_transforms,
        AudioTransformRegistration,
        "audio transform",
        dyn AudioTransform
    );
    explicit_registry_kind!(
        trainers,
        load_trainer,
        trainers,
        TrainerRegistration,
        "trainer",
        dyn Trainer
    );
    explicit_registry_kind!(
        captioners,
        load_captioner,
        captioners,
        CaptionerRegistration,
        "captioner",
        dyn Captioner
    );
    explicit_registry_kind!(
        transcribers,
        load_transcriber,
        transcribers,
        TranscriberRegistration,
        "transcriber",
        dyn Transcriber
    );
    explicit_registry_kind!(
        image_embedders,
        load_image_embedder,
        image_embedders,
        ImageEmbedderRegistration,
        "image embedder",
        dyn ImageEmbedder
    );
    explicit_registry_kind!(
        text_embedders,
        load_text_embedder,
        text_embedders,
        TextEmbedderRegistration,
        "text embedder",
        dyn TextEmbedder
    );
    explicit_registry_kind!(
        voice_embedders,
        load_voice_embedder,
        voice_embedders,
        VoiceEmbedderRegistration,
        "voice embedder",
        dyn VoiceEmbedder
    );
    explicit_registry_kind!(
        audio_embedders,
        load_audio_embedder,
        audio_embedders,
        AudioEmbedderRegistration,
        "audio embedder",
        dyn AudioEmbedder
    );

    /// Return the provider-owned on-disk component footprint for generator `id`, when declared.
    ///
    /// The lookup is scoped to this explicit runtime catalog. `Ok(None)` means the provider does not
    /// declare a split; unknown ids and provider accounting failures remain errors so consumers can
    /// deliberately choose whether to fail open.
    pub fn footprint(&self, id: &str, spec: &LoadSpec) -> Result<Option<PerComponentBytes>> {
        let registration = self
            .generators()
            .find(|registration| (registration.descriptor)().id == id)
            .ok_or_else(|| Error::Msg(format!("no generator registered for id '{id}'")))?;
        match registration.footprint {
            Some(footprint) => footprint(spec).map(Some),
            None => Ok(None),
        }
    }

    /// Run the weights-free descriptor conformance sweep over this explicit catalog.
    pub fn descriptor_conformance_errors(&self) -> Vec<String> {
        descriptor_conformance_errors_for(
            &self.generators,
            &self.transforms,
            &self.audio_transforms,
            &self.trainers,
            &self.captioners,
            &self.transcribers,
            &self.image_embedders,
            &self.text_embedders,
            &self.voice_embedders,
            &self.audio_embedders,
        )
    }
}

// ---------------------------------------------------------------------------------------------
// Descriptor-level conformance sweep (sc-9098, F-009)
// ---------------------------------------------------------------------------------------------

/// An identifier-shaped registry string: non-empty lowercase `a-z0-9` with `_`/`-`/`.`/`/`
/// separators — the shape every shipped id/family/backend uses (`z_image_turbo`, `image-embed`,
/// `mlx`, and HF-repo-style captioner ids like `fancyfeast/llama-joycaption-beta-one-hf-llava`).
/// Rejects whitespace/uppercase/unicode, which would break worker payload routing and log grepping.
fn is_registry_ident(s: &str) -> bool {
    !s.is_empty()
        && s.chars().all(|c| {
            c.is_ascii_lowercase() || c.is_ascii_digit() || matches!(c, '_' | '-' | '.' | '/')
        })
}

/// Push an error for every malformed identity field (shared by all descriptor kinds).
fn check_identity(errs: &mut Vec<String>, ctx: &str, fields: &[(&str, &str)]) {
    for (name, value) in fields {
        if !is_registry_ident(value) {
            errs.push(format!(
                "{ctx}: {name} {value:?} is not a valid registry identifier \
                 (non-empty lowercase [a-z0-9_.-/])"
            ));
        }
    }
}

/// Push an error for empty/whitespace/duplicate entries in a descriptor's curated name list
/// (samplers / schedulers / guidance methods).
fn check_name_list(errs: &mut Vec<String>, ctx: &str, list_name: &str, names: &[&str]) {
    for (i, n) in names.iter().enumerate() {
        if n.is_empty() || n.chars().any(char::is_whitespace) {
            errs.push(format!(
                "{ctx}: {list_name}[{i}] {n:?} is empty or contains whitespace"
            ));
        }
        if names[..i].contains(n) {
            errs.push(format!("{ctx}: duplicate {list_name} entry {n:?}"));
        }
    }
}

/// The weights-free invariants a generator [`ModelDescriptor`] must satisfy — everything checkable
/// from `(registration.descriptor)()` alone, with no model load (sc-9098, F-009):
///
/// - `id` / `family` / `backend` are non-empty registry identifiers,
/// - `max_count ≥ 1`, and `1 ≤ min_size ≤ max_size` for the visual modalities (a `Default` 0 bound
///   rejects every request with a confusing "out of range 0..=0" — the F-084 footgun, enforced here
///   for *every* linked visual descriptor rather than only when a request happens to reach
///   `validate_request`); the size range is **skipped for `Modality::Audio`**, whose generators emit
///   a track with no width/height and leave the bounds at the unused 0 (matching the size-skipping
///   `validate_request_audio` floor, sc-12834/sc-13314),
/// - `samplers` / `schedulers` / `supported_guidance_methods` entries are non-empty, whitespace-free
///   and duplicate-free (name *shape* only — resolvability is per-engine: several families advertise
///   native sampler names alongside the gen-core curated set),
/// - `conditioning` is duplicate-free, and the video-clip kinds
///   ([`Keyframe`](ConditioningKind::Keyframe) / [`VideoClip`](ConditioningKind::VideoClip) /
///   [`ControlClip`](ConditioningKind::ControlClip)) are only advertised by `Video`/`Both`-modality
///   models — an `Image` model cannot consume a clip.
///
/// Returns one message per violation (empty = conformant). Public so a provider's own tests can
/// target a single descriptor; [`ProviderRegistry::descriptor_conformance_errors`] sweeps a catalog.
pub fn model_descriptor_errors(d: &ModelDescriptor) -> Vec<String> {
    let mut errs = Vec::new();
    let ctx = format!("generator '{}'", d.id);
    check_identity(
        &mut errs,
        &ctx,
        &[("id", d.id), ("family", d.family), ("backend", d.backend)],
    );
    let caps = &d.capabilities;
    if caps.max_count == 0 {
        errs.push(format!(
            "{ctx}: max_count is 0 — every request would be rejected"
        ));
    }
    // Size bounds only mean something for the visual modalities. A `Modality::Audio` generator emits
    // a `GenerationOutput::Audio` track with no width/height, so its `min_size`/`max_size` are unused
    // and left at the natural `Default` 0 — exactly the fields `validate_request_audio` skips the
    // range check for (sc-12834). Enforcing the visual `1 <= min_size <= max_size` floor on an audio
    // descriptor would force a nominal placeholder bound purely to satisfy the sweep (the sc-13314
    // wart), so the check is skipped for `Audio` and stays strict for `Image`/`Video`/`Both`, which
    // genuinely carry a spatial size range.
    if d.modality != Modality::Audio {
        if caps.min_size == 0 || caps.max_size == 0 {
            errs.push(format!(
                "{ctx}: min_size={} max_size={} — size bounds left at the Default 0",
                caps.min_size, caps.max_size
            ));
        } else if caps.min_size > caps.max_size {
            errs.push(format!(
                "{ctx}: min_size {} > max_size {}",
                caps.min_size, caps.max_size
            ));
        }
    }
    check_name_list(&mut errs, &ctx, "sampler", &caps.samplers);
    check_name_list(&mut errs, &ctx, "scheduler", &caps.schedulers);
    check_name_list(
        &mut errs,
        &ctx,
        "guidance_method",
        &caps.supported_guidance_methods,
    );
    for (i, k) in caps.conditioning.iter().enumerate() {
        if caps.conditioning[..i].contains(k) {
            errs.push(format!("{ctx}: duplicate conditioning kind {k:?}"));
        }
        let is_video_kind = matches!(
            k,
            ConditioningKind::Keyframe
                | ConditioningKind::VideoClip
                | ConditioningKind::ControlClip
        );
        if is_video_kind && d.modality == Modality::Image {
            errs.push(format!(
                "{ctx}: advertises video conditioning {k:?} but modality is Image"
            ));
        }
    }
    errs
}

/// Push duplicate-id errors for one registry kind.
fn check_unique_ids(errs: &mut Vec<String>, kind: &str, ids: &[&str]) {
    for (i, id) in ids.iter().enumerate() {
        if ids[..i].contains(id) {
            errs.push(format!(
                "{kind} id '{id}' is registered more than once (first-wins shadows the rest)"
            ));
        }
    }
}

/// Weights-free descriptor-level conformance sweep over one explicit provider catalog (sc-9098,
/// F-009): generators through [`model_descriptor_errors`], plus identity
/// and capability-bound checks and per-kind id uniqueness for trainers, captioners, transforms and
/// image/text/voice embedders. No `load` is ever called, so it runs by default (no weights, no Metal) —
/// each provider crate invokes it from a default test, giving every cataloged id at least
/// descriptor-level coverage; behavioral conformance (progress/cancel/seed) stays weights-gated in
/// the `gen-core-testkit` suite.
///
/// Returns one message per violation (empty = conformant).
// One `&[Registration]` slice per provider kind, so the arg count tracks the number of kinds (10 as
// of the sc-12851 audio-embedder kind) rather than any avoidable coupling — the alternative is a
// throwaway "all registrations" struct that adds no clarity.
#[allow(clippy::too_many_arguments)]
fn descriptor_conformance_errors_for(
    generator_registrations: &[ModelRegistration],
    transform_registrations: &[TransformRegistration],
    audio_transform_registrations: &[AudioTransformRegistration],
    trainer_registrations: &[TrainerRegistration],
    captioner_registrations: &[CaptionerRegistration],
    transcriber_registrations: &[TranscriberRegistration],
    image_embedder_registrations: &[ImageEmbedderRegistration],
    text_embedder_registrations: &[TextEmbedderRegistration],
    voice_embedder_registrations: &[VoiceEmbedderRegistration],
    audio_embedder_registrations: &[AudioEmbedderRegistration],
) -> Vec<String> {
    let mut errs = Vec::new();

    let gen_descs: Vec<ModelDescriptor> = generator_registrations
        .iter()
        .map(|r| (r.descriptor)())
        .collect();
    for d in &gen_descs {
        errs.extend(model_descriptor_errors(d));
    }
    let gen_ids: Vec<&str> = gen_descs.iter().map(|d| d.id).collect();
    check_unique_ids(&mut errs, "generator", &gen_ids);

    let trainer_descs: Vec<TrainerDescriptor> = trainer_registrations
        .iter()
        .map(|r| (r.descriptor)())
        .collect();
    for d in &trainer_descs {
        let ctx = format!("trainer '{}'", d.id);
        check_identity(
            &mut errs,
            &ctx,
            &[("id", d.id), ("family", d.family), ("backend", d.backend)],
        );
    }
    let trainer_ids: Vec<&str> = trainer_descs.iter().map(|d| d.id).collect();
    check_unique_ids(&mut errs, "trainer", &trainer_ids);

    let cap_descs: Vec<CaptionerDescriptor> = captioner_registrations
        .iter()
        .map(|r| (r.descriptor)())
        .collect();
    for d in &cap_descs {
        let ctx = format!("captioner '{}'", d.id);
        check_identity(
            &mut errs,
            &ctx,
            &[("id", d.id), ("family", d.family), ("backend", d.backend)],
        );
        let c = &d.capabilities;
        if c.min_image_size == 0 || c.max_image_size < c.min_image_size {
            errs.push(format!(
                "{ctx}: image-size bounds incoherent (min {} max {})",
                c.min_image_size, c.max_image_size
            ));
        }
        if c.max_new_tokens == 0 {
            errs.push(format!(
                "{ctx}: max_new_tokens is 0 — no caption could be produced"
            ));
        }
    }
    let cap_ids: Vec<&str> = cap_descs.iter().map(|d| d.id).collect();
    check_unique_ids(&mut errs, "captioner", &cap_ids);

    // Transcribers (sc-12850): identity, capability-bound coherence (a non-zero token ceiling and a
    // positive max clip duration — the audio twin of the captioner's max_new_tokens/image-size
    // checks), and id uniqueness.
    let asr_descs: Vec<TranscriberDescriptor> = transcriber_registrations
        .iter()
        .map(|r| (r.descriptor)())
        .collect();
    for d in &asr_descs {
        let ctx = format!("transcriber '{}'", d.id);
        check_identity(
            &mut errs,
            &ctx,
            &[("id", d.id), ("family", d.family), ("backend", d.backend)],
        );
        let c = &d.capabilities;
        if c.max_new_tokens == 0 {
            errs.push(format!(
                "{ctx}: max_new_tokens is 0 — no transcript could be produced"
            ));
        }
        if !c.max_audio_seconds.is_finite() || c.max_audio_seconds <= 0.0 {
            errs.push(format!(
                "{ctx}: max_audio_seconds is {} — no audio could be accepted",
                c.max_audio_seconds
            ));
        }
    }
    let asr_ids: Vec<&str> = asr_descs.iter().map(|d| d.id).collect();
    check_unique_ids(&mut errs, "transcriber", &asr_ids);

    let tf_descs: Vec<TransformDescriptor> = transform_registrations
        .iter()
        .map(|r| (r.descriptor)())
        .collect();
    for d in &tf_descs {
        let ctx = format!("transform '{}'", d.id);
        check_identity(
            &mut errs,
            &ctx,
            &[("id", d.id), ("family", d.family), ("backend", d.backend)],
        );
    }
    let tf_ids: Vec<&str> = tf_descs.iter().map(|d| d.id).collect();
    check_unique_ids(&mut errs, "transform", &tf_ids);

    // Audio transforms (sc-12839): identity, a kind/stem-count coherence check (a separator must
    // advertise ≥ 2 stems; the single-output kinds advertise 0), and id uniqueness.
    let atf_descs: Vec<AudioTransformDescriptor> = audio_transform_registrations
        .iter()
        .map(|r| (r.descriptor)())
        .collect();
    for d in &atf_descs {
        let ctx = format!("audio transform '{}'", d.id);
        check_identity(
            &mut errs,
            &ctx,
            &[("id", d.id), ("family", d.family), ("backend", d.backend)],
        );
        let caps = &d.capabilities;
        match caps.kind {
            AudioTransformKind::StemSeparation if caps.stem_count < 2 => errs.push(format!(
                "{ctx}: StemSeparation advertises stem_count {} (a separator must produce ≥ 2 stems)",
                caps.stem_count
            )),
            AudioTransformKind::VoiceConversion | AudioTransformKind::SuperResolution
                if caps.stem_count != 0 =>
            {
                errs.push(format!(
                    "{ctx}: {:?} advertises stem_count {} — only StemSeparation produces stems",
                    caps.kind, caps.stem_count
                ))
            }
            _ => {}
        }
    }
    let atf_ids: Vec<&str> = atf_descs.iter().map(|d| d.id).collect();
    check_unique_ids(&mut errs, "audio transform", &atf_ids);

    let ie_descs: Vec<ImageEmbedderDescriptor> = image_embedder_registrations
        .iter()
        .map(|r| (r.descriptor)())
        .collect();
    let te_descs: Vec<TextEmbedderDescriptor> = text_embedder_registrations
        .iter()
        .map(|r| (r.descriptor)())
        .collect();
    // Audio embedders (sc-12851) carry a joint audio-text `space` exactly like image/text
    // embedders, so they run through the same identity + non-zero-dim + non-empty-space check.
    let ae_descs: Vec<AudioEmbedderDescriptor> = audio_embedder_registrations
        .iter()
        .map(|r| (r.descriptor)())
        .collect();
    for (ctx_kind, id, family, backend, dim, space) in ie_descs
        .iter()
        .map(|d| {
            (
                "image embedder",
                d.id,
                d.family,
                d.backend,
                d.embedding_dim,
                d.space,
            )
        })
        .chain(te_descs.iter().map(|d| {
            (
                "text embedder",
                d.id,
                d.family,
                d.backend,
                d.embedding_dim,
                d.space,
            )
        }))
        .chain(ae_descs.iter().map(|d| {
            (
                "audio embedder",
                d.id,
                d.family,
                d.backend,
                d.embedding_dim,
                d.space,
            )
        }))
    {
        let ctx = format!("{ctx_kind} '{id}'");
        check_identity(
            &mut errs,
            &ctx,
            &[("id", id), ("family", family), ("backend", backend)],
        );
        if dim == 0 {
            errs.push(format!("{ctx}: embedding_dim is 0"));
        }
        if space.is_empty() {
            errs.push(format!("{ctx}: embedding space is empty"));
        }
    }
    let ie_ids: Vec<&str> = ie_descs.iter().map(|d| d.id).collect();
    check_unique_ids(&mut errs, "image embedder", &ie_ids);
    let te_ids: Vec<&str> = te_descs.iter().map(|d| d.id).collect();
    check_unique_ids(&mut errs, "text embedder", &te_ids);
    let ae_ids: Vec<&str> = ae_descs.iter().map(|d| d.id).collect();
    check_unique_ids(&mut errs, "audio embedder", &ae_ids);

    // Voice embedders (sc-12838) carry no embedding `space` (they are the audio-identity sibling of
    // the face embedder, not a cross-encoder cosine space) — check identity, a non-zero dim, and
    // id uniqueness only.
    let ve_descs: Vec<VoiceEmbedderDescriptor> = voice_embedder_registrations
        .iter()
        .map(|r| (r.descriptor)())
        .collect();
    for d in &ve_descs {
        let ctx = format!("voice embedder '{}'", d.id);
        check_identity(
            &mut errs,
            &ctx,
            &[("id", d.id), ("family", d.family), ("backend", d.backend)],
        );
        if d.embedding_dim == 0 {
            errs.push(format!("{ctx}: embedding_dim is 0"));
        }
    }
    let ve_ids: Vec<&str> = ve_descs.iter().map(|d| d.id).collect();
    check_unique_ids(&mut errs, "voice embedder", &ve_ids);

    errs
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::audio_embed::{AudioEmbedder, AudioEmbedderDescriptor};
    use crate::audio_transform::{
        AudioTarget, AudioTransform, AudioTransformCapabilities, AudioTransformDescriptor,
        AudioTransformKind, AudioTransformRequest,
    };
    use crate::caption::{
        CaptionCapabilities, CaptionOutput, CaptionRequest, Captioner, CaptionerDescriptor,
    };
    use crate::generator::{
        Capabilities, GenerationOutput, GenerationRequest, Modality, ModelDescriptor,
    };
    use crate::image_embed::{ImageEmbedder, ImageEmbedderDescriptor};
    use crate::media::{AudioTrack, Image};
    use crate::runtime::{Progress, WeightsSource};
    use crate::text_embed::{TextEmbedder, TextEmbedderDescriptor};
    use crate::train::{
        Trainer, TrainerDescriptor, TrainingOutput, TrainingProgress, TrainingRequest,
    };
    use crate::transcribe::{
        TranscribeCapabilities, TranscribeRequest, Transcriber, TranscriberDescriptor,
        TranscriptOutput,
    };
    use crate::voice_embed::{VoiceEmbedder, VoiceEmbedderDescriptor, VoiceEmbedding};
    use std::path::PathBuf;

    struct DummyGen {
        desc: ModelDescriptor,
    }

    impl Generator for DummyGen {
        fn descriptor(&self) -> &ModelDescriptor {
            &self.desc
        }
        fn validate(&self, _req: &GenerationRequest) -> Result<()> {
            Ok(())
        }
        fn generate(
            &self,
            _req: &GenerationRequest,
            _on_progress: &mut dyn FnMut(Progress),
        ) -> Result<GenerationOutput> {
            Ok(GenerationOutput::Images(vec![Image::default()]))
        }
    }

    /// Small-but-coherent capabilities for the dummy registrations: the descriptor sweep runs over
    /// the explicit fixture catalog, so the dummies must carry real bounds (a
    /// `Capabilities::default()` has the F-084 all-zero bounds the sweep exists to reject).
    fn dummy_caps() -> Capabilities {
        Capabilities {
            min_size: 64,
            max_size: 512,
            max_count: 1,
            ..Default::default()
        }
    }

    fn dummy_descriptor() -> ModelDescriptor {
        ModelDescriptor {
            id: "dummy_test_model",
            family: "test",
            backend: "mlx",
            modality: Modality::Image,
            capabilities: dummy_caps(),
        }
    }

    fn dummy_load(_spec: &LoadSpec) -> Result<Box<dyn Generator>> {
        Ok(Box::new(DummyGen {
            desc: dummy_descriptor(),
        }))
    }

    crate::register_generators! {
        const DUMMY_GENERATOR_REGISTRATION = dummy_descriptor => dummy_load
    }

    struct DummyDelegatedGen {
        descriptor: ModelDescriptor,
    }

    impl DummyDelegatedGen {
        fn generate_impl(
            &self,
            _req: &GenerationRequest,
            _on_progress: &mut dyn FnMut(Progress),
        ) -> Result<GenerationOutput> {
            Ok(GenerationOutput::Images(vec![Image::default()]))
        }
    }

    crate::impl_generator!(DummyDelegatedGen {
        validate: |_s, _req| Ok::<(), Error>(()),
        generate: generate_impl,
    });

    fn dummy_delegated_descriptor() -> ModelDescriptor {
        ModelDescriptor {
            id: "dummy_delegated_test_model",
            family: "test",
            backend: "mlx",
            modality: Modality::Image,
            capabilities: dummy_caps(),
        }
    }

    fn dummy_delegated_load(_spec: &LoadSpec) -> Result<Box<dyn Generator>> {
        Ok(Box::new(DummyDelegatedGen {
            descriptor: dummy_delegated_descriptor(),
        }))
    }

    crate::register_generators! {
        const DUMMY_DELEGATED_GENERATOR_REGISTRATION =
            dummy_delegated_descriptor => dummy_delegated_load
    }

    // A dummy generator that DECLARES a per-component footprint (sc-10894), exercising the
    // `; footprint = …` macro arm and the [`footprint`] entry point. Its text encoder is under a
    // non-standard `mllm/` subdir (the real boogu layout) — a naming a `text_encoder*` guesser would
    // read as ZERO — so the provider-owned split is what finds it.
    fn dummy_footprint_descriptor() -> ModelDescriptor {
        ModelDescriptor {
            id: "dummy_footprint_model",
            family: "test",
            backend: "mlx",
            modality: Modality::Image,
            capabilities: dummy_caps(),
        }
    }

    fn dummy_footprint_load(_spec: &LoadSpec) -> Result<Box<dyn Generator>> {
        Ok(Box::new(DummyGen {
            desc: dummy_footprint_descriptor(),
        }))
    }

    fn dummy_footprint(spec: &LoadSpec) -> Result<PerComponentBytes> {
        PerComponentBytes::from_spec_subdirs(spec, &["mllm"], &["transformer"], &["vae"])
    }

    crate::register_generators! {
        const DUMMY_FOOTPRINT_GENERATOR_REGISTRATION =
            dummy_footprint_descriptor => dummy_footprint_load;
        footprint = dummy_footprint
    }

    struct DummyTrainer {
        desc: TrainerDescriptor,
    }

    impl Trainer for DummyTrainer {
        fn descriptor(&self) -> &TrainerDescriptor {
            &self.desc
        }

        fn validate(&self, _req: &TrainingRequest) -> Result<()> {
            Ok(())
        }

        fn train(
            &mut self,
            _req: &TrainingRequest,
            _on_progress: &mut dyn FnMut(TrainingProgress),
        ) -> Result<TrainingOutput> {
            Ok(TrainingOutput {
                adapter_path: PathBuf::from("/tmp/dummy.safetensors"),
                steps: 0,
                final_loss: 0.0,
            })
        }
    }

    fn dummy_trainer_descriptor() -> TrainerDescriptor {
        TrainerDescriptor {
            id: "dummy_test_trainer",
            family: "test",
            backend: "mlx",
            modality: Modality::Image,
            supports_lora: true,
            supports_lokr: false,
            supports_control: false,
        }
    }

    fn dummy_trainer_load(_spec: &LoadSpec) -> Result<Box<dyn Trainer>> {
        Ok(Box::new(DummyTrainer {
            desc: dummy_trainer_descriptor(),
        }))
    }

    crate::register_trainer! {
        const DUMMY_TRAINER_REGISTRATION = dummy_trainer_descriptor => dummy_trainer_load
    }

    // Multi-provider fixtures verify that independently named constants compose into one catalog.
    fn dummy_multi_gen_a_descriptor() -> ModelDescriptor {
        ModelDescriptor {
            id: "dummy_multi_gen_a",
            family: "test",
            backend: "mlx",
            modality: Modality::Image,
            capabilities: dummy_caps(),
        }
    }

    fn dummy_multi_gen_b_descriptor() -> ModelDescriptor {
        ModelDescriptor {
            id: "dummy_multi_gen_b",
            family: "test",
            backend: "mlx",
            modality: Modality::Image,
            capabilities: dummy_caps(),
        }
    }

    fn dummy_multi_gen_a_load(_spec: &LoadSpec) -> Result<Box<dyn Generator>> {
        Ok(Box::new(DummyGen {
            desc: dummy_multi_gen_a_descriptor(),
        }))
    }

    fn dummy_multi_gen_b_load(_spec: &LoadSpec) -> Result<Box<dyn Generator>> {
        Ok(Box::new(DummyGen {
            desc: dummy_multi_gen_b_descriptor(),
        }))
    }

    crate::register_generators! {
        const DUMMY_MULTI_GENERATOR_A_REGISTRATION =
            dummy_multi_gen_a_descriptor => dummy_multi_gen_a_load
    }
    crate::register_generators! {
        const DUMMY_MULTI_GENERATOR_B_REGISTRATION =
            dummy_multi_gen_b_descriptor => dummy_multi_gen_b_load
    }

    fn dummy_multi_trainer_a_descriptor() -> TrainerDescriptor {
        TrainerDescriptor {
            id: "dummy_multi_trainer_a",
            family: "test",
            backend: "mlx",
            modality: Modality::Image,
            supports_lora: true,
            supports_lokr: false,
            supports_control: false,
        }
    }

    fn dummy_multi_trainer_b_descriptor() -> TrainerDescriptor {
        TrainerDescriptor {
            id: "dummy_multi_trainer_b",
            family: "test",
            backend: "mlx",
            modality: Modality::Image,
            supports_lora: true,
            supports_lokr: false,
            supports_control: false,
        }
    }

    fn dummy_multi_trainer_a_load(_spec: &LoadSpec) -> Result<Box<dyn Trainer>> {
        Ok(Box::new(DummyTrainer {
            desc: dummy_multi_trainer_a_descriptor(),
        }))
    }

    fn dummy_multi_trainer_b_load(_spec: &LoadSpec) -> Result<Box<dyn Trainer>> {
        Ok(Box::new(DummyTrainer {
            desc: dummy_multi_trainer_b_descriptor(),
        }))
    }

    crate::register_trainer! {
        const DUMMY_MULTI_TRAINER_A_REGISTRATION =
            dummy_multi_trainer_a_descriptor => dummy_multi_trainer_a_load
    }
    crate::register_trainer! {
        const DUMMY_MULTI_TRAINER_B_REGISTRATION =
            dummy_multi_trainer_b_descriptor => dummy_multi_trainer_b_load
    }

    struct DummyCaptioner {
        desc: CaptionerDescriptor,
    }

    impl Captioner for DummyCaptioner {
        fn descriptor(&self) -> &CaptionerDescriptor {
            &self.desc
        }
        fn validate(&self, _req: &CaptionRequest) -> Result<()> {
            Ok(())
        }
        fn caption(
            &self,
            _req: &CaptionRequest,
            _on_progress: &mut dyn FnMut(Progress),
        ) -> Result<CaptionOutput> {
            Ok(CaptionOutput {
                text: "caption".to_owned(),
                generated_tokens: Some(1),
                finish_reason: None,
            })
        }
    }

    fn dummy_captioner_descriptor() -> CaptionerDescriptor {
        CaptionerDescriptor {
            id: "dummy_test_captioner",
            family: "test",
            backend: "mlx",
            capabilities: CaptionCapabilities {
                min_image_size: 1,
                max_image_size: 4096,
                max_prompt_chars: 4000,
                max_name_chars: 120,
                max_extra_options: 16,
                max_extra_option_chars: 500,
                max_trigger_words: 32,
                max_trigger_word_chars: 120,
                max_new_tokens: 1024,
                ..Default::default()
            },
        }
    }

    fn dummy_captioner_load(_spec: &LoadSpec) -> Result<Box<dyn Captioner>> {
        Ok(Box::new(DummyCaptioner {
            desc: dummy_captioner_descriptor(),
        }))
    }

    crate::register_captioner! {
        const DUMMY_CAPTIONER_REGISTRATION = dummy_captioner_descriptor => dummy_captioner_load
    }

    struct DummyTranscriber {
        desc: TranscriberDescriptor,
    }

    impl Transcriber for DummyTranscriber {
        fn descriptor(&self) -> &TranscriberDescriptor {
            &self.desc
        }
        fn validate(&self, _req: &TranscribeRequest) -> Result<()> {
            Ok(())
        }
        fn transcribe(
            &self,
            _req: &TranscribeRequest,
            _on_progress: &mut dyn FnMut(Progress),
        ) -> Result<TranscriptOutput> {
            Ok(TranscriptOutput {
                text: "transcript".to_owned(),
                generated_tokens: Some(1),
                ..Default::default()
            })
        }
    }

    fn dummy_transcriber_descriptor() -> TranscriberDescriptor {
        TranscriberDescriptor {
            id: "dummy_test_transcriber",
            family: "test",
            backend: "candle",
            capabilities: TranscribeCapabilities {
                languages: vec!["en"],
                supports_segment_timestamps: true,
                max_audio_seconds: 30.0,
                max_new_tokens: 448,
                ..Default::default()
            },
        }
    }

    fn dummy_transcriber_load(_spec: &LoadSpec) -> Result<Box<dyn Transcriber>> {
        Ok(Box::new(DummyTranscriber {
            desc: dummy_transcriber_descriptor(),
        }))
    }

    crate::register_transcriber! {
        const DUMMY_TRANSCRIBER_REGISTRATION =
            dummy_transcriber_descriptor => dummy_transcriber_load
    }

    fn dummy_registry() -> ProviderRegistry {
        ProviderRegistryBuilder::new()
            .register_generator(DUMMY_GENERATOR_REGISTRATION)
            .register_generator(DUMMY_DELEGATED_GENERATOR_REGISTRATION)
            .register_generator(DUMMY_FOOTPRINT_GENERATOR_REGISTRATION)
            .register_generator(DUMMY_MULTI_GENERATOR_A_REGISTRATION)
            .register_generator(DUMMY_MULTI_GENERATOR_B_REGISTRATION)
            .register_trainer(DUMMY_TRAINER_REGISTRATION)
            .register_trainer(DUMMY_MULTI_TRAINER_A_REGISTRATION)
            .register_trainer(DUMMY_MULTI_TRAINER_B_REGISTRATION)
            .register_captioner(DUMMY_CAPTIONER_REGISTRATION)
            .register_transcriber(DUMMY_TRANSCRIBER_REGISTRATION)
            .register_text_embedder(DUMMY_TEXT_EMBEDDER_REGISTRATION)
            .register_image_embedder(DUMMY_IMAGE_EMBEDDER_REGISTRATION)
            .register_voice_embedder(DUMMY_VOICE_EMBEDDER_REGISTRATION)
            .register_audio_embedder(DUMMY_AUDIO_EMBEDDER_REGISTRATION)
            .build()
            .unwrap()
    }

    #[test]
    fn registry_resolves_by_id() {
        let registry = dummy_registry();
        let spec = LoadSpec::new(WeightsSource::Dir("/nonexistent".into()));
        let g = registry
            .load("dummy_test_model", &spec)
            .expect("dummy is registered");
        assert_eq!(g.descriptor().id, "dummy_test_model");
        assert_eq!(g.descriptor().modality, Modality::Image);
    }

    #[test]
    fn explicit_registry_resolves_minimal_catalog() {
        let registry = ProviderRegistryBuilder::new()
            .register_generator(ModelRegistration {
                descriptor: dummy_descriptor,
                load: dummy_load,
                footprint: None,
            })
            .build()
            .unwrap();
        assert_eq!(registry.generators().len(), 1);
        let spec = LoadSpec::new(WeightsSource::Dir(PathBuf::from("/tmp")));
        assert_eq!(
            registry
                .load("dummy_test_model", &spec)
                .unwrap()
                .descriptor()
                .id,
            "dummy_test_model"
        );
        assert!(registry.trainers().next().is_none());
    }

    /// A tier the platform declared unimplemented is rejected at the load boundary — loudly, naming
    /// the tier, the id, and the platform's reason (epic 11037 SC#5: a quant tier is a creative
    /// choice, never silently substituted). `dummy_load` would otherwise *succeed*, so this pins that
    /// the guard fires ahead of the provider rather than leaving the coercion to the backend.
    #[test]
    fn rejected_quant_tier_fails_loudly_at_load() {
        let registry = ProviderRegistryBuilder::new()
            .register_generator(ModelRegistration {
                descriptor: dummy_descriptor,
                load: dummy_load,
                footprint: None,
            })
            .reject_quant(Quant::Nvfp4, "no FP4 quantizer on this backend")
            .build()
            .unwrap();

        let mut spec = LoadSpec::new(WeightsSource::Dir("/nonexistent".into()));
        spec.quantize = Some(Quant::Nvfp4);
        let error = registry
            .load("dummy_test_model", &spec)
            .err()
            .expect("a rejected tier must not reach the provider");
        match error {
            Error::Unsupported(message) => assert_eq!(
                message,
                "quant tier Nvfp4 is not implemented by this runtime's backend \
                 (requested for 'dummy_test_model'): no FP4 quantizer on this backend. Refusing to \
                 load rather than silently serving a different tier's numerics."
            ),
            other => panic!("a rejected quant tier is a capability gap, got {other:?}"),
        }
    }

    /// The guard is scoped to the declared tiers: an unrejected tier (and a dense, `None` load) still
    /// reaches the provider untouched, and a catalog that declares nothing rejects nothing.
    #[test]
    fn unrejected_quant_tiers_still_load() {
        let registry = ProviderRegistryBuilder::new()
            .register_generator(ModelRegistration {
                descriptor: dummy_descriptor,
                load: dummy_load,
                footprint: None,
            })
            .reject_quant(Quant::Nvfp4, "no FP4 quantizer on this backend")
            .build()
            .unwrap();

        for quant in [None, Some(Quant::Q4), Some(Quant::Q8)] {
            let mut spec = LoadSpec::new(WeightsSource::Dir("/nonexistent".into()));
            spec.quantize = quant;
            assert!(
                registry.load("dummy_test_model", &spec).is_ok(),
                "{quant:?} must still load"
            );
        }

        // A catalog whose backend implements every tier (the CUDA candle catalog) declares no
        // rejection and is unaffected.
        let permissive = ProviderRegistryBuilder::new()
            .register_generator(ModelRegistration {
                descriptor: dummy_descriptor,
                load: dummy_load,
                footprint: None,
            })
            .build()
            .unwrap();
        let mut spec = LoadSpec::new(WeightsSource::Dir("/nonexistent".into()));
        spec.quantize = Some(Quant::Nvfp4);
        assert!(permissive.load("dummy_test_model", &spec).is_ok());
    }

    /// An unknown id reports as an unknown id even when the spec also carries a rejected tier — the
    /// guard runs after id resolution so the caller sees the primary fault.
    #[test]
    fn unknown_id_wins_over_rejected_quant() {
        let registry = ProviderRegistryBuilder::new()
            .reject_quant(Quant::Nvfp4, "no FP4 quantizer on this backend")
            .build()
            .unwrap();
        let mut spec = LoadSpec::new(WeightsSource::Dir("/nonexistent".into()));
        spec.quantize = Some(Quant::Nvfp4);
        let error = registry
            .load("nope", &spec)
            .err()
            .expect("unknown id must fail")
            .to_string();
        assert!(
            error.contains("no generator registered for id 'nope'"),
            "{error}"
        );
    }

    #[test]
    fn explicit_registry_rejects_duplicate_ids_deterministically() {
        let registration = ModelRegistration {
            descriptor: dummy_descriptor,
            load: dummy_load,
            footprint: None,
        };
        let error = ProviderRegistryBuilder::new()
            .register_generator(registration)
            .register_generator(registration)
            .build()
            .err()
            .expect("duplicate registry must fail");
        assert_eq!(
            error.to_string(),
            "duplicate generator id 'dummy_test_model' in explicit registry"
        );
    }

    #[test]
    fn unknown_id_errors() {
        let registry = dummy_registry();
        let spec = LoadSpec::new(WeightsSource::Dir("/nonexistent".into()));
        assert!(registry.load("no_such_model", &spec).is_err());
    }

    #[test]
    fn dummy_appears_in_iteration() {
        assert!(dummy_registry()
            .generators()
            .any(|r| (r.descriptor)().id == "dummy_test_model"));
    }

    #[test]
    fn macro_delegated_generator_resolves_by_id() {
        let registry = dummy_registry();
        let spec = LoadSpec::new(WeightsSource::Dir("/nonexistent".into()));
        let g = registry
            .load("dummy_delegated_test_model", &spec)
            .expect("dummy is registered");
        assert_eq!(g.descriptor().id, "dummy_delegated_test_model");
        g.validate(&GenerationRequest::default()).unwrap();
    }

    #[test]
    fn macro_registered_trainer_resolves_by_id() {
        let registry = dummy_registry();
        let spec = LoadSpec::new(WeightsSource::Dir("/nonexistent".into()));
        let t = registry
            .load_trainer("dummy_test_trainer", &spec)
            .expect("dummy trainer is registered");
        assert_eq!(t.descriptor().id, "dummy_test_trainer");
        assert!(registry
            .trainers()
            .any(|r| (r.descriptor)().id == "dummy_test_trainer"));
    }

    #[test]
    fn multiple_generator_constants_compose() {
        let registry = dummy_registry();
        let spec = LoadSpec::new(WeightsSource::Dir("/nonexistent".into()));
        for id in ["dummy_multi_gen_a", "dummy_multi_gen_b"] {
            let g = registry
                .load(id, &spec)
                .unwrap_or_else(|_| panic!("{id} is registered"));
            assert_eq!(g.descriptor().id, id);
        }
    }

    #[test]
    fn multiple_trainer_constants_compose() {
        let registry = dummy_registry();
        let spec = LoadSpec::new(WeightsSource::Dir("/nonexistent".into()));
        for id in ["dummy_multi_trainer_a", "dummy_multi_trainer_b"] {
            let t = registry
                .load_trainer(id, &spec)
                .unwrap_or_else(|_| panic!("{id} is registered"));
            assert_eq!(t.descriptor().id, id);
        }
    }

    #[test]
    fn captioner_registry_resolves_by_id() {
        let registry = dummy_registry();
        let spec = LoadSpec::new(WeightsSource::Dir("/nonexistent".into()));
        let c = registry
            .load_captioner("dummy_test_captioner", &spec)
            .expect("dummy captioner is registered");
        assert_eq!(c.descriptor().id, "dummy_test_captioner");
    }

    #[test]
    fn unknown_captioner_id_errors() {
        let registry = dummy_registry();
        let spec = LoadSpec::new(WeightsSource::Dir("/nonexistent".into()));
        assert!(registry.load_captioner("no_such_captioner", &spec).is_err());
    }

    #[test]
    fn dummy_captioner_appears_in_iteration() {
        assert!(dummy_registry()
            .captioners()
            .any(|r| (r.descriptor)().id == "dummy_test_captioner"));
    }

    #[test]
    fn transcriber_registry_resolves_by_id() {
        let registry = dummy_registry();
        let spec = LoadSpec::new(WeightsSource::Dir("/nonexistent".into()));
        let t = registry
            .load_transcriber("dummy_test_transcriber", &spec)
            .expect("dummy transcriber is registered");
        assert_eq!(t.descriptor().id, "dummy_test_transcriber");
    }

    #[test]
    fn unknown_transcriber_id_errors() {
        let registry = dummy_registry();
        let spec = LoadSpec::new(WeightsSource::Dir("/nonexistent".into()));
        assert!(registry
            .load_transcriber("no_such_transcriber", &spec)
            .is_err());
    }

    #[test]
    fn dummy_transcriber_appears_in_iteration() {
        assert!(dummy_registry()
            .transcribers()
            .any(|r| (r.descriptor)().id == "dummy_test_transcriber"));
    }

    struct DummyTextEmbedder {
        desc: TextEmbedderDescriptor,
    }

    impl TextEmbedder for DummyTextEmbedder {
        fn descriptor(&self) -> &TextEmbedderDescriptor {
            &self.desc
        }

        fn embed_text(&self, text: &str) -> Result<Vec<f32>> {
            Ok(vec![text.len() as f32, 1.0])
        }
    }

    fn dummy_text_embedder_descriptor() -> TextEmbedderDescriptor {
        TextEmbedderDescriptor {
            id: "dummy_test_text_embedder",
            family: "test",
            backend: "mlx",
            embedding_dim: 2,
            space: "test-space",
            mac_only: true,
        }
    }

    fn dummy_text_embedder_load(_spec: &LoadSpec) -> Result<Box<dyn TextEmbedder>> {
        Ok(Box::new(DummyTextEmbedder {
            desc: dummy_text_embedder_descriptor(),
        }))
    }

    crate::register_text_embedder! {
        const DUMMY_TEXT_EMBEDDER_REGISTRATION =
            dummy_text_embedder_descriptor => dummy_text_embedder_load
    }

    #[test]
    fn text_embedder_registry_resolves_by_id() {
        let registry = dummy_registry();
        let spec = LoadSpec::new(WeightsSource::Dir("/nonexistent".into()));
        let e = registry
            .load_text_embedder("dummy_test_text_embedder", &spec)
            .expect("dummy text embedder is registered");
        assert_eq!(e.descriptor().id, "dummy_test_text_embedder");
        assert_eq!(e.embed_text("clip").unwrap(), vec![4.0, 1.0]);
        assert_eq!(
            e.embed_text_batch(&["a", "abcd"]).unwrap(),
            vec![vec![1.0, 1.0], vec![4.0, 1.0]]
        );
    }

    #[test]
    fn unknown_text_embedder_id_errors() {
        let registry = dummy_registry();
        let spec = LoadSpec::new(WeightsSource::Dir("/nonexistent".into()));
        assert!(registry
            .load_text_embedder("no_such_text_embedder", &spec)
            .is_err());
    }

    #[test]
    fn dummy_text_embedder_appears_in_iteration() {
        assert!(dummy_registry()
            .text_embedders()
            .any(|r| (r.descriptor)().id == "dummy_test_text_embedder"));
    }

    struct DummyImageEmbedder {
        desc: ImageEmbedderDescriptor,
    }

    impl ImageEmbedder for DummyImageEmbedder {
        fn descriptor(&self) -> &ImageEmbedderDescriptor {
            &self.desc
        }

        fn embed(&self, image: &Image) -> Result<Vec<f32>> {
            Ok(vec![image.width as f32, image.height as f32])
        }
    }

    fn dummy_image_embedder_descriptor() -> ImageEmbedderDescriptor {
        ImageEmbedderDescriptor {
            id: "dummy_test_image_embedder",
            family: "test",
            backend: "mlx",
            embedding_dim: 2,
            space: "test-space",
            mac_only: true,
        }
    }

    fn dummy_image_embedder_load(_spec: &LoadSpec) -> Result<Box<dyn ImageEmbedder>> {
        Ok(Box::new(DummyImageEmbedder {
            desc: dummy_image_embedder_descriptor(),
        }))
    }

    crate::register_image_embedder! {
        const DUMMY_IMAGE_EMBEDDER_REGISTRATION =
            dummy_image_embedder_descriptor => dummy_image_embedder_load
    }

    #[test]
    fn image_embedder_registry_resolves_by_id() {
        let registry = dummy_registry();
        let spec = LoadSpec::new(WeightsSource::Dir("/nonexistent".into()));
        let e = registry
            .load_image_embedder("dummy_test_image_embedder", &spec)
            .expect("dummy image embedder is registered");
        assert_eq!(e.descriptor().id, "dummy_test_image_embedder");
        let img = Image {
            width: 7,
            height: 3,
            pixels: vec![0; 7 * 3 * 3],
        };
        assert_eq!(e.embed(&img).unwrap(), vec![7.0, 3.0]);
    }

    #[test]
    fn unknown_image_embedder_id_errors() {
        let registry = dummy_registry();
        let spec = LoadSpec::new(WeightsSource::Dir("/nonexistent".into()));
        assert!(registry
            .load_image_embedder("no_such_image_embedder", &spec)
            .is_err());
    }

    #[test]
    fn dummy_image_embedder_appears_in_iteration() {
        assert!(dummy_registry()
            .image_embedders()
            .any(|r| (r.descriptor)().id == "dummy_test_image_embedder"));
    }

    // ---- audio transforms (sc-12839) --------------------------------------------------------

    /// A weights-free stub audio transform: `apply` returns `out_tracks` copies of the source clip
    /// (one for the single-output kinds, one per stem for separation), retargeted to the request's
    /// sample rate — enough to exercise register→resolve→apply for all three shapes without a tensor
    /// backend.
    struct DummyAudioTransform {
        desc: AudioTransformDescriptor,
        out_tracks: usize,
    }

    impl AudioTransform for DummyAudioTransform {
        fn descriptor(&self) -> &AudioTransformDescriptor {
            &self.desc
        }
        fn validate(&self, _req: &AudioTransformRequest) -> Result<()> {
            Ok(())
        }
        fn apply(
            &self,
            req: &AudioTransformRequest,
            _on_progress: &mut dyn FnMut(Progress),
        ) -> Result<Vec<AudioTrack>> {
            let rate = match req.target {
                AudioTarget::Preserve => req.audio.sample_rate,
                AudioTarget::SampleRate(r) => r,
            };
            Ok(vec![
                AudioTrack {
                    sample_rate: rate,
                    ..req.audio.clone()
                };
                self.out_tracks
            ])
        }
    }

    fn dummy_voice_conversion_descriptor() -> AudioTransformDescriptor {
        AudioTransformDescriptor {
            id: "dummy_voice_conversion",
            family: "audio",
            backend: "candle",
            capabilities: AudioTransformCapabilities {
                kind: AudioTransformKind::VoiceConversion,
                ..Default::default()
            },
        }
    }

    fn dummy_stem_separation_descriptor() -> AudioTransformDescriptor {
        AudioTransformDescriptor {
            id: "dummy_stem_separation",
            family: "audio",
            backend: "candle",
            capabilities: AudioTransformCapabilities {
                kind: AudioTransformKind::StemSeparation,
                stem_count: 4,
                ..Default::default()
            },
        }
    }

    fn dummy_super_resolution_descriptor() -> AudioTransformDescriptor {
        AudioTransformDescriptor {
            id: "dummy_super_resolution",
            family: "audio",
            backend: "candle",
            capabilities: AudioTransformCapabilities {
                kind: AudioTransformKind::SuperResolution,
                is_diffusion: true,
                supports_resample: true,
                ..Default::default()
            },
        }
    }

    fn dummy_voice_conversion_load(_spec: &LoadSpec) -> Result<Box<dyn AudioTransform>> {
        Ok(Box::new(DummyAudioTransform {
            desc: dummy_voice_conversion_descriptor(),
            out_tracks: 1,
        }))
    }

    fn dummy_stem_separation_load(_spec: &LoadSpec) -> Result<Box<dyn AudioTransform>> {
        Ok(Box::new(DummyAudioTransform {
            desc: dummy_stem_separation_descriptor(),
            out_tracks: 4,
        }))
    }

    fn dummy_super_resolution_load(_spec: &LoadSpec) -> Result<Box<dyn AudioTransform>> {
        Ok(Box::new(DummyAudioTransform {
            desc: dummy_super_resolution_descriptor(),
            out_tracks: 1,
        }))
    }

    crate::register_audio_transform! {
        const DUMMY_VOICE_CONVERSION_REGISTRATION =
            dummy_voice_conversion_descriptor => dummy_voice_conversion_load
    }
    crate::register_audio_transform! {
        const DUMMY_STEM_SEPARATION_REGISTRATION =
            dummy_stem_separation_descriptor => dummy_stem_separation_load
    }
    crate::register_audio_transform! {
        const DUMMY_SUPER_RESOLUTION_REGISTRATION =
            dummy_super_resolution_descriptor => dummy_super_resolution_load
    }

    fn audio_transform_registry() -> ProviderRegistry {
        ProviderRegistryBuilder::new()
            .register_audio_transform(DUMMY_VOICE_CONVERSION_REGISTRATION)
            .register_audio_transform(DUMMY_STEM_SEPARATION_REGISTRATION)
            .register_audio_transform(DUMMY_SUPER_RESOLUTION_REGISTRATION)
            .build()
            .unwrap()
    }

    #[test]
    fn audio_transform_registry_resolves_by_id() {
        let registry = audio_transform_registry();
        let spec = LoadSpec::new(WeightsSource::Dir("/nonexistent".into()));
        let t = registry
            .load_audio_transform("dummy_voice_conversion", &spec)
            .expect("dummy voice conversion is registered");
        assert_eq!(t.descriptor().id, "dummy_voice_conversion");
    }

    #[test]
    fn unknown_audio_transform_id_errors() {
        let registry = audio_transform_registry();
        let spec = LoadSpec::new(WeightsSource::Dir("/nonexistent".into()));
        assert!(registry
            .load_audio_transform("no_such_audio_transform", &spec)
            .is_err());
    }

    #[test]
    fn audio_transforms_appear_in_iteration() {
        let registry = audio_transform_registry();
        assert_eq!(registry.audio_transforms().len(), 3);
        assert!(registry
            .audio_transforms()
            .any(|r| (r.descriptor)().id == "dummy_stem_separation"));
    }

    #[test]
    fn duplicate_audio_transform_id_is_rejected() {
        let err = ProviderRegistryBuilder::new()
            .register_audio_transform(DUMMY_VOICE_CONVERSION_REGISTRATION)
            .register_audio_transform(DUMMY_VOICE_CONVERSION_REGISTRATION)
            .build()
            .err()
            .expect("duplicate audio transform id must fail");
        assert_eq!(
            err.to_string(),
            "duplicate audio transform id 'dummy_voice_conversion' in explicit registry"
        );
    }

    #[test]
    fn audio_transform_descriptors_pass_conformance() {
        assert!(audio_transform_registry()
            .descriptor_conformance_errors()
            .is_empty());
    }

    /// The sc-12839 acceptance path end to end, weights-free: register the three non-prompt
    /// audio→audio shapes, then resolve + `apply` each and assert the output shape.
    #[test]
    fn all_three_audio_transform_shapes_resolve_and_apply() {
        let registry = audio_transform_registry();
        let spec = LoadSpec::new(WeightsSource::Dir("/nonexistent".into()));
        let source = dummy_audio_track(1024, 16_000);

        // audio→audio: voice conversion, rate-preserving, single output.
        let vc = registry
            .load_audio_transform("dummy_voice_conversion", &spec)
            .unwrap();
        let converted = vc
            .apply(
                &AudioTransformRequest {
                    audio: source.clone(),
                    ..Default::default()
                },
                &mut |_| {},
            )
            .unwrap();
        assert_eq!(converted.len(), 1);
        assert_eq!(converted[0].sample_rate, 16_000);

        // audio→Vec<audio>: stem separation, multi output.
        let stems = registry
            .load_audio_transform("dummy_stem_separation", &spec)
            .unwrap();
        let separated = stems
            .apply(
                &AudioTransformRequest {
                    audio: source.clone(),
                    ..Default::default()
                },
                &mut |_| {},
            )
            .unwrap();
        assert_eq!(separated.len(), 4);
        assert_eq!(
            stems.descriptor().capabilities.stem_count as usize,
            separated.len()
        );

        // audio→audio: super-resolution / bandwidth extension to a higher rate, single output.
        let sr = registry
            .load_audio_transform("dummy_super_resolution", &spec)
            .unwrap();
        let restored = sr
            .apply(
                &AudioTransformRequest {
                    audio: source,
                    target: AudioTarget::SampleRate(48_000),
                    ..Default::default()
                },
                &mut |_| {},
            )
            .unwrap();
        assert_eq!(restored.len(), 1);
        assert_eq!(restored[0].sample_rate, 48_000);
    }

    fn dummy_audio_track(samples: usize, rate: u32) -> AudioTrack {
        AudioTrack {
            samples: vec![0.0; samples],
            sample_rate: rate,
            channels: 1,
            ..Default::default()
        }
    }

    // Malformed audio-transform descriptors exercising the kind/stem_count coherence guard's
    // *rejection* branches (an inverted condition would otherwise pass the whole suite, since the
    // positive test only asserts an empty sweep). The `load` fn is never invoked by the sweep.
    fn bad_load(_spec: &LoadSpec) -> Result<Box<dyn AudioTransform>> {
        Ok(Box::new(DummyAudioTransform {
            desc: dummy_voice_conversion_descriptor(),
            out_tracks: 1,
        }))
    }

    fn separator_stem_count(id: &'static str, stem_count: u16) -> AudioTransformDescriptor {
        AudioTransformDescriptor {
            id,
            family: "audio",
            backend: "candle",
            capabilities: AudioTransformCapabilities {
                kind: AudioTransformKind::StemSeparation,
                stem_count,
                ..Default::default()
            },
        }
    }
    fn single_output_stem_count(
        id: &'static str,
        kind: AudioTransformKind,
    ) -> AudioTransformDescriptor {
        AudioTransformDescriptor {
            id,
            family: "audio",
            backend: "candle",
            capabilities: AudioTransformCapabilities {
                kind,
                stem_count: 3,
                ..Default::default()
            },
        }
    }

    fn bad_stems_zero_descriptor() -> AudioTransformDescriptor {
        separator_stem_count("bad_stems_zero", 0)
    }
    fn bad_stems_one_descriptor() -> AudioTransformDescriptor {
        separator_stem_count("bad_stems_one", 1)
    }
    fn bad_vc_stems_descriptor() -> AudioTransformDescriptor {
        single_output_stem_count("bad_vc_stems", AudioTransformKind::VoiceConversion)
    }
    fn bad_sr_stems_descriptor() -> AudioTransformDescriptor {
        single_output_stem_count("bad_sr_stems", AudioTransformKind::SuperResolution)
    }

    crate::register_audio_transform! {
        const BAD_STEMS_ZERO_REGISTRATION = bad_stems_zero_descriptor => bad_load
    }
    crate::register_audio_transform! {
        const BAD_STEMS_ONE_REGISTRATION = bad_stems_one_descriptor => bad_load
    }
    crate::register_audio_transform! {
        const BAD_VC_STEMS_REGISTRATION = bad_vc_stems_descriptor => bad_load
    }
    crate::register_audio_transform! {
        const BAD_SR_STEMS_REGISTRATION = bad_sr_stems_descriptor => bad_load
    }

    #[test]
    fn audio_transform_kind_stem_count_incoherence_is_rejected() {
        let errs = ProviderRegistryBuilder::new()
            .register_audio_transform(BAD_STEMS_ZERO_REGISTRATION)
            .register_audio_transform(BAD_STEMS_ONE_REGISTRATION)
            .register_audio_transform(BAD_VC_STEMS_REGISTRATION)
            .register_audio_transform(BAD_SR_STEMS_REGISTRATION)
            .build()
            .unwrap()
            .descriptor_conformance_errors();
        let has = |needle: &str| errs.iter().any(|e| e.contains(needle));

        // A separator advertising < 2 stems (0 and 1) is rejected with the specific message.
        assert!(
            has("audio transform 'bad_stems_zero': StemSeparation advertises stem_count 0 (a separator must produce ≥ 2 stems)"),
            "{errs:?}"
        );
        assert!(
            has("audio transform 'bad_stems_one': StemSeparation advertises stem_count 1 (a separator must produce ≥ 2 stems)"),
            "{errs:?}"
        );
        // A single-output kind advertising any stems is rejected.
        assert!(
            has("audio transform 'bad_vc_stems': VoiceConversion advertises stem_count 3 — only StemSeparation produces stems"),
            "{errs:?}"
        );
        assert!(
            has("audio transform 'bad_sr_stems': SuperResolution advertises stem_count 3 — only StemSeparation produces stems"),
            "{errs:?}"
        );
    }

    struct DummyVoiceEmbedder {
        desc: VoiceEmbedderDescriptor,
    }

    impl VoiceEmbedder for DummyVoiceEmbedder {
        fn descriptor(&self) -> &VoiceEmbedderDescriptor {
            &self.desc
        }

        fn embed(&self, audio: &AudioTrack) -> Result<VoiceEmbedding> {
            Ok(vec![audio.samples.len() as f32; self.desc.embedding_dim])
        }
    }

    fn dummy_voice_embedder_descriptor() -> VoiceEmbedderDescriptor {
        VoiceEmbedderDescriptor {
            id: "dummy_test_voice_embedder",
            family: "voice",
            backend: "candle",
            embedding_dim: 4,
            mac_only: false,
        }
    }

    fn dummy_voice_embedder_load(_spec: &LoadSpec) -> Result<Box<dyn VoiceEmbedder>> {
        Ok(Box::new(DummyVoiceEmbedder {
            desc: dummy_voice_embedder_descriptor(),
        }))
    }

    crate::register_voice_embedder! {
        const DUMMY_VOICE_EMBEDDER_REGISTRATION =
            dummy_voice_embedder_descriptor => dummy_voice_embedder_load
    }

    fn dummy_audio(samples: usize) -> AudioTrack {
        AudioTrack {
            samples: vec![0.0; samples],
            sample_rate: 24_000,
            channels: 1,
            ..Default::default()
        }
    }

    #[test]
    fn voice_embedder_registry_resolves_by_id() {
        let registry = dummy_registry();
        let spec = LoadSpec::new(WeightsSource::Dir("/nonexistent".into()));
        let e = registry
            .load_voice_embedder("dummy_test_voice_embedder", &spec)
            .expect("dummy voice embedder is registered");
        assert_eq!(e.descriptor().id, "dummy_test_voice_embedder");
        assert_eq!(e.embed(&dummy_audio(3)).unwrap(), vec![3.0; 4]);
    }

    #[test]
    fn unknown_voice_embedder_id_errors() {
        let registry = dummy_registry();
        let spec = LoadSpec::new(WeightsSource::Dir("/nonexistent".into()));
        assert!(registry
            .load_voice_embedder("no_such_voice_embedder", &spec)
            .is_err());
    }

    #[test]
    fn dummy_voice_embedder_appears_in_iteration() {
        assert!(dummy_registry()
            .voice_embedders()
            .any(|r| (r.descriptor)().id == "dummy_test_voice_embedder"));
    }

    // ---- audio embedders (sc-12851) ---------------------------------------------------------

    /// A weights-free joint audio-text embedder: both `embed` and `embed_text` return a **one-hot**
    /// unit vector, keyed off the audio clip's length (a proxy for its semantic "category") and off
    /// the text's length respectively. That is deliberately enough structure to drive the DoD
    /// cross-modal ranking test without a tensor backend: a text query lands on the one-hot index of
    /// exactly one clip, so cosine ranks that clip first — and the test FAILS if the stub ignored the
    /// audio (every clip identical) or if the audio/text vectors were not the same length (not joint).
    struct DummyAudioEmbedder {
        desc: AudioEmbedderDescriptor,
    }

    impl DummyAudioEmbedder {
        fn one_hot(&self, index: usize) -> Vec<f32> {
            let mut v = vec![0.0; self.desc.embedding_dim];
            v[index % self.desc.embedding_dim] = 1.0;
            v
        }
    }

    impl AudioEmbedder for DummyAudioEmbedder {
        fn descriptor(&self) -> &AudioEmbedderDescriptor {
            &self.desc
        }
        fn embed(&self, audio: &AudioTrack) -> Result<Vec<f32>> {
            Ok(self.one_hot(audio.samples.len()))
        }
        fn embed_text(&self, text: &str) -> Result<Vec<f32>> {
            Ok(self.one_hot(text.len()))
        }
    }

    fn dummy_audio_embedder_descriptor() -> AudioEmbedderDescriptor {
        AudioEmbedderDescriptor {
            id: "dummy_test_audio_embedder",
            family: "audio-embed",
            backend: "candle",
            embedding_dim: 4,
            space: "test-space",
            mac_only: false,
        }
    }

    fn dummy_audio_embedder_load(_spec: &LoadSpec) -> Result<Box<dyn AudioEmbedder>> {
        Ok(Box::new(DummyAudioEmbedder {
            desc: dummy_audio_embedder_descriptor(),
        }))
    }

    crate::register_audio_embedder! {
        const DUMMY_AUDIO_EMBEDDER_REGISTRATION =
            dummy_audio_embedder_descriptor => dummy_audio_embedder_load
    }

    #[test]
    fn audio_embedder_registry_resolves_by_id() {
        let registry = dummy_registry();
        let spec = LoadSpec::new(WeightsSource::Dir("/nonexistent".into()));
        let e = registry
            .load_audio_embedder("dummy_test_audio_embedder", &spec)
            .expect("dummy audio embedder is registered");
        assert_eq!(e.descriptor().id, "dummy_test_audio_embedder");
        // audio and text embeddings share the joint dim.
        assert_eq!(e.embed(&dummy_audio(1)).unwrap().len(), 4);
        assert_eq!(e.embed_text("q").unwrap().len(), 4);
    }

    #[test]
    fn unknown_audio_embedder_id_errors() {
        let registry = dummy_registry();
        let spec = LoadSpec::new(WeightsSource::Dir("/nonexistent".into()));
        assert!(registry
            .load_audio_embedder("no_such_audio_embedder", &spec)
            .is_err());
    }

    #[test]
    fn dummy_audio_embedder_appears_in_iteration() {
        assert!(dummy_registry()
            .audio_embedders()
            .any(|r| (r.descriptor)().id == "dummy_test_audio_embedder"));
    }

    /// The sc-12851 acceptance path end to end, weights-free: resolve the joint embedder, embed a
    /// SET of audio clips spanning "categories", embed a TEXT query, and assert the semantically
    /// matching clip ranks HIGHEST by cosine over the others — the cross-modal retrieval the DoD
    /// pins. Designed to fail if the embedder ignored the audio (all clips equidistant) or the audio
    /// and text vectors were not in one joint space.
    #[test]
    fn audio_text_query_ranks_the_matching_clip_highest() {
        fn cosine(a: &[f32], b: &[f32]) -> f32 {
            let dot: f32 = a.iter().zip(b).map(|(x, y)| x * y).sum();
            let na: f32 = a.iter().map(|x| x * x).sum::<f32>().sqrt();
            let nb: f32 = b.iter().map(|x| x * x).sum::<f32>().sqrt();
            dot / (na * nb)
        }

        let registry = dummy_registry();
        let spec = LoadSpec::new(WeightsSource::Dir("/nonexistent".into()));
        let e = registry
            .load_audio_embedder("dummy_test_audio_embedder", &spec)
            .expect("registered");

        // Three clips whose lengths map to distinct one-hot categories (1, 2, 3).
        let clips = [dummy_audio(1), dummy_audio(2), dummy_audio(3)];
        let clip_vecs: Vec<Vec<f32>> = clips.iter().map(|c| e.embed(c).unwrap()).collect();

        // A 3-char query lands on category 3 → the third clip is the match.
        let query = e.embed_text("abc").unwrap();
        let scores: Vec<f32> = clip_vecs.iter().map(|v| cosine(&query, v)).collect();

        let best = scores
            .iter()
            .enumerate()
            .max_by(|a, b| a.1.total_cmp(b.1))
            .map(|(i, _)| i)
            .unwrap();
        assert_eq!(
            best, 2,
            "the matching clip must rank first, scores={scores:?}"
        );
        assert!(scores[2] > scores[0] && scores[2] > scores[1], "{scores:?}");
    }

    /// The sc-12838 acceptance path end to end, weights-free: resolve a registered voice embedder,
    /// drive `embed()` over a reference clip, then feed the resulting embedding into a stub audio
    /// [`Generator`]'s conditioning (`Conditioning::VoiceEmbedding`) and validate — a cloned voice
    /// driving TTS, the audio mirror of a face embedding conditioning InstantID/PuLID.
    #[test]
    fn voice_embedding_resolves_embeds_and_conditions_a_generator() {
        use crate::generator::{Conditioning, ConditioningKind};

        let registry = dummy_registry();
        let spec = LoadSpec::new(WeightsSource::Dir("/nonexistent".into()));

        // resolve → embed
        let embedder = registry
            .load_voice_embedder("dummy_test_voice_embedder", &spec)
            .expect("registered");
        let embedding = embedder.embed(&dummy_audio(5)).unwrap();
        assert_eq!(embedding.len(), 4);

        // A stub audio generator that advertises VoiceEmbedding conditioning.
        let tts = DummyGen {
            desc: ModelDescriptor {
                id: "dummy_tts",
                family: "test",
                backend: "mlx",
                modality: Modality::Audio,
                capabilities: Capabilities {
                    max_count: 1,
                    conditioning: vec![ConditioningKind::VoiceEmbedding],
                    audio_sample_rates: vec![24_000],
                    ..Default::default()
                },
            },
        };

        // feed the embedding into the generator's conditioning and validate (size-skipping audio floor)
        let req = GenerationRequest {
            conditioning: vec![Conditioning::VoiceEmbedding {
                embedding,
                strength: None,
            }],
            steps: Some(1),
            ..Default::default()
        };
        tts.descriptor()
            .capabilities
            .validate_request_audio("dummy_tts", &req)
            .expect("a cloned-voice embedding is accepted conditioning for the TTS generator");

        // A generator that does NOT advertise it rejects the same conditioning.
        let no_voice = Capabilities {
            max_count: 1,
            audio_sample_rates: vec![24_000],
            ..Default::default()
        };
        assert!(no_voice.validate_request_audio("dummy_tts", &req).is_err());
    }

    /// The sweep (sc-9098, F-009) is clean over the explicit dummy catalog.
    #[test]
    fn descriptor_sweep_is_clean_over_dummy_catalog() {
        let errs = dummy_registry().descriptor_conformance_errors();
        assert!(
            errs.is_empty(),
            "descriptor conformance FAILED:\n  - {}",
            errs.join("\n  - ")
        );
    }

    /// Each per-descriptor invariant fires: identity shape, zero/inverted bounds, duplicate or
    /// malformed curated names, duplicate conditioning, video conditioning on an Image model.
    #[test]
    fn model_descriptor_errors_flags_each_violation() {
        // A fully-coherent descriptor produces no errors.
        assert!(model_descriptor_errors(&dummy_descriptor()).is_empty());

        let broken = ModelDescriptor {
            id: "Bad Id", // uppercase + whitespace
            family: "",   // empty
            backend: "mlx",
            modality: Modality::Image,
            capabilities: Capabilities {
                min_size: 512,
                max_size: 256,                                // inverted
                max_count: 0,                                 // zero
                samplers: vec!["euler", "euler", "bad name"], // duplicate + whitespace
                conditioning: vec![
                    ConditioningKind::Reference,
                    ConditioningKind::Reference, // duplicate
                    ConditioningKind::VideoClip, // video kind on an Image model
                ],
                ..Default::default()
            },
        };
        let errs = model_descriptor_errors(&broken);
        let has = |needle: &str| errs.iter().any(|e| e.contains(needle));
        assert!(has("id \"Bad Id\""), "{errs:?}");
        assert!(has("family \"\""), "{errs:?}");
        assert!(has("max_count is 0"), "{errs:?}");
        assert!(has("min_size 512 > max_size 256"), "{errs:?}");
        assert!(has("duplicate sampler entry \"euler\""), "{errs:?}");
        assert!(has("sampler[2] \"bad name\""), "{errs:?}");
        assert!(has("duplicate conditioning kind Reference"), "{errs:?}");
        assert!(has("video conditioning VideoClip"), "{errs:?}");

        // All-zero bounds report the Default-0 message (F-084), not the inverted-bounds one.
        let zeroed = ModelDescriptor {
            id: "zeroed",
            family: "test",
            backend: "mlx",
            modality: Modality::Image,
            capabilities: Capabilities::default(),
        };
        assert!(model_descriptor_errors(&zeroed)
            .iter()
            .any(|e| e.contains("left at the Default 0")));
    }

    /// The size-bounds floor is exempt for `Modality::Audio` (sc-13314): a pure-audio generator has
    /// no width/height, so `min_size`/`max_size` left at the unused `Default` 0 must NOT be flagged —
    /// mirroring the size-skipping `validate_request_audio` floor. The exemption is scoped to the
    /// size axis: every other invariant (identity, `max_count`, curated-name shape) still fires for an
    /// audio descriptor.
    #[test]
    fn audio_descriptor_with_zero_size_bounds_passes_sweep() {
        let audio = ModelDescriptor {
            id: "zeroed_audio",
            family: "test",
            backend: "candle",
            modality: Modality::Audio,
            capabilities: Capabilities {
                // No spatial dims — bounds stay at the natural unused 0.
                min_size: 0,
                max_size: 0,
                max_count: 1,
                audio_sample_rates: vec![24_000],
                ..Default::default()
            },
        };
        assert!(
            model_descriptor_errors(&audio).is_empty(),
            "an audio descriptor with unused (0) size bounds must pass the sweep: {:?}",
            model_descriptor_errors(&audio)
        );

        // The exemption is only the size axis: a broken `max_count` on the same audio descriptor is
        // still reported.
        let audio_bad_count = ModelDescriptor {
            capabilities: Capabilities {
                max_count: 0,
                ..audio.capabilities.clone()
            },
            ..audio.clone()
        };
        assert!(
            model_descriptor_errors(&audio_bad_count)
                .iter()
                .any(|e| e.contains("max_count is 0")),
            "the audio exemption must not relax the non-size invariants"
        );
    }

    /// The strictness the sweep exists for is preserved for the visual modalities: an `Image`/`Video`
    /// descriptor with invalid size bounds is STILL flagged (the audio exemption above does not weaken
    /// the check for modalities that genuinely carry a spatial size range).
    #[test]
    fn visual_descriptor_with_invalid_size_bounds_still_fails_sweep() {
        // Video, zero bounds → the Default-0 footgun still fires.
        let video_zero = ModelDescriptor {
            id: "video_zero",
            family: "test",
            backend: "mlx",
            modality: Modality::Video,
            capabilities: Capabilities {
                min_size: 0,
                max_size: 0,
                max_count: 1,
                ..Default::default()
            },
        };
        assert!(
            model_descriptor_errors(&video_zero)
                .iter()
                .any(|e| e.contains("left at the Default 0")),
            "a Video descriptor with zero size bounds must still be rejected"
        );

        // Image, inverted bounds → the inverted-bounds message still fires.
        let image_inverted = ModelDescriptor {
            id: "image_inverted",
            family: "test",
            backend: "mlx",
            modality: Modality::Image,
            capabilities: Capabilities {
                min_size: 512,
                max_size: 256,
                max_count: 1,
                ..Default::default()
            },
        };
        assert!(
            model_descriptor_errors(&image_inverted)
                .iter()
                .any(|e| e.contains("min_size 512 > max_size 256")),
            "an Image descriptor with inverted size bounds must still be rejected"
        );

        // `Both` (emits image or video) is a visual modality too — zero bounds still fail.
        let both_zero = ModelDescriptor {
            id: "both_zero",
            family: "test",
            backend: "mlx",
            modality: Modality::Both,
            capabilities: Capabilities {
                min_size: 0,
                max_size: 0,
                max_count: 1,
                ..Default::default()
            },
        };
        assert!(
            model_descriptor_errors(&both_zero)
                .iter()
                .any(|e| e.contains("left at the Default 0")),
            "a Both-modality descriptor with zero size bounds must still be rejected"
        );
    }

    /// Build a synthetic diffusers-style snapshot with a `bytes`-sized `model.safetensors` under each
    /// named subdir, returning the root. The caller cleans it up.
    fn synthetic_snapshot(tag: &str, subdirs: &[(&str, usize)]) -> PathBuf {
        let root = std::env::temp_dir().join(format!(
            "gencore_footprint_{tag}_{}_{}",
            std::process::id(),
            line!()
        ));
        for (sub, bytes) in subdirs {
            let dir = root.join(sub);
            std::fs::create_dir_all(&dir).unwrap();
            std::fs::write(dir.join("model.safetensors"), vec![0u8; *bytes]).unwrap();
        }
        root
    }

    /// sc-10894: a provider that declared a footprint returns the per-component on-disk split, resolved
    /// from the exact subdirs its loader uses — including a text encoder under a NON-`text_encoder`
    /// subdir (`mllm/`, the boogu layout) that a name-guessing consumer would read as zero.
    #[test]
    fn footprint_returns_provider_component_split() {
        let root = synthetic_snapshot(
            "split",
            &[("mllm", 1500), ("transformer", 9000), ("vae", 400)],
        );
        let spec = LoadSpec::new(WeightsSource::Dir(root.clone()));

        let fp = dummy_registry()
            .footprint("dummy_footprint_model", &spec)
            .expect("registered + declares a footprint")
            .expect("Some — the provider computed the split");
        assert_eq!(
            fp,
            PerComponentBytes {
                text_encoder: 1500,
                dit: 9000,
                vae: 400,
            }
        );
        // The whole point: the text encoder is NON-zero even though it is not under `text_encoder*`.
        assert!(fp.text_encoder > 0, "mllm/ text encoder must be measured");

        std::fs::remove_dir_all(&root).ok();
    }

    /// A registered generator that declares NO footprint yields `Ok(None)` (the consumer falls back);
    /// an unknown id is an `Err`.
    #[test]
    fn footprint_is_none_without_declaration_and_errs_on_unknown_id() {
        let spec = LoadSpec::new(WeightsSource::Dir("/nonexistent".into()));
        // `dummy_test_model` is registered but declares no footprint.
        let registry = dummy_registry();
        assert_eq!(registry.footprint("dummy_test_model", &spec).unwrap(), None);
        // Unknown id → Err (a fail-open consumer treats it like None).
        assert!(registry.footprint("no_such_model", &spec).is_err());
    }

    /// sc-10894: `from_spec_subdirs` sums each component's subdir(s) (SD3's three text encoders here),
    /// treats a missing subdir as `0`, and errors on a single-`File` source (no tree to split).
    #[test]
    fn per_component_bytes_from_spec_subdirs_and_file_guard() {
        let root = synthetic_snapshot(
            "sd3",
            &[
                ("text_encoder", 100),
                ("text_encoder_2", 200),
                ("text_encoder_3", 4000),
                ("transformer", 8000),
                ("vae", 300),
            ],
        );
        let spec = LoadSpec::new(WeightsSource::Dir(root.clone()));
        let fp = PerComponentBytes::from_spec_subdirs(
            &spec,
            &["text_encoder", "text_encoder_2", "text_encoder_3"],
            &["transformer"],
            &["vae"],
        )
        .unwrap();
        assert_eq!(fp.text_encoder, 4300); // 100 + 200 + 4000
        assert_eq!(fp.dit, 8000);
        assert_eq!(fp.vae, 300);

        // A named-but-absent subdir contributes 0 (does not error).
        let fp_missing =
            PerComponentBytes::from_spec_subdirs(&spec, &["nope"], &["transformer"], &["vae"])
                .unwrap();
        assert_eq!(fp_missing.text_encoder, 0);

        // A single-file source has no component tree → Err (consumer falls back to whole-file).
        let file_spec = LoadSpec::new(WeightsSource::File(
            root.join("transformer/model.safetensors"),
        ));
        assert!(
            PerComponentBytes::from_spec_subdirs(&file_spec, &["te"], &["dit"], &["vae"]).is_err()
        );

        std::fs::remove_dir_all(&root).ok();
    }

    /// sc-10894: `from_root_subdirs` sums a component named by a flat FILE (the bernini/anima layout —
    /// `t5_encoder.safetensors` at the root, not a `text_encoder/` subdir) as well as a subdir, against
    /// an already-resolved root.
    #[test]
    fn per_component_bytes_from_root_subdirs_handles_flat_files() {
        let root = std::env::temp_dir().join(format!(
            "gencore_footprint_flat_{}_{}",
            std::process::id(),
            line!()
        ));
        std::fs::create_dir_all(&root).unwrap();
        // bernini-style flat component files at the root.
        std::fs::write(root.join("t5_encoder.safetensors"), vec![0u8; 2000]).unwrap();
        std::fs::write(root.join("low_noise_model.safetensors"), vec![0u8; 6000]).unwrap();
        std::fs::write(root.join("high_noise_model.safetensors"), vec![0u8; 6000]).unwrap();
        std::fs::write(root.join("vae.safetensors"), vec![0u8; 500]).unwrap();

        let fp = PerComponentBytes::from_root_subdirs(
            &root,
            &["t5_encoder.safetensors"],
            &[
                "low_noise_model.safetensors",
                "high_noise_model.safetensors",
            ],
            &["vae.safetensors"],
        );
        assert_eq!(
            fp,
            PerComponentBytes {
                text_encoder: 2000,
                dit: 12000,
                vae: 500,
            }
        );

        std::fs::remove_dir_all(&root).ok();
    }
}
