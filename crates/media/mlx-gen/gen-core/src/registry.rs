//! Explicit model + transform discovery: provider crates publish registration constants, family
//! crates add them to a [`ProviderRegistryBuilder`], and platform catalogs select the families they
//! ship. This is the Rust equivalent of an ordinary DI composition root with resolve-by-id.

use crate::caption::{Captioner, CaptionerDescriptor};
use crate::generator::{ConditioningKind, Generator, Modality, ModelDescriptor};
use crate::image_embed::{ImageEmbedder, ImageEmbedderDescriptor};
use crate::runtime::LoadSpec;
use crate::text_embed::{TextEmbedder, TextEmbedderDescriptor};
use crate::train::{Trainer, TrainerDescriptor};
use crate::transform::{Transform, TransformDescriptor};
use crate::{Error, Result};

/// A generator provider's registration — `descriptor` for introspection (no weights loaded),
/// `load` to construct the model. ≈ `services.AddKeyedSingleton<IGenerator>("id", factory)`.
#[derive(Clone, Copy)]
pub struct ModelRegistration {
    pub descriptor: fn() -> ModelDescriptor,
    pub load: fn(&LoadSpec) -> Result<Box<dyn Generator>>,
}

/// A transform provider's registration (parallel to [`ModelRegistration`]).
#[derive(Clone, Copy)]
pub struct TransformRegistration {
    pub descriptor: fn() -> TransformDescriptor,
    pub load: fn(&LoadSpec) -> Result<Box<dyn Transform>>,
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

/// Builder for an ordinary, explicit generative-media provider registry.
///
/// Platform bundles add exactly the registrations they ship.
#[derive(Default)]
pub struct ProviderRegistryBuilder {
    generators: Vec<ModelRegistration>,
    transforms: Vec<TransformRegistration>,
    trainers: Vec<TrainerRegistration>,
    captioners: Vec<CaptionerRegistration>,
    image_embedders: Vec<ImageEmbedderRegistration>,
    text_embedders: Vec<TextEmbedderRegistration>,
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
    builder_registration_method!(register_trainer, trainers, TrainerRegistration);
    builder_registration_method!(register_captioner, captioners, CaptionerRegistration);
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
        ensure_unique!(trainers, "trainer");
        ensure_unique!(captioners, "captioner");
        ensure_unique!(image_embedders, "image embedder");
        ensure_unique!(text_embedders, "text embedder");

        Ok(ProviderRegistry {
            generators: self.generators.into_boxed_slice(),
            transforms: self.transforms.into_boxed_slice(),
            trainers: self.trainers.into_boxed_slice(),
            captioners: self.captioners.into_boxed_slice(),
            image_embedders: self.image_embedders.into_boxed_slice(),
            text_embedders: self.text_embedders.into_boxed_slice(),
        })
    }
}

/// An immutable, explicit catalog of generative-media providers.
pub struct ProviderRegistry {
    generators: Box<[ModelRegistration]>,
    transforms: Box<[TransformRegistration]>,
    trainers: Box<[TrainerRegistration]>,
    captioners: Box<[CaptionerRegistration]>,
    image_embedders: Box<[ImageEmbedderRegistration]>,
    text_embedders: Box<[TextEmbedderRegistration]>,
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
            (registration.load)(spec)
        }
    };
}

impl ProviderRegistry {
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

    /// Run the weights-free descriptor conformance sweep over this explicit catalog.
    pub fn descriptor_conformance_errors(&self) -> Vec<String> {
        descriptor_conformance_errors_for(
            &self.generators,
            &self.transforms,
            &self.trainers,
            &self.captioners,
            &self.image_embedders,
            &self.text_embedders,
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
/// - `max_count ≥ 1` and `1 ≤ min_size ≤ max_size` (a `Default` 0 bound rejects every request with
///   a confusing "out of range 0..=0" — the F-084 footgun, enforced here for *every* linked
///   descriptor rather than only when a request happens to reach `validate_request`),
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
/// image/text embedders. No `load` is ever called, so it runs by default (no weights, no Metal) —
/// each provider crate invokes it from a default test, giving every cataloged id at least
/// descriptor-level coverage; behavioral conformance (progress/cancel/seed) stays weights-gated in
/// the `gen-core-testkit` suite.
///
/// Returns one message per violation (empty = conformant).
fn descriptor_conformance_errors_for(
    generator_registrations: &[ModelRegistration],
    transform_registrations: &[TransformRegistration],
    trainer_registrations: &[TrainerRegistration],
    captioner_registrations: &[CaptionerRegistration],
    image_embedder_registrations: &[ImageEmbedderRegistration],
    text_embedder_registrations: &[TextEmbedderRegistration],
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

    let ie_descs: Vec<ImageEmbedderDescriptor> = image_embedder_registrations
        .iter()
        .map(|r| (r.descriptor)())
        .collect();
    let te_descs: Vec<TextEmbedderDescriptor> = text_embedder_registrations
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

    errs
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::caption::{
        CaptionCapabilities, CaptionOutput, CaptionRequest, Captioner, CaptionerDescriptor,
    };
    use crate::generator::{
        Capabilities, GenerationOutput, GenerationRequest, Modality, ModelDescriptor,
    };
    use crate::image_embed::{ImageEmbedder, ImageEmbedderDescriptor};
    use crate::media::Image;
    use crate::runtime::{Progress, WeightsSource};
    use crate::text_embed::{TextEmbedder, TextEmbedderDescriptor};
    use crate::train::{
        Trainer, TrainerDescriptor, TrainingOutput, TrainingProgress, TrainingRequest,
    };
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

    fn dummy_registry() -> ProviderRegistry {
        ProviderRegistryBuilder::new()
            .register_generator(DUMMY_GENERATOR_REGISTRATION)
            .register_generator(DUMMY_DELEGATED_GENERATOR_REGISTRATION)
            .register_generator(DUMMY_MULTI_GENERATOR_A_REGISTRATION)
            .register_generator(DUMMY_MULTI_GENERATOR_B_REGISTRATION)
            .register_trainer(DUMMY_TRAINER_REGISTRATION)
            .register_trainer(DUMMY_MULTI_TRAINER_A_REGISTRATION)
            .register_trainer(DUMMY_MULTI_TRAINER_B_REGISTRATION)
            .register_captioner(DUMMY_CAPTIONER_REGISTRATION)
            .register_text_embedder(DUMMY_TEXT_EMBEDDER_REGISTRATION)
            .register_image_embedder(DUMMY_IMAGE_EMBEDDER_REGISTRATION)
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

    #[test]
    fn explicit_registry_rejects_duplicate_ids_deterministically() {
        let registration = ModelRegistration {
            descriptor: dummy_descriptor,
            load: dummy_load,
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
}
