//! The built, immutable `ProviderRegistry` and its resolve-by-id surface.

use super::*;

/// An immutable, explicit catalog of generative-media providers.
pub struct ProviderRegistry {
    pub(super) generators: Box<[ModelRegistration]>,
    pub(super) checkpoint_adapters: Box<[CheckpointAdapterRegistration]>,
    pub(super) checkpoint_codecs: CheckpointCodecRegistry,
    pub(super) imported_models: Box<[ImportedModelRegistration]>,
    pub(super) encoder_contract_routes: Box<[EncoderContractRouteRegistration]>,
    pub(super) memory_strategy: Box<[MemoryRegistration]>,
    pub(super) memory_contract_fixture: Box<[MemoryContractFixtureRegistration]>,
    pub(super) memory_contract_surface_resolver: Box<[MemoryContractSurfaceResolverRegistration]>,
    pub(super) resident_only_memory_contract: Box<[ResidentOnlyMemoryContractRegistration]>,
    pub(super) memory_behavior: Box<[MemoryBehaviorRegistration]>,
    pub(super) activation_memory: Box<[ActivationMemoryRegistration]>,
    pub(super) reference_workspace: Box<[ReferenceWorkspaceRegistration]>,
    pub(super) composed_memory_strategy_ids: Box<[&'static str]>,
    pub(super) transforms: Box<[TransformRegistration]>,
    pub(super) audio_transforms: Box<[AudioTransformRegistration]>,
    pub(super) trainers: Box<[TrainerRegistration]>,
    pub(super) captioners: Box<[CaptionerRegistration]>,
    pub(super) transcribers: Box<[TranscriberRegistration]>,
    pub(super) image_embedders: Box<[ImageEmbedderRegistration]>,
    pub(super) text_embedders: Box<[TextEmbedderRegistration]>,
    pub(super) voice_embedders: Box<[VoiceEmbedderRegistration]>,
    pub(super) audio_embedders: Box<[AudioEmbedderRegistration]>,
    pub(super) rejected_quants: Box<[(Quant, &'static str)]>,
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
            self.ensure_quant_supported(id, spec, $kind)?;
            let _activity = spec.load_activity_context().install();
            spec.read_prepared_files_unchanged(|| (registration.load)(spec))
        }
    };
}

impl ProviderRegistry {
    /// Every validated portable family checkpoint adapter in this platform catalog.
    pub fn checkpoint_adapters(
        &self,
    ) -> impl ExactSizeIterator<Item = &CheckpointAdapterRegistration> {
        self.checkpoint_adapters.iter()
    }

    /// The validated codec table this platform catalog registered (engine-level, registered once).
    pub fn checkpoint_codecs(&self) -> &CheckpointCodecRegistry {
        &self.checkpoint_codecs
    }

    /// Check that two platform catalogs agree on the stable family/adapter-id mapping and portable
    /// checkpoint metadata, and collectively bind every backend and operation they actually ship.
    ///
    /// A family that declares only MLX eligibility is allowed to be absent from a Candle catalog
    /// (and vice versa). A family eligible for both must be present in both platform catalogs and
    /// retain at least one real binding for each backend. Per-platform provider ids and operation
    /// subsets remain intentionally asymmetric.
    pub fn checkpoint_adapter_catalog_conformance_errors(&self, other: &Self) -> Vec<String> {
        fn catalog_backends(
            registry: &ProviderRegistry,
        ) -> std::collections::BTreeSet<CheckpointBackend> {
            registry
                .generators
                .iter()
                .filter_map(|registration| match (registration.descriptor)().backend {
                    "mlx" => Some(CheckpointBackend::Mlx),
                    "candle" => Some(CheckpointBackend::Candle),
                    _ => None,
                })
                .collect()
        }

        fn ordered_pair<'a>(left: &'a str, right: &'a str) -> (&'a str, &'a str) {
            if left <= right {
                (left, right)
            } else {
                (right, left)
            }
        }

        let self_backends = catalog_backends(self);
        let other_backends = catalog_backends(other);
        let catalog_backends: std::collections::BTreeSet<_> =
            self_backends.union(&other_backends).copied().collect();
        let self_adapters_by_id: std::collections::BTreeMap<_, _> = self
            .checkpoint_adapters
            .iter()
            .map(|adapter| (adapter.adapter_id, adapter))
            .collect();
        let other_adapters_by_id: std::collections::BTreeMap<_, _> = other
            .checkpoint_adapters
            .iter()
            .map(|adapter| (adapter.adapter_id, adapter))
            .collect();
        let adapter_ids: std::collections::BTreeSet<_> = self_adapters_by_id
            .keys()
            .chain(other_adapters_by_id.keys())
            .copied()
            .collect();
        let self_adapters_by_family: std::collections::BTreeMap<_, _> = self
            .checkpoint_adapters
            .iter()
            .map(|adapter| (adapter.family, adapter))
            .collect();
        let other_adapters_by_family: std::collections::BTreeMap<_, _> = other
            .checkpoint_adapters
            .iter()
            .map(|adapter| (adapter.family, adapter))
            .collect();
        let families: std::collections::BTreeSet<_> = self_adapters_by_family
            .keys()
            .chain(other_adapters_by_family.keys())
            .copied()
            .collect();
        let self_adapters_by_compatibility_family: std::collections::BTreeMap<_, _> = self
            .checkpoint_adapters
            .iter()
            .map(|adapter| (adapter.compatibility_projection.family, adapter))
            .collect();
        let other_adapters_by_compatibility_family: std::collections::BTreeMap<_, _> = other
            .checkpoint_adapters
            .iter()
            .map(|adapter| (adapter.compatibility_projection.family, adapter))
            .collect();
        let compatibility_families: std::collections::BTreeSet<_> =
            self_adapters_by_compatibility_family
                .keys()
                .chain(other_adapters_by_compatibility_family.keys())
                .copied()
                .collect();
        let mut errors = Vec::new();

        for compatibility_family in compatibility_families {
            let left = self_adapters_by_compatibility_family
                .get(compatibility_family)
                .copied();
            let right = other_adapters_by_compatibility_family
                .get(compatibility_family)
                .copied();
            if let (Some(left), Some(right)) = (left, right) {
                if left.family != right.family && left.adapter_id != right.adapter_id {
                    let (first, second) =
                        if (left.family, left.adapter_id) <= (right.family, right.adapter_id) {
                            (left, right)
                        } else {
                            (right, left)
                        };
                    errors.push(format!(
                        "checkpoint-adapter compatibility family '{compatibility_family}' maps to different portable authorities '{}' (id '{}') and '{}' (id '{}') across catalogs",
                        first.family, first.adapter_id, second.family, second.adapter_id
                    ));
                }
            }
        }

        for adapter_id in adapter_ids {
            let left = self_adapters_by_id.get(adapter_id).copied();
            let right = other_adapters_by_id.get(adapter_id).copied();
            if let (Some(left), Some(right)) = (left, right) {
                if left.family != right.family {
                    let (first_family, second_family) = ordered_pair(left.family, right.family);
                    errors.push(format!(
                        "checkpoint-adapter id '{adapter_id}' maps to different families '{}' and '{}' across catalogs",
                        first_family, second_family
                    ));
                }
            }
        }

        for family in families {
            let left = self_adapters_by_family.get(family).copied();
            let right = other_adapters_by_family.get(family).copied();
            let authority = left.or(right).expect("family came from one catalog");

            match (left, right) {
                (Some(left), Some(right)) if left.adapter_id != right.adapter_id => {
                    let (first_id, second_id) = ordered_pair(left.adapter_id, right.adapter_id);
                    errors.push(format!(
                        "checkpoint-adapter family '{family}' maps to different adapter ids '{}' and '{}' across catalogs",
                        first_id, second_id
                    ));
                    continue;
                }
                (Some(left), Some(right)) if !left.has_same_portable_metadata(right) => {
                    errors.push(format!(
                        "checkpoint-adapter family '{family}' (id '{}') portable metadata differs across catalogs",
                        left.adapter_id
                    ));
                    continue;
                }
                (Some(left), None)
                    if other_adapters_by_id
                        .get(left.adapter_id)
                        .is_some_and(|right| right.family != family) =>
                {
                    continue;
                }
                (None, Some(right))
                    if self_adapters_by_id
                        .get(right.adapter_id)
                        .is_some_and(|left| left.family != family) =>
                {
                    continue;
                }
                (Some(_), None)
                    if authority
                        .eligible_backends
                        .iter()
                        .any(|backend| other_backends.contains(backend)) =>
                {
                    errors.push(format!(
                        "checkpoint-adapter family '{family}' (id '{}') is missing from a catalog that ships an eligible backend",
                        authority.adapter_id
                    ));
                }
                (None, Some(_))
                    if authority
                        .eligible_backends
                        .iter()
                        .any(|backend| self_backends.contains(backend)) =>
                {
                    errors.push(format!(
                        "checkpoint-adapter family '{family}' (id '{}') is missing from a catalog that ships an eligible backend",
                        authority.adapter_id
                    ));
                }
                _ => {}
            }

            for backend in authority
                .eligible_backends
                .iter()
                .copied()
                .filter(|backend| catalog_backends.contains(backend))
            {
                let bound = left
                    .into_iter()
                    .chain(right)
                    .flat_map(|adapter| adapter.backend_bindings)
                    .any(|binding| binding.backend == backend);
                if !bound {
                    errors.push(format!(
                        "checkpoint-adapter family '{family}' (id '{}') has no binding for shipped eligible backend '{}'",
                        authority.adapter_id,
                        backend.descriptor_label()
                    ));
                }
            }

            for operation in authority.operations {
                let bound = left
                    .into_iter()
                    .chain(right)
                    .flat_map(|adapter| adapter.backend_bindings)
                    .any(|binding| binding.operation == *operation);
                if !bound {
                    errors.push(format!(
                        "checkpoint-adapter family '{family}' (id '{}') operation {:?} has no binding across eligible catalogs",
                        authority.adapter_id, operation
                    ));
                }
            }
        }

        errors
    }

    /// Resolve the encoder contract for an ordinary generator or an explicitly registered bespoke
    /// route. Unknown routes, and ordinary generators that do not support substitution, return
    /// `None`; aliases are validated at registry construction and cannot target another alias or a
    /// provider without a contract.
    pub fn provider_encoder_contract(&self, id: &str) -> Option<crate::EncoderContract> {
        if let Some(descriptor) = self
            .generators
            .iter()
            .map(|registration| (registration.descriptor)())
            .find(|descriptor| descriptor.id == id)
        {
            return descriptor.encoder_contract;
        }
        let provider_id = self
            .encoder_contract_routes
            .iter()
            .find(|registration| registration.route_id == id)?
            .provider_id;
        self.generators
            .iter()
            .map(|registration| (registration.descriptor)())
            .find(|descriptor| descriptor.id == provider_id)?
            .encoder_contract
    }

    /// Every provider-owned bespoke route participating in encoder-contract lookup.
    pub fn encoder_contract_routes(
        &self,
    ) -> impl ExactSizeIterator<Item = &EncoderContractRouteRegistration> {
        self.encoder_contract_routes.iter()
    }

    /// Compatibility projection of every adapter-owned imported route in this platform catalog.
    ///
    /// New code consumes [`Self::checkpoint_adapters`]. Existing SceneWorks consumers can continue
    /// using this view until their family migration moves them to persisted import plans.
    pub fn imported_models(&self) -> impl ExactSizeIterator<Item = &ImportedModelRegistration> {
        self.imported_models.iter()
    }

    /// Resolve an exact imported source shape and operation to the descriptor of the generator that
    /// will actually load it. Missing is an explicit unsupported answer.
    ///
    /// A route whose binding declares `inherit_adapters = false` has `supports_lora`/`supports_lokr`
    /// **withdrawn** from the descriptor returned here. That withdrawal is *advertisement only* —
    /// this function admits nothing and refuses nothing. Turning it into a typed refusal is the job
    /// of the **provider-build seam gate**: every provider's `build(spec, descriptor)` calls
    /// [`crate::reject_unsupported_adapters`] with the descriptor it was handed (see
    /// `candle_gen_krea`'s `build`), which is the single gate for the sc-21483 / epic-11037-E6
    /// silent-un-adapted-render class. There is deliberately no second, registry-side wrapper: a
    /// guard that no load route calls is indistinguishable from no guard at all, so a consumer that
    /// resolves a descriptor here is responsible for passing *that* descriptor — not the provider's
    /// own — into the build seam.
    pub fn imported_model_descriptor(
        &self,
        family: &str,
        source: ImportedModelSource,
        operation: ImportedModelOperation,
    ) -> Option<ModelDescriptor> {
        let route = self.imported_models.iter().find(|route| {
            route.family == family && route.source == source && route.operation == operation
        })?;
        let mut descriptor = self
            .generators
            .iter()
            .find(|registration| (registration.descriptor)().id == route.provider_id)
            .map(|registration| (registration.descriptor)())?;
        if !route.inherit_adapters {
            descriptor.capabilities.supports_lora = false;
            descriptor.capabilities.supports_lokr = false;
        }
        if let Some(required_components) = route.required_components {
            descriptor.required_components = required_components;
        }
        Some(descriptor)
    }

    /// Provider-owned warm 1024² activation transient for `id`.
    /// `Ok(None)` is the compatibility-safe unmeasured state, including for a known platform-composed
    /// memory route with no standalone generator registration; genuinely unknown ids remain errors.
    pub fn activation_memory_bytes_1024(&self, id: &str) -> Result<Option<u64>> {
        let has_generator = self
            .generators()
            .any(|registration| (registration.descriptor)().id == id);
        let is_composed_route = self.composed_memory_strategy_ids.contains(&id);
        if !has_generator && !is_composed_route {
            return Err(Error::Msg(format!("no generator registered for id '{id}'")));
        }
        Ok(self
            .activation_memory
            .iter()
            .find(|registration| registration.provider_id == id)
            .map(|registration| registration.anchor.bytes_1024))
    }

    /// Provider-owned PRE-FIT bytes of a request's reference work for generator `id` on `spec`.
    /// `facts == None` (no references) is `Ok(Some(0))` for a registered provider, so decisions
    /// without references are unchanged; an unregistered provider is `Ok(None)` (unpriced); an
    /// unknown id is an error.
    pub fn reference_workspace_bytes(
        &self,
        id: &str,
        spec: &LoadSpec,
        facts: Option<&crate::MemoryReferenceResourceFacts>,
    ) -> Result<Option<u64>> {
        if !self
            .generators()
            .any(|registration| (registration.descriptor)().id == id)
        {
            return Err(Error::Msg(format!("no generator registered for id '{id}'")));
        }
        let Some(registration) = self
            .reference_workspace
            .iter()
            .find(|registration| registration.provider_id == id)
        else {
            return Ok(None);
        };
        match facts {
            None => Ok(Some(0)),
            Some(facts) => (registration.reference_workspace_bytes)(spec, facts).map(Some),
        }
    }

    /// Every adopted memory-strategy registration in this explicit catalog.
    pub fn memory_strategy_registrations(
        &self,
    ) -> impl ExactSizeIterator<Item = &MemoryRegistration> {
        self.memory_strategy.iter()
    }

    pub fn memory_contract_fixture_registrations(
        &self,
    ) -> impl ExactSizeIterator<Item = &MemoryContractFixtureRegistration> {
        self.memory_contract_fixture.iter()
    }

    pub fn memory_contract_surface_resolver_registrations(
        &self,
    ) -> impl ExactSizeIterator<Item = &MemoryContractSurfaceResolverRegistration> {
        self.memory_contract_surface_resolver.iter()
    }

    pub fn resident_only_memory_contract_registrations(
        &self,
    ) -> impl ExactSizeIterator<Item = &ResidentOnlyMemoryContractRegistration> {
        self.resident_only_memory_contract.iter()
    }

    /// Construct every provider-owned registry-load contract witness without opening weights.
    ///
    /// Coverage is fail-closed in both directions: each memory registration must have exactly one
    /// paired surface fixture or explicit resident-only witness (builder validation rejects overlap
    /// and orphans). Every declaration must publish a non-empty unique selector set and construct a
    /// contract for the paired provider id. Resident-only declarations are checked across the same
    /// axes but omitted from the returned optimized-surface inventory. This is the seam consumed by
    /// generated capability dumps; it deliberately has no caller-supplied `LoadSpec`.
    pub fn memory_contract_surfaces(&self) -> Result<Vec<MemoryContractSurface>> {
        let mut out = Vec::new();
        for registration in &self.memory_strategy {
            let fixture = self
                .memory_contract_fixture
                .iter()
                .find(|fixture| fixture.provider_id == registration.provider_id);
            let resident_only = self
                .resident_only_memory_contract
                .iter()
                .find(|witness| witness.provider_id == registration.provider_id);
            let surface_resolver = self
                .memory_contract_surface_resolver
                .iter()
                .find(|resolver| resolver.provider_id == registration.provider_id);
            let (surface_specs, contract_factory, is_resident_only) = match (fixture, resident_only) {
                (Some(fixture), None) => (fixture.surface_specs, fixture.contract, false),
                (None, Some(witness)) => (witness.surface_specs, witness.contract, true),
                (None, None) => {
                    return Err(Error::Msg(format!(
                        "memory-strategy registration '{}' has neither a weights-free contract-surface fixture nor a resident-only witness",
                        registration.provider_id
                    )))
                }
                (Some(_), Some(_)) => {
                    return Err(Error::Msg(format!(
                        "memory-strategy registration '{}' has both a contract-surface fixture and a resident-only witness",
                        registration.provider_id
                    )))
                }
            };
            let surface_specs = surface_specs();
            if surface_specs.is_empty() {
                return Err(Error::Msg(format!(
                    "memory-contract witness '{}' publishes no surface selectors",
                    registration.provider_id
                )));
            }
            let mut selectors = std::collections::BTreeSet::new();
            for surface in surface_specs {
                if !surface.selector.matches_spec(&surface.spec) {
                    return Err(Error::Msg(format!(
                        "memory-contract witness '{}' selector '{}' does not match its LoadSpec",
                        registration.provider_id,
                        surface.selector.id()
                    )));
                }
                if !selectors.insert(surface.selector.id()) {
                    return Err(Error::Msg(format!(
                        "memory-contract witness '{}' repeats surface selector '{}'",
                        registration.provider_id,
                        surface.selector.id()
                    )));
                }
                let contract = match surface_resolver {
                    Some(resolver) => (resolver.contract)(&surface),
                    None => contract_factory(&surface.spec),
                }
                .map_err(|error| {
                    Error::Msg(format!(
                        "memory-contract witness '{}' failed surface '{}': {error}",
                        registration.provider_id,
                        surface.selector.id()
                    ))
                })?;
                if contract.provider_id != registration.provider_id {
                    return Err(Error::Msg(format!(
                        "memory-contract witness '{}' surface '{}' returned contract for '{}'",
                        registration.provider_id,
                        surface.selector.id(),
                        contract.provider_id
                    )));
                }
                if is_resident_only {
                    let optimized = contract.strategies.iter().find(|capability| {
                        capability.strategy != MemoryStrategy::Resident
                            && capability.support != MemoryStrategySupport::Missing
                    });
                    if let Some(capability) = optimized {
                        return Err(Error::Msg(format!(
                            "resident-only memory-contract witness '{}' surface '{}' exposes {:?} as {:?}",
                            registration.provider_id,
                            surface.selector.id(),
                            capability.strategy,
                            capability.support
                        )));
                    }
                    let errors = contract.conformance_errors();
                    if !errors.is_empty() {
                        return Err(Error::Msg(format!(
                            "resident-only memory-contract witness '{}' surface '{}' is malformed: {}",
                            registration.provider_id,
                            surface.selector.id(),
                            errors.join("; ")
                        )));
                    }
                    continue;
                }
                out.push(MemoryContractSurface {
                    selector: surface.selector,
                    spec: surface.spec,
                    contract,
                    composed: self
                        .composed_memory_strategy_ids
                        .contains(&registration.provider_id),
                });
            }
        }
        Ok(out)
    }

    pub fn memory_behavior_registrations(
        &self,
    ) -> impl ExactSizeIterator<Item = &MemoryBehaviorRegistration> {
        self.memory_behavior.iter()
    }

    /// Reject a [`LoadSpec`] whose requested quant tier this platform's backend does not implement,
    /// as declared by [`ProviderRegistryBuilder::reject_quant`], or whose YuE2 AR mode a different
    /// provider would silently ignore.
    ///
    /// The single boundary every registry-routed load of every provider kind passes through, so one
    /// check covers the whole catalog — the composition root states the platform's tier support once
    /// instead of each provider re-deriving it. Runs *after* id resolution so an unknown id still
    /// reports as an unknown id.
    pub(super) fn ensure_quant_supported(
        &self,
        id: &str,
        spec: &LoadSpec,
        kind: &str,
    ) -> Result<()> {
        if spec.yue2_ar_mode != crate::Yue2ArMode::Native && id != "yue2" {
            return Err(Error::Unsupported(format!(
                "the YuE2 AR mode {:?} cannot be used by provider '{id}'",
                spec.yue2_ar_mode
            )));
        }
        if spec.yue2_compute_policy != crate::Yue2ComputePolicy::Legacy && id != "yue2" {
            return Err(Error::Unsupported(format!(
                "the YuE2 compute policy {:?} cannot be used by provider '{id}'",
                spec.yue2_compute_policy
            )));
        }
        let Some(quant) = spec.quantize else {
            return Ok(());
        };
        match self.rejected_quants.iter().find(|(q, _)| *q == quant) {
            Some((_, reason)) => Err(Error::Unsupported(format!(
                "quant tier {quant:?} is not implemented by this runtime's backend \
                 (requested for '{id}'): {reason}. Refusing to load rather than silently \
                 serving a different tier's numerics."
            ))),
            None => {
                // A NEW packed float format never inherits affine Q8 support. Reuse
                // the descriptor's exact typed list as the one opt-in source; no
                // parallel builder allow-list or model-id exception exists.
                if quant == Quant::Fp8Rowwise
                    && (kind != "generator"
                        || !self.generators().any(|registration| {
                            let descriptor = (registration.descriptor)();
                            descriptor.id == id
                                && descriptor.capabilities.supported_quants.contains(&quant)
                        }))
                {
                    return Err(Error::Unsupported(format!("provider '{id}' does not advertise the exact fp8r packed format (it is not Q8); install/convert a compatible artifact and select a supported native tier")));
                }
                Ok(())
            }
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
        if spec.quantize == Some(Quant::Fp8Rowwise) {
            self.ensure_quant_supported(id, spec, "generator")?;
        }
        match registration.footprint {
            Some(footprint) => spec
                .read_prepared_files_unchanged(|| footprint(spec))
                .map(Some),
            None => Ok(None),
        }
    }

    /// Return the provider-owned memory-strategy contract for `id`, when adopted.
    ///
    /// `Ok(None)` is the compatibility-safe resident-only/unverified state. Unknown provider ids
    /// remain errors; a malformed adopted contract also fails instead of being silently trusted.
    pub fn memory_strategy_contract(
        &self,
        id: &str,
        spec: &LoadSpec,
    ) -> Result<Option<MemoryProviderContract>> {
        let has_generator = self
            .generators()
            .any(|registration| (registration.descriptor)().id == id);
        let is_composed_route = self.composed_memory_strategy_ids.contains(&id);
        if !has_generator && !is_composed_route {
            return Err(Error::Msg(format!("no generator registered for id '{id}'")));
        }
        if spec.quantize == Some(Quant::Fp8Rowwise) {
            self.ensure_quant_supported(id, spec, "generator")?;
        }
        let Some(registration) = self
            .memory_strategy
            .iter()
            .find(|registration| registration.provider_id == id)
        else {
            return Ok(None);
        };
        let contract = spec.read_prepared_files_unchanged(|| (registration.contract)(spec))?;
        let errors = contract.conformance_errors();
        if !errors.is_empty() {
            return Err(Error::Msg(format!(
                "memory-strategy contract '{id}' is malformed: {}",
                errors.join("; ")
            )));
        }
        Ok(Some(contract))
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
