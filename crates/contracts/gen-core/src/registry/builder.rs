//! `ProviderRegistryBuilder` and the registration-time validation it runs.

use super::*;

/// Builder for an ordinary, explicit generative-media provider registry.
///
/// Platform bundles add exactly the registrations they ship.
#[derive(Default)]
pub struct ProviderRegistryBuilder {
    pub(super) generators: Vec<ModelRegistration>,
    pub(super) checkpoint_adapters: Vec<CheckpointAdapterRegistration>,
    pub(super) checkpoint_codecs: Vec<CheckpointCodecRegistration>,
    pub(super) encoder_contract_routes: Vec<EncoderContractRouteRegistration>,
    pub(super) memory_strategy: Vec<MemoryRegistration>,
    pub(super) memory_contract_fixture: Vec<MemoryContractFixtureRegistration>,
    pub(super) memory_contract_surface_resolver: Vec<MemoryContractSurfaceResolverRegistration>,
    pub(super) resident_only_memory_contract: Vec<ResidentOnlyMemoryContractRegistration>,
    pub(super) memory_behavior: Vec<MemoryBehaviorRegistration>,
    pub(super) activation_memory: Vec<ActivationMemoryRegistration>,
    pub(super) reference_workspace: Vec<ReferenceWorkspaceRegistration>,
    pub(super) composed_memory_strategy_ids: Vec<&'static str>,
    pub(super) transforms: Vec<TransformRegistration>,
    pub(super) audio_transforms: Vec<AudioTransformRegistration>,
    pub(super) trainers: Vec<TrainerRegistration>,
    pub(super) captioners: Vec<CaptionerRegistration>,
    pub(super) transcribers: Vec<TranscriberRegistration>,
    pub(super) image_embedders: Vec<ImageEmbedderRegistration>,
    pub(super) text_embedders: Vec<TextEmbedderRegistration>,
    pub(super) voice_embedders: Vec<VoiceEmbedderRegistration>,
    pub(super) audio_embedders: Vec<AudioEmbedderRegistration>,
    pub(super) rejected_quants: Vec<(Quant, &'static str)>,
}

macro_rules! builder_registration_method {
    ($name:ident, $field:ident, $registration:ty) => {
        pub fn $name(mut self, registration: $registration) -> Self {
            self.$field.push(registration);
            self
        }
    };
}

pub(super) fn validate_checkpoint_adapters(
    adapters: &[CheckpointAdapterRegistration],
    generators: &[ModelRegistration],
) -> Result<Vec<ImportedModelRegistration>> {
    fn require_surface(
        adapter: &CheckpointAdapterRegistration,
        name: &str,
        present: bool,
    ) -> Result<()> {
        if present {
            Ok(())
        } else {
            Err(Error::Msg(format!(
                "checkpoint-adapter '{}' {name} must not be empty",
                adapter.adapter_id
            )))
        }
    }

    let mut adapter_ids = std::collections::BTreeSet::new();
    let mut families = std::collections::BTreeMap::new();
    let mut compatibility_families = std::collections::BTreeMap::new();
    let mut projected_routes = std::collections::BTreeSet::new();
    let mut imported_models = Vec::new();

    for adapter in adapters {
        if !is_registry_ident(adapter.adapter_id) || !is_registry_ident(adapter.family) {
            return Err(Error::Msg(format!(
                "checkpoint-adapter identity {:?}/{:?} must use non-empty lowercase registry identifiers",
                adapter.adapter_id, adapter.family
            )));
        }
        if !is_registry_ident(adapter.compatibility_projection.family) {
            return Err(Error::Msg(format!(
                "checkpoint-adapter '{}' compatibility family {:?} must use a non-empty lowercase registry identifier",
                adapter.adapter_id, adapter.compatibility_projection.family
            )));
        }
        if !adapter_ids.insert(adapter.adapter_id) {
            return Err(Error::Msg(format!(
                "duplicate checkpoint-adapter id '{}'",
                adapter.adapter_id
            )));
        }
        if families
            .insert(adapter.family, adapter.adapter_id)
            .is_some()
        {
            return Err(Error::Msg(format!(
                "duplicate checkpoint-adapter family '{}'",
                adapter.family
            )));
        }
        if compatibility_families
            .insert(adapter.compatibility_projection.family, adapter.adapter_id)
            .is_some()
        {
            return Err(Error::Msg(format!(
                "duplicate checkpoint-adapter compatibility family '{}'",
                adapter.compatibility_projection.family
            )));
        }
    }
    for (compatibility_family, adapter_id) in &compatibility_families {
        if let Some(portable_owner) = families.get(compatibility_family) {
            if portable_owner != adapter_id {
                return Err(Error::Msg(format!(
                    "checkpoint-adapter compatibility family '{}' for '{}' collides with portable family owned by '{}'",
                    compatibility_family, adapter_id, portable_owner
                )));
            }
        }
    }

    for adapter in adapters {
        require_surface(adapter, "signatures", !adapter.signatures.is_empty())?;
        require_surface(adapter, "dialects", !adapter.dialects.is_empty())?;
        require_surface(
            adapter,
            "component topology",
            !adapter.component_topology.is_empty(),
        )?;
        require_surface(
            adapter,
            "base compatibility",
            !adapter.base_compatibility.is_empty(),
        )?;
        require_surface(
            adapter,
            "canonical mappings",
            !adapter.canonical_mappings.is_empty(),
        )?;
        require_surface(
            adapter,
            "config recovery",
            !adapter.config_recovery.is_empty(),
        )?;
        require_surface(
            adapter,
            "eligible backends",
            !adapter.eligible_backends.is_empty(),
        )?;
        if adapter.backend_bindings.is_empty() {
            return Err(Error::Msg(format!(
                "checkpoint-adapter '{}' has no backend bindings",
                adapter.adapter_id
            )));
        }
        require_surface(adapter, "operations", !adapter.operations.is_empty())?;
        require_surface(adapter, "capabilities", !adapter.capabilities.is_empty())?;

        let mut dialect_ids = std::collections::BTreeSet::new();
        for dialect in adapter.dialects {
            if !is_registry_ident(dialect.id) || !dialect_ids.insert(dialect.id) {
                return Err(Error::Msg(format!(
                    "checkpoint-adapter '{}' has malformed or duplicate dialect '{}'",
                    adapter.adapter_id, dialect.id
                )));
            }
        }

        let mut signature_ids = std::collections::BTreeSet::new();
        let mut signature_dialects = std::collections::BTreeSet::new();
        for signature in adapter.signatures {
            if !is_registry_ident(signature.id) || !signature_ids.insert(signature.id) {
                return Err(Error::Msg(format!(
                    "checkpoint-adapter '{}' has malformed or duplicate signature '{}'",
                    adapter.adapter_id, signature.id
                )));
            }
            if !dialect_ids.contains(signature.dialect) {
                return Err(Error::Msg(format!(
                    "checkpoint-adapter '{}' signature '{}' targets unknown dialect '{}'",
                    adapter.adapter_id, signature.id, signature.dialect
                )));
            }
            if signature.required_tensor_names.is_empty() {
                return Err(Error::Msg(format!(
                    "checkpoint-adapter '{}' signature '{}' has no required tensor names",
                    adapter.adapter_id, signature.id
                )));
            }
            let mut tensor_names = std::collections::BTreeSet::new();
            for name in signature.required_tensor_names {
                if name.is_empty()
                    || name.chars().any(char::is_whitespace)
                    || !tensor_names.insert(*name)
                {
                    return Err(Error::Msg(format!(
                        "checkpoint-adapter '{}' signature '{}' has an empty, whitespace-containing, or duplicate tensor name",
                        adapter.adapter_id, signature.id
                    )));
                }
            }
            signature_dialects.insert(signature.dialect);
        }
        for dialect in adapter.dialects {
            if !signature_dialects.contains(dialect.id) {
                return Err(Error::Msg(format!(
                    "checkpoint-adapter '{}' dialect '{}' has no signature",
                    adapter.adapter_id, dialect.id
                )));
            }
        }

        let mut component_roles = std::collections::BTreeSet::new();
        for component in adapter.component_topology {
            if !is_registry_ident(component.role)
                || !component_roles.insert(component.role)
                || component.min_count == 0
                || component.max_count < component.min_count
            {
                return Err(Error::Msg(format!(
                    "checkpoint-adapter '{}' has malformed or duplicate component role '{}'",
                    adapter.adapter_id, component.role
                )));
            }
        }
        let mut base_roles = std::collections::BTreeSet::new();
        for base in adapter.base_compatibility {
            if !component_roles.contains(base.component_role)
                || !base_roles.insert(base.component_role)
                || base.compatible_families.is_empty()
                || base
                    .compatible_families
                    .iter()
                    .any(|family| !is_registry_ident(family))
            {
                return Err(Error::Msg(format!(
                    "checkpoint-adapter '{}' has malformed, duplicate, or dangling base compatibility for role '{}'",
                    adapter.adapter_id, base.component_role
                )));
            }
            let unique: std::collections::BTreeSet<_> =
                base.compatible_families.iter().copied().collect();
            if unique.len() != base.compatible_families.len() {
                return Err(Error::Msg(format!(
                    "checkpoint-adapter '{}' repeats a compatible base family for role '{}'",
                    adapter.adapter_id, base.component_role
                )));
            }
        }

        let mut mapped_dialects = std::collections::BTreeSet::new();
        for mapping in adapter.canonical_mappings {
            if !dialect_ids.contains(mapping.dialect)
                || !is_registry_ident(mapping.mapping_id)
                || !mapped_dialects.insert(mapping.dialect)
            {
                return Err(Error::Msg(format!(
                    "checkpoint-adapter '{}' has malformed, duplicate, or dangling canonical mapping for dialect '{}'",
                    adapter.adapter_id, mapping.dialect
                )));
            }
        }
        if mapped_dialects.len() != dialect_ids.len() {
            return Err(Error::Msg(format!(
                "checkpoint-adapter '{}' must map every declared dialect",
                adapter.adapter_id
            )));
        }

        let mut recovery_fields = std::collections::BTreeSet::new();
        for recovery in adapter.config_recovery {
            if !is_registry_ident(recovery.field)
                || !is_registry_ident(recovery.recovery_id)
                || !recovery_fields.insert(recovery.field)
            {
                return Err(Error::Msg(format!(
                    "checkpoint-adapter '{}' has malformed or duplicate config recovery for field '{}'",
                    adapter.adapter_id, recovery.field
                )));
            }
        }

        let eligible_backends: std::collections::BTreeSet<_> =
            adapter.eligible_backends.iter().copied().collect();
        if eligible_backends.len() != adapter.eligible_backends.len() {
            return Err(Error::Msg(format!(
                "checkpoint-adapter '{}' repeats an eligible backend",
                adapter.adapter_id
            )));
        }
        // A dialect cannot be plan-driven on a backend the adapter is not eligible for, and cannot
        // name the same backend twice: the catalog conformance tests read this list as the exact
        // set of backends that must ship an implementation of `mapping_id`.
        for mapping in adapter.canonical_mappings {
            let plan_driven: std::collections::BTreeSet<_> =
                mapping.plan_driven_backends.iter().copied().collect();
            if plan_driven.len() != mapping.plan_driven_backends.len()
                || !plan_driven.is_subset(&eligible_backends)
            {
                return Err(Error::Msg(format!(
                    "checkpoint-adapter '{}' canonical mapping for dialect '{}' declares a repeated or ineligible plan-driven backend",
                    adapter.adapter_id, mapping.dialect
                )));
            }
        }

        let operations: std::collections::BTreeSet<_> =
            adapter.operations.iter().copied().collect();
        if operations.len() != adapter.operations.len() {
            return Err(Error::Msg(format!(
                "checkpoint-adapter '{}' repeats an operation",
                adapter.adapter_id
            )));
        }
        let mut capability_operations = std::collections::BTreeSet::new();
        for capability in adapter.capabilities {
            if !operations.contains(&capability.operation)
                || !capability_operations.insert(capability.operation)
                || !capability.inherit_provider_capabilities
            {
                return Err(Error::Msg(format!(
                    "checkpoint-adapter '{}' has a duplicate, undeclared, or non-projectable capability for {:?}",
                    adapter.adapter_id, capability.operation
                )));
            }
        }
        if capability_operations != operations {
            return Err(Error::Msg(format!(
                "checkpoint-adapter '{}' must own capability policy for every operation",
                adapter.adapter_id
            )));
        }

        let mut binding_keys = std::collections::BTreeSet::new();
        let mut binding_backends = std::collections::BTreeSet::new();
        for binding in adapter.backend_bindings {
            if !adapter
                .dialects
                .iter()
                .any(|dialect| dialect.source == binding.source)
            {
                return Err(Error::Msg(format!(
                    "checkpoint-adapter '{}' binding {:?}/{:?} has no dialect for its source shape",
                    adapter.adapter_id, binding.backend, binding.operation
                )));
            }
            if !eligible_backends.contains(&binding.backend)
                || !operations.contains(&binding.operation)
            {
                return Err(Error::Msg(format!(
                    "checkpoint-adapter '{}' binding {:?}/{:?} is not declared portable metadata",
                    adapter.adapter_id, binding.backend, binding.operation
                )));
            }
            let capability = adapter
                .capabilities
                .iter()
                .find(|capability| capability.operation == binding.operation)
                .expect("capability-operation equality validated above");
            if binding.inherit_adapters && !capability.supports_adapter_inheritance {
                return Err(Error::Msg(format!(
                    "checkpoint-adapter '{}' binding {:?}/{:?} inherits adapters contrary to capability policy",
                    adapter.adapter_id, binding.backend, binding.operation
                )));
            }
            let binding_key = (binding.backend, binding.source, binding.operation);
            if !binding_keys.insert(binding_key) {
                return Err(Error::Msg(format!(
                    "duplicate checkpoint-adapter binding '{}' ({:?}/{:?}/{:?})",
                    adapter.adapter_id, binding.backend, binding.source, binding.operation
                )));
            }
            binding_backends.insert(binding.backend);
            let projected_key = (
                adapter.compatibility_projection.family,
                binding.source,
                binding.operation,
            );
            if !projected_routes.insert(projected_key) {
                return Err(Error::Msg(format!(
                    "duplicate imported-model route for family '{}' ({:?}/{:?})",
                    adapter.compatibility_projection.family, binding.source, binding.operation
                )));
            }
            if !is_registry_ident(binding.provider_id) {
                return Err(Error::Msg(format!(
                    "checkpoint-adapter '{}' binding provider id is malformed",
                    adapter.adapter_id
                )));
            }
            if let Some(required) = binding.required_components {
                if required.is_empty() {
                    return Err(Error::Msg(format!(
                        "checkpoint-adapter '{}' binding {:?}/{:?} declares an empty required-components override; use None instead",
                        adapter.adapter_id, binding.source, binding.operation
                    )));
                }
                let mut component_ids = std::collections::BTreeSet::new();
                for component in required {
                    if !is_registry_ident(component) || !component_ids.insert(*component) {
                        return Err(Error::Msg(format!(
                            "checkpoint-adapter '{}' binding {:?}/{:?} has a malformed or duplicate required component",
                            adapter.adapter_id, binding.source, binding.operation
                        )));
                    }
                }
            }
            let Some(descriptor) = generators
                .iter()
                .map(|generator| (generator.descriptor)())
                .find(|descriptor| descriptor.id == binding.provider_id)
            else {
                return Err(Error::Msg(format!(
                    "checkpoint-adapter '{}' binding {:?}/{:?} targets unregistered generator '{}'",
                    adapter.adapter_id, binding.source, binding.operation, binding.provider_id
                )));
            };
            if descriptor.backend != binding.backend.descriptor_label() {
                return Err(Error::Msg(format!(
                    "checkpoint-adapter '{}' binding backend '{}' does not match generator backend '{}'",
                    adapter.adapter_id,
                    binding.backend.descriptor_label(),
                    descriptor.backend
                )));
            }
            if descriptor.family != adapter.family {
                return Err(Error::Msg(format!(
                    "checkpoint-adapter '{}' family '{}' does not match generator '{}' family '{}'",
                    adapter.adapter_id, adapter.family, binding.provider_id, descriptor.family
                )));
            }
            imported_models.push(ImportedModelRegistration {
                family: adapter.compatibility_projection.family,
                source: binding.source,
                operation: binding.operation,
                provider_id: binding.provider_id,
                required_components: binding.required_components,
                inherit_adapters: binding.inherit_adapters,
            });
        }
        if let [sole_backend] = adapter.eligible_backends {
            for operation in adapter.operations {
                if !adapter.backend_bindings.iter().any(|binding| {
                    binding.backend == *sole_backend && binding.operation == *operation
                }) {
                    return Err(Error::Msg(format!(
                        "checkpoint-adapter '{}' operation {:?} has no binding on sole eligible backend '{}'",
                        adapter.adapter_id,
                        operation,
                        sole_backend.descriptor_label()
                    )));
                }
            }
        }
        let shipped_family_backends: std::collections::BTreeSet<_> = generators
            .iter()
            .map(|generator| (generator.descriptor)())
            .filter(|descriptor| descriptor.family == adapter.family)
            .filter_map(|descriptor| match descriptor.backend {
                "mlx" => Some(CheckpointBackend::Mlx),
                "candle" => Some(CheckpointBackend::Candle),
                _ => None,
            })
            .filter(|backend| eligible_backends.contains(backend))
            .collect();
        for backend in shipped_family_backends {
            if !binding_backends.contains(&backend) {
                return Err(Error::Msg(format!(
                    "checkpoint-adapter '{}' has no binding for shipped eligible backend '{}'",
                    adapter.adapter_id,
                    backend.descriptor_label()
                )));
            }
        }
    }
    Ok(imported_models)
}

impl ProviderRegistryBuilder {
    /// Start an empty explicit registry.
    pub fn new() -> Self {
        Self::default()
    }

    builder_registration_method!(register_generator, generators, ModelRegistration);
    builder_registration_method!(
        register_checkpoint_adapter,
        checkpoint_adapters,
        CheckpointAdapterRegistration
    );
    /// Register one portable checkpoint codec row. Codecs are engine-level: a platform catalog
    /// registers its backend's baseline table exactly once, and a family adapter never carries a
    /// codec table of its own. `build` refuses duplicate ids and two codecs claiming one stored
    /// encoding, so a composed catalog that registers the same row twice fails closed instead of
    /// silently shipping duplicate rows.
    pub fn register_checkpoint_codec(mut self, registration: CheckpointCodecRegistration) -> Self {
        self.checkpoint_codecs.push(registration);
        self
    }
    builder_registration_method!(
        register_encoder_contract_route,
        encoder_contract_routes,
        EncoderContractRouteRegistration
    );
    builder_registration_method!(
        register_activation_memory,
        activation_memory,
        ActivationMemoryRegistration
    );
    builder_registration_method!(
        register_reference_workspace,
        reference_workspace,
        ReferenceWorkspaceRegistration
    );
    builder_registration_method!(
        register_memory_strategy,
        memory_strategy,
        MemoryRegistration
    );
    builder_registration_method!(
        register_memory_contract_fixture,
        memory_contract_fixture,
        MemoryContractFixtureRegistration
    );
    builder_registration_method!(
        register_memory_contract_surface_resolver,
        memory_contract_surface_resolver,
        MemoryContractSurfaceResolverRegistration
    );
    builder_registration_method!(
        register_resident_only_memory_contract,
        resident_only_memory_contract,
        ResidentOnlyMemoryContractRegistration
    );
    builder_registration_method!(
        register_memory_behavior,
        memory_behavior,
        MemoryBehaviorRegistration
    );

    /// Register memory policy for a real platform-composed route that is not represented by a
    /// standalone gen-core [`Generator`] registration.
    ///
    /// This explicit seam preserves the ordinary registration invariant: calling
    /// [`register_memory_strategy`](Self::register_memory_strategy) for an unmatched id is still an
    /// error. Composition roots must opt in route by route, so a typo cannot silently become a
    /// provider contract with no executable owner.
    pub fn register_composed_memory_strategy(mut self, registration: MemoryRegistration) -> Self {
        self.composed_memory_strategy_ids
            .push(registration.provider_id);
        self.memory_strategy.push(registration);
        self
    }
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
        {
            let mut route_ids = std::collections::BTreeSet::new();
            for registration in &self.encoder_contract_routes {
                if !is_registry_ident(registration.route_id) {
                    return Err(Error::Msg(
                        "encoder-contract route id must be a non-empty lowercase registry identifier"
                            .to_owned(),
                    ));
                }
                if !route_ids.insert(registration.route_id) {
                    return Err(Error::Msg(format!(
                        "duplicate encoder-contract route id '{}'",
                        registration.route_id
                    )));
                }
            }
            for registration in &self.encoder_contract_routes {
                if self
                    .generators
                    .iter()
                    .any(|generator| (generator.descriptor)().id == registration.route_id)
                {
                    return Err(Error::Msg(format!(
                        "encoder-contract route '{}' shadows a registered generator",
                        registration.route_id
                    )));
                }
                let Some(descriptor) = self
                    .generators
                    .iter()
                    .map(|generator| (generator.descriptor)())
                    .find(|descriptor| descriptor.id == registration.provider_id)
                else {
                    return Err(Error::Msg(format!(
                        "encoder-contract route '{}' targets unregistered generator '{}'",
                        registration.route_id, registration.provider_id
                    )));
                };
                if descriptor.encoder_contract.is_none() {
                    return Err(Error::Msg(format!(
                        "encoder-contract route '{}' targets generator '{}' with no encoder contract",
                        registration.route_id, registration.provider_id
                    )));
                }
            }
        }
        let imported_models =
            validate_checkpoint_adapters(&self.checkpoint_adapters, &self.generators)?;
        {
            let mut ids = std::collections::BTreeSet::new();
            for registration in &self.reference_workspace {
                if !ids.insert(registration.provider_id) {
                    return Err(Error::Msg(format!(
                        "duplicate reference-workspace provider id '{}'",
                        registration.provider_id
                    )));
                }
                if !self
                    .generators
                    .iter()
                    .any(|generator| (generator.descriptor)().id == registration.provider_id)
                {
                    return Err(Error::Msg(format!(
                        "reference-workspace registration '{}' has no matching generator",
                        registration.provider_id
                    )));
                }
            }
        }
        {
            let mut ids = std::collections::BTreeSet::new();
            for registration in &self.activation_memory {
                if !ids.insert(registration.provider_id) {
                    return Err(Error::Msg(format!(
                        "duplicate activation-memory provider id '{}'",
                        registration.provider_id
                    )));
                }
                if !self
                    .generators
                    .iter()
                    .any(|generator| (generator.descriptor)().id == registration.provider_id)
                {
                    return Err(Error::Msg(format!(
                        "activation-memory registration '{}' has no matching generator",
                        registration.provider_id
                    )));
                }
                if registration.anchor.bytes_1024 == 0 {
                    return Err(Error::Msg(format!(
                        "activation-memory registration '{}' is zero — omit unmeasured routes",
                        registration.provider_id
                    )));
                }
            }
        }
        {
            let mut ids = std::collections::BTreeSet::new();
            for registration in &self.memory_contract_fixture {
                if !ids.insert(registration.provider_id) {
                    return Err(Error::Msg(format!(
                        "duplicate memory-contract fixture provider id '{}'",
                        registration.provider_id
                    )));
                }
                if !self
                    .memory_strategy
                    .iter()
                    .any(|memory| memory.provider_id == registration.provider_id)
                {
                    return Err(Error::Msg(format!(
                        "memory-contract fixture '{}' has no matching memory strategy",
                        registration.provider_id
                    )));
                }
            }
            let mut resolver_ids = std::collections::BTreeSet::new();
            for registration in &self.memory_contract_surface_resolver {
                if !resolver_ids.insert(registration.provider_id) {
                    return Err(Error::Msg(format!(
                        "duplicate memory-contract surface resolver provider id '{}'",
                        registration.provider_id
                    )));
                }
                if !ids.contains(registration.provider_id) {
                    return Err(Error::Msg(format!(
                        "memory-contract surface resolver '{}' has no matching contract-surface fixture",
                        registration.provider_id
                    )));
                }
            }
            let mut resident_only_ids = std::collections::BTreeSet::new();
            for registration in &self.resident_only_memory_contract {
                if !resident_only_ids.insert(registration.provider_id) {
                    return Err(Error::Msg(format!(
                        "duplicate resident-only memory-contract witness provider id '{}'",
                        registration.provider_id
                    )));
                }
                if ids.contains(registration.provider_id) {
                    return Err(Error::Msg(format!(
                        "memory-strategy registration '{}' has both a contract-surface fixture and a resident-only witness",
                        registration.provider_id
                    )));
                }
                if !self
                    .memory_strategy
                    .iter()
                    .any(|memory| memory.provider_id == registration.provider_id)
                {
                    return Err(Error::Msg(format!(
                        "resident-only memory-contract witness '{}' has no matching memory strategy",
                        registration.provider_id
                    )));
                }
            }
        }
        {
            let mut ids = std::collections::BTreeSet::new();
            for registration in &self.memory_behavior {
                if !ids.insert(registration.provider_id) {
                    return Err(Error::Msg(format!(
                        "duplicate memory-behavior provider id '{}'",
                        registration.provider_id
                    )));
                }
                if !self
                    .memory_strategy
                    .iter()
                    .any(|memory| memory.provider_id == registration.provider_id)
                {
                    return Err(Error::Msg(format!(
                        "memory-behavior registration '{}' has no matching memory strategy",
                        registration.provider_id
                    )));
                }
            }
        }
        {
            let mut ids = std::collections::BTreeSet::new();
            let mut composed_ids = std::collections::BTreeSet::new();
            for id in &self.composed_memory_strategy_ids {
                if !composed_ids.insert(*id) {
                    return Err(Error::Msg(format!(
                        "duplicate composed memory-strategy route id '{id}'"
                    )));
                }
            }
            for registration in &self.memory_strategy {
                if !ids.insert(registration.provider_id) {
                    return Err(Error::Msg(format!(
                        "duplicate memory-strategy provider id '{}'",
                        registration.provider_id
                    )));
                }
                if !self
                    .generators
                    .iter()
                    .any(|generator| (generator.descriptor)().id == registration.provider_id)
                    && !composed_ids.contains(registration.provider_id)
                {
                    return Err(Error::Msg(format!(
                        "memory-strategy contract '{}' has no matching generator registration",
                        registration.provider_id
                    )));
                }
            }
        }
        ensure_unique!(transforms, "transform");
        ensure_unique!(audio_transforms, "audio transform");
        ensure_unique!(trainers, "trainer");
        ensure_unique!(captioners, "captioner");
        ensure_unique!(transcribers, "transcriber");
        ensure_unique!(image_embedders, "image embedder");
        ensure_unique!(text_embedders, "text embedder");
        ensure_unique!(voice_embedders, "voice embedder");
        ensure_unique!(audio_embedders, "audio embedder");
        let checkpoint_codecs =
            CheckpointCodecRegistry::new(self.checkpoint_codecs.iter().copied())?;

        Ok(ProviderRegistry {
            generators: self.generators.into_boxed_slice(),
            checkpoint_adapters: self.checkpoint_adapters.into_boxed_slice(),
            checkpoint_codecs,
            imported_models: imported_models.into_boxed_slice(),
            encoder_contract_routes: self.encoder_contract_routes.into_boxed_slice(),
            memory_strategy: self.memory_strategy.into_boxed_slice(),
            memory_contract_fixture: self.memory_contract_fixture.into_boxed_slice(),
            memory_contract_surface_resolver: self
                .memory_contract_surface_resolver
                .into_boxed_slice(),
            resident_only_memory_contract: self.resident_only_memory_contract.into_boxed_slice(),
            memory_behavior: self.memory_behavior.into_boxed_slice(),
            activation_memory: self.activation_memory.into_boxed_slice(),
            reference_workspace: self.reference_workspace.into_boxed_slice(),
            composed_memory_strategy_ids: self.composed_memory_strategy_ids.into_boxed_slice(),
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
