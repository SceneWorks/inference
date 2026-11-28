//! Shared Candle/CUDA image-memory ladder for the bespoke PuLID-FLUX route (SC-15839).
//!
//! PuLID reuses FLUX.1-dev's staged/windowed reference backbone, but owns a distinct admission
//! identity because EVA-CLIP, IDFormer, the face stack, and the 20 CA modules stay resident. Base
//! FLUX evidence must therefore never authorize this route.

use candle_gen::gen_core::{
    self, LoadShape, LoadSpec, MemoryComponentKind, MemoryComponentResidency, MemoryMode,
    MemoryProviderContract, MemoryResidentComponent, MemoryRunContext, MemorySafetyDecision,
    OffloadPolicy, WeightsSource,
};

use crate::PulidFluxPaths;

pub const PROVIDER_ID: &str = "pulid_flux";
pub const CALIBRATION_FINGERPRINT: &str =
    "pulid-flux-cuda-identity-stack-staged-decode-attention-block-window-v1";

fn base_spec(paths: &PulidFluxPaths) -> LoadSpec {
    let mut spec = LoadSpec::new(WeightsSource::Dir(paths.flux_base.clone()))
        .with_offload_policy(OffloadPolicy::Sequential)
        .with_load_shape(LoadShape::DeferredMaterialization);
    // Adapter facts must participate in FLUX admission. The reference backbone cannot safely use
    // transformer block streaming with additive adapters, so omitting this stack could admit an
    // adapted PuLID request to a rung that only fails later during load.
    spec.adapters = paths.adapters.clone();
    spec
}

fn base_spec_from_load_spec(spec: &LoadSpec) -> LoadSpec {
    let mut base = spec.clone();
    base.identity = None;
    base
}

fn resident_component(
    id: &str,
    path: &std::path::Path,
    require_float_source: bool,
) -> gen_core::Result<MemoryResidentComponent> {
    let resident_bytes = candle_gen::logical_weights::component_f32_bytes(
        path,
        require_float_source,
        |name, shape| {
            shape.len() == 2 && name.ends_with(".weight") && !name.ends_with("embedding.weight")
        },
    )?;
    if resident_bytes == 0 {
        return Err(gen_core::Error::Msg(format!(
            "{PROVIDER_ID}: resident identity component {id} has no loadable safetensors bytes at {}",
            path.display()
        )));
    }
    Ok(MemoryResidentComponent {
        id: id.to_owned(),
        kind: MemoryComponentKind::IdentityEncoder,
        resident_bytes,
        bounded_by: None,
        residency: MemoryComponentResidency::WholeRender,
    })
}

fn resident_components(paths: &PulidFluxPaths) -> gen_core::Result<Vec<MemoryResidentComponent>> {
    Ok(vec![
        resident_component("pulid_idformer_ca", &paths.pulid_weights, true)?,
        resident_component("pulid_eva_clip", &paths.eva_weights, true)?,
        resident_component(
            "pulid_face_scrfd",
            &paths.face_dir.join("scrfd_10g.safetensors"),
            false,
        )?,
        resident_component(
            "pulid_face_arcface",
            &paths.face_dir.join("arcface_iresnet100.safetensors"),
            false,
        )?,
        resident_component(
            "pulid_face_bisenet",
            &paths.face_dir.join("bisenet_parsing.safetensors"),
            false,
        )?,
    ])
}

/// Exact contract for the fully composed PuLID request route.
pub fn provider_contract(paths: &PulidFluxPaths) -> gen_core::Result<MemoryProviderContract> {
    candle_gen_flux::memory_strategy::reference_backbone_contract(
        PROVIDER_ID,
        &base_spec(paths),
        resident_components(paths)?,
        CALIBRATION_FINGERPRINT,
    )
}

pub fn resolved_numeric_tier(
    paths: &PulidFluxPaths,
) -> gen_core::Result<gen_core::MemoryNumericTier> {
    candle_gen_flux::memory_strategy::resolved_numeric_tier(&base_spec(paths), PROVIDER_ID)
}

/// The PuLID-FLUX memory profile (sc-27307): the shared pipeline, then the identity-route refusals
/// once it has admitted.
const MEMORY_PROFILE: gen_core::ProviderMemoryProfile = gen_core::ProviderMemoryProfile {
    tier: gen_core::MemoryProfileTier::Resolved(profile_tier),
    rejection: gen_core::MemoryRejectionRendering::ValidationError,
    fixtures: gen_core::MemoryProfileFixtures::routes(&[gen_core::MemoryFixtureRoute::new(
        gen_core::MemoryFixtureMode::Other("character_image"),
        1,
    )
    .with_overlay(gen_core::MemoryFixtureOverlay::Fixed("identity"))]),
    gates: gen_core::MemoryProfileGates::after(identity_route_gate),
    ..gen_core::ProviderMemoryProfile::STANDARD
};

fn profile_tier(
    spec: &LoadSpec,
    _contract: &MemoryProviderContract,
) -> gen_core::Result<gen_core::MemoryNumericTier> {
    candle_gen_flux::memory_strategy::resolved_numeric_tier(
        &base_spec_from_load_spec(spec),
        PROVIDER_ID,
    )
}

/// Weights-free contract surface specs carrying the mandatory PuLID identity slots.
pub fn memory_contract_surface_specs() -> Vec<gen_core::MemoryContractSurfaceSpec> {
    gen_core::candle_memory_contract_surface_specs()
        .into_iter()
        .map(|mut surface| {
            surface.spec.identity = Some(gen_core::IdentityWeights {
                encoder: Some(WeightsSource::File(
                    "/__sceneworks_memory_contract_pulid_weights__".into(),
                )),
                eva: Some(WeightsSource::File(
                    "/__sceneworks_memory_contract_pulid_eva__".into(),
                )),
                face_dir: Some(WeightsSource::Dir(
                    "/__sceneworks_memory_contract_pulid_face__".into(),
                )),
            });
            surface
        })
        .collect()
}

/// Registered [`LoadSpec`]-driven contract factory for `pulid_flux`.
///
/// Prices all 5 resident PuLID identity networks when materialized on disk; on weights-free
/// sentinel surfaces where none of the overlay paths exist, produces the exact structural contract
/// without reading weights.
pub fn registered_provider_contract(spec: &LoadSpec) -> gen_core::Result<MemoryProviderContract> {
    let base = base_spec_from_load_spec(spec);
    let components = match crate::registry::paths_from_spec(spec) {
        Ok(paths)
            if paths.pulid_weights.exists()
                || paths.eva_weights.exists()
                || paths.face_dir.exists() =>
        {
            resident_components(&paths)?
        }
        _ => Vec::new(),
    };
    candle_gen_flux::memory_strategy::reference_backbone_contract(
        PROVIDER_ID,
        &base,
        components,
        CALIBRATION_FINGERPRINT,
    )
}

pub(crate) fn registered_safety_check(
    spec: &LoadSpec,
    contract: &MemoryProviderContract,
    context: &MemoryRunContext,
) -> MemorySafetyDecision {
    gen_core::standard_registered_safety_check(&MEMORY_PROFILE, spec, contract, context)
}

pub(crate) fn registered_valid_fixture(
    spec: &LoadSpec,
    contract: &MemoryProviderContract,
    strategy: gen_core::MemoryStrategy,
) -> gen_core::Result<Vec<gen_core::MemoryBehaviorFixture>> {
    gen_core::standard_registered_valid_fixture(&MEMORY_PROFILE, spec, contract, strategy)
}

fn registered_scope(
    provider_id: &'static str,
    _spec: &LoadSpec,
    contract: &MemoryProviderContract,
    context: &MemoryRunContext,
    _loaded_tier: gen_core::MemoryNumericTier,
) -> gen_core::Result<Option<Box<dyn gen_core::MemoryRequestScope>>> {
    Ok(Some(
        candle_gen_flux::memory_strategy::reference_backbone_request_scope(
            provider_id,
            candle_core::Device::Cpu,
            contract,
            context,
        ),
    ))
}

pub(crate) fn registered_begin_request(
    spec: &LoadSpec,
    contract: &MemoryProviderContract,
    context: &MemoryRunContext,
) -> gen_core::Result<Option<Box<dyn gen_core::MemoryRequestScope>>> {
    gen_core::standard_begin_request(
        &MEMORY_PROFILE,
        PROVIDER_ID,
        spec,
        contract,
        context,
        registered_scope,
    )
}

pub(crate) const PULID_MEMORY_REGISTRATION: gen_core::MemoryRegistration =
    gen_core::MemoryRegistration {
        provider_id: PROVIDER_ID,
        contract: registered_provider_contract,
        safety_check: registered_safety_check,
    };

pub(crate) const PULID_MEMORY_BEHAVIOR: gen_core::MemoryBehaviorRegistration =
    gen_core::MemoryBehaviorRegistration {
        provider_id: PROVIDER_ID,
        valid_fixtures: registered_valid_fixture,
        begin_request: registered_begin_request,
    };

fn identity_route_gate(
    _contract: &MemoryProviderContract,
    context: &MemoryRunContext,
    _loaded_tier: gen_core::MemoryNumericTier,
) -> gen_core::Result<()> {
    if context.mode != MemoryMode::Other("character_image".to_owned())
        || !context.has_reference
        || context.geometry.batch != 1
        || context.geometry.reference_count != 1
        || context.overlay.as_deref() != Some("identity")
    {
        return Err(gen_core::Error::Msg(format!(
            "{PROVIDER_ID}: memory admission is bound to one character_image identity reference"
        )));
    }
    if context.has_phases {
        return Err(gen_core::Error::Msg(format!(
            "{PROVIDER_ID}: multi-phase denoise is not covered"
        )));
    }
    if context.use_pid {
        return Err(gen_core::Error::Msg(format!(
            "{PROVIDER_ID}: PiD has no admitted native-VAE memory ladder"
        )));
    }
    Ok(())
}

/// Provider-owned safety check for the bespoke identity route.
pub fn safety_check(
    paths: &PulidFluxPaths,
    contract: &MemoryProviderContract,
    context: &MemoryRunContext,
) -> MemorySafetyDecision {
    match resolved_numeric_tier(paths) {
        Ok(tier) => gen_core::standard_safety_check(&MEMORY_PROFILE, contract, context, tier),
        Err(error) => MemorySafetyDecision::rejected(error),
    }
}

pub fn validate_context(
    paths: &PulidFluxPaths,
    contract: &MemoryProviderContract,
    context: &MemoryRunContext,
) -> gen_core::Result<()> {
    candle_gen::request_scope::admit(safety_check(paths, contract, context))
}

/// Revalidate the context without filesystem reads after the exact contract has been retained by a
/// loaded provider. Path/tier validation already happened before materialization.
pub fn validate_context_from_loaded(
    contract: &MemoryProviderContract,
    context: &MemoryRunContext,
) -> gen_core::Result<()> {
    if contract.provider_id != PROVIDER_ID
        || context.mode != MemoryMode::Other("character_image".to_owned())
        || !context.has_reference
        || context.geometry.batch != 1
        || context.geometry.reference_count != 1
        || context.overlay.as_deref() != Some("identity")
        || context.has_phases
        || context.use_pid
    {
        return Err(gen_core::Error::Unsupported(format!(
            "{PROVIDER_ID}: loaded memory context no longer matches the admitted identity route"
        )));
    }
    contract.validate_selection(&context.selection)
}

#[cfg(test)]
mod tests {
    use super::*;
    use candle_gen::gen_core::{
        MemoryBehaviorRoute, MemoryStrategy, MemoryStrategySupport, TransformerComponent,
    };

    fn write_safetensors(path: &std::path::Path) {
        std::fs::create_dir_all(path.parent().unwrap()).unwrap();
        let mut header = br#"{"weight":{"dtype":"F32","shape":[1],"data_offsets":[0,4]}}"#.to_vec();
        while !header.len().is_multiple_of(8) {
            header.push(b' ');
        }
        let mut bytes = (header.len() as u64).to_le_bytes().to_vec();
        bytes.extend(header);
        bytes.extend([0_u8; 4]);
        std::fs::write(path, bytes).unwrap();
    }

    fn write_typed_safetensors(
        path: &std::path::Path,
        dtype: &str,
        elements: usize,
        stored_bytes: usize,
    ) {
        std::fs::create_dir_all(path.parent().unwrap()).unwrap();
        let mut header = format!(
            r#"{{"weight":{{"dtype":"{dtype}","shape":[{elements}],"data_offsets":[0,{stored_bytes}]}}}}"#
        )
        .into_bytes();
        while !header.len().is_multiple_of(8) {
            header.push(b' ');
        }
        let mut bytes = (header.len() as u64).to_le_bytes().to_vec();
        bytes.extend(header);
        bytes.extend(vec![0_u8; stored_bytes]);
        std::fs::write(path, bytes).unwrap();
    }

    fn paths(tmp: &tempfile::TempDir) -> PulidFluxPaths {
        let root = tmp.path().join("pulid-memory");
        for component in ["text_encoder", "text_encoder_2", "transformer", "vae"] {
            write_safetensors(&root.join("base").join(component).join("model.safetensors"));
        }
        std::fs::write(
            root.join("base/transformer/config.json"),
            r#"{"quantization":{"bits":4,"group_size":64}}"#,
        )
        .unwrap();
        for name in [
            "scrfd_10g.safetensors",
            "arcface_iresnet100.safetensors",
            "bisenet_parsing.safetensors",
        ] {
            write_safetensors(&root.join("face").join(name));
        }
        write_safetensors(&root.join("pulid.safetensors"));
        write_safetensors(&root.join("eva.safetensors"));
        PulidFluxPaths {
            flux_base: root.join("base"),
            pulid_weights: root.join("pulid.safetensors"),
            eva_weights: root.join("eva.safetensors"),
            face_dir: root.join("face"),
            adapters: Vec::new(),
        }
    }

    #[test]
    fn contract_is_full_but_has_a_distinct_identity_and_prices_every_resident_network() {
        let tmp = tempfile::tempdir().unwrap();
        let paths = paths(&tmp);
        let contract = provider_contract(&paths).unwrap();
        assert!(contract.conformance_errors().is_empty());
        assert_eq!(contract.provider_id, PROVIDER_ID);
        assert_eq!(
            contract.calibration.as_ref().unwrap().fingerprint,
            CALIBRATION_FINGERPRINT
        );
        assert_ne!(
            CALIBRATION_FINGERPRINT,
            candle_gen_flux::memory_strategy::CALIBRATION_FINGERPRINT
        );
        assert_eq!(contract.resident_components().len(), 5);
        assert_eq!(
            contract.asset_facts.overlay_bytes,
            contract
                .resident_components()
                .iter()
                .map(|component| component.resident_bytes)
                .sum::<u64>()
        );
        for strategy in MemoryStrategy::ALL {
            assert_eq!(
                contract.capability(strategy).unwrap().support,
                MemoryStrategySupport::Implemented
            );
        }
        assert_eq!(
            contract
                .capability(MemoryStrategy::BoundedTransformerResidency)
                .unwrap()
                .parameters
                .transformer_window_components,
            vec![TransformerComponent::Dit]
        );
    }

    #[test]
    fn adapted_route_disables_transformer_block_streaming_before_load() {
        let tmp = tempfile::tempdir().unwrap();
        let mut paths = paths(&tmp);
        paths.adapters.push(candle_gen::gen_core::AdapterSpec::new(
            tmp.path().join("identity-style.safetensors"),
            1.0,
            candle_gen::gen_core::AdapterKind::Lora,
        ));

        let spec = base_spec(&paths);
        assert_eq!(
            spec.adapters.len(),
            1,
            "the composed PuLID route must retain its stack"
        );
        assert_eq!(spec.adapters[0].path, paths.adapters[0].path);
        let contract = provider_contract(&paths).unwrap();
        assert_eq!(
            contract
                .capability(MemoryStrategy::BoundedTransformerResidency)
                .unwrap()
                .support,
            MemoryStrategySupport::Missing,
            "adapted FLUX reference backbones require resident transformer blocks"
        );
        assert!(contract
            .capability(MemoryStrategy::StagedResidency)
            .is_some_and(|capability| capability.support == MemoryStrategySupport::Implemented));
    }

    #[test]
    fn exact_identity_route_is_accepted_but_base_overlay_reuse_fails_closed() {
        let tmp = tempfile::tempdir().unwrap();
        let paths = paths(&tmp);
        let contract = provider_contract(&paths).unwrap();
        let tier = resolved_numeric_tier(&paths).unwrap();
        let mut context = gen_core::standard_memory_behavior_context(
            &contract,
            MemoryStrategy::BoundedTransformerResidency,
            tier,
            MemoryBehaviorRoute {
                mode: MemoryMode::Other("character_image".to_owned()),
                reference_count: 1,
                use_pid: false,
                has_phases: false,
                overlay: Some("identity".to_owned()),
            },
        )
        .unwrap();
        assert_eq!(
            safety_check(&paths, &contract, &context),
            MemorySafetyDecision::Accept
        );
        context.overlay = None;
        assert!(matches!(
            safety_check(&paths, &contract, &context),
            MemorySafetyDecision::Reject { .. }
        ));
    }

    #[test]
    fn pid_is_rejected_for_every_strategy_including_resident() {
        let tmp = tempfile::tempdir().unwrap();
        let paths = paths(&tmp);
        let contract = provider_contract(&paths).unwrap();
        let tier = resolved_numeric_tier(&paths).unwrap();
        for strategy in MemoryStrategy::ALL {
            let mut context = gen_core::standard_memory_behavior_context(
                &contract,
                strategy,
                tier,
                MemoryBehaviorRoute {
                    mode: MemoryMode::Other("character_image".to_owned()),
                    reference_count: 1,
                    use_pid: true,
                    has_phases: false,
                    overlay: Some("identity".to_owned()),
                },
            )
            .unwrap();
            assert!(matches!(
                safety_check(&paths, &contract, &context),
                MemorySafetyDecision::Reject { .. }
            ));
            assert!(validate_context_from_loaded(&contract, &context).is_err());

            // The same exact route without PiD remains admitted at every implemented rung.
            context.use_pid = false;
            assert_eq!(
                safety_check(&paths, &contract, &context),
                MemorySafetyDecision::Accept
            );
            assert!(validate_context_from_loaded(&contract, &context).is_ok());
        }
    }

    #[test]
    fn batch_must_be_exactly_one_at_admission_and_loaded_context_validation() {
        let tmp = tempfile::tempdir().unwrap();
        let paths = paths(&tmp);
        let contract = provider_contract(&paths).unwrap();
        let tier = resolved_numeric_tier(&paths).unwrap();
        let mut context = gen_core::standard_memory_behavior_context(
            &contract,
            MemoryStrategy::Resident,
            tier,
            MemoryBehaviorRoute {
                mode: MemoryMode::Other("character_image".to_owned()),
                reference_count: 1,
                use_pid: false,
                has_phases: false,
                overlay: Some("identity".to_owned()),
            },
        )
        .unwrap();

        for batch in [0, 2] {
            context.geometry.batch = batch;
            assert!(matches!(
                safety_check(&paths, &contract, &context),
                MemorySafetyDecision::Reject { .. }
            ));
            assert!(validate_context_from_loaded(&contract, &context).is_err());
        }
        context.geometry.batch = 1;
        assert_eq!(
            safety_check(&paths, &contract, &context),
            MemorySafetyDecision::Accept
        );
        assert!(validate_context_from_loaded(&contract, &context).is_ok());
    }

    #[test]
    fn resident_accounting_prices_loaded_f32_tensors_not_serialized_file_bytes() {
        let root_tmp = tempfile::tempdir().unwrap();
        let root = root_tmp.path().to_path_buf();
        for (name, dtype, elements, stored_bytes, expected_resident) in [
            ("bf16.safetensors", "BF16", 3, 6, 12),
            ("f16.safetensors", "F16", 5, 10, 20),
            ("f32.safetensors", "F32", 7, 28, 28),
            // Rank-one FP8 coefficients use the dense F32 operator even on a native-GEMM device.
            ("fp8.safetensors", "F8_E5M2", 7, 7, 28),
        ] {
            let path = root.join(name);
            write_typed_safetensors(&path, dtype, elements, stored_bytes);
            let component = resident_component(name, &path, true).unwrap();
            assert_eq!(
                component.resident_bytes, expected_resident,
                "{dtype} must be priced at its actual F32 materialization"
            );
        }
    }

    #[test]
    fn pulid_and_eva_non_float_tensors_fail_closed_instead_of_being_underpriced() {
        let root_tmp = tempfile::tempdir().unwrap();
        let root = root_tmp.path().to_path_buf();
        let path = root.join("i64.safetensors");
        write_typed_safetensors(&path, "I64", 3, 24);

        let error = resident_component("pulid_idformer_ca", &path, true)
            .unwrap_err()
            .to_string();
        assert!(error.contains("requires floating weights"), "{error}");

        // The face loader differs intentionally: it casts every tensor, including integer sources,
        // so the same three elements really do occupy three F32 values after load.
        assert_eq!(
            resident_component("pulid_face_scrfd", &path, false)
                .unwrap()
                .resident_bytes,
            12
        );
    }
}
