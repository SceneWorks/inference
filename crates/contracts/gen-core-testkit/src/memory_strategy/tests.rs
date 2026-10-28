use super::*;
use gen_core::{
    standard_memory_strategy_safety_check, MemoryBackendRealization, MemoryBehaviorBeginRequest,
    MemoryCacheState, MemoryCalibrationIdentity, MemoryDecodeRouteDomain, MemoryGeometry,
    MemoryLifecycleCapabilities, MemoryMode, MemoryNumericTier, MemoryOptimizationAuthority,
    MemoryParameterRanges, MemoryPidDecodeRoutes, MemoryRequestScope, MemoryRunOutcome,
    MemorySelection, MemoryStrategyCapability, WeightsSource,
};

fn route_context(
    calibration_abi: u32,
    calibration_fingerprint: &str,
    load_shape: gen_core::LoadShape,
    edge: u32,
    overlap: u32,
    use_pid: bool,
) -> MemoryRunContext {
    MemoryRunContext {
        reference_resources: None,
        selection: MemorySelection {
            strategy: MemoryStrategy::BoundedDecode,
            parameters: MemoryStrategyParameters {
                decode_tile_edge: Some(edge),
                decode_overlap: Some(overlap),
                ..Default::default()
            },
            tier: MemoryNumericTier {
                precision: Precision::Bf16,
                quant: None,
                component_precision_floors: &[],
            },
        },
        optimization_authority: MemoryOptimizationAuthority::Calibrated,
        calibration_abi,
        calibration_fingerprint: calibration_fingerprint.to_owned(),
        load_shape,
        mode: MemoryMode::TextToImage,
        has_reference: false,
        use_pid,
        has_phases: true,
        geometry: MemoryGeometry {
            width: 1024,
            height: 1024,
            batch: 1,
            frames: 1,
            reference_count: 0,
        },
        overlay: None,
        budget: MemoryBudget {
            total_bytes: 8 * 1024 * 1024 * 1024,
            committed_bytes: 0,
            reclaimable_bytes: 0,
            reserved_headroom_bytes: 0,
        },
        predicted_peak_bytes: 1024 * 1024 * 1024,
        cache_state: MemoryCacheState::Cold,
        evidence_revision: "weights-free-registry-conformance".to_owned(),
    }
}

fn backend() -> MemoryBackendRealization {
    MemoryBackendRealization::MlxMetal {
        bounded_wired_residency: true,
        lazy_or_mmap_materialization: true,
        explicit_evaluation_and_synchronization: true,
        cache_eviction: true,
    }
}

/// Honest baseline for the facts check: a decomposable base total, three distinct component
/// totals, a lifecycle-phase formula, and one declared architecture axis.
fn facts_contract() -> MemoryProviderContract {
    let mut contract = MemoryProviderContract::compatibility_default("facts", backend());
    contract.asset_facts = gen_core::MemoryAssetFacts {
        base_bytes: 60,
        conditioning_bytes: 10,
        transformer_bytes: 20,
        decoder_bytes: 30,
        overlay_bytes: 0,
    };
    contract.formula = gen_core::MemoryFormulaKind::PhaseEnvelope {
        phases: vec![MemoryPhase::Denoise],
        variables: vec![gen_core::MemoryFormulaVariable::AssetBytes],
    };
    contract.architecture_facts = gen_core::MemoryArchitectureFacts {
        attention_heads: Some(30),
        head_dim: Some(128),
        transformer_blocks: Some(30),
        patch_size: Some(2),
        latent_channels: Some(16),
        vae_spatial_scale: Some(8),
        vae_temporal_scale: None,
        activation_dtype_width: Some(2),
    };
    contract
}

#[test]
fn honest_contract_facts_conform() {
    let contract = facts_contract();
    check_memory_contract_facts(&contract).unwrap();
    assert_memory_contract_facts_conform(&contract);
}

#[test]
fn base_bytes_that_is_not_its_own_decomposition_is_rejected() {
    let mut contract = facts_contract();
    // The classic dishonest shape: an overlay folded into the total but into no component.
    contract.asset_facts.base_bytes += 5;
    let errors = check_memory_contract_facts(&contract).unwrap_err();
    assert!(
        errors
            .iter()
            .any(|error| error.contains("must equal conditioning")),
        "{errors:?}"
    );
}

#[test]
fn one_total_repeated_in_two_component_fields_is_rejected() {
    let mut contract = facts_contract();
    // Decoder bytes borrowed for the conditioning phase: the sum still adds up, the
    // decomposition is still a lie.
    contract.asset_facts.conditioning_bytes = 30;
    contract.asset_facts.base_bytes = 30 + 20 + 30;
    let errors = check_memory_contract_facts(&contract).unwrap_err();
    assert!(
        errors.iter().any(|error| error
            .contains("asset_facts.conditioning_bytes and asset_facts.decoder_bytes repeat")),
        "{errors:?}"
    );
}

#[test]
fn a_zero_component_pair_is_not_treated_as_a_repeat() {
    let mut contract = facts_contract();
    contract.asset_facts.conditioning_bytes = 0;
    contract.asset_facts.decoder_bytes = 0;
    contract.asset_facts.base_bytes = contract.asset_facts.transformer_bytes;
    check_memory_contract_facts(&contract).unwrap();
}

#[test]
fn all_absent_architecture_facts_on_a_phase_provider_are_rejected() {
    for formula in [
        gen_core::MemoryFormulaKind::PhaseEnvelope {
            phases: vec![MemoryPhase::Denoise],
            variables: vec![gen_core::MemoryFormulaVariable::AssetBytes],
        },
        gen_core::MemoryFormulaKind::ComponentPhaseEnvelope {
            phases: vec![MemoryPhase::Denoise],
            variables: vec![gen_core::MemoryFormulaVariable::AssetBytes],
            resident_components: Vec::new(),
        },
    ] {
        let mut contract = facts_contract();
        contract.formula = formula;
        contract.architecture_facts = gen_core::MemoryArchitectureFacts::default();
        let errors = check_memory_contract_facts(&contract).unwrap_err();
        assert!(
            errors.iter().any(|error| error
                .contains("must declare at least one config-derived architecture fact")),
            "{errors:?}"
        );
    }
}

/// The E2 gate must not be satisfiable by a compile-time constant. `activation_dtype_width` is
/// emitted from the provider's pinned dtype whether or not a single component `config.json` was
/// found, so a contract carrying only that axis has read nothing and must still be rejected.
#[test]
fn activation_dtype_width_alone_does_not_satisfy_the_architecture_gate() {
    let mut contract = facts_contract();
    contract.formula = gen_core::MemoryFormulaKind::PhaseEnvelope {
        phases: vec![MemoryPhase::Denoise],
        variables: vec![gen_core::MemoryFormulaVariable::AssetBytes],
    };
    contract.architecture_facts = gen_core::MemoryArchitectureFacts {
        activation_dtype_width: Some(2),
        ..Default::default()
    };
    assert!(!contract.architecture_facts.is_empty());
    assert!(!contract.architecture_facts.has_declared_architecture_axis());
    let errors = check_memory_contract_facts(&contract).unwrap_err();
    assert!(
        errors
            .iter()
            .any(|error| error.contains("must declare at least one config-derived")),
        "{errors:?}"
    );
    // One genuinely config-derived axis alongside it is accepted.
    contract.architecture_facts.transformer_blocks = Some(30);
    check_memory_contract_facts(&contract).unwrap();
}

// ---- the backend-keyed weights-free architecture rule (SC-22661 reconciliation) ------------

const CANDLE_BACKEND: MemoryBackendRealization = MemoryBackendRealization::CandleCuda {
    device_residency: true,
    host_backed_weights: false,
    host_to_device_block_materialization: true,
    block_materialization: gen_core::MemoryWindowMaterialization::DeviceFormatTransfer,
};

/// One config-derived axis set, the shape either backend publishes once geometry is known.
fn declared_axes() -> gen_core::MemoryArchitectureFacts {
    gen_core::MemoryArchitectureFacts {
        transformer_blocks: Some(30),
        activation_dtype_width: Some(2),
        ..Default::default()
    }
}

/// `activation_dtype_width` only — the crate-wide constant that must satisfy neither arm.
fn dtype_only_axes() -> gen_core::MemoryArchitectureFacts {
    gen_core::MemoryArchitectureFacts {
        activation_dtype_width: Some(2),
        ..Default::default()
    }
}

fn surface_contract(
    provider_id: &str,
    backend: MemoryBackendRealization,
    facts: gen_core::MemoryArchitectureFacts,
) -> gen_core::Result<MemoryProviderContract> {
    let mut contract = MemoryProviderContract::compatibility_default(provider_id, backend);
    contract.architecture_facts = facts;
    Ok(contract)
}

/// The honest Candle shape: axes appear only once the snapshot root exists on disk, exactly
/// like `candle_gen::architecture_facts::snapshot_root` gates them.
fn candle_honest(spec: &LoadSpec) -> gen_core::Result<MemoryProviderContract> {
    let materialized = matches!(&spec.weights, WeightsSource::Dir(root) if root.is_dir());
    surface_contract(
        "candle_route",
        CANDLE_BACKEND,
        if materialized {
            declared_axes()
        } else {
            Default::default()
        },
    )
}

/// The Candle defect the weights-free arm exists to catch: geometry with nothing to read it
/// from, so it came from the provider id.
fn candle_fabricating(spec: &LoadSpec) -> gen_core::Result<MemoryProviderContract> {
    let _ = spec;
    surface_contract("candle_route", CANDLE_BACKEND, declared_axes())
}

/// The Candle defect only the materialized arm can catch: `default()` unconditionally, which
/// the weights-free walk cannot distinguish from honesty.
fn candle_never_derives(spec: &LoadSpec) -> gen_core::Result<MemoryProviderContract> {
    let _ = spec;
    surface_contract("candle_route", CANDLE_BACKEND, Default::default())
}

/// A Candle route that publishes only the crate-wide dtype constant on a materialized root has
/// still derived no geometry.
fn candle_dtype_only(spec: &LoadSpec) -> gen_core::Result<MemoryProviderContract> {
    let _ = spec;
    surface_contract("candle_route", CANDLE_BACKEND, dtype_only_axes())
}

/// The honest MLX shape: preset constants exist before any snapshot does.
fn mlx_preset(spec: &LoadSpec) -> gen_core::Result<MemoryProviderContract> {
    let _ = spec;
    surface_contract("mlx_route", backend(), declared_axes())
}

/// The MLX defect: a weights-free surface withholding geometry the crate already holds.
fn mlx_silent(spec: &LoadSpec) -> gen_core::Result<MemoryProviderContract> {
    let _ = spec;
    surface_contract("mlx_route", backend(), Default::default())
}

/// ...and the constant that must not be mistaken for it.
fn mlx_dtype_only(spec: &LoadSpec) -> gen_core::Result<MemoryProviderContract> {
    let _ = spec;
    surface_contract("mlx_route", backend(), dtype_only_axes())
}

fn registry_of(
    provider_id: &'static str,
    contract: fn(&LoadSpec) -> gen_core::Result<MemoryProviderContract>,
) -> ProviderRegistry {
    gen_core::ProviderRegistryBuilder::new()
        .register_composed_memory_strategy(MemoryRegistration {
            provider_id,
            contract,
            safety_check: gen_core::default_registered_memory_strategy_safety_check,
        })
        .register_memory_contract_fixture(MemoryContractFixtureRegistration {
            provider_id,
            contract,
            surface_specs: gen_core::mlx_memory_contract_surface_specs,
        })
        .build()
        .expect("a single paired memory route builds")
}

/// AC (SC-22661 reconciliation, arm 1): a **Candle** weights-free surface must publish
/// `MemoryArchitectureFacts::default()`. Its axes are gated on a materialized snapshot root the
/// registry deliberately does not create, so an axis here was inferred from the provider id.
#[test]
fn a_candle_weights_free_surface_may_not_declare_an_architecture_axis() {
    let honest = registry_of("candle_route", candle_honest);
    let coverage = memory_contract_surface_registry_facts_conformance(&honest, None);
    assert!(
        coverage.surfaces_checked > 0,
        "the walk had nothing to check"
    );
    assert_eq!(coverage.materialized_providers_checked, 0);

    let errors = check_memory_contract_surface_registry_facts(
        &registry_of("candle_route", candle_fabricating),
        None,
    )
    .unwrap_err();
    assert!(
        errors
            .iter()
            .any(|error| error.contains("Candle weights-free contract surface must publish")),
        "{errors:?}"
    );
    // The crate-wide dtype constant is not an axis, so publishing only it is still honest here.
    check_memory_contract_surface_registry_facts(
        &registry_of("candle_route", candle_dtype_only),
        None,
    )
    .unwrap();
}

/// AC (SC-22661 reconciliation, arm 2): an **MLX** weights-free surface must declare at least
/// one axis. MLX geometry is mirrored from compile-time presets that exist before a snapshot
/// does, so declaring nothing withholds facts the crate already holds.
#[test]
fn an_mlx_weights_free_surface_must_declare_an_architecture_axis() {
    memory_contract_surface_registry_facts_conformance(&registry_of("mlx_route", mlx_preset), None);

    for (label, contract) in [
        ("nothing at all", mlx_silent as fn(&LoadSpec) -> _),
        ("only the dtype constant", mlx_dtype_only),
    ] {
        let errors =
            check_memory_contract_surface_registry_facts(&registry_of("mlx_route", contract), None)
                .unwrap_err();
        assert!(
            errors.iter().any(|error| error
                .contains("MLX weights-free contract surface must declare at least one")),
            "{label}: {errors:?}"
        );
    }
}

/// AC (SC-22661): the materialized arm is what makes the registry walk non-vacuous for E2. A
/// provider that returns `MemoryArchitectureFacts::default()` unconditionally passes every
/// weights-free assertion and fails only here.
#[test]
fn a_provider_that_never_derives_an_axis_fails_only_the_materialized_arm() {
    let root = tempfile::tempdir().unwrap();
    let path = root.path().to_path_buf();
    let lookup = |_: &str| Some(path.clone());

    // Honest: the same root yields geometry, and the walk records that it checked it.
    let coverage = memory_contract_surface_registry_facts_conformance(
        &registry_of("candle_route", candle_honest),
        Some(&lookup),
    );
    assert_eq!(coverage.materialized_providers_checked, 1);

    // The `::default()` mutation: invisible weights-free, rejected here.
    let never = registry_of("candle_route", candle_never_derives);
    check_memory_contract_surface_registry_facts(&never, None)
        .expect("the weights-free walk cannot see this defect");
    let errors = check_memory_contract_surface_registry_facts(&never, Some(&lookup)).unwrap_err();
    assert!(
        errors
            .iter()
            .any(|error| error.contains("still declares no architecture axis")),
        "{errors:?}"
    );

    // A provider the lookup declines is checked on the weights-free arm only.
    let none = |_: &str| None;
    let coverage = memory_contract_surface_registry_facts_conformance(&never, Some(&none));
    assert_eq!(coverage.materialized_providers_checked, 0);
}

/// A component sum that overflows `u64` must be reported, not saturated into agreement with a
/// `base_bytes` of `u64::MAX`.
#[test]
fn an_overflowing_base_component_sum_is_rejected_rather_than_saturated() {
    let mut contract = facts_contract();
    contract.asset_facts.conditioning_bytes = u64::MAX;
    contract.asset_facts.transformer_bytes = 1;
    contract.asset_facts.decoder_bytes = 2;
    contract.asset_facts.base_bytes = u64::MAX;
    let errors = check_memory_contract_facts(&contract).unwrap_err();
    assert!(
        errors
            .iter()
            .any(|error| error == "base component byte sum overflow"),
        "{errors:?}"
    );
}

/// The E1 half stands alone for the weights-free path, and the E2 gate is *not* applied there.
#[test]
fn the_asset_facts_entry_point_checks_e1_without_the_architecture_gate() {
    let mut contract = facts_contract();
    contract.architecture_facts = gen_core::MemoryArchitectureFacts::default();
    check_memory_contract_asset_facts(&contract).unwrap();
    assert_memory_contract_asset_facts_conform(&contract);
    // ...and it still rejects a dishonest decomposition.
    contract.asset_facts.base_bytes += 5;
    let errors = check_memory_contract_asset_facts(&contract).unwrap_err();
    assert!(
        errors
            .iter()
            .any(|error| error.contains("must equal conditioning")),
        "{errors:?}"
    );
}

#[test]
#[should_panic(expected = "memory-contract asset-facts conformance FAILED for 'facts'")]
fn the_panicking_asset_facts_entry_point_names_the_provider() {
    let mut contract = facts_contract();
    contract.asset_facts.base_bytes += 1;
    assert_memory_contract_asset_facts_conform(&contract);
}

#[test]
fn a_provider_without_a_phase_formula_may_still_declare_no_architecture_facts() {
    let mut contract = facts_contract();
    contract.formula = gen_core::MemoryFormulaKind::AssetBytesPlusHeadroom;
    contract.architecture_facts = gen_core::MemoryArchitectureFacts::default();
    check_memory_contract_facts(&contract).unwrap();
}

#[test]
fn a_zero_valued_architecture_axis_is_rejected() {
    let mut contract = facts_contract();
    contract.architecture_facts.vae_temporal_scale = Some(0);
    let errors = check_memory_contract_facts(&contract).unwrap_err();
    assert!(
        errors
            .iter()
            .any(|error| error.contains("architecture_facts.vae_temporal_scale is Some(0)")),
        "{errors:?}"
    );
}

#[test]
#[should_panic(expected = "memory-contract facts conformance FAILED for 'facts'")]
fn the_panicking_entry_point_names_the_provider() {
    let mut contract = facts_contract();
    contract.asset_facts.base_bytes += 1;
    assert_memory_contract_facts_conform(&contract);
}

#[test]
fn resident_only_compatibility_contract_conforms_without_claiming_optimization() {
    let contract = MemoryProviderContract::compatibility_default("legacy", backend());
    check_memory_strategy_contract(&contract).unwrap();
    assert!(contract.calibration.is_none());
}

#[test]
fn malformed_strategy_table_is_reported() {
    let mut contract = MemoryProviderContract::compatibility_default("bad", backend());
    contract.strategies.pop();
    let errors = check_memory_strategy_contract(&contract).unwrap_err();
    assert!(errors.iter().any(|error| error.contains("exactly once")));
}

fn eager_only_contract(spec: &LoadSpec) -> gen_core::Result<MemoryProviderContract> {
    if spec.load_shape != gen_core::LoadShape::EagerMaterialization {
        return Err(gen_core::Error::Unsupported("eager only".into()));
    }
    let mut contract = MemoryProviderContract::compatibility_default("eager-only", backend());
    contract.load_shape = spec.load_shape;
    Ok(contract)
}

fn always_reject_safety(
    _spec: &LoadSpec,
    _contract: &MemoryProviderContract,
    _context: &MemoryRunContext,
) -> MemorySafetyDecision {
    MemorySafetyDecision::Reject {
        reason: "no optimized strategies".into(),
    }
}

#[test]
fn mixed_catalog_conformance_uses_each_providers_valid_load_shape() {
    let registration = MemoryRegistration {
        provider_id: "eager-only",
        contract: eager_only_contract,
        safety_check: always_reject_safety,
    };
    let spec = LoadSpec::new(WeightsSource::Dir("/nonexistent".into()))
        .with_load_shape(gen_core::LoadShape::DeferredMaterialization);
    let mut errors = Vec::new();

    check_memory_registration(&registration, None, None, None, &spec, &mut errors);

    assert_eq!(errors, Vec::<String>::new());
}

fn pid_contract(_spec: &LoadSpec) -> gen_core::Result<MemoryProviderContract> {
    let mut contract = MemoryProviderContract::compatibility_default("pid-provider", backend());
    let bounded = contract
        .strategies
        .iter_mut()
        .find(|capability| capability.strategy == MemoryStrategy::BoundedDecode)
        .unwrap();
    *bounded = MemoryStrategyCapability {
        strategy: MemoryStrategy::BoundedDecode,
        support: MemoryStrategySupport::Implemented,
        parameters: MemoryParameterRanges {
            decode_tile_edges: vec![2048, 512],
            decode_overlaps: vec![256, 64],
            ..Default::default()
        },
    };
    contract.pid_decode_routes = Some(MemoryPidDecodeRoutes {
        native: MemoryDecodeRouteDomain {
            tile_edges: vec![512],
            tile_overlap: 64,
        },
        pid: MemoryDecodeRouteDomain {
            tile_edges: vec![2048],
            tile_overlap: 256,
        },
    });
    contract.lifecycle = MemoryLifecycleCapabilities {
        decode_tiling: true,
        ..Default::default()
    };
    contract.calibration = Some(MemoryCalibrationIdentity::new(
        "pid-provider-v1",
        contract.load_shape,
    ));
    Ok(contract)
}

fn route_aware_safety(
    _spec: &LoadSpec,
    contract: &MemoryProviderContract,
    context: &MemoryRunContext,
) -> MemorySafetyDecision {
    let routes = contract.pid_decode_routes.as_ref().unwrap();
    let edge = context.selection.parameters.decode_tile_edge;
    let overlap = context.selection.parameters.decode_overlap;
    let domain = if context.use_pid {
        &routes.pid
    } else {
        &routes.native
    };
    if edge.is_some_and(|edge| domain.tile_edges.contains(&edge))
        && overlap == Some(domain.tile_overlap)
    {
        standard_memory_strategy_safety_check(
            contract,
            context,
            Some(MemoryNumericTier {
                precision: Precision::Bf16,
                quant: None,
                component_precision_floors: &[],
            }),
            None,
        )
    } else {
        MemorySafetyDecision::Reject {
            reason: "cross-route geometry".to_owned(),
        }
    }
}

struct FixtureScope {
    memory: Option<gen_core::GenerationMemory>,
    finished: bool,
}

impl MemoryRequestScope for FixtureScope {
    fn configure_request(
        &mut self,
        request: &mut gen_core::GenerationRequest,
    ) -> gen_core::Result<()> {
        if self.finished {
            return Err(gen_core::Error::Msg("finished".into()));
        }
        request.memory = self.memory;
        Ok(())
    }
    fn enter_phase(&mut self, _phase: MemoryPhase) -> gen_core::Result<()> {
        if self.finished {
            Err(gen_core::Error::Msg("finished".into()))
        } else {
            Ok(())
        }
    }
    fn leave_phase(&mut self, phase: MemoryPhase) -> gen_core::Result<()> {
        self.enter_phase(phase)
    }
    fn configure_decode(
        &mut self,
        _edge: u32,
        _overlap: u32,
        _geometry: MemoryGeometry,
    ) -> gen_core::Result<()> {
        self.enter_phase(MemoryPhase::Decode)
    }
    fn configure_attention(&mut self, _chunk: u32) -> gen_core::Result<()> {
        self.enter_phase(MemoryPhase::Denoise)
    }
    fn materialize_transformer_window(&mut self, _first: u32, _count: u32) -> gen_core::Result<()> {
        self.enter_phase(MemoryPhase::Denoise)
    }
    fn finish(&mut self, _outcome: MemoryRunOutcome) -> gen_core::Result<()> {
        if self.finished {
            Err(gen_core::Error::Msg("finished".into()))
        } else {
            self.finished = true;
            Ok(())
        }
    }
}

fn pid_fixture(
    _spec: &LoadSpec,
    contract: &MemoryProviderContract,
    strategy: MemoryStrategy,
) -> gen_core::Result<Vec<gen_core::MemoryBehaviorFixture>> {
    if strategy != MemoryStrategy::BoundedDecode {
        return Ok(Vec::new());
    }
    let routes = contract.pid_decode_routes.as_ref().unwrap();
    let calibration = contract.calibration.as_ref().unwrap();
    Ok(vec![
        gen_core::MemoryBehaviorFixture::new(route_context(
            calibration.abi,
            &calibration.fingerprint,
            calibration.load_shape,
            routes.native.tile_edges[0],
            routes.native.tile_overlap,
            false,
        )),
        gen_core::MemoryBehaviorFixture::new(route_context(
            calibration.abi,
            &calibration.fingerprint,
            calibration.load_shape,
            routes.pid.tile_edges[0],
            routes.pid.tile_overlap,
            true,
        )),
    ])
}

fn pid_begin(
    _spec: &LoadSpec,
    contract: &MemoryProviderContract,
    context: &MemoryRunContext,
) -> gen_core::Result<Option<Box<dyn MemoryRequestScope>>> {
    Ok(Some(Box::new(FixtureScope {
        memory: contract.generation_memory(&context.selection),
        finished: false,
    })))
}

const PID_BEHAVIOR: MemoryBehaviorRegistration = MemoryBehaviorRegistration {
    provider_id: "pid-provider",
    valid_fixtures: pid_fixture,
    begin_request: pid_begin,
};

#[test]
fn route_aware_registration_accepts_matching_routes_and_rejects_cross_routes() {
    let registration = MemoryRegistration {
        provider_id: "pid-provider",
        contract: pid_contract,
        safety_check: route_aware_safety,
    };
    let mut errors = Vec::new();
    check_memory_registration(
        &registration,
        None,
        None,
        Some(&PID_BEHAVIOR),
        &LoadSpec::new(WeightsSource::Dir("/nonexistent".into())),
        &mut errors,
    );
    assert_eq!(errors, Vec::<String>::new());
}

#[test]
fn conformance_uses_the_explicit_weights_free_factory() {
    fn production_requires_assets(_spec: &LoadSpec) -> gen_core::Result<MemoryProviderContract> {
        Err(gen_core::Error::Msg(
            "production contract touched required assets".to_owned(),
        ))
    }

    let registration = MemoryRegistration {
        provider_id: "pid-provider",
        contract: production_requires_assets,
        safety_check: route_aware_safety,
    };
    let contract_fixture = MemoryContractFixtureRegistration {
        provider_id: "pid-provider",
        contract: pid_contract,
        surface_specs: gen_core::mlx_memory_contract_surface_specs,
    };
    let mut errors = Vec::new();
    check_memory_registration(
        &registration,
        Some(&contract_fixture),
        None,
        Some(&PID_BEHAVIOR),
        &LoadSpec::new(WeightsSource::Dir("/nonexistent".into())),
        &mut errors,
    );
    assert_eq!(errors, Vec::<String>::new());
}

#[test]
fn conformance_uses_the_explicit_resident_only_witness_factory() {
    fn production_requires_assets(_spec: &LoadSpec) -> gen_core::Result<MemoryProviderContract> {
        Err(gen_core::Error::Msg(
            "production contract touched required assets".to_owned(),
        ))
    }

    let registration = MemoryRegistration {
        provider_id: "eager-only",
        contract: production_requires_assets,
        safety_check: always_reject_safety,
    };
    let witness = ResidentOnlyMemoryContractRegistration {
        provider_id: "eager-only",
        contract: eager_only_contract,
        surface_specs: gen_core::mlx_memory_contract_surface_specs,
    };
    let spec = LoadSpec::new(WeightsSource::Dir("/nonexistent".into()))
        .with_load_shape(gen_core::LoadShape::DeferredMaterialization);
    let mut errors = Vec::new();
    check_memory_registration(
        &registration,
        None,
        Some(&witness),
        None,
        &spec,
        &mut errors,
    );
    assert_eq!(errors, Vec::<String>::new());
}

#[test]
fn route_blind_safety_fails_both_cross_route_probes() {
    let registration = MemoryRegistration {
        provider_id: "pid-provider",
        contract: pid_contract,
        safety_check: |_spec, contract, context| {
            standard_memory_strategy_safety_check(
                contract,
                context,
                Some(MemoryNumericTier {
                    precision: Precision::Bf16,
                    quant: None,
                    component_precision_floors: &[],
                }),
                None,
            )
        },
    };
    let mut errors = Vec::new();
    check_memory_registration(
        &registration,
        None,
        None,
        Some(&PID_BEHAVIOR),
        &LoadSpec::new(WeightsSource::Dir("/nonexistent".into())),
        &mut errors,
    );
    assert_eq!(errors.len(), 6, "{errors:#?}");
    assert!(errors
        .iter()
        .any(|error| error.contains("edge 512 + overlap 256 on native route")));
    assert!(errors
        .iter()
        .any(|error| error.contains("edge 2048 + overlap 64 on PiD route")));
}

fn always_accept(
    _spec: &LoadSpec,
    _contract: &MemoryProviderContract,
    _context: &MemoryRunContext,
) -> MemorySafetyDecision {
    MemorySafetyDecision::Accept
}

fn always_reject(
    _spec: &LoadSpec,
    _contract: &MemoryProviderContract,
    _context: &MemoryRunContext,
) -> MemorySafetyDecision {
    MemorySafetyDecision::Reject {
        reason: "always".into(),
    }
}

fn handshake_blind(
    _spec: &LoadSpec,
    contract: &MemoryProviderContract,
    context: &MemoryRunContext,
) -> MemorySafetyDecision {
    let mut trusted = context.clone();
    let calibration = contract.calibration.as_ref().unwrap();
    trusted.calibration_abi = calibration.abi;
    trusted.calibration_fingerprint = calibration.fingerprint.clone();
    route_aware_safety(
        &LoadSpec::new(WeightsSource::Dir("/nonexistent".into())),
        contract,
        &trusted,
    )
}

fn tier_blind(
    _spec: &LoadSpec,
    contract: &MemoryProviderContract,
    context: &MemoryRunContext,
) -> MemorySafetyDecision {
    let routes = contract.pid_decode_routes.as_ref().unwrap();
    let domain = if context.use_pid {
        &routes.pid
    } else {
        &routes.native
    };
    if context
        .selection
        .parameters
        .decode_tile_edge
        .is_some_and(|edge| domain.tile_edges.contains(&edge))
        && context.selection.parameters.decode_overlap == Some(domain.tile_overlap)
    {
        gen_core::default_memory_strategy_safety_check(contract, context)
    } else {
        MemorySafetyDecision::Reject {
            reason: "route".into(),
        }
    }
}

fn budget_blind(
    spec: &LoadSpec,
    contract: &MemoryProviderContract,
    context: &MemoryRunContext,
) -> MemorySafetyDecision {
    let mut admitted = context.clone();
    admitted.predicted_peak_bytes = 0;
    admitted.budget.total_bytes = u64::MAX;
    route_aware_safety(spec, contract, &admitted)
}

fn overlap_blind(
    spec: &LoadSpec,
    contract: &MemoryProviderContract,
    context: &MemoryRunContext,
) -> MemorySafetyDecision {
    let mut admitted = context.clone();
    let routes = contract.pid_decode_routes.as_ref().unwrap();
    admitted.selection.parameters.decode_overlap = Some(if admitted.use_pid {
        routes.pid.tile_overlap
    } else {
        routes.native.tile_overlap
    });
    route_aware_safety(spec, contract, &admitted)
}

fn wrong_begin(
    _spec: &LoadSpec,
    _contract: &MemoryProviderContract,
    _context: &MemoryRunContext,
) -> gen_core::Result<Option<Box<dyn MemoryRequestScope>>> {
    Ok(Some(Box::new(FixtureScope {
        memory: None,
        finished: false,
    })))
}

fn no_scope(
    _spec: &LoadSpec,
    _contract: &MemoryProviderContract,
    _context: &MemoryRunContext,
) -> gen_core::Result<Option<Box<dyn MemoryRequestScope>>> {
    Ok(None)
}

fn native_only_begin(
    spec: &LoadSpec,
    contract: &MemoryProviderContract,
    context: &MemoryRunContext,
) -> gen_core::Result<Option<Box<dyn MemoryRequestScope>>> {
    if context.use_pid {
        Ok(None)
    } else {
        pid_begin(spec, contract, context)
    }
}

struct DoubleFinishScope {
    memory: Option<gen_core::GenerationMemory>,
}
impl MemoryRequestScope for DoubleFinishScope {
    fn configure_request(
        &mut self,
        request: &mut gen_core::GenerationRequest,
    ) -> gen_core::Result<()> {
        request.memory = self.memory;
        Ok(())
    }
    fn enter_phase(&mut self, _phase: MemoryPhase) -> gen_core::Result<()> {
        Err(gen_core::Error::Msg("finished".into()))
    }
    fn leave_phase(&mut self, _phase: MemoryPhase) -> gen_core::Result<()> {
        Err(gen_core::Error::Msg("finished".into()))
    }
    fn configure_decode(
        &mut self,
        _edge: u32,
        _overlap: u32,
        _geometry: MemoryGeometry,
    ) -> gen_core::Result<()> {
        Err(gen_core::Error::Msg("finished".into()))
    }
    fn configure_attention(&mut self, _chunk: u32) -> gen_core::Result<()> {
        Err(gen_core::Error::Msg("finished".into()))
    }
    fn materialize_transformer_window(&mut self, _first: u32, _count: u32) -> gen_core::Result<()> {
        Err(gen_core::Error::Msg("finished".into()))
    }
    fn finish(&mut self, _outcome: MemoryRunOutcome) -> gen_core::Result<()> {
        Ok(())
    }
}
fn double_finish_begin(
    _spec: &LoadSpec,
    contract: &MemoryProviderContract,
    context: &MemoryRunContext,
) -> gen_core::Result<Option<Box<dyn MemoryRequestScope>>> {
    Ok(Some(Box::new(DoubleFinishScope {
        memory: contract.generation_memory(&context.selection),
    })))
}

struct PostFinishHooksScope {
    memory: Option<gen_core::GenerationMemory>,
    finished: bool,
}
impl MemoryRequestScope for PostFinishHooksScope {
    fn configure_request(
        &mut self,
        request: &mut gen_core::GenerationRequest,
    ) -> gen_core::Result<()> {
        request.memory = self.memory;
        Ok(())
    }
    fn enter_phase(&mut self, _phase: MemoryPhase) -> gen_core::Result<()> {
        Ok(())
    }
    fn leave_phase(&mut self, _phase: MemoryPhase) -> gen_core::Result<()> {
        Ok(())
    }
    fn configure_decode(
        &mut self,
        _edge: u32,
        _overlap: u32,
        _geometry: MemoryGeometry,
    ) -> gen_core::Result<()> {
        Ok(())
    }
    fn configure_attention(&mut self, _chunk: u32) -> gen_core::Result<()> {
        Ok(())
    }
    fn materialize_transformer_window(&mut self, _first: u32, _count: u32) -> gen_core::Result<()> {
        Ok(())
    }
    fn finish(&mut self, _outcome: MemoryRunOutcome) -> gen_core::Result<()> {
        if self.finished {
            Err(gen_core::Error::Msg("finished".into()))
        } else {
            self.finished = true;
            Ok(())
        }
    }
}
fn post_finish_hooks_begin(
    _spec: &LoadSpec,
    contract: &MemoryProviderContract,
    context: &MemoryRunContext,
) -> gen_core::Result<Option<Box<dyn MemoryRequestScope>>> {
    Ok(Some(Box::new(PostFinishHooksScope {
        memory: contract.generation_memory(&context.selection),
        finished: false,
    })))
}

fn errors_for(
    safety_check: fn(&LoadSpec, &MemoryProviderContract, &MemoryRunContext) -> MemorySafetyDecision,
    begin_request: MemoryBehaviorBeginRequest,
) -> Vec<String> {
    let registration = MemoryRegistration {
        provider_id: "pid-provider",
        contract: pid_contract,
        safety_check,
    };
    let behavior = MemoryBehaviorRegistration {
        provider_id: "pid-provider",
        valid_fixtures: pid_fixture,
        begin_request,
    };
    let mut errors = Vec::new();
    check_memory_registration(
        &registration,
        None,
        None,
        Some(&behavior),
        &LoadSpec::new(WeightsSource::Dir("/nonexistent".into())),
        &mut errors,
    );
    errors
}

#[test]
fn overlap_blind_admission_is_killed_by_the_edge_overlap_cross_product() {
    let errors = errors_for(overlap_blind, pid_begin);
    assert!(
        errors
            .iter()
            .any(|error| error.contains("overlap 256 on native route")),
        "{errors:#?}"
    );
    assert!(
        errors
            .iter()
            .any(|error| error.contains("overlap 64 on PiD route")),
        "{errors:#?}"
    );
}

#[test]
fn native_only_scope_is_killed_by_the_pid_lifecycle_fixture() {
    let errors = errors_for(route_aware_safety, native_only_begin);
    assert!(
        errors.iter().any(|error| {
            error.contains("returned no scope") && error.contains("BoundedDecode")
        }),
        "{errors:#?}"
    );
}

#[test]
fn rejecting_a_second_finish_does_not_mask_succeeding_post_finish_hooks() {
    let errors = errors_for(route_aware_safety, post_finish_hooks_begin);
    assert!(
        !errors
            .iter()
            .any(|error| error.contains("second finish succeeded")),
        "{errors:#?}"
    );
    for hook in [
        "enter_phase",
        "leave_phase",
        "configure_decode",
        "configure_attention",
        "materialize_transformer_window",
    ] {
        assert!(
            errors
                .iter()
                .any(|error| { error.contains(hook) && error.contains("succeeded after finish") }),
            "{hook} was not independently killed: {errors:#?}"
        );
    }
}

#[test]
fn mutation_probes_reject_deliberately_broken_adopters() {
    let cases = [
        (
            "always accept",
            errors_for(always_accept, pid_begin),
            "mutated calibration ABI",
        ),
        (
            "always reject",
            errors_for(always_reject, pid_begin),
            "rejects its provider-owned valid context",
        ),
        (
            "handshake blind",
            errors_for(handshake_blind, pid_begin),
            "mutated calibration ABI",
        ),
        (
            "tier blind",
            errors_for(tier_blind, pid_begin),
            "numeric-tier mutation",
        ),
        (
            "budget blind",
            errors_for(budget_blind, pid_begin),
            "impossible budget",
        ),
        (
            "overlap blind",
            errors_for(overlap_blind, pid_begin),
            "overlap 256 on native route",
        ),
        (
            "wrong translation",
            errors_for(route_aware_safety, wrong_begin),
            "configured None",
        ),
        (
            "no scope",
            errors_for(route_aware_safety, no_scope),
            "returned no scope",
        ),
        (
            "native-only scope",
            errors_for(route_aware_safety, native_only_begin),
            "returned no scope",
        ),
        (
            "double finish",
            errors_for(route_aware_safety, double_finish_begin),
            "second finish succeeded",
        ),
        (
            "post-finish hooks",
            errors_for(route_aware_safety, post_finish_hooks_begin),
            "succeeded after finish",
        ),
    ];
    for (name, errors, expected) in cases {
        assert!(
            errors.iter().any(|error| error.contains(expected)),
            "{name} was not killed by {expected:?}: {errors:#?}"
        );
    }
}
