use super::super::tests::{admitted_context, adopted_contract, pid_contract};
use super::*;
use crate::{LoadSpec, WeightsSource};

fn spec() -> LoadSpec {
    LoadSpec::new(WeightsSource::Dir("/weights-free-profile-sentinel".into())).with_quant(Quant::Q4)
}

fn refuse_before(
    _: &MemoryProviderContract,
    _: &MemoryRunContext,
    _: MemoryNumericTier,
) -> Result<()> {
    Err(Error::Msg("before refused".to_owned()))
}

fn refuse_within(
    _: &MemoryProviderContract,
    _: &MemoryRunContext,
    _: MemoryNumericTier,
) -> Result<()> {
    Err(Error::Unsupported("within refused".to_owned()))
}

fn refuse_after(
    _: &MemoryProviderContract,
    _: &MemoryRunContext,
    _: MemoryNumericTier,
) -> Result<()> {
    Err(Error::Msg("after refused".to_owned()))
}

fn refuse_q8(
    _: &MemoryProviderContract,
    _: &MemoryRunContext,
    tier: MemoryNumericTier,
) -> Result<()> {
    if tier.quant == Some(Quant::Q8) {
        return Err(Error::Msg("q8 refused".to_owned()));
    }
    Ok(())
}

fn with_gates(gates: MemoryProfileGates) -> ProviderMemoryProfile {
    ProviderMemoryProfile {
        gates,
        ..ProviderMemoryProfile::STANDARD
    }
}

fn reason(decision: MemorySafetyDecision) -> String {
    match decision {
        MemorySafetyDecision::Reject { reason } => reason,
        MemorySafetyDecision::Accept => panic!("expected a rejection"),
    }
}

#[test]
fn tier_policies_resolve_the_loaded_tier() {
    let contract = adopted_contract();
    let mut spec = spec();
    spec.precision = Precision::Fp32;
    let requested = MemoryProfileTier::Requested
        .resolve(&spec, &contract)
        .unwrap();
    assert_eq!(
        (requested.precision, requested.quant),
        (Precision::Fp32, Some(Quant::Q4))
    );
    let fixed = MemoryProfileTier::FixedPrecision(Precision::Bf16)
        .resolve(&spec, &contract)
        .unwrap();
    assert_eq!(
        (fixed.precision, fixed.quant),
        (Precision::Bf16, Some(Quant::Q4))
    );
    fn packed(_: &LoadSpec, contract: &MemoryProviderContract) -> Result<MemoryNumericTier> {
        Err(Error::Unsupported(format!(
            "{}: unreadable tier",
            contract.provider_id
        )))
    }
    assert!(matches!(
        MemoryProfileTier::Resolved(packed).resolve(&spec, &contract),
        Err(Error::Unsupported(message)) if message == "test-provider: unreadable tier"
    ));
}

#[test]
fn a_standard_profile_is_exactly_the_shared_pipeline() {
    let contract = adopted_contract();
    let context = admitted_context(&contract);
    let tier = context.selection.tier;
    let profile = ProviderMemoryProfile::STANDARD;
    assert_eq!(
        standard_safety_check(&profile, &contract, &context, tier),
        standard_memory_strategy_safety_check(&contract, &context, Some(tier), None)
    );
    let mut over_budget = context.clone();
    over_budget.predicted_peak_bytes = 1025;
    assert_eq!(
        standard_safety_check(&profile, &contract, &over_budget, tier),
        standard_memory_strategy_safety_check(&contract, &over_budget, Some(tier), None)
    );
    assert_eq!(
        standard_registered_safety_check(&profile, &spec(), &contract, &context),
        default_registered_memory_strategy_safety_check(&spec(), &contract, &context)
    );
    let q8 = spec().with_quant(Quant::Q8);
    assert_eq!(
        standard_registered_safety_check(&profile, &q8, &contract, &context),
        default_registered_memory_strategy_safety_check(&q8, &contract, &context)
    );
}

#[test]
fn gate_positions_fix_rejection_precedence() {
    let contract = adopted_contract();
    let context = admitted_context(&contract);
    let tier = context.selection.tier;
    let mut crossed = context.clone();
    crossed.calibration_fingerprint = "other-layout".to_owned();

    // A `before` gate wins over a shared rejection; its reason is its error's rendering.
    let before = with_gates(MemoryProfileGates {
        before: Some(refuse_before),
        ..Default::default()
    });
    assert_eq!(
        reason(standard_safety_check(&before, &contract, &crossed, tier)),
        "before refused"
    );

    // A `within` gate runs after the handshake, so the shared refusal wins.
    let within = with_gates(MemoryProfileGates {
        within: Some(refuse_within),
        ..Default::default()
    });
    assert!(
        reason(standard_safety_check(&within, &contract, &crossed, tier))
            .contains("calibration handshake mismatch")
    );
    assert_eq!(
        reason(standard_safety_check(&within, &contract, &context, tier)),
        "unsupported: within refused"
    );
    // ... and before the budget check.
    let mut over_budget = context.clone();
    over_budget.predicted_peak_bytes = 1025;
    assert_eq!(
        reason(standard_safety_check(
            &within,
            &contract,
            &over_budget,
            tier
        )),
        "unsupported: within refused"
    );

    // An `after` gate never masks a shared rejection.
    let after = with_gates(MemoryProfileGates {
        after: Some(refuse_after),
        ..Default::default()
    });
    assert!(
        reason(standard_safety_check(&after, &contract, &over_budget, tier))
            .contains("exceeds effective budget")
    );
    assert_eq!(
        reason(standard_safety_check(&after, &contract, &context, tier)),
        "after refused"
    );

    // Gates see the loaded tier.
    let tiered = with_gates(MemoryProfileGates {
        within: Some(refuse_q8),
        ..Default::default()
    });
    assert_eq!(
        standard_safety_check(&tiered, &contract, &context, tier),
        MemorySafetyDecision::Accept
    );
}

#[test]
fn validate_context_keeps_a_gate_error_typed() {
    let contract = adopted_contract();
    let context = admitted_context(&contract);
    let tier = context.selection.tier;
    let after = with_gates(MemoryProfileGates {
        after: Some(refuse_after),
        ..Default::default()
    });
    assert!(matches!(
        standard_validate_context(&after, &contract, &context, tier),
        Err(Error::Msg(message)) if message == "after refused"
    ));
    let mut over_budget = context.clone();
    over_budget.predicted_peak_bytes = 1025;
    assert!(matches!(
        standard_validate_context(&after, &contract, &over_budget, tier),
        Err(Error::Unsupported(message)) if message.contains("exceeds effective budget")
    ));
    assert!(
        standard_validate_context(&ProviderMemoryProfile::STANDARD, &contract, &context, tier)
            .is_ok()
    );
}

#[test]
fn validation_error_rendering_matches_the_result_form() {
    let contract = adopted_contract();
    let mut over_budget = admitted_context(&contract);
    over_budget.predicted_peak_bytes = 1025;
    let tier = over_budget.selection.tier;
    let profile = ProviderMemoryProfile {
        rejection: MemoryRejectionRendering::ValidationError,
        gates: MemoryProfileGates::after(refuse_after),
        ..ProviderMemoryProfile::STANDARD
    };
    let shared = reason(standard_safety_check(
        &profile,
        &contract,
        &over_budget,
        tier,
    ));
    assert!(shared.starts_with("unsupported: test-provider: incremental live demand"));
    assert_eq!(
        shared,
        standard_validate_context(&profile, &contract, &over_budget, tier)
            .unwrap_err()
            .to_string()
    );
    let context = admitted_context(&contract);
    assert_eq!(
        reason(standard_safety_check(&profile, &contract, &context, tier)),
        "after refused"
    );
    assert_eq!(
        standard_validate_context(&profile, &contract, &context, tier)
            .unwrap_err()
            .to_string(),
        "after refused"
    );
}

#[test]
fn registered_safety_check_refuses_an_unresolvable_tier() {
    fn unreadable(_: &LoadSpec, contract: &MemoryProviderContract) -> Result<MemoryNumericTier> {
        Err(Error::Msg(format!("{}: no tier", contract.provider_id)))
    }
    let profile = ProviderMemoryProfile {
        tier: MemoryProfileTier::Resolved(unreadable),
        ..ProviderMemoryProfile::STANDARD
    };
    let contract = adopted_contract();
    let context = admitted_context(&contract);
    assert_eq!(
        reason(standard_registered_safety_check(
            &profile,
            &spec(),
            &contract,
            &context
        )),
        "test-provider: no tier"
    );
}

#[test]
fn begin_request_resolves_admits_then_builds_the_scope() {
    struct NoopScope;
    impl MemoryRequestScope for NoopScope {
        fn configure_request(&mut self, _: &mut GenerationRequest) -> Result<()> {
            Ok(())
        }
        fn enter_phase(&mut self, _: MemoryPhase) -> Result<()> {
            Ok(())
        }
        fn leave_phase(&mut self, _: MemoryPhase) -> Result<()> {
            Ok(())
        }
        fn configure_decode(&mut self, _: u32, _: u32, _: MemoryGeometry) -> Result<()> {
            Ok(())
        }
        fn configure_attention(&mut self, _: u32) -> Result<()> {
            Ok(())
        }
        fn materialize_transformer_window(&mut self, _: u32, _: u32) -> Result<()> {
            Ok(())
        }
        fn finish(&mut self, _: MemoryRunOutcome) -> Result<()> {
            Ok(())
        }
    }
    fn scope(
        provider_id: &'static str,
        _: &LoadSpec,
        _: &MemoryProviderContract,
        _: &MemoryRunContext,
        tier: MemoryNumericTier,
    ) -> Result<Option<Box<dyn MemoryRequestScope>>> {
        assert_eq!(
            (provider_id, tier.quant),
            ("test-provider", Some(Quant::Q4))
        );
        Ok(Some(Box::new(NoopScope)))
    }
    let contract = adopted_contract();
    let context = admitted_context(&contract);
    let profile = ProviderMemoryProfile::STANDARD;
    assert!(standard_begin_request(
        &profile,
        "test-provider",
        &spec(),
        &contract,
        &context,
        scope
    )
    .unwrap()
    .is_some());
    // The tier the context selected is not the tier this spec loads: refused before the scope.
    assert!(matches!(
        standard_begin_request(&profile, "test-provider", &spec().with_quant(Quant::Q8), &contract, &context, scope),
        Err(Error::Unsupported(message)) if message.contains("does not match loaded tier")
    ));
    // A gate's typed error reaches the caller as typed: one rendering, no doubled prefix.
    let gated = ProviderMemoryProfile {
        gates: MemoryProfileGates::after(refuse_after),
        ..profile
    };
    assert!(matches!(
        standard_begin_request(&gated, "test-provider", &spec(), &contract, &context, scope),
        Err(Error::Msg(message)) if message == "after refused"
    ));
    fn unreadable(_: &LoadSpec, _: &MemoryProviderContract) -> Result<MemoryNumericTier> {
        Err(Error::Msg("no tier".to_owned()))
    }
    let unresolvable = ProviderMemoryProfile {
        tier: MemoryProfileTier::Resolved(unreadable),
        ..profile
    };
    assert!(matches!(
        standard_begin_request(&unresolvable, "test-provider", &spec(), &contract, &context, scope),
        Err(Error::Msg(message)) if message == "no tier"
    ));
}

#[test]
fn fixtures_follow_the_declared_routes_gates_and_pid_copies() {
    const ROUTES: &[MemoryFixtureRoute] = &[
        MemoryFixtureRoute::TEXT_TO_IMAGE,
        MemoryFixtureRoute::new(MemoryFixtureMode::Other("character_image"), 1)
            .with_phases(MemoryFixturePhases::WhenStaged)
            .with_overlay(MemoryFixtureOverlay::WhenAdapters("lora")),
    ];
    let contract = pid_contract();
    let profile = ProviderMemoryProfile {
        fixtures: MemoryProfileFixtures::routes(ROUTES)
            .with_pid(MemoryFixturePid::WhenPidDecodeRoutes)
            .with_prompt("profile fixture"),
        ..ProviderMemoryProfile::STANDARD
    };
    let fixtures = |strategy| {
        standard_registered_valid_fixture(&profile, &spec(), &contract, strategy).unwrap()
    };
    assert!(fixtures(MemoryStrategy::Resident).is_empty());

    let staged = fixtures(MemoryStrategy::StagedResidency);
    assert_eq!(staged.len(), 2, "no decode rung engaged, so no PiD copies");
    assert_eq!(staged[0].context.mode, MemoryMode::TextToImage);
    assert!(!staged[0].context.has_phases);
    assert_eq!(
        staged[1].context.mode,
        MemoryMode::Other("character_image".to_owned())
    );
    assert_eq!(staged[1].context.geometry.reference_count, 1);
    assert!(staged[1].context.has_phases);
    assert_eq!(
        staged[1].context.overlay, None,
        "the spec carries no adapters"
    );
    assert!(staged
        .iter()
        .all(|fixture| fixture.request.prompt == "profile fixture"));

    let decode = fixtures(MemoryStrategy::BoundedDecode);
    let pid: Vec<_> = decode
        .iter()
        .map(|fixture| fixture.context.use_pid)
        .collect();
    assert_eq!(pid, [false, true, false, true]);

    let mut native_only = contract.clone();
    native_only.pid_decode_routes = None;
    native_only.strategies = adopted_contract().strategies;
    assert_eq!(
        standard_registered_valid_fixture(
            &profile,
            &spec(),
            &native_only,
            MemoryStrategy::BoundedDecode
        )
        .unwrap()
        .len(),
        2
    );

    let mut unimplemented = adopted_contract();
    unimplemented.strategies[3].support = MemoryStrategySupport::Missing;
    let implemented_only = ProviderMemoryProfile {
        fixtures: MemoryProfileFixtures::TEXT_TO_IMAGE.implemented_only(),
        ..ProviderMemoryProfile::STANDARD
    };
    assert!(standard_registered_valid_fixture(
        &implemented_only,
        &spec(),
        &unimplemented,
        MemoryStrategy::BoundedAttention
    )
    .unwrap()
    .is_empty());

    let mut uncalibrated = adopted_contract();
    uncalibrated.calibration = None;
    let calibrated_only = ProviderMemoryProfile {
        fixtures: MemoryProfileFixtures::TEXT_TO_IMAGE.calibrated_only(),
        ..ProviderMemoryProfile::STANDARD
    };
    assert!(standard_registered_valid_fixture(
        &calibrated_only,
        &spec(),
        &uncalibrated,
        MemoryStrategy::StagedResidency
    )
    .unwrap()
    .is_empty());
    assert!(standard_registered_valid_fixture(
        &ProviderMemoryProfile::STANDARD,
        &spec(),
        &uncalibrated,
        MemoryStrategy::StagedResidency
    )
    .is_err());
}

#[test]
fn fixture_overlay_and_phase_sources_read_the_load_spec() {
    let contract = adopted_contract();
    let route = MemoryFixtureRoute::TEXT_TO_IMAGE
        .with_phases(MemoryFixturePhases::WhenSequentialOffload)
        .with_overlay(MemoryFixtureOverlay::WhenAdapters("lora"));
    let mut spec = spec();
    let plain = route.route(&spec, &contract, MemoryStrategy::StagedResidency, false);
    assert_eq!((plain.has_phases, plain.overlay), (false, None));
    spec.offload_policy = crate::OffloadPolicy::Sequential;
    spec.adapters.push(AdapterSpec::new(
        "/a.safetensors".into(),
        1.0,
        crate::AdapterKind::Lora,
    ));
    let loaded = route.route(&spec, &contract, MemoryStrategy::StagedResidency, false);
    assert_eq!(
        (loaded.has_phases, loaded.overlay.as_deref()),
        (true, Some("lora"))
    );
    let fixed = MemoryFixtureRoute::TEXT_TO_IMAGE
        .with_phases(MemoryFixturePhases::Always)
        .with_overlay(MemoryFixtureOverlay::Fixed("identity"))
        .route(&spec, &contract, MemoryStrategy::StagedResidency, false);
    assert_eq!(
        (fixed.has_phases, fixed.overlay.as_deref()),
        (true, Some("identity"))
    );
}

#[test]
fn fingerprint_table_spells_routes_tiers_and_pins() {
    const TABLE: MemoryFingerprintTable = MemoryFingerprintTable {
        template: "family-{route}-{tier}-mlx-shared-ladder-v1",
        routes: &[("family_dev", "dev"), ("family_fast", "fast")],
        pins: &[MemoryFingerprintPin {
            route: "dev",
            tier: "q4",
            fingerprint: "sc-1-measured-family-dev-q4",
        }],
    };
    assert_eq!(TABLE.route("family_fast"), Some("fast"));
    assert_eq!(TABLE.route("family_unknown"), None);
    assert_eq!(
        TABLE.fingerprint("family_dev", "q4").as_deref(),
        Some("sc-1-measured-family-dev-q4")
    );
    assert_eq!(
        TABLE.fingerprint("family_dev", "q8").as_deref(),
        Some("family-dev-q8-mlx-shared-ladder-v1")
    );
    assert_eq!(
        TABLE.fingerprint("family_fast", "q4").as_deref(),
        Some("family-fast-q4-mlx-shared-ladder-v1")
    );
    assert_eq!(TABLE.fingerprint("family_unknown", "q4"), None);
    assert_eq!(
        TABLE.render("dev", "bf16"),
        "family-dev-bf16-mlx-shared-ladder-v1"
    );
}

#[test]
fn architecture_source_guards_the_weights_free_sentinel() {
    const FACTS: MemoryArchitectureFacts = MemoryArchitectureFacts {
        attention_heads: Some(24),
        head_dim: Some(128),
        transformer_blocks: Some(42),
        patch_size: Some(1),
        latent_channels: Some(128),
        vae_spatial_scale: Some(16),
        vae_temporal_scale: None,
        activation_dtype_width: Some(4),
    };
    fn from_snapshot(provider_id: &str, root: &Path, _: &LoadSpec) -> MemoryArchitectureFacts {
        assert_eq!(provider_id, "p");
        assert!(root.is_dir());
        FACTS
    }
    let declared = ProviderMemoryProfile {
        architecture: Some(MemoryArchitectureSource::Declared(FACTS)),
        ..ProviderMemoryProfile::STANDARD
    };
    assert_eq!(standard_architecture_facts(&declared, "p", &spec()), FACTS);
    assert_eq!(
        standard_architecture_facts(&ProviderMemoryProfile::STANDARD, "p", &spec()),
        MemoryArchitectureFacts::default()
    );
    let snapshot = ProviderMemoryProfile {
        architecture: Some(MemoryArchitectureSource::Snapshot(from_snapshot)),
        ..ProviderMemoryProfile::STANDARD
    };
    assert_eq!(
        standard_architecture_facts(&snapshot, "p", &spec()),
        MemoryArchitectureFacts::default(),
        "a sentinel path that is not on disk declares no axis"
    );
    let root = tempfile::tempdir().unwrap();
    let resolved = LoadSpec::new(WeightsSource::Dir(root.path().to_path_buf()));
    assert_eq!(resolved_snapshot_root(&resolved), Some(root.path()));
    assert_eq!(
        standard_architecture_facts(&snapshot, "p", &resolved),
        FACTS
    );
    let file = LoadSpec::new(WeightsSource::File(root.path().join("model.safetensors")));
    assert_eq!(resolved_snapshot_root(&file), None);
}
