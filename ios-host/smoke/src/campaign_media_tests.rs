use super::*;

#[test]
fn stage_sidecar_is_bounded_nonce_bound_and_preserves_prior_transition_on_write_failure() {
    let docs = tempfile::tempdir().unwrap();
    let nonce = "1".repeat(32);
    let case = format!("recipe-{}", "2".repeat(64));
    let build = "3".repeat(40);
    let memory = runtime_ios::ProcessMemory {
        footprint: Some(u64::MAX),
        available: Some(u64::MAX),
    };
    write_stage(
        docs.path(),
        &nonce,
        &case,
        "before-tokenizer",
        0,
        memory,
        &build,
    )
    .unwrap();
    let path = docs.path().join("campaign-progress.txt");
    let first = std::fs::read_to_string(&path).unwrap();
    assert!(first.contains(&format!(
        "run_id={nonce} build={build} case={case} stage=before-tokenizer"
    )));
    assert!(write_stage(docs.path(), "bad", &case, "wrong", 0, memory, &build).is_err());
    assert!(write_stage(
        docs.path(),
        &nonce,
        &case,
        "wrong",
        0,
        memory,
        &"x".repeat(1024)
    )
    .is_err());
    assert_eq!(std::fs::read_to_string(&path).unwrap(), first);
    write_stage(
        docs.path(),
        &nonce,
        &case,
        &"z".repeat(1024),
        u64::MAX,
        memory,
        &build,
    )
    .unwrap();
    let last = std::fs::read_to_string(&path).unwrap();
    assert!(last.len() <= 512);
    assert!(last.contains(&format!("stage={} bytes=", "z".repeat(128))));
    assert!(!docs.path().join("campaign-progress.next").exists());
}

#[test]
fn campaign_grouping_uses_actual_control_carriers_not_lossy_mode_labels() {
    let catalog = runtime_ios::catalog().unwrap();
    let registry = catalog.media();
    let provider = runtime_ios::media::zimage::Z_IMAGE_TURBO_CONTROL_ID;
    let kinds = accepted_kinds(registry, provider);
    assert!(!kinds.is_empty());
    let mut control_variants = std::collections::BTreeSet::new();
    let mut init_variants = std::collections::BTreeSet::new();
    for kind in &kinds {
        for (prefix, references) in [("control", 0), ("reference", 1)] {
            let mode = format!("{prefix}_{}", kind_token(kind));
            let case = Case {
                provider: provider.into(),
                route: provider.into(),
                tier: "q4".into(),
                edge: 32,
                mode: mode.clone(),
                rung: 0,
                selection: campaign::BASE_SELECTION.into(),
                steps: 1,
                recipe: None,
            };
            let runtime_ios::media::IosWork::Generate(request) =
                work(&case, &CancelFlag::default(), registry).unwrap()
            else {
                panic!("expected generation carrier")
            };
            let actual = (
                request.memory_reference_count(),
                request
                    .conditioning
                    .iter()
                    .filter(|role| matches!(role, Conditioning::Mask { .. }))
                    .count() as u32,
                request
                    .conditioning
                    .iter()
                    .filter(|role| matches!(role, Conditioning::Control { .. }))
                    .count() as u32,
            );
            assert_eq!(
                actual,
                (references + 1, 0, 1),
                "geometry counts ALL image carriers"
            );
            assert_eq!(
                request
                    .conditioning
                    .iter()
                    .filter(|role| matches!(role, Conditioning::Reference { .. }))
                    .count() as u32,
                references,
                "true Reference role remains separate from aggregate geometry"
            );
            let planned =
                variant_for_registry(registry, provider, &mode, "q4", Some("actual-same-class"));
            assert!(
                planned.is_ok(),
                "accepted actual carrier must retain its complete scalar role tuple"
            );
            let planned = planned.unwrap();
            assert_eq!(
                planned,
                runtime_ios::media::pricing::anchor_variant_for_roles(
                    provider,
                    "control",
                    "q4",
                    Some("actual-same-class"),
                    actual.0,
                    actual.1,
                    actual.2,
                )
                .unwrap()
            );
            if references == 0 {
                control_variants.insert(planned);
            } else {
                init_variants.insert(planned);
            }
        }
    }
    assert_eq!(
        control_variants.len(),
        1,
        "role-identical accepted union-control aliases share"
    );
    assert_eq!(init_variants.len(), 1, "role-identical init aliases share");
    assert!(control_variants.is_disjoint(&init_variants));

    // Same actual SDXL carrier: only configured spec + resolved native route decides IP/init.
    let mut spec = runtime_ios::media::sdxl::sdxl_load_spec(
        "sdxl",
        WeightsSource::Dir("metadata-only".into()),
    )
    .unwrap();
    spec.ip_adapter = Some(WeightsSource::File("metadata-only-ip".into()));
    let mut request = GenerationRequest {
        prompt: "role fixture".into(),
        conditioning: vec![Conditioning::Reference {
            image: synthetic_image(32),
            strength: Some(0.4),
        }],
        ..Default::default()
    };
    let sampling = runtime_ios::media::sampling::resolve("sdxl", &request).unwrap();
    let class = sampling.allocation_class();
    let work = runtime_ios::media::IosWork::Generate(request.clone());
    let ip = variant_for_work("sdxl", "q4", Some(class), &spec, &work).unwrap();
    assert_eq!(ip, "ip_adapter");
    spec.ip_adapter = None;
    assert_eq!(
        variant_for_work("sdxl", "q4", Some(class), &spec, &work).unwrap(),
        "image_to_image"
    );
    assert!(
        variant_for_registry(registry, "sdxl", "ip_adapter", "q4", Some(class)).is_err(),
        "unprovisioned spec must stay unresolved"
    );
    spec.ip_adapter = Some(WeightsSource::File("metadata-only-ip".into()));
    let case = Case {
        provider: "sdxl".into(),
        route: "sdxl".into(),
        tier: "q4".into(),
        edge: 32,
        mode: "ip_adapter".into(),
        rung: 0,
        selection: campaign::BASE_SELECTION.into(),
        steps: sampling.steps(),
        recipe: Some(sampling.token().into()),
    };
    let projected = variant_for_case(registry, &case, &spec, Some(class));
    assert!(
        projected.is_ok(),
        "actual configured ordinary case must replay before grouping"
    );
    assert_eq!(projected.unwrap(), "ip_adapter");
    let masked_case = Case {
        mode: "mask".into(),
        ..case.clone()
    };
    let masked = variant_for_case(registry, &masked_case, &spec, Some(class));
    assert!(
        masked.is_ok(),
        "the actual supported single-reference mask case must replay"
    );
    assert_ne!(
        masked.unwrap(),
        "ip_adapter",
        "a real Mask carrier suppresses configured IP"
    );
    let acceleration = runtime_ios::media::sampling::resolve(
        "sdxl",
        &GenerationRequest {
            sampler: Some("lcm".into()),
            ..Default::default()
        },
    )
    .unwrap();
    let rejected_case = Case {
        steps: acceleration.steps(),
        recipe: Some(acceleration.token().into()),
        ..case
    };
    assert!(
        variant_for_case(
            registry,
            &rejected_case,
            &spec,
            Some(acceleration.allocation_class())
        )
        .is_err(),
        "the real campaign case must apply its captured recipe, not silently use ordinary defaults"
    );
    request.sampler = Some("lcm".into());
    let rejected = runtime_ios::media::IosWork::Generate(request);
    assert!(
        variant_for_work("sdxl", "q4", None, &spec, &rejected).is_err(),
        "actual early accel+Ref refusal is not an accepted INIT family"
    );
}

#[test]
fn capture_output_cardinality_includes_serial_count_and_prompt_rows() {
    let mut request = GenerationRequest {
        count: 3,
        ..Default::default()
    };
    assert_eq!(support::expected_generation_outputs(&request).unwrap(), 3);
    request.prompt_batch = vec!["first".into(), "second".into()];
    assert_eq!(support::expected_generation_outputs(&request).unwrap(), 6);
    assert_eq!(request.count, 3);
    request.count = 0;
    assert!(support::expected_generation_outputs(&request).is_err());
}

#[test]
fn campaign_endpoints_use_literal_load_forms_before_contract_selection() {
    let catalog = runtime_ios::catalog().unwrap();
    let registry = catalog.media();
    let discovered = surfaces(registry).unwrap();
    let variant_of = |p: &str, _route: &str, m: &str, t: &str, recipe: &campaign::CaptureRecipe| {
        variant(p, m, t, recipe.memory_class.as_deref())
    };
    let (_, cases) = campaign::plan(&discovered, &campaign::per_route(&variant_of));
    let contracts = registry.memory_contract_surfaces().unwrap();
    for planned in cases.iter().filter(|c| {
        matches!(
            c.purpose,
            campaign::Purpose::Resident | campaign::Purpose::Optimized
        )
    }) {
        let c = &planned.case;
        let spec = load_spec(c, WeightsSource::Dir(PathBuf::new())).unwrap();
        let (policy, shape) = if planned.purpose == campaign::Purpose::Resident {
            (OffloadPolicy::Resident, LoadShape::EagerMaterialization)
        } else {
            (
                OffloadPolicy::Sequential,
                LoadShape::DeferredMaterialization,
            )
        };
        assert_eq!(
            (spec.offload_policy, spec.load_shape),
            (policy, shape),
            "{}",
            c.key()
        );
        let surface = contracts
            .iter()
            .find(|s| {
                s.contract.provider_id == c.provider
                    && s.selector.offload_policy == policy
                    && s.selector.load_shape == shape
                    && match s.selector.tier {
                        MemoryContractSurfaceTier::Q4 => c.tier == "q4",
                        MemoryContractSurfaceTier::Q8 => c.tier == "q8",
                        MemoryContractSurfaceTier::Bf16 => c.tier == "bf16",
                        _ => false,
                    }
            })
            .expect("endpoint load form must have a declared contract surface");
        let strategy = MemoryStrategy::ALL[usize::from(c.rung)];
        let selection = surface
            .contract
            .selection_space(strategy, tier(c), false)
            .unwrap()
            .into_iter()
            .find(|s| selection_token(&s.parameters) == c.selection)
            .expect("enumeration must use the endpoint's own contract");
        let execution = match surface.contract.generation_memory(&selection) {
            Some(execution) => execution,
            None if strategy == MemoryStrategy::Resident
                && surface.contract.resident_request_memory
                    == ResidentRequestMemory::PreserveLoadDefaults =>
            {
                // A declared resident load-default policy legitimately has no request override.
                // The adapter still binds literal Resident+Eager and writes explicit no-stage work.
                GenerationMemory::default()
            }
            None => panic!("optimized endpoint lacks execution levers"),
        };
        if planned.purpose == campaign::Purpose::Resident {
            assert!(!execution.stage_residency);
            assert!(!execution.stream_transformer_blocks);
        }
    }
}
#[test]
fn sdxl_capture_families_follow_real_recipe_classes_and_preserve_unsupported_combinations() {
    let catalog = runtime_ios::catalog().unwrap();
    let registry = catalog.media();
    let discovered = surfaces(registry).unwrap();
    let surface = discovered.iter().find(|s| s.provider == "sdxl").unwrap();
    let descriptor = registry
        .generators()
        .map(|r| (r.descriptor)())
        .find(|d| d.id == "sdxl")
        .unwrap();
    let variant_of = |p: &str, _route: &str, m: &str, t: &str, recipe: &campaign::CaptureRecipe| {
        variant(p, m, t, recipe.memory_class.as_deref())
    };
    let (families, planned) = campaign::plan(&discovered, &campaign::per_route(&variant_of));
    for sampler in descriptor.capabilities.samplers {
        assert!(
            surface.recipes.values().flatten().any(|entry| {
                let recipe = runtime_ios::media::sdxl::sampling::SamplingRecipe::from_token(
                    entry.token.as_deref().unwrap(),
                )
                .unwrap();
                runtime_ios::media::sdxl::sampling::ResolvedSampling::resolve(&recipe.to_request())
                    .unwrap()
                    .sampler_name()
                    == sampler
            }),
            "advertised sampler {sampler} has no reproducible carrier"
        );
    }
    for ((route, recipe_tier, mode), entries) in &surface.recipes {
        for entry in entries {
            let token = entry.token.as_ref().unwrap();
            let recipe =
                runtime_ios::media::sdxl::sampling::SamplingRecipe::from_token(token).unwrap();
            let class =
                runtime_ios::media::sdxl::sampling::ResolvedSampling::resolve(&recipe.to_request())
                    .unwrap()
                    .allocation_class();
            assert_eq!(entry.memory_class.as_deref(), Some(class.as_str()));
            let case = Case {
                provider: "sdxl".into(),
                route: route.clone(),
                tier: recipe_tier.clone(),
                edge: 256,
                mode: mode.clone(),
                rung: 0,
                selection: campaign::BASE_SELECTION.into(),
                steps: entry.steps,
                recipe: Some(token.clone()),
            };
            let actual = work(&case, &CancelFlag::default(), registry);
            if entry.refusal.is_some() {
                assert!(
                    matches!(actual, Err(AdmissionError::Refused(_))),
                    "unsupported combinations must be visible typed refusals"
                );
            } else {
                let runtime_ios::media::IosWork::Generate(request) = actual.unwrap() else {
                    panic!("wrong work carrier")
                };
                let resolved =
                    runtime_ios::media::sdxl::sampling::ResolvedSampling::resolve(&request)
                        .unwrap();
                assert_eq!(resolved.allocation_class(), class);
                let member = format!("sdxl/{route}/{mode}");
                let homes: Vec<_> = families
                    .iter()
                    .filter(|f| {
                        f.tier == *recipe_tier
                            && f.variant
                                == format!(
                                    "{}/{class}",
                                    variant("sdxl", mode, recipe_tier, Some(&class))
                                )
                            && f.members.contains(&member)
                    })
                    .collect();
                assert_eq!(homes.len(), 1, "{member} {class}");
                for purpose in [campaign::Purpose::Resident, campaign::Purpose::Optimized] {
                    assert!(planned.iter().any(|p| p.purpose == purpose
                        && p.case.tier == "bf16"
                        && p.case.recipe == homes[0].recipe
                        && (
                            p.case.provider.clone(),
                            p.case.route.clone(),
                            p.case.mode.clone()
                        ) == homes[0].anchor));
                }
            }
        }
    }
}

#[test]
fn campaign_actual_catalog_covers_each_registered_mode_without_loading() {
    let catalog = runtime_ios::catalog().unwrap();
    let registry = catalog.media();
    let discovered = surfaces(registry).unwrap();
    let variant_of = |p: &str, _route: &str, m: &str, t: &str, recipe: &campaign::CaptureRecipe| {
        variant(p, m, t, recipe.memory_class.as_deref())
    };
    let (families, cases) = campaign::plan(&discovered, &campaign::per_route(&variant_of));
    for registration in registry.generators() {
        let descriptor = (registration.descriptor)();
        let surface = discovered
            .iter()
            .find(|s| s.provider == descriptor.id)
            .unwrap();
        for kind in &descriptor.capabilities.conditioning {
            let mode = format!("{kind:?}").to_ascii_lowercase();
            // A control-required route's modes are per accepted control kind (`control_pose`).
            assert!(
                surface
                    .modes
                    .iter()
                    .any(|m| *m == mode || m.starts_with(&format!("{mode}_"))),
                "{}: {mode}",
                descriptor.id
            );
        }
        // Every registered mode remains represented; each supported allocation-graph family has
        // both endpoints. Unsupported recipe/mode combinations are explicit listing refusals.
        for route in &surface.routes {
            for tier in ["q4", "q8", "bf16"] {
                for mode in &surface.modes {
                    let member = format!("{}/{route}/{mode}", descriptor.id);
                    let homes: Vec<_> = families
                        .iter()
                        .filter(|f| f.tier == tier && f.members.contains(&member))
                        .collect();
                    if descriptor.id == "iris_3b" {
                        // A weights-free listing cannot know this checkpoint's resolved defaults.
                        // Require explicit visible templates, not guessed classes or fake anchors.
                        let entries = surface
                            .recipes
                            .get(&(route.clone(), tier.into(), mode.clone()))
                            .unwrap();
                        assert!(
                            !entries.is_empty(),
                            "{member} {tier}: unresolved capability omitted"
                        );
                        assert!(entries
                            .iter()
                            .all(|entry| entry.memory_class.is_none() && entry.refusal.is_some()));
                        assert!(
                            homes.is_empty(),
                            "unresolved templates cannot form physical families"
                        );
                        continue;
                    }
                    assert!(!homes.is_empty(), "{member} {tier}");
                    for home in homes {
                        let anchored = |purpose| {
                            cases.iter().any(|p| {
                                p.purpose == purpose
                                    && (
                                        p.case.provider.clone(),
                                        p.case.route.clone(),
                                        p.case.mode.clone(),
                                    ) == home.anchor
                                    && p.case.tier == tier
                                    && p.case.recipe == home.recipe
                            })
                        };
                        assert!(
                            anchored(campaign::Purpose::Resident),
                            "{member} {tier} {}",
                            home.variant
                        );
                        assert!(
                            anchored(campaign::Purpose::Optimized),
                            "{member} {tier} {}: no fully optimized anchor",
                            home.variant
                        );
                    }
                }
            }
        }
    }
    for r in registry.transforms() {
        assert!(discovered
            .iter()
            .any(|s| s.provider == (r.descriptor)().id && s.modes.contains(&"restore".into())));
    }
    for r in registry.memory_strategy_registrations() {
        assert!(discovered.iter().any(|s| s.provider == r.provider_id));
    }
    let listing: Value = serde_json::from_str(&listing_with(None).unwrap()).unwrap();
    assert_eq!(listing["schema"], 3);
    let listed = listing["cases"].as_array().unwrap();
    assert!(
        listed.len() < cases.len(),
        "registered settings must not become a capture grid"
    );
    for entry in listed {
        Case::parse(entry["case"].as_str().unwrap()).unwrap();
    }
    let mut expected = Vec::new();
    for surface in &discovered {
        for ((route, tier, mode), recipes) in &surface.recipes {
            for recipe in recipes.iter().filter(|recipe| recipe.refusal.is_none()) {
                expected.push(format!(
                    "{:?}",
                    (
                        &surface.provider,
                        route,
                        tier,
                        mode,
                        &recipe.token,
                        &recipe.memory_class,
                        &recipe.configured_variant
                    )
                ));
            }
        }
    }
    let coverage = listing["recipe_coverage"].as_array().unwrap();
    let mut actual = Vec::new();
    for entry in coverage {
        actual.push(format!(
            "{:?}",
            (
                entry["provider"].as_str().unwrap(),
                entry["route"].as_str().unwrap(),
                entry["tier"].as_str().unwrap(),
                entry["mode"].as_str().unwrap(),
                entry["recipe"].as_str(),
                entry["memory_class"].as_str(),
                entry["configured_variant"].as_str()
            )
        ));
        let family = listing["families"]
            .as_array()
            .unwrap()
            .iter()
            .find(|family| {
                family["architecture"] == entry["architecture"]
                    && family["tier"] == entry["tier"]
                    && family["variant"] == entry["variant"]
                    && family["sampling_recipe"] == entry["representative"]
                    && family["members"]
                        .as_array()
                        .unwrap()
                        .contains(&json!(format!(
                            "{}/{}/{}",
                            entry["provider"].as_str().unwrap(),
                            entry["route"].as_str().unwrap(),
                            entry["mode"].as_str().unwrap()
                        )))
            })
            .expect("every accepted setting has a measured representative family");
        for purpose in ["resident", "optimized"] {
            assert!(listed.iter().any(|case| case["purpose"] == purpose
                && Case::parse(case["case"].as_str().unwrap())
                    .unwrap()
                    .recipe
                    .as_deref()
                    == family["sampling_recipe"].as_str()
                && case["case"].as_str().unwrap().starts_with(&format!(
                    "{}/{}/{}/",
                    entry["provider"].as_str().unwrap(),
                    entry["route"].as_str().unwrap(),
                    entry["tier"].as_str().unwrap()
                ))
                && Case::parse(case["case"].as_str().unwrap()).unwrap().mode
                    == family["anchor"]
                        .as_str()
                        .unwrap()
                        .rsplit_once('/')
                        .unwrap()
                        .1));
        }
    }
    expected.sort();
    expected.dedup();
    actual.sort();
    assert_eq!(actual.len(), expected.len(), "accepted setting count");
    for (index, (found, wanted)) in actual.iter().zip(expected.iter()).enumerate() {
        assert_eq!(found, wanted, "accepted setting mapping at index {index}");
    }
    for purpose in ["validation_steps", "validation_geometry"] {
        assert_eq!(
            listed.iter().filter(|e| e["purpose"] == purpose).count(),
            1,
            "{purpose}"
        );
    }
    // Without an actual depth checkpoint there is no source-owned allocation
    // class. Keep that registered capability visible, not a fabricated anchor.
    assert!(listing["unserved_sampling"]
        .as_array()
        .unwrap()
        .iter()
        .any(|entry| entry["provider"] == "iris_3b_depth"
            && entry["tier"] == "bf16"
            && entry["mode"] == "depth"
            && entry["memory_class"].is_null()));
    assert!(discovered
        .iter()
        .any(|surface| surface.provider == "iris_3b_depth"
            && surface.modes.contains(&"depth".into())
            && surface.serves("bf16")));
    // A tier a route does not serve (the dense-only Iris tasks at q8) is listed as unserved, never
    // silently dropped and never a planned cell.
    assert!(listing["unserved"]
        .as_array()
        .unwrap()
        .iter()
        .any(|u| u["provider"] == "iris_3b_depth" && u["tier"] == "q8"));
    assert!(!listed.iter().any(|e| e["case"]
        .as_str()
        .unwrap()
        .starts_with("iris_3b_depth/iris_3b_depth/q8/")));
    assert!(listed.iter().any(|e| e["case"]
        .as_str()
        .unwrap()
        .starts_with("sdxl/illustrious_xl_v2/q4/1024/text/")));
    // Component-collapse is not inferred from a factory name. Until a corrected per-class mode
    // proof is accepted, mask/IP remain visible independent endpoint families.
    for mode in ["mask", "ip_adapter"] {
        assert!(listed.iter().any(|e| matches!(
            e["purpose"].as_str(),
            Some("resident" | "optimized")
        ) && e["case"]
            .as_str()
            .unwrap()
            .contains(&format!("/{mode}/"))));
    }
    let additive_control = families.iter().any(|f| {
        f.tier == "q4" && f.anchor.2 == "text" && f.members.iter().any(|m| m.ends_with("/control"))
    });
    assert_eq!(
        listed
            .iter()
            .filter(|e| e["purpose"] == "validation_mode")
            .count(),
        usize::from(additive_control)
    );
}
#[test]
fn configured_depth_matrix_has_real_bf16_1024_endpoints_and_guarded_source_keys() {
    let root = tempfile::tempdir().unwrap();
    let fixture = Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("../../crates/media/mlx-gen/mlx-gen-iris/tests/fixtures/tiny-depth");
    super::admission_tests::copy_tree(&fixture, &root.path().join("iris_3b_depth/bf16"));
    let catalog = runtime_ios::catalog().unwrap();
    let discovered = surfaces_with_snapshots(catalog.media(), Some(root.path())).unwrap();
    let variant_of =
        |p: &str, _route: &str, mode: &str, tier: &str, recipe: &campaign::CaptureRecipe| {
            variant(p, mode, tier, recipe.memory_class.as_deref())
        };
    let (_, planned) = campaign::plan(&discovered, &campaign::per_route(&variant_of));
    for purpose in [campaign::Purpose::Resident, campaign::Purpose::Optimized] {
        let endpoints: Vec<_> = planned
            .iter()
            .filter(|entry| {
                entry.purpose == purpose
                    && entry.case.provider == "iris_3b_depth"
                    && entry.case.route == "iris_3b_depth"
                    && entry.case.tier == "bf16"
                    && entry.case.edge == 1024
                    && entry.case.mode == "depth"
            })
            .collect();
        assert!(
            !endpoints.is_empty(),
            "actual configured depth is missing {purpose:?}"
        );
        for entry in endpoints {
            assert!(
                entry.case.recipe.is_some(),
                "a real source-owned sampling recipe is required"
            );
            let mut adapter =
                super::admission_tests::adapter(catalog.media(), root.path().to_owned());
            adapter
                .prepare(&entry.case)
                .unwrap_or_else(|error| panic!("{}: {error:?}", entry.case.key()));
            let key = adapter
                .evidence_key
                .as_ref()
                .expect("guarded source capture produced an actual key");
            assert_eq!(key.mode, "depth");
            assert!(
                key.layout_sha256.len() == 64
                    && key
                        .layout_sha256
                        .bytes()
                        .all(|byte| byte.is_ascii_hexdigit())
            );
            assert!(
                adapter.permit.is_none(),
                "metadata/keys do not grant execution or fake evidence"
            );
        }
    }
}

#[cfg(not(target_os = "ios"))]
#[test]
fn campaign_public_report_host_execution_is_failure_before_allocator_or_load() {
    for diagnostic in [false, true] {
        let output = report(
            &super::admission_tests::cell("anima_turbo/anima_turbo/q8/768/text/4").key(),
            diagnostic,
        );
        assert!(output.starts_with("SMOKE: FAIL"));
        let row: Value = serde_json::from_str(
            output
                .lines()
                .find_map(|l| l.strip_prefix("CAMPAIGN_JSON "))
                .unwrap(),
        )
        .unwrap();
        assert_eq!(row["status"], "diagnostic_failure");
        assert_eq!(row["production_success"], false);
        assert!(row["reason"].as_str().unwrap().contains("real iOS device"));
        assert!(row["phys_footprint_peak_bytes"].is_null());
    }
}
#[test]
fn campaign_missing_snapshot_is_not_a_refusal_proof() {
    let case = super::admission_tests::cell("iris_3b/iris_3b/q4/768/text/3");
    assert!(
        snapshot(Path::new("/definitely-not-a-provisioned-campaign"), &case)
            .unwrap_err()
            .contains("not admission/refusal evidence")
    );
}

/// The capture's kernel lifetime peak splits by phase through the quantum marks: growth inside an
/// opened quantum is that quantum's phase's, glue after a synchronization the next quantum's; a
/// phase that never raised the peak carries the masking reading as an upper bound (not exact).
#[test]
fn a_capture_splits_its_lifetime_peak_by_phase_through_the_quantum_marks() {
    use mlx_gen::ios_pricing::IosPhase::*;
    let m = |phase, opens, lifetime_peak| support::Mark {
        phase: Some(phase),
        opens,
        lifetime_peak,
        label: String::new(),
        scratch: 0,
        current: 0,
        preceding_sampling_mark: None,
    };
    let marks = [
        m(Load, true, 100),          // load floor checked (pre-load glue → load)
        m(Conditioning, true, 900),  // the TE weights landed inside the load quantum → load 900
        m(Conditioning, false, 950), // text layer ran → conditioning 950
        m(Denoise, true, 950),       // glue
        m(Denoise, false, 2000),     // denoise block → denoise 2000
        m(Decode, true, 2000),       // glue
        m(Decode, false, 2000),      // decode never rose above denoise: masked
        support::Mark {
            phase: None,
            opens: true,
            lifetime_peak: 2000,
            label: String::new(),
            scratch: 0,
            current: 0,
            preceding_sampling_mark: None,
        },
    ];
    let split = support::phase_peaks(50, &marks, 2100);
    assert_eq!(split[&Load], (900, true));
    assert_eq!(split[&Conditioning], (950, true));
    assert_eq!(split[&Denoise], (2000, true));
    assert_eq!(split[&Decode], (2000, false));
    // The tail after an unattributed mark is dropped, never credited to another phase.
    assert!(!split.contains_key(&Encode));
    assert!(split.values().all(|(peak, _)| *peak <= 2100));
}

/// The per-formula DEVICE calibration of a capture: each opened quantum's growth over its check's
/// footprint, exact only when the lifetime peak rose inside it; per label the worst
/// observed/priced invocation (exact before masked).
#[test]
fn a_capture_observes_each_quantum_formula_against_its_priced_scratch() {
    let q = |label: &str, opens, scratch, current, lifetime_peak| support::Mark {
        phase: None,
        opens,
        lifetime_peak,
        label: label.into(),
        scratch,
        current,
        preceding_sampling_mark: None,
    };
    let marks = [
        q("tile", true, 400, 1000, 1500), // peak rose to 1700 inside: 700 observed, exact
        q("tile", false, 0, 1100, 1700),
        q("tile", true, 400, 1100, 1700), // rose to 1750: 650, exact, ratio 650/400 < 700/400
        q("tile", false, 0, 1100, 1750),
        q("block", true, 100, 1600, 1750), // never rose: masked, 150 upper bound
        q("block", false, 0, 1600, 1750),
        q("last", true, 50, 1700, 1750), // the run's end peak 1800 is after it: exact 100
    ];
    let o = support::quantum_observations(&marks, 1800);
    assert_eq!(
        o["tile"],
        support::Observation {
            priced: 400,
            observed: 700,
            exact: true,
            preceding_sampling_mark: None,
        }
    );
    assert_eq!(
        o["block"],
        support::Observation {
            priced: 100,
            observed: 150,
            exact: false,
            preceding_sampling_mark: None,
        }
    );
    assert_eq!(
        o["last"],
        support::Observation {
            priced: 50,
            observed: 100,
            exact: true,
            preceding_sampling_mark: None,
        }
    );
    assert_eq!(o.len(), 3, "synchronizations are not quanta");
}

#[test]
fn unprovisioned_iris_templates_are_visible_unresolved_and_never_physical_anchor_classes() {
    let catalog = runtime_ios::catalog().unwrap();
    let registry = catalog.media();
    let discovered = surfaces(registry).unwrap();
    let variant_of = |p: &str, _route: &str, m: &str, t: &str, recipe: &campaign::CaptureRecipe| {
        variant(p, m, t, recipe.memory_class.as_deref())
    };
    let (_, planned) = campaign::plan(&discovered, &campaign::per_route(&variant_of));
    for surface in discovered
        .iter()
        .filter(|surface| surface.provider.starts_with("iris_"))
    {
        assert!(!surface.modes.is_empty());
        for route in &surface.routes {
            for tier in ["q4", "q8", "bf16"] {
                for mode in &surface.modes {
                    let entries = surface
                        .recipes
                        .get(&(route.clone(), tier.into(), mode.clone()))
                        .unwrap();
                    assert!(
                        !entries.is_empty(),
                        "missing unresolved {route}/{tier}/{mode}"
                    );
                    assert!(entries.iter().all(|entry| entry.memory_class.is_none()
                        && entry.configured_variant.is_none()
                        && entry
                            .refusal
                            .as_deref()
                            .is_some_and(|reason| reason.starts_with("unresolved controls:"))));
                    if surface.provider == "iris_3b" {
                        assert!(entries.iter().all(|entry| entry
                            .token
                            .as_deref()
                            .is_some_and(|token| token.starts_with("iris-template1-"))));
                    }
                }
            }
        }
        assert!(
            !planned
                .iter()
                .any(|entry| entry.case.provider == surface.provider),
            "unresolved templates cannot masquerade as resolved physical two-anchor families"
        );
    }
    let listed: Value = serde_json::from_str(&listing_with(None).unwrap()).unwrap();
    assert!(listed["unserved_sampling"]
        .as_array()
        .unwrap()
        .iter()
        .any(|entry| entry["provider"] == "iris_3b"
            && entry["tier"] == "q8"
            && entry["memory_class"].is_null()
            && entry["reason"]
                .as_str()
                .unwrap()
                .starts_with("unresolved controls:")));
}

fn copy_tiny_iris_metadata_slot(root: &Path, tier: &str, shift: f64, prediction: &str) {
    fn copy_dir(from: &Path, to: &Path) {
        std::fs::create_dir_all(to).unwrap();
        for entry in std::fs::read_dir(from).unwrap() {
            let entry = entry.unwrap();
            if entry.file_type().unwrap().is_dir() {
                copy_dir(&entry.path(), &to.join(entry.file_name()));
            } else {
                std::fs::copy(entry.path(), to.join(entry.file_name())).unwrap();
            }
        }
    }
    // Existing synthetic source fixture, never downloaded/real model weights or Array evaluation.
    let fixture = Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("../../crates/media/mlx-gen/mlx-gen-iris/tests/fixtures/tiny-snapshot");
    let slot = root.join("iris_3b").join(tier);
    copy_dir(&fixture.join("iris"), &slot);
    copy_dir(&fixture.join("text_encoder"), &slot.join("text_encoder"));
    if tier == "q8" {
        // The dense numerical fixture's MLP width 48 cannot use affine group 32
        // or 64. Build a coherent synthetic width-64 tower for this q8 metadata
        // test, changing BOTH config and every affected tensor shape (not merely
        // declaring a quantization policy over incompatible source geometry).
        let config = slot.join("text_encoder/config.json");
        let mut value: Value = serde_json::from_slice(&std::fs::read(&config).unwrap()).unwrap();
        assert_eq!(value["text_config"]["intermediate_size"], 48);
        value["text_config"]["intermediate_size"] = json!(64);
        std::fs::write(config, serde_json::to_vec(&value).unwrap()).unwrap();
        let weights = slot.join("text_encoder/model.safetensors");
        let mut headers =
            runtime_ios::gen_core::weightsmeta::safetensors_path_tensor_headers(&weights).unwrap();
        let mut changed = 0;
        for tensor in &mut headers {
            if tensor.name.contains(".mlp.") {
                for dimension in &mut tensor.shape {
                    if *dimension == 48 {
                        *dimension = 64;
                        changed += 1;
                    }
                }
            }
        }
        assert!(
            changed > 0,
            "the quantizable fixture must change actual MLP matrices"
        );
        let dtypes: Vec<_> = headers
            .iter()
            .map(|tensor| format!("{:?}", tensor.dtype))
            .collect();
        let tensors: Vec<_> = headers
            .iter()
            .zip(&dtypes)
            .map(|(tensor, dtype)| {
                (
                    tensor.name.as_str(),
                    dtype.as_str(),
                    tensor.shape.as_slice(),
                )
            })
            .collect();
        gen_core_testkit::wan_i2v::write_safetensors(&weights, &tensors);
    }
    let config = slot.join("config.yaml");
    let original = std::fs::read_to_string(&config).unwrap();
    assert_eq!(original.matches("  shift: 4.0").count(), 1);
    assert_eq!(original.matches("  prediction: v").count(), 1);
    let changed = original
        .replacen("  shift: 4.0", &format!("  shift: {shift:?}"), 1)
        .replacen("  prediction: v", &format!("  prediction: {prediction}"), 1);
    std::fs::write(config, changed).unwrap();
    std::fs::write(
        slot.join("components.json"),
        r#"{"text_encoder":"text_encoder"}"#,
    )
    .unwrap();
}

#[test]
fn configured_iris_recipes_read_actual_tier_and_mode_and_dedup_only_normalized_classes() {
    let root = tempfile::tempdir().unwrap();
    copy_tiny_iris_metadata_slot(root.path(), "q8", 3.0, "x");
    copy_tiny_iris_metadata_slot(root.path(), "bf16", 4.0, "v");
    let catalog = runtime_ios::catalog().unwrap();
    let registry = catalog.media();
    let routes = vec!["iris_3b".into()];
    let modes = vec!["text".into(), "prompt_batch".into()];
    let recipes = capture_recipes(registry, "iris_3b", &routes, &modes, Some(root.path()));
    for (tier, shift, prediction) in [("q8", 3.0f64, "x"), ("bf16", 4.0f64, "v")] {
        for (mode, rows) in [("text", 1u64), ("prompt_batch", 2u64)] {
            let key = ("iris_3b".into(), tier.into(), mode.into());
            let entries = recipes.get(&key).unwrap();
            assert!(
                entries.iter().any(|entry| entry.refusal.is_none()),
                "{key:?}: {entries:?}"
            );
            let mut classes = std::collections::BTreeSet::new();
            for entry in entries.iter().filter(|entry| entry.refusal.is_none()) {
                let actual = runtime_ios::media::sampling::from_token(
                    "iris_3b",
                    entry.token.as_deref().unwrap(),
                )
                .unwrap();
                assert_eq!(
                    entry.memory_class.as_deref(),
                    Some(actual.allocation_class())
                );
                assert!(
                    classes.insert(actual.allocation_class().to_string()),
                    "duplicate configured class"
                );
                assert_eq!(
                    actual.provenance()["prediction"].as_str(),
                    Some(prediction),
                    "wrong actual tier's artifact prediction semantics"
                );
                assert_eq!(
                    actual.provenance()["prompt_rows"].as_u64(),
                    Some(rows),
                    "wrong actual mode carrier"
                );
                assert!(entry.configured_variant.is_some());
                let case = Case {
                    provider: "iris_3b".into(),
                    route: "iris_3b".into(),
                    tier: tier.into(),
                    edge: campaign::reference_edge("iris_3b"),
                    mode: mode.into(),
                    rung: 0,
                    selection: campaign::BASE_SELECTION.into(),
                    steps: entry.steps,
                    recipe: entry.token.clone(),
                };
                let spec = snapshot(root.path(), &case).unwrap();
                let raw = GenerationRequest {
                    prompt: prompt("iris_3b"),
                    width: case.edge,
                    height: case.edge,
                    ..Default::default()
                };
                let default =
                    runtime_ios::media::sampling::resolve_for_spec("iris_3b", &spec, &raw).unwrap();
                assert_eq!(default.provenance()["flow_shift_bits"].as_u64(), Some(shift.to_bits()),
                    "None must resolve the actual tier's f64 default, not a unit-shift template override");
                let runtime_ios::media::IosWork::Generate(request) =
                    work_with_spec(&case, &CancelFlag::default(), registry, Some(&spec)).unwrap()
                else {
                    panic!("wrong actual work kind")
                };
                assert_eq!(
                    runtime_ios::media::sampling::resolve_for_spec("iris_3b", &spec, &request)
                        .unwrap()
                        .token(),
                    entry.token.as_deref().unwrap()
                );
            }
        }
    }
    // The absent q4 slot must not borrow q8/bf16 defaults or a fabricated class.
    assert!(recipes
        .get(&("iris_3b".into(), "q4".into(), "text".into()))
        .unwrap()
        .iter()
        .all(|entry| entry.refusal.is_some() && entry.memory_class.is_none()));
}
