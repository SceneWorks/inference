use super::*;
#[test]
fn campaign_actual_catalog_covers_each_registered_mode_without_loading() {
    let catalog = runtime_ios::catalog().unwrap();
    let registry = catalog.media();
    let discovered = surfaces(registry);
    let variant_of = |p: &str, m: &str| variant(p, m);
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
        // Every registered mode of every route and served tier belongs to exactly one family, and
        // that family has its resident anchor and — the fully optimized end must exist on the phone —
        // its optimized anchor.
        for route in &surface.routes {
            for tier in ["q4", "q8", "bf16"] {
                for mode in &surface.modes {
                    let member = format!("{}/{route}/{mode}", descriptor.id);
                    let homes: Vec<_> = families
                        .iter()
                        .filter(|f| f.tier == tier && f.members.contains(&member))
                        .collect();
                    assert_eq!(homes.len(), 1, "{member} {tier}");
                    let anchored = |purpose| {
                        cases.iter().any(|p| {
                            p.purpose == purpose
                                && (
                                    p.case.provider.clone(),
                                    p.case.route.clone(),
                                    p.case.mode.clone(),
                                ) == homes[0].anchor
                                && p.case.tier == tier
                        })
                    };
                    assert!(anchored(campaign::Purpose::Resident), "{member} {tier}");
                    assert!(
                        anchored(campaign::Purpose::Optimized),
                        "{member} {tier}: no fully optimized anchor"
                    );
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
    assert_eq!(listing["schema"], 2);
    let listed = listing["cases"].as_array().unwrap();
    assert_eq!(listed.len(), cases.len());
    for (entry, planned) in listed.iter().zip(&cases) {
        assert_eq!(entry["case"], planned.case.key());
        assert_eq!(entry["purpose"], planned.purpose.label());
        assert_eq!(entry["repeats"], planned.repeats);
        assert_eq!(
            Case::parse(entry["case"].as_str().unwrap()).unwrap(),
            planned.case
        );
    }
    for purpose in ["validation_steps", "validation_geometry", "validation_mode"] {
        assert_eq!(
            listed.iter().filter(|e| e["purpose"] == purpose).count(),
            1,
            "{purpose}"
        );
    }
    assert!(listed.iter().any(|e| e["case"]
        .as_str()
        .unwrap()
        .starts_with("iris_3b_depth/iris_3b_depth/bf16/1024/depth/")));
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
    // Additive modes never get their own anchors; their family's text member does.
    assert!(!listed.iter().any(|e| {
        let c = e["case"].as_str().unwrap();
        matches!(e["purpose"].as_str(), Some("resident" | "optimized"))
            && (c.contains("/mask/") || c.contains("/ip_adapter/"))
    }));
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
            exact: true
        }
    );
    assert_eq!(
        o["block"],
        support::Observation {
            priced: 100,
            observed: 150,
            exact: false
        }
    );
    assert_eq!(
        o["last"],
        support::Observation {
            priced: 50,
            observed: 100,
            exact: true
        }
    );
    assert_eq!(o.len(), 3, "synchronizations are not quanta");
}
