use super::*;
#[test]
fn campaign_actual_catalog_covers_each_registered_mode_without_loading() {
    let catalog = runtime_ios::catalog().unwrap();
    let registry = catalog.media();
    let discovered = surfaces(registry);
    let cases = campaign::enumerate(&discovered);
    for registration in registry.generators() {
        let descriptor = (registration.descriptor)();
        let surface = discovered
            .iter()
            .find(|s| s.provider == descriptor.id)
            .unwrap();
        for kind in &descriptor.capabilities.conditioning {
            assert!(surface
                .modes
                .contains(&format!("{kind:?}").to_ascii_lowercase()));
        }
        for route in &surface.routes {
            for tier in ["q4", "q8", "bf16"] {
                for edge in [512, 768, 1024] {
                    assert!(cases.iter().any(|c| c.provider == descriptor.id
                        && c.route == *route
                        && c.tier == tier
                        && c.edge == edge));
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
    let listing: Value = serde_json::from_str(&listing().unwrap()).unwrap();
    assert_eq!(listing["schema"], 1);
    assert!(listing["cases"]
        .as_array()
        .unwrap()
        .iter()
        .any(|k| k == "iris_3b_depth/iris_3b_depth/q8/768/depth/3"));
    assert!(listing["cases"]
        .as_array()
        .unwrap()
        .iter()
        .any(|k| k == "sdxl/illustrious_xl_v2/q4/1024/mask/4"));
}
#[cfg(not(target_os = "ios"))]
#[test]
fn campaign_public_report_host_execution_is_failure_before_allocator_or_load() {
    for diagnostic in [false, true] {
        let output = report("anima_turbo/anima_turbo/q8/768/text/4", diagnostic);
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
    let case = Case::parse("iris_3b/iris_3b/q4/768/text/3").unwrap();
    assert!(
        snapshot(Path::new("/definitely-not-a-provisioned-campaign"), &case)
            .unwrap_err()
            .contains("not admission/refusal evidence")
    );
}
