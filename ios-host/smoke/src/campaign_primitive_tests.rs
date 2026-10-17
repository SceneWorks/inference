use super::*;
use std::cell::Cell;

/// Injected device: fixed headroom, scripted readings, and a record of every side effect.
struct Fixture {
    available: Option<u64>,
    before: Reading,
    after: Reading,
    reads: usize,
    resets: Cell<usize>,
    allocations: Cell<usize>,
    fail: Option<String>,
}
impl Fixture {
    fn new(available: Option<u64>) -> Self {
        Self {
            available,
            before: Reading {
                available_bytes: available,
                footprint_bytes: Some(100),
                lifetime_peak_bytes: Some(200),
                ..Reading::default()
            },
            after: Reading {
                available_bytes: available,
                footprint_bytes: Some(150),
                lifetime_peak_bytes: Some(1000),
                mlx_active_bytes: 0,
                mlx_cache_bytes: 0,
                mlx_peak_bytes: 300,
            },
            reads: 0,
            resets: Cell::new(0),
            allocations: Cell::new(0),
            fail: None,
        }
    }
}
impl ProbeDevice for Fixture {
    fn read(&mut self) -> Reading {
        self.reads += 1;
        let mut reading = if self.reads == 1 {
            self.before
        } else {
            self.after
        };
        reading.available_bytes = self.available;
        reading
    }
    fn reset_mlx_peak(&mut self) {
        self.resets.set(self.resets.get() + 1);
    }
    fn allocate_and_run(&mut self, _: &Plan) -> Result<Reading, String> {
        self.allocations.set(self.allocations.get() + 1);
        match &self.fail {
            Some(reason) => Err(reason.clone()),
            None => Ok(Reading {
                footprint_bytes: Some(130),
                ..self.before
            }),
        }
    }
}

#[test]
fn requirement_is_the_budget_modules_sana_block_floor_plus_reserve() {
    let plan = plan().unwrap();
    let cfg = mlx_gen_sana::SanaTransformerConfig::sana_1600m();
    let side = (EDGE / mlx_gen_sana::pipeline::SPATIAL_SCALE) as i32;
    let geometry = mlx_gen_sana::transformer::configured_block_geometry(
        &cfg,
        1,
        side,
        side,
        mlx_gen_sana::text_encoder::MAX_SEQUENCE_LENGTH,
    );
    assert_eq!(plan.geometry, geometry);
    assert_eq!(plan.scratch_bytes, ios_quantum::sana_block_bytes(geometry));
    assert_eq!(plan.reserve_bytes, PROCESS_RESERVE_BYTES);
    // Residents are the probe's own tensors at their stored dtypes: f32 activation, F16 weight.
    let elements = |shape: &[i32; 4]| shape.iter().map(|d| *d as u64).product::<u64>();
    assert_eq!(
        plan.resident_bytes,
        4 * elements(&plan.activation_shape) + 2 * elements(&plan.weight_shape)
    );
    assert_eq!(
        plan.required_bytes,
        plan.scratch_bytes + plan.resident_bytes + PROCESS_RESERVE_BYTES
    );
    // Shapes are the block's own: [1, H, W, dim] activation, [2*hidden, 1, 1, dim] weight.
    assert_eq!(plan.activation_shape, [1, side, side, cfg.inner_dim()]);
    let hidden = (cfg.mlp_ratio * cfg.inner_dim() as f32) as i32;
    assert_eq!(plan.weight_shape, [2 * hidden, 1, 1, cfg.inner_dim()]);
    assert_eq!(geometry.tokens, (side * side) as u64);
    // The floor charges the FULL f32 promotion of every GLUMBConv tensor, which strictly exceeds
    // the probe's own promoted weight copy.
    let probe_weight_elements = plan.weight_shape.iter().map(|d| *d as u64).product::<u64>();
    assert!(plan.scratch_bytes > 4 * probe_weight_elements);
}

#[test]
fn insufficient_or_unavailable_headroom_refuses_before_any_allocation() {
    let plan = plan().unwrap();
    let required = plan.required_bytes;
    // `scratch + reserve` alone is the window where the inner live recheck would refuse only
    // AFTER residents were allocated; the pre-check must close it.
    let without_residents = plan.scratch_bytes + plan.reserve_bytes;
    for available in [None, Some(0), Some(without_residents), Some(required - 1)] {
        let mut device = Fixture::new(available);
        let row = characterize(&mut device, true);
        assert_eq!(device.allocations.get(), 0, "{available:?}");
        assert_eq!(device.resets.get(), 0, "{available:?}");
        assert_eq!(row["status"], "diagnostic_failure");
        assert_eq!(row["refused_before_allocation"], true);
        assert_eq!(row["allocation_attempted"], false);
        assert!(row["reason"]
            .as_str()
            .unwrap()
            .contains(&required.to_string()));
        assert!(row.get("after").is_none());
    }
    // Exactly the requirement is admitted for the probe (>= comparison, same as the guard).
    let mut device = Fixture::new(Some(required));
    assert_eq!(
        characterize(&mut device, true)["status"],
        "diagnostic_success"
    );
    assert_eq!(device.allocations.get(), 1);
}

#[test]
fn non_diagnostic_launch_refuses_before_reading_or_allocating() {
    let mut device = Fixture::new(Some(u64::MAX));
    let row = characterize(&mut device, false);
    assert_eq!(row["status"], "diagnostic_failure");
    assert_eq!((device.reads, device.allocations.get()), (0, 0));
}

#[test]
fn success_row_schema_is_primitive_characterization_never_admission() {
    let mut device = Fixture::new(Some(u64::MAX));
    let row = characterize(&mut device, true);
    assert_eq!(row["status"], "diagnostic_success");
    assert_eq!(row["evidence_class"], EVIDENCE_CLASS);
    assert_eq!(row["model_admission_evidence"], false);
    assert_eq!(row["production_success"], false);
    assert_eq!(row["cap_validated"], false);
    assert_eq!(row["diagnostic"], true);
    assert_eq!(row["allocation_attempted"], true);
    assert_eq!(device.resets.get(), 1);
    for phase in ["before", "resident", "after"] {
        for key in [
            "available_bytes",
            "footprint_bytes",
            "kernel_lifetime_peak_bytes",
            "mlx_active_bytes",
            "mlx_cache_bytes",
            "mlx_peak_bytes",
        ] {
            assert!(row[phase].get(key).is_some(), "{phase}.{key}");
        }
    }
    assert_eq!(row["resident"]["footprint_bytes"], 130);
    let primitive = &row["primitive"];
    assert_eq!(primitive["weight_dtype"], "float16");
    assert_eq!(primitive["activation_dtype"], "float32");
    assert_eq!(
        primitive["required_headroom_bytes"],
        plan().unwrap().required_bytes
    );
    // New lifetime peak 1000 over a 100-byte baseline footprint; MLX accounted 300 of it.
    assert_eq!(row["quantum_peak_over_baseline_bytes"], 900);
    assert_eq!(row["unaccounted_by_mlx_peak_bytes"], 600);
}

#[test]
fn a_quantum_masked_by_an_earlier_lifetime_peak_reports_null_not_zero() {
    let mut device = Fixture::new(Some(u64::MAX));
    device.after.lifetime_peak_bytes = device.before.lifetime_peak_bytes;
    let row = characterize(&mut device, true);
    assert_eq!(row["status"], "diagnostic_success");
    assert!(row["quantum_peak_over_baseline_bytes"].is_null());
    assert!(row["unaccounted_by_mlx_peak_bytes"].is_null());
    assert!(row["reason"].as_str().unwrap().contains("unresolved"));
}

#[test]
fn allocation_failure_is_a_diagnostic_failure_with_an_after_reading() {
    let mut device = Fixture::new(Some(u64::MAX));
    device.fail = Some("backend refused".into());
    let row = characterize(&mut device, true);
    assert_eq!(row["status"], "diagnostic_failure");
    assert_eq!(row["reason"], "backend refused");
    assert_eq!(row["allocation_attempted"], true);
    assert!(row.get("after").is_some());
}

#[cfg(not(target_os = "ios"))]
#[test]
fn host_public_report_is_primitive_labelled_and_never_allocates() {
    let output = super::super::report(PROBE_KEY, true);
    assert!(output.starts_with("PRIMITIVE: diagnostic_failure\n"));
    assert!(!output.contains("SMOKE: PASS"));
    let row: Value = serde_json::from_str(
        output
            .lines()
            .find_map(|l| l.strip_prefix("CAMPAIGN_JSON "))
            .unwrap(),
    )
    .unwrap();
    assert_eq!(row["case"], PROBE_KEY);
    assert_eq!(row["evidence_class"], EVIDENCE_CLASS);
    assert_eq!(row["model_admission_evidence"], false);
    assert_eq!(row["production_success"], false);
    assert_eq!(row["allocation_attempted"], false);
    assert!(row["reason"].as_str().unwrap().contains("real iOS device"));
}

#[test]
fn probe_touches_no_model_files_and_is_not_a_catalog_case() {
    let source = include_str!("campaign_primitive.rs");
    for forbidden in [
        "snapshot",
        "LoadSpec",
        "WeightsSource",
        "safetensors",
        "join(\"campaign\")",
        "load_sana",
        "File::open",
        "read_dir",
    ] {
        assert!(!source.contains(forbidden), "{forbidden}");
    }
    // The only filesystem write is the empty render manifest.
    assert_eq!(source.matches("std::fs::").count(), 1);
    let listing = super::super::listing().unwrap();
    assert!(!listing.contains(PROBE_KEY));
    assert!(crate::campaign::Case::parse(PROBE_KEY).is_err());
}

#[cfg(target_os = "macos")]
#[path = "../../../crates/media/mlx-gen/mlx-gen-iris/tests/common/metal_guard.rs"]
mod metal_guard;

/// The REAL primitive on the MLX CPU device (Metal trapped), through the engine's guard API with
/// deterministic fixture headroom: exactly-required runs and promotes; one byte short allocates
/// nothing. Runs in a child process so the CPU default device cannot leak into other tests.
#[cfg(target_os = "macos")]
#[test]
fn cpu_real_primitive_runs_at_the_requirement_and_allocates_nothing_below_it() {
    const CHILD: &str = "IOS_PRIMITIVE_CPU_CHILD";
    const NAME: &str = "campaign_media::primitive::tests::cpu_real_primitive_runs_at_the_requirement_and_allocates_nothing_below_it";
    if std::env::var_os(CHILD).is_none() {
        let status = std::process::Command::new(std::env::current_exe().unwrap())
            .args(["--exact", NAME, "--nocapture", "--test-threads", "1"])
            .env(CHILD, "1")
            .status()
            .unwrap();
        assert!(status.success());
        return;
    }
    mlx_rs::Device::set_default(&mlx_rs::Device::cpu());
    assert!(metal_guard::forbid_metal());
    assert!(metal_guard::metal_forbidden());
    let plan = plan().unwrap();
    use ios_quantum::QuantumEvent;
    use std::{cell::RefCell, rc::Rc};
    let observe = |headroom: u64| {
        let events = Rc::new(RefCell::new(Vec::new()));
        let sink = events.clone();
        let row = ios_quantum::with_cpu_diagnostic_headroom(headroom, || {
            ios_quantum::with_cpu_diagnostic_observer(
                move |event| sink.borrow_mut().push(event.clone()),
                || {
                    let active = mlx_rs::memory::get_active_memory();
                    let row = characterize(&mut Device, true);
                    if row["allocation_attempted"] == false {
                        assert_eq!(mlx_rs::memory::get_active_memory(), active);
                    }
                    Ok(row)
                },
            )
        })
        .unwrap();
        let events = events.borrow().clone();
        (row, events)
    };
    let (below, events) = observe(plan.required_bytes - 1);
    assert_eq!(below["refused_before_allocation"], true);
    assert_eq!(below["allocation_attempted"], false);
    assert!(events.is_empty(), "{events:?}");
    let (row, events) = observe(plan.required_bytes);
    assert_eq!(row["status"], "diagnostic_success", "{row}");
    // The quantum went through the engine's guard API: exactly one pre-construction check for
    // the block floor and one stream synchronization, both under the probe's label.
    let checked: Vec<_> = events
        .iter()
        .filter(|e| matches!(e, QuantumEvent::Checked { .. }))
        .collect();
    assert_eq!(
        checked,
        [&QuantumEvent::Checked {
            label: LABEL.into(),
            scratch: plan.scratch_bytes
        }],
        "{events:?}"
    );
    let synchronized = events
        .iter()
        .filter(|e| matches!(e, QuantumEvent::Synchronized { label, .. } if label == LABEL))
        .count();
    assert_eq!(synchronized, 1, "{events:?}");
    // F16 weight resident at its half size, then the quantum's peak includes the F32 promotion.
    let weight_elements = plan.weight_shape.iter().map(|d| *d as u64).product::<u64>();
    let resident = row["resident"]["mlx_active_bytes"].as_u64().unwrap();
    assert!(resident >= 2 * weight_elements, "{row}");
    let peak = row["after"]["mlx_peak_bytes"].as_u64().unwrap();
    assert!(peak >= 2 * weight_elements + 4 * weight_elements, "{row}");
    // Scope drained the cache and released every array.
    assert_eq!(row["after"]["mlx_cache_bytes"], 0, "{row}");
}
