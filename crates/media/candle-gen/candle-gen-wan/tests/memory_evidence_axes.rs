//! sc-20799: the Candle evidence emitter must be able to carry the E2 axes, and must refuse a video
//! probe that leaves the frame rate out.
//!
//! `candle_gen::testkit` lives behind the `testkit` feature, which this crate already enables as a
//! dev-dependency; Wan is also the concrete provider the collapse bites — Wan2.2 Ti2V-5B admits
//! *disjoint* frame menus at 16 fps and at 24 fps, so a record that pins `frames_per_second: None`
//! folds two genuinely different calibration cells (different frame counts, different peaks) into a
//! single evidence key.

use candle_gen::gen_core::{
    LoadShape, MemoryCalibrationIdentity, MemoryGeometry, MemoryMode, MemoryNumericTier,
    MemoryParityContract, MemoryParityResult, MemoryReferenceShape, MemoryStrategy,
    MemoryStrategyParameters, Precision, MEMORY_EVIDENCE_V1_PREFIX,
};
use candle_gen::testkit::{
    memory_evidence_v1_line_with_axes, MemoryEvidenceAxes, MemoryEvidenceProbe,
};

fn calibration() -> MemoryCalibrationIdentity {
    MemoryCalibrationIdentity {
        abi: candle_gen::gen_core::MEMORY_CALIBRATION_ABI,
        fingerprint: "sc-20799-wan-evidence-axes-v1".to_owned(),
        load_shape: LoadShape::EagerMaterialization,
    }
}

fn probe(frames: u32, reference_count: u32) -> MemoryEvidenceProbe<'static> {
    MemoryEvidenceProbe {
        resolved_route: "wan2_2_ti2v_5b",
        declared_calibration: calibration(),
        observed_calibration: calibration(),
        tier: MemoryNumericTier {
            precision: Precision::Bf16,
            quant: None,
            component_precision_floors: &[],
        },
        load_shape: LoadShape::EagerMaterialization,
        mode: MemoryMode::Other("image_to_video".to_owned()),
        overlay: None,
        geometry: MemoryGeometry {
            width: 832,
            height: 480,
            batch: 1,
            frames,
            reference_count,
        },
        strategy: MemoryStrategy::Resident,
        engaged_composition: vec![MemoryStrategy::Resident],
        parameters: MemoryStrategyParameters::default(),
        observed_peak_bytes: 1 << 30,
        harness_version: "sc-20799-axes-test",
        output_bytes: b"frames",
    }
}

/// The record body behind the strict `MEMORY_EVIDENCE_V1 ` line prefix.
fn payload(line: &str) -> &str {
    line.strip_prefix(MEMORY_EVIDENCE_V1_PREFIX)
        .expect("emitted line carries the strict evidence prefix")
}

fn set_required_revision_env() {
    std::env::set_var("INFERENCE_REVISION", "a".repeat(40));
    std::env::set_var("SCENEWORKS_REVISION", "b".repeat(40));
    std::env::set_var("MEMORY_MODEL_REVISION", "c".repeat(40));
    std::env::set_var("MEMORY_MODEL_INVENTORY_SHA256", "d".repeat(64));
}

#[test]
fn a_video_probe_without_a_frame_rate_is_refused_at_the_emitter() {
    set_required_revision_env();
    let panic = std::panic::catch_unwind(|| {
        memory_evidence_v1_line_with_axes(
            probe(81, 1),
            MemoryParityContract::Exact,
            MemoryParityResult::NotRun,
            MemoryEvidenceAxes::default(),
        )
    })
    .unwrap_err();
    let message = panic
        .downcast_ref::<String>()
        .cloned()
        .unwrap_or_else(|| "<non-string panic>".to_owned());
    assert!(message.contains("frames_per_second"), "{message}");
}

#[test]
fn the_two_ti2v_frame_rates_produce_distinguishable_evidence_cells() {
    set_required_revision_env();
    let line = |fps: u32, frames: u32| {
        memory_evidence_v1_line_with_axes(
            probe(frames, 1),
            MemoryParityContract::Exact,
            MemoryParityResult::NotRun,
            MemoryEvidenceAxes {
                frames_per_second: Some(fps),
                reference_shape: Some(MemoryReferenceShape::Image),
            },
        )
    };
    // The same frame count is reachable from both public menus, so geometry alone cannot separate
    // the cells — only the rate can.
    let at_16 = line(16, 97);
    let at_24 = line(24, 97);
    assert_ne!(at_16, at_24);
    let parsed: serde_json::Value = serde_json::from_str(payload(&at_16)).unwrap();
    assert_eq!(parsed["key"]["frames_per_second"], 16);
    assert_eq!(parsed["key"]["reference_shape"], "image");
    let parsed: serde_json::Value = serde_json::from_str(payload(&at_24)).unwrap();
    assert_eq!(parsed["key"]["frames_per_second"], 24);
}

#[test]
fn an_untyped_probe_keeps_the_opaque_legacy_carrier_shape() {
    set_required_revision_env();
    let line = memory_evidence_v1_line_with_axes(
        probe(1, 2),
        MemoryParityContract::Exact,
        MemoryParityResult::NotRun,
        MemoryEvidenceAxes::default(),
    );
    let parsed: serde_json::Value = serde_json::from_str(payload(&line)).unwrap();
    assert_eq!(
        parsed["key"]["reference_shape"],
        "legacy-untyped-reference-count-2"
    );
    assert_eq!(
        parsed["key"]["frames_per_second"],
        serde_json::Value::Null,
        "a still probe legitimately carries no rate"
    );
}

#[test]
fn a_typed_shape_that_contradicts_the_carrier_count_is_refused() {
    set_required_revision_env();
    let panic = std::panic::catch_unwind(|| {
        memory_evidence_v1_line_with_axes(
            probe(1, 2),
            MemoryParityContract::Exact,
            MemoryParityResult::NotRun,
            MemoryEvidenceAxes {
                frames_per_second: None,
                reference_shape: Some(MemoryReferenceShape::None),
            },
        )
    })
    .unwrap_err();
    let message = panic
        .downcast_ref::<String>()
        .cloned()
        .unwrap_or_else(|| "<non-string panic>".to_owned());
    assert!(
        message.contains("contradicts reference_count=2"),
        "{message}"
    );
}
