//! The iOS anchor law lives in [`mlx_gen::ios_anchor_law`] (shared with the providers' CPU
//! additivity proofs); this module binds it to the evidence table.

use crate::evidence::AnchorRow;
pub use mlx_gen::ios_anchor_law::*;
use mlx_gen::ios_pricing::{
    IosAnchorObservation, IosMemoryCoordinate, IosMemoryDomain, IosObservedCheck,
    IosObservedMetadata, IosPhase, IosPricing,
};

/// Validated whole-run PROCESS upper in the SDK's actual coordinate. A masked lifetime peak
/// remains an upper, not an accurate attained demand. Original pre-request phase numbers are
/// deliberately not copied into this different coordinate.
pub fn measured_anchor(row: &AnchorRow) -> Result<MeasuredAnchor, String> {
    Ok(MeasuredAnchor {
        domain: IosMemoryDomain::ProcessGrowth,
        coordinate: IosMemoryCoordinate::BeforeModelLoad,
        demand: row.upper_growth_at(IosMemoryCoordinate::BeforeModelLoad)?,
        phases: Default::default(),
        area: area(row.key.width, row.key.height, row.key.batch),
    })
}

/// Transport only REAL recorded metadata/checks into pricing. No predicted epochs, source-op
/// names, logical sizes or class labels manufacture witnesses. `telemetry_complete` reports
/// bounded transport integrity; provider code must still establish its complete class coverage
/// and physical/domain/coordinate correspondence. Coalesced check occurrences prove existence,
/// not distinct per-iteration resource ownership.
pub fn attach_observations(pricing: &mut IosPricing, row: &AnchorRow) -> Result<(), String> {
    if row.captures().is_empty() {
        return Err("anchor has no completed capture observations".into());
    }
    let mut observations = Vec::with_capacity(row.captures().len());
    for capture in row.captures() {
        let growth = capture.kernel_growth(IosMemoryCoordinate::BeforeModelLoad)?;
        let metadata = capture
            .sampling_coverage
            .as_ref()
            .ok_or("anchor actual metadata coverage missing")?;
        metadata.validate()?;
        let checks = capture
            .quantum_coverage
            .as_ref()
            .ok_or("anchor actual checked/context coverage missing")?;
        checks.validate(metadata)?;
        let checks = checks
            .checks
            .iter()
            .map(|check| {
                let phase = check
                    .phase
                    .as_deref()
                    .map(|key| {
                        IosPhase::from_key(key)
                            .ok_or_else(|| format!("unknown actual quantum phase {key}"))
                    })
                    .transpose()?;
                Ok(IosObservedCheck {
                    phase,
                    label: check.label.clone(),
                    context: check.context.clone(),
                })
            })
            .collect::<Result<Vec<_>, String>>()?;
        observations.push(IosAnchorObservation {
            domain: IosMemoryDomain::ProcessGrowth,
            coordinate: IosMemoryCoordinate::BeforeModelLoad,
            baseline_bytes: growth.starting_current,
            metadata: metadata
                .marks
                .iter()
                .map(|mark| IosObservedMetadata {
                    provider: mark.provider.clone(),
                    kind: mark.kind.clone(),
                    fields: mark.fields.clone(),
                })
                .collect(),
            checks,
            telemetry_complete: !metadata.truncated
                && capture
                    .quantum_coverage
                    .as_ref()
                    .is_some_and(|coverage| !coverage.truncated)
                && capture.quantum_query_gaps == 0,
        });
    }
    pricing.observations = observations;
    Ok(())
}
