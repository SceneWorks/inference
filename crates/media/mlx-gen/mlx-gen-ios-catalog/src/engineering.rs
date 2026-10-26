//! Explicit target-device engineering estimates. These are NEVER physical certificates.
//!
//! Source-generated phase pressure is a predictor, not a bound on process demand. Enabling this
//! model requires approval of the response, unproved host/driver residual and preparation laws.
use crate::evidence::{AnchorOutcome, AnchorRow};
use mlx_gen::ios_pricing::{IosCertificatePremise, IosHostGrowth, IosPhase, IosPricing};
use serde::{Deserialize, Serialize};

pub const PROCESS_PRESSURE_PREMISE_ID: &str = "ios/target-process-pressure-estimate-v1";
pub const PROCESS_RESIDUAL_PREMISE_ID: &str = "ios/target-process-pressure-host-residual-v1";
pub const PROCESS_PREPARATION_PREMISE_ID: &str = "ios/target-source-preparation-estimate-v1";

/// Reviewable configuration, scoped to one hardware identity, OS build and source revision.
/// No default model or default approval exists. The runtime verifies `os_build` against sysctl.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct EngineeringProcessModel {
    pub machine: String,
    pub os_build: String,
    pub build_identity: crate::source_identity::BuildIdentity,
    /// Human authorization and the reviewed endpoint receipt identities, not a proof claim.
    pub approval_reference: String,
}

/// Identity known before component manifests, recipes or carriers are constructed. Preparation
/// uses the largest reviewed upper in this route/tier/mode, never a guessed configured graph.
#[derive(Clone, Copy, Debug)]
pub struct PreparationScope<'a> {
    pub machine: &'a str,
    pub provider: &'a str,
    pub route: &'a str,
    pub mode: &'a str,
    pub tier: &'a str,
}

/// Explicit harness/product configuration. Listing premise IDs records approval; carrying a
/// model by itself never activates it. This is optional, with no environment-dependent default.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct EngineeringPolicyConfig {
    pub model: EngineeringProcessModel,
    pub approved_premises: Vec<String>,
}
impl EngineeringPolicyConfig {
    pub fn into_policy(self) -> Result<crate::PremisePolicy, String> {
        self.model.validate()?;
        for id in [
            PROCESS_PRESSURE_PREMISE_ID,
            PROCESS_RESIDUAL_PREMISE_ID,
            PROCESS_PREPARATION_PREMISE_ID,
        ] {
            if !self.approved_premises.iter().any(|approved| approved == id) {
                return Err(format!(
                    "engineering policy does not explicitly approve {id}"
                ));
            }
        }
        Ok(crate::PremisePolicy::approving(self.approved_premises)
            .with_engineering_process(self.model))
    }
}

impl EngineeringProcessModel {
    pub fn validate(&self) -> Result<(), String> {
        if self.machine.is_empty()
            || self.os_build.is_empty()
            || self.approval_reference.is_empty()
            || !self
                .build_identity
                .same_production_execution(&self.build_identity)
        {
            return Err("engineering process model needs hardware, OS build, verified executable/native identity and approval provenance".into());
        }
        Ok(())
    }

    pub fn validate_runtime(
        &self,
        machine: &str,
        os_build: &str,
        build_identity: &crate::source_identity::BuildIdentity,
    ) -> Result<(), String> {
        self.validate()?;
        if self.machine != machine
            || self.os_build != os_build
            || !self
                .build_identity
                .same_production_execution(build_identity)
        {
            return Err("engineering process model hardware/OS/source revision differs from the executing build".into());
        }
        Ok(())
    }

    pub(crate) fn validate_row(&self, row: &AnchorRow) -> Result<(), String> {
        if row.key.machine != self.machine {
            return Err("engineering estimate has a foreign hardware anchor".into());
        }
        match &row.outcome {
            AnchorOutcome::Completed { captures }
                if !captures.is_empty()
                    && captures.iter().all(|c| !c.build_sha.is_empty() && c.os_build.as_deref()==Some(self.os_build.as_str()) && c.build_identity.as_ref().is_some_and(|identity|self.build_identity.same_production_execution(identity))) =>
            {
                Ok(())
            }
            AnchorOutcome::Refused { build_identity:Some(identity), build_sha, os_build:Some(os_build), .. } if !build_sha.is_empty() && os_build==&self.os_build && self.build_identity.same_production_execution(identity) => {
                Ok(())
            }
            _ => Err(
                "engineering estimate anchor executable source/native identity differs from its approved model"
                    .into(),
            ),
        }
    }

    /// Preparation-interval physical UPPER: lifetime highwater at BeforeModelLoad minus the
    /// earlier BeforeRequestConstruction current baseline. An earlier highwater can mask the
    /// interval; it still bounds its peak and is never called attained demand or variability.
    /// This excludes later render growth and avoids charging a whole render as parser scratch.
    pub fn preparation_estimate(&self, optimized: &AnchorRow) -> Result<u64, String> {
        self.validate()?;
        self.validate_row(optimized)?;
        let mut upper = None;
        for capture in optimized.captures() {
            let (baseline, old_highwater) = capture
                .before_request_snapshot
                .ok_or("preparation lacks BeforeRequestConstruction snapshot")?
                .complete_pair()?;
            let (current, preparation_highwater) = capture
                .before_model_load_snapshot
                .ok_or("preparation lacks BeforeModelLoad snapshot")?
                .complete_pair()?;
            if baseline != capture.baseline_footprint_bytes
                || preparation_highwater < old_highwater
                || preparation_highwater < current
            {
                return Err("incompatible preparation interval kernel snapshots".into());
            }
            let growth = preparation_highwater
                .checked_sub(baseline)
                .ok_or("preparation peak precedes baseline")?;
            upper = Some(upper.map_or(growth, |old: u64| old.max(growth)));
        }
        let (_, margin) = optimized.exact_growth_and_margin_at(
            mlx_gen::ios_pricing::IosMemoryCoordinate::BeforeModelLoad,
        )?;
        upper
            .ok_or("preparation has no completed capture")?
            .checked_add(margin)
            .ok_or_else(|| "engineering source preparation envelope overflows bytes".into())
    }

    /// Exact endpoint pressure normalization. The floor of one preserves the entire measured
    /// demand (including fixed overhead) for smaller pressures. No resident subtraction, phase
    /// borrowing, underprediction tolerance, native-counter conversion, or lower credit occurs.
    pub fn estimate(
        &self,
        optimized: &AnchorRow,
        optimized_price: &IosPricing,
        request_price: &IosPricing,
    ) -> Result<(u64, u64), String> {
        self.validate()?;
        self.validate_row(optimized)?;
        let (measured, margin) = optimized.exact_growth_and_margin_at(
            mlx_gen::ios_pricing::IosMemoryCoordinate::BeforeModelLoad,
        )?;
        Ok((
            pressure_estimate(measured, optimized_price, request_price)?,
            margin,
        ))
    }

    /// A refused resident is endpoint provenance, never a measured demand. A completed resident
    /// tests the empirical response with zero tolerated underprediction. Neither outcome proves
    /// that an unobserved intermediate cannot exceed the model; that is the approved premise.
    pub fn validate_resident(
        &self,
        optimized: &AnchorRow,
        optimized_price: &IosPricing,
        resident: &AnchorRow,
        resident_price: Option<&IosPricing>,
    ) -> Result<(), String> {
        self.validate_row(resident)?;
        if !same_pressure_family(&optimized.key, &resident.key) {
            return Err("engineering endpoint families differ".into());
        }
        if matches!(resident.outcome, AnchorOutcome::Refused { .. }) {
            return Ok(());
        }
        let (predicted, _) = self.estimate(
            optimized,
            optimized_price,
            resident_price.ok_or("completed resident lacks source pricing")?,
        )?;
        let (measured, _) = resident.exact_growth_and_margin_at(
            mlx_gen::ios_pricing::IosMemoryCoordinate::BeforeModelLoad,
        )?;
        if predicted < measured {
            return Err(format!("engineering process estimate under-predicts completed resident: {predicted} B estimated vs {measured} B measured"));
        }
        Ok(())
    }

    pub fn premises(
        &self,
        optimized: &AnchorRow,
        resident: &AnchorRow,
        host: &IosHostGrowth,
    ) -> Vec<IosCertificatePremise> {
        let endpoint = if matches!(resident.outcome, AnchorOutcome::Refused { .. }) {
            "resident refused before allocation; no second numeric endpoint validates response"
        } else {
            "completed resident checked with zero underprediction tolerance"
        };
        let captures = optimized
            .captures()
            .iter()
            .map(|capture| capture.run_id.as_str())
            .collect::<Vec<_>>()
            .join(",");
        vec![
            IosCertificatePremise { id: PROCESS_PREPARATION_PREMISE_ID.into(), statement: format!("ENGINEERING preparation estimate, not source allocation proof: source inventory verification, replay/config/header parsing and tokenizer/key preparation fit the optimized capture's preparation-interval physical UPPER (kernel lifetime highwater at BeforeModelLoad minus current footprint at BeforeRequestConstruction), plus the optimized render repeat-spread margin floored at 32 MiB. Earlier highwater can mask this upper; it is not claimed as attained preparation demand or measured preparation variability. The interval ends before model load, not at render completion. The actual guard checks this envelope before parser work; new metadata growth remains an empirical assumption. Capture runs={captures}; machine={}, OS={}, source={}; authorization={}", self.machine,self.os_build,self.build_identity.source_closure_sha256,self.approval_reference) },
            IosCertificatePremise { id: PROCESS_PRESSURE_PREMISE_ID.into(), statement: format!("ENGINEERING ESTIMATE, not source-proved physical bound: D_request <= D_optimized_exact_max * max(1, max_phase(pressure_request/pressure_optimized)), rounded upward; pressure is the source-generated resident+scratch phase sum for each actual registered solver/scheduler/guidance/memory setting; sharing across those settings is empirical only, never native certificate authority; optimized repeat spread floored at 32 MiB is added once. {endpoint}. Scope machine={}, OS={}, source={}; authorization/receipts: {}", self.machine,self.os_build,self.build_identity.source_closure_sha256,self.approval_reference) },
            IosCertificatePremise { id: PROCESS_RESIDUAL_PREMISE_ID.into(), statement: format!("ENGINEERING residual assumption, not HOST proof: ALL Rust host, thread stack, allocator/cache and driver process charges obey the same measured process-pressure envelope; no absent host bound is zeroed or certified. Source host status: {host:?}. A refused resident supplies no residual upper; live pre-allocation headroom checks remain required. Scope/authorization: {}", self.approval_reference) },
        ]
    }
}

/// The explicitly empirical sharing domain. Native certificate families retain their exact
/// sampling classes. Here actual factory/settings changes are re-priced, while architecture,
/// tier, true input roles and the configured graph stay fixed. Exact artifact layouts are also
/// checked against current sources before an estimate is sealed.
pub fn same_pressure_family(
    a: &crate::evidence::EvidenceKey,
    b: &crate::evidence::EvidenceKey,
) -> bool {
    let retirement = mlx_gen_sana::ios_pricing::NATIVE_RETIREMENT_EXECUTION_SUFFIX;
    a.machine == b.machine
        && a.provider == b.provider
        && a.route == b.route
        && a.mode == b.mode
        && a.architecture_sha256 == b.architecture_sha256
        && a.tier == b.tier
        && a.configured_variant_sha256 == b.configured_variant_sha256
        && (a.reference_count, a.mask_count, a.control_count)
            == (b.reference_count, b.mask_count, b.control_count)
        && a.execution.ends_with(retirement) == b.execution.ends_with(retirement)
}

pub(crate) fn pressure_family_anchors<'a>(
    rows: &'a [AnchorRow],
    key: &crate::evidence::EvidenceKey,
) -> Result<(Option<&'a AnchorRow>, Option<&'a AnchorRow>), String> {
    // Prefer an already captured exact native family. Otherwise use the lexically first actual
    // captured representative, keeping its resident/optimized recipe pair together. Recording
    // run identities in the premises makes this choice auditable, never a native-family merge.
    let family = key.family()?;
    let exact = crate::evidence::family_anchors(rows, &family);
    if exact
        .1
        .is_some_and(|row| same_pressure_family(&row.key, key))
    {
        return Ok(exact);
    }
    let optimized = rows
        .iter()
        .filter(|row| {
            row.anchor_kind == crate::evidence::AnchorKind::Optimized
                && same_pressure_family(&row.key, key)
        })
        .min_by(|a, b| a.key.execution.cmp(&b.key.execution));
    let resident = match optimized {
        Some(optimized) => {
            let family = optimized.key.family()?;
            rows.iter().find(|row| {
                row.anchor_kind == crate::evidence::AnchorKind::Resident
                    && same_pressure_family(&row.key, key)
                    && row.key.family().as_ref() == Ok(&family)
            })
        }
        None => None,
    };
    Ok((resident, optimized))
}

fn pressure(pricing: &IosPricing, phase: IosPhase) -> Result<Option<u64>, String> {
    let mut found = pricing.phases.iter().filter(|price| price.phase == phase);
    let value = match found.next() {
        Some(price) => Some(
            price
                .resident_bytes
                .checked_add(price.scratch_bytes)
                .filter(|p| *p > 0)
                .ok_or("invalid/overflowing phase pressure")?,
        ),
        None if pricing.absent_phases.contains_key(&phase) => None,
        None => {
            return Err(format!(
                "engineering estimate has an unaccounted {} phase",
                phase.as_key()
            ))
        }
    };
    if found.next().is_some() {
        return Err("engineering estimate has duplicate phase summaries".into());
    }
    Ok(value)
}

fn pressure_estimate(
    measured: u64,
    optimized: &IosPricing,
    request: &IosPricing,
) -> Result<u64, String> {
    let mut estimate = measured;
    for phase in IosPhase::ALL {
        let base = pressure(optimized, phase)?;
        let Some(next) = pressure(request, phase)? else {
            continue;
        };
        let base = base.ok_or_else(|| {
            format!(
                "engineering estimate cannot borrow an unobserved {} phase",
                phase.as_key()
            )
        })?;
        let scaled = u128::from(measured)
            .checked_mul(u128::from(next))
            .ok_or("engineering pressure multiplication overflow")?
            .div_ceil(u128::from(base));
        estimate = estimate.max(
            u64::try_from(scaled).map_err(|_| "engineering pressure estimate overflows bytes")?,
        );
    }
    Ok(estimate)
}

#[cfg(test)]
pub(crate) fn test_identity() -> crate::source_identity::BuildIdentity {
    crate::source_identity::BuildIdentity {
        source_closure_sha256: "a".repeat(64),
        backend: "mlx".into(),
        target: "aarch64-apple-ios".into(),
        rustc: "fixture-rustc".into(),
        features: String::new(),
        build_config_sha256: "c".repeat(64),
        native_mlx_fingerprint: "d".repeat(64),
    }
}

/// The same reviewed DATA configuration is compiled into SDK and campaign. An absent model
/// grants no empirical authority; malformed configuration refuses rather than silently defaults.
pub fn reviewed_policy() -> Result<crate::PremisePolicy, String> {
    policy_from_json(include_str!("../evidence/ios_engineering_policy.json"))
}
fn policy_from_json(json: &str) -> Result<crate::PremisePolicy, String> {
    let config: Option<EngineeringPolicyConfig> = serde_json::from_str(json)
        .map_err(|error| format!("invalid reviewed iOS engineering policy: {error}"))?;
    match config {
        Some(config) => config.into_policy(),
        None => Ok(crate::PremisePolicy::default()),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    fn profile(scale: u64) -> IosPricing {
        let mut profile = IosPricing::default();
        for phase in IosPhase::ALL {
            profile.take(phase, 100, scale, "fixture");
        }
        profile
    }
    #[test]
    fn smaller_pressure_keeps_measured_demand_and_larger_pressure_rounds_up() {
        assert_eq!(
            pressure_estimate(1001, &profile(100), &profile(0)).unwrap(),
            1001
        );
        assert_eq!(
            pressure_estimate(1001, &profile(100), &profile(101)).unwrap(),
            1007
        );
        assert_eq!(
            pressure_estimate(u64::MAX, &profile(0), &profile(100)).unwrap_err(),
            "engineering pressure estimate overflows bytes"
        );
    }
    #[test]
    fn missing_duplicate_or_new_phases_never_borrow_an_anchor_residual() {
        let mut missing = profile(100);
        missing.phases.pop();
        assert!(pressure_estimate(1, &missing, &profile(100)).is_err());
        missing.phase_not_run(IosPhase::Decode, "did not run");
        assert!(pressure_estimate(1, &missing, &profile(100)).is_err());
        let mut duplicate = profile(100);
        duplicate.phases.push(duplicate.phases[0].clone());
        assert!(pressure_estimate(1, &profile(100), &duplicate).is_err());
    }
    #[test]
    fn absent_reviewed_model_has_no_implicit_approval() {
        assert!(policy_from_json("null")
            .unwrap()
            .engineering_process()
            .is_none());
        assert!(policy_from_json("{}").is_err());
    }

    /// Source-only validation after reviewed DATA is filled. Device compatibility is checked
    /// again by the actual SDK/campaign runtime; this host test never relabels a phone receipt.
    #[test]
    fn reviewed_policy_and_anchor_receipts_have_matching_scope() {
        let policy = reviewed_policy().unwrap();
        let rows = crate::evidence::rows().unwrap();
        if let Some(model) = policy.engineering_process() {
            let mut matching = 0;
            for row in rows.iter().filter(|row| row.key.machine == model.machine) {
                model.validate_row(row).unwrap();
                matching += 1;
                if row.anchor_kind == crate::evidence::AnchorKind::Optimized {
                    model.preparation_estimate(row).unwrap();
                }
            }
            assert!(
                matching > 0,
                "reviewed engineering policy requires actual scoped receipts"
            );
        }
    }

    #[test]
    fn scope_is_explicit_and_invalidates_on_os_hardware_or_source_change() {
        let model = EngineeringProcessModel {
            machine: "phone".into(),
            os_build: "23G90".into(),
            build_identity: test_identity(),
            approval_reference: "reviewed receipts and authorization".into(),
        };
        model
            .validate_runtime("phone", "23G90", &test_identity())
            .unwrap();
        let mut other_source = test_identity();
        other_source.source_closure_sha256 = "b".repeat(64);
        assert!(model
            .validate_runtime("other", "23G90", &test_identity())
            .is_err());
        assert!(model
            .validate_runtime("phone", "23G91", &test_identity())
            .is_err());
        assert!(model
            .validate_runtime("phone", "23G90", &other_source)
            .is_err());
    }
}
