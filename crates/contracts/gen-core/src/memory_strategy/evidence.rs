//! Calibration evidence keys, verdicts, log records, and their canonical JSON.

use super::*;

/// Five catalog conformance states from epic SC-15448.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum MemoryConformanceState {
    Verified,
    ImplementedUnverified,
    StructurallyNotApplicable,
    Missing,
    RouteUnavailableOrBroken,
}

/// Six independent evidence dimensions from epic SC-15448.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum MemoryEvidenceDimension {
    StaticImplementation,
    DeclaredCalibration,
    HistoricalVerification,
    CurrentEnvironmentVerification,
    CanonicalRouteLoadability,
    ExactStrategyParameters,
}

impl MemoryEvidenceDimension {
    pub const ALL: [Self; 6] = [
        Self::StaticImplementation,
        Self::DeclaredCalibration,
        Self::HistoricalVerification,
        Self::CurrentEnvironmentVerification,
        Self::CanonicalRouteLoadability,
        Self::ExactStrategyParameters,
    ];
}

/// Why one evidence dimension does or does not cover a request.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum MemoryEvidenceVerdict {
    Satisfied,
    Missing,
    Unverified,
    Stale,
    FingerprintMismatch,
    CompositionMismatch,
    OutOfEnvelope,
    Invalid,
}

/// Explicit six-dimensional evidence result.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct MemoryEvidenceDimensions {
    pub static_implementation: MemoryEvidenceVerdict,
    pub declared_calibration: MemoryEvidenceVerdict,
    pub historical_verification: MemoryEvidenceVerdict,
    pub current_environment_verification: MemoryEvidenceVerdict,
    pub canonical_route_loadability: MemoryEvidenceVerdict,
    pub exact_strategy_parameters: MemoryEvidenceVerdict,
}

impl MemoryEvidenceDimensions {
    pub const VERIFIED: Self = Self {
        static_implementation: MemoryEvidenceVerdict::Satisfied,
        declared_calibration: MemoryEvidenceVerdict::Satisfied,
        historical_verification: MemoryEvidenceVerdict::Satisfied,
        current_environment_verification: MemoryEvidenceVerdict::Satisfied,
        canonical_route_loadability: MemoryEvidenceVerdict::Satisfied,
        exact_strategy_parameters: MemoryEvidenceVerdict::Satisfied,
    };

    pub fn verdict(self, dimension: MemoryEvidenceDimension) -> MemoryEvidenceVerdict {
        match dimension {
            MemoryEvidenceDimension::StaticImplementation => self.static_implementation,
            MemoryEvidenceDimension::DeclaredCalibration => self.declared_calibration,
            MemoryEvidenceDimension::HistoricalVerification => self.historical_verification,
            MemoryEvidenceDimension::CurrentEnvironmentVerification => {
                self.current_environment_verification
            }
            MemoryEvidenceDimension::CanonicalRouteLoadability => self.canonical_route_loadability,
            MemoryEvidenceDimension::ExactStrategyParameters => self.exact_strategy_parameters,
        }
    }

    pub fn all_satisfied(self) -> bool {
        MemoryEvidenceDimension::ALL
            .into_iter()
            .all(|dimension| self.verdict(dimension) == MemoryEvidenceVerdict::Satisfied)
    }
}

/// Fully-qualified key for one evidence cell.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct MemoryEvidenceKey {
    /// Resolved catalog family.  Engines shared by two catalog families must not share evidence.
    pub model_family: String,
    pub resolved_route: String,
    pub backend: MemoryBackend,
    pub tier: MemoryNumericTier,
    /// Exact intra-phase materialization shape measured by this evidence cell.
    pub load_shape: LoadShape,
    pub mode: MemoryMode,
    /// Carrier shape for the exact number of references in [`Self::geometry`].
    pub reference_shape: MemoryReferenceShape,
    pub overlay: Option<String>,
    pub geometry: MemoryGeometry,
    /// Exact output frame rate when the request has temporal output. `None` is the non-temporal
    /// identity; it is not interchangeable with an arbitrary numeric rate.
    pub frames_per_second: Option<u32>,
    pub strategy: MemoryStrategy,
    /// Exact, canonically ordered strategy set active when this evidence was measured.
    pub engaged_composition: Vec<MemoryStrategy>,
    pub parameters: MemoryStrategyParameters,
}

impl MemoryEvidenceKey {
    fn has_canonical_engaged_composition(&self) -> bool {
        !self.engaged_composition.is_empty()
            && self
                .engaged_composition
                .windows(2)
                .all(|pair| pair[0] < pair[1])
    }

    fn validation_errors(&self) -> Vec<String> {
        let mut errors = Vec::new();
        if self.model_family.trim().is_empty() {
            errors.push("evidence model family must be non-empty".to_owned());
        }
        if self.reference_shape.is_none() != (self.geometry.reference_count == 0) {
            errors.push(
                "evidence reference shape must be none exactly when reference count is zero"
                    .to_owned(),
            );
        }
        if matches!(&self.reference_shape, MemoryReferenceShape::Other(shape) if shape.trim().is_empty())
        {
            errors.push("evidence other reference shape must be non-empty".to_owned());
        }
        if self.frames_per_second == Some(0) {
            errors.push("evidence frame rate must be positive when present".to_owned());
        }
        errors
    }
}

/// Numerical parity contract for an evidence record.
#[derive(Clone, Debug, PartialEq)]
pub enum MemoryParityContract {
    Exact,
    Tolerance {
        metric: String,
        maximum_error: f64,
    },
    Golden {
        fixture: String,
        metric: String,
        maximum_error: f64,
    },
}

/// Result of executing the declared numerical contract.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum MemoryParityResult {
    Passed,
    Failed { reason: String },
    NotRun,
}

/// Dynamic/static evidence handshake populated and consumed by caller-side admission selectors.
#[derive(Clone, Debug, PartialEq)]
pub struct MemoryEvidence {
    pub key: MemoryEvidenceKey,
    pub conformance: MemoryConformanceState,
    pub dimensions: MemoryEvidenceDimensions,
    pub calibration_abi: u32,
    pub calibration_fingerprint: String,
    pub sceneworks_revision: String,
    pub inference_revision: String,
    pub harness_version: String,
    /// Absolute request peak **live allocation** the formula predicts — bytes held by the request's
    /// live tensors at the high-water mark, including request-resident assets but excluding any
    /// allocator reuse cache. A caller converts this to `MemoryRunContext::predicted_peak_bytes` by
    /// subtracting exactly the request-resident bytes already charged in its budget snapshot. See
    /// [`observed_peak_bytes`](Self::observed_peak_bytes) for why the distinction is load-bearing
    /// and not pedantic.
    pub predicted_peak_bytes: u64,
    /// Absolute request peak **live allocation** measured by the calibration harness. `None` until
    /// measured; it uses the same baseline as `predicted_peak_bytes` above.
    ///
    /// **Live allocation, not process footprint, and the two are not close.** An allocator's
    /// freed-buffer reuse cache is excluded here and counted by the OS: a Z-Image 1024² render
    /// measured 3102 MiB of live peak and 6488 MiB of `active + cache` on the same run. Comparing the
    /// wrong one against a cap is not conservative in a predictable direction — of three
    /// configurations measured together, the one with the **lowest** live peak had the **highest**
    /// footprint and was the only one an iOS device refused. A ladder built on the wrong quantity can
    /// rank a fatal configuration as the safest of its set.
    ///
    /// Live allocation is the right quantity *here* because it is the irreducible working set — the
    /// part no tuning removes. The cache is modelled on the other side of the comparison, by
    /// [`MemoryBudget::reclaimable_bytes`], whose docs carry the precondition that makes that
    /// modelling valid. A calibration that records footprint in this field silently double-counts
    /// the cache against a budget that already credits it.
    pub observed_peak_bytes: Option<u64>,
    pub parity: MemoryParityContract,
    pub parity_result: MemoryParityResult,
}

/// One machine-readable calibration observation emitted by a real-weight harness.
///
/// This is deliberately separate from [`MemoryEvidence`]. A harness observation is the provenance
/// input used to promote evidence; it must carry both the provider identity declared before the run
/// and the identity observed at the measurement seam. Making those two fields explicit lets the
/// verifier reject the exact stale-layout failure calibration fingerprints exist to detect.
#[derive(Clone, Debug, PartialEq)]
pub struct MemoryEvidenceLogRecord {
    pub key: MemoryEvidenceKey,
    pub declared_calibration: MemoryCalibrationIdentity,
    pub observed_calibration: MemoryCalibrationIdentity,
    pub predicted_peak_bytes: u64,
    pub observed_peak_bytes: u64,
    pub inference_revision: String,
    pub sceneworks_revision: String,
    /// Exact immutable model revision used by the probe.
    pub model_revision: String,
    /// SHA-256 of the canonical, dereferenced model-file inventory persisted with the evidence.
    pub model_inventory_sha256: String,
    pub harness_version: String,
    /// SHA-256 of the exact output bytes written by this probe. The A/B verifier binds each record
    /// to its artifact before it may promote the declared parity contract.
    pub output_sha256: String,
    pub parity: MemoryParityContract,
    pub parity_result: MemoryParityResult,
}

impl MemoryEvidenceLogRecord {
    /// Validate the persisted protocol invariants before emitting a line.
    pub fn validation_errors(&self) -> Vec<String> {
        let mut errors = Vec::new();
        errors.extend(self.key.validation_errors());
        if !self.key.has_canonical_engaged_composition() {
            errors.push(
                "evidence engaged composition must be a non-empty canonical strategy set"
                    .to_owned(),
            );
        } else if self.key.engaged_composition.first() != Some(&MemoryStrategy::Resident)
            || self.key.engaged_composition.last() != Some(&self.key.strategy)
        {
            errors.push(
                "evidence engaged composition must start with resident and end with the measured strategy"
                    .to_owned(),
            );
        }
        if self.predicted_peak_bytes == 0 {
            errors.push("predicted peak bytes must be positive".to_owned());
        }
        if self.observed_peak_bytes == 0 {
            errors.push("observed peak bytes must be positive".to_owned());
        }
        for (label, identity) in [
            ("declared", &self.declared_calibration),
            ("observed", &self.observed_calibration),
        ] {
            if identity.abi != MEMORY_CALIBRATION_ABI {
                errors.push(format!(
                    "{label} calibration ABI {} does not match contract ABI {MEMORY_CALIBRATION_ABI}",
                    identity.abi
                ));
            }
            if identity.load_shape != self.key.load_shape {
                errors.push(format!(
                    "{label} calibration load shape {:?} does not match evidence key {:?}",
                    identity.load_shape, self.key.load_shape
                ));
            }
            if let Err(reason) = validate_calibration_fingerprint(&identity.fingerprint) {
                errors.push(format!("{label} calibration fingerprint {reason}"));
            }
        }
        if self.declared_calibration != self.observed_calibration {
            errors.push("declared and observed calibration identities do not match".to_owned());
        }
        for (label, revision) in [
            ("inference", self.inference_revision.as_str()),
            ("SceneWorks", self.sceneworks_revision.as_str()),
            ("model", self.model_revision.as_str()),
        ] {
            if revision.len() != 40
                || !revision
                    .bytes()
                    .all(|byte| byte.is_ascii_digit() || (b'a'..=b'f').contains(&byte))
            {
                errors.push(format!(
                    "{label} revision must be an exact 40-character lowercase Git commit"
                ));
            }
        }
        if self.model_inventory_sha256.len() != 64
            || !self
                .model_inventory_sha256
                .bytes()
                .all(|byte| byte.is_ascii_digit() || (b'a'..=b'f').contains(&byte))
        {
            errors.push(
                "model inventory SHA-256 must be 64 lowercase hexadecimal characters".to_owned(),
            );
        }
        if self.harness_version.trim().is_empty() {
            errors.push("harness version must be non-empty".to_owned());
        }
        if self.output_sha256.len() != 64
            || !self
                .output_sha256
                .bytes()
                .all(|byte| byte.is_ascii_digit() || (b'a'..=b'f').contains(&byte))
        {
            errors.push("output SHA-256 must be 64 lowercase hexadecimal characters".to_owned());
        }
        match &self.parity {
            MemoryParityContract::Exact => {}
            MemoryParityContract::Tolerance {
                metric,
                maximum_error,
            } => validate_parity_limit(metric, *maximum_error, &mut errors),
            MemoryParityContract::Golden {
                fixture,
                metric,
                maximum_error,
            } => {
                if fixture.trim().is_empty() {
                    errors.push("golden parity fixture must be non-empty".to_owned());
                }
                validate_parity_limit(metric, *maximum_error, &mut errors);
            }
        }
        errors
    }

    /// Serialize this observation as one canonical `MEMORY_EVIDENCE_V1` line.
    pub fn to_json_line(&self) -> Result<String> {
        let errors = self.validation_errors();
        if !errors.is_empty() {
            return Err(Error::Msg(format!(
                "invalid memory evidence log record: {}",
                errors.join("; ")
            )));
        }

        let json = serde_json::json!({
            "schema_version": MEMORY_EVIDENCE_SCHEMA_VERSION,
            "key": evidence_key_json(&self.key),
            "declared_calibration": calibration_identity_json(&self.declared_calibration),
            "observed_calibration": calibration_identity_json(&self.observed_calibration),
            "predicted_peak_bytes": self.predicted_peak_bytes,
            "observed_peak_bytes": self.observed_peak_bytes,
            "inference_revision": self.inference_revision,
            "sceneworks_revision": self.sceneworks_revision,
            "model_revision": self.model_revision,
            "model_inventory_sha256": self.model_inventory_sha256,
            "harness_version": self.harness_version,
            "output_sha256": self.output_sha256,
            "parity": parity_contract_json(&self.parity),
            "parity_result": parity_result_json(&self.parity_result),
        });
        serde_json::to_string(&json)
            .map(|payload| format!("{MEMORY_EVIDENCE_V1_PREFIX}{payload}"))
            .map_err(|error| Error::Msg(format!("serialize memory evidence log record: {error}")))
    }
}

/// Validate the stable content-fingerprint grammar used at the persisted evidence boundary.
///
/// Fingerprints are lowercase kebab tokens and contain exactly one positive `vN` token. Backend,
/// mode, tier, load shape, geometry, and strategy parameters are typed evidence-key axes and must
/// not be encoded as substitutes for those fields. Existing descriptive tokens may still mention
/// implementation facts; the version token is what makes a deliberate calibration change lintable.
pub fn validate_calibration_fingerprint(fingerprint: &str) -> std::result::Result<(), String> {
    if fingerprint.is_empty() {
        return Err("must be non-empty".to_owned());
    }
    let tokens = fingerprint.split('-').collect::<Vec<_>>();
    if tokens.iter().any(|token| {
        token.is_empty()
            || !token
                .bytes()
                .all(|byte| byte.is_ascii_lowercase() || byte.is_ascii_digit())
    }) {
        return Err("must contain only non-empty lowercase ASCII kebab tokens".to_owned());
    }
    let version_tokens = tokens
        .iter()
        .filter(|token| {
            token.strip_prefix('v').is_some_and(|digits| {
                !digits.is_empty() && digits.bytes().all(|b| b.is_ascii_digit())
            })
        })
        .collect::<Vec<_>>();
    if version_tokens.len() != 1 {
        return Err("must contain exactly one vN version token".to_owned());
    }
    let digits = version_tokens[0].strip_prefix('v').unwrap_or_default();
    if digits.starts_with('0') {
        return Err(
            "version token must be positive and must not contain leading zeroes".to_owned(),
        );
    }
    Ok(())
}

fn calibration_identity_json(identity: &MemoryCalibrationIdentity) -> serde_json::Value {
    serde_json::json!({
        "abi": identity.abi,
        "fingerprint": identity.fingerprint,
        "load_shape": load_shape_key(identity.load_shape),
    })
}

pub(super) fn evidence_key_json(key: &MemoryEvidenceKey) -> serde_json::Value {
    let component_precision_floors = key
        .tier
        .component_precision_floors
        .iter()
        .map(|floor| {
            serde_json::json!({
                "component": floor.component.as_str(),
                "selected_tier": quant_key(floor.selected_tier),
                "resident_tier": quant_key(floor.resident_tier),
            })
        })
        .collect::<Vec<_>>();
    let mut value = serde_json::json!({
        "model_family": key.model_family,
        "resolved_route": key.resolved_route,
        "backend": key.backend.as_key(),
        "tier": {
            "precision": precision_key(key.tier.precision),
            "quant": key.tier.quant.map(quant_key),
            "component_precision_floors": component_precision_floors,
        },
        "load_shape": load_shape_key(key.load_shape),
        "mode": key.mode.as_key(),
        "reference_shape": key.reference_shape.as_key(),
        "overlay": key.overlay,
        "geometry": {
            "width": key.geometry.width,
            "height": key.geometry.height,
            "batch": key.geometry.batch,
            "frames": key.geometry.frames,
            "reference_count": key.geometry.reference_count,
        },
        "frames_per_second": key.frames_per_second,
        "strategy": memory_strategy_key(key.strategy),
        "engaged_composition": key.engaged_composition.iter().copied().map(memory_strategy_key).collect::<Vec<_>>(),
        "parameters": {
            "decode_tile_edge": key.parameters.decode_tile_edge,
            "decode_overlap": key.parameters.decode_overlap,
            "attention_chunk_size": key.parameters.attention_chunk_size,
            "transformer_window_size": key.parameters.transformer_window_size,
            "transformer_window_component": key.parameters.transformer_window_component.map(transformer_component_key),
        },
    });
    // Preserve the byte identity of every retained legacy evidence key.
    if let Some(staged) = key.parameters.stage_residency {
        value["parameters"]["stage_residency"] = serde_json::json!(staged);
    }
    value
}

fn parity_contract_json(parity: &MemoryParityContract) -> serde_json::Value {
    match parity {
        MemoryParityContract::Exact => serde_json::json!({ "kind": "exact" }),
        MemoryParityContract::Tolerance {
            metric,
            maximum_error,
        } => serde_json::json!({
            "kind": "tolerance",
            "metric": metric,
            "maximum_error": maximum_error,
        }),
        MemoryParityContract::Golden {
            fixture,
            metric,
            maximum_error,
        } => serde_json::json!({
            "kind": "golden",
            "fixture": fixture,
            "metric": metric,
            "maximum_error": maximum_error,
        }),
    }
}

fn parity_result_json(result: &MemoryParityResult) -> serde_json::Value {
    match result {
        MemoryParityResult::Passed => serde_json::json!({ "kind": "passed" }),
        MemoryParityResult::Failed { reason } => {
            serde_json::json!({ "kind": "failed", "reason": reason })
        }
        MemoryParityResult::NotRun => serde_json::json!({ "kind": "not_run" }),
    }
}

const fn memory_strategy_key(strategy: MemoryStrategy) -> &'static str {
    match strategy {
        MemoryStrategy::Resident => "resident",
        MemoryStrategy::StagedResidency => "staged_residency",
        MemoryStrategy::BoundedDecode => "bounded_decode",
        MemoryStrategy::BoundedAttention => "bounded_attention",
        MemoryStrategy::BoundedTransformerResidency => "bounded_transformer_residency",
    }
}

const fn transformer_component_key(component: TransformerComponent) -> &'static str {
    match component {
        TransformerComponent::Dit => "dit",
        TransformerComponent::TextEncoder => "text_encoder",
        TransformerComponent::Both => "both",
    }
}

pub(super) const fn load_shape_key(load_shape: LoadShape) -> &'static str {
    match load_shape {
        LoadShape::EagerMaterialization => "eager_materialization",
        LoadShape::DeferredMaterialization => "deferred_materialization",
    }
}

pub(super) const fn precision_key(precision: Precision) -> &'static str {
    match precision {
        Precision::Bf16 => "bf16",
        Precision::Fp32 => "fp32",
    }
}

pub(super) const fn quant_key(quant: Quant) -> &'static str {
    quant.key()
}

impl MemoryEvidence {
    pub fn predicted_peak_breakdown(
        &self,
        contract: &MemoryProviderContract,
    ) -> MemoryPeakBreakdown {
        contract.decompose_predicted_peak(self.predicted_peak_bytes)
    }

    pub fn validation_errors(&self) -> Vec<String> {
        let mut errors = Vec::new();
        if !self.key.has_canonical_engaged_composition() {
            errors.push(
                "evidence engaged composition must be a non-empty canonical strategy set"
                    .to_owned(),
            );
        }
        if self.observed_peak_bytes.is_none() {
            errors.push("verified evidence requires an observed peak".to_owned());
        }
        match &self.parity {
            MemoryParityContract::Exact => {}
            MemoryParityContract::Tolerance {
                metric,
                maximum_error,
            } => validate_parity_limit(metric, *maximum_error, &mut errors),
            MemoryParityContract::Golden {
                fixture,
                metric,
                maximum_error,
            } => {
                if fixture.trim().is_empty() {
                    errors.push("golden parity fixture must be non-empty".to_owned());
                }
                validate_parity_limit(metric, *maximum_error, &mut errors);
            }
        }
        match &self.parity_result {
            MemoryParityResult::Passed => {}
            MemoryParityResult::Failed { reason } if reason.trim().is_empty() => {
                errors.push("failed parity result requires a reason".to_owned());
            }
            MemoryParityResult::Failed { .. } | MemoryParityResult::NotRun => {}
        }
        errors
    }

    /// Only a fully verified, six-dimension record matching the current provider handshake may
    /// authorize an optimized fit.
    pub fn optimized_eligibility(
        &self,
        contract: &MemoryProviderContract,
    ) -> std::result::Result<(), MemoryEvidenceVerdict> {
        if !self.key.has_canonical_engaged_composition() {
            return Err(MemoryEvidenceVerdict::Invalid);
        }
        let selection = MemorySelection {
            strategy: self.key.strategy,
            parameters: self.key.parameters,
            tier: self.key.tier,
        };
        if self.key.engaged_composition != contract.engaged_composition_for_selection(&selection) {
            return Err(MemoryEvidenceVerdict::CompositionMismatch);
        }
        if !self.key.strategy.is_optimized() {
            return Ok(());
        }
        if self.conformance != MemoryConformanceState::Verified {
            return Err(MemoryEvidenceVerdict::Unverified);
        }
        if !self.validation_errors().is_empty() {
            return Err(MemoryEvidenceVerdict::Invalid);
        }
        if self.parity_result != MemoryParityResult::Passed {
            return Err(MemoryEvidenceVerdict::Unverified);
        }
        for dimension in MemoryEvidenceDimension::ALL {
            let verdict = self.dimensions.verdict(dimension);
            if verdict != MemoryEvidenceVerdict::Satisfied {
                return Err(verdict);
            }
        }
        let Some(identity) = &contract.calibration else {
            return Err(MemoryEvidenceVerdict::Unverified);
        };
        if self.key.load_shape != contract.load_shape || self.key.load_shape != identity.load_shape
        {
            return Err(MemoryEvidenceVerdict::FingerprintMismatch);
        }
        if self.calibration_abi != identity.abi
            || self.calibration_fingerprint != identity.fingerprint
        {
            return Err(MemoryEvidenceVerdict::FingerprintMismatch);
        }
        Ok(())
    }
}

fn validate_parity_limit(metric: &str, maximum_error: f64, errors: &mut Vec<String>) {
    if metric.trim().is_empty() {
        errors.push("parity metric must be non-empty".to_owned());
    }
    if !maximum_error.is_finite() || maximum_error < 0.0 {
        errors.push("parity maximum_error must be finite and non-negative".to_owned());
    }
}
