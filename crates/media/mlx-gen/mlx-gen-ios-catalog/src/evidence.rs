//! The single source-controlled iOS device evidence table (epic 16774, E4/E7): ANCHOR rows.
//!
//! `evidence/ios_device_evidence.json` is compiled in. Following the recorded memory-anchor rule
//! (SceneWorks `docs/memory-strategy-contract.md`: "the anchor + analytic derivation is the
//! complete, self-sufficient admission basis"; "every non-anchor rung and geometry is derived
//! analytically from the anchor rather than measured"), a row is one of at most TWO measured
//! anchors per anchor FAMILY `(machine, provider, route, tier, layout digest, mode)`:
//!
//! * [`AnchorKind::Resident`] — the rung-0 composition. On the 8 GB phone it may be refused before
//!   allocation by a quantum guard; that typed refusal IS its evidence ([`AnchorOutcome::Refused`]).
//! * [`AnchorKind::Optimized`] — the deepest rung with every saving engaged at its cheapest
//!   selection. It must complete, and it is captured at least [`MIN_OPTIMIZED_REPEATS`] times: the
//!   spread of those repeats is the admission margin at the low end ([`margin_bytes`]).
//!
//! Every other (rung, selection, geometry) of the family is DERIVED from these anchors by the
//! anchor law ([`crate::anchor`]). A request of another family (another mode, tier, artifact
//! layout or route) has no anchor and is refused `no_evidence`.
//!
//! The key is MEMORY-RELEVANT facts only: hardware, provider, route, tier, a path-free artifact
//! layout digest, mode, the reference geometry, reference count, execution levers and token window.
//! Prompt text, seed, install path and conditioning pixels are deliberately absent. Nothing at
//! runtime writes this table; a capture prints candidate rows that a human reviews and commits.

use mlx_gen::gen_core::{MemoryStrategy, MemoryStrategyParameters, TransformerComponent};
use mlx_gen::ios_pricing::IosPhase;
use serde::{Deserialize, Serialize};
use std::collections::BTreeMap;
use std::sync::OnceLock;

/// The repository path of the table, for capture tooling and review messages.
pub const EVIDENCE_TABLE_PATH: &str =
    "crates/media/mlx-gen/mlx-gen-ios-catalog/evidence/ios_device_evidence.json";

/// The only evidence class a row may carry: a physical-device capture run.
pub const DEVICE_CAPTURE_CLASS: &str = "device_capture";

/// The table schema. 3 = anchor rows (two anchors per family + derivation). Schema 2's exact-match
/// grid rows are refused, never migrated.
pub const EVIDENCE_SCHEMA: u32 = 3;

/// An optimized anchor must carry at least this many repeat captures: the margin at the low end is
/// their measured spread (floored at [`ANCHOR_MARGIN_FLOOR_BYTES`]), never a blanket reserve. Two
/// suffice because the floor already backstops a spread two repeats happen to under-sample.
pub const MIN_OPTIMIZED_REPEATS: usize = 2;

/// Floor of the low-end margin. Two repeats can agree to the byte, and a zero margin would be a
/// claim that the next render is bit-identical in footprint. This floor is an ENGINEERING choice,
/// not a measurement: 32 MiB is about 0.5% of the 6142 MiB entitled launch headroom and two orders of
/// magnitude below the 512 MiB formula reserve it replaces at an anchor. Replace it only with a
/// measured spread.
pub const ANCHOR_MARGIN_FLOOR_BYTES: u64 = 32 << 20;

const EVIDENCE_JSON: &str = include_str!("../evidence/ios_device_evidence.json");

/// Actual configured load form, not inferred from endpoint/rung names. An optimized configured
/// operation may require Eager (e.g. valid Iris adapters); ordinary catalog defaults are untouched.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum AnchorLoadForm {
    ResidentEager,
    ResidentDeferred,
    SequentialEager,
    SequentialDeferred,
}
impl AnchorLoadForm {
    pub fn from_spec(spec: &mlx_gen::LoadSpec) -> Self {
        use mlx_gen::gen_core::{LoadShape, OffloadPolicy};
        match (spec.offload_policy, spec.load_shape) {
            (OffloadPolicy::Resident, LoadShape::EagerMaterialization) => Self::ResidentEager,
            (OffloadPolicy::Resident, LoadShape::DeferredMaterialization) => Self::ResidentDeferred,
            (OffloadPolicy::Sequential, LoadShape::EagerMaterialization) => Self::SequentialEager,
            (OffloadPolicy::Sequential, LoadShape::DeferredMaterialization) => {
                Self::SequentialDeferred
            }
        }
    }
    pub fn apply(self, spec: &mut mlx_gen::LoadSpec) {
        use mlx_gen::gen_core::{LoadShape, OffloadPolicy};
        (spec.offload_policy, spec.load_shape) = match self {
            Self::ResidentEager => (OffloadPolicy::Resident, LoadShape::EagerMaterialization),
            Self::ResidentDeferred => (OffloadPolicy::Resident, LoadShape::DeferredMaterialization),
            Self::SequentialEager => (OffloadPolicy::Sequential, LoadShape::EagerMaterialization),
            Self::SequentialDeferred => (
                OffloadPolicy::Sequential,
                LoadShape::DeferredMaterialization,
            ),
        };
    }
    pub fn execution_prefix(self) -> &'static str {
        match self {
            Self::ResidentEager => "Resident|EagerMaterialization|",
            Self::ResidentDeferred => "Resident|DeferredMaterialization|",
            Self::SequentialEager => "Sequential|EagerMaterialization|",
            Self::SequentialDeferred => "Sequential|DeferredMaterialization|",
        }
    }
}

/// The memory-relevant identity of one request (or anchor) on one artifact and device.
#[derive(Clone, Debug, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct EvidenceKey {
    /// `hw.machine` (e.g. `iPhone17,2`), never total RAM or launch headroom.
    pub machine: String,
    /// The registered memory-strategy provider id.
    pub provider: String,
    /// The checkpoint route (SDXL family) or the provider id.
    pub route: String,
    /// `bf16`, `q8` or `q4`: the LOADED tier.
    pub tier: String,
    /// Path-free tensor/config layout digest of every configured source (hex SHA-256).
    pub layout_sha256: String,
    /// Path-free ARCHITECTURE digest of the base components (tensor names, shapes, dtypes and
    /// packing from headers; [`crate::layout::architecture_digest`]): the anchor FAMILY axis. Two
    /// checkpoints of one architecture share it; the per-checkpoint `layout_sha256` above stays the
    /// exact-key artifact check.
    pub architecture_sha256: String,
    /// Actual normalized configured overlay graph. Separate from base architecture; no paths,
    /// payload values or continuous nonzero-scale grid. None only for a genuinely unconfigured
    /// provider without an overlay exporter.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub configured_variant_sha256: Option<String>,
    /// `MemoryMode::as_key` of the request route.
    pub mode: String,
    pub width: u32,
    pub height: u32,
    pub batch: u32,
    /// Existing AGGREGATE image-carrier geometry (Reference + Mask + Control in the current
    /// validated closed set), identical to GenerationRequest::memory_reference_count.
    pub reference_count: u32,
    /// Typed subcounts WITHIN that aggregate, not additional images. A mode alone cannot
    /// distinguish control+reference or reconstruct a mask; replay must not invent media.
    pub mask_count: u32,
    pub control_count: u32,
    pub load_form: AnchorLoadForm,
    /// Canonical selection + request execution levers ([`crate::ios_admission::execution_key`]).
    pub execution: String,
    /// Canonical, validated provider sampling recipe. The family derives its class from the real
    /// resolver; a capture cannot claim a compatible class with an arbitrary string.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub sampling_recipe: Option<String>,
    /// Exact source context only; excluded from structural family/class identity.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub sdxl_work: Option<crate::SdxlWorkBinding>,
    /// Encoded text window: CLIP windows x 77 for SDXL, 0 where the provider pads to a fixed,
    /// artifact-determined window (covered by the layout digest).
    pub token_window: u32,
}

impl EvidenceKey {
    /// The anchor family this key belongs to: everything a derivation must hold EQUAL.
    pub fn family(&self) -> Result<AnchorFamily, String> {
        if self.provider != crate::sdxl::SDXL_ID && self.sdxl_work.is_some() {
            return Err("SDXL source binding belongs to another provider".into());
        }
        let sampling = self
            .sampling_recipe
            .as_ref()
            .map(|token| {
                crate::sampling::from_token(&self.provider, token).map_err(|e| e.to_string())
            })
            .transpose()?;
        if self.provider == crate::sdxl::SDXL_ID {
            if let Some(token) = &self.sampling_recipe {
                // Token decoding alone has no conditioning carrier. Revalidate its actual typed
                // roles through the SAME SDXL gate before claiming a family (no RGB/request).
                let recipe = mlx_gen_sdxl::sampling::SamplingRecipe::from_token(token)
                    .map_err(|error| error.to_string())?;
                mlx_gen_sdxl::sampling::ResolvedSampling::resolve_recipe(
                    &recipe,
                    mlx_gen_sdxl::sampling::SamplingWorkFacts {
                        reference_count: crate::pricing::reference_role_count(
                            self.reference_count,
                            self.mask_count,
                            self.control_count,
                        )
                        .map_err(|error| error.to_string())?
                            as usize,
                        mask_count: self.mask_count as usize,
                        control_count: self.control_count as usize,
                        reference_image: None,
                    },
                )
                .map_err(|error| error.to_string())?;
            }
        }
        // Active multi-branch topology belongs to the actual configured overlay, not an
        // unbounded raw-count family grid or an assumed component-additivity proof.
        if self.control_count > 1 && self.configured_variant_sha256.is_none() {
            return Err("multi-control anchor lacks its configured branch graph identity".into());
        }
        let mut variant = crate::pricing::anchor_variant_for_roles(
            &self.provider,
            &self.mode,
            &self.tier,
            sampling
                .as_ref()
                .map(|sampling| sampling.allocation_class()),
            self.reference_count,
            self.mask_count,
            self.control_count,
        )?;
        if let Some(sampling) = &sampling {
            variant.push('/');
            variant.push_str(sampling.allocation_class());
        } else if crate::sampling::requires_recipe(&self.provider) {
            return Err(format!(
                "{} evidence is missing its resolved sampling recipe",
                self.provider
            ));
        }
        // Joined DC-AE/native-retirement is a distinct memory execution
        // program, NOT a pin/currency event. Ordinary families remain intact.
        if matches!(self.provider.as_str(), "sana_1600m" | "sana_sprint_1600m")
            && self
                .execution
                .ends_with(mlx_gen_sana::ios_pricing::NATIVE_RETIREMENT_EXECUTION_SUFFIX)
        {
            variant.push_str("/sana-native-retirement-v1");
        }
        if let Some(configured) = &self.configured_variant_sha256 {
            variant.push_str("/configured=");
            variant.push_str(configured);
        }
        Ok(AnchorFamily {
            machine: self.machine.clone(),
            architecture_sha256: self.architecture_sha256.clone(),
            tier: self.tier.clone(),
            variant,
        })
    }
}

/// What must be equal between a request and an anchor for the anchor law to apply: the device, the
/// base ARCHITECTURE (so checkpoints of one architecture — the SDXL family's routes, SANA base and
/// Sprint — share anchors, and a header-different checkpoint or decoder is its own family), the
/// tier, and the anchor VARIANT: `base` for every mode the provider proves component-additive (its
/// components and extra phases are priced on top of the text-mode anchor), else the mode itself.
#[derive(Clone, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct AnchorFamily {
    pub machine: String,
    pub architecture_sha256: String,
    pub tier: String,
    pub variant: String,
}

/// Which extreme of the ladder an anchor measures.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum AnchorKind {
    /// Rung 0: every component resident.
    Resident,
    /// The deepest implemented rung at its cheapest selection.
    Optimized,
}

impl AnchorKind {
    /// The literal endpoint load form, shared by capture and anchor re-pricing. This is NOT an
    /// ordinary catalog default: a resident anchor must not arm a deferred full-cover stream.
    pub fn apply_load_form(self, spec: &mut mlx_gen::gen_core::LoadSpec) {
        use mlx_gen::gen_core::{LoadShape, OffloadPolicy};
        (spec.offload_policy, spec.load_shape) = match self {
            Self::Resident => (OffloadPolicy::Resident, LoadShape::EagerMaterialization),
            Self::Optimized => (
                OffloadPolicy::Sequential,
                LoadShape::DeferredMaterialization,
            ),
        };
    }
}

/// The anchor's memory selection, structurally (the anchor law re-prices the anchor's own
/// configuration through the provider's counted pricing).
#[derive(Clone, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct AnchorSelection {
    /// `MemoryStrategy::ALL` index (0 = resident … 4 = bounded transformer residency).
    pub rung: u8,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub stage_residency: Option<bool>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub decode_tile_edge: Option<u32>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub decode_overlap: Option<u32>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub attention_chunk_size: Option<u32>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub transformer_window_size: Option<u32>,
    /// `dit`, `te` or `both`.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub transformer_window_component: Option<String>,
}

impl AnchorSelection {
    pub fn from_parts(strategy: MemoryStrategy, p: &MemoryStrategyParameters) -> Self {
        Self {
            rung: MemoryStrategy::ALL
                .iter()
                .position(|s| *s == strategy)
                .unwrap_or(0) as u8,
            stage_residency: p.stage_residency,
            decode_tile_edge: p.decode_tile_edge,
            decode_overlap: p.decode_overlap,
            attention_chunk_size: p.attention_chunk_size,
            transformer_window_size: p.transformer_window_size,
            transformer_window_component: p.transformer_window_component.map(|c| {
                match c {
                    TransformerComponent::Dit => "dit",
                    TransformerComponent::TextEncoder => "te",
                    TransformerComponent::Both => "both",
                }
                .to_string()
            }),
        }
    }

    pub fn strategy(&self) -> Option<MemoryStrategy> {
        MemoryStrategy::ALL.get(usize::from(self.rung)).copied()
    }

    pub fn parameters(&self) -> Option<MemoryStrategyParameters> {
        let component = match self.transformer_window_component.as_deref() {
            None => None,
            Some("dit") => Some(TransformerComponent::Dit),
            Some("te") => Some(TransformerComponent::TextEncoder),
            Some("both") => Some(TransformerComponent::Both),
            Some(_) => return None,
        };
        Some(MemoryStrategyParameters {
            stage_residency: self.stage_residency,
            decode_tile_edge: self.decode_tile_edge,
            decode_overlap: self.decode_overlap,
            attention_chunk_size: self.attention_chunk_size,
            transformer_window_size: self.transformer_window_size,
            transformer_window_component: component,
        })
    }
}

/// One phase's measured peak: the kernel lifetime `phys_footprint` peak (absolute) when the phase
/// ended. `exact` = the lifetime peak ROSE during this phase (so it is this phase's own maximum);
/// otherwise an earlier phase masked it and the value is an upper bound.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct PhasePeak {
    pub peak_bytes: u64,
    pub exact: bool,
}

/// Engineering bounds on observation storage, NOT a memory estimate, step cap or fit gate.
pub const BASE_SAMPLING_MARKS: usize = 1024;
/// Per actually retained Checked: native epoch + joined receipt + at most one
/// GLU/selected-owner shape mark. Root start/end and owned-encoder release add3.
/// This composes the EXISTING observation/check bounds, not a private step cap.
pub const NATIVE_RETIREMENT_MARKS_PER_CHECK: usize = 3;
pub const NATIVE_RETIREMENT_ROOT_MARKS: usize = 3;
pub const MAX_NATIVE_RETIREMENT_CHECK_MARKS: usize =
    NATIVE_RETIREMENT_MARKS_PER_CHECK * MAX_QUANTUM_CHECKS;
pub const MAX_SAMPLING_MARKS: usize =
    BASE_SAMPLING_MARKS + MAX_NATIVE_RETIREMENT_CHECK_MARKS + NATIVE_RETIREMENT_ROOT_MARKS;
pub const MAX_SAMPLING_FIELDS: usize = 64;
pub const MAX_SAMPLING_NAME_BYTES: usize = 128;

/// Scalar metadata emitted at an ACTUAL production observation site. A request/serial mark is
/// not a model evaluation; kind/fields retain that distinction. No tensors/values are serialized.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct SamplingMark {
    pub provider: String,
    pub kind: String,
    pub fields: BTreeMap<String, u64>,
}

/// Bounded observed metadata and retained per-kind counters. This records what ran, not a claim
/// that every epoch required by a class was covered. Missing/truncated observations are explicit.
#[derive(Clone, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct SamplingCoverage {
    pub marks: Vec<SamplingMark>,
    pub counts: BTreeMap<String, BTreeMap<String, u64>>,
    pub total_marks: u64,
    pub dropped_marks: u64,
    pub malformed_marks: u64,
    pub truncated: bool,
}

/// Execution identity, not native/physical proof or a currency gate. Completed
/// SANA captures cannot label the joined program as ordinary (or vice versa).
/// Old ordinary keys with no new native marks remain valid unchanged.
pub(crate) fn validate_sana_capture_program(
    provider: &str,
    execution: &str,
    coverage: &SamplingCoverage,
    quantum: Option<&QuantumCoverage>,
) -> Result<(), String> {
    if !matches!(provider, "sana_1600m" | "sana_sprint_1600m") {
        return Ok(());
    }
    let tagged = execution.ends_with(mlx_gen_sana::ios_pricing::NATIVE_RETIREMENT_EXECUTION_SUFFIX);
    let mut start = None;
    let mut end = None;
    for mark in &coverage.marks {
        if mark.provider != provider {
            continue;
        }
        if mark.kind == "native-lifecycle" && start.replace(mark).is_some() {
            return Err("SANA capture has multiple native invocation identities".into());
        }
        if mark.kind == "native-lifecycle-completed" && end.replace(mark).is_some() {
            return Err("SANA capture has multiple native completions".into());
        }
    }
    if !tagged && start.is_none() && end.is_none() {
        if coverage
            .marks
            .iter()
            .any(|mark| observation_storage_class(&mark.provider, &mark.kind) != 0)
        {
            return Err(
                "ordinary SANA completed evidence contains unbound retirement claims".into(),
            );
        }
        return Ok(());
    }
    let valid = match (start, end) {
        (Some(start), Some(end)) => {
            tagged
                && start.fields.get("schema") == Some(&1)
                && start.fields.get("started") == Some(&1)
                && start.fields.get("configured") == Some(&1)
                && start.fields.get("invocation").is_some_and(|id| *id > 0)
                && start.fields.get("invocation") == end.fields.get("invocation")
                && end.fields.get("schema") == Some(&1)
                && end.fields.get("status") == Some(&1)
        }
        _ => false,
    };
    if !valid {
        return Err(
            "SANA capture execution key disagrees with its actual configured native invocation"
                .into(),
        );
    }
    let quantum = quantum.ok_or("successful native SANA claim lacks actual checked receipts")?;
    // Reuse the SAME source-owned receipt validator through borrowed views;
    // no metadata/media clone, duplicate protocol, invented epoch or unsafe cast.
    use mlx_gen_sana::ios_pricing::{RetirementCheckView, RetirementMetadataView};
    mlx_gen_sana::ios_pricing::validate_retirement_receipts(
        coverage.marks.len(),
        |index| {
            coverage
                .marks
                .get(index)
                .map(|mark| RetirementMetadataView {
                    provider: &mark.provider,
                    kind: &mark.kind,
                    fields: &mark.fields,
                })
        },
        quantum.checks.iter().map(|check| RetirementCheckView {
            phase: check.phase.as_deref().and_then(IosPhase::from_key),
            label: &check.label,
            context: &check.context,
            count: check.count,
        }),
        !coverage.truncated
            && coverage.dropped_marks == 0
            && coverage.malformed_marks == 0
            && !quantum.truncated
            && quantum.dropped_checks == 0,
    )
    .map_err(|error| error.to_string())
}

fn observation_storage_class(provider: &str, kind: &str) -> usize {
    match (provider, kind) {
        ("mlx", "native-quantum") => 1,
        ("mlx", "native-joined") => 2,
        ("sana", "native-dcae-glu" | "native-selected") => 3,
        ("sana_1600m" | "sana_sprint_1600m", "native-lifecycle" | "native-lifecycle-completed")
        | ("sana", "native-owned-phase-retired") => 4,
        _ => 0,
    }
}
impl SamplingCoverage {
    fn category_usage(&self) -> Option<[u64; 5]> {
        let mut used = [0u64; 5];
        for (provider, kinds) in &self.counts {
            for (kind, count) in kinds {
                let class = observation_storage_class(provider, kind);
                used[class] = used[class].checked_add(*count)?;
            }
        }
        Some(used)
    }
    /// Return the retained mark ordinal for temporal quantum attribution. An oversized/malformed
    /// event cannot leave a stale mark active; callers clear their preceding ordinal on None.
    pub fn record(&mut self, provider: &str, kind: &str, fields: &[(&str, u64)]) -> Option<u32> {
        if self.total_marks == u64::MAX {
            self.truncated = true;
            return None;
        }
        self.total_marks += 1;
        let name_ok = |name: &str| !name.is_empty() && name.len() <= MAX_SAMPLING_NAME_BYTES;
        let malformed = !name_ok(provider)
            || !name_ok(kind)
            || fields.len() > MAX_SAMPLING_FIELDS
            || fields.iter().any(|(name, _)| !name_ok(name))
            || fields
                .iter()
                .enumerate()
                .any(|(i, (name, _))| fields[..i].iter().any(|(prior, _)| prior == name));
        let class = observation_storage_class(provider, kind);
        let limits = [
            BASE_SAMPLING_MARKS as u64,
            MAX_QUANTUM_CHECKS as u64,
            MAX_QUANTUM_CHECKS as u64,
            MAX_QUANTUM_CHECKS as u64,
            NATIVE_RETIREMENT_ROOT_MARKS as u64,
        ];
        let category_full = self
            .category_usage()
            .is_none_or(|used| used[class] >= limits[class]);
        if malformed || category_full || self.marks.len() >= MAX_SAMPLING_MARKS {
            self.dropped_marks += 1;
            if malformed {
                self.malformed_marks += 1;
            }
            self.truncated = true;
            return None;
        }
        let index = self.marks.len() as u32;
        self.marks.push(SamplingMark {
            provider: provider.into(),
            kind: kind.into(),
            fields: fields
                .iter()
                .map(|(key, value)| ((*key).into(), *value))
                .collect(),
        });
        *self
            .counts
            .entry(provider.into())
            .or_default()
            .entry(kind.into())
            .or_default() += 1;
        Some(index)
    }

    /// Presence at actual published execution sites, NOT proof of complete class coverage or
    /// materialized/free storage. Request, serial declarations and allocator policy alone do not
    /// count as solver/forward observations.
    pub fn has_execution_observation(&self) -> bool {
        self.marks.iter().any(|mark| {
            mark.kind.starts_with("solver")
                || mark.kind.starts_with("forward")
                || mark.kind.starts_with("discrete-")
                || mark.kind.starts_with("scm_")
                || mark.kind.starts_with("predict_")
                || matches!(
                    mark.kind.as_str(),
                    "solve_complete" | "callback_evaluated" | "denoised_pair_return"
                )
        })
    }

    pub fn validate(&self) -> Result<(), String> {
        let used = self
            .category_usage()
            .ok_or("sampling category counter overflow")?;
        if used[0] > BASE_SAMPLING_MARKS as u64
            || used[1..4]
                .iter()
                .any(|used| *used > MAX_QUANTUM_CHECKS as u64)
            || used[4] > NATIVE_RETIREMENT_ROOT_MARKS as u64
        {
            return Err("ordinary/native observation storage categories exceed their independent source budgets".into());
        }
        if self.marks.len() > MAX_SAMPLING_MARKS
            || self.malformed_marks > self.dropped_marks
            || self.total_marks
                != (self.marks.len() as u64)
                    .checked_add(self.dropped_marks)
                    .ok_or("sampling counter overflow")?
            || self.truncated != (self.dropped_marks > 0)
        {
            return Err("inconsistent bounded sampling observation counts".into());
        }
        let mut rebuilt = Self::default();
        for mark in &self.marks {
            let fields: Vec<_> = mark
                .fields
                .iter()
                .map(|(key, value)| (key.as_str(), *value))
                .collect();
            if rebuilt
                .record(&mark.provider, &mark.kind, &fields)
                .is_none()
            {
                return Err("malformed retained sampling observation".into());
            }
        }
        if rebuilt.counts != self.counts {
            return Err("sampling per-kind counters disagree with retained observations".into());
        }
        Ok(())
    }
}

/// Bounded actual Checked/context witnesses. Identical repeated label/context entries retain a
/// count; this does NOT disambiguate different operations sharing that label. Providers must use
/// conservative plausible-candidate MAX and source-proved correspondence.
pub const MAX_QUANTUM_CHECKS: usize = 4096;
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct QuantumCheck {
    pub phase: Option<String>,
    pub label: String,
    pub context: Vec<u32>,
    pub count: u64,
}
#[derive(Clone, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct QuantumCoverage {
    pub checks: Vec<QuantumCheck>,
    pub total_checks: u64,
    pub dropped_checks: u64,
    pub truncated: bool,
}
impl QuantumCoverage {
    pub fn record(&mut self, phase: Option<IosPhase>, label: &str, context: Vec<u32>) {
        if self.total_checks == u64::MAX {
            self.truncated = true;
            return;
        }
        self.total_checks += 1;
        let phase = phase.map(|p| p.as_key().into());
        if let Some(check) = self
            .checks
            .iter_mut()
            .find(|check| check.phase == phase && check.label == label && check.context == context)
        {
            check.count += 1;
            return;
        }
        if self.checks.len() >= MAX_QUANTUM_CHECKS
            || context.len() > MAX_SAMPLING_FIELDS
            || label.is_empty()
            || label.len() > MAX_SAMPLING_NAME_BYTES
        {
            self.dropped_checks += 1;
            self.truncated = true;
            return;
        }
        self.checks.push(QuantumCheck {
            phase,
            label: label.into(),
            context,
            count: 1,
        });
    }
    pub fn validate(&self, metadata: &SamplingCoverage) -> Result<(), String> {
        if self.checks.len() > MAX_QUANTUM_CHECKS || self.truncated != (self.dropped_checks > 0) {
            return Err("inconsistent bounded quantum coverage".into());
        }
        let mut total = self.dropped_checks;
        for check in &self.checks {
            if check.count == 0
                || check.label.is_empty()
                || check.label.len() > MAX_SAMPLING_NAME_BYTES
                || check.context.len() > MAX_SAMPLING_FIELDS
                || check
                    .phase
                    .as_ref()
                    .is_some_and(|phase| IosPhase::from_key(phase).is_none())
                || check
                    .context
                    .iter()
                    .any(|index| *index as usize >= metadata.marks.len())
            {
                return Err("invalid actual quantum/context witness".into());
            }
            total = total
                .checked_add(check.count)
                .ok_or("quantum coverage counter overflow")?;
        }
        if total != self.total_checks {
            return Err("quantum coverage total disagrees with checks".into());
        }
        Ok(())
    }
}

/// One quantum formula as the device saw it in a capture: the guard's priced scratch at the
/// invocation with the largest observed/priced ratio, the kernel-footprint growth across it (the
/// lifetime peak at its end minus the footprint at its check), and whether that growth is exact
/// (the lifetime peak rose inside the quantum) or masked by an earlier peak (an upper bound).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct QuantumObservation {
    pub priced_scratch_bytes: u64,
    pub observed_bytes: u64,
    pub exact: bool,
    /// Last actual sampling metadata mark before this quantum's check. This is a temporal
    /// association, not an assertion that a reported logical owner remains live after its scope.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub preceding_sampling_mark: Option<u32>,
}

/// Raw same-task kernel query provenance. Required field word counts come from the exact SDK
/// layout used by the capture decoder; absent/short/error fields remain None, never synthetic zero.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum KernelFootprintMethod {
    TaskVmInfo,
    UnsupportedPlatform,
}
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct KernelFootprintSnapshot {
    pub method: KernelFootprintMethod,
    pub query_status: i32,
    pub requested_words: u32,
    pub returned_words: u32,
    pub current_required_words: u32,
    pub peak_required_words: u32,
    pub current_bytes: Option<u64>,
    pub lifetime_peak_bytes: Option<u64>,
}
// Darwin LP64 TASK_VM_INFO, pack(4): phys_footprint ends at word38; the REV3
// ledger_phys_footprint_peak member ends at word84. These are ABI offsets, not memory evidence.
pub const TASK_VM_INFO_CURRENT_REQUIRED_WORDS: u32 = 38;
pub const TASK_VM_INFO_PEAK_REQUIRED_WORDS: u32 = 84;
impl KernelFootprintSnapshot {
    pub fn current(&self) -> Option<u64> {
        if self.method != KernelFootprintMethod::TaskVmInfo
            || self.query_status != 0
            || self.current_required_words != TASK_VM_INFO_CURRENT_REQUIRED_WORDS
            || self.returned_words < self.current_required_words
            || self.returned_words > self.requested_words
        {
            return None;
        }
        self.current_bytes.filter(|current| *current > 0)
    }
    pub fn complete_pair(&self) -> Result<(u64, u64), String> {
        let current = self
            .current()
            .ok_or("kernel current footprint unavailable/short/error")?;
        if self.peak_required_words != TASK_VM_INFO_PEAK_REQUIRED_WORDS
            || self.returned_words < self.peak_required_words
            || self.requested_words < self.peak_required_words
        {
            return Err("kernel lifetime footprint peak unavailable/short/error".into());
        }
        let peak = self
            .lifetime_peak_bytes
            .ok_or("kernel lifetime peak missing")?;
        if peak < current {
            return Err("kernel lifetime peak below current footprint".into());
        }
        Ok((current, peak))
    }
}

/// One capture run of an anchor.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Capture {
    pub run_id: String,
    pub build_sha: String,
    /// Resolved requested denoise steps; actual callback/update coverage is recorded separately.
    /// Short reference recipes must still exercise their class's required warm-up/terminal epochs.
    pub steps: u32,
    /// The capture process's `phys_footprint` before any of the case's loads.
    pub baseline_footprint_bytes: u64,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub before_request_snapshot: Option<KernelFootprintSnapshot>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub before_model_load_snapshot: Option<KernelFootprintSnapshot>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub ending_snapshot: Option<KernelFootprintSnapshot>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub observed_current_peak_bytes: Option<u64>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub observed_model_current_peak_bytes: Option<u64>,
    #[serde(default)]
    pub valid_kernel_queries: u64,
    #[serde(default)]
    pub valid_model_kernel_queries: u64,
    /// Kernel-ledger LIFETIME `phys_footprint` peak of the capture process (absolute).
    pub footprint_bytes: u64,
    /// Per-phase peaks split through the quantum marks (`IosPhase::as_key` → peak). A phase the
    /// capture could not attribute is absent and is priced at the whole-run peak (erring large).
    pub phase_peaks: BTreeMap<String, PhasePeak>,
    /// Actual production-emitted bounded observations, never a predicted plan/progress replay.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub sampling_coverage: Option<SamplingCoverage>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub quantum_coverage: Option<QuantumCoverage>,
    /// Missing/unreadable quantum footprint marks invalidate phase attribution, not a valid
    /// starting/ending whole-run kernel upper. Consumers must then use the whole-run upper.
    #[serde(default)]
    pub quantum_query_gaps: u64,
    /// Per quantum label, the device's observation of that formula (the per-formula DEVICE
    /// calibration: priced scratch vs observed peak).
    #[serde(default, skip_serializing_if = "BTreeMap::is_empty")]
    pub quanta: BTreeMap<String, QuantumObservation>,
    /// MLX `get_peak_memory`, bytes (informational).
    pub mlx_peak_bytes: u64,
    /// Minimum sampled `os_proc_available_memory`, MiB (informational).
    pub min_headroom_mib: f64,
    pub elapsed_seconds: f64,
}

impl Capture {
    /// Construct a validated physical upper in ONE explicit baseline coordinate. Both original
    /// and SDK/pre-load ledgers remain available; this is never a free baseline conversion.
    pub fn kernel_growth(
        &self,
        coordinate: mlx_gen::ios_pricing::IosMemoryCoordinate,
    ) -> Result<mlx_gen::ios_anchor_law::KernelGrowth, String> {
        use mlx_gen::ios_pricing::IosMemoryCoordinate;
        let (start, current_peak, queries) = match coordinate {
            IosMemoryCoordinate::BeforeRequestConstruction => (
                self.before_request_snapshot.as_ref(),
                self.observed_current_peak_bytes,
                self.valid_kernel_queries,
            ),
            IosMemoryCoordinate::BeforeModelLoad => (
                self.before_model_load_snapshot.as_ref(),
                self.observed_model_current_peak_bytes,
                self.valid_model_kernel_queries,
            ),
            _ => return Err("capture has no measurement in requested baseline coordinate".into()),
        };
        let (current, old_highwater) = start
            .ok_or("capture starting kernel snapshot missing")?
            .complete_pair()?;
        if coordinate == IosMemoryCoordinate::BeforeRequestConstruction
            && current != self.baseline_footprint_bytes
        {
            return Err(
                "original baseline scalar disagrees with its actual kernel snapshot".into(),
            );
        }
        let (_, ending_highwater) = self
            .ending_snapshot
            .as_ref()
            .ok_or("capture ending kernel snapshot missing")?
            .complete_pair()?;
        if ending_highwater != self.footprint_bytes {
            return Err("capture peak disagrees with ending kernel query".into());
        }
        let peak = current_peak.ok_or("actual run current-footprint observations missing")?;
        mlx_gen::ios_anchor_law::KernelGrowth::from_queries(
            current,
            old_highwater,
            ending_highwater,
            peak,
            queries,
        )
        .map_err(|e| e.to_string())
    }

    /// The render's own growth over the capture's starting footprint.
    pub fn demand_bytes(&self) -> u64 {
        self.footprint_bytes
            .saturating_sub(self.baseline_footprint_bytes)
    }
}

/// The pre-construction quantum refusal that is a refused anchor's evidence.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct RefusedQuantum {
    pub label: String,
    pub scratch_bytes: u64,
    pub reserve_bytes: u64,
    pub available_bytes: Option<u64>,
}

/// What the anchor capture established.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case", deny_unknown_fields)]
pub enum AnchorOutcome {
    /// Completed captures (repeats of the same key).
    Completed { captures: Vec<Capture> },
    /// Refused before allocation by a quantum guard (resident anchors only).
    Refused {
        refused_quantum: RefusedQuantum,
        run_id: String,
        build_sha: String,
    },
}

/// One reviewed anchor capture.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct AnchorRow {
    pub key: EvidenceKey,
    pub anchor_kind: AnchorKind,
    pub selection: AnchorSelection,
    pub outcome: AnchorOutcome,
    /// Always [`DEVICE_CAPTURE_CLASS`].
    pub evidence_class: String,
}

impl AnchorRow {
    /// Completed captures (empty for a refused anchor).
    pub fn captures(&self) -> &[Capture] {
        match &self.outcome {
            AnchorOutcome::Completed { captures } => captures,
            AnchorOutcome::Refused { .. } => &[],
        }
    }

    /// Validated upper in one declared coordinate. Masked highwater remains an upper only.
    pub fn upper_growth_at(
        &self,
        coordinate: mlx_gen::ios_pricing::IosMemoryCoordinate,
    ) -> Result<u64, String> {
        self.captures()
            .iter()
            .map(|capture| {
                capture
                    .kernel_growth(coordinate)
                    .map(|growth| growth.growth_upper)
            })
            .collect::<Result<Vec<_>, _>>()?
            .into_iter()
            .max()
            .ok_or_else(|| "anchor has no completed capture".into())
    }

    /// Accurate reference demand/spread requires exact attained peaks for every repeat; no masked
    /// upper is silently called measured variability or rescued by a smaller exact repeat.
    pub fn exact_growth_and_margin_at(
        &self,
        coordinate: mlx_gen::ios_pricing::IosMemoryCoordinate,
    ) -> Result<(u64, u64), String> {
        let demands = self
            .captures()
            .iter()
            .map(|capture| {
                capture
                    .kernel_growth(coordinate)?
                    .exact_growth()
                    .map_err(|e| e.to_string())
            })
            .collect::<Result<Vec<_>, String>>()?;
        let max = demands
            .iter()
            .copied()
            .max()
            .ok_or("anchor has no completed capture")?;
        let min = demands
            .iter()
            .copied()
            .min()
            .ok_or("anchor has no completed capture")?;
        Ok((max, (max - min).max(ANCHOR_MARGIN_FLOOR_BYTES)))
    }

    /// The ORIGINAL pre-request ledger, retained for diagnostics/compatibility, not silently
    /// substituted for SDK BeforeModelLoad growth.
    /// The anchor's measured demand: the largest growth over the starting footprint across its
    /// repeats (`None` for a refused anchor).
    pub fn demand_bytes(&self) -> Option<u64> {
        self.captures().iter().map(Capture::demand_bytes).max()
    }

    /// Per phase, the largest measured growth over the starting footprint across repeats.
    pub fn phase_demands(&self) -> BTreeMap<IosPhase, u64> {
        let mut out = BTreeMap::new();
        for capture in self.captures() {
            for (phase, peak) in &capture.phase_peaks {
                if let Some(phase) = IosPhase::from_key(phase) {
                    let growth = peak
                        .peak_bytes
                        .saturating_sub(capture.baseline_footprint_bytes);
                    let slot = out.entry(phase).or_insert(0u64);
                    *slot = (*slot).max(growth);
                }
            }
        }
        out
    }

    /// The low-end margin: the measured spread (max - min demand) of the repeats, floored at
    /// [`ANCHOR_MARGIN_FLOOR_BYTES`]. `None` for a refused anchor.
    pub fn margin_bytes(&self) -> Option<u64> {
        margin_bytes(self.captures())
    }

    /// The anchor's per-quantum DEVICE calibration: per label, over every repeat's EXACT
    /// observation, the one with the largest observed/priced ratio (the formula's worst
    /// under-pricing on the device). A label never observed exactly has no entry and keeps its
    /// formula.
    pub fn quantum_calibration(
        &self,
    ) -> BTreeMap<String, mlx_gen::ios_quantum::QuantumCalibration> {
        let mut out: BTreeMap<String, mlx_gen::ios_quantum::QuantumCalibration> = BTreeMap::new();
        for capture in self.captures() {
            for (label, q) in &capture.quanta {
                if !q.exact || q.priced_scratch_bytes == 0 {
                    continue;
                }
                // Keep the larger observed/priced ratio (cross-multiplied, exact).
                let worse = out.get(label).is_none_or(|existing| {
                    u128::from(q.observed_bytes) * u128::from(existing.priced_bytes)
                        > u128::from(existing.observed_bytes) * u128::from(q.priced_scratch_bytes)
                });
                if worse {
                    out.insert(
                        label.clone(),
                        mlx_gen::ios_quantum::QuantumCalibration {
                            priced_bytes: q.priced_scratch_bytes,
                            observed_bytes: q.observed_bytes,
                        },
                    );
                }
            }
        }
        out
    }
}

/// The admission margin of a set of repeat captures: their demand spread, floored.
pub fn margin_bytes(captures: &[Capture]) -> Option<u64> {
    let demands = captures.iter().map(Capture::demand_bytes);
    let max = demands.clone().max()?;
    let min = demands.min()?;
    Some((max - min).max(ANCHOR_MARGIN_FLOOR_BYTES))
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct Table {
    schema: u32,
    #[allow(dead_code)]
    about: String,
    anchors: Vec<AnchorRow>,
}

fn validate_key(k: &EvidenceKey) -> Result<(), String> {
    for digest in [&k.layout_sha256, &k.architecture_sha256]
        .into_iter()
        .chain(k.configured_variant_sha256.iter())
    {
        if digest.len() != 64
            || !digest
                .bytes()
                .all(|b| b.is_ascii_hexdigit() && !b.is_ascii_uppercase())
        {
            return Err(format!("row {k:?}: digests must be 64 lowercase hex"));
        }
    }
    if !["bf16", "q8", "q4"].contains(&k.tier.as_str())
        || k.machine.is_empty()
        || k.provider.is_empty()
        || k.route.is_empty()
        || k.mode.is_empty()
        || k.execution.is_empty()
        || k.width == 0
        || k.height == 0
        || k.batch == 0
    {
        return Err(format!("row {k:?}: incomplete key"));
    }
    k.family()?;
    if !k.execution.starts_with(k.load_form.execution_prefix()) {
        return Err("typed load form disagrees with actual execution provenance".into());
    }
    if let Some(token) = &k.sampling_recipe {
        if !k.execution.ends_with(&format!("|recipe={token}")) {
            return Err("sampling recipe disagrees with execution provenance".into());
        }
    }
    Ok(())
}

pub(crate) fn validate_rows(rows: &[AnchorRow]) -> Result<(), String> {
    let mut families = std::collections::BTreeSet::new();
    let mut keys = std::collections::BTreeSet::new();
    for row in rows {
        let k = &row.key;
        if row.evidence_class != DEVICE_CAPTURE_CLASS {
            return Err(format!(
                "row {k:?}: evidence_class must be {DEVICE_CAPTURE_CLASS}"
            ));
        }
        validate_key(k)?;
        if matches!(&row.outcome, AnchorOutcome::Completed { .. })
            && k.provider == crate::sdxl::SDXL_ID
            && k.sdxl_work.is_some()
        {
            let facts = k
                .sdxl_work
                .as_ref()
                .ok_or("completed SDXL row lacks actual source context")?
                .require_facts()
                .map_err(|e| e.to_string())?;
            crate::source_work::validate_key_roles(k, facts).map_err(|e| e.to_string())?;
        }
        if row.selection.strategy().is_none() || row.selection.parameters().is_none() {
            return Err(format!("row {k:?}: malformed anchor selection"));
        }
        if row.anchor_kind == AnchorKind::Resident && k.load_form != AnchorLoadForm::ResidentEager {
            return Err(format!("row {k:?}: resident endpoint must be actual Resident+Eager, not staged/stream-armed"));
        }
        if (row.anchor_kind == AnchorKind::Resident) != (row.selection.rung == 0) {
            return Err(format!(
                "row {k:?}: a resident anchor is rung 0 and an optimized anchor is not"
            ));
        }
        match &row.outcome {
            AnchorOutcome::Refused {
                refused_quantum,
                run_id,
                build_sha,
            } => {
                // The fully optimized anchor must fit: its refusal is a defect, never evidence.
                if row.anchor_kind == AnchorKind::Optimized {
                    return Err(format!(
                        "row {k:?}: an optimized anchor must be a completed capture"
                    ));
                }
                if refused_quantum.label.is_empty() || run_id.is_empty() || build_sha.is_empty() {
                    return Err(format!("row {k:?}: refusal without its quantum/provenance"));
                }
            }
            AnchorOutcome::Completed { captures } => {
                let needed = match row.anchor_kind {
                    AnchorKind::Optimized => MIN_OPTIMIZED_REPEATS,
                    AnchorKind::Resident => 1,
                };
                if captures.len() < needed {
                    return Err(format!(
                        "row {k:?}: {:?} anchor needs >= {needed} repeat captures",
                        row.anchor_kind
                    ));
                }
                for c in captures {
                    if c.run_id.is_empty() || c.build_sha.is_empty() || c.steps == 0 {
                        return Err(format!("row {k:?}: capture missing run/build/steps"));
                    }
                    c.kernel_growth(
                        mlx_gen::ios_pricing::IosMemoryCoordinate::BeforeRequestConstruction,
                    )
                    .map_err(|error| {
                        format!("row {k:?}: original footprint ledger invalid: {error}")
                    })?;
                    c.kernel_growth(mlx_gen::ios_pricing::IosMemoryCoordinate::BeforeModelLoad)
                        .map_err(|error| {
                            format!("row {k:?}: SDK footprint ledger invalid: {error}")
                        })?;
                    if let Some(coverage) = &c.sampling_coverage {
                        coverage.validate().map_err(|e| format!("row {k:?}: {e}"))?;
                        validate_sana_capture_program(
                            &k.provider,
                            &k.execution,
                            coverage,
                            c.quantum_coverage.as_ref(),
                        )
                        .map_err(|error| format!("row {k:?}: {error}"))?;
                    }
                    if crate::sampling::requires_recipe(&k.provider) {
                        let mut roles = c
                            .sampling_coverage
                            .as_ref()
                            .ok_or("actual conditioning role witness missing")?
                            .marks
                            .iter()
                            .filter(|mark| {
                                mark.provider == "ios_capture" && mark.kind == "conditioning-roles"
                            });
                        let role = roles
                            .next()
                            .ok_or("actual conditioning role witness missing")?;
                        if roles.next().is_some()
                            || role.fields.len() != 3
                            || role.fields.get("reference_count")
                                != Some(&u64::from(k.reference_count))
                            || role.fields.get("mask_count") != Some(&u64::from(k.mask_count))
                            || role.fields.get("control_count") != Some(&u64::from(k.control_count))
                        {
                            return Err(format!("row {k:?}: conditioning roles disagree with actual prepared-carrier witness"));
                        }
                    }
                    if let Some(quantum) = &c.quantum_coverage {
                        let metadata = c
                            .sampling_coverage
                            .as_ref()
                            .ok_or("quantum contexts have no metadata witness list")?;
                        quantum.validate(metadata)?;
                    }
                    if crate::sampling::requires_generation_recipe(&k.provider)
                        && c.quantum_coverage
                            .as_ref()
                            .is_none_or(|quantum| quantum.checks.is_empty() || quantum.truncated)
                    {
                        return Err(format!(
                            "row {k:?}: actual quantum/context coverage missing/truncated"
                        ));
                    }
                    if crate::sampling::requires_generation_recipe(&k.provider)
                        && c.sampling_coverage.as_ref().is_none_or(|coverage| {
                            !coverage.has_execution_observation() || coverage.truncated
                        })
                    {
                        return Err(format!(
                            "row {k:?}: actual sampling observations are missing/truncated"
                        ));
                    }
                    if c.baseline_footprint_bytes == 0
                        || c.baseline_footprint_bytes >= c.footprint_bytes
                    {
                        return Err(format!(
                            "row {k:?}: capture baseline footprint must be present and below its peak"
                        ));
                    }
                    for (label, q) in &c.quanta {
                        if let Some(index) = q.preceding_sampling_mark {
                            if c.sampling_coverage
                                .as_ref()
                                .and_then(|coverage| coverage.marks.get(index as usize))
                                .is_none()
                            {
                                return Err(format!("row {k:?}: quantum {label} refers to missing sampling observation"));
                            }
                        }
                        if label.is_empty() || q.observed_bytes > c.footprint_bytes {
                            return Err(format!(
                                "row {k:?}: quantum {label} observed above the lifetime peak"
                            ));
                        }
                    }
                    for (phase, peak) in &c.phase_peaks {
                        if IosPhase::from_key(phase).is_none()
                            || peak.peak_bytes < c.baseline_footprint_bytes
                            || peak.peak_bytes > c.footprint_bytes
                        {
                            return Err(format!(
                                "row {k:?}: phase {phase} peak outside [baseline, lifetime peak]"
                            ));
                        }
                    }
                }
                let run_ids: std::collections::BTreeSet<_> =
                    captures.iter().map(|c| &c.run_id).collect();
                if run_ids.len() != captures.len() {
                    return Err(format!("row {k:?}: repeat captures must be distinct runs"));
                }
            }
        }
        if !keys.insert(k.clone()) {
            return Err(format!("row {k:?}: duplicate key"));
        }
        if !families.insert((k.family()?, row.anchor_kind)) {
            return Err(format!(
                "row {k:?}: a family has at most one {:?} anchor",
                row.anchor_kind
            ));
        }
    }
    Ok(())
}

pub(crate) fn parse(json: &str) -> Result<Vec<AnchorRow>, String> {
    let mut budget = mlx_gen_sdxl::sampling::SourceRequestBudget::bounded(json.len() as u64);
    parse_with_source_budget(json, &mut budget)
}

/// Explicit finite requested-layout allowance. This does not certify tokenizer or process memory.
pub fn parse_with_source_budget(
    json: &str,
    budget: &mut mlx_gen_sdxl::sampling::SourceRequestBudget<'_>,
) -> Result<Vec<AnchorRow>, String> {
    let (schema, anchors) = crate::source_work::parse_table(json, budget)?;
    let table = Table {
        schema,
        about: String::new(),
        anchors,
    };
    // Schema 3 replaced the exact-match grid with anchors; an older table is refused (fail
    // closed), never migrated by guess.
    if table.schema != EVIDENCE_SCHEMA {
        return Err(format!("unknown evidence schema {}", table.schema));
    }
    validate_rows(&table.anchors)?;
    Ok(table.anchors)
}

/// The compiled-in, reviewed table. A malformed table admits nothing (every lookup refuses).
static ROWS: OnceLock<Result<Vec<AnchorRow>, String>> = OnceLock::new();
pub fn rows() -> Result<&'static [AnchorRow], String> {
    ROWS.get_or_init(|| parse(EVIDENCE_JSON))
        .as_ref()
        .map(Vec::as_slice)
        .map_err(|e| format!("{EVIDENCE_TABLE_PATH}: {e}"))
}

/// Runtime initialization with an explicit byte-sensitive NEXT-allocation hook. A transient
/// capacity refusal is not cached as a permanent artifact parse failure. An already initialized
/// table allocates no new source payload; its retained bytes are already in the process snapshot.
pub fn rows_with_source_capacity(
    next: &mut dyn FnMut(u64) -> mlx_gen::Result<()>,
) -> Result<&'static [AnchorRow], String> {
    if ROWS.get().is_none() {
        let mut budget = mlx_gen_sdxl::sampling::SourceRequestBudget::with_capacity(
            EVIDENCE_JSON.len() as u64,
            next,
        );
        let rows = parse_with_source_budget(EVIDENCE_JSON, &mut budget)?;
        // Another initializer may win. Dropping this source copy is not physical retirement proof.
        let _ = ROWS.set(Ok(rows));
    }
    ROWS.get()
        .expect("reviewed rows initialized")
        .as_ref()
        .map(Vec::as_slice)
        .map_err(|e| format!("{EVIDENCE_TABLE_PATH}: {e}"))
}

/// Exact-key lookup (an anchor's own key).
pub fn lookup<'a>(rows: &'a [AnchorRow], key: &EvidenceKey) -> Option<&'a AnchorRow> {
    rows.iter().find(|row| &row.key == key)
}

/// The anchors of `family`: `(resident, optimized)`.
pub fn family_anchors<'a>(
    rows: &'a [AnchorRow],
    family: &AnchorFamily,
) -> (Option<&'a AnchorRow>, Option<&'a AnchorRow>) {
    let of = |kind| {
        rows.iter()
            .find(|r| r.anchor_kind == kind && r.key.family().as_ref() == Ok(family))
    };
    (of(AnchorKind::Resident), of(AnchorKind::Optimized))
}

/// Test/fixture helpers: build well-formed rows without a device.
#[cfg(test)]
pub(crate) mod fixture {
    use super::*;

    pub fn capture(run: &str, baseline: u64, peak: u64, phases: &[(IosPhase, u64)]) -> Capture {
        // Synthetic admission/schema fixture ONLY; native execution coverage is proved by actual
        // callback tests, never by this fabricated measurement record.
        let mut sampling = SamplingCoverage::default();
        sampling
            .record("synthetic_fixture", "solve_complete", &[("fixture", 1)])
            .unwrap();
        let mut quantum = QuantumCoverage::default();
        quantum.record(Some(IosPhase::Denoise), "synthetic_kernel", vec![0]);
        let snapshot = |current, peak| KernelFootprintSnapshot {
            method: KernelFootprintMethod::TaskVmInfo,
            query_status: 0,
            requested_words: 93,
            returned_words: 93,
            current_required_words: 38,
            peak_required_words: 84,
            current_bytes: Some(current),
            lifetime_peak_bytes: Some(peak),
        };
        Capture {
            run_id: run.into(),
            build_sha: "synthetic".into(),
            steps: 1,
            baseline_footprint_bytes: baseline,
            before_request_snapshot: Some(snapshot(baseline, baseline)),
            before_model_load_snapshot: Some(snapshot(baseline, baseline)),
            ending_snapshot: Some(snapshot(peak, peak)),
            observed_current_peak_bytes: Some(peak),
            observed_model_current_peak_bytes: Some(peak),
            valid_kernel_queries: 2,
            valid_model_kernel_queries: 2,
            footprint_bytes: peak,
            phase_peaks: phases
                .iter()
                .map(|(p, b)| {
                    (
                        p.as_key().to_string(),
                        PhasePeak {
                            peak_bytes: *b,
                            exact: true,
                        },
                    )
                })
                .collect(),
            sampling_coverage: Some(sampling),
            quantum_coverage: Some(quantum),
            quantum_query_gaps: 0,
            quanta: BTreeMap::new(),
            mlx_peak_bytes: 1,
            min_headroom_mib: 1.0,
            elapsed_seconds: 1.0,
        }
    }

    /// Change a synthetic capture's WHOLE-RUN kernel ledger consistently. Preserve role and
    /// sampling observations; callers changing phase-attributed peaks must update those separately.
    pub fn set_kernel_footprints(capture: &mut Capture, baseline: u64, peak: u64) {
        assert!(baseline > 0 && peak > baseline);
        let ledger = self::capture("synthetic-ledger", baseline, peak, &[]);
        capture.baseline_footprint_bytes = ledger.baseline_footprint_bytes;
        capture.before_request_snapshot = ledger.before_request_snapshot;
        capture.before_model_load_snapshot = ledger.before_model_load_snapshot;
        capture.ending_snapshot = ledger.ending_snapshot;
        capture.observed_current_peak_bytes = ledger.observed_current_peak_bytes;
        capture.observed_model_current_peak_bytes = ledger.observed_model_current_peak_bytes;
        capture.valid_kernel_queries = ledger.valid_kernel_queries;
        capture.valid_model_kernel_queries = ledger.valid_model_kernel_queries;
        capture.footprint_bytes = ledger.footprint_bytes;
    }

    /// A completed anchor whose repeats all measure `demand` over a 1-byte baseline (phase-blind).
    pub fn completed(
        key: EvidenceKey,
        kind: AnchorKind,
        selection: AnchorSelection,
        demand: u64,
    ) -> AnchorRow {
        let repeats = match kind {
            AnchorKind::Optimized => MIN_OPTIMIZED_REPEATS,
            AnchorKind::Resident => 1,
        };
        let roles = [
            ("reference_count", u64::from(key.reference_count)),
            ("mask_count", u64::from(key.mask_count)),
            ("control_count", u64::from(key.control_count)),
        ];
        AnchorRow {
            key,
            anchor_kind: kind,
            selection,
            outcome: AnchorOutcome::Completed {
                captures: (0..repeats)
                    .map(|i| {
                        let mut capture = capture(&format!("r{i}"), 1, demand + 1, &[]);
                        // Deliberately synthetic schema fixture, not a production witness/proof.
                        capture
                            .sampling_coverage
                            .as_mut()
                            .unwrap()
                            .record("ios_capture", "conditioning-roles", &roles)
                            .unwrap();
                        capture
                    })
                    .collect(),
            },
            evidence_class: DEVICE_CAPTURE_CLASS.into(),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::fixture::*;
    use super::*;

    #[test]
    fn sampling_observations_are_bounded_and_dangling_quantum_contexts_fail_closed() {
        let mut coverage = SamplingCoverage::default();
        assert_eq!(
            coverage.record(
                "curated",
                "solver_primary",
                &[("step", 0), ("history_denoised", 1)]
            ),
            Some(0)
        );
        assert_eq!(coverage.counts["curated"]["solver_primary"], 1);
        assert_eq!(
            coverage.record("curated", "solver_primary", &[("step", 1), ("step", 2)]),
            None
        );
        assert_eq!(coverage.malformed_marks, 1);
        assert!(coverage.truncated);
        coverage.validate().unwrap();
        let mut bounded = SamplingCoverage::default();
        for i in 0..BASE_SAMPLING_MARKS {
            assert!(bounded
                .record("curated", "solver_update", &[("step", i as u64)])
                .is_some());
        }
        assert!(bounded
            .record(
                "curated",
                "solve_complete",
                &[("evaluations", BASE_SAMPLING_MARKS as u64)]
            )
            .is_none());
        assert_eq!(bounded.marks.len(), BASE_SAMPLING_MARKS);
        assert_eq!(bounded.dropped_marks, 1);
        assert!(bounded.truncated);
        bounded.validate().unwrap();
        let mut row = completed(key(), AnchorKind::Optimized, optimized(), 1000);
        let AnchorOutcome::Completed { captures } = &mut row.outcome else {
            unreachable!()
        };
        captures[0].quanta.insert(
            "q".into(),
            QuantumObservation {
                priced_scratch_bytes: 1,
                observed_bytes: 1,
                exact: true,
                preceding_sampling_mark: Some(99),
            },
        );
        assert!(parse(&table(&[row]))
            .unwrap_err()
            .contains("missing sampling observation"));
    }

    fn key() -> EvidenceKey {
        EvidenceKey {
            machine: "iPhone17,2".into(),
            provider: "synthetic_task".into(),
            route: "synthetic_task".into(),
            tier: "q4".into(),
            layout_sha256: "ab".repeat(32),
            architecture_sha256: "cd".repeat(32),
            configured_variant_sha256: None,
            mode: "text_to_image".into(),
            width: 512,
            height: 512,
            batch: 1,
            reference_count: 0,
            mask_count: 0,
            control_count: 0,
            load_form: AnchorLoadForm::SequentialDeferred,
            execution: "Sequential|DeferredMaterialization|x".into(),
            sampling_recipe: None,
            sdxl_work: None,
            token_window: 0,
        }
    }

    #[test]
    fn evidence_family_keeps_actual_control_init_and_mask_roles() {
        let mut control = key();
        control.mode = "control".into();
        control.reference_count = 1; // aggregate: one Control, no true Reference.
        control.control_count = 1;
        let mut init = control.clone();
        init.reference_count = 2; // aggregate: one Control + one true Reference.
        let mut mask = init.clone();
        mask.mode = "edit".into();
        mask.control_count = 0;
        mask.mask_count = 1;
        assert_ne!(control.family().unwrap(), init.family().unwrap());
        assert_ne!(init.family().unwrap(), mask.family().unwrap());
        let mut alias = init.clone();
        alias.mode = "control_alias".into();
        assert_eq!(init.family().unwrap(), alias.family().unwrap());
        let mut other_geometry = init.clone();
        other_geometry.width *= 2;
        other_geometry.height *= 2;
        other_geometry.batch = 3;
        assert_eq!(init.family().unwrap(), other_geometry.family().unwrap());
        let mut multi = control.clone();
        multi.reference_count = 2; // aggregate of two loaded control branches.
        multi.control_count = 2;
        assert!(multi.family().is_err());
        multi.configured_variant_sha256 = Some("ef".repeat(32));
        assert_ne!(control.family().unwrap(), multi.family().unwrap());

        // Actual SDXL metadata replay revalidates roles before family classification, without
        // fabricating a media payload or accepting rejected combinations as an INIT family.
        let mut native = key();
        native.provider = crate::sdxl::SDXL_ID.into();
        native.route = "sdxl".into();
        let mut scalar = mlx_gen::GenerationRequest {
            sampler: Some("lcm".into()),
            ..Default::default()
        };
        native.sampling_recipe = Some(
            mlx_gen_sdxl::sampling::SamplingRecipe::from_request(&scalar)
                .unwrap()
                .canonical_token(),
        );
        native.mode = "ip_adapter".into();
        native.reference_count = 1;
        assert!(native.family().is_err()); // Actual early acceleration+Reference refusal.
        scalar.sampler = None;
        native.sampling_recipe = Some(
            mlx_gen_sdxl::sampling::SamplingRecipe::from_request(&scalar)
                .unwrap()
                .canonical_token(),
        );
        native.mode = "edit".into();
        native.reference_count = 3; // Reference + Mask + Control, before native policy refusal.
        native.mask_count = 1;
        native.control_count = 1;
        assert!(native.family().is_err()); // Actual mask+Control refusal.
        native.control_count = 0;
        native.reference_count = 2; // supported Reference + Mask aggregate.
        assert!(native.family().is_ok()); // Actual supported single-reference mask graph.
    }

    fn optimized() -> AnchorSelection {
        AnchorSelection {
            rung: 4,
            stage_residency: Some(true),
            decode_tile_edge: Some(256),
            ..Default::default()
        }
    }

    fn table(rows: &[AnchorRow]) -> String {
        serde_json::json!({"schema": 3, "about": "t", "anchors": rows}).to_string()
    }

    #[test]
    fn the_committed_table_parses_and_is_fail_closed_until_reviewed_rows_land() {
        let rows = rows().expect("the committed evidence table parses");
        assert!(rows
            .iter()
            .all(|r| r.evidence_class == DEVICE_CAPTURE_CLASS));
        assert!(lookup(rows, &key()).is_none());
    }

    #[test]
    fn anchor_rows_round_trip_and_malformed_rows_fail_the_whole_table_closed() {
        let good = completed(key(), AnchorKind::Optimized, optimized(), 3 << 30);
        assert_eq!(
            parse(&table(std::slice::from_ref(&good))).unwrap(),
            vec![good.clone()]
        );
        // A resident refusal is evidence; an optimized refusal is a defect.
        let refused = |kind, rung| AnchorRow {
            key: EvidenceKey {
                execution: "Resident|EagerMaterialization|resident".into(),
                load_form: AnchorLoadForm::ResidentEager,
                ..key()
            },
            anchor_kind: kind,
            selection: AnchorSelection {
                rung,
                ..Default::default()
            },
            outcome: AnchorOutcome::Refused {
                refused_quantum: RefusedQuantum {
                    label: "SDXL UNet load floor".into(),
                    scratch_bytes: 5 << 30,
                    reserve_bytes: 512 << 20,
                    available_bytes: Some(4 << 30),
                },
                run_id: "r".into(),
                build_sha: "s".into(),
            },
            evidence_class: DEVICE_CAPTURE_CLASS.into(),
        };
        assert_eq!(
            parse(&table(&[good.clone(), refused(AnchorKind::Resident, 0)]))
                .unwrap()
                .len(),
            2
        );
        assert!(parse(&table(&[refused(AnchorKind::Optimized, 4)])).is_err());
        // Fewer than two optimized repeats: no margin basis.
        let mut short = good.clone();
        if let AnchorOutcome::Completed { captures } = &mut short.outcome {
            captures.pop();
        }
        assert!(parse(&table(&[short])).is_err());
        // Repeats must be distinct runs.
        let mut same = good.clone();
        if let AnchorOutcome::Completed { captures } = &mut same.outcome {
            for c in captures.iter_mut() {
                c.run_id = "one".into();
            }
        }
        assert!(parse(&table(&[same])).is_err());
        // Baseline missing / at the peak, a phase peak outside [baseline, peak], an unknown phase.
        for mutate in [
            (|c: &mut Capture| c.baseline_footprint_bytes = 0) as fn(&mut Capture),
            |c| c.baseline_footprint_bytes = c.footprint_bytes,
            |c| {
                c.phase_peaks.insert(
                    "decode".into(),
                    PhasePeak {
                        peak_bytes: c.footprint_bytes + 1,
                        exact: true,
                    },
                );
            },
            |c| {
                c.phase_peaks.insert(
                    "vibes".into(),
                    PhasePeak {
                        peak_bytes: c.footprint_bytes,
                        exact: true,
                    },
                );
            },
            |c| c.steps = 0,
        ] {
            let mut bad = good.clone();
            if let AnchorOutcome::Completed { captures } = &mut bad.outcome {
                mutate(&mut captures[0]);
            }
            assert!(parse(&table(&[bad])).is_err());
        }
        // Rung/kind mismatch, a second optimized anchor in one family, non-device class, hex.
        let mut rung0 = good.clone();
        rung0.selection.rung = 0;
        assert!(parse(&table(&[rung0])).is_err());
        let mut twin = good.clone();
        twin.key.execution = "Sequential|DeferredMaterialization|y".into();
        assert!(parse(&table(&[good.clone(), twin])).is_err());
        let mut host = good.clone();
        host.evidence_class = "mac_host_peak".into();
        assert!(parse(&table(&[host])).is_err());
        let mut upper = good.clone();
        upper.key.layout_sha256 = "AB".repeat(32);
        assert!(parse(&table(&[upper])).is_err());
        // Schema 2 (the grid) and unknown fields are refused.
        assert!(parse(r#"{"schema":2,"about":"","anchors":[]}"#).is_err());
        assert!(parse(r#"{"schema":2,"about":"","rows":[]}"#).is_err());
        assert!(parse(r#"{"schema":3,"about":"","anchors":[],"extra":1}"#).is_err());
    }

    #[test]
    fn demand_phases_and_margin_come_from_the_repeats() {
        let mut row = completed(key(), AnchorKind::Optimized, optimized(), 0);
        row.outcome = AnchorOutcome::Completed {
            captures: vec![
                capture("a", 100, 1100, &[(IosPhase::Decode, 900)]),
                capture("b", 200, 1260, &[(IosPhase::Decode, 1260)]),
                capture("c", 100, 1150, &[(IosPhase::Denoise, 700)]),
            ],
        };
        assert_eq!(row.demand_bytes(), Some(1060));
        assert_eq!(
            row.phase_demands(),
            BTreeMap::from([(IosPhase::Decode, 1060), (IosPhase::Denoise, 600)])
        );
        // Spread 1060 - 1000 = 60 bytes: below the floor, so the floor.
        assert_eq!(row.margin_bytes(), Some(ANCHOR_MARGIN_FLOOR_BYTES));
        let wide = [
            capture("a", 1, 1 + (100 << 20), &[]),
            capture("b", 1, 1 + (180 << 20), &[]),
            capture("c", 1, 1 + (150 << 20), &[]),
        ];
        assert_eq!(margin_bytes(&wide), Some(80 << 20));
        assert_eq!(margin_bytes(&[]), None);
    }

    #[test]
    fn the_quantum_calibration_keeps_each_formulas_worst_exact_device_ratio() {
        let mut row = completed(key(), AnchorKind::Optimized, optimized(), 1000);
        let q = |priced, observed, exact| QuantumObservation {
            priced_scratch_bytes: priced,
            observed_bytes: observed,
            exact,
            preceding_sampling_mark: None,
        };
        if let AnchorOutcome::Completed { captures } = &mut row.outcome {
            captures.push(capture("r2", 1, 1001, &[]));
            captures[0]
                .quanta
                .insert("decode tile".into(), q(400, 200, true));
            captures[1]
                .quanta
                .insert("decode tile".into(), q(400, 260, true));
            // Masked (an upper bound) never calibrates, even when it reads higher.
            captures[2]
                .quanta
                .insert("decode tile".into(), q(400, 390, false));
            captures[0]
                .quanta
                .insert("denoise block".into(), q(100, 90, false));
            captures[1]
                .quanta
                .insert("text layer".into(), q(0, 10, true));
        }
        let calibration = row.quantum_calibration();
        assert_eq!(
            calibration.keys().collect::<Vec<_>>(),
            ["decode tile"],
            "masked and zero-priced observations do not calibrate"
        );
        assert_eq!(calibration["decode tile"].observed_bytes, 260);
        assert_eq!(calibration["decode tile"].priced_bytes, 400);
        // Round-trips, and an observation above the lifetime peak fails the table closed.
        assert_eq!(
            parse(&table(std::slice::from_ref(&row))).unwrap(),
            vec![row.clone()]
        );
        if let AnchorOutcome::Completed { captures } = &mut row.outcome {
            let peak = captures[0].footprint_bytes;
            captures[0].quanta.insert("x".into(), q(1, peak + 1, true));
        }
        assert!(parse(&table(&[row])).is_err());
    }

    #[test]
    fn selections_round_trip_structurally() {
        let p = MemoryStrategyParameters {
            stage_residency: Some(true),
            decode_tile_edge: Some(256),
            decode_overlap: Some(64),
            attention_chunk_size: Some(1 << 20),
            transformer_window_size: Some(1),
            transformer_window_component: Some(TransformerComponent::Both),
        };
        let s = AnchorSelection::from_parts(MemoryStrategy::BoundedTransformerResidency, &p);
        assert_eq!(s.rung, 4);
        assert_eq!(s.parameters(), Some(p));
        assert_eq!(
            s.strategy(),
            Some(MemoryStrategy::BoundedTransformerResidency)
        );
        let bad = AnchorSelection {
            transformer_window_component: Some("x".into()),
            ..s.clone()
        };
        assert!(bad.parameters().is_none());
    }
    /// Wire identity/receipt protocol only: synthetic metadata never obtains a
    /// source/native/physical certificate or a successful model admission.
    #[test]
    fn completed_sana_programme_key_requires_actual_complete_receipts() {
        let provider = "sana_1600m";
        let execution = format!(
            "ordinary{}",
            mlx_gen_sana::ios_pricing::NATIVE_RETIREMENT_EXECUTION_SUFFIX
        );
        let mut metadata = SamplingCoverage::default();
        let cache = [
            ("selected_stream", 1),
            ("stream_mask", 2),
            ("stream_count", 1),
            ("stream_0", 0),
            ("stream_1", 1),
            ("stream_2", 0),
            ("compute_cpu", 1),
        ];
        let start = metadata
            .record(
                provider,
                "native-lifecycle",
                &[
                    ("schema", 1),
                    ("invocation", 9),
                    ("started", 1),
                    ("configured", 1),
                    ("epoch", 0),
                    ("exclusive_request_ownership", 0),
                    ("host_reserve_declared", 0),
                    cache[0],
                    cache[1],
                    cache[2],
                    cache[3],
                    cache[4],
                    cache[5],
                    cache[6],
                ],
            )
            .unwrap();
        let label = mlx_gen_sana::ios_pricing::GEMMA_LABELS[0];
        let label_id = mlx_gen::ios_quantum::retirement_label_id(label);
        let pre = metadata
            .record(
                "mlx",
                "native-quantum",
                &[
                    ("schema", 1),
                    ("invocation", 9),
                    ("epoch", 1),
                    ("label", label_id),
                ],
            )
            .unwrap();
        let mut quantum = QuantumCoverage::default();
        quantum.record(Some(IosPhase::Conditioning), label, vec![start, pre]);
        let joined = metadata
            .record(
                "mlx",
                "native-joined",
                &[
                    ("schema", 1),
                    ("invocation", 9),
                    ("epoch", 1),
                    ("label", label_id),
                    ("all_source_streams", 1),
                    ("status", 1),
                    cache[0],
                    cache[1],
                    cache[2],
                    cache[3],
                    cache[4],
                    cache[5],
                    cache[6],
                    ("range_captured_prefix", 1),
                    ("exclusive_request_ownership", 0),
                    ("alias_release_credit", 0),
                    ("cache_release_credit", 0),
                    ("process_release_credit", 0),
                    ("caller_receipts_live_upper", 1),
                    ("host_reserve_declared", 0),
                ],
            )
            .unwrap();
        metadata
            .record(
                provider,
                "native-lifecycle-completed",
                &[
                    ("schema", 1),
                    ("invocation", 9),
                    ("status", 1),
                    ("epochs", 1),
                ],
            )
            .unwrap();
        validate_sana_capture_program(provider, &execution, &metadata, Some(&quantum)).unwrap();
        assert!(
            validate_sana_capture_program(provider, "ordinary", &metadata, Some(&quantum)).is_err(),
            "capture query must describe forced programme"
        );
        assert!(validate_sana_capture_program(provider, &execution, &metadata, None).is_err());
        let mut missing = metadata.clone();
        missing.marks[joined as usize].kind = "missing-native-joined".into();
        assert!(
            validate_sana_capture_program(provider, &execution, &missing, Some(&quantum)).is_err()
        );
        let mut repeated = quantum;
        repeated.checks[0].count = 2;
        assert!(
            validate_sana_capture_program(provider, &execution, &metadata, Some(&repeated))
                .is_err()
        );
        validate_sana_capture_program(provider, "ordinary", &SamplingCoverage::default(), None)
            .unwrap();
        // Failed/cancelled rows never call this completed-evidence validator;
        // their existing diagnostic serialization and primary reason survive.
    }

    #[test]
    fn source_composed_observation_budget_covers_existing_check_bound() {
        let mut coverage = SamplingCoverage::default();
        for _ in 0..MAX_QUANTUM_CHECKS {
            for (provider, kind) in [
                ("mlx", "native-quantum"),
                ("mlx", "native-joined"),
                ("sana", "native-dcae-glu"),
            ] {
                assert!(coverage.record(provider, kind, &[("schema", 1)]).is_some());
            }
        }
        for (provider, kind) in [
            ("sana_1600m", "native-lifecycle"),
            ("sana_1600m", "native-lifecycle-completed"),
            ("sana", "native-owned-phase-retired"),
        ] {
            assert!(coverage.record(provider, kind, &[("schema", 1)]).is_some());
        }
        for _ in 0..BASE_SAMPLING_MARKS {
            assert!(coverage
                .record("existing", "ordinary", &[("schema", 1)])
                .is_some());
        }
        assert_eq!(coverage.marks.len(), MAX_SAMPLING_MARKS);
        assert!(!coverage.truncated);
        coverage.validate().unwrap();
        assert!(coverage
            .record("overflow", "actual-extra", &[("schema", 1)])
            .is_none());
        assert!(coverage.truncated);
        // Storage shape/guard chronology only; no step/render/RAM allowance.
    }

    #[test]
    fn ordinary_and_native_observation_categories_cannot_borrow_each_others_capacity() {
        for (provider, kind, limit) in [
            ("ordinary", "actual", BASE_SAMPLING_MARKS),
            ("mlx", "native-quantum", MAX_QUANTUM_CHECKS),
            ("mlx", "native-joined", MAX_QUANTUM_CHECKS),
            ("sana", "native-dcae-glu", MAX_QUANTUM_CHECKS),
            (
                "sana_1600m",
                "native-lifecycle",
                NATIVE_RETIREMENT_ROOT_MARKS,
            ),
        ] {
            let mut coverage = SamplingCoverage::default();
            for _ in 0..limit {
                assert!(coverage.record(provider, kind, &[("schema", 1)]).is_some());
            }
            assert!(coverage.marks.len() < MAX_SAMPLING_MARKS);
            assert!(
                coverage.record(provider, kind, &[("schema", 1)]).is_none(),
                "category {provider}/{kind} escaped independent bound"
            );
            assert!(coverage.truncated);
            assert_eq!(coverage.dropped_marks, 1);
        }
    }
}
