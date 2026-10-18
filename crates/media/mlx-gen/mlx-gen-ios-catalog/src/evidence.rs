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
/// their measured spread, never a blanket reserve.
pub const MIN_OPTIMIZED_REPEATS: usize = 3;

/// Floor of the low-end margin. Three repeats can agree to the byte, and a zero margin would be a
/// claim that the next render is bit-identical in footprint. This floor is an ENGINEERING choice,
/// not a measurement: 32 MiB is about 0.5% of the 6142 MiB entitled launch headroom and two orders of
/// magnitude below the 512 MiB formula reserve it replaces at an anchor. Replace it only with a
/// measured spread.
pub const ANCHOR_MARGIN_FLOOR_BYTES: u64 = 32 << 20;

const EVIDENCE_JSON: &str = include_str!("../evidence/ios_device_evidence.json");

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
    /// `MemoryMode::as_key` of the request route (the anchor family's variant axis).
    pub mode: String,
    pub width: u32,
    pub height: u32,
    pub batch: u32,
    pub reference_count: u32,
    /// Canonical selection + request execution levers ([`crate::ios_admission::execution_key`]).
    pub execution: String,
    /// Encoded text window: CLIP windows x 77 for SDXL, 0 where the provider pads to a fixed,
    /// artifact-determined window (covered by the layout digest).
    pub token_window: u32,
}

impl EvidenceKey {
    /// The anchor family this key belongs to: everything a derivation must hold EQUAL.
    pub fn family(&self) -> AnchorFamily {
        AnchorFamily {
            machine: self.machine.clone(),
            provider: self.provider.clone(),
            route: self.route.clone(),
            tier: self.tier.clone(),
            layout_sha256: self.layout_sha256.clone(),
            mode: self.mode.clone(),
        }
    }
}

/// What must be equal between a request and an anchor for the anchor law to apply. A different
/// mode (variant), tier, artifact layout (decoder / components), route or device is NOT derivable.
#[derive(Clone, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct AnchorFamily {
    pub machine: String,
    pub provider: String,
    pub route: String,
    pub tier: String,
    pub layout_sha256: String,
    pub mode: String,
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

/// One capture run of an anchor.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Capture {
    pub run_id: String,
    pub build_sha: String,
    /// Denoise steps the capture ran (anchors capture short: 1-2 steps).
    pub steps: u32,
    /// The capture process's `phys_footprint` before any of the case's loads.
    pub baseline_footprint_bytes: u64,
    /// Kernel-ledger LIFETIME `phys_footprint` peak of the capture process (absolute).
    pub footprint_bytes: u64,
    /// Per-phase peaks split through the quantum marks (`IosPhase::as_key` → peak). A phase the
    /// capture could not attribute is absent and is priced at the whole-run peak (erring large).
    pub phase_peaks: BTreeMap<String, PhasePeak>,
    /// MLX `get_peak_memory`, bytes (informational).
    pub mlx_peak_bytes: u64,
    /// Minimum sampled `os_proc_available_memory`, MiB (informational).
    pub min_headroom_mib: f64,
    pub elapsed_seconds: f64,
}

impl Capture {
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
    if k.layout_sha256.len() != 64
        || !k
            .layout_sha256
            .bytes()
            .all(|b| b.is_ascii_hexdigit() && !b.is_ascii_uppercase())
    {
        return Err(format!("row {k:?}: layout_sha256 must be 64 lowercase hex"));
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
    Ok(())
}

fn validate(rows: &[AnchorRow]) -> Result<(), String> {
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
        if row.selection.strategy().is_none() || row.selection.parameters().is_none() {
            return Err(format!("row {k:?}: malformed anchor selection"));
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
                    if c.baseline_footprint_bytes == 0
                        || c.baseline_footprint_bytes >= c.footprint_bytes
                    {
                        return Err(format!(
                            "row {k:?}: capture baseline footprint must be present and below its peak"
                        ));
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
        if !families.insert((k.family(), row.anchor_kind)) {
            return Err(format!(
                "row {k:?}: a family has at most one {:?} anchor",
                row.anchor_kind
            ));
        }
    }
    Ok(())
}

pub(crate) fn parse(json: &str) -> Result<Vec<AnchorRow>, String> {
    let table: Table = serde_json::from_str(json).map_err(|e| e.to_string())?;
    // Schema 3 replaced the exact-match grid with anchors; an older table is refused (fail
    // closed), never migrated by guess.
    if table.schema != EVIDENCE_SCHEMA {
        return Err(format!("unknown evidence schema {}", table.schema));
    }
    validate(&table.anchors)?;
    Ok(table.anchors)
}

/// The compiled-in, reviewed table. A malformed table admits nothing (every lookup refuses).
pub fn rows() -> Result<&'static [AnchorRow], String> {
    static ROWS: OnceLock<Result<Vec<AnchorRow>, String>> = OnceLock::new();
    ROWS.get_or_init(|| parse(EVIDENCE_JSON))
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
            .find(|r| r.anchor_kind == kind && &r.key.family() == family)
    };
    (of(AnchorKind::Resident), of(AnchorKind::Optimized))
}

/// Test/fixture helpers: build well-formed rows without a device.
#[cfg(test)]
pub(crate) mod fixture {
    use super::*;

    pub fn capture(run: &str, baseline: u64, peak: u64, phases: &[(IosPhase, u64)]) -> Capture {
        Capture {
            run_id: run.into(),
            build_sha: "synthetic".into(),
            steps: 1,
            baseline_footprint_bytes: baseline,
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
            mlx_peak_bytes: 1,
            min_headroom_mib: 1.0,
            elapsed_seconds: 1.0,
        }
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
        AnchorRow {
            key,
            anchor_kind: kind,
            selection,
            outcome: AnchorOutcome::Completed {
                captures: (0..repeats)
                    .map(|i| capture(&format!("r{i}"), 1, demand + 1, &[]))
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

    fn key() -> EvidenceKey {
        EvidenceKey {
            machine: "iPhone17,2".into(),
            provider: "sana_1600m".into(),
            route: "sana_1600m".into(),
            tier: "q4".into(),
            layout_sha256: "ab".repeat(32),
            mode: "text_to_image".into(),
            width: 512,
            height: 512,
            batch: 1,
            reference_count: 0,
            execution: "x".into(),
            token_window: 0,
        }
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
        assert_eq!(parse(&table(&[good.clone()])).unwrap(), vec![good.clone()]);
        // A resident refusal is evidence; an optimized refusal is a defect.
        let refused = |kind, rung| AnchorRow {
            key: EvidenceKey {
                execution: "resident".into(),
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
        // Fewer than three optimized repeats: no margin basis.
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
        twin.key.execution = "y".into();
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
}
