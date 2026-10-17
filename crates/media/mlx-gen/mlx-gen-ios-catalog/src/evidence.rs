//! The single source-controlled iOS device evidence table (epic 16774, E4/E7).
//!
//! `evidence/ios_device_evidence.json` is compiled in. A row is a reviewed device capture keyed on
//! MEMORY-RELEVANT facts only: hardware, provider, route, tier, a path-free artifact layout digest,
//! mode, geometry, reference count, execution levers and token window. Prompt text, seed, install
//! path and conditioning pixels are deliberately absent (they do not change the footprint at a fixed
//! token window and geometry). Admission is fail-closed: no matching row, no admission. Nothing at
//! runtime writes this table; a capture prints candidate rows that a human reviews and commits.

use serde::{Deserialize, Serialize};
use std::sync::OnceLock;

/// The repository path of the table, for capture tooling and review messages.
pub const EVIDENCE_TABLE_PATH: &str =
    "crates/media/mlx-gen/mlx-gen-ios-catalog/evidence/ios_device_evidence.json";

/// The only evidence class a row may carry: a physical-device capture run.
pub const DEVICE_CAPTURE_CLASS: &str = "device_capture";

const EVIDENCE_JSON: &str = include_str!("../evidence/ios_device_evidence.json");

/// The memory-relevant identity of one request on one artifact and device.
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
    /// `MemoryMode::as_key` of the request route.
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

/// One reviewed device capture.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct EvidenceRow {
    pub key: EvidenceKey,
    /// Kernel-ledger lifetime `phys_footprint` peak of the capture process, bytes (ABSOLUTE: it
    /// includes the harness's own baseline).
    pub footprint_bytes: u64,
    /// The capture process's `phys_footprint` when the case started, before any of its loads.
    /// The render's own demand is `footprint_bytes - baseline_footprint_bytes`; admission charges
    /// only that growth against live headroom (never credits a product's unrelated residents).
    pub baseline_footprint_bytes: u64,
    /// MLX `get_peak_memory`, bytes (informational).
    pub mlx_peak_bytes: u64,
    /// Minimum sampled `os_proc_available_memory`, MiB (informational).
    pub min_headroom_mib: f64,
    pub elapsed_seconds: f64,
    /// Always [`DEVICE_CAPTURE_CLASS`].
    pub evidence_class: String,
    pub run_id: String,
    pub build_sha: String,
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct Table {
    schema: u32,
    #[allow(dead_code)]
    about: String,
    rows: Vec<EvidenceRow>,
}

fn validate(rows: &[EvidenceRow]) -> Result<(), String> {
    let mut seen = std::collections::BTreeSet::new();
    for row in rows {
        let k = &row.key;
        if row.evidence_class != DEVICE_CAPTURE_CLASS {
            return Err(format!(
                "row {k:?}: evidence_class must be {DEVICE_CAPTURE_CLASS}"
            ));
        }
        if row.footprint_bytes == 0 || row.run_id.is_empty() || row.build_sha.is_empty() {
            return Err(format!("row {k:?}: missing footprint/run/build provenance"));
        }
        if row.baseline_footprint_bytes == 0 || row.baseline_footprint_bytes >= row.footprint_bytes
        {
            return Err(format!(
                "row {k:?}: capture baseline footprint must be present and below its peak"
            ));
        }
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
        if !seen.insert(k.clone()) {
            return Err(format!("row {k:?}: duplicate key"));
        }
    }
    Ok(())
}

pub(crate) fn parse(json: &str) -> Result<Vec<EvidenceRow>, String> {
    let table: Table = serde_json::from_str(json).map_err(|e| e.to_string())?;
    // Schema 2 added the capture baseline footprint; a schema-1 row cannot say how much of its
    // peak was the harness itself, so it is refused (fail closed), never migrated by guess.
    if table.schema != 2 {
        return Err(format!("unknown evidence schema {}", table.schema));
    }
    validate(&table.rows)?;
    Ok(table.rows)
}

/// The compiled-in, reviewed table. A malformed table admits nothing (every lookup refuses).
pub fn rows() -> Result<&'static [EvidenceRow], String> {
    static ROWS: OnceLock<Result<Vec<EvidenceRow>, String>> = OnceLock::new();
    ROWS.get_or_init(|| parse(EVIDENCE_JSON))
        .as_ref()
        .map(Vec::as_slice)
        .map_err(|e| format!("{EVIDENCE_TABLE_PATH}: {e}"))
}

/// Exact-key lookup. No nearest neighbour, no interpolation, no cross-key borrowing.
pub fn lookup<'a>(rows: &'a [EvidenceRow], key: &EvidenceKey) -> Option<&'a EvidenceRow> {
    rows.iter().find(|row| &row.key == key)
}

#[cfg(test)]
mod tests {
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

    fn row(key: EvidenceKey) -> EvidenceRow {
        EvidenceRow {
            key,
            footprint_bytes: 2,
            baseline_footprint_bytes: 1,
            mlx_peak_bytes: 1,
            min_headroom_mib: 1.0,
            elapsed_seconds: 1.0,
            evidence_class: DEVICE_CAPTURE_CLASS.into(),
            run_id: "r".into(),
            build_sha: "s".into(),
        }
    }

    fn table(rows: &[EvidenceRow]) -> String {
        serde_json::json!({"schema": 2, "about": "t", "rows": rows}).to_string()
    }

    #[test]
    fn the_committed_table_parses_and_is_fail_closed_until_reviewed_rows_land() {
        let rows = rows().expect("the committed evidence table parses");
        // Each committed row must be a device capture; an empty table admits nothing.
        assert!(rows
            .iter()
            .all(|r| r.evidence_class == DEVICE_CAPTURE_CLASS));
        assert!(lookup(rows, &key()).is_none());
    }

    #[test]
    fn malformed_or_non_device_rows_are_rejected_not_admitted() {
        assert_eq!(parse(&table(&[row(key())])).unwrap().len(), 1);
        let mut host = row(key());
        host.evidence_class = "mac_host_peak".into();
        assert!(parse(&table(&[host])).is_err());
        let mut zero = row(key());
        zero.footprint_bytes = 0;
        assert!(parse(&table(&[zero])).is_err());
        // The capture baseline is required, nonzero and below the peak: a row without it fails
        // the whole table closed (nothing is admitted from a malformed table).
        let mut no_baseline = row(key());
        no_baseline.baseline_footprint_bytes = 0;
        assert!(parse(&table(&[no_baseline])).is_err());
        let mut above = row(key());
        above.baseline_footprint_bytes = above.footprint_bytes;
        assert!(parse(&table(&[above])).is_err());
        let mut missing = serde_json::to_value(row(key())).unwrap();
        missing
            .as_object_mut()
            .unwrap()
            .remove("baseline_footprint_bytes");
        let missing = serde_json::json!({"schema": 2, "about": "t", "rows": [missing]});
        assert!(parse(&missing.to_string()).is_err());
        assert!(
            parse(&serde_json::json!({"schema": 1, "about": "t", "rows": []}).to_string()).is_err()
        );
        let mut bad = key();
        bad.layout_sha256 = "AB".repeat(32);
        assert!(parse(&table(&[row(bad)])).is_err());
        assert!(parse(&table(&[row(key()), row(key())])).is_err());
        assert!(parse(r#"{"schema":3,"about":"","rows":[]}"#).is_err());
        assert!(parse(r#"{"schema":2,"about":"","rows":[],"extra":1}"#).is_err());
    }

    #[test]
    fn lookup_is_exact_on_every_memory_relevant_field() {
        let rows = vec![row(key())];
        assert!(lookup(&rows, &key()).is_some());
        type Change = Box<dyn Fn(&mut EvidenceKey)>;
        let variants: Vec<Change> = vec![
            Box::new(|k| k.machine = "iPhone18,2".into()),
            Box::new(|k| k.provider = "sana_sprint_1600m".into()),
            Box::new(|k| k.route = "other".into()),
            Box::new(|k| k.tier = "q8".into()),
            Box::new(|k| k.layout_sha256 = "cd".repeat(32)),
            Box::new(|k| k.mode = "image_to_image".into()),
            Box::new(|k| k.width = 768),
            Box::new(|k| k.height = 768),
            Box::new(|k| k.batch = 2),
            Box::new(|k| k.reference_count = 1),
            Box::new(|k| k.execution = "y".into()),
            Box::new(|k| k.token_window = 154),
        ];
        for change in variants {
            let mut k = key();
            change(&mut k);
            assert!(
                lookup(&rows, &k).is_none(),
                "{k:?} borrowed another key's row"
            );
        }
    }
}
