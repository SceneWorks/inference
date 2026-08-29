//! Source-owned evidence primitives for the SC-20671 dense baseline campaign.
//!
//! This module deliberately keeps the receipt producer beside the product decoder.  The JSON
//! harness can validate evidence, but it must not be the component inventing model identity,
//! geometry, or lifecycle observations.  Device collection is supplied by the campaign runner;
//! these helpers make its inputs deterministic and testable without weights or Metal.

use sha2::{Digest, Sha256};
use std::fs;
use std::path::{Path, PathBuf};
use std::time::{SystemTime, UNIX_EPOCH};

use core_llm::{Message, Role, Sampling, StreamEvent, TextLlm, TextLlmOutput, TextLlmRequest};

pub const REQUIRED_PHASES: [&str; 8] = [
    "process-start",
    "weights-loaded",
    "prefill-peak",
    "first-token",
    "decode-steady",
    "prompt-cache-reuse",
    "cancellation-cleanup",
    "post-run-release",
];

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct InventoryFile {
    pub path: String,
    pub bytes: u64,
    pub sha256: String,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct SnapshotInventory {
    pub root: PathBuf,
    pub files: Vec<InventoryFile>,
    pub bytes: u64,
    pub sha256: String,
}

/// Hash every resolved file in a HF snapshot.  Symlink names and blob names are never trusted.
pub fn inventory_snapshot(root: impl AsRef<Path>) -> std::io::Result<SnapshotInventory> {
    fn visit(root: &Path, dir: &Path, out: &mut Vec<InventoryFile>) -> std::io::Result<()> {
        let mut entries = fs::read_dir(dir)?.collect::<Result<Vec<_>, _>>()?;
        entries.sort_by_key(|e| e.file_name());
        for entry in entries {
            let path = entry.path();
            let metadata = fs::symlink_metadata(&path)?;
            if metadata.is_dir() {
                visit(root, &path, out)?;
                continue;
            }
            if !metadata.is_file() && !metadata.file_type().is_symlink() {
                continue;
            }
            let resolved = fs::canonicalize(&path)?;
            let bytes = fs::metadata(&resolved)?.len();
            if bytes == 0 {
                return Err(std::io::Error::new(
                    std::io::ErrorKind::InvalidData,
                    format!("empty snapshot file {}", path.display()),
                ));
            }
            let data = fs::read(&resolved)?;
            let sha = hex(&Sha256::digest(&data));
            let relative = path
                .strip_prefix(root)
                .unwrap_or(&path)
                .to_string_lossy()
                .replace('\\', "/");
            out.push(InventoryFile {
                path: relative,
                bytes,
                sha256: sha,
            });
        }
        Ok(())
    }
    let root = fs::canonicalize(root)?;
    let mut files = Vec::new();
    visit(&root, &root, &mut files)?;
    files.sort_by(|a, b| a.path.cmp(&b.path));
    if files.is_empty() {
        return Err(std::io::Error::new(
            std::io::ErrorKind::InvalidData,
            "empty snapshot",
        ));
    }
    let mut digest = Sha256::new();
    for file in &files {
        digest.update(file.path.as_bytes());
        digest.update([0]);
        digest.update(file.bytes.to_string().as_bytes());
        digest.update([0]);
        digest.update(file.sha256.as_bytes());
        digest.update([b'\n']);
    }
    Ok(SnapshotInventory {
        root,
        bytes: files.iter().map(|f| f.bytes).sum(),
        sha256: hex(&digest.finalize()),
        files,
    })
}

fn hex(bytes: &[u8]) -> String {
    bytes.iter().map(|b| format!("{b:02x}")).collect()
}

/// Exact dense KV theoretical size, including batch and K+V.
pub fn dense_kv_bytes(
    batch: u64,
    layers: u64,
    kv_heads: u64,
    capacity: u64,
    head_dimension: u64,
    element_bytes: u64,
) -> u64 {
    batch
        .saturating_mul(layers)
        .saturating_mul(kv_heads)
        .saturating_mul(capacity)
        .saturating_mul(head_dimension)
        .saturating_mul(element_bytes)
        .saturating_mul(2)
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct PhaseSample {
    pub phase: String,
    pub pid: u32,
    pub source: String,
    pub captured_at: String,
    pub footprint_bytes: u64,
    pub mlx_active_bytes: u64,
    pub mlx_cache_bytes: u64,
    pub mlx_peak_bytes: u64,
}

/// Validate the non-forgeable structural part of phase evidence before serialization.
pub fn validate_phases(samples: &[PhaseSample]) -> Result<(), String> {
    if samples.len() != REQUIRED_PHASES.len() {
        return Err("phase set is incomplete or duplicated".into());
    }
    let pid = samples
        .first()
        .map(|s| s.pid)
        .filter(|p| *p > 0)
        .ok_or("worker PID is missing")?;
    for (sample, expected) in samples.iter().zip(REQUIRED_PHASES) {
        if sample.phase != expected {
            return Err(format!("expected phase {expected}, got {}", sample.phase));
        }
        if sample.pid != pid || sample.source.is_empty() || sample.captured_at.is_empty() {
            return Err(format!("invalid evidence for {expected}"));
        }
    }
    Ok(())
}

/// The producer's cancellation invariant: cleanup evidence is required even if cancellation was
/// already set before inference began.
pub fn cancellation_cleanup_required(
    aborted_before_start: bool,
    cleanup_observed: bool,
    released: bool,
) -> Result<(), String> {
    if (aborted_before_start || cleanup_observed) && !released {
        return Err("cancellation did not release resources".into());
    }
    Ok(())
}

/// Stable seal over the exact bytes written to a receipt or fixture artifact.
pub fn seal_bytes(bytes: &[u8]) -> String {
    hex(&Sha256::digest(bytes))
}

pub const REQUIRED_FIXTURES: [&str; 4] = [
    "kernel-fp32-reference",
    "structured-tool-call",
    "long-context-needle",
    "multi-turn-prompt-cache",
];

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct FixtureEvidence {
    pub name: String,
    pub artifact_sha256: String,
    pub independent_reference: String,
    pub passed: bool,
}

pub fn validate_fixture_evidence(evidence: &[FixtureEvidence]) -> Result<(), String> {
    if evidence.len() != REQUIRED_FIXTURES.len() || !evidence.iter().all(|e| e.passed) {
        return Err("all four independent quality fixtures must pass".into());
    }
    for (item, expected) in evidence.iter().zip(REQUIRED_FIXTURES) {
        if item.name != expected
            || item.artifact_sha256.len() != 64
            || !item.artifact_sha256.bytes().all(|b| b.is_ascii_hexdigit())
            || item.independent_reference.is_empty()
        {
            return Err(format!("invalid sealed evidence for {expected}"));
        }
    }
    Ok(())
}

/// A monotonic timestamp suitable for the producer's internal sequencing tests.
pub fn sequence_marker() -> u128 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_nanos()
}

/// The narrow observation seam used by a real campaign runner.  The runner owns platform probes
/// (`footprint` and `mlx_rs::memory`); the product path owns the phase boundaries and generation.
pub trait Observer {
    fn phase(&mut self, name: &'static str);
    fn allocation(&mut self, role: &'static str, lifetime: &'static str, bytes: u64);
}

/// Run one dense product-path coordinate.  This intentionally loads through `LlamaProvider` and
/// `core-llm::TextLlm`, rather than a test double or HTTP route.  The lower-level cache observation
/// callbacks remain a separate seam because MLX arrays are backend-owned and cannot cross the
/// backend-neutral contract.  A campaign runner must call this in a fresh process per coordinate.
pub fn run_dense_coordinate(
    snapshot: impl AsRef<Path>,
    prompt: &str,
    max_new_tokens: u32,
    observer: &mut dyn Observer,
) -> core_llm::Result<TextLlmOutput> {
    observer.phase("process-start");
    let _inventory = inventory_snapshot(snapshot.as_ref())
        .map_err(|e| core_llm::Error::Load(format!("snapshot inventory: {e}")))?;
    let provider = crate::provider::LlamaProvider::load(&core_llm::LoadSpec::dense(
        snapshot.as_ref().to_string_lossy().to_string(),
    ))?;
    observer.phase("weights-loaded");
    let request = TextLlmRequest {
        messages: vec![Message::text(Role::User, prompt)],
        sampling: Sampling {
            temperature: 0.0,
            top_p: 1.0,
            ..Default::default()
        },
        max_new_tokens,
        seed: Some(0),
        ..Default::default()
    };
    let mut saw_token = false;
    let output = provider.generate(&request, &mut |event| {
        if !saw_token && matches!(event, StreamEvent::Token { .. }) {
            saw_token = true;
            observer.phase("first-token");
        }
    })?;
    if !saw_token {
        return Err(core_llm::Error::InvalidRequest(
            "dense campaign produced no first-token observation".into(),
        ));
    }
    observer.phase("decode-steady");
    Ok(output)
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Write;

    #[test]
    fn formula_includes_batch_and_key_value_pair() {
        assert_eq!(dense_kv_bytes(2, 3, 4, 5, 6, 2), 1440);
    }

    #[test]
    fn inventory_hashes_resolved_shards_deterministically() {
        let dir = tempfile::tempdir().unwrap();
        fs::write(dir.path().join("b.safetensors"), b"b").unwrap();
        let mut file = fs::File::create(dir.path().join("a.safetensors")).unwrap();
        file.write_all(b"a").unwrap();
        let first = inventory_snapshot(dir.path()).unwrap();
        let second = inventory_snapshot(dir.path()).unwrap();
        assert_eq!(first.sha256, second.sha256);
        assert_eq!(first.files[0].path, "a.safetensors");
    }

    #[test]
    fn phases_are_exact_and_single_pid() {
        let samples = REQUIRED_PHASES
            .iter()
            .map(|phase| PhaseSample {
                phase: (*phase).into(),
                pid: 42,
                source: "test".into(),
                captured_at: "2026-01-01T00:00:00Z".into(),
                footprint_bytes: 1,
                mlx_active_bytes: 1,
                mlx_cache_bytes: 1,
                mlx_peak_bytes: 1,
            })
            .collect::<Vec<_>>();
        assert!(validate_phases(&samples).is_ok());
    }

    #[test]
    fn cancellation_before_start_still_requires_release() {
        assert!(cancellation_cleanup_required(true, false, false).is_err());
        assert!(cancellation_cleanup_required(true, false, true).is_ok());
    }

    #[test]
    fn seal_includes_exact_newline_bytes() {
        assert_ne!(seal_bytes(b"{}"), seal_bytes(b"{}\n"));
    }

    #[test]
    fn fixture_evidence_is_fail_closed() {
        let evidence = REQUIRED_FIXTURES
            .iter()
            .map(|name| FixtureEvidence {
                name: (*name).into(),
                artifact_sha256: "a".repeat(64),
                independent_reference: "fp32-reference".into(),
                passed: true,
            })
            .collect::<Vec<_>>();
        assert!(validate_fixture_evidence(&evidence).is_ok());
        assert!(validate_fixture_evidence(&evidence[..3]).is_err());
    }
}
