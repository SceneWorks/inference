//! Pinned-revision hf-hub weight resolution into [`gen_core::WeightsSource`] — the
//! `LoadSpec` snapshot path for candle audio providers (sc-12835).
//!
//! This is the exact idiom the existing candle media providers use for runtime hub
//! fetches (`candle-gen-sdxl::hf_get`, sc-9013 / F-029): every download is pinned to an
//! **immutable commit SHA**, never the hub's mutable `main` default, so an upstream
//! force-push or account compromise cannot silently alter weights at request time. Each
//! audio provider owns its own pin table (repo → 40-hex revision) next to its descriptor,
//! the way `candle-gen-sdxl` owns `HUB_PINS`; this module refuses anything that is not a
//! full commit SHA rather than falling back to a mutable ref.
//!
//! Resolved files land in the ordinary HF cache (`$HF_HUB_CACHE` → `$HF_HOME/hub` →
//! `~/.cache/huggingface/hub`), so a provider's `LoadSpec` weights interoperate with
//! snapshots prepared out-of-band through the core-llm snapshot-preparer flow — the audio
//! lane carries the candle preparer in every bundle (see `runtime-catalog`'s audio lane).

use std::path::PathBuf;

use gen_core::WeightsSource;

use crate::{AudioError, Result};

/// Whether `revision` is a full 40-hex-digit commit SHA — the only revision shape the
/// F-029 discipline accepts for a runtime download (branch names and tags are mutable).
fn is_commit_sha(revision: &str) -> bool {
    revision.len() == 40 && revision.chars().all(|c| c.is_ascii_hexdigit())
}

/// Resolve (download-or-cache) one file from an HF hub repo at a **pinned immutable
/// revision**. Mirrors `candle-gen-sdxl`'s `hf_get` (sc-9013 / F-029), with the pin passed
/// by the caller because each audio provider owns its own repo→SHA table.
///
/// Errors when `revision` is not a full 40-hex commit SHA — an unpinned runtime download on
/// the synthesis path is a supply-chain risk, so this refuses rather than silently
/// resolving a mutable ref.
pub fn hf_get_pinned(repo: &str, revision: &str, path: &str) -> Result<PathBuf> {
    use hf_hub::api::sync::Api;
    use hf_hub::{Repo, RepoType};
    if !is_commit_sha(revision) {
        return Err(AudioError::Msg(format!(
            "hf-hub fetch {repo}/{path}: revision {revision:?} is not a full 40-hex commit SHA — \
             pin an immutable revision (F-029), never a branch or tag"
        )));
    }
    Api::new()
        .and_then(|api| {
            api.repo(Repo::with_revision(
                repo.to_string(),
                RepoType::Model,
                revision.to_string(),
            ))
            .get(path)
        })
        .map_err(|e| AudioError::Msg(format!("hf-hub fetch {repo}/{path}@{revision}: {e}")))
}

/// [`hf_get_pinned`] wrapped as a single-file [`WeightsSource::File`] — for a provider whose
/// `LoadSpec` names one checkpoint file (e.g. a single `.safetensors`).
pub fn pinned_weights_file(repo: &str, revision: &str, path: &str) -> Result<WeightsSource> {
    Ok(WeightsSource::File(hf_get_pinned(repo, revision, path)?))
}

/// Resolve a pinned repo's **snapshot directory** as a [`WeightsSource::Dir`], by fetching
/// `probe_file` (a small, always-present file such as `config.json`) and taking its parent —
/// the hf-hub cache lays every file of one revision under a single `snapshots/<rev>/` dir.
/// For a provider whose `LoadSpec` names a snapshot directory (weights + voices + config).
///
/// Note this materializes only `probe_file`; the provider fetches its remaining files through
/// [`hf_get_pinned`] with the same pin (they land in the same snapshot dir).
pub fn pinned_snapshot_dir(repo: &str, revision: &str, probe_file: &str) -> Result<WeightsSource> {
    let probe = hf_get_pinned(repo, revision, probe_file)?;
    let dir = probe.parent().ok_or_else(|| {
        AudioError::Msg(format!(
            "hf-hub fetch {repo}/{probe_file}@{revision}: resolved cache path {} has no parent \
             directory",
            probe.display()
        ))
    })?;
    Ok(WeightsSource::Dir(dir.to_path_buf()))
}

#[cfg(test)]
mod tests {
    use super::*;

    // Offline tests only: the pin-shape gate must reject every mutable ref up front, before
    // any network activity. Successful fetches are exercised by the provider real-weight
    // tests (sc-12836), which are `#[ignore]`d and snapshot-gated like every other family's.

    #[test]
    fn rejects_unpinned_revisions_before_any_network_use() {
        for bad in ["main", "v1.0", "", "abc123", "MAIN", &"a".repeat(41)] {
            let err = hf_get_pinned("owner/repo", bad, "config.json").unwrap_err();
            assert!(
                err.to_string().contains("not a full 40-hex commit SHA"),
                "revision {bad:?} must be refused as unpinned, got: {err}"
            );
        }
    }

    #[test]
    fn accepts_only_full_hex_shas() {
        assert!(is_commit_sha("91b3b1eb141d1a1b30bd5a58c2b1c9dfd7b31469"));
        assert!(!is_commit_sha("91B3B1EB141D1A1B30BD5A58C2B1C9DFD7B3146")); // 39 chars
        assert!(!is_commit_sha("zzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzz")); // not hex
    }
}
