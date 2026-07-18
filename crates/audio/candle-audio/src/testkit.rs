//! Test-support helpers for the candle audio provider crates (`testkit` feature,
//! sc-12835) — the audio family's copy of `candle_gen::testkit`'s HF-cache resolution
//! (sc-9055 / F-069). Duplicated rather than imported: `crates/audio/` deliberately takes
//! no dependency on the media engine families (`crates/media/`), so these ~40 lines are
//! carried here with the same F-071 semantics. Keep the resolution order in sync with
//! `candle-gen/src/testkit.rs` if either changes.

use std::path::PathBuf;

/// The candidate HF Hub cache roots, in resolution order: `$HF_HUB_CACHE`, then
/// `$HF_HOME/hub`, then the user-home `.cache/huggingface/hub` default (`USERPROFILE` on
/// Windows, then `HOME`).
///
/// The Windows-primary dev box keeps the cache at `D:\.cache\huggingface` via `HF_HOME`,
/// where `HOME` is usually unset — resolvers that only consulted `$HOME/.cache/huggingface`
/// silently missed it (F-071 / sc-9057).
pub fn hf_cache_roots() -> Vec<PathBuf> {
    let mut roots: Vec<PathBuf> = Vec::new();
    if let Ok(c) = std::env::var("HF_HUB_CACHE") {
        roots.push(PathBuf::from(c));
    }
    if let Ok(h) = std::env::var("HF_HOME") {
        roots.push(PathBuf::from(h).join("hub"));
    }
    for home_var in ["USERPROFILE", "HOME"] {
        if let Ok(home) = std::env::var(home_var) {
            roots.push(PathBuf::from(home).join(".cache/huggingface/hub"));
        }
    }
    roots
}

/// Resolve the first existing `snapshots/<rev>/` directory for an HF repo under the
/// [`hf_cache_roots`], or `None` if the repo isn't cached anywhere. `repo` is the
/// `owner/name` form (e.g. `"hexgrad/Kokoro-82M"`) — it is normalized to the
/// `models--owner--name` cache dir.
pub fn hf_snapshot_dir(repo: &str) -> Option<PathBuf> {
    let repo_dir = format!("models--{}", repo.replace('/', "--"));
    for snapshots in hf_cache_roots()
        .into_iter()
        .map(|r| r.join(&repo_dir).join("snapshots"))
    {
        let Ok(revs) = std::fs::read_dir(&snapshots) else {
            continue;
        };
        if let Some(dir) = revs
            .filter_map(std::result::Result::ok)
            .map(|e| e.path())
            .find(|p| p.is_dir())
        {
            return dir.into();
        }
    }
    None
}

/// [`hf_snapshot_dir`] that panics with an actionable message if the repo isn't cached —
/// for real-weight tests that require the snapshot present.
pub fn require_hf_snapshot_dir(repo: &str) -> PathBuf {
    hf_snapshot_dir(repo).unwrap_or_else(|| {
        panic!(
            "{repo} snapshot not cached under any HF cache root \
             (HF_HUB_CACHE / HF_HOME/hub / <home>/.cache/huggingface/hub)"
        )
    })
}
