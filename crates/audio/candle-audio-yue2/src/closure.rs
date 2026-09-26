//! Self-contained model closures (sc-22994): save the verified generation closure to one
//! directory and load an engine back from it, offline.
//!
//! Ported from the pinned upstream `YuE2Pipeline.save_pretrained` / `from_pretrained` on a saved
//! directory and `storage.copy_model_files` (`src/yue2/pipeline.py`, `src/yue2/storage.py`). A saved
//! closure is
//!
//! ```text
//! <dir>/pipeline.json          format, sub-directories, generation config, source identities, licence
//! <dir>/YuE2-3B/…              every pinned file of YuE2-3B (incl. qwen.tiktoken, LICENSE,
//!                              THIRD_PARTY_NOTICES.md, licenses/*)
//! <dir>/YuE2-Vae/…             the standard decoder, when saved
//! <dir>/YuE2-Vae-legacy/…      the legacy decoder, when saved
//! ```
//!
//! [`save_closure`] resolves and verifies every component from its local snapshot, authorizes the
//! copy for [`IntendedUse::NoncommercialExperimentation`] (a local working copy; sharing it with
//! others is [`IntendedUse::Redistribution`], which the CC BY-NC 4.0 closure does not permit and
//! this module never does), copies every pinned file — the weights **and** the licence, notice
//! and model-card files, never relicensed or stripped — and re-hashes each copied file against its
//! pin. The directory is assembled in `<dir>.partial` and renamed into place only when complete.
//!
//! [`load_closure`] maps `pipeline.json`'s sub-directories back to [`SnapshotDirs`]; the engine then
//! verifies every file against its pin **at load time**, exactly as for any other snapshot, so a
//! changed or truncated file in a saved closure is refused.
//!
//! # Deliberate differences from upstream
//!
//! * Upstream rewrites `config.json` through the model class and omits `README.md`; here every file
//!   is copied byte for byte, because each is pinned by SHA-256 (a rewritten config would fail
//!   verification) and the model card is the document that declares the weights' licence.
//! * Upstream copies its installed `modeling_*.py`; the native runtime has no remote code to copy.

use std::path::Path;

use candle_audio::candle_core::{DType, Device};
use candle_audio::gen_core;
use serde_json::{json, Map, Value};

use crate::engine::{component_identity, EngineOptions, Yue2Engine};
use crate::inventory::{Closure, ComponentId, UpstreamRepo, VaeVariant};
use crate::license::{self, IntendedUse};
use crate::protocol::GenerationConfig;
use crate::run::{
    file_digest, is_nonempty_dir, partial_dir, read_json, sync_dir, write_json, RunError,
};
use crate::snapshot::{self, AssetError, SnapshotDirs, VerifiedComponent};

/// `pipeline.json`.
pub const PIPELINE_JSON: &str = "pipeline.json";
/// The closure format recorded in `pipeline.json`.
pub const CLOSURE_FORMAT: &str = "yue2-closure-v1";

/// The sub-directory a repository is saved under: its name without the owner (`YuE2-3B`).
pub fn repo_dir_name(repo: &UpstreamRepo) -> &'static str {
    repo.id.rsplit('/').next().unwrap_or(repo.id)
}

/// A saved closure, as [`load_closure`] reads it.
#[derive(Clone, Debug, PartialEq)]
pub struct SavedClosure {
    /// The snapshot directories of every saved repository.
    pub dirs: SnapshotDirs,
    /// The decoders saved with it.
    pub decoders: Vec<VaeVariant>,
    /// The generation configuration saved with it.
    pub generation: GenerationConfig,
    /// `pipeline.json` as written.
    pub metadata: Value,
}

fn components_for(decoders: &[VaeVariant]) -> Vec<ComponentId> {
    let mut ids = vec![ComponentId::Lm, ComponentId::QwenTiktoken];
    for d in decoders {
        let id = match d {
            VaeVariant::Standard => ComponentId::VaeStandard,
            VaeVariant::Legacy => ComponentId::VaeLegacy,
        };
        if !ids.contains(&id) {
            ids.push(id);
        }
    }
    ids
}

/// Save the generation closure resolved from `dirs` — YuE2-3B, `qwen.tiktoken` and each decoder
/// in `decoders` — with `generation` into `dest` (see the [module docs](self)). `dest` must be
/// absent or empty. Returns the written `pipeline.json`.
pub fn save_closure(
    dirs: &SnapshotDirs,
    decoders: &[VaeVariant],
    generation: &GenerationConfig,
    dest: &Path,
) -> Result<Value, RunError> {
    // A derived tier snapshot staged as YuE2-3B is copied as a tier (verified by its own manifest
    // when it is copied, sc-22995); the pinned original is resolved like every other component.
    let tier = dirs
        .snapshot_dir(&ComponentId::Lm.component().repo)
        .ok()
        .filter(|dir| crate::tier::is_tier_snapshot(dir));
    save_resolved(
        &|id| snapshot::resolve_component(id, dirs),
        tier.as_deref(),
        decoders,
        generation,
        dest,
    )
}

/// Resolves and verifies one component (the production resolver is
/// [`snapshot::resolve_component`]).
type Resolve<'a> = dyn Fn(ComponentId) -> Result<VerifiedComponent, AssetError> + 'a;

/// [`save_closure`] over any resolver of verified components (unit tests pass synthetic ones).
fn save_resolved(
    resolve: &Resolve<'_>,
    tier: Option<&Path>,
    decoders: &[VaeVariant],
    generation: &GenerationConfig,
    dest: &Path,
) -> Result<Value, RunError> {
    if decoders.is_empty() {
        return Err(RunError::Invalid(
            "a generation closure needs at least one decoder".into(),
        ));
    }
    if is_nonempty_dir(dest)? || (dest.exists() && !dest.is_dir()) {
        return Err(RunError::Exists(dest.to_path_buf()));
    }
    let mut attributions: Vec<&str> = Vec::new();
    for &vae in decoders {
        let auth = license::authorize_closure(
            Closure::Generation { vae },
            IntendedUse::NoncommercialExperimentation,
        )
        .map_err(|e| RunError::Engine(gen_core::Error::Unsupported(e.to_string())))?;
        for a in auth.attributions() {
            if !attributions.contains(a) {
                attributions.push(a);
            }
        }
    }
    let work = partial_dir(dest);
    if work.exists() {
        return Err(RunError::Interrupted(work));
    }
    if let Some(parent) = dest.parent().filter(|p| !p.as_os_str().is_empty()) {
        std::fs::create_dir_all(parent).map_err(|source| RunError::Io {
            path: parent.to_path_buf(),
            source,
        })?;
    }
    std::fs::create_dir(&work).map_err(|source| RunError::Io {
        path: work.clone(),
        source,
    })?;
    // The working directory is this call's own: a failure removes it again (the error that
    // caused it is the one returned).
    let metadata = match assemble(resolve, tier, decoders, generation, &attributions, &work) {
        Ok(metadata) => metadata,
        Err(e) => {
            let _ = std::fs::remove_dir_all(&work);
            return Err(e);
        }
    };
    if dest.is_dir() {
        std::fs::remove_dir(dest).map_err(|source| RunError::Io {
            path: dest.to_path_buf(),
            source,
        })?;
    }
    std::fs::rename(&work, dest).map_err(|source| RunError::Io {
        path: dest.to_path_buf(),
        source,
    })?;
    Ok(metadata)
}

/// Copy and verify every closure file into `work` and write `pipeline.json` there.
fn assemble(
    resolve: &Resolve<'_>,
    tier: Option<&Path>,
    decoders: &[VaeVariant],
    generation: &GenerationConfig,
    attributions: &[&str],
    work: &Path,
) -> Result<Value, RunError> {
    let mut sources = Map::new();
    let mut subdirs = Map::new();
    for id in components_for(decoders) {
        if let (ComponentId::Lm, Some(dir)) = (id, tier) {
            let lm = ComponentId::Lm.component();
            let name = repo_dir_name(&lm.repo);
            subdirs.insert(lm.repo.id.to_string(), json!(name));
            let mut identity = component_identity(lm);
            identity["tier"] = copy_tier(dir, &work.join(name))?;
            sources.insert(lm.key.to_string(), identity);
            continue;
        }
        // Verify the snapshot immediately before reading the bytes to copy.
        let verified = resolve(id).map_err(gen_core::Error::from)?;
        let component = verified.component();
        let name = repo_dir_name(&component.repo);
        subdirs.insert(component.repo.id.to_string(), json!(name));
        let root = work.join(name);
        for file in verified.files() {
            let pinned = component
                .file(file.file)
                .expect("a verified file is a pinned file");
            let to = file
                .file
                .split('/')
                .fold(root.clone(), |p, part| p.join(part));
            if let Some(parent) = to.parent() {
                std::fs::create_dir_all(parent).map_err(|source| RunError::Io {
                    path: parent.to_path_buf(),
                    source,
                })?;
            }
            std::fs::copy(&file.path, &to).map_err(|source| RunError::Io {
                path: to.clone(),
                source,
            })?;
            crate::run::sync_file(&to)?;
            // The copy is checked against the pin, over the bytes now on disk.
            let (sha, bytes) = file_digest(&to)?;
            if sha != pinned.sha256 || bytes != pinned.bytes {
                return Err(RunError::Corrupt {
                    path: to,
                    detail: format!(
                        "copied bytes hash to {sha} ({bytes} bytes); pinned {} ({} bytes)",
                        pinned.sha256, pinned.bytes
                    ),
                });
            }
        }
        sync_dir(&root)?;
        sources.insert(component.key.to_string(), component_identity(component));
    }
    let metadata = json!({
        "format": CLOSURE_FORMAT,
        "engine": crate::engine::ENGINE_ID,
        "model": repo_dir_name(&ComponentId::Lm.component().repo),
        "repositories": subdirs,
        "decoders": decoders.iter().map(|&d| crate::vae::variant_name(d)).collect::<Vec<_>>(),
        "generation_config": generation.to_json(),
        "source_weights": sources,
        "license": {
            "intended_use": "noncommercial_experimentation",
            "attributions": attributions,
            "note": "A local working copy of the YuE2 closure. The weights are CC BY-NC 4.0: \
                     noncommercial use only, and sharing this copy with others is \
                     redistribution, which is not authorized. Licence and notice files are \
                     retained unmodified beside the weights.",
        },
    });
    write_json(&work.join(PIPELINE_JSON), &metadata)?;
    sync_dir(work)?;
    Ok(metadata)
}

/// Copy a derived tier snapshot ([`crate::tier`]) — verified immediately before its bytes are read
/// — into `root`, re-hashing every copy against the manifest record it was verified with. Returns
/// the tier's identity for `pipeline.json`.
fn copy_tier(dir: &Path, root: &Path) -> Result<Value, RunError> {
    copy_tier_from(crate::tier::TierSource::pinned(), dir, root)
}

pub(crate) fn copy_tier_from(
    source: crate::tier::TierSource,
    dir: &Path,
    root: &Path,
) -> Result<Value, RunError> {
    let verified = crate::tier::verify_tier_from(source, dir).map_err(gen_core::Error::from)?;
    let files = verified
        .manifest()
        .get("files")
        .and_then(Value::as_object)
        .cloned()
        .unwrap_or_default();
    let manifest_file = crate::tier::TIER_MANIFEST;
    let mut copies: Vec<(String, Option<(String, u64)>)> = files
        .iter()
        .map(|(rel, rec)| {
            let want = rec
                .get("sha256")
                .and_then(Value::as_str)
                .zip(rec.get("bytes").and_then(Value::as_u64))
                .map(|(s, b)| (s.to_string(), b));
            (rel.clone(), want)
        })
        .collect();
    // The manifest itself is re-verified when the copy loads; it is copied byte for byte.
    copies.push((manifest_file.to_string(), None));
    for (rel, want) in copies {
        let from = crate::snapshot::join_rel(dir, &rel);
        let to = crate::snapshot::join_rel(root, &rel);
        if let Some(parent) = to.parent() {
            std::fs::create_dir_all(parent).map_err(|source| RunError::Io {
                path: parent.to_path_buf(),
                source,
            })?;
        }
        std::fs::copy(&from, &to).map_err(|source| RunError::Io {
            path: to.clone(),
            source,
        })?;
        let (sha, bytes) = file_digest(&to)?;
        let expected = match want {
            Some(w) => w,
            None => file_digest(&from)?,
        };
        if (sha.clone(), bytes) != expected {
            return Err(RunError::Corrupt {
                path: to,
                detail: format!(
                    "copied bytes hash to {sha} ({bytes} bytes); verified {} ({} bytes)",
                    expected.0, expected.1
                ),
            });
        }
    }
    sync_dir(root)?;
    Ok(json!({
        "tier": verified.tier().name(),
        "conversion": crate::tier::CONVERSION_ID,
        "weights_sha256": verified.weights_sha256(),
    }))
}

/// Read a closure saved by [`save_closure`]. Only `pipeline.json` is parsed here; every weight and
/// licence file is verified against its pin when the engine loads from the returned directories.
pub fn load_closure(dir: &Path) -> Result<SavedClosure, RunError> {
    let path = dir.join(PIPELINE_JSON);
    let metadata = read_json(&path)?;
    let corrupt = |detail: &str| RunError::Corrupt {
        path: path.clone(),
        detail: detail.to_string(),
    };
    if metadata.get("format").and_then(Value::as_str) != Some(CLOSURE_FORMAT) {
        return Err(corrupt("not a YuE2 closure"));
    }
    let repos = metadata
        .get("repositories")
        .and_then(Value::as_object)
        .ok_or_else(|| corrupt("no repositories"))?;
    let mut dirs = SnapshotDirs::new();
    for (repo, sub) in repos {
        let sub = sub.as_str().ok_or_else(|| corrupt("repository dir"))?;
        let plain = !sub.is_empty()
            && Path::new(sub)
                .components()
                .all(|c| matches!(c, std::path::Component::Normal(_)))
            && Path::new(sub).components().count() == 1;
        if !plain {
            return Err(corrupt("a repository directory must be a plain child name"));
        }
        dirs = dirs.with(repo.clone(), dir.join(sub));
    }
    let decoders = metadata
        .get("decoders")
        .and_then(Value::as_array)
        .ok_or_else(|| corrupt("no decoders"))?
        .iter()
        .map(|d| match d.as_str() {
            Some("standard") => Ok(VaeVariant::Standard),
            Some("legacy") => Ok(VaeVariant::Legacy),
            _ => Err(corrupt("unknown decoder")),
        })
        .collect::<Result<Vec<_>, _>>()?;
    let generation = GenerationConfig::from_json(
        metadata
            .get("generation_config")
            .ok_or_else(|| corrupt("no generation_config"))?,
    )
    .map_err(|e| corrupt(&e.to_string()))?;
    Ok(SavedClosure {
        dirs,
        decoders,
        generation,
        metadata,
    })
}

impl Yue2Engine {
    /// Load an engine from a closure saved by [`save_closure`] (upstream `from_pretrained` on a
    /// saved directory), offline, with the saved generation configuration. Every file is verified
    /// against its pin as it is loaded.
    pub fn load_saved(
        dir: &Path,
        dtype: DType,
        device: &Device,
        options: EngineOptions,
    ) -> gen_core::Result<Self> {
        let saved = load_closure(dir)?;
        Self::load(&saved.dirs, dtype, device, saved.generation, options)
    }
}

/// Whether `dir` holds a saved closure (`pipeline.json`).
pub fn is_saved_closure(dir: &Path) -> bool {
    dir.join(PIPELINE_JSON).is_file()
}

#[cfg(test)]
mod tests;
