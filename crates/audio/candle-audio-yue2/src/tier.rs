//! Derived YuE2 tier snapshots (sc-22995): the deterministic, local `q8` / `q4` conversion of the
//! verified pinned YuE2-3B original, and their verification at the load boundary.
//!
//! # Why local
//!
//! SceneWorks normally ships pre-quantized tier snapshots rehosted on a model hub. YuE2's weights
//! are CC BY-NC 4.0 and [`crate::license`] refuses [`IntendedUse::Redistribution`] until an owner
//! records a distribution basis, so **nothing here uploads or rehosts anything**: a tier snapshot is
//! derived on the user's machine from the original they acquired, with [`convert`] (also reachable
//! as the audio-lane snapshot preparer, [`crate::prepare`]). Rehosting derived tier snapshots stays
//! gated — [`crate::precision::OWNER_DECISIONS`]' `tier_snapshot_rehost`.
//!
//! # Layout
//!
//! ```text
//! <dir>/yue2-tier.json       the conversion manifest (below)
//! <dir>/model.safetensors    the tier weights
//! <dir>/config.json, qwen.tiktoken, LICENSE, README.md, THIRD_PARTY_NOTICES.md, licenses/*, …
//!                            every other pinned file of YuE2-3B, byte for byte
//! ```
//!
//! A tier snapshot is self-contained and loads wherever the YuE2-3B snapshot directory is expected
//! (the `m-a-p/YuE2-3B` entry of [`SnapshotDirs`], or `LoadSpec::weights`). The licence, notice and
//! model-card files travel with it unmodified (epic E7: never relicensed, never stripped).
//!
//! `model.safetensors` holds exactly the tensors of the released checkpoint, each in the storage its
//! tier assigns ([`crate::precision`]): a BF16 tensor byte-identical to the original, or a GGML
//! block-quantized matrix as a `U8` `[rows, blocks_per_row, block_bytes]` GGML block tensor — the
//! workspace's stored-GGML convention (`candle_llm::primitives::quant::to_ggml_block_tensor`), which
//! the loader rebuilds on the device exactly as stored.
//!
//! # The conversion manifest
//!
//! `yue2-tier.json` pins the source identity (repository, revision, the original weights file's
//! SHA-256 and size), the conversion code version ([`CONVERSION_ID`] — bumped whenever the bytes a
//! conversion writes could change), the tier, every tensor's class, storage, logical and stored
//! shape and the SHA-256 of its stored bytes, and every file's size and SHA-256.
//!
//! The conversion is deterministic: Candle's CPU GGML quantizer is per-block scalar code and the
//! safetensors writer orders tensors by (dtype, name), so the same original and conversion code give
//! byte-identical output (asserted by `conversion_is_deterministic_and_verifies`).
//!
//! # Verification ([`verify_tier`])
//!
//! Nothing in the manifest is trusted that can be derived: the precision plan is recomputed from
//! the committed original's tensor table and the tier; every copied file is checked against the
//! **inventory pins** (not the manifest); every BF16 tensor's recorded digest must equal the pinned
//! original's; the weights file's header must be exactly the plan's stored table; no other weights
//! file may be present; and the weights file must hash to the manifest's SHA-256.

use std::collections::{BTreeMap, HashMap};
use std::path::{Path, PathBuf};

use candle_audio::candle_core::quantized::QTensor;
use candle_audio::candle_core::{self, CpuStorage, Device, Storage as CStorage, Tensor};
use candle_audio::gen_core;
use candle_llm::primitives::quant::to_ggml_block_tensor;
use serde_json::{json, Map, Value};
use sha2::{Digest, Sha256};

use crate::inventory::{Component, ComponentId, FileRole, PinnedFile};
use crate::license::{self, IntendedUse};
use crate::precision::{self, PlannedTensor, Storage, Tier};
use crate::run::{
    file_digest, is_nonempty_dir, partial_dir, read_json, sync_dir, write_json, RunError,
};
use crate::snapshot::{self, AssetError, SnapshotDirs};

/// The conversion manifest's file name.
pub const TIER_MANIFEST: &str = "yue2-tier.json";
/// The tier-snapshot format this crate writes and reads.
pub const TIER_FORMAT: &str = "yue2-tier-v1";
/// The conversion code version. Bump it whenever a conversion's output bytes could change; a
/// snapshot written by another version is refused (re-derive it).
pub const CONVERSION_ID: &str = "yue2-ggml-tier-v1";
/// The tier weights file.
pub const WEIGHTS_FILE: &str = "model.safetensors";
/// The component key tier errors are reported under.
pub const COMPONENT_KEY: &str = "yue2_3b_tier";

/// Whether `dir` holds a tier snapshot (a `yue2-tier.json`).
pub fn is_tier_snapshot(dir: &Path) -> bool {
    dir.join(TIER_MANIFEST).is_file()
}

/// The pinned components a tier is derived from: the MoT and its tokenizer file.
#[derive(Clone, Copy, Debug)]
pub(crate) struct TierSource {
    /// The MoT component (YuE2-3B).
    pub(crate) lm: &'static Component,
    /// The tokenizer component (`qwen.tiktoken`, same repository).
    pub(crate) tok: &'static Component,
}

impl TierSource {
    /// The pinned YuE2-3B original.
    pub(crate) fn pinned() -> Self {
        Self {
            lm: ComponentId::Lm.component(),
            tok: ComponentId::QwenTiktoken.component(),
        }
    }

    /// Every file a tier snapshot copies byte for byte: each pinned MoT file but the weights, and
    /// the tokenizer file.
    fn copied_files(self) -> Vec<&'static PinnedFile> {
        self.lm
            .files
            .iter()
            .filter(|f| f.role != FileRole::Weights)
            .chain(self.tok.files.iter())
            .collect()
    }

    /// The original a tier is derived from (repository, revision, weights file identity).
    fn identity(self) -> Value {
        let w = self.lm.weights().expect("the MoT pins a weights file");
        json!({
            "component": self.lm.key,
            "repo": self.lm.repo.id,
            "revision": self.lm.repo.revision,
            "file": w.path,
            "sha256": w.sha256,
            "bytes": w.bytes,
        })
    }
}

/// The precision plan of `tier` over the committed original's tensor table.
pub fn tier_plan(tier: Tier) -> Result<Vec<PlannedTensor>, AssetError> {
    plan_from(TierSource::pinned(), tier)
}

fn plan_from(source: TierSource, tier: Tier) -> Result<Vec<PlannedTensor>, AssetError> {
    let manifest = source.lm.conversion_manifest()?;
    precision::plan(
        manifest
            .tensors
            .iter()
            .map(|t| (t.name.as_str(), t.dtype.as_str(), t.shape.as_slice())),
        tier,
    )
    .map_err(|e| manifest_error(e.to_string()))
}

fn manifest_error(detail: impl Into<String>) -> AssetError {
    AssetError::Manifest {
        component: COMPONENT_KEY.to_string(),
        detail: detail.into(),
    }
}

/// A tier snapshot whose every file was verified ([`verify_tier`]).
#[derive(Clone, Debug)]
pub struct VerifiedTier {
    dir: PathBuf,
    tier: Tier,
    plan: Vec<PlannedTensor>,
    weights: PathBuf,
    weights_sha256: String,
    manifest: Value,
}

impl VerifiedTier {
    /// The snapshot directory.
    pub fn dir(&self) -> &Path {
        &self.dir
    }

    /// The tier it holds.
    pub fn tier(&self) -> Tier {
        self.tier
    }

    /// The precision plan its weights follow.
    pub fn plan(&self) -> &[PlannedTensor] {
        &self.plan
    }

    /// The verified weights file.
    pub fn weights_path(&self) -> &Path {
        &self.weights
    }

    /// The verified weights file's SHA-256.
    pub fn weights_sha256(&self) -> &str {
        &self.weights_sha256
    }

    /// The verified `config.json`.
    pub fn config_path(&self) -> PathBuf {
        self.dir.join("config.json")
    }

    /// The conversion manifest as read.
    pub fn manifest(&self) -> &Value {
        &self.manifest
    }

    /// The plan as a name → storage map (what the loader reads each tensor as).
    pub fn storage_map(&self) -> BTreeMap<String, Storage> {
        self.plan
            .iter()
            .map(|t| (t.name.clone(), t.storage))
            .collect()
    }
}

fn file_check(dir: &Path, file: &str, bytes: u64, sha256: &str) -> Result<PathBuf, AssetError> {
    let path = snapshot::join_rel(dir, file);
    let meta = match std::fs::metadata(&path) {
        Ok(m) if m.is_file() => m,
        Ok(_) => {
            return Err(AssetError::MissingFile {
                component: COMPONENT_KEY.into(),
                file: file.into(),
                path,
            })
        }
        Err(e) if e.kind() == std::io::ErrorKind::NotFound => {
            return Err(AssetError::MissingFile {
                component: COMPONENT_KEY.into(),
                file: file.into(),
                path,
            })
        }
        Err(source) => {
            return Err(AssetError::Io {
                component: COMPONENT_KEY.into(),
                path,
                source,
            })
        }
    };
    if meta.len() != bytes {
        return Err(AssetError::SizeMismatch {
            component: COMPONENT_KEY.into(),
            file: file.into(),
            expected: bytes,
            actual: meta.len(),
        });
    }
    let actual = snapshot::sha256_file(COMPONENT_KEY, &path)?;
    if actual != sha256 {
        return Err(AssetError::HashMismatch {
            component: COMPONENT_KEY.into(),
            file: file.into(),
            expected: sha256.into(),
            actual,
        });
    }
    Ok(path)
}

/// Verify the tier snapshot in `dir` (see the [module docs](self#verification-verify_tier)). Call it
/// immediately before loading and load only the returned paths.
pub fn verify_tier(dir: &Path) -> Result<VerifiedTier, AssetError> {
    verify_tier_from(TierSource::pinned(), dir)
}

pub(crate) fn verify_tier_from(source: TierSource, dir: &Path) -> Result<VerifiedTier, AssetError> {
    let manifest_path = dir.join(TIER_MANIFEST);
    let text = std::fs::read(&manifest_path).map_err(|source| AssetError::Io {
        component: COMPONENT_KEY.into(),
        path: manifest_path.clone(),
        source,
    })?;
    let manifest: Value =
        serde_json::from_slice(&text).map_err(|e| manifest_error(format!("not JSON: {e}")))?;
    let str_at = |v: &Value, key: &str| -> Result<String, AssetError> {
        v.get(key)
            .and_then(Value::as_str)
            .map(str::to_string)
            .ok_or_else(|| manifest_error(format!("missing `{key}`")))
    };
    if str_at(&manifest, "format")? != TIER_FORMAT {
        return Err(manifest_error(format!("not a {TIER_FORMAT} snapshot")));
    }
    let conversion = manifest
        .get("conversion")
        .ok_or_else(|| manifest_error("missing `conversion`"))?;
    let id = str_at(conversion, "id")?;
    if id != CONVERSION_ID {
        return Err(manifest_error(format!(
            "written by conversion {id}; this runtime reads {CONVERSION_ID} — re-derive the tier \
             from the original"
        )));
    }
    let tier_name = str_at(&manifest, "tier")?;
    let tier = match Tier::parse(&tier_name) {
        Some(Tier::Bf16) | None => {
            return Err(manifest_error(format!(
                "`{tier_name}` is not a derived tier (bf16 is the released checkpoint itself)"
            )))
        }
        Some(t) => t,
    };

    // The source must be the pinned original.
    let recorded = manifest
        .get("source")
        .ok_or_else(|| manifest_error("missing `source`"))?;
    let want_source = source.identity();
    if recorded != &want_source {
        return Err(manifest_error(format!(
            "derived from {recorded}, not the pinned original {want_source}"
        )));
    }

    // Every copied file, against the inventory pins.
    for pinned in source.copied_files() {
        file_check(dir, pinned.path, pinned.bytes, pinned.sha256)?;
    }

    // No other weights file anywhere in the tree.
    let mut found = Vec::new();
    snapshot::collect_weight_files(COMPONENT_KEY, dir, "", &mut found)?;
    found.sort();
    if let Some(stray) = found.into_iter().find(|rel| rel != WEIGHTS_FILE) {
        return Err(AssetError::UnexpectedWeightFile {
            component: COMPONENT_KEY.into(),
            file: stray,
            dir: dir.to_path_buf(),
        });
    }

    // The tensor list: the recomputed plan, with the pinned digests for every BF16 tensor.
    let plan = plan_from(source, tier)?;
    let pinned_digests: BTreeMap<String, String> = source
        .lm
        .conversion_manifest()?
        .tensors
        .into_iter()
        .map(|t| (t.name, t.sha256))
        .collect();
    let listed = manifest
        .get("tensors")
        .and_then(Value::as_array)
        .ok_or_else(|| manifest_error("missing `tensors`"))?;
    if listed.len() != plan.len() {
        return Err(manifest_error(format!(
            "lists {} tensors, the {tier} plan has {}",
            listed.len(),
            plan.len()
        )));
    }
    for (entry, want) in listed.iter().zip(&plan) {
        let (dtype, shape) = want.storage.stored(&want.logical_shape);
        let expected = json!({
            "name": want.name,
            "class": want.class.name(),
            "storage": want.storage.label(),
            "logical_shape": want.logical_shape,
            "dtype": dtype,
            "shape": shape,
        });
        let mut got = entry.clone();
        let sha = got
            .as_object_mut()
            .and_then(|o| o.remove("sha256"))
            .and_then(|v| v.as_str().map(str::to_string))
            .ok_or_else(|| manifest_error(format!("`{}` has no sha256", want.name)))?;
        if got != expected {
            return Err(manifest_error(format!(
                "tensor entry {got} differs from the {tier} plan {expected}"
            )));
        }
        if want.storage == Storage::Bf16 && pinned_digests.get(&want.name) != Some(&sha) {
            return Err(manifest_error(format!(
                "`{}` is recorded BF16 but its digest {sha} is not the pinned original's",
                want.name
            )));
        }
    }

    // The weights file: recorded digest, then the header against the plan.
    let files = manifest
        .get("files")
        .and_then(Value::as_object)
        .ok_or_else(|| manifest_error("missing `files`"))?;
    let weights_rec = files
        .get(WEIGHTS_FILE)
        .ok_or_else(|| manifest_error(format!("`files` has no {WEIGHTS_FILE}")))?;
    let weights_sha256 = str_at(weights_rec, "sha256")?;
    let weights_bytes = weights_rec
        .get("bytes")
        .and_then(Value::as_u64)
        .ok_or_else(|| manifest_error(format!("{WEIGHTS_FILE} has no size")))?;
    let weights = file_check(dir, WEIGHTS_FILE, weights_bytes, &weights_sha256)?;
    let table = snapshot::read_tensor_table(COMPONENT_KEY, WEIGHTS_FILE, &weights)?;
    let expected: BTreeMap<String, (String, Vec<usize>)> = plan
        .iter()
        .map(|t| {
            let (d, s) = t.storage.stored(&t.logical_shape);
            (t.name.clone(), (d.to_string(), s))
        })
        .collect();
    if table != expected {
        let mut problems: Vec<String> = expected
            .iter()
            .filter(|(k, v)| table.get(*k) != Some(v))
            .map(|(k, v)| format!("`{k}` should be {v:?}, is {:?}", table.get(k)))
            .chain(
                table
                    .keys()
                    .filter(|k| !expected.contains_key(*k))
                    .map(|k| format!("`{k}` is not in the plan")),
            )
            .collect();
        let n = problems.len();
        problems.truncate(8);
        return Err(AssetError::TensorTableMismatch {
            component: COMPONENT_KEY.into(),
            detail: format!("{n} difference(s): {}", problems.join("; ")),
        });
    }
    Ok(VerifiedTier {
        dir: dir.to_path_buf(),
        tier,
        plan,
        weights,
        weights_sha256,
        manifest,
    })
}

/// The pinned original a tier is derived from (repository, revision, weights file identity).
pub fn source_identity() -> Value {
    TierSource::pinned().identity()
}

/// SHA-256 of a CPU tensor's little-endian element bytes (the manifests' tensor digest).
fn tensor_digest(t: &Tensor) -> candle_core::Result<String> {
    let t = t.to_device(&Device::Cpu)?.contiguous()?;
    let (storage, layout) = t.storage_and_layout();
    let (start, n) = (layout.start_offset(), layout.shape().elem_count());
    let mut hasher = Sha256::new();
    match &*storage {
        CStorage::Cpu(CpuStorage::BF16(v)) => {
            snapshot::hash_le(&mut hasher, &v[start..start + n], |x| {
                x.to_bits().to_le_bytes()
            })
        }
        CStorage::Cpu(CpuStorage::U8(v)) => hasher.update(&v[start..start + n]),
        _ => candle_core::bail!("tier digest: unexpected storage {:?}", t.dtype()),
    }
    Ok(crate::engine::hex(&hasher.finalize()))
}

fn candle_run(what: &str) -> impl Fn(candle_core::Error) -> RunError + '_ {
    move |e| RunError::Engine(gen_core::Error::Msg(format!("YuE2 tier {what}: {e}")))
}

/// Derive the `tier` snapshot from the pinned YuE2-3B original in `dirs` into `dest` (absent or
/// empty), deterministically (see the [module docs](self)). The original and `qwen.tiktoken` are
/// verified immediately before they are read; the result is assembled in `<dest>.partial`,
/// verified with [`verify_tier`], and only then renamed into place. Returns the manifest.
///
/// The copy is a local working copy for [`IntendedUse::NoncommercialExperimentation`]; sharing it
/// is redistribution, which is not authorized ([`crate::license`]).
pub fn convert(dirs: &SnapshotDirs, tier: Tier, dest: &Path) -> Result<Value, RunError> {
    convert_from(TierSource::pinned(), dirs, tier, dest)
}

pub(crate) fn convert_from(
    source: TierSource,
    dirs: &SnapshotDirs,
    tier: Tier,
    dest: &Path,
) -> Result<Value, RunError> {
    if tier == Tier::Bf16 {
        return Err(RunError::Invalid(
            "the bf16 tier is the released checkpoint itself; there is nothing to derive".into(),
        ));
    }
    license::authorize(
        &[ComponentId::Lm, ComponentId::QwenTiktoken],
        IntendedUse::NoncommercialExperimentation,
    )
    .map_err(|e| RunError::Engine(gen_core::Error::Unsupported(e.to_string())))?;
    if is_nonempty_dir(dest)? || (dest.exists() && !dest.is_dir()) {
        return Err(RunError::Exists(dest.to_path_buf()));
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
    let manifest = match assemble(source, dirs, tier, &work) {
        Ok(m) => m,
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
    Ok(manifest)
}

fn assemble(
    source: TierSource,
    dirs: &SnapshotDirs,
    tier: Tier,
    work: &Path,
) -> Result<Value, RunError> {
    let asset = |e: AssetError| RunError::Engine(gen_core::Error::from(e));
    // Verified immediately before the bytes are read.
    let resolve = |c: &'static Component| {
        dirs.snapshot_dir(&c.repo)
            .and_then(|dir| snapshot::verify_component(c, &dir))
    };
    let lm = resolve(source.lm).map_err(asset)?;
    let tok = resolve(source.tok).map_err(asset)?;
    let mut files = Map::new();
    for verified in [&lm, &tok] {
        for file in verified.files() {
            let pinned = verified
                .component()
                .file(file.file)
                .expect("a verified file is pinned");
            if pinned.role == FileRole::Weights {
                continue;
            }
            let to = snapshot::join_rel(work, file.file);
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
            let (sha, bytes) = file_digest(&to)?;
            if sha != pinned.sha256 || bytes != pinned.bytes {
                return Err(RunError::Corrupt {
                    path: to,
                    detail: format!("copied bytes hash to {sha}; pinned {}", pinned.sha256),
                });
            }
            files.insert(
                file.file.to_string(),
                json!({"sha256": sha, "bytes": bytes}),
            );
        }
    }

    let plan = plan_from(source, tier).map_err(asset)?;
    let pinned: BTreeMap<&str, &str> = lm
        .manifest()
        .tensors
        .iter()
        .map(|t| (t.name.as_str(), t.sha256.as_str()))
        .collect();
    let path = lm.weights_path().expect("YuE2-3B has a weights file");
    // SAFETY: the file was just verified and is only read; same contract as every Candle load.
    let st = unsafe { candle_core::safetensors::MmapedSafetensors::new(path) }
        .map_err(candle_run("open original"))?;
    let mut out: HashMap<String, Tensor> = HashMap::with_capacity(plan.len());
    let mut entries = Vec::with_capacity(plan.len());
    for t in &plan {
        let err = candle_run(&t.name);
        let src = st.load(&t.name, &Device::Cpu).map_err(&err)?;
        let stored = match t.storage {
            Storage::Bf16 => src,
            Storage::Ggml(dtype) => {
                let q = QTensor::quantize(&src, dtype).map_err(&err)?;
                to_ggml_block_tensor(&q).map_err(|e| {
                    RunError::Engine(gen_core::Error::Msg(format!(
                        "YuE2 tier {}: {e}",
                        t.name
                    )))
                })?
            }
        };
        let sha = tensor_digest(&stored).map_err(&err)?;
        if t.storage == Storage::Bf16 && pinned.get(t.name.as_str()) != Some(&sha.as_str()) {
            return Err(RunError::Corrupt {
                path: path.to_path_buf(),
                detail: format!("`{}` does not reproduce its pinned digest", t.name),
            });
        }
        let (dtype, shape) = t.storage.stored(&t.logical_shape);
        entries.push(json!({
            "name": t.name,
            "class": t.class.name(),
            "storage": t.storage.label(),
            "logical_shape": t.logical_shape,
            "dtype": dtype,
            "shape": shape,
            "sha256": sha,
        }));
        out.insert(t.name.clone(), stored);
    }
    drop(st);
    let weights = work.join(WEIGHTS_FILE);
    candle_core::safetensors::save(&out, &weights).map_err(candle_run("write"))?;
    drop(out);
    std::fs::File::open(&weights)
        .and_then(|f| f.sync_all())
        .map_err(|source| RunError::Io {
            path: weights.clone(),
            source,
        })?;
    let (sha, bytes) = file_digest(&weights)?;
    files.insert(
        WEIGHTS_FILE.to_string(),
        json!({"sha256": sha, "bytes": bytes}),
    );
    let manifest = json!({
        "format": TIER_FORMAT,
        "tier": tier.name(),
        "conversion": {
            "id": CONVERSION_ID,
            "crate": env!("CARGO_PKG_NAME"),
            "crate_version": env!("CARGO_PKG_VERSION"),
            "method": "candle CPU GGML quantizer (QTensor::quantize) over the BF16 original upcast \
                       to F32; matmul weights of the tier stored as GGML block tensors, every \
                       other tensor byte-identical BF16",
        },
        "source": source.identity(),
        "files": files,
        "tensors": entries,
        "license": {
            "derived_from": "m-a-p/YuE2-3B (CC BY-NC 4.0)",
            "intended_use": "noncommercial_experimentation",
            "note": "A locally derived working copy of CC BY-NC 4.0 weights. Sharing or \
                     rehosting it is redistribution, which is not authorized until an owner \
                     records a distribution basis. Licence and notice files are retained \
                     unmodified beside the weights.",
        },
    });
    write_json(&work.join(TIER_MANIFEST), &manifest)?;
    sync_dir(work)?;
    verify_tier_from(source, work).map_err(asset)?;
    Ok(manifest)
}

/// Read a tier snapshot's manifest without verifying the snapshot (for reporting).
pub fn read_manifest(dir: &Path) -> Result<Value, RunError> {
    read_json(&dir.join(TIER_MANIFEST))
}

#[cfg(test)]
mod tests;
