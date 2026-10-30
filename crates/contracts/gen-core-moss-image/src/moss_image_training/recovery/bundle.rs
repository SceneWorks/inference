//! Complete directory bundles: raw factors, optimizer, pending gradients and run metadata become
//! visible together. Backend writers operate only inside a newly created run-owned staging dir.
use gen_core::{Error, Result};
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use std::{
    collections::BTreeMap,
    fs,
    io::Read,
    path::{Path, PathBuf},
    sync::atomic::{AtomicU64, Ordering},
};
static NONCE: AtomicU64 = AtomicU64::new(0);
const MANIFEST: &str = "complete.json";
const FILES: [&str; 3] = ["run.json", "tensors.safetensors", "optimizer.safetensors"];
#[derive(Serialize, Deserialize)]
struct Manifest {
    format: String,
    files: BTreeMap<String, (u64, String)>,
}
fn error(e: impl std::fmt::Display) -> Error {
    Error::Msg(format!("MOSS resume bundle: {e}"))
}
fn digest(path: &Path) -> Result<(u64, String)> {
    let mut file = fs::File::open(path).map_err(error)?;
    let mut hash = Sha256::new();
    let mut total = 0u64;
    let mut buffer = [0u8; 65536];
    loop {
        let n = file.read(&mut buffer).map_err(error)?;
        if n == 0 {
            break;
        }
        hash.update(&buffer[..n]);
        total = total
            .checked_add(n as u64)
            .ok_or_else(|| error("file length overflow"))?;
    }
    Ok((total, format!("{:x}", hash.finalize())))
}
fn name(stem: &str, step: u32) -> Result<String> {
    if stem.is_empty() || Path::new(stem).file_name().and_then(|v| v.to_str()) != Some(stem) {
        return Err(error("invalid output stem"));
    }
    Ok(format!("{stem}-step{step:06}.moss-state"))
}
pub fn path(dir: &Path, stem: &str, step: u32) -> Result<PathBuf> {
    Ok(dir.join(name(stem, step)?))
}
struct Staging(PathBuf);
impl Drop for Staging {
    fn drop(&mut self) {
        let _ = fs::remove_dir_all(&self.0);
    }
}

pub fn publish(
    dir: &Path,
    stem: &str,
    state: &super::RunState,
    write: impl FnOnce(&Path) -> Result<()>,
) -> Result<PathBuf> {
    fs::create_dir_all(dir).map_err(error)?;
    let final_path = dir.join(name(stem, state.step)?);
    if final_path.exists() {
        return Err(error("checkpoint bundle already exists"));
    }
    let staging = loop {
        let path = dir.join(format!(
            ".moss-state-{}-{}",
            std::process::id(),
            NONCE.fetch_add(1, Ordering::Relaxed)
        ));
        match fs::create_dir(&path) {
            Ok(()) => break Staging(path),
            Err(e) if e.kind() == std::io::ErrorKind::AlreadyExists => continue,
            Err(e) => return Err(error(e)),
        }
    };
    fs::write(
        staging.0.join("run.json"),
        serde_json::to_vec(state).map_err(error)?,
    )
    .map_err(error)?;
    write(&staging.0)?;
    let actual = fs::read_dir(&staging.0)
        .map_err(error)?
        .map(|e| e.map(|e| e.file_name().to_string_lossy().into_owned()))
        .collect::<std::io::Result<std::collections::BTreeSet<_>>>()
        .map_err(error)?;
    let expected = FILES.iter().map(|s| (*s).to_owned()).collect();
    if actual != expected {
        return Err(error(
            "writer did not produce exactly the complete tensor/optimizer/state population",
        ));
    }
    let mut files = BTreeMap::new();
    for name in FILES {
        let path = staging.0.join(name);
        if !fs::symlink_metadata(&path)
            .map_err(error)?
            .file_type()
            .is_file()
        {
            return Err(error("bundle files must be regular owned files"));
        }
        fs::OpenOptions::new()
            .read(true)
            .write(true)
            .open(&path)
            .map_err(error)?
            .sync_all()
            .map_err(error)?;
        files.insert(name.into(), digest(&path)?);
    }
    fs::write(
        staging.0.join(MANIFEST),
        serde_json::to_vec(&Manifest {
            format: super::STATE_VERSION.into(),
            files,
        })
        .map_err(error)?,
    )
    .map_err(error)?;
    fs::OpenOptions::new()
        .read(true)
        .write(true)
        .open(staging.0.join(MANIFEST))
        .map_err(error)?
        .sync_all()
        .map_err(error)?;
    #[cfg(unix)]
    fs::File::open(&staging.0)
        .map_err(error)?
        .sync_all()
        .map_err(error)?;
    fs::rename(&staging.0, &final_path).map_err(error)?;
    #[cfg(unix)]
    fs::File::open(dir)
        .map_err(error)?
        .sync_all()
        .map_err(error)?;
    Ok(final_path)
}

/// Verify the complete manifest and every file before parsing state or allocating tensor payloads.
pub fn read(path: &Path) -> Result<super::RunState> {
    if !fs::symlink_metadata(path)
        .map_err(error)?
        .file_type()
        .is_dir()
    {
        return Err(error("bundle must be an owned directory"));
    }
    let manifest: Manifest =
        serde_json::from_slice(&fs::read(path.join(MANIFEST)).map_err(error)?).map_err(error)?;
    if manifest.format != super::STATE_VERSION
        || manifest
            .files
            .keys()
            .map(String::as_str)
            .collect::<Vec<_>>()
            != FILES
                .iter()
                .copied()
                .collect::<std::collections::BTreeSet<_>>()
                .into_iter()
                .collect::<Vec<_>>()
    {
        return Err(error("unsupported or incomplete manifest"));
    }
    for (name, expected) in manifest.files {
        let file = path.join(name);
        if !fs::symlink_metadata(&file)
            .map_err(error)?
            .file_type()
            .is_file()
            || digest(&file)? != expected
        {
            return Err(error("missing/changed checkpoint bundle file"));
        }
    }
    serde_json::from_slice(&fs::read(path.join("run.json")).map_err(error)?).map_err(error)
}

fn candidates(dir: &Path, stem: &str) -> Result<Vec<(u32, PathBuf)>> {
    name(stem, 0)?;
    let prefix = format!("{stem}-step");
    let suffix = ".moss-state";
    let entries = match fs::read_dir(dir) {
        Ok(entries) => entries,
        Err(e) if e.kind() == std::io::ErrorKind::NotFound => return Ok(Vec::new()),
        Err(e) => return Err(error(e)),
    };
    let mut result = Vec::new();
    for entry in entries {
        let entry = entry.map_err(error)?;
        let file = entry.file_name();
        let text = file.to_string_lossy();
        let Some(digits) = text
            .strip_prefix(&prefix)
            .and_then(|s| s.strip_suffix(suffix))
        else {
            continue;
        };
        let step = digits
            .parse::<u32>()
            .map_err(|_| error("malformed owned checkpoint step name"))?;
        if !digits.bytes().all(|b| b.is_ascii_digit()) || name(stem, step)? != text {
            return Err(error("noncanonical owned checkpoint step name"));
        }
        result.push((step, entry.path()));
    }
    result.sort_by_key(|(step, _)| *step);
    Ok(result)
}
/// Only genuine absence returns None. Corrupt highest candidates are never skipped/fallen back.
pub fn latest_optional(dir: &Path, stem: &str) -> Result<Option<PathBuf>> {
    Ok(candidates(dir, stem)?.pop().map(|(_, path)| path))
}
/// Requested resume is fail-closed: absent state is not permission to start a new run.
pub fn latest(dir: &Path, stem: &str) -> Result<PathBuf> {
    latest_optional(dir, stem)?
        .ok_or_else(|| error("resume requested but no complete training state exists"))
}
/// Explicit restart only, after the caller has stopped the run and validated ownership of dir.
/// Refuse symlinks/unexpected contents; remove only the four native files, never recursive caches.
pub fn clear(dir: &Path, stem: &str) -> Result<usize> {
    let candidates = candidates(dir, stem)?;
    let mut count = 0;
    for (_, path) in candidates {
        if !fs::symlink_metadata(&path)
            .map_err(error)?
            .file_type()
            .is_dir()
        {
            return Err(error(
                "restart checkpoint candidate is not an owned directory",
            ));
        }
        let mut files = Vec::new();
        for entry in fs::read_dir(&path).map_err(error)? {
            let entry = entry.map_err(error)?;
            let name = entry.file_name();
            let name = name
                .to_str()
                .ok_or_else(|| error("unexpected checkpoint filename"))?;
            if !FILES.contains(&name) && name != MANIFEST {
                return Err(error(
                    "restart refuses unexpected files inside checkpoint bundle",
                ));
            }
            if !fs::symlink_metadata(entry.path())
                .map_err(error)?
                .file_type()
                .is_file()
            {
                return Err(error("restart refuses checkpoint symlinks/non-files"));
            }
            files.push(entry.path());
        }
        for file in files {
            fs::remove_file(file).map_err(error)?;
        }
        fs::remove_dir(&path).map_err(error)?;
        count += 1;
    }
    Ok(count)
}

#[cfg(test)]
mod tests {
    use super::*;
    fn state() -> crate::moss_image_training::recovery::RunState {
        super::super::RunState {
            version: super::super::STATE_VERSION.into(),
            stream_version: super::super::STREAM_VERSION.into(),
            effective_identity: super::super::RunIdentity {
                digest: "identity".into(),
                source_checkpoint: "fixture-only".into(),
                dataset_fingerprint: "dataset".into(),
                effective_config: "controls".into(),
            },
            assignment_policy: "fixture-policy".into(),
            crop_policy: "fixture-crop".into(),
            geometry: Vec::new(),
            plan_identity: "plan".into(),
            resolved_seed: 173,
            step: 1,
            update: 0,
            epoch: 0,
            batch: 1,
            sample_cursor: 2,
            accumulated_tokens: 9,
            final_loss: 0.25,
        }
    }
    #[test]
    fn incomplete_writes_are_invisible_and_complete_bundle_rejects_changed_files() {
        let dir = tempfile::tempdir().unwrap();
        let fail = publish(dir.path(), "adapter", &state(), |path| {
            fs::write(path.join("tensors.safetensors"), b"partial").unwrap();
            Err(error("injected interruption"))
        });
        assert!(fail.is_err());
        assert!(latest(dir.path(), "adapter").is_err());
        assert_eq!(fs::read_dir(dir.path()).unwrap().count(), 0);
        let path = publish(dir.path(), "adapter", &state(), |path| {
            for file in &FILES[1..] {
                fs::write(path.join(file), b"owned fixture").map_err(error)?;
            }
            Ok(())
        })
        .unwrap();
        assert_eq!(read(&path).unwrap(), state());
        assert_eq!(latest(dir.path(), "adapter").unwrap(), path);
        fs::write(path.join("optimizer.safetensors"), b"changed").unwrap();
        assert!(read(&path).is_err());
    }
    #[test]
    fn optional_discovery_distinguishes_missing_from_corrupt_latest_and_restart_is_scoped() {
        let root = tempfile::tempdir().unwrap();
        assert!(latest_optional(root.path(), "adapter").unwrap().is_none());
        let first = publish(root.path(), "adapter", &state(), |path| {
            for file in &FILES[1..] {
                fs::write(path.join(file), b"fixture").map_err(error)?;
            }
            Ok(())
        })
        .unwrap();
        let mut newer = state();
        newer.step = 2;
        let second = publish(root.path(), "adapter", &newer, |path| {
            for file in &FILES[1..] {
                fs::write(path.join(file), b"fixture").map_err(error)?;
            }
            Ok(())
        })
        .unwrap();
        fs::write(second.join("run.json"), b"corrupt").unwrap();
        assert_eq!(
            latest_optional(root.path(), "adapter").unwrap(),
            Some(second.clone())
        );
        assert!(read(&second).is_err());
        assert!(read(&first).is_ok());
        let foreign = root.path().join("foreign.safetensors");
        fs::write(&foreign, b"not this run").unwrap();
        assert_eq!(clear(root.path(), "adapter").unwrap(), 2);
        assert!(foreign.is_file());
        assert!(latest_optional(root.path(), "adapter").unwrap().is_none());
        let path = publish(root.path(), "adapter", &state(), |path| {
            for file in &FILES[1..] {
                fs::write(path.join(file), b"fixture").map_err(error)?;
            }
            Ok(())
        })
        .unwrap();
        fs::write(path.join("user-data"), b"foreign").unwrap();
        assert!(clear(root.path(), "adapter").is_err());
        assert!(path.join("user-data").is_file());
    }
}
