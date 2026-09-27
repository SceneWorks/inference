//! Resolve index-selected safetensors files before a backend opens or maps them.
//!
//! The caller supplies authorized physical roots. A snapshot may contain links to its own
//! repository's `blobs/`, so [`snapshot_shard_roots`] recognizes that layout without deriving a
//! cache location or authorizing a whole cache tree.

use std::path::{Component, Path, PathBuf};

use crate::{Error, Result};

/// Authorized physical roots for a caller-provisioned component directory. Ordinary directories
/// authorize only themselves. A component beneath `models--*/snapshots/<revision>/` also
/// authorizes that same repository's `blobs/` directory, if present.
pub fn snapshot_shard_roots(dir: &Path) -> Result<Vec<PathBuf>> {
    let component = std::fs::canonicalize(dir)
        .map_err(|e| Error::Msg(format!("resolve shard directory {}: {e}", dir.display())))?;
    if !component.is_dir() {
        return Err(Error::Msg(format!(
            "shard directory {} is not a directory",
            dir.display()
        )));
    }
    let mut roots = vec![component.clone()];
    for revision in component.ancestors().skip(1) {
        let Some(snapshots) = revision.parent() else {
            continue;
        };
        let Some(repository) = snapshots.parent() else {
            continue;
        };
        if snapshots
            .file_name()
            .is_some_and(|name| name == "snapshots")
            && repository
                .file_name()
                .and_then(|name| name.to_str())
                .is_some_and(|name| name.starts_with("models--"))
        {
            let blobs = repository.join("blobs");
            if std::fs::symlink_metadata(&blobs).is_ok_and(|meta| meta.file_type().is_dir()) {
                roots.push(std::fs::canonicalize(blobs)?);
            }
            break;
        }
    }
    Ok(roots)
}

/// Return all distinct shards selected by `{stem}.safetensors.index.json`, or the single-file
/// fallback. Every index entry must be a relative path without parent traversal. Every selected
/// file must be regular and its canonical target must lie beneath an explicitly authorized root.
/// The complete set is validated before the caller can open or map any shard. Returned paths are
/// canonical, avoiding a second resolution of a shard symlink at the mmap boundary.
pub fn resolve_safetensors_shards(
    dir: &Path,
    stem: &str,
    allowed_roots: &[PathBuf],
) -> Result<Vec<PathBuf>> {
    let index_name = format!("{stem}.safetensors.index.json");
    if let Some(paths) = resolve_indexed_safetensors_shards(dir, &index_name, allowed_roots)? {
        return Ok(paths.into_iter().map(|path| path.canonical_path).collect());
    }
    let roots = canonical_roots(allowed_roots)?;
    Ok(vec![validate_file(
        &dir.join(format!("{stem}.safetensors")),
        &roots,
    )?])
}

/// A selected shard's original loader path and authorized canonical target. Candle can mmap the
/// target directly. MLX requires the `.safetensors` extension on the path passed to its loader,
/// so it uses `loader_path` for repository `blobs/<digest>` symlinks after full-set validation.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ResolvedShardPath {
    pub loader_path: PathBuf,
    pub canonical_path: PathBuf,
}

/// Resolve every distinct shard named by a particular index, including variant indexes such as
/// `diffusion_pytorch_model.safetensors.index.bf16.json`. Returns `None` only when the index does
/// not exist; a present malformed or non-regular index is an error. Callers retain their existing
/// no-index fallback. The entire selected set is checked before any path is returned for loading.
pub fn resolve_indexed_safetensors_shards(
    dir: &Path,
    index_name: &str,
    allowed_roots: &[PathBuf],
) -> Result<Option<Vec<ResolvedShardPath>>> {
    if !matches!(
        Path::new(index_name)
            .components()
            .collect::<Vec<_>>()
            .as_slice(),
        [Component::Normal(_)]
    ) {
        return Err(Error::Msg(format!(
            "invalid shard index name {index_name:?}"
        )));
    }
    let roots = canonical_roots(allowed_roots)?;
    let index = dir.join(index_name);
    let index_target = match std::fs::symlink_metadata(&index) {
        Ok(_) => validate_file(&index, &roots)?,
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => return Ok(None),
        Err(error) => return Err(Error::Msg(format!("inspect {}: {error}", index.display()))),
    };
    let text = std::fs::read_to_string(&index_target)
        .map_err(|e| Error::Msg(format!("read {}: {e}", index.display())))?;
    let json: serde_json::Value = serde_json::from_str(&text)
        .map_err(|e| Error::Msg(format!("parse {}: {e}", index.display())))?;
    let map = json
        .get("weight_map")
        .and_then(|value| value.as_object())
        .ok_or_else(|| Error::Msg(format!("{}: no weight_map", index.display())))?;
    if map.is_empty() {
        return Err(Error::Msg(format!("{}: empty weight_map", index.display())));
    }
    let mut names = Vec::with_capacity(map.len());
    for (tensor, value) in map {
        let name = value.as_str().ok_or_else(|| {
            Error::Msg(format!(
                "{}: shard for {tensor:?} is not a filename",
                index.display()
            ))
        })?;
        names.push(name.to_owned());
    }
    names.sort();
    names.dedup();
    let mut paths = Vec::with_capacity(names.len());
    for name in names {
        let path = Path::new(&name);
        if name.is_empty()
            || path.components().any(|component| {
                matches!(
                    component,
                    Component::ParentDir | Component::RootDir | Component::Prefix(_)
                )
            })
        {
            return Err(Error::Msg(format!(
                "{}: invalid shard path {name:?}; expected a relative path without parent traversal",
                index.display()
            )));
        }
        let loader_path = dir.join(path);
        paths.push(ResolvedShardPath {
            canonical_path: validate_file(&loader_path, &roots)?,
            loader_path,
        });
    }
    Ok(Some(paths))
}

fn canonical_roots(allowed_roots: &[PathBuf]) -> Result<Vec<PathBuf>> {
    if allowed_roots.is_empty() {
        return Err(Error::Msg(
            "safetensors shards have no authorized roots".into(),
        ));
    }
    Ok(allowed_roots
        .iter()
        .map(std::fs::canonicalize)
        .collect::<std::io::Result<Vec<_>>>()?)
}

/// Preserve the existing direct-child `.safetensors` fallback enumeration used by Candle and
/// MLX: skip hidden sidecars, sort lexical paths, and require at least one shard. Validate the
/// whole set before returning paths so no fallback loader can open a FIFO or an
/// unauthorized symlink before discovering a later bad entry.
pub fn resolve_directory_safetensors_shards(
    dir: &Path,
    allowed_roots: &[PathBuf],
) -> Result<Vec<ResolvedShardPath>> {
    let roots = canonical_roots(allowed_roots)?;
    let mut files: Vec<PathBuf> = std::fs::read_dir(dir)
        .map_err(|e| Error::Msg(format!("read shard directory {}: {e}", dir.display())))?
        .filter_map(|entry| entry.ok().map(|entry| entry.path()))
        .filter(|path| {
            path.extension()
                .is_some_and(|extension| extension == "safetensors")
        })
        .filter(|path| !crate::weightsmeta::is_hidden_file(path))
        .collect();
    files.sort();
    if files.is_empty() {
        return Err(Error::Msg(format!(
            "no .safetensors files in {}",
            dir.display()
        )));
    }
    files
        .into_iter()
        .map(|loader_path| {
            Ok(ResolvedShardPath {
                canonical_path: validate_file(&loader_path, &roots)?,
                loader_path,
            })
        })
        .collect()
}

fn validate_file(path: &Path, roots: &[PathBuf]) -> Result<PathBuf> {
    let target = std::fs::canonicalize(path)
        .map_err(|e| Error::Msg(format!("resolve {}: {e}", path.display())))?;
    if !roots.iter().any(|root| target.starts_with(root)) {
        return Err(Error::Msg(format!(
            "{} resolves outside authorized shard roots ({})",
            path.display(),
            target.display()
        )));
    }
    if !std::fs::metadata(&target)
        .map_err(|e| Error::Msg(format!("inspect {}: {e}", path.display())))?
        .is_file()
    {
        return Err(Error::Msg(format!(
            "{} is not a regular file",
            path.display()
        )));
    }
    Ok(target)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn index(dir: &Path, names: &[&str]) {
        let map: serde_json::Map<String, serde_json::Value> = names
            .iter()
            .enumerate()
            .map(|(i, name)| (format!("tensor_{i}"), (*name).into()))
            .collect();
        std::fs::write(
            dir.join("model.safetensors.index.json"),
            serde_json::json!({ "weight_map": map }).to_string(),
        )
        .unwrap();
    }

    #[test]
    fn resolves_only_complete_regular_local_set() {
        let temp = tempfile::tempdir().unwrap();
        let dir = temp.path();
        std::fs::write(dir.join("a.safetensors"), b"a").unwrap();
        std::fs::write(dir.join("b.safetensors"), b"b").unwrap();
        std::fs::create_dir(dir.join("nested")).unwrap();
        std::fs::write(dir.join("nested/c.safetensors"), b"c").unwrap();
        let roots = snapshot_shard_roots(dir).unwrap();
        index(
            dir,
            &[
                "b.safetensors",
                "a.safetensors",
                "a.safetensors",
                "nested/c.safetensors",
            ],
        );
        assert_eq!(
            resolve_safetensors_shards(dir, "model", &roots).unwrap(),
            vec![
                std::fs::canonicalize(dir.join("a.safetensors")).unwrap(),
                std::fs::canonicalize(dir.join("b.safetensors")).unwrap(),
                std::fs::canonicalize(dir.join("nested/c.safetensors")).unwrap()
            ]
        );
        index(dir, &["a.safetensors", "missing.safetensors"]);
        assert!(resolve_safetensors_shards(dir, "model", &roots)
            .unwrap_err()
            .to_string()
            .contains("missing.safetensors"));
    }

    #[test]
    fn named_variant_index_and_directory_fallback_keep_selection_order() {
        let temp = tempfile::tempdir().unwrap();
        let dir = temp.path();
        std::fs::write(dir.join("z.safetensors"), b"z").unwrap();
        std::fs::write(dir.join("a.safetensors"), b"a").unwrap();
        std::fs::write(dir.join("._sidecar.safetensors"), b"sidecar").unwrap();
        std::fs::create_dir(dir.join("nested")).unwrap();
        std::fs::write(dir.join("nested/part.bin"), b"nested").unwrap();
        let roots = snapshot_shard_roots(dir).unwrap();
        let variant = "diffusion_pytorch_model.safetensors.index.bf16.json";
        assert!(resolve_indexed_safetensors_shards(dir, variant, &roots)
            .unwrap()
            .is_none());
        std::fs::write(
            dir.join(variant),
            r#"{"weight_map":{"a":"z.safetensors","b":"nested/part.bin","c":"z.safetensors"}}"#,
        )
        .unwrap();
        assert_eq!(
            resolve_indexed_safetensors_shards(dir, variant, &roots)
                .unwrap()
                .unwrap()
                .into_iter()
                .map(|path| path.canonical_path)
                .collect::<Vec<_>>(),
            vec![
                std::fs::canonicalize(dir.join("nested/part.bin")).unwrap(),
                std::fs::canonicalize(dir.join("z.safetensors")).unwrap()
            ]
        );
        assert_eq!(
            resolve_directory_safetensors_shards(dir, &roots)
                .unwrap()
                .into_iter()
                .map(|path| path.canonical_path)
                .collect::<Vec<_>>(),
            vec![
                std::fs::canonicalize(dir.join("a.safetensors")).unwrap(),
                std::fs::canonicalize(dir.join("z.safetensors")).unwrap()
            ]
        );
    }

    #[cfg(unix)]
    #[test]
    fn directory_fallback_validates_every_target_before_loading() {
        use std::os::unix::fs::symlink;
        let temp = tempfile::tempdir().unwrap();
        let dir = temp.path().join("component");
        std::fs::create_dir(&dir).unwrap();
        std::fs::write(dir.join("a.safetensors"), b"a").unwrap();
        let roots = snapshot_shard_roots(&dir).unwrap();
        let fifo = dir.join("z.safetensors");
        assert!(std::process::Command::new("mkfifo")
            .arg(&fifo)
            .status()
            .unwrap()
            .success());
        assert!(resolve_directory_safetensors_shards(&dir, &roots)
            .unwrap_err()
            .to_string()
            .contains("not a regular file"));
        std::fs::remove_file(&fifo).unwrap();
        let outside = temp.path().join("outside.safetensors");
        std::fs::write(&outside, b"outside").unwrap();
        symlink(&outside, dir.join("z.safetensors")).unwrap();
        assert!(resolve_directory_safetensors_shards(&dir, &roots)
            .unwrap_err()
            .to_string()
            .contains("outside authorized shard roots"));
    }

    #[test]
    fn rejects_invalid_index_values_and_names() {
        let temp = tempfile::tempdir().unwrap();
        let dir = temp.path();
        let roots = snapshot_shard_roots(dir).unwrap();
        for name in ["../outside.safetensors", "/tmp/outside.safetensors", ".."] {
            index(dir, &[name]);
            assert!(
                resolve_safetensors_shards(dir, "model", &roots)
                    .unwrap_err()
                    .to_string()
                    .contains("invalid shard path"),
                "{name:?}"
            );
        }
        std::fs::write(
            dir.join("model.safetensors.index.json"),
            r#"{"weight_map":{"x":42}}"#,
        )
        .unwrap();
        assert!(resolve_safetensors_shards(dir, "model", &roots)
            .unwrap_err()
            .to_string()
            .contains("not a filename"));
    }

    #[cfg(unix)]
    #[test]
    fn rejects_external_links_and_fifo_without_opening_them() {
        use std::os::unix::fs::symlink;
        let temp = tempfile::tempdir().unwrap();
        let dir = temp.path().join("component");
        std::fs::create_dir(&dir).unwrap();
        let outside = temp.path().join("outside.safetensors");
        std::fs::write(&outside, b"outside").unwrap();
        symlink(&outside, dir.join("link.safetensors")).unwrap();
        let roots = snapshot_shard_roots(&dir).unwrap();
        index(&dir, &["link.safetensors"]);
        assert!(resolve_safetensors_shards(&dir, "model", &roots)
            .unwrap_err()
            .to_string()
            .contains("outside authorized shard roots"));

        let fifo = dir.join("pipe.safetensors");
        assert!(std::process::Command::new("mkfifo")
            .arg(&fifo)
            .status()
            .unwrap()
            .success());
        index(&dir, &["pipe.safetensors"]);
        assert!(resolve_safetensors_shards(&dir, "model", &roots)
            .unwrap_err()
            .to_string()
            .contains("not a regular file"));

        let index_path = dir.join("model.safetensors.index.json");
        std::fs::remove_file(&index_path).unwrap();
        assert!(std::process::Command::new("mkfifo")
            .arg(&index_path)
            .status()
            .unwrap()
            .success());
        assert!(resolve_safetensors_shards(&dir, "model", &roots)
            .unwrap_err()
            .to_string()
            .contains("not a regular file"));
    }

    #[cfg(unix)]
    #[test]
    fn only_provisioned_repository_blobs_are_authorized() {
        use std::os::unix::fs::symlink;
        let temp = tempfile::tempdir().unwrap();
        let repository = temp.path().join("models--org--repo");
        let dir = repository.join("snapshots/revision/text_encoder");
        let blobs = repository.join("blobs");
        std::fs::create_dir_all(&dir).unwrap();
        std::fs::create_dir(&blobs).unwrap();
        std::fs::write(blobs.join("digest"), b"data").unwrap();
        symlink("../../../blobs/digest", dir.join("model.safetensors")).unwrap();
        let roots = snapshot_shard_roots(&dir).unwrap();
        assert_eq!(
            roots,
            vec![
                std::fs::canonicalize(&dir).unwrap(),
                std::fs::canonicalize(&blobs).unwrap()
            ]
        );
        assert_eq!(
            resolve_safetensors_shards(&dir, "model", &roots).unwrap(),
            vec![std::fs::canonicalize(blobs.join("digest")).unwrap()]
        );

        let other = repository.join("other.safetensors");
        std::fs::write(&other, b"outside blobs").unwrap();
        symlink("../../../other.safetensors", dir.join("other.safetensors")).unwrap();
        index(&dir, &["other.safetensors"]);
        assert!(resolve_safetensors_shards(&dir, "model", &roots)
            .unwrap_err()
            .to_string()
            .contains("outside authorized shard roots"));

        let linked_repository = temp.path().join("models--org--linked");
        let linked_dir = linked_repository.join("snapshots/revision/text_encoder");
        std::fs::create_dir_all(&linked_dir).unwrap();
        symlink(&blobs, linked_repository.join("blobs")).unwrap();
        symlink(
            "../../../blobs/digest",
            linked_dir.join("model.safetensors"),
        )
        .unwrap();
        let linked_roots = snapshot_shard_roots(&linked_dir).unwrap();
        assert_eq!(
            linked_roots,
            vec![std::fs::canonicalize(&linked_dir).unwrap()]
        );
        assert!(
            resolve_safetensors_shards(&linked_dir, "model", &linked_roots)
                .unwrap_err()
                .to_string()
                .contains("outside authorized shard roots")
        );
    }
}
