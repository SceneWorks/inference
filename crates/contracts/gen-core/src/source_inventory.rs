//! Retained source inventory for deferred consumers. Uses the existing replacement-sensitive
//! file tokens, not a header hash masquerading as immutable tensor bytes. Persistent replacement,
//! in-place stat changes and directory membership changes fail before/after each guarded read.
//! Like PinnedWeightsFile, this is not a filesystem transaction against restored-path attacks.
use crate::{Error, LoadSpec, PinnedWeightsFile, Result};
use std::path::{Path, PathBuf};

#[derive(Clone, Debug)]
pub struct SourceInventory {
    roots: Vec<PathBuf>,
    entries: Vec<PathBuf>,
    pins: Vec<PinnedWeightsFile>,
}

fn collect(path: &Path, entries: &mut Vec<PathBuf>, files: &mut Vec<PathBuf>) -> Result<()> {
    entries.push(path.to_owned());
    if path.is_file() {
        files.push(path.to_owned());
        return Ok(());
    }
    for entry in std::fs::read_dir(path)? {
        let entry = entry?;
        let path = entry.path();
        if crate::weightsmeta::is_hidden_file(&path) || crate::weightsmeta::is_hidden_dir(&path) {
            continue;
        }
        if path.is_dir() && entry.file_type()?.is_symlink() {
            return Err(Error::Unsupported(
                "source inventory refuses a symlinked subdirectory".into(),
            ));
        }
        collect(&path, entries, files)?;
    }
    Ok(())
}

impl SourceInventory {
    pub fn capture(spec: &LoadSpec) -> Result<Self> {
        spec.validate_prepared_file_pins()?;
        let mut roots = spec
            .file_source_paths()
            .into_iter()
            .chain(spec.directory_source_paths())
            .map(std::path::absolute)
            .collect::<std::io::Result<Vec<_>>>()?;
        roots.sort();
        roots.dedup();
        let (mut entries, mut files) = (Vec::new(), Vec::new());
        for root in &roots {
            collect(root, &mut entries, &mut files)?;
        }
        entries.sort();
        entries.dedup();
        files.sort();
        files.dedup();
        // Preserve already prepared tokens. New members acquire the ordinary content seal once;
        // guarded deferred reads only verify replacement-sensitive identities, never rehash weights.
        let pins = files
            .iter()
            .map(|path| match spec.prepared_file_pins().get(path) {
                Some(pin) => Ok(pin.clone()),
                None => PinnedWeightsFile::pin(path),
            })
            .collect::<Result<Vec<_>>>()?;
        let guard = Self {
            roots,
            entries,
            pins,
        };
        guard.verify_unchanged()?;
        Ok(guard)
    }

    pub(crate) fn configured_file_pins(&self, spec: &LoadSpec) -> Result<Vec<PinnedWeightsFile>> {
        let mut paths = spec
            .file_source_paths()
            .into_iter()
            .map(std::path::absolute)
            .collect::<std::io::Result<Vec<_>>>()?;
        paths.sort();
        paths.dedup();
        paths
            .iter()
            .map(|path| {
                self.pins
                    .iter()
                    .find(|pin| pin.loader_path() == path)
                    .cloned()
                    .ok_or_else(|| {
                        Error::Unsupported(
                            "configured File is absent from retained inventory".into(),
                        )
                    })
            })
            .collect()
    }

    pub fn verify_for(&self, spec: &LoadSpec) -> Result<()> {
        let mut roots = spec
            .file_source_paths()
            .into_iter()
            .chain(spec.directory_source_paths())
            .map(std::path::absolute)
            .collect::<std::io::Result<Vec<_>>>()?;
        roots.sort();
        roots.dedup();
        if roots != self.roots {
            return Err(Error::Unsupported(
                "source inventory does not cover the configured sources".into(),
            ));
        }
        self.verify_unchanged()
    }

    pub fn verify_unchanged(&self) -> Result<()> {
        let (mut entries, mut files) = (Vec::new(), Vec::new());
        for root in &self.roots {
            collect(root, &mut entries, &mut files)?;
        }
        entries.sort();
        entries.dedup();
        if entries != self.entries {
            return Err(Error::Unsupported(
                "source inventory changed after admission".into(),
            ));
        }
        for pin in &self.pins {
            pin.verify_unchanged()?;
        }
        Ok(())
    }

    pub fn read_unchanged<T, E: From<Error>>(
        &self,
        read: impl FnOnce() -> std::result::Result<T, E>,
    ) -> std::result::Result<T, E> {
        self.verify_unchanged().map_err(E::from)?;
        let result = read();
        self.verify_unchanged().map_err(E::from)?;
        result
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::WeightsSource;
    use std::cell::Cell;
    #[test]
    fn retained_inventory_refuses_files_directory_additions_and_aux_replacement_at_read() {
        for form in ["file", "directory", "override"] {
            let root = tempfile::tempdir().unwrap();
            let file = root.path().join("source.safetensors");
            std::fs::write(&file, b"original").unwrap();
            let aux = tempfile::tempdir().unwrap();
            let component = aux.path().join("vae.safetensors");
            std::fs::write(&component, b"original").unwrap();
            let mut spec = LoadSpec::new(if form == "file" {
                WeightsSource::File(file.clone())
            } else {
                WeightsSource::Dir(root.path().to_owned())
            });
            if form == "override" {
                spec.components.insert(
                    "vae_fp16_fix".into(),
                    WeightsSource::File(component.clone()),
                );
            }
            let guard = SourceInventory::capture(&spec).unwrap();
            spec.bind_source_inventory(guard).unwrap();
            if form == "directory" {
                std::fs::write(root.path().join("new-shard.safetensors"), b"new").unwrap();
            } else {
                let path = if form == "override" {
                    &component
                } else {
                    &file
                };
                let replacement = path.with_extension("tmp");
                std::fs::write(&replacement, b"original").unwrap();
                std::fs::rename(replacement, path).unwrap();
            }
            let invoked = Cell::new(false);
            let read: Result<()> = spec.read_prepared_files_unchanged(|| {
                invoked.set(true);
                Ok(())
            });
            assert!(
                read.is_err(),
                "changed {form} must fail before deferred consumption"
            );
            assert!(!invoked.get());
        }
    }
}
