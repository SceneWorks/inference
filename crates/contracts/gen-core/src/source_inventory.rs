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

/// Read-only requested-layout facts of the ALREADY retained source inventory. No paths or pins
/// are cloned and no directory/stat/hash operation is performed by allocation_facts.
/// Vec capacities are element counts. Path lengths are encoded bytes; native path capacities are
/// platform capacity units (one byte on the Unix targets consuming the iOS host-layout law).
/// Referenced pin-path lengths describe future clone inputs, not a second owned path allocation.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct SourceInventoryAllocationFacts {
    pub roots_len: u64,
    pub roots_capacity: u64,
    pub entries_len: u64,
    pub entries_capacity: u64,
    pub pins_len: u64,
    pub pins_capacity: u64,
    pub roots_path_bytes: u64,
    pub roots_path_capacity_units: u64,
    pub entries_path_bytes: u64,
    pub entries_path_capacity_units: u64,
    pub referenced_pin_path_bytes: u64,
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
    /// Scalar facts only, preserving the inventory's replacement-sensitive ownership/identity.
    /// This is requested-layout metadata, NOT a malloc/RSS/process-footprint certificate.
    pub fn allocation_facts(&self) -> Result<SourceInventoryAllocationFacts> {
        fn n(value: usize) -> Result<u64> {
            u64::try_from(value)
                .map_err(|_| Error::Msg("source inventory layout extent overflow".into()))
        }
        fn add(a: u64, b: u64) -> Result<u64> {
            a.checked_add(b)
                .ok_or_else(|| Error::Msg("source inventory path layout sum overflow".into()))
        }
        fn paths(paths: &[PathBuf]) -> Result<(u64, u64)> {
            paths.iter().try_fold((0, 0), |(bytes, capacity), path| {
                Ok((
                    add(bytes, n(path.as_os_str().as_encoded_bytes().len())?)?,
                    add(capacity, n(path.capacity())?)?,
                ))
            })
        }
        let (roots_path_bytes, roots_path_capacity_units) = paths(&self.roots)?;
        let (entries_path_bytes, entries_path_capacity_units) = paths(&self.entries)?;
        let referenced_pin_path_bytes = self.pins.iter().try_fold(0, |bytes, pin| {
            add(
                bytes,
                n(pin.loader_path().as_os_str().as_encoded_bytes().len())?,
            )
        })?;
        Ok(SourceInventoryAllocationFacts {
            roots_len: n(self.roots.len())?,
            roots_capacity: n(self.roots.capacity())?,
            entries_len: n(self.entries.len())?,
            entries_capacity: n(self.entries.capacity())?,
            pins_len: n(self.pins.len())?,
            pins_capacity: n(self.pins.capacity())?,
            roots_path_bytes,
            roots_path_capacity_units,
            entries_path_bytes,
            entries_path_capacity_units,
            referenced_pin_path_bytes,
        })
    }

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
    /// HOST: the accessor reports actual retained cardinalities/capacities without touching IO.
    #[test]
    fn allocation_facts_report_retained_inventory_cardinalities_and_capacities() {
        let root = tempfile::tempdir().unwrap();
        let file = root.path().join("source.safetensors");
        std::fs::write(&file, b"inventory fact fixture").unwrap();
        let spec = LoadSpec::new(WeightsSource::Dir(root.path().to_owned()));
        let mut inventory = SourceInventory::capture(&spec).unwrap();
        inventory.roots[0].reserve(64);
        let facts = inventory.allocation_facts().unwrap();
        assert_eq!(facts.roots_len, inventory.roots.len() as u64);
        assert_eq!(facts.entries_len, inventory.entries.len() as u64);
        assert_eq!(facts.pins_len, inventory.pins.len() as u64);
        assert!(facts.pins_len > 0);
        assert_eq!(facts.roots_capacity, inventory.roots.capacity() as u64);
        assert_eq!(facts.entries_capacity, inventory.entries.capacity() as u64);
        assert_eq!(facts.pins_capacity, inventory.pins.capacity() as u64);
        assert_eq!(
            facts.roots_path_bytes,
            inventory
                .roots
                .iter()
                .map(|p| p.as_os_str().as_encoded_bytes().len() as u64)
                .sum::<u64>()
        );
        assert_eq!(
            facts.roots_path_capacity_units,
            inventory
                .roots
                .iter()
                .map(|p| p.capacity() as u64)
                .sum::<u64>()
        );
        assert!(facts.roots_path_capacity_units > facts.roots_path_bytes);
        assert_eq!(
            facts.entries_path_bytes,
            inventory
                .entries
                .iter()
                .map(|p| p.as_os_str().as_encoded_bytes().len() as u64)
                .sum::<u64>()
        );
        assert_eq!(
            facts.referenced_pin_path_bytes,
            inventory
                .pins
                .iter()
                .map(|p| p.loader_path().as_os_str().as_encoded_bytes().len() as u64)
                .sum::<u64>()
        );
    }
}
