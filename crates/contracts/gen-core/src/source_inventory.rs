//! Retained source inventory for deferred consumers. Uses the existing replacement-sensitive
//! file tokens, not a header hash masquerading as immutable tensor bytes. Persistent replacement,
//! in-place stat changes and directory membership changes fail before/after each guarded read.
//! Like PinnedWeightsFile, this is not a filesystem transaction against restored-path attacks.
//! Guarded capacity APIs use an owned, source-proven Apple (macOS/iOS) backend. Other targets
//! retain explicit source-gap refusals; their ordinary/native inventory APIs are unchanged.
use crate::{Error, LoadSpec, PinnedWeightsFile, Result};
use std::path::{Path, PathBuf};

#[cfg(target_vendor = "apple")]
#[path = "bounded_clone.rs"]
pub(crate) mod bounded_clone;
#[cfg(target_vendor = "apple")]
#[path = "bounded_fs.rs"]
pub(crate) mod bounded_fs;

#[derive(Clone, Debug)]
pub struct SourceInventory {
    roots: Vec<PathBuf>,
    entries: Vec<PathBuf>,
    pins: Vec<PinnedWeightsFile>,
}

#[cfg(target_vendor = "apple")]
impl bounded_clone::CloneLaw for SourceInventory {
    fn clone_bytes(&self) -> Result<u64> {
        let Self {
            roots,
            entries,
            pins,
        } = self;
        bounded_clone::sum([
            roots.clone_bytes(),
            entries.clone_bytes(),
            pins.clone_bytes(),
        ])
    }
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

#[cfg(target_vendor = "apple")]
fn collect_with_capacity(
    roots: &[PathBuf],
    next: &mut dyn FnMut(u64) -> Result<()>,
) -> Result<(Vec<PathBuf>, Vec<PathBuf>)> {
    use bounded_fs as fs;
    let mut entries = fs::vec(roots.len(), next)?;
    let mut files = Vec::new();
    for root in roots {
        entries.push(fs::copy_path(root, next)?);
    }
    // Iterative traversal: directory depth does not consume a recursive native stack.
    let mut cursor = 0;
    while cursor < entries.len() {
        let path = fs::copy_path(&entries[cursor], next)?;
        cursor += 1;
        let target = fs::stat(&path, true, next)?;
        if fs::is_file(&target) {
            fs::push(&mut files, path, next)?;
            continue;
        }
        let mut directory = fs::Directory::open(&path, next)?;
        while let Some(name) = directory.next(next)? {
            let leaf = Path::new(name);
            if crate::weightsmeta::is_hidden_file(leaf) || crate::weightsmeta::is_hidden_dir(leaf) {
                continue;
            }
            let child = fs::join(&path, name, next)?;
            let target = fs::stat(&child, true, next)?;
            if fs::is_dir(&target) && fs::is_link(&fs::stat(&child, false, next)?) {
                return SourceInventory::metadata_gap(
                    next,
                    "source inventory refuses a symlinked subdirectory",
                );
            }
            fs::push(&mut entries, child, next)?;
        }
    }
    entries.sort_unstable();
    entries.dedup();
    files.sort_unstable();
    files.dedup();
    Ok((entries, files))
}

#[cfg(target_vendor = "apple")]
fn roots_with_capacity(
    spec: &LoadSpec,
    next: &mut dyn FnMut(u64) -> Result<()>,
) -> Result<Vec<PathBuf>> {
    let mut roots = Vec::new();
    let mut result = Ok(());
    spec.visit_source_paths(|path, _| {
        if result.is_ok() {
            result = (|| {
                let path = bounded_fs::absolute(path, next)?;
                bounded_fs::push(&mut roots, path, next)
            })();
        }
    });
    result?;
    roots.sort_unstable();
    roots.dedup();
    Ok(roots)
}

impl SourceInventory {
    /// Exact unresolved filesystem operations; these named gaps never stand for zero bytes.
    pub const READDIR_METADATA_GAP: &str = "source inventory: READDIR_METADATA_GAP (exact ReadDir/DirEntry/Apple libc allocation law unproved)";
    pub const PIN_VERIFY_METADATA_GAP: &str = "source inventory: PIN_VERIFY_METADATA_GAP (exact canonicalize/component-check allocation law unproved)";
    pub const PIN_CAPTURE_METADATA_GAP: &str = "source inventory: PIN_CAPTURE_METADATA_GAP (exact initial seal/canonicalize allocation law unproved)";
    pub const ABSOLUTE_PATH_METADATA_GAP: &str = "source inventory: ABSOLUTE_PATH_METADATA_GAP (relative-path current-directory allocation law unproved)";
    pub const MISSING_INVENTORY_METADATA_GAP: &str = "source inventory: MISSING_INVENTORY_METADATA_GAP (guarded reader requires retained source closure)";

    /// Allocate only a checked fixed diagnostic, then refuse BEFORE the unproved operation.
    /// This is not a successful filesystem allocation certificate or a capacity default.
    pub(crate) fn metadata_gap<T>(
        next: &mut dyn FnMut(u64) -> Result<()>,
        reason: &'static str,
    ) -> Result<T> {
        let layout = std::alloc::Layout::array::<u8>(reason.len())
            .map_err(|_| Error::Io(std::io::ErrorKind::InvalidInput.into()))?;
        next(layout.size() as u64)?;
        Err(Error::Unsupported(reason.into()))
    }

    fn changed_with_capacity<T>(next: &mut dyn FnMut(u64) -> Result<()>) -> Result<T> {
        Self::metadata_gap(next, "source inventory changed after admission")
    }

    /// Earliest directory-source discovery boundary. On Apple, walk actual current
    /// members using caller-owned, prechecked buffers, without acquiring content seals.
    /// This is discovery, not a retained membership receipt or a static manifest.
    pub fn preflight_directory_with_capacity(
        directory: &Path,
        next: &mut dyn FnMut(u64) -> Result<()>,
    ) -> Result<()> {
        #[cfg(target_vendor = "apple")]
        {
            let directory = bounded_fs::absolute(directory, next)?;
            collect_with_capacity(std::slice::from_ref(&directory), next)?;
            Ok(())
        }
        #[cfg(not(target_vendor = "apple"))]
        {
            let _ = directory;
            Self::metadata_gap(next, Self::READDIR_METADATA_GAP)
        }
    }

    /// Compatibility preflight for the aggregate owning Clone of this inventory
    /// and spec. This does not authorize native filesystem operations: callers
    /// still use bind_source_inventory_with_capacity for verification/binding.
    /// New code should prefer the per-operation clone_with_capacity methods.
    pub fn preflight_admission_binding_with_capacity(
        &self,
        spec: &LoadSpec,
        next: &mut dyn FnMut(u64) -> Result<()>,
    ) -> Result<()> {
        #[cfg(target_vendor = "apple")]
        {
            use bounded_clone::CloneLaw;
            self.verify_for_with_capacity(spec, next)?;
            next(bounded_clone::sum([
                self.clone_bytes(),
                spec.clone_bytes(),
            ])?)
        }
        #[cfg(not(target_vendor = "apple"))]
        {
            let _ = spec;
            Self::metadata_gap(next, Self::PIN_CAPTURE_METADATA_GAP)
        }
    }

    /// Initial guarded inventory acquisition. No current-directory Vec or seal acquisition runs
    /// before its real source law. The native `capture` API remains unchanged.
    pub fn capture_with_capacity(
        spec: &LoadSpec,
        next: &mut dyn FnMut(u64) -> Result<()>,
    ) -> Result<Self> {
        #[cfg(target_vendor = "apple")]
        {
            spec.validate_prepared_file_pins_with_capacity(next)?;
            if let Some(retained) = spec.source_inventory() {
                retained.verify_for_with_capacity(spec, next)?;
            }
            let roots = roots_with_capacity(spec, next)?;
            let (entries, files) = collect_with_capacity(&roots, next)?;
            let mut pins = bounded_fs::vec(files.len(), next)?;
            for path in files {
                pins.push(match spec.prepared_file_pins().get(&path) {
                    Some(pin) => pin.clone_with_capacity(next)?,
                    None => PinnedWeightsFile::pin_with_capacity(&path, next)?,
                });
            }
            let inventory = Self {
                roots,
                entries,
                pins,
            };
            inventory.verify_unchanged_with_capacity(next)?;
            Ok(inventory)
        }
        #[cfg(not(target_vendor = "apple"))]
        {
            let mut directory = false;
            spec.visit_source_paths(|_, is_directory| directory |= is_directory);
            Self::metadata_gap(
                next,
                if directory {
                    Self::READDIR_METADATA_GAP
                } else {
                    Self::PIN_CAPTURE_METADATA_GAP
                },
            )
        }
    }

    /// Borrow the original token after guarded lexical normalization. Apple's
    /// caller-buffer current-directory path supports relative bindings; other
    /// targets retain their explicit relative-path source gap.
    pub fn member_pin_with_capacity(
        &self,
        loader_path: &Path,
        next: &mut dyn FnMut(u64) -> Result<()>,
    ) -> Result<Option<&PinnedWeightsFile>> {
        #[cfg(target_vendor = "apple")]
        {
            let loader_path = bounded_fs::absolute(loader_path, next)?;
            Ok(self
                .pins
                .iter()
                .find(|pin| pin.loader_path() == loader_path))
        }
        #[cfg(not(target_vendor = "apple"))]
        {
            if !loader_path.is_absolute() {
                return Self::metadata_gap(next, Self::ABSOLUTE_PATH_METADATA_GAP);
            }
            Ok(self
                .pins
                .iter()
                .find(|pin| pin.loader_path() == loader_path))
        }
    }

    /// Normalize configured roots with prechecked ownership, compare the sorted
    /// lexical binding, then independently verify current membership and pins.
    pub fn verify_for_with_capacity(
        &self,
        spec: &LoadSpec,
        next: &mut dyn FnMut(u64) -> Result<()>,
    ) -> Result<()> {
        #[cfg(target_vendor = "apple")]
        {
            spec.validate_prepared_file_pins_with_capacity(next)?;
            if roots_with_capacity(spec, next)? != self.roots {
                return Self::metadata_gap(
                    next,
                    "source inventory does not cover the configured sources",
                );
            }
            self.verify_unchanged_with_capacity(next)
        }
        #[cfg(not(target_vendor = "apple"))]
        {
            let _ = spec;
            Self::metadata_gap(next, Self::READDIR_METADATA_GAP)
        }
    }

    fn expected_is_file(&self, path: &Path) -> bool {
        // Both acquisition paths retain pins in sorted file-path order.
        self.pins
            .binary_search_by(|pin| pin.loader_path().cmp(path))
            .is_ok()
    }

    fn expected_child_index(&self, directory: &Path, name: &std::ffi::OsStr) -> Option<usize> {
        // Compare a BORROWED joined component sequence against the sorted closure.
        // No joined PathBuf, name allocation or quadratic full-member scan.
        self.entries
            .binary_search_by(|path| {
                path.components().cmp(
                    directory
                        .components()
                        .chain(std::iter::once(std::path::Component::Normal(name))),
                )
            })
            .ok()
    }

    /// Reopen EVERY retained directory, borrowing names from an owned bounded
    /// syscall buffer. Current cardinality never sizes a vector. An extra entry
    /// fails before joining/owning its path; missing members are independently
    /// detected by a sealed-count seen vector. Pins retain all identity checks.
    pub fn verify_unchanged_with_capacity(
        &self,
        next: &mut dyn FnMut(u64) -> Result<()>,
    ) -> Result<()> {
        #[cfg(target_vendor = "apple")]
        {
            use bounded_fs as fs;
            let mut seen = fs::vec::<bool>(self.entries.len(), next)?;
            seen.resize(self.entries.len(), false);
            for (index, path) in self.entries.iter().enumerate() {
                if self.roots.contains(path) {
                    seen[index] = true;
                }
                if self.expected_is_file(path) {
                    continue;
                }
                let target = fs::stat(path, true, next)?;
                if !fs::is_dir(&target) {
                    return Self::changed_with_capacity(next);
                }
                // Configured root directory symlinks are historically allowed;
                // a discovered symlinked subdirectory is not.
                if !self.roots.contains(path) && fs::is_link(&fs::stat(path, false, next)?) {
                    return Self::metadata_gap(
                        next,
                        "source inventory refuses a symlinked subdirectory",
                    );
                }
                let mut current = fs::Directory::open(path, next)?;
                while let Some(name) = current.next(next)? {
                    let leaf = Path::new(name);
                    if crate::weightsmeta::is_hidden_file(leaf)
                        || crate::weightsmeta::is_hidden_dir(leaf)
                    {
                        continue;
                    }
                    let Some(child) = self.expected_child_index(path, name) else {
                        return Self::changed_with_capacity(next);
                    };
                    let expected = &self.entries[child];
                    let target = fs::stat(expected, true, next)?;
                    let expected_file = self.expected_is_file(expected);
                    if (expected_file && !fs::is_file(&target))
                        || (!expected_file && !fs::is_dir(&target))
                    {
                        return Self::changed_with_capacity(next);
                    }
                    if !expected_file && fs::is_link(&fs::stat(expected, false, next)?) {
                        return Self::metadata_gap(
                            next,
                            "source inventory refuses a symlinked subdirectory",
                        );
                    }
                    seen[child] = true;
                }
            }
            if seen.iter().any(|seen| !seen) {
                return Self::changed_with_capacity(next);
            }
            for pin in &self.pins {
                pin.verify_unchanged_with_capacity(next)?;
            }
            Ok(())
        }
        #[cfg(not(target_vendor = "apple"))]
        Self::metadata_gap(next, Self::READDIR_METADATA_GAP)
    }

    /// Same source closure and SAME capacity check both before and after the read, including a
    /// failed read. The closure receives the check explicitly; there is no TLS or no-op fallback.
    pub fn read_unchanged_with_capacity<T, E: From<Error>>(
        &self,
        next: &mut dyn FnMut(u64) -> Result<()>,
        read: impl FnOnce(&mut dyn FnMut(u64) -> Result<()>) -> std::result::Result<T, E>,
    ) -> std::result::Result<T, E> {
        self.verify_unchanged_with_capacity(next).map_err(E::from)?;
        let result = read(next);
        self.verify_unchanged_with_capacity(next).map_err(E::from)?;
        result
    }

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

    /// Clone only known retained owners, checking each requested layout before
    /// ownership is acquired. This does not authorize a LoadSpec/DTO clone.
    pub fn clone_with_capacity(&self, next: &mut dyn FnMut(u64) -> Result<()>) -> Result<Self> {
        #[cfg(target_vendor = "apple")]
        {
            let mut roots = bounded_fs::vec(self.roots.len(), next)?;
            for root in &self.roots {
                roots.push(bounded_fs::copy_path(root, next)?);
            }
            let mut entries = bounded_fs::vec(self.entries.len(), next)?;
            for entry in &self.entries {
                entries.push(bounded_fs::copy_path(entry, next)?);
            }
            let mut pins = bounded_fs::vec(self.pins.len(), next)?;
            for pin in &self.pins {
                pins.push(pin.clone_with_capacity(next)?);
            }
            Ok(Self {
                roots,
                entries,
                pins,
            })
        }
        #[cfg(not(target_vendor = "apple"))]
        Self::metadata_gap(next, Self::PIN_CAPTURE_METADATA_GAP)
    }

    #[cfg(target_vendor = "apple")]
    pub(crate) fn configured_file_pins_with_capacity(
        &self,
        spec: &LoadSpec,
        next: &mut dyn FnMut(u64) -> Result<()>,
    ) -> Result<Vec<(PathBuf, PinnedWeightsFile)>> {
        let mut paths = Vec::new();
        let mut result: Result<()> = Ok(());
        spec.visit_source_paths(|path, directory| {
            if !directory && result.is_ok() {
                result = (|| {
                    let path = bounded_fs::absolute(path, next)?;
                    bounded_fs::push(&mut paths, path, next)
                })();
            }
        });
        result?;
        paths.sort_unstable();
        paths.dedup();
        let mut pins = bounded_fs::vec(paths.len(), next)?;
        for path in paths {
            let Some(pin) = self.pins.iter().find(|pin| pin.loader_path() == path) else {
                return Self::metadata_gap(
                    next,
                    "configured File is absent from retained inventory",
                );
            };
            pins.push((path, pin.clone_with_capacity(next)?));
        }
        Ok(pins)
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

    /// Borrow an existing file token using capture's absolute lexical path convention.
    /// This never re-pins the file, rehashes payloads, canonicalizes a symlink, or builds an inventory.
    pub fn member_pin(&self, loader_path: &Path) -> Result<Option<&PinnedWeightsFile>> {
        let absolute = std::path::absolute(loader_path)?;
        Ok(self
            .pins
            .iter()
            .find(|pin| pin.loader_path() == absolute.as_path()))
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

#[cfg(test)]
mod member_pin_tests {
    use super::*;
    use crate::WeightsSource;

    #[test]
    fn borrowed_members_are_original_tokens_and_inventory_still_detects_replacement() {
        let root = tempfile::tempdir().unwrap();
        let vocab = root.path().join("vocab.json");
        std::fs::write(&vocab, b"{}").unwrap();
        let spec = LoadSpec::new(WeightsSource::Dir(root.path().to_owned()));
        let inventory = SourceInventory::capture(&spec).unwrap();
        let first = inventory.member_pin(&vocab).unwrap().unwrap();
        let second = inventory.member_pin(&vocab).unwrap().unwrap();
        assert!(std::ptr::eq(first, second));
        assert_eq!(first.target_fingerprint().size, 2);
        assert!(inventory
            .member_pin(&root.path().join("missing"))
            .unwrap()
            .is_none());
        inventory.verify_for(&spec).unwrap();
        let replacement = root.path().join("replacement");
        std::fs::write(&replacement, b"{}").unwrap();
        std::fs::rename(&replacement, &vocab).unwrap();
        assert!(inventory.verify_unchanged().is_err());
    }
}

#[cfg(all(test, not(target_vendor = "apple")))]
mod guarded_source_inventory_tests {
    use super::*;
    use crate::WeightsSource;
    use std::cell::Cell;

    #[test]
    fn extra_many_long_named_entries_refuse_before_private_directory_wrapper() {
        let root = tempfile::tempdir().unwrap();
        std::fs::write(root.path().join("sealed"), b"original").unwrap();
        let spec = LoadSpec::new(WeightsSource::Dir(root.path().to_owned()));
        let inventory = SourceInventory::capture(&spec).unwrap();
        for index in 0..257 {
            let name = format!("{index:03}-{}", "x".repeat(240));
            std::fs::write(root.path().join(name), b"extra").unwrap();
        }
        // Legacy semantics still detect directory addition; the guarded implementation
        // does NOT pretend that retained counts bound that CURRENT-member collection.
        assert!(inventory.verify_unchanged().is_err());
        let mut requests = Vec::new();
        let mut check = |bytes| {
            requests.push(bytes);
            if bytes > 4096 {
                Err(Error::Msg("fixture capacity refusal".into()))
            } else {
                Ok(())
            }
        };
        let error = inventory
            .verify_for_with_capacity(&spec, &mut check)
            .unwrap_err();
        match error {
            Error::Unsupported(reason) => assert_eq!(reason, SourceInventory::READDIR_METADATA_GAP),
            other => panic!("unexpected error: {other}"),
        }
        assert_eq!(
            requests,
            vec![SourceInventory::READDIR_METADATA_GAP.len() as u64]
        );
        // No N-current vector/PathBuf reserve, ReadDir or entry-name operation was authorized.
    }

    #[test]
    fn source_gap_precedes_read_and_checks_diagnostic_before_owning_it() {
        let root = tempfile::tempdir().unwrap();
        let file = root.path().join("sealed");
        std::fs::write(&file, b"original").unwrap();
        let spec = LoadSpec::new(WeightsSource::File(file));
        let inventory = SourceInventory::capture(&spec).unwrap();
        let read = Cell::new(false);
        let mut calls = 0;
        let mut check = |_bytes| {
            calls += 1;
            Err(Error::Msg("fixture capacity refusal".into()))
        };
        let result: Result<()> =
            inventory.read_unchanged_with_capacity(&mut check, |_same_check| {
                read.set(true);
                Ok(())
            });
        assert!(result
            .unwrap_err()
            .to_string()
            .contains("fixture capacity refusal"));
        assert_eq!(calls, 1);
        assert!(!read.get());
    }

    #[test]
    fn replacement_and_missing_members_preserve_native_identity_checks() {
        for mutation in ["replace", "remove", "rename"] {
            let root = tempfile::tempdir().unwrap();
            let file = root.path().join("sealed");
            std::fs::write(&file, b"original").unwrap();
            let spec = LoadSpec::new(WeightsSource::Dir(root.path().to_owned()));
            let inventory = SourceInventory::capture(&spec).unwrap();
            match mutation {
                "replace" => {
                    let tmp = root.path().join("replacement");
                    std::fs::write(&tmp, b"original").unwrap();
                    std::fs::rename(tmp, &file).unwrap();
                }
                "remove" => std::fs::remove_file(&file).unwrap(),
                "rename" => std::fs::rename(&file, root.path().join("renamed")).unwrap(),
                _ => unreachable!(),
            }
            assert!(inventory.verify_unchanged().is_err(), "{mutation}");
            let mut check = |bytes| {
                if bytes <= 4096 {
                    Ok(())
                } else {
                    Err(Error::Msg("fixture capacity refusal".into()))
                }
            };
            let result = inventory.verify_unchanged_with_capacity(&mut check);
            assert!(
                matches!(result, Err(Error::Unsupported(reason)) if reason == SourceInventory::READDIR_METADATA_GAP)
            );
        }
    }

    #[test]
    fn sealed_child_lookup_rejects_unknown_name_without_path_join_or_growth() {
        let root = tempfile::tempdir().unwrap();
        std::fs::write(root.path().join("sealed"), b"original").unwrap();
        let spec = LoadSpec::new(WeightsSource::Dir(root.path().to_owned()));
        let inventory = SourceInventory::capture(&spec).unwrap();
        assert!(inventory
            .expected_child_index(root.path(), std::ffi::OsStr::new("sealed"))
            .is_some());
        let long = "x".repeat(10_000);
        assert_eq!(
            inventory.expected_child_index(root.path(), std::ffi::OsStr::new(&long)),
            None
        );
        assert_eq!(
            inventory.expected_child_index(root.path(), std::ffi::OsStr::new("renamed")),
            None
        );
    }

    #[test]
    fn guarded_capture_and_relative_path_have_explicit_preoperation_gaps() {
        let spec = LoadSpec::new(WeightsSource::Dir(PathBuf::from("relative-snapshot")));
        let mut requested = Vec::new();
        let mut check = |bytes| {
            requested.push(bytes);
            if bytes <= 4096 {
                Ok(())
            } else {
                Err(Error::Msg("fixture capacity refusal".into()))
            }
        };
        let result = SourceInventory::capture_with_capacity(&spec, &mut check);
        assert!(
            matches!(result, Err(Error::Unsupported(reason)) if reason == SourceInventory::READDIR_METADATA_GAP)
        );
        assert_eq!(requested.len(), 1);
    }
}

#[cfg(all(test, target_vendor = "apple"))]
#[path = "source_inventory_guarded_tests.rs"]
pub(crate) mod guarded_allocation_tests;

#[cfg(all(test, not(target_vendor = "apple")))]
mod source_prelude_boundary_tests {
    use super::*;

    #[test]
    fn borrowed_prelude_checks_capacity_then_refuses_before_any_directory_operation() {
        // Borrowed nonexistent path is deliberate: this operation must not stat/open it.
        let root = Path::new("/source-prelude-fixture-never-opened");
        let mut requested = Vec::new();
        let mut finite_fixture_check = |bytes| {
            requested.push(bytes);
            if bytes == SourceInventory::READDIR_METADATA_GAP.len() as u64 {
                Ok(())
            } else {
                Err(Error::Msg("unexpected fixture allocation".into()))
            }
        };
        let result =
            SourceInventory::preflight_directory_with_capacity(root, &mut finite_fixture_check);
        assert!(
            matches!(result, Err(Error::Unsupported(reason)) if reason == SourceInventory::READDIR_METADATA_GAP)
        );
        assert_eq!(
            requested,
            vec![SourceInventory::READDIR_METADATA_GAP.len() as u64]
        );
        let mut refuse = |_bytes| Err(Error::Msg("real prelude check refused".into()));
        let result = SourceInventory::preflight_directory_with_capacity(root, &mut refuse);
        assert!(result
            .unwrap_err()
            .to_string()
            .contains("real prelude check refused"));
    }
}

#[cfg(all(test, not(target_vendor = "apple")))]
mod admission_binding_preflight_tests {
    use super::*;

    #[test]
    fn receipt_binding_checks_capacity_before_its_fixed_source_gap() {
        let source = SourceInventory {
            roots: Vec::new(),
            entries: Vec::new(),
            pins: Vec::new(),
        };
        let spec = LoadSpec::new(crate::WeightsSource::File(PathBuf::from(
            "/binding-fixture-never-opened",
        )));
        let mut checked = Vec::new();
        let mut next = |bytes| {
            checked.push(bytes);
            Ok(())
        };
        let result = source.preflight_admission_binding_with_capacity(&spec, &mut next);
        assert!(
            matches!(result, Err(Error::Unsupported(reason)) if reason == SourceInventory::PIN_CAPTURE_METADATA_GAP)
        );
        assert_eq!(
            checked,
            [SourceInventory::PIN_CAPTURE_METADATA_GAP.len() as u64]
        );
        let mut refuse = |_| Err(Error::Canceled);
        assert!(matches!(
            source.preflight_admission_binding_with_capacity(&spec, &mut refuse),
            Err(Error::Canceled)
        ));
    }
}
