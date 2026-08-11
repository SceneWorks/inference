use runtime_macos::perf_bench::{
    inventory_artifact, ArtifactInventoryReceipt, ArtifactReceipt, ArtifactSnapshotReceipt,
    ARTIFACT_SNAPSHOT_FORMAT,
};
use std::collections::BTreeMap;
use std::ffi::{CString, OsStr};
use std::fs::{self, File, OpenOptions};
use std::io::{self, Seek, SeekFrom};
use std::os::fd::AsRawFd;
use std::os::macos::fs::MetadataExt as MacMetadataExt;
use std::os::unix::ffi::OsStrExt;
use std::os::unix::fs::{MetadataExt as UnixMetadataExt, OpenOptionsExt, PermissionsExt};
use std::path::{Path, PathBuf};

// From <sys/clonefile.h>. The source is already an opened regular file; this flag only prevents
// privileged runs from copying source ownership onto the private destination.
const CLONE_NOOWNERCOPY: u32 = 1 << 1;

#[derive(Clone, Copy)]
enum ClonePreference {
    CloneOrCopy,
    #[cfg(test)]
    CopyOnly,
}

#[derive(Default)]
struct MaterializationCounts {
    cloned_files: u64,
    copied_files: u64,
    source_identities: BTreeMap<PathBuf, (u64, u64)>,
}

impl MaterializationCounts {
    fn record_clone(&mut self) -> Result<(), String> {
        self.cloned_files = self
            .cloned_files
            .checked_add(1)
            .ok_or_else(|| "private snapshot cloned-file count overflowed".to_owned())?;
        Ok(())
    }

    fn record_copy(&mut self) -> Result<(), String> {
        self.copied_files = self
            .copied_files
            .checked_add(1)
            .ok_or_else(|| "private snapshot copied-file count overflowed".to_owned())?;
        Ok(())
    }

    fn total(&self) -> Result<u64, String> {
        self.cloned_files
            .checked_add(self.copied_files)
            .ok_or_else(|| "private snapshot materialized-file count overflowed".to_owned())
    }

    fn record_source_identity(
        &mut self,
        destination: PathBuf,
        source: (u64, u64),
    ) -> Result<(), String> {
        if self.source_identities.insert(destination, source).is_some() {
            return Err("private snapshot repeated a destination file".to_owned());
        }
        Ok(())
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum SnapshotNodeKind {
    Directory,
    File,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
struct SnapshotNodeIdentity {
    kind: SnapshotNodeKind,
    device: u64,
    inode: u64,
}

struct SnapshotOwner {
    directory: Option<tempfile::TempDir>,
}

impl SnapshotOwner {
    fn new(parent: Option<&Path>) -> Result<Self, String> {
        let mut builder = tempfile::Builder::new();
        builder.prefix("sceneworks-mlx-perf-pinned-");
        let directory = match parent {
            Some(parent) => builder.tempdir_in(parent),
            None => builder.tempdir(),
        }
        .map_err(|error| format!("create private artifact snapshot directory: {error}"))?;
        Ok(Self {
            directory: Some(directory),
        })
    }

    fn path(&self) -> &Path {
        self.directory
            .as_ref()
            .expect("snapshot owner path is unavailable after cleanup")
            .path()
    }

    fn cleanup(&mut self) -> Result<(), String> {
        let Some(directory) = self.directory.as_ref() else {
            return Ok(());
        };
        let path = directory.path().to_path_buf();
        match fs::symlink_metadata(&path) {
            Ok(_) => {
                unseal_tree(&path)?;
                fs::remove_dir_all(&path).map_err(|error| {
                    format!("remove private snapshot {}: {error}", path.display())
                })?;
            }
            Err(error) if error.kind() == io::ErrorKind::NotFound => {}
            Err(error) => {
                return Err(format!(
                    "inspect private snapshot for removal {}: {error}",
                    path.display()
                ));
            }
        }
        // The owned path is already gone. Disarm TempDir so it cannot report a second removal.
        let _ = self
            .directory
            .take()
            .expect("snapshot owner remained present")
            .keep();
        Ok(())
    }
}

impl Drop for SnapshotOwner {
    fn drop(&mut self) {
        let _ = self.cleanup();
    }
}

fn combine_snapshot_cleanup(primary: String, cleanup: Result<(), String>) -> String {
    match cleanup {
        Ok(()) => primary,
        Err(cleanup) => {
            format!("{primary}; additionally failed to clean private snapshot: {cleanup}")
        }
    }
}

/// Owns the only path exposed to benchmark providers for one frozen artifact.
///
/// Every leaf is independently cloned or copied from an already-open source file, the complete
/// tree is content-verified against the frozen receipt, and Darwin's user-immutable flag seals the
/// private tree before this value is returned. The original mutable path is never used again.
pub(super) struct OwnedPinnedArtifact {
    owner: SnapshotOwner,
    artifact_path: PathBuf,
    identities: BTreeMap<PathBuf, SnapshotNodeIdentity>,
    counts: MaterializationCounts,
    inventory: ArtifactInventoryReceipt,
}

impl OwnedPinnedArtifact {
    pub(super) fn create(artifact: &ArtifactReceipt) -> Result<Self, String> {
        Self::create_with_hooks(
            artifact,
            None,
            ClonePreference::CloneOrCopy,
            || Ok(()),
            |_| Ok(()),
        )
    }

    fn create_with_hooks<AfterPrecheck, AfterMaterialized>(
        artifact: &ArtifactReceipt,
        temp_parent: Option<&Path>,
        clone_preference: ClonePreference,
        after_precheck: AfterPrecheck,
        after_materialized: AfterMaterialized,
    ) -> Result<Self, String>
    where
        AfterPrecheck: FnOnce() -> Result<(), String>,
        AfterMaterialized: FnOnce(&Path) -> Result<(), String>,
    {
        verify_original_artifact(artifact)?;
        after_precheck()?;

        let mut owner = SnapshotOwner::new(temp_parent)?;
        let construction = (|| {
            let artifact_path = owner.path().join("artifact");
            fs::create_dir(&artifact_path).map_err(|error| {
                format!(
                    "create private artifact root {}: {error}",
                    artifact_path.display()
                )
            })?;
            let mut counts = MaterializationCounts::default();
            materialize_directory(
                &artifact.canonical_path,
                &artifact_path,
                clone_preference,
                &mut counts,
            )?;
            after_materialized(&artifact_path)?;

            // Seal before inventorying. A filesystem that cannot enforce UF_IMMUTABLE is unsafe
            // for path-based providers and is refused rather than silently using writable files.
            seal_tree(owner.path())?;
            let actual = inventory_artifact(&artifact_path).map_err(|error| error.to_string())?;
            if actual != artifact.inventory {
                return Err(format!(
                    "private snapshot for artifact {} does not match the frozen inventory",
                    artifact.key
                ));
            }
            let total = counts.total()?;
            if total != actual.file_count {
                return Err(format!(
                    "private snapshot for artifact {} materialized {total} files but inventoried {}",
                    artifact.key, actual.file_count
                ));
            }
            let identities = collect_identities(owner.path())?;
            verify_source_independence(&counts)?;
            verify_sealed_tree(owner.path())?;
            Ok((artifact_path, identities, counts, actual))
        })();
        let (artifact_path, identities, counts, actual) = match construction {
            Ok(snapshot) => snapshot,
            Err(error) => {
                return Err(combine_snapshot_cleanup(error, owner.cleanup()));
            }
        };
        let mut snapshot = Self {
            owner,
            artifact_path,
            identities,
            counts,
            inventory: actual,
        };
        if let Err(error) = snapshot.verify_integrity() {
            return Err(combine_snapshot_cleanup(error, snapshot.cleanup()));
        }
        Ok(snapshot)
    }

    pub(super) fn path(&self) -> &Path {
        &self.artifact_path
    }

    pub(super) fn verify_integrity(&self) -> Result<(), String> {
        verify_sealed_tree(self.owner.path())?;
        let actual_identities = collect_identities(self.owner.path())?;
        if actual_identities != self.identities {
            return Err(
                "private artifact snapshot path identities changed during execution".to_owned(),
            );
        }
        let actual = inventory_artifact(&self.artifact_path).map_err(|error| error.to_string())?;
        if actual.file_count != self.counts.total()? {
            return Err("private artifact snapshot file count changed during execution".to_owned());
        }
        verify_source_independence(&self.counts)?;
        if actual != self.inventory {
            return Err("private artifact snapshot content changed during execution".to_owned());
        }
        Ok(())
    }

    pub(super) fn cleanup(&mut self) -> Result<(), String> {
        self.owner.cleanup()
    }

    #[cfg(test)]
    fn root_path(&self) -> &Path {
        self.owner.path()
    }
}

/// A child-process view of the parent-owned sealed snapshot.
pub(super) struct VerifiedPinnedArtifact {
    root_path: PathBuf,
    artifact_path: PathBuf,
    identities: BTreeMap<PathBuf, SnapshotNodeIdentity>,
    receipt: ArtifactSnapshotReceipt,
}

impl VerifiedPinnedArtifact {
    pub(super) fn admit(artifact: &ArtifactReceipt, path: &Path) -> Result<Self, String> {
        if !path.is_absolute() {
            return Err("child artifact snapshot path must be absolute".to_owned());
        }
        let artifact_path = path
            .canonicalize()
            .map_err(|error| format!("canonicalize private artifact snapshot: {error}"))?;
        if artifact_path == artifact.canonical_path {
            return Err("child must not load from the original mutable artifact path".to_owned());
        }
        let root_path = artifact_path
            .parent()
            .ok_or_else(|| "private artifact snapshot has no parent".to_owned())?
            .to_path_buf();
        verify_sealed_tree(&root_path)?;
        let inventory = inventory_artifact(&artifact_path).map_err(|error| error.to_string())?;
        if inventory != artifact.inventory {
            return Err(format!(
                "private snapshot for artifact {} does not match the frozen inventory",
                artifact.key
            ));
        }
        let snapshot = Self {
            identities: collect_identities(&root_path)?,
            root_path,
            artifact_path,
            receipt: ArtifactSnapshotReceipt {
                format: ARTIFACT_SNAPSHOT_FORMAT.to_owned(),
                inventory,
            },
        };
        snapshot.verify_integrity()?;
        Ok(snapshot)
    }

    pub(super) fn path(&self) -> &Path {
        &self.artifact_path
    }

    pub(super) fn receipt(&self) -> &ArtifactSnapshotReceipt {
        &self.receipt
    }

    pub(super) fn verify_integrity(&self) -> Result<(), String> {
        verify_sealed_tree(&self.root_path)?;
        if collect_identities(&self.root_path)? != self.identities {
            return Err(
                "private artifact snapshot path identities changed during execution".to_owned(),
            );
        }
        let inventory =
            inventory_artifact(&self.artifact_path).map_err(|error| error.to_string())?;
        if inventory != self.receipt.inventory {
            return Err("private artifact snapshot content changed during execution".to_owned());
        }
        Ok(())
    }
}

fn verify_original_artifact(artifact: &ArtifactReceipt) -> Result<(), String> {
    let metadata = fs::symlink_metadata(&artifact.canonical_path).map_err(|error| {
        format!(
            "inspect frozen artifact path {}: {error}",
            artifact.canonical_path.display()
        )
    })?;
    if metadata.file_type().is_symlink() || !metadata.is_dir() {
        return Err(format!(
            "frozen artifact path is not a real directory: {}",
            artifact.canonical_path.display()
        ));
    }
    let actual = inventory_artifact(&artifact.canonical_path).map_err(|error| error.to_string())?;
    if actual != artifact.inventory {
        return Err(format!(
            "artifact {} changed after the campaign was frozen",
            artifact.key
        ));
    }
    Ok(())
}

fn sorted_entries(path: &Path) -> Result<Vec<PathBuf>, String> {
    let mut entries = fs::read_dir(path)
        .map_err(|error| format!("read snapshot directory {}: {error}", path.display()))?
        .map(|entry| {
            entry
                .map(|entry| entry.path())
                .map_err(|error| format!("read snapshot directory entry: {error}"))
        })
        .collect::<Result<Vec<_>, _>>()?;
    entries.sort();
    Ok(entries)
}

fn materialize_directory(
    source: &Path,
    destination: &Path,
    clone_preference: ClonePreference,
    counts: &mut MaterializationCounts,
) -> Result<(), String> {
    for source_path in sorted_entries(source)? {
        let name = source_path
            .file_name()
            .ok_or_else(|| format!("artifact entry has no file name: {}", source_path.display()))?;
        let destination_path = destination.join(name);
        let link_metadata = fs::symlink_metadata(&source_path).map_err(|error| {
            format!("inspect artifact entry {}: {error}", source_path.display())
        })?;
        let metadata = fs::metadata(&source_path).map_err(|error| {
            format!("resolve artifact entry {}: {error}", source_path.display())
        })?;
        if metadata.is_dir() {
            if link_metadata.file_type().is_symlink() {
                return Err(format!(
                    "private artifact snapshot refuses symlinked directory {}",
                    source_path.display()
                ));
            }
            fs::create_dir(&destination_path).map_err(|error| {
                format!(
                    "create private snapshot directory {}: {error}",
                    destination_path.display()
                )
            })?;
            materialize_directory(&source_path, &destination_path, clone_preference, counts)?;
        } else if metadata.is_file() {
            materialize_file(
                &source_path,
                destination,
                &destination_path,
                name,
                clone_preference,
                counts,
            )?;
        } else {
            return Err(format!(
                "private artifact snapshot refuses special entry {}",
                source_path.display()
            ));
        }
    }
    Ok(())
}

fn materialize_file(
    source_path: &Path,
    destination_parent: &Path,
    destination_path: &Path,
    destination_name: &OsStr,
    clone_preference: ClonePreference,
    counts: &mut MaterializationCounts,
) -> Result<(), String> {
    // Opening first makes subsequent content transfer independent of source path retargeting.
    // File symlinks are intentionally dereferenced into ordinary private files.
    let mut source = File::open(source_path)
        .map_err(|error| format!("open artifact file {}: {error}", source_path.display()))?;
    let source_metadata = source
        .metadata()
        .map_err(|error| format!("stat artifact file {}: {error}", source_path.display()))?;
    if !source_metadata.is_file() {
        return Err(format!(
            "artifact entry stopped being a regular file while snapshotting: {}",
            source_path.display()
        ));
    }

    let try_clone = matches!(clone_preference, ClonePreference::CloneOrCopy)
        && MacMetadataExt::st_flags(&source_metadata) == 0;
    let cloned = if try_clone {
        match clone_open_file(&source, destination_parent, destination_name) {
            Ok(()) => true,
            Err(error) if clone_is_unsupported(&error) => {
                remove_failed_clone(destination_path)?;
                false
            }
            Err(error) => {
                return Err(format!(
                    "clone artifact file {} to {}: {error}",
                    source_path.display(),
                    destination_path.display()
                ));
            }
        }
    } else {
        false
    };
    if cloned {
        counts.record_clone()?;
    } else {
        source
            .seek(SeekFrom::Start(0))
            .map_err(|error| format!("rewind artifact file {}: {error}", source_path.display()))?;
        let mut destination = OpenOptions::new()
            .create_new(true)
            .write(true)
            .open(destination_path)
            .map_err(|error| {
                format!(
                    "create copied snapshot file {}: {error}",
                    destination_path.display()
                )
            })?;
        io::copy(&mut source, &mut destination).map_err(|error| {
            format!(
                "copy artifact file {} to {}: {error}",
                source_path.display(),
                destination_path.display()
            )
        })?;
        destination.sync_all().map_err(|error| {
            format!(
                "sync copied snapshot file {}: {error}",
                destination_path.display()
            )
        })?;
        counts.record_copy()?;
    }

    let destination_metadata = fs::symlink_metadata(destination_path).map_err(|error| {
        format!(
            "stat materialized snapshot file {}: {error}",
            destination_path.display()
        )
    })?;
    if destination_metadata.file_type().is_symlink() || !destination_metadata.is_file() {
        return Err(format!(
            "private snapshot produced a non-regular file: {}",
            destination_path.display()
        ));
    }
    if file_identity(&source_metadata) == file_identity(&destination_metadata) {
        return Err(format!(
            "private snapshot file {} shares source identity; hard links are forbidden",
            destination_path.display()
        ));
    }
    counts.record_source_identity(
        destination_path.to_path_buf(),
        file_identity(&source_metadata),
    )?;
    Ok(())
}

fn verify_source_independence(counts: &MaterializationCounts) -> Result<(), String> {
    if counts.source_identities.len() as u64 != counts.total()? {
        return Err("private snapshot source-identity receipt is incomplete".to_owned());
    }
    for (destination, source_identity) in &counts.source_identities {
        let metadata = fs::symlink_metadata(destination).map_err(|error| {
            format!(
                "stat private snapshot destination {}: {error}",
                destination.display()
            )
        })?;
        if metadata.file_type().is_symlink() || !metadata.is_file() {
            return Err(format!(
                "private snapshot destination is no longer a regular file: {}",
                destination.display()
            ));
        }
        if file_identity(&metadata) == *source_identity {
            return Err(format!(
                "private snapshot destination {} shares its source identity",
                destination.display()
            ));
        }
    }
    Ok(())
}

fn clone_open_file(source: &File, destination_parent: &Path, name: &OsStr) -> io::Result<()> {
    let destination_directory = OpenOptions::new()
        .read(true)
        .custom_flags(libc::O_DIRECTORY | libc::O_NOFOLLOW)
        .open(destination_parent)?;
    let name = CString::new(name.as_bytes())
        .map_err(|_| io::Error::new(io::ErrorKind::InvalidInput, "file name contains NUL"))?;
    // SAFETY: both descriptors remain owned and open for the call, `name` is NUL-terminated, and
    // fclonefileat creates a new entry below the private destination directory.
    let result = unsafe {
        libc::fclonefileat(
            source.as_raw_fd(),
            destination_directory.as_raw_fd(),
            name.as_ptr(),
            CLONE_NOOWNERCOPY,
        )
    };
    if result == 0 {
        Ok(())
    } else {
        Err(io::Error::last_os_error())
    }
}

fn clone_is_unsupported(error: &io::Error) -> bool {
    error
        .raw_os_error()
        .is_some_and(|code| code == libc::EXDEV || code == libc::ENOTSUP || code == libc::ENOSYS)
}

fn remove_failed_clone(path: &Path) -> Result<(), String> {
    match fs::symlink_metadata(path) {
        Ok(metadata) if metadata.is_file() && !metadata.file_type().is_symlink() => {
            fs::remove_file(path)
                .map_err(|error| format!("remove failed clone {}: {error}", path.display()))
        }
        Ok(_) => Err(format!(
            "failed clone left an unsafe destination entry {}",
            path.display()
        )),
        Err(error) if error.kind() == io::ErrorKind::NotFound => Ok(()),
        Err(error) => Err(format!(
            "inspect failed clone destination {}: {error}",
            path.display()
        )),
    }
}

fn file_identity(metadata: &fs::Metadata) -> (u64, u64) {
    (
        UnixMetadataExt::dev(metadata),
        UnixMetadataExt::ino(metadata),
    )
}

#[cfg(test)]
fn set_user_immutable(path: &Path, immutable: bool) -> Result<(), String> {
    let file = OpenOptions::new()
        .read(true)
        .custom_flags(libc::O_NOFOLLOW)
        .open(path)
        .map_err(|error| format!("open snapshot entry {}: {error}", path.display()))?;
    set_open_file_immutable(&file, path, immutable)
}

fn set_open_file_immutable(file: &File, path: &Path, immutable: bool) -> Result<(), String> {
    let metadata = file
        .metadata()
        .map_err(|error| format!("stat snapshot entry {}: {error}", path.display()))?;
    let current = MacMetadataExt::st_flags(&metadata);
    let desired = if immutable {
        current | libc::UF_IMMUTABLE
    } else {
        current & !libc::UF_IMMUTABLE
    };
    // SAFETY: `file` owns a valid descriptor for the complete call; fchflags changes only that
    // opened object and avoids a path-retargeting race.
    let result = unsafe { libc::fchflags(file.as_raw_fd(), desired) };
    if result == 0 {
        Ok(())
    } else {
        Err(format!(
            "{} Darwin user-immutable flag on {}: {}. Use a local macOS filesystem that supports UF_IMMUTABLE",
            if immutable { "set" } else { "clear" },
            path.display(),
            io::Error::last_os_error()
        ))
    }
}

fn set_snapshot_protection(path: &Path, mode: u32, immutable: bool) -> Result<(), String> {
    let file = OpenOptions::new()
        .read(true)
        .custom_flags(libc::O_NOFOLLOW)
        .open(path)
        .map_err(|error| format!("open snapshot entry {}: {error}", path.display()))?;
    if !immutable {
        set_open_file_immutable(&file, path, false)?;
    }
    // SAFETY: `file` is an owned descriptor opened without following the final path component.
    let result = unsafe { libc::fchmod(file.as_raw_fd(), mode as libc::mode_t) };
    if result != 0 {
        return Err(format!(
            "set snapshot mode {mode:o} on {}: {}",
            path.display(),
            io::Error::last_os_error()
        ));
    }
    if immutable {
        set_open_file_immutable(&file, path, true)?;
    }
    Ok(())
}

fn seal_tree(path: &Path) -> Result<(), String> {
    let metadata = fs::symlink_metadata(path)
        .map_err(|error| format!("inspect private snapshot {}: {error}", path.display()))?;
    if metadata.file_type().is_symlink() {
        return Err(format!(
            "private snapshot unexpectedly contains symlink {}",
            path.display()
        ));
    }
    if metadata.is_dir() {
        for entry in sorted_entries(path)? {
            seal_tree(&entry)?;
        }
        set_snapshot_protection(path, 0o500, true)?;
    } else if metadata.is_file() {
        set_snapshot_protection(path, 0o400, true)?;
    } else {
        return Err(format!(
            "private snapshot contains unsupported entry {}",
            path.display()
        ));
    }
    Ok(())
}

fn unseal_tree(path: &Path) -> Result<(), String> {
    let metadata = match fs::symlink_metadata(path) {
        Ok(metadata) => metadata,
        Err(error) if error.kind() == io::ErrorKind::NotFound => return Ok(()),
        Err(error) => {
            return Err(format!(
                "inspect private snapshot for cleanup {}: {error}",
                path.display()
            ));
        }
    };
    if metadata.file_type().is_symlink() {
        return Err(format!(
            "refuse unsafe symlink while cleaning private snapshot {}",
            path.display()
        ));
    }
    if metadata.is_dir() {
        set_snapshot_protection(path, 0o700, false)?;
        for entry in sorted_entries(path)? {
            unseal_tree(&entry)?;
        }
    } else if metadata.is_file() {
        set_snapshot_protection(path, 0o600, false)?;
    } else {
        return Err(format!(
            "refuse unsupported entry while cleaning private snapshot {}",
            path.display()
        ));
    }
    Ok(())
}

fn collect_identities(root: &Path) -> Result<BTreeMap<PathBuf, SnapshotNodeIdentity>, String> {
    fn collect(
        root: &Path,
        path: &Path,
        identities: &mut BTreeMap<PathBuf, SnapshotNodeIdentity>,
    ) -> Result<(), String> {
        let metadata = fs::symlink_metadata(path)
            .map_err(|error| format!("inspect private snapshot {}: {error}", path.display()))?;
        let kind = if metadata.file_type().is_symlink() {
            return Err(format!(
                "private snapshot unexpectedly contains symlink {}",
                path.display()
            ));
        } else if metadata.is_dir() {
            SnapshotNodeKind::Directory
        } else if metadata.is_file() {
            SnapshotNodeKind::File
        } else {
            return Err(format!(
                "private snapshot contains unsupported entry {}",
                path.display()
            ));
        };
        let relative = path
            .strip_prefix(root)
            .expect("snapshot traversal stays below its root")
            .to_path_buf();
        identities.insert(
            relative,
            SnapshotNodeIdentity {
                kind,
                device: UnixMetadataExt::dev(&metadata),
                inode: UnixMetadataExt::ino(&metadata),
            },
        );
        if kind == SnapshotNodeKind::Directory {
            for entry in sorted_entries(path)? {
                collect(root, &entry, identities)?;
            }
        }
        Ok(())
    }

    let mut identities = BTreeMap::new();
    collect(root, root, &mut identities)?;
    Ok(identities)
}

fn verify_sealed_tree(path: &Path) -> Result<(), String> {
    let metadata = fs::symlink_metadata(path)
        .map_err(|error| format!("inspect sealed snapshot {}: {error}", path.display()))?;
    if metadata.file_type().is_symlink() {
        return Err(format!(
            "sealed snapshot contains symlink {}",
            path.display()
        ));
    }
    if MacMetadataExt::st_flags(&metadata) & libc::UF_IMMUTABLE == 0 {
        return Err(format!(
            "private snapshot entry is not Darwin user-immutable: {}",
            path.display()
        ));
    }
    let mode = metadata.permissions().mode();
    if mode & 0o222 != 0 {
        return Err(format!(
            "private snapshot entry remains writable: {}",
            path.display()
        ));
    }
    if metadata.is_dir() {
        if mode & 0o100 == 0 {
            return Err(format!(
                "private snapshot directory is not traversable: {}",
                path.display()
            ));
        }
        for entry in sorted_entries(path)? {
            verify_sealed_tree(&entry)?;
        }
    } else if !metadata.is_file() {
        return Err(format!(
            "sealed snapshot contains unsupported entry {}",
            path.display()
        ));
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use runtime_macos::perf_bench::{ModelTier, INVENTORY_ALGORITHM};
    use std::cell::Cell;
    use std::os::unix::fs::symlink;

    fn receipt(path: &Path) -> ArtifactReceipt {
        let canonical_path = path.canonicalize().unwrap();
        ArtifactReceipt {
            key: "fixture".to_owned(),
            repository: "fixture/repository".to_owned(),
            resolved_revision: "a".repeat(40),
            tier: ModelTier::Q4,
            input_path: path.to_path_buf(),
            inventory: inventory_artifact(&canonical_path).unwrap(),
            canonical_path,
        }
    }

    #[test]
    fn private_snapshot_materializes_layout_file_symlinks_and_independent_identities() {
        let source = tempfile::tempdir().unwrap();
        let artifact_path = source.path().join("artifact");
        fs::create_dir_all(artifact_path.join("nested")).unwrap();
        fs::create_dir(artifact_path.join("empty")).unwrap();
        fs::write(artifact_path.join("weights.bin"), b"weights").unwrap();
        fs::write(artifact_path.join("nested/config.json"), b"{}").unwrap();
        let blob = source.path().join("blob.bin");
        fs::write(&blob, b"external").unwrap();
        symlink("../blob.bin", artifact_path.join("weights-link.bin")).unwrap();
        fs::hard_link(
            artifact_path.join("weights.bin"),
            artifact_path.join("weights-hardlink.bin"),
        )
        .unwrap();
        let artifact = receipt(&artifact_path);
        let snapshots = tempfile::tempdir().unwrap();

        let mut pinned = OwnedPinnedArtifact::create_with_hooks(
            &artifact,
            Some(snapshots.path()),
            ClonePreference::CopyOnly,
            || Ok(()),
            |_| Ok(()),
        )
        .unwrap();
        assert_eq!(pinned.counts.copied_files, 4);
        assert_eq!(pinned.counts.cloned_files, 0);
        assert_eq!(pinned.inventory, artifact.inventory);
        assert!(pinned.path().join("empty").is_dir());
        let copied_link = pinned.path().join("weights-link.bin");
        assert!(fs::symlink_metadata(&copied_link).unwrap().is_file());
        assert!(!fs::symlink_metadata(&copied_link)
            .unwrap()
            .file_type()
            .is_symlink());
        let source_identity =
            file_identity(&fs::metadata(artifact_path.join("weights.bin")).unwrap());
        let copied_identity =
            file_identity(&fs::metadata(pinned.path().join("weights.bin")).unwrap());
        assert_ne!(source_identity, copied_identity);
        let copied_alias_identity =
            file_identity(&fs::metadata(pinned.path().join("weights-hardlink.bin")).unwrap());
        assert_ne!(source_identity, copied_alias_identity);
        assert_ne!(copied_identity, copied_alias_identity);
        fs::write(&blob, b"attacked").unwrap();
        fs::write(artifact_path.join("weights.bin"), b"changed").unwrap();
        assert_eq!(fs::read(&copied_link).unwrap(), b"external");
        assert_eq!(
            fs::read(pinned.path().join("weights.bin")).unwrap(),
            b"weights"
        );
        assert!(fs::set_permissions(
            pinned.path().join("weights.bin"),
            fs::Permissions::from_mode(0o600)
        )
        .is_err());
        assert!(fs::write(pinned.path().join("weights.bin"), b"attack").is_err());
        assert!(fs::remove_file(pinned.path().join("weights.bin")).is_err());
        assert!(fs::rename(
            pinned.path().join("weights.bin"),
            pinned.path().join("replacement.bin")
        )
        .is_err());
        pinned.verify_integrity().unwrap();

        let root = pinned.root_path().to_path_buf();
        pinned.cleanup().unwrap();
        assert!(!root.exists());
        assert_eq!(fs::read_dir(snapshots.path()).unwrap().count(), 0);
    }

    #[test]
    fn real_clone_or_explicit_safe_fallback_has_independent_cow_content() {
        let temp = tempfile::tempdir().unwrap();
        let source_path = temp.path().join("source.bin");
        let clone_path = temp.path().join("clone.bin");
        fs::write(&source_path, b"before").unwrap();
        let source = File::open(&source_path).unwrap();

        match clone_open_file(&source, temp.path(), OsStr::new("clone.bin")) {
            Ok(()) => {}
            Err(error) if clone_is_unsupported(&error) => {
                remove_failed_clone(&clone_path).unwrap();
                let mut counts = MaterializationCounts::default();
                materialize_file(
                    &source_path,
                    temp.path(),
                    &clone_path,
                    OsStr::new("clone.bin"),
                    ClonePreference::CopyOnly,
                    &mut counts,
                )
                .unwrap();
                assert_eq!(counts.copied_files, 1);
            }
            Err(error) => panic!("fclonefileat failed unexpectedly: {error}"),
        }
        assert_ne!(
            file_identity(&fs::metadata(&source_path).unwrap()),
            file_identity(&fs::metadata(&clone_path).unwrap())
        );
        fs::write(&source_path, b"source").unwrap();
        assert_eq!(fs::read(&clone_path).unwrap(), b"before");
        fs::write(&clone_path, b"cloned").unwrap();
        assert_eq!(fs::read(&source_path).unwrap(), b"source");
    }

    #[test]
    fn clone_fallback_is_limited_to_unsupported_or_cross_volume_errors() {
        assert!(clone_is_unsupported(&io::Error::from_raw_os_error(
            libc::EXDEV
        )));
        assert!(clone_is_unsupported(&io::Error::from_raw_os_error(
            libc::ENOTSUP
        )));
        assert!(clone_is_unsupported(&io::Error::from_raw_os_error(
            libc::ENOSYS
        )));
        assert!(!clone_is_unsupported(&io::Error::from_raw_os_error(
            libc::EACCES
        )));
        assert!(!clone_is_unsupported(&io::Error::from_raw_os_error(
            libc::ENOSPC
        )));
        assert!(!clone_is_unsupported(&io::Error::from_raw_os_error(
            libc::EIO
        )));
    }

    #[test]
    fn immutable_source_flags_select_stream_copy_and_drop_cleans_the_sealed_tree() {
        let source = tempfile::tempdir().unwrap();
        let artifact_path = source.path().join("artifact");
        fs::create_dir(&artifact_path).unwrap();
        let weights = artifact_path.join("weights.bin");
        fs::write(&weights, b"weights").unwrap();
        let artifact = receipt(&artifact_path);
        set_user_immutable(&weights, true).unwrap();
        let result = OwnedPinnedArtifact::create(&artifact);
        set_user_immutable(&weights, false).unwrap();
        let pinned = result.unwrap();
        assert_eq!(pinned.counts.cloned_files, 0);
        assert_eq!(pinned.counts.copied_files, 1);
        let root = pinned.root_path().to_path_buf();
        drop(pinned);
        assert!(!root.exists());
    }

    #[test]
    fn source_aba_after_snapshot_creation_cannot_reach_child_execution() {
        let source = tempfile::tempdir().unwrap();
        let artifact_path = source.path().join("artifact");
        fs::create_dir(&artifact_path).unwrap();
        let weights = artifact_path.join("weights.bin");
        fs::write(&weights, b"trusted").unwrap();
        let artifact = receipt(&artifact_path);
        let mut owned = OwnedPinnedArtifact::create(&artifact).unwrap();

        fs::write(&weights, b"attack!").unwrap();
        let child = VerifiedPinnedArtifact::admit(&artifact, owned.path()).unwrap();
        assert_eq!(
            fs::read(child.path().join("weights.bin")).unwrap(),
            b"trusted"
        );
        fs::write(&weights, b"trusted").unwrap();
        child.verify_integrity().unwrap();
        owned.cleanup().unwrap();
    }

    #[test]
    fn transient_source_aba_after_precheck_cannot_admit_or_publish_false_snapshot() {
        let source = tempfile::tempdir().unwrap();
        let artifact_path = source.path().join("artifact");
        fs::create_dir(&artifact_path).unwrap();
        let weights = artifact_path.join("weights.bin");
        fs::write(&weights, b"trusted").unwrap();
        let artifact = receipt(&artifact_path);
        let snapshots = tempfile::tempdir().unwrap();
        let output = source.path().join("run.json");
        let observed_substitution = Cell::new(false);

        let admission = OwnedPinnedArtifact::create_with_hooks(
            &artifact,
            Some(snapshots.path()),
            ClonePreference::CopyOnly,
            || {
                fs::write(&weights, b"attack!").map_err(|error| error.to_string())?;
                Ok(())
            },
            |snapshot| {
                observed_substitution.set(
                    fs::read(snapshot.join("weights.bin")).map_err(|error| error.to_string())?
                        == b"attack!",
                );
                fs::write(&weights, b"trusted").map_err(|error| error.to_string())?;
                Ok(())
            },
        );
        // Publication is reachable only after the parent admits a sealed, equivalent snapshot.
        // The former pre/post-original-path hashes would both see `trusted` in this ABA sequence
        // and reach this write even though the simulated execution observation saw `attack!`.
        if admission.is_ok() {
            fs::write(&output, b"false provenance").unwrap();
        }
        let error = admission
            .err()
            .expect("the substituted snapshot must be refused");
        assert!(observed_substitution.get());
        assert!(
            error.contains("does not match the frozen inventory"),
            "{error}"
        );
        assert_eq!(
            inventory_artifact(&artifact.canonical_path).unwrap(),
            artifact.inventory
        );
        assert!(!output.exists());
        assert_eq!(fs::read_dir(snapshots.path()).unwrap().count(), 0);
    }

    #[test]
    fn construction_error_removes_partially_materialized_tree() {
        let source = tempfile::tempdir().unwrap();
        let artifact_path = source.path().join("artifact");
        fs::create_dir(&artifact_path).unwrap();
        fs::write(artifact_path.join("weights.bin"), b"weights").unwrap();
        let artifact = receipt(&artifact_path);
        let snapshots = tempfile::tempdir().unwrap();

        let error = OwnedPinnedArtifact::create_with_hooks(
            &artifact,
            Some(snapshots.path()),
            ClonePreference::CopyOnly,
            || Ok(()),
            |_| Err("injected construction failure".to_owned()),
        )
        .err()
        .expect("injected construction failure must be returned");
        assert!(error.contains("injected construction failure"), "{error}");
        assert_eq!(fs::read_dir(snapshots.path()).unwrap().count(), 0);
    }

    #[test]
    fn directory_symlink_is_refused_without_leaking_a_private_snapshot() {
        let source = tempfile::tempdir().unwrap();
        let artifact_path = source.path().join("artifact");
        let target = source.path().join("target");
        fs::create_dir(&artifact_path).unwrap();
        fs::create_dir(&target).unwrap();
        fs::write(target.join("weights.bin"), b"weights").unwrap();
        symlink(&target, artifact_path.join("linked-directory")).unwrap();
        let canonical_path = artifact_path.canonicalize().unwrap();
        let artifact = ArtifactReceipt {
            key: "fixture".to_owned(),
            repository: "fixture/repository".to_owned(),
            resolved_revision: "a".repeat(40),
            tier: ModelTier::Q4,
            input_path: artifact_path,
            canonical_path,
            inventory: ArtifactInventoryReceipt {
                algorithm: INVENTORY_ALGORITHM.to_owned(),
                file_count: 1,
                total_bytes: 7,
                sha256: "0".repeat(64),
            },
        };
        let snapshots = tempfile::tempdir().unwrap();
        let error = OwnedPinnedArtifact::create_with_hooks(
            &artifact,
            Some(snapshots.path()),
            ClonePreference::CopyOnly,
            || Ok(()),
            |_| Ok(()),
        )
        .err()
        .expect("directory symlink must be refused");
        assert!(error.contains("refuses symlinked directory"), "{error}");
        assert_eq!(fs::read_dir(snapshots.path()).unwrap().count(), 0);
    }
}
