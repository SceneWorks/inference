use runtime_macos::perf_bench::{
    inventory_artifact, ArtifactInventoryReceipt, ArtifactReceipt, ArtifactSnapshotReceipt,
    ARTIFACT_SNAPSHOT_FORMAT,
};
use serde::{de::DeserializeOwned, Deserialize, Serialize};
use sha2::{Digest, Sha256};
use std::cell::Cell;
use std::collections::{BTreeMap, BTreeSet};
use std::ffi::{CStr, CString, OsStr, OsString};
use std::fs::{self, File, OpenOptions};
use std::io::{self, Read, Seek, SeekFrom, Write};
use std::os::fd::{AsRawFd, FromRawFd, IntoRawFd, RawFd};
use std::os::macos::fs::MetadataExt as MacMetadataExt;
use std::os::unix::ffi::{OsStrExt, OsStringExt};
use std::os::unix::fs::{
    DirBuilderExt, MetadataExt as UnixMetadataExt, OpenOptionsExt, PermissionsExt,
};
use std::os::unix::process::CommandExt;
use std::path::{Path, PathBuf};
use std::process::Command;

// From <sys/clonefile.h>. The source is already an opened regular file; this flag only prevents
// privileged runs from copying source ownership onto the private destination.
const CLONE_NOOWNERCOPY: u32 = 1 << 1;
const SNAPSHOT_NAMESPACE: &str = "sceneworks-mlx-perf-pinned-v1";
const NAMESPACE_LOCK_FILE: &str = ".namespace.lock";
const OWNER_SIDECAR_PREFIX: &str = "owner-";
const OWNER_STAGE_PREFIX: &str = ".owner-stage-";
const SNAPSHOT_ROOT_PREFIX: &str = "snapshot-";
const OWNER_MARKER_FILE: &str = ".sceneworks-owner.json";
const CLEANUP_RECEIPT_FILE: &str = ".sceneworks-cleanup.json";
const CLEANUP_RECEIPT_TEMP_FILE: &str = ".sceneworks-cleanup.tmp";
const OWNERSHIP_SCHEMA: &str = "sceneworks.mlx-perf-snapshot-owner.v2";
const OWNERSHIP_STAGE_SCHEMA: &str = "sceneworks.mlx-perf-snapshot-owner-stage.v1";
const CLEANUP_SCHEMA: &str = "sceneworks.mlx-perf-snapshot-cleanup.v2";
const OWNERSHIP_HARNESS: &str = "sceneworks-mlx-perf-bench";
const MAX_OWNERSHIP_BYTES: u64 = 64 * 1024;
const MAX_CLEANUP_RECEIPT_BYTES: u64 = 64 * 1024 * 1024;
const CHILD_LEASE_FD: RawFd = 198;

#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(rename_all = "camelCase", deny_unknown_fields)]
struct SnapshotLease {
    schema_version: String,
    harness: String,
    token: String,
    sidecar_name: String,
    sidecar_device: u64,
    sidecar_inode: u64,
    root_name: String,
    root_device: u64,
    root_inode: u64,
}

#[derive(Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(rename_all = "camelCase", deny_unknown_fields)]
struct SnapshotStageIntent {
    schema_version: String,
    harness: String,
    token: String,
    sidecar_name: String,
    sidecar_device: u64,
    sidecar_inode: u64,
    root_name: String,
}

#[derive(Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(rename_all = "camelCase", deny_unknown_fields)]
struct SnapshotOwnership {
    schema_version: String,
    harness: String,
    token: String,
    sidecar_name: String,
    sidecar_device: u64,
    sidecar_inode: u64,
    root_name: String,
    root_device: u64,
    root_inode: u64,
}

#[derive(Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(rename_all = "camelCase", deny_unknown_fields)]
struct SnapshotCleanupReceipt {
    schema_version: String,
    node_count: u64,
    identity_sha256: String,
    nodes: Vec<SnapshotCleanupNode>,
}

#[derive(Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(rename_all = "camelCase", deny_unknown_fields)]
struct SnapshotCleanupNode {
    components_hex: Vec<String>,
    kind: SnapshotNodeKind,
    device: u64,
    inode: u64,
    links: u64,
}

enum LockDisposition {
    Acquired,
    Busy,
}

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

#[derive(Clone, Copy, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(rename_all = "snake_case")]
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

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
struct CleanupNodeIdentity {
    kind: SnapshotNodeKind,
    device: u64,
    inode: u64,
    links: u64,
}

struct SnapshotOwner {
    directory: Option<PathBuf>,
    sidecar: Option<File>,
    sidecar_path: Option<PathBuf>,
    lease: SnapshotLease,
}

fn snapshot_namespace(parent: Option<&Path>) -> Result<PathBuf, String> {
    let require_private_mode = parent.is_none();
    let path = parent.map_or_else(
        || std::env::temp_dir().join(SNAPSHOT_NAMESPACE),
        Path::to_path_buf,
    );
    match fs::DirBuilder::new().mode(0o700).create(&path) {
        Ok(()) => {}
        Err(error) if error.kind() == io::ErrorKind::AlreadyExists => {}
        Err(error) => {
            return Err(format!(
                "create private snapshot namespace {}: {error}",
                path.display()
            ));
        }
    }
    let metadata = fs::symlink_metadata(&path).map_err(|error| {
        format!(
            "inspect private snapshot namespace {}: {error}",
            path.display()
        )
    })?;
    if metadata.file_type().is_symlink()
        || !metadata.is_dir()
        || UnixMetadataExt::uid(&metadata) != unsafe { libc::geteuid() }
        || (require_private_mode && metadata.permissions().mode() & 0o077 != 0)
    {
        return Err(format!(
            "private snapshot namespace must be a private real directory owned by this user: {}",
            path.display()
        ));
    }
    path.canonicalize().map_err(|error| {
        format!(
            "canonicalize private snapshot namespace {}: {error}",
            path.display()
        )
    })
}

fn open_namespace_lock(namespace: &Path) -> Result<File, String> {
    let path = namespace.join(NAMESPACE_LOCK_FILE);
    let file = OpenOptions::new()
        .read(true)
        .write(true)
        .create(true)
        .mode(0o600)
        .custom_flags(libc::O_NOFOLLOW)
        .open(&path)
        .map_err(|error| format!("open snapshot namespace lock {}: {error}", path.display()))?;
    validate_owned_regular_file(&file, &path)?;
    Ok(file)
}

fn validate_owned_regular_file(file: &File, path: &Path) -> Result<fs::Metadata, String> {
    let metadata = file
        .metadata()
        .map_err(|error| format!("stat ownership file {}: {error}", path.display()))?;
    if !metadata.is_file()
        || UnixMetadataExt::nlink(&metadata) != 1
        || UnixMetadataExt::uid(&metadata) != unsafe { libc::geteuid() }
    {
        return Err(format!(
            "ownership file must be a singly-linked regular file owned by this user: {}",
            path.display()
        ));
    }
    Ok(metadata)
}

fn lock_exclusive(file: &File, nonblocking: bool) -> Result<LockDisposition, String> {
    let operation = libc::LOCK_EX | if nonblocking { libc::LOCK_NB } else { 0 };
    // SAFETY: `file` owns a live descriptor and flock neither borrows nor retains pointers.
    if unsafe { libc::flock(file.as_raw_fd(), operation) } == 0 {
        return Ok(LockDisposition::Acquired);
    }
    let error = io::Error::last_os_error();
    if nonblocking
        && error
            .raw_os_error()
            .is_some_and(|code| code == libc::EWOULDBLOCK || code == libc::EAGAIN)
    {
        Ok(LockDisposition::Busy)
    } else {
        Err(format!("lock private snapshot ownership file: {error}"))
    }
}

fn sync_directory(path: &Path) -> Result<(), String> {
    let directory = OpenOptions::new()
        .read(true)
        .custom_flags(libc::O_DIRECTORY | libc::O_NOFOLLOW)
        .open(path)
        .map_err(|error| {
            format!(
                "open snapshot directory {} for sync: {error}",
                path.display()
            )
        })?;
    directory
        .sync_all()
        .map_err(|error| format!("sync snapshot directory {}: {error}", path.display()))
}

fn open_directory(path: &Path) -> Result<File, String> {
    OpenOptions::new()
        .read(true)
        .custom_flags(libc::O_DIRECTORY | libc::O_NOFOLLOW)
        .open(path)
        .map_err(|error| format!("open snapshot directory {}: {error}", path.display()))
}

fn c_name(name: &OsStr) -> Result<CString, String> {
    CString::new(name.as_bytes()).map_err(|_| "snapshot entry name contains NUL".to_owned())
}

fn open_entry_at(parent: &File, name: &OsStr, directory: bool) -> Result<File, String> {
    let name = c_name(name)?;
    let flags = libc::O_RDONLY
        | libc::O_NOFOLLOW
        | libc::O_CLOEXEC
        | if directory { libc::O_DIRECTORY } else { 0 };
    // SAFETY: the parent descriptor and NUL-terminated name remain valid for the call. O_NOFOLLOW
    // prevents the final component from rebinding traversal outside the already-open parent.
    let descriptor = unsafe { libc::openat(parent.as_raw_fd(), name.as_ptr(), flags) };
    if descriptor < 0 {
        Err(format!(
            "open descriptor-relative snapshot entry {:?}: {}",
            OsStr::from_bytes(name.as_bytes()),
            io::Error::last_os_error()
        ))
    } else {
        // SAFETY: openat returned a new owned descriptor.
        Ok(unsafe { File::from_raw_fd(descriptor) })
    }
}

fn duplicate_cloexec(file: &File) -> Result<File, String> {
    // SAFETY: fcntl duplicates the valid descriptor and returns a new owned descriptor.
    let descriptor = unsafe { libc::fcntl(file.as_raw_fd(), libc::F_DUPFD_CLOEXEC, 3) };
    if descriptor < 0 {
        Err(format!(
            "duplicate snapshot directory descriptor: {}",
            io::Error::last_os_error()
        ))
    } else {
        // SAFETY: fcntl returned a new owned descriptor.
        Ok(unsafe { File::from_raw_fd(descriptor) })
    }
}

fn set_cloexec(file: &File, label: &str) -> Result<(), String> {
    // SAFETY: F_GETFD and F_SETFD only inspect and update flags on this owned descriptor.
    let flags = unsafe { libc::fcntl(file.as_raw_fd(), libc::F_GETFD) };
    if flags < 0 {
        return Err(format!(
            "read {label} descriptor flags: {}",
            io::Error::last_os_error()
        ));
    }
    if unsafe { libc::fcntl(file.as_raw_fd(), libc::F_SETFD, flags | libc::FD_CLOEXEC) } < 0 {
        return Err(format!(
            "restore close-on-exec for {label}: {}",
            io::Error::last_os_error()
        ));
    }
    Ok(())
}

fn directory_entry_names(directory: &File) -> Result<Vec<OsString>, String> {
    struct DirectoryStream(*mut libc::DIR);
    impl Drop for DirectoryStream {
        fn drop(&mut self) {
            // SAFETY: the stream is owned by this guard and closed exactly once.
            unsafe { libc::closedir(self.0) };
        }
    }

    // Opening "." creates an independent open-file description. A plain dup would share the
    // directory offset, making one traversal silently exhaust later descriptor-relative reads.
    let independent = open_entry_at(directory, OsStr::new("."), true)?;
    let descriptor = independent.into_raw_fd();
    // SAFETY: fdopendir takes ownership of the duplicated directory descriptor.
    let stream = unsafe { libc::fdopendir(descriptor) };
    if stream.is_null() {
        // SAFETY: fdopendir failed and therefore did not take ownership.
        unsafe { libc::close(descriptor) };
        return Err(format!(
            "open snapshot directory stream: {}",
            io::Error::last_os_error()
        ));
    }
    let stream = DirectoryStream(stream);
    let mut names = Vec::new();
    loop {
        // SAFETY: macOS exposes thread-local errno through __error.
        unsafe { *libc::__error() = 0 };
        // SAFETY: the directory stream remains live and exclusively used by this loop.
        let entry = unsafe { libc::readdir(stream.0) };
        if entry.is_null() {
            // SAFETY: read thread-local errno after readdir returned null.
            let error = unsafe { *libc::__error() };
            if error != 0 {
                return Err(format!(
                    "read snapshot directory stream: {}",
                    io::Error::from_raw_os_error(error)
                ));
            }
            break;
        }
        // SAFETY: d_name is NUL-terminated for a live dirent returned by readdir.
        let bytes = unsafe { CStr::from_ptr((*entry).d_name.as_ptr()) }.to_bytes();
        if bytes != b"." && bytes != b".." {
            names.push(OsString::from_vec(bytes.to_vec()));
        }
    }
    names.sort_by(|left, right| left.as_bytes().cmp(right.as_bytes()));
    Ok(names)
}

fn cleanup_identity(metadata: &fs::Metadata) -> Result<CleanupNodeIdentity, String> {
    let kind = if metadata.is_dir() {
        SnapshotNodeKind::Directory
    } else if metadata.is_file() {
        SnapshotNodeKind::File
    } else {
        return Err("snapshot cleanup refuses a non-file, non-directory inode".to_owned());
    };
    Ok(CleanupNodeIdentity {
        kind,
        device: UnixMetadataExt::dev(metadata),
        inode: UnixMetadataExt::ino(metadata),
        links: UnixMetadataExt::nlink(metadata),
    })
}

fn validate_cleanup_file(
    file: &File,
    expected: CleanupNodeIdentity,
    label: &str,
) -> Result<fs::Metadata, String> {
    let metadata = file
        .metadata()
        .map_err(|error| format!("stat cleanup inode {label}: {error}"))?;
    let actual = cleanup_identity(&metadata)?;
    if actual != expected || UnixMetadataExt::uid(&metadata) != unsafe { libc::geteuid() } {
        return Err(format!(
            "snapshot cleanup inode changed before mutation at {label}: expected {expected:?}, actual {actual:?}"
        ));
    }
    if actual.kind == SnapshotNodeKind::File && actual.links != 1 {
        return Err(format!(
            "snapshot cleanup refuses multiply-linked file at {label}"
        ));
    }
    Ok(metadata)
}

fn identity_at_optional(
    parent: &File,
    name: &OsStr,
) -> Result<Option<CleanupNodeIdentity>, String> {
    let name = c_name(name)?;
    let mut stat = std::mem::MaybeUninit::<libc::stat>::uninit();
    // SAFETY: parent and name are valid, stat points to writable storage, and AT_SYMLINK_NOFOLLOW
    // prevents a final-component symlink from redirecting the check.
    let result = unsafe {
        libc::fstatat(
            parent.as_raw_fd(),
            name.as_ptr(),
            stat.as_mut_ptr(),
            libc::AT_SYMLINK_NOFOLLOW,
        )
    };
    if result != 0 {
        let error = io::Error::last_os_error();
        if error.kind() == io::ErrorKind::NotFound {
            return Ok(None);
        }
        return Err(format!(
            "stat descriptor-relative snapshot entry {:?}: {}",
            OsStr::from_bytes(name.as_bytes()),
            error
        ));
    }
    // SAFETY: fstatat succeeded and initialized stat.
    let stat = unsafe { stat.assume_init() };
    let kind = match stat.st_mode & libc::S_IFMT {
        libc::S_IFDIR => SnapshotNodeKind::Directory,
        libc::S_IFREG => SnapshotNodeKind::File,
        _ => return Err("snapshot cleanup refuses a linked or special entry".to_owned()),
    };
    Ok(Some(CleanupNodeIdentity {
        kind,
        device: stat.st_dev as u64,
        inode: stat.st_ino,
        links: stat.st_nlink as u64,
    }))
}

fn identity_at(parent: &File, name: &OsStr) -> Result<CleanupNodeIdentity, String> {
    identity_at_optional(parent, name)?
        .ok_or_else(|| format!("descriptor-relative snapshot entry {:?} is missing", name))
}

fn unlink_entry_at(
    parent: &File,
    name: &OsStr,
    expected: CleanupNodeIdentity,
) -> Result<(), String> {
    let actual = identity_at(parent, name)?;
    if actual != expected {
        return Err(format!(
            "snapshot entry changed immediately before removal: expected {expected:?}, actual {actual:?}"
        ));
    }
    if actual.kind == SnapshotNodeKind::File && actual.links != 1 {
        return Err("snapshot cleanup refuses to unlink a multiply-linked file".to_owned());
    }
    let name = c_name(name)?;
    let flags = if expected.kind == SnapshotNodeKind::Directory {
        libc::AT_REMOVEDIR
    } else {
        0
    };
    // SAFETY: the descriptor and NUL-terminated relative name are valid for the call.
    if unsafe { libc::unlinkat(parent.as_raw_fd(), name.as_ptr(), flags) } != 0 {
        Err(format!(
            "remove descriptor-relative snapshot entry {:?}: {}",
            OsStr::from_bytes(name.as_bytes()),
            io::Error::last_os_error()
        ))
    } else {
        Ok(())
    }
}

fn rename_noreplace_at(directory: &File, from: &OsStr, to: &OsStr) -> Result<(), String> {
    let from = c_name(from)?;
    let to = c_name(to)?;
    // SAFETY: both names are relative to the same owned directory descriptor. RENAME_EXCL makes
    // publication fail rather than replacing an existing ownership authority.
    let result = unsafe {
        libc::renameatx_np(
            directory.as_raw_fd(),
            from.as_ptr(),
            directory.as_raw_fd(),
            to.as_ptr(),
            libc::RENAME_EXCL,
        )
    };
    if result != 0 {
        Err(format!(
            "publish snapshot ownership {:?} -> {:?}: {}",
            OsStr::from_bytes(from.as_bytes()),
            OsStr::from_bytes(to.as_bytes()),
            io::Error::last_os_error()
        ))
    } else {
        Ok(())
    }
}

fn create_directory_at(parent: &File, name: &OsStr, mode: u32) -> Result<File, String> {
    let name_c = c_name(name)?;
    // SAFETY: the parent descriptor and NUL-terminated relative name are valid for mkdirat.
    if unsafe { libc::mkdirat(parent.as_raw_fd(), name_c.as_ptr(), mode as libc::mode_t) } != 0 {
        return Err(format!(
            "create descriptor-relative snapshot directory {:?}: {}",
            name,
            io::Error::last_os_error()
        ));
    }
    open_entry_at(parent, name, true)
}

fn write_json_to_open_file(file: &mut File, value: &impl Serialize) -> Result<(), String> {
    file.set_len(0)
        .and_then(|()| file.seek(SeekFrom::Start(0)).map(|_| ()))
        .map_err(|error| format!("prepare snapshot ownership file: {error}"))?;
    serde_json::to_writer_pretty(&mut *file, value)
        .map_err(|error| format!("serialize snapshot ownership: {error}"))?;
    file.write_all(b"\n")
        .and_then(|()| file.sync_all())
        .map_err(|error| format!("persist snapshot ownership: {error}"))
}

#[cfg(test)]
fn create_json_file(path: &Path, value: &impl Serialize) -> Result<(), String> {
    let mut file = OpenOptions::new()
        .create_new(true)
        .write(true)
        .mode(0o600)
        .custom_flags(libc::O_NOFOLLOW)
        .open(path)
        .map_err(|error| format!("create snapshot ownership file {}: {error}", path.display()))?;
    serde_json::to_writer_pretty(&mut file, value)
        .map_err(|error| format!("serialize snapshot ownership {}: {error}", path.display()))?;
    file.write_all(b"\n")
        .and_then(|()| file.sync_all())
        .map_err(|error| format!("persist snapshot ownership {}: {error}", path.display()))
}

fn create_json_file_at(
    parent: &File,
    name: &OsStr,
    value: &impl Serialize,
) -> Result<File, String> {
    let name_c = c_name(name)?;
    // SAFETY: openat creates a new regular file below the already-open directory without following
    // a final-component link.
    let descriptor = unsafe {
        libc::openat(
            parent.as_raw_fd(),
            name_c.as_ptr(),
            libc::O_WRONLY | libc::O_CREAT | libc::O_EXCL | libc::O_NOFOLLOW | libc::O_CLOEXEC,
            0o600,
        )
    };
    if descriptor < 0 {
        return Err(format!(
            "create descriptor-relative ownership file {:?}: {}",
            name,
            io::Error::last_os_error()
        ));
    }
    // SAFETY: openat returned a new owned descriptor.
    let mut file = unsafe { File::from_raw_fd(descriptor) };
    serde_json::to_writer_pretty(&mut file, value)
        .map_err(|error| format!("serialize snapshot ownership {:?}: {error}", name))?;
    file.write_all(b"\n")
        .and_then(|()| file.sync_all())
        .map_err(|error| format!("persist snapshot ownership {:?}: {error}", name))?;
    Ok(file)
}

fn read_json_from_open_file<T: DeserializeOwned>(
    file: &mut File,
    path: &Path,
) -> Result<T, String> {
    read_json_from_open_file_with_limit(file, path, MAX_OWNERSHIP_BYTES)
}

fn read_json_from_open_file_with_limit<T: DeserializeOwned>(
    file: &mut File,
    path: &Path,
    maximum_bytes: u64,
) -> Result<T, String> {
    let metadata = validate_owned_regular_file(file, path)?;
    if metadata.len() > maximum_bytes {
        return Err(format!(
            "snapshot ownership file is unexpectedly large: {}",
            path.display()
        ));
    }
    file.seek(SeekFrom::Start(0))
        .map_err(|error| format!("rewind snapshot ownership {}: {error}", path.display()))?;
    let mut bytes = Vec::with_capacity(metadata.len() as usize);
    file.read_to_end(&mut bytes)
        .map_err(|error| format!("read snapshot ownership {}: {error}", path.display()))?;
    serde_json::from_slice(&bytes)
        .map_err(|error| format!("parse snapshot ownership {}: {error}", path.display()))
}

fn read_json_at<T: DeserializeOwned>(parent: &File, name: &OsStr) -> Result<T, String> {
    let mut file = open_entry_at(parent, name, false)?;
    read_json_from_open_file(&mut file, Path::new(name))
}

fn read_cleanup_receipt_at(
    parent: &File,
    name: &OsStr,
) -> Result<(SnapshotCleanupReceipt, File), String> {
    let mut file = open_entry_at(parent, name, false)?;
    let receipt =
        read_json_from_open_file_with_limit(&mut file, Path::new(name), MAX_CLEANUP_RECEIPT_BYTES)?;
    Ok((receipt, file))
}

fn sidecar_token(name: &str) -> Option<&str> {
    let token = name.strip_prefix(OWNER_SIDECAR_PREFIX)?;
    (!token.is_empty() && token.bytes().all(|byte| byte.is_ascii_alphanumeric())).then_some(token)
}

fn stage_token(name: &str) -> Option<&str> {
    let token = name.strip_prefix(OWNER_STAGE_PREFIX)?;
    (!token.is_empty() && token.bytes().all(|byte| byte.is_ascii_alphanumeric())).then_some(token)
}

fn expected_sidecar_name(token: &str) -> String {
    format!("{OWNER_SIDECAR_PREFIX}{token}")
}

fn expected_root_name(token: &str) -> String {
    format!("{SNAPSHOT_ROOT_PREFIX}{token}")
}

#[cfg(test)]
fn snapshot_publication_pause_requested_for_test(phase: &str) -> bool {
    const PHASE_ENV: &str = "SCENEWORKS_P6_PUBLICATION_PAUSE_PHASE";
    std::env::var(PHASE_ENV).as_deref() == Ok(phase)
}

#[cfg(test)]
fn pause_snapshot_publication_for_test(phase: &str) {
    const READY_ENV: &str = "SCENEWORKS_P6_PUBLICATION_PAUSE_READY";
    if !snapshot_publication_pause_requested_for_test(phase) {
        return;
    }
    let ready = std::env::var(READY_ENV).expect("publication pause requires a ready path");
    fs::write(ready, phase).expect("publication pause must publish readiness");
    loop {
        std::thread::park_timeout(std::time::Duration::from_secs(1));
    }
}

fn remove_unpublished_root_at(
    namespace: &File,
    root_name: &OsStr,
    expected_root: CleanupNodeIdentity,
) -> Result<(), String> {
    let root = open_entry_at(namespace, root_name, true)?;
    let names = directory_entry_names(&root)?;
    if names
        .iter()
        .any(|name| name.as_os_str() != OsStr::new(OWNER_MARKER_FILE))
    {
        return Err("unpublished snapshot root contains unexpected entries".to_owned());
    }
    let current_metadata = root
        .metadata()
        .map_err(|error| format!("stat unpublished snapshot root: {error}"))?;
    let current_root = cleanup_identity(&current_metadata)?;
    if UnixMetadataExt::uid(&current_metadata) != unsafe { libc::geteuid() }
        || current_metadata.permissions().mode() & 0o777 != 0o700
    {
        return Err("unpublished snapshot root is not an owned private directory".to_owned());
    }
    if current_root.kind != expected_root.kind
        || current_root.device != expected_root.device
        || current_root.inode != expected_root.inode
    {
        return Err("unpublished snapshot root identity changed".to_owned());
    }
    validate_open_and_entry(
        &root,
        namespace,
        root_name,
        current_root,
        Path::new(root_name),
    )?;
    if names.len() == 1 {
        let marker_name = OsStr::new(OWNER_MARKER_FILE);
        let marker = open_entry_at(&root, marker_name, false)?;
        let identity = cleanup_identity(
            &marker
                .metadata()
                .map_err(|error| format!("stat unpublished owner marker: {error}"))?,
        )?;
        validate_open_and_entry(
            &marker,
            &root,
            marker_name,
            identity,
            Path::new(OWNER_MARKER_FILE),
        )?;
        unlink_entry_at(&root, marker_name, identity)?;
    }
    root.sync_all()
        .map_err(|error| format!("sync unpublished snapshot root: {error}"))?;
    let terminal = cleanup_identity(
        &root
            .metadata()
            .map_err(|error| format!("stat emptied unpublished snapshot root: {error}"))?,
    )?;
    if terminal.kind != expected_root.kind
        || terminal.device != expected_root.device
        || terminal.inode != expected_root.inode
    {
        return Err("unpublished snapshot root changed during cleanup".to_owned());
    }
    validate_open_and_entry(&root, namespace, root_name, terminal, Path::new(root_name))?;
    unlink_entry_at(namespace, root_name, terminal)
}

fn validate_stage_intent(
    intent: &SnapshotStageIntent,
    token: &str,
    sidecar_metadata: &fs::Metadata,
) -> Result<(), String> {
    if intent.schema_version != OWNERSHIP_STAGE_SCHEMA
        || intent.harness != OWNERSHIP_HARNESS
        || intent.token != token
        || intent.sidecar_name != expected_sidecar_name(token)
        || intent.sidecar_device != UnixMetadataExt::dev(sidecar_metadata)
        || intent.sidecar_inode != UnixMetadataExt::ino(sidecar_metadata)
        || intent.root_name != expected_root_name(token)
    {
        return Err("foreign or malformed staged snapshot ownership intent".to_owned());
    }
    Ok(())
}

fn unpublished_root_identity_at(
    namespace: &File,
    root_name: &OsStr,
) -> Result<CleanupNodeIdentity, String> {
    let root = open_entry_at(namespace, root_name, true)?;
    let metadata = root
        .metadata()
        .map_err(|error| format!("stat unpublished snapshot root: {error}"))?;
    let identity = cleanup_identity(&metadata)?;
    if identity.kind != SnapshotNodeKind::Directory
        || UnixMetadataExt::uid(&metadata) != unsafe { libc::geteuid() }
        || metadata.permissions().mode() & 0o777 != 0o700
    {
        return Err("unpublished snapshot root is not an owned private directory".to_owned());
    }
    validate_open_and_entry(&root, namespace, root_name, identity, Path::new(root_name))?;
    Ok(identity)
}

fn validate_unpublished_marker_authority(
    namespace: &File,
    root_name: &OsStr,
    token: &str,
    sidecar_metadata: &fs::Metadata,
) -> Result<CleanupNodeIdentity, String> {
    let root = open_entry_at(namespace, root_name, true)?;
    let root_metadata = root
        .metadata()
        .map_err(|error| format!("stat unpublished snapshot root: {error}"))?;
    let root_identity = cleanup_identity(&root_metadata)?;
    if root_identity.kind != SnapshotNodeKind::Directory
        || UnixMetadataExt::uid(&root_metadata) != unsafe { libc::geteuid() }
        || root_metadata.permissions().mode() & 0o777 != 0o700
    {
        return Err("unpublished snapshot root is not an owned private directory".to_owned());
    }
    validate_open_and_entry(
        &root,
        namespace,
        root_name,
        root_identity,
        Path::new(root_name),
    )?;
    let marker: SnapshotOwnership = read_json_at(&root, OsStr::new(OWNER_MARKER_FILE))?;
    if marker.schema_version != OWNERSHIP_SCHEMA
        || marker.harness != OWNERSHIP_HARNESS
        || marker.token != token
        || marker.sidecar_name != expected_sidecar_name(token)
        || marker.sidecar_device != UnixMetadataExt::dev(sidecar_metadata)
        || marker.sidecar_inode != UnixMetadataExt::ino(sidecar_metadata)
        || marker.root_name != expected_root_name(token)
        || marker.root_device != root_identity.device
        || marker.root_inode != root_identity.inode
    {
        return Err("unpublished snapshot marker does not bind staged authority".to_owned());
    }
    Ok(root_identity)
}

impl SnapshotOwner {
    fn new(parent: Option<&Path>) -> Result<Self, String> {
        let namespace = snapshot_namespace(parent)?;
        let namespace_lock = open_namespace_lock(&namespace)?;
        match lock_exclusive(&namespace_lock, false)? {
            LockDisposition::Acquired => {}
            LockDisposition::Busy => unreachable!("blocking namespace lock cannot be busy"),
        }

        let namespace_directory = open_directory(&namespace)?;
        let mut builder = tempfile::Builder::new();
        builder.prefix(OWNER_STAGE_PREFIX);
        let mut staged_sidecar = builder
            .tempfile_in(&namespace)
            .map_err(|error| format!("create staged snapshot ownership sidecar: {error}"))?;
        match lock_exclusive(staged_sidecar.as_file(), false)? {
            LockDisposition::Acquired => {}
            LockDisposition::Busy => unreachable!("new ownership sidecar cannot be busy"),
        }
        #[cfg(test)]
        pause_snapshot_publication_for_test("empty-stage");
        let staged_name = staged_sidecar
            .path()
            .file_name()
            .and_then(OsStr::to_str)
            .ok_or_else(|| "staged snapshot ownership name is not UTF-8".to_owned())?
            .to_owned();
        let token = stage_token(&staged_name)
            .ok_or_else(|| "staged snapshot ownership sidecar has an unsafe name".to_owned())?
            .to_owned();
        let sidecar_name = expected_sidecar_name(&token);
        let root_name = expected_root_name(&token);
        let sidecar_metadata =
            validate_owned_regular_file(staged_sidecar.as_file(), staged_sidecar.path())?;
        let stage_intent = SnapshotStageIntent {
            schema_version: OWNERSHIP_STAGE_SCHEMA.to_owned(),
            harness: OWNERSHIP_HARNESS.to_owned(),
            token: token.clone(),
            sidecar_name: sidecar_name.clone(),
            sidecar_device: UnixMetadataExt::dev(&sidecar_metadata),
            sidecar_inode: UnixMetadataExt::ino(&sidecar_metadata),
            root_name: root_name.clone(),
        };
        write_json_to_open_file(staged_sidecar.as_file_mut(), &stage_intent)?;
        namespace_directory
            .sync_all()
            .map_err(|error| format!("sync staged snapshot intent: {error}"))?;
        #[cfg(test)]
        pause_snapshot_publication_for_test("intent-stage");
        let root_directory =
            create_directory_at(&namespace_directory, OsStr::new(&root_name), 0o700)?;
        let root_metadata = root_directory
            .metadata()
            .map_err(|error| format!("stat staged snapshot root: {error}"))?;
        #[cfg(test)]
        pause_snapshot_publication_for_test("root-stage");
        let lease = SnapshotLease {
            schema_version: OWNERSHIP_SCHEMA.to_owned(),
            harness: OWNERSHIP_HARNESS.to_owned(),
            token,
            sidecar_name: sidecar_name.clone(),
            sidecar_device: UnixMetadataExt::dev(&sidecar_metadata),
            sidecar_inode: UnixMetadataExt::ino(&sidecar_metadata),
            root_name: root_name.clone(),
            root_device: UnixMetadataExt::dev(&root_metadata),
            root_inode: UnixMetadataExt::ino(&root_metadata),
        };
        let ownership = SnapshotOwnership {
            schema_version: OWNERSHIP_SCHEMA.to_owned(),
            harness: OWNERSHIP_HARNESS.to_owned(),
            token: lease.token.clone(),
            sidecar_name: sidecar_name.clone(),
            sidecar_device: lease.sidecar_device,
            sidecar_inode: lease.sidecar_inode,
            root_name: lease.root_name.clone(),
            root_device: lease.root_device,
            root_inode: lease.root_inode,
        };
        let marker_publication =
            create_json_file_at(&root_directory, OsStr::new(OWNER_MARKER_FILE), &ownership)
                .and_then(|_| {
                    root_directory
                        .sync_all()
                        .map_err(|error| format!("sync staged snapshot root: {error}"))
                });
        if let Err(error) = marker_publication {
            let cleanup = remove_unpublished_root_at(
                &namespace_directory,
                OsStr::new(&root_name),
                cleanup_identity(&root_metadata)?,
            );
            return Err(combine_snapshot_cleanup(error, cleanup));
        }
        #[cfg(test)]
        pause_snapshot_publication_for_test("marker-stage");
        #[cfg(test)]
        if snapshot_publication_pause_requested_for_test("truncated-lease-stage") {
            staged_sidecar
                .as_file_mut()
                .set_len(0)
                .and_then(|()| staged_sidecar.as_file().sync_all())
                .expect("truncate staged lease for crash injection");
            pause_snapshot_publication_for_test("truncated-lease-stage");
        }
        let lease_publication = write_json_to_open_file(staged_sidecar.as_file_mut(), &lease)
            .and_then(|()| {
                namespace_directory
                    .sync_all()
                    .map_err(|error| format!("sync staged snapshot namespace: {error}"))
            });
        if let Err(error) = lease_publication {
            let cleanup = remove_unpublished_root_at(
                &namespace_directory,
                OsStr::new(&root_name),
                cleanup_identity(&root_metadata)?,
            );
            return Err(combine_snapshot_cleanup(error, cleanup));
        }
        #[cfg(test)]
        pause_snapshot_publication_for_test("complete-stage");
        let (sidecar, _staged_path) = match staged_sidecar.keep() {
            Ok(kept) => kept,
            Err(error) => {
                let primary = format!("persist private snapshot ownership sidecar: {error}");
                return Err(combine_snapshot_cleanup(
                    primary,
                    remove_unpublished_root_at(
                        &namespace_directory,
                        OsStr::new(&root_name),
                        cleanup_identity(&root_metadata)?,
                    ),
                ));
            }
        };
        if let Err(error) = rename_noreplace_at(
            &namespace_directory,
            OsStr::new(&staged_name),
            OsStr::new(&sidecar_name),
        )
        .and_then(|()| {
            namespace_directory
                .sync_all()
                .map_err(|error| format!("sync published snapshot authority: {error}"))
        }) {
            let sidecar_cleanup =
                remove_locked_sidecar_at(&namespace_directory, &sidecar, OsStr::new(&staged_name));
            let root_cleanup = remove_unpublished_root_at(
                &namespace_directory,
                OsStr::new(&root_name),
                cleanup_identity(&root_metadata)?,
            );
            return Err(combine_snapshot_cleanup(
                combine_snapshot_cleanup(error, sidecar_cleanup),
                root_cleanup,
            ));
        }
        let root = namespace.join(&root_name);
        let sidecar_path = namespace.join(&sidecar_name);
        drop(namespace_lock);
        Ok(Self {
            directory: Some(root),
            sidecar: Some(sidecar),
            sidecar_path: Some(sidecar_path),
            lease,
        })
    }

    fn path(&self) -> &Path {
        self.directory
            .as_deref()
            .expect("snapshot owner path is unavailable after cleanup")
    }

    fn inherit_child_lease(&self, command: &mut Command) -> Result<(), String> {
        let source = self
            .sidecar
            .as_ref()
            .ok_or_else(|| "snapshot ownership lease is unavailable after cleanup".to_owned())?
            .as_raw_fd();
        // SAFETY: after fork and before exec, the closure invokes only async-signal-safe libc
        // descriptor operations. dup2 creates a child-only reference to the same locked open-file
        // description; clearing FD_CLOEXEC makes that reference survive exec.
        unsafe {
            command.pre_exec(move || {
                if libc::dup2(source, CHILD_LEASE_FD) < 0 {
                    return Err(io::Error::last_os_error());
                }
                if libc::fcntl(CHILD_LEASE_FD, libc::F_SETFD, 0) < 0 {
                    return Err(io::Error::last_os_error());
                }
                Ok(())
            });
        }
        Ok(())
    }

    fn configure_child_lease(&self, command: &mut Command) -> Result<(), String> {
        self.inherit_child_lease(command)?;
        command
            .arg("--artifact-lease-fd")
            .arg(CHILD_LEASE_FD.to_string());
        Ok(())
    }

    #[cfg(test)]
    pub(super) fn duplicate_lease_fd(&self) -> Result<RawFd, String> {
        let sidecar = self
            .sidecar
            .as_ref()
            .ok_or_else(|| "snapshot ownership lease is unavailable after cleanup".to_owned())?;
        Ok(duplicate_cloexec(sidecar)?.into_raw_fd())
    }

    fn prepare_for_sealing(&self) -> Result<(), String> {
        let temporary = self.path().join(CLEANUP_RECEIPT_TEMP_FILE);
        let mut temporary_file = OpenOptions::new()
            .create_new(true)
            .read(true)
            .write(true)
            .mode(0o600)
            .custom_flags(libc::O_NOFOLLOW)
            .open(&temporary)
            .map_err(|error| format!("create snapshot cleanup receipt: {error}"))?;
        // Collect while the staging inode is linked so the receipt records the directory's exact
        // post-publication link identity. rename below preserves both that link count and inode.
        let root = open_directory(self.path())?;
        let mut identities = collect_cleanup_identity_map(&root)?;
        identities
            .remove(Path::new(CLEANUP_RECEIPT_TEMP_FILE))
            .ok_or_else(|| "cleanup receipt staging inode disappeared".to_owned())?;
        let receipt = cleanup_receipt_from_identities(&identities)?;
        write_json_to_open_file(&mut temporary_file, &receipt)?;
        let published = self.path().join(CLEANUP_RECEIPT_FILE);
        if let Err(error) = fs::rename(&temporary, &published)
            .map_err(|error| format!("publish snapshot cleanup receipt: {error}"))
            .and_then(|()| sync_directory(self.path()))
        {
            let _ = fs::remove_file(&temporary);
            return Err(error);
        }
        Ok(())
    }

    fn cleanup(&mut self) -> Result<(), String> {
        let Some(path) = self.directory.as_ref() else {
            return Ok(());
        };
        let namespace = path
            .parent()
            .ok_or_else(|| "snapshot root has no namespace".to_owned())?;
        let namespace_directory = open_directory(namespace)?;
        if identity_at_optional(&namespace_directory, OsStr::new(&self.lease.root_name))?.is_some()
        {
            remove_owned_snapshot_tree_at(
                &namespace_directory,
                &self.lease,
                false,
                &mut |_, _| Ok(()),
            )?;
        }
        self.directory.take();
        if let (Some(sidecar), Some(sidecar_path)) =
            (self.sidecar.as_ref(), self.sidecar_path.as_ref())
        {
            let sidecar_name = sidecar_path
                .file_name()
                .ok_or_else(|| "snapshot ownership sidecar has no filename".to_owned())?;
            remove_locked_sidecar_at(&namespace_directory, sidecar, sidecar_name)?;
            namespace_directory
                .sync_all()
                .map_err(|error| format!("sync removed snapshot authority: {error}"))?;
            self.sidecar.take();
            self.sidecar_path.take();
        }
        Ok(())
    }

    #[cfg(test)]
    fn abandon_for_termination_test(mut self) -> PathBuf {
        let root = self
            .directory
            .take()
            .expect("termination fixture retains its snapshot root");
        let sidecar = self
            .sidecar
            .take()
            .expect("termination fixture retains its ownership sidecar");
        self.sidecar_path
            .take()
            .expect("termination fixture retains its ownership path");
        drop(sidecar);
        root
    }

    #[cfg(test)]
    fn abandon_with_lease_for_test(mut self) -> (PathBuf, SnapshotLease) {
        let root = self
            .directory
            .take()
            .expect("termination fixture retains its snapshot root");
        let sidecar = self
            .sidecar
            .take()
            .expect("termination fixture retains its ownership sidecar");
        self.sidecar_path
            .take()
            .expect("termination fixture retains its ownership path");
        drop(sidecar);
        (root, self.lease.clone())
    }
}

impl Drop for SnapshotOwner {
    fn drop(&mut self) {
        let _ = self.cleanup();
    }
}

fn validate_lease(
    lease: &SnapshotLease,
    token: &str,
    sidecar_metadata: &fs::Metadata,
) -> Result<(), String> {
    if lease.schema_version != OWNERSHIP_SCHEMA
        || lease.harness != OWNERSHIP_HARNESS
        || lease.token != token
        || lease.sidecar_name != expected_sidecar_name(token)
        || lease.sidecar_device != UnixMetadataExt::dev(sidecar_metadata)
        || lease.sidecar_inode != UnixMetadataExt::ino(sidecar_metadata)
        || lease.root_name != expected_root_name(token)
    {
        return Err(format!(
            "foreign or malformed private snapshot ownership metadata {:?}",
            lease.sidecar_name
        ));
    }
    Ok(())
}

fn validate_lease_root(lease: &SnapshotLease, root: &File) -> Result<CleanupNodeIdentity, String> {
    let root_metadata = root
        .metadata()
        .map_err(|error| format!("stat owned private snapshot root: {error}"))?;
    if root_metadata.file_type().is_symlink()
        || !root_metadata.is_dir()
        || UnixMetadataExt::uid(&root_metadata) != unsafe { libc::geteuid() }
        || UnixMetadataExt::dev(&root_metadata) != lease.root_device
        || UnixMetadataExt::ino(&root_metadata) != lease.root_inode
    {
        return Err("owned private snapshot root no longer matches sidecar authority".to_owned());
    }
    cleanup_identity(&root_metadata)
}

fn validate_snapshot_ownership_fd(root: &File, lease: &SnapshotLease) -> Result<(), String> {
    validate_lease_root(lease, root)?;
    let marker: SnapshotOwnership = read_json_at(root, OsStr::new(OWNER_MARKER_FILE))?;
    if marker.schema_version != OWNERSHIP_SCHEMA
        || marker.harness != OWNERSHIP_HARNESS
        || marker.token != lease.token
        || marker.sidecar_name != lease.sidecar_name
        || marker.sidecar_device != lease.sidecar_device
        || marker.sidecar_inode != lease.sidecar_inode
        || marker.root_name != lease.root_name
        || marker.root_device != lease.root_device
        || marker.root_inode != lease.root_inode
    {
        return Err("private snapshot ownership marker does not bind sidecar authority".to_owned());
    }
    Ok(())
}

fn collect_cleanup_identity_map(
    root: &File,
) -> Result<BTreeMap<PathBuf, CleanupNodeIdentity>, String> {
    fn collect(
        file: &File,
        relative: &Path,
        identities: &mut BTreeMap<PathBuf, CleanupNodeIdentity>,
    ) -> Result<(), String> {
        let metadata = file
            .metadata()
            .map_err(|error| format!("stat descriptor-relative cleanup inode: {error}"))?;
        if UnixMetadataExt::uid(&metadata) != unsafe { libc::geteuid() } {
            return Err(format!(
                "snapshot cleanup refuses an inode owned by another user at {}",
                relative.display()
            ));
        }
        let identity = cleanup_identity(&metadata)?;
        if identity.kind == SnapshotNodeKind::File && identity.links != 1 {
            return Err(format!(
                "snapshot cleanup refuses multiply-linked file at {}",
                relative.display()
            ));
        }
        if identities
            .insert(relative.to_path_buf(), identity)
            .is_some()
        {
            return Err("snapshot cleanup repeated a relative path".to_owned());
        }
        if identity.kind == SnapshotNodeKind::Directory {
            for name in directory_entry_names(file)? {
                let entry_identity = identity_at(file, &name)?;
                let child = open_entry_at(
                    file,
                    &name,
                    entry_identity.kind == SnapshotNodeKind::Directory,
                )?;
                let opened_identity =
                    cleanup_identity(&child.metadata().map_err(|error| {
                        format!("stat opened snapshot entry {:?}: {error}", name)
                    })?)?;
                if opened_identity != entry_identity {
                    return Err(format!(
                        "snapshot entry changed while collecting cleanup identities: {:?}",
                        name
                    ));
                }
                collect(&child, &relative.join(&name), identities)?;
            }
        }
        Ok(())
    }

    let mut identities = BTreeMap::new();
    collect(root, Path::new(""), &mut identities)?;
    Ok(identities)
}

fn cleanup_receipt_from_identities(
    identities: &BTreeMap<PathBuf, CleanupNodeIdentity>,
) -> Result<SnapshotCleanupReceipt, String> {
    let mut nodes = Vec::new();
    for (relative, identity) in identities {
        if relative == Path::new(CLEANUP_RECEIPT_FILE) {
            continue;
        }
        let mut components_hex = Vec::new();
        for component in relative.components() {
            let std::path::Component::Normal(name) = component else {
                return Err(format!(
                    "cleanup receipt contains unsafe relative path {}",
                    relative.display()
                ));
            };
            components_hex.push(hex_bytes(name.as_bytes()));
        }
        nodes.push(SnapshotCleanupNode {
            components_hex,
            kind: identity.kind,
            device: identity.device,
            inode: identity.inode,
            links: identity.links,
        });
    }
    let node_count = u64::try_from(nodes.len())
        .map_err(|_| "snapshot cleanup node count overflowed".to_owned())?;
    let identity_sha256 = cleanup_nodes_digest(&nodes)?;
    Ok(SnapshotCleanupReceipt {
        schema_version: CLEANUP_SCHEMA.to_owned(),
        node_count,
        identity_sha256,
        nodes,
    })
}

fn hex_bytes(bytes: &[u8]) -> String {
    const HEX: &[u8; 16] = b"0123456789abcdef";
    let mut encoded = String::with_capacity(bytes.len() * 2);
    for byte in bytes {
        encoded.push(HEX[(byte >> 4) as usize] as char);
        encoded.push(HEX[(byte & 0x0f) as usize] as char);
    }
    encoded
}

fn decode_hex_name(encoded: &str) -> Result<OsString, String> {
    if encoded.is_empty() || !encoded.len().is_multiple_of(2) {
        return Err("cleanup receipt contains an invalid encoded path component".to_owned());
    }
    fn nibble(byte: u8) -> Option<u8> {
        match byte {
            b'0'..=b'9' => Some(byte - b'0'),
            b'a'..=b'f' => Some(byte - b'a' + 10),
            _ => None,
        }
    }
    let mut decoded = Vec::with_capacity(encoded.len() / 2);
    for pair in encoded.as_bytes().chunks_exact(2) {
        let high = nibble(pair[0])
            .ok_or_else(|| "cleanup receipt contains non-canonical hex".to_owned())?;
        let low = nibble(pair[1])
            .ok_or_else(|| "cleanup receipt contains non-canonical hex".to_owned())?;
        decoded.push((high << 4) | low);
    }
    if decoded == b"." || decoded == b".." || decoded.contains(&0) || decoded.contains(&b'/') {
        return Err("cleanup receipt contains an unsafe path component".to_owned());
    }
    Ok(OsString::from_vec(decoded))
}

fn cleanup_nodes_digest(nodes: &[SnapshotCleanupNode]) -> Result<String, String> {
    let mut digest = Sha256::new();
    for node in nodes {
        digest.update(
            u64::try_from(node.components_hex.len())
                .map_err(|_| "cleanup receipt component count overflowed".to_owned())?
                .to_le_bytes(),
        );
        for component in &node.components_hex {
            let decoded = decode_hex_name(component)?;
            let bytes = decoded.as_bytes();
            digest.update(
                u64::try_from(bytes.len())
                    .map_err(|_| "cleanup receipt path component is too large".to_owned())?
                    .to_le_bytes(),
            );
            digest.update(bytes);
        }
        digest.update([if node.kind == SnapshotNodeKind::Directory {
            b'd'
        } else {
            b'f'
        }]);
        digest.update(node.device.to_le_bytes());
        digest.update(node.inode.to_le_bytes());
        digest.update(node.links.to_le_bytes());
    }
    Ok(format!("{:x}", digest.finalize()))
}

fn cleanup_receipt_identity_map(
    receipt: &SnapshotCleanupReceipt,
) -> Result<BTreeMap<PathBuf, CleanupNodeIdentity>, String> {
    if receipt.schema_version != CLEANUP_SCHEMA
        || receipt.node_count != receipt.nodes.len() as u64
        || receipt.identity_sha256 != cleanup_nodes_digest(&receipt.nodes)?
    {
        return Err(
            "foreign or internally inconsistent private snapshot cleanup receipt".to_owned(),
        );
    }
    let mut identities = BTreeMap::new();
    let mut previous: Option<PathBuf> = None;
    for node in &receipt.nodes {
        let mut relative = PathBuf::new();
        for component in &node.components_hex {
            relative.push(decode_hex_name(component)?);
        }
        if previous.as_ref().is_some_and(|path| path >= &relative)
            || identities
                .insert(
                    relative.clone(),
                    CleanupNodeIdentity {
                        kind: node.kind,
                        device: node.device,
                        inode: node.inode,
                        links: node.links,
                    },
                )
                .is_some()
        {
            return Err("cleanup receipt paths are duplicated or out of order".to_owned());
        }
        previous = Some(relative);
    }
    if !identities
        .get(Path::new(""))
        .is_some_and(|identity| identity.kind == SnapshotNodeKind::Directory)
    {
        return Err("cleanup receipt omits its root directory".to_owned());
    }
    Ok(identities)
}

fn expected_directory_links(
    expected: &BTreeMap<PathBuf, CleanupNodeIdentity>,
    current: &BTreeMap<PathBuf, CleanupNodeIdentity>,
    directory: &Path,
) -> Result<u64, String> {
    let original = expected
        .get(directory)
        .ok_or_else(|| format!("cleanup receipt omitted {}", directory.display()))?
        .links;
    let missing_entries = expected
        .keys()
        .filter(|path| path.parent() == Some(directory) && !current.contains_key(*path))
        .count() as u64;
    original.checked_sub(missing_entries).ok_or_else(|| {
        format!(
            "cleanup receipt has impossible directory link accounting at {}",
            directory.display()
        )
    })
}

fn validate_cleanup_receipt_fd(
    root: &File,
    allow_partial: bool,
) -> Result<
    (
        BTreeMap<PathBuf, CleanupNodeIdentity>,
        File,
        CleanupNodeIdentity,
    ),
    String,
> {
    let receipt_name = OsStr::new(CLEANUP_RECEIPT_FILE);
    let (receipt, receipt_file) = read_cleanup_receipt_at(root, receipt_name)?;
    let receipt_identity = cleanup_identity(
        &receipt_file
            .metadata()
            .map_err(|error| format!("stat cleanup receipt: {error}"))?,
    )?;
    validate_cleanup_file(&receipt_file, receipt_identity, CLEANUP_RECEIPT_FILE)?;
    let expected = cleanup_receipt_identity_map(&receipt)?;
    let mut current = collect_cleanup_identity_map(root)?;
    let collected_receipt = current
        .remove(Path::new(CLEANUP_RECEIPT_FILE))
        .ok_or_else(|| "cleanup receipt disappeared during validation".to_owned())?;
    if collected_receipt != receipt_identity {
        return Err("cleanup receipt changed during validation".to_owned());
    }
    if !allow_partial && current.keys().ne(expected.keys()) {
        return Err("private snapshot cleanup membership changed before cleanup".to_owned());
    }
    for (path, actual) in &current {
        let wanted = expected.get(path).ok_or_else(|| {
            format!(
                "private snapshot contains unexpected entry not authorized by cleanup receipt: {}",
                path.display()
            )
        })?;
        if actual.kind != wanted.kind
            || actual.device != wanted.device
            || actual.inode != wanted.inode
            || (actual.kind == SnapshotNodeKind::File && actual.links != wanted.links)
        {
            return Err(format!(
                "private snapshot cleanup identity changed at {}: expected {wanted:?}, actual {actual:?}",
                path.display()
            ));
        }
    }
    for (path, actual) in &current {
        if actual.kind == SnapshotNodeKind::Directory {
            let wanted_links = expected_directory_links(&expected, &current, path)?;
            if actual.links != wanted_links {
                return Err(format!(
                    "private snapshot directory link identity changed at {}: expected {wanted_links}, actual {}",
                    path.display(),
                    actual.links
                ));
            }
        }
    }
    Ok((current, receipt_file, receipt_identity))
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum CleanupMutation {
    UnsealFlags,
    UnsealMode,
    RemoveEntry,
}

fn expected_child_names(
    identities: &BTreeMap<PathBuf, CleanupNodeIdentity>,
    parent: &Path,
) -> Vec<OsString> {
    let mut names = identities
        .keys()
        .filter_map(|relative| {
            (relative.parent() == Some(parent))
                .then(|| relative.file_name().map(OsStr::to_os_string))
                .flatten()
        })
        .collect::<Vec<_>>();
    names.sort_by(|left, right| left.as_bytes().cmp(right.as_bytes()));
    names
}

fn set_open_file_mode(file: &File, mode: u32, label: &Path) -> Result<(), String> {
    // SAFETY: file is an owned live descriptor and fchmod mutates only that opened inode.
    if unsafe { libc::fchmod(file.as_raw_fd(), mode as libc::mode_t) } != 0 {
        Err(format!(
            "set snapshot mode {mode:o} on {}: {}",
            label.display(),
            io::Error::last_os_error()
        ))
    } else {
        Ok(())
    }
}

fn verify_unsealed_tree_fd(
    root: &File,
    identities: &BTreeMap<PathBuf, CleanupNodeIdentity>,
) -> Result<(), String> {
    for (relative, expected) in identities {
        let file = open_relative_from_root(root, relative, expected.kind)?;
        let metadata = validate_cleanup_file(&file, *expected, &relative.display().to_string())?;
        if MacMetadataExt::st_flags(&metadata) & libc::UF_IMMUTABLE != 0 {
            return Err(format!(
                "unfinished private snapshot remains immutable at {}",
                relative.display()
            ));
        }
    }
    Ok(())
}

fn open_relative_from_root(
    root: &File,
    relative: &Path,
    kind: SnapshotNodeKind,
) -> Result<File, String> {
    if relative.as_os_str().is_empty() {
        return duplicate_cloexec(root);
    }
    let mut current = duplicate_cloexec(root)?;
    let components = relative.components().collect::<Vec<_>>();
    for (index, component) in components.iter().enumerate() {
        let name = component.as_os_str();
        let directory = index + 1 != components.len() || kind == SnapshotNodeKind::Directory;
        current = open_entry_at(&current, name, directory)?;
    }
    Ok(current)
}

struct OpenedCleanupChild {
    name: OsString,
    relative: PathBuf,
    initial: CleanupNodeIdentity,
    file: File,
}

struct CleanupBinding<'a> {
    file: &'a File,
    parent: &'a File,
    name: &'a OsStr,
    relative: &'a Path,
    identity: CleanupNodeIdentity,
    links: Cell<u64>,
    ancestor: Option<&'a CleanupBinding<'a>>,
}

impl CleanupBinding<'_> {
    fn expected(&self) -> CleanupNodeIdentity {
        CleanupNodeIdentity {
            links: self.links.get(),
            ..self.identity
        }
    }

    fn validate_chain(&self) -> Result<(), String> {
        if let Some(ancestor) = self.ancestor {
            ancestor.validate_chain()?;
        }
        validate_open_and_entry(
            self.file,
            self.parent,
            self.name,
            self.expected(),
            self.relative,
        )
    }

    fn record_child_unlink(&self) -> Result<(), String> {
        let remaining = self.links.get().checked_sub(1).ok_or_else(|| {
            format!(
                "snapshot directory link count underflow at {}",
                self.relative.display()
            )
        })?;
        self.links.set(remaining);
        self.validate_chain()
    }
}

fn validate_open_and_entry(
    file: &File,
    parent: &File,
    name: &OsStr,
    expected: CleanupNodeIdentity,
    label: &Path,
) -> Result<(), String> {
    validate_cleanup_file(file, expected, &label.display().to_string())?;
    let linked = identity_at(parent, name)?;
    if linked != expected {
        return Err(format!(
            "snapshot cleanup path rebound at {}: expected {expected:?}, actual {linked:?}",
            label.display()
        ));
    }
    Ok(())
}

fn cleanup_directory<F>(
    binding: &CleanupBinding<'_>,
    relative: &Path,
    current: &BTreeMap<PathBuf, CleanupNodeIdentity>,
    preserve_root_authority: bool,
    hook: &mut F,
) -> Result<(), String>
where
    F: FnMut(&Path, CleanupMutation) -> Result<(), String>,
{
    let directory = binding.file;
    let initial = *current
        .get(relative)
        .ok_or_else(|| format!("cleanup identity map omitted {}", relative.display()))?;
    if binding.expected() != initial {
        return Err(format!(
            "cleanup binding disagrees with receipt at {}",
            relative.display()
        ));
    }
    binding.validate_chain()?;

    let mut expected_names = expected_child_names(current, relative);
    if preserve_root_authority && relative.as_os_str().is_empty() {
        expected_names.push(OsString::from(CLEANUP_RECEIPT_FILE));
        expected_names.sort_by(|left, right| left.as_bytes().cmp(right.as_bytes()));
    }
    let actual_names = directory_entry_names(directory)?;
    if actual_names != expected_names {
        return Err(format!(
            "snapshot directory membership changed before cleanup at {}: expected {expected_names:?}, actual {actual_names:?}",
            relative.display(),
        ));
    }

    let mut children = Vec::new();
    for name in expected_child_names(current, relative) {
        let child_relative = relative.join(&name);
        let child_initial = *current
            .get(&child_relative)
            .expect("current cleanup membership has an identity");
        let child = open_entry_at(
            directory,
            &name,
            child_initial.kind == SnapshotNodeKind::Directory,
        )?;
        validate_open_and_entry(&child, directory, &name, child_initial, &child_relative)?;
        children.push(OpenedCleanupChild {
            name,
            relative: child_relative,
            initial: child_initial,
            file: child,
        });
    }

    for child in children
        .iter()
        .filter(|child| child.initial.kind == SnapshotNodeKind::Directory)
    {
        let child_binding = CleanupBinding {
            file: &child.file,
            parent: directory,
            name: &child.name,
            relative: &child.relative,
            identity: child.initial,
            links: Cell::new(child.initial.links),
            ancestor: Some(binding),
        };
        cleanup_directory(
            &child_binding,
            &child.relative,
            current,
            preserve_root_authority,
            hook,
        )?;
    }

    if directory_entry_names(directory)? != expected_names {
        return Err(format!(
            "snapshot directory membership rebound before unsealing at {}",
            relative.display()
        ));
    }
    hook(relative, CleanupMutation::UnsealFlags)?;
    if directory_entry_names(directory)? != expected_names {
        return Err(format!(
            "snapshot directory membership rebound immediately before flag mutation at {}",
            relative.display()
        ));
    }
    binding.validate_chain()?;
    set_open_file_immutable(directory, relative, false)?;
    hook(relative, CleanupMutation::UnsealMode)?;
    if directory_entry_names(directory)? != expected_names {
        return Err(format!(
            "snapshot directory membership rebound immediately before mode mutation at {}",
            relative.display()
        ));
    }
    binding.validate_chain()?;
    set_open_file_mode(directory, 0o700, relative)?;

    for child in children
        .iter()
        .filter(|child| child.initial.kind == SnapshotNodeKind::File)
    {
        let file_binding = CleanupBinding {
            file: &child.file,
            parent: directory,
            name: &child.name,
            relative: &child.relative,
            identity: child.initial,
            links: Cell::new(child.initial.links),
            ancestor: Some(binding),
        };
        hook(&child.relative, CleanupMutation::UnsealFlags)?;
        file_binding.validate_chain()?;
        set_open_file_immutable(&child.file, &child.relative, false)?;
        hook(&child.relative, CleanupMutation::UnsealMode)?;
        file_binding.validate_chain()?;
        set_open_file_mode(&child.file, 0o600, &child.relative)?;
        if preserve_root_authority
            && relative.as_os_str().is_empty()
            && child.relative == Path::new(OWNER_MARKER_FILE)
        {
            // The external sidecar binds this marker, and receipt-present recovery still uses it
            // to authenticate the root. Keep the now-unsealed marker until the cleanup receipt is
            // durably retired. A restart in the following state can then authenticate the marker
            // through the sidecar and finish the ordinary receipt-free cleanup path.
            continue;
        }
        hook(&child.relative, CleanupMutation::RemoveEntry)?;
        file_binding.validate_chain()?;
        unlink_entry_at(directory, &child.name, child.initial)?;
        binding.record_child_unlink()?;
    }

    for child in children
        .iter()
        .filter(|child| child.initial.kind == SnapshotNodeKind::Directory)
    {
        let direct_entries = current
            .keys()
            .filter(|path| path.parent() == Some(child.relative.as_path()))
            .count() as u64;
        let terminal = CleanupNodeIdentity {
            links: child
                .initial
                .links
                .checked_sub(direct_entries)
                .ok_or_else(|| {
                    format!(
                        "cleanup identity has impossible link accounting at {}",
                        child.relative.display()
                    )
                })?,
            ..child.initial
        };
        if !directory_entry_names(&child.file)?.is_empty() {
            return Err(format!(
                "snapshot child directory is not empty at {}",
                child.relative.display()
            ));
        }
        let child_binding = CleanupBinding {
            file: &child.file,
            parent: directory,
            name: &child.name,
            relative: &child.relative,
            identity: terminal,
            links: Cell::new(terminal.links),
            ancestor: Some(binding),
        };
        hook(&child.relative, CleanupMutation::RemoveEntry)?;
        child_binding.validate_chain()?;
        unlink_entry_at(directory, &child.name, terminal)?;
        binding.record_child_unlink()?;
    }

    let remaining = directory_entry_names(directory)?;
    let wanted = if preserve_root_authority && relative.as_os_str().is_empty() {
        let mut authority = vec![
            OsString::from(CLEANUP_RECEIPT_FILE),
            OsString::from(OWNER_MARKER_FILE),
        ];
        authority.sort_by(|left, right| left.as_bytes().cmp(right.as_bytes()));
        authority
    } else {
        Vec::new()
    };
    if remaining != wanted {
        return Err(format!(
            "snapshot directory has unexpected entries after cleanup at {}",
            relative.display()
        ));
    }
    directory
        .sync_all()
        .map_err(|error| format!("sync cleaned snapshot directory: {error}"))
}

#[cfg(test)]
fn validate_cleanup_receipt(root: &Path) -> Result<(), String> {
    let root = open_directory(root)?;
    validate_cleanup_receipt_fd(&root, false)?;
    Ok(())
}

fn remove_owned_snapshot_tree_at<F>(
    namespace: &File,
    lease: &SnapshotLease,
    allow_partial: bool,
    hook: &mut F,
) -> Result<(), String>
where
    F: FnMut(&Path, CleanupMutation) -> Result<(), String>,
{
    let root_name = OsStr::new(&lease.root_name);
    let root = open_entry_at(namespace, root_name, true)?;
    let root_authority = validate_lease_root(lease, &root)?;
    let root_relative = Path::new("");
    let root_binding = CleanupBinding {
        file: &root,
        parent: namespace,
        name: root_name,
        relative: Path::new(&lease.root_name),
        identity: root_authority,
        links: Cell::new(root_authority.links),
        ancestor: None,
    };
    let receipt_name = OsStr::new(CLEANUP_RECEIPT_FILE);
    if identity_at_optional(&root, receipt_name)?.is_some() {
        validate_snapshot_ownership_fd(&root, lease)?;
        let (current, receipt_file, receipt_identity) =
            validate_cleanup_receipt_fd(&root, allow_partial)?;
        cleanup_directory(&root_binding, root_relative, &current, true, hook)?;
        let receipt_relative = Path::new(CLEANUP_RECEIPT_FILE);
        let receipt_binding = CleanupBinding {
            file: &receipt_file,
            parent: &root,
            name: receipt_name,
            relative: receipt_relative,
            identity: receipt_identity,
            links: Cell::new(receipt_identity.links),
            ancestor: Some(&root_binding),
        };
        hook(receipt_relative, CleanupMutation::UnsealFlags)?;
        receipt_binding.validate_chain()?;
        set_open_file_immutable(&receipt_file, receipt_relative, false)?;
        hook(receipt_relative, CleanupMutation::UnsealMode)?;
        receipt_binding.validate_chain()?;
        set_open_file_mode(&receipt_file, 0o600, receipt_relative)?;
        hook(receipt_relative, CleanupMutation::RemoveEntry)?;
        receipt_binding.validate_chain()?;
        unlink_entry_at(&root, receipt_name, receipt_identity)?;
        root_binding.record_child_unlink()?;
        root.sync_all()
            .map_err(|error| format!("sync retired snapshot cleanup receipt: {error}"))?;
    }
    // Receipt-free trees are unfinished construction state, or the terminal cleanup state after
    // the receipt was durably retired. The receipt-present pass deliberately preserves and
    // unseals the owner marker, so a crash immediately after receipt retirement still reaches
    // this authenticated, resumable state instead of becoming permanently quarantined. Root
    // dev+ino from the external lease remains mandatory, and sealed content is never dynamically
    // authorized for mutation.
    let current = collect_cleanup_identity_map(&root)?;
    if current.len() != 1 {
        validate_snapshot_ownership_fd(&root, lease)?;
    }
    verify_unsealed_tree_fd(&root, &current)?;
    cleanup_directory(&root_binding, root_relative, &current, false, hook)?;
    if !directory_entry_names(&root)?.is_empty() {
        return Err("private snapshot root is not empty after descriptor cleanup".to_owned());
    }
    hook(root_relative, CleanupMutation::RemoveEntry)?;
    root_binding.validate_chain()?;
    unlink_entry_at(namespace, root_name, root_binding.expected())
}

fn remove_locked_sidecar_at(namespace: &File, file: &File, name: &OsStr) -> Result<(), String> {
    let metadata = validate_owned_regular_file(file, Path::new(name))?;
    let identity = cleanup_identity(&metadata)?;
    validate_open_and_entry(file, namespace, name, identity, Path::new(name))?;
    unlink_entry_at(namespace, name, identity)
}

fn recover_stale_sidecar(
    namespace: &Path,
    namespace_directory: &File,
    path: &Path,
    staged: bool,
) -> Result<bool, String> {
    let sidecar_name = path
        .file_name()
        .and_then(OsStr::to_str)
        .ok_or_else(|| "snapshot ownership sidecar name is not UTF-8".to_owned())?;
    let token = if staged {
        stage_token(sidecar_name)
    } else {
        sidecar_token(sidecar_name)
    }
    .ok_or_else(|| format!("unsafe snapshot ownership sidecar name {sidecar_name:?}"))?;
    let mut file = OpenOptions::new()
        .read(true)
        .write(true)
        .custom_flags(libc::O_NOFOLLOW)
        .open(path)
        .map_err(|error| format!("open stale snapshot ownership {}: {error}", path.display()))?;
    let sidecar_metadata = validate_owned_regular_file(&file, path)?;
    match lock_exclusive(&file, true)? {
        LockDisposition::Busy => return Ok(false),
        LockDisposition::Acquired => {}
    }
    let lease = match read_json_from_open_file::<SnapshotLease>(&mut file, path) {
        Ok(lease) => lease,
        Err(lease_error) if staged => {
            let root_name = expected_root_name(token);
            if identity_at_optional(namespace_directory, OsStr::new(&root_name))?.is_some() {
                let expected_root = match read_json_from_open_file::<SnapshotStageIntent>(
                    &mut file, path,
                ) {
                    Ok(intent) => {
                        validate_stage_intent(&intent, token, &sidecar_metadata)?;
                        unpublished_root_identity_at(
                            namespace_directory,
                            OsStr::new(&root_name),
                        )?
                    }
                    Err(intent_error) => validate_unpublished_marker_authority(
                        namespace_directory,
                        OsStr::new(&root_name),
                        token,
                        &sidecar_metadata,
                    )
                    .map_err(|marker_error| {
                        format!(
                            "{lease_error}; {intent_error}; {marker_error}; malformed staged authority has a candidate root and was quarantined: {}",
                            namespace.join(&root_name).display()
                        )
                    })?,
                };
                remove_unpublished_root_at(
                    namespace_directory,
                    OsStr::new(&root_name),
                    expected_root,
                )?;
            }
            remove_locked_sidecar_at(namespace_directory, &file, OsStr::new(sidecar_name))?;
            namespace_directory
                .sync_all()
                .map_err(|error| format!("sync removed staging authority: {error}"))?;
            return Ok(true);
        }
        Err(error) => return Err(error),
    };
    validate_lease(&lease, token, &sidecar_metadata)?;
    if identity_at_optional(namespace_directory, OsStr::new(&lease.root_name))?.is_some() {
        remove_owned_snapshot_tree_at(namespace_directory, &lease, true, &mut |_, _| Ok(()))?;
    }
    remove_locked_sidecar_at(namespace_directory, &file, OsStr::new(sidecar_name))?;
    namespace_directory
        .sync_all()
        .map_err(|error| format!("sync recovered snapshot namespace: {error}"))?;
    Ok(true)
}

fn scavenge_stale_snapshots_in(parent: Option<&Path>) -> Result<usize, String> {
    let namespace = snapshot_namespace(parent)?;
    let namespace_lock = open_namespace_lock(&namespace)?;
    match lock_exclusive(&namespace_lock, false)? {
        LockDisposition::Acquired => {}
        LockDisposition::Busy => unreachable!("blocking namespace lock cannot be busy"),
    }
    let namespace_directory = open_directory(&namespace)?;
    let mut recovered = 0usize;
    let mut errors = Vec::new();
    let mut referenced_roots = BTreeSet::new();
    for path in sorted_entries(&namespace)? {
        let Some(name) = path.file_name().and_then(OsStr::to_str) else {
            errors.push(format!(
                "private snapshot namespace contains a non-UTF-8 entry: {}",
                path.display()
            ));
            continue;
        };
        let token = sidecar_token(name).or_else(|| stage_token(name));
        if let Some(token) = token {
            referenced_roots.insert(expected_root_name(token));
            match recover_stale_sidecar(
                &namespace,
                &namespace_directory,
                &path,
                stage_token(name).is_some(),
            ) {
                Ok(true) => recovered += 1,
                Ok(false) => {}
                Err(error) => errors.push(error),
            }
        }
    }
    for path in sorted_entries(&namespace)? {
        let Some(name) = path.file_name().and_then(OsStr::to_str) else {
            continue;
        };
        if name.starts_with(SNAPSHOT_ROOT_PREFIX) && !referenced_roots.contains(name) {
            errors.push(format!(
                "orphan private snapshot has no trustworthy ownership sidecar; left untouched: {}",
                path.display()
            ));
        }
    }
    if errors.is_empty() {
        Ok(recovered)
    } else {
        Err(format!(
            "stale snapshot recovery refused unsafe ownership state: {}",
            errors.join("; ")
        ))
    }
}

pub(super) fn scavenge_stale_snapshots() -> Result<usize, String> {
    scavenge_stale_snapshots_in(None)
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

            // Publish an exact cleanup identity receipt before sealing begins. A terminated parent
            // can then distinguish an unsealed construction tree from a partially or fully sealed
            // tree and never needs to clear flags on an unrelated inode.
            owner.prepare_for_sealing()?;
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

    pub(super) fn configure_child_lease(&self, command: &mut Command) -> Result<(), String> {
        self.owner.configure_child_lease(command)
    }

    #[cfg(test)]
    fn inherit_child_lease_for_test(&self, command: &mut Command) -> Result<(), String> {
        self.owner.inherit_child_lease(command)
    }

    #[cfg(test)]
    pub(super) fn duplicate_lease_fd(&self) -> Result<RawFd, String> {
        self.owner.duplicate_lease_fd()
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
    _lease: File,
    root_path: PathBuf,
    artifact_path: PathBuf,
    identities: BTreeMap<PathBuf, SnapshotNodeIdentity>,
    receipt: ArtifactSnapshotReceipt,
}

impl VerifiedPinnedArtifact {
    pub(super) fn admit(
        artifact: &ArtifactReceipt,
        path: &Path,
        lease_fd: RawFd,
    ) -> Result<Self, String> {
        if lease_fd < 0 {
            return Err("child artifact lease descriptor must be non-negative".to_owned());
        }
        // SAFETY: F_GETFD only inspects the numeric descriptor and does not take ownership.
        if unsafe { libc::fcntl(lease_fd, libc::F_GETFD) } < 0 {
            return Err(format!(
                "child artifact lease descriptor is not open: {}",
                io::Error::last_os_error()
            ));
        }
        // SAFETY: the child CLI transfers ownership of its explicitly inherited descriptor to the
        // verified snapshot guard. Callers must pass a live descriptor exactly once.
        let mut lease_file = unsafe { File::from_raw_fd(lease_fd) };
        set_cloexec(&lease_file, "child artifact lease")?;
        let lease_metadata =
            validate_owned_regular_file(&lease_file, Path::new("<inherited snapshot lease>"))?;
        match lock_exclusive(&lease_file, true)? {
            LockDisposition::Acquired => {}
            LockDisposition::Busy => {
                return Err(
                    "child snapshot lease descriptor is not the inherited locked authority"
                        .to_owned(),
                )
            }
        }
        let lease: SnapshotLease =
            read_json_from_open_file(&mut lease_file, Path::new("<inherited snapshot lease>"))?;
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
        let root_name = root_path
            .file_name()
            .and_then(OsStr::to_str)
            .ok_or_else(|| "private artifact snapshot root name is not UTF-8".to_owned())?;
        let token = root_name
            .strip_prefix(SNAPSHOT_ROOT_PREFIX)
            .filter(|token| {
                !token.is_empty() && token.bytes().all(|byte| byte.is_ascii_alphanumeric())
            })
            .ok_or_else(|| "private artifact snapshot root has an unsafe name".to_owned())?;
        validate_lease(&lease, token, &lease_metadata)?;
        if lease.root_name != root_name {
            return Err("child snapshot path does not match its inherited lease".to_owned());
        }
        let namespace_path = root_path
            .parent()
            .ok_or_else(|| "private artifact snapshot root has no namespace".to_owned())?;
        let namespace = open_directory(namespace_path)?;
        let root = open_entry_at(&namespace, OsStr::new(root_name), true)?;
        validate_snapshot_ownership_fd(&root, &lease)?;
        verify_sealed_tree(&root_path)?;
        let inventory = inventory_artifact(&artifact_path).map_err(|error| error.to_string())?;
        if inventory != artifact.inventory {
            return Err(format!(
                "private snapshot for artifact {} does not match the frozen inventory",
                artifact.key
            ));
        }
        let snapshot = Self {
            _lease: lease_file,
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

#[cfg(test)]
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
    use std::process::{Command, Stdio};
    use std::thread;
    use std::time::Duration;

    const TERMINATION_NAMESPACE_ENV: &str = "SCENEWORKS_P6_TERMINATION_NAMESPACE";
    const TERMINATION_ARTIFACT_ENV: &str = "SCENEWORKS_P6_TERMINATION_ARTIFACT";
    const TERMINATION_READY_ENV: &str = "SCENEWORKS_P6_TERMINATION_READY";
    const PUBLICATION_HELPER_ENV: &str = "SCENEWORKS_P6_PUBLICATION_HELPER";
    const PUBLICATION_PHASE_ENV: &str = "SCENEWORKS_P6_PUBLICATION_PAUSE_PHASE";
    const PUBLICATION_READY_ENV: &str = "SCENEWORKS_P6_PUBLICATION_PAUSE_READY";
    const CHILD_LEASE_PARENT_ENV: &str = "SCENEWORKS_P6_CHILD_LEASE_PARENT";
    const CHILD_LEASE_CHILD_ENV: &str = "SCENEWORKS_P6_CHILD_LEASE_CHILD";
    const CHILD_LEASE_NAMESPACE_ENV: &str = "SCENEWORKS_P6_CHILD_LEASE_NAMESPACE";
    const CHILD_LEASE_ARTIFACT_ENV: &str = "SCENEWORKS_P6_CHILD_LEASE_ARTIFACT";
    const CHILD_LEASE_SNAPSHOT_ENV: &str = "SCENEWORKS_P6_CHILD_LEASE_SNAPSHOT";
    const CHILD_LEASE_READY_ENV: &str = "SCENEWORKS_P6_CHILD_LEASE_READY";

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

    fn assert_no_owned_snapshots(namespace: &Path) {
        let entries = fs::read_dir(namespace)
            .unwrap()
            .map(|entry| entry.unwrap().file_name().into_string().unwrap())
            .filter(|name| {
                name.starts_with(OWNER_SIDECAR_PREFIX)
                    || name.starts_with(SNAPSHOT_ROOT_PREFIX)
                    || name.starts_with(OWNER_STAGE_PREFIX)
            })
            .collect::<Vec<_>>();
        assert!(
            entries.is_empty(),
            "leaked owned snapshot entries: {entries:?}"
        );
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
        assert_no_owned_snapshots(snapshots.path());
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
    fn released_crash_lease_is_scavenged_but_a_live_lease_is_not() {
        let source = tempfile::tempdir().unwrap();
        let artifact_path = source.path().join("artifact");
        fs::create_dir(&artifact_path).unwrap();
        fs::write(artifact_path.join("weights.bin"), b"weights").unwrap();
        let artifact = receipt(&artifact_path);
        let snapshots = tempfile::tempdir().unwrap();

        let mut live = OwnedPinnedArtifact::create_with_hooks(
            &artifact,
            Some(snapshots.path()),
            ClonePreference::CopyOnly,
            || Ok(()),
            |_| Ok(()),
        )
        .unwrap();
        let live_root = live.root_path().to_path_buf();
        assert_eq!(
            scavenge_stale_snapshots_in(Some(snapshots.path())).unwrap(),
            0
        );
        assert!(live_root.exists());
        live.cleanup().unwrap();

        let crashed = OwnedPinnedArtifact::create_with_hooks(
            &artifact,
            Some(snapshots.path()),
            ClonePreference::CopyOnly,
            || Ok(()),
            |_| Ok(()),
        )
        .unwrap();
        let OwnedPinnedArtifact { owner, .. } = crashed;
        let stale_root = owner.abandon_for_termination_test();
        assert!(stale_root.exists());
        assert_eq!(
            scavenge_stale_snapshots_in(Some(snapshots.path())).unwrap(),
            1
        );
        assert!(!stale_root.exists());
        assert_no_owned_snapshots(snapshots.path());
    }

    #[test]
    fn startup_scavenger_finishes_removal_interrupted_after_unsealing() {
        let source = tempfile::tempdir().unwrap();
        let artifact_path = source.path().join("artifact");
        fs::create_dir(&artifact_path).unwrap();
        fs::write(artifact_path.join("weights.bin"), b"weights").unwrap();
        let artifact = receipt(&artifact_path);
        let snapshots = tempfile::tempdir().unwrap();
        let pinned = OwnedPinnedArtifact::create_with_hooks(
            &artifact,
            Some(snapshots.path()),
            ClonePreference::CopyOnly,
            || Ok(()),
            |_| Ok(()),
        )
        .unwrap();
        let OwnedPinnedArtifact { owner, .. } = pinned;
        let stale_root = owner.abandon_for_termination_test();

        validate_cleanup_receipt(&stale_root).unwrap();
        unseal_tree(&stale_root).unwrap();
        fs::remove_file(stale_root.join(CLEANUP_RECEIPT_FILE)).unwrap();
        sync_directory(&stale_root).unwrap();
        fs::remove_file(stale_root.join("artifact/weights.bin")).unwrap();

        assert_eq!(
            scavenge_stale_snapshots_in(Some(snapshots.path())).unwrap(),
            1
        );
        assert!(!stale_root.exists());
        assert_no_owned_snapshots(snapshots.path());
    }

    #[test]
    fn termination_snapshot_helper() {
        let Ok(namespace) = std::env::var(TERMINATION_NAMESPACE_ENV) else {
            return;
        };
        let artifact = std::env::var(TERMINATION_ARTIFACT_ENV).unwrap();
        let ready = std::env::var(TERMINATION_READY_ENV).unwrap();
        let artifact = receipt(Path::new(&artifact));
        let pinned = OwnedPinnedArtifact::create_with_hooks(
            &artifact,
            Some(Path::new(&namespace)),
            ClonePreference::CopyOnly,
            || Ok(()),
            |_| Ok(()),
        )
        .unwrap();
        fs::write(ready, pinned.root_path().as_os_str().as_bytes()).unwrap();
        loop {
            thread::park_timeout(Duration::from_secs(1));
        }
    }

    #[test]
    fn publication_interruption_helper() {
        if std::env::var_os(PUBLICATION_HELPER_ENV).is_none() {
            return;
        }
        let namespace = std::env::var(CHILD_LEASE_NAMESPACE_ENV).unwrap();
        let artifact = std::env::var(CHILD_LEASE_ARTIFACT_ENV).unwrap();
        let artifact = receipt(Path::new(&artifact));
        let _pinned = OwnedPinnedArtifact::create_with_hooks(
            &artifact,
            Some(Path::new(&namespace)),
            ClonePreference::CopyOnly,
            || Ok(()),
            |_| Ok(()),
        )
        .unwrap();
        panic!("publication helper passed its requested pause point");
    }

    #[test]
    fn child_lease_grandchild_helper() {
        if std::env::var_os(CHILD_LEASE_CHILD_ENV).is_none() {
            return;
        }
        let artifact_path = std::env::var(CHILD_LEASE_ARTIFACT_ENV).unwrap();
        let snapshot_path = std::env::var(CHILD_LEASE_SNAPSHOT_ENV).unwrap();
        let ready = std::env::var(CHILD_LEASE_READY_ENV).unwrap();
        let artifact = receipt(Path::new(&artifact_path));
        let _verified =
            VerifiedPinnedArtifact::admit(&artifact, Path::new(&snapshot_path), CHILD_LEASE_FD)
                .unwrap();
        fs::write(ready, std::process::id().to_string()).unwrap();
        loop {
            thread::park_timeout(Duration::from_secs(1));
        }
    }

    #[test]
    #[allow(clippy::zombie_processes)]
    fn child_lease_parent_helper() {
        if std::env::var_os(CHILD_LEASE_PARENT_ENV).is_none() {
            return;
        }
        let namespace = std::env::var(CHILD_LEASE_NAMESPACE_ENV).unwrap();
        let artifact_path = std::env::var(CHILD_LEASE_ARTIFACT_ENV).unwrap();
        let ready = PathBuf::from(std::env::var(CHILD_LEASE_READY_ENV).unwrap());
        let child_ready = ready.with_extension("child");
        let artifact = receipt(Path::new(&artifact_path));
        let pinned = OwnedPinnedArtifact::create_with_hooks(
            &artifact,
            Some(Path::new(&namespace)),
            ClonePreference::CopyOnly,
            || Ok(()),
            |_| Ok(()),
        )
        .unwrap();
        let test_binary = std::env::current_exe().unwrap();
        let mut command = Command::new(test_binary);
        command
            .arg("--exact")
            .arg("pinned_artifact::tests::child_lease_grandchild_helper")
            .arg("--nocapture")
            .env(CHILD_LEASE_CHILD_ENV, "1")
            .env(CHILD_LEASE_ARTIFACT_ENV, &artifact_path)
            .env(CHILD_LEASE_SNAPSHOT_ENV, pinned.path())
            .env(CHILD_LEASE_READY_ENV, &child_ready)
            .stdin(Stdio::null())
            .stdout(Stdio::null())
            .stderr(Stdio::null());
        pinned.inherit_child_lease_for_test(&mut command).unwrap();
        let _child = command.spawn().unwrap();
        for _ in 0..500 {
            if child_ready.exists() {
                break;
            }
            thread::sleep(Duration::from_millis(10));
        }
        let child_pid = fs::read_to_string(&child_ready).unwrap();
        fs::write(
            &ready,
            format!("{}\n{}\n", pinned.root_path().display(), child_pid.trim()),
        )
        .unwrap();
        loop {
            thread::park_timeout(Duration::from_secs(1));
        }
    }

    #[test]
    fn inherited_child_lease_survives_parent_death_and_blocks_scavenging() {
        let temp = tempfile::tempdir().unwrap();
        let artifact_path = temp.path().join("artifact");
        let namespace = temp.path().join("snapshots");
        let ready = temp.path().join("lease-ready");
        fs::create_dir(&artifact_path).unwrap();
        fs::create_dir(&namespace).unwrap();
        fs::write(artifact_path.join("weights.bin"), b"weights").unwrap();
        let test_binary = std::env::current_exe().unwrap();
        let mut parent = Command::new(test_binary)
            .arg("--exact")
            .arg("pinned_artifact::tests::child_lease_parent_helper")
            .arg("--nocapture")
            .env(CHILD_LEASE_PARENT_ENV, "1")
            .env(CHILD_LEASE_NAMESPACE_ENV, &namespace)
            .env(CHILD_LEASE_ARTIFACT_ENV, &artifact_path)
            .env(CHILD_LEASE_READY_ENV, &ready)
            .stdin(Stdio::null())
            .stdout(Stdio::null())
            .stderr(Stdio::null())
            .spawn()
            .unwrap();
        for _ in 0..500 {
            if ready.exists() {
                break;
            }
            thread::sleep(Duration::from_millis(10));
        }
        let state = fs::read_to_string(&ready).expect("lease parent did not become ready");
        let mut lines = state.lines();
        let root = PathBuf::from(lines.next().unwrap());
        let child_pid = lines.next().unwrap().parse::<libc::pid_t>().unwrap();
        assert_eq!(
            unsafe { libc::kill(parent.id() as libc::pid_t, libc::SIGKILL) },
            0
        );
        assert!(!parent.wait().unwrap().success());
        assert_eq!(scavenge_stale_snapshots_in(Some(&namespace)).unwrap(), 0);
        assert!(root.exists());
        assert_eq!(unsafe { libc::kill(child_pid, libc::SIGKILL) }, 0);
        let mut recovered = 0;
        for _ in 0..500 {
            recovered = scavenge_stale_snapshots_in(Some(&namespace)).unwrap();
            if recovered == 1 {
                break;
            }
            thread::sleep(Duration::from_millis(10));
        }
        assert_eq!(recovered, 1);
        assert!(!root.exists());
        assert_no_owned_snapshots(&namespace);
    }

    #[test]
    fn staging_authority_interruptions_never_publish_partial_owner_files() {
        let temp = tempfile::tempdir().unwrap();
        let artifact_path = temp.path().join("artifact");
        let namespace = temp.path().join("snapshots");
        fs::create_dir(&artifact_path).unwrap();
        fs::create_dir(&namespace).unwrap();
        fs::write(artifact_path.join("weights.bin"), b"weights").unwrap();
        let test_binary = std::env::current_exe().unwrap();

        for (phase, signal) in [
            ("empty-stage", libc::SIGKILL),
            ("intent-stage", libc::SIGABRT),
            ("root-stage", libc::SIGKILL),
            ("marker-stage", libc::SIGABRT),
            ("truncated-lease-stage", libc::SIGKILL),
            ("complete-stage", libc::SIGABRT),
        ] {
            let ready = temp.path().join(format!("publication-{phase}"));
            let mut child = Command::new(&test_binary)
                .arg("--exact")
                .arg("pinned_artifact::tests::publication_interruption_helper")
                .arg("--nocapture")
                .env(PUBLICATION_HELPER_ENV, "1")
                .env(PUBLICATION_PHASE_ENV, phase)
                .env(PUBLICATION_READY_ENV, &ready)
                .env(CHILD_LEASE_NAMESPACE_ENV, &namespace)
                .env(CHILD_LEASE_ARTIFACT_ENV, &artifact_path)
                .stdin(Stdio::null())
                .stdout(Stdio::null())
                .stderr(Stdio::null())
                .spawn()
                .unwrap();
            for _ in 0..500 {
                if ready.exists() {
                    break;
                }
                thread::sleep(Duration::from_millis(10));
            }
            assert!(ready.exists(), "publication helper did not reach {phase}");
            let names = fs::read_dir(&namespace)
                .unwrap()
                .map(|entry| entry.unwrap().file_name().into_string().unwrap())
                .collect::<Vec<_>>();
            assert!(names
                .iter()
                .any(|name| name.starts_with(OWNER_STAGE_PREFIX)));
            assert!(
                names
                    .iter()
                    .all(|name| !name.starts_with(OWNER_SIDECAR_PREFIX)),
                "a final owner file became visible before atomic publication: {names:?}"
            );
            assert_eq!(unsafe { libc::kill(child.id() as libc::pid_t, signal) }, 0);
            assert!(!child.wait().unwrap().success());
            assert_eq!(scavenge_stale_snapshots_in(Some(&namespace)).unwrap(), 1);
            assert_no_owned_snapshots(&namespace);
        }
    }

    #[test]
    fn startup_scavenger_recovers_sigint_sigterm_sigkill_and_abort_trees() {
        let temp = tempfile::tempdir().unwrap();
        let artifact_path = temp.path().join("artifact");
        let namespace = temp.path().join("snapshots");
        fs::create_dir(&artifact_path).unwrap();
        fs::create_dir(&namespace).unwrap();
        fs::write(artifact_path.join("weights.bin"), b"weights").unwrap();
        let test_binary = std::env::current_exe().unwrap();

        for signal in [libc::SIGINT, libc::SIGTERM, libc::SIGKILL, libc::SIGABRT] {
            let ready = temp.path().join(format!("ready-{signal}"));
            let mut child = Command::new(&test_binary)
                .arg("--exact")
                .arg("pinned_artifact::tests::termination_snapshot_helper")
                .arg("--nocapture")
                .env(TERMINATION_NAMESPACE_ENV, &namespace)
                .env(TERMINATION_ARTIFACT_ENV, &artifact_path)
                .env(TERMINATION_READY_ENV, &ready)
                .stdin(Stdio::null())
                .stdout(Stdio::null())
                .stderr(Stdio::null())
                .spawn()
                .unwrap();
            for _ in 0..500 {
                if ready.exists() {
                    break;
                }
                thread::sleep(Duration::from_millis(10));
            }
            assert!(
                ready.exists(),
                "termination helper did not publish readiness"
            );
            let root = PathBuf::from(OsStr::from_bytes(&fs::read(&ready).unwrap()));
            assert!(root.exists());
            assert_eq!(scavenge_stale_snapshots_in(Some(&namespace)).unwrap(), 0);
            // SIGKILL is also the relevant lock-release model for kernel OOM termination.
            assert_eq!(unsafe { libc::kill(child.id() as libc::pid_t, signal) }, 0);
            assert!(!child.wait().unwrap().success());
            assert_eq!(scavenge_stale_snapshots_in(Some(&namespace)).unwrap(), 1);
            assert!(!root.exists());
        }
        assert_no_owned_snapshots(&namespace);
    }

    #[test]
    fn malformed_and_foreign_ownership_metadata_is_never_scavenged() {
        for foreign in [false, true] {
            let namespace = tempfile::tempdir().unwrap();
            let token = if foreign { "FOREIGN" } else { "MALFORMED" };
            let sidecar = namespace
                .path()
                .join(format!("{OWNER_SIDECAR_PREFIX}{token}"));
            let root = namespace.path().join(expected_root_name(token));
            fs::create_dir(&root).unwrap();
            fs::write(root.join("sentinel"), b"unrelated").unwrap();
            if foreign {
                create_json_file(
                    &sidecar,
                    &SnapshotLease {
                        schema_version: OWNERSHIP_SCHEMA.to_owned(),
                        harness: "foreign-harness".to_owned(),
                        token: token.to_owned(),
                        sidecar_name: expected_sidecar_name(token),
                        sidecar_device: 0,
                        sidecar_inode: 0,
                        root_name: expected_root_name(token),
                        root_device: 0,
                        root_inode: 0,
                    },
                )
                .unwrap();
            } else {
                fs::write(&sidecar, b"{not-json").unwrap();
            }

            let error = scavenge_stale_snapshots_in(Some(namespace.path())).unwrap_err();
            assert!(error.contains("refused unsafe ownership state"), "{error}");
            assert_eq!(fs::read(root.join("sentinel")).unwrap(), b"unrelated");
            assert!(sidecar.exists());
        }
    }

    #[test]
    fn foreign_staging_intent_with_a_candidate_root_is_never_scavenged() {
        let namespace = tempfile::tempdir().unwrap();
        let token = "FOREIGNSTAGE";
        let sidecar = namespace
            .path()
            .join(format!("{OWNER_STAGE_PREFIX}{token}"));
        let root = namespace.path().join(expected_root_name(token));
        fs::create_dir(&root).unwrap();
        fs::write(root.join("sentinel"), b"unrelated").unwrap();
        create_json_file(
            &sidecar,
            &SnapshotStageIntent {
                schema_version: OWNERSHIP_STAGE_SCHEMA.to_owned(),
                harness: "foreign-harness".to_owned(),
                token: token.to_owned(),
                sidecar_name: expected_sidecar_name(token),
                sidecar_device: 0,
                sidecar_inode: 0,
                root_name: expected_root_name(token),
            },
        )
        .unwrap();

        let error = scavenge_stale_snapshots_in(Some(namespace.path())).unwrap_err();
        assert!(
            error.contains("foreign or malformed staged snapshot ownership intent"),
            "{error}"
        );
        assert_eq!(fs::read(root.join("sentinel")).unwrap(), b"unrelated");
        assert!(sidecar.exists());
    }

    #[test]
    fn receipt_free_recovery_refuses_a_replacement_root_bound_to_the_stale_sidecar() {
        let source = tempfile::tempdir().unwrap();
        let artifact_path = source.path().join("artifact");
        fs::create_dir(&artifact_path).unwrap();
        fs::write(artifact_path.join("weights.bin"), b"weights").unwrap();
        let artifact = receipt(&artifact_path);
        let snapshots = tempfile::tempdir().unwrap();
        let pinned = OwnedPinnedArtifact::create_with_hooks(
            &artifact,
            Some(snapshots.path()),
            ClonePreference::CopyOnly,
            || Ok(()),
            |_| Ok(()),
        )
        .unwrap();
        let OwnedPinnedArtifact { owner, .. } = pinned;
        let (root, _lease) = owner.abandon_with_lease_for_test();
        unseal_tree(&root).unwrap();
        fs::remove_file(root.join(CLEANUP_RECEIPT_FILE)).unwrap();
        let displaced = snapshots.path().join("displaced-owned-root");
        fs::rename(&root, &displaced).unwrap();
        fs::create_dir(&root).unwrap();
        fs::write(root.join("sentinel"), b"replacement").unwrap();

        let error = scavenge_stale_snapshots_in(Some(snapshots.path())).unwrap_err();
        assert!(
            error.contains("no longer matches sidecar authority"),
            "{error}"
        );
        assert_eq!(fs::read(root.join("sentinel")).unwrap(), b"replacement");

        fs::remove_dir_all(&root).unwrap();
        fs::rename(&displaced, &root).unwrap();
        assert_eq!(
            scavenge_stale_snapshots_in(Some(snapshots.path())).unwrap(),
            1
        );
        assert_no_owned_snapshots(snapshots.path());
    }

    #[test]
    fn child_admission_refuses_missing_and_independent_lease_descriptors() {
        let source = tempfile::tempdir().unwrap();
        let artifact_path = source.path().join("artifact");
        fs::create_dir(&artifact_path).unwrap();
        fs::write(artifact_path.join("weights.bin"), b"weights").unwrap();
        let artifact = receipt(&artifact_path);
        let mut pinned = OwnedPinnedArtifact::create(&artifact).unwrap();

        let missing = VerifiedPinnedArtifact::admit(&artifact, pinned.path(), -1)
            .err()
            .expect("missing lease descriptor must be refused");
        assert!(missing.contains("non-negative"), "{missing}");

        let unrelated_path = source.path().join("unrelated-lease");
        fs::write(&unrelated_path, b"{}").unwrap();
        let unrelated = OpenOptions::new()
            .read(true)
            .write(true)
            .open(&unrelated_path)
            .unwrap();
        let wrong =
            VerifiedPinnedArtifact::admit(&artifact, pinned.path(), unrelated.into_raw_fd())
                .err()
                .expect("independent lease descriptor must be refused");
        assert!(wrong.contains("parse snapshot ownership"), "{wrong}");
        pinned.cleanup().unwrap();
    }

    #[test]
    fn cleanup_revalidates_nlink_immediately_before_mode_mutation() {
        let source = tempfile::tempdir().unwrap();
        let artifact_path = source.path().join("artifact");
        fs::create_dir(&artifact_path).unwrap();
        fs::write(artifact_path.join("weights.bin"), b"weights").unwrap();
        let artifact = receipt(&artifact_path);
        let snapshots = tempfile::tempdir().unwrap();
        let pinned = OwnedPinnedArtifact::create_with_hooks(
            &artifact,
            Some(snapshots.path()),
            ClonePreference::CopyOnly,
            || Ok(()),
            |_| Ok(()),
        )
        .unwrap();
        let OwnedPinnedArtifact { owner, .. } = pinned;
        let (root, lease) = owner.abandon_with_lease_for_test();
        let namespace = open_directory(snapshots.path()).unwrap();
        let target = Path::new("artifact/weights.bin");
        let outside_link = source.path().join("outside-link.bin");
        let original_mode = fs::metadata(root.join(target))
            .unwrap()
            .permissions()
            .mode();
        let injected = Cell::new(false);
        let error =
            remove_owned_snapshot_tree_at(&namespace, &lease, false, &mut |relative, mutation| {
                if relative == target
                    && mutation == CleanupMutation::UnsealMode
                    && !injected.replace(true)
                {
                    fs::hard_link(root.join(target), &outside_link)
                        .map_err(|error| error.to_string())?;
                }
                Ok(())
            })
            .unwrap_err();
        assert!(
            error.contains("multiply-linked") || error.contains("changed"),
            "{error}"
        );
        assert_eq!(
            fs::metadata(&outside_link).unwrap().permissions().mode(),
            original_mode
        );
        fs::remove_file(&outside_link).unwrap();
        assert_eq!(
            scavenge_stale_snapshots_in(Some(snapshots.path())).unwrap(),
            1
        );
        assert_no_owned_snapshots(snapshots.path());
    }

    #[test]
    fn cleanup_manifest_resumes_with_missing_expected_nodes_and_keeps_receipt_until_last() {
        let source = tempfile::tempdir().unwrap();
        let artifact_path = source.path().join("artifact");
        fs::create_dir(&artifact_path).unwrap();
        fs::write(artifact_path.join("weights.bin"), b"weights").unwrap();
        let artifact = receipt(&artifact_path);
        let snapshots = tempfile::tempdir().unwrap();
        let pinned = OwnedPinnedArtifact::create_with_hooks(
            &artifact,
            Some(snapshots.path()),
            ClonePreference::CopyOnly,
            || Ok(()),
            |_| Ok(()),
        )
        .unwrap();
        let OwnedPinnedArtifact { owner, .. } = pinned;
        let (root, lease) = owner.abandon_with_lease_for_test();
        let namespace = open_directory(snapshots.path()).unwrap();
        let removed_file = Cell::new(false);
        let interruption =
            remove_owned_snapshot_tree_at(&namespace, &lease, false, &mut |relative, mutation| {
                if relative == Path::new("artifact/weights.bin")
                    && mutation == CleanupMutation::RemoveEntry
                {
                    removed_file.set(true);
                } else if removed_file.get() {
                    return Err("injected cleanup interruption".to_owned());
                }
                Ok(())
            })
            .unwrap_err();
        assert!(interruption.contains("injected cleanup interruption"));
        assert!(!root.join("artifact/weights.bin").exists());
        assert!(root.join(CLEANUP_RECEIPT_FILE).exists());

        assert_eq!(
            scavenge_stale_snapshots_in(Some(snapshots.path())).unwrap(),
            1
        );
        assert!(!root.exists());
        assert_no_owned_snapshots(snapshots.path());
    }

    #[test]
    fn cleanup_preserves_owner_marker_until_after_receipt_retirement() {
        let source = tempfile::tempdir().unwrap();
        let artifact_path = source.path().join("artifact");
        fs::create_dir(&artifact_path).unwrap();
        fs::write(artifact_path.join("weights.bin"), b"weights").unwrap();
        let artifact = receipt(&artifact_path);
        let snapshots = tempfile::tempdir().unwrap();
        let pinned = OwnedPinnedArtifact::create_with_hooks(
            &artifact,
            Some(snapshots.path()),
            ClonePreference::CopyOnly,
            || Ok(()),
            |_| Ok(()),
        )
        .unwrap();
        let OwnedPinnedArtifact { owner, .. } = pinned;
        let (root, lease) = owner.abandon_with_lease_for_test();
        let namespace = open_directory(snapshots.path()).unwrap();
        let interruption =
            remove_owned_snapshot_tree_at(&namespace, &lease, false, &mut |relative, mutation| {
                if relative == Path::new(OWNER_MARKER_FILE)
                    && mutation == CleanupMutation::RemoveEntry
                {
                    if root.join(CLEANUP_RECEIPT_FILE).exists() {
                        return Err(
                            "owner marker removal preceded cleanup receipt retirement".to_owned()
                        );
                    }
                    return Err("injected post-receipt marker interruption".to_owned());
                }
                Ok(())
            })
            .unwrap_err();
        assert!(
            interruption.contains("injected post-receipt marker interruption"),
            "{interruption}"
        );
        assert!(root.is_dir());
        assert!(!root.join(CLEANUP_RECEIPT_FILE).exists());
        assert!(root.join(OWNER_MARKER_FILE).is_file());

        assert_eq!(
            scavenge_stale_snapshots_in(Some(snapshots.path())).unwrap(),
            1
        );
        assert!(!root.exists());
        assert_no_owned_snapshots(snapshots.path());
    }

    #[test]
    fn cleanup_resumes_after_receipt_retirement_before_root_unlink() {
        let source = tempfile::tempdir().unwrap();
        let artifact_path = source.path().join("artifact");
        fs::create_dir(&artifact_path).unwrap();
        fs::write(artifact_path.join("weights.bin"), b"weights").unwrap();
        let artifact = receipt(&artifact_path);
        let snapshots = tempfile::tempdir().unwrap();
        let pinned = OwnedPinnedArtifact::create_with_hooks(
            &artifact,
            Some(snapshots.path()),
            ClonePreference::CopyOnly,
            || Ok(()),
            |_| Ok(()),
        )
        .unwrap();
        let OwnedPinnedArtifact { owner, .. } = pinned;
        let (root, lease) = owner.abandon_with_lease_for_test();
        let namespace = open_directory(snapshots.path()).unwrap();
        let receipt_retired = Cell::new(false);
        let interruption =
            remove_owned_snapshot_tree_at(&namespace, &lease, false, &mut |relative, mutation| {
                if relative == Path::new(CLEANUP_RECEIPT_FILE)
                    && mutation == CleanupMutation::RemoveEntry
                {
                    receipt_retired.set(true);
                } else if relative.as_os_str().is_empty()
                    && mutation == CleanupMutation::RemoveEntry
                    && receipt_retired.get()
                {
                    return Err("injected terminal cleanup interruption".to_owned());
                }
                Ok(())
            })
            .unwrap_err();
        assert!(interruption.contains("terminal cleanup interruption"));
        assert!(root.is_dir());
        assert!(fs::read_dir(&root).unwrap().next().is_none());

        assert_eq!(
            scavenge_stale_snapshots_in(Some(snapshots.path())).unwrap(),
            1
        );
        assert!(!root.exists());
        assert_no_owned_snapshots(snapshots.path());
    }

    #[test]
    fn cleanup_refuses_an_intermediate_directory_swap_before_unsealing() {
        let source = tempfile::tempdir().unwrap();
        let artifact_path = source.path().join("artifact");
        fs::create_dir(&artifact_path).unwrap();
        fs::write(artifact_path.join("weights.bin"), b"weights").unwrap();
        let artifact = receipt(&artifact_path);
        let snapshots = tempfile::tempdir().unwrap();
        let pinned = OwnedPinnedArtifact::create_with_hooks(
            &artifact,
            Some(snapshots.path()),
            ClonePreference::CopyOnly,
            || Ok(()),
            |_| Ok(()),
        )
        .unwrap();
        let OwnedPinnedArtifact { owner, .. } = pinned;
        let (root, lease) = owner.abandon_with_lease_for_test();
        let namespace = open_directory(snapshots.path()).unwrap();
        let artifact = root.join("artifact");
        let displaced = snapshots.path().join("artifact-displaced");
        let injected = Cell::new(false);
        let error =
            remove_owned_snapshot_tree_at(&namespace, &lease, false, &mut |relative, mutation| {
                if relative == Path::new("artifact/weights.bin")
                    && mutation == CleanupMutation::UnsealFlags
                    && !injected.replace(true)
                {
                    set_snapshot_protection(&root, 0o700, false)?;
                    fs::rename(&artifact, &displaced).map_err(|error| error.to_string())?;
                    fs::create_dir(&artifact).map_err(|error| error.to_string())?;
                    fs::write(artifact.join("sentinel"), b"replacement")
                        .map_err(|error| error.to_string())?;
                }
                Ok(())
            })
            .unwrap_err();
        assert!(error.contains("path rebound"), "{error}");
        assert_eq!(fs::read(artifact.join("sentinel")).unwrap(), b"replacement");

        fs::remove_dir_all(&artifact).unwrap();
        fs::rename(&displaced, &artifact).unwrap();
        assert_eq!(
            scavenge_stale_snapshots_in(Some(snapshots.path())).unwrap(),
            1
        );
        assert_no_owned_snapshots(snapshots.path());
    }

    #[test]
    fn cleanup_receipt_refuses_a_foreign_hardlink_without_touching_its_target() {
        let source = tempfile::tempdir().unwrap();
        let artifact_path = source.path().join("artifact");
        fs::create_dir(&artifact_path).unwrap();
        fs::write(artifact_path.join("weights.bin"), b"weights").unwrap();
        let artifact = receipt(&artifact_path);
        let snapshots = tempfile::tempdir().unwrap();
        let pinned = OwnedPinnedArtifact::create_with_hooks(
            &artifact,
            Some(snapshots.path()),
            ClonePreference::CopyOnly,
            || Ok(()),
            |_| Ok(()),
        )
        .unwrap();
        let stale_root = pinned.root_path().to_path_buf();
        let sidecar = fs::read_dir(snapshots.path())
            .unwrap()
            .map(|entry| entry.unwrap().path())
            .find(|path| {
                path.file_name()
                    .and_then(OsStr::to_str)
                    .is_some_and(|name| name.starts_with(OWNER_SIDECAR_PREFIX))
            })
            .unwrap();
        unseal_tree(&stale_root).unwrap();
        let weights = stale_root.join("artifact/weights.bin");
        fs::remove_file(&weights).unwrap();
        let unrelated = source.path().join("unrelated.bin");
        fs::write(&unrelated, b"do-not-touch").unwrap();
        let unrelated_mode = fs::metadata(&unrelated).unwrap().permissions().mode();
        fs::hard_link(&unrelated, &weights).unwrap();
        // Drop retries normal cleanup, but the identity mismatch must leave both the root and its
        // durable sidecar available for a later conservative startup audit.
        drop(pinned);
        assert!(sidecar.exists());

        let error = scavenge_stale_snapshots_in(Some(snapshots.path())).unwrap_err();
        assert!(error.contains("multiply-linked"), "{error}");
        assert_eq!(fs::read(&unrelated).unwrap(), b"do-not-touch");
        assert_eq!(
            fs::metadata(&unrelated).unwrap().permissions().mode(),
            unrelated_mode
        );
        assert!(stale_root.exists());
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
        let lease_fd = owned.duplicate_lease_fd().unwrap();
        let child = VerifiedPinnedArtifact::admit(&artifact, owned.path(), lease_fd).unwrap();
        let lease_flags = unsafe { libc::fcntl(child._lease.as_raw_fd(), libc::F_GETFD) };
        assert!(lease_flags >= 0);
        assert_ne!(lease_flags & libc::FD_CLOEXEC, 0);
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
        assert_no_owned_snapshots(snapshots.path());
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
        assert_no_owned_snapshots(snapshots.path());
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
        assert_no_owned_snapshots(snapshots.path());
    }
}
