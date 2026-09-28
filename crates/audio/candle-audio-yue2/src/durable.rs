//! The one implementation of this crate's digests and durable writes (sc-22988 review): lower-case
//! hex, streaming file SHA-256, atomic (`fsync`ed, renamed, parent-synced) file writes, and the
//! file / directory syncs a transactional publication needs. Every artifact writer of this crate —
//! run directories, saved plans, closures, derived tiers — and of the gated
//! `candle-audio-sheetsage2` crate goes through these, so a durability fix lands everywhere at once.

use std::fs;
use std::io::{Read as _, Write as _};
use std::path::{Path, PathBuf};

use serde_json::Value;
use sha2::{Digest, Sha256};

/// A file-system operation on `path` failed.
#[derive(Debug, thiserror::Error)]
#[error("{}: {source}", path.display())]
pub struct IoAt {
    /// The path operated on.
    pub path: PathBuf,
    /// The error.
    pub source: std::io::Error,
}

fn at(path: &Path) -> impl FnOnce(std::io::Error) -> IoAt + '_ {
    move |source| IoAt {
        path: path.to_path_buf(),
        source,
    }
}

/// Lower-case hex of `bytes`.
pub fn hex(bytes: &[u8]) -> String {
    bytes.iter().map(|b| format!("{b:02x}")).collect()
}

/// SHA-256 (lower-case hex) of `bytes`.
pub fn sha256_hex(bytes: &[u8]) -> String {
    hex(&Sha256::digest(bytes))
}

/// SHA-256 (lower-case hex) and size of the bytes of `path` as they are on disk now, streamed in
/// 8 MiB blocks (a multi-GB weight file is never read whole).
pub fn sha256_file(path: &Path) -> Result<(String, u64), IoAt> {
    let mut file = fs::File::open(path).map_err(at(path))?;
    let mut hasher = Sha256::new();
    let mut buf = vec![0u8; 8 << 20];
    let mut bytes = 0u64;
    loop {
        let n = file.read(&mut buf).map_err(at(path))?;
        if n == 0 {
            break;
        }
        bytes += n as u64;
        hasher.update(&buf[..n]);
    }
    Ok((hex(&hasher.finalize()), bytes))
}

/// Write `bytes` to `path` durably: a sibling temporary file, `fsync`, rename onto `path`, then
/// [`sync_dir`] of the parent so the rename itself survives a crash (upstream `write_json`'s shape,
/// plus the directory sync). A reader never sees a partially written `path`.
pub fn write_atomic(path: &Path, bytes: &[u8]) -> Result<(), IoAt> {
    let name = path
        .file_name()
        .map(|n| n.to_string_lossy().into_owned())
        .unwrap_or_default();
    let tmp = path.with_file_name(format!("{name}.{}.tmp", std::process::id()));
    let mut file = fs::File::create(&tmp).map_err(at(&tmp))?;
    file.write_all(bytes)
        .and_then(|()| file.sync_all())
        .map_err(at(&tmp))?;
    drop(file);
    fs::rename(&tmp, path).map_err(at(path))?;
    match path.parent().filter(|p| !p.as_os_str().is_empty()) {
        Some(parent) => sync_dir(parent),
        None => sync_dir(Path::new(".")),
    }
}

/// [`write_atomic`] of `value` as pretty JSON with a final newline.
pub fn write_json(path: &Path, value: &Value) -> Result<(), IoAt> {
    let mut bytes = serde_json::to_vec_pretty(value)
        .map_err(|e| at(path)(std::io::Error::new(std::io::ErrorKind::InvalidData, e)))?;
    bytes.push(b'\n');
    write_atomic(path, &bytes)
}

/// `fsync` a file that is already written, without modifying its bytes.
///
/// * Unix: through a read-only handle (`fsync` needs no write access), so a read-only file — for
///   example a copy of a `0444` pinned snapshot file, whose mode `fs::copy` preserves — syncs too.
/// * Windows: `FlushFileBuffers` needs a handle with write access, and a read-only-attribute file
///   cannot be opened for writing, so the read-only attribute is cleared first. Only files this
///   crate itself wrote or copied are synced, so this changes nothing a caller owns.
pub fn sync_file(path: &Path) -> Result<(), IoAt> {
    #[cfg(not(windows))]
    {
        fs::File::open(path)
            .and_then(|f| f.sync_all())
            .map_err(at(path))
    }
    #[cfg(windows)]
    {
        let mut permissions = fs::metadata(path).map_err(at(path))?.permissions();
        if permissions.readonly() {
            // Windows has one read-only attribute (no Unix mode bits to widen): clearing it is
            // exactly what makes the copy openable for the flush.
            #[allow(clippy::permissions_set_readonly_false)]
            permissions.set_readonly(false);
            fs::set_permissions(path, permissions).map_err(at(path))?;
        }
        fs::OpenOptions::new()
            .write(true)
            .open(path)
            .and_then(|f| f.sync_all())
            .map_err(at(path))
    }
}

/// `fsync` a directory, making the renames and creations inside it durable (POSIX). On Windows a
/// directory is not opened for flushing: that needs `FILE_FLAG_BACKUP_SEMANTICS` plus write
/// access, and NTFS journals the metadata of a rename (`MoveFileEx`) itself, so the call only
/// checks that `dir` is a directory there — file contents are still flushed by [`sync_file`].
pub fn sync_dir(dir: &Path) -> Result<(), IoAt> {
    #[cfg(unix)]
    {
        fs::File::open(dir)
            .and_then(|d| d.sync_all())
            .map_err(at(dir))
    }
    #[cfg(not(unix))]
    {
        if dir.is_dir() {
            Ok(())
        } else {
            Err(at(dir)(std::io::Error::new(
                std::io::ErrorKind::NotFound,
                "not a directory",
            )))
        }
    }
}

/// Rename the fully written and synced directory `work` onto `dest` (an absent or empty `dest`
/// directory is replaced), then [`sync_dir`] `dest`'s parent so the publication survives a crash.
/// `work` itself is synced first, so every entry created in it is durable before it becomes
/// visible under `dest`.
pub fn publish_dir(work: &Path, dest: &Path) -> Result<(), IoAt> {
    sync_dir(work)?;
    if dest.is_dir() {
        fs::remove_dir(dest).map_err(at(dest))?;
    }
    fs::rename(work, dest).map_err(at(dest))?;
    match dest.parent().filter(|p| !p.as_os_str().is_empty()) {
        Some(parent) => sync_dir(parent),
        None => sync_dir(Path::new(".")),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn digests_are_lower_hex_sha256() {
        // SHA-256("abc"), FIPS 180-2 appendix B.1.
        let abc = "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad";
        assert_eq!(sha256_hex(b"abc"), abc);
        assert_eq!(hex(&[0x00, 0x0f, 0xa0, 0xff]), "000fa0ff");
        let tmp = tempfile::tempdir().unwrap();
        let path = tmp.path().join("abc");
        fs::write(&path, b"abc").unwrap();
        assert_eq!(sha256_file(&path).unwrap(), (abc.to_string(), 3));
    }

    #[test]
    fn atomic_writes_leave_no_temporary_and_publish_replaces_an_empty_target() {
        let tmp = tempfile::tempdir().unwrap();
        let path = tmp.path().join("x.json");
        write_json(&path, &serde_json::json!({"a": 1})).unwrap();
        assert_eq!(fs::read_to_string(&path).unwrap(), "{\n  \"a\": 1\n}\n");
        let names: Vec<_> = fs::read_dir(tmp.path())
            .unwrap()
            .map(|e| e.unwrap().file_name())
            .collect();
        assert_eq!(names, ["x.json"], "no temporary file is left behind");

        let work = tmp.path().join("d.partial");
        let dest = tmp.path().join("d");
        fs::create_dir(&work).unwrap();
        fs::create_dir(&dest).unwrap();
        write_atomic(&work.join("f"), b"1").unwrap();
        publish_dir(&work, &dest).unwrap();
        assert_eq!(fs::read(dest.join("f")).unwrap(), b"1");
        assert!(!work.exists());
    }
}
