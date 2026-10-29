//! One snapshot-inventory policy for provider receipts and sealed inventories (sc-27253).
//!
//! Seven Candle providers (sdxl, sd3, sana, kolors, boogu, ideogram, chroma) each carried a private
//! recursive walker, a streaming SHA-256, a length-framed digest update, and a hex encoder. The
//! walkers had drifted: one followed directory symlinks, another refused them, one pushed a
//! symlink to a directory (or a dangling one) as a file, one ran its hidden/partial-download name
//! check only on regular files so a symlinked `*.incomplete` passed (every Hugging Face cache entry
//! is a symlink), some skipped FIFOs and sockets while others refused them, and only one knew the
//! `.partial` suffix.
//!
//! [`collect_regular_files`] is the single policy:
//!
//! - The `root` itself may be a regular file (a one-file artifact) or a directory; a symlinked root is
//!   followed, because it is the caller's chosen path.
//! - Below the root, a symlink to a regular file is collected (the Hugging Face cache layout) and a
//!   symlink to a directory is followed (e.g. a shared `vae/` linked into a snapshot). A dangling
//!   link is refused, and so is a directory link that resolves to a directory already on the
//!   current descent (a cycle), so a walk always terminates.
//! - Directories are recursed (whatever their name).
//! - Special files (FIFOs, sockets, devices) are refused, never silently skipped.
//! - With [`InventoryPolicy::reject_markers`], a hidden (leading `.`) or partial-download
//!   ([`PARTIAL_DOWNLOAD_SUFFIXES`]) **file name** is refused whether the entry is a regular file or a
//!   symlink. Exact-inventory providers that compare against an expected file set instead (sd3, sana)
//!   leave it off: a stray marker already fails their set comparison as an extra entry.
//! - The result is the sorted, de-duplicated list of absolute lexical paths (not canonicalized):
//!   a file reached through a directory link is named by its path under the link, never by the
//!   link target, so a receipt keyed on the root-relative path is the same whether a component
//!   directory is real or linked to identical content.
//!
//! The hashing helpers are byte-for-byte the per-crate copies they replace, so every receipt and
//! fingerprint digest computed through them is unchanged.

use std::io::Read;
use std::path::{Path, PathBuf};

use sha2::{Digest, Sha256};

use crate::{Error, Result};

/// File-name suffixes that mark an unfinished or in-flight download (union of the per-crate lists).
pub const PARTIAL_DOWNLOAD_SUFFIXES: [&str; 5] =
    [".part", ".partial", ".tmp", ".lock", ".incomplete"];

/// What [`collect_regular_files`] refuses beyond the fixed structural rules (see the module docs).
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct InventoryPolicy {
    /// Refuse hidden (leading `.`) and [`PARTIAL_DOWNLOAD_SUFFIXES`] file names, including symlinks.
    pub reject_markers: bool,
}

impl InventoryPolicy {
    /// Structural rules only: exact-inventory providers whose set comparison already rejects extras.
    pub const STRUCTURAL: Self = Self {
        reject_markers: false,
    };
    /// Structural rules plus the hidden / partial-download name refusal.
    pub const REJECT_MARKERS: Self = Self {
        reject_markers: true,
    };
}

/// Whether `name` is a hidden or partial-download marker under [`InventoryPolicy::reject_markers`].
pub fn is_marker_name(name: &str) -> bool {
    name.starts_with('.')
        || PARTIAL_DOWNLOAD_SUFFIXES
            .iter()
            .any(|suffix| name.ends_with(suffix))
}

/// Every regular file under `root` (or `root` itself when it is a file), as sorted, de-duplicated
/// absolute lexical paths, under the single policy in the module docs. `label` prefixes refusals so
/// providers keep their diagnostics (e.g. `"sdxl"`).
pub fn collect_regular_files(
    label: &str,
    root: &Path,
    policy: InventoryPolicy,
) -> Result<Vec<PathBuf>> {
    let root = std::path::absolute(root)?;
    let metadata = std::fs::metadata(&root)?;
    let mut files = Vec::new();
    if metadata.is_file() {
        push_file(label, root, policy, &mut files)?;
    } else if metadata.is_dir() {
        let mut descent = vec![std::fs::canonicalize(&root)?];
        visit(label, &root, policy, &mut descent, &mut files)?;
    } else {
        return Err(Error::Unsupported(format!(
            "{label}: artifact source is neither a file nor a directory: {}",
            root.display()
        )));
    }
    files.sort();
    files.dedup();
    Ok(files)
}

/// Recurse into `dir`, whose canonical path is the last entry of `descent` (the canonical paths of
/// every directory from the root down to `dir`), so a link back into the descent is a cycle.
fn visit(
    label: &str,
    dir: &Path,
    policy: InventoryPolicy,
    descent: &mut Vec<PathBuf>,
    files: &mut Vec<PathBuf>,
) -> Result<()> {
    for entry in std::fs::read_dir(dir)? {
        let path = entry?.path();
        let metadata = std::fs::symlink_metadata(&path)?;
        if metadata.file_type().is_symlink() {
            // `metadata` follows the link; a dangling link surfaces as this error, not a skip.
            let target = std::fs::metadata(&path).map_err(|error| {
                Error::Unsupported(format!(
                    "{label}: symlink does not resolve: {}: {error}",
                    path.display()
                ))
            })?;
            if target.is_file() {
                push_file(label, path, policy, files)?;
            } else if target.is_dir() {
                descend(label, path, policy, descent, files)?;
            } else {
                return Err(Error::Unsupported(format!(
                    "{label}: symlink resolves to a special file: {}",
                    path.display()
                )));
            }
        } else if metadata.is_dir() {
            descend(label, path, policy, descent, files)?;
        } else if metadata.is_file() {
            push_file(label, path, policy, files)?;
        } else {
            return Err(Error::Unsupported(format!(
                "{label}: refusing special file in artifact: {}",
                path.display()
            )));
        }
    }
    Ok(())
}

/// Visit the sub-directory `path` (real or linked) under its lexical path, refusing it when its
/// canonical path is already on the current descent.
fn descend(
    label: &str,
    path: PathBuf,
    policy: InventoryPolicy,
    descent: &mut Vec<PathBuf>,
    files: &mut Vec<PathBuf>,
) -> Result<()> {
    let canonical = std::fs::canonicalize(&path)?;
    if descent.contains(&canonical) {
        return Err(Error::Unsupported(format!(
            "{label}: directory symlink cycle: {} resolves to {}, which encloses it",
            path.display(),
            canonical.display()
        )));
    }
    descent.push(canonical);
    let visited = visit(label, &path, policy, descent, files);
    descent.pop();
    visited
}

fn push_file(
    label: &str,
    path: PathBuf,
    policy: InventoryPolicy,
    files: &mut Vec<PathBuf>,
) -> Result<()> {
    if policy.reject_markers {
        let name = path
            .file_name()
            .and_then(|name| name.to_str())
            .unwrap_or_default();
        if is_marker_name(name) {
            return Err(Error::Unsupported(format!(
                "{label}: refusing incomplete or hidden artifact {}",
                path.display()
            )));
        }
    }
    files.push(path);
    Ok(())
}

/// Streaming SHA-256 of `path`'s bytes through a fixed 64 KiB buffer (no whole-file buffer).
pub fn sha256_file(path: &Path) -> Result<[u8; 32]> {
    Ok(sha256_file_counted(path)?.0)
}

/// [`sha256_file`] plus the number of bytes hashed, for callers that cross-check the length they
/// read against file metadata.
pub fn sha256_file_counted(path: &Path) -> Result<([u8; 32], u64)> {
    let mut file = std::fs::File::open(path)?;
    let mut digest = Sha256::new();
    let mut buffer = vec![0_u8; 64 * 1024];
    let mut total = 0_u64;
    loop {
        let read = file.read(&mut buffer)?;
        if read == 0 {
            break;
        }
        digest.update(&buffer[..read]);
        total += read as u64;
    }
    Ok((digest.finalize().into(), total))
}

/// Feed `bytes` into `digest` behind its little-endian `u64` length, so adjacent fields cannot run
/// together.
pub fn update_framed(digest: &mut Sha256, bytes: &[u8]) {
    digest.update((bytes.len() as u64).to_le_bytes());
    digest.update(bytes);
}

/// Lower-case hexadecimal encoding of `bytes`.
pub fn hex_lower(bytes: &[u8]) -> String {
    use std::fmt::Write;
    let mut encoded = String::with_capacity(bytes.len() * 2);
    for byte in bytes {
        write!(&mut encoded, "{byte:02x}").expect("writing to a String cannot fail");
    }
    encoded
}

#[cfg(test)]
mod tests {
    use super::*;

    fn names(root: &Path, files: &[PathBuf]) -> Vec<String> {
        files
            .iter()
            .map(|path| {
                path.strip_prefix(root)
                    .unwrap()
                    .to_string_lossy()
                    .replace('\\', "/")
            })
            .collect()
    }

    #[test]
    fn collects_sorted_absolute_files_recursively_including_hidden_directories() {
        let tmp = tempfile::tempdir().unwrap();
        let root = std::path::absolute(tmp.path()).unwrap();
        std::fs::create_dir_all(root.join("b/c")).unwrap();
        std::fs::create_dir_all(root.join(".cache")).unwrap();
        std::fs::write(root.join("b/c/z.safetensors"), b"z").unwrap();
        std::fs::write(root.join("a.json"), b"a").unwrap();
        std::fs::write(root.join(".cache/meta"), b"m").unwrap();
        let files = collect_regular_files("t", &root, InventoryPolicy::REJECT_MARKERS).unwrap();
        assert!(files.iter().all(|path| path.is_absolute()));
        assert_eq!(
            names(&root, &files),
            [".cache/meta", "a.json", "b/c/z.safetensors"]
        );
    }

    #[test]
    fn a_file_root_is_its_own_inventory() {
        let tmp = tempfile::tempdir().unwrap();
        let file = tmp.path().join("model.safetensors");
        std::fs::write(&file, b"x").unwrap();
        let files = collect_regular_files("t", &file, InventoryPolicy::REJECT_MARKERS).unwrap();
        assert_eq!(files, [std::path::absolute(&file).unwrap()]);
    }

    #[test]
    fn marker_names_are_refused_only_under_reject_markers() {
        for marker in [
            ".hidden",
            "x.part",
            "x.partial",
            "x.tmp",
            "x.lock",
            "x.incomplete",
        ] {
            let tmp = tempfile::tempdir().unwrap();
            std::fs::write(tmp.path().join("ok.json"), b"{}").unwrap();
            std::fs::write(tmp.path().join(marker), b"partial").unwrap();
            assert!(
                collect_regular_files("t", tmp.path(), InventoryPolicy::REJECT_MARKERS).is_err(),
                "{marker} must be refused"
            );
            assert_eq!(
                collect_regular_files("t", tmp.path(), InventoryPolicy::STRUCTURAL)
                    .unwrap()
                    .len(),
                2,
                "{marker} is an ordinary entry under the structural policy"
            );
        }
    }

    #[cfg(unix)]
    #[test]
    fn symlinked_marker_is_refused_like_a_regular_one() {
        let tmp = tempfile::tempdir().unwrap();
        let blobs = tmp.path().join("blobs");
        let snapshot = tmp.path().join("snapshot");
        std::fs::create_dir_all(&blobs).unwrap();
        std::fs::create_dir_all(&snapshot).unwrap();
        std::fs::write(blobs.join("abc"), b"partial").unwrap();
        std::os::unix::fs::symlink(blobs.join("abc"), snapshot.join("model.incomplete")).unwrap();
        assert!(collect_regular_files("t", &snapshot, InventoryPolicy::REJECT_MARKERS).is_err());
        // A symlink to a regular file with an ordinary name is collected at its lexical path.
        std::fs::remove_file(snapshot.join("model.incomplete")).unwrap();
        std::os::unix::fs::symlink(blobs.join("abc"), snapshot.join("model.safetensors")).unwrap();
        let files = collect_regular_files("t", &snapshot, InventoryPolicy::REJECT_MARKERS).unwrap();
        assert_eq!(
            files,
            [std::path::absolute(snapshot.join("model.safetensors")).unwrap()]
        );
    }

    /// Root-relative names joined with their content digest, the shape a receipt keys on.
    fn keyed_digests(root: &Path, files: &[PathBuf]) -> Vec<(String, [u8; 32])> {
        names(root, files)
            .into_iter()
            .zip(files.iter().map(|path| sha256_file(path).unwrap()))
            .collect()
    }

    #[cfg(unix)]
    #[test]
    fn directory_symlinks_are_followed_under_their_lexical_path() {
        let tmp = tempfile::tempdir().unwrap();
        let shared = tmp.path().join("shared-vae");
        std::fs::create_dir_all(shared.join("nested")).unwrap();
        std::fs::write(shared.join("config.json"), b"{}").unwrap();
        std::fs::write(shared.join("nested/weights.safetensors"), b"w").unwrap();

        let linked = std::path::absolute(tmp.path().join("linked")).unwrap();
        std::fs::create_dir_all(&linked).unwrap();
        std::fs::write(linked.join("model_index.json"), b"{}").unwrap();
        std::os::unix::fs::symlink(&shared, linked.join("vae")).unwrap();

        let real = std::path::absolute(tmp.path().join("real")).unwrap();
        std::fs::create_dir_all(real.join("vae/nested")).unwrap();
        std::fs::write(real.join("model_index.json"), b"{}").unwrap();
        std::fs::write(real.join("vae/config.json"), b"{}").unwrap();
        std::fs::write(real.join("vae/nested/weights.safetensors"), b"w").unwrap();

        for policy in [InventoryPolicy::STRUCTURAL, InventoryPolicy::REJECT_MARKERS] {
            let through_link = collect_regular_files("t", &linked, policy).unwrap();
            assert!(through_link.iter().all(|path| path.starts_with(&linked)));
            assert_eq!(
                names(&linked, &through_link),
                [
                    "model_index.json",
                    "vae/config.json",
                    "vae/nested/weights.safetensors"
                ]
            );
            let direct = collect_regular_files("t", &real, policy).unwrap();
            assert_eq!(
                keyed_digests(&linked, &through_link),
                keyed_digests(&real, &direct),
                "a linked component directory inventories like the real one"
            );
        }
    }

    #[cfg(unix)]
    #[test]
    fn markers_below_a_directory_symlink_are_still_refused() {
        let tmp = tempfile::tempdir().unwrap();
        let shared = tmp.path().join("shared");
        let snapshot = tmp.path().join("snapshot");
        std::fs::create_dir_all(&shared).unwrap();
        std::fs::create_dir_all(&snapshot).unwrap();
        std::fs::write(shared.join("weights.safetensors.incomplete"), b"w").unwrap();
        std::os::unix::fs::symlink(&shared, snapshot.join("vae")).unwrap();
        assert!(collect_regular_files("t", &snapshot, InventoryPolicy::REJECT_MARKERS).is_err());
    }

    #[cfg(unix)]
    #[test]
    fn dangling_symlinks_are_refused() {
        let tmp = tempfile::tempdir().unwrap();
        let snapshot = tmp.path().join("snapshot");
        std::fs::create_dir_all(&snapshot).unwrap();
        std::fs::write(snapshot.join("ok.json"), b"{}").unwrap();
        std::os::unix::fs::symlink(tmp.path().join("missing"), snapshot.join("vae")).unwrap();
        for policy in [InventoryPolicy::STRUCTURAL, InventoryPolicy::REJECT_MARKERS] {
            let error = collect_regular_files("t", &snapshot, policy).unwrap_err();
            assert!(
                matches!(&error, Error::Unsupported(message) if message.contains("does not resolve")),
                "{error}"
            );
        }
    }

    #[cfg(unix)]
    #[test]
    fn directory_symlink_cycles_are_refused_not_followed_forever() {
        let tmp = tempfile::tempdir().unwrap();
        let snapshot = tmp.path().join("snapshot");
        std::fs::create_dir_all(snapshot.join("a/b")).unwrap();
        std::fs::write(snapshot.join("a/b/w.safetensors"), b"w").unwrap();
        // A link back to the root, and a link back to an intermediate directory.
        for (link, target) in [
            ("a/b/root", snapshot.clone()),
            ("a/b/up", snapshot.join("a")),
        ] {
            std::os::unix::fs::symlink(&target, snapshot.join(link)).unwrap();
            for policy in [InventoryPolicy::STRUCTURAL, InventoryPolicy::REJECT_MARKERS] {
                let error = collect_regular_files("t", &snapshot, policy).unwrap_err();
                assert!(
                    matches!(&error, Error::Unsupported(message) if message.contains("cycle")),
                    "{error}"
                );
            }
            std::fs::remove_file(snapshot.join(link)).unwrap();
        }
        // The same root reached through a linked root is still a cycle.
        let linked_root = tmp.path().join("linked-root");
        std::os::unix::fs::symlink(&snapshot, &linked_root).unwrap();
        std::os::unix::fs::symlink(&snapshot, snapshot.join("a/again")).unwrap();
        assert!(collect_regular_files("t", &linked_root, InventoryPolicy::STRUCTURAL).is_err());
    }

    #[cfg(unix)]
    #[test]
    fn sibling_links_to_one_shared_directory_are_not_a_cycle() {
        let tmp = tempfile::tempdir().unwrap();
        let shared = tmp.path().join("shared");
        let snapshot = tmp.path().join("snapshot");
        std::fs::create_dir_all(&shared).unwrap();
        std::fs::create_dir_all(&snapshot).unwrap();
        std::fs::write(shared.join("w.safetensors"), b"w").unwrap();
        std::os::unix::fs::symlink(&shared, snapshot.join("text_encoder")).unwrap();
        std::os::unix::fs::symlink(&shared, snapshot.join("text_encoder_2")).unwrap();
        let files = collect_regular_files("t", &snapshot, InventoryPolicy::STRUCTURAL).unwrap();
        assert_eq!(
            names(&snapshot, &files),
            ["text_encoder/w.safetensors", "text_encoder_2/w.safetensors"]
        );
    }

    #[cfg(unix)]
    #[test]
    fn special_files_are_refused_not_skipped() {
        let tmp = tempfile::tempdir().unwrap();
        std::fs::write(tmp.path().join("ok.json"), b"{}").unwrap();
        let fifo = tmp.path().join("pipe");
        let status = std::process::Command::new("mkfifo")
            .arg(&fifo)
            .status()
            .expect("mkfifo is available on unix hosts");
        assert!(status.success());
        for policy in [InventoryPolicy::STRUCTURAL, InventoryPolicy::REJECT_MARKERS] {
            assert!(collect_regular_files("t", tmp.path(), policy).is_err());
        }
    }

    #[test]
    fn hashing_helpers_match_the_reference_encodings() {
        let tmp = tempfile::tempdir().unwrap();
        let file = tmp.path().join("f");
        let bytes = vec![7_u8; 200_000];
        std::fs::write(&file, &bytes).unwrap();
        let expected: [u8; 32] = Sha256::digest(&bytes).into();
        assert_eq!(sha256_file(&file).unwrap(), expected);
        assert_eq!(
            sha256_file_counted(&file).unwrap(),
            (expected, bytes.len() as u64)
        );

        let mut framed = Sha256::new();
        update_framed(&mut framed, b"abc");
        let mut manual = Sha256::new();
        manual.update(3_u64.to_le_bytes());
        manual.update(b"abc");
        assert_eq!(framed.finalize(), manual.finalize());

        assert_eq!(hex_lower(&[0x00, 0x0f, 0xa5, 0xff]), "000fa5ff");
    }
}
