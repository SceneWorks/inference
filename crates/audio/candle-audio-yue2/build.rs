//! Emits `YUE2_SOURCE_DIGEST`: a SHA-256 over this crate's `src/` tree (relative path and bytes of
//! every file, in sorted path order, CRLF normalized to LF). It is the native runtime's build
//! identity — upstream binds `runtime_sha256` (a digest of its Python sources) into every run
//! identity — so a code change invalidates earlier runs and stage checkpoints, while the same
//! sources give the same digest on every machine.

use std::path::{Path, PathBuf};

use sha2::{Digest, Sha256};

fn files(dir: &Path, out: &mut Vec<PathBuf>) {
    let mut entries: Vec<_> = std::fs::read_dir(dir)
        .unwrap_or_else(|e| panic!("reading {}: {e}", dir.display()))
        .map(|e| e.expect("directory entry").path())
        .collect();
    entries.sort();
    for path in entries {
        if path.is_dir() {
            files(&path, out);
        } else {
            out.push(path);
        }
    }
}

fn main() {
    let root = PathBuf::from(std::env::var("CARGO_MANIFEST_DIR").expect("cargo sets it"));
    let src = root.join("src");
    println!("cargo:rerun-if-changed=src");
    let mut paths = Vec::new();
    files(&src, &mut paths);
    let mut rel: Vec<(String, PathBuf)> = paths
        .into_iter()
        .map(|p| {
            let r = p
                .strip_prefix(&src)
                .expect("below src")
                .components()
                .map(|c| c.as_os_str().to_string_lossy().into_owned())
                .collect::<Vec<_>>()
                .join("/");
            (r, p)
        })
        .collect();
    rel.sort();
    let mut hasher = Sha256::new();
    for (name, path) in rel {
        let bytes = std::fs::read(&path).unwrap_or_else(|e| panic!("{}: {e}", path.display()));
        let mut normalized = Vec::with_capacity(bytes.len());
        let mut i = 0;
        while i < bytes.len() {
            if bytes[i] == b'\r' && bytes.get(i + 1) == Some(&b'\n') {
                i += 1;
                continue;
            }
            normalized.push(bytes[i]);
            i += 1;
        }
        hasher.update((name.len() as u64).to_le_bytes());
        hasher.update(name.as_bytes());
        hasher.update((normalized.len() as u64).to_le_bytes());
        hasher.update(&normalized);
    }
    let digest: String = hasher
        .finalize()
        .iter()
        .map(|b| format!("{b:02x}"))
        .collect();
    println!("cargo:rustc-env=YUE2_SOURCE_DIGEST={digest}");
}
