//! Synthetic snapshot fixtures for the YuE2 asset verifier (sc-22989) — no weights, so they run in
//! every CI lane (the audio family runs `--lib`).
//!
//! Each test builds a tiny safetensors snapshot plus its conversion manifest, wraps it in a
//! synthetic [`Component`], and drives the **same public entry points** the real closure uses
//! ([`verify_component`], [`SnapshotDirs::snapshot_dir`], [`native_tensor_digests`] /
//! [`verify_native_tensors`]). The manifest's per-tensor digests are computed here from the raw
//! little-endian element bytes the test wrote — independently of Candle — so the native check
//! passing proves Candle's loaded values reproduce the file's.

use std::path::PathBuf;

use crate::inventory::{Component, ComponentId, FileRole, PinnedFile, UpstreamRepo, YUE2_3B_REPO};
use crate::snapshot::{
    native_tensor_digests, resolve_closure, verify_component, verify_native_tensors,
};
use crate::{AssetError, Closure, SnapshotDirs, VaeVariant};
use sha2::{Digest, Sha256};

const REPO: UpstreamRepo = UpstreamRepo {
    id: "m-a-p/YuE2-Synthetic",
    revision: "0123456789abcdef0123456789abcdef01234567",
    gated: false,
    card_license: "cc-by-nc-4.0",
};

fn sha(bytes: &[u8]) -> String {
    Sha256::digest(bytes)
        .iter()
        .map(|b| format!("{b:02x}"))
        .collect()
}

fn leak<T>(v: T) -> &'static T {
    Box::leak(Box::new(v))
}

fn leak_str(s: String) -> &'static str {
    Box::leak(s.into_boxed_str())
}

/// Two tensors: `a` BF16 [2, 3] and `b` F32 [4], with deterministic, non-trivial bit patterns.
fn tensors() -> Vec<(&'static str, &'static str, Vec<usize>, Vec<u8>)> {
    let a: Vec<u8> = (0u16..6)
        .flat_map(|i| (0x3f80u16 + i * 0x11).to_le_bytes())
        .collect();
    let b: Vec<u8> = [1.5f32, -2.25, 1e-3, 7.0]
        .iter()
        .flat_map(|x| x.to_le_bytes())
        .collect();
    vec![("a", "BF16", vec![2, 3], a), ("b", "F32", vec![4], b)]
}

/// A safetensors file for `tensors`.
fn safetensors_bytes(tensors: &[(&str, &str, Vec<usize>, Vec<u8>)]) -> Vec<u8> {
    let mut header = serde_json::Map::new();
    let mut offset = 0usize;
    for (name, dtype, shape, data) in tensors {
        header.insert(
            name.to_string(),
            serde_json::json!({"dtype": dtype, "shape": shape, "data_offsets": [offset, offset + data.len()]}),
        );
        offset += data.len();
    }
    let mut h = serde_json::to_vec(&serde_json::Value::Object(header)).unwrap();
    while !h.len().is_multiple_of(8) {
        h.push(b' ');
    }
    let mut out = (h.len() as u64).to_le_bytes().to_vec();
    out.extend_from_slice(&h);
    for (_, _, _, data) in tensors {
        out.extend_from_slice(data);
    }
    out
}

struct Fixture {
    _tmp: tempfile::TempDir,
    dir: PathBuf,
    component: &'static Component,
}

/// A complete synthetic snapshot directory: weights, a config and a LICENSE, all pinned.
/// `edit_manifest` may alter the manifest's tensor rows before it is frozen.
fn fixture_with(edit_manifest: impl FnOnce(&mut Vec<serde_json::Value>)) -> Fixture {
    let tmp = tempfile::tempdir().unwrap();
    let dir = tmp.path().join("snapshot");
    std::fs::create_dir_all(&dir).unwrap();
    let t = tensors();
    let weights = safetensors_bytes(&t);
    let config = br#"{"model_type": "yue2_synthetic"}"#.to_vec();
    let license = b"Attribution-NonCommercial 4.0 International (synthetic fixture)\n".to_vec();
    std::fs::write(dir.join("model.safetensors"), &weights).unwrap();
    std::fs::write(dir.join("config.json"), &config).unwrap();
    std::fs::write(dir.join("LICENSE"), &license).unwrap();

    let files: Vec<(&str, &Vec<u8>, FileRole)> = vec![
        ("config.json", &config, FileRole::Config),
        ("model.safetensors", &weights, FileRole::Weights),
        ("LICENSE", &license, FileRole::License),
    ];
    let pinned: Vec<PinnedFile> = files
        .iter()
        .map(|(path, bytes, role)| PinnedFile {
            path,
            bytes: bytes.len() as u64,
            sha256: leak_str(sha(bytes)),
            role: *role,
        })
        .collect();
    let mut rows: Vec<serde_json::Value> = t
        .iter()
        .map(|(name, dtype, shape, data)| {
            serde_json::json!({"name": name, "dtype": dtype, "shape": shape, "sha256": sha(data)})
        })
        .collect();
    edit_manifest(&mut rows);
    let source_files: serde_json::Map<String, serde_json::Value> = files
        .iter()
        .map(|(path, bytes, _)| {
            (
                path.to_string(),
                serde_json::json!({"bytes": bytes.len(), "sha256": sha(bytes)}),
            )
        })
        .collect();
    let manifest = serde_json::json!({
        "schema": 1,
        "component": "yue2_synthetic",
        "source": {"repo": REPO.id, "revision": REPO.revision, "files": source_files},
        "conversion": {"kind": "identity"},
        "native": {"file": "model.safetensors", "bytes": weights.len(), "sha256": sha(&weights)},
        "tensors": rows,
    });
    let component = leak(Component {
        id: ComponentId::Lm,
        key: "yue2_synthetic",
        repo: REPO,
        files: Box::leak(pinned.into_boxed_slice()),
        manifest_json: leak_str(manifest.to_string()),
    });
    Fixture {
        _tmp: tmp,
        dir,
        component,
    }
}

fn fixture() -> Fixture {
    fixture_with(|_| {})
}

fn weights_path(f: &Fixture) -> PathBuf {
    f.dir.join("model.safetensors")
}

#[test]
fn a_complete_snapshot_verifies_and_its_native_values_reproduce_the_original() {
    let f = fixture();
    let v = verify_component(f.component, &f.dir).expect("complete snapshot verifies");
    // The verifier hands back the exact paths it hashed, inside the snapshot.
    assert_eq!(v.weights_path(), Some(weights_path(&f).as_path()));
    assert_eq!(v.files().len(), 3);
    for file in v.files() {
        assert!(file.path.starts_with(&f.dir), "{}", file.path.display());
    }
    // Candle's loaded values, re-hashed, equal the raw bytes the fixture wrote.
    assert_eq!(verify_native_tensors(&v).unwrap(), 2);
    let native = native_tensor_digests(&v).unwrap();
    assert_eq!(native[0].name, "a");
    assert_eq!(native[0].dtype, "BF16");
    assert_eq!(native[0].shape, vec![2, 3]);
    assert_eq!(native[0].sha256, sha(&tensors()[0].3));
}

#[test]
fn a_missing_shard_is_an_explicit_failure() {
    let f = fixture();
    std::fs::remove_file(weights_path(&f)).unwrap();
    match verify_component(f.component, &f.dir) {
        Err(AssetError::MissingFile { file, .. }) => assert_eq!(file, "model.safetensors"),
        other => panic!("expected MissingFile, got {other:?}"),
    }
}

#[test]
fn a_truncated_shard_is_an_explicit_failure() {
    let f = fixture();
    let bytes = std::fs::read(weights_path(&f)).unwrap();
    std::fs::write(weights_path(&f), &bytes[..bytes.len() - 4]).unwrap();
    match verify_component(f.component, &f.dir) {
        Err(AssetError::SizeMismatch {
            file,
            expected,
            actual,
            ..
        }) => {
            assert_eq!(file, "model.safetensors");
            assert_eq!(expected, actual + 4);
        }
        other => panic!("expected SizeMismatch, got {other:?}"),
    }
}

#[test]
fn a_corrupt_shard_of_the_right_size_is_an_explicit_failure() {
    let f = fixture();
    let mut bytes = std::fs::read(weights_path(&f)).unwrap();
    let last = bytes.len() - 1; // a tensor value byte, not the header
    bytes[last] ^= 0x01;
    std::fs::write(weights_path(&f), &bytes).unwrap();
    match verify_component(f.component, &f.dir) {
        Err(AssetError::HashMismatch { file, .. }) => assert_eq!(file, "model.safetensors"),
        other => panic!("expected HashMismatch, got {other:?}"),
    }
}

#[test]
fn a_modified_licence_file_is_an_explicit_failure() {
    let f = fixture();
    let mut bytes = std::fs::read(f.dir.join("LICENSE")).unwrap();
    bytes[0] = b'X';
    std::fs::write(f.dir.join("LICENSE"), &bytes).unwrap();
    match verify_component(f.component, &f.dir) {
        Err(AssetError::HashMismatch { file, .. }) => assert_eq!(file, "LICENSE"),
        other => panic!("expected HashMismatch on LICENSE, got {other:?}"),
    }
}

#[test]
fn an_unpinned_extra_shard_is_refused() {
    let f = fixture();
    std::fs::write(f.dir.join("model-00002-of-00002.safetensors"), b"stray").unwrap();
    match verify_component(f.component, &f.dir) {
        Err(AssetError::UnexpectedWeightFile { file, .. }) => {
            assert_eq!(file, "model-00002-of-00002.safetensors")
        }
        other => panic!("expected UnexpectedWeightFile, got {other:?}"),
    }
}

/// A stray shard or index one or more directories down is found too, and reported by its
/// snapshot-relative path; a harmless nested non-weights file (like the pinned `licenses/` texts)
/// is not.
#[test]
fn a_nested_stray_shard_or_index_is_refused() {
    let f = fixture();
    let nested = f.dir.join("licenses");
    std::fs::create_dir_all(&nested).unwrap();
    std::fs::write(nested.join("NOTICE.txt"), b"not weights").unwrap();
    verify_component(f.component, &f.dir).expect("a nested non-weights file is fine");

    let deep = f.dir.join("sub/deeper");
    std::fs::create_dir_all(&deep).unwrap();
    std::fs::write(deep.join("model.safetensors.index.json"), b"{}").unwrap();
    match verify_component(f.component, &f.dir) {
        Err(AssetError::UnexpectedWeightFile { file, .. }) => {
            assert_eq!(file, "sub/deeper/model.safetensors.index.json")
        }
        other => panic!("expected UnexpectedWeightFile, got {other:?}"),
    }
}

/// A symlink loop in the snapshot tree is refused as an I/O failure (the OS's ELOOP, propagated)
/// instead of recursing forever, being skipped, or being misreported as a stray shard.
#[cfg(unix)]
#[test]
fn a_symlink_loop_in_the_snapshot_tree_is_refused() {
    let f = fixture();
    std::os::unix::fs::symlink(&f.dir, f.dir.join("loop")).unwrap();
    match verify_component(f.component, &f.dir) {
        Err(AssetError::Io { source, .. }) => {
            let msg = source.to_string();
            assert!(msg.contains("symbolic links"), "{msg}")
        }
        other => panic!("expected an Io refusal, got {other:?}"),
    }
}

#[test]
fn a_manifest_tensor_table_that_disagrees_with_the_file_is_refused() {
    let f = fixture_with(|rows| rows[1]["shape"] = serde_json::json!([2, 2]));
    match verify_component(f.component, &f.dir) {
        Err(AssetError::TensorTableMismatch { detail, .. }) => {
            assert!(
                detail.contains("`b` is F32 [4], manifest says F32 [2, 2]"),
                "{detail}"
            )
        }
        other => panic!("expected TensorTableMismatch, got {other:?}"),
    }
    let f = fixture_with(|rows| {
        rows.pop();
    });
    match verify_component(f.component, &f.dir) {
        Err(AssetError::TensorTableMismatch { detail, .. }) => {
            assert!(detail.contains("`b` is not in the manifest"), "{detail}")
        }
        other => panic!("expected TensorTableMismatch, got {other:?}"),
    }
}

#[test]
fn native_values_that_differ_from_the_original_digest_are_refused() {
    let f = fixture_with(|rows| rows[0]["sha256"] = serde_json::json!("0".repeat(64)));
    let v = verify_component(f.component, &f.dir).expect("file-level checks pass");
    match verify_native_tensors(&v) {
        Err(AssetError::TensorValueMismatch { detail, .. }) => {
            assert!(
                detail.starts_with("1 difference(s): `a` loaded as BF16"),
                "{detail}"
            )
        }
        other => panic!("expected TensorValueMismatch, got {other:?}"),
    }
}

#[test]
fn an_offline_cache_miss_is_explicit() {
    // Nothing provisioned for the repository.
    match SnapshotDirs::new().snapshot_dir(&YUE2_3B_REPO) {
        Err(AssetError::CacheMiss { repo, revision, .. }) => {
            assert_eq!(repo, "m-a-p/YuE2-3B");
            assert_eq!(revision, YUE2_3B_REPO.revision);
        }
        other => panic!("expected CacheMiss, got {other:?}"),
    }
    // Provisioned, but the directory is not there (evicted cache, unmounted volume).
    let tmp = tempfile::tempdir().unwrap();
    let gone = tmp.path().join("evicted");
    let dirs = SnapshotDirs::new().with(YUE2_3B_REPO.id, &gone);
    match dirs.snapshot_dir(&YUE2_3B_REPO) {
        Err(AssetError::CacheMiss { looked_in, .. }) => assert_eq!(looked_in, gone),
        other => panic!("expected CacheMiss, got {other:?}"),
    }
    // A directory provisioned under another repository's id does not satisfy this one.
    let other = SnapshotDirs::new().with("m-a-p/YuE2-Vae", tmp.path());
    assert!(matches!(
        other.snapshot_dir(&YUE2_3B_REPO),
        Err(AssetError::CacheMiss { .. })
    ));
    let present = SnapshotDirs::new().with(YUE2_3B_REPO.id, tmp.path());
    assert_eq!(present.snapshot_dir(&YUE2_3B_REPO).unwrap(), tmp.path());
}

#[test]
fn a_generation_closure_on_a_cold_cache_fails_explicitly_without_fetching() {
    let tmp = tempfile::tempdir().unwrap();
    let closure = Closure::Generation {
        vae: VaeVariant::Legacy,
    };
    // Nothing provisioned: the first component is a cache miss.
    match resolve_closure(closure, &SnapshotDirs::new()) {
        Err(AssetError::CacheMiss { repo, .. }) => assert_eq!(repo, "m-a-p/YuE2-3B"),
        other => panic!("expected CacheMiss, got {other:?}"),
    }
    // An empty directory provisioned for the LM: its first pinned file is missing.
    let dirs = SnapshotDirs::new().with(YUE2_3B_REPO.id, tmp.path());
    match resolve_closure(closure, &dirs) {
        Err(AssetError::MissingFile {
            component, file, ..
        }) => {
            assert_eq!(component, "yue2_3b");
            assert_eq!(file, "config.json");
        }
        other => panic!("expected MissingFile, got {other:?}"),
    }
    // Nothing was written into the provisioned directory.
    assert_eq!(std::fs::read_dir(tmp.path()).unwrap().count(), 0);
}

/// Download caches commonly store snapshot entries as symlinks into a blob store. The verifier follows them —
/// hashing the bytes a loader reads — and a dangling link (an interrupted download) is a missing
/// file, not a pass.
#[cfg(unix)]
#[test]
fn hub_cache_symlinks_are_followed_and_a_dangling_one_is_missing() {
    let f = fixture();
    let blobs = f.dir.parent().unwrap().join("blobs");
    std::fs::create_dir_all(&blobs).unwrap();
    let blob = blobs.join("deadbeef");
    std::fs::rename(weights_path(&f), &blob).unwrap();
    std::os::unix::fs::symlink(&blob, weights_path(&f)).unwrap();
    let v = verify_component(f.component, &f.dir).expect("symlinked snapshot verifies");
    assert_eq!(verify_native_tensors(&v).unwrap(), 2);

    std::fs::remove_file(&blob).unwrap();
    match verify_component(f.component, &f.dir) {
        Err(AssetError::MissingFile { file, .. }) => assert_eq!(file, "model.safetensors"),
        other => panic!("expected MissingFile, got {other:?}"),
    }
}

#[test]
fn a_garbage_header_with_a_matching_pin_is_an_invalid_header() {
    // A "weights" file whose pinned hash matches but which is not safetensors at all.
    let tmp = tempfile::tempdir().unwrap();
    let dir = tmp.path();
    let junk = vec![0xffu8; 64];
    std::fs::write(dir.join("model.safetensors"), &junk).unwrap();
    let files: &'static [PinnedFile] = Box::leak(Box::new([PinnedFile {
        path: "model.safetensors",
        bytes: 64,
        sha256: leak_str(sha(&junk)),
        role: FileRole::Weights,
    }]));
    let manifest = serde_json::json!({
        "schema": 1, "component": "yue2_junk",
        "source": {"repo": REPO.id, "revision": REPO.revision,
                   "files": {"model.safetensors": {"bytes": 64, "sha256": sha(&junk)}}},
        "conversion": {"kind": "identity"},
        "native": {"file": "model.safetensors", "bytes": 64, "sha256": sha(&junk)},
        "tensors": [],
    });
    let component = leak(Component {
        id: ComponentId::Lm,
        key: "yue2_junk",
        repo: REPO,
        files,
        manifest_json: leak_str(manifest.to_string()),
    });
    assert!(matches!(
        verify_component(component, dir),
        Err(AssetError::InvalidHeader { .. })
    ));
}
