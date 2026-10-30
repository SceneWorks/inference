use super::*;
use crate::conversion::open_conversion;
use gen_core::safetensors_path_tensor_headers;

fn fixture() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("../../media/mlx-gen/mlx-gen-flux3/tests/fixtures/tiny-snapshot")
}

/// A private copy of the committed tiny dense package plus the license files a real one ships.
fn source_copy(dir: &Path) -> ConversionSource {
    source_copy_kind(dir, false)
}
fn source_copy_kind(dir: &Path, base: bool) -> ConversionSource {
    let selected = if base {
        fixture().parent().unwrap().join("tiny-base-snapshot")
    } else {
        fixture()
    };
    let root = dir.join("source");
    std::fs::create_dir_all(root.join("text_encoder")).unwrap();
    for entry in ["config.json", "model.safetensors", "ae.safetensors"] {
        let input = if entry == "ae.safetensors" {
            fixture().join(entry)
        } else {
            selected.join(entry)
        };
        std::fs::copy(input, root.join(entry)).unwrap();
    }
    for entry in std::fs::read_dir(fixture().join("text_encoder")).unwrap() {
        let entry = entry.unwrap();
        std::fs::copy(
            entry.path(),
            root.join("text_encoder").join(entry.file_name()),
        )
        .unwrap();
    }
    std::fs::write(root.join("LICENSE.md"), b"license").unwrap();
    std::fs::write(root.join("NOTICE.md"), b"notice").unwrap();
    ConversionSource {
        root,
        repository: if base {
            crate::BASE_WEIGHTS_REPO
        } else {
            crate::DISTILLED_WEIGHTS_REPO
        }
        .into(),
        revision: if base {
            crate::BASE_WEIGHTS_REVISION
        } else {
            crate::DISTILLED_WEIGHTS_REVISION
        }
        .into(),
    }
}

/// *Mutations that red this:* unsorted/streamed-out-of-order output, a timestamp or path in the
/// metadata, a companion written for the wrong stem, or quantizing anything but the dense source.
#[test]
fn affine_conversion_is_deterministic_complete_and_dequantizes_to_the_source() {
    let dir = tempfile::tempdir().unwrap();
    let source = source_copy(dir.path());
    for (bits, group) in [(4, 64), (8, 32)] {
        let one = dir.path().join(format!("q{bits}g{group}-a"));
        let two = dir.path().join(format!("q{bits}g{group}-b"));
        let r1 = convert_affine(&source, bits, group, &one, &CancelFlag::new()).unwrap();
        let r2 = convert_affine(&source, bits, group, &two, &CancelFlag::new()).unwrap();
        assert_eq!(r1, r2);
        assert_eq!(
            std::fs::read(one.join(DIT_WEIGHTS_FILE)).unwrap(),
            std::fs::read(two.join(DIT_WEIGHTS_FILE)).unwrap()
        );
        assert_eq!(open_conversion(&one).unwrap(), r1);
        r1.verify_external(&source.root).unwrap();
        let storage = NativeDitStorage::MlxAffine {
            bits,
            group_size: group,
        };
        assert_eq!(NativeDitStorage::from_package(&one).unwrap(), storage);
        let cfg = super::super::resolve_checkpoint(&one).unwrap();
        storage
            .validate_headers(
                &cfg.dit,
                &safetensors_path_tensor_headers(one.join(DIT_WEIGHTS_FILE)).unwrap(),
            )
            .unwrap();
        // One packed target reconstructs its dense source within the affine bound.
        let out = SafetensorsSource::open(&one.join(DIT_WEIGHTS_FILE)).unwrap();
        let src = SafetensorsSource::open(&source.root.join(DIT_WEIGHTS_FILE)).unwrap();
        let target = "single_blocks.0.q_proj";
        let [n, k] = src.get(&format!("{target}.weight")).unwrap().shape[..] else {
            panic!("matrix")
        };
        let words = |name: &str, width: usize| -> Vec<u8> {
            let bytes = out.read(name).unwrap();
            assert_eq!(bytes.len() % width, 0);
            bytes
        };
        let codes = words(&format!("{target}.weight"), 4);
        let scales = words(&format!("{target}.scales"), 2);
        let biases = words(&format!("{target}.biases"), 2);
        let packed = crate::affine::AffinePacked {
            codes: codes
                .chunks_exact(4)
                .map(|b| u32::from_le_bytes([b[0], b[1], b[2], b[3]]))
                .collect(),
            scales: scales
                .chunks_exact(2)
                .map(|b| u16::from_le_bytes([b[0], b[1]]))
                .collect(),
            biases: biases
                .chunks_exact(2)
                .map(|b| u16::from_le_bytes([b[0], b[1]]))
                .collect(),
        };
        let back = crate::affine::dequantize(&packed, n, k, bits, group).unwrap();
        let tensor = src.get(&format!("{target}.weight")).unwrap();
        let dense = crate::affine::decode_dense(
            &tensor.dtype,
            &src.read(&format!("{target}.weight")).unwrap(),
        )
        .unwrap();
        let peak = dense.iter().fold(0f32, |m, v| m.max(v.abs()));
        let err = back
            .iter()
            .zip(&dense)
            .fold(0f32, |m, (a, b)| m.max((a - b).abs()));
        let bound = if bits == 4 { 0.2 } else { 0.02 };
        assert!(err <= bound * peak, "q{bits}: {err} vs peak {peak}");
        // Non-targets are byte-identical copies.
        assert_eq!(
            out.read("txt_in.weight").unwrap(),
            src.read("txt_in.weight").unwrap()
        );
    }
}

#[test]
fn chains_publisher_targets_cancellation_and_existing_outputs_are_refused() {
    let dir = tempfile::tempdir().unwrap();
    let source = source_copy(dir.path());
    let cancel = CancelFlag::new();
    cancel.cancel();
    let out = dir.path().join("cancelled");
    assert!(matches!(
        convert_affine(&source, 4, 64, &out, &cancel),
        Err(Error::Canceled)
    ));
    assert!(!out.exists());
    let first = dir.path().join("q8");
    convert_affine(&source, 8, 64, &first, &CancelFlag::new()).unwrap();
    let packed = ConversionSource {
        root: first.clone(),
        ..source.clone()
    };
    assert!(convert_affine(
        &packed,
        4,
        64,
        &dir.path().join("chain"),
        &CancelFlag::new()
    )
    .is_err());
    assert!(convert_affine(&source, 8, 64, &first, &CancelFlag::new()).is_err());
    assert!(convert_native(
        &source,
        NativeDitStorage::Publisher(DitFormat::Fp8Rowwise),
        &dir.path().join("fp8"),
        &CancelFlag::new(),
        |_, _, _| unreachable!()
    )
    .is_err());
    // A pack that omits a companion is refused before publication.
    let bad = dir.path().join("bad");
    assert!(convert_native(
        &source,
        NativeDitStorage::MlxAffine {
            bits: 4,
            group_size: 64
        },
        &bad,
        &CancelFlag::new(),
        |target, _, _| Ok(Packed::from([(target.to_owned(), vec![0u8; 4])]))
    )
    .is_err());
    assert!(!bad.exists());
}

#[cfg(unix)]
fn link_file(target: &Path, link: &Path) -> std::io::Result<()> {
    std::os::unix::fs::symlink(target, link)
}
#[cfg(windows)]
fn link_file(target: &Path, link: &Path) -> std::io::Result<()> {
    std::os::windows::fs::symlink_file(target, link)
}

/// A Hugging Face cache snapshot links every file to a content-addressed blob. Conversion follows
/// those FILE links, hashes the real bytes, and produces the same receipt as a plain copy; the
/// external components verify through the links too.
///
/// *Mutations that red this:* hashing with the strict `hash_file` (refuses links), or recording
/// the link target path instead of the package-relative name.
#[test]
fn symlinked_snapshot_converts_like_a_plain_copy() {
    let dir = tempfile::tempdir().unwrap();
    let plain = source_copy(dir.path());
    let blobs = plain.root.clone();
    let snapshot = dir.path().join("snapshot");
    std::fs::create_dir_all(snapshot.join("text_encoder")).unwrap();
    let mut files = vec![
        "config.json".to_string(),
        "model.safetensors".into(),
        "ae.safetensors".into(),
        "LICENSE.md".into(),
        "NOTICE.md".into(),
    ];
    for entry in std::fs::read_dir(blobs.join("text_encoder")).unwrap() {
        files.push(format!(
            "text_encoder/{}",
            entry.unwrap().file_name().to_string_lossy()
        ));
    }
    for name in &files {
        if let Err(e) = link_file(&blobs.join(name), &snapshot.join(name)) {
            if cfg!(windows) {
                eprintln!("SKIP: this Windows host cannot create symlinks ({e}); Linux CI runs it");
                return;
            }
            panic!("symlink {name}: {e}");
        }
    }
    let linked = ConversionSource {
        root: snapshot.clone(),
        ..plain.clone()
    };
    let from_plain =
        convert_affine(&plain, 4, 64, &dir.path().join("plain"), &CancelFlag::new()).unwrap();
    let from_links = convert_affine(
        &linked,
        4,
        64,
        &dir.path().join("links"),
        &CancelFlag::new(),
    )
    .unwrap();
    assert_eq!(from_plain, from_links);
    from_links.verify_external(&snapshot).unwrap();
    // A linked DIRECTORY is still refused: it could escape the package.
    let escape = dir.path().join("escape");
    std::fs::create_dir_all(&escape).unwrap();
    for name in [
        "config.json",
        "model.safetensors",
        "ae.safetensors",
        "LICENSE.md",
        "NOTICE.md",
    ] {
        std::fs::copy(blobs.join(name), escape.join(name)).unwrap();
    }
    #[cfg(unix)]
    std::os::unix::fs::symlink(blobs.join("text_encoder"), escape.join("text_encoder")).unwrap();
    #[cfg(windows)]
    if std::os::windows::fs::symlink_dir(blobs.join("text_encoder"), escape.join("text_encoder"))
        .is_err()
    {
        return;
    }
    let escaped = ConversionSource {
        root: escape,
        ..plain
    };
    assert!(convert_affine(
        &escaped,
        4,
        64,
        &dir.path().join("escaped"),
        &CancelFlag::new()
    )
    .is_err());
}

#[test]
fn base_conversion_preserves_absent_metadata_and_binds_receipt_to_header_role() {
    let temp = tempfile::tempdir().unwrap();
    let source = source_copy_kind(temp.path(), true);
    let output = temp.path().join("base-q4");
    let receipt = convert_affine(&source, 4, 64, &output, &CancelFlag::new()).unwrap();
    let metadata = crate::Flux3MetadataConfig::from_dir(&output).unwrap();
    assert_eq!(
        metadata.dit.flow_map, None,
        "conversion must not fabricate an explicit base flag"
    );
    let resolved = crate::resolve_checkpoint(&output).unwrap();
    resolved.validate_route(crate::BASE_MODEL_ID).unwrap();
    assert_eq!(receipt.source_repository, crate::BASE_WEIGHTS_REPO);
    assert_eq!(open_conversion(&output).unwrap(), receipt);
    let mut mislabeled = receipt;
    mislabeled.source_repository = crate::DISTILLED_WEIGHTS_REPO.into();
    mislabeled.source_revision = crate::DISTILLED_WEIGHTS_REVISION.into();
    for component in &mut mislabeled.components {
        component.source_repository = mislabeled.source_repository.clone();
        component.source_revision = mislabeled.source_revision.clone();
    }
    mislabeled.validate().unwrap(); // Repo/revision are a valid pair, but the WRONG header role.
    std::fs::write(
        output.join(crate::conversion::RECEIPT_FILE),
        serde_json::to_vec(&mislabeled.to_json().unwrap()).unwrap(),
    )
    .unwrap();
    assert!(
        matches!(open_conversion(&output),Err(Error::Unsupported(message)) if message.contains("flow_map=false") && message.contains(crate::BASE_WEIGHTS_REPO))
    );
}

#[test]
fn conversion_refuses_wrong_source_role_before_pack_or_publication() {
    let temp = tempfile::tempdir().unwrap();
    let mut source = source_copy(temp.path());
    source.repository = crate::BASE_WEIGHTS_REPO.into();
    source.revision = crate::BASE_WEIGHTS_REVISION.into();
    let output = temp.path().join("never-published");
    let calls = std::cell::Cell::new(0);
    let result = convert_native(
        &source,
        NativeDitStorage::MlxAffine {
            bits: 4,
            group_size: 64,
        },
        &output,
        &CancelFlag::new(),
        |_, _, _| {
            calls.set(calls.get() + 1);
            panic!("mode/repository must be checked before any tensor packing")
        },
    );
    assert!(
        matches!(result,Err(Error::Unsupported(message)) if message.contains("flow_map=true") && message.contains(crate::DISTILLED_WEIGHTS_REPO))
    );
    assert_eq!(calls.get(), 0);
    assert!(!output.exists());
}
