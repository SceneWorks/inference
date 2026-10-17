use super::*;
use crate::flux3::conversion::open_conversion;

fn fixture() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("../../media/mlx-gen/mlx-gen-flux3/tests/fixtures/tiny-snapshot")
}

/// A private copy of the committed tiny dense package plus the license files a real one ships.
fn source_copy(dir: &Path) -> ConversionSource {
    let root = dir.join("source");
    std::fs::create_dir_all(root.join("text_encoder")).unwrap();
    for entry in ["config.json", "model.safetensors", "ae.safetensors"] {
        std::fs::copy(fixture().join(entry), root.join(entry)).unwrap();
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
        repository: crate::flux3::DISTILLED_WEIGHTS_REPO.into(),
        revision: crate::flux3::DISTILLED_WEIGHTS_REVISION.into(),
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
        let cfg = Flux3Config::from_dir(&one).unwrap();
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
        let packed = crate::flux3::affine::AffinePacked {
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
        let back = crate::flux3::affine::dequantize(&packed, n, k, bits, group).unwrap();
        let tensor = src.get(&format!("{target}.weight")).unwrap();
        let dense = crate::flux3::affine::decode_dense(
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
