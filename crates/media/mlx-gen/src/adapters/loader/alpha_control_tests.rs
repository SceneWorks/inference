use super::read_control_scalar_f32;
fn tensor(
    key: &str,
    root: &std::path::Path,
    dtype: &str,
    shape: &[usize],
    bytes: &[u8],
) -> std::path::PathBuf {
    let path = root.join("alpha.safetensors");
    let mut entries = serde_json::Map::new();
    entries.insert(
        key.into(),
        serde_json::json!({"dtype":dtype,"shape":shape,"data_offsets":[0,bytes.len()]}),
    );
    let mut header = serde_json::to_vec(&entries).unwrap();
    while !header.len().is_multiple_of(8) {
        header.push(b' ');
    }
    let mut file = (header.len() as u64).to_le_bytes().to_vec();
    file.extend(header);
    file.extend_from_slice(bytes);
    std::fs::write(&path, file).unwrap();
    path
}
fn control(
    root: &std::path::Path,
    dtype: &str,
    shape: &[usize],
    bytes: &[u8],
) -> std::path::PathBuf {
    tensor("module.alpha", root, dtype, shape, bytes)
}
#[test]
fn alpha_control_reader_matches_native_f32_zero_and_nonzero_semantics() {
    let root = tempfile::tempdir().unwrap();
    for (dtype, bytes, expected) in [
        ("F32", 0.0f32.to_le_bytes().to_vec(), 0.0f32),
        ("F32", 1.5f32.to_le_bytes().to_vec(), 1.5f32),
        ("BF16", 0x3fc0u16.to_le_bytes().to_vec(), 1.5),
        ("F16", 1u16.to_le_bytes().to_vec(), 2.0_f32.powi(-24)),
        ("F16", 0x3e00u16.to_le_bytes().to_vec(), 1.5),
        ("BOOL", vec![1], 1.0),
        ("I8", (-3i8).to_le_bytes().to_vec(), -3.0),
        ("U8", vec![4], 4.0),
        ("I16", (-123i16).to_le_bytes().to_vec(), -123.0),
        ("U16", 321u16.to_le_bytes().to_vec(), 321.0),
        ("I32", (-12345i32).to_le_bytes().to_vec(), -12345.0),
        ("U32", 123456u32.to_le_bytes().to_vec(), 123456.0),
        ("I64", (-3i64).to_le_bytes().to_vec(), -3.0),
        ("U64", 1234567u64.to_le_bytes().to_vec(), 1234567.0),
        // Native load maps these to U8; it does NOT opt into floating-point fp8 decoding.
        ("F8_E4M3", vec![128], 128.0),
        ("F8_E8M0", vec![0], 0.0),
    ] {
        let path = control(root.path(), dtype, &[1], &bytes);
        assert_eq!(
            read_control_scalar_f32(&path, "module.alpha")
                .unwrap()
                .to_bits(),
            expected.to_bits()
        );
    }
}
#[test]
fn alpha_control_reader_preserves_core_first_element_and_alpha_only_selector() {
    let root = tempfile::tempdir().unwrap();
    let mut values = 0.5f32.to_le_bytes().to_vec();
    values.extend_from_slice(&2.0f32.to_le_bytes());
    values.extend_from_slice(&3.0f32.to_le_bytes());
    let path = control(root.path(), "F32", &[3], &values);
    assert!(matches!(read_control_scalar_f32(&path, "module.alpha"), Ok(value) if value == 0.5));
    let forbidden = tensor(
        "module.weight",
        root.path(),
        "F32",
        &[1],
        &1.5f32.to_le_bytes(),
    );
    assert!(read_control_scalar_f32(&forbidden, "module.weight").is_err());
}
#[test]
fn alpha_control_reader_rejects_empty_unsupported_and_truncated_controls() {
    let root = tempfile::tempdir().unwrap();
    let empty = control(root.path(), "F32", &[0], &[]);
    assert!(read_control_scalar_f32(&empty, "module.alpha").is_err());
    for (dtype, bytes) in [
        ("F64", 1.0f64.to_le_bytes().to_vec()),
        ("F8_E5M2", vec![1]),
        ("C64", vec![0; 8]),
    ] {
        let unsupported = control(root.path(), dtype, &[1], &bytes);
        assert!(read_control_scalar_f32(&unsupported, "module.alpha").is_err());
    }
    let truncated = control(root.path(), "F32", &[1], &[0u8; 3]);
    assert!(read_control_scalar_f32(&truncated, "module.alpha").is_err());
}
