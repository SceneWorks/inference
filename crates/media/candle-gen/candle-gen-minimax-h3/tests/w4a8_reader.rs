//! W4A8 reader conformance (sc-26685): the native `from_files_native_int8` path admits the frozen
//! `asym_w4a8_int8` dialect and preserves the packed I8, F8_E4M3 `s_rel`, F32 codebook and F32
//! `s_channel` companions — the source format a packed W4A8 checkpoint actually stores.
use candle_gen::{candle_core::Device, Weights};
use safetensors::{tensor::TensorView, serialize_to_file, Dtype};
use serde_json::json;

/// out=16, in=512 (two ConvRot groups): packed `[16,256]`, s_rel F8 `[16,32]`, codebook F32[16].
fn write_parts(path: &std::path::Path, s_rel_bytes: Vec<u8>, descriptor: serde_json::Value) {
    let n = 16usize;
    let k = 512usize;
    let packed: Vec<u8> = (0..n * (k / 2)).map(|i| (i % 256) as u8).collect();
    let codebook: Vec<u8> = (0..16usize)
        .flat_map(|j| ((j as f32 / 16.0 - 0.5).to_le_bytes()))
        .collect();
    let channel: Vec<u8> = (0..n)
        .flat_map(|i| ((i + 1) as f32 / 128.0).to_le_bytes())
        .collect();
    let desc = serde_json::to_vec(&descriptor).unwrap();
    let tensors = std::collections::HashMap::from([
        ("p.weight", TensorView::new(Dtype::I8, vec![n, k / 2], &packed).unwrap()),
        (
            "p.weight_codebook",
            TensorView::new(Dtype::F32, vec![16], &codebook).unwrap(),
        ),
        (
            "p.weight_s_rel",
            TensorView::new(Dtype::F8_E4M3, vec![n, k / 16], &s_rel_bytes).unwrap(),
        ),
        (
            "p.weight_s_channel",
            TensorView::new(Dtype::F32, vec![n], &channel).unwrap(),
        ),
        ("p.comfy_quant", TensorView::new(Dtype::U8, vec![desc.len()], &desc).unwrap()),
    ]);
    serialize_to_file(tensors, &None, path).unwrap();
}

/// The frozen dialect descriptor, as a parsed JSON object (not a quoted string).
fn w4a8_descriptor() -> serde_json::Value {
    json!({"convrot_groupsize":256,"format":"asym_w4a8_int8","group_size":16})
}


#[test]
fn native_reader_admits_w4a8_and_preserves_the_packed_operands() {
    let dir = tempfile::tempdir().unwrap();
    let file = dir.path().join("w4a8.safetensors");
    // One valid E4M3 scale per group: 0x38 = 1.0.
    write_parts(&file, vec![0x38u8; 16 * 32], w4a8_descriptor());
    let w = Weights::from_files_native_int8(&[&file], &Device::Cpu).unwrap();
    assert_eq!(w.require("p.weight").unwrap().dtype(), candle_gen::candle_core::DType::U8);
    // candle loads F8_E4M3 natively (not U8): the reader must not corrupt the source dtype.
    assert!(matches!(
        w.require("p.weight_s_rel").unwrap().dtype(),
        candle_gen::candle_core::DType::F8E4M3
    ));
    assert_eq!(w.require("p.weight_codebook").unwrap().dtype(), candle_gen::candle_core::DType::F32);
    assert_eq!(w.require("p.weight_s_channel").unwrap().dtype(), candle_gen::candle_core::DType::F32);
}

#[test]
fn native_reader_refuses_malformed_w4a8_operands_before_any_tensor_exists() {
    let dir = tempfile::tempdir().unwrap();
    // A NaN E4M3 group scale (0x7F) must be refused by value, ahead of any projection/CUDA handle.
    let nan = dir.path().join("nan.safetensors");
    let mut bytes = vec![0x38u8; 16 * 32];
    bytes[0] = 0x7F;
    write_parts(&nan, bytes, w4a8_descriptor());
    assert!(Weights::from_files_native_int8(&[&nan], &Device::Cpu).is_err());

    // A negative group scale (0xA8 = -0.25) must refuse too.
    let neg = dir.path().join("neg.safetensors");
    let mut bytes = vec![0x38u8; 16 * 32];
    bytes[0] = 0xA8;
    write_parts(&neg, bytes, w4a8_descriptor());
    assert!(Weights::from_files_native_int8(&[&neg], &Device::Cpu).is_err());

    // A contradictory `convrot:true` bool is not part of the W4A8 dialect.
    let convrot = dir.path().join("convrot.safetensors");
    write_parts(
        &convrot,
        vec![0x38u8; 16 * 32],
        json!({"convrot":true,"convrot_groupsize":256,"format":"asym_w4a8_int8","group_size":16}),
    );
    assert!(Weights::from_files_native_int8(&[&convrot], &Device::Cpu).is_err());
}