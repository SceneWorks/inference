//! Explicit Metal proof of source-native projections in both Iris backbone lifecycles.

use std::collections::BTreeMap;
use std::path::Path;

use mlx_gen::gen_core::checkpoint_codec::FP8_E4M3_SCALAR_CODEC;
use mlx_gen::gen_core::checkpoint_facts::ExecutionRepresentation;
use mlx_gen::gen_core::{convrot_int8, CheckpointReceiptScope};
use mlx_rs::{DeviceType, Stream};
use serde_json::{json, Value};
use sha2::{Digest, Sha256};

use super::{backbone_dir, exec, host, Case, Dtype, LoadShape, LoadSpec, TextBatch, WeightsSource};
use crate::IrisDiT;

const FP8: [&str; 2] = ["t_embedder.mlp.2.weight", "blocks.0.attn.q_proj_x.weight"];
const CONVROT: [&str; 2] = [
    "modulation_cores.adaln_img.weight",
    "blocks.1.attn.v_proj_y.weight",
];
const FP8_CODES: [u8; 7] = [0, 0x18, 0xa0, 0x28, 0xa8, 0x20, 0x98];

fn push_tensor(
    headers: &mut BTreeMap<String, Value>,
    payload: &mut Vec<u8>,
    name: String,
    dtype: &str,
    shape: Value,
    bytes: &[u8],
) {
    let start = payload.len();
    payload.extend_from_slice(bytes);
    assert!(headers
        .insert(
            name,
            json!({"dtype": dtype, "shape": shape, "data_offsets": [start, payload.len()]}),
        )
        .is_none());
}

/// Keep every other source payload intact. The replacement codes are a fixed, signed codebook
/// with non-unit scales, since this test witnesses constructor/forward wiring, not conversion.
fn write_native_backbone(original: &Path, output: &Path) -> (String, String) {
    let source = std::fs::read(original).unwrap();
    let prefix = 8 + u64::from_le_bytes(source[..8].try_into().unwrap()) as usize;
    let original_headers: BTreeMap<String, Value> =
        serde_json::from_slice(&source[8..prefix]).unwrap();
    let mut headers = BTreeMap::new();
    let mut payload = Vec::new();
    let mut replaced = 0;
    for (name, entry) in original_headers {
        if name == "__metadata__" {
            headers.insert(name, entry);
            continue;
        }
        let dtype = entry["dtype"].as_str().unwrap();
        let shape = entry["shape"].clone();
        let offsets = entry["data_offsets"].as_array().unwrap();
        let start = offsets[0].as_u64().unwrap() as usize;
        let end = offsets[1].as_u64().unwrap() as usize;
        if !FP8.contains(&name.as_str()) && !CONVROT.contains(&name.as_str()) {
            push_tensor(
                &mut headers,
                &mut payload,
                name,
                dtype,
                shape,
                &source[prefix + start..prefix + end],
            );
            continue;
        }

        assert_eq!(dtype, "F32");
        let [rows, cols] = shape.as_array().unwrap().as_slice() else {
            panic!("encoded fixture projection {name} must be rank two");
        };
        let rows = rows.as_u64().unwrap() as usize;
        let cols = cols.as_u64().unwrap() as usize;
        assert_eq!(cols, 64);
        let base = name.strip_suffix(".weight").unwrap();
        let encoded_fp8 = FP8.contains(&name.as_str());
        let (encoding, codes, descriptor, scale_shape, scale) = if encoded_fp8 {
            (
                "F8_E4M3",
                (0..rows * cols)
                    .map(|i| FP8_CODES[(i + i / cols) % FP8_CODES.len()])
                    .collect::<Vec<_>>(),
                br#"{"format":"float8_e4m3fn"}"#.as_slice(),
                json!([]),
                0.125f32.to_le_bytes().to_vec(),
            )
        } else {
            (
                "I8",
                (0..rows * cols)
                    .map(|i| ((i % 23) as i8 - 11) as u8)
                    .collect::<Vec<_>>(),
                br#"{"format":"int8_tensorwise","convrot":true,"per_row":true,"convrot_groupsize":64}"#
                    .as_slice(),
                json!([rows, 1]),
                (0..rows)
                    .flat_map(|row| ((1 + row % 3) as f32 / 256.).to_le_bytes())
                    .collect::<Vec<_>>(),
            )
        };
        push_tensor(
            &mut headers,
            &mut payload,
            format!("{base}.weight_scale"),
            "F32",
            scale_shape,
            &scale,
        );
        push_tensor(
            &mut headers,
            &mut payload,
            format!("{base}.comfy_quant"),
            "U8",
            json!([descriptor.len()]),
            descriptor,
        );
        if encoded_fp8 {
            push_tensor(
                &mut headers,
                &mut payload,
                format!("{base}.input_scale"),
                "F32",
                json!([]),
                &0.0625f32.to_le_bytes(),
            );
        }
        push_tensor(&mut headers, &mut payload, name, encoding, shape, &codes);
        replaced += 1;
    }
    assert_eq!(replaced, FP8.len() + CONVROT.len());

    let mut header = serde_json::to_vec(&headers).unwrap();
    header.resize(header.len().next_multiple_of(8), b' ');
    let mut encoded = (header.len() as u64).to_le_bytes().to_vec();
    encoded.extend(header);
    encoded.extend(payload);
    std::fs::write(output, &encoded).unwrap();
    (
        format!("{:x}", Sha256::digest(source)),
        format!("{:x}", Sha256::digest(encoded)),
    )
}

fn forward(
    case: &Case,
    dit: &IrisDiT,
    scope: &CheckpointReceiptScope,
    window: Option<usize>,
) -> Vec<f32> {
    scope.with(|| {
        super::metal_guard::record_model_inference_device();
        let output = dit
            .forward_exec(
                &case.x,
                &case.t,
                &TextBatch {
                    states: &case.states,
                    mask: &case.mask,
                },
                exec(window, false),
            )
            .unwrap();
        assert_eq!(output.shape(), case.x.shape());
        let pixels = host(&output);
        assert!(pixels.iter().all(|pixel| pixel.is_finite()));
        assert!(pixels.iter().any(|pixel| *pixel != pixels[0]));
        pixels
    })
}

fn assert_native_receipt(scope: &CheckpointReceiptScope, path: &Path) {
    let facts = scope.try_snapshots().unwrap();
    assert_eq!(facts.len(), 1, "the backbone is one physical checkpoint");
    let facts = &facts[0];
    let binding = facts.source_binding().expect("verified fixture binding");
    assert_eq!(
        binding.canonical_path(),
        std::fs::canonicalize(path).unwrap().as_path()
    );
    for (codec, bytes) in [
        (FP8_E4M3_SCALAR_CODEC.codec_id, 2 * (64 * 64 + 4 + 4)),
        (convrot_int8::CODEC_ID, (384 + 32) * (64 + 4)),
    ] {
        assert!(facts.capability().executes_natively(codec));
        let row = facts
            .materialized_as(codec, ExecutionRepresentation::NativePacked)
            .expect("the forward must observe both source formats");
        assert_eq!(row.tensor_count, 2, "{codec}: both configured projections");
        assert_eq!(
            row.resident_bytes, bytes,
            "{codec}: codes and publisher scales"
        );
        assert!(facts
            .materialized_as(codec, ExecutionRepresentation::DenseFallback)
            .is_none());
    }
    assert!(facts.receipt().demotions.is_empty());
}

#[test]
#[ignore = "model inference; explicit model-proof workflow"]
fn source_native_fp8_and_convrot_match_eager_and_windowed_on_metal() {
    assert!(mlx_gen::metal_available());
    mlx_rs::with_new_default_stream(Stream::gpu(), || {
        assert!(matches!(
            Stream::task_local_or_default()
                .get_device()
                .unwrap()
                .get_type()
                .unwrap(),
            DeviceType::Gpu
        ));

        let original = backbone_dir();
        let temp = tempfile::tempdir().unwrap();
        std::fs::copy(
            original.join("config.yaml"),
            temp.path().join("config.yaml"),
        )
        .unwrap();
        let file = temp.path().join(gen_core_iris::BACKBONE_WEIGHTS_FILE);
        let (original_digest, fixture_digest) =
            write_native_backbone(&original.join(gen_core_iris::BACKBONE_WEIGHTS_FILE), &file);
        let config = gen_core_iris::IrisConfig::from_dir(temp.path()).unwrap();
        let case = Case::new(&config);
        let scope = |shape| {
            let spec = LoadSpec::new(WeightsSource::File(file.clone())).with_load_shape(shape);
            CheckpointReceiptScope::for_load(&spec).unwrap()
        };
        let eager_scope = scope(LoadShape::EagerMaterialization);
        let eager = eager_scope
            .with(|| crate::model::load_backbone(temp.path(), &config, Dtype::Bfloat16).unwrap());
        assert!(!eager.is_streamed());
        let retained = eager.arrays();
        for (dtype, shape, count) in [
            (Dtype::Uint8, [64, 64], 2),
            (Dtype::Int8, [384, 64], 1),
            (Dtype::Int8, [32, 64], 1),
        ] {
            assert_eq!(
                retained
                    .iter()
                    .filter(|array| array.dtype() == dtype && array.shape() == shape)
                    .count(),
                count,
                "eager model must retain its encoded operand arrays"
            );
        }
        let expected = forward(&case, &eager, &eager_scope, None);
        assert_native_receipt(&eager_scope, &file);

        let streamed_scope = scope(LoadShape::DeferredMaterialization);
        let streamed = streamed_scope.with(|| {
            crate::model::load_backbone_streamed(temp.path(), &config, Dtype::Bfloat16, None, None)
                .unwrap()
        });
        assert!(streamed.is_streamed());
        let skeleton = streamed.arrays();
        assert_eq!(
            skeleton
                .iter()
                .filter(|array| array.dtype() == Dtype::Uint8 && array.shape() == [64, 64])
                .count(),
            1,
            "streamed model retains the skeleton FP8 projection only"
        );
        assert!(skeleton
            .iter()
            .any(|array| array.dtype() == Dtype::Int8 && array.shape() == [384, 64]));
        assert!(!skeleton
            .iter()
            .any(|array| array.dtype() == Dtype::Int8 && array.shape() == [32, 64]));
        for window in [Some(1), Some(2), None] {
            assert_eq!(
                forward(&case, &streamed, &streamed_scope, window),
                expected,
                "same source-native operators at window {window:?}"
            );
            assert_native_receipt(&streamed_scope, &file);
        }
        eprintln!("Iris native checkpoint proof: original_sha256={original_digest} encoded_sha256={fixture_digest} BF16 2x3x16x24 windows=1,2,full");
        eprintln!("\nMODEL_INFERENCE_BODY_COMPLETED stream_tests::native_formats::source_native_fp8_and_convrot_match_eager_and_windowed_on_metal");
    });
}
