//! W4A8 external-fixture comparison (sc-26685) on the MLX/Metal lane.
//!
//! Expected values come from the **pinned external comfy-kitchen executable** (its CPU kernel replay
//! of `w4a8_gemm.cu` / `cutlass_gemm_int8.cu` / `int8_linear.cu`), never a shared host oracle. This
//! drives the production MLX packed projection against that JSON's full `output`, reached through the
//! real `Weights` reader + `quant::lin` (the same path a published checkpoint takes). The fixture's
//! optional provenance fields are ignored.
use mlx_gen::weights::Weights;
use mlx_gen_minimax_h3::quant;
use mlx_rs::{transforms::eval, Array, Dtype};
use serde_json::Value;

fn fixture() -> Value {
    serde_json::from_str(include_str!(concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/../../../../tests/fixtures/minimax_h3_w4a8.json"
    )))
    .unwrap()
}
fn floats(v: &Value) -> Vec<f32> {
    serde_json::from_value(v.clone()).unwrap()
}
fn unsigned(v: &Value) -> Vec<u8> {
    serde_json::from_value(v.clone()).unwrap()
}

/// The MLX reader maps `F8_E4M3` to `uint8` (its safetensors ABI has no fp8 storage dtype), so the
/// `s_rel` bytes reach `from_weights` as U8 — reproduced here by serializing the raw bytes directly.
/// The packed weight is the original I8 pattern (two int4 codes per byte).
#[test]
#[ignore = "requires coordinator-reserved exclusive local Metal token; never skip-as-pass"]
fn native_forward_matches_the_external_reference() {
    let fx = fixture();
    assert_eq!(fx["reference"]["noDenseFallback"], true, "reference must declare no dense fallback");
    for case in fx["cases"].as_array().unwrap() {
        let (m, n, k) = (
            case["m"].as_u64().unwrap() as i32,
            case["n"].as_u64().unwrap() as i32,
            case["k"].as_u64().unwrap() as i32,
        );
        let packed: Vec<i8> = unsigned(&case["packedWeight"]).into_iter().map(|b| b as i8).collect();
        let desc = serde_json::to_vec(&serde_json::json!({
            "convrot_groupsize": 256,
            "format": "asym_w4a8_int8",
            "group_size": 16,
        }))
        .unwrap();
        let mut parts: Vec<(String, Array)> = vec![
            ("p.weight".into(), Array::from_slice(&packed, &[n, k / 2])),
            ("p.weight_codebook".into(), Array::from_slice(&floats(&case["codebook"]), &[16])),
            ("p.weight_s_rel".into(), Array::from_slice(&unsigned(&case["sRelBits"]), &[n, k / 16])),
            ("p.weight_s_channel".into(), Array::from_slice(&floats(&case["sChannel"]), &[n])),
            (
                "p.comfy_quant".into(),
                Array::from_slice(&desc, &[desc.len() as i32]),
            ),
        ];
        if let Some(bias) = case.get("bias").filter(|b| !b.is_null()) {
            parts.push(("p.bias".into(), Array::from_slice(&floats(bias), &[n])));
        }
        // Serialize and reload through the production Weights reader, exactly like a published file.
        let dir = tempfile::tempdir().unwrap();
        let file = dir.path().join("w4a8.safetensors");
        Array::save_safetensors(parts.iter().map(|(k, v)| (k.as_str(), v)), None, &file).unwrap();
        let w = Weights::from_file(&file).unwrap();
        let linear = quant::lin(&w, "p", case.get("bias").map_or(false, |b| !b.is_null())).unwrap();
        assert!(linear.w4a8_base().is_some(), "case {} must be W4A8", case["id"]);
        assert!(linear.convrot_base().is_none(), "case {} is W4A8, not ConvRot", case["id"]);
        assert!(linear.dense_weight().is_none(), "case {} must not be a dense fallback", case["id"]);
        drop(w);
        // The H3 hidden state is BF16; `compute_dtype` is Bfloat16 for W4A8, and the external
        // reference rounds the store to BF16 once (`single BF16 RN at the store`).
        let x = Array::from_slice(&floats(&case["x"]), &[m, k]).as_dtype(Dtype::Bfloat16).unwrap();
        let out = linear.forward(&x).unwrap().as_dtype(Dtype::Float32).unwrap();
        eval([&out]).unwrap();
        assert_eq!(out.as_slice::<f32>(), floats(&case["expected"]["output"]).as_slice(), "case {} forward", case["id"]);
    }
}