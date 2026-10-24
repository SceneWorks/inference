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
    assert_eq!(
        fx["reference"]["noDenseFallback"], true,
        "reference must declare no dense fallback"
    );
    for case in fx["cases"].as_array().unwrap() {
        let (m, n, k) = (
            case["m"].as_u64().unwrap() as i32,
            case["n"].as_u64().unwrap() as i32,
            case["k"].as_u64().unwrap() as i32,
        );
        let packed: Vec<i8> = unsigned(&case["packedWeight"])
            .into_iter()
            .map(|b| b as i8)
            .collect();
        let desc = serde_json::to_vec(&serde_json::json!({
            "convrot_groupsize": 256,
            "format": "asym_w4a8_int8",
            "group_size": 16,
        }))
        .unwrap();
        let mut parts: Vec<(String, Array)> = vec![
            ("p.weight".into(), Array::from_slice(&packed, &[n, k / 2])),
            (
                "p.weight_codebook".into(),
                Array::from_slice(&floats(&case["codebook"]), &[16]),
            ),
            (
                "p.weight_s_rel".into(),
                Array::from_slice(&unsigned(&case["sRelBits"]), &[n, k / 16]),
            ),
            (
                "p.weight_s_channel".into(),
                Array::from_slice(&floats(&case["sChannel"]), &[n]),
            ),
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
        assert!(
            linear.w4a8_base().is_some(),
            "case {} must be W4A8",
            case["id"]
        );
        assert!(
            linear.convrot_base().is_none(),
            "case {} is W4A8, not ConvRot",
            case["id"]
        );
        assert!(
            linear.dense_weight().is_none(),
            "case {} must not be a dense fallback",
            case["id"]
        );
        drop(w);
        // The H3 hidden state is BF16; `compute_dtype` is Bfloat16 for W4A8, and the external
        // reference rounds the store to BF16 once (`single BF16 RN at the store`).
        let x = Array::from_slice(&floats(&case["x"]), &[m, k])
            .as_dtype(Dtype::Bfloat16)
            .unwrap();
        let out = linear
            .forward(&x)
            .unwrap()
            .as_dtype(Dtype::Float32)
            .unwrap();
        eval([&out]).unwrap();
        assert_eq!(
            out.as_slice::<f32>(),
            floats(&case["expected"]["output"]).as_slice(),
            "case {} forward",
            case["id"]
        );
    }
}

/// The operand sets the published contract refuses — the negative/NaN `s_rel` bytes and the zero
/// channel scale the old success case wrongly carried. Driven through the production loader.
#[test]
#[ignore = "requires coordinator-reserved exclusive local Metal token; never skip-as-pass"]
fn invalid_operands_are_refused_by_the_published_contract() {
    let fx = fixture();
    let refusals = fx["refusedOperands"]
        .as_array()
        .expect("fixture must pin expected refusals");
    assert!(!refusals.is_empty());
    for case in refusals {
        let (n, k) = (
            case["n"].as_u64().unwrap() as i32,
            case["k"].as_u64().unwrap() as i32,
        );
        let packed: Vec<i8> = unsigned(&case["packedWeight"])
            .into_iter()
            .map(|b| b as i8)
            .collect();
        let desc = serde_json::to_vec(&serde_json::json!({
            "convrot_groupsize": 256,
            "format": "asym_w4a8_int8",
            "group_size": 16,
        }))
        .unwrap();
        let mut parts: Vec<(String, Array)> = vec![
            ("p.weight".into(), Array::from_slice(&packed, &[n, k / 2])),
            (
                "p.weight_codebook".into(),
                Array::from_slice(&floats(&case["codebook"]), &[16]),
            ),
            (
                "p.weight_s_rel".into(),
                Array::from_slice(&unsigned(&case["sRelBits"]), &[n, k / 16]),
            ),
            (
                "p.weight_s_channel".into(),
                Array::from_slice(&floats(&case["sChannel"]), &[n]),
            ),
            (
                "p.comfy_quant".into(),
                Array::from_slice(&desc, &[desc.len() as i32]),
            ),
        ];
        if case.get("bias").is_some_and(|b| !b.is_null()) {
            parts.push((
                "p.bias".into(),
                Array::from_slice(&floats(&case["bias"]), &[n]),
            ));
        }
        let dir = tempfile::tempdir().unwrap();
        let file = dir.path().join("refused.safetensors");
        Array::save_safetensors(parts.iter().map(|(k, v)| (k.as_str(), v)), None, &file).unwrap();
        let w = Weights::from_file(&file).expect("reader");
        let err = quant::lin(&w, "p", true)
            .err()
            .unwrap_or_else(|| panic!("case {} must be refused", case["id"]))
            .to_string();
        let expected = case["expectedError"].as_str().unwrap();
        assert!(
            err.contains(expected),
            "case {} must be refused with {expected:?}, got {err:?}",
            case["id"]
        );
    }
}

/// One published payload holding a W4A8 layer and a ConvRot mixed-layer operand set: each prefix
/// must keep its own format (no densification in either direction) and reproduce its own pinned
/// arithmetic — W4A8 against the comfy-kitchen PR#90 replay, ConvRot against the S4 fixture's
/// pinned activation with the cutlass int8 GEMM/epilogue.
#[test]
#[ignore = "requires coordinator-reserved exclusive local Metal token; never skip-as-pass"]
fn mixed_w4a8_and_convrot_layers_keep_their_own_arithmetic() {
    let fx = fixture();
    let mixed = &fx["mixedLayer"];
    let w4a8_id = mixed["w4a8Case"].as_str().unwrap();
    let w4a8 = fx["cases"]
        .as_array()
        .unwrap()
        .iter()
        .find(|c| c["id"] == w4a8_id)
        .expect("mixed-layer W4A8 case");
    let convrot = &mixed["convrot"];
    let (cm, cn, ck) = (
        convrot["m"].as_u64().unwrap() as i32,
        convrot["n"].as_u64().unwrap() as i32,
        convrot["k"].as_u64().unwrap() as i32,
    );
    let (pn, pk) = (
        w4a8["n"].as_i64().unwrap() as i32,
        w4a8["k"].as_i64().unwrap() as i32,
    );
    let pm = w4a8["m"].as_i64().unwrap() as i32;
    let packed: Vec<i8> = unsigned(&w4a8["packedWeight"])
        .into_iter()
        .map(|b| b as i8)
        .collect();
    let desc = serde_json::to_vec(&serde_json::json!({
        "convrot_groupsize": 256,
        "format": "asym_w4a8_int8",
        "group_size": 16,
    }))
    .unwrap();
    let q_codes: Vec<i8> = serde_json::from_value(convrot["weight"].clone()).unwrap();
    let q_desc = serde_json::to_vec(&serde_json::json!({
        "format": "int8_tensorwise",
        "convrot": true,
        "convrot_groupsize": 256,
    }))
    .unwrap();
    let mut parts: Vec<(String, Array)> = vec![
        ("p.weight".into(), Array::from_slice(&packed, &[pn, pk / 2])),
        (
            "p.weight_codebook".into(),
            Array::from_slice(&floats(&w4a8["codebook"]), &[16]),
        ),
        (
            "p.weight_s_rel".into(),
            Array::from_slice(&unsigned(&w4a8["sRelBits"]), &[pn, pk / 16]),
        ),
        (
            "p.weight_s_channel".into(),
            Array::from_slice(&floats(&w4a8["sChannel"]), &[pn]),
        ),
        (
            "p.comfy_quant".into(),
            Array::from_slice(&desc, &[desc.len() as i32]),
        ),
        ("q.weight".into(), Array::from_slice(&q_codes, &[cn, ck])),
        (
            "q.weight_scale".into(),
            Array::from_slice(&floats(&convrot["weightScale"]), &[cn, 1]),
        ),
        (
            "q.bias".into(),
            Array::from_slice(&floats(&convrot["bias"]), &[cn]),
        ),
        (
            "q.comfy_quant".into(),
            Array::from_slice(&q_desc, &[q_desc.len() as i32]),
        ),
    ];
    if w4a8.get("bias").is_some_and(|b| !b.is_null()) {
        parts.push((
            "p.bias".into(),
            Array::from_slice(&floats(&w4a8["bias"]), &[pn]),
        ));
    }
    let dir = tempfile::tempdir().unwrap();
    let file = dir.path().join("mixed.safetensors");
    Array::save_safetensors(parts.iter().map(|(k, v)| (k.as_str(), v)), None, &file).unwrap();
    let w = Weights::from_file(&file).expect("reader");

    let p = quant::lin(&w, "p", w4a8.get("bias").is_some_and(|b| !b.is_null())).expect("load p");
    assert!(p.w4a8_base().is_some(), "p must stay W4A8");
    assert!(p.convrot_base().is_none());
    assert!(p.dense_weight().is_none(), "p must not densify");
    let px = Array::from_slice(&floats(&w4a8["x"]), &[pm, pk])
        .as_dtype(Dtype::Bfloat16)
        .unwrap();
    let py = p.forward(&px).unwrap().as_dtype(Dtype::Float32).unwrap();
    eval([&py]).unwrap();
    assert_eq!(
        py.as_slice::<f32>(),
        floats(&w4a8["expected"]["output"]).as_slice(),
        "mixed-layer W4A8 half"
    );

    let q = quant::lin(&w, "q", true).expect("load q");
    assert!(q.convrot_base().is_some(), "q must stay ConvRot INT8");
    assert!(q.w4a8_base().is_none());
    assert!(q.dense_weight().is_none(), "q must not densify");
    let qx = Array::from_slice(&floats(&convrot["input"]), &[cm, ck]);
    let qy = q.forward(&qx).unwrap().as_dtype(Dtype::Float32).unwrap();
    eval([&qy]).unwrap();
    assert_eq!(
        qy.as_slice::<f32>(),
        floats(&convrot["expected"]).as_slice(),
        "mixed-layer ConvRot half"
    );
}
