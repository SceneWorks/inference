//! W4A8 external-fixture comparison (sc-26685).
//!
//! Expected values come from the **pinned external comfy-kitchen executable** (its CPU kernel replay
//! of `w4a8_gemm.cu` / `cutlass_gemm_int8.cu` / `int8_linear.cu`), never a shared host oracle. This
//! test (a) crosses the host packed decode against the external per-element `decodedWeightI8`, and
//! (b) on a CUDA device drives the production W4A8 forward and compares its full output against the
//! external `output`. The JSON's optional provenance fields are ignored.
use candle_gen::gen_core;
use candle_gen::{
    candle_core::{DType, Device, Tensor},
    Weights,
};
use candle_gen_minimax_h3::quant::{lin, DIT};
use safetensors::{serialize_to_file, tensor::TensorView, Dtype};
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
fn signed(v: &Value) -> Vec<i8> {
    serde_json::from_value(v.clone()).unwrap()
}

/// The host packed decode is a bit-identical transcription of the pinned CUDA decode, so it must
/// reproduce the external `decodedWeightI8` element for element.
#[test]
fn host_packed_decode_matches_the_external_reference() {
    let fx = fixture();
    assert_eq!(
        fx["reference"]["noDenseFallback"], true,
        "reference must declare no dense fallback"
    );
    for case in fx["cases"].as_array().unwrap() {
        let (n, k) = (
            case["n"].as_u64().unwrap() as usize,
            case["k"].as_u64().unwrap() as usize,
        );
        let packed: Vec<i8> = unsigned(&case["packedWeight"])
            .into_iter()
            .map(|b| b as i8)
            .collect();
        let mut got = Vec::new();
        gen_core::w4a8_int8::dequant_packed_to_int8(
            &packed,
            &unsigned(&case["sRelBits"]),
            &floats(&case["codebook"]),
            n,
            k,
            &mut got,
        )
        .expect("decode");
        assert_eq!(
            got,
            signed(&case["expected"]["decodedWeightI8"]),
            "case {}",
            case["id"]
        );
    }
}

/// The activation leg is the one numeric fork a CPU lane can still exercise end to end: the F32
/// fast-Hadamard butterfly, the row absmax/scale taken from its **unrounded** values, and
/// `quant_div_float_to_float<bf16>`'s three cast positions.
///
/// `convrot_fht_f32` is device-generic, so this drives the production rotation against the external
/// `rotatedF32`, `rowAbsmax` and `xScale` bit for bit, and re-derives `xQuantized` from the pinned
/// quantizer transcription below (the same shape `h3_convrot_int8.json`'s host test uses). A dense
/// `x·H` rotation — in F32 or in the source dtype — fails this on every case: the shuffled
/// additions move `rotatedF32`, and a BF16-rounded peak turns the review's `257/4096` into `0.0625`,
/// which selects different int8 codes.
#[test]
fn cpu_fht_and_row_scale_match_the_external_reference() {
    let fx = fixture();
    for case in fx["cases"].as_array().unwrap() {
        let (m, k) = (
            case["m"].as_u64().unwrap() as usize,
            case["k"].as_u64().unwrap() as usize,
        );
        let group = case["convrotGroup"].as_u64().unwrap() as usize;
        assert_eq!(group, 256, "case {} rotation group", case["id"]);
        let x = Tensor::from_vec(floats(&case["x"]), (m, k), &Device::Cpu)
            .unwrap()
            .to_dtype(DType::BF16)
            .unwrap();
        let rotated = candle_gen::quant::convrot_fht_f32(&x, group)
            .expect("production FHT")
            .flatten_all()
            .unwrap()
            .to_vec1::<f32>()
            .unwrap();
        assert_eq!(
            rotated,
            floats(&case["expected"]["rotatedF32"]),
            "case {} rotatedF32 (F32 butterfly order)",
            case["id"]
        );

        let scale_recip = 1.0f32 / 127.0;
        let max_finite = 3.3895314e38f32; // f32(3.38953139e38), the BF16 max
        let mut peaks = Vec::new();
        let mut scales = Vec::new();
        for row in rotated.chunks_exact(k) {
            let peak = row.iter().fold(0f32, |acc, v| acc.max(v.abs()));
            peaks.push(peak);
            scales.push((peak.min(max_finite) * scale_recip).max(1.0e-30f32));
        }
        assert_eq!(
            peaks,
            floats(&case["expected"]["rowAbsmax"]),
            "case {} rowAbsmax (measured before any dtype rounding)",
            case["id"]
        );
        assert_eq!(
            scales,
            floats(&case["expected"]["xScale"]),
            "case {} xScale",
            case["id"]
        );

        // int8_linear.cu `quant_div_float_to_float<bf16>`: numerator -> bf16, divisor -> bf16,
        // quotient -> bf16, `nearbyintf`, then the asymmetric [-128, 127] clamp.
        let round = |v: f32| half::bf16::from_f32(v).to_f32();
        let mut codes: Vec<i32> = Vec::with_capacity(m * k);
        for (row, scale) in rotated.chunks_exact(k).zip(&scales) {
            let divisor = round(*scale);
            for &value in row {
                let z = round(round(value) / divisor);
                codes.push(if z.is_nan() {
                    -128
                } else {
                    z.round_ties_even().clamp(-128.0, 127.0) as i32
                });
            }
        }
        assert_eq!(
            codes,
            serde_json::from_value::<Vec<i32>>(case["expected"]["xQuantized"].clone()).unwrap(),
            "case {} xQuantized",
            case["id"]
        );
    }
}

/// Operand sets the published contract refuses. These are the bytes the old success case wrongly
/// carried (negative/NaN `s_rel`), plus a zero channel scale; each is driven through the production
/// loader so the refusal is the contract's, not the test's.
#[test]
fn invalid_operands_are_refused_by_the_published_contract() {
    let fx = fixture();
    let refusals = fx["refusedOperands"]
        .as_array()
        .expect("fixture must pin expected refusals");
    assert!(!refusals.is_empty());
    for refusal in refusals {
        let dir = tempfile::tempdir().unwrap();
        let file = dir.path().join("refused.safetensors");
        write_operands(&file, "p", refusal);
        // The reader validates packed W4A8 operands while loading, so the refusal surfaces there
        // (before any backend is chosen); a reader that let these bytes through must still refuse
        // at projection load.
        let err = match Weights::from_files_native_int8(&[&file], &Device::Cpu) {
            Ok(w) => lin(&w, DIT, "p", true, DType::BF16)
                .err()
                .unwrap_or_else(|| panic!("case {} must be refused", refusal["id"]))
                .to_string(),
            Err(e) => e.to_string(),
        };
        let expected = refusal["expectedError"].as_str().unwrap();
        assert!(
            err.contains(expected),
            "case {} must be refused with {expected:?}, got {err:?}",
            refusal["id"]
        );
        // The entry has to actually carry the violation it claims (a refused-but-valid operand set
        // would make this test pass for the wrong reason).
        let rule = refusal["rule"].as_str().unwrap();
        let violates = if rule.contains("s_rel") {
            refusal["sRelBits"].as_array().unwrap().iter().any(|b| {
                let b = b.as_u64().unwrap() as u8;
                b & 0x80 != 0 || (b & 0x7F) == 0x7F
            })
        } else {
            refusal["sChannel"]
                .as_array()
                .unwrap()
                .iter()
                .any(|v| v.as_f64().unwrap() <= 0.0)
        };
        assert!(
            violates,
            "case {} must actually violate the rule it claims",
            refusal["id"]
        );
    }
}

/// One published operand set carrying a packed W4A8 layer (`p`) and a ConvRot INT8 layer (`q`).
/// The W4A8 half must stay W4A8 and reproduce its own case; the ConvRot half must stay ConvRot (no
/// densification in either direction) and reproduce the pinned S4 activation/GEMM/epilogue.
#[test]
#[cfg(feature = "cuda")]
#[ignore = "requires coordinator-reserved CUDA device; absence is failure, not skip"]
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
        convrot["m"].as_u64().unwrap() as usize,
        convrot["n"].as_u64().unwrap() as usize,
        convrot["k"].as_u64().unwrap() as usize,
    );
    let device = Device::new_cuda(0).expect("reserved CUDA runner must provide a real device");
    let dir = tempfile::tempdir().unwrap();
    let file = dir.path().join("mixed.safetensors");
    let p_op = w4a8_operands(w4a8);
    let codes: Vec<i8> = serde_json::from_value(convrot["weight"].clone()).unwrap();
    let scales: Vec<u8> = floats(&convrot["weightScale"])
        .iter()
        .flat_map(|&v| v.to_le_bytes())
        .collect();
    let q_bias: Vec<u8> = floats(&convrot["bias"])
        .iter()
        .flat_map(|&v| v.to_le_bytes())
        .collect();
    let q_desc = serde_json::to_vec(&serde_json::json!({
        "format": "int8_tensorwise",
        "convrot": true,
        "convrot_groupsize": 256,
    }))
    .unwrap();
    let mut views = w4a8_views("p", &p_op);
    views.extend(convrot_views(
        "q", convrot, &codes, &scales, &q_bias, &q_desc,
    ));
    serialize_to_file(views, &None, &file).unwrap();
    let w = Weights::from_files_native_int8(&[&file], &device).expect("reader");

    let p = lin(
        &w,
        DIT,
        "p",
        w4a8.get("bias").is_some_and(|b| !b.is_null()),
        DType::BF16,
    )
    .expect("load p");
    assert!(p.linear.is_w4a8(), "p must stay W4A8");
    assert!(!p.linear.is_convrot_int8());
    assert!(p.linear.is_packed(), "p must not densify");
    let (pm, pk) = (
        w4a8["m"].as_u64().unwrap() as usize,
        w4a8["k"].as_u64().unwrap() as usize,
    );
    let px = Tensor::from_slice(&floats(&w4a8["x"]), (pm, pk), &device)
        .unwrap()
        .to_dtype(DType::BF16)
        .unwrap();
    let py = p.linear.forward(&px).expect("p forward");
    assert_eq!(
        py.to_dtype(DType::F32)
            .unwrap()
            .flatten_all()
            .unwrap()
            .to_vec1::<f32>()
            .unwrap(),
        floats(&w4a8["expected"]["output"]),
        "mixed-layer W4A8 half"
    );

    let q = lin(&w, DIT, "q", true, DType::F32).expect("load q");
    assert!(q.linear.is_convrot_int8(), "q must stay ConvRot INT8");
    assert!(!q.linear.is_w4a8());
    assert!(q.linear.is_packed(), "q must not densify");
    let qx = Tensor::from_vec(floats(&convrot["input"]), (cm, ck), &device).unwrap();
    let qy = q.linear.forward(&qx).expect("q forward");
    assert_eq!(
        qy.to_dtype(DType::F32)
            .unwrap()
            .flatten_all()
            .unwrap()
            .to_vec1::<f32>()
            .unwrap(),
        floats(&convrot["expected"]),
        "mixed-layer ConvRot half ({}x{}x{})",
        cm,
        cn,
        ck
    );
}

/// The canonical W4A8 operand set a reader loads: packed codes, codebook, `s_rel` bytes and
/// `s_channel`, under `prefix`. Owned buffers so a second (ConvRot) prefix can share the file.
struct W4A8Operands {
    packed: Vec<u8>,
    s_rel: Vec<u8>,
    codebook: Vec<u8>,
    channel: Vec<u8>,
    bias: Option<Vec<u8>>,
    desc: Vec<u8>,
    n: usize,
    k: usize,
}

fn w4a8_operands(case: &Value) -> W4A8Operands {
    let (n, k) = (
        case["n"].as_u64().unwrap() as usize,
        case["k"].as_u64().unwrap() as usize,
    );
    W4A8Operands {
        packed: unsigned(&case["packedWeight"]),
        s_rel: unsigned(&case["sRelBits"]),
        codebook: floats(&case["codebook"])
            .iter()
            .flat_map(|&v| v.to_le_bytes())
            .collect(),
        channel: floats(&case["sChannel"])
            .iter()
            .flat_map(|&v| v.to_le_bytes())
            .collect(),
        bias: case
            .get("bias")
            .filter(|b| !b.is_null())
            .map(|b| floats(b).iter().flat_map(|&v| v.to_le_bytes()).collect()),
        desc: serde_json::to_vec(&serde_json::json!({
            "convrot_groupsize": 256,
            "format": "asym_w4a8_int8",
            "group_size": 16,
        }))
        .unwrap(),
        n,
        k,
    }
}

fn w4a8_views<'a>(prefix: &str, op: &'a W4A8Operands) -> Vec<(String, TensorView<'a>)> {
    let mut views = vec![
        (
            format!("{prefix}.weight"),
            TensorView::new(Dtype::I8, vec![op.n, op.k / 2], &op.packed).unwrap(),
        ),
        (
            format!("{prefix}.weight_codebook"),
            TensorView::new(Dtype::F32, vec![16], &op.codebook).unwrap(),
        ),
        (
            format!("{prefix}.weight_s_rel"),
            TensorView::new(Dtype::F8_E4M3, vec![op.n, op.k / 16], &op.s_rel).unwrap(),
        ),
        (
            format!("{prefix}.weight_s_channel"),
            TensorView::new(Dtype::F32, vec![op.n], &op.channel).unwrap(),
        ),
        (
            format!("{prefix}.comfy_quant"),
            TensorView::new(Dtype::U8, vec![op.desc.len()], &op.desc).unwrap(),
        ),
    ];
    if let Some(bytes) = &op.bias {
        views.push((
            format!("{prefix}.bias"),
            TensorView::new(Dtype::F32, vec![op.n], bytes).unwrap(),
        ));
    }
    views
}

/// Write one case as the same canonical operands the native reader loads, so the forward runs the
/// production loader path rather than a hand-built `Weights` map.
fn write_operands(path: &std::path::Path, prefix: &str, case: &Value) {
    let op = w4a8_operands(case);
    serialize_to_file(w4a8_views(prefix, &op), &None, path).unwrap();
}

#[cfg(feature = "cuda")]
fn write_case(path: &std::path::Path, prefix: &str, case: &Value) {
    write_operands(path, prefix, case);
}

/// The S4 ConvRot INT8 operand views of the mixed case's second layer.
#[cfg(feature = "cuda")]
fn convrot_views<'a>(
    prefix: &str,
    convrot: &'a Value,
    codes: &'a [i8],
    scales: &'a [u8],
    bias: &'a [u8],
    desc: &'a [u8],
) -> Vec<(String, TensorView<'a>)> {
    let (n, k) = (
        convrot["n"].as_u64().unwrap() as usize,
        convrot["k"].as_u64().unwrap() as usize,
    );
    vec![
        (
            format!("{prefix}.weight"),
            TensorView::new(Dtype::I8, vec![n, k], codes).unwrap(),
        ),
        (
            format!("{prefix}.weight_scale"),
            TensorView::new(Dtype::F32, vec![n, 1], scales).unwrap(),
        ),
        (
            format!("{prefix}.bias"),
            TensorView::new(Dtype::F32, vec![n], bias).unwrap(),
        ),
        (
            format!("{prefix}.comfy_quant"),
            TensorView::new(Dtype::U8, vec![desc.len()], desc).unwrap(),
        ),
    ]
}

/// Full production forward against the external `output`. CUDA-only: the published W4A8 path has no
/// floating fallback (matches `#[cfg(not(feature = "cuda"))]` refusing in `W4A8Linear`).
#[test]
#[cfg(feature = "cuda")]
#[ignore = "requires coordinator-reserved CUDA device; absence is failure, not skip"]
fn native_forward_matches_the_external_reference() {
    let fx = fixture();
    let device = Device::new_cuda(0).expect("reserved CUDA runner must provide a real device");
    for case in fx["cases"].as_array().unwrap() {
        let (n, k) = (
            case["n"].as_u64().unwrap() as usize,
            case["k"].as_u64().unwrap() as usize,
        );
        let m = case["m"].as_u64().unwrap() as usize;
        // The shared cuBLASLt INT8 IGEMM requires an output width that is a multiple of 16, and the
        // loader refuses anything else — so a case with any other N can never be executed here.
        // Every fixture case is generated in that supported geometry (the generator asserts it);
        // this test asserts instead of skipping, so a new case cannot silently assert nothing.
        assert_eq!(
            n % 16,
            0,
            "case {} n={n} is outside the CUDA INT8 output-width geometry",
            case["id"]
        );
        let dir = tempfile::tempdir().unwrap();
        let file = dir.path().join("case.safetensors");
        write_case(&file, "p", case);
        let w = Weights::from_files_native_int8(&[&file], &device).expect("reader");
        let loaded = lin(
            &w,
            DIT,
            "p",
            case.get("bias").map_or(false, |b| !b.is_null()),
            DType::BF16,
        )
        .expect("load W4A8");
        assert!(loaded.linear.is_w4a8(), "case {} must be W4A8", case["id"]);
        // The H3 hidden state is BF16; the production forward casts its output to the input dtype,
        // so BF16 input reproduces the reference's single BF16 RN store.
        let x = Tensor::from_slice(&floats(&case["x"]), (m, k), &device)
            .expect("x f32")
            .to_dtype(DType::BF16)
            .expect("x bf16");
        let y = loaded.linear.forward(&x).expect("forward");
        let got = y
            .to_dtype(DType::F32)
            .expect("f32")
            .flatten_all()
            .expect("flat")
            .to_vec1::<f32>()
            .expect("vec");
        assert_eq!(
            got,
            floats(&case["expected"]["output"]),
            "case {} forward",
            case["id"]
        );
    }
}
