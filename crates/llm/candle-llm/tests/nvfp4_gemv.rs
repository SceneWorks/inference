//! The fused NVFP4 decode GEMV behind `Projection::Nvfp4` (sc-24136, epic sc-24128).
//!
//! - **AC1 parity:** for every NVFP4 projection shape of Qwen3.8-27B — enumerated from the
//!   checkpoint's own `config.json` (`docs/reference/qwen38/config.json`, identical to the pinned
//!   snapshot's) — and every decode row count 1..=8, the GEMV output matches the
//!   dequantize-then-matmul reference within the declared tolerance
//!   (`candle_quant_kernels::GEMV_REL_RMS_TOL` on the relative RMS, `gemv_abs_bound` per element),
//!   plus the edge shapes whose `K` is not a multiple of the 16-block. The delta against the
//!   cuBLASLt W4A4 path on the same weight is measured and printed (it is not bit-identical by
//!   design: the GEMV does not quantize the activation). `NVFP4_GEMV_PARITY_OUTPUT=<json>` writes
//!   the table.
//! - **Dispatch:** ≤ 8 bf16 rows take the GEMV; 9+ rows, an f32 activation or the switch off take
//!   cuBLASLt — each recorded in the per-thread `Nvfp4PathTally` with its reason.
//! - **Seam:** the GEMV compiles once per device through `candle_quant_kernels::nvrtc`.
//! - **Microbench** (`#[ignore]`, `NVFP4_GEMV_BENCH_OUTPUT=<json>`): µs per call and effective GB/s,
//!   GEMV vs cuBLASLt, per 27B shape and row count 1..=8.
//!
//! GPU tests skip (loudly) without an sm_120 CUDA device. They share the process-wide GEMV switch,
//! so they serialize on one lock.

#![cfg(feature = "cuda")]

use std::sync::Mutex;

use candle_core::{DType, Device, Tensor};
use candle_llm::models::Qwen35Config;
use candle_llm::primitives::{
    nvfp4_path_tally, set_nvfp4_gemv, Projection, ProjectionFormat, ProjectionKind,
};
use candle_quant_kernels::{
    e4m3_to_f32, gemv_abs_bound, Nvfp4Weight, E2M1_LUT, GEMV_REL_RMS_TOL, NVFP4_BLOCK,
    NVFP4_GEMV_MAX_ROWS, NVFP4_GEMV_SRC,
};
use serde_json::{json, Value};

static SWITCH: Mutex<()> = Mutex::new(());

fn lock() -> std::sync::MutexGuard<'static, ()> {
    SWITCH.lock().unwrap_or_else(|p| p.into_inner())
}

fn nvfp4() -> Option<(Device, ProjectionFormat)> {
    let device = Device::new_cuda(0).ok()?;
    let format = ProjectionFormat::nvfp4(&device).ok()?;
    Some((device, format))
}

/// Every NVFP4 projection of Qwen3.8-27B as `(name, out = N, in = K)`, derived from the config the
/// way `Qwen35Model::from_weights_format` / `Qwen35Mtp::from_weights_with` load them: the full
/// attention layers' q (doubled by the output gate) / k / v / o, the Gated DeltaNet in (qkv, z) and
/// out projections (`in_proj_a` / `in_proj_b` stay dense), the SwiGLU gate / up / down, the MTP
/// head's `fc` (its decoder layer repeats the attention and MLP shapes) and the untied `lm_head`.
fn qwen38_27b_projection_shapes() -> Vec<(&'static str, usize, usize)> {
    let config: Value = serde_json::from_str(include_str!(
        "../../../../docs/reference/qwen38/config.json"
    ))
    .unwrap();
    let c = Qwen35Config::from_json(&config).unwrap();
    assert!(c.moe.is_none() && !c.tie_word_embeddings && c.mtp_num_hidden_layers > 0);
    let h = c.hidden_size as usize;
    let q_heads = (c.num_heads * c.head_dim) as usize;
    let kv = (c.num_kv_heads * c.head_dim) as usize;
    let key_dim = (c.linear_key_head_dim * c.linear_num_key_heads) as usize;
    let value_dim = (c.linear_value_head_dim * c.linear_num_value_heads) as usize;
    let inter = c.intermediate_size as usize;
    let shapes = vec![
        ("self_attn.q_proj (+gate)", 2 * q_heads, h),
        ("self_attn.k_proj", kv, h),
        ("self_attn.v_proj", kv, h),
        ("self_attn.o_proj", h, q_heads),
        ("linear_attn.in_proj_qkv", 2 * key_dim + value_dim, h),
        ("linear_attn.in_proj_z", value_dim, h),
        ("linear_attn.out_proj", h, value_dim),
        ("mlp.gate_proj / up_proj", inter, h),
        ("mlp.down_proj", h, inter),
        ("mtp.fc", h, 2 * h),
        ("lm_head", c.vocab_size as usize, h),
    ];
    // Pin the enumeration to the published 27B dims so a config drift is loud.
    assert_eq!(
        shapes.iter().map(|&(_, n, k)| (n, k)).collect::<Vec<_>>(),
        vec![
            (12288, 5120),
            (1024, 5120),
            (1024, 5120),
            (5120, 6144),
            (10240, 5120),
            (6144, 5120),
            (5120, 6144),
            (17408, 5120),
            (5120, 17408),
            (5120, 10240),
            (248_320, 5120),
        ]
    );
    shapes
}

/// Shapes whose `K` is not a multiple of the 16-block (or of 8, the activation's vector width),
/// served through the same projection.
const EDGE_SHAPES: &[(&str, usize, usize)] = &[
    ("edge K=17", 16, 17),
    ("edge K=80 (pads to 96)", 96, 80),
    ("edge K=1000", 32, 1000),
    ("edge K=5130", 128, 5130),
];

fn weight(n: usize, k: usize, device: &Device) -> Tensor {
    // Normal(0, 0.02) like a trained projection; generated on-device (the lm_head is 1.27 G
    // elements).
    Tensor::randn(0f32, 0.02, (n, k), device)
        .unwrap()
        .to_dtype(DType::BF16)
        .unwrap()
}

fn activation(m: usize, k: usize, device: &Device) -> Tensor {
    Tensor::randn(0f32, 1.0, (1, m, k), device)
        .unwrap()
        .to_dtype(DType::BF16)
        .unwrap()
}

/// The reference rows checked for an `n`-row projection: every row up to 17408 (every 27B shape
/// but the head); for the 248320-row `lm_head`, the first and last 128-row scale atoms plus every
/// 16th row — 15.8 k rows across all 1940 row atoms.
fn reference_rows(n: usize) -> Vec<usize> {
    if n <= 17_408 {
        return (0..n).collect();
    }
    (0..n)
        .filter(|&r| r < 128 || r >= n - 128 || r % 16 == 0)
        .collect()
}

/// `dequant(W)[rows]` as an f64 `[rows.len(), k]` CPU tensor, decoded from the resident weight
/// through the codec's canonical layout (`Nvfp4Weight::to_host` inverts the device layout), so the
/// reference shares none of the kernel's layout arithmetic.
fn dequant_rows(w: &Nvfp4Weight, rows: &[usize]) -> Tensor {
    let host = w.to_host().unwrap();
    let (_, k) = w.shape();
    let row_bytes = host.cols_padded / 2;
    let mut out = Vec::with_capacity(rows.len() * k);
    for &r in rows {
        for c in 0..k {
            let blk = c / NVFP4_BLOCK;
            let scale = e4m3_to_f32(host.scales[host.scale_offset(r, blk)]) as f64
                * host.global_scale as f64;
            let byte = host.packed[r * row_bytes + c / 2];
            let code = if c % 2 == 0 { byte & 0x0f } else { byte >> 4 };
            out.push(E2M1_LUT[code as usize] as f64 * scale);
        }
    }
    Tensor::from_vec(out, (rows.len(), k), &Device::Cpu).unwrap()
}

fn host_f64(t: &Tensor, m: usize, cols: usize) -> Vec<f64> {
    t.reshape((m, cols))
        .unwrap()
        .to_device(&Device::Cpu)
        .unwrap()
        .to_dtype(DType::F64)
        .unwrap()
        .flatten_all()
        .unwrap()
        .to_vec1::<f64>()
        .unwrap()
}

#[derive(Default)]
struct Stats {
    num: f64,
    den: f64,
    max_abs: f64,
    max_ratio_to_bound: f64,
}

impl Stats {
    fn rel_rms(&self) -> f64 {
        (self.num / self.den.max(1e-300)).sqrt()
    }
}

/// Compare `got` (`[m, n]`) on the reference rows; `bound` enforces the declared per-element
/// tolerance (the GEMV) or only measures (the cuBLASLt delta).
fn compare(
    got: &[f64],
    n: usize,
    rows: &[usize],
    reference: &[f64],
    abs_dot: &[f64],
    k: usize,
    enforce: Option<&str>,
) -> Stats {
    let mut s = Stats::default();
    let m = got.len() / n;
    for mi in 0..m {
        for (j, &r) in rows.iter().enumerate() {
            let want = reference[mi * rows.len() + j];
            let err = (got[mi * n + r] - want).abs();
            s.num += err * err;
            s.den += want * want;
            s.max_abs = s.max_abs.max(err);
            let bound = gemv_abs_bound(want, abs_dot[mi * rows.len() + j], k);
            s.max_ratio_to_bound = s.max_ratio_to_bound.max(err / bound);
            if let Some(what) = enforce {
                assert!(
                    err <= bound,
                    "{what}: m={mi} row {r}: |{} - {want}| = {err} > declared bound {bound}",
                    got[mi * n + r]
                );
            }
        }
    }
    s
}

/// AC1: GEMV vs the dequantize-then-matmul reference on every 27B projection shape (and the edge
/// shapes) for rows 1..=8, under the declared tolerance; the cuBLASLt delta is recorded alongside.
#[test]
fn gemv_matches_the_dequant_reference_on_every_qwen38_27b_projection_shape() {
    let _guard = lock();
    let Some((device, format)) = nvfp4() else {
        eprintln!("skipping: no sm_120 CUDA device");
        return;
    };
    device.set_seed(24_136).unwrap();
    let mut table = Vec::new();
    let shapes: Vec<_> = qwen38_27b_projection_shapes()
        .into_iter()
        .chain(EDGE_SHAPES.iter().copied())
        .collect();
    let mut seen = std::collections::HashSet::new();
    for (name, n, k) in shapes {
        if !seen.insert((n, k)) {
            continue; // v_proj repeats k_proj, out_proj repeats o_proj
        }
        let p = Projection::load_as(weight(n, k, &device), None, Some(&format)).unwrap();
        assert_eq!(p.kind(), ProjectionKind::Nvfp4);
        let Projection::Nvfp4(w) = &p else {
            unreachable!()
        };
        let rows = reference_rows(n);
        let w_ref = dequant_rows(w, &rows);
        for m in 1..=NVFP4_GEMV_MAX_ROWS {
            let x = activation(m, k, &device);
            set_nvfp4_gemv(Some(true));
            let before = nvfp4_path_tally();
            let y_gemv = p.forward(&x).unwrap();
            let d = nvfp4_path_tally().since(&before);
            assert_eq!((d.gemv, d.cublaslt), (1, 0), "{name} m={m} ran the GEMV");
            set_nvfp4_gemv(Some(false));
            let y_lt = p.forward(&x).unwrap();
            set_nvfp4_gemv(None);
            assert_eq!(y_gemv.dims(), &[1, m, n]);
            assert_eq!(y_gemv.dtype(), DType::BF16);

            let xh = x
                .reshape((m, k))
                .unwrap()
                .to_device(&Device::Cpu)
                .unwrap()
                .to_dtype(DType::F64)
                .unwrap();
            let reference = xh.matmul(&w_ref.t().unwrap()).unwrap();
            let abs_dot = xh
                .abs()
                .unwrap()
                .matmul(&w_ref.abs().unwrap().t().unwrap())
                .unwrap();
            let reference = reference.flatten_all().unwrap().to_vec1::<f64>().unwrap();
            let abs_dot = abs_dot.flatten_all().unwrap().to_vec1::<f64>().unwrap();
            let what = format!("{name} [{n},{k}] m={m}");
            let gemv = compare(
                &host_f64(&y_gemv, m, n),
                n,
                &rows,
                &reference,
                &abs_dot,
                k,
                Some(&what),
            );
            assert!(
                gemv.rel_rms() <= GEMV_REL_RMS_TOL,
                "{what}: rel-RMS {} > {GEMV_REL_RMS_TOL}",
                gemv.rel_rms()
            );
            let lt = compare(
                &host_f64(&y_lt, m, n),
                n,
                &rows,
                &reference,
                &abs_dot,
                k,
                None,
            );
            let y_g = host_f64(&y_gemv, m, n);
            let y_l = host_f64(&y_lt, m, n);
            let (mut dn, mut dd) = (0f64, 0f64);
            for (a, b) in y_g.iter().zip(&y_l) {
                dn += (a - b) * (a - b);
                dd += b * b;
            }
            let gemv_vs_cublaslt = (dn / dd.max(1e-300)).sqrt();
            eprintln!(
                "[nvfp4_gemv parity] {what:<44} gemv rel-RMS {:.3e} (max |err| {:.3e}, {:.3} of \
                 bound) | cuBLASLt W4A4 rel-RMS {:.3e} | gemv vs cuBLASLt {:.3e}",
                gemv.rel_rms(),
                gemv.max_abs,
                gemv.max_ratio_to_bound,
                lt.rel_rms(),
                gemv_vs_cublaslt
            );
            table.push(json!({
                "projection": name,
                "n": n,
                "k": k,
                "rows": m,
                "reference_rows_checked": rows.len(),
                "gemv_rel_rms": gemv.rel_rms(),
                "gemv_max_abs_err": gemv.max_abs,
                "gemv_max_err_over_declared_bound": gemv.max_ratio_to_bound,
                "cublaslt_w4a4_rel_rms": lt.rel_rms(),
                "cublaslt_w4a4_max_abs_err": lt.max_abs,
                "gemv_vs_cublaslt_rel_rms": gemv_vs_cublaslt,
            }));
        }
    }
    if let Ok(path) = std::env::var("NVFP4_GEMV_PARITY_OUTPUT") {
        let doc = json!({
            "suite": "nvfp4_gemv_parity",
            "story": "sc-24136",
            "label": std::env::var("NVFP4_GEMV_LABEL").unwrap_or_else(|_| "RTX Pro 6000 / sm_120".into()),
            "declared_tolerance": {
                "rel_rms_max": GEMV_REL_RMS_TOL,
                "per_element": "|y - ref| <= |ref|/128 + K * 2^-24 * sum_k |x*W|",
                "reference": "f64 x · dequant(W)^T over the resident weight read back through the codec",
            },
            "weights": "Normal(0, 0.02) bf16, quantized at load; activations Normal(0, 1) bf16",
            "rows": table,
        });
        std::fs::write(&path, serde_json::to_string_pretty(&doc).unwrap()).unwrap();
        eprintln!("[nvfp4_gemv parity] wrote {path}");
    }
}

/// The dispatch rule, visible in telemetry: ≤ 8 bf16 rows → GEMV; more rows (a prefill), an f32
/// activation or the switch off → cuBLASLt with the reason. The outputs agree either way.
#[test]
fn dispatch_takes_the_gemv_for_decode_rows_and_cublaslt_otherwise_visibly() {
    let _guard = lock();
    let Some((device, format)) = nvfp4() else {
        eprintln!("skipping: no sm_120 CUDA device");
        return;
    };
    let (n, k) = (256, 512);
    let p = Projection::load_as(weight(n, k, &device), None, Some(&format)).unwrap();
    let run = |x: &Tensor, switch: bool| {
        set_nvfp4_gemv(Some(switch));
        let before = nvfp4_path_tally();
        let y = p.forward(x).unwrap();
        set_nvfp4_gemv(None);
        (y, nvfp4_path_tally().since(&before))
    };
    for m in [1, 3, 5, 7, 8] {
        let (_, d) = run(&activation(m, k, &device), true);
        assert_eq!(d.label(), "gemv", "m={m}");
        assert_eq!(d.cublaslt_reason, None);
    }
    for m in [9, 16, 97] {
        let (y, d) = run(&activation(m, k, &device), true);
        assert_eq!(y.dims(), &[1, m, n]);
        assert_eq!(d.label(), "cublaslt", "m={m}");
        assert_eq!(d.cublaslt_reason, Some("rows"), "m={m}");
    }
    let f32x = activation(1, k, &device).to_dtype(DType::F32).unwrap();
    let (y, d) = run(&f32x, true);
    assert_eq!(y.dtype(), DType::F32);
    assert_eq!((d.label(), d.cublaslt_reason), ("cublaslt", Some("dtype")));
    let x = activation(4, k, &device);
    let (y_off, d) = run(&x, false);
    assert_eq!(
        (d.label(), d.cublaslt_reason),
        ("cublaslt", Some("disabled"))
    );
    let (y_on, _) = run(&x, true);
    // Different arithmetic (W4A16 vs W4A4), same projection: close, not identical.
    let rel = ((&y_on.to_dtype(DType::F32).unwrap() - &y_off.to_dtype(DType::F32).unwrap())
        .unwrap()
        .sqr()
        .unwrap()
        .sum_all()
        .unwrap()
        .to_scalar::<f32>()
        .unwrap()
        / y_off
            .to_dtype(DType::F32)
            .unwrap()
            .sqr()
            .unwrap()
            .sum_all()
            .unwrap()
            .to_scalar::<f32>()
            .unwrap())
    .sqrt();
    assert!(rel < 0.2, "gemv vs cuBLASLt rel {rel}");
}

/// The GEMV is compiled through the shared nvrtc seam, once per device, however many NVFP4
/// projections and row counts use it.
#[test]
fn gemv_compiles_once_per_device_through_the_seam() {
    let _guard = lock();
    let Some((device, format)) = nvfp4() else {
        eprintln!("skipping: no sm_120 CUDA device");
        return;
    };
    let Device::Cuda(dev) = &device else {
        unreachable!()
    };
    set_nvfp4_gemv(Some(true));
    for (n, k) in [(16, 64), (64, 256), (32, 1000)] {
        let p = Projection::load_as(weight(n, k, &device), None, Some(&format)).unwrap();
        for m in 1..=NVFP4_GEMV_MAX_ROWS {
            p.forward(&activation(m, k, &device)).unwrap();
        }
    }
    set_nvfp4_gemv(None);
    assert_eq!(NVFP4_GEMV_SRC.compile_attempts(dev), 1);
    assert!(matches!(NVFP4_GEMV_SRC.cached(dev), Some(Ok(_))));
}

fn time_us(device: &Device, iters: usize, mut f: impl FnMut()) -> f64 {
    for _ in 0..5 {
        f();
    }
    device.synchronize().unwrap();
    let start = std::time::Instant::now();
    for _ in 0..iters {
        f();
    }
    device.synchronize().unwrap();
    start.elapsed().as_secs_f64() * 1e6 / iters as f64
}

/// Per-shape kernel microbench: µs per call and effective GB/s (packed weight + scales + bf16
/// activation in + bf16 output out, per call) for the GEMV and for the cuBLASLt W4A4 forward, on
/// every 27B projection shape at rows 1..=8.
#[test]
#[ignore = "microbench: run in release on an idle sm_120 GPU (NVFP4_GEMV_BENCH_OUTPUT=<json>)"]
fn nvfp4_gemv_microbench() {
    let _guard = lock();
    let (device, format) = nvfp4().expect("an sm_120 CUDA device");
    device.set_seed(7).unwrap();
    let mut seen = std::collections::HashSet::new();
    let mut rows_json = Vec::new();
    for (name, n, k) in qwen38_27b_projection_shapes() {
        if !seen.insert((n, k)) {
            continue;
        }
        let p = Projection::load_as(weight(n, k, &device), None, Some(&format)).unwrap();
        let Projection::Nvfp4(w) = &p else {
            unreachable!()
        };
        let weight_bytes = w.resident_bytes() as f64;
        let iters = if n > 100_000 { 50 } else { 200 };
        for m in 1..=NVFP4_GEMV_MAX_ROWS {
            let x = activation(m, k, &device);
            let io_bytes = weight_bytes + (m * k * 2 + m * n * 2) as f64;
            let gemv = time_us(&device, iters, || {
                w.forward_gemv(&x).unwrap();
            });
            let lt = time_us(&device, iters, || {
                w.forward(&x).unwrap();
            });
            eprintln!(
                "[nvfp4_gemv bench] {name:<28} [{n:>6},{k:>5}] m={m}  gemv {gemv:>8.1} us \
                 ({:>6.0} GB/s)  cuBLASLt {lt:>8.1} us  ({:.2}x)",
                io_bytes / gemv / 1e3,
                lt / gemv
            );
            rows_json.push(json!({
                "projection": name,
                "n": n,
                "k": k,
                "rows": m,
                "bytes_per_call": io_bytes,
                "gemv": {"us_per_call": gemv, "effective_gb_per_s": io_bytes / gemv / 1e3},
                "cublaslt_w4a4": {"us_per_call": lt, "effective_gb_per_s": io_bytes / lt / 1e3},
                "speedup_vs_cublaslt": lt / gemv,
            }));
        }
    }
    if let Ok(path) = std::env::var("NVFP4_GEMV_BENCH_OUTPUT") {
        let doc = json!({
            "suite": "nvfp4_gemv_microbench",
            "story": "sc-24136",
            "label": std::env::var("NVFP4_GEMV_LABEL").unwrap_or_else(|_| "RTX Pro 6000 / sm_120".into()),
            "timing": "wall clock over N back-to-back calls between two device synchronizes, after 5 warm-up calls",
            "effective_bytes": "packed NVFP4 weight + UE4M3 scales + bf16 activation + bf16 output, per call",
            "rows": rows_json,
        });
        std::fs::write(&path, serde_json::to_string_pretty(&doc).unwrap()).unwrap();
        eprintln!("[nvfp4_gemv bench] wrote {path}");
    }
}
