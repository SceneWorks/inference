//! Tiled INT8 GEMM parity and throughput on Metal (sc-26691).
//!
//! The production ConvRot INT8 and packed W4A8 projections now run `mlx_gen::int8_gemm`'s tiled
//! kernel (windowed FP32 `simdgroup_matrix` products flushed exactly into int32). The contract is
//! the serial sc-26683 / sc-26685 kernels' **exact int32 accumulation**, so every comparison here is
//! bit-for-bit against `forward_reference` (those serial kernels, kept test-only) — at every
//! activation dtype, including F32, because both paths share the rotation and the epilogue.
//!
//! Every body is `#[ignore]`d: it executes Metal kernels, and kernel execution never runs in
//! ordinary CI. The explicit `synthetic-w4a8-mlx` route of `real-weights.yml` runs them with an
//! exact pass-count guard; `real_checkpoint_slice_matches_serial_reference` additionally needs
//! `H3_INT8_GEMM_REAL_SAFETENSORS` (a published ConvRot H3 checkpoint file) and FAILS without it.
use mlx_gen::adapters::AdaptableLinear;
use mlx_gen::gen_core::weightsmeta::{self, Dtype as StDtype};
use mlx_gen::weights::Weights;
use mlx_gen_minimax_h3::quant;
use mlx_rs::{transforms::eval, Array, Dtype};
use serde_json::json;
use std::collections::HashMap;
use std::io::{Read, Seek, SeekFrom};
use std::time::Instant;

/// Deterministic splitmix64; no RNG dependency and identical streams on every host.
struct Rng(u64);
impl Rng {
    fn next(&mut self) -> u64 {
        self.0 = self.0.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^ (z >> 31)
    }
    fn unit(&mut self) -> f32 {
        (self.next() >> 40) as f32 / (1u64 << 24) as f32
    }
    /// Roughly normal (Irwin-Hall of four), with occasional per-channel outliers like real
    /// activations.
    fn activation(&mut self) -> f32 {
        let v = (self.unit() + self.unit() + self.unit() + self.unit() - 2.0) * 1.7;
        if self.next() % 97 == 0 {
            v * 24.0
        } else {
            v
        }
    }
    fn code(&mut self) -> i8 {
        self.next() as u8 as i8
    }
}

fn require_metal() {
    let device = mlx_rs::Device::try_default().expect("INT8 GEMM proof requires an MLX device");
    assert!(
        matches!(device.get_type().unwrap(), mlx_rs::DeviceType::Gpu),
        "a CPU fallback cannot prove the Metal kernel"
    );
    println!("INT8_GEMM_DEVICE {device}");
}

fn descriptor(group: i32) -> Array {
    let blob = serde_json::to_vec(
        &json!({"format":"int8_tensorwise","convrot":true,"convrot_groupsize":group}),
    )
    .unwrap();
    Array::from_slice(&blob, &[blob.len() as i32])
}

fn convrot(
    codes: &[i8],
    scales: &[f32],
    bias: Option<&[f32]>,
    n: i32,
    k: i32,
    group: i32,
) -> AdaptableLinear {
    let mut map = HashMap::from([
        ("p.weight".to_string(), Array::from_slice(codes, &[n, k])),
        (
            "p.weight_scale".to_string(),
            Array::from_slice(scales, &[n, 1]),
        ),
        ("p.comfy_quant".to_string(), descriptor(group)),
    ]);
    if let Some(b) = bias {
        map.insert("p.bias".into(), Array::from_slice(b, &[n]));
    }
    let linear = quant::lin(&Weights::from_map(map), "p", bias.is_some()).unwrap();
    assert!(
        linear.convrot_base().is_some(),
        "must reach the native ConvRot base"
    );
    linear
}

fn random_convrot(rng: &mut Rng, n: i32, k: i32, group: i32, bias: bool) -> AdaptableLinear {
    let codes: Vec<i8> = (0..n * k).map(|_| rng.code()).collect();
    let scales: Vec<f32> = (0..n).map(|_| 1e-3 + rng.unit() * 0.05).collect();
    let b: Vec<f32> = (0..n).map(|_| rng.activation()).collect();
    convrot(&codes, &scales, bias.then_some(b.as_slice()), n, k, group)
}

fn activations(rng: &mut Rng, m: i32, k: i32, dtype: Dtype) -> Array {
    let x: Vec<f32> = (0..m * k).map(|_| rng.activation()).collect();
    Array::from_slice(&x, &[m, k]).as_dtype(dtype).unwrap()
}

fn bits(a: &Array) -> Vec<u32> {
    let f = a.as_dtype(Dtype::Float32).unwrap();
    eval([&f]).unwrap();
    f.as_slice::<f32>().iter().map(|v| v.to_bits()).collect()
}

/// Bit-for-bit comparison of the tiled forward against the serial reference.
fn assert_identical(label: &str, tiled: &Array, reference: &Array) {
    assert_eq!(tiled.shape(), reference.shape(), "{label}: shape");
    assert_eq!(tiled.dtype(), reference.dtype(), "{label}: dtype");
    let (t, r) = (bits(tiled), bits(reference));
    let mismatches = t.iter().zip(&r).filter(|(a, b)| a != b).count();
    if let Some(i) = t.iter().zip(&r).position(|(a, b)| a != b) {
        panic!(
            "{label}: {mismatches}/{} outputs differ; first at {i}: tiled {} != reference {}",
            t.len(),
            f32::from_bits(t[i]),
            f32::from_bits(r[i])
        );
    }
    assert!(!t.is_empty(), "{label}: empty output proves nothing");
    println!("INT8_GEMM_PARITY {label} outputs={} identical", t.len());
}

fn convrot_pair(linear: &AdaptableLinear, x: &Array) -> (Array, Array) {
    let base = linear.convrot_base().unwrap();
    let tiled = base.forward(x).unwrap();
    let reference = base.forward_reference(x).unwrap();
    eval([&tiled, &reference]).unwrap();
    (tiled, reference)
}

/// Ragged tiles (M, N not multiples of 64), both ConvRot groups, every activation dtype, with and
/// without bias, and a 3-D activation.
#[test]
#[ignore = "executes the Metal INT8 GEMM; explicit synthetic-w4a8-mlx route only"]
fn tiled_convrot_matches_serial_reference_on_random_codes() {
    require_metal();
    let mut rng = Rng(0x26691);
    let cases = [
        (5, 16, 192, 64, true),
        (67, 130, 768, 256, true),
        (129, 65, 1024, 64, false),
        (1, 1, 64, 64, true),
        (64, 64, 512, 256, false),
        (200, 333, 2560, 256, true),
    ];
    for (m, n, k, g, bias) in cases {
        let linear = random_convrot(&mut rng, n, k, g, bias);
        for dtype in [Dtype::Float32, Dtype::Bfloat16, Dtype::Float16] {
            let x = activations(&mut rng, m, k, dtype);
            let (t, r) = convrot_pair(&linear, &x);
            assert_identical(&format!("convrot m{m} n{n} k{k} g{g} {dtype:?}"), &t, &r);
        }
    }
    let linear = random_convrot(&mut rng, 96, 256, 64, true);
    let x = activations(&mut rng, 3 * 7, 256, Dtype::Bfloat16)
        .reshape(&[3, 7, 256])
        .unwrap();
    let (t, r) = convrot_pair(&linear, &x);
    assert_eq!(t.shape(), &[3, 7, 96]);
    assert_identical("convrot 3-D bf16", &t, &r);
}

/// The case the windowed flush exists for: H3's FFN-down depth (K = 14336) with saturated codes, so
/// `|acc|` reaches ~K·127² ≈ 2.3e8 — far beyond FP32's 2^24 contiguous-integer range, where a plain
/// FP32 accumulation would round. The activation is built as `v·R` (R the symmetric orthogonal
/// rotation), so the rotated row is ≈ ±1 and quantizes to ≈ ±127; aligned columns then sum K
/// same-sign products.
#[test]
#[ignore = "executes the Metal INT8 GEMM; explicit synthetic-w4a8-mlx route only"]
fn extreme_accumulation_beyond_f32_integer_range_is_exact() {
    require_metal();
    let (m, n, k, g) = (70i32, 72i32, 14336i32, 256i32);
    let mut rng = Rng(0x2_26691);
    let signs: Vec<f32> = (0..m * k)
        .map(|_| if rng.next() & 1 == 0 { 1.0 } else { -1.0 })
        .collect();
    // Hadamard-like H4 Kronecker basis, exactly as `ConvRotLinear::from_weights` builds it.
    let h4 = [
        [1f64, 1., 1., -1.],
        [1., 1., -1., 1.],
        [1., -1., 1., 1.],
        [-1., 1., 1., 1.],
    ];
    let gu = g as usize;
    let rot: Vec<f64> = (0..gu * gu)
        .map(|i| {
            let (mut r, mut c, mut s) = (i / gu, i % gu, 1f64);
            while r != 0 || c != 0 {
                s *= h4[r % 4][c % 4];
                r /= 4;
                c /= 4;
            }
            s / (gu as f64).sqrt()
        })
        .collect();
    let mut x = vec![0f32; (m * k) as usize];
    for row in 0..m as usize {
        for blk in 0..(k / g) as usize {
            let base = row * k as usize + blk * gu;
            for c in 0..gu {
                let mut acc = 0f64;
                for r in 0..gu {
                    acc += signs[base + r] as f64 * rot[r * gu + c];
                }
                x[base + c] = acc as f32;
            }
        }
    }
    // Columns 0..24 aligned with row 0's signs (+127 / -128), 24..48 anti-aligned, the rest random.
    let mut codes = vec![0i8; (n * k) as usize];
    for col in 0..n as usize {
        for j in 0..k as usize {
            let s = signs[j];
            codes[col * k as usize + j] = match col {
                0..24 => {
                    if s > 0.0 {
                        127
                    } else {
                        -128
                    }
                }
                24..48 => {
                    if s > 0.0 {
                        -128
                    } else {
                        127
                    }
                }
                _ => rng.code(),
            };
        }
    }
    let scales = vec![1.0f32; n as usize];
    let linear = convrot(&codes, &scales, None, n, k, g);
    for dtype in [Dtype::Float32, Dtype::Bfloat16, Dtype::Float16] {
        let x = Array::from_slice(&x, &[m, k]).as_dtype(dtype).unwrap();
        let (t, r) = convrot_pair(&linear, &x);
        assert_identical(&format!("convrot extreme K{k} {dtype:?}"), &t, &r);
        if dtype == Dtype::Float32 {
            // With sw = 1 and no bias, out = float(acc)·sx and sx ≈ 1/127: the aligned column of
            // row 0 must carry |acc| > 2^24, or this case did not exercise the range it claims.
            let out = t.as_dtype(Dtype::Float32).unwrap();
            eval([&out]).unwrap();
            let peak = out.as_slice::<f32>()[0].abs() * 127.0;
            assert!(
                peak > (1u64 << 24) as f32,
                "accumulator peak {peak} never left f32's exact range"
            );
            println!("INT8_GEMM_EXTREME accumulator_peak~{peak:.3e} (> 2^24 = 1.678e7)");
        }
    }
}

/// Every E4M3 group-scale byte the published contract admits: `validate_s_rel` accepts only
/// non-negative finite values, i.e. sign-clear bytes other than the 0x7F NaN code (normals,
/// subnormals, +0) plus the sign-SET 0x80 byte, which decodes to -0.0. Every other sign-set byte is
/// negative and refused at load (see `negative_s_rel_codes_are_refused_before_any_kernel`), so it
/// can never reach either kernel.
fn admissible_s_rel(rng: &mut Rng) -> u8 {
    match rng.next() % 16 {
        0 => 0x80,
        _ => (rng.next() % 0x7F) as u8,
    }
}

fn w4a8_operands(
    rng: &mut Rng,
    n: i32,
    k: i32,
    bias: bool,
    s_rel: impl Fn(&mut Rng) -> u8,
) -> Weights {
    let packed: Vec<i8> = (0..n * k / 2).map(|_| rng.code()).collect();
    let s_rel: Vec<u8> = (0..n * k / 16).map(|_| s_rel(rng)).collect();
    let codebook: Vec<f32> = (0..16)
        .map(|i| (i as f32 - 7.5) * (0.6 + rng.unit()))
        .collect();
    let s_channel: Vec<f32> = (0..n).map(|_| 1e-3 + rng.unit() * 0.02).collect();
    let desc = serde_json::to_vec(&json!({
        "convrot_groupsize": 256, "format": "asym_w4a8_int8", "group_size": 16,
    }))
    .unwrap();
    let mut map = HashMap::from([
        (
            "p.weight".to_string(),
            Array::from_slice(&packed, &[n, k / 2]),
        ),
        (
            "p.weight_codebook".to_string(),
            Array::from_slice(&codebook, &[16]),
        ),
        (
            "p.weight_s_rel".to_string(),
            Array::from_slice(&s_rel, &[n, k / 16]),
        ),
        (
            "p.weight_s_channel".to_string(),
            Array::from_slice(&s_channel, &[n]),
        ),
        (
            "p.comfy_quant".to_string(),
            Array::from_slice(&desc, &[desc.len() as i32]),
        ),
    ]);
    if bias {
        let b: Vec<f32> = (0..n).map(|_| rng.activation()).collect();
        map.insert(
            "p.bias".into(),
            Array::from_slice(&b, &[n])
                .as_dtype(Dtype::Bfloat16)
                .unwrap(),
        );
    }
    Weights::from_map(map)
}

fn w4a8(rng: &mut Rng, n: i32, k: i32, bias: bool) -> AdaptableLinear {
    let linear = quant::lin(&w4a8_operands(rng, n, k, bias, admissible_s_rel), "p", bias).unwrap();
    assert!(
        linear.w4a8_base().is_some(),
        "must reach the native W4A8 base"
    );
    linear
}

/// Packed W4A8: in-tile int4 decode vs the serial `[TILE_COLS, K]` decode scratch, including a
/// width that spans several reference tiles and ragged GEMM tiles, depths that are whole flush
/// windows (512, 1024, 3584) AND depths that end in a partial window (768, 1280, and H3's own
/// W4A8 K = 5376 = 10.5 x 512), and the -0.0 (0x80) group-scale byte.
#[test]
#[ignore = "executes the Metal INT8 GEMM; explicit synthetic-w4a8-mlx route only"]
fn tiled_w4a8_matches_serial_reference_on_random_packed_codes() {
    require_metal();
    // The 0x80 (-0.0) scale actually reaches the kernels, and a negative one never does.
    let mut probe = Rng(7);
    assert!((0..4096).any(|_| admissible_s_rel(&mut probe) == 0x80));
    negative_s_rel_codes_are_refused_before_any_kernel();
    let mut rng = Rng(0x3_26691);
    for (m, n, k, bias) in [
        (37, 300, 512, true),
        (65, 64, 1024, false),
        (3, 520, 3584, true),
        (70, 130, 768, true),
        (9, 66, 1280, false),
        (129, 200, 5376, true),
    ] {
        let linear = w4a8(&mut rng, n, k, bias);
        let base = linear.w4a8_base().unwrap();
        assert_eq!(
            base.tile_scratch_bytes(),
            0,
            "no decoded weight scratch on the tiled path"
        );
        for dtype in [Dtype::Float32, Dtype::Bfloat16, Dtype::Float16] {
            let x = activations(&mut rng, m, k, dtype);
            let tiled = base.forward(&x).unwrap();
            let reference = base.forward_reference(&x).unwrap();
            eval([&tiled, &reference]).unwrap();
            assert_identical(
                &format!("w4a8 m{m} n{n} k{k} {dtype:?}"),
                &tiled,
                &reference,
            );
        }
    }
}

/// Read only the named tensors of a (possibly ~21 GB) safetensors file into a `Weights` map.
fn real_slice(path: &std::path::Path, base: &str) -> Weights {
    let layout = weightsmeta::safetensors_file_tensor_locations(path).unwrap();
    let mut file = std::fs::File::open(path).unwrap();
    let mut map = HashMap::new();
    for suffix in ["weight", "weight_scale", "comfy_quant"] {
        let name = format!("{base}.{suffix}");
        let t = layout
            .tensors
            .iter()
            .find(|t| t.header.name == name)
            .unwrap_or_else(|| panic!("{} has no {name}", path.display()));
        let mut bytes = vec![0u8; t.header.data_bytes as usize];
        file.seek(SeekFrom::Start(t.file_offset)).unwrap();
        file.read_exact(&mut bytes).unwrap();
        let shape: Vec<i32> = t.header.shape.iter().map(|&d| d as i32).collect();
        let array = match t.header.dtype {
            StDtype::I8 => {
                let v: Vec<i8> = bytes.iter().map(|&b| b as i8).collect();
                Array::from_slice(&v, &shape)
            }
            StDtype::U8 => Array::from_slice(&bytes, &shape),
            StDtype::F32 => {
                let v: Vec<f32> = bytes
                    .chunks_exact(4)
                    .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
                    .collect();
                Array::from_slice(&v, &shape)
            }
            other => panic!("{name}: unexpected dtype {other:?}"),
        };
        map.insert(name, array);
    }
    Weights::from_map(map)
}

fn real_checkpoint() -> std::path::PathBuf {
    let path = std::env::var_os("H3_INT8_GEMM_REAL_SAFETENSORS").expect(
        "H3_INT8_GEMM_REAL_SAFETENSORS must name a ConvRot H3 checkpoint; absence is failure",
    );
    let path = std::path::PathBuf::from(path);
    assert!(path.is_file(), "{} is not a file", path.display());
    path
}

/// The published checkpoint's own I8 codes, F32 scales and `.comfy_quant` descriptor for every H3
/// projection geometry (attention QKV and out, FFN up and the K = 14336 FFN down), BF16 activations.
#[test]
#[ignore = "executes the Metal INT8 GEMM on a real checkpoint slice; explicit synthetic-w4a8-mlx route only"]
fn real_checkpoint_slice_matches_serial_reference() {
    require_metal();
    let path = real_checkpoint();
    println!("INT8_GEMM_REAL_SOURCE {}", path.display());
    let mut rng = Rng(0x4_26691);
    let mut checked = 0;
    for base in [
        "blocks.10.attn.qkv_proj",
        "blocks.10.attn.out_proj",
        "blocks.10.mlp.fc1",
        "blocks.10.mlp.fc2",
    ] {
        let w = real_slice(&path, base);
        let linear = quant::lin(&w, base, false).unwrap();
        let q = linear
            .convrot_base()
            .expect("real H3 projection must be native ConvRot INT8");
        let d = q.descriptor();
        println!(
            "INT8_GEMM_REAL {base} in={} out={} group={}",
            d.in_features, d.out_features, d.rotation_group
        );
        let x = activations(&mut rng, 96, d.in_features as i32, Dtype::Bfloat16);
        let (t, r) = convrot_pair(&linear, &x);
        assert_identical(&format!("real {base} bf16"), &t, &r);
        checked += 1;
    }
    assert_eq!(checked, 4);
}

fn time_forward(f: impl Fn() -> Array, iterations: u32) -> f64 {
    let warm = f();
    eval([&warm]).unwrap();
    drop(warm);
    let start = Instant::now();
    for _ in 0..iterations {
        let y = f();
        eval([&y]).unwrap();
    }
    start.elapsed().as_secs_f64() / f64::from(iterations)
}

/// Throughput of the four H3 projection geometries (synthetic codes, real shapes and group). The
/// serial reference is timed at a small row count (it is linear in rows) and scaled per row; the
/// per-step estimate scales the tiled time to H3's ~52.7k packed rows x 50 blocks. A measurement,
/// never a gate: it prints and asserts only that both paths ran and agreed.
#[test]
#[ignore = "Metal INT8 GEMM microbenchmark; explicit synthetic-w4a8-mlx route only"]
fn microbench_h3_projection_shapes() {
    require_metal();
    const STEP_ROWS: f64 = 52_683.0;
    const BLOCKS: f64 = 50.0;
    let (rows, ref_rows) = (4096i32, 64i32);
    let mut rng = Rng(0x5_26691);
    let (mut step_tiled, mut step_reference) = (0.0f64, 0.0f64);
    for (name, k, n) in [
        ("qkv_proj", 5376, 21504),
        ("out_proj", 7168, 5376),
        ("fc1", 5376, 28672),
        ("fc2", 14336, 5376),
    ] {
        let linear = random_convrot(&mut rng, n, k, 256, false);
        let base = linear.convrot_base().unwrap();
        let x = activations(&mut rng, rows, k, Dtype::Bfloat16);
        let small = activations(&mut rng, ref_rows, k, Dtype::Bfloat16);
        let (t, r) = convrot_pair(&linear, &small);
        assert_identical(&format!("bench {name}"), &t, &r);
        let tiled = time_forward(|| base.forward(&x).unwrap(), 5);
        let reference = time_forward(|| base.forward_reference(&small).unwrap(), 1);
        let (tiled_row, reference_row) = (tiled / f64::from(rows), reference / f64::from(ref_rows));
        let tflops = 2.0 * f64::from(rows) * f64::from(n) * f64::from(k) / tiled / 1e12;
        step_tiled += tiled_row * STEP_ROWS * BLOCKS;
        step_reference += reference_row * STEP_ROWS * BLOCKS;
        println!(
            "INT8_GEMM_BENCH {name} k={k} n={n} tiled_rows={rows} tiled_ms={:.3} tiled_tflops={tflops:.2} \
             reference_rows={ref_rows} reference_ms={:.3} speedup_per_row={:.1}x",
            tiled * 1e3,
            reference * 1e3,
            reference_row / tiled_row
        );
    }
    let linear = w4a8(&mut rng, 5376, 14336, false);
    let base = linear.w4a8_base().unwrap();
    let x = activations(&mut rng, rows, 14336, Dtype::Bfloat16);
    let tiled = time_forward(|| base.forward(&x).unwrap(), 5);
    println!(
        "INT8_GEMM_BENCH w4a8_fc2 k=14336 n=5376 tiled_rows={rows} tiled_ms={:.3} tiled_tflops={:.2}",
        tiled * 1e3,
        2.0 * f64::from(rows) * 5376.0 * 14336.0 / tiled / 1e12
    );
    println!(
        "INT8_GEMM_STEP_ESTIMATE rows={STEP_ROWS} blocks={BLOCKS} projections_tiled_s={step_tiled:.1} \
         projections_reference_s={step_reference:.1} speedup={:.1}x",
        step_reference / step_tiled
    );
}

/// The sign-set E4M3 bytes the parity generator omits are not an untested decode path: the
/// published contract refuses them at load, before any kernel exists. A plain helper (called by
/// the W4A8 parity body) so the route's executed-test count is unchanged.
fn negative_s_rel_codes_are_refused_before_any_kernel() {
    for byte in [0x81u8, 0x88, 0xB8, 0xFE] {
        let mut rng = Rng(u64::from(byte));
        let refused = quant::lin(
            &w4a8_operands(&mut rng, 8, 256, false, |_| byte),
            "p",
            false,
        );
        assert!(
            refused.is_err(),
            "negative s_rel byte {byte:#04x} must be refused at load"
        );
    }
}
