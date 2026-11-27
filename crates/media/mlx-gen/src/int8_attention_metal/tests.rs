//! SW-6141 tests: pure envelope / dispatch / layout-model tests (ordinary CI), and the one
//! registered Metal proof that measures correctness, the bench and the go/no-go in a single run.

use super::*;
use mlx_rs::memory::{clear_cache, get_active_memory, get_peak_memory, reset_peak_memory};
use mlx_rs::ops::{add, multiply, subtract};
use mlx_rs::transforms::eval;
use std::time::Instant;

const BF16: Dtype = Dtype::Bfloat16;

#[test]
fn check_admits_dit_shapes_and_refuses_the_rest() {
    let cfg = Int8AttentionConfig::default();
    let dit = [1, 24, 16384, 128];
    assert_eq!(
        check((&dit, BF16), (&dit, BF16), (&dit, BF16), &cfg),
        Ok(Plan {
            batch: 1,
            heads: 24,
            q_len: 16384,
            k_len: 16384
        })
    );
    let half = Dtype::Float16;
    assert!(check((&dit, half), (&dit, half), (&dit, half), &cfg).is_ok());
    // Cross-attention lengths: Sq and Sk may differ.
    let (q, kv) = ([2, 24, 4096, 128], [2, 24, 333, 128]);
    assert_eq!(
        check((&q, BF16), (&kv, BF16), (&kv, BF16), &cfg).map(|p| (p.q_len, p.k_len)),
        Ok((4096, 333))
    );

    let d64 = [1, 24, 4096, 64];
    assert_eq!(
        check((&d64, BF16), (&d64, BF16), (&d64, BF16), &cfg),
        Err(Refusal::HeadDim(64))
    );
    let rank3 = [24, 4096, 128];
    assert_eq!(
        check((&rank3, BF16), (&dit, BF16), (&dit, BF16), &cfg),
        Err(Refusal::Rank {
            operand: "q",
            rank: 3
        })
    );
    let float = Dtype::Float32;
    assert!(matches!(
        check((&dit, float), (&dit, float), (&dit, float), &cfg),
        Err(Refusal::Dtype(_))
    ));
    assert!(matches!(
        check((&dit, BF16), (&dit, half), (&dit, BF16), &cfg),
        Err(Refusal::Dtype(_))
    ));
    let other_len = [1, 24, 8192, 128];
    assert!(matches!(
        check((&dit, BF16), (&dit, BF16), (&other_len, BF16), &cfg),
        Err(Refusal::Shape(_))
    ));
    let other_heads = [1, 30, 16384, 128];
    assert!(matches!(
        check(
            (&dit, BF16),
            (&other_heads, BF16),
            (&other_heads, BF16),
            &cfg
        ),
        Err(Refusal::Shape(_))
    ));
    let empty = [1, 24, 0, 128];
    assert_eq!(
        check((&empty, BF16), (&dit, BF16), (&dit, BF16), &cfg),
        Err(Refusal::Empty)
    );
    // E6: 24 · 2^20 · 128 elements exceed i32::MAX.
    let huge = [1, 24, 1 << 20, 128];
    assert_eq!(
        check((&huge, BF16), (&huge, BF16), (&huge, BF16), &cfg),
        Err(Refusal::IndexOverflow)
    );
    for block_tokens in [0, MAX_BLOCK_TOKENS + 1] {
        let bad = Int8AttentionConfig {
            block_tokens,
            ..cfg
        };
        assert_eq!(
            check((&dit, BF16), (&dit, BF16), (&dit, BF16), &bad),
            Err(Refusal::BlockTokens(block_tokens))
        );
    }
    // A refusal surfaces as the typed `Unsupported` error.
    assert!(matches!(
        Error::from(Refusal::HeadDim(64)),
        Error::Unsupported(_)
    ));
}

#[test]
fn dispatch_geometry_covers_every_row_exactly_once() {
    for (rows, len, block) in [
        (1, 1, 1),
        (24, 4096, 32),
        (2, 333, 32),
        (2, 333, 5),
        (30, 16416, 1),
    ] {
        let ((gx, gy, gz), (tx, ty, tz)) = quantize_dispatch(rows, len, block);
        assert_eq!((gz, ty, tz), (1, 1, 1));
        assert_eq!(
            (tx as usize, gy as usize),
            (32 * block.min(QUANT_MAX_SIMD_GROUPS), rows)
        );
        assert_eq!(gx % tx, 0);
        let groups = (gx / tx) as usize;
        assert!(groups * block >= len && (groups - 1) * block < len);
    }
    for (route, tile) in [
        (QkRoute::Simd, SIMD_QUERY_ROWS),
        (QkRoute::Nax, NAX_QUERY_ROWS),
    ] {
        for (rows, len) in [(1, 1), (2, 333), (24, 4096), (24, 16384), (30, 16416)] {
            let ((gx, gy, gz), (tx, ty, tz)) = attend_dispatch(route, rows, len);
            assert_eq!((gz, ty, tz, gy as usize), (1, 1, 1, rows));
            assert_eq!(gx % tx, 0);
            let tiles = (gx / tx) as usize;
            assert!(tiles * tile >= len && (tiles - 1) * tile < len);
        }
    }
    // The kernels' fixed layouts: NAX eight SIMD groups of 16 rows, SIMD four of 8.
    assert_eq!(NAX_THREADS / 32 * 16, NAX_QUERY_ROWS);
    assert_eq!(SIMD_THREADS / 32 * 8, SIMD_QUERY_ROWS);
}

#[test]
fn the_assumed_nax_fragment_layout_is_a_bijection_and_the_probe_model_multiplies() {
    let mut seen = [[false; 16]; 16];
    for lane in 0..32 {
        for i in 0..8 {
            let (r, c) = nax_element(lane, i);
            assert!(!seen[r][c], "lane {lane} element {i} duplicates ({r}, {c})");
            seen[r][c] = true;
        }
    }
    assert!(seen.iter().flatten().all(|&x| x));

    // With A = I the model's product is C[m][n] = Bᵀ[n][m].
    let mut pa = vec![0i8; 256];
    for lane in 0..32 {
        for i in 0..8 {
            let (r, c) = nax_element(lane, i);
            pa[lane * 8 + i] = i8::from(r == c);
        }
    }
    let pb = probe_values(512, 7);
    let mut bt = [[0i8; 16]; 32];
    for lane in 0..32 {
        for i in 0..8 {
            let (r, c) = nax_element(lane, i);
            bt[r][c] = pb[lane * 16 + i];
            bt[16 + r][c] = pb[lane * 16 + 8 + i];
        }
    }
    let product = assumed_probe_product(&pa, &pb);
    for lane in 0..32 {
        for i in 0..16 {
            let (m, n) = nax_element(lane, i % 8);
            let n = n + 16 * (i / 8);
            assert_eq!(product[lane * 16 + i], f32::from(bt[n][m]));
        }
    }
    let values = probe_values(4096, 99);
    assert!(values.iter().all(|v| (-8..=8).contains(v)));
    assert!(values.iter().any(|&v| v == -8) && values.iter().any(|&v| v == 8));
}

#[test]
fn the_metal_sources_carry_the_cuda_twin_sign_words() {
    // SW-6131's `SAGE_SIGN_WORDS`, written out so a drift of either copy fails here.
    assert_eq!(
        SIGN_WORDS,
        [0x1035_997b, 0x8087_f5ee, 0xee2e_4e1a, 0x7113_2418]
    );
    for word in SIGN_WORDS {
        assert!(HEADER.contains(&format!("{word:#010x}u")), "{word:#010x}");
    }
    // Each body references only the kernel's declared inputs/outputs and template names.
    assert!(QUANTIZE.contains("QB") && QUANTIZE.contains("ROTATE"));
    // The quantizer's SIMD-group cap matches the host dispatch.
    assert!(QUANTIZE.contains(&format!("uint(QB) < {QUANT_MAX_SIMD_GROUPS}u")));
    assert!(QUANTIZE.contains(&format!("block_peak[{QUANT_MAX_SIMD_GROUPS}]")));
    assert!(ATTEND_NAX.contains("QUANT"));
    assert!(NAX_HEADER.contains("MetalPerformancePrimitives"));
    assert!(!HEADER.contains("MetalPerformancePrimitives"));
    // The array-form diagnostic helper stays out of the attention kernels' headers, so a compile
    // failure of either form cannot take the other down.
    assert!(NAX_ARRAY_HEADER.contains("sw6141_nax_mma_arr"));
    assert!(!NAX_HEADER.contains("sw6141_nax_mma_arr") && !HEADER.contains("sw6141_nax_mma_arr"));
}

// ---------------------------------------------------------------------------------------------
// The registered Metal proof.
// ---------------------------------------------------------------------------------------------

/// Heads of the bench shape (Krea 2 / FLUX.1-class DiTs: 24 heads of 128).
const HEADS: usize = 24;
/// Sequence lengths of the AC: 1024² and 2048² image tokens.
const LENGTHS: [usize; 2] = [4096, 16384];
/// The AC's accuracy floor against MLX's fused SDPA.
const MIN_COSINE: f64 = 0.999;
/// GO requires the INT8 call (quantization included) at most this fraction of fused SDPA's time
/// at every length: a >= 5% win, well outside run-to-run noise.
const GO_TIME_RATIO: f64 = 0.95;

fn chip() -> String {
    std::process::Command::new("sysctl")
        .args(["-n", "machdep.cpu.brand_string"])
        .output()
        .ok()
        .and_then(|o| String::from_utf8(o.stdout).ok())
        .map(|s| s.trim().to_string())
        .filter(|s| !s.is_empty())
        .unwrap_or_else(|| "unknown chip".to_string())
}

fn mib(bytes: usize) -> f64 {
    bytes as f64 / (1024.0 * 1024.0)
}

fn normal(shape: &[i32], seed: u64) -> Array {
    let key = mlx_rs::random::key(seed).unwrap();
    mlx_rs::random::normal::<f32>(shape, None, None, Some(&key)).unwrap()
}

fn settle(a: Array) -> Array {
    let a = a.as_dtype(BF16).unwrap();
    eval([&a]).unwrap();
    a
}

/// Pinned bf16 Q/K/V `[1, heads, s, 128]`.
struct Operands {
    q: Array,
    k: Array,
    v: Array,
}

impl Operands {
    /// i.i.d. N(0, 1).
    fn random(heads: usize, s: usize, seed: u64) -> Self {
        let shape = [1, heads as i32, s as i32, HEAD_DIM as i32];
        Self {
            q: settle(normal(&shape, seed)),
            k: settle(normal(&shape, seed + 1)),
            v: settle(normal(&shape, seed + 2)),
        }
    }

    /// DiT-like statistics: per-channel gains 2 with four outlier channels at 6 (logit std ~7.5,
    /// a peaked softmax) and a large shared K offset (+6 on eight channels) — the structure the
    /// rotation and the K centering exist for.
    fn dit_like(heads: usize, s: usize, seed: u64) -> Self {
        let shape = [1, heads as i32, s as i32, HEAD_DIM as i32];
        let mut gain = vec![2.0f32; HEAD_DIM];
        for c in [3, 37, 64, 101] {
            gain[c] = 6.0;
        }
        let mut offset = vec![0.0f32; HEAD_DIM];
        offset[..8].fill(6.0);
        let gain = Array::from_slice(&gain, &[HEAD_DIM as i32]);
        let offset = Array::from_slice(&offset, &[HEAD_DIM as i32]);
        Self {
            q: settle(multiply(normal(&shape, seed), &gain).unwrap()),
            k: settle(add(multiply(normal(&shape, seed + 1), &gain).unwrap(), &offset).unwrap()),
            v: settle(normal(&shape, seed + 2)),
        }
    }
}

/// `(cosine, relative L2 error)` of `a` against `reference`, in f32.
fn agreement(a: &Array, reference: &Array) -> Result<(f64, f64)> {
    let a = a.as_dtype(Dtype::Float32)?;
    let b = reference.as_dtype(Dtype::Float32)?;
    let d = subtract(&a, &b)?;
    let dot = multiply(&a, &b)?.sum(None)?;
    let na = multiply(&a, &a)?.sum(None)?;
    let nb = multiply(&b, &b)?.sum(None)?;
    let nd = multiply(&d, &d)?.sum(None)?;
    eval([&dot, &na, &nb, &nd])?;
    let (dot, na, nb, nd) = (
        f64::from(dot.item::<f32>()),
        f64::from(na.item::<f32>()),
        f64::from(nb.item::<f32>()),
        f64::from(nd.item::<f32>()),
    );
    Ok((dot / (na * nb).sqrt(), (nd / nb).sqrt()))
}

/// Median ms/iter and the transient peak (MiB above the pinned operands) of `f`; warmup includes
/// the JIT compile.
fn time_it(
    warmup: usize,
    iters: usize,
    mut f: impl FnMut() -> Result<Vec<Array>>,
) -> Result<(f64, f64)> {
    for _ in 0..warmup {
        let out = f()?;
        eval(out.iter())?;
    }
    clear_cache();
    let base = get_active_memory();
    reset_peak_memory();
    let mut times = Vec::with_capacity(iters);
    for _ in 0..iters {
        let t0 = Instant::now();
        let out = f()?;
        eval(out.iter())?;
        times.push(t0.elapsed().as_secs_f64() * 1e3);
    }
    let peak = get_peak_memory().saturating_sub(base);
    times.sort_by(f64::total_cmp);
    Ok((times[times.len() / 2], mib(peak)))
}

/// Peak MiB of one call whose Q/K/V are **intermediates** of the evaluated graph (`x · 1` of the
/// pinned operands, every handle dropped before `eval`), as inside a DiT forward: MLX frees Q and
/// K once their last consumer ran, so the INT8 path's codes replace them instead of adding to them.
fn graph_peak(ops: &Operands, f: &AttentionFn<'_>) -> Result<f64> {
    let one = Array::from_f32(1.0).as_dtype(ops.q.dtype())?;
    eval([&one])?;
    clear_cache();
    let base = get_active_memory();
    reset_peak_memory();
    let out = {
        let q = multiply(&ops.q, &one)?;
        let k = multiply(&ops.k, &one)?;
        let v = multiply(&ops.v, &one)?;
        f(&q, &k, &v)?
    };
    eval([&out])?;
    let peak = get_peak_memory().saturating_sub(base);
    drop(out);
    Ok(mib(peak))
}

/// The NAX INT8 product's standing after the layout probe.
#[derive(Clone, Debug, PartialEq)]
enum NaxInt8 {
    /// No neural accelerators (MLX's predicate).
    Unavailable,
    /// The Metal compiler rejected the INT8 `matmul2d` (conclusive: no INT8 accelerator path).
    Rejected(String),
    /// Compiles; the INT8 cooperative tensors use the 16-bit layout.
    LayoutOk,
    /// Compiles, but the INT8 layout differs from the 16-bit one (kernel needs a layout fix).
    LayoutMismatch,
    /// INT8 `matmul2d` compiles over thread arrays but not over 8-wide INT8 vectors: the kernel's
    /// fragment form needs a fix before the route can be measured.
    VectorFormRejected(String),
}

fn report_probe(probe: &LayoutProbe) {
    println!(
        "SW6141 NAX layout probe: 16-bit product matches the BaseNAXFrag model: {}; INT8 product \
         matches the 16-bit product: {}",
        probe.sixteen_bit_matches_assumed(),
        probe.int8_matches_sixteen_bit()
    );
    if !probe.int8_matches_sixteen_bit() || !probe.sixteen_bit_matches_assumed() {
        for lane in 0..32 {
            let span = lane * 16..lane * 16 + 16;
            println!(
                "SW6141 probe lane {lane:2}: int8 {:?} | bf16 {:?} | model {:?}",
                &probe.int8[span.clone()],
                &probe.sixteen_bit[span.clone()],
                &probe.assumed[span]
            );
        }
    }
}

struct Accuracy {
    s: usize,
    inputs: &'static str,
    arm: &'static str,
    result: std::result::Result<(f64, f64, f64), String>,
}

struct Bench {
    s: usize,
    arm: &'static str,
    result: std::result::Result<(f64, f64), String>,
    graph_peak: Option<std::result::Result<f64, String>>,
}

fn accuracy_of(
    out: Result<Array>,
    ref16: &Array,
    ref32: &Array,
) -> std::result::Result<(f64, f64, f64), String> {
    let out = out.map_err(|e| e.to_string())?;
    let (c16, l2) = agreement(&out, ref16).map_err(|e| e.to_string())?;
    let (c32, _) = agreement(&out, ref32).map_err(|e| e.to_string())?;
    Ok((c16, l2, c32))
}

/// One attention call over `(q, k, v)`, for the graph-regime peak.
type AttentionFn<'a> = dyn Fn(&Array, &Array, &Array) -> Result<Array> + 'a;

/// One accuracy arm: a lazily built output over pinned operands.
type Arm<'a> = Box<dyn Fn() -> Result<Array> + 'a>;

const SIMD_DEFAULT: &str = "simd int8 (block 32, K centered)";
const SIMD_PER_TOKEN: &str = "simd int8 (per-token, K centered)";
const SIMD_UNCENTERED: &str = "simd int8 (block 32, no centering)";
const MUTANT: &str = "MUTANT simd int8 (K not rotated)";
const NAX_CONTROL: &str = "nax bf16 control (same kernel, no quant)";
const NAX_DEFAULT: &str = "nax int8 (block 32, K centered)";

#[test]
#[ignore = "model inference; explicit model-proof workflow"]
fn int8_attention_metal_parity_bench_and_go_no_go() {
    let started = Instant::now();
    assert!(crate::metal_available());
    assert!(matches!(
        mlx_rs::Device::try_default().unwrap().get_type().unwrap(),
        mlx_rs::DeviceType::Gpu
    ));
    let chip = chip();
    let nax = nax_available();
    println!(
        "\nSW6141 chip: {chip}; MLX neural accelerators (NAX): {nax}; bench shape B=1, H={HEADS}, \
         D={HEAD_DIM}, bf16 Q/K/V, S in {LENGTHS:?}"
    );
    let mut failures: Vec<String> = Vec::new();
    let default = Int8AttentionConfig::default();
    let per_token = Int8AttentionConfig {
        block_tokens: 1,
        ..default
    };
    let uncentered = Int8AttentionConfig {
        center_k: false,
        ..default
    };
    let k_unrotated = Int8AttentionConfig {
        rotate_k: false,
        ..default
    };

    // 1. Ragged tails (S = 333, and Sq = 200 over Sk = 333) on the portable route and the NAX
    //    control: every kernel's partial query tile, key block and quantization block.
    let small = Operands::random(2, 333, 11);
    let cross_q = settle(normal(&[1, 2, 200, HEAD_DIM as i32], 17));
    let mut small_checks: Vec<(&str, Result<Array>, Array)> = vec![
        (
            "ragged S=333 simd int8",
            int8_attention(&small.q, &small.k, &small.v, &default, QkRoute::Simd),
            fused_sdpa(&small.q, &small.k, &small.v).unwrap(),
        ),
        (
            "ragged Sq=200/Sk=333 simd int8",
            int8_attention(&cross_q, &small.k, &small.v, &default, QkRoute::Simd),
            fused_sdpa(&cross_q, &small.k, &small.v).unwrap(),
        ),
    ];
    if nax {
        small_checks.push((
            "ragged S=333 nax bf16 control",
            attend_nax_unquantized(&small.q, &small.k, &small.v),
            fused_sdpa(&small.q, &small.k, &small.v).unwrap(),
        ));
    }
    for (name, out, reference) in small_checks {
        match out.and_then(|o| agreement(&o, &reference)) {
            Ok((cos, l2)) => {
                println!("SW6141 {name}: cosine {cos:.6}, rel-L2 {l2:.3e}");
                if name.contains("simd") && cos < MIN_COSINE {
                    failures.push(format!("{name}: cosine {cos:.6} < {MIN_COSINE}"));
                }
            }
            Err(e) => {
                println!("SW6141 {name}: ERROR {e}");
                if name.contains("simd") {
                    failures.push(format!("{name}: {e}"));
                }
            }
        }
    }

    // 2. The INT8 `matmul2d` layout probe (after the proven 16-bit NAX control compiled).
    const PROBE_SEED: u64 = 0x5eed_6141;
    let mut nax_int8 = if !nax {
        NaxInt8::Unavailable
    } else {
        let vector_probe = nax_layout_probe(PROBE_SEED);
        let array_probe = nax_layout_probe_arrays(PROBE_SEED);
        let (pa, pb) = probe_inputs(PROBE_SEED);
        let model = assumed_probe_product(&pa, &pb);
        match &array_probe {
            Ok(ci) => println!(
                "SW6141 NAX INT8 matmul2d over thread arrays compiles; matches the BaseNAXFrag \
                 model: {}",
                ci.iter().zip(&model).all(|(&i, &m)| i as f32 == m)
            ),
            Err(e) => println!("SW6141 NAX INT8 matmul2d over thread arrays: ERROR {e}"),
        }
        match vector_probe {
            Ok(probe) => {
                report_probe(&probe);
                if probe.int8_matches_sixteen_bit() {
                    NaxInt8::LayoutOk
                } else {
                    NaxInt8::LayoutMismatch
                }
            }
            Err(e) => {
                println!("SW6141 NAX INT8 matmul2d over 8-wide vectors: ERROR {e}");
                if array_probe.is_ok() {
                    NaxInt8::VectorFormRejected(e.to_string())
                } else {
                    NaxInt8::Rejected(e.to_string())
                }
            }
        }
    };
    let nax_int8_runs = matches!(nax_int8, NaxInt8::LayoutOk | NaxInt8::LayoutMismatch);
    if nax_int8_runs {
        let out = int8_attention(&small.q, &small.k, &small.v, &default, QkRoute::Nax);
        let reference = fused_sdpa(&small.q, &small.k, &small.v).unwrap();
        match out.and_then(|o| agreement(&o, &reference)) {
            Ok((cos, l2)) => {
                println!("SW6141 ragged S=333 nax int8: cosine {cos:.6}, rel-L2 {l2:.3e}")
            }
            Err(e) => println!("SW6141 ragged S=333 nax int8: ERROR {e}"),
        }
    }

    // 3. Accuracy and bench at the AC lengths.
    let mut accuracy: Vec<Accuracy> = Vec::new();
    let mut bench: Vec<Bench> = Vec::new();
    for &s in &LENGTHS {
        let (warmup, iters) = if s <= 4096 { (3, 10) } else { (2, 6) };
        let random = Operands::random(HEADS, s, 1000 + s as u64);
        let dit = Operands::dit_like(HEADS, s, 2000 + s as u64);
        for (inputs, ops) in [("random", &random), ("dit-like", &dit)] {
            let ref16 = fused_sdpa(&ops.q, &ops.k, &ops.v).unwrap();
            let (q32, k32, v32) = (
                ops.q.as_dtype(Dtype::Float32).unwrap(),
                ops.k.as_dtype(Dtype::Float32).unwrap(),
                ops.v.as_dtype(Dtype::Float32).unwrap(),
            );
            let ref32 = fused_sdpa(&q32, &k32, &v32).unwrap();
            eval([&ref16, &ref32]).unwrap();
            drop((q32, k32, v32));
            let mut arms: Vec<(&'static str, Arm<'_>)> = Vec::new();
            for (arm, config) in [
                (SIMD_DEFAULT, default),
                (SIMD_PER_TOKEN, per_token),
                (SIMD_UNCENTERED, uncentered),
                (MUTANT, k_unrotated),
            ] {
                arms.push((
                    arm,
                    Box::new(move || {
                        int8_attention(&ops.q, &ops.k, &ops.v, &config, QkRoute::Simd)
                    }),
                ));
            }
            if nax {
                arms.push((
                    NAX_CONTROL,
                    Box::new(move || attend_nax_unquantized(&ops.q, &ops.k, &ops.v)),
                ));
            }
            if nax_int8_runs {
                arms.push((
                    NAX_DEFAULT,
                    Box::new(move || {
                        int8_attention(&ops.q, &ops.k, &ops.v, &default, QkRoute::Nax)
                    }),
                ));
            }
            for (arm, run) in arms {
                let result = accuracy_of(run(), &ref16, &ref32);
                match &result {
                    Ok((c16, l2, c32)) => println!(
                        "SW6141 accuracy S={s} {inputs} {arm}: cosine vs fused SDPA {c16:.6} \
                         (rel-L2 {l2:.3e}), vs f32 SDPA {c32:.6}"
                    ),
                    Err(e) => println!("SW6141 accuracy S={s} {inputs} {arm}: ERROR {e}"),
                }
                accuracy.push(Accuracy {
                    s,
                    inputs,
                    arm,
                    result,
                });
            }
        }
        drop(dit);

        // Bench on the random operands (time does not depend on the values).
        let ops = &random;
        let pre = prequantize(&ops.q, &ops.k, &ops.v, &default).unwrap();
        eval([&pre.q_codes, &pre.q_scales, &pre.k_codes, &pre.k_scales]).unwrap();
        let sdpa_fn = |q: &Array, k: &Array, v: &Array| fused_sdpa(q, k, v);
        let simd_fn =
            |q: &Array, k: &Array, v: &Array| int8_attention(q, k, v, &default, QkRoute::Simd);
        let nax_fn =
            |q: &Array, k: &Array, v: &Array| int8_attention(q, k, v, &default, QkRoute::Nax);
        let mut row =
            |arm: &'static str, timed: Result<(f64, f64)>, graph: Option<&AttentionFn<'_>>| {
                let graph_peak = graph.map(|f| graph_peak(ops, f).map_err(|e| e.to_string()));
                bench.push(Bench {
                    s,
                    arm,
                    result: timed.map_err(|e| e.to_string()),
                    graph_peak,
                });
            };
        row(
            "mlx fused sdpa (bf16)",
            time_it(warmup, iters, || {
                Ok(vec![fused_sdpa(&ops.q, &ops.k, &ops.v)?])
            }),
            Some(&sdpa_fn),
        );
        row(
            "quantize Q+K only (block 32, K centered)",
            time_it(warmup, iters, || {
                let qk = prequantize(&ops.q, &ops.k, &ops.v, &default)?;
                Ok(vec![qk.q_codes, qk.q_scales, qk.k_codes, qk.k_scales])
            }),
            None,
        );
        row(
            "simd int8 attend only (prequantized)",
            time_it(warmup, iters, || {
                Ok(vec![attend_quantized(&pre, &ops.v, QkRoute::Simd)?])
            }),
            None,
        );
        row(
            "simd int8 total (quantize + attend)",
            time_it(warmup, iters, || Ok(vec![simd_fn(&ops.q, &ops.k, &ops.v)?])),
            Some(&simd_fn),
        );
        if nax {
            row(
                "nax bf16 control (no quant)",
                time_it(warmup, iters, || {
                    Ok(vec![attend_nax_unquantized(&ops.q, &ops.k, &ops.v)?])
                }),
                None,
            );
        }
        if nax_int8_runs {
            row(
                "nax int8 attend only (prequantized)",
                time_it(warmup, iters, || {
                    Ok(vec![attend_quantized(&pre, &ops.v, QkRoute::Nax)?])
                }),
                None,
            );
            row(
                "nax int8 total (quantize + attend)",
                time_it(warmup, iters, || Ok(vec![nax_fn(&ops.q, &ops.k, &ops.v)?])),
                Some(&nax_fn),
            );
        }
        drop(pre);
    }

    // 4. Correctness verdicts per arm.
    let correct_at = |arm: &str, s: usize| {
        let rows: Vec<_> = accuracy
            .iter()
            .filter(|a| a.arm == arm && a.s == s)
            .collect();
        !rows.is_empty()
            && rows
                .iter()
                .all(|a| matches!(a.result, Ok((c16, _, _)) if c16 >= MIN_COSINE))
    };
    for &s in &LENGTHS {
        if !correct_at(SIMD_DEFAULT, s) {
            failures.push(format!("{SIMD_DEFAULT} below cosine {MIN_COSINE} at S={s}"));
        }
        // The mutation must be caught: skipping the rotation on K only breaks Q·Kᵀ.
        for a in accuracy.iter().filter(|a| a.arm == MUTANT && a.s == s) {
            match a.result {
                Ok((c16, _, _)) if c16 < MIN_COSINE => {}
                Ok((c16, _, _)) => failures.push(format!(
                    "mutation not caught: {MUTANT} at S={s} {} has cosine {c16:.6}",
                    a.inputs
                )),
                Err(ref e) => failures.push(format!("{MUTANT} at S={s}: {e}")),
            }
        }
        if nax_int8_runs && !correct_at(NAX_DEFAULT, s) && nax_int8 == NaxInt8::LayoutOk {
            // Layout matched the probe but the kernel is still wrong: a kernel defect, not a
            // measurement.
            nax_int8 = NaxInt8::LayoutMismatch;
        }
    }

    // 5. Tables.
    println!("\n### SW-6141 INT8 QK attention vs MLX fused SDPA — {chip} (NAX: {nax})\n");
    println!("| S | inputs | arm | cosine vs fused SDPA | rel-L2 | cosine vs f32 SDPA |");
    println!("|---:|---|---|---:|---:|---:|");
    for a in &accuracy {
        match &a.result {
            Ok((c16, l2, c32)) => println!(
                "| {} | {} | {} | {c16:.6} | {l2:.3e} | {c32:.6} |",
                a.s, a.inputs, a.arm
            ),
            Err(e) => println!("| {} | {} | {} | ERROR: {e} | | |", a.s, a.inputs, a.arm),
        }
    }
    println!(
        "\n| S | arm | median ms/iter | x fused SDPA | transient peak MiB (pinned Q/K/V) | peak MiB \
         (Q/K/V as graph intermediates) |"
    );
    println!("|---:|---|---:|---:|---:|---:|");
    let sdpa_of = |s: usize| {
        bench
            .iter()
            .find(|b| b.s == s && b.arm.starts_with("mlx fused sdpa"))
            .and_then(|b| b.result.as_ref().ok().map(|r| r.0))
    };
    for b in &bench {
        let graph = match &b.graph_peak {
            Some(Ok(p)) => format!("{p:.1}"),
            Some(Err(e)) => format!("ERROR: {e}"),
            None => "-".into(),
        };
        match &b.result {
            Ok((ms, peak)) => {
                let ratio = sdpa_of(b.s).map_or("-".into(), |base| format!("{:.3}", ms / base));
                println!(
                    "| {} | {} | {ms:.3} | {ratio} | {peak:.1} | {graph} |",
                    b.s, b.arm
                )
            }
            Err(e) => println!("| {} | {} | ERROR: {e} | | | {graph} |", b.s, b.arm),
        }
    }

    // 6. Go / no-go: an INT8 route that is correct at every length, at most GO_TIME_RATIO of fused
    //    SDPA's time (quantization included) and no higher graph-regime peak, at every length.
    let measured = |arm: &str, s: usize| {
        bench
            .iter()
            .find(|b| b.s == s && b.arm == arm)
            .and_then(|b| {
                let (ms, _) = *b.result.as_ref().ok()?;
                let peak = *b.graph_peak.as_ref()?.as_ref().ok()?;
                Some((ms, peak))
            })
    };
    let candidates: Vec<(&str, &str)> = {
        let mut c = vec![(SIMD_DEFAULT, "simd int8 total (quantize + attend)")];
        if nax_int8_runs {
            c.push((NAX_DEFAULT, "nax int8 total (quantize + attend)"));
        }
        c
    };
    let mut reasons = Vec::new();
    let mut go_everywhere = true;
    for &s in &LENGTHS {
        let Some((sdpa_ms, sdpa_peak)) = measured("mlx fused sdpa (bf16)", s) else {
            failures.push(format!("fused SDPA bench missing at S={s}"));
            go_everywhere = false;
            continue;
        };
        let best = candidates
            .iter()
            .filter(|candidate| correct_at(candidate.0, s))
            .filter_map(|&(_, bench_arm)| measured(bench_arm, s).map(|m| (bench_arm, m)))
            .min_by(|a, b| a.1 .0.total_cmp(&b.1 .0));
        match best {
            Some((arm, (ms, peak))) => {
                let go = ms <= GO_TIME_RATIO * sdpa_ms && peak <= sdpa_peak;
                reasons.push(format!(
                    "S={s}: best correct INT8 {arm} {ms:.3} ms = {:.3}x fused SDPA {sdpa_ms:.3} ms, \
                     graph peak {peak:.1} vs {sdpa_peak:.1} MiB -> {}",
                    ms / sdpa_ms,
                    if go { "go" } else { "no-go" }
                ));
                go_everywhere &= go;
            }
            None => {
                reasons.push(format!("S={s}: no correct, measured INT8 route"));
                go_everywhere = false;
            }
        }
    }
    let verdict = match &nax_int8 {
        NaxInt8::LayoutMismatch | NaxInt8::VectorFormRejected(_) => "INCONCLUSIVE",
        _ if go_everywhere => "GO",
        _ => "NO-GO",
    };
    let nax_note = match &nax_int8 {
        NaxInt8::Unavailable => {
            "no neural accelerators: INT8 has no matrix-unit path on this GPU".to_string()
        }
        NaxInt8::Rejected(e) => {
            format!("Metal rejected INT8 matmul2d ({e}): no INT8 accelerator path")
        }
        NaxInt8::LayoutOk => "NAX INT8 matmul2d measured".to_string(),
        NaxInt8::LayoutMismatch => {
            "NAX INT8 kernel incorrect (see the probe lines): fix and re-run".to_string()
        }
        NaxInt8::VectorFormRejected(e) => format!(
            "INT8 matmul2d works over thread arrays but not 8-wide INT8 vectors ({e}): fix the \
             fragment form and re-run"
        ),
    };
    println!(
        "\nSW6141_VERDICT {verdict} on {chip}: {}; {nax_note}; criterion: correct (cosine >= \
         {MIN_COSINE}) INT8 total <= {GO_TIME_RATIO} x fused SDPA ms and graph peak <= fused SDPA \
         at every S in {LENGTHS:?}",
        reasons.join("; ")
    );
    println!(
        "SW6141 proof wall time {:.1} s",
        started.elapsed().as_secs_f64()
    );

    assert!(failures.is_empty(), "SW-6141 failures: {failures:#?}");
    assert_ne!(
        verdict, "INCONCLUSIVE",
        "the NAX INT8 kernel did not produce a correct result; the go/no-go is not measured"
    );
    crate::model_inference_device::record_model_inference_device();
    println!(
        "\nMODEL_INFERENCE_BODY_COMPLETED int8_attention_metal::tests::int8_attention_metal_parity_bench_and_go_no_go"
    );
}
