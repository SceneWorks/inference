//! sc-11044 GPU gates for the **on-device NVFP4 W4A4 activation-quant path** (epic 11037) on consumer
//! Blackwell `sm_120`. CUDA-only (needs a live cuBLASLt handle **and** an `sm_120` device); a graceful
//! no-op on CPU/Metal and on pre-Blackwell CUDA.
//!
//! What this pins on the real GPU:
//!
//! 1. **On-device activation quantize == the CPU `Nvfp4Tensor::pack` reference** — quantizing the
//!    activation entirely on the GPU (`CublasLt::quantize_nvfp4_activation`) and running the FP4 GEMM
//!    matches, within a tiny rel-RMS, the same GEMM fed by the old CPU-round-trip pack. The W4A4
//!    forward is therefore fully on-device (no host round-trip) and still numerically faithful.
//! 2. **W4A4 forward is finite + tracks bf16** — `Nvfp4Linear::forward_checked` (NaN/inf guard) never
//!    NaNs across repeated forwards (a stand-in for denoise steps) and stays within NVFP4 tolerance of
//!    a bf16-dense reference.
//! 3. **Throughput: W4A4 vs W4A16 (and bf16)** — on representative Sana-DiT GEMM shapes, on the
//!    now-exclusive GPU, timed layer forwards. Reports the multiple (informational, not asserted —
//!    hardware-dependent).
//! 4. **Outlier-sparsity capture confirms the partition** — the spike residual-gate metric
//!    (`OutlierSparsity`) on synthetic benign / sparse / dense activations classifies as expected, and
//!    the benign→W4A4 / dense→W4A16 partition holds.
//! 5. **Real Sana-1.6B DiT weight (`#[ignore]`, env-gated)** — if `SC11044_SANA_DIT_SAFETENSORS`
//!    points at a Sana transformer shard, a real projection weight is quantized and run W4A4 vs bf16.
//!
//! Alongside (1), and deliberately **not** folded into it, sit the sc-12078 **byte-level tie gates**
//! (`*_at_exact_ties`): crafted E4M3/E2M1 midpoints asserted at the raw emitted scale bytes / element
//! nibbles against the canonical `e4m3_from_f32` / `e2m1_from_f32`, on both the fused and unfused
//! quantize routes. Every rel-RMS gate above is blind to these — exact midpoints are measure-zero
//! under random activations, so (1) reads 0.000000 while every tie is encoded wrong.

#![cfg(feature = "cuda")]

use candle_gen::candle_core::{DType, Device, Tensor};
use candle_gen::quant::nvfp4::{e2m1_from_f32, e4m3_from_f32, e4m3_to_f32, Nvfp4Tensor, E2M1_LUT};
use candle_gen::quant::{
    ActPrecision, CublasLt, DevNvfp4, Nvfp4Linear, Nvfp4Regime, OutlierClass, OutlierSparsity,
};
use std::time::Instant;

/// splitmix64-hashed deterministic pseudo-random in ~[-1, 1] (launch-portable; no device RNG).
fn pseudo_random(n: usize, seed: u64) -> Vec<f32> {
    (0..n)
        .map(|i| {
            let mut z = (i as u64)
                .wrapping_add(seed)
                .wrapping_add(0x9E37_79B9_7F4A_7C15);
            z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
            z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
            z ^= z >> 31;
            (z as f64 / u64::MAX as f64) as f32 * 2.0 - 1.0
        })
        .collect()
}

fn rel_rms(got: &[f32], reference: &[f32]) -> f32 {
    let (mut num, mut den) = (0f64, 0f64);
    for (g, r) in got.iter().zip(reference) {
        num += (*g as f64 - *r as f64).powi(2);
        den += (*r as f64).powi(2);
    }
    (num / den.max(1e-30)).sqrt() as f32
}

/// CPU f32 reference `X·Wᵀ` for `X=[M,K]`, `W=[N,K]` row-major → `[M,N]`.
fn ref_matmul(x: &[f32], w: &[f32], m: usize, k: usize, n: usize) -> Vec<f32> {
    let mut out = vec![0f32; m * n];
    for r in 0..m {
        for c in 0..n {
            let mut acc = 0f64;
            for kk in 0..k {
                acc += (x[r * k + kk] as f64) * (w[c * k + kk] as f64);
            }
            out[r * n + c] = acc as f32;
        }
    }
    out
}

/// A CUDA device iff one exists and it is Blackwell `sm_120`+ — else `None` with a SKIP note.
fn nvfp4_device() -> Option<Device> {
    let dev = match Device::cuda_if_available(0) {
        Ok(d @ Device::Cuda(_)) => d,
        _ => {
            eprintln!("[sc-11044] no CUDA device; skipping on-device W4A4 GPU gate");
            return None;
        }
    };
    let lt = CublasLt::new(&dev).expect("cuBLASLt handle");
    match lt.meets_nvfp4_floor() {
        Ok(true) => {
            eprintln!(
                "[sc-11044] device cap = {:?} (NVFP4 eligible)",
                lt.compute_cap().unwrap()
            );
            Some(dev)
        }
        _ => {
            eprintln!(
                "[sc-11044] device not sm_120 ({:?}); skipping",
                lt.compute_cap().ok()
            );
            None
        }
    }
}

fn to_vec_f32(t: &Tensor) -> Vec<f32> {
    t.to_dtype(DType::F32)
        .unwrap()
        .flatten_all()
        .unwrap()
        .to_vec1::<f32>()
        .unwrap()
}

/// (1) The on-device activation quantize matches the CPU `Nvfp4Tensor::pack` reference: feeding the
/// same resident weight, the FP4 GEMM output from the on-device-quantized activation tracks the output
/// from the CPU-packed activation within a tiny rel-RMS — proving the W4A4 forward is fully on-device
/// (no host round-trip) without changing the numerics. Also checks both track the CPU dequant math.
#[test]
fn nvfp4_w4a4_ondevice_activation_quant_matches_cpu_pack_ref() {
    let Some(dev) = nvfp4_device() else { return };
    let lt = CublasLt::new(&dev).unwrap();
    let (m, k, n) = (256usize, 256usize, 128usize); // K%32==0, N%16==0, M%16==0

    let x_f32 = pseudo_random(m * k, 101);
    let w_f32 = pseudo_random(n * k, 202);
    let x = Tensor::from_vec(x_f32.clone(), (m, k), &dev)
        .unwrap()
        .to_dtype(DType::BF16)
        .unwrap();
    let w = Tensor::from_vec(w_f32.clone(), (n, k), &dev)
        .unwrap()
        .to_dtype(DType::BF16)
        .unwrap();

    // Resident weight (staged once from the CPU packer, as in sc-11041).
    let w_pk = Nvfp4Tensor::pack(&w).unwrap();
    let w_stg = lt.stage_nvfp4(&w_pk).unwrap();

    // On-device activation quantize (the sc-11044 net-new path) vs the old CPU-round-trip pack.
    let x_dev = lt.quantize_nvfp4_activation(&x, w_pk.cols_padded).unwrap();
    let x_cpu = lt.stage_nvfp4(&Nvfp4Tensor::pack(&x).unwrap()).unwrap();

    let y_dev = to_vec_f32(&lt.matmul_nvfp4_staged(&w_stg, &x_dev).unwrap());
    let y_cpu = to_vec_f32(&lt.matmul_nvfp4_staged(&w_stg, &x_cpu).unwrap());
    assert!(
        y_dev.iter().all(|v| v.is_finite()),
        "on-device quant produced NaN/Inf"
    );

    let rr = rel_rms(&y_dev, &y_cpu);
    eprintln!("[sc-11044] on-device vs CPU-pack activation quant: GEMM rel-RMS = {rr:.6}");
    assert!(
        rr < 0.02,
        "on-device activation quant diverges from the CPU pack ref (rel-RMS {rr:.6})"
    );

    // Both also track the CPU dequant reference (X_dq · W_dqᵀ).
    let x_pk = Nvfp4Tensor::pack(&x).unwrap();
    let dq_ref = ref_matmul(
        &x_pk.dequantize_to_vec(),
        &w_pk.dequantize_to_vec(),
        m,
        k,
        n,
    );
    let rr_dq = rel_rms(&y_dev, &dq_ref);
    eprintln!("[sc-11044] on-device W4A4 vs CPU dequant reference rel-RMS = {rr_dq:.5}");
    assert!(
        rr_dq < 0.03,
        "on-device W4A4 does not track the dequant reference (rel-RMS {rr_dq:.5})"
    );
}

/// (1b) **sc-12078 bit-faithfulness gate:** the FUSED activation quantizer produces the same FP4 GEMM
/// output as (a) the CPU `Nvfp4Tensor::pack` reference and (b) the unfused on-device candle path.
///
/// ⚠ **This test cannot see an E4M3 or E2M1 rounding tie** — exact midpoints are measure-zero under
/// random activations, so it reported 0.000000 while both routes were emitting the wrong byte for every
/// tie. The byte-level gates are [`nvfp4_fused_e4m3_block_scale_bytes_match_cpu_at_exact_ties`],
/// [`nvfp4_unfused_e4m3_block_scale_bytes_match_cpu_at_exact_ties`] and
/// [`nvfp4_e2m1_element_nibbles_match_cpu_at_exact_ties`]; keep all of them. This gate covers the bulk
/// (non-tie) numerics that the crafted byte gates deliberately do not exercise.
#[test]
fn nvfp4_fused_activation_quant_matches_cpu_pack_ref() {
    let Some(dev) = nvfp4_device() else { return };
    let lt = CublasLt::new(&dev).unwrap();
    let (m, k, n) = (256usize, 256usize, 128usize); // K%32==0, N%16==0, M%16==0

    let x_f32 = pseudo_random(m * k, 101);
    let w_f32 = pseudo_random(n * k, 202);
    let x = Tensor::from_vec(x_f32, (m, k), &dev)
        .unwrap()
        .to_dtype(DType::BF16)
        .unwrap();
    let w = Tensor::from_vec(w_f32, (n, k), &dev)
        .unwrap()
        .to_dtype(DType::BF16)
        .unwrap();

    let w_pk = Nvfp4Tensor::pack(&w).unwrap();
    let w_stg = lt.stage_nvfp4(&w_pk).unwrap();

    // Three activation-quantize routes against the same resident weight.
    let x_fused = lt
        .quantize_nvfp4_activation_fused(&x, w_pk.cols_padded)
        .unwrap();
    let x_candle = lt.quantize_nvfp4_activation(&x, w_pk.cols_padded).unwrap();
    let x_cpu = lt.stage_nvfp4(&Nvfp4Tensor::pack(&x).unwrap()).unwrap();

    let y_fused = to_vec_f32(&lt.matmul_nvfp4_staged(&w_stg, &x_fused).unwrap());
    let y_candle = to_vec_f32(&lt.matmul_nvfp4_staged(&w_stg, &x_candle).unwrap());
    let y_cpu = to_vec_f32(&lt.matmul_nvfp4_staged(&w_stg, &x_cpu).unwrap());
    assert!(
        y_fused.iter().all(|v| v.is_finite()),
        "fused quant produced NaN/Inf"
    );

    let rr_cpu = rel_rms(&y_fused, &y_cpu);
    let rr_candle = rel_rms(&y_fused, &y_candle);
    eprintln!(
        "[sc-12078] fused vs CPU-pack GEMM rel-RMS = {rr_cpu:.6}; fused vs unfused on-device = {rr_candle:.6}"
    );
    assert!(
        rr_cpu < 0.02,
        "fused activation quant diverges from the CPU pack ref (rel-RMS {rr_cpu:.6})"
    );
    assert!(
        rr_candle < 0.001,
        "fused must match the unfused on-device recipe (rel-RMS {rr_candle:.6})"
    );
}

/// Byte offset of a logical `(row, 16-block)` UE4M3 scale within the **row-major 128×4 SF-atom** layout
/// cuBLASLt consumes — mirrors `cublaslt_scale_layout` and the quantizer's gather index.
fn rowmajor_scale_offset(r: usize, blk: usize, sf_cols: usize) -> usize {
    let (m_atom, mr) = (r / 128, r % 128);
    let num_k_atoms = sf_cols / 4;
    let (k_atom, kc) = (blk / 4, blk % 4);
    (k_atom + num_k_atoms * m_atom) * 512 + (mr % 32) * 16 + (mr / 32) * 4 + kc
}

/// Exact E4M3 grid midpoints. e.g. 8.5 sits between 8.0 (mantissa 0 → even) and 9.0 (mantissa 1 →
/// odd), so ties-to-even must yield 8.0 (0x50), NOT 9.0 (0x51). Covers the normal grid at several
/// exponents and one subnormal.
const E4M3_TIES: [f32; 6] = [
    8.5,             // e=3, between 8.0 (even) and 9.0 (odd)      -> 8.0
    9.5,             // e=3, between 9.0 (odd)  and 10.0 (even)    -> 10.0
    17.0,            // e=4, between 16.0 (even) and 18.0 (odd)    -> 16.0
    19.0,            // e=4, between 18.0 (odd)  and 20.0 (even)   -> 20.0
    0.097_656_25,    // e=-4 (12.5 ULPs), between 12 and 13 ULPs   -> 12 ULPs
    0.004_882_812_5, // subnormal, between 2 and 3 ULPs of 2^-9    -> 2 ULPs
];

/// The crafted E4M3-tie activation `[1, 16·(1+E4M3_TIES.len())]` + its `cols`.
///
/// A per-tensor amax of exactly **448** gives `global_scale = 448/(6·448) = 1/6`, hence `sf_real =
/// block_amax / (6·global_scale) = block_amax` exactly — so each block's amax *is* the value handed to
/// the E4M3 rounder and can be placed precisely on a midpoint. Block 0 carries the amax; block `i+1`
/// carries tie `i`.
fn e4m3_tie_activation(dev: &Device) -> (Tensor, usize) {
    let cols = 16 * (1 + E4M3_TIES.len());
    let mut data = vec![0f32; cols];
    data[0] = 448.0; // block 0 carries the per-tensor amax => global_scale = 1/6
    for (i, &t) in E4M3_TIES.iter().enumerate() {
        data[16 * (i + 1)] = t; // block i+1 has amax == t
    }
    (Tensor::from_vec(data, (1, cols), dev).unwrap(), cols)
}

/// Assert a quantize route's emitted UE4M3 block-scale bytes equal the canonical ties-to-even
/// [`e4m3_from_f32`] at every crafted midpoint. `route` labels the path in the failure output.
fn assert_e4m3_tie_scale_bytes(lt: &CublasLt, stg: &DevNvfp4, route: &str) {
    assert_eq!(
        stg.global_scale(),
        448.0f32 / (6.0f32 * 448.0f32),
        "the construction relies on global_scale == 1/6 so that sf_real == block_amax"
    );
    let scales = stg.scales_to_host(lt).unwrap();
    let sf_cols = (1 + E4M3_TIES.len()).div_ceil(4) * 4;
    let mut bad = Vec::new();
    for (i, &t) in E4M3_TIES.iter().enumerate() {
        let got = scales[rowmajor_scale_offset(0, i + 1, sf_cols)];
        let want = e4m3_from_f32(t); // the canonical encoder: RN, ties-to-even
        eprintln!(
            "[sc-12078] {route}: tie sf_real={t} -> 0x{got:02X} ({}) | CPU e4m3_from_f32 0x{want:02X} ({})",
            e4m3_to_f32(got),
            e4m3_to_f32(want)
        );
        if got != want {
            bad.push(format!(
                "sf_real={t}: {route} 0x{got:02X} ({}) != CPU 0x{want:02X} ({})",
                e4m3_to_f32(got),
                e4m3_to_f32(want)
            ));
        }
    }
    assert!(
        bad.is_empty(),
        "{route}: E4M3 block-scale bytes disagree with the canonical ties-to-even encoder at exact \
         midpoints — each wrong byte rescales an entire 16-element block: {bad:?}"
    );
}

/// (1c) **sc-12078 byte-exactness at E4M3 rounding TIES — the case a GEMM rel-RMS test cannot see.**
///
/// [`e4m3_from_f32`] is round-to-nearest **ties-to-even**: its scan tie-breaks on
/// `code.is_multiple_of(2)`, and an even E4M3 byte *is* an even mantissa LSB. A kernel that rounds with
/// `roundf` / candle's `.round()` instead rounds halves **away from zero**, so it emits the wrong byte for
/// every block whose scale lands exactly on a grid midpoint — changing the dequant scale of all 16 values
/// in that block.
///
/// Random activations never land on an exact midpoint (measure zero), so
/// [`nvfp4_fused_activation_quant_matches_cpu_pack_ref`] happily reports rel-RMS 0.000000 while this is
/// broken. **Byte-exactness has to be tested at the bytes, on crafted inputs.**
#[test]
fn nvfp4_fused_e4m3_block_scale_bytes_match_cpu_at_exact_ties() {
    let Some(dev) = nvfp4_device() else { return };
    let lt = CublasLt::new(&dev).unwrap();
    let (x, cols) = e4m3_tie_activation(&dev);
    let stg = lt.quantize_nvfp4_activation_fused(&x, cols).unwrap();
    assert_e4m3_tie_scale_bytes(&lt, &stg, "fused kernel");
}

/// (1d) The **unfused** twin of [`nvfp4_fused_e4m3_block_scale_bytes_match_cpu_at_exact_ties`]
/// (sc-12078 follow-up). The candle path had the identical defect: `e4m3_round_tensor` rounded
/// `v/ulp` with candle's `.round()` (half-away-from-zero), so it emitted 0x51 (9.0) for `sf_real` 8.5
/// where the canonical encoder emits 0x50 (8.0) — wrong at 4 of these 6 ties. candle has no RN-even
/// round, so the tie is now derived explicitly; this gate pins that it agrees with `e4m3_from_f32` at
/// the bytes. The unfused path is the fused kernel's nvrtc-unavailable fallback and must be *exactly*
/// as correct — a fallback that silently rescales blocks is worse than no fallback.
#[test]
fn nvfp4_unfused_e4m3_block_scale_bytes_match_cpu_at_exact_ties() {
    let Some(dev) = nvfp4_device() else { return };
    let lt = CublasLt::new(&dev).unwrap();
    let (x, cols) = e4m3_tie_activation(&dev);
    let stg = lt.quantize_nvfp4_activation(&x, cols).unwrap();
    assert_e4m3_tie_scale_bytes(&lt, &stg, "unfused candle path");
}

/// The exact E2M1 grid midpoints — the 7 gaps in `{0,.5,1,1.5,2,3,4,6}`. Ties-to-even sends each to
/// its **even**-index neighbour: 0.25→0.0, 0.75→1.0, 1.25→1.0, 1.75→2.0, 2.5→2.0, 3.5→4.0, 5.0→4.0.
const E2M1_TIES: [f32; 7] = [0.25, 0.75, 1.25, 1.75, 2.5, 3.5, 5.0];

/// `global_scale` this construction pins: `336/(6·448) = 1/8`, an exact power of two.
const E2M1_TIE_GLOBAL_SCALE: f32 = 0.125;

/// The crafted E2M1-tie activation `[1, 32]` + its `cols`.
///
/// A tie only fires if `ratio = value / elem_scale` lands **exactly** on a midpoint, so `elem_scale`
/// must be exactly a power of two — otherwise every crafted value drifts a few ULPs off the midpoint
/// and the test silently measures nothing (it reads as "ties round up" whatever the code does).
///
/// The recipe's factor of 6 makes that delicate: `sf_real = block_amax / (6·global_scale)` and
/// `elem_scale = block_scale · global_scale` cannot *both* be exact for an arbitrary amax. The way
/// through is to let `sf_real` be inexact but land solidly inside one E4M3 grid cell, and pick a
/// power-of-two `global_scale` so the *rounded* block scale times it is exact:
///
/// - per-tensor amax **336** ⟹ `global_scale = 336/(6·448) = 1/8` exactly (336·8 = 2688);
/// - block 1's amax **6.0** ⟹ `sf_real = 6/(6·⅛) = 8.0` — mid-cell, nowhere near the 8.5 tie, so the
///   E4M3 rounder returns the block scale **8.0 exactly** regardless of the last bit of its `log`/`exp`;
/// - hence `elem_scale = 8.0 · ⅛ = 1.0` exactly, and `ratio == value`.
///
/// Block 0 just carries the per-tensor amax. All ties are < 6.0, so block 1's amax stays 6.0.
fn e2m1_tie_activation(dev: &Device) -> (Tensor, usize) {
    let cols = 32; // 2 blocks
    let mut data = vec![0f32; cols];
    data[0] = 336.0; // per-tensor amax -> global_scale = 1/8 exactly
    data[16] = 6.0; // block 1 amax -> sf_real 8.0 -> block scale 8.0 -> elem_scale 1.0
    for (i, &t) in E2M1_TIES.iter().enumerate() {
        data[17 + i] = t; // +tie
        data[24 + i] = -t; // -tie (sign in bit 3; magnitude rounds the same)
    }
    (Tensor::from_vec(data, (1, cols), dev).unwrap(), cols)
}

/// Assert a quantize route's emitted E2M1 nibbles equal the canonical ties-to-even [`e2m1_from_f32`]
/// at every crafted midpoint, both signs.
fn assert_e2m1_tie_nibbles(lt: &CublasLt, stg: &DevNvfp4, route: &str) {
    // Both halves of `elem_scale = block_scale · global_scale == 1.0` are checked, not assumed. If
    // either drifts, `ratio != value`, no crafted value sits on a midpoint any more, and the tie
    // assertions below would quietly pass (or fail) for the wrong reason — which is exactly how the
    // first cut of this test fooled itself.
    assert_eq!(
        stg.global_scale(),
        E2M1_TIE_GLOBAL_SCALE,
        "{route}: the construction relies on global_scale == 1/8 so that elem_scale is exact"
    );
    let sf_cols = 4; // 2 blocks padded to the 4-block SF atom
    let block1_scale = stg.scales_to_host(lt).unwrap()[rowmajor_scale_offset(0, 1, sf_cols)];
    assert_eq!(
        (
            e4m3_to_f32(block1_scale) * E2M1_TIE_GLOBAL_SCALE,
            block1_scale
        ),
        (1.0, e4m3_from_f32(8.0)),
        "{route}: the construction relies on block 1's scale being exactly 8.0 (0x50) so that \
         elem_scale == 1.0 and ratio == value"
    );
    let packed = stg.packed_to_host(lt).unwrap();
    // Row 0, row-major `[1, cols/2]`: even column = low nibble, odd column = high nibble.
    let nibble = |c: usize| {
        if c.is_multiple_of(2) {
            packed[c / 2] & 0x0F
        } else {
            packed[c / 2] >> 4
        }
    };
    let mut bad = Vec::new();
    for (i, &t) in E2M1_TIES.iter().enumerate() {
        for (c, v) in [(17 + i, t), (24 + i, -t)] {
            let got = nibble(c);
            let want = e2m1_from_f32(v); // the canonical encoder: RN, ties-to-even
            eprintln!(
                "[sc-12078] {route}: E2M1 tie ratio={v} -> 0x{got:X} ({}) | CPU e2m1_from_f32 0x{want:X} ({})",
                E2M1_LUT[got as usize], E2M1_LUT[want as usize]
            );
            if got != want {
                bad.push(format!(
                    "ratio={v}: {route} 0x{got:X} ({}) != CPU 0x{want:X} ({})",
                    E2M1_LUT[got as usize], E2M1_LUT[want as usize]
                ));
            }
        }
    }
    assert!(
        bad.is_empty(),
        "{route}: E2M1 element nibbles disagree with the canonical ties-to-even encoder at exact \
         midpoints: {bad:?}"
    );
}

/// (1e) The E2M1 analogue of the E4M3 tie gate (sc-12078 follow-up), for **both** quantize routes.
///
/// `e2m1_code_tensor` counted `mag.ge(mid)` at all 7 midpoints, which rounds every tie **UP**, whereas
/// the canonical [`e2m1_from_f32`] is ties-to-even — wrong at 4 of the 7 (0.25, 1.25, 2.5, 5.0). Same
/// measure-zero invisibility as the E4M3 defect: a GEMM rel-RMS gate never sees it. The fused kernel's
/// `e2m1_code` already spells the thresholds out (`<=`/`<`) and is the reference; running both routes
/// through one assertion pins them to the same encoder and to each other.
#[test]
fn nvfp4_e2m1_element_nibbles_match_cpu_at_exact_ties() {
    let Some(dev) = nvfp4_device() else { return };
    let lt = CublasLt::new(&dev).unwrap();
    let (x, cols) = e2m1_tie_activation(&dev);
    assert_e2m1_tie_nibbles(
        &lt,
        &lt.quantize_nvfp4_activation(&x, cols).unwrap(),
        "unfused candle path",
    );
    assert_e2m1_tie_nibbles(
        &lt,
        &lt.quantize_nvfp4_activation_fused(&x, cols).unwrap(),
        "fused kernel",
    );
}

/// (2) The full `Nvfp4Linear` W4A4 forward runs the on-device quantize end-to-end, is finite across
/// repeated forwards (a denoise-step stand-in), and stays within NVFP4 tolerance of a bf16-dense
/// reference. Exercises the `forward_checked` NaN/inf guard.
#[test]
fn nvfp4_w4a4_forward_ondevice_no_nan_vs_bf16() {
    let Some(dev) = nvfp4_device() else { return };
    let (m, k, n) = (512usize, 512usize, 512usize);
    let x_f32 = pseudo_random(m * k, 7);
    let w_f32 = pseudo_random(n * k, 8);
    let x = Tensor::from_vec(x_f32.clone(), (m, k), &dev)
        .unwrap()
        .to_dtype(DType::BF16)
        .unwrap();
    let w = Tensor::from_vec(w_f32.clone(), (n, k), &dev)
        .unwrap()
        .to_dtype(DType::BF16)
        .unwrap();

    let lin = Nvfp4Linear::from_dense(&w, None, &dev, ActPrecision::W4A4).unwrap();
    assert_eq!(
        lin.regime(),
        Nvfp4Regime::Fp4W4A4,
        "W4A4 must light the FP4 cores on sm_120"
    );

    // Repeat to emulate steps: the guard must never trip, output must stay finite.
    let mut last = None;
    for step in 0..8 {
        let y = lin
            .forward_checked(&x)
            .unwrap_or_else(|e| panic!("W4A4 forward_checked tripped at step {step}: {e}"));
        assert_eq!(y.dims(), &[m, n]);
        last = Some(to_vec_f32(&y));
    }
    let got = last.unwrap();
    assert!(got.iter().all(|v| v.is_finite()));
    let bf16_ref = ref_matmul(&x_f32, &w_f32, m, k, n);
    let rr = rel_rms(&got, &bf16_ref);
    eprintln!("[sc-11044] on-device W4A4 forward vs bf16-dense rel-RMS = {rr:.5}");
    assert!(
        rr < 0.2,
        "on-device W4A4 vs bf16 {rr:.5} exceeds NVFP4 tolerance"
    );
}

/// (2b) **sc-12078 fallback policy: no fused quantizer ⇒ W4A16, never W4A4-via-unfused.**
///
/// The branch a healthy rig can never reach by itself. `SC12078_DISABLE_FUSED_QUANT` makes the fused
/// kernel report uncompilable — the same `Err` the compile closure produces for a broken nvrtc, cached
/// on the handle the same way — so the real capability gate in `Nvfp4Linear::from_packed` executes
/// here rather than shipping unexecuted.
///
/// The policy under test: an FP4-eligible sm_120 W4A4 request whose fused quantizer is unavailable
/// resolves to [`Nvfp4Regime::DequantBf16`] — it does **not** error, and it does **not** stay
/// `Fp4W4A4` and route the forward through the unfused reference chain. That last option is what this
/// gate exists to forbid: it is numerically identical (so no rel-RMS gate can see it) but costs ~19 ms
/// per projection against ~0.38 ms fused, which measured **0.01× vs dense bf16** end-to-end — ~100×
/// worse than the W4A16 this now selects (~1.00×).
///
/// Also asserts the reported accounting stays honest through the fallback: a layer that fell back must
/// not claim the packed NVFP4 footprint it no longer has.
///
/// Env-var mutation is safe here: `.cargo/config.toml` forces `RUST_TEST_THREADS=1`, so no concurrent
/// test can observe the window, and the var is removed before any assertion can unwind past it.
#[test]
fn nvfp4_fused_unavailable_forces_w4a16() {
    let Some(dev) = nvfp4_device() else { return };
    let (n, k) = (256usize, 512usize);
    let w_f32 = pseudo_random(n * k, 0x1_2078);
    let w = Tensor::from_vec(w_f32.clone(), (n, k), &dev)
        .unwrap()
        .to_dtype(DType::BF16)
        .unwrap();

    // Baseline: with the fused kernel available, this exact weight lights the FP4 cores. Without this
    // the test could pass for the boring reason that the shape/device was never eligible.
    let lit = Nvfp4Linear::from_dense(&w, None, &dev, ActPrecision::W4A4).unwrap();
    assert_eq!(
        lit.regime(),
        Nvfp4Regime::Fp4W4A4,
        "precondition: this shape must be FP4-eligible, or the gate below proves nothing"
    );
    assert!(lit.lights_up_fp4());

    // Now build the same layer with the fused quantizer forced unavailable. Each `Nvfp4Linear` builds
    // its own `CublasLt`, so this one re-probes and sees the failure.
    std::env::set_var(CublasLt::NVFP4_FORCE_NO_FUSED_QUANT_ENV, "1");
    let gated = Nvfp4Linear::from_dense(&w, None, &dev, ActPrecision::W4A4);
    std::env::remove_var(CublasLt::NVFP4_FORCE_NO_FUSED_QUANT_ENV);
    let gated = gated.expect("no fused quantizer must DEGRADE to W4A16, not fail the build");

    assert_eq!(
        gated.regime(),
        Nvfp4Regime::DequantBf16,
        "with no fused quantizer, W4A4 must fall back to W4A16 — staying Fp4W4A4 means the forward \
         is serving the unfused chain at 0.01× vs bf16 (~100× worse than this fallback)"
    );
    assert!(
        !gated.lights_up_fp4(),
        "a fallback layer must not report the FP4 cores as lit"
    );

    // The probe is a capability gate, not a numerics change: the fallback still computes.
    let m = 64usize;
    let x_f32 = pseudo_random(m * k, 0x2_2078);
    let x = Tensor::from_vec(x_f32.clone(), (m, k), &dev)
        .unwrap()
        .to_dtype(DType::BF16)
        .unwrap();
    let y = gated
        .forward_checked(&x)
        .expect("W4A16 fallback must forward");
    assert_eq!(y.dims(), &[m, n]);
    let got = to_vec_f32(&y);
    assert!(got.iter().all(|v| v.is_finite()));
    let rr = rel_rms(&got, &ref_matmul(&x_f32, &w_f32, m, k, n));
    eprintln!("[sc-12078] no-fused-quantizer W4A16 fallback vs bf16-dense rel-RMS = {rr:.5}");
    assert!(
        rr < 0.2,
        "W4A16 fallback vs bf16 {rr:.5} exceeds NVFP4 tolerance"
    );

    // Honest accounting through the fallback (the sc-11045 MAJOR-3 shape): this layer holds a dense
    // bf16 weight now, and must say so rather than reporting its packed host container's size.
    assert_eq!(
        gated.resident_weight_bytes(),
        gated.bf16_footprint_bytes(),
        "a W4A16 fallback holds dense bf16 resident — it must not report the packed NVFP4 footprint"
    );
    assert_eq!(
        gated.resident_device_bytes(),
        None,
        "nothing is staged packed on-device in W4A16"
    );
}

/// (3) Throughput: on-device **W4A4** vs **W4A16** (and a bf16-dense baseline) layer forwards on
/// representative Sana-DiT GEMM shapes, on the (now exclusive) GPU. Informational — the multiple is
/// hardware-dependent, so it is reported, not asserted (the correctness/no-NaN gates above are the
/// hard checks).
#[test]
fn nvfp4_w4a4_vs_w4a16_throughput() {
    let Some(dev) = nvfp4_device() else { return };
    // (M tokens, K in, N out): Sana-1.6B-ish attn proj (2240²) and FF up-proj (2240→5600).
    let shapes = [(1024usize, 2240usize, 2240usize), (1024, 2240, 5600)];
    let iters = 40usize;

    for (m, k, n) in shapes {
        let x = Tensor::from_vec(pseudo_random(m * k, 1), (m, k), &dev)
            .unwrap()
            .to_dtype(DType::BF16)
            .unwrap();
        let w = Tensor::from_vec(pseudo_random(n * k, 2), (n, k), &dev)
            .unwrap()
            .to_dtype(DType::BF16)
            .unwrap();

        let w4a4 = Nvfp4Linear::from_dense(&w, None, &dev, ActPrecision::W4A4).unwrap();
        let w4a16 = Nvfp4Linear::from_dense(&w, None, &dev, ActPrecision::W4A16).unwrap();
        assert_eq!(w4a4.regime(), Nvfp4Regime::Fp4W4A4);
        assert_eq!(w4a16.regime(), Nvfp4Regime::DequantBf16);

        let time = |lin: &Nvfp4Linear| -> f64 {
            // warmup
            for _ in 0..5 {
                let _ = lin.forward(&x).unwrap();
            }
            dev.synchronize().unwrap();
            let t0 = Instant::now();
            for _ in 0..iters {
                let y = lin.forward(&x).unwrap();
                std::hint::black_box(&y);
            }
            dev.synchronize().unwrap();
            t0.elapsed().as_secs_f64() / iters as f64
        };
        // bf16-dense baseline (candle matmul).
        let time_bf16 = || -> f64 {
            let wt = w.t().unwrap().contiguous().unwrap();
            for _ in 0..5 {
                let _ = x.matmul(&wt).unwrap();
            }
            dev.synchronize().unwrap();
            let t0 = Instant::now();
            for _ in 0..iters {
                let y = x.matmul(&wt).unwrap();
                std::hint::black_box(&y);
            }
            dev.synchronize().unwrap();
            t0.elapsed().as_secs_f64() / iters as f64
        };

        let t_w4a4 = time(&w4a4);
        let t_w4a16 = time(&w4a16);
        let t_bf16 = time_bf16();
        eprintln!(
            "[sc-11044] LAYER shape M={m} K={k} N={n}: W4A4 {:.3} ms/fwd (incl. on-device act quant), \
             W4A16 {:.3} ms/fwd, bf16-dense {:.3} ms/fwd | W4A4 vs W4A16 = {:.2}×, vs bf16 = {:.2}× \
             (exclusive GPU)",
            t_w4a4 * 1e3,
            t_w4a16 * 1e3,
            t_bf16 * 1e3,
            t_w4a16 / t_w4a4,
            t_bf16 / t_w4a4,
        );

        // GEMM-CORE isolation: pre-stage both operands once, then time ONLY the FP4 GEMM vs the bf16
        // GEMM — the real FP4 tensor-core win, separated from the (unfused, candle-op) activation
        // quantize tax that the layer number above includes.
        let lt = CublasLt::new(&dev).unwrap();
        let w_pk = Nvfp4Tensor::pack(&w).unwrap();
        let w_stg = lt.stage_nvfp4(&w_pk).unwrap();
        let x_stg = lt.quantize_nvfp4_activation(&x, w_pk.cols_padded).unwrap();
        let wt = w.t().unwrap().contiguous().unwrap();
        let time_fp4_gemm = || -> f64 {
            for _ in 0..5 {
                let _ = lt.matmul_nvfp4_staged(&w_stg, &x_stg).unwrap();
            }
            dev.synchronize().unwrap();
            let t0 = Instant::now();
            for _ in 0..iters {
                let y = lt.matmul_nvfp4_staged(&w_stg, &x_stg).unwrap();
                std::hint::black_box(&y);
            }
            dev.synchronize().unwrap();
            t0.elapsed().as_secs_f64() / iters as f64
        };
        let time_bf16_gemm = || -> f64 {
            for _ in 0..5 {
                let _ = x.matmul(&wt).unwrap();
            }
            dev.synchronize().unwrap();
            let t0 = Instant::now();
            for _ in 0..iters {
                let y = x.matmul(&wt).unwrap();
                std::hint::black_box(&y);
            }
            dev.synchronize().unwrap();
            t0.elapsed().as_secs_f64() / iters as f64
        };
        let t_fp4g = time_fp4_gemm();
        let t_bf16g = time_bf16_gemm();
        eprintln!(
            "[sc-11044] GEMM-CORE shape M={m} K={k} N={n}: FP4 {:.3} ms, bf16 {:.3} ms | FP4 tensor-core \
             speedup = {:.2}× (pre-staged operands, exclusive GPU)",
            t_fp4g * 1e3,
            t_bf16g * 1e3,
            t_bf16g / t_fp4g,
        );
        assert!(t_w4a4.is_finite() && t_w4a4 > 0.0 && t_fp4g > 0.0);
    }
}

/// (3b) **sc-12207 re-measurement — per-projection activation-quantizer cost on the real Krea 2 Turbo
/// GEMM shapes.** The sc-12110 review decomposed `quantize_nvfp4_activation` here and found **76% of it
/// was a `scatter_add` atomic bijection** (250.04 ms of 328.10 ms at K=6144; 666.29 ms of 879.42 ms at
/// K=16384). sc-12207 turned that swizzle into an `index_select` gather over a cached inverse
/// permutation; this reports the **corrected** per-projection quantizer cost — the residual sc-12078's
/// fused-kernel target must be re-derived from. It also times the FP4 GEMM at the same shapes so the
/// implied W4A4 per-projection (quantize + GEMM) and Krea step time can be reconstructed **without
/// loading the 12.5 B trunk**.
///
/// Krea 2 Turbo at 1024²: M ≈ 4118 tokens (image seq + text context); K = 6144 (hidden) and 16384
/// (SwiGLU intermediate); the DiT is ~260 quantized projections/step. Throughput → run EXCLUSIVE GPU.
#[test]
#[ignore = "GPU perf measurement (sc-12207): needs an sm_120 device; run on an EXCLUSIVE GPU"]
fn nvfp4_quantize_activation_perf_krea_shapes() {
    let Some(dev) = nvfp4_device() else { return };
    let lt = CublasLt::new(&dev).unwrap();
    let m = 4118usize;
    let iters = 30usize;

    eprintln!(
        "\n[sc-12207] ===== per-projection activation-quantizer cost, Krea shapes, exclusive GPU ====="
    );
    for k in [6144usize, 16384usize] {
        // cols_padded == K: both 6144 and 16384 are multiples of NVFP4_BLOCK and NVFP4_K_ALIGN.
        let x = Tensor::from_vec(pseudo_random(m * k, 0x5C1_2207 ^ k as u64), (m, k), &dev)
            .unwrap()
            .to_dtype(DType::BF16)
            .unwrap();

        // The unfused quantizer (the sc-12207 target). First call also builds+caches the gather index,
        // so warm up before timing.
        for _ in 0..5 {
            std::hint::black_box(lt.quantize_nvfp4_activation(&x, k).unwrap());
        }
        dev.synchronize().unwrap();
        let t0 = Instant::now();
        for _ in 0..iters {
            std::hint::black_box(lt.quantize_nvfp4_activation(&x, k).unwrap());
        }
        dev.synchronize().unwrap();
        let ms_quant = t0.elapsed().as_secs_f64() * 1000.0 / iters as f64;

        // The FUSED quantizer (sc-12078). First call also nvrtc-compiles the module, so warm up first.
        for _ in 0..5 {
            std::hint::black_box(lt.quantize_nvfp4_activation_fused(&x, k).unwrap());
        }
        dev.synchronize().unwrap();
        let tf = Instant::now();
        for _ in 0..iters {
            std::hint::black_box(lt.quantize_nvfp4_activation_fused(&x, k).unwrap());
        }
        dev.synchronize().unwrap();
        let ms_fused = tf.elapsed().as_secs_f64() * 1000.0 / iters as f64;

        // FP4 GEMM at a representative N (pre-staged operands) — N does not affect the quantizer, this is
        // just the compute leg to reconstruct a full W4A4 projection cost.
        let n = k;
        let w = Tensor::from_vec(pseudo_random(n * k, 2), (n, k), &dev)
            .unwrap()
            .to_dtype(DType::BF16)
            .unwrap();
        let w_pk = Nvfp4Tensor::pack(&w).unwrap();
        let w_stg = lt.stage_nvfp4(&w_pk).unwrap();
        let x_stg = lt.quantize_nvfp4_activation(&x, w_pk.cols_padded).unwrap();
        for _ in 0..5 {
            std::hint::black_box(lt.matmul_nvfp4_staged(&w_stg, &x_stg).unwrap());
        }
        dev.synchronize().unwrap();
        let t1 = Instant::now();
        for _ in 0..iters {
            std::hint::black_box(lt.matmul_nvfp4_staged(&w_stg, &x_stg).unwrap());
        }
        dev.synchronize().unwrap();
        let ms_gemm = t1.elapsed().as_secs_f64() * 1000.0 / iters as f64;

        eprintln!(
            "[sc-12207] M={m} K={k:>5}: unfused {ms_quant:>8.3} ms | FUSED {ms_fused:>7.3} ms \
             ({:>5.1}× vs unfused) | FP4 GEMM (N={n}) {ms_gemm:>7.3} ms | W4A4 proj: unfused {:>8.3} \
             / fused {:>6.3} ms",
            ms_quant / ms_fused,
            ms_quant + ms_gemm,
            ms_fused + ms_gemm
        );
        assert!(ms_quant.is_finite() && ms_quant > 0.0 && ms_gemm > 0.0 && ms_fused > 0.0);
    }
    eprintln!(
        "[sc-12207/12078] pre-fix ref (sc-12110): 328.10 ms (K=6144) / 879.42 ms (K=16384). sc-12207 \
         (scatter→gather) → the 'unfused' column; sc-12078 (fused kernel) → the 'FUSED' column."
    );
}

/// (4) Outlier-sparsity capture (the spike residual gate) on synthetic activations: benign / sparse /
/// dense classify as expected on-device, confirming the metric that governs the benign→W4A4 /
/// dense→W4A16 partition.
#[test]
fn nvfp4_w4a4_outlier_sparsity_capture_confirms_partition() {
    let Some(dev) = nvfp4_device() else { return };
    let (m, k) = (256usize, 512usize);

    // Benign self-attn/FF-style activation.
    let benign = Tensor::from_vec(
        pseudo_random(m * k, 11)
            .iter()
            .map(|v| v * 0.3)
            .collect::<Vec<_>>(),
        (m, k),
        &dev,
    )
    .unwrap()
    .to_dtype(DType::BF16)
    .unwrap();
    let s_benign = OutlierSparsity::from_tensor(&benign, OutlierSparsity::DEFAULT_TAU).unwrap();
    eprintln!(
        "[sc-11044] benign layer: benign_fraction={:.4} class={:?}",
        s_benign.benign_fraction,
        s_benign.class()
    );
    assert!(
        s_benign.w4a4_viable(),
        "benign activation must be W4A4-viable"
    );

    // Dense-outlier (caption/cross-attn-style) activation: an outlier in most blocks.
    let mut dense = pseudo_random(m * k, 22)
        .iter()
        .map(|v| v * 0.3)
        .collect::<Vec<_>>();
    for r in 0..m {
        for b in 0..(k / OutlierSparsity::BLOCK) {
            dense[r * k + b * OutlierSparsity::BLOCK + 1] = 100.0;
        }
    }
    let dense_t = Tensor::from_vec(dense, (m, k), &dev)
        .unwrap()
        .to_dtype(DType::BF16)
        .unwrap();
    let s_dense = OutlierSparsity::from_tensor(&dense_t, OutlierSparsity::DEFAULT_TAU).unwrap();
    eprintln!(
        "[sc-11044] dense-outlier layer: benign_fraction={:.4} class={:?} crush={:.0}",
        s_dense.benign_fraction,
        s_dense.class(),
        s_dense.max_crush_ratio
    );
    assert_eq!(
        s_dense.class(),
        OutlierClass::Dense,
        "dense outliers must flag collapse (W4A16)"
    );
    assert!(
        !s_dense.w4a4_viable(),
        "dense-outlier layer must NOT be W4A4-viable — partition holds"
    );
}

/// (5) Real Sana-1.6B DiT projection weight, W4A4 vs bf16 (env-gated, `#[ignore]` per repo convention
/// for real-weight tests). Set `SC11044_SANA_DIT_SAFETENSORS` to a Sana transformer shard; the test
/// loads the first eligible 2-D linear weight (K%32==0, N%16==0), quantizes W4A4, and reports the
/// quality delta vs the bf16 weight on a synthetic activation. Activations are synthetic — the live
/// per-step denoise activation capture is deferred to sc-11045.
#[test]
#[ignore = "real-weight test: set SC11044_SANA_DIT_SAFETENSORS to a Sana DiT shard"]
fn nvfp4_w4a4_real_sana_dit_weight() {
    let Some(dev) = nvfp4_device() else { return };
    let path = match std::env::var("SC11044_SANA_DIT_SAFETENSORS") {
        Ok(p) => p,
        Err(_) => {
            eprintln!("[sc-11044] SC11044_SANA_DIT_SAFETENSORS unset; skipping real-weight test");
            return;
        }
    };
    let tensors = candle_gen::candle_core::safetensors::load(&path, &Device::Cpu)
        .expect("load Sana DiT safetensors shard");
    // Find the first eligible 2-D weight: K a multiple of 32, N a multiple of 16, reasonably large.
    let mut chosen: Option<(String, Tensor)> = None;
    for (name, t) in tensors.iter() {
        if t.rank() == 2 {
            let (n, k) = t.dims2().unwrap();
            if k.is_multiple_of(32) && n.is_multiple_of(16) && k >= 256 && n >= 256 {
                chosen = Some((name.clone(), t.clone()));
                break;
            }
        }
    }
    let (name, w_cpu) = chosen.expect("no eligible 2-D linear weight in the shard");
    let (n, k) = w_cpu.dims2().unwrap();
    eprintln!("[sc-11044] real Sana DiT weight '{name}' shape [N={n}, K={k}]");
    let w = w_cpu
        .to_dtype(DType::BF16)
        .unwrap()
        .to_device(&dev)
        .unwrap();

    let m = 1024usize;
    let x_f32 = pseudo_random(m * k, 42);
    let x = Tensor::from_vec(x_f32.clone(), (m, k), &dev)
        .unwrap()
        .to_dtype(DType::BF16)
        .unwrap();

    let lin = Nvfp4Linear::from_dense(&w, None, &dev, ActPrecision::W4A4).unwrap();
    assert_eq!(lin.regime(), Nvfp4Regime::Fp4W4A4);
    let got = to_vec_f32(&lin.forward_checked(&x).unwrap());
    assert!(
        got.iter().all(|v| v.is_finite()),
        "real-weight W4A4 produced NaN/Inf"
    );

    let w_ref = to_vec_f32(&w);
    let bf16_ref = ref_matmul(&x_f32, &w_ref, m, k, n);
    let rr = rel_rms(&got, &bf16_ref);
    // Weight-outlier sparsity of the real weight, for the record.
    let ws = OutlierSparsity::from_tensor(&w, OutlierSparsity::DEFAULT_TAU).unwrap();
    eprintln!(
        "[sc-11044] real Sana DiT '{name}': W4A4 vs bf16 rel-RMS = {rr:.5}; weight benign_fraction \
         {:.4} ({:?})",
        ws.benign_fraction,
        ws.class()
    );
    assert!(
        rr < 0.25,
        "real-weight W4A4 vs bf16 {rr:.5} unexpectedly large"
    );
}
