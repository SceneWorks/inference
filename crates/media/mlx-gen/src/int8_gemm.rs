//! Tiled, bit-exact INT8 GEMM shared by the ConvRot INT8 and packed W4A8 projections (sc-26691).
//!
//! # Why this exists
//!
//! The first native kernels (sc-26683 / sc-26685) ran one 32-lane SIMD group per **output
//! element**: a scalar `int` dot product over `K` with no tiling and no threadgroup reuse, so every
//! activation row re-read the whole weight matrix. At MiniMax-H3 geometry (hidden 5376, FFN 14336,
//! ~52.7k packed rows, 50 blocks) that is ~1e15 bytes of device traffic per denoise step, measured at
//! 46-49 minutes per step on the Mac (SceneWorks campaign run 40797515762) against ~55 s on CUDA.
//!
//! # The exactness contract this kernel keeps
//!
//! The published arithmetic (Kitchen `int8_linear.cu`, mirrored by the candle cuBLASLt path and the
//! MLX oracle tests `convrot_int8::native_projections_match_pinned_kitchen` /
//! `w4a8_compare::native_forward_matches_the_external_reference`) is: **exact int32 accumulation**
//! of `int8 x int8` products, then the FP32 epilogue `((float(acc) * sx) * sw) + bias`, rounded once
//! to the activation dtype. Those oracle tests compare BF16/F16 outputs with a tolerance of **zero**,
//! so any route that changes `acc` by a single unit is a contract break, not a precision trade-off.
//!
//! A plain FP32 GEMM over the whole `K` is **not** exact: `|acc|` can reach
//! `K * 128 * 128 = 14336 * 2^14 ~= 2.3e8 > 2^24`, beyond which FP32 can no longer hold every
//! integer and the summation order starts to round. (That rules out feeding the codes as floats to
//! `quantized_matmul` / `matmul` over the full depth, which the diagnosis proposed and this story
//! verified against the bound.)
//!
//! This kernel uses the FP32 `simdgroup_matrix` units for throughput but **flushes** the FP32
//! accumulators into int32 registers every [`FLUSH_K`] (512) terms. Inside one window every partial
//! sum — in *any* association order the hardware chooses — is an integer of magnitude at most
//! `512 * 2^14 = 2^23 < 2^24`, and every product `|x * w| <= 2^14` is exact, so each window's FP32
//! result is exactly its integer sum. Window sums are then added in int32 (no rounding), which
//! reproduces the reference `acc` bit for bit; the epilogue is the reference expression verbatim.
//! Codes are integers in `[-128, 127]`, so staging them as `float` in threadgroup memory is exact.
//!
//! # Shape
//!
//! One threadgroup computes a `BM x BN` (64 x 64) output tile with four SIMD groups (2 x 2, each a
//! 32 x 32 sub-tile of sixteen 8x8 fragments), streaming `BK` (32) deep slices of the activation
//! codes and the weight through threadgroup memory. The W4A8 variant decodes its packed int4 codes
//! to the int8 weight **inside** the tile load (the `w4a8_gemm.cu` decode arithmetic, unchanged), so
//! no decoded weight scratch of any size exists. Residency is unchanged: the I8 (or packed) codes are
//! the only weight bytes, and no persistent float copy of any weight is created.
//!
//! The per-row activation quantizers are one threadgroup per row with a cooperative absmax (an
//! order-independent `max`, so bit-identical to the serial loop) and coalesced code writes.

use crate::Result;
use mlx_rs::{fast::MetalKernel, Array, Dtype};
use std::cell::RefCell;

/// Output rows per threadgroup tile.
pub const BM: usize = 64;
/// Output columns per threadgroup tile.
pub const BN: usize = 64;
/// Reduction depth staged through threadgroup memory per step.
pub const BK: usize = 32;
/// Threads per GEMM threadgroup: four SIMD groups of 32.
pub const GEMM_THREADS: usize = 128;
/// Reduction terms accumulated in FP32 before an exact flush into int32. See the module docs.
pub const FLUSH_K: usize = 512;
/// Largest magnitude of one `int8 x int8` product: `(-128) * (-128)`.
pub const MAX_PRODUCT: u64 = 128 * 128;
/// Threads per activation-quantizer threadgroup (one row each).
pub const QUANT_THREADS: usize = 256;
/// Deepest reduction whose int32 accumulator cannot overflow (`K * 2^14 < 2^31`).
pub const MAX_K: usize = (i32::MAX as u64 / MAX_PRODUCT) as usize;

// The exactness argument, checked at compile time: a whole flush window of worst-case products must
// stay strictly inside FP32's contiguous-integer range, and the window must be whole BK steps.
const _: () = assert!(FLUSH_K as u64 * MAX_PRODUCT < (1u64 << 24));
const _: () = assert!(FLUSH_K.is_multiple_of(BK));
const _: () = assert!(GEMM_THREADS * 16 == BM * BK && GEMM_THREADS * 16 == BN * BK);

/// Validate a projection's geometry for the tiled kernel: `K` a whole number of `BK` steps (both
/// codecs guarantee this — the ConvRot group and the W4A8 rotation group are multiples of 64), an
/// int32 accumulator that cannot overflow, and `u32` element indices for activations, weights and
/// outputs.
pub fn check_geometry(m: usize, n: usize, k: usize) -> std::result::Result<(), String> {
    if m == 0 || n == 0 || k == 0 {
        return Err("INT8 GEMM: empty operand".into());
    }
    if !k.is_multiple_of(BK) {
        return Err(format!("INT8 GEMM: K={k} is not a multiple of {BK}"));
    }
    if k > MAX_K {
        return Err(format!(
            "INT8 GEMM: K={k} could overflow the int32 accumulator"
        ));
    }
    let fits = |a: usize, b: usize| a.checked_mul(b).is_some_and(|v| v <= u32::MAX as usize);
    if !fits(m, k) || !fits(n, k) || !fits(m, n) || m > i32::MAX as usize {
        return Err("INT8 GEMM: operands exceed native index geometry".into());
    }
    Ok(())
}

/// `(grid, threadgroup)` of the tiled GEMM in MLX's thread-count convention: one threadgroup per
/// `BM x BN` output tile, tile columns along `x`, tile rows along `y`.
pub fn gemm_dispatch(m: usize, n: usize) -> ((i32, i32, i32), (i32, i32, i32)) {
    let tiles_n = n.div_ceil(BN);
    let tiles_m = m.div_ceil(BM);
    (
        ((tiles_n * GEMM_THREADS) as i32, tiles_m as i32, 1),
        (GEMM_THREADS as i32, 1, 1),
    )
}

/// Exact FP32 flush windows a `K`-deep reduction is split into.
pub fn flush_windows(k: usize) -> usize {
    k.div_ceil(FLUSH_K)
}

const HEADER: &str = r#"#include <metal_stdlib>
using namespace metal;

// Row and first column of this lane's two elements of an 8x8 simdgroup matrix (the Apple GPU
// fragment layout MLX's steel kernels use; the second element is the next column).
inline ushort2 sc_i8_frag(uint lane) {
    const uint qid = lane / 4u;
    return ushort2((qid & 4u) + ((lane / 2u) % 4u), (qid & 2u) * 2u + (lane % 2u) * 2u);
}

// OCP E4M3FN byte -> f32, bit-identical to `gen_core::comfy_quant::fp8_e4m3fn_to_f32` and to the
// former `sc_w4a8_int8_dequant_v1` kernel (same expression).
inline float sc_w4a8_group_scale(uint scale_code) {
    uint exponent = (scale_code >> 3) & 0x0Fu;
    uint mantissa = scale_code & 0x07u;
    float scale;
    if (exponent == 0x0Fu && mantissa == 0x07u) {
        // NaN code: `validate_s_rel` refuses it at load, and the host oracle's `NaN as i8` is 0.
        scale = 0.0f;
    } else {
        float magnitude = exponent == 0u ? float(mantissa) * 0.125f * 0.015625f
                                         : (1.0f + float(mantissa) * 0.125f)
                                               * exp2(float(exponent) - 7.0f);
        scale = (scale_code & 0x80u) == 0u ? magnitude : -magnitude;
    }
    return scale;
}
"#;

/// Row-wise symmetric quantizer, one threadgroup per row. `{NUMERATOR}` is the codec's numerator
/// expression: the S4 ConvRot input is already dtype-rounded (`float(x)`), while the W4A8 input is
/// the F32 butterfly and Kitchen casts its numerator to the activation dtype (`float(T(float(x)))`).
/// Everything else is the former serial kernel's expression, verbatim.
const QUANT_TEMPLATE: &str = r#"
threadgroup float partial[QUANT_SIMDGROUPS];
const uint row = threadgroup_position_in_grid.y;
// Uniform per threadgroup (one row each), so no thread skips the barrier below alone.
if (row >= uint(M)) return;
const uint tid = thread_index_in_threadgroup;
const uint base = row * uint(K);
float peak = 0.0f;
for (uint j = tid; j < uint(K); j += QUANT_THREADS) {
    float v = abs(float(x[base + j]));
    if (!isnan(v)) peak = max(peak, v);
}
// max is exact and order-independent: the same peak as the serial left-to-right loop.
peak = simd_max(peak);
if (thread_index_in_simdgroup == 0u) partial[simdgroup_index_in_threadgroup] = peak;
threadgroup_barrier(mem_flags::mem_threadgroup);
peak = partial[0];
for (uint i = 1u; i < QUANT_SIMDGROUPS; ++i) peak = max(peak, partial[i]);
peak = min(peak, DTYPE==1 ? 3.38953139e38f : DTYPE==2 ? 65504.0f : 3.402823466e38f);
// Kitchen rounds scale and quotient to the activation dtype before nearbyintf.
float scale = max(peak * (1.0f / 127.0f), 1.0e-30f);
if (tid == 0u) s[row] = scale;
float divisor = float(T(scale));
for (uint j = tid; j < uint(K); j += QUANT_THREADS) {
    float z = float(T({NUMERATOR} / divisor));
    q[base + j] = char(isnan(z) ? -128.0f : clamp(rint(z), -128.0f, 127.0f));
}
"#;

/// The tiled GEMM. `{STAGE_B}` stages this thread's 16 weight codes of row `bcol`, columns
/// `k0..k0+16`, into `Bs[lr * LDS + lk ..]` as exact floats (zero past `N`).
const GEMM_TEMPLATE: &str = r#"
constexpr uint BM = 64u, BN = 64u, BK = 32u, LDS = BK + 4u;
constexpr uint FLUSH_STEPS = FLUSH_K / BK;
threadgroup float As[BM * LDS];
threadgroup float Bs[BN * LDS];
const uint tid = thread_index_in_threadgroup;
const uint lane = thread_index_in_simdgroup;
const uint sg = simdgroup_index_in_threadgroup;
const uint n0 = threadgroup_position_in_grid.x * BN;
const uint m0 = threadgroup_position_in_grid.y * BM;
const uint sm = (sg / 2u) * 32u;
const uint sn = (sg % 2u) * 32u;
// Each thread stages 16 consecutive K codes of one tile row of A and one tile row of B.
const uint lr = tid / 2u;
const uint lk = (tid % 2u) * 16u;
const uint arow = m0 + lr;
const uint bcol = n0 + lr;
const ushort2 frag = sc_i8_frag(lane);

simdgroup_float8x8 c[4][4];
int2 acc[4][4];
#pragma clang loop unroll(full)
for (int i = 0; i < 4; ++i) {
#pragma clang loop unroll(full)
    for (int j = 0; j < 4; ++j) {
        c[i][j] = make_filled_simdgroup_matrix<float, 8, 8>(0.0f);
        acc[i][j] = int2(0);
    }
}

const uint steps = uint(K) / BK;
for (uint kstep = 0u; kstep < steps; ++kstep) {
    const uint k0 = kstep * BK + lk;
    if (arow < uint(M)) {
        const uint src = arow * uint(K) + k0;
#pragma clang loop unroll(full)
        for (uint e = 0u; e < 16u; ++e) As[lr * LDS + lk + e] = float(q[src + e]);
    } else {
#pragma clang loop unroll(full)
        for (uint e = 0u; e < 16u; ++e) As[lr * LDS + lk + e] = 0.0f;
    }
    if (bcol < uint(N)) {
{STAGE_B}
    } else {
#pragma clang loop unroll(full)
        for (uint e = 0u; e < 16u; ++e) Bs[lr * LDS + lk + e] = 0.0f;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
#pragma clang loop unroll(full)
    for (uint kk = 0u; kk < BK; kk += 8u) {
        simdgroup_float8x8 a[4];
        simdgroup_float8x8 bm[4];
#pragma clang loop unroll(full)
        for (int i = 0; i < 4; ++i) simdgroup_load(a[i], As + (sm + uint(i) * 8u) * LDS + kk, LDS);
#pragma clang loop unroll(full)
        for (int j = 0; j < 4; ++j)
            simdgroup_load(bm[j], Bs + (sn + uint(j) * 8u) * LDS + kk, LDS, ulong2(0, 0), true);
#pragma clang loop unroll(full)
        for (int i = 0; i < 4; ++i) {
#pragma clang loop unroll(full)
            for (int j = 0; j < 4; ++j) simdgroup_multiply_accumulate(c[i][j], a[i], bm[j], c[i][j]);
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    // Exact flush: every FP32 partial in the window is an integer of magnitude <= 2^23.
    if ((kstep + 1u) % FLUSH_STEPS == 0u || kstep + 1u == steps) {
#pragma clang loop unroll(full)
        for (int i = 0; i < 4; ++i) {
#pragma clang loop unroll(full)
            for (int j = 0; j < 4; ++j) {
                acc[i][j] += int2(int(c[i][j].thread_elements()[0]), int(c[i][j].thread_elements()[1]));
                c[i][j] = make_filled_simdgroup_matrix<float, 8, 8>(0.0f);
            }
        }
    }
}

// The reference epilogue, verbatim: ((acc * sx) * sw) + bias, one rounding to T at the store.
#pragma clang loop unroll(full)
for (int i = 0; i < 4; ++i) {
    const uint row = m0 + sm + uint(i) * 8u + uint(frag.x);
    if (row >= uint(M)) continue;
#pragma clang loop unroll(full)
    for (int j = 0; j < 4; ++j) {
#pragma clang loop unroll(full)
        for (int e = 0; e < 2; ++e) {
            const uint out = n0 + sn + uint(j) * 8u + uint(frag.y) + uint(e);
            if (out < uint(N)) {
                float value = (float(acc[i][j][e]) * sx[row]) * sw[out];
                y[row * uint(N) + out] = T(value + b[out]);
            }
        }
    }
}
"#;

/// ConvRot: an I8 `[N, K]` weight, read as stored.
const STAGE_B_CONVROT: &str = r#"
        const uint wsrc = bcol * uint(K) + k0;
#pragma clang loop unroll(full)
        for (uint e = 0u; e < 16u; ++e) Bs[lr * LDS + lk + e] = float(w[wsrc + e]);
"#;

/// W4A8: `k0` is a multiple of 16, so the 16 codes are 8 packed bytes (even column in the LOW
/// nibble) under ONE E4M3 group scale. The per-code arithmetic is `w4a8_gemm.cu`'s
/// `dequant_int4_grouped_to_int8`: `round_ties_even(codebook[q] * scale)`, clamped to `[-127, 127]`.
const STAGE_B_W4A8: &str = r#"
        const uint psrc = bcol * (uint(K) / 2u) + k0 / 2u;
        const float scale = sc_w4a8_group_scale(uint(uchar(srel[bcol * (uint(K) / 16u) + k0 / 16u])));
#pragma clang loop unroll(full)
        for (uint e = 0u; e < 8u; ++e) {
            const uint packed_byte = uint(uchar(packed[psrc + e]));
            Bs[lr * LDS + lk + 2u * e] =
                float(char(clamp(rint(codebook[packed_byte & 0x0Fu] * scale), -127.0f, 127.0f)));
            Bs[lr * LDS + lk + 2u * e + 1u] =
                float(char(clamp(rint(codebook[packed_byte >> 4] * scale), -127.0f, 127.0f)));
        }
"#;

struct Kernels {
    quant_convrot: MetalKernel,
    quant_w4a8: MetalKernel,
    gemm_convrot: MetalKernel,
    gemm_w4a8: MetalKernel,
}

thread_local! {
    // MetalKernel is thread-affine; never place one in a Send provider/model base. These handles
    // capture no arrays and share MLX's compiled-pipeline cache.
    static KERNELS: RefCell<Option<Kernels>> = const { RefCell::new(None) };
}

fn quant_source(numerator: &str) -> String {
    QUANT_TEMPLATE
        .replace("{NUMERATOR}", numerator)
        .replace("QUANT_THREADS", &format!("{QUANT_THREADS}u"))
        .replace("QUANT_SIMDGROUPS", &format!("{}u", QUANT_THREADS / 32))
}

fn gemm_source(stage_b: &str) -> String {
    GEMM_TEMPLATE
        .replace("{STAGE_B}", stage_b)
        .replace("FLUSH_K", &format!("{FLUSH_K}u"))
}

fn with_kernels<T>(f: impl FnOnce(&Kernels) -> Result<T>) -> Result<T> {
    crate::nn::prepare_retained_compilation_thread();
    KERNELS.with(|cache| {
        let mut cache = cache.borrow_mut();
        if cache.is_none() {
            let make = |name: &str, inputs: &[&str], outputs: &[&str], body: &str| {
                MetalKernel::with_options(name, inputs, outputs, body, HEADER, true, false)
            };
            *cache = Some(Kernels {
                quant_convrot: make(
                    "sc_convrot_int8_quant_v2",
                    &["x"],
                    &["q", "s"],
                    &quant_source("float(x[base + j])"),
                )?,
                quant_w4a8: make(
                    "sc_w4a8_int8_quant_v2",
                    &["x"],
                    &["q", "s"],
                    &quant_source("float(T(float(x[base + j])))"),
                )?,
                gemm_convrot: make(
                    "sc_int8_gemm_convrot_v1",
                    &["q", "w", "sx", "sw", "b"],
                    &["y"],
                    &gemm_source(STAGE_B_CONVROT),
                )?,
                gemm_w4a8: make(
                    "sc_int8_gemm_w4a8_v1",
                    &["q", "packed", "srel", "codebook", "sx", "sw", "b"],
                    &["y"],
                    &gemm_source(STAGE_B_W4A8),
                )?,
            });
        }
        f(cache.as_ref().expect("initialized above"))
    })
}

/// Which codec's activation-quantizer numerator to use: the two codecs differ only in whether the
/// numerator is cast to the activation dtype before the divide.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Numerator {
    /// S4 ConvRot: the rotated input is already in the activation dtype.
    ConvRot,
    /// W4A8: the F32 butterfly output, cast to the activation dtype before the divide.
    W4A8,
}

fn dtype_code(dtype: Dtype) -> i32 {
    match dtype {
        Dtype::Bfloat16 => 1,
        Dtype::Float16 => 2,
        _ => 0,
    }
}

/// Row-wise symmetric int8 quantization of a contiguous `[m, k]` operand: `(codes I8 [m, k],
/// scales F32 [m])`, `act` being the activation dtype that defines Kitchen's cast chain.
pub fn quantize_rows(
    x: &Array,
    m: usize,
    k: usize,
    act: Dtype,
    numerator: Numerator,
) -> Result<(Array, Array)> {
    let (m, k) = (m as i32, k as i32);
    with_kernels(|kernels| {
        let kernel = match numerator {
            Numerator::ConvRot => &kernels.quant_convrot,
            Numerator::W4A8 => &kernels.quant_w4a8,
        };
        let mut out = kernel
            .apply()
            .inputs([x])
            .output_shape([m, k], Dtype::Int8)
            .output_shape([m], Dtype::Float32)
            .template_arg("T", act)
            .template_arg("M", m)
            .template_arg("K", k)
            .template_arg("DTYPE", dtype_code(act))
            .grid(QUANT_THREADS as i32, m, 1)
            .thread_group(QUANT_THREADS as i32, 1, 1)
            .run()?
            .into_iter();
        let codes = out.next().ok_or("INT8 quantizer returned no codes")?;
        let scales = out.next().ok_or("INT8 quantizer returned no scales")?;
        Ok((codes, scales))
    })
}

/// The weight operand of [`project`].
pub enum Weight<'a> {
    /// ConvRot I8 `[n, k]` codes.
    ConvRot { codes: &'a Array },
    /// Packed W4A8: I8 `[n, k/2]` two-int4 bytes, U8 `[n, k/16]` E4M3 group scales, F32 `[16]`
    /// codebook.
    W4A8 {
        packed: &'a Array,
        s_rel: &'a Array,
        codebook: &'a Array,
    },
}

/// `y[m, n] = T(((acc · sx) · sw) + bias)` with `acc` the exact int32 `codes · weightᵀ`.
/// `sw` and `bias` are F32 with `n` elements; `out` is the activation dtype.
#[allow(clippy::too_many_arguments)]
pub fn project(
    codes: &Array,
    sx: &Array,
    weight: Weight<'_>,
    sw: &Array,
    bias: &Array,
    m: usize,
    n: usize,
    k: usize,
    out: Dtype,
) -> Result<Array> {
    check_geometry(m, n, k).map_err(crate::Error::Msg)?;
    let ((gx, gy, gz), (tx, ty, tz)) = gemm_dispatch(m, n);
    with_kernels(|kernels| {
        let dispatch = match weight {
            Weight::ConvRot { codes: w } => kernels
                .gemm_convrot
                .apply()
                .inputs([codes, w, sx, sw, bias]),
            Weight::W4A8 {
                packed,
                s_rel,
                codebook,
            } => kernels
                .gemm_w4a8
                .apply()
                .inputs([codes, packed, s_rel, codebook, sx, sw, bias]),
        };
        let mut y = dispatch
            .output_shape([m as i32, n as i32], out)
            .template_arg("T", out)
            .template_arg("M", m as i32)
            .template_arg("N", n as i32)
            .template_arg("K", k as i32)
            .grid(gx, gy, gz)
            .thread_group(tx, ty, tz)
            .run()?;
        Ok(y.remove(0))
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_flush_window_of_worst_case_products_is_exact_in_f32() {
        // Every partial sum of a window is an integer of magnitude <= FLUSH_K * 2^14, and f32 holds
        // every integer up to 2^24 exactly.
        let bound = FLUSH_K as u64 * MAX_PRODUCT;
        assert!(bound < 1 << 24, "{bound}");
        assert_eq!(bound as f32 as u64, bound);
        // ...whereas a single H3 FFN-down reduction is NOT: this is why the K axis is windowed.
        assert!(14_336 * MAX_PRODUCT > 1 << 24);
    }

    #[test]
    fn the_int32_accumulator_cannot_overflow_at_max_k() {
        assert!(MAX_K as u64 * MAX_PRODUCT <= i32::MAX as u64);
        assert!((MAX_K as u64 + 1) * MAX_PRODUCT > i32::MAX as u64);
        assert!(check_geometry(1, 1, MAX_K - MAX_K % BK).is_ok());
        assert!(check_geometry(1, 1, MAX_K + BK).is_err());
    }

    #[test]
    fn geometry_admits_every_h3_projection_and_refuses_ragged_depth() {
        let rows = 52_683;
        for (k, n) in [(5376, 21504), (7168, 5376), (5376, 28672), (14336, 5376)] {
            assert!(check_geometry(rows, n, k).is_ok(), "{k}x{n}");
        }
        assert!(check_geometry(5, 16, 192).is_ok());
        assert!(check_geometry(5, 16, 200).is_err());
        assert!(check_geometry(0, 16, 192).is_err());
        assert!(check_geometry(1 << 20, 1 << 13, 64).is_err());
    }

    #[test]
    fn dispatch_covers_every_output_with_one_threadgroup_per_tile() {
        for (m, n) in [(1, 1), (5, 16), (64, 64), (65, 129), (52_683, 28_672)] {
            let ((gx, gy, gz), (tx, ty, tz)) = gemm_dispatch(m, n);
            assert_eq!((tx, ty, tz), (GEMM_THREADS as i32, 1, 1));
            assert_eq!(gz, 1);
            assert_eq!(gx as usize % GEMM_THREADS, 0);
            let (tiles_n, tiles_m) = (gx as usize / GEMM_THREADS, gy as usize);
            assert!(tiles_n * BN >= n && (tiles_n - 1) * BN < n);
            assert!(tiles_m * BM >= m && (tiles_m - 1) * BM < m);
        }
        assert_eq!(flush_windows(14_336), 28);
        assert_eq!(flush_windows(192), 1);
    }

    #[test]
    fn templates_are_fully_substituted() {
        for source in [
            quant_source("float(x[base + j])"),
            gemm_source(STAGE_B_CONVROT),
            gemm_source(STAGE_B_W4A8),
        ] {
            for token in [
                "{NUMERATOR}",
                "{STAGE_B}",
                "QUANT_THREADS",
                "QUANT_SIMDGROUPS",
                "FLUSH_K",
            ] {
                assert!(!source.contains(token), "{token}");
            }
        }
        assert!(gemm_source(STAGE_B_CONVROT).contains("FLUSH_STEPS = 512u / BK"));
        assert!(quant_source("N").contains("j += 256u"));
        assert!(quant_source("N").contains("partial[8u]"));
    }
}
