//! Row-chunk planning and exact transient accounting for packed FLUX3 projections.
//!
//! A packed GEMM is row-independent: every activation row is quantized and dotted against the
//! packed weight on its own, so splitting the rows into launches changes no output bit. Chunking
//! therefore bounds the per-launch transient (and keeps every launch inside the i32 element range
//! the device kernels index with) without touching steps, dimensions, references or numerics.
use super::artifacts::NativeDitStorage;
use super::quantization::DitFormat;
use crate::{Error, Result};

/// Per-launch transient ceiling for one packed projection. The memory contract charges this
/// bound, so it must stay the planner's actual cap.
pub const PACKED_GEMM_TRANSIENT_BUDGET: u64 = 256 * 1024 * 1024;

/// Device quantized-matmul kernels index activations and outputs with i32.
pub const MAX_LAUNCH_ELEMENTS: u64 = i32::MAX as u64;

fn invalid(detail: impl std::fmt::Display) -> Error {
    Error::Unsupported(format!("flux3 packed gemm: {detail}"))
}

/// Exact transient bytes ONE activation row of a `[k] -> [n]` packed projection allocates:
///
/// * the f32 copy of the compute-dtype input row the quantized kernels read (`4k`);
/// * its dynamic activation quantization: GGML Q8_1 blocks (36 B per 32, the CUDA path for both
///   formats and the CPU path for Q4_1) bound the CPU Q8_0 path's 34 B; the publisher FP8R/NVFP4
///   paths quantize to one byte (FP8) or half a byte plus scales (NVFP4) per element;
/// * the f32 kernel output (`4n`) and its single rounding to the compute dtype (`n · width`);
///   the publisher NVFP4 GEMM writes BF16 directly (`2n`), but first copies its rows into a
///   zero-padded compute-dtype buffer (`k · width`).
///
/// MLX affine is W4/8A16: it reads the compute-dtype input directly with no activation copy.
/// Per-launch fixed costs that do not scale with the chunk are [`launch_overhead_bytes`].
pub fn transient_bytes_per_row(
    storage: NativeDitStorage,
    k: usize,
    n: usize,
    compute_width: u64,
) -> Result<u64> {
    if k == 0 || n == 0 || !matches!(compute_width, 2 | 4) {
        return Err(invalid("empty projection or non-float compute width"));
    }
    let (k, n) = (k as u64, n as u64);
    let activation = match storage {
        NativeDitStorage::GgmlQ4_1 | NativeDitStorage::GgmlQ8_0 => {
            if k % 32 != 0 {
                return Err(invalid("GGML width must be whole 32-element blocks"));
            }
            4 * k + k / 32 * 36
        }
        NativeDitStorage::Publisher(DitFormat::Fp8Rowwise) => 4 * k + k + 4,
        NativeDitStorage::Publisher(DitFormat::Nvfp4) => {
            if k % 16 != 0 {
                return Err(invalid("NVFP4 width must be whole 16-element blocks"));
            }
            k * compute_width + 4 * k + k / 2 + k / 16
        }
        NativeDitStorage::MlxAffine { .. } => 0,
        NativeDitStorage::Publisher(DitFormat::Dense) => {
            return Err(invalid("a dense projection is not a packed GEMM"))
        }
    };
    let output = match storage {
        NativeDitStorage::MlxAffine { .. } => n * compute_width,
        NativeDitStorage::Publisher(DitFormat::Nvfp4) => 2 * n,
        _ => 4 * n + n * compute_width,
    };
    activation
        .checked_add(output)
        .ok_or_else(|| invalid("transient byte overflow"))
}

/// NVFP4 W4A4 pads every launch to the FP4 GEMM row alignment.
pub const NVFP4_M_ALIGN: u64 = 16;

/// Bytes one launch allocates regardless of its row count: the NVFP4 path pads up to
/// `NVFP4_M_ALIGN - 1` extra rows (each costing a full row's transient) and its block scales up
/// to a 128-row x 4-block atom (`127 · ceil4(k/16)` bytes). Other formats have none. A planner
/// caller subtracts this from the budget so the WHOLE launch stays within it.
pub fn launch_overhead_bytes(
    storage: NativeDitStorage,
    k: usize,
    n: usize,
    compute_width: u64,
) -> Result<u64> {
    if storage != NativeDitStorage::Publisher(DitFormat::Nvfp4) {
        return Ok(0);
    }
    let per_row = transient_bytes_per_row(storage, k, n, compute_width)?;
    let blocks = (k as u64 / 16).div_ceil(4) * 4;
    (NVFP4_M_ALIGN - 1)
        .checked_mul(per_row)
        .and_then(|v| v.checked_add(127 * blocks))
        .ok_or_else(|| invalid("launch overhead overflow"))
}

/// Contiguous `(start, len)` launches covering `rows`, each within `budget` transient bytes and
/// within [`MAX_LAUNCH_ELEMENTS`] for both the `[len, k]` input and the `[len, n]` output.
/// A single row that cannot fit is a typed refusal, never a silently dropped or degraded row.
pub fn plan_row_chunks(
    rows: usize,
    k: usize,
    n: usize,
    per_row_bytes: u64,
    budget: u64,
) -> Result<Vec<(usize, usize)>> {
    if rows == 0 {
        return Ok(Vec::new());
    }
    if per_row_bytes == 0 || k == 0 || n == 0 {
        return Err(invalid("empty row plan"));
    }
    let by_budget = budget / per_row_bytes;
    let by_index = MAX_LAUNCH_ELEMENTS / k.max(n) as u64;
    let chunk = by_budget.min(by_index).min(rows as u64) as usize;
    if chunk == 0 {
        return Err(invalid(format!(
            "one {k}->{n} row needs {per_row_bytes} B, above the {budget} B launch budget"
        )));
    }
    Ok((0..rows)
        .step_by(chunk)
        .map(|start| (start, chunk.min(rows - start)))
        .collect())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn transient_is_exact_per_format_and_rejects_partial_blocks() {
        // Q4_1 / Q8_0: 4k f32 input + k/32·36 Q8_1 + 4n f32 out + 2n bf16 out.
        let ggml = 4 * 3072 + 3072 / 32 * 36 + 4 * 9216 + 2 * 9216;
        for storage in [NativeDitStorage::GgmlQ4_1, NativeDitStorage::GgmlQ8_0] {
            assert_eq!(
                transient_bytes_per_row(storage, 3072, 9216, 2).unwrap(),
                ggml
            );
        }
        assert_eq!(
            transient_bytes_per_row(
                NativeDitStorage::MlxAffine {
                    bits: 4,
                    group_size: 64
                },
                3072,
                9216,
                2
            )
            .unwrap(),
            2 * 9216
        );
        assert!(transient_bytes_per_row(NativeDitStorage::GgmlQ4_1, 48, 8, 2).is_err());
        assert!(
            transient_bytes_per_row(NativeDitStorage::Publisher(DitFormat::Dense), 32, 8, 2)
                .is_err()
        );
    }

    /// NVFP4 prices its padded compute-dtype copy, writes BF16 directly, and carries a fixed
    /// per-launch padding overhead; FP8R prices its E4M3 bytes and row scale.
    #[test]
    fn publisher_transients_include_padding_and_direct_bf16_output() {
        let (k, n) = (3072, 9216);
        let nvfp4 = NativeDitStorage::Publisher(DitFormat::Nvfp4);
        let per_row = transient_bytes_per_row(nvfp4, k, n, 2).unwrap();
        assert_eq!(
            per_row,
            2 * 3072 + 4 * 3072 + 3072 / 2 + 3072 / 16 + 2 * 9216
        );
        assert_eq!(
            launch_overhead_bytes(nvfp4, k, n, 2).unwrap(),
            15 * per_row + 127 * 192
        );
        let fp8 = NativeDitStorage::Publisher(DitFormat::Fp8Rowwise);
        assert_eq!(
            transient_bytes_per_row(fp8, k, n, 2).unwrap(),
            4 * 3072 + 3072 + 4 + 4 * 9216 + 2 * 9216
        );
        assert_eq!(launch_overhead_bytes(fp8, k, n, 2).unwrap(), 0);
        assert_eq!(
            launch_overhead_bytes(NativeDitStorage::GgmlQ4_1, k, n, 2).unwrap(),
            0
        );
    }

    /// *Mutations that red this:* ignoring the budget, the i32 index cap, or the tail chunk.
    #[test]
    fn chunks_cover_every_row_once_within_budget_and_index_range() {
        let plan = plan_row_chunks(10, 4, 4, 30, 100).unwrap();
        assert_eq!(plan, vec![(0, 3), (3, 3), (6, 3), (9, 1)]);
        let total: usize = plan.iter().map(|(_, len)| len).sum();
        assert_eq!(total, 10);
        assert_eq!(plan_row_chunks(5, 4, 4, 1, u64::MAX).unwrap(), vec![(0, 5)]);
        // 40960 reference+image tokens of a 61440-wide MLP output: rows·n would overflow i32.
        let n = 61_440;
        let plan = plan_row_chunks(40_960, 3072, n, 1, u64::MAX).unwrap();
        assert!(plan
            .iter()
            .all(|(_, len)| (*len as u64) * (n as u64) <= MAX_LAUNCH_ELEMENTS));
        assert_eq!(plan.iter().map(|(_, l)| l).sum::<usize>(), 40_960);
        assert!(plan_row_chunks(3, 4, 4, 101, 100).is_err());
        assert!(plan_row_chunks(0, 4, 4, 1, 100).unwrap().is_empty());
    }
}
