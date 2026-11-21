//! MLX-packed → GGML **repack** primitives (sc-9085 spike; the byte-level half of the shared
//! packed-load module, sc-9086, epic 9083). The packed-**detect** loaders that call these to build a
//! [`QLinear`](super::QLinear) / [`QEmbedding`](super::QEmbedding) live in the parent [`super`]
//! module; this file owns only the pure MLX-triple → GGML-`Q4_1` / dequant conversions and their
//! order-sensitivity unit tests.
//!
//! The hosted quant tiers (epic 8506, e.g. `SceneWorks/z-image-turbo-mlx`) store each quantized
//! Linear as the MLX packed triple `{base}.weight` (u32 codes) + `{base}.scales` + `{base}.biases`:
//! **group-wise affine** quantization along the input dimension, dequantized as
//! `w = scale · q + bias` with the codes packed LSB-first (4-bit: element *k* of a row is nibble
//! `k % 8` of u32 `k / 8`; 8-bit: byte `k % 4` of u32 `k / 4`). The **group size** is a per-tier
//! quantizer choice ([`MLX_GROUP_SIZE`] = 64 for the z-image / flux tiers; the boogu tier packs at
//! group 32, sc-9410). It is not recoverable from the packed shapes alone (a `[out, in/8]` /
//! `[out, in/g]` Q4 pair and a `[out, in/4]` / `[out, in/g']` Q8 pair collide for some `g`/`g'`), so
//! the group-size-aware entry points ([`repack_mlx_q4_to_q4_1_gs`] / [`dequant_mlx_q8_gs`], and the
//! `*_gs` inference helpers) take it explicitly — read from the component `config.json`'s
//! `quantization.group_size` ([`super::PackedConfig`]). The group-64 wrappers are the historical
//! default the z-image / flux seams call.
//!
//! GGML's `Q4_1` is the **same affine form** over 32-element blocks (`block_q4_1` = f16 `d` +
//! f16 `m` + 16 nibble bytes; element `j` in the low nibble of `qs[j]`, element `j + 16` in the
//! high nibble — `w = d · q + m`). One MLX group of size `g` (a multiple of 32) therefore splits
//! **losslessly** into `g / 32` `Q4_1` blocks sharing `d = scale`, `m = bias` (a group-64 pack
//! yields two blocks, a group-32 pack one): [`repack_mlx_q4_to_q4_1`] is a pure nibble permutation
//! plus a bf16 → f16 cast of the per-group scale/bias (exact whenever the bf16 value's exponent is
//! in f16 range — its 7-bit mantissa always fits f16's 10; the real-weight spike test censuses the
//! exceptions). The repacked [`QTensor`] feeds the existing `QLinear` dequant-on-forward machinery
//! (sc-7702: the weight is dequantized to a dense matmul; the int8 activation fast path stays off).
//!
//! 8-bit has no affine GGML container (`Q8_0`/`Q8_1` are symmetric, no bias/min), so the MLX Q8
//! tier cannot be repacked losslessly: [`dequant_mlx_q8`] materializes the exact MLX grid values
//! and the caller re-quantizes to `Q8_0` (`QTensor::quantize`), a second 8-bit rounding the spike
//! measured at 0.56 % mean / 0.87 % worst relative RMS on the real z-image Q8 tier (~10× below
//! Q4's inherent error) — the accepted Q8 path per the sc-9085 decision; an exact affine custom
//! dequant path stays an option if a Q8 A/B ever shows the gap.

use std::borrow::Cow;

use candle_core::quantized::{GgmlDType, QStorage, QTensor};
use candle_core::{DType, Device, Result, Tensor};
use half::f16;

/// Cumulative per-phase wall-clock counters for packed loads (sc-26691), read by the real-weight load
/// profiler. Nanoseconds, relaxed atomics: negligible next to the work they time.
pub mod profile {
    use std::sync::atomic::{AtomicU64, Ordering::Relaxed};
    pub static PARTS_NS: AtomicU64 = AtomicU64::new(0);
    pub static REPACK_NS: AtomicU64 = AtomicU64::new(0);
    pub static UPLOAD_NS: AtomicU64 = AtomicU64::new(0);
    pub static CALLS: AtomicU64 = AtomicU64::new(0);
    pub(super) fn add(c: &AtomicU64, t: std::time::Instant) {
        c.fetch_add(t.elapsed().as_nanos() as u64, Relaxed);
    }
    /// `(calls, parts_s, repack_s, upload_s)` accumulated since process start.
    pub fn snapshot() -> (u64, f64, f64, f64) {
        let s = |c: &AtomicU64| c.load(Relaxed) as f64 / 1e9;
        (
            CALLS.load(Relaxed),
            s(&PARTS_NS),
            s(&REPACK_NS),
            s(&UPLOAD_NS),
        )
    }
}

/// MLX's default (and the hosted tiers' only) quantization group size along the input dim.
pub const MLX_GROUP_SIZE: usize = 64;

/// GGML `Q4_1` block: 32 elements in 20 bytes (f16 `d`, f16 `m`, 16 nibble bytes).
pub const Q4_1_BLOCK: usize = 32;
pub const Q4_1_BLOCK_BYTES: usize = 20;

/// Derive the quant bit-width from an MLX packed pair's shapes at the default group size 64 — the
/// z-image / flux tiers' convention: `scales` is `[out, in/64]` ⇒ `in = cols · 64`; the u32-packed
/// `weight` is `[out, in·bits/32]` ⇒ `bits = wq_cols · 32 / in`. For a non-64 group tier (boogu packs
/// at 32) use [`mlx_packed_bits_gs`] with the `config.json` group size.
pub fn mlx_packed_bits(wq_cols: usize, scales_cols: usize) -> usize {
    mlx_packed_bits_gs(wq_cols, scales_cols, MLX_GROUP_SIZE)
}

/// Derive the quant bit-width from an MLX packed pair's shapes at an explicit `group_size` (sc-9410):
/// `scales` is `[out, in/group_size]` ⇒ `in = scales_cols · group_size`; the u32-packed `weight` is
/// `[out, in·bits/32]` ⇒ `bits = wq_cols · 32 / in`. The group size is the tier's quantizer choice
/// (read from `config.json`), not recoverable from the shapes alone.
pub fn mlx_packed_bits_gs(wq_cols: usize, scales_cols: usize, group_size: usize) -> usize {
    let in_dim = scales_cols * group_size;
    wq_cols * 32 / in_dim
}

/// Whether an f32 value survives the f32 → f16 → f32 round-trip exactly — the only lossy step the
/// Q4 repack can have (bf16 scales/biases whose exponent falls outside f16's range). Used by the
/// spike census; production repack proceeds regardless (the deviation is one f16 ulp of a scale).
pub fn f16_exact(x: f32) -> bool {
    f16::from_f32(x).to_f32() == x
}

/// The exact affine grid values an MLX pack represents, computed the way candle's `Q4_1`/CPU
/// dequant computes them (`f32(scale) · q + f32(bias)` per element, f32 accumulate): the repack's
/// loss-free reference. `codes` are the unpacked per-element quant codes of one row-major
/// `[out, in]` tensor; `scales`/`biases` are the per-group f32 values.
pub fn affine_grid(
    codes: &[u8],
    scales: &[f32],
    biases: &[f32],
    in_dim: usize,
    group_size: usize,
) -> Vec<f32> {
    let groups_per_row = in_dim / group_size;
    codes
        .iter()
        .enumerate()
        .map(|(i, &q)| {
            let (row, col) = (i / in_dim, i % in_dim);
            let g = row * groups_per_row + col / group_size;
            scales[g] * q as f32 + biases[g]
        })
        .collect()
}

/// Unpack MLX u32-packed 4-bit codes (LSB-first nibbles) into one `u8` code per element.
pub fn unpack_mlx_q4(wq: &[u32], out_dim: usize, in_dim: usize) -> Vec<u8> {
    let words_per_row = in_dim / 8;
    let mut codes = vec![0u8; out_dim * in_dim];
    for row in 0..out_dim {
        for (w, &word) in wq[row * words_per_row..(row + 1) * words_per_row]
            .iter()
            .enumerate()
        {
            let base = row * in_dim + w * 8;
            for nib in 0..8 {
                codes[base + nib] = ((word >> (4 * nib)) & 0xF) as u8;
            }
        }
    }
    codes
}

/// An MLX packed triple pulled to CPU vectors: u32 code words + f32 scales/biases + the tensor's
/// `[out, in]` dims.
type MlxParts = (Vec<u32>, Vec<f32>, Vec<f32>, usize, usize);

/// Pull an MLX packed triple to CPU vectors, validating the Q4 shape contract at `group_size`
/// (`wq [out, in/8]`, `scales`/`biases` `[out, in/group_size]`).
fn q4_parts(wq: &Tensor, scales: &Tensor, biases: &Tensor, group_size: usize) -> Result<MlxParts> {
    let (out_dim, wq_cols) = wq.dims2()?;
    let (s_rows, s_cols) = scales.dims2()?;
    let in_dim = s_cols * group_size;
    if s_rows != out_dim || biases.dims2()? != (s_rows, s_cols) || wq_cols * 8 != in_dim {
        candle_core::bail!(
            "not an MLX group-{group_size} Q4 pack: wq {:?}, scales {:?}, biases {:?}",
            wq.shape(),
            scales.shape(),
            biases.shape()
        );
    }
    let cpu = Device::Cpu;
    let wq = wq.to_device(&cpu)?.flatten_all()?.to_vec1::<u32>()?;
    let scales = scales
        .to_device(&cpu)?
        .to_dtype(DType::F32)?
        .flatten_all()?
        .to_vec1::<f32>()?;
    let biases = biases
        .to_device(&cpu)?
        .to_dtype(DType::F32)?
        .flatten_all()?
        .to_vec1::<f32>()?;
    Ok((wq, scales, biases, out_dim, in_dim))
}

/// Repack an MLX group-64 affine **Q4** triple into a GGML **`Q4_1`** [`QTensor`] on `device` — the
/// z-image / flux tiers' default group size. See [`repack_mlx_q4_to_q4_1_gs`] for the general form.
pub fn repack_mlx_q4_to_q4_1(
    wq: &Tensor,
    scales: &Tensor,
    biases: &Tensor,
    device: &Device,
) -> Result<QTensor> {
    repack_mlx_q4_to_q4_1_gs(wq, scales, biases, MLX_GROUP_SIZE, device)
}

/// Spread the 8 nibbles of `w` into the low nibbles of 8 bytes (nibble `k` -> byte `k`, little-endian).
#[inline(always)]
fn spread_nibbles(w: u32) -> u64 {
    let mut x = w as u64;
    x = (x | (x << 16)) & 0x0000_FFFF_0000_FFFF;
    x = (x | (x << 8)) & 0x00FF_00FF_00FF_00FF;
    x = (x | (x << 4)) & 0x0F0F_0F0F_0F0F_0F0F;
    x
}

/// Build the GGML `Q4_1` byte stream of an MLX Q4 triple (sc-26691). Word-wise, row-parallel
/// equivalent of the per-byte unpack/rebuild: a Q4_1 block's 32 elements are 4 consecutive u32 words
/// `w0..w3`, and `qs[j] = nib(w[j / 8], j % 8) | nib(w[2 + j / 8], j % 8) << 4`, so each pair
/// `(w0, w2)` / `(w1, w3)` yields 8 `qs` bytes with two nibble-spreads. Output is byte-identical to
/// the reference ([`q4_1_bytes_reference`] under test). `group_size` must be a multiple of 32.
pub fn q4_1_bytes(
    wq: &[u32],
    scales: &[f32],
    biases: &[f32],
    out_dim: usize,
    in_dim: usize,
    group_size: usize,
) -> Vec<u8> {
    use rayon::prelude::*;
    let groups_per_row = in_dim / group_size;
    let blocks_per_group = group_size / Q4_1_BLOCK;
    let words_per_row = in_dim / 8;
    let row_bytes = in_dim / Q4_1_BLOCK * Q4_1_BLOCK_BYTES;
    let mut bytes = vec![0u8; out_dim * row_bytes];
    if bytes.is_empty() {
        return bytes;
    }
    bytes
        .par_chunks_mut(row_bytes)
        .zip(wq.par_chunks(words_per_row))
        .enumerate()
        .for_each(|(row, (out, words))| {
            let mut blocks = out.chunks_exact_mut(Q4_1_BLOCK_BYTES);
            let mut words = words.chunks_exact(4);
            for g in 0..groups_per_row {
                let d = f16::from_f32(scales[row * groups_per_row + g]).to_le_bytes();
                let m = f16::from_f32(biases[row * groups_per_row + g]).to_le_bytes();
                for _ in 0..blocks_per_group {
                    let (blk, w) = (blocks.next().unwrap(), words.next().unwrap());
                    blk[0..2].copy_from_slice(&d);
                    blk[2..4].copy_from_slice(&m);
                    blk[4..12].copy_from_slice(
                        &(spread_nibbles(w[0]) | spread_nibbles(w[2]) << 4).to_le_bytes(),
                    );
                    blk[12..20].copy_from_slice(
                        &(spread_nibbles(w[1]) | spread_nibbles(w[3]) << 4).to_le_bytes(),
                    );
                }
            }
        });
    bytes
}

/// The pre-sc-26691 per-byte, single-threaded repack, kept verbatim as the bit-identity oracle for
/// [`q4_1_bytes`] (and the old side of the timing micro-benchmark).
pub fn q4_1_bytes_reference(
    wq: &[u32],
    scales: &[f32],
    biases: &[f32],
    out_dim: usize,
    in_dim: usize,
    group_size: usize,
) -> Vec<u8> {
    let codes = unpack_mlx_q4(wq, out_dim, in_dim);

    let blocks = out_dim * in_dim / Q4_1_BLOCK;
    let mut bytes = Vec::with_capacity(blocks * Q4_1_BLOCK_BYTES);
    let groups_per_row = in_dim / group_size;
    for row in 0..out_dim {
        for g in 0..groups_per_row {
            let d = f16::from_f32(scales[row * groups_per_row + g]);
            let m = f16::from_f32(biases[row * groups_per_row + g]);
            let group = &codes[row * in_dim + g * group_size..][..group_size];
            // One MLX group of `group_size` = `group_size / 32` consecutive Q4_1 blocks, same d/m.
            for block in group.chunks_exact(Q4_1_BLOCK) {
                bytes.extend_from_slice(&d.to_le_bytes());
                bytes.extend_from_slice(&m.to_le_bytes());
                for j in 0..Q4_1_BLOCK / 2 {
                    bytes.push(block[j] | (block[j + Q4_1_BLOCK / 2] << 4));
                }
            }
        }
    }
    bytes
}

/// Repack an MLX affine **Q4** triple at an explicit `group_size` (a multiple of 32) into a GGML
/// **`Q4_1`** [`QTensor`] on `device` (sc-9410) — lossless up to the bf16 → f16 scale/bias cast (see
/// the module docs). One MLX group splits into `group_size / 32` consecutive `Q4_1` blocks sharing
/// the group's `d = scale` / `m = bias`. The result plugs straight into the per-crate
/// `QLinear::Quantized` dequant-on-forward path.
pub fn repack_mlx_q4_to_q4_1_gs(
    wq: &Tensor,
    scales: &Tensor,
    biases: &Tensor,
    group_size: usize,
    device: &Device,
) -> Result<QTensor> {
    if !group_size.is_multiple_of(Q4_1_BLOCK) {
        candle_core::bail!(
            "MLX Q4 group_size {group_size} must be a multiple of the Q4_1 block ({Q4_1_BLOCK})"
        );
    }
    let t0 = std::time::Instant::now();
    let (wq, scales, biases, out_dim, in_dim) = q4_parts(wq, scales, biases, group_size)?;
    profile::add(&profile::PARTS_NS, t0);
    let t0 = std::time::Instant::now();
    let bytes = q4_1_bytes(&wq, &scales, &biases, out_dim, in_dim, group_size);
    profile::add(&profile::REPACK_NS, t0);

    // MUST be `Cow::Borrowed`: candle's `as_t_slice` takes the Cow by value and returns a slice
    // borrowed from it, so an `Owned` cow's backing Vec is dropped before `from_data` copies the
    // blocks out — a use-after-free that reads freed memory (garbage weights, found in the sc-9085
    // spike). Borrowed keeps `bytes` alive across the call; `from_data` clones into its own Vec.
    let t0 = std::time::Instant::now();
    let storage = QStorage::from_data(Cow::Borrowed(&bytes), device, GgmlDType::Q4_1)?;
    let q = QTensor::new(storage, (out_dim, in_dim));
    profile::add(&profile::UPLOAD_NS, t0);
    profile::CALLS.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
    q
}

/// The exact f32 grid values of an MLX group-64 Q4 pack — the repack's loss-free reference (spike
/// verification; not a load path). See [`dequant_mlx_q4_reference_gs`] for the general form.
pub fn dequant_mlx_q4_reference(wq: &Tensor, scales: &Tensor, biases: &Tensor) -> Result<Tensor> {
    dequant_mlx_q4_reference_gs(wq, scales, biases, MLX_GROUP_SIZE)
}

/// The exact f32 grid values of an MLX Q4 pack at an explicit `group_size` (sc-9410) — the repack's
/// loss-free reference.
pub fn dequant_mlx_q4_reference_gs(
    wq: &Tensor,
    scales: &Tensor,
    biases: &Tensor,
    group_size: usize,
) -> Result<Tensor> {
    let (wq, scales, biases, out_dim, in_dim) = q4_parts(wq, scales, biases, group_size)?;
    // Reference through the same f16 cast the repack bakes in, so "lossless" is measured against
    // what a Q4_1 block can represent (the f16-exactness census reports the cast's own deviation).
    let scales: Vec<f32> = scales.iter().map(|&s| f16::from_f32(s).to_f32()).collect();
    let biases: Vec<f32> = biases.iter().map(|&b| f16::from_f32(b).to_f32()).collect();
    let codes = unpack_mlx_q4(&wq, out_dim, in_dim);
    Tensor::from_vec(
        affine_grid(&codes, &scales, &biases, in_dim, group_size),
        (out_dim, in_dim),
        &Device::Cpu,
    )
}

/// Materialize an MLX group-64 affine **Q8** triple as its exact f32 grid values — the z-image /
/// flux tiers' default group size. See [`dequant_mlx_q8_gs`] for the general form.
pub fn dequant_mlx_q8(wq: &Tensor, scales: &Tensor, biases: &Tensor) -> Result<Tensor> {
    dequant_mlx_q8_gs(wq, scales, biases, MLX_GROUP_SIZE)
}

/// Materialize an MLX affine **Q8** triple (`wq [out, in/4]` u32, LSB-first bytes) at an explicit
/// `group_size` (sc-9410) as its exact f32 grid values. 8-bit has no affine GGML container, so the
/// Q8 tier path is dequant-then-`QTensor::quantize(…, Q8_0)` — this is the dequant half (and the
/// error reference the spike measures the `Q8_0` re-quantization against).
pub fn dequant_mlx_q8_gs(
    wq: &Tensor,
    scales: &Tensor,
    biases: &Tensor,
    group_size: usize,
) -> Result<Tensor> {
    let (out_dim, wq_cols) = wq.dims2()?;
    let (s_rows, s_cols) = scales.dims2()?;
    let in_dim = s_cols * group_size;
    if s_rows != out_dim || biases.dims2()? != (s_rows, s_cols) || wq_cols * 4 != in_dim {
        candle_core::bail!(
            "not an MLX group-{group_size} Q8 pack: wq {:?}, scales {:?}, biases {:?}",
            wq.shape(),
            scales.shape(),
            biases.shape()
        );
    }
    let cpu = Device::Cpu;
    let words = wq.to_device(&cpu)?.flatten_all()?.to_vec1::<u32>()?;
    let scales = scales
        .to_device(&cpu)?
        .to_dtype(DType::F32)?
        .flatten_all()?
        .to_vec1::<f32>()?;
    let biases = biases
        .to_device(&cpu)?
        .to_dtype(DType::F32)?
        .flatten_all()?
        .to_vec1::<f32>()?;

    let mut codes = vec![0u8; out_dim * in_dim];
    for (w, &word) in words.iter().enumerate() {
        for b in 0..4 {
            codes[w * 4 + b] = ((word >> (8 * b)) & 0xFF) as u8;
        }
    }
    Tensor::from_vec(
        affine_grid(&codes, &scales, &biases, in_dim, group_size),
        (out_dim, in_dim),
        &Device::Cpu,
    )
}

/// **Producer** side of the packed-tier format (sc-10026): quantize a dense `[out, in]` `weight` into
/// the MLX affine packed triple `(wq, scales, biases)` the consume path ([`repack_mlx_q4_to_q4_1_gs`] /
/// [`dequant_mlx_q8_gs`] / [`super::QLinear::from_packed_gs`]) reads — the exact inverse. Lets candle
/// **host** its own diffusers-keyed packed tiers (rather than reverse-remapping the native-keyed MLX
/// tiers), so the packed-detect seam loads them with no key/layout translation.
///
/// Group-wise **affine** along the input dim, matching the dequant `w = scale·q + bias`: for each
/// group of `group_size` consecutive elements of a row, `bias = min`, `scale = (max − min) / (2^bits −
/// 1)`, `q = round((w − bias) / scale)` clamped to `[0, 2^bits − 1]` (a constant group has `scale = 0`
/// and every `q = 0`, so it dequantizes back to `bias = w` exactly). Codes pack LSB-first into u32
/// (`bits = 4`: 8 codes/word, `wq` is `[out, in/8]`; `bits = 8`: 4 codes/word, `wq` is `[out, in/4]`),
/// and `scales`/`biases` are `[out, in/group_size]` f32 — the shapes [`mlx_packed_bits_gs`] and the
/// loaders assume. `bits` must be 4 or 8; `in` must be a multiple of `group_size`, which for `bits = 4`
/// must itself be a multiple of 8 (a u32 word holds 8 nibbles) — group 64 satisfies both.
///
/// Q4 round-trips **losslessly** through the `Q4_1` repack (same affine form); Q8's consume path
/// re-quantizes to `Q8_0` (see the module docs), so a Q8 tier carries that second rounding — packing
/// here is exact either way (the codes reproduce `scale·q + bias`).
pub fn pack_mlx_affine(
    weight: &Tensor,
    bits: usize,
    group_size: usize,
) -> Result<(Tensor, Tensor, Tensor)> {
    if bits != 4 && bits != 8 {
        candle_core::bail!("pack_mlx_affine: bits must be 4 or 8 (got {bits})");
    }
    let (out_dim, in_dim) = weight.dims2()?;
    if !in_dim.is_multiple_of(group_size) {
        candle_core::bail!("pack_mlx_affine: in dim {in_dim} not a multiple of group {group_size}");
    }
    let codes_per_word = 32 / bits; // 8 for Q4, 4 for Q8
    if !group_size.is_multiple_of(codes_per_word) {
        candle_core::bail!(
            "pack_mlx_affine: group {group_size} must be a multiple of {codes_per_word} for {bits}-bit packing"
        );
    }
    let levels = ((1u32 << bits) - 1) as f32; // 15 (Q4) or 255 (Q8)
    let w = weight
        .to_device(&Device::Cpu)?
        .to_dtype(DType::F32)?
        .flatten_all()?
        .to_vec1::<f32>()?;
    let groups_per_row = in_dim / group_size;
    let mut codes = vec![0u8; out_dim * in_dim];
    let mut scales = Vec::with_capacity(out_dim * groups_per_row);
    let mut biases = Vec::with_capacity(out_dim * groups_per_row);
    for row in 0..out_dim {
        for g in 0..groups_per_row {
            let base = row * in_dim + g * group_size;
            let group = &w[base..base + group_size];
            let (mut lo, mut hi) = (group[0], group[0]);
            for &v in &group[1..] {
                lo = lo.min(v);
                hi = hi.max(v);
            }
            let scale = (hi - lo) / levels;
            for (k, &v) in group.iter().enumerate() {
                let q = if scale > 0.0 {
                    ((v - lo) / scale).round().clamp(0.0, levels) as u8
                } else {
                    0
                };
                codes[base + k] = q;
            }
            scales.push(scale);
            biases.push(lo);
        }
    }
    // Pack codes LSB-first into u32 words (nibble/byte `k` of word into bit `bits·k`).
    let words: Vec<u32> = codes
        .chunks_exact(codes_per_word)
        .map(|c| {
            c.iter()
                .enumerate()
                .fold(0u32, |acc, (i, &q)| acc | ((q as u32) << (bits * i)))
        })
        .collect();
    let dev = Device::Cpu;
    let wq = Tensor::from_vec(words, (out_dim, in_dim / codes_per_word), &dev)?;
    let scales = Tensor::from_vec(scales, (out_dim, groups_per_row), &dev)?;
    let biases = Tensor::from_vec(biases, (out_dim, groups_per_row), &dev)?;
    Ok((wq, scales, biases))
}
