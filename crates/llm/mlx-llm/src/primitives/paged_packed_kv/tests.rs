use super::*;
use crate::primitives::packed_attention::{attention_f32_masked, PackedAttentionShape};
use crate::primitives::packed_group_affine_kv::RetainedPackedKernel;
use crate::primitives::packed_metal::{PackedKernelPath, PackedMetalGpuFamily, PackedMetalKernel};
use crate::primitives::{select_decoder_cache_with_reader, PackedCacheRequest};
use std::sync::Arc;

/// Deterministic K/V/Q values with a distinct range per 32-channel group and per token, so a
/// wrong page, group or metadata index is numerically visible.
fn values(seed: u64, tokens: usize, heads: usize, width: usize) -> Vec<f32> {
    let mut state = seed.wrapping_mul(0x9e37_79b9_7f4a_7c15) | 1;
    (0..heads * tokens * width)
        .map(|i| {
            state ^= state << 13;
            state ^= state >> 7;
            state ^= state << 17;
            let unit = (state >> 40) as f32 / (1u64 << 24) as f32 - 0.5;
            let (channel, token) = (i % width, (i / width) % tokens);
            unit * (1.0 + (channel / 32) as f32 * 0.5) + (channel / 32) as f32 * 0.75
                - (token % 7) as f32 * 0.125
        })
        .collect()
}

fn bhsd(data: &[f32], heads: usize, tokens: usize, width: usize, dtype: Dtype) -> Array {
    Array::from_slice(data, &[1, heads as i32, tokens as i32, width as i32])
        .as_dtype(dtype)
        .unwrap()
}

fn host<T: mlx_rs::ArrayElement + Copy>(array: &Array) -> Vec<T> {
    let array = crate::primitives::nn::contiguous(array).unwrap();
    array.eval().unwrap();
    array.as_slice::<T>().to_vec()
}

fn host_f32(array: &Array) -> Vec<f32> {
    host::<f32>(&array.as_dtype(Dtype::Float32).unwrap())
}

#[allow(clippy::arc_with_non_send_sync)]
fn reader(bits: PackedCodeBits, family: PackedMetalGpuFamily) -> CompiledKernelHandle {
    CompiledKernelHandle::new(Arc::new(
        PackedMetalKernel::for_identity_family_and_bits("paged-test", family, bits).unwrap(),
    ))
}

/// Independent host dequantization of one layer of `cache`: K/V of every live token, read by
/// walking the host page ids over host copies of the pool arrays with this test's own code
/// unpacking (no MLX dequantization, no kernel). `[Hkv][tokens][D]` flattened.
fn oracle_kv(cache: &PagedPackedKvCache, layer: usize) -> (Vec<f32>, Vec<f32>) {
    let pool = cache.pool.borrow();
    let (h, d, pt, bits) = (
        pool.kv_heads,
        pool.head_dimension,
        pool.page_tokens,
        pool.bits,
    );
    let group = PACKED_METAL_QUANT_GROUP_SIZE;
    let arrays = &pool.arrays[layer];
    let kc = host::<u8>(&arrays.key_codes);
    let ks = host_f32(&arrays.key_scales);
    let kz = host_f32(&arrays.key_zeros);
    let vc = host::<u8>(&arrays.value_codes);
    let vs = host_f32(&arrays.value_scales);
    let vz = host_f32(&arrays.value_zeros);
    let state = &cache.layers[layer];
    let kt = host_f32(state.key_tail.as_ref().unwrap());
    let vt = host_f32(state.value_tail.as_ref().unwrap());
    let per_byte = 8 / bits.bits() as usize;
    let mask = (1u32 << bits.bits()) - 1;
    let code = |bytes: &[u8], base: usize, index: usize| -> f32 {
        let byte = u32::from(bytes[base + index / per_byte]);
        ((byte >> ((index % per_byte) * bits.bits() as usize)) & mask) as f32
    };
    let (pg, kw, vw, vg) = (pt / group, group * d / per_byte, d / per_byte, d / group);
    let tokens = state.packed + state.tail_rows;
    let mut keys = vec![0.0; h * tokens * d];
    let mut vals = vec![0.0; h * tokens * d];
    for head in 0..h {
        for t in 0..tokens {
            for c in 0..d {
                let out = (head * tokens + t) * d + c;
                if t < state.packed {
                    let page = cache.page_ids[t / pt];
                    let kg = (page * h + head) * pg + (t % pt) / group;
                    keys[out] =
                        kz[kg * d + c] + ks[kg * d + c] * code(&kc, kg * kw, (t % group) * d + c);
                    let row = (page * h + head) * pt + t % pt;
                    vals[out] = vz[row * vg + c / group]
                        + vs[row * vg + c / group] * code(&vc, row * vw, c);
                } else {
                    let r = (head * group + t - state.packed) * d + c;
                    keys[out] = kt[r];
                    vals[out] = vt[r];
                }
            }
        }
    }
    (keys, vals)
}

/// Append one layer-0 chunk directly (kernel-level fixtures; no attention).
fn append(cache: &mut PagedPackedKvCache, keys: &[f32], vals: &[f32], tokens: usize, dtype: Dtype) {
    let (h, d) = {
        let pool = cache.pool.borrow();
        (pool.kv_heads, pool.head_dimension)
    };
    cache
        .append(
            0,
            &bhsd(keys, h, tokens, d, dtype),
            &bhsd(vals, h, tokens, d, dtype),
        )
        .unwrap();
    cache.logical_len += tokens;
}

struct KernelCase {
    dtype: Dtype,
    width: usize,
    kv_heads: usize,
    query_heads: usize,
    page_tokens: usize,
    lengths: [usize; 3],
    query_len: usize,
    mask: PackedAttentionMask,
}

/// The fused paged reader against an independent dequantize-then-attend fp32 oracle, for
/// several sequences of one dispatch with ragged lengths whose pages are interleaved and
/// recycled (non-contiguous, non-monotonic page order), partially filled last pages, residual
/// tails, a sequence with no full group, GQA, causal/sliding/no masks, decode and multi-row
/// queries, every code width, and every kernel path (per-row single/split on both families,
/// fp32 tiled single/split, and NAX where MLX runs it). Unread page-table columns hold an
/// out-of-range id, so a read past a sequence's pages corrupts the output.
#[test]
fn paged_reader_matches_the_independent_fp32_oracle_across_paths_and_page_orders() {
    let case =
        |dtype, width, kv_heads, query_heads, page_tokens, lengths, query_len, mask| KernelCase {
            dtype,
            width,
            kv_heads,
            query_heads,
            page_tokens,
            lengths,
            query_len,
            mask,
        };
    let causal = PackedAttentionMask::Causal;
    let cases = [
        case(Dtype::Float32, 64, 2, 4, 32, [161, 70, 20], 1, causal),
        case(Dtype::Float16, 128, 2, 6, 64, [300, 129, 97], 3, causal),
        case(
            Dtype::Bfloat16,
            256,
            1,
            2,
            96,
            [200, 33, 100],
            1,
            PackedAttentionMask::None,
        ),
        case(
            Dtype::Float32,
            128,
            2,
            4,
            64,
            [257, 140, 100],
            2,
            PackedAttentionMask::SlidingWindow(45),
        ),
        case(Dtype::Bfloat16, 64, 2, 4, 32, [230, 96, 40], 32, causal),
        case(Dtype::Float16, 128, 1, 4, 128, [190, 120, 50], 16, causal),
    ];
    let mut nax_ran = false;
    for bits in PackedCodeBits::ALL {
        for (index, case) in cases.iter().enumerate() {
            nax_ran |= check_kernel_case(bits, index, case);
        }
    }
    eprintln!("NAX paged path exercised: {nax_ran}");
}

/// Run one [`KernelCase`]; returns whether the NAX path ran.
fn check_kernel_case(bits: PackedCodeBits, index: usize, case: &KernelCase) -> bool {
    let (h, d, pt) = (case.kv_heads, case.width, case.page_tokens);
    let pool = PackedPagePool::new(1, h, d, pt, bits).unwrap();
    let handle = reader(bits, PackedMetalGpuFamily::Apple7OrNewer);
    let mut caches = (0..3)
        .map(|_| PagedPackedKvCache::with_pool(pool.clone(), handle.clone()).unwrap())
        .collect::<Vec<_>>();
    // Recycle: a decoy takes pages, then frees them, so the sequences reuse those ids in LIFO
    // (descending) order before fresh ones.
    {
        let mut decoy = PagedPackedKvCache::with_pool(pool.clone(), handle.clone()).unwrap();
        let data = values(999, 5 * pt, h, d);
        append(&mut decoy, &data, &data, 5 * pt, case.dtype);
    }
    assert_eq!(
        pool.borrow().live_pages(),
        0,
        "the decoy returned its pages"
    );
    // Interleave the sequences' appends in uneven chunks so their pages interleave.
    let seed = index as u64 + 1;
    let keys = (0..3)
        .map(|b| values(17 * seed + b as u64, case.lengths[b], h, d))
        .collect::<Vec<_>>();
    let vals = (0..3)
        .map(|b| values(101 * seed + b as u64, case.lengths[b], h, d))
        .collect::<Vec<_>>();
    let mut fed = [0usize; 3];
    let chunks = [37, 5, 64, 1, 29];
    let mut turn = 0;
    while (0..3).any(|b| fed[b] < case.lengths[b]) {
        for b in 0..3 {
            let chunk = chunks[(turn + b) % chunks.len()].min(case.lengths[b] - fed[b]);
            if chunk == 0 {
                continue;
            }
            let take = |data: &[f32]| {
                (0..h)
                    .flat_map(|head| {
                        let start = (head * case.lengths[b] + fed[b]) * d;
                        data[start..start + chunk * d].to_vec()
                    })
                    .collect::<Vec<_>>()
            };
            append(
                &mut caches[b],
                &take(&keys[b]),
                &take(&vals[b]),
                chunk,
                case.dtype,
            );
            fed[b] += chunk;
        }
        turn += 1;
    }
    let ids = caches
        .iter()
        .map(|c| c.page_ids.clone())
        .collect::<Vec<_>>();
    assert!(
        ids.iter()
            .any(|pages| pages.windows(2).any(|w| w[1] != w[0] + 1)),
        "case {index}: some sequence's pages are non-contiguous: {ids:?}"
    );
    let query = Array::from_slice(
        &(0..3 * case.query_heads * case.query_len * d)
            .map(|i| ((i * 37 + 3) % 61) as f32 * 0.02 - 0.6)
            .collect::<Vec<_>>(),
        &[3, case.query_heads as i32, case.query_len as i32, d as i32],
    )
    .as_dtype(case.dtype)
    .unwrap();
    let query_f32 = host_f32(&query);
    let stack = |parts: Vec<Array>| concatenate_axis(&parts.iter().collect::<Vec<_>>(), 0).unwrap();
    let key_tail = stack(
        caches
            .iter()
            .map(|c| c.layers[0].key_tail.clone().unwrap())
            .collect(),
    );
    let value_tail = stack(
        caches
            .iter()
            .map(|c| c.layers[0].value_tail.clone().unwrap())
            .collect(),
    );
    let columns = ids.iter().map(Vec::len).max().unwrap().max(1);
    let table = ids
        .iter()
        .flat_map(|pages| {
            let mut row = pages.iter().map(|&p| p as i32).collect::<Vec<_>>();
            row.resize(columns, i32::MAX / 2);
            row
        })
        .collect::<Vec<_>>();
    let page_table = Array::from_slice(&table, &[3, columns as i32]);
    let sequences = caches
        .iter()
        .map(|c| PagedSequenceExtent {
            packed_tokens: c.layers[0].packed,
            kv_tokens: c.layers[0].packed + c.layers[0].tail_rows,
        })
        .collect::<Vec<_>>();
    // A partially filled last page (or a sequence with no page yet) in every case whose page
    // is larger than one group.
    assert!(
        pt == PACKED_METAL_QUANT_GROUP_SIZE
            || sequences
                .iter()
                .any(|s| s.packed_tokens == 0 || s.packed_tokens % pt != 0)
    );
    let packed_mask = match case.mask {
        PackedAttentionMask::None => PackedMask::None,
        PackedAttentionMask::Causal => PackedMask::Causal,
        PackedAttentionMask::SlidingWindow(w) => PackedMask::SlidingWindow(w),
        PackedAttentionMask::Additive => unreachable!(),
    };
    // Independent oracle, per sequence.
    let mut reference = Vec::new();
    for (b, cache) in caches.iter().enumerate() {
        let (k, v) = oracle_kv(cache, 0);
        let kv = sequences[b].kv_tokens;
        let rows = case.query_heads * case.query_len * d;
        reference.extend(
            attention_f32_masked(
                PackedAttentionShape {
                    batch: 1,
                    query_heads: case.query_heads,
                    kv_heads: h,
                    query_len: case.query_len,
                    kv_len: kv,
                    head_dim: d,
                },
                &query_f32[b * rows..(b + 1) * rows],
                |_, hh, t, c| k[(hh * kv + t) * d + c],
                |_, hh, t, c| v[(hh * kv + t) * d + c],
                (d as f32).powf(-0.5),
                case.mask,
            )
            .unwrap(),
        );
    }
    let pool_ref = pool.borrow();
    let arrays = &pool_ref.arrays[0];
    let args = PagedPackedAttentionArgs {
        query: &query,
        key_codes: &arrays.key_codes,
        key_scales: &arrays.key_scales,
        key_zeros: &arrays.key_zeros,
        key_tail: &key_tail,
        value_codes: &arrays.value_codes,
        value_scales: &arrays.value_scales,
        value_zeros: &arrays.value_zeros,
        value_tail: &value_tail,
        page_table: &page_table,
        page_tokens: pt,
        sequences: &sequences,
        code_bits: bits,
        mask: packed_mask,
    };
    // About twice the worst observed oracle error per query dtype (and, for bf16, head
    // dimension: its 8-bit mantissa rounding of a D = 256 query dominates), so the negative
    // control below — one wrong page — lands far outside it.
    let tolerance = match (case.dtype, d) {
        (Dtype::Float32, _) => 1e-5,
        (Dtype::Float16, _) => 6e-3,
        (_, 256) => 3e-2,
        _ => 8e-3,
    };
    use PackedMetalGpuFamily::{Apple7OrNewer, ConservativeUnknownApple};
    let mut paths = vec![
        (
            ConservativeUnknownApple,
            PackedKernelPath::PerRow { splits: 1 },
        ),
        (
            ConservativeUnknownApple,
            PackedKernelPath::PerRow { splits: 3 },
        ),
        (Apple7OrNewer, PackedKernelPath::PerRow { splits: 1 }),
        (Apple7OrNewer, PackedKernelPath::PerRow { splits: 64 }),
        (Apple7OrNewer, PackedKernelPath::Tiled { splits: 1 }),
        (Apple7OrNewer, PackedKernelPath::Tiled { splits: 2 }),
    ];
    let qualified =
        PackedMetalKernel::for_identity_family_and_bits("paged-test", Apple7OrNewer, bits).unwrap();
    let nax = qualified.nax_selection(d, case.dtype).selected;
    if nax {
        paths.push((Apple7OrNewer, PackedKernelPath::NaxTiled { splits: 1 }));
        paths.push((Apple7OrNewer, PackedKernelPath::NaxTiled { splits: 3 }));
    }
    let mut case_max = 0.0f32;
    for (family, path) in paths {
        let kernel =
            PackedMetalKernel::for_identity_family_and_bits("paged-test", family, bits).unwrap();
        let output = host_f32(&kernel.dispatch_paged_path(&args, path).unwrap());
        assert_eq!(output.len(), reference.len());
        for (i, (actual, expected)) in output.iter().zip(&reference).enumerate() {
            case_max = case_max.max((actual - expected).abs());
            assert!(
                (actual - expected).abs() <= tolerance,
                "{bits:?} case {index} {family:?} {path:?} element {i}: {actual} != {expected}"
            );
        }
    }
    // The production planner picks one of those paths for this shape.
    let planned = host_f32(&qualified.dispatch_paged(&args).unwrap());
    for (actual, expected) in planned.iter().zip(&reference) {
        assert!((actual - expected).abs() <= tolerance);
    }
    // Negative control: a wrong-but-valid page table (two sequences' last pages swapped, so
    // every mask's visible range reads the wrong page) must move the output well past the
    // tolerance, or the tolerance could not catch a page-walk error.
    let paged = (0..3).filter(|&b| !ids[b].is_empty()).collect::<Vec<_>>();
    let (a, b) = (paged[0], paged[1]);
    let mut swapped = table.clone();
    let (last_a, last_b) = (
        a * columns + ids[a].len() - 1,
        b * columns + ids[b].len() - 1,
    );
    swapped.swap(last_a, last_b);
    let swapped = Array::from_slice(&swapped, &[3, columns as i32]);
    let wrong = host_f32(
        &qualified
            .dispatch_paged(&PagedPackedAttentionArgs {
                page_table: &swapped,
                ..args
            })
            .unwrap(),
    );
    let wrong_error = wrong
        .iter()
        .zip(&reference)
        .map(|(actual, expected)| (actual - expected).abs())
        .fold(0.0f32, f32::max);
    assert!(
        wrong_error > 2.0 * tolerance,
        "{bits:?} case {index}: a swapped page moved the output only {wrong_error} \
             (tolerance {tolerance})"
    );
    eprintln!(
        "paged parity {bits:?} case {index} {:?} D={d}: max abs {case_max}",
        case.dtype
    );
    nax
}

/// A paged dispatch is refused before encoding when a sequence's quantized tokens are not
/// covered by its page-table row, the page size disagrees with the pool arrays, or the
/// reader's code width differs; a pool refuses a page that is not whole groups.
#[test]
fn paged_dispatch_validates_page_geometry_before_encoding() {
    let bits = PackedCodeBits::Eight;
    assert!(PackedPagePool::new(1, 1, 64, 48, bits).is_err());
    assert!(PackedPagePool::new(1, 1, 96, 32, bits).is_err());
    let pool = PackedPagePool::new(1, 1, 64, 32, bits).unwrap();
    let handle = reader(bits, PackedMetalGpuFamily::Apple7OrNewer);
    let mut cache = PagedPackedKvCache::with_pool(pool.clone(), handle).unwrap();
    let data = values(3, 70, 1, 64);
    append(&mut cache, &data, &data, 70, Dtype::Float32);
    assert_eq!(cache.page_ids.len(), 2);
    let query = Array::from_slice(&[0.1f32; 64], &[1, 1, 1, 64]);
    let pool_ref = pool.borrow();
    let arrays = &pool_ref.arrays[0];
    let state = &cache.layers[0];
    let short_table = Array::from_slice(&[0i32], &[1, 1]);
    let table = cache.page_table.clone().unwrap();
    let sequences = [PagedSequenceExtent {
        packed_tokens: 64,
        kv_tokens: 70,
    }];
    let kernel = PackedMetalKernel::for_identity_family_and_bits(
        "paged-test",
        PackedMetalGpuFamily::Apple7OrNewer,
        bits,
    )
    .unwrap();
    let args = |table, page_tokens, code_bits| PagedPackedAttentionArgs {
        query: &query,
        key_codes: &arrays.key_codes,
        key_scales: &arrays.key_scales,
        key_zeros: &arrays.key_zeros,
        key_tail: state.key_tail.as_ref().unwrap(),
        value_codes: &arrays.value_codes,
        value_scales: &arrays.value_scales,
        value_zeros: &arrays.value_zeros,
        value_tail: state.value_tail.as_ref().unwrap(),
        page_table: table,
        page_tokens,
        sequences: &sequences,
        code_bits,
        mask: PackedMask::Causal,
    };
    assert!(kernel.dispatch_paged(&args(&table, 32, bits)).is_ok());
    assert!(kernel
        .dispatch_paged(&args(&short_table, 32, bits))
        .is_err());
    assert!(kernel.dispatch_paged(&args(&table, 64, bits)).is_err());
    assert!(kernel
        .dispatch_paged(&args(&table, 32, PackedCodeBits::Four))
        .is_err());
}

const LAYERS: usize = 2;

fn contiguous_cache(
    handle: &CompiledKernelHandle,
    layers: usize,
    kv_heads: usize,
    width: usize,
) -> Box<dyn KvCache> {
    select_decoder_cache_with_reader(
        PackedCacheRequest {
            enabled: true,
            backend: "mlx-metal".into(),
            identity: handle.cache_identity().to_owned(),
            layers,
            batch: 1,
            kv_heads,
            head_dimension: width,
            group_size: PACKED_METAL_QUANT_GROUP_SIZE,
            bits: handle.code_bits(),
            query_length: 1,
            has_mask: false,
        },
        handle.clone(),
    )
    .into_cache()
}

/// Q/K/V of one synthetic layer step.
fn qkv(
    seed: u64,
    step: usize,
    (heads, kv_heads, width): (usize, usize, usize),
    dtype: Dtype,
) -> (Array, Array, Array) {
    (
        bhsd(&values(seed, step, heads, width), heads, step, width, dtype),
        bhsd(
            &values(seed + 7, step, kv_heads, width),
            kv_heads,
            step,
            width,
            dtype,
        ),
        bhsd(
            &values(seed + 13, step, kv_heads, width),
            kv_heads,
            step,
            width,
            dtype,
        ),
    )
}

/// One synthetic model step on a cache: per layer, fresh Q/K/V of `step` tokens attended
/// causally; the fused route where the cache takes it, else `update` + dense SDPA.
fn attend_step(
    cache: &mut dyn KvCache,
    seed: u64,
    step: usize,
    dims: (usize, usize, usize),
    dtype: Dtype,
) -> Vec<Vec<f32>> {
    let scale = (dims.2 as f32).powf(-0.5);
    (0..cache.num_layers())
        .map(|layer| {
            let (q, k, v) = qkv(seed * 31 + layer as u64, step, dims, dtype);
            let out = match cache
                .try_packed_attention(layer, &q, &k, &v, PackedAttentionMask::Causal, scale, false)
                .unwrap()
            {
                Some(out) => out,
                None => {
                    let (keys, vals) = cache.update(layer, &k, &v).unwrap();
                    sdpa(&q, &keys, &vals, scale, AttnMask::Causal).unwrap()
                }
            };
            host_f32(&out)
        })
        .collect()
}

fn max_abs(a: &[Vec<f32>], b: &[Vec<f32>]) -> f32 {
    a.iter()
        .zip(b)
        .flat_map(|(x, y)| x.iter().zip(y))
        .map(|(x, y)| (x - y).abs())
        .fold(0.0, f32::max)
}

fn pool(kv_heads: usize, width: usize, page_tokens: usize) -> Rc<RefCell<PackedPagePool>> {
    PackedPagePool::new(LAYERS, kv_heads, width, page_tokens, PackedCodeBits::Eight).unwrap()
}

fn key(name: &str) -> PagedModelKey {
    PagedModelKey::new(name, "test-decoder")
}

fn k8v8_reader() -> CompiledKernelHandle {
    crate::kv_policy::group_affine_reader(PackedCodeBits::Eight).unwrap()
}

/// Cache-level differential parity over a synthetic two-layer attention stack: the paged
/// compressed cache matches the contiguous compressed cache (same codes, same kernels) and the
/// dense paged cache within 8-bit rounding, through a fresh prefill, decode across page
/// boundaries, a short multi-row step (per-row kernel), and a long one (tiled/NAX kernel). The
/// fused calls build no dense copy: no gather is ever counted.
#[test]
fn paged_compressed_matches_contiguous_compressed_and_dense_paged_through_a_decode() {
    let dims = (4, 2, 64);
    for dtype in [Dtype::Float32, Dtype::Bfloat16] {
        let handle = k8v8_reader();
        let mut paged = PagedPackedKvCache::with_pool(pool(2, 64, 64), handle.clone()).unwrap();
        let mut contiguous = contiguous_cache(&handle, LAYERS, 2, 64);
        let mut dense = PagedKvCache::new(LAYERS, 16);
        let steps = [70, 1, 1, 1, 5]
            .into_iter()
            .chain(std::iter::repeat_n(1, 30))
            .chain([20])
            .chain(std::iter::repeat_n(1, 40))
            .collect::<Vec<_>>();
        let mut to_dense = 0.0f32;
        let mut tokens = 0;
        for (i, &step) in steps.iter().enumerate() {
            let p = attend_step(&mut paged, i as u64, step, dims, dtype);
            let c = attend_step(contiguous.as_mut(), i as u64, step, dims, dtype);
            let d = attend_step(&mut dense, i as u64, step, dims, dtype);
            tokens += step;
            // Same codes, same kernels: only where they live differs.
            assert_eq!(p, c, "{dtype:?} step {i}: paged vs contiguous compressed");
            to_dense = to_dense.max(max_abs(&p, &d));
            assert_eq!(paged.offset(), tokens as i32);
        }
        let dense_bound = match dtype {
            Dtype::Float32 => 3e-2,
            _ => 8e-2,
        };
        assert!(
            to_dense <= dense_bound,
            "{dtype:?}: paged vs dense {to_dense}"
        );
        eprintln!("{dtype:?}: paged == contiguous, vs dense paged {to_dense}");
        assert_eq!(paged.page_ids.len(), (tokens / 32 * 32).div_ceil(64));
        let evidence = paged.packed_evidence().unwrap();
        assert!(evidence.dense_gather_calls.is_empty(), "{evidence:?}");
        assert_eq!(evidence.full_cache_dequantizations, 0);
        assert_eq!(
            evidence.representation_identity,
            paged_packed_identity(PackedCodeBits::Eight)
        );
        // Every attention call after the fresh prefill ran fused.
        let fused = (LAYERS * (steps.len() - 1)) as u64;
        assert_eq!(paged.fused_calls(), fused);
        assert!(evidence
            .kernel_paths
            .iter()
            .any(|path| path.kernel != crate::primitives::PACKED_PER_ROW_KERNEL));
    }
}

/// Physical-byte accounting includes page metadata: the sequence's storage is its pages'
/// share of the pool's measured arrays (codes; f16 scale/zero) plus its page table and dense
/// residuals, and it stays well under the dense bf16 K/V of the same tokens despite the
/// partially filled last page.
#[test]
fn physical_bytes_count_pages_page_metadata_table_and_residuals() {
    let (kv_heads, width) = (4, 128);
    let pool = pool(kv_heads, width, 64);
    let mut cache = PagedPackedKvCache::with_pool(pool.clone(), k8v8_reader()).unwrap();
    let dims = (4, kv_heads, width);
    attend_step(&mut cache, 1, 2020, dims, Dtype::Bfloat16);
    for i in 0..10 {
        attend_step(&mut cache, 2 + i, 1, dims, Dtype::Bfloat16);
    }
    let tokens = 2030_u64;
    let storage = cache.compressed_storage().unwrap().unwrap();
    assert_eq!(storage.tokens, tokens);
    let pool_storage = pool.borrow().storage();
    let (page_codes, page_metadata) = pool.borrow().page_bytes();
    assert_eq!(
        pool_storage.code_bytes,
        pool_storage.capacity_pages * page_codes
    );
    assert_eq!(
        pool_storage.metadata_bytes,
        pool_storage.capacity_pages * page_metadata
    );
    let pages = cache.page_ids.len() as u64;
    assert_eq!(
        pages, 32,
        "2016 quantized tokens: 31 full pages and a partial 32nd"
    );
    assert_eq!(pool_storage.live_pages, pages);
    assert_eq!(storage.device_code_bytes, pages * page_codes);
    let residuals = (LAYERS * 2 * kv_heads * 32 * width * 2) as u64;
    let table = array_bytes(cache.page_table.as_ref().unwrap());
    assert_eq!(table, 32 * 4, "a 32-column Int32 page table");
    assert_eq!(
        storage.device_metadata_bytes,
        pages * page_metadata + residuals + table
    );
    // Per page and layer and head: K and V codes 64·128 bytes each; metadata K 2 groups ×
    // 128 channels × (scale + zero) f16, V 64 tokens × 4 channel groups × (scale + zero) f16.
    assert_eq!(page_codes, (LAYERS * kv_heads * 64 * 128 * 2) as u64);
    assert_eq!(
        page_metadata,
        (LAYERS * kv_heads * (2 * 128 * 4 + 64 * 4 * 4)) as u64
    );
    let dense_bf16 = LAYERS as u64 * 2 * kv_heads as u64 * tokens * width as u64 * 2;
    let ratio = storage.device_bytes() as f64 / dense_bf16 as f64;
    assert!(ratio < 0.6, "paged K8V8 holds {ratio:.3} of dense bf16");
    // Slack: 32 unused rows of the last page and 18 unused residual rows.
    assert_eq!(cache.slack_tokens(), 32 * 64 - 2016 + 32 - 14);
    eprintln!(
        "paged storage {} B for {tokens} tokens = {ratio:.3} of dense bf16 (slack {} tokens)",
        storage.device_bytes(),
        cache.slack_tokens()
    );
}

/// Page reuse and fragmentation: sequences interleaved on one pool, one freed mid-way and its
/// pages recycled by a third (lowest free id first), the second freed while the third still
/// grows — so the third's page table runs back to a lower id — each produce exactly the
/// outputs they produce alone on a fresh pool, and the pool returns to zero live pages when
/// every sequence is dropped.
#[test]
fn interleaved_freed_and_recycled_pages_do_not_change_any_sequence() {
    let dims = (2, 1, 64);
    let dtype = Dtype::Float32;
    let handle = k8v8_reader();
    let alone = |seed: u64, steps: &[usize]| {
        let mut cache = PagedPackedKvCache::with_pool(pool(1, 64, 32), handle.clone()).unwrap();
        steps
            .iter()
            .enumerate()
            .map(|(i, &step)| attend_step(&mut cache, seed * 1000 + i as u64, step, dims, dtype))
            .collect::<Vec<_>>()
    };
    let a_steps = [40, 1, 30, 1, 1, 33];
    let b_steps = [50, 1, 1, 40, 1, 1];
    let c_steps = [100, 1, 64, 1, 1, 33];
    let shared = pool(1, 64, 32);
    let mut a = PagedPackedKvCache::with_pool(shared.clone(), handle.clone()).unwrap();
    let mut b = PagedPackedKvCache::with_pool(shared.clone(), handle.clone()).unwrap();
    let (mut a_out, mut b_out) = (Vec::new(), Vec::new());
    for i in 0..a_steps.len() {
        a_out.push(attend_step(
            &mut a,
            1000 + i as u64,
            a_steps[i],
            dims,
            dtype,
        ));
        b_out.push(attend_step(
            &mut b,
            2000 + i as u64,
            b_steps[i],
            dims,
            dtype,
        ));
    }
    assert_eq!(a_out, alone(1, &a_steps), "A interleaved with B");
    assert_eq!(b_out, alone(2, &b_steps), "B interleaved with A");
    assert!(
        a.page_ids().windows(2).any(|w| w[1] != w[0] + 1),
        "{:?}",
        a.page_ids()
    );
    let a_pages = a.page_ids().to_vec();
    drop(a);
    assert_eq!(shared.borrow().live_pages(), b.page_ids().len());
    let mut c = PagedPackedKvCache::with_pool(shared.clone(), handle.clone()).unwrap();
    let mut b = Some(b);
    let c_out = c_steps
        .iter()
        .enumerate()
        .map(|(i, &step)| {
            if i == 3 {
                // B's (lower) pages go back; C's next page takes the lowest of them.
                drop(b.take());
            }
            attend_step(&mut c, 3000 + i as u64, step, dims, dtype)
        })
        .collect::<Vec<_>>();
    assert_eq!(c_out, alone(3, &c_steps), "C on recycled pages");
    assert!(
        c.page_ids()
            .iter()
            .filter(|id| a_pages.contains(id))
            .count()
            == a_pages.len(),
        "C reused every page A freed: {:?} vs {a_pages:?}",
        c.page_ids()
    );
    assert!(
        c.page_ids().windows(2).any(|w| w[1] < w[0]),
        "non-monotonic page order: {:?}",
        c.page_ids()
    );
    drop(c);
    assert_eq!(shared.borrow().live_pages(), 0);
}

/// A reader that cancels (or faults) on one chosen paged dispatch.
#[derive(Debug)]
struct Interrupting {
    inner: PackedMetalKernel,
    fail_at: std::cell::Cell<usize>,
    canceled: bool,
}

impl RetainedPackedKernel for Interrupting {
    fn cache_identity(&self) -> &str {
        "interrupting"
    }
    fn backend(&self) -> &str {
        "mlx-metal"
    }
    fn retained_host_bytes_estimate(&self) -> usize {
        0
    }
    fn dispatch(&self, args: &crate::primitives::PackedAttentionArgs<'_>) -> Result<Array> {
        self.inner.dispatch(args)
    }
    fn dispatch_paged(&self, args: &PagedPackedAttentionArgs<'_>) -> Result<Array> {
        let left = self.fail_at.get();
        self.fail_at.set(left.wrapping_sub(1));
        match left {
            0 if self.canceled => Err(Error::Canceled),
            0 => Err(Error::Msg("injected paged dispatch fault".into())),
            _ => self.inner.dispatch_paged(args),
        }
    }
    fn code_bits(&self) -> PackedCodeBits {
        PackedCodeBits::Eight
    }
}

#[allow(clippy::arc_with_non_send_sync)]
pub(crate) fn interrupting(fail_at: usize, canceled: bool) -> CompiledKernelHandle {
    CompiledKernelHandle::new(Arc::new(Interrupting {
        inner: PackedMetalKernel::for_identity_family_and_bits(
            "interrupting",
            PackedMetalGpuFamily::Apple7OrNewer,
            PackedCodeBits::Eight,
        )
        .unwrap(),
        fail_at: std::cell::Cell::new(fail_at),
        canceled,
    }))
}

/// Cancellation or a fault at a later layer of a step that crossed a page boundary rolls back
/// the whole step: offset, residuals, and the page it allocated (the pool's live pages return
/// to their pre-step count); retrying produces the uninterrupted outputs. Dropping a sequence
/// mid-generation (a cancelled request) returns every page.
#[test]
fn a_cancelled_or_faulted_step_returns_its_pages_and_retries_exactly() {
    let dims = (2, 1, 64);
    let dtype = Dtype::Float32;
    let steps = [63, 1, 1];
    let run = |cache: &mut PagedPackedKvCache| {
        steps
            .iter()
            .enumerate()
            .map(|(i, &s)| attend_step(cache, i as u64, s, dims, dtype))
            .collect::<Vec<_>>()
    };
    let reference =
        run(&mut PagedPackedKvCache::with_pool(pool(1, 64, 32), k8v8_reader()).unwrap());
    for canceled in [true, false] {
        // Dispatch 0 is step 1's layer 0 (the 63-token prefill is fresh dense SDPA); fail
        // dispatch 1, step 1's layer 1. Step 1 completes token 64's group, whose page the
        // step allocates at layer 0.
        let shared = pool(1, 64, 32);
        let mut cache =
            PagedPackedKvCache::with_pool(shared.clone(), interrupting(1, canceled)).unwrap();
        let mut outputs = vec![attend_step(&mut cache, 0, steps[0], dims, dtype)];
        assert_eq!(cache.offset(), 63);
        let live = shared.borrow().live_pages();
        assert_eq!(live, 1);
        let scale = 0.125;
        let (q0, k0, v0) = qkv(31, 1, dims, dtype);
        assert!(cache
            .try_packed_attention(0, &q0, &k0, &v0, PackedAttentionMask::Causal, scale, false)
            .unwrap()
            .is_some());
        assert_eq!(
            shared.borrow().live_pages(),
            live + 1,
            "the step allocated a page"
        );
        let (q1, k1, v1) = qkv(32, 1, dims, dtype);
        let error = cache
            .try_packed_attention(1, &q1, &k1, &v1, PackedAttentionMask::Causal, scale, false)
            .unwrap_err();
        assert_eq!(matches!(error, Error::Canceled), canceled, "{error}");
        assert_eq!(cache.offset(), 63, "the interrupted step did not commit");
        assert_eq!(
            shared.borrow().live_pages(),
            live,
            "the step's page went back"
        );
        assert!(cache.pending.is_none());
        assert_eq!(cache.layers[0].tail_rows, 31, "layer 0's append was undone");
        // Retry the whole step, then finish: identical to the uninterrupted run.
        for (i, &s) in steps.iter().enumerate().skip(1) {
            outputs.push(attend_step(&mut cache, i as u64, s, dims, dtype));
        }
        assert_eq!(outputs, reference, "canceled={canceled}");
        assert_eq!(cache.fused_calls(), 4, "rolled-back calls are not accepted");
        drop(cache);
        assert_eq!(
            shared.borrow().live_pages(),
            0,
            "a dropped sequence returns its pages"
        );
    }
}

/// Truncation (speculative rollback) into a quantized group re-stages the cut group from its
/// codes and frees the pages past it, exactly as the contiguous compressed cache trims.
#[test]
fn truncate_into_a_quantized_group_matches_the_contiguous_cache_and_frees_pages() {
    let dims = (2, 1, 64);
    let dtype = Dtype::Float32;
    let handle = k8v8_reader();
    let shared = pool(1, 64, 32);
    let mut paged = PagedPackedKvCache::with_pool(shared.clone(), handle.clone()).unwrap();
    let mut contiguous = contiguous_cache(&handle, LAYERS, 1, 64);
    for (i, step) in [100, 1, 1].into_iter().enumerate() {
        attend_step(&mut paged, i as u64, step, dims, dtype);
        attend_step(contiguous.as_mut(), i as u64, step, dims, dtype);
    }
    assert_eq!(paged.page_ids.len(), 3);
    paged.truncate(45).unwrap();
    contiguous.truncate(45).unwrap();
    assert_eq!((paged.offset(), contiguous.offset()), (45, 45));
    assert_eq!(
        paged.page_ids.len(),
        1,
        "pages past the kept group are freed"
    );
    assert_eq!(shared.borrow().live_pages(), 1);
    let mut worst = 0.0f32;
    for (i, step) in [1, 1, 30, 1].into_iter().enumerate() {
        let p = attend_step(&mut paged, 50 + i as u64, step, dims, dtype);
        let c = attend_step(contiguous.as_mut(), 50 + i as u64, step, dims, dtype);
        worst = worst.max(max_abs(&p, &c));
    }
    assert!(worst <= 1e-5, "paged vs contiguous after truncate: {worst}");
    paged.reset().unwrap();
    assert_eq!(shared.borrow().live_pages(), 0);
    assert_eq!(paged.offset(), 0);
}

/// An unsupported call takes the dense gather fallback: the caller's `update` returns the
/// sequence's dequantized K/V (the independent oracle's values), the reason is counted, the
/// history stays compressed, and the next supported call is fused again. The report names the
/// gather.
#[test]
fn unsupported_calls_take_the_counted_dense_gather_and_history_stays_paged() {
    let dims = (2, 1, 64);
    let dtype = Dtype::Float32;
    let mut cache = PagedPackedKvCache::with_pool(pool(1, 64, 32), k8v8_reader()).unwrap();
    attend_step(&mut cache, 0, 70, dims, dtype);
    let fused = cache.fused_calls();
    let pages = cache.page_ids.clone();
    let (q, k, v) = qkv(9, 1, dims, dtype);
    for (layer, mask, scale, reason) in [
        (
            0,
            PackedAttentionMask::Additive,
            0.125,
            PagedFallbackReason::AdditiveMask,
        ),
        (
            1,
            PackedAttentionMask::Causal,
            0.5,
            PagedFallbackReason::AttentionScale,
        ),
    ] {
        assert!(cache
            .try_packed_attention(layer, &q, &k, &v, mask, scale, false)
            .unwrap()
            .is_none());
        let (keys, vals) = cache.update(layer, &k, &v).unwrap();
        assert_eq!(keys.shape(), &[1, 1, 71, 64]);
        let (ok, ov) = oracle_kv(&cache, layer);
        assert_eq!(
            host_f32(&keys),
            ok,
            "{reason:?}: gathered keys are the paged values"
        );
        assert_eq!(host_f32(&vals), ov);
        assert!(cache.dense_gathers().contains(&(reason, 1)));
    }
    assert_eq!(cache.offset(), 71);
    assert_eq!(
        cache.page_ids[..pages.len()],
        pages[..],
        "history stayed in its pages"
    );
    assert_eq!(cache.fused_calls(), fused);
    attend_step(&mut cache, 12, 1, dims, dtype);
    assert_eq!(
        cache.fused_calls(),
        fused + LAYERS as u64,
        "fused again after the gathers"
    );
    // A soft-cap preparation and a direct `update` (a caller wanting dense K/V) count too.
    cache
        .prepare_dense_fallback("score-softcap", "tanh before softmax")
        .unwrap();
    cache.update(0, &k, &v).unwrap();
    cache.update(1, &k, &v).unwrap();
    assert!(cache
        .dense_gathers()
        .contains(&(PagedFallbackReason::DenseCaller, 2)));
    let report =
        crate::kv_policy::compressed_report(KvCompressionFormat::GroupAffineK8V8, None, &cache)
            .unwrap();
    assert_eq!(
        report.fallback,
        Some(core_llm::KvCacheFallbackReason::DenseGather)
    );
    assert_eq!(report.counters.dense_gather_fallbacks, 4);
    assert_eq!(report.counters.full_cache_dequantizations, 0);
    assert_eq!(report.counters.dense_fallback_events, 0);
    assert_eq!(report.counters.fused_attention_calls, fused + LAYERS as u64);
    let detail = report.detail.unwrap();
    for reason in ["additive_mask", "attention_scale", "dense_caller"] {
        assert!(detail.contains(reason), "{detail}");
    }
    let ids = PagedFallbackReason::ALL.map(PagedFallbackReason::id);
    assert_eq!(
        ids.iter().collect::<std::collections::BTreeSet<_>>().len(),
        ids.len()
    );
}

/// The paged kernels are their own pipelines: a reader the contiguous cache has already
/// warmed is still cold for paged dispatch, so a failing first paged dispatch takes the counted
/// reader-unavailable gather rather than surfacing as a hard error.
#[test]
fn a_contiguously_warmed_reader_still_degrades_on_a_cold_paged_fault() {
    let dims = (2, 1, 64);
    let dtype = Dtype::Float32;
    let handle = interrupting(0, false);
    let mut contiguous = contiguous_cache(&handle, LAYERS, 1, 64);
    for (i, step) in [3, 1].into_iter().enumerate() {
        attend_step(contiguous.as_mut(), i as u64, step, dims, dtype);
    }
    assert!(
        handle.is_warmed(),
        "the contiguous kernels warmed the shared handle"
    );
    assert!(!handle.is_paged_warmed());
    let mut paged = PagedPackedKvCache::with_pool(pool(1, 64, 32), handle).unwrap();
    let (q, k, v) = qkv(0, 3, dims, dtype);
    let scale = (64f32).powf(-0.5);
    assert!(paged
        .try_packed_attention(0, &q, &k, &v, PackedAttentionMask::Causal, scale, false)
        .unwrap()
        .is_none());
    paged.update(0, &k, &v).unwrap();
    assert_eq!(
        paged.dense_gathers(),
        vec![(PagedFallbackReason::ReaderUnavailable, 1)]
    );
    assert!(paged.handle.is_none());
}

/// A reader whose paged kernels fail on their cold dispatch is dropped before any packed output
/// is published; that call and every later one are served by the gather fallback with the
/// reader-unavailable reason, and the outputs stay the fused paged attention's.
#[test]
fn a_cold_paged_reader_fault_degrades_to_counted_gathers() {
    let dims = (2, 1, 64);
    let dtype = Dtype::Float32;
    let mut cache = PagedPackedKvCache::with_pool(pool(1, 64, 32), interrupting(0, false)).unwrap();
    let mut healthy = PagedPackedKvCache::with_pool(pool(1, 64, 32), k8v8_reader()).unwrap();
    for (i, step) in [3, 1, 1, 40].into_iter().enumerate() {
        let a = attend_step(&mut cache, i as u64, step, dims, dtype);
        let b = attend_step(&mut healthy, i as u64, step, dims, dtype);
        // Same values either way; only the attention arithmetic differs. MLX's vector SDPA
        // (up to 8 query rows) agrees with the fused reader to f32 rounding; its multi-row
        // steel kernel differs from the fp32 tiled reader (itself within 1e-4 of the fp32
        // oracle) by up to ~1e-3.
        let bound = if step <= SDPA_MAX_FUSED_QLEN as usize {
            1e-5
        } else {
            2e-3
        };
        let worst = max_abs(&a, &b);
        assert!(
            worst <= bound,
            "step {i}: gathered vs fused paged attention {worst}"
        );
    }
    assert_eq!(cache.fused_calls(), 0);
    assert_eq!(
        cache.dense_gathers(),
        vec![(PagedFallbackReason::ReaderUnavailable, 4 * LAYERS as u64)]
    );
    assert!(cache.handle.is_none());
    assert_eq!(cache.offset(), 45);
}

/// sc-20688 review (finding 8): a sequence whose reader faulted is never stored in the
/// long-lived prefix store, so a later hit never inherits the faulted reader and its dense
/// gathers; the healthy sequence's prefix is stored and a hit on it reads its pages fused.
#[test]
fn a_prefix_store_never_keeps_a_sequence_whose_reader_faulted() {
    let dims = (2, 1, 64);
    let dtype = Dtype::Float32;
    let shared = pool(1, 64, 32);
    let model = key("faulted-prefix");
    let mut store = crate::decode::PagedPrefixCache::new(shared.clone(), model.clone(), 4);
    let identity = store.identity().clone();
    let tokens = (0..71).collect::<Vec<i32>>();
    // A prompt step, then a decode step whose cold paged dispatch faults.
    let mut faulted =
        PagedPackedKvCache::with_pool(shared.clone(), interrupting(0, false)).unwrap();
    attend_step(&mut faulted, 0, 70, dims, dtype);
    attend_step(&mut faulted, 1, 1, dims, dtype);
    assert!(faulted.reader_faulted());
    assert!(!store.insert(&identity, &tokens, &faulted).unwrap());
    assert!(store.is_empty(), "the faulted sequence was not stored");
    assert!(matches!(
        store.lookup(&identity, &tokens).unwrap(),
        crate::decode::PagedPrefixLookup::Miss
    ));

    let mut healthy = PagedPackedKvCache::with_pool(shared.clone(), k8v8_reader()).unwrap();
    attend_step(&mut healthy, 0, 70, dims, dtype);
    attend_step(&mut healthy, 1, 1, dims, dtype);
    assert!(!healthy.reader_faulted());
    assert!(store.insert(&identity, &tokens, &healthy).unwrap());
    let crate::decode::PagedPrefixLookup::Hit {
        mut cache,
        tokens: held,
    } = store.lookup(&identity, &tokens).unwrap()
    else {
        panic!("the healthy prefix is stored");
    };
    assert_eq!(held, 70, "the stored prefix, up to the prompt's last token");
    assert!(!cache.reader_faulted());
    attend_step(cache.as_mut(), 2, 1, dims, dtype);
    assert_eq!(
        cache.fused_calls(),
        LAYERS as u64,
        "the hit reads its pages fused"
    );
    assert!(cache.dense_gathers().is_empty());
}

/// Synthetic decode timing (measurement, not a gate): one decode token's attention over a
/// long history on the fused paged reader, the fused contiguous reader (same codes), and the
/// dense paged cache (gather + SDPA). Hq = 32, Hkv = 8, D = 128, bf16, 64-token pages.
#[test]
#[ignore = "GPU micro-benchmark; run explicitly with --ignored --nocapture"]
fn paged_decode_timing_versus_contiguous_and_dense_paged() {
    let dims = (32, 8, 128);
    let dtype = Dtype::Bfloat16;
    let scale = (128f32).powf(-0.5);
    let median = |mut samples: Vec<f64>| {
        samples.sort_by(f64::total_cmp);
        samples[samples.len() / 2]
    };
    for context in [4096, 16384, 32768] {
        let handle = k8v8_reader();
        let pages = PackedPagePool::new(1, 8, 128, 64, PackedCodeBits::Eight).unwrap();
        let mut paged = PagedPackedKvCache::with_pool(pages, handle.clone()).unwrap();
        let mut contiguous = contiguous_cache(&handle, 1, 8, 128);
        let mut dense = PagedKvCache::new(1, 16);
        let chunk = 2048;
        for start in (0..context).step_by(chunk) {
            let (q, k, v) = qkv(start as u64, chunk, dims, dtype);
            for cache in [&mut paged as &mut dyn KvCache, contiguous.as_mut()] {
                cache
                    .try_packed_attention(0, &q, &k, &v, PackedAttentionMask::Causal, scale, false)
                    .unwrap()
                    .expect("prefill chunk accepted")
                    .eval()
                    .unwrap();
            }
            let (kk, vv) = dense.update(0, &k, &v).unwrap();
            mlx_rs::transforms::eval([&kk, &vv]).unwrap();
        }
        let (mut p, mut c, mut d) = (Vec::new(), Vec::new(), Vec::new());
        for i in 0..45 {
            let (q, k, v) = qkv(i, 1, dims, dtype);
            let timed = |run: &mut dyn FnMut() -> Array, into: &mut Vec<f64>| {
                let started = std::time::Instant::now();
                run().eval().unwrap();
                into.push(started.elapsed().as_secs_f64() * 1e3);
            };
            timed(
                &mut || {
                    paged
                        .try_packed_attention(
                            0,
                            &q,
                            &k,
                            &v,
                            PackedAttentionMask::Causal,
                            scale,
                            false,
                        )
                        .unwrap()
                        .unwrap()
                },
                &mut p,
            );
            timed(
                &mut || {
                    contiguous
                        .try_packed_attention(
                            0,
                            &q,
                            &k,
                            &v,
                            PackedAttentionMask::Causal,
                            scale,
                            false,
                        )
                        .unwrap()
                        .unwrap()
                },
                &mut c,
            );
            timed(
                &mut || {
                    let (kk, vv) = dense.update(0, &k, &v).unwrap();
                    sdpa(&q, &kk, &vv, scale, AttnMask::Causal).unwrap()
                },
                &mut d,
            );
        }
        eprintln!(
            "decode attention @ {context} tokens (median of 40 after 5 warm-up, ms): \
                 fused-paged {:.3}, fused-contiguous {:.3}, dense-paged {:.3}",
            median(p[5..].to_vec()),
            median(c[5..].to_vec()),
            median(d[5..].to_vec())
        );
    }
}

// ---- sc-20681: prefix sharing, copy-on-write, exact speculative rollback, restore, batching.

/// A fresh cache on its own pool fed `steps` (`(seed, tokens)`), with each step's outputs.
fn alone(steps: &[(u64, usize)], dims: (usize, usize, usize), dtype: Dtype) -> Run {
    let mut cache = PagedPackedKvCache::with_pool(pool(dims.1, dims.2, 64), k8v8_reader()).unwrap();
    let outputs = steps
        .iter()
        .map(|&(seed, step)| attend_step(&mut cache, seed, step, dims, dtype))
        .collect();
    (cache, outputs)
}

type Run = (PagedPackedKvCache, Vec<Vec<Vec<f32>>>);

/// A fork shares the source's pages by reference (no copy); a fork inside a quantized group
/// keeps whole groups only. When both write into the shared, partially filled page, the first
/// writer copies it (the last holder then writes in place), and each sequence's outputs and
/// independently dequantized K/V equal its own history appended to an empty cache: neither
/// sees the other's writes. The untouched full page stays shared; dropping both frees all.
#[test]
fn a_forked_prefix_shares_pages_and_copy_on_write_keeps_sequences_apart() {
    let dims = (2, 1, 64);
    let dtype = Dtype::Float32;
    let shared = pool(1, 64, 64);
    let mut source = PagedPackedKvCache::with_pool(shared.clone(), k8v8_reader()).unwrap();
    attend_step(&mut source, 0, 100, dims, dtype);
    assert_eq!(
        source.page_ids().len(),
        2,
        "96 quantized tokens: one full, one half page"
    );
    let (mut fork, held) = source.fork_prefix(100).unwrap();
    assert_eq!((held, fork.offset()), (100, 100));
    assert_eq!(fork.page_ids(), source.page_ids());
    for &id in source.page_ids() {
        assert_eq!(shared.borrow().refcount(id), 2);
    }
    assert_eq!(shared.borrow().live_pages(), 2, "sharing allocated nothing");
    assert_eq!(shared.borrow().shared_pages(), 2);
    let (inner, inner_held) = source.fork_prefix(70).unwrap();
    assert_eq!(inner_held, 64, "a cut quantized group is not shared");
    assert_eq!(inner.page_ids(), &source.page_ids()[..1]);
    assert_eq!(shared.borrow().refcount(source.page_ids()[0]), 3);
    drop(inner);
    assert_eq!(shared.borrow().refcount(source.page_ids()[0]), 2);

    let half_page = source.page_ids()[1];
    let source_out = (11..14)
        .map(|seed| attend_step(&mut source, seed, 20, dims, dtype))
        .collect::<Vec<_>>();
    assert_ne!(
        source.page_ids()[1],
        half_page,
        "the source copied the page it wrote"
    );
    assert_eq!(
        shared.borrow().refcount(half_page),
        1,
        "only the fork holds it now"
    );
    let fork_out = (21..24)
        .map(|seed| attend_step(&mut fork, seed, 20, dims, dtype))
        .collect::<Vec<_>>();
    assert_eq!(
        fork.page_ids()[1],
        half_page,
        "the last holder writes in place"
    );
    assert_eq!(
        source.page_ids()[0],
        fork.page_ids()[0],
        "the full page stays shared"
    );
    assert_eq!(shared.borrow().refcount(source.page_ids()[0]), 2);

    let (source_ref, source_ref_out) =
        alone(&[(0, 100), (11, 20), (12, 20), (13, 20)], dims, dtype);
    let (fork_ref, fork_ref_out) = alone(&[(0, 100), (21, 20), (22, 20), (23, 20)], dims, dtype);
    assert_eq!(source_out, source_ref_out[1..]);
    assert_eq!(fork_out, fork_ref_out[1..]);
    for layer in 0..LAYERS {
        assert_eq!(oracle_kv(&source, layer), oracle_kv(&source_ref, layer));
        assert_eq!(oracle_kv(&fork, layer), oracle_kv(&fork_ref, layer));
    }
    drop(source);
    drop(fork);
    assert_eq!(shared.borrow().live_pages(), 0);
}

/// A step that copied a shared page and then faults returns the copy and restores the shared
/// page (refcount and page table); the retried step equals the uninterrupted history, and the
/// source never sees the fork's attempt.
#[test]
fn a_rolled_back_step_returns_its_copy_and_restores_the_shared_page() {
    let dims = (2, 1, 64);
    let dtype = Dtype::Float32;
    let scale = 0.125;
    let shared = pool(1, 64, 64);
    // Dispatches 0 and 1 are the source's decode step; 2 is the fork's layer 0; 3 faults.
    let mut source = PagedPackedKvCache::with_pool(shared.clone(), interrupting(3, false)).unwrap();
    attend_step(&mut source, 0, 100, dims, dtype);
    attend_step(&mut source, 1, 1, dims, dtype);
    let (mut fork, _) = source.fork_prefix(101).unwrap();
    let before = fork.page_ids().to_vec();
    let live = shared.borrow().live_pages();
    let (q0, k0, v0) = qkv(7 * 31, 28, dims, dtype);
    let (q1, k1, v1) = qkv(7 * 31 + 1, 28, dims, dtype);
    assert!(fork
        .try_packed_attention(0, &q0, &k0, &v0, PackedAttentionMask::Causal, scale, false)
        .unwrap()
        .is_some());
    assert_ne!(
        fork.page_ids()[1],
        before[1],
        "layer 0 copied the shared page"
    );
    assert_eq!(shared.borrow().live_pages(), live + 1);
    fork.try_packed_attention(1, &q1, &k1, &v1, PackedAttentionMask::Causal, scale, false)
        .unwrap_err();
    assert_eq!(
        fork.page_ids(),
        before.as_slice(),
        "the shared page is back"
    );
    assert_eq!(shared.borrow().live_pages(), live, "the copy went back");
    assert_eq!(shared.borrow().refcount(before[1]), 2);
    assert_eq!(fork.offset(), 101);
    let retried = attend_step(&mut fork, 7, 28, dims, dtype);
    let (fork_ref, fork_ref_out) = alone(&[(0, 100), (1, 1), (7, 28)], dims, dtype);
    assert_eq!(retried, fork_ref_out[2]);
    let (source_ref, _) = alone(&[(0, 100), (1, 1)], dims, dtype);
    for layer in 0..LAYERS {
        assert_eq!(oracle_kv(&fork, layer), oracle_kv(&fork_ref, layer));
        assert_eq!(oracle_kv(&source, layer), oracle_kv(&source_ref, layer));
    }
}

/// Append each layer's rows `rows` of one synthetic verify step (seed `seed`, `step` rows).
fn attend_rows(
    cache: &mut PagedPackedKvCache,
    seed: u64,
    step: usize,
    rows: usize,
    dims: (usize, usize, usize),
    dtype: Dtype,
) {
    for layer in 0..LAYERS {
        let (q, k, v) = qkv(seed * 31 + layer as u64, step, dims, dtype);
        let cut = |a: &Array| rows_range(a, 0, rows).unwrap();
        cache
            .try_packed_attention(
                layer,
                &cut(&q),
                &cut(&k),
                &cut(&v),
                PackedAttentionMask::Causal,
                (dims.2 as f32).powf(-0.5),
                false,
            )
            .unwrap()
            .unwrap();
    }
}

/// Speculative rollback with the window armed is byte- and position-exact: after a 20-row
/// verify (which completes and quantizes a group) is cut back to `kept` rows, the cache's
/// codes, scale/zero metadata, residual rows, offset and pages equal a cache that appended
/// only the kept rows, and both continue identically. Without the window the same cut
/// re-stages the group from its codes, which is not exact (the control).
#[test]
fn speculative_rollback_is_byte_and_position_exact() {
    let dims = (2, 1, 64);
    for dtype in [Dtype::Float32, Dtype::Bfloat16] {
        for kept in [1, 9, 14, 19] {
            let prefill = |cache: &mut PagedPackedKvCache| {
                attend_step(cache, 0, 50, dims, dtype);
            };
            let shared = pool(1, 64, 32);
            let mut speculated =
                PagedPackedKvCache::with_pool(shared.clone(), k8v8_reader()).unwrap();
            prefill(&mut speculated);
            speculated.begin_speculation().unwrap();
            attend_rows(&mut speculated, 9, 20, 20, dims, dtype);
            assert_eq!(
                shared.borrow().live_pages(),
                2,
                "the verify completed a group"
            );
            speculated.truncate((50 + kept) as i32).unwrap();

            let reference_pool = pool(1, 64, 32);
            let mut reference =
                PagedPackedKvCache::with_pool(reference_pool.clone(), k8v8_reader()).unwrap();
            prefill(&mut reference);
            attend_rows(&mut reference, 9, 20, kept, dims, dtype);

            assert_eq!(speculated.offset(), reference.offset());
            assert_eq!(
                speculated.snapshot(&key("m")).unwrap(),
                reference.snapshot(&key("m")).unwrap(),
                "{dtype:?} kept {kept}"
            );
            assert_eq!(
                shared.borrow().live_pages(),
                reference_pool.borrow().live_pages()
            );
            for (seed, step) in [(5, 1), (6, 33), (7, 1)] {
                assert_eq!(
                    attend_step(&mut speculated, seed, step, dims, dtype),
                    attend_step(&mut reference, seed, step, dims, dtype)
                );
            }

            if 50 + kept < 64 {
                let mut control =
                    PagedPackedKvCache::with_pool(pool(1, 64, 32), k8v8_reader()).unwrap();
                prefill(&mut control);
                attend_rows(&mut control, 9, 20, 20, dims, dtype);
                control.truncate((50 + kept) as i32).unwrap();
                let mut fresh_reference =
                    PagedPackedKvCache::with_pool(pool(1, 64, 32), k8v8_reader()).unwrap();
                prefill(&mut fresh_reference);
                attend_rows(&mut fresh_reference, 9, 20, kept, dims, dtype);
                assert_ne!(
                    control.snapshot(&key("m")).unwrap(),
                    fresh_reference.snapshot(&key("m")).unwrap(),
                    "{dtype:?} kept {kept}: the unarmed cut must be visibly inexact"
                );
            }
        }
    }
}

/// A snapshot round-trips through bytes and restores into another pool exactly (same live
/// bytes, same continued outputs). A restore under any other identity — model, KV format
/// version, page layout version, page geometry, code width — is refused before allocating, and
/// a corrupted or foreign byte string is refused when parsed.
#[test]
fn a_snapshot_restores_exactly_across_bytes_and_refuses_any_identity_mismatch() {
    let dims = (2, 1, 64);
    let dtype = Dtype::Bfloat16;
    let handle = k8v8_reader();
    let mut original = PagedPackedKvCache::with_pool(pool(1, 64, 32), handle.clone()).unwrap();
    for (i, step) in [100, 1, 1].into_iter().enumerate() {
        attend_step(&mut original, i as u64, step, dims, dtype);
    }
    let snapshot = original.snapshot(&key("model-a")).unwrap();
    assert_eq!(snapshot.tokens(), 102);
    let bytes = snapshot.to_bytes().unwrap();
    let parsed = PagedCacheSnapshot::from_bytes(&bytes).unwrap();
    assert_eq!(parsed, snapshot);
    let target = pool(1, 64, 32);
    let mut restored =
        PagedPackedKvCache::restore(&parsed, target.clone(), handle.clone(), &key("model-a"))
            .unwrap();
    assert_eq!(restored.offset(), 102);
    assert_eq!(restored.snapshot(&key("model-a")).unwrap(), snapshot);
    for layer in 0..LAYERS {
        assert_eq!(oracle_kv(&restored, layer), oracle_kv(&original, layer));
    }
    for (i, step) in [1, 40, 1].into_iter().enumerate() {
        let seed = 50 + i as u64;
        assert_eq!(
            attend_step(&mut restored, seed, step, dims, dtype),
            attend_step(&mut original, seed, step, dims, dtype)
        );
    }

    let fresh = pool(1, 64, 32);
    let refused = |snapshot: &PagedCacheSnapshot,
                   pool: Rc<RefCell<PackedPagePool>>,
                   reader: CompiledKernelHandle,
                   model: &PagedModelKey,
                   field: &str| {
        let error = PagedPackedKvCache::restore(snapshot, pool.clone(), reader, model)
            .unwrap_err()
            .to_string();
        assert!(error.contains(field), "{field}: {error}");
        assert_eq!(pool.borrow().live_pages(), 0, "{field}: nothing allocated");
    };
    refused(
        &snapshot,
        fresh.clone(),
        handle.clone(),
        &key("model-b"),
        "model",
    );
    refused(
        &snapshot,
        fresh.clone(),
        handle.clone(),
        &PagedModelKey::new("model-a", "another decoder"),
        "model fingerprint",
    );
    // An empty identity matches nothing, not even the same empty identity.
    for unnamed in [key(""), PagedModelKey::new("model-a", "")] {
        let mut anonymous = snapshot.clone();
        anonymous.identity = fresh.borrow().identity(&unnamed);
        refused(
            &anonymous,
            fresh.clone(),
            handle.clone(),
            &unnamed,
            "no model identity",
        );
    }
    let mut format = snapshot.clone();
    format.identity.format_version += 1;
    refused(
        &format,
        fresh.clone(),
        handle.clone(),
        &key("model-a"),
        "KV format version",
    );
    let mut layout = snapshot.clone();
    layout.identity.layout_version += 1;
    refused(
        &layout,
        fresh.clone(),
        handle.clone(),
        &key("model-a"),
        "page layout version",
    );
    refused(
        &snapshot,
        pool(1, 64, 64),
        handle.clone(),
        &key("model-a"),
        "page tokens",
    );
    let four_bit = PackedPagePool::new(LAYERS, 1, 64, 32, PackedCodeBits::Four).unwrap();
    refused(
        &snapshot,
        four_bit,
        reader(PackedCodeBits::Four, PackedMetalGpuFamily::Apple7OrNewer),
        &key("model-a"),
        "representation",
    );
    // A version-mismatched snapshot is still refused after a trip through bytes.
    let reread = PagedCacheSnapshot::from_bytes(&format.to_bytes().unwrap()).unwrap();
    refused(
        &reread,
        fresh.clone(),
        handle.clone(),
        &key("model-a"),
        "KV format version",
    );

    let mut corrupt = bytes.clone();
    let last = corrupt.len() - 1;
    corrupt[last] ^= 1;
    let error = PagedCacheSnapshot::from_bytes(&corrupt)
        .unwrap_err()
        .to_string();
    assert!(error.contains("digest"), "{error}");
    assert!(PagedCacheSnapshot::from_bytes(b"not a paged KV snapshot").is_err());
    assert!(PagedCacheSnapshot::from_bytes(&bytes[..bytes.len() / 2]).is_err());
    // The digest covers the header: a dtype label flipped in place (same length, still valid
    // JSON) is refused, not restored with the wrong residual dtype.
    let mut relabelled = bytes.clone();
    let label = br#""Bfloat16""#;
    let at = relabelled
        .windows(label.len())
        .position(|window| window == label)
        .expect("the header names the residual dtype");
    relabelled[at..at + label.len()].copy_from_slice(br#""Float16" "#);
    let error = PagedCacheSnapshot::from_bytes(&relabelled)
        .unwrap_err()
        .to_string();
    assert!(error.contains("digest"), "{error}");

    // Crafted extents are refused before any arithmetic or allocation: a residual of more
    // than a group (formerly an overflowing sum), quantized tokens off the group grid, layers
    // disagreeing on their quantized extent, a layer holding positions without dtypes.
    type Defect = fn(&mut PagedCacheSnapshot);
    let defects: [(&str, Defect); 5] = [
        ("overflowing residual", |s| {
            s.layers[0].tail_rows = usize::MAX
        }),
        ("oversized residual", |s| {
            s.layers[0].packed -= 32;
            s.layers[0].tail_rows += 32;
        }),
        ("off-grid quantized extent", |s| {
            s.layers[1].packed += 1;
            s.layers[1].tail_rows -= 1;
        }),
        ("disagreeing layers", |s| s.layers[1].packed -= 32),
        ("missing dtypes", |s| s.layers[0].dtypes = None),
    ];
    // A layer of whole groups and no residual must still name its dtypes.
    let mut whole = PagedPackedKvCache::with_pool(pool(1, 64, 32), handle.clone()).unwrap();
    attend_step(&mut whole, 0, 96, dims, dtype);
    let mut untyped = whole.snapshot(&key("model-a")).unwrap();
    assert_eq!(untyped.layers[1].tail_rows, 0);
    untyped.layers[1].dtypes = None;
    let target = pool(1, 64, 32);
    let error =
        PagedPackedKvCache::restore(&untyped, target.clone(), handle.clone(), &key("model-a"))
            .map(|_| ())
            .unwrap_err()
            .to_string();
    assert!(error.contains("refused"), "untyped whole groups: {error}");
    assert_eq!(target.borrow().live_pages(), 0);
    for (name, defect) in defects {
        let mut crafted = snapshot.clone();
        defect(&mut crafted);
        // The digest is not a secret: a crafted header re-serialized with a matching digest
        // reads back, so restore itself must refuse it (a typed error, never a panic).
        let crafted = PagedCacheSnapshot::from_bytes(&crafted.to_bytes().unwrap()).unwrap();
        let target = pool(1, 64, 32);
        let result =
            PagedPackedKvCache::restore(&crafted, target.clone(), handle.clone(), &key("model-a"));
        let error = result.map(|_| ()).unwrap_err().to_string();
        assert!(error.contains("refused"), "{name}: {error}");
        assert_eq!(target.borrow().live_pages(), 0, "{name}: nothing allocated");
    }
}

/// Three jagged sequences on one pool decode through one batched paged dispatch per layer: each
/// row equals that sequence decoded alone through the single-sequence fused path, every call is
/// fused and batched, and nothing is gathered. A call outside the batched reader (an additive
/// mask, a sequence of another pool, another scale) is declined with nothing mutated.
#[test]
fn a_batched_paged_dispatch_serves_jagged_sequences_like_each_alone() {
    let dims = (4, 2, 64);
    let scale = (64f32).powf(-0.5);
    for dtype in [Dtype::Float32, Dtype::Bfloat16] {
        let handle = k8v8_reader();
        let shared = pool(2, 64, 32);
        let lengths = [40, 75, 130];
        let mut batch =
            lengths.map(|_| PagedPackedKvCache::with_pool(shared.clone(), handle.clone()).unwrap());
        let mut single = lengths
            .map(|_| PagedPackedKvCache::with_pool(pool(2, 64, 32), handle.clone()).unwrap());
        for (b, &len) in lengths.iter().enumerate() {
            for cache in [&mut batch[b], &mut single[b]] {
                attend_step(cache, b as u64, len, dims, dtype);
                attend_step(cache, 10 + b as u64, 1, dims, dtype);
            }
        }
        let mut worst = 0.0f32;
        for step in 0..40u64 {
            for layer in 0..LAYERS {
                let rows = (0..3u64)
                    .map(|b| qkv(1000 * step + 10 * b + layer as u64, 1, dims, dtype))
                    .collect::<Vec<_>>();
                let stack = |pick: fn(&(Array, Array, Array)) -> &Array| {
                    concatenate_axis(&rows.iter().map(pick).collect::<Vec<_>>(), 0).unwrap()
                };
                let (q, k, v) = (stack(|r| &r.0), stack(|r| &r.1), stack(|r| &r.2));
                let mut members = batch.iter_mut().collect::<Vec<_>>();
                let out = paged_attention_batch(
                    &mut members,
                    layer,
                    &q,
                    &k,
                    &v,
                    PackedAttentionMask::Causal,
                    scale,
                )
                .unwrap()
                .expect("one pool, warmed reader, causal: batched");
                for (b, (qb, kb, vb)) in rows.iter().enumerate() {
                    let alone = single[b]
                        .try_packed_attention(
                            layer,
                            qb,
                            kb,
                            vb,
                            PackedAttentionMask::Causal,
                            scale,
                            false,
                        )
                        .unwrap()
                        .unwrap();
                    let row = out.try_index((b as i32..b as i32 + 1, .., .., ..)).unwrap();
                    worst = worst.max(
                        host_f32(&row)
                            .iter()
                            .zip(host_f32(&alone))
                            .map(|(a, b)| (a - b).abs())
                            .fold(0.0, f32::max),
                    );
                }
            }
        }
        eprintln!("{dtype:?}: batched vs alone worst |Δ| {worst}");
        assert_eq!(
            worst, 0.0,
            "{dtype:?}: a batched row differs from its sequence alone"
        );
        for (b, cache) in batch.iter().enumerate() {
            assert_eq!(cache.offset() as usize, lengths[b] + 41);
            assert_eq!(cache.batched_calls(), 2 * 40);
            assert!(cache.dense_gathers().is_empty());
        }

        let offsets = batch.iter().map(|cache| cache.offset()).collect::<Vec<_>>();
        let (q, k, v) = qkv(5, 1, (12, 6, 64), dtype);
        let (q, k, v) = (
            q.reshape(&[3, 4, 1, 64]).unwrap(),
            k.reshape(&[3, 2, 1, 64]).unwrap(),
            v.reshape(&[3, 2, 1, 64]).unwrap(),
        );
        let decline = |members: &mut [&mut PagedPackedKvCache], mask, scale| {
            assert!(paged_attention_batch(members, 0, &q, &k, &v, mask, scale)
                .unwrap()
                .is_none());
        };
        decline(
            &mut batch.iter_mut().collect::<Vec<_>>(),
            PackedAttentionMask::Additive,
            scale,
        );
        decline(
            &mut batch.iter_mut().collect::<Vec<_>>(),
            PackedAttentionMask::Causal,
            0.5,
        );
        let [first, second, _] = &mut batch;
        decline(
            &mut [first, second, &mut single[2]],
            PackedAttentionMask::Causal,
            scale,
        );
        for (cache, offset) in batch.iter().zip(offsets) {
            assert_eq!(cache.offset(), offset);
            assert!(cache.pending.is_none());
        }
    }
}

/// Bounded growth (sc-20681): a sequence that grows its pool page by page, with no
/// reservation, never leaves the pool holding more than its live pages plus one growth chunk,
/// and once the pool is past its first few chunks the bytes it holds stay at or under 0.60 of
/// the dense bf16 K/V of the same positions (doubling would hold up to twice the live pages).
#[test]
fn unreserved_growth_stays_within_one_chunk_of_the_live_pages() {
    let dims = (2, 1, 64);
    let dtype = Dtype::Float32;
    let shared = pool(1, 64, 32);
    let mut cache = PagedPackedKvCache::with_pool(shared.clone(), k8v8_reader()).unwrap();
    // Dense bf16 K/V per position: LAYERS × K and V × 1 KV head × 64 dims × 2 bytes.
    let dense_per_position = (LAYERS * 2 * 64 * 2) as u64;
    let mut ratio_checks = 0;
    for step in 0..260u64 {
        attend_step(&mut cache, step, 32, dims, dtype);
        let pool = shared.borrow();
        let (capacity, live) = (pool.capacity_pages(), pool.live_pages());
        assert!(
            capacity <= live + pool.growth_chunk(),
            "step {step}: {capacity} pages held for {live} live"
        );
        if live >= 128 {
            let storage = pool.storage();
            let held = (storage.code_bytes + storage.metadata_bytes) as f64;
            let dense = (cache.offset() as u64 * dense_per_position) as f64;
            assert!(
                held / dense <= 0.60,
                "step {step}: pool holds {:.3} of dense bf16",
                held / dense
            );
            ratio_checks += 1;
        }
    }
    assert!(ratio_checks > 100);
}

/// Lowest-id-first allocation and trim: after the first and last of three sequences go, a
/// trim returns the capacity above the middle one, and a new sequence takes the lowest freed
/// pages instead of growing the pool again. The kept sequence's pages survive the trim's copy
/// (its continued outputs equal it alone), and an emptied pool trims to nothing.
#[test]
fn a_trimmed_pool_reuses_its_lowest_pages_without_growing() {
    let dims = (2, 1, 64);
    let dtype = Dtype::Float32;
    let handle = k8v8_reader();
    let shared = pool(1, 64, 32);
    let mut sequences = (0..3)
        .map(|_| PagedPackedKvCache::with_pool(shared.clone(), handle.clone()).unwrap())
        .collect::<Vec<_>>();
    for (i, sequence) in sequences.iter_mut().enumerate() {
        attend_step(sequence, i as u64, 64, dims, dtype);
    }
    assert_eq!(
        sequences
            .iter()
            .map(|s| s.page_ids().to_vec())
            .collect::<Vec<_>>(),
        vec![vec![0, 1], vec![2, 3], vec![4, 5]]
    );
    let mut middle = sequences.remove(1);
    drop(sequences);
    shared.borrow_mut().trim().unwrap();
    assert_eq!(shared.borrow().capacity_pages(), 4, "pages 4.. went back");
    let mut late = PagedPackedKvCache::with_pool(shared.clone(), handle.clone()).unwrap();
    attend_step(&mut late, 9, 64, dims, dtype);
    assert_eq!(late.page_ids(), &[0, 1], "the lowest freed pages");
    assert_eq!(shared.borrow().capacity_pages(), 4, "no growth");
    let (_, middle_alone) = alone(&[(1, 64), (5, 1), (6, 40)], dims, dtype);
    assert_eq!(attend_step(&mut middle, 5, 1, dims, dtype), middle_alone[1]);
    assert_eq!(
        attend_step(&mut middle, 6, 40, dims, dtype),
        middle_alone[2]
    );
    drop((middle, late));
    shared.borrow_mut().trim().unwrap();
    assert_eq!(shared.borrow().capacity_pages(), 0);
    assert_eq!(shared.borrow().storage().code_bytes, 0);
}

/// Copy-on-write inside one batched dispatch: two forks of one 100-token source (sharing its
/// half-full second page with it) step together through batched paged dispatches; the step
/// that completes their next group has both write into the shared page in the same call, so
/// each takes a private copy. Every row equals that fork stepped alone, each fork's
/// independently dequantized K/V equals its alone reference, the source never sees either,
/// the shared page ends held by the source alone, and dropping everything frees every page.
#[test]
fn copy_on_write_inside_one_batched_dispatch_keeps_forks_apart() {
    let dims = (4, 2, 64);
    let dtype = Dtype::Float32;
    let scale = (64f32).powf(-0.5);
    let handle = k8v8_reader();
    let shared = pool(2, 64, 64);
    let mut source = PagedPackedKvCache::with_pool(shared.clone(), handle.clone()).unwrap();
    attend_step(&mut source, 0, 100, dims, dtype);
    // One single-sequence decode step warms the paged reader (batching needs it warm).
    attend_step(&mut source, 1, 1, dims, dtype);
    let (mut first, _) = source.fork_prefix(101).unwrap();
    let (mut second, _) = source.fork_prefix(101).unwrap();
    let half_page = source.page_ids()[1];
    assert_eq!(shared.borrow().refcount(half_page), 3);
    let mut alone_refs = [
        alone(&[(0, 100), (1, 1)], dims, dtype).0,
        alone(&[(0, 100), (1, 1)], dims, dtype).0,
    ];
    for step in 0..40u64 {
        for layer in 0..LAYERS {
            let rows = (0..2u64)
                .map(|b| qkv(7_000 + 100 * step + 10 * b + layer as u64, 1, dims, dtype))
                .collect::<Vec<_>>();
            let stack = |pick: fn(&(Array, Array, Array)) -> &Array| {
                concatenate_axis(&rows.iter().map(pick).collect::<Vec<_>>(), 0).unwrap()
            };
            let (q, k, v) = (stack(|r| &r.0), stack(|r| &r.1), stack(|r| &r.2));
            let out = paged_attention_batch(
                &mut [&mut first, &mut second],
                layer,
                &q,
                &k,
                &v,
                PackedAttentionMask::Causal,
                scale,
            )
            .unwrap()
            .expect("both forks on one pool: batched");
            for (b, (qb, kb, vb)) in rows.iter().enumerate() {
                let expected = alone_refs[b]
                    .try_packed_attention(
                        layer,
                        qb,
                        kb,
                        vb,
                        PackedAttentionMask::Causal,
                        scale,
                        false,
                    )
                    .unwrap()
                    .unwrap();
                let row = out.try_index((b as i32..b as i32 + 1, .., .., ..)).unwrap();
                assert_eq!(host_f32(&row), host_f32(&expected), "step {step} fork {b}");
            }
        }
    }
    assert_ne!(
        first.page_ids()[1],
        half_page,
        "the first fork copied the page"
    );
    assert_ne!(
        second.page_ids()[1],
        half_page,
        "the second fork copied the page"
    );
    assert_ne!(first.page_ids()[1], second.page_ids()[1]);
    assert_eq!(
        shared.borrow().refcount(half_page),
        1,
        "held by the source alone"
    );
    assert_eq!(
        first.page_ids()[0],
        source.page_ids()[0],
        "the full page stays shared"
    );
    for layer in 0..LAYERS {
        assert_eq!(oracle_kv(&first, layer), oracle_kv(&alone_refs[0], layer));
        assert_eq!(oracle_kv(&second, layer), oracle_kv(&alone_refs[1], layer));
    }
    let (source_ref, _) = alone(&[(0, 100), (1, 1)], dims, dtype);
    for layer in 0..LAYERS {
        assert_eq!(oracle_kv(&source, layer), oracle_kv(&source_ref, layer));
    }
    drop((first, second, source));
    assert_eq!(shared.borrow().live_pages(), 0);
}

/// A batched dispatch that faults (here at its second layer, after the step allocated a
/// page) rolls back every sequence's step: offsets, residuals, pages and accepted calls; the
/// retried step then succeeds.
#[test]
fn a_faulted_batched_dispatch_rolls_back_every_sequence() {
    let dims = (4, 2, 64);
    let dtype = Dtype::Float32;
    let scale = (64f32).powf(-0.5);
    let shared = pool(2, 64, 32);
    // Six single-sequence dispatches warm the reader; batched dispatch 6 succeeds, 7 faults.
    let handle = interrupting(7, false);
    let mut batch = [40, 62, 95]
        .map(|_| PagedPackedKvCache::with_pool(shared.clone(), handle.clone()).unwrap());
    for (b, len) in [40, 62, 95].into_iter().enumerate() {
        attend_step(&mut batch[b], b as u64, len, dims, dtype);
        attend_step(&mut batch[b], 10 + b as u64, 1, dims, dtype);
    }
    let live = shared.borrow().live_pages();
    let before = batch
        .iter()
        .map(|cache| {
            (
                cache.offset(),
                cache.fused_calls(),
                cache.page_ids().to_vec(),
            )
        })
        .collect::<Vec<_>>();
    let layer_rows = |layer: u64| {
        let rows = (0..3u64)
            .map(|b| qkv(77 + 10 * b + layer, 1, dims, dtype))
            .collect::<Vec<_>>();
        let stack = |pick: fn(&(Array, Array, Array)) -> &Array| {
            concatenate_axis(&rows.iter().map(pick).collect::<Vec<_>>(), 0).unwrap()
        };
        (stack(|r| &r.0), stack(|r| &r.1), stack(|r| &r.2))
    };
    let attend = |batch: &mut [PagedPackedKvCache; 3], layer: usize| {
        let (q, k, v) = layer_rows(layer as u64);
        paged_attention_batch(
            &mut batch.iter_mut().collect::<Vec<_>>(),
            layer,
            &q,
            &k,
            &v,
            PackedAttentionMask::Causal,
            scale,
        )
    };
    assert!(attend(&mut batch, 0).unwrap().is_some());
    assert_eq!(
        shared.borrow().live_pages(),
        live + 1,
        "the 63rd → 64th token took a page"
    );
    attend(&mut batch, 1).unwrap_err();
    assert_eq!(
        shared.borrow().live_pages(),
        live,
        "the step's page went back"
    );
    for (cache, (offset, fused, pages)) in batch.iter().zip(&before) {
        assert_eq!(cache.offset(), *offset);
        assert_eq!(cache.fused_calls(), *fused);
        assert_eq!(cache.page_ids(), pages.as_slice());
        assert!(cache.pending.is_none());
    }
    for layer in 0..LAYERS {
        assert!(attend(&mut batch, layer).unwrap().is_some());
    }
    for (cache, (offset, _, _)) in batch.iter().zip(&before) {
        assert_eq!(cache.offset(), offset + 1);
    }
}
