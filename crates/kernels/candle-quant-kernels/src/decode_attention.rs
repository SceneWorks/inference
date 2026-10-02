//! Length-aware decode attention and device-indexed copies (sc-24441, epic sc-24432),
//! nvrtc-compiled from [`DECODE_ATTENTION_SRC`] through the [`nvrtc`](crate::nvrtc) seam.
//!
//! A CUDA graph records a decode step once and replays it at every later position, so nothing a
//! recorded kernel needs may be a host-side number that changes per step. These primitives read
//! the step's position from a **device** `u32` (`start[0]`, staged by the caller before the step)
//! instead:
//!
//! * [`decode_attention`] — attention of `M` new queries over a **static** (preallocated)
//!   `[B, Hkv, capacity, D]` K/V cache. Query `i` sits at position `start + i` and attends keys
//!   `lo..=start + i` (`lo = 0`, or the sliding window's start), with grouped-query head sharing,
//!   an optional score soft-cap and key/value widths that may differ (MLA). The grid is sized to
//!   the capacity; blocks past the visible range exit before touching memory, so the cost follows
//!   the cache's fill, not its capacity.
//! * [`write_rows_at`] — write a step's rows into a buffer at a device index (the KV write).
//! * [`read_slot`] — copy one slot of a slotted buffer at a device index (a checkpoint-ring read).
//!
//! **Determinism.** The attention result depends only on the operands and the fixed key chunking
//! ([`DECODE_ATTN_CHUNK`]): every reduction runs in an order fixed by the code, never by
//! scheduling (see `decode_attention.cu`). An eager launch and a graph replay at the same position
//! are therefore bit-identical by construction — which is why the static-KV decode path uses this
//! kernel whether or not a graph is captured.
//!
//! **CPU reference.** [`decode_attention_reference`] is the same chunked arithmetic in f32 on the
//! host (per-chunk max / exp / sum / value accumulation, then the ascending-chunk combine), so a
//! CPU build exercises the algorithm and a GPU test compares the kernel against it. On a CPU
//! device [`decode_attention`], [`write_rows_at`] and [`read_slot`] run the reference / host
//! copies, reading the index from the (host-resident) tensor.

use candle_core::{DType, Device, Tensor};

use crate::nvrtc::KernelSource;

/// The decode-attention kernel source; compiled once per device on first use.
pub const DECODE_ATTENTION_SRC: KernelSource = KernelSource {
    name: "candle_quant_kernels_decode_attention_v2",
    src: include_str!("decode_attention.cu"),
    // Builtins, warp shuffles and software bf16 conversions only.
    cc_floor: (7, 0),
};

/// Keys per chunk of the partial pass — the fixed chunking the reduction order depends on. Must
/// equal `DECODE_ATTN_CHUNK` in `decode_attention.cu`. Small (sc-24446): at decode shapes the
/// attention is latency-bound, and a smaller chunk puts more blocks in flight over a short cache.
pub const DECODE_ATTN_CHUNK: usize = 64;

/// Threads per block of both passes (`DECODE_ATTN_THREADS` in the source).
const DECODE_ATTN_THREADS: u32 = 256;

/// Widest key or value head the kernel serves (its per-block query row lives in shared memory).
pub const DECODE_ATTN_MAX_HEAD_DIM: usize = 1024;

/// Most query heads one partial block serves (`DECODE_ATTN_MAX_GROUP_TILE` in the source): the
/// query heads sharing a KV head are split into tiles of at most this many, and each tile reads
/// the KV head's chunk once. Bounded by the per-thread accumulators of the score pass (one per
/// head for each of the keys a warp has in flight).
pub const DECODE_ATTN_MAX_GROUP_TILE: usize = 8;

/// The shared memory a partial block may take without opting in to more (48 KiB, every device).
const DECODE_ATTN_SHARED_LIMIT: usize = 48 * 1024;

/// The value pass's key slices at value width `value_dim` (`value_slices` in the source): a
/// thread owns one pair of value elements, so `ceil(dv / 2)` threads cover a value row and the
/// block's other threads take further slices of the chunk's keys. The slice sums are added in a
/// fixed order, so this is part of the arithmetic — a function of the value width only.
fn value_slices(value_dim: usize) -> usize {
    let pairs = value_dim.div_ceil(2).max(1);
    let threads = DECODE_ATTN_THREADS as usize;
    if pairs >= threads {
        1
    } else {
        threads / pairs
    }
}

/// Dynamic shared memory of a partial block serving `tile` query heads: per head, the query row
/// (whose region the value pass then reuses for its per-slice sums) and the chunk's scores —
/// `tile·(max(dk, S·dv) + chunk)` f32.
fn partial_shared_bytes(tile: usize, key_dim: usize, value_dim: usize) -> usize {
    let region = key_dim.max(value_slices(value_dim) * value_dim);
    tile * (region + DECODE_ATTN_CHUNK) * 4
}

/// How many of the `groups` query heads sharing one KV head a partial block serves at key width
/// `key_dim` and value width `value_dim`: as many as fit [`DECODE_ATTN_MAX_GROUP_TILE`] and the
/// block's shared memory (see `partial_shared_bytes`). The result never depends on it (every
/// head's reductions run in one fixed order, see `decode_attention.cu`); only how often a K/V
/// chunk is read does. At least `1`.
pub fn group_tile(groups: usize, key_dim: usize, value_dim: usize) -> usize {
    let mut tile = groups.clamp(1, DECODE_ATTN_MAX_GROUP_TILE);
    while tile > 1 && partial_shared_bytes(tile, key_dim, value_dim) > DECODE_ATTN_SHARED_LIMIT {
        tile -= 1;
    }
    tile
}

/// The attention policy of one call: the score scale, Gemma-2's optional score soft-cap
/// (`c·tanh(s/c)`, applied after scaling) and an optional sliding window (a query sees at most
/// `window` keys, its own position included).
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct DecodeAttnSpec {
    /// Multiplier on `q·k` (usually `head_dim^-0.5`).
    pub scale: f32,
    /// Score soft-cap `c` (`None` ⇒ none).
    pub softcap: Option<f32>,
    /// Sliding window in keys (`None` ⇒ the whole causal prefix).
    pub window: Option<usize>,
}

/// The validated geometry of one [`decode_attention`] call.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct DecodeAttnPlan {
    /// `B`.
    pub batch: usize,
    /// Query heads `H`.
    pub heads: usize,
    /// Cached K/V heads `Hkv` (`H % Hkv == 0`).
    pub kv_heads: usize,
    /// New queries `M` per sequence.
    pub queries: usize,
    /// Cache capacity (positions).
    pub capacity: usize,
    /// Key (and query) head width.
    pub key_dim: usize,
    /// Value head width.
    pub value_dim: usize,
    /// Operand dtype (`F32` or `BF16`).
    pub dtype: DType,
    /// Chunks of [`DECODE_ATTN_CHUNK`] keys covering the capacity.
    pub chunks: usize,
}

fn msg(text: String) -> candle_core::Error {
    candle_core::Error::Msg(text)
}

/// Whether `dtype` is served ([`decode_attention`] takes `F32` or `BF16`).
pub fn served_dtype(dtype: DType) -> bool {
    matches!(dtype, DType::F32 | DType::BF16)
}

/// Validate `q: [B, H, M, dk]`, `k: [B, Hkv, cap, dk]`, `v: [B, Hkv, cap, dv]` and the `u32`
/// position tensor (its first element is the step start) for [`decode_attention`].
pub fn check_decode_attention(
    q: &Tensor,
    k: &Tensor,
    v: &Tensor,
    start: &Tensor,
) -> candle_core::Result<DecodeAttnPlan> {
    let (batch, heads, queries, key_dim) = q.dims4()?;
    let (kb, kv_heads, capacity, kd) = k.dims4()?;
    let (vb, vh, vcap, value_dim) = v.dims4()?;
    if kb != batch || vb != batch || vh != kv_heads || vcap != capacity || kd != key_dim {
        return Err(msg(format!(
            "decode_attention: q {:?}, k {:?}, v {:?} are not [B, H, M, dk] / [B, Hkv, cap, dk] / \
             [B, Hkv, cap, dv]",
            q.dims(),
            k.dims(),
            v.dims()
        )));
    }
    if kv_heads == 0 || heads % kv_heads != 0 || batch == 0 || queries == 0 || capacity == 0 {
        return Err(msg(format!(
            "decode_attention: {heads} query heads do not group over {kv_heads} KV heads (or an \
             axis is empty)"
        )));
    }
    if key_dim == 0 || value_dim == 0 {
        return Err(msg("decode_attention: empty head".into()));
    }
    if key_dim > DECODE_ATTN_MAX_HEAD_DIM || value_dim > DECODE_ATTN_MAX_HEAD_DIM {
        return Err(msg(format!(
            "decode_attention: head width {key_dim}/{value_dim} exceeds {DECODE_ATTN_MAX_HEAD_DIM}"
        )));
    }
    let dtype = q.dtype();
    if k.dtype() != dtype || v.dtype() != dtype || !served_dtype(dtype) {
        return Err(msg(format!(
            "decode_attention: operands must share one F32/BF16 dtype, got {:?}/{:?}/{:?}",
            dtype,
            k.dtype(),
            v.dtype()
        )));
    }
    if start.dtype() != DType::U32 || start.elem_count() == 0 {
        return Err(msg(
            "decode_attention: the step start must be a non-empty U32 tensor".into(),
        ));
    }
    let same = |t: &Tensor| t.device().same_device(q.device());
    if !(same(k) && same(v) && same(start)) {
        return Err(msg("decode_attention: operands on different devices".into()));
    }
    let chunks = capacity.div_ceil(DECODE_ATTN_CHUNK);
    let fits = |n: usize| i32::try_from(n).is_ok();
    if !(fits(capacity) && fits(heads) && fits(batch * queries) && fits(chunks)) {
        return Err(msg("decode_attention: geometry exceeds one launch".into()));
    }
    Ok(DecodeAttnPlan {
        batch,
        heads,
        kv_heads,
        queries,
        capacity,
        key_dim,
        value_dim,
        dtype,
        chunks,
    })
}

/// Device bytes one [`decode_attention`] call holds for its partial-softmax workspace (freed when
/// the call returns; a graph memory node inside a captured step): `(2 + value_dim)` f32 per
/// (batch·query, head, chunk), the chunks covering `capacity`. Saturating.
pub fn decode_attention_workspace_bytes(
    batch: usize,
    heads: usize,
    queries: usize,
    capacity: usize,
    value_dim: usize,
) -> usize {
    batch
        .saturating_mul(queries)
        .saturating_mul(heads)
        .saturating_mul(capacity.div_ceil(DECODE_ATTN_CHUNK))
        .saturating_mul(value_dim.saturating_add(2))
        .saturating_mul(4)
}

/// The keys query `i` of a step starting at `start` sees: `[lo, hi)`.
fn visible_range(start: usize, i: usize, capacity: usize, window: Option<usize>) -> (usize, usize) {
    let pos = start + i;
    let hi = (pos + 1).min(capacity);
    let lo = match window {
        Some(w) if w > 0 => (pos + 1).saturating_sub(w),
        _ => 0,
    };
    (lo, hi)
}

/// The chunked arithmetic of [`decode_attention`] on the host, in f32 (see the module docs):
/// `start` is the step start the device tensor would hold. Returns `[B, H, M, dv]` in `q`'s
/// dtype on `q`'s device. Used as the CPU path and as the GPU kernel's reference.
pub fn decode_attention_reference(
    q: &Tensor,
    k: &Tensor,
    v: &Tensor,
    start: usize,
    spec: DecodeAttnSpec,
) -> candle_core::Result<Tensor> {
    let (batch, heads, queries, key_dim) = q.dims4()?;
    let (_, kv_heads, capacity, _) = k.dims4()?;
    let value_dim = v.dim(3)?;
    if kv_heads == 0 || heads % kv_heads != 0 {
        return Err(msg("decode_attention_reference: heads do not group".into()));
    }
    if start + queries > capacity {
        return Err(msg(format!(
            "decode_attention_reference: a {queries}-query step at {start} ends past the \
             capacity {capacity}"
        )));
    }
    let host = |t: &Tensor| -> candle_core::Result<Vec<f32>> {
        t.to_dtype(DType::F32)?.flatten_all()?.to_vec1::<f32>()
    };
    let (qh, kh, vh) = (host(q)?, host(k)?, host(v)?);
    let groups = heads / kv_heads;
    let chunk = DECODE_ATTN_CHUNK;
    let mut out = vec![0f32; batch * heads * queries * value_dim];
    let mut scores = vec![0f32; chunk];
    for b in 0..batch {
        for h in 0..heads {
            let kvh = h / groups;
            let kbase = (b * kv_heads + kvh) * capacity * key_dim;
            let vbase = (b * kv_heads + kvh) * capacity * value_dim;
            for i in 0..queries {
                let qrow = &qh[((b * heads + h) * queries + i) * key_dim..][..key_dim];
                let (lo, hi) = visible_range(start, i, capacity, spec.window);
                // Per-chunk partial state (max, sum, acc), visible chunks only, ascending.
                let mut partials: Vec<(f32, f32, Vec<f32>)> = Vec::new();
                let (c_lo, c_hi) = (lo / chunk, (hi - 1) / chunk);
                for c in c_lo..=c_hi {
                    let k0 = c * chunk;
                    let ks = k0.max(lo);
                    let k1 = (k0 + chunk).min(hi);
                    for j in ks..k1 {
                        let krow = &kh[kbase + j * key_dim..][..key_dim];
                        let dot = qrow
                            .iter()
                            .zip(krow)
                            .fold(0f32, |acc, (a, b)| a.mul_add(*b, acc));
                        let mut s = dot * spec.scale;
                        if let Some(cap) = spec.softcap.filter(|c| *c > 0.0) {
                            s = cap * (s / cap).tanh();
                        }
                        scores[j - k0] = s;
                    }
                    let m = scores[ks - k0..k1 - k0]
                        .iter()
                        .fold(f32::NEG_INFINITY, |a, &s| a.max(s));
                    let mut sum = 0f32;
                    for s in &mut scores[ks - k0..k1 - k0] {
                        *s = (*s - m).exp();
                        sum += *s;
                    }
                    let mut acc = vec![0f32; value_dim];
                    for j in ks..k1 {
                        let p = scores[j - k0];
                        let vrow = &vh[vbase + j * value_dim..][..value_dim];
                        for (a, x) in acc.iter_mut().zip(vrow) {
                            *a = p.mul_add(*x, *a);
                        }
                    }
                    partials.push((m, sum, acc));
                }
                let mx = partials
                    .iter()
                    .fold(f32::NEG_INFINITY, |a, (m, _, _)| a.max(*m));
                let l = partials
                    .iter()
                    .fold(0f32, |acc, (m, s, _)| s.mul_add((m - mx).exp(), acc));
                let orow = &mut out[((b * heads + h) * queries + i) * value_dim..][..value_dim];
                for (d, o) in orow.iter_mut().enumerate() {
                    let a = partials
                        .iter()
                        .fold(0f32, |acc, (m, _, pa)| pa[d].mul_add((m - mx).exp(), acc));
                    *o = a / l;
                }
            }
        }
    }
    Tensor::from_vec(out, (batch, heads, queries, value_dim), q.device())?.to_dtype(q.dtype())
}

/// The first element of a (host-resident) `u32` index tensor.
fn host_index(index: &Tensor) -> candle_core::Result<usize> {
    let first = index.flatten_all()?.narrow(0, 0, 1)?.to_vec1::<u32>()?;
    Ok(first[0] as usize)
}

/// Length-aware decode attention (see the module docs). `q: [B, H, M, dk]` are the step's
/// queries at positions `start[0] .. start[0] + M`, `k` / `v` the **whole** preallocated
/// `[B, Hkv, capacity, dk | dv]` cache buffers with the step's own keys already written at
/// those positions, `start` a `u32` tensor whose first element is the step start. Returns
/// `[B, H, M, dv]` in the operands' dtype.
///
/// On CUDA the position is read on the device (a graph replays it correctly); on the CPU the
/// reference runs with the index read from the host tensor.
pub fn decode_attention(
    q: &Tensor,
    k: &Tensor,
    v: &Tensor,
    start: &Tensor,
    spec: DecodeAttnSpec,
) -> candle_core::Result<Tensor> {
    let plan = check_decode_attention(q, k, v, start)?;
    match q.device() {
        Device::Cpu => decode_attention_reference(q, k, v, host_index(start)?, spec),
        #[cfg(feature = "cuda")]
        Device::Cuda(_) => cuda_impl::decode_attention(q, k, v, start, spec, plan),
        other => {
            let _ = plan;
            Err(msg(format!("decode_attention: no kernel for {other:?}")))
        }
    }
}

/// The axis split of a `write_rows_at`: `dst` `[outer, cap, inner]` and `src` `[outer, rows,
/// inner]` around `dim`.
fn row_split(
    dst: &Tensor,
    src: &Tensor,
    dim: usize,
) -> candle_core::Result<(usize, usize, usize, usize)> {
    let (dd, sd) = (dst.dims(), src.dims());
    if dd.len() != sd.len()
        || dim >= dd.len()
        || dd[..dim] != sd[..dim]
        || dd[dim + 1..] != sd[dim + 1..]
        || sd[dim] > dd[dim]
    {
        return Err(msg(format!(
            "write_rows_at: src {sd:?} does not fit dst {dd:?} along axis {dim}"
        )));
    }
    let outer = dd[..dim].iter().product();
    let inner = dd[dim + 1..].iter().product();
    Ok((outer, sd[dim], inner, dd[dim]))
}

/// Write `src` into `dst` along axis `dim` starting at row `index[0]` (a device `u32` on CUDA):
/// `dst` and `src` agree on every other axis, share a dtype and are contiguous. In place. On the
/// CPU a write past `dst`'s extent is an error; on CUDA the rows past it are dropped (the caller
/// refuses such a step on the host before any launch).
pub fn write_rows_at(
    dst: &Tensor,
    src: &Tensor,
    index: &Tensor,
    dim: usize,
) -> candle_core::Result<()> {
    let (outer, rows, inner, cap) = row_split(dst, src, dim)?;
    if dst.dtype() != src.dtype() || index.dtype() != DType::U32 || index.elem_count() == 0 {
        return Err(msg(
            "write_rows_at: dst/src dtypes differ, or the index is not a U32 tensor".into(),
        ));
    }
    if !dst.is_contiguous() || !src.is_contiguous() {
        return Err(msg("write_rows_at: dst and src must be contiguous".into()));
    }
    match dst.device() {
        Device::Cpu => {
            let at = host_index(index)?;
            if at + rows > cap {
                return Err(msg(format!(
                    "write_rows_at: rows {at}..{} past the extent {cap}",
                    at + rows
                )));
            }
            dst.slice_set(src, dim, at)
        }
        #[cfg(feature = "cuda")]
        Device::Cuda(_) => dst.inplace_op3(
            src,
            index,
            &cuda_impl::WriteRowsAt {
                outer,
                rows,
                inner,
                cap,
            },
        ),
        other => {
            let _ = (outer, inner);
            Err(msg(format!("write_rows_at: no kernel for {other:?}")))
        }
    }
}

/// Copy slot `index[0]` (a device `u32` on CUDA) of `src: [slots, ...]` into a fresh contiguous
/// tensor of shape `src.dims()[1..]`. `src` must be contiguous. An out-of-range index is an error
/// on the CPU and reads zeros on CUDA.
pub fn read_slot(src: &Tensor, index: &Tensor) -> candle_core::Result<Tensor> {
    let dims = src.dims();
    if dims.is_empty() || !src.is_contiguous() {
        return Err(msg(
            "read_slot: src must be a contiguous [slots, ...] tensor".into(),
        ));
    }
    if index.dtype() != DType::U32 || index.elem_count() == 0 {
        return Err(msg(
            "read_slot: the index must be a non-empty U32 tensor".into()
        ));
    }
    match src.device() {
        Device::Cpu => {
            let at = host_index(index)?;
            if at >= dims[0] {
                return Err(msg(format!("read_slot: slot {at} of {}", dims[0])));
            }
            src.narrow(0, at, 1)?.squeeze(0)?.copy()
        }
        #[cfg(feature = "cuda")]
        Device::Cuda(_) => cuda_impl::read_slot(src, index),
        other => Err(msg(format!("read_slot: no kernel for {other:?}"))),
    }
}

/// [`read_slot`] times a per-row scale in one pass (sc-24446): slot `index[0]` of the `F32`
/// `src: [slots, ...]`, each element multiplied by `g` — an `F32` tensor whose shape is a leading
/// prefix of the slot's (`[B, H]` over a `[B, H, Dv, Dk]` state), broadcast over the rest. The
/// bits of `read_slot(src, index)?.broadcast_mul(g)` (one IEEE multiply per element) without the
/// intermediate copy: a Gated DeltaNet step's first state decay fused into its checkpoint-ring
/// read. Errors as [`read_slot`] does, and on a dtype or shape `g` cannot scale.
pub fn read_slot_scaled(src: &Tensor, index: &Tensor, g: &Tensor) -> candle_core::Result<Tensor> {
    let dims = src.dims();
    if src.dtype() != DType::F32 || g.dtype() != DType::F32 || !g.is_contiguous() {
        return Err(msg(
            "read_slot_scaled: src and a contiguous g must both be F32".into(),
        ));
    }
    if dims.is_empty() || g.rank() >= dims.len() || g.dims() != &dims[1..=g.rank()] {
        return Err(msg(format!(
            "read_slot_scaled: g {:?} is not a leading prefix of the slot {:?}",
            g.dims(),
            dims.get(1..).unwrap_or(&[])
        )));
    }
    match src.device() {
        Device::Cpu => {
            let slot = read_slot(src, index)?;
            let mut shape = g.dims().to_vec();
            shape.resize(slot.rank(), 1);
            slot.broadcast_mul(&g.reshape(shape)?)
        }
        #[cfg(feature = "cuda")]
        Device::Cuda(_) => cuda_impl::read_slot_scaled(src, index, g),
        other => Err(msg(format!("read_slot_scaled: no kernel for {other:?}"))),
    }
}

/// Whether these primitives serve `device`: the CPU (the host reference) always; CUDA when the
/// kernels compile and load there (the reason is the nvrtc seam's cached error); nothing else.
pub fn available(device: &Device) -> Result<(), String> {
    match device {
        Device::Cpu => Ok(()),
        #[cfg(feature = "cuda")]
        Device::Cuda(dev) => DECODE_ATTENTION_SRC
            .compiled(dev)
            .map(|_| ())
            .map_err(|e| e.to_string()),
        other => Err(format!("decode_attention: no kernel for {other:?}")),
    }
}

#[cfg(feature = "cuda")]
mod cuda_impl {
    use super::*;
    use candle_core::backend::BackendStorage;
    use candle_core::cuda_backend::cudarc::driver::{LaunchConfig, PushKernelArg};
    use candle_core::cuda_backend::CudaDType;
    use candle_core::op::BackpropOp;
    use candle_core::{CudaStorage, InplaceOp3, Layout, Shape, Storage};

    fn drv(e: impl std::fmt::Debug) -> candle_core::Error {
        candle_core::Error::Cuda(format!("decode_attention kernel: {e:?}").into())
    }

    fn function(
        dev: &candle_core::CudaDevice,
        name: &str,
    ) -> candle_core::Result<candle_core::cuda_backend::cudarc::driver::CudaFunction> {
        let module = DECODE_ATTENTION_SRC
            .compiled(dev)
            .map_err(|e| msg(e.to_string()))?;
        module.function(name).map_err(|e| msg(e.to_string()))
    }

    fn cuda_storage(storage: &Storage) -> candle_core::Result<&CudaStorage> {
        match storage {
            Storage::Cuda(c) => Ok(c),
            _ => Err(msg("decode_attention: expected a CUDA tensor".into())),
        }
    }

    fn launch_typed<T: CudaDType + candle_core::cuda_backend::cudarc::driver::DeviceRepr>(
        q: &Tensor,
        k: &Tensor,
        v: &Tensor,
        start: &Tensor,
        spec: DecodeAttnSpec,
        plan: DecodeAttnPlan,
        suffix: &str,
    ) -> candle_core::Result<Tensor> {
        let q = q.contiguous()?;
        let (qs, ql) = q.storage_and_layout();
        let (ks, kl) = k.storage_and_layout();
        let (vs, vl) = v.storage_and_layout();
        let (ss, sl) = start.storage_and_layout();
        let (qc, kc, vc, sc) = (
            cuda_storage(&qs)?,
            cuda_storage(&ks)?,
            cuda_storage(&vs)?,
            cuda_storage(&ss)?,
        );
        let dev = qc.device.clone();
        let q_slice = qc.as_cuda_slice::<T>()?.slice(ql.start_offset()..);
        let k_slice = kc.as_cuda_slice::<T>()?.slice(kl.start_offset()..);
        let v_slice = vc.as_cuda_slice::<T>()?.slice(vl.start_offset()..);
        let s_slice = sc.as_cuda_slice::<u32>()?.slice(sl.start_offset()..);
        let bm = plan.batch * plan.queries;
        let ws_len = bm * plan.heads * plan.chunks * (plan.value_dim + 2);
        // SAFETY: every workspace cell the combine reads is written by the partial pass first
        // (the combine reads exactly the chunks the partial pass does not skip).
        let ws = unsafe { dev.alloc::<f32>(ws_len) }?;
        // SAFETY: the combine writes every output element.
        let out = unsafe { dev.alloc::<T>(bm * plan.heads * plan.value_dim) }?;
        let (heads, kv_heads, queries, cap) = (
            plan.heads as i32,
            plan.kv_heads as i32,
            plan.queries as i32,
            plan.capacity as i32,
        );
        let (dk, dv, chunks) = (
            plan.key_dim as i32,
            plan.value_dim as i32,
            plan.chunks as i32,
        );
        let scale = spec.scale;
        let softcap = spec.softcap.filter(|c| *c > 0.0).unwrap_or(0.0);
        let window = spec
            .window
            .map(|w| i32::try_from(w).unwrap_or(i32::MAX))
            .unwrap_or(0);
        let stream = dev.cuda_stream();
        let partial = function(&dev, &format!("decode_attn_partial_{suffix}"))?;
        // One block per (chunk, KV head × tile of the query heads sharing it, batch·query).
        let groups = plan.heads / plan.kv_heads;
        let tile = group_tile(groups, plan.key_dim, plan.value_dim);
        let tiles = groups.div_ceil(tile);
        let shared = partial_shared_bytes(tile, plan.key_dim, plan.value_dim);
        let cfg = LaunchConfig {
            grid_dim: (
                plan.chunks as u32,
                (plan.kv_heads * tiles) as u32,
                bm as u32,
            ),
            block_dim: (DECODE_ATTN_THREADS, 1, 1),
            shared_mem_bytes: shared as u32,
        };
        let tile = tile as i32;
        let mut b = stream.launch_builder(&partial);
        b.arg(&q_slice)
            .arg(&k_slice)
            .arg(&v_slice)
            .arg(&s_slice)
            .arg(&ws)
            .arg(&heads)
            .arg(&kv_heads)
            .arg(&queries)
            .arg(&cap)
            .arg(&dk)
            .arg(&dv)
            .arg(&scale)
            .arg(&softcap)
            .arg(&window)
            .arg(&chunks)
            .arg(&tile);
        // SAFETY: argument list matches `decode_attn_partial_*` in `decode_attention.cu`.
        unsafe { b.launch(cfg) }.map_err(drv)?;
        let combine = function(&dev, &format!("decode_attn_combine_{suffix}"))?;
        let cfg = LaunchConfig {
            grid_dim: (plan.heads as u32, bm as u32, 1),
            block_dim: (DECODE_ATTN_THREADS, 1, 1),
            shared_mem_bytes: 0,
        };
        let mut b = stream.launch_builder(&combine);
        b.arg(&ws)
            .arg(&s_slice)
            .arg(&out)
            .arg(&heads)
            .arg(&queries)
            .arg(&cap)
            .arg(&dv)
            .arg(&window)
            .arg(&chunks);
        // SAFETY: argument list matches `decode_attn_combine_*`.
        unsafe { b.launch(cfg) }.map_err(drv)?;
        drop(ws);
        Ok(Tensor::from_storage(
            Storage::Cuda(CudaStorage::wrap_cuda_slice(out, dev)),
            Shape::from((plan.batch, plan.heads, plan.queries, plan.value_dim)),
            BackpropOp::none(),
            false,
        ))
    }

    pub(super) fn decode_attention(
        q: &Tensor,
        k: &Tensor,
        v: &Tensor,
        start: &Tensor,
        spec: DecodeAttnSpec,
        plan: DecodeAttnPlan,
    ) -> candle_core::Result<Tensor> {
        if !k.is_contiguous() || !v.is_contiguous() {
            return Err(msg(
                "decode_attention: the K/V buffers must be contiguous".into()
            ));
        }
        match plan.dtype {
            DType::F32 => launch_typed::<f32>(q, k, v, start, spec, plan, "f32"),
            _ => launch_typed::<half::bf16>(q, k, v, start, spec, plan, "bf16"),
        }
    }

    /// The in-place row write ([`write_rows_at`]).
    pub(super) struct WriteRowsAt {
        pub(super) outer: usize,
        pub(super) rows: usize,
        pub(super) inner: usize,
        pub(super) cap: usize,
    }

    fn grid(n: usize) -> LaunchConfig {
        let block = 256u32;
        let blocks = n.div_ceil(block as usize).clamp(1, 65535) as u32;
        LaunchConfig {
            grid_dim: (blocks, 1, 1),
            block_dim: (block, 1, 1),
            shared_mem_bytes: 0,
        }
    }

    impl InplaceOp3 for WriteRowsAt {
        fn name(&self) -> &'static str {
            "write_rows_at"
        }

        fn cpu_fwd(
            &self,
            _: &mut candle_core::CpuStorage,
            _: &Layout,
            _: &candle_core::CpuStorage,
            _: &Layout,
            _: &candle_core::CpuStorage,
            _: &Layout,
        ) -> candle_core::Result<()> {
            Err(msg("write_rows_at: the CPU path is slice_set".into()))
        }

        fn cuda_fwd(
            &self,
            dst: &mut CudaStorage,
            dl: &Layout,
            src: &CudaStorage,
            sl: &Layout,
            index: &CudaStorage,
            il: &Layout,
        ) -> candle_core::Result<()> {
            let dev = dst.device.clone();
            let dtype = dst.dtype();
            let index = index.as_cuda_slice::<u32>()?.slice(il.start_offset()..);
            let n = self.outer * self.rows * self.inner;
            let (outer, rows, inner, cap) = (
                self.outer as u64,
                self.rows as u64,
                self.inner as u64,
                self.cap as u64,
            );
            let stream = dev.cuda_stream();
            let dst_offset = dl.start_offset();
            macro_rules! run {
                ($t:ty, $kernel:literal) => {{
                    let s = src.as_cuda_slice::<$t>()?.slice(sl.start_offset()..);
                    let f = function(&dev, $kernel)?;
                    let mut d = dst.as_cuda_slice_mut::<$t>()?.slice_mut(dst_offset..);
                    let mut b = stream.launch_builder(&f);
                    b.arg(&mut d)
                        .arg(&s)
                        .arg(&index)
                        .arg(&outer)
                        .arg(&rows)
                        .arg(&inner)
                        .arg(&cap);
                    // SAFETY: argument list matches `write_rows_at_*` in `decode_attention.cu`.
                    unsafe { b.launch(grid(n)) }.map_err(drv)?;
                    Ok(())
                }};
            }
            match dtype {
                DType::F32 => run!(f32, "write_rows_at_u32"),
                DType::U32 => run!(u32, "write_rows_at_u32"),
                DType::BF16 => run!(half::bf16, "write_rows_at_u16"),
                DType::F16 => run!(half::f16, "write_rows_at_u16"),
                other => Err(msg(format!("write_rows_at: dtype {other:?} is not served"))),
            }
        }
    }

    pub(super) fn read_slot(src: &Tensor, index: &Tensor) -> candle_core::Result<Tensor> {
        let (ss, sl) = src.storage_and_layout();
        let (is, il) = index.storage_and_layout();
        let (sc, ic) = (cuda_storage(&ss)?, cuda_storage(&is)?);
        let dev = sc.device.clone();
        let dims = src.dims();
        let slots = dims[0];
        let n: usize = dims[1..].iter().product();
        let index = ic.as_cuda_slice::<u32>()?.slice(il.start_offset()..);
        let stream = dev.cuda_stream();
        let (n64, slots64) = (n as u64, slots as u64);
        let shape = Shape::from(&dims[1..]);
        macro_rules! run {
            ($t:ty, $kernel:literal) => {{
                let s = sc.as_cuda_slice::<$t>()?.slice(sl.start_offset()..);
                // SAFETY: the kernel writes every element.
                let out = unsafe { dev.alloc::<$t>(n) }?;
                let f = function(&dev, $kernel)?;
                let mut b = stream.launch_builder(&f);
                b.arg(&s).arg(&index).arg(&out).arg(&n64).arg(&slots64);
                // SAFETY: argument list matches `read_slot_*`.
                unsafe { b.launch(grid(n)) }.map_err(drv)?;
                Ok(Tensor::from_storage(
                    Storage::Cuda(CudaStorage::wrap_cuda_slice(out, dev.clone())),
                    shape,
                    BackpropOp::none(),
                    false,
                ))
            }};
        }
        match src.dtype() {
            DType::F32 => run!(f32, "read_slot_u32"),
            DType::U32 => run!(u32, "read_slot_u32"),
            DType::BF16 => run!(half::bf16, "read_slot_u16"),
            DType::F16 => run!(half::f16, "read_slot_u16"),
            other => Err(msg(format!("read_slot: dtype {other:?} is not served"))),
        }
    }

    pub(super) fn read_slot_scaled(
        src: &Tensor,
        index: &Tensor,
        g: &Tensor,
    ) -> candle_core::Result<Tensor> {
        if !src.is_contiguous() {
            return Err(msg(
                "read_slot_scaled: src must be a contiguous [slots, ...] tensor".into(),
            ));
        }
        if index.dtype() != DType::U32 || index.elem_count() == 0 {
            return Err(msg(
                "read_slot_scaled: the index must be a non-empty U32 tensor".into(),
            ));
        }
        let (ss, sl) = src.storage_and_layout();
        let (is, il) = index.storage_and_layout();
        let (gs, gl) = g.storage_and_layout();
        let (sc, ic, gc) = (cuda_storage(&ss)?, cuda_storage(&is)?, cuda_storage(&gs)?);
        let dev = sc.device.clone();
        let dims = src.dims();
        let n: usize = dims[1..].iter().product();
        let inner = n / g.elem_count().max(1);
        let s = sc.as_cuda_slice::<f32>()?.slice(sl.start_offset()..);
        let index = ic.as_cuda_slice::<u32>()?.slice(il.start_offset()..);
        let gv = gc.as_cuda_slice::<f32>()?.slice(gl.start_offset()..);
        let (n64, slots64, inner64) = (n as u64, dims[0] as u64, inner as u64);
        // SAFETY: the kernel writes every element.
        let out = unsafe { dev.alloc::<f32>(n) }?;
        let f = function(&dev, "read_slot_scaled_f32")?;
        let stream = dev.cuda_stream();
        let mut b = stream.launch_builder(&f);
        b.arg(&s)
            .arg(&index)
            .arg(&gv)
            .arg(&out)
            .arg(&n64)
            .arg(&slots64)
            .arg(&inner64);
        // SAFETY: argument list matches `read_slot_scaled_f32`.
        unsafe { b.launch(grid(n)) }.map_err(drv)?;
        Ok(Tensor::from_storage(
            Storage::Cuda(CudaStorage::wrap_cuda_slice(out, dev.clone())),
            Shape::from(&dims[1..]),
            BackpropOp::none(),
            false,
        ))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn ramp(dims: &[usize], seed: u32, scale: f32) -> Tensor {
        let n: usize = dims.iter().product();
        let mut x = seed.max(1);
        let data: Vec<f32> = (0..n)
            .map(|_| {
                x ^= x << 13;
                x ^= x >> 17;
                x ^= x << 5;
                ((x % 2001) as f32 / 1000.0 - 1.0) * scale
            })
            .collect();
        Tensor::from_vec(data, dims, &Device::Cpu).unwrap()
    }

    /// Plain (unchunked) masked softmax attention over the visible keys — the math the chunked
    /// reference must reproduce to rounding.
    fn naive(q: &Tensor, k: &Tensor, v: &Tensor, start: usize, spec: DecodeAttnSpec) -> Vec<f32> {
        let (b, h, m, dk) = q.dims4().unwrap();
        let (_, hkv, cap, _) = k.dims4().unwrap();
        let dv = v.dim(3).unwrap();
        let (qh, kh, vh) = (
            q.flatten_all().unwrap().to_vec1::<f32>().unwrap(),
            k.flatten_all().unwrap().to_vec1::<f32>().unwrap(),
            v.flatten_all().unwrap().to_vec1::<f32>().unwrap(),
        );
        let groups = h / hkv;
        let mut out = Vec::new();
        for bi in 0..b {
            for hi in 0..h {
                let kvh = hi / groups;
                for i in 0..m {
                    let (lo, end) = visible_range(start, i, cap, spec.window);
                    let q0 = ((bi * h + hi) * m + i) * dk;
                    let scores: Vec<f64> = (lo..end)
                        .map(|j| {
                            let k0 = ((bi * hkv + kvh) * cap + j) * dk;
                            let dot: f64 = (0..dk)
                                .map(|d| f64::from(qh[q0 + d]) * f64::from(kh[k0 + d]))
                                .sum();
                            let mut s = dot * f64::from(spec.scale);
                            if let Some(c) = spec.softcap {
                                s = f64::from(c) * (s / f64::from(c)).tanh();
                            }
                            s
                        })
                        .collect();
                    let mx = scores.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
                    let w: Vec<f64> = scores.iter().map(|s| (s - mx).exp()).collect();
                    let l: f64 = w.iter().sum();
                    for d in 0..dv {
                        let a: f64 = (lo..end)
                            .zip(&w)
                            .map(|(j, p)| p * f64::from(vh[((bi * hkv + kvh) * cap + j) * dv + d]))
                            .sum();
                        out.push((a / l) as f32);
                    }
                }
            }
        }
        out
    }

    fn close(got: &[f32], want: &[f32], tol: f32) {
        assert_eq!(got.len(), want.len());
        for (i, (g, w)) in got.iter().zip(want).enumerate() {
            assert!((g - w).abs() <= tol * (1.0 + w.abs()), "[{i}] {g} vs {w}");
        }
    }

    fn start(at: u32) -> Tensor {
        Tensor::new(&[at], &Device::Cpu).unwrap()
    }

    /// The chunked reference equals plain softmax attention across chunk boundaries, GQA,
    /// multi-query steps, a sliding window, a soft-cap and MLA-style unequal K/V widths.
    #[test]
    fn chunked_reference_matches_plain_attention() {
        let cap = 3 * DECODE_ATTN_CHUNK + 17;
        let cases: &[(usize, usize, usize, usize, usize, usize, DecodeAttnSpec)] = &[
            // (b, h, hkv, m, dk, dv, spec)
            (
                1,
                4,
                2,
                1,
                16,
                16,
                DecodeAttnSpec {
                    scale: 0.25,
                    softcap: None,
                    window: None,
                },
            ),
            (
                2,
                6,
                2,
                5,
                8,
                8,
                DecodeAttnSpec {
                    scale: 0.35,
                    softcap: None,
                    window: None,
                },
            ),
            (
                1,
                2,
                2,
                3,
                12,
                8,
                DecodeAttnSpec {
                    scale: 0.3,
                    softcap: Some(5.0),
                    window: None,
                },
            ),
            (
                1,
                4,
                1,
                4,
                8,
                8,
                DecodeAttnSpec {
                    scale: 0.3,
                    softcap: None,
                    window: Some(100),
                },
            ),
        ];
        for (n, &(b, h, hkv, m, dk, dv, spec)) in cases.iter().enumerate() {
            let q = ramp(&[b, h, m, dk], 11 + n as u32, 1.0);
            let k = ramp(&[b, hkv, cap, dk], 23 + n as u32, 1.0);
            let v = ramp(&[b, hkv, cap, dv], 37 + n as u32, 1.0);
            for at in [
                0usize,
                5,
                DECODE_ATTN_CHUNK - 1,
                DECODE_ATTN_CHUNK,
                2 * DECODE_ATTN_CHUNK + 3,
                cap - m,
            ] {
                let got = decode_attention(&q, &k, &v, &start(at as u32), spec).unwrap();
                assert_eq!(got.dims(), &[b, h, m, dv]);
                let got = got.flatten_all().unwrap().to_vec1::<f32>().unwrap();
                close(&got, &naive(&q, &k, &v, at, spec), 2e-5);
            }
        }
    }

    /// Keys past the step's last position (the stale tail of a rolled-back cache) and before a
    /// window never reach the output, whatever they hold.
    #[test]
    fn invisible_keys_do_not_leak() {
        let cap = 2 * DECODE_ATTN_CHUNK;
        let spec = DecodeAttnSpec {
            scale: 0.5,
            softcap: None,
            window: Some(40),
        };
        let q = ramp(&[1, 2, 2, 8], 3, 1.0);
        let k = ramp(&[1, 1, cap, 8], 5, 1.0);
        let v = ramp(&[1, 1, cap, 8], 7, 1.0);
        let at = 100usize;
        let base = decode_attention(&q, &k, &v, &start(at as u32), spec).unwrap();
        // Poison every key neither query sees: query 0 (position `at`) sees `at - 39 ..= at`,
        // query 1 sees `at - 38 ..= at + 1`.
        let poison = |t: &Tensor| {
            let mut data = t.flatten_all().unwrap().to_vec1::<f32>().unwrap();
            for j in (0..at + 1 - 40).chain(at + 2..cap) {
                for d in 0..8 {
                    data[j * 8 + d] = 1e6;
                }
            }
            Tensor::from_vec(data, t.dims(), &Device::Cpu).unwrap()
        };
        let got = decode_attention(&q, &poison(&k), &poison(&v), &start(at as u32), spec).unwrap();
        assert_eq!(
            base.flatten_all().unwrap().to_vec1::<f32>().unwrap(),
            got.flatten_all().unwrap().to_vec1::<f32>().unwrap()
        );
    }

    #[test]
    fn checks_refuse_malformed_operands() {
        let q = ramp(&[1, 4, 1, 8], 1, 1.0);
        let k = ramp(&[1, 3, 16, 8], 2, 1.0);
        assert!(
            check_decode_attention(&q, &k, &k, &start(0)).is_err(),
            "4 % 3"
        );
        let k = ramp(&[1, 2, 16, 8], 2, 1.0);
        let v = ramp(&[1, 2, 15, 8], 2, 1.0);
        assert!(
            check_decode_attention(&q, &k, &v, &start(0)).is_err(),
            "cap"
        );
        assert!(
            check_decode_attention(&q, &k, &k, &q).is_err(),
            "start dtype"
        );
        let plan = check_decode_attention(&q, &k, &k, &start(0)).unwrap();
        assert_eq!((plan.heads, plan.kv_heads, plan.chunks), (4, 2, 1));
        let f16 = q.to_dtype(DType::F16).unwrap();
        assert!(check_decode_attention(&f16, &k, &k, &start(0)).is_err());
        assert!(decode_attention_reference(
            &q,
            &k,
            &k,
            16,
            DecodeAttnSpec {
                scale: 1.0,
                softcap: None,
                window: None
            }
        )
        .is_err());
    }

    /// The partial pass's head tile: the whole group up to the register bound of 8 heads, and —
    /// with a 64-key chunk — every served width fits 48 KiB of shared memory at that bound.
    #[test]
    fn group_tile_takes_the_whole_group_within_the_bounds() {
        assert_eq!(group_tile(1, 128, 128), 1);
        assert_eq!(group_tile(4, 128, 128), 4);
        assert_eq!(group_tile(6, 256, 256), 6, "Qwen3.8: one tile of six");
        assert_eq!(group_tile(8, 128, 128), 8);
        assert_eq!(group_tile(16, 128, 128), 8, "tiled past the register bound");
        assert_eq!(
            group_tile(16, 1024, 1024),
            8,
            "8 × (1024 + 64) f32 fit 48 KiB"
        );
        assert_eq!(group_tile(0, 1024, 64), 1);
        for (dk, dv) in [
            (1024, 1024),
            (1024, 64),
            (64, 1024),
            (512, 512),
            (96, 64),
            (1, 1),
        ] {
            let tile = group_tile(16, dk, dv);
            assert!(partial_shared_bytes(tile, dk, dv) <= 48 * 1024, "{dk}/{dv}");
        }
    }

    /// The value pass's key slices: a value row's element pairs, then as many slices of the
    /// chunk as the block's remaining threads cover; the slice sums never outgrow the query
    /// rows' region by more than twice the block's threads.
    #[test]
    fn value_slices_cover_the_block() {
        assert_eq!(value_slices(256), 2, "Qwen3.8 head dim");
        assert_eq!(value_slices(128), 4);
        assert_eq!(value_slices(96), 5);
        assert_eq!(value_slices(64), 8);
        assert_eq!(value_slices(511), 1);
        assert_eq!(value_slices(512), 1);
        assert_eq!(value_slices(1024), 1);
        assert_eq!(value_slices(1), 256);
        for dv in 1..=DECODE_ATTN_MAX_HEAD_DIM {
            assert!(
                value_slices(dv) * dv <= 2 * 256 || value_slices(dv) == 1,
                "{dv}"
            );
        }
    }

    #[test]
    fn workspace_bytes_cover_every_chunk() {
        assert_eq!(
            decode_attention_workspace_bytes(1, 32, 1, 4096, 128),
            32 * (4096 / DECODE_ATTN_CHUNK) * 130 * 4
        );
        assert_eq!(
            decode_attention_workspace_bytes(1, 1, 2, DECODE_ATTN_CHUNK + 1, 6),
            2 * 2 * 8 * 4
        );
    }

    #[test]
    fn indexed_copies_on_the_host() {
        let dst = Tensor::zeros((2, 6, 3), DType::F32, &Device::Cpu).unwrap();
        let src = ramp(&[2, 2, 3], 9, 1.0);
        write_rows_at(&dst, &src, &start(3), 1).unwrap();
        assert_eq!(
            dst.narrow(1, 3, 2)
                .unwrap()
                .flatten_all()
                .unwrap()
                .to_vec1::<f32>()
                .unwrap(),
            src.flatten_all().unwrap().to_vec1::<f32>().unwrap()
        );
        assert!(
            write_rows_at(&dst, &src, &start(5), 1).is_err(),
            "past the extent"
        );
        let ring = ramp(&[4, 2, 3], 13, 1.0);
        let slot = read_slot(&ring, &start(2)).unwrap();
        assert_eq!(slot.dims(), &[2, 3]);
        assert_eq!(
            slot.flatten_all().unwrap().to_vec1::<f32>().unwrap(),
            ring.get(2)
                .unwrap()
                .flatten_all()
                .unwrap()
                .to_vec1::<f32>()
                .unwrap()
        );
        assert!(read_slot(&ring, &start(4)).is_err());
    }

    /// sc-24446: `read_slot_scaled` is `read_slot` then a broadcast multiply by the leading
    /// `[B, H]` scale, bit for bit, and refuses a scale that is not a leading prefix or not F32.
    #[test]
    fn a_scaled_slot_read_is_the_read_times_the_scale() {
        let ring = ramp(&[3, 2, 4, 5, 6], 31, 1.0);
        let g = ramp(&[2, 4], 7, 1.0);
        let at = Tensor::new(&[1u32], &Device::Cpu).unwrap();
        let got = read_slot_scaled(&ring, &at, &g).unwrap();
        let want = read_slot(&ring, &at)
            .unwrap()
            .broadcast_mul(&g.reshape((2, 4, 1, 1)).unwrap())
            .unwrap();
        assert_eq!(got.dims(), &[2, 4, 5, 6]);
        let bits = |t: &Tensor| -> Vec<u32> {
            t.flatten_all()
                .unwrap()
                .to_vec1::<f32>()
                .unwrap()
                .iter()
                .map(|x| x.to_bits())
                .collect()
        };
        assert_eq!(bits(&got), bits(&want));
        assert!(read_slot_scaled(&ring, &at, &ramp(&[4, 2], 7, 1.0)).is_err());
        assert!(read_slot_scaled(&ring, &at, &g.to_dtype(DType::BF16).unwrap()).is_err());
    }
}

#[cfg(all(test, feature = "cuda"))]
mod cuda_tests {
    use super::*;

    fn device() -> Option<Device> {
        Device::new_cuda(0).ok()
    }

    fn ramp(dims: &[usize], seed: u32, dev: &Device, dtype: DType) -> Tensor {
        let n: usize = dims.iter().product();
        let mut x = seed.max(1);
        let data: Vec<f32> = (0..n)
            .map(|_| {
                x ^= x << 13;
                x ^= x >> 17;
                x ^= x << 5;
                (x % 2001) as f32 / 1000.0 - 1.0
            })
            .collect();
        Tensor::from_vec(data, dims, dev)
            .unwrap()
            .to_dtype(dtype)
            .unwrap()
    }

    fn host(t: &Tensor) -> Vec<f32> {
        t.to_dtype(DType::F32)
            .unwrap()
            .flatten_all()
            .unwrap()
            .to_vec1::<f32>()
            .unwrap()
    }

    /// The kernel agrees with the CPU reference (f32 within rounding; bf16 within one output
    /// ulp-scale tolerance, the reference being f32 math on the same bf16 inputs), and two
    /// launches at the same position are bit-identical — the determinism graphs rely on.
    #[test]
    fn kernel_matches_the_reference_and_is_deterministic() {
        let Some(dev) = device() else { return };
        let cap = 3 * DECODE_ATTN_CHUNK + 5;
        for dtype in [DType::F32, DType::BF16] {
            for (h, hkv, m, dk, dv, spec) in [
                (
                    8usize,
                    2usize,
                    1usize,
                    64usize,
                    64usize,
                    DecodeAttnSpec {
                        scale: 0.125,
                        softcap: None,
                        window: None,
                    },
                ),
                (
                    4,
                    4,
                    5,
                    128,
                    128,
                    DecodeAttnSpec {
                        scale: 0.09,
                        softcap: Some(30.0),
                        window: None,
                    },
                ),
                (
                    4,
                    1,
                    3,
                    96,
                    64,
                    DecodeAttnSpec {
                        scale: 0.1,
                        softcap: None,
                        window: Some(70),
                    },
                ),
                (
                    2,
                    2,
                    2,
                    512,
                    512,
                    DecodeAttnSpec {
                        scale: 0.044,
                        softcap: None,
                        window: None,
                    },
                ),
                // GQA groups 1, 4 and 8 on one geometry (sc-24441: a partial block serves the
                // query heads sharing a KV head).
                (
                    8,
                    8,
                    3,
                    128,
                    128,
                    DecodeAttnSpec {
                        scale: 0.088,
                        softcap: None,
                        window: None,
                    },
                ),
                (
                    8,
                    2,
                    3,
                    128,
                    128,
                    DecodeAttnSpec {
                        scale: 0.088,
                        softcap: None,
                        window: None,
                    },
                ),
                (
                    8,
                    1,
                    3,
                    128,
                    128,
                    DecodeAttnSpec {
                        scale: 0.088,
                        softcap: None,
                        window: None,
                    },
                ),
            ] {
                let q = ramp(&[1, h, m, dk], 3, &dev, dtype);
                let k = ramp(&[1, hkv, cap, dk], 5, &dev, dtype);
                let v = ramp(&[1, hkv, cap, dv], 7, &dev, dtype);
                for at in [0usize, 17, DECODE_ATTN_CHUNK, cap - m] {
                    let s = Tensor::new(&[at as u32], &dev).unwrap();
                    let got = decode_attention(&q, &k, &v, &s, spec).unwrap();
                    let again = decode_attention(&q, &k, &v, &s, spec).unwrap();
                    assert_eq!(host(&got), host(&again), "bit-identical relaunch");
                    let want = decode_attention_reference(
                        &q.to_device(&Device::Cpu).unwrap(),
                        &k.to_device(&Device::Cpu).unwrap(),
                        &v.to_device(&Device::Cpu).unwrap(),
                        at,
                        spec,
                    )
                    .unwrap();
                    let tol = if dtype == DType::F32 { 1e-5 } else { 1e-2 };
                    for (i, (g, w)) in host(&got).iter().zip(host(&want)).enumerate() {
                        assert!(
                            (g - w).abs() <= tol * (1.0 + w.abs()),
                            "{dtype:?} h={h} m={m} at={at} [{i}] {g} vs {w}"
                        );
                    }
                }
            }
        }
    }

    /// sc-24441: a partial block serving a tile of the query heads that share a KV head runs
    /// every head's reductions in the order a one-head block does, so a grouped launch is
    /// bit-identical to launching each query head alone against its KV head — groups 1, 4, 8,
    /// and 16 at the widest head (two tiles of 8), with a soft-cap and a sliding window.
    #[test]
    fn grouped_heads_are_bit_identical_to_one_head_per_launch() {
        let Some(dev) = device() else { return };
        let cap = 2 * DECODE_ATTN_CHUNK + 9;
        for dtype in [DType::F32, DType::BF16] {
            for (h, hkv, m, dk, dv, spec) in [
                (4usize, 4usize, 2usize, 64usize, 64usize, None),
                (8, 2, 3, 128, 128, Some(30.0)),
                (8, 1, 1, 128, 96, None),
                (16, 1, 2, 1024, 64, None),
            ] {
                let spec = DecodeAttnSpec {
                    scale: 0.07,
                    softcap: spec,
                    window: Some(90),
                };
                let q = ramp(&[1, h, m, dk], 11, &dev, dtype);
                let k = ramp(&[1, hkv, cap, dk], 13, &dev, dtype);
                let v = ramp(&[1, hkv, cap, dv], 17, &dev, dtype);
                for at in [0usize, DECODE_ATTN_CHUNK + 3, cap - m] {
                    let s = Tensor::new(&[at as u32], &dev).unwrap();
                    let grouped = host(&decode_attention(&q, &k, &v, &s, spec).unwrap());
                    let groups = h / hkv;
                    let mut alone = Vec::with_capacity(grouped.len());
                    for head in 0..h {
                        let one =
                            |t: &Tensor, i: usize| t.narrow(1, i, 1).unwrap().contiguous().unwrap();
                        let out = decode_attention(
                            &one(&q, head),
                            &one(&k, head / groups),
                            &one(&v, head / groups),
                            &s,
                            spec,
                        )
                        .unwrap();
                        alone.extend(host(&out));
                    }
                    assert_eq!(grouped, alone, "{dtype:?} h={h} hkv={hkv} at={at}");
                }
            }
        }
    }

    /// The kernel against the reference at every chunk boundary (sc-24446): a step whose last
    /// query sees 1, C − 1, C, C + 1, 2C − 1, 2C, 2C + 1 keys, at the Qwen3.8 decode shape (24
    /// query heads over 4 KV heads, head dim 256, one query) and at a short multi-query step,
    /// both dtypes; plus widths the 16-byte loads do not divide (key 100, value 37), which take
    /// the scalar fallback.
    #[test]
    fn kernel_matches_the_reference_at_every_chunk_boundary() {
        let Some(dev) = device() else { return };
        let c = DECODE_ATTN_CHUNK;
        let cap = 3 * c + 7;
        for dtype in [DType::F32, DType::BF16] {
            for (h, hkv, m, dk, dv, window) in [
                (24usize, 4usize, 1usize, 256usize, 256usize, None),
                (24, 4, 3, 256, 256, None),
                (4, 2, 2, 100, 37, Some(c + 5)),
            ] {
                let spec = DecodeAttnSpec {
                    scale: (dk as f32).powf(-0.5),
                    softcap: None,
                    window,
                };
                let q = ramp(&[1, h, m, dk], 19, &dev, dtype);
                let k = ramp(&[1, hkv, cap, dk], 23, &dev, dtype);
                let v = ramp(&[1, hkv, cap, dv], 29, &dev, dtype);
                let (qc, kc, vc) = (
                    q.to_device(&Device::Cpu).unwrap(),
                    k.to_device(&Device::Cpu).unwrap(),
                    v.to_device(&Device::Cpu).unwrap(),
                );
                // The last query sees `len` keys: `at = len - m`.
                for len in [1, c - 1, c, c + 1, 2 * c - 1, 2 * c, 2 * c + 1, cap] {
                    if len < m {
                        continue;
                    }
                    let at = len - m;
                    let s = Tensor::new(&[at as u32], &dev).unwrap();
                    let got = host(&decode_attention(&q, &k, &v, &s, spec).unwrap());
                    let want = host(&decode_attention_reference(&qc, &kc, &vc, at, spec).unwrap());
                    let tol = if dtype == DType::F32 { 1e-5 } else { 1e-2 };
                    for (i, (g, w)) in got.iter().zip(&want).enumerate() {
                        assert!(
                            (g - w).abs() <= tol * (1.0 + w.abs()),
                            "{dtype:?} h={h} m={m} dk={dk} dv={dv} len={len} [{i}] {g} vs {w}"
                        );
                    }
                }
            }
        }
    }

    /// The chunking is aligned to absolute key positions (sc-24446), so the result does not
    /// depend on the cache's capacity: attention over a wide static buffer is bit-identical to
    /// the same step over a buffer exactly as long as the step's last position (the growing
    /// cache's concat with device positions) — at chunk boundaries and mid-chunk, with a window.
    #[test]
    fn the_result_does_not_depend_on_the_capacity() {
        let Some(dev) = device() else { return };
        let c = DECODE_ATTN_CHUNK;
        let cap = 4 * c + 11;
        for dtype in [DType::F32, DType::BF16] {
            for (h, hkv, m, dk, dv, window) in [
                (24usize, 4usize, 1usize, 256usize, 256usize, None),
                (8, 2, 3, 128, 96, Some(2 * c + 1)),
            ] {
                let spec = DecodeAttnSpec {
                    scale: 0.06,
                    softcap: None,
                    window,
                };
                let q = ramp(&[1, h, m, dk], 31, &dev, dtype);
                let k = ramp(&[1, hkv, cap, dk], 37, &dev, dtype);
                let v = ramp(&[1, hkv, cap, dv], 41, &dev, dtype);
                for at in [0usize, c - 1, c, 2 * c + 5, 3 * c] {
                    let s = Tensor::new(&[at as u32], &dev).unwrap();
                    let wide = host(&decode_attention(&q, &k, &v, &s, spec).unwrap());
                    let tight = |t: &Tensor| t.narrow(2, 0, at + m).unwrap().contiguous().unwrap();
                    let narrow =
                        host(&decode_attention(&q, &tight(&k), &tight(&v), &s, spec).unwrap());
                    assert_eq!(wide, narrow, "{dtype:?} h={h} m={m} at={at}");
                }
            }
        }
    }

    /// The 16-byte (key) and pair (value) loads and their scalar fallback run one fma sequence
    /// (sc-24446): K/V buffers that start one element past an aligned address — the scalar path —
    /// give bit-identical output to the same values aligned.
    #[test]
    fn vector_and_scalar_loads_are_bit_identical() {
        let Some(dev) = device() else { return };
        let cap = 3 * DECODE_ATTN_CHUNK + 3;
        // `[n + 1]` narrowed past its first element: contiguous, misaligned by one element.
        let misaligned = |t: &Tensor| {
            let flat = t.flatten_all().unwrap();
            let pad = Tensor::zeros(1, t.dtype(), &dev).unwrap();
            Tensor::cat(&[&pad, &flat], 0)
                .unwrap()
                .narrow(0, 1, flat.elem_count())
                .unwrap()
                .reshape(t.dims())
                .unwrap()
        };
        for dtype in [DType::F32, DType::BF16] {
            for (h, hkv, m, dk, dv) in [
                (24usize, 4usize, 1usize, 256usize, 256usize),
                (4, 1, 2, 64, 128),
            ] {
                let spec = DecodeAttnSpec {
                    scale: 0.07,
                    softcap: None,
                    window: None,
                };
                let q = ramp(&[1, h, m, dk], 43, &dev, dtype);
                let k = ramp(&[1, hkv, cap, dk], 47, &dev, dtype);
                let v = ramp(&[1, hkv, cap, dv], 53, &dev, dtype);
                let (km, vm) = (misaligned(&k), misaligned(&v));
                assert!(km.is_contiguous() && km.layout().start_offset() == 1);
                for at in [0usize, DECODE_ATTN_CHUNK, cap - m] {
                    let s = Tensor::new(&[at as u32], &dev).unwrap();
                    let aligned = host(&decode_attention(&q, &k, &v, &s, spec).unwrap());
                    let scalar = host(&decode_attention(&q, &km, &vm, &s, spec).unwrap());
                    assert_eq!(aligned, scalar, "{dtype:?} h={h} dk={dk} at={at}");
                }
            }
        }
    }

    #[test]
    fn indexed_copies_follow_the_device_index() {
        let Some(dev) = device() else { return };
        for dtype in [DType::F32, DType::BF16] {
            let dst = Tensor::zeros((2, 3, 10, 4), dtype, &dev).unwrap();
            let src = ramp(&[2, 3, 2, 4], 9, &dev, dtype);
            let at = Tensor::new(&[6u32], &dev).unwrap();
            write_rows_at(&dst, &src, &at, 2).unwrap();
            assert_eq!(host(&dst.narrow(2, 6, 2).unwrap()), host(&src));
            assert!(host(&dst.narrow(2, 0, 6).unwrap())
                .iter()
                .all(|x| *x == 0.0));
            let ring = ramp(&[3, 5, 7], 21, &dev, dtype);
            let slot = read_slot(&ring, &Tensor::new(&[1u32], &dev).unwrap()).unwrap();
            assert_eq!(host(&slot), host(&ring.get(1).unwrap()));
        }
    }

    /// sc-24446: the fused scaled slot read is bit-identical to `read_slot` then the ops path's
    /// broadcast multiply — the first decay of a Gated DeltaNet step — at a Qwen3.8 state shape
    /// and an odd one, at every slot.
    #[test]
    fn the_scaled_slot_read_is_bit_identical_to_the_ops_path() {
        let Some(dev) = device() else { return };
        for (dims, g_dims) in [
            (vec![4usize, 1, 48, 128, 128], vec![1usize, 48]),
            (vec![3, 2, 3, 5, 7], vec![2, 3]),
        ] {
            let ring = ramp(&dims, 17, &dev, DType::F32);
            let g = ramp(&g_dims, 5, &dev, DType::F32);
            let mut shape = g_dims.clone();
            shape.resize(dims.len() - 1, 1);
            for slot in 0..dims[0] as u32 {
                let at = Tensor::new(&[slot], &dev).unwrap();
                let fused = read_slot_scaled(&ring, &at, &g).unwrap();
                let ops = read_slot(&ring, &at)
                    .unwrap()
                    .broadcast_mul(&g.reshape(shape.clone()).unwrap())
                    .unwrap();
                let bits = |t: &Tensor| -> Vec<u32> {
                    t.flatten_all()
                        .unwrap()
                        .to_vec1::<f32>()
                        .unwrap()
                        .iter()
                        .map(|x| x.to_bits())
                        .collect()
                };
                assert_eq!(bits(&fused), bits(&ops), "{dims:?} slot {slot}");
            }
        }
    }
}
