//! Krea Realtime 14B **causal autoregressive** transformer forward + persistent KV cache (sc-8436, S3).
//!
//! The S1 audit established that Krea Realtime 14B is Wan 2.1 T2V 14B weight-for-weight, so the whole
//! DiT — patchify, embeddings, adaLN-6vec modulation, text cross-attention, gated-GELU FFN, 3-axis
//! RoPE, the modulated head — is **reused verbatim** from [`mlx_gen_wan::WanTransformer`]. The single
//! net-new compute delta of the autoregressive regime is confined to **self-attention**, and it is
//! three coupled pieces **adapted from** the reference `transformer/causal_model.py`
//! (`krea-ai/realtime-video`) — reimplemented in native MLX, not copied source:
//!
//!   1. **Block-causal attention mask** ([`build_block_causal_mask`]) — a query attends to every token
//!      up to the END of its own frame-block (intra-block bidirectional) but no later block
//!      (inter-block strictly causal): `allowed(q, kv) = (kv < end_of_block(q)) || (q == kv)`, built as
//!      the additive SDPA form (`0` allowed / `-inf` masked). Mirrors `get_sdpa_mask`/`get_block_mask`.
//!      A frame-block spans `frame_seq_length × num_frames_per_block` tokens
//!      ([`KreaArConfig::block_size`](crate::KreaArConfig::block_size)).
//!   2. **Persistent per-layer KV cache** ([`CausalKvCache`]) — post-RoPE keys + raw values retained
//!      across chunks; a new chunk's queries attend over `k[max(0, end - max_attention_size):end]`
//!      (global on this checkpoint: `max_attention_size = seq_length`), plus any always-attended sink
//!      prefix. Mirrors `_initialize_kv_cache` + the cached `CausalWanSelfAttention.forward`. The text
//!      cross-attention cache is the **separate**, position-independent
//!      [`WanTransformer::prepare_cross_kv`] (computed once per prompt — `_initialize_crossattn_cache`).
//!   3. **Causal RoPE temporal offset** ([`WanTransformer::prepare_rope_with_frame_offset`]) — a chunk
//!      that begins at global latent frame `start_frame = current_start / frame_seq_length` indexes the
//!      temporal RoPE band from `start_frame` (the spatial bands are unchanged), so cached keys line up
//!      with a full-sequence pass. Mirrors `causal_rope_apply`.
//!
//! The reused Wan attention gained one additive compute variant for this
//! ([`WanTransformer::forward_causal_chunk`], the AR analogue of `forward_packed`); the cache
//! append/read/windowing, the mask construction, the RoPE-offset driving, and the chunk orchestration
//! live here. S3 is the forward + cache; the Self-Forcing few-step renoise scheduler and AR chunk loop
//! are S4; the clean-context KV recompute + the bounded ([`CausalKvCache`]) window are S5; VAE decode +
//! `Generator` registration are S6/S7. Note the causal structure here is **entirely** the block-causal
//! mask + KV read window + causal RoPE offset — the released reference applies no additive attention
//! bias / `score_mod` in its sampling path (see the crate-root reconciliation note).

use mlx_gen::adapters::{AdaptableHost, AdaptableLinear, DiffPatchPart};
use mlx_gen::{Error, Result};
use mlx_gen_wan::patchify::unpatchify;
use mlx_gen_wan::{normalize_wan_key, WanTransformer};
use mlx_rs::ops::{concatenate_axis, dequantize, quantize};
use mlx_rs::{Array, Dtype};

use crate::config::{KreaRealtimeConfig, KvCacheQuant};

#[cfg(test)]
thread_local! {
    /// Per-test-thread count of full `Sq x Sk` host-mask materializations. A thread-local counter
    /// keeps assertions deterministic under Rust's parallel test runner.
    static MASK_MATERIALIZATIONS: std::cell::Cell<usize> = const { std::cell::Cell::new(0) };
}

#[cfg(test)]
fn mask_materialization_count() -> usize {
    MASK_MATERIALIZATIONS.with(std::cell::Cell::get)
}

/// One layer's cached self-attention `(post-RoPE keys, raw values)`, each `[B, n, S, head_dim]`.
///
/// This is the **dense** form the attention path produces and consumes. The cache may store it
/// group-wise-quantized ([`KvCacheQuant`]) and dequantize on read — see [`CausalKvCache`].
pub type LayerKv = (Array, Array);

/// One group-wise affine-quantized cached tensor: the packed payload plus its per-group scales and
/// biases (sc-17807).
///
/// All three share the cached tensor's **leading** axes `[B, n, S, …]` — packing runs along the last
/// axis (`head_dim`) — so the cache's token-axis (`axis 2`) `concatenate_axis` / `take_axis` algebra
/// applies to each part unchanged, and eviction/windowing need no special case.
#[derive(Clone)]
struct PackedKv {
    /// Packed elements, `[B, n, S, head_dim · bits / 32]` (uint32).
    w: Array,
    /// Per-group scale, `[B, n, S, head_dim / group_size]`, in the quantized input's dtype.
    scales: Array,
    /// Per-group bias, same shape/dtype as [`scales`](Self::scales).
    biases: Array,
}

impl PackedKv {
    /// Pack a dense `[B, n, S, head_dim]` cached tensor at `q`. Errors when `head_dim` is not a
    /// multiple of the group size — MLX's own requirement, surfaced here with the config knob named
    /// rather than as an opaque exception inside a chunk forward.
    fn pack(dense: &Array, q: KvCacheQuant) -> Result<Self> {
        q.validate()?;
        let shape = dense.shape();
        let head_dim = *shape.last().ok_or_else(|| {
            Error::Msg("krea causal: cannot quantize a zero-dimensional KV tensor".into())
        })?;
        if head_dim <= 0 || head_dim % q.group_size != 0 {
            return Err(Error::Msg(format!(
                "krea causal: KV head_dim {head_dim} is not a positive multiple of the \
                 kv_cache_quant group size {} — pick a group size that divides head_dim, or leave \
                 kv_cache_quant unset for the bf16 cache",
                q.group_size
            )));
        }
        let (w, scales, biases) = quantize(dense, q.group_size, q.bits)?;
        Ok(Self { w, scales, biases })
    }

    /// The dense `[B, n, S, head_dim]` tensor, in the dtype that was packed (bf16 on the AR path).
    fn unpack(&self, q: KvCacheQuant) -> Result<Array> {
        Ok(dequantize(
            &self.w,
            &self.scales,
            &self.biases,
            q.group_size,
            q.bits,
        )?)
    }

    /// Concatenate `other`'s tokens after this one's, on the token axis.
    fn concat(&self, other: &Self) -> Result<Self> {
        Ok(Self {
            w: concatenate_axis(&[&self.w, &other.w], 2)?,
            scales: concatenate_axis(&[&self.scales, &other.scales], 2)?,
            biases: concatenate_axis(&[&self.biases, &other.biases], 2)?,
        })
    }

    /// Gather the token-axis positions `idx`.
    fn take(&self, idx: &Array) -> Result<Self> {
        Ok(Self {
            w: self.w.take_axis(idx, 2)?,
            scales: self.scales.take_axis(idx, 2)?,
            biases: self.biases.take_axis(idx, 2)?,
        })
    }

    fn nbytes(&self) -> usize {
        self.w.nbytes() + self.scales.nbytes() + self.biases.nbytes()
    }
}

/// One layer's physically retained K/V, in whichever representation the cache is configured for.
#[derive(Clone)]
enum StoredKv {
    /// The shipped default: post-RoPE keys + raw values as produced by attention (bf16).
    Dense { k: Array, v: Array },
    /// Group-wise affine-quantized K and V ([`KreaArConfig::kv_cache_quant`](crate::KreaArConfig::kv_cache_quant)).
    Packed { k: PackedKv, v: PackedKv },
}

impl StoredKv {
    /// Store a chunk's dense `(k, v)`, packing it when the cache is quantized.
    fn store(kv: LayerKv, quant: Option<KvCacheQuant>) -> Result<Self> {
        let (k, v) = kv;
        match quant {
            None => Ok(Self::Dense { k, v }),
            Some(q) => Ok(Self::Packed {
                k: PackedKv::pack(&k, q)?,
                v: PackedKv::pack(&v, q)?,
            }),
        }
    }

    /// This layer's retained tokens followed by `next`'s, in the same representation.
    fn concat(&self, next: &Self) -> Result<Self> {
        match (self, next) {
            (Self::Dense { k, v }, Self::Dense { k: nk, v: nv }) => Ok(Self::Dense {
                k: concatenate_axis(&[k, nk], 2)?,
                v: concatenate_axis(&[v, nv], 2)?,
            }),
            (Self::Packed { k, v }, Self::Packed { k: nk, v: nv }) => Ok(Self::Packed {
                k: k.concat(nk)?,
                v: v.concat(nv)?,
            }),
            _ => Err(Error::Msg(
                "krea causal: KV cache mixes dense and quantized layer storage".into(),
            )),
        }
    }

    /// Gather the token-axis positions `idx` out of this layer's retained buffer.
    fn take(&self, idx: &Array) -> Result<Self> {
        match self {
            Self::Dense { k, v } => Ok(Self::Dense {
                k: k.take_axis(idx, 2)?,
                v: v.take_axis(idx, 2)?,
            }),
            Self::Packed { k, v } => Ok(Self::Packed {
                k: k.take(idx)?,
                v: v.take(idx)?,
            }),
        }
    }

    /// The dense `(k, v)` the attention path consumes — a handle pair when the cache is dense, a
    /// per-call `dequantize` when it is packed.
    fn dense(&self, quant: Option<KvCacheQuant>) -> Result<LayerKv> {
        match (self, quant) {
            (Self::Dense { k, v }, _) => Ok((k.clone(), v.clone())),
            (Self::Packed { k, v }, Some(q)) => Ok((k.unpack(q)?, v.unpack(q)?)),
            (Self::Packed { .. }, None) => Err(Error::Msg(
                "krea causal: KV cache holds quantized layers but carries no quantization tier"
                    .into(),
            )),
        }
    }

    /// Bytes physically retained for this layer — the **stored** representation, so a quantized cache
    /// reports its packed cost rather than what it would cost dequantized.
    fn nbytes(&self) -> usize {
        match self {
            Self::Dense { k, v } => k.nbytes() + v.nbytes(),
            Self::Packed { k, v } => k.nbytes() + v.nbytes(),
        }
    }
}

/// End of the frame-block containing global token index `q` (exclusive): `(q / block + 1) * block`.
/// The widened result represents the final boundary past `i64::MAX` exactly instead of overflowing;
/// i128 division keeps Rust's existing truncation-toward-zero behavior for negative test positions.
#[inline]
fn end_of_block(q: i64, block_size: i64) -> i128 {
    let q = i128::from(q);
    let block_size = i128::from(block_size);
    (q / block_size + 1) * block_size
}

fn checked_block_size(block_size: usize) -> Result<i64> {
    let block_size = i64::try_from(block_size)
        .map_err(|_| Error::Msg("krea causal mask: block_size exceeds i64::MAX".into()))?;
    if block_size == 0 {
        return Err(Error::Msg(
            "krea causal mask: block_size must be greater than zero".into(),
        ));
    }
    Ok(block_size)
}

#[inline]
fn is_masked(q: i64, kv: i64, block_size: i64) -> bool {
    !(i128::from(kv) < end_of_block(q, block_size) || q == kv)
}

/// Decide whether the scalar block-causal rule masks any `(query, key)` pair without building the
/// `Sq x Sk` matrix. For each query only the largest key other than that query can matter: if it is
/// below the query block's end, every smaller key is allowed; otherwise that largest key is masked.
/// Tracking the two largest distinct key positions preserves the rule's `q == kv` exception.
fn mask_is_needed(q_pos: &[i64], kv_pos: &[i64], block_size: usize) -> Result<bool> {
    let bs = checked_block_size(block_size)?;
    let (mut largest, mut second_largest) = (None, None);
    for &kv in kv_pos {
        match largest {
            None => largest = Some(kv),
            Some(max) if kv > max => {
                second_largest = Some(max);
                largest = Some(kv);
            }
            Some(max) if kv < max && second_largest.is_none_or(|second| kv > second) => {
                second_largest = Some(kv);
            }
            Some(_) => {}
        }
    }

    Ok(q_pos.iter().any(|&q| {
        let largest_other = match largest {
            Some(max) if max != q => Some(max),
            Some(_) => second_largest,
            None => None,
        };
        largest_other.is_some_and(|kv| is_masked(q, kv, bs))
    }))
}

/// The raw additive block-causal mask data for query global positions `q_pos` against key global
/// positions `kv_pos` (block size `block_size` tokens), plus whether **any** entry is masked. Entry
/// `(i, j)` is `0.0` when `kv_pos[j] < end_of_block(q_pos[i]) || q_pos[i] == kv_pos[j]`, else `-inf`.
fn mask_data(q_pos: &[i64], kv_pos: &[i64], block_size: usize) -> Result<(Vec<f32>, bool)> {
    let bs = checked_block_size(block_size)?;
    #[cfg(test)]
    MASK_MATERIALIZATIONS.with(|count| count.set(count.get() + 1));
    let (sq, sk) = (q_pos.len(), kv_pos.len());
    let mut data = vec![0f32; sq * sk];
    let mut any_masked = false;
    for (i, &q) in q_pos.iter().enumerate() {
        for (j, &kv) in kv_pos.iter().enumerate() {
            if is_masked(q, kv, bs) {
                data[i * sk + j] = f32::NEG_INFINITY;
                any_masked = true;
            }
        }
    }
    Ok((data, any_masked))
}

/// Build the **additive** block-causal SDPA mask (bf16) for query global token positions `q_pos`
/// against key global token positions `kv_pos`, with `block_size` tokens per frame-block. Shape
/// `[Sq, Sk]` — broadcasts over the batch/head axes in [`mlx_rs::fast::scaled_dot_product_attention`].
/// `0.0` where a query may attend the key, `-inf` where it may not. Mirrors
/// `causal_model.py::get_sdpa_mask`/`get_block_mask`. See [`block_causal_mask`] for the forward path's
/// "all-allowed ⇒ no mask" optimization.
pub fn build_block_causal_mask(q_pos: &[i64], kv_pos: &[i64], block_size: usize) -> Result<Array> {
    let (data, _) = mask_data(q_pos, kv_pos, block_size)?;
    let arr = Array::from_slice(&data, &[q_pos.len() as i32, kv_pos.len() as i32]);
    Ok(arr.as_dtype(Dtype::Bfloat16)?)
}

/// Like [`build_block_causal_mask`] but returns `None` when **every** query may attend **every** key
/// (the additive mask would be all-zeros) — the common single-block AR step, where passing no mask lets
/// SDPA take its unmasked fast path (adding an all-zero mask is a numeric no-op). `Some(mask)` otherwise
/// (e.g. a multi-block full-recompute pass).
pub fn block_causal_mask(
    q_pos: &[i64],
    kv_pos: &[i64],
    block_size: usize,
) -> Result<Option<Array>> {
    if !mask_is_needed(q_pos, kv_pos, block_size)? {
        return Ok(None);
    }
    let (data, any_masked) = mask_data(q_pos, kv_pos, block_size)?;
    debug_assert!(any_masked, "analytic and scalar mask decisions diverged");
    let arr = Array::from_slice(&data, &[q_pos.len() as i32, kv_pos.len() as i32]);
    Ok(Some(arr.as_dtype(Dtype::Bfloat16)?))
}

/// Persistent per-layer self-attention KV cache for the Krea Realtime AR forward (sc-8436 S3; bounded
/// window + clean-context recompute in sc-8438 S5).
///
/// Stores, per transformer layer, the running **post-RoPE keys + raw values** accumulated across the
/// chunks generated so far, in two regimes selected by the read-window geometry:
///
///   * **Global** (`max_attention_size ≥` the whole clip — the shipped `local_attn_size = -1`
///     checkpoint, where `max_attention_size = seq_length`, `sink = 0`) — every key is retained and the
///     read window is the whole cache. Byte-for-byte the S3/S4 behaviour.
///   * **Bounded / streaming** (a finite `local_attn_size`, the Mac memory-feasible path) — physical
///     storage is **capped**: only the always-attended sink prefix `[0, sink_tokens)` plus the
///     most-recent `max_attention_size` tokens are kept; older tail K/V are evicted on
///     [`append`](Self::append) so a long clip does not grow KV without bound. Mirrors the reference's
///     rolling KV buffer (`causal_model.py::CausalWanSelfAttention.forward`, the `local_attn_size != -1`
///     roll at lines 363-385) — pure cache slicing, no VAE. The first-frame VAE re-anchor
///     (`release_server.py::get_clean_context_frames` re-encoding the first output frame) is a
///     *separate* mechanism, deferred to S6.
///
/// The running global token count is [`stored_tokens`](Self::stored_tokens) (never shrinks — it is the
/// reference's `global_end_index` / `current_start`); the physically retained length
/// ([`retained_tokens`](Self::retained_tokens)) may be smaller once eviction begins. The **read window**
/// `k[max(0, end - max_attention_size):end]` (plus the sink prefix) is applied on read by
/// [`window_prev`](Self::window_prev).
///
/// **Per-token cost (sc-17807).** The cache holds *activations*, so the DiT's weight tier does not
/// shrink it: at the Wan-14B geometry a token of KV is `2 (K and V) × 40 layers × 5120 dim × 2 bytes`
/// = **800 KiB**, which for an autoregressive video model dominates the ~9 GiB of Q4 weights. The
/// storage representation is therefore a knob:
/// [`KreaArConfig::kv_cache_quant`](crate::KreaArConfig::kv_cache_quant) selects group-wise affine
/// quantization ([`KvCacheQuant`]) and the cache stores packed K/V, **dequantizing the read window
/// per layer** in [`window_prev`](Self::window_prev). Attention still runs the dense fused SDPA —
/// consuming a packed cache directly would mean the decomposed quantized-matmul form, whose per-layer
/// `Sq × Sk` score matrix costs ~37× the dequantized window at AR-video chunk widths (and, for one
/// layer, as much as the whole cache's Q8 saving) — see [`KvCacheQuant`].
/// The dequantized window is an anonymous graph intermediate, dropped before the AR loop's per-step
/// `eval`, so what stays resident across chunks is the packed cache.
/// [`retained_bytes`](Self::retained_bytes) reports the **stored** cost either way.
pub struct CausalKvCache {
    /// Per layer: `Some(stored)` once populated — the physically retained tokens in global order (the
    /// sink prefix `[0, sink_kept)` followed by the rolling tail `[tail_base, stored_tokens)`), each
    /// `[B, n, retained_tokens, d]` (post-RoPE k, raw v), dense or packed per [`quant`](Self::quant).
    layers: Vec<Option<StoredKv>>,
    /// Running global token count committed so far (the reference's `global_end_index`); grows by the
    /// chunk length on every [`append`](Self::append) and never shrinks, even under eviction.
    committed_tokens: usize,
    /// Global position of the first physically-retained **tail** token. Equals `sink_kept` until
    /// eviction rolls it forward; tokens in `[sink_kept, tail_base)` have been evicted from storage.
    tail_base: usize,
    max_attention_size: usize,
    sink_tokens: usize,
    /// Storage tier for the retained K/V: `None` = bf16 (the shipped default), `Some(q)` = group-wise
    /// affine-quantized, dequantized on read (sc-17807).
    quant: Option<KvCacheQuant>,
}

impl CausalKvCache {
    /// An empty cache for `num_layers` transformer blocks with the given read-window geometry (in
    /// tokens) and storage tier. See
    /// [`KreaArConfig::max_attention_size`](crate::KreaArConfig::max_attention_size) /
    /// [`sink_tokens`](crate::KreaArConfig::sink_tokens) /
    /// [`kv_cache_quant`](crate::KreaArConfig::kv_cache_quant).
    ///
    /// `quant` is an explicit parameter rather than a defaulted one so a caller cannot silently get
    /// the bf16 cache while its config asks for a quantized one — the failure mode would be a
    /// correct-looking run at 1.9× the intended memory.
    pub fn new(
        num_layers: usize,
        max_attention_size: usize,
        sink_tokens: usize,
        quant: Option<KvCacheQuant>,
    ) -> Self {
        Self {
            layers: (0..num_layers).map(|_| None).collect(),
            committed_tokens: 0,
            tail_base: 0,
            max_attention_size,
            sink_tokens,
            quant,
        }
    }

    /// The storage tier this cache retains K/V at — `None` for the shipped bf16 cache.
    pub fn quant(&self) -> Option<KvCacheQuant> {
        self.quant
    }

    /// Bytes **physically retained** right now, summed over every layer, in the representation the
    /// cache actually stores. A quantized cache reports its packed cost, not what its contents would
    /// occupy dequantized — which is the whole point of measuring it.
    pub fn retained_bytes(&self) -> usize {
        self.layers
            .iter()
            .flatten()
            .map(StoredKv::nbytes)
            .sum::<usize>()
    }

    /// Force MLX to materialize the retained buffers **as stored** (packed, when the cache is
    /// quantized).
    ///
    /// MLX is lazy: [`append`](Self::append)'s concat (and eviction's gather) build graph nodes, and
    /// until something evaluates them the cache occupies no allocator bytes. A residency measurement
    /// taken without this is a shape calculation. Evaluating [`layer_kv`](Self::layer_kv) instead
    /// would materialize the *dequantized* copies and leave the packed cache itself unevaluated —
    /// which is precisely the wrong thing to measure.
    pub fn eval_retained(&self) -> Result<()> {
        let mut arrays: Vec<&Array> = Vec::with_capacity(self.layers.len() * 6);
        for stored in self.layers.iter().flatten() {
            match stored {
                StoredKv::Dense { k, v } => arrays.extend([k, v]),
                StoredKv::Packed { k, v } => {
                    arrays.extend([&k.w, &k.scales, &k.biases, &v.w, &v.scales, &v.biases]);
                }
            }
        }
        mlx_rs::transforms::eval(arrays)?;
        Ok(())
    }

    /// Number of tokens (per layer) **committed** so far (the global running end) — grows by the chunk
    /// length on each [`append`](Self::append) and never shrinks. Mirrors the reference's running
    /// `current_start` / `global_end_index`.
    pub fn stored_tokens(&self) -> usize {
        self.committed_tokens
    }

    /// Number of tokens (per layer) **physically retained** right now: the sink prefix plus the rolling
    /// tail. Equals [`stored_tokens`](Self::stored_tokens) in the global regime; smaller once the bounded
    /// window starts evicting older tail K/V.
    pub fn retained_tokens(&self) -> usize {
        self.sink_kept() + (self.committed_tokens - self.tail_base)
    }

    /// Number of transformer layers this cache serves.
    pub fn num_layers(&self) -> usize {
        self.layers.len()
    }

    /// `true` before the first chunk is appended.
    pub fn is_empty(&self) -> bool {
        self.committed_tokens == 0
    }

    /// The physically-retained sink-prefix length (`min(sink_tokens, stored)`).
    #[inline]
    fn sink_kept(&self) -> usize {
        self.sink_tokens.min(self.committed_tokens)
    }

    /// The retained `(k, v)` per layer as **dense** post-RoPE k / raw v, for verification (S5
    /// recompute / bounded-window tests) and the S6 pipeline. `Ok(None)` before the first append or
    /// for an unknown layer.
    ///
    /// Returns handles to the stored arrays when the cache is dense, and dequantizes when it is
    /// quantized — so a caller reading it back always sees the values attention sees. Use
    /// [`retained_bytes`](Self::retained_bytes), not the returned arrays' `nbytes`, to measure
    /// residency: the dequantized form is *not* what a quantized cache costs.
    pub fn layer_kv(&self, layer: usize) -> Result<Option<LayerKv>> {
        self.layers
            .get(layer)
            .and_then(|l| l.as_ref())
            .map(|stored| stored.dense(self.quant))
            .transpose()
    }

    /// The **global token positions** of the read window's cached (prev) keys for a query chunk of
    /// `s_new` new tokens: the always-attended sink prefix `[0, min(sink, stored))` unioned with the
    /// sliding tail `[read_start, stored)`, where
    /// `read_start = max(sink, (stored + s_new) − max_attention_size)` (clamped to `stored`). These are
    /// exactly the prev tokens the new chunk's queries attend, and the caller uses them to build the
    /// matching mask column positions. Every returned position is physically retained (the eviction
    /// invariant guarantees `tail_base ≤ read_start`). Empty before the first append.
    fn window_positions(&self, s_new: usize) -> Vec<i64> {
        if self.committed_tokens == 0 {
            return Vec::new();
        }
        let stored = self.committed_tokens;
        let current_end = stored + s_new;
        let sink_end = self.sink_tokens.min(stored);
        // Sliding-window start; saturating so a window larger than history keeps everything.
        let read_start = self
            .sink_tokens
            .max(current_end.saturating_sub(self.max_attention_size))
            .min(stored);
        let tail_start = read_start.max(sink_end);
        let mut pos: Vec<i64> = Vec::with_capacity(sink_end + stored.saturating_sub(tail_start));
        pos.extend((0..sink_end).map(|p| p as i64));
        pos.extend((tail_start..stored).map(|p| p as i64));
        pos
    }

    /// The **physical** index (into the retained buffer) of a currently-retained global token position
    /// `g`: `g < sink_kept` maps to itself; a tail position `g ≥ tail_base` maps past the sink prefix.
    #[inline]
    fn phys_index(&self, g: usize) -> usize {
        let sink_kept = self.sink_kept();
        if g < sink_kept {
            g
        } else {
            sink_kept + (g - self.tail_base)
        }
    }

    /// The windowed cached `(k, v)` per layer for a query chunk of `s_new` new tokens, alongside the
    /// **global token positions** of those keys (for mask construction). Returns `(vec![], vec![])`
    /// before the first append (first chunk has no history). When the window is exactly the whole
    /// retained buffer (the global / fits case) no gather is issued; otherwise the physically-indexed
    /// window is gathered.
    ///
    /// The returned pairs are always **dense** — a quantized cache gathers packed and dequantizes here
    /// (sc-17807), so the reused Wan attention keeps its fused SDPA path. Gathering *before*
    /// dequantizing is what makes it pay: only the window is widened, and only for as long as the
    /// caller holds it (one chunk forward, dropped before the AR loop's per-step `eval`, so MLX frees
    /// each layer's copy as that layer's attention completes rather than holding all `num_layers` at
    /// once).
    ///
    /// One asymmetry worth naming: in the **global** regime the read window *is* the whole retained
    /// buffer, so the dense path returns handles with no copy at all while the quantized path still
    /// dequantizes. The transient is per layer and bounded by the window, and it is far smaller than
    /// the packed cache's saving over a global-window clip — but it is not zero, and it is the reason
    /// the tier's value grows with how much of the cache is *retained* rather than read.
    pub fn window_prev(&self, s_new: usize) -> Result<(Vec<LayerKv>, Vec<i64>)> {
        let positions = self.window_positions(s_new);
        if positions.is_empty() {
            return Ok((Vec::new(), Vec::new()));
        }
        let phys: Vec<i32> = positions
            .iter()
            .map(|&g| self.phys_index(g as usize) as i32)
            .collect();
        // Reading the entire retained buffer in order ⇒ no gather (the stored buffers are the window).
        let whole = phys.len() == self.retained_tokens()
            && phys.iter().enumerate().all(|(i, &p)| p as usize == i);
        let idx = if whole {
            None
        } else {
            Some(Array::from_slice(&phys, &[phys.len() as i32]))
        };
        let mut prev = Vec::with_capacity(self.layers.len());
        for layer in &self.layers {
            let stored = layer.as_ref().ok_or_else(|| {
                Error::Msg("krea causal: KV cache has tokens but a layer slot is empty".into())
            })?;
            let windowed = match &idx {
                None => stored.dense(self.quant)?,
                Some(ix) => stored.take(ix)?.dense(self.quant)?,
            };
            prev.push(windowed);
        }
        Ok((prev, positions))
    }

    /// Append this chunk's per-layer post-RoPE `(k, v)` `[B, n, s_new, d]` to the running cache. In the
    /// global regime this is full retention (the read window is applied on
    /// [`window_prev`](Self::window_prev)); in the bounded regime the oldest tail tokens beyond
    /// `sink_tokens + max_attention_size` are **evicted** from physical storage so a long clip stays
    /// bounded. `new_kv` must carry exactly one `(k, v)` per layer.
    pub fn append(&mut self, new_kv: Vec<LayerKv>) -> Result<()> {
        if new_kv.len() != self.layers.len() {
            return Err(Error::Msg(format!(
                "krea causal: append expected {} layers, got {}",
                self.layers.len(),
                new_kv.len()
            )));
        }
        // Concatenate this chunk's K/V onto the physical tail (each layer identically). When the cache
        // is quantized the chunk is packed **here**, at commit — once per chunk, not once per denoise
        // step, since only the committing forward reaches `append` (sc-17807).
        let mut s_new = 0usize;
        for (slot, kv) in self.layers.iter_mut().zip(new_kv) {
            s_new = kv.0.shape()[2] as usize;
            let incoming = StoredKv::store(kv, self.quant)?;
            *slot = Some(match slot.take() {
                None => incoming,
                Some(prev) => prev.concat(&incoming)?,
            });
        }

        let tail_base_old = self.tail_base;
        let committed_before = self.committed_tokens;
        self.committed_tokens += s_new;

        // Bounded window: evict the oldest tail tokens beyond the sink prefix + read window. In the
        // global regime `max_attention_size ≥ committed`, so `tail_base_new` stays at `sink_kept` and
        // nothing is dropped (identical to full retention).
        let sink_kept = self.sink_kept();
        let tail_base_new = sink_kept.max(
            self.committed_tokens
                .saturating_sub(self.max_attention_size),
        );
        // Only a *real* tail drop (past the sink prefix) needs a re-gather; advancing `tail_base` while
        // the sink is still filling (tail empty) is a no-op.
        let drop_start = tail_base_old.max(sink_kept);
        if tail_base_new > drop_start {
            // Physical indices to KEEP in the current (**pre-eviction**) buffer. That buffer's layout
            // is the sink prefix `[0, sink_kept_prev)` followed by the rolling tail
            // `[tail_base_old, committed)`, where `sink_kept_prev` is the sink length *before* this
            // append — the physical layout predates any sink growth contributed by this chunk. The
            // desired keep set is the (possibly grown) sink prefix `[0, sink_kept)` ‖ retained tail
            // `[tail_base_new, committed)`, but each global position must be mapped through the
            // pre-eviction sink length: using the post-append `sink_kept` here would overshoot the
            // tail's true physical start by `sink_kept - sink_kept_prev` whenever the sink is still
            // filling on the evicting append (e.g. one large first chunk), running the gather past the
            // axis (`take`/`take_axis` is not bounds-checked on Metal → silent KV corruption). Once the
            // sink is full `sink_kept_prev == sink_kept`, so this is byte-for-byte the steady state.
            let sink_kept_prev = self.sink_tokens.min(committed_before);
            let keep: Vec<i32> = (0..sink_kept)
                .chain(tail_base_new..self.committed_tokens)
                .map(|g| {
                    let p = if g < sink_kept_prev {
                        g
                    } else {
                        sink_kept_prev + (g - tail_base_old)
                    };
                    p as i32
                })
                .collect();
            let idx = Array::from_slice(&keep, &[keep.len() as i32]);
            for slot in self.layers.iter_mut() {
                if let Some(stored) = slot.as_ref() {
                    *slot = Some(stored.take(&idx)?);
                }
            }
        }
        self.tail_base = tail_base_new;
        Ok(())
    }
}

/// The Krea Realtime 14B **causal autoregressive** transformer: the reused Wan 2.1 14B DiT
/// ([`WanTransformer`]) plus the AR self-attention regime (block-causal mask + KV cache + causal RoPE
/// offset). Load the DiT with [`load_krea_realtime_transformer`](crate::load_krea_realtime_transformer),
/// then wrap it here. S3 exposes the per-chunk forward + cache; the Self-Forcing scheduler / AR chunk
/// loop (S4), KV-cache recompute (S5), and VAE/`Generator` (S6/S7) are not part of this crate yet.
pub struct CausalKreaTransformer {
    inner: WanTransformer,
    frame_seq_length: usize,
    block_size: usize,
    max_attention_size: usize,
    sink_tokens: usize,
    kv_cache_quant: Option<KvCacheQuant>,
    num_layers: usize,
    out_dim: usize,
    patch_size: (usize, usize, usize),
}

impl CausalKreaTransformer {
    /// Wrap a loaded Wan DiT with the Krea Realtime AR config. The DiT is loaded (weight-for-weight
    /// Wan 2.1 14B) via [`load_krea_realtime_transformer`](crate::load_krea_realtime_transformer); this
    /// only layers the AR self-attention regime on top.
    pub fn new(inner: WanTransformer, cfg: &KreaRealtimeConfig) -> Self {
        Self {
            frame_seq_length: cfg.ar.frame_seq_length,
            block_size: cfg.ar.block_size(),
            max_attention_size: cfg.ar.max_attention_size(),
            sink_tokens: cfg.ar.sink_tokens(),
            kv_cache_quant: cfg.ar.kv_cache_quant,
            num_layers: cfg.wan.num_layers,
            out_dim: cfg.wan.out_dim,
            patch_size: cfg.wan.patch_size,
            inner,
        }
    }

    /// The reused Wan DiT (e.g. for its stock non-causal [`WanTransformer::forward_packed`] anchor).
    pub fn inner(&self) -> &WanTransformer {
        &self.inner
    }

    /// Tokens per frame-block (the block-causal unit).
    pub fn block_size(&self) -> usize {
        self.block_size
    }

    /// A fresh, empty [`CausalKvCache`] sized to this model (layers + read-window geometry) and
    /// carrying the config's KV storage tier
    /// ([`KreaArConfig::kv_cache_quant`](crate::KreaArConfig::kv_cache_quant)).
    pub fn new_cache(&self) -> CausalKvCache {
        CausalKvCache::new(
            self.num_layers,
            self.max_attention_size,
            self.sink_tokens,
            self.kv_cache_quant,
        )
    }

    /// Precompute the per-block text cross-attention K/V once per prompt — the **separate**
    /// (position-independent) cross-attention cache. `context_batch`: `[B, text_len, dim]` (bf16) from
    /// [`WanTransformer::embed_text`]. Mirrors `_initialize_crossattn_cache`.
    pub fn prepare_cross_kv(&self, context_batch: &Array) -> Result<Vec<(Array, Array)>> {
        self.inner.prepare_cross_kv(context_batch)
    }

    /// Denoise one autoregressive chunk `latent_chunk` `[C, F_chunk, H, W]` (f32) at timestep `t`,
    /// attending over the persistent `cache` (earlier chunks) plus itself, and appending this chunk's
    /// self-attention K/V to `cache`. `current_start_token` is the global token index where this chunk
    /// begins (must equal `cache.stored_tokens()` and be frame-aligned); it fixes the causal RoPE frame
    /// offset (`start_frame = current_start_token / frame_seq_length`). `cross_kv` is the per-prompt
    /// cross-attention cache from [`prepare_cross_kv`](Self::prepare_cross_kv). Returns the denoised
    /// velocity `[out_dim, F_chunk, H, W]` (f32).
    ///
    /// This is the S3 forward: it does **not** renoise / few-step / advance a schedule (S4). The block
    /// size means one AR chunk of `num_frames_per_block` frames is exactly one attention block, so the
    /// in-step mask is all-allowed (no mask); the cache + RoPE offset carry the causal structure.
    pub fn forward_chunk(
        &self,
        latent_chunk: &Array,
        t: f32,
        cross_kv: &[(Array, Array)],
        current_start_token: usize,
        cache: &mut CausalKvCache,
    ) -> Result<Array> {
        let (velocity, new_kv) =
            self.denoise_chunk_inner(latent_chunk, t, cross_kv, current_start_token, cache)?;
        cache.append(new_kv)?;
        Ok(velocity)
    }

    /// Like [`forward_chunk`](Self::forward_chunk) but **does not append** this chunk's self-attention
    /// K/V to `cache` — the read-only denoise forward the S4 few-step loop uses for every denoising
    /// step *except* the last. Every step of one AR chunk attends the **same** committed
    /// previous-chunk window (the cache is untouched by the intermediate steps), so only the final
    /// (near-clean) step commits its K/V via [`forward_chunk`](Self::forward_chunk) and the cache grows
    /// by exactly one chunk per chunk. `current_start_token` must still equal `cache.stored_tokens()`
    /// (the chunk begins where the committed history ends) and be frame-aligned; it fixes the causal
    /// RoPE frame offset identically to [`forward_chunk`](Self::forward_chunk). Returns the denoised
    /// velocity `[out_dim, F_chunk, H, W]` (f32).
    pub fn forward_chunk_readonly(
        &self,
        latent_chunk: &Array,
        t: f32,
        cross_kv: &[(Array, Array)],
        current_start_token: usize,
        cache: &CausalKvCache,
    ) -> Result<Array> {
        let (velocity, _new_kv) =
            self.denoise_chunk_inner(latent_chunk, t, cross_kv, current_start_token, cache)?;
        Ok(velocity)
    }

    /// Shared per-chunk causal denoise forward: validate the chunk start, patch-embed the chunk, build
    /// the offset RoPE, window the cache, assemble the block-causal mask over `[prev-window ‖ this-chunk]`,
    /// run [`WanTransformer::forward_causal_chunk`], and unpatchify. Returns the denoised velocity
    /// `[out_dim, F_chunk, H, W]` (f32) **and** this chunk's per-layer post-RoPE self-attention `(k, v)`
    /// for the caller to append or discard — it does **not** mutate `cache`. The two public entries
    /// differ only in whether they commit `new_kv`: [`forward_chunk`](Self::forward_chunk) appends,
    /// [`forward_chunk_readonly`](Self::forward_chunk_readonly) drops it.
    fn denoise_chunk_inner(
        &self,
        latent_chunk: &Array,
        t: f32,
        cross_kv: &[(Array, Array)],
        current_start_token: usize,
        cache: &CausalKvCache,
    ) -> Result<(Array, Vec<LayerKv>)> {
        if current_start_token != cache.stored_tokens() {
            return Err(Error::Msg(format!(
                "krea causal: current_start_token {current_start_token} must equal the cache's stored \
                 tokens {} (chunks are contiguous)",
                cache.stored_tokens()
            )));
        }
        if !current_start_token.is_multiple_of(self.frame_seq_length) {
            return Err(Error::Msg(format!(
                "krea causal: current_start_token {current_start_token} is not frame-aligned \
                 (frame_seq_length {})",
                self.frame_seq_length
            )));
        }

        let (tokens, grid) = self.inner.patch_embed_tokens(latent_chunk)?;
        let s_new = grid.0 * grid.1 * grid.2;
        let start_frame = current_start_token / self.frame_seq_length;
        let (cos, sin) = self
            .inner
            .prepare_rope_with_frame_offset(grid, start_frame)?;

        // Windowed cache + the block-causal mask over [prev-window ‖ this-chunk].
        let (prev_kv, prev_positions) = cache.window_prev(s_new)?;
        let q_positions: Vec<i64> =
            (current_start_token as i64..(current_start_token + s_new) as i64).collect();
        let mut kv_positions = prev_positions;
        kv_positions.extend(q_positions.iter().copied());
        let mask = block_causal_mask(&q_positions, &kv_positions, self.block_size)?;

        let (velocity, new_kv) = self.inner.forward_causal_chunk(
            &tokens,
            t,
            cross_kv,
            &cos,
            &sin,
            &prev_kv,
            mask.as_ref(),
        )?;

        // Unpatchify the per-token velocity [1, S, out_dim·∏patch] → [out_dim, F_chunk, H, W].
        let op = velocity.shape()[2];
        let xb = velocity.reshape(&[s_new as i32, op])?;
        Ok((
            unpatchify(&xb, grid, self.out_dim, self.patch_size)?,
            new_kv,
        ))
    }
}

/// Every LoRA-adaptable target in the Krea Realtime DiT as a dotted path in the **Wan reference /
/// diffusers** naming a Wan-family LoRA file carries (once its `diffusion_model.`/`transformer.`
/// namespace is stripped), for a model with `num_layers` blocks. Krea Realtime 14B *is* Wan-2.1-14B
/// T2V weight-for-weight, so these are exactly the per-block attention + FFN Linears a Wan-family
/// style LoRA (musubi-tuner / diffusion-pipe / ComfyUI) targets. The reference spells the FFN as
/// `ffn.0`/`ffn.2` (the converted DiT renames them to `ffn.fc1`/`ffn.fc2`); [`normalize_wan_key`]
/// bridges that in [`CausalKreaTransformer::adaptable_mut`], so this file-naming surface stays the one
/// a LoRA file speaks. Single source of truth for [`AdaptableHost::adaptable_paths`] (the kohya
/// `flattened → dotted` table), kept in lock-step with the resolver by tests.
///
/// **The globals decision (sc-8446, S13) — SETTLED: widen, don't soft-skip.** The seven whole-model
/// Linears (`patch_embedding`, `text_embedding.0/.2`, `time_embedding.0/.2`, `time_projection.1`,
/// `head.head`) ARE exposed, in the same reference/file spelling a LoRA carries, and route to the reused
/// inner [`WanTransformer`]'s own fields via [`WanTransformer::global_adaptable_mut`] — matching the
/// SCAIL-2 host, which has always exposed its globals.
///
/// Settled against the safetensors headers of real published Wan LoRA files rather than a guess:
/// * **Plain style LoRAs** (`shauray/Origami_WanLora`, `motimalu/wan-flat-color-v2`) carry exactly
///   `num_layers × 10` per-block stems and **no** globals — they loaded on the pre-widening surface and
///   are unaffected by it.
/// * **Step-distill / lightning LoRAs for this very backbone** (`lightx2v` Wan2.1-T2V-14B
///   cfg-step-distill v2 and `FastWan` T2V-14B — headers read, structurally identical) carry genuine
///   `lora_down`/`lora_up` factors for **six** of the seven: `text_embedding.0/.2`,
///   `time_embedding.0/.2`, `time_projection.1`, `head.head`. **`patch_embedding` carries only a
///   `.diff_b` bias delta, no low-rank pair** — which is exactly why a real install reports **406**
///   targets (400 per-block + 6) against a 407-wide surface, not 407. `patch_embedding` stays exposed
///   because the surface is defined by what the model *has*, not by what one file happens to populate.
///   On the narrow surface `apply_adapters_strict` rejected the whole file; soft-skipping the globals
///   instead would have *silently* installed a step-distill LoRA with its text/time/output projections
///   missing — a wrong render that still looks like a success, precisely the failure mode the strict
///   installer exists to prevent. Widening applies them.
///
/// **The other 647 keys — RESOLVED in sc-15326.** Widening the *Linear* surface alone did not make a
/// step-distill file fully applied: the same lightx2v/FastWan file carries 647 keys the low-rank pass
/// does not consume — 447 `.diff_b` bias deltas (including `patch_embedding`'s) and 200 `.diff` weight
/// deltas on the qk/`norm3` **norms**, which are not `AdaptableLinear`s at any surface width — and Krea
/// used to drop them without a word. `t2v::load_transformer` now calls
/// [`apply_adapters_strict_with_diff_patch`](mlx_gen::adapters::loader::apply_adapters_strict_with_diff_patch),
/// the `.diff_b` deltas fold into the Linears' (always-dense) biases, and the norm deltas fold through
/// [`AdaptableHost::diff_patch_param_mut`] — all 647 land, on every tier. So "a step-distill LoRA
/// installs" now means the whole file installs, and `ApplyReport::diff_patch_unapplied` reports
/// anything that ever does not.
///
/// Still deliberately absent, and still a loud error: the I2V-only image cross-attention
/// (`cross_attn.k_img`/`v_img`, which `Remade-AI/Squish`-style Wan-I2V LoRAs carry). Krea Realtime is the
/// **T2V** backbone — those modules do not exist here at any surface width, so erroring is the honest
/// answer, not a gap.
pub(crate) fn krea_adaptable_paths(num_layers: usize) -> Vec<String> {
    let mut paths: Vec<String> = KREA_GLOBAL_ADAPTABLE_PATHS
        .iter()
        .map(|s| (*s).to_string())
        .collect();
    paths.reserve(num_layers * 10);
    for i in 0..num_layers {
        for attn in ["self_attn", "cross_attn"] {
            for proj in ["q", "k", "v", "o"] {
                paths.push(format!("blocks.{i}.{attn}.{proj}"));
            }
        }
        paths.push(format!("blocks.{i}.ffn.0"));
        paths.push(format!("blocks.{i}.ffn.2"));
    }
    paths
}

/// The whole-model adaptable targets in the **reference / LoRA-file** spelling (what a Wan-family LoRA
/// actually names), the counterpart of [`mlx_gen_wan::WAN_GLOBAL_ADAPTABLE_PATHS`]'s converted spelling.
/// [`normalize_wan_key`] is the bridge between the two, so this list and that one must stay in
/// correspondence — pinned by `krea_global_paths_normalize_onto_the_wan_globals`.
pub(crate) const KREA_GLOBAL_ADAPTABLE_PATHS: &[&str] = &[
    "patch_embedding",
    "text_embedding.0",
    "text_embedding.2",
    "time_embedding.0",
    "time_embedding.2",
    "time_projection.1",
    "head.head",
];

/// Install inference LoRA(s) onto the Krea Realtime DiT as forward-time residuals (sc-15015, S14).
/// Krea Realtime 14B is Wan-2.1-14B T2V weight-for-weight, so the family-agnostic
/// [`mlx_gen::adapters::loader`] path resolves a diffusers / PEFT / kohya / LoKr / LoHa file directly
/// against the DiT's module names — the same residual install the Z-Image / Qwen / SCAIL-2 providers
/// use. One deliberate difference from SCAIL-2, so "follows the SCAIL-2 template" is not read as
/// "identical to it": Krea's DiT is the *converted* [`WanTransformer`] (whose adaptable surface names the
/// FFN `ffn.fc1`/`ffn.fc2` and the globals `text_embedding_0`/`patch_embedding_proj`/…), so a
/// reference-named key is normalized to the converted layout via the shared Wan key-normalizer before
/// routing. The **surface width now matches** SCAIL-2's: per-block Linears plus the whole-model globals
/// (sc-8446 S13 settled this against real published Wan LoRAs — see `krea_adaptable_paths`).
///
/// Adapters apply as a forward-time residual (`base(x) + scale·x·A·B`) rather than folding into the
/// weights, which makes them **tier-agnostic** (sc-15203): the identical install runs over a dense bf16
/// base and over a packed Q4/Q8 one, with the packed base never dequantized — the additive-on-packed
/// property epic 10043 / sc-10578 established for the Wan family, delivered here by the shared
/// [`AdaptableLinear`] rather than by a separate `apply_wan_adapters_additive` path. `t2v` therefore
/// installs adapters *after* any quantization, so the residual is a dense add over the quantized matmul.
impl AdaptableHost for CausalKreaTransformer {
    fn adaptable_mut(&mut self, path: &[&str]) -> Option<&mut AdaptableLinear> {
        // Normalize the Wan reference / diffusers key to the inner converted [`WanTransformer`] layout
        // (`ffn.0`→`ffn.fc1`, `text_embedding.0`→`text_embedding_0`, `patch_embedding`→
        // `patch_embedding_proj`, …), then route: per-block first, then the whole-model globals
        // (sc-8446 S13 — see `krea_adaptable_paths` for why they are exposed). `q/k/v/o` pass through
        // the normalizer unchanged.
        let dotted = path.join(".");
        let native = normalize_wan_key(&dotted);
        let parts: Vec<&str> = native.split('.').collect();
        // Disjoint by first segment, so this is a single borrow rather than a try-then-fall-back (which
        // the borrow checker would reject for two `&mut self.inner` lookups in one expression).
        if parts.first() == Some(&"blocks") {
            self.inner.adaptable_mut(&parts)
        } else {
            self.inner.global_adaptable_mut(&parts)
        }
    }

    fn adaptable_paths(&self) -> Vec<String> {
        krea_adaptable_paths(self.num_layers)
    }

    /// Route a ComfyUI/lightx2v diff-patch delta to a dense **norm** parameter (sc-15326) — the
    /// surface the 200 `.diff` (qk-RMSNorm gains, `norm3` gain) and 40 `norm3.diff_b` deltas a Wan-T2V
    /// step-distill file carries need, and which the `AdaptableLinear` surface cannot reach at any
    /// width. Same reference→converted normalization as [`Self::adaptable_mut`], then straight through
    /// to the reused [`WanTransformer::norm_param_mut`](mlx_gen_wan::WanTransformer::norm_param_mut).
    fn diff_patch_param_mut(
        &mut self,
        path: &[&str],
        part: DiffPatchPart,
    ) -> Option<&mut mlx_rs::Array> {
        let native = normalize_wan_key(&path.join("."));
        let parts: Vec<&str> = native.split('.').collect();
        self.inner.norm_param_mut(&parts, part)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::BTreeSet;

    /// A Wan-2.1-14B-T2V style LoRA (musubi-tuner / diffusion-pipe / ComfyUI), once its
    /// `diffusion_model.`/`transformer.` namespace is stripped, names exactly these per-block dotted
    /// targets — every one must be an adaptable Krea path or the strict installer would reject the file.
    /// The FFN is named `ffn.0`/`ffn.2` (the reference layout the file carries; the resolver normalizes
    /// it to the converted `ffn.fc1`/`ffn.fc2`). Mirrors the SCAIL-2 / Wan target guards.
    #[test]
    fn wan_family_style_lora_target_keys_are_adaptable_paths() {
        let paths: BTreeSet<String> = krea_adaptable_paths(40).into_iter().collect();
        for k in [
            "blocks.0.self_attn.q",
            "blocks.0.self_attn.k",
            "blocks.0.self_attn.v",
            "blocks.0.self_attn.o",
            "blocks.0.cross_attn.q",
            "blocks.0.cross_attn.o",
            "blocks.0.ffn.0",
            "blocks.0.ffn.2",
            "blocks.39.self_attn.q",
            "blocks.39.ffn.2",
        ] {
            assert!(
                paths.contains(k),
                "`{k}` is not an adaptable Krea LoRA target"
            );
        }
        // T2V backbone: the I2V-only image cross-attention does NOT exist on this model at any surface
        // width, so it must stay unexposed (a Wan-I2V LoRA naming it is a loud, honest error).
        assert!(!paths.contains("blocks.0.cross_attn.k_img"));
        assert!(!paths.contains("blocks.0.cross_attn.v_img"));
    }

    /// sc-8446 S13 — the settled globals decision, pinned. A real Wan-T2V **step-distill** LoRA
    /// (lightx2v `Wan2.1-T2V-14B` cfg-step-distill v2, `FastWan` T2V-14B) carries low-rank factors for
    /// six of the seven whole-model Linears in exactly these file spellings (`patch_embedding` ships a
    /// `.diff_b` bias delta only) — and every one of the seven must be an adaptable target or
    /// `apply_adapters_strict` rejects the whole file. The surface is seven wide because it describes
    /// the model; the *install* count is six, which is the 406-vs-407 gap pinned just below.
    #[test]
    fn step_distill_lora_global_target_keys_are_adaptable_paths() {
        let paths: BTreeSet<String> = krea_adaptable_paths(40).into_iter().collect();
        for k in [
            "patch_embedding",
            "text_embedding.0",
            "text_embedding.2",
            "time_embedding.0",
            "time_embedding.2",
            "time_projection.1",
            "head.head",
        ] {
            assert!(
                paths.contains(k),
                "`{k}` is a real lightx2v/FastWan global target but is not an adaptable Krea path"
            );
        }
    }

    /// sc-8446 — the **406 vs 407** gap, pinned where it can be checked rather than left in a commit
    /// message. The surface is 407 wide (400 per-block + 7 globals), but a real lightx2v / FastWan
    /// step-distill file installs 406: `patch_embedding` ships a `.diff_b` bias delta only, with no
    /// `lora_down`/`lora_up` pair for the low-rank pass to consume. Both facts have to stay true —
    /// the surface must keep `patch_embedding` (the model *has* that Linear), and the expected real
    /// install count must stay one below the surface width.
    #[test]
    fn step_distill_install_count_is_one_below_the_surface_width() {
        let paths = krea_adaptable_paths(40);
        assert_eq!(paths.len(), 407, "400 per-block + 7 globals");
        assert!(
            paths.iter().any(|p| p == "patch_embedding"),
            "patch_embedding stays exposed even though step-distill files carry no low-rank pair for it"
        );
        // The six globals a real step-distill file DOES carry low-rank factors for.
        let low_rank_globals = [
            "text_embedding.0",
            "text_embedding.2",
            "time_embedding.0",
            "time_embedding.2",
            "time_projection.1",
            "head.head",
        ];
        for g in low_rank_globals {
            assert!(paths.iter().any(|p| p == g), "`{g}` must be adaptable");
        }
        assert_eq!(
            400 + low_rank_globals.len(),
            406,
            "the expected real-weight install count for a step-distill file"
        );
        assert_eq!(
            paths.len() - (400 + low_rank_globals.len()),
            1,
            "exactly one exposed global (patch_embedding) is unmatched by a step-distill file"
        );
    }

    /// The file-spelled globals must normalize onto the converted spellings the inner Wan host routes —
    /// otherwise the path list would advertise a target `adaptable_mut` cannot reach. Discriminating: it
    /// compares the normalized set against `mlx_gen_wan`'s own constant, so a rename on either side that
    /// is not mirrored fails here rather than silently at install time.
    #[test]
    fn krea_global_paths_normalize_onto_the_wan_globals() {
        let normalized: BTreeSet<String> = KREA_GLOBAL_ADAPTABLE_PATHS
            .iter()
            .map(|p| normalize_wan_key(p))
            .collect();
        let wan: BTreeSet<String> = mlx_gen_wan::WAN_GLOBAL_ADAPTABLE_PATHS
            .iter()
            .map(|s| (*s).to_string())
            .collect();
        assert_eq!(normalized, wan);
        // None of them normalizes into the `blocks.` namespace, which is what makes the routing split in
        // `adaptable_mut` (first segment == "blocks") sound.
        assert!(normalized.iter().all(|p| !p.starts_with("blocks")));
    }

    /// sc-15326 — the **norm diff-patch surface**, pinned in the same style as the globals above. A
    /// Wan-T2V step-distill file's `.diff`/`.diff_b` norm keys are spelled in the reference layout;
    /// [`normalize_wan_key`] must carry every one of them onto a `(suffix, part)` the reused Wan host
    /// actually routes ([`mlx_gen_wan::WAN_BLOCK_NORM_DIFF_PATCH_TARGETS`]), or
    /// `diff_patch_param_mut` would advertise a target it cannot reach and the delta would be reported
    /// unmatched instead of applied.
    ///
    /// Discriminating: it compares against `mlx-gen-wan`'s own constant, so a rename on either side
    /// that is not mirrored fails here rather than silently at install time.
    #[test]
    fn krea_norm_diff_patch_keys_normalize_onto_the_wan_norm_targets() {
        // The five per-block norm stems a real lightx2v / FastWan T2V file carries, in FILE spelling.
        let file_stems = [
            "self_attn.norm_q",
            "self_attn.norm_k",
            "cross_attn.norm_q",
            "cross_attn.norm_k",
            "norm3",
        ];
        let routed: BTreeSet<&str> = mlx_gen_wan::WAN_BLOCK_NORM_DIFF_PATCH_TARGETS
            .iter()
            .map(|(suffix, _)| *suffix)
            .collect();
        for stem in file_stems {
            let native = normalize_wan_key(&format!("diffusion_model.blocks.7.{stem}"));
            let suffix = native
                .strip_prefix("blocks.7.")
                .unwrap_or_else(|| panic!("`{stem}` did not normalize under the block namespace"));
            assert!(
                routed.contains(suffix),
                "`{stem}` normalizes to `{suffix}`, which the Wan norm diff-patch surface does not route"
            );
        }
        // `norm3` is the only affine one — it alone carries a Bias part, matching the file's
        // `norm3.diff_b`; the qk-RMSNorms are weight-only, so a `.diff_b` on them is honestly skipped.
        let bias_targets: Vec<&str> = mlx_gen_wan::WAN_BLOCK_NORM_DIFF_PATCH_TARGETS
            .iter()
            .filter(|(_, part)| *part == mlx_gen::adapters::DiffPatchPart::Bias)
            .map(|(suffix, _)| *suffix)
            .collect();
        assert_eq!(bias_targets, vec!["norm3"]);
    }

    /// The path set must be duplicate-free AND stay collision-free under the kohya `.`→`_` flattening
    /// (the [`AdaptableHost::adaptable_paths`] contract — the `flattened → dotted` table would otherwise
    /// lose a target). 10 per-block Linears × `num_layers`, plus the 7 whole-model globals.
    #[test]
    fn krea_adaptable_paths_unique_and_kohya_collision_free() {
        let paths = krea_adaptable_paths(40);
        let n = paths.len();
        assert_eq!(n, 40 * 10 + 7);
        let uniq: BTreeSet<&String> = paths.iter().collect();
        assert_eq!(uniq.len(), n, "duplicate adaptable path");
        let flat: BTreeSet<String> = paths.iter().map(|p| p.replace('.', "_")).collect();
        assert_eq!(flat.len(), n, "kohya-flattened path collision");
    }

    #[test]
    fn end_of_block_math() {
        // block size 8: tokens 0..8 → end 8; 8..16 → end 16.
        assert_eq!(end_of_block(0, 8), 8);
        assert_eq!(end_of_block(7, 8), 8);
        assert_eq!(end_of_block(8, 8), 16);
        assert_eq!(end_of_block(15, 8), 16);
    }

    #[test]
    fn end_of_block_is_exact_at_i64_extremes() {
        assert_eq!(
            end_of_block(i64::MAX, 2),
            i128::from(i64::MAX) + 1,
            "the final representable two-token block ends one past i64::MAX"
        );
        assert_eq!(
            end_of_block(i64::MIN, 2),
            i128::from(i64::MIN) + 2,
            "negative division must retain the previous truncation-toward-zero rule"
        );
        assert!(
            !is_masked(i64::MAX - 1, i64::MAX, 2),
            "both final representable positions occupy the same block"
        );
        let final_block = [i64::MAX - 1, i64::MAX];
        let (data, any_masked) = mask_data(&final_block, &final_block, 2).unwrap();
        assert!(!any_masked);
        assert_eq!(data, vec![0.0; 4]);
        assert_eq!(
            build_block_causal_mask(&final_block, &final_block, 2)
                .unwrap()
                .shape(),
            &[2, 2]
        );
    }

    /// (C) The explicit mask rule: 2 blocks × 2 tokens (block_size 2) → the hand-derived [4,4] additive
    /// matrix. Block 0 (q0,q1) sees only block 0; block 1 (q2,q3) sees blocks 0 and 1.
    #[test]
    fn block_causal_mask_rule_is_exact() {
        let q_pos = [0i64, 1, 2, 3];
        let kv_pos = [0i64, 1, 2, 3];
        let (data, any) = mask_data(&q_pos, &kv_pos, 2).unwrap();
        assert!(any, "a 2-block case must mask block 0 → block 1");
        let ninf = f32::NEG_INFINITY;
        #[rustfmt::skip]
        let expected = vec![
            0.0, 0.0, ninf, ninf, // q0 (block 0, end 2): kv 0,1 allowed; 2,3 future-masked
            0.0, 0.0, ninf, ninf, // q1 (block 0, end 2)
            0.0, 0.0, 0.0,  0.0,  // q2 (block 1, end 4): all allowed
            0.0, 0.0, 0.0,  0.0,  // q3 (block 1, end 4)
        ];
        assert_eq!(
            data, expected,
            "block-causal mask must match the hand-derived matrix"
        );
    }

    #[test]
    fn single_block_mask_is_all_allowed() {
        // One full block of queries+keys is fully bidirectional ⇒ no mask (None).
        let pos = [0i64, 1, 2, 3];
        let before = mask_materialization_count();
        let m = block_causal_mask(&pos, &pos, 4).unwrap();
        assert!(m.is_none(), "a single block must need no mask");
        assert_eq!(
            mask_materialization_count(),
            before,
            "the common all-allowed path must not materialize an Sq x Sk host matrix"
        );

        let masked = block_causal_mask(&[0], &[0, 4], 4).unwrap();
        assert!(masked.is_some(), "a future block must materialize a mask");
        assert_eq!(
            mask_materialization_count(),
            before + 1,
            "a genuinely masked path must materialize exactly one host matrix"
        );
    }

    #[test]
    fn zero_block_size_is_a_typed_error() {
        let err = block_causal_mask(&[0], &[0], 0)
            .expect_err("zero block size must return an error instead of dividing by zero");
        assert!(matches!(err, Error::Msg(_)), "got: {err:?}");
        let err = build_block_causal_mask(&[0], &[0], 0)
            .expect_err("the explicit builder must also reject zero block size");
        assert!(matches!(err, Error::Msg(_)), "got: {err:?}");
    }

    #[test]
    fn analytic_mask_decision_matches_scalar_rule_across_positions_and_blocks() {
        for block_size in 1..=9 {
            for q_start in 0..=12 {
                for q_len in 0..=6 {
                    let q_pos = (q_start..q_start + q_len)
                        .map(i64::from)
                        .collect::<Vec<_>>();
                    for kv_start in 0..=12 {
                        for kv_len in 0..=9 {
                            let kv_pos = (kv_start..kv_start + kv_len)
                                .map(i64::from)
                                .collect::<Vec<_>>();
                            let scalar = q_pos.iter().any(|&q| {
                                kv_pos
                                    .iter()
                                    .any(|&kv| is_masked(q, kv, i64::from(block_size)))
                            });
                            assert_eq!(
                                mask_is_needed(&q_pos, &kv_pos, block_size as usize).unwrap(),
                                scalar,
                                "q={q_pos:?}, kv={kv_pos:?}, block_size={block_size}"
                            );
                        }
                    }
                }
            }
        }

        // Gaps, duplicates, reverse order, and negative positions exercise the equality exception
        // and prove the decision does not depend on sorted contiguous production positions.
        let unusual = [
            vec![],
            vec![0],
            vec![8, 3, 8],
            vec![-9, -1, 0, 7],
            vec![i64::MIN, i64::MIN + 1, 1, i64::MAX - 1, i64::MAX],
        ];
        for block_size in 1..=9 {
            for q_pos in &unusual {
                for kv_pos in &unusual {
                    let bs = i64::from(block_size);
                    let scalar = q_pos
                        .iter()
                        .any(|&q| kv_pos.iter().any(|&kv| is_masked(q, kv, bs)));
                    assert_eq!(
                        mask_is_needed(q_pos, kv_pos, block_size as usize).unwrap(),
                        scalar,
                        "q={q_pos:?}, kv={kv_pos:?}, block_size={block_size}"
                    );
                }
            }
        }
    }

    #[test]
    fn incremental_step_mask_over_cached_plus_new_is_all_allowed() {
        // Chunk 1 (one block) attending cached block 0 + itself: every key < the query block's end ⇒
        // all allowed ⇒ None. This is why the AR step needs the cache + RoPE offset, not a mask.
        let q_pos = [4i64, 5, 6, 7]; // block 1
        let kv_pos = [0i64, 1, 2, 3, 4, 5, 6, 7]; // cached block 0 ‖ block 1
        let m = block_causal_mask(&q_pos, &kv_pos, 4).unwrap();
        assert!(m.is_none());
    }

    #[test]
    fn window_positions_global_keeps_everything() {
        // Global window (max_attention_size huge, no sink): the prev window is the whole history.
        let mut cache = CausalKvCache::new(1, 1_000_000, 0, None);
        cache.committed_tokens = 8; // simulate one 8-token block cached
        let pos = cache.window_positions(8);
        assert_eq!(pos, (0..8).collect::<Vec<i64>>());
    }

    #[test]
    fn window_positions_sliding_and_sink() {
        // stored 16, new 4, window 8, sink 2: current_end=20, read_start=max(2, 20-8)=12,
        // window = sink [0,2) ∪ tail [12,16).
        let mut cache = CausalKvCache::new(1, 8, 2, None);
        cache.committed_tokens = 16;
        let pos = cache.window_positions(4);
        assert_eq!(pos, vec![0, 1, 12, 13, 14, 15]);
    }

    /// A one-token-wide `(k, v)` per layer with a recognizable value at a given position — lets the
    /// eviction tests assert *which* global tokens survived by reading the retained buffer back.
    fn kv_block(pos_values: &[f32]) -> Vec<LayerKv> {
        let s = pos_values.len() as i32;
        // [B=1, n=1, s, d=1]
        let k = Array::from_slice(pos_values, &[1, 1, s, 1]);
        let v = Array::from_slice(pos_values, &[1, 1, s, 1]);
        vec![(k, v)]
    }

    fn retained_key_values(cache: &CausalKvCache) -> Vec<f32> {
        let (k, _) = cache.layer_kv(0).unwrap().expect("layer 0 populated");
        k.as_slice::<f32>().to_vec()
    }

    // --- sc-17807: the quantized KV cache ------------------------------------------------------
    //
    // A quantization group must divide the cached tensor's LAST axis (`head_dim`), so these use a
    // realistic `d = 64` rather than the `d = 1` the eviction tests above get away with.
    const QD: i32 = 64;

    /// A `[B=1, n=1, s, QD]` `(k, v)` block whose token `g` carries the row `g + j/QD` for
    /// `j in 0..QD`. Two properties this buys:
    ///   * **token identity is recoverable** — the row mean is `g + (QD-1)/(2·QD)`, and adjacent
    ///     tokens differ by a full 1.0, which is orders of magnitude outside the quantization error
    ///     below. A gather that returns the wrong token cannot pass a tolerance that a correct one does.
    ///   * **the quantization is real** — a constant row would quantize with zero scale (exactly), so
    ///     the intra-row ramp is what makes the round-trip error non-trivial.
    ///
    /// `v` is offset so a k/v mix-up is caught too.
    fn wide_kv_block(positions: &[usize], dtype: Dtype) -> Vec<LayerKv> {
        let s = positions.len() as i32;
        let build = |offset: f32| {
            let data: Vec<f32> = positions
                .iter()
                .flat_map(|&g| (0..QD).map(move |j| offset + g as f32 + j as f32 / QD as f32))
                .collect();
            Array::from_slice(&data, &[1, 1, s, QD])
                .as_dtype(dtype)
                .expect("cast the synthetic KV block")
        };
        vec![(build(0.0), build(100.0))]
    }

    fn row_means(a: &Array) -> Vec<f32> {
        let flat = a
            .as_dtype(Dtype::Float32)
            .expect("read back as f32")
            .as_slice::<f32>()
            .to_vec();
        flat.chunks(QD as usize)
            .map(|row| row.iter().sum::<f32>() / QD as f32)
            .collect()
    }

    /// **The sc-17807 feasibility gate, as arithmetic rather than prose.**
    ///
    /// MLX at the pinned revision has no fused quantized SDPA, so "let attention consume the packed
    /// cache" means the decomposed `quantized_matmul → softmax → quantized_matmul` form — which
    /// materializes the full `[B, heads, Sq, Sk]` score matrix that the fused kernel never builds. For
    /// LLM decode that is free (`Sq = 1`); for one Krea AR chunk `Sq` is a whole frame-block.
    ///
    /// Two numbers decide it, and both are computed here rather than asserted in prose:
    ///
    ///   1. **One layer's** score matrix is of the same order as the Q8 saving on the **whole
    ///      40-layer cache** — so the route trades a persistent saving for a transient of comparable
    ///      size, and a `precise` f32 softmax (what mlx-lm uses) doubles the transient.
    ///   2. Against the route actually taken — gather packed, dequantize the window — the decomposed
    ///      transient is well over an order of magnitude larger *per layer*. That margin, not (1), is
    ///      what makes this a settled decision rather than a judgement call.
    #[test]
    fn decomposed_quantized_attention_costs_far_more_than_dequantizing_the_window() {
        // Shipped bounded geometry at 832x480: 1560 tokens/latent frame, 3 frames per AR chunk, a
        // 6-latent-frame read window, 40 heads, dim 5120.
        let cfg = KreaRealtimeConfig::krea_realtime_14b();
        let (tokens_per_frame, frames_per_block, window_frames) = (1560usize, 3, 6);
        let sq = tokens_per_frame * frames_per_block; // 4,680 query tokens in one chunk
        let sk = tokens_per_frame * window_frames; // read window ‖ this chunk = 9,360 keys
        let prev = sk - sq; // 4,680 cached keys this chunk actually reads
        assert_eq!((sq, sk), (4_680, 9_360));

        // (a) The decomposed form's score matrix, for ONE layer, in bf16 — the optimistic bound.
        let scores_bytes = cfg.wan.num_heads * sq * sk * 2;
        // (b) What Q8 saves on the whole cache across all 40 layers.
        let dense_per_token = cfg.kv_bytes_per_token().unwrap();
        let mut q8 = cfg.clone();
        q8.ar.kv_cache_quant = Some(KvCacheQuant::Q8);
        let saving_bytes = (dense_per_token - q8.kv_bytes_per_token().unwrap()) * sk;
        // (c) The route taken: one layer's dequantized read window (K and V, bf16).
        let dequantized_window_bytes = 2 * prev * cfg.wan.dim * 2;

        assert!(
            scores_bytes * 10 >= saving_bytes * 9,
            "ONE layer's decomposed-attention score matrix is {scores_bytes} bytes against a \
             whole-cache Q8 saving of {saving_bytes}. If that ratio ever collapses, consuming the \
             packed cache directly becomes worth reconsidering — as would a fused quantized SDPA, \
             which would settle it outright"
        );
        assert!(
            scores_bytes > 30 * dequantized_window_bytes,
            "per layer, the decomposed route costs {scores_bytes} bytes against \
             {dequantized_window_bytes} for dequantizing the read window — that margin is why the \
             cache dequantizes on read instead of attending over packed K/V"
        );
    }

    /// MLX emits the quantization `scales`/`biases` in the **input's** dtype. `KvCacheQuant::row_bytes`
    /// prices a group's overhead at 4 bytes on exactly that basis (bf16 scale + bf16 bias), so if this
    /// ever changed the published per-token table would silently understate a quantized cache.
    #[test]
    fn quantize_emits_scales_and_biases_in_the_input_dtype() {
        let kv = wide_kv_block(&[0, 1, 2, 3], Dtype::Bfloat16);
        let packed = PackedKv::pack(&kv[0].0, KvCacheQuant::Q8).unwrap();
        assert_eq!(packed.scales.dtype(), Dtype::Bfloat16);
        assert_eq!(packed.biases.dtype(), Dtype::Bfloat16);
        assert_eq!(packed.w.dtype(), Dtype::Uint32);
        // Packed payload keeps the leading [B, n, S] axes — which is what makes the cache's
        // token-axis concat/gather apply to it unchanged — and packs only the last one.
        assert_eq!(packed.w.shape()[..3], [1, 1, 4]);
        assert_eq!(packed.scales.shape(), &[1, 1, 4, QD / 64]);
        // Dequantizing returns the dense shape and dtype attention expects.
        let back = packed.unpack(KvCacheQuant::Q8).unwrap();
        assert_eq!(back.shape(), kv[0].0.shape());
        assert_eq!(back.dtype(), Dtype::Bfloat16);
    }

    /// A group size that does not divide `head_dim` is a **config** error, and must be reported as one
    /// (naming the knob) rather than surfacing as an opaque MLX exception from inside a chunk forward.
    #[test]
    fn a_group_size_that_does_not_divide_head_dim_is_a_typed_error() {
        let mut cache = CausalKvCache::new(
            1,
            1_000_000,
            0,
            Some(KvCacheQuant {
                bits: 8,
                group_size: 128,
            }),
        );
        // QD = 64 < 128, so no whole group fits.
        let err = cache
            .append(wide_kv_block(&[0, 1], Dtype::Float32))
            .expect_err("a group size larger than head_dim must be refused");
        let msg = format!("{err}");
        assert!(
            msg.contains("head_dim") && msg.contains("kv_cache_quant"),
            "the error must name the geometry and the knob: {msg}"
        );
        // An unsupported width is refused before it reaches MLX at all.
        let bad = KvCacheQuant {
            bits: 7,
            group_size: 64,
        };
        assert!(format!("{}", bad.validate().unwrap_err()).contains("affine quantization width"));
    }

    /// **The cache algebra must commute with packing.** The quantized cache has to evict, window and
    /// order *exactly* the tokens the dense one does — packing changes the values slightly, never
    /// which values. Driven through the same eviction-and-roll sequence the dense bounded-window test
    /// uses, then compared token-for-token.
    ///
    /// Discriminating: adjacent tokens differ by 1.0 while the Q8 round trip is bounded at 0.01, so a
    /// one-token-off gather (the failure mode `take_axis` on packed parts would introduce) fails by
    /// two orders of magnitude. The dense arm is asserted against the same expectations, so a bug that
    /// broke *both* representations identically still fails.
    #[test]
    fn quantized_cache_evicts_and_windows_exactly_the_dense_tokens() {
        let window = 4usize;
        let mut dense = CausalKvCache::new(1, window, 0, None);
        let mut packed = CausalKvCache::new(1, window, 0, Some(KvCacheQuant::Q8));
        for start in (0..12).step_by(2) {
            dense
                .append(wide_kv_block(&[start, start + 1], Dtype::Float32))
                .unwrap();
            packed
                .append(wide_kv_block(&[start, start + 1], Dtype::Float32))
                .unwrap();
        }
        assert_eq!(packed.stored_tokens(), dense.stored_tokens());
        assert_eq!(packed.retained_tokens(), dense.retained_tokens());

        // The retained buffers hold the same global tokens, in the same order.
        let (dk, _) = dense.layer_kv(0).unwrap().unwrap();
        let (pk, pv) = packed.layer_kv(0).unwrap().unwrap();
        let (dense_rows, packed_rows) = (row_means(&dk), row_means(&pk));
        assert_eq!(dense_rows.len(), packed_rows.len());
        // Token `g`'s row mean is `g + 63/128`; the retained window is the last 4 committed tokens.
        let expected: Vec<f32> = (8..12).map(|g| g as f32 + 63.0 / 128.0).collect();
        for (i, want) in expected.iter().enumerate() {
            assert!(
                (dense_rows[i] - want).abs() < 1e-4,
                "dense token {i}: {} != {want}",
                dense_rows[i]
            );
            assert!(
                (packed_rows[i] - want).abs() < 0.01,
                "quantized token {i}: {} != {want} — an error this large is a wrong token, not \
                 quantization noise (adjacent tokens differ by 1.0)",
                packed_rows[i]
            );
        }
        // V is offset by 100 in the fixture, so a k/v swap in the packed path cannot pass.
        let packed_v_rows = row_means(&pv);
        assert!((packed_v_rows[0] - (100.0 + expected[0])).abs() < 0.02);

        // ...and the same holds for the READ window, which gathers a strict subset of the retained
        // buffer (the path that has to index three packed parts consistently).
        let (dense_prev, dense_pos) = dense.window_prev(2).unwrap();
        let (packed_prev, packed_pos) = packed.window_prev(2).unwrap();
        assert_eq!(packed_pos, dense_pos, "the read window must not move");
        assert_eq!(packed_pos, vec![10, 11]);
        let dense_window = row_means(&dense_prev[0].0);
        let packed_window = row_means(&packed_prev[0].0);
        for (i, (d, p)) in dense_window.iter().zip(&packed_window).enumerate() {
            assert!(
                (d - p).abs() < 0.01,
                "read-window token {i}: dense {d} vs quantized {p}"
            );
        }
    }

    /// The sink prefix is the other retention path (a permanent head plus a rolling tail, gathered
    /// through a keep-map), and it must survive packing too — this is the same case the
    /// `bounded_window_retains_sink_prefix` dense test pins.
    #[test]
    fn quantized_cache_retains_the_sink_prefix_across_eviction() {
        let mut cache = CausalKvCache::new(1, 4, 2, Some(KvCacheQuant::Q8));
        for start in (0..8).step_by(2) {
            cache
                .append(wide_kv_block(&[start, start + 1], Dtype::Float32))
                .unwrap();
        }
        let (k, _) = cache.layer_kv(0).unwrap().unwrap();
        let got = row_means(&k);
        // Sink [0,1] pinned; oldest non-sink tail [2,3] evicted; [4..8) retained.
        let expected: Vec<f32> = [0usize, 1, 4, 5, 6, 7]
            .iter()
            .map(|&g| g as f32 + 63.0 / 128.0)
            .collect();
        assert_eq!(got.len(), expected.len());
        for (i, want) in expected.iter().enumerate() {
            assert!(
                (got[i] - want).abs() < 0.01,
                "retained slot {i}: {} != {want}",
                got[i]
            );
        }
    }

    /// **The measurement instrument.** `retained_bytes` must report the STORED representation, and it
    /// must agree with the published per-token table — otherwise the memory claim and the memory
    /// measurement are two independent stories. Uses bf16 blocks because that is what the AR path
    /// caches, and what `KvCacheQuant::row_bytes` prices the per-group scale/bias at.
    #[test]
    fn retained_bytes_reports_the_packed_cost_and_matches_the_published_row_cost() {
        let tokens: Vec<usize> = (0..8).collect();
        let mut dense = CausalKvCache::new(1, 1_000_000, 0, None);
        let mut packed = CausalKvCache::new(1, 1_000_000, 0, Some(KvCacheQuant::Q8));
        dense
            .append(wide_kv_block(&tokens, Dtype::Bfloat16))
            .unwrap();
        packed
            .append(wide_kv_block(&tokens, Dtype::Bfloat16))
            .unwrap();

        let n = tokens.len();
        let d = QD as usize;
        // 2 (K and V) x 1 layer x n tokens x the tier's row cost.
        assert_eq!(dense.retained_bytes(), 2 * n * d * 2);
        assert_eq!(
            packed.retained_bytes(),
            2 * n * KvCacheQuant::Q8.row_bytes(d).unwrap(),
            "the measured packed residency must equal the published row cost"
        );
        // The whole point: it is materially smaller, and it is NOT the dequantized size.
        assert!(packed.retained_bytes() < dense.retained_bytes() * 55 / 100);
        let (k, _) = packed.layer_kv(0).unwrap().unwrap();
        assert!(
            k.nbytes() * 2 > packed.retained_bytes(),
            "reading nbytes off the dequantized handles would report the bf16 size — the accessor \
             doc says to use retained_bytes for exactly this reason"
        );
    }

    /// A quantized cache must be selected by config, and only by config: the shipped preset stays
    /// dense, so nothing changes for an existing caller.
    #[test]
    fn the_shipped_config_builds_a_dense_cache_and_the_knob_builds_a_packed_one() {
        let base = KreaRealtimeConfig::krea_realtime_14b();
        assert_eq!(base.ar.kv_cache_quant, None);
        assert_eq!(
            CausalKvCache::new(1, 4, 0, base.ar.kv_cache_quant).quant(),
            None
        );
        let mut q8 = base;
        q8.ar.kv_cache_quant = Some(KvCacheQuant::Q8);
        assert_eq!(
            CausalKvCache::new(1, 4, 0, q8.ar.kv_cache_quant).quant(),
            Some(KvCacheQuant::Q8)
        );
    }

    #[test]
    fn global_regime_retains_everything() {
        // max_attention_size huge ⇒ no eviction ever: retained == committed, buffer is [0, committed).
        let mut cache = CausalKvCache::new(1, 1_000_000, 0, None);
        cache.append(kv_block(&[0.0, 1.0, 2.0, 3.0])).unwrap();
        cache.append(kv_block(&[4.0, 5.0, 6.0, 7.0])).unwrap();
        assert_eq!(cache.stored_tokens(), 8);
        assert_eq!(
            cache.retained_tokens(),
            8,
            "global regime keeps every token"
        );
        assert_eq!(
            retained_key_values(&cache),
            (0..8).map(|i| i as f32).collect::<Vec<_>>()
        );
        // The whole-history read hands back all 8 in order.
        let (_prev, positions) = cache.window_prev(4).unwrap();
        assert_eq!(positions, (0..8).collect::<Vec<i64>>());
    }

    #[test]
    fn bounded_window_evicts_oldest_tail_no_sink() {
        // Window 4 tokens, no sink. Append four 2-token chunks (global 0..8). After each append only the
        // most-recent ≤ 4 tokens survive physically; committed keeps counting.
        let mut cache = CausalKvCache::new(1, 4, 0, None);
        cache.append(kv_block(&[0.0, 1.0])).unwrap(); // committed 2, retained [0,1]
        assert_eq!(cache.retained_tokens(), 2);
        cache.append(kv_block(&[2.0, 3.0])).unwrap(); // committed 4, retained [0,1,2,3]
        assert_eq!(cache.retained_tokens(), 4);
        assert_eq!(retained_key_values(&cache), vec![0.0, 1.0, 2.0, 3.0]);
        cache.append(kv_block(&[4.0, 5.0])).unwrap(); // committed 6, evict [0,1] ⇒ retained [2,3,4,5]
        assert_eq!(cache.stored_tokens(), 6);
        assert_eq!(
            cache.retained_tokens(),
            4,
            "storage stays bounded to the window"
        );
        assert_eq!(
            retained_key_values(&cache),
            vec![2.0, 3.0, 4.0, 5.0],
            "the oldest tail tokens are physically evicted"
        );
        cache.append(kv_block(&[6.0, 7.0])).unwrap(); // committed 8, retained [4,5,6,7]
        assert_eq!(cache.retained_tokens(), 4);
        assert_eq!(retained_key_values(&cache), vec![4.0, 5.0, 6.0, 7.0]);
    }

    #[test]
    fn bounded_window_read_is_last_max_attention_size_tokens() {
        // A clip much longer than the window: the read window for the next chunk is exactly the last
        // `max_attention_size` tokens (global positions), and every one is physically retained (no OOB).
        let window = 4usize;
        let mut cache = CausalKvCache::new(1, window, 0, None);
        for start in (0..12).step_by(2) {
            cache
                .append(kv_block(&[start as f32, (start + 1) as f32]))
                .unwrap();
        }
        assert_eq!(cache.stored_tokens(), 12);
        assert!(cache.retained_tokens() <= window + 2, "bounded storage");
        // Next chunk of s_new = 2: current_end = 14, read_start = max(0, 14 - 4) = 10 ⇒ window [10, 12).
        let (prev, positions) = cache.window_prev(2).unwrap();
        assert_eq!(
            positions,
            vec![10, 11],
            "reads only the last max_attention_size tokens"
        );
        // The gathered K carries exactly those global positions' values (retained, correct, no OOB).
        let (k, _) = &prev[0];
        assert_eq!(k.as_slice::<f32>(), &[10.0, 11.0]);
    }

    #[test]
    fn bounded_window_retains_sink_prefix() {
        // sink 2, window 4 (window includes the sink). The first two global tokens [0,1] are always
        // retained; the tail rolls under them.
        let mut cache = CausalKvCache::new(1, 4, 2, None);
        cache.append(kv_block(&[0.0, 1.0])).unwrap(); // committed 2 (all sink)
        cache.append(kv_block(&[2.0, 3.0])).unwrap(); // committed 4, retained [0,1,2,3]
        cache.append(kv_block(&[4.0, 5.0])).unwrap(); // committed 6
                                                      // sink_kept 2 ([0,1]); tail_base = max(2, 6-4)=2 ⇒ still contiguous, retained [0,1,2,3,4,5]?
                                                      // committed 6, window 4: tail_base = max(2, 2)=2, retained = 2 + (6-2)=6, no drop yet.
        assert_eq!(
            retained_key_values(&cache),
            vec![0.0, 1.0, 2.0, 3.0, 4.0, 5.0]
        );
        cache.append(kv_block(&[6.0, 7.0])).unwrap(); // committed 8, tail_base = max(2, 8-4)=4 ⇒ drop [2,3]
        assert_eq!(cache.stored_tokens(), 8);
        assert_eq!(
            retained_key_values(&cache),
            vec![0.0, 1.0, 4.0, 5.0, 6.0, 7.0],
            "sink prefix [0,1] retained; oldest non-sink tail [2,3] evicted"
        );
    }

    #[test]
    fn eviction_before_sink_fills_gathers_pre_eviction_layout() {
        // Discriminating regression (the reviewer's repro): a single chunk that overshoots
        // `sink_tokens + max_attention_size` *before* the sink prefix has filled. With
        // `max_attention_size = 2`, `sink_tokens = 4`, ONE 10-token append (global 0..9) fires the
        // first eviction while `committed_before (0) < sink_tokens (4)`. The pre-eviction buffer is
        // contiguous (phys == global, `sink_kept_prev = 0`, `tail_base_old = 0`), so the keep map must
        // gather physical `[0,1,2,3,8,9]` → retained global `[0,1,2,3,8,9]` (sink `[0..4)` ‖ the last
        // `max_attention_size = 2` tokens `[8,9]`). The old map used the *post-append* sink length and
        // gathered `[0,1,2,3,12,13]` — out of bounds on a length-10 axis (unchecked `take_axis` on
        // Metal ⇒ silent KV corruption / crash). This test therefore fails on the buggy map and passes
        // on the pre-eviction-layout fix.
        let mut cache = CausalKvCache::new(1, 2, 4, None);
        cache
            .append(kv_block(&[
                0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0,
            ]))
            .unwrap();
        assert_eq!(cache.stored_tokens(), 10);
        assert_eq!(
            cache.retained_tokens(),
            6,
            "sink (4) + last max_attention_size (2) = 6 physically retained"
        );
        // The retained global positions are exactly [0,1,2,3,8,9] (values mirror global position),
        // gathered without OOB — NOT the buggy [0,1,2,3,12,13] map.
        assert_eq!(
            retained_key_values(&cache),
            vec![0.0, 1.0, 2.0, 3.0, 8.0, 9.0],
            "sink prefix [0..4) retained; window is the last two tokens [8,9]"
        );
        // Values (v) mirror keys (k): the same retained source rows, no OOB.
        let (_, v) = cache.layer_kv(0).unwrap().expect("layer 0 populated");
        assert_eq!(v.as_slice::<f32>(), &[0.0, 1.0, 2.0, 3.0, 8.0, 9.0]);
        // The next chunk's read window is well-formed and gathers retained rows only (no OOB on read).
        // With this tiny window (max_attention_size = 2), the sliding tail lands at/after `stored`, so
        // the next chunk attends just the always-on sink prefix [0,1,2,3].
        let (prev, positions) = cache.window_prev(2).unwrap();
        assert_eq!(positions, vec![0, 1, 2, 3]);
        let (k, _) = &prev[0];
        assert_eq!(k.as_slice::<f32>(), &[0.0, 1.0, 2.0, 3.0]);
    }
}
