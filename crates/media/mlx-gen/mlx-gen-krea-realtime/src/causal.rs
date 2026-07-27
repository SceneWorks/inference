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

use mlx_gen::adapters::{AdaptableHost, AdaptableLinear};
use mlx_gen::{Error, Result};
use mlx_gen_wan::patchify::unpatchify;
use mlx_gen_wan::{normalize_wan_key, WanTransformer};
use mlx_rs::ops::concatenate_axis;
use mlx_rs::{Array, Dtype};

use crate::config::KreaRealtimeConfig;

/// One layer's cached self-attention `(post-RoPE keys, raw values)`, each `[B, n, S, head_dim]`.
pub type LayerKv = (Array, Array);

/// End of the frame-block containing global token index `q` (exclusive): `(q / block + 1) * block`.
#[inline]
fn end_of_block(q: i64, block_size: i64) -> i64 {
    (q / block_size + 1) * block_size
}

/// The raw additive block-causal mask data for query global positions `q_pos` against key global
/// positions `kv_pos` (block size `block_size` tokens), plus whether **any** entry is masked. Entry
/// `(i, j)` is `0.0` when `kv_pos[j] < end_of_block(q_pos[i]) || q_pos[i] == kv_pos[j]`, else `-inf`.
fn mask_data(q_pos: &[i64], kv_pos: &[i64], block_size: usize) -> (Vec<f32>, bool) {
    let bs = block_size as i64;
    let (sq, sk) = (q_pos.len(), kv_pos.len());
    let mut data = vec![0f32; sq * sk];
    let mut any_masked = false;
    for (i, &q) in q_pos.iter().enumerate() {
        let end = end_of_block(q, bs);
        for (j, &kv) in kv_pos.iter().enumerate() {
            if !(kv < end || q == kv) {
                data[i * sk + j] = f32::NEG_INFINITY;
                any_masked = true;
            }
        }
    }
    (data, any_masked)
}

/// Build the **additive** block-causal SDPA mask (bf16) for query global token positions `q_pos`
/// against key global token positions `kv_pos`, with `block_size` tokens per frame-block. Shape
/// `[Sq, Sk]` — broadcasts over the batch/head axes in [`mlx_rs::fast::scaled_dot_product_attention`].
/// `0.0` where a query may attend the key, `-inf` where it may not. Mirrors
/// `causal_model.py::get_sdpa_mask`/`get_block_mask`. See [`block_causal_mask`] for the forward path's
/// "all-allowed ⇒ no mask" optimization.
pub fn build_block_causal_mask(q_pos: &[i64], kv_pos: &[i64], block_size: usize) -> Result<Array> {
    let (data, _) = mask_data(q_pos, kv_pos, block_size);
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
    let (data, any_masked) = mask_data(q_pos, kv_pos, block_size);
    if !any_masked {
        return Ok(None);
    }
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
pub struct CausalKvCache {
    /// Per layer: `Some((k, v))` once populated — the physically retained tokens in global order (the
    /// sink prefix `[0, sink_kept)` followed by the rolling tail `[tail_base, stored_tokens)`), each
    /// `[B, n, retained_tokens, d]` (post-RoPE k, raw v).
    layers: Vec<Option<LayerKv>>,
    /// Running global token count committed so far (the reference's `global_end_index`); grows by the
    /// chunk length on every [`append`](Self::append) and never shrinks, even under eviction.
    committed_tokens: usize,
    /// Global position of the first physically-retained **tail** token. Equals `sink_kept` until
    /// eviction rolls it forward; tokens in `[sink_kept, tail_base)` have been evicted from storage.
    tail_base: usize,
    max_attention_size: usize,
    sink_tokens: usize,
}

impl CausalKvCache {
    /// An empty cache for `num_layers` transformer blocks with the given read-window geometry (in
    /// tokens). See [`KreaArConfig::max_attention_size`](crate::KreaArConfig::max_attention_size) /
    /// [`sink_tokens`](crate::KreaArConfig::sink_tokens).
    pub fn new(num_layers: usize, max_attention_size: usize, sink_tokens: usize) -> Self {
        Self {
            layers: (0..num_layers).map(|_| None).collect(),
            committed_tokens: 0,
            tail_base: 0,
            max_attention_size,
            sink_tokens,
        }
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

    /// The retained `(k, v)` per layer (post-RoPE k, raw v), for verification (S5 recompute /
    /// bounded-window tests) and the S6 pipeline. `None` before the first append or for an unknown
    /// layer.
    pub fn layer_kv(&self, layer: usize) -> Option<&LayerKv> {
        self.layers.get(layer).and_then(|l| l.as_ref())
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
    /// retained buffer (the global / fits case) the stored handles are returned directly (no gather);
    /// otherwise the physically-indexed window is gathered.
    pub fn window_prev(&self, s_new: usize) -> Result<(Vec<LayerKv>, Vec<i64>)> {
        let positions = self.window_positions(s_new);
        if positions.is_empty() {
            return Ok((Vec::new(), Vec::new()));
        }
        let phys: Vec<i32> = positions
            .iter()
            .map(|&g| self.phys_index(g as usize) as i32)
            .collect();
        // Reading the entire retained buffer in order ⇒ hand the stored buffers back (no gather).
        let whole = phys.len() == self.retained_tokens()
            && phys.iter().enumerate().all(|(i, &p)| p as usize == i);
        let idx = if whole {
            None
        } else {
            Some(Array::from_slice(&phys, &[phys.len() as i32]))
        };
        let mut prev = Vec::with_capacity(self.layers.len());
        for layer in &self.layers {
            let (k, v) = layer.as_ref().ok_or_else(|| {
                Error::Msg("krea causal: KV cache has tokens but a layer slot is empty".into())
            })?;
            match &idx {
                None => prev.push((k.clone(), v.clone())),
                Some(ix) => prev.push((k.take_axis(ix, 2)?, v.take_axis(ix, 2)?)),
            }
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
        // Concatenate this chunk's K/V onto the physical tail (each layer identically).
        let mut s_new = 0usize;
        for (slot, (nk, nv)) in self.layers.iter_mut().zip(new_kv) {
            s_new = nk.shape()[2] as usize;
            *slot = Some(match slot.take() {
                None => (nk, nv),
                Some((pk, pv)) => (
                    concatenate_axis(&[&pk, &nk], 2)?,
                    concatenate_axis(&[&pv, &nv], 2)?,
                ),
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
                if let Some((k, v)) = slot.as_ref() {
                    *slot = Some((k.take_axis(&idx, 2)?, v.take_axis(&idx, 2)?));
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

    /// A fresh, empty [`CausalKvCache`] sized to this model (layers + read-window geometry).
    pub fn new_cache(&self) -> CausalKvCache {
        CausalKvCache::new(self.num_layers, self.max_attention_size, self.sink_tokens)
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
/// ⚠️ **Widening does NOT make a step-distill file fully applied, and this comment must not be read as
/// claiming it does.** The same lightx2v/FastWan file carries **647 further keys the low-rank pass does
/// not consume**: 447 `.diff_b` bias deltas (including `patch_embedding`'s) and 200 `.diff` weight
/// deltas on the qk/`norm3` **norms**, which are not `AdaptableLinear`s at any surface width. Krea calls
/// [`apply_adapters_strict`](mlx_gen::adapters::loader::apply_adapters_strict), not the
/// `_with_diff_patch` variant, so those are dropped **without a word** — the same silent
/// under-application this decision argues against, merely at a different seam. Tracked as **sc-15326**;
/// until it lands, "a step-distill LoRA installs" means its low-rank half installs.
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
    /// all seven whole-model Linears in exactly these file spellings; every one must now be an adaptable
    /// target or `apply_adapters_strict` rejects the whole file.
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

    /// (C) The explicit mask rule: 2 blocks × 2 tokens (block_size 2) → the hand-derived [4,4] additive
    /// matrix. Block 0 (q0,q1) sees only block 0; block 1 (q2,q3) sees blocks 0 and 1.
    #[test]
    fn block_causal_mask_rule_is_exact() {
        let q_pos = [0i64, 1, 2, 3];
        let kv_pos = [0i64, 1, 2, 3];
        let (data, any) = mask_data(&q_pos, &kv_pos, 2);
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
        let m = block_causal_mask(&pos, &pos, 4).unwrap();
        assert!(m.is_none(), "a single block must need no mask");
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
        let mut cache = CausalKvCache::new(1, 1_000_000, 0);
        cache.committed_tokens = 8; // simulate one 8-token block cached
        let pos = cache.window_positions(8);
        assert_eq!(pos, (0..8).collect::<Vec<i64>>());
    }

    #[test]
    fn window_positions_sliding_and_sink() {
        // stored 16, new 4, window 8, sink 2: current_end=20, read_start=max(2, 20-8)=12,
        // window = sink [0,2) ∪ tail [12,16).
        let mut cache = CausalKvCache::new(1, 8, 2);
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
        let (k, _) = cache.layer_kv(0).expect("layer 0 populated");
        k.as_slice::<f32>().to_vec()
    }

    #[test]
    fn global_regime_retains_everything() {
        // max_attention_size huge ⇒ no eviction ever: retained == committed, buffer is [0, committed).
        let mut cache = CausalKvCache::new(1, 1_000_000, 0);
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
        let mut cache = CausalKvCache::new(1, 4, 0);
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
        let mut cache = CausalKvCache::new(1, window, 0);
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
        let mut cache = CausalKvCache::new(1, 4, 2);
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
        let mut cache = CausalKvCache::new(1, 2, 4);
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
        let (_, v) = cache.layer_kv(0).expect("layer 0 populated");
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
