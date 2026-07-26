//! Krea Realtime 14B **causal autoregressive** transformer forward + persistent KV cache (sc-8436, S3).
//!
//! The S1 audit established that Krea Realtime 14B is Wan 2.1 T2V 14B weight-for-weight, so the whole
//! DiT — patchify, embeddings, adaLN-6vec modulation, text cross-attention, gated-GELU FFN, 3-axis
//! RoPE, the modulated head — is **reused verbatim** from [`mlx_gen_wan::WanTransformer`]. The single
//! net-new compute delta of the autoregressive regime is confined to **self-attention**, and it is
//! three coupled pieces (mirroring the reference `transformer/causal_model.py`):
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
//! live here. **S3 is the forward + cache only** — the Self-Forcing few-step renoise scheduler, the AR
//! chunk loop, and KV-cache recompute are S4/S5, VAE decode + `Generator` registration are S6/S7.

use mlx_gen::{Error, Result};
use mlx_gen_wan::patchify::unpatchify;
use mlx_gen_wan::WanTransformer;
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

/// Persistent per-layer self-attention KV cache for the Krea Realtime AR forward (sc-8436, S3).
///
/// Stores, per transformer layer, the running **post-RoPE keys + raw values** `[B, n, T, d]`
/// accumulated across every chunk generated so far (`T` = [`stored_tokens`](Self::stored_tokens)) —
/// mirroring `causal_model.py::_initialize_kv_cache` + the cached-forward append. Storage keeps every
/// key (global attention needs them all); the **read window**
/// `k[max(0, end - max_attention_size):end]` — plus any always-attended `[0, sink)` prefix — is applied
/// on read by [`window_prev`](Self::window_prev). For the shipped global checkpoint
/// (`max_attention_size = seq_length`, `sink = 0`) the window is the whole cache.
pub struct CausalKvCache {
    /// Per layer: `Some((k, v))` once populated, each `[B, n, stored_tokens, d]` (post-RoPE k, raw v).
    layers: Vec<Option<LayerKv>>,
    stored_tokens: usize,
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
            stored_tokens: 0,
            max_attention_size,
            sink_tokens,
        }
    }

    /// Number of tokens (per layer) currently cached — grows by the chunk length on each
    /// [`append`](Self::append). Mirrors the reference's running `current_start`.
    pub fn stored_tokens(&self) -> usize {
        self.stored_tokens
    }

    /// Number of transformer layers this cache serves.
    pub fn num_layers(&self) -> usize {
        self.layers.len()
    }

    /// `true` before the first chunk is appended.
    pub fn is_empty(&self) -> bool {
        self.stored_tokens == 0
    }

    /// The **global token positions** of the read window's cached (prev) keys for a query chunk of
    /// `s_new` new tokens: the always-attended sink prefix `[0, min(sink, stored))` unioned with the
    /// sliding tail `[read_start, stored)`, where
    /// `read_start = max(sink, (stored + s_new) − max_attention_size)` (clamped to `stored`). These are
    /// exactly the prev tokens the new chunk's queries attend, and the caller uses them to build the
    /// matching mask column positions. Empty before the first append.
    fn window_positions(&self, s_new: usize) -> Vec<i64> {
        if self.stored_tokens == 0 {
            return Vec::new();
        }
        let stored = self.stored_tokens;
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

    /// The windowed cached `(k, v)` per layer for a query chunk of `s_new` new tokens, alongside the
    /// **global token positions** of those keys (for mask construction). Returns `(vec![], vec![])`
    /// before the first append (first chunk has no history). When the window covers the whole stored
    /// history (the global / fits case) the stored handles are returned directly (no gather).
    pub fn window_prev(&self, s_new: usize) -> Result<(Vec<LayerKv>, Vec<i64>)> {
        let positions = self.window_positions(s_new);
        if positions.is_empty() {
            return Ok((Vec::new(), Vec::new()));
        }
        // Whole-history window ⇒ return stored handles as-is (cheap refcount clones, no gather).
        let whole = positions.len() == self.stored_tokens;
        let idx = if whole {
            None
        } else {
            Some(Array::from_slice(
                &positions.iter().map(|&p| p as i32).collect::<Vec<i32>>(),
                &[positions.len() as i32],
            ))
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

    /// Append this chunk's per-layer post-RoPE `(k, v)` `[B, n, s_new, d]` to the running cache (full
    /// retention — the read window is applied on [`window_prev`](Self::window_prev)). `new_kv` must
    /// carry exactly one `(k, v)` per layer.
    pub fn append(&mut self, new_kv: Vec<LayerKv>) -> Result<()> {
        if new_kv.len() != self.layers.len() {
            return Err(Error::Msg(format!(
                "krea causal: append expected {} layers, got {}",
                self.layers.len(),
                new_kv.len()
            )));
        }
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
        self.stored_tokens += s_new;
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

#[cfg(test)]
mod tests {
    use super::*;

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
        cache.stored_tokens = 8; // simulate one 8-token block cached
        let pos = cache.window_positions(8);
        assert_eq!(pos, (0..8).collect::<Vec<i64>>());
    }

    #[test]
    fn window_positions_sliding_and_sink() {
        // stored 16, new 4, window 8, sink 2: current_end=20, read_start=max(2, 20-8)=12,
        // window = sink [0,2) ∪ tail [12,16).
        let mut cache = CausalKvCache::new(1, 8, 2);
        cache.stored_tokens = 16;
        let pos = cache.window_positions(4);
        assert_eq!(pos, vec![0, 1, 12, 13, 14, 15]);
    }
}
