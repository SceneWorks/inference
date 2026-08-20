//! The H3 condition-encoder forward: token embedding → causal Qwen3 decoder layers → the hidden
//! state at [`SELECT_HIDDEN`](super::SELECT_HIDDEN). That `[B, L, 5120]` tensor is the `context`
//! the H3 DiT consumes, **one row per presentation token, none dropped** — the reference applies no
//! chat template, so there is no prefix to slice (sc-18741; see
//! [`APPLIES_CHAT_TEMPLATE`](super::tokenizer::APPLIES_CHAT_TEMPLATE)).
//!
//! # The off-by-one, stated once
//!
//! HF `output_hidden_states` indexing: `hidden_states[k]` is the state **after running `k` decoder
//! layers**, so `hidden_states[0]` is the raw embedding and `hidden_states[50]` — the card's "50th
//! layer" — is the OUTPUT of 0-indexed layer **49**. This encoder therefore:
//!
//! - loads and runs layers `0..=49` only (50 of the checkpoint's 64);
//! - never applies the final `norm` (the selected state is pre-final-norm);
//! - never loads `lm_head`.
//!
//! Layers 50-63 plus `lm_head` are consequently **dead weight for generation**. `tests/te_parity.rs`
//! proves this by asserting the port matches `hidden_states[50]` *and* differs from both
//! `hidden_states[49]` and `hidden_states[51]`, so a shift in either direction fails.

use mlx_rs::ops::{add, concatenate_axis};
use mlx_rs::{Array, Dtype};

use mlx_gen::nn::{build_mask, TextRope, TokenEmbedding};
use mlx_gen::weights::Weights;
use mlx_gen::{Error, Result};

use super::{embedding, join_key, MiniMaxH3TeConfig, Qwen3DecoderLayer, SPATIAL_MERGE};

/// MiniMax-H3's Qwen3-VL-32B condition encoder.
pub struct MiniMaxH3TextEncoder {
    embed_tokens: TokenEmbedding,
    /// Empty under a deferred load (sc-18662, rung 4) — the layers live in `stream` instead.
    layers: Vec<Qwen3DecoderLayer>,
    /// `Some` under [`LoadShape::DeferredMaterialization`](mlx_gen::gen_core::LoadShape). Mutually
    /// exclusive with a populated `layers`.
    stream: Option<crate::block_stream::TeBlockStream>,
    /// The per-step block window. `None` runs the resident stack.
    window: Option<mlx_gen::block_residency::BlockPlan>,
    /// Checked at every window boundary — the only cancellation point inside one encoder forward.
    window_cancel: mlx_gen::CancelFlag,
    rope: TextRope,
    /// 0-indexed decoder layer whose output is the context (`select_hidden - 1`).
    out_layer: usize,
    image_token_id: i32,
    /// `<|video_pad|>` — the pad a `ref2va` **video** reference's blocks occupy. `fl2va` never
    /// emits one, which is why the base grounded path scans for `image_token_id` alone.
    video_token_id: i32,
    mrope_section: [i32; 3],
    head_dim: i32,
    rope_theta: f32,
}

impl MiniMaxH3TextEncoder {
    /// Load from the `text_encoder` weights under `prefix` (normally
    /// [`LM_PREFIX`](super::LM_PREFIX) = `"model.language_model"`):
    /// `{prefix}.embed_tokens.weight` and `{prefix}.layers.{i}.…` for `i` in `0..select_hidden`.
    ///
    /// Deliberately loads **only** the layers it will run — `{prefix}.norm.weight`, layers
    /// `select_hidden..num_layers` and `lm_head.weight` are never touched.
    pub fn from_weights(w: &Weights, prefix: &str, cfg: &MiniMaxH3TeConfig) -> Result<Self> {
        let out_layer = cfg.out_layer()?;
        if out_layer as i32 >= cfg.num_layers {
            return Err(Error::Msg(format!(
                "minimax-h3 te: select_hidden {} needs layer {out_layer} but the encoder has {} \
                 layers",
                cfg.select_hidden, cfg.num_layers
            )));
        }

        let mut layers = Vec::with_capacity(out_layer + 1);
        for i in 0..=out_layer {
            layers.push(Qwen3DecoderLayer::from_weights(
                w,
                &join_key(prefix, &format!("layers.{i}")),
                cfg.num_heads,
                cfg.num_kv_heads,
                cfg.head_dim,
                cfg.rms_norm_eps,
            )?);
        }
        Ok(Self {
            embed_tokens: embedding(w, &join_key(prefix, "embed_tokens"))?,
            layers,
            stream: None,
            window: None,
            window_cancel: mlx_gen::CancelFlag::default(),
            rope: TextRope::new(cfg.head_dim, cfg.rope_theta),
            out_layer,
            image_token_id: cfg.image_token_id,
            video_token_id: cfg.video_token_id,
            mrope_section: cfg.mrope_section,
            head_dim: cfg.head_dim,
            rope_theta: cfg.rope_theta,
        })
    }

    /// **The rung-4 loader** (sc-18662): load only `embed_tokens` and defer the 50 decoder layers.
    ///
    /// `embed_tokens` stays resident because it is consumed **once**, before the stack runs, and a
    /// window that re-read it would pay the token table per window for a tensor already spent. It is
    /// the encoder's counterpart to the DiT's 17 I/O projections.
    ///
    /// The [`Weights`] map is dropped before returning: it is only lazily-mapped handles, but a
    /// retained one keeps every layer tensor reachable and makes the first window's release free
    /// nothing.
    pub fn from_dir_deferred(
        dir: impl AsRef<std::path::Path>,
        prefix: &str,
        cfg: &MiniMaxH3TeConfig,
    ) -> Result<Self> {
        let dir = dir.as_ref();
        let out_layer = cfg.out_layer()?;
        if out_layer as i32 >= cfg.num_layers {
            return Err(Error::Msg(format!(
                "minimax-h3 te: select_hidden {} needs layer {out_layer} but the encoder has {} \
                 layers",
                cfg.select_hidden, cfg.num_layers
            )));
        }
        let stream = crate::block_stream::TeBlockStream::new(dir, prefix, cfg.clone())?;
        let embed_tokens = {
            let w = Weights::from_dir(dir)?;
            embedding(&w, &join_key(prefix, "embed_tokens"))?
        };
        Ok(Self {
            embed_tokens,
            layers: Vec::new(),
            stream: Some(stream),
            window: None,
            window_cancel: mlx_gen::CancelFlag::default(),
            rope: TextRope::new(cfg.head_dim, cfg.rope_theta),
            out_layer,
            image_token_id: cfg.image_token_id,
            video_token_id: cfg.video_token_id,
            mrope_section: cfg.mrope_section,
            head_dim: cfg.head_dim,
            rope_theta: cfg.rope_theta,
        })
    }

    /// Select the per-step block window for a deferred load. `window` blocks materialized at once;
    /// `1` is the floor.
    pub fn set_block_window(&mut self, window: usize, cancel: mlx_gen::CancelFlag) -> Result<()> {
        let stream = self.stream.as_ref().ok_or_else(|| {
            Error::Msg(
                "minimax-h3 te: a block window was set on a RESIDENT encoder; load with \
                 `from_dir_deferred` for rung 4"
                    .into(),
            )
        })?;
        self.window = Some(stream.plan(window)?);
        self.window_cancel = cancel;
        Ok(())
    }

    /// Whether this load defers layer materialization.
    pub fn is_deferred(&self) -> bool {
        self.stream.is_some()
    }

    /// Layers currently materialized. `0` under a deferred load — the residency claim.
    pub fn resident_layers(&self) -> usize {
        self.layers.len()
    }

    /// The block window in force, or `None` on a resident stack.
    pub fn block_window(&self) -> Option<&mlx_gen::block_residency::BlockPlan> {
        self.window.as_ref()
    }

    /// **The one decoder-stack walk**, under either residency mode.
    ///
    /// Both public forwards route through this rather than each growing its own windowed twin: the
    /// grounded path differs only by a per-layer post-hook (the `deepstack` injection), which is what
    /// `after` carries. A second loop would be a second chance for the two to disagree about the
    /// residency, and the grounded path is the one a `ref2va` request takes — a rung reachable on
    /// only `t2va` would be declared for routes it does not serve.
    fn run_layers(
        &self,
        hidden: Array,
        cos: &Array,
        sin: &Array,
        mask: &Array,
        mut after: impl FnMut(usize, Array) -> Result<Array>,
    ) -> Result<Array> {
        match (&self.stream, &self.window) {
            (None, None) => {
                let mut hidden = hidden;
                for (i, layer) in self.layers.iter().enumerate() {
                    hidden = layer.forward(&hidden, cos, sin, mask)?;
                    hidden = after(i, hidden)?;
                }
                Ok(hidden)
            }
            (Some(stream), Some(plan)) => mlx_gen::block_residency::run_windowed(
                plan,
                &self.window_cancel,
                hidden,
                || stream.open(),
                |mut hidden: Array, view: &mut Weights, range: std::ops::Range<usize>| {
                    for i in range {
                        let layer = stream.materialize(view, i)?;
                        hidden = layer.forward(&hidden, cos, sin, mask)?;
                        hidden = after(i, hidden)?;
                        // The layer drops per iteration, not per window: at `window > 1` the
                        // alternative holds every layer of the window plus the activation.
                    }
                    Ok(hidden)
                },
                // LOAD-BEARING: MLX is lazy, so the carried activation still references this
                // window's weights until it is forced.
                |hidden: &Array| mlx_rs::transforms::eval([hidden]).map_err(Into::into),
            ),
            (Some(_), None) => Err(Error::Msg(
                "minimax-h3 te: a deferred encoder has no block window; call `set_block_window` \
                 before the forward, or load resident"
                    .into(),
            )),
            (None, Some(_)) => Err(Error::Msg(
                "minimax-h3 te: a resident encoder carries a block window; it would bound nothing"
                    .into(),
            )),
        }
    }

    /// How many decoder layers were actually loaded — `select_hidden`, not `num_layers`. Exposed so
    /// a caller (and the real-weight smoke) can prove the trim is real.
    ///
    /// Under a deferred load this is the **tapped depth the forward runs**, which is the question
    /// every existing caller is asking; [`Self::resident_layers`] is the residency observable.
    pub fn num_loaded_layers(&self) -> usize {
        match &self.stream {
            Some(stream) => stream.n_layers(),
            None => self.layers.len(),
        }
    }

    /// Quantize the token table + every loaded projection in place. Norms stay dense.
    ///
    /// **This is the definition [`crate::convert::is_te_pack_target`] is held against**, not a
    /// render path. Nothing on the render path calls it and nothing should: quantizing at load
    /// requires the dense weights resident first, which is the 53.07 GB this model's tiering exists
    /// to avoid ([`crate::memory_strategy::CONDITIONING_STAGE_PEAK_BYTES`]). Tiers ship
    /// **pre-quantized**; a packed `text_encoder/` auto-detects through
    /// [`mlx_gen::quant::lin`] / [`mlx_gen::quant::embedding`] with no dense transient.
    ///
    /// What it *is* good for is stating the pack set in executable form. `tests/quant_policy.rs`
    /// builds one encoder by quantizing in place and another from a
    /// [`quantize_minimax_h3_text_encoder`](crate::convert::quantize_minimax_h3_text_encoder)
    /// artifact and asserts the two are bit-identical, so the converter's suffix list cannot drift
    /// away from the modules' own idea of what is packable.
    pub fn quantize(&mut self, bits: i32) -> Result<()> {
        self.embed_tokens.quantize(bits, true)?;
        for layer in &mut self.layers {
            layer.quantize(bits)?;
        }
        Ok(())
    }

    /// The bit width every packed projection in this encoder was built at, or `None` if it loaded
    /// dense.
    ///
    /// `Err` when the loaded layers disagree — a tier assembled from a mixture of widths is a
    /// mis-built artifact, and reporting the first layer's width would hide it. The token table is
    /// checked separately by [`Self::token_table_is_quantized`]: it is the one tensor that a
    /// converter can leave dense while every Linear packs.
    pub fn packed_bits(&self) -> Result<Option<i32>> {
        let mut seen: Option<Option<i32>> = None;
        for (i, layer) in self.layers.iter().enumerate() {
            let bits = layer.packed_bits();
            match seen {
                None => seen = Some(bits),
                Some(first) if first != bits => {
                    return Err(Error::Msg(format!(
                        "minimax-h3 te: layer {i} is {bits:?}-bit but layer 0 is {first:?}-bit — \
                         the staged tier mixes quantization widths"
                    )))
                }
                Some(_) => {}
            }
        }
        Ok(seen.flatten())
    }

    /// `true` when the encoder's projections loaded packed rather than dense.
    pub fn is_quantized(&self) -> Result<bool> {
        Ok(self.packed_bits()?.is_some())
    }

    /// `true` when the token table loaded packed. Reported separately from [`Self::packed_bits`]
    /// because the embedding takes a different loader
    /// ([`mlx_gen::quant::embedding`]) and a converter can miss it while packing every Linear.
    pub fn token_table_is_quantized(&self) -> bool {
        self.embed_tokens.is_quantized()
    }

    /// Device bytes this encoder holds — packed triples summed as they actually sit in memory, not
    /// as the logical shape they decode to.
    ///
    /// This is the quantity a tier is judged on. `get_active_memory` after a forced materialization
    /// must land within a small margin of it; anything larger means something dense is still
    /// resident that the tier believed it had packed.
    pub fn nbytes(&self) -> usize {
        self.embed_tokens_nbytes()
            + self
                .layers
                .iter()
                .map(Qwen3DecoderLayer::nbytes)
                .sum::<usize>()
    }

    fn embed_tokens_nbytes(&self) -> usize {
        use mlx_gen::nn::TokenEmbedding;
        match &self.embed_tokens {
            TokenEmbedding::Dense(w) => w.nbytes(),
            TokenEmbedding::Quantized {
                wq, scales, biases, ..
            } => wq.nbytes() + scales.nbytes() + biases.nbytes(),
        }
    }

    /// Text-only (t2va) conditioning. `input_ids` / `attention_mask`: `[b, s]` int32. Returns the
    /// DiT context `[b, s, hidden]` — one row per presentation token.
    ///
    /// Uses plain 1-D RoPE: with no vision tokens Qwen3-VL's interleaved MRoPE sections all index
    /// the same sequential position, so it reduces exactly to standard RoPE.
    pub fn forward(&self, input_ids: &Array, attention_mask: &Array) -> Result<Array> {
        let sh = input_ids.shape();
        let (b, s) = (sh[0], sh[1]);
        let (cos, sin) = self.rope.forward(s)?;
        let mask = build_mask(attention_mask, b, s)?;

        let hidden = self.embed_tokens.forward(input_ids)?;
        self.run_layers(hidden, &cos, &sin, &mask, |_, h| Ok(h))
    }

    /// **Vision-grounded** conditioning (the fl2va / Ref2VA image path): run the encoder with each
    /// reference's tower features spliced over its `<|image_pad|>` block and 3-D interleaved MRoPE
    /// positions, so the LM "sees" the references while reading the prompt.
    ///
    /// Mirrors [`forward`](Self::forward) but (a) replaces the `<|image_pad|>` embeddings with the
    /// tower's merged `image_embeds` `[nⱼ, hidden]`, (b) additively injects each reference's
    /// `deepstack` feature at those positions for the first `deepstack.len()` layers, and (c) uses
    /// interleaved MRoPE — the image block carries its 2-D merged grid position, text stays
    /// sequential. Returns the same `[b, s, hidden]` context. `b = 1`.
    pub fn forward_with_images(
        &self,
        input_ids: &Array,
        attention_mask: &Array,
        image_embeds: &[Array],
        deepstack: &[Vec<Array>],
        grids: &[[i32; 3]],
    ) -> Result<Array> {
        let pads = [self.image_token_id];
        self.forward_grounded(
            input_ids,
            attention_mask,
            image_embeds,
            deepstack,
            grids,
            &pads,
        )
    }

    /// The **`ref2va`** grounded forward: splice reference features into runs of *both*
    /// `<|image_pad|>` and `<|video_pad|>`.
    ///
    /// `embeds`, `deepstack` and `grids` are in **sequence order** — the order the pad runs appear
    /// in the presentation, which for `ref2va` is request order across modalities.
    ///
    /// # Why sequence order reproduces the reference's per-modality batching
    ///
    /// Qwen3-VL batches vision tensors *per modality* and fills the n-th pad run of a modality with
    /// the n-th entry of that modality's batch. Relative order **within** a modality is preserved by
    /// request order, so walking runs in sequence order while consuming features in request order
    /// yields exactly the same assignment — with one cursor instead of two, and therefore no way
    /// for the two to drift apart on an interleaved request.
    ///
    /// A run never spans two *different* pads, so an image block immediately followed by a video
    /// block stays two runs and two references.
    pub fn forward_with_references(
        &self,
        input_ids: &Array,
        attention_mask: &Array,
        embeds: &[Array],
        deepstack: &[Vec<Array>],
        grids: &[[i32; 3]],
    ) -> Result<Array> {
        let pads = [self.image_token_id, self.video_token_id];
        self.forward_grounded(input_ids, attention_mask, embeds, deepstack, grids, &pads)
    }

    fn forward_grounded(
        &self,
        input_ids: &Array,
        attention_mask: &Array,
        image_embeds: &[Array],
        deepstack: &[Vec<Array>],
        grids: &[[i32; 3]],
        pad_ids: &[i32],
    ) -> Result<Array> {
        let sh = input_ids.shape();
        let (b, s) = (sh[0], sh[1]);
        let ids_arr = input_ids.as_dtype(Dtype::Int32)?;
        let ids: Vec<i32> = ids_arr.as_slice::<i32>().to_vec();

        let runs = image_token_runs(&ids, pad_ids, s);
        if runs.is_empty() {
            return Err(Error::Msg(
                "minimax-h3 te (grounded): prompt has no vision-pad tokens".into(),
            ));
        }
        if runs.len() != image_embeds.len() || runs.len() != grids.len() {
            return Err(Error::Msg(format!(
                "minimax-h3 te (grounded): {} vision-pad run(s) but {} embeds / {} grids",
                runs.len(),
                image_embeds.len(),
                grids.len()
            )));
        }

        let mut hidden = self.embed_tokens.forward(input_ids)?;
        let dt = hidden.dtype();
        for (&(start, end), emb) in runs.iter().zip(image_embeds) {
            let img = emb.expand_dims(0)?.as_dtype(dt)?;
            hidden = replace_seq(&hidden, &img, start, end)?;
        }

        let (pt, ph, pw) = mrope_positions_multi(&ids, pad_ids, grids);
        let (cos, sin) = mrope_cos_sin(
            &pt,
            &ph,
            &pw,
            self.head_dim,
            self.rope_theta,
            self.mrope_section,
            dt,
        )?;
        let mask = build_mask(attention_mask, b, s)?;

        let hidden = self.run_layers(hidden, &cos, &sin, &mask, |i, mut h| {
            for (&(start, end), ds_img) in runs.iter().zip(deepstack) {
                if i < ds_img.len() {
                    let mid = slice_seq(&h, start, end)?;
                    let inj = add(&mid, &ds_img[i].expand_dims(0)?.as_dtype(dt)?)?;
                    h = replace_seq(&h, &inj, start, end)?;
                }
            }
            Ok(h)
        })?;
        let _ = self.out_layer; // the walk runs exactly `out_layer + 1` layers by construction
        Ok(hidden)
    }
}

// ── Vision-grounded helpers ──────────────────────────────────────────────────────────────────────

/// Slice `[b, s, d]` along the sequence axis to `[start, end)` via a contiguous split.
fn slice_seq(x: &Array, start: i32, end: i32) -> Result<Array> {
    Ok(x.split_axis(&[start, end], 1)?.swap_remove(1))
}

/// Replace `x[:, start:end, :]` with `repl` by concatenating the surrounding contiguous splits.
fn replace_seq(x: &Array, repl: &Array, start: i32, end: i32) -> Result<Array> {
    let mut parts = x.split_axis(&[start, end], 1)?;
    let after = parts.swap_remove(2);
    let before = parts.swap_remove(0);
    Ok(concatenate_axis(&[&before, repl, &after], 1)?)
}

/// Contiguous runs of `image_token_id` in `ids` (`[start, end)` per run), in sequence order — one
/// run per reference (the template separates references with `<|vision_end|><|vision_start|>`).
fn image_token_runs(ids: &[i32], pad_ids: &[i32], s: i32) -> Vec<(i32, i32)> {
    let is_pad = |v: i32| pad_ids.contains(&v);
    let mut runs = Vec::new();
    let mut i = 0i32;
    while i < s {
        if is_pad(ids[i as usize]) {
            let start = i;
            // A run does not span two DIFFERENT pads: `<|image_pad|>` and `<|video_pad|>` blocks
            // are separate references even when adjacent, and merging them would splice one
            // reference's embeds across two.
            let first = ids[i as usize];
            while i < s && ids[i as usize] == first {
                i += 1;
            }
            runs.push((start, i));
        } else {
            i += 1;
        }
    }
    runs
}

/// Multi-reference 3-D MRoPE positions (mirrors Qwen3-VL `get_rope_index` over `image_grid_thw`):
/// text tokens advance `(i, i, i)`; the `k`-th image block at offset `cur` gets `t = cur`,
/// `h = cur + row`, `w = cur + col` over its `(h/merge)×(w/merge)` merged grid, then
/// `cur += max(h, w) / merge`.
fn mrope_positions_multi(
    ids: &[i32],
    pad_ids: &[i32],
    grids: &[[i32; 3]],
) -> (Vec<i32>, Vec<i32>, Vec<i32>) {
    let (mut pt, mut ph, mut pw) = (Vec::new(), Vec::new(), Vec::new());
    let mut cur = 0i32;
    let mut img_i = 0usize;
    let mut i = 0usize;
    while i < ids.len() {
        if pad_ids.contains(&ids[i]) && img_i < grids.len() {
            let g = grids[img_i];
            let (llm_h, llm_w) = (g[1] / SPATIAL_MERGE, g[2] / SPATIAL_MERGE);
            let step = g[1].max(g[2]) / SPATIAL_MERGE;
            for idx in 0..(llm_h * llm_w) {
                pt.push(cur);
                ph.push(cur + idx / llm_w);
                pw.push(cur + idx % llm_w);
            }
            cur += step;
            i += (llm_h * llm_w) as usize;
            img_i += 1;
        } else {
            pt.push(cur);
            ph.push(cur);
            pw.push(cur);
            cur += 1;
            i += 1;
        }
    }
    (pt, ph, pw)
}

/// Build the **interleaved** MRoPE `cos`/`sin` `[1, s, head_dim]`, cast to `dt`.
///
/// For each of the `head_dim/2` frequencies `j`: within the first `section[1]·3` indices
/// `j % 3 == 1` takes the H position, within `section[2]·3` `j % 3 == 2` takes W, else T. This is
/// the `mrope_interleaved: true` assignment the shipped config declares — see
/// `MiniMaxH3TeConfig::mrope_interleaved`, which no tensor can witness.
fn mrope_cos_sin(
    pt: &[i32],
    ph: &[i32],
    pw: &[i32],
    head_dim: i32,
    theta: f32,
    section: [i32; 3],
    dt: Dtype,
) -> Result<(Array, Array)> {
    let s = pt.len();
    let half = (head_dim / 2) as usize;
    let sec_h = (section[1] * 3) as usize;
    let sec_w = (section[2] * 3) as usize;
    let inv: Vec<f32> = (0..half)
        .map(|j| (theta as f64).powf(-(2.0 * j as f64) / head_dim as f64) as f32)
        .collect();

    let hd = head_dim as usize;
    let mut emb = vec![0f32; s * hd];
    for i in 0..s {
        for j in 0..half {
            let pos = if j < sec_h && j % 3 == 1 {
                ph[i]
            } else if j < sec_w && j % 3 == 2 {
                pw[i]
            } else {
                pt[i]
            };
            let angle = pos as f32 * inv[j];
            emb[i * hd + j] = angle;
            emb[i * hd + half + j] = angle;
        }
    }
    let arr = Array::from_slice(&emb, &[1, s as i32, head_dim]);
    Ok((arr.cos()?.as_dtype(dt)?, arr.sin()?.as_dtype(dt)?))
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Text-only MRoPE is a plain sequential ramp on all three axes — the reduction that lets the
    /// text path keep using 1-D `TextRope`. If this ever stopped holding, `forward` would need the
    /// full MRoPE build too.
    #[test]
    fn mrope_positions_text_only_is_sequential() {
        let (pt, ph, pw) = mrope_positions_multi(&[10, 11, 12, 13], &[151655], &[]);
        assert_eq!(pt, vec![0, 1, 2, 3]);
        assert_eq!(pt, ph);
        assert_eq!(pt, pw);
    }

    /// Text tokens advance sequentially; an image block sits at the running offset with its 2-D
    /// merged grid on the H/W axes, and the cursor jumps by `max(h,w)/merge` after it.
    #[test]
    fn mrope_positions_lays_out_text_then_image_grid() {
        let img = 151655;
        // [txt, txt, img×4, txt] with a 4×4 patch grid → merged 2×2 = 4 image tokens.
        let ids = vec![1, 2, img, img, img, img, 3];
        let (pt, ph, pw) = mrope_positions_multi(&ids, &[img], &[[1, 4, 4]]);
        assert_eq!(pt, vec![0, 1, 2, 2, 2, 2, 4]);
        assert_eq!(ph, vec![0, 1, 2, 2, 3, 3, 4]);
        assert_eq!(pw, vec![0, 1, 2, 3, 2, 3, 4]);
    }

    /// Two references each get their own grid and their own cursor advance.
    #[test]
    fn mrope_positions_handles_two_references() {
        let img = 151655;
        let ids = vec![1, img, img, img, img, 2, img, 3];
        let (pt, _, _) = mrope_positions_multi(&ids, &[img], &[[1, 4, 4], [1, 2, 2]]);
        // txt@0; img0 (2×2 merged, 4 tokens) @cur=1, then cur += 4/2 = 2 → 3; txt@3 → cur 4;
        // img1 (1×1 merged, 1 token) @cur=4, then cur += 2/2 = 1 → 5; txt@5.
        assert_eq!(pt, vec![0, 1, 1, 1, 1, 3, 4, 5]);
    }

    /// Contiguous `<|image_pad|>` runs are found one per reference.
    #[test]
    fn image_token_runs_are_per_reference() {
        let img = 151655;
        let ids = vec![1, img, img, 2, img, 3];
        assert_eq!(
            image_token_runs(&ids, &[img], ids.len() as i32),
            vec![(1, 3), (4, 5)]
        );
        assert!(image_token_runs(&[1, 2, 3], &[img], 3).is_empty());
    }

    /// The interleaved assignment must actually route H and W frequencies to different axes. With
    /// distinct T/H/W positions the resulting angles must differ across the three groups — a
    /// contiguous (non-interleaved) implementation would put them in different index ranges and
    /// this pins that we use the interleaved one.
    #[test]
    fn interleaved_mrope_routes_h_and_w_to_distinct_frequencies() {
        let head_dim = 24;
        let section = [4, 4, 4]; // sums to head_dim/2 = 12
        let (t, h, w) = (vec![0], vec![5], vec![9]);
        let (cos_a, _) =
            mrope_cos_sin(&t, &h, &w, head_dim, 10_000.0, section, Dtype::Float32).unwrap();
        // Same T, but H and W swapped → different tensor, proving both axes are actually read.
        let (cos_b, _) =
            mrope_cos_sin(&t, &w, &h, head_dim, 10_000.0, section, Dtype::Float32).unwrap();
        let a: Vec<f32> = cos_a.as_slice::<f32>().to_vec();
        let b: Vec<f32> = cos_b.as_slice::<f32>().to_vec();
        assert_ne!(a, b, "H and W must land on distinct frequency slots");
        // And the two halves of the embedding are duplicated (`emb = cat(f, f)`).
        let half = (head_dim / 2) as usize;
        assert_eq!(a[..half], a[half..]);
    }

    /// A text-only prompt must produce the same `cos`/`sin` as plain 1-D RoPE, since all three
    /// sections index the same position. This is the invariant `forward` relies on.
    #[test]
    fn text_only_mrope_matches_plain_rope() {
        let head_dim = 24;
        let theta = 5_000_000.0;
        let pos = vec![0, 1, 2, 3];
        let (cos_m, sin_m) =
            mrope_cos_sin(&pos, &pos, &pos, head_dim, theta, [4, 4, 4], Dtype::Float32).unwrap();
        let (cos_r, sin_r) = TextRope::new(head_dim, theta).forward(4).unwrap();
        let close = |a: &Array, b: &Array| {
            let a = a.as_slice::<f32>().to_vec();
            let b = b.as_slice::<f32>().to_vec();
            a.iter().zip(&b).all(|(x, y)| (x - y).abs() < 1e-5)
        };
        assert!(close(&cos_m, &cos_r), "cos diverged from 1-D RoPE");
        assert!(close(&sin_m, &sin_r), "sin diverged from 1-D RoPE");
    }
}
