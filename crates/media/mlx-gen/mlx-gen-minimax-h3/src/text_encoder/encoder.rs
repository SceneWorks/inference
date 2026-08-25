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

/// The two vision-grounded routes.
///
/// One value carries the pad set the presentation uses *and* both refusal labels, so a route
/// cannot be spliced as one thing and named another in a refusal — the pairing was previously two
/// independent parameters that every call site had to keep in step.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum GroundedRoute {
    /// `fl2va` — keyframes only, so `<|image_pad|>` alone.
    Fl2va,
    /// `ref2va` — image *and* video references, so both pads.
    Ref2va,
}

impl GroundedRoute {
    /// The route name, used when a whole context comes back degenerate.
    fn context_label(self) -> &'static str {
        match self {
            Self::Fl2va => "fl2va",
            Self::Ref2va => "ref2va",
        }
    }

    /// The route name at the token-embedding stage — the sc-17153 detection point.
    fn embed_label(self) -> &'static str {
        match self {
            Self::Fl2va => "fl2va token embedding",
            Self::Ref2va => "ref2va token embedding",
        }
    }
}

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
    ///
    /// # The token table is FORCED before that map is dropped (sc-17153)
    ///
    /// MLX maps per tensor and materializes on first read, so `embedding` alone hands back a lazy
    /// handle: measured, `get_active_memory()` was **0 B** when this constructor returned while
    /// [`Self::nbytes`] already claimed 1,555,824,640 B, and the table only became resident
    /// (1,556,316,352 B active) part-way through the *first window*. Two things were wrong with
    /// that:
    ///
    /// * the residency this constructor documents — "`embed_tokens` stays resident because it is
    ///   consumed once, before the stack runs" — was not true at the point it is stated, and
    ///   [`Self::nbytes`]'s own contract ("`get_active_memory` after a forced materialization must
    ///   land within a small margin of it") had no forced materialization to be measured against;
    /// * the 1.556 GB read landed *inside* a bounded window, so the window-1 peak this rung
    ///   publishes ([`RUNG4_TE_WINDOWED_PEAK_BYTES`](crate::memory_strategy::RUNG4_TE_WINDOWED_PEAK_BYTES))
    ///   was charged the token table's first materialization rather than the window's own
    ///   residency, and the table's `Load` was still outstanding against a *dropped* map while the
    ///   window held a second, independent map over the same 14 shard files.
    ///
    /// Forcing it here costs no extra bytes — the table has to be resident for the forward either
    /// way — and is the idiom `mlx-gen-krea`'s encoder already uses
    /// ([`TokenEmbedding::materialize_weights`]).
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
            let embed_tokens = embedding(&w, &join_key(prefix, "embed_tokens"))?;
            // LOAD-BEARING, and it must happen before `w` goes out of scope — see the doc above.
            embed_tokens.materialize_weights()?;
            embed_tokens
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
    ///
    /// Every context this encoder hands the DiT is this function's return value, so the sc-17153
    /// screen here is the **whole-stack backstop**: it catches a context that came out degenerate
    /// however it got that way. It is explicitly *not* sufficient on its own — see
    /// [`Self::embed_screened`] for the screen that catches the hazard this epic is actually
    /// chasing, and why a post-stack screen cannot. `producer` is the route's name, so a refusal
    /// says which one hit it.
    fn run_layers(
        &self,
        producer: &'static str,
        hidden: Array,
        cos: &Array,
        sin: &Array,
        mask: &Array,
        mut after: impl FnMut(usize, Array) -> Result<Array>,
    ) -> Result<Array> {
        let context = self.walk_layers(hidden, cos, sin, mask, &mut after)?;
        // A cancelled walk short-circuits above; anything that reaches here is a context the DiT
        // would otherwise denoise against, so it is screened before it can leave the encoder.
        super::degeneracy::refuse_if_degenerate(producer, &context)?;
        Ok(context)
    }

    /// The token-embedding lookup, screened — **this is the sc-17153 detection point.**
    ///
    /// `hidden_states[0]` is the tensor the hazard actually zeroes: the fault is a first-evaluation
    /// read of the 1.556 GB dense `embed_tokens` table returning zeros, and everything downstream is
    /// propagation. Screening here rather than only after the stack matters because **on the
    /// grounded routes the post-stack screen cannot fire at all**:
    /// [`Self::forward_grounded`] *overwrites* the `<|image_pad|>` / `<|video_pad|>` rows with the
    /// vision tower's live embeds (`replace_seq`, a replace and not a blend) before the stack runs.
    /// With a zeroed table the text rows are dead but the vision rows are not, so `max|context| > 0`
    /// and a whole-context screen returns `Ok(None)` while the render proceeds carrying **zero
    /// prompt signal** — for a 1024px keyframe that is ~1024 live vision rows against ~100 dead text
    /// rows. The tower's embeds are produced independently and screened separately in
    /// [`run_vision`](super::run_vision), so the two cannot go zero together and mask each other.
    ///
    /// The grounded routes are not the safer case, either: they map an extra `VISION_SHARD` and run
    /// the tower before the encoder, so their race window is at least as wide as `t2va`'s.
    ///
    /// Forcing the embedding here does not add work — the stack's first layer consumes it
    /// immediately — and it costs no extra peak, since the token table has to be resident for the
    /// forward regardless.
    fn embed_screened(&self, producer: &'static str, input_ids: &Array) -> Result<Array> {
        let hidden = self.embed_tokens.forward(input_ids)?;
        super::degeneracy::refuse_if_degenerate(producer, &hidden)?;
        Ok(hidden)
    }

    /// [`Self::run_layers`] without the screen — the residency dispatch on its own.
    fn walk_layers(
        &self,
        hidden: Array,
        cos: &Array,
        sin: &Array,
        mask: &Array,
        after: &mut impl FnMut(usize, Array) -> Result<Array>,
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

        let hidden = self.embed_screened("t2va token embedding", input_ids)?;
        self.run_layers("t2va", hidden, &cos, &sin, &mask, |_, h| Ok(h))
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
        self.forward_grounded(
            GroundedRoute::Fl2va,
            input_ids,
            attention_mask,
            image_embeds,
            deepstack,
            grids,
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
        self.forward_grounded(
            GroundedRoute::Ref2va,
            input_ids,
            attention_mask,
            embeds,
            deepstack,
            grids,
        )
    }

    fn forward_grounded(
        &self,
        route: GroundedRoute,
        input_ids: &Array,
        attention_mask: &Array,
        image_embeds: &[Array],
        deepstack: &[Vec<Array>],
        grids: &[[i32; 3]],
    ) -> Result<Array> {
        let sh = input_ids.shape();
        let (b, s) = (sh[0], sh[1]);
        let ids_arr = input_ids.as_dtype(Dtype::Int32)?;
        let ids: Vec<i32> = ids_arr.as_slice::<i32>().to_vec();

        let image_only = [self.image_token_id];
        let image_and_video = [self.image_token_id, self.video_token_id];
        let pad_ids: &[i32] = match route {
            GroundedRoute::Fl2va => &image_only,
            GroundedRoute::Ref2va => &image_and_video,
        };

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

        // SCREENED BEFORE THE SPLICE. Once `replace_seq` has written the tower's live embeds over
        // the pad rows, a zeroed token table is no longer visible in any whole-tensor reduction —
        // see `embed_screened`. This is the only point on a grounded route where it still is.
        let mut hidden = self.embed_screened(route.embed_label(), input_ids)?;
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

        let hidden = self.run_layers(
            route.context_label(),
            hidden,
            &cos,
            &sin,
            &mask,
            |i, mut h| {
                for (&(start, end), ds_img) in runs.iter().zip(deepstack) {
                    if i < ds_img.len() {
                        let mid = slice_seq(&h, start, end)?;
                        let inj = add(&mid, &ds_img[i].expand_dims(0)?.as_dtype(dt)?)?;
                        h = replace_seq(&h, &inj, start, end)?;
                    }
                }
                Ok(h)
            },
        )?;
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

    const TEST_PREFIX: &str = "language_model";

    /// A tiny but structurally faithful encoder config — same module graph, ~1e-7 of the weights.
    fn tiny_config() -> MiniMaxH3TeConfig {
        MiniMaxH3TeConfig {
            hidden_size: 16,
            num_layers: 2,
            num_heads: 2,
            num_kv_heads: 1,
            head_dim: 8,
            intermediate_size: 32,
            vocab_size: 32,
            // Inside `vocab_size`, so the grounded fixtures can put real pad ids in `input_ids`
            // without the embedding gather indexing off the end of a 32-row table.
            image_token_id: 7,
            video_token_id: 8,
            select_hidden: 2,
            mrope_section: [2, 1, 1],
            ..MiniMaxH3TeConfig::qwen3_vl_32b()
        }
    }

    /// A deterministic, non-constant, sign-mixed bf16 weight.
    fn weight(seed: u64, shape: &[i32]) -> Array {
        let n: usize = shape.iter().map(|&d| d as usize).product();
        let mut state = seed.wrapping_mul(6_364_136_223_846_793_005).wrapping_add(1);
        let vals: Vec<f32> = (0..n)
            .map(|_| {
                state = state
                    .wrapping_mul(6_364_136_223_846_793_005)
                    .wrapping_add(1_442_695_040_888_963_407);
                ((state >> 33) as f32 / (1u64 << 31) as f32) - 0.5
            })
            .collect();
        Array::from_slice(&vals, shape)
            .as_dtype(Dtype::Bfloat16)
            .expect("bf16 cast")
    }

    /// A complete dense weight map for [`tiny_config`]. `live_token_table` chooses whether
    /// `embed_tokens.weight` is populated or **zeroed** — every other tensor is live either way.
    ///
    /// Zeroing only the token table is how the sc-17153 hazard would present if it originates at the
    /// embedding lookup, and it is enough on its own to produce an exactly-zero context: RMSNorm(0),
    /// SDPA over zeroed q/k/v, SwiGLU(0) and both residual adds all preserve zero, so a zeroed
    /// `hidden_states[0]` walks the whole stack unchanged. That property is what
    /// [`a_zeroed_token_table_produces_an_exactly_zero_context`] pins.
    fn tiny_weights(cfg: &MiniMaxH3TeConfig, live_token_table: bool) -> Weights {
        let mut w = Weights::empty();
        let (h, q_out, kv_out) = (
            cfg.hidden_size,
            cfg.num_heads * cfg.head_dim,
            cfg.num_kv_heads * cfg.head_dim,
        );
        let table = if live_token_table {
            weight(1, &[cfg.vocab_size, h])
        } else {
            Array::from_slice(
                &vec![0f32; (cfg.vocab_size * h) as usize],
                &[cfg.vocab_size, h],
            )
            .as_dtype(Dtype::Bfloat16)
            .expect("bf16 cast")
        };
        w.insert(join_key(TEST_PREFIX, "embed_tokens.weight"), table);

        let mut seed = 1u64;
        for i in 0..cfg.num_layers {
            let layer = join_key(TEST_PREFIX, &format!("layers.{i}"));
            for (leaf, shape) in [
                ("input_layernorm.weight", vec![h]),
                ("post_attention_layernorm.weight", vec![h]),
                ("self_attn.q_proj.weight", vec![q_out, h]),
                ("self_attn.k_proj.weight", vec![kv_out, h]),
                ("self_attn.v_proj.weight", vec![kv_out, h]),
                ("self_attn.o_proj.weight", vec![h, q_out]),
                ("self_attn.q_norm.weight", vec![cfg.head_dim]),
                ("self_attn.k_norm.weight", vec![cfg.head_dim]),
                ("mlp.gate_proj.weight", vec![cfg.intermediate_size, h]),
                ("mlp.up_proj.weight", vec![cfg.intermediate_size, h]),
                ("mlp.down_proj.weight", vec![h, cfg.intermediate_size]),
            ] {
                seed += 1;
                w.insert(join_key(&layer, leaf), weight(seed, &shape));
            }
        }
        w
    }

    /// A 6-token presentation and its all-ones mask.
    fn tiny_prompt(cfg: &MiniMaxH3TeConfig) -> (Array, Array) {
        let ids: Vec<i32> = (0..6).map(|i| (i * 5 + 3) % cfg.vocab_size).collect();
        let n = ids.len() as i32;
        (
            Array::from_slice(&ids, &[1, n]),
            Array::from_slice(&vec![1i32; ids.len()], &[1, n]),
        )
    }

    /// One grounded presentation and the tower output that grounds it.
    struct TinyGrounded {
        ids: Vec<i32>,
        input_ids: Array,
        mask: Array,
        embeds: Vec<Array>,
        deepstack: Vec<Vec<Array>>,
        grids: Vec<[i32; 3]>,
    }

    /// A grounded presentation: `[txt, txt, img, img, txt, txt]` with a 2-token merged image block,
    /// plus one reference's **live** embeds. The grid `[1, 4, 2]` merges to `(4/2)·(2/2) = 2`
    /// tokens, matching the pad run. `deepstack` is empty — the injection is not what is under test.
    fn tiny_grounded(cfg: &MiniMaxH3TeConfig) -> TinyGrounded {
        let img = cfg.image_token_id;
        let ids = vec![1, 2, img, img, 3, 4];
        let n = ids.len() as i32;
        TinyGrounded {
            input_ids: Array::from_slice(&ids, &[1, n]),
            mask: Array::from_slice(&vec![1i32; ids.len()], &[1, n]),
            embeds: vec![weight(99, &[2, cfg.hidden_size])],
            deepstack: vec![vec![]],
            grids: vec![[1, 4, 2]],
            ids,
        }
    }

    /// **The blocker this screen exists for.** On a grounded route the vision embeds *overwrite* the
    /// pad rows before the stack runs, so a zeroed token table leaves `max|context| > 0` and a
    /// post-stack screen returns `Ok(None)` — the render proceeds carrying zero prompt signal.
    ///
    /// Here: zeroed table, **live** embeds. The refusal must come from
    /// [`MiniMaxH3TextEncoder::embed_screened`], before the splice. Delete that call and this test
    /// reds while `forward_refuses_an_all_zero_context` stays green — which is exactly the hole.
    #[test]
    fn forward_with_images_refuses_a_zeroed_token_table_the_context_screen_cannot_see() {
        let cfg = tiny_config();
        let te = MiniMaxH3TextEncoder::from_weights(&tiny_weights(&cfg, false), TEST_PREFIX, &cfg)
            .expect("build tiny encoder");
        let g = tiny_grounded(&cfg);

        let message = te
            .forward_with_images(&g.input_ids, &g.mask, &g.embeds, &g.deepstack, &g.grids)
            .expect_err("a zeroed token table must be refused on the grounded route too")
            .to_string();
        assert!(
            message.contains("fl2va token embedding"),
            "the refusal must name the embedding stage, not the context: {message}"
        );
        assert!(message.contains("sc-17153"), "{message}");
    }

    /// The same fixture with a live table must pass, or the arm above would be green for a screen
    /// that rejects every grounded forward.
    #[test]
    fn forward_with_images_returns_a_healthy_grounded_context() {
        let cfg = tiny_config();
        let te = MiniMaxH3TextEncoder::from_weights(&tiny_weights(&cfg, true), TEST_PREFIX, &cfg)
            .expect("build tiny encoder");
        let g = tiny_grounded(&cfg);

        let context = te
            .forward_with_images(&g.input_ids, &g.mask, &g.embeds, &g.deepstack, &g.grids)
            .expect("a live grounded encoder must forward");
        assert_eq!(context.shape(), &[1, 6, cfg.hidden_size]);
    }

    /// **Why the post-stack screen is not enough**, stated as a measurement rather than an argument:
    /// with a zeroed token table and live vision embeds, the context the stack produces has
    /// `max|context| > 0`, so [`super::degeneracy::inspect_conditioning`] over it reports *nothing
    /// wrong*. If this ever starts reporting a defect, the grounded splice stopped overwriting the
    /// pad rows and `embed_screened`'s justification needs re-reading.
    #[test]
    fn a_grounded_context_hides_a_zeroed_token_table_from_the_post_stack_screen() {
        let cfg = tiny_config();
        let te = MiniMaxH3TextEncoder::from_weights(&tiny_weights(&cfg, false), TEST_PREFIX, &cfg)
            .expect("build tiny encoder");
        let g = tiny_grounded(&cfg);

        // Rebuild `forward_grounded`'s graph without the embedding screen, to observe what a
        // context-only screen would have seen.
        let hidden = te.embed_tokens.forward(&g.input_ids).unwrap();
        let dt = hidden.dtype();
        let img = g.embeds[0].expand_dims(0).unwrap().as_dtype(dt).unwrap();
        let hidden = replace_seq(&hidden, &img, 2, 4).unwrap();
        let (pt, ph, pw) = mrope_positions_multi(&g.ids, &[cfg.image_token_id], &g.grids);
        let (cos, sin) = mrope_cos_sin(
            &pt,
            &ph,
            &pw,
            cfg.head_dim,
            cfg.rope_theta,
            cfg.mrope_section,
            dt,
        )
        .unwrap();
        let m = build_mask(&g.mask, 1, 6).unwrap();
        let context = te
            .walk_layers(hidden, &cos, &sin, &m, &mut |_, h| Ok(h))
            .unwrap();

        assert!(
            crate::text_encoder::degeneracy::inspect_conditioning("fl2va", &context)
                .unwrap()
                .is_none(),
            "premise of `embed_screened`: a whole-context screen is blind to this"
        );
    }

    /// **The mutation target for the post-stack backstop.** A context that goes bad *inside* the
    /// stack is invisible to [`MiniMaxH3TextEncoder::embed_screened`] — the embedding is healthy —
    /// so only [`MiniMaxH3TextEncoder::run_layers`]'s screen can catch it.
    ///
    /// Produced, not faked: one layer's `input_layernorm` weight carries a NaN, so `hidden_states[0]`
    /// is fine and every state after layer 0 is not. Delete the `refuse_if_degenerate` call in
    /// `run_layers` and this reds while every other arm stays green.
    #[test]
    fn forward_refuses_a_context_the_stack_itself_poisoned() {
        let cfg = tiny_config();
        let mut w = tiny_weights(&cfg, true);
        let poisoned = Array::from_slice(
            &{
                let mut v = vec![1.0f32; cfg.hidden_size as usize];
                v[3] = f32::NAN;
                v
            },
            &[cfg.hidden_size],
        )
        .as_dtype(Dtype::Bfloat16)
        .unwrap();
        w.insert(
            join_key(TEST_PREFIX, "layers.0.input_layernorm.weight"),
            poisoned,
        );
        let te = MiniMaxH3TextEncoder::from_weights(&w, TEST_PREFIX, &cfg).expect("build encoder");
        let (ids, mask) = tiny_prompt(&cfg);

        let message = te
            .forward(&ids, &mask)
            .expect_err("a NaN context must be refused")
            .to_string();
        assert!(
            message.contains("NaN or infinite"),
            "the backstop must report the non-finite defect: {message}"
        );
        // Named as the context stage, not the embedding stage — proof it was the post-stack screen.
        assert!(
            message.contains("te (t2va):"),
            "must be the post-stack screen, not the embedding one: {message}"
        );
    }

    /// A `t2va` forward whose context comes out all-zero must return the typed refusal, not the
    /// tensor. The degenerate output is *produced*, not faked: the fixture zeroes only the token
    /// table and lets the real 2-layer stack run.
    ///
    /// On `t2va` this is satisfied by the **embedding** screen, which fires first — the post-stack
    /// backstop has its own arm above.
    #[test]
    fn forward_refuses_an_all_zero_context() {
        let cfg = tiny_config();
        let te = MiniMaxH3TextEncoder::from_weights(&tiny_weights(&cfg, false), TEST_PREFIX, &cfg)
            .expect("build tiny encoder");
        let (ids, mask) = tiny_prompt(&cfg);

        let message = te
            .forward(&ids, &mask)
            .expect_err("an all-zero context must be refused, never returned")
            .to_string();
        assert!(
            message.contains("t2va"),
            "the route must be named: {message}"
        );
        assert!(
            message.contains("degenerate conditioning tensor"),
            "{message}"
        );
        assert!(message.contains("sc-17153"), "{message}");
    }

    /// The refusal must be **specific**. The same fixture with a live token table has to pass, or
    /// the arm above would be green for a guard that rejects every forward.
    #[test]
    fn forward_returns_a_healthy_context_untouched() {
        let cfg = tiny_config();
        let te = MiniMaxH3TextEncoder::from_weights(&tiny_weights(&cfg, true), TEST_PREFIX, &cfg)
            .expect("build tiny encoder");
        let (ids, mask) = tiny_prompt(&cfg);

        let context = te
            .forward(&ids, &mask)
            .expect("a live encoder must forward");
        assert_eq!(context.shape(), &[1, 6, cfg.hidden_size]);
        assert!(
            crate::text_encoder::degeneracy::inspect_conditioning("t2va", &context)
                .unwrap()
                .is_none()
        );
    }

    /// The propagation claim the sc-17153 investigation rests on, stated executably: zeroing **only**
    /// `embed_tokens` yields a context that is *exactly* zero, through a stack whose 22 other tensors
    /// are all live. Residual adds, RMSNorm, SDPA and SwiGLU each map 0 to 0, so an embedding-lookup
    /// zero is indistinguishable at the output from a whole-stack zero — which is why the observed
    /// `max|out| = 0.0` does not by itself locate the fault at the end of the stack.
    ///
    /// If this ever stops holding, the "a zeroed token table explains the signature" line of the
    /// investigation is dead and the bisect has to start again from the intermediates.
    #[test]
    fn a_zeroed_token_table_produces_an_exactly_zero_context() {
        let cfg = tiny_config();
        let w = tiny_weights(&cfg, false);
        let te = MiniMaxH3TextEncoder::from_weights(&w, TEST_PREFIX, &cfg).unwrap();
        let (ids, mask) = tiny_prompt(&cfg);

        // Walk the stack directly, bypassing the screen, to observe what it screens.
        let hidden = te.embed_tokens.forward(&ids).unwrap();
        let (cos, sin) = te.rope.forward(6).unwrap();
        let m = build_mask(&mask, 1, 6).unwrap();
        let context = te
            .walk_layers(hidden, &cos, &sin, &m, &mut |_, h| Ok(h))
            .unwrap();

        let peak = context
            .abs()
            .unwrap()
            .max(None)
            .unwrap()
            .as_dtype(Dtype::Float32)
            .unwrap()
            .item::<f32>();
        assert_eq!(
            peak, 0.0,
            "a zeroed token table must walk the stack unchanged"
        );
    }

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
