//! The Qwen3-VL-4B language model itself: token embedding → 36 causal decoder layers → the final
//! RMSNorm. Port of `Qwen3VLTextModel` as re-bound by the reference's `qwen3_patch_forward`
//! (`_vendor/mage_flow/models/modules/text_encoder.py:204-295`, `:368`).
//!
//! ## The output IS the final post-norm hidden state
//!
//! `model_forward` ends with `hidden_states = self.norm(hidden_states)` (`:290`) and returns it as
//! `last_hidden_state`; `CustomQwen3VLForConditionalGeneration` runs permanently in
//! `output_mode="embedding"` (`:83-84`) and hands `outputs[0]` straight back (`:156`, `:172-178`).
//! `output_hidden_states: False` (`:521`) means no intermediate layer is even materialised on this
//! path. So [`Qwen3VlTextEncoder::forward_embeds`](Qwen3VlTextEncoder::forward_embeds) applies **every** layer and then
//! the final norm — the `mlx-gen-z-image` sibling's penultimate-layer convention is wrong here by
//! measured max_abs 10433.29 (penultimate) / 4225.29 (final, pre-norm), against 0.0 bit-exact for
//! the post-norm final state (`_vendor/MAGE_FLOW_GAPS.md` GAP 1).
//!
//! ## Packing (`cu_seqlens`) — and why a per-segment loop *is* the port
//!
//! The reference encodes several prompts in one varlen forward: `input_ids` is the flat
//! concatenation `[Total_L]`, `cu_seqlens` `[B+1]` carries the boundaries, **position ids restart
//! at 0 for every segment** (`:501-504`), and attention is `causal=True` with per-segment
//! `cu_seqlens` isolation (`:344-356`). Segments therefore cannot see each other, and the
//! reference states as much (`:477-478`).
//!
//! That is not merely a claim we are trusting. The committed goldens were dumped on this Mac with
//! `attn: "sdpa"` (golden metadata), and the vendored SDPA backend implements varlen by
//! **dispatching one `F.scaled_dot_product_attention` per sequence and concatenating**
//! (`_vendor/mage_flow/models/modules/_attn_backend.py`, `_resolve_sdpa`). The oracle this port is
//! measured against was produced by exactly the loop [`Qwen3VlTextEncoder::forward_packed`] runs, so the structure
//! matches rather than merely commutes — and the golden's `neg_txt` (the *second* segment) is the
//! tensor that proves it, since a leaking implementation would corrupt it while leaving `gen_txt`
//! intact.
//!
//! Running per segment also bounds attention memory by `max(Lᵢ)²` instead of `(ΣLᵢ)²`, which at the
//! 2082-token cap ([`max_prompt_tokens`](crate::config::max_prompt_tokens)) is a 4× difference for
//! the ordinary positive+negative pair.
//!
//! ## Seam for the vision/edit path (sc-14048)
//!
//! Editing keeps this LM unchanged and adds the Qwen3-VL vision tower around it. The three pieces
//! it needs are already public and do not require restructuring:
//!
//! 1. [`embed`](Qwen3VlTextEncoder::embed) returns the token embeddings **before** any layer runs,
//!    so the merged image features can be spliced over the `<|image_pad|>` run;
//! 2. [`layers`](Qwen3VlTextEncoder::layers) plus [`Qwen3VlDecoderLayer::forward`] let that path
//!    drive the stack itself and inject the Qwen3-VL **deepstack** features — the one thing
//!    [`Qwen3VlTextEncoder::forward_embeds`] cannot express. Inject into **LM layers
//!    `0..deepstack.len()`** (i.e. 0/1/2), **additively**, and **only over the `<|image_pad|>`
//!    token run**;
//! 3. [`final_norm`](Qwen3VlTextEncoder::final_norm) closes it, and
//!    [`MRopePositions`] already carries three independent axes.
//!
//! **`deepstack_visual_indexes` (`[5, 11, 17]`) is NOT an LM layer list — do not inject there.**
//! Those are the **vision-tower block indices the features are EXTRACTED from**, read inside
//! `Qwen3VLVisionModel` (`transformers/models/qwen3_vl/modeling_qwen3_vl.py:590`), and the field
//! lives in `vision_config`, not `text_config`. The LM side consumes
//! `deepstack_visual_embeds[layer_idx] for layer_idx in range(len(deepstack_visual_embeds))`
//! (`:862-866` — identically in the vendored patch at
//! `_vendor/mage_flow/models/modules/text_encoder.py:282-288`) and applies it via
//! `_deepstack_process` (`:876-882`), which adds **only at `visual_pos_masks`**. The class
//! docstring is explicit: DeepStack "integrates visual features into the **early** hidden states"
//! (`:759`). Injecting at 5/11/17, or over all tokens instead of the image run, produces silently
//! wrong edit conditioning — no shape error, no missing key, and there is no edit-path golden until
//! e2e. `mlx-gen-boogu/src/text_encoder/encoder.rs:9,134` and
//! `mlx-gen-krea/src/text_encoder/encoder.rs:155-156,248-249` already do it correctly and are the
//! executable references.
//!
//! (Note that the reference's *own* edit path still feeds flat per-segment positions — see the
//! [`super::rope`] module docs — so the 3-axis freedom is available but not currently exercised.)

use mlx_rs::ops::{add, concatenate_axis};
use mlx_rs::Array;

use mlx_gen::nn::{build_mask, TokenEmbedding};
use mlx_gen::weights::Weights;
use mlx_gen::{Error, Result};

use crate::config::QwenVlTextConfig;

use super::rope::{mrope_cos_sin, MRopePositions};
use super::{embedding, join, Qwen3VlDecoderLayer};

/// Qwen3-VL-4B, text path.
pub struct Qwen3VlTextEncoder {
    embed_tokens: TokenEmbedding,
    layers: Vec<Qwen3VlDecoderLayer>,
    /// The **final** RMSNorm scale (`{prefix}.norm.weight`). Loading it is the whole GAP-1
    /// correction: the z-image sibling deliberately does not load its fork's equivalent because it
    /// returns a penultimate, un-normed state.
    norm: Array,
    eps: f32,
    head_dim: i32,
    rope_theta: f64,
    mrope_section: [i32; 3],
    block_stream: Option<TextBlockStream>,
}

#[derive(Clone)]
struct TextBlockStream {
    source: mlx_gen::WeightsSource,
    prefix: String,
    cfg: QwenVlTextConfig,
    eps: f32,
    layer_quant_bits: Option<i32>,
}

impl TextBlockStream {
    fn open(&self) -> Result<Weights> {
        match &self.source {
            mlx_gen::WeightsSource::Dir(dir) => Weights::from_dir(dir),
            mlx_gen::WeightsSource::File(file) => Weights::from_file(file),
        }
    }

    fn materialize(&self, view: &mut Weights, index: usize) -> Result<Qwen3VlDecoderLayer> {
        let mut layer = Qwen3VlDecoderLayer::from_weights(
            view,
            &join(&self.prefix, &format!("layers.{index}")),
            self.cfg.num_attention_heads,
            self.cfg.num_key_value_heads,
            self.cfg.head_dim,
            self.eps,
        )?;
        view.remove_accessed();
        if let Some(bits) = self.layer_quant_bits {
            layer.quantize(bits)?;
        }
        Ok(layer)
    }
}

impl Qwen3VlTextEncoder {
    /// Quantize token embeddings and every attention/MLP projection; RMSNorms stay dense.
    ///
    /// The 36 decoder layers are held at their 8-bit floor (sc-15071) — a uniformly-Q4 text encoder
    /// is the second half of the defect that made the Q4 tier render a tiled texture instead of the
    /// prompt, and the SwiGLU MLP is the specific offender. The token embedding takes the requested
    /// width. [`crate::convert::quant_floor_bits`] documents both floors and their measurements, and
    /// is the same seam the offline converter calls.
    pub fn quantize(&mut self, bits: i32) -> Result<()> {
        self.embed_tokens.quantize(bits, true)?;
        let layer_bits = crate::quant::floor_bits(crate::quant::LM_LAYER_PREFIX, bits);
        for layer in &mut self.layers {
            layer.quantize(layer_bits)?;
        }
        let got = self.quantized_linear_count();
        let expected = 1 + self.layers.len() * 7;
        if got != expected {
            return Err(mlx_gen::Error::Msg(format!(
                "mage_flow: text encoder quantization packed {got}/{expected} required projections"
            )));
        }
        Ok(())
    }

    pub(crate) fn quantized_linear_count(&self) -> usize {
        usize::from(self.embed_tokens.is_quantized())
            + self
                .layers
                .iter()
                .map(Qwen3VlDecoderLayer::quantized_linear_count)
                .sum::<usize>()
    }

    /// Load the LM under `prefix` — `"model.language_model"` for the published
    /// `text_encoder/` checkpoint (`{prefix}.embed_tokens.weight`, `{prefix}.layers.{i}.…`,
    /// `{prefix}.norm.weight`).
    ///
    /// Every one of `cfg.num_layers` layers is loaded and run: unlike the select-layer encoders in
    /// this workspace there is no "later layers cannot matter" shortcut, because the conditioning
    /// is the last layer's output.
    pub fn from_weights(
        w: &Weights,
        prefix: &str,
        cfg: &QwenVlTextConfig,
        eps: f32,
        rope_theta: f64,
    ) -> Result<Self> {
        let mut layers = Vec::with_capacity(cfg.num_layers);
        for i in 0..cfg.num_layers {
            layers.push(Qwen3VlDecoderLayer::from_weights(
                w,
                &join(prefix, &format!("layers.{i}")),
                cfg.num_attention_heads,
                cfg.num_key_value_heads,
                cfg.head_dim,
                eps,
            )?);
        }
        Ok(Self {
            embed_tokens: embedding(w, &join(prefix, "embed_tokens"))?,
            layers,
            norm: w.require(&join(prefix, "norm.weight"))?.clone(),
            eps,
            head_dim: cfg.head_dim,
            rope_theta,
            mrope_section: cfg.mrope_section,
            block_stream: None,
        })
    }

    /// Consuming production variant of [`from_weights`](Self::from_weights). Each completed
    /// decoder layer is removed from the source map immediately, bounding the load/quantize
    /// transient instead of retaining all 36 source-layer handles to the end.
    pub fn from_weights_draining(
        w: &mut Weights,
        prefix: &str,
        cfg: &QwenVlTextConfig,
        eps: f32,
        rope_theta: f64,
    ) -> Result<Self> {
        let embed_prefix = join(prefix, "embed_tokens");
        let embed_tokens = embedding(w, &embed_prefix)?;
        w.remove_prefix(&format!("{embed_prefix}."));
        let mut layers = Vec::with_capacity(cfg.num_layers);
        for i in 0..cfg.num_layers {
            let layer_prefix = join(prefix, &format!("layers.{i}"));
            layers.push(Qwen3VlDecoderLayer::from_weights(
                w,
                &layer_prefix,
                cfg.num_attention_heads,
                cfg.num_key_value_heads,
                cfg.head_dim,
                eps,
            )?);
            w.remove_prefix(&format!("{layer_prefix}."));
        }
        let norm_key = join(prefix, "norm.weight");
        let norm = w.require(&norm_key)?.clone();
        w.remove(&norm_key);
        Ok(Self {
            embed_tokens,
            layers,
            norm,
            eps,
            head_dim: cfg.head_dim,
            rope_theta,
            mrope_section: cfg.mrope_section,
            block_stream: None,
        })
    }

    pub(crate) fn arm_block_stream(
        &mut self,
        source: mlx_gen::WeightsSource,
        prefix: &str,
        cfg: QwenVlTextConfig,
        layer_quant_bits: Option<i32>,
    ) -> Result<()> {
        if self.layers.len() != cfg.num_layers {
            return Err(Error::Msg(format!(
                "mage_flow text encoder: cannot arm the {}-layer stream from {} resident layers",
                cfg.num_layers,
                self.layers.len()
            )));
        }
        self.block_stream = Some(TextBlockStream {
            source,
            prefix: prefix.to_owned(),
            cfg,
            eps: self.eps,
            layer_quant_bits,
        });
        self.layers.clear();
        if !self.layers.is_empty() {
            return Err(Error::Msg(
                "mage_flow text encoder: deferred stream retained resident layers".into(),
            ));
        }
        Ok(())
    }

    pub(crate) fn resident_layer_count(&self) -> usize {
        self.layers.len()
    }

    /// The 36 decoder blocks, in order. Public for the deepstack-injecting edit path (sc-14048).
    pub fn layers(&self) -> &[Qwen3VlDecoderLayer] {
        &self.layers
    }

    /// Token ids `[b, s]` (int32) → embeddings `[b, s, hidden]` (f32), before any layer runs.
    pub fn embed(&self, ids: &Array) -> Result<Array> {
        self.embed_tokens.forward(ids)
    }

    /// The model's final RMSNorm — the last operation before the conditioning is returned.
    pub fn final_norm(&self, hidden: &Array) -> Result<Array> {
        Ok(mlx_rs::fast::rms_norm(hidden, &self.norm, self.eps)?)
    }

    /// Run every decoder layer over `hidden` `[1, s, hidden]` under the given M-RoPE positions,
    /// then the final norm. Returns `[1, s, hidden]`.
    ///
    /// Attention is causal over the whole `s`, so this is **one** packed segment — callers with
    /// several prompts go through [`Qwen3VlTextEncoder::forward_packed`](Self::forward_packed).
    pub fn forward_embeds(&self, hidden: &Array, pos: &MRopePositions) -> Result<Array> {
        self.forward_embeds_with_cancel(hidden, pos, None)
    }

    pub(crate) fn forward_embeds_with_cancel(
        &self,
        hidden: &Array,
        pos: &MRopePositions,
        cancel: Option<&mlx_gen::CancelFlag>,
    ) -> Result<Array> {
        let sh = hidden.shape();
        let (b, s) = (sh[0], sh[1]);
        if b != 1 {
            return Err(Error::Msg(format!(
                "mage_flow text encoder: expected a single packed row, got batch {b}"
            )));
        }
        if pos.len() != s as usize {
            return Err(Error::Msg(format!(
                "mage_flow text encoder: {} M-RoPE positions for {s} token(s)",
                pos.len()
            )));
        }

        let (cos, sin) = mrope_cos_sin(
            pos,
            self.head_dim,
            self.rope_theta,
            self.mrope_section,
            hidden.dtype(),
        )?;
        // All-real tokens: the reference never pads a packed segment, so the mask is purely causal.
        let ones = Array::from_slice(&vec![1i32; s as usize], &[1, s]);
        let mask = build_mask(&ones, 1, s)?;

        let h = self.run_layers(hidden.clone(), &cos, &sin, &mask, cancel, &mut |h, _| Ok(h))?;
        self.final_norm(&h)
    }

    fn run_layers(
        &self,
        mut hidden: Array,
        cos: &Array,
        sin: &Array,
        mask: &Array,
        cancel: Option<&mlx_gen::CancelFlag>,
        after: &mut dyn FnMut(Array, usize) -> Result<Array>,
    ) -> Result<Array> {
        let Some(stream) = &self.block_stream else {
            for (index, layer) in self.layers.iter().enumerate() {
                hidden = after(layer.forward(&hidden, cos, sin, mask)?, index)?;
            }
            return Ok(hidden);
        };
        if !self.layers.is_empty() {
            return Err(Error::Msg(
                "mage_flow text encoder: deferred stream still owns resident layers".into(),
            ));
        }
        let fallback = mlx_gen::CancelFlag::default();
        let cancel = cancel.unwrap_or(&fallback);
        let plan = mlx_gen::block_residency::BlockPlan::new(stream.cfg.num_layers, 1)?;
        mlx_gen::block_residency::run_windowed(
            &plan,
            cancel,
            hidden,
            || stream.open(),
            |mut state, view, range| {
                for index in range {
                    let layer = stream.materialize(view, index)?;
                    state = after(layer.forward(&state, cos, sin, mask)?, index)?;
                }
                Ok(state)
            },
            |state: &Array| Ok(mlx_rs::transforms::eval([state])?),
        )
    }

    /// Run the Qwen3-VL language stack with merged vision features spliced at each image-token run.
    /// Deepstack features extracted by vision blocks 5/11/17 are injected into LM layers 0/1/2,
    /// additively and only at the corresponding visual positions.
    pub fn forward_grounded(
        &self,
        ids: &[i32],
        image_token_id: i32,
        image_embeds: &[Array],
        deepstack: &[Vec<Array>],
        grids: &[[i32; 3]],
    ) -> Result<Array> {
        Ok(self
            .forward_grounded_trace(ids, image_token_id, image_embeds, deepstack, grids)?
            .0)
    }

    /// Grounded forward plus the pre-deepstack outputs of LM layers 0/1/2.
    pub fn forward_grounded_trace(
        &self,
        ids: &[i32],
        image_token_id: i32,
        image_embeds: &[Array],
        deepstack: &[Vec<Array>],
        grids: &[[i32; 3]],
    ) -> Result<(Array, Vec<Array>)> {
        self.forward_grounded_trace_with_cancel(
            ids,
            image_token_id,
            image_embeds,
            deepstack,
            grids,
            None,
        )
    }

    pub(crate) fn forward_grounded_trace_with_cancel(
        &self,
        ids: &[i32],
        image_token_id: i32,
        image_embeds: &[Array],
        deepstack: &[Vec<Array>],
        grids: &[[i32; 3]],
        cancel: Option<&mlx_gen::CancelFlag>,
    ) -> Result<(Array, Vec<Array>)> {
        if image_embeds.is_empty()
            || image_embeds.len() != deepstack.len()
            || image_embeds.len() != grids.len()
        {
            return Err(Error::Msg(format!(
                "mage_flow grounded TE: need matching non-empty embeds/deepstack/grids, got {}/{}/{}",
                image_embeds.len(),
                deepstack.len(),
                grids.len()
            )));
        }
        let s = ids.len() as i32;
        let runs = image_token_runs(ids, image_token_id);
        if runs.len() != image_embeds.len() {
            return Err(Error::Msg(format!(
                "mage_flow grounded TE: {} image-token run(s) but {} vision embedding(s)",
                runs.len(),
                image_embeds.len()
            )));
        }
        validate_deepstack(&runs, image_embeds, deepstack)?;
        let ids_arr = Array::from_slice(ids, &[1, s]);
        let mut hidden = self.embed(&ids_arr)?;
        let dtype = hidden.dtype();
        for (&(start, end), embeds) in runs.iter().zip(image_embeds) {
            if embeds.shape()[0] != end - start {
                return Err(Error::Msg(format!(
                    "mage_flow grounded TE: image run has {} token(s) but vision produced {}",
                    end - start,
                    embeds.shape()[0]
                )));
            }
            hidden = replace_seq(
                &hidden,
                &embeds.expand_dims(0)?.as_dtype(dtype)?,
                start,
                end,
            )?;
        }

        // Mage's packed wrapper explicitly passes flat per-segment `position_ids` even for image
        // inputs, so HF expands that one axis across T/H/W instead of invoking Qwen3-VL's native
        // grid position builder. Keep the distinct-axis builder executable below as the oracle for
        // that lower-level capability, but follow the frozen edit pipeline here.
        let positions = MRopePositions::text(ids.len());
        let (cos, sin) = mrope_cos_sin(
            &positions,
            self.head_dim,
            self.rope_theta,
            self.mrope_section,
            dtype,
        )?;
        let mask = build_mask(&Array::from_slice(&vec![1i32; ids.len()], &[1, s]), 1, s)?;
        let mut early = Vec::with_capacity(3);
        let hidden = self.run_layers(
            hidden,
            &cos,
            &sin,
            &mask,
            cancel,
            &mut |mut hidden, layer_index| {
                if layer_index < 3 {
                    early.push(hidden.clone());
                }
                for (&(start, end), features) in runs.iter().zip(deepstack) {
                    if layer_index < 3 {
                        let visual = slice_seq(&hidden, start, end)?;
                        let injected = add(
                            &visual,
                            &features[layer_index].expand_dims(0)?.as_dtype(dtype)?,
                        )?;
                        hidden = replace_seq(&hidden, &injected, start, end)?;
                    }
                }
                Ok(hidden)
            },
        )?;
        Ok((self.final_norm(&hidden)?, early))
    }

    /// Encode ONE sequence of token ids → `[s, hidden]`, the post-final-norm hidden state with no
    /// tokens dropped.
    pub fn forward_segment(&self, ids: &[i32]) -> Result<Array> {
        if ids.is_empty() {
            return Err(Error::Msg(
                "mage_flow text encoder: cannot encode an empty token sequence".into(),
            ));
        }
        let ids_arr = Array::from_slice(ids, &[1, ids.len() as i32]);
        let hidden = self.embed(&ids_arr)?;
        let out = self.forward_embeds(&hidden, &MRopePositions::text(ids.len()))?;
        Ok(out.reshape(&[ids.len() as i32, -1])?)
    }

    /// Encode a **packed** batch: `ids` is the flat concatenation of every segment and
    /// `cu_seqlens` `[B+1]` its cumulative boundaries (`cu_seqlens[0] == 0`,
    /// `cu_seqlens[B] == ids.len()`). Returns the full `[Total_L, hidden]` post-final-norm state,
    /// in packed order — nothing dropped yet.
    ///
    /// See the module docs for why this is a per-segment loop rather than a block-diagonal mask.
    pub fn forward_packed(&self, ids: &[i32], cu_seqlens: &[i32]) -> Result<Array> {
        self.forward_packed_with_cancel(ids, cu_seqlens, None)
    }

    pub(crate) fn forward_packed_with_cancel(
        &self,
        ids: &[i32],
        cu_seqlens: &[i32],
        cancel: Option<&mlx_gen::CancelFlag>,
    ) -> Result<Array> {
        let lens = seq_lens_from_cu(cu_seqlens, ids.len())?;
        let mut parts = Vec::with_capacity(lens.len());
        let mut start = 0usize;
        for len in lens {
            let segment = &ids[start..start + len];
            let ids_arr = Array::from_slice(segment, &[1, len as i32]);
            let hidden = self.embed(&ids_arr)?;
            let out =
                self.forward_embeds_with_cancel(&hidden, &MRopePositions::text(len), cancel)?;
            parts.push(out.reshape(&[len as i32, -1])?);
            start += len;
        }
        if parts.len() == 1 {
            return Ok(parts.pop().expect("checked non-empty"));
        }
        let refs: Vec<&Array> = parts.iter().collect();
        Ok(concatenate_axis(&refs, 0)?)
    }
}

const IMAGE_SPATIAL_MERGE: i32 = 2;

fn validate_deepstack(
    runs: &[(i32, i32)],
    image_embeds: &[Array],
    deepstack: &[Vec<Array>],
) -> Result<()> {
    for ((&(start, end), embeds), features) in runs.iter().zip(image_embeds).zip(deepstack) {
        let visual_tokens = end - start;
        if features.len() != 3 {
            return Err(Error::Msg(format!(
                "mage_flow grounded TE: each image requires exactly 3 deepstack features, got {}",
                features.len()
            )));
        }
        for (index, feature) in features.iter().enumerate() {
            if feature.shape() != [visual_tokens, embeds.shape()[1]] {
                return Err(Error::Msg(format!(
                    "mage_flow grounded TE: deepstack {index} must be [{visual_tokens}, {}], got {:?}",
                    embeds.shape()[1],
                    feature.shape()
                )));
            }
        }
    }
    Ok(())
}

fn slice_seq(x: &Array, start: i32, end: i32) -> Result<Array> {
    Ok(x.split_axis(&[start, end], 1)?.swap_remove(1))
}

fn replace_seq(x: &Array, replacement: &Array, start: i32, end: i32) -> Result<Array> {
    let mut parts = x.split_axis(&[start, end], 1)?;
    let after = parts.swap_remove(2);
    let before = parts.swap_remove(0);
    Ok(concatenate_axis(&[&before, replacement, &after], 1)?)
}

fn image_token_runs(ids: &[i32], image_token_id: i32) -> Vec<(i32, i32)> {
    let mut runs = Vec::new();
    let mut index = 0usize;
    while index < ids.len() {
        if ids[index] == image_token_id {
            let start = index;
            while index < ids.len() && ids[index] == image_token_id {
                index += 1;
            }
            runs.push((start as i32, index as i32));
        } else {
            index += 1;
        }
    }
    runs
}

pub(crate) fn grounded_mrope_positions(
    ids: &[i32],
    image_token_id: i32,
    grids: &[[i32; 3]],
) -> Result<MRopePositions> {
    let runs = image_token_runs(ids, image_token_id);
    if runs.len() != grids.len() {
        return Err(Error::Msg(format!(
            "mage_flow grounded TE: {} image-token run(s) but {} image grid(s)",
            runs.len(),
            grids.len()
        )));
    }
    let (mut t, mut h, mut w) = (
        Vec::with_capacity(ids.len()),
        Vec::with_capacity(ids.len()),
        Vec::with_capacity(ids.len()),
    );
    let mut cursor = 0i32;
    let mut image_index = 0usize;
    let mut index = 0usize;
    while index < ids.len() {
        if ids[index] == image_token_id {
            let grid = grids[image_index];
            let (rows, cols) = (grid[1] / IMAGE_SPATIAL_MERGE, grid[2] / IMAGE_SPATIAL_MERGE);
            let tokens = rows * cols * grid[0];
            let run = runs[image_index];
            if run.1 - run.0 != tokens {
                return Err(Error::Msg(format!(
                    "mage_flow grounded TE: image run has {} token(s), grid {:?} requires {tokens}",
                    run.1 - run.0,
                    grid
                )));
            }
            for frame in 0..grid[0] {
                for row in 0..rows {
                    for col in 0..cols {
                        t.push(cursor + frame);
                        h.push(cursor + row);
                        w.push(cursor + col);
                    }
                }
            }
            cursor += grid[0].max(rows.max(cols));
            index += tokens as usize;
            image_index += 1;
        } else {
            t.push(cursor);
            h.push(cursor);
            w.push(cursor);
            cursor += 1;
            index += 1;
        }
    }
    MRopePositions::from_axes(t, h, w)
}

/// Validate `cu_seqlens` against the packed token count and return the per-segment lengths.
///
/// Rejects a malformed pack rather than silently producing a shorter conditioning: a non-zero
/// first entry, a non-monotonic step, a zero-length segment, or a final entry that disagrees with
/// `total` would each yield plausible-looking but wrong `txt`.
pub(crate) fn seq_lens_from_cu(cu_seqlens: &[i32], total: usize) -> Result<Vec<usize>> {
    if cu_seqlens.len() < 2 {
        return Err(Error::Msg(format!(
            "mage_flow text encoder: cu_seqlens needs at least 2 entries, got {}",
            cu_seqlens.len()
        )));
    }
    if cu_seqlens[0] != 0 {
        return Err(Error::Msg(format!(
            "mage_flow text encoder: cu_seqlens must start at 0, got {}",
            cu_seqlens[0]
        )));
    }
    if *cu_seqlens.last().expect("checked non-empty") as usize != total {
        return Err(Error::Msg(format!(
            "mage_flow text encoder: cu_seqlens ends at {} but {total} token(s) were packed",
            cu_seqlens.last().expect("checked non-empty")
        )));
    }
    cu_seqlens
        .windows(2)
        .map(|pair| {
            let len = pair[1] - pair[0];
            if len <= 0 {
                return Err(Error::Msg(format!(
                    "mage_flow text encoder: cu_seqlens segment {}..{} is not positive",
                    pair[0], pair[1]
                )));
            }
            Ok(len as usize)
        })
        .collect()
}

/// Cumulative-sequence-length vector `[0, L₀, L₀+L₁, …]` for a list of segment lengths — the
/// reference's `_lens_to_cu` (`pipeline.py`).
pub fn cu_seqlens_from_lens(lens: &[usize]) -> Vec<i32> {
    let mut cu = Vec::with_capacity(lens.len() + 1);
    let mut acc = 0i32;
    cu.push(acc);
    for &len in lens {
        acc += len as i32;
        cu.push(acc);
    }
    cu
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn cu_seqlens_round_trip() {
        let lens = [54usize, 40];
        let cu = cu_seqlens_from_lens(&lens);
        assert_eq!(cu, vec![0, 54, 94]);
        assert_eq!(seq_lens_from_cu(&cu, 94).unwrap(), vec![54, 40]);
    }

    /// A malformed pack must be an error, not a silently shorter conditioning.
    #[test]
    fn malformed_cu_seqlens_are_rejected() {
        assert!(seq_lens_from_cu(&[0], 0).is_err(), "needs >= 2 entries");
        assert!(seq_lens_from_cu(&[1, 5], 5).is_err(), "must start at 0");
        assert!(seq_lens_from_cu(&[0, 5], 6).is_err(), "must end at total");
        assert!(
            seq_lens_from_cu(&[0, 3, 3], 3).is_err(),
            "zero-length segment"
        );
        assert!(
            seq_lens_from_cu(&[0, 5, 3], 3).is_err(),
            "non-monotonic segment"
        );
    }

    #[test]
    fn grounded_positions_use_distinct_t_h_w_axes_for_every_image() {
        let image = 151_655;
        let ids = [7, image, image, image, image, 8, image, image, 9];
        let positions = grounded_mrope_positions(&ids, image, &[[1, 4, 4], [1, 2, 4]]).unwrap();
        assert_eq!(positions.len(), ids.len());
        let (t, h, w) = positions.axes();
        // First image starts at cursor 1 and forms a 2x2 merged grid.
        assert_eq!(&t[1..5], &[1, 1, 1, 1]);
        assert_eq!(&h[1..5], &[1, 1, 2, 2]);
        assert_eq!(&w[1..5], &[1, 2, 1, 2]);
        // The second image has a distinct 1x2 grid and advances from the intervening text token.
        assert_eq!(&t[6..8], &[4, 4]);
        assert_eq!(&h[6..8], &[4, 4]);
        assert_eq!(&w[6..8], &[4, 5]);
        assert_ne!(t, h);
        assert_ne!(h, w);
        assert!(
            grounded_mrope_positions(&ids, image, &[[1, 4, 4]]).is_err(),
            "dropping a grid must not silently flatten the second image"
        );
        assert!(
            grounded_mrope_positions(&ids, image, &[[1, 2, 4], [1, 2, 4]]).is_err(),
            "a grid whose merged-token count disagrees with its placeholder run must fail"
        );
    }

    #[test]
    fn grounded_deepstack_requires_all_three_shape_exact_features() {
        let a = || Array::from_slice(&[0f32; 32], &[4, 8]);
        let embeds = vec![a()];
        let good = vec![vec![a(), a(), a()]];
        assert!(validate_deepstack(&[(1, 5)], &embeds, &good).is_ok());
        assert!(validate_deepstack(&[(1, 5)], &embeds, &[good[0][..2].to_vec()]).is_err());
        let wrong = vec![vec![a(), Array::from_slice(&[0f32; 24], &[3, 8]), a()]];
        assert!(validate_deepstack(&[(1, 5)], &embeds, &wrong).is_err());
    }
}
