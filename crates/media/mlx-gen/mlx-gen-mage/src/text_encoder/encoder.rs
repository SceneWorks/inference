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
//!    drive the stack itself and additively inject the deepstack features at layers
//!    `deepstack_visual_indexes` — the one thing [`Qwen3VlTextEncoder::forward_embeds`] cannot express;
//! 3. [`final_norm`](Qwen3VlTextEncoder::final_norm) closes it, and
//!    [`MRopePositions`] already carries three independent axes.
//!
//! (Note that the reference's *own* edit path still feeds flat per-segment positions — see the
//! [`super::rope`] module docs — so the 3-axis freedom is available but not currently exercised.)

use mlx_rs::ops::concatenate_axis;
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
}

impl Qwen3VlTextEncoder {
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
        })
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

        let mut h = hidden.clone();
        for layer in &self.layers {
            h = layer.forward(&h, &cos, &sin, &mask)?;
        }
        self.final_norm(&h)
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
        let lens = seq_lens_from_cu(cu_seqlens, ids.len())?;
        let mut parts = Vec::with_capacity(lens.len());
        let mut start = 0usize;
        for len in lens {
            parts.push(self.forward_segment(&ids[start..start + len])?);
            start += len;
        }
        if parts.len() == 1 {
            return Ok(parts.pop().expect("checked non-empty"));
        }
        let refs: Vec<&Array> = parts.iter().collect();
        Ok(concatenate_axis(&refs, 0)?)
    }
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
}
