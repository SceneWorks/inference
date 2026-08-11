//! The H3 condition-encoder forward: token embedding → causal Qwen3 decoder layers → the hidden
//! state at [`SELECT_HIDDEN`](super::SELECT_HIDDEN), with the leading chat-template tokens dropped.
//! That `[B, L - prefix, 5120]` tensor is the `context` the H3 DiT consumes.
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
    layers: Vec<Qwen3DecoderLayer>,
    rope: TextRope,
    /// 0-indexed decoder layer whose output is the context (`select_hidden - 1`).
    out_layer: usize,
    prefix_tokens: i32,
    image_token_id: i32,
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
            rope: TextRope::new(cfg.head_dim, cfg.rope_theta),
            out_layer,
            prefix_tokens: cfg.prefix_tokens as i32,
            image_token_id: cfg.image_token_id,
            mrope_section: cfg.mrope_section,
            head_dim: cfg.head_dim,
            rope_theta: cfg.rope_theta,
        })
    }

    /// How many decoder layers were actually loaded — `select_hidden`, not `num_layers`. Exposed so
    /// a caller (and the real-weight smoke) can prove the trim is real.
    pub fn num_loaded_layers(&self) -> usize {
        self.layers.len()
    }

    /// Quantize the token table + every loaded projection in place. Norms stay dense.
    pub fn quantize(&mut self, bits: i32) -> Result<()> {
        self.embed_tokens.quantize(bits, true)?;
        for layer in &mut self.layers {
            layer.quantize(bits)?;
        }
        Ok(())
    }

    /// Text-only (t2va) conditioning. `input_ids` / `attention_mask`: `[b, s]` int32. Returns the
    /// DiT context `[b, s - prefix_tokens, hidden]`.
    ///
    /// Uses plain 1-D RoPE: with no vision tokens Qwen3-VL's interleaved MRoPE sections all index
    /// the same sequential position, so it reduces exactly to standard RoPE.
    pub fn forward(&self, input_ids: &Array, attention_mask: &Array) -> Result<Array> {
        let sh = input_ids.shape();
        let (b, s) = (sh[0], sh[1]);
        let (cos, sin) = self.rope.forward(s)?;
        let mask = build_mask(attention_mask, b, s)?;

        let mut hidden = self.embed_tokens.forward(input_ids)?;
        for layer in &self.layers {
            hidden = layer.forward(&hidden, &cos, &sin, &mask)?;
        }
        self.trim_prefix(hidden)
    }

    /// Drop the leading template-prefix tokens. Needs strictly more tokens than the prefix; a
    /// shorter sequence would otherwise build an empty index and hit an opaque `split_axis` panic.
    fn trim_prefix(&self, hidden: Array) -> Result<Array> {
        let n = hidden.shape()[1];
        if n <= self.prefix_tokens {
            return Err(Error::Msg(format!(
                "minimax-h3 text encoder: prompt has {n} token(s), must exceed the {} dropped \
                 template-prefix tokens",
                self.prefix_tokens
            )));
        }
        // Contiguous split rather than an arange-gather: MLX's `as_slice` ignores strides, so a
        // gather round-trip is both slower and stride-fragile.
        Ok(hidden.split_axis(&[self.prefix_tokens], 1)?.swap_remove(1))
    }

    /// **Vision-grounded** conditioning (the fl2va / Ref2VA image path): run the encoder with each
    /// reference's tower features spliced over its `<|image_pad|>` block and 3-D interleaved MRoPE
    /// positions, so the LM "sees" the references while reading the prompt.
    ///
    /// Mirrors [`forward`](Self::forward) but (a) replaces the `<|image_pad|>` embeddings with the
    /// tower's merged `image_embeds` `[nⱼ, hidden]`, (b) additively injects each reference's
    /// `deepstack` feature at those positions for the first `deepstack.len()` layers, and (c) uses
    /// interleaved MRoPE — the image block carries its 2-D merged grid position, text stays
    /// sequential. Returns the same `[b, s - prefix_tokens, hidden]` context. `b = 1`.
    pub fn forward_with_images(
        &self,
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

        let runs = image_token_runs(&ids, self.image_token_id, s);
        if runs.is_empty() {
            return Err(Error::Msg(
                "minimax-h3 te (grounded): prompt has no <|image_pad|> tokens".into(),
            ));
        }
        if runs.len() != image_embeds.len() || runs.len() != grids.len() {
            return Err(Error::Msg(format!(
                "minimax-h3 te (grounded): {} <|image_pad|> run(s) but {} embeds / {} grids",
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

        let (pt, ph, pw) = mrope_positions_multi(&ids, self.image_token_id, grids);
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

        for (i, layer) in self.layers.iter().enumerate() {
            hidden = layer.forward(&hidden, &cos, &sin, &mask)?;
            for (&(start, end), ds_img) in runs.iter().zip(deepstack) {
                if i < ds_img.len() {
                    let mid = slice_seq(&hidden, start, end)?;
                    let inj = add(&mid, &ds_img[i].expand_dims(0)?.as_dtype(dt)?)?;
                    hidden = replace_seq(&hidden, &inj, start, end)?;
                }
            }
        }
        let _ = self.out_layer; // the loop runs exactly `out_layer + 1` layers by construction
        self.trim_prefix(hidden)
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
fn image_token_runs(ids: &[i32], image_token_id: i32, s: i32) -> Vec<(i32, i32)> {
    let mut runs = Vec::new();
    let mut i = 0i32;
    while i < s {
        if ids[i as usize] == image_token_id {
            let start = i;
            while i < s && ids[i as usize] == image_token_id {
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
    image_token_id: i32,
    grids: &[[i32; 3]],
) -> (Vec<i32>, Vec<i32>, Vec<i32>) {
    let (mut pt, mut ph, mut pw) = (Vec::new(), Vec::new(), Vec::new());
    let mut cur = 0i32;
    let mut img_i = 0usize;
    let mut i = 0usize;
    while i < ids.len() {
        if ids[i] == image_token_id && img_i < grids.len() {
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
        let (pt, ph, pw) = mrope_positions_multi(&[10, 11, 12, 13], 151655, &[]);
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
        let (pt, ph, pw) = mrope_positions_multi(&ids, img, &[[1, 4, 4]]);
        assert_eq!(pt, vec![0, 1, 2, 2, 2, 2, 4]);
        assert_eq!(ph, vec![0, 1, 2, 2, 3, 3, 4]);
        assert_eq!(pw, vec![0, 1, 2, 3, 2, 3, 4]);
    }

    /// Two references each get their own grid and their own cursor advance.
    #[test]
    fn mrope_positions_handles_two_references() {
        let img = 151655;
        let ids = vec![1, img, img, img, img, 2, img, 3];
        let (pt, _, _) = mrope_positions_multi(&ids, img, &[[1, 4, 4], [1, 2, 2]]);
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
            image_token_runs(&ids, img, ids.len() as i32),
            vec![(1, 3), (4, 5)]
        );
        assert!(image_token_runs(&[1, 2, 3], img, 3).is_empty());
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
