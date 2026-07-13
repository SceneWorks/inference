//! Boogu Qwen3-VL condition encoder: token embedding → all `num_layers` causal decoder layers →
//! final RMSNorm → **last_hidden_state** `[B, L, 4096]` (the per-token instruction features the DiT
//! caption embedder consumes). Differs from the ideogram TE only in the head: Boogu applies the
//! final norm and returns a single layer, vs ideogram's 13-layer pre-final-norm interleave.
//!
//! Two forwards: [`BooguTextEncoder::last_hidden`] (text-only, plain 1-D RoPE) and
//! [`BooguTextEncoder::last_hidden_with_image`] (Edit / E7b-2 — splices the vision tower's image
//! embeds at the `<|image_pad|>` positions, switches to the 3-D **interleaved MRoPE**, and injects
//! the deepstack features into LM layers 0/1/2 at the image positions).

use mlx_rs::fast::rms_norm;
use mlx_rs::ops::{add, concatenate_axis};
use mlx_rs::{Array, Dtype};

use mlx_gen::array::host_i32;
use mlx_gen::nn::{build_mask, TextRope, TokenEmbedding};
use mlx_gen::weights::Weights;
use mlx_gen::Result;

use super::{join, BooguTextEncoderConfig, Qwen3DecoderLayer};

/// Qwen3-VL MRoPE section split (`text_config.rope_parameters.mrope_section`) — T/H/W frequency
/// counts over `head_dim/2 = 64`.
const MROPE_SECTION: [i32; 3] = [24, 20, 20];
/// Vision spatial merge (the LM sees one token per `merge²` patches).
const SPATIAL_MERGE: i32 = 2;

pub struct BooguTextEncoder {
    embed_tokens: TokenEmbedding,
    layers: Vec<Qwen3DecoderLayer>,
    rope: TextRope,
    final_norm: Array,
    eps: f32,
    head_dim: i32,
    rope_theta: f32,
}

impl BooguTextEncoder {
    /// Load from the `mllm` weights under `prefix` (`"model.language_model"`):
    /// `{prefix}.embed_tokens.weight`, `{prefix}.layers.{i}.…`, `{prefix}.norm.weight`.
    pub fn from_weights(w: &Weights, prefix: &str, cfg: &BooguTextEncoderConfig) -> Result<Self> {
        let mut layers = Vec::with_capacity(cfg.num_layers as usize);
        for i in 0..cfg.num_layers {
            layers.push(Qwen3DecoderLayer::from_weights(
                w,
                &join(prefix, &format!("layers.{i}")),
                cfg.num_heads,
                cfg.num_kv_heads,
                cfg.head_dim,
                cfg.rms_norm_eps,
            )?);
        }
        Ok(Self {
            embed_tokens: crate::quant::embedding(w, &join(prefix, "embed_tokens"))?,
            layers,
            rope: TextRope::new(cfg.head_dim, cfg.rope_theta),
            final_norm: w.require(&join(prefix, "norm.weight"))?.clone(),
            eps: cfg.rms_norm_eps,
            head_dim: cfg.head_dim,
            rope_theta: cfg.rope_theta,
        })
    }

    /// Quantize every decoder-layer projection in place (group-wise Q4/Q8 at [`crate::quant::GROUP_SIZE`]
    /// = 32). The **token embedding stays dense**: its only quantizer (`TokenEmbedding::quantize`)
    /// hardcodes group 64 in shared gen-core, which would clash with the group-32 Linears under the
    /// single-group-size packed loader — and the embedding is a precision-sensitive lookup table
    /// (~1.2 GB bf16), a standard dense-keep. The per-layer norms + final norm also stay dense.
    pub fn quantize(&mut self, bits: i32) -> Result<()> {
        for layer in &mut self.layers {
            layer.quantize(bits)?;
        }
        Ok(())
    }

    /// `input_ids` / `attention_mask`: `[b, s]` int32. Returns `last_hidden_state` `[b, s, 4096]`
    /// (f32) — all layers run, final norm applied.
    pub fn last_hidden(&self, input_ids: &Array, attention_mask: &Array) -> Result<Array> {
        let sh = input_ids.shape();
        let (b, s) = (sh[0], sh[1]);
        let (cos, sin) = self.rope.forward(s)?;
        let mask = build_mask(attention_mask, b, s)?;

        let mut hidden = self.embed_tokens.forward(input_ids)?;
        for layer in &self.layers {
            hidden = layer.forward(&hidden, &cos, &sin, &mask)?;
        }
        Ok(rms_norm(&hidden, &self.final_norm, self.eps)?)
    }

    /// Image-conditioned forward (Edit / E7b-2). Splices `image_embeds` (`[n, 4096]`, the vision
    /// tower's merged output) into the token embeddings at the `image_token_id` positions, runs the
    /// 36 decoder layers under the 3-D **interleaved MRoPE**, and injects the 3 `deepstack` features
    /// (`[n, 4096]` each) at the image positions after layers 0/1/2 — mirroring `Qwen3VLTextModel`.
    /// `grid_thw` is the image's patch grid `[t, h, w]`. `b = 1`.
    pub fn last_hidden_with_image(
        &self,
        input_ids: &Array,
        attention_mask: &Array,
        image_embeds: &Array,
        deepstack: &[Array],
        grid_thw: [i32; 3],
        image_token_id: i32,
    ) -> Result<Array> {
        let sh = input_ids.shape();
        let (b, s) = (sh[0], sh[1]);
        let ids = host_i32(input_ids)?;

        // Image-token block (contiguous, single reference).
        let img_idx: Vec<i32> = (0..s)
            .filter(|&i| ids[i as usize] == image_token_id)
            .collect();
        let img_start = *img_idx.first().ok_or_else(|| {
            mlx_gen::Error::Msg(format!(
                "boogu image-conditioned encode: no image tokens (id {image_token_id}) in input_ids"
            ))
        })?;
        let img_end = img_start + img_idx.len() as i32;

        // Token embeddings, then splice the vision embeds at the image positions.
        let mut hidden = self.embed_tokens.forward(input_ids)?;
        let dt = hidden.dtype();
        let img = image_embeds.expand_dims(0)?.as_dtype(dt)?; // [1, n, 4096]
        hidden = replace_seq(&hidden, &img, img_start, img_end, s)?;

        // 3-D interleaved MRoPE + causal mask.
        let (pt, ph, pw) = mrope_positions(&ids, image_token_id, grid_thw[1], grid_thw[2]);
        let (cos, sin) = mrope_cos_sin(&pt, &ph, &pw, self.head_dim, self.rope_theta, dt)?;
        let mask = build_mask(attention_mask, b, s)?;

        for (i, layer) in self.layers.iter().enumerate() {
            hidden = layer.forward(&hidden, &cos, &sin, &mask)?;
            // Deepstack: add the layer-i feature at the image positions (LM layers 0/1/2).
            if i < deepstack.len() {
                let ds = deepstack[i].expand_dims(0)?.as_dtype(dt)?; // [1, n, 4096]
                let mid = add(&slice_seq(&hidden, img_start, img_end)?, &ds)?;
                hidden = replace_seq(&hidden, &mid, img_start, img_end, s)?;
            }
        }
        Ok(rms_norm(&hidden, &self.final_norm, self.eps)?)
    }

    /// Multi-image-conditioned forward (Edit, `N ∈ [1, 5]` references). Splices each reference's
    /// `image_embedsⱼ` (`[nⱼ, 4096]`) at its own contiguous `<|image_pad|>` run, runs the 36 decoder
    /// layers under the 3-D interleaved MRoPE (positions advance `max(hⱼ, wⱼ)/merge` per image), and
    /// injects each image's 3 `deepstack` features at its run after layers 0/1/2 — the multi-image
    /// generalization of [`Self::last_hidden_with_image`] (`N = 1` is identical). `grids[j]` is image
    /// `j`'s patch grid `[t, h, w]`, in the same order the runs appear in `input_ids`. `b = 1`.
    #[allow(clippy::too_many_arguments)]
    pub fn last_hidden_with_image_multi(
        &self,
        input_ids: &Array,
        attention_mask: &Array,
        image_embeds: &[Array],
        deepstack: &[Vec<Array>],
        grids: &[[i32; 3]],
        image_token_id: i32,
    ) -> Result<Array> {
        let sh = input_ids.shape();
        let (b, s) = (sh[0], sh[1]);
        let ids = host_i32(input_ids)?;

        // One contiguous image-token run per reference image, in sequence order.
        let runs = image_token_runs(&ids, image_token_id, s);
        if runs.len() != image_embeds.len() || runs.len() != grids.len() {
            return Err(mlx_gen::Error::Msg(format!(
                "boogu multi-image encode: {} image-token runs but {} embeds / {} grids",
                runs.len(),
                image_embeds.len(),
                grids.len()
            )));
        }

        // Token embeddings, then splice each image's vision embeds at its run.
        let mut hidden = self.embed_tokens.forward(input_ids)?;
        let dt = hidden.dtype();
        for ((start, end), emb) in runs.iter().zip(image_embeds) {
            let img = emb.expand_dims(0)?.as_dtype(dt)?; // [1, nⱼ, 4096]
            hidden = replace_seq(&hidden, &img, *start, *end, s)?;
        }

        // 3-D interleaved MRoPE (per-image position advance) + causal mask.
        let (pt, ph, pw) = mrope_positions_multi(&ids, image_token_id, grids);
        let (cos, sin) = mrope_cos_sin(&pt, &ph, &pw, self.head_dim, self.rope_theta, dt)?;
        let mask = build_mask(attention_mask, b, s)?;

        for (i, layer) in self.layers.iter().enumerate() {
            hidden = layer.forward(&hidden, &cos, &sin, &mask)?;
            // Deepstack: add each image's layer-i feature at its run (LM layers 0/1/2).
            for ((start, end), ds_img) in runs.iter().zip(deepstack) {
                if i < ds_img.len() {
                    let ds = ds_img[i].expand_dims(0)?.as_dtype(dt)?; // [1, nⱼ, 4096]
                    let mid = add(&slice_seq(&hidden, *start, *end)?, &ds)?;
                    hidden = replace_seq(&hidden, &mid, *start, *end, s)?;
                }
            }
        }
        Ok(rms_norm(&hidden, &self.final_norm, self.eps)?)
    }
}

/// Slice `[b, s, d]` along the sequence axis (axis 1) to `[start, end)`.
fn slice_seq(x: &Array, start: i32, end: i32) -> Result<Array> {
    let idx: Vec<i32> = (start..end).collect();
    Ok(x.take_axis(Array::from_slice(&idx, &[end - start]), 1)?)
}

/// Replace `x[:, start:end, :]` with `repl` (`[b, end-start, d]`) via concat of the surrounding slices.
fn replace_seq(x: &Array, repl: &Array, start: i32, end: i32, s: i32) -> Result<Array> {
    let before = slice_seq(x, 0, start)?;
    let after = slice_seq(x, end, s)?;
    Ok(concatenate_axis(&[&before, repl, &after], 1)?)
}

/// 3-D MRoPE positions per token (mirrors `get_rope_index` + `get_vision_position_ids`): text tokens
/// advance `(i, i, i)`; an image block (at offset `cur`) gets `t = cur`, `h = cur + row`,
/// `w = cur + col` over its `(h/merge)×(w/merge)` merged grid, then `cur += max(h, w) / merge`.
fn mrope_positions(
    ids: &[i32],
    image_token_id: i32,
    grid_h: i32,
    grid_w: i32,
) -> (Vec<i32>, Vec<i32>, Vec<i32>) {
    let (llm_h, llm_w) = (grid_h / SPATIAL_MERGE, grid_w / SPATIAL_MERGE);
    let step = grid_h.max(grid_w) / SPATIAL_MERGE;
    let (mut pt, mut ph, mut pw) = (Vec::new(), Vec::new(), Vec::new());
    let mut cur = 0i32;
    let mut i = 0usize;
    while i < ids.len() {
        if ids[i] == image_token_id {
            for idx in 0..(llm_h * llm_w) {
                pt.push(cur);
                ph.push(cur + idx / llm_w);
                pw.push(cur + idx % llm_w);
            }
            cur += step;
            i += (llm_h * llm_w) as usize;
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

/// Contiguous runs of `image_token_id` in `ids` (`[start, end)` per run), in sequence order — one run
/// per reference image (the tokenizer separates images with `<|vision_end|><|vision_start|>` markers).
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

/// Multi-image 3-D MRoPE positions (mirrors `get_rope_index` over `image_grid_thw`): text tokens
/// advance `(i, i, i)`; the `k`-th image block (using `grids[k]`) at offset `cur` gets `t = cur`,
/// `h = cur + row`, `w = cur + col` over its `(h/merge)×(w/merge)` merged grid, then
/// `cur += max(h, w) / merge`. The single-image [`mrope_positions`] is the `grids.len() == 1` case.
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
        if ids[i] == image_token_id {
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

/// Build the interleaved-MRoPE `cos`/`sin` `[1, s, head_dim]` (cast to `dt`). Each of the `head_dim/2`
/// frequencies takes its position from the T/H/W axis per the interleave: within the first
/// `mrope_section[1]·3` indices, `j%3==1 → H`, `j%3==2 → W`, else `T` (the tail stays `T`).
fn mrope_cos_sin(
    pt: &[i32],
    ph: &[i32],
    pw: &[i32],
    head_dim: i32,
    theta: f32,
    dt: Dtype,
) -> Result<(Array, Array)> {
    let s = pt.len();
    let half = (head_dim / 2) as usize;
    let sec_h = (MROPE_SECTION[1] * 3) as usize;
    let sec_w = (MROPE_SECTION[2] * 3) as usize;
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
            emb[i * hd + half + j] = angle; // emb = cat(freqs, freqs)
        }
    }
    let arr = Array::from_slice(&emb, &[1, s as i32, head_dim]);
    Ok((arr.cos()?.as_dtype(dt)?, arr.sin()?.as_dtype(dt)?))
}

#[cfg(test)]
mod tests {
    use super::{image_token_runs, mrope_positions, mrope_positions_multi};

    const IMG: i32 = 999;

    /// Two image blocks of different sizes interleaved with text: the contiguous `<|image_pad|>` runs
    /// are located correctly and in order.
    #[test]
    fn image_token_runs_finds_each_block() {
        // [t t | img0×4 | t | img1×1 | t]
        let ids = [1, 1, IMG, IMG, IMG, IMG, 1, IMG, 1];
        assert_eq!(
            image_token_runs(&ids, IMG, ids.len() as i32),
            vec![(2, 6), (7, 8)]
        );
        // No image tokens → no runs.
        assert!(image_token_runs(&[1, 2, 3], IMG, 3).is_empty());
    }

    /// MRoPE positions advance per image: image `k` sits at the running `cur`, its merged grid fills
    /// the (t,h,w) axes, and `cur` then jumps by `max(h,w)/merge` — so the second image starts past
    /// the first, and trailing text continues from there.
    #[test]
    fn mrope_positions_multi_advances_per_image() {
        // img0 grid 4×4 (merge 2 ⇒ 2×2 = 4 tokens, step 2); img1 grid 2×2 (⇒ 1 token, step 1).
        let ids = [1, 1, IMG, IMG, IMG, IMG, 1, IMG, 1];
        let grids = [[1, 4, 4], [1, 2, 2]];
        let (pt, ph, pw) = mrope_positions_multi(&ids, IMG, &grids);
        assert_eq!(pt, vec![0, 1, 2, 2, 2, 2, 4, 5, 6]);
        assert_eq!(ph, vec![0, 1, 2, 2, 3, 3, 4, 5, 6]);
        assert_eq!(pw, vec![0, 1, 2, 3, 2, 3, 4, 5, 6]);
    }

    /// The single-image [`mrope_positions`] is exactly the one-grid case of the multi version.
    #[test]
    fn mrope_single_matches_multi_one_grid() {
        let ids = [1, IMG, IMG, IMG, IMG, 1];
        let single = mrope_positions(&ids, IMG, 4, 4);
        let multi = mrope_positions_multi(&ids, IMG, &[[1, 4, 4]]);
        assert_eq!(single, multi);
    }
}
