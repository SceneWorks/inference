//! DeepStack multi-level ViT feature fusion + the shared multimodal splice / interleaved-M-RoPE
//! position logic for the Qwen-VL family (Qwen3-VL `qwen3_vl` and Qwen3.6 `qwen3_5`).
//!
//! Both vision backbones drive their image/video prefill through these backbone-agnostic helpers
//! (sc-8080, the candle mirror of mlx-llm sc-8073/8074 + the sc-8314 `VlmDecode` refactor):
//!
//! - [`mrope_positions_mm`] — the `get_rope_index` port (B=1): interleaved-M-RoPE 3-D position rows
//!   over image **and** video grids, plus the `mrope_delta` continuation offset.
//! - [`splice_vision_features`] — replace placeholder-token rows of the text embeds with the vision
//!   tower's merged patch rows, in sequence order (mixed image+video).
//! - [`deepstack_fused_decoder_layers`] / [`add_visual_features`] — run the decoder layers, adding
//!   each tapped/merged ViT feature set to the visual-token rows at the first decoder layers.

use candle_core::Tensor;

use crate::error::{Error, Result};

/// Interleaved M-RoPE position rows (temporal / height / width, each length `S`) plus the
/// `mrope_delta` (`max_position + 1 − len`) the decode loop adds to continue positions after the
/// multimodal prompt.
pub type MropePositions = (Vec<i32>, Vec<i32>, Vec<i32>, i32);

/// The Qwen3-VL `get_rope_index` port (B=1) over image **and** video vision runs, returning the
/// interleaved-M-RoPE `(t, h, w)` rows (each length `S`) plus the `mrope_delta`
/// (`max_position + 1 − len`) the decode loop adds to continue positions after the prompt.
///
/// Text tokens advance all three axes by 1. A vision run lays its tokens over the
/// `(t, h/merge, w/merge)` grid offset by the shared cursor `cur` (`= max(prev) + 1`), then advances
/// the cursor by `max(grid_t, h/merge, w/merge)`. Qwen3-VL emits one video-token run **per frame**
/// (timestamp-separated), so each `[t, h, w]` video grid expands to `t × [1, h, w]` per-frame
/// blocks. Image grids are consumed one run per `image_grid_thw` entry (always `gt = 1`).
pub fn mrope_positions_mm(
    input_ids: &[i32],
    image_grid_thw: &[[i32; 3]],
    image_token_id: i32,
    video_grid_thw: &[[i32; 3]],
    video_token_id: i32,
    spatial_merge_size: i32,
) -> Result<MropePositions> {
    let merge = spatial_merge_size.max(1);
    // Each video grid `[t, h, w]` expands into `t` per-frame `[1, h, w]` blocks (one timestamp-
    // separated run per frame).
    let mut video_frames: Vec<[i32; 3]> = Vec::new();
    for &[t, h, w] in video_grid_thw {
        if t <= 0 {
            return Err(Error::Msg(format!("vlm mrope: bad video grid {:?}", [t, h, w])));
        }
        for _ in 0..t {
            video_frames.push([1, h, w]);
        }
    }

    let (mut t, mut h, mut w) = (Vec::new(), Vec::new(), Vec::new());
    let mut cur = 0i32;
    let (mut img_i, mut vid_i) = (0usize, 0usize);
    let mut i = 0usize;
    while i < input_ids.len() {
        let id = input_ids[i];
        let is_image = id == image_token_id;
        let is_video = !is_image && id == video_token_id;
        if is_image || is_video {
            let (grid, label): ([i32; 3], &str) = if is_image {
                let g = *image_grid_thw.get(img_i).ok_or_else(|| {
                    Error::Msg("vlm mrope: more image runs than image_grid_thw entries".into())
                })?;
                img_i += 1;
                (g, "image")
            } else {
                let g = *video_frames.get(vid_i).ok_or_else(|| {
                    Error::Msg("vlm mrope: more video frame runs than video_grid_thw frames".into())
                })?;
                vid_i += 1;
                (g, "video")
            };
            let (gt, gh, gw) = (grid[0], grid[1] / merge, grid[2] / merge);
            if gt <= 0 || gh <= 0 || gw <= 0 {
                return Err(Error::Msg(format!("vlm mrope: bad {label} grid {grid:?}")));
            }
            let count = (gt * gh * gw) as usize;
            let run = input_ids[i..].iter().take_while(|&&x| x == id).count();
            if run != count {
                return Err(Error::Msg(format!(
                    "vlm mrope: {label} run length {run} != grid tokens {count}"
                )));
            }
            let frame = gh * gw;
            for k in 0..count as i32 {
                t.push(k / frame + cur);
                let rem = k % frame;
                h.push(rem / gw + cur);
                w.push(rem % gw + cur);
            }
            cur += gt.max(gh).max(gw);
            i += count;
        } else {
            t.push(cur);
            h.push(cur);
            w.push(cur);
            cur += 1;
            i += 1;
        }
    }
    let maxpos = t
        .iter()
        .chain(h.iter())
        .chain(w.iter())
        .copied()
        .max()
        .unwrap_or(-1);
    let delta = maxpos + 1 - input_ids.len() as i32;
    Ok((t, h, w, delta))
}

/// Replace every row of `embeds` `[1, S, hidden]` whose id is any of `placeholder_tokens`
/// (`<|image_pad|>` and/or `<|video_pad|>`) with the next `vision_features`
/// `[num_vision_tokens, hidden]` row, in sequence order — the multimodal splice for a mixed
/// image+video prompt. The count of placeholder positions must equal the feature-row count.
///
/// The vision features are cast to the embeds' dtype/device, then text spans (from `embeds`) and
/// vision spans (from the features) are stitched in order via `narrow` + `cat` — no scatter.
pub fn splice_vision_features(
    embeds: &Tensor,
    input_ids: &[i32],
    vision_features: &Tensor,
    placeholder_tokens: &[i32],
) -> Result<Tensor> {
    let hidden = embeds.dim(2)?;
    let s = embeds.dim(1)?;
    let feats = vision_features
        .to_dtype(embeds.dtype())?
        .to_device(embeds.device())?;
    let is_ph = |id: i32| placeholder_tokens.contains(&id);
    let num_ph = input_ids.iter().filter(|&&x| is_ph(x)).count();
    if num_ph != feats.dim(0)? {
        return Err(Error::Msg(format!(
            "vlm splice: {num_ph} placeholder tokens != {} feature rows",
            feats.dim(0)?
        )));
    }
    if num_ph == 0 {
        return Ok(embeds.clone());
    }
    let mut pieces: Vec<Tensor> = Vec::new();
    let mut feat_off = 0usize;
    let mut i = 0usize;
    while i < s {
        let vis = is_ph(input_ids[i]);
        let mut j = i;
        while j < s && is_ph(input_ids[j]) == vis {
            j += 1;
        }
        let n = j - i;
        if vis {
            pieces.push(feats.narrow(0, feat_off, n)?.reshape((1, n, hidden))?);
            feat_off += n;
        } else {
            pieces.push(embeds.narrow(1, i, n)?);
        }
        i = j;
    }
    let refs: Vec<&Tensor> = pieces.iter().collect();
    Ok(Tensor::cat(&refs, 1)?)
}

/// Residually add a tapped/merged ViT feature set `visual` `[num_visual, hidden]` to the
/// visual-token rows of decoder hidden states `h` `[1, S, hidden]` (the rows where
/// `visual_pos_mask` is `true`), leaving text rows unchanged. The DeepStack fusion step. The number
/// of `true` mask positions must equal the feature-row count; batch must be 1.
pub fn add_visual_features(h: &Tensor, visual_pos_mask: &[bool], visual: &Tensor) -> Result<Tensor> {
    let (b, s, hidden) = h.dims3()?;
    if b != 1 {
        return Err(Error::Msg(format!(
            "deepstack fusion expects batch 1, got {b}"
        )));
    }
    if visual_pos_mask.len() != s {
        return Err(Error::Msg(format!(
            "deepstack mask length {} != seq {s}",
            visual_pos_mask.len()
        )));
    }
    let num_visual = visual_pos_mask.iter().filter(|&&m| m).count();
    if num_visual != visual.dim(0)? {
        return Err(Error::Msg(format!(
            "deepstack: {num_visual} visual positions != {} feature rows",
            visual.dim(0)?
        )));
    }
    if num_visual == 0 {
        return Ok(h.clone());
    }
    let vis = visual.to_dtype(h.dtype())?.to_device(h.device())?;
    let mut pieces: Vec<Tensor> = Vec::new();
    let mut off = 0usize;
    let mut i = 0usize;
    while i < s {
        let v = visual_pos_mask[i];
        let mut j = i;
        while j < s && visual_pos_mask[j] == v {
            j += 1;
        }
        let n = j - i;
        let h_span = h.narrow(1, i, n)?;
        if v {
            let add = vis.narrow(0, off, n)?.reshape((1, n, hidden))?;
            pieces.push(h_span.broadcast_add(&add)?);
            off += n;
        } else {
            pieces.push(h_span);
        }
        i = j;
    }
    let refs: Vec<&Tensor> = pieces.iter().collect();
    Ok(Tensor::cat(&refs, 1)?)
}

/// Run `num_layers` decoder layers via the `layer_forward` closure, fusing the `i`-th DeepStack
/// feature set into the visual-token rows after decoder layer `i` (for `i < deepstack.len()`).
///
/// The ViT taps at `deepstack_visual_indexes` produce `deepstack.len()` merged feature sets; these
/// are fused into the **first** `deepstack.len()` decoder layers (decoder layer `i` ← tap `i`), the
/// Qwen3-VL `_deepstack_process` convention. With an empty `deepstack` this is a plain decoder run.
pub fn deepstack_fused_decoder_layers<F>(
    h0: &Tensor,
    visual_pos_mask: &[bool],
    deepstack: &[Tensor],
    num_layers: usize,
    mut layer_forward: F,
) -> Result<Tensor>
where
    F: FnMut(usize, &Tensor) -> Result<Tensor>,
{
    let mut h = h0.clone();
    for i in 0..num_layers {
        h = layer_forward(i, &h)?;
        if let Some(feature) = deepstack.get(i) {
            h = add_visual_features(&h, visual_pos_mask, feature)?;
        }
    }
    Ok(h)
}

#[cfg(test)]
mod tests {
    use super::*;
    use candle_core::{Device, IndexOp};

    fn oracle() -> serde_json::Value {
        serde_json::from_str(include_str!("testdata/qwen3vl_mrope_oracle.json")).unwrap()
    }

    fn vec_i32(j: &serde_json::Value, k: &str) -> Vec<i32> {
        j[k].as_array()
            .unwrap()
            .iter()
            .map(|x| x.as_i64().unwrap() as i32)
            .collect()
    }

    fn grids(j: &serde_json::Value, k: &str) -> Vec<[i32; 3]> {
        j[k].as_array()
            .unwrap()
            .iter()
            .map(|g| {
                let a = g.as_array().unwrap();
                [
                    a[0].as_i64().unwrap() as i32,
                    a[1].as_i64().unwrap() as i32,
                    a[2].as_i64().unwrap() as i32,
                ]
            })
            .collect()
    }

    /// `mrope_positions_mm` (the `get_rope_index` port) must reproduce the real Qwen3-VL HF 3-D
    /// position rows + `mrope_delta` exactly, for both an image prompt and a multi-frame video prompt
    /// (the mlx-llm sc-8074 oracle). Pure index math — the error-prone novelty — verified weightless.
    #[test]
    fn mrope_positions_match_reference() {
        let j = oracle();
        let img_id = j["image_token_id"].as_i64().unwrap() as i32;
        let vid_id = j["video_token_id"].as_i64().unwrap() as i32;
        let merge = j["merge"].as_i64().unwrap() as i32;

        // Image-only prompt.
        let ri = &j["rope_index_image"];
        let ids = vec_i32(ri, "input_ids");
        let img_grids = grids(ri, "image_grid_thw");
        let (t, h, w, delta) =
            mrope_positions_mm(&ids, &img_grids, img_id, &[], vid_id, merge).unwrap();
        assert_eq!(t, vec_i32(ri, "t"));
        assert_eq!(h, vec_i32(ri, "h"));
        assert_eq!(w, vec_i32(ri, "w"));
        assert_eq!(delta, ri["delta"].as_i64().unwrap() as i32);

        // Video prompt (the `[t, h, w]` grid expands into `t` per-frame runs).
        let rv = &j["rope_index_video"];
        let ids = vec_i32(rv, "input_ids");
        let vid_grids = grids(rv, "video_grid_thw");
        let (t, h, w, delta) =
            mrope_positions_mm(&ids, &[], img_id, &vid_grids, vid_id, merge).unwrap();
        assert_eq!(t, vec_i32(rv, "t"));
        assert_eq!(h, vec_i32(rv, "h"));
        assert_eq!(w, vec_i32(rv, "w"));
        assert_eq!(delta, rv["delta"].as_i64().unwrap() as i32);
    }

    /// `splice_vision_features` replaces exactly the placeholder rows (image and video tokens),
    /// preserving text rows and order; a placeholder/feature count mismatch errors.
    #[test]
    fn splice_replaces_placeholder_rows() {
        let dev = Device::Cpu;
        let hidden = 4usize;
        // ids: text, IMG, IMG, text, VID, text  (img=9, vid=8)
        let ids = [1i32, 9, 9, 2, 8, 3];
        let s = ids.len();
        let embeds = Tensor::arange(0f32, (s * hidden) as f32, &dev)
            .unwrap()
            .reshape((1, s, hidden))
            .unwrap();
        // 3 vision rows (2 image + 1 video), each a distinct constant.
        let feats = Tensor::from_vec(
            vec![
                10f32, 10., 10., 10., // img row 0
                20., 20., 20., 20., // img row 1
                30., 30., 30., 30., // vid row 0
            ],
            (3, hidden),
            &dev,
        )
        .unwrap();
        let out = splice_vision_features(&embeds, &ids, &feats, &[9, 8]).unwrap();
        assert_eq!(out.dims(), &[1, s, hidden]);
        let rows: Vec<Vec<f32>> = out
            .i(0)
            .unwrap()
            .to_vec2::<f32>()
            .unwrap();
        assert_eq!(rows[1], vec![10., 10., 10., 10.]); // image row 0
        assert_eq!(rows[2], vec![20., 20., 20., 20.]); // image row 1
        assert_eq!(rows[4], vec![30., 30., 30., 30.]); // video row 0
        assert_eq!(rows[0], vec![0., 1., 2., 3.]); // text untouched
        assert_eq!(rows[3], vec![12., 13., 14., 15.]); // text untouched

        // Count mismatch is an error.
        assert!(splice_vision_features(&embeds, &ids, &feats, &[9]).is_err());
    }

    /// `add_visual_features` (DeepStack fusion) adds the feature rows only at masked positions.
    #[test]
    fn add_visual_features_adds_at_masked_positions() {
        let dev = Device::Cpu;
        let hidden = 3usize;
        let s = 4usize;
        let h = Tensor::ones((1, s, hidden), candle_core::DType::F32, &dev).unwrap();
        let mask = [false, true, true, false];
        let visual = Tensor::from_vec(
            vec![5f32, 5., 5., 7., 7., 7.],
            (2, hidden),
            &dev,
        )
        .unwrap();
        let out = add_visual_features(&h, &mask, &visual).unwrap();
        let rows: Vec<Vec<f32>> = out.i(0).unwrap().to_vec2::<f32>().unwrap();
        assert_eq!(rows[0], vec![1., 1., 1.]); // unmasked
        assert_eq!(rows[1], vec![6., 6., 6.]); // 1 + 5
        assert_eq!(rows[2], vec![8., 8., 8.]); // 1 + 7
        assert_eq!(rows[3], vec![1., 1., 1.]); // unmasked
    }
}
