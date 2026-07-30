//! The crate-facing text encoder: tokenizer + [`Qwen3VlTextEncoder`] + the packed-encode /
//! system-prompt-drop policy. Port of the reference's `TextEncoder` wrapper
//! (`_vendor/mage_flow/models/modules/text_encoder.py:425-577`) and the `_encode_texts_packed`
//! helper that drives it (`pipeline.py:218-231`).

use mlx_rs::ops::concatenate_axis;
use mlx_rs::Array;

use image::{imageops::FilterType, RgbImage};
use mlx_gen::tokenizer::TextTokenizer;
use mlx_gen::{Error, Result};
use mlx_gen_boogu::vision::preprocess::preprocess_image;
use mlx_gen_boogu::VisionTower;

use super::encoder::{
    cu_seqlens_from_lens, grounded_mrope_positions, seq_lens_from_cu, Qwen3VlTextEncoder,
};
use super::prompt::{edit_body, truncate, PromptKind};

const IMAGE_TOKEN_ID: i32 = 151_655;

/// The DiT-consumable text conditioning: the packed `txt` stream and its per-prompt lengths.
///
/// **There is no pooled vector.** The reference also computes `vec` = the mean of each segment's
/// valid tokens (`text_encoder.py:565`), but the DiT discards it outright — `mage_flow.py:116`
/// builds `txt_vec = torch.zeros(...)` and adds that to `temb`, never reading the encoder's
/// pooling. Carrying it would be an unused knob whose only effect is to imply the DiT uses it.
pub struct Conditioning {
    /// `[Σ(Lᵢ − drop_idx), hidden]` — every prompt's post-drop hidden states, concatenated in
    /// prompt order. This is the reference's `txt`.
    pub txt: Array,
    /// Per-prompt token counts **after** the drop (`txt_seq_lens`), so a consumer can split `txt`
    /// back apart or build the DiT's `cu_seqlens`.
    pub seq_lens: Vec<usize>,
}

impl Conditioning {
    /// Cumulative boundaries `[0, L₀, L₀+L₁, …]` over [`seq_lens`](Self::seq_lens).
    pub fn cu_seqlens(&self) -> Vec<i32> {
        cu_seqlens_from_lens(&self.seq_lens)
    }

    /// Prompt count.
    pub fn len(&self) -> usize {
        self.seq_lens.len()
    }

    /// `true` when no prompt was encoded.
    pub fn is_empty(&self) -> bool {
        self.seq_lens.is_empty()
    }

    /// The `i`-th prompt's slice of [`txt`](Self::txt), `[Lᵢ, hidden]`.
    pub fn segment(&self, i: usize) -> Result<Array> {
        let start: usize = self.seq_lens.iter().take(i).sum();
        let len = *self.seq_lens.get(i).ok_or_else(|| {
            Error::Msg(format!(
                "mage_flow conditioning: prompt {i} of {}",
                self.seq_lens.len()
            ))
        })?;
        Ok(self
            .txt
            .split_axis(&[start as i32, (start + len) as i32], 0)?
            .swap_remove(1))
    }
}

/// Tokenizer + Qwen3-VL LM, with the Mage-Flow prompt policy applied.
pub struct MageTextEncoder {
    tokenizer: TextTokenizer,
    lm: Qwen3VlTextEncoder,
    vision: Option<VisionTower>,
}

impl MageTextEncoder {
    pub fn quantize(&mut self, bits: i32) -> Result<()> {
        self.lm.quantize(bits)?;
        if let Some(vision) = &mut self.vision {
            vision.quantize(bits)?;
        }
        Ok(())
    }

    pub fn quantized_linear_count(&self) -> usize {
        self.lm.quantized_linear_count()
    }

    /// Pair an already-loaded tokenizer and LM. [`load`](super::load()) builds both from a snapshot.
    pub fn new(tokenizer: TextTokenizer, lm: Qwen3VlTextEncoder) -> Self {
        Self {
            tokenizer,
            lm,
            vision: None,
        }
    }

    pub fn new_multimodal(
        tokenizer: TextTokenizer,
        lm: Qwen3VlTextEncoder,
        vision: VisionTower,
    ) -> Self {
        Self {
            tokenizer,
            lm,
            vision: Some(vision),
        }
    }

    /// The Qwen2 fast tokenizer shipped in `text_encoder/tokenizer.json`.
    pub fn tokenizer(&self) -> &TextTokenizer {
        &self.tokenizer
    }

    /// The language model — the seam the vision/edit path (sc-14048) drives directly.
    pub fn lm(&self) -> &Qwen3VlTextEncoder {
        &self.lm
    }

    /// Template `body` under `kind`, tokenize it, and right-truncate to `kind`'s budget.
    ///
    /// `add_special_tokens` is irrelevant for this tokenizer (its post-processor is a bare
    /// `ByteLevel` with no BOS/EOS template — the ChatML markers come from the template text), but
    /// it is passed as `true` to match HF's default `tokenizer(text)` call in `pipeline.py:226`.
    pub fn token_ids(&self, body: &str, kind: PromptKind) -> Result<Vec<i32>> {
        let text = kind.render(body);
        let mut ids = self.tokenizer.encode_ids(&text, true)?;
        truncate(&mut ids, kind);
        if ids.len() <= kind.drop_idx() {
            return Err(Error::Msg(format!(
                "mage_flow text encoder: the templated prompt is {} token(s) but {} leading \
                 system-prompt token(s) are dropped — the conditioning would be empty",
                ids.len(),
                kind.drop_idx()
            )));
        }
        Ok(ids)
    }

    /// Encode a batch of prompt bodies in ONE packed varlen forward and drop each segment's
    /// leading `kind.drop_idx()` system-prompt tokens — the reference's `_encode_texts_packed`.
    ///
    /// The positive and negative prompts of a CFG run are encoded through a **single** call, as
    /// the reference does (`pipeline.py:318-323`); per-segment isolation is what makes that
    /// equivalent to separate calls (see [`Qwen3VlTextEncoder::forward_packed`]).
    pub fn encode(&self, bodies: &[&str], kind: PromptKind) -> Result<Conditioning> {
        if bodies.is_empty() {
            return Err(Error::Msg(
                "mage_flow text encoder: no prompts to encode".into(),
            ));
        }
        let mut ids = Vec::new();
        let mut lens = Vec::with_capacity(bodies.len());
        for body in bodies {
            let seg = self.token_ids(body, kind)?;
            lens.push(seg.len());
            ids.extend_from_slice(&seg);
        }
        self.encode_packed_ids(&ids, &cu_seqlens_from_lens(&lens), kind.drop_idx())
    }

    /// Encode pre-built packed token ids. Separated from [`encode`](Self::encode) because the edit
    /// path builds its own ids: the `<|image_pad|>` placeholder has to be repeated once per merged
    /// vision token, which needs the reference image's patch grid (sc-14048).
    ///
    /// `drop_idx` is explicit here for the same reason the reference exposes
    /// `drop_idx_override` (`text_encoder.py:470`).
    pub fn encode_packed_ids(
        &self,
        ids: &[i32],
        cu_seqlens: &[i32],
        drop_idx: usize,
    ) -> Result<Conditioning> {
        let lens = seq_lens_from_cu(cu_seqlens, ids.len())?;
        let hidden = self.lm.forward_packed(ids, cu_seqlens)?;
        drop_system_prompt(&hidden, &lens, drop_idx)
    }

    /// Encode one edit instruction grounded by one or more source images through Qwen3-VL.
    pub fn encode_edit(&self, instruction: &str, references: &[RgbImage]) -> Result<Conditioning> {
        if references.is_empty() {
            return Err(Error::Msg(
                "mage_flow edit: at least one reference image is required".into(),
            ));
        }
        let mut embeds = Vec::with_capacity(references.len());
        let mut deepstack = Vec::with_capacity(references.len());
        let mut grids = Vec::with_capacity(references.len());
        for reference in references {
            let (image_embeds, image_deepstack, grid) = self.vision_features(reference)?;
            embeds.push(image_embeds);
            deepstack.push(image_deepstack);
            grids.push(grid);
        }
        self.encode_edit_with_features(instruction, &embeds, &deepstack, &grids)
    }

    /// Encode an edit instruction from precomputed vision/deepstack boundaries.
    pub fn encode_edit_with_features(
        &self,
        instruction: &str,
        embeds: &[Array],
        deepstack: &[Vec<Array>],
        grids: &[[i32; 3]],
    ) -> Result<Conditioning> {
        let counts = embeds
            .iter()
            .map(|embed| embed.shape()[0] as usize)
            .collect::<Vec<_>>();
        let ids = self.edit_input_ids(instruction, &counts)?;
        let hidden = self
            .lm
            .forward_grounded(&ids, IMAGE_TOKEN_ID, embeds, deepstack, grids)?
            .reshape(&[ids.len() as i32, -1])?;
        drop_system_prompt(&hidden, &[ids.len()], PromptKind::Edit.drop_idx())
    }

    /// Return the shared Qwen3-VL vision boundaries for one edit reference.
    pub fn vision_features(&self, reference: &RgbImage) -> Result<(Array, Vec<Array>, [i32; 3])> {
        let vision = self.vision.as_ref().ok_or_else(|| {
            Error::Msg("mage_flow edit: text encoder was loaded without the vision tower".into())
        })?;
        let capped = cap_long_edge(reference, crate::config::VL_COND_LONG_EDGE);
        let (pixels, grid) = preprocess_image(&capped)?;
        let (embeds, deepstack) = vision.forward(&pixels, &[grid])?;
        Ok((embeds, deepstack, grid))
    }

    /// Render/tokenize an edit prompt and expand each visual placeholder to its merged-token run.
    pub fn edit_input_ids(
        &self,
        instruction: &str,
        image_token_counts: &[usize],
    ) -> Result<Vec<i32>> {
        let base = self.token_ids(
            &edit_body(instruction, image_token_counts.len()),
            PromptKind::Edit,
        )?;
        expand_image_tokens(&base, image_token_counts)
    }

    /// Return the distinct temporal/height/width M-RoPE axes for an expanded edit prompt.
    pub fn edit_mrope_axes(
        &self,
        ids: &[i32],
        grids: &[[i32; 3]],
    ) -> Result<(Vec<i32>, Vec<i32>, Vec<i32>)> {
        let positions = grounded_mrope_positions(ids, IMAGE_TOKEN_ID, grids)?;
        let (t, h, w) = positions.axes();
        Ok((t.to_vec(), h.to_vec(), w.to_vec()))
    }

    /// Return pre-deepstack outputs from LM layers 0/1/2 for parity localization.
    pub fn edit_early_lm_trace(
        &self,
        instruction: &str,
        embeds: &[Array],
        deepstack: &[Vec<Array>],
        grids: &[[i32; 3]],
    ) -> Result<Vec<Array>> {
        let counts = embeds
            .iter()
            .map(|embed| embed.shape()[0] as usize)
            .collect::<Vec<_>>();
        let ids = self.edit_input_ids(instruction, &counts)?;
        Ok(self
            .lm
            .forward_grounded_trace(&ids, IMAGE_TOKEN_ID, embeds, deepstack, grids)?
            .1)
    }
}

fn cap_long_edge(image: &RgbImage, max_edge: u32) -> RgbImage {
    let long = image.width().max(image.height());
    if long <= max_edge {
        return image.clone();
    }
    let scale = max_edge as f64 / long as f64;
    let width = (image.width() as f64 * scale).round().max(1.0) as u32;
    let height = (image.height() as f64 * scale).round().max(1.0) as u32;
    image::imageops::resize(image, width, height, FilterType::CatmullRom)
}

fn expand_image_tokens(ids: &[i32], counts: &[usize]) -> Result<Vec<i32>> {
    let placeholders = ids.iter().filter(|&&id| id == IMAGE_TOKEN_ID).count();
    if placeholders != counts.len() {
        return Err(Error::Msg(format!(
            "mage_flow edit: template has {placeholders} image placeholder(s) but {} reference(s)",
            counts.len()
        )));
    }
    let mut out = Vec::with_capacity(ids.len() + counts.iter().sum::<usize>());
    let mut image = 0usize;
    for &id in ids {
        if id == IMAGE_TOKEN_ID {
            out.extend(std::iter::repeat_n(id, counts[image]));
            image += 1;
        } else {
            out.push(id);
        }
    }
    Ok(out)
}

/// Drop the first `drop_idx` tokens of every packed segment and re-concatenate — `h[drop_idx:]`
/// per sequence (`text_encoder.py:551-567`).
///
/// This is the *second* half of the GAP-1 correction: taking the right layer but the wrong
/// `drop_idx` yields a tensor of the wrong length whose content is offset by the tail of the
/// system prompt.
pub(crate) fn drop_system_prompt(
    hidden: &Array,
    lens: &[usize],
    drop_idx: usize,
) -> Result<Conditioning> {
    let total: usize = lens.iter().sum();
    let rows = hidden.shape()[0] as usize;
    if rows != total {
        return Err(Error::Msg(format!(
            "mage_flow text encoder: hidden state has {rows} row(s) but the pack declares {total}"
        )));
    }

    let mut parts = Vec::with_capacity(lens.len());
    let mut seq_lens = Vec::with_capacity(lens.len());
    let mut start = 0usize;
    for &len in lens {
        if len <= drop_idx {
            return Err(Error::Msg(format!(
                "mage_flow text encoder: a packed segment is {len} token(s) but {drop_idx} leading \
                 system-prompt token(s) are dropped"
            )));
        }
        let keep_from = start + drop_idx;
        let keep_to = start + len;
        parts.push(
            hidden
                .split_axis(&[keep_from as i32, keep_to as i32], 0)?
                .swap_remove(1),
        );
        seq_lens.push(len - drop_idx);
        start = keep_to;
    }

    let txt = if parts.len() == 1 {
        parts.swap_remove(0)
    } else {
        let refs: Vec<&Array> = parts.iter().collect();
        concatenate_axis(&refs, 0)?
    };
    Ok(Conditioning { txt, seq_lens })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn ramp(rows: i32, cols: i32) -> Array {
        let data: Vec<f32> = (0..rows * cols).map(|v| v as f32).collect();
        Array::from_slice(&data, &[rows, cols])
    }

    /// The drop is applied **per segment**, not once to the packed tensor — the difference a
    /// single-segment test cannot see. Two segments of 5 and 4 tokens with `drop_idx = 2` must
    /// yield rows `[2,3,4]` of the first and `[7,8]` of the second, i.e. 3 + 2 rows.
    #[test]
    fn drop_is_per_segment_not_once_over_the_pack() {
        let hidden = ramp(9, 2);
        let cond = drop_system_prompt(&hidden, &[5, 4], 2).unwrap();
        assert_eq!(cond.seq_lens, vec![3, 2]);
        assert_eq!(cond.txt.shape(), [5, 2]);
        // Row r of the ramp is [2r, 2r+1]; kept rows are 2,3,4 then 7,8.
        assert_eq!(
            cond.txt.as_slice::<f32>(),
            &[4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 14.0, 15.0, 16.0, 17.0]
        );
        assert_eq!(cond.cu_seqlens(), vec![0, 3, 5]);
        // `segment` returns whole rows: the second prompt is txt rows 3..5.
        let seg = cond.segment(1).unwrap();
        assert_eq!(seg.shape(), [2, 2]);
        assert_eq!(seg.as_slice::<f32>(), &[14.0, 15.0, 16.0, 17.0]);
        assert_eq!(cond.segment(0).unwrap().shape(), [3, 2]);
        assert!(cond.segment(2).is_err(), "out-of-range prompt index");
    }

    /// A segment no longer than `drop_idx` is an error, not an empty slice.
    #[test]
    fn a_segment_shorter_than_the_drop_is_rejected() {
        let hidden = ramp(4, 2);
        assert!(drop_system_prompt(&hidden, &[4], 4).is_err());
        assert!(drop_system_prompt(&hidden, &[1, 3], 2).is_err());
    }

    /// The declared pack must match the hidden state it is applied to.
    #[test]
    fn a_pack_that_disagrees_with_the_hidden_state_is_rejected() {
        let hidden = ramp(9, 2);
        assert!(drop_system_prompt(&hidden, &[5, 5], 2).is_err());
    }

    #[test]
    fn image_placeholders_expand_one_run_per_reference() {
        let ids = [1, IMAGE_TOKEN_ID, 2, IMAGE_TOKEN_ID, 3];
        assert_eq!(
            expand_image_tokens(&ids, &[2, 3]).unwrap(),
            [
                1,
                IMAGE_TOKEN_ID,
                IMAGE_TOKEN_ID,
                2,
                IMAGE_TOKEN_ID,
                IMAGE_TOKEN_ID,
                IMAGE_TOKEN_ID,
                3
            ]
        );
        assert!(expand_image_tokens(&ids, &[2]).is_err());
        assert!(expand_image_tokens(&[1, 2], &[2]).is_err());
    }

    #[test]
    fn vision_conditioning_caps_only_the_long_edge() {
        let wide = RgbImage::new(768, 192);
        let capped = cap_long_edge(&wide, 384);
        assert_eq!(capped.dimensions(), (384, 96));
        let small = RgbImage::new(320, 160);
        assert_eq!(cap_long_edge(&small, 384).dimensions(), (320, 160));
    }
}
