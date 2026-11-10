//! The `text2image` packed sequence: `[text | generated image]` (`build_image_packed_sequence`,
//! before_denoise.py, with no reference grids).
//!
//! * text rows carry `(k, 0, 0)` for `k` in `0..num_text_tokens` — H3's own
//!   [`text_position_ids`];
//! * the generated image is **one frame on the rotary clock**, at `t = num_text_tokens`, with H3's
//!   aspect-normalized `(h, w)` grid ([`frame_grid`]);
//! * tags are [`TEXT_TAG`] then [`IMAGE_TAG`] (gen-core's checkpoint constants; image rows reuse
//!   H3's video slot of the three-modality AdaLN table).
//!
//! The geometry is a single latent frame (`T = 1`). That is legal for **this** family only: H3's
//! video `JointGeometry` keeps refusing it, and nothing here goes through that type.

use mlx_rs::Array;

use mlx_gen::gen_core::h3_image::{IMAGE_TAG, TEXT_TAG};
use mlx_gen::{Error, Result};

use crate::dit::positions::{frame_grid, text_position_ids, to_array};

/// One `text2image` request's packed layout.
#[derive(Debug, Clone)]
pub struct ImageLayout {
    num_text_tokens: i32,
    latent_height: i32,
    latent_width: i32,
    patch: [i32; 3],
    position_ids: Array,
    token_tags: Array,
}

/// The `(t, h, w)` coordinates of every row, text first. Host math, f64, as the reference builds
/// them; [`to_array`] narrows once to f32.
pub fn t2i_position_rows(
    num_text_tokens: i32,
    latent_height: i32,
    latent_width: i32,
    patch: [i32; 3],
) -> Result<Vec<[f64; 3]>> {
    if num_text_tokens <= 0 {
        return Err(Error::Msg(format!(
            "minimax-h3-image layout: a text2image sequence needs text rows, got {num_text_tokens}"
        )));
    }
    if patch[0] != 1 {
        return Err(Error::Msg(format!(
            "minimax-h3-image layout: an image is one latent frame, so the temporal patch must be \
             1, got {patch:?}"
        )));
    }
    let mut rows = text_position_ids(num_text_tokens)?;
    let (grid, _) = frame_grid(latent_height, latent_width, patch[1], patch[2])?;
    let t = f64::from(num_text_tokens);
    rows.extend(grid.into_iter().map(|[h, w]| [t, h, w]));
    Ok(rows)
}

/// Per-row modality tags: [`TEXT_TAG`] for the text rows, [`IMAGE_TAG`] for the image rows.
pub fn t2i_token_tags(num_text_tokens: i32, num_image_rows: i32) -> Vec<i32> {
    let mut tags = vec![TEXT_TAG; num_text_tokens.max(0) as usize];
    tags.extend(std::iter::repeat_n(
        IMAGE_TAG,
        num_image_rows.max(0) as usize,
    ));
    tags
}

impl ImageLayout {
    /// The `text2image` layout over a `latent_height x latent_width` latent.
    pub fn t2i(
        num_text_tokens: i32,
        latent_height: i32,
        latent_width: i32,
        patch: [i32; 3],
    ) -> Result<Self> {
        let rows = t2i_position_rows(num_text_tokens, latent_height, latent_width, patch)?;
        let num_image_rows = (latent_height / patch[1]) * (latent_width / patch[2]);
        let tags = t2i_token_tags(num_text_tokens, num_image_rows);
        if tags.len() != rows.len() {
            return Err(Error::Msg(format!(
                "minimax-h3-image layout: {} tags for {} rows",
                tags.len(),
                rows.len()
            )));
        }
        Ok(Self {
            num_text_tokens,
            latent_height,
            latent_width,
            patch,
            position_ids: to_array(&rows)?,
            token_tags: Array::from_slice(&tags, &[tags.len() as i32]),
        })
    }

    /// Rows the text condition occupies — the leading `0..num_text_tokens`.
    pub fn num_text_tokens(&self) -> i32 {
        self.num_text_tokens
    }

    /// Rows of the generated image — the trailing `num_text_tokens..seq_len`.
    pub fn num_image_rows(&self) -> i32 {
        (self.latent_height / self.patch[1]) * (self.latent_width / self.patch[2])
    }

    /// Whole sequence length.
    pub fn seq_len(&self) -> i32 {
        self.num_text_tokens + self.num_image_rows()
    }

    /// Latent height of the generated image.
    pub fn latent_height(&self) -> i32 {
        self.latent_height
    }

    /// Latent width of the generated image.
    pub fn latent_width(&self) -> i32 {
        self.latent_width
    }

    /// The DiT patch the rows were packed with.
    pub fn patch(&self) -> [i32; 3] {
        self.patch
    }

    /// `[seq_len, 3]` f32 rotary coordinates.
    pub fn position_ids(&self) -> &Array {
        &self.position_ids
    }

    /// `[seq_len]` int32 modality tags.
    pub fn token_tags(&self) -> &Array {
        &self.token_tags
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The reference's t2i layout: text `(k, 0, 0)`, then one frame at `t = num_text` whose square
    /// spatial grid spans `[0, 32)` with the right endpoint excluded.
    #[test]
    fn t2i_layout_matches_the_reference_convention() {
        let rows = t2i_position_rows(3, 4, 4, [1, 2, 2]).unwrap();
        assert_eq!(rows.len(), 3 + 4);
        assert_eq!(rows[0], [0.0, 0.0, 0.0]);
        assert_eq!(rows[2], [2.0, 0.0, 0.0]);
        // 4x4 latent, patch 2 -> a 2x2 grid; square, so left = 0 and the step is 32 / 2.
        assert_eq!(rows[3], [3.0, 0.0, 0.0]);
        assert_eq!(rows[4], [3.0, 0.0, 16.0]);
        assert_eq!(rows[5], [3.0, 16.0, 0.0]);
        assert_eq!(rows[6], [3.0, 16.0, 16.0]);
        assert_eq!(t2i_token_tags(3, 4), vec![1, 1, 1, 0, 0, 0, 0]);
    }

    /// A non-square canvas is aspect-normalized: the wider axis spans more than 32.
    #[test]
    fn a_wide_canvas_spans_a_wider_w_axis() {
        let rows = t2i_position_rows(1, 4, 16, [1, 2, 2]).unwrap();
        let ws: Vec<f64> = rows[1..].iter().map(|r| r[2]).collect();
        let hs: Vec<f64> = rows[1..].iter().map(|r| r[1]).collect();
        let span = |v: &[f64]| {
            v.iter().cloned().fold(f64::MIN, f64::max) - v.iter().cloned().fold(f64::MAX, f64::min)
        };
        assert!(span(&ws) > span(&hs));
        // sqrt_area = 8, so the w axis covers ratio 2 of the unit interval, starting at -0.5.
        assert_eq!(ws[0], -16.0);
    }

    #[test]
    fn layout_counts_and_refusals() {
        let layout = ImageLayout::t2i(5, 128, 128, [1, 2, 2]).unwrap();
        assert_eq!(layout.num_image_rows(), 4096);
        assert_eq!(layout.seq_len(), 4101);
        assert_eq!(layout.position_ids().shape(), &[4101, 3]);
        assert_eq!(layout.token_tags().shape(), &[4101]);
        assert!(ImageLayout::t2i(0, 4, 4, [1, 2, 2]).is_err());
        assert!(ImageLayout::t2i(2, 4, 4, [2, 2, 2]).is_err());
        assert!(ImageLayout::t2i(2, 5, 4, [1, 2, 2]).is_err());
    }
}
