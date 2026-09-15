//! **What one DiT forward costs at a given packed sequence length** (sc-17152) — the pure
//! arithmetic the duration bound is derived from, and the `i32::MAX` element-count gate.
//!
//! MiniMax-H3 is trained with sparse attention, but **the sparse-attention inference implementation
//! is not released**, so the open weights run dense attention over one packed
//! text + audio + video sequence ([`crate::denoise::PackedLayout`]). The sequence grows with
//! *both* axes of a request — the canvas (rows per latent frame) and the duration (latent frames) —
//! and the two enter differently:
//!
//! | quantity | grows as | at 768×1344 / 345 frames |
//! |---|---|---|
//! | projections, feed-forward, norms | `O(S)` | 0.75–2.98 G elements |
//! | attention score domain | `O(S²)` | **606 G elements** |
//!
//! # Two separate questions, and they have different answers
//!
//! **1. What is the widest tensor the forward actually writes?** That is
//! [`DitSequenceCost::widest_materialized_elements`], and on the shipped geometry it is the SwiGLU
//! projection's `[1, S, 2·ffn_dim]` — **not** the attention. It crosses [`MAX_WRITABLE_ELEMS`] on
//! the default 768×1344 canvas at 260 frames for `t2va` and at 243 for a two-keyframe `fl2va`
//! request ([`largest_writable_frame_count`]) — a real, reachable geometry well inside the
//! advertised envelope. Crossing it is a **memory** cost, not a correctness one:
//! `tests/sequence_cost_real.rs` measures MLX's `matmul` as exact on both sides of the bound at
//! these exact widths.
//!
//! **2. What would a *materializing* attention build?** That is
//! [`DitSequenceCost::dense_score_elements`], `heads · S²`, and it is over
//! [`MAX_WRITABLE_ELEMS`] at **every legal geometry this model has** — 1.3× at the smallest legal
//! render. It is reported rather than gated because MLX's fused
//! `fast::scaled_dot_product_attention` **streams** the scores and never writes that tensor
//! (`mlx_gen::attention`'s measured contract; `tests/sequence_cost_real.rs` re-measures it at
//! *this* model's shape, which is 6× longer than the Z-Image shape that contract was measured at).
//! The number is kept because it is exactly what a hand-rolled `matmul → softmax → matmul` or a
//! shape the fused kernel rejects would cost, and it is the input sc-18661's bounded-attention rung
//! is sized on.
//!
//! Conflating the two is the trap: "the attention is 606 G elements, therefore we are over the
//! bound" is true of a tensor that does not exist, while the tensor that *does* cross is the one
//! nobody looks at.
//!
//! # Why `i32::MAX` at all
//!
//! `MAX_WRITABLE_ELEMS` is `mlx_gen::gen_core::tiling`'s threshold, and its doc is explicit that it
//! is **operation-specific, not a universal law** — sc-12748 measured `conv3d`, `pad`,
//! `concatenate`, `reshape` and `as_slice` as int64-safe on this pin, with `from_slice` the one
//! residual. `matmul` was not on that list, and the SwiGLU projection is a matmul, so
//! `tests/sequence_cost_real.rs::the_feed_forward_projection_is_exact_above_the_writable_bound`
//! measures it at these exact widths rather than inheriting a neighbouring op's result.

use mlx_gen::{Error, Result};

use crate::denoise::{JointGeometry, LEGAL_FRAME_COUNTS, MINIMAX_H3_FPS};
use crate::dit::config::MiniMaxH3DitConfig;

/// The MLX element-count threshold, in elements: `i32::MAX`.
///
/// Re-exported from `mlx_gen::gen_core::tiling` rather than redeclared, so this crate gates on the
/// same number — and inherits the same per-operation caveats — as the tiled VAE decoders.
pub const MAX_WRITABLE_ELEMS: i64 = mlx_gen::tiling::MAX_WRITABLE_ELEMS;

/// Element counts of one DiT block's intermediates at a packed sequence length.
///
/// Batch is always 1: [`crate::model::descriptor`] declares `max_count: 1` and the checkpoint is
/// guidance-distilled, so there is no unconditional branch to double the batch either.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct DitSequenceCost {
    /// Rows in the packed text + audio + video sequence.
    pub seq_len: i64,
    /// One of `to_q` / `to_k` / `to_v`'s output: `S · heads · head_dim`.
    pub projection_elements: i64,
    /// The residual stream: `S · hidden_size`.
    pub residual_elements: i64,
    /// `ff.net.0.proj`'s output: `S · 2 · ffn_dim`. The **widest tensor the forward writes**.
    pub ff_proj_elements: i64,
    /// `heads · S²` — the score domain a *materializing* attention would build. MLX's fused kernel
    /// does not; see the module docs.
    pub dense_score_elements: i64,
}

impl DitSequenceCost {
    /// Derive every count from a sequence length and the DiT geometry.
    ///
    /// Everything is computed in `i64`. The whole point of this module is counts above `i32::MAX`,
    /// so an `i32` product here would wrap into a small positive number and report that the
    /// geometry is comfortably inside the bound.
    pub fn new(seq_len: i64, cfg: &MiniMaxH3DitConfig) -> Result<Self> {
        cfg.validate()?;
        if seq_len <= 0 {
            return Err(Error::Msg(format!(
                "minimax-h3 cost: the packed sequence must have at least one row, got {seq_len}"
            )));
        }
        let heads = i64::from(cfg.num_attention_heads);
        Ok(Self {
            seq_len,
            projection_elements: seq_len * heads * i64::from(cfg.attention_head_dim),
            residual_elements: seq_len * i64::from(cfg.hidden_size),
            // `ff.net.0.proj` emits `[value | gate]`, i.e. TWO ffn_dim halves in one tensor — see
            // `crate::dit::layers::DitFeedForward`. Reading this as one `ffn_dim` halves the number
            // and moves the bound crossing a whole canvas tier.
            ff_proj_elements: seq_len * 2 * i64::from(cfg.ffn_dim),
            dense_score_elements: heads.saturating_mul(seq_len.saturating_mul(seq_len)),
        })
    }

    /// The widest tensor one forward **materializes**, in elements.
    ///
    /// The SwiGLU projection on every shipped geometry (`2 · 14336` is four times the attention
    /// inner width of 7168 and over five times `hidden_size`). Stated as a `max` rather than
    /// hardcoded to the feed-forward so a future config whose attention is wider is not silently
    /// mis-bounded.
    pub fn widest_materialized_elements(&self) -> i64 {
        self.ff_proj_elements
            .max(self.projection_elements)
            .max(self.residual_elements)
    }

    /// Whether the widest materialized tensor crosses [`MAX_WRITABLE_ELEMS`].
    pub fn over_writable_bound(&self) -> bool {
        self.widest_materialized_elements() > MAX_WRITABLE_ELEMS
    }

    /// Bytes the widest materialized tensor occupies at `element_bytes` (2 for bf16).
    pub fn widest_materialized_bytes(&self, element_bytes: i64) -> i64 {
        self.widest_materialized_elements()
            .saturating_mul(element_bytes)
    }
}

/// Rows one latent frame contributes to the packed sequence: `(H/ph) · (W/pw)` latent patches.
///
/// The same arithmetic [`crate::denoise::PackedLayout::build`] applies, extracted so a bound can be
/// computed **before** any weight is read and without building the layout's position tensors.
pub fn rows_per_latent_frame(geometry: &JointGeometry, patch: [i32; 3]) -> Result<i64> {
    let [_, ph, pw] = patch;
    if ph <= 0 || pw <= 0 {
        return Err(Error::Msg(format!(
            "minimax-h3 cost: the patch {patch:?} must be positive"
        )));
    }
    let rows = i64::from(geometry.latent_height / ph) * i64::from(geometry.latent_width / pw);
    if rows <= 0 {
        return Err(Error::Msg(format!(
            "minimax-h3 cost: a {}x{} latent has no whole patch at {patch:?}",
            geometry.latent_height, geometry.latent_width
        )));
    }
    Ok(rows)
}

/// Packed sequence length: `text + condition video + audio + video` rows.
///
/// Mirrors [`crate::denoise::PackedLayout::build`]'s own sum in `i64` and without tensors, so
/// `validate` can reach it. `num_condition_video_rows` is `keyframe_anchors · rows_per_frame`
/// (zero for `t2va`; sc-17148's `fl2va` anchors are the non-zero case).
pub fn packed_seq_len(
    geometry: &JointGeometry,
    patch: [i32; 3],
    num_text_tokens: i32,
    num_keyframe_anchors: i32,
    audio_channels: i32,
) -> Result<i64> {
    if num_text_tokens <= 0 || audio_channels <= 0 || num_keyframe_anchors < 0 {
        return Err(Error::Msg(format!(
            "minimax-h3 cost: {num_text_tokens} text tokens / {num_keyframe_anchors} anchors / \
             {audio_channels} audio channels is not a packable request"
        )));
    }
    let rows_per_frame = rows_per_latent_frame(geometry, patch)?;
    Ok(i64::from(num_text_tokens)
        + i64::from(num_keyframe_anchors) * rows_per_frame
        + i64::from(geometry.num_audio_latents) * i64::from(audio_channels)
        + i64::from(geometry.num_latent_frames) * rows_per_frame)
}

/// The **largest legal frame count** whose widest materialized tensor stays inside
/// [`MAX_WRITABLE_ELEMS`] on a given latent canvas, or `None` if even the shortest legal clip is
/// over.
///
/// Walks [`LEGAL_FRAME_COUNTS`] rather than solving for a frame count, because the answer must be a
/// lattice point — an interpolated frame count is not a renderable one.
pub fn largest_writable_frame_count(
    latent_height: i32,
    latent_width: i32,
    patch: [i32; 3],
    num_text_tokens: i32,
    num_keyframe_anchors: i32,
    audio_channels: i32,
    cfg: &MiniMaxH3DitConfig,
) -> Result<Option<i32>> {
    let mut best = None;
    for &frames in &LEGAL_FRAME_COUNTS {
        let geometry = JointGeometry::new(frames, latent_height, latent_width)?;
        let seq = packed_seq_len(
            &geometry,
            patch,
            num_text_tokens,
            num_keyframe_anchors,
            audio_channels,
        )?;
        if DitSequenceCost::new(seq, cfg)?.over_writable_bound() {
            break;
        }
        best = Some(frames);
    }
    Ok(best)
}

/// That frame count as a duration in seconds.
pub fn writable_duration_seconds(frames: i32) -> f64 {
    f64::from(frames) / MINIMAX_H3_FPS
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::VAE_RATIO;
    use crate::pipeline::PATCH_SIZE;

    /// The text-token count the bound tables are quoted at. `tests/sequence_cost_real.rs` prints
    /// the real tokenizer's count for the first-light prompt; the bound moves by one sequence row
    /// per token, i.e. 28672 elements, so a ±100-token error moves the crossing by 2.9 M elements
    /// out of 2.1 G — under a thousandth of the bound, and nowhere near a lattice rung.
    const TEXT_TOKENS: i32 = 64;

    fn latent(width: u32, height: u32) -> (i32, i32) {
        (
            (height / VAE_RATIO as u32) as i32,
            (width / VAE_RATIO as u32) as i32,
        )
    }

    fn seq_at(width: u32, height: u32, frames: i32) -> i64 {
        let (lh, lw) = latent(width, height);
        let g = JointGeometry::new(frames, lh, lw).unwrap();
        packed_seq_len(&g, PATCH_SIZE, TEXT_TOKENS, 0, 2).unwrap()
    }

    /// The packed sequence length really is ~31 k at the smallest legal clip on the default canvas
    /// and ~104 k at the longest — the epic's headline "~94 k tokens at 15 s" is the right order and
    /// slightly LOW, because it assumed a `4k + 1` frame lattice rather than `17n + 5`.
    #[test]
    fn the_packed_sequence_is_thirty_eight_thousand_rows_at_the_floor_and_one_hundred_four_at_the_top(
    ) {
        // The default canvas: 768 short edge, 16:9 -> 1344 x 768.
        assert_eq!(seq_at(1344, 768, 124), 37_774);
        assert_eq!(seq_at(1344, 768, 345), 104_030);
        // ...and the gating canvas the cost curve was measured at.
        assert_eq!(seq_at(576, 320, 124), 7_138);
        assert_eq!(seq_at(576, 320, 345), 19_574);

        // It really is dominated by the video rows: audio is 1150 rows at 345 frames, video 102 816.
        let g = JointGeometry::new(345, 48, 84).unwrap();
        assert_eq!(i64::from(g.num_audio_latents) * 2, 1_150);
        assert_eq!(
            i64::from(g.num_latent_frames) * rows_per_latent_frame(&g, PATCH_SIZE).unwrap(),
            102_816
        );
    }

    /// **The widest materialized tensor is the SwiGLU projection, not the attention.**
    ///
    /// Four times the attention projection and over five times the residual stream, at every legal
    /// geometry. An implementation that bounded the projections instead would be bounding a tensor
    /// a quarter the size of the one that actually crosses.
    #[test]
    fn the_widest_materialized_tensor_is_the_feed_forward_projection() {
        let cfg = MiniMaxH3DitConfig::default();
        for &frames in &LEGAL_FRAME_COUNTS {
            for (w, h) in [(576u32, 320u32), (960, 544), (1344, 768)] {
                let c = DitSequenceCost::new(seq_at(w, h, frames), &cfg).unwrap();
                assert_eq!(c.widest_materialized_elements(), c.ff_proj_elements);
                assert_eq!(c.ff_proj_elements, c.projection_elements * 4);
                assert!(c.ff_proj_elements > c.residual_elements * 5);
            }
        }
    }

    /// **The correctness gate.** On the default 768×1344 canvas the feed-forward projection crosses
    /// `i32::MAX` between 243 and 260 frames — inside the advertised duration envelope — while the
    /// gating 576×320 canvas never gets within a quarter of the bound.
    ///
    /// This is the number sc-17152's `hardMaxDuration` is derived from, so it is pinned as an exact
    /// lattice rung rather than as an inequality.
    #[test]
    fn the_feed_forward_projection_crosses_the_writable_bound_inside_the_advertised_envelope() {
        let cfg = MiniMaxH3DitConfig::default();
        let over = |w, h, f| {
            DitSequenceCost::new(seq_at(w, h, f), &cfg)
                .unwrap()
                .over_writable_bound()
        };

        // The default canvas: 243 frames (10.125 s) is under, 260 frames (10.833 s) is over.
        assert!(!over(1344, 768, 243), "243 frames must be writable");
        assert!(over(1344, 768, 260), "260 frames must cross the bound");
        assert!(over(1344, 768, 345), "the lattice top is well over");

        // ...and the largest writable lattice rung is exactly 243.
        assert_eq!(
            largest_writable_frame_count(48, 84, PATCH_SIZE, TEXT_TOKENS, 0, 2, &cfg).unwrap(),
            Some(243)
        );
        assert!((writable_duration_seconds(243) - 10.125).abs() < 1e-9);

        // The gating canvas stays far inside it at every legal duration.
        for &frames in &LEGAL_FRAME_COUNTS {
            let c = DitSequenceCost::new(seq_at(576, 320, frames), &cfg).unwrap();
            assert!(!c.over_writable_bound(), "{frames} frames at 576x320");
            assert!(
                c.widest_materialized_elements() * 3 < MAX_WRITABLE_ELEMS,
                "{frames} frames at 576x320 is {} elements, closer to the bound than expected",
                c.widest_materialized_elements()
            );
        }
        assert_eq!(
            largest_writable_frame_count(20, 36, PATCH_SIZE, TEXT_TOKENS, 0, 2, &cfg).unwrap(),
            Some(345),
            "every legal duration is writable at the gating canvas"
        );
    }

    /// **A materializing attention would be over the bound at every geometry this model has** —
    /// including the smallest legal render, at 1.3×. Reported, not gated: MLX's fused SDPA streams
    /// the scores (`tests/sequence_cost_real.rs` measures it at this model's shape).
    ///
    /// This is the number handed to sc-18661: bounded attention on this family is a *graph-cut*
    /// lever, not a score-tensor one, because there is no score tensor.
    #[test]
    fn a_materializing_attention_would_be_over_the_bound_at_every_legal_geometry() {
        let cfg = MiniMaxH3DitConfig::default();
        for (w, h) in [(576u32, 320u32), (960, 544), (1344, 768)] {
            for &frames in &LEGAL_FRAME_COUNTS {
                let c = DitSequenceCost::new(seq_at(w, h, frames), &cfg).unwrap();
                assert!(
                    c.dense_score_elements > MAX_WRITABLE_ELEMS,
                    "{w}x{h} / {frames} frames: {} score elements is somehow under the bound",
                    c.dense_score_elements
                );
            }
        }
        // The smallest legal render, and the top of the default canvas — the two ends of the range.
        let smallest = DitSequenceCost::new(seq_at(576, 320, 124), &cfg).unwrap();
        assert_eq!(smallest.dense_score_elements, 56 * 7_138 * 7_138);
        assert!(
            (smallest.dense_score_elements as f64 / MAX_WRITABLE_ELEMS as f64 - 1.33).abs() < 0.01
        );
        let top = DitSequenceCost::new(seq_at(1344, 768, 345), &cfg).unwrap();
        assert!(
            top.dense_score_elements > 600_000_000_000,
            "{} score elements at the top of the envelope",
            top.dense_score_elements
        );
        // 282x the bound — and 1.2 TB at bf16, which is why it cannot be a materialized tensor.
        assert!(top.dense_score_elements as f64 / MAX_WRITABLE_ELEMS as f64 > 280.0);
    }

    /// **The counts are computed in `i64`.** The same arithmetic in `i32` wraps to a small positive
    /// number and reports a wildly over-bound geometry as comfortably inside it — the exact failure
    /// this module exists to prevent.
    #[test]
    fn the_counts_do_not_wrap() {
        let cfg = MiniMaxH3DitConfig::default();
        let c = DitSequenceCost::new(seq_at(1344, 768, 345), &cfg).unwrap();
        assert_eq!(c.ff_proj_elements, 104_030 * 2 * 14_336);
        assert!(c.ff_proj_elements > MAX_WRITABLE_ELEMS);
        // What an i32 product would have said.
        let wrapped = (104_030i32).wrapping_mul(2 * 14_336);
        assert!(
            i64::from(wrapped) < MAX_WRITABLE_ELEMS,
            "the i32 form must be the false-negative this test names"
        );
        assert_ne!(i64::from(wrapped), c.ff_proj_elements);
        // The score count is 6e11 — past i64 is not a concern, but the saturating form is checked.
        assert!(c.dense_score_elements > 0);
    }

    /// Degenerate inputs are errors rather than zero-element costs.
    #[test]
    fn rejects_degenerate_geometry() {
        let cfg = MiniMaxH3DitConfig::default();
        assert!(DitSequenceCost::new(0, &cfg).is_err());
        assert!(DitSequenceCost::new(-1, &cfg).is_err());
        let g = JointGeometry::new(124, 20, 36).unwrap();
        assert!(rows_per_latent_frame(&g, [1, 0, 2]).is_err());
        assert!(packed_seq_len(&g, PATCH_SIZE, 0, 0, 2).is_err());
        assert!(packed_seq_len(&g, PATCH_SIZE, 8, -1, 2).is_err());
        assert!(packed_seq_len(&g, PATCH_SIZE, 8, 0, 0).is_err());
    }

    /// `packed_seq_len` must agree with the layout the render actually builds, at every legal
    /// duration — otherwise the bound is computed on a different sequence than the one that runs.
    /// Swept over **both** conditioning shapes. `t2va` (no anchors) is the path sc-17152 measured;
    /// `fl2va` (sc-17148) prepends a whole frame of conditioning rows per anchor, which is 1008 rows
    /// at the default canvas. A mirror validated only against the empty-anchor case would silently
    /// under-report every keyframe request.
    #[test]
    fn the_pure_sequence_length_agrees_with_the_built_layout() {
        use crate::dit::positions::KeyframeAnchor;
        for &frames in &LEGAL_FRAME_COUNTS {
            let g = JointGeometry::new(frames, 20, 36).unwrap();
            for anchors in [
                &[][..],
                &[KeyframeAnchor::First][..],
                &[KeyframeAnchor::First, KeyframeAnchor::Last][..],
            ] {
                let built = crate::denoise::PackedLayout::build(
                    g,
                    PATCH_SIZE,
                    &vec![crate::denoise::TEXT_TAG; TEXT_TOKENS as usize],
                    2,
                    anchors,
                )
                .unwrap();
                assert_eq!(
                    packed_seq_len(&g, PATCH_SIZE, TEXT_TOKENS, anchors.len() as i32, 2).unwrap(),
                    i64::from(built.seq_len()),
                    "{frames} frames with {} anchors",
                    anchors.len()
                );
            }
        }
        // ...and the anchors really do change the answer, or the sweep above proves nothing.
        let g = JointGeometry::new(345, 48, 84).unwrap();
        let plain = packed_seq_len(&g, PATCH_SIZE, TEXT_TOKENS, 0, 2).unwrap();
        let anchored = packed_seq_len(&g, PATCH_SIZE, TEXT_TOKENS, 2, 2).unwrap();
        assert_eq!(anchored - plain, 2 * 1008, "one frame of rows per anchor");
    }

    /// **`fl2va` keyframes move the `i32::MAX` crossing down a lattice rung** (sc-17148).
    ///
    /// This test was written asserting the opposite and failed, which is the reason it exists in
    /// this form: a keyframe anchor prepends **one whole frame** of conditioning rows — 1008 at the
    /// default canvas — and `t2va`'s largest writable rung, 243 frames, sits at 0.981× the bound.
    /// Two anchors add 2016 rows = 57.8 M feed-forward elements and carry it to **1.008×**. So:
    ///
    /// | anchors | largest writable rung | margin | first rung over |
    /// |---|---|---|---|
    /// | 0 (`t2va`) | 243 frames / 10.125 s | 0.981× | 260 / 10.833 s |
    /// | 1 | 243 frames / 10.125 s | 0.994× | 260 / 10.833 s |
    /// | 2 (first **and** last) | **226 frames / 9.4167 s** | 0.940× | 243 / 10.125 s |
    ///
    /// It is a characterization, not a gate — `tests/sequence_cost_real.rs` measured MLX's `matmul`
    /// as exact on both sides of the bound, so crossing costs memory rather than correctness. It is
    /// pinned because the margin is thin enough that the *next* conditioning block to arrive
    /// (sc-17149's omni-reference) can move it again, and because a bound derived on `t2va` alone
    /// would be quoted 0.7 s too long for every two-keyframe request.
    #[test]
    fn keyframe_anchors_move_the_writable_crossing_down_one_rung() {
        let cfg = MiniMaxH3DitConfig::default();
        let largest = |anchors| {
            largest_writable_frame_count(48, 84, PATCH_SIZE, TEXT_TOKENS, anchors, 2, &cfg).unwrap()
        };
        assert_eq!(largest(0), Some(243), "t2va");
        assert_eq!(largest(1), Some(243), "one anchor still fits");
        assert_eq!(largest(2), Some(226), "two anchors drop a rung");
        assert!((writable_duration_seconds(226) - 9.41667).abs() < 1e-4);

        // The margins, pinned as numbers — "comfortably inside" is what made the first version of
        // this test wrong.
        let ratio = |frames, anchors| {
            let g = JointGeometry::new(frames, 48, 84).unwrap();
            DitSequenceCost::new(
                packed_seq_len(&g, PATCH_SIZE, TEXT_TOKENS, anchors, 2).unwrap(),
                &cfg,
            )
            .unwrap()
            .widest_materialized_elements() as f64
                / MAX_WRITABLE_ELEMS as f64
        };
        assert!((ratio(243, 0) - 0.981).abs() < 0.002, "{}", ratio(243, 0));
        assert!((ratio(243, 1) - 0.994).abs() < 0.002, "{}", ratio(243, 1));
        assert!((ratio(243, 2) - 1.008).abs() < 0.002, "{}", ratio(243, 2));
        assert!((ratio(226, 2) - 0.940).abs() < 0.002, "{}", ratio(226, 2));

        // One anchor is one frame of rows, at every canvas — the quantity that drives all of it.
        let g = JointGeometry::new(243, 48, 84).unwrap();
        assert_eq!(rows_per_latent_frame(&g, PATCH_SIZE).unwrap(), 1008);
        assert_eq!(
            packed_seq_len(&g, PATCH_SIZE, TEXT_TOKENS, 1, 2).unwrap()
                - packed_seq_len(&g, PATCH_SIZE, TEXT_TOKENS, 0, 2).unwrap(),
            1008
        );
    }
}
