//! Temporal chunk bookkeeping for the video VAE decode (`klvae.py::decode_temporal`).
//!
//! This is the part of the port with the most room to be subtly wrong — the transformer blocks
//! themselves are conventional. It is deliberately kept as **pure integer arithmetic** with no
//! tensors, so the whole plan can be asserted directly.
//!
//! ## Why the decode is chunked at all
//!
//! An encode emits `ceil(clip_length / vae_ratio_t)` latent tokens per clip and then drops
//! `token_drop` tokens off the tail (`klvae.py:493-494` — this happens at *inference*, it is not a
//! training-time augmentation). Decode therefore has to reconstruct those dropped frames from the
//! neighbouring clips, which it does by decoding each token chunk together with a `token_overlap`
//! of the *next* chunk's tokens and blending the overlapping frames:
//!
//! ```text
//! clip_length 17, token_drop 3, vae_ratio_t 4
//!   frame_pre_padding = (-17) mod 4  = 3     frames trimmed off the FRONT of every split
//!   tokens_chunk_size = ceil(17 / 4) = 5     tokens advanced per chunk
//!   token_overlap     = (-3)  mod 5  = 2     extra tokens read past the chunk
//!   frame_overlap     = max(2·4 - 3, 0) = 5  frames cross-faded at the seam
//!   split_count       = 1 + [token_drop > 0] = 2
//! ```
//!
//! Each chunk reads `5 + 2 = 7` tokens → 28 decoded frames, split into `chunk_dec = 5·4 = 20`
//! frame groups: split 0 yields `20 - 3 = 17` frames (exactly `clip_length`) that are emitted, and
//! split 1 yields `8 - 3 = 5` frames (exactly `frame_overlap`) that are held back and cross-faded
//! into the *next* chunk's split-0 output. With `token_drop = 0` the second split disappears
//! entirely and chunks simply abut — the two-pass alignment path.
//!
//! The reference supports `isolated_first_frame` / `isolated_last_frame` / `isolated_key_frame`
//! (its `z_head`/`z_tail` limbs). The shipped `MiniMax-H3` checkpoint enables none of them —
//! `MiniMaxH3VideoVAE.from_pretrained` passes only `clip_length` and `token_drop`, so all three
//! keep their `False` defaults — hence [`TemporalPlan`] models the no-isolate path and
//! [`TemporalGeometry::isolated_frames_unsupported`] documents the deliberate boundary.

use mlx_gen::{Error, Result};

use crate::config::MiniMaxH3VaeConfig;

/// `(-a) mod m` with Python's sign convention (Rust's `%` would return a negative here).
fn neg_mod(a: i32, m: i32) -> i32 {
    (-a).rem_euclid(m)
}

/// The token/frame constants derived once from `clip_length`, `token_drop` and `vae_ratio_t`
/// (`klvae.py::setup_forward`, lines 100-106).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct TemporalGeometry {
    /// Frames per clip.
    pub clip_length: i32,
    /// Latent tokens dropped at the encode tail.
    pub token_drop: i32,
    /// Temporal compression factor.
    pub vae_ratio_t: i32,
    /// `token_drop · vae_ratio_t` — frames the drop corresponds to.
    pub frame_drop: i32,
    /// `(-clip_length) mod vae_ratio_t` — frames trimmed off the front of every decoded split.
    pub frame_pre_padding: i32,
    /// `ceil(clip_length / vae_ratio_t)` — tokens advanced per chunk.
    pub tokens_chunk_size: i32,
    /// `(-token_drop) mod tokens_chunk_size` — extra tokens each chunk reads past its own span.
    pub token_overlap: i32,
    /// `max(token_overlap · vae_ratio_t - frame_pre_padding, 0)` — cross-faded seam length.
    pub frame_overlap: i32,
}

impl TemporalGeometry {
    /// Derive the geometry from a config.
    pub fn new(cfg: &MiniMaxH3VaeConfig) -> Result<Self> {
        Self::from_parts(cfg.clip_length, cfg.token_drop, cfg.patch_size_t)
    }

    /// Derive the geometry from the three raw knobs.
    pub fn from_parts(clip_length: i32, token_drop: i32, vae_ratio_t: i32) -> Result<Self> {
        if clip_length <= 0 || vae_ratio_t <= 0 {
            return Err(Error::Msg(format!(
                "minimax-h3 temporal geometry: clip_length {clip_length} and vae_ratio_t \
                 {vae_ratio_t} must be positive"
            )));
        }
        if token_drop < 0 {
            return Err(Error::Msg(format!(
                "minimax-h3 temporal geometry: token_drop {token_drop} must be >= 0"
            )));
        }
        // `ceil(clip_length / vae_ratio_t)`; both are positive here, so the +denominator-1 form is
        // exact (`i32::div_ceil` is not stable on this toolchain).
        let tokens_chunk_size = (clip_length + vae_ratio_t - 1) / vae_ratio_t;
        let frame_pre_padding = neg_mod(clip_length, vae_ratio_t);
        let token_overlap = neg_mod(token_drop, tokens_chunk_size);
        Ok(Self {
            clip_length,
            token_drop,
            vae_ratio_t,
            frame_drop: token_drop * vae_ratio_t,
            frame_pre_padding,
            tokens_chunk_size,
            token_overlap,
            frame_overlap: (token_overlap * vae_ratio_t - frame_pre_padding).max(0),
        })
    }

    /// Decoded frames per chunk split — `tokens_chunk_size · vae_ratio_t`.
    pub fn chunk_dec(&self) -> i32 {
        self.tokens_chunk_size * self.vae_ratio_t
    }

    /// `1 + [token_drop > 0]` — the second split exists only to produce the blended seam.
    pub fn split_count(&self) -> i32 {
        1 + i32::from(self.token_drop > 0)
    }

    /// Fewest temporal tokens that yield at least one decodable chunk.
    ///
    /// With `token_drop > 0` the plan discards one chunk's worth of grid up front (`num_chunks =
    /// pseudo_num_chunks - 1`), so a latent shorter than `tokens_chunk_size - token_drop + 1`
    /// leaves nothing to decode — the reference hits its own "produced no output tensor" error
    /// there. For the shipped config that floor is 3 tokens.
    pub fn min_decodable_tokens(&self) -> i32 {
        if self.token_drop > 0 {
            (self.tokens_chunk_size - self.token_drop + 1).max(1)
        } else {
            1
        }
    }

    /// The reference's `isolated_first_frame` / `isolated_last_frame` / `isolated_key_frame` limbs
    /// are not modelled: the shipped checkpoint leaves all three `False`. Kept as a named
    /// constant so the boundary is greppable rather than implicit.
    pub const fn isolated_frames_unsupported() -> bool {
        true
    }
}

/// One chunk's token span, half-open over the padded latent.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ChunkSpan {
    /// First token index (inclusive).
    pub start: i32,
    /// Last token index (exclusive).
    pub end: i32,
}

impl ChunkSpan {
    /// Tokens in the span.
    pub fn tokens(&self) -> i32 {
        self.end - self.start
    }
}

/// The full decode plan for a latent with `n_tokens` temporal tokens.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TemporalPlan {
    /// Derived constants.
    pub geometry: TemporalGeometry,
    /// Temporal tokens the caller supplied.
    pub input_tokens: i32,
    /// Tokens appended by repeating the last one so the chunk grid divides evenly.
    pub pad_tokens: i32,
    /// `input_tokens + pad_tokens`.
    pub padded_tokens: i32,
    /// Chunk spans over the padded latent.
    pub chunks: Vec<ChunkSpan>,
    /// Frames produced before the padding trim.
    pub total_frames: i32,
    /// Frames trimmed off the tail to undo `pad_tokens`.
    pub pad_frames: i32,
    /// Frames actually returned — `total_frames - pad_frames`.
    pub output_frames: i32,
}

impl TemporalPlan {
    /// Build the plan for `n_tokens` temporal latent tokens.
    pub fn new(geometry: TemporalGeometry, n_tokens: i32) -> Result<Self> {
        if n_tokens <= 0 {
            return Err(Error::Msg(format!(
                "minimax-h3 decode: need at least one temporal latent token, got {n_tokens}"
            )));
        }
        let chunk = geometry.tokens_chunk_size;

        // `pseudo_total_tokens` counts the tokens the encode *would* have had before the drop, so
        // the chunk grid is aligned to the encoder's clips rather than to what survived.
        let mut pseudo_total = n_tokens + geometry.token_drop;
        let remainder = pseudo_total.rem_euclid(chunk);
        let pad_tokens = if remainder == 0 { 0 } else { chunk - remainder };
        pseudo_total += pad_tokens;

        let pseudo_num_chunks = pseudo_total / chunk;
        let num_chunks = pseudo_num_chunks - i32::from(geometry.token_drop > 0);
        if num_chunks <= 0 {
            return Err(Error::Msg(format!(
                "minimax-h3 decode: {n_tokens} temporal tokens yield no decodable chunk \
                 (tokens_chunk_size {chunk}, token_drop {})",
                geometry.token_drop
            )));
        }
        let padded_tokens = n_tokens + pad_tokens;

        let chunks: Vec<ChunkSpan> = (0..num_chunks)
            .map(|i| {
                let start = i * chunk;
                ChunkSpan {
                    start,
                    end: (start + chunk + geometry.token_overlap).min(padded_tokens),
                }
            })
            .collect();

        // Frame accounting mirrors `_decode_temporal_output_frame_plan`: every split-0 part is
        // emitted, and only the LAST split-1 part survives (each earlier one is consumed by the
        // blend into the following chunk, which contributes no frames of its own).
        let chunk_dec = geometry.chunk_dec();
        let mut total_frames = 0;
        let mut final_overlap_frames = 0;
        for span in &chunks {
            let clip_frames = span.tokens() * geometry.vae_ratio_t;
            for j in 0..geometry.split_count() {
                let f_start = j * chunk_dec;
                let f_end = (f_start + chunk_dec).min(clip_frames);
                let part = (f_end - f_start - geometry.frame_pre_padding).max(0);
                if j == 0 {
                    total_frames += part;
                } else {
                    final_overlap_frames = part;
                }
            }
        }
        total_frames += final_overlap_frames;

        let pad_frames = Self::pad_frames(geometry, padded_tokens, pad_tokens);
        let output_frames = total_frames - pad_frames;
        if output_frames <= 0 {
            return Err(Error::Msg(format!(
                "minimax-h3 decode: planned non-positive output_frames {output_frames} \
                 (total {total_frames}, pad {pad_frames})"
            )));
        }
        Ok(Self {
            geometry,
            input_tokens: n_tokens,
            pad_tokens,
            padded_tokens,
            chunks,
            total_frames,
            pad_frames,
            output_frames,
        })
    }

    /// `_decode_temporal_pad_frames`. Each padding token normally expands to `vae_ratio_t` frames,
    /// but a token that lands exactly on a chunk boundary only contributes
    /// `clip_length mod vae_ratio_t` (the clip's ragged tail), so the trim is not simply
    /// `pad_tokens · vae_ratio_t`.
    fn pad_frames(geometry: TemporalGeometry, padded_tokens: i32, pad_tokens: i32) -> i32 {
        if pad_tokens <= 0 {
            return 0;
        }
        let intra_tail = geometry.clip_length.rem_euclid(geometry.vae_ratio_t);
        if intra_tail == 0 {
            return pad_tokens * geometry.vae_ratio_t;
        }
        let before_pad = padded_tokens - pad_tokens;
        (0..pad_tokens)
            .map(|k| {
                if (before_pad + k).rem_euclid(geometry.tokens_chunk_size) == 0 {
                    intra_tail
                } else {
                    geometry.vae_ratio_t
                }
            })
            .sum()
    }

    /// The frame span `[start, end)` a chunk's split `j` contributes, after the
    /// `frame_pre_padding` front trim, given how many frames the chunk decoded to.
    pub fn split_span(&self, clip_frames: i32, j: i32) -> (i32, i32) {
        let chunk_dec = self.geometry.chunk_dec();
        let f_start = j * chunk_dec;
        let f_end = (f_start + chunk_dec).min(clip_frames);
        (
            (f_start + self.geometry.frame_pre_padding).min(f_end),
            f_end,
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn shipped() -> TemporalGeometry {
        TemporalGeometry::from_parts(17, 3, 4).unwrap()
    }

    /// The five derived constants for the SHIPPED config, checked against the reference's own
    /// values (printed by `tools/dump_minimax_h3_video_vae.py`, which instantiates the reference).
    #[test]
    fn shipped_geometry_matches_the_reference() {
        let g = shipped();
        assert_eq!(g.tokens_chunk_size, 5);
        assert_eq!(g.token_overlap, 2);
        assert_eq!(g.frame_pre_padding, 3);
        assert_eq!(g.frame_overlap, 5);
        assert_eq!(g.frame_drop, 12);
        assert_eq!(g.chunk_dec(), 20);
        assert_eq!(g.split_count(), 2);
        // Split 0 emits exactly one clip; split 1 emits exactly the seam.
        assert_eq!(g.chunk_dec() - g.frame_pre_padding, g.clip_length);
    }

    /// `token_drop = 0` collapses the overlap entirely — the two-pass alignment path.
    #[test]
    fn zero_token_drop_removes_the_seam() {
        let g = TemporalGeometry::from_parts(17, 0, 4).unwrap();
        assert_eq!(g.tokens_chunk_size, 5);
        assert_eq!(g.token_overlap, 0);
        assert_eq!(g.frame_overlap, 0);
        assert_eq!(g.split_count(), 1);
        assert_eq!(g.frame_drop, 0);
        // frame_pre_padding is a property of clip_length alone and does NOT vanish.
        assert_eq!(g.frame_pre_padding, 3);
    }

    /// Python's `(-a) % m` is non-negative; Rust's `%` is not. A `-3 % 5 == -3` here would give
    /// `token_overlap = -3`, silently producing SHORTER chunks and a wrong frame count.
    #[test]
    fn negative_modulo_follows_python() {
        assert_eq!(neg_mod(3, 5), 2);
        assert_eq!(neg_mod(17, 4), 3);
        assert_eq!(neg_mod(0, 5), 0);
        assert_eq!(neg_mod(5, 5), 0);
        assert_ne!(neg_mod(3, 5), -3);
    }

    /// Frame counts for token counts that straddle the `clip_length`-17 chunk boundary, each
    /// verified against the reference implementation's actual decode output shape (recorded by
    /// the fixture dump: 5→17, 7→22, 9→30, 12→39, 17→56 frames).
    #[test]
    fn output_frame_counts_match_the_reference_decode() {
        let g = shipped();
        for (tokens, frames) in [(5, 17), (7, 22), (9, 30), (12, 39), (17, 56)] {
            let plan = TemporalPlan::new(g, tokens).unwrap();
            assert_eq!(
                plan.output_frames, frames,
                "token_drop=3, {tokens} tokens should decode to {frames} frames"
            );
        }
    }

    #[test]
    fn output_frame_counts_match_the_reference_with_no_token_drop() {
        let g = TemporalGeometry::from_parts(17, 0, 4).unwrap();
        for (tokens, frames) in [(5, 17), (10, 34)] {
            let plan = TemporalPlan::new(g, tokens).unwrap();
            assert_eq!(plan.output_frames, frames);
            assert_eq!(plan.pad_tokens, 0);
        }
    }

    /// The padded cases: 5 and 9 tokens are NOT chunk-aligned and pull in repeat padding, whose
    /// frames are trimmed back off with the ragged-tail rule.
    #[test]
    fn padding_uses_the_ragged_tail_rule_not_a_flat_multiple() {
        let g = shipped();
        let plan = TemporalPlan::new(g, 5).unwrap();
        assert_eq!(plan.pad_tokens, 2);
        assert_eq!(plan.padded_tokens, 7);
        // A flat `pad_tokens · vae_ratio_t` would be 8; the boundary token contributes only
        // `17 mod 4 = 1`, so the real trim is 1 + 4 = 5.
        assert_eq!(plan.pad_frames, 5);
        assert_ne!(plan.pad_frames, plan.pad_tokens * g.vae_ratio_t);
        assert_eq!(plan.total_frames, 22);
        assert_eq!(plan.output_frames, 17);

        let plan = TemporalPlan::new(g, 9).unwrap();
        assert_eq!(plan.pad_tokens, 3);
        // 9%5=4 -> 4, 10%5=0 -> 1, 11%5=1 -> 4  ==> 9, not 3·4 = 12.
        assert_eq!(plan.pad_frames, 9);
        assert_eq!(plan.output_frames, 30);
    }

    /// Every chunk must read a FULL `tokens_chunk_size + token_overlap` window. If the tail chunk
    /// were short the second split would be empty and the seam would silently vanish.
    #[test]
    fn every_chunk_reads_a_full_window() {
        let g = shipped();
        for tokens in g.min_decodable_tokens()..60 {
            let plan = TemporalPlan::new(g, tokens).unwrap();
            for span in &plan.chunks {
                assert_eq!(
                    span.tokens(),
                    g.tokens_chunk_size + g.token_overlap,
                    "{tokens} tokens: chunk {span:?} is short"
                );
                assert!(span.end <= plan.padded_tokens);
            }
            // Chunks advance by exactly one chunk size and cover the padded latent.
            for pair in plan.chunks.windows(2) {
                assert_eq!(pair[1].start - pair[0].start, g.tokens_chunk_size);
            }
            assert_eq!(plan.chunks.last().unwrap().end, plan.padded_tokens);
        }
    }

    /// The plan's own frame accounting must agree with walking the splits.
    #[test]
    fn split_spans_sum_to_the_planned_frame_total() {
        let g = shipped();
        for tokens in g.min_decodable_tokens()..40 {
            let plan = TemporalPlan::new(g, tokens).unwrap();
            let mut emitted = 0;
            let mut overlap = 0;
            for span in &plan.chunks {
                let clip_frames = span.tokens() * g.vae_ratio_t;
                for j in 0..g.split_count() {
                    let (s, e) = plan.split_span(clip_frames, j);
                    if j == 0 {
                        emitted += e - s;
                    } else {
                        overlap = e - s;
                    }
                }
            }
            assert_eq!(emitted + overlap, plan.total_frames, "{tokens} tokens");
            // Split 0 always yields exactly one clip's worth; split 1 exactly the seam.
            assert_eq!(emitted, plan.chunks.len() as i32 * g.clip_length);
            assert_eq!(overlap, g.frame_overlap);
        }
    }

    /// Output length must be monotonically non-decreasing in the token count — an off-by-one in
    /// the pad arithmetic typically shows up as a dip.
    #[test]
    fn output_frames_are_monotonic_in_token_count() {
        let g = shipped();
        let mut prev = 0;
        for tokens in g.min_decodable_tokens()..60 {
            let plan = TemporalPlan::new(g, tokens).unwrap();
            assert!(
                plan.output_frames >= prev,
                "{tokens} tokens: {} frames dipped below {prev}",
                plan.output_frames
            );
            prev = plan.output_frames;
        }
    }

    /// A `token_drop` that is a MULTIPLE of `tokens_chunk_size` gives `token_overlap = 0` while
    /// still asking for two splits — so split 1 has no frames left. The plan must count it as
    /// zero rather than negative, and `MiniMaxH3VideoVae::decode_temporal` skips it to match.
    /// The shipped config (drop 3, overlap 2) never reaches this, but a caller-supplied
    /// `token_drop` can.
    #[test]
    fn a_token_drop_that_zeroes_the_overlap_yields_an_empty_second_split() {
        let g = TemporalGeometry::from_parts(17, 5, 4).unwrap();
        assert_eq!(g.token_overlap, 0);
        assert_eq!(g.frame_overlap, 0);
        assert_eq!(
            g.split_count(),
            2,
            "token_drop > 0 still asks for two splits"
        );

        let plan = TemporalPlan::new(g, 7).unwrap();
        for span in &plan.chunks {
            let clip_frames = span.tokens() * g.vae_ratio_t;
            let (s0, e0) = plan.split_span(clip_frames, 0);
            assert_eq!(e0 - s0, g.clip_length, "split 0 still emits a full clip");
            let (s1, e1) = plan.split_span(clip_frames, 1);
            assert!(s1 >= e1, "split 1 must be empty when there is no overlap");
        }
        assert_eq!(
            plan.total_frames,
            plan.chunks.len() as i32 * g.clip_length,
            "no seam frames are produced"
        );
    }

    #[test]
    fn rejects_degenerate_inputs() {
        let g = shipped();
        assert!(TemporalPlan::new(g, 0).is_err());
        assert!(TemporalPlan::new(g, -1).is_err());
        assert!(TemporalGeometry::from_parts(0, 3, 4).is_err());
        assert!(TemporalGeometry::from_parts(17, -1, 4).is_err());
        assert!(TemporalGeometry::from_parts(17, 3, 0).is_err());
    }

    #[test]
    fn geometry_from_config_matches_the_raw_knobs() {
        let cfg = MiniMaxH3VaeConfig::default();
        assert_eq!(TemporalGeometry::new(&cfg).unwrap(), shipped());
        assert!(TemporalGeometry::isolated_frames_unsupported());
    }
}
