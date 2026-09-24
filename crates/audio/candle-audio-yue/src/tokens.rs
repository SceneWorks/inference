//! The YuE token-space facts shared by every stage: special-token ids, the xcodec codebook
//! offsets inside the mm vocabulary, and the stage-1 codebook-0 de-interleave.
//!
//! These are pipeline facts (epic sc-19373), not stage internals: the tokenizer (sc-19376) emits
//! them, stage 1 (sc-19380) samples inside them, stage 2 (sc-19381) slices its logits by them, and
//! the engine de-interleaves stage 1's output with [`split_stage1_tokens`].

/// Start-of-audio marker (`<SOA>`).
pub const SOA: u32 = 32_001;
/// End-of-audio marker (`<EOA>`) — the stage-1 segment terminator.
pub const EOA: u32 = 32_002;
/// `<stage_1>` marker.
pub const STAGE_1: u32 = 32_013;
/// The `<xcodec>` separator that follows `<SOA>` in every audio block.
pub const XCODEC_SEP: u32 = 32_016;
/// `<stage_2>` marker.
pub const STAGE_2: u32 = 32_017;

/// First xcodec token id in the mm vocabulary (codebook 0, code 0).
pub const CODEC_OFFSET: u32 = 45_334;
/// Codes per xcodec codebook.
pub const CODEBOOK_SIZE: u32 = 1_024;
/// Codebooks the xcodec decoder consumes (stage 2 upsamples codebook 0 to all of them).
pub const NUM_CODEBOOKS: usize = 8;
/// Codec frames per second per track (xcodec at 16 kHz with a 320-sample hop).
pub const FRAMES_PER_SECOND: u32 = 50;

/// Last token id stage 1's sampling allow-list admits besides [`EOA`] (the upper bound of
/// `[CODEC_OFFSET, STAGE1_ALLOW_MAX]`).
pub const STAGE1_ALLOW_MAX: u32 = 56_721;
/// Stage 2's residual-codebook logit slice `[STAGE2_SLICE_MIN, STAGE2_SLICE_MAX]` — codebooks 1..=7.
pub const STAGE2_SLICE_MIN: u32 = CODEC_OFFSET + CODEBOOK_SIZE;
/// See [`STAGE2_SLICE_MIN`].
pub const STAGE2_SLICE_MAX: u32 = CODEC_OFFSET + CODEBOOK_SIZE * NUM_CODEBOOKS as u32 - 1;

/// The mm-vocabulary id of `code` in xcodec codebook `codebook`.
pub fn codec_token(codebook: usize, code: u32) -> u32 {
    CODEC_OFFSET + codebook as u32 * CODEBOOK_SIZE + code
}

/// The codebook-0 code a stage-1 token carries, or `None` when the token is not a codebook-0 id.
pub fn cb0_code(token: u32) -> Option<u32> {
    (CODEC_OFFSET..CODEC_OFFSET + CODEBOOK_SIZE)
        .contains(&token)
        .then(|| token - CODEC_OFFSET)
}

/// Codebook-0 codes for the two tracks stage 1 interleaves.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct TrackCodes {
    /// Vocal-track codebook-0 codes (`0..CODEBOOK_SIZE`), one per 20 ms frame.
    pub vocals: Vec<u32>,
    /// Instrumental-track codebook-0 codes, frame-aligned with [`vocals`](Self::vocals).
    pub instrumental: Vec<u32>,
}

/// One track's full xcodec code grid — stage 2's output and the codec decoder's input.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct CodecFrames {
    /// [`NUM_CODEBOOKS`] rows of `0..CODEBOOK_SIZE` codes, all the same length (frames).
    pub codebooks: Vec<Vec<u32>>,
}

impl CodecFrames {
    /// Frames per codebook (0 for an empty grid).
    pub fn frames(&self) -> usize {
        self.codebooks.first().map_or(0, Vec::len)
    }

    /// Check the grid shape: [`NUM_CODEBOOKS`] equal-length rows of in-range codes whose row 0 is
    /// `cb0` (stage 2 is teacher-forced on codebook 0, so it must not rewrite it).
    pub fn check_against(&self, cb0: &[u32]) -> Result<(), String> {
        if self.codebooks.len() != NUM_CODEBOOKS {
            return Err(format!(
                "expected {NUM_CODEBOOKS} codebooks, got {}",
                self.codebooks.len()
            ));
        }
        if self.codebooks.iter().any(|row| row.len() != cb0.len()) {
            return Err(format!(
                "every codebook must hold {} frames (the codebook-0 length)",
                cb0.len()
            ));
        }
        if self.codebooks[0] != cb0 {
            return Err("codebook 0 differs from the teacher-forced input".into());
        }
        if self.codebooks.iter().flatten().any(|&c| c >= CODEBOOK_SIZE) {
            return Err(format!("a code is outside 0..{CODEBOOK_SIZE}"));
        }
        Ok(())
    }
}

/// De-interleave stage 1's per-segment token streams into the two codebook-0 tracks.
///
/// Stage 1 emits `vocal, instrumental, vocal, instrumental, …` codebook-0 ids per frame. Each
/// segment is trimmed to a whole number of frames (an odd trailing token is a half frame) before
/// concatenation, as the reference does. A token outside codebook 0 is a stage-1 allow-list defect
/// and is refused rather than wrapped into range.
pub fn split_stage1_tokens(segments: &[Vec<u32>]) -> Result<TrackCodes, String> {
    let mut out = TrackCodes::default();
    for (index, tokens) in segments.iter().enumerate() {
        let whole = tokens.len() - tokens.len() % 2;
        for pair in tokens[..whole].chunks_exact(2) {
            let code = |t: u32| {
                cb0_code(t).ok_or_else(|| {
                    format!(
                        "stage-1 segment {index} emitted token {t}, which is not a codebook-0 id"
                    )
                })
            };
            out.vocals.push(code(pair[0])?);
            out.instrumental.push(code(pair[1])?);
        }
    }
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn stage2_slice_spans_codebooks_one_through_seven() {
        assert_eq!(STAGE2_SLICE_MIN, 46_358);
        assert_eq!(STAGE2_SLICE_MAX, 53_525);
        assert_eq!(codec_token(1, 0), STAGE2_SLICE_MIN);
        assert_eq!(codec_token(7, CODEBOOK_SIZE - 1), STAGE2_SLICE_MAX);
    }

    #[test]
    fn split_deinterleaves_and_trims_half_frames() {
        let seg0 = vec![codec_token(0, 1), codec_token(0, 2), codec_token(0, 3)];
        let seg1 = vec![codec_token(0, 4), codec_token(0, 5)];
        let codes = split_stage1_tokens(&[seg0, seg1]).unwrap();
        assert_eq!(codes.vocals, [1, 4]);
        assert_eq!(codes.instrumental, [2, 5]);
    }

    #[test]
    fn split_refuses_tokens_outside_codebook_zero() {
        let err = split_stage1_tokens(&[vec![codec_token(0, 1), codec_token(1, 1)]]).unwrap_err();
        assert!(err.contains("not a codebook-0 id"), "{err}");
        assert!(split_stage1_tokens(&[vec![EOA, codec_token(0, 1)]]).is_err());
    }
}
