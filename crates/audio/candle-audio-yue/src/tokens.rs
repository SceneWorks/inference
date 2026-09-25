//! The YuE token-space facts shared by every stage: special-token ids, the xcodec codebook
//! offsets inside the mm vocabulary, and the stage-1 codebook-0 de-interleave.
//!
//! These are pipeline facts (epic sc-19373), not stage internals: the tokenizer (sc-19376) emits
//! them, stage 1 (sc-19380) samples inside them, stage 2 (sc-19381) slices its logits by them, and
//! the engine de-interleaves stage 1's output with [`split_raw_output`].

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
    /// Vocal-track codes, one per 20 ms frame: the stage-1 token minus [`CODEC_OFFSET`] (the
    /// reference `ids2npy`). A token the stage-1 allow-list admits beyond codebook 0 yields a code
    /// `>= CODEBOOK_SIZE`, as in the reference; the engine refuses such a code before stage 2, where
    /// the reference's `offset_tok_ids` asserts on it.
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

/// Split a render's stage-1 sequence (every prompt block, generated token and `<EOA>`, in order)
/// into the two codebook-0 tracks, with the reference `Stage1Pipeline.save` semantics:
///
/// * `<SOA>` and `<EOA>` must occur equally often; the `i`-th `<SOA>` pairs with the `i`-th `<EOA>`;
/// * the first `skip_pairs` pairs are prompt audio (an ICL reference block), not generated audio;
/// * each remaining pair's body drops a leading `<xcodec>` separator, is truncated to an even
///   length, and is de-interleaved `vocal, instrumental, vocal, …`;
/// * a track's code is `token - CODEC_OFFSET`. A track whose first token is not a codebook-0 id, or
///   that holds a token below [`CODEC_OFFSET`], is refused (the reference asserts on both).
///
/// One deliberate difference: a pair whose body is empty after the separator drop and the even
/// truncation contributes no frames, where the reference raises an `IndexError`.
pub fn split_raw_output(raw: &[u32], skip_pairs: usize) -> Result<TrackCodes, String> {
    let soa: Vec<usize> = (0..raw.len()).filter(|&i| raw[i] == SOA).collect();
    let eoa: Vec<usize> = (0..raw.len()).filter(|&i| raw[i] == EOA).collect();
    if soa.len() != eoa.len() {
        return Err(format!(
            "invalid pairs of soa and eoa, Num of soa: {}, Num of eoa: {}",
            soa.len(),
            eoa.len()
        ));
    }
    let mut out = TrackCodes::default();
    for (pair, (&s, &e)) in soa.iter().zip(&eoa).enumerate().skip(skip_pairs) {
        if e <= s {
            return Err(format!(
                "audio pair {pair}: <EOA> at {e} does not follow its <SOA> at {s}"
            ));
        }
        let mut body = &raw[s + 1..e];
        if body.first() == Some(&XCODEC_SEP) {
            body = &body[1..];
        }
        let body = &body[..body.len() - body.len() % 2];
        if body.is_empty() {
            continue;
        }
        for (track, dest) in [(0, &mut out.vocals), (1, &mut out.instrumental)] {
            let first = body[track];
            if cb0_code(first).is_none() {
                return Err(format!(
                    "audio pair {pair}: the {} track starts with token {first}, which is not a \
                     codebook-0 id",
                    if track == 0 { "vocal" } else { "instrumental" }
                ));
            }
            for &t in body[track..].iter().step_by(2) {
                if t < CODEC_OFFSET {
                    return Err(format!(
                        "audio pair {pair}: token {t} is below the codec range"
                    ));
                }
                dest.push(t - CODEC_OFFSET);
            }
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
    fn split_pairs_soa_eoa_drops_the_separator_and_trims_half_frames() {
        let c = |code| codec_token(0, code);
        let raw = [
            7,
            SOA,
            XCODEC_SEP,
            c(1),
            c(2),
            c(3),
            EOA,
            9,
            SOA,
            XCODEC_SEP,
            c(4),
            c(5),
            EOA,
        ];
        let codes = split_raw_output(&raw, 0).unwrap();
        assert_eq!(codes.vocals, [1, 4]);
        assert_eq!(codes.instrumental, [2, 5]);
        // A skipped leading pair is prompt audio (an ICL reference block).
        let codes = split_raw_output(&raw, 1).unwrap();
        assert_eq!(codes.vocals, [4]);
        assert_eq!(codes.instrumental, [5]);
    }

    #[test]
    fn split_refuses_unpaired_markers_and_non_codebook0_track_starts() {
        let c = |code| codec_token(0, code);
        let err = split_raw_output(&[SOA, c(1), c(2)], 0).unwrap_err();
        assert!(err.contains("invalid pairs"), "{err}");
        let err = split_raw_output(&[SOA, codec_token(1, 1), c(1), EOA], 0).unwrap_err();
        assert!(err.contains("not a codebook-0 id"), "{err}");
        // A mid-track token beyond codebook 0 keeps the reference's raw offset.
        let raw = [SOA, c(1), c(2), codec_token(3, 5), c(3), EOA];
        let codes = split_raw_output(&raw, 0).unwrap();
        assert_eq!(codes.vocals, [1, 3 * CODEBOOK_SIZE + 5]);
        assert_eq!(codes.instrumental, [2, 3]);
    }
}
