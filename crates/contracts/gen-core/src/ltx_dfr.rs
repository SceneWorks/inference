//! LTX-2.5 **DFR** (Diffusion Fidelity Rendering) shared geometry + step coefficients (sc-18789).
//!
//! Port of the backend-neutral half of the reference `ltx_pipelines/dfr_layout.py` +
//! `ltx_core/components/diffusion_steps.py::EulerAncestralDiffusionStep` (Lightricks/LTX-2 v1.2.0,
//! `d1511477`): the keyframe segment grid, temporal tile ranges with their lead-in/handover
//! arithmetic, the generated-keyframe position resolvers, and the **rectified-flow** ancestral
//! Euler step coefficients. Everything here is pure integer/scalar math so both tensor backends
//! (`mlx-gen-ltx`, `candle-gen-ltx`) consume one implementation — the tensor halves (latent
//! stitching, slot token appends, the denoise loops) live in the backend crates.
//!
//! **The RF ancestral step is deliberately not [`crate::sampling::solvers::EulerAncestral`].**
//! That curated solver is the k-diffusion *variance-exploding* ancestral parameterization; LTX-2 is
//! rectified flow (`alpha = 1 - sigma`) and its reference stepper documents that the two "agree
//! only at `eta = 0`" — a different `sigma_down` and a different amount of injected noise for the
//! same `eta`. Reusing the curated solver here would be an aliasing bug of exactly the shape the
//! story brief warns about.

use crate::error::Error;

type Result<T> = std::result::Result<T, Error>;

/// Keyframe segment-length candidates, in pixel frames (`dfr_layout.SEGMENT_CANDIDATES`).
pub const SEGMENT_CANDIDATES: [i64; 2] = [24, 32];

/// Lead-in for non-first temporal tiles, in canvas segments (`dfr_layout.TILE_LEAD_SEGMENTS`). A
/// tile's local latent 0 is an *image* latent (one pixel frame) and its local latent 1 was denoised
/// against it; one segment of lead-in puts both inside the discarded prefix while keeping the
/// window start on a keyframe anchor.
pub const TILE_LEAD_SEGMENTS: usize = 1;

/// Strength for anchor keyframes carried between temporal rounds — pinned just short of fully
/// clean so a tile can still settle its seam frame (`dfr_pipeline._ANCHOR_KEYFRAME_STRENGTH`).
pub const ANCHOR_KEYFRAME_STRENGTH: f32 = 0.95;

/// Ancestral eta for the temporal-round tile denoise (`dfr_pipeline._TEMPORAL_ANCESTRAL_ETA`).
pub const TEMPORAL_ANCESTRAL_ETA: f32 = 0.5;

/// Conditioning-fps cap for temporal rounds (`dfr_pipeline._MAX_CONDITIONING_FPS`): RoPE time is
/// `pixel_frame / fps`, and a 120 fps time base halves every token's temporal span versus the
/// trained distribution. Playback fps is unaffected (decode-side only).
pub const MAX_CONDITIONING_FPS: f32 = 60.0;

/// Stage-1 ancestral sampler defaults for LTX >= 2.5 (`distilled.ANCESTRAL_ETA` /
/// `ANCESTRAL_S_NOISE` / `ANCESTRAL_NOISE_SEED_OFFSET` — the seed offset keeps the loop's first
/// draw from being bit-identical to the initial latent noise).
pub const ANCESTRAL_ETA: f32 = 1.0;
pub const ANCESTRAL_S_NOISE: f32 = 1.0;
pub const ANCESTRAL_NOISE_SEED_OFFSET: u64 = 10_000;

/// Per-(round, tile) ancestral noise-seed stride (`dfr_pipeline`: `seed + 1000·round + tile`).
/// Tiles are positionally identical, so a shared ancestral seed would inject byte-identical noise
/// into every one of them.
pub const TEMPORAL_TILE_SEED_STRIDE: u64 = 1000;

/// The DFR CLI bounds `--temporal-upsample-rounds` to `{0, 1, 2}`.
pub const MAX_TEMPORAL_UPSAMPLE_ROUNDS: u32 = 2;

/// One temporal denoise tile in global pixel / latent coordinates (`dfr_layout.TileRange`).
///
/// `pixel_start` / `pixel_end` are inclusive, `latent_end_exclusive` is half-open. Non-first tiles
/// start [`TILE_LEAD_SEGMENTS`] segments before the region they own, so the seam shared with the
/// previous tile falls inside the window. `anchor_kf_global` are the seam keyframes inside the
/// window (frame 0 is never a keyframe, so the first tile's window start contributes no anchor);
/// `slot_kf_global` are the mid-segment positions this window invents. `drop_latent_prefix` is the
/// stitch handover: the previous tile keeps the seam latent and this tile resumes strictly after
/// it, so the prefix drops the lead-in plus that seam latent.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct DfrTileRange {
    pub pixel_start: i64,
    pub pixel_end: i64,
    pub latent_start: usize,
    pub latent_end_exclusive: usize,
    pub anchor_kf_global: Vec<i64>,
    pub slot_kf_global: Vec<i64>,
    pub drop_latent_prefix: usize,
}

impl DfrTileRange {
    /// Pixel-frame count of this tile's denoise window (`(T_lat − 1)·scale + 1`).
    pub fn local_frames(&self, temporal_scale: i64) -> i64 {
        (self.latent_end_exclusive as i64 - self.latent_start as i64 - 1) * temporal_scale + 1
    }

    /// Latent-frame count of this tile's denoise window.
    pub fn latent_frames(&self) -> usize {
        self.latent_end_exclusive - self.latent_start
    }
}

/// Pick the keyframe segment length from [`SEGMENT_CANDIDATES`], preferring whichever pads least;
/// ties keep the larger segment. `content_frames` is `num_frames − 1`.
pub fn choose_segment_length(content_frames: i64) -> Result<i64> {
    if content_frames < 1 {
        return Err(Error::Msg(format!(
            "ltx dfr: content_frames must be >= 1, got {content_frames}"
        )));
    }
    let pad = |segment: i64| (segment - content_frames % segment) % segment;
    let mut best = SEGMENT_CANDIDATES[0];
    let mut best_pad = pad(best);
    for &candidate in &SEGMENT_CANDIDATES[1..] {
        let p = pad(candidate);
        if p < best_pad || (p == best_pad && candidate > best) {
            best = candidate;
            best_pad = p;
        }
    }
    Ok(best)
}

/// Pad `(num_frames − 1)` up to a multiple of the segment length, returning
/// `(padded_num_frames, segment, positions)`. Positions are `[S, 2S, …, N' − 1]`: frame 0 is
/// excluded (already a single-pixel-frame token under causal encoding) and the terminal frame is
/// included (`dfr_layout.resolve_canvas`).
pub fn resolve_canvas(num_frames: i64, temporal_scale: i64) -> Result<(i64, i64, Vec<i64>)> {
    if num_frames < 1 {
        return Err(Error::Msg(format!(
            "ltx dfr: num_frames must be >= 1, got {num_frames}"
        )));
    }
    if (num_frames - 1) % temporal_scale != 0 {
        return Err(Error::Msg(format!(
            "ltx dfr: num_frames must satisfy (num_frames - 1) % {temporal_scale} == 0 \
             (got {num_frames})"
        )));
    }
    let content = num_frames - 1;
    if content == 0 {
        return Err(Error::Msg(
            "ltx dfr: the canvas needs at least 2 pixel frames".into(),
        ));
    }
    let segment = choose_segment_length(content)?;
    let content_padded = content + (segment - content % segment) % segment;
    let positions: Vec<i64> = (1..=content_padded / segment).map(|i| segment * i).collect();
    Ok((content_padded + 1, segment, positions))
}

/// Map an x`temporal_scale`-border pixel frame to its latent index.
pub fn pixel_to_latent_index(pixel_frame: i64, temporal_scale: i64) -> Result<usize> {
    if pixel_frame < 0 {
        return Err(Error::Msg(format!(
            "ltx dfr: pixel_frame must be >= 0, got {pixel_frame}"
        )));
    }
    if pixel_frame != 0 && pixel_frame % temporal_scale != 0 {
        return Err(Error::Msg(format!(
            "ltx dfr: pixel_frame {pixel_frame} is not on the x{temporal_scale} latent border"
        )));
    }
    Ok((pixel_frame / temporal_scale) as usize)
}

/// Split `n_segments` into `num_tiles` contiguous owned runs, largest first.
fn owned_segment_counts(n_segments: usize, num_tiles: usize) -> Vec<usize> {
    let base = n_segments / num_tiles;
    let remainder = n_segments % num_tiles;
    (0..num_tiles)
        .map(|i| base + usize::from(i < remainder))
        .collect()
}

/// One window owning segments `[own_lo, own_hi)`, preceded by `lead_segments` of lead-in.
fn build_tile(
    boundaries: &[i64],
    own_lo: usize,
    own_hi: usize,
    lead_segments: usize,
    temporal_scale: i64,
) -> Result<DfrTileRange> {
    let window_lo = own_lo.saturating_sub(lead_segments);
    let pixel_start = boundaries[window_lo];
    let pixel_end = boundaries[own_hi];
    let latent_start = pixel_to_latent_index(pixel_start, temporal_scale)?;

    // Handover exactly at the shared keyframe: the previous tile keeps the seam latent (the token
    // ending on the KF mark) and this tile resumes strictly after it.
    let mut drop_latent_prefix =
        pixel_to_latent_index(boundaries[own_lo], temporal_scale)? - latent_start;
    if own_lo > 0 {
        drop_latent_prefix += 1;
    }

    Ok(DfrTileRange {
        pixel_start,
        pixel_end,
        latent_start,
        latent_end_exclusive: pixel_to_latent_index(pixel_end, temporal_scale)? + 1,
        anchor_kf_global: (window_lo..=own_hi)
            .map(|i| boundaries[i])
            .filter(|&b| b != 0)
            .collect(),
        slot_kf_global: (window_lo..own_hi)
            .map(|i| (boundaries[i] + boundaries[i + 1]) / 2)
            .collect(),
        drop_latent_prefix,
    })
}

/// Partition the canvas into `num_tiles` keyframe-seam tiles, gapless, with a lead-in
/// (`dfr_layout.tile_ranges`). Owned segment runs are contiguous; each non-first window reaches
/// `lead_segments` back so it denoises through the shared seam keyframe. `num_tiles` is clamped to
/// the segment count.
pub fn tile_ranges(
    seam_positions: &[i64],
    num_frames: i64,
    num_tiles: usize,
    temporal_scale: i64,
    lead_segments: usize,
) -> Result<Vec<DfrTileRange>> {
    if num_frames < 2 {
        return Err(Error::Msg(format!(
            "ltx dfr: num_frames must be >= 2, got {num_frames}"
        )));
    }
    if seam_positions.is_empty() {
        return Err(Error::Msg("ltx dfr: seam_positions must be non-empty".into()));
    }
    if *seam_positions.last().expect("non-empty") != num_frames - 1 {
        return Err(Error::Msg(format!(
            "ltx dfr: last seam must be the terminal frame {}, got {}",
            num_frames - 1,
            seam_positions.last().expect("non-empty")
        )));
    }
    if lead_segments < 1 {
        return Err(Error::Msg(format!(
            "ltx dfr: lead_segments must be >= 1, got {lead_segments}"
        )));
    }

    let mut boundaries = Vec::with_capacity(seam_positions.len() + 1);
    boundaries.push(0i64);
    boundaries.extend_from_slice(seam_positions);
    for i in 1..boundaries.len() {
        let span = boundaries[i] - boundaries[i - 1];
        if span <= 0 {
            return Err(Error::Msg(format!(
                "ltx dfr: seam_positions must be strictly increasing, got {seam_positions:?}"
            )));
        }
        if span % temporal_scale != 0 {
            return Err(Error::Msg(format!(
                "ltx dfr: segment span {span} is not a multiple of temporal scale {temporal_scale}"
            )));
        }
        if span / temporal_scale < 2 {
            return Err(Error::Msg(format!(
                "ltx dfr: segment span {span} is under 2 latent frames, too short to carry a tile \
                 lead-in"
            )));
        }
    }

    let n_segments = boundaries.len() - 1;
    let mut tiles = Vec::new();
    let mut own_lo = 0usize;
    for (index, count) in owned_segment_counts(n_segments, num_tiles.min(n_segments))
        .into_iter()
        .enumerate()
    {
        tiles.push(build_tile(
            &boundaries,
            own_lo,
            own_lo + count,
            if index > 0 { lead_segments } else { 0 },
            temporal_scale,
        )?);
        own_lo += count;
    }
    Ok(tiles)
}

/// Shift global pixel indices into a tile-local frame (`local = global − pixel_start`).
pub fn remap_positions_to_local(positions: &[i64], pixel_start: i64) -> Vec<i64> {
    positions.iter().map(|p| p - pixel_start).collect()
}

/// Evenly spaced interior keyframe positions for a `--num-generated-keyframes` count request:
/// `linspace(0, num_frames − 1, k + 2)` rounded **half-to-even** (torch semantics), interior points
/// only (`helpers.evenly_spaced_keyframe_positions`).
pub fn evenly_spaced_keyframe_positions(num_keyframes: u32, num_frames: i64) -> Vec<i64> {
    if num_keyframes == 0 || num_frames < 2 {
        return Vec::new();
    }
    let last = (num_frames - 1) as f64;
    let denom = (num_keyframes + 1) as f64;
    (1..=num_keyframes as i64)
        .map(|i| (last * i as f64 / denom).round_ties_even() as i64)
        .collect()
}

/// Validate + normalize explicit generated-keyframe pixel positions: sorted, deduped, all in
/// `[0, num_frames)` (`helpers.resolve_generated_keyframes`, sequence arm).
pub fn resolve_generated_keyframe_positions(
    positions: &[i64],
    num_frames: i64,
) -> Result<Vec<i64>> {
    let mut out: Vec<i64> = positions.to_vec();
    out.sort_unstable();
    out.dedup();
    if let (Some(&first), Some(&last)) = (out.first(), out.last()) {
        if first < 0 || last >= num_frames {
            return Err(Error::Msg(format!(
                "ltx dfr: generated keyframe positions must lie in [0, {num_frames}), got \
                 {positions:?}"
            )));
        }
    }
    Ok(out)
}

/// Frame count after one temporal x2 round: `N → 2(N − 1) + 1` (pixel frames; the same recurrence
/// holds for latent frames, which is what the temporal upsampler's `T → 2T − 1` implements).
pub fn temporal_upsampled_frames(num_frames: i64) -> i64 {
    2 * (num_frames - 1) + 1
}

/// The caller's frame contract after `rounds` temporal rounds over a possibly padded canvas:
/// `(requested − 1) · 2^rounds + 1` (`requested − 1` is a multiple of the VAE temporal scale, so
/// the trim always lands on a latent boundary).
pub fn dfr_target_frames(requested_frames: i64, rounds: u32) -> i64 {
    (requested_frames - 1) * (1i64 << rounds) + 1
}

/// Where one contiguous run of generated-keyframe slot tokens sits in a token sequence
/// (`ltx_core.types.GeneratedKeyframeLayout`): appended by a single conditioning item so the run
/// is exactly locatable — `first_token` is the sequence length before the append, and each of the
/// `pixel_frame_indices` owns `tokens_per_keyframe` consecutive tokens.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct GeneratedKeyframeLayout {
    pub pixel_frame_indices: Vec<i64>,
    pub tokens_per_keyframe: usize,
    pub first_token: usize,
}

/// Coefficients of one **rectified-flow** ancestral Euler step
/// (`EulerAncestralDiffusionStep.step`, computed in f64 like the reference's f32-upcast):
///
/// ```text
/// downstep_ratio = 1 + (σ_next/σ − 1)·η
/// σ_down         = σ_next · downstep_ratio
/// x'             = (σ_down/σ)·x + (1 − σ_down/σ)·x0
/// η > 0:  α_next = 1 − σ_next;  α_down = 1 − σ_down
///         x''    = (α_next/α_down)·x' + noise · s_noise · √max(σ_next² − σ_down²·α_next²/α_down², 0)
/// ```
///
/// At `η = 0` this reduces exactly to the deterministic Euler interpolation
/// (`alpha_ratio = 1`, `renoise_coeff = 0`). The terminal step (`σ_next = 0`) is the caller's
/// short-circuit to the denoised prediction and never reaches these coefficients.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct RfAncestralCoeffs {
    /// `σ_down / σ` — the interpolation weight on the current sample (`1 − it` on the denoised).
    pub sigma_down_ratio: f32,
    /// `α_next / α_down` — the variance-preserving rescale on the interpolated sample.
    pub alpha_ratio: f32,
    /// `s_noise · √max(σ_next² − σ_down²·α_next²/α_down², 0)` — the fresh-noise scale.
    pub renoise_coeff: f32,
}

impl RfAncestralCoeffs {
    /// Compute the step coefficients for `σ → σ_next` at stochasticity `eta` / `s_noise`.
    pub fn new(sigma: f32, sigma_next: f32, eta: f32, s_noise: f32) -> Result<Self> {
        if sigma <= 0.0 {
            return Err(Error::Msg(format!(
                "ltx dfr: rf ancestral step needs sigma > 0, got {sigma}"
            )));
        }
        if sigma_next <= 0.0 {
            return Err(Error::Msg(format!(
                "ltx dfr: the terminal step (sigma_next = {sigma_next}) short-circuits to the \
                 denoised prediction; it has no ancestral coefficients"
            )));
        }
        let (s, sn, eta, s_noise) = (sigma as f64, sigma_next as f64, eta as f64, s_noise as f64);
        let downstep_ratio = 1.0 + (sn / s - 1.0) * eta;
        let sigma_down = sn * downstep_ratio;
        let sigma_down_ratio = sigma_down / s;
        if eta <= 0.0 {
            return Ok(Self {
                sigma_down_ratio: sigma_down_ratio as f32,
                alpha_ratio: 1.0,
                renoise_coeff: 0.0,
            });
        }
        let alpha_next = 1.0 - sn;
        let alpha_down = 1.0 - sigma_down;
        let alpha_ratio = alpha_next / alpha_down;
        let renoise = (sn * sn - sigma_down * sigma_down * alpha_ratio * alpha_ratio).max(0.0);
        Ok(Self {
            sigma_down_ratio: sigma_down_ratio as f32,
            alpha_ratio: alpha_ratio as f32,
            renoise_coeff: (s_noise * renoise.sqrt()) as f32,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Golden canvases computed by executing the reference `dfr_layout.resolve_canvas` at
    /// `d1511477` (torch-free extraction, run 2026-08-27).
    #[test]
    fn resolve_canvas_matches_reference_goldens() {
        for (n, want_n, want_s, want_pos) in [
            (9i64, 25i64, 24i64, vec![24i64]),
            (25, 25, 24, vec![24]),
            (49, 49, 24, vec![24, 48]),
            (97, 97, 32, vec![32, 64, 96]),
            (121, 121, 24, vec![24, 48, 72, 96, 120]),
            (145, 145, 24, vec![24, 48, 72, 96, 120, 144]),
            (153, 161, 32, vec![32, 64, 96, 128, 160]),
            (193, 193, 32, vec![32, 64, 96, 128, 160, 192]),
            (
                241,
                241,
                24,
                vec![24, 48, 72, 96, 120, 144, 168, 192, 216, 240],
            ),
        ] {
            let (got_n, got_s, got_pos) = resolve_canvas(n, 8).unwrap();
            assert_eq!((got_n, got_s, got_pos), (want_n, want_s, want_pos), "n={n}");
        }
    }

    #[test]
    fn segment_choice_matches_reference() {
        for (content, want) in [
            (24i64, 24i64),
            (32, 32),
            (48, 24),
            (96, 32),
            (120, 24),
            (144, 24),
            (152, 32),
            (192, 32),
            (240, 24),
            (28, 32),
            (40, 24),
        ] {
            assert_eq!(choose_segment_length(content).unwrap(), want, "content={content}");
        }
    }

    #[test]
    fn canvas_rejects_off_grid_and_degenerate_inputs() {
        assert!(resolve_canvas(0, 8).is_err());
        assert!(resolve_canvas(1, 8).is_err(), "content == 0");
        assert!(resolve_canvas(10, 8).is_err(), "(n-1) % 8 != 0");
    }

    /// Golden round-1 tiling for the 121-frame canvas (seams x2 on the 241-frame upsampled canvas,
    /// 2 tiles) — from the reference `tile_ranges` at `d1511477`.
    #[test]
    fn round1_tiles_match_reference_goldens() {
        let (n, _, pos) = resolve_canvas(121, 8).unwrap();
        let seams: Vec<i64> = pos.iter().map(|p| 2 * p).collect();
        let n1 = temporal_upsampled_frames(n);
        assert_eq!(n1, 241);
        let tiles = tile_ranges(&seams, n1, 2, 8, TILE_LEAD_SEGMENTS).unwrap();
        assert_eq!(
            tiles,
            vec![
                DfrTileRange {
                    pixel_start: 0,
                    pixel_end: 144,
                    latent_start: 0,
                    latent_end_exclusive: 19,
                    anchor_kf_global: vec![48, 96, 144],
                    slot_kf_global: vec![24, 72, 120],
                    drop_latent_prefix: 0,
                },
                DfrTileRange {
                    pixel_start: 96,
                    pixel_end: 240,
                    latent_start: 12,
                    latent_end_exclusive: 31,
                    anchor_kf_global: vec![96, 144, 192, 240],
                    slot_kf_global: vec![120, 168, 216],
                    drop_latent_prefix: 7,
                },
            ]
        );
        // The stitch is gapless by construction: kept latents sum to the full canvas.
        let kept: usize = tiles
            .iter()
            .map(|t| t.latent_frames() - t.drop_latent_prefix)
            .sum();
        assert_eq!(kept as i64, (n1 - 1) / 8 + 1);
    }

    /// Golden round-2 tiling: the carry bag after round 1 (seams U slot midpoints) re-doubled onto
    /// the 481-frame canvas, 4 tiles with largest-first ownership — reference `tile_ranges`.
    #[test]
    fn round2_tiles_match_reference_goldens() {
        let (n, _, pos) = resolve_canvas(121, 8).unwrap();
        let seams1: Vec<i64> = pos.iter().map(|p| 2 * p).collect();
        let n1 = temporal_upsampled_frames(n);
        let round1 = tile_ranges(&seams1, n1, 2, 8, TILE_LEAD_SEGMENTS).unwrap();
        let mut carry: Vec<i64> = seams1.clone();
        carry.extend(round1.iter().flat_map(|t| t.slot_kf_global.iter().copied()));
        carry.sort_unstable();
        carry.dedup();
        let seams2: Vec<i64> = carry.iter().map(|p| 2 * p).collect();
        let n2 = temporal_upsampled_frames(n1);
        assert_eq!(n2, 481);
        let tiles = tile_ranges(&seams2, n2, 4, 8, TILE_LEAD_SEGMENTS).unwrap();
        let brief: Vec<(i64, i64, usize, usize, usize)> = tiles
            .iter()
            .map(|t| {
                (
                    t.pixel_start,
                    t.pixel_end,
                    t.latent_start,
                    t.latent_end_exclusive,
                    t.drop_latent_prefix,
                )
            })
            .collect();
        assert_eq!(
            brief,
            vec![
                (0, 144, 0, 19, 0),
                (96, 288, 12, 37, 7),
                (240, 384, 30, 49, 7),
                (336, 480, 42, 61, 7),
            ]
        );
        assert_eq!(tiles[1].anchor_kf_global, vec![96, 144, 192, 240, 288]);
        assert_eq!(tiles[1].slot_kf_global, vec![120, 168, 216, 264]);
        let kept: usize = tiles
            .iter()
            .map(|t| t.latent_frames() - t.drop_latent_prefix)
            .sum();
        assert_eq!(kept as i64, (n2 - 1) / 8 + 1);
    }

    #[test]
    fn tile_ranges_validates_inputs() {
        // Wrong terminal seam.
        assert!(tile_ranges(&[24], 33, 1, 8, 1).is_err());
        // Non-increasing seams.
        assert!(tile_ranges(&[24, 24, 48], 49, 1, 8, 1).is_err());
        // Span off the latent grid.
        assert!(tile_ranges(&[20], 21, 1, 8, 1).is_err());
        // Span under 2 latent frames.
        assert!(tile_ranges(&[8, 48], 49, 1, 8, 1).is_err());
        // num_tiles clamps to the segment count instead of erroring.
        let t = tile_ranges(&[24, 48], 49, 8, 8, 1).unwrap();
        assert_eq!(t.len(), 2);
    }

    #[test]
    fn evenly_spaced_positions_round_ties_even_like_torch() {
        // linspace(0, 120, 5)[1:-1] = [30, 60, 90] — exact, no ties.
        assert_eq!(evenly_spaced_keyframe_positions(3, 121), vec![30, 60, 90]);
        // linspace(0, 9, 3)[1:-1] = [4.5] → round-half-even → 4 (torch.round semantics; a
        // round-half-up port would say 5).
        assert_eq!(evenly_spaced_keyframe_positions(1, 10), vec![4]);
        assert!(evenly_spaced_keyframe_positions(0, 121).is_empty());
    }

    #[test]
    fn explicit_positions_validate_range() {
        assert_eq!(
            resolve_generated_keyframe_positions(&[40, 20, 20], 49).unwrap(),
            vec![20, 40]
        );
        assert!(resolve_generated_keyframe_positions(&[-1], 49).is_err());
        assert!(resolve_generated_keyframe_positions(&[49], 49).is_err());
    }

    /// RF-ancestral coefficients against hand-computed reference values
    /// (`EulerAncestralDiffusionStep.step` math evaluated in f64).
    #[test]
    fn rf_ancestral_coeffs_match_reference_math() {
        // σ=0.975 → σn=0.909375 at η=0.5 (the temporal-round configuration).
        let c = RfAncestralCoeffs::new(0.975, 0.909375, 0.5, 1.0).unwrap();
        let (s, sn) = (0.975f64, 0.909375f64);
        let dsr = 1.0 + (sn / s - 1.0) * 0.5;
        let sd = sn * dsr;
        let ar = (1.0 - sn) / (1.0 - sd);
        let rn = (sn * sn - sd * sd * ar * ar).max(0.0).sqrt();
        assert!((c.sigma_down_ratio as f64 - sd / s).abs() < 1e-7);
        assert!((c.alpha_ratio as f64 - ar).abs() < 1e-7);
        assert!((c.renoise_coeff as f64 - rn).abs() < 1e-7);
        assert!(c.renoise_coeff > 0.0, "η>0 must inject fresh noise");

        // η=0 reduces to the deterministic Euler interpolation: ratio σn/σ, no rescale, no noise.
        let c0 = RfAncestralCoeffs::new(0.975, 0.909375, 0.0, 1.0).unwrap();
        assert!((c0.sigma_down_ratio as f64 - sn / s).abs() < 1e-7);
        assert_eq!(c0.alpha_ratio, 1.0);
        assert_eq!(c0.renoise_coeff, 0.0);

        // s_noise scales only the injected-noise term.
        let c2 = RfAncestralCoeffs::new(0.975, 0.909375, 0.5, 2.0).unwrap();
        assert!((c2.renoise_coeff - 2.0 * c.renoise_coeff).abs() < 1e-6);
        assert_eq!(c2.sigma_down_ratio, c.sigma_down_ratio);

        // Full ancestral η=1 (the 2.5 stage-1 configuration): σ_down = σn²/σ.
        let c1 = RfAncestralCoeffs::new(0.99375, 0.9875, 1.0, 1.0).unwrap();
        let (s, sn) = (0.99375f64, 0.9875f64);
        assert!((c1.sigma_down_ratio as f64 - (sn * sn / s) / s).abs() < 1e-7);
    }

    #[test]
    fn rf_ancestral_rejects_terminal_and_degenerate_sigmas() {
        assert!(RfAncestralCoeffs::new(0.0, 0.5, 1.0, 1.0).is_err());
        assert!(RfAncestralCoeffs::new(0.5, 0.0, 1.0, 1.0).is_err());
    }

    #[test]
    fn frame_contracts() {
        assert_eq!(temporal_upsampled_frames(121), 241);
        assert_eq!(dfr_target_frames(121, 0), 121);
        assert_eq!(dfr_target_frames(121, 1), 241);
        assert_eq!(dfr_target_frames(121, 2), 481);
        // A padded canvas trims back to the caller's contract: 153 → canvas 161, 2 rounds → 641
        // generated but the contract is (153−1)·4+1 = 609.
        assert_eq!(dfr_target_frames(153, 2), 609);
    }
}
