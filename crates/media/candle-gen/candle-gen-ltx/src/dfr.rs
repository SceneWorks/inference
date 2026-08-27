//! The LTX-2.5 **DFR** pipeline (sc-18789) — candle twin of `mlx-gen-ltx/src/dfr.rs`; keep the two
//! in step. Generated keyframe slots, the full-resolution detailing pass, and tiled temporal
//! upsampling rounds, ported from the reference `ltx_pipelines/dfr_pipeline.py` (Lightricks/LTX-2
//! v1.2.0 @ `d1511477`) over this crate's token-native conditioning and denoise loops, with the
//! shared integer geometry in [`candle_gen::gen_core::ltx_dfr`].
//!
//! [`run_temporal_rounds`] owns the round/tile/seam arithmetic behind injected closures (temporal
//! upsampler + per-tile denoise) so the exact production code path is drivable by CPU tests
//! without the 22B DiT; [`denoise_dfr_tile`] is the production tile closure body.

use candle_gen::candle_core::{Error, Result, Tensor};
use candle_gen::gen_core::ltx_dfr as gen_core_dfr;
use candle_gen::gen_core::ltx_dfr::DfrTileRange;
use candle_gen::gen_core::{CancelFlag, Progress};
use rand::rngs::StdRng;
use rand::SeedableRng;

use crate::conditioning::{
    append_generated_keyframe_slots, append_single_frame_keyframes, VideoTokenState,
};
use crate::config::{SPATIAL_SCALE, TEMPORAL_SCALE, TEMPORAL_SIGMAS};
use crate::pipeline::{denoise_tokens_rf_ancestral, unflatten_latent};
use crate::transformer::AvDiT;

/// Gather latent frames (axis 2) by index.
fn take_frames(x: &Tensor, indices: &[u32]) -> Result<Tensor> {
    let idx = Tensor::from_vec(indices.to_vec(), indices.len(), x.device())?;
    x.contiguous()?.index_select(&idx, 2)
}

/// Temporal window `x[:, :, start..end]`.
fn frame_window(x: &Tensor, start: usize, end: usize) -> Result<Tensor> {
    x.narrow(2, start, end - start)
}

/// Stack the nearest video latent frames as `(B, C, K, H, W)` slot seeds
/// (`dfr_pipeline._slot_initials_from_video`): tile-local pixel `position / temporal_scale`,
/// rounded half-to-even like the reference's `round()`, clamped to the window.
pub fn slot_initials_from_video(
    video_latent: &Tensor,
    positions_local: &[i64],
    temporal_scale: i64,
) -> Result<Tensor> {
    let frames = video_latent.dim(2)? as i64;
    let idx: Vec<u32> = positions_local
        .iter()
        .map(|&p| {
            ((p as f64 / temporal_scale as f64).round_ties_even() as i64).clamp(0, frames - 1)
                as u32
        })
        .collect();
    take_frames(video_latent, &idx)
}

/// Concatenate tile video latents along T, each contributing `latent[drop_latent_prefix..]`
/// (`dfr_layout.stitch_tile_latents`). The prefix drop is the seam handover: the previous tile
/// keeps the shared seam latent, so a wrong prefix double-writes or gaps the seam.
pub fn stitch_tile_latents(tile_latents: &[Tensor], ranges: &[DfrTileRange]) -> Result<Tensor> {
    if tile_latents.len() != ranges.len() {
        return Err(Error::Msg(format!(
            "ltx dfr: expected {} tile latents, got {}",
            ranges.len(),
            tile_latents.len()
        )));
    }
    if tile_latents.is_empty() {
        return Err(Error::Msg("ltx dfr: tile_latents must be non-empty".into()));
    }
    let mut pieces = Vec::with_capacity(tile_latents.len());
    for (latent, tile) in tile_latents.iter().zip(ranges) {
        let (_b, _c, t, _h, _w) = latent.dims5()?;
        let expected_t = tile.latent_frames();
        if t != expected_t {
            return Err(Error::Msg(format!(
                "ltx dfr: tile latent T={t} != expected {expected_t} for range [{}, {})",
                tile.latent_start, tile.latent_end_exclusive
            )));
        }
        if tile.drop_latent_prefix >= t {
            return Err(Error::Msg(format!(
                "ltx dfr: drop_latent_prefix={} invalid for tile T={t}",
                tile.drop_latent_prefix
            )));
        }
        pieces.push(frame_window(latent, tile.drop_latent_prefix, t)?);
    }
    Tensor::cat(&pieces.iter().collect::<Vec<_>>(), 2)
}

/// Build the next round's anchor bag: carried keyframe stills plus this round's denoised slots
/// (`dfr_pipeline._merge_carry_forward_keyframes`). Positions are on the current round's pixel
/// grid; the next round remaps (×2). On a (structurally impossible) collision the slot wins.
pub fn merge_carry_forward_keyframes(
    anchor_positions: &[i64],
    anchor_latents: Option<&Tensor>,
    slot_positions: &[i64],
    slot_latents: Option<&Tensor>,
) -> Result<(Vec<i64>, Tensor)> {
    let mut by_position: std::collections::BTreeMap<i64, (u8, u32)> = Default::default();
    for (which, (positions, latents)) in [
        (anchor_positions, anchor_latents),
        (slot_positions, slot_latents),
    ]
    .into_iter()
    .enumerate()
    {
        if positions.is_empty() {
            continue;
        }
        let Some(latents) = latents else {
            let label = if which == 0 { "anchor" } else { "slot" };
            return Err(Error::Msg(format!(
                "ltx dfr: missing {label} keyframe latents for carry-forward merge"
            )));
        };
        if latents.dim(2)? != positions.len() {
            return Err(Error::Msg(format!(
                "ltx dfr: carry-forward latents K={} != {} positions",
                latents.dim(2)?,
                positions.len()
            )));
        }
        for (index, &position) in positions.iter().enumerate() {
            by_position.insert(position, (which as u8, index as u32));
        }
    }
    if by_position.is_empty() {
        return Err(Error::Msg(
            "ltx dfr: carry-forward keyframe bag is empty".into(),
        ));
    }
    let ordered: Vec<i64> = by_position.keys().copied().collect();
    let mut frames = Vec::with_capacity(ordered.len());
    for (which, index) in by_position.values() {
        let src = if *which == 0 {
            anchor_latents.expect("checked above")
        } else {
            slot_latents.expect("checked above")
        };
        frames.push(take_frames(src, &[*index])?);
    }
    Ok((ordered, Tensor::cat(&frames.iter().collect::<Vec<_>>(), 2)?))
}

/// A replace-latent image anchor carried into the temporal rounds (twin of the MLX
/// `DfrRoundImage`; see that doc for the window-filter projection note): the conditioning latent
/// at stage-2 resolution (`(B, C, 1, H, W)`), its **pre-round** x8-aligned pixel position, and
/// its strength.
pub struct DfrRoundImage {
    pub pixel_position: i64,
    pub latent: Tensor,
    pub strength: f32,
}

/// One image re-attached to a tile, in tile-local latent coordinates.
pub struct DfrTileImage {
    pub local_latent_index: usize,
    pub latent: Tensor,
    pub strength: f32,
}

/// One tile's denoise inputs, in tile-local coordinates (twin of the MLX `DfrTileJob`).
pub struct DfrTileJob<'a> {
    pub round: u32,
    pub tile_index: usize,
    pub tile: &'a DfrTileRange,
    /// `(B, C, T_tile, H, W)` window slice of the temporally upsampled video latent.
    pub tile_video: Tensor,
    /// Tile-local anchor positions; every non-first tile's first anchor is exactly `0` (the
    /// shared seam).
    pub anchor_positions_local: Vec<i64>,
    pub anchor_latents: Option<Tensor>,
    pub slot_positions_local: Vec<i64>,
    pub slot_initials: Option<Tensor>,
    /// Replace-latent image anchors whose (round-scaled) pixel position falls inside this window,
    /// remapped tile-locally.
    pub image_keyframes: Vec<DfrTileImage>,
    pub local_frames: i64,
    pub cond_fps: f32,
    /// `seed + 1000·round + tile_index` — tiles are positionally identical, so a shared ancestral
    /// seed would inject byte-identical noise into every one.
    pub noise_seed: u64,
}

/// One tile's denoise result.
pub struct DfrTileResult {
    pub latent: Tensor,
    pub generated_keyframes: Option<Tensor>,
}

/// The output of [`run_temporal_rounds`].
#[derive(Debug)]
pub struct TemporalRoundsOutput {
    pub video_latent: Tensor,
    pub num_frames: i64,
    pub fps: f32,
}

/// Tiled temporal x2 upsampling rounds (`dfr_pipeline.__call__`'s round loop) — see the MLX twin
/// for the full semantics. Every seam/round decision here is testable without the DiT.
#[allow(clippy::too_many_arguments)]
pub fn run_temporal_rounds(
    video_latent: &Tensor,
    carry_positions: &[i64],
    carry_keyframes: &Tensor,
    images: &[DfrRoundImage],
    num_frames: i64,
    fps: f32,
    seed: u64,
    rounds: u32,
    upsample: &mut dyn FnMut(&Tensor) -> Result<Tensor>,
    denoise_tile: &mut dyn FnMut(&DfrTileJob) -> candle_gen::Result<DfrTileResult>,
) -> candle_gen::Result<TemporalRoundsOutput> {
    use candle_gen::CandleError;
    for image in images {
        if image.pixel_position < 0 || image.pixel_position % (TEMPORAL_SCALE as i64) != 0 {
            return Err(CandleError::Msg(format!(
                "ltx dfr: image anchor pixel position {} is not on the x{TEMPORAL_SCALE} latent \
                 border",
                image.pixel_position
            )));
        }
    }
    if rounds > gen_core_dfr::MAX_TEMPORAL_UPSAMPLE_ROUNDS {
        return Err(CandleError::Msg(format!(
            "ltx dfr: temporal_upsample_rounds must be 0..={}, got {rounds}",
            gen_core_dfr::MAX_TEMPORAL_UPSAMPLE_ROUNDS
        )));
    }
    let temporal_scale = TEMPORAL_SCALE as i64;
    let mut video = video_latent.clone();
    let mut num_frames = num_frames;
    let mut fps = fps;
    let mut carry_positions: Vec<i64> = carry_positions.to_vec();
    let mut carry_keyframes = carry_keyframes.clone();

    for round in 1..=rounds {
        if carry_positions.is_empty() {
            return Err(CandleError::Msg(format!(
                "ltx dfr: temporal round {round}: missing carry-forward keyframes"
            )));
        }
        video = upsample(&video)?;
        num_frames = gen_core_dfr::temporal_upsampled_frames(num_frames);
        fps *= 2.0;
        let expected_t = (num_frames - 1) / temporal_scale + 1;
        if video.dim(2)? as i64 != expected_t {
            return Err(CandleError::Msg(format!(
                "ltx dfr: temporal upsampler produced T={} for round {round}, expected \
                 {expected_t}",
                video.dim(2)?
            )));
        }
        let seam_positions: Vec<i64> = carry_positions.iter().map(|p| 2 * p).collect();
        let anchor_keyframes = carry_keyframes.clone();
        let seam_to_index: std::collections::HashMap<i64, u32> = seam_positions
            .iter()
            .enumerate()
            .map(|(i, &s)| (s, i as u32))
            .collect();
        let cond_fps = fps.min(gen_core_dfr::MAX_CONDITIONING_FPS);
        let tiles = gen_core_dfr::tile_ranges(
            &seam_positions,
            num_frames,
            1usize << round,
            temporal_scale,
            gen_core_dfr::TILE_LEAD_SEGMENTS,
        )
        .map_err(|e| CandleError::Msg(e.to_string()))?;

        let mut tile_latents: Vec<Tensor> = Vec::with_capacity(tiles.len());
        let mut slot_positions: Vec<i64> = Vec::new();
        let mut slot_latent_slices: Vec<Tensor> = Vec::new();

        for (tile_index, tile) in tiles.iter().enumerate() {
            let tile_video = frame_window(&video, tile.latent_start, tile.latent_end_exclusive)?;

            let anchor_global = &tile.anchor_kf_global;
            let (anchor_positions_local, anchor_latents) = if anchor_global.is_empty() {
                (Vec::new(), None)
            } else {
                let missing: Vec<i64> = anchor_global
                    .iter()
                    .copied()
                    .filter(|p| !seam_to_index.contains_key(p))
                    .collect();
                if !missing.is_empty() {
                    return Err(CandleError::Msg(format!(
                        "ltx dfr: anchor seams {missing:?} missing from the carry-forward bag"
                    )));
                }
                let idx: Vec<u32> = anchor_global.iter().map(|p| seam_to_index[p]).collect();
                (
                    gen_core_dfr::remap_positions_to_local(anchor_global, tile.pixel_start),
                    Some(take_frames(&anchor_keyframes, &idx)?),
                )
            };

            let slot_global = &tile.slot_kf_global;
            let (slot_positions_local, slot_initials) = if slot_global.is_empty() {
                (Vec::new(), None)
            } else {
                let local = gen_core_dfr::remap_positions_to_local(slot_global, tile.pixel_start);
                let initials = slot_initials_from_video(&tile_video, &local, temporal_scale)?;
                (local, Some(initials))
            };

            // Image conditioning is tile-local (reference: only in-window images re-attach,
            // remapped by −pixel_start; re-applying an outside image would pin the wrong frame
            // onto the seam).
            let image_keyframes: Vec<DfrTileImage> = images
                .iter()
                .filter_map(|image| {
                    let pos_r = image.pixel_position << round;
                    if pos_r < tile.pixel_start || pos_r > tile.pixel_end {
                        return None;
                    }
                    Some(DfrTileImage {
                        local_latent_index: ((pos_r - tile.pixel_start) / temporal_scale) as usize,
                        latent: image.latent.clone(),
                        strength: image.strength,
                    })
                })
                .collect();

            let job = DfrTileJob {
                round,
                tile_index,
                tile,
                local_frames: tile.local_frames(temporal_scale),
                tile_video,
                anchor_positions_local,
                anchor_latents,
                slot_positions_local,
                slot_initials,
                image_keyframes,
                cond_fps,
                noise_seed: seed
                    .wrapping_add(gen_core_dfr::TEMPORAL_TILE_SEED_STRIDE * u64::from(round))
                    .wrapping_add(tile_index as u64),
            };
            let has_slots = !job.slot_positions_local.is_empty();
            let result = denoise_tile(&job)?;
            tile_latents.push(result.latent);

            if has_slots {
                let Some(generated) = result.generated_keyframes else {
                    return Err(CandleError::Msg(format!(
                        "ltx dfr: temporal round {round}: tile {tile_index} produced no keyframe \
                         slots"
                    )));
                };
                if generated.dim(2)? != slot_global.len() {
                    return Err(CandleError::Msg(format!(
                        "ltx dfr: tile {tile_index} returned {} slot keyframes for {} slots",
                        generated.dim(2)?,
                        slot_global.len()
                    )));
                }
                slot_positions.extend_from_slice(slot_global);
                slot_latent_slices.push(generated);
            }
        }

        let stitched = stitch_tile_latents(&tile_latents, &tiles)?;
        if stitched.dim(2)? as i64 != expected_t {
            return Err(CandleError::Msg(format!(
                "ltx dfr: stitched latent T={} != expected {expected_t}",
                stitched.dim(2)?
            )));
        }
        video = stitched;

        // Lead-in segments repeat the previous tile's slots; the earlier tile's version wins.
        let slot_latents = if slot_latent_slices.is_empty() {
            None
        } else {
            Some(Tensor::cat(
                &slot_latent_slices.iter().collect::<Vec<_>>(),
                2,
            )?)
        };
        let (slot_positions, slot_latents) = match slot_latents {
            None => (Vec::new(), None),
            Some(all) => {
                let mut first_index: std::collections::BTreeMap<i64, u32> = Default::default();
                for (index, &position) in slot_positions.iter().enumerate() {
                    first_index.entry(position).or_insert(index as u32);
                }
                let positions: Vec<i64> = first_index.keys().copied().collect();
                let idx: Vec<u32> = first_index.values().copied().collect();
                (positions, Some(take_frames(&all, &idx)?))
            }
        };

        let (next_positions, next_keyframes) = merge_carry_forward_keyframes(
            &seam_positions,
            Some(&anchor_keyframes),
            &slot_positions,
            slot_latents.as_ref(),
        )?;
        carry_positions = next_positions;
        carry_keyframes = next_keyframes;
    }

    Ok(TemporalRoundsOutput {
        video_latent: video,
        num_frames,
        fps,
    })
}

/// Trim a possibly padded DFR canvas back to the caller's frame contract
/// (`(requested − 1) · 2^rounds + 1`; the trim always lands on a latent boundary).
pub fn trim_to_target_frames(
    video_latent: &Tensor,
    canvas_frames: i64,
    requested_frames: i64,
    rounds: u32,
) -> Result<(Tensor, i64)> {
    let target = gen_core_dfr::dfr_target_frames(requested_frames, rounds);
    if target > canvas_frames {
        return Err(Error::Msg(format!(
            "ltx dfr: target {target} frames exceeds the generated canvas {canvas_frames}"
        )));
    }
    if target == canvas_frames {
        return Ok((video_latent.clone(), canvas_frames));
    }
    let keep = ((target - 1) / (TEMPORAL_SCALE as i64) + 1) as usize;
    Ok((frame_window(video_latent, 0, keep)?, target))
}

/// Stage-entry noising for appended generated-keyframe slot tokens: where the token is a slot
/// (`keyframes_mask > 0`, `denoise_mask = 1`), lerp the slot's seeded latent toward fresh seeded
/// noise at the stage-entry `sigma` — the reference stage noiser applied to the appended run only.
pub fn noise_slot_tokens(state: &VideoTokenState, sigma: f32, seed: u64) -> Result<Tensor> {
    let Some(mask) = state.keyframes_mask.as_ref() else {
        return Ok(state.latent.clone());
    };
    let dims = state.latent.dims();
    let n: usize = dims.iter().product();
    let mut rng = StdRng::seed_from_u64(seed);
    let noise = Tensor::from_vec(
        candle_gen::seeded_normal_vec(&mut rng, n),
        dims,
        state.latent.device(),
    )?
    .to_dtype(state.latent.dtype())?;
    let gate = (mask.to_dtype(state.latent.dtype())? * sigma as f64)?
        .broadcast_as(state.latent.shape())?;
    let keep = (Tensor::ones_like(&gate)? - &gate)?;
    (noise * gate)? + (&state.latent * keep)?
}

/// The production tile denoise (`dfr_pipeline`'s per-tile stage call): re-noise the window at
/// `TEMPORAL_SIGMAS[0]` (seeded per tile), append the seam anchors as hard single-frame keyframes
/// ([`gen_core_dfr::ANCHOR_KEYFRAME_STRENGTH`]), append the window's generated slots seeded from
/// the video, run the **video-only** RF-ancestral loop at
/// [`gen_core_dfr::TEMPORAL_ANCESTRAL_ETA`], and read the slots back.
pub fn denoise_dfr_tile(
    dit: &AvDiT,
    job: &DfrTileJob<'_>,
    video_ctx: &Tensor,
    positions: &Tensor,
    cancel: &CancelFlag,
    on_model_forward: &mut dyn FnMut() -> Result<()>,
    on_progress: &mut dyn FnMut(Progress),
) -> candle_gen::Result<DfrTileResult> {
    let (_b, c, t, h, w) = job.tile_video.dims5()?;
    let spatial_scale = SPATIAL_SCALE as i64;

    // Tile-entry re-noise: noise·σ₀ + latent·(1 − σ₀), seeded per tile.
    let dims = job.tile_video.dims();
    let n: usize = dims.iter().product();
    let mut rng = StdRng::seed_from_u64(job.noise_seed);
    let noise = Tensor::from_vec(
        candle_gen::seeded_normal_vec(&mut rng, n),
        dims,
        job.tile_video.device(),
    )?
    .to_dtype(job.tile_video.dtype())?;
    let mut state = if job.image_keyframes.is_empty() {
        let renoised = crate::pipeline::renoise(&job.tile_video, &noise, TEMPORAL_SIGMAS[0])?;
        VideoTokenState::base(&renoised, positions)?
    } else {
        // Window-local image anchors: the replace-latent state pins the image frames instead of
        // re-noising them (the reference's per-tile image conditionings + the stage noiser).
        let dt = job.tile_video.dtype();
        let cast: Vec<Tensor> = job
            .image_keyframes
            .iter()
            .map(|image| image.latent.to_dtype(dt))
            .collect::<Result<_>>()?;
        let keyframes: Vec<crate::conditioning::Keyframe> = job
            .image_keyframes
            .iter()
            .zip(&cast)
            .map(|(image, latent)| crate::conditioning::Keyframe {
                latent,
                frame_idx: image.local_latent_index,
                strength: image.strength,
            })
            .collect();
        let i2v = crate::conditioning::apply_keyframes(&job.tile_video, &keyframes)?
            .noised(&noise, TEMPORAL_SIGMAS[0])?;
        VideoTokenState::from_i2v(&i2v, positions)?
    };
    if let (false, Some(anchors)) = (
        job.anchor_positions_local.is_empty(),
        job.anchor_latents.as_ref(),
    ) {
        state = append_single_frame_keyframes(
            &state,
            anchors,
            &job.anchor_positions_local,
            gen_core_dfr::ANCHOR_KEYFRAME_STRENGTH,
            spatial_scale,
            job.cond_fps,
        )?;
    }
    let has_slots = !job.slot_positions_local.is_empty();
    if has_slots {
        state = append_generated_keyframe_slots(
            &state,
            &job.slot_positions_local,
            job.slot_initials.as_ref(),
            job.local_frames,
            h,
            w,
            spatial_scale,
            job.cond_fps,
        )?;
        state.latent =
            noise_slot_tokens(&state, TEMPORAL_SIGMAS[0], job.noise_seed.wrapping_add(1))?;
    }

    let (state, _) = denoise_tokens_rf_ancestral(
        dit,
        &state,
        video_ctx,
        None,
        &TEMPORAL_SIGMAS,
        gen_core_dfr::TEMPORAL_ANCESTRAL_ETA,
        1.0,
        job.noise_seed,
        cancel,
        on_model_forward,
        on_progress,
    )?;

    let grid = state.latent.narrow(1, 0, state.target_tokens)?;
    let latent = unflatten_latent(&grid, t, h, w)?;
    debug_assert_eq!(latent.dim(1)?, c);
    let generated_keyframes = if has_slots {
        Some(crate::conditioning::take_generated_keyframes(&state, h, w)?)
    } else {
        None
    };
    Ok(DfrTileResult {
        latent,
        generated_keyframes,
    })
}

/// A replace-latent keyframe VAE-encoded at both stage resolutions (twin of the MLX
/// `StageKeyframe`, owned because the candle engine materializes stage latents up front).
pub struct DfrStageKeyframe {
    pub stage1: Tensor,
    pub stage2: Tensor,
    pub frame_idx: usize,
    pub strength: f32,
}

/// Everything [`generate_dfr_av_latents`] needs beyond the request-shaped parameters.
pub struct DfrComponents<'a> {
    pub dit: &'a AvDiT,
    /// Latent normalize/denormalize around the learned upsamplers (the candle upsampler runs in
    /// VAE space; see the two-stage renderer's `learned_upsample`).
    pub vae: &'a crate::vae::LtxVideoVae,
    pub spatial_upsampler: &'a crate::upsampler::LatentUpsampler,
    /// Required when `temporal_upsample_rounds > 0`.
    pub temporal_upsampler: Option<&'a crate::upsampler::LatentUpsampler>,
    pub video_ctx: &'a Tensor,
    pub audio_ctx: &'a Tensor,
    pub audio_grid: &'a Tensor,
    pub audio_frames: usize,
}

/// The DFR request shape (twin of the MLX `DfrRequest`; `canvas_frames` / `keyframe_positions`
/// come from [`gen_core_dfr::resolve_canvas`] over the pre-padding `requested_frames`).
pub struct DfrRequest<'a> {
    pub canvas_frames: i64,
    pub requested_frames: i64,
    pub keyframe_positions: &'a [i64],
    pub geometry: crate::pipeline::TwoStageGeometry,
    pub fps: f32,
    pub seed: u64,
    pub temporal_upsample_rounds: u32,
    /// `Some(downscale)` appends the reserved half-res stage-1 video as the detailing IC-LoRA
    /// reference in stage 2; the detailing LoRA weights themselves are installed by the engine's
    /// adapter layer, scoped to the stage-2 pass.
    pub detailing_downscale: Option<i64>,
    /// Replace-latent image conditioning (I2V / first-last-frame), empty for T2V.
    pub video_keyframes: &'a [DfrStageKeyframe],
}

/// The DFR pipeline output (video latent, the **stage-1** audio latent — the shipped audio — the
/// final pixel-frame count and the playback fps).
pub struct DfrOutput {
    pub video_latent: Tensor,
    pub audio_latent: Tensor,
    pub num_frames: i64,
    pub playback_fps: f32,
}

/// The full DFR latent pipeline — candle twin of the MLX `generate_dfr_av_latents`; see that doc
/// comment for stage semantics (deterministic distilled stages, ancestral tiles, stage-1 audio
/// shipped, pass 0 for stage 1 + tiles, pass 1 for the detailing stage 2).
pub fn generate_dfr_av_latents(
    parts: &DfrComponents<'_>,
    req: &DfrRequest<'_>,
    cancel: &CancelFlag,
    on_progress: &mut dyn FnMut(Progress),
) -> candle_gen::Result<DfrOutput> {
    use crate::config::{STAGE1_SIGMAS, STAGE2_SIGMAS};
    use crate::pipeline::{
        create_audio_noise, create_noise, denoise_av_conditioned, renoise, unflatten_latent,
    };

    let rounds = req.temporal_upsample_rounds;
    if rounds > gen_core_dfr::MAX_TEMPORAL_UPSAMPLE_ROUNDS {
        return Err(candle_gen::CandleError::Msg(format!(
            "ltx dfr: temporal_upsample_rounds must be 0..={}, got {rounds}",
            gen_core_dfr::MAX_TEMPORAL_UPSAMPLE_ROUNDS
        )));
    }
    if rounds > 0 && parts.temporal_upsampler.is_none() {
        return Err(candle_gen::CandleError::Msg(
            "ltx dfr: temporal_upsample_rounds > 0 requires the temporal latent upsampler \
             component"
                .into(),
        ));
    }
    if req.keyframe_positions.is_empty() {
        return Err(candle_gen::CandleError::Msg(
            "ltx dfr: the DFR canvas resolved no keyframe positions".into(),
        ));
    }
    let temporal_scale = TEMPORAL_SCALE as i64;
    let g = req.geometry;
    let expected_lf = ((req.canvas_frames - 1) / temporal_scale + 1) as usize;
    if g.t != expected_lf {
        return Err(candle_gen::CandleError::Msg(format!(
            "ltx dfr: geometry t={} must match the canvas' {expected_lf} latent frames",
            g.t
        )));
    }
    let device = parts.video_ctx.device().clone();
    let normalize_upsample =
        |up: &crate::upsampler::LatentUpsampler, x: &Tensor| -> Result<Tensor> {
            parts
                .vae
                .normalize_latents(&up.forward(&parts.vae.denormalize_latents(x)?)?)
        };

    // --- Stage 1: half-res base + keyframe slots -------------------------------------------------
    let vnoise1 = create_noise(req.seed, g.t, g.h1, g.w1, &device)?;
    let anoise1 = create_audio_noise(req.seed.wrapping_add(2), parts.audio_frames, &device)?;
    let grid1 = crate::rope::create_position_grid(g.t, g.h1, g.w1, req.fps, &device)?;
    let mut state = if req.video_keyframes.is_empty() {
        VideoTokenState::base(&vnoise1, &grid1)?
    } else {
        let zeros = Tensor::zeros_like(&vnoise1)?;
        let borrowed: Vec<crate::conditioning::Keyframe> = req
            .video_keyframes
            .iter()
            .map(|k| crate::conditioning::Keyframe {
                latent: &k.stage1,
                frame_idx: k.frame_idx,
                strength: k.strength,
            })
            .collect();
        let i2v = crate::conditioning::apply_keyframes(&zeros, &borrowed)?
            .noised(&vnoise1, STAGE1_SIGMAS[0])?;
        VideoTokenState::from_i2v(&i2v, &grid1)?
    };
    state = append_generated_keyframe_slots(
        &state,
        req.keyframe_positions,
        None,
        req.canvas_frames,
        g.h1,
        g.w1,
        SPATIAL_SCALE as i64,
        req.fps,
    )?;
    state.latent = noise_slot_tokens(&state, STAGE1_SIGMAS[0], req.seed.wrapping_add(11))?;

    parts.dit.set_adapter_pass(0);
    let mut on_forward = || Ok(());
    let (state, audio_s1) = denoise_av_conditioned(
        parts.dit,
        &state,
        &anoise1,
        parts.video_ctx,
        parts.audio_ctx,
        parts.audio_frames,
        parts.audio_grid,
        &STAGE1_SIGMAS,
        cancel,
        &mut on_forward,
        on_progress,
    )?;
    let stage1_audio_latent = audio_s1.clone();
    let grid_tokens = state.latent.narrow(1, 0, state.target_tokens)?;
    let reserved_half_res = unflatten_latent(&grid_tokens, g.t, g.h1, g.w1)?;
    let slot_keyframes = crate::conditioning::take_generated_keyframes(&state, g.h1, g.w1)?;

    // Spatial x2 of video and slots (slots' K rides the frame axis, untouched by the spatial
    // checkpoint).
    let upscaled_video = normalize_upsample(parts.spatial_upsampler, &reserved_half_res)?;
    let upscaled_slots = normalize_upsample(parts.spatial_upsampler, &slot_keyframes)?;

    // --- Stage 2: full-res detailing -------------------------------------------------------------
    let s2_entry = STAGE2_SIGMAS[0];
    let vnoise2 = create_noise(req.seed.wrapping_add(1), g.t, g.h2, g.w2, &device)?;
    let anoise2 = create_audio_noise(req.seed.wrapping_add(3), parts.audio_frames, &device)?;
    let grid2 = crate::rope::create_position_grid(g.t, g.h2, g.w2, req.fps, &device)?;
    let mut state2 = if req.video_keyframes.is_empty() {
        let renoised = renoise(&upscaled_video, &vnoise2, s2_entry)?;
        VideoTokenState::base(&renoised, &grid2)?
    } else {
        let borrowed: Vec<crate::conditioning::Keyframe> = req
            .video_keyframes
            .iter()
            .map(|k| crate::conditioning::Keyframe {
                latent: &k.stage2,
                frame_idx: k.frame_idx,
                strength: k.strength,
            })
            .collect();
        let i2v = crate::conditioning::apply_keyframes(&upscaled_video, &borrowed)?
            .noised(&vnoise2, s2_entry)?;
        VideoTokenState::from_i2v(&i2v, &grid2)?
    };
    state2 = append_generated_keyframe_slots(
        &state2,
        req.keyframe_positions,
        Some(&upscaled_slots),
        req.canvas_frames,
        g.h2,
        g.w2,
        SPATIAL_SCALE as i64,
        req.fps,
    )?;
    state2.latent = noise_slot_tokens(&state2, s2_entry, req.seed.wrapping_add(13))?;
    if let Some(downscale) = req.detailing_downscale {
        state2 = crate::conditioning::append_reference_latent(
            &state2,
            &reserved_half_res,
            downscale,
            1.0,
            req.fps,
        )?;
    }
    let audio2 = renoise(&stage1_audio_latent, &anoise2, s2_entry)?;

    parts.dit.set_adapter_pass(1);
    let mut on_forward = || Ok(());
    let (state2, _audio2) = denoise_av_conditioned(
        parts.dit,
        &state2,
        &audio2,
        parts.video_ctx,
        parts.audio_ctx,
        parts.audio_frames,
        parts.audio_grid,
        &STAGE2_SIGMAS,
        cancel,
        &mut on_forward,
        on_progress,
    )?;
    let grid_tokens2 = state2.latent.narrow(1, 0, state2.target_tokens)?;
    let mut video = unflatten_latent(&grid_tokens2, g.t, g.h2, g.w2)?;
    let carry_keyframes = crate::conditioning::take_generated_keyframes(&state2, g.h2, g.w2)?;

    // --- Temporal rounds -------------------------------------------------------------------------
    let mut num_frames = req.canvas_frames;
    let mut playback_fps = req.fps;
    if rounds > 0 {
        let upsampler = parts.temporal_upsampler.expect("validated above");
        // Temporal tiles run the base (stage-1) adapter pass, like the reference's non-detailing
        // stage.
        parts.dit.set_adapter_pass(0);
        // Replace-latent image anchors ride into every round, re-attached tile-locally; a
        // multi-latent-frame conditioning has no per-tile replace-latent projection — typed
        // refusal, never a silent drop (twin of the mlx arm).
        let round_images: Vec<DfrRoundImage> = req
            .video_keyframes
            .iter()
            .map(|keyframe| {
                let cf = keyframe.stage2.dim(2)?;
                if cf != 1 {
                    return Err(candle_gen::CandleError::Msg(format!(
                        "ltx dfr: a {cf}-latent-frame replace-latent conditioning cannot ride \
                         temporal_upsample_rounds > 0 (single-frame image anchors only)"
                    )));
                }
                Ok(DfrRoundImage {
                    pixel_position: keyframe.frame_idx as i64 * TEMPORAL_SCALE as i64,
                    latent: keyframe.stage2.clone(),
                    strength: keyframe.strength,
                })
            })
            .collect::<candle_gen::Result<_>>()?;
        let mut upsample = |v: &Tensor| normalize_upsample(upsampler, v);
        let mut denoise_tile = |job: &DfrTileJob| -> candle_gen::Result<DfrTileResult> {
            let t_tile = job.tile.latent_frames();
            let positions =
                crate::rope::create_position_grid(t_tile, g.h2, g.w2, job.cond_fps, &device)?;
            let mut on_forward = || Ok(());
            let mut tile_progress = |_p: Progress| {};
            denoise_dfr_tile(
                parts.dit,
                job,
                parts.video_ctx,
                &positions,
                cancel,
                &mut on_forward,
                &mut tile_progress,
            )
        };
        let out = run_temporal_rounds(
            &video,
            req.keyframe_positions,
            &carry_keyframes,
            &round_images,
            num_frames,
            req.fps,
            req.seed,
            rounds,
            &mut upsample,
            &mut denoise_tile,
        )?;
        video = out.video_latent;
        num_frames = out.num_frames;
        playback_fps = out.fps;
    }

    let (video, num_frames) =
        trim_to_target_frames(&video, num_frames, req.requested_frames, rounds)?;
    Ok(DfrOutput {
        video_latent: video,
        audio_latent: stage1_audio_latent,
        num_frames,
        playback_fps,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use candle_gen::candle_core::Device;
    use std::cell::RefCell;

    fn dev() -> Device {
        Device::Cpu
    }

    /// `(1, C, T, 1, 1)` latent whose frame t is the constant `base + t`.
    fn ramp_latent(c: usize, t: usize, base: f32) -> Tensor {
        let mut data = Vec::with_capacity(c * t);
        for _ in 0..c {
            for frame in 0..t {
                data.push(base + frame as f32);
            }
        }
        Tensor::from_vec(data, (1, c, t, 1, 1), &dev()).unwrap()
    }

    fn const_latent(c: usize, t: usize, v: f32) -> Tensor {
        Tensor::from_vec(vec![v; c * t], (1, c, t, 1, 1), &dev()).unwrap()
    }

    fn frame_value(x: &Tensor, t: usize) -> f32 {
        x.flatten_all().unwrap().to_vec1::<f32>().unwrap()[t]
    }

    fn fake_upsample(x: &Tensor) -> Result<Tensor> {
        let t = x.dim(2)?;
        let mut idx: Vec<u32> = Vec::new();
        for i in 0..t as u32 {
            idx.push(i);
            if (i as usize) + 1 < t {
                idx.push(i);
            }
        }
        take_frames(x, &idx)
    }

    #[derive(Clone, Debug)]
    struct Call {
        round: u32,
        tile_index: usize,
        seed: u64,
        cond_fps: f32,
        anchors_global: Vec<i64>,
        anchor_first_value: Option<f32>,
        images_local: Vec<(usize, f32)>,
    }

    fn recording_denoiser(
        calls: &RefCell<Vec<Call>>,
    ) -> impl FnMut(&DfrTileJob) -> candle_gen::Result<DfrTileResult> + '_ {
        move |job| {
            let (_b, c, t, _h, _w) = job.tile_video.dims5()?;
            calls.borrow_mut().push(Call {
                round: job.round,
                tile_index: job.tile_index,
                seed: job.noise_seed,
                cond_fps: job.cond_fps,
                anchors_global: job
                    .anchor_positions_local
                    .iter()
                    .map(|p| p + job.tile.pixel_start)
                    .collect(),
                anchor_first_value: job.anchor_latents.as_ref().map(|a| frame_value(a, 0)),
                images_local: job
                    .image_keyframes
                    .iter()
                    .map(|image| (image.local_latent_index, frame_value(&image.latent, 0)))
                    .collect(),
            });
            let generated_keyframes = if job.slot_positions_local.is_empty() {
                None
            } else {
                Some(const_latent(
                    c,
                    job.slot_positions_local.len(),
                    (1000 * job.round as i32 + job.tile_index as i32) as f32,
                ))
            };
            Ok(DfrTileResult {
                latent: const_latent(
                    c,
                    t,
                    (100 * job.round as i32) as f32 + job.tile_index as f32 + 1.0,
                ),
                generated_keyframes,
            })
        }
    }

    fn stage2_carry(c: usize) -> (Vec<i64>, Tensor) {
        let positions: Vec<i64> = vec![24, 48, 72, 96, 120];
        let latents = const_latent(c, positions.len(), 7.0);
        (positions, latents)
    }

    /// Twin of the MLX `round_count_is_honoured_not_inert_above_1` — rounds=2 must produce the
    /// deeper canvas AND the round-2 tile calls, with distinct per-tile seeds and the capped
    /// conditioning fps.
    #[test]
    fn round_count_is_honoured_not_inert_above_1() {
        let video = ramp_latent(2, 16, 0.0);
        let (carry_pos, carry_kf) = stage2_carry(2);

        for (rounds, want_calls, want_t, want_frames, want_fps) in [
            (1u32, vec![(1u32, 2usize)], 31usize, 241i64, 48.0f32),
            (2, vec![(1, 2), (2, 4)], 61, 481, 96.0),
        ] {
            let calls = RefCell::new(Vec::new());
            let out = run_temporal_rounds(
                &video,
                &carry_pos,
                &carry_kf,
                &[],
                121,
                24.0,
                77,
                rounds,
                &mut fake_upsample,
                &mut recording_denoiser(&calls),
            )
            .unwrap();
            assert_eq!(out.video_latent.dim(2).unwrap(), want_t, "rounds={rounds}");
            assert_eq!(out.num_frames, want_frames);
            assert_eq!(out.fps, want_fps);
            let calls = calls.borrow();
            for (round, count) in want_calls {
                assert_eq!(
                    calls.iter().filter(|c| c.round == round).count(),
                    count,
                    "rounds={rounds}: round {round} tile count"
                );
            }
            let mut seeds: Vec<u64> = calls.iter().map(|c| c.seed).collect();
            seeds.sort_unstable();
            seeds.dedup();
            assert_eq!(seeds.len(), calls.len(), "per-tile seeds must be distinct");
            for c in calls.iter() {
                let want = if c.round == 1 { 48.0 } else { 60.0 };
                assert_eq!(c.cond_fps, want, "round {} cond fps", c.round);
            }
        }
    }

    /// Twin of the MLX seam test: with per-tile constant fills, every latent frame of the stitched
    /// canvas carries the constant of the tile that owns it, handing over exactly at the seam
    /// latent (tile 0 keeps latents 0..=18, tile 1's kept run is 19..31).
    #[test]
    fn stitch_hands_over_exactly_at_the_seam_latent() {
        let video = ramp_latent(1, 16, 0.0);
        let (carry_pos, carry_kf) = stage2_carry(1);
        let calls = RefCell::new(Vec::new());
        let out = run_temporal_rounds(
            &video,
            &carry_pos,
            &carry_kf,
            &[],
            121,
            24.0,
            5,
            1,
            &mut fake_upsample,
            &mut recording_denoiser(&calls),
        )
        .unwrap();
        assert_eq!(out.video_latent.dim(2).unwrap(), 31);
        for t in 0..=18 {
            assert_eq!(frame_value(&out.video_latent, t), 101.0, "latent {t}");
        }
        for t in 19..31 {
            assert_eq!(frame_value(&out.video_latent, t), 102.0, "latent {t}");
        }
    }

    /// Twin of the MLX inter-round continuity test: round-2 anchors must include seams that exist
    /// only because round-1 slots were merged into the carry bag, carrying round-1 tile content.
    #[test]
    fn round2_anchors_carry_round1_slot_content() {
        let video = ramp_latent(1, 16, 0.0);
        let (carry_pos, carry_kf) = stage2_carry(1);
        let calls = RefCell::new(Vec::new());
        run_temporal_rounds(
            &video,
            &carry_pos,
            &carry_kf,
            &[],
            121,
            24.0,
            5,
            2,
            &mut fake_upsample,
            &mut recording_denoiser(&calls),
        )
        .unwrap();
        let calls = calls.borrow();
        let round2: Vec<&Call> = calls.iter().filter(|c| c.round == 2).collect();
        assert_eq!(round2.len(), 4);
        let mid_derived: Vec<i64> = vec![48, 144, 240, 336, 432];
        let mut seen_mid_derived = false;
        for call in &round2 {
            assert!(
                !call.anchors_global.is_empty(),
                "round-2 tiles are anchored"
            );
            for a in &call.anchors_global {
                assert_eq!(a % 48, 0, "round-2 anchor {a} must be a merged-bag seam");
            }
            if call.anchors_global.iter().any(|a| mid_derived.contains(a)) {
                seen_mid_derived = true;
            }
        }
        assert!(seen_mid_derived, "carry-forward merge is not flowing");
        let tile0 = round2.iter().find(|c| c.tile_index == 0).unwrap();
        assert_eq!(tile0.anchors_global[0], 48);
        let v = tile0.anchor_first_value.unwrap();
        assert!(
            v == 1000.0 || v == 1001.0,
            "round-2 anchor at seam 48 must carry round-1 slot content, got {v}"
        );
    }

    /// Twin of the MLX image re-attachment test: image anchors land only on the tiles whose
    /// window contains their round-scaled position, remapped tile-locally.
    #[test]
    fn images_reattach_tile_locally_per_round() {
        let video = ramp_latent(1, 16, 0.0);
        let (carry_pos, carry_kf) = stage2_carry(1);
        let images = [
            DfrRoundImage {
                pixel_position: 0,
                latent: const_latent(1, 1, 41.0),
                strength: 1.0,
            },
            DfrRoundImage {
                pixel_position: 120,
                latent: const_latent(1, 1, 42.0),
                strength: 0.8,
            },
        ];
        for rounds in [1u32, 2] {
            let calls = RefCell::new(Vec::new());
            run_temporal_rounds(
                &video,
                &carry_pos,
                &carry_kf,
                &images,
                121,
                24.0,
                7,
                rounds,
                &mut fake_upsample,
                &mut recording_denoiser(&calls),
            )
            .unwrap();
            let calls = calls.borrow();
            let last: Vec<&Call> = calls.iter().filter(|c| c.round == rounds).collect();
            for (index, call) in last.iter().enumerate() {
                let want: Vec<(usize, f32)> = if index == 0 {
                    vec![(0, 41.0)]
                } else if index == last.len() - 1 {
                    vec![(18, 42.0)]
                } else {
                    vec![]
                };
                assert_eq!(
                    call.images_local, want,
                    "rounds={rounds} tile {index}: window-local image re-attachment"
                );
            }
        }
    }

    #[test]
    fn slot_initials_pick_nearest_latent_frame() {
        let window = ramp_latent(1, 19, 0.0);
        let init = slot_initials_from_video(&window, &[24, 72, 120], 8).unwrap();
        assert_eq!(init.dims(), &[1, 1, 3, 1, 1]);
        assert_eq!(frame_value(&init, 0), 3.0);
        assert_eq!(frame_value(&init, 1), 9.0);
        assert_eq!(frame_value(&init, 2), 15.0);
    }

    #[test]
    fn carry_forward_merge_orders_and_validates() {
        let anchors = const_latent(1, 2, 5.0);
        let slots = const_latent(1, 1, 9.0);
        let (positions, latents) =
            merge_carry_forward_keyframes(&[48, 96], Some(&anchors), &[72], Some(&slots)).unwrap();
        assert_eq!(positions, vec![48, 72, 96]);
        assert_eq!(frame_value(&latents, 0), 5.0);
        assert_eq!(frame_value(&latents, 1), 9.0);
        assert_eq!(frame_value(&latents, 2), 5.0);
        assert!(merge_carry_forward_keyframes(&[48], None, &[], None).is_err());
        assert!(merge_carry_forward_keyframes(&[], None, &[], None).is_err());
    }

    #[test]
    fn missing_slot_keyframes_is_a_hard_error() {
        let video = ramp_latent(1, 16, 0.0);
        let (carry_pos, carry_kf) = stage2_carry(1);
        let mut no_slots = |job: &DfrTileJob| -> candle_gen::Result<DfrTileResult> {
            let (_b, c, t, _h, _w) = job.tile_video.dims5()?;
            Ok(DfrTileResult {
                latent: const_latent(c, t, 1.0),
                generated_keyframes: None,
            })
        };
        let err = run_temporal_rounds(
            &video,
            &carry_pos,
            &carry_kf,
            &[],
            121,
            24.0,
            5,
            1,
            &mut fake_upsample,
            &mut no_slots,
        )
        .unwrap_err();
        assert!(
            err.to_string().contains("produced no keyframe slots"),
            "got: {err}"
        );
    }

    #[test]
    fn trim_to_target_frames_lands_on_latent_boundary() {
        let canvas = ramp_latent(1, 21, 0.0);
        let (trimmed, frames) = trim_to_target_frames(&canvas, 161, 153, 0).unwrap();
        assert_eq!(frames, 153);
        assert_eq!(trimmed.dim(2).unwrap(), 20);
        let (same, frames) = trim_to_target_frames(&canvas, 161, 161, 0).unwrap();
        assert_eq!((frames, same.dim(2).unwrap()), (161, 21));
        assert!(trim_to_target_frames(&canvas, 161, 169, 0).is_err());
    }

    /// Slot append marks exactly its run; given keyframes stay unmarked (candle twins of the MLX
    /// conditioning assertions, at the same seams).
    #[test]
    fn slot_and_keyframe_appends_mark_correctly() {
        let noise = Tensor::ones(
            (1usize, 2, 2, 1, 1),
            candle_gen::candle_core::DType::F32,
            &dev(),
        )
        .unwrap();
        let positions = Tensor::zeros(
            (1usize, 3, 2, 2),
            candle_gen::candle_core::DType::F32,
            &dev(),
        )
        .unwrap();
        let st = VideoTokenState::base(&noise, &positions).unwrap();
        let out = append_generated_keyframe_slots(&st, &[5, 9], None, 17, 1, 1, 32, 24.0).unwrap();
        let layout = out.generated_keyframe_layout.as_ref().unwrap();
        assert_eq!((layout.first_token, layout.tokens_per_keyframe), (2, 1));
        let mask = out
            .keyframes_mask
            .as_ref()
            .expect("slots must mark")
            .flatten_all()
            .unwrap()
            .to_vec1::<f32>()
            .unwrap();
        assert_eq!(mask, vec![0.0, 0.0, 1.0, 1.0]);
        // Read the (still-initial) slots back through the layout.
        let init = Tensor::from_vec(vec![3.0f32, 4.0, 5.0, 6.0], (1, 2, 2, 1, 1), &dev()).unwrap();
        let seeded =
            append_generated_keyframe_slots(&st, &[5, 9], Some(&init), 17, 1, 1, 32, 24.0).unwrap();
        let back = crate::conditioning::take_generated_keyframes(&seeded, 1, 1).unwrap();
        assert_eq!(back.dims(), &[1, 2, 2, 1, 1]);
        assert_eq!(
            back.flatten_all().unwrap().to_vec1::<f32>().unwrap(),
            vec![3.0, 4.0, 5.0, 6.0]
        );
        // Given single-frame keyframes never mark.
        let kf = Tensor::from_vec(vec![7.0f32, 8.0], (1, 2, 1, 1, 1), &dev()).unwrap();
        let given = append_single_frame_keyframes(&st, &kf, &[6], 0.95, 32, 24.0).unwrap();
        assert!(given.keyframes_mask.is_none());
        assert!(given.generated_keyframe_layout.is_none());
        // And the reference latent never marks either.
        let refl = Tensor::ones(
            (1usize, 2, 1, 1, 1),
            candle_gen::candle_core::DType::F32,
            &dev(),
        )
        .unwrap();
        let with_ref =
            crate::conditioning::append_reference_latent(&st, &refl, 2, 1.0, 24.0).unwrap();
        assert!(with_ref.keyframes_mask.is_none());

        // --- Numeric parity with the mlx twins (sc-18789 review) --------------------------------
        // 1. Slot RoPE positions span exactly one pixel frame — `[t, t+1)/fps`.
        let slot_pos = crate::conditioning::single_frame_positions(1, 1, 9, 32, 24.0, &dev())
            .unwrap()
            .flatten_all()
            .unwrap()
            .to_vec1::<f32>()
            .unwrap();
        assert_eq!(
            slot_pos,
            vec![9.0 / 24.0, 10.0 / 24.0, 0.0, 32.0, 0.0, 32.0],
            "single-pixel-frame span on the frame axis; x32 spatial spans"
        );
        // 2. Slot initials seed the NOISY latent while clean stays zero.
        let lat = seeded
            .latent
            .flatten_all()
            .unwrap()
            .to_vec1::<f32>()
            .unwrap();
        // Token layout is (B, S, C): slot 0 = (c0=3, c1=5), slot 1 = (c0=4, c1=6).
        assert_eq!(
            &lat[4..8],
            &[3.0, 5.0, 4.0, 6.0],
            "initials seed the latent"
        );
        let clean = seeded
            .clean_latent
            .flatten_all()
            .unwrap()
            .to_vec1::<f32>()
            .unwrap();
        assert_eq!(&clean[4..8], &[0.0, 0.0, 0.0, 0.0], "slot clean stays zero");
        // 3. Reference latent: strength 1.0 pins fully (mask 0) and its spatial positions ride the
        // x(32·d) grid — the appended token's height span is [0, 64) at d = 2.
        let dm = with_ref
            .denoise_mask
            .flatten_all()
            .unwrap()
            .to_vec1::<f32>()
            .unwrap();
        assert_eq!(dm[2], 0.0, "strength 1.0 pins the reference fully");
        let rp = with_ref
            .positions
            .flatten_all()
            .unwrap()
            .to_vec1::<f32>()
            .unwrap();
        // positions (1, 3, 3, 2): height axis starts at index 3·2; appended token is index 2.
        assert_eq!((rp[3 * 2 + 2 * 2], rp[3 * 2 + 2 * 2 + 1]), (0.0, 64.0));
    }

    /// Candle twin of the mlx real-anchor-list test: the non-first tile's anchor list (first
    /// element exactly 0, the shared seam) must flow through `append_single_frame_keyframes` —
    /// the production `denoise_dfr_tile` path that the position-0 guard broke pre-review.
    #[test]
    fn real_tile_anchor_list_flows_through_the_keyframe_append() {
        let (n, _, pos) = gen_core_dfr::resolve_canvas(121, 8).unwrap();
        let seams: Vec<i64> = pos.iter().map(|p| 2 * p).collect();
        let n1 = gen_core_dfr::temporal_upsampled_frames(n);
        let tiles =
            gen_core_dfr::tile_ranges(&seams, n1, 2, 8, gen_core_dfr::TILE_LEAD_SEGMENTS).unwrap();
        let tile1 = &tiles[1];
        let local =
            gen_core_dfr::remap_positions_to_local(&tile1.anchor_kf_global, tile1.pixel_start);
        assert_eq!(local[0], 0, "the non-first tile anchors its window start");

        let noise = Tensor::ones(
            (1usize, 2, tile1.latent_frames(), 1, 1),
            candle_gen::candle_core::DType::F32,
            &dev(),
        )
        .unwrap();
        let positions = Tensor::zeros(
            (1usize, 3, tile1.latent_frames(), 2),
            candle_gen::candle_core::DType::F32,
            &dev(),
        )
        .unwrap();
        let st = VideoTokenState::base(&noise, &positions).unwrap();
        let anchors = Tensor::ones(
            (1usize, 2, local.len(), 1, 1),
            candle_gen::candle_core::DType::F32,
            &dev(),
        )
        .unwrap();
        let out = append_single_frame_keyframes(
            &st,
            &anchors,
            &local,
            gen_core_dfr::ANCHOR_KEYFRAME_STRENGTH,
            32,
            48.0,
        )
        .expect("the real tile-1 anchor list must be appendable");
        let dm = out
            .denoise_mask
            .flatten_all()
            .unwrap()
            .to_vec1::<f32>()
            .unwrap();
        assert!((dm[tile1.latent_frames()] - 0.05).abs() < 1e-6);
    }
}
