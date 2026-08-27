//! The LTX-2.5 **DFR** pipeline (sc-18789) — generated keyframe slots, the full-resolution
//! detailing pass, and tiled temporal upsampling rounds. Port of the reference
//! `ltx_pipelines/dfr_pipeline.py` (Lightricks/LTX-2 v1.2.0 @ `d1511477`) over this crate's
//! token-native conditioning ([`crate::conditioning`]) and denoise loops ([`crate::pipeline`]),
//! with the shared integer geometry in [`mlx_gen::gen_core::ltx_dfr`].
//!
//! Layering: [`run_temporal_rounds`] owns every piece of round/tile/seam arithmetic — window
//! slicing, anchor gathering, slot seeding, per-tile noise seeds, the latent stitch, slot
//! dedup and the carry-forward merge — behind two injected closures (the temporal upsampler and
//! the per-tile denoise), so the exact code path production runs is drivable by tests without the
//! 22B DiT. [`denoise_dfr_tile`] is the production tile closure body: it assembles the tile's
//! token state (anchors → hard keyframes, slots → generated-keyframe slots) and runs the
//! rectified-flow ancestral loop at [`gen_core_dfr::TEMPORAL_ANCESTRAL_ETA`].

use mlx_rs::ops::{broadcast_to, concatenate_axis, multiply};
use mlx_rs::Array;

use mlx_gen::gen_core::ltx_dfr as gen_core_dfr;
use mlx_gen::gen_core::ltx_dfr::DfrTileRange;
use mlx_gen::{CancelFlag, Error, Result};

use crate::conditioning::{
    append_generated_keyframe_slots, append_single_frame_keyframes, token_timesteps,
    VideoTokenState,
};
use crate::pipeline::{denoise_tokens_rf_ancestral, TEMPORAL_SIGMAS};
use crate::positions::{SPATIAL_SCALE, TEMPORAL_SCALE};
use crate::transformer::AvDiT;

/// Gather latent frames (axis 2) by index.
fn take_frames(x: &Array, indices: &[i32]) -> Result<Array> {
    Ok(x.take_axis(Array::from_slice(indices, &[indices.len() as i32]), 2)?)
}

/// Temporal window `x[:, :, start..end]`.
fn frame_window(x: &Array, start: usize, end: usize) -> Result<Array> {
    let idx: Vec<i32> = (start as i32..end as i32).collect();
    take_frames(x, &idx)
}

/// Stack the nearest video latent frames as `(B, C, K, H, W)` slot seeds
/// (`dfr_pipeline._slot_initials_from_video`): tile-local pixel `position / temporal_scale`,
/// rounded half-to-even like the reference's `round()`, clamped to the window.
pub fn slot_initials_from_video(
    video_latent: &Array,
    positions_local: &[i64],
    temporal_scale: i64,
) -> Result<Array> {
    let frames = video_latent.shape()[2];
    let idx: Vec<i32> = positions_local
        .iter()
        .map(|&p| {
            ((p as f64 / temporal_scale as f64).round_ties_even() as i64)
                .clamp(0, i64::from(frames) - 1) as i32
        })
        .collect();
    take_frames(video_latent, &idx)
}

/// Concatenate tile video latents along T, each contributing `latent[drop_latent_prefix..]`
/// (`dfr_layout.stitch_tile_latents`). The prefix drop is the seam handover: the previous tile
/// keeps the shared seam latent, so a wrong prefix double-writes or gaps the seam — the checks
/// here sit on the exact tensors where that fault appears.
pub fn stitch_tile_latents(tile_latents: &[Array], ranges: &[DfrTileRange]) -> Result<Array> {
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
        let sh = latent.shape();
        if sh.len() != 5 {
            return Err(Error::Msg(format!(
                "ltx dfr: expected tile latent (B, C, T, H, W), got {sh:?}"
            )));
        }
        let expected_t = tile.latent_frames();
        if sh[2] as usize != expected_t {
            return Err(Error::Msg(format!(
                "ltx dfr: tile latent T={} != expected {expected_t} for range [{}, {})",
                sh[2], tile.latent_start, tile.latent_end_exclusive
            )));
        }
        if tile.drop_latent_prefix >= sh[2] as usize {
            return Err(Error::Msg(format!(
                "ltx dfr: drop_latent_prefix={} invalid for tile T={}",
                tile.drop_latent_prefix, sh[2]
            )));
        }
        pieces.push(frame_window(latent, tile.drop_latent_prefix, sh[2] as usize)?);
    }
    concatenate_axis(&pieces.iter().collect::<Vec<_>>(), 2).map_err(Into::into)
}

/// Build the next round's anchor bag: carried keyframe stills plus this round's denoised slots
/// (`dfr_pipeline._merge_carry_forward_keyframes`). Positions are on the current round's pixel
/// grid; the next round remaps (×2). Anchor and slot positions are disjoint by construction
/// (seams vs. segment midpoints); on a collision the slot's version wins, like the reference's
/// insertion order.
pub fn merge_carry_forward_keyframes(
    anchor_positions: &[i64],
    anchor_latents: Option<&Array>,
    slot_positions: &[i64],
    slot_latents: Option<&Array>,
) -> Result<(Vec<i64>, Array)> {
    let mut by_position: std::collections::BTreeMap<i64, (u8, i32)> = Default::default();
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
        if latents.shape()[2] as usize != positions.len() {
            return Err(Error::Msg(format!(
                "ltx dfr: carry-forward latents K={} != {} positions",
                latents.shape()[2],
                positions.len()
            )));
        }
        for (index, &position) in positions.iter().enumerate() {
            by_position.insert(position, (which as u8, index as i32));
        }
    }
    if by_position.is_empty() {
        return Err(Error::Msg(
            "ltx dfr: carry-forward keyframe bag is empty".into(),
        ));
    }
    let ordered: Vec<i64> = by_position.keys().copied().collect();
    let mut frames = Vec::with_capacity(ordered.len());
    for (_, (which, index)) in &by_position {
        let src = if *which == 0 {
            anchor_latents.expect("checked above")
        } else {
            slot_latents.expect("checked above")
        };
        frames.push(take_frames(src, &[*index])?);
    }
    Ok((
        ordered,
        concatenate_axis(&frames.iter().collect::<Vec<_>>(), 2)?,
    ))
}

/// One tile's denoise inputs, in tile-local coordinates.
pub struct DfrTileJob<'a> {
    /// 1-based temporal round.
    pub round: u32,
    /// Tile index within the round.
    pub tile_index: usize,
    pub tile: &'a DfrTileRange,
    /// `(B, C, T_tile, H, W)` — the window's slice of the temporally upsampled video latent.
    pub tile_video: Array,
    /// Hard-keyframe anchors inside the window, tile-local pixel positions (all `> 0`).
    pub anchor_positions_local: Vec<i64>,
    /// `(B, C, Ka, H, W)` anchor latents, ordered like `anchor_positions_local` (`None` iff empty).
    pub anchor_latents: Option<Array>,
    /// Generated-slot positions this window invents, tile-local.
    pub slot_positions_local: Vec<i64>,
    /// `(B, C, Ks, H, W)` slot seeds from the nearest video latents (`None` iff no slots).
    pub slot_initials: Option<Array>,
    /// Pixel-frame count of the window.
    pub local_frames: i64,
    /// Conditioning fps for RoPE time (playback fps capped at
    /// [`gen_core_dfr::MAX_CONDITIONING_FPS`]).
    pub cond_fps: f32,
    /// The tile's ancestral noise seed (`seed + 1000·round + tile_index` — tiles are positionally
    /// identical, so a shared seed would inject byte-identical noise into every one).
    pub noise_seed: u64,
}

/// One tile's denoise result.
pub struct DfrTileResult {
    /// `(B, C, T_tile, H, W)` denoised window.
    pub latent: Array,
    /// `(B, C, Ks, H, W)` denoised slot keyframes — required when the job carried slots.
    pub generated_keyframes: Option<Array>,
}

/// The output of [`run_temporal_rounds`].
#[derive(Debug)]
pub struct TemporalRoundsOutput {
    /// `(B, C, T, H, W)` stitched video latent after the final round.
    pub video_latent: Array,
    /// Pixel-frame count after the final round (`N → 2(N−1)+1` per round).
    pub num_frames: i64,
    /// Playback fps after the final round (doubles per round).
    pub fps: f32,
}

/// Tiled temporal x2 upsampling rounds (`dfr_pipeline.__call__`'s round loop): each round
/// temporally upsamples the latent, splits the canvas into `2^round` keyframe-seam tiles, seeds
/// mid-segment slots per tile, denoises every tile (ancestral, per-tile seed), stitches on the
/// seam handover, and folds the denoised slots into the next round's anchor bag.
///
/// `upsample` maps `(B,C,T,H,W) → (B,C,2T−1,H,W)` (the sc-18773 temporal upsampler wrapped in the
/// caller's normalization); `denoise_tile` runs one window (production: [`denoise_dfr_tile`]).
/// Both are injected so every seam/round decision in this function is testable without the DiT —
/// the checks live on the exact tensors where a seam fault would appear.
#[allow(clippy::too_many_arguments)]
pub fn run_temporal_rounds(
    video_latent: &Array,
    carry_positions: &[i64],
    carry_keyframes: &Array,
    num_frames: i64,
    fps: f32,
    seed: u64,
    rounds: u32,
    upsample: &mut dyn FnMut(&Array) -> Result<Array>,
    denoise_tile: &mut dyn FnMut(&DfrTileJob) -> Result<DfrTileResult>,
) -> Result<TemporalRoundsOutput> {
    if rounds > gen_core_dfr::MAX_TEMPORAL_UPSAMPLE_ROUNDS {
        return Err(Error::Msg(format!(
            "ltx dfr: temporal_upsample_rounds must be 0..={}, got {rounds}",
            gen_core_dfr::MAX_TEMPORAL_UPSAMPLE_ROUNDS
        )));
    }
    let temporal_scale = TEMPORAL_SCALE;
    let mut video = video_latent.clone();
    let mut num_frames = num_frames;
    let mut fps = fps;
    let mut carry_positions: Vec<i64> = carry_positions.to_vec();
    let mut carry_keyframes = carry_keyframes.clone();

    for round in 1..=rounds {
        if carry_positions.is_empty() {
            return Err(Error::Msg(format!(
                "ltx dfr: temporal round {round}: missing carry-forward keyframes"
            )));
        }
        video = upsample(&video)?;
        num_frames = gen_core_dfr::temporal_upsampled_frames(num_frames);
        fps *= 2.0;
        let expected_t = (num_frames - 1) / temporal_scale + 1;
        if i64::from(video.shape()[2]) != expected_t {
            return Err(Error::Msg(format!(
                "ltx dfr: temporal upsampler produced T={} for round {round}, expected \
                 {expected_t}",
                video.shape()[2]
            )));
        }
        // Carried keyframes are single-frame latents: only their positions scale with the round.
        let seam_positions: Vec<i64> = carry_positions.iter().map(|p| 2 * p).collect();
        let anchor_keyframes = carry_keyframes.clone();
        let seam_to_index: std::collections::HashMap<i64, i32> = seam_positions
            .iter()
            .enumerate()
            .map(|(i, &s)| (s, i as i32))
            .collect();
        let cond_fps = fps.min(gen_core_dfr::MAX_CONDITIONING_FPS);
        let tiles = gen_core_dfr::tile_ranges(
            &seam_positions,
            num_frames,
            1usize << round,
            temporal_scale,
            gen_core_dfr::TILE_LEAD_SEGMENTS,
        )?;

        let mut tile_latents: Vec<Array> = Vec::with_capacity(tiles.len());
        let mut slot_positions: Vec<i64> = Vec::new();
        let mut slot_latent_slices: Vec<Array> = Vec::new();

        for (tile_index, tile) in tiles.iter().enumerate() {
            let tile_video = frame_window(&video, tile.latent_start, tile.latent_end_exclusive)?;

            // Every seam in the window is a hard keyframe, including the one at local frame 0 of
            // non-first tiles.
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
                    return Err(Error::Msg(format!(
                        "ltx dfr: anchor seams {missing:?} missing from the carry-forward bag"
                    )));
                }
                let idx: Vec<i32> = anchor_global.iter().map(|p| seam_to_index[p]).collect();
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
                    return Err(Error::Msg(format!(
                        "ltx dfr: temporal round {round}: tile {tile_index} produced no keyframe \
                         slots"
                    )));
                };
                if generated.shape()[2] as usize != slot_global.len() {
                    return Err(Error::Msg(format!(
                        "ltx dfr: tile {tile_index} returned {} slot keyframes for {} slots",
                        generated.shape()[2],
                        slot_global.len()
                    )));
                }
                slot_positions.extend_from_slice(slot_global);
                slot_latent_slices.push(generated);
            }
        }

        let stitched = stitch_tile_latents(&tile_latents, &tiles)?;
        if i64::from(stitched.shape()[2]) != expected_t {
            return Err(Error::Msg(format!(
                "ltx dfr: stitched latent T={} != expected {expected_t}",
                stitched.shape()[2]
            )));
        }
        video = stitched;

        // Lead-in segments repeat the previous tile's slots; the earlier tile's version wins.
        let slot_latents = if slot_latent_slices.is_empty() {
            None
        } else {
            Some(concatenate_axis(
                &slot_latent_slices.iter().collect::<Vec<_>>(),
                2,
            )?)
        };
        let (slot_positions, slot_latents) = match slot_latents {
            None => (Vec::new(), None),
            Some(all) => {
                let mut first_index: std::collections::BTreeMap<i64, i32> = Default::default();
                for (index, &position) in slot_positions.iter().enumerate() {
                    first_index.entry(position).or_insert(index as i32);
                }
                let positions: Vec<i64> = first_index.keys().copied().collect();
                let idx: Vec<i32> = first_index.values().copied().collect();
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
/// (`(requested − 1) · 2^rounds + 1`; the trim always lands on a latent boundary). Returns the
/// trimmed latent and the final frame count.
pub fn trim_to_target_frames(
    video_latent: &Array,
    canvas_frames: i64,
    requested_frames: i64,
    rounds: u32,
) -> Result<(Array, i64)> {
    let target = gen_core_dfr::dfr_target_frames(requested_frames, rounds);
    if target > canvas_frames {
        return Err(Error::Msg(format!(
            "ltx dfr: target {target} frames exceeds the generated canvas {canvas_frames}"
        )));
    }
    if target == canvas_frames {
        return Ok((video_latent.clone(), canvas_frames));
    }
    let temporal_scale = TEMPORAL_SCALE;
    let keep_latents = (target - 1) / temporal_scale + 1;
    Ok((frame_window(video_latent, 0, keep_latents as usize)?, target))
}

/// The production tile denoise (`dfr_pipeline`'s per-tile stage call): re-noise the window at
/// `TEMPORAL_SIGMAS[0]`, append the seam anchors as hard single-frame keyframes
/// ([`gen_core_dfr::ANCHOR_KEYFRAME_STRENGTH`]), append the window's generated slots seeded from
/// the video, then run the **video-only** rectified-flow ancestral loop
/// (`eta = `[`gen_core_dfr::TEMPORAL_ANCESTRAL_ETA`]) and read the slots back.
///
/// `positions` is the window's token position grid at `cond_fps` (built by the caller — tile
/// windows are positionally identical, only their conditioning differs).
pub fn denoise_dfr_tile(
    dit: &AvDiT,
    job: &DfrTileJob<'_>,
    video_ctx: &Array,
    positions: &Array,
    cancel: &CancelFlag,
    on_step: &mut dyn FnMut(usize),
) -> Result<DfrTileResult> {
    let sh = job.tile_video.shape();
    let (h, w) = (sh[3] as usize, sh[4] as usize);
    let spatial_scale = SPATIAL_SCALE;

    // Tile-entry re-noise: noise·σ₀ + latent·(1 − σ₀), seeded per tile.
    let key = mlx_rs::random::key(job.noise_seed)?;
    let noise = mlx_rs::random::normal::<f32>(sh, None, None, Some(&key))?
        .as_dtype(job.tile_video.dtype())?;
    let renoised = crate::pipeline::renoise(&job.tile_video, &noise, TEMPORAL_SIGMAS[0])?;

    let mut state = VideoTokenState::base(&renoised, positions)?;
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
        // The appended slot latents seed from `initial_keyframes`; noise them to the tile's entry
        // σ like the stage noiser would (`denoise_mask = 1` ⇒ plain lerp toward noise).
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
        on_step,
    )?;

    let grid_tokens: Vec<i32> = (0..state.target_tokens).collect();
    let grid = state
        .latent
        .take_axis(Array::from_slice(&grid_tokens, &[state.target_tokens]), 1)?;
    let latent = crate::conditioning::unpatchify_grid(
        &grid,
        state.latent.shape()[2],
        sh[2],
        sh[3],
        sh[4],
    )?;
    let generated_keyframes = if has_slots {
        Some(crate::conditioning::take_generated_keyframes(
            &state, sh[3], sh[4],
        )?)
    } else {
        None
    };
    Ok(DfrTileResult {
        latent,
        generated_keyframes,
    })
}

/// Stage-entry noising for appended generated-keyframe slot tokens: where the token is a slot
/// (`keyframes_mask > 0`, `denoise_mask = 1`), lerp the slot's seeded latent toward fresh noise at
/// the stage-entry `sigma` — the reference stage noiser applied to the appended run only (the grid
/// half of the state is noised as a grid by the caller).
pub fn noise_slot_tokens(state: &VideoTokenState, sigma: f32, seed: u64) -> Result<Array> {
    let Some(mask) = state.keyframes_mask.as_ref() else {
        return Ok(state.latent.clone());
    };
    let dt = state.latent.dtype();
    let key = mlx_rs::random::key(seed)?;
    let noise = mlx_rs::random::normal::<f32>(state.latent.shape(), None, None, Some(&key))?
        .as_dtype(dt)?;
    let sigma = Array::from_slice(&[sigma], &[1]).as_dtype(dt)?;
    let gate = multiply(mask, &sigma)?; // (B, S, 1): σ on slot tokens, 0 elsewhere
    let gate = broadcast_to(&gate, state.latent.shape())?;
    let one = Array::from_slice(&[1.0f32], &[1]).as_dtype(dt)?;
    let keep = mlx_rs::ops::subtract(&one, &gate)?;
    Ok(mlx_rs::ops::add(
        &multiply(&noise, &gate)?,
        &multiply(&state.latent, &keep)?,
    )?)
}

/// Sanity companion for [`denoise_dfr_tile`]'s caller: per-token σ at the tile's entry sigma —
/// exposed for tests that pin the conditioning-token timestep contract (anchors at
/// `σ·(1 − strength)`, slots at `σ`).
pub fn tile_entry_timesteps(state: &VideoTokenState) -> Result<Array> {
    token_timesteps(&state.denoise_mask, state.latent.dtype(), TEMPORAL_SIGMAS[0])
}

/// Everything [`generate_dfr_av_latents`] needs beyond the request-shaped parameters: the loaded
/// components and the encoded text contexts.
pub struct DfrComponents<'a> {
    pub dit: &'a AvDiT,
    pub spatial_upsampler: &'a crate::upsampler::LatentUpsampler,
    /// Required when `temporal_upsample_rounds > 0`; a rounds request without it is a typed error
    /// (mirrors the reference's up-front `temporal_upsampler_path` validation).
    pub temporal_upsampler: Option<&'a crate::upsampler::LatentUpsampler>,
    pub latent_mean: &'a Array,
    pub latent_std: &'a Array,
    pub video_ctx: &'a Array,
    pub audio_ctx: &'a Array,
    pub audio_pos: &'a Array,
}

/// The DFR request shape. `canvas_frames` and `keyframe_positions` come from
/// [`gen_core_dfr::resolve_canvas`] over the request's (auto-)resolved frame count —
/// `requested_frames` is that pre-padding count, and the pipeline trims back to
/// `(requested − 1)·2^rounds + 1` at the end.
pub struct DfrRequest<'a> {
    pub canvas_frames: i64,
    pub requested_frames: i64,
    pub keyframe_positions: &'a [i64],
    pub fps: f32,
    pub seed: u64,
    pub temporal_upsample_rounds: u32,
    /// `Some(downscale)` appends the reserved half-res stage-1 video as the detailing IC-LoRA
    /// reference in stage 2 (reference `--detailing-lora` + `VideoConditionByReferenceLatent`).
    /// The detailing LoRA weights themselves are installed on the DiT by the engine's adapter
    /// layer, scoped to the stage-2 pass; this flag only controls the reference conditioning.
    pub detailing_downscale: Option<i64>,
    /// Replace-latent image conditioning (I2V / first-last-frame / multi-keyframe), VAE-encoded at
    /// both stage resolutions — empty for T2V. Stage 2's state is built over the upscaled stage-1
    /// video, exactly like [`crate::pipeline::generate_av_latents`].
    pub video_keyframes: &'a [crate::pipeline::StageKeyframe<'a>],
}

/// The DFR pipeline output: the final video latent, the **stage-1** audio latent (the shipped
/// audio — stage 2 re-noises audio only because video needs the cross-modal attention), the final
/// pixel-frame count and the playback fps (`fps · 2^rounds`).
pub struct DfrOutput {
    pub video_latent: Array,
    pub audio_latent: Array,
    pub num_frames: i64,
    pub playback_fps: f32,
}

/// The full DFR latent pipeline (`dfr_pipeline.DFRPipeline.__call__`): stage-1 half-res base with
/// generated keyframe slots on the segment grid → spatial x2 upsample of video + slots → stage-2
/// full-res re-denoise with slot warm starts and the optional detailing reference → up to two
/// tiled temporal rounds ([`run_temporal_rounds`]) → trim to the caller's frame contract.
///
/// Stages run the deterministic distilled Euler ([`crate::pipeline::denoise_av_tokens`], per the
/// reference's default stage loop); only the temporal-round tiles run the RF-ancestral loop.
/// LoRA passes: stage 1 and the temporal tiles select pass 0, stage 2 selects pass 1 (where the
/// engine scopes any detailing LoRA).
#[allow(clippy::too_many_arguments)]
pub fn generate_dfr_av_latents(
    parts: &DfrComponents<'_>,
    req: &DfrRequest<'_>,
    video_s1_noise: &Array,
    video_pos1: &Array,
    video_s2_noise: &Array,
    video_pos2: &Array,
    audio_s1_noise: &Array,
    audio_s2_noise: &Array,
    cancel: &CancelFlag,
    on_step: &mut dyn FnMut(usize),
) -> Result<DfrOutput> {
    use crate::pipeline::{denoise_av_tokens, renoise, STAGE1_SIGMAS, STAGE2_SIGMAS};

    let rounds = req.temporal_upsample_rounds;
    if rounds > gen_core_dfr::MAX_TEMPORAL_UPSAMPLE_ROUNDS {
        return Err(Error::Msg(format!(
            "ltx dfr: temporal_upsample_rounds must be 0..={}, got {rounds}",
            gen_core_dfr::MAX_TEMPORAL_UPSAMPLE_ROUNDS
        )));
    }
    if rounds > 0 && parts.temporal_upsampler.is_none() {
        return Err(Error::Msg(
            "ltx dfr: temporal_upsample_rounds > 0 requires the temporal latent upsampler \
             component"
                .into(),
        ));
    }
    if req.keyframe_positions.is_empty() {
        return Err(Error::Msg(
            "ltx dfr: the DFR canvas resolved no keyframe positions".into(),
        ));
    }
    let temporal_scale = TEMPORAL_SCALE;
    let s1 = video_s1_noise.shape();
    let (h1, w1) = (s1[3] as usize, s1[4] as usize);
    let s2 = video_s2_noise.shape();
    let (h2, w2) = (s2[3] as usize, s2[4] as usize);
    let expected_lf = (req.canvas_frames - 1) / temporal_scale + 1;
    if i64::from(s1[2]) != expected_lf || i64::from(s2[2]) != expected_lf {
        return Err(Error::Msg(format!(
            "ltx dfr: stage noise T ({} / {}) must match the canvas' {expected_lf} latent frames",
            s1[2], s2[2]
        )));
    }

    // --- Stage 1: half-res base + keyframe slots ------------------------------------------------
    let zeros1 = Array::zeros::<f32>(video_s1_noise.shape())?.as_dtype(video_s1_noise.dtype())?;
    let mut state =
        match crate::pipeline::stage_keyframe_state(&zeros1, req.video_keyframes, true)? {
            Some(i2v) => {
                let noised = i2v.noised(video_s1_noise, STAGE1_SIGMAS[0])?;
                VideoTokenState::from_i2v(&noised, video_pos1)?
            }
            None => VideoTokenState::base(video_s1_noise, video_pos1)?,
        };
    state = append_generated_keyframe_slots(
        &state,
        req.keyframe_positions,
        None,
        req.canvas_frames,
        h1,
        w1,
        SPATIAL_SCALE,
        req.fps,
    )?;
    // Zero-seeded slots still enter at full noise: lerp them toward the stage-entry σ.
    state.latent = noise_slot_tokens(&state, STAGE1_SIGMAS[0], req.seed.wrapping_add(11))?;

    parts.dit.set_lora_pass(0);
    let (state, audio_s1) = denoise_av_tokens(
        parts.dit,
        &state,
        audio_s1_noise,
        parts.video_ctx,
        parts.audio_ctx,
        parts.audio_pos,
        &STAGE1_SIGMAS,
        cancel,
        on_step,
    )?;
    let stage1_audio_latent = audio_s1.clone();
    let grid_tokens: Vec<i32> = (0..state.target_tokens).collect();
    let grid = state
        .latent
        .take_axis(Array::from_slice(&grid_tokens, &[state.target_tokens]), 1)?;
    let reserved_half_res = crate::conditioning::unpatchify_grid(
        &grid,
        state.latent.shape()[2],
        s1[2],
        s1[3],
        s1[4],
    )?;
    let slot_keyframes = crate::conditioning::take_generated_keyframes(&state, s1[3], s1[4])?;

    // Spatial x2: video and slots ride the same upsampler (slots' K sits on the frame axis, which
    // the spatial checkpoint leaves untouched).
    let upscaled_video = crate::upsampler::upsample_latents(
        &reserved_half_res,
        parts.spatial_upsampler,
        parts.latent_mean,
        parts.latent_std,
    )?;
    let upscaled_slots = crate::upsampler::upsample_latents(
        &slot_keyframes,
        parts.spatial_upsampler,
        parts.latent_mean,
        parts.latent_std,
    )?;
    mlx_rs::transforms::eval([&upscaled_video, &upscaled_slots, &stage1_audio_latent])?;

    // --- Stage 2: full-res detailing --------------------------------------------------------------
    let s2_entry = STAGE2_SIGMAS[0];
    let mut state2 = match crate::pipeline::stage_keyframe_state(
        &upscaled_video,
        req.video_keyframes,
        false,
    )? {
        Some(i2v) => {
            // Replace-latent conditioning over the upscaled base, then the stage noiser.
            let noised = i2v.noised(video_s2_noise, s2_entry)?;
            VideoTokenState::from_i2v(&noised, video_pos2)?
        }
        None => {
            let renoised = renoise(&upscaled_video, video_s2_noise, s2_entry)?;
            VideoTokenState::base(&renoised, video_pos2)?
        }
    };
    state2 = append_generated_keyframe_slots(
        &state2,
        req.keyframe_positions,
        Some(&upscaled_slots),
        req.canvas_frames,
        h2,
        w2,
        SPATIAL_SCALE,
        req.fps,
    )?;
    state2.latent = noise_slot_tokens(&state2, s2_entry, req.seed.wrapping_add(13))?;
    if let Some(downscale) = req.detailing_downscale {
        state2 = crate::conditioning::append_reference_latent(
            &state2,
            &reserved_half_res,
            downscale,
            1.0,
            temporal_scale,
            SPATIAL_SCALE,
            req.fps,
        )?;
    }
    let audio2 = renoise(&stage1_audio_latent, audio_s2_noise, s2_entry)?;

    parts.dit.set_lora_pass(1);
    let (state2, _audio2) = denoise_av_tokens(
        parts.dit,
        &state2,
        &audio2,
        parts.video_ctx,
        parts.audio_ctx,
        parts.audio_pos,
        &STAGE2_SIGMAS,
        cancel,
        on_step,
    )?;
    let grid_tokens: Vec<i32> = (0..state2.target_tokens).collect();
    let grid2 = state2
        .latent
        .take_axis(Array::from_slice(&grid_tokens, &[state2.target_tokens]), 1)?;
    let mut video = crate::conditioning::unpatchify_grid(
        &grid2,
        state2.latent.shape()[2],
        s2[2],
        s2[3],
        s2[4],
    )?;
    let carry_keyframes = crate::conditioning::take_generated_keyframes(&state2, s2[3], s2[4])?;
    mlx_rs::transforms::eval([&video, &carry_keyframes])?;

    // --- Temporal rounds --------------------------------------------------------------------------
    let mut num_frames = req.canvas_frames;
    let mut playback_fps = req.fps;
    if rounds > 0 {
        let upsampler = parts.temporal_upsampler.expect("validated above");
        // Temporal tiles run the base (stage-1) LoRA pass, like the reference's non-detailing
        // stage.
        parts.dit.set_lora_pass(0);
        let mut upsample = |v: &Array| {
            crate::upsampler::upsample_latents(
                v,
                upsampler,
                parts.latent_mean,
                parts.latent_std,
            )
        };
        let mut denoise_tile = |job: &DfrTileJob| {
            let t_tile = job.tile.latent_frames();
            let positions = crate::positions::create_position_grid_with(
                1,
                t_tile,
                h2,
                w2,
                temporal_scale,
                SPATIAL_SCALE,
                job.cond_fps,
                true,
            );
            denoise_dfr_tile(parts.dit, job, parts.video_ctx, &positions, cancel, on_step)
        };
        let out = run_temporal_rounds(
            &video,
            req.keyframe_positions,
            &carry_keyframes,
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
    use mlx_rs::ops::indexing::IndexOp;
    use mlx_rs::Device;
    use std::cell::RefCell;

    /// All orchestration tests run on the CPU stream — the gpu-local lane is queued and nothing
    /// here needs Metal. RAII-restored so the process-global default device does not leak into
    /// other tests in this binary (the vocoder STFT test is bit-exact and device-sensitive).
    struct CpuGuard(Device);
    impl CpuGuard {
        fn new() -> Self {
            let previous = Device::try_default().expect("default device");
            Device::set_default(&Device::cpu());
            Self(previous)
        }
    }
    impl Drop for CpuGuard {
        fn drop(&mut self) {
            Device::set_default(&self.0);
        }
    }

    /// `(1, C, T, 1, 1)` latent whose frame t is the constant `base + t`.
    fn ramp_latent(c: i32, t: i32, base: f32) -> Array {
        let mut data = Vec::with_capacity((c * t) as usize);
        for _ in 0..c {
            for frame in 0..t {
                data.push(base + frame as f32);
            }
        }
        Array::from_slice(&data, &[1, c, t, 1, 1])
    }

    fn const_latent(c: i32, t: i32, v: f32) -> Array {
        Array::from_slice(&vec![v; (c * t) as usize], &[1, c, t, 1, 1])
    }

    fn frame_value(x: &Array, t: i32) -> f32 {
        x.index((0, 0, t, 0, 0)).item::<f32>()
    }

    /// Fake temporal upsampler: T → 2T−1 by repeating frames (`[f0, f0, f1, f1, …, fT−1]`-style
    /// midpoint fill; values don't matter to the orchestration, the shape contract does).
    fn fake_upsample(x: &Array) -> Result<Array> {
        let t = x.shape()[2];
        let mut idx: Vec<i32> = Vec::new();
        for i in 0..t {
            idx.push(i);
            if i + 1 < t {
                idx.push(i);
            }
        }
        take_frames(x, &idx)
    }

    /// One recorded tile call.
    #[derive(Clone, Debug)]
    struct Call {
        round: u32,
        tile_index: usize,
        seed: u64,
        cond_fps: f32,
        anchors_global: Vec<i64>,
        slots_global: Vec<i64>,
        anchor_first_value: Option<f32>,
        window_t: i32,
    }

    /// Recording fake denoiser: returns the window filled with the constant
    /// `100·round + tile_index + 1` and slot keyframes filled with `1000·round + tile_index`.
    fn recording_denoiser(
        calls: &RefCell<Vec<Call>>,
    ) -> impl FnMut(&DfrTileJob) -> Result<DfrTileResult> + '_ {
        move |job| {
            let sh = job.tile_video.shape();
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
                slots_global: job
                    .slot_positions_local
                    .iter()
                    .map(|p| p + job.tile.pixel_start)
                    .collect(),
                anchor_first_value: job.anchor_latents.as_ref().map(|a| frame_value(a, 0)),
                window_t: sh[2],
            });
            let generated_keyframes = if job.slot_positions_local.is_empty() {
                None
            } else {
                Some(const_latent(
                    sh[1],
                    job.slot_positions_local.len() as i32,
                    (1000 * job.round as i32 + job.tile_index as i32) as f32,
                ))
            };
            Ok(DfrTileResult {
                latent: const_latent(sh[1], sh[2], (100 * job.round as i32) as f32
                    + job.tile_index as f32
                    + 1.0),
                generated_keyframes,
            })
        }
    }

    /// The 121-frame canvas: stage-2 slots at the segment marks are the round-1 carry bag.
    fn stage2_carry(c: i32) -> (Vec<i64>, Array) {
        let positions: Vec<i64> = vec![24, 48, 72, 96, 120];
        let latents = const_latent(c, positions.len() as i32, 7.0);
        (positions, latents)
    }

    /// Round/tile bookkeeping is honoured mechanically: 1 round → 2 tiles and T 16→31; 2 rounds →
    /// 2 then 4 tiles and T 61, fps doubling with the conditioning cap, per-tile seeds all
    /// distinct. A `temporal_upsample_rounds` knob that is inert above 1 — the known failure
    /// shape — cannot pass this: rounds=2 must produce the deeper canvas AND the round-2 calls.
    #[test]
    fn round_count_is_honoured_not_inert_above_1() {
        let _cpu = CpuGuard::new();
        let video = ramp_latent(2, 16, 0.0); // 121 frames → 16 latents
        let (carry_pos, carry_kf) = stage2_carry(2);

        for (rounds, want_calls, want_t, want_frames, want_fps) in [
            (1u32, vec![(1u32, 2usize)], 31i32, 241i64, 48.0f32),
            (2, vec![(1, 2), (2, 4)], 61, 481, 96.0),
        ] {
            let calls = RefCell::new(Vec::new());
            let out = run_temporal_rounds(
                &video,
                &carry_pos,
                &carry_kf,
                121,
                24.0,
                77,
                rounds,
                &mut fake_upsample,
                &mut recording_denoiser(&calls),
            )
            .unwrap();
            assert_eq!(out.video_latent.shape()[2], want_t, "rounds={rounds}");
            assert_eq!(out.num_frames, want_frames);
            assert_eq!(out.fps, want_fps);
            let calls = calls.borrow();
            let per_round: Vec<(u32, usize)> = want_calls.clone();
            for (round, count) in per_round {
                assert_eq!(
                    calls.iter().filter(|c| c.round == round).count(),
                    count,
                    "rounds={rounds}: round {round} tile count"
                );
            }
            // Per-tile ancestral seeds are all distinct (a shared seed would inject
            // byte-identical noise into positionally identical tiles).
            let mut seeds: Vec<u64> = calls.iter().map(|c| c.seed).collect();
            seeds.sort_unstable();
            seeds.dedup();
            assert_eq!(seeds.len(), calls.len(), "per-tile seeds must be distinct");
            // Conditioning fps is capped at 60 while playback fps doubles freely.
            for c in calls.iter() {
                let want = if c.round == 1 { 48.0 } else { 60.0 };
                assert_eq!(c.cond_fps, want, "round {} cond fps", c.round);
            }
        }
    }

    /// The stitch keeps exactly each tile's owned run: with per-tile constant fills, every latent
    /// frame of the stitched canvas must carry the constant of the tile that owns it, and the
    /// handover happens exactly at the seam latent (previous tile keeps it). This is the check on
    /// the exact tensor positions where a wrong `drop_latent_prefix` would double-write or gap.
    #[test]
    fn stitch_hands_over_exactly_at_the_seam_latent() {
        let _cpu = CpuGuard::new();
        let video = ramp_latent(1, 16, 0.0);
        let (carry_pos, carry_kf) = stage2_carry(1);
        let calls = RefCell::new(Vec::new());
        let out = run_temporal_rounds(
            &video,
            &carry_pos,
            &carry_kf,
            121,
            24.0,
            5,
            1,
            &mut fake_upsample,
            &mut recording_denoiser(&calls),
        )
        .unwrap();
        // Round-1 tiles over the 241-frame canvas (from the reference goldens): tile 0 owns
        // latents [0, 19) (fill 101), tile 1 spans [12, 31) with drop prefix 7, so its kept run is
        // latents 19..31 (fill 102). Latent 18 is the last of tile 0; latent 19 the first of
        // tile 1's kept half.
        assert_eq!(out.video_latent.shape()[2], 31);
        for t in 0..=18 {
            assert_eq!(frame_value(&out.video_latent, t), 101.0, "latent {t}");
        }
        for t in 19..31 {
            assert_eq!(frame_value(&out.video_latent, t), 102.0, "latent {t}");
        }
    }

    /// Inter-round continuity: round 2's seam anchors must include positions that exist ONLY
    /// because round 1's denoised slots were merged into the carry bag, and their latents must be
    /// the round-1 tile outputs (fill 1000·1 + tile), not the stage-2 stills. Dropping the
    /// carry-forward merge (the M3 mutation) leaves those seams unanchored or anchored with stale
    /// content — both fail here.
    #[test]
    fn round2_anchors_carry_round1_slot_content() {
        let _cpu = CpuGuard::new();
        let video = ramp_latent(1, 16, 0.0);
        let (carry_pos, carry_kf) = stage2_carry(1);
        let calls = RefCell::new(Vec::new());
        run_temporal_rounds(
            &video,
            &carry_pos,
            &carry_kf,
            121,
            24.0,
            5,
            2,
            &mut fake_upsample,
            &mut recording_denoiser(&calls),
        )
        .unwrap();
        let calls = calls.borrow();
        // Round-1 slots sit at the odd multiples of 24 on the 241 grid (24, 72, 120, 168, 216);
        // doubled onto the 481 grid they are the round-2 seams 48, 144, 240, 336, 432. Every
        // round-2 anchor set must be non-empty and drawn from the merged bag {48·k}.
        let round2: Vec<&Call> = calls.iter().filter(|c| c.round == 2).collect();
        assert_eq!(round2.len(), 4);
        let mid_derived: Vec<i64> = vec![48, 144, 240, 336, 432];
        let mut seen_mid_derived = false;
        for call in &round2 {
            assert!(!call.anchors_global.is_empty(), "round-2 tiles are anchored");
            for a in &call.anchors_global {
                assert_eq!(a % 48, 0, "round-2 anchor {a} must be a merged-bag seam");
            }
            if call.anchors_global.iter().any(|a| mid_derived.contains(a)) {
                seen_mid_derived = true;
            }
        }
        assert!(
            seen_mid_derived,
            "no round-2 tile was anchored on a round-1 slot seam — the carry-forward merge is \
             not flowing"
        );
        // The first round-2 tile's first anchor (seam 48) is a round-1 slot: its latent must be a
        // round-1 tile output (1000·1 + tile ∈ {1000, 1001}), not the stage-2 still (7).
        let tile0 = round2.iter().find(|c| c.tile_index == 0).unwrap();
        assert_eq!(tile0.anchors_global[0], 48);
        let v = tile0.anchor_first_value.unwrap();
        assert!(
            v == 1000.0 || v == 1001.0,
            "round-2 anchor at seam 48 must carry round-1 slot content, got {v}"
        );
    }

    /// Slot seeds pick the nearest latent frame of the window, tile-locally.
    #[test]
    fn slot_initials_pick_nearest_latent_frame() {
        let _cpu = CpuGuard::new();
        let window = ramp_latent(1, 19, 0.0);
        let init = slot_initials_from_video(&window, &[24, 72, 120], 8).unwrap();
        assert_eq!(init.shape(), &[1, 1, 3, 1, 1]);
        assert_eq!(frame_value(&init, 0), 3.0); // 24/8
        assert_eq!(frame_value(&init, 1), 9.0); // 72/8
        assert_eq!(frame_value(&init, 2), 15.0); // 120/8
    }

    /// Carry-forward merge orders by position and errors when a latent bag is missing.
    #[test]
    fn carry_forward_merge_orders_and_validates() {
        let _cpu = CpuGuard::new();
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

    /// A tile that was asked for slots but returns none is a hard error, not a silent skip.
    #[test]
    fn missing_slot_keyframes_is_a_hard_error() {
        let _cpu = CpuGuard::new();
        let video = ramp_latent(1, 16, 0.0);
        let (carry_pos, carry_kf) = stage2_carry(1);
        let mut no_slots = |job: &DfrTileJob| {
            let sh = job.tile_video.shape();
            Ok(DfrTileResult {
                latent: const_latent(sh[1], sh[2], 1.0),
                generated_keyframes: None,
            })
        };
        let err = run_temporal_rounds(
            &video,
            &carry_pos,
            &carry_kf,
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

    /// The padded-canvas trim honours the caller's frame contract on a latent boundary.
    #[test]
    fn trim_to_target_frames_lands_on_latent_boundary() {
        let _cpu = CpuGuard::new();
        // 153 requested → canvas 161 (21 latents); 0 rounds → keep (153−1)/8+1 = 20 latents.
        let canvas = ramp_latent(1, 21, 0.0);
        let (trimmed, frames) = trim_to_target_frames(&canvas, 161, 153, 0).unwrap();
        assert_eq!(frames, 153);
        assert_eq!(trimmed.shape()[2], 20);
        // Exact fit is a no-op.
        let (same, frames) = trim_to_target_frames(&canvas, 161, 161, 0).unwrap();
        assert_eq!((frames, same.shape()[2]), (161, 21));
        // A target beyond the canvas is a hard error.
        assert!(trim_to_target_frames(&canvas, 161, 169, 0).is_err());
    }
}
