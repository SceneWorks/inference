//! The live SCAIL-2 generation pipeline — the runnable end-to-end denoise loop.
//!
//! Ports `wan/scail.py::SCAIL2Pipeline.generate`: preprocess the reference character (+ optional
//! extra characters) and the driving video into conditioning latents, then run a plain-CFG
//! flow-matching denoise (UniPC) over one or more 81-frame **segments** with clean-history continuity,
//! and VAE-decode each segment back to pixels.
//!
//! Reuse map — the heavy components are `candle-gen-wan`'s (SCAIL-2 *is* Wan2.1-14B I2V): the z16
//! [`ProviderVae`] (encode/decode; its decode already streams one latent frame at a time = the
//! temporal-tiled decode the high-res fix needs), the [`Umt5Encoder`] text encoder, and the
//! flow-matching [`FlowScheduler`] (UniPC). SCAIL-2's own pieces are the [`Scail2Dit`] forward, the
//! open-CLIP [`ScailClip`] image encode, the 28-channel [`extract_and_compress_mask_to_latent`] mask
//! build, and the [`interpolate`]/[`downsample_half`] resizes.

use candle_gen::gen_core::render::video::{run_video_steps, VideoDenoisePhase};
use candle_gen::gen_core::{MemoryGeometry, MemoryRequestScope};

use gen_core_family::scail2::{resolve_segments, Scail2Segment, SegmentTail};
use std::ops::Deref;
use std::path::{Path, PathBuf};

use candle_gen::candle_core::{DType, Device, Tensor};
use candle_gen::gen_core::runtime::CancelFlag;
use candle_gen::gen_core::{GenerationOutput, Image, Progress};
use candle_gen::{CandleError, Result as CResult};
use candle_gen_wan::config::TextEncoderConfig;
use candle_gen_wan::pipeline::frames_to_images;
use candle_gen_wan::scheduler::{FlowScheduler, Sampler};
use candle_gen_wan::text_encoder::Umt5Encoder;
use gen_core_tokenizer::{ChatTemplate, TextTokenizer, TokenizerConfig};

use crate::clip::ScailClip;
use crate::model::{Scail2Dit, Scail2Inputs};
use crate::preprocess::{extract_and_compress_mask_to_latent, TEMPORAL_STRIDE};
use crate::resize::{clip_preprocess, downsample_half, interpolate, Interp};
use crate::ProviderVae;

/// Inputs must be divisible by 32: the pose path halves spatially (→ ÷16) before the ÷8 VAE stride, and
/// the 28-channel mask pools 8×, so both the full and half grids stay integer + even.
pub(crate) const DIM_ALIGN: u32 = crate::VAE_TILING.spatial_scale as u32 * 4;

/// The loaded SCAIL-2 components (resident in the [`crate::pipeline::Scail2`] generator's cache). All
/// run f32 (the DiT's high-token-length NaN avoidance, and z16 VAE / UMT5 / CLIP are f32 anyway).
pub struct Components {
    pub te: Umt5Encoder,
    pub dit: Scail2Dit,
    pub vae: ProviderVae,
    pub clip: ScailClip,
    /// UMT5 tokenizer, loaded+parsed **once** at component load and reused across the pos/neg encodes
    /// (sc-8991 / F-011) instead of re-parsing `tokenizer.json` per generate call.
    pub tok: TextTokenizer,
}

/// One masked character reference (the primary subject or an extra character): an RGB image paired with
/// its color-coded segmentation mask.
#[derive(Debug)]
pub struct CharacterRef<'a> {
    pub image: &'a Image,
    pub mask: &'a Image,
}

/// A fully-specified SCAIL-2 generation job (the engine-internal form the worker maps a
/// `GenerationRequest` onto). `width` and `height` must already be non-zero multiples of the
/// 32-pixel render lattice; [`generate`] rejects an invalid job rather than silently changing its geometry.
/// All images are decoded + resized to `(width, height)` here.
pub struct Scail2Job<'a> {
    pub prompt: &'a str,
    pub negative_prompt: &'a str,
    pub width: u32,
    pub height: u32,
    pub reference: CharacterRef<'a>,
    pub additional: Vec<CharacterRef<'a>>,
    pub driving_frames: &'a [Image],
    pub driving_masks: &'a [Image],
    /// `true` = cross-identity replacement, `false` = animation.
    pub replace_flag: bool,
    pub seed: u64,
    pub steps: usize,
    pub shift: f64,
    pub guidance: f64,
    pub sampler: Sampler,
    pub fps: u32,
    pub segment_len: usize,
    pub segment_overlap: usize,
}

/// Encode every additional character while observing cancellation before, between, and after the
/// individual VAE/mask passes. Extra references are input-side work and intentionally emit no
/// progress events: the existing global denoise sweep remains the sole `Progress::Step` sequence
/// and therefore stays monotonic.
fn map_additional_references<T, U>(
    additional: &[CharacterRef<'_>],
    cancel: &CancelFlag,
    mut encode_image: impl FnMut(&CharacterRef<'_>) -> CResult<T>,
    mut encode_mask: impl FnMut(&CharacterRef<'_>) -> CResult<U>,
) -> CResult<Vec<(T, U)>> {
    let mut encoded = Vec::with_capacity(additional.len());
    for reference in additional {
        if cancel.is_cancelled() {
            return Err(CandleError::Canceled);
        }
        let image = encode_image(reference)?;
        if cancel.is_cancelled() {
            return Err(CandleError::Canceled);
        }
        let mask = encode_mask(reference)?;
        if cancel.is_cancelled() {
            return Err(CandleError::Canceled);
        }
        encoded.push((image, mask));
    }
    Ok(encoded)
}

/// Validate the exact render geometry carried by the public [`Scail2Job`] API.
///
/// Direct callers do not carry the explicit-versus-resolved origin available to the provider
/// adapter, so this lower-level API accepts exact render geometry only.
fn validate_job_geometry(width: u32, height: u32) -> CResult<(usize, usize)> {
    if width < DIM_ALIGN
        || height < DIM_ALIGN
        || !width.is_multiple_of(DIM_ALIGN)
        || !height.is_multiple_of(DIM_ALIGN)
    {
        return Err(CandleError::Msg(format!(
            "scail2: Scail2Job width/height must be non-zero multiples of {DIM_ALIGN} \
             (got {width}×{height})"
        )));
    }
    Ok((width as usize, height as usize))
}

/// Project a dimension onto the SCAIL-2 render lattice for the provider's area calculation.
///
/// Since sc-16198, the provider rejects off-grid explicit requests before the area gate, so this is
/// a no-op for every reachable explicit request. Keeping the lattice projection in the area helper
/// preserves sc-16197's rendered-geometry rule for direct helper probes and any future resolved-size
/// path. The public low-level [`generate`] API does not call this function: it requires exact geometry
/// through [`validate_job_geometry`].
pub(crate) fn align(value: u32) -> usize {
    (value / DIM_ALIGN).max(1) as usize * DIM_ALIGN as usize
}

/// Decode and resize an RGB24 image on the host. Keeping the native-resolution tensor on CPU avoids
/// uploading it only for [`interpolate`] to read it back into a host buffer (F-070 / sc-12517).
fn image_to_chw_host(img: &Image, tw: usize, th: usize, mode: Interp) -> CResult<Tensor> {
    let (iw, ih) = (img.width as usize, img.height as usize);
    if img.pixels.len()
        != candle_gen::gen_core::imageops::checked_image_buffer_len(iw, ih, 3).unwrap_or(usize::MAX)
    {
        return Err(CandleError::Msg(format!(
            "scail2: image pixel buffer {} != {iw}x{ih}x3",
            img.pixels.len()
        )));
    }
    let px: Vec<f32> = img.pixels.iter().map(|&p| p as f32 / 127.5 - 1.0).collect();
    let chw = Tensor::from_vec(px, (ih, iw, 3), &Device::Cpu)?.permute((2, 0, 1))?; // [3,H,W]
    let nchw = chw.reshape((1, 3, ih, iw))?;
    let out = if (ih, iw) != (th, tw) {
        interpolate(&nchw, th, tw, mode)?
    } else {
        nchw
    };
    Ok(out.reshape((3, th, tw))?)
}

/// Host-first image preparation followed by exactly one upload of the resized tensor.
fn image_to_chw(img: &Image, tw: usize, th: usize, mode: Interp, dev: &Device) -> CResult<Tensor> {
    Ok(image_to_chw_host(img, tw, th, mode)?.to_device(dev)?)
}

/// Stack driving frames on CPU → `[T, 3, H, W]`. Segments stay host-resident through the half-size
/// resize and cross the device boundary only immediately before model preprocessing.
fn stack_frames(frames: &[Image], tw: usize, th: usize) -> CResult<Tensor> {
    let chw: Vec<Tensor> = frames
        .iter()
        .map(|f| -> CResult<Tensor> {
            Ok(image_to_chw_host(f, tw, th, Interp::Bicubic)?.reshape((1, 3, th, tw))?)
        })
        .collect::<CResult<_>>()?;
    let refs: Vec<&Tensor> = chw.iter().collect();
    Ok(Tensor::cat(&refs, 0)?) // [T,3,H,W]
}

/// Stack per-frame masks → `[3, T, H, W]` (the `extract_and_compress` input layout).
fn stack_masks(masks: &[Image], tw: usize, th: usize) -> CResult<Tensor> {
    let chw: Vec<Tensor> = masks
        .iter()
        .map(|m| -> CResult<Tensor> {
            Ok(image_to_chw_host(m, tw, th, Interp::Bilinear)?.reshape((3, 1, th, tw))?)
        })
        .collect::<CResult<_>>()?;
    let refs: Vec<&Tensor> = chw.iter().collect();
    Ok(Tensor::cat(&refs, 1)?) // [3,T,H,W]
}

/// VAE-encode a `[3, T, H, W]` pixel clip (`[-1,1]`) → `[16, T_lat, H/8, W/8]` (drops the batch dim).
fn vae_encode_cthw(vae: &ProviderVae, cthw: &Tensor) -> CResult<Tensor> {
    let (c, t, h, w) = cthw.dims4()?;
    let z = vae.encode(&cthw.reshape((1, c, t, h, w))?)?; // [1,16,T_lat,h,w]
    let (_, zc, zt, zh, zw) = z.dims5()?;
    Ok(z.reshape((zc, zt, zh, zw))?)
}

/// `uncond + guidance·(cond − uncond)`.
fn cfg_combine(uncond: &Tensor, cond: &Tensor, guidance: f64) -> CResult<Tensor> {
    Ok((uncond + (cond - uncond)?.affine(guidance, 0.0)?)?)
}

/// Overwrite the leading `min(history_t, T)` latent frames of `latent [16,T,h,w]` with the clean
/// history (upstream `apply_clean_history`). No-op without history.
fn apply_clean_history(latent: &Tensor, history: Option<&Tensor>) -> CResult<Tensor> {
    let Some(h) = history else {
        return Ok(latent.clone());
    };
    let lt = latent.dim(1)?;
    let ht = h.dim(1)?.min(lt);
    if ht == 0 {
        return Ok(latent.clone());
    }
    let head = h.narrow(1, 0, ht)?;
    if ht == lt {
        return Ok(head);
    }
    let tail = latent.narrow(1, ht, lt - ht)?;
    Ok(Tensor::cat(&[&head, &tail], 1)?)
}

/// Global denoise progress for step `i` (0-based) of segment `seg_idx` (0-based), given `steps` denoise
/// steps per segment over `num_segments` segments (F-125 / sc-11225).
///
/// A multi-segment SCAIL-2 job runs the same `steps` denoise once per segment. Reporting `current = i+1`
/// / `total = steps` per segment makes percent-complete jump backwards at every segment boundary — on
/// exactly the long clips where progress matters most. Instead this accumulates prior segments' steps so
/// the whole job is one monotonic `1 → steps·num_segments` sweep, matching the single-segment worker-UI
/// contract the testkit monotonicity check encodes.
fn segment_step_progress(
    seg_idx: usize,
    i: usize,
    steps: usize,
    num_segments: usize,
) -> (u32, u32) {
    let current = (seg_idx * steps + i + 1) as u32;
    let total = (steps * num_segments) as u32;
    (current, total)
}

/// Tokenize + UMT5-encode `prompt` → `[L, 4096]` (f32, un-padded; the DiT's `embed_text` pads to
/// `text_len`).
fn encode_text(
    te: &Umt5Encoder,
    tok: &TextTokenizer,
    prompt: &str,
    pad_token_id: u32,
    dev: &Device,
) -> CResult<Tensor> {
    let out = tok
        .tokenize(prompt)
        .map_err(|e| CandleError::Msg(format!("scail2: tokenize: {e}")))?;
    let mut ids: Vec<u32> = out.ids.iter().map(|&i| i as u32).collect();
    if ids.is_empty() {
        // The gen_core tokenizer short-circuits an empty prompt to zero ids, but UMT5/T5 encode the
        // empty string as a single token. A 0-length sequence here would build a degenerate `(1,1)`
        // tensor (the old `.max(1)` padded the *shape* but not the data), and the f32 embedding gather
        // over zero indices is a 0-element CUDA `index_select` that reads out of bounds →
        // `CUDA_ERROR_ILLEGAL_ADDRESS` (it surfaced deferred at the next cublas call). Emit one pad
        // token so the unconditional (empty negative-prompt) branch encodes a valid 1-token context.
        ids.push(pad_token_id);
    }
    let len = ids.len();
    let input_ids = Tensor::from_vec(ids, (1, len), dev)?;
    let embeds = te.encode(&input_ids)?; // [1, L, 4096]
    let (_, l, d) = embeds.dims3()?;
    Ok(embeds.reshape((l, d))?)
}

/// Build the SCAIL-2 UMT5 tokenizer from an explicit `tokenizer.json` path **once** (sc-8991 /
/// F-011). The legacy candle snapshot stores it under `tokenizer/tokenizer.json`; the shared
/// `SceneWorks/scail2-mlx` tier stores the byte-identical tokenizer at the tier root. Keeping this
/// path-shaped lets both layouts share one parser without copying the file.
pub fn build_tokenizer_from_path(
    tokenizer_path: PathBuf,
    te_cfg: &TextEncoderConfig,
) -> CResult<TextTokenizer> {
    TextTokenizer::from_file(
        tokenizer_path,
        TokenizerConfig {
            max_length: te_cfg.max_length,
            pad_token_id: te_cfg.pad_token_id,
            chat_template: ChatTemplate::None,
            pad_to_max_length: false,
        },
    )
    .map_err(|e| CandleError::Msg(format!("scail2: load tokenizer: {e}")))
}

/// Legacy component-directory wrapper retained for caller compatibility.
pub fn build_tokenizer(root: &Path, te_cfg: &TextEncoderConfig) -> CResult<TextTokenizer> {
    build_tokenizer_from_path(root.join("tokenizer/tokenizer.json"), te_cfg)
}

/// Run the full SCAIL-2 generation for `job` against the resident `comps`. `cancel` is polled each
/// denoise step.
#[allow(clippy::too_many_arguments)]
pub fn generate(
    comps: &Components,
    te_cfg: &TextEncoderConfig,
    job: &Scail2Job,
    cancel: &CancelFlag,
    on_progress: &mut dyn FnMut(Progress),
) -> CResult<GenerationOutput> {
    generate_scoped(|| Ok(comps), te_cfg, job, cancel, None, on_progress)
}

pub(crate) fn generate_scoped<C: Deref<Target = Components>>(
    load: impl Fn() -> CResult<C>,
    te_cfg: &TextEncoderConfig,
    job: &Scail2Job,
    cancel: &CancelFlag,
    mut scope: Option<&mut dyn MemoryRequestScope>,
    on_progress: &mut dyn FnMut(Progress),
) -> CResult<GenerationOutput> {
    if job.driving_frames.is_empty() {
        return Err(CandleError::Msg(
            "scail2: a driving video is required".into(),
        ));
    }
    if job.driving_masks.len() != job.driving_frames.len() {
        return Err(CandleError::Msg(format!(
            "scail2: driving_masks ({}) must match driving_frames ({})",
            job.driving_masks.len(),
            job.driving_frames.len()
        )));
    }
    let segments = job_segments(job)?;
    if job.steps == 0 {
        return Err(CandleError::Msg("scail2: steps must be positive".into()));
    }
    // This is the exact geometry all image and latent paths consume. Keeping validation in the
    // conversion makes it impossible for the public job API to snap before rendering.
    let (tw, th) = validate_job_geometry(job.width, job.height)?;
    let cfg_disabled = job.guidance <= 1.0;
    candle_gen::execution::policy(
        "scail2_segment_policy",
        serde_json::json!({
            "length": job.segment_len, "overlap": job.segment_overlap,
            "guidance_bits": job.guidance.to_bits(), "cfg_disabled": cfg_disabled,
            "replace": job.replace_flag, "vae": "z16_causal_streaming_no_spatial_tiles"
        }),
    );

    let mut out_pieces: Vec<Tensor> = Vec::new();
    let mut prev_history_pixel: Option<Tensor> = None;

    // Progress is reported **globally** across all segments (F-125 / sc-11225) — see
    // `segment_step_progress`. `Decoding` is likewise emitted once (on the final segment) rather than
    // mid-sweep, so the step count reaches 100% before the single decode signal.
    let seg_count = segments.len();
    let mut prepared = None;

    for segment in &segments {
        let seg_idx = segment.identity.pass;
        let (seg_start, seg_end) = (segment.source.start, segment.source.end);
        let mut sched = FlowScheduler::new(job.sampler, job.steps, job.shift);
        let phase = segment.denoise_phase(
            job.sampler,
            sched.sigmas().iter().map(|&sigma| sigma as f32).collect(),
            job.seed,
        );
        let active = phase.enter(
            cancel,
            scope
                .as_mut()
                .map(|s| &mut **s as &mut dyn MemoryRequestScope),
        )?;
        // This route loads all resident components together. Its first admitted segment owns
        // the load and conditioning. Later segments reuse the request's prepared inputs.
        let (comps, conditioning) = match &mut prepared {
            Some(prepared) => prepared,
            None => {
                let comps = load()?;
                let conditioning = prepare(
                    &comps,
                    te_cfg,
                    job,
                    cancel,
                    on_progress,
                    (tw, th),
                    cfg_disabled,
                )?;
                prepared.insert((comps, conditioning))
            }
        };
        let (lat_h, lat_w) = conditioning.latent_size;
        let dev = comps.dit.device();
        // Pose latent (half spatial res) + driving mask for this segment.
        let pose_seg = conditioning
            .driving
            .narrow(0, seg_start, seg_end - seg_start)?; // [T,3,H,W]
        let pose_half = downsample_half(&pose_seg)?; // [T,3,H/2,W/2]
        let pose_cthw = pose_half
            .permute((1, 0, 2, 3))?
            .contiguous()?
            .to_device(dev)?; // one upload: [3,T,H/2,W/2]
        let pose_latent = vae_encode_cthw(&comps.vae, &pose_cthw)?; // [16,T_lat,h/2,w/2]
        let lat_t = pose_latent.dim(1)?;
        if lat_t != phase.latent_frames as usize {
            return Err(CandleError::Msg(format!(
                "scail2: pose encoder produced {lat_t} latent frames for {} source frames, expected {}",
                seg_end - seg_start, phase.latent_frames
            )));
        }

        let dmask_seg = conditioning
            .driving_mask
            .narrow(1, seg_start, seg_end - seg_start)?; // [3,T,H,W]
        let dmask_half = downsample_half(&dmask_seg)?.to_device(dev)?; // one upload: [3,T,H/2,W/2]
                                                                       // Keep threshold/pool/pack on the model device exactly as before; only its input transfer
                                                                       // moved from full resolution to half resolution.
        let driving_masks = extract_and_compress_mask_to_latent(&dmask_half, TEMPORAL_STRIDE)?;

        // ref_masks = ref_mask_28 ++ zero null-noisy-mask over the latent length.
        let null_noisy = Tensor::zeros((28, lat_t, lat_h, lat_w), DType::F32, dev)?;
        let ref_masks = Tensor::cat(&[&conditioning.ref_mask_28, &null_noisy], 1)?; // [28,1+T,h,w]

        // Clean-history latent + i2v mask for segments after the first.
        let (history_latent, history_mask) = match &prev_history_pixel {
            Some(hp) if seg_idx > 0 => {
                let hl = vae_encode_cthw(&comps.vae, hp)?; // [16,h_t,h,w]
                let h_t = hl.dim(1)?.min(lat_t);
                let ones = Tensor::ones((4, h_t, lat_h, lat_w), DType::F32, dev)?;
                let hm = if h_t < lat_t {
                    let z = Tensor::zeros((4, lat_t - h_t, lat_h, lat_w), DType::F32, dev)?;
                    Tensor::cat(&[&ones, &z], 1)?
                } else {
                    ones
                };
                (Some(hl), Some(hm))
            }
            _ => (None, None),
        };

        // Seeded init noise (per-segment).
        let noise =
            candle_gen_wan::pipeline::create_noise(phase.seed, 16, lat_t, lat_h, lat_w, dev)?
                .reshape((16, lat_t, lat_h, lat_w))?;

        // --- denoise (plain CFG, clean-history pinned) ---
        let latent = denoise_segment(
            &phase,
            seg_count,
            &noise,
            history_latent.as_ref(),
            cancel,
            on_progress,
            |i, x, latent| {
                let t = sched.timestep(i);
                let mut inp = Scail2Inputs {
                    x,
                    ref_latent: &conditioning.ref_latent,
                    ref_masks: &ref_masks,
                    pose_latent: &pose_latent,
                    driving_masks: &driving_masks,
                    history_mask: history_mask.as_ref(),
                    additional_ref_latent: conditioning.additional_ref_latent.as_ref(),
                    additional_ref_masks: conditioning.additional_ref_masks.as_ref(),
                    clip_fea: &conditioning.clip_fea,
                    context: &conditioning.context,
                    t,
                    replace_flag: job.replace_flag,
                };
                let pred_cond = comps.dit.forward(&inp)?;
                let pred = if cfg_disabled {
                    pred_cond
                } else {
                    if cancel.is_cancelled() {
                        return Err(CandleError::Canceled);
                    }
                    inp.context = &conditioning.context_null;
                    let pred_uncond = comps.dit.forward(&inp)?;
                    cfg_combine(&pred_uncond, &pred_cond, job.guidance)?
                };
                Ok(sched.step(&pred, latent)?)
            },
        )?;
        active.finish()?;

        // --- decode this segment → pixels; stitch + carry history ---
        // Emit `Decoding` once, after the final segment's denoise completes, so the global step sweep
        // reaches 100% before the single decode signal (rather than firing mid-sweep per segment).
        if seg_idx + 1 == seg_count {
            on_progress(Progress::Decoding);
        }
        let video = comps
            .vae
            .decode_with_cancel(&latent.reshape((1, 16, lat_t, lat_h, lat_w))?, cancel)?; // [1,3,T_out,H,W]
        let (_, vc, vt, vh, vw) = video.dims5()?;
        let seg_video = video.reshape((vc, vt, vh, vw))?; // [3,T_out,H,W]
        let t_out = seg_video.dim(1)?;

        let keep_from = if seg_idx == 0 { 0 } else { job.segment_overlap };
        let piece = seg_video.narrow(1, keep_from, t_out - keep_from)?;
        out_pieces.push(piece);

        if seg_idx + 1 < segments.len() {
            let ov = job.segment_overlap;
            prev_history_pixel = Some(seg_video.narrow(1, t_out - ov, ov)?.contiguous()?);
        }
    }

    let piece_refs: Vec<&Tensor> = out_pieces.iter().collect();
    let full = Tensor::cat(&piece_refs, 1)?; // [3,T_total,H,W]
    let (fc, ft, fh, fw) = full.dims4()?;
    let frames = frames_to_images(&full.reshape((1, fc, ft, fh, fw))?)?;
    Ok(GenerationOutput::Video {
        frames,
        fps: job.fps,
        audio: None,
    })
}

struct Prepared {
    ref_latent: Tensor,
    ref_mask_28: Tensor,
    driving: Tensor,
    driving_mask: Tensor,
    additional_ref_latent: Option<Tensor>,
    additional_ref_masks: Option<Tensor>,
    context: Tensor,
    context_null: Tensor,
    clip_fea: Tensor,
    latent_size: (usize, usize),
}

fn prepare(
    comps: &Components,
    te_cfg: &TextEncoderConfig,
    job: &Scail2Job,
    cancel: &CancelFlag,
    on_progress: &mut dyn FnMut(Progress),
    render_size: (usize, usize),
    cfg_disabled: bool,
) -> CResult<Prepared> {
    let (tw, th) = render_size;
    let dev = comps.dit.device();
    // --- decode + resize all pixel inputs to (tw, th) ---
    let ref_chw = image_to_chw(job.reference.image, tw, th, Interp::Bicubic, dev)?; // [3,H,W]
    let ref_mask_chw = image_to_chw(job.reference.mask, tw, th, Interp::Bilinear, dev)?; // [3,H,W]
    let driving = stack_frames(job.driving_frames, tw, th)?; // CPU [T,3,H,W]
    let driving_mask = stack_masks(job.driving_masks, tw, th)?; // CPU [3,T,H,W]

    // Reference char latent + its 28-ch mask (1 latent frame).
    let ref_latent = vae_encode_cthw(&comps.vae, &ref_chw.reshape((3, 1, th, tw))?)?;
    let ref_mask_28 = extract_and_compress_mask_to_latent(
        &ref_mask_chw.reshape((3, 1, th, tw))?,
        TEMPORAL_STRIDE,
    )?;
    let lat_h = ref_latent.dim(2)?;
    let lat_w = ref_latent.dim(3)?;

    // Extra characters (multi-reference): cat latents + masks on the frame axis.
    let (additional_ref_latent, additional_ref_masks) = if job.additional.is_empty() {
        (None, None)
    } else {
        let encoded = map_additional_references(
            &job.additional,
            cancel,
            |c| {
                let img =
                    image_to_chw(c.image, tw, th, Interp::Bicubic, dev)?.reshape((3, 1, th, tw))?;
                vae_encode_cthw(&comps.vae, &img)
            },
            |c| {
                let mk =
                    image_to_chw(c.mask, tw, th, Interp::Bilinear, dev)?.reshape((3, 1, th, tw))?;
                Ok(extract_and_compress_mask_to_latent(&mk, TEMPORAL_STRIDE)?)
            },
        )?;
        let (lats, masks): (Vec<_>, Vec<_>) = encoded.into_iter().unzip();
        let lr: Vec<&Tensor> = lats.iter().collect();
        let mr: Vec<&Tensor> = masks.iter().collect();
        (Some(Tensor::cat(&lr, 1)?), Some(Tensor::cat(&mr, 1)?))
    };
    if cancel.is_cancelled() {
        return Err(CandleError::Canceled);
    }

    // --- UMT5 text encode + CLIP reference-image features (once) ---
    // The tokenizer is loaded+parsed once at component load (sc-8991 / F-011) — reuse the cached one.
    let tok = &comps.tok;
    let pad_id = te_cfg.pad_token_id.max(0) as u32;
    let (context, context_null) = candle_gen::gen_core::with_prompt_encoding(
        on_progress,
        || -> std::result::Result<_, candle_gen::CandleError> {
            let context = encode_text(&comps.te, tok, job.prompt, pad_id, dev)?;
            let context_null = if cfg_disabled {
                context.clone()
            } else {
                encode_text(&comps.te, tok, job.negative_prompt, pad_id, dev)?
            };
            Ok((context, context_null))
        },
    )?;
    let clip_fea = {
        let pixel = clip_preprocess(&ref_chw.reshape((1, 3, th, tw))?, 224)?;
        comps.clip.encode(&pixel)?
    };

    Ok(Prepared {
        ref_latent,
        ref_mask_28,
        driving,
        driving_mask,
        additional_ref_latent,
        additional_ref_masks,
        context,
        context_null,
        clip_fea,
        latent_size: (lat_h, lat_w),
    })
}

/// Pin decoded-and-reencoded history on both sides of the native scheduler update.
fn denoise_segment(
    phase: &VideoDenoisePhase<(), Sampler>,
    segment_count: usize,
    noise: &Tensor,
    history: Option<&Tensor>,
    cancel: &CancelFlag,
    on_progress: &mut dyn FnMut(Progress),
    mut advance: impl FnMut(usize, &Tensor, &Tensor) -> CResult<Tensor>,
) -> CResult<Tensor> {
    let segment = phase.identity.pass;
    let steps = phase.sigmas.len() - 1;
    let mut latent = apply_clean_history(noise, history)?;
    run_video_steps(&phase.sigmas, cancel, |step| -> CResult<()> {
        let x = apply_clean_history(&latent, history)?;
        let updated = advance(step.index, &x, &latent)?;
        latent = apply_clean_history(&updated, history)?;
        let (current, total) = segment_step_progress(segment, step.index, steps, segment_count);
        on_progress(Progress::Step { current, total });
        Ok(())
    })?;
    Ok(latent)
}

fn job_segments(job: &Scail2Job) -> candle_gen::gen_core::Result<Vec<Scail2Segment>> {
    let frames = u32::try_from(job.driving_frames.len()).map_err(|_| {
        candle_gen::gen_core::Error::Unsupported("scail2: too many driving frames".into())
    })?;
    let reference_count = u32::try_from(job.additional.len())
        .ok()
        .and_then(|n| n.checked_add(1))
        .ok_or_else(|| {
            candle_gen::gen_core::Error::Unsupported("scail2: too many character images".into())
        })?;
    resolve_segments(
        SegmentTail::AlignedRemainder,
        MemoryGeometry {
            width: job.width,
            height: job.height,
            batch: 1,
            frames,
            reference_count,
        },
        job.segment_len,
        job.segment_overlap,
    )
}

#[cfg(test)]
mod tests {
    use super::{
        image_to_chw_host, map_additional_references, segment_step_progress, stack_frames,
        stack_masks, validate_job_geometry, CharacterRef, Image, Interp, TEMPORAL_STRIDE,
    };
    use crate::preprocess::extract_and_compress_mask_to_latent;
    use crate::resize::{downsample_half, interpolate};
    use candle_gen::candle_core::{Device, Tensor};
    use candle_gen::gen_core::runtime::CancelFlag;
    use candle_gen::CandleError;

    fn validate_segment_params(
        total: usize,
        window: usize,
        overlap: usize,
    ) -> candle_gen::gen_core::Result<Vec<super::Scail2Segment>> {
        super::resolve_segments(
            super::SegmentTail::AlignedRemainder,
            super::MemoryGeometry {
                width: 32,
                height: 32,
                batch: 1,
                frames: total as u32,
                reference_count: 1,
            },
            window,
            overlap,
        )
    }

    fn build_segments(total: usize, window: usize, overlap: usize) -> Vec<(usize, usize)> {
        validate_segment_params(total, window, overlap)
            .unwrap()
            .into_iter()
            .map(|s| (s.source.start, s.source.end))
            .collect()
    }

    #[test]
    fn native_segment_preserves_window_seed_progress_and_clean_history() {
        use super::{denoise_segment, job_segments, Scail2Job};
        use candle_gen::gen_core::Progress;

        let image = Image {
            width: 64,
            height: 32,
            pixels: Vec::new(),
        };
        let frames = vec![image.clone(); 25];
        let job = Scail2Job {
            prompt: "",
            negative_prompt: "",
            width: 64,
            height: 32,
            reference: CharacterRef {
                image: &image,
                mask: &image,
            },
            additional: vec![CharacterRef {
                image: &image,
                mask: &image,
            }],
            driving_frames: &frames,
            driving_masks: &frames,
            replace_flag: true,
            seed: u64::MAX,
            steps: 2,
            shift: 5.0,
            guidance: 5.0,
            sampler: super::Sampler::parse(Some("dpm++")),
            fps: 16,
            segment_len: 13,
            segment_overlap: 5,
        };
        let segments = job_segments(&job).unwrap();
        let windows: Vec<_> = segments
            .iter()
            .map(|s| (s.source.start, s.source.end))
            .collect();
        assert_eq!(windows, vec![(0, 13), (8, 21), (16, 25)]);
        for (index, (start, end)) in windows.iter().enumerate() {
            let phase = segments[index].denoise_phase(job.sampler, vec![1., 0.5, 0.], job.seed);
            assert_eq!(phase.identity.pass, index);
            assert_eq!(phase.identity.geometry.width, 64);
            assert_eq!(phase.identity.geometry.height, 32);
            assert_eq!(phase.identity.geometry.reference_count, 2);
            assert_eq!(phase.identity.geometry.frames as usize, end - start);
            assert_eq!(phase.latent_frames as usize, (end - start - 1) / 4 + 1);
            assert_eq!(phase.seed, u64::MAX.wrapping_add(index as u64));
            assert_eq!(phase.solver, super::Sampler::UniPC);
        }

        // At the real generation entry, a refused denoise geometry must stop the
        // lazy all-components load, including on a cold resident cache.
        let admitted = candle_gen::gen_core::MemoryGeometry {
            width: 32,
            height: 32,
            batch: 1,
            frames: 13,
            reference_count: 2,
        };
        let config = candle_gen::request_scope::CandleRequestScopeConfig::new(
            "scail2-test",
            Device::Cpu,
            admitted,
            None,
            false,
            0,
            |_, _, _| Ok(()),
        )
        .unwrap();
        let mut scope = candle_gen::request_scope::CandleRequestScopeCore::new(config);
        let refused = super::generate_scoped(
            || -> super::CResult<&super::Components> { panic!("loaded before admission") },
            &super::TextEncoderConfig::umt5_xxl(),
            &job,
            &CancelFlag::new(),
            Some(&mut scope),
            &mut |_| panic!("progress before admission"),
        )
        .unwrap_err();
        assert!(refused.to_string().contains("geometry"), "{refused}");

        let short_windows = Scail2Job {
            driving_frames: &frames[..13],
            driving_masks: &frames[..13],
            segment_len: 5,
            segment_overlap: 1,
            ..job
        };
        let segments = job_segments(&short_windows).unwrap();
        assert_eq!(segments.len(), 3);
        assert_eq!(segments[1].latent_frames, 2);
        // Poison the clean first frame on every update. The next forward, solver input,
        // and returned segment must see the restored frame while the other frame evolves.
        let dev = Device::Cpu;
        let noise = Tensor::from_vec(vec![1.0f32, 2.0], (1, 2, 1, 1), &dev).unwrap();
        let history = Tensor::from_vec(vec![19.0f32], (1, 1, 1, 1), &dev).unwrap();
        let values = |t: &Tensor| t.flatten_all().unwrap().to_vec1::<f32>().unwrap();
        let phase = segments[1].denoise_phase(job.sampler, vec![1., 0.5, 0.], job.seed);
        let mut progress = Vec::new();
        let solved = denoise_segment(
            &phase,
            3,
            &noise,
            Some(&history),
            &CancelFlag::new(),
            &mut |event| match event {
                Progress::Step { current, total } => progress.push((current, total)),
                _ => panic!("unexpected progress during a native segment"),
            },
            |i, x, previous| {
                assert_eq!(values(x), [19.0, 2.0 + i as f32]);
                assert_eq!(values(previous), values(x));
                Ok(Tensor::from_vec(
                    vec![-10.0, 3.0 + i as f32],
                    (1, 2, 1, 1),
                    &dev,
                )?)
            },
        )
        .unwrap();
        assert_eq!(values(&solved), [19.0, 4.0]);
        assert_eq!(progress, [(3, 6), (4, 6)]);
    }

    #[test]
    fn public_job_geometry_is_exact_or_rejected() {
        assert_eq!(
            validate_job_geometry(1280, 704).expect("on-grid geometry is exact"),
            (1280, 704)
        );
        for (width, height) in [(1280, 730), (730, 1280), (0, 704), (704, 0)] {
            let err = validate_job_geometry(width, height)
                .expect_err("the public low-level job must never be silently snapped")
                .to_string();
            assert!(
                err.contains("non-zero multiples of 32")
                    && err.contains(&format!("{width}×{height}")),
                "got: {err}"
            );
        }
    }

    /// The pre-sc-12517 ordering: upload the native-sized normalized tensor to `dev`, then call the
    /// host-backed resize, which reads it back and recreates the output on `dev`.
    fn legacy_image_to_chw(
        img: &Image,
        tw: usize,
        th: usize,
        mode: Interp,
        dev: &Device,
    ) -> Tensor {
        let (iw, ih) = (img.width as usize, img.height as usize);
        let px: Vec<f32> = img.pixels.iter().map(|&p| p as f32 / 127.5 - 1.0).collect();
        let chw = Tensor::from_vec(px, (ih, iw, 3), dev)
            .unwrap()
            .permute((2, 0, 1))
            .unwrap();
        let nchw = chw.reshape((1, 3, ih, iw)).unwrap();
        let out = if (ih, iw) != (th, tw) {
            interpolate(&nchw, th, tw, mode).unwrap()
        } else {
            nchw
        };
        out.reshape((3, th, tw)).unwrap()
    }

    /// The pre-sc-12517 half-resize contract: the host kernel recreates its result on the input
    /// device immediately. On the CPU test lane this has the same values while remaining an
    /// independent route through `interpolate`; on CUDA it was the redundant host→device leg.
    fn legacy_downsample_half(x: &Tensor) -> Tensor {
        let (_, _, h, w) = x.dims4().unwrap();
        interpolate(x, h / 2, w / 2, Interp::Bilinear).unwrap()
    }

    fn sample_image(width: u32, height: u32, salt: u8) -> Image {
        let len = width as usize * height as usize * 3;
        Image {
            width,
            height,
            pixels: (0..len)
                .map(|i| (i as u8).wrapping_mul(37).wrapping_add(salt))
                .collect(),
        }
    }

    fn values(t: &Tensor) -> Vec<f32> {
        t.flatten_all().unwrap().to_vec1::<f32>().unwrap()
    }

    // Shipped defaults (see `pipeline.rs`).
    const LEN: usize = 81;
    const OVERLAP: usize = 5;

    #[test]
    fn host_first_image_resize_is_bit_identical_to_legacy_order() {
        let image = sample_image(5, 4, 11);
        for mode in [Interp::Bicubic, Interp::Bilinear] {
            let legacy = legacy_image_to_chw(&image, 7, 6, mode, &Device::Cpu);
            let host = image_to_chw_host(&image, 7, 6, mode).unwrap();
            assert_eq!(host.dims(), legacy.dims());
            assert_eq!(values(&host), values(&legacy));
            assert!(matches!(host.device(), Device::Cpu));
        }
    }

    #[test]
    fn driving_stacks_and_half_resizes_remain_cpu_until_transfer() {
        let frames: Vec<Image> = (0..5).map(|i| sample_image(5, 4, 3 + i * 16)).collect();
        let masks: Vec<Image> = (0..5).map(|i| sample_image(5, 4, 71 + i * 16)).collect();
        let driving = stack_frames(&frames, 16, 16).unwrap();
        let driving_mask = stack_masks(&masks, 16, 16).unwrap();
        assert_eq!(driving.dims(), &[5, 3, 16, 16]);
        assert_eq!(driving_mask.dims(), &[3, 5, 16, 16]);
        assert!(matches!(driving.device(), Device::Cpu));
        assert!(matches!(driving_mask.device(), Device::Cpu));

        let pose_half = downsample_half(&driving).unwrap();
        let mask_half = downsample_half(&driving_mask).unwrap();
        assert_eq!(pose_half.dims(), &[5, 3, 8, 8]);
        assert_eq!(mask_half.dims(), &[3, 5, 8, 8]);
        assert!(matches!(pose_half.device(), Device::Cpu));
        assert!(matches!(mask_half.device(), Device::Cpu));

        let legacy_pose_half = legacy_downsample_half(&driving);
        let legacy_mask_half = legacy_downsample_half(&driving_mask);

        // These are the exact weight-free conditioning boundaries consumed by the models. Pose is
        // permuted/contiguous immediately before VAE encode; mask is thresholded into seven colors,
        // 8× pooled, and temporally packed immediately before the DiT. Bit equality here proves the
        // transfer-only optimization cannot change generated output without invoking model weights.
        let pose_condition = pose_half
            .permute((1, 0, 2, 3))
            .unwrap()
            .contiguous()
            .unwrap();
        let legacy_pose_condition = legacy_pose_half
            .permute((1, 0, 2, 3))
            .unwrap()
            .contiguous()
            .unwrap();
        assert_eq!(pose_condition.dims(), &[3, 5, 8, 8]);
        assert_eq!(values(&pose_condition), values(&legacy_pose_condition));

        let mask_condition =
            extract_and_compress_mask_to_latent(&mask_half, TEMPORAL_STRIDE).unwrap();
        let legacy_mask_condition =
            extract_and_compress_mask_to_latent(&legacy_mask_half, TEMPORAL_STRIDE).unwrap();
        assert_eq!(mask_condition.dims(), &[28, 2, 1, 1]);
        assert_eq!(values(&mask_condition), values(&legacy_mask_condition));
    }

    /// Every emitted window is VAE-temporal-aligned (`4k + 1`) so the z16 encoder accepts it, and non
    /// empty.
    fn assert_windows_valid(segs: &[(usize, usize)]) {
        for &(s, e) in segs {
            assert!(e > s, "empty/backwards window ({s}, {e})");
            let n = e - s;
            assert_eq!(
                n % TEMPORAL_STRIDE,
                1 % TEMPORAL_STRIDE,
                "window len {n} is not 4k+1"
            );
        }
    }

    #[test]
    fn single_segment_when_clip_fits() {
        // total <= len: one VAE-aligned window.
        let segs = build_segments(50, LEN, OVERLAP);
        assert_eq!(segs, vec![(0, 49)]); // vae_align(50) = 49
        assert_windows_valid(&segs);
    }

    #[test]
    fn single_frame_and_tiny_clips_do_not_panic() {
        assert_eq!(build_segments(1, LEN, OVERLAP), vec![(0, 1)]);
        assert_eq!(build_segments(2, LEN, OVERLAP), vec![(0, 1)]); // vae_align(2) = 1
        assert_eq!(build_segments(5, LEN, OVERLAP), vec![(0, 5)]);
    }

    #[test]
    fn empty_clip_refuses_segment_resolution() {
        assert!(validate_segment_params(0, LEN, OVERLAP).is_err());
    }

    /// The core F-006 regression: a clip a bit longer than one window must NOT silently drop most of its
    /// tail. With len=81, overlap=5, stride=76 the old code emitted only `[0,81)` and dropped frames
    /// 81..120 (39 frames). The tail may still lose up to `TEMPORAL_STRIDE-1` frames to VAE alignment —
    /// the same unavoidable rounding the single-segment path applies — but no more.
    #[test]
    fn trailing_frames_are_not_dropped_just_over_one_window() {
        let total = 120;
        let segs = build_segments(total, LEN, OVERLAP);
        assert_windows_valid(&segs);
        // Full window then a shortened tail.
        assert_eq!(segs.first().copied(), Some((0, 81)));
        let (_, last_end) = *segs.last().unwrap();
        // Covers to the clip end, modulo VAE alignment (≤ 3 frames lost, not a whole window).
        assert_eq!(last_end, 117);
        assert!(total - last_end < TEMPORAL_STRIDE);
    }

    /// A long clip (the review's motivating case): frames must be covered essentially end-to-end (only
    /// the ≤3-frame VAE-alignment tail is dropped) and every interior join must be exactly `overlap`
    /// frames wide so the fixed `keep_from = overlap` stitch is correct.
    #[test]
    fn long_clip_covers_all_frames_with_uniform_overlap() {
        let total = 200;
        let segs = build_segments(total, LEN, OVERLAP);
        assert_windows_valid(&segs);
        assert!(segs.len() >= 3, "expected multiple windows, got {segs:?}");

        // Coverage: windows start at 0 and each next window starts before the previous ends, with the
        // join exactly `overlap` frames wide.
        assert_eq!(segs[0].0, 0);
        for pair in segs.windows(2) {
            let (prev_start, prev_end) = pair[0];
            let (next_start, _) = pair[1];
            assert!(next_start < prev_end, "gap between windows (no overlap)");
            assert!(next_start > prev_start);
            let overlap = prev_end - next_start;
            assert_eq!(overlap, OVERLAP, "join overlap must equal segment_overlap");
        }
        // The last window reaches the VAE-aligned clip end: at most TEMPORAL_STRIDE-1 frames lost.
        let (_, last_end) = *segs.last().unwrap();
        assert_eq!(last_end, 197);
        assert!(total - last_end < TEMPORAL_STRIDE);
    }

    /// Exact tiling: when the windows land flush on the clip end there must be no spurious zero/duplicate
    /// tail window.
    #[test]
    fn exact_multiple_boundary_has_no_spurious_tail() {
        // stride = 76. Two windows: [0,81) and [76,157). total = 157 → second window ends exactly at
        // total, remainder is 0, no tail appended.
        let total = 157;
        let segs = build_segments(total, LEN, OVERLAP);
        assert_windows_valid(&segs);
        assert_eq!(segs, vec![(0, 81), (76, 157)]);
        assert_eq!(segs.last().unwrap().1, total);
    }

    /// Remainder smaller than the overlap contributes nothing new (it lives entirely inside the previous
    /// window's tail), so no degenerate tail window is emitted.
    #[test]
    fn tiny_remainder_within_overlap_is_not_appended() {
        // total = 159: after [0,81),[76,157) the remainder is frames 157..159 (2 frames), vae_align = 1
        // which is < overlap(5) → skipped.
        let total = 159;
        let segs = build_segments(total, LEN, OVERLAP);
        assert_windows_valid(&segs);
        assert_eq!(segs, vec![(0, 81), (76, 157)]);
    }

    /// Underflow guard: a pathological direct-consumer job with `overlap >= len` must not underflow
    /// `len - overlap` (panic in debug / wrap in release) — segment resolution returns an error and `generate`
    /// rejects it up front.
    #[test]
    fn overlap_ge_len_does_not_underflow() {
        assert!(validate_segment_params(200, 81, 81).is_err());
        assert!(validate_segment_params(200, 81, 200).is_err());
        assert!(validate_segment_params(200, 0, 0).is_err());
    }

    /// Replays the exact `(current, total)` sequence `generate`'s denoise loop emits over a multi-segment
    /// job and asserts the worker-UI progress contract (F-125 / sc-11225): one strictly monotonic sweep
    /// that never resets per segment and lands exactly on `total` at the final step.
    fn assert_global_progress_monotonic(steps: usize, num_segments: usize) {
        let expected_total = (steps * num_segments) as u32;
        let mut sweep = Vec::new();
        for seg_idx in 0..num_segments {
            for i in 0..steps {
                let (current, total) = segment_step_progress(seg_idx, i, steps, num_segments);
                assert_eq!(total, expected_total, "total must be the GLOBAL step count");
                sweep.push(current);
            }
        }
        // First reported step is 1, last is exactly `total`.
        assert_eq!(sweep.first().copied(), Some(1));
        assert_eq!(sweep.last().copied(), Some(expected_total));
        // Strictly increasing by 1 — no per-segment reset to 1, no backwards jump at a segment boundary.
        for pair in sweep.windows(2) {
            assert_eq!(
                pair[1],
                pair[0] + 1,
                "progress must advance by exactly 1 and never restart per segment"
            );
        }
        assert_eq!(sweep.len(), steps * num_segments);
    }

    /// The core F-125 regression: with >1 segment the old per-segment counting reset `current` to 1 at
    /// every boundary (percent-complete jumped backwards). Progress must instead be one global 1→total
    /// sweep. A driving clip of 200 frames tiles into multiple 81-frame segments (see
    /// `long_clip_covers_all_frames_with_uniform_overlap`).
    #[test]
    fn multi_segment_progress_is_global_and_monotonic() {
        let num_segments = build_segments(200, LEN, OVERLAP).len();
        assert!(
            num_segments >= 3,
            "expected a multi-segment tiling for the regression"
        );
        assert_global_progress_monotonic(30, num_segments);
        // A couple of other shapes for good measure.
        assert_global_progress_monotonic(1, 4);
        assert_global_progress_monotonic(50, 2);
    }

    /// A single-segment clip (the common short case) is unchanged: a plain 1→steps sweep with
    /// `total == steps`.
    #[test]
    fn single_segment_progress_matches_step_count() {
        let steps = 20;
        for i in 0..steps {
            let (current, total) = segment_step_progress(0, i, steps, 1);
            assert_eq!(current, (i + 1) as u32);
            assert_eq!(total, steps as u32);
        }
    }

    #[test]
    fn cancellation_during_one_extra_vae_stops_before_mask_and_later_conditioning() {
        let first = sample_image(2, 2, 1);
        let refs = [CharacterRef {
            image: &first,
            mask: &first,
        }];
        let cancel = CancelFlag::new();
        let mut vae_passes = 0;
        let mut mask_passes = 0;
        let mut later_conditioning_entered = false;
        let result = map_additional_references(
            &refs,
            &cancel,
            |_| {
                vae_passes += 1;
                cancel.cancel();
                Ok(())
            },
            |_| {
                mask_passes += 1;
                Ok(())
            },
        );
        if result.is_ok() {
            later_conditioning_entered = true;
        }
        let error = result.expect_err("cancellation during VAE must stop the single extra pair");
        assert!(matches!(error, CandleError::Canceled), "got: {error}");
        assert_eq!(vae_passes, 1, "the extra reference entered its VAE pass");
        assert_eq!(mask_passes, 0, "mask preprocessing must not begin");
        assert!(
            !later_conditioning_entered,
            "later text/image conditioning must not begin"
        );
    }

    #[test]
    fn cancellation_during_last_extra_mask_stops_before_later_conditioning() {
        let first = sample_image(2, 2, 1);
        let refs = [CharacterRef {
            image: &first,
            mask: &first,
        }];
        let cancel = CancelFlag::new();
        let mut later_conditioning_entered = false;
        let result = map_additional_references(
            &refs,
            &cancel,
            |_| Ok(()),
            |_| {
                cancel.cancel();
                Ok(())
            },
        );
        if result.is_ok() {
            later_conditioning_entered = true;
        }
        let error = result.expect_err("cancellation after the last mask must stop conditioning");
        assert!(matches!(error, CandleError::Canceled), "got: {error}");
        assert!(
            !later_conditioning_entered,
            "later text/image conditioning must not begin"
        );
    }
}
