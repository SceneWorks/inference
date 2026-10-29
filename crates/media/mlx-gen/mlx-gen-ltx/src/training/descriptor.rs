//! Trainer descriptors, request validation and the LTX-2.5 refusal floors (subject mask,
//! depth anchoring) shared by `validate` and the lazy loaders.

use super::*;

pub(super) fn trainer_descriptor() -> TrainerDescriptor {
    trainer_descriptor_for(MODEL_ID)
}

pub(super) fn trainer_descriptor_25() -> TrainerDescriptor {
    trainer_descriptor_for(MODEL_25_ID)
}

pub(super) fn trainer_descriptor_for(id: &'static str) -> TrainerDescriptor {
    TrainerDescriptor {
        id,
        family: "ltx",
        backend: "mlx",
        modality: Modality::Video,
        supports_lora: true,
        // The reference LTX MLX trainer is LoRA-only; LoKr training is unsupported (see `validate`).
        supports_lokr: false,
        // No control-branch training path (F-006). The LTX trainer must reject a control request
        // rather than silently training a plain LoRA (F-055).
        supports_control: false,
        // Adapter-only: no full base fine-tune path (sc-14056). The shared
        // `validate_full_finetune_request` floor makes a `full_finetune` request a typed reject.
        supports_full_finetune: false,
        max_reference_images: 0,
        // Epic 2123 S2 (sc-24827): weight noise + gradient noise at the adapter optimizer
        // update.
        // sc-2127 (epic 2123): multi-resolution buckets on the 2.3 still-image path only (one cached
        // latent + RoPE grid per spatial edge). 2.5 trains on externally prepared latent packs
        // (`ltxPreparedBundlePath`) whose `videoShape`/`audioShape`, condition masks and token plan
        // are fixed by the pack — there are no pixels to re-encode at another edge and
        // `cfg.resolution` is unused — so it keeps `NONE` and the shared floor refuses buckets.
        // sc-24828 (epic 2123): LTX-2.3 trains on decoded, centre-cropped images and honours the
        // subject-masked loss (one weight per cached (item, bucket) latent). LTX-2.5 trains only on
        // preprocessed latent bundles with no image aligned to the latent, so it does not declare
        // it (refused with the reason by [`refuse_ltx25_subject_mask`]).
        // sc-24830 (epic 2123): depth anchoring through TAELTX2.3 (upstream's tiny decoder for
        // LTX-2.3 and LTX-2.5) — on the LTX-2.3 still-image path (dense and block-checkpointed)
        // and on every LTX-2.5 workflow that generates video (the prepared bundle's clean video
        // latent's selected frames decoded, each `k > 0` with its predecessor; audio ignored). The
        // six 2.5 workflows with no generated video are
        // refused with the reason by [`refuse_ltx25_depth_anchoring`].
        // sc-24833 (epic 2123): the VAE anchor (same TAELTX2.3 decode → FLUX.2 encoder taps, per
        // decoded frame) wherever depth anchoring runs — through the same builder and the same
        // video-workflow refusal. No E-LatentLPIPS: no published weights match this latent family.
        techniques: gen_core::train::TrainingTechniques {
            resolution_buckets: id != MODEL_25_ID,
            subject_mask_loss: id == MODEL_ID,
            depth_anchoring: true,
            // sc-24831: the face losses ride the same shared builder arms + x0 decoder.
            identity_loss: true,
            face_landmark_loss: true,
            // sc-24832: the body losses ride the same builder arms as depth anchoring.
            body_proportion_loss: true,
            body_shape_loss: true,
            normal_loss: true,
            vae_anchor_loss: true,
            ..gen_core::train::TrainingTechniques::ADAPTER_NOISE
        },
    }
}

/// sc-24828 — LTX-2.5 cannot honour subject-masked loss: every one of its workflows trains on a
/// preprocessed video/audio latent bundle (`model_options`), with no decodable image whose crop is
/// aligned to the latent, so a single image mask has nothing to line up with. A typed
/// `Unsupported` naming that reason, raised by `validate`, the start of `train` (before any
/// caching) and the weights-free [`validate_ltx25_training_request`] preflight — ahead of the
/// generic technique floor, so the caller sees why.
pub(super) fn refuse_ltx25_subject_mask(id: &str, req: &TrainingRequest) -> gen_core::Result<()> {
    if id == MODEL_25_ID && req.config.subject_mask_loss.is_some() {
        return Err(gen_core::Error::Unsupported(format!(
            "{id} trainer: subject-masked loss needs a training image aligned to the latent, but \
             LTX-2.5 trains on preprocessed video/audio latent bundles that carry none; turn \
             subject-masked loss off for this trainer"
        )));
    }
    // sc-24832: the subject-restricted normal loss reads the same (absent) aligned mask.
    if id == MODEL_25_ID
        && mlx_gen::train::subject_mask::PerceptualSubjectMasks::needed(&req.config)
    {
        return Err(gen_core::Error::Unsupported(format!(
            "{id} trainer: the normal loss restricted to the subject needs a training image aligned \
             to the latent, but LTX-2.5 trains on preprocessed latent bundles that carry none; turn \
             body_losses.normal_restrict_to_subject off for this trainer"
        )));
    }
    Ok(())
}

/// sc-24830 — LTX-2.5 runs depth anchoring on its generated video stream, so it refuses it for the
/// six workflows that generate no video (`v2a_lora`, whose video is frozen conditioning, and the
/// audio-only `t2a_lora`, `audio_extend_lora`, `audio_inpainting_lora`, `audio_suffix_lora`,
/// `a2a_ic_lora`): a typed `Unsupported` naming the workflow, raised by `validate`, the start of
/// `train` (before any caching) and the weights-free [`validate_ltx25_training_request`] preflight
/// — ahead of the generic technique floor, so the caller sees why. Also validates the
/// [`DEPTH_ANCHORING_FRAMES_KEY`] knob.
pub(super) fn refuse_ltx25_depth_anchoring(
    id: &str,
    req: &TrainingRequest,
) -> gen_core::Result<()> {
    // Every decoded-x0 perceptual loss (depth anchoring, the sc-24831 identity / face-landmark
    // losses, the sc-24833 VAE anchor, the sc-24832 body losses) decodes the same generated video
    // frames, so the refusal covers whichever ones are on and names them.
    let pixel_losses = mlx_gen_perceptual::enabled_pixel_aux_losses(&req.config);
    if id != MODEL_25_ID || pixel_losses.is_empty() {
        return Ok(());
    }
    depth_anchoring_frames(&req.config)?;
    // The resolved plan (per-request modality overrides applied) when it parses; else the named
    // workflow's canonical plan — an otherwise-malformed request is reported by the plan
    // validation that follows, but never slips a video-less workflow past this refusal.
    let (workflow, video) = match Ltx25TrainingPlan::from_request(req) {
        Ok(plan) => (plan.workflow, plan.video.map(|v| v.is_generated)),
        Err(_) => {
            let Some(workflow) = req
                .config
                .model_options
                .get("ltxWorkflow")
                .and_then(serde_json::Value::as_str)
                .and_then(|id| Ltx25Workflow::parse(id).ok())
            else {
                return Ok(());
            };
            (workflow, workflow.plan().video.map(|v| v.is_generated))
        }
    };
    if video != Some(true) {
        return Err(gen_core::Error::Unsupported(format!(
            "{id} trainer: depth anchoring and the other decoded-x0 losses ({}) decode the \
             generated video stream, but workflow `{}` generates no video (the video stream is \
             {}); turn them off for this workflow",
            pixel_losses.join(", "),
            workflow.id(),
            if video.is_some() {
                "frozen conditioning"
            } else {
                "absent"
            }
        )));
    }
    Ok(())
}

/// `model_options` key selecting how many latent frames of an LTX-2.5 video clip the depth loss
/// decodes per step (sc-24830). Each selected latent frame decodes through TAELTX2.3 to one pixel
/// frame — frame 0 as a `T = 1` clip, a frame `k > 0` together with its predecessor `k − 1` as a
/// `T = 2` clip, keeping its last output frame; the frames are evenly spaced over the clip's latent
/// frames that carry generated (loss) tokens, fixed per example. Positive integer; default
/// [`DEFAULT_DEPTH_ANCHORING_FRAMES`]. Bounds the per-step decode + Depth-Anything-V2 memory.
pub const DEPTH_ANCHORING_FRAMES_KEY: &str = "depthAnchoringFrames";

/// The default [`DEPTH_ANCHORING_FRAMES_KEY`]: two latent frames per step.
pub const DEFAULT_DEPTH_ANCHORING_FRAMES: usize = 2;

/// The validated [`DEPTH_ANCHORING_FRAMES_KEY`] of `cfg` (default when absent).
pub(super) fn depth_anchoring_frames(cfg: &TrainingConfig) -> Result<usize> {
    match cfg.model_options.get(DEPTH_ANCHORING_FRAMES_KEY) {
        None => Ok(DEFAULT_DEPTH_ANCHORING_FRAMES),
        Some(value) => value
            .as_u64()
            .filter(|&n| n >= 1)
            .map(|n| n as usize)
            .ok_or_else(|| {
                mlx_gen::Error::Msg(format!(
                    "ltx_2_5 trainer: `{DEPTH_ANCHORING_FRAMES_KEY}` must be a positive integer, \
                     got {value}"
                ))
            }),
    }
}

/// The latent frames of a `(F, H, W)`-token clip the depth loss decodes (sc-24830): `k` frames
/// evenly spaced over the frames holding at least one generated (`loss_mask > 0`) target token —
/// conditioning frames would decode their clean latent and train nothing. Empty when no frame is
/// generated.
pub(super) fn select_depth_frames(
    loss_mask: &[f32],
    frames: usize,
    frame_tokens: usize,
    k: usize,
) -> Vec<i32> {
    let candidates: Vec<i32> = (0..frames)
        .filter(|&f| {
            loss_mask[f * frame_tokens..(f + 1) * frame_tokens]
                .iter()
                .any(|&m| m > 0.0)
        })
        .map(|f| f as i32)
        .collect();
    let n = candidates.len();
    if k >= n {
        return candidates;
    }
    if k == 1 {
        return vec![candidates[n / 2]];
    }
    let mut picked: Vec<i32> = (0..k)
        .map(|i| candidates[(i * (n - 1) + (k - 1) / 2) / (k - 1)])
        .collect();
    picked.dedup();
    picked
}

/// Capability-free request validation, factored out of [`Trainer::validate`] so it can be
/// unit-tested without a loaded trainer. Rejects an empty dataset, zero rank, LoKr (LoRA-only
/// trainer), and unsupported optimizers. The single-use / text-encoder-present check stays in
/// [`Trainer::validate`], which has the trainer state to inspect (F-055).
pub(super) fn validate_request(req: &TrainingRequest, label: &str) -> Result<()> {
    if req.items.is_empty() {
        return Err(format!("{label}: dataset is empty").into());
    }
    if req.config.rank == 0 {
        return Err(format!("{label}: rank must be > 0").into());
    }
    if req.config.network_type == NetworkType::Lokr {
        return Err(format!(
            "{label}: LoKr training is not supported — the reference LTX MLX \
                    trainer is LoRA-only. (LTX *inference* supports LoKr via sc-2393, but no \
                    LoKr trainer exists yet; that would be a separate extension.)"
        )
        .into());
    }
    if !TrainOptimizer::is_supported(&req.config.optimizer) {
        return Err(format!(
            "{label}: optimizer '{}' is not available on MLX training (supported: \
             adamw, adam, rose, prodigy)",
            req.config.optimizer
        )
        .into());
    }
    Ok(())
}

/// Weights-free (with respect to the multi-gigabyte base model) LTX-2.5 request preflight. Product
/// callers can run this before constructing a trainer: it resolves the complete workflow and Dev
/// validation contract, requires every prepared-pack path, and validates each safetensors header,
/// schema, modality shape and tensor-backed condition key. The same function is the request half of
/// [`LtxTrainer::validate`], so preflight and execution cannot drift.
pub fn validate_ltx25_training_request(req: &TrainingRequest) -> Result<()> {
    let descriptor = trainer_descriptor_25();
    refuse_ltx25_subject_mask(descriptor.id, req)?;
    refuse_ltx25_depth_anchoring(descriptor.id, req)?;
    // The shared floors keep their typed variant across the seam (`?` maps
    // `gen_core::Error::Unsupported` 1:1): a capability gap must stay `Unsupported` for the worker,
    // never be flattened to a message (sc-24161).
    gen_core::train::validate_control_request(&descriptor, req)?;
    gen_core::train::validate_full_finetune_request(&descriptor, req)?;
    gen_core::train::validate_training_techniques(&descriptor, req)?;
    gen_core::train::validate_edit_request(&descriptor, req)?;
    validate_request(req, "ltx_2_5 trainer")?;
    validate_ltx25_adapter_scale(req.config.alpha)?;
    let plan = Ltx25TrainingPlan::from_request(req)?;
    for item in &req.items {
        let path = prepared_bundle_path(item)?;
        if !path.is_file() {
            return Err(format!(
                "ltx_2_5 trainer: prepared bundle does not exist: {}",
                path.display()
            )
            .into());
        }
        let weights = Weights::from_file(path)?;
        validate_prepared_bundle_contract(&weights, path, &plan)?;
    }
    Ok(())
}

/// Every weights-free [`Trainer::validate`] floor — all of it but the single-use check.
pub(super) fn validate_floors(
    descriptor: &TrainerDescriptor,
    req: &TrainingRequest,
) -> gen_core::Result<()> {
    // Shared control-training floor (F-006 / F-055): the LTX trainer has no control branch, so a
    // request carrying `control_type` / per-item control images is rejected (typed `Unsupported`)
    // rather than silently training a plain LoRA and reporting success.
    gen_core::train::validate_control_request(descriptor, req)?;
    // Shared full-base-fine-tune floor (sc-14056): an adapter-only trainer must reject a
    // `full_finetune` request (typed `Unsupported`) rather than silently training a LoRA.
    gen_core::train::validate_full_finetune_request(descriptor, req)?;
    refuse_ltx25_subject_mask(descriptor.id, req)?;
    refuse_ltx25_depth_anchoring(descriptor.id, req)?;
    // Shared training-technique floor (epic 2123 E3): a technique this trainer does not
    // declare (e.g. `weight_noise_sigma > 0`) is a typed refusal, never silently ignored.
    gen_core::train::validate_training_techniques(descriptor, req)?;
    gen_core::train::validate_edit_request(descriptor, req)?;
    if descriptor.id == MODEL_25_ID {
        validate_ltx25_training_request(req)?;
    } else {
        validate_request(req, &format!("{} trainer", descriptor.id))?;
    }
    Ok(())
}
