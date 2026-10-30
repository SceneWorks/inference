//! Shared request validation floors trainers apply before expensive work.

use super::*;

/// The validated stem of a training adapter output name (sc-27282): `file_name` must be a single
/// plain file name ending in `.safetensors` with a non-empty stem. Path separators (`/`, `\\`), drive
/// or stream colons, NUL and other control characters, `.`/`..`, and absolute or multi-component
/// names are refused with [`gen_core::Error::Unsupported`], so `output_dir.join(file_name)` (and every
/// `{stem}-step…` checkpoint name derived from it) can never leave [`TrainingRequest::output_dir`].
/// [`validate_training_techniques`] applies it, so every trainer that runs that floor gets it.
pub fn output_stem(file_name: &str) -> gen_core::Result<&str> {
    let invalid = |detail: &str| {
        gen_core::Error::Unsupported(format!("training adapter output {file_name:?}: {detail}"))
    };
    let path = std::path::Path::new(file_name);
    let mut components = path.components();
    if file_name.is_empty()
        || file_name.contains(['/', '\\', ':', '\0'])
        || file_name.chars().any(char::is_control)
        || !matches!(components.next(), Some(std::path::Component::Normal(_)))
        || components.next().is_some()
    {
        return Err(invalid("must be a single safe file name"));
    }
    file_name
        .strip_suffix(".safetensors")
        .filter(|s| !s.is_empty())
        .ok_or_else(|| invalid("must end in .safetensors"))
}

/// The shared **training-technique floor** (epic 2123 E3/E5) — every trainer's `validate` *and*
/// `train` entry point calls it before any expensive work, so a requested technique the trainer
/// does not implement is refused instead of silently ignored.
///
/// - `weight_noise_sigma` / `gradient_noise_eta` / `gradient_noise_gamma` not finite or negative
///   ⇒ [`gen_core::Error::Msg`] (malformed request).
/// - `weight_noise_sigma > 0` on a trainer whose [`TrainerDescriptor::techniques`] lacks
///   [`weight_noise`](TrainingTechniques::weight_noise) ⇒ typed [`gen_core::Error::Unsupported`];
///   likewise `gradient_noise_eta > 0` without
///   [`gradient_noise`](TrainingTechniques::gradient_noise).
/// - either noise technique with [`TrainingConfig::full_finetune`] ⇒ typed
///   [`gen_core::Error::Unsupported`]: both perturb adapter factors / adapter gradients only and must
///   never touch base weights (E5).
/// - any auxiliary loss ([`enabled_aux_losses`]) with [`TrainingConfig::full_finetune`] ⇒ typed
///   [`gen_core::Error::Unsupported`]: the aux losses train the adapter surface only.
/// - `resolution_buckets` non-empty but malformed (a zero resolution or repeat count, a resolution
///   off the [`RESOLUTION_BUCKET_STRIDE`], a duplicate resolution, more than
///   [`MAX_RESOLUTION_BUCKETS`]) ⇒ [`gen_core::Error::Msg`]; well formed on a
///   trainer that lacks [`resolution_buckets`](TrainingTechniques::resolution_buckets) ⇒ typed
///   [`gen_core::Error::Unsupported`].
/// - a malformed [`TrainingConfig::depth_anchoring`] schedule ⇒ [`gen_core::Error::Msg`].
/// - depth anchoring enabled on a trainer whose descriptor lacks
///   [`depth_anchoring`](TrainingTechniques::depth_anchoring) ⇒ typed [`gen_core::Error::Unsupported`].
/// - depth anchoring enabled without a [`DepthAnchoringConfig::model_dir`] or a
///   [`TrainingConfig::perceptual_decoder_dir`] ⇒ [`gen_core::Error::Msg`] naming the missing model.
/// - `subject_mask_loss` set with malformed weights ⇒ [`gen_core::Error::Msg`] naming the field.
/// - `subject_mask_loss` set on a trainer whose [`TrainerDescriptor::techniques`] lacks
///   [`subject_mask_loss`](TrainingTechniques::subject_mask_loss) ⇒ typed
///   [`gen_core::Error::Unsupported`].
/// - `subject_mask_loss` set while any item has no
///   [`subject_mask_path`](TrainingItem::subject_mask_path) ⇒ [`gen_core::Error::Msg`] naming the
///   images that lack one (defence in depth behind the product-layer preflight).
/// - a malformed identity / face-landmark schedule, or `identity_loss.min_cos` outside `[-1, 1]`
///   ⇒ [`gen_core::Error::Msg`]; either loss enabled on a trainer that lacks
///   [`identity_loss`](TrainingTechniques::identity_loss) /
///   [`face_landmark_loss`](TrainingTechniques::face_landmark_loss) ⇒ typed
///   [`gen_core::Error::Unsupported`]; enabled without [`TrainingConfig::face_analysis_dir`], the
///   x0 decoder, or (landmarks) [`FaceLandmarkLossConfig::model_dir`] ⇒ [`gen_core::Error::Msg`].
/// - a malformed [`TrainingConfig::vae_anchor`] / [`TrainingConfig::latent_lpips`] schedule ⇒
///   [`gen_core::Error::Msg`]; either loss enabled on a trainer lacking
///   [`vae_anchor_loss`](TrainingTechniques::vae_anchor_loss) /
///   [`latent_lpips_loss`](TrainingTechniques::latent_lpips_loss) ⇒ typed
///   [`gen_core::Error::Unsupported`]; enabled without its model directory (or, for the VAE anchor,
///   without [`TrainingConfig::perceptual_decoder_dir`]) ⇒ [`gen_core::Error::Msg`].
/// - [`TrainingConfig::training_adapter`] set with [`TrainingConfig::full_finetune`], or on a
///   trainer whose descriptor lacks [`training_adapter`](TrainingTechniques::training_adapter) ⇒
///   typed [`gen_core::Error::Unsupported`].
/// - an unsafe or non-`.safetensors` [`TrainingRequest::file_name`] ⇒ typed
///   [`gen_core::Error::Unsupported`] (see [`output_stem`]; sc-27282).
/// - every technique off ⇒ no-op.
pub fn validate_training_techniques(
    desc: &TrainerDescriptor,
    req: &TrainingRequest,
) -> gen_core::Result<()> {
    output_stem(&req.file_name)?;
    let cfg = &req.config;
    validate_memory_limit_fraction(cfg.memory_limit_fraction)?;
    if cfg.memory_limit_fraction.is_some() && desc.family != "moss_image" {
        return Err(gen_core::Error::Unsupported(format!(
            "{}: training memory fraction is not routed by this trainer",
            desc.id
        )));
    }
    if cfg.memory_limit_bytes == Some(0) {
        return Err(gen_core::Error::Unsupported(
            "training memory cap must be positive; None means no caller cap".into(),
        ));
    }
    if cfg.memory_limit_bytes.is_some() && desc.family != "moss_image" {
        return Err(gen_core::Error::Unsupported(format!(
            "{}: training memory byte cap is not routed by this trainer",
            desc.id
        )));
    }
    for (name, value) in [
        ("weight_noise_sigma", cfg.weight_noise_sigma),
        ("gradient_noise_eta", cfg.gradient_noise_eta),
        ("gradient_noise_gamma", cfg.gradient_noise_gamma),
    ] {
        if !value.is_finite() || value < 0.0 {
            return Err(gen_core::Error::Msg(format!(
                "{}: {name} must be a finite value >= 0, got {value}",
                desc.id
            )));
        }
    }
    let sigma = cfg.weight_noise_sigma;
    if sigma > 0.0 {
        if cfg.full_finetune {
            return Err(gen_core::Error::Unsupported(format!(
                "{}: weight noising (weight_noise_sigma {sigma}) perturbs adapter factors only and \
                 cannot be combined with a full base fine-tune",
                desc.id
            )));
        }
        if !desc.techniques.weight_noise {
            return Err(gen_core::Error::Unsupported(format!(
                "{}: weight noising (weight_noise_sigma {sigma}) is not supported by this trainer",
                desc.id
            )));
        }
    }
    let eta = cfg.gradient_noise_eta;
    if eta > 0.0 {
        if cfg.full_finetune {
            return Err(gen_core::Error::Unsupported(format!(
                "{}: gradient noise (gradient_noise_eta {eta}) perturbs adapter gradients only and \
                 cannot be combined with a full base fine-tune",
                desc.id
            )));
        }
        if !desc.techniques.gradient_noise {
            return Err(gen_core::Error::Unsupported(format!(
                "{}: gradient noise (gradient_noise_eta {eta}) is not supported by this trainer",
                desc.id
            )));
        }
    }
    let aux = enabled_aux_losses(cfg);
    if cfg.full_finetune && !aux.is_empty() {
        return Err(gen_core::Error::Unsupported(format!(
            "{}: the auxiliary losses ({}) train a LoRA/LoKr adapter only and cannot be combined \
             with a full base fine-tune",
            desc.id,
            aux.join(", ")
        )));
    }
    let depth = &req.config.depth_anchoring;
    depth
        .schedule
        .validate("depth anchoring")
        .map_err(|m| gen_core::Error::Msg(format!("{}: {m}", desc.id)))?;
    if depth.schedule.is_enabled() {
        if !desc.techniques.depth_anchoring {
            return Err(gen_core::Error::Unsupported(format!(
                "{}: depth anchoring (weight {}) is not supported by this trainer",
                desc.id, depth.schedule.weight
            )));
        }
        if depth.model_dir.is_none() {
            return Err(gen_core::Error::Msg(format!(
                "{}: depth anchoring needs the Depth-Anything-V2 {} checkpoint \
                 (depth_anchoring.model_dir is unset)",
                desc.id,
                depth.model_size.as_str()
            )));
        }
        if lacks_x0_decoder(desc, &req.config) {
            return Err(gen_core::Error::Msg(format!(
                "{}: depth anchoring needs the family's small x0 decoder \
                 (perceptual_decoder_dir is unset)",
                desc.id
            )));
        }
    }
    validate_face_losses(desc, &req.config)?;
    validate_latent_perceptual_losses(desc, req)?;
    if let Some(mask_loss) = &req.config.subject_mask_loss {
        mask_loss.validate(desc.id)?;
        if !desc.techniques.subject_mask_loss {
            return Err(gen_core::Error::Unsupported(format!(
                "{}: subject-masked loss weighting is not supported by this trainer",
                desc.id
            )));
        }
        subject_mask::require_subject_masks(desc.id, &req.items)?;
    }
    validate_body_losses(desc, req)?;
    validate_resolution_buckets(desc, &cfg.resolution_buckets)?;
    aspect_ratio::validate(cfg)?;
    if !cfg.aspect_ratio_buckets.is_empty()
        && (!desc.techniques.resolution_buckets || desc.family != "moss_image")
    {
        return Err(gen_core::Error::Unsupported(format!(
            "{}: rectangular aspect-assigned training is not supported by this trainer",
            desc.id
        )));
    }
    if let Some(adapter) = &cfg.training_adapter {
        if cfg.full_finetune {
            return Err(gen_core::Error::Unsupported(format!(
                "{}: a training adapter ({}) is applied to the frozen base of a LoRA/LoKr run and \
                 cannot be combined with a full base fine-tune",
                desc.id,
                adapter.display()
            )));
        }
        if !desc.techniques.training_adapter {
            return Err(gen_core::Error::Unsupported(format!(
                "{}: a training adapter ({}) is not supported by this trainer",
                desc.id,
                adapter.display()
            )));
        }
    }
    Ok(())
}

/// Body-loss half of [`validate_training_techniques`] (sc-24832): a malformed schedule/knob is a
/// `Msg`; an enabled loss the trainer does not declare is `Unsupported`; an enabled loss without
/// its checkpoint (or the x0 decoder, or — normal restricted to the subject — every item's mask)
/// is a `Msg` naming what is missing.
fn validate_body_losses(desc: &TrainerDescriptor, req: &TrainingRequest) -> gen_core::Result<()> {
    let body = &req.config.body_losses;
    body.validate(desc.id).map_err(gen_core::Error::Msg)?;
    let losses = [
        (
            "body proportion loss",
            &body.proportion,
            desc.techniques.body_proportion_loss,
        ),
        (
            "body shape loss",
            &body.shape,
            desc.techniques.body_shape_loss,
        ),
        ("normal loss", &body.normal, desc.techniques.normal_loss),
    ];
    for (name, schedule, declared) in losses {
        if schedule.is_enabled() && !declared {
            return Err(gen_core::Error::Unsupported(format!(
                "{}: the {name} (weight {}) is not supported by this trainer",
                desc.id, schedule.weight
            )));
        }
    }
    if !body.any_enabled() {
        return Ok(());
    }
    let missing = |what: &str| {
        Err(gen_core::Error::Msg(format!(
            "{}: the body losses need {what}",
            desc.id
        )))
    };
    if body.pose_model_dir.is_none() {
        return missing(
            "the ViTPose+ base checkpoint — the proportion encoder and every body loss's person \
             detector (body_losses.pose_model_dir is unset)",
        );
    }
    if body.shape.is_enabled() && body.shape_model_dir.is_none() {
        return missing("the HybrIK checkpoint (body_losses.shape_model_dir is unset)");
    }
    if body.normal.is_enabled() && body.normal_model_dir.is_none() {
        return missing("the Sapiens normal checkpoint (body_losses.normal_model_dir is unset)");
    }
    if lacks_x0_decoder(desc, &req.config) {
        return missing("the family's small x0 decoder (perceptual_decoder_dir is unset)");
    }
    if body.normal.is_enabled() && body.normal_restrict_to_subject {
        let lacking: Vec<String> = req
            .items
            .iter()
            .filter(|i| i.subject_mask_path.is_none())
            .map(|i| {
                i.image_path
                    .file_name()
                    .map(|n| n.to_string_lossy().into_owned())
                    .unwrap_or_else(|| i.image_path.display().to_string())
            })
            .collect();
        if !lacking.is_empty() {
            return Err(gen_core::Error::Msg(format!(
                "{}: the normal loss restricted to the subject needs a subject mask for every \
                 dataset image; {} of {} have none: {}",
                desc.id,
                lacking.len(),
                req.items.len(),
                subject_mask::name_list(&lacking)
            )));
        }
    }
    Ok(())
}

/// Face-loss half of [`validate_training_techniques`] (sc-24831): malformed schedules / `min_cos`
/// ⇒ `Msg`; an enabled loss the trainer does not declare ⇒ `Unsupported`; an enabled loss without
/// its models (the face-analysis stack, the FaceMesh checkpoint, the x0 decoder) ⇒ `Msg` naming the
/// missing one.
fn validate_face_losses(desc: &TrainerDescriptor, cfg: &TrainingConfig) -> gen_core::Result<()> {
    let id = &cfg.identity_loss;
    let lm = &cfg.face_landmark_loss;
    let msg = |m: String| gen_core::Error::Msg(format!("{}: {m}", desc.id));
    id.schedule.validate("identity loss").map_err(msg)?;
    lm.schedule.validate("face-landmark loss").map_err(msg)?;
    let (lo, hi) = IdentityLossConfig::MIN_COS_RANGE;
    if !id.min_cos.is_finite() || id.min_cos < lo || id.min_cos > hi {
        return Err(gen_core::Error::Msg(format!(
            "{}: identity_loss.min_cos must be in [{lo}, {hi}], got {}",
            desc.id, id.min_cos
        )));
    }
    let checks = [
        (
            id.schedule.is_enabled(),
            desc.techniques.identity_loss,
            "the ArcFace identity loss",
            id.schedule.weight,
        ),
        (
            lm.schedule.is_enabled(),
            desc.techniques.face_landmark_loss,
            "the face-landmark loss",
            lm.schedule.weight,
        ),
    ];
    for (enabled, declared, name, weight) in checks {
        if !enabled {
            continue;
        }
        if !declared {
            return Err(gen_core::Error::Unsupported(format!(
                "{}: {name} (weight {weight}) is not supported by this trainer",
                desc.id
            )));
        }
        if cfg.face_analysis_dir.is_none() {
            return Err(gen_core::Error::Msg(format!(
                "{}: {name} needs the SCRFD + ArcFace face-analysis stack \
                 (face_analysis_dir is unset)",
                desc.id
            )));
        }
        if lacks_x0_decoder(desc, cfg) {
            return Err(gen_core::Error::Msg(format!(
                "{}: {name} needs the family's small x0 decoder (perceptual_decoder_dir is unset)",
                desc.id
            )));
        }
    }
    if lm.schedule.is_enabled() && lm.model_dir.is_none() {
        return Err(gen_core::Error::Msg(format!(
            "{}: the face-landmark loss needs the MediaPipe FaceMesh-v2 checkpoint \
             (face_landmark_loss.model_dir is unset)",
            desc.id
        )));
    }
    Ok(())
}

/// Latent-perceptual half of [`validate_training_techniques`] (sc-24833): each loss's schedule must
/// be well formed (`Msg`), and when enabled the trainer must declare it (`Unsupported`) and its
/// frozen model (plus, for the decoded-x0 VAE anchor, the family's small decoder) must be named
/// (`Msg`).
fn validate_latent_perceptual_losses(
    desc: &TrainerDescriptor,
    req: &TrainingRequest,
) -> gen_core::Result<()> {
    let cfg = &req.config;
    let va = &cfg.vae_anchor;
    va.schedule
        .validate("VAE anchor")
        .map_err(|m| gen_core::Error::Msg(format!("{}: {m}", desc.id)))?;
    if va.schedule.is_enabled() {
        if !desc.techniques.vae_anchor_loss {
            return Err(gen_core::Error::Unsupported(format!(
                "{}: the VAE anchor loss (weight {}) is not supported by this trainer",
                desc.id, va.schedule.weight
            )));
        }
        if va.model_dir.is_none() {
            return Err(gen_core::Error::Msg(format!(
                "{}: the VAE anchor loss needs the FLUX.2 VAE (vae_anchor.model_dir is unset)",
                desc.id
            )));
        }
        if lacks_x0_decoder(desc, cfg) {
            return Err(gen_core::Error::Msg(format!(
                "{}: the VAE anchor loss needs the family's small x0 decoder \
                 (perceptual_decoder_dir is unset)",
                desc.id
            )));
        }
    }
    let lp = &cfg.latent_lpips;
    lp.schedule
        .validate("E-LatentLPIPS")
        .map_err(|m| gen_core::Error::Msg(format!("{}: {m}", desc.id)))?;
    if lp.schedule.is_enabled() {
        if !desc.techniques.latent_lpips_loss {
            return Err(gen_core::Error::Unsupported(format!(
                "{}: the E-LatentLPIPS loss (weight {}) is not supported by this trainer (no \
                 E-LatentLPIPS weights match its latent family)",
                desc.id, lp.schedule.weight
            )));
        }
        if lp.model_dir.is_none() {
            return Err(gen_core::Error::Msg(format!(
                "{}: the E-LatentLPIPS loss needs its weights (latent_lpips.model_dir is unset)",
                desc.id
            )));
        }
    }
    Ok(())
}

/// Resolution-bucket half of [`validate_training_techniques`] (sc-2127): an empty list is off; a
/// non-empty one must be well formed (`Msg`) and declared by the trainer (`Unsupported`).
fn validate_resolution_buckets(
    desc: &TrainerDescriptor,
    buckets: &[ResolutionBucket],
) -> gen_core::Result<()> {
    if buckets.is_empty() {
        return Ok(());
    }
    if buckets.len() > MAX_RESOLUTION_BUCKETS {
        return Err(gen_core::Error::Msg(format!(
            "{}: resolution_buckets has {} buckets; at most {MAX_RESOLUTION_BUCKETS} are allowed",
            desc.id,
            buckets.len()
        )));
    }
    for (i, b) in buckets.iter().enumerate() {
        if b.resolution == 0 || b.repeats == 0 {
            return Err(gen_core::Error::Msg(format!(
                "{}: resolution_buckets[{i}] needs a resolution and a repeat count >= 1, got \
                 resolution {} repeats {}",
                desc.id, b.resolution, b.repeats
            )));
        }
        if b.resolution % RESOLUTION_BUCKET_STRIDE != 0 {
            return Err(gen_core::Error::Msg(format!(
                "{}: resolution_buckets[{i}] resolution {} is not a multiple of \
                 {RESOLUTION_BUCKET_STRIDE} (the trainers' latent stride)",
                desc.id, b.resolution
            )));
        }
        if buckets[..i].iter().any(|p| p.resolution == b.resolution) {
            return Err(gen_core::Error::Msg(format!(
                "{}: resolution_buckets lists resolution {} twice",
                desc.id, b.resolution
            )));
        }
    }
    if !desc.techniques.resolution_buckets {
        return Err(gen_core::Error::Unsupported(format!(
            "{}: multi-resolution buckets (resolution_buckets) are not supported by this trainer",
            desc.id
        )));
    }
    Ok(())
}

/// The shared control-training validation floor (F-006) — the training analog of
/// [`Capabilities::validate_request`](gen_core::generator::Capabilities::validate_request). A family
/// trainer's `validate` calls this so the "control_type set ⇒ every item carries a control image"
/// invariant (and the "unsupported ⇒ typed reject" rule) lives in one place instead of being
/// re-implemented per family — the recurring fixes-don't-travel gap sc-10163 opened.
///
/// - `control_type` unset ⇒ the plain LoRA/LoKr path: no-op.
/// - `control_type` set on a trainer that does **not** advertise
///   [`supports_control`](TrainerDescriptor::supports_control) ⇒ typed
///   [`gen_core::Error::Unsupported`] (the
///   LoRA-only trainers must reject a control request, not silently produce a non-control adapter).
/// - `control_type` set on a control-capable trainer ⇒ every [`TrainingItem`] must carry a
///   `control_image_path`, else [`gen_core::Error::Msg`].
pub fn validate_control_request(
    desc: &TrainerDescriptor,
    req: &TrainingRequest,
) -> gen_core::Result<()> {
    let Some(control_type) = req.config.control_type.as_deref() else {
        return Ok(());
    };
    if !desc.supports_control {
        return Err(gen_core::Error::Unsupported(format!(
            "{}: control-branch training (control_type {control_type:?}) is not supported by this \
             trainer",
            desc.id
        )));
    }
    if let Some(idx) = req
        .items
        .iter()
        .position(|it| it.control_image_path.is_none())
    {
        return Err(gen_core::Error::Msg(format!(
            "{}: control_type {control_type:?} requires a control image on every training item \
             (item {idx} has none)",
            desc.id
        )));
    }
    Ok(())
}

/// The shared **full-base-fine-tune** validation floor (F-006, sc-14056) — the exact analog of
/// [`validate_control_request`] for the [`TrainingConfig::full_finetune`] selector. A family
/// trainer's `validate` calls it so the "unsupported ⇒ typed reject" rule lives in one place instead
/// of each adapter-only trainer independently deciding to ignore the flag.
///
/// - `full_finetune == false` ⇒ the plain LoRA/LoKr path: no-op.
/// - `full_finetune == true` on a trainer that does **not** advertise
///   [`supports_full_finetune`](TrainerDescriptor::supports_full_finetune) ⇒ typed
///   [`gen_core::Error::Unsupported`]. Silently training an adapter instead would hand the caller a
///   ~100 MB LoRA where they asked for a fine-tuned base checkpoint, and the consumer's memory gate,
///   output registration, and UI would all describe the wrong artifact — the F-055 silent-adapter
///   class, which this repo already refuses for `control_type`.
/// - `full_finetune == true` on a full-tune-capable trainer ⇒ ok.
pub fn validate_full_finetune_request(
    desc: &TrainerDescriptor,
    req: &TrainingRequest,
) -> gen_core::Result<()> {
    if req.config.full_finetune && !desc.supports_full_finetune {
        return Err(gen_core::Error::Unsupported(format!(
            "{}: full base fine-tune (full_finetune) is not supported by this trainer — it can only \
             train a LoRA/LoKr adapter",
            desc.id
        )));
    }
    Ok(())
}

/// The shared **instruction-edit dataset** floor (sc-24161) — the analog of
/// [`validate_control_request`] for [`TrainingItem::reference_image_paths`]. Every family trainer's
/// `validate` calls it, so "a trainer that cannot use references refuses an edit dataset" and the
/// per-model reference cap live in one place.
///
/// - no item carries references ⇒ the plain captioned / control path: no-op.
/// - any item carries references on a trainer whose
///   [`max_reference_images`](TrainerDescriptor::max_reference_images) is `0` ⇒ typed
///   [`gen_core::Error::Unsupported`] — never a text-to-image adapter silently trained on the targets.
/// - an edit-capable trainer, but some item has **no** references (a mixed dataset) ⇒
///   [`gen_core::Error::Msg`] naming the item: an edit run conditions every step on its references, so
///   a reference-less item has no defined training input.
/// - an item also carries a `control_image_path` ⇒ [`gen_core::Error::Msg`]: edit pairs and control
///   pairs are different workflows.
/// - an item carries more references than the cap ⇒ [`gen_core::Error::Msg`] naming the item, its
///   count and the cap.
/// - an edit pair with an empty (whitespace-only) instruction `caption` ⇒ [`gen_core::Error::Msg`]: the
///   instruction is the edit's whole conditioning.
///
/// It also carries the **item-shape floor** every trainer gets for free (and the reason
/// [`TrainingItem`] can derive `Default` safely): any item whose `image_path` — or any of whose
/// `reference_image_paths` — is empty is refused with [`gen_core::Error::Msg`] naming the item, so a
/// `..Default::default()` literal that forgot a path fails `validate` instead of the run.
pub fn validate_edit_request(
    desc: &TrainerDescriptor,
    req: &TrainingRequest,
) -> gen_core::Result<()> {
    for (idx, item) in req.items.iter().enumerate() {
        if item.image_path.as_os_str().is_empty() {
            return Err(gen_core::Error::Msg(format!(
                "{}: item {idx} has an empty image_path",
                desc.id
            )));
        }
        if let Some(r) = item
            .reference_image_paths
            .iter()
            .position(|p| p.as_os_str().is_empty())
        {
            return Err(gen_core::Error::Msg(format!(
                "{}: item {idx} has an empty path at reference {r}",
                desc.id
            )));
        }
    }
    let Some(first_edit) = req.items.iter().position(TrainingItem::is_edit_pair) else {
        return Ok(());
    };
    let cap = desc.max_reference_images as usize;
    if cap == 0 {
        return Err(gen_core::Error::Unsupported(format!(
            "{}: instruction-edit training (item {first_edit} carries ordered reference images) is \
             not supported by this trainer — it trains on captioned images only, and would \
             otherwise silently learn text-to-image from the edit targets",
            desc.id
        )));
    }
    for (idx, item) in req.items.iter().enumerate() {
        let count = item.reference_image_paths.len();
        if count == 0 {
            return Err(gen_core::Error::Msg(format!(
                "{}: an instruction-edit dataset needs at least one reference image on every item \
                 (item {idx} has none); split captioned and edit items into separate runs",
                desc.id
            )));
        }
        if item.control_image_path.is_some() {
            return Err(gen_core::Error::Msg(format!(
                "{}: item {idx} carries both reference images (edit pair) and a control image \
                 (control pair); an item is one or the other",
                desc.id
            )));
        }
        if count > cap {
            return Err(gen_core::Error::Msg(format!(
                "{}: item {idx} carries {count} reference images, but this model accepts at most \
                 {cap} reference images per edit",
                desc.id
            )));
        }
        if item.caption.trim().is_empty() {
            return Err(gen_core::Error::Msg(format!(
                "{}: item {idx} is an edit pair with an empty instruction caption",
                desc.id
            )));
        }
    }
    Ok(())
}

/// `model_options` keys naming a workflow an image LoRA/LoKr trainer does not read from
/// `model_options`: ordered reference images (instruction-edit training travels as
/// `TrainingItem::reference_image_paths`, under the shared edit floor) and control conditioning.
/// Every other key — the worker's `mixedPrecision`, `cacheLatents`, `networkType`, `sampleEvery`, … —
/// is not the trainer's to read and is ignored.
pub const REFERENCE_CONTROL_MODEL_OPTIONS: [&str; 8] = [
    "references",
    "referenceImages",
    "reference_images",
    "referenceImagePaths",
    "controlType",
    "control_type",
    "controlImage",
    "control_image",
];

/// Whether a [`REFERENCE_CONTROL_MODEL_OPTIONS`] value actually selects a workflow. The worker's
/// `advanced` map carries these keys in their *off* state too — `null`, `[]`, `""` (or whitespace),
/// `{}`, `false`, `"none"` — and those select nothing.
pub fn model_option_selects_something(value: &JsonValue) -> bool {
    match value {
        JsonValue::Null | JsonValue::Bool(false) => false,
        JsonValue::String(s) => {
            let s = s.trim();
            !s.is_empty() && !s.eq_ignore_ascii_case("none")
        }
        JsonValue::Array(items) => !items.is_empty(),
        JsonValue::Object(map) => !map.is_empty(),
        JsonValue::Bool(true) | JsonValue::Number(_) => true,
    }
}

/// The first [`REFERENCE_CONTROL_MODEL_OPTIONS`] key that selects something, on the config's
/// `model_options` first and then on each item's.
pub fn selected_reference_control_model_option(req: &TrainingRequest) -> Option<&'static str> {
    let refused = |options: &JsonMap<String, JsonValue>| {
        REFERENCE_CONTROL_MODEL_OPTIONS
            .iter()
            .find(|key| {
                options
                    .get(**key)
                    .is_some_and(model_option_selects_something)
            })
            .copied()
    };
    refused(&req.config.model_options).or_else(|| {
        req.items
            .iter()
            .find_map(|item| refused(&item.model_options))
    })
}

/// Refuse (as [`gen_core::Error::Msg`], `label`-prefixed) a request whose `model_options` select
/// reference / control conditioning ([`selected_reference_control_model_option`]) rather than
/// silently training without it.
pub fn refuse_reference_control_model_options(
    label: &str,
    req: &TrainingRequest,
) -> gen_core::Result<()> {
    match selected_reference_control_model_option(req) {
        None => Ok(()),
        Some(key) => Err(gen_core::Error::Msg(format!(
            "{label}: model_options `{key}` selects reference/control conditioning, which this \
             trainer does not read from model_options (instruction-edit references travel as \
             TrainingItem::reference_image_paths); refusing rather than silently training \
             without it"
        ))),
    }
}

/// The shared **load-overlay floor** for image LoRA/LoKr trainers (sc-24163): a control,
/// extra-control, IP-adapter or identity overlay on the trainer's
/// [`LoadSpec`](gen_core::runtime::LoadSpec) is not part of the trained adapter, so it is a typed
/// [`gen_core::Error::Unsupported`] rather than silently dropped. `label` names the trainer.
pub fn refuse_trainer_load_overlays(
    label: &str,
    spec: &gen_core::runtime::LoadSpec,
) -> gen_core::Result<()> {
    if spec.control.is_some()
        || !spec.extra_controls.is_empty()
        || spec.ip_adapter.is_some()
        || spec.identity.is_some()
    {
        return Err(gen_core::Error::Unsupported(format!(
            "{label}: control / IP-adapter / identity overlays are not part of text-to-image \
             LoRA/LoKr training"
        )));
    }
    Ok(())
}
