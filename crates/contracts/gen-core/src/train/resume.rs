//! Backend-neutral **resume identity** shared by the candle and MLX trainers (sc-24163), so the two
//! backends refuse the same resumes with the same words instead of drifting apart.
//!
//! * [`request_fingerprint`] — a stable digest of every input that selects cached training data
//!   (resolution, item order, captions, image / control / ordered edit-reference paths **and file
//!   contents**). A resume bundle records it, and a resume whose dataset changed is refused rather
//!   than silently continuing a run on different data.
//! * [`training_config_fingerprint`] — the training-config knobs a resume must not change (steps,
//!   accumulation, schedule, rank / alpha, seed, resolution, loss, dtype, checkpointing, timestep
//!   sampling).
//! * [`check_resume_fingerprints`] — the comparison both backends run against a bundle's metadata.
//!
//! The digest format is byte-identical to the one candle-gen's resume bundles have always recorded
//! (`candle-training-request-v1`), so existing candle bundles keep resuming.

use std::collections::HashMap;
use std::io::Read;
use std::path::Path;

use sha2::{Digest, Sha256};

use super::{TrainingConfig, TrainingRequest};

/// The resume-bundle metadata key holding [`training_config_fingerprint`].
pub const TRAINING_CONFIG_KEY: &str = "training_config";
/// The resume-bundle metadata key holding [`request_fingerprint`].
pub const REQUEST_FINGERPRINT_KEY: &str = "request_fingerprint";

fn field(hasher: &mut Sha256, tag: &[u8], bytes: &[u8]) {
    hasher.update((tag.len() as u64).to_le_bytes());
    hasher.update(tag);
    hasher.update((bytes.len() as u64).to_le_bytes());
    hasher.update(bytes);
}

fn cancelled(req: &TrainingRequest) -> crate::Result<()> {
    if req.cancel.is_cancelled() {
        Err(crate::Error::Canceled)
    } else {
        Ok(())
    }
}

fn file(hasher: &mut Sha256, tag: &[u8], path: &Path, req: &TrainingRequest) -> crate::Result<()> {
    cancelled(req)?;
    field(hasher, tag, path.to_string_lossy().as_bytes());
    let io = |what: &str, e: std::io::Error| {
        crate::Error::Msg(format!(
            "training resume fingerprint: {what} {}: {e}",
            path.display()
        ))
    };
    let mut handle = std::fs::File::open(path).map_err(|e| io("open", e))?;
    let size = handle.metadata().map_err(|e| io("stat", e))?.len();
    hasher.update(size.to_le_bytes());
    let mut buffer = [0u8; 64 * 1024];
    loop {
        cancelled(req)?;
        let n = handle.read(&mut buffer).map_err(|e| io("read", e))?;
        if n == 0 {
            break;
        }
        hasher.update(&buffer[..n]);
    }
    Ok(())
}

/// Stable digest of every input that selects cached training data. Item order, captions, paths, file
/// contents, optional control inputs, ordered edit references and resolution are length-delimited to
/// prevent ambiguity. A tripped `req.cancel` is a typed [`crate::Error::Canceled`].
pub fn request_fingerprint(req: &TrainingRequest) -> crate::Result<String> {
    cancelled(req)?;
    let mut hasher = Sha256::new();
    field(&mut hasher, b"format", b"candle-training-request-v1");
    field(
        &mut hasher,
        b"resolution",
        &req.config.resolution.to_le_bytes(),
    );
    field(
        &mut hasher,
        b"item_count",
        &(req.items.len() as u64).to_le_bytes(),
    );
    for (index, item) in req.items.iter().enumerate() {
        cancelled(req)?;
        field(&mut hasher, b"item_index", &(index as u64).to_le_bytes());
        field(&mut hasher, b"caption", item.caption.as_bytes());
        file(&mut hasher, b"image", &item.image_path, req)?;
        match &item.control_image_path {
            Some(path) => {
                field(&mut hasher, b"has_control", &[1]);
                file(&mut hasher, b"control", path, req)?;
            }
            None => field(&mut hasher, b"has_control", &[0]),
        }
        // Instruction-edit references (sc-24161), in order. Hashed only when present, so every
        // captioned / control request keeps the fingerprint (and the resume bundles) it had.
        if !item.reference_image_paths.is_empty() {
            field(
                &mut hasher,
                b"reference_count",
                &(item.reference_image_paths.len() as u64).to_le_bytes(),
            );
            for reference in &item.reference_image_paths {
                file(&mut hasher, b"reference", reference, req)?;
            }
        }
        // Subject mask (sc-24828), hashed only when present — the worker sets it only for a
        // masked-loss run — so every unmasked request keeps its fingerprint (and resume bundles).
        if let Some(mask) = &item.subject_mask_path {
            file(&mut hasher, b"subject_mask", mask, req)?;
        }
    }
    Ok(format!("{:x}", hasher.finalize()))
}

/// The training-config knobs a resume must continue unchanged, as one comparable string.
///
/// Adapter noise (epic 2123, sc-24826/sc-24827) changes the trained trajectory, so a non-zero
/// `weight_noise_sigma` appends `;weight_noise=<sigma>` and a non-zero `gradient_noise_eta` appends
/// `;gradient_noise=<eta>/<gamma>` (gamma only matters while eta is on). With both off nothing is
/// appended, so every pre-noise resume bundle keeps the fingerprint it was written with.
///
/// Resolution buckets (sc-2127) change the cache layout and the sample order, so a non-empty
/// bucket list is appended (`;buckets=<res>x<repeats>,…`); an empty list appends nothing, so every
/// pre-bucket resume bundle keeps the fingerprint it was written with.
///
/// Subject-masked loss (sc-24828) changes the objective, so it appends
/// `;subject_mask_loss=<background>,<subject>` when on and nothing when off.
///
/// The identity / face-landmark losses (sc-24831) change the objective, so each appends its
/// schedule (and the identity loss its gate + reference mode) when on and nothing when off.
/// Each enabled body loss (sc-24832) appends its schedule and knob
/// (`;body_proportion=<w>/<t_min>-<t_max>/<every_n>,head=<bool>`, `;body_shape=…,min_cos=<c>`,
/// `;normal=…,subject=<bool>`); a disabled one appends nothing.
pub fn training_config_fingerprint(cfg: &TrainingConfig) -> String {
    let mut fingerprint = format!(
        "steps={};accum={};scheduler={:?};warmup={};rank={};alpha={};seed={};resolution={};loss={};dtype={};\
         checkpoint={};timestep_type={};timestep_bias={}",
        cfg.steps,
        cfg.gradient_accumulation.max(1),
        cfg.lr_scheduler,
        cfg.lr_warmup_steps,
        cfg.rank,
        cfg.alpha,
        cfg.seed,
        cfg.resolution,
        cfg.loss_type,
        cfg.train_dtype,
        cfg.gradient_checkpointing,
        cfg.timestep_type,
        cfg.timestep_bias
    );
    if cfg.weight_noise_sigma != 0.0 {
        fingerprint.push_str(&format!(";weight_noise={:?}", cfg.weight_noise_sigma));
    }
    if cfg.gradient_noise_eta != 0.0 {
        fingerprint.push_str(&format!(
            ";gradient_noise={:?}/{:?}",
            cfg.gradient_noise_eta, cfg.gradient_noise_gamma
        ));
    }
    if !cfg.resolution_buckets.is_empty() {
        let buckets: Vec<String> = cfg
            .resolution_buckets
            .iter()
            .map(|b| format!("{}x{}", b.resolution, b.repeats))
            .collect();
        fingerprint.push_str(&format!(";buckets={}", buckets.join(",")));
    }
    if let Some(m) = &cfg.subject_mask_loss {
        fingerprint.push_str(&format!(
            ";subject_mask_loss={:?},{:?}",
            m.background_weight, m.subject_weight
        ));
    }
    let body = &cfg.body_losses;
    let sched = |s: &crate::train::AuxLossSchedule| {
        format!("{:?}/{:?}-{:?}/{}", s.weight, s.t_min, s.t_max, s.every_n)
    };
    if body.proportion.is_enabled() {
        fingerprint.push_str(&format!(
            ";body_proportion={},head={}",
            sched(&body.proportion),
            body.include_head
        ));
    }
    if body.shape.is_enabled() {
        fingerprint.push_str(&format!(
            ";body_shape={},min_cos={:?}",
            sched(&body.shape),
            body.shape_min_cos
        ));
    }
    if body.normal.is_enabled() {
        fingerprint.push_str(&format!(
            ";normal={},subject={}",
            sched(&body.normal),
            body.normal_restrict_to_subject
        ));
    }
    let id = &cfg.identity_loss;
    if id.schedule.is_enabled() {
        let s = id.schedule;
        fingerprint.push_str(&format!(
            ";identity_loss={:?}/{:?}-{:?}/{}/{:?}/{}",
            s.weight,
            s.t_min,
            s.t_max,
            s.every_n,
            id.min_cos,
            id.reference_mode.as_str()
        ));
    }
    let lm = cfg.face_landmark_loss.schedule;
    if lm.is_enabled() {
        fingerprint.push_str(&format!(
            ";face_landmark_loss={:?}/{:?}-{:?}/{}",
            lm.weight, lm.t_min, lm.t_max, lm.every_n
        ));
    }
    fingerprint
}

/// Refuse a resume bundle (by its safetensors `meta`) whose recorded training config or dataset
/// fingerprint differs from this run's — or that records neither, so its provenance is unknown.
pub fn check_resume_fingerprints(
    meta: &HashMap<String, String>,
    cfg: &TrainingConfig,
    request_fingerprint: &str,
) -> crate::Result<()> {
    let saved_config = meta.get(TRAINING_CONFIG_KEY).ok_or_else(|| {
        crate::Error::Msg(format!("resume: missing {TRAINING_CONFIG_KEY} metadata"))
    })?;
    let requested_config = training_config_fingerprint(cfg);
    if saved_config != &requested_config {
        return Err(crate::Error::Msg(format!(
            "resume: training configuration differs (saved {saved_config:?}, requested \
             {requested_config:?})"
        )));
    }
    let saved_request = meta.get(REQUEST_FINGERPRINT_KEY).ok_or_else(|| {
        crate::Error::Msg(format!(
            "resume: missing {REQUEST_FINGERPRINT_KEY} metadata"
        ))
    })?;
    if saved_request != request_fingerprint {
        return Err(crate::Error::Msg(format!(
            "resume: dataset/request fingerprint differs (saved {saved_request}, requested \
             {request_fingerprint})"
        )));
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::path::PathBuf;

    use crate::train::TrainingItem;

    fn request(dir: &Path) -> TrainingRequest {
        let image = dir.join("a.png");
        std::fs::write(&image, b"image a").unwrap();
        TrainingRequest {
            items: vec![TrainingItem::captioned(image, "a cat".into())],
            config: TrainingConfig::default(),
            output_dir: dir.to_path_buf(),
            file_name: "out.safetensors".into(),
            trigger_words: vec![],
            cancel: Default::default(),
        }
    }

    /// The digest is the exact bytes candle-gen's resume bundles have always recorded: a change here
    /// strands every existing candle resume snapshot. The hex strings are **captured, not derived**:
    /// they are what the pre-sc-24163 `candle_gen::train::flow_match::request_fingerprint`
    /// (feature-branch base `d6f6388ad`) returned for these exact requests over the committed
    /// `tests/fixtures/request_fingerprint/` files, addressed by the same relative paths (cargo runs
    /// a package's tests from its manifest directory, so the path bytes are machine-independent).
    ///
    /// *Mutation that reds this:* any change to a field tag byte, the field order, the length
    /// framing, the file-size prefix or the format string.
    #[test]
    fn the_request_fingerprint_format_is_pinned() {
        let fixture =
            |name: &str| PathBuf::from(format!("tests/fixtures/request_fingerprint/{name}"));
        let req = |items: Vec<TrainingItem>| TrainingRequest {
            items,
            config: TrainingConfig {
                resolution: 512,
                ..Default::default()
            },
            output_dir: PathBuf::from("out"),
            file_name: "out.safetensors".into(),
            trigger_words: vec![],
            cancel: Default::default(),
        };
        let captioned = req(vec![TrainingItem::captioned(
            fixture("target.bin"),
            "a cat".into(),
        )]);
        let control = req(vec![TrainingItem::with_control(
            fixture("target.bin"),
            "a cat".into(),
            fixture("control.bin"),
        )]);
        let edit = req(vec![TrainingItem::edit_pair(
            fixture("target.bin"),
            "make it blue".into(),
            vec![fixture("ref_a.bin"), fixture("ref_b.bin")],
        )]);
        for (name, req, pinned) in [
            (
                "captioned",
                captioned,
                "d1b0e0c1313cd521a3bff8698c6ac823c5f021f7b6dc5e2233894bc1ad39a8f1",
            ),
            (
                "control",
                control,
                "ac7457a124e2ed138e301c7821117f320b8e08bda98b931e2d298a944ad0f401",
            ),
            (
                "edit",
                edit,
                "9a7112d8f77cb0ebd266d8b87fef582858b914f92dbb2184b539df1c52b6c4b7",
            ),
        ] {
            assert_eq!(
                request_fingerprint(&req).unwrap(),
                pinned,
                "{name}: the digest must stay the one candle resume bundles already record"
            );
        }
    }

    /// Ordered references are part of the identity: adding, reordering or editing one changes it.
    #[test]
    fn ordered_references_change_the_fingerprint() {
        let dir = tempfile::tempdir().unwrap();
        let mut req = request(dir.path());
        let plain = request_fingerprint(&req).unwrap();
        let (ra, rb) = (dir.path().join("ra.png"), dir.path().join("rb.png"));
        std::fs::write(&ra, b"ref a").unwrap();
        std::fs::write(&rb, b"ref b").unwrap();
        req.items[0].reference_image_paths = vec![ra.clone(), rb.clone()];
        let edit = request_fingerprint(&req).unwrap();
        assert_ne!(plain, edit);
        req.items[0].reference_image_paths = vec![rb, ra.clone()];
        assert_ne!(edit, request_fingerprint(&req).unwrap());
        req.items[0].reference_image_paths.reverse();
        assert_eq!(edit, request_fingerprint(&req).unwrap());
        std::fs::write(&ra, b"ref a, edited").unwrap();
        assert_ne!(edit, request_fingerprint(&req).unwrap());
    }

    /// sc-24828: a subject mask (and its contents) is part of the dataset identity, and the
    /// masked-loss weights part of the config identity — both only when present, so an unmasked
    /// request/config keeps the digest `the_request_fingerprint_format_is_pinned` pins.
    #[test]
    fn subject_masks_and_mask_loss_change_the_fingerprints() {
        let dir = tempfile::tempdir().unwrap();
        let mut req = request(dir.path());
        let plain = request_fingerprint(&req).unwrap();
        let mask = dir.path().join("mask.png");
        std::fs::write(&mask, b"mask a").unwrap();
        req.items[0].subject_mask_path = Some(mask.clone());
        let masked = request_fingerprint(&req).unwrap();
        assert_ne!(plain, masked);
        std::fs::write(&mask, b"mask a, repainted").unwrap();
        assert_ne!(masked, request_fingerprint(&req).unwrap());

        let off = TrainingConfig::default();
        let mut on = off.clone();
        on.subject_mask_loss = Some(crate::train::SubjectMaskLoss {
            background_weight: 0.0,
            subject_weight: 1.0,
        });
        let base = training_config_fingerprint(&off);
        assert!(!base.contains("subject_mask"), "{base}");
        let with = training_config_fingerprint(&on);
        assert_eq!(with, format!("{base};subject_mask_loss=0.0,1.0"));
    }

    /// sc-24832: each enabled body loss joins the config fingerprint; all off leaves it unchanged.
    /// Mutation: drop the `;normal=` append ⇒ the normal-weight change is invisible ⇒ red.
    #[test]
    fn body_losses_join_the_config_fingerprint_only_when_on() {
        let base_cfg = TrainingConfig::default();
        let base = training_config_fingerprint(&base_cfg);
        assert!(
            !base.contains("body") && !base.contains("normal="),
            "{base}"
        );
        let mut on = base_cfg.clone();
        on.body_losses.proportion.weight = 0.1;
        let p = training_config_fingerprint(&on);
        assert_eq!(
            p,
            format!("{base};body_proportion=0.1/0.0-1.0/2,head=false")
        );
        on.body_losses.shape.weight = 0.2;
        on.body_losses.normal.weight = 0.3;
        let all = training_config_fingerprint(&on);
        assert!(
            all.ends_with(
                ";body_shape=0.2/0.0-1.0/2,min_cos=0.2;normal=0.3/0.0-1.0/2,subject=false"
            ),
            "{all}"
        );
        let mut moved = on.clone();
        moved.body_losses.normal.weight = 0.4;
        assert_ne!(training_config_fingerprint(&moved), all);
    }

    /// sc-24831: the face losses join the config fingerprint only when on (an off config keeps its
    /// pre-sc-24831 fingerprint byte for byte), and a changed gate refuses the resume.
    /// Mutation: drop the identity append ⇒ `with` == `base` ⇒ red.
    #[test]
    fn face_losses_change_the_config_fingerprint_only_when_on() {
        let base_cfg = TrainingConfig::default();
        let base = training_config_fingerprint(&base_cfg);
        assert!(
            !base.contains("identity") && !base.contains("landmark"),
            "{base}"
        );
        let mut on = base_cfg.clone();
        on.identity_loss.schedule.weight = 0.1;
        let with = training_config_fingerprint(&on);
        assert_eq!(
            with,
            format!("{base};identity_loss=0.1/0.0-1.0/2/0.2/dataset_average")
        );
        on.face_landmark_loss.schedule.weight = 0.05;
        assert!(training_config_fingerprint(&on).ends_with(";face_landmark_loss=0.05/0.0-1.0/2"));
        let mut gate = on.clone();
        gate.identity_loss.min_cos = 0.3;
        assert_ne!(
            training_config_fingerprint(&gate),
            training_config_fingerprint(&on)
        );
    }

    #[test]
    fn a_cancelled_fingerprint_is_typed() {
        let dir = tempfile::tempdir().unwrap();
        let req = request(dir.path());
        req.cancel.cancel();
        assert!(matches!(
            request_fingerprint(&req),
            Err(crate::Error::Canceled)
        ));
    }

    #[test]
    fn resume_fingerprints_refuse_a_changed_config_dataset_or_unknown_bundle() {
        let cfg = TrainingConfig::default();
        let meta = HashMap::from([
            (
                TRAINING_CONFIG_KEY.to_string(),
                training_config_fingerprint(&cfg),
            ),
            (REQUEST_FINGERPRINT_KEY.to_string(), "fp".to_string()),
        ]);
        check_resume_fingerprints(&meta, &cfg, "fp").unwrap();
        let err = check_resume_fingerprints(&meta, &cfg, "other")
            .unwrap_err()
            .to_string();
        assert!(err.contains("dataset/request fingerprint differs"), "{err}");
        let changed = TrainingConfig {
            rank: cfg.rank + 1,
            ..cfg.clone()
        };
        let err = check_resume_fingerprints(&meta, &changed, "fp")
            .unwrap_err()
            .to_string();
        assert!(err.contains("training configuration differs"), "{err}");
        let err = check_resume_fingerprints(&HashMap::new(), &cfg, "fp")
            .unwrap_err()
            .to_string();
        assert!(err.contains("missing training_config"), "{err}");
    }

    /// sc-24827: the adapter-noise knobs are part of the resume config fingerprint — changing any
    /// of them refuses the resume — while a knobs-off config keeps the exact pre-noise string, so
    /// existing resume bundles still match. The expected string is the literal pre-change format
    /// for `TrainingConfig::default()`.
    ///
    /// *Mutations that red this:* dropping either append; appending unconditionally (the knobs-off
    /// string changes); omitting gamma from the gradient-noise append.
    #[test]
    fn adapter_noise_knobs_join_the_config_fingerprint_without_moving_the_off_value() {
        let off = TrainingConfig::default();
        let pre_change = format!(
            "steps={};accum={};scheduler={:?};warmup={};rank={};alpha={};seed={};resolution={};\
             loss={};dtype={};checkpoint={};timestep_type={};timestep_bias={}",
            off.steps,
            off.gradient_accumulation.max(1),
            off.lr_scheduler,
            off.lr_warmup_steps,
            off.rank,
            off.alpha,
            off.seed,
            off.resolution,
            off.loss_type,
            off.train_dtype,
            off.gradient_checkpointing,
            off.timestep_type,
            off.timestep_bias
        );
        assert_eq!(training_config_fingerprint(&off), pre_change);
        // Gamma alone (eta off) trains identically, so it must not strand a bundle either.
        let gamma_only = TrainingConfig {
            gradient_noise_gamma: 0.9,
            ..off.clone()
        };
        assert_eq!(training_config_fingerprint(&gamma_only), pre_change);

        let weight = TrainingConfig {
            weight_noise_sigma: 0.0125,
            ..off.clone()
        };
        let weight2 = TrainingConfig {
            weight_noise_sigma: 0.02,
            ..off.clone()
        };
        let grad = TrainingConfig {
            gradient_noise_eta: 0.01,
            ..off.clone()
        };
        let grad_eta2 = TrainingConfig {
            gradient_noise_eta: 0.02,
            ..off.clone()
        };
        let grad_gamma2 = TrainingConfig {
            gradient_noise_gamma: 0.9,
            ..grad.clone()
        };
        let fps: Vec<String> = [&off, &weight, &weight2, &grad, &grad_eta2, &grad_gamma2]
            .iter()
            .map(|c| training_config_fingerprint(c))
            .collect();
        for i in 0..fps.len() {
            for j in (i + 1)..fps.len() {
                assert_ne!(
                    fps[i], fps[j],
                    "configs {i} and {j} must fingerprint differently"
                );
            }
        }
        // ...and the resume check refuses a changed knob.
        let meta = HashMap::from([
            (TRAINING_CONFIG_KEY.to_owned(), fps[1].clone()),
            (REQUEST_FINGERPRINT_KEY.to_owned(), "fp".to_owned()),
        ]);
        check_resume_fingerprints(&meta, &weight, "fp").unwrap();
        assert!(check_resume_fingerprints(&meta, &off, "fp").is_err());
    }

    /// sc-2127: buckets are part of the resume config identity — a bundle written with one bucket
    /// list refuses a resume under another (or under none) — while a bucket-free config keeps its
    /// pre-bucket fingerprint byte for byte.
    #[test]
    fn resolution_buckets_are_part_of_the_resume_config_fingerprint() {
        use crate::train::ResolutionBucket;
        let plain = TrainingConfig::default();
        assert!(!training_config_fingerprint(&plain).contains("buckets"));
        let bucketed = TrainingConfig {
            resolution_buckets: vec![
                ResolutionBucket {
                    resolution: 512,
                    repeats: 16,
                },
                ResolutionBucket {
                    resolution: 1024,
                    repeats: 1,
                },
            ],
            ..plain.clone()
        };
        assert!(training_config_fingerprint(&bucketed).ends_with(";buckets=512x16,1024x1"));
        let meta = HashMap::from([
            (
                TRAINING_CONFIG_KEY.to_string(),
                training_config_fingerprint(&bucketed),
            ),
            (REQUEST_FINGERPRINT_KEY.to_string(), "fp".to_string()),
        ]);
        check_resume_fingerprints(&meta, &bucketed, "fp").unwrap();
        let mut remixed = bucketed.clone();
        remixed.resolution_buckets[0].repeats = 4;
        for other in [plain, remixed] {
            let err = check_resume_fingerprints(&meta, &other, "fp")
                .unwrap_err()
                .to_string();
            assert!(err.contains("training configuration differs"), "{err}");
        }
    }
}
