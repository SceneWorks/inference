//! Shared MOSS sample geometry, absolute draw cursor and complete non-tensor recovery metadata.
//! Tensor/optimizer bundles remain backend-owned; this module does not allocate device memory.
pub mod bundle;
pub mod host;
mod inputs;

use crate::train::{
    aspect_ratio::{self, AspectRatioBucket, AspectSchedule},
    subject_mask::CropBox,
};
use crate::{BucketSchedule, Error, Result, TrainingRequest};
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};

pub const STREAM_VERSION: &str = "moss-stateless-draw-v1";
pub const STATE_VERSION: &str = "moss-complete-state-v2";

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct EntryGeometry {
    pub item: usize,
    pub bucket: usize,
    pub width: u32,
    pub height: u32,
    pub crop: [u32; 4],
}
impl EntryGeometry {
    pub fn crop_box(&self) -> CropBox {
        CropBox {
            x: self.crop[0],
            y: self.crop[1],
            width: self.crop[2],
            height: self.crop[3],
        }
    }
}
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct BatchDraw {
    pub epoch: u64,
    pub batch: u64,
    pub sample_cursor: u64,
    pub entries: Vec<usize>,
}
#[derive(Clone, Debug)]
pub struct DataPlan {
    pub entries: Vec<EntryGeometry>,
    pub identity: String,
    legacy: std::cell::OnceCell<BucketSchedule>,
    legacy_items: usize,
    legacy_buckets: Vec<crate::ResolutionBucket>,
    legacy_seed: u64,
    rectangular: Option<AspectSchedule>,
    batch_size: usize,
    inputs: Option<inputs::Inputs>,
}
impl DataPlan {
    pub fn from_request(req: &TrainingRequest) -> Result<Self> {
        Self::from_request_budget(req, host::available_bytes()?)
    }
    pub fn from_request_budget(req: &TrainingRequest, host_budget: u64) -> Result<Self> {
        let metadata = Self::metadata_bound(req)?;
        if metadata > host_budget {
            return Err(Error::Unsupported(
                "dataset metadata/schedule exceeds available host budget".into(),
            ));
        }
        let inputs = inputs::Inputs::capture(req, host_budget)?;
        let mut plan = Self::new(req, &inputs.dimensions())?;
        plan.inputs = Some(inputs);
        Ok(plan)
    }
    fn pinned(&self) -> Result<&inputs::Inputs> {
        self.inputs
            .as_ref()
            .ok_or_else(|| Error::Msg("dataset inputs have not been captured".into()))
    }
    pub fn verify_inputs(&self) -> Result<()> {
        self.pinned()?.verify()
    }
    pub fn decode_rgb(&self, item: usize, budget: u64) -> Result<image::RgbImage> {
        self.pinned()?.rgb(item, budget)
    }
    pub fn subject_mask(
        &self,
        req: &TrainingRequest,
        item: usize,
        budget: u64,
    ) -> Result<Option<crate::train::subject_mask::PreparedSubjectMask>> {
        self.pinned()?
            .subject_mask(req, item, req.config.subject_mask_loss.as_ref(), budget)
    }
    pub fn decode_buffer_bound(&self) -> Result<u64> {
        self.pinned()?.decode_buffer_bound()
    }
    pub fn dataset_fingerprint(&self, req: &TrainingRequest) -> Result<String> {
        self.pinned()?.fingerprint(req)
    }
    pub fn metadata_bound(req: &TrainingRequest) -> Result<u64> {
        let mut bytes = 0u64;
        for item in &req.items {
            if let Some(trigger) = req
                .config
                .trigger_word
                .as_ref()
                .filter(|s| !s.trim().is_empty())
            {
                if !item.caption.contains(trigger) {
                    bytes = bytes.checked_add(trigger.len() as u64).ok_or_else(|| {
                        Error::Unsupported("caption prefix metadata overflow".into())
                    })?;
                }
            }

            bytes = bytes
                .checked_add(item.caption.len() as u64)
                .ok_or_else(|| Error::Unsupported("dataset metadata overflow".into()))?;
            for path in
                std::iter::once(&item.image_path).chain(if req.config.subject_mask_loss.is_some() {
                    item.subject_mask_path.as_ref()
                } else {
                    None
                })
            {
                let path = std::path::absolute(path)?;
                for parent in path.ancestors() {
                    bytes = bytes
                        .checked_add(
                            (parent.as_os_str().len() as u64)
                                .checked_mul(4)
                                .and_then(|n| n.checked_add(512))
                                .ok_or_else(|| {
                                    Error::Unsupported("dataset pin metadata overflow".into())
                                })?,
                        )
                        .ok_or_else(|| {
                            Error::Unsupported("dataset pin metadata overflow".into())
                        })?;
                }
                bytes = bytes
                    .checked_add(8192)
                    .ok_or_else(|| Error::Unsupported("dataset pin metadata overflow".into()))?;
            }
        }
        let c = &req.config;
        let repeats = if !c.aspect_ratio_buckets.is_empty() {
            u64::from(
                c.aspect_ratio_buckets
                    .iter()
                    .map(|r| r.repeats)
                    .max()
                    .ok_or_else(|| Error::Unsupported("no rectangular rows".into()))?,
            )
        } else if c.training_buckets().len() > 1 {
            c.training_buckets().iter().try_fold(0u64, |n, row| {
                n.checked_add(u64::from(row.repeats))
                    .ok_or_else(|| Error::Unsupported("bucket repeat overflow".into()))
            })?
        } else {
            0
        };
        let schedules = (req.items.len() as u64)
            .checked_mul(repeats)
            .and_then(|n| n.checked_mul(128))
            .ok_or_else(|| Error::Unsupported("dataset schedule metadata overflow".into()))?;
        bytes
            .checked_add(schedules)
            .and_then(|n| n.checked_add((req.items.len() as u64).checked_mul(1024)?))
            .ok_or_else(|| Error::Unsupported("dataset metadata overflow".into()))
    }
    pub fn image_digest(&self, item: usize) -> Result<[u8; 32]> {
        self.pinned()?.image_digest(item)
    }
    pub fn mask_digest(&self, item: usize) -> Result<Option<[u8; 32]>> {
        self.pinned()?.mask_digest(item)
    }
    pub fn new(req: &TrainingRequest, dimensions: &[(u32, u32)]) -> Result<Self> {
        let c = &req.config;
        aspect_ratio::validate(c)?;
        if req.items.is_empty() || dimensions.len() != req.items.len() || c.batch_size == 0 {
            return Err(Error::Msg(
                "MOSS data plan needs matching nonempty items/geometries and positive batch size"
                    .into(),
            ));
        }
        let rectangular = if c.aspect_ratio_buckets.is_empty() {
            None
        } else {
            Some(AspectSchedule::new(dimensions, c, c.seed)?)
        };
        let mut entries = Vec::new();
        for (item, &(source_w, source_h)) in dimensions.iter().enumerate() {
            let rows: Vec<(usize, AspectRatioBucket)> = if let Some(schedule) = &rectangular {
                let b = schedule.assignments[item];
                vec![(b, c.aspect_ratio_buckets[b])]
            } else {
                c.training_buckets()
                    .iter()
                    .enumerate()
                    .map(|(b, r)| {
                        let edge = if r.resolution == 0 {
                            512
                        } else {
                            ((r.resolution / 32) * 32).max(32)
                        };
                        (
                            b,
                            AspectRatioBucket {
                                width: edge,
                                height: edge,
                                repeats: r.repeats,
                            },
                        )
                    })
                    .collect()
            };
            for (bucket, row) in rows {
                let crop = aspect_ratio::center_crop(source_w, source_h, row.width, row.height)?;
                entries.push(EntryGeometry {
                    item,
                    bucket,
                    width: row.width,
                    height: row.height,
                    crop: [crop.x, crop.y, crop.width, crop.height],
                });
            }
        }
        let policy = if rectangular.is_some() {
            aspect_ratio::RECTANGULAR_POLICY
        } else {
            "legacy-square-bucket-v1"
        };
        let identity = format!(
            "{:x}",
            Sha256::digest(
                serde_json::to_vec(&(policy, aspect_ratio::CROP_POLICY, STREAM_VERSION, &entries))
                    .map_err(|e| Error::Msg(e.to_string()))?
            )
        );
        Ok(Self {
            entries,
            identity,
            legacy: std::cell::OnceCell::new(),
            legacy_items: req.items.len(),
            legacy_buckets: c.training_buckets(),
            legacy_seed: c.seed,
            rectangular,
            batch_size: c.batch_size as usize,
            inputs: None,
        })
    }
    /// Zero-based micro-step to the exact sample draw cursor and homogeneous rectangular batch.
    pub fn batch(&self, micro_step: u32) -> Result<BatchDraw> {
        if let Some(schedule) = &self.rectangular {
            let n = schedule.epoch(0).len();
            if n == 0 {
                return Err(Error::Msg("MOSS rectangular schedule is empty".into()));
            }
            let epoch = u64::from(micro_step) / (n as u64);
            let batch = micro_step as usize % n;
            let batches = schedule.epoch(epoch);
            let draws = batches.iter().try_fold(0u64, |sum, b| {
                sum.checked_add(b.items.len() as u64)
                    .ok_or_else(|| Error::Msg("MOSS epoch draw overflow".into()))
            })?;
            let prior = batches[..batch]
                .iter()
                .map(|b| b.items.len() as u64)
                .sum::<u64>();
            let sample_cursor = epoch
                .checked_mul(draws)
                .and_then(|v| v.checked_add(prior))
                .ok_or_else(|| Error::Msg("MOSS absolute sample cursor overflow".into()))?;
            let selected = &batches[batch];
            // Rectangular entries are item-major, exactly one assigned entry per item.
            Ok(BatchDraw {
                epoch,
                batch: batch as u64,
                sample_cursor,
                entries: selected.items.clone(),
            })
        } else {
            let legacy = self.legacy.get_or_init(|| {
                BucketSchedule::new(self.legacy_items, &self.legacy_buckets, self.legacy_seed)
            });
            let first = (micro_step as usize)
                .checked_mul(self.batch_size)
                .ok_or_else(|| Error::Msg("MOSS sample cursor overflow".into()))?;
            let entries = (0..self.batch_size)
                .map(|i| {
                    first
                        .checked_add(i)
                        .map(|draw| legacy.cache_index(draw))
                        .ok_or_else(|| Error::Msg("MOSS sample cursor overflow".into()))
                })
                .collect::<Result<_>>()?;
            Ok(BatchDraw {
                epoch: (first / legacy.epoch_len()) as u64,
                batch: u64::from(micro_step),
                sample_cursor: first as u64,
                entries,
            })
        }
    }
    pub fn cached_token_count(&self) -> Result<u64> {
        self.entries.iter().try_fold(0u64, |sum, e| {
            sum.checked_add(u64::from(e.width / 16) * u64::from(e.height / 16))
                .ok_or_else(|| Error::Msg("MOSS cache token overflow".into()))
        })
    }
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct RunIdentity {
    pub digest: String,
    pub source_checkpoint: String,
    pub dataset_fingerprint: String,
    /// Complete canonical effective controls, including original horizon/LR/scheduler/optimizer.
    pub effective_config: String,
}

/// Captions actually encoded by training, including trigger injection and selected previews.
pub fn conditioning_prompts(req: &TrainingRequest) -> Vec<String> {
    let mut prompts = req
        .items
        .iter()
        .map(|item| {
            match req
                .config
                .trigger_word
                .as_deref()
                .filter(|t| !t.trim().is_empty())
            {
                Some(trigger) if !item.caption.contains(trigger) => {
                    format!("{trigger}, {}", item.caption)
                }
                _ => item.caption.clone(),
            }
        })
        .collect::<Vec<_>>();
    if req.config.sample_every > 0 && !req.config.sample_prompts.is_empty() {
        prompts.extend(req.config.sample_prompts.iter().take(4).cloned());
        prompts.push(String::new());
    }
    prompts
}

/// All progress needed in addition to raw factors, complete optimizer tensors and pending gradients.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct RunState {
    pub version: String,
    pub stream_version: String,
    pub effective_identity: RunIdentity,
    pub assignment_policy: String,
    pub crop_policy: String,
    pub geometry: Vec<EntryGeometry>,
    pub plan_identity: String,
    pub resolved_seed: u64,
    pub step: u32,
    pub update: u32,
    pub epoch: u64,
    pub batch: u64,
    pub sample_cursor: u64,
    pub accumulated_tokens: u64,
    pub final_loss: f32,
}
impl RunState {
    pub fn after_step(
        req: &TrainingRequest,
        plan: &DataPlan,
        identity: &RunIdentity,
        step: u32,
        update: u32,
        tokens: u64,
        loss: f32,
    ) -> Result<Self> {
        let next = plan.batch(step)?;
        if !loss.is_finite() {
            return Err(Error::Msg("MOSS checkpoint loss is nonfinite".into()));
        }
        Ok(Self {
            version: STATE_VERSION.into(),
            stream_version: STREAM_VERSION.into(),
            effective_identity: identity.clone(),
            assignment_policy: if req.config.aspect_ratio_buckets.is_empty() {
                "legacy-square-bucket-v1"
            } else {
                aspect_ratio::RECTANGULAR_POLICY
            }
            .into(),
            crop_policy: aspect_ratio::CROP_POLICY.into(),
            geometry: plan.entries.clone(),
            plan_identity: plan.identity.clone(),
            resolved_seed: req.config.seed,
            step,
            update,
            epoch: next.epoch,
            batch: next.batch,
            sample_cursor: next.sample_cursor,
            accumulated_tokens: tokens,
            final_loss: loss,
        })
    }
    pub fn validate(
        &self,
        req: &TrainingRequest,
        plan: &DataPlan,
        identity: &RunIdentity,
    ) -> Result<()> {
        if req.config.gradient_accumulation == 0 {
            return Err(Error::Unsupported(
                "MOSS zero accumulation is invalid".into(),
            ));
        }
        let expected = Self::after_step(
            req,
            plan,
            identity,
            self.step,
            self.update,
            self.accumulated_tokens,
            self.final_loss,
        )?;
        if self != &expected
            || self.step == 0
            || self.step >= req.config.steps
            || self.update != self.step / req.config.gradient_accumulation
        {
            return Err(Error::Unsupported(
                "MOSS complete resume state/version/seed/geometry/cursor/config differs".into(),
            ));
        }
        let boundary = self.step.is_multiple_of(req.config.gradient_accumulation);
        if boundary != (self.accumulated_tokens == 0) {
            return Err(Error::Unsupported(
                "MOSS pending gradient/token state differs from accumulation cursor".into(),
            ));
        }
        Ok(())
    }
}

/// Reject missing/mismatched requested recovery before eager tensor loading.
pub fn validate_resume_request(req: &TrainingRequest, source: &str) -> Result<()> {
    if !req.config.resume {
        return Ok(());
    }
    let plan = DataPlan::from_request(req)?;
    validate_resume_plan(req, source, &plan)
}
pub fn validate_resume_plan(req: &TrainingRequest, source: &str, plan: &DataPlan) -> Result<()> {
    if !req.config.resume {
        return Ok(());
    }
    let stem = req
        .file_name
        .strip_suffix(".safetensors")
        .ok_or_else(|| Error::Unsupported("MOSS output must use .safetensors".into()))?;
    let identity = effective_identity(req, source, plan)?;
    let path = bundle::latest(&req.output_dir, stem)?;
    bundle::read(&path)?.validate(req, plan, &identity)
}

/// Versioned effective identity includes original horizon, optimizer spelling, LR, all semantic
/// controls, exact source identity, dataset/caption contents and the actual assignment/crop plan.
/// Retry intent and a lower operational admission cap do not change the training algorithm.
/// Read a complete sealed state for a consumer resolving an originally unspecified seed. The
/// native trainer still validates the full request/source identity before any eager allocation.
pub fn resume_state_if_present(
    output_dir: &std::path::Path,
    file_name: &str,
) -> Result<Option<RunState>> {
    let stem = file_name
        .strip_suffix(".safetensors")
        .ok_or_else(|| Error::Unsupported("MOSS output must use .safetensors".into()))?;
    bundle::latest_optional(output_dir, stem)?
        .map(|path| bundle::read(&path))
        .transpose()
}
pub fn resume_state(output_dir: &std::path::Path, file_name: &str) -> Result<RunState> {
    resume_state_if_present(output_dir, file_name)?.ok_or_else(|| {
        Error::Unsupported("resume requested but no complete training state exists".into())
    })
}
/// Canonical locator only. Call resume_state_if_present first; a directory name is not proof.
pub fn resume_bundle_path(
    output_dir: &std::path::Path,
    file_name: &str,
    step: u32,
) -> Result<std::path::PathBuf> {
    let stem = file_name
        .strip_suffix(".safetensors")
        .ok_or_else(|| Error::Unsupported("MOSS output must use .safetensors".into()))?;
    bundle::path(output_dir, stem, step)
}
pub fn clear_resume_states(output_dir: &std::path::Path, file_name: &str) -> Result<usize> {
    let stem = file_name
        .strip_suffix(".safetensors")
        .ok_or_else(|| Error::Unsupported("MOSS output must use .safetensors".into()))?;
    bundle::clear(output_dir, stem)
}

fn canonical_controls(value: serde_json::Value) -> serde_json::Value {
    match value {
        serde_json::Value::Object(map) => serde_json::Value::Object(
            map.into_iter()
                .collect::<std::collections::BTreeMap<_, _>>()
                .into_iter()
                .map(|(k, v)| (k, canonical_controls(v)))
                .collect(),
        ),
        serde_json::Value::Array(values) => {
            serde_json::Value::Array(values.into_iter().map(canonical_controls).collect())
        }
        value => value,
    }
}

pub fn effective_identity(
    req: &TrainingRequest,
    source_identity: &str,
    plan: &DataPlan,
) -> Result<RunIdentity> {
    let c = &req.config;
    let scheduler = match c.lr_scheduler {
        crate::LrSchedule::Constant => "constant",
        crate::LrSchedule::Linear => "linear",
        crate::LrSchedule::Cosine => "cosine",
    };
    let dtype = match c.train_dtype.to_ascii_lowercase().as_str() {
        "bf16" | "bfloat16" => "bf16",
        "f32" | "fp32" | "float32" => "f32",
        _ => {
            return Err(Error::Unsupported(
                "unknown MOSS effective compute dtype".into(),
            ))
        }
    };
    let masks=c.subject_mask_loss.as_ref().map(|m|serde_json::json!({"background_f32_bits":m.background_weight.to_bits(),"subject_f32_bits":m.subject_weight.to_bits()}));
    let controls = serde_json::json!({
        "format":"moss-effective-controls-v2","rank":c.rank,"alpha_f32_bits":c.alpha.to_bits(),
        "learning_rate_f32_bits":c.learning_rate.to_bits(),"optimizer":c.optimizer,"weight_decay_f32_bits":c.weight_decay.to_bits(),
        "original_horizon_steps":c.steps,"batch_size":c.batch_size,"gradient_accumulation":c.gradient_accumulation,
        "gradient_checkpointing":c.gradient_checkpointing,"compute_dtype":dtype,"seed":c.seed,
        "lr_scheduler":scheduler,"lr_warmup_microsteps":c.lr_warmup_steps,
        "network_type":match c.network_type{crate::NetworkType::Lora=>"lora",crate::NetworkType::Lokr=>"lokr"},
        "decompose_factor":c.decompose_factor,"target_modules":c.lora_target_modules,
        "resolution":c.resolution,"square_buckets":c.resolution_buckets.iter().map(|b|serde_json::json!({"resolution":b.resolution,"repeats":b.repeats})).collect::<Vec<_>>(),
        "aspect_ratio_buckets":c.aspect_ratio_buckets,"cache_latents":c.cache_latents,"cache_text_embeddings":c.cache_text_embeddings,
        "timestep_type":c.timestep_type,"timestep_bias":c.timestep_bias,"loss_type":c.loss_type,"caption_trigger":c.trigger_word,
        "weight_noise_sigma_f32_bits":c.weight_noise_sigma.to_bits(),"gradient_noise_eta_f32_bits":c.gradient_noise_eta.to_bits(),"gradient_noise_gamma_f32_bits":c.gradient_noise_gamma.to_bits(),
        "subject_mask_loss":masks,"vae_posterior":super::POSTERIOR,
        "save_every":c.save_every,"sample_every":c.sample_every,"sample_prompts":c.sample_prompts,"sample_steps":c.sample_steps,"sample_guidance_f32_bits":c.sample_guidance_scale.to_bits()
    });
    let effective_config = serde_json::to_string(&canonical_controls(controls))
        .map_err(|e| Error::Msg(e.to_string()))?;
    let dataset_fingerprint = plan.dataset_fingerprint(req)?;
    let digest=format!("{:x}",Sha256::digest(format!("{STATE_VERSION}:{STREAM_VERSION}:{source_identity}:{dataset_fingerprint}:{}:{effective_config}",plan.identity)));
    Ok(RunIdentity {
        digest,
        source_checkpoint: source_identity.into(),
        dataset_fingerprint,
        effective_config,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn caller_cap_never_raises_physical_and_zero_refuses() {
        assert_eq!(
            crate::train::effective_memory_budget(1024, Some(4096)).unwrap(),
            1024
        );
        assert_eq!(
            crate::train::effective_memory_budget(1024, Some(173)).unwrap(),
            173
        );
        assert!(crate::train::effective_memory_budget(1024, Some(0)).is_err());
        assert!(crate::train::effective_memory_budget(0, None).is_err());
    }
    #[test]
    fn rectangular_cursor_counts_samples_not_padded_microbatch_slots() {
        let req = TrainingRequest {
            config: crate::TrainingConfig {
                aspect_ratio_buckets: vec![
                    AspectRatioBucket {
                        width: 64,
                        height: 32,
                        repeats: 1,
                    },
                    AspectRatioBucket {
                        width: 32,
                        height: 64,
                        repeats: 1,
                    },
                ],
                batch_size: 2,
                steps: 20,
                gradient_accumulation: 3,
                ..Default::default()
            },
            items: (0..4)
                .map(|i| crate::TrainingItem::captioned(format!("item{i}").into(), String::new()))
                .collect(),
            output_dir: "output".into(),
            file_name: "adapter.safetensors".into(),
            trigger_words: Vec::new(),
            cancel: crate::CancelFlag::new(),
        };
        let plan = DataPlan::new(&req, &[(80, 40), (80, 40), (80, 40), (40, 80)]).unwrap();
        let mut cursor = 0;
        for step in 0..12 {
            let batch = plan.batch(step).unwrap();
            assert_eq!(batch.sample_cursor, cursor);
            cursor += batch.entries.len() as u64;
            let geometry = &plan.entries[batch.entries[0]];
            assert!(batch
                .entries
                .iter()
                .all(|&i| plan.entries[i].bucket == geometry.bucket));
        }
        let identity = RunIdentity {
            digest: "identity".into(),
            source_checkpoint: "fixture-only".into(),
            dataset_fingerprint: "dataset".into(),
            effective_config: "controls".into(),
        };
        let state = RunState::after_step(&req, &plan, &identity, 4, 1, 123, 0.25).unwrap();
        state.validate(&req, &plan, &identity).unwrap();
        let mut wrong = state.clone();
        wrong.sample_cursor += 1;
        assert!(wrong.validate(&req, &plan, &identity).is_err());
        let mut wrong = state;
        wrong.accumulated_tokens = 0;
        assert!(wrong.validate(&req, &plan, &identity).is_err());
    }
    #[test]
    fn canonical_moss_identity_distinguishes_cache_controls_but_not_retry_cap() {
        let root = tempfile::tempdir().unwrap();
        let image = root.path().join("data.png");
        image::RgbImage::from_pixel(32, 32, image::Rgb([11, 27, 53]))
            .save(&image)
            .unwrap();
        let request = TrainingRequest {
            items: vec![crate::TrainingItem::captioned(image, "caption".into())],
            config: crate::TrainingConfig {
                steps: 6,
                batch_size: 1,
                gradient_accumulation: 2,
                resolution: 32,
                ..Default::default()
            },
            output_dir: root.path().into(),
            file_name: "adapter.safetensors".into(),
            trigger_words: Vec::new(),
            cancel: crate::CancelFlag::new(),
        };
        let plan = DataPlan::from_request_budget(&request, 2 * 1024 * 1024 * 1024).unwrap();
        let baseline = effective_identity(&request, "fixture-parent-a", &plan).unwrap();
        let legacy = crate::train::resume::training_config_fingerprint(&request.config);
        let mut retry = request.clone();
        retry.config.resume = true;
        retry.config.memory_limit_bytes = Some(173);
        assert_eq!(
            effective_identity(&retry, "fixture-parent-a", &plan).unwrap(),
            baseline
        );
        for mutation in 0..6 {
            let mut changed = request.clone();
            match mutation {
                0 => changed.config.cache_latents = false,
                1 => changed.config.cache_text_embeddings = false,
                2 => changed.config.optimizer = "adam".into(),
                3 => changed.config.learning_rate *= 2.,
                4 => changed.config.steps += 1,
                _ => changed.items[0].caption.push_str(" changed"),
            };
            assert_ne!(
                effective_identity(&changed, "fixture-parent-a", &plan).unwrap(),
                baseline,
                "semantic identity mutation {mutation}"
            );
            if mutation < 2 {
                assert_eq!(
                    crate::train::resume::training_config_fingerprint(&changed.config),
                    legacy,
                    "legacy explicit-field fingerprint must not acquire MOSS transport defaults"
                );
            }
        }
        assert_ne!(
            effective_identity(&request, "fixture-parent-b", &plan).unwrap(),
            baseline
        );
    }
    #[test]
    fn retry_prepared_path_relocation_preserves_content_identity_not_mutation() {
        let root = tempfile::tempdir().unwrap();
        let first = root.path().join("first.png");
        let second = root.path().join("retry.png");
        image::RgbImage::from_pixel(32, 32, image::Rgb([11, 27, 53]))
            .save(&first)
            .unwrap();
        std::fs::copy(&first, &second).unwrap();
        let request = TrainingRequest {
            items: vec![crate::TrainingItem::captioned(first, "caption".into())],
            config: crate::TrainingConfig {
                resolution: 32,
                ..Default::default()
            },
            output_dir: root.path().into(),
            file_name: "adapter.safetensors".into(),
            trigger_words: Vec::new(),
            cancel: crate::CancelFlag::new(),
        };
        let original = DataPlan::from_request_budget(&request, 2 * 1024 * 1024 * 1024).unwrap();
        let original = effective_identity(&request, "fixture-parent", &original).unwrap();
        let mut retry = request.clone();
        retry.items[0].image_path = second.clone();
        let plan = DataPlan::from_request_budget(&retry, 2 * 1024 * 1024 * 1024).unwrap();
        assert_eq!(
            effective_identity(&retry, "fixture-parent", &plan).unwrap(),
            original
        );
        image::RgbImage::from_pixel(32, 32, image::Rgb([11, 27, 54]))
            .save(&second)
            .unwrap();
        let changed = DataPlan::from_request_budget(&retry, 2 * 1024 * 1024 * 1024).unwrap();
        assert_ne!(
            effective_identity(&retry, "fixture-parent", &changed).unwrap(),
            original
        );
    }
}
