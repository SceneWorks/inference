//! Shared MOSS sample geometry, absolute draw cursor and complete non-tensor recovery metadata.
//! Tensor/optimizer bundles remain backend-owned; this module does not allocate device memory.
pub mod bundle;
pub mod host;
mod inputs;

use gen_core::train::aspect_ratio::AspectRatioBucket;
use gen_core::{train::BucketSchedule, train::TrainingRequest};
use gen_core::{Error, Result};
use gen_core_train::{
    aspect_ratio::{self, AspectSchedule},
    canvas::training_canvas,
    subject_mask::CropBox,
};
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};

pub const STREAM_VERSION: &str = "moss-stateless-draw-v1";
pub const STATE_VERSION: &str = "moss-complete-state-v2";
/// Assignment policy of the default (no `aspect_ratio_buckets`) path: each item at each square
/// bucket edge trains on the shared aspect-preserving canvas
/// ([`gen_core_train::canvas::training_canvas`]) at stride `CANVAS_STRIDE`, and micro-batches
/// larger than one are homogeneous in canvas size. It replaced `legacy-square-bucket-v1` (centre
/// square at `edge²`), so a resume state written under the square policy is refused by the plan
/// identity rather than resumed on different geometry.
pub const LEGACY_CANVAS_POLICY: &str = "aspect-canvas-bucket-v1";
/// Canvas side multiple for the default path: the same multiple-of-32 rule rectangular rows obey,
/// which also keeps every side a multiple of the 16-pixel packed-token grid.
pub const CANVAS_STRIDE: u32 = 32;
/// Batch-list shuffle domain of the default path's homogeneous micro-batches (distinct from the
/// rectangular schedule's per-bucket and batch-list constants).
const LEGACY_BATCH_SEED: u64 = 0x4341_4e56_4241_0001;

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
    /// Homogeneous micro-batches per epoch of the default path (`batch_size > 1` only); constant
    /// because every epoch permutes the same `(item, bucket)` multiset.
    legacy_batches_per_epoch: std::cell::OnceCell<usize>,
    /// The most recently built default-path epoch `(epoch, batches)`.
    legacy_epoch: std::cell::RefCell<Option<(u64, Vec<Vec<usize>>)>>,
    legacy_items: usize,
    legacy_buckets: Vec<gen_core::train::ResolutionBucket>,
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
    ) -> Result<Option<gen_core_train::subject_mask::PreparedSubjectMask>> {
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
            if let Some(schedule) = &rectangular {
                let bucket = schedule.assignments[item];
                let row: AspectRatioBucket = c.aspect_ratio_buckets[bucket];
                let crop = aspect_ratio::center_crop(source_w, source_h, row.width, row.height)?;
                entries.push(EntryGeometry {
                    item,
                    bucket,
                    width: row.width,
                    height: row.height,
                    crop: [crop.x, crop.y, crop.width, crop.height],
                });
                continue;
            }
            if source_w == 0 || source_h == 0 {
                return Err(Error::Msg(format!(
                    "MOSS dataset item {item} has a zero-pixel side ({source_w}x{source_h})"
                )));
            }
            // Default path: each square bucket edge is an area budget; the item keeps its own
            // aspect on a canvas of at most edge² pixels, both sides multiples of 32.
            for (bucket, row) in c.training_buckets().iter().enumerate() {
                let edge = if row.resolution == 0 {
                    512
                } else {
                    ((row.resolution / 32) * 32).max(32)
                };
                let canvas = training_canvas((source_w, source_h), edge, CANVAS_STRIDE, None);
                entries.push(EntryGeometry {
                    item,
                    bucket,
                    width: canvas.width,
                    height: canvas.height,
                    crop: [
                        canvas.crop.x,
                        canvas.crop.y,
                        canvas.crop.width,
                        canvas.crop.height,
                    ],
                });
            }
        }
        let policy = Self::policy(c);
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
            legacy_batches_per_epoch: std::cell::OnceCell::new(),
            legacy_epoch: std::cell::RefCell::new(None),
            legacy_items: req.items.len(),
            legacy_buckets: c.training_buckets(),
            legacy_seed: c.seed,
            rectangular,
            batch_size: c.batch_size as usize,
            inputs: None,
        })
    }
    /// The assignment policy a config's plan is built under (part of the plan identity).
    pub fn policy(config: &gen_core::train::TrainingConfig) -> &'static str {
        if config.aspect_ratio_buckets.is_empty() {
            LEGACY_CANVAS_POLICY
        } else {
            aspect_ratio::RECTANGULAR_POLICY
        }
    }
    fn legacy_schedule(&self) -> &BucketSchedule {
        self.legacy.get_or_init(|| {
            BucketSchedule::new(self.legacy_items, &self.legacy_buckets, self.legacy_seed)
        })
    }
    /// Default-path epoch `epoch` as homogeneous micro-batches: that epoch's ordered draws
    /// grouped by canvas `(width, height)` (draw order kept within a group, groups in order of
    /// first draw), each group chunked by `batch_size` keeping its tail, then the batch list
    /// shuffled with the seed and epoch.
    fn build_legacy_epoch(&self, epoch: u64) -> Result<Vec<Vec<usize>>> {
        let legacy = self.legacy_schedule();
        let len = legacy.epoch_len();
        let first = usize::try_from(epoch)
            .ok()
            .and_then(|e| e.checked_mul(len))
            .filter(|f| f.checked_add(len).is_some())
            .ok_or_else(|| Error::Msg("MOSS sample cursor overflow".into()))?;
        let mut slots = std::collections::HashMap::<(u32, u32), usize>::new();
        let mut groups: Vec<Vec<usize>> = Vec::new();
        for k in 0..len {
            let entry = legacy.cache_index(first + k);
            let geometry = self
                .entries
                .get(entry)
                .ok_or_else(|| Error::Msg("MOSS schedule entry outside the plan".into()))?;
            let slot = *slots
                .entry((geometry.width, geometry.height))
                .or_insert_with(|| {
                    groups.push(Vec::new());
                    groups.len() - 1
                });
            groups[slot].push(entry);
        }
        let mut batches: Vec<Vec<usize>> = groups
            .iter()
            .flat_map(|group| group.chunks(self.batch_size).map(<[usize]>::to_vec))
            .collect();
        BucketSchedule::shuffle_items(&mut batches, self.legacy_seed ^ LEGACY_BATCH_SEED, epoch, 0);
        Ok(batches)
    }
    /// Zero-based micro-step to the exact sample draw cursor and a micro-batch whose entries
    /// share one canvas size (the rectangular bucket, or the default path's canvas).
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
        } else if self.batch_size > 1 {
            let len = self.legacy_schedule().epoch_len();
            if len == 0 {
                return Err(Error::Msg("MOSS schedule is empty".into()));
            }
            let per_epoch = match self.legacy_batches_per_epoch.get() {
                Some(&n) => n,
                None => {
                    let n = self.build_legacy_epoch(0)?.len();
                    *self.legacy_batches_per_epoch.get_or_init(|| n)
                }
            };
            if per_epoch == 0 {
                return Err(Error::Msg("MOSS schedule is empty".into()));
            }
            let epoch = u64::from(micro_step) / (per_epoch as u64);
            let batch = micro_step as usize % per_epoch;
            let mut cached = self.legacy_epoch.borrow_mut();
            if cached.as_ref().is_none_or(|(e, _)| *e != epoch) {
                *cached = Some((epoch, self.build_legacy_epoch(epoch)?));
            }
            let batches = &cached.as_ref().expect("epoch built above").1;
            let prior = batches[..batch].iter().map(|b| b.len() as u64).sum::<u64>();
            // Every epoch draws each scheduled (item, bucket) exactly once: `len` samples.
            let sample_cursor = epoch
                .checked_mul(len as u64)
                .and_then(|v| v.checked_add(prior))
                .ok_or_else(|| Error::Msg("MOSS absolute sample cursor overflow".into()))?;
            Ok(BatchDraw {
                epoch,
                batch: batch as u64,
                sample_cursor,
                entries: batches[batch].clone(),
            })
        } else {
            let legacy = self.legacy_schedule();
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

/// Conservative simultaneous host workspace for prompt rendering/tokenizer normalization,
/// pretokenizer offsets, full token encodings and retained configuration copies. Admission
/// happens from borrowed byte lengths BEFORE constructing any rendered strings or tokenizer.
/// The model's 4096-token limit is not a tokenizer-allocation limit.
pub fn conditioning_preprocess_bytes(req: &TrainingRequest) -> Result<u64> {
    let trigger = req
        .config
        .trigger_word
        .as_deref()
        .filter(|t| !t.trim().is_empty());
    let mut largest = 0u64;
    for item in &req.items {
        let prefix = trigger
            .filter(|t| !item.caption.contains(t))
            .map_or(0, |t| t.len().saturating_add(2));
        largest = largest.max(
            (item.caption.len() as u64)
                .checked_add(prefix as u64)
                .ok_or_else(|| Error::Unsupported("MOSS prompt size overflow".into()))?,
        );
    }
    let mut config_copies = 0u64;
    for prompt in &req.config.sample_prompts {
        config_copies = config_copies
            .checked_add(prompt.len() as u64)
            .and_then(|n| n.checked_add(std::mem::size_of::<String>() as u64))
            .ok_or_else(|| Error::Unsupported("MOSS preview size overflow".into()))?;
    }
    if req.config.sample_every > 0 {
        for prompt in req.config.sample_prompts.iter().take(4) {
            largest = largest.max(prompt.len() as u64);
        }
    }
    // 256 bytes of scratch per UTF-8 byte: normalized/pretokenized text, offset pairs,
    // token/word/type/attention vectors and token strings, plus rendered trigger/chat copies.
    // Fixed slack covers an empty template and tokenizer per-call bookkeeping.
    largest
        .checked_add(128)
        .and_then(|n| n.checked_mul(256))
        .and_then(|n| n.checked_add(config_copies.checked_mul(4)?))
        .and_then(|n| n.checked_add(1024 * 1024))
        .ok_or_else(|| Error::Unsupported("MOSS prompt preprocessing bound overflow".into()))
}
pub fn admit_conditioning_preprocess(
    req: &TrainingRequest,
    budget: u64,
    source_metadata: u64,
) -> Result<()> {
    let demand = conditioning_preprocess_bytes(req)?
        .checked_add(DataPlan::metadata_bound(req)?)
        .and_then(|n| n.checked_add(source_metadata))
        .ok_or_else(|| Error::Unsupported("MOSS prompt preprocessing demand overflow".into()))?;
    if demand > budget {
        return Err(Error::Unsupported(
            "MOSS prompt preprocessing exceeds admitted host budget before tokenization".into(),
        ));
    }
    Ok(())
}

/// Borrow one conditioning prompt at a time; only trigger injection allocates a single caption.
/// Call `admit_conditioning_preprocess` before advancing this iterator or tokenizing.
pub fn conditioning_prompts_iter(
    req: &TrainingRequest,
) -> impl Iterator<Item = std::borrow::Cow<'_, str>> {
    let captions = req.items.iter().map(|item| {
        match req
            .config
            .trigger_word
            .as_deref()
            .filter(|t| !t.trim().is_empty())
        {
            Some(trigger) if !item.caption.contains(trigger) => {
                std::borrow::Cow::Owned(format!("{trigger}, {}", item.caption))
            }
            _ => std::borrow::Cow::Borrowed(item.caption.as_str()),
        }
    });
    let previews = req
        .config
        .sample_prompts
        .iter()
        .take(if req.config.sample_every > 0 { 4 } else { 0 })
        .map(|s| std::borrow::Cow::Borrowed(s.as_str()));
    let negative = (req.config.sample_every > 0 && !req.config.sample_prompts.is_empty())
        .then_some(std::borrow::Cow::Borrowed(""));
    captions.chain(previews).chain(negative)
}
/// Convenience collection for explicit fixture controls, not the native pre-admission route.
pub fn conditioning_prompts(req: &TrainingRequest) -> Vec<String> {
    conditioning_prompts_iter(req)
        .map(|s| s.into_owned())
        .collect()
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
            assignment_policy: DataPlan::policy(&req.config).into(),
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
        gen_core::train::LrSchedule::Constant => "constant",
        gen_core::train::LrSchedule::Linear => "linear",
        gen_core::train::LrSchedule::Cosine => "cosine",
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
        "network_type":match c.network_type{gen_core::train::NetworkType::Lora=>"lora",gen_core::train::NetworkType::Lokr=>"lokr"},
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
    fn serialized_studio_square_defaults_schedule_each_image_at_each_resolution() {
        // Frozen consumer shape: snapshot -> draft -> snapshot emits these exact rows.
        const DEFAULTS: &str = r#"[{"resolution":256,"repeats":1},{"resolution":512,"repeats":1},{"resolution":768,"repeats":1},{"resolution":1024,"repeats":1}]"#;
        #[derive(serde::Serialize, serde::Deserialize)]
        struct WireBucket {
            resolution: u32,
            repeats: u32,
        }
        let wire: Vec<WireBucket> = serde_json::from_str(DEFAULTS).unwrap();
        assert_eq!(serde_json::to_string(&wire).unwrap(), DEFAULTS);
        let rows: Vec<gen_core::train::ResolutionBucket> = wire
            .into_iter()
            .map(|r| gen_core::train::ResolutionBucket {
                resolution: r.resolution,
                repeats: r.repeats,
            })
            .collect();
        let mut req = TrainingRequest {
            items: (0..3)
                .map(|i| {
                    gen_core::train::TrainingItem::captioned(
                        format!("image{i}.png").into(),
                        String::new(),
                    )
                })
                .collect(),
            output_dir: "unused".into(),
            file_name: "unused".into(),
            trigger_words: vec![],
            config: gen_core::train::TrainingConfig {
                resolution_buckets: rows.clone(),
                batch_size: 1,
                ..Default::default()
            },
            cancel: gen_core::CancelFlag::new(),
        };
        let dimensions = [(640, 480), (480, 640), (1024, 1024)];
        let plan = DataPlan::new(&req, &dimensions).unwrap();
        let mut observed = std::collections::BTreeMap::new();
        for step in 0..12 {
            for entry in plan.batch(step).unwrap().entries {
                let geometry = &plan.entries[entry];
                let edge = rows[geometry.bucket].resolution;
                let (source_w, source_h) = dimensions[geometry.item];
                // Each resolution is an area budget: the item keeps its own aspect (whole frame,
                // never a centre square) on a 32-multiple canvas of at most edge² pixels.
                assert_eq!(geometry.crop, [0, 0, source_w, source_h]);
                assert_eq!((geometry.width % 32, geometry.height % 32), (0, 0));
                assert!(
                    u64::from(geometry.width) * u64::from(geometry.height)
                        <= u64::from(edge).pow(2)
                );
                assert_eq!(
                    geometry.width > geometry.height,
                    source_w > source_h,
                    "canvas keeps the source orientation"
                );
                assert_eq!(geometry.width == geometry.height, source_w == source_h);
                *observed
                    .entry((geometry.item, geometry.bucket))
                    .or_insert(0) += 1;
            }
        }
        for item in 0..3 {
            for bucket in 0..rows.len() {
                assert_eq!(observed.get(&(item, bucket)), Some(&1));
            }
        }
        assert_eq!(observed.len(), 12);
        let canvas = |item: usize, bucket: usize| {
            let g = plan
                .entries
                .iter()
                .find(|g| g.item == item && g.bucket == bucket)
                .unwrap();
            (g.width, g.height)
        };
        // 640×480 at 256: scaled to ~65 536 px and 32-floored; at 768 and 1024 it is below the
        // budget and trains at its own size (never upscaled).
        assert_eq!(canvas(0, 0), (288, 192));
        assert_eq!(canvas(0, 2), (640, 480));
        assert_eq!(canvas(0, 3), (640, 480));
        assert_eq!(canvas(1, 3), (480, 640));
        // A square source still trains at the square edge.
        for (bucket, row) in rows.iter().enumerate() {
            assert_eq!(canvas(2, bucket), (row.resolution, row.resolution));
        }
        req.config.resolution_buckets.clear();
        req.config.aspect_ratio_buckets = rows
            .iter()
            .map(|r| AspectRatioBucket {
                width: r.resolution,
                height: r.resolution,
                repeats: r.repeats,
            })
            .collect();
        let tied = DataPlan::new(&req, &dimensions).unwrap();
        assert_eq!(tied.entries.len(), 3);
        assert!(tied.entries.iter().all(|e|e.bucket==0 && e.width==256 && e.height==256),
            "rectangular equal-aspect rows choose the earliest row, NOT the legacy resolution ladder");
    }

    fn default_path_request(
        n_items: usize,
        buckets: Vec<gen_core::train::ResolutionBucket>,
        batch_size: u32,
    ) -> TrainingRequest {
        TrainingRequest {
            items: (0..n_items)
                .map(|i| {
                    gen_core::train::TrainingItem::captioned(
                        format!("image{i}.png").into(),
                        String::new(),
                    )
                })
                .collect(),
            output_dir: "unused".into(),
            file_name: "adapter.safetensors".into(),
            trigger_words: vec![],
            config: gen_core::train::TrainingConfig {
                resolution_buckets: buckets,
                batch_size,
                steps: 1000,
                gradient_accumulation: 1,
                seed: 173,
                ..Default::default()
            },
            cancel: gen_core::CancelFlag::new(),
        }
    }

    #[test]
    fn default_path_portrait_trains_its_whole_frame_on_an_aspect_canvas() {
        let bucket = |resolution| gen_core::train::ResolutionBucket {
            resolution,
            repeats: 1,
        };
        let req = default_path_request(1, vec![bucket(1024), bucket(2048)], 1);
        let plan = DataPlan::new(&req, &[(1365, 2048)]).unwrap();
        assert_eq!(plan.entries.len(), 2);
        for (geometry, edge) in plan.entries.iter().zip([1024u64, 2048]) {
            // The whole frame — not the old centred 1365² square.
            assert_eq!(geometry.crop, [0, 0, 1365, 2048]);
            assert_eq!((geometry.width % 32, geometry.height % 32), (0, 0));
            assert!(geometry.height > geometry.width);
            assert!(u64::from(geometry.width) * u64::from(geometry.height) <= edge * edge);
            let aspect = f64::from(geometry.width) / f64::from(geometry.height);
            assert!((aspect - 1365.0 / 2048.0).abs() < 0.03, "{aspect}");
        }
        // At 2048 the portrait is within the area budget: its own size, 32-floored, not upscaled.
        assert_eq!(
            (plan.entries[1].width, plan.entries[1].height),
            (1344, 2048)
        );
        // Packed-token math still holds on the non-square canvas (16-pixel tokens).
        let tokens: u64 = plan
            .entries
            .iter()
            .map(|g| u64::from(g.width / 16) * u64::from(g.height / 16))
            .sum();
        assert_eq!(plan.cached_token_count().unwrap(), tokens);
        assert_eq!(plan.entries[1].width % 16, 0);
        assert!(DataPlan::new(&req, &[(0, 2048)]).is_err());
    }

    #[test]
    fn default_path_batches_are_homogeneous_complete_and_deterministic() {
        let rows = vec![
            gen_core::train::ResolutionBucket {
                resolution: 256,
                repeats: 1,
            },
            gen_core::train::ResolutionBucket {
                resolution: 512,
                repeats: 2,
            },
        ];
        // Squares (one smaller-than-1024 square shares the square canvases) and portraits.
        let dims = [
            (1024, 1024),
            (1365, 2048),
            (1024, 1024),
            (1365, 2048),
            (800, 800),
        ];
        let req = default_path_request(dims.len(), rows.clone(), 2);
        let plan = DataPlan::new(&req, &dims).unwrap();
        let again = DataPlan::new(&req, &dims).unwrap();
        let draws_per_epoch: u64 = (dims.len() as u64) * 3;
        let mut step = 0u32;
        let mut cursor = 0u64;
        let mut epoch_orders = Vec::new();
        for epoch in 0..3u64 {
            let mut counts = std::collections::BTreeMap::new();
            let mut order = Vec::new();
            let mut saw_tail = false;
            loop {
                let draw = plan.batch(step).unwrap();
                if draw.epoch != epoch {
                    break;
                }
                assert_eq!(
                    draw,
                    again.batch(step).unwrap(),
                    "deterministic across plans"
                );
                assert_eq!(
                    draw,
                    plan.batch(step).unwrap(),
                    "deterministic across calls"
                );
                assert_eq!(draw.sample_cursor, cursor);
                assert!(!draw.entries.is_empty() && draw.entries.len() <= 2);
                saw_tail |= draw.entries.len() == 1;
                let first = &plan.entries[draw.entries[0]];
                for &entry in &draw.entries {
                    let g = &plan.entries[entry];
                    assert_eq!((g.width, g.height), (first.width, first.height));
                    *counts.entry(entry).or_insert(0u32) += 1;
                }
                cursor += draw.entries.len() as u64;
                order.push(draw.entries);
                step += 1;
            }
            assert_eq!(cursor, (epoch + 1) * draws_per_epoch);
            assert!(saw_tail, "odd groups keep their tails");
            assert_eq!(counts.len(), plan.entries.len());
            for (entry, count) in counts {
                assert_eq!(count, rows[plan.entries[entry].bucket].repeats);
            }
            epoch_orders.push(order);
        }
        // Constant batches per epoch; the batch order reshuffles per epoch.
        assert!(epoch_orders
            .iter()
            .all(|o| o.len() == epoch_orders[0].len()));
        assert_ne!(epoch_orders[0], epoch_orders[1]);
        // A resume state on the homogeneous path validates; the replaced square policy is refused.
        let identity = RunIdentity {
            digest: "identity".into(),
            source_checkpoint: "fixture-only".into(),
            dataset_fingerprint: "dataset".into(),
            effective_config: "controls".into(),
        };
        let state = RunState::after_step(&req, &plan, &identity, 7, 7, 0, 0.5).unwrap();
        assert_eq!(state.assignment_policy, LEGACY_CANVAS_POLICY);
        state.validate(&req, &plan, &identity).unwrap();
        let mut old = state.clone();
        old.assignment_policy = "legacy-square-bucket-v1".into();
        assert!(old.validate(&req, &plan, &identity).is_err());
        // The 1365×2048 portrait's 256 entry (item-major: item 1, bucket 0) under the old
        // centre-square geometry.
        let mut old = state;
        assert_eq!((old.geometry[2].item, old.geometry[2].bucket), (1, 0));
        assert_ne!(old.geometry[2].width, old.geometry[2].height);
        old.geometry[2].width = 256;
        old.geometry[2].height = 256;
        old.geometry[2].crop = [0, 341, 1365, 1365];
        assert!(old.validate(&req, &plan, &identity).is_err());

        // batch_size 1 keeps the historical one-draw-per-step schedule exactly.
        let single = default_path_request(dims.len(), rows.clone(), 1);
        let plan = DataPlan::new(&single, &dims).unwrap();
        let schedule = BucketSchedule::new(dims.len(), &rows, 173);
        for step in 0..40u32 {
            let draw = plan.batch(step).unwrap();
            let k = step as usize;
            assert_eq!(draw.entries, vec![schedule.cache_index(k)]);
            assert_eq!(draw.sample_cursor, u64::from(step));
            assert_eq!(draw.batch, u64::from(step));
            assert_eq!(draw.epoch, (k / schedule.epoch_len()) as u64);
        }
    }

    #[test]
    fn huge_preview_refuses_before_render_or_tokenizer_and_lazy_prompts_preserve_semantics() {
        let mut req = TrainingRequest {
            items: vec![gen_core::train::TrainingItem::captioned(
                "small.png".into(),
                "caption".into(),
            )],
            output_dir: "unused".into(),
            file_name: "unused".into(),
            trigger_words: vec![],
            config: gen_core::train::TrainingConfig::default(),
            cancel: gen_core::CancelFlag::new(),
        };
        req.config.sample_every = 1;
        req.config.sample_prompts = vec!["short ".repeat(1_000_000)];
        let budget = 64 * 1024 * 1024;
        assert!(matches!(admit_conditioning_preprocess(&req, budget, 0),
            Err(Error::Unsupported(message)) if message.contains("before tokenization")));
        req.config.sample_prompts = vec!["preview".into()];
        req.config.trigger_word = Some("ohwx".into());
        admit_conditioning_preprocess(&req, budget, 0).unwrap();
        assert_eq!(conditioning_prompts(&req), ["ohwx, caption", "preview", ""]);
        let mut iter = conditioning_prompts_iter(&req);
        assert!(matches!(iter.next(), Some(std::borrow::Cow::Owned(_))));
        assert!(matches!(
            iter.next(),
            Some(std::borrow::Cow::Borrowed("preview"))
        ));
    }

    #[test]
    fn caller_cap_never_raises_physical_and_zero_refuses() {
        assert_eq!(
            gen_core::train::effective_memory_budget(1024, Some(4096)).unwrap(),
            1024
        );
        assert_eq!(
            gen_core::train::effective_memory_budget(1024, Some(173)).unwrap(),
            173
        );
        assert!(gen_core::train::effective_memory_budget(1024, Some(0)).is_err());
        assert!(gen_core::train::effective_memory_budget(0, None).is_err());
    }
    #[test]
    fn rectangular_cursor_counts_samples_not_padded_microbatch_slots() {
        let req = TrainingRequest {
            config: gen_core::train::TrainingConfig {
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
                .map(|i| {
                    gen_core::train::TrainingItem::captioned(
                        format!("item{i}").into(),
                        String::new(),
                    )
                })
                .collect(),
            output_dir: "output".into(),
            file_name: "adapter.safetensors".into(),
            trigger_words: Vec::new(),
            cancel: gen_core::CancelFlag::new(),
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
            items: vec![gen_core::train::TrainingItem::captioned(
                image,
                "caption".into(),
            )],
            config: gen_core::train::TrainingConfig {
                steps: 6,
                batch_size: 1,
                gradient_accumulation: 2,
                resolution: 32,
                ..Default::default()
            },
            output_dir: root.path().into(),
            file_name: "adapter.safetensors".into(),
            trigger_words: Vec::new(),
            cancel: gen_core::CancelFlag::new(),
        };
        let plan = DataPlan::from_request_budget(&request, 2 * 1024 * 1024 * 1024).unwrap();
        let baseline = effective_identity(&request, "fixture-parent-a", &plan).unwrap();
        let legacy = gen_core_train::resume::training_config_fingerprint(&request.config);
        let mut retry = request.clone();
        retry.config.resume = true;
        retry.config.memory_limit_bytes = Some(173);
        retry.config.memory_limit_fraction = Some(0.5);
        assert_eq!(
            gen_core_train::resume::training_config_fingerprint(&retry.config),
            legacy
        );
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
                    gen_core_train::resume::training_config_fingerprint(&changed.config),
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
            items: vec![gen_core::train::TrainingItem::captioned(
                first,
                "caption".into(),
            )],
            config: gen_core::train::TrainingConfig {
                resolution: 32,
                ..Default::default()
            },
            output_dir: root.path().into(),
            file_name: "adapter.safetensors".into(),
            trigger_words: Vec::new(),
            cancel: gen_core::CancelFlag::new(),
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
