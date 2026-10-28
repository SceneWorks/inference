//! Small explicit native-training acceptance probe. No device/tensor dependency, global hooks,
//! production request options, optimizer access, or training random-stream mutation.
mod snapshot;
pub use snapshot::{AdapterSnapshot, CheckpointObservation, FactorDelta};
/// Existing admitted audit allowance, shared with native preparation and bounded file readers.
pub const AUDIT_WORKSPACE_BYTES: u64 = 64 * 1024 * 1024;
use gen_core::{Error, Result, TrainingItem, TrainingRequest};
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use std::{
    collections::BTreeMap,
    path::PathBuf,
    sync::{Arc, Mutex},
};

#[derive(Clone, Debug)]
pub struct EvaluationManifest {
    pub item: TrainingItem,
    pub width: u32,
    pub height: u32,
    pub noise_seed: u64,
    pub sigma: f32,
    pub identity: String,
    image_digest: [u8; 32],
    mask_digest: Option<[u8; 32]>,
}
impl EvaluationManifest {
    pub fn new(
        item: TrainingItem,
        width: u32,
        height: u32,
        noise_seed: u64,
        sigma: f32,
    ) -> Result<Self> {
        if width == 0
            || height == 0
            || !width.is_multiple_of(32)
            || !height.is_multiple_of(32)
            || !sigma.is_finite()
            || !(0. ..=1.).contains(&sigma)
        {
            return Err(Error::Msg("invalid fixed evaluation manifest".into()));
        }
        fn file_hash(path: &std::path::Path) -> Result<[u8; 32]> {
            use std::io::Read;
            let mut file = std::fs::File::open(path)?;
            let mut hash = Sha256::new();
            let mut buffer = [0u8; 65536];
            loop {
                let n = file.read(&mut buffer)?;
                if n == 0 {
                    break;
                }
                hash.update(&buffer[..n]);
            }
            Ok(hash.finalize().into())
        }
        let image_digest = file_hash(&item.image_path)?;
        let mask_digest = item
            .subject_mask_path
            .as_deref()
            .map(file_hash)
            .transpose()?;
        let identity=format!("{:x}",Sha256::digest(format!("moss-heldout-v1:{}:{image_digest:?}:{mask_digest:?}:{width}:{height}:{noise_seed}:{sigma:?}",item.caption)));
        Ok(Self {
            item,
            width,
            height,
            noise_seed,
            sigma,
            identity,
            image_digest,
            mask_digest,
        })
    }
    pub fn verify(&self) -> Result<()> {
        let actual = Self::new(
            self.item.clone(),
            self.width,
            self.height,
            self.noise_seed,
            self.sigma,
        )?;
        if actual.identity != self.identity {
            return Err(Error::Unsupported(
                "immutable heldout evaluation input changed".into(),
            ));
        }
        Ok(())
    }
    pub fn verify_snapshot(
        &self,
        plan: &gen_core::moss_image_training::recovery::DataPlan,
        masked: bool,
    ) -> Result<()> {
        if plan.image_digest(0)? != self.image_digest
            || (masked && plan.mask_digest(0)? != self.mask_digest)
        {
            return Err(Error::Unsupported(
                "captured heldout bytes differ from immutable evaluation manifest".into(),
            ));
        }
        Ok(())
    }
    pub fn request(&self, request: &TrainingRequest) -> Result<TrainingRequest> {
        self.verify()?;
        let mut evaluation = request.clone();
        evaluation.items = vec![self.item.clone()];
        evaluation.config.resolution_buckets.clear();
        evaluation.config.aspect_ratio_buckets = vec![gen_core::AspectRatioBucket {
            width: self.width,
            height: self.height,
            repeats: 1,
        }];
        evaluation.config.resolution = self.width;
        evaluation.config.sample_every = 0;
        evaluation.config.sample_prompts.clear();
        evaluation.config.resume = false;
        Ok(evaluation)
    }
}
#[derive(Clone, Debug, Default, Serialize, Deserialize)]
pub struct FrozenReceipt {
    pub before: String,
    pub after: String,
}
/// Canonical hash encoding shared by both native diagnostic implementations.
pub const EVALUATION_CONTENT_ENCODING: &str = "logical-row-major-f32-le-v1";

/// Streaming canonical serialization: source values converted to F32, visited in
/// logical row-major order, each serialized with `f32::to_le_bytes` (no header/padding).
#[derive(Default)]
pub struct EvaluationContentHasher(Sha256);
impl EvaluationContentHasher {
    pub fn update(&mut self, values: impl IntoIterator<Item = f32>) {
        for value in values {
            self.0.update(value.to_le_bytes());
        }
    }
    pub fn finish(self) -> String {
        format!("{:x}", self.0.finalize())
    }
}

/// Actual prepared tensor metadata; hashes cover logical row-major F32 little-endian
/// values, not storage padding or a regenerated input. Dtype/strides describe the source.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct EvaluationTensorDescriptor {
    pub shape: Vec<usize>,
    pub strides: Vec<usize>,
    pub dtype: String,
    pub content_encoding: String,
    /// SHA256 of the canonical bytes defined by `content_encoding`. Both backends use
    /// EvaluationContentHasher, but acceptance compares before/after ONLY WITHIN a run;
    /// cross-backend numerical/hash equality is neither required nor expected. Never
    /// compare content hashes with different encoding labels.
    pub content_sha256: String,
}
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct EvaluationObservation {
    pub tensors: BTreeMap<String, EvaluationTensorDescriptor>,
    pub sigma: f32,
    pub loss: f32,
}
#[derive(Clone, Debug, Default, Serialize, Deserialize)]
pub struct PreparedEvaluationReceipt {
    pub before: Option<EvaluationObservation>,
    pub after: Option<EvaluationObservation>,
}
#[derive(Clone, Debug, Default, Serialize, Deserialize)]
pub struct RunReceipt {
    /// Test-support-only evidence, populated exclusively by an attached native probe.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub prepared_evaluation: Option<PreparedEvaluationReceipt>,
    #[serde(default)]
    pub request_identity: Option<gen_core::moss_image_training::recovery::RunIdentity>,
    #[serde(default)]
    pub resolved_seed: Option<u64>,
    #[serde(default)]
    pub memory_limit_bytes: Option<u64>,
    #[serde(default)]
    pub memory_limit_fraction: Option<f32>,
    #[serde(default)]
    pub run_start_adapter: Option<AdapterSnapshot>,
    #[serde(default)]
    pub final_adapter_path: Option<PathBuf>,
    #[serde(default)]
    pub final_native_path: Option<PathBuf>,
    #[serde(default)]
    pub final_parameter_delta: Option<FactorDelta>,
    #[serde(default)]
    pub checkpoints: Vec<CheckpointObservation>,
    pub cpu_reference: bool,
    pub evaluation_manifest: String,
    pub evaluation_before: Option<f32>,
    pub evaluation_after: Option<f32>,
    pub factor_before: String,
    pub factor_after: String,
    pub gradient_tensors: u64,
    pub gradients_finite: bool,
    pub nonzero_gradient: bool,
    pub frozen: BTreeMap<String, FrozenReceipt>,
    pub context_bound: u64,
    pub actual_context_max: u64,
    pub microsteps: Vec<u32>,
    pub masks: BTreeMap<String, Vec<f32>>,
    pub complete: bool,
    pub run_scope_returned: bool,
    pub error: Option<String>,
}
#[derive(Clone, Debug)]
pub struct TrainingProbe {
    evaluation: EvaluationManifest,
    receipts: Arc<Mutex<Vec<RunReceipt>>>,
    generation: Arc<Mutex<Option<serde_json::Value>>>,
    // Output-only diagnostic path supplied by the owned CPU supervisor, never a model/source knob.
    artifact_path: Option<PathBuf>,
}
#[derive(Clone, Debug)]
pub struct RunProbe {
    owner: TrainingProbe,
    index: usize,
}
#[derive(Serialize)]
struct EvaluationArtifact<'a> {
    identity: &'a str,
    width: u32,
    height: u32,
    noise_seed: u64,
    sigma: f32,
    caption: &'a str,
    image_sha256: &'a [u8; 32],
    mask_sha256: Option<&'a [u8; 32]>,
}
#[derive(Serialize)]
struct ObservationArtifact<'a> {
    version: &'static str,
    source_head: Option<String>,
    admitted_audit_workspace_bytes: u64,
    evaluation: EvaluationArtifact<'a>,
    receipts: &'a [RunReceipt],
    generation: &'a Option<serde_json::Value>,
    parameter_hash_changed: Vec<Option<bool>>,
}
#[derive(Serialize)]
struct CheckpointArtifact<'a> {
    source_head: Option<String>,
    run_index: usize,
    checkpoint: &'a CheckpointObservation,
}
impl TrainingProbe {
    pub fn new(evaluation: EvaluationManifest) -> Self {
        Self {
            evaluation,
            receipts: Arc::new(Mutex::new(Vec::new())),
            generation: Arc::new(Mutex::new(None)),
            artifact_path: std::env::var_os("MOSS_CPU_RECEIPT_PATH").map(PathBuf::from),
        }
    }
    /// Keep native export witnesses available in ordinary fixture tests too, without
    /// changing the supervisor-selected output location or mutating process-global env.
    pub fn with_artifact_fallback(mut self, path: PathBuf) -> Self {
        if self.artifact_path.is_none() {
            self.artifact_path = Some(path);
        }
        self
    }
    fn persist(&self, receipts: &[RunReceipt]) -> Result<()> {
        let Some(path) = &self.artifact_path else {
            return Ok(());
        };
        let parent = path
            .parent()
            .ok_or_else(|| Error::Msg("receipt output parent missing".into()))?;
        if !path.is_absolute() || !std::fs::symlink_metadata(parent)?.is_dir() {
            return Err(Error::Unsupported(
                "receipt output must have an owned absolute directory".into(),
            ));
        }
        match std::fs::symlink_metadata(path) {
            Ok(metadata) if !metadata.is_file() => {
                return Err(Error::Unsupported(
                    "receipt output is not a regular file".into(),
                ))
            }
            Ok(_) => (),
            Err(error) if error.kind() == std::io::ErrorKind::NotFound => (),
            Err(error) => return Err(error.into()),
        }
        let parameter_delta: Vec<_> = receipts
            .iter()
            .map(|r| {
                if r.factor_before.is_empty() || r.factor_after.is_empty() {
                    None
                } else {
                    Some(r.factor_before != r.factor_after)
                }
            })
            .collect();
        let generation = self
            .generation
            .lock()
            .map_err(|_| Error::Msg("generation observation poisoned".into()))?;
        let artifact = ObservationArtifact {
            version: "moss-cpu-training-observations-v1",
            source_head: std::env::var("GITHUB_SHA").ok(),
            admitted_audit_workspace_bytes: AUDIT_WORKSPACE_BYTES,
            evaluation: EvaluationArtifact {
                identity: &self.evaluation.identity,
                width: self.evaluation.width,
                height: self.evaluation.height,
                noise_seed: self.evaluation.noise_seed,
                sigma: self.evaluation.sigma,
                caption: &self.evaluation.item.caption,
                image_sha256: &self.evaluation.image_digest,
                mask_sha256: self.evaluation.mask_digest.as_ref(),
            },
            receipts,
            generation: &generation,
            parameter_hash_changed: parameter_delta,
        };
        let staging = path.with_extension("receipts.partial");
        let mut file = std::fs::OpenOptions::new()
            .write(true)
            .create_new(true)
            .open(&staging)?;
        let result = (|| -> Result<()> {
            serde_json::to_writer(&mut file, &artifact).map_err(|e| Error::Msg(e.to_string()))?;
            file.sync_all()?;
            std::fs::rename(&staging, path)?;
            Ok(())
        })();
        if result.is_err() {
            let _ = std::fs::remove_file(staging);
        }
        result
    }
    pub fn begin(&self, cpu_reference: bool) -> Result<RunProbe> {
        self.evaluation.verify()?;
        let mut receipts = self
            .receipts
            .lock()
            .map_err(|_| Error::Msg("training probe poisoned".into()))?;
        let index = receipts.len();
        receipts.push(RunReceipt {
            cpu_reference,
            evaluation_manifest: self.evaluation.identity.clone(),
            gradients_finite: true,
            ..Default::default()
        });
        self.persist(&receipts)?;
        Ok(RunProbe {
            owner: self.clone(),
            index,
        })
    }
    /// The actual native step-zero export of the completed uninterrupted run, never TrainingOutput.
    pub fn initial_untrained_adapter(&self) -> Result<Option<AdapterSnapshot>> {
        let receipts = self
            .receipts
            .lock()
            .map_err(|_| Error::Msg("training probe poisoned".into()))?;
        let initial = receipts
            .iter()
            .find(|r| {
                r.complete
                    && r.microsteps.first() == Some(&1)
                    && r.run_start_adapter
                        .as_ref()
                        .is_some_and(|s| s.start_step == 0)
            })
            .and_then(|r| r.run_start_adapter.clone());
        if let Some(initial) = &initial {
            if snapshot::hash_file(&initial.path)? != initial.sha256 {
                return Err(Error::Unsupported(
                    "initial diagnostic adapter changed".into(),
                ));
            }
        }
        Ok(initial)
    }
    /// Observations from real FP32 route-matched controls; presence/zero controls alone are not learning proof.
    /// This records diagnostics only; every original acceptance assertion remains in the caller.
    pub fn generation_observation(
        &self,
        request: &gen_core::GenerationRequest,
        baseline: (&[usize; 4], &[f32]),
        trained: (&[usize; 4], &[f32]),
        disabled: (&[usize; 4], &[f32]),
        initial: Option<(&[usize; 4], &[f32], bool)>,
        trained_adapter: (&std::path::Path, bool),
    ) -> Result<()> {
        let compare = |reference: &[f32], values: &[f32]| {
            let finite = reference.iter().chain(values).all(|v| v.is_finite());
            let nonzero = reference.iter().zip(values).filter(|(a, b)| a != b).count();
            let maximum = if finite && reference.len() == values.len() && !values.is_empty() {
                Some(
                    reference
                        .iter()
                        .zip(values)
                        .map(|(a, b)| (f64::from(*a) - f64::from(*b)).abs())
                        .fold(0.0_f64, f64::max),
                )
            } else {
                None
            };
            serde_json::json!({"elements":values.len(),"finite":finite,
                "nonzero_elements_in_overlap":nonzero,"max_abs_delta":maximum})
        };
        let initial_export = self.initial_untrained_adapter()?;
        let matched_binding = if self.artifact_path.is_some() {
            let actual_sha = snapshot::hash_file(trained_adapter.0)?;
            let all = self
                .receipts
                .lock()
                .map_err(|_| Error::Msg("training probe poisoned".into()))?;
            let (index, trained_run) = all
                .iter()
                .enumerate()
                .find(|(_, r)| r.final_native_path.as_deref() == Some(trained_adapter.0))
                .ok_or_else(|| {
                    Error::Unsupported(
                        "generation adapter is not an observed native TrainingOutput".into(),
                    )
                })?;
            let start = trained_run
                .run_start_adapter
                .as_ref()
                .ok_or_else(|| Error::Unsupported("trained adapter start export missing".into()))?;
            let initial = initial_export
                .as_ref()
                .ok_or_else(|| Error::Unsupported("real untrained control missing".into()))?;
            if initial.start_step != 0
                || initial.layout_sha256 != start.layout_sha256
                || initial.export_identity != start.export_identity
                || trained_run.final_parameter_delta.as_ref().is_none_or(|d| {
                    d.final_sha256 != actual_sha || d.layout_sha256 != initial.layout_sha256
                })
            {
                return Err(Error::Unsupported(
                    "generation control parent/source/layout/final export differs".into(),
                ));
            }
            Some(
                serde_json::json!({"trained_native_path":trained_adapter.0,"trained_export_sha256":actual_sha,
                "matched_run_index":index,"matched_run_start_step":start.start_step,
                "initial_factor_state_sha256":all.iter().find(|r|r.run_start_adapter.as_ref().is_some_and(|s|s.path==initial.path)).map(|r|&r.factor_before),
                "trained_factor_state_sha256":&trained_run.factor_after,
                "layout_sha256":&initial.layout_sha256,"export_identity":&initial.export_identity}),
            )
        } else {
            None
        };
        let observation = serde_json::json!({
            "request":{"prompt":&request.prompt,"width":request.width,"height":request.height,
                "seed":request.seed,"steps":request.steps,"guidance":request.guidance,
                "max_sequence_length":request.moss_image.as_ref().map(|m|m.max_sequence_length)},
            "baseline_shape":baseline.0,"baseline_elements":baseline.1.len(),
            "trained_shape":trained.0,"zero_strength_shape":disabled.0,
            "trained":compare(baseline.1,trained.1),"zero_strength":compare(baseline.1,disabled.1),
            "route_matched_initial":initial.map(|(shape,values,applied)| serde_json::json!({
                "shape":shape,"adapter_applied":applied,
                "initial_vs_no_adapter":compare(baseline.1,values),
                "trained_vs_initial":compare(values,trained.1),
            })),
            "initial_native_export":initial_export,
            "native_export_binding":matched_binding,
            "adapter_applied":trained_adapter.1,
        });
        // Same lock order as update/persist: receipts, then diagnostic generation metadata.
        let receipts = self
            .receipts
            .lock()
            .map_err(|_| Error::Msg("training probe poisoned".into()))?;
        *self
            .generation
            .lock()
            .map_err(|_| Error::Msg("generation observation poisoned".into()))? = Some(observation);
        self.persist(&receipts)
    }
    pub fn receipts(&self) -> Result<Vec<RunReceipt>> {
        Ok(self
            .receipts
            .lock()
            .map_err(|_| Error::Msg("training probe poisoned".into()))?
            .clone())
    }
    pub fn evaluation(&self) -> &EvaluationManifest {
        &self.evaluation
    }
}
impl RunProbe {
    fn artifact_target(&self, label: &str) -> Result<Option<PathBuf>> {
        let Some(path) = &self.owner.artifact_path else {
            return Ok(None);
        };
        let stem = path
            .file_stem()
            .ok_or_else(|| Error::Msg("diagnostic artifact stem missing".into()))?
            .to_string_lossy();
        let target = path.with_file_name(format!("{stem}.run{}.{}.safetensors", self.index, label));
        match std::fs::symlink_metadata(&target) {
            Err(e) if e.kind() == std::io::ErrorKind::NotFound => Ok(Some(target)),
            Err(e) => Err(e.into()),
            Ok(_) => Err(Error::Unsupported(
                "diagnostic target already exists; no overwrite".into(),
            )),
        }
    }
    pub fn run_start_export_target(&self) -> Result<Option<PathBuf>> {
        self.artifact_target("native-start")
    }
    pub fn record_run_start_export(
        &self,
        path: &std::path::Path,
        step: u32,
        update: u32,
        variant: gen_core::moss_image::MossImageVariant,
    ) -> Result<()> {
        let info = snapshot::inspect(path, step, update, AUDIT_WORKSPACE_BYTES)?;
        let metadata: BTreeMap<_, _> = info.export_identity.iter().cloned().collect();
        gen_core::moss_image_training::validate_metadata(variant, &metadata)?;
        let request = self
            .owner
            .receipts
            .lock()
            .map_err(|_| Error::Msg("training probe poisoned".into()))?
            .get(self.index)
            .and_then(|r| r.request_identity.clone())
            .ok_or_else(|| Error::Unsupported("native export request binding missing".into()))?;
        if metadata.get("source_checkpoint_identity") != Some(&request.source_checkpoint)
            || metadata.get("dataset_fingerprint") != Some(&request.dataset_fingerprint)
            || metadata.get("effective_training_controls") != Some(&request.effective_config)
        {
            return Err(Error::Unsupported(
                "native export does not match actual request source/config/dataset".into(),
            ));
        }
        self.update(|r| r.run_start_adapter = Some(info))
    }
    fn start_snapshot(&self) -> Result<Option<AdapterSnapshot>> {
        Ok(self
            .owner
            .receipts
            .lock()
            .map_err(|_| Error::Msg("training probe poisoned".into()))?
            .get(self.index)
            .and_then(|r| r.run_start_adapter.clone()))
    }
    pub fn record_checkpoint(
        &self,
        bundle: &std::path::Path,
        adapter: &std::path::Path,
    ) -> Result<()> {
        if self.owner.artifact_path.is_none() {
            return Ok(());
        }
        let initial = self
            .start_snapshot()?
            .ok_or_else(|| Error::Unsupported("run-start diagnostic export missing".into()))?;
        let state_bytes = std::fs::metadata(bundle.join("run.json"))?.len();
        let state_workspace = state_bytes
            .checked_mul(6)
            .and_then(|n| n.checked_add(128 * 1024))
            .ok_or_else(|| {
                Error::Unsupported("checkpoint observation workspace overflow".into())
            })?;
        if state_workspace > AUDIT_WORKSPACE_BYTES {
            return Err(Error::Unsupported(
                "checkpoint observations exceed admitted audit workspace".into(),
            ));
        }
        let state = gen_core::moss_image_training::recovery::bundle::read(bundle)?;
        let reader_budget = AUDIT_WORKSPACE_BYTES
            .checked_sub(state_workspace)
            .ok_or_else(|| {
                Error::Unsupported("checkpoint observation workspace exceeds admission".into())
            })?;
        let factor_delta = snapshot::delta(&initial.path, adapter, reader_budget)?;
        if factor_delta.initial_sha256 != initial.sha256 {
            return Err(Error::Unsupported("run-start export changed".into()));
        }
        let (optimizer_scalars_as_stored, scalar_workspace) =
            snapshot::optimizer_scalars(&bundle.join("optimizer.safetensors"), reader_budget)?;
        let workspace_bytes = state_workspace + factor_delta.workspace_bytes.max(scalar_workspace);
        let observation = CheckpointObservation {
            state,
            factor_delta,
            optimizer_scalars_as_stored,
            workspace_bytes,
        };
        let path = self
            .artifact_target(&format!("checkpoint-{:06}", observation.state.step))?
            .ok_or_else(|| Error::Unsupported("checkpoint artifact target missing".into()))?
            .with_extension("json");
        let mut file = std::fs::OpenOptions::new()
            .write(true)
            .create_new(true)
            .open(path)?;
        serde_json::to_writer(
            &mut file,
            &CheckpointArtifact {
                source_head: std::env::var("GITHUB_SHA").ok(),
                run_index: self.index,
                checkpoint: &observation,
            },
        )
        .map_err(|e| Error::Msg(e.to_string()))?;
        file.sync_all()?;
        // Durable per-checkpoint files hold the timeline; retain only the first/latest anchors.
        self.update(|r| {
            if r.checkpoints.len() < 2 {
                r.checkpoints.push(observation);
            } else {
                r.checkpoints[1] = observation;
            }
        })
    }
    pub fn record_final_export(&self, path: &std::path::Path) -> Result<()> {
        let Some(target) = self.artifact_target("native-final-copy")? else {
            return Ok(());
        };
        let initial = self
            .start_snapshot()?
            .ok_or_else(|| Error::Unsupported("run-start diagnostic export missing".into()))?;
        let before = snapshot::hash_file(path)?;
        snapshot::copy_export(path, &target)?;
        let delta = snapshot::delta(&initial.path, &target, AUDIT_WORKSPACE_BYTES)?;
        if delta.initial_sha256 != initial.sha256
            || delta.final_sha256 != before
            || snapshot::hash_file(path)? != before
        {
            return Err(Error::Unsupported(
                "native export changed during diagnostic capture".into(),
            ));
        }
        self.update(|r| {
            r.final_adapter_path = Some(target);
            r.final_native_path = Some(path.to_path_buf());
            r.final_parameter_delta = Some(delta);
        })
    }
    pub fn evaluation(&self) -> &EvaluationManifest {
        self.owner.evaluation()
    }
    pub fn update(&self, write: impl FnOnce(&mut RunReceipt)) -> Result<()> {
        let mut receipts = self
            .owner
            .receipts
            .lock()
            .map_err(|_| Error::Msg("training probe poisoned".into()))?;
        let receipt = receipts
            .get_mut(self.index)
            .ok_or_else(|| Error::Msg("training probe run missing".into()))?;
        write(receipt);
        // Persist before frozen/gradient/heldout assertions can fail or unwind.
        self.owner.persist(&receipts)
    }
    pub fn bind_request(
        &self,
        request: &TrainingRequest,
        plan: &gen_core::moss_image_training::recovery::DataPlan,
        source: &str,
    ) -> Result<()> {
        let identity =
            gen_core::moss_image_training::recovery::effective_identity(request, source, plan)?;
        self.update(|r| {
            r.request_identity = Some(identity);
            r.resolved_seed = Some(request.config.seed);
            r.memory_limit_bytes = request.config.memory_limit_bytes;
            r.memory_limit_fraction = request.config.memory_limit_fraction;
        })
    }
    pub fn frozen(&self, name: &str, before: &str, after: &str) -> Result<()> {
        self.update(|r| {
            r.frozen.insert(
                name.into(),
                FrozenReceipt {
                    before: before.into(),
                    after: after.into(),
                },
            );
        })?;
        if before != after {
            return Err(Error::Unsupported(format!(
                "frozen {name} changed during actual native training"
            )));
        }
        Ok(())
    }
    pub fn gradients(&self, count: u64, finite: bool, nonzero: bool) -> Result<()> {
        self.update(|r| {
            r.gradient_tensors = r.gradient_tensors.saturating_add(count);
            r.gradients_finite &= finite;
            r.nonzero_gradient |= nonzero;
        })?;
        if !finite {
            return Err(Error::Unsupported(
                "nonfinite observed native gradients".into(),
            ));
        }
        Ok(())
    }
    pub fn finish(&self, error: Option<String>) -> Result<()> {
        self.evaluation().verify()?;
        self.update(|r| {
            r.run_scope_returned = true;
            r.complete = error.is_none();
            r.error = error;
        })
    }
}

/// These assertions consume actual native observations; missing/default receipt fields cannot pass.
pub fn assert_complete_case(probe: &TrainingProbe, masks_required: bool) -> Result<()> {
    let receipts = probe.receipts()?;
    let full = receipts
        .iter()
        .find(|r| r.complete && r.microsteps == [1, 2, 3, 4, 5, 6])
        .expect("actual uninterrupted native run receipt");
    let resumed = receipts
        .iter()
        .find(|r| r.complete && r.microsteps == [4, 5, 6])
        .expect("actual resumed native run receipt");
    let canceled = receipts
        .iter()
        .find(|r| !r.complete && r.microsteps == [1, 2, 3])
        .expect("actual mid-accumulation cancellation receipt");
    assert!(canceled.run_scope_returned);
    for receipt in [full, resumed] {
        assert!(
            receipt.cpu_reference,
            "fixture proof must be labeled CPU, not GPU execution"
        );
        assert!(
            receipt.run_scope_returned
                && receipt.gradient_tensors > 0
                && receipt.gradients_finite
                && receipt.nonzero_gradient
        );
        let initial = receipt
            .run_start_adapter
            .as_ref()
            .expect("actual native run-start portable export");
        let delta = receipt
            .final_parameter_delta
            .as_ref()
            .expect("actual streamed native portable factor delta");
        assert!(
            delta.tensors > 0
                && delta.elements > 0
                && delta.changed_elements > 0
                && delta.changed_elements <= delta.elements
                && delta.max_abs_delta.is_finite()
                && delta.max_abs_delta > 0.0
                && delta.squared_l2_delta.is_finite()
                && delta.squared_l2_delta > 0.0,
            "native portable factors did not numerically update"
        );
        assert_eq!(delta.initial_sha256, initial.sha256);
        assert_eq!(delta.layout_sha256, initial.layout_sha256);
        assert_eq!(delta.tensors, initial.factor_tensors);
        assert!(receipt.final_adapter_path.is_some() && receipt.final_native_path.is_some());
        assert!(!receipt.factor_before.is_empty() && !receipt.factor_after.is_empty());
        assert_ne!(
            receipt.factor_before, receipt.factor_after,
            "actual adapter factors did not change"
        );
        assert!(
            receipt.actual_context_max > 0 && receipt.actual_context_max <= receipt.context_bound,
            "executed ragged context exceeded the actual pre-load bound"
        );
        for component in ["backbone", "text_encoder", "vae"] {
            let frozen = receipt
                .frozen
                .get(component)
                .expect("actual loaded tensor invariant missing");
            assert!(!frozen.before.is_empty());
            assert_eq!(frozen.before, frozen.after, "frozen {component} changed");
        }
        let before = receipt
            .evaluation_before
            .expect("actual heldout before objective");
        let after = receipt
            .evaluation_after
            .expect("actual heldout after objective");
        // Independent heldout loss remains diagnostic: three stochastic updates do not
        // imply universal generalization. Fixed-evaluation learning belongs to the
        // epic's real-weight terminal cases, not this CPU recovery/update contract.
        assert!(before.is_finite() && after.is_finite());
        let prepared = receipt
            .prepared_evaluation
            .as_ref()
            .expect("prepared evaluation evidence");
        let observed_before = prepared
            .before
            .as_ref()
            .expect("prepared before evaluation");
        let observed_after = prepared.after.as_ref().expect("prepared after evaluation");
        assert_eq!(observed_before.loss.to_bits(), before.to_bits());
        assert_eq!(observed_after.loss.to_bits(), after.to_bits());
        assert!(observed_before.loss.is_finite() && observed_after.loss.is_finite());
        assert!(observed_before.sigma.is_finite());
        assert_eq!(
            observed_before.sigma.to_bits(),
            observed_after.sigma.to_bits()
        );
        assert_eq!(
            observed_before.tensors.keys().collect::<Vec<_>>(),
            observed_after.tensors.keys().collect::<Vec<_>>(),
            "prepared evaluation tensor key sets changed"
        );
        assert_eq!(
            observed_before.tensors, observed_after.tensors,
            "before/after must consume identical prepared tensors"
        );
        let expected = if masks_required {
            vec!["context", "latent", "mask", "noise"]
        } else {
            vec!["context", "latent", "noise"]
        };
        assert_eq!(
            observed_before
                .tensors
                .keys()
                .map(String::as_str)
                .collect::<Vec<_>>(),
            expected
        );
        for descriptor in observed_before.tensors.values() {
            assert!(!descriptor.shape.is_empty() && descriptor.shape.iter().all(|&d| d > 0));
            assert_eq!(descriptor.shape.len(), descriptor.strides.len());
            assert!(!descriptor.dtype.is_empty());
            assert_eq!(descriptor.content_encoding, "logical-row-major-f32-le-v1");
            assert_eq!(descriptor.content_sha256.len(), 64);
            assert!(descriptor
                .content_sha256
                .bytes()
                .all(|c| c.is_ascii_hexdigit()));
        }
    }
    if masks_required {
        assert_eq!(full.masks.get("64x32"), Some(&vec![1., 1., 0.25, 0.25]));
        assert_eq!(full.masks.get("32x64"), Some(&vec![1., 0.25]));
        assert_eq!(full.masks, resumed.masks);
    }
    assert_eq!(full.run_start_adapter.as_ref().unwrap().start_step, 0);
    assert_eq!(resumed.run_start_adapter.as_ref().unwrap().start_step, 3);
    assert!(full.request_identity.is_some());
    assert_eq!(full.request_identity, resumed.request_identity);
    assert_eq!(full.resolved_seed, resumed.resolved_seed);
    assert_eq!(full.evaluation_manifest, resumed.evaluation_manifest);
    assert_eq!(full.factor_after, resumed.factor_after);
    assert_eq!(
        full.evaluation_after.map(f32::to_bits),
        resumed.evaluation_after.map(f32::to_bits),
        "evaluation must not advance the original training RNG/cursor/optimizer"
    );
    Ok(())
}

#[cfg(test)]
mod prepared_evaluation_tests {
    use super::*;

    #[test]
    fn canonical_content_hash_is_little_endian_and_chunk_independent() {
        let expected = format!(
            "{:x}",
            Sha256::digest([
                0x00, 0x00, 0x80, 0x3f, // 1.0
                0x00, 0x00, 0x00, 0x80, // -0.0
                0x00, 0x00, 0x20, 0xc0, // -2.5
            ])
        );
        let mut whole = EvaluationContentHasher::default();
        whole.update([1.0, -0.0, -2.5]);
        let mut chunks = EvaluationContentHasher::default();
        chunks.update([1.0]);
        chunks.update([-0.0, -2.5]);
        assert_eq!(whole.finish(), expected);
        assert_eq!(chunks.finish(), expected);
    }

    #[test]
    fn unattached_receipt_omits_prepared_evaluation_and_old_receipts_default_to_absent() {
        let value = serde_json::to_value(RunReceipt::default()).unwrap();
        assert!(value.get("prepared_evaluation").is_none());
        let restored: RunReceipt = serde_json::from_value(value).unwrap();
        assert!(restored.prepared_evaluation.is_none());
    }
}
