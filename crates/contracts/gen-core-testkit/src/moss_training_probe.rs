//! Small explicit native-training acceptance probe. No device/tensor dependency, global hooks,
//! production request options, optimizer access, or training random-stream mutation.
use gen_core::{Error, Result, TrainingItem, TrainingRequest};
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use std::{
    collections::BTreeMap,
    io::Write,
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
#[derive(Clone, Debug, Default, Serialize, Deserialize)]
pub struct RunReceipt {
    #[serde(default)]
    pub request_identity: Option<gen_core::moss_image_training::recovery::RunIdentity>,
    #[serde(default)]
    pub resolved_seed: Option<u64>,
    #[serde(default)]
    pub memory_limit_bytes: Option<u64>,
    #[serde(default)]
    pub memory_limit_fraction: Option<f32>,
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
    // Output-only diagnostic path supplied by the owned CPU supervisor, never a model/source knob.
    artifact_path: Option<PathBuf>,
}
#[derive(Clone, Debug)]
pub struct RunProbe {
    owner: TrainingProbe,
    index: usize,
}
impl TrainingProbe {
    pub fn new(evaluation: EvaluationManifest) -> Self {
        Self {
            evaluation,
            receipts: Arc::new(Mutex::new(Vec::new())),
            artifact_path: std::env::var_os("MOSS_CPU_RECEIPT_PATH").map(PathBuf::from),
        }
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
        let bytes = serde_json::to_vec(&serde_json::json!({
            "version":"moss-cpu-training-observations-v1",
            "source_head":std::env::var("GITHUB_SHA").ok(),
            "evaluation":{
                "identity":&self.evaluation.identity,
                "width":self.evaluation.width,"height":self.evaluation.height,
                "noise_seed":self.evaluation.noise_seed,"sigma":self.evaluation.sigma,
                "caption":&self.evaluation.item.caption,
                "image_sha256":self.evaluation.image_digest,
                "mask_sha256":self.evaluation.mask_digest,
            },
            "receipts":receipts,
            // Hash-observed changes, not an invented numeric norm of the parameter delta.
            "parameter_hash_changed":parameter_delta,
        }))
        .map_err(|e| Error::Msg(e.to_string()))?;
        let staging = path.with_extension("receipts.partial");
        let mut file = std::fs::OpenOptions::new()
            .write(true)
            .create_new(true)
            .open(&staging)?;
        let result = (|| -> Result<()> {
            file.write_all(&bytes)?;
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
        assert!(before.is_finite() && after.is_finite());
        assert!(
            after < before,
            "fixed independent heldout objective did not decrease: {before} → {after}"
        );
    }
    if masks_required {
        assert_eq!(full.masks.get("64x32"), Some(&vec![1., 1., 0.25, 0.25]));
        assert_eq!(full.masks.get("32x64"), Some(&vec![1., 0.25]));
        assert_eq!(full.masks, resumed.masks);
    }
    assert_eq!(full.evaluation_manifest, resumed.evaluation_manifest);
    assert_eq!(full.factor_after, resumed.factor_after);
    assert_eq!(
        full.evaluation_after.map(f32::to_bits),
        resumed.evaluation_after.map(f32::to_bits),
        "evaluation must not advance the original training RNG/cursor/optimizer"
    );
    Ok(())
}
