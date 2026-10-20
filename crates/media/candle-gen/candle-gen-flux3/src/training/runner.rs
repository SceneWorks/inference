//! Real eager Candle session behind the shared source controller. Only declared
//! LoRA Var leaves reach the optimizer; AE/teacher caches are frozen CPU tensors.
use super::{
    checkpoint::{self, Snapshot},
    TrainingGraph,
};
use crate::SampleTokens;
use candle_gen::candle_core::{backprop::GradStore, DType, Device, Tensor};
use candle_gen::gen_core::flux3::{
    latent_ids, scaled_image_ids,
    training::{
        caption_choice,
        checkpoint::{ResumePolicy, TrainingCursor},
        controller::{checkpoint_name, output_stem, ExportReceipt, SourceTrainingSession},
        density_index,
        options::{TrainingOptions, Workflow},
        policy::source_clip_multiplier,
        rng::NativeRng,
        training_condition_ids, training_sigma, CaptionChoice, ResumeIdentity,
    },
    IMAGE_ROPE_SCALE,
};
use candle_gen::gen_core::train::TrainingRequest;
use candle_gen::gen_core::{self, AdapterKind, AdapterSpec, CancelFlag};
use candle_gen::train::optim::{accumulate_grads, scale_grads, TrainOptimizer};
use candle_gen::{CandleError as Error, Result};
use std::{
    collections::{BTreeMap, BTreeSet},
    path::Path,
};

pub(crate) struct CachedExample {
    pub latent: Tensor,
    pub text: Tensor,
    pub text_ids: Vec<[f32; 4]>,
    pub empty_text: Option<Tensor>,
    pub reference: Option<Tensor>,
    pub grid: (usize, usize),
}
pub(crate) struct CandleSession {
    pub graph: TrainingGraph,
    pub optimizer: TrainOptimizer,
    pub examples: Vec<CachedExample>,
    pub options: TrainingOptions,
    pub identity: ResumeIdentity,
    pub seed: u64,
    pub accumulation: u32,
    pending: Option<GradStore>,
}
fn invalid(detail: impl std::fmt::Display) -> Error {
    Error::Unsupported(format!("flux3 training session: {detail}"))
}
fn check(flag: &CancelFlag) -> Result<()> {
    if flag.is_cancelled() {
        Err(Error::Canceled)
    } else {
        Ok(())
    }
}
impl CandleSession {
    pub fn new(
        graph: TrainingGraph,
        optimizer: TrainOptimizer,
        examples: Vec<CachedExample>,
        options: TrainingOptions,
        identity: ResumeIdentity,
        seed: u64,
        accumulation: u32,
    ) -> Result<Self> {
        if examples.is_empty() || accumulation == 0 {
            return Err(invalid("empty prepared cache or invalid accumulation"));
        }
        for example in &examples {
            let count = example
                .grid
                .0
                .checked_mul(example.grid.1)
                .ok_or_else(|| invalid("cache grid overflow"))?;
            if example.latent.dtype() != DType::F32
                || example.latent.dims() != [count, graph.dit.config().image_in_channels]
                || example.text.dim(1)? != graph.dit.config().text_in_channels
                || example.text.dim(0)? != example.text_ids.len()
                || (example.reference.is_some()) != (options.workflow == Workflow::PairedEdit)
            {
                return Err(invalid("prepared cache geometry/workflow/dtype mismatch"));
            }
            if let Some(reference) = &example.reference {
                if reference.dtype() != DType::F32 || reference.dims() != example.latent.dims() {
                    return Err(invalid("paired conditioning must share target geometry and normalized F32 latent dtype"));
                }
            }
            if options.caption_dropout > 0.0
                && example.empty_text.as_ref().is_none_or(|empty| {
                    empty.dims() != example.text.dims() || empty.dtype() != example.text.dtype()
                })
            {
                return Err(invalid(
                    "empty-caption cache must match each original window and selected dtype",
                ));
            }
        }
        Ok(Self {
            graph,
            optimizer,
            examples,
            options,
            identity,
            seed,
            accumulation,
            pending: None,
        })
    }
    fn named_pending(&self) -> BTreeMap<String, Tensor> {
        let Some(pending) = &self.pending else {
            return BTreeMap::new();
        };
        self.graph
            .factors
            .iter()
            .flat_map(|f| {
                [
                    (format!("{}.lora_A.weight", f.target.native), &f.down),
                    (format!("{}.lora_B.weight", f.target.native), &f.up),
                ]
            })
            .filter_map(|(name, var)| {
                pending
                    .get(var.as_tensor())
                    .map(|tensor| (name, tensor.clone()))
            })
            .collect()
    }
    fn export_identity(&self, cursor: &TrainingCursor) -> ResumeIdentity {
        let mut identity = self.identity.clone();
        identity.configuration.insert(
            "completed_micro_steps".into(),
            cursor.micro_step.to_string(),
        );
        if let Some(bits) = cursor.last_loss_bits {
            identity
                .configuration
                .insert("observed_last_loss_bits".into(), format!("{bits:08x}"));
        }
        identity
    }
    fn existing_export_matches(&self, path: &Path, identity: &ResumeIdentity) -> Result<()> {
        let spec = AdapterSpec::new(path.to_path_buf(), 1.0, AdapterKind::Lora);
        let plan = gen_core::flux3::adapters::inspect(&spec, self.graph.dit.config())
            .map_err(Error::from)?;
        if plan.len() != self.graph.factors.len()
            || plan
                .iter()
                .any(|p| (p.scale as f32).to_bits() != (self.graph.scale as f32).to_bits())
        {
            return Err(invalid(
                "existing output is not this trained adapter's target/gain surface",
            ));
        }
        let bytes =
            std::fs::read(path).map_err(|e| invalid(format!("read existing output: {e}")))?;
        let (_, header) =
            safetensors::SafeTensors::read_metadata(&bytes).map_err(|e| invalid(e.to_string()))?;
        if header
            .metadata()
            .as_ref()
            .and_then(|m| m.get("training_identity"))
            != Some(&identity.to_metadata().map_err(Error::from)?)
        {
            return Err(invalid(
                "existing output belongs to another training identity/progress; refusing overwrite",
            ));
        }
        let tensors = candle_gen::candle_core::safetensors::load_buffer(&bytes, &Device::Cpu)?;
        let mut expected = BTreeSet::new();
        for factor in &self.graph.factors {
            for (suffix, var) in [
                ("lora_A.weight", &factor.down),
                ("lora_B.weight", &factor.up),
            ] {
                let name = format!("{}.{}", factor.target.diffusers, suffix);
                expected.insert(name.clone());
                let saved = tensors
                    .get(&name)
                    .ok_or_else(|| invalid("existing output is missing a trained factor"))?;
                if saved.dtype() != DType::F32 || saved.dims() != var.dims() {
                    return Err(invalid("existing output shape/master precision differs"));
                }
                let actual = saved.flatten_all()?.to_vec1::<f32>()?;
                let wanted = var
                    .as_tensor()
                    .to_device(&Device::Cpu)?
                    .flatten_all()?
                    .to_vec1::<f32>()?;
                if actual
                    .iter()
                    .zip(wanted)
                    .any(|(a, b)| a.to_bits() != b.to_bits())
                {
                    return Err(invalid(
                        "existing output factor content differs; refusing overwrite",
                    ));
                }
            }
        }
        if tensors.keys().cloned().collect::<BTreeSet<_>>() != expected {
            return Err(invalid("existing output contains undeclared factors"));
        }
        Ok(())
    }
}
impl SourceTrainingSession for CandleSession {
    fn identity(&self) -> &ResumeIdentity {
        &self.identity
    }
    fn resume_format(&self) -> &'static str {
        "flux3-native-training-resume-v1"
    }
    fn restore(
        &mut self,
        path: &Path,
        policy: ResumePolicy,
        flag: &CancelFlag,
    ) -> gen_core::Result<TrainingCursor> {
        let state = checkpoint::restore(
            &self.graph,
            &mut self.optimizer,
            &self.identity,
            policy,
            path,
            flag,
        )
        .map_err(gen_core::Error::from)?;
        let mut pending = GradStore::default();
        for factor in &self.graph.factors {
            for (suffix, var) in [
                ("lora_A.weight", &factor.down),
                ("lora_B.weight", &factor.up),
            ] {
                if let Some(gradient) = state
                    .pending
                    .get(&format!("{}.{}", factor.target.native, suffix))
                {
                    pending.insert(var.as_tensor(), gradient.clone());
                }
            }
        }
        self.pending = (!state.pending.is_empty()).then_some(pending);
        Ok(state.cursor)
    }
    fn micro_batch(
        &mut self,
        indices: &[usize],
        micro_step: u64,
        flag: &CancelFlag,
    ) -> gen_core::Result<f32> {
        let result = (|| -> Result<f32> {
            if indices.is_empty() {
                return Err(invalid("empty micro-batch"));
            }
            let vars = self.graph.vars();
            let mut raw_loss = 0.0f32;
            for (position, index) in indices.iter().enumerate() {
                check(flag)?;
                let example = self
                    .examples
                    .get(*index)
                    .ok_or_else(|| invalid("sampler selected unknown cache item"))?;
                let stream = micro_step
                    .checked_mul(indices.len() as u64)
                    .and_then(|n| n.checked_add(position as u64))
                    .ok_or_else(|| invalid("RNG stream index overflow"))?;
                let mut density = NativeRng::keyed(self.seed, "timestep-density", stream);
                let uniform = density.uniform();
                let normal = if self.options.weighting
                    == gen_core::flux3::training::Weighting::LogitNormal
                {
                    density.normal()
                } else {
                    0.0
                };
                let u = self
                    .options
                    .weighting
                    .density(
                        uniform,
                        normal,
                        self.options.logit_mean,
                        self.options.logit_std,
                        self.options.mode_scale,
                    )
                    .map_err(Error::from)?;
                let sigma =
                    training_sigma(density_index(u).map_err(Error::from)?).map_err(Error::from)?;
                let mut noise = NativeRng::keyed(self.seed, "training-normal-noise", stream);
                let data = (0..example.latent.elem_count())
                    .map(|_| noise.normal())
                    .collect::<Vec<_>>();
                let noise = Tensor::from_vec(data, example.latent.shape(), &Device::Cpu)?
                    .to_device(self.graph.dit.device())?;
                let clean = example.latent.to_device(self.graph.dit.device())?;
                let image = ((clean.clone() * (1.0 - sigma) as f64)? + (&noise * sigma as f64)?)?;
                let target = noise.sub(&clean)?;
                let mut caption = NativeRng::keyed(self.seed, "caption-dropout", stream);
                let choice = caption_choice(
                    caption.uniform(),
                    self.options.caption_dropout,
                    false,
                    example.text_ids.len(),
                )
                .map_err(Error::from)?;
                let text = match choice {
                    CaptionChoice::Original => &example.text,
                    CaptionChoice::EmptyAtWindow(_) => example
                        .empty_text
                        .as_ref()
                        .ok_or_else(|| invalid("missing matched-window empty caption"))?,
                }
                .to_device(self.graph.dit.device())?;
                let reference = example
                    .reference
                    .as_ref()
                    .map(|r| r.to_device(self.graph.dit.device()))
                    .transpose()?;
                let image_ids = scaled_image_ids(
                    &latent_ids(example.grid.0, example.grid.1),
                    IMAGE_ROPE_SCALE,
                );
                let cond_ids =
                    training_condition_ids(example.grid.0, example.grid.1).map_err(Error::from)?;
                let samples = [SampleTokens {
                    text: &text,
                    text_ids: &example.text_ids,
                    image: &image,
                    image_ids: &image_ids,
                    image_cond: reference.as_ref().map(|r| (r, cond_ids.as_slice())),
                    timestep: sigma,
                    target_timestep: None,
                }];
                let weight = self
                    .options
                    .weighting
                    .loss_weight(sigma)
                    .map_err(Error::from)?;
                let loss = self.graph.loss(&samples, &target, weight as f64)?;
                // Source Accelerator.backward: divide by CONFIGURED N, including short tails.
                let scale =
                    gen_core::flux3::training::policy::accumulation_loss_scale(self.accumulation)
                        .map_err(Error::from)?
                        / indices.len() as f32;
                let gradients = (&loss * scale as f64)?.backward()?;
                raw_loss += loss.to_scalar::<f32>()? / indices.len() as f32;
                accumulate_grads(&mut self.pending, gradients, &vars)?;
            }
            check(flag)?;
            Ok(raw_loss)
        })();
        result.map_err(Into::into)
    }
    fn optimizer_update(
        &mut self,
        lr_multiplier: f32,
        max_grad_norm: f32,
        flag: &CancelFlag,
    ) -> gen_core::Result<()> {
        let result = (|| -> Result<()> {
            check(flag)?;
            let vars = self.graph.vars();
            let pending = self
                .pending
                .as_mut()
                .ok_or_else(|| invalid("optimizer update has no pending native gradients"))?;
            let norms = vars
                .iter()
                .filter_map(|v| pending.get(v.as_tensor()))
                .map(|g| Ok(g.sqr()?.sum_all()?.sqrt()?))
                .collect::<Result<Vec<_>>>()?;
            if norms.is_empty() {
                return Err(invalid(
                    "no declared adapter target affects this workflow; refusing simulated training",
                ));
            }
            let norm = Tensor::stack(&norms, 0)?
                .sqr()?
                .sum_all()?
                .sqrt()?
                .to_scalar::<f32>()?;
            let coefficient = source_clip_multiplier(norm, max_grad_norm).map_err(Error::from)?;
            if coefficient != 1.0 {
                scale_grads(pending, &vars, coefficient as f64)?;
            }
            self.optimizer.set_lr_scaled(lr_multiplier);
            self.optimizer.step(pending)?;
            self.pending = None;
            check(flag)
        })();
        result.map_err(Into::into)
    }
    fn checkpoint(
        &mut self,
        req: &TrainingRequest,
        cursor: &TrainingCursor,
        policy: ResumePolicy,
        flag: &CancelFlag,
    ) -> gen_core::Result<()> {
        let result = (|| -> Result<()> {
            let pending = self.named_pending();
            let stem = output_stem(&req.file_name).map_err(Error::from)?;
            checkpoint::save(
                &self.graph,
                &self.optimizer,
                Snapshot {
                    identity: &self.identity,
                    cursor,
                    policy,
                    pending: &pending,
                },
                &req.output_dir,
                &checkpoint_name(stem, cursor.micro_step, true),
                flag,
            )?;
            checkpoint::export_adapter(
                &self.graph,
                &self.export_identity(cursor),
                &req.output_dir,
                &checkpoint_name(stem, cursor.micro_step, false),
                flag,
            )?;
            check(flag)
        })();
        result.map_err(Into::into)
    }
    fn export(
        &mut self,
        req: &TrainingRequest,
        cursor: &TrainingCursor,
        flag: &CancelFlag,
    ) -> gen_core::Result<ExportReceipt> {
        let result = (|| -> Result<ExportReceipt> {
            check(flag)?;
            let path = req.output_dir.join(&req.file_name);
            let identity = self.export_identity(cursor);
            if path.exists() {
                self.existing_export_matches(&path, &identity)?;
                return Ok(ExportReceipt {
                    path,
                    created: false,
                });
            }
            let path = checkpoint::export_adapter(
                &self.graph,
                &identity,
                &req.output_dir,
                &req.file_name,
                flag,
            )?;
            Ok(ExportReceipt {
                path,
                created: true,
            })
        })();
        result.map_err(Into::into)
    }
    fn discard_created_export(&self, receipt: &ExportReceipt) -> gen_core::Result<()> {
        if receipt.created {
            std::fs::remove_file(&receipt.path)?;
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    //! Real native loop on SYNTHETIC tiny weights/cache: actual backward/AdamW,
    //! interrupted pending-gradient checkpoint, strict resume and generation reload.
    use super::*;
    use crate::Flux3Dit;
    use candle_gen::gen_core::flux3::training::{
        controller::{run, RunPlan},
        options::TrainingPrecision,
        policy::SourceLrKind,
        sampler::BatchSampler,
        Weighting,
    };
    use candle_gen::gen_core::train::{TrainingConfig, TrainingItem, TrainingProgress};
    use std::path::PathBuf;
    fn fixtures() -> PathBuf {
        Path::new(env!("CARGO_MANIFEST_DIR")).join("../../mlx-gen/mlx-gen-flux3/tests/fixtures")
    }
    fn ids(t: &Tensor) -> Vec<[f32; 4]> {
        t.flatten_all()
            .unwrap()
            .to_vec1::<f32>()
            .unwrap()
            .chunks_exact(4)
            .map(|v| [v[0], v[1], v[2], v[3]])
            .collect()
    }
    fn options() -> TrainingOptions {
        TrainingOptions {
            precision: TrainingPrecision::F32,
            workflow: Workflow::PairedEdit,
            caption_dropout: 0.5,
            max_sequence_length: 512,
            center_crop: true,
            random_flip: false,
            weighting: Weighting::LogitNormal,
            logit_mean: 0.0,
            logit_std: 1.0,
            mode_scale: 1.29,
            max_grad_norm: 0.05,
            scheduler: SourceLrKind::Cosine,
            cycles: 1,
            power: 1.0,
        }
    }
    fn session() -> (CandleSession, gen_core::flux3::Flux3Config) {
        session_for(Workflow::PairedEdit)
    }
    fn session_for(workflow: Workflow) -> (CandleSession, gen_core::flux3::Flux3Config) {
        let config = gen_core::flux3::Flux3Config::parse(
            &std::fs::read_to_string(fixtures().join("tiny-base-snapshot/config.json")).unwrap(),
        )
        .unwrap();
        let dit = Flux3Dit::load(
            &fixtures().join("tiny-base-snapshot/model.safetensors"),
            &config.dit,
            DType::F32,
            &Device::Cpu,
        )
        .unwrap();
        let golden = candle_gen::candle_core::safetensors::load(
            fixtures().join("flux3_training_golden.safetensors"),
            &Device::Cpu,
        )
        .unwrap();
        let targets = gen_core::flux3::training::lora_targets(&config.dit, &[]).unwrap();
        let initial = targets
            .iter()
            .map(|t| {
                (
                    t.native.clone(),
                    golden[&format!("initial/{}.lora_A.weight", t.native)].clone(),
                )
            })
            .collect();
        let graph = TrainingGraph::new(dit, 3, 1.5, &[], initial).unwrap();
        let optimizer = TrainOptimizer::from_config("adamw", graph.vars(), 0.0003, 0.0001).unwrap();
        let image_ids = ids(&golden["image_ids"]);
        let h = image_ids
            .iter()
            .map(|i| (i[1] * 2.0) as usize)
            .max()
            .unwrap()
            + 1;
        let w = image_ids
            .iter()
            .map(|i| (i[2] * 2.0) as usize)
            .max()
            .unwrap()
            + 1;
        let text = golden["text"].clone();
        let text_ids = ids(&golden["text_ids"]);
        let examples = (0..3)
            .map(|i| CachedExample {
                latent: (&golden["clean"] * (1.0 + 0.1 * i as f64)).unwrap(),
                text: text.clone(),
                text_ids: text_ids.clone(),
                empty_text: Some((&text * 0.5).unwrap()),
                reference: (workflow == Workflow::PairedEdit).then(|| golden["reference"].clone()),
                grid: (h, w),
            })
            .collect();
        let mut factors = BTreeMap::new();
        for f in &graph.factors {
            factors.insert(
                format!("{}.lora_A.weight", f.target.native),
                f.down.dims().to_vec(),
            );
            factors.insert(
                format!("{}.lora_B.weight", f.target.native),
                f.up.dims().to_vec(),
            );
        }
        let identity = ResumeIdentity {
            base_repo: gen_core::flux3::BASE_WEIGHTS_REPO.into(),
            base_revision: gen_core::flux3::BASE_WEIGHTS_REVISION.into(),
            artifacts: BTreeMap::from([("synthetic-tiny-base".into(), "a".repeat(64))]),
            dataset: "b".repeat(64),
            preprocessing: "c".repeat(64),
            factors,
            configuration: BTreeMap::from([
                ("optimizer".into(), "adamw".into()),
                ("learning_rate".into(), "0.0003".into()),
                ("rank".into(), "3".into()),
                ("alpha".into(), "1.5".into()),
                ("scaling".into(), "alpha/rank".into()),
                ("precision".into(), "f32".into()),
                ("backend".into(), "candle-cpu".into()),
                ("scheduler".into(), "cosine".into()),
                ("accumulation".into(), "2".into()),
                (
                    "rng".into(),
                    gen_core::flux3::training::rng::RNG_ALGORITHM.into(),
                ),
                (
                    "fixture".into(),
                    "SYNTHETIC tiny weights, NOT downloaded BFL weights".into(),
                ),
            ]),
        };
        (
            CandleSession::new(
                graph,
                optimizer,
                examples,
                TrainingOptions {
                    workflow,
                    ..options()
                },
                identity,
                73,
                2,
            )
            .unwrap(),
            config,
        )
    }
    fn request(dir: &Path, resume: bool) -> TrainingRequest {
        TrainingRequest {
            items: (0..3)
                .map(|i| TrainingItem::captioned(dir.join(format!("{i}.png")), "synthetic".into()))
                .collect(),
            config: TrainingConfig {
                rank: 3,
                alpha: 1.5,
                learning_rate: 0.0003,
                weight_decay: 0.0001,
                steps: 5,
                batch_size: 1,
                gradient_accumulation: 2,
                save_every: 1,
                seed: 73,
                resume,
                lr_warmup_steps: 1,
                ..TrainingConfig::default()
            },
            output_dir: dir.to_path_buf(),
            file_name: "trained.safetensors".into(),
            trigger_words: vec![],
            cancel: CancelFlag::new(),
        }
    }
    fn plan(req: &TrainingRequest) -> RunPlan {
        let sampler = BatchSampler::new(&[0, 0, 0], 1, 73).unwrap();
        RunPlan {
            schedule: options()
                .schedule(&req.config, sampler.len() as u64)
                .unwrap(),
            sampler,
            max_grad_norm: 0.05,
        }
    }
    fn bits(path: &Path) -> BTreeMap<String, Vec<u32>> {
        candle_gen::candle_core::safetensors::load(path, &Device::Cpu)
            .unwrap()
            .into_iter()
            .map(|(k, t)| {
                (
                    k,
                    t.flatten_all()
                        .unwrap()
                        .to_vec1::<f32>()
                        .unwrap()
                        .into_iter()
                        .map(f32::to_bits)
                        .collect(),
                )
            })
            .collect()
    }
    #[test]
    fn real_loop_interrupts_with_pending_gradient_resumes_identically_and_reloads() {
        let continuous_dir = tempfile::tempdir().unwrap();
        let (mut continuous, config) = session();
        let req = request(continuous_dir.path(), false);
        let mut losses = Vec::new();
        let output = run(&mut continuous, &req, &plan(&req), &mut |event| {
            if let TrainingProgress::Training { loss, .. } = event {
                losses.push(loss)
            }
        })
        .unwrap();
        assert_eq!(output.steps, 5);
        assert_eq!(losses.len(), 5);
        assert!(losses.iter().all(|l| l.is_finite() && *l > 0.0));
        let interrupted_dir = tempfile::tempdir().unwrap();
        let (mut first, _) = session();
        let req = request(interrupted_dir.path(), false);
        let cancel = req.cancel.clone();
        let error = run(&mut first, &req, &plan(&req), &mut |event| {
            if event == (TrainingProgress::Checkpoint { step: 4 }) {
                cancel.cancel()
            }
        })
        .unwrap_err();
        assert!(matches!(error, gen_core::Error::Canceled));
        assert!(
            !interrupted_dir.path().join("trained.safetensors").exists(),
            "cancel must not publish success"
        );
        let (mut resumed, _) = session();
        let req = request(interrupted_dir.path(), true);
        let resumed_output = run(&mut resumed, &req, &plan(&req), &mut |_| {}).unwrap();
        assert_eq!(resumed_output.steps, 1);
        assert_eq!(
            resumed_output.final_loss.to_bits(),
            output.final_loss.to_bits()
        );
        assert_eq!(
            bits(&interrupted_dir.path().join("trained.safetensors")),
            bits(&output.adapter_path),
            "step4 pending gradient + epoch-tail + LR/RNG cursor resume must be exact"
        );
        let spec = gen_core::AdapterSpec::new(
            output.adapter_path.clone(),
            1.0,
            gen_core::AdapterKind::Lora,
        );
        assert_eq!(
            gen_core::flux3::adapters::inspect(&spec, &config.dit)
                .unwrap()
                .len(),
            54
        );
        let golden = candle_gen::candle_core::safetensors::load(
            fixtures().join("flux3_training_golden.safetensors"),
            &Device::Cpu,
        )
        .unwrap();
        let text_ids = ids(&golden["text_ids"]);
        let image_ids = ids(&golden["image_ids"]);
        let ref_ids = ids(&golden["reference_ids"]);
        let sample = [SampleTokens {
            text: &golden["text"],
            text_ids: &text_ids,
            image: &golden["noise"],
            image_ids: &image_ids,
            image_cond: Some((&golden["reference"], ref_ids.as_slice())),
            timestep: 0.6,
            target_timestep: None,
        }];
        let load = || {
            Flux3Dit::load(
                &fixtures().join("tiny-base-snapshot/model.safetensors"),
                &config.dit,
                DType::F32,
                &Device::Cpu,
            )
            .unwrap()
        };
        let mut on = load();
        let bare = load().forward(&sample).unwrap();
        crate::adapters::install(&mut on, std::slice::from_ref(&spec)).unwrap();
        assert!(
            on.forward(&sample)
                .unwrap()
                .sub(&bare)
                .unwrap()
                .abs()
                .unwrap()
                .max_all()
                .unwrap()
                .to_scalar::<f32>()
                .unwrap()
                > 1e-7
        );
        let mut zero = load();
        let mut off = spec;
        off.scale = 0.0;
        crate::adapters::install(&mut zero, &[off]).unwrap();
        assert_eq!(
            zero.forward(&sample)
                .unwrap()
                .flatten_all()
                .unwrap()
                .to_vec1::<f32>()
                .unwrap(),
            bare.flatten_all().unwrap().to_vec1::<f32>().unwrap()
        );
        let req = request(continuous_dir.path(), false);
        assert!(
            gen_core::flux3::training::controller::ensure_new_output(&req).is_err(),
            "a completed run must not be silently overwritten"
        );
    }
    fn factor_bits(session: &CandleSession) -> BTreeMap<String, Vec<u32>> {
        session
            .graph
            .factors
            .iter()
            .flat_map(|f| {
                [
                    (format!("{}.lora_A.weight", f.target.native), &f.down),
                    (format!("{}.lora_B.weight", f.target.native), &f.up),
                ]
            })
            .map(|(name, var)| {
                let values = var
                    .as_tensor()
                    .flatten_all()
                    .unwrap()
                    .to_vec1::<f32>()
                    .unwrap();
                (name, values.into_iter().map(f32::to_bits).collect())
            })
            .collect()
    }
    /// PR #1288 review: the source (PyTorch) never steps or decays a parameter whose `.grad` is
    /// `None`. Candle's autograd is the ground truth for which factors a workflow reaches; this
    /// pins that the shared `Workflow::trains` predicate (which the MLX twin must apply, because
    /// its functional gradient returns zeros) is EXACTLY that partition, and that an unreached
    /// factor is left bit-identical by the update (no moment, no decoupled weight decay).
    #[test]
    fn each_workflow_steps_exactly_the_factors_its_native_graph_reaches() {
        for workflow in [Workflow::Captioned, Workflow::PairedEdit] {
            let (mut session, _) = session_for(workflow);
            let flag = CancelFlag::new();
            let before = factor_bits(&session);
            session.micro_batch(&[0], 0, &flag).unwrap();
            let reached: BTreeSet<String> = session.named_pending().into_keys().collect();
            let declared: BTreeSet<String> = before
                .keys()
                .filter(|name| workflow.trains(name))
                .cloned()
                .collect();
            assert_eq!(
                reached, declared,
                "{workflow:?}: Workflow::trains must equal the native autograd partition"
            );
            let unreached = before.len() - declared.len();
            match workflow {
                Workflow::Captioned => assert!(
                    unreached > 0,
                    "a captioned run must leave the reference-only factors unreached"
                ),
                Workflow::PairedEdit => assert_eq!(unreached, 0),
            }
            session.optimizer_update(1.0, 0.05, &flag).unwrap();
            let after = factor_bits(&session);
            for (name, bits) in &before {
                if !workflow.trains(name) {
                    assert_eq!(
                        &after[name], bits,
                        "{workflow:?}: unreached {name} must be neither stepped nor decayed"
                    );
                } else if name.ends_with(".lora_B.weight") {
                    assert_ne!(
                        &after[name], bits,
                        "{workflow:?}: reached {name} must update"
                    );
                }
            }
        }
    }
    /// PR #1288 review: a TRAINED export (not a hand-made fixture) loads through the public
    /// provider path SceneWorks uses, `LoadSpec.adapters` -> `load("flux3_klein_base")` ->
    /// `generate`, applies on every trained projection, changes the image, and is inert at 0.
    #[test]
    fn trained_export_loads_through_the_public_base_provider_and_changes_generation() {
        use gen_core::flux3::{AUTOENCODER_COMPONENT, TEXT_ENCODER_COMPONENT};
        use gen_core::{
            GenerationOutput, GenerationRequest, Generator, LoadSpec, Precision, WeightsSource,
        };
        let dir = tempfile::tempdir().unwrap();
        let (mut session, _) = session();
        let req = request(dir.path(), false);
        let output = run(&mut session, &req, &plan(&req), &mut |_| {}).unwrap();
        let registry = crate::provider_registry().unwrap();
        let spec_with = |adapters: Vec<AdapterSpec>| {
            let mut spec = LoadSpec::new(WeightsSource::Dir(fixtures().join("tiny-base-snapshot")));
            spec.precision = Precision::Fp32;
            spec.components.insert(
                TEXT_ENCODER_COMPONENT.into(),
                WeightsSource::Dir(fixtures().join("tiny-snapshot/text_encoder")),
            );
            spec.components.insert(
                AUTOENCODER_COMPONENT.into(),
                WeightsSource::File(fixtures().join("tiny-snapshot/ae.safetensors")),
            );
            spec.adapters = adapters;
            spec
        };
        let generation = GenerationRequest {
            prompt: "a red fox in the snow".into(),
            width: 80,
            height: 64,
            steps: Some(3),
            seed: Some(42),
            ..Default::default()
        };
        let render = |generator: &dyn Generator| -> Vec<u8> {
            generator.validate(&generation).unwrap();
            let GenerationOutput::Images(images) =
                generator.generate(&generation, &mut |_| {}).unwrap()
            else {
                panic!("image output required")
            };
            assert_eq!((images[0].width, images[0].height), (80, 64));
            images[0].pixels.clone()
        };
        let bare = render(
            registry
                .load("flux3_klein_base", &spec_with(vec![]))
                .unwrap()
                .as_ref(),
        );
        let trained = AdapterSpec::new(output.adapter_path.clone(), 1.0, AdapterKind::Lora);
        let adapted = registry
            .load("flux3_klein_base", &spec_with(vec![trained.clone()]))
            .unwrap();
        assert_eq!(adapted.descriptor().id, "flux3_klein_base");
        let reports = adapted.adapter_apply_reports();
        assert_eq!(reports.len(), 1);
        assert_eq!(reports[0].applied, 54, "every trained projection applies");
        assert!(reports[0].skipped.is_empty());
        assert_ne!(
            render(adapted.as_ref()),
            bare,
            "the trained adapter must change the generated image"
        );
        let mut inert = trained;
        inert.scale = 0.0;
        let zero = registry
            .load("flux3_klein_base", &spec_with(vec![inert]))
            .unwrap();
        assert_eq!(render(zero.as_ref()), bare, "scale 0 must be bit-inert");
    }
}
