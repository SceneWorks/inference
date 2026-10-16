//! Registered MLX FLUX3 base LoRA trainer: the same shared source controller,
//! identities, pixel recipe and checkpoint semantics as the Candle trainer.
//! Supported-host execution only; Windows can edit/format but not build MLX.
use super::{
    checkpoint::{self, Snapshot},
    optim::SourceAdamW,
    TrainingGraph,
};
use crate::autoencoder::FrozenTrainingEncoder;
use crate::{Flux3TextEncoder, SampleTokens};
use gen_core::flux3::training::{
    caption_choice,
    checkpoint::{ResumePolicy, TrainingCursor},
    controller::{
        self, checkpoint_name, output_stem, ExportReceipt, RunPlan, SourceTrainingSession,
    },
    density_index, identity, lora_targets,
    options::{TrainingOptions, TrainingPrecision, Workflow},
    policy::{accumulation_loss_scale, source_clip_multiplier},
    preprocess::{self, PixelOptions},
    rng::NativeRng,
    sampler::BatchSampler,
    training_condition_ids, training_sigma, CaptionChoice, ResumeIdentity, Weighting,
};
use gen_core::flux3::{
    assemble_window, latent_ids, scaled_image_ids, Flux3Config, GenerationResources,
    AUTOENCODER_COMPONENT, BASE_MODEL_ID, CONFIG_FILE, DIT_WEIGHTS_FILE, FAMILY, IMAGE_ROPE_SCALE,
    TEXT_ENCODER_COMPONENT,
};
use gen_core::train::{
    Trainer, TrainerDescriptor, TrainingOutput, TrainingProgress, TrainingRequest,
    TrainingTechniques,
};
use mlx_gen::gen_core::{
    self, AdapterKind, AdapterSpec, CancelFlag, LoadSpec, Modality, Precision, WeightsSource,
};
use mlx_gen::{train::lora::LoraParams, weights::Weights, Error, Result};
use mlx_rs::{transforms::eval, Array, Dtype};
use std::{collections::BTreeMap, path::Path};
fn invalid(detail: impl std::fmt::Display) -> Error {
    Error::Unsupported(format!("flux3 trainer: {detail}"))
}
fn check(flag: &CancelFlag) -> Result<()> {
    if flag.is_cancelled() {
        Err(Error::Canceled)
    } else {
        Ok(())
    }
}
pub fn trainer_descriptor() -> TrainerDescriptor {
    TrainerDescriptor {
        id: BASE_MODEL_ID,
        family: FAMILY,
        backend: "mlx",
        modality: Modality::Image,
        supports_lora: true,
        supports_lokr: false,
        supports_control: false,
        supports_full_finetune: false,
        max_reference_images: 1,
        techniques: TrainingTechniques::NONE,
    }
}
pub struct Flux3Trainer {
    spec: LoadSpec,
    descriptor: TrainerDescriptor,
}
pub fn load_trainer(spec: &LoadSpec) -> Result<Box<dyn Trainer>> {
    gen_core::train::refuse_trainer_load_overlays(BASE_MODEL_ID, spec).map_err(Error::from)?;
    if spec.load_shape != gen_core::LoadShape::EagerMaterialization
        || !matches!(spec.weights, WeightsSource::Dir(_))
        || spec.quantize.is_some()
        || !spec.adapters.is_empty()
        || spec.pid.is_some()
        || spec.text_encoder.is_some()
    {
        return Err(invalid("requires the dense pinned base directory and no quantization/overlaid adapters/auxiliary graph"));
    }
    gen_core::control::reject_unknown_components(
        spec,
        &[TEXT_ENCODER_COMPONENT, AUTOENCODER_COMPONENT],
        BASE_MODEL_ID,
    )
    .map_err(Error::from)?;
    Ok(Box::new(Flux3Trainer {
        spec: spec.clone(),
        descriptor: trainer_descriptor(),
    }))
}
mlx_gen::register_trainer! {pub(crate) const TRAINER_REGISTRATION = trainer_descriptor => load_trainer}
impl Trainer for Flux3Trainer {
    fn descriptor(&self) -> &TrainerDescriptor {
        &self.descriptor
    }
    fn validate(&self, req: &TrainingRequest) -> gen_core::Result<()> {
        gen_core::train::validate_full_finetune_request(self.descriptor(), req)?;
        gen_core::train::validate_control_request(self.descriptor(), req)?;
        gen_core::train::validate_training_techniques(self.descriptor(), req)?;
        gen_core::train::validate_edit_request(self.descriptor(), req)?;
        let opts = TrainingOptions::parse(req)?;
        controller::ensure_new_output(req)?;
        if self.spec.precision == Precision::Fp32 && opts.precision != TrainingPrecision::F32 {
            return Err(gen_core::Error::Unsupported(
                "explicit Fp32 LoadSpec and selected training precision disagree".into(),
            ));
        }
        Ok(())
    }
    fn train(
        &mut self,
        req: &TrainingRequest,
        on_progress: &mut dyn FnMut(TrainingProgress),
    ) -> gen_core::Result<TrainingOutput> {
        self.validate(req)?;
        execute(&self.spec, req, on_progress).map_err(Into::into)
    }
}

pub(crate) struct CachedExample {
    pub latent: Array,
    pub text: Array,
    pub text_ids: Vec<[f32; 4]>,
    pub empty_text: Option<Array>,
    pub reference: Option<Array>,
    pub grid: (i32, i32),
}
pub(crate) struct MlxSession {
    pub graph: TrainingGraph,
    pub params: LoraParams,
    pub optimizer: SourceAdamW,
    pub examples: Vec<CachedExample>,
    pub options: TrainingOptions,
    pub identity: ResumeIdentity,
    pub seed: u64,
    pub accumulation: u32,
    pending: LoraParams,
}
impl MlxSession {
    pub fn new(
        graph: TrainingGraph,
        params: LoraParams,
        optimizer: SourceAdamW,
        examples: Vec<CachedExample>,
        options: TrainingOptions,
        identity: ResumeIdentity,
        seed: u64,
        accumulation: u32,
    ) -> Result<Self> {
        graph.validate_factors(&params)?;
        if examples.is_empty() || accumulation == 0 {
            return Err(invalid("empty prepared cache or invalid accumulation"));
        }
        for e in &examples {
            if e.latent.dtype() != Dtype::Float32
                || e.latent.shape()
                    != [
                        e.grid.0 * e.grid.1,
                        graph.dit.config().image_in_channels as i32,
                    ]
                || (e.reference.is_some()) != (options.workflow == Workflow::PairedEdit)
                || e.reference
                    .as_ref()
                    .is_some_and(|r| r.shape() != e.latent.shape() || r.dtype() != Dtype::Float32)
                || (options.caption_dropout > 0.0
                    && e.empty_text
                        .as_ref()
                        .is_none_or(|t| t.shape() != e.text.shape() || t.dtype() != e.text.dtype()))
            {
                return Err(invalid("prepared cache geometry/workflow/dtype mismatch"));
            }
        }
        Ok(Self {
            graph,
            params,
            optimizer,
            examples,
            options,
            identity,
            seed,
            accumulation,
            pending: LoraParams::new(),
        })
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
}
fn host_bits(array: &Array) -> Result<Vec<u32>> {
    let flat = array.as_dtype(Dtype::Float32)?.reshape(&[-1])?;
    eval([&flat])?;
    Ok(flat.as_slice::<f32>().iter().map(|v| v.to_bits()).collect())
}
impl SourceTrainingSession for MlxSession {
    fn identity(&self) -> &ResumeIdentity {
        &self.identity
    }
    fn resume_format(&self) -> &'static str {
        "flux3-mlx-training-resume-v1"
    }
    fn restore(
        &mut self,
        path: &Path,
        policy: ResumePolicy,
        flag: &CancelFlag,
    ) -> gen_core::Result<TrainingCursor> {
        let state = checkpoint::restore(
            &self.graph,
            &mut self.params,
            &mut self.optimizer,
            &self.identity,
            policy,
            path,
            flag,
        )?;
        self.pending = state.pending;
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
                let normal = if self.options.weighting == Weighting::LogitNormal {
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
                let count = example.latent.shape().iter().product::<i32>() as usize;
                let noise = Array::from_slice(
                    &(0..count).map(|_| noise.normal()).collect::<Vec<_>>(),
                    example.latent.shape(),
                );
                let image = example
                    .latent
                    .multiply(Array::from_f32(1.0 - sigma))?
                    .add(noise.multiply(Array::from_f32(sigma))?)?;
                let target = noise.subtract(&example.latent)?;
                let mut caption = NativeRng::keyed(self.seed, "caption-dropout", stream);
                let text = match caption_choice(
                    caption.uniform(),
                    self.options.caption_dropout,
                    false,
                    example.text_ids.len(),
                )
                .map_err(Error::from)?
                {
                    CaptionChoice::Original => &example.text,
                    CaptionChoice::EmptyAtWindow(_) => example
                        .empty_text
                        .as_ref()
                        .ok_or_else(|| invalid("missing matched-window empty caption"))?,
                };
                let (h, w) = (example.grid.0 as usize, example.grid.1 as usize);
                let image_ids = scaled_image_ids(&latent_ids(h, w), IMAGE_ROPE_SCALE);
                let cond_ids = training_condition_ids(h, w).map_err(Error::from)?;
                let samples = [SampleTokens {
                    text,
                    text_ids: &example.text_ids,
                    image: &image,
                    image_ids: &image_ids,
                    image_cond: example.reference.as_ref().map(|r| (r, cond_ids.as_slice())),
                    timestep: sigma,
                    target_timestep: None,
                }];
                let scale = accumulation_loss_scale(self.accumulation).map_err(Error::from)?
                    / indices.len() as f32;
                let weight = self
                    .options
                    .weighting
                    .loss_weight(sigma)
                    .map_err(Error::from)?;
                let (loss, grads) =
                    self.graph
                        .loss_and_grads(&self.params, &samples, &target, weight * scale)?;
                eval([&loss])?;
                raw_loss += loss.item::<f32>() / scale / indices.len() as f32;
                for (name, gradient) in grads {
                    let next = match self.pending.remove(&name) {
                        Some(previous) => previous.add(&gradient)?,
                        None => gradient,
                    };
                    eval([&next])?;
                    self.pending.insert(name, next);
                }
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
            if self.pending.len() != self.params.len() {
                return Err(invalid(
                    "pending gradients do not cover every declared trainable factor",
                ));
            }
            let mut total = Array::from_f32(0.0);
            for gradient in self.pending.values() {
                total = total.add(gradient.square()?.sum(None)?)?;
            }
            let norm = total.sqrt()?;
            eval([&norm])?;
            let coefficient =
                source_clip_multiplier(norm.item::<f32>(), max_grad_norm).map_err(Error::from)?;
            let grads: LoraParams = if coefficient == 1.0 {
                std::mem::take(&mut self.pending)
            } else {
                std::mem::take(&mut self.pending)
                    .into_iter()
                    .map(|(k, g)| Ok((k, g.multiply(Array::from_f32(coefficient))?)))
                    .collect::<Result<_>>()?
            };
            self.optimizer.set_lr_scaled(lr_multiplier)?;
            self.optimizer.step(&mut self.params, &grads)?;
            let arrays = self.params.values().collect::<Vec<_>>();
            eval(arrays)?;
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
            let stem = output_stem(&req.file_name).map_err(Error::from)?;
            checkpoint::save(
                &self.graph,
                Snapshot {
                    params: &self.params,
                    optimizer: &self.optimizer,
                    identity: &self.identity,
                    cursor,
                    policy,
                    pending: &self.pending,
                },
                &req.output_dir,
                &checkpoint_name(stem, cursor.micro_step, true),
                flag,
            )?;
            checkpoint::export_adapter(
                &self.graph,
                &self.params,
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
                let spec = AdapterSpec::new(path.clone(), 1.0, AdapterKind::Lora);
                gen_core::flux3::adapters::inspect(&spec, self.graph.dit.config())
                    .map_err(Error::from)?;
                let saved = Weights::from_file(&path)?;
                let metadata =
                    gen_core::weightsmeta::safetensors_file_metadata(&path).map_err(Error::from)?;
                if metadata.get("training_identity")
                    != Some(&identity.to_metadata().map_err(Error::from)?)
                {
                    return Err(invalid("existing output belongs to another training identity/progress; refusing overwrite"));
                }
                for target in &self.graph.targets {
                    for suffix in ["lora_A.weight", "lora_B.weight"] {
                        if host_bits(saved.require(&format!("{}.{}", target.diffusers, suffix))?)?
                            != host_bits(
                                &self.params[format!("{}.{}", target.native, suffix).as_str()],
                            )?
                        {
                            return Err(invalid(
                                "existing output factor content differs; refusing overwrite",
                            ));
                        }
                    }
                }
                if saved.len() != self.params.len() {
                    return Err(invalid("existing output contains undeclared factors"));
                }
                return Ok(ExportReceipt {
                    path,
                    created: false,
                });
            }
            Ok(ExportReceipt {
                path: checkpoint::export_adapter(
                    &self.graph,
                    &self.params,
                    &identity,
                    &req.output_dir,
                    &req.file_name,
                    flag,
                )?,
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

fn host_pixels(image: &gen_core::Image) -> Result<Array> {
    Ok(Array::from_slice(
        &preprocess::normalized_hwc(image).map_err(Error::from)?,
        &[1, image.height as i32, image.width as i32, 3],
    ))
}
fn execute(
    spec: &LoadSpec,
    req: &TrainingRequest,
    on_progress: &mut dyn FnMut(TrainingProgress),
) -> Result<TrainingOutput> {
    check(&req.cancel)?;
    let opts = TrainingOptions::parse(req).map_err(Error::from)?;
    on_progress(TrainingProgress::Preparing);
    check(&req.cancel)?;
    let WeightsSource::Dir(root) = &spec.weights else {
        return Err(invalid("base directory required"));
    };
    let cfg = Flux3Config::parse(
        &std::fs::read_to_string(root.join(CONFIG_FILE))
            .map_err(|e| invalid(format!("read base config: {e}")))?,
    )
    .map_err(Error::from)?;
    identity::validate_release_config(&cfg).map_err(Error::from)?;
    let resources =
        GenerationResources::from_spec(spec, &cfg, BASE_MODEL_ID).map_err(Error::from)?;
    let artifacts =
        identity::observe_pinned_resources(&resources, &req.cancel).map_err(Error::from)?;
    let targets = lora_targets(&cfg.dit, &req.config.lora_target_modules).map_err(Error::from)?;
    let pixels = req
        .items
        .iter()
        .enumerate()
        .map(|(index, item)| {
            preprocess::decode_pair(
                &item.image_path,
                item.reference_image_paths.first().map(|p| p.as_path()),
                PixelOptions {
                    width: req.config.resolution,
                    height: req.config.resolution,
                    center_crop: opts.center_crop,
                    random_flip: opts.random_flip,
                    seed: req.config.seed,
                },
                index as u64,
            )
            .map_err(Error::from)
        })
        .collect::<Result<Vec<_>>>()?;
    let geometry = pixels
        .iter()
        .map(|p| p.geometry.clone())
        .collect::<Vec<_>>();
    let sampler = BatchSampler::new(
        &vec![0usize; pixels.len()],
        req.config.batch_size,
        req.config.seed,
    )
    .map_err(Error::from)?;
    let plan = RunPlan {
        schedule: opts
            .schedule(&req.config, sampler.len() as u64)
            .map_err(Error::from)?,
        sampler,
        max_grad_norm: opts.max_grad_norm,
    };
    let identity = identity::build(req, &cfg, &opts, artifacts, &targets, &geometry, "mlx")
        .map_err(Error::from)?;
    if req.config.resume {
        controller::find_resume(
            &req.output_dir,
            output_stem(&req.file_name).map_err(Error::from)?,
            &identity,
            plan.policy(req),
            "flux3-mlx-training-resume-v1",
            req.config.seed,
            &req.cancel,
        )
        .map_err(Error::from)?
        .ok_or_else(|| {
            invalid("resume requested but no compatible saved state exists; refusing restart")
        })?;
    }
    on_progress(TrainingProgress::LoadingModel);
    check(&req.cancel)?;
    let dtype = match opts.precision {
        TrainingPrecision::F32 => Dtype::Float32,
        TrainingPrecision::Bf16 => Dtype::Bfloat16,
    };
    let encoder = FrozenTrainingEncoder::from_weights(
        &Weights::from_file(&resources.autoencoder_file)?,
        dtype,
    )?;
    let teacher = Flux3TextEncoder::load_for_training(
        &resources.text_encoder_dir,
        &cfg.text_encoder,
        cfg.dit.text_in_channels,
        dtype,
        opts.max_sequence_length,
    )?;
    let mut windows = req
        .items
        .iter()
        .map(|item| teacher.window(&item.caption))
        .collect::<Result<Vec<_>>>()?;
    if req.config.batch_size > 1 {
        let length = windows
            .iter()
            .map(|w| w.length())
            .max()
            .ok_or_else(|| invalid("empty caption windows"))?;
        for window in &mut windows {
            let truncated = window.truncated_tokens;
            *window = assemble_window(
                &window.input_ids[..window.real],
                teacher.pad_id(),
                length,
                opts.max_sequence_length,
            )
            .map_err(Error::from)?;
            window.truncated_tokens = truncated;
        }
    }
    let empty = teacher.window("")?;
    let mut empties: BTreeMap<usize, Array> = BTreeMap::new();
    let mut captions: BTreeMap<(String, usize), Array> = BTreeMap::new();
    let mut examples = Vec::new();
    for (index, ((item, pair), window)) in req.items.iter().zip(&pixels).zip(windows).enumerate() {
        check(&req.cancel)?;
        let (latent, grid) = encoder.encode(&host_pixels(&pair.target)?)?;
        let reference = pair
            .reference
            .as_ref()
            .map(|image| -> Result<Array> {
                let (tokens, g) = encoder.encode(&host_pixels(image)?)?;
                if g != grid {
                    return Err(invalid("paired encoder grids differ"));
                }
                Ok(tokens)
            })
            .transpose()?;
        let key = (item.caption.clone(), window.length());
        let text = match captions.get(&key) {
            Some(t) => t.clone(),
            None => {
                let t = teacher.encode_window(window.clone())?.tokens;
                captions.insert(key, t.clone());
                t
            }
        };
        let empty_text = if opts.caption_dropout > 0.0 {
            let length = window.length();
            Some(match empties.get(&length) {
                Some(t) => t.clone(),
                None => {
                    let t = teacher
                        .encode_window(
                            assemble_window(
                                &empty.input_ids[..empty.real],
                                teacher.pad_id(),
                                length,
                                opts.max_sequence_length,
                            )
                            .map_err(Error::from)?,
                        )?
                        .tokens;
                    empties.insert(length, t.clone());
                    t
                }
            })
        } else {
            None
        };
        let mut arrays = vec![&latent, &text];
        if let Some(r) = &reference {
            arrays.push(r);
        }
        if let Some(t) = &empty_text {
            arrays.push(t);
        }
        eval(arrays)?;
        examples.push(CachedExample {
            latent,
            text,
            text_ids: gen_core::flux3::text_ids(window.length()),
            empty_text,
            reference,
            grid,
        });
        on_progress(TrainingProgress::Caching {
            current: (index + 1) as u32,
            total: req.items.len() as u32,
        });
        check(&req.cancel)?;
    }
    drop(teacher);
    drop(encoder);
    drop(pixels);
    check(&req.cancel)?;
    if gen_core::train::resume::request_fingerprint(req).map_err(Error::from)? != identity.dataset
        || identity::observe_pinned_resources(&resources, &req.cancel).map_err(Error::from)?
            != identity.artifacts
    {
        return Err(invalid(
            "dataset/base/component bytes changed during frozen caching; refusing training",
        ));
    }
    let dit = crate::model::load_dit(&resources.package_dir.join(DIT_WEIGHTS_FILE), &cfg, dtype)?;
    let graph = TrainingGraph::new(
        dit,
        req.config.rank as usize,
        req.config.alpha as f64,
        &req.config.lora_target_modules,
    )?;
    let mut params = LoraParams::new();
    for (index, target) in graph.targets.iter().enumerate() {
        check(&req.cancel)?;
        let mut rng = NativeRng::keyed(req.config.seed, "lora-gaussian-init", index as u64);
        let rank = req.config.rank as usize;
        let values = (0..rank * target.input)
            .map(|_| rng.normal() / req.config.rank as f32)
            .collect::<Vec<_>>();
        params.insert(
            format!("{}.lora_A.weight", target.native).into(),
            Array::from_slice(&values, &[rank as i32, target.input as i32]),
        );
        params.insert(
            format!("{}.lora_B.weight", target.native).into(),
            Array::zeros::<f32>(&[target.output as i32, rank as i32])?,
        );
    }
    let mut session = MlxSession::new(
        graph,
        params,
        SourceAdamW::new(req.config.learning_rate, req.config.weight_decay)?,
        examples,
        opts,
        identity,
        req.config.seed,
        req.config.gradient_accumulation,
    )?;
    let output = controller::run(&mut session, req, &plan, on_progress).map_err(Error::from)?;
    check(&req.cancel)?;
    Ok(output)
}

#[cfg(test)]
mod tests {
    //! Supported-host only (MLX cannot build on Windows): the same real interrupted
    //! pending-gradient resume / bit-identical continuation / reload contract as Candle.
    use super::*;
    use gen_core::flux3::training::policy::SourceLrKind;
    use gen_core::train::{TrainingConfig, TrainingItem};
    use std::path::PathBuf;
    fn fixtures() -> PathBuf {
        Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures")
    }
    fn ids(a: &Array) -> Vec<[f32; 4]> {
        let f = a.as_dtype(Dtype::Float32).unwrap().reshape(&[-1]).unwrap();
        eval([&f]).unwrap();
        f.as_slice::<f32>()
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
    fn session() -> (MlxSession, Flux3Config) {
        let config = Flux3Config::parse(
            &std::fs::read_to_string(fixtures().join("tiny-base-snapshot/config.json")).unwrap(),
        )
        .unwrap();
        let dit = crate::model::load_dit(
            &fixtures().join("tiny-base-snapshot/model.safetensors"),
            &config,
            Dtype::Float32,
        )
        .unwrap();
        let golden =
            Weights::from_file(&fixtures().join("flux3_training_golden.safetensors")).unwrap();
        let g = |k: &str| golden.require(k).unwrap().clone();
        let graph = TrainingGraph::new(dit, 3, 1.5, &[]).unwrap();
        let mut params = LoraParams::new();
        for t in &graph.targets {
            for s in ["lora_A.weight", "lora_B.weight"] {
                let n = format!("{}.{}", t.native, s);
                params.insert(n.clone().into(), g(&format!("initial/{n}")));
            }
        }
        let image_ids = ids(&g("image_ids"));
        let h = image_ids.iter().map(|i| (i[1] * 2.0) as i32).max().unwrap() + 1;
        let w = image_ids.iter().map(|i| (i[2] * 2.0) as i32).max().unwrap() + 1;
        let examples = (0..3)
            .map(|i| CachedExample {
                latent: g("clean")
                    .multiply(Array::from_f32(1.0 + 0.1 * i as f32))
                    .unwrap(),
                text: g("text"),
                text_ids: ids(&g("text_ids")),
                empty_text: Some(g("text").multiply(Array::from_f32(0.5)).unwrap()),
                reference: Some(g("reference")),
                grid: (h, w),
            })
            .collect();
        let factors = params
            .iter()
            .map(|(k, a)| {
                (
                    k.to_string(),
                    a.shape().iter().map(|d| *d as usize).collect(),
                )
            })
            .collect();
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
                ("backend".into(), "mlx".into()),
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
            MlxSession::new(
                graph,
                params,
                SourceAdamW::new(0.0003, 0.0001).unwrap(),
                examples,
                options(),
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
        let w = Weights::from_file(path).unwrap();
        w.keys()
            .map(|k| (k.to_owned(), host_bits(w.require(k).unwrap()).unwrap()))
            .collect()
    }
    #[test]
    fn real_loop_interrupts_with_pending_gradient_resumes_identically_and_reloads() {
        let a = tempfile::tempdir().unwrap();
        let (mut continuous, config) = session();
        let req = request(a.path(), false);
        let output = controller::run(&mut continuous, &req, &plan(&req), &mut |_| {}).unwrap();
        assert_eq!(output.steps, 5);
        assert!(output.final_loss.is_finite());
        let b = tempfile::tempdir().unwrap();
        let (mut first, _) = session();
        let req = request(b.path(), false);
        let cancel = req.cancel.clone();
        let error = controller::run(&mut first, &req, &plan(&req), &mut |event| {
            if event == (TrainingProgress::Checkpoint { step: 4 }) {
                cancel.cancel()
            }
        })
        .unwrap_err();
        assert!(matches!(error, gen_core::Error::Canceled));
        assert!(!b.path().join("trained.safetensors").exists());
        let (mut resumed, _) = session();
        let req = request(b.path(), true);
        let out = controller::run(&mut resumed, &req, &plan(&req), &mut |_| {}).unwrap();
        assert_eq!(out.steps, 1);
        assert_eq!(out.final_loss.to_bits(), output.final_loss.to_bits());
        assert_eq!(
            bits(&b.path().join("trained.safetensors")),
            bits(&output.adapter_path)
        );
        let spec = AdapterSpec::new(output.adapter_path.clone(), 1.0, AdapterKind::Lora);
        assert_eq!(
            gen_core::flux3::adapters::inspect(&spec, &config.dit)
                .unwrap()
                .len(),
            54
        );
        assert!(controller::ensure_new_output(&request(a.path(), false)).is_err());
    }
}
