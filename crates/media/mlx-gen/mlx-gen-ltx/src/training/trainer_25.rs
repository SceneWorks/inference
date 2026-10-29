//! The LTX-2.5 flexible AV trainer: `train_25_impl` over `AvDiT`, its expanded adapter surface
//! and its per-modality step loss.

use super::*;

impl LtxTrainer {
    pub(super) fn train_25_impl(
        &mut self,
        req: &TrainingRequest,
        on_progress: &mut dyn FnMut(TrainingProgress),
    ) -> Result<TrainingOutput> {
        self.validate(req)?;
        let plan = Ltx25TrainingPlan::from_request(req)?;
        let cfg = &req.config;
        let compute_dtype = Dtype::Float32;
        on_progress(TrainingProgress::Preparing);
        // sc-24830 (epic 2123 E7 + E8): depth anchoring on the generated video stream — the
        // memory guard (decoded frames × the prepared video grid) and the frozen models load
        // BEFORE caching, so a missing checkpoint or an over-budget run fails fast.
        let depth_frames = if mlx_gen_perceptual::any_aux_loss(cfg) {
            let k = depth_anchoring_frames(cfg)?;
            let (f, h, w) = ltx25_video_grids(&req.items)?;
            let aux_gb = ltx25_perceptual_footprint_gb(
                cfg,
                h,
                w,
                k.min(f.max(1) as usize) as u32,
                req.items.len(),
            );
            ltx25_aux_memory_guard(cfg, aux_gb, get_memory_limit())?;
            Some(k)
        } else {
            None
        };
        let perceptual = load_perceptual_path_for(cfg, "ltx_2_5 trainer", ltx25_decoder())?;
        on_progress(TrainingProgress::LoadingModel);

        let mut cached = Vec::with_capacity(req.items.len());
        let mut sample_contexts: Vec<(String, Array, Array)> = Vec::new();
        let mut negative_contexts: Option<(Array, Array)> = None;
        {
            let encoder = self.text_encoder.as_ref().ok_or_else(|| {
                mlx_gen::Error::Msg("ltx_2_5 trainer: text encoder missing".into())
            })?;
            let tokenizer = self
                .tokenizer
                .as_ref()
                .ok_or_else(|| mlx_gen::Error::Msg("ltx_2_5 trainer: tokenizer missing".into()))?;
            for (index, item) in req.items.iter().enumerate() {
                if req.cancel.is_cancelled() {
                    break;
                }
                on_progress(TrainingProgress::Caching {
                    current: index as u32 + 1,
                    total: req.items.len() as u32,
                });
                let (ids, mask) = tokenizer.encode(&item.caption, MAX_PROMPT_TOKENS)?;
                let (video_context, audio_context) = encoder.encode_av(&ids, &mask)?;
                let audio_context = audio_context.ok_or_else(|| {
                    mlx_gen::Error::Msg(
                        "ltx_2_5 trainer: Gemma-4 encoder has no audio connector".into(),
                    )
                })?;
                let video_context = to_dtype(&video_context, Dtype::Float32)?;
                let audio_context = to_dtype(&audio_context, Dtype::Float32)?;
                let example = load_prepared_example(
                    item,
                    &plan,
                    &video_context,
                    Some(&audio_context),
                    cfg.seed,
                    index as u64,
                    depth_frames,
                )?;
                eval([&video_context, &audio_context])?;
                cached.push(example);
            }
            if cfg.sample_every > 0 && !cfg.sample_prompts.is_empty() && !req.cancel.is_cancelled()
            {
                for prompt in cfg.sample_prompts.iter().take(SAMPLE_PROMPT_CAP) {
                    let (ids, mask) = tokenizer.encode(prompt, MAX_PROMPT_TOKENS)?;
                    let (video, audio) = encoder.encode_av(&ids, &mask)?;
                    let video = to_dtype(&video, Dtype::Float32)?;
                    let audio = to_dtype(
                        &audio.ok_or_else(|| {
                            mlx_gen::Error::Msg(
                                "ltx_2_5 trainer: validation requires audio text context".into(),
                            )
                        })?,
                        Dtype::Float32,
                    )?;
                    eval([&video, &audio])?;
                    sample_contexts.push((prompt.clone(), video, audio));
                }
                let (ids, mask) = tokenizer.encode("", MAX_PROMPT_TOKENS)?;
                let (video, audio) = encoder.encode_av(&ids, &mask)?;
                let video = to_dtype(&video, Dtype::Float32)?;
                let audio = to_dtype(
                    &audio.ok_or_else(|| {
                        mlx_gen::Error::Msg(
                            "ltx_2_5 trainer: validation requires negative audio context".into(),
                        )
                    })?,
                    Dtype::Float32,
                )?;
                eval([&video, &audio])?;
                negative_contexts = Some((video, audio));
            }
        }
        if cached.is_empty() {
            return if req.cancel.is_cancelled() {
                Err(mlx_gen::Error::Canceled)
            } else {
                Err("ltx_2_5 trainer: no usable prepared examples".into())
            };
        }
        self.text_encoder = None;
        self.tokenizer = None;

        let TrainingTransformer::Ltx25(transformer) = &mut self.transformer else {
            unreachable!("dispatcher selected the 2.5 path")
        };
        let suffixes: Vec<String> = if cfg.lora_target_modules.is_empty() {
            DEFAULT_TARGET_SUFFIXES
                .iter()
                .map(|value| value.to_string())
                .collect()
        } else {
            cfg.lora_target_modules.clone()
        };
        let (targets, mut params) = build_targets_25(
            transformer,
            self.cfg.num_layers,
            &suffixes,
            cfg.rank as i32,
            cfg.seed,
        )?;
        if targets.is_empty() {
            return Err(
                "ltx_2_5 trainer: no AV LoRA targets resolved (check lora_target_modules)".into(),
            );
        }
        // AV whole-block checkpointing is not yet a separate path; the attention-segment checkpoint
        // still bounds all six attention score graphs while preserving the full AV forward.
        transformer.set_sdpa_checkpoint(true);

        let weight_decay = if cfg.optimizer.eq_ignore_ascii_case("adam") {
            0.0
        } else {
            cfg.weight_decay
        };
        let mut optimizer =
            TrainOptimizer::from_config(&cfg.optimizer, cfg.learning_rate, weight_decay)?;
        let accumulation = cfg.gradient_accumulation.max(1);
        let (total_updates, warmup_updates) =
            schedule_updates(cfg.steps, accumulation, cfg.lr_warmup_steps);
        let stem = Path::new(&req.file_name)
            .file_stem()
            .and_then(|value| value.to_str())
            .unwrap_or("lora")
            .to_string();
        let mut update_index = 0;
        let mut start_step = 0;
        if cfg.resume {
            if let Some((snapshot, _)) = checkpoint::find_latest_resume(&req.output_dir, &stem) {
                let (loaded, metadata) = checkpoint::load_resume(&snapshot, &mut optimizer)?;
                params = loaded;
                start_step = metadata.step;
                update_index = metadata.update_idx;
            }
        }

        let mae = matches!(cfg.loss_type.to_ascii_lowercase().as_str(), "mae" | "l1");
        // sc-24830: each example's depth reference (its clean video frames decoded → DA2) once,
        // and the alternation over the round-robin windows (a pure function of the step, so a
        // resumed run needs no replay).
        let mut aux_driver = match perceptual {
            Some(path) => Some(ltx25_aux_driver(
                path,
                &cached,
                accumulation,
                &Default::default(),
            )?),
            None => None,
        };
        let mut accumulated: Option<LoraParams> = None;
        let mut final_loss = 0.0;
        let mut steps_run = start_step;
        for step in start_step + 1..=cfg.steps {
            if req.cancel.is_cancelled() {
                break;
            }
            let mut sigma = {
                let key =
                    random::key(cfg.seed.wrapping_mul(0x9E37_79B9).wrapping_add(step as u64))?;
                random::uniform::<_, f32>(1e-3, 1.0 - 1e-3, &[1], Some(&key))?.item::<f32>()
            };
            let item = (step as usize - 1) % cached.len();
            let example = &cached[item];
            // sc-24830: plan the step; an aux-only step trains at σ remapped into the loss window.
            let step_plan = match aux_driver.as_mut() {
                Some(driver) => {
                    let key = driver.key(step);
                    let plan = driver.path().plan(key, item, sigma)?;
                    sigma = plan.noise_level;
                    Some(plan)
                }
                _ => None,
            };
            let prepared = prepare_cached_ltx25(
                example,
                sigma,
                cfg.seed.wrapping_add(step as u64).wrapping_mul(2),
            )?;
            let aux = match (
                aux_driver.as_ref().map(AuxDriver::path),
                step_plan.as_ref(),
                example.depth.as_ref(),
            ) {
                (Some(path), Some(plan), Some(depth)) => Some(Ltx25AuxStep {
                    path,
                    plan,
                    image: item,
                    depth,
                }),
                (Some(_), Some(_), None) => {
                    return Err("ltx_2_5 trainer: depth anchoring has no video frames".into())
                }
                _ => None,
            };
            let (losses, grads) = compute_ltx25_step_loss_grads(
                transformer,
                &params,
                &targets,
                cfg.alpha,
                cfg.rank as f32,
                &prepared,
                mae,
                compute_dtype,
                aux,
            )?;
            final_loss = losses.total;
            steps_run = step;
            accumulate_grads(&mut accumulated, grads)?;
            if step % accumulation == 0 || step == cfg.steps {
                optimizer.set_lr_scaled(lr_multiplier(
                    cfg.lr_scheduler,
                    update_index,
                    total_updates,
                    warmup_updates,
                ));
                let window = if step % accumulation == 0 {
                    accumulation
                } else {
                    step % accumulation
                };
                let average = average_grads(
                    accumulated.take().expect("update requires gradients"),
                    window,
                )?;
                // Epic 2123 (sc-24827): clip → gradient noise → step → weight noise.
                adapter_optimizer_update(
                    &mut optimizer,
                    &mut params,
                    &average,
                    cfg,
                    update_index,
                    cfg.seed,
                )?;
                update_index += 1;
            }
            on_progress(TrainingProgress::Training {
                step,
                total: cfg.steps,
                loss: final_loss,
            });
            if cfg.save_every > 0 && step % cfg.save_every == 0 && step != cfg.steps {
                std::fs::create_dir_all(&req.output_dir)?;
                save_lora(
                    &params,
                    &targets,
                    cfg.alpha,
                    cfg.rank,
                    &req.output_dir.join(checkpoint_filename(&stem, step)),
                )?;
                checkpoint::save_resume(
                    &req.output_dir,
                    &stem,
                    step,
                    update_index,
                    &optimizer,
                    &params,
                )?;
                on_progress(TrainingProgress::Checkpoint { step });
            }

            if cfg.sample_every > 0 && step % cfg.sample_every == 0 && !sample_contexts.is_empty() {
                install_train_lora(
                    transformer,
                    &params,
                    &targets,
                    cfg.alpha,
                    cfg.rank as f32,
                    None,
                )?;
                let negatives = negative_contexts.as_ref().ok_or_else(|| {
                    mlx_gen::Error::Msg(
                        "ltx_2_5 trainer: negative validation context missing".into(),
                    )
                })?;
                for (index, (prompt, video_context, audio_context)) in
                    sample_contexts.iter().enumerate()
                {
                    let image = render_ltx25_validation_sample(
                        transformer,
                        &self.vae,
                        video_context,
                        audio_context,
                        &negatives.0,
                        &negatives.1,
                        &plan.validation,
                        cfg.seed
                            .wrapping_add(step as u64)
                            .wrapping_add(index as u64),
                        &req.cancel,
                    )?;
                    on_progress(TrainingProgress::Sample {
                        step,
                        index: index as u32 + 1,
                        total: sample_contexts.len() as u32,
                        prompt: prompt.clone(),
                        image,
                    });
                }
            }
        }
        if steps_run == 0 {
            return Err(mlx_gen::Error::Canceled);
        }
        on_progress(TrainingProgress::Saving);
        std::fs::create_dir_all(&req.output_dir)?;
        let adapter_path = req.output_dir.join(&req.file_name);
        save_lora(&params, &targets, cfg.alpha, cfg.rank, &adapter_path)?;
        Ok(TrainingOutput {
            adapter_path,
            steps: steps_run,
            final_loss,
        })
    }
}

pub(super) const LTX25_ATTENTION_MODULES: [&str; 6] = [
    "attn1",
    "attn2",
    "audio_attn1",
    "audio_attn2",
    "audio_to_video_attn",
    "video_to_audio_attn",
];

/// Expand the reference's suffix matching against the complete AV model.  Short attention suffixes
/// intentionally hit all six attention modules; FF patterns remain modality-specific exactly as in
/// the upstream YAML. No bias target is manufactured: `ff_bias=false` means the 2.5 video FF base
/// legitimately has no bias while its two weight projections remain trainable.
pub(super) fn ltx25_target_paths(num_layers: i32, suffixes: &[String]) -> Vec<String> {
    let mut paths = Vec::new();
    let push_unique = |paths: &mut Vec<String>, path: String| {
        if !paths.contains(&path) {
            paths.push(path);
        }
    };
    for i in 0..num_layers {
        for suffix in suffixes {
            match suffix.as_str() {
                "to_q" | "to_k" | "to_v" | "to_out.0" | "to_gate_logits" => {
                    for module in LTX25_ATTENTION_MODULES {
                        push_unique(
                            &mut paths,
                            format!("transformer_blocks.{i}.{module}.{suffix}"),
                        );
                    }
                }
                "ff.net.0.proj" | "ff.net.2" => {
                    push_unique(&mut paths, format!("transformer_blocks.{i}.{suffix}"));
                    push_unique(&mut paths, format!("transformer_blocks.{i}.audio_{suffix}"));
                }
                "ff" => {
                    for projection in ["ff.net.0.proj", "ff.net.2"] {
                        push_unique(&mut paths, format!("transformer_blocks.{i}.{projection}"));
                        push_unique(
                            &mut paths,
                            format!("transformer_blocks.{i}.audio_{projection}"),
                        );
                    }
                }
                "audio_ff.net.0.proj" | "audio_ff.net.2" => {
                    push_unique(&mut paths, format!("transformer_blocks.{i}.{suffix}"));
                }
                qualified if qualified.contains('.') => {
                    push_unique(&mut paths, format!("transformer_blocks.{i}.{qualified}"));
                }
                _ => {}
            }
        }
    }
    paths
}

pub(super) fn build_targets_25(
    dit: &mut AvDiT,
    num_layers: i32,
    suffixes: &[String],
    rank: i32,
    seed: u64,
) -> Result<(Vec<LtxLoraTarget>, LoraParams)> {
    build_targets_from_paths(dit, ltx25_target_paths(num_layers, suffixes), rank, seed)
}

pub(super) fn masked_modality_loss(
    prediction: &Array,
    modality: &Ltx25PreparedModality,
    mae: bool,
) -> MlxResult<Array> {
    if modality.loss_denominator <= 0.0 {
        return Err(Exception::custom(
            "ltx_2_5 trainer: generated modality has an empty loss mask",
        ));
    }
    let diff = subtract(prediction, &modality.target)?;
    let point = if mae { diff.abs()? } else { diff.square()? };
    let masked = multiply(&point, &modality.loss_mask)?;
    divide(
        &masked.sum(None)?,
        Array::from_f32(modality.loss_denominator),
    )
}

/// One LTX-2.5 aux-loss step's view of the perceptual path (sc-24830): the plan, the example's
/// reference key and its depth frame selection.
pub(super) struct Ltx25AuxStep<'a> {
    pub(super) path: &'a PerceptualPath,
    pub(super) plan: &'a StepPlan,
    /// The example index (selects its cached reference).
    pub(super) image: usize,
    pub(super) depth: &'a Ltx25DepthFrames,
}

/// The depth x0 estimate of the generated video stream (sc-24830): over the target tokens (the
/// appended reference tokens dropped), `x0 = x_t − t·v` with the per-token timestep `t` (the
/// planned σ on generated tokens, `0` on conditioning tokens — whose x0 is then their clean
/// latent), the velocity regressing `noise − clean`; the selected frames as the decoder's NCHW.
pub(super) fn ltx25_video_x0(
    prediction: &Array,
    video: &Ltx25PreparedModality,
    depth: &Ltx25DepthFrames,
) -> Result<Array> {
    use mlx_rs::ops::indexing::IndexOp;
    let tokens = video.target_tokens as i32;
    let noisy = video.noisy.index((.., ..tokens));
    let timestep = video.timestep.index((.., ..tokens)).expand_dims(2)?;
    let velocity = prediction.index((.., ..tokens)).as_dtype(Dtype::Float32)?;
    let x0 = subtract(&noisy, &multiply(&timestep, &velocity)?)?;
    depth.nchw(&x0)
}

/// [`compute_ltx25_loss_grads`] with the step's perceptual plan (sc-24830): on an aux-only step the
/// (video + audio) velocity term is not computed and the loss is the weighted depth term on the
/// generated video's x0 estimate ([`ltx25_video_x0`]); summed (`every_n == 1`) aux terms ride along
/// on a diffusion step. With `aux = None` (or a plan with no aux loss) the traced graph is exactly
/// the pre-epic-2123 one.
#[allow(clippy::too_many_arguments)]
pub(super) fn compute_ltx25_step_loss_grads(
    dit: &mut AvDiT,
    params: &LoraParams,
    targets: &[LtxLoraTarget],
    alpha: f32,
    rank: f32,
    batch: &Ltx25PreparedBatch,
    mae: bool,
    dtype: Dtype,
    aux: Option<Ltx25AuxStep<'_>>,
) -> Result<(StepLosses, LoraParams)> {
    if batch.video.is_none() && batch.audio.is_none() {
        return Err("ltx_2_5 trainer: prepared batch has no modalities".into());
    }
    let (diffusion_on, aux_on) = match &aux {
        Some(a) => (a.plan.diffusion, !a.plan.aux.is_empty()),
        None => (true, false),
    };
    if aux_on && batch.video.is_none() {
        return Err("ltx_2_5 trainer: depth anchoring needs the generated video stream".into());
    }
    let prepared = batch.clone();
    let lora_dtype = (dtype != Dtype::Float32).then_some(dtype);
    let loss_fn = move |p: LoraParams, _: i32| -> MlxResult<Vec<Array>> {
        install_train_lora(dit, &p, targets, alpha, rank, lora_dtype)?;
        let mut video_prediction_out: Option<Array> = None;
        let diffusion = match (&prepared.video, &prepared.audio) {
            (Some(video), Some(audio)) => {
                let (video_prediction, audio_prediction) = dit
                    .forward(
                        &video.noisy,
                        &video.timestep,
                        &video.context,
                        None,
                        &video.positions,
                        &audio.noisy,
                        &audio.timestep,
                        &audio.context,
                        None,
                        &audio.positions,
                        None,
                        None,
                    )
                    .map_err(|error| Exception::custom(error.to_string()))?;
                let diffusion = if diffusion_on {
                    let video_loss = (video.loss_denominator > 0.0)
                        .then(|| masked_modality_loss(&video_prediction, video, mae))
                        .transpose()?;
                    let audio_loss = (audio.loss_denominator > 0.0)
                        .then(|| masked_modality_loss(&audio_prediction, audio, mae))
                        .transpose()?;
                    Some(match (video_loss, audio_loss) {
                        (Some(video), Some(audio)) => add(&video, &audio)?,
                        (Some(loss), None) | (None, Some(loss)) => loss,
                        (None, None) => {
                            return Err(Exception::custom(
                                "ltx_2_5 trainer: both modality loss masks are empty",
                            ))
                        }
                    })
                } else {
                    None
                };
                video_prediction_out = Some(video_prediction);
                diffusion
            }
            (Some(video), None) => {
                let prediction = dit
                    .forward_video_only(
                        &video.noisy,
                        &video.timestep,
                        &video.context,
                        None,
                        &video.positions,
                        None,
                        None,
                    )
                    .map_err(|error| Exception::custom(error.to_string()))?;
                let diffusion = diffusion_on
                    .then(|| masked_modality_loss(&prediction, video, mae))
                    .transpose()?;
                video_prediction_out = Some(prediction);
                diffusion
            }
            (None, Some(audio)) => {
                let prediction = dit
                    .forward_audio_only(
                        &audio.noisy,
                        &audio.timestep,
                        &audio.context,
                        None,
                        &audio.positions,
                        None,
                    )
                    .map_err(|error| Exception::custom(error.to_string()))?;
                diffusion_on
                    .then(|| masked_modality_loss(&prediction, audio, mae))
                    .transpose()?
            }
            (None, None) => unreachable!(),
        };
        let aux_term = match (&aux, &prepared.video, &video_prediction_out) {
            (Some(a), Some(video), Some(prediction)) if aux_on => {
                let x0 = ltx25_video_x0(prediction, video, a.depth)
                    .map_err(|e| Exception::custom(e.to_string()))?;
                a.path
                    .aux_loss(a.plan, a.image, &x0)
                    .map_err(|e| Exception::custom(e.to_string()))?
                    .map(|t| t.weighted)
            }
            _ => None,
        };
        let zero = || Array::from_f32(0.0);
        let d_out = diffusion.clone().unwrap_or_else(zero);
        let a_out = aux_term.clone().unwrap_or_else(zero);
        let total =
            combine_step_loss(diffusion, aux_term).map_err(|e| Exception::custom(e.to_string()))?;
        Ok(vec![total, d_out, a_out])
    };
    let mut vg = keyed_value_and_grad(loss_fn);
    let (value, grads) = vg(params.clone(), 0)?;
    let losses = StepLosses {
        total: value[0].item::<f32>(),
        diffusion: diffusion_on.then(|| value[1].item::<f32>()),
        aux: aux_on.then(|| value[2].item::<f32>()),
    };
    Ok((losses, grads))
}
