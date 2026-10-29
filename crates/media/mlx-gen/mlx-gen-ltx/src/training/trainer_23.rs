//! The LTX-2.3 video-only trainer: `train_23_impl` over `LtxDiT`, its single-frame latent
//! layout and its step loss.

use super::*;

impl LtxTrainer {
    pub(super) fn train_23_impl(
        &mut self,
        req: &TrainingRequest,
        on_progress: &mut dyn FnMut(TrainingProgress),
        warnings: &mut Vec<gen_core::GenerationWarning>,
    ) -> Result<TrainingOutput> {
        self.validate(req)?;
        let TrainingTransformer::Ltx23(transformer) = &mut self.transformer else {
            unreachable!("dispatcher selected the 2.3 path")
        };
        let cfg = &req.config;
        on_progress(TrainingProgress::Preparing);
        // sc-2127 — one pixel edge (multiple of 32) per resolution bucket (just `[resolution]` when
        // buckets are off), and its latent tokens per side. The memory guard and previews size for
        // the largest (epic 2123 E7).
        let edges = bucket_edges(cfg);
        let latent_edges: Vec<usize> = edges.iter().map(|&e| latent_edge_of(e)).collect();
        let latent_edge = guard_latent_edge(&edges);

        // sc-4942 — LTX trains in **f32 activations** (× the Q-packed base), NOT bf16, even though the
        // SceneWorks worker passes `train_dtype=bf16` (sc-4881). MEASURED on real weights (the
        // `bf16_grads_direction_and_memory_vs_f32` harness): a bf16 activation cast DECORRELATES the
        // gradient from the f32 (quality) path — global cosine 0.31–0.45, with the early/deep K
        // projections of BOTH attentions pointing ~opposite — because the 48-block distilled DiT is
        // chaos-sensitive (the same reason inference uses `quant_f32`, not `quant_bf16`, for quality;
        // see `transformer::Precision`/`Mode::QuantF32`). bf16 would save ~30% memory (1024: 28 vs 43 GB)
        // but the f32 working set already fits the video tier with the checkpointing levers below
        // (1024 ≈ 43 GB attn-ckpt / 27 GB block-ckpt, measured 128 GB box; the guard auto-scales), so
        // honoring bf16 would trade training quality for memory we do not need. (`LtxDiT::cast_weights`
        // stays available — the bf16 harness that produced this finding exercises it — but the
        // production trainer never invokes it; this trainer is f32-only.)
        let compute_dtype = Dtype::Float32;

        // sc-4942 — fail-fast pre-flight memory guard (the sc-4874 mechanism, ported to LTX). The dense
        // (non-block-checkpointed) first step materializes the whole forward graph in one MLX `eval`; at
        // high resolution that working set can exceed unified memory and the OS hard-kills the worker
        // with an UNCATCHABLE SIGKILL. We predict it and refuse up front with an actionable, catchable
        // error — BEFORE the (~minutes-long) latent caching — when gradient checkpointing is not
        // enabled. (LTX is LoRA-only, so the LoRA-path condition is always met.)
        let will_checkpoint = cfg.gradient_checkpointing;
        // Epic 2123 E7 (sc-24830): the aux-loss models (TAELTX2.3 + every enabled arm)
        // count against the budget on BOTH paths — a checkpointed depth job must not skip it.
        let aux_gb = perceptual_footprint_gb(
            cfg,
            edges.iter().copied().max().unwrap_or(0),
            req.items.len() * edges.len(),
        );
        if !will_checkpoint || aux_gb > 0.0 {
            preflight_memory_guard(cfg, latent_edge, aux_gb, will_checkpoint)?;
        }

        // Epic 2123 depth anchoring (sc-24830): load the frozen TAELTX2.3 decoder +
        // Depth-Anything-V2 before the caching pass, so a missing checkpoint fails fast.
        let mut perceptual = load_perceptual_path(cfg)?;

        // --- prepare → load → cache: normalized latents + prompt embeds (then free the TE) ---
        on_progress(TrainingProgress::LoadingModel);
        let total = req.items.len() as u32;
        // Item-major: `cache[item * edges.len() + bucket]` (sc-2127). Per entry: patchified clean
        // latent, prompt embeds, and (subject-masked loss, sc-24828) that latent's patchified
        // loss-weight map — `None` when the technique is off.
        let mut cache: Vec<(Array, Array, Option<Array>)> =
            Vec::with_capacity(req.items.len() * edges.len());
        // sc-5637 — preview-sample prompts, pre-encoded inside the `te`/`tok` scope below (the Gemma
        // encoder is freed before the train loop). LTX is distilled (no CFG) → one ctx per prompt.
        let mut sample_ctxs: Vec<(String, Array)> = Vec::new();
        {
            let te = self.text_encoder.as_ref().ok_or_else(|| {
                mlx_gen::Error::Msg("ltx_2_3 trainer: text encoder missing".into())
            })?;
            let tok = self
                .tokenizer
                .as_ref()
                .ok_or_else(|| mlx_gen::Error::Msg("ltx_2_3 trainer: tokenizer missing".into()))?;
            for (i, item) in req.items.iter().enumerate() {
                if req.cancel.is_cancelled() {
                    break;
                }
                on_progress(TrainingProgress::Caching {
                    current: i as u32 + 1,
                    total,
                });
                let img = center_crop_square(&decode_image(&item.image_path)?);
                // sc-24828: the item's mask is read and checked once, resampled per bucket below.
                let subject_mask = PreparedSubjectMask::load_if_enabled(
                    "ltx_2_3 trainer",
                    item,
                    cfg.subject_mask_loss.as_ref(),
                )?;
                let (ids, mask) = tok.encode(&item.caption, MAX_PROMPT_TOKENS)?;
                let ctx = to_dtype(&te.encode(&ids, &mask)?, Dtype::Float32)?; // (1, L, 4096)
                eval([&ctx])?;
                for (clean, mask_weight) in encode_bucket_latents(
                    &edges,
                    |edge| {
                        let prep = preprocess_conditioning_image(&img, edge, edge)?; // (1,3,1,edge,edge)
                        self.vae.encode(&prep) // (1,128,1,le,le), normalized, f32
                    },
                    |shape| still_subject_weight("ltx_2_3 trainer", subject_mask.as_ref(), shape),
                )? {
                    cache.push((clean, ctx.clone(), mask_weight));
                }
            }
            // sc-5637 — pre-encode the preview-sample prompts while the encoder is still resident.
            if cfg.sample_every > 0 && !cfg.sample_prompts.is_empty() && !req.cancel.is_cancelled()
            {
                for prompt in cfg.sample_prompts.iter().take(SAMPLE_PROMPT_CAP) {
                    let (ids, mask) = tok.encode(prompt, MAX_PROMPT_TOKENS)?;
                    let ctx = to_dtype(&te.encode(&ids, &mask)?, Dtype::Float32)?;
                    eval([&ctx])?;
                    sample_ctxs.push((prompt.clone(), ctx));
                }
            }
        }
        if cache.is_empty() {
            // sc-4895 — a cancel tripped during caching is a genuine cancellation → typed
            // `Error::Canceled` (bridged 1:1 to `gen_core::Error::Canceled`); an empty cache with no
            // cancel is a real "no usable dataset items" error.
            if req.cancel.is_cancelled() {
                return Err(mlx_gen::Error::Canceled);
            }
            return Err("ltx_2_3 trainer: no usable dataset items".into());
        }
        // Free the Gemma text encoder + tokenizer (~24 GB) before training — they are only needed for
        // the one-time embed cache (mirrors the reference `prepare_dataset` release).
        self.text_encoder = None;
        self.tokenizer = None;

        let sampling_enabled = !sample_ctxs.is_empty();

        // Epic 2123 E8: each (item, bucket) entry's perceptual reference (TAELTX2.3 decode of its
        // cached clean latent → DA2 depth) is computed exactly once per job, by the `AuxDriver`
        // before the loop.
        if let Some(path) = perceptual.as_mut() {
            // sc-24832: the job's subject masks (restricted normal loss) reach every reference,
            // cropped like the image and resampled onto its decoded size.
            path.attach_subject_masks(mlx_gen::train::subject_mask::PerceptualSubjectMasks::load(
                "ltx trainer",
                &req.items,
                cfg,
                latent_edges.len(),
                CropBox::center_square,
            )?);
        }

        // The RoPE position grid is identical across items at a fixed latent resolution (single
        // frame) — build it once per bucket (sc-2127: each step uses its sampled bucket's grid).
        // Preview-sample rendering (sc-5637) uses the largest bucket's grid at `latent_edge`.
        let bucket_positions: Vec<Array> = latent_edges
            .iter()
            .map(|&le| create_position_grid(1, 1, le, le))
            .collect();
        let preview_positions = &bucket_positions[largest_bucket(&latent_edges)];

        // --- adapter targets + trainable factors ---
        let suffixes: Vec<String> = if cfg.lora_target_modules.is_empty() {
            DEFAULT_TARGET_SUFFIXES
                .iter()
                .map(|s| s.to_string())
                .collect()
        } else {
            cfg.lora_target_modules.clone()
        };
        let (targets, mut params) = build_targets(
            transformer,
            self.cfg.num_layers,
            &suffixes,
            cfg.rank as i32,
            cfg.seed,
        )?;
        if targets.is_empty() {
            return Err(
                "ltx_2_3 trainer: no LoRA targets resolved (check lora_target_modules)".into(),
            );
        }
        let alpha = cfg.alpha;
        let rank = cfg.rank as f32;
        let mae = {
            let lt = cfg.loss_type.to_ascii_lowercase();
            lt == "mae" || lt == "l1"
        };

        // sc-4942 — gradient checkpointing. Group the resolved targets by their owning block so the
        // checkpointed forward can thread each block's LoRA factors as explicit recompute inputs. Every
        // target of this trainer lives in a `transformer_blocks.{i}.attn{1,2}` leaf, so the grouping
        // covers the whole adapter surface.
        let n_layers = self.cfg.num_layers as usize;
        let block_targets = group_block_targets(&targets, n_layers);
        // Gradient checkpointing is an OPT-IN OPTION (the SceneWorks "Gradient Checkpointing" toggle),
        // never auto-forced — a run that would OOM is caught instead by the fail-fast pre-flight guard
        // above, which recommends this flag rather than silently changing the user's training dynamics.
        let use_checkpoint = cfg.gradient_checkpointing;
        let checkpoint_block: Option<&[Vec<BlockLoraRef>]> = if use_checkpoint {
            Some(&block_targets)
        } else {
            None
        };
        // sc-4942 — attention-segment checkpointing is on for the dense (non-block-checkpointed) path:
        // it is numerically identical to the retained backward (same decomposed attention, recomputed)
        // and removes the dominant seq² per-block retention — the flash-backward surrogate every torch
        // trainer gets from its fused SDPA kernel. When whole-block checkpointing is on it goes OFF (the
        // block recompute already covers attention; nesting would recompute it twice for no win).
        transformer.set_sdpa_checkpoint(!use_checkpoint);

        // AdamW with wd=0 is identical to Adam, so the one optimizer covers both choices.
        let weight_decay = if cfg.optimizer.eq_ignore_ascii_case("adam") {
            0.0
        } else {
            cfg.weight_decay
        };
        let mut opt = TrainOptimizer::from_config(&cfg.optimizer, cfg.learning_rate, weight_decay)?;

        let accum = cfg.gradient_accumulation.max(1);
        let (total_updates, warmup_updates) =
            schedule_updates(cfg.steps, accum, cfg.lr_warmup_steps);
        let stem = Path::new(&req.file_name)
            .file_stem()
            .and_then(|s| s.to_str())
            .unwrap_or("lora")
            .to_string();

        // --- resume (F-125): continue from the latest snapshot of THIS adapter in output_dir, if any ---
        let mut update_idx: u32 = 0;
        let mut start_step: u32 = 0;
        if cfg.resume {
            if let Some((snapshot, _)) = checkpoint::find_latest_resume(&req.output_dir, &stem) {
                let (loaded, meta) = checkpoint::load_resume(&snapshot, &mut opt)?;
                params = loaded;
                start_step = meta.step;
                update_idx = meta.update_idx;
                eprintln!(
                    "[F-125] resuming from step {start_step} (optimizer update {update_idx})"
                );
            }
        }

        // --- train loop ---
        // sc-2127: which cached (item, bucket) latent each step trains on (round-robin over items
        // for a single bucket — the pre-bucket order; a seeded per-epoch shuffle otherwise).
        let schedule =
            BucketSchedule::new(cache.len() / edges.len(), &cfg.training_buckets(), cfg.seed);
        // Epic 2123 E8: the perceptual alternation interleaves optimizer windows over the
        // schedule's epochs (sc-2124) — a pure function of the step, so a resumed run needs no
        // replay.
        let mut aux_driver = match perceptual {
            Some(path) => Some(aux_driver(
                path,
                &cache,
                &latent_edges,
                &schedule,
                accum,
                &Default::default(),
            )?),
            None => None,
        };
        let mut accumulated: Option<LoraParams> = None;
        let mut last_loss = 0.0f32;
        let mut steps_run = start_step;
        for step in start_step + 1..=cfg.steps {
            if req.cancel.is_cancelled() {
                break;
            }
            let (entry, bucket) = step_entry(&schedule, step);
            let (clean, ctx, mask_weight) = &cache[entry];
            let positions = &bucket_positions[bucket];
            // σ ~ U(1e-3, 1-1e-3), deterministic in seed (the reference's uniform timestep).
            let mut sigma = {
                let k = random::key(cfg.seed.wrapping_mul(0x9E37_79B9).wrapping_add(step as u64))?;
                random::uniform::<_, f32>(1e-3f32, 1.0 - 1e-3, &[1], Some(&k))?.item::<f32>()
            };
            // Epic 2123 E8: plan the step's loss terms; an aux-only step trains at the σ remapped
            // into the loss window.
            let planned = match aux_driver.as_mut() {
                Some(d) => d.sample(step, &schedule).plan(sigma)?,
                None => None,
            };
            if let Some(p) = &planned {
                sigma = p.plan.noise_level;
            }
            let aux = planned.as_ref().map(|p| AuxStep {
                path: p.path,
                plan: &p.plan,
                image: p.entry,
                latent_edge: latent_edges[bucket],
            });
            let noise = random::normal::<f32>(
                clean.shape(),
                None,
                None,
                Some(&random::key(
                    cfg.seed.wrapping_add(step as u64).wrapping_mul(2) + 1,
                )?),
            )?;
            let (losses, grads) = compute_step_loss_grads(
                transformer,
                &params,
                &targets,
                alpha,
                rank,
                clean,
                ctx,
                positions,
                sigma,
                &noise,
                mae,
                mask_weight.as_ref(),
                checkpoint_block,
                compute_dtype,
                aux,
            )?;
            last_loss = losses.total;
            steps_run = step;
            accumulate_grads(&mut accumulated, grads)?;

            if step % accum == 0 || step == cfg.steps {
                let mult =
                    lr_multiplier(cfg.lr_scheduler, update_idx, total_updates, warmup_updates);
                opt.set_lr_scaled(mult);
                // F-017: average by the ACTUAL in-window count, not the full `accum`. The final-step
                // flush is usually a partial window (cfg.steps % accum != 0); dividing by `accum`
                // down-scaled that update (halved effective LR on the tail). Mirrors z-image/lens
                // F-069. (When step%accum==0 the window is the full `accum`.)
                let window = if step % accum == 0 {
                    accum
                } else {
                    step % accum
                };
                let avg = average_grads(
                    accumulated
                        .take()
                        .expect("an update fires only after accumulation"),
                    window,
                )?;
                // Epic 2123 (sc-24827): clip → gradient noise → step → weight noise.
                adapter_optimizer_update(&mut opt, &mut params, &avg, cfg, update_idx, cfg.seed)?;
                update_idx += 1;
            }

            on_progress(TrainingProgress::Training {
                step,
                total: cfg.steps,
                loss: last_loss,
            });

            if cfg.save_every > 0 && step % cfg.save_every == 0 && step != cfg.steps {
                std::fs::create_dir_all(&req.output_dir)?;
                let ckpt = req.output_dir.join(checkpoint_filename(&stem, step));
                save_lora(&params, &targets, alpha, cfg.rank, &ckpt)?;
                checkpoint::save_resume(&req.output_dir, &stem, step, update_idx, &opt, &params)?;
                on_progress(TrainingProgress::Checkpoint { step });
            }

            // sc-5637 — periodic best-effort preview frames from the in-progress adapter. Install the
            // current factors concretely for the forward-only render (the next step's traced `loss_fn`
            // re-installs them); a render failure logs and is skipped, never failing the training run.
            if sampling_enabled && cfg.sample_every > 0 && step % cfg.sample_every == 0 {
                let lora_dtype = (compute_dtype != Dtype::Float32).then_some(compute_dtype);
                install_train_lora(transformer, &params, &targets, alpha, rank, lora_dtype)?;
                let total = sample_ctxs.len() as u32;
                for (i, (prompt, ctx)) in sample_ctxs.iter().enumerate() {
                    if req.cancel.is_cancelled() {
                        break;
                    }
                    let sample_seed = cfg
                        .seed
                        .wrapping_add(step as u64)
                        .wrapping_mul(0xA24B_AED4_4AC9_5F2D)
                        .wrapping_add(i as u64);
                    match crate::pipeline::render_sample(
                        transformer,
                        &self.vae,
                        ctx,
                        preview_positions,
                        sample_seed,
                        latent_edge,
                        compute_dtype,
                    ) {
                        Ok(image) => on_progress(TrainingProgress::Sample {
                            step,
                            index: i as u32 + 1,
                            total,
                            prompt: prompt.clone(),
                            image,
                        }),
                        Err(e) => {
                            eprintln!(
                                "[sc-5637] {MODEL_ID} preview sample failed at step {step} \
                                 (prompt {}): {e} — skipping this preview, training continues",
                                i + 1
                            );
                            warnings.push(gen_core::train::preview_sample_failed(
                                step,
                                i as u32 + 1,
                                &e,
                            ));
                        }
                    }
                }
            }
        }

        // Cancelled before completing a single step (`steps == 0` is rejected upstream by
        // `validate`): the LoRA factors are still freshly initialized with `B = 0`, a no-op adapter.
        // Surface the typed `Error::Canceled` (sc-4895, bridged 1:1 to `gen_core::Error::Canceled`)
        // rather than writing a valid-looking `.safetensors` and returning `Ok` — downstream tooling
        // would otherwise ship an identity LoRA as a trained artifact (F-040).
        if steps_run == 0 {
            return Err(mlx_gen::Error::Canceled);
        }

        // --- save final adapter ---
        on_progress(TrainingProgress::Saving);
        std::fs::create_dir_all(&req.output_dir)?;
        let adapter_path = req.output_dir.join(&req.file_name);
        save_lora(&params, &targets, alpha, cfg.rank, &adapter_path)?;
        Ok(TrainingOutput {
            adapter_path,
            steps: steps_run,
            final_loss: last_loss,
        })
    }
}

/// Flatten a single-frame VAE latent `(1, 128, 1, le, le)` to the patchified `(1, S, 128)` the DiT
/// consumes (`S = le·le`) — the reference's `transpose(reshape(latent, (B, C, -1)), (0, 2, 1))`.
pub(super) fn flatten_latent(latent: &Array) -> Result<Array> {
    let sh = latent.shape(); // [1, 128, 1, le, le]
    let (b, c) = (sh[0], sh[1]);
    let s = sh[2] * sh[3] * sh[4];
    let flat = latent.reshape(&[b, c, s])?; // (1, 128, S)
    Ok(flat.transpose_axes(&[0, 2, 1])?) // (1, S, 128)
}

/// The inverse of [`flatten_latent`] for a square single-frame latent: the patchified
/// `(1, S, 128)` with `S = latent_edge²` back to the decoder's NCHW `(1, 128, le, le)` (the
/// per-channel-normalized space `LtxVideoVae::encode` produces).
pub(super) fn unflatten_latent(flat: &Array, latent_edge: usize) -> Result<Array> {
    let sh = flat.shape();
    let le = latent_edge as i32;
    if sh.len() != 3 || sh[1] != le * le {
        return Err(mlx_gen::Error::Msg(format!(
            "ltx_2_3 trainer: expected a patchified (B, {}, C) still latent, got {sh:?}",
            le * le
        )));
    }
    Ok(flat
        .transpose_axes(&[0, 2, 1])?
        .reshape(&[sh[0], sh[2], le, le])?)
}

/// One LTX-2.3 item's cached latents, one per bucket edge in `edges` order (the item-major layout
/// [`BucketSchedule::cache_index`] indexes, sc-2127): `encode(edge)` the unpatchified
/// `(1, 128, 1, le, le)` latent, `mask_weight(latent.shape())` — the item's subject mask resampled
/// onto **that** latent's grid (sc-24828) — then both patchified by the same [`flatten_latent`] →
/// `(1, S, 128)` so the weight lines up token-for-token with that bucket's clean latent; each pair
/// evaluated so it survives the encoder's drop.
pub(super) fn encode_bucket_latents(
    edges: &[u32],
    mut encode: impl FnMut(u32) -> Result<Array>,
    mut mask_weight: impl FnMut(&[i32]) -> Result<Option<Array>>,
) -> Result<Vec<(Array, Option<Array>)>> {
    edges
        .iter()
        .map(|&edge| {
            let latent = encode(edge)?;
            let weight = mask_weight(latent.shape())?
                .map(|w| flatten_latent(&w))
                .transpose()?;
            let clean = flatten_latent(&latent)?;
            eval(std::iter::once(&clean).chain(weight.as_ref()))?;
            Ok((clean, weight))
        })
        .collect()
}

/// The subject-masked loss weight (sc-24828) of one unpatchified still latent
/// `(1, 128, 1, le, le)`: the image went through `center_crop_square`, so the mask's crop is
/// [`CropBox::center_square`], area-averaged onto the latent's `(le, le)` grid and broadcast over
/// every channel — the latent's exact shape. `None` when off.
pub(super) fn still_subject_weight(
    label: &str,
    mask: Option<&PreparedSubjectMask>,
    latent_shape: &[i32],
) -> Result<Option<Array>> {
    prepared_subject_mask_weight(label, mask, CropBox::center_square, latent_shape)
}

/// Enumerate the `attn1`/`attn2` × `suffixes` targets across the DiT's `num_layers` blocks, resolve
/// each on the (mutable) DiT, read its `[out,in]` base shape, and initialise the trainable factors
/// the reference `_MlxLoRALinear` way — `A ~ N(0, 0.02)` `[rank,in]`, `B = 0` `[out,rank]` — keyed
/// `{save_path}.lora_a` / `.lora_b`. Targets that do not resolve (a missing gated branch, a typo'd
/// suffix) are skipped.
pub(super) fn build_targets(
    dit: &mut LtxDiT,
    num_layers: i32,
    suffixes: &[String],
    rank: i32,
    seed: u64,
) -> Result<(Vec<LtxLoraTarget>, LoraParams)> {
    let mut paths = Vec::new();
    for i in 0..num_layers {
        for attn in ["attn1", "attn2"] {
            for suf in suffixes {
                paths.push(format!("transformer_blocks.{i}.{attn}.{suf}"));
            }
        }
    }
    build_targets_from_paths(dit, paths, rank, seed)
}

/// [`compute_loss_grads`] with the step's perceptual plan (epic 2123 E8): on an aux-only step the
/// diffusion term is not computed and the loss is the weighted perceptual term on the DiT's x0
/// estimate `x_t − σ·v` (the raw output regresses `noise − clean`), unpatchified to the decoder's
/// NCHW layout and decoded by TAELTX2.3; summed (`every_n == 1`) aux terms ride along on a
/// diffusion step. The dense and block-checkpointed forwards land in the same closure, so both
/// carry it. With `aux = None` (or a plan with no aux loss) the traced graph is exactly the
/// pre-epic-2123 one.
#[allow(clippy::too_many_arguments)]
pub(super) fn compute_step_loss_grads(
    dit: &mut LtxDiT,
    params: &LoraParams,
    targets: &[LtxLoraTarget],
    alpha: f32,
    rank: f32,
    clean: &Array,
    context: &Array,
    positions: &Array,
    sigma: f32,
    noise: &Array,
    mae: bool,
    mask_weight: Option<&Array>,
    checkpoint_block: Option<&[Vec<BlockLoraRef>]>,
    dtype: Dtype,
    aux: Option<AuxStep<'_>>,
) -> Result<(StepLosses, LoraParams)> {
    let (diffusion_on, aux_on) = match &aux {
        Some(a) => (a.plan.diffusion, !a.plan.aux.is_empty()),
        None => (true, false),
    };
    // x_t = (1-σ)·clean + σ·noise; target = noise - clean (the raw-output velocity); timestep = σ.
    // x_t / context stay f32 here; `preprocess` casts the activation stream to the compute dtype.
    let one_minus = Array::from_slice(&[1.0 - sigma], &[1]);
    let s = Array::from_slice(&[sigma], &[1]);
    let x_t = add(&multiply(clean, &one_minus)?, &multiply(noise, &s)?)?;
    let x_t_f32 = x_t.clone();
    let target = subtract(noise, clean)?;
    let timestep = Array::from_slice(&[sigma], &[1, 1]); // (B, 1), broadcast over tokens
    let ctx = context.clone();
    let pos = positions.clone();
    let mask_weight = mask_weight.cloned();
    let lora_dtype = (dtype != Dtype::Float32).then_some(dtype);
    let loss_fn = move |p: LoraParams, _: i32| -> MlxResult<Vec<Array>> {
        let v = match checkpoint_block {
            Some(bt) => dit
                .forward_with_main_checkpointed(&x_t, &timestep, &ctx, None, &pos, &p, bt, alpha)
                .map_err(|e| Exception::custom(e.to_string()))?,
            None => {
                install_train_lora(dit, &p, targets, alpha, rank, lora_dtype)?;
                // `None`: content-keyed RoPE memo (the per-stage epoch fast path is inference-only;
                // training positions are constant within a step, so the content compare hits — sc-7141).
                dit.forward(&x_t, &timestep, &ctx, None, &pos, None)
                    .map_err(|e| Exception::custom(e.to_string()))?
            }
        };
        let diffusion = if diffusion_on {
            let diff = subtract(&v, &target)?;
            // MSE / MAE, subject-mask weighted when on (sc-24828) — reduces to a 0-d scalar (grad
            // requires a scalar cotangent). Dense and block-checkpointed forwards both land here.
            Some(reduce_loss(&diff, mask_weight.as_ref(), mae)?)
        } else {
            None
        };
        let aux_term = match &aux {
            Some(a) if aux_on => {
                // x0 estimate in f32 from the raw velocity (`noise − clean`): x0 = x_t − σ·v.
                let x0_hat = Parameterization::FlowNoiseMinusX0 { sigma }
                    .recover_x0(&x_t_f32, &v.as_dtype(Dtype::Float32)?)
                    .and_then(|x| unflatten_latent(&x, a.latent_edge))
                    .map_err(|e| Exception::custom(e.to_string()))?;
                a.path
                    .aux_loss(a.plan, a.image, &x0_hat)
                    .map_err(|e| Exception::custom(e.to_string()))?
                    .map(|t| t.weighted)
            }
            _ => None,
        };
        // Only the first output is differentiated; the other two are reported terms.
        let zero = || Array::from_f32(0.0);
        let d_out = diffusion.clone().unwrap_or_else(zero);
        let a_out = aux_term.clone().unwrap_or_else(zero);
        let total =
            combine_step_loss(diffusion, aux_term).map_err(|e| Exception::custom(e.to_string()))?;
        Ok(vec![total, d_out, a_out])
    };
    let mut vg = keyed_value_and_grad(loss_fn);
    let (val, grads) = vg(params.clone(), 0)?;
    let losses = StepLosses {
        total: val[0].item::<f32>(),
        diffusion: diffusion_on.then(|| val[1].item::<f32>()),
        aux: aux_on.then(|| val[2].item::<f32>()),
    };
    Ok((losses, grads))
}
