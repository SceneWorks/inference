//! **MiniMax-H3-Image LoRA/LoKr training** on MLX (SW-7097) — the `minimax_h3_image` trainer:
//! text-to-image on captioned datasets and edit on edit-pair datasets (1–16 ordered references
//! and a target), over a frozen — optionally packed q4/q8 — 33 B DiT.
//!
//! The backend-neutral half (identity, trainable surface, the flow-matching contract, the training
//! canvas and the train-step memory envelope) is [`gen_core_train::h3_image`], shared with the
//! Candle trainer. This module is the MLX half; it realizes the core [`Trainer`] contract the way
//! every MLX family trainer does — trainable factors outside the model in a [`LoraParams`] map,
//! re-installed on the target [`AdaptableLinear`](mlx_gen::adapters::AdaptableLinear)s each step,
//! stepped by the shared [`run_train_loop`] (LR schedule, accumulation, checkpoints + resume,
//! cancel, previews, progress).
//!
//! # Fitting a 33 B DiT by design
//!
//! Nothing heavy is ever co-resident, and nothing heavy is ever whole:
//!
//! 1. **Preflight** ([`memory`]) — the run's peak, derived from the staged tier's headers the way
//!    inference admission derives a render's, is checked against this device before any weight is
//!    read.
//! 2. **Captions** — H3's Qwen3-VL-32B text encoder, streamed one layer at a time (the generator's
//!    own rung-4 encoder), encodes every caption once; an edit item also runs the vision tower over
//!    its references, exactly as the render presents them.
//! 3. **Latents** — the image VAE's encoder encodes every target (once per bucket edge, the
//!    posterior mode) and every reference (the render's own posterior sample), then is released.
//! 4. **Train** — the DiT loads **deferred**: only its trunk (I/O projections, the six heads, the
//!    token refiner) is mapped; the 50 blocks stream from the staged tier — dense bf16 or packed
//!    q4/q8, used in its published form — one at a time, forward and again in reverse for the
//!    backward ([`step`]). The AdaLN modulation of the six published timesteps is projected once,
//!    a projection at a time. Gradient checkpointing is therefore structural: each block's input is
//!    all the forward keeps. `gradient_checkpointing = false` is honored as the resident
//!    alternative (whole DiT mapped, every activation retained), and the preflight refuses it
//!    unless the device holds it.
//!
//! # The saved adapter
//!
//! PEFT keys `transformer_blocks.{i}.<leaf>.lora_A.weight` / `.lora_B.weight` + a per-target
//! `.alpha` (LoRA), or the LyCORIS `lokr_*` factors (LoKr), with `networkType` / `rank` / `alpha`
//! and the [`gen_core_train::h3_image::ADAPTER_PROVENANCE`] metadata — the layout the SW-7096
//! H3-Image adapter install reads on both backends, with no key conversion. The install mirrors
//! the loader op for op (`alpha / rank` folded into `B`, residual over the untouched base on every
//! tier), so a saved adapter renders what the trainer trained.
//!
//! # Previews
//!
//! A preview is rendered by the **registered generator itself** with the in-progress factors
//! saved as an adapter file — the same load, install and sampler a user's render runs — so a
//! preview at the final step is, by construction, what the saved adapter renders.

pub mod memory;
mod step;

use std::path::{Path, PathBuf};
use std::rc::Rc;

use mlx_rs::ops::concatenate_axis;
use mlx_rs::{Array, Dtype};

use mlx_gen::adapters::AdaptableHost;
use mlx_gen::gen_core::h3_image::{
    reference_vae_size, reference_vision_size, t2i_prompt_ids, TEXT_TAG,
};
use mlx_gen::gen_core::{
    self, AdapterKind, AdapterSpec, BucketSchedule, Conditioning, GenerationMemory,
    GenerationOutput, GenerationRequest, LoadShape, LoadSpec, Precision, TransformerComponent,
};
use mlx_gen::media::Image;
use mlx_gen::train::checkpoint::{self, checkpoint_filename};
use mlx_gen::train::dataset::{bucket_edges, center_crop_square, decode_image};
use mlx_gen::train::lora::{
    build_lokr_targets, build_lora_targets, factorization, LoraParams, TrainAdapter,
};
use mlx_gen::train::trainer_loop::{
    adapter_optimizer, adapter_stem, build_adapter, cache_items, draw_flow_step, ensure_cached,
    largest_edge, mae_loss, render_preview_prompts, resolve_compute_dtype, run_train_loop,
    AdapterSaver, CheckpointAt, LoopStart, LossTypeFold, PreviewLog, PreviewSink, TrainLoopFamily,
    SAMPLE_PROMPT_CAP,
};
use mlx_gen::train::vocab::validate_common_request;
use mlx_gen::{
    CancelFlag, Error, NetworkType, Result, TrainOptimizer, Trainer, TrainerDescriptor,
    TrainingConfig, TrainingItem, TrainingOutput, TrainingProgress, TrainingReport,
    TrainingRequest,
};

use gen_core_train::h3_image::{
    adapter_surface, target_block, trainer_descriptor, training_canvas, training_step,
    ADAPTER_PROVENANCE, EDIT_ADAPTER_MARKER, TRAINER_ID,
};

use crate::dit::config::MiniMaxH3ImageDitConfig;
use crate::h3_image::admission::ImageMemoryFacts;
use crate::h3_image::dit::{MiniMaxH3ImageDit, Modulation};
use crate::h3_image::layout::ImageLayout;
use crate::h3_image::model::{
    load_image, reference_condition_rows, reference_noise, reference_vae_pixels, resize_reference,
    resize_to_request, MiniMaxH3Image,
};
use crate::image_vae::MiniMaxH3ImageVaeEncoder;
use crate::pipeline::patchify_video_latents;
use crate::text_encoder::MiniMaxH3Tokenizer;

use memory::{
    admit_training, budget_of, contract_and_tier, device_budget, estimate_training,
    TrainingItemShape, TrainingRunShape,
};
use step::{loss_and_grads, noised_rows, BlockTrainables, LossSpec, StepInputs};

/// The streamed window: one block (and one text-encoder layer) at a time.
const WINDOW: usize = 1;

/// The production [`Trainer`] for `minimax_h3_image`.
pub struct MiniMaxH3ImageTrainer {
    descriptor: TrainerDescriptor,
    /// The trainer's load, at [`LoadShape::DeferredMaterialization`] (every phase streams).
    spec: LoadSpec,
    /// The generator over the same snapshot: its phases encode the dataset and render previews.
    generator: MiniMaxH3Image,
    /// The DiT trunk mapped lazily (no tensor read) with the blocks described by the stream —
    /// target resolution and base shapes for `validate` and the preflight.
    probe: MiniMaxH3ImageDit,
    facts: ImageMemoryFacts,
    /// Test seam: replaces [`device_budget`].
    memory_budget_override: Option<u64>,
}

/// Construct the trainer from a `MiniMaxAI/MiniMax-H3-Image` snapshot (the staged DiT tier as the
/// `transformer_diffusers` component, MiniMax-H3's `text_encoder/` as the `text_encoder` component
/// when the snapshot lacks it). Reads configs and headers only.
pub fn load_trainer(spec: &LoadSpec) -> Result<Box<dyn Trainer>> {
    Ok(Box::new(MiniMaxH3ImageTrainer::load(spec)?))
}

mlx_gen::register_trainer! {
    pub(crate) const TRAINER_REGISTRATION = descriptor => load_trainer
}

fn descriptor() -> TrainerDescriptor {
    trainer_descriptor("mlx")
}

impl MiniMaxH3ImageTrainer {
    /// [`load_trainer`] without the box.
    pub fn load(spec: &LoadSpec) -> Result<Self> {
        if spec.precision != Precision::Bf16 {
            return Err(Error::Unsupported(format!(
                "{TRAINER_ID} trainer: the compute dtype comes from TrainingConfig::train_dtype; \
                 drop the LoadSpec precision override"
            )));
        }
        if spec.text_encoder.is_some() {
            return Err(Error::Unsupported(format!(
                "{TRAINER_ID} trainer: the text encoder is staged as the 'text_encoder' \
                 component; LoadSpec::text_encoder substitution is not supported"
            )));
        }
        if !spec.adapters.is_empty() {
            return Err(Error::Unsupported(format!(
                "{TRAINER_ID} trainer: training starts from the bare base; LoadSpec adapters \
                 would not be part of the trained adapter, so they are refused rather than \
                 silently dropped"
            )));
        }
        gen_core_train::refuse_trainer_load_overlays(&format!("{TRAINER_ID} trainer"), spec)?;
        let spec = spec
            .clone()
            .with_load_shape(LoadShape::DeferredMaterialization);
        let generator = load_image(&spec)?;
        let probe = MiniMaxH3ImageDit::load_dir_deferred(generator.dit_dir(), Dtype::Bfloat16)?;
        let facts = ImageMemoryFacts::resolve(&spec)?;
        Ok(Self {
            descriptor: descriptor(),
            spec,
            generator,
            probe,
            facts,
            memory_budget_override: None,
        })
    }

    /// Replace the device budget the preflight compares against (bytes). A test seam.
    pub fn set_memory_budget_override(&mut self, bytes: Option<u64>) {
        self.memory_budget_override = bytes;
    }
}

/// The trainable target paths of `cfg` on the shipped geometry, in block order.
fn resolve_target_paths(cfg: &TrainingConfig) -> Result<Vec<String>> {
    let c = MiniMaxH3ImageDitConfig::default();
    let surface = adapter_surface(
        c.base.num_layers as usize,
        c.base.num_refiner_layers as usize,
        c.num_pdd_heads as usize,
    );
    Ok(gen_core_train::h3_image::resolve_target_paths(
        &surface, cfg,
    )?)
}

/// `(trainable elements, largest block's LoKr delta, all LoKr deltas)` for `paths` under `cfg` —
/// exact, from the host's base shapes (LoRA `rank·(in + out)`; LoKr `w1` plus a full or low-rank
/// `w2` by PEFT's `use_w2` rule, exactly as [`build_lokr_targets`] sizes them).
fn adapter_elements(
    host: &mut MiniMaxH3ImageDit,
    paths: &[String],
    cfg: &TrainingConfig,
) -> Result<(u64, u64, u64)> {
    let rank = i64::from(cfg.rank);
    let mut total = 0u64;
    let mut per_block = std::collections::BTreeMap::<usize, u64>::new();
    for path in paths {
        let segs: Vec<&str> = path.split('.').collect();
        let facts = host.adaptable_facts(&segs).ok_or_else(|| {
            Error::Msg(format!(
                "{TRAINER_ID} trainer: target {path} does not resolve on the DiT"
            ))
        })?;
        let (out_f, in_f) = (
            i64::from(facts.base_shape[0]),
            i64::from(facts.base_shape[1]),
        );
        let elements = match cfg.network_type {
            NetworkType::Lora => rank * (in_f + out_f),
            NetworkType::Lokr => {
                let (out_a, out_b) = factorization(out_f as i32, cfg.decompose_factor);
                let (in_a, in_b) = factorization(in_f as i32, cfg.decompose_factor);
                let (out_a, out_b, in_a, in_b) = (
                    i64::from(out_a),
                    i64::from(out_b),
                    i64::from(in_a),
                    i64::from(in_b),
                );
                let w2 = if (rank as f32) < (out_b.max(in_b) as f32) / 2.0 {
                    rank * (out_b + in_b)
                } else {
                    out_b * in_b
                };
                out_a * in_a + w2
            }
        };
        total += elements.max(0) as u64;
        if cfg.network_type == NetworkType::Lokr {
            *per_block
                .entry(target_block(path).unwrap_or(0))
                .or_default() += (out_f * in_f).max(0) as u64;
        }
    }
    let block_max = per_block.values().copied().max().unwrap_or(0);
    Ok((total, block_max, per_block.values().sum()))
}

/// Per block, an adapter over that block's targets (bookkeeping only — the live factors come from
/// `params` under the same keys) and the factor keys it reads.
fn block_trainables(
    dit: &mut MiniMaxH3ImageDit,
    paths: &[String],
    params: &LoraParams,
    cfg: &TrainingConfig,
) -> Result<Vec<BlockTrainables>> {
    let mut out = Vec::with_capacity(dit.num_layers());
    for index in 0..dit.num_layers() {
        let local: Vec<String> = paths
            .iter()
            .filter(|p| target_block(p) == Some(index))
            .cloned()
            .collect();
        if local.is_empty() {
            out.push(BlockTrainables {
                adapter: None,
                keys: Vec::new(),
            });
            continue;
        }
        let adapter = match cfg.network_type {
            NetworkType::Lora => TrainAdapter::Lora {
                targets: build_lora_targets(dit, &local, cfg.rank as i32, 0)?.0,
            },
            NetworkType::Lokr => TrainAdapter::Lokr {
                targets: build_lokr_targets(dit, &local, cfg.rank as i32, cfg.decompose_factor, 0)?
                    .0,
            },
        };
        let mut keys: Vec<Rc<str>> = params
            .keys()
            .filter(|key| {
                local.iter().any(|path| {
                    key.strip_prefix(path.as_str())
                        .and_then(|rest| rest.strip_prefix('.'))
                        .is_some_and(|factor| !factor.contains('.'))
                })
            })
            .cloned()
            .collect();
        keys.sort();
        out.push(BlockTrainables {
            adapter: Some(adapter),
            keys,
        });
    }
    Ok(out)
}

/// The request-derived setup every phase of one run reads.
struct RunPlan<'r> {
    req: &'r TrainingRequest,
    edit: bool,
    target_paths: Vec<String>,
    compute_dtype: Dtype,
    lora_dtype: Option<Dtype>,
    checkpointed: bool,
    sample_prompts: Vec<String>,
    edges: Vec<u32>,
    edge: u32,
}

/// One cached `(item, edge)` entry.
struct CachedItem {
    /// The caption's layer-50 context `[1, T, 5120]` (shared across an item's edges).
    context: Array,
    /// The packed `[1, rows, 96]` target latent (normalized posterior mode), float32.
    x0: Array,
    /// The target's `[1, 24, 1, h, w]` latent shape, for the step's noise draw.
    latent_shape: [i32; 5],
    /// An edit's reference latents `[1, 24, h, w]`, in order (empty for a captioned item).
    references: Vec<Array>,
    layout: ImageLayout,
}

impl MiniMaxH3ImageTrainer {
    fn plan_run<'r>(&self, req: &'r TrainingRequest) -> Result<RunPlan<'r>> {
        gen_core_train::h3_image::validate_floors(&self.descriptor, req)?;
        validate_common_request(&format!("{TRAINER_ID} trainer"), req)?;
        let cfg = &req.config;
        let target_paths = resolve_target_paths(cfg)?;
        let compute_dtype = resolve_compute_dtype(&cfg.train_dtype);
        let sample_prompts = if cfg.sample_every > 0 {
            cfg.sample_prompts
                .iter()
                .take(SAMPLE_PROMPT_CAP)
                .cloned()
                .collect()
        } else {
            Vec::new()
        };
        let edges = bucket_edges(cfg);
        Ok(RunPlan {
            req,
            edit: req.items.iter().any(TrainingItem::is_edit_pair),
            target_paths,
            compute_dtype,
            lora_dtype: (compute_dtype != Dtype::Float32).then_some(compute_dtype),
            checkpointed: cfg.gradient_checkpointing,
            sample_prompts,
            edge: largest_edge(&edges),
            edges,
        })
    }

    /// The run's shape as the estimate reads it — image headers and caption token counts only.
    fn run_shape(&mut self, plan: &RunPlan<'_>) -> Result<TrainingRunShape> {
        let tok = MiniMaxH3Tokenizer::from_snapshot(self.generator.root())?;
        let mut items = Vec::with_capacity(plan.req.items.len());
        for item in &plan.req.items {
            let dims = image_dimensions(&item.image_path)?;
            let references = item
                .reference_image_paths
                .iter()
                .map(|p| image_dimensions(p))
                .collect::<Result<Vec<_>>>()?;
            items.push(TrainingItemShape {
                canvas: training_canvas(item.is_edit_pair(), dims, plan.edge),
                caption_tokens: t2i_prompt_ids(tok.ids(&item.caption)?)?.len() as u32,
                references,
            });
        }
        let cfg = &plan.req.config;
        let (trainable, block_max, total) =
            adapter_elements(&mut self.probe, &plan.target_paths, cfg)?;
        let preview = (!plan.sample_prompts.is_empty()).then(|| {
            let longest = plan
                .sample_prompts
                .iter()
                .map(|p| tok.ids(p).map(|ids| ids.len().min(4096) as u32))
                .collect::<Result<Vec<u32>>>();
            longest.map(|l| TrainingItemShape {
                caption_tokens: l.into_iter().max().unwrap_or(0),
                ..items[0].clone()
            })
        });
        Ok(TrainingRunShape {
            items,
            edges: plan.edges.len() as u32,
            width: if plan.compute_dtype == Dtype::Float32 {
                4
            } else {
                2
            },
            trainable_elements: trainable,
            optimizer_state_per_element: optimizer_state_per_element(&cfg.optimizer),
            lokr_delta_block_max: block_max,
            lokr_delta_total: total,
            checkpointed: plan.checkpointed,
            chunked_attention: true,
            preview: preview.transpose()?,
        })
    }

    fn preflight(&mut self, plan: &RunPlan<'_>) -> Result<()> {
        let run = self.run_shape(plan)?;
        let (contract, tier) = contract_and_tier(&self.spec)?;
        let estimate = estimate_training(&contract, &self.facts, tier, &run)?;
        let budget = self
            .memory_budget_override
            .map_or_else(device_budget, budget_of);
        admit_training(&estimate, budget, &run, plan.edge)?;
        Ok(())
    }

    /// Every caption (and an edit's references) through the generator's own conditioning phase,
    /// the text encoder streamed; then every target and reference latent through the VAE encoder.
    fn cache(
        &self,
        plan: &RunPlan<'_>,
        on_progress: &mut dyn FnMut(TrainingProgress),
    ) -> Result<Vec<CachedItem>> {
        let req = plan.req;
        let cfg = &req.config;
        let patch = MiniMaxH3ImageDitConfig::default().base.patch_size;
        let mut contexts: Vec<(Array, Vec<i32>, Vec<Image>)> = Vec::with_capacity(req.items.len());
        let mut no_progress = |_: gen_core::Progress| {};
        for item in &req.items {
            if req.cancel.is_cancelled() {
                return Err(Error::Canceled);
            }
            if item.is_edit_pair() {
                let references = item
                    .reference_image_paths
                    .iter()
                    .map(|p| decode_image(p))
                    .collect::<Result<Vec<_>>>()?;
                let vision = references
                    .iter()
                    .map(|r| resize_reference(r, reference_vision_size(r.width, r.height)))
                    .collect::<Result<Vec<_>>>()?;
                let (context, tags) = self.generator.encode_edit_prompt(
                    &item.caption,
                    &vision,
                    Some(WINDOW as u32),
                    &req.cancel,
                    &mut no_progress,
                )?;
                contexts.push((context, tags, references));
            } else {
                let context = self.generator.encode_prompt(
                    &item.caption,
                    Some(WINDOW as u32),
                    &req.cancel,
                    &mut no_progress,
                )?;
                let n = context.shape()[1] as usize;
                contexts.push((context, vec![TEXT_TAG; n], Vec::new()));
            }
        }
        mlx_rs::memory::clear_cache();

        let encoder = MiniMaxH3ImageVaeEncoder::load(self.generator.root())?;
        let mut cache = Vec::with_capacity(req.items.len() * plan.edges.len());
        let mut contexts = contexts.into_iter();
        cache_items(req, on_progress, |item| {
            let (context, tags, references) = contexts
                .next()
                .expect("one encoded caption per dataset item");
            // The references' latents: the render's own resize and posterior sample.
            let reference_latents = if references.is_empty() {
                Vec::new()
            } else {
                let vae_images = references
                    .iter()
                    .map(|r| resize_reference(r, reference_vae_size(r.width, r.height)))
                    .collect::<Result<Vec<_>>>()?;
                let mut latents = Vec::with_capacity(vae_images.len());
                for (i, image) in vae_images.iter().enumerate() {
                    let posterior = encoder.encode(&reference_vae_pixels(image)?)?;
                    let noise = reference_noise(cfg.seed, 0, i, posterior.mean().shape())?;
                    let latent =
                        encoder.reference_latent(&posterior, &noise, (patch[1], patch[2]))?;
                    mlx_rs::transforms::eval([&latent])?;
                    latents.push(latent);
                }
                latents
            };
            let target = decode_image(&item.image_path)?;
            for &edge in &plan.edges {
                let (w, h) =
                    training_canvas(item.is_edit_pair(), (target.width, target.height), edge);
                let image = if item.is_edit_pair() {
                    resize_to_request(target.clone(), w, h)?
                } else {
                    resize_to_request(center_crop_square(&target), w, h)?
                };
                let posterior = encoder.encode(&reference_vae_pixels(&image)?)?;
                // The posterior mode: the sample at zero noise, normalized like a reference.
                let zero = Array::zeros::<f32>(posterior.mean().shape())?;
                let latent = encoder.reference_latent(&posterior, &zero, (patch[1], patch[2]))?;
                let s = latent.shape().to_vec();
                let latent_shape = [s[0], s[1], 1, s[2], s[3]];
                let x0 = patchify_video_latents(&latent.reshape(&latent_shape)?, patch)?
                    .as_dtype(Dtype::Float32)?;
                mlx_rs::transforms::eval([&x0])?;
                let layout = if reference_latents.is_empty() {
                    ImageLayout::t2i(context.shape()[1], s[2], s[3], patch)?
                } else {
                    let grids: Vec<(i32, i32)> = reference_latents
                        .iter()
                        .map(|l| (l.shape()[2], l.shape()[3]))
                        .collect();
                    ImageLayout::edit(&tags, &grids, s[2], s[3], patch)?
                };
                cache.push(CachedItem {
                    context: context.clone(),
                    x0,
                    latent_shape,
                    references: reference_latents.clone(),
                    layout,
                });
            }
            Ok(())
        })?;
        ensure_cached(cache.len(), &req.cancel, &format!("{TRAINER_ID} trainer"))?;
        drop(encoder);
        mlx_gen::residency::drain_allocator_cache();
        if req.cancel.is_cancelled() {
            return Err(Error::Canceled);
        }
        Ok(cache)
    }

    fn train_impl(
        &mut self,
        req: &TrainingRequest,
        on_progress: &mut dyn FnMut(TrainingProgress),
        warnings: &mut Vec<gen_core::GenerationWarning>,
    ) -> Result<TrainingOutput> {
        let plan = self.plan_run(req)?;
        let cfg = &req.config;
        on_progress(TrainingProgress::Preparing);
        self.preflight(&plan)?;
        let admission = self.admit_resume(&plan)?;

        on_progress(TrainingProgress::LoadingModel);
        let cache = self.cache(&plan, on_progress)?;

        // The DiT: deferred (trunk mapped, blocks streamed) or, without checkpointing, resident.
        let dit_dir = self.generator.dit_dir().to_path_buf();
        let mut dit = if plan.checkpointed {
            MiniMaxH3ImageDit::load_dir_deferred(&dit_dir, plan.compute_dtype)?
        } else {
            MiniMaxH3ImageDit::load_dir(&dit_dir, plan.compute_dtype)?
        };
        let modulation = dit.training_modulation(
            plan.edit,
            plan.checkpointed.then_some((WINDOW, &req.cancel)),
        )?;

        let (adapter, params) = build_adapter(&mut dit, &plan.target_paths, cfg)?;
        let blocks = block_trainables(&mut dit, &plan.target_paths, &params, cfg)?;
        let identity = checkpoint::ResumeIdentity {
            config: cfg,
            request_fingerprint: &admission.fingerprint,
        };
        let start = resume_admitted(&admission, identity, adapter_optimizer(cfg)?, params)?;
        let provenance: Vec<(&str, &str)> = ADAPTER_PROVENANCE
            .iter()
            .copied()
            .chain(plan.edit.then_some(EDIT_ADAPTER_MARKER))
            .collect();
        let spec = LossSpec {
            adapter: &adapter,
            blocks: &blocks,
            alpha: cfg.alpha,
            rank: cfg.rank as f32,
            mae: mae_loss(&cfg.loss_type, LossTypeFold::Normalized),
            lora_dtype: plan.lora_dtype,
            streamed: plan.checkpointed,
            chunked_attention: true,
        };
        let schedule = BucketSchedule::new(
            cache.len() / plan.edges.len(),
            &cfg.training_buckets(),
            cfg.seed,
        );
        let previews = PreviewSetup::new(self, &plan)?;
        let mut family = H3ImageLoop {
            dit: &mut dit,
            modulation: &modulation,
            req,
            saver: AdapterSaver {
                adapter: &adapter,
                cfg,
                provenance: &provenance,
            },
            spec: &spec,
            identity,
            cache: &cache,
            schedule: &schedule,
            previews,
        };
        run_train_loop(
            &mut family,
            req,
            &admission.stem,
            start,
            on_progress,
            warnings,
        )
    }

    /// The run's identity: refuse a resume whose config, dataset or mode changed, from the
    /// snapshot's metadata alone, before any model loads.
    fn admit_resume(&self, plan: &RunPlan<'_>) -> Result<ResumeAdmission> {
        let (req, cfg) = (plan.req, &plan.req.config);
        let stem = adapter_stem(&req.file_name, "lora");
        let fingerprint = checkpoint::request_fingerprint(req)?;
        let resume_from = if cfg.resume {
            checkpoint::find_latest_resume(&req.output_dir, &stem)
        } else {
            None
        };
        if let Some((snapshot, step)) = &resume_from {
            let meta = gen_core::weightsmeta::safetensors_file_metadata(
                req.output_dir.join(checkpoint_filename(&stem, *step)),
            )?;
            check_resumed_mode(
                meta.get(EDIT_ADAPTER_MARKER.0).map(String::as_str),
                plan.edit,
            )?;
            checkpoint::check_resume_fingerprints(
                &gen_core::weightsmeta::safetensors_file_metadata(snapshot)?
                    .into_iter()
                    .collect::<std::collections::HashMap<_, _>>(),
                cfg,
                &fingerprint,
            )?;
        }
        Ok(ResumeAdmission {
            stem,
            fingerprint,
            resume_from,
        })
    }
}

/// `(width, height)` of an image file from its header.
fn image_dimensions(path: &Path) -> Result<(u32, u32)> {
    image::image_dimensions(path).map_err(|e| {
        Error::Msg(format!(
            "{TRAINER_ID} trainer: reading {}: {e}",
            path.display()
        ))
    })
}

/// f32 optimizer-state elements per trainable element: AdamW/Adam two, Rose none, Prodigy four
/// (names normalized as the optimizer picker does).
fn optimizer_state_per_element(optimizer: &str) -> u64 {
    let name: String = optimizer
        .trim()
        .to_ascii_lowercase()
        .chars()
        .filter(|c| *c != '-' && *c != '_')
        .collect();
    match name.as_str() {
        "rose" | "roseopt" => 0,
        "prodigy" | "prodigyopt" => 4,
        _ => 2,
    }
}

struct ResumeAdmission {
    stem: String,
    fingerprint: String,
    resume_from: Option<(PathBuf, u32)>,
}

fn check_resumed_mode(checkpoint_mode: Option<&str>, edit: bool) -> Result<()> {
    let resumed_edit = checkpoint_mode == Some(EDIT_ADAPTER_MARKER.1);
    if resumed_edit != edit {
        let name = |edit: bool| {
            if edit {
                "an edit"
            } else {
                "a text-to-image"
            }
        };
        return Err(Error::Msg(format!(
            "{TRAINER_ID} trainer: the resume snapshot is from {} run but this is {} run; start a \
             fresh run or resume with the original dataset",
            name(resumed_edit),
            name(edit)
        )));
    }
    Ok(())
}

/// A resumed snapshot must hold exactly this run's factors (same keys, same shapes).
fn check_resumed_params(built: &LoraParams, resumed: &LoraParams) -> Result<()> {
    let mismatch = built.len() != resumed.len()
        || built
            .iter()
            .any(|(key, value)| resumed.get(key).is_none_or(|r| r.shape() != value.shape()));
    if mismatch {
        return Err(Error::Msg(format!(
            "{TRAINER_ID} trainer: the resume snapshot's factors do not match this run's adapter \
             (rank, network type or target modules changed since it was written); start a fresh \
             run or resume with the original settings"
        )));
    }
    Ok(())
}

fn resume_admitted(
    admission: &ResumeAdmission,
    identity: checkpoint::ResumeIdentity<'_>,
    mut opt: TrainOptimizer,
    params: LoraParams,
) -> Result<LoopStart> {
    let Some((snapshot, _)) = &admission.resume_from else {
        return Ok(LoopStart::fresh(opt, params));
    };
    let (loaded, meta) = checkpoint::load_resume_with_identity(snapshot, &mut opt, identity)?;
    check_resumed_params(&params, &loaded)?;
    Ok(LoopStart {
        opt,
        params: loaded,
        start_step: meta.step,
        update_idx: meta.update_idx,
    })
}

/// The seed of micro-step `step`'s reference-augmentation noise: a pure function of the run seed
/// and the step, so a resumed run draws what an uninterrupted one would.
fn step_seed(seed: u64, step: u32) -> u64 {
    seed.wrapping_add(u64::from(step))
        .wrapping_mul(0xD1B5_4A32_D192_ED03)
}

/// Draw micro-step `step`'s inputs from `entry`: the noise level (the configured distribution,
/// quantized onto the six published steps), the noised target and, for an edit, the references
/// noised to `0.999` with this step's augmentation draw.
fn step_inputs(
    cfg: &TrainingConfig,
    step: u32,
    entry: &CachedItem,
    t: f32,
    noise: &Array,
) -> Result<(Array, Array, gen_core_train::h3_image::TrainingStep)> {
    let st = training_step(t);
    let noise = patchify_video_latents(noise, entry.layout.patch())?;
    let (x_t, target) = noised_rows(&entry.x0, &noise, st)?;
    let packed = if entry.references.is_empty() {
        x_t
    } else {
        let seed = step_seed(cfg.seed, step);
        let aug = entry
            .references
            .iter()
            .enumerate()
            .map(|(i, l)| reference_noise(seed, 1, i, l.shape()))
            .collect::<Result<Vec<_>>>()?;
        let condition = reference_condition_rows(&entry.references, &aug, entry.layout.patch())?;
        concatenate_axis(&[condition, x_t], 1)?
    };
    Ok((packed, target, st))
}

/// What a preview needs: the prompts, the request geometry, and (edit) item 0's references.
struct PreviewSetup {
    spec: LoadSpec,
    prompts: Vec<String>,
    size: (u32, u32),
    references: Vec<Image>,
    adapter_kind: AdapterKind,
}

impl PreviewSetup {
    fn new(trainer: &MiniMaxH3ImageTrainer, plan: &RunPlan<'_>) -> Result<Option<Self>> {
        if plan.sample_prompts.is_empty() {
            return Ok(None);
        }
        let first = &plan.req.items[0];
        let (size, references) = if plan.edit {
            let dims = image_dimensions(&first.image_path)?;
            let refs = first
                .reference_image_paths
                .iter()
                .map(|p| decode_image(p))
                .collect::<Result<Vec<_>>>()?;
            (training_canvas(true, dims, plan.edge), refs)
        } else {
            ((plan.edge, plan.edge), Vec::new())
        };
        Ok(Some(Self {
            spec: trainer.spec.clone(),
            prompts: plan.sample_prompts.clone(),
            size,
            references,
            adapter_kind: match plan.req.config.network_type {
                NetworkType::Lora => AdapterKind::Lora,
                NetworkType::Lokr => AdapterKind::Lokr,
            },
        }))
    }

    /// Render `prompt` at `seed` through the registered generator with the adapter at `adapter`:
    /// both transformers streamed, every other setting the generator's default.
    fn render(
        &self,
        adapter: &Path,
        prompt: &str,
        seed: u64,
        cancel: &CancelFlag,
    ) -> Result<Image> {
        let mut spec = self.spec.clone();
        spec.adapters = vec![AdapterSpec::new(
            adapter.to_path_buf(),
            1.0,
            self.adapter_kind,
        )];
        let generator = load_image(&spec)?;
        let req = preview_request(prompt, seed, self.size, &self.references, cancel.clone());
        match gen_core::Generator::generate(&generator, &req, &mut |_| {})? {
            GenerationOutput::Images(mut images) if images.len() == 1 => Ok(images.remove(0)),
            _ => Err(Error::Msg(format!(
                "{TRAINER_ID} trainer: the preview render returned no single image"
            ))),
        }
    }
}

/// The request a preview renders — and the one a user reproduces it with: the prompt at the
/// training canvas and `seed`, item 0's references for an edit, both transformers streamed one
/// block at a time.
pub fn preview_request(
    prompt: &str,
    seed: u64,
    (width, height): (u32, u32),
    references: &[Image],
    cancel: CancelFlag,
) -> GenerationRequest {
    GenerationRequest {
        prompt: prompt.to_owned(),
        width,
        height,
        count: 1,
        seed: Some(seed),
        conditioning: references
            .iter()
            .map(|image| Conditioning::Reference {
                image: image.clone(),
                strength: None,
            })
            .collect(),
        memory: Some(GenerationMemory {
            stage_residency: true,
            stream_transformer_blocks: true,
            transformer_window_size: Some(WINDOW as u32),
            transformer_window_component: Some(TransformerComponent::Both),
            ..Default::default()
        }),
        cancel,
        ..Default::default()
    }
}

/// The H3-Image [`TrainLoopFamily`].
struct H3ImageLoop<'a> {
    dit: &'a mut MiniMaxH3ImageDit,
    modulation: &'a Modulation,
    req: &'a TrainingRequest,
    saver: AdapterSaver<'a>,
    spec: &'a LossSpec<'a>,
    identity: checkpoint::ResumeIdentity<'a>,
    cache: &'a [CachedItem],
    schedule: &'a BucketSchedule,
    previews: Option<PreviewSetup>,
}

impl TrainLoopFamily for H3ImageLoop<'_> {
    fn micro_step(&mut self, step: u32, params: &LoraParams) -> Result<(f32, LoraParams)> {
        if self.req.cancel.is_cancelled() {
            return Err(Error::Canceled);
        }
        // The shared draw: the (item, bucket) entry, the noise level and the noise, all pure
        // functions of the seed and the step.
        let cache = self.cache;
        let draw = draw_flow_step(None, step, self.schedule, &self.req.config, |entry| {
            cache[entry].latent_shape
        })?;
        let entry = &cache[draw.entry];
        let (packed, target, st) = step_inputs(&self.req.config, step, entry, draw.t, &draw.noise)?;
        loss_and_grads(
            self.dit,
            self.modulation,
            params,
            self.spec,
            &StepInputs {
                context: &entry.context,
                layout: &entry.layout,
                packed: &packed,
                target: &target,
                step: st,
            },
        )
    }

    /// The adapter checkpoint and, on an optimizer-update boundary, the resume bundle (factors +
    /// optimizer state, never an in-flight accumulation window).
    fn save_checkpoint(&mut self, at: CheckpointAt<'_>) -> Result<()> {
        self.saver.save_checkpoint_adapter(&at)?;
        if at.step.is_multiple_of(at.accum) {
            checkpoint::save_resume_with_identity(
                at.output_dir,
                at.stem,
                at.step,
                at.update_idx,
                at.opt,
                at.params,
                self.identity,
            )?;
        }
        Ok(())
    }

    fn previews_enabled(&self) -> bool {
        self.previews.is_some()
    }

    /// Best-effort previews through the registered generator, the current factors saved as an
    /// adapter file beside the run's output (removed afterwards). A failure never aborts training.
    fn render_previews(
        &mut self,
        step: u32,
        params: &LoraParams,
        sink: &mut PreviewSink<'_>,
    ) -> Result<()> {
        let Some(previews) = &self.previews else {
            return Ok(());
        };
        std::fs::create_dir_all(&self.req.output_dir)?;
        let file = self.req.output_dir.join(format!(
            ".{}-preview.safetensors",
            adapter_stem(&self.req.file_name, "lora")
        ));
        self.saver.save(params, &file)?;
        render_preview_prompts(
            sink,
            step,
            self.req.config.seed,
            &self.req.cancel,
            &previews.prompts,
            |p| p.as_str(),
            &PreviewLog {
                tag: "[SW-7097]",
                trainer: TRAINER_ID,
                what: "preview sample",
                canceled_stops: true,
                report_failures: true,
            },
            |prompt, seed| previews.render(&file, prompt, seed, &self.req.cancel),
        );
        let _ = std::fs::remove_file(&file);
        mlx_gen::residency::drain_allocator_cache();
        Ok(())
    }

    fn save_final(&mut self, params: &LoraParams) -> Result<PathBuf> {
        self.saver.save_final(self.req, params)
    }
}

impl Trainer for MiniMaxH3ImageTrainer {
    fn descriptor(&self) -> &TrainerDescriptor {
        &self.descriptor
    }

    fn validate(&self, req: &TrainingRequest) -> gen_core::Result<()> {
        gen_core_train::h3_image::validate_floors(&self.descriptor, req)?;
        validate_common_request(&format!("{TRAINER_ID} trainer"), req)?;
        resolve_target_paths(&req.config)?;
        Ok(())
    }

    fn train(
        &mut self,
        req: &TrainingRequest,
        on_progress: &mut dyn FnMut(TrainingProgress),
    ) -> gen_core::Result<TrainingOutput> {
        self.train_with_report(req, on_progress)
            .map(|report| report.output)
    }

    fn train_with_report(
        &mut self,
        req: &TrainingRequest,
        on_progress: &mut dyn FnMut(TrainingProgress),
    ) -> gen_core::Result<TrainingReport> {
        self.validate(req)?;
        let mut warnings = Vec::new();
        let output = self.train_impl(req, on_progress, &mut warnings)?;
        Ok(TrainingReport { output, warnings })
    }
}

#[cfg(test)]
mod real_tests;
#[cfg(test)]
mod tests;
