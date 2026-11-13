//! **MiniMax-H3-Image LoRA/LoKr training** on candle/CUDA (SW-7098) — the `minimax_h3_image`
//! trainer: text-to-image on captioned datasets and edit on edit-pair datasets (1–16 ordered
//! references and a target), over a frozen — optionally packed q4/q8 — 33 B DiT. The twin of
//! `mlx_gen_minimax_h3::training` (SW-7097).
//!
//! The backend-neutral half (identity, trainable surface, the flow-matching contract, the training
//! canvas and the train-step memory envelope) is [`gen_core_train::h3_image`], shared with the MLX
//! trainer. This module is the candle half; it realizes the core [`Trainer`] contract over the
//! shared [`candle_gen::train`] harness: the trainable factors are `Var`-backed residuals pushed
//! onto the DiT's own [`AdaptLinear`] leaves (the SW-7096 loader's module paths), stepped by the
//! shared optimizer, LR schedule, accumulation, checkpoint and resume helpers.
//!
//! # Fitting a 33 B DiT by design
//!
//! Nothing heavy is ever co-resident on the device, and nothing heavy is ever whole there:
//!
//! 1. **Preflight** ([`memory`]) — the run's GPU and host peaks, derived from the staged tier's
//!    headers the way inference admission derives a render's, are checked against this device
//!    before any weight is read.
//! 2. **Captions** — H3's Qwen3-VL-32B text encoder, streamed one layer at a time (the generator's
//!    own rung-4 conditioning phase), encodes every caption once; an edit item also runs the
//!    vision tower over its references and VAE-encodes them, exactly as the render does.
//! 3. **Latents** — the image VAE's encoder encodes every target (once per bucket edge, the
//!    posterior mode), then is released.
//! 4. **Train** — the DiT loads in runtime format on the **host**; only its trunk (I/O
//!    projections, the six heads, the token refiner) moves to the device. The AdaLN modulation of
//!    the six published timesteps is projected once, one block's projection at a time, and every
//!    projection is evicted. The 50 blocks then stream into a one-block device window, forward and
//!    again in reverse for the backward ([`step`]), so gradient checkpointing is structural: each
//!    block's input is all the forward keeps. `gradient_checkpointing = false` is honored as the
//!    resident alternative (whole DiT on the device, every activation retained), and the preflight
//!    refuses it unless the device holds it.
//!
//! # Where this differs from the MLX twin
//!
//! * The stream is host-resident runtime-format blocks copied into a device window
//!   ([`MiniMaxH3Dit::window_block`]) rather than blocks rebuilt from the tier's shards, because
//!   CUDA memory is separate from the host's. The memory estimate prices the host stack in its own
//!   column.
//! * Training attention is bounded by design through candle's shared training SDPA
//!   ([`crate::nn::with_differentiable_attention`]): its reference owner keeps only q / k / v and the
//!   output, and recomputes the float32 probabilities one query tile at a time in the backward,
//!   where MLX chunks whole heads under a checkpoint. The estimate prices one tile, rounded up to
//!   whole heads, through the shared envelope.
//! * The trainable factors are installed once on the host blocks; a window copy carries them as
//!   the same `Var` leaves, so no per-step re-install exists.
//!
//! # The saved adapter
//!
//! PEFT keys `transformer_blocks.{i}.<leaf>.lora_A.weight` / `.lora_B.weight` + a per-target
//! `.alpha` (LoRA), or the LyCORIS `lokr_*` factors in PEFT's `use_w2` surface (LoKr), with
//! `networkType` / `rank` / `alpha` and the [`gen_core_train::h3_image::ADAPTER_PROVENANCE`]
//! metadata — the layout the SW-7096 H3-Image adapter install reads on both backends, with no key
//! conversion.
//!
//! # Previews
//!
//! A preview is rendered by the **registered generator itself** with the in-progress factors
//! saved as an adapter file — the same load, install and sampler a user's render runs — so a
//! preview at the final step is, by construction, what the saved adapter renders.

pub mod memory;
mod step;

use std::collections::HashMap;
use std::path::Path;

use candle_gen::candle_core::backprop::GradStore;
use candle_gen::candle_core::{DType, Device, Tensor};
use candle_gen::gen_core::h3_image::MAX_PROMPT_TOKENS;
use candle_gen::gen_core::train::{
    Trainer, TrainerDescriptor, TrainingConfig, TrainingItem, TrainingOutput, TrainingProgress,
    TrainingReport, TrainingRequest,
};
use candle_gen::gen_core::{
    self, AdapterKind, AdapterSpec, CancelFlag, Conditioning, GenerationMemory, GenerationOutput,
    GenerationRequest, Image, LoadShape, LoadSpec, NetworkType, Precision, TransformerComponent,
};
use candle_gen::quant::AdaptLinear;
use candle_gen::train::checkpoint::{
    checkpoint_filename, file_stem, find_latest_resume, load_resume, save_resume,
};
use candle_gen::train::dataset::{bucket_edges, decode_square};
use candle_gen::train::flow_match::{
    self, apply_update, create_output_dir, effective_weight_decay, item_major_schedule,
    request_fingerprint, sample_seed, save_adapter, validate_flow_match_request,
};
use candle_gen::train::lora::{
    build_adapt_lokr_targets_peft, build_adapt_lora_targets, factorization, AdaptLoraHost, LoraSet,
};
use candle_gen::train::optim::{accumulate_grads, TrainOptimizer};
use candle_gen::train::schedule::schedule_updates;
use candle_gen::{CandleError as Error, Result};

use gen_core_train::h3_image::{
    adapter_surface, target_block, trainer_descriptor, training_canvas, training_step,
    ADAPTER_PROVENANCE, EDIT_ADAPTER_MARKER, TRAINER_ID,
};

use crate::dit::model::{image_modulation, ImageModulation};
use crate::dit::{MiniMaxH3Dit, MiniMaxH3DitConfig};
use crate::image_memory_strategy::admission::ImageMemoryFacts;
use crate::image_model::{image_dit_config_at, MiniMaxH3Image};
use crate::image_pipeline::{
    reference_condition_rows, reference_noise, reference_vae_pixels, resize_reference, ImageLayout,
};
use crate::image_vae::MiniMaxH3ImageVaeEncoder;
use crate::pipeline::patchify_video_latents;
use crate::text_encoder::MiniMaxH3Tokenizer;

use memory::{
    admit_training, budget_of, contract_and_tier, estimate_training, TrainingItemShape,
    TrainingRunShape,
};
use step::{loss_and_grads, noised_rows, StepInputs};

/// The streamed window: one block (and one text-encoder layer) at a time.
const WINDOW: u32 = 1;

/// Max preview prompts rendered per sample cadence (the shared preview contract's cap).
const SAMPLE_PROMPT_CAP: usize = 4;

/// The production [`Trainer`] for `minimax_h3_image` on candle.
pub struct MiniMaxH3ImageTrainer {
    descriptor: TrainerDescriptor,
    /// The trainer's load, at [`LoadShape::DeferredMaterialization`] (every phase streams).
    spec: LoadSpec,
    /// The generator over the same snapshot: its phases encode the dataset and render previews.
    generator: MiniMaxH3Image,
    /// The staged DiT's geometry (shapes for `validate` and the preflight).
    dit_cfg: MiniMaxH3DitConfig,
    facts: ImageMemoryFacts,
    /// Test seam: replaces the device's training budget (bytes).
    memory_budget_override: Option<u64>,
}

/// Construct the trainer from a `MiniMaxAI/MiniMax-H3-Image` snapshot (the staged DiT tier as the
/// `transformer_diffusers` component, MiniMax-H3's `text_encoder/` as the `text_encoder` component
/// when the snapshot lacks it). Reads configs and headers only.
pub fn load_trainer(spec: &LoadSpec) -> Result<Box<dyn Trainer>> {
    Ok(Box::new(MiniMaxH3ImageTrainer::load(spec)?))
}

candle_gen::register_trainer! {
    pub(crate) const TRAINER_REGISTRATION = descriptor => load_trainer
}

fn descriptor() -> TrainerDescriptor {
    trainer_descriptor("candle")
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
        // The generator's own load boundary: the tier marker, the partitions, and — refused here,
        // before any weight — a source whose bases cannot migrate into the device window (the
        // native int8 / NVFP4 formats, whose kernels have no backward either).
        let generator = MiniMaxH3Image::load(&spec)?;
        let dit_cfg = image_dit_config_at(generator.dit_dir())?;
        let facts = ImageMemoryFacts::resolve(&spec)?;
        Ok(Self {
            descriptor: descriptor(),
            spec,
            generator,
            dit_cfg,
            facts,
            memory_budget_override: None,
        })
    }

    /// Replace the device budget the preflight compares against (bytes). A test seam.
    pub fn set_memory_budget_override(&mut self, bytes: Option<u64>) {
        self.memory_budget_override = bytes;
    }
}

/// The trainable target paths of `cfg` on `dit_cfg`, in block order — the shared resolver over
/// the SW-7096 loader's surface.
fn resolve_target_paths(dit_cfg: &MiniMaxH3DitConfig, cfg: &TrainingConfig) -> Result<Vec<String>> {
    let surface = adapter_surface(
        dit_cfg.num_layers,
        dit_cfg.num_refiner_layers,
        dit_cfg.num_pdd_heads().unwrap_or(0),
    );
    Ok(gen_core_train::h3_image::resolve_target_paths(
        &surface, cfg,
    )?)
}

/// `(out_features, in_features)` of a block leaf on `cfg`.
fn leaf_shape(cfg: &MiniMaxH3DitConfig, path: &str) -> Result<(usize, usize)> {
    let inner = cfg.num_attention_heads * cfg.attention_head_dim;
    let leaf = path
        .strip_prefix("transformer_blocks.")
        .and_then(|rest| rest.split_once('.'))
        .map_or("", |(_, leaf)| leaf);
    Ok(match leaf {
        "attn.to_q" | "attn.to_k" | "attn.to_v" => (inner, cfg.hidden_size),
        "attn.to_out.0" => (cfg.hidden_size, inner),
        "ff.net.0.proj" => (2 * cfg.ffn_dim, cfg.hidden_size),
        "ff.net.2" => (cfg.hidden_size, cfg.ffn_dim),
        _ => {
            return Err(Error::Msg(format!(
                "{TRAINER_ID} trainer: {path} is not a DiT block leaf"
            )))
        }
    })
}

/// `(trainable elements, largest block's LoKr delta, all LoKr deltas)` for `paths` under `cfg` —
/// exact, from the leaf shapes (LoRA `rank·(in + out)`; LoKr `w1` plus a full or low-rank `w2` by
/// PEFT's `use_w2` rule, exactly as [`build_adapt_lokr_targets_peft`] sizes them).
fn adapter_elements(
    dit_cfg: &MiniMaxH3DitConfig,
    paths: &[String],
    cfg: &TrainingConfig,
) -> Result<(u64, u64, u64)> {
    let rank = cfg.rank as u64;
    let mut total = 0u64;
    let mut per_block = std::collections::BTreeMap::<usize, u64>::new();
    for path in paths {
        let (out_f, in_f) = leaf_shape(dit_cfg, path)?;
        let elements = match cfg.network_type {
            NetworkType::Lora => rank * (in_f + out_f) as u64,
            NetworkType::Lokr => {
                let (out_a, out_b) = factorization(out_f, cfg.decompose_factor);
                let (in_a, in_b) = factorization(in_f, cfg.decompose_factor);
                let w2 = if (rank as f32) < (out_b.max(in_b) as f32) / 2.0 {
                    rank * (out_b + in_b) as u64
                } else {
                    (out_b * in_b) as u64
                };
                (out_a * in_a) as u64 + w2
            }
        };
        total += elements;
        if cfg.network_type == NetworkType::Lokr {
            *per_block
                .entry(target_block(path).unwrap_or(0))
                .or_default() += (out_f * in_f) as u64;
        }
    }
    let block_max = per_block.values().copied().max().unwrap_or(0);
    Ok((total, block_max, per_block.values().sum()))
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

/// The resolved target leaves of a DiT, as the shared trainable-adapter builders visit them.
struct TargetHost<'a> {
    dit: &'a mut MiniMaxH3Dit,
    paths: &'a [String],
}

impl AdaptLoraHost for TargetHost<'_> {
    fn visit_adapt_lora_mut(
        &mut self,
        f: &mut dyn FnMut(&str, &mut AdaptLinear) -> Result<()>,
    ) -> Result<()> {
        for path in self.paths {
            let segs: Vec<&str> = path.split('.').collect();
            let leaf = self.dit.adaptable_mut(&segs).ok_or_else(|| {
                Error::Msg(format!(
                    "{TRAINER_ID} trainer: target {path} does not resolve on the DiT"
                ))
            })?;
            f(path, leaf.trainable_mut())?;
        }
        Ok(())
    }
}

/// Install the run's trainable factors (on `device`) onto the DiT's target leaves.
fn install_adapters(
    dit: &mut MiniMaxH3Dit,
    paths: &[String],
    cfg: &TrainingConfig,
    device: &Device,
) -> Result<LoraSet> {
    let mut host = TargetHost { dit, paths };
    match cfg.network_type {
        NetworkType::Lora => {
            build_adapt_lora_targets(&mut host, paths, cfg.rank, cfg.alpha, cfg.seed, device)
        }
        NetworkType::Lokr => build_adapt_lokr_targets_peft(
            &mut host,
            paths,
            cfg.rank,
            cfg.alpha,
            cfg.decompose_factor,
            cfg.seed,
            device,
        ),
    }
}

/// The request-derived setup every phase of one run reads.
struct RunPlan<'r> {
    req: &'r TrainingRequest,
    edit: bool,
    target_paths: Vec<String>,
    compute_dtype: DType,
    checkpointed: bool,
    sample_prompts: Vec<String>,
    edges: Vec<u32>,
    edge: u32,
}

/// One cached `(item, edge)` entry.
struct CachedItem {
    /// The item's index (the cache is item-major over the edges).
    item: usize,
    /// The packed `[1, rows, 96]` target latent (normalized posterior mode), float32.
    x0: Tensor,
    /// The target's `[1, 24, 1, h, w]` latent shape, for the step's noise draw.
    latent_shape: Vec<usize>,
    /// An edit's reference latents `[1, 24, h, w]`, in order (empty for a captioned item).
    references: Vec<Tensor>,
    layout: ImageLayout,
}

/// The dataset as the train phase reads it.
struct Cache {
    entries: Vec<CachedItem>,
    /// Per item, the caption's layer-50 context `[1, T, 5120]` — replaced by its refined text rows
    /// once the DiT trunk is on the device.
    text: Vec<Tensor>,
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

fn decode_image(path: &Path) -> Result<Image> {
    let img = image::open(path)
        .map_err(|e| {
            Error::Msg(format!(
                "{TRAINER_ID} trainer: decoding {}: {e}",
                path.display()
            ))
        })?
        .to_rgb8();
    Ok(Image {
        width: img.width(),
        height: img.height(),
        pixels: img.into_raw(),
    })
}

/// A resume continues only the mode it was written in.
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

/// The seed of micro-step `step`'s reference-augmentation noise: a pure function of the run seed
/// and the step, so a resumed run draws what an uninterrupted one would (the MLX twin's mix).
fn step_seed(seed: u64, step: u32) -> u64 {
    seed.wrapping_add(u64::from(step))
        .wrapping_mul(0xD1B5_4A32_D192_ED03)
}

/// Every saved adapter's metadata: the provenance, and the edit marker on an edit run.
fn provenance(edit: bool) -> HashMap<String, String> {
    ADAPTER_PROVENANCE
        .iter()
        .copied()
        .chain(edit.then_some(EDIT_ADAPTER_MARKER))
        .map(|(k, v)| (k.to_owned(), v.to_owned()))
        .collect()
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
            transformer_window_size: Some(WINDOW),
            transformer_window_component: Some(TransformerComponent::Both),
            ..Default::default()
        }),
        cancel,
        ..Default::default()
    }
}

/// The scope the conditioning phase streams under: the one-block window
/// over the text encoder and the DiT, observing the run's cancel flag.
fn streamed_scope(cancel: &CancelFlag, device: &Device) -> Result<crate::StreamedRequestScope> {
    let req = preview_request("", 0, (0, 0), &[], cancel.clone());
    crate::StreamedRequestScope::enter(&req, LoadShape::DeferredMaterialization, device)
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

    /// Render `prompt` at `seed` through the registered generator with the adapter at `adapter`.
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
        let generator = crate::image_model::load_image(&spec)?;
        let req = preview_request(prompt, seed, self.size, &self.references, cancel.clone());
        generator.validate(&req)?;
        match generator.generate(&req, &mut |_| {})? {
            GenerationOutput::Images(mut images) if images.len() == 1 => Ok(images.remove(0)),
            _ => Err(Error::Msg(format!(
                "{TRAINER_ID} trainer: the preview render returned no single image"
            ))),
        }
    }
}

impl MiniMaxH3ImageTrainer {
    fn plan_run<'r>(&self, req: &'r TrainingRequest) -> Result<RunPlan<'r>> {
        gen_core_train::h3_image::validate_floors(&self.descriptor, req)?;
        validate_flow_match_request(req, &format!("{TRAINER_ID} trainer"))?;
        let cfg = &req.config;
        let target_paths = resolve_target_paths(&self.dit_cfg, cfg)?;
        let device = self.generator.device();
        // candle's CPU backend has no bf16 matmul, so a CPU device trains in f32.
        let compute_dtype = if device.is_cpu() {
            DType::F32
        } else {
            flow_match::parse_compute_dtype(&cfg.train_dtype)
        };
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
            checkpointed: cfg.gradient_checkpointing,
            sample_prompts,
            edge: edges.iter().copied().max().unwrap_or(0),
            edges,
        })
    }

    /// The run's shape as the estimate reads it — image headers and caption token counts only.
    fn run_shape(&self, plan: &RunPlan<'_>) -> Result<TrainingRunShape> {
        let tok = MiniMaxH3Tokenizer::from_snapshot(self.generator.root())?;
        let tokens = |text: &str| -> Result<u32> {
            Ok(tok
                .encode_prompt_truncated(text, MAX_PROMPT_TOKENS, &Device::Cpu)?
                .0
                .dims()[1] as u32)
        };
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
                caption_tokens: tokens(&item.caption)?,
                references,
            });
        }
        let cfg = &plan.req.config;
        let (trainable, block_max, total) =
            adapter_elements(&self.dit_cfg, &plan.target_paths, cfg)?;
        let preview = match plan.sample_prompts.is_empty() {
            true => None,
            false => {
                let longest = plan
                    .sample_prompts
                    .iter()
                    .map(|p| tokens(p))
                    .collect::<Result<Vec<u32>>>()?;
                Some(TrainingItemShape {
                    caption_tokens: longest.into_iter().max().unwrap_or(0),
                    ..items[0].clone()
                })
            }
        };
        Ok(TrainingRunShape {
            items,
            edges: plan.edges.len() as u32,
            width: plan.compute_dtype.size_in_bytes() as u64,
            trainable_elements: trainable,
            optimizer_state_per_element: optimizer_state_per_element(&cfg.optimizer),
            lokr_delta_block_max: block_max,
            lokr_delta_total: total,
            checkpointed: plan.checkpointed,
            chunked_attention: true,
            preview,
        })
    }

    fn preflight(&self, plan: &RunPlan<'_>) -> Result<()> {
        let run = self.run_shape(plan)?;
        let (contract, tier) = contract_and_tier(&self.spec)?;
        let estimate = estimate_training(&contract, &self.facts, tier, &run)?;
        let budget = self.memory_budget_override.unwrap_or_else(|| {
            flow_match::device_training_budget_bytes(
                self.generator.device(),
                &format!("{TRAINER_ID} trainer"),
            )
        });
        admit_training(&estimate, budget_of(budget), &run, plan.edge)?;
        Ok(())
    }

    /// Every caption (and an edit's references) through the generator's own conditioning phase,
    /// the text encoder streamed; then every target latent through the VAE encoder.
    fn cache(
        &self,
        plan: &RunPlan<'_>,
        on_progress: &mut dyn FnMut(TrainingProgress),
    ) -> Result<Cache> {
        let req = plan.req;
        let cfg = &req.config;
        let device = self.generator.device();
        let patch = self.dit_cfg.patch_size;
        let mut conditioned: Vec<(Tensor, Vec<u32>, Vec<Tensor>)> =
            Vec::with_capacity(req.items.len());
        {
            let _scope = streamed_scope(&req.cancel, device)?;
            let mut no_progress = |_: gen_core::Progress| {};
            for item in &req.items {
                if req.cancel.is_cancelled() {
                    return Err(Error::Canceled);
                }
                let references = item
                    .reference_image_paths
                    .iter()
                    .map(|p| decode_image(p))
                    .collect::<Result<Vec<_>>>()?;
                let refs: Vec<&Image> = references.iter().collect();
                // The render's own conditioning: the vision tower, the conditioner and each
                // reference's posterior sample (`reference_noise` stream 0 at the run seed).
                conditioned.push(self.generator.encode_conditioning(
                    &item.caption,
                    &refs,
                    cfg.seed,
                    &mut no_progress,
                )?);
            }
        }
        crate::dit::release_device_memory(device)?;

        let encoder = MiniMaxH3ImageVaeEncoder::load(self.generator.root(), device)?;
        let total = req.items.len() as u32;
        let mut entries = Vec::with_capacity(req.items.len() * plan.edges.len());
        let mut text = Vec::with_capacity(req.items.len());
        for (index, (item, (context, tags, references))) in
            req.items.iter().zip(conditioned).enumerate()
        {
            if req.cancel.is_cancelled() {
                return Err(Error::Canceled);
            }
            on_progress(TrainingProgress::Caching {
                current: index as u32 + 1,
                total,
            });
            let target = if item.is_edit_pair() {
                decode_image(&item.image_path)?
            } else {
                let square = decode_square(&item.image_path)?;
                Image {
                    width: square.width(),
                    height: square.height(),
                    pixels: square.into_raw(),
                }
            };
            for &edge in &plan.edges {
                let canvas =
                    training_canvas(item.is_edit_pair(), (target.width, target.height), edge);
                let image = resize_reference(&target, canvas)?;
                let posterior = encoder.encode(&reference_vae_pixels(&image, device)?)?;
                // The posterior mode: the sample at zero noise, normalized like a reference.
                let zero = posterior.mean().zeros_like()?;
                let latent = encoder.reference_latent(&posterior, &zero, (patch[1], patch[2]))?;
                let (_, _, h, w) = latent.dims4()?;
                let latent_shape = vec![1, crate::config::LATENT_CHANNELS, 1, h, w];
                let x0 = patchify_video_latents(&latent.reshape(latent_shape.as_slice())?, patch)?
                    .to_dtype(DType::F32)?;
                let layout = if references.is_empty() {
                    ImageLayout::text_to_image(context.dims()[1], h, w, patch)?
                } else {
                    let grids: Vec<(usize, usize)> = references
                        .iter()
                        .map(|l| (l.dims()[2], l.dims()[3]))
                        .collect();
                    ImageLayout::edit(&tags, &grids, h, w, patch)?
                };
                entries.push(CachedItem {
                    item: index,
                    x0,
                    latent_shape,
                    references: references.clone(),
                    layout,
                });
            }
            text.push(context);
        }
        drop(encoder);
        crate::dit::release_device_memory(device)?;
        if req.cancel.is_cancelled() {
            return Err(Error::Canceled);
        }
        if entries.is_empty() {
            return Err(Error::Msg(format!(
                "{TRAINER_ID} trainer: no usable dataset items"
            )));
        }
        Ok(Cache { entries, text })
    }

    /// The DiT: on the host with its trunk on the device (streamed, checkpointed), or resident;
    /// the run's factors installed; the six timesteps' modulation projected and `adaln_proj`
    /// evicted.
    fn build_dit(
        &self,
        plan: &RunPlan<'_>,
        device: &Device,
    ) -> Result<(MiniMaxH3Dit, ImageModulation, LoraSet)> {
        let cfg = &plan.req.config;
        let load_device = if plan.checkpointed {
            &Device::Cpu
        } else {
            device
        };
        let mut dit = self.generator.load_dit(load_device, plan.compute_dtype)?;
        let set = install_adapters(&mut dit, &plan.target_paths, cfg, device)?;
        let schedule = gen_core_h3::h3_schedule::SigmaSchedule::h3_image_distilled()
            .map_err(|e| Error::Msg(e.to_string()))?;
        if plan.checkpointed {
            dit.prepare_streamed(device, WINDOW as usize)?;
        }
        let modulation =
            image_modulation(&mut dit, schedule.timesteps(), plan.edit, &plan.req.cancel)?;
        Ok((dit, modulation, set))
    }

    fn train_impl(
        &mut self,
        req: &TrainingRequest,
        on_progress: &mut dyn FnMut(TrainingProgress),
        warnings: &mut Vec<gen_core::GenerationWarning>,
    ) -> Result<TrainingOutput> {
        let plan = self.plan_run(req)?;
        let cfg = &req.config;
        let device = self.generator.device().clone();
        on_progress(TrainingProgress::Preparing);
        self.preflight(&plan)?;
        let stem = file_stem(&req.file_name).to_string();
        let fingerprint = request_fingerprint(req)?;
        // The run's mode against the snapshot's, from its metadata alone, before any model loads.
        let resume_from = if cfg.resume {
            find_latest_resume(&req.output_dir, &stem)
        } else {
            None
        };
        if let Some((_, step)) = &resume_from {
            let meta = gen_core::weightsmeta::safetensors_file_metadata(
                req.output_dir.join(checkpoint_filename(&stem, *step)),
            )?;
            check_resumed_mode(
                meta.get(EDIT_ADAPTER_MARKER.0).map(String::as_str),
                plan.edit,
            )?;
        }

        on_progress(TrainingProgress::LoadingModel);
        let mut cache = self.cache(&plan, on_progress)?;
        let (dit, modulation, set) = self.build_dit(&plan, &device)?;
        // The caption rows are constants of every step: refine each item's once.
        for text in &mut cache.text {
            *text = dit.embed_context(text)?;
        }

        let mut opt = TrainOptimizer::from_config(
            &cfg.optimizer,
            set.vars.clone(),
            cfg.learning_rate,
            effective_weight_decay(cfg),
        )?;
        let accum = cfg.gradient_accumulation.max(1);
        let (total_updates, warmup_updates) =
            schedule_updates(cfg.steps, accum, cfg.lr_warmup_steps);
        let mut start_step = 0u32;
        let mut update_idx = 0u32;
        if let Some((snapshot, _)) = &resume_from {
            let restored = load_resume(snapshot, &mut opt, &set, cfg, &fingerprint)?;
            if restored.step > cfg.steps {
                return Err(Error::Msg(format!(
                    "{TRAINER_ID} trainer: resume snapshot step {} exceeds requested total steps {}",
                    restored.step, cfg.steps
                )));
            }
            start_step = restored.step;
            update_idx = restored.update_idx;
        }
        let meta = provenance(plan.edit);
        let mae = flow_match::is_mae(cfg);
        let schedule = item_major_schedule(
            TRAINER_ID,
            cache.entries.len(),
            &cfg.training_buckets(),
            cfg.seed,
        )?;
        let previews = PreviewSetup::new(self, &plan)?;

        let mut accumulated: Option<GradStore> = None;
        let mut pending = 0u32;
        let mut last_loss = 0.0f32;
        let mut steps_run = start_step;
        // A checkpoint cadence may fall mid-accumulation; its resume bundle is deferred to the
        // next completed optimizer-update boundary, so resuming from it is always exact.
        let mut resume_due = false;
        for step in start_step.saturating_add(1)..=cfg.steps {
            if req.cancel.is_cancelled() {
                break;
            }
            let entry = &cache.entries[flow_match::step_sample(None, step, &schedule).entry];
            let (loss, grads) = micro_step(
                &dit,
                &modulation,
                &set,
                cfg,
                step,
                entry,
                &cache.text[entry.item],
                mae,
                &device,
            )?;
            last_loss = loss;
            steps_run = step;
            accumulate_grads(&mut accumulated, grads, &set.vars)?;
            pending += 1;
            if step.is_multiple_of(accum) {
                apply_update(
                    &mut opt,
                    &mut accumulated,
                    &set,
                    pending,
                    cfg,
                    update_idx,
                    total_updates,
                    warmup_updates,
                    cfg.seed,
                )?;
                pending = 0;
                update_idx += 1;
            }
            on_progress(TrainingProgress::Training {
                step,
                total: cfg.steps,
                loss: last_loss,
            });

            if let Some(previews) = previews.as_ref() {
                if step % cfg.sample_every.max(1) == 0 {
                    render_previews(
                        previews,
                        req,
                        &set,
                        &meta,
                        &stem,
                        step,
                        on_progress,
                        warnings,
                    )?;
                }
            }

            if cfg.save_every > 0 && step % cfg.save_every == 0 && step != cfg.steps {
                create_output_dir(&req.output_dir)?;
                save_adapter(
                    &set,
                    &meta,
                    &req.output_dir.join(checkpoint_filename(&stem, step)),
                )?;
                resume_due = true;
                on_progress(TrainingProgress::Checkpoint { step });
            }
            if resume_due && pending == 0 {
                save_resume(
                    &req.output_dir,
                    &stem,
                    step,
                    update_idx,
                    &opt,
                    &set,
                    cfg,
                    &fingerprint,
                )?;
                resume_due = false;
            }
        }

        // Cancelled before a single step completed: the factors are still the no-op init. Surface
        // the cancellation rather than writing a valid-looking identity adapter.
        if steps_run == 0 {
            return Err(Error::Canceled);
        }
        // Flush a pending (sub-`accum`) window as a true mean of the grads it holds.
        if accumulated.is_some() {
            apply_update(
                &mut opt,
                &mut accumulated,
                &set,
                pending,
                cfg,
                update_idx,
                total_updates,
                warmup_updates,
                cfg.seed,
            )?;
            if resume_due {
                save_resume(
                    &req.output_dir,
                    &stem,
                    steps_run,
                    update_idx + 1,
                    &opt,
                    &set,
                    cfg,
                    &fingerprint,
                )?;
            }
        }
        on_progress(TrainingProgress::Saving);
        create_output_dir(&req.output_dir)?;
        let adapter_path = req.output_dir.join(&req.file_name);
        save_adapter(&set, &meta, &adapter_path)?;
        Ok(TrainingOutput {
            adapter_path,
            steps: steps_run,
            final_loss: last_loss,
        })
    }
}

/// Draw micro-step `step`'s inputs from `entry` — the noise level (the configured distribution,
/// quantized onto the six published steps), the noised target and, for an edit, the references
/// noised to `0.999` with this step's augmentation draw, all pure functions of the seed and the
/// step — and run its forward and backward.
#[allow(clippy::too_many_arguments)]
fn micro_step(
    dit: &MiniMaxH3Dit,
    modulation: &ImageModulation,
    set: &LoraSet,
    cfg: &TrainingConfig,
    step: u32,
    entry: &CachedItem,
    text_rows: &Tensor,
    mae: bool,
    device: &Device,
) -> Result<(f32, GradStore)> {
    let t = flow_match::sample_unit_timestep(
        &cfg.timestep_type,
        &cfg.timestep_bias,
        flow_match::timestep_seed(cfg.seed, step),
    );
    let st = training_step(t);
    let noise = flow_match::sample_noise(
        &entry.latent_shape,
        flow_match::noise_seed(cfg.seed, step),
        device,
    )?;
    let noise = patchify_video_latents(&noise, entry.layout.patch())?;
    let (x_t, target) = noised_rows(&entry.x0, &noise, st)?;
    let packed = if entry.references.is_empty() {
        x_t
    } else {
        let seed = step_seed(cfg.seed, step);
        let aug = entry
            .references
            .iter()
            .enumerate()
            .map(|(i, l)| reference_noise(seed, 1, i, l.dims(), device))
            .collect::<Result<Vec<_>>>()?;
        let condition = reference_condition_rows(&entry.references, &aug, entry.layout.patch())?;
        Tensor::cat(&[&condition, &x_t], 1)?
    };
    loss_and_grads(
        dit,
        modulation,
        &set.vars,
        &StepInputs {
            text_rows,
            layout: &entry.layout,
            packed: &packed,
            target: &target,
            step: st,
        },
        mae,
    )
}

/// Best-effort previews through the registered generator, the current factors saved as an adapter
/// file beside the run's output (removed afterwards). A failure never aborts training.
#[allow(clippy::too_many_arguments)]
fn render_previews(
    previews: &PreviewSetup,
    req: &TrainingRequest,
    set: &LoraSet,
    meta: &HashMap<String, String>,
    stem: &str,
    step: u32,
    on_progress: &mut dyn FnMut(TrainingProgress),
    warnings: &mut Vec<gen_core::GenerationWarning>,
) -> Result<()> {
    create_output_dir(&req.output_dir)?;
    let file = req.output_dir.join(format!(".{stem}-preview.safetensors"));
    save_adapter(set, meta, &file)?;
    let total = previews.prompts.len() as u32;
    for (index, prompt) in previews.prompts.iter().enumerate() {
        if req.cancel.is_cancelled() {
            break;
        }
        let seed = sample_seed(req.config.seed, step, index);
        match previews.render(&file, prompt, seed, &req.cancel) {
            Ok(image) => on_progress(TrainingProgress::Sample {
                step,
                index: index as u32 + 1,
                total,
                prompt: prompt.clone(),
                image,
            }),
            Err(Error::Canceled) => break,
            Err(e) => {
                eprintln!(
                    "[SW-7098] {TRAINER_ID} trainer: preview sample failed at step {step} (prompt \
                     {}): {e} — skipping this preview, training continues",
                    index + 1
                );
                warnings.push(gen_core::train::preview_sample_failed(
                    step,
                    index as u32 + 1,
                    &e,
                ));
            }
        }
    }
    let _ = std::fs::remove_file(&file);
    Ok(())
}

impl Trainer for MiniMaxH3ImageTrainer {
    fn descriptor(&self) -> &TrainerDescriptor {
        &self.descriptor
    }

    fn validate(&self, req: &TrainingRequest) -> gen_core::Result<()> {
        gen_core_train::h3_image::validate_floors(&self.descriptor, req)?;
        validate_flow_match_request(req, &format!("{TRAINER_ID} trainer"))?;
        resolve_target_paths(&self.dit_cfg, &req.config)?;
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
