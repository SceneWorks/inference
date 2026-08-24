//! # candle-gen-sensenova
//!
//! The **SenseNova-U1** (NEO-Unify) provider crate for [`candle-gen`](candle_gen) — the candle
//! (Windows/CUDA) sibling of `mlx-gen-sensenova`. It implements the backend-neutral
//! [`gen_core::Generator`] contract and exposes both variants through its explicit family catalog.
//!
//! **Image generation:** SenseNova-U1 is a *unified* multimodal model — a dense dual-path Qwen3 "MoT"
//! backbone (understanding + generation paths) with a flow-matching image head; there is no separate
//! VAE or text encoder. The registered non-think routes cover T2I plus instruction/reference it2i:
//! build the `neo1_0` prompt and optional vision prefix, prefill it on the understanding path, then
//! run the flow-matching denoise loop on the generation path (`crate::t2i`) and unpatchify to RGB.
//! Deterministic CPU-seeded noise (sc-3673) makes output launch-portable per seed.
//!
//! Two registered ids share the loader: **`sensenova_u1_8b`** (50 NFE, CFG 4.0) and
//! **`sensenova_u1_8b_fast`** (8 NFE, CFG 1.0 — its loader merges the 8-step distill LoRA into the
//! dense generation path). Both advertise T2I plus reference-conditioned instruction edit and
//! Character Studio through the `Generator` contract. User LoRAs remain unsupported and are rejected
//! rather than silently dropped. `backend` is `"candle"` and `mac_only` is `false`.
//!
//! **Tiers (sc-14249, epic 9083).** The crate consumes the SceneWorks turnkey's `bf16/`, `q8/` and
//! `q4/` tiers directly, through one seam (`quant::detect_linear`): a projection whose
//! `.scales` sibling is present builds packed from the MLX triple, otherwise it loads dense at the
//! checkpoint's own store dtype and widens to f32 per op. So the tier is chosen by the DIRECTORY the
//! caller resolved and `Quant` is a tier label, not a request to quantize anything here. This
//! replaced a hard `DType::F32` mmap that widened the bf16 checkpoint for no extra precision — a
//! measured 70.5 GB peak on sm_120 for a 32.7 GiB checkpoint.
//!
//! **Live denoise preview (epic 16948, sc-16960).** Both ids advertise `supports_preview` and emit one
//! RGB8 frame per outer solver step from the bespoke flow-match loop in `crate::t2i`. SenseNova is
//! the epic's **Tier 2** family: it drives no shared `candle_gen` sampler, so the emission is a direct
//! `PreviewHook::emit_step` inside its own loop; and — contrary to the epic's scoping — it has **no
//! VAE at all**, so its fit could not be inherited from epic 16624 and had to be measured. It
//! denoises in pixel space, which is why [`preview`]'s fit is over **three** channels rather than a
//! VAE latent width. See that module and `tests/fit_preview_rgb.rs`.
//!
//! **Understanding surface (VQA + interleave, sc-5501):** SenseNova-U1's text / text+image modes
//! ([`T2iModel::vqa`], [`T2iModel::interleave_gen`]) output what the neutral
//! `GenerationOutput` contract can't express, so they are exposed as
//! typed admitted methods on [`SenseNovaUnderstanding`] (built via
//! [`load_understanding_with_spec`]). The worker never receives the raw model, so route/geometry,
//! request memory, lifecycle cleanup, and checkpoint identity remain one operation.

mod config;
mod distill;
mod fm;
pub mod memory_strategy;
pub mod preview;
mod quant;
mod qwen3;
mod runtime;
mod t2i;
mod text;
mod vision;

use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex};

use candle_gen::candle_core::{DType, Device, Tensor};
use candle_gen::candle_nn::VarBuilder;
use candle_gen::gen_core::{
    self, reject_unknown_components, CancelFlag, Capabilities, Conditioning, ConditioningKind,
    GenerationOutput, GenerationRequest, Generator, Image, LoadSpec, Modality, ModelDescriptor,
    Progress, Quant, WeightsSource,
};
use candle_gen::{CandleError, Result};

use distill::{resolve_distill_lora, DistillLora, DISTILL_LORA_FILE, DISTILL_MERGED_MARKER};

// The understanding surface (VQA + Document-Studio interleave) is driven by the worker **directly**
// off the concrete `T2iModel` (its text / text+image output the neutral `Generator` contract can't
// express), so re-export the types + a dense loader the worker assembles them from.
pub use config::NeoChatConfig;
pub use runtime::Sampler;
pub use t2i::{
    interleave_resolution_for, smart_resize, tensor_to_image, CfgNorm, InterleaveOutput, T2iModel,
    T2iOptions, INTERLEAVE_RESOLUTIONS,
};
pub use text::{SenseNovaTokenizer, INTERLEAVE_SYSTEM_MESSAGE};

/// Registry id — the base 8B-MoT variant.
pub const MODEL_ID: &str = "sensenova_u1_8b";
/// The 8-step distilled variant (same base weights, distill LoRA merged at load).
pub const MODEL_ID_FAST: &str = "sensenova_u1_8b_fast";

const DEFAULT_STEPS: u32 = 50;
const DEFAULT_GUIDANCE: f32 = 4.0;
/// Distilled defaults (`docs/base_vs_distill.md`): 8 NFE at CFG 1.0 (guidance off).
const DEFAULT_STEPS_FAST: u32 = 8;
const DEFAULT_GUIDANCE_FAST: f32 = 1.0;
/// The product inference timestep-shift for the t2i path when the request doesn't override it
/// (`req.scheduler_shift`). This is the *pipeline* shift the reference `t2i_generate` applies at
/// sampling time — deliberately **3.0**, which is distinct from the checkpoint's config-declared
/// `timestep_shift` (`NeoChatConfig::timestep_shift`, `1.0` for the shipped 8B-MoT). See
/// [`NeoChatConfig::inference_timestep_shift`] for why the config field does *not* feed inference.
const DEFAULT_TIMESTEP_SHIFT: f32 = 3.0;
const REF_MIN_PIXELS: i64 = 512 * 512;
const REF_MAX_PIXELS: i64 = 2048 * 2048;
/// Cell = patch·merge: every side must be a multiple of this (the patchify grid).
pub const SIZE_MULTIPLE: u32 = 32;
/// Tokenizer-level sentinels owned by the internal image-prefix builder. Letting user text inject
/// one would corrupt the token/grid relationship that the understanding path derives.
const RESERVED_IMAGE_MARKERS: [&str; 3] = ["<IMG_CONTEXT>", "<img>", "</img>"];

/// The base descriptor (`sensenova_u1_8b`).
pub fn descriptor() -> ModelDescriptor {
    descriptor_for(MODEL_ID)
}

/// The 8-step distilled descriptor (`sensenova_u1_8b_fast`). Identical capability surface to the
/// base — only the id and the generation defaults differ.
pub fn descriptor_fast() -> ModelDescriptor {
    descriptor_for(MODEL_ID_FAST)
}

/// SenseNova-U1's registered Candle surface: classifier-free text guidance plus Reference and
/// MultiReference it2i with image guidance, over the q4/q8/bf16 turnkey tiers. VQA and interleave are
/// direct worker APIs because their text-bearing outputs do not fit `GenerationOutput`; user LoRA is
/// not advertised. Backend-correct deviations from MLX are `backend = "candle"` and `mac_only = false`.
fn descriptor_for(id: &'static str) -> ModelDescriptor {
    ModelDescriptor {
        encoder_contract: None,
        denoiser_output_latent_space: None,
        control_kinds: None,
        required_components: &[],
        id,
        family: "sensenova-u1",
        backend: "candle",
        modality: Modality::Image,
        capabilities: Capabilities {
            supports_guidance: true,
            // `guidance` is text CFG; `true_cfg` is the image-guidance scale on it2i.
            supports_true_cfg: true,
            conditioning: vec![
                ConditioningKind::Reference,
                ConditioningKind::MultiReference,
            ],
            // Bespoke-by-architecture (epic 7114 P4, sc-7123 — mirrors the mlx-gen SenseNova won't-do):
            // SenseNova-U1 is an AUTOREGRESSIVE backbone whose `predict_v` mutates a per-step `KvCache`
            // (`cache.len()` feeds the RoPE/position build) shared across the cond/uncond passes. The
            // unified curated solvers do multiple model evals per step (heun = 2; dpmpp_2m/uni_pc reuse
            // prior-step state) and would append to the cache multiple times → desynced AR positions →
            // corrupt output. The native shifted-Euler (single eval/step, `req.scheduler_shift`) is the
            // only valid integrator, so no curated sampler/scheduler menu is advertised (N3: empty list).
            samplers: Vec::new(),
            min_size: 256,
            max_size: 2048,
            max_count: 8,
            // The SceneWorks turnkey's pre-quantized q4/q8 tiers load natively (sc-14249): every
            // backbone projection packed-detects its MLX triple, so a Q4/Q8 here is a turnkey tier
            // SELECT (which subdir the caller resolved), not an on-the-fly quantize. bf16 resolves
            // to `None` and loads dense. Same contract as flux1/qwen/kolors.
            supported_quants: &[Quant::Q4, Quant::Q8],
            // The backbone uses a KV cache for the AR prefix + denoise.
            supports_kv_cache: true,
            // Flow-match schedule uses a timestep shift (mapped from scheduler_shift).
            requires_sigma_shift: true,
            // Per-step latent previews (epic 16948, sc-16960). Both ids reach the one bespoke
            // flow-match denoise loop in `t2i.rs`, which emits directly through
            // `PreviewHook::emit_step`; `preview` owns the fit and the pool to the token grid.
            supports_preview: true,
            ..Default::default()
        },
    }
}

/// The loaded SenseNova-U1 components, `Arc`-shared so the generator can cache them across calls.
#[derive(Clone)]
struct Components {
    tokenizer: Arc<SenseNovaTokenizer>,
    model: Arc<T2iModel>,
    /// The parsed checkpoint config, kept so `options` can resolve the inference timestep shift from
    /// the checkpoint's own `timestep_shift` rather than always the product default (sc-9029).
    cfg: Arc<NeoChatConfig>,
}

/// A loaded candle SenseNova-U1 generator. Loading is **lazy**: `load` does no backbone file I/O
/// (registry introspection against a missing path still resolves), and the heavy unified model is
/// built on the first [`generate`](Generator::generate) call and then cached. The one exception is the
/// `fast` variant, which resolves + existence-checks its distill LoRA at `load` (sc-13664) so a
/// missing/unprovisioned LoRA fails fast at load rather than at first generate.
pub struct SenseNovaGenerator {
    descriptor: ModelDescriptor,
    root: PathBuf,
    device: Device,
    /// The 8-step distilled variant — merges the distill LoRA at build + applies distilled defaults.
    fast: bool,
    /// The distill LoRA path resolved from the caller-supplied `distill_lora` component / co-located
    /// snapshot file at load (sc-13664), so the merge in [`Self::components`] reads a passed-in path
    /// with no env / HF-cache derivation. `Some` for the `fast` variant UNLESS the pre-merged
    /// `distill_merged.json` marker ([`DISTILL_MERGED_MARKER`]) is present — a turnkey tier bakes the
    /// merge into its on-disk weights (sc-8775/sc-13787), so resolve+merge is skipped and the dense
    /// weights load as-is; `None` for the base id and for a pre-merged fast tier.
    distill_lora: Option<PathBuf>,
    components: Mutex<Option<Components>>,
    loaded_spec: LoadSpec,
    memory_strategy: gen_core::MemoryProviderContract,
    inventory: Option<memory_strategy::CheckpointInventory>,
}

impl SenseNovaGenerator {
    fn ensure_inventory_unchanged(&self) -> Result<()> {
        if let Some(inventory) = &self.inventory {
            inventory.ensure_unchanged().map_err(CandleError::from)?;
        }
        Ok(())
    }

    fn components(&self) -> Result<Components> {
        candle_gen::cached(&self.components, || {
            self.ensure_inventory_unchanged()?;
            memory_strategy::validate_resolved_artifact_binding(&self.loaded_spec)
                .map_err(CandleError::from)?;
            memory_strategy::validate_artifact_tier(&self.loaded_spec)
                .map_err(CandleError::from)?;
            let cfg = NeoChatConfig::from_dir(&self.root)?;
            let vb = backbone_vb(&self.root, &self.device)?;
            let mut model =
                if memory_strategy::streamable_spec(self.descriptor.id, &self.loaded_spec) {
                    let inventory = memory_strategy::CheckpointInventory::capture(&self.root)
                        .map_err(CandleError::from)?;
                    T2iModel::from_weights_with_deferred_gen(&vb, &cfg, inventory)?
                } else {
                    T2iModel::from_weights(&vb, &cfg)?
                };
            // Merge the 8-step distill LoRA into the dense generation path — ONLY when the fast loader
            // resolved one at `load` (sc-13664/sc-13787). A **pre-merged** turnkey tier ships the
            // `distill_merged.json` marker and no LoRA, so `self.distill_lora` is `None`; its merge is
            // already baked into the on-disk weights — skip and load the dense weights as-is (the base
            // id is likewise `None`). When there IS a LoRA, assert full coverage — `7 · layers` gen-path
            // projections + the 2 FM-head Linears — so a stale/mismatched LoRA fails loudly rather than
            // silently merging a subset. (Mirrors `mlx-gen-sensenova`'s marker-guarded merge.)
            if let Some(lora_path) = &self.distill_lora {
                let lora = DistillLora::from_file(lora_path)?;
                let applied = model.merge_distill_lora(&lora)?;
                let expected = cfg.llm.num_hidden_layers * 7 + 2;
                if applied != expected {
                    return Err(CandleError::Msg(format!(
                        "{}: distill LoRA merged {applied} targets, expected {expected} \
                         (7·{} gen-path linears + 2 fm_head) — wrong LoRA file?",
                        self.descriptor.id, cfg.llm.num_hidden_layers
                    )));
                }
            }
            let tokenizer = SenseNovaTokenizer::from_dir(&self.root)?;
            Ok(Components {
                tokenizer: Arc::new(tokenizer),
                model: Arc::new(model),
                cfg: Arc::new(cfg),
            })
        })
    }

    /// Map a request to [`T2iOptions`] (distilled vs base defaults; explicit request values win).
    ///
    /// `cfg` is the loaded checkpoint config: the timestep shift is resolved through
    /// [`NeoChatConfig::inference_timestep_shift`] so a future checkpoint variant that declares its
    /// own inference shift is honored instead of being shadowed by the product default (sc-9029).
    fn options(&self, req: &GenerationRequest, cfg: &NeoChatConfig, seed: u64) -> T2iOptions {
        let (def_steps, def_guidance) = if self.fast {
            (DEFAULT_STEPS_FAST, DEFAULT_GUIDANCE_FAST)
        } else {
            (DEFAULT_STEPS, DEFAULT_GUIDANCE)
        };
        T2iOptions {
            cfg_scale: req.guidance.unwrap_or(def_guidance),
            img_cfg_scale: req.true_cfg.unwrap_or(1.0),
            num_steps: req.steps.unwrap_or(def_steps) as usize,
            // Precedence: explicit request wins; else the checkpoint's own inference shift if it
            // declares one; else the product default (3.0). Reads the parsed config so the field is
            // no longer a silent shadow (sc-9029 / F-045).
            timestep_shift: req
                .scheduler_shift
                .unwrap_or_else(|| cfg.inference_timestep_shift(DEFAULT_TIMESTEP_SHIFT)),
            seed,
            attention_score_budget: Some(memory_strategy::request_attention_budget(req)),
            transformer_window_size: req
                .memory
                .filter(|memory| memory.stream_transformer_blocks)
                .and_then(|memory| memory.transformer_window_size)
                .map(|window| window as usize),
            ..Default::default()
        }
    }

    /// The rich-`Result` body behind [`Generator::generate`].
    fn generate_impl(
        &self,
        req: &GenerationRequest,
        on_progress: &mut dyn FnMut(Progress),
    ) -> Result<GenerationOutput> {
        let comps = self.components()?;
        let base_seed = req.seed.unwrap_or_else(gen_core::default_seed);
        let (w, h) = (req.width as usize, req.height as usize);
        let references = references(req)?;

        // The per-step preview seam (epic 16948, sc-16960): ONE hook, built over the REQUEST's own
        // sink, carrying the token cell of the very model whose denoise loop the frames come from.
        // It is threaded into `T2iModel::generate` as a non-`Option` `&PreviewHook` — this crate
        // drives no shared sampler, so there is no driver argument `candle-gen-catalog`'s route
        // inventory could classify, and an `Option` anywhere on the path would be blankable here.
        // An inert sink costs one branch per denoise step and leaves the render byte-identical.
        let preview = preview::t2i_hook(&req.preview, comps.model.cell());
        let images = candle_gen::for_each_image_seed(base_seed, req.count, |seed| {
            // A 50-step 8B run is multi-minute; check cancellation between images too (the per-step
            // check lives in the denoise loop).
            if req.cancel.is_cancelled() {
                return Err(CandleError::Canceled);
            }
            let opts = self.options(req, &comps.cfg, seed);
            let img = if references.is_empty() {
                comps.model.generate(
                    &comps.tokenizer,
                    &req.prompt,
                    w,
                    h,
                    &opts,
                    &req.cancel,
                    on_progress,
                    &preview,
                )?
            } else {
                comps.model.it2i_generate(
                    &comps.tokenizer,
                    &req.prompt,
                    &references,
                    w,
                    h,
                    &opts,
                    &req.cancel,
                    on_progress,
                    &preview,
                )?
            };
            // `?` bridges the candle-side `tensor_to_image` error into `CandleError`.
            Ok(tensor_to_image(&img)?)
        })?;
        Ok(GenerationOutput::Images(images))
    }
}

impl Generator for SenseNovaGenerator {
    fn descriptor(&self) -> &ModelDescriptor {
        &self.descriptor
    }

    fn memory_strategy_contract(&self) -> Option<&gen_core::MemoryProviderContract> {
        Some(&self.memory_strategy)
    }

    fn memory_strategy_safety_check(
        &self,
        context: &gen_core::MemoryRunContext,
    ) -> gen_core::MemorySafetyDecision {
        memory_strategy::safety_check(
            self.descriptor.id,
            &self.memory_strategy,
            context,
            self.loaded_spec.quantize,
        )
    }

    fn begin_memory_strategy_request(
        &self,
        context: &gen_core::MemoryRunContext,
    ) -> gen_core::Result<Option<Box<dyn gen_core::MemoryRequestScope + '_>>> {
        memory_strategy::validate_context(
            self.descriptor.id,
            &self.memory_strategy,
            context,
            self.loaded_spec.quantize,
        )?;
        Ok(Some(Box::new(memory_strategy::request_scope(
            self.descriptor.id,
            self.device.clone(),
            &self.memory_strategy,
            context,
            42,
        )?)))
    }

    fn validate(&self, req: &GenerationRequest) -> gen_core::Result<()> {
        let id = self.descriptor.id;
        // Capability floor (count/size range, guidance, and Reference/MultiReference only).
        self.descriptor.capabilities.validate_request(id, req)?;
        if let Some(marker) = RESERVED_IMAGE_MARKERS
            .iter()
            .find(|marker| req.prompt.contains(**marker))
        {
            return Err(gen_core::Error::Unsupported(format!(
                "{id}: prompt contains reserved internal image marker {marker}"
            )));
        }
        // SenseNova consumes every supplied reference at full model-native conditioning. It has no
        // strength-weighted blend or schedule-tail primitive, so accepting either strength carrier
        // would silently discard a user control. Validate the per-reference carrier first: when a
        // caller supplies both forms, the more specific field is the actionable error and the
        // request-level fallback is considered only when the reference omitted its own value.
        for conditioning in &req.conditioning {
            match conditioning {
                Conditioning::Reference {
                    strength: Some(_), ..
                } => {
                    return Err(gen_core::Error::Unsupported(format!(
                        "{id}: Conditioning::Reference.strength is unsupported; omit it or set it to null"
                    )));
                }
                Conditioning::MultiReference { images } if images.is_empty() => {
                    return Err(gen_core::Error::Unsupported(format!(
                        "{id}: MultiReference conditioning requires at least one image"
                    )));
                }
                _ => {}
            }
        }
        if req.strength.is_some() {
            return Err(gen_core::Error::Unsupported(format!(
                "{id}: request-level strength is unsupported; omit it or set it to null"
            )));
        }
        if req.true_cfg.is_some() && !has_reference(req) {
            return Err(gen_core::Error::Unsupported(format!(
                "{id}: true_cfg is image guidance and requires Reference or non-empty \
                 MultiReference conditioning"
            )));
        }
        if req.prompt.trim().is_empty() {
            return Err(gen_core::Error::Msg(format!(
                "{id}: prompt must not be empty"
            )));
        }
        if !req.width.is_multiple_of(SIZE_MULTIPLE) || !req.height.is_multiple_of(SIZE_MULTIPLE) {
            return Err(gen_core::Error::Msg(format!(
                "{id}: width/height must be multiples of {SIZE_MULTIPLE} (got {}x{})",
                req.width, req.height
            )));
        }
        // `steps == 0` builds an empty denoise trajectory; `None` falls back to the variant default.
        if req.steps == Some(0) {
            return Err(gen_core::Error::Msg(format!("{id}: steps must be >= 1")));
        }
        Ok(())
    }

    fn generate(
        &self,
        req: &GenerationRequest,
        on_progress: &mut dyn FnMut(Progress),
    ) -> gen_core::Result<GenerationOutput> {
        self.validate(req)?;
        self.ensure_inventory_unchanged()
            .map_err(gen_core::Error::from)?;
        let result = self.generate_impl(req, on_progress);
        self.ensure_inventory_unchanged()
            .map_err(gen_core::Error::from)?;
        result.map_err(Into::into)
    }
}

/// Whether this request actually carries image guidance (an empty multi-reference is absent).
fn has_reference(req: &GenerationRequest) -> bool {
    req.conditioning
        .iter()
        .any(|conditioning| match conditioning {
            Conditioning::Reference { .. } => true,
            Conditioning::MultiReference { images } => !images.is_empty(),
            _ => false,
        })
}

fn references(req: &GenerationRequest) -> Result<Vec<Tensor>> {
    let mut out = Vec::new();
    for conditioning in &req.conditioning {
        match conditioning {
            Conditioning::Reference { image, .. } => out.push(image_to_chw01(image)?),
            Conditioning::MultiReference { images } => {
                for image in images {
                    out.push(image_to_chw01(image)?);
                }
            }
            _ => {}
        }
    }
    Ok(out)
}

/// Decode one RGB8 reference to a smart-resized `[3,H,W]` f32 tensor in `[0,1]`, matching the MLX
/// SenseNova request boundary and the checkpoint's understanding-vision preprocessing contract.
fn image_to_chw01(image: &Image) -> Result<Tensor> {
    let (in_w, in_h) = (image.width as usize, image.height as usize);
    let expected =
        gen_core::imageops::checked_image_buffer_len(in_w, in_h, 3).ok_or_else(|| {
            CandleError::Msg(format!(
                "sensenova: invalid reference dimensions {}x{}",
                image.width, image.height
            ))
        })?;
    if image.pixels.len() != expected {
        return Err(CandleError::Msg(format!(
            "sensenova: reference pixel buffer {} != {in_w}x{in_h}x3",
            image.pixels.len()
        )));
    }
    let (out_h, out_w) = smart_resize(
        image.height as i32,
        image.width as i32,
        SIZE_MULTIPLE as i32,
        REF_MIN_PIXELS,
        REF_MAX_PIXELS,
    );
    let resized = gen_core::imageops::resize_bicubic_u8(
        &image.pixels,
        in_h,
        in_w,
        out_h as usize,
        out_w as usize,
    )?;
    let data: Vec<f32> = resized.into_iter().map(|pixel| pixel / 255.0).collect();
    Ok(
        Tensor::from_vec(data, (out_h as usize, out_w as usize, 3), &Device::Cpu)?
            .permute((2, 0, 1))?
            .contiguous()?,
    )
}

/// The SenseNova-U1 checkpoint shards under `root` — the flat `*.safetensors`, excluding the optional
/// co-located distill LoRA (and any AppleDouble sidecar).
fn backbone_files(root: &Path) -> Result<Vec<PathBuf>> {
    let mut files: Vec<PathBuf> = std::fs::read_dir(root)
        .map_err(|e| CandleError::Msg(format!("sensenova: read {}: {e}", root.display())))?
        .filter_map(|e| e.ok().map(|e| e.path()))
        .filter(|p| p.extension().is_some_and(|x| x == "safetensors"))
        .filter(|p| !candle_gen::gen_core::weightsmeta::is_hidden_file(p))
        .filter(|p| p.file_name().and_then(|n| n.to_str()) != Some(DISTILL_LORA_FILE))
        .collect();
    files.sort();
    if files.is_empty() {
        return Err(CandleError::Msg(format!(
            "sensenova: no .safetensors found in {} (expected a SenseNova-U1-8B-MoT snapshot)",
            root.display()
        )));
    }
    Ok(files)
}

/// mmap a [`VarBuilder`] over the SenseNova-U1 checkpoint at its **on-disk store dtype** (sc-14249).
///
/// This used to be `f32_vb`, pinned to [`DType::F32`]. The shipped checkpoint is bf16 (`config.json`
/// `llm_config.torch_dtype: "bfloat16"`, 32.7 GiB on disk), so that pin *widened* every weight for no
/// extra precision — a measured **70.5 GB** peak on sm_120. The store now follows the checkpoint
/// (`quant::store_dtype_for`: bf16 stays bf16, anything else keeps loading f32 — never widen, never
/// truncate), and each projection widens to f32 per op via `QLinear::forward_upcast`, so **every
/// matmul is still f32** and the arithmetic is unchanged. The dense leaves the model multiplies
/// directly (norms, conv kernels, FM/timestep Linears) are read at f32 through `quant::get_f32`.
///
/// The probe reads the FIRST shard's header rather than trusting `config.json`: the config's
/// `torch_dtype` describes the upstream release, not necessarily what a given tier's packer emitted,
/// and getting this wrong in the truncating direction would silently round every weight.
fn backbone_vb(root: &Path, device: &Device) -> Result<VarBuilder<'static>> {
    let files = backbone_files(root)?;
    let dtype = crate::quant::store_dtype_for(checkpoint_dtype(&files));
    // Shared audited unsafe-mmap surface (sc-8999 / F-019). The distill-LoRA exclusion filter above
    // is a genuine per-site variation, so the read_dir/sort stays local; only the mmap is shared.
    candle_gen::mmap_var_builder(&files, dtype, device)
}

/// The dtype the checkpoint stores its weights in, probed from one small always-dense tensor.
///
/// Reads the final backbone RMSNorm (`language_model.model.norm.weight`, `[4096]`): it is present and
/// **dense** in every tier — the packer quantizes only the 588 layer projections — so it reports the
/// tier's real store dtype on a packed q4/q8 tier just as it does on `bf16/`, without being fooled by
/// the `U32` code tensors sitting next to it. Anything unreadable falls back to `F32`, the
/// pre-sc-14249 behavior. Mirrors `candle-gen-ideogram`'s `te_store_dtype` (sc-12828).
fn checkpoint_dtype(files: &[PathBuf]) -> DType {
    const PROBE_KEY: &str = "language_model.model.norm.weight";
    // SAFETY: read-only mmap of weight files; the standard candle loading path.
    unsafe { candle_gen::candle_core::safetensors::MmapedSafetensors::multi(files) }
        .ok()
        .and_then(|st| st.load(PROBE_KEY, &Device::Cpu).ok())
        .map(|t| t.dtype())
        .unwrap_or(DType::F32)
}

/// Build the dense f32 understanding model ([`T2iModel`]) + tokenizer for a SenseNova-U1-8B-MoT
/// snapshot — the VQA / interleave entry the worker drives directly (the modes the neutral
/// `Generator` contract can't express). Loads **dense** (no distill LoRA, no quantization), exactly
/// like the base registry path, so the understanding decode uses the full base model. The heavy mmap
/// happens here, so the worker calls this once on its blocking thread per job.
#[cfg(feature = "real-weight-diagnostics")]
#[doc(hidden)]
pub fn load_understanding_for_diagnostics(root: &Path) -> Result<(T2iModel, SenseNovaTokenizer)> {
    let cfg = NeoChatConfig::from_dir(root)?;
    let device = candle_gen::default_device()?;
    let vb = backbone_vb(root, &device)?;
    let model = T2iModel::from_weights(&vb, &cfg)?;
    let tokenizer = SenseNovaTokenizer::from_dir(root)?;
    Ok((model, tokenizer))
}

/// Load the direct VQA/interleave runtime through the same exact spec and memory contract as the
/// registered generator. This is the worker-facing seam for request-scoped direct modes.
pub struct SenseNovaUnderstanding {
    model: T2iModel,
    tokenizer: SenseNovaTokenizer,
    device: Device,
    spec: LoadSpec,
    contract: gen_core::MemoryProviderContract,
    inventory: memory_strategy::CheckpointInventory,
}

impl SenseNovaUnderstanding {
    pub fn memory_strategy_contract(&self) -> &gen_core::MemoryProviderContract {
        &self.contract
    }

    pub fn begin_memory_strategy_request(
        &self,
        context: &gen_core::MemoryRunContext,
    ) -> gen_core::Result<Box<dyn gen_core::MemoryRequestScope + '_>> {
        memory_strategy::validate_context(MODEL_ID, &self.contract, context, self.spec.quantize)?;
        Ok(Box::new(memory_strategy::request_scope(
            MODEL_ID,
            self.device.clone(),
            &self.contract,
            context,
            42,
        )?))
    }

    /// Run a direct VQA/interleave operation against the exact pinned checkpoint inventory. The
    /// post-check runs after success, cancellation, or an ordinary error; an artifact mutation
    /// takes precedence so crossed evidence cannot release a stale result.
    fn with_runtime<T>(
        &self,
        operation: impl FnOnce(&T2iModel, &SenseNovaTokenizer) -> Result<T>,
    ) -> Result<T> {
        self.inventory
            .ensure_unchanged()
            .map_err(CandleError::from)?;
        let result = operation(&self.model, &self.tokenizer);
        self.inventory
            .ensure_unchanged()
            .map_err(CandleError::from)?;
        result
    }

    fn validate_direct_request(
        &self,
        context: &gen_core::MemoryRunContext,
        actual_mode: gen_core::MemoryMode,
        request: &GenerationRequest,
        actual_reference_count: usize,
    ) -> Result<()> {
        let actual_reference_count = u32::try_from(actual_reference_count)
            .map_err(|_| CandleError::Msg("sensenova: too many direct image references".into()))?;
        if request.image_reference_count() != actual_reference_count {
            return Err(CandleError::Msg(format!(
                "sensenova: request declares {} image references but direct execution received {actual_reference_count}",
                request.image_reference_count()
            )));
        }
        memory_strategy::validate_direct_operation_identity(
            MODEL_ID,
            context,
            &actual_mode,
            gen_core::MemoryGeometry {
                width: request.width,
                height: request.height,
                batch: request.count,
                frames: request.frames.unwrap_or(context.geometry.frames),
                reference_count: actual_reference_count,
            },
        )
        .map_err(CandleError::from)
    }

    fn apply_request_memory(request: &GenerationRequest, options: &mut T2iOptions) {
        let memory = request.memory;
        options.attention_score_budget = memory
            .filter(|memory| memory.chunk_attention)
            .and_then(|memory| memory.attention_chunk_size)
            .map(u64::from);
        options.transformer_window_size = memory
            .filter(|memory| memory.stream_transformer_blocks)
            .and_then(|memory| memory.transformer_window_size)
            .map(|window| window as usize);
    }

    fn finish_direct<T>(
        scope: &mut dyn gen_core::MemoryRequestScope,
        result: Result<T>,
    ) -> Result<T> {
        let outcome = match &result {
            Ok(_) => gen_core::MemoryRunOutcome::Complete,
            Err(CandleError::Canceled) => gen_core::MemoryRunOutcome::Canceled,
            Err(error) => gen_core::MemoryRunOutcome::Error {
                message: error.to_string(),
            },
        };
        scope.finish(outcome).map_err(CandleError::from)?;
        result
    }

    fn run_direct_scope<T>(
        scope: &mut dyn gen_core::MemoryRequestScope,
        request: &mut GenerationRequest,
        operation: impl FnOnce(&GenerationRequest) -> Result<T>,
    ) -> Result<T> {
        let result = scope
            .configure_request(request)
            .map_err(CandleError::from)
            .and_then(|()| operation(request));
        Self::finish_direct(scope, result)
    }

    /// Run VQA only under the exact request that was admitted. The concrete operation is invoked
    /// internally, so it cannot discard the configured memory selection or cross the admitted
    /// reference count. The request scope is finished exactly once on every terminal path.
    #[allow(clippy::too_many_arguments)]
    pub fn run_vqa(
        &self,
        context: &gen_core::MemoryRunContext,
        mut request: GenerationRequest,
        question: &str,
        images: &[Tensor],
        max_new_tokens: usize,
        sampler: Sampler,
        mut options: T2iOptions,
        cancel: Option<&CancelFlag>,
    ) -> Result<String> {
        self.validate_direct_request(
            context,
            gen_core::MemoryMode::Other("vqa".into()),
            &request,
            images.len(),
        )?;
        let mut scope = self
            .begin_memory_strategy_request(context)
            .map_err(CandleError::from)?;
        Self::run_direct_scope(&mut *scope, &mut request, |configured_request| {
            Self::apply_request_memory(configured_request, &mut options);
            self.with_runtime(|model, tokenizer| {
                model.vqa_with_options(
                    tokenizer,
                    question,
                    images,
                    max_new_tokens,
                    sampler,
                    &options,
                    cancel,
                )
            })
        })
    }

    /// Run Document Studio interleave under the exact admitted request and lifecycle scope.
    #[allow(clippy::too_many_arguments)]
    pub fn run_interleave(
        &self,
        context: &gen_core::MemoryRunContext,
        mut request: GenerationRequest,
        prompt: &str,
        input_images: &[Tensor],
        width: usize,
        height: usize,
        mut options: T2iOptions,
        system_message: &str,
        max_new_tokens: usize,
        max_images: usize,
        cancel: &CancelFlag,
    ) -> Result<InterleaveOutput> {
        validate_interleave_count(&request, max_images)?;
        if request.width != width as u32 || request.height != height as u32 {
            return Err(CandleError::Msg(format!(
                "sensenova: interleave execution {width}x{height} does not match request {}x{}",
                request.width, request.height
            )));
        }
        self.validate_direct_request(
            context,
            gen_core::MemoryMode::Other("interleave".into()),
            &request,
            input_images.len(),
        )?;
        let mut scope = self
            .begin_memory_strategy_request(context)
            .map_err(CandleError::from)?;
        Self::run_direct_scope(&mut *scope, &mut request, |configured_request| {
            Self::apply_request_memory(configured_request, &mut options);
            self.with_runtime(|model, tokenizer| {
                model.interleave_gen(
                    tokenizer,
                    prompt,
                    input_images,
                    width,
                    height,
                    &options,
                    system_message,
                    max_new_tokens,
                    max_images,
                    cancel,
                )
            })
        })
    }
}

fn validate_interleave_count(request: &GenerationRequest, max_images: usize) -> Result<()> {
    if !(1..=10).contains(&max_images) || request.count != max_images as u32 {
        return Err(CandleError::Msg(format!(
            "sensenova: interleave max_images must be 1..=10 and exactly match admitted request count (max_images={max_images}, count={})",
            request.count
        )));
    }
    Ok(())
}

pub fn load_understanding_with_spec(spec: &LoadSpec) -> Result<SenseNovaUnderstanding> {
    memory_strategy::validate_load_spec(MODEL_ID, spec).map_err(CandleError::from)?;
    memory_strategy::validate_resolved_artifact_binding(spec).map_err(CandleError::from)?;
    let WeightsSource::Dir(root) = &spec.weights else {
        unreachable!("validated directory source")
    };
    let inventory =
        memory_strategy::CheckpointInventory::capture(root).map_err(CandleError::from)?;
    inventory
        .validate_numeric_tier(spec)
        .map_err(CandleError::from)?;
    let cfg = NeoChatConfig::from_dir(root)?;
    let device = candle_gen::default_device()?;
    let vb = backbone_vb(root, &device)?;
    let model = if memory_strategy::streamable_spec(MODEL_ID, spec) {
        T2iModel::from_weights_with_deferred_gen(&vb, &cfg, inventory.clone())?
    } else {
        T2iModel::from_weights(&vb, &cfg)?
    };
    let tokenizer = SenseNovaTokenizer::from_dir(root)?;
    let contract = memory_strategy::provider_contract(MODEL_ID, spec).map_err(CandleError::from)?;
    inventory.ensure_unchanged().map_err(CandleError::from)?;
    Ok(SenseNovaUnderstanding {
        model,
        tokenizer,
        device,
        spec: spec.clone(),
        contract,
        inventory,
    })
}

/// Construct the (lazy) base candle SenseNova-U1 generator (`sensenova_u1_8b`).
pub fn load(spec: &LoadSpec) -> gen_core::Result<Box<dyn Generator>> {
    load_inner(spec, false)
}

/// Construct the (lazy) 8-step distilled generator (`sensenova_u1_8b_fast`).
pub fn load_fast(spec: &LoadSpec) -> gen_core::Result<Box<dyn Generator>> {
    load_inner(spec, true)
}

fn load_inner(spec: &LoadSpec, fast: bool) -> gen_core::Result<Box<dyn Generator>> {
    let id = if fast { MODEL_ID_FAST } else { MODEL_ID };
    memory_strategy::validate_load_spec(id, spec)?;
    memory_strategy::validate_resolved_artifact_binding(spec)?;
    let root = match &spec.weights {
        WeightsSource::Dir(p) => p.clone(),
        WeightsSource::File(_) => {
            return Err(gen_core::Error::Msg(format!(
                "{id} expects a SenseNova-U1-8B-MoT snapshot directory, not a single .safetensors file"
            )));
        }
    };
    if spec.resolved_route.is_some() && !root.is_dir() {
        return Err(gen_core::Error::Unsupported(format!(
            "{id}: a resolved public route requires an existing checkpoint directory at {}",
            root.display()
        )));
    }
    let inventory = root
        .is_dir()
        .then(|| memory_strategy::CheckpointInventory::capture(&root))
        .transpose()?;
    if let Some(inventory) = &inventory {
        inventory.validate_numeric_tier(spec)?;
    }
    // User-supplied LoRAs are unsupported on both ids — the distill LoRA is merged internally by the
    // fast loader, never stacked via `spec.adapters`.
    if !spec.adapters.is_empty() {
        return Err(gen_core::Error::Unsupported(format!(
            "{id}: user-supplied adapters are not supported (supports_lora=false)"
        )));
    }
    // NOTE (sc-14249): there is deliberately no `spec.quantize` reject here any more, and `quantize`
    // is deliberately never READ either. The SceneWorks turnkey ships pre-quantized `q4/`/`q8/`
    // tiers, and `quant::detect_linear` picks the precision from the WEIGHTS on disk (a `.scales`
    // sibling), so the tier is selected by the DIRECTORY the caller resolved — exactly as it is for
    // flux1/qwen/kolors. An accepted `Q4`/`Q8` is therefore a no-op on an already-packed tier rather
    // than an on-the-fly quantize, which is why the descriptor can honestly advertise both without
    // this crate owning a quantizer. A `bf16/` tier still loads dense at the checkpoint's own dtype.
    if spec.control.is_some() || !spec.extra_controls.is_empty() || spec.ip_adapter.is_some() {
        return Err(gen_core::Error::Unsupported(format!(
            "{id}: registered text/reference generator does not support control / IP-adapter overlays"
        )));
    }
    // Named-component contract (sc-13658/sc-13664): the fast variant reads the `distill_lora`
    // component; the base id reads none. Reject any unrecognized component key up front.
    let known: &[&str] = if fast { &["distill_lora"] } else { &[] };
    reject_unknown_components(spec, known, id)?;
    // Resolve (and existence-check) the fast variant's distill LoRA at load — from the caller-supplied
    // `distill_lora` component, else the co-located snapshot file — so a missing LoRA fails fast here,
    // not at first generate (sc-13664: no env side-channel, no HF-cache scan). A **pre-merged** turnkey
    // tier (`DISTILL_MERGED_MARKER` present, sc-8775/sc-13787) bakes the merge into its on-disk weights
    // and ships no LoRA, so it is exempt — resolve+merge is skipped and the dense weights load as-is
    // (mirrors `mlx-gen-sensenova`'s `load_inner`). The base id needs no LoRA.
    let distill_lora = if fast && !root.join(DISTILL_MERGED_MARKER).exists() {
        Some(resolve_distill_lora(
            spec.components.get("distill_lora"),
            &root,
        )?)
    } else {
        None
    };
    let device = candle_gen::default_device()?;
    // ONE contract for the loaded spec, on every load shape and from every entry point (this
    // generator seam and `load_understanding_with_spec`). `provider_contract` reads the component
    // bytes off the same on-disk inventory an eager load already captured above, so an eager load
    // advertises the same real `asset_facts` as a deferred one for the same weights. Load shape is
    // still honored *inside* the contract: `build_contract` declares
    // `BoundedTransformerResidency` `Missing` on a non-streamable spec. Eager loads remain lazy
    // until the first generation request, so a not-yet-populated root is still contract-legal (no
    // inventory, zero bytes); an *existing* root must be complete and tier-consistent on both
    // shapes.
    let memory_strategy = memory_strategy::provider_contract(id, spec)?;
    Ok(Box::new(SenseNovaGenerator {
        descriptor: descriptor_for(id),
        root,
        device,
        fast,
        distill_lora,
        components: Mutex::new(None),
        loaded_spec: spec.clone(),
        memory_strategy,
        inventory,
    }))
}

// Link-time self-registration of both ids into gen-core's model registry.
candle_gen::register_generators! {
    pub(crate) const QUALITY_REGISTRATION = descriptor => load
}
candle_gen::register_generators! {
    pub(crate) const FAST_REGISTRATION = descriptor_fast => load_fast
}

/// Add all Candle SenseNova providers to an explicit media registry builder.
pub fn register_providers(
    registry: candle_gen::gen_core::ProviderRegistryBuilder,
) -> candle_gen::gen_core::ProviderRegistryBuilder {
    let registry = registry
        .register_generator(QUALITY_REGISTRATION)
        .register_generator(FAST_REGISTRATION);
    let registry = register_memory_contract_surfaces(registry);
    registry
        .register_memory_behavior(QUALITY_MEMORY_BEHAVIOR)
        .register_memory_behavior(FAST_MEMORY_BEHAVIOR)
}

pub fn register_memory_contract_surfaces(
    registry: gen_core::ProviderRegistryBuilder,
) -> gen_core::ProviderRegistryBuilder {
    registry
        .register_memory_strategy(QUALITY_MEMORY_REGISTRATION)
        .register_memory_contract_fixture(gen_core::MemoryContractFixtureRegistration {
            surface_specs: gen_core::candle_memory_contract_surface_specs,
            provider_id: MODEL_ID,
            contract: |spec| memory_strategy::weights_free_contract(MODEL_ID, spec),
        })
        .register_memory_strategy(FAST_MEMORY_REGISTRATION)
        .register_memory_contract_fixture(gen_core::MemoryContractFixtureRegistration {
            surface_specs: gen_core::candle_memory_contract_surface_specs,
            provider_id: MODEL_ID_FAST,
            contract: |spec| memory_strategy::weights_free_contract(MODEL_ID_FAST, spec),
        })
}

const QUALITY_MEMORY_REGISTRATION: gen_core::MemoryRegistration = gen_core::MemoryRegistration {
    provider_id: MODEL_ID,
    contract: |spec| memory_strategy::provider_contract(MODEL_ID, spec),
    safety_check: memory_strategy::registered_safety_check,
};

const FAST_MEMORY_REGISTRATION: gen_core::MemoryRegistration = gen_core::MemoryRegistration {
    provider_id: MODEL_ID_FAST,
    contract: |spec| memory_strategy::provider_contract(MODEL_ID_FAST, spec),
    safety_check: memory_strategy::registered_safety_check,
};

const QUALITY_MEMORY_BEHAVIOR: gen_core::MemoryBehaviorRegistration =
    gen_core::MemoryBehaviorRegistration {
        provider_id: MODEL_ID,
        valid_fixtures: memory_strategy::registered_valid_fixture,
        begin_request: |spec, contract, context| {
            memory_strategy::registered_begin_request(MODEL_ID, spec, contract, context)
        },
    };

const FAST_MEMORY_BEHAVIOR: gen_core::MemoryBehaviorRegistration =
    gen_core::MemoryBehaviorRegistration {
        provider_id: MODEL_ID_FAST,
        valid_fixtures: memory_strategy::registered_valid_fixture,
        begin_request: |spec, contract, context| {
            memory_strategy::registered_begin_request(MODEL_ID_FAST, spec, contract, context)
        },
    };

/// Build the complete explicit Candle SenseNova provider catalog.
pub fn provider_registry() -> candle_gen::gen_core::Result<candle_gen::gen_core::ProviderRegistry> {
    register_providers(candle_gen::gen_core::ProviderRegistryBuilder::new()).build()
}

#[cfg(test)]
mod explicit_registry_tests {
    #[test]
    fn explicit_catalog_has_stable_surface() {
        let registry = super::provider_registry().unwrap();
        let explicit: Vec<String> = registry
            .generators()
            .map(|registration| (registration.descriptor)().id.to_string())
            .collect();

        assert_eq!(explicit, ["sensenova_u1_8b", "sensenova_u1_8b_fast"]);
        let behaviors: Vec<&str> = registry
            .memory_behavior_registrations()
            .map(|registration| registration.provider_id)
            .collect();
        assert_eq!(behaviors, [super::MODEL_ID, super::MODEL_ID_FAST]);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use candle_gen::gen_core::{AdapterKind, AdapterSpec, Conditioning, Image, Quant};
    use std::collections::HashMap;

    #[derive(Default)]
    struct RecordingScope {
        configured: usize,
        outcomes: Vec<gen_core::MemoryRunOutcome>,
        reject_configure: bool,
    }

    impl gen_core::MemoryRequestScope for RecordingScope {
        fn configure_request(&mut self, request: &mut GenerationRequest) -> gen_core::Result<()> {
            self.configured += 1;
            if self.reject_configure {
                return Err(gen_core::Error::Unsupported("crossed request".into()));
            }
            request.memory = Some(gen_core::GenerationMemory {
                chunk_attention: true,
                attention_chunk_size: Some(123),
                ..Default::default()
            });
            Ok(())
        }

        fn enter_phase(&mut self, _phase: gen_core::MemoryPhase) -> gen_core::Result<()> {
            Ok(())
        }
        fn leave_phase(&mut self, _phase: gen_core::MemoryPhase) -> gen_core::Result<()> {
            Ok(())
        }
        fn configure_decode(
            &mut self,
            _tile_edge: u32,
            _overlap: u32,
            _geometry: gen_core::MemoryGeometry,
        ) -> gen_core::Result<()> {
            Ok(())
        }
        fn configure_attention(&mut self, _chunk_size: u32) -> gen_core::Result<()> {
            Ok(())
        }
        fn materialize_transformer_window(
            &mut self,
            _first_block: u32,
            _block_count: u32,
        ) -> gen_core::Result<()> {
            Ok(())
        }
        fn finish(&mut self, outcome: gen_core::MemoryRunOutcome) -> gen_core::Result<()> {
            if !self.outcomes.is_empty() {
                return Err(gen_core::Error::Msg("finished twice".into()));
            }
            self.outcomes.push(outcome);
            Ok(())
        }
    }

    fn write_minimal_dense_checkpoint(root: &Path) {
        let key = "language_model.model.layers.0.self_attn.k_proj.weight".to_owned();
        let tensor = Tensor::zeros((2, 64), DType::BF16, &Device::Cpu).unwrap();
        candle_gen::candle_core::safetensors::save(
            &HashMap::from([(key, tensor)]),
            root.join("model.safetensors"),
        )
        .unwrap();
        std::fs::write(root.join("config.json"), "{}").unwrap();
    }

    #[test]
    fn registers_both_ids_as_candle() {
        let ids: Vec<&str> = crate::provider_registry()
            .unwrap()
            .generators()
            .copied()
            .map(|r| (r.descriptor)().id)
            .collect();
        assert!(ids.contains(&MODEL_ID), "{MODEL_ID} not registered");
        assert!(
            ids.contains(&MODEL_ID_FAST),
            "{MODEL_ID_FAST} not registered"
        );

        let spec = LoadSpec::new(WeightsSource::Dir("/nonexistent".into()));
        let g = crate::provider_registry()
            .unwrap()
            .load(MODEL_ID, &spec)
            .expect("candle sensenova is registered");
        assert_eq!(g.descriptor().id, "sensenova_u1_8b");
        assert_eq!(g.descriptor().family, "sensenova-u1");
        assert_eq!(g.descriptor().backend, "candle");
        assert_eq!(g.descriptor().modality, Modality::Image);
    }

    #[test]
    fn typed_direct_scope_configures_and_finishes_every_terminal_outcome_once() {
        let mut complete = RecordingScope::default();
        let mut request = GenerationRequest::default();
        let value =
            SenseNovaUnderstanding::run_direct_scope(&mut complete, &mut request, |configured| {
                assert_eq!(
                    configured
                        .memory
                        .and_then(|memory| memory.attention_chunk_size),
                    Some(123)
                );
                Ok(7)
            })
            .unwrap();
        assert_eq!(value, 7);
        assert_eq!(complete.configured, 1);
        assert_eq!(complete.outcomes, [gen_core::MemoryRunOutcome::Complete]);

        let mut canceled = RecordingScope::default();
        let result: Result<()> = SenseNovaUnderstanding::run_direct_scope(
            &mut canceled,
            &mut GenerationRequest::default(),
            |_| Err(CandleError::Canceled),
        );
        assert!(matches!(result, Err(CandleError::Canceled)));
        assert_eq!(canceled.outcomes, [gen_core::MemoryRunOutcome::Canceled]);

        let mut failed = RecordingScope::default();
        let result: Result<()> = SenseNovaUnderstanding::run_direct_scope(
            &mut failed,
            &mut GenerationRequest::default(),
            |_| Err(CandleError::Msg("operation failed".into())),
        );
        assert!(result.is_err());
        assert!(matches!(
            failed.outcomes.as_slice(),
            [gen_core::MemoryRunOutcome::Error { message }] if message == "operation failed"
        ));

        let mut crossed = RecordingScope {
            reject_configure: true,
            ..Default::default()
        };
        let mut called = false;
        let result: Result<()> = SenseNovaUnderstanding::run_direct_scope(
            &mut crossed,
            &mut GenerationRequest::default(),
            |_| {
                called = true;
                Ok(())
            },
        );
        assert!(result.is_err());
        assert!(!called);
        assert_eq!(crossed.configured, 1);
        assert!(matches!(
            crossed.outcomes.as_slice(),
            [gen_core::MemoryRunOutcome::Error { .. }]
        ));
    }

    #[test]
    fn interleave_output_count_is_exactly_admission_bound() {
        let request = GenerationRequest {
            count: 4,
            ..Default::default()
        };
        assert!(validate_interleave_count(&request, 4).is_ok());
        for crossed in [0, 3, 5, 11] {
            assert!(validate_interleave_count(&request, crossed).is_err());
        }
    }

    /// One model, one set of bytes. The eager (non-streamable) load must publish the same real
    /// component bytes as the deferred load for the same weights — previously the eager branch
    /// built an `uncalibrated_contract` whose `asset_facts` were all zero, so an eager load of a
    /// ~35GB checkpoint advertised 0 bytes while the registry advertised full size.
    #[test]
    fn eager_and_deferred_contracts_publish_the_same_real_asset_bytes() {
        let tmp = tempfile::tempdir().unwrap();
        let root = tmp.path().join("sensenova-u1-8b-mlx").join("bf16");
        std::fs::create_dir_all(&root).unwrap();
        write_minimal_dense_checkpoint(&root);
        // The real value: the shard bytes actually on disk, not a placeholder.
        let on_disk = std::fs::metadata(root.join("model.safetensors"))
            .unwrap()
            .len();
        assert!(on_disk > 0);

        let eager = LoadSpec::new(WeightsSource::Dir(root.clone()))
            .with_resolved_route("sensenova_u1_8b")
            .with_load_shape(gen_core::LoadShape::EagerMaterialization);
        let deferred = LoadSpec::new(WeightsSource::Dir(root.clone()))
            .with_resolved_route("sensenova_u1_8b")
            .with_load_shape(gen_core::LoadShape::DeferredMaterialization);
        assert!(!memory_strategy::streamable_spec(MODEL_ID, &eager));
        assert!(memory_strategy::streamable_spec(MODEL_ID, &deferred));

        let facts = |spec: &LoadSpec| {
            load(spec)
                .unwrap()
                .memory_strategy_contract()
                .expect("SenseNova publishes a memory contract")
                .asset_facts
        };
        let eager_facts = facts(&eager);
        let deferred_facts = facts(&deferred);
        assert_eq!(eager_facts.base_bytes, on_disk);
        assert_eq!(eager_facts.transformer_bytes, on_disk);
        assert_eq!(eager_facts.conditioning_bytes, on_disk);
        assert_eq!(eager_facts, deferred_facts);
        // ...and the registry's own answer for the same spec is that same number.
        assert_eq!(
            memory_strategy::provider_contract(MODEL_ID, &eager)
                .unwrap()
                .asset_facts,
            eager_facts
        );
        // The load shape still shows up *inside* the contract, not as a different byte count.
        let rung = |spec: &LoadSpec| {
            load(spec)
                .unwrap()
                .memory_strategy_contract()
                .unwrap()
                .strategies
                .iter()
                .find(|capability| {
                    capability.strategy == gen_core::MemoryStrategy::BoundedTransformerResidency
                })
                .unwrap()
                .support
                .clone()
        };
        assert_eq!(rung(&eager), gen_core::MemoryStrategySupport::Missing);
        assert_eq!(
            rung(&deferred),
            gen_core::MemoryStrategySupport::Implemented
        );
    }

    #[test]
    fn eager_public_route_refuses_checkpoint_replacement_before_lazy_materialization() {
        let tmp = tempfile::tempdir().unwrap();
        let root = tmp.path().join("sensenova-u1-8b-mlx").join("bf16");
        std::fs::create_dir_all(&root).unwrap();
        write_minimal_dense_checkpoint(&root);
        let spec =
            LoadSpec::new(WeightsSource::Dir(root.clone())).with_resolved_route("sensenova_u1_8b");
        let generator = load(&spec).unwrap();

        let replacement = root.join("replacement.safetensors.tmp");
        std::fs::write(&replacement, [7_u8; 8]).unwrap();
        std::fs::rename(&replacement, root.join("model.safetensors")).unwrap();
        let request = GenerationRequest {
            prompt: "must fail before config/model materialization".into(),
            width: 512,
            height: 512,
            ..Default::default()
        };
        let error = generator
            .generate(&request, &mut |_| {})
            .expect_err("replaced pinned artifact must fail closed")
            .to_string();
        assert!(
            error.contains("changed") || error.contains("identity"),
            "{error}"
        );
    }

    #[test]
    fn descriptor_advertises_t2i_and_it2i_surface() {
        let d = descriptor();
        assert!(d.capabilities.supports_guidance);
        assert!(d.capabilities.supports_true_cfg);
        assert!(!d.capabilities.mac_only);
        assert!(d.capabilities.accepts(ConditioningKind::Reference));
        assert!(d.capabilities.accepts(ConditioningKind::MultiReference));
        assert!(!d.capabilities.supports_lora);
        assert!(!d.capabilities.supports_lokr);
        // sc-14249: the turnkey's pre-quantized q4/q8 tiers load natively (packed-detect per
        // projection), so both tiers ARE advertised — a Q4/Q8 here selects which tier subdir the
        // caller resolved, it does not ask this crate to quantize anything.
        assert_eq!(d.capabilities.supported_quants, &[Quant::Q4, Quant::Q8]);
        assert!(d.capabilities.supports_kv_cache);
        assert!(d.capabilities.requires_sigma_shift);
        // sc-16960: the bespoke denoise loop emits per-step frames on BOTH ids.
        assert!(d.capabilities.supports_preview);
        assert!(descriptor_fast().capabilities.supports_preview);
        // The fast variant shares the capability surface; only id + defaults differ.
        let f = descriptor_fast();
        assert_eq!(f.id, MODEL_ID_FAST);
        assert_eq!(f.family, d.family);
        assert_eq!(f.capabilities.max_size, d.capabilities.max_size);
        assert_eq!(
            f.capabilities.supported_quants,
            d.capabilities.supported_quants
        );
    }

    #[test]
    fn validate_accepts_t2i_and_reference_shapes_and_rejects_invalid_guidance() {
        let spec = LoadSpec::new(WeightsSource::Dir("/nonexistent".into()));
        let g = crate::provider_registry()
            .unwrap()
            .load(MODEL_ID, &spec)
            .unwrap();

        let ok = GenerationRequest {
            prompt: "a cat holding a lit candle".into(),
            width: 512,
            height: 512,
            guidance: Some(4.0),
            ..Default::default()
        };
        assert!(g.validate(&ok).is_ok());
        let reference = GenerationRequest {
            true_cfg: Some(1.5),
            conditioning: vec![Conditioning::Reference {
                image: Image::default(),
                strength: None,
            }],
            ..ok.clone()
        };
        assert!(g.validate(&reference).is_ok());
        let multi = GenerationRequest {
            conditioning: vec![Conditioning::MultiReference {
                images: vec![Image::default(), Image::default()],
            }],
            ..ok.clone()
        };
        assert!(g.validate(&multi).is_ok());

        for bad in [
            GenerationRequest {
                width: 512,
                height: 512,
                ..Default::default()
            }, // empty prompt
            GenerationRequest {
                prompt: "x".into(),
                width: 300, // not a multiple of 32
                height: 512,
                ..Default::default()
            },
            GenerationRequest {
                prompt: "x".into(),
                width: 512,
                height: 512,
                steps: Some(0),
                ..Default::default()
            },
            GenerationRequest {
                prompt: "x".into(),
                width: 512,
                height: 512,
                true_cfg: Some(1.5),
                ..Default::default()
            },
        ] {
            assert!(g.validate(&bad).is_err(), "should reject: {bad:?}");
        }

        // sc-12612: `SIZE_MULTIPLE` is the pinned stride SceneWorks ties every advertised SenseNova
        // bucket to. Pin the value and mutation-check that a size which is a multiple of 16 but not
        // SIZE_MULTIPLE (32) is still rejected with the stride error, and an on-stride size passes.
        assert_eq!(SIZE_MULTIPLE, 32);
        let off_stride = g
            .validate(&GenerationRequest {
                prompt: "x".into(),
                width: 1040, // 65×16 — a multiple of 16 but not SIZE_MULTIPLE
                height: 512,
                ..Default::default()
            })
            .unwrap_err()
            .to_string();
        assert!(
            off_stride.contains("multiples of 32"),
            "expected the stride error, got: {off_stride}"
        );
        assert!(g
            .validate(&GenerationRequest {
                prompt: "x".into(),
                width: 1024, // 32×32 — on-stride
                height: 512,
                ..Default::default()
            })
            .is_ok());
    }

    #[test]
    fn validate_rejects_empty_multi_reference_without_falling_back_to_t2i() {
        let spec = LoadSpec::new(WeightsSource::Dir("/nonexistent".into()));
        let g = crate::provider_registry()
            .unwrap()
            .load(MODEL_ID, &spec)
            .unwrap();
        let empty = GenerationRequest {
            prompt: "use these references".into(),
            width: 512,
            height: 512,
            // Deliberately no true_cfg: the empty shape must fail on its own instead of passing as
            // an unconditioned T2I request.
            conditioning: vec![Conditioning::MultiReference { images: vec![] }],
            ..Default::default()
        };

        let err = g
            .validate(&empty)
            .expect_err("empty MultiReference must fail");
        assert!(matches!(err, gen_core::Error::Unsupported(_)));
        assert_eq!(
            err.to_string(),
            "unsupported: sensenova_u1_8b: MultiReference conditioning requires at least one image"
        );
    }

    #[test]
    fn validate_rejects_reserved_image_context_marker_as_typed_error() {
        let spec = LoadSpec::new(WeightsSource::Dir("/nonexistent".into()));
        let g = crate::provider_registry()
            .unwrap()
            .load(MODEL_ID, &spec)
            .unwrap();
        let request = GenerationRequest {
            prompt: "preserve the literal <IMG_CONTEXT> label".into(),
            width: 512,
            height: 512,
            conditioning: vec![Conditioning::Reference {
                image: Image::default(),
                strength: None,
            }],
            ..Default::default()
        };

        let err = g
            .validate(&request)
            .expect_err("reserved tokenizer markers must never reach the position builder");
        assert!(matches!(err, gen_core::Error::Unsupported(_)));
        assert_eq!(
            err.to_string(),
            "unsupported: sensenova_u1_8b: prompt contains reserved internal image marker <IMG_CONTEXT>"
        );
    }

    #[test]
    fn validate_strength_fields_are_fail_closed_and_null_is_accepted() {
        let spec = LoadSpec::new(WeightsSource::Dir("/nonexistent".into()));
        let g = crate::provider_registry()
            .unwrap()
            .load(MODEL_ID, &spec)
            .unwrap();
        let reference = Image {
            width: 1,
            height: 1,
            pixels: vec![0, 0, 0],
        };
        let null_strengths = GenerationRequest {
            prompt: "edit this reference".into(),
            width: 512,
            height: 512,
            strength: None,
            conditioning: vec![Conditioning::Reference {
                image: reference.clone(),
                strength: None,
            }],
            ..Default::default()
        };
        assert!(
            g.validate(&null_strengths).is_ok(),
            "absent/null strength carriers preserve the supported reference shape"
        );

        let request_strength = GenerationRequest {
            strength: Some(0.6),
            ..null_strengths.clone()
        };
        let err = g
            .validate(&request_strength)
            .expect_err("request-level strength must not be discarded");
        assert!(matches!(err, gen_core::Error::Unsupported(_)));
        assert_eq!(
            err.to_string(),
            "unsupported: sensenova_u1_8b: request-level strength is unsupported; omit it or set it to null"
        );

        let per_reference_strength = GenerationRequest {
            conditioning: vec![Conditioning::Reference {
                image: reference.clone(),
                strength: Some(0.4),
            }],
            ..null_strengths.clone()
        };
        let err = g
            .validate(&per_reference_strength)
            .expect_err("per-reference strength must not be discarded");
        assert!(matches!(err, gen_core::Error::Unsupported(_)));
        assert_eq!(
            err.to_string(),
            "unsupported: sensenova_u1_8b: Conditioning::Reference.strength is unsupported; omit it or set it to null"
        );

        // Per-reference strength is the specific carrier and therefore wins when both are present;
        // this pins the request-level fallback precedence documented by GenerationRequest.
        let both = GenerationRequest {
            strength: Some(0.8),
            conditioning: vec![Conditioning::Reference {
                image: reference,
                strength: Some(0.4),
            }],
            ..null_strengths
        };
        assert_eq!(
            g.validate(&both).unwrap_err().to_string(),
            "unsupported: sensenova_u1_8b: Conditioning::Reference.strength is unsupported; omit it or set it to null"
        );
    }

    #[test]
    fn reference_preprocess_rejects_malformed_pixels_and_preserves_a_bounded_grid() {
        assert!(image_to_chw01(&Image::default()).is_err());

        let image = Image {
            width: 4,
            height: 2,
            pixels: vec![127; 4 * 2 * 3],
        };
        let tensor = image_to_chw01(&image).unwrap();
        let (channels, height, width) = tensor.dims3().unwrap();
        assert_eq!(channels, 3);
        assert!(height.is_multiple_of(SIZE_MULTIPLE as usize));
        assert!(width.is_multiple_of(SIZE_MULTIPLE as usize));
        let area = height * width;
        assert!(area >= REF_MIN_PIXELS as usize);
        assert!(area <= REF_MAX_PIXELS as usize);
        let values = tensor.flatten_all().unwrap().to_vec1::<f32>().unwrap();
        assert!(values.iter().all(|value| (0.0..=1.0).contains(value)));
    }

    /// sc-9029 / F-045: `options` must route the timestep shift through the checkpoint config, not a
    /// hardcoded 3.0. Pins the resolved `T2iOptions.timestep_shift` against request/config precedence.
    #[test]
    fn options_resolves_timestep_shift_from_request_then_config() {
        let loaded_spec = LoadSpec::new(WeightsSource::Dir("/nonexistent".into()));
        let gen = SenseNovaGenerator {
            descriptor: descriptor(),
            root: "/nonexistent".into(),
            device: candle_gen::default_device().unwrap(),
            fast: false,
            distill_lora: None,
            components: Mutex::new(None),
            memory_strategy: memory_strategy::weights_free_contract(MODEL_ID, &loaded_spec)
                .unwrap(),
            loaded_spec,
            inventory: None,
        };
        let req = GenerationRequest {
            prompt: "a cat".into(),
            width: 512,
            height: 512,
            ..Default::default()
        };

        // Shipped 8B-MoT (config timestep_shift = 1.0 identity) → product default 3.0, unchanged render.
        let shipped = crate::config::mot_8b();
        assert_eq!(gen.options(&req, &shipped, 0).timestep_shift, 3.0);

        // An explicit request scheduler_shift wins over everything.
        let req_shift = GenerationRequest {
            scheduler_shift: Some(2.5),
            ..req.clone()
        };
        assert_eq!(gen.options(&req_shift, &shipped, 0).timestep_shift, 2.5);

        // A checkpoint variant declaring its own inference shift is honored (no longer shadowed).
        let variant = crate::config::variant_with_timestep_shift(7.0);
        assert_eq!(gen.options(&req, &variant, 0).timestep_shift, 7.0);
        // ...but an explicit request still overrides the variant's config value.
        assert_eq!(gen.options(&req_shift, &variant, 0).timestep_shift, 2.5);
    }

    #[test]
    fn load_rejects_unwired_surfaces_and_single_file() {
        // Use a guaranteed-missing path: `/snap` is an existing system directory on Ubuntu CI, and
        // existing directories are intentionally inspected for exact artifact-tier provenance.
        let temp = tempfile::tempdir().unwrap();
        let missing_snapshot = temp.path().join("missing-snapshot");
        let lora = LoadSpec::new(WeightsSource::Dir(missing_snapshot.clone())).with_adapters(vec![
            AdapterSpec::new("/lora.safetensors".into(), 1.0, AdapterKind::Lora),
        ]);
        assert!(matches!(
            load(&lora).err().expect("err"),
            gen_core::Error::Unsupported(_)
        ));
        // The fast loader rejects user adapters too (its distill LoRA is internal, not user-supplied).
        assert!(matches!(
            load_fast(&lora).err().expect("err"),
            gen_core::Error::Unsupported(_)
        ));

        // sc-14249: a tier `Quant` is ACCEPTED now (it was `Unsupported` while the loader was dense-f32
        // only). Both tiers load lazily against a not-yet-present resolved dir; once a directory
        // exists, its converter-written provenance and tensor dtypes are validated before load.
        for q in [Quant::Q4, Quant::Q8] {
            let quant = LoadSpec::new(WeightsSource::Dir(missing_snapshot.clone())).with_quant(q);
            assert!(load(&quant).is_ok(), "{q:?} tier select must be accepted");
        }

        let single = LoadSpec::new(WeightsSource::File("/x.safetensors".into()));
        let err = load(&single).err().expect("err").to_string();
        assert!(err.contains("snapshot directory"), "got: {err}");
    }

    /// sc-13664: the fast variant resolves its distill LoRA from caller-supplied paths only (the
    /// `distill_lora` component, else the co-located snapshot file) — no `$SENSENOVA_DISTILL_LORA`, no
    /// HF-cache scan. A fast load with neither staged fails **at load** (fail-fast) with an actionable
    /// error naming the component. Weights-free: `/nonexistent` has no co-located LoRA.
    #[test]
    fn fast_load_requires_distill_lora_path() {
        let spec = LoadSpec::new(WeightsSource::Dir("/nonexistent".into()));
        let err = load_fast(&spec).err().expect("err").to_string();
        assert!(err.contains("distill_lora"), "got: {err}");
        assert!(!err.contains("SENSENOVA_DISTILL_LORA"), "got: {err}");
        // The base id needs no LoRA — it loads lazily against the same missing path.
        assert!(load(&spec).is_ok(), "base load stays lazy");
    }

    /// sc-13787 (epic 13678): the **inverse** of [`fast_load_requires_distill_lora_path`]. When a fast
    /// tier ships the pre-merged `distill_merged.json` marker (a turnkey snapshot bakes the 8-step
    /// distill merge into its on-disk weights, sc-8775), the loader must SKIP resolving the distill
    /// LoRA — so a fast load succeeds with the marker present and NO LoRA staged (no `distill_lora`
    /// component, no co-located file), where the marker-less load above fails. Mirrors
    /// `mlx-gen-sensenova`'s marker-guarded load. Weights-free: load is lazy, so this exercises only the
    /// resolve-skip, no backbone I/O.
    #[test]
    fn fast_load_skips_distill_lora_when_premerged_marker_present() {
        let dir_tmp = tempfile::tempdir().unwrap();
        let dir = dir_tmp.path().to_path_buf();
        write_minimal_dense_checkpoint(&dir);
        // The marker keys off existence only; an empty file is enough. No LoRA file, no component.
        std::fs::write(dir.join(DISTILL_MERGED_MARKER), b"").expect("write marker");

        let spec = LoadSpec::new(WeightsSource::Dir(dir.clone()));
        let loaded = load_fast(&spec);
        // Clean up before asserting so a failing assert doesn't leak the tempdir.
        assert!(
            loaded.is_ok(),
            "pre-merged fast tier (marker present, no LoRA) must load, got: {:?}",
            loaded.err()
        );
    }

    /// sc-13664: a staged-but-nonexistent `distill_lora` component errors at load, naming the
    /// component (not a bare later I/O error), and a `Dir` staged where a file is required is rejected.
    #[test]
    fn fast_load_validates_distill_lora_component() {
        let missing = LoadSpec::new(WeightsSource::Dir("/nonexistent".into())).with_component(
            "distill_lora",
            WeightsSource::File("/nope/lora.safetensors".into()),
        );
        let err = load_fast(&missing).err().expect("err").to_string();
        assert!(err.contains("distill_lora"), "got: {err}");
        assert!(err.contains("does not exist"), "got: {err}");

        let as_dir = LoadSpec::new(WeightsSource::Dir("/nonexistent".into()))
            .with_component("distill_lora", WeightsSource::Dir("/some/dir".into()));
        let err = load_fast(&as_dir).err().expect("err").to_string();
        assert!(err.contains("distill_lora"), "got: {err}");
    }

    /// sc-13658/sc-13664: an unrecognized component key is rejected at load (typed `Unsupported`).
    #[test]
    fn fast_load_rejects_unknown_component() {
        let spec = LoadSpec::new(WeightsSource::Dir("/nonexistent".into())).with_component(
            "bogus_component",
            WeightsSource::File("/x.safetensors".into()),
        );
        assert!(matches!(
            load_fast(&spec).err().expect("err"),
            gen_core::Error::Unsupported(_)
        ));
    }
}
