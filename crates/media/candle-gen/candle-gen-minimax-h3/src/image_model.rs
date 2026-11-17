//! The **`minimax_h3_image`** generator on candle/CUDA (SW-7091): MiniMax-H3-Image
//! text-to-image at bf16, prompt → one image.
//!
//! Its own engine id, [`IMAGE_MODEL_ID`], with an **image** descriptor — none of H3 video's
//! audio/video, frame, keyframe or reference surface, and none of its canvas caps
//! (`CANVAS_MAX_PIXELS`, `MAX_CANVAS_EDGE`): the request contract is `gen_core::h3_image`.
//!
//! # Phases
//!
//! A staged render runs exactly as H3's does — at bf16 the layer-50 Qwen3-VL-32B conditioner
//! (~58 GB at the tap) and the 66 GB DiT cannot be resident together on a 96 GB card:
//!
//! 1. **conditioning** — tokenize the prompt verbatim (no chat template, no special tokens,
//!    truncated to 4096 tokens), run H3's text-only conditioner pass over the byte-identical
//!    `text_encoder/` (this snapshot's, or MiniMax-H3's staged as the `text_encoder` component) (`crate::model::run_conditioner`), drop it and synchronize;
//! 2. **denoise** — map `transformer_diffusers/` (no audio tensors), evict `adaln_proj` after
//!    projecting the six fixed timesteps' modulation ([`crate::dit::ImageDit`]), run the six-step
//!    velocity loop ([`crate::image_pipeline::denoise_image`]) with step `i` through head `i`,
//!    drop it and synchronize;
//! 3. **decode** — map `vae_diffusers/`'s decode half ([`MiniMaxH3ImageVae`]), decode with its
//!    shipped 256 / 64 tiling, and resize back from the generation canvas to the requested size.
//!
//! A request whose admitted selection stages or streams (`GenerationMemory::stage_residency` /
//! `stream_transformer_blocks`) always runs those phases. A **resident** request (no request
//! memory, or one that neither stages nor streams) keeps the three components warm instead when
//! they fit together on the device: the conditioner, the DiT after its AdaLN precompute-and-evict,
//! and the VAE decode half stay in [`MiniMaxH3Image`] between requests, so the next render reuses
//! them rather than re-reading them from disk. "Fit" is the header-derived
//! `admission::warm_estimate` (every phase plus the other components' warm weights) plus a 2 GiB
//! reserve against the device's free memory plus what is already held warm. A resident request
//! whose warm set does not fit (bf16 on a 96 GB card: ~126 GB) releases any warm components and
//! runs staged, exactly as before — which is why rung 0 is admitted at the staged price.
//!
//! # Streamed route (rung 4, SW-7095)
//!
//! On a [`LoadShape::DeferredMaterialization`] load, a request whose admitted selection streams
//! the transformer blocks keeps the text encoder and the DiT on the host in runtime format and
//! runs each through a one-block device window (`crate::block_stream`); the DiT's root modules
//! and six heads stay on the device, and the AdaLN precompute projects one block at a time.
//!
//! # Size
//!
//! Any positive size: the image is generated on `gen_core::h3_image::generation_canvas` (a
//! 32-multiple canvas at least `min_generation_area` = 4 MP by default) and resized back. A
//! `0 x 0` request is the upstream 2048 x 2048 default.

use std::path::{Path, PathBuf};

use candle_gen::candle_core::{DType, Device, Tensor};
use candle_gen::gen_core::h3_image as contract;
use candle_gen::gen_core::{
    Capabilities, ConditioningKind, GenerationOutput, GenerationRequest, Generator, LoadShape,
    LoadSpec, MemoryProviderContract, Modality, ModelDescriptor, OffloadPolicy, Precision,
    Progress, Quant, SizeFloor, StepSupport, WeightsSource,
};
use candle_gen::{CandleError, Result};
use gen_core_h3::h3_schedule::SigmaSchedule;

use crate::dit::{ImageDit, MiniMaxH3Dit, PreparedImageDit};
use crate::image_pipeline::{
    decoded_to_image, denoise_image, initial_image_rows, latent_grid, reference_condition_rows,
    reference_noise, reference_vae_pixels, resize_reference, unpack_image_rows, ImageLayout,
};
use crate::image_vae::{MiniMaxH3ImageVae, MiniMaxH3ImageVaeEncoder, IMAGE_VAE_COMPONENT};
use crate::model::{
    conditioner_forward, require_component_shards, resolve_seed, run_conditioner, PhaseLog,
};
use crate::text_encoder::MiniMaxH3Tokenizer;

/// Caption context, per-row modality tags, and ordered reference latents.
type ImageConditioning = (Tensor, Vec<u32>, Vec<Tensor>);

/// The engine id. A separate family from H3 video (`minimax_h3`): a different checkpoint, a
/// different modality and a different request contract.
pub const IMAGE_MODEL_ID: &str = "minimax_h3_image";

/// The DiT partition: the diffusers-converted `MiniMaxH3ImageTransformer2DModel`. The native
/// `transformer/` copy is never read.
pub const IMAGE_DIT_PARTITION: &str = "transformer_diffusers";

/// The conditioner partition — the same Qwen3-VL-32B shards MiniMax-H3 ships — and the
/// `LoadSpec::components` key that stages it from elsewhere (an H3 snapshot's `text_encoder/`),
/// the MLX twin's `TEXT_ENCODER_COMPONENT`.
pub const IMAGE_TEXT_ENCODER: &str = "text_encoder";

/// The conditioner directory a load reads: the staged [`IMAGE_TEXT_ENCODER`] component when it is
/// a directory, else `root/text_encoder`.
pub fn resolve_text_encoder_dir(root: &Path, spec: &LoadSpec) -> PathBuf {
    match spec.components.get(IMAGE_TEXT_ENCODER) {
        Some(WeightsSource::Dir(dir)) => dir.clone(),
        _ => root.join(IMAGE_TEXT_ENCODER),
    }
}

/// The DiT directory a load reads: the staged [`IMAGE_DIT_PARTITION`] component (a packed tier's
/// `transformer_diffusers/`, SW-7093) when it is a directory, else `root/transformer_diffusers`.
pub fn resolve_dit_dir(root: &Path, spec: &LoadSpec) -> PathBuf {
    match spec.components.get(IMAGE_DIT_PARTITION) {
        Some(WeightsSource::Dir(dir)) => dir.clone(),
        _ => root.join(IMAGE_DIT_PARTITION),
    }
}

/// The upstream default output size when a request names none (`height = width = 2048`).
pub const DEFAULT_SIZE: u32 = 2048;

/// The largest edge the descriptor advertises. Upstream has **no** maximum; gen-core needs a
/// finite range, and the memory admission is the real bound, so this is a generous ceiling
/// rather than a published limit.
pub const MAX_EDGE: u32 = 16_384;

/// The identity, modality and capability surface.
pub fn image_descriptor() -> ModelDescriptor {
    ModelDescriptor {
        encoder_contract: None,
        denoiser_output_latent_space: None,
        control_kinds: None,
        required_components: &[],
        id: IMAGE_MODEL_ID,
        family: "minimax_h3_image",
        backend: "candle",
        modality: Modality::Image,
        capabilities: Capabilities {
            // Guidance-distilled: no negative prompt, no CFG (`gen_core::h3_image` refuses both).
            supports_negative_prompt: false,
            supports_guidance: false,
            supports_true_cfg: false,
            // `edit` (SW-7092): 1..=16 reference images, as single references and/or a
            // multi-reference list, in request order. No reference is `text2image`.
            conditioning: vec![
                ConditioningKind::Reference,
                ConditioningKind::MultiReference,
            ],
            // SW-7096: H3's adapter install over the image DiT's block/refiner surface --
            // diffusers/PEFT, kohya `lora_unet_`, ComfyUI and LoKr, each at its own scale.
            supports_lora: true,
            supports_lokr: true,
            // Pre-quantized MLX-affine tiers (SW-7093), the artifacts the MLX lane reads: staged as
            // the DiT component; `spec.quantize` asserts which tier was staged.
            supported_quants: &[Quant::Q4, Quant::Q8],
            // A staged request maps and releases its three components in turn (see the module docs).
            supports_sequential_offload: true,
            max_count: 1,
            // PDD-distilled: one output head per step, so six is the only legal count.
            supported_steps: StepSupport::Exact(vec![contract::FIXED_STEPS]),
            component_precision_floors: &[],
            // SW-7106: ComfyUI's stochastic LCM (the default, 1.1 → 0.9) or the released Euler on
            // the same six-step grid, named by `sampler` and parameterized by `h3_image_sampler`.
            samplers: contract::SAMPLER_NAMES.to_vec(),
            supports_h3_image_sampler: true,
            schedulers: Vec::new(),
            min_size: 1,
            max_size: MAX_EDGE,
            // Any positive size: the canvas rounding is internal and resized back.
            size_floor: SizeFloor::RangeChecked,
            ..Default::default()
        },
    }
}

/// The components a resident request keeps on the device between requests.
#[derive(Default)]
struct WarmComponents {
    /// The layer-50 conditioner.
    conditioner: Option<crate::text_encoder::MiniMaxH3TextEncoder>,
    /// The DiT after the AdaLN precompute-and-evict, keyed on the timesteps and reference presence
    /// its modulation was projected for.
    dit: Option<crate::dit::PreparedImageDit>,
    /// The image VAE's decode half.
    decoder: Option<MiniMaxH3ImageVae>,
    /// The per-component warm weights of the resident request that last kept this set.
    priced: Option<crate::image_memory_strategy::admission::WarmWeights>,
}

impl WarmComponents {
    fn is_empty(&self) -> bool {
        self.conditioner.is_none() && self.dit.is_none() && self.decoder.is_none()
    }

    /// The held components' bytes at `weights`.
    fn held_bytes(&self, weights: &crate::image_memory_strategy::admission::WarmWeights) -> u64 {
        [
            self.conditioner.as_ref().map(|_| weights.conditioner),
            self.dit.as_ref().map(|_| weights.dit),
            self.decoder.as_ref().map(|_| weights.decoder),
        ]
        .into_iter()
        .flatten()
        .fold(0u64, u64::saturating_add)
    }
}

/// Whether a request may keep its components warm: one whose request memory neither stages nor
/// streams ([`crate::warm::keeps_components_warm`]), or one with no request memory (a resident
/// admission carries none) on a generator not loaded with [`OffloadPolicy::Sequential`]. Without
/// request memory the load-time policy stands, and a sequential load asked for each component to
/// be dropped after its phase.
fn keeps_components_warm(req: &GenerationRequest, load_policy: OffloadPolicy) -> bool {
    match req.memory {
        Some(_) => crate::warm::keeps_components_warm(req),
        None => load_policy != OffloadPolicy::Sequential,
    }
}

/// The loaded `minimax_h3_image` provider. A staged or streamed request maps and releases each
/// component in its own phase; a resident request keeps them warm in its `warm` field (see the
/// module docs).
pub struct MiniMaxH3Image {
    descriptor: ModelDescriptor,
    root: PathBuf,
    dit_dir: PathBuf,
    text_encoder_dir: PathBuf,
    device: Device,
    contract: MemoryProviderContract,
    spec: LoadSpec,
    /// [`LoadShape::DeferredMaterialization`] when the source supports rung 4 (SW-7095): a
    /// request that selects it keeps the text encoder and DiT blocks on the host and runs them
    /// through a one-block device window.
    load_shape: LoadShape,
    /// The header-only facts admission prices each request's phases from (SW-7095).
    memory_facts: Option<crate::image_memory_strategy::admission::ImageMemoryFacts>,
    /// The components a resident request left on the device for the next one.
    warm: std::sync::Mutex<WarmComponents>,
}

impl MiniMaxH3Image {
    /// Validate the snapshot and record the paths a render reads: `transformer_diffusers/`,
    /// `text_encoder/` and `vae_diffusers/` with shards, plus `tokenizer/`.
    pub fn load(spec: &LoadSpec) -> Result<Self> {
        refuse_precision_override(spec)?;
        let root = match &spec.weights {
            WeightsSource::Dir(p) => p.clone(),
            WeightsSource::File(p) => p
                .parent()
                .map(Path::to_path_buf)
                .unwrap_or_else(|| p.clone()),
        };
        let dit_dir = resolve_dit_dir(&root, spec);
        require_component_shards(&dit_dir, IMAGE_DIT_PARTITION)?;
        require_component_shards(&root.join(IMAGE_VAE_COMPONENT), IMAGE_VAE_COMPONENT)?;
        let text_encoder_dir = resolve_text_encoder_dir(&root, spec);
        require_component_shards(&text_encoder_dir, IMAGE_TEXT_ENCODER)?;
        // The tier is an assertion about what is staged (the marker is authoritative); `None`
        // asserts nothing and the loaders auto-detect from `.scales`. The group size is checked
        // either way, and for the text encoder -- MiniMax-H3's own tier artifact -- by H3's check.
        let tier =
            crate::model::requested_tier(spec).map_err(|e| CandleError::Msg(e.to_string()))?;
        crate::tier::require_partition(&dit_dir, tier, IMAGE_DIT_PARTITION)?;
        crate::tier::MiniMaxH3TierPaths::resolve(&root, None, Some(&text_encoder_dir))
            .require_text_encoder()?;
        if !root.join("tokenizer").join("tokenizer.json").is_file() {
            return Err(CandleError::Msg(format!(
                "{IMAGE_MODEL_ID}: the snapshot has no tokenizer/tokenizer.json at {}",
                root.display()
            )));
        }
        // The partition must be the IMAGE transformer: H3 video's `transformer/` config would
        // load and run with an audio stream this pipeline never feeds.
        let cfg = image_dit_config_at(&dit_dir)?;
        if cfg.num_pdd_heads() != Some(contract::FIXED_STEPS as usize) {
            return Err(CandleError::Msg(format!(
                "{IMAGE_MODEL_ID}: `{IMAGE_DIT_PARTITION}` carries {:?} per-step heads; the fixed \
                 sampler runs {} steps",
                cfg.num_pdd_heads(),
                contract::FIXED_STEPS
            )));
        }
        if spec.load_shape == LoadShape::DeferredMaterialization
            && !image_streaming_supported(spec)?
        {
            return Err(CandleError::Msg(format!(
                "{IMAGE_MODEL_ID}: this source does not support runtime-format block migration; \
                 use the resident load"
            )));
        }
        Ok(Self {
            descriptor: image_descriptor(),
            root,
            dit_dir,
            text_encoder_dir,
            device: candle_gen::default_device()?,
            contract: crate::image_memory_strategy::contract_for(spec)
                .map_err(|e| CandleError::Msg(e.to_string()))?,
            spec: spec.clone(),
            load_shape: spec.load_shape,
            memory_facts: crate::image_memory_strategy::facts_for(spec),
            warm: std::sync::Mutex::new(WarmComponents::default()),
        })
    }

    /// The resolved DiT's config — the staged tier directory when `LoadSpec.components` names
    /// one, which is the directory [`Self::load`] validated and the denoise loads, not the snapshot
    /// root's `transformer_diffusers/`.
    fn dit_config(&self) -> Result<crate::dit::MiniMaxH3DitConfig> {
        image_dit_config_at(&self.dit_dir)
    }

    /// The snapshot root a render reads from.
    pub fn root(&self) -> &Path {
        &self.root
    }

    /// The requested output size: the upstream 2048 x 2048 default for `0 x 0`.
    fn output_size(req: &GenerationRequest) -> (u32, u32) {
        if req.width == 0 && req.height == 0 {
            (DEFAULT_SIZE, DEFAULT_SIZE)
        } else {
            (req.width, req.height)
        }
    }

    /// The `edit` conditioning (SW-7092): H3's Qwen3-VL vision tower over every vision-resized
    /// reference (a 1024 short edge), spliced into the `Picture i: ` + vision-block presentation
    /// and run through the layer-50 conditioner; then every VAE-resized reference (a
    /// `[1024, 2048]` short edge) encoded on its own latent grid, its posterior sampled with
    /// [`reference_noise`] stream 0. Each component is mapped, used, dropped and released in turn.
    /// Returns the context, its per-row tags and the reference latents `[1, 24, h, w]`.
    fn encode_edit(
        &self,
        conditioner: Option<&crate::text_encoder::MiniMaxH3TextEncoder>,
        prompt: &str,
        references: &[&candle_gen::gen_core::Image],
        seed: u64,
        on_progress: &mut dyn FnMut(Progress),
    ) -> Result<ImageConditioning> {
        let vision_images = references
            .iter()
            .map(|r| resize_reference(r, contract::reference_vision_size(r.width, r.height)))
            .collect::<Result<Vec<_>>>()?;
        let phase = PhaseLog::start("vision_tower");
        let vision = crate::text_encoder::load_vision_tower(&self.text_encoder_dir, &self.device)?;
        let refs: Vec<&candle_gen::gen_core::Image> = vision_images.iter().collect();
        let grounded = crate::text_encoder::run_vision(&vision, &refs, &self.device)?;
        drop(vision);
        drop(vision_images);
        crate::dit::release_device_memory(&self.device)?;
        phase.done();

        let phase = PhaseLog::start("text_encoder");
        let tok = MiniMaxH3Tokenizer::from_snapshot(&self.root)?;
        let (ids, mask, tags) = tok.encode_image_edit(prompt, &grounded.counts, &self.device)?;
        let context = match conditioner {
            Some(te) => conditioner_forward(te, &ids, &mask, Some(&grounded), on_progress)?,
            None => run_conditioner(
                &self.text_encoder_dir,
                &ids,
                &mask,
                Some(&grounded),
                &self.device,
                on_progress,
            )?,
        };
        drop((ids, mask, grounded));
        phase.done();
        if context.dims()[1] != tags.len() {
            return Err(CandleError::Msg(format!(
                "{IMAGE_MODEL_ID}: the edit context has {} rows but {} modality tags",
                context.dims()[1],
                tags.len()
            )));
        }

        let phase = PhaseLog::start("image_vae_encode");
        let patch = self.dit_config()?.patch_size;
        let encoder = MiniMaxH3ImageVaeEncoder::load(&self.root, &self.device)?;
        let mut latents = Vec::with_capacity(references.len());
        for (i, r) in references.iter().enumerate() {
            let image = resize_reference(r, contract::reference_vae_size(r.width, r.height))?;
            let posterior = encoder.encode(&reference_vae_pixels(&image, &self.device)?)?;
            let noise = reference_noise(seed, 0, i, posterior.mean().dims(), &self.device)?;
            latents.push(encoder.reference_latent(&posterior, &noise, (patch[1], patch[2]))?);
        }
        drop(encoder);
        crate::dit::release_device_memory(&self.device)?;
        phase.done();
        Ok((context, tags, latents))
    }

    /// The conditioning phase (step 1 of a render): the prompt through the layer-50 conditioner,
    /// or — for an edit — the vision tower, the conditioner and every reference's VAE latent
    /// ([`Self::encode_edit`], posterior noise from `seed`). Returns the context, its per-row tags
    /// and the reference latents (empty for text-to-image). The trainer (SW-7098) encodes its
    /// dataset through this same phase.
    pub(crate) fn encode_conditioning(
        &self,
        prompt: &str,
        references: &[&candle_gen::gen_core::Image],
        seed: u64,
        on_progress: &mut dyn FnMut(Progress),
    ) -> Result<ImageConditioning> {
        self.encode_conditioning_with(None, prompt, references, seed, on_progress)
    }

    /// [`Self::encode_conditioning`] through `conditioner` when one is already loaded (a resident
    /// request's warm encoder), else through a conditioner mapped and released for this call.
    fn encode_conditioning_with(
        &self,
        conditioner: Option<&crate::text_encoder::MiniMaxH3TextEncoder>,
        prompt: &str,
        references: &[&candle_gen::gen_core::Image],
        seed: u64,
        on_progress: &mut dyn FnMut(Progress),
    ) -> Result<ImageConditioning> {
        Ok(if references.is_empty() {
            let phase = PhaseLog::start("text_encoder");
            let tok = MiniMaxH3Tokenizer::from_snapshot(&self.root)?;
            let (ids, mask) =
                tok.encode_prompt_truncated(prompt, contract::MAX_PROMPT_TOKENS, &self.device)?;
            let context = match conditioner {
                Some(te) => conditioner_forward(te, &ids, &mask, None, on_progress)?,
                None => run_conditioner(
                    &self.text_encoder_dir,
                    &ids,
                    &mask,
                    None,
                    &self.device,
                    on_progress,
                )?,
            };
            drop((ids, mask));
            phase.done();
            let n = context.dims()[1];
            (context, vec![contract::TEXT_TAG as u32; n], Vec::new())
        } else {
            self.encode_edit(conditioner, prompt, references, seed, on_progress)?
        })
    }

    /// Cache a text-to-image dataset with one tokenizer and one host-resident conditioner.
    /// No VAE or DiT is loaded until the entire caption phase has released the encoder.
    pub(crate) fn encode_training_captions(
        &self,
        items: &[candle_gen::gen_core::train::TrainingItem],
        cancel: &candle_gen::gen_core::CancelFlag,
        mut on_caption: impl FnMut(usize),
    ) -> Result<Vec<ImageConditioning>> {
        if cancel.is_cancelled() {
            return Err(CandleError::Canceled);
        }
        let phase = PhaseLog::start("text_encoder");
        let tok = MiniMaxH3Tokenizer::from_snapshot(&self.root)?;
        let result = crate::model::with_conditioner(&self.text_encoder_dir, &self.device, |te| {
            let mut contexts = Vec::with_capacity(items.len());
            for (index, item) in items.iter().enumerate() {
                if cancel.is_cancelled() {
                    return Err(CandleError::Canceled);
                }
                let (ids, mask) = tok.encode_prompt_truncated(
                    &item.caption,
                    contract::MAX_PROMPT_TOKENS,
                    &self.device,
                )?;
                let context = __model_proof_output!(te.forward(&ids, &mask))?;
                let n = context.dims()[1];
                contexts.push((context, vec![contract::TEXT_TAG as u32; n], Vec::new()));
                on_caption(index + 1);
            }
            Ok(contexts)
        });
        phase.done();
        result
    }

    /// Map the validated DiT partition onto `device` with its block stack at `dtype` — the one
    /// image DiT load site, shared by the render and the trainer (SW-7098).
    pub(crate) fn load_dit(&self, device: &Device, dtype: DType) -> Result<MiniMaxH3Dit> {
        MiniMaxH3Dit::load_from_dir(&self.dit_dir, device, dtype)
    }

    /// The DiT partition directory this load validated.
    pub(crate) fn dit_dir(&self) -> &Path {
        &self.dit_dir
    }

    /// The compute device.
    pub(crate) fn device(&self) -> &Device {
        &self.device
    }

    /// Whether this request's warm set fits the device now (see the module docs), logging the
    /// figures the decision used. `false` when the sources' headers could not be read.
    fn warm_set_fits_now(&self, req: &GenerationRequest, warm: &mut WarmComponents) -> bool {
        use crate::image_memory_strategy::admission::{
            ladder, warm_estimate, warm_weights, ImageRequestShape,
        };
        let Some(facts) = self.memory_facts.as_ref() else {
            return false;
        };
        let tier = candle_gen::gen_core::MemoryNumericTier {
            precision: self.spec.precision,
            quant: self.spec.quantize,
            component_precision_floors: &[],
        };
        let plan = ImageRequestShape::from_request(req).and_then(|shape| {
            let selection = ladder(&self.contract, tier)
                .into_iter()
                .find(|s| s.strategy == candle_gen::gen_core::MemoryStrategy::Resident)
                .ok_or_else(|| {
                    candle_gen::gen_core::Error::Unsupported(format!(
                        "{IMAGE_MODEL_ID}: the contract offers no resident rung"
                    ))
                })?;
            let plan = warm_estimate(&self.contract, facts, &shape, &selection)?;
            Ok((plan, !shape.references.is_empty()))
        });
        let (plan, has_references) = match plan {
            Ok(plan) => plan,
            Err(error) => {
                tracing::info!(%error, "minimax-h3-image: no warm estimate; staging this request");
                return false;
            }
        };
        let weights = warm_weights(facts, has_references);
        let held = warm.held_bytes(&weights);
        let decision =
            crate::warm::WarmDecision::decide(plan.peak_device_bytes(), held, &self.device);
        warm.priced = decision.keep_warm.then_some(weights);
        decision.log(
            "minimax_h3_image_warm_decision",
            "minimax-h3-image",
            "text encoder, DiT and VAE decoder",
        );
        decision.keep_warm
    }

    fn generate_impl(
        &self,
        req: &GenerationRequest,
        on_progress: &mut dyn FnMut(Progress),
    ) -> Result<GenerationOutput> {
        // Rung 4 (SW-7095): a request whose admitted selection streams the transformer blocks
        // installs the one-block window policy for this request only; the text encoder
        // (`run_conditioner`) and the DiT load below read it.
        let _stream_request =
            crate::block_stream::RequestGuard::enter(req, self.load_shape, &self.device)?;
        Generator::validate(self, req)?;
        // Admission priced the references by count; re-price the admitted selection at their
        // actual sizes before any phase runs (a non-square reference outgrows the count bound).
        if let Some(context) = crate::memory_strategy::active_context() {
            crate::image_memory_strategy::admit_exact_request(
                self.memory_facts.as_ref(),
                &self.contract,
                &context,
                req,
            )?;
        }
        if req.cancel.is_cancelled() {
            return Err(CandleError::Canceled);
        }
        let (width, height) = Self::output_size(req);
        // The same canvas admission sized this request's resources from.
        let (canvas_w, canvas_h) = crate::image_memory_strategy::admission_canvas(width, height)?;
        let (latent_h, latent_w) = latent_grid(canvas_w, canvas_h)?;
        let schedule =
            SigmaSchedule::h3_image_distilled().map_err(|e| CandleError::Msg(e.to_string()))?;
        let seed = resolve_seed(req);
        let sampler =
            contract::request_sampler(req).map_err(|r| CandleError::Msg(r.to_string()))?;
        tracing::info!(
            width,
            height,
            canvas_w,
            canvas_h,
            seed,
            "minimax-h3-image: text-to-image request"
        );

        let references = contract::edit_reference_images(&req.conditioning)
            .map_err(|r| CandleError::Msg(r.to_string()))?;

        // Resident requests whose warm set fits reuse (and leave behind) the warm components; any
        // other request releases them first, so its staged phases have the device to themselves.
        let mut warm = candle_gen::lock_recover(&self.warm);
        let resident = keeps_components_warm(req, self.spec.offload_policy)
            && crate::block_stream::current().is_none()
            && self.warm_set_fits_now(req, &mut warm);
        if !resident && !warm.is_empty() {
            *warm = WarmComponents::default();
            crate::dit::release_device_memory(&self.device)?;
            tracing::info!("minimax-h3-image: released the warm components for a staged request");
        }
        if resident {
            tracing::info!(
                conditioner_warm = warm.conditioner.is_some(),
                dit_warm = warm.dit.is_some(),
                decoder_warm = warm.decoder.is_some(),
                "minimax-h3-image: resident request; reusing warm components"
            );
        }

        // --- 1. conditioning ---------------------------------------------------------------
        if resident && warm.conditioner.is_none() {
            let phase = PhaseLog::start("text_encoder_load");
            warm.conditioner = Some(crate::model::load_conditioner(
                &self.text_encoder_dir,
                &self.device,
            )?);
            phase.done();
        }
        let (context, text_tags, reference_latents) = self.encode_conditioning_with(
            warm.conditioner.as_ref(),
            &req.prompt,
            &references,
            seed,
            on_progress,
        )?;
        if req.cancel.is_cancelled() {
            return Err(CandleError::Canceled);
        }

        // --- 2. denoise --------------------------------------------------------------------
        let has_references = !reference_latents.is_empty();
        let timesteps = schedule.timesteps();
        let warm_dit = warm
            .dit
            .take()
            .filter(|prepared| prepared.serves(timesteps, has_references));
        // A warm DiT prepared for the other reference presence was dropped by the filter above;
        // synchronize so its memory is reusable before the reload.
        crate::dit::release_device_memory(&self.device)?;
        let prepared = match warm_dit {
            Some(prepared) => prepared,
            None => {
                let phase = PhaseLog::start("dit_load");
                // Streamed (rung 4): the stack is built on the host in runtime format and only the
                // root modules (projections, refiner, the six heads) move to the device; each block
                // is copied in for its window. The adapters are installed on the host copy first,
                // so every window carries its block's factors.
                let policy = crate::block_stream::current();
                let load_device = if policy.is_some() {
                    &Device::Cpu
                } else {
                    &self.device
                };
                let mut dit = self.load_dit(load_device, DType::BF16)?;
                if !self.spec.adapters.is_empty() {
                    // Strict, like H3 video: an unmatched target or a file that matched nothing
                    // fails the render here rather than producing an image the LoRA barely
                    // touched. The image entry's absent-alpha fallback is `alpha = rank` (the
                    // diffusers/PEFT convention).
                    crate::adapters::apply_minimax_h3_image_adapters(
                        &mut dit,
                        &self.spec.adapters,
                    )?;
                }
                if let Some(policy) = policy {
                    if policy.cancel.is_cancelled() {
                        return Err(CandleError::Canceled);
                    }
                    dit.prepare_streamed(&self.device, policy.window)?;
                }
                let prepared = PreparedImageDit::prepare(dit, timesteps, has_references)?;
                phase.done();
                prepared
            }
        };
        let patch = prepared.dit().config().patch_size;
        let layout = if reference_latents.is_empty() {
            ImageLayout::text_to_image(context.dims()[1], latent_h, latent_w, patch)?
        } else {
            let grids = reference_latents
                .iter()
                .map(|l| {
                    let d = l.dims();
                    (d[2], d[3])
                })
                .collect::<Vec<_>>();
            ImageLayout::edit(&text_tags, &grids, latent_h, latent_w, patch)?
        };
        tracing::info!(
            seq_len = layout.seq_len(),
            text = layout.num_text_tokens(),
            image = layout.num_image_rows(),
            "minimax-h3-image: packed DiT sequence"
        );
        let mut model = ImageDit::from_prepared(prepared, &layout, &context)?;
        model.set_cancel(&req.cancel);
        drop(context);
        let noise = initial_image_rows(latent_h, latent_w, patch, seed, &self.device)?;
        // An edit's reference rows lead the packed image stream, noised to 0.999 with one draw per
        // reference after the generated image's own.
        let initial = if reference_latents.is_empty() {
            noise
        } else {
            let aug = reference_latents
                .iter()
                .enumerate()
                .map(|(i, l)| reference_noise(seed, 1, i, l.dims(), &self.device))
                .collect::<Result<Vec<_>>>()?;
            let condition = reference_condition_rows(&reference_latents, &aug, patch)?;
            Tensor::cat(&[&condition, &noise], 1)?
        };
        drop(reference_latents);
        let total = schedule.num_evals() as u32;
        let phase = PhaseLog::start("denoise");
        let rows = denoise_image(
            &mut model,
            &schedule,
            &sampler,
            seed,
            &initial,
            layout.num_reference_rows(),
            &self.device,
            &req.cancel,
            &mut |completed| {
                on_progress(Progress::Step {
                    current: completed as u32,
                    total,
                })
            },
        )?;
        phase.done();
        if resident {
            warm.dit = Some(model.into_prepared());
        } else {
            drop(model);
        }
        drop(initial);
        crate::dit::release_device_memory(&self.device)?;

        // --- 3. decode ---------------------------------------------------------------------
        on_progress(Progress::Decoding);
        if req.cancel.is_cancelled() {
            return Err(CandleError::Canceled);
        }
        let phase = PhaseLog::start("image_vae_decode");
        let latents = unpack_image_rows(&rows, &layout)?;
        drop(rows);
        let staged_vae;
        let vae = if resident {
            if warm.decoder.is_none() {
                warm.decoder = Some(MiniMaxH3ImageVae::load(
                    &self.root,
                    &self.device,
                    DType::F32,
                )?);
            }
            warm.decoder
                .as_ref()
                .expect("the warm decoder was just loaded")
        } else {
            staged_vae = MiniMaxH3ImageVae::load(&self.root, &self.device, DType::F32)?;
            &staged_vae
        };
        let decoded = vae.decode(&latents)?;
        drop(latents);
        crate::dit::release_device_memory(&self.device)?;
        let image = decoded_to_image(&decoded, width, height)?;
        phase.done();
        Ok(GenerationOutput::Images(vec![image]))
    }
}

/// Whether the resolved DiT and text encoder sources support byte-preserving host-to-device block
/// migration (rung 4) — header-only, shared by the contract declaration and the load boundary.
pub(crate) fn image_streaming_supported(spec: &LoadSpec) -> Result<bool> {
    let root = match &spec.weights {
        WeightsSource::Dir(p) => p.as_path(),
        WeightsSource::File(p) => p.parent().unwrap_or(p),
    };
    Ok(
        crate::block_stream::component_supported(&resolve_dit_dir(root, spec))?
            && crate::block_stream::component_supported(&resolve_text_encoder_dir(root, spec))?,
    )
}

/// Parse `transformer_diffusers/config.json` of a snapshot root and require it to be the image
/// transformer.
pub fn image_dit_config(root: &Path) -> Result<crate::dit::MiniMaxH3DitConfig> {
    image_dit_config_at(&root.join(IMAGE_DIT_PARTITION))
}

/// [`image_dit_config`] against an already-resolved (possibly staged tier) DiT directory.
pub fn image_dit_config_at(dit_dir: &Path) -> Result<crate::dit::MiniMaxH3DitConfig> {
    let path = dit_dir.join("config.json");
    let text = std::fs::read_to_string(&path).map_err(|e| {
        CandleError::Msg(format!("{IMAGE_MODEL_ID}: reading {}: {e}", path.display()))
    })?;
    crate::dit::MiniMaxH3DitConfig::from_image_diffusers_json(&text)
}

impl Generator for MiniMaxH3Image {
    fn descriptor(&self) -> &ModelDescriptor {
        &self.descriptor
    }

    /// The warm set a resident request left on the device, at the weights that request priced it
    /// at. `None` when nothing is held, or while a render holds the set (no credit is the safe
    /// answer).
    fn retained_weight_bytes(&self) -> Option<u64> {
        let warm = match self.warm.try_lock() {
            Ok(warm) => warm,
            Err(std::sync::TryLockError::Poisoned(poisoned)) => poisoned.into_inner(),
            Err(std::sync::TryLockError::WouldBlock) => return None,
        };
        let held = warm
            .priced
            .as_ref()
            .map_or(0, |weights| warm.held_bytes(weights));
        (held > 0).then_some(held)
    }

    fn validate(&self, req: &GenerationRequest) -> candle_gen::gen_core::Result<()> {
        validate_image_request(&self.descriptor, req)
    }

    fn memory_strategy_contract(&self) -> Option<&MemoryProviderContract> {
        Some(&self.contract)
    }

    fn memory_strategy_safety_check(
        &self,
        context: &candle_gen::gen_core::MemoryRunContext,
    ) -> candle_gen::gen_core::MemorySafetyDecision {
        crate::image_memory_strategy::safety_check_with(
            self.memory_facts.as_ref(),
            &self.spec,
            &self.contract,
            context,
        )
    }

    fn begin_memory_strategy_request(
        &self,
        context: &candle_gen::gen_core::MemoryRunContext,
    ) -> candle_gen::gen_core::Result<Option<Box<dyn candle_gen::gen_core::MemoryRequestScope>>>
    {
        crate::image_memory_strategy::registered_begin_request(
            self.memory_facts.as_ref(),
            &self.spec,
            &self.contract,
            context,
        )
    }

    fn generate(
        &self,
        req: &GenerationRequest,
        on_progress: &mut dyn FnMut(Progress),
    ) -> candle_gen::gen_core::Result<GenerationOutput> {
        let result = self.generate_impl(req, on_progress);
        if matches!(&result, Err(error) if !matches!(error, CandleError::Canceled)) {
            // A failed render (an out-of-memory load among them) leaves no half-built warm set
            // behind for the next request to inherit; a cancelled one keeps what it loaded.
            *candle_gen::lock_recover(&self.warm) = WarmComponents::default();
            if let Err(release) = crate::dit::release_device_memory(&self.device) {
                tracing::warn!(%release, "minimax-h3-image: releasing the warm set after a failed render");
            }
        }
        Ok(result?)
    }
}

/// The request floor plus the H3-Image contract: a non-empty prompt, any positive size (or the
/// `0 x 0` default), exactly six steps, no guidance other than 1, no negative prompt, and 0..=16
/// references of at least 32 px a side with no per-reference strength.
pub fn validate_image_request(
    descriptor: &ModelDescriptor,
    req: &GenerationRequest,
) -> candle_gen::gen_core::Result<()> {
    if req.prompt.trim().is_empty() {
        return Err(candle_gen::gen_core::Error::Msg(format!(
            "{IMAGE_MODEL_ID}: prompt must not be empty"
        )));
    }
    let (width, height) = MiniMaxH3Image::output_size(req);
    contract::validate_request(
        width,
        height,
        None,
        req.steps.unwrap_or(contract::FIXED_STEPS),
        req.guidance,
        req.negative_prompt
            .as_deref()
            .is_some_and(|n| !n.trim().is_empty()),
        contract::edit_reference_images(&req.conditioning)?.len(),
    )?;
    contract::request_sampler(req)?;
    let mut sized = req.clone();
    sized.width = width;
    sized.height = height;
    // `guidance = 1.0` is the contract's "no CFG" value and already passed it; the shared floor
    // refuses ANY guidance on a model without CFG, so the admitted 1.0 is not handed to it.
    sized.guidance = None;
    descriptor
        .capabilities
        .validate_request(IMAGE_MODEL_ID, &sized)
}

/// Registry entry point: refuse every `LoadSpec` slot this provider does not read, then load.
pub fn load_image(spec: &LoadSpec) -> candle_gen::gen_core::Result<Box<dyn Generator>> {
    // Q4 / Q8 are the published MLX-affine tiers; NVFP4 and FP8 are different formats and are
    // refused by name rather than reconciled against a q4/q8 marker.
    crate::model::requested_tier(spec)?;
    candle_gen::gen_core::reject_unknown_components(
        spec,
        &[IMAGE_TEXT_ENCODER, IMAGE_DIT_PARTITION],
        IMAGE_MODEL_ID,
    )?;
    crate::model::reject_unread_slots_for(spec, IMAGE_MODEL_ID)?;
    Ok(Box::new(MiniMaxH3Image::load(spec)?))
}

/// Refuse a `precision` override: this route does not wire one.
///
/// [`Precision::Bf16`] is gen-core's "provider default, no override" sentinel (see
/// [`candle_gen::gen_core::Precision`]); only [`Precision::Fp32`] is an ask. The candle route
/// loads the DiT at the checkpoint's bf16 block store (float32 heads), the conditioner at bf16 and
/// the VAE at f32 whatever the spec says, and its block stream, q8/q4 tiers and memory contract are
/// all sized for that. Honoring the ask would mean an f32 DiT through every one of those paths,
/// so, as gen-core's contract prescribes for an unwired override, it is refused with a typed
/// [`candle_gen::gen_core::Error::Unsupported`] rather than rendered (and priced) as bf16. The MLX
/// twin does wire it: its loaders take the requested dtype.
pub(crate) fn refuse_precision_override(spec: &LoadSpec) -> candle_gen::gen_core::Result<()> {
    if spec.precision == Precision::Bf16 {
        return Ok(());
    }
    Err(candle_gen::gen_core::Error::Unsupported(format!(
        "{IMAGE_MODEL_ID}: the candle route does not wire a precision override; it loads the \
         checkpoint's bf16 DiT block store (float32 heads) and an f32 VAE. Got {:?}; drop the \
         override (or use the MLX backend, which honors Fp32)",
        spec.precision
    )))
}

candle_gen::register_generators! {
    pub const IMAGE_REGISTRATION = image_descriptor => load_image
}

#[cfg(test)]
mod tests {
    use super::*;
    use candle_gen::gen_core::Conditioning;

    fn request(width: u32, height: u32) -> GenerationRequest {
        GenerationRequest {
            prompt: "a lighthouse at dusk".into(),
            width,
            height,
            ..Default::default()
        }
    }

    /// A generator loaded with `OffloadPolicy::Sequential` keeps nothing warm for a request with
    /// no request memory: the load asked for each component to be dropped after its phase. A
    /// request's own memory policy still decides when it carries one.
    #[test]
    fn a_sequential_load_keeps_nothing_warm_without_request_memory() {
        use candle_gen::gen_core::GenerationMemory;
        let sequential = OffloadPolicy::Sequential;
        let mut req = request(2048, 2048);
        assert!(
            !keeps_components_warm(&req, sequential),
            "a sequential load stages a request with no request memory"
        );
        req.memory = Some(GenerationMemory::default());
        assert!(
            keeps_components_warm(&req, sequential),
            "an explicit resident request memory overrides the load default"
        );
    }

    /// The descriptor is its own image family — not H3 video's id, modality or flags.
    #[test]
    fn the_descriptor_is_an_image_family_of_its_own() {
        let d = image_descriptor();
        assert_eq!(d.id, IMAGE_MODEL_ID);
        assert_ne!(d.id, crate::MODEL_ID);
        assert_eq!(d.modality, Modality::Image);
        assert_eq!(d.backend, "candle");
        let c = &d.capabilities;
        assert!(!c.supports_negative_prompt && !c.supports_guidance && !c.supports_true_cfg);
        assert_eq!(
            c.conditioning,
            vec![
                ConditioningKind::Reference,
                ConditioningKind::MultiReference
            ]
        );
        assert_eq!(c.supported_steps, StepSupport::Exact(vec![6]));
        assert_eq!(c.max_count, 1);
        // Not H3 video's 1.03 MP / 2016 caps.
        assert!(c.max_size > crate::MAX_CANVAS_EDGE);
        assert_eq!(c.min_size, 1);
        assert!(candle_gen::gen_core::registry::model_descriptor_errors(&d).is_empty());
    }

    /// The H3-Image contract at the engine boundary: 2048² and odd sizes pass, steps other than
    /// six, guidance other than 1, a negative prompt and an empty prompt are refused.
    #[test]
    fn validate_enforces_the_h3_image_contract() {
        let d = image_descriptor();
        validate_image_request(&d, &request(2048, 2048)).unwrap();
        validate_image_request(&d, &request(0, 0)).unwrap();
        validate_image_request(&d, &request(1000, 1001)).unwrap();
        let mut r = request(2048, 2048);
        r.steps = Some(6);
        r.guidance = Some(1.0);
        validate_image_request(&d, &r).unwrap();
        r.steps = Some(8);
        assert!(validate_image_request(&d, &r).is_err());
        let mut r = request(2048, 2048);
        r.guidance = Some(4.0);
        assert!(validate_image_request(&d, &r).is_err());
        let mut r = request(2048, 2048);
        r.negative_prompt = Some("blurry".into());
        assert!(validate_image_request(&d, &r).is_err());
        let mut r = request(2048, 2048);
        r.prompt = "  ".into();
        assert!(validate_image_request(&d, &r).is_err());
        assert!(validate_image_request(&d, &request(0, 2048)).is_err());
        // SW-7106: euler / lcm by name or block; a bad LCM parameter, an unknown name and a
        // name that contradicts the block are refused.
        assert_eq!(d.capabilities.samplers, vec!["euler", "lcm"]);
        let mut r = request(2048, 2048);
        r.sampler = Some("lcm".into());
        validate_image_request(&d, &r).unwrap();
        r.h3_image_sampler = Some(contract::H3ImageSampler::Lcm {
            s_noise: 1.1,
            s_noise_end: 0.9,
            noise_clip_std: 2.0,
        });
        validate_image_request(&d, &r).unwrap();
        r.sampler = Some("euler".into());
        assert!(validate_image_request(&d, &r).is_err());
        r.sampler = None;
        r.h3_image_sampler = Some(contract::H3ImageSampler::lcm(-1.0));
        assert!(validate_image_request(&d, &r).is_err());
        let mut r = request(2048, 2048);
        r.sampler = Some("dpmpp_2m".into());
        assert!(validate_image_request(&d, &r).is_err());
        let mut r = request(2048, 2048);
        r.conditioning.push(Conditioning::Reference {
            image: candle_gen::gen_core::Image {
                width: 1,
                height: 1,
                pixels: vec![0; 3],
            },
            strength: None,
        });
        assert!(validate_image_request(&d, &r).is_err());
    }

    /// A 2048² request generates on a 2048² canvas (128² latent); a 1024² request generates at
    /// 4 MP and is resized back.
    #[test]
    fn the_canvas_comes_from_the_h3_image_contract() {
        let (w, h) = MiniMaxH3Image::output_size(&request(0, 0));
        assert_eq!((w, h), (2048, 2048));
        let canvas = contract::generation_canvas(w, h, contract::MIN_GENERATION_AREA_DEFAULT);
        assert_eq!(latent_grid(canvas.0, canvas.1).unwrap(), (128, 128));
        let canvas = contract::generation_canvas(1024, 1024, contract::MIN_GENERATION_AREA_DEFAULT);
        assert_eq!(canvas, (2048, 2048));
    }

    /// An `Fp32` precision override is refused with a typed `Unsupported` on every entry point
    /// before any path is touched — the registry load, the generator's own loader and the memory
    /// contract — instead of loading and pricing the route's bf16 DiT as though it were honored.
    /// The `Bf16` sentinel passes the precision check (and then fails on the missing snapshot).
    #[test]
    fn an_fp32_precision_override_is_refused_not_ignored() {
        use candle_gen::gen_core::Error as GenError;
        let bf16 = LoadSpec::new(WeightsSource::Dir("/nonexistent-h3-image".into()));
        let mut fp32 = bf16.clone();
        fp32.precision = Precision::Fp32;
        assert!(refuse_precision_override(&bf16).is_ok());
        assert!(matches!(
            refuse_precision_override(&fp32),
            Err(GenError::Unsupported(m)) if m.contains("precision")
        ));
        assert!(matches!(load_image(&fp32), Err(GenError::Unsupported(_))));
        assert!(matches!(
            MiniMaxH3Image::load(&fp32),
            Err(CandleError::Unsupported(m)) if m.contains("precision")
        ));
        assert!(matches!(
            crate::image_memory_strategy::contract_for(&fp32),
            Err(GenError::Unsupported(m)) if m.contains("precision")
        ));
        assert!(!matches!(
            MiniMaxH3Image::load(&bf16),
            Err(CandleError::Unsupported(m)) if m.contains("precision")
        ));
    }

    /// Every slot the provider does not read is refused before any path is touched.
    #[test]
    fn unread_load_slots_are_refused() {
        let spec = LoadSpec::new(WeightsSource::Dir("/nonexistent-h3-image".into()));
        // NVFP4 is a different format from the published MLX-affine tiers.
        let mut q = spec.clone();
        q.quantize = Some(candle_gen::gen_core::Quant::Nvfp4);
        assert!(matches!(
            load_image(&q),
            Err(candle_gen::gen_core::Error::Unsupported(_))
        ));
        // SW-7096: adapters are a read slot now — an adapter spec over a missing snapshot fails
        // at the component check, not as an unsupported slot.
        let mut a = spec.clone();
        a.adapters.push(candle_gen::gen_core::AdapterSpec::new(
            "/x.safetensors".into(),
            1.0,
            candle_gen::gen_core::AdapterKind::Lora,
        ));
        let e = load_image(&a).err().expect("missing snapshot");
        assert!(e.to_string().contains(IMAGE_DIT_PARTITION), "{e}");
        // SW-7095: a deferred load is a read slot too — over a missing snapshot it fails at the
        // component check, not as an unsupported slot.
        let mut d = spec.clone();
        d.load_shape = LoadShape::DeferredMaterialization;
        let e = load_image(&d).err().expect("missing snapshot");
        assert!(e.to_string().contains(IMAGE_DIT_PARTITION), "{e}");
        // A plain spec over a missing snapshot fails at the component check, naming the path.
        let e = load_image(&spec).err().expect("missing snapshot");
        assert!(e.to_string().contains(IMAGE_DIT_PARTITION), "{e}");
    }

    fn with_refs(n: usize, w: u32, h: u32) -> GenerationRequest {
        let mut r = request(1024, 1024);
        r.conditioning.push(Conditioning::MultiReference {
            images: (0..n)
                .map(|_| candle_gen::gen_core::Image {
                    width: w,
                    height: h,
                    pixels: vec![0; (w * h * 3) as usize],
                })
                .collect(),
        });
        r
    }

    /// `edit` admits 1..=16 references of at least 32 px a side and refuses 17 or a 31 px one.
    #[test]
    fn edit_admits_one_to_sixteen_references_of_at_least_32_px() {
        let d = image_descriptor();
        validate_image_request(&d, &with_refs(1, 32, 32)).unwrap();
        validate_image_request(&d, &with_refs(16, 64, 40)).unwrap();
        let e = validate_image_request(&d, &with_refs(17, 64, 64))
            .unwrap_err()
            .to_string();
        assert!(e.contains("17"), "{e}");
        let e = validate_image_request(&d, &with_refs(2, 31, 64))
            .unwrap_err()
            .to_string();
        assert!(e.contains("32"), "{e}");
    }

    /// SW-7093: Q4 and Q8 are advertised, and the staged `transformer_diffusers` component -- not the
    /// snapshot root -- is the DiT a tier load reads.
    #[test]
    fn the_packed_tiers_are_advertised_and_staged_as_the_dit_component() {
        assert_eq!(
            image_descriptor().capabilities.supported_quants,
            &[Quant::Q4, Quant::Q8]
        );
        let mut spec = LoadSpec::new(WeightsSource::Dir("/snap".into()));
        assert_eq!(
            resolve_dit_dir(Path::new("/snap"), &spec),
            Path::new("/snap/transformer_diffusers")
        );
        spec.components.insert(
            IMAGE_DIT_PARTITION.into(),
            WeightsSource::Dir("/tiers/q4/transformer_diffusers".into()),
        );
        assert_eq!(
            resolve_dit_dir(Path::new("/snap"), &spec),
            Path::new("/tiers/q4/transformer_diffusers")
        );
        // The staged key is accepted; any other component is still refused.
        let mut other = spec.clone();
        other
            .components
            .insert("transformer".into(), WeightsSource::Dir("/x".into()));
        assert!(load_image(&other).is_err());
    }

    /// Small real tensor/tokenizer fixture: exercise the production batch and loader on CPU.
    fn caption_fixture() -> (tempfile::TempDir, MiniMaxH3Image) {
        let tmp = tempfile::tempdir().unwrap();
        let root = tmp.path().to_path_buf();
        let te_dir = root.join(IMAGE_TEXT_ENCODER);
        let tok_dir = root.join("tokenizer");
        std::fs::create_dir_all(&te_dir).unwrap();
        std::fs::create_dir_all(&tok_dir).unwrap();
        std::fs::write(
            te_dir.join("config.json"),
            r#"{
            "hidden_size":8,"num_hidden_layers":50,"num_attention_heads":2,
            "num_key_value_heads":1,"head_dim":4,"intermediate_size":12,"vocab_size":16,
            "rope_scaling":{"mrope_section":[1,1,0],"mrope_interleaved":true}
        }"#,
        )
        .unwrap();
        std::fs::write(
            tok_dir.join("tokenizer.json"),
            r#"{
            "version":"1.0","truncation":null,"padding":null,"added_tokens":[],
            "normalizer":null,"pre_tokenizer":{"type":"Whitespace"},
            "post_processor":null,"decoder":null,
            "model":{"type":"WordLevel","vocab":{"[UNK]":0,"red":1,"blue":2},"unk_token":"[UNK]"}
        }"#,
        )
        .unwrap();
        std::fs::write(
            tok_dir.join("tokenizer_config.json"),
            r#"{"additional_special_tokens":["<d>"]}"#,
        )
        .unwrap();
        let mut weights = std::collections::HashMap::new();
        let mut add = |name: String, shape: &[usize]| {
            let values: Vec<f32> = (0..shape.iter().product())
                .map(|i| (i as f32 * 0.37 + 0.1).sin() * 0.02)
                .collect();
            weights.insert(name, Tensor::from_vec(values, shape, &Device::Cpu).unwrap());
        };
        let prefix = crate::text_encoder::LM_PREFIX;
        add(format!("{prefix}.embed_tokens.weight"), &[16, 8]);
        for i in 0..50 {
            let layer = format!("{prefix}.layers.{i}");
            for (name, shape) in [
                ("input_layernorm", vec![8]),
                ("post_attention_layernorm", vec![8]),
                ("self_attn.q_proj", vec![8, 8]),
                ("self_attn.k_proj", vec![4, 8]),
                ("self_attn.v_proj", vec![4, 8]),
                ("self_attn.o_proj", vec![8, 8]),
                ("self_attn.q_norm", vec![4]),
                ("self_attn.k_norm", vec![4]),
                ("mlp.gate_proj", vec![12, 8]),
                ("mlp.up_proj", vec![12, 8]),
                ("mlp.down_proj", vec![8, 12]),
            ] {
                add(format!("{layer}.{name}.weight"), &shape);
            }
        }
        candle_gen::candle_core::safetensors::save(&weights, te_dir.join("model.safetensors"))
            .unwrap();
        let spec = LoadSpec::new(WeightsSource::Dir(root.clone()));
        let model = MiniMaxH3Image {
            descriptor: image_descriptor(),
            root: root.clone(),
            dit_dir: root.join(IMAGE_DIT_PARTITION),
            text_encoder_dir: te_dir,
            device: Device::Cpu,
            contract: crate::image_memory_strategy::weights_free_contract(&spec).unwrap(),
            spec,
            load_shape: LoadShape::DeferredMaterialization,
            warm: Default::default(),
            memory_facts: None,
        };
        (tmp, model)
    }

    /// A provider that holds nothing warm grants no reclaimable credit.
    #[test]
    fn an_empty_warm_set_retains_nothing() {
        let (_tmp, model) = caption_fixture();
        assert_eq!(model.retained_weight_bytes(), None);
    }

    #[test]
    #[ignore = "model inference; explicit model-proof workflow"]
    fn training_captions_reuse_one_encoder_and_match_individual_conditioning() {
        let (_tmp, model) = caption_fixture();
        let items: Vec<_> = ["red", "blue red", "red"]
            .iter()
            .map(|caption| {
                candle_gen::gen_core::train::TrainingItem::captioned(
                    "unused.png".into(),
                    (*caption).into(),
                )
            })
            .collect();
        let mut noop = |_| {};
        let expected: Vec<_> = items
            .iter()
            .map(|item| {
                model
                    .encode_conditioning(&item.caption, &[], 0, &mut noop)
                    .unwrap()
            })
            .collect();
        let cancel = candle_gen::gen_core::CancelFlag::default();
        let mut progress = Vec::new();
        let req = crate::training::preview_request("", 0, (0, 0), &[], cancel.clone());
        let _scope =
            crate::StreamedRequestScope::enter(&req, model.load_shape, &model.device).unwrap();
        let actual = model
            .encode_training_captions(&items, &cancel, |current| {
                progress.push(current);
                if current == 1 {
                    // A second load would now fail. The prepared host stack must serve the rest.
                    std::fs::remove_file(model.text_encoder_dir.join("model.safetensors")).unwrap();
                    std::fs::remove_file(model.root.join("tokenizer/tokenizer.json")).unwrap();
                }
            })
            .unwrap();
        assert_eq!(progress, vec![1, 2, 3]);
        for ((context, tags, refs), (expected, expected_tags, _)) in actual.iter().zip(expected) {
            assert_eq!(
                context.to_vec3::<f32>().unwrap(),
                expected.to_vec3::<f32>().unwrap()
            );
            assert_eq!(*tags, expected_tags);
            assert!(refs.is_empty());
        }

        #[cfg(feature = "model-inference-proof")]
        println!("\nMODEL_INFERENCE_BODY_COMPLETED image_model::tests::training_captions_reuse_one_encoder_and_match_individual_conditioning");
    }

    #[test]
    #[ignore = "model inference; explicit model-proof workflow"]
    fn training_caption_cancellation_stops_before_the_next_item() {
        let (_tmp, model) = caption_fixture();
        let items = vec![
            candle_gen::gen_core::train::TrainingItem::captioned(
                "unused.png".into(),
                "red".into()
            );
            2
        ];
        let cancel = candle_gen::gen_core::CancelFlag::default();
        let mut progress = Vec::new();
        let result = model.encode_training_captions(&items, &cancel, |current| {
            progress.push(current);
            cancel.cancel();
        });
        assert!(matches!(result, Err(CandleError::Canceled)));
        assert_eq!(progress, vec![1]);
        // Pre-cancelled input must not load tokenizer or encoder assets.
        std::fs::remove_file(model.root.join("tokenizer/tokenizer_config.json")).unwrap();
        assert!(matches!(
            model.encode_training_captions(&items, &cancel, |_| {}),
            Err(CandleError::Canceled)
        ));

        #[cfg(feature = "model-inference-proof")]
        println!("\nMODEL_INFERENCE_BODY_COMPLETED image_model::tests::training_caption_cancellation_stops_before_the_next_item");
    }

    /// The edit reads its VAE patch size from the resolved DiT — a staged tier directory — so a
    /// snapshot root that holds only the VAE and tokenizer assets still edits, as it text-to-images.
    #[test]
    fn the_edit_patch_size_comes_from_the_staged_dit() {
        let tmp = tempfile::tempdir().unwrap();
        let root = tmp.path().join("root");
        std::fs::create_dir_all(&root).unwrap();
        let dit_dir = tmp.path().join("q4").join(IMAGE_DIT_PARTITION);
        std::fs::create_dir_all(&dit_dir).unwrap();
        std::fs::write(
            dit_dir.join("config.json"),
            r#"{
                "_class_name": "MiniMaxH3ImageTransformer2DModel",
                "num_attention_heads": 56, "attention_head_dim": 128, "hidden_size": 5376,
                "num_layers": 50, "num_refiner_layers": 2, "ffn_dim": 14336, "in_channels": 24,
                "patch_size": [1, 2, 2], "text_dim": 5120, "freq_dim": 256,
                "time_embed_hidden_dim": 5376, "time_embed_dim": 2688, "rope_freq_dim": 16,
                "rope_theta": 10000.0, "norm_eps": 1e-05, "qk_norm_eps": 1e-05,
                "final_norm_eps": 1e-05, "num_pdd_heads": 6
            }"#,
        )
        .unwrap();
        let spec = LoadSpec::new(WeightsSource::Dir(root.clone()));
        let image = MiniMaxH3Image {
            descriptor: image_descriptor(),
            root: root.clone(),
            dit_dir,
            text_encoder_dir: root.join(IMAGE_TEXT_ENCODER),
            device: Device::Cpu,
            contract: crate::image_memory_strategy::weights_free_contract(&spec).unwrap(),
            spec,
            load_shape: LoadShape::EagerMaterialization,
            memory_facts: None,
            warm: Default::default(),
        };
        assert_eq!(image.dit_config().unwrap().patch_size, [1, 2, 2]);
        // The root has no `transformer_diffusers/`: reading it there is what failed every edit.
        assert!(image_dit_config(&root).is_err());
    }

    /// The on-disk `quantization` marker is authoritative: a q4 DiT staged under a Q8 request is
    /// refused before any tensor is read, and the text encoder resolves to the staged H3 artifact.
    #[test]
    fn a_staged_tier_must_agree_with_the_request() {
        let tmp = tempfile::tempdir().unwrap();
        let comp = |dir: &Path, config: &str| {
            std::fs::create_dir_all(dir).unwrap();
            std::fs::write(dir.join("config.json"), config).unwrap();
            std::fs::write(dir.join("m.safetensors"), b"").unwrap();
        };
        let root = tmp.path().join("root");
        comp(&root.join(IMAGE_VAE_COMPONENT), "{}");
        let dit = tmp.path().join("q4").join(IMAGE_DIT_PARTITION);
        comp(&dit, r#"{"quantization": {"bits": 4, "group_size": 64}}"#);
        let te = tmp.path().join("q4").join(IMAGE_TEXT_ENCODER);
        comp(&te, r#"{"quantization": {"bits": 4, "group_size": 64}}"#);
        let mut spec = LoadSpec::new(WeightsSource::Dir(root));
        spec.components
            .insert(IMAGE_DIT_PARTITION.into(), WeightsSource::Dir(dit));
        spec.components
            .insert(IMAGE_TEXT_ENCODER.into(), WeightsSource::Dir(te.clone()));
        assert_eq!(resolve_text_encoder_dir(Path::new("/unused"), &spec), te);
        spec.quantize = Some(Quant::Q8);
        let e = load_image(&spec).err().expect("q4 staged, q8 asked");
        assert!(e.to_string().contains("bits 4"), "{e}");
    }
}
