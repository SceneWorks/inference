//! `QwenImage21` — the Qwen-Image 2.1 text-to-image implementation of [`mlx_gen::Generator`],
//! plus its [`descriptor`]/[`load`] entry points and the explicit registration constant.
//!
//! [`load`] assembles the model from a `Qwen/Qwen-Image-2.1` snapshot directory (see
//! [`crate::loader`]) and [`QwenImage21::generate`] runs prompt → image: template + tokenize →
//! Qwen3-VL conditioning (drop the system prefix) → seeded packed noise → resolution-shifted
//! flow-match Euler denoise (true CFG when a negative prompt and a scale above 1 are given) →
//! unpack → denormalise → RGBA decode → RGB composited over white.

use std::path::Path;

use mlx_gen::tokenizer::TextTokenizer;
use mlx_gen::{
    curated_sampler_names, curated_scheduler_names, default_seed, gen_core, resolve_flow_schedule,
    Capabilities, Error, GenerationOutput, GenerationRequest, Generator, LoadSpec, Modality,
    ModelDescriptor, Precision, Progress, Quant, Residency, Result, SizeFloor,
};

use crate::config::{SchedulerConfig, DEFAULT_STEPS, DEFAULT_TRUE_CFG, PRESETS, SIZE_MULTIPLE};
use crate::loader;
use crate::pipeline::{
    create_noise, decode_rgb, denoise, encode_references, joint_layout, text_rows, DenoiseInputs,
    ReferenceConditioning,
};
use crate::reference::{collect_references, prepare_references};
use crate::scheduler;
use crate::text_encoder::{system_prompt_drop_count, QwenImage21TextEncoder};
use crate::transformer::QwenImage21Transformer;
use crate::vae::QwenImage21Vae;

/// Registry id for Qwen-Image 2.1 (the SceneWorks worker's `payload.model`).
pub const MODEL_ID: &str = "qwen_image_2_1";
/// Descriptor family — distinct from the 2512 crate's `qwen-image`.
pub const FAMILY: &str = "qwen-image-2-1";
/// Smallest side upstream's pipeline can render: one vision slot (a 2×2 latent) at 16 px/token.
pub const MIN_SIZE: u32 = 32;

/// Qwen-Image 2.1's identity + capabilities — constructible without loading weights.
pub fn descriptor() -> ModelDescriptor {
    let max_size = PRESETS
        .iter()
        .map(|p| p.width.max(p.height))
        .max()
        .unwrap_or(2048);
    ModelDescriptor {
        // The Qwen3-VL tower is loaded from the snapshot's own `text_encoder/`; substituting one
        // through `LoadSpec::text_encoder` is not advertised for this route.
        encoder_contract: None,
        denoiser_output_latent_space: Some(&gen_core::QWEN_IMAGE_2_1_Z64_LATENT_SPACE),
        control_kinds: None,
        required_components: &[],
        id: MODEL_ID,
        family: FAMILY,
        backend: "mlx",
        modality: Modality::Image,
        capabilities: Capabilities {
            // True CFG: a negative prompt plus `true_cfg`/`guidance` above 1 runs the negative
            // branch (upstream default 1.0 = off).
            supports_negative_prompt: true,
            supports_guidance: true,
            supports_true_cfg: true,
            // Reference conditioning (sc-24110). `Reference` and `MultiReference` reach the same
            // upstream call — one ordered list of one to ten condition images — so both kinds are
            // advertised and flattened in request order, matching the seam the 2512 edit provider
            // already exposes to SceneWorks routing. `Mask` is deliberately **not** advertised:
            // upstream has no mask input (see UPSTREAM.md), and a mask travels as an ordinary
            // extra reference the prompt names.
            conditioning: vec![
                gen_core::ConditioningKind::Reference,
                gen_core::ConditioningKind::MultiReference,
            ],
            supports_lora: false,
            supports_lokr: false,
            samplers: curated_sampler_names(),
            schedulers: curated_scheduler_names(),
            min_size: MIN_SIZE,
            max_size,
            max_count: 8,
            mac_only: true,
            // Both affine tiers are **installable** as pre-quantized snapshots ([`crate::convert`])
            // and are also reachable by quantizing a dense snapshot at load.
            supported_quants: &[Quant::Q4, Quant::Q8],
            // No `component_precision_floors`: a tier is a whole-pipeline contract, so every
            // packable component runs the tier the caller selected (`crate::quant`).
            requires_sigma_shift: true,
            supports_sequential_offload: true,
            // Both sides must be multiples of 32 px (16× VAE × 2×2 vision slot).
            size_floor: SizeFloor::RangeCheckedOnGrid {
                multiple: SIZE_MULTIPLE,
            },
            ..Default::default()
        },
    }
}

/// A loaded Qwen-Image 2.1 generator.
pub struct QwenImage21 {
    descriptor: ModelDescriptor,
    tokenizer: TextTokenizer,
    /// Tokens of the system-role prefix the conditioning drops, derived from the tokenizer.
    drop_count: usize,
    scheduler: SchedulerConfig,
    residency: Residency<QwenImage21TextEncoder, Heavy>,
}

/// The heavy render-phase components — everything but the text encoder.
pub(crate) struct Heavy {
    transformer: QwenImage21Transformer,
    vae: QwenImage21Vae,
}

/// Construct a [`QwenImage21`] from a [`LoadSpec`] whose `weights` is a `Qwen/Qwen-Image-2.1`
/// snapshot directory — dense, or one of the pre-quantized tiers [`crate::convert`] produces.
///
/// `spec.quantize` (Q4/Q8) **selects a tier**. Against a pre-quantized snapshot it is a pure
/// selector: every projection packed-detects off its `{base}.scales` sibling and no quantization
/// pass runs, so a Q4 tier lands at ~4 bits/weight with no dense bf16 transient. Against a dense
/// snapshot it still means "quantize the DiT at load", the historical behaviour. A request that
/// disagrees with the tier on disk is a hard error rather than a silent mis-serve — see
/// [`crate::quant::needs_load_time_quant`].
///
/// `Resident` (default) holds every component warm; `Sequential` loads the text encoder, encodes,
/// drops it, then loads the DiT + VAE — bounding peak memory to `max(text encoder, DiT + VAE)`.
pub fn load(spec: &LoadSpec) -> Result<Box<dyn Generator>> {
    gen_core::reject_unknown_components(spec, &[], MODEL_ID)?;
    if spec.precision != Precision::Bf16 {
        return Err(Error::Msg(
            "qwen_image_2_1: weights load at their on-disk dtype; drop the precision override"
                .into(),
        ));
    }
    if !spec.adapters.is_empty() {
        return Err(Error::Unsupported(
            "qwen_image_2_1: LoRA/LoKr adapters are not wired for Qwen-Image 2.1 yet".into(),
        ));
    }
    if spec.text_encoder.is_some() {
        return Err(Error::Unsupported(
            "qwen_image_2_1: the Qwen3-VL text encoder is loaded from the snapshot's own \
             text_encoder/; LoadSpec::text_encoder substitution is not advertised"
                .into(),
        ));
    }
    if let Some(q) = spec.quantize {
        if !matches!(q, Quant::Q4 | Quant::Q8) {
            return Err(Error::Unsupported(format!(
                "qwen_image_2_1: {q:?} is not an MLX affine tier (Q4/Q8)"
            )));
        }
    }
    let root = loader::snapshot_root(&spec.weights)?;
    // Resolve the request against the tier actually on disk before anything is read, so a
    // mismatched pair fails with the actionable message rather than rendering the wrong tier.
    crate::quant::needs_load_time_quant(root, spec.quantize)?;
    let tokenizer = loader::load_tokenizer(root)?;
    let drop_count = system_prompt_drop_count(&tokenizer)?;
    let scheduler = loader::load_scheduler_config(root)?;
    let residency = build_residency(spec)?;
    Ok(Box::new(QwenImage21 {
        descriptor: descriptor(),
        tokenizer,
        drop_count,
        scheduler,
        residency,
    }))
}

fn build_residency(spec: &LoadSpec) -> Result<Residency<QwenImage21TextEncoder, Heavy>> {
    let text_spec = spec.clone();
    let heavy_spec = spec.clone();
    Residency::from_policy(
        spec.offload_policy,
        move || load_text_encoder(&text_spec),
        move |_use_pid| load_heavy(&heavy_spec),
    )
}

/// The Qwen3 language tower at the tier's declared text-encoder width.
///
/// A packed tier loads packed. A dense snapshot with a Q4/Q8 request is quantized here to
/// [`crate::quant::Tier::text_encoder_bits`] — which is the selected tier, because a tier is a
/// whole-pipeline contract — so a load-time tier and the installable tier of the same name hold the
/// same resident layout rather than two different ones.
///
/// A geometry the single declared group size cannot cover is a **typed refusal**, not a silent
/// dense tower: serving a "q4" load whose text encoder is secretly bf16 is the mixed-tier failure
/// this route does not have.
fn load_text_encoder(spec: &LoadSpec) -> Result<QwenImage21TextEncoder> {
    let root = loader::snapshot_root(&spec.weights)?;
    let mut encoder = loader::load_text_encoder(root)?;
    if crate::quant::needs_load_time_quant(root, spec.quantize)? {
        if let Some(bits) = crate::quant::Tier::from_selected(spec.quantize)
            .and_then(crate::quant::Tier::text_encoder_bits)
        {
            if !encoder.quantize(bits)? {
                return Err(Error::Msg(format!(
                    "qwen_image_2_1: this snapshot's Qwen3 tower is {} wide with a {} SwiGLU \
                     intermediate, and a tier is written at group {}; a load-time Q{bits} tier \
                     would leave the tower dense, which is not the tier that was asked for. Point \
                     at a pre-quantized snapshot, or drop the quantize request.",
                    encoder.hidden_size(),
                    encoder.intermediate_size(),
                    crate::quant::GROUP_SIZE,
                )));
            }
        }
    }
    Ok(encoder)
}

fn load_heavy(spec: &LoadSpec) -> Result<Heavy> {
    let root: &Path = loader::snapshot_root(&spec.weights)?;
    let mut transformer = loader::load_transformer(root)?;
    // A pre-quantized tier is already packed — `quantize` would be a no-op over packed weights, so
    // the guard decides rather than the call silently doing nothing.
    if crate::quant::needs_load_time_quant(root, spec.quantize)? {
        let bits = spec
            .quantize
            .expect("needs_load_time_quant is false without a requested tier")
            .bits();
        transformer.quantize(bits)?;
    }
    let vae = loader::load_vae(root)?;
    Ok(Heavy { transformer, vae })
}

impl Generator for QwenImage21 {
    fn descriptor(&self) -> &ModelDescriptor {
        &self.descriptor
    }

    fn validate(&self, req: &GenerationRequest) -> gen_core::Result<()> {
        validate_request(&self.descriptor.capabilities, req).map_err(Into::into)
    }

    fn generate(
        &self,
        req: &GenerationRequest,
        on_progress: &mut dyn FnMut(Progress),
    ) -> gen_core::Result<GenerationOutput> {
        self.generate_impl(req, on_progress).map_err(Into::into)
    }
}

/// The resolved run parameters of one request.
pub(crate) struct RunParams {
    pub(crate) true_cfg: f32,
    pub(crate) use_negative: bool,
    pub(crate) base_seed: u64,
    pub(crate) sigmas: Vec<f32>,
}

pub(crate) fn resolve_run_params(
    scheduler: &SchedulerConfig,
    req: &GenerationRequest,
) -> Result<RunParams> {
    let steps = req.steps.unwrap_or(DEFAULT_STEPS) as usize;
    let tokens = scheduler::image_tokens(req.width, req.height);
    let native = scheduler::sigmas(scheduler, steps, tokens)?;
    let mu = scheduler::mu_for_tokens(scheduler, tokens) as f32;
    let sigmas = resolve_flow_schedule(req.scheduler.as_deref(), mu, steps, &native);
    // `true_cfg_scale`: honour a caller-supplied `true_cfg`, then `guidance`, then the upstream
    // default (off). Guidance engages only with a negative prompt, as upstream's `do_true_cfg`.
    let true_cfg = req.true_cfg.or(req.guidance).unwrap_or(DEFAULT_TRUE_CFG);
    let has_negative = req
        .negative_prompt
        .as_deref()
        .is_some_and(|n| !n.trim().is_empty());
    Ok(RunParams {
        true_cfg,
        use_negative: true_cfg > 1.0 && has_negative,
        base_seed: req.seed.unwrap_or_else(default_seed),
        sigmas,
    })
}

impl QwenImage21 {
    fn generate_impl(
        &self,
        req: &GenerationRequest,
        on_progress: &mut dyn FnMut(Progress),
    ) -> Result<GenerationOutput> {
        self.validate(req)?;
        let params = resolve_run_params(&self.scheduler, req)?;
        let tiling = crate::pipeline::decode_tiling(req);
        let drop = self.drop_count;
        self.residency.run(
            &req.cancel,
            false,
            on_progress,
            |te: &QwenImage21TextEncoder| {
                // The ordered reference list, host-preprocessed against the snapshot's own
                // Qwen3-VL processor geometry. Empty ⇒ the text-to-image route, unchanged.
                let images = collect_references(req)?;
                let references = if images.is_empty() {
                    Vec::new()
                } else {
                    let vision = te.vision_config().ok_or_else(|| {
                        Error::Unsupported(
                            "qwen_image_2_1: reference conditioning needs the snapshot's Qwen3-VL \
                             vision tower (`text_encoder/config.json` `vision_config` + \
                             `model.visual.*`), which this snapshot does not carry"
                                .into(),
                        )
                    })?;
                    prepare_references(&images, vision)?
                };
                let pos =
                    te.encode_conditioning(&self.tokenizer, &req.prompt, drop, &references)?;
                let neg = if params.use_negative {
                    Some(te.encode_conditioning(
                        &self.tokenizer,
                        req.negative_prompt.as_deref().unwrap_or(""),
                        drop,
                        &references,
                    )?)
                } else {
                    None
                };
                // MLX is lazy: force the conditioning while the encoder is alive, so a Sequential
                // drop cannot leave an unevaluated graph pointing at freed weights.
                match &neg {
                    Some(neg) => mlx_rs::transforms::eval([&pos.hidden, &neg.hidden])?,
                    None => mlx_rs::transforms::eval([&pos.hidden])?,
                }
                Ok((pos, neg, references))
            },
            |_| Ok(()),
            |heavy, (pos, neg, references), on_progress| {
                let channels = heavy.transformer.config().in_channels;
                // The joint layout + the condition latents: both branches share one reference
                // encode, but a different prompt is a different text length, hence two layouts.
                let reference_latents = encode_references(&heavy.vae, &references)?;
                let pos_layout =
                    joint_layout(&pos.image_pad_mask, &references, req.width, req.height)?;
                let neg_layout = neg
                    .as_ref()
                    .map(|neg| {
                        joint_layout(&neg.image_pad_mask, &references, req.width, req.height)
                    })
                    .transpose()?;
                let pos_text = text_rows(&pos.hidden, &pos.image_pad_mask)?;
                let neg_text = neg
                    .as_ref()
                    .map(|neg| text_rows(&neg.hidden, &neg.image_pad_mask))
                    .transpose()?;
                let conditioning = (!references.is_empty()).then(|| ReferenceConditioning {
                    latents: &reference_latents,
                    layout: &pos_layout,
                    negative_layout: neg_layout.as_ref(),
                });
                let mut images = Vec::with_capacity(req.count as usize);
                for i in 0..req.count {
                    let seed = params.base_seed.wrapping_add(i as u64);
                    let latents = create_noise(seed, req.width, req.height, channels)?;
                    let latents = denoise(
                        DenoiseInputs {
                            transformer: &heavy.transformer,
                            sigmas: &params.sigmas,
                            latents,
                            prompt_embeds: &pos_text,
                            negative_embeds: neg_text.as_ref(),
                            true_cfg_scale: params.true_cfg,
                            width: req.width,
                            height: req.height,
                            sampler: req.sampler.as_deref(),
                            seed,
                            cancel: &req.cancel,
                            references: conditioning.as_ref().map(|c| ReferenceConditioning {
                                latents: c.latents,
                                layout: c.layout,
                                negative_layout: c.negative_layout,
                            }),
                        },
                        on_progress,
                    )?;
                    on_progress(Progress::Decoding);
                    if req.cancel.is_cancelled() {
                        return Err(Error::Canceled);
                    }
                    images.push(decode_rgb(
                        &heavy.vae,
                        &latents,
                        req.width,
                        req.height,
                        tiling.as_ref(),
                        Some(&req.cancel),
                    )?);
                }
                Ok(GenerationOutput::Images(images))
            },
        )
    }
}

/// Capability-driven request validation: the shared floor (count, size range + 32-px grid,
/// negative/guidance support, sampler/scheduler membership, finiteness) plus the family's own
/// `steps >= 2` (the terminal-sigma stretch is undefined at one step).
pub(crate) fn validate_request(caps: &Capabilities, req: &GenerationRequest) -> Result<()> {
    // The reference list first, so a caller sees the actionable message (the `Mask` workaround,
    // the one-to-ten window) rather than the shared floor's generic "unsupported conditioning",
    // and so a bad count is refused before any weight is touched.
    collect_references(req)?;
    caps.validate_request(MODEL_ID, req)?;
    if req.prompt.trim().is_empty() && req.negative_prompt.is_none() {
        // Upstream renders an empty prompt as a single space; accept it, but a whitespace-only
        // prompt with nothing else is almost certainly a caller bug, so say so.
        return Err(Error::Msg(
            "qwen_image_2_1: the prompt is empty (upstream would condition on a single space)"
                .into(),
        ));
    }
    if let Some(steps) = req.steps {
        if steps < 2 {
            return Err(Error::Msg(format!(
                "qwen_image_2_1: steps must be >= 2 (got {steps}); the terminal-sigma stretch is undefined at one step"
            )));
        }
    }
    Ok(())
}

pub(crate) fn component_footprint(spec: &LoadSpec) -> gen_core::Result<mlx_gen::PerComponentBytes> {
    mlx_gen::PerComponentBytes::from_spec_subdirs(
        spec,
        &["text_encoder"],
        &["transformer"],
        &["vae"],
    )
}

mlx_gen::register_generators! {
    pub(crate) const REGISTRATION = descriptor => load;
    footprint = component_footprint
}

#[cfg(test)]
mod tests {
    use super::*;
    use mlx_gen::{OffloadPolicy, WeightsSource};

    fn req(width: u32, height: u32) -> GenerationRequest {
        GenerationRequest {
            prompt: "a red fox".into(),
            width,
            height,
            ..Default::default()
        }
    }

    #[test]
    fn descriptor_is_distinct_from_the_2512_route() {
        let d = descriptor();
        assert_eq!(d.id, "qwen_image_2_1");
        assert_eq!(d.family, "qwen-image-2-1");
        assert_eq!(d.modality, Modality::Image);
        assert_eq!(d.capabilities.max_size, 2752);
        assert_eq!(d.capabilities.min_size, 32);
        assert_eq!(
            d.capabilities.size_floor,
            SizeFloor::RangeCheckedOnGrid { multiple: 32 }
        );
        assert!(d.capabilities.supports_true_cfg);
        assert_eq!(
            d.capabilities.conditioning,
            vec![
                gen_core::ConditioningKind::Reference,
                gen_core::ConditioningKind::MultiReference,
            ],
            "the reference/edit route is the same upstream call as text-to-image"
        );
        assert!(
            !d.capabilities.accepts(gen_core::ConditioningKind::Mask),
            "upstream has no mask input; a mask travels as an ordinary extra reference"
        );
        assert_eq!(d.denoiser_output_latent_space.map(|s| s.channels), Some(64));
        assert!(gen_core::registry::model_descriptor_errors(&d).is_empty());
    }

    #[test]
    fn every_preset_validates_and_off_grid_sizes_do_not() {
        let caps = descriptor().capabilities;
        for p in PRESETS {
            assert!(
                validate_request(&caps, &req(p.width, p.height)).is_ok(),
                "{}",
                p.ratio
            );
        }
        let err = validate_request(&caps, &req(1000, 1024))
            .unwrap_err()
            .to_string();
        assert!(err.contains("32"), "{err}");
        assert!(validate_request(&caps, &req(4096, 4096)).is_err());
        assert!(validate_request(&caps, &req(32, 32)).is_ok());
    }

    #[test]
    fn steps_below_two_and_empty_prompts_are_refused() {
        let caps = descriptor().capabilities;
        let mut r = req(2048, 2048);
        r.steps = Some(1);
        let err = validate_request(&caps, &r).unwrap_err().to_string();
        assert!(err.contains("steps must be >= 2"), "{err}");
        r.steps = Some(2);
        assert!(validate_request(&caps, &r).is_ok());
        r.prompt = "   ".into();
        let err = validate_request(&caps, &r).unwrap_err().to_string();
        assert!(err.contains("prompt is empty"), "{err}");
    }

    #[test]
    fn reference_shapes_are_refused_at_validate_before_any_weight_loads() {
        use mlx_gen::gen_core::{Conditioning, Image};

        let caps = descriptor().capabilities;
        let image = |n: u32| Image {
            width: n,
            height: n,
            pixels: vec![0; (n * n * 3) as usize],
        };
        let mut r = req(2048, 2048);
        assert!(
            validate_request(&caps, &r).is_ok(),
            "no conditioning is T2I"
        );

        r.conditioning = vec![Conditioning::MultiReference {
            images: (0..11).map(|_| image(8)).collect(),
        }];
        let err = validate_request(&caps, &r).unwrap_err().to_string();
        assert!(err.contains("at most 10"), "{err}");

        // The `Mask` refusal must be this route's actionable message, not the shared floor's
        // generic "unsupported conditioning" — which is why the reference check runs first.
        r.conditioning = vec![Conditioning::Mask { image: image(8) }];
        let err = validate_request(&caps, &r).unwrap_err().to_string();
        assert!(err.contains("no mask input"), "{err}");
        assert!(err.contains("extra reference"), "{err}");

        r.conditioning = vec![Conditioning::MultiReference {
            images: (0..10).map(|_| image(8)).collect(),
        }];
        assert!(
            validate_request(&caps, &r).is_ok(),
            "ten references is the documented boundary"
        );
    }

    #[test]
    fn run_params_follow_upstream_defaults() {
        let s = SchedulerConfig::production();
        let p = resolve_run_params(&s, &req(2048, 2048)).unwrap();
        assert_eq!(p.sigmas.len(), 41, "40 default steps + the trailing 0");
        assert_eq!(p.true_cfg, 1.0);
        assert!(!p.use_negative);
        let mut r = req(2048, 2048);
        r.negative_prompt = Some("blurry".into());
        r.guidance = Some(3.0);
        let p = resolve_run_params(&s, &r).unwrap();
        assert!(p.use_negative);
        assert_eq!(p.true_cfg, 3.0);
        r.true_cfg = Some(1.0);
        let p = resolve_run_params(&s, &r).unwrap();
        assert!(
            !p.use_negative,
            "true_cfg 1.0 disables guidance even with a negative"
        );
    }

    #[test]
    fn load_rejects_single_file_and_unwired_overlays() {
        let err = load(&LoadSpec::new(WeightsSource::File(
            "/tmp/q.safetensors".into(),
        )))
        .err()
        .expect("a single file is refused")
        .to_string();
        assert!(err.contains("snapshot directory"), "{err}");
        let spec = LoadSpec::new(WeightsSource::Dir("/nonexistent".into()))
            .with_text_encoder(WeightsSource::Dir("/elsewhere".into()));
        let err = load(&spec)
            .err()
            .expect("substitution is refused")
            .to_string();
        assert!(err.contains("text_encoder"), "{err}");
        let spec = LoadSpec::new(WeightsSource::Dir("/nonexistent".into())).with_quant(Quant::Q8);
        let err = load(&spec).err().expect("missing snapshot").to_string();
        assert!(!err.contains("not an MLX affine tier"), "{err}");
        assert!(
            load(
                &LoadSpec::new(WeightsSource::Dir("/nonexistent".into()))
                    .with_offload_policy(OffloadPolicy::Sequential)
            )
            .is_err(),
            "a missing snapshot fails at the tokenizer, for both policies"
        );
    }
}
