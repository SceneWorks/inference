//! `Flux2DevControl` — the FLUX.2-dev **Fun-Controlnet-Union** variant (sc-2292): strict-pose
//! (VACE-style) conditioning via `alibaba-pai/FLUX.2-dev-Fun-Controlnet-Union`, registered as its
//! own `Generator` (`flux2_dev_control`).
//!
//! Mirrors the Z-Image-turbo control port (sc-2257) onto the dev base: the transformer is a
//! [`Flux2ControlTransformer`] (the parity-proven dev DiT + the control branch) and `generate`
//! threads a VAE-encoded control context through it under the embedded-guidance denoise (dev is
//! guidance-distilled — a single forward, no true-CFG). [`load_dev_control`] needs the dev snapshot
//! (`spec.weights`) **and** the control checkpoint (`spec.control`); the base loads manifest-aware
//! (a pre-quantized dev snapshot loads packed, sc-5917) and the bf16 control overlay loads dense,
//! The selected text encoder follows the base transformer's effective tier, including a packed base
//! selected without `spec.quantize`. An explicit `spec.quantize` also packs the dense control branch
//! and VAE in place (the packed base no-ops); without that explicit request those two components stay
//! dense. The control patch embedder always stays dense (its 260 in-features is not a multiple of the
//! quant group size).
//!
//! Architecture (`videox_fun/models/flux2_transformer2d_control.py`): a VACE ControlNet on the first
//! 4 of dev's 8 base double blocks. The control context is the VAE-encoded pose/union skeleton
//! (`control_latents` 128) concatenated with a zero inpaint mask (4) and a zero inpaint latent (128)
//! = 260 channels per image token (the union ControlNet's pose-only layout). See
//! [`crate::transformer::Flux2ControlBranch`] for the hint-injection forward.

use mlx_gen::image::decoded_to_image;
use mlx_gen::tokenizer::TextTokenizer;
use mlx_gen::{
    curated_sampler_names, curated_scheduler_names, default_seed, gen_core, require_base_dir,
    require_control, run_flow_sampler_with_latent_hook, Capabilities, Conditioning,
    ConditioningKind, ControlBranch, Error, GenerationOutput, GenerationRequest, Generator, Image,
    LoadSpec, Modality, ModelDescriptor, OffloadPolicy, Precision, Progress, Quant, Residency,
    Result, SizeFloor, TimestepConvention,
};
use mlx_rs::transforms::eval;
use mlx_rs::Array;

use crate::config::{Flux2Config, FLUX2_DEV_CONTROL_ID};
use crate::model::{crop_to_even, match_latent_spatial_size, validate_request, Flux2TextOwned};
use crate::pipeline::{
    add_noise_by_interpolation, create_noise, fun_control_context_from_latents, init_time_step,
    pack_latents, patchify_latents, prepare_grid_ids, prepare_text_ids, preprocess_ref_image,
    schedule_with,
};
use crate::transformer::Flux2ControlTransformer;
use crate::vae::Flux2Vae;
use crate::{loader, CONTROL_IN_DIM};

/// The control variant's identity + capabilities. The guidance-distilled dev base (embedded
/// guidance, no negative prompt / true-CFG) plus `Control` conditioning (the required pose/union
/// skeleton) and an optional `Reference` (an img2img init image, the fork's `inpaint_image`/`image`
/// init seed). Mac-only, like every FLUX.2 variant.
pub fn descriptor_dev_control() -> ModelDescriptor {
    ModelDescriptor {
        encoder_contract: Some(crate::config::DEV_ENCODER_CONTRACT),
        denoiser_output_latent_space: Some(&mlx_gen::gen_core::FLUX2_PACKED_LATENT_SPACE),
        // Deliberately input-agnostic, not undeclared: the Fun-Controlnet-Union checkpoint runs
        // pose / canny / depth down one VAE-encoded path with no mode index, so every kind is
        // genuinely accepted. Declaring `Some(Any)` says that; leaving it `None` would say "nobody
        // checked".
        control_kinds: Some(mlx_gen::AcceptedControlKinds::Any),
        required_components: &[],
        id: FLUX2_DEV_CONTROL_ID,
        family: "flux2",
        backend: "mlx",
        modality: Modality::Image,
        capabilities: Capabilities {
            supports_negative_prompt: false,
            // dev consumes its guidance scale as an embedded scalar (FLUX.1-dev pattern), not CFG.
            supports_guidance: true,
            supports_true_cfg: false,
            // Control (required, the pose/union skeleton) + an optional img2img Reference init.
            conditioning: vec![ConditioningKind::Control, ConditioningKind::Reference],
            // LoRA/LoKr target the base DiT (the control branch is never an adapter target).
            supports_lora: true,
            supports_lokr: true,
            supported_quants: &[Quant::Q4, Quant::Q8],
            component_precision_floors: &[],
            // Curated unified-framework integrator menu (epic 7114 P3), as the base FLUX.2 path.
            samplers: curated_sampler_names(),
            // Curated scheduler menu (epic 7114), as the base FLUX.2 path — native default + curated.
            schedulers: {
                let mut s = curated_scheduler_names();
                s.push("flow_match_euler");
                s
            },
            supported_guidance_methods: vec![],
            min_size: 256,
            max_size: 2048,
            max_count: 8,
            mac_only: true,
            supports_kv_cache: false,
            requires_sigma_shift: true,
            // Wired onto the shared `Residency` seam (sc-10840); honors Sequential offload — the
            // Mistral-3 text encoder drops after the prompt encode, then the control transformer (dev
            // DiT + control branch) + VAE load, bounding peak to `max(TE, DiT+control+VAE)`.
            supports_sequential_offload: true,
            supports_preview: true,
            supports_streaming: false,
            supports_multi_speaker: false,
            supports_conversation_history: false,
            supports_conversation_session: false,
            max_speakers: None,
            // No audio surface (sc-12834): pure image/video model.
            audio_sample_rates: vec![],
            max_audio_duration_secs: None,
            audio_voices: vec![],
            audio_languages: vec![],
            audio_edit_modes: vec![],
            size_floor: SizeFloor::RangeChecked,
        },
    }
}

/// The heavy render-phase components for the FLUX.2 control variant — the control transformer (the dev
/// DiT plus the Fun-Controlnet-Union control branch, loaded together) and the VAE. No PiD overlay (the
/// FLUX.2 PiD story scoped the control path out — sc-7847). Owned by the `Resident` components or by a
/// `Sequential` generate.
pub(crate) struct Flux2ControlHeavyOwned {
    transformer: Flux2ControlTransformer,
    vae: Flux2Vae,
}

/// A loaded control generator: the dev Mistral-3 text encoder + the control transformer + VAE, held via
/// the shared [`Residency`] seam (sc-10840). `Resident` (default) keeps every component warm;
/// `Sequential` drops the text encoder after the prompt encode, then loads the control transformer
/// (base DiT + control branch) + VAE, bounding peak to `max(TE, DiT+control+VAE)`.
pub struct Flux2DevControl {
    descriptor: ModelDescriptor,
    config: Flux2Config,
    memory_strategy: gen_core::MemoryProviderContract,
    memory_numeric_tier: gen_core::MemoryNumericTier,
    tokenizer: Option<TextTokenizer>,
    residency: Residency<Flux2TextOwned, Flux2ControlHeavyOwned>,
}

/// FLUX.2-dev strict pose (sc-2292): load the dev snapshot + the Fun-Controlnet-Union control
/// checkpoint and assemble the [`Flux2DevControl`] generator, honoring [`LoadSpec::offload_policy`]
/// (sc-10840).
///
/// `spec.weights` must be the dev snapshot directory (tokenizer/ text_encoder/ transformer/ vae/);
/// `spec.control` (required) the Fun-Controlnet-Union checkpoint (a single `.safetensors` `File`, or
/// a `Dir`). The base loads manifest-aware (pre-quantized dev → packed); the bf16 control overlay
/// loads dense. The selected text encoder inherits the effective base transformer tier. An explicit
/// `spec.quantize` (Q4/Q8) additionally packs the dense control branch + VAE — a no-op on an already
/// packed base (the control patch embedder stays dense, since its in-features is not a multiple of
/// 64). A packed base selected without an explicit request leaves the control branch + VAE dense.
pub fn load_dev_control(spec: &LoadSpec) -> Result<Box<dyn Generator>> {
    if spec.precision != Precision::Bf16 {
        return Err(Error::Msg(format!(
            "{FLUX2_DEV_CONTROL_ID}: only the default precision is wired; drop the precision \
             override (Q4/Q8 = spec.quantize)"
        )));
    }
    // Shared load boilerplate (sc-8241): the base must be a snapshot dir, the control checkpoint is
    // required — checked up front (fail fast) so a missing control checkpoint errors before any
    // component loads.
    let root = require_base_dir(
        spec,
        FLUX2_DEV_CONTROL_ID,
        "a FLUX.2-dev snapshot directory",
    )?;
    require_control(
        spec,
        FLUX2_DEV_CONTROL_ID,
        "FLUX.2-dev-Fun-Controlnet-Union",
    )?;
    // F-181: a `Sequential` + `spec.quantize` load over a dense snapshot re-quantizes every generate.
    if let Some(q) = spec.quantize {
        if matches!(spec.offload_policy, OffloadPolicy::Sequential) {
            mlx_gen::residency::warn_sequential_requantize(FLUX2_DEV_CONTROL_ID, q.bits());
        }
    }
    let text_encoder_source = crate::config::DEV_ENCODER_CONTRACT.source_for_load(spec, root)?;
    let tokenizer = loader::load_validated_tokenizer_dev(&text_encoder_source)?;
    let memory_numeric_tier =
        crate::model::effective_dev_memory_numeric_tier(spec, FLUX2_DEV_CONTROL_ID)?;
    Ok(Box::new(Flux2DevControl {
        descriptor: descriptor_dev_control(),
        config: Flux2Config::dev(),
        memory_strategy: crate::memory_strategy::registered_dev_control_contract(spec)?,
        memory_numeric_tier,
        tokenizer: Some(tokenizer),
        residency: build_control_residency_with_source(spec, text_encoder_source)?,
    }))
}

/// The policy→[`Residency`] dispatch for the FLUX.2 control variant (sc-10840). `Resident` eager-loads
/// the text encoder + control heavy bundle now; `Sequential` captures the two per-phase loaders and
/// loads nothing now. The text phase is the dev Mistral-3 encoder only (no caption upsample — the
/// control variant has no vision tower), reusing the shared [`Flux2TextOwned`]; the heavy loader builds
/// the control transformer (base + control branch) + VAE.
#[cfg(test)]
fn build_control_residency(
    spec: &LoadSpec,
) -> Result<Residency<Flux2TextOwned, Flux2ControlHeavyOwned>> {
    let root = require_base_dir(
        spec,
        FLUX2_DEV_CONTROL_ID,
        "a FLUX.2-dev snapshot directory",
    )?;
    let text_encoder_source = crate::config::DEV_ENCODER_CONTRACT.source_for_load(spec, root)?;
    build_control_residency_with_source(spec, text_encoder_source)
}

fn build_control_residency_with_source(
    spec: &LoadSpec,
    text_encoder_source: gen_core::ValidatedEncoderSource,
) -> Result<Residency<Flux2TextOwned, Flux2ControlHeavyOwned>> {
    let root = require_base_dir(
        spec,
        FLUX2_DEV_CONTROL_ID,
        "a FLUX.2-dev snapshot directory",
    )?;
    let effective_quant_bits =
        crate::model::effective_base_quant(spec, root, FLUX2_DEV_CONTROL_ID)?.map(Quant::bits);
    let text_encoder_load_time_quant_bits =
        text_encoder_source.load_time_quant_bits(effective_quant_bits, FLUX2_DEV_CONTROL_ID)?;
    build_control_residency_from_admitted_source(
        spec,
        text_encoder_source,
        text_encoder_load_time_quant_bits,
    )
}

fn build_control_residency_from_admitted_source(
    spec: &LoadSpec,
    text_encoder_source: gen_core::ValidatedEncoderSource,
    text_encoder_load_time_quant_bits: Option<i32>,
) -> Result<Residency<Flux2TextOwned, Flux2ControlHeavyOwned>> {
    let spec_heavy = spec.clone();
    Residency::from_policy(
        spec.offload_policy,
        move || load_control_text(&text_encoder_source, text_encoder_load_time_quant_bits),
        // The control variant has no PiD overlay, so the heavy loader ignores `use_pid`.
        move |_use_pid| load_control_heavy(&spec_heavy),
    )
}

/// Load the dev Mistral-3 text encoder (+ optional Q4/Q8) — the phase-A component dropped first under
/// `Sequential`. No vision tower / projector (the control variant does not caption-upsample), so it
/// wraps the encoder in a text-only [`Flux2TextOwned`].
fn load_control_text(
    text_encoder_source: &gen_core::ValidatedEncoderSource,
    text_encoder_load_time_quant_bits: Option<i32>,
) -> Result<Flux2TextOwned> {
    text_encoder_source.read_unchanged(|source| {
        let mut text_encoder = loader::load_text_encoder_dev_from_source(source)?;
        if let Some(bits) = text_encoder_load_time_quant_bits {
            text_encoder.quantize(bits)?;
        }
        Ok(Flux2TextOwned {
            text_encoder,
            vision_tower: None,
            projector: None,
        })
    })
}

/// Load the control heavy bundle — the control transformer (dev DiT + the Fun-Controlnet-Union control
/// branch, from `spec.control`) and the VAE (+ Q4/Q8 + LoRA/LoKr residuals on the base DiT) — everything
/// but the text encoder. The control branch loads here with the DiT (the heavy phase), not the
/// text-encoder phase. Byte-identical to the pre-seam composition.
fn load_control_heavy(spec: &LoadSpec) -> Result<Flux2ControlHeavyOwned> {
    let root = require_base_dir(
        spec,
        FLUX2_DEV_CONTROL_ID,
        "a FLUX.2-dev snapshot directory",
    )?;
    let control = require_control(
        spec,
        FLUX2_DEV_CONTROL_ID,
        "FLUX.2-dev-Fun-Controlnet-Union",
    )?;
    let mut transformer = loader::load_control_transformer_dev(root, control)?;
    let mut vae = loader::load_vae(root)?;
    if let Some(q) = spec.quantize {
        let bits = q.bits();
        transformer.quantize(bits)?;
        vae.quantize(bits)?;
    }
    // LoRA/LoKr (sc-2646): applied to the base DiT (the control branch is never an adapter target),
    // after quantization, as forward-time residuals. No-op when empty.
    if !spec.adapters.is_empty() {
        crate::adapters::apply_flux2_adapters(transformer.base_mut(), &spec.adapters)?;
    }
    Ok(Flux2ControlHeavyOwned { transformer, vae })
}

impl Flux2DevControl {
    /// Tokenize + encode the prompt into `(prompt_embeds, text_ids)` (the dev Mistral TE path; same
    /// as [`crate::model::Flux2`]'s `encode`). Takes the encoder as an argument so the residency seam's
    /// phase-A closure supplies either the warm-resident or the just-loaded `Sequential` encoder.
    fn encode(
        tokenizer: &TextTokenizer,
        text: &Flux2TextOwned,
        prompt: &str,
    ) -> Result<(Array, Array)> {
        let tok = tokenizer.tokenize(prompt)?;
        let (input_ids, attention_mask) = mlx_gen::tokenizer::to_arrays(&tok);
        let embeds = text
            .text_encoder
            .prompt_embeds(&input_ids, &attention_mask)?;
        let ids = prepare_text_ids(embeds.shape()[1] as usize);
        Ok((embeds, ids))
    }

    /// The optional img2img init image (a single `Reference`) + its strength (the per-reference
    /// strength wins over `req.strength`). More than one `Reference` is an error.
    fn resolve_reference<'a>(
        &self,
        req: &'a GenerationRequest,
    ) -> Result<Option<(&'a Image, Option<f32>)>> {
        let mut reference = None;
        for c in &req.conditioning {
            if let Conditioning::Reference { image, strength } = c {
                if reference.is_some() {
                    return Err(Error::Msg(format!(
                        "{FLUX2_DEV_CONTROL_ID}: a single img2img init reference is supported"
                    )));
                }
                reference = Some((image, strength.or(req.strength)));
            }
        }
        Ok(reference)
    }

    /// img2img init conditioning (same encode chain as [`crate::model::Flux2`]): resize → VAE-encode
    /// → NCHW → crop-to-even → match the target latent grid → 2×2 patchify → BN-normalize → pack.
    /// Returns the **clean** packed init latents `[1, lat_h·lat_w, 128]` (seed-independent).
    fn encode_init_latents(
        vae: &Flux2Vae,
        image: &Image,
        width: u32,
        height: u32,
    ) -> Result<Array> {
        let pre = preprocess_ref_image(image, width, height)?;
        let enc = vae.encode_mean(&pre)?;
        let enc = enc.transpose_axes(&[0, 3, 1, 2])?;
        let enc = crop_to_even(&enc)?;
        let enc = match_latent_spatial_size(&enc, (height / 8) as i32, (width / 8) as i32)?;
        let patchified = patchify_latents(&enc)?;
        let normed = vae.bn_normalize_nchw(&patchified)?;
        pack_latents(&normed)
    }

    /// Build the packed control context `[1, seq, 260]` from the pose/union control image — the
    /// fork's `pipeline_flux2_control.py`: VAE-encode → 2×2 patchify → BN-normalize → pack
    /// (`control_latents`, 128), concatenated with a zero inpaint **mask** (4) and a zero **inpaint
    /// latent** (128). For pure pose (no inpaint image / mask) the fork's mask is `1 − ones = 0` and
    /// the inpaint latent is a zeros tensor, so both are all-zero here. `seq` equals the target
    /// latent sequence (built at the same `width`/`height`), so the control context aligns 1:1 with
    /// the base image tokens.
    fn encode_control_context(
        &self,
        vae: &Flux2Vae,
        image: &Image,
        width: u32,
        height: u32,
    ) -> Result<Array> {
        let pre = preprocess_ref_image(image, width, height)?;
        let enc = vae.encode_mean(&pre)?; // NHWC [1,H/8,W/8,32]
        let enc = enc.transpose_axes(&[0, 3, 1, 2])?; // NCHW
        let enc = crop_to_even(&enc)?;
        let enc = match_latent_spatial_size(&enc, (height / 8) as i32, (width / 8) as i32)?;
        let patchified = patchify_latents(&enc)?; // [1,128,h,w]
        let control_lat = vae.bn_normalize_nchw(&patchified)?;
        // Union pose-only layout: pack the control latent → [1, seq, 128], then concat a zero mask
        // (1 latent channel × 2×2 patch = 4) + zero inpaint latent (= in_channels, 128) on the packed
        // feature axis → 260 = CONTROL_IN_DIM. The pack + channel-fill is `fun_control_context_from_latents`
        // (byte-golden'd against the fork's `pipeline_flux2_control` in `tests/fun_control_parity.rs`).
        let in_ch = self.config.in_channels as i32;
        let num_latent_channels = self.config.num_latent_channels as i32;
        let cc = fun_control_context_from_latents(&control_lat, in_ch, num_latent_channels)?;
        debug_assert_eq!(
            cc.shape()[2],
            CONTROL_IN_DIM,
            "control context must be 260ch"
        );
        Ok(cc)
    }

    fn generate_impl(
        &self,
        req: &GenerationRequest,
        on_progress: &mut dyn FnMut(Progress),
    ) -> Result<GenerationOutput> {
        self.validate(req)?;
        let tokenizer = self
            .tokenizer
            .as_ref()
            .ok_or_else(|| Error::Msg(format!("{FLUX2_DEV_CONTROL_ID}: model is not loaded")))?;
        // F-037: bail before the TE encode + the control-context / img2img VAE encodes (all pre-denoise).
        if req.cancel.is_cancelled() {
            return Err(Error::Canceled);
        }
        let base_seed = req.seed.unwrap_or_else(default_seed);
        let steps = req.steps.unwrap_or(self.config_default_steps()) as usize;
        let guidance = req.guidance.unwrap_or(crate::config::DEFAULT_GUIDANCE_DEV);
        // dev is guidance-distilled: the scale is an embedded scalar (single forward), never true-CFG.
        let embedded_guidance = Some(guidance);

        // `resolve_control` is cheap (reads `req.conditioning`), so resolve it before the residency
        // seam runs; the control-context VAE-encode happens in the heavy phase.
        let (control_image, control_scale) = self.resolve_control(req)?;
        // Optional img2img init (the fork's `image`/`inpaint_image` seed) via a single `Reference`.
        let img2img = self.resolve_reference(req)?;
        let start_step = match &img2img {
            Some((_, strength)) => init_time_step(steps, *strength),
            None => 0,
        };

        let sched = schedule_with(steps, req.width, req.height, req.scheduler.as_deref())?;
        let lat_h = (req.height / 16) as usize;
        let lat_w = (req.width / 16) as usize;
        let latent_ids = prepare_grid_ids(lat_h, lat_w, 0);
        let in_channels = self.config.in_channels as i32;

        // Staged residency lifecycle (sc-10840): under `Sequential` the seam loads the Mistral-3
        // encoder, encodes the prompt, materializes, then DROPS it + `clear_cache()` before the control
        // transformer + VAE load. The control-context + img2img-init VAE-encodes run in the heavy phase
        // (byte-identical — deterministic, TE-independent encodes).
        self.residency.run(
            &req.cancel,
            // The control variant has no PiD overlay; `use_pid` is inert for the heavy loader.
            req.use_pid,
            on_progress,
            |text: &Flux2TextOwned| Self::encode(tokenizer, text, &req.prompt),
            |encoded| {
                let Some((prompt_embeds, _text_ids)) = encoded else {
                    return Ok(());
                };
                eval([prompt_embeds])?;
                Ok(())
            },
            |heavy, (prompt_embeds, text_ids), on_progress| {
                let vae = &heavy.vae;
                // The control context + the clean img2img init latents are constant across steps + the
                // batch (they depend only on the image + dims, not the per-seed noise) — encode once.
                let control_context =
                    self.encode_control_context(vae, control_image, req.width, req.height)?;
                let clean_init = match &img2img {
                    Some((image, _)) if start_step > 0 => Some(Self::encode_init_latents(
                        vae, image, req.width, req.height,
                    )?),
                    _ => None,
                };
                // F-037: force the control-context (and any img2img init) VAE encode so the check
                // observes it, then honor a cancel arriving during that encode before the denoise loop.
                match &clean_init {
                    Some(ci) => eval([&control_context, ci])?,
                    None => eval([&control_context])?,
                }
                if req.cancel.is_cancelled() {
                    return Err(Error::Canceled);
                }

                // Compiled elementwise glue (sc-2963), shared with the base flux2 path. Scoped +
                // restored on drop by the RAII guard (F-007).
                let _compile_glue = crate::transformer::CompileGlueGuard::enable();

                let sampler_name = req.sampler.as_deref();
                let mut images = Vec::with_capacity(req.count as usize);
                for i in 0..req.count {
                    let seed = base_seed.wrapping_add(i as u64);
                    let noise = create_noise(seed, req.width, req.height, self.config.in_channels)?;
                    let latents = match &clean_init {
                        Some(clean) => {
                            add_noise_by_interpolation(clean, &noise, sched.sigmas[start_step])?
                        }
                        None => noise,
                    };
                    // Curated unified-framework solver (epic 7114 P3); the control branch is the
                    // `predict` closure. FLUX.2 feeds `sigma · 1000` as the transformer timestep (Sigma
                    // convention). Cancellation, the per-step `eval`, and progress live in
                    // `run_flow_sampler`.
                    let predict = |latents: &Array, sigma: f32| -> Result<Array> {
                        heavy.transformer.forward(
                            latents,
                            &prompt_embeds,
                            &latent_ids,
                            &text_ids,
                            sigma * 1000.0,
                            embedded_guidance,
                            &control_context,
                            control_scale,
                        )
                    };
                    let denoise_sigmas = &sched.sigmas[start_step..];
                    let previews = mlx_gen::preview::PreviewCounter::new(denoise_sigmas);
                    let final_latents = run_flow_sampler_with_latent_hook(
                        sampler_name,
                        TimestepConvention::Sigma,
                        denoise_sigmas,
                        latents,
                        seed,
                        &req.cancel,
                        on_progress,
                        |latents, sigma| {
                            crate::preview::emit_flux_preview(
                                &req.preview,
                                &previews,
                                denoise_sigmas,
                                sigma,
                                latents,
                                lat_h as i32,
                                lat_w as i32,
                                vae,
                            );
                        },
                        predict,
                    )?;
                    on_progress(Progress::Decoding);
                    // The PiD decode overlay (sc-7847) is wired on the FLUX.2 txt2img/edit path;
                    // `flux2_dev_control` is NOT in that story's model list, so it stays on the native
                    // VAE — mirroring the sc-7846 Z-Image Fun-ControlNet scoping decision.
                    let packed =
                        final_latents.reshape(&[1, lat_h as i32, lat_w as i32, in_channels])?;
                    let decoded = heavy.vae.decode_packed_latents(&packed)?; // NHWC [1,H,W,3]
                    let nchw = decoded.transpose_axes(&[0, 3, 1, 2])?;
                    images.push(decoded_to_image(&nchw)?);
                }
                Ok(GenerationOutput::Images(images))
            },
        )
    }

    fn config_default_steps(&self) -> u32 {
        crate::config::DEFAULT_STEPS_DEV
    }
}

/// The Fun-Controlnet-Union is a *union* ControlNet (pose / canny / depth / … share one VAE-encoded
/// control path), so the input-agnostic default [`mlx_gen::AcceptedControlKinds::Any`] applies and all the
/// control boilerplate (resolve/validate-present + the load helpers above) comes from the shared
/// trait (sc-8241). The default message bodies already match this variant's wording, so no override
/// is needed.
impl ControlBranch for Flux2DevControl {
    fn model_id(&self) -> &'static str {
        FLUX2_DEV_CONTROL_ID
    }
}

impl Generator for Flux2DevControl {
    fn descriptor(&self) -> &ModelDescriptor {
        &self.descriptor
    }

    fn validate(&self, req: &GenerationRequest) -> gen_core::Result<()> {
        // Shared capability floor (size/count/guidance/negative/accepted conditioning + multiple-of-16),
        // then the shared control-present check (sc-8241's `ControlBranch::require_control_present`).
        // `is_edit = false`: the control variant requires a *Control* image, not an edit reference.
        validate_request(&self.descriptor, false, false, req)?;
        self.require_control_present(req)?;
        Ok(())
    }

    fn generate(
        &self,
        req: &GenerationRequest,
        on_progress: &mut dyn FnMut(Progress),
    ) -> gen_core::Result<GenerationOutput> {
        self.generate_impl(req, on_progress).map_err(Into::into)
    }

    fn memory_strategy_contract(&self) -> Option<&gen_core::MemoryProviderContract> {
        Some(&self.memory_strategy)
    }

    fn memory_strategy_safety_check(
        &self,
        context: &gen_core::MemoryRunContext,
    ) -> gen_core::MemorySafetyDecision {
        crate::memory_strategy::dev_control_safety_check(
            &self.memory_strategy,
            context,
            self.memory_numeric_tier,
        )
    }
}

// The registration constant bridges the crate's rich `Result` into backend-neutral
// `gen_core::Result`. The `impl Generator`
// above stays hand-written because `validate` adds a control-conditioning check beyond the shared
// `validate_request`, so it is not the plain delegation `impl_generator!` expresses.
mlx_gen::register_generators! {
    pub(crate) const DEV_CONTROL_REGISTRATION = descriptor_dev_control => load_dev_control;
    footprint = crate::model::dev_control_component_footprint
}

pub(crate) const DEV_CONTROL_MEMORY_REGISTRATION: gen_core::MemoryRegistration =
    gen_core::MemoryRegistration {
        provider_id: FLUX2_DEV_CONTROL_ID,
        contract: crate::memory_strategy::registered_dev_control_contract,
        safety_check: crate::memory_strategy::registered_dev_control_safety_check,
    };

#[cfg(test)]
mod tests {
    use super::*;
    use mlx_gen::gen_core::{
        MemoryBudget, MemoryCacheState, MemoryGeometry, MemoryMode, MemoryNumericTier,
        MemoryRunContext, MemorySafetyDecision, MemorySelection, MemoryStrategy,
    };
    use mlx_gen::WeightsSource;

    #[test]
    fn descriptor_is_flux2_dev_control() {
        let d = descriptor_dev_control();
        assert_eq!(d.id, "flux2_dev_control");
        assert_eq!(d.family, "flux2");
        assert!(d.capabilities.accepts(ConditioningKind::Control));
        assert!(d.capabilities.accepts(ConditioningKind::Reference));
        // dev embedded guidance: guidance on, negative / true-CFG off; no KV cache; mac-only.
        assert!(d.capabilities.supports_guidance);
        assert!(!d.capabilities.supports_negative_prompt);
        assert!(!d.capabilities.supports_true_cfg);
        assert!(!d.capabilities.supports_kv_cache);
        assert!(d.capabilities.mac_only);
    }

    #[test]
    fn load_rejects_missing_control_weights() {
        // Without `spec.control`, load must fail on the missing control weights (proving the control
        // overlay is wired as a hard requirement) — not on the missing snapshot.
        let spec = LoadSpec::new(WeightsSource::Dir("/nonexistent".into()));
        let err = load_dev_control(&spec)
            .err()
            .expect("expected error")
            .to_string();
        assert!(err.contains("Fun-Controlnet-Union"), "got: {err}");
    }

    #[test]
    fn load_rejects_single_file_base() {
        let spec = LoadSpec::new(WeightsSource::File("/tmp/dev.safetensors".into()))
            .with_control(WeightsSource::File("/tmp/control.safetensors".into()));
        let err = load_dev_control(&spec)
            .err()
            .expect("expected error")
            .to_string();
        assert!(err.contains("snapshot directory"), "got: {err}");
    }

    // ── sc-10840: weight-free, default-run proof that the FLUX.2 control dispatch HONORS
    // `offload_policy`. `build_control_residency` uses a validation-complete sparse encoder plus a
    // missing control checkpoint: `Sequential` admits the encoder and defers payload loads;
    // `Resident` immediately enters the unchanged payload bracket without materializing the sparse
    // production-size file. The real-weight A/B remains hosted.
    fn validation_complete_snapshot_spec(
        root: &std::path::Path,
        policy: OffloadPolicy,
    ) -> LoadSpec {
        gen_core_testkit::write_encoder_contract_fixture(
            &root.join("text_encoder"),
            crate::config::DEV_ENCODER_CONTRACT,
        )
        .unwrap();
        LoadSpec::new(WeightsSource::Dir(root.to_path_buf()))
            .with_control(WeightsSource::File(
                "/nonexistent/control.safetensors".into(),
            ))
            .with_offload_policy(policy)
    }

    fn tier_spec(
        root: &std::path::Path,
        packed_bits: Option<i32>,
        requested: Option<Quant>,
    ) -> LoadSpec {
        let mut spec = validation_complete_snapshot_spec(root, OffloadPolicy::Sequential);
        std::fs::create_dir_all(root.join("transformer")).unwrap();
        std::fs::write(
            root.join("transformer/config.json"),
            packed_bits.map_or_else(
                || "{}".to_owned(),
                |bits| format!(r#"{{"quantization":{{"bits":{bits},"group_size":64}}}}"#),
            ),
        )
        .unwrap();
        spec.quantize = requested;
        spec
    }

    fn public_control_context(
        contract: &gen_core::MemoryProviderContract,
        tier: MemoryNumericTier,
    ) -> MemoryRunContext {
        MemoryRunContext {
            optimization_authority: mlx_gen::gen_core::MemoryOptimizationAuthority::Estimated,
            selection: MemorySelection {
                strategy: MemoryStrategy::Resident,
                parameters: Default::default(),
                tier,
            },
            calibration_abi: 0,
            calibration_fingerprint: String::new(),
            load_shape: contract.load_shape,
            mode: MemoryMode::TextToImage,
            has_reference: false,
            use_pid: false,
            has_phases: false,
            geometry: MemoryGeometry {
                width: 768,
                height: 768,
                batch: 1,
                frames: 1,
                reference_count: 0,
            },
            overlay: Some(crate::memory_strategy::DEV_CONTROL_OVERLAY.to_owned()),
            budget: MemoryBudget {
                total_bytes: 96 * 1024 * 1024 * 1024,
                committed_bytes: 0,
                reclaimable_bytes: 0,
                reserved_headroom_bytes: 0,
            },
            predicted_peak_bytes: 1,
            cache_state: MemoryCacheState::Cold,
            evidence_revision: "flux2-dev-control-estimated-fallback".to_owned(),
        }
    }

    #[test]
    fn build_residency_sequential_defers_all_component_loads() {
        let fixture = tempfile::tempdir().unwrap();
        let spec = validation_complete_snapshot_spec(fixture.path(), OffloadPolicy::Sequential);
        let res = build_control_residency(&spec)
            .expect("Sequential must validate the encoder and defer payload loads");
        assert!(
            res.is_sequential(),
            "Sequential policy must build a Sequential (deferred) residency"
        );
    }

    #[test]
    fn build_residency_resident_enters_payload_bracket_after_admission() {
        let fixture = tempfile::tempdir().unwrap();
        let spec = validation_complete_snapshot_spec(fixture.path(), OffloadPolicy::Resident);
        let root = require_base_dir(
            &spec,
            FLUX2_DEV_CONTROL_ID,
            "a FLUX.2-dev snapshot directory",
        )
        .unwrap();
        let text_encoder_source = crate::config::DEV_ENCODER_CONTRACT
            .source_for_load(&spec, root)
            .unwrap();
        let effective_quant_bits =
            crate::model::effective_base_quant(&spec, root, FLUX2_DEV_CONTROL_ID)
                .unwrap()
                .map(Quant::bits);
        let text_encoder_load_time_quant_bits = text_encoder_source
            .load_time_quant_bits(effective_quant_bits, FLUX2_DEV_CONTROL_ID)
            .unwrap();

        // Do not materialize the sparse production-shape fixture. A post-admission shard addition
        // lets the real Resident closure prove eager entry through the unchanged-read bracket.
        std::fs::write(
            root.join("text_encoder/added-after-admission.safetensors"),
            [],
        )
        .unwrap();
        let err = build_control_residency_from_admitted_source(
            &spec,
            text_encoder_source,
            text_encoder_load_time_quant_bits,
        )
        .err()
        .expect("Resident must immediately enter the admitted payload-load bracket");
        assert!(
            err.to_string()
                .contains("shard inventory changed after validation"),
            "expected eager payload-bracket mutation detection: {err}"
        );
    }

    #[test]
    fn loaded_and_registered_control_contracts_use_the_effective_base_tier() {
        let registry = crate::provider_registry().unwrap();
        let registration = registry
            .memory_strategy_registrations()
            .find(|registration| registration.provider_id == FLUX2_DEV_CONTROL_ID)
            .expect("DevControl memory registration");
        assert_eq!(
            registry
                .memory_strategy_registrations()
                .filter(|registration| registration.provider_id == FLUX2_DEV_CONTROL_ID)
                .count(),
            1
        );

        for (quant, bits) in [(Quant::Q4, 4), (Quant::Q8, 8)] {
            for prepacked in [false, true] {
                let fixture = tempfile::tempdir().unwrap();
                let spec = tier_spec(
                    fixture.path(),
                    prepacked.then_some(bits),
                    (!prepacked).then_some(quant),
                );
                let loaded = load_dev_control(&spec)
                    .unwrap_or_else(|error| panic!("Q{bits} prepacked={prepacked}: {error}"));
                let contract = loaded
                    .memory_strategy_contract()
                    .expect("loaded control contract");
                assert_eq!(contract.provider_id, FLUX2_DEV_CONTROL_ID);
                assert_eq!(contract.load_shape, spec.load_shape);
                assert!(contract.calibration.is_none());
                assert_eq!(contract.asset_facts, Default::default());
                assert!(contract.conformance_errors().is_empty());
                assert!(contract.strategies.iter().all(|capability| {
                    capability.strategy == MemoryStrategy::Resident
                        && capability.support == gen_core::MemoryStrategySupport::Implemented
                        || capability.strategy != MemoryStrategy::Resident
                            && capability.support == gen_core::MemoryStrategySupport::Missing
                }));
                let tier = MemoryNumericTier {
                    precision: Precision::Bf16,
                    quant: Some(quant),
                    component_precision_floors: &[],
                };
                let context = public_control_context(contract, tier);
                assert_eq!(
                    loaded.memory_strategy_safety_check(&context),
                    MemorySafetyDecision::Accept,
                    "loaded Q{bits} prepacked={prepacked}"
                );

                let registered_contract = (registration.contract)(&spec).unwrap();
                assert_eq!(registered_contract, *contract);
                assert_eq!(
                    (registration.safety_check)(&spec, &registered_contract, &context),
                    MemorySafetyDecision::Accept,
                    "registered Q{bits} prepacked={prepacked}"
                );
                let mut wrong_tier = context.clone();
                wrong_tier.selection.tier.quant = Some(if quant == Quant::Q4 {
                    Quant::Q8
                } else {
                    Quant::Q4
                });
                for decision in [
                    loaded.memory_strategy_safety_check(&wrong_tier),
                    (registration.safety_check)(&spec, &registered_contract, &wrong_tier),
                ] {
                    let MemorySafetyDecision::Reject { reason } = decision else {
                        panic!("wrong Q{bits} control tier must reject")
                    };
                    assert!(reason.contains("does not match loaded tier"), "{reason}");
                }
                assert!(
                    registry
                        .footprint(FLUX2_DEV_CONTROL_ID, &spec)
                        .unwrap()
                        .is_some(),
                    "the resident-only contract must retain its estimated-fallback footprint"
                );
            }
        }
    }

    #[test]
    fn control_contract_rejects_tier_mismatches_and_non_control_public_contexts() {
        let registry = crate::provider_registry().unwrap();
        let registration = registry
            .memory_strategy_registrations()
            .find(|registration| registration.provider_id == FLUX2_DEV_CONTROL_ID)
            .unwrap();
        for (stored_bits, requested) in [(4, Quant::Q8), (8, Quant::Q4)] {
            let fixture = tempfile::tempdir().unwrap();
            let mismatch = tier_spec(fixture.path(), Some(stored_bits), Some(requested));
            let load_error = load_dev_control(&mismatch)
                .err()
                .expect("control load must reject a requested/stored mismatch")
                .to_string();
            assert!(
                load_error.contains(FLUX2_DEV_CONTROL_ID) && load_error.contains("pre-quantized"),
                "{load_error}"
            );
            let contract = (registration.contract)(&mismatch).unwrap();
            let context = public_control_context(
                &contract,
                MemoryNumericTier {
                    precision: Precision::Bf16,
                    quant: Some(requested),
                    component_precision_floors: &[],
                },
            );
            let MemorySafetyDecision::Reject { reason } =
                (registration.safety_check)(&mismatch, &contract, &context)
            else {
                panic!("registered mismatch must reject")
            };
            assert!(
                reason.contains(FLUX2_DEV_CONTROL_ID) && reason.contains("pre-quantized"),
                "{reason}"
            );
        }

        let fixture = tempfile::tempdir().unwrap();
        let spec = tier_spec(fixture.path(), Some(4), None);
        let loaded = load_dev_control(&spec).unwrap();
        let contract = loaded.memory_strategy_contract().unwrap();
        let tier = MemoryNumericTier {
            precision: Precision::Bf16,
            quant: Some(Quant::Q4),
            component_precision_floors: &[],
        };
        let exact = public_control_context(contract, tier);
        for wrong in [
            {
                let mut context = exact.clone();
                context.overlay = None;
                context
            },
            {
                let mut context = exact.clone();
                context.mode = MemoryMode::ImageToImage;
                context.has_reference = true;
                context.geometry.reference_count = 1;
                context
            },
            {
                let mut context = exact.clone();
                context.use_pid = true;
                context
            },
        ] {
            let MemorySafetyDecision::Reject { reason } =
                loaded.memory_strategy_safety_check(&wrong)
            else {
                panic!("non-control public context must reject: {wrong:?}")
            };
            assert!(reason.contains(FLUX2_DEV_CONTROL_ID), "{reason}");
        }

        let mut missing_control = spec.clone();
        missing_control.control = None;
        let registered_contract = (registration.contract)(&missing_control).unwrap();
        let MemorySafetyDecision::Reject { reason } =
            (registration.safety_check)(&missing_control, &registered_contract, &exact)
        else {
            panic!("registered control safety must reject a missing control artifact")
        };
        assert!(reason.contains("Fun-Controlnet-Union"), "{reason}");
    }
}
