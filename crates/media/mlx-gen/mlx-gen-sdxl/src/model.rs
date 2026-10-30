//! `Sdxl` — the Stable Diffusion XL implementation of [`mlx_gen::Generator`], plus its
//! [`descriptor`]/[`load`] entry points and explicit registration under the id `"sdxl"` (the
//! SceneWorks worker's `payload.model`).
//!
//! SDXL is the in-process Apple `mlx-examples/stable_diffusion` path (vendored at
//! `_vendor/mlx_sd/`) brought into Rust — a **U-Net** generator (conv ResBlocks + spatial/cross
//! attention + time/`text_time` micro-conditioning), dual CLIP text encoders, an SDXL VAE, and a
//! discrete Euler-Ancestral sampler with real classifier-free guidance. Parity target = the
//! vendored fp16 reference path (`StableDiffusionXL.generate_latents`), validated stage-by-stage.
//!
//! Slices land incrementally (sc-2400): this module starts as the contract + capability surface;
//! [`load`] assembles components as each slice (tokenizer → text encoders → U-Net → VAE → sampler)
//! is wired and parity-proven.

use mlx_gen::{
    curated_scheduler_names, default_seed, ActivationMemoryAnchor, AlphaSchedule, Capabilities,
    Conditioning, ConditioningKind, DiffusionSampler, DiscreteModelSampling, Error,
    GenerationOutput, GenerationRequest, Generator, Image, LatentDecoder, LoadSpec, Modality,
    ModelDescriptor, OffloadPolicy, Precision, Progress, Quant, Residency, Result, StagedHeavy,
    WeightsSource,
};
use mlx_rs::ops::{add, concatenate_axis, multiply};
use mlx_rs::Dtype;
use std::path::Path;

use gen_core_family::sdxl_ldm::SdxlComponent as SdxlLdmComponent;
use mlx_gen::array::scalar;
use mlx_gen::gen_core::sampling::{vp_capture_plan, VpCapturePlan};
use mlx_gen_pid::{resolve_pid_decoder_at_sigma, PidDecoder, PidEngine};

use crate::config::{DiffusionConfig, UNetConfig};
use crate::inpaint::{preprocess_mask, InpaintBlend};
use crate::ip_adapter::IpImageEncoder;
use crate::loader;
use crate::pipeline::{
    denoise_cfgpp_with_preview, denoise_curated_with_preview, denoise_inpaint_with_preview,
    denoise_ip_with_preview, denoise_multi_control_with_preview, denoise_with_preview,
    encode_conditioning_windows, encode_init_latents_tiled, preprocess_control_image,
    text_time_ids, ControlContext, Denoiser,
};
use crate::sampler::{AncestralEuler, EulerSampler};
use crate::text_encoder::ClipTextEncoder;
use crate::tokenizer::ClipBpeTokenizer;
use crate::unet::{ControlNet, UNet2DConditionModel};
use crate::vae::Autoencoder;

/// Caller-staged tokenizer snapshot used by the registry path for a fused imported checkpoint.
pub const LDM_TOKENIZER_COMPONENT: &str = "ldm_tokenizer";

/// img2img default strength (the vendored `generate_latents_from_image` default).
pub(crate) const DEFAULT_STRENGTH: f32 = 0.8;
/// Masked-inpaint / outpaint default strength — the worker's `SdxlDiffusersAdapter` uses 0.85 for
/// `use_inpaint`/`outpaint` (vs 0.6 for a plain edit). An explicit request strength still wins.
pub(crate) const INPAINT_DEFAULT_STRENGTH: f32 = 0.85;
/// Default `ip_adapter_scale` (sc-3059) when a request doesn't override it (the worker's plus-face
/// default ≈ 0.6). In IP mode the `Reference` strength field carries the IP scale.
const IP_DEFAULT_SCALE: f32 = 0.6;

/// The actual serial-output owner: keep every completed latent through the next denoise, seed
/// each output exactly once with wrapping addition, then return the same Vec to staged decode.
/// Observations borrow only; logical byte sums are not unique-backing or physical certificates.
pub(crate) fn collect_serial_latents(
    count: u32,
    base_seed: u64,
    mut generate: impl FnMut(u64, &mut Vec<mlx_rs::Array>) -> Result<()>,
) -> Result<Vec<mlx_rs::Array>> {
    let mut finals = Vec::with_capacity(count as usize);
    for output in 0..count {
        mlx_gen::ios_quantum::diagnostic_metadata(
            "sdxl",
            "serial-output-start",
            &[
                ("schema", 1),
                ("output", u64::from(output)),
                ("count", u64::from(count)),
                ("completed_latents", finals.len() as u64),
                (
                    "completed_latent_bytes",
                    finals.iter().map(mlx_gen::ios_quantum::bytes).sum(),
                ),
            ],
        );
        let seed = base_seed.wrapping_add(output as u64);
        mlx_rs::random::seed(seed)?;
        // The closure appends BEFORE its sampler/blend locals leave scope, preserving the
        // original inline loop's drop/retention order. No extra tensor return/clone is introduced.
        generate(seed, &mut finals)?;
    }
    Ok(finals)
}

/// Actual image-role predicate, shared with payload-free pricing. Merely configuring IP weights
/// does not activate image-token work when a mask/control/acceleration route owns the request.
pub fn uses_ip_reference(
    has_ip: bool,
    has_reference: bool,
    has_mask: bool,
    control_count: usize,
    acceleration: bool,
) -> bool {
    has_ip && has_reference && !has_mask && control_count == 0 && !acceleration
}
/// Default per-branch `conditioning_scale` for a ControlNet `Conditioning::Control` that leaves
/// `scale = None` (F-085) — the diffusers `controlnet_conditioning_scale` full-strength default. An
/// explicit `Some(x)`, including `Some(0.0)` for an inert branch, overrides it.
const DEFAULT_CONTROLNET_SCALE: f32 = 1.0;

/// Resolve an img2img-style strength, preserving the existing default-and-clamp semantics.
pub(crate) fn resolve_strength(requested: Option<f32>, default: f32) -> f32 {
    requested.unwrap_or(default).clamp(0.0, 1.0)
}

/// Resolve the ancestral schedule window for an img2img or inpaint run.
pub(crate) fn ancestral_strength_schedule(
    steps: usize,
    max_time: f32,
    strength: f32,
) -> (usize, f32) {
    ((steps as f32 * strength) as usize, max_time * strength)
}

/// Select the strength tail of a curated sigma schedule.
fn curated_strength_schedule(full_sigmas: &[f32], steps: usize, strength: f32) -> Vec<f32> {
    crate::sampling::strength_tail(full_sigmas, steps, strength)
}

/// The SDXL compute dtype: the U-Net + both CLIP text encoders run **fp16** (the production
/// reference `StableDiffusionXL(float16=True)`); the VAE loads f32 inside its own loader. Shared by
/// the eager (`Resident`) load and the per-generation (`Sequential`) component loaders so both build
/// byte-identical components.
pub(crate) const DTYPE: Dtype = Dtype::Float16;

/// SDXL-base-1.0 production defaults (the SceneWorks `MlxSdxlAdapter`): 30 inference steps,
/// CFG 7.0, native 1024². Used when a request omits the corresponding field (consumed by the
/// `generate` pipeline slice, sc-2400 S5).
pub(crate) const DEFAULT_STEPS: u32 = 30;
pub(crate) const DEFAULT_GUIDANCE: f32 = 7.0;

/// The few-step acceleration samplers (sc-2769). Selected per request via `req.sampler`; each is
/// paired with its acceleration LoRA at load (`spec.adapters`) by the caller (the SceneWorks
/// variant manifest, epic 2755) — selecting one without its LoRA loaded yields undertrained noise.
pub(crate) const ACCEL_SAMPLERS: [&str; 3] = ["lcm", "lightning", "hyper"];

/// ControlNet can share the CFG-free Lightning denoise loop: it builds a one-row control context,
/// produces residuals for that same row, and [`crate::pipeline::forward_eps`] injects them before
/// the U-Net prediction. The other acceleration profiles have not been validated with ControlNet,
/// so keep that boundary explicit rather than accepting a combination just because the common loop
/// happens to be mechanically capable of running it.
///
/// Lightning is distilled CFG-free. In particular, an omitted guidance value resolves to its normal
/// `1.0` default; an explicitly larger value must be rejected rather than silently running a
/// two-row CFG path against a checkpoint trained for a single conditioned prediction.
pub(crate) fn validate_control_acceleration(sampler: &str, guidance: Option<f32>) -> Result<()> {
    if !ACCEL_SAMPLERS.contains(&sampler) {
        return Ok(());
    }
    if sampler != "lightning" {
        return Err(Error::Msg(format!(
            "sdxl: ControlNet is supported with the CFG-free `lightning` acceleration sampler only \
             (got {sampler:?})"
        )));
    }
    if guidance.is_some_and(|value| value > 1.0) {
        return Err(Error::Msg(
            "sdxl: ControlNet with the CFG-free `lightning` sampler requires guidance <= 1.0"
                .into(),
        ));
    }
    Ok(())
}

/// Registry id — matches the SceneWorks worker's `payload.model` (`MODEL_TARGETS["sdxl"]`).
pub const MODEL_ID: &str = "sdxl";

/// PiD latent-space backbone tag (epic 7840, sc-7848): the `sdxl` student in
/// [`mlx_gen_pid::registry`] (SDXL's 4-ch, `0.13025`-affine VAE latent). The whole SDXL family
/// shares this latent space, so `mlx-gen-kolors` (and the RealVisXL variants,
/// which register under this same `"sdxl"` generator) reuse this tag rather than redeclaring it.
pub const PID_BACKBONE: &str = "sdxl";

/// SDXL's identity + capabilities — constructible without loading weights (registry
/// introspection). Capability flags are turned on as each slice lands and is parity-proven, so the
/// descriptor never advertises a path that isn't wired (avoids the false-capability trap —
/// [[false-green-gates-mask-descope]]).
pub fn descriptor() -> ModelDescriptor {
    ModelDescriptor {
        encoder_contract: None,
        denoiser_output_latent_space: Some(&mlx_gen::gen_core::SDXL_LATENT_SPACE),
        control_kinds: None,
        required_components: &[],
        id: MODEL_ID,
        family: "sdxl",
        backend: "mlx",
        modality: Modality::Image,
        capabilities: Capabilities {
            // SDXL uses real classifier-free guidance: honors the negative prompt + a CFG scale.
            supports_negative_prompt: true,
            supports_guidance: true,
            // img2img Reference (sc-2638) + masked inpaint/outpaint (Mask, sc-3057) + tile-ControlNet
            // detail (Control, sc-3058 — requires a control checkpoint via LoadSpec::control). LoRA
            // (kohya `lora_unet_` + PEFT, sc-2639) and LoKr (sc-2640 — Rust is more capable than the
            // vendored path, which rejects LoKr) are wired.
            conditioning: vec![
                ConditioningKind::Reference,
                ConditioningKind::Mask,
                ConditioningKind::Control,
            ],
            supports_lora: true,
            supports_lokr: true,
            // `euler_ancestral` is the production default (full-CFG, 30-step, the bespoke vendored
            // ancestral loop); `lcm`/`lightning`/`hyper` are the few-step acceleration samplers
            // (sc-2769), each driven by its diffusers-faithful schedule and paired with an acceleration
            // LoRA at load. The remaining names (`euler`/`heun`/`dpmpp_2m`/`dpmpp_sde`/`uni_pc`/`ddim`)
            // are the unified curated solvers (epic 7114, sc-7121) — the additive k-diffusion path over
            // `DiscreteModelSampling`; selecting one (or a non-`discrete` scheduler) routes to
            // `denoise_curated` while the default stays byte-exact. A request naming any other sampler
            // is rejected in `validate_request` rather than silently downgraded.
            samplers: vec![
                "euler_ancestral",
                "lcm",
                "lightning",
                "hyper",
                "euler",
                "heun",
                "dpmpp_2m",
                "dpmpp_sde",
                "uni_pc",
                "ddim",
            ],
            // `discrete` is the native ancestral schedule; the rest are the curated σ schedulers
            // (epic 7114 scheduler axis) usable with any curated sampler.
            schedulers: {
                let mut s = vec!["discrete"];
                s.extend(curated_scheduler_names());
                s
            },
            // Plain CFG (the shared `gen_core::guidance::cfg` over `MlxLatentOps`, epic 7434 P3 sc-7443;
            // byte-identical to the retired hand form, shared by Kolors/InstantID/PuLID via
            // `denoise_core`), plus CFG++ (`cfg_pp`, sc-8256) on the curated path — gated at dispatch to a
            // CFG++-compatible base solver (euler/ddim/dpmpp_2m) + an active guidance gap.
            supported_guidance_methods: vec!["cfg", "cfg_pp"],
            min_size: 512,
            max_size: 2048,
            max_count: 8,
            mac_only: false,
            // On-the-fly Q4/Q8 over the U-Net + CLIP encoders + IdentityNet, conv_shortcut kept
            // dense (sc-2769 / sc-3329). Read by the worker capability advertisement (sc-3723).
            supported_quants: &[Quant::Q4, Quant::Q8],
            // Wired onto the shared `Residency` seam (epic 10834); honors Sequential offload (F-176).
            supports_sequential_offload: true,
            supports_preview: true,
            // sc-26268: both CFG batchings are implemented on every denoise route (ancestral,
            // curated, CFG++), through `pipeline::guided_eps_pair`.
            execution: mlx_gen::gen_core::ExecutionSurface {
                cfg_batching: mlx_gen::gen_core::CfgBatchingDomain::Modes(
                    CFG_BATCHING_MODES.to_vec(),
                ),
                ..Default::default()
            },
            ..Default::default()
        },
    }
}

/// The CFG batchings this provider implements (sc-26268): the historical doubled batch and the
/// two-forward sequential mode a constrained device selects.
pub const CFG_BATCHING_MODES: [mlx_gen::gen_core::CfgBatching; 2] = [
    mlx_gen::gen_core::CfgBatching::Batched,
    mlx_gen::gen_core::CfgBatching::Sequential,
];

/// The request's CFG batching. `None` ⇒ [`CfgBatching::Batched`](mlx_gen::gen_core::CfgBatching),
/// exactly what every SDXL denoise ran before sc-26268. The shared request floor
/// (`Capabilities::validate_request`) has already refused a mode outside [`CFG_BATCHING_MODES`].
pub(crate) fn resolve_cfg_batching(req: &GenerationRequest) -> mlx_gen::gen_core::CfgBatching {
    req.memory
        .and_then(|memory| memory.cfg_batching)
        .unwrap_or(mlx_gen::gen_core::CfgBatching::Batched)
}

/// A loaded SDXL generator: the dual CLIP encoders + tokenizer, the U-Net, the VAE, the
/// Euler-Ancestral sampler (production default), and the `alphas_cumprod` schedule the few-step
/// acceleration samplers (LCM/Lightning/Hyper) build on — assembled from a snapshot directory.
pub struct Sdxl {
    descriptor: ModelDescriptor,
    tokenizer: ClipBpeTokenizer,
    sampler: EulerSampler,
    /// DDPM `alphas_cumprod` from the SDXL `scaled_linear` betas — shared by the acceleration
    /// samplers (sc-2769). Built once at load (the ancestral `sampler` keeps its own σ table).
    alpha_schedule: AlphaSchedule,
    /// Number of loaded ControlNet branches (`spec.control` + `spec.extra_controls`), computed at
    /// load so `generate` can reject a `Control`-count mismatch under either residency (the
    /// `Sequential` seam holds loader closures, not the spec, so this is captured up front).
    control_count: usize,
    /// Component-residency strategy (epic 10834 Phase 1, sc-10839; hoisted to the shared seam in
    /// sc-11125), selected from [`LoadSpec::offload_policy`] at [`load`]. `Resident` (default) holds
    /// every heavy component (both CLIP encoders + U-Net + control/IP/VAE/PiD) warm for the whole job
    /// and across jobs; `Sequential` holds only the per-phase loader closures and re-loads each
    /// component in phase order per generation (text encode → **drop the encoders** → U-Net/VAE
    /// denoise+decode), bounding peak unified memory to the largest single working set instead of the
    /// sum, at the cost of the warm cache — for Macs where the resident set would OOM. The
    /// [`Residency`] seam owns the eval/drop/clear discipline, the stage-boundary cancel checks, and
    /// the error-safe cache flush once for all providers.
    residency: Residency<(ClipTextEncoder, ClipTextEncoder), SdxlHeavyOwned>,
    /// The load-time `OffloadPolicy` verdict, kept as the DEFAULT for a request that names no
    /// [`GenerationMemory::stage_residency`](mlx_gen::gen_core::GenerationMemory). Rung 1 is
    /// request-scoped from SC-15525 onward; the policy is no longer the authority.
    default_stage_residency: bool,
    /// Whether THIS load can execute ladder rung 4 — see
    /// [`memory_strategy::streamable`](crate::memory_strategy::streamable).
    streamable: bool,
    loaded_spec: LoadSpec,
    memory_strategy: mlx_gen::gen_core::MemoryProviderContract,
}

/// One correctness-only production-latent decode comparison.
///
/// This test seam deliberately carries no duration, allocator, peak-memory, or calibration field.
/// The latent is captured from the normal [`Sdxl::generate_impl`] denoise path immediately before
/// its production decode, then decoded once densely and once with the caller's candidate tiling.
#[doc(hidden)]
pub struct DecodeQualitySample {
    pub production_latent: mlx_rs::Array,
    pub dense: Image,
    pub tiled: Image,
}

type FinalLatentObserver<'a> =
    &'a mut dyn FnMut(&Autoencoder, &mlx_rs::Array, Option<&dyn LatentDecoder>) -> Result<()>;

/// The heavy render-phase components (everything but the text encoders): the U-Net, its ControlNet
/// branches / IP-Adapter, the VAE, and the optional PiD decoder. Owned by the `Resident` components
/// (held for the whole job) or by a `Sequential` generate (loaded after the encoders are dropped,
/// freed when the job ends).
pub(crate) struct SdxlHeavyOwned {
    unet: UNet2DConditionModel,
    /// ControlNet branches (sc-3058; MultiControlNet sc-3378), loaded from `LoadSpec::control` +
    /// `LoadSpec::extra_controls`. Empty when no control checkpoint was supplied. `generate` requires
    /// exactly one `Control` conditioning per loaded branch (paired by order); their residuals are
    /// summed (the diffusers `MultiControlNetModel` rule).
    controls: Vec<ControlNet>,
    /// Optional IP-Adapter image-token source (sc-3059), loaded from `LoadSpec::ip_adapter`. When
    /// present, the model is in "IP mode": a `Reference` conditioning is the image prompt (txt2img +
    /// IP), not an img2img init. The decoupled-attn K/V projections are installed into `unet`.
    ///
    /// Behind a lock so a STAGED request can shed it once the image prompt is encoded (epic 16774
    /// lever: [`Self::ip_tokens`]): the ViT-H tower and the Resampler are not read again by the
    /// denoise — only the installed K/V pairs are — so the denoise runs without their bytes.
    ip_adapter: std::sync::Mutex<Option<IpImageEncoder>>,
    vae: Autoencoder,
    /// Optional PiD super-resolving decoder overlay (epic 7840, sc-7848): loaded when the spec
    /// carries [`LoadSpec::pid`]. `Some` ⇒ a `req.use_pid` generation decodes the final SDXL latent
    /// through the `sdxl` PiD student (4× SR) instead of the VAE. `None` ⇒ the default byte-exact
    /// VAE decode.
    pid: Option<PidEngine>,
}

/// A borrow of the heavy render-phase components, so the denoise/decode body is written once and
/// runs identically whether the components are held resident (borrowed out of [`ResidentComponents`])
/// or were just loaded by the `Sequential` path — mirrors candle's `DitRef` (sc-10769). Cheap refs.
struct SdxlHeavy<'a> {
    unet: &'a UNet2DConditionModel,
    controls: &'a [ControlNet],
    vae: &'a Autoencoder,
    pid: Option<&'a PidEngine>,
}

impl SdxlHeavyOwned {
    fn as_ref(&self) -> SdxlHeavy<'_> {
        SdxlHeavy {
            unet: &self.unet,
            controls: &self.controls,
            vae: &self.vae,
            pid: self.pid.as_ref(),
        }
    }

    /// A heavy bundle of `unet`, `vae` and an optional IP image encoder (epic 16774 tests).
    #[cfg(test)]
    pub(crate) fn for_test(
        unet: UNet2DConditionModel,
        vae: Autoencoder,
        ip: Option<IpImageEncoder>,
    ) -> Self {
        Self {
            unet,
            controls: Vec::new(),
            ip_adapter: std::sync::Mutex::new(ip),
            vae,
            pid: None,
        }
    }

    /// A heavy bundle of every component a mode loads (epic 16774 additivity proofs).
    #[cfg(test)]
    pub(crate) fn with_controls_for_test(
        unet: UNet2DConditionModel,
        controls: Vec<ControlNet>,
        vae: Autoencoder,
        ip: Option<IpImageEncoder>,
    ) -> Self {
        Self {
            controls,
            ..Self::for_test(unet, vae, ip)
        }
    }

    /// `(U-Net, ControlNet branches, VAE)` of the bundle (epic 16774 additivity proofs).
    #[cfg(test)]
    pub(crate) fn parts_for_test(&self) -> (&UNet2DConditionModel, &[ControlNet], &Autoencoder) {
        (&self.unet, &self.controls, &self.vae)
    }

    #[cfg(test)]
    pub(crate) fn has_ip_adapter_for_test(&self) -> bool {
        self.has_ip_adapter().unwrap()
    }

    #[cfg(test)]
    pub(crate) fn ip_tokens_for_test(
        &self,
        image: &Image,
        cfg_on: bool,
        shed: bool,
    ) -> Result<mlx_rs::Array> {
        self.ip_tokens(image, cfg_on, shed)
    }

    /// Whether the IP-Adapter image encoder is loaded (and not yet shed).
    fn has_ip_adapter(&self) -> Result<bool> {
        Ok(self.ip_lock()?.is_some())
    }

    fn ip_lock(&self) -> Result<std::sync::MutexGuard<'_, Option<IpImageEncoder>>> {
        self.ip_adapter
            .lock()
            .map_err(|_| Error::Msg("sdxl: the IP-Adapter encoder lock is poisoned".into()))
    }

    /// The request's IP image-prompt tokens — CFG-batched with a zeros uncond row when `cfg_on` —
    /// and, when `shed` ([`sheds_ip_image_encoder`]: a staged request), the release of the image
    /// encoder and Resampler the moment the tokens are materialized: nothing after reads them, so
    /// the denoise runs without their bytes (epic 16774 lever; the K/V pairs stay installed in the
    /// U-Net). The tokens are the same values either way.
    fn ip_tokens(&self, image: &Image, cfg_on: bool, shed: bool) -> Result<mlx_rs::Array> {
        let mut slot = self.ip_lock()?;
        let enc = slot.as_ref().ok_or_else(|| {
            Error::Msg("sdxl: the IP-Adapter image encoder was already released".into())
        })?;
        let tokens = enc.tokens(image)?;
        let tokens = if cfg_on {
            // Its own conditioning quantum (epic 16774), so the pair is materialized — and, staged,
            // the encoder released below — before the denoise's first quantum: no allocation runs
            // beside the image encoder outside the conditioning phase.
            mlx_gen::ios_quantum::run(
                crate::quantum::IP_CFG_TOKENS,
                crate::quantum::ip_cfg_tokens_bytes(&tokens),
                || crate::ip_adapter::cfg_pair(&tokens),
            )?
        } else {
            tokens
        };
        if shed {
            // Materialize before the release: a lazy token graph still references the encoder's
            // weights, and dropping it first would free nothing.
            tokens.eval()?;
            *slot = None;
            drop(slot);
            mlx_gen::residency::drain_allocator_cache();
        }
        crate::quantum::diagnostic_owned_tensors(
            "ip-persistent-token-owner",
            0,
            std::iter::once(&tokens),
        );
        mlx_gen::sampler::diagnostic_sampling_tensor(
            "sdxl",
            "ip-persistent-token-layout",
            0,
            tokens.shape()[0] as u64,
            &tokens,
        );
        mlx_gen::ios_quantum::diagnostic_metadata(
            "sdxl",
            "ip-encoder-release-boundary",
            &[("schema", 1), ("shed", u64::from(shed))],
        );
        Ok(tokens)
    }
}

/// The decode-only half of [`SdxlHeavyOwned`] that survives a **staged** request's denoise
/// (sc-26268): the VAE and the optional PiD overlay. [`StagedHeavy::shed_dit`] drops the U-Net, the
/// ControlNet branches and the IP-Adapter encoder before decode starts, so the decode's
/// full-resolution working set no longer stacks on top of a resident U-Net — the dominant term of an
/// SDXL request's high-water, because the decode is where that high-water is reached.
pub(crate) struct SdxlLight {
    vae: Autoencoder,
    #[allow(dead_code)] // carried for symmetry: the PiD decoder is minted during denoise
    pid: Option<PidEngine>,
}

/// A borrowed decode view — from the still-warm [`SdxlHeavyOwned`] on a resident request, or from
/// the post-shed [`SdxlLight`] on a staged one — so the decode body is written once.
pub(crate) struct SdxlDecodeView<'a> {
    vae: &'a Autoencoder,
}

#[cfg(test)]
impl SdxlDecodeView<'_> {
    pub(crate) fn vae_for_test(&self) -> &Autoencoder {
        self.vae
    }
}

impl StagedHeavy for SdxlHeavyOwned {
    type Light = SdxlLight;
    type DecodeView<'a> = SdxlDecodeView<'a>;

    fn shed_dit(self) -> SdxlLight {
        // `unet`, `controls` and `ip_adapter` drop here; the shared driver then runs the runtime's
        // `after_component_drop` (an eval barrier + `clear_cache`) before decode begins.
        SdxlLight {
            vae: self.vae,
            pid: self.pid,
        }
    }

    fn decode_view(&self) -> SdxlDecodeView<'_> {
        SdxlDecodeView { vae: &self.vae }
    }

    fn light_view(light: &SdxlLight) -> SdxlDecodeView<'_> {
        SdxlDecodeView { vae: &light.vae }
    }
}

/// Whether a request sheds the IP-Adapter image encoder once its image prompt is encoded (epic 16774
/// lever): exactly a staged request, whose heavy bundle is its own (a warm bundle serves the next
/// request and keeps it).
pub(crate) fn sheds_ip_image_encoder(stage_residency: bool) -> bool {
    stage_residency
}

/// What the denoise phase hands the decode phase: every image's final latent (evaluated), plus the
/// PiD decoder minted for this request (owned, so it outlives the shed U-Net bundle).
type SdxlDenoised = (Vec<mlx_rs::Array>, Option<PidDecoder>);

/// Construct an [`Sdxl`] from a [`LoadSpec`].
///
/// `spec.weights` must be a [`WeightsSource::Dir`] pointing at a
/// `stabilityai/stable-diffusion-xl-base-1.0` snapshot (the diffusers multi-component tree —
/// `tokenizer/`, `tokenizer_2/`, `text_encoder/`, `text_encoder_2/`, `unet/`, `vae/`).
///
/// **Dtype:** the U-Net + both CLIP text encoders run **fp16**, matching the production reference
/// (`StableDiffusionXL(float16=True)`); the **VAE stays f32** (the vendored always loads the
/// autoencoder f32 — the SDXL VAE is fp16-unstable). The whole fp16 path is byte-identical to the
/// reference at the matched MLX — now 0.32.0, RE-CONFIRMED by sc-12896: the fp16 UNet golden was
/// re-dumped on the 0.32.0 non-NAX env and `unet_single_forward_matches_vendored_fp16` measures
/// 100.00% byte-exact (16384/16384). Established on 0.31.2 by sc-2721 (needs sc-2772's NAX 16-bit
/// fix + the compiled `gelu_exact`); fp16 is bit-identical across the 0.31.2→0.32.0 bump (on
/// SDXL's path the sc-12896 cross-stack drift is f32-only; dense-bf16 paths drift in other crates).
/// The lower-level `load_unet`/`load_text_encoder_*` keep an f32 path for the tight stage gates.
pub fn load(spec: &LoadSpec) -> Result<Box<dyn Generator>> {
    if spec.precision != Precision::Bf16 {
        return Err(Error::Msg(
            "sdxl: precision override is not wired; the dense path runs fp16 (the production \
             reference dtype) — drop the precision override"
                .into(),
        ));
    }
    if matches!(spec.weights, WeightsSource::File(_)) {
        mlx_gen::gen_core::reject_unknown_components(
            spec,
            &[LDM_TOKENIZER_COMPONENT, loader::VAE_FP16_FIX_COMPONENT],
            MODEL_ID,
        )?;
        return load_from_ldm_file(spec, tokenizer_root(spec)?);
    }
    Ok(Box::new(load_concrete(spec)?))
}

/// The directory whose `tokenizer/` the generator [`load`]ed for `spec` reads: the snapshot root
/// of a diffusers directory, or the caller-staged [`LDM_TOKENIZER_COMPONENT`] directory of a fused
/// single-file checkpoint. The one resolution both [`load`] and [`conditioning_window_count`] use.
fn tokenizer_root(spec: &LoadSpec) -> Result<&Path> {
    match &spec.weights {
        WeightsSource::Dir(root) => Ok(root),
        WeightsSource::File(_) => match spec.components.get(LDM_TOKENIZER_COMPONENT) {
            Some(WeightsSource::Dir(path)) => Ok(path),
            Some(WeightsSource::File(path)) => Err(Error::Msg(format!(
                "sdxl: imported fused checkpoint component '{LDM_TOKENIZER_COMPONENT}' must be a directory, got {}",
                path.display()
            ))),
            None => Err(Error::Msg(format!(
                "sdxl: imported fused checkpoint requires caller-staged '{LDM_TOKENIZER_COMPONENT}' tokenizer assets"
            ))),
        },
    }
}

/// How many 77-token CLIP windows a generate under `spec` encodes `prompt` (and, when CFG is on,
/// `negative`) into — the long-prompt window count (sc-20528) the iOS admission keys evidence on
/// (sc-26270). Pass `negative` exactly as `generate` does: `Some(negative_prompt or "")` when the
/// request's guidance enables CFG, `None` otherwise.
///
/// Weights-free: it reads only the tokenizer's `vocab.json` / `merges.txt`, resolved exactly as the
/// generator's load path resolves them for `spec`, never a tensor payload, and builds no array.
pub fn conditioning_window_count(
    spec: &LoadSpec,
    prompt: &str,
    negative: Option<&str>,
) -> Result<u32> {
    let tokenizer = loader::load_tokenizer(tokenizer_root(spec)?)?;
    let mut rows = vec![tokenizer.tokenize(prompt)?];
    if let Some(negative) = negative {
        rows.push(tokenizer.tokenize(negative)?);
    }
    let plan = crate::long_prompt::ChunkPlan::from_tokenizer(&tokenizer)?;
    u32::try_from(crate::long_prompt::common_chunks(&plan, &rows))
        .map_err(|_| Error::Msg("sdxl: CLIP window count overflows u32".into()))
}

/// Pure-host pre-key CLIP window count, guarded before fresh requested storage.
/// Uses borrowed members of the caller's retained source inventory. Both CLIP encoders use
/// the loader's one shared tokenizer and the existing common-window/empty-probe contract.
pub fn conditioning_window_count_with_source_capacity(
    spec: &LoadSpec,
    prompt: &str,
    negative: Option<&str>,
    source: &mlx_gen::gen_core::SourceInventory,
    next: &mut dyn FnMut(u64) -> Result<()>,
) -> Result<u32> {
    let root = tokenizer_root(spec)?;
    let vocab_path = root.join("tokenizer").join("vocab.json");
    let merges_path = root.join("tokenizer").join("merges.txt");
    let (vocab, merges) = mlx_gen::ios_pricing::with_core_source_capacity(next, |core_next| {
        Ok((
            source.member_pin_with_capacity(&vocab_path, core_next)?,
            source.member_pin_with_capacity(&merges_path, core_next)?,
        ))
    })?;
    // Path lookup storage is source metadata, not a whole-key/RSS certificate.
    crate::tokenizer::count_pinned_with_source_capacity(vocab, merges, prompt, negative, next)
}

/// Construct the concrete generator for the correctness-only production-latent admission harness.
/// Normal registry callers should use [`load`].
#[doc(hidden)]
pub fn load_concrete(spec: &LoadSpec) -> Result<Sdxl> {
    if spec.precision != Precision::Bf16 {
        // `Precision::Bf16` is the registry's dense sentinel; the dense path runs fp16 (the
        // production dtype). A non-default precision flag is rejected rather than silently ignored.
        return Err(Error::Msg(
            "sdxl: precision override is not wired; the dense path runs fp16 (the production \
             reference dtype) — drop the precision override"
                .into(),
        ));
    }
    // Resolve the snapshot dir up front — a fail-fast for BOTH residencies (Sequential defers the
    // heavy component build to each generate, but a single-file source is still wrong, so reject it
    // here rather than at the first generate).
    let root = resolve_root(spec)?;

    let cfg = DiffusionConfig::sdxl_base();
    let alpha_schedule =
        AlphaSchedule::scaled_linear(cfg.num_train_steps, cfg.beta_start, cfg.beta_end);
    // Component residency (epic 10834 Phase 1, sc-10839): the default `Resident` builds every heavy
    // component now and holds it warm; `Sequential` keeps only the spec and re-loads per generate in
    // phase order (encode → drop encoders → denoise/decode) to bound peak memory. The `Resident`
    // build is byte-identical to the pre-sc-10839 `load` — the same loaders, adapter/quant order,
    // and PiD overlay, just assembled through the shared per-phase helpers.
    let control_count = spec.control.is_some() as usize + spec.extra_controls.len();
    // F-181: Sequential + a load-time quant over a dense snapshot re-quantizes every generate. An
    // already-packed turnkey loads packed (no re-quant); `Resident` quantizes once. So warn only for
    // the Sequential-over-dense combination that actually pays the repeated cost.
    if let Some(q) = spec.quantize {
        if matches!(spec.offload_policy, OffloadPolicy::Sequential)
            && mlx_gen::quant::needs_load_time_quant(root, "unet", q.bits(), descriptor().id)?
        {
            mlx_gen::residency::warn_sequential_requantize(descriptor().id, q.bits());
        }
    }
    let residency = build_residency(spec)?;
    let memory_strategy = crate::memory_strategy::memory_strategy_contract(descriptor().id, spec)?;
    Ok(Sdxl {
        descriptor: descriptor(),
        default_stage_residency: crate::memory_strategy::default_stage_residency(spec),
        streamable: crate::memory_strategy::streamable(spec),
        loaded_spec: spec.clone(),
        memory_strategy,
        tokenizer: loader::load_tokenizer(root)?,
        sampler: EulerSampler::new_with_dtype(&cfg, true, DTYPE)?,
        alpha_schedule,
        control_count,
        residency,
    })
}

/// Load a fused SDXL LDM/A1111 checkpoint in memory. The fused file supplies both text encoders,
/// UNet, and VAE; only the model-agnostic tokenizer assets come from `tokenizer_root`.
pub fn load_from_ldm_file(spec: &LoadSpec, tokenizer_root: &Path) -> Result<Box<dyn Generator>> {
    spec.validate_prepared_file_pins()?;
    let file_pin = match &spec.weights {
        WeightsSource::File(_) => spec
            .weights_file_pin()?
            .expect("File weights must resolve to a pin"),
        WeightsSource::Dir(_) => {
            return Err(Error::Msg(
                "sdxl LDM loader expects a fused .safetensors file".into(),
            ))
        }
    };
    if spec.precision != Precision::Bf16 {
        return Err(Error::Msg(
            "sdxl: precision override is not wired; the dense path runs fp16".into(),
        ));
    }
    let residency = build_ldm_residency(spec, &file_pin)?;
    let cfg = DiffusionConfig::sdxl_base();
    let alpha_schedule =
        AlphaSchedule::scaled_linear(cfg.num_train_steps, cfg.beta_start, cfg.beta_end);
    // A fused LDM/A1111 checkpoint has no re-openable **per-component** source — every component is
    // cut out of one file by `ldm::split_ldm_checkpoint` — so `memory_strategy::streamable` reports
    // `false` for its `WeightsSource::File` and rung 4 is declared Missing for this load rather than
    // silently executing resident. Rung 1 is a different question and is genuinely executable here:
    // the fused *file* is re-openable through the retained pin, which is what
    // [`build_ldm_residency`] keeps for both phase loaders (sc-18317).
    // The public helper retains the same prepared identity guarantee as registry dispatch: contract
    // projection reopens the fused header and tokenizer loading opens staged files, so complete both
    // under the retained source + full prepared-file guards before exposing the generator.
    let (memory_strategy, tokenizer) = spec.read_prepared_files_unchanged(|| {
        file_pin.read_unchanged(|_| {
            Ok::<_, Error>((
                crate::memory_strategy::memory_strategy_contract(descriptor().id, spec)?,
                loader::load_tokenizer(tokenizer_root)?,
            ))
        })
    })?;
    Ok(Box::new(Sdxl {
        descriptor: descriptor(),
        default_stage_residency: crate::memory_strategy::default_stage_residency(spec),
        streamable: false,
        loaded_spec: spec.clone(),
        memory_strategy,
        tokenizer,
        sampler: EulerSampler::new_with_dtype(&cfg, true, DTYPE)?,
        alpha_schedule,
        control_count: spec.control.is_some() as usize + spec.extra_controls.len(),
        residency,
    }))
}

/// The fused-LDM policy→[`Residency`] dispatch, mirroring [`build_residency`]'s single-seam shape
/// for the imported-checkpoint route (sc-18317).
///
/// **Every arm retains the pin and keeps reload loaders.** Before sc-18317 the `Resident` arm read
/// the fused file once through a valid pin, dropped both, and built `Residency::resident` — a
/// non-rebuildable owner whose loaders return errors. That made rung 1 a *declared but unexecutable*
/// capability on this route: the descriptor publishes `supports_sequential_offload` (so
/// `staged_residency_availability()` is `Selectable`), the contract builder declares
/// `StagedResidency` `Implemented` (it derives only `load_shape` and rung 4 from the spec), rung 1
/// declares no prerequisite for `validate_selection` to refuse, and the shared
/// `Capabilities::validate_request` floor does not inspect `stage_residency` at all. A staged
/// selection was therefore admitted everywhere and then refused inside `generate` by
/// `Residency::ensure_rebuildable` — the epic's core defect class.
///
/// Nothing here is new machinery: the `Sequential` arm has always re-split the fused checkpoint
/// through this same pin on every generate, and `PinnedWeightsFile::read_unchanged` documents
/// repeated reopen as its intended use ("keep this same pin for every lazy or sequential reopen").
/// The nearest neighbour agrees — `mlx-gen-z-image`'s fused ComfyUI `Checkpoint` source builds a
/// request-scoped residency whose per-phase loaders re-split one pinned file. So the `Resident` arm
/// now routes through [`Residency::from_policy_with_resident`], which keeps a **distinct warm
/// aggregate**: the warm pair is still built by ONE fused read (byte-identical composition to the
/// pre-sc-18317 eager build, `load_pid = true` for the reusable PiD superset), while the two staged
/// phase loaders re-split the file per phase exactly as `Sequential` does.
///
/// One behavioural delta, and it is deliberate: under `Resident` the warm pair is now built on first
/// use instead of at load. Structural validity of the checkpoint is still proven at load — the
/// contract projection reads and classifies every tensor header through the pin
/// (`model::fused_ldm_component_footprint`) and rejects a file missing a text encoder, U-Net or VAE
/// — and `load` is only ever reached from the job that then calls `generate`, so a tensor-level
/// failure surfaces in the same job either way. `Sequential` on this route already deferred
/// everything.
pub(crate) fn build_ldm_residency(
    spec: &LoadSpec,
    file_pin: &mlx_gen::gen_core::PinnedWeightsFile,
) -> Result<Residency<(ClipTextEncoder, ClipTextEncoder), SdxlHeavyOwned>> {
    let warm_file = file_pin.clone();
    let warm_spec = spec.clone();
    let text_file = file_pin.clone();
    let text_quant = spec.quantize;
    let heavy_file = file_pin.clone();
    let heavy_spec = spec.clone();
    Residency::from_policy_with_resident(
        spec.offload_policy,
        move || {
            warm_spec.read_prepared_files_unchanged(|| {
                warm_file.read_unchanged(|file| {
                    crate::ldm::before_component_load(
                        crate::quantum::LDM_TEXT_LOAD,
                        file,
                        &[SdxlLdmComponent::ClipL, SdxlLdmComponent::ClipBigG],
                    )?;
                    let crate::ldm::LdmComponents {
                        unet,
                        clip_l,
                        clip_bigg,
                        vae,
                    } = crate::ldm::split_ldm_checkpoint(file)?;
                    let text = build_ldm_text(clip_l, clip_bigg, warm_spec.quantize)?;
                    crate::ldm::before_component_load(
                        crate::quantum::LDM_HEAVY_LOAD,
                        file,
                        ldm_heavy_components(&warm_spec),
                    )?;
                    let heavy = build_ldm_heavy(&warm_spec, unet, vae, true)?;
                    Ok::<_, Error>((text, heavy))
                })
            })
        },
        move || {
            text_file.read_unchanged(|file| {
                crate::ldm::before_component_load(
                    crate::quantum::LDM_TEXT_LOAD,
                    file,
                    &[SdxlLdmComponent::ClipL, SdxlLdmComponent::ClipBigG],
                )?;
                let crate::ldm::LdmComponents {
                    clip_l, clip_bigg, ..
                } = crate::ldm::split_ldm_checkpoint(file)?;
                build_ldm_text(clip_l, clip_bigg, text_quant)
            })
        },
        move |load_pid| {
            heavy_spec.read_prepared_files_unchanged(|| {
                heavy_file.read_unchanged(|file| {
                    crate::ldm::before_component_load(
                        crate::quantum::LDM_HEAVY_LOAD,
                        file,
                        ldm_heavy_components(&heavy_spec),
                    )?;
                    let crate::ldm::LdmComponents { unet, vae, .. } =
                        crate::ldm::split_ldm_checkpoint(file)?;
                    build_ldm_heavy(&heavy_spec, unet, vae, load_pid)
                })
            })
        },
    )
}

/// The fused-file components the heavy LDM phase constructs: the U-Net always, and the fused VAE
/// only when no staged fp16-fix VAE replaces it (that one is checked by its own loader).
fn ldm_heavy_components(spec: &LoadSpec) -> &'static [SdxlLdmComponent] {
    if spec.components.contains_key(loader::VAE_FP16_FIX_COMPONENT) {
        &[SdxlLdmComponent::Unet]
    } else {
        &[SdxlLdmComponent::Unet, SdxlLdmComponent::Vae]
    }
}

fn build_ldm_text(
    clip_l: mlx_gen::weights::Weights,
    clip_bigg: mlx_gen::weights::Weights,
    quantize: Option<Quant>,
) -> Result<(ClipTextEncoder, ClipTextEncoder)> {
    build_ldm_text_using(
        clip_l,
        clip_bigg,
        quantize,
        loader::load_text_encoder_1_from_weights,
        loader::load_text_encoder_2_from_weights,
        ClipTextEncoder::materialize_weights,
    )
}

/// The real fused component-map owner/order, with private constructor factories so tiny tests
/// exercise this same control flow. Does not bypass the outer immutable-file guard or remapper;
/// callers still own those boundaries. No public runtime architecture injection.
pub(crate) fn build_ldm_text_using(
    clip_l: mlx_gen::weights::Weights,
    clip_bigg: mlx_gen::weights::Weights,
    quantize: Option<Quant>,
    one: impl FnOnce(mlx_gen::weights::Weights, Dtype) -> Result<ClipTextEncoder>,
    two: impl FnOnce(mlx_gen::weights::Weights, Dtype) -> Result<ClipTextEncoder>,
    mut materialize: impl FnMut(&ClipTextEncoder) -> Result<()>,
) -> Result<(ClipTextEncoder, ClipTextEncoder)> {
    // Consume each lazy map exactly as before; no parallel source owner survives its constructor.
    let mut te1 = one(clip_l, DTYPE)?;
    let mut te2 = two(clip_bigg, DTYPE)?;
    if let Some(quant) = quantize {
        crate::text_encoder::quantize_pair(&mut te1, &mut te2, quant.bits())?;
    }
    materialize(&te1)?;
    materialize(&te2)?;
    Ok((te1, te2))
}

fn build_ldm_heavy(
    spec: &LoadSpec,
    unet_weights: mlx_gen::weights::Weights,
    vae_weights: mlx_gen::weights::Weights,
    load_pid: bool,
) -> Result<SdxlHeavyOwned> {
    // The U-Net component map is consumed by the constructor so no second full source map retains
    // arrays while the packed model is walked below. The VAE intentionally stays dense and is
    // materialized as its own bounded component under the same immutable-file guard.
    let mut unet = loader::load_unet_from_weights(unet_weights, DTYPE)?;
    // A staged fp16-fix VAE replaces the fused file's VAE (sc-26268); the fused VAE's lazy arrays are
    // then dropped unread rather than materialized for nothing.
    let vae = match spec.components.get(loader::VAE_FP16_FIX_COMPONENT) {
        Some(source) => {
            drop(vae_weights);
            loader::load_vae_fp16_fix(source)?
        }
        None => {
            vae_weights.materialize()?;
            loader::load_vae_from_weights(vae_weights)?
        }
    };
    if !spec.adapters.is_empty() {
        let coverage = crate::adapters::configured_lora_coverage();
        crate::adapters::apply_sdxl_adapters_with(&mut unet, &spec.adapters, coverage)?;
    }
    let mut controls = Vec::new();
    if let Some(source) = &spec.control {
        controls.push(loader::load_controlnet(source, DTYPE)?);
    }
    for source in &spec.extra_controls {
        controls.push(loader::load_controlnet(source, DTYPE)?);
    }
    let ip_adapter = match &spec.ip_adapter {
        Some(WeightsSource::Dir(path)) => {
            let (encoder, pairs) = loader::load_ip_adapter(path, DTYPE)?;
            unet.install_ip_adapter(pairs)?;
            Some(encoder)
        }
        Some(WeightsSource::File(_)) => {
            return Err(Error::Msg(
                "sdxl ip_adapter expects an h94/IP-Adapter snapshot directory".into(),
            ))
        }
        None => None,
    };
    if let Some(quant) = spec.quantize {
        let bits = quant.bits();
        unet.quantize(bits)?;
        for control in &mut controls {
            control.quantize(bits)?;
        }
    }
    unet.materialize_weights()?;
    let pid = if load_pid {
        spec.pid
            .as_ref()
            .map(|weights| PidEngine::from_spec(weights, PID_BACKBONE))
            .transpose()?
    } else {
        None
    };
    if let Some(guard) = spec.source_inventory() {
        unet.bind_source_inventory(guard);
    }
    Ok(SdxlHeavyOwned {
        unet,
        controls,
        ip_adapter: std::sync::Mutex::new(ip_adapter),
        vae,
        pid,
    })
}

/// The policy→[`Residency`] dispatch, routed through the single [`Residency::from_policy`] seam
/// (sc-10839; hoisted to the shared seam in sc-11126, F-180) so the `match offload_policy` lives in
/// exactly one place. `Resident` eager-loads the dual CLIP text encoders + heavy bundle now (the heavy
/// loader with `use_pid = true`, loading any PiD overlay once and reusing it); `Sequential` captures
/// the two per-phase loaders and loads nothing now, deferring each to [`Residency::run`]. Both use the
/// same [`load_text_encoders`] / [`load_heavy`], so the `Resident` composition is byte-identical to the
/// pre-seam one. The deferral is weight-free-testable: under `Sequential` this touches no component
/// weights, so a dispatch that ignored `offload_policy` would eager-load and fail the "Sequential
/// defers" unit test.
pub(crate) fn build_residency(
    spec: &LoadSpec,
) -> Result<Residency<(ClipTextEncoder, ClipTextEncoder), SdxlHeavyOwned>> {
    let spec_text = spec.clone();
    let spec_heavy = spec.clone();
    Residency::from_policy(
        spec.offload_policy,
        move || {
            spec_text.read_prepared_files_unchanged(|| {
                load_text_encoders(resolve_root(&spec_text)?, spec_text.quantize)
            })
        },
        move |use_pid| {
            spec_heavy.read_prepared_files_unchanged(|| {
                load_heavy(&spec_heavy, resolve_root(&spec_heavy)?, use_pid)
            })
        },
    )
}

/// Resolve the snapshot directory from the load spec, rejecting a single-file source (SDXL needs the
/// diffusers multi-component tree). Shared by [`load`] and the `Sequential` per-phase loaders.
fn resolve_root(spec: &LoadSpec) -> Result<&Path> {
    match &spec.weights {
        WeightsSource::Dir(p) => Ok(p),
        WeightsSource::File(_) => Err(Error::Msg(
            "sdxl expects a snapshot directory (tokenizer/ text_encoder/ unet/ vae/ …), not a \
             single .safetensors file"
                .into(),
        )),
    }
}

/// Load the dual CLIP text encoders (the phase-A component under `Sequential`, dropped before the
/// U-Net loads). Factored out of [`load`] so the `Resident` and `Sequential` paths build byte-
/// identical encoders: both run fp16 with the same optional Q4/Q8 (`group_size 64`) over every
/// quantizable Linear + the token Embedding, matching the sc-2604/sc-1975 quant scope.
fn load_text_encoders(
    root: &Path,
    quant: Option<Quant>,
) -> Result<(ClipTextEncoder, ClipTextEncoder)> {
    load_text_encoders_using(
        root,
        quant,
        loader::load_text_encoder_1_dtype,
        loader::load_text_encoder_2_dtype,
    )
}

/// The actual directory load/tier-check/pack sequence. Private factories only replace encoder
/// construction in tests; requested-vs-packed tier validation remains the real production check.
pub(crate) fn load_text_encoders_using(
    root: &Path,
    quant: Option<Quant>,
    one: impl FnOnce(&Path, Dtype) -> Result<ClipTextEncoder>,
    two: impl FnOnce(&Path, Dtype) -> Result<ClipTextEncoder>,
) -> Result<(ClipTextEncoder, ClipTextEncoder)> {
    let mut te1 = one(root, DTYPE)?;
    let mut te2 = two(root, DTYPE)?;
    if let Some(q) = quant {
        // F-144 (sc-11129): reject a requested-vs-packed tier mismatch up front. `quantize()` silently
        // no-ops on already-packed weights, so a Q4 request over a pre-quantized Q8 turnkey would
        // otherwise serve Q8 with no diagnostic. `needs_load_time_quant` errors on a mismatch; on a
        // matching-packed or dense snapshot it returns Ok and the quantize below stands (a no-op on the
        // already-packed encoders, a real pack on a dense snapshot).
        mlx_gen::quant::needs_load_time_quant(root, "unet", q.bits(), descriptor().id)?;
        let bits = q.bits();
        crate::text_encoder::quantize_pair(&mut te1, &mut te2, bits)?;
    }
    Ok((te1, te2))
}

/// Load the heavy render-phase components — U-Net (+ LoRA/LoKr merge + IP-Adapter install + Q4/Q8),
/// ControlNet branches (+ Q4/Q8), VAE (f32), and the optional PiD overlay — everything but the text
/// encoders. Factored out of [`load`] so the `Sequential` path can load these AFTER the encoders are
/// dropped (bounding peak to `max(encoders, U-Net+VAE)`), and the `Resident` path builds the same
/// bundle up front. The operation order matches the pre-sc-10839 `load` (adapter merge before quant),
/// and the components are independent of the text encoders, so both residencies are byte-identical.
fn load_heavy(spec: &LoadSpec, root: &Path, load_pid: bool) -> Result<SdxlHeavyOwned> {
    // Materialized at load (sc-24245); a load that arms rung 4 below leaves the streamed
    // transformer-block bodies to the block stream.
    let mut unet = loader::load_unet_with_config_streamed(
        root,
        DTYPE,
        &UNetConfig::sdxl_base(),
        crate::memory_strategy::streamable(spec),
    )?;
    if !spec.adapters.is_empty() {
        // Merge LoRA (kohya `lora_unet_` / PEFT, sc-2639) and LoKr (sc-2640) into the dense fp16
        // U-Net weights at load — the production reference merges into the `float16=True` U-Net too,
        // and merging (not a
        // forward-time residual) keeps the chaos-sensitive ancestral sampler bit-exact. Out-of-surface
        // keys (mid_block/ff/conv) are surfaced in the report, not dropped.
        //
        // Coverage (sc-2671): default to the strictly-more-correct COMPLETE surface — mid_block +
        // the GEGLU FF the vendored `lora.py` silently drops — so SDXL LoRAs apply in full, matching
        // diffusers (Michael's correctness-over-parity call, 2026-06-03). `SDXL_LORA_VENDORED` is the
        // escape hatch back to the legacy 515-module surface for byte-parity with the retired Python
        // path.
        let coverage = crate::adapters::configured_lora_coverage();
        if coverage == crate::adapters::LoraCoverage::Vendored {
            eprintln!(
                "sdxl: SDXL_LORA_VENDORED set — restricting LoRA to the legacy vendored 515-module \
                 surface (mid_block + ff dropped; byte-parity with the retired Python path)"
            );
        }
        crate::adapters::apply_sdxl_adapters_with(&mut unet, &spec.adapters, coverage)?;
    }
    // The in-snapshot VAE at f32 (the vendored reference loads the autoencoder `float16=False`), or
    // the staged fp16-fix VAE at f16 when the spec carries one (sc-26268).
    let vae = loader::load_vae_for_spec(spec, root)?;

    // ControlNet branches (sc-3058; MultiControlNet sc-3378) — `spec.control` first, then each
    // `spec.extra_controls`, all at the U-Net dtype (fp16). Quantized with the U-Net below when
    // `spec.quantize` is set (the encoder-copy Linears; conv stem / cond-embedding / zero-convs stay
    // dense, matching the U-Net scope).
    let mut controls: Vec<ControlNet> = Vec::new();
    if let Some(src) = &spec.control {
        controls.push(loader::load_controlnet(src, DTYPE)?);
    }
    for src in &spec.extra_controls {
        controls.push(loader::load_controlnet(src, DTYPE)?);
    }

    // Optional IP-Adapter (sc-3059) — install the decoupled-attn K/V pairs into the still-mutable,
    // pre-quant U-Net (so they quantize with it) and keep the image-token encoder.
    let ip_adapter = match &spec.ip_adapter {
        Some(WeightsSource::Dir(p)) => {
            let (enc, pairs) = loader::load_ip_adapter(p, DTYPE)?;
            unet.install_ip_adapter(pairs)?;
            Some(enc)
        }
        Some(WeightsSource::File(_)) => {
            return Err(Error::Msg(
                "sdxl ip_adapter expects an h94/IP-Adapter snapshot directory, not a single file"
                    .into(),
            ));
        }
        None => None,
    };

    if let Some(q) = spec.quantize {
        // Q4/Q8 (group_size 64) over every quantizable Linear of the U-Net + control branches —
        // applied AFTER the adapter merge (the merge needs the dense weight; `merge_dense_delta`
        // errors on a quantized base, matching the fork's "LoRA merged pre-quantization"). The core
        // `AdaptableLinear::quantize` casts each weight to bf16 before packing (sc-2604): SDXL ships
        // fp16/fp32 on disk, and quantizing the as-loaded dtype would give drifted group scales — the
        // sc-1975 "Q8 broken on base-1.0". Convs / norms / token & position embeddings stay dense
        // (gather lookups, not matmuls). The text encoders are quantized in [`load_text_encoders`];
        // the **VAE stays f32** — its only Linears are the tiny quant/post-quant projections
        // (negligible memory), and a dense decode preserves output quality. Scope verified
        // empirically by the full `load(Q).generate()` gate (sc-2641).
        //
        // F-144 (sc-11129): reject a requested-vs-packed U-Net tier mismatch first — `quantize()`
        // no-ops on an already-packed snapshot, so a Q4 request over a pre-quantized Q8 turnkey would
        // silently serve Q8. On a matching-packed or dense snapshot this returns Ok and the quantize
        // below stands (a no-op on the packed U-Net, a real pack on the dense control branches).
        mlx_gen::quant::needs_load_time_quant(root, "unet", q.bits(), descriptor().id)?;
        let bits = q.bits();
        crate::quant::guarded_load_quantize(|| -> Result<()> {
            unet.quantize(bits)?;
            for cn in &mut controls {
                cn.quantize(bits)?;
            }
            Ok(())
        })?;
    }

    // PiD decoder overlay (epic 7840, sc-7848): load the `sdxl` student + Gemma caption encoder once
    // when the spec carries it AND this generate uses it (`load_pid`, F-177) — Resident passes `true`
    // (loaded once, reused), Sequential passes `req.use_pid` so a non-PiD generate skips the student +
    // its Gemma caption encoder entirely. Shared across the whole SDXL family (sdxl/realvisxl) — and
    // Kolors, which loads its own engine via `mlx_gen_sdxl::model::PID_BACKBONE`.
    let pid = if load_pid {
        spec.pid
            .as_ref()
            .map(|p| PidEngine::from_spec(p, PID_BACKBONE))
            .transpose()?
    } else {
        None
    };

    // Ladder rung 4 (SC-15525, closing SC-16355) — armed LAST, and that ordering is the whole
    // correctness argument. Each `Transformer2D`'s stream captures the installed IP-Adapter K/V
    // projections and the forward-time residual adapters off its FINISHED resident blocks, so the
    // streamed and resident paths cannot disagree about which state landed where. Arming before the
    // IP install or the adapter merge would capture an empty state and silently render without the
    // image prompt.
    //
    // `memory_strategy::streamable` is the single authority for whether this load may arm at all: it
    // refuses a fused single file, an eager load shape, a load-time quantization that materializes
    // the trunk, and an adapter load that merged its delta into weights the snapshot does not carry.
    if crate::memory_strategy::streamable(spec) {
        let unet_file =
            loader::resolve_weight_file(root, "unet", "diffusion_pytorch_model", DTYPE)?;
        unet.arm_block_streams(
            &WeightsSource::File(unet_file),
            DTYPE,
            spec.quantize.map(|q| q.bits()),
        );
    }

    if let Some(guard) = spec.source_inventory() {
        unet.bind_source_inventory(guard);
    }
    Ok(SdxlHeavyOwned {
        unet,
        controls,
        ip_adapter: std::sync::Mutex::new(ip_adapter),
        vae,
        pid,
    })
}

// Written out rather than `mlx_gen::impl_generator!` because SDXL now answers the memory-strategy
// hooks (SC-15525); the macro covers only `descriptor`/`validate`/`generate`.
impl mlx_gen::gen_core::Generator for Sdxl {
    fn descriptor(&self) -> &ModelDescriptor {
        &self.descriptor
    }

    fn validate(&self, req: &GenerationRequest) -> mlx_gen::gen_core::Result<()> {
        validate_request(&self.descriptor.capabilities, req).map_err(Into::into)
    }

    fn generate(
        &self,
        req: &GenerationRequest,
        on_progress: &mut dyn FnMut(Progress),
    ) -> mlx_gen::gen_core::Result<GenerationOutput> {
        self.generate_impl(req, on_progress).map_err(Into::into)
    }

    fn memory_strategy_contract(&self) -> Option<&mlx_gen::gen_core::MemoryProviderContract> {
        Some(&self.memory_strategy)
    }

    fn memory_strategy_safety_check(
        &self,
        context: &mlx_gen::gen_core::MemoryRunContext,
    ) -> mlx_gen::gen_core::MemorySafetyDecision {
        crate::memory_strategy::safety_check(&self.loaded_spec, &self.memory_strategy, context)
    }

    fn begin_memory_strategy_request(
        &self,
        context: &mlx_gen::gen_core::MemoryRunContext,
    ) -> mlx_gen::gen_core::Result<Option<Box<dyn mlx_gen::gen_core::MemoryRequestScope + '_>>>
    {
        crate::memory_strategy::begin_request(
            self.descriptor.id,
            &self.loaded_spec,
            &self.memory_strategy,
            context,
        )
    }
}

impl Sdxl {
    /// Run the ordinary production denoise path and compare the exact final latent under dense and
    /// candidate tiled decode. This is an explicit correctness-only seam for admission tooling; it
    /// does not read clocks or allocator/memory counters and does not change the production output.
    #[doc(hidden)]
    pub fn generate_decode_quality(
        &self,
        req: &GenerationRequest,
        tiling: &mlx_gen::tiling::TilingConfig,
        on_progress: &mut dyn FnMut(Progress),
    ) -> Result<(GenerationOutput, Vec<DecodeQualitySample>)> {
        let mut samples = Vec::new();
        let output = {
            let mut capture = |vae: &Autoencoder,
                               latent: &mlx_rs::Array,
                               pid: Option<&dyn LatentDecoder>|
             -> Result<()> {
                latent.eval()?;
                let dense =
                    crate::pipeline::decode_image_tiled(vae, latent, pid, None, Some(&req.cancel))?;
                let tiled = crate::pipeline::decode_image_tiled(
                    vae,
                    latent,
                    pid,
                    Some(tiling),
                    Some(&req.cancel),
                )?;
                samples.push(DecodeQualitySample {
                    production_latent: latent.clone(),
                    dense,
                    tiled,
                });
                Ok(())
            };
            self.generate_impl_with_final_latent_observer(req, on_progress, Some(&mut capture))?
        };
        Ok((output, samples))
    }

    /// The rich-`Result` body behind [`Generator::generate`]. Kept on the crate's own
    /// [`mlx_gen::Error`] so the `?` operator lifts both `mlx_rs` device exceptions and the family
    /// helpers transparently; the trait wrapper bridges the tail into [`gen_core::Error`] (epic 3720).
    ///
    /// The staged residency lifecycle (text encode → drop the CLIP encoders under `Sequential` → load
    /// the U-Net/control/IP/VAE/PiD → denoise/decode → free the heavy bundle) is driven by the shared
    /// [`Residency::run`] seam (sc-11125), which owns the eval/drop/clear discipline, the
    /// stage-boundary cancel checks, and the error-safe cache flush.
    fn generate_impl(
        &self,
        req: &GenerationRequest,
        on_progress: &mut dyn FnMut(Progress),
    ) -> Result<GenerationOutput> {
        self.generate_impl_with_final_latent_observer(req, on_progress, None)
    }

    fn generate_impl_with_final_latent_observer(
        &self,
        req: &GenerationRequest,
        on_progress: &mut dyn FnMut(Progress),
        mut final_latent_observer: Option<FinalLatentObserver<'_>>,
    ) -> Result<GenerationOutput> {
        self.validate(req)?;
        // sc-26270: the request's iOS quantum scope (cache policy for the whole generate, every
        // component load and quantum inside it). A no-op off device.
        let _quantum_scope = mlx_gen::ios_quantum::Scope::try_new()?;
        let request = self.resolve_request(req)?;

        self.residency.run_staged_request_scoped(
            request.stage_residency,
            false,
            &req.cancel,
            req.use_pid,
            on_progress,
            |text: &(ClipTextEncoder, ClipTextEncoder)| {
                encode_conditioning_windows(&text.0, &text.1, &request.tokens)
            },
            // Materialize the conditioning + pooled while the encoders are still alive (Sequential
            // only) — MLX is lazy, so an un-evaluated output keeps the encoders referenced through the
            // graph and the drop would free nothing (cf. Wan's `encode_text_staged`).
            |enc| match enc {
                Some(enc) => Ok(mlx_rs::transforms::eval([&enc.0, &enc.1])?),
                None => Ok(()),
            },
            // ── Phase B (denoise): the heavy render components (U-Net + control/IP + VAE + PiD) run
            // every image's denoise against the `heavy` borrow — identical for both residencies — and
            // hand the evaluated latents to phase C. `on_progress` is threaded through the seam
            // (F-179) and shadows the outer sink here.
            |heavy_owned: &SdxlHeavyOwned, enc, on_progress| {
                self.denoise_phase(&request, heavy_owned, enc, on_progress)
            },
            // Materialize every final latent while the U-Net is still alive: MLX is lazy, so an
            // un-evaluated latent keeps the whole denoise graph — and through it the U-Net — referenced,
            // and the shed below would free nothing.
            |(finals, _pid): &SdxlDenoised| Ok(mlx_rs::transforms::eval(finals.iter())?),
            // ── Phase C (decode): on a staged request the U-Net, control branches and IP encoder
            // have been dropped (`StagedHeavy::shed_dit`) and the cache cleared before this runs; a
            // resident request decodes against the warm bundle exactly as before.
            |view: SdxlDecodeView<'_>, denoised: SdxlDenoised, on_progress| {
                decode_phase(
                    &view,
                    denoised,
                    &mut final_latent_observer,
                    request.decode_tiling.as_ref(),
                    &req.cancel,
                    on_progress,
                )
            },
        )
    }

    /// Everything the request resolves before the residency lifecycle starts: the sampling route,
    /// the image roles (validated), the tokenized prompt windows and the memory-ladder rungs.
    fn resolve_request<'r>(&self, req: &'r GenerationRequest) -> Result<SdxlRequest<'r>> {
        let sampling = crate::sampling::ResolvedSampling::resolve(req)?;
        let sampler_name = sampling.sampler_name();
        let is_accel = sampling.route().is_acceleration();
        let use_curated = sampling.route().is_curated();
        let cfg_on = sampling.guidance() > 1.0;
        let negative = req.negative_prompt.as_deref().unwrap_or("");
        let base_seed = req.seed.unwrap_or_else(default_seed);
        let reference = self.resolve_reference(req)?;
        let mask = self.resolve_mask(req)?;

        // Acceleration variants are txt2img-only in v1 (epic 2755 "Image-only v1"); reject an init
        // image rather than silently ignoring it.
        validate_image_roles(
            sampler_name,
            is_accel,
            use_curated,
            reference.is_some(),
            mask.is_some(),
        )?;
        // ControlNet (sc-3058; MultiControlNet sc-3378): each `Control` conditioning pairs, in order,
        // with a loaded control branch (`spec.control` + `spec.extra_controls`); their residuals are
        // summed. The common denoiser correctly carries those residuals through Lightning's one-row,
        // CFG-free loop; LCM/Hyper remain deliberately unsupported until separately characterized.
        // Control still does not combine with an inpaint mask in this build.
        let control_reqs = self.resolve_control(req)?;
        validate_control_roles(
            sampler_name,
            req.guidance,
            control_reqs.len(),
            self.control_count,
            mask.is_some(),
        )?;

        // ── Phase A input: text tokens (epic 10834 Phase 1, sc-10839; sc-11125). Seed-independent
        // (no RNG) — tokenizing above the residency lifecycle, then hoisting the encode above the
        // control/IP builds and the per-image loop, is byte-identical to the F-068 order. Under
        // `Sequential` the shared seam LOADS the dual CLIP encoders, encodes, materializes, then
        // DROPS them + `clear_cache()` so their ~1 GB frees before the U-Net/control/IP bundle
        // loads — bounding peak to `max(encoders, U-Net+VAE)`. Under `Resident` it borrows the
        // warm encoders (byte-identical to the pre-sc-10839 `encode_conditioning`).
        // sc-20528: a prompt past CLIP's 77-token context is split into windows, not truncated. The
        // window count is decided ONCE here, as the max over both CFG rows, so `[cond, uncond]` stay
        // stackable and the negative prompt takes exactly the same path as the positive one. A
        // request whose rows both fit is a single window that IS the pre-sc-20528 token batch.
        let tokens = self
            .tokenizer
            .tokenize_windows(&req.prompt, if cfg_on { Some(negative) } else { None })?;

        // ── Ladder request resolution (SC-15525) ────────────────────────────────────────────────
        // Rung 1 is request-scoped from here on: the load-time `OffloadPolicy` is only the default a
        // request that names nothing keeps.
        let stage_residency =
            crate::memory_strategy::stage_residency(req, self.default_stage_residency);
        let window_size = crate::memory_strategy::transformer_window_size(req)?;
        // Two fail-closed guards a calibration harness driving `generate` with a hand-built
        // `GenerationMemory` must still cross — it never went through `safety_check`.
        if window_size.is_some() && !self.streamable {
            return Err(Error::Unsupported(
                "sdxl: bounded transformer residency needs a DeferredMaterialization load over a \
                 snapshot directory whose U-Net stays lazy and whose adapters (if any) are \
                 replayable; this generator cannot stream its blocks"
                    .into(),
            ));
        }
        if window_size.is_some() && !stage_residency {
            return Err(Error::Unsupported(
                "sdxl: bounded transformer residency requires staged residency engaged in the same \
                 request — without the phase release both CLIP towers stay resident through the \
                 denoise and the request peak does not move"
                    .into(),
            ));
        }
        let decode_tiling =
            crate::memory_strategy::decode_tiling_for_contract(req, &self.memory_strategy)?;
        let forward_plan = crate::plan::SdxlForwardPlan::with_attention(
            crate::memory_strategy::attention_plan(req)?,
        )
        .with_window(window_size.map(|size| crate::plan::SdxlBlockWindow {
            size,
            cancel: &req.cancel,
        }))
        .with_cfg_batching(sampling.cfg_batching());

        Ok(SdxlRequest {
            req,
            sampling,
            base_seed,
            reference,
            mask,
            control_reqs,
            tokens,
            stage_residency,
            decode_tiling,
            forward_plan,
        })
    }

    /// Phase B: prepare the seed-independent conditioning, encode the init, resolve the PiD
    /// `from_ldm` plan, then denoise every output serially through the shared stage skeleton.
    fn denoise_phase(
        &self,
        request: &SdxlRequest<'_>,
        heavy_owned: &SdxlHeavyOwned,
        (conditioning, pooled): (mlx_rs::Array, mlx_rs::Array),
        on_progress: &mut dyn FnMut(Progress),
    ) -> Result<SdxlDenoised> {
        let req = request.req;
        let heavy = heavy_owned.as_ref();
        let (mode, guidance) = prepare_conditioning(request, heavy_owned, &heavy)?;
        let time_ids = text_time_ids(pooled.shape()[0]);
        let latent_shape = [1, (req.height / 8) as i32, (req.width / 8) as i32, 4];
        let init = build_init_latents(request, heavy.vae, mode)?;
        let vp_plan = self.resolve_vp_plan(request, &init)?;
        let capture_sigma = vp_plan.map(|p| p.sigma).unwrap_or(0.0);
        let pid_decoder = resolve_pid_decoder_at_sigma(
            heavy.pid,
            req,
            request.base_seed,
            self.descriptor.id,
            capture_sigma,
        )?;

        mlx_gen::diagnostics::record_phase_boundary(
            mlx_gen::diagnostics::BenchmarkPhaseBoundary::DenoiseStart,
        );
        let plan = SdxlRenderPlan {
            request,
            unet: heavy.unet,
            sampler: &self.sampler,
            alpha_schedule: &self.alpha_schedule,
            guidance: &guidance,
            init: &init,
            conditioning: &conditioning,
            pooled: &pooled,
            time_ids: &time_ids,
            latent_shape,
            vp_plan,
        };
        // The common serial owner seeds before this closure and retains the appended latents
        // through every subsequent output, exactly as the historical inline count loop did.
        let finals = collect_serial_latents(req.count, request.base_seed, |seed, finals| {
            crate::stages::run_output(&plan, seed, on_progress, |latents, _| {
                mlx_gen::sampler::diagnostic_sampling_tensor(
                    "sdxl",
                    "serial-final-latent",
                    finals.len() as u64,
                    1,
                    &latents,
                );
                finals.push(latents);
                Ok(())
            })
        })?;
        Ok((finals, pid_decoder))
    }

    /// PiD decode overlay (epic 7840, sc-7848) + `from_ldm` early-stop (sc-8049). SDXL is the lone
    /// **variance-preserving** PiD student. Its two denoise paths keep the latent in DIFFERENT
    /// frames, so the from_ldm capture is handled per-path:
    ///   • ancestral (the default; txt2img / img2img / control / IP via `denoise_core`) stores the
    ///     latent ALREADY renormalized to the VP frame `(x0+σε)/√(σ²+1)` = `√(1−σ_vp²)·x0 + σ_vp·ε`
    ///     at every node (see `EulerSampler::step`/`add_noise`), so a truncated x_k is handed to PiD
    ///     as-is (no rescale);
    ///   • curated (opt-in k-diffusion) stores RAW VE latents `x0+σ·ε`, so a truncated x_k is mapped
    ///     into the VP frame by the plan's rescale (`1/√(1+σ²)`) before decode.
    ///
    /// Both stay 0.13025-normalized throughout (the loop runs in the scaled latent space
    /// `vae.decode` consumes), so no extra normalization is applied. The plan is image-independent
    /// (the schedule is fixed by steps / scheduler / strength), so it is resolved once and the
    /// decoder is minted at the achieved degrade σ. The clean σ=0 decode (`None`) is byte-identical
    /// to before. The few-step accel path (decode-bound → no from_ldm benefit, sc-7993) and the
    /// inpaint mask-blend (needs the full schedule to σ=0) keep the clean decode; a from_ldm request
    /// on either errors loudly rather than silently dropping the knob.
    fn resolve_vp_plan(
        &self,
        request: &SdxlRequest<'_>,
        init: &SdxlInitLatents,
    ) -> Result<Option<VpCapturePlan>> {
        let req = request.req;
        if !(req.use_pid && req.pid_capture_sigma.is_some()) {
            return Ok(None);
        }
        let is_accel = request.sampling.route().is_acceleration();
        let is_inpaint = matches!(init, SdxlInitLatents::Inpaint { .. });
        if is_accel || is_inpaint {
            return Err(Error::Msg(format!(
                "{}: pid_capture_sigma (from_ldm early-stop) is not supported on the SDXL {} path \
                 (it keeps the clean σ=0 decode); use the standard ancestral or curated denoise for \
                 from_ldm (sc-8049)",
                self.descriptor.id,
                if is_accel {
                    "few-step accel"
                } else {
                    "inpaint mask-blend"
                }
            )));
        }
        // Resolve the VP capture against the EXACT σ schedule this run will denoise, so `keep` and
        // the achieved degrade σ agree with the truncated trajectory. (This mirrors the per-path
        // schedule build in `build_init` — deterministic host math, no RNG, so it does not perturb
        // the ancestral noise stream.)
        let steps = request.sampling.steps();
        let max_time = self.sampler.max_time();
        let init_strength = init
            .requested_strength()
            .map(|strength| resolve_strength(strength, DEFAULT_STRENGTH));
        let edm_sigmas = if request.sampling.route().is_curated() {
            let full_sigmas = request.sampling.curated_full_sigmas(&self.alpha_schedule)?;
            match init_strength {
                Some(strength) => curated_strength_schedule(&full_sigmas, steps, strength),
                None => full_sigmas,
            }
        } else {
            let (eff, start_time) = match init_strength {
                Some(strength) => ancestral_strength_schedule(steps, max_time, strength),
                None => (steps, max_time),
            };
            AncestralEuler::new(&self.sampler, eff, start_time)?.edm_sigmas()
        };
        Ok(vp_capture_plan(&edm_sigmas, req.pid_capture_sigma))
    }
}

/// The request-level SDXL plan resolved before the residency lifecycle (sc-27336): sampling,
/// validated image roles, tokens and the memory-ladder rungs. Borrowed by every later stage.
struct SdxlRequest<'r> {
    req: &'r GenerationRequest,
    sampling: crate::sampling::ResolvedSampling,
    base_seed: u64,
    reference: Option<(&'r Image, Option<f32>)>,
    mask: Option<&'r Image>,
    control_reqs: Vec<(&'r Image, f32)>,
    tokens: crate::ChunkedTokens,
    stage_residency: bool,
    decode_tiling: Option<mlx_gen::tiling::TilingConfig>,
    forward_plan: crate::plan::SdxlForwardPlan<'r>,
}

/// The role the request's Reference / Mask / Control inputs play (sc-27336), resolved once from
/// the validated request and the loaded bundle's IP presence. Exactly one mode holds, so the stage
/// functions pattern-match instead of re-deriving it from coupled `Option`s.
#[derive(Clone, Copy, Debug)]
pub(crate) enum SdxlMode<'r> {
    /// No Reference (an acceleration request is always this or [`Self::Control`] with no init).
    Txt2Img,
    /// The Reference is the img2img init; `strength` is the raw requested strength.
    Img2Img {
        image: &'r Image,
        strength: Option<f32>,
    },
    /// The Reference is the IP-Adapter image prompt (sc-3059), not an init.
    Ip { image: &'r Image, scale: f32 },
    /// ControlNet residuals (sc-3058), optionally on an img2img init from the Reference.
    Control {
        init: Option<(&'r Image, Option<f32>)>,
    },
    /// Masked inpaint (sc-3057) over the Reference.
    Inpaint {
        image: &'r Image,
        strength: Option<f32>,
        mask: &'r Image,
    },
}

impl<'r> SdxlMode<'r> {
    /// The mode matrix. IP mode needs loaded IP weights and a Reference with no mask, control or
    /// acceleration ([`uses_ip_reference`]); otherwise a non-accelerated Reference is an init
    /// (inpaint when masked, ControlNet-on-img2img when controlled, plain img2img otherwise). A Mask
    /// with no init and Control with a Mask are refused by [`validate_image_roles`] /
    /// [`validate_control_roles`] before this runs.
    pub(crate) fn resolve(
        has_ip_adapter: bool,
        reference: Option<(&'r Image, Option<f32>)>,
        mask: Option<&'r Image>,
        control_count: usize,
        is_accel: bool,
    ) -> Self {
        if let Some((image, strength)) = reference {
            if uses_ip_reference(
                has_ip_adapter,
                true,
                mask.is_some(),
                control_count,
                is_accel,
            ) {
                return Self::Ip {
                    image,
                    scale: ip_reference_scale(strength),
                };
            }
            if !is_accel {
                return match mask {
                    Some(mask) => Self::Inpaint {
                        image,
                        strength,
                        mask,
                    },
                    None if control_count > 0 => Self::Control {
                        init: Some((image, strength)),
                    },
                    None => Self::Img2Img { image, strength },
                };
            }
        }
        if control_count > 0 {
            Self::Control { init: None }
        } else {
            Self::Txt2Img
        }
    }
}

/// The seed-independent guidance inputs the denoise reads besides the text conditioning.
enum SdxlGuidance<'h> {
    /// Text conditioning only.
    Plain,
    /// The IP image-prompt tokens (CFG-batched with a zeros uncond row when CFG is on).
    Ip { tokens: mlx_rs::Array, scale: f32 },
    /// One context per loaded ControlNet branch, paired by order.
    Control(Vec<ControlContext<'h>>),
}

impl SdxlGuidance<'_> {
    /// The `(controls, ip)` pair the curated / CFG++ denoise takes.
    fn curated_inputs(&self) -> (&[ControlContext<'_>], Option<(&mlx_rs::Array, f32)>) {
        match self {
            Self::Plain => (&[], None),
            Self::Ip { tokens, scale } => (&[], Some((tokens, *scale))),
            Self::Control(contexts) => (contexts.as_slice(), None),
        }
    }
}

/// The seed-independent init (F-068): img2img / inpaint init latents are the f32 VAE encode and
/// the inpaint mask its preprocess — computed once, not per output.
enum SdxlInitLatents {
    /// Start from the seeded prior.
    Noise,
    /// img2img (or ControlNet-on-img2img) from the encoded Reference.
    Image {
        x0: mlx_rs::Array,
        strength: Option<f32>,
    },
    /// Masked inpaint over the encoded Reference.
    Inpaint {
        x0: mlx_rs::Array,
        mask: mlx_rs::Array,
        strength: Option<f32>,
    },
}

impl SdxlInitLatents {
    /// The raw requested strength of an init-image start, `None` for a prior start.
    fn requested_strength(&self) -> Option<Option<f32>> {
        match self {
            Self::Noise => None,
            Self::Image { strength, .. } | Self::Inpaint { strength, .. } => Some(*strength),
        }
    }
}

/// Stage 1 (sc-27336 `prepare_conditioning`): the ControlNet contexts, the request mode and the IP
/// tokens — all seed-independent, built once against the heavy bundle.
fn prepare_conditioning<'r, 'h>(
    request: &SdxlRequest<'r>,
    heavy_owned: &SdxlHeavyOwned,
    heavy: &SdxlHeavy<'h>,
) -> Result<(SdxlMode<'r>, SdxlGuidance<'h>)> {
    let req = request.req;
    let cfg_on = request.sampling.guidance() > 1.0;
    // Build the ControlNet contexts once (seed-independent): preprocess each control image to
    // [0,1] NHWC and CFG-batch it to match the U-Net input, paired by order with a loaded branch.
    let mut control_ctxs: Vec<ControlContext> = Vec::with_capacity(request.control_reqs.len());
    for ((image, scale), cn) in request.control_reqs.iter().zip(heavy.controls) {
        let img = preprocess_control_image(image, req.width, req.height)?;
        let img = if cfg_on {
            concatenate_axis(&[&img, &img], 0)?
        } else {
            img
        };
        control_ctxs.push(ControlContext {
            controlnet: cn,
            // Precompute the step-invariant conditioning embedding once per run (F-069).
            cond_embed: cn.embed_cond(&img)?,
            scale: *scale,
        });
    }

    // IP-Adapter (sc-3059): when the model carries IP weights and a Reference is present (no
    // mask/control/accel), the Reference is the image prompt (txt2img + IP), NOT an img2img init.
    // The IP scale rides the Reference `strength` field (default 0.6). Tokens are seed-independent
    // → built once, CFG-batched with a zeros uncond row so the negative pass gets no IP signal.
    let mode = SdxlMode::resolve(
        heavy_owned.has_ip_adapter()?,
        request.reference,
        request.mask,
        request.control_reqs.len(),
        request.sampling.route().is_acceleration(),
    );
    let guidance = match mode {
        SdxlMode::Ip { image, scale } => SdxlGuidance::Ip {
            tokens: heavy_owned.ip_tokens(
                image,
                cfg_on,
                sheds_ip_image_encoder(request.stage_residency),
            )?,
            scale,
        },
        SdxlMode::Control { .. } => SdxlGuidance::Control(control_ctxs),
        SdxlMode::Txt2Img | SdxlMode::Img2Img { .. } | SdxlMode::Inpaint { .. } => {
            SdxlGuidance::Plain
        }
    };
    Ok((mode, guidance))
}

/// The seed-independent half of stage 2 (`build_init`): encode the init image (sc-26270: the
/// bounded-decode rung's tiling bounds the encode too) and preprocess the inpaint mask.
fn build_init_latents(
    request: &SdxlRequest<'_>,
    vae: &Autoencoder,
    mode: SdxlMode<'_>,
) -> Result<SdxlInitLatents> {
    let req = request.req;
    let encode = |image: &Image| {
        encode_init_latents_tiled(
            vae,
            image,
            req.width,
            req.height,
            request.decode_tiling.as_ref(),
            Some(&req.cancel),
        )
    };
    Ok(match mode {
        SdxlMode::Img2Img { image, strength }
        | SdxlMode::Control {
            init: Some((image, strength)),
        } => SdxlInitLatents::Image {
            x0: encode(image)?,
            strength,
        },
        SdxlMode::Inpaint {
            image,
            strength,
            mask,
        } => {
            let x0 = encode(image)?;
            SdxlInitLatents::Inpaint {
                x0,
                mask: preprocess_mask(mask, req.width, req.height)?,
                strength,
            }
        }
        SdxlMode::Txt2Img | SdxlMode::Ip { .. } | SdxlMode::Control { init: None } => {
            SdxlInitLatents::Noise
        }
    })
}

/// Stages 2–3 for one SDXL output, over the request's prepared conditioning (sc-27336).
struct SdxlRenderPlan<'a> {
    request: &'a SdxlRequest<'a>,
    unet: &'a UNet2DConditionModel,
    sampler: &'a EulerSampler,
    alpha_schedule: &'a AlphaSchedule,
    guidance: &'a SdxlGuidance<'a>,
    init: &'a SdxlInitLatents,
    conditioning: &'a mlx_rs::Array,
    pooled: &'a mlx_rs::Array,
    time_ids: &'a mlx_rs::Array,
    latent_shape: [i32; 4],
    vp_plan: Option<VpCapturePlan>,
}

/// What an SDXL output's start keeps alive until its final latent has been collected — the same
/// locals the historical inline loop held to the end of each iteration.
enum SdxlInit<'a> {
    /// Curated k-diffusion (epic 7114, sc-7121): the model sampling, the run schedule and the drawn
    /// noise.
    Curated {
        ms: DiscreteModelSampling,
        run_sigmas: Vec<f32>,
        _noise: mlx_rs::Array,
    },
    /// The ancestral / acceleration sampler.
    Ancestral {
        sampler: Box<dyn DiffusionSampler + 'a>,
    },
    /// The masked-inpaint ancestral sampler and its per-step blend.
    Inpaint {
        sampler: Box<dyn DiffusionSampler + 'a>,
        blend: InpaintBlend<'a>,
    },
}

impl<'a> crate::stages::FamilyStages for SdxlRenderPlan<'a> {
    type Start = mlx_rs::Array;
    type Retained = SdxlInit<'a>;

    fn build_init(&self, seed: u64) -> Result<(mlx_rs::Array, SdxlInit<'a>)> {
        if self.request.sampling.route().is_curated() {
            self.build_curated_init()
        } else {
            self.build_ancestral_init(seed)
        }
    }

    fn denoise(
        &self,
        latents: mlx_rs::Array,
        init: &SdxlInit<'a>,
        seed: u64,
        on_progress: &mut dyn FnMut(Progress),
    ) -> Result<mlx_rs::Array> {
        match init {
            SdxlInit::Curated { ms, run_sigmas, .. } => {
                self.denoise_curated(ms, run_sigmas, latents, seed, on_progress)
            }
            SdxlInit::Ancestral { sampler, .. } => {
                self.denoise_ancestral(sampler.as_ref(), None, latents, on_progress)
            }
            SdxlInit::Inpaint { sampler, blend } => {
                self.denoise_ancestral(sampler.as_ref(), Some(blend), latents, on_progress)
            }
        }
    }
}

impl<'a> SdxlRenderPlan<'a> {
    /// Curated unified-sampler path (epic 7114, sc-7121): k-diffusion VE-σ sampling over a
    /// `DiscreteModelSampling`, additive alongside the bespoke ancestral default. The latents live
    /// in raw σ-space; the curated solver + scheduler are selected per request. Supports
    /// txt2img / img2img / ControlNet / IP-Adapter (inpaint is refused by `validate_image_roles`).
    fn build_curated_init(&self) -> Result<(mlx_rs::Array, SdxlInit<'a>)> {
        let steps = self.request.sampling.steps();
        let ms = DiscreteModelSampling::sdxl(self.alpha_schedule);
        let full_sigmas = self
            .request
            .sampling
            .curated_full_sigmas(self.alpha_schedule)?;
        let noise = mlx_rs::random::normal::<f32>(&self.latent_shape, None, None, None)?;
        // Raw k-diffusion σ-space init: txt2img `ε·σ_max`; img2img runs the strength-tail of the
        // schedule, seeded `x₀ + ε·σ_start` (diffusers EulerDiscrete add_noise). A strength that
        // rounds to 0 effective steps leaves the schedule at `[0.0]` → the init is returned.
        let (run_sigmas, init) = match self.init {
            SdxlInitLatents::Image { x0, strength }
            | SdxlInitLatents::Inpaint { x0, strength, .. } => {
                let strength = resolve_strength(*strength, DEFAULT_STRENGTH);
                let rs = curated_strength_schedule(&full_sigmas, steps, strength);
                let init = add(x0, &multiply(&noise, scalar(rs[0]))?)?;
                (rs, init)
            }
            SdxlInitLatents::Noise => {
                let init = multiply(&noise, scalar(full_sigmas[0]))?;
                (full_sigmas, init)
            }
        };
        Ok((
            init,
            SdxlInit::Curated {
                ms,
                run_sigmas,
                _noise: noise,
            },
        ))
    }

    /// Build the run's ancestral (or acceleration) sampler + its seeded init latents. The denoise
    /// loop is driven entirely by the sampler's own schedule (`sampler.num_steps()`), so the trait
    /// owns the per-step timestep, the input scaling, and the step math.
    fn build_ancestral_init(&self, seed: u64) -> Result<(mlx_rs::Array, SdxlInit<'a>)> {
        let steps = self.request.sampling.steps();
        let max_time = self.sampler.max_time();
        if self.request.sampling.route().is_acceleration() {
            // Few-step acceleration (txt2img): unit-noise prior scaled into the sampler's space.
            let s = self
                .request
                .sampling
                .build_acceleration_sampler(self.alpha_schedule, seed)?;
            let noise = mlx_rs::random::normal::<f32>(&self.latent_shape, None, None, None)?;
            let lat = s.scale_initial_noise(&noise)?;
            return Ok((lat, SdxlInit::Ancestral { sampler: s }));
        }
        Ok(match self.init {
            SdxlInitLatents::Inpaint { x0, mask, strength } => {
                // Masked inpaint (sc-3057): same ancestral img2img start, but keep the FIXED prior
                // noise so the per-step blend can pin the black (keep) region to the init noised to
                // each step's σ. Default strength 0.85 (the worker's inpaint default).
                let strength = resolve_strength(*strength, INPAINT_DEFAULT_STRENGTH);
                let (eff, start_step) = ancestral_strength_schedule(steps, max_time, strength);
                let noise = mlx_rs::random::normal::<f32>(&self.latent_shape, None, None, None)?;
                let x_t = self.sampler.add_noise_with(x0, &noise, start_step)?;
                // The kept region is noised to each step's "next" time `t_prev` (schedule[i].1).
                let t_prev: Vec<f32> = self
                    .sampler
                    .timesteps(eff, start_step)?
                    .into_iter()
                    .map(|(_, tp)| tp)
                    .collect();
                let blend =
                    InpaintBlend::new(self.sampler, mask.clone(), x0.clone(), noise, t_prev);
                (
                    x_t,
                    SdxlInit::Inpaint {
                        sampler: Box::new(AncestralEuler::new(self.sampler, eff, start_step)?),
                        blend,
                    },
                )
            }
            SdxlInitLatents::Image { x0, strength } => {
                // img2img (ancestral; the vendored `generate_latents_from_image`): start at
                // `max_time·strength`, run `int(steps·strength)` steps — NO min-1 floor (strength ≤
                // 1/steps ⇒ 0 steps ⇒ init returned unchanged, dodging the σ=0 ancestral `σ_up` 0/0
                // → NaN).
                let strength = resolve_strength(*strength, DEFAULT_STRENGTH);
                let (eff, start_step) = ancestral_strength_schedule(steps, max_time, strength);
                let x_t = self.sampler.add_noise(x0, start_step)?;
                // PiD from_ldm early-stop (sc-8049): truncate to the VP-capture `keep` steps; the
                // stored ancestral latent is already the VP frame, so it is handed to PiD as-is.
                // Clean path (`vp_plan = None`) keeps the full schedule.
                let sampler = AncestralEuler::new(self.sampler, eff, start_step)?;
                (x_t, self.ancestral(sampler))
            }
            SdxlInitLatents::Noise => {
                // txt2img (ancestral): seeded prior.
                let prior = self.sampler.sample_prior(&self.latent_shape)?;
                // PiD from_ldm early-stop (sc-8049): truncate to the VP-capture `keep` steps (see
                // the img2img arm); clean path (`vp_plan = None`) keeps the full schedule.
                let sampler = AncestralEuler::new(self.sampler, steps, max_time)?;
                (prior, self.ancestral(sampler))
            }
        })
    }

    /// An ancestral sampler truncated to the VP-capture `keep` steps when `from_ldm` asks for it.
    fn ancestral(&self, sampler: AncestralEuler<'a>) -> SdxlInit<'a> {
        SdxlInit::Ancestral {
            sampler: Box::new(match &self.vp_plan {
                Some(p) => sampler.truncate_to(p.keep - 1),
                None => sampler,
            }),
        }
    }

    fn denoise_curated(
        &self,
        ms: &DiscreteModelSampling,
        run_sigmas: &[f32],
        init: mlx_rs::Array,
        seed: u64,
        on_progress: &mut dyn FnMut(Progress),
    ) -> Result<mlx_rs::Array> {
        let req = self.request.req;
        let sampling = &self.request.sampling;
        // PiD from_ldm early-stop (sc-8049): truncate the curated k-diffusion schedule to the
        // VP-capture `keep` nodes so the solver stops at the achieved degrade σ; the clean path
        // (`vp_plan = None`) runs the full schedule byte-identically.
        let keep_sigmas: &[f32] = match &self.vp_plan {
            Some(p) => &run_sigmas[..p.keep],
            None => run_sigmas,
        };
        let (controls, ip) = self.guidance.curated_inputs();
        // CFG++ (sc-8256): opt-in via `guidance_method == "cfg_pp"`, only with a CFG++-compatible
        // base solver (euler/ddim/dpmpp_2m) and an active guidance gap (`cfg > 1`). Anything else —
        // including `cfg_pp` on an incompatible sampler — falls back to the plain curated path (N3,
        // never a hard-fail), so the default is byte-untouched.
        let latents = if matches!(sampling.route(), crate::sampling::SamplingRoute::CfgPp(_)) {
            denoise_cfgpp_with_preview(
                self.unet,
                Some(sampling.sampler_name()),
                ms,
                keep_sigmas,
                init,
                self.conditioning,
                self.pooled,
                self.time_ids,
                sampling.guidance(),
                &req.cancel,
                on_progress,
                &req.preview,
                controls,
                ip,
                None,
                self.request.forward_plan,
            )?
        } else {
            denoise_curated_with_preview(
                self.unet,
                Some(sampling.sampler_name()),
                ms,
                keep_sigmas,
                init,
                self.conditioning,
                self.pooled,
                self.time_ids,
                sampling.guidance(),
                seed,
                &req.cancel,
                on_progress,
                &req.preview,
                controls,
                ip,
                None,
                self.request.forward_plan,
            )?
        };
        // Curated latents live in RAW VE σ-space (`x0+σ·ε`); an early-stop leaves x_k at σ>0, so map
        // it into the student's VP frame with the plan's rescale (`1/√(1+σ²)`) before decode. The
        // clean path (`vp_plan = None`) leaves it byte-identical. (sc-8049)
        crate::stages::rescale_ve_capture(latents, self.vp_plan.as_ref())
    }

    fn denoise_ancestral(
        &self,
        sampler: &dyn DiffusionSampler,
        blend: Option<&InpaintBlend>,
        latents: mlx_rs::Array,
        on_progress: &mut dyn FnMut(Progress),
    ) -> Result<mlx_rs::Array> {
        let req = self.request.req;
        let cfg = self.request.sampling.guidance();
        let d = Denoiser::with_plan(self.unet, sampler, self.request.forward_plan);
        if let Some(b) = blend {
            return denoise_inpaint_with_preview(
                &d,
                latents,
                self.conditioning,
                self.pooled,
                self.time_ids,
                cfg,
                &req.cancel,
                on_progress,
                &req.preview,
                b,
            );
        }
        match self.guidance {
            SdxlGuidance::Ip { tokens, scale } => denoise_ip_with_preview(
                &d,
                latents,
                self.conditioning,
                self.pooled,
                self.time_ids,
                cfg,
                &req.cancel,
                on_progress,
                &req.preview,
                tokens,
                *scale,
            ),
            SdxlGuidance::Control(contexts) => denoise_multi_control_with_preview(
                &d,
                latents,
                self.conditioning,
                self.pooled,
                self.time_ids,
                cfg,
                &req.cancel,
                on_progress,
                &req.preview,
                contexts,
            ),
            SdxlGuidance::Plain => denoise_with_preview(
                &d,
                latents,
                self.conditioning,
                self.pooled,
                self.time_ids,
                cfg,
                &req.cancel,
                on_progress,
                &req.preview,
            ),
        }
    }
}

/// Phase C (stage 4): decode every retained final latent, in order, through the VAE (or the PiD
/// decoder minted in phase B), offering each to the correctness-only observer first.
fn decode_phase(
    view: &SdxlDecodeView<'_>,
    (finals, pid_decoder): SdxlDenoised,
    final_latent_observer: &mut Option<FinalLatentObserver<'_>>,
    decode_tiling: Option<&mlx_gen::tiling::TilingConfig>,
    cancel: &mlx_gen::CancelFlag,
    on_progress: &mut dyn FnMut(Progress),
) -> Result<GenerationOutput> {
    let pid_ref = pid_decoder.as_ref().map(|d| d as &dyn LatentDecoder);
    let mut images = Vec::with_capacity(finals.len());
    for (output, latents) in finals.iter().enumerate() {
        mlx_gen::ios_quantum::diagnostic_metadata(
            "sdxl",
            "serial-decode-start",
            &[
                ("schema", 1),
                ("output", output as u64),
                ("retained_latents", finals.len() as u64),
                (
                    "retained_latent_bytes",
                    finals.iter().map(mlx_gen::ios_quantum::bytes).sum(),
                ),
                ("completed_images", images.len() as u64),
            ],
        );
        if let Some(observer) = final_latent_observer.as_deref_mut() {
            observer(view.vae, latents, pid_ref)?;
        }
        mlx_gen::diagnostics::record_phase_boundary(
            mlx_gen::diagnostics::BenchmarkPhaseBoundary::DecodeStart,
        );
        images.push(crate::stages::decode_output(
            view.vae,
            latents,
            pid_ref,
            decode_tiling,
            cancel,
            on_progress,
        )?);
    }
    Ok(GenerationOutput::Images(images))
}

impl Sdxl {
    /// Extract the single img2img init image + its strength from the request's conditioning (the
    /// per-reference strength wins over `req.strength`). SDXL img2img conditions on exactly one init
    /// image, so more than one `Reference` is an error.
    fn resolve_reference<'a>(
        &self,
        req: &'a GenerationRequest,
    ) -> Result<Option<(&'a Image, Option<f32>)>> {
        reference_inputs(req)
    }

    /// Extract the single inpaint mask from the request's conditioning (sc-3057). White = repaint,
    /// black = keep. SDXL supports one mask; more than one is an error.
    fn resolve_mask<'a>(&self, req: &'a GenerationRequest) -> Result<Option<&'a Image>> {
        mask_input(req)
    }

    /// Collect the ControlNet control images + `conditioning_scale`s (sc-3058; MultiControlNet
    /// sc-3378), in request order. Each pairs with a loaded control branch (`spec.control` +
    /// `spec.extra_controls`); the count must match (validated in `generate`). A single `Control` is
    /// the common case; more than one runs as MultiControlNet (residuals summed).
    fn resolve_control<'a>(&self, req: &'a GenerationRequest) -> Result<Vec<(&'a Image, f32)>> {
        control_inputs(req)
    }
}

/// SDXL's VAE produces a `/8` latent and the three-block U-Net applies two stride-2 downsamplers
/// before mirroring them through exact skip concatenations. Each image axis must therefore be a
/// multiple of `8 * 2² = 32`; accepting only the VAE stride can produce an odd intermediate whose
/// upsampled extent no longer matches its skip tensor. `validate_request` enforces this structural
/// multiple before production denoise.
pub const SIZE_MULTIPLE: u32 = 32;

/// Capability-driven request validation, factored out so it can be unit-tested without loaded
/// weights. Rejects unsupported guidance / negative prompt / conditioning / size / count.
pub(crate) fn validate_request(caps: &Capabilities, req: &GenerationRequest) -> Result<()> {
    // Shared capability floor (F-022): count/steps range, size range, negative_prompt/guidance/
    // true_cfg support gating + finiteness, sampler/scheduler/guidance_method membership, and accepted
    // conditioning kinds. Delegating to core (like Kolors, F-132) restores the `true_cfg` and
    // `guidance_method` checks this hand-rolled copy had dropped — a `cfg_pp` typo in `guidance_method`
    // previously slipped through and silently rendered plain CFG. `steps == Some(0)` is now the floor's
    // job too. The `?` keeps the typed `Error::Unsupported` for capability gaps.
    caps.validate_request(MODEL_ID, req)?;

    // SDXL-specific checks layered on top of the shared floor:
    if req.prompt.is_empty() {
        return Err(Error::Msg("sdxl: prompt must not be empty".into()));
    }
    // The /8 VAE plus two exact U-Net downsample/upsample joins require SIZE_MULTIPLE.
    if !req.width.is_multiple_of(SIZE_MULTIPLE) || !req.height.is_multiple_of(SIZE_MULTIPLE) {
        return Err(Error::Msg(format!(
            "sdxl: width/height must be multiples of {SIZE_MULTIPLE} (got {}x{})",
            req.width, req.height
        )));
    }
    Ok(())
}

// The registration constant bridges the crate's rich `Result` into backend-neutral
// `gen_core::Result`.
pub(crate) fn component_footprint(
    spec: &mlx_gen::LoadSpec,
) -> mlx_gen::gen_core::Result<mlx_gen::PerComponentBytes> {
    let mut footprint = match &spec.weights {
        WeightsSource::Dir(root) => snapshot_component_footprint(root),
        WeightsSource::File(file) => fused_ldm_component_footprint(file, spec.quantize)?,
    };
    // sc-26268: a staged fp16-fix VAE replaces the in-snapshot (or fused) VAE and is resident at
    // f16 — price the file the load opens at the width it materializes, like every other component.
    // A staged component the contract cannot resolve or read is an error, not a silent fallback to
    // pricing the f32 VAE it replaces: the load would fail on the same file anyway.
    if let Some(source) = spec.components.get(loader::VAE_FP16_FIX_COMPONENT) {
        use mlx_gen::asset_facts::{projected_safetensors_bytes, ResidentProjection};
        let file = loader::resolve_vae_fp16_fix_file(source)?;
        footprint.vae = projected_safetensors_bytes(file, |_| ResidentProjection::Bfloat16)
            .map_err(Error::from)?;
    }
    Ok(footprint)
}

/// Header-only resident projection for a diffusers **snapshot tree** — the directory twin of
/// the fused-checkpoint projection, and for the same two reasons (SC-22667).
///
/// `pub` for `mlx-gen-instantid` (SC-22667): InstantID layers its IdentityNet and face IP-Adapter
/// onto a stock SDXL base loaded through this crate's own `load_unet_dtype` / `load_vae`, so it
/// needs the same per-component resolution and dtype projection the resident load performs rather
/// than a second derivation of the rule.
///
/// A plain `.safetensors` sum over `unet/`, `text_encoder*/` and `vae/` was wrong twice over:
///
/// * **The VAE was under-priced by exactly 2x, at every tier.** `loader::load_vae` does
///   `w.cast_all(Dtype::Float32)` unconditionally — the SDXL VAE is fp16-unstable — while every
///   SceneWorks SDXL-family tier ships only `diffusion_pytorch_model.fp16.safetensors`.
///   `loader::resolve_vae_weight_file`'s own doc has named this since sc-15839, down to the
///   `ResidentProjection::Float32` remedy; nothing had ever called it from a contract. The fused
///   branch below already prices its VAE tensors at `materialized_bytes(4)`, so the two snapshot
///   shapes disagreed with each other.
/// * **Side-by-side dtype variants were counted twice.** A component directory holding both
///   `<stem>.safetensors` and `<stem>.fp16.safetensors` contributed both, while
///   `loader::resolve_weight_file` opens exactly one of them. gen-core's own
///   `PerComponentBytes` doc names this hazard.
///
/// The two errors point in opposite directions and land on different components, so they never
/// cancelled. Each component is now sized from the exact file the resident load resolves, at the
/// width that load materializes it in: the VAE f32, the U-Net and the two CLIP towers 16-bit
/// (`DTYPE`), which is the same width the fused branch charges them. The shared projection detects
/// an existing affine pack by its `.scales` companion and leaves those tensors at stored width, so
/// a turnkey packed snapshot is not re-inflated to dense.
///
/// A component whose file cannot be resolved or read keeps its previous directory sum. This runs at
/// contract time, ahead of any deferred component load, so an incomplete tree must not become a
/// contract-time refusal.
pub fn snapshot_component_footprint(root: &Path) -> mlx_gen::PerComponentBytes {
    use mlx_gen::asset_facts::{projected_safetensors_bytes, ResidentProjection};

    let mut footprint = mlx_gen::PerComponentBytes::from_root_subdirs(
        root,
        &["text_encoder", "text_encoder_2"],
        &["unet"],
        &["vae"],
    );
    let projected =
        |file: Option<std::path::PathBuf>, projection: ResidentProjection| -> Option<u64> {
            projected_safetensors_bytes(file?, move |_| projection).ok()
        };
    let half = ResidentProjection::Bfloat16;

    let clip_l = projected(
        loader::resolve_weight_file(root, "text_encoder", "model", DTYPE).ok(),
        half,
    );
    let clip_g = projected(
        loader::resolve_weight_file(root, "text_encoder_2", "model", DTYPE).ok(),
        half,
    );
    if let (Some(clip_l), Some(clip_g)) = (clip_l, clip_g) {
        footprint.text_encoder = clip_l.saturating_add(clip_g);
    }
    if let Some(unet) = projected(loader::resolve_unet_weight_file(root, DTYPE).ok(), half) {
        footprint.dit = unet;
    }
    if let Some(vae) = projected(
        loader::resolve_vae_weight_file(root).ok(),
        ResidentProjection::Float32,
    ) {
        footprint.vae = vae;
    }
    footprint
}

/// Header-only resident projection for a fused LDM/A1111 checkpoint.  Unlike a snapshot tree, a
/// fused file must first be classified by the exact shared LDM key remapper.  Text/UNet values are
/// retained at fp16 (or at the selected affine Q4/Q8 tier); VAE values are retained at f32.  Keys the
/// executable splitter ignores are omitted here too.
fn fused_ldm_component_footprint(
    file: &Path,
    quantize: Option<Quant>,
) -> mlx_gen::gen_core::Result<mlx_gen::PerComponentBytes> {
    use gen_core_family::sdxl_ldm::{remap_sdxl_ldm_key, SdxlComponent, TensorRemap};

    let headers = mlx_gen::gen_core::safetensors_path_tensor_headers(file)?;
    if headers.is_empty() {
        return Err(mlx_gen::gen_core::Error::Msg(
            "sdxl fused checkpoint contains no tensor headers".to_owned(),
        ));
    }
    let mut out = mlx_gen::PerComponentBytes::default();
    for tensor in headers {
        let Some(remap) = remap_sdxl_ldm_key(&tensor.name) else {
            continue;
        };
        let (component, targets, source_shape_is_pack_shape) = match remap {
            TensorRemap::Rename(component, target) => (component, vec![target], true),
            // Row-wise Q/K/V packing is additive, so packing the fused row stack has the same byte
            // total as packing the three remapped projections independently.
            TensorRemap::SplitQkv(component, targets) => {
                (component, targets.into_iter().collect(), true)
            }
            // Transpose/squeeze preserve element count but can change which axis is grouped. Keep
            // them dense in the estimate rather than risk understating the exact packed footprint.
            TensorRemap::Transpose(component, target) | TensorRemap::Squeeze(component, target) => {
                (component, vec![target], false)
            }
        };
        let bytes = match component {
            SdxlComponent::Vae => tensor.materialized_bytes(4)?,
            SdxlComponent::Unet | SdxlComponent::ClipL | SdxlComponent::ClipBigG => {
                let target_is_packable = source_shape_is_pack_shape
                    && tensor.shape.len() == 2
                    && targets.iter().all(|target| {
                        let Some(base) = target.strip_suffix(".weight") else {
                            return false;
                        };
                        component == SdxlComponent::Unet
                            || (!base.ends_with(".token_embedding")
                                && !base.ends_with(".position_embedding"))
                    });
                match (quantize, target_is_packable) {
                    (Some(quant), true) => {
                        let group = crate::quant::GROUP_SIZE as usize;
                        let [rows, columns] = tensor.shape.as_slice() else {
                            unreachable!("target_is_packable requires a weight target; shape is checked below")
                        };
                        if *columns < group || *columns % group != 0 {
                            tensor.materialized_bytes(2)?
                        } else {
                            let rows = u64::try_from(*rows).map_err(|_| {
                                mlx_gen::gen_core::Error::Msg(
                                    "sdxl fused projection row count overflow".to_owned(),
                                )
                            })?;
                            let columns = u64::try_from(*columns).map_err(|_| {
                                mlx_gen::gen_core::Error::Msg(
                                    "sdxl fused projection column count overflow".to_owned(),
                                )
                            })?;
                            let codes = rows
                                .checked_mul(columns)
                                .and_then(|elements| elements.checked_mul(quant.bits() as u64))
                                .map(|bits| bits / 8)
                                .ok_or_else(|| {
                                    mlx_gen::gen_core::Error::Msg(
                                        "sdxl fused quantized code size overflow".to_owned(),
                                    )
                                })?;
                            let tables = rows
                                .checked_mul(columns / group as u64)
                                .and_then(|entries| entries.checked_mul(4))
                                .ok_or_else(|| {
                                    mlx_gen::gen_core::Error::Msg(
                                        "sdxl fused quantization table size overflow".to_owned(),
                                    )
                                })?;
                            codes.checked_add(tables).ok_or_else(|| {
                                mlx_gen::gen_core::Error::Msg(
                                    "sdxl fused quantized resident size overflow".to_owned(),
                                )
                            })?
                        }
                    }
                    _ => tensor.materialized_bytes(2)?,
                }
            }
        };
        let slot = match component {
            SdxlComponent::ClipL | SdxlComponent::ClipBigG => &mut out.text_encoder,
            SdxlComponent::Unet => &mut out.dit,
            SdxlComponent::Vae => &mut out.vae,
        };
        *slot = slot.checked_add(bytes).ok_or_else(|| {
            mlx_gen::gen_core::Error::Msg("sdxl fused component byte sum overflow".to_owned())
        })?;
    }
    if out.text_encoder == 0 || out.dit == 0 || out.vae == 0 {
        return Err(mlx_gen::gen_core::Error::Msg(
            "sdxl fused checkpoint is missing a text encoder, UNet, or VAE component".to_owned(),
        ));
    }
    Ok(out)
}

mlx_gen::register_generators! {
    pub(crate) const REGISTRATION = descriptor => load;
    footprint = component_footprint
}

/// The weights-free memory-strategy registration (SC-15525) — the shared registry conformance
/// suite's entry point into this provider's declaration.
pub const MEMORY_REGISTRATION: mlx_gen::gen_core::MemoryRegistration =
    mlx_gen::gen_core::MemoryRegistration {
        provider_id: MODEL_ID,
        contract: |spec| crate::memory_strategy::memory_strategy_contract(MODEL_ID, spec),
        safety_check: crate::memory_strategy::safety_check,
    };

/// The weights-free **behavioral** registration: valid fixtures plus a request scope, so the shared
/// suite can drive every declared rung's lifecycle without loading 6.6 GB of weights.
pub const MEMORY_BEHAVIOR_REGISTRATION: mlx_gen::gen_core::MemoryBehaviorRegistration =
    mlx_gen::gen_core::MemoryBehaviorRegistration {
        provider_id: MODEL_ID,
        valid_fixtures: crate::memory_strategy::registered_valid_fixture,
        begin_request: |spec, contract, context| {
            crate::memory_strategy::registered_begin_request(MODEL_ID, spec, contract, context)
        },
    };

/// sc-16195 Apple-Silicon warm sweep: q8 and dense both peaked at 14.039 GiB at 1024².
/// The 14.05 GiB family ceiling is deliberately upward-rounded.
pub const ACTIVATION_MEMORY_REGISTRATION: mlx_gen::gen_core::ActivationMemoryRegistration =
    mlx_gen::gen_core::ActivationMemoryRegistration {
        provider_id: MODEL_ID,
        anchor: ActivationMemoryAnchor {
            bytes_1024: 15_086_072_628,
        },
    };

#[cfg(test)]
mod tests {
    use super::*;

    /// sc-27336: the typed SDXL mode matrix — which role the Reference / Mask / Control inputs
    /// play — weights-free.
    #[test]
    fn mode_matrix_assigns_each_image_role_once() {
        let image = Image::default();
        let mask = Image::default();
        let reference = Some((&image, Some(0.4)));
        assert!(matches!(
            SdxlMode::resolve(true, None, None, 0, false),
            SdxlMode::Txt2Img
        ));
        assert!(matches!(
            SdxlMode::resolve(false, None, None, 2, true),
            SdxlMode::Control { init: None }
        ));
        // Loaded IP weights + a bare Reference: the image prompt, its strength the IP scale.
        assert!(matches!(
            SdxlMode::resolve(true, reference, None, 0, false),
            SdxlMode::Ip { scale, .. } if scale == 0.4
        ));
        assert!(matches!(
            SdxlMode::resolve(true, Some((&image, None)), None, 0, false),
            SdxlMode::Ip { scale, .. } if scale == IP_DEFAULT_SCALE
        ));
        // Without IP weights the Reference is the init.
        assert!(matches!(
            SdxlMode::resolve(false, reference, None, 0, false),
            SdxlMode::Img2Img { strength: Some(s), .. } if s == 0.4
        ));
        // A mask or a control owns the request even with IP weights loaded.
        assert!(matches!(
            SdxlMode::resolve(true, reference, Some(&mask), 0, false),
            SdxlMode::Inpaint { strength: Some(s), .. } if s == 0.4
        ));
        assert!(matches!(
            SdxlMode::resolve(true, reference, None, 1, false),
            SdxlMode::Control { init: Some(_) }
        ));
        // Acceleration is txt2img-only: its Reference is refused by `validate_image_roles` and is
        // never an init here.
        assert!(validate_image_roles("lightning", true, false, true, false).is_err());
        assert!(matches!(
            SdxlMode::resolve(false, reference, None, 0, true),
            SdxlMode::Txt2Img
        ));
        // A Mask needs its Reference, and never combines with Control.
        assert!(validate_image_roles("euler_ancestral", false, false, false, true).is_err());
        assert!(validate_control_roles("euler_ancestral", None, 1, 1, true).is_err());
    }

    /// Feature-end review (SC-22667, E1): the snapshot footprint must price the exact file each
    /// component load resolves, at the width that load materializes it in.
    ///
    /// Two independent errors, in opposite directions, on different components:
    /// `loader::load_vae` `cast_all(Dtype::Float32)`s unconditionally over the fp16 variant every
    /// SceneWorks tier ships (2x under, named by `resolve_vae_weight_file`'s own doc since
    /// sc-15839), while a directory sum counted BOTH `<stem>.safetensors` and
    /// `<stem>.fp16.safetensors` when a snapshot carries the pair (over, on the U-Net).
    ///
    /// Mutation that fails this: restoring `PerComponentBytes::from_spec_subdirs` for the `Dir`
    /// branch — `vae` halves to the stored 100 and `dit` inflates to the sum of both U-Net
    /// variants plus their headers.
    #[test]
    fn snapshot_footprint_resolves_one_file_per_component_at_its_materialized_width() {
        /// One-tensor safetensors file; `dtype` must agree with `width` because the shared header
        /// reader refuses a payload length that is not `elements * dtype.size()`.
        fn write_tensor(path: &std::path::Path, dtype: &str, elements: usize, width: usize) {
            let data_bytes = elements * width;
            let mut header = format!(
                "{{\"weight\":{{\"dtype\":\"{dtype}\",\"shape\":[{elements}],\"data_offsets\":[0,{data_bytes}]}}}}"
            )
            .into_bytes();
            while !header.len().is_multiple_of(8) {
                header.push(b' ');
            }
            let mut bytes = (header.len() as u64).to_le_bytes().to_vec();
            bytes.extend(header);
            bytes.resize(bytes.len() + data_bytes, 0);
            std::fs::write(path, bytes).unwrap();
        }

        let tmp = tempfile::tempdir().unwrap();
        let root = tmp.path();
        for component in ["unet", "vae", "text_encoder", "text_encoder_2"] {
            std::fs::create_dir_all(root.join(component)).unwrap();
        }
        // The U-Net ships BOTH dtype variants side by side; only the fp16 one is opened at `DTYPE`.
        write_tensor(
            &root.join("unet/diffusion_pytorch_model.fp16.safetensors"),
            "F16",
            100,
            2,
        );
        write_tensor(
            &root.join("unet/diffusion_pytorch_model.safetensors"),
            "F32",
            100,
            4,
        );
        // The VAE ships ONLY the fp16 variant — the shape every SceneWorks SDXL tier has — and is
        // upcast to f32 on load regardless.
        write_tensor(
            &root.join("vae/diffusion_pytorch_model.fp16.safetensors"),
            "F16",
            50,
            2,
        );
        write_tensor(
            &root.join("text_encoder/model.fp16.safetensors"),
            "F16",
            20,
            2,
        );
        write_tensor(
            &root.join("text_encoder_2/model.fp16.safetensors"),
            "F16",
            30,
            2,
        );

        let spec = mlx_gen::LoadSpec::new(WeightsSource::Dir(root.to_path_buf()));
        let footprint = component_footprint(&spec).unwrap();
        assert_eq!(
            footprint.dit, 200,
            "only the resolved fp16 U-Net may be counted, at 2 bytes per element"
        );
        assert_eq!(
            footprint.vae, 200,
            "the VAE is materialized f32: 50 elements at 4 bytes, not the 100 stored"
        );
        assert_eq!(footprint.text_encoder, 20 * 2 + 30 * 2);

        // A directory sum would have been strictly larger on the U-Net and exactly half on the VAE.
        let stored = mlx_gen::PerComponentBytes::from_root_subdirs(
            root,
            &["text_encoder", "text_encoder_2"],
            &["unet"],
            &["vae"],
        );
        assert!(stored.dit > footprint.dit, "{stored:?}");
        assert!(stored.vae < footprint.vae, "{stored:?}");

        // An incomplete tree keeps the previous directory sum rather than refusing at contract time.
        let bare = tempfile::tempdir().unwrap();
        let bare_spec = mlx_gen::LoadSpec::new(WeightsSource::Dir(bare.path().to_path_buf()));
        let bare_footprint = component_footprint(&bare_spec).unwrap();
        assert_eq!(bare_footprint.dit, 0);
        assert_eq!(bare_footprint.vae, 0);
        assert_eq!(bare_footprint.text_encoder, 0);

        // A staged fp16-fix VAE replaces the snapshot VAE and is priced at f16 from its own file.
        let fix = tempfile::tempdir().unwrap();
        write_tensor(
            &fix.path().join("diffusion_pytorch_model.safetensors"),
            "F32",
            40,
            4,
        );
        let staged = mlx_gen::LoadSpec::new(WeightsSource::Dir(root.to_path_buf())).with_component(
            loader::VAE_FP16_FIX_COMPONENT,
            WeightsSource::Dir(fix.path().to_path_buf()),
        );
        assert_eq!(component_footprint(&staged).unwrap().vae, 40 * 2);
        // A staged component the contract cannot resolve is an error — never a silent fallback to
        // pricing the f32 VAE it replaces.
        let empty = tempfile::tempdir().unwrap();
        let unresolvable = mlx_gen::LoadSpec::new(WeightsSource::Dir(root.to_path_buf()))
            .with_component(
                loader::VAE_FP16_FIX_COMPONENT,
                WeightsSource::Dir(empty.path().to_path_buf()),
            );
        let err = component_footprint(&unresolvable)
            .expect_err("an unresolvable staged fp16-fix VAE must not be priced as the f32 VAE")
            .to_string();
        assert!(err.contains(loader::VAE_FP16_FIX_COMPONENT), "{err}");
    }

    #[test]
    fn strength_resolution_preserves_defaults_explicit_values_and_clamping() {
        for (requested, default) in [
            (None, DEFAULT_STRENGTH),
            (None, INPAINT_DEFAULT_STRENGTH),
            (Some(0.0), DEFAULT_STRENGTH),
            (Some(0.37), DEFAULT_STRENGTH),
            (Some(1.0), DEFAULT_STRENGTH),
            (Some(-0.25), DEFAULT_STRENGTH),
            (Some(1.25), INPAINT_DEFAULT_STRENGTH),
        ] {
            let previous = requested.unwrap_or(default).clamp(0.0, 1.0);
            assert_eq!(resolve_strength(requested, default), previous);
        }
    }

    #[test]
    fn ancestral_strength_schedule_matches_previous_formula() {
        let steps = 30;
        let max_time = 999.0;
        for strength in [0.0, 0.01, 0.6, DEFAULT_STRENGTH, 0.85, 1.0] {
            let previous = ((steps as f32 * strength) as usize, max_time * strength);
            assert_eq!(
                ancestral_strength_schedule(steps, max_time, strength),
                previous
            );
        }
    }

    #[test]
    fn control_acceleration_admission_is_lightning_only_and_cfg_free() {
        // The ordinary no-ControlNet acceleration paths keep their established defaults. This
        // contract applies only to the control-aware path, where a Lightning checkpoint must never
        // be silently driven through CFG.
        let request = GenerationRequest {
            sampler: Some("lightning".into()),
            ..Default::default()
        };
        let sampling = crate::sampling::ResolvedSampling::resolve(&request).unwrap();
        assert_eq!((sampling.steps(), sampling.guidance()), (4, 1.0));
        assert!(validate_control_acceleration("euler_ancestral", Some(7.0)).is_ok());
        assert!(validate_control_acceleration("lightning", None).is_ok());
        assert!(validate_control_acceleration("lightning", Some(1.0)).is_ok());
        assert!(validate_control_acceleration("lightning", Some(0.5)).is_ok());

        let guidance = validate_control_acceleration("lightning", Some(1.01))
            .unwrap_err()
            .to_string();
        assert!(guidance.contains("guidance <= 1.0"), "got: {guidance}");

        for sampler in ["lcm", "hyper"] {
            let err = validate_control_acceleration(sampler, None)
                .unwrap_err()
                .to_string();
            assert!(err.contains("lightning"), "{sampler}: {err}");
        }
    }

    #[test]
    fn curated_strength_schedule_matches_previous_tail_selection() {
        let full_sigmas = vec![14.0, 8.0, 4.0, 2.0, 1.0, 0.0];
        let steps = 5;
        for strength in [0.0, 0.01, 0.2, 0.6, DEFAULT_STRENGTH, 1.0] {
            let effective_steps = (steps as f32 * strength) as usize;
            let run_start = full_sigmas
                .len()
                .saturating_sub(1)
                .saturating_sub(effective_steps);
            let previous = full_sigmas[run_start..].to_vec();
            assert_eq!(
                curated_strength_schedule(&full_sigmas, steps, strength),
                previous
            );
        }
    }

    /// sc-26268 AC (bullet 1): **the U-Net is not resident during a staged request's decode**,
    /// asserted with a memory probe. The production generator's heavy bundle type runs through the
    /// shared three-phase driver it now uses (`Residency::run_staged_request_scoped`); MLX's active
    /// byte count is read inside the denoise phase and again inside the decode phase. On a staged
    /// request the decode must see the tiny U-Net's whole weight set gone; on a resident (warm)
    /// request — the control — it must still be there, since a warm bundle is kept across requests
    /// by design.
    #[test]
    fn a_staged_request_releases_the_unet_before_decode() {
        use crate::test_support::{tiny_unet, tiny_unet_weights};
        use mlx_rs::memory::{clear_cache, get_active_memory};
        use std::sync::{Arc, Mutex};

        let _cpu = crate::test_device::cpu_only();
        let unet_bytes: usize = tiny_unet_weights(11)
            .unwrap()
            .into_tensors()
            .values()
            .map(mlx_rs::Array::nbytes)
            .sum();
        assert!(
            unet_bytes > 1 << 20,
            "the probe needs a U-Net it can resolve"
        );

        let heavy = || -> Result<SdxlHeavyOwned> {
            let unet = tiny_unet(11)?;
            unet.materialize_weights()?;
            let (weights, cfg) = crate::vae::tests::fixture();
            let vae = Autoencoder::from_weights(&weights, &cfg)?;
            weights.materialize()?;
            Ok(SdxlHeavyOwned {
                unet,
                controls: Vec::new(),
                ip_adapter: std::sync::Mutex::new(None),
                vae,
                pid: None,
            })
        };
        let probe = |stage_residency: bool| -> (usize, usize) {
            let residency =
                Residency::<u8, SdxlHeavyOwned>::request_scoped(|_| Ok(0_u8), move |_, _| heavy());
            clear_cache();
            let seen = Arc::new(Mutex::new((0_usize, 0_usize)));
            let (in_denoise, in_decode) = (Arc::clone(&seen), Arc::clone(&seen));
            residency
                .run_staged_request_scoped(
                    stage_residency,
                    false,
                    &mlx_gen::CancelFlag::new(),
                    false,
                    &mut |_| {},
                    |_: &u8| Ok(()),
                    |_| Ok(()),
                    move |heavy: &SdxlHeavyOwned, (), _| {
                        heavy.unet.materialize_weights()?;
                        in_denoise.lock().unwrap().0 = get_active_memory();
                        Ok(())
                    },
                    |_| Ok(()),
                    move |view: SdxlDecodeView<'_>, (), _| {
                        assert_eq!(view.vae.dtype(), mlx_rs::Dtype::Float32);
                        in_decode.lock().unwrap().1 = get_active_memory();
                        Ok(())
                    },
                )
                .unwrap();
            let seen = *seen.lock().unwrap();
            seen
        };

        let (denoise, decode) = probe(true);
        let released = denoise.saturating_sub(decode);
        assert!(
            released * 10 >= unet_bytes * 9,
            "staged decode must not hold the U-Net: active {denoise} B in denoise, {decode} B in \
             decode, released {released} B of a {unet_bytes} B U-Net"
        );
        // The control: the warm arm keeps its bundle, so the same probe sees no release. Without
        // it, a probe that read zero in every phase would pass.
        let (denoise, decode) = probe(false);
        assert!(
            decode + unet_bytes / 10 >= denoise,
            "a resident request keeps its warm bundle through decode: {denoise} B vs {decode} B"
        );
        assert!(
            denoise >= unet_bytes,
            "the probe must see the U-Net while it denoises"
        );
    }

    /// sc-26268: the descriptor declares both CFG batchings, the shared request floor admits each
    /// and refuses nothing it declares, and an unset request resolves to the historical batched
    /// mode.
    #[test]
    fn cfg_batching_is_declared_admitted_and_defaults_to_batched() {
        use mlx_gen::gen_core::{CfgBatching, GenerationMemory};
        let caps = descriptor().capabilities;
        for mode in CfgBatching::ALL {
            assert!(caps.execution.cfg_batching.accepts(mode), "{mode:?}");
            let req = GenerationRequest {
                prompt: "p".into(),
                width: 1024,
                height: 1024,
                count: 1,
                memory: Some(GenerationMemory {
                    cfg_batching: Some(mode),
                    ..Default::default()
                }),
                ..Default::default()
            };
            validate_request(&caps, &req).unwrap();
            assert_eq!(resolve_cfg_batching(&req), mode);
        }
        assert_eq!(
            resolve_cfg_batching(&GenerationRequest::default()),
            CfgBatching::Batched
        );
    }

    #[test]
    fn descriptor_is_sdxl() {
        let d = descriptor();
        assert_eq!(d.id, "sdxl");
        assert_eq!(d.family, "sdxl");
        assert_eq!(d.modality, Modality::Image);
        assert!(d.capabilities.supports_guidance);
        assert!(d.capabilities.supports_negative_prompt);
        assert!(d.capabilities.supports_preview);
    }

    #[test]
    fn registered_in_core_registry() {
        // The family catalog must expose the model registration.
        assert!(
            crate::provider_registry()
                .unwrap()
                .generators()
                .copied()
                .any(|r| (r.descriptor)().id == "sdxl"),
            "sdxl is not registered in mlx_gen's generator registry"
        );
    }

    #[test]
    fn validate_rejects_empty_prompt() {
        let caps = descriptor().capabilities;
        let req = GenerationRequest::default(); // default prompt is empty
        let err = validate_request(&caps, &req).unwrap_err().to_string();
        assert!(err.contains("empty"), "got: {err}");
    }

    /// sc-12612: `SIZE_MULTIPLE` is the pinned stride SceneWorks ties every advertised SDXL bucket to.
    /// Pin the value and mutation-check that a VAE-valid multiple of 8 which is not the full U-Net
    /// multiple is rejected with the stride error, and an on-stride size passes.
    #[test]
    fn size_multiple_is_the_pinned_stride() {
        assert_eq!(SIZE_MULTIPLE, 32);
        let caps = descriptor().capabilities;
        let off = validate_request(
            &caps,
            &GenerationRequest {
                prompt: "a fox".into(),
                width: 1000, // 125×8 — VAE-valid but not SIZE_MULTIPLE
                height: 1024,
                ..Default::default()
            },
        )
        .unwrap_err()
        .to_string();
        assert!(off.contains("multiples of 32"), "got: {off}");
        assert!(validate_request(
            &caps,
            &GenerationRequest {
                prompt: "a fox".into(),
                width: 1024,
                height: 1024,
                ..Default::default()
            }
        )
        .is_ok());
    }

    #[test]
    fn validate_rejects_explicit_zero_steps() {
        let caps = descriptor().capabilities;
        // F-073: an explicit `steps: Some(0)` would VAE-decode pure scaled noise → reject loudly.
        let zero = GenerationRequest {
            prompt: "a fox".into(),
            steps: Some(0),
            ..Default::default()
        };
        let err = validate_request(&caps, &zero).unwrap_err().to_string();
        assert!(err.contains("steps"), "got: {err}");
        // `steps: None` (use the production default) and an explicit positive count are accepted.
        let unset = GenerationRequest {
            prompt: "a fox".into(),
            ..Default::default()
        };
        assert!(validate_request(&caps, &unset).is_ok());
        let one = GenerationRequest {
            prompt: "a fox".into(),
            steps: Some(1),
            ..Default::default()
        };
        assert!(validate_request(&caps, &one).is_ok());
    }

    #[test]
    fn validate_rejects_unadvertised_guidance_method_and_true_cfg() {
        // F-022: the hand-rolled copy dropped the `guidance_method` membership + `true_cfg` gate. A
        // `cfg_pp` typo (e.g. "cfgpp") previously slipped through and silently rendered plain CFG.
        let caps = descriptor().capabilities;
        // Advertised methods are ["cfg", "cfg_pp"]; a typo must be rejected (typed Unsupported).
        let typo = GenerationRequest {
            prompt: "a fox".into(),
            guidance_method: Some("cfgpp".into()),
            ..Default::default()
        };
        let err = validate_request(&caps, &typo).unwrap_err();
        assert!(
            matches!(err, Error::Unsupported(_)),
            "cfg_pp typo should be a typed Unsupported gap, got {err:?}"
        );
        // The correct spelling passes.
        let ok = GenerationRequest {
            prompt: "a fox".into(),
            guidance_method: Some("cfg_pp".into()),
            ..Default::default()
        };
        assert!(validate_request(&caps, &ok).is_ok());
        // SDXL doesn't support true_cfg — a request must be rejected, not ignored.
        let tcfg = GenerationRequest {
            prompt: "a fox".into(),
            true_cfg: Some(4.0),
            ..Default::default()
        };
        assert!(validate_request(&caps, &tcfg).is_err());
    }

    #[test]
    fn validate_accepts_cfg_and_negative_prompt_rejects_bad_size() {
        let caps = descriptor().capabilities;
        // Real CFG + negative prompt are supported.
        let mut req = GenerationRequest {
            prompt: "a fox".into(),
            guidance: Some(7.0),
            negative_prompt: Some("blurry".into()),
            ..Default::default()
        };
        assert!(validate_request(&caps, &req).is_ok());
        // VAE-valid but U-Net-invalid size is rejected.
        req = GenerationRequest {
            prompt: "a fox".into(),
            width: 1000,
            height: 1024,
            ..Default::default()
        };
        assert!(validate_request(&caps, &req).is_err());
        // Out-of-range size is rejected.
        req = GenerationRequest {
            prompt: "a fox".into(),
            width: 256,
            height: 256,
            ..Default::default()
        };
        assert!(validate_request(&caps, &req).is_err());
    }

    #[test]
    fn validate_sampler_selection() {
        let caps = descriptor().capabilities;
        let base = GenerationRequest {
            prompt: "a fox".into(),
            ..Default::default()
        };
        // The default + every wired sampler is accepted (an unset sampler defaults to ancestral): the
        // accel profiles AND the unified curated solvers (epic 7114, sc-7121).
        assert!(validate_request(&caps, &base).is_ok());
        for ok in [
            "euler_ancestral",
            "lcm",
            "lightning",
            "hyper",
            "euler",
            "heun",
            "dpmpp_2m",
            "dpmpp_sde",
            "uni_pc",
            "ddim",
        ] {
            assert!(
                validate_request(
                    &caps,
                    &GenerationRequest {
                        sampler: Some(ok.into()),
                        ..base.clone()
                    }
                )
                .is_ok(),
                "sampler {ok:?} should be accepted"
            );
        }
        // An unknown sampler is rejected, not silently downgraded.
        for bad in ["plms", "dpm_fast", "nonsense"] {
            let err = validate_request(
                &caps,
                &GenerationRequest {
                    sampler: Some(bad.into()),
                    ..base.clone()
                },
            )
            .unwrap_err()
            .to_string();
            assert!(err.contains("unsupported sampler"), "got: {err}");
        }
    }

    #[test]
    fn validate_scheduler_selection() {
        let caps = descriptor().capabilities;
        let base = GenerationRequest {
            prompt: "a fox".into(),
            ..Default::default()
        };
        // `discrete` (the native ancestral schedule) + every curated σ scheduler is accepted (sc-7121).
        for ok in [
            "discrete",
            "normal",
            "simple",
            "karras",
            "exponential",
            "sgm_uniform",
            "beta",
            "ddim_uniform",
        ] {
            assert!(
                validate_request(
                    &caps,
                    &GenerationRequest {
                        scheduler: Some(ok.into()),
                        ..base.clone()
                    }
                )
                .is_ok(),
                "scheduler {ok:?} should be accepted"
            );
        }
        // diffusers timestep-spacing names are NOT curated scheduler names → rejected.
        for bad in ["leading", "trailing", "nonsense"] {
            let err = validate_request(
                &caps,
                &GenerationRequest {
                    scheduler: Some(bad.into()),
                    ..base.clone()
                },
            )
            .unwrap_err()
            .to_string();
            assert!(err.contains("unsupported scheduler"), "got: {err}");
        }
    }

    #[test]
    fn ordinary_load_does_not_treat_an_undeclared_file_as_a_snapshot() {
        let spec = LoadSpec::new(WeightsSource::File("/tmp/sdxl.safetensors".into()));
        let err = load(&spec).err().expect("expected an error").to_string();
        assert!(
            err.contains("ldm_tokenizer") || err.contains("snapshot directory"),
            "the file source must fail at an imported-route structural gate, not load as a snapshot: {err}"
        );
    }

    // ── F-180 (sc-11126): weight-free, default-run proof that SDXL's dispatch HONORS `offload_policy`.
    // `build_residency` points at a non-existent snapshot *directory* (so the single-file guard in
    // `resolve_root` passes) and the discriminator is deferral:
    //   * `Sequential` captures the two per-phase loaders, touches NO weights → `Ok` + `is_sequential`.
    //   * `Resident` eager-loads the CLIP text encoders from the missing dir → `Err`.
    // A dispatch that ignored `offload_policy` (always `Resident`) would eager-load under a `Sequential`
    // request and fail the first assertion. The `sequential_residency_real_weights.rs` A/B is
    // `#[ignore]`d; this is the default-run guard.
    fn missing_snapshot_spec(policy: OffloadPolicy) -> LoadSpec {
        LoadSpec::new(WeightsSource::Dir(
            "/nonexistent/sdxl-residency-test-snapshot".into(),
        ))
        .with_offload_policy(policy)
    }

    #[test]
    fn prepared_directory_change_is_rejected_by_deferred_text_loader() {
        let root = tempfile::tempdir().unwrap();
        let file = root.path().join("placeholder.safetensors");
        std::fs::write(&file, b"original").unwrap();
        let mut spec = LoadSpec::new(WeightsSource::Dir(root.path().to_owned()))
            .with_offload_policy(OffloadPolicy::Sequential);
        spec.bind_source_inventory(mlx_gen::gen_core::SourceInventory::capture(&spec).unwrap())
            .unwrap();
        let residency = build_residency(&spec).unwrap();
        std::fs::write(root.path().join("new-shard.safetensors"), b"new").unwrap();
        let result: Result<()> = residency.run(
            &mlx_gen::CancelFlag::new(),
            false,
            &mut |_| {},
            |_| Ok(()),
            |_| Ok(()),
            |_, _, _| Ok(()),
        );
        let error = result.unwrap_err().to_string();
        assert!(
            error.contains("source inventory changed"),
            "must reject at phase guard, not attempt a tensor load: {error}"
        );
    }

    #[test]
    fn build_residency_sequential_defers_all_component_loads() {
        let res = build_residency(&missing_snapshot_spec(OffloadPolicy::Sequential))
            .expect("Sequential must defer loads and not touch the (missing) snapshot dir");
        assert!(
            res.is_sequential(),
            "Sequential policy must build a Sequential (deferred) residency"
        );
    }

    #[test]
    fn build_residency_resident_eager_loads_and_fails_on_missing_snapshot() {
        let err = build_residency(&missing_snapshot_spec(OffloadPolicy::Resident))
            .err()
            .expect("Resident must eager-load and fail on a missing snapshot dir");
        let msg = err.to_string();
        assert!(
            !msg.contains("single .safetensors file") && !msg.contains("precision override"),
            "expected an eager-load failure, not the up-front guard: {msg}"
        );
    }

    // ── sc-18317: the fused-LDM route's rung-1 capability is EXECUTABLE, not merely declared.
    //
    // The defect these cover: `load_from_ldm_file` under `OffloadPolicy::Resident` used to obtain a
    // valid file pin, read through it once, drop it, and build `Residency::resident` — whose loaders
    // return errors and whose `rebuildable` flag is `false`. Nothing upstream refused a staged
    // selection on that instance (the descriptor publishes `Selectable`, the contract declares rung 1
    // `Implemented`, rung 1 has no prerequisite for `validate_selection` to check, and the shared
    // `Capabilities::validate_request` floor never inspects `stage_residency`), so the composition was
    // admitted everywhere and then refused inside `generate` by `Residency::ensure_rebuildable`.
    //
    // These run weight-free. A real fused SDXL checkpoint is ~7 GB, so the fixture is a structurally
    // valid safetensors carrying one SDXL-irrelevant tensor: `Weights::from_file` opens it, the shared
    // LDM key remapper classifies nothing, and `split_ldm_checkpoint` returns its own typed
    // "missing the … component" error. That error arriving is the proof the staged phase loader
    // re-opened the pinned fused file; the resident-only refusal arriving instead is the defect.
    // Each test mints its fixture root with `tempfile::tempdir()` and holds the `TempDir` guard for
    // its own duration: the guard IS the cleanup, including out of a panicking test (sc-17791).
    //
    /// One structurally valid safetensors file with a single F32 tensor and no SDXL/LDM keys.
    ///
    /// Deliberately valid rather than garbage: the point is to reach `split_ldm_checkpoint`'s key
    /// classification through the pin, not to probe the safetensors reader's malformed-input path.
    fn write_stub_fused_checkpoint(dir: &Path) -> std::path::PathBuf {
        let path = dir.join("stub-fused.safetensors");
        let mut header =
            br#"{"not_an_sdxl_tensor":{"dtype":"F32","shape":[2],"data_offsets":[0,8]}}"#.to_vec();
        // Pad to an 8-byte boundary so the payload starts aligned, as the format expects.
        while !header.len().is_multiple_of(8) {
            header.push(b' ');
        }
        let mut bytes = (header.len() as u64).to_le_bytes().to_vec();
        bytes.extend_from_slice(&header);
        bytes.extend_from_slice(&1.0f32.to_le_bytes());
        bytes.extend_from_slice(&2.0f32.to_le_bytes());
        std::fs::write(&path, bytes).expect("write stub fused checkpoint");
        path
    }

    fn stub_fused_spec(dir: &Path, policy: OffloadPolicy) -> LoadSpec {
        LoadSpec::new(WeightsSource::File(write_stub_fused_checkpoint(dir)))
            .with_offload_policy(policy)
    }

    /// The narrowest public reach into `ensure_rebuildable`: a resident-only owner refuses to evict.
    #[test]
    fn fused_ldm_resident_load_retains_its_reload_loaders() {
        let dir = tempfile::tempdir().expect("fixture root");
        let spec = stub_fused_spec(dir.path(), OffloadPolicy::Resident);
        let pin = spec
            .weights_file_pin()
            .expect("pin the stub checkpoint")
            .expect("a File source resolves to a pin");
        let residency = build_ldm_residency(&spec, &pin)
            .expect("the fused Resident arm must build a residency");
        let evicted = residency
            .evict_warm()
            .expect("a fused Resident load must retain reload loaders");
        assert!(
            !evicted,
            "the warm pair is built on first use, so there is nothing to evict yet"
        );
    }

    /// The whole point: a staged selection on a fused `Resident` instance reaches the real pinned
    /// phase loaders instead of being refused for having none.
    #[test]
    fn fused_ldm_staged_selection_reaches_the_real_pinned_phase_loaders() {
        let dir = tempfile::tempdir().expect("fixture root");
        let spec = stub_fused_spec(dir.path(), OffloadPolicy::Resident);
        let pin = spec
            .weights_file_pin()
            .expect("pin the stub checkpoint")
            .expect("a File source resolves to a pin");
        let residency = build_ldm_residency(&spec, &pin)
            .expect("the fused Resident arm must build a residency");
        let cancel = mlx_gen::CancelFlag::new();
        let err = residency
            .run_request_scoped(
                // The selection admission accepts today and used to refuse here.
                true,
                // Rung 4 is `Missing` on a fused file; `generate` always passes `false`.
                false,
                &cancel,
                false,
                &mut |_| {},
                |_text: &(ClipTextEncoder, ClipTextEncoder)| Ok::<(), Error>(()),
                |_| Ok(()),
                |_heavy: &SdxlHeavyOwned, (), _: &mut dyn FnMut(Progress)| Ok::<(), Error>(()),
            )
            .expect_err("the stub checkpoint carries no SDXL tensors, so the text phase must fail");
        let msg = err.to_string();
        assert!(
            !msg.contains("no reload loaders"),
            "a staged selection must reach the pinned phase loaders, not the resident-only \
             refusal: {msg}"
        );
        assert!(
            msg.contains("sdxl LDM checkpoint is missing the"),
            "the staged text phase must have re-split the pinned fused file: {msg}"
        );
    }

    /// The reach half of the chain: every layer between a request and `generate` accepts a staged
    /// selection on a fused `Resident` load, which is exactly why the instance has to back it.
    #[test]
    fn fused_ldm_staged_residency_is_declared_and_admitted_on_every_layer() {
        use mlx_gen::gen_core::{
            MemoryNumericTier, MemorySelection, MemoryStrategy, MemoryStrategySupport,
            StagedResidencyAvailability,
        };

        assert_eq!(
            descriptor().capabilities.staged_residency_availability(),
            StagedResidencyAvailability::Selectable,
            "the static descriptor publishes rung 1 as selectable for every SDXL route"
        );

        let dir = tempfile::tempdir().expect("fixture root");
        let spec = stub_fused_spec(dir.path(), OffloadPolicy::Resident);
        let contract =
            crate::memory_strategy::weights_free_memory_strategy_contract(descriptor().id, &spec)
                .expect("declaration-equivalent contract for a fused load");
        assert_eq!(
            contract
                .capability(MemoryStrategy::StagedResidency)
                .map(|capability| &capability.support),
            Some(&MemoryStrategySupport::Implemented),
            "rung 1 is declared Implemented on a fused File + Resident load"
        );
        assert!(
            !crate::memory_strategy::streamable(&spec),
            "rung 4 stays Missing on a fused file — only rung 1 is at issue here"
        );
        contract
            .validate_selection(&MemorySelection {
                strategy: MemoryStrategy::StagedResidency,
                parameters: Default::default(),
                tier: MemoryNumericTier {
                    precision: Precision::Bf16,
                    quant: None,
                    component_precision_floors: &[],
                },
            })
            .expect("admission accepts a staged selection on a fused load");
    }
}
#[test]
#[ignore = "model inference; explicit model-proof workflow"]
// Prior ignore attribute: #[ignore = "real fused SDXL checkpoint + Metal; set SDXL_LDM_CHECKPOINT and SDXL_TOKENIZER_ROOT"]
fn fused_ldm_real_weight_render_is_nonconstant() {
    let checkpoint = std::env::var_os("SDXL_LDM_CHECKPOINT")
        .map(std::path::PathBuf::from)
        .expect("set SDXL_LDM_CHECKPOINT");
    let tokenizer_root = std::env::var_os("SDXL_TOKENIZER_ROOT")
        .map(std::path::PathBuf::from)
        .expect("set SDXL_TOKENIZER_ROOT");
    let spec = LoadSpec::new(WeightsSource::File(checkpoint));
    let generator = load_from_ldm_file(&spec, &tokenizer_root).expect("load fused SDXL");
    let output = (
        crate::test_device::record_model_inference_device(),
        generator.generate(
            &GenerationRequest {
                prompt: "a red fox in a snowy forest, cinematic photograph".into(),
                negative_prompt: Some("blurry, low quality".into()),
                width: 512,
                height: 512,
                steps: Some(2),
                guidance: Some(5.0),
                seed: Some(14024),
                ..Default::default()
            },
            &mut |_| {},
        ),
    )
        .1
        .expect("render fused SDXL");
    let GenerationOutput::Images(images) = output else {
        panic!("SDXL returned non-image output");
    };
    let image = images.first().expect("one image");
    assert_eq!((image.width, image.height), (512, 512));
    let min = *image.pixels.iter().min().expect("pixels");
    let max = *image.pixels.iter().max().expect("pixels");
    assert!(max.saturating_sub(min) > 32, "render is nearly constant");

    println!("\nMODEL_INFERENCE_BODY_COMPLETED model::fused_ldm_real_weight_render_is_nonconstant");
}

pub(crate) fn reference_inputs(req: &GenerationRequest) -> Result<Option<(&Image, Option<f32>)>> {
    let mut reference = None;
    for c in &req.conditioning {
        if let Conditioning::Reference { image, strength } = c {
            if reference.is_some() {
                return Err(Error::Msg(
                    "sdxl: multiple reference images are not supported (single img2img init only)"
                        .into(),
                ));
            }
            reference = Some((image, strength.or(req.strength)));
        }
    }
    Ok(reference)
}

pub(crate) fn mask_input(req: &GenerationRequest) -> Result<Option<&Image>> {
    let mut mask = None;
    for c in &req.conditioning {
        if let Conditioning::Mask { image } = c {
            if mask.is_some() {
                return Err(Error::Msg(
                    "sdxl: multiple inpaint masks are not supported".into(),
                ));
            }
            mask = Some(image);
        }
    }
    Ok(mask)
}

pub(crate) fn control_inputs(req: &GenerationRequest) -> Result<Vec<(&Image, f32)>> {
    let mut controls = Vec::new();
    for c in &req.conditioning {
        if let Conditioning::Control { image, scale, .. } = c {
            // `None` → the diffusers `controlnet_conditioning_scale` default (full strength);
            // `Some(x)` — including `Some(0.0)` for an inert branch — is used verbatim (F-085).
            controls.push((image, scale.unwrap_or(DEFAULT_CONTROLNET_SCALE)));
        }
    }
    Ok(controls)
}

pub(crate) fn validate_image_roles(
    sampler_name: &str,
    is_accel: bool,
    use_curated: bool,
    has_reference: bool,
    has_mask: bool,
) -> Result<()> {
    if is_accel && has_reference {
        return Err(Error::Msg(format!(
            "sdxl: the {sampler_name:?} acceleration sampler is txt2img-only (no img2img \
                 reference) in this build"
        )));
    }
    // Inpaint (Mask) rides the ancestral img2img path and needs an init image to blend against.
    if has_mask {
        if is_accel {
            return Err(Error::Msg(
                "sdxl: inpaint masks are not supported with the acceleration samplers".into(),
            ));
        }
        if use_curated {
            return Err(Error::Msg(
                    "sdxl: curated samplers/schedulers are not supported with an inpaint Mask (its \
                     per-step blend has no post-step hook in the callback Sampler) — use the default \
                     euler_ancestral"
                        .into(),
                ));
        }
        if !has_reference {
            return Err(Error::Msg(
                "sdxl: inpaint requires an init image (a Reference) alongside the Mask".into(),
            ));
        }
    }
    Ok(())
}

/// The provider allows an unused loaded Control branch, but every actual Control needs a loaded peer.
pub(crate) fn control_count_matches_loaded(control_count: usize, loaded_count: usize) -> bool {
    control_count == 0 || control_count == loaded_count
}

pub(crate) fn validate_control_roles(
    sampler_name: &str,
    guidance: Option<f32>,
    control_count: usize,
    loaded_count: usize,
    has_mask: bool,
) -> Result<()> {
    if control_count != 0 {
        validate_control_acceleration(sampler_name, guidance)?;
        if !control_count_matches_loaded(control_count, loaded_count) {
            return Err(Error::Msg(format!(
                    "sdxl: {} Control conditioning(s) passed but the model was loaded with {} control \
                     checkpoint(s) (set LoadSpec::control + extra_controls, one per Control, in order)",
                    control_count,
                    loaded_count
                )));
        }
        if has_mask {
            return Err(Error::Msg(
                "sdxl: combining a ControlNet (Control) with an inpaint Mask is not supported"
                    .into(),
            ));
        }
    }

    Ok(())
}

/// IP strength is a raw image-prompt scalar, NOT the clamped img2img schedule strength.
pub(crate) fn ip_reference_scale(requested: Option<f32>) -> f32 {
    requested.unwrap_or(IP_DEFAULT_SCALE)
}
