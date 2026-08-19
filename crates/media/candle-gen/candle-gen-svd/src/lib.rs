//! # candle-gen-svd
//!
//! **Stable Video Diffusion (img2vid-xt)** image-to-video provider for [`candle-gen`](candle_gen) —
//! the candle (Windows/CUDA) sibling of `mlx-gen-svd`. SVD has **no** `candle-transformers` reference:
//! the `UNetSpatioTemporalConditionModel` ([`unet`]), the `AutoencoderKLTemporalDecoder` temporal VAE
//! ([`vae`], built on a from-scratch causal conv3d since candle ships none), the OpenCLIP ViT-H
//! `CLIPVisionModelWithProjection` image encoder ([`image_encoder`]), and the EDM `EulerDiscreteScheduler`
//! ([`scheduler`]) are all ported here from the `stabilityai/stable-video-diffusion-img2vid-xt`
//! checkpoint.
//!
//! **img2vid (sc-5493):** a single [`Conditioning::Reference`] image is CLIP-encoded for the UNet
//! cross-attention conditioning and (noise-augmented) VAE-encoded into the per-frame image latent that
//! is channel-concatenated into the UNet input. `motion_bucket_id` / `noise_aug_strength` /
//! `conditioning_fps` / `decode_chunk_size` / `frames` / `steps` / the CFG ceiling come from the
//! request; `req.fps` is the decoupled output/playback cadence.
//!
//! **Dtypes:** every component defaults to **f32** — the VAE always (`force_upcast=True`), and the
//! UNet + image encoder too, because fp16 overflows to NaN in the deep spatio-temporal UNet and bf16's
//! coarse mantissa collapses the wide-σ EDM denoise (see `Components::load` for the full rationale).
//! The experimental fp16/bf16 paths are reachable only via the `SVD_FORCE_F16` / `SVD_FORCE_BF16` env
//! vars (the sc-5493 GPU follow-up). `backend = "candle"`, `mac_only = false`.

pub mod config;
pub mod conv3d;
pub mod embeddings;
pub mod image_encoder;
pub mod pipeline;
pub mod preprocess;
pub mod scheduler;
pub mod transformer;
pub mod unet;
pub mod vae;

use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex};

use candle_gen::candle_core::{DType, Device, Tensor};
use candle_gen::candle_nn::VarBuilder;
use candle_gen::gen_core::runtime::LoadPhase;
use candle_gen::gen_core::{
    self, Capabilities, Conditioning, ConditioningKind, GenerationOutput, GenerationRequest,
    Generator, Image, LoadSpec, Modality, ModelDescriptor, OffloadPolicy, PerComponentBytes,
    Progress, StepSupport, WeightsSource,
};
use candle_gen::{check_cancel, run_three_stage_sequential, CandleError, Result as CResult};

use config::{
    ImageEncoderConfig, SchedulerConfig, UnetConfig, VaeConfig, MODEL_ID, SIZE_ALIGN, VAE_SCALE,
};
use image_encoder::SvdImageEncoder;
use pipeline::SvdParams;
use scheduler::EdmSchedule;
use unet::SvdUnet;
/// The concrete VAE assigned to the SVD-XT route.
pub type ProviderVae = vae::SvdVae;
/// Provider-facing SVD geometry, derived from the decoder implementation.
pub const VAE_TILING: candle_gen::gen_core::tiling::VaeTiling = ProviderVae::VAE_TILING;

/// OpenCLIP ViT-H image-normalization mean/std (the SVD `feature_extractor`).
#[allow(clippy::excessive_precision)]
const CLIP_MEAN: [f32; 3] = [0.481_454_66, 0.457_827_5, 0.408_210_73];
#[allow(clippy::excessive_precision)]
const CLIP_STD: [f32; 3] = [0.268_629_54, 0.261_302_58, 0.275_777_11];
const CLIP_SIZE: usize = 224;

/// The lazily-loaded SVD components (image encoder + VAE + UNet), cached behind the generator's
/// `Mutex` for the worker's `Arc<dyn Generator>` reuse.
#[derive(Clone)]
struct Components {
    image_encoder: Arc<SvdImageEncoder>,
    vae: Arc<ProviderVae>,
    unet: Arc<SvdUnet>,
}

struct ConditioningComponents {
    image_encoder: SvdImageEncoder,
    vae: ProviderVae,
}

fn load_image_encoder(root: &Path, device: &Device) -> CResult<SvdImageEncoder> {
    SvdImageEncoder::new(
        &ImageEncoderConfig::default(),
        component_vb(root, "image_encoder", "model", dense_dtype(), device)?,
    )
    .map_err(Into::into)
}

fn load_vae(root: &Path, device: &Device) -> CResult<ProviderVae> {
    ProviderVae::new(
        &VaeConfig::default(),
        component_vb(root, "vae", "diffusion_pytorch_model", DType::F32, device)?,
    )
    .map_err(Into::into)
}

fn load_unet(root: &Path, device: &Device) -> CResult<SvdUnet> {
    SvdUnet::new(
        &UnetConfig::default(),
        component_vb(
            root,
            "unet",
            "diffusion_pytorch_model",
            dense_dtype(),
            device,
        )?,
    )
    .map_err(Into::into)
}

impl ConditioningComponents {
    fn load(root: &Path, device: &Device) -> CResult<Self> {
        Ok(Self {
            image_encoder: load_image_encoder(root, device)?,
            vae: load_vae(root, device)?,
        })
    }
}

impl Components {
    /// Load every component from a checkpoint snapshot dir (`vae/` + `unet/` + `image_encoder/`). Every
    /// component defaults to **f32** (the VAE always does, `force_upcast=True`; the UNet + image encoder
    /// too — see the rationale below). The experimental fp16/bf16 paths are opt-in via `SVD_FORCE_F16` /
    /// `SVD_FORCE_BF16`.
    fn load(root: &Path, device: &Device) -> CResult<Self> {
        let vae = load_vae(root, device)?;
        let unet = load_unet(root, device)?;
        let image_encoder = load_image_encoder(root, device)?;
        Ok(Self {
            image_encoder: Arc::new(image_encoder),
            vae: Arc::new(vae),
            unet: Arc::new(unet),
        })
    }
}

/// The dtype the UNet + image encoder load at.
///
/// The UNet + image encoder run **f32** (the VAE always does). SVD ships an fp16 checkpoint, but the
/// candle GPU dtype story is a tradeoff this provider can't yet square at fp16: fp16's narrow exponent
/// range overflows to NaN in the deep spatio-temporal UNet (max-abs climbs to Inf at the last up-block —
/// candle accumulates fp16 conv/matmul in fp16 where torch's cudnn/cublas use f32), while bf16's 8-bit
/// mantissa is too coarse for the wide-σ (700→0.002) EDM denoise and collapses to noise. f32 is the only
/// dtype that produces correct video today. The fp16 path with targeted f32-upcast of the overflowing ops
/// (so native-res clips fit in VRAM) is the sc-5493 GPU follow-up. `SVD_FORCE_F16` / `SVD_FORCE_BF16`
/// reach those paths for that work.
///
/// Read by BOTH the load path and [`component_footprint`], so the footprint sizes the file the loader
/// will actually pick — including under the env escapes (sc-12397).
fn dense_dtype() -> DType {
    if std::env::var("SVD_FORCE_F16").is_ok() {
        DType::F16
    } else if std::env::var("SVD_FORCE_BF16").is_ok() {
        DType::BF16
    } else {
        DType::F32
    }
}

/// Resolve the ONE `.safetensors` a component loads, preferring the on-disk `.fp16` variant when loading
/// at [`DType::F16`] (half the load IO), else the full-precision file.
///
/// **The single source of truth for which file a component reads** — [`component_vb`] mmaps whatever this
/// returns and [`component_footprint`] sizes it (sc-12397). Keeping the selection here is the whole point:
/// the upstream `stabilityai/stable-video-diffusion-img2vid-xt` snapshot ships `{stem}.safetensors` AND
/// `{stem}.fp16.safetensors` side by side in every component dir, so a consumer that sums the DIRECTORY
/// roughly DOUBLES the model and can false-reject a card that runs it fine. Only the provider knows which
/// of the pair loads, which is exactly what `gen_core::PerComponentBytes` exists to let it say.
fn component_file(root: &Path, sub: &str, stem: &str, dtype: DType) -> CResult<PathBuf> {
    let dir = root.join(sub);
    if !dir.is_dir() {
        return Err(CandleError::Msg(format!(
            "svd_xt: snapshot is missing the {sub}/ dir (expected a \
             stable-video-diffusion-img2vid-xt snapshot at {})",
            root.display()
        )));
    }
    let fp16 = dir.join(format!("{stem}.fp16.safetensors"));
    let full = dir.join(format!("{stem}.safetensors"));
    if dtype == DType::F16 && fp16.exists() {
        Ok(fp16)
    } else if full.exists() {
        Ok(full)
    } else if fp16.exists() {
        Ok(fp16)
    } else {
        Err(CandleError::Msg(format!(
            "svd_xt: no {stem}.safetensors in {sub}/ (at {})",
            dir.display()
        )))
    }
}

/// Build a `VarBuilder` over the component file [`component_file`] resolves.
fn component_vb(
    root: &Path,
    sub: &str,
    stem: &str,
    dtype: DType,
    device: &Device,
) -> CResult<VarBuilder<'static>> {
    let path = component_file(root, sub, stem, dtype)?;
    // Shared audited unsafe-mmap surface (sc-8999 / F-019). The `{stem}.fp16`/`.safetensors`
    // resolution is a genuine per-site variation, so only the mmap is shared.
    candle_gen::mmap_var_builder(&[path], dtype, device)
}

/// The provider-owned per-component on-disk footprint (sc-12397, epic 1788) — the size of the exact
/// files [`Components::load`] will mmap, NOT a directory sum.
///
/// Lets a pre-load fit gate size an SVD job honestly. The consumer (`sceneworks-worker`'s candle video
/// VRAM gate) cannot compute this itself: each component dir ships both a full-precision and an `.fp16`
/// file, and only this crate knows the [`dense_dtype`] + [`component_file`] rules that pick one.
///
/// Mapping onto [`PerComponentBytes`]' three slots: `text_encoder` = the OpenCLIP ViT-H **image**
/// encoder. SVD is image-conditioned and has no prompt encoder, but `image_encoder/` is the phase-A
/// conditioning encoder — it runs once over the driving frame before the denoise — which is the slot's
/// role. Under sequential residency the image encoder and source-image VAE encode are phase A, the
/// UNet is phase B, and a reloaded VAE decode is phase C.
///
/// A component whose file cannot be resolved contributes `0` rather than erroring: the footprint is a
/// pre-load ADMISSION signal, and reporting no signal (⇒ the caller admits) is always safer than
/// refusing a job over an unreadable path. `Components::load` reports the real error moments later.
pub(crate) fn component_footprint(spec: &LoadSpec) -> gen_core::Result<PerComponentBytes> {
    let root = match &spec.weights {
        WeightsSource::Dir(p) => p.clone(),
        WeightsSource::File(p) => p
            .parent()
            .map(Path::to_path_buf)
            .unwrap_or_else(|| p.clone()),
    };
    let dense = dense_dtype();
    let bytes = |sub: &str, stem: &str, dtype: DType| -> u64 {
        component_file(&root, sub, stem, dtype)
            .map(gen_core::safetensors_path_bytes)
            .unwrap_or(0)
    };
    Ok(PerComponentBytes {
        text_encoder: bytes("image_encoder", "model", dense),
        dit: bytes("unet", "diffusion_pytorch_model", dense),
        // The VAE always loads f32 (`force_upcast=True`), regardless of the dense dtype.
        vae: bytes("vae", "diffusion_pytorch_model", DType::F32),
    })
}

/// Upper bound on a `Reference` image's dimensions (caps host allocations on the input buffer + the
/// resize's f32 intermediates). 8192 is far above any real photo (F-164).
const MAX_REFERENCE_DIM: u32 = 8192;
/// Upper bound on requested output `frames` — SVD-XT is the 25-frame variant; per-frame latents +
/// `added_time_ids` scale linearly, so cap the allocation.
const MAX_FRAMES: u32 = 64;
/// Upper bound on requested denoise `steps` (guards a pathological value pinning the GPU).
const MAX_STEPS: u32 = 200;

/// SVD-XT img2vid descriptor — image→video via a single `Reference`, a frame-wise guidance ramp
/// (`req.guidance` overrides the ceiling), no negative prompt / sampler / scheduler / LoRA / quant.
pub fn descriptor() -> ModelDescriptor {
    ModelDescriptor {
        encoder_contract: None,
        denoiser_output_latent_space: Some(&candle_gen::gen_core::SVD_LATENT_SPACE),
        control_kinds: None,
        required_components: &[],
        id: MODEL_ID,
        family: "svd",
        backend: "candle",
        modality: Modality::Video,
        capabilities: Capabilities {
            supports_guidance: true,
            conditioning: vec![ConditioningKind::Reference],
            // Unified curated SAMPLER menu (epic 7114 P4, sc-7125, decision 3b: sampler-only, NO
            // scheduler axis — SVD keeps its native Karras EDM σ schedule). SVD is EDM v-prediction;
            // the default `euler` over `EdmModelSampling` reproduces the native v-pred Euler loop (N1).
            samplers: candle_gen::curated_sampler_names(),
            min_size: 256,
            max_size: 1024,
            max_count: 1,
            // SVD-XT's engine bound, advertised rather than hidden (sc-19559) — the candle twin
            // of `mlx-gen-svd`'s declaration, from this lane's own `MAX_STEPS` (both 200).
            supported_steps: StepSupport::Range {
                min: 1,
                max: MAX_STEPS,
            },
            supports_sequential_offload: true,
            ..Default::default()
        },
    }
}

/// The lazy candle SVD generator. Components (image encoder + VAE + UNet) are loaded on first
/// `generate` and cached behind a `Mutex` for the worker's `Arc<dyn Generator>` cache.
pub struct SvdGenerator {
    descriptor: ModelDescriptor,
    root: PathBuf,
    device: Device,
    /// Resolved once at load. Resident preserves the historical warm aggregate; Sequential loads
    /// conditioner → UNet → VAE in disjoint phases through the shared Candle lifecycle.
    offload: OffloadPolicy,
    components: Mutex<Option<Components>>,
}

/// The SVD-specific request validation the core `Capabilities::validate_request` leaves to each model
/// (size alignment + the allocation/compute knob bounds) — F-165.
fn validate_output_params(req: &GenerationRequest) -> gen_core::Result<()> {
    if !req.width.is_multiple_of(SIZE_ALIGN) || !req.height.is_multiple_of(SIZE_ALIGN) {
        return Err(gen_core::Error::Msg(format!(
            "svd_xt: {}x{} must be a multiple of {SIZE_ALIGN} (VAE 8× × UNet 8×)",
            req.width, req.height
        )));
    }
    if let Some(frames) = req.frames {
        if frames == 0 || frames > MAX_FRAMES {
            return Err(gen_core::Error::Msg(format!(
                "svd_xt: frames {frames} out of range 1..={MAX_FRAMES}"
            )));
        }
    }
    if let Some(steps) = req.steps {
        if steps == 0 || steps > MAX_STEPS {
            return Err(gen_core::Error::Msg(format!(
                "svd_xt: steps {steps} out of range 1..={MAX_STEPS}"
            )));
        }
    }
    Ok(())
}

/// Reject a `Reference` image with zero/oversized dims or a buffer that isn't `w*h*3` RGB8 (usize math
/// so the length never wraps — F-164).
fn validate_reference_image(img: &Image) -> gen_core::Result<()> {
    if img.width == 0 || img.height == 0 {
        return Err(gen_core::Error::Msg(format!(
            "svd_xt: reference image has a zero dimension ({}x{})",
            img.width, img.height
        )));
    }
    if img.width > MAX_REFERENCE_DIM || img.height > MAX_REFERENCE_DIM {
        return Err(gen_core::Error::Msg(format!(
            "svd_xt: reference image {}x{} exceeds the {MAX_REFERENCE_DIM}px dimension cap",
            img.width, img.height
        )));
    }
    if img.pixels.len()
        != candle_gen::gen_core::imageops::checked_image_buffer_len(
            img.width as usize,
            img.height as usize,
            3,
        )
        .unwrap_or(usize::MAX)
    {
        return Err(gen_core::Error::Msg(format!(
            "svd_xt: reference image pixel buffer {} != {}x{}x3 (RGB8)",
            img.pixels.len(),
            img.width,
            img.height
        )));
    }
    Ok(())
}

impl SvdGenerator {
    /// Resolve the single conditioning reference image (image→video input).
    fn reference<'a>(&self, req: &'a GenerationRequest) -> gen_core::Result<&'a Image> {
        req.conditioning
            .iter()
            .find_map(|c| match c {
                Conditioning::Reference { image, .. } => Some(image),
                _ => None,
            })
            .ok_or_else(|| {
                gen_core::Error::Msg("svd_xt: image→video requires a Reference image".into())
            })
    }

    /// Lazily load + cache the SVD components. `cached` recovers a poisoned lock (sc-9015) internally.
    fn components(&self) -> CResult<Components> {
        candle_gen::cached(&self.components, || {
            Components::load(&self.root, &self.device)
        })
    }

    /// CLIP `image_embeds` `[1, 1, 1024]` from the reference: diffusers `_resize_with_antialiasing` to
    /// 224 (gaussian-blur + align-corners bicubic, in `[-1,1]`) → CLIP mean/std normalize.
    fn clip_embeds(&self, image_encoder: &SvdImageEncoder, img: &Image) -> CResult<Tensor> {
        let unit = preprocess::resize_with_antialiasing_unit(
            &img.pixels,
            img.height as usize,
            img.width as usize,
            CLIP_SIZE,
            CLIP_SIZE,
        ); // HWC [224,224,3] in [0,1]
        let plane = CLIP_SIZE * CLIP_SIZE;
        let mut chw = vec![0f32; 3 * plane];
        for y in 0..CLIP_SIZE {
            for x in 0..CLIP_SIZE {
                for c in 0..3 {
                    let v = unit[(y * CLIP_SIZE + x) * 3 + c];
                    chw[c * plane + y * CLIP_SIZE + x] = (v - CLIP_MEAN[c]) / CLIP_STD[c];
                }
            }
        }
        let pix = Tensor::from_vec(chw, (1, 3, CLIP_SIZE, CLIP_SIZE), &self.device)?;
        let embeds = image_encoder.image_embeds(&pix)?; // [1, 1024]
        let d = embeds.dim(1)?;
        Ok(embeds.reshape((1, 1, d))?)
    }

    /// Per-frame VAE image latent `[1, F, 4, h, w]`: lanczos resize to the output size, scale to
    /// `[-1,1]`, add `noise_aug·N(0,1)`, VAE-encode (`mode()`), repeat over frames.
    #[allow(clippy::too_many_arguments)]
    fn image_latents(
        &self,
        vae: &ProviderVae,
        img: &Image,
        height: u32,
        width: u32,
        num_frames: usize,
        noise_aug: f32,
        seed: u64,
    ) -> CResult<Tensor> {
        let (oh, ow) = (height as usize, width as usize);
        let resized = candle_gen::gen_core::imageops::resize_lanczos_u8(
            &img.pixels,
            img.height as usize,
            img.width as usize,
            oh,
            ow,
        )?; // HWC [0,255] f32
        let plane = oh * ow;
        let mut chw = vec![0f32; 3 * plane];
        for y in 0..oh {
            for x in 0..ow {
                for c in 0..3 {
                    chw[c * plane + y * ow + x] = resized[(y * ow + x) * 3 + c] / 255.0;
                }
            }
        }
        let unit = Tensor::from_vec(chw, (1, 3, oh, ow), &self.device)?;
        let centered = unit.affine(2.0, -1.0)?; // [-1,1]
        let noise = pipeline::seeded_normal(seed.wrapping_add(7), (1, 3, oh, ow), &self.device)?;
        let augmented = (centered + noise.affine(noise_aug as f64, 0.0)?)?;
        let latent = vae.encode_mode(&augmented)?; // [1, 4, h, w]
        let (b, c, lh, lw) = latent.dims4()?;
        latent
            .reshape((b, 1, c, lh, lw))?
            .broadcast_as((b, num_frames, c, lh, lw))?
            .contiguous()
            .map_err(Into::into)
    }
}

struct SequentialState<'a> {
    image_embeds: Option<Tensor>,
    image_latents: Option<Tensor>,
    latents: Option<Tensor>,
    on_progress: &'a mut dyn FnMut(Progress),
}

/// Production residency-policy dispatch seam. Passing the mutable request state into only the
/// selected closure keeps the two routes mutually exclusive and makes that selection hermetically
/// testable without loading model weights.
fn dispatch_svd_offload<T, S: ?Sized>(
    offload: OffloadPolicy,
    state: &mut S,
    resident: impl FnOnce(&mut S) -> CResult<T>,
    sequential: impl FnOnce(&mut S) -> CResult<T>,
) -> CResult<T> {
    match offload {
        OffloadPolicy::Resident => resident(state),
        OffloadPolicy::Sequential => sequential(state),
    }
}

impl Generator for SvdGenerator {
    fn descriptor(&self) -> &ModelDescriptor {
        &self.descriptor
    }

    fn validate(&self, req: &GenerationRequest) -> gen_core::Result<()> {
        // Shared capability floor: size range (256..=1024), count, unsupported negative-prompt /
        // true_cfg / sampler / scheduler, and conditioning (`Reference` only). `guidance` IS supported
        // — it overrides the frame-wise CFG ceiling.
        self.descriptor
            .capabilities
            .validate_request(MODEL_ID, req)?;
        validate_output_params(req)?;
        let img = self.reference(req)?;
        validate_reference_image(img)?;
        Ok(())
    }

    fn generate(
        &self,
        req: &GenerationRequest,
        on_progress: &mut dyn FnMut(Progress),
    ) -> gen_core::Result<GenerationOutput> {
        self.validate(req)?;
        let img = self.reference(req)?;

        let mut params = SvdParams::default();
        if let Some(f) = req.frames {
            params.num_frames = f as usize;
            // Default the decode chunk to the full clip unless the request overrides it below.
            params.decode_chunk_size = f as usize;
        }
        if let Some(s) = req.steps {
            params.num_inference_steps = s as usize;
        }
        // `params.fps` is the MOTION-conditioning cadence (from `conditioning_fps`), distinct from
        // `req.fps` (the output/playback cadence applied at return time).
        if let Some(cfps) = req.conditioning_fps {
            params.fps = cfps;
        }
        if let Some(g) = req.guidance {
            params.max_guidance_scale = g;
        }
        if let Some(m) = req.motion_bucket_id {
            params.motion_bucket_id = m;
        }
        if let Some(n) = req.noise_aug_strength {
            params.noise_aug_strength = n;
        }
        if let Some(c) = req.decode_chunk_size {
            params.decode_chunk_size = c as usize;
        }
        let seed = req.seed.unwrap_or_else(gen_core::default_seed);

        // Opt-in tensor diagnostics (`SVD_DEBUG=1`) — localizes any degeneracy (NaN / all-constant)
        // across the conditioning / denoise / decode boundaries during GPU bring-up.
        let dbg = |name: &str, t: &Tensor| {
            if std::env::var("SVD_DEBUG").is_err() {
                return;
            }
            match t
                .to_dtype(DType::F32)
                .and_then(|t| t.flatten_all())
                .and_then(|t| t.to_vec1::<f32>())
            {
                Ok(v) => {
                    let n = v.len();
                    let nan = v.iter().filter(|x| x.is_nan()).count();
                    let (mut mn, mut mx, mut sum) = (f32::INFINITY, f32::NEG_INFINITY, 0f64);
                    for &x in &v {
                        if x.is_finite() {
                            mn = mn.min(x);
                            mx = mx.max(x);
                            sum += x as f64;
                        }
                    }
                    eprintln!(
                        "[svd_dbg] {name}: n={n} nan={nan} min={mn:.5} max={mx:.5} mean={:.5}",
                        sum / n as f64
                    );
                }
                Err(e) => eprintln!("[svd_dbg] {name}: stats err {e}"),
            }
        };

        let atid = pipeline::added_time_ids(&params, &self.device)?;

        // Seeded init noise scaled by `init_noise_sigma`; shared by both residency policies.
        let sched_cfg = SchedulerConfig::default();
        let sched = EdmSchedule::karras(params.num_inference_steps, &sched_cfg);
        let lh = (req.height / VAE_SCALE) as usize;
        let lw = (req.width / VAE_SCALE) as usize;
        let noise = pipeline::create_noise(seed, params.num_frames, lh, lw, &self.device)?;
        let latents = noise
            .affine(sched.init_noise_sigma() as f64, 0.0)
            .map_err(CandleError::from)?;
        dbg("init_latents", &latents);

        let mut dispatch_state = (on_progress, Some(latents));
        let frames = dispatch_svd_offload(
            self.offload,
            &mut dispatch_state,
            |dispatch| {
                let (on_progress, latents) = dispatch;
                let comps = self.components()?;
                let image_embeds = self.clip_embeds(&comps.image_encoder, img)?;
                dbg("image_embeds", &image_embeds);
                let image_latents = self.image_latents(
                    &comps.vae,
                    img,
                    req.height,
                    req.width,
                    params.num_frames,
                    params.noise_aug_strength,
                    seed,
                )?;
                dbg("image_latents", &image_latents);
                let final_latents = pipeline::denoise(
                    &comps.unet,
                    &sched_cfg,
                    latents
                        .as_ref()
                        .expect("init latents seeded before residency dispatch"),
                    &image_embeds,
                    &image_latents,
                    &atid,
                    params.num_frames,
                    params.num_inference_steps,
                    params.min_guidance_scale,
                    params.max_guidance_scale,
                    req.sampler.as_deref(),
                    seed,
                    &req.cancel,
                    &mut **on_progress,
                )?;
                dbg("final_latents", &final_latents);
                pipeline::decode_to_images_incremental(
                    &comps.vae,
                    &final_latents,
                    params.num_frames,
                    params.decode_chunk_size,
                    &req.cancel,
                    &mut **on_progress,
                )
            },
            |dispatch| {
                let (on_progress, latents) = dispatch;
                let mut state = SequentialState {
                    image_embeds: None,
                    image_latents: None,
                    latents: latents.take(),
                    on_progress: &mut **on_progress,
                };
                run_three_stage_sequential(
                    &mut state,
                    |st| {
                        check_cancel(&req.cancel)?;
                        (st.on_progress)(Progress::Loading(LoadPhase::TextEncoder));
                        ConditioningComponents::load(&self.root, &self.device)
                    },
                    |conditioners, st| {
                        let image_embeds = self.clip_embeds(&conditioners.image_encoder, img)?;
                        dbg("image_embeds", &image_embeds);
                        let image_latents = self.image_latents(
                            &conditioners.vae,
                            img,
                            req.height,
                            req.width,
                            params.num_frames,
                            params.noise_aug_strength,
                            seed,
                        )?;
                        dbg("image_latents", &image_latents);
                        st.image_embeds = Some(image_embeds);
                        st.image_latents = Some(image_latents);
                        Ok(())
                    },
                    |st| {
                        check_cancel(&req.cancel)?;
                        (st.on_progress)(Progress::Loading(LoadPhase::Renderer));
                        load_unet(&self.root, &self.device)
                    },
                    |unet, st| {
                        let final_latents = pipeline::denoise(
                            unet,
                            &sched_cfg,
                            st.latents
                                .as_ref()
                                .expect("init latents seeded before staging"),
                            st.image_embeds
                                .as_ref()
                                .expect("image embeds produced in conditioning phase"),
                            st.image_latents
                                .as_ref()
                                .expect("image latents produced in conditioning phase"),
                            &atid,
                            params.num_frames,
                            params.num_inference_steps,
                            params.min_guidance_scale,
                            params.max_guidance_scale,
                            req.sampler.as_deref(),
                            seed,
                            &req.cancel,
                            st.on_progress,
                        )?;
                        dbg("final_latents", &final_latents);
                        st.latents = Some(final_latents);
                        Ok(())
                    },
                    |st| {
                        // The denoise boundary sync has completed before this closure runs, so the
                        // conditioning tensors can release before the decoder weights materialize.
                        st.image_embeds.take();
                        st.image_latents.take();
                        check_cancel(&req.cancel)?;
                        (st.on_progress)(Progress::Loading(LoadPhase::Renderer));
                        load_vae(&self.root, &self.device)
                    },
                    |vae, st| {
                        pipeline::decode_to_images_incremental(
                            vae,
                            st.latents
                                .as_ref()
                                .expect("denoised latents produced in UNet phase"),
                            params.num_frames,
                            params.decode_chunk_size,
                            &req.cancel,
                            st.on_progress,
                        )
                    },
                    || Ok(self.device.synchronize()?),
                )
            },
        )?;

        Ok(GenerationOutput::Video {
            frames,
            // Output/playback cadence = `req.fps` (decoupled from the motion-conditioning fps); falls
            // back to the conditioning fps when unset.
            fps: req.fps.unwrap_or(params.fps),
            audio: None,
        })
    }
}

fn load_generator(spec: &LoadSpec) -> gen_core::Result<SvdGenerator> {
    let root = match &spec.weights {
        WeightsSource::Dir(p) => p.clone(),
        WeightsSource::File(_) => {
            return Err(gen_core::Error::Msg(
                "svd_xt: expected a checkpoint directory (vae/ + unet/ + image_encoder/), not a \
                 single .safetensors file"
                    .into(),
            ));
        }
    };
    if !spec.adapters.is_empty() {
        return Err(gen_core::Error::Unsupported(
            "candle svd does not support LoRA/LoKr".into(),
        ));
    }
    if spec.quantize.is_some() {
        return Err(gen_core::Error::Unsupported(
            "candle svd does not support quantization".into(),
        ));
    }
    if spec.control.is_some() || !spec.extra_controls.is_empty() || spec.ip_adapter.is_some() {
        return Err(gen_core::Error::Unsupported(
            "candle svd does not support control / IP-adapter overlays".into(),
        ));
    }
    let device = candle_gen::default_device()?;
    Ok(SvdGenerator {
        descriptor: descriptor(),
        root,
        device,
        offload: spec.offload_policy,
        components: Mutex::new(None),
    })
}

/// Construct a lazy candle SVD generator. `spec.weights` must be a [`WeightsSource::Dir`] pointing at a
/// `stabilityai/stable-video-diffusion-img2vid-xt` snapshot (`vae/` + `unet/` + `image_encoder/`).
/// Adapters / quantization / control overlays are rejected (SVD is image→video only).
pub fn load(spec: &LoadSpec) -> gen_core::Result<Box<dyn Generator>> {
    Ok(Box::new(load_generator(spec)?))
}

candle_gen::register_generators! {
    pub(crate) const REGISTRATION = descriptor => load;
    footprint = component_footprint
}

/// Add the Candle SVD provider to an explicit media registry builder.
pub fn register_providers(
    registry: candle_gen::gen_core::ProviderRegistryBuilder,
) -> candle_gen::gen_core::ProviderRegistryBuilder {
    registry.register_generator(REGISTRATION)
}

/// Build the complete explicit Candle SVD provider catalog.
pub fn provider_registry() -> candle_gen::gen_core::Result<candle_gen::gen_core::ProviderRegistry> {
    register_providers(candle_gen::gen_core::ProviderRegistryBuilder::new()).build()
}

/// Resolve the load-bearing VAE geometry for the Candle SVD generator id.
///
/// The write cap applies to one actual VAE decode pass. With the library default, the 25-frame clip
/// is one 25-frame decode pass and therefore exceeds the 14-frame cap at both shipped 1024x576 and
/// 576x1024 geometries. SceneWorks, however, resolves both product lanes to an 8-frame chunk, which
/// is below this write cap (although the live-memory budget can still require spatial tiling).
/// Consumers must therefore classify the pass using
/// `min(request_frames, max(1, decode_chunk_size))`, not the whole clip length. Neither "the
/// shipped default is always tiled" nor "an SVD clip is always single-pass" is a truthful model.
pub fn vae_tiling(provider_id: &str) -> Option<candle_gen::gen_core::tiling::VaeTiling> {
    (provider_id == MODEL_ID).then_some(VAE_TILING)
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
        assert_eq!(explicit, ["svd_xt"]);
    }

    #[test]
    fn provider_id_resolves_to_the_concrete_decoder_geometry() {
        assert_eq!(super::VAE_TILING, super::ProviderVae::VAE_TILING);
        assert_eq!(
            super::vae_tiling(super::MODEL_ID),
            Some(super::ProviderVae::VAE_TILING)
        );
        assert_eq!(super::vae_tiling("not_svd"), None);
        for (width, height) in [(1024, 576), (576, 1024)] {
            assert_eq!(
                super::VAE_TILING.writable_frame_cap(height, width),
                14,
                "{width}x{height}"
            );
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn registers_and_resolves_as_candle_video() {
        let spec = LoadSpec::new(WeightsSource::Dir("/nonexistent".into()));
        let g = crate::provider_registry()
            .unwrap()
            .load(MODEL_ID, &spec)
            .expect("svd is registered");
        assert_eq!(g.descriptor().id, MODEL_ID);
        assert_eq!(g.descriptor().family, "svd");
        assert_eq!(g.descriptor().backend, "candle");
        assert_eq!(g.descriptor().modality, Modality::Video);
    }

    #[test]
    fn descriptor_surface() {
        let d = descriptor();
        assert!(d.capabilities.supports_guidance);
        assert!(!d.capabilities.supports_negative_prompt);
        assert!(!d.capabilities.supports_true_cfg);
        assert!(!d.capabilities.mac_only);
        assert!(d.capabilities.accepts(ConditioningKind::Reference));
        // sc-7125: curated sampler menu (default euler); NO scheduler axis (decision 3b).
        assert_eq!(d.capabilities.samplers, candle_gen::curated_sampler_names());
        assert!(d.capabilities.schedulers.is_empty());
        assert_eq!(d.capabilities.min_size, 256);
        assert_eq!(d.capabilities.max_size, 1024);
        assert!(
            d.capabilities.supports_sequential_offload,
            "svd_xt must advertise the staged conditioner → UNet → VAE lifecycle"
        );
    }

    /// sc-19559 — the candle twin of `mlx-gen-svd`'s ceiling test. The bound both lanes have
    /// always enforced privately is now readable off the descriptor and rejected by the SHARED
    /// floor, so the two lanes refuse the same counts from the same declaration.
    ///
    /// Reading `ceiling()` alone would pass against a descriptor nothing enforces; asserting only
    /// the refusal would pass against a bound that stays undiscoverable. Both are asserted.
    #[test]
    fn the_step_ceiling_is_advertised_and_enforced_by_the_shared_floor() {
        let caps = descriptor().capabilities;
        assert_eq!(
            caps.supported_steps.ceiling(),
            Some(MAX_STEPS),
            "the descriptor must advertise the engine's own ceiling, weights-free"
        );
        assert_eq!(caps.supported_steps.floor(), Some(1));

        let at = |steps: u32| {
            caps.validate_request(
                MODEL_ID,
                &GenerationRequest {
                    width: 512,
                    height: 512,
                    count: 1,
                    steps: Some(steps),
                    ..Default::default()
                },
            )
        };
        assert!(
            at(MAX_STEPS).is_ok(),
            "the advertised ceiling itself must be renderable"
        );
        let err = at(MAX_STEPS + 1)
            .expect_err("the shared floor must refuse an over-ceiling count")
            .to_string();
        assert!(
            err.contains(MODEL_ID) && err.contains(&format!("1..={MAX_STEPS}")),
            "the refusal must name the model and the advertised range: {err}"
        );
    }

    #[test]
    fn load_honors_requested_sequential_policy_without_populating_resident_cache() {
        let spec = LoadSpec::new(WeightsSource::Dir("/nonexistent".into()))
            .with_offload_policy(OffloadPolicy::Sequential);
        let generator = load_generator(&spec).unwrap();
        assert_eq!(generator.offload, OffloadPolicy::Sequential);
        assert!(
            candle_gen::lock_recover(&generator.components).is_none(),
            "the sequential route must not populate the all-resident component aggregate"
        );
    }

    /// Engine-level drop-order witness for the exact SVD phases. The conditioning aggregate contains
    /// the image encoder + source-image VAE encode; it must release before the UNet loads, and the
    /// UNet must release before the decode VAE reloads. An all-resident mutation raises `max_live`.
    #[test]
    fn sequential_svd_phases_never_make_conditioner_unet_and_decoder_co_resident() {
        use std::cell::{Cell, RefCell};

        struct Phase<'a> {
            name: &'static str,
            live: &'a Cell<usize>,
            log: &'a RefCell<Vec<&'static str>>,
        }
        impl Drop for Phase<'_> {
            fn drop(&mut self) {
                self.live.set(self.live.get() - 1);
                self.log.borrow_mut().push(match self.name {
                    "conditioner" => "drop-conditioner",
                    "unet" => "drop-unet",
                    "decoder" => "drop-decoder",
                    _ => unreachable!(),
                });
            }
        }

        let live = Cell::new(0usize);
        let max_live = Cell::new(0usize);
        let log = RefCell::new(Vec::new());
        let load = |name: &'static str| {
            let next = live.get() + 1;
            live.set(next);
            max_live.set(max_live.get().max(next));
            log.borrow_mut().push(match name {
                "conditioner" => "load-conditioner",
                "unet" => "load-unet",
                "decoder" => "load-decoder",
                _ => unreachable!(),
            });
            Phase {
                name,
                live: &live,
                log: &log,
            }
        };
        let mut state = ();
        dispatch_svd_offload(
            OffloadPolicy::Sequential,
            &mut state,
            |_| {
                log.borrow_mut().push("resident-route");
                Err(CandleError::Msg(
                    "the sequential policy selected the resident route".into(),
                ))
            },
            |state| {
                run_three_stage_sequential(
                    state,
                    |_| Ok(load("conditioner")),
                    |_, _| {
                        log.borrow_mut().push("use-conditioner");
                        Ok(())
                    },
                    |_| Ok(load("unet")),
                    |_, _| {
                        log.borrow_mut().push("use-unet");
                        Ok(())
                    },
                    |_| Ok(load("decoder")),
                    |_, _| {
                        log.borrow_mut().push("use-decoder");
                        Ok(())
                    },
                    || Ok(()),
                )
            },
        )
        .unwrap();

        assert_eq!(max_live.get(), 1, "no two SVD phases may be co-resident");
        assert_eq!(
            *log.borrow(),
            vec![
                "load-conditioner",
                "use-conditioner",
                "drop-conditioner",
                "load-unet",
                "use-unet",
                "drop-unet",
                "load-decoder",
                "use-decoder",
                "drop-decoder"
            ]
        );
    }

    fn ref_req(w: u32, h: u32) -> GenerationRequest {
        GenerationRequest {
            width: w,
            height: h,
            conditioning: vec![Conditioning::Reference {
                image: Image {
                    width: w,
                    height: h,
                    pixels: vec![0u8; w as usize * h as usize * 3],
                },
                strength: None,
            }],
            ..Default::default()
        }
    }

    #[test]
    fn validate_accepts_img2vid_and_rejects_unsupported() {
        let spec = LoadSpec::new(WeightsSource::Dir("/nonexistent".into()));
        let g = crate::provider_registry()
            .unwrap()
            .load(MODEL_ID, &spec)
            .unwrap();
        // 1024×576 = 16×64 / 9×64 with a well-formed reference passes.
        assert!(g.validate(&ref_req(1024, 576)).is_ok());
        // Missing reference image.
        assert!(g
            .validate(&GenerationRequest {
                width: 512,
                height: 512,
                ..Default::default()
            })
            .is_err());
        // Unaligned size (not a multiple of 64).
        assert!(g.validate(&ref_req(700, 704)).is_err());
        // sc-12587: `SIZE_ALIGN` is the pinned stride SceneWorks ties `requiresDimensionsMultipleOf`
        // to. A multiple of 32 that is not a multiple of SIZE_ALIGN (64) is still rejected — pin the
        // value and mutation-check it (the stride is VAE 8× × UNet 8×, not the bare VAE scale).
        assert_eq!(SIZE_ALIGN, 64);
        // 736 = 23×32 — a multiple of 32 but not SIZE_ALIGN (64).
        let off_align = g.validate(&ref_req(736, 576)).unwrap_err().to_string();
        assert!(off_align.contains("multiple of 64"), "got: {off_align}");

        // Out-of-range frames.
        assert!(g
            .validate(&GenerationRequest {
                frames: Some(MAX_FRAMES + 1),
                ..ref_req(512, 512)
            })
            .is_err());
    }

    #[test]
    fn load_rejects_unwired_surfaces() {
        use candle_gen::gen_core::{AdapterKind, AdapterSpec, Quant};
        let lora = LoadSpec::new(WeightsSource::Dir("/snap".into())).with_adapters(vec![
            AdapterSpec::new("/lora.safetensors".into(), 1.0, AdapterKind::Lora),
        ]);
        assert!(matches!(
            load(&lora).err().expect("err"),
            gen_core::Error::Unsupported(_)
        ));
        let quant = LoadSpec::new(WeightsSource::Dir("/snap".into())).with_quant(Quant::Q8);
        assert!(matches!(
            load(&quant).err().expect("err"),
            gen_core::Error::Unsupported(_)
        ));
    }

    #[test]
    fn load_rejects_single_file_source() {
        let spec = LoadSpec::new(WeightsSource::File("/tmp/w.safetensors".into()));
        let err = load(&spec).err().expect("expected an error").to_string();
        assert!(err.contains("checkpoint directory"), "got: {err}");
    }

    /// sc-12397: the footprint must size the ONE file per component that [`component_vb`] mmaps — NOT the
    /// directory.
    ///
    /// This is the whole reason SVD owns its own footprint rather than letting the consumer sum subdirs.
    /// The upstream `stabilityai/stable-video-diffusion-img2vid-xt` snapshot ships `X.safetensors` AND
    /// `X.fp16.safetensors` side by side in every component dir, so a directory sum roughly DOUBLES a
    /// ~8.9 GiB model — enough for a pre-load VRAM gate to false-reject a card that renders it fine today.
    ///
    /// Kills the mutation that matters: swapping `component_file` for a `safetensors_dir_bytes(dir)` sum
    /// makes every field here read `full + fp16` and the assert fails.
    #[test]
    fn component_footprint_sizes_the_selected_file_not_the_whole_dir() {
        let root_tmp = tempfile::tempdir().unwrap();
        let root = root_tmp.path().to_path_buf();
        // Both dtype variants side by side, as the real snapshot ships them.
        for (sub, stem, full, fp16) in [
            ("unet", "diffusion_pytorch_model", 6_000_u64, 3_000_u64),
            ("vae", "diffusion_pytorch_model", 400, 200),
            ("image_encoder", "model", 2_500, 1_250),
        ] {
            let dir = root.join(sub);
            std::fs::create_dir_all(&dir).unwrap();
            for (name, len) in [
                (format!("{stem}.safetensors"), full),
                (format!("{stem}.fp16.safetensors"), fp16),
            ] {
                std::fs::File::create(dir.join(name))
                    .unwrap()
                    .set_len(len)
                    .unwrap();
            }
        }

        let spec = LoadSpec::new(WeightsSource::Dir(root.clone()));
        let fp = component_footprint(&spec).expect("footprint");
        // Default dtype is F32 ⇒ every component takes the FULL file, never the fp16 sibling…
        assert_eq!(fp.dit, 6_000, "unet: the f32 file, not full+fp16 (9_000)");
        assert_eq!(fp.vae, 400, "vae: always f32");
        assert_eq!(fp.text_encoder, 2_500, "image_encoder: the f32 file");
        // …so the total is the load, not the directory. A dir sum would read 13_350.
        assert_eq!(fp.text_encoder + fp.dit + fp.vae, 8_900);
    }

    /// A component the loader cannot resolve contributes `0`, and never errors: the footprint is a
    /// pre-load ADMISSION signal, so "no signal" (⇒ the caller admits) beats refusing a job over an
    /// unreadable path. `Components::load` surfaces the real error moments later.
    #[test]
    fn component_footprint_reports_no_signal_rather_than_failing() {
        let spec = LoadSpec::new(WeightsSource::Dir("/nonexistent-svd-snapshot".into()));
        let fp = component_footprint(&spec).expect("a missing snapshot is not a footprint error");
        assert_eq!(
            (fp.text_encoder, fp.dit, fp.vae),
            (0, 0, 0),
            "an unreadable snapshot must read as no signal"
        );
    }
}
