//! SDXL-family header-only iOS pricing (epic 16774): the per-phase COUNTED price of one request —
//! the component bytes resident in each phase under the request's regime plus the phase's largest
//! counted quantum — built from the SAME formulas the `ios_quantum` guards check.
//!
//! [`ios_pricing`] constructs every component the request loads LAZILY from its safetensors
//! (`Weights::from_file` handles, the device's load-time cast rule; never read, never evaluated)
//! and walks the request's execution with each module's own guard formulas
//! (`price_forward` / `price_encode` / `price_decode` beside every guarded forward), over
//! shape-only activations. Load floors and resident bytes are the loaders' header-only floors
//! (`loader::cast_load_parts`). So the table a device capture is scaled by and the guards the
//! runtime checks cannot drift: `quantum_tests::pricing` runs the real pipeline under the CPU
//! diagnostic observer and asserts, per phase, that the largest checked scratch IS the priced one.
//!
//! The regime is the request's [`GenerationMemory`] (the selection's `generation_memory`, as the
//! catalog writes it onto the request): rung 1 (`stage_residency`) sheds the text encoders before
//! the heavy load, the IP-Adapter image encoder and Resampler once the image prompt is encoded
//! (`model::sheds_ip_image_encoder`) and the U-Net / ControlNet / IP state before the decode; rung 2 tiles the decode (and the init-image encode) at its tile
//! edge; rung 4 streams the U-Net's transformer blocks — and a streamable load (iOS's
//! `DeferredMaterialization` over a snapshot whose U-Net stays lazy) streams them on EVERY request,
//! each block read inside its own priced quantum, so the U-Net's resident bytes are its non-block
//! remainder whichever rung is selected. Guided requests are priced with classifier-free guidance
//! on (the larger of the two shapes) at the request's CFG batching.
//!
//! Not priced (a typed refusal, never a silent under-price): adapter (LoRA/LoKr/LoHa) installs, the
//! PiD decoder, a fused single-file (LDM) source, and a sealed geometry decode policy.

use mlx_gen::gen_core::{CfgBatching, GenerationMemory, LoadSpec, MemoryMode, WeightsSource};
use mlx_gen::ios_pricing::{IosPhase, IosPricing, IosPricingRequest};
use mlx_gen::{Error, Result};
use mlx_rs::{Array, Dtype};
use std::path::{Path, PathBuf};

use crate::config::{ClipTextConfig, UNetConfig, VaeConfig};
use crate::ip_adapter::{IpImageEncoder, ResamplerConfig};
use crate::quantum::{self as q, Backend, Quanta};
use crate::text_encoder::ClipTextEncoder;
use crate::unet::{ControlNet, UNet2DConditionModel};
use crate::vae::Autoencoder;
use crate::vision_encoder::VisionConfig;

/// Every quantum label this crate checks (`ios_quantum::{run, before, before_load*}`) and the phase
/// it runs in. A device capture attributes its kernel peak through [`ios_quantum_phase`].
const PHASES: &[(&str, IosPhase)] = &[
    (q::TEXT_ENCODER_1_LOAD, IosPhase::Load),
    (q::TEXT_ENCODER_2_LOAD, IosPhase::Load),
    (q::UNET_LOAD, IosPhase::Load),
    (q::VAE_LOAD, IosPhase::Load),
    (q::VAE_FP16_FIX_LOAD, IosPhase::Load),
    (q::CONTROLNET_LOAD, IosPhase::Load),
    (q::IP_IMAGE_ENCODER_LOAD, IosPhase::Load),
    (q::IP_ADAPTER_LOAD, IosPhase::Load),
    (q::LDM_TEXT_LOAD, IosPhase::Load),
    (q::LDM_HEAVY_LOAD, IosPhase::Load),
    (q::ADAPTER_LOAD, IosPhase::Load),
    (q::ADAPTER_MERGE, IosPhase::Load),
    (q::CLIP_EMBEDDINGS, IosPhase::Conditioning),
    (q::CLIP_LAYER, IosPhase::Conditioning),
    (q::CLIP_POOLED, IosPhase::Conditioning),
    (q::VISION_EMBEDDINGS, IosPhase::Conditioning),
    (q::VISION_LAYER, IosPhase::Conditioning),
    (q::RESAMPLER_PROJ_IN, IosPhase::Conditioning),
    (q::RESAMPLER_LAYER, IosPhase::Conditioning),
    (q::RESAMPLER_PROJ_OUT, IosPhase::Conditioning),
    (q::VAE_ENCODE_IN, IosPhase::Encode),
    (q::VAE_ENCODE_RESNET, IosPhase::Encode),
    (q::VAE_ENCODE_ATTENTION, IosPhase::Encode),
    (q::VAE_DOWNSAMPLE, IosPhase::Encode),
    (q::VAE_ENCODE_TILE, IosPhase::Encode),
    (q::VAE_ENCODE_TILED_STAGE, IosPhase::Encode),
    (q::VAE_ENCODE_OUT, IosPhase::Encode),
    (q::CONTROL_COND_CONV, IosPhase::Encode),
    (q::UNET_STEM, IosPhase::Denoise),
    (q::UNET_RESNET, IosPhase::Denoise),
    (q::UNET_DOWNSAMPLE, IosPhase::Denoise),
    (q::UNET_UPSAMPLE, IosPhase::Denoise),
    (q::TRANSFORMER_PROJ_IN, IosPhase::Denoise),
    (q::TRANSFORMER_BLOCK, IosPhase::Denoise),
    (q::TRANSFORMER_PROJ_OUT, IosPhase::Denoise),
    (q::CONTROL_DOWN_INJECTION, IosPhase::Denoise),
    (q::CONTROL_MID_INJECTION, IosPhase::Denoise),
    (q::UNET_CONV_OUT, IosPhase::Denoise),
    (q::CONTROL_STEM, IosPhase::Denoise),
    (q::CONTROL_ZERO_CONV, IosPhase::Denoise),
    (q::SAMPLER_STEP, IosPhase::Denoise),
    // The shared sampler-step guard a few-step acceleration sampler checks inside SDXL's denoise.
    (mlx_gen::sampler::SAMPLER_STEP_QUANTUM, IosPhase::Denoise),
    (q::VAE_DECODE_IN, IosPhase::Decode),
    (q::VAE_RESNET, IosPhase::Decode),
    (q::VAE_ATTENTION, IosPhase::Decode),
    (q::VAE_UPSAMPLE, IosPhase::Decode),
    (q::VAE_DECODE_OUT, IosPhase::Decode),
    (q::VAE_TILE, IosPhase::Decode),
    (q::VAE_TILED_STAGE, IosPhase::Decode),
    (q::DECODED_TO_IMAGE, IosPhase::Decode),
];

/// The phase a quantum label of this crate runs in: the longest label constant `label` starts with
/// (so `SDXL VAE tiled stage` is not read as `SDXL VAE tile`). `None` for a label this crate never
/// checks (e.g. the shared tiled-convolution core's synchronization).
pub fn ios_quantum_phase(label: &str) -> Option<IosPhase> {
    PHASES
        .iter()
        .filter(|(constant, _)| label.starts_with(constant))
        .max_by_key(|(constant, _)| constant.len())
        .map(|(_, phase)| *phase)
}

/// The request's counted per-phase price on the device (Metal kernel transients), header-only.
pub fn ios_pricing(req: &IosPricingRequest<'_>) -> Result<IosPricing> {
    ios_pricing_on(req, Backend::Metal)
}

/// [`ios_pricing`] with `backend`'s kernel transients (the CPU proof fixture prices its own).
pub(crate) fn ios_pricing_on(req: &IosPricingRequest<'_>, backend: Backend) -> Result<IosPricing> {
    let regime = Regime::from_request(req)?;
    let inputs = Inputs::from_spec(req.spec, &regime)?;
    q::with_backend(backend, || {
        let priced = Priced::build(&inputs)?;
        Ok(priced.price(&regime)?.0)
    })
}

/// The SDXL-family architecture every route loads (`model::load_heavy` and the text loader pin
/// these configs); a tiny proof fixture substitutes its own.
#[derive(Clone)]
pub(crate) struct Architecture {
    pub unet: UNetConfig,
    pub vae: VaeConfig,
    pub te1: ClipTextConfig,
    pub te2: ClipTextConfig,
    pub vision: VisionConfig,
    pub resampler: ResamplerConfig,
    /// CLIP's per-window token count (77).
    pub text_tokens: i32,
    /// The text memory's width (`cross_attention_dim`, 2048).
    pub context_dim: i32,
    /// Output pixels per latent cell along each axis (the VAE's spatial scale, 8).
    pub latent_scale: i32,
    /// Control-image pixels per latent cell (the ControlNet conditioning embedding's stride, 8).
    pub control_scale: i32,
}

impl Architecture {
    pub(crate) fn sdxl() -> Self {
        Self {
            unet: UNetConfig::sdxl_base(),
            vae: VaeConfig::sdxl_base(),
            te1: ClipTextConfig::sdxl_te1(),
            te2: ClipTextConfig::sdxl_te2(),
            vision: VisionConfig::vit_h_14(),
            resampler: ResamplerConfig::plus_sdxl_vit_h(),
            text_tokens: 77,
            context_dim: 2048,
            latent_scale: 8,
            control_scale: 8,
        }
    }
}

/// The memory route of a request.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum Route {
    Text,
    /// img2img: an init-image encode, then the denoise from the f32 encoded latent.
    Reference,
    /// Inpaint: the img2img encode plus the per-step mask blend.
    Mask,
    /// ControlNet: the conditioning embedding, then the branch beside every U-Net forward.
    Control,
    /// IP-Adapter: the image prompt encode, then the decoupled cross-attention.
    IpAdapter,
}

impl Route {
    fn from_mode(mode: &MemoryMode) -> Result<Self> {
        Ok(match mode {
            MemoryMode::TextToImage => Self::Text,
            MemoryMode::ImageToImage => Self::Reference,
            MemoryMode::Edit => Self::Mask,
            MemoryMode::Other(other) if other == "control" => Self::Control,
            MemoryMode::Other(other) if other == "ip_adapter" => Self::IpAdapter,
            MemoryMode::Other(other) => {
                return Err(Error::Unsupported(format!(
                    "sdxl ios pricing: no memory route `{other}`"
                )))
            }
        })
    }

    fn encodes_init(self) -> bool {
        matches!(self, Self::Reference | Self::Mask)
    }
}

/// The execution regime one request runs under.
#[derive(Clone)]
pub(crate) struct Regime {
    pub route: Route,
    pub width: i32,
    pub height: i32,
    /// Rung 1: shed the text encoders before the heavy load, the IP image encoder after the
    /// image prompt, and the U-Net / ControlNet / IP state before the decode.
    pub staged: bool,
    pub cfg_batching: CfgBatching,
    /// Rung 2's decode tiling (the init encode tiles at the same edge).
    pub tiling: Option<mlx_gen::tiling::TilingConfig>,
    /// The text memory's token count (CLIP windows x 77).
    pub context_tokens: i32,
}

impl Regime {
    fn from_request(req: &IosPricingRequest<'_>) -> Result<Self> {
        if !req.geometry.width.is_multiple_of(8) || !req.geometry.height.is_multiple_of(8) {
            return Err(Error::Unsupported(format!(
                "sdxl ios pricing: {}x{} is not a multiple of the latent stride",
                req.geometry.width, req.geometry.height
            )));
        }
        let memory: &GenerationMemory = req.execution;
        let tiling = if memory.tile_vae_decode {
            let (edge, overlap) = crate::memory_strategy::resolve_decode_geometry(
                false,
                memory.decode_tile_edge,
                memory.decode_overlap,
            );
            Some(mlx_gen::tiling::TilingConfig::spatial_only(
                edge.unwrap_or(crate::memory_strategy::DECODE_TILE_EDGE) as i32,
                overlap.unwrap_or(crate::memory_strategy::DECODE_OVERLAP) as i32,
            ))
        } else {
            None
        };
        if memory.chunk_attention {
            return Err(Error::Unsupported(
                "sdxl ios pricing: bounded attention is not selectable on this provider".into(),
            ));
        }
        if memory.stream_transformer_blocks && !crate::memory_strategy::streamable(req.spec) {
            return Err(Error::Unsupported(
                "sdxl ios pricing: bounded transformer residency needs a streamable load".into(),
            ));
        }
        let windows = (req.token_window as i32).max(77);
        Ok(Self {
            route: Route::from_mode(req.mode)?,
            width: req.geometry.width as i32,
            height: req.geometry.height as i32,
            staged: memory.stage_residency,
            cfg_batching: memory.cfg_batching.unwrap_or(CfgBatching::Batched),
            tiling,
            context_tokens: windows,
        })
    }
}

/// The files and load facts one request's components come from.
pub(crate) struct Inputs {
    pub arch: Architecture,
    pub te1: PathBuf,
    pub te2: PathBuf,
    pub unet: PathBuf,
    /// `(file, dtype)` of the VAE the load runs (the staged fp16-fix at f16, else the in-snapshot
    /// one at f32).
    pub vae: (PathBuf, Dtype),
    pub controls: Vec<PathBuf>,
    /// `(image encoder file, IP weights file)`.
    pub ip: Option<(PathBuf, PathBuf)>,
    pub quant_bits: Option<i32>,
    /// Whether the load arms the rung-4 block stream ([`crate::memory_strategy::streamable`]).
    pub streamable: bool,
}

impl Inputs {
    fn from_spec(spec: &LoadSpec, regime: &Regime) -> Result<Self> {
        let WeightsSource::Dir(root) = &spec.weights else {
            return Err(Error::Unsupported(
                "sdxl ios pricing: a fused single-file (LDM) source is not priced".into(),
            ));
        };
        if !spec.adapters.is_empty() {
            return Err(Error::Unsupported(
                "sdxl ios pricing: adapter (LoRA/LoKr/LoHa) installs are not priced".into(),
            ));
        }
        if spec.pid.is_some() {
            return Err(Error::Unsupported(
                "sdxl ios pricing: the PiD decoder overlay is not priced".into(),
            ));
        }
        if spec.decode_geometry_policy_authoritative {
            return Err(Error::Unsupported(
                "sdxl ios pricing: a sealed geometry decode policy is not priced".into(),
            ));
        }
        if regime.route == Route::Control && spec.control.is_none() {
            return Err(Error::Unsupported(
                "sdxl ios pricing: a control request needs a loaded ControlNet".into(),
            ));
        }
        if regime.route == Route::IpAdapter && spec.ip_adapter.is_none() {
            return Err(Error::Unsupported(
                "sdxl ios pricing: an IP-Adapter request needs a loaded IP-Adapter".into(),
            ));
        }
        let dtype = crate::model::DTYPE;
        let resolve =
            |sub: &str, stem: &str| crate::loader::resolve_weight_file(root, sub, stem, dtype);
        let vae = match spec.components.get(crate::loader::VAE_FP16_FIX_COMPONENT) {
            Some(source) => (
                crate::loader::resolve_vae_fp16_fix_file(source)?,
                crate::loader::VAE_FP16_FIX_DTYPE,
            ),
            None => (
                crate::loader::resolve_vae_weight_file(root)?,
                Dtype::Float32,
            ),
        };
        let controls = spec
            .control
            .iter()
            .chain(&spec.extra_controls)
            .map(|source| match source {
                WeightsSource::File(path) | WeightsSource::Dir(path) => path.clone(),
            })
            .collect();
        let ip = match &spec.ip_adapter {
            Some(WeightsSource::Dir(dir)) => Some((
                dir.join(crate::loader::IP_ADAPTER_IMAGE_ENCODER_FILE),
                crate::loader::IP_ADAPTER_WEIGHT_FILES
                    .iter()
                    .map(|file| dir.join(file))
                    .find(|path| path.exists())
                    .ok_or_else(|| {
                        Error::Msg(format!(
                            "sdxl ios pricing: no plus/plus-face sdxl_vit-h weights under {}",
                            dir.display()
                        ))
                    })?,
            )),
            Some(WeightsSource::File(_)) => {
                return Err(Error::Unsupported(
                    "sdxl ip_adapter expects an h94/IP-Adapter snapshot directory".into(),
                ))
            }
            None => None,
        };
        Ok(Self {
            arch: Architecture::sdxl(),
            te1: resolve("text_encoder", "model")?,
            te2: resolve("text_encoder_2", "model")?,
            unet: resolve("unet", "diffusion_pytorch_model")?,
            vae,
            controls,
            ip,
            quant_bits: spec.quantize.map(|quant| quant.bits()),
            streamable: crate::memory_strategy::streamable(spec),
        })
    }
}

/// `(resident bytes after the load, the load's floor scratch)` of one component.
#[derive(Clone, Copy, Debug, Default)]
pub(crate) struct Footprint {
    pub resident: u64,
    pub floor: u64,
}

/// Every component of the request, lazily constructed, with its header-only footprints.
pub(crate) struct Priced {
    te: (ClipTextEncoder, ClipTextEncoder),
    unet: UNet2DConditionModel,
    vae: Autoencoder,
    controls: Vec<ControlNet>,
    ip: Option<IpImageEncoder>,
    pub te_bytes: [Footprint; 2],
    pub unet_bytes: Footprint,
    pub vae_bytes: Footprint,
    /// Per branch: `(load footprint at the compute dtype, resident after the tier quantization)`.
    pub control_bytes: Vec<(Footprint, u64)>,
    pub ip_encoder_bytes: Footprint,
    /// The IP weights file's load footprint (Resampler + K/V pairs at the compute dtype).
    pub ip_adapter_bytes: Footprint,
    /// The Resampler's resident bytes (shed with the image encoder after the image prompt).
    pub ip_resampler_bytes: u64,
    /// The installed K/V pairs' resident bytes (quantized with the U-Net at a packed tier).
    pub ip_kv_bytes: u64,
    arch: Architecture,
}

fn footprint(files: &[PathBuf], dtype: Dtype, keep: impl Fn(&str) -> bool) -> Result<Footprint> {
    let (resident, floor) = crate::loader::cast_load_parts(files, dtype, true, keep)?;
    Ok(Footprint { resident, floor })
}

/// Bytes of a `[out, in]` projection group-quantized at `bits` (u32 codes + one bf16 scale and
/// bias per 64-group), the pack `AdaptableLinear::quantize` produces.
fn packed_bytes(elements: u64, bits: i32) -> u64 {
    let group = crate::quant::GROUP_SIZE as u64;
    elements
        .saturating_mul(bits as u64)
        .div_ceil(8)
        .saturating_add(elements.div_ceil(group).saturating_mul(4))
}

impl Priced {
    pub(crate) fn build(inputs: &Inputs) -> Result<Self> {
        let dtype = crate::model::DTYPE;
        let arch = inputs.arch.clone();
        let one = |p: &Path| vec![p.to_path_buf()];
        let te1 = ClipTextEncoder::from_weights(
            &crate::loader::lazy_weights(&one(&inputs.te1), dtype, false)?,
            "text_model",
            &arch.te1,
        )?;
        let te2 = ClipTextEncoder::from_weights(
            &crate::loader::lazy_weights(&one(&inputs.te2), dtype, false)?,
            "text_model",
            &arch.te2,
        )?;
        let te_bytes = [
            footprint(&one(&inputs.te1), dtype, |_| true)?,
            footprint(&one(&inputs.te2), dtype, |_| true)?,
        ];

        let mut unet = UNet2DConditionModel::from_weights(
            &crate::loader::lazy_weights(&one(&inputs.unet), dtype, false)?,
            &arch.unet,
        )?;
        let (resident, floor) = crate::loader::unet_load_parts(
            &one(&inputs.unet),
            dtype,
            true,
            inputs.streamable,
            &arch.unet,
        )?;
        let unet_bytes = Footprint { resident, floor };

        let (vae_file, vae_dtype) = &inputs.vae;
        let vae = Autoencoder::from_weights(
            &crate::loader::lazy_weights(&one(vae_file), *vae_dtype, true)?,
            &arch.vae,
        )?;
        let vae_bytes = footprint(&one(vae_file), *vae_dtype, |_| true)?;

        let mut controls = Vec::new();
        let mut control_bytes = Vec::new();
        for file in &inputs.controls {
            let mut control = ControlNet::from_weights(
                &crate::loader::lazy_weights(&one(file), dtype, false)?,
                &arch.unet,
            )?;
            let load = footprint(&one(file), dtype, |_| true)?;
            let resident = match inputs.quant_bits {
                Some(bits) => {
                    crate::quant::with_lazy_quantize(|| control.quantize(bits))?;
                    crate::memory_strategy::control_resident_bytes(file, Some(bits))
                        .map_err(|e| Error::Msg(e.to_string()))?
                }
                None => load.resident,
            };
            controls.push(control);
            control_bytes.push((load, resident));
        }

        let (mut ip, mut ip_encoder_bytes, mut ip_adapter_bytes) =
            (None, Footprint::default(), Footprint::default());
        let (mut ip_resampler_bytes, mut ip_kv_bytes) = (0, 0);
        if let Some((encoder_file, ip_file)) = &inputs.ip {
            let encoder = crate::vision_encoder::ClipVisionEncoder::from_weights(
                &crate::loader::lazy_weights(&one(encoder_file), dtype, false)?,
                &arch.vision,
            )?;
            let ip_w = crate::loader::lazy_weights(&one(ip_file), dtype, false)?;
            let resampler =
                crate::ip_adapter::Resampler::from_weights(&ip_w, "image_proj", &arch.resampler)?;
            unet.install_ip_adapter(crate::ip_adapter::load_ip_kv_pairs(&ip_w)?)?;
            ip = Some(IpImageEncoder::with_image_size(
                encoder,
                resampler,
                arch.vision.image_size as usize,
            ));
            ip_encoder_bytes = footprint(&one(encoder_file), dtype, |_| true)?;
            ip_adapter_bytes = footprint(&one(ip_file), dtype, |_| true)?;
            ip_resampler_bytes =
                footprint(&one(ip_file), dtype, |name| name.starts_with("image_proj."))?.resident;
            ip_kv_bytes = match inputs.quant_bits {
                Some(bits) => {
                    mlx_gen::gen_core::weightsmeta::safetensors_path_tensor_headers(ip_file)
                        .map_err(|e| Error::Msg(e.to_string()))?
                        .iter()
                        .filter(|h| h.name.starts_with("ip_adapter."))
                        .map(|h| packed_bytes(h.shape.iter().product::<usize>() as u64, bits))
                        .sum()
                }
                None => {
                    footprint(&one(ip_file), dtype, |name| name.starts_with("ip_adapter."))?
                        .resident
                }
            };
        }
        if inputs.streamable {
            unet.arm_block_streams(
                &WeightsSource::File(inputs.unet.clone()),
                dtype,
                inputs.quant_bits,
            );
        }
        Ok(Self {
            te: (te1, te2),
            unet,
            vae,
            controls,
            ip,
            te_bytes,
            unet_bytes,
            vae_bytes,
            control_bytes,
            ip_encoder_bytes,
            ip_adapter_bytes,
            ip_resampler_bytes,
            ip_kv_bytes,
            arch,
        })
    }

    /// The heavy bundle's resident bytes in the denoise (`shed_ip_encoder`: lever d) and the
    /// load order's running totals.
    fn heavy(&self, shed_ip_encoder: bool) -> u64 {
        let controls: u64 = self.control_bytes.iter().map(|(_, r)| r).sum();
        let ip = if self.ip.is_some() {
            let encoder = if shed_ip_encoder {
                0
            } else {
                self.ip_encoder_bytes.resident + self.ip_resampler_bytes
            };
            encoder + self.ip_kv_bytes
        } else {
            0
        };
        self.unet_bytes.resident + self.vae_bytes.resident + controls + ip
    }

    fn text(&self) -> u64 {
        self.te_bytes[0].resident + self.te_bytes[1].resident
    }

    /// Price `regime`: the per-phase pricing and every `(phase, resident, label, scratch)` quantum.
    pub(crate) fn price(&self, regime: &Regime) -> Result<(IosPricing, Vec<PricedQuantum>)> {
        let mut items: Vec<PricedQuantum> = Vec::new();
        let mut take = |phase: IosPhase, resident: u64, quanta: Quanta| {
            for (label, scratch) in quanta {
                items.push(PricedQuantum {
                    phase,
                    resident,
                    label,
                    scratch,
                });
            }
        };
        let staged = regime.staged;
        let warm_text = if staged { 0 } else { self.text() };

        // Load floors in load order: the text encoders (staged: into an empty process), then the
        // heavy bundle — the U-Net, the VAE, the ControlNets (each at the compute dtype until the
        // bundle's tier quantization), the IP image encoder and IP weights.
        take(
            IosPhase::Load,
            0,
            vec![(q::TEXT_ENCODER_1_LOAD, self.te_bytes[0].floor)],
        );
        take(
            IosPhase::Load,
            self.te_bytes[0].resident,
            vec![(q::TEXT_ENCODER_2_LOAD, self.te_bytes[1].floor)],
        );
        let mut loaded = warm_text;
        take(
            IosPhase::Load,
            loaded,
            vec![(q::UNET_LOAD, self.unet_bytes.floor)],
        );
        loaded += self.unet_bytes.resident;
        let vae_label = if self.vae.dtype() == Dtype::Float32 {
            q::VAE_LOAD
        } else {
            q::VAE_FP16_FIX_LOAD
        };
        take(
            IosPhase::Load,
            loaded,
            vec![(vae_label, self.vae_bytes.floor)],
        );
        loaded += self.vae_bytes.resident;
        for (load, _) in &self.control_bytes {
            take(
                IosPhase::Load,
                loaded,
                vec![(q::CONTROLNET_LOAD, load.floor)],
            );
            loaded += load.resident;
        }
        if self.ip.is_some() {
            take(
                IosPhase::Load,
                loaded,
                vec![(q::IP_IMAGE_ENCODER_LOAD, self.ip_encoder_bytes.floor)],
            );
            loaded += self.ip_encoder_bytes.resident;
            take(
                IosPhase::Load,
                loaded,
                vec![(q::IP_ADAPTER_LOAD, self.ip_adapter_bytes.floor)],
            );
        }

        let heavy = self.heavy(false);
        // Conditioning: both CLIP towers over the guided pair's two token rows, one window at a
        // time (each window is its own forward).
        let rows = 2;
        let mut text = Quanta::new();
        let (h1, _) = self
            .te
            .0
            .price_forward(rows, self.arch.text_tokens, &mut text);
        let (h2, pooled) = self
            .te
            .1
            .price_forward(rows, self.arch.text_tokens, &mut text);
        let text_resident = self.text() + if staged { 0 } else { heavy };
        take(IosPhase::Conditioning, text_resident, text);
        let conditioning_dtype = Dtype::from_promoting_types(h1, h2);

        let (h, w) = (regime.height, regime.width);
        let (lh, lw) = (h / self.arch.latent_scale, w / self.arch.latent_scale);
        let (ch, cw) = (lh * self.arch.control_scale, lw * self.arch.control_scale);
        let ip_tokens = match (&self.ip, regime.route) {
            (Some(ip), Route::IpAdapter) => {
                let mut quanta = Quanta::new();
                let tokens = ip.price_tokens(&mut quanta)?;
                take(IosPhase::Conditioning, heavy + warm_text, quanta);
                let t = tokens.shape();
                Some(q::shaped(&[rows, t[1], t[2]], tokens.dtype())?)
            }
            _ => None,
        };

        // Encode: the img2img / inpaint init latent, or each ControlNet's conditioning embedding.
        let mut cond_embeds = Vec::new();
        if regime.route.encodes_init() {
            let mut quanta = Quanta::new();
            let tile = regime
                .tiling
                .as_ref()
                .and_then(|cfg| cfg.spatial.as_ref().map(|s| s.tile_px))
                .unwrap_or(crate::vae::IOS_ENCODE_TILE_PX);
            self.vae.price_encode(
                &q::shaped(&[1, h, w, 3], Dtype::Float32)?,
                tile,
                &mut quanta,
            )?;
            take(IosPhase::Encode, heavy + warm_text, quanta);
        }
        if regime.route == Route::Control {
            let mut quanta = Quanta::new();
            for control in &self.controls {
                cond_embeds.push(control.price_embed_cond(
                    &q::shaped(&[rows, ch, cw, 3], Dtype::Float32)?,
                    &mut quanta,
                )?);
            }
            take(IosPhase::Encode, heavy + warm_text, quanta);
        }

        // Denoise: one guided step (two sequential batch-1 halves, or one batched pair), the
        // ControlNet branches beside the U-Net, the IP tokens in the cross-attention.
        let latent = if regime.route.encodes_init() {
            Dtype::Float32
        } else {
            crate::model::DTYPE
        };
        let sequential = !regime.cfg_batching.is_batched();
        let batch = if sequential { 1 } else { rows };
        let plan = crate::plan::SdxlForwardPlan::UNBOUNDED
            .with_cfg_batching(regime.cfg_batching)
            .with_cfg_row(sequential.then_some(0));
        let encoder_x = q::shaped(
            &[rows, regime.context_tokens, self.arch.context_dim],
            conditioning_dtype,
        )?;
        let text_emb = q::shaped(&[rows, self.te.1.pooled_width()], pooled)?;
        // Step 1 runs the U-Net on the init latent's dtype; every later step on the sampler's
        // output, which the step promotes to the U-Net output's dtype (a packed tier's f32) — so
        // the denoise prices every distinct latent dtype the loop feeds the U-Net.
        let mut quanta = Quanta::new();
        let mut step_dtype = latent;
        loop {
            let eps = self.price_step(
                regime,
                q::shaped(&[batch, lh, lw, 4], step_dtype)?,
                &encoder_x,
                &text_emb,
                &cond_embeds,
                ip_tokens.as_ref(),
                plan,
                &mut quanta,
            )?;
            let next = Dtype::from_promoting_types(step_dtype, eps);
            if next == step_dtype {
                break;
            }
            step_dtype = next;
        }
        let latents = q::shaped(&[1, lh, lw, 4], step_dtype)?;
        quanta.push((
            q::SAMPLER_STEP,
            q::sampler_step_bytes(&latents, regime.route == Route::Mask),
        ));
        let shed_ip =
            crate::model::sheds_ip_image_encoder(staged) && regime.route == Route::IpAdapter;
        take(IosPhase::Denoise, self.heavy(shed_ip) + warm_text, quanta);

        // Decode: the staged decode holds the VAE alone; a warm one decodes beside everything.
        let mut quanta = Quanta::new();
        let decoded = self
            .vae
            .price_decode(&latents, regime.tiling.as_ref(), &mut quanta)?;
        quanta.push((q::DECODED_TO_IMAGE, q::decoded_to_image_bytes(&decoded)));
        let decode_resident = if staged {
            self.vae_bytes.resident
        } else {
            heavy + warm_text
        };
        take(IosPhase::Decode, decode_resident, quanta);

        let mut pricing = IosPricing::default();
        for item in &items {
            pricing.take(item.phase, item.resident, item.scratch, item.label);
        }
        Ok((pricing, items))
    }

    /// One guided denoise forward's quanta over the shape-only latent `x` (the ControlNet branches
    /// beside the U-Net, the IP tokens in the cross-attention). Returns the U-Net output's dtype.
    #[allow(clippy::too_many_arguments)]
    fn price_step(
        &self,
        regime: &Regime,
        x: Array,
        encoder_x: &Array,
        text_emb: &Array,
        cond_embeds: &[Array],
        ip_tokens: Option<&Array>,
        plan: crate::plan::SdxlForwardPlan<'_>,
        quanta: &mut Quanta,
    ) -> Result<Dtype> {
        let batch = x.shape()[0];
        let mut control: Option<(Vec<Array>, Array)> = None;
        if regime.route == Route::Control {
            for (branch, embed) in self.controls.iter().zip(cond_embeds) {
                let s = embed.shape();
                let row = q::shaped(&[batch, s[1], s[2], s[3]], embed.dtype())?;
                let (down, mid) =
                    branch.price_forward(&x, &row, encoder_x, text_emb, plan, quanta)?;
                control = Some(match control {
                    None => (down, mid),
                    Some((prev_down, prev_mid)) => (
                        prev_down
                            .iter()
                            .zip(&down)
                            .map(|(a, b)| {
                                q::shaped(
                                    a.shape(),
                                    Dtype::from_promoting_types(a.dtype(), b.dtype()),
                                )
                            })
                            .collect::<Result<_>>()?,
                        q::shaped(
                            prev_mid.shape(),
                            Dtype::from_promoting_types(prev_mid.dtype(), mid.dtype()),
                        )?,
                    ),
                });
            }
        }
        self.unet.price_forward(
            &x,
            encoder_x,
            text_emb,
            control.as_ref(),
            ip_tokens.map(|t| (t, 1.0)),
            plan,
            quanta,
        )
    }
}

/// One priced quantum: the phase it runs in, the component bytes resident around it, its label and
/// its counted scratch.
#[derive(Clone, Debug, PartialEq, Eq)]
pub(crate) struct PricedQuantum {
    pub phase: IosPhase,
    pub resident: u64,
    pub label: &'static str,
    pub scratch: u64,
}
