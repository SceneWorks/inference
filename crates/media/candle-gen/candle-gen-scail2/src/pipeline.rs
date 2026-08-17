//! SCAIL-2 provider: capability surface, registration, snapshot/config resolution, and the
//! [`Generator`] entrypoint — the candle (Windows/CUDA) sibling of `mlx-gen-scail2`'s pipeline.
//!
//! [`Generator::generate`] maps the [`GenerationRequest`] conditioning onto the SCAIL-2 inputs and runs
//! the live `crate::generate` denoise pipeline: the primary **reference character** is a
//! [`Conditioning::Reference`] image paired with its color-coded [`Conditioning::Mask`]; the **driving
//! video + per-frame color masks** are a `ControlClip`; `video_mode == "replacement"` toggles the
//! cross-identity `replace_flag` (else animation). Inference adapters (`spec.adapters`) — LoRA / LoKr /
//! LoHa, the lightx2v lightning diff-patch, and the Bias-Aware DPO refinement LoRA — are folded into the
//! dense DiT before build ([`crate::adapters`], sc-6838). Multi-reference awaits the worker request
//! contract (sc-5583: gen-core has no way to pair an extra reference image with its color-coded mask —
//! `Conditioning::MultiReference` carries images only); `crate::generate` already supports extra
//! characters via [`crate::generate::CharacterRef`], so until that contract lands `MultiReference` is
//! deliberately NOT advertised and [`Generator::validate`] rejects it loudly (sc-8985).

use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex};

use candle_gen::candle_core::{
    safetensors as cst, DType, Device, Error as CoreError, Result as CoreResult, Shape, Tensor,
};
use candle_gen::candle_nn::var_builder::{Rename, SimpleBackend};
use candle_gen::candle_nn::{Init, VarBuilder};
use candle_gen::gen_core::{
    self, AdapterSpec, Capabilities, Conditioning, ConditioningKind, GenerationOutput,
    GenerationRequest, Generator, Image, LoadSpec, Modality, ModelDescriptor, Progress, Quant,
    SizeFloor, WeightsSource,
};
use candle_gen::{CandleError, Result as CResult};
use candle_gen_wan::config::{TextEncoderConfig, Vae16Config, MAX_AREA_14B};
use candle_gen_wan::scheduler::Sampler;
use candle_gen_wan::text_encoder::Umt5Encoder;
use cst::Load;

use crate::clip::{ClipVisionConfig, ScailClip};
use crate::config::Scail2Config;
use crate::generate::{align, CharacterRef, Components, Scail2Job, DIM_ALIGN};
use crate::model::Scail2Dit;
use crate::ProviderVae;

/// Default driving-segment window + clean-history overlap (upstream `scail.py` defaults).
const SEGMENT_LEN: usize = 81;
const SEGMENT_OVERLAP: usize = 5;
/// Upstream `generate()` sampler defaults: 40 steps, shift 5.0, guide 5.0, 16 fps.
const DEFAULT_STEPS: u32 = 40;
const DEFAULT_SHIFT: f32 = 5.0;
const DEFAULT_GUIDANCE: f32 = 5.0;
const DEFAULT_FPS: u32 = 16;

/// SceneWorks/engine model id (matches `mlx-gen-scail2` so a consumer resolves the same engine across
/// backends). A still image is `num_frames == 1`.
pub const MODEL_ID: &str = "scail2_14b";

/// The exact files in one self-contained `SceneWorks/scail2-mlx` tier. The MLX and candle providers
/// consume these same bytes; q4/q8 remain MLX-only because their DiT is MLX-packed, while the dense
/// bf16 tier is backend-shared.
pub const SHARED_TIER_FILES: &[&str] = &[
    "config.json",
    "dit.safetensors",
    "t5_encoder.safetensors",
    "tokenizer.json",
    "clip.safetensors",
    "vae.safetensors",
];

/// On-disk SCAIL-2 layouts accepted by the candle provider.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum SnapshotLayout {
    /// Flat, self-contained tier from `SceneWorks/scail2-mlx` (the Model Manager package).
    SharedMlxTier,
    /// Original candle-only component tree retained for explicit/manual overrides.
    LegacyComponents,
}

/// Classify a SCAIL-2 snapshot without loading tensor bytes. Completeness fails closed: a partial
/// shared tier is not mistaken for the legacy layout, and a legacy tree must contain every component
/// the provider opens.
pub fn snapshot_layout(root: &Path) -> gen_core::Result<SnapshotLayout> {
    if SHARED_TIER_FILES
        .iter()
        .all(|file| root.join(file).is_file())
    {
        return Ok(SnapshotLayout::SharedMlxTier);
    }
    let legacy_dirs = ["transformer", "text_encoder", "vae", "clip"];
    if legacy_dirs.iter().all(|sub| root.join(sub).is_dir())
        && root.join("tokenizer/tokenizer.json").is_file()
    {
        return Ok(SnapshotLayout::LegacyComponents);
    }
    let missing_shared: Vec<&str> = SHARED_TIER_FILES
        .iter()
        .copied()
        .filter(|file| !root.join(file).is_file())
        .collect();
    let mut missing_legacy: Vec<String> = legacy_dirs
        .iter()
        .filter(|sub| !root.join(sub).is_dir())
        .map(|sub| format!("{sub}/"))
        .collect();
    if !root.join("tokenizer/tokenizer.json").is_file() {
        missing_legacy.push("tokenizer/tokenizer.json".to_owned());
    }
    Err(gen_core::Error::Msg(format!(
        "scail2: incomplete snapshot at {}: shared SceneWorks/scail2-mlx tier is missing [{}]; \
         legacy candle layout is missing [{}]",
        root.display(),
        missing_shared.join(", "),
        missing_legacy.join(", ")
    )))
}

/// Stable identity + advertised capabilities for SCAIL-2 (Wan2.1-14B I2V end-to-end character
/// animation: reference image + driving video + color-coded masks → animated/identity-replaced video;
/// plain single-scale CFG; packed-token conditioning + per-source RoPE + CLIP image cross-attn).
/// `backend = "candle"`, `mac_only = false` (the off-Mac CUDA lane).
pub fn descriptor() -> ModelDescriptor {
    ModelDescriptor {
        encoder_contract: None,
        denoiser_output_latent_space: Some(&candle_gen::gen_core::WAN_Z16_VIDEO_LATENT_SPACE),
        control_kinds: None,
        required_components: &[],
        id: MODEL_ID,
        family: "scail2",
        backend: "candle",
        modality: Modality::Video,
        capabilities: Capabilities {
            supports_negative_prompt: true,
            supports_guidance: true,
            supports_true_cfg: false,
            // Reference character image (Reference) + its color-coded segmentation mask (Mask); the
            // driving video + its per-frame color masks map to ControlClip. `MultiReference` (extra
            // characters) is deliberately NOT advertised: gen-core's `Conditioning::MultiReference`
            // carries images only, with no way to pair each extra reference with its required
            // color-coded mask, so the request contract can't reach `Scail2Job.additional` yet —
            // sc-5583 tracks the paired ref+mask contract + worker plumbing (sc-8985: advertising it
            // let multi-ref requests validate, render for minutes, and silently drop the extras).
            conditioning: vec![
                ConditioningKind::Reference,
                ConditioningKind::Mask,
                ConditioningKind::ControlClip,
            ],
            // Inference LoRA / LoKr / LoHa + the lightx2v lightning diff-patch + the Bias-Aware DPO
            // refinement LoRA, merged into the dense DiT before build (sc-6838,
            // [`crate::adapters::merge_adapters`]).
            supports_lora: true,
            supports_lokr: true,
            // candle's FlowScheduler is UniPC/Euler; "dpm++" resolves to UniPC (bh2). Advertised to
            // match the mlx-gen-scail2 descriptor for cross-backend routing parity.
            samplers: vec!["unipc", "dpm++"],
            schedulers: Vec::new(),
            supported_guidance_methods: vec![],
            min_size: 32,
            max_size: 1280,
            max_count: 1,
            mac_only: false,
            supported_quants: &[] as &[Quant],
            component_precision_floors: &[],
            supports_kv_cache: false,
            requires_sigma_shift: false,
            supports_sequential_offload: false,
            unconditionally_engages_staged_residency: false,
            supports_preview: false,
            supports_prompt_enhancement: false,
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
            // Match the MLX sibling: `0×0` resolves from the driving clip, while every explicit
            // request remains exact-or-rejected on SCAIL-2's 32-pixel render lattice (sc-16199).
            size_floor: SizeFloor::ResolvedDownstreamExplicitGrid {
                multiple: DIM_ALIGN,
            },
            execution: Default::default(),
        },
    }
}

/// Load all `.safetensors` in the snapshot subdir `sub` as one f32 mmapped [`VarBuilder`].
fn component_vb(root: &Path, device: &Device, sub: &str) -> CResult<VarBuilder<'static>> {
    candle_gen::component_vb(root, sub, DType::F32, device, "scail2")
}

/// A safetensors backend that performs precision widening on the CPU before the final device
/// transfer. Candle's stock mmap backend does these operations in the opposite order
/// (`load(name, target_device)?.to_dtype(dtype)`), which turns one bf16 SCAIL tensor into a bf16 CUDA
/// staging allocation followed by its f32 resident allocation. The CUDA caching allocator retains
/// the freed staging blocks; across the 14B DiT that is roughly 28 GiB of avoidable device pressure.
///
/// This backend remains bounded: it maps the checkpoint, materializes one requested tensor on CPU,
/// casts that tensor on CPU, uploads the final f32 bytes, and drops the host tensor when `get` returns.
/// It therefore does not need the whole 47.2 GB shared package in host memory at once.
struct CpuCastMmap {
    tensors: cst::MmapedSafetensors,
}

impl CpuCastMmap {
    fn load_cpu(&self, name: &str, dtype: DType) -> CoreResult<Tensor> {
        let tensor = self.tensors.load(name, &Device::Cpu)?;
        let tensor = if tensor.dtype() == dtype {
            tensor
        } else {
            tensor.to_dtype(dtype)?
        };
        debug_assert!(tensor.device().is_cpu());
        Ok(tensor)
    }
}

impl SimpleBackend for CpuCastMmap {
    fn get(
        &self,
        shape: Shape,
        name: &str,
        _: Init,
        dtype: DType,
        device: &Device,
    ) -> CoreResult<Tensor> {
        let tensor = self.load_cpu(name, dtype)?;
        if tensor.shape() != &shape {
            return Err(CoreError::Msg(format!(
                "scail2: shape mismatch for {name}: expected {shape:?}, got {:?}",
                tensor.shape()
            )));
        }
        tensor.to_device(device)
    }

    fn get_unchecked(&self, name: &str, dtype: DType, device: &Device) -> CoreResult<Tensor> {
        self.load_cpu(name, dtype)?.to_device(device)
    }

    fn contains_tensor(&self, name: &str) -> bool {
        self.tensors.get(name).is_ok()
    }
}

/// Build a bounded mmap var-builder whose source tensors are always cast on CPU before upload.
fn cpu_cast_mmap_var_builder(
    files: &[PathBuf],
    dtype: DType,
    device: &Device,
) -> CResult<VarBuilder<'static>> {
    // SAFETY: the same invariant as `candle_gen::mmap_var_builder`: these are process-owned,
    // read-only model files and are not mutated or truncated while the mapping is live.
    let tensors = unsafe { cst::MmapedSafetensors::multi(files)? };
    Ok(VarBuilder::from_backend(
        Box::new(CpuCastMmap { tensors }),
        dtype,
        device.clone(),
    ))
}

/// Load a checkpoint to a CPU tensor map and widen every tensor to f32 there. Adapter merging and the
/// shared VAE key remap require a mutable whole-model map, so they cannot use the bounded backend;
/// this helper gives them the same no-bf16-on-CUDA invariant as the stock dense path.
fn cpu_f32_tensor_map(files: &[PathBuf]) -> CResult<HashMap<String, Tensor>> {
    // SAFETY: the same read-only, process-owned model-file invariant as the builder above.
    let tensors = unsafe { cst::MmapedSafetensors::multi(files)? };
    let mut out = HashMap::new();
    for (name, view) in tensors.tensors() {
        let tensor = view.load(&Device::Cpu)?;
        let tensor = if tensor.dtype() == DType::F32 {
            tensor
        } else {
            tensor.to_dtype(DType::F32)?
        };
        debug_assert!(tensor.device().is_cpu());
        out.insert(name, tensor);
    }
    Ok(out)
}

/// Translate the candle/Hugging Face UMT5 key requested by [`Umt5Encoder`] to the normalized key in
/// the shared MLX tier's `t5_encoder.safetensors`. This is a pure key projection: tensor bytes and
/// shapes are unchanged.
fn shared_umt5_key(key: &str) -> String {
    if key == "shared.weight" {
        return "token_embedding.weight".to_owned();
    }
    if key == "encoder.final_layer_norm.weight" {
        return "norm.weight".to_owned();
    }
    let Some(rest) = key.strip_prefix("encoder.block.") else {
        return key.to_owned();
    };
    let Some((block, leaf)) = rest.split_once('.') else {
        return key.to_owned();
    };
    let mapped = match leaf {
        "layer.0.layer_norm.weight" => "norm1.weight",
        "layer.0.SelfAttention.q.weight" => "attn.q.weight",
        "layer.0.SelfAttention.k.weight" => "attn.k.weight",
        "layer.0.SelfAttention.v.weight" => "attn.v.weight",
        "layer.0.SelfAttention.o.weight" => "attn.o.weight",
        "layer.0.SelfAttention.relative_attention_bias.weight" => "pos_embedding.embedding.weight",
        "layer.1.layer_norm.weight" => "norm2.weight",
        "layer.1.DenseReluDense.wi_0.weight" => "ffn.gate_proj.weight",
        "layer.1.DenseReluDense.wi_1.weight" => "ffn.fc1.weight",
        "layer.1.DenseReluDense.wo.weight" => "ffn.fc2.weight",
        _ => return key.to_owned(),
    };
    format!("blocks.{block}.{mapped}")
}

fn shared_umt5_vb(root: &Path, device: &Device) -> CResult<VarBuilder<'static>> {
    let file = root.join("t5_encoder.safetensors");
    let inner = cpu_cast_mmap_var_builder(std::slice::from_ref(&file), DType::F32, device)?;
    let renamer: Box<dyn Fn(&str) -> String + Send + Sync> = Box::new(shared_umt5_key);
    Ok(VarBuilder::from_backend(
        Box::new(Rename::new(inner, renamer)),
        DType::F32,
        device.clone(),
    ))
}

fn shared_vae_vb(root: &Path, device: &Device) -> CResult<VarBuilder<'static>> {
    let map = cpu_f32_tensor_map(&[root.join("vae.safetensors")])?;
    let map = candle_gen::remap_vae_wan_mlx_to_diffusers(map)?;
    Ok(VarBuilder::from_tensors(map, DType::F32, device))
}

/// The loaded SCAIL-2 model: resolved config + snapshot dir, with the heavy components (DiT / VAE /
/// UMT5 / CLIP) loaded lazily on first generate and cached.
pub struct Scail2 {
    descriptor: ModelDescriptor,
    config: Scail2Config,
    root: PathBuf,
    layout: SnapshotLayout,
    device: Device,
    /// Inference adapters (LoRA / LoKr / LoHa / lightx2v lightning diff-patch) folded into the DiT
    /// before build; empty for the stock path (sc-6838).
    adapters: Vec<AdapterSpec>,
    components: Mutex<Option<Arc<Components>>>,
}

impl Scail2 {
    /// Build the DiT [`VarBuilder`] over the shared flat file or legacy `transformer/` snapshot. The
    /// stock path streams one tensor at a time through the CPU f32 cast; with adapters, the base
    /// tensors are loaded to a CPU map and each delta is folded in
    /// ([`crate::adapters::merge_adapters`], f32 math — merge not residual, the chaos-sensitive-sampler
    /// rationale), the **whole map is cast to f32 on the CPU**, then the DiT is built from it.
    ///
    /// The host-side f32 cast is load-bearing for memory: SCAIL-2's DiT is f32, so a bf16 base tensor
    /// served through Candle's stock mmap backend would upload bf16 and cast bf16→f32 *on the GPU*,
    /// and candle's CUDA
    /// caching allocator retains the freed bf16 staging blocks — ~28 GiB piled on top of the ~56 GiB
    /// f32 DiT, OOM-ing at the VAE-decode peak even on a 96 GiB card. Casting host-side (host RAM is
    /// ample, the map is transient) makes `get` a pure f32 host→device move. The Wan-14B merge path
    /// doesn't need this because its DiT is bf16, so `from_tensors` never casts on the GPU.
    fn transformer_vb(&self) -> CResult<VarBuilder<'static>> {
        let files = match self.layout {
            SnapshotLayout::SharedMlxTier => vec![self.root.join("dit.safetensors")],
            SnapshotLayout::LegacyComponents => {
                candle_gen::sorted_safetensors(&self.root.join("transformer"), "scail2")?
            }
        };
        if self.adapters.is_empty() {
            return cpu_cast_mmap_var_builder(&files, DType::F32, &self.device);
        }
        let mut tensors = cpu_f32_tensor_map(&files)?;
        // Discard the merge report — the silent twin (`candle-gen-z-image`'s
        // `transformer_vb_with_adapters`) does the same; a mismatched adapter surface already errors
        // inside `merge_adapters`, so library code stays quiet on stderr (sc-9035 / F-051).
        crate::adapters::merge_adapters(&mut tensors, &self.adapters)?;
        // Cast host-side so `from_tensors` does no GPU-side bf16→f32 staging (see the doc note above).
        for v in tensors.values_mut() {
            if v.dtype() != DType::F32 {
                *v = v.to_dtype(DType::F32)?;
            }
            debug_assert!(v.device().is_cpu());
        }
        Ok(VarBuilder::from_tensors(tensors, DType::F32, &self.device))
    }

    fn load_components(&self) -> CResult<Components> {
        let te_vb = match self.layout {
            SnapshotLayout::SharedMlxTier => shared_umt5_vb(&self.root, &self.device)?,
            SnapshotLayout::LegacyComponents => {
                component_vb(&self.root, &self.device, "text_encoder")?
            }
        };
        let te = Umt5Encoder::new(&TextEncoderConfig::umt5_xxl(), te_vb)?;
        let dit = Scail2Dit::new(self.transformer_vb()?, &self.config)?;
        let vae_vb = match self.layout {
            SnapshotLayout::SharedMlxTier => shared_vae_vb(&self.root, &self.device)?,
            SnapshotLayout::LegacyComponents => component_vb(&self.root, &self.device, "vae")?,
        };
        let vae = ProviderVae::new_with_encoder(&Vae16Config::wan21(), vae_vb)?;
        let clip_vb = match self.layout {
            SnapshotLayout::SharedMlxTier => cpu_cast_mmap_var_builder(
                &[self.root.join("clip.safetensors")],
                DType::F32,
                &self.device,
            )?,
            SnapshotLayout::LegacyComponents => component_vb(&self.root, &self.device, "clip")?,
        };
        let clip = ScailClip::new(clip_vb, &ClipVisionConfig::vit_h_14())?;
        let tok = match self.layout {
            SnapshotLayout::SharedMlxTier => crate::generate::build_tokenizer_from_path(
                self.root.join("tokenizer.json"),
                &TextEncoderConfig::umt5_xxl(),
            )?,
            SnapshotLayout::LegacyComponents => {
                crate::generate::build_tokenizer(&self.root, &TextEncoderConfig::umt5_xxl())?
            }
        };
        Ok(Components {
            te,
            dit,
            vae,
            clip,
            tok,
        })
    }

    fn components(&self) -> CResult<Arc<Components>> {
        candle_gen::cached(&self.components, || Ok(Arc::new(self.load_components()?)))
    }
}

/// Construct a candle SCAIL-2 generator. `spec.weights` must be a [`WeightsSource::Dir`] containing
/// either the flat dense `SceneWorks/scail2-mlx` bf16 tier in [`SHARED_TIER_FILES`] or the legacy
/// candle component tree (`text_encoder/`, `transformer/`, `vae/`, `clip/`, and
/// `tokenizer/tokenizer.json`). Inference
/// adapters (`spec.adapters` — LoRA / LoKr / LoHa / lightx2v lightning diff-patch / Bias-Aware DPO) are
/// merged into the dense DiT before build (sc-6838); on-the-fly quantization is still rejected.
pub fn load(spec: &LoadSpec) -> gen_core::Result<Box<dyn Generator>> {
    let root = match &spec.weights {
        WeightsSource::Dir(p) => p.clone(),
        WeightsSource::File(_) => {
            return Err(gen_core::Error::Msg(
                "scail2: expected a snapshot directory (a shared SceneWorks/scail2-mlx bf16 tier or \
                 the legacy text_encoder/ transformer/ vae/ clip/ tokenizer/ tree), not a single \
                 .safetensors file"
                    .into(),
            ));
        }
    };
    if spec.quantize.is_some() {
        return Err(gen_core::Error::Unsupported(
            "candle scail2 does not support on-the-fly Q4/Q8 quantization yet".into(),
        ));
    }
    if !root.exists() {
        return Err(gen_core::Error::Msg(format!(
            "scail2: snapshot dir does not exist: {}",
            root.display()
        )));
    }
    let layout = snapshot_layout(&root)?;
    let config = Scail2Config::from_model_dir(&root)?;
    let device = candle_gen::default_device()?;
    Ok(Box::new(Scail2 {
        descriptor: descriptor(),
        config,
        root,
        layout,
        device,
        adapters: spec.adapters.clone(),
        components: Mutex::new(None),
    }))
}

candle_gen::register_generators! { pub(crate) const REGISTRATION = descriptor => load }

impl Generator for Scail2 {
    fn descriptor(&self) -> &ModelDescriptor {
        &self.descriptor
    }

    fn validate(&self, req: &GenerationRequest) -> gen_core::Result<()> {
        // Actionable multi-reference rejection first, so the caller learns WHY (pending contract,
        // sc-5583) rather than the generic capability-floor "not supported" from `validate_request`
        // (which also rejects it now that `MultiReference` is unadvertised, sc-8985).
        reject_multi_reference(self.descriptor.id, req)?;
        reject_zero_steps(self.descriptor.id, req)?;
        self.descriptor
            .capabilities
            .validate_request(self.descriptor.id, req)?;
        // The shared floor can validate an explicit size but cannot see what the `0×0` sentinel
        // resolves to. Bound that driving-frame geometry here so preflight and render agree.
        resolve_pre_flight_size(req).map_or(Ok(()), |(width, height)| {
            reject_unrenderable_geometry(&self.descriptor.capabilities, req, width, height)
        })
    }

    fn generate(
        &self,
        req: &GenerationRequest,
        on_progress: &mut dyn FnMut(Progress),
    ) -> gen_core::Result<GenerationOutput> {
        self.validate(req)?;
        Ok(self.run(req, on_progress)?)
    }
}

/// Reject `MultiReference` conditioning loudly instead of letting a multi-character request render
/// for minutes and silently drop the extra characters (sc-8985). The engine core already supports
/// extra characters ([`crate::generate::CharacterRef`] / `Scail2Job.additional`), but gen-core's
/// `Conditioning::MultiReference` carries images only — there is no way to pair each extra reference
/// with its required color-coded segmentation mask until the paired ref+mask request contract lands
/// (sc-5583).
fn reject_multi_reference(id: &str, req: &GenerationRequest) -> gen_core::Result<()> {
    if req
        .conditioning
        .iter()
        .any(|c| matches!(c, Conditioning::MultiReference { .. }))
    {
        return Err(gen_core::Error::Unsupported(format!(
            "{id}: MultiReference (extra reference characters) is not supported yet — each extra \
             character needs its own color-coded segmentation mask and the paired reference+mask \
             request contract is pending (sc-5583); pass exactly one Reference + Mask"
        )));
    }
    Ok(())
}

/// Reject an explicit `steps: Some(0)` loudly instead of running zero denoise iterations and
/// VAE-decoding the pure prior — on video that is MINUTES of GPU time for garbage (sc-9016, F-032).
/// Mirrors the registered `SdxlGenerator::validate` steps floor; this worker-driven video path has no
/// gen-core steps floor upstream of it. A `None` legitimately falls through to `DEFAULT_STEPS`.
fn reject_zero_steps(id: &str, req: &GenerationRequest) -> gen_core::Result<()> {
    if req.steps == Some(0) {
        return Err(gen_core::Error::Msg(format!(
            "{id}: steps must be >= 1 (an explicit 0 renders undenoised noise)"
        )));
    }
    Ok(())
}

/// The geometry the pipeline will render: explicit dimensions where non-zero, otherwise the first
/// driving frame. Only the exact `0×0` pair reaches this policy through the advertised floor; the
/// per-axis form keeps this helper identical to the assignment in [`Scail2::run`].
fn resolve_target_size(req: &GenerationRequest, first: &Image) -> (u32, u32) {
    (
        if req.width > 0 {
            req.width
        } else {
            first.width
        },
        if req.height > 0 {
            req.height
        } else {
            first.height
        },
    )
}

/// [`resolve_target_size`] for preflight, where conditioning may not have been assembled yet.
fn resolve_pre_flight_size(req: &GenerationRequest) -> Option<(u32, u32)> {
    if req.width > 0 && req.height > 0 {
        return Some((req.width, req.height));
    }
    req.control_clip()
        .and_then(|clip| clip.frames.first())
        .map(|first| resolve_target_size(req, first))
}

/// Resolve, bound, and lattice-project the geometry consumed by [`Scail2Job`].
fn resolve_render_size(
    caps: &Capabilities,
    req: &GenerationRequest,
    first: &Image,
) -> gen_core::Result<(u32, u32)> {
    let (width, height) = resolve_target_size(req, first);
    reject_unrenderable_geometry(caps, req, width, height)?;
    if req.width == 0 && req.height == 0 {
        Ok((align(width) as u32, align(height) as u32))
    } else {
        Ok((width, height))
    }
}

/// Reject a requested or driving-video-resolved geometry outside SCAIL-2's advertised edge and area
/// envelope. Sentinel-resolved edges and every area are measured after projection onto the same
/// lattice the render uses; source media can be off-grid, while explicit off-grid requests have
/// already failed the shared floor and retain exact requested-edge policy here.
fn reject_unrenderable_geometry(
    caps: &Capabilities,
    req: &GenerationRequest,
    width: u32,
    height: u32,
) -> gen_core::Result<()> {
    let auto_sized = req.width == 0 && req.height == 0;
    let origin = if auto_sized {
        "resolved from the driving video"
    } else {
        "requested"
    };
    let (bounded_width, bounded_height) = if auto_sized {
        (align(width) as u32, align(height) as u32)
    } else {
        (width, height)
    };
    let outside_range = bounded_width < caps.min_size
        || bounded_width > caps.max_size
        || bounded_height < caps.min_size
        || bounded_height > caps.max_size;

    if !outside_range {
        return reject_over_area(MODEL_ID, width, height);
    }

    let reason = format!(
        "each edge must be within {}..={}",
        caps.min_size, caps.max_size
    );
    let advice = match suggest_in_envelope(width, height, caps.min_size, caps.max_size) {
        Some((suggested_width, suggested_height)) => format!(
            "pass an explicit width/height — the largest geometry this engine accepts at this \
             aspect is {suggested_width}×{suggested_height} (the same driving clip is resized to \
             whatever target you set)"
        ),
        None => "pass an explicit width/height inside the advertised range".to_string(),
    };
    Err(gen_core::Error::Msg(format!(
        "{MODEL_ID}: {width}×{height} ({origin}) is outside this model's advertised size envelope — \
         {reason}; {advice}"
    )))
}

/// Largest on-lattice geometry at the source aspect that fits the advertised edge and area bounds.
/// Integer arithmetic avoids a float round-trip dropping an extra lattice step at `x.999…`.
fn suggest_in_envelope(
    width: u32,
    height: u32,
    min_size: u32,
    max_size: u32,
) -> Option<(u32, u32)> {
    if width == 0 || height == 0 {
        return None;
    }
    let step = u64::from(DIM_ALIGN);
    let clamp = |value: u64| u32::try_from(value).unwrap_or(u32::MAX);
    let floor = min_size.max(DIM_ALIGN);
    let mut suggested_width = clamp(u64::from(width.min(max_size)) / step * step);
    while suggested_width >= floor {
        let numerator = u64::from(suggested_width) * u64::from(height);
        let denominator = u64::from(width);
        let rounded_up = clamp(numerator.div_ceil(denominator * step) * step);
        let rounded_down = clamp(numerator / denominator / step * step);
        for suggested_height in [rounded_up, rounded_down] {
            if suggested_height >= min_size
                && suggested_height <= max_size
                && suggested_width as usize * suggested_height as usize <= MAX_AREA_14B
            {
                return Some((suggested_width, suggested_height));
            }
        }
        suggested_width -= DIM_ALIGN;
    }
    None
}

/// Reject an over-area request loudly instead of letting the 14B DiT run for minutes and OOM. SCAIL-2's
/// DiT runs **f32** (≈ 56 GiB resident) with a packed conditioning sequence >2× the plain token count,
/// so a far-over-envelope request (e.g. 1280×1280×81) validates and dies with an opaque CUDA OOM at the
/// VAE-decode peak. Reject past the shared A14B cap with an actionable message, mirroring the A14B MoE
/// lane (`wan14b.rs`, sc-9028 / F-044); the incident class F-090 (sc-11215) left this lane open. `max_size`
/// alone only bounds each edge, so 1280×1280 (both ≤ 1280) slips through without the area check.
///
/// sc-16197 made this helper measure the lattice-projected geometry rather than the raw product.
/// `1280×730`, for example, projects to `1280×704`. sc-16198 then advertised that same
/// [`DIM_ALIGN`] lattice through the descriptor's explicit-size policy. Sentinel-resolved source
/// geometry can be off-grid, so [`reject_unrenderable_geometry`] also calls this helper before the
/// render projects it to the lattice.
fn reject_over_area(id: &str, width: u32, height: u32) -> gen_core::Result<()> {
    let (w, h) = (align(width), align(height));
    let area = w * h;
    if area > MAX_AREA_14B {
        // Report the geometry the gate measured, and name the snap when it happened. Quoting the raw
        // product alone would send an off-lattice caller hunting for an off-by-one that isn't there.
        // (The wording is candle-only: `mlx-gen-scail2` reaches the same end by naming the requested
        // geometry in the head of its message and the rendered one in the reason clause.)
        let snapped = if (w, h) == (width as usize, height as usize) {
            String::new()
        } else {
            format!(
                " (the requested {}×{} snaps onto the {DIM_ALIGN}-px lattice)",
                width, height
            )
        };
        return Err(gen_core::Error::Msg(format!(
            "{id}: width×height ({w}×{h} = {area} px){snapped} exceeds the max area \
             {MAX_AREA_14B} px (1280×720); reduce the resolution"
        )));
    }
    Ok(())
}

/// The first conditioning input matching `f`.
fn find_conditioning<'a, T>(
    req: &'a GenerationRequest,
    f: impl Fn(&'a Conditioning) -> Option<T>,
) -> Option<T> {
    req.conditioning.iter().find_map(f)
}

impl Scail2 {
    /// Map the request conditioning onto a [`Scail2Job`] and run the denoise pipeline.
    fn run(
        &self,
        req: &GenerationRequest,
        on_progress: &mut dyn FnMut(Progress),
    ) -> CResult<GenerationOutput> {
        let reference = find_conditioning(req, |c| match c {
            Conditioning::Reference { image, .. } => Some(image),
            _ => None,
        })
        .ok_or_else(|| {
            CandleError::Msg("scail2: a Reference character image is required".into())
        })?;
        let ref_mask = find_conditioning(req, |c| match c {
            Conditioning::Mask { image } => Some(image),
            _ => None,
        })
        .ok_or_else(|| {
            CandleError::Msg(
                "scail2: a Mask (the reference character's color-coded segmentation mask) is required"
                    .into(),
            )
        })?;
        let driving = req.control_clip().ok_or_else(|| {
            CandleError::Msg(
                "scail2: a ControlClip (driving video frames + per-frame color masks) is required"
                    .into(),
            )
        })?;

        let first: &Image = driving.frames.first().ok_or_else(|| {
            CandleError::Msg("scail2: the ControlClip has no driving frames".into())
        })?;
        // The shared floor exempts `0×0` because it cannot inspect the driving clip. Resolve and
        // enforce the edge/area envelope before loading components, then project source media to the
        // same render lattice used by the MLX sibling.
        let (width, height) = resolve_render_size(&self.descriptor.capabilities, req, first)?;

        let neg = req.negative_prompt.clone().unwrap_or_default();
        let job = Scail2Job {
            prompt: &req.prompt,
            negative_prompt: &neg,
            width,
            height,
            reference: CharacterRef {
                image: reference,
                mask: ref_mask,
            },
            // Extra characters await the paired ref+mask request contract (sc-5583); `validate`
            // rejects `MultiReference` until then (sc-8985).
            additional: Vec::new(),
            driving_frames: driving.frames,
            driving_masks: driving.mask,
            replace_flag: req.video_mode.as_deref() == Some("replacement"),
            seed: req.seed.unwrap_or_else(gen_core::default_seed),
            steps: req.steps.unwrap_or(DEFAULT_STEPS) as usize,
            shift: req.scheduler_shift.unwrap_or(DEFAULT_SHIFT) as f64,
            guidance: req.guidance.unwrap_or(DEFAULT_GUIDANCE) as f64,
            sampler: Sampler::parse(req.sampler.as_deref()),
            fps: req.fps.unwrap_or(DEFAULT_FPS),
            segment_len: SEGMENT_LEN,
            segment_overlap: SEGMENT_OVERLAP,
        };
        let comps = self.components()?;
        let te_cfg = TextEncoderConfig::umt5_xxl();
        crate::generate::generate(&comps, &te_cfg, &job, &req.cancel, on_progress)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::BTreeSet;

    fn touch(path: &Path) {
        if let Some(parent) = path.parent() {
            std::fs::create_dir_all(parent).unwrap();
        }
        std::fs::write(path, b"").unwrap();
    }

    #[test]
    fn shared_model_manager_tier_is_complete_or_fails_closed() {
        let tmp = tempfile::tempdir().unwrap();
        for file in SHARED_TIER_FILES {
            touch(&tmp.path().join(file));
        }
        assert_eq!(
            snapshot_layout(tmp.path()).unwrap(),
            SnapshotLayout::SharedMlxTier
        );

        std::fs::remove_file(tmp.path().join("t5_encoder.safetensors")).unwrap();
        let error = snapshot_layout(tmp.path()).unwrap_err().to_string();
        assert!(error.contains("t5_encoder.safetensors"), "got: {error}");
        assert!(error.contains("text_encoder/"), "got: {error}");
    }

    #[test]
    fn legacy_component_layout_remains_supported() {
        let tmp = tempfile::tempdir().unwrap();
        for sub in ["transformer", "text_encoder", "vae", "clip"] {
            std::fs::create_dir_all(tmp.path().join(sub)).unwrap();
        }
        touch(&tmp.path().join("tokenizer/tokenizer.json"));
        assert_eq!(
            snapshot_layout(tmp.path()).unwrap(),
            SnapshotLayout::LegacyComponents
        );
    }

    #[test]
    fn shared_umt5_projection_covers_every_loaded_tensor_key() {
        let mut requested = Vec::new();
        requested.push("shared.weight".to_owned());
        requested.push("encoder.final_layer_norm.weight".to_owned());
        let leaves = [
            "layer.0.layer_norm.weight",
            "layer.0.SelfAttention.q.weight",
            "layer.0.SelfAttention.k.weight",
            "layer.0.SelfAttention.v.weight",
            "layer.0.SelfAttention.o.weight",
            "layer.0.SelfAttention.relative_attention_bias.weight",
            "layer.1.layer_norm.weight",
            "layer.1.DenseReluDense.wi_0.weight",
            "layer.1.DenseReluDense.wi_1.weight",
            "layer.1.DenseReluDense.wo.weight",
        ];
        for block in 0..TextEncoderConfig::umt5_xxl().num_layers {
            requested.extend(
                leaves
                    .iter()
                    .map(|leaf| format!("encoder.block.{block}.{leaf}")),
            );
        }
        assert_eq!(requested.len(), 242, "UMT5-XXL loads exactly 242 tensors");
        let projected: BTreeSet<String> =
            requested.iter().map(|key| shared_umt5_key(key)).collect();
        assert_eq!(
            projected.len(),
            requested.len(),
            "projection must be bijective"
        );
        assert!(projected.contains("token_embedding.weight"));
        assert!(projected.contains("blocks.23.pos_embedding.embedding.weight"));
        assert!(projected.contains("blocks.23.ffn.fc2.weight"));
        assert!(projected.contains("norm.weight"));
        assert_eq!(
            shared_umt5_key("encoder.block.0.unexpected.weight"),
            "encoder.block.0.unexpected.weight",
            "unknown keys stay loud at the underlying missing-tensor gate"
        );
    }

    fn write_bf16_safetensors(path: &Path) {
        let cpu = Device::Cpu;
        let mut tensors = HashMap::new();
        tensors.insert(
            "weight".to_owned(),
            Tensor::new(&[[1.0_f32, 2.0], [3.0, 4.0]], &cpu)
                .unwrap()
                .to_dtype(DType::BF16)
                .unwrap(),
        );
        tensors.insert(
            "bias".to_owned(),
            Tensor::new(&[5.0_f32, 6.0], &cpu)
                .unwrap()
                .to_dtype(DType::BF16)
                .unwrap(),
        );
        cst::save(&tensors, path).unwrap();
    }

    #[test]
    fn shared_dense_loader_casts_on_cpu_and_releases_source_mapping() {
        let tmp = tempfile::tempdir().unwrap();
        let source = tmp.path().join("source.safetensors");
        let moved = tmp.path().join("moved.safetensors");
        write_bf16_safetensors(&source);

        let map = cpu_f32_tensor_map(std::slice::from_ref(&source)).unwrap();
        assert_eq!(map.len(), 2);
        for tensor in map.values() {
            assert_eq!(tensor.dtype(), DType::F32);
            assert!(tensor.device().is_cpu());
        }
        // The returned CPU/F32 map owns its tensors rather than retaining a live file mapping. This
        // is especially load-bearing on Windows, where an open mmap would make the rename fail.
        std::fs::rename(&source, &moved).unwrap();
        assert_eq!(
            map["weight"]
                .flatten_all()
                .unwrap()
                .to_vec1::<f32>()
                .unwrap(),
            vec![1.0, 2.0, 3.0, 4.0]
        );
    }

    #[test]
    fn bounded_shared_backend_returns_final_dtype_without_retaining_bf16_tensor() {
        let tmp = tempfile::tempdir().unwrap();
        let source = tmp.path().join("source.safetensors");
        write_bf16_safetensors(&source);

        // Exercise the backend's pre-upload seam directly: the only tensor it can hand to the
        // target-device transfer is already CPU/F32.
        let mapped =
            unsafe { cst::MmapedSafetensors::multi(std::slice::from_ref(&source)) }.unwrap();
        let backend = CpuCastMmap { tensors: mapped };
        let prepared = backend.load_cpu("weight", DType::F32).unwrap();
        assert_eq!(prepared.dtype(), DType::F32);
        assert!(prepared.device().is_cpu());
        assert_eq!(prepared.dims(), &[2, 2]);

        // And exercise the exact VarBuilder path used by dense DiT/T5/CLIP. A mutation back to the
        // stock mmap backend makes the source-structure gate below fail even though CPU-only CI
        // cannot observe the intermediate CUDA allocation.
        let vb = cpu_cast_mmap_var_builder(std::slice::from_ref(&source), DType::F32, &Device::Cpu)
            .unwrap();
        let loaded = vb.get((2, 2), "weight").unwrap();
        assert_eq!(loaded.dtype(), DType::F32);
        assert!(loaded.device().is_cpu());
    }

    #[test]
    fn every_shared_component_avoids_the_stock_device_then_cast_backend() {
        let source = include_str!("pipeline.rs")
            .split("#[cfg(test)]")
            .next()
            .expect("production source before tests");
        let forbidden = [
            "SnapshotLayout::SharedMlxTier => candle_gen::",
            "mmap_var_builder",
        ]
        .concat();
        assert!(
            !source.contains(&forbidden),
            "a shared component must never upload its stored dtype before widening to f32"
        );
        for required in [
            "let inner = cpu_cast_mmap_var_builder",
            "return cpu_cast_mmap_var_builder(&files",
            "SnapshotLayout::SharedMlxTier => cpu_cast_mmap_var_builder",
            "let map = cpu_f32_tensor_map(&[root.join(\"vae.safetensors\")])",
            "remap_vae_wan_mlx_to_diffusers(map)",
        ] {
            assert!(
                source.contains(required),
                "shared dense component lost the CPU-first loader seam: {required}"
            );
        }
        assert!(
            source.contains("let mut tensors = cpu_f32_tensor_map(&files)?"),
            "the adapter DiT must begin from the same CPU/F32 invariant as the stock path"
        );
    }

    #[test]
    fn shared_cuda_profile_reuses_one_stable_idle_policy_and_validates_before_publish() {
        const SOURCE: &str = include_str!("pipeline.rs");
        const WORKFLOW: &str = include_str!("../../../../../.github/workflows/real-weights.yml");
        let profile_start = SOURCE
            .rfind("\n    fn shared_bf16_real_weights_cuda_loads_and_renders_with_measured_peak()")
            .expect("exact SCAIL profile declaration");
        let profile_end = SOURCE[profile_start..]
            .find("\n    fn registers_and_resolves_as_candle_video()")
            .map(|offset| profile_start + offset)
            .expect("profile boundary");
        let profile = &SOURCE[profile_start..profile_end];

        for required in [
            "const STABLE_IDLE: candle_gen::testkit::StableIdleConfig =",
            ".assert_stable_idle(STABLE_IDLE)",
            ".assert_trustworthy(STABLE_IDLE.max_baseline_gb)",
        ] {
            assert!(
                profile.contains(required),
                "missing shared idle policy: {required}"
            );
        }
        assert_eq!(
            profile.matches("STABLE_IDLE").count(),
            3,
            "the one stable-idle config must be declared once and consumed by both validations"
        );
        assert!(
            !profile.contains("assert_trustworthy(1.0)"),
            "the legacy headless ceiling must not override the validated WDDM policy"
        );

        let raw = profile
            .find("[[SCAIL2_CUDA_VRAM_RAW]]")
            .expect("raw diagnostic marker");
        let validation = profile
            .find(".assert_trustworthy(STABLE_IDLE.max_baseline_gb)")
            .expect("post-report validation");
        let validated = profile
            .find("[[SCAIL2_CUDA_VRAM]]")
            .expect("validated publication marker");
        assert!(raw < validation && validation < validated);
        assert!(
            WORKFLOW.contains("grep -Fq '[[SCAIL2_CUDA_VRAM]]'"),
            "workflow must accept only the final validated marker"
        );
        assert!(
            !WORKFLOW.contains("grep -Fq '[[SCAIL2_CUDA_VRAM_RAW]]'"),
            "raw diagnostics must never satisfy publication"
        );
    }

    /// Exact shared-package CUDA gate. The official real-weight workflow provisions
    /// `SceneWorks/scail2-mlx@ce88cfdb.../bf16`, runs this test on an otherwise-idle CUDA device,
    /// and records both driver-reserved (the admission unit) and concurrent-live pool peaks.
    #[cfg(feature = "cuda")]
    #[test]
    #[ignore = "needs the exact 47.2 GB shared SCAIL bf16 package and an idle >=96 GB CUDA GPU"]
    fn shared_bf16_real_weights_cuda_loads_and_renders_with_measured_peak() {
        const REPOSITORY: &str = "SceneWorks/scail2-mlx";
        const REVISION: &str = "ce88cfdb1008f395e9c820e525e6db7b6695f7b3";
        const WIDTH: u32 = 832;
        const HEIGHT: u32 = 480;
        const FRAMES: usize = 81;
        const STEPS: u32 = 1;
        const STABLE_IDLE: candle_gen::testkit::StableIdleConfig =
            candle_gen::testkit::StableIdleConfig::new(2.0, 6, 64, 200);

        let root = PathBuf::from(
            std::env::var("SCAIL2_SHARED_BF16_DIR")
                .expect("set SCAIL2_SHARED_BF16_DIR to the exact Model Manager bf16 tier"),
        );
        let canonical = root.canonicalize().expect("canonical shared snapshot path");
        let normalized = canonical.to_string_lossy().replace('\\', "/");
        let expected_suffix = format!("models--SceneWorks--scail2-mlx/snapshots/{REVISION}/bf16");
        assert!(
            normalized.ends_with(&expected_suffix),
            "snapshot must be the exact fixed repository/revision/tier, got {normalized}"
        );
        assert_eq!(
            snapshot_layout(&canonical).unwrap(),
            SnapshotLayout::SharedMlxTier
        );

        // Bind the runtime proof to the exact complete VAE header. The FNV-1a input is the sorted
        // `name<TAB>dtype<TAB>dimxdim...<LF>` stream; this guards all 194 names and source shapes
        // without checking a hand-selected subset. The loader independently validates every tensor
        // class and its post-transpose weight/bias relationship before device construction.
        let vae_mapped = unsafe { cst::MmapedSafetensors::new(canonical.join("vae.safetensors")) }
            .expect("map exact VAE header");
        let mut vae_specs = vae_mapped
            .tensors()
            .into_iter()
            .map(|(name, view)| {
                format!(
                    "{name}\t{:?}\t{}\n",
                    view.dtype(),
                    view.shape()
                        .iter()
                        .map(usize::to_string)
                        .collect::<Vec<_>>()
                        .join("x")
                )
            })
            .collect::<Vec<_>>();
        vae_specs.sort_unstable();
        assert_eq!(vae_specs.len(), 194, "exact SCAIL VAE tensor count");
        let vae_header_fnv = vae_specs
            .concat()
            .bytes()
            .fold(0xcbf29ce484222325_u64, |hash, byte| {
                (hash ^ u64::from(byte)).wrapping_mul(0x100000001b3)
            });
        assert_eq!(
            // Canonical LF is part of the digest input. Do not derive this through a platform text
            // pipeline: PowerShell's CRLF rendering produces a different value for the same header.
            vae_header_fnv,
            0xa7b83f24867477ab,
            "exact SceneWorks/scail2-mlx@ce88 VAE name/dtype/shape header drift"
        );

        let mut source_dtypes = Vec::new();
        for file in [
            "dit.safetensors",
            "t5_encoder.safetensors",
            "vae.safetensors",
            "clip.safetensors",
        ] {
            let path = canonical.join(file);
            let mapped = unsafe { cst::MmapedSafetensors::new(&path) }.unwrap();
            let mut counts = std::collections::BTreeMap::<String, usize>::new();
            for (_, view) in mapped.tensors() {
                *counts.entry(format!("{:?}", view.dtype())).or_default() += 1;
            }
            source_dtypes.push(format!("{file}={counts:?}"));
        }
        assert!(
            source_dtypes[0].contains("BF16"),
            "DiT must exercise widening"
        );
        assert!(
            source_dtypes[1].contains("BF16"),
            "T5 must exercise widening"
        );
        assert!(
            source_dtypes[2].contains("F32"),
            "VAE is the shared f32 companion"
        );
        assert!(
            source_dtypes[3].contains("F32"),
            "CLIP is the shared f32 companion"
        );

        let pool =
            candle_gen::cuda_mempool::MemPool::device_default(0).expect("CUDA default memory pool");
        assert!(pool.reset_high_water(), "reset CUDA pool high-water marks");
        // The self-hosted Windows lane's isolated GPU has a stable ~1.6 GB WDDM/UI baseline even
        // when pmon shows no pure compute process. Prove that exact condition instead of either
        // accepting a one-shot busy sample or pretending this runner is a headless <1 GB device.
        let mut probe =
            candle_gen::testkit::VramProbe::start_rendered().assert_stable_idle(STABLE_IDLE);

        let load_phase = probe.phase();
        let load_started = std::time::Instant::now();
        // Use the public provider entry point that worker dispatch calls. `load` deliberately keeps
        // component construction lazy, so its settled residency is near zero; the generate phase
        // below covers both real component materialization and the smallest valid render.
        let model = load(&LoadSpec::new(WeightsSource::Dir(canonical)))
            .expect("production SCAIL provider accepts the shared package");
        probe.end_load(load_phase);
        let load_seconds = load_started.elapsed().as_secs_f64();

        let image = |rgb: [u8; 3]| Image {
            width: WIDTH,
            height: HEIGHT,
            pixels: std::iter::repeat_n(rgb, (WIDTH * HEIGHT) as usize)
                .flatten()
                .collect(),
        };
        let driving_frame = image([96, 128, 160]);
        let driving_mask = image([0, 0, 255]);
        let req = GenerationRequest {
            prompt: "a character follows the driving motion".to_owned(),
            negative_prompt: Some(String::new()),
            width: WIDTH,
            height: HEIGHT,
            count: 1,
            steps: Some(STEPS),
            guidance: Some(5.0),
            seed: Some(18473),
            fps: Some(16),
            video_mode: Some("animation".to_owned()),
            conditioning: vec![
                Conditioning::Reference {
                    image: image([120, 80, 40]),
                    strength: None,
                },
                Conditioning::Mask {
                    image: image([0, 0, 255]),
                },
                Conditioning::ControlClip {
                    frames: vec![driving_frame; FRAMES],
                    mask: vec![driving_mask; FRAMES],
                    masking_strength: 1.0,
                    start_frame: 0,
                    mode: Default::default(),
                },
            ],
            ..Default::default()
        };
        let render_phase = probe.phase();
        let render_started = std::time::Instant::now();
        let output = model
            .generate(&req, &mut |_| {})
            .expect("minimal production-provider render");
        probe.end_gen(render_phase);
        let render_seconds = render_started.elapsed().as_secs_f64();
        let GenerationOutput::Video { frames, fps, .. } = output else {
            panic!("SCAIL must return video");
        };
        assert_eq!(frames.len(), FRAMES);
        assert_eq!(fps, 16);
        assert!(frames
            .iter()
            .any(|frame| frame.pixels.iter().any(|&v| v != 0)));

        let report = probe.report();
        let used_high = pool.used_high().expect("USED_MEM_HIGH") as f64 / 1.0e9;
        let reserved_high = pool.reserved_high().expect("RESERVED_MEM_HIGH") as f64 / 1.0e9;
        eprintln!(
            "[[SCAIL2_CUDA_VRAM_RAW]] baselineGb={:.3} loadPeakGb={:.3} steadyGb={:.3} \
             overallPeakGb={:.3} poolUsedHighGb={used_high:.3} poolReservedHighGb={reserved_high:.3}",
            report.baseline_gb, report.load_peak_gb, report.steady_gb, report.peak_gb,
        );
        let report = report.assert_trustworthy(STABLE_IDLE.max_baseline_gb);
        assert!(
            report.peak_gb > 70.0,
            "real F32 stack was not observed through the production provider: {report}"
        );
        println!(
            "[[SCAIL2_CUDA_VRAM]] repository={REPOSITORY} revision={REVISION} tier=bf16 width={WIDTH} \
             height={HEIGHT} frames={FRAMES} steps={STEPS} baselineGb={:.3} loadPeakGb={:.3} \
             steadyGb={:.3} overallPeakGb={:.3} poolUsedHighGb={used_high:.3} \
             poolReservedHighGb={reserved_high:.3} loadSeconds={load_seconds:.3} \
             renderSeconds={render_seconds:.3} sourceDtypes=\"{}\"",
            report.baseline_gb,
            report.load_peak_gb,
            report.steady_gb,
            report.peak_gb,
            source_dtypes.join(";"),
        );
    }

    #[test]
    fn registers_and_resolves_as_candle_video() {
        let spec = LoadSpec::new(WeightsSource::Dir("/nonexistent".into()));
        // The snapshot dir doesn't exist, so `load` errors — but the engine must be REGISTERED (the
        // registry resolves the id to this provider's `load`).
        let err = crate::provider_registry()
            .unwrap()
            .load(MODEL_ID, &spec)
            .err()
            .expect("dir missing");
        assert!(
            err.to_string().contains("does not exist"),
            "expected a missing-dir error from the scail2 loader, got: {err}"
        );
    }

    #[test]
    fn descriptor_surface() {
        let d = descriptor();
        assert_eq!(d.id, MODEL_ID);
        assert_eq!(d.family, "scail2");
        assert_eq!(d.backend, "candle");
        assert_eq!(d.modality, Modality::Video);
        assert!(!d.capabilities.mac_only);
        assert!(d.capabilities.supports_guidance);
        assert!(d.capabilities.supports_negative_prompt);
        assert!(!d.capabilities.supports_true_cfg);
        assert!(d.capabilities.supports_lora);
        assert!(d.capabilities.supports_lokr);
        assert!(d.capabilities.accepts(ConditioningKind::Reference));
        assert!(d.capabilities.accepts(ConditioningKind::Mask));
        assert!(d.capabilities.accepts(ConditioningKind::ControlClip));
        // MultiReference is deliberately NOT advertised until the paired ref+mask request contract
        // lands (sc-5583) — advertising it silently dropped the extra characters (sc-8985).
        assert!(!d.capabilities.accepts(ConditioningKind::MultiReference));
        assert!(d.capabilities.samplers.contains(&"unipc"));
        assert_eq!(
            d.capabilities.size_floor.explicit_size_multiple(),
            Some(DIM_ALIGN)
        );
        assert_eq!(
            d.capabilities.size_floor,
            SizeFloor::ResolvedDownstreamExplicitGrid {
                multiple: DIM_ALIGN
            }
        );
    }

    #[test]
    fn explicit_off_grid_size_is_refused_weights_free() {
        for (width, height) in [(1280, 730), (730, 1280)] {
            let req = GenerationRequest {
                width,
                height,
                count: 1,
                ..Default::default()
            };

            let err = descriptor()
                .capabilities
                .validate_request(MODEL_ID, &req)
                .expect_err("an explicit off-grid size must be refused before loading")
                .to_string();
            assert!(
                err.contains("multiples of 32") && err.contains(&format!("{width}×{height}")),
                "the refusal must name the required grid and requested size, got: {err}"
            );
        }
    }

    #[test]
    fn multi_reference_is_rejected_loudly() {
        let img = Image {
            width: 64,
            height: 64,
            pixels: vec![0u8; 64 * 64 * 3],
        };
        let req = GenerationRequest {
            prompt: "a character".into(),
            width: 64,
            height: 64,
            count: 1,
            conditioning: vec![Conditioning::MultiReference {
                images: vec![img.clone(), img],
            }],
            ..Default::default()
        };
        // The dedicated guard fires with the actionable pending-contract message.
        let err = reject_multi_reference(MODEL_ID, &req).expect_err("err");
        assert!(matches!(err, gen_core::Error::Unsupported(_)), "got: {err}");
        let msg = err.to_string();
        assert!(msg.contains("MultiReference"), "got: {msg}");
        assert!(msg.contains("sc-5583"), "got: {msg}");
        // Backstop: with `MultiReference` unadvertised, the shared capability floor rejects it too.
        assert!(
            descriptor()
                .capabilities
                .validate_request(MODEL_ID, &req)
                .is_err(),
            "the capability floor must reject unadvertised MultiReference conditioning"
        );
        // A request without MultiReference passes the guard (the floor still enforces the rest).
        let single = GenerationRequest {
            conditioning: Vec::new(),
            ..req
        };
        assert!(reject_multi_reference(MODEL_ID, &single).is_ok());
    }

    #[test]
    fn zero_steps_is_rejected_loudly() {
        // An explicit `steps: Some(0)` is a fast, actionable error — NOT minutes of video decoded from
        // undenoised prior noise (sc-9016, F-032).
        let zero = GenerationRequest {
            prompt: "a character".into(),
            steps: Some(0),
            ..Default::default()
        };
        let err = reject_zero_steps(MODEL_ID, &zero).expect_err("steps==0 must be rejected");
        let msg = err.to_string();
        assert!(msg.contains("steps must be >= 1"), "got: {msg}");
        // A valid step count and an unset (default) step count both pass this guard.
        let valid = GenerationRequest {
            steps: Some(40),
            ..zero.clone()
        };
        assert!(reject_zero_steps(MODEL_ID, &valid).is_ok());
        let unset = GenerationRequest {
            steps: None,
            ..zero
        };
        assert!(reject_zero_steps(MODEL_ID, &unset).is_ok());
    }

    #[test]
    fn over_area_is_rejected_loudly() {
        // A far-over-envelope request (1280×1280, both edges ≤ `max_size` so `max_size` alone lets it
        // through) must be a fast, actionable rejection — NOT minutes of the f32 14B DiT running to an
        // opaque CUDA OOM (F-090 / sc-11215, mirroring the A14B MoE lane's sc-9028 guard).
        let over = GenerationRequest {
            prompt: "a character".into(),
            width: 1280,
            height: 1280,
            ..Default::default()
        };
        let err = reject_over_area(MODEL_ID, over.width, over.height)
            .expect_err("over-area must be rejected");
        let msg = err.to_string();
        assert!(msg.contains("max area"), "message names the cap: {msg}");
        // Both edges are already on the lattice, so nothing was snapped and the message must not
        // claim otherwise — the parenthetical is reserved for a request whose measured geometry
        // differs from the typed one.
        assert!(
            msg.contains("1280×1280 = 1638400 px") && !msg.contains("snaps onto"),
            "an on-lattice refusal quotes the geometry as typed, with no snap note: {msg}"
        );
        // The canonical 720p and a small in-bounds request both pass the guard. 1280×720 is
        // `MAX_AREA_14B` exactly *as typed* — it did not pass while this cap carried the TI2V-5B's
        // 901 120 (sc-12308) — but since sc-16197 the gate measures the rendered geometry, and
        // SCAIL-2's own lattice is 32, not the z16 VAE's 16: `720` floors to `704`, so what is
        // actually weighed here is 1280×704 = 901 120, with 20 480 px of slack. The strict-`>`
        // boundary duty therefore moved to the on-lattice 960×960 row in
        // `over_area_is_judged_on_the_aligned_geometry`; this row's job is now the headline
        // resolution passing, which the raw reading would have left one pixel of slack from refusing.
        assert_eq!(1280 * 720, MAX_AREA_14B);
        let at_cap = GenerationRequest {
            width: 1280,
            height: 720,
            ..over.clone()
        };
        assert!(reject_over_area(MODEL_ID, at_cap.width, at_cap.height).is_ok());
        let small = GenerationRequest {
            width: 512,
            height: 512,
            ..over
        };
        assert!(reject_over_area(MODEL_ID, small.width, small.height).is_ok());
    }

    /// The helper projects onto the render lattice before measuring the cap (sc-16197).
    ///
    /// The area helper projects every edge down to the [`DIM_ALIGN`] lattice. The provider now
    /// rejects these off-grid requests before calling it (sc-16198), but direct rows still pin the
    /// sc-16197 arithmetic independently: a regression to the raw product turns this test RED.
    ///
    /// Every row appears in **both orientations**. The fix aligns two edges, and a landscape-only
    /// table gates only one of them: with `1280` and `1216` already on the lattice, dropping the
    /// `align` from the *width* alone would leave a landscape-only table green. `mlx-gen-wan` pairs
    /// its own area rows the same way.
    #[test]
    fn over_area_is_judged_on_the_aligned_geometry() {
        for (w, h, aw, ah) in [
            // The story's headline row: 934 400 raw, 901 120 rendered.
            (1280u32, 730u32, 1280usize, 704usize),
            (730, 1280, 704, 1280),
            // …and one where neither edge is the one the headline row snapped.
            (1216, 760, 1216, 736),
            (760, 1216, 736, 1216),
        ] {
            // Pin the area helper's retained projection arithmetic independently of provider
            // validation and low-level generation.
            assert_eq!(
                (align(w), align(h)),
                (aw, ah),
                "{w}×{h} renders at {aw}×{ah}"
            );
            assert!(
                w as usize * h as usize > MAX_AREA_14B,
                "{w}×{h} must be over the cap AS TYPED, or this row proves nothing"
            );
            assert!(
                aw * ah <= MAX_AREA_14B,
                "{aw}×{ah} must be inside the cap once aligned"
            );
            reject_over_area(MODEL_ID, w, h)
                .unwrap_or_else(|e| panic!("{w}×{h} renders at {aw}×{ah}, inside the cap: {e}"));
        }

        // The band is a band, not a hole: once the ALIGNED geometry is over the cap, the request is
        // still refused — and the message reports the measured geometry plus the snap that produced
        // it, so the caller is not left hunting for an off-by-one against the number they typed.
        // Both orientations again, so the reported dims cannot come out transposed.
        for (w, h, aw, ah) in [
            (1280u32, 760u32, 1280usize, 736usize),
            (760, 1280, 736, 1280),
        ] {
            assert_eq!((align(w), align(h)), (aw, ah));
            assert!(aw * ah > MAX_AREA_14B, "{aw}×{ah} is over the cap");
            let err = reject_over_area(MODEL_ID, w, h)
                .expect_err("942 080 px is over the cap even after aligning");
            let msg = err.to_string();
            assert!(
                msg.contains(&format!("{aw}×{ah} = {} px", aw * ah)),
                "the refusal quotes the RENDERED geometry: {msg}"
            );
            assert!(
                msg.contains(&format!(
                    "the requested {w}×{h} snaps onto the 32-px lattice"
                )),
                "the refusal names the snap that produced it: {msg}"
            );
        }

        // The cap stays a strict `>`: 960×960 is on-lattice and EXACTLY `MAX_AREA_14B`, so aligning
        // changes nothing and it must still pass. (The mirror of `mlx-gen-scail2`'s own 960×960 row.)
        assert_eq!(960 * 960, MAX_AREA_14B);
        assert!(reject_over_area(MODEL_ID, 960, 960).is_ok());

        // [`align`]'s min-one-tile floor is load-bearing for the paragraph in `reject_over_area`'s
        // doc that reasons about it: a sub-lattice edge snaps UP to one tile, so the measured area
        // can only ever grow, and the `0×0` sentinel measures 32×32 = 1024 px rather than 0. Drop
        // the `.max(1)` and the doc's argument silently stops being true.
        assert_eq!((align(0), align(1), align(31)), (32, 32, 32));
        assert!(reject_over_area(MODEL_ID, 0, 0).is_ok());
        // …and that raise is not a hole: an edge under one tile still refuses when the OTHER edge
        // carries it past the cap. `1×30000` → `32×29984` = 959 488 px.
        assert_eq!((align(1), align(30000)), (32, 29984));
        assert!(reject_over_area(MODEL_ID, 1, 30000).is_err());
    }

    fn unloaded() -> Scail2 {
        Scail2 {
            descriptor: descriptor(),
            config: Scail2Config::default(),
            root: PathBuf::from("/nonexistent-scail2-snapshot"),
            layout: SnapshotLayout::LegacyComponents,
            device: Device::Cpu,
            adapters: Vec::new(),
            components: Mutex::new(None),
        }
    }

    fn img(width: u32, height: u32) -> Image {
        Image {
            width,
            height,
            pixels: Vec::new(),
        }
    }

    fn auto_sized(width: u32, height: u32) -> GenerationRequest {
        GenerationRequest {
            prompt: "a character".into(),
            width: 0,
            height: 0,
            count: 1,
            conditioning: vec![
                Conditioning::Reference {
                    image: img(64, 64),
                    strength: None,
                },
                Conditioning::Mask { image: img(64, 64) },
                Conditioning::ControlClip {
                    frames: vec![img(width, height)],
                    mask: vec![img(width, height)],
                    masking_strength: 1.0,
                    start_frame: 0,
                    mode: Default::default(),
                },
            ],
            ..Default::default()
        }
    }

    #[test]
    fn sentinel_is_advertised_without_weakening_explicit_size_validation() {
        let caps = descriptor().capabilities;
        assert!(
            caps.validate_request(MODEL_ID, &auto_sized(832, 480))
                .is_ok(),
            "0×0 must advertise the resolve-from-driving-video convention"
        );
        for (width, height) in [(0, 512), (512, 0), (1280, 730)] {
            let req = GenerationRequest {
                width,
                height,
                count: 1,
                ..Default::default()
            };
            assert!(
                caps.validate_request(MODEL_ID, &req).is_err(),
                "{width}×{height} is not the sentinel or an explicit on-grid size"
            );
        }
    }

    #[test]
    fn auto_size_bounds_the_resolved_geometry_in_preflight_and_render() {
        let m = unloaded();

        Generator::validate(&m, &auto_sized(832, 480))
            .expect("an in-envelope driving clip must clear preflight");

        for (width, height, suggestion, over_area) in [
            (3840, 2160, "1280×704", false),
            (1280, 1280, "960×960", true),
        ] {
            let over = auto_sized(width, height);
            let preflight = Generator::validate(&m, &over)
                .expect_err("unsafe resolved geometry must fail before model loading")
                .to_string();
            if over_area {
                assert_eq!(
                    preflight,
                    reject_over_area(MODEL_ID, width, height)
                        .expect_err("fixture is over-area")
                        .to_string(),
                    "the geometry seam must preserve the canonical area refusal verbatim"
                );
            } else {
                assert!(
                    preflight.contains("advertised size envelope")
                        && preflight.contains(&format!("{width}×{height}"))
                        && preflight.contains("resolved from the driving video")
                        && preflight.contains(suggestion),
                    "the edge refusal must name the unsafe resolved geometry and a usable \
                     alternative: {preflight}"
                );
            }

            let rendered = m
                .run(&over, &mut |_| {})
                .expect_err("the render path must independently bound resolved geometry")
                .to_string();
            assert_eq!(
                rendered, preflight,
                "preflight and render must expose the same geometry refusal"
            );
        }
    }

    #[test]
    fn auto_size_projects_source_media_onto_the_render_lattice() {
        for (source, rendered) in [((640, 360), (640, 352)), ((1290, 704), (1280, 704))] {
            let req = auto_sized(source.0, source.1);
            let first = req
                .control_clip()
                .and_then(|clip| clip.frames.first())
                .expect("fixture has a driving frame");
            assert_eq!(
                resolve_render_size(&descriptor().capabilities, &req, first)
                    .expect("the rendered geometry is inside the advertised envelope"),
                rendered
            );
        }

        let explicit = GenerationRequest {
            width: 1290,
            height: 704,
            ..Default::default()
        };
        let error = resolve_render_size(
            &descriptor().capabilities,
            &explicit,
            &img(explicit.width, explicit.height),
        )
        .expect_err("explicit geometry keeps the exact edge-bound policy")
        .to_string();
        assert!(error.contains("requested") && error.contains("each edge must be within"));
    }

    #[test]
    fn load_rejects_single_file_and_unwired_surfaces() {
        use candle_gen::gen_core::{AdapterKind, AdapterSpec};
        // single-file source
        let f = LoadSpec::new(WeightsSource::File("/tmp/w.safetensors".into()));
        assert!(load(&f).is_err());
        // LoRA adapters are now ACCEPTED (sc-6838) — `load` proceeds past the adapter check and fails
        // only on the missing snapshot dir, NOT with an Unsupported("LoRA") error.
        let lora = LoadSpec::new(WeightsSource::Dir("/nonexistent".into())).with_adapters(vec![
            AdapterSpec::new("/lora.safetensors".into(), 1.0, AdapterKind::Lora),
        ]);
        let err = load(&lora).err().expect("missing dir");
        assert!(
            !matches!(err, gen_core::Error::Unsupported(_)),
            "got: {err}"
        );
        assert!(err.to_string().contains("does not exist"), "got: {err}");
        // on-the-fly quant is still rejected
        let quant = LoadSpec::new(WeightsSource::Dir("/snap".into())).with_quant(Quant::Q8);
        assert!(matches!(
            load(&quant).err().expect("err"),
            gen_core::Error::Unsupported(_)
        ));
    }
}
