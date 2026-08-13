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

use candle_gen::candle_core::{DType, Device, Tensor};
use candle_gen::candle_nn::VarBuilder;
use candle_gen::gen_core::{
    self, AdapterSpec, Capabilities, Conditioning, ConditioningKind, GenerationOutput,
    GenerationRequest, Generator, Image, LoadSpec, Modality, ModelDescriptor, Progress, Quant,
    SizeFloor, WeightsSource,
};
use candle_gen::{CandleError, Result as CResult};
use candle_gen_wan::config::{TextEncoderConfig, Vae16Config, MAX_AREA_14B};
use candle_gen_wan::scheduler::Sampler;
use candle_gen_wan::text_encoder::Umt5Encoder;
use candle_gen_wan::vae16::WanVae16;

use crate::clip::{ClipVisionConfig, ScailClip};
use crate::config::Scail2Config;
use crate::generate::{align, CharacterRef, Components, Scail2Job, DIM_ALIGN};
use crate::model::Scail2Dit;

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

/// Stable identity + advertised capabilities for SCAIL-2 (Wan2.1-14B I2V end-to-end character
/// animation: reference image + driving video + color-coded masks → animated/identity-replaced video;
/// plain single-scale CFG; packed-token conditioning + per-source RoPE + CLIP image cross-attn).
/// `backend = "candle"`, `mac_only = false` (the off-Mac CUDA lane).
pub fn descriptor() -> ModelDescriptor {
    ModelDescriptor {
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
        },
    }
}

/// Load all `.safetensors` in the snapshot subdir `sub` as one f32 mmapped [`VarBuilder`].
fn component_vb(root: &Path, device: &Device, sub: &str) -> CResult<VarBuilder<'static>> {
    candle_gen::component_vb(root, sub, DType::F32, device, "scail2")
}

/// The loaded SCAIL-2 model: resolved config + snapshot dir, with the heavy components (DiT / VAE /
/// UMT5 / CLIP) loaded lazily on first generate and cached.
pub struct Scail2 {
    descriptor: ModelDescriptor,
    config: Scail2Config,
    root: PathBuf,
    device: Device,
    /// Inference adapters (LoRA / LoKr / LoHa / lightx2v lightning diff-patch) folded into the DiT
    /// before build; empty for the stock path (sc-6838).
    adapters: Vec<AdapterSpec>,
    components: Mutex<Option<Arc<Components>>>,
}

impl Scail2 {
    /// Build the DiT [`VarBuilder`] over the `transformer/` snapshot. With no adapters this is the
    /// stock f32 mmap build — **byte-identical** to the pre-sc-6838 path (the empty-adapter regression
    /// gate). With adapters, the base tensors are loaded to a CPU map, each delta is folded in
    /// ([`crate::adapters::merge_adapters`], f32 math — merge not residual, the chaos-sensitive-sampler
    /// rationale), the **whole map is cast to f32 on the CPU**, then the DiT is built from it.
    ///
    /// The host-side f32 cast is load-bearing for memory: SCAIL-2's DiT is f32, so a bf16 base tensor
    /// served through `from_tensors(F32, gpu)` would cast bf16→f32 *on the GPU*, and candle's CUDA
    /// caching allocator retains the freed bf16 staging blocks — ~28 GiB piled on top of the ~56 GiB
    /// f32 DiT, OOM-ing at the VAE-decode peak even on a 96 GiB card. Casting host-side (host RAM is
    /// ample, the map is transient) makes `get` a pure f32 host→device move, so the GPU footprint
    /// matches the stock mmap path exactly. (The Wan-14B merge path doesn't need this — its DiT is
    /// bf16, so `from_tensors` never casts on the GPU.)
    fn transformer_vb(&self) -> CResult<VarBuilder<'static>> {
        if self.adapters.is_empty() {
            return component_vb(&self.root, &self.device, "transformer");
        }
        let dir = self.root.join("transformer");
        let files = candle_gen::sorted_safetensors(&dir, "scail2")?;
        let mut tensors: HashMap<String, Tensor> = HashMap::new();
        for f in &files {
            let part = candle_gen::candle_core::safetensors::load(f, &Device::Cpu)?;
            tensors.extend(part);
        }
        // Discard the merge report — the silent twin (`candle-gen-z-image`'s
        // `transformer_vb_with_adapters`) does the same; a mismatched adapter surface already errors
        // inside `merge_adapters`, so library code stays quiet on stderr (sc-9035 / F-051).
        crate::adapters::merge_adapters(&mut tensors, &self.adapters)?;
        // Cast host-side so `from_tensors` does no GPU-side bf16→f32 staging (see the doc note above).
        for v in tensors.values_mut() {
            if v.dtype() != DType::F32 {
                *v = v.to_dtype(DType::F32)?;
            }
        }
        Ok(VarBuilder::from_tensors(tensors, DType::F32, &self.device))
    }

    fn load_components(&self) -> CResult<Components> {
        let te = Umt5Encoder::new(
            &TextEncoderConfig::umt5_xxl(),
            component_vb(&self.root, &self.device, "text_encoder")?,
        )?;
        let dit = Scail2Dit::new(self.transformer_vb()?, &self.config)?;
        let vae = WanVae16::new_with_encoder(
            &Vae16Config::wan21(),
            component_vb(&self.root, &self.device, "vae")?,
        )?;
        let clip = ScailClip::new(
            component_vb(&self.root, &self.device, "clip")?,
            &ClipVisionConfig::vit_h_14(),
        )?;
        let tok = crate::generate::build_tokenizer(&self.root, &TextEncoderConfig::umt5_xxl())?;
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

/// Construct a candle SCAIL-2 generator. `spec.weights` must be a [`WeightsSource::Dir`] pointing at a
/// snapshot with `text_encoder/`, `transformer/` (the converted SCAIL2Model DiT), `vae/` (z16 Wan VAE
/// with encoder), `clip/` (open-CLIP ViT-H/14 visual tower), and `tokenizer/tokenizer.json`. Inference
/// adapters (`spec.adapters` — LoRA / LoKr / LoHa / lightx2v lightning diff-patch / Bias-Aware DPO) are
/// merged into the dense DiT before build (sc-6838); on-the-fly quantization is still rejected.
pub fn load(spec: &LoadSpec) -> gen_core::Result<Box<dyn Generator>> {
    let root = match &spec.weights {
        WeightsSource::Dir(p) => p.clone(),
        WeightsSource::File(_) => {
            return Err(gen_core::Error::Msg(
                "scail2: expected a snapshot directory (text_encoder/ transformer/ vae/ clip/ \
                 tokenizer/), not a single .safetensors file"
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
    let config = Scail2Config::from_model_dir(&root)?;
    let device = candle_gen::default_device()?;
    Ok(Box::new(Scail2 {
        descriptor: descriptor(),
        config,
        root,
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
