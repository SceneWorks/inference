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
            supports_kv_cache: false,
            requires_sigma_shift: false,
            supports_sequential_offload: false,
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
        reject_over_area(self.descriptor.id, req)?;
        self.descriptor
            .capabilities
            .validate_request(self.descriptor.id, req)
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

/// Reject an over-area request loudly instead of letting the 14B DiT run for minutes and OOM. SCAIL-2's
/// DiT runs **f32** (≈ 56 GiB resident) with a packed conditioning sequence >2× the plain token count,
/// so a far-over-envelope request (e.g. 1280×1280×81) validates and dies with an opaque CUDA OOM at the
/// VAE-decode peak. Reject past the shared A14B cap with an actionable message, mirroring the A14B MoE
/// lane (`wan14b.rs`, sc-9028 / F-044); the incident class F-090 (sc-11215) left this lane open. `max_size`
/// alone only bounds each edge, so 1280×1280 (both ≤ 1280) slips through without the area check.
///
/// **sc-16197 — the cap is measured on the geometry that actually renders**: each edge floored onto
/// the [`DIM_ALIGN`] lattice by the very [`align`] the denoise loop calls
/// ([`crate::generate::generate`]), not the raw `req.width × req.height`. Judging the raw product
/// refused requests this engine would then have rendered at a perfectly legal size — `1280×730` is
/// 934 400 px as typed but renders at `1280×704` = 901 120, comfortably inside the cap. `mlx-gen-wan`
/// settled this reading for the family (its `over_area_is_judged_on_the_aligned_geometry`) and
/// `mlx-gen-scail2` inherited it in sc-16167; this is candle joining them, so one manifest entry means
/// one thing on both backends.
///
/// The divergence band exists at all only because SCAIL-2 — alone in the wan family — has no off-grid
/// rejection: its siblings (`wan14b.rs`, `model_vace.rs`) refuse a non-multiple size *before* the area
/// check, so their raw and aligned areas are always equal. Whether SCAIL-2 should refuse an off-lattice
/// explicit size rather than snap it is sc-16198; this gate is correct either way, because on-lattice
/// input makes the alignment a no-op.
///
/// [`align`]'s min-one-tile floor (a sub-32 edge snaps *up* to 32 rather than down to 0) is carried
/// deliberately — it is what renders — and cannot hide an over-area request: it applies only to an edge
/// below one lattice step, and it only ever *raises* the measured area. It does move the `0×0` sentinel
/// from 0 px to 1 024 px, both far under the cap, so that path is unchanged in outcome.
///
/// This gate reads `req` rather than the resolved dims, which is only sound because the descriptor
/// declares [`SizeFloor::RangeChecked`]: `validate_request` refuses any edge below `min_size`, so `0×0`
/// never reaches [`Scail2::run`]'s resolve-from-the-driving-clip branch and the gate is never asked
/// about a geometry it cannot see. Were that floor relaxed (sc-16199 revisits exactly this), the
/// sentinel would measure 32×32 here while rendering at the clip's own size — so
/// `descriptor_declares_the_size_floor_this_gate_depends_on` pins the dependency rather than leaving it
/// implicit. `mlx-gen-wan` solves the same hazard structurally, with a dims-taking
/// `reject_over_area_dims`, because on that backend the sentinel *is* reachable.
fn reject_over_area(id: &str, req: &GenerationRequest) -> gen_core::Result<()> {
    let (w, h) = (align(req.width), align(req.height));
    let area = w * h;
    if area > MAX_AREA_14B {
        // Report the geometry the gate measured, and name the snap when it happened. Quoting the raw
        // product alone would send an off-lattice caller hunting for an off-by-one that isn't there.
        // (The wording is candle-only: `mlx-gen-scail2` reaches the same end by naming the requested
        // geometry in the head of its message and the rendered one in the reason clause.)
        let snapped = if (w, h) == (req.width as usize, req.height as usize) {
            String::new()
        } else {
            format!(
                " (the requested {}×{} snaps onto the {DIM_ALIGN}-px lattice)",
                req.width, req.height
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
        let width = if req.width > 0 {
            req.width
        } else {
            first.width
        };
        let height = if req.height > 0 {
            req.height
        } else {
            first.height
        };

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
        let err = reject_over_area(MODEL_ID, &over).expect_err("over-area must be rejected");
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
        assert!(reject_over_area(MODEL_ID, &at_cap).is_ok());
        let small = GenerationRequest {
            width: 512,
            height: 512,
            ..over
        };
        assert!(reject_over_area(MODEL_ID, &small).is_ok());
    }

    /// The cap measures the geometry the pipeline **renders**, not the one that was typed (sc-16197).
    ///
    /// SCAIL-2 snaps every edge down to the [`DIM_ALIGN`] lattice before rendering and — alone in the
    /// wan family — refuses no off-grid request first (sc-16198), so raw and aligned areas genuinely
    /// diverge here. Measuring the raw product refused requests that render legally; the rows below
    /// are exactly that band, and each one is over the cap *as typed*, so a regression to the raw
    /// measurement turns this test RED rather than leaving it green for the wrong reason.
    ///
    /// Every row appears in **both orientations**. The fix aligns two edges, and a landscape-only
    /// table gates only one of them: with `1280` and `1216` already on the lattice, dropping the
    /// `align` from the *width* alone would leave a landscape-only table green. `mlx-gen-wan` pairs
    /// its own area rows the same way.
    #[test]
    fn over_area_is_judged_on_the_aligned_geometry() {
        let req = |w, h| GenerationRequest {
            prompt: "a character".into(),
            width: w,
            height: h,
            ..Default::default()
        };
        for (w, h, aw, ah) in [
            // The story's headline row: 934 400 raw, 901 120 rendered.
            (1280u32, 730u32, 1280usize, 704usize),
            (730, 1280, 704, 1280),
            // …and one where neither edge is the one the headline row snapped.
            (1216, 760, 1216, 736),
            (760, 1216, 736, 1216),
        ] {
            // The gate's alignment is `generate`'s own, so it cannot drift from what renders.
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
            reject_over_area(MODEL_ID, &req(w, h))
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
            let err = reject_over_area(MODEL_ID, &req(w, h))
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
        assert!(reject_over_area(MODEL_ID, &req(960, 960)).is_ok());

        // [`align`]'s min-one-tile floor is load-bearing for the paragraph in `reject_over_area`'s
        // doc that reasons about it: a sub-lattice edge snaps UP to one tile, so the measured area
        // can only ever grow, and the `0×0` sentinel measures 32×32 = 1024 px rather than 0. Drop
        // the `.max(1)` and the doc's argument silently stops being true.
        assert_eq!((align(0), align(1), align(31)), (32, 32, 32));
        assert!(reject_over_area(MODEL_ID, &req(0, 0)).is_ok());
        // …and that raise is not a hole: an edge under one tile still refuses when the OTHER edge
        // carries it past the cap. `1×30000` → `32×29984` = 959 488 px.
        assert_eq!((align(1), align(30000)), (32, 29984));
        assert!(reject_over_area(MODEL_ID, &req(1, 30000)).is_err());
    }

    /// [`reject_over_area`] measures the **requested** dims, which equal the rendered ones only
    /// because [`SizeFloor::RangeChecked`] refuses every edge below `min_size` — so the `0×0`
    /// sentinel can never reach [`Scail2::run`]'s resolve-from-the-driving-clip branch, where the
    /// rendered geometry would be the clip's rather than the request's.
    ///
    /// That is a cross-crate dependency (the floor lives in gen-core's `validate_request`), and
    /// sc-16199 exists to revisit this very declaration. Pinning it here means relaxing the floor
    /// turns this test RED at the site that depends on it, instead of silently opening a gap where an
    /// auto-sized 4K clip measures 1 024 px and renders 8.2 Mpx.
    #[test]
    fn descriptor_declares_the_size_floor_this_gate_depends_on() {
        assert_eq!(
            descriptor().capabilities.size_floor,
            SizeFloor::RangeChecked
        );
        let sentinel = GenerationRequest {
            prompt: "a character".into(),
            width: 0,
            height: 0,
            ..Default::default()
        };
        // The area gate itself passes the sentinel (32×32 once aligned) — the floor is what stops it.
        assert!(reject_over_area(MODEL_ID, &sentinel).is_ok());
        assert!(
            descriptor()
                .capabilities
                .validate_request(MODEL_ID, &sentinel)
                .is_err(),
            "the capability floor must refuse 0×0, or the area gate is measuring the wrong geometry"
        );
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
