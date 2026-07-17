//! Krea 2 Turbo **pose-ControlNet inference** provider (sc-8464, epic 8459) — candle (Windows/CUDA).
//!
//! The deployable sibling of the sc-8460 spike harness (`examples/krea-control-infer.rs`): loads the
//! frozen Krea 2 Turbo base (through the composable [`KreaTrainDit`] — the same forward the branch
//! trains against) plus a trained [`ControlBranch`] overlay, and renders
//! the standard 8-step CFG-free Turbo denoise conditioned on a rendered OpenPose skeleton.
//!
//! **How it conditions:** the pose skeleton is VAE-encoded (Qwen-Image VAE) into a control latent, then
//! [`forward_with_control`] — a drop-in for the base
//! `dit.forward` — adds the branch residual into the frozen main stream after each of the first N
//! single-stream blocks, scaled by `control_scale` and RMS-clamped at τ (the S0 recipe: τ = 0.15,
//! applied identically train/infer). `control_scale = 0` is engine-proven **byte-identical** to the
//! un-branched base generation at the same seed (the spike's identity contract).
//!
//! Bespoke provider (NOT gen-core-registered), worker-invoked by name — the candle pattern for
//! conditioned surfaces (mirrors [`crate::control_train`]'s trainer and the FLUX.2 control provider).
//! Krea 2 Turbo is CFG-free + distilled few-step: a single guidance-inert forward per step, no
//! negative pass. The base DiT keeps a packed q4/q8 tier packed in VRAM (dequant-on-forward, sc-11727)
//! and the control-branch overlay quantizes to a matching q4/q8 packed footprint on request
//! (`branch_quant`, sc-11743) — the small-card path — while the studio-trained overlay is published
//! bf16. `generate` takes `&self` so one load serves many poses; the residual clamp is a fixed recipe
//! constant set at load, not a knob.

use std::path::{Path, PathBuf};

use candle_gen::candle_core::{DType, Device, Tensor};
use candle_gen::gen_core::runtime::CancelFlag;
use candle_gen::gen_core::sampling::TimestepConvention;
use candle_gen::gen_core::{AdapterSpec, Image, OffloadPolicy, Progress, Quant};
use candle_gen::train::flow_match::component_vb;
use candle_gen::{CandleError, Result};
use candle_gen_qwen_image::vae::{QwenVae, QwenVaeEncoder};
use rand::{rngs::StdRng, SeedableRng};

use crate::config::Krea2Config;
use crate::control::{forward_with_control, ControlBranch, DEFAULT_RESIDUAL_CLAMP};
use crate::loader::Weights;
use crate::pipeline::maybe_apply_style_gain;
use crate::pipeline::to_image;
use crate::pipeline::MAX_TEXT_TOKENS;
use crate::text_encoder::{KreaTeConfig, KreaTextEncoder};
use crate::tokenizer::KreaTokenizer;
use crate::train_dit::{KreaTrainDit, KREA_ATTN_CHUNK_BUDGET};
use crate::{load_vae, turbo_sigmas, TURBO_STEPS};

/// Qwen-Image VAE 8× spatial compression (latent side = pixels / 8).
const SPATIAL_SCALE: u32 = 8;
/// Latent channel count (Qwen-Image VAE).
const LATENT_CHANNELS: usize = 16;
/// Width/height must be a multiple of this (VAE 8× × 2×2 patchify), matching the base txt2img guard.
/// Single source of truth = the crate-root [`crate::SIZE_MULTIPLE`] (sc-12612).
use crate::SIZE_MULTIPLE;

/// Default `control_scale` for the distilled CFG-free Turbo base. The S0 spike found the usable band
/// ~0.5–0.75 (widening to ~0.7–0.9 with more data); ship a comfortable mid default and hard-cap the
/// exposed range ≤ 0.85 (over-drive haloes to halftone above that). The worker applies the cap.
pub const DEFAULT_CONTROL_SCALE: f32 = 0.6;

/// Paths to the Krea 2 control checkpoints: the Krea 2 Turbo diffusers snapshot dir (`text_encoder/`,
/// `transformer/`, `vae/`, `tokenizer/`) + the trained control-branch overlay (a single `.safetensors`).
pub struct Krea2ControlPaths {
    /// Krea 2 Turbo diffusers snapshot dir (the deployed base the overlay applies on).
    pub root: PathBuf,
    /// The trained control-branch overlay checkpoint (`.safetensors`, e.g. `control_step5000.safetensors`).
    pub control: PathBuf,
    /// User LoRA/LoKr adapters applied **additively** to the frozen base DiT (sc-11720) — a character /
    /// style adapter reshapes the generated subject while the control branch keeps the pose lock. The
    /// control branch is never adapted. Empty ⇒ the stock control build.
    pub adapters: Vec<AdapterSpec>,
    /// Quantize the control-branch overlay to q4/q8 and keep it packed in VRAM (dequant-on-forward,
    /// sc-11743). **`None` (bf16) is the default and the norm** — branch quant is a *quality cost* (the
    /// residual is precision-sensitive, RMS-clamped at τ), so it is the **last-resort rung** on the Krea
    /// control VRAM fit ladder (sc-11754): the worker's fit-gate engages it only when the predicted peak
    /// still exceeds free VRAM after the cheaper rungs (VAE-decode tiling, activation chunking). It is
    /// **not** auto-mirrored to the base tier — a q4 base on a card with headroom keeps a bf16 branch.
    pub branch_quant: Option<Quant>,
    /// Engage sc-6217-style **query-row attention chunking** on the composable base stack + the control
    /// branch (sc-11745) — the fit-ladder rung **between** VAE-decode tiling (sc-11744) and branch-quant
    /// (sc-11743). `false` (the default and the norm) runs each single-stream block's joint `[ctx; img]`
    /// attention unchunked at the i32-guard budget — full speed, the ~11 GB-of-activations 1024² denoise
    /// peak. The worker's Krea control fit-gate (sc-11754) flips this to `true` **only** when the
    /// predicted *denoise*-phase peak exceeds free VRAM, lowering the scores budget to
    /// [`KREA_ATTN_CHUNK_BUDGET`] so each per-block attention block is bounded (a small speed cost, no
    /// quality cost — the chunked result is numerically identical). A **resolution cap** is a separate,
    /// sharper lever the worker owns by choosing smaller render dims; it needs no knob here. On a card
    /// with headroom this stays `false`.
    pub chunk_attention: bool,
    /// Component residency for this bespoke provider. [`OffloadPolicy::Resident`] keeps Qwen3-VL,
    /// DiT, control branch, and both VAE halves warm. [`OffloadPolicy::Sequential`] loads and encodes
    /// Qwen3-VL first, drops it, then loads the DiT + control branch + VAE bundle for the render. The
    /// worker selects this directly from the control-lane fit gate; this provider is intentionally not
    /// registered, so there is no capability bit to consult.
    pub offload_policy: OffloadPolicy,
}

/// One Krea 2 strict-pose control request. Krea 2 Turbo is CFG-free (no guidance / negative pass) —
/// the conditioning knobs beyond the prompt are `control_scale` and the optional `text_style_gain`.
#[derive(Clone)]
pub struct Krea2ControlRequest {
    pub prompt: String,
    pub width: u32,
    pub height: u32,
    pub steps: usize,
    /// How strongly the control branch locks the base (S0 usable ~0.5–0.85). `0.0` ⇒ base passthrough
    /// (byte-identical to un-branched generation at the same seed).
    pub control_scale: f32,
    /// Optional "text style" tap-reweight gain (sc-12009): reweights the 12 stacked Qwen3-VL taps of the
    /// single CFG-free conditional context before the DiT's TextFusion. `None`/g≈1 is a byte-exact no-op;
    /// the worker clamps to the GPU-validated `[0.25, 1.75]`. Mirrors the txt2img/img2img knob.
    pub text_style_gain: Option<f32>,
    pub seed: u64,
    /// Route the final latent→pixel VAE decode through the seam-free **tiled tail** even below the
    /// im2col-overflow threshold (sc-11744). `false` (the default) is the monolithic decode — full speed,
    /// the ~30 GB end-of-render spike. The worker's Krea control fit-ladder (sc-11754) flips this to
    /// `true` **only** when the predicted decode-phase peak exceeds free VRAM — the cheapest rung (a speed
    /// cost, no quality cost) ahead of branch-quant. On a card with headroom it stays `false`.
    pub tile_vae_decode: bool,
    /// Cooperative cancellation, checked before each denoise step (the engine contract).
    pub cancel: CancelFlag,
}

impl Default for Krea2ControlRequest {
    fn default() -> Self {
        Self {
            prompt: String::new(),
            width: 1024,
            height: 1024,
            steps: TURBO_STEPS,
            control_scale: DEFAULT_CONTROL_SCALE,
            text_style_gain: None,
            seed: 0,
            tile_vae_decode: false,
            cancel: CancelFlag::default(),
        }
    }
}

/// The phase-A Qwen3-VL prompt encoder. Under sequential residency this value drops before any heavy
/// component is loaded.
struct Krea2ControlText {
    tokenizer: KreaTokenizer,
    te: KreaTextEncoder,
}

impl Krea2ControlText {
    fn encode(&self, req: &Krea2ControlRequest) -> Result<Tensor> {
        maybe_apply_style_gain(
            self.te
                .forward(&self.tokenizer.encode_prompt(&req.prompt, MAX_TEXT_TOKENS)?)?,
            req.text_style_gain,
        )
    }
}

/// The heavy render phase: composable Turbo DiT + pose-control branch + both Qwen-Image VAE halves.
/// The control branch deliberately stays beside the DiT rather than spanning the two phases.
struct Krea2ControlHeavy {
    dit: KreaTrainDit,
    branch: ControlBranch,
    vae: QwenVae,
    vae_encoder: QwenVaeEncoder,
}

/// A loaded Krea 2 control model whose residency value exclusively owns either the warm text/heavy pair
/// or the deferred phase loaders. Sequential bounds peak at `max(Qwen3-VL, DiT + branch + VAE)`.
pub struct Krea2Control {
    device: Device,
    residency: candle_gen::Residency<Krea2ControlText, Krea2ControlHeavy>,
}

impl Krea2Control {
    /// Build the selected component residency. Both policies use the same phase loaders; Resident loads
    /// both now, while Sequential defers them until the shared seam runs inside `generate`.
    pub fn load(paths: &Krea2ControlPaths) -> Result<Self> {
        let device = candle_gen::default_device()?;
        let policy = candle_gen::effective_offload_policy(paths.offload_policy);
        let text_root = paths.root.clone();
        let text_device = device.clone();
        let heavy_root = paths.root.clone();
        let heavy_control = paths.control.clone();
        let heavy_adapters = paths.adapters.clone();
        let heavy_branch_quant = paths.branch_quant;
        let heavy_chunk_attention = paths.chunk_attention;
        let heavy_device = device.clone();
        let residency = candle_gen::Residency::from_policy(
            policy,
            move || load_control_text(&text_root, &text_device),
            move |_use_pid| {
                load_control_heavy(
                    &heavy_root,
                    &heavy_control,
                    &heavy_adapters,
                    heavy_branch_quant,
                    heavy_chunk_attention,
                    &heavy_device,
                )
            },
        )?;
        Ok(Self { device, residency })
    }

    /// Generate one strict-pose-conditioned image from a rendered OpenPose skeleton. The control image
    /// must already match the request dimensions; the worker renders it at those exact dimensions.
    pub fn generate(
        &self,
        req: &Krea2ControlRequest,
        control_image: &Image,
        on_progress: &mut dyn FnMut(Progress),
    ) -> Result<Image> {
        validate_request(req)?;
        self.residency.run(
            &req.cancel,
            &self.device,
            false,
            on_progress,
            |text| text.encode(req),
            |heavy, context, on_progress| {
                heavy.render(&self.device, req, control_image, context, on_progress)
            },
        )
    }
}

/// Load the Qwen3-VL text phase exactly once per resident model or once per sequential generation.
fn load_control_text(root: &Path, device: &Device) -> Result<Krea2ControlText> {
    let tokenizer = KreaTokenizer::from_snapshot(root, device)?;
    let te_cfg = KreaTeConfig::from_snapshot(root)?;
    let te_w = Weights::from_dir(&root.join("text_encoder"), device, DType::F32)?;
    let te = KreaTextEncoder::load(&te_w, "language_model", &te_cfg, MAX_TEXT_TOKENS)?;
    drop(te_w);
    Ok(Krea2ControlText { tokenizer, te })
}

/// Load the render phase after the text value has dropped on the sequential path.
fn load_control_heavy(
    root: &Path,
    control: &Path,
    adapters: &[AdapterSpec],
    branch_quant: Option<Quant>,
    chunk_attention: bool,
    device: &Device,
) -> Result<Krea2ControlHeavy> {
    let cfg = Krea2Config::from_snapshot(root)?;
    let dit_w = Weights::from_dir(&root.join("transformer"), device, DType::BF16)?;
    let mut dit = KreaTrainDit::load_inference(&dit_w, &cfg)?;
    drop(dit_w);
    if !adapters.is_empty() {
        crate::adapters::install_additive(&mut dit, adapters)?;
    }

    let mut branch = match branch_quant {
        Some(quant) => ControlBranch::from_checkpoint_quantized(control, &cfg, device, quant)?,
        None => ControlBranch::from_checkpoint(control, &cfg, device)?,
    };
    branch.freeze();
    branch.set_residual_clamp(Some(DEFAULT_RESIDUAL_CLAMP));
    if chunk_attention {
        dit.set_attention_budget(KREA_ATTN_CHUNK_BUDGET);
        branch.set_attention_budget(KREA_ATTN_CHUNK_BUDGET);
    }

    let vae = load_vae(root, device)?;
    let vae_encoder = QwenVaeEncoder::new(component_vb(
        root,
        "vae",
        device,
        DType::F32,
        "krea control infer",
    )?)?;
    Ok(Krea2ControlHeavy {
        dit,
        branch,
        vae,
        vae_encoder,
    })
}

impl Krea2ControlHeavy {
    fn render(
        &self,
        device: &Device,
        req: &Krea2ControlRequest,
        control_image: &Image,
        context: Tensor,
        on_progress: &mut dyn FnMut(Progress),
    ) -> Result<Image> {
        let ctrl_nchw = control_image_to_nchw(control_image, req.width, req.height, device)?;
        let ctrl_latent = self.vae_encoder.encode(&ctrl_nchw)?;
        let scale = req.control_scale as f64;

        let (lat_h, lat_w) = (
            (req.height / SPATIAL_SCALE) as usize,
            (req.width / SPATIAL_SCALE) as usize,
        );
        let mut rng = StdRng::seed_from_u64(req.seed);
        let noise = candle_gen::seeded_normal_vec(&mut rng, LATENT_CHANNELS * lat_h * lat_w);
        let noise = Tensor::from_vec(noise, (1, LATENT_CHANNELS, lat_h, lat_w), &Device::Cpu)?
            .to_device(device)?;

        let sigmas = turbo_sigmas(req.steps);
        let latent = candle_gen::run_flow_sampler(
            None,
            TimestepConvention::Sigma,
            &sigmas,
            noise,
            req.seed,
            &req.cancel,
            on_progress,
            |x, timestep| -> Result<Tensor> {
                let t = Tensor::from_vec(vec![timestep], (1,), device)?;
                let v = forward_with_control(
                    &self.dit,
                    &self.branch,
                    x,
                    &t,
                    &context,
                    &ctrl_latent,
                    scale,
                )?;
                Ok(v.to_dtype(DType::F32)?)
            },
        )?;

        on_progress(Progress::Decoding);
        let decoded = self
            .vae
            .decode_with(&latent, req.tile_vae_decode)?
            .to_dtype(DType::F32)?;
        to_image(&decoded)
    }
}

/// Validate the seed-independent request knobs before any tensor work. The empty-prompt guard mirrors
/// the registered txt2img `validate` (an empty prompt reaches the TE as a zero-length sequence and
/// surfaces as a deep tensor-shape error instead of a clean validation error).
fn validate_request(req: &Krea2ControlRequest) -> Result<()> {
    if req.prompt.trim().is_empty() {
        return Err(CandleError::Msg("krea control: prompt is required".into()));
    }
    if !req.width.is_multiple_of(SIZE_MULTIPLE) || !req.height.is_multiple_of(SIZE_MULTIPLE) {
        return Err(CandleError::Msg(format!(
            "krea control: width/height must be multiples of {SIZE_MULTIPLE} (got {}x{})",
            req.width, req.height
        )));
    }
    if req.steps == 0 {
        return Err(CandleError::Msg("krea control: steps must be >= 1".into()));
    }
    Ok(())
}

/// The rendered OpenPose skeleton (HWC RGB u8, already at `width`×`height`) → `[1, 3, H, W]` f32 in
/// `[-1, 1]`, channel-first — the exact normalization `candle_gen::train::dataset::load_image_tensor`
/// produces at train time, so the VAE-encoded control latent is identical to what the branch was
/// trained on. The worker driver renders the control map at the provider's output dims, so a size
/// mismatch is a wiring bug, not a resize case (the lib carries no image codec) — it errors loudly.
fn control_image_to_nchw(
    image: &Image,
    width: u32,
    height: u32,
    device: &Device,
) -> Result<Tensor> {
    let (iw, ih) = (image.width, image.height);
    if (iw, ih) != (width, height) {
        return Err(CandleError::Msg(format!(
            "krea control: control image {iw}x{ih} must match the render size {width}x{height}"
        )));
    }
    let (rw, rh) = (width as usize, height as usize);
    if image.pixels.len() != rw * rh * 3 {
        return Err(CandleError::Msg(format!(
            "krea control: control pixel buffer {} != {width}x{height}x3",
            image.pixels.len()
        )));
    }
    let mut data = vec![0f32; 3 * rh * rw];
    for y in 0..rh {
        for x in 0..rw {
            let base = (y * rw + x) * 3;
            for c in 0..3 {
                // HWC u8 [0,255] → channel-first [3, H, W]; [-1, 1].
                data[c * rh * rw + y * rw + x] = image.pixels[base + c] as f32 / 127.5 - 1.0;
            }
        }
    }
    Ok(Tensor::from_vec(data, (1, 3, rh, rw), &Device::Cpu)?.to_device(device)?)
}

#[cfg(test)]
mod tests {
    use super::*;

    struct OffloadEnvGuard(Option<String>);

    impl OffloadEnvGuard {
        fn unset() -> Self {
            let prior = std::env::var(candle_gen::OFFLOAD_ENV).ok();
            std::env::remove_var(candle_gen::OFFLOAD_ENV);
            Self(prior)
        }
    }

    impl Drop for OffloadEnvGuard {
        fn drop(&mut self) {
            match self.0.take() {
                Some(value) => std::env::set_var(candle_gen::OFFLOAD_ENV, value),
                None => std::env::remove_var(candle_gen::OFFLOAD_ENV),
            }
        }
    }

    fn missing_paths(offload_policy: OffloadPolicy) -> Krea2ControlPaths {
        Krea2ControlPaths {
            root: PathBuf::from("/nonexistent/krea-control-residency-test-snapshot"),
            control: PathBuf::from("/nonexistent/krea-control-residency-test-overlay.safetensors"),
            adapters: Vec::new(),
            branch_quant: None,
            chunk_attention: false,
            offload_policy,
        }
    }

    /// Weight-free proof that this bespoke lane honors its direct policy even though it has no registry
    /// descriptor/capability bit: Sequential captures loaders and touches no component weights.
    #[test]
    fn sequential_policy_defers_all_component_loads() {
        let _env = OffloadEnvGuard::unset();
        let model = Krea2Control::load(&missing_paths(OffloadPolicy::Sequential))
            .expect("Sequential must not touch the missing snapshot at provider construction");
        assert!(model.residency.is_sequential());
    }

    /// The resident twin uses the same phase loaders eagerly, so the missing snapshot fails at load.
    #[test]
    fn resident_policy_eager_loads_components() {
        let _env = OffloadEnvGuard::unset();
        assert!(Krea2Control::load(&missing_paths(OffloadPolicy::Resident)).is_err());
    }

    /// The request defaults match the Turbo control production knobs (1024², 8 CFG-free steps,
    /// control scale 0.6).
    #[test]
    fn request_defaults() {
        let r = Krea2ControlRequest::default();
        assert_eq!((r.width, r.height), (1024, 1024));
        assert_eq!(r.steps, TURBO_STEPS);
        assert_eq!(r.control_scale, DEFAULT_CONTROL_SCALE);
        // Untiled by default (sc-11744): the monolithic full-speed decode — the fit-ladder flips it on
        // only when the decode-phase peak won't fit.
        assert!(!r.tile_vae_decode);
        assert!(!r.cancel.is_cancelled());
    }

    /// The empty-prompt guard: an empty or whitespace-only prompt is a clean validation error; a real
    /// prompt with valid size passes.
    #[test]
    fn validate_request_rejects_empty_prompt() {
        let empty = Krea2ControlRequest::default();
        assert!(validate_request(&empty)
            .unwrap_err()
            .to_string()
            .contains("prompt is required"));

        let whitespace = Krea2ControlRequest {
            prompt: " \t\n".into(),
            ..Default::default()
        };
        assert!(validate_request(&whitespace)
            .unwrap_err()
            .to_string()
            .contains("prompt is required"));

        let ok = Krea2ControlRequest {
            prompt: "a dancer mid-leap".into(),
            ..Default::default()
        };
        assert!(validate_request(&ok).is_ok());
    }

    /// The size/steps guards fire.
    #[test]
    fn validate_request_keeps_size_and_steps_guards() {
        let odd = Krea2ControlRequest {
            prompt: "a dancer".into(),
            height: 1000,
            ..Default::default()
        };
        assert!(validate_request(&odd)
            .unwrap_err()
            .to_string()
            .contains("multiples"));

        let zero_steps = Krea2ControlRequest {
            prompt: "a dancer".into(),
            steps: 0,
            ..Default::default()
        };
        assert!(validate_request(&zero_steps)
            .unwrap_err()
            .to_string()
            .contains("steps"));
    }

    /// Real-weight two-process resident/sequential parity + peak harness for the bespoke control lane.
    /// Run once per mode in separate processes because candle's CUDA allocator retains its pool:
    ///
    /// ```text
    /// KREA_TURBO_DIR=<tier> KREA_CONTROL_CKPT=<overlay> KREA_CONTROL_POSE=<png> \
    /// KREA_OUT=resident.rgb cargo test -p candle-gen-krea --features cuda \
    ///   control_probed_generate_for_offload_ab -- --ignored --nocapture
    /// CANDLE_GEN_OFFLOAD=sequential KREA_TURBO_DIR=<tier> KREA_CONTROL_CKPT=<overlay> \
    /// KREA_CONTROL_POSE=<png> KREA_OUT=sequential.rgb cargo test -p candle-gen-krea \
    ///   --features cuda control_probed_generate_for_offload_ab -- --ignored --nocapture
    /// ```
    ///
    /// Compare the raw pixel files byte-for-byte and use the printed rendered-device `overall-peak`
    /// deltas as the resident/sequential calibration. `KREA_CONTROL_BRANCH_QUANT=q8|q4` selects the
    /// branch tier; omitted means bf16. `KREA_AB_RES` defaults to 768 and `KREA_AB_STEPS` defaults to
    /// eight. One step is sufficient for packed-tier peak calibration because the same denoise working
    /// set is reused at every step, but both processes in a parity pair must use the same value.
    #[cfg(feature = "cuda")]
    #[test]
    #[ignore]
    fn control_probed_generate_for_offload_ab() {
        let root = PathBuf::from(std::env::var("KREA_TURBO_DIR").expect("set KREA_TURBO_DIR"));
        let control =
            PathBuf::from(std::env::var("KREA_CONTROL_CKPT").expect("set KREA_CONTROL_CKPT"));
        let pose_path =
            std::env::var("KREA_CONTROL_POSE").expect("set KREA_CONTROL_POSE to a pose PNG");
        let out = std::env::var("KREA_OUT").expect("set KREA_OUT to the raw pixel-dump path");
        let res = std::env::var("KREA_AB_RES")
            .ok()
            .and_then(|value| value.trim().parse().ok())
            .unwrap_or(768u32);
        let steps = std::env::var("KREA_AB_STEPS")
            .ok()
            .and_then(|value| value.trim().parse().ok())
            .unwrap_or(TURBO_STEPS);
        let branch_quant = match std::env::var("KREA_CONTROL_BRANCH_QUANT")
            .unwrap_or_default()
            .trim()
            .to_ascii_lowercase()
            .as_str()
        {
            "" | "bf16" | "none" => None,
            "q8" => Some(Quant::Q8),
            "q4" => Some(Quant::Q4),
            other => panic!("KREA_CONTROL_BRANCH_QUANT must be bf16|q8|q4, got {other}"),
        };
        let spec_mode = std::env::var("KREA_OFFLOAD_MODE").unwrap_or_default();
        let offload_policy = if spec_mode == "spec-sequential" {
            OffloadPolicy::Sequential
        } else {
            OffloadPolicy::Resident
        };

        let pose = image::open(pose_path).expect("decode pose PNG").to_rgb8();
        let pose = image::imageops::resize(&pose, res, res, image::imageops::FilterType::Lanczos3);
        let pose = Image {
            width: res,
            height: res,
            pixels: pose.into_raw(),
        };
        let paths = Krea2ControlPaths {
            root,
            control,
            adapters: Vec::new(),
            branch_quant,
            chunk_attention: false,
            offload_policy,
        };
        let request = Krea2ControlRequest {
            prompt: "a dancer in a colorful studio, cinematic lighting".into(),
            width: res,
            height: res,
            steps,
            control_scale: DEFAULT_CONTROL_SCALE,
            seed: 42,
            ..Default::default()
        };

        let mut probe = candle_gen::testkit::VramProbe::start_rendered();
        let load_phase = probe.phase();
        let model = Krea2Control::load(&paths).expect("load Krea control provider");
        probe.end_load(load_phase);
        let gen_phase = probe.phase();
        let image = model
            .generate(&request, &pose, &mut |_| {})
            .expect("generate Krea control image");
        probe.end_gen(gen_phase);
        let report = probe.report();
        std::fs::write(&out, &image.pixels).expect("write raw pixels");

        let env_mode = std::env::var(candle_gen::OFFLOAD_ENV).unwrap_or_default();
        let mode = if spec_mode == "spec-sequential" {
            "spec-sequential"
        } else if env_mode.eq_ignore_ascii_case("sequential") {
            "env-sequential"
        } else {
            "resident"
        };
        eprintln!(
            "SEQ_AB id=krea_2_turbo_control mode={mode} gpu={} {}x{} steps={} branch_quant={branch_quant:?} | {report} | bytes={} out={out}",
            candle_gen::testkit::probe_gpu(),
            image.width,
            image.height,
            request.steps,
            image.pixels.len(),
        );
    }
}
