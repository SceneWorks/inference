//! Mochi 1 provider — the `Generator` seam: stable identity + advertised [`Capabilities`],
//! request [`validate`](validate_request), model [`load`], and the explicit
//! [`REGISTRATION`]/[`register_providers`] the MLX catalog enumerates.
//!
//! Mochi is **text-to-video only** in the base preview (no audio, no I2V — those are epic 1788
//! follow-ons sc-11998/sc-11997). It is **not** distilled, so it exposes **true CFG**
//! (negative prompt + `guidance` scale). Quant tiers ship as **pre-quantized per-tier checkpoints**
//! (epic 1788 architecture / A6 sc-11990), *not* on-the-fly requant, so [`Capabilities::supported_quants`]
//! is empty and `load` rejects a stray `spec.quantize` rather than silently ignoring it. The
//! denoise/decode itself lives in [`crate::pipeline`].

use mlx_rs::ops::concatenate_axis;
use mlx_rs::{random, Dtype};

use mlx_gen::tokenizer::TextTokenizer;
use mlx_gen::{
    default_seed, Capabilities, Error, GenerationOutput, GenerationRequest, Generator, Image,
    LoadSpec, Modality, ModelDescriptor, Progress, Result, WeightsSource,
};
use mlx_gen_flux::T5TextEncoder;

use crate::config::{MochiConfig, MochiVaeConfig};
use crate::text_encoder::{encode_prompt, load_t5_encoder};
use crate::tokenizer::load_tokenizer;
use crate::transformer::{load_transformer_weights, MochiDitConfig, MochiTransformer3DModel};
use crate::vae::{load_vae_decoder, MochiVaeDecoder};

/// Public provider id: `"mochi_1"`.
pub const MODEL_ID: &str = "mochi_1";

/// AsymmVAE latent channels fed to the DiT / seeded as init noise.
const LATENT_CHANNELS: i32 = 12;
/// Resolution `shift` for the flow-match schedule — Mochi config `shift = 1.0` (no shift).
const MOCHI_SHIFT: f32 = 1.0;

// Production defaults when the request leaves a knob unset (the fixture point overrides these).
/// Diffusers `MochiPipeline` default `num_inference_steps` for the preview.
const DEFAULT_STEPS: u32 = 64;
/// Diffusers `MochiPipeline` default `guidance_scale`.
const DEFAULT_GUIDANCE: f32 = 4.5;
/// A safe default frame count on the `6·k + 1` lattice (`6·3 + 1`).
const DEFAULT_FRAMES: u32 = 19;
/// Mochi renders ~30 fps.
const DEFAULT_FPS: u32 = 30;

/// Stable identity + advertised capabilities for Mochi 1 (text-to-video, true CFG, no audio).
pub fn descriptor() -> ModelDescriptor {
    ModelDescriptor {
        id: MODEL_ID,
        family: "mochi",
        backend: "mlx",
        modality: Modality::Video,
        capabilities: Capabilities {
            // Not distilled → true classifier-free guidance over a [neg, pos] batch.
            supports_negative_prompt: true,
            supports_guidance: true,
            supports_true_cfg: true,
            // Text-to-video only in the base preview (I2V = sc-11998, a follow-on).
            conditioning: Vec::new(),
            supports_lora: false,
            supports_lokr: false,
            // A single fixed flow-match Euler integrator; a selectable sampler/scheduler axis is not
            // wired for Mochi, so advertising one would be a false capability.
            samplers: Vec::new(),
            schedulers: Vec::new(),
            supported_guidance_methods: Vec::new(),
            // Width/height must be divisible by 16 (VAE 8× spatial × DiT patch 2). 480p target = 848×480.
            min_size: 16,
            max_size: 1280,
            max_count: 1,
            mac_only: true,
            // Quant tiers are pre-quantized per-tier checkpoints (epic 1788 / A6 sc-11990), selected by
            // pointing `WeightsSource` at the tier dir — NOT on-the-fly requant. So no on-the-fly levels.
            supported_quants: &[],
            supports_kv_cache: false,
            requires_sigma_shift: false,
            supports_sequential_offload: false,
        },
    }
}

/// The loaded Mochi 1 model: the reused T5-XXL encoder + tokenizer, the AsymmDiT transformer, and the
/// AsymmVAE decoder, all held resident for the whole generation. (Sequential residency is a follow-on;
/// A4 loads everything up front — the milestone is a working end-to-end path, not a memory-minimal one.)
pub struct Mochi {
    descriptor: ModelDescriptor,
    tokenizer: TextTokenizer,
    t5: T5TextEncoder,
    transformer: MochiTransformer3DModel,
    vae: MochiVaeDecoder,
    /// T5 conditioning geometry (documentation / text_len policy).
    #[allow(dead_code)]
    config: MochiConfig,
    vae_config: MochiVaeConfig,
}

/// Assemble the full Mochi 1 model from a snapshot directory (T5-XXL text encoder + AsymmDiT
/// transformer + AsymmVAE decoder). The DiT is loaded at f32 compute precision (its on-disk shards are
/// bf16; the port upcasts them — the parity-verified regime, see `tests/dit_parity.rs`).
pub fn load(spec: &LoadSpec) -> Result<Box<dyn Generator>> {
    let root =
        match &spec.weights {
            WeightsSource::Dir(p) => p.clone(),
            WeightsSource::File(_) => return Err(Error::Msg(
                "mochi_1: expected a model directory (split-weight snapshot), not a single file"
                    .into(),
            )),
        };
    // On-the-fly requant is not the Mochi tier mechanism (epic 1788: self-contained pre-quantized
    // q4/q8/bf16 tier dirs, A6 sc-11990). Reject a stray `spec.quantize` rather than silently ignore
    // it — a mismatch would otherwise run bf16 while the caller believed it asked for Q4.
    if let Some(q) = spec.quantize {
        return Err(Error::Msg(format!(
            "mochi_1: spec.quantize={q:?} unsupported — Mochi ships pre-quantized per-tier \
             checkpoints; point WeightsSource at the q4/q8/bf16 tier dir (no on-the-fly requant)"
        )));
    }

    let config = MochiConfig::from_model_dir(&root)?;
    let vae_config = MochiVaeConfig::from_model_dir(&root)?;
    let dit_cfg = MochiDitConfig::default();

    let tokenizer = load_tokenizer()?;
    let t5 = load_t5_encoder(&root)?;
    let dit_w = load_transformer_weights(&root)?;
    let transformer = MochiTransformer3DModel::from_weights(&dit_w, &dit_cfg, Dtype::Float32)?;
    let vae = load_vae_decoder(&root)?;

    Ok(Box::new(Mochi {
        descriptor: descriptor(),
        tokenizer,
        t5,
        transformer,
        vae,
        config,
        vae_config,
    }))
}

/// Reject a request Mochi cannot serve: the shared capability floor (size range, count,
/// negative/guidance/true_cfg gating, sampler/scheduler membership, conditioning allowlist) plus the
/// model-specific constraints — non-empty prompt, 16-divisible width/height, and `num_frames = 1 + 6·k`.
pub(crate) fn validate_request(caps: &Capabilities, req: &GenerationRequest) -> Result<()> {
    if req.prompt.is_empty() {
        return Err(Error::Msg("mochi_1: prompt must not be empty".into()));
    }
    caps.validate_request(MODEL_ID, req)?;
    if !req.width.is_multiple_of(16) || !req.height.is_multiple_of(16) {
        return Err(Error::Msg(format!(
            "mochi_1: width/height must be divisible by 16 (got {}x{})",
            req.width, req.height
        )));
    }
    if let Some(frames) = req.frames {
        // The AsymmVAE has a 6× temporal ratio, so a valid clip length is `1 + 6·k` latent-aligned.
        if frames % 6 != 1 {
            return Err(Error::Msg(format!(
                "mochi_1: num_frames must be 1 + 6·k (got {frames})"
            )));
        }
    }
    Ok(())
}

impl Mochi {
    /// The rich-`Result` body behind [`Generator::generate`]: T5-XXL masked encode (positive + CFG
    /// unconditional) → seeded latents → the flow-match true-CFG denoise loop → VAE decode →
    /// `Vec<Image>`. Kept on the crate's own [`mlx_gen::Error`] so `?` lifts device exceptions and the
    /// family helpers; the `impl_generator!` wrapper bridges the tail into `gen_core::Error`.
    fn generate_impl(
        &self,
        req: &GenerationRequest,
        on_progress: &mut dyn FnMut(Progress),
    ) -> Result<GenerationOutput> {
        self.validate(req)?;
        // Honor a cancel before the (unbounded) text-encode stage.
        if req.cancel.is_cancelled() {
            return Err(Error::Canceled);
        }

        // T5-XXL masked encode (`_get_t5_prompt_embeds`), positive + the CFG-unconditional branch (the
        // reference encodes the negative prompt, or "" when unset, with `do_classifier_free_guidance`).
        let pos = encode_prompt(&self.tokenizer, &self.t5, &req.prompt)?;
        let neg_prompt = req.negative_prompt.as_deref().unwrap_or("");
        let neg = encode_prompt(&self.tokenizer, &self.t5, neg_prompt)?;
        // CFG batch order [neg, pos] — matches `scheduler::cfg_combine` (uncond = half 0, cond = half 1)
        // and the reference `torch.cat([negative_prompt_embeds, prompt_embeds])`.
        let enc = concatenate_axis(&[&neg.prompt_embeds, &pos.prompt_embeds], 0)?;
        let enc_mask =
            concatenate_axis(&[&neg.prompt_attention_mask, &pos.prompt_attention_mask], 0)?;

        // Geometry: AsymmVAE 6× temporal / 8× spatial; the DiT sees the `[1, 12, F_lat, H/8, W/8]`
        // latent (frames already gated to `1 + 6·k`, size to multiple-of-16 by `validate`).
        let t_ratio = self.vae_config.temporal_compression_ratio() as u32;
        let s_ratio = self.vae_config.spatial_compression_ratio() as u32;
        let frames = req.frames.unwrap_or(DEFAULT_FRAMES);
        let steps = req.steps.unwrap_or(DEFAULT_STEPS) as usize;
        let guidance = req.guidance.unwrap_or(DEFAULT_GUIDANCE);
        let lf = 1 + (frames - 1) / t_ratio;
        let lh = req.height / s_ratio;
        let lw = req.width / s_ratio;

        // Seeded init noise. FlowMatchEuler `init_noise_sigma = 1`, so the seeded standard-normal latent
        // is used unscaled (RNG is not portable to the reference; the e2e parity gate teacher-forces the
        // init latents instead of matching MLX's RNG).
        let seed = req.seed.unwrap_or_else(default_seed);
        let key = random::key(seed)?;
        let init = random::normal::<f32>(
            &[1, LATENT_CHANNELS, lf as i32, lh as i32, lw as i32],
            None,
            None,
            Some(&key),
        )?;

        let latents = crate::pipeline::denoise(
            &self.transformer,
            &init,
            &enc,
            &enc_mask,
            steps,
            guidance,
            MOCHI_SHIFT,
            &req.cancel,
            on_progress,
        )?;

        on_progress(Progress::Decoding);
        let frames_arr = crate::pipeline::decode_to_frames(&self.vae, &latents, &req.cancel)?;
        let images: Vec<Image> = crate::pipeline::frames_to_images(&frames_arr)?;
        Ok(GenerationOutput::Video {
            frames: images,
            fps: req.fps.unwrap_or(DEFAULT_FPS),
            audio: None,
        })
    }
}

mlx_gen::impl_generator!(Mochi {
    validate: |s, req| validate_request(&s.descriptor.capabilities, req),
    generate: generate_impl,
});

// The registration constant bridges the crate's rich `Result` into backend-neutral `gen_core::Result`.
mlx_gen::register_generators! {
    pub(crate) const REGISTRATION = descriptor => load
}

#[cfg(test)]
mod tests {
    use super::*;
    use mlx_gen::CancelFlag;

    fn base_req() -> GenerationRequest {
        GenerationRequest {
            prompt: "a calico kitten".into(),
            width: 64,
            height: 64,
            count: 1,
            frames: Some(7),
            ..Default::default()
        }
    }

    #[test]
    fn descriptor_identity_and_capabilities() {
        let d = descriptor();
        assert_eq!(d.id, "mochi_1");
        assert_eq!(d.family, "mochi");
        assert_eq!(d.backend, "mlx");
        assert_eq!(d.modality, Modality::Video);
        let c = &d.capabilities;
        assert!(c.supports_negative_prompt);
        assert!(c.supports_guidance);
        assert!(c.supports_true_cfg);
        assert!(c.mac_only);
        assert_eq!(c.max_count, 1);
        assert!(c.conditioning.is_empty(), "t2v-only: no conditioning kinds");
        assert!(c.supported_quants.is_empty(), "no on-the-fly requant");
    }

    #[test]
    fn validate_accepts_a_well_formed_request() {
        let caps = descriptor().capabilities;
        validate_request(&caps, &base_req()).expect("well-formed request should validate");
    }

    #[test]
    fn validate_rejects_empty_prompt() {
        let caps = descriptor().capabilities;
        let req = GenerationRequest {
            prompt: String::new(),
            ..base_req()
        };
        assert!(validate_request(&caps, &req).is_err());
    }

    #[test]
    fn validate_rejects_misaligned_size() {
        let caps = descriptor().capabilities;
        let req = GenerationRequest {
            width: 72, // not a multiple of 16
            ..base_req()
        };
        assert!(validate_request(&caps, &req).is_err());
    }

    #[test]
    fn validate_rejects_bad_frame_count() {
        let caps = descriptor().capabilities;
        let req = GenerationRequest {
            frames: Some(8), // not 1 + 6·k
            ..base_req()
        };
        assert!(validate_request(&caps, &req).is_err());
        let ok = GenerationRequest {
            frames: Some(13), // 1 + 6·2
            ..base_req()
        };
        validate_request(&caps, &ok).expect("13 = 1 + 6·2 is valid");
    }

    #[test]
    fn validate_rejects_guidance_only_when_unsupported() {
        // Mochi supports guidance + negative prompt, so a request carrying them must pass the floor.
        let caps = descriptor().capabilities;
        let req = GenerationRequest {
            negative_prompt: Some("blurry".into()),
            guidance: Some(4.5),
            true_cfg: Some(1.0),
            ..base_req()
        };
        validate_request(&caps, &req).expect("guidance/negative are advertised");
    }

    #[test]
    fn load_rejects_single_file_source() {
        let spec = LoadSpec::new(WeightsSource::File(
            "/tmp/does-not-exist.safetensors".into(),
        ));
        assert!(load(&spec).is_err(), "single-file source must be rejected");
    }

    #[test]
    fn load_rejects_on_the_fly_quant() {
        let dir = std::env::temp_dir().join(format!("mochi_load_{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let spec = LoadSpec::new(WeightsSource::Dir(dir.clone())).with_quant(mlx_gen::Quant::Q4);
        assert!(
            load(&spec).is_err(),
            "on-the-fly quant is not the Mochi tier mechanism — must reject"
        );
        std::fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn cancel_flag_field_is_reachable() {
        // Sanity: the cooperative-cancel field used by generate_impl exists and is default-false.
        let req = base_req();
        let _c: &CancelFlag = &req.cancel;
        assert!(!req.cancel.is_cancelled());
    }
}
