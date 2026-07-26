//! Krea Realtime 14B provider: capability surface, registration, snapshot resolution, and the
//! [`Generator`] entrypoint (sc-8439, S6).
//!
//! [`Generator::generate`] maps a text-to-video [`GenerationRequest`] onto a [`KreaRealtimeJob`] and
//! runs the live [`crate::t2v::generate_t2v`] pipeline (prompt → UMT5 → AR few-step latents → z16 Wan
//! VAE decode → clip). Krea Realtime is an **autoregressive, self-forcing, CFG-off** text-to-video
//! model: no negative prompt, no guidance scale, a fixed Self-Forcing few-step schedule. Image/video
//! conditioning (i2v / v2v) is **S7**; the streaming realtime decode is the streaming epic.
//!
//! Registration mirrors the sibling Wan-2.1-14B video provider `mlx-gen-scail2`: an explicit
//! `ModelRegistration` composed by the platform catalog (never linker-discovered), so
//! `provider_registry().load("krea_realtime_14b", spec)` yields this generator.

use std::path::PathBuf;

use mlx_gen::{
    default_seed, Capabilities, Error, GenerationOutput, GenerationRequest, Generator, Modality,
    ModelDescriptor, Progress, Result, WeightsSource,
};

use crate::config::{KreaRealtimeConfig, MODEL_ID};
use crate::t2v::KreaRealtimeJob;

/// The Self-Forcing few-step sampler name Krea Realtime advertises (a fixed short per-block flow-match
/// renoise schedule, not a selectable classic solver). Distinct from Wan's UniPC/DPM++ so a consumer
/// knows this model runs the AR few-step regime.
pub const SELF_FORCING_SAMPLER: &str = "self_forcing";

/// Default output cadence when a request omits `fps` (the reference realtime-video default).
const DEFAULT_FPS: u32 = 16;
/// Default output frame count when a request omits `frames`/`duration` — the reference canonical clip
/// is 21 latent frames = 81 output frames (`(81 − 1)/4 + 1 = 21`).
const DEFAULT_FRAMES: u32 = 81;

/// Stable identity + advertised capabilities for Krea Realtime 14B (Wan-2.1-T2V-14B backbone,
/// autoregressive self-forcing **text-to-video**; **CFG off** → no negative prompt / no guidance; a
/// fixed Self-Forcing few-step sampler; a rolling causal KV cache).
pub fn descriptor() -> ModelDescriptor {
    ModelDescriptor {
        required_components: &[],
        id: MODEL_ID,
        family: "krea_realtime",
        backend: "mlx",
        modality: Modality::Video,
        capabilities: Capabilities {
            // CFG-off model: the AR few-step denoise runs a single batch-1 forward per step with no
            // unconditional branch, so there is no negative-prompt / guidance axis (sc-8437 S4).
            supports_negative_prompt: false,
            supports_guidance: false,
            supports_true_cfg: false,
            // Text-to-video only in S6 (no image/video conditioning); i2v/v2v is S7.
            conditioning: Vec::new(),
            // No inference LoRA/quant wired yet (the base DiT loads dense bf16). Advertising a knob the
            // load path does not honor would violate capability honesty — kept off until wired.
            supports_lora: false,
            supports_lokr: false,
            samplers: vec![SELF_FORCING_SAMPLER],
            schedulers: Vec::new(),
            supported_guidance_methods: vec![],
            // H/W align to patch×vae_stride = 16 (z16 VAE spatial stride 8, patch 2); mirror Wan's cap.
            min_size: 16,
            max_size: 1280,
            max_count: 1,
            mac_only: true,
            // Q4/Q8 load-time quant is not wired in S6 (dense bf16 only) — advertise none until it is.
            supported_quants: &[],
            // The AR regime is built on a rolling causal KV cache (sc-8436 S3 / sc-8438 S5).
            supports_kv_cache: true,
            requires_sigma_shift: false,
            supports_sequential_offload: false,
            // Batch whole-clip form in S6; the realtime streaming decode is the streaming epic.
            supports_streaming: false,
            supports_multi_speaker: false,
            supports_conversation_history: false,
            supports_conversation_session: false,
            max_speakers: None,
            // No audio surface: pure video model.
            audio_sample_rates: vec![],
            max_audio_duration_secs: None,
            audio_voices: vec![],
            audio_languages: vec![],
            audio_edit_modes: vec![],
        },
    }
}

/// The loaded Krea Realtime 14B model: the resolved config + the snapshot dir. The heavy components
/// (DiT / UMT5 / z16 VAE) are staged inside [`crate::t2v::generate_t2v`].
pub struct KreaRealtime {
    descriptor: ModelDescriptor,
    config: KreaRealtimeConfig,
    root: PathBuf,
}

/// Load Krea Realtime from a converted MLX snapshot directory (`dit.safetensors` + the stock Wan
/// `t5_encoder.safetensors` / `vae.safetensors` / `tokenizer.json` — Krea Realtime ships
/// transformer-only, reusing stock Wan for the TE / VAE / tokenizer).
pub fn load(spec: &mlx_gen::LoadSpec) -> Result<Box<dyn Generator>> {
    let root = match &spec.weights {
        WeightsSource::Dir(p) => p.clone(),
        WeightsSource::File(_) => return Err(Error::Msg(
            "krea_realtime: expected a model directory (converted MLX snapshot), not a single file"
                .into(),
        )),
    };
    if !root.exists() {
        return Err(Error::Msg(format!(
            "krea_realtime: snapshot dir does not exist: {}",
            root.display()
        )));
    }
    let config = KreaRealtimeConfig::from_model_dir(&root)?;
    Ok(Box::new(KreaRealtime {
        descriptor: descriptor(),
        config,
        root,
    }))
}

// The registration constant the platform catalog composes (explicit, not linker-discovered); bridges
// the crate `Result` into backend-neutral `gen_core::Result`.
mlx_gen::register_generators! { pub(crate) const REGISTRATION = descriptor => load }

mlx_gen::impl_generator!(KreaRealtime {
    validate: |s, req| s
        .descriptor
        .capabilities
        .validate_request(s.descriptor.id, req),
    generate: run,
});

impl KreaRealtime {
    /// Map the text-to-video request onto a [`KreaRealtimeJob`] and run the AR pipeline. Self-validates
    /// the shared capability floor first (`impl_generator!`'s `generate` does not call `validate`), so a
    /// direct `Generator::generate` still rejects an out-of-surface request (count / size / sampler /
    /// finiteness) before any weight load.
    fn run(
        &self,
        req: &GenerationRequest,
        on_progress: &mut dyn FnMut(Progress),
    ) -> Result<GenerationOutput> {
        self.descriptor
            .capabilities
            .validate_request(self.descriptor.id, req)?;

        // Output frame count: explicit `frames`, else derived from `duration × fps`, else the default.
        let fps = req.fps.unwrap_or(DEFAULT_FPS);
        let num_frames = req.frames.unwrap_or_else(|| {
            req.duration
                .map(|d| ((d * fps as f32).round() as u32).max(1))
                .unwrap_or(DEFAULT_FRAMES)
        });

        let job = KreaRealtimeJob {
            prompt: &req.prompt,
            width: req.width,
            height: req.height,
            num_frames,
            fps,
            seed: req.seed.unwrap_or_else(default_seed),
            steps: req.steps.map(|s| s as usize),
        };
        crate::t2v::generate_t2v(&self.root, &self.config, &job, &req.cancel, on_progress)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A weights-free [`KreaRealtime`] whose [`KreaRealtime::run`] hits the shared floor before any
    /// load. Sound only because validation runs first: a *passing* request would then try to load
    /// `root` and fail.
    fn unloaded() -> KreaRealtime {
        KreaRealtime {
            descriptor: descriptor(),
            config: KreaRealtimeConfig::default(),
            root: PathBuf::from("/nonexistent-krea-realtime-snapshot"),
        }
    }

    /// The advertised surface is text-to-video, CFG-off, with the Self-Forcing few-step sampler.
    #[test]
    fn descriptor_is_cfg_off_text_to_video() {
        let d = descriptor();
        assert_eq!(d.id, "krea_realtime_14b");
        assert_eq!(d.family, "krea_realtime");
        assert_eq!(d.backend, "mlx");
        assert_eq!(d.modality, Modality::Video);
        let c = &d.capabilities;
        assert!(!c.supports_guidance, "Krea Realtime is CFG-off");
        assert!(!c.supports_negative_prompt, "CFG-off ⇒ no negative prompt");
        assert!(!c.supports_true_cfg);
        assert!(c.conditioning.is_empty(), "t2v only in S6 (i2v/v2v is S7)");
        assert_eq!(c.samplers, vec!["self_forcing"]);
        assert!(c.supports_kv_cache, "the AR regime runs a rolling KV cache");
        assert!(c.mac_only);
        assert!(!c.supports_streaming, "batch form in S6");
        // The descriptor passes the weights-free conformance sweep.
        assert!(
            mlx_gen::gen_core::registry::model_descriptor_errors(&d).is_empty(),
            "descriptor must be conformant: {:?}",
            mlx_gen::gen_core::registry::model_descriptor_errors(&d)
        );
    }

    /// The shared floor rejects an unadvertised sampler before any weight load.
    #[test]
    fn floor_rejects_unadvertised_sampler() {
        let m = unloaded();
        let mut noop = |_: Progress| {};
        let bad = GenerationRequest {
            width: 512,
            height: 512,
            count: 1,
            sampler: Some("unipc".into()),
            ..Default::default()
        };
        let err = m
            .run(&bad, &mut noop)
            .expect_err("unadvertised sampler must be rejected");
        assert!(err.to_string().contains("sampler"), "got: {err}");
    }

    /// A guidance scale is rejected (CFG-off): the floor gates `supports_guidance`.
    #[test]
    fn floor_rejects_guidance_on_cfg_off_model() {
        let m = unloaded();
        let mut noop = |_: Progress| {};
        let bad = GenerationRequest {
            width: 512,
            height: 512,
            count: 1,
            guidance: Some(5.0),
            ..Default::default()
        };
        let err = m
            .run(&bad, &mut noop)
            .expect_err("guidance must be rejected on a CFG-off model");
        assert!(err.to_string().contains("guidance"), "got: {err}");
    }
}
