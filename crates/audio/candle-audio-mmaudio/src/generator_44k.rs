//! The **shipping MMAudio 44.1 kHz quality-ceiling video→audio (Foley) generator** (sc-13441, epic
//! sc-12833) — the [`gen_core::Generator`] that assembles the large_44k_v2 MM-DiT + the 44k mel-VAE +
//! the NVIDIA BigVGAN v2 vocoder (alongside the shared CLIP + Synchformer conditioners) into one
//! synchronized-soundtrack pipeline and registers into `candle-audio-catalog` under the sibling id
//! **`mmaudio_large_44k`** (a distinct id beside `mmaudio_small_16k`, not a runtime quality selector —
//! cleaner for the ordered-surface ship-gate, and it lets each quality tier carry its own descriptor
//! sample-rate / composite license).
//!
//! ## The pipeline (reference `MMAudio/demo.py` with `--variant large_44k_v2`)
//!
//! Identical conditioning to the 16k provider — the CLIP visual/text encoder and the Synchformer sync
//! encoder are the **same weights** — feeding the larger 1.03B `large_44k_v2` MM-DiT (14 heads, hidden
//! 896, depth 21, `v2=True`; see [`mmdit::Config::large_44k_v2`]) which denoises `(1, latent_seq_len,
//! 40)` audio latents at the 44.1 kHz latent frame rate, then the 44k [`crate::output::AudioDecoder44k`]
//! decodes latent → 128-band mel (44k VAE) → 44.1 kHz waveform (NVIDIA BigVGAN v2, 512× upsample).
//!
//! ## Weights + license
//!
//! Five checkpoints across **three** HF repos: the CLIP DFN5B encoder (`apple/…`, Apple ML Research —
//! research-only), the Synchformer + large_44k_v2 MM-DiT + 44k mel-VAE (`hkchengrex/MMAudio` — MIT /
//! CC-BY-NC-4.0), and the NVIDIA BigVGAN v2 vocoder (`nvidia/bigvgan_v2_44khz_128band_512x` — MIT). The
//! composite the provider ships under is the **intersection** (strictest): research / non-commercial,
//! set by the Apple DFN5B conditioner exactly as the 16k provider (the NVIDIA BigVGAN v2 MIT term is
//! more permissive, so it neither blocks nor relaxes the composite). See [`WEIGHT_LICENSE`].

use std::sync::{Arc, Mutex};

use candle_audio::candle_core::{Device, Tensor};
use candle_audio::gen_core::{
    self, reject_unknown_components, require_component, AudioTrack, Capabilities, ConditioningKind,
    GenerationOutput, GenerationRequest, Generator, Image, LoadSpec, Modality, ModelDescriptor,
    Progress, WeightsSource,
};
use candle_audio::{AudioError, Result as AudioResult};

use crate::clip::{self, DfnClipEncoder};
use crate::generator::{
    cfg_flow, check_seq, effective_duration, lock_recover, resample_frames, seeded_prior,
    validate_request, video_sync_frames, PipelineProgress, CLIP_FPS, REQUIRED_COMPONENTS, SYNC_FPS,
};
use crate::output::{AudioDecoder44k, SAMPLE_RATE_44K};
use crate::sync::SynchformerVisualEncoder;
use crate::{mmdit, model, preprocess};

/// Registry id (the SceneWorks worker routes `payload.model` to this exact id) — the 44.1 kHz sibling
/// of `mmaudio_small_16k`.
pub const MODEL_ID: &str = "mmaudio_large_44k";

/// Provider family (shared with the 16k provider).
pub const FAMILY: &str = "mmaudio";

/// Native output sample rate (Hz) — the 44.1 kHz output path.
pub const SAMPLE_RATE: u32 = SAMPLE_RATE_44K as u32;

/// The trained latent window (`CONFIG_44K.duration`), and the longest clip this model synthesizes.
pub const MAX_DURATION_SECS: f32 = 8.0;

/// The default duration cap when a request supplies none — the reference `demo.py --duration` default.
pub const DEFAULT_DURATION_SECS: f32 = 8.0;

/// Shortest renderable duration (one Synchformer segment = `16 / 25 fps = 0.64 s`) — same conditioning
/// floor as the 16k provider.
pub const MIN_DURATION_SECS: f32 = 0.68;

/// Default Euler flow-matching steps and CFG strength (reference defaults, shared with 16k).
pub const DEFAULT_STEPS: u32 = mmdit::NUM_STEPS as u32;
pub const DEFAULT_CFG: f32 = mmdit::CFG_STRENGTH as f32;

/// Largest solver step count accepted.
pub const MAX_STEPS: u32 = 500;

/// Prompt language the CLIP text tower was trained on.
pub const LANGUAGES: &[&str] = &["en"];

/// The **composite** model-weight license for the shipping `mmaudio_large_44k` provider (sc-13441).
///
/// The 44k pipeline assembles **five** checkpoints across **three** repos under three license
/// families — recorded per-component in [`crate::WEIGHT_LICENSES_44K`]. As with the 16k provider, the
/// catalog keys one composite row per registered id: the **intersection** (strictest) of the five,
/// which is **research / non-commercial** — the Apple ML Research Model License on the DFN5B-CLIP
/// conditioner (research-only) remains the strictest, the MMAudio large_44k_v2 MM-DiT + 44k mel-VAE
/// add CC-BY-NC-4.0 (non-commercial), and the Synchformer encoder + NVIDIA BigVGAN v2 vocoder are MIT
/// (permissive — they neither block nor relax the composite). SceneWorks is non-commercial, so the
/// weights are usable, but the composite restriction MUST be surfaced.
pub const WEIGHT_LICENSE: gen_core::WeightLicense = gen_core::WeightLicense {
    spdx_id: "LicenseRef-MMAudio-large-44k-composite",
    name: "MMAudio large_44k_v2 composite (Apple ML Research + CC-BY-NC-4.0 + MIT)",
    source_url: "https://huggingface.co/hkchengrex/MMAudio",
    attribution: Some(
        "MMAudio video→audio 44.1 kHz (mmaudio_large_44k) assembles five checkpoints: the \
         large_44k_v2 MM-DiT network + 44k mel-VAE (© Sony Research Inc. / MMAudio — CC-BY-NC-4.0), \
         the DFN5B-CLIP ViT-H/14-384 conditioner (© Apple Inc. — Apple ML Research Model License, \
         research-only), the Synchformer visual encoder (© 2024 Vladimir Iashin — MIT), and the \
         NVIDIA BigVGAN v2 44 kHz vocoder (© 2024 NVIDIA CORPORATION — MIT).",
    ),
    commercial_use: false,
    restriction: Some(
        "Research / non-commercial only — the intersection of five component licenses. The strictest, \
         the Apple ML Research Model License on the DFN5B-CLIP conditioner, limits use to scientific \
         research and academic development and excludes any commercial product or service; the MMAudio \
         large_44k_v2 MM-DiT / 44k mel-VAE add CC-BY-NC-4.0 (non-commercial); the Synchformer encoder \
         and NVIDIA BigVGAN v2 vocoder are MIT (permissive). See candle-audio-mmaudio::WEIGHT_LICENSES_44K \
         for each checkpoint's full terms. A legal read is warranted before any commercial use.",
    ),
};

/// This provider's single composite weight-license entry (keyed by [`MODEL_ID`]) — what
/// `candle-audio-catalog` aggregates into the model-licenses manifest.
pub const WEIGHT_LICENSE_ENTRY: gen_core::WeightLicenseEntry = gen_core::WeightLicenseEntry {
    provider_id: MODEL_ID,
    // The composite / effective-restriction row (component == None); the per-checkpoint
    // attribution rows live in `crate::SHIPPED_WEIGHT_LICENSES` beside it (sc-13493).
    component: None,
    license: WEIGHT_LICENSE,
};

/// The 44k provider's identity + capabilities — constructible without weights.
pub fn descriptor() -> ModelDescriptor {
    ModelDescriptor {
        // Same five named component ids as the 16k provider (epic 13657, sc-13666) — `clip` /
        // `synchformer` / `dit` / `vae` / `vocoder`. The ids match; the underlying checkpoints differ
        // (the `large_44k_v2` MM-DiT + 44k mel-VAE from `hkchengrex/MMAudio`, and — unlike the 16k
        // path's in-repo BigVGAN — the `vocoder` from `nvidia/bigvgan_v2_44khz_128band_512x`).
        required_components: REQUIRED_COMPONENTS,
        id: MODEL_ID,
        family: FAMILY,
        backend: "candle",
        modality: Modality::Audio,
        capabilities: Capabilities {
            supports_negative_prompt: true,
            supports_guidance: true,
            supports_true_cfg: false,
            conditioning: vec![ConditioningKind::VideoSync],
            supports_lora: false,
            supports_lokr: false,
            samplers: vec![],
            schedulers: vec![],
            supported_guidance_methods: vec![],
            min_size: 0,
            max_size: 0,
            max_count: 1,
            mac_only: false,
            audio_sample_rates: vec![SAMPLE_RATE],
            max_audio_duration_secs: Some(MAX_DURATION_SECS),
            audio_voices: vec![],
            audio_languages: LANGUAGES.to_vec(),
            audio_edit_modes: vec![],
            supported_quants: &[],
            supports_kv_cache: false,
            requires_sigma_shift: false,
            supports_sequential_offload: false,
            supports_streaming: false,
            supports_multi_speaker: false,
            supports_conversation_history: false,
            supports_conversation_session: false,
            max_speakers: None,
        },
    }
}

/// The `(latent_seq_len, clip_seq_len, sync_seq_len)` for a duration — the port of `SequenceConfig`
/// for `CONFIG_44K` (`sampling_rate=44100`, `spectrogram_frame_rate=512`, `latent_downsample_rate=2`,
/// `clip_frame_rate=8`, `sync_frame_rate=25`, 16-frame / step-8 segments, `sync_downsample_rate=2`).
/// Only the latent length differs from the 16k config (44.1 kHz / 512-hop mel → ~43.07 latents/s).
pub(crate) fn seq_lengths(duration: f32) -> (usize, usize, usize) {
    let duration = duration as f64;
    // ceil(duration * 44100 / 512 / 2).
    let latent = (duration * 44100.0 / 512.0 / 2.0).ceil() as usize;
    let clip = (duration * CLIP_FPS as f64) as usize; // int(duration * 8)
    let sync_frames = (duration * SYNC_FPS as f64) as usize; // int(duration * 25)
    let num_segments = if sync_frames >= 16 {
        (sync_frames - 16) / 8 + 1
    } else {
        0
    };
    let sync = num_segments * 8; // num_segments * 16 / 2
    (latent, clip, sync)
}

/// The assembled MMAudio 44.1 kHz synthesis pipeline: the two shared conditioners + the large_44k_v2
/// MM-DiT + the 44k decoder, all resident on one device. The DiT is behind a `Mutex` because each
/// request reconfigures its sequence lengths for the clip's duration.
pub struct MmAudio44kPipeline {
    clip: DfnClipEncoder,
    sync: SynchformerVisualEncoder,
    dit: Mutex<mmdit::MmAudioDit>,
    decoder: AudioDecoder44k,
    device: Device,
}

impl MmAudio44kPipeline {
    /// Load all five components from their individually-provisioned [`WeightsSource`]s (epic 13657) —
    /// `clip` / `synchformer` / `dit` / `vae` / `vocoder`, staged by the caller in
    /// [`LoadSpec::components`] and validated at [`load`]. The 44k twin of
    /// [`MmAudioPipeline::from_components`](crate::generator::MmAudioPipeline::from_components): the
    /// `dit` is the `large_44k_v2` preset and the decoder pairs the 44k mel-VAE with the external
    /// NVIDIA BigVGAN v2 vocoder. No assembled snapshot directory (sc-13666).
    pub fn from_components(
        clip_src: &WeightsSource,
        sync_src: &WeightsSource,
        dit_src: &WeightsSource,
        vae_src: &WeightsSource,
        vocoder_src: &WeightsSource,
        device: &Device,
    ) -> AudioResult<Self> {
        let clip = clip::load(clip_src, device)?;
        let sync = model::load(sync_src, device)?;
        let dit = mmdit::load_large_44k_v2(dit_src, device)?;
        let decoder = AudioDecoder44k::load(vae_src, vocoder_src, device)?;
        Ok(Self {
            clip,
            sync,
            dit: Mutex::new(dit),
            decoder,
            device: device.clone(),
        })
    }

    /// The compute device the weights live on.
    pub fn device(&self) -> &Device {
        &self.device
    }

    /// Encode the frames + prompt, reconfigure the DiT for the clip's duration, run the Euler / CFG
    /// sampler from a `seed`-seeded Gaussian prior, and decode to a 44.1 kHz waveform.
    #[allow(clippy::too_many_arguments)]
    pub fn synthesize(
        &self,
        frames: &[Image],
        src_fps: f32,
        duration: f32,
        prompt: &str,
        negative_prompt: &str,
        steps: usize,
        cfg: f64,
        seed: u64,
        progress: &mut dyn FnMut(PipelineProgress),
        probe: &dyn Fn() -> bool,
    ) -> AudioResult<Vec<f32>> {
        let (latent_seq_len, clip_seq_len, sync_seq_len) = seq_lengths(duration);
        let dev = &self.device;

        let clip_rgb = resample_frames(frames, src_fps, CLIP_FPS, clip_seq_len);
        let sync_rgb_count = (duration * SYNC_FPS).floor() as usize;
        let sync_rgb = resample_frames(frames, src_fps, SYNC_FPS, sync_rgb_count);

        let clip_feat = self.encode_clip_visual(&clip_rgb)?; // (1, clip_seq_len, 1024)
        let text_feat = self.encode_text(prompt)?; // (1, 77, 1024)
        let neg_text_feat = self.encode_text(negative_prompt)?; // (1, 77, 1024)
        if probe() {
            return Err(AudioError::Canceled);
        }
        let sync_feat = self.encode_sync(&sync_rgb)?; // (1, sync_seq_len, 768)
        check_seq(&clip_feat, 1, clip_seq_len, "clip")?;
        check_seq(&sync_feat, 1, sync_seq_len, "sync")?;

        let x0 = seeded_prior(
            seed,
            latent_seq_len,
            mmdit::Config::large_44k_v2().latent_dim,
            dev,
        )?;

        self.synthesize_from_features(
            &clip_feat,
            &sync_feat,
            &text_feat,
            &neg_text_feat,
            &x0,
            cfg,
            steps,
            progress,
            probe,
        )
    }

    /// The **injectable assembly core** — from already-encoded conditioning features + a prior `x0`,
    /// reconfigure the DiT, run the Euler / CFG flow-matching sampler, and decode latent → mel →
    /// 44.1 kHz waveform. Split out so the end-to-end reference-parity harness can inject the
    /// reference's own dumped features + prior noise (the 44k twin of the 16k parity core).
    #[allow(clippy::too_many_arguments)]
    pub fn synthesize_from_features(
        &self,
        clip_feat: &Tensor,     // (1, clip_seq_len, 1024)
        sync_feat: &Tensor,     // (1, sync_seq_len, 768)
        text_feat: &Tensor,     // (1, 77, 1024)
        neg_text_feat: &Tensor, // (1, 77, 1024)
        x0: &Tensor,            // (1, latent_seq_len, 40)
        cfg: f64,
        steps: usize,
        progress: &mut dyn FnMut(PipelineProgress),
        probe: &dyn Fn() -> bool,
    ) -> AudioResult<Vec<f32>> {
        let latent_seq_len = x0.dim(1)?;
        let clip_seq_len = clip_feat.dim(1)?;
        let sync_seq_len = sync_feat.dim(1)?;

        let mut dit = lock_recover(&self.dit);
        dit.update_seq_lengths(latent_seq_len, clip_seq_len, sync_seq_len)
            .map_err(AudioError::from)?;

        let cond = dit
            .preprocess_conditions(clip_feat, sync_feat, text_feat)
            .map_err(AudioError::from)?;
        let empty = dit
            .empty_conditions_with_text(1, neg_text_feat)
            .map_err(AudioError::from)?;

        let mut x = x0.clone();
        for i in 0..steps {
            if probe() {
                return Err(AudioError::Canceled);
            }
            let t = i as f64 / steps as f64;
            let dt = (i + 1) as f64 / steps as f64 - t;
            let flow = cfg_flow(&dit, &x, t, &cond, &empty, cfg).map_err(AudioError::from)?;
            x = (x + (flow * dt)?).map_err(AudioError::from)?;
            progress(PipelineProgress::Step(i + 1));
        }
        let latent = dit.unnormalize(&x).map_err(AudioError::from)?; // (1, N, 40)
        drop(dit);

        if probe() {
            return Err(AudioError::Canceled);
        }
        progress(PipelineProgress::Decoding);
        // The VAE consumes (B, latent_dim, N); the DiT emits (B, N, latent_dim).
        let latent = latent.transpose(1, 2)?.contiguous()?;
        let wav = self
            .decoder
            .latent_to_waveform(&latent)
            .map_err(AudioError::from)?; // (1, 1, S)
        let samples: Vec<f32> = wav.flatten_all()?.to_vec1()?;
        Ok(samples)
    }

    fn encode_clip_visual(&self, frames: &[image::RgbImage]) -> AudioResult<Tensor> {
        let input = clip::frames_to_clip_input(frames, &self.device)?; // (M, 3, 384, 384)
        let feat = self.clip.encode_image(&input)?; // (M, 1024)
        Ok(feat.unsqueeze(0)?) // (1, M, 1024)
    }

    fn encode_text(&self, text: &str) -> AudioResult<Tensor> {
        let row = clip::tokenize_str(text).to_vec();
        let tokens = clip::tokenize(&[row], &self.device)?; // (1, 77)
        Ok(self.clip.encode_text(&tokens)?) // (1, 77, 1024)
    }

    fn encode_sync(&self, frames: &[image::RgbImage]) -> AudioResult<Tensor> {
        let segments = preprocess::frames_to_segments(frames, &self.device)?; // (S, 3, 16, 224, 224)
        let feat = self.sync.encode(&segments)?; // (S, 8, 768)
        let (s, per_seg, d) = feat.dims3()?;
        Ok(feat.reshape((1, s * per_seg, d))?) // (1, sync_seq_len, 768)
    }
}

/// The five caller-provisioned component sources resolved at [`load`] (epic 13657), held so the heavy
/// 44k pipeline can be built lazily on first `generate`.
struct ComponentSources {
    clip: WeightsSource,
    sync: WeightsSource,
    dit: WeightsSource,
    vae: WeightsSource,
    vocoder: WeightsSource,
}

/// A loaded (lazy) MMAudio 44k generator. The heavy pipeline (CLIP ViT-H + Synchformer +
/// large_44k_v2 MM-DiT + 44k VAE + NVIDIA BigVGAN v2, several GB resident in f32) is built on first
/// use and cached; `load` does no file I/O beyond argument checks.
pub struct MmAudioLarge44kGenerator {
    descriptor: ModelDescriptor,
    components: ComponentSources,
    pipeline: Mutex<Option<Arc<MmAudio44kPipeline>>>,
}

impl MmAudioLarge44kGenerator {
    fn pipeline(&self) -> gen_core::Result<Arc<MmAudio44kPipeline>> {
        let mut guard = lock_recover(&self.pipeline);
        if let Some(p) = guard.as_ref() {
            return Ok(p.clone());
        }
        let device = candle_audio::default_device()?;
        let c = &self.components;
        let built = Arc::new(MmAudio44kPipeline::from_components(
            &c.clip, &c.sync, &c.dit, &c.vae, &c.vocoder, &device,
        )?);
        *guard = Some(built.clone());
        Ok(built)
    }
}

/// Construct the (lazy) 44k generator from a [`LoadSpec`]. The five checkpoints are provisioned as
/// named [`LoadSpec::components`] — `clip` / `synchformer` / `dit` / `vae` / `vocoder`
/// ([`REQUIRED_COMPONENTS`]) — each required at load via [`require_component`], so a missing component
/// is a **load-time** contract error (epic 13657, sc-13666). `spec.weights` is unused (mmaudio is a
/// pure assembly of the five named components). Unknown component keys, quantization, adapters, and
/// control / IP-adapter overlays are rejected.
pub fn load(spec: &LoadSpec) -> gen_core::Result<Box<dyn Generator>> {
    reject_unknown_components(spec, REQUIRED_COMPONENTS, MODEL_ID)?;
    let clip =
        require_component(spec, "clip", MODEL_ID, "DFN5B-CLIP ViT-H/14 conditioner")?.clone();
    let sync =
        require_component(spec, "synchformer", MODEL_ID, "Synchformer visual encoder")?.clone();
    let dit = require_component(spec, "dit", MODEL_ID, "large_44k_v2 MM-DiT network")?.clone();
    let vae = require_component(spec, "vae", MODEL_ID, "44k mel-VAE decoder")?.clone();
    let vocoder =
        require_component(spec, "vocoder", MODEL_ID, "NVIDIA BigVGAN v2 vocoder")?.clone();
    if spec.quantize.is_some() {
        return Err(gen_core::Error::Unsupported(format!(
            "{MODEL_ID} does not support on-the-fly quantization"
        )));
    }
    if !spec.adapters.is_empty() {
        return Err(gen_core::Error::Unsupported(format!(
            "{MODEL_ID} does not support LoRA/LoKr adapters"
        )));
    }
    if spec.control.is_some() || !spec.extra_controls.is_empty() || spec.ip_adapter.is_some() {
        return Err(gen_core::Error::Unsupported(format!(
            "{MODEL_ID} does not support control/IP-adapter overlays"
        )));
    }
    Ok(Box::new(MmAudioLarge44kGenerator {
        descriptor: descriptor(),
        components: ComponentSources {
            clip,
            sync,
            dit,
            vae,
            vocoder,
        },
        pipeline: Mutex::new(None),
    }))
}

impl Generator for MmAudioLarge44kGenerator {
    fn descriptor(&self) -> &ModelDescriptor {
        &self.descriptor
    }

    fn validate(&self, req: &GenerationRequest) -> gen_core::Result<()> {
        // Same conditioning/sampling floor as the 16k provider (identical 8 s window + sync segment
        // floor); the 44.1 kHz sample rate is enforced via this descriptor's audio_sample_rates.
        validate_request(&self.descriptor, req)
    }

    fn generate(
        &self,
        req: &GenerationRequest,
        on_progress: &mut dyn FnMut(Progress),
    ) -> gen_core::Result<GenerationOutput> {
        self.validate(req)?;
        if req.cancel.is_cancelled() {
            return Err(gen_core::Error::Canceled);
        }
        let frames = video_sync_frames(req)?.ok_or_else(|| {
            gen_core::Error::Msg(format!(
                "{MODEL_ID}: a VideoSync clip is required (video→audio Foley conditions on frames)"
            ))
        })?;
        let fps = req.fps.expect("validate ensured req.fps is present") as f32;
        let clip_secs = frames.len() as f32 / fps;
        let duration = effective_duration(req, clip_secs);

        let steps = req.steps.unwrap_or(DEFAULT_STEPS) as usize;
        let cfg = req.guidance.unwrap_or(DEFAULT_CFG) as f64;
        let seed = req.seed.unwrap_or_else(gen_core::default_seed);
        let negative = req.negative_prompt.clone().unwrap_or_default();

        let pipeline = self.pipeline()?;
        let total = steps as u32;
        let cancel = req.cancel.clone();
        let probe = move || cancel.is_cancelled();
        let mut progress = |p: PipelineProgress| match p {
            PipelineProgress::Step(k) => on_progress(Progress::Step {
                current: k as u32,
                total,
            }),
            PipelineProgress::Decoding => on_progress(Progress::Decoding),
        };
        let samples = pipeline
            .synthesize(
                frames,
                fps,
                duration,
                &req.prompt,
                &negative,
                steps,
                cfg,
                seed,
                &mut progress,
                &probe,
            )
            .map_err(gen_core::Error::from)?;

        Ok(GenerationOutput::Audio(AudioTrack {
            samples,
            sample_rate: SAMPLE_RATE,
            channels: 1,
            ..Default::default()
        }))
    }
}

// Explicit catalog registration for `mmaudio_large_44k` (composed by `candle-audio-catalog`).
candle_audio::register_generators! {
    pub const REGISTRATION = descriptor => load
}

#[cfg(test)]
mod tests {
    use super::*;
    use candle_audio::gen_core::{CancelFlag, Conditioning};

    fn foley_frames(n: usize, w: u32, h: u32, seed: u8) -> Vec<Image> {
        (0..n)
            .map(|f| {
                let mut pixels = vec![0u8; (w * h * 3) as usize];
                for (i, p) in pixels.iter_mut().enumerate() {
                    *p = ((i as u32 + f as u32 * 37 + seed as u32 * 101) % 251) as u8;
                }
                Image {
                    width: w,
                    height: h,
                    pixels,
                }
            })
            .collect()
    }

    fn foley_req(frames: Vec<Image>, fps: u32) -> GenerationRequest {
        GenerationRequest {
            prompt: "footsteps on gravel".into(),
            fps: Some(fps),
            seed: Some(7),
            conditioning: vec![Conditioning::VideoSync { frames }],
            ..Default::default()
        }
    }

    #[test]
    fn descriptor_advertises_the_44k_surface() {
        let d = descriptor();
        assert_eq!(d.id, "mmaudio_large_44k");
        assert_eq!(d.family, "mmaudio");
        assert_eq!(d.backend, "candle");
        assert!(matches!(d.modality, Modality::Audio));
        assert_eq!(d.capabilities.audio_sample_rates, [44_100]);
        assert_eq!(d.capabilities.max_audio_duration_secs, Some(8.0));
        assert_eq!(
            d.capabilities.conditioning,
            vec![ConditioningKind::VideoSync]
        );
        assert!(d.capabilities.supports_guidance);
        assert!(d.capabilities.supports_negative_prompt);
        assert_eq!(d.capabilities.max_count, 1);
        assert!(d.capabilities.audio_voices.is_empty());
        // The five named components (epic 13657, sc-13666) — same ids as the 16k provider; pins the
        // input contract so a change is deliberate (CONTRIBUTING.md).
        assert_eq!(
            d.required_components,
            ["clip", "synchformer", "dit", "vae", "vocoder"]
        );
    }

    #[test]
    fn seq_lengths_match_the_44k_reference_config() {
        // 8 s → CONFIG_44K (latent 345, clip 64, sync 192).
        assert_eq!(seq_lengths(8.0), (345, 64, 192));
        // 1 s → latent ceil(43.066) = 44, clip 8, sync 16 (2 segments × 8).
        assert_eq!(seq_lengths(1.0), (44, 8, 16));
        // A short-but-valid clip just above the 0.64 s floor → exactly one sync segment.
        let (_l, _c, sync) = seq_lengths(0.72);
        assert_eq!(sync, 8, "~0.7s is exactly one 16-frame segment");
    }

    #[test]
    fn weight_license_is_the_research_noncommercial_composite() {
        assert!(WEIGHT_LICENSE.is_well_formed());
        assert!(!WEIGHT_LICENSE.is_permissive());
        let commercial_use = WEIGHT_LICENSE.commercial_use;
        assert!(!commercial_use, "44k composite is research/non-commercial");
        assert!(WEIGHT_LICENSE.restriction.is_some());
        assert_eq!(WEIGHT_LICENSE_ENTRY.provider_id, "mmaudio_large_44k");
    }

    #[test]
    fn validate_gates_the_conditioning_surface() {
        let d = descriptor();
        let ok = foley_req(foley_frames(8, 16, 16, 0), 8);
        assert!(validate_request(&d, &ok).is_ok());
        let short = foley_req(foley_frames(8, 16, 16, 0), 25);
        assert!(validate_request(&d, &short).is_err());
    }

    /// A weights-free [`LoadSpec`] that stages every required component (placeholder paths — `load`
    /// is lazy and reads no file). `weights` is an ignored placeholder.
    fn staged_spec() -> LoadSpec {
        let dir = std::env::temp_dir().join("mmaudio-44k-staged");
        LoadSpec::new(WeightsSource::Dir(dir))
            .with_component("clip", WeightsSource::File("/nonexistent/clip.bin".into()))
            .with_component(
                "synchformer",
                WeightsSource::File("/nonexistent/sync.pth".into()),
            )
            .with_component("dit", WeightsSource::File("/nonexistent/dit.pth".into()))
            .with_component("vae", WeightsSource::File("/nonexistent/vae.pth".into()))
            .with_component(
                "vocoder",
                WeightsSource::File("/nonexistent/vocoder.pt".into()),
            )
    }

    #[test]
    fn load_requires_every_component_and_rejects_unsupported_spec_shapes() {
        // Bare spec (no components) → load fails at the first missing component gate.
        let bare = LoadSpec::new(WeightsSource::Dir(std::env::temp_dir()));
        let err = match load(&bare) {
            Err(e) => e,
            Ok(_) => panic!("bare spec (no components) must fail at load"),
        };
        assert!(err.to_string().contains("clip"), "got: {err}");

        // Every required component staged → load succeeds (lazy; no weight read).
        assert!(load(&staged_spec()).is_ok());

        // Quantization is still rejected as Unsupported even with components staged.
        let mut spec = staged_spec();
        spec.quantize = Some(gen_core::Quant::Q4);
        assert!(matches!(load(&spec), Err(gen_core::Error::Unsupported(_))));
    }

    #[test]
    fn pre_tripped_cancel_returns_typed_canceled_before_any_heavy_work() {
        let g = load(&staged_spec()).unwrap();
        let flag = CancelFlag::new();
        flag.cancel();
        let mut req = foley_req(foley_frames(8, 16, 16, 0), 8);
        req.cancel = flag;
        let err = g.generate(&req, &mut |_| {}).unwrap_err();
        assert!(matches!(err, gen_core::Error::Canceled));
    }
}
