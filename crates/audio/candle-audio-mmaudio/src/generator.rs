//! The **shipping MMAudio video→audio (Foley) generator** (sc-12843, epic sc-12833) — the
//! [`gen_core::Generator`] that assembles this crate's four parity-verified components into one
//! synchronized-soundtrack pipeline and registers into `candle-audio-catalog` under the id
//! **`mmaudio_small_16k`**.
//!
//! ## The pipeline (reference `MMAudio/demo.py` + `eval_utils.generate`)
//!
//! A [`Conditioning::VideoSync`] clip's RGB frames (rate on [`GenerationRequest::fps`]) plus an
//! **optional** text prompt drive:
//!
//! 1. **CLIP visual** ([`crate::clip`]) — frames resampled to 8 fps, encoded per frame → `(1, clip_seq_len, 1024)`.
//! 2. **CLIP text** — `prompt` (may be empty; video-only Foley is first-class) → `(1, 77, 1024)`.
//! 3. **Synchformer** ([`crate::sync`]) — frames resampled to 25 fps, windowed into 16-frame /
//!    step-8 segments → `(1, sync_seq_len, 768)` frame-aligned sync features.
//! 4. **MM-DiT** ([`crate::mmdit`]) — the Euler-25 / CFG-4.5 flow-matching sampler seeded by
//!    `req.seed`, from a Gaussian prior → `(1, latent_seq_len, 20)` audio latents. The negative/CFG
//!    branch's text is `encode_text(negative_prompt)` (default `""`), faithful to the reference.
//! 5. **16k decoder** ([`crate::output`]) — latent → mel (VAE) → 16 kHz waveform (BigVGAN) → one mono
//!    [`AudioTrack`].
//!
//! ## Duration (variable, ≤ 8 s — the trained window)
//!
//! MMAudio's `SequenceConfig` derives every sequence length from the clip duration, and `demo.py`
//! sets `duration = min(--duration, video_length)` before `net.update_seq_lengths(...)`. This provider
//! mirrors that: the effective duration is `min(req.audio.target_duration ?? 8 s, clip_length, 8 s)`,
//! then [`mmdit::MmAudioDit::update_seq_lengths`] rebuilds the length-dependent RoPE / upsample tensors so a
//! 1 s clip renders ~1 s of audio (not a fixed 8 s block). A `target_duration` above the trained 8 s
//! window is rejected by the shared audio floor; a clip too short for one Synchformer segment
//! (< 0.64 s) is a typed [`gen_core::Error::Msg`].

use std::sync::{Arc, Mutex};

use candle_audio::candle_core::{Device, Result as CResult, Tensor};
use candle_audio::gen_core::{
    self, reject_unknown_components, require_component, AudioTrack, Capabilities, Conditioning,
    ConditioningKind, GenerationOutput, GenerationRequest, Generator, Image, LoadSpec, Modality,
    ModelDescriptor, Progress, WeightsSource,
};
use candle_audio::{AudioError, Result as AudioResult};
use rand::rngs::StdRng;
use rand::SeedableRng;
use rand_distr::{Distribution, StandardNormal};

use crate::clip::{self, DfnClipEncoder};
use crate::output::{AudioDecoder16k, SAMPLE_RATE as OUT_SAMPLE_RATE};
use crate::sync::SynchformerVisualEncoder;
use crate::{mmdit, model, preprocess};

/// Registry id (the SceneWorks worker routes `payload.model` to this exact id). The 44.1 kHz
/// quality-ceiling sibling is [`crate::generator_44k`]'s `mmaudio_large_44k` (sc-13441).
pub const MODEL_ID: &str = "mmaudio_small_16k";

/// Provider family.
pub const FAMILY: &str = "mmaudio";

/// Native output sample rate (Hz) — the 16k output path.
pub const SAMPLE_RATE: u32 = OUT_SAMPLE_RATE as u32;

/// The trained latent window (`CONFIG_16K.duration`), and the longest clip this model synthesizes.
pub const MAX_DURATION_SECS: f32 = 8.0;

/// The default duration cap when a request supplies none — the reference `demo.py --duration` default.
pub const DEFAULT_DURATION_SECS: f32 = 8.0;

/// Shortest renderable duration: one Synchformer segment is `16 / 25 fps = 0.64 s`, below which the
/// sync stream has no full 16-frame window. Set just above the algebraic 0.64 s so the reference's
/// own `int(duration * 25)` truncation (0.64 × 25 = 15.999… → 15 frames in both Python and Rust) can
/// never collapse the window to zero segments at the boundary.
pub const MIN_DURATION_SECS: f32 = 0.68;

/// CLIP / Synchformer sampling rates (frames per second) — the reference `_CLIP_FPS` / `_SYNC_FPS`.
pub(crate) const CLIP_FPS: f32 = 8.0;
pub(crate) const SYNC_FPS: f32 = 25.0;

/// Default Euler flow-matching steps and CFG strength (reference defaults).
pub const DEFAULT_STEPS: u32 = mmdit::NUM_STEPS as u32;
pub const DEFAULT_CFG: f32 = mmdit::CFG_STRENGTH as f32;

/// Largest solver step count accepted — a finer ladder than a few hundred Euler steps only adds cost.
pub const MAX_STEPS: u32 = 500;

/// CFG guidance bounds: 1.0 turns guidance off (single forward per step); the reference default is 4.5.
pub const GUIDANCE_RANGE: (f32, f32) = (1.0, 20.0);

/// Prompt language the CLIP text tower was trained on (English; the prompt is advisory and optional).
pub const LANGUAGES: &[&str] = &["en"];

/// The **composite** model-weight license for the shipping `mmaudio_small_16k` provider (sc-13332).
///
/// MMAudio's assembled pipeline pulls **five** checkpoints across two repos, under three different
/// licenses — the crate's per-component [`crate::WEIGHT_LICENSES`] records each in full. The
/// `candle-audio-catalog` ship-gate keys exactly one license row per *registered* provider id, so the
/// governing license the provider ships under is surfaced here as one entry keyed by [`MODEL_ID`]: the
/// **intersection** of all five, i.e. the strictest terms. That is **research / non-commercial only** —
/// the DFN5B-CLIP conditioner's Apple ML Research Model License limits use to scientific research and
/// academic development (excluding commercial products), and the MM-DiT / mel-VAE / BigVGAN checkpoints
/// add CC-BY-NC-4.0 (non-commercial); the Synchformer visual encoder is MIT. SceneWorks is
/// non-commercial, so the weights are usable, but the composite restriction MUST be surfaced.
pub const WEIGHT_LICENSE: gen_core::WeightLicense = gen_core::WeightLicense {
    spdx_id: "LicenseRef-MMAudio-small-16k-composite",
    name: "MMAudio small_16k composite (Apple ML Research + CC-BY-NC-4.0 + MIT)",
    source_url: "https://huggingface.co/hkchengrex/MMAudio",
    attribution: Some(
        "MMAudio video→audio (mmaudio_small_16k) assembles five checkpoints: the MM-DiT network + 16k \
         mel-VAE + 16k BigVGAN (© Sony Research Inc. / MMAudio — CC-BY-NC-4.0), the DFN5B-CLIP \
         ViT-H/14-384 conditioner (© Apple Inc. — Apple ML Research Model License, research-only), and \
         the Synchformer visual encoder (© 2024 Vladimir Iashin — MIT).",
    ),
    commercial_use: false,
    restriction: Some(
        "Research / non-commercial only — the intersection of five component licenses. The strictest, \
         the Apple ML Research Model License on the DFN5B-CLIP conditioner, limits use to scientific \
         research and academic development and excludes any commercial product or service; the MMAudio \
         MM-DiT / mel-VAE / BigVGAN checkpoints add CC-BY-NC-4.0 (non-commercial); the Synchformer \
         encoder is MIT. See candle-audio-mmaudio::WEIGHT_LICENSES for each checkpoint's full terms. A \
         legal read is warranted before any commercial use.",
    ),
};

/// This provider's single composite weight-license entry (keyed by [`MODEL_ID`]) — what
/// `candle-audio-catalog` aggregates into the model-licenses manifest (one row per registered
/// provider). The five per-component entries live in [`crate::WEIGHT_LICENSES`].
pub const WEIGHT_LICENSE_ENTRY: gen_core::WeightLicenseEntry = gen_core::WeightLicenseEntry {
    provider_id: MODEL_ID,
    // The composite / effective-restriction row (component == None) — the at-a-glance
    // "can we use this provider" signal. The per-checkpoint attribution rows live in
    // `crate::SHIPPED_WEIGHT_LICENSES` beside it (sc-13493).
    component: None,
    license: WEIGHT_LICENSE,
};

/// The five named model components (epic 13657) `mmaudio_small_16k` requires, provisioned by the
/// caller in [`LoadSpec::components`] and read at [`load`] via [`require_component`]. The composite
/// assembles five checkpoints across two repos: `clip` (the DFN5B-CLIP ViT-H/14 conditioner, from
/// `apple/DFN5B-CLIP-ViT-H-14-384`) and `synchformer` / `dit` / `vae` / `vocoder` (the Synchformer
/// visual encoder, the small_16k MM-DiT, the 16k mel-VAE, and the 16k BigVGAN vocoder, all from
/// `hkchengrex/MMAudio`). Advertised weights-free on the descriptor so a consumer knows what to stage.
pub const REQUIRED_COMPONENTS: &[&str] = &["clip", "synchformer", "dit", "vae", "vocoder"];

/// MMAudio's identity + capabilities — constructible without weights.
pub fn descriptor() -> ModelDescriptor {
    ModelDescriptor {
        required_components: REQUIRED_COMPONENTS,
        id: MODEL_ID,
        family: FAMILY,
        backend: "candle",
        modality: Modality::Audio,
        capabilities: Capabilities {
            supports_negative_prompt: true,
            supports_guidance: true,
            supports_true_cfg: false,
            // The one video→audio conditioning: a silent clip's RGB frames (the Foley condition).
            conditioning: vec![ConditioningKind::VideoSync],
            supports_lora: false,
            supports_lokr: false,
            samplers: vec![],
            schedulers: vec![],
            supported_guidance_methods: vec![],
            // Pure audio: no visual size floor (the audio descriptor sweep exempts Audio, sc-13314).
            min_size: 0,
            max_size: 0,
            // One clip per request (GenerationOutput::Audio carries a single track).
            max_count: 1,
            mac_only: false,
            audio_sample_rates: vec![SAMPLE_RATE],
            max_audio_duration_secs: Some(MAX_DURATION_SECS),
            // No voice / edit-mode / speaker surface — this is video-conditioned Foley.
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

/// The single `VideoSync` clip carried by a request, or `None` — factored out so `validate`/`generate`
/// agree on the extraction. More than one `VideoSync` is rejected (the model conditions on one clip).
pub(crate) fn video_sync_frames(req: &GenerationRequest) -> gen_core::Result<Option<&[Image]>> {
    let mut found: Option<&[Image]> = None;
    for c in &req.conditioning {
        if let Conditioning::VideoSync { frames } = c {
            if found.is_some() {
                return Err(gen_core::Error::Msg(format!(
                    "{MODEL_ID}: more than one VideoSync clip supplied — condition on a single clip"
                )));
            }
            found = Some(frames);
        }
    }
    Ok(found)
}

/// Capability-driven request validation, factored out for weightless unit tests. Shared audio floor
/// (which gates un-advertised/empty `VideoSync`, the duration range, sample rate, language, …) plus
/// this model's own sampling-knob and clip-length bounds.
pub(crate) fn validate_request(
    desc: &ModelDescriptor,
    req: &GenerationRequest,
) -> gen_core::Result<()> {
    let id = desc.id;
    let caps = &desc.capabilities;
    // The shared floor first: un-advertised or empty VideoSync, out-of-range duration/rate/language,
    // and every audio-shaped cross-check. Prompt is OPTIONAL for Foley, so no empty-prompt gate.
    caps.validate_request_audio(id, req)?;

    if let Some(steps) = req.steps {
        if steps == 0 || steps > MAX_STEPS {
            return Err(gen_core::Error::Msg(format!(
                "{id}: steps {steps} outside 1..={MAX_STEPS} (the Euler flow-matching ladder; \
                 default {DEFAULT_STEPS})"
            )));
        }
    }
    if let Some(g) = req.guidance {
        if !(GUIDANCE_RANGE.0..=GUIDANCE_RANGE.1).contains(&g) {
            return Err(gen_core::Error::Msg(format!(
                "{id}: guidance (CFG scale) {g} outside {GUIDANCE_RANGE:?} (1.0 disables CFG; the \
                 reference default is {DEFAULT_CFG})"
            )));
        }
    }

    // The clip must carry a frame rate (the reference derives every length from fps; the variant
    // deliberately does not duplicate it — it rides req.fps) and enough frames for one sync segment.
    let frames = video_sync_frames(req)?;
    if let Some(frames) = frames {
        let fps = req.fps.ok_or_else(|| {
            gen_core::Error::Msg(format!(
                "{id}: a VideoSync clip requires a frame rate on req.fps (the clip's fps)"
            ))
        })?;
        if fps == 0 {
            return Err(gen_core::Error::Msg(format!("{id}: req.fps must be > 0")));
        }
        let clip_secs = frames.len() as f32 / fps as f32;
        let duration = effective_duration(req, clip_secs);
        if duration < MIN_DURATION_SECS {
            return Err(gen_core::Error::Msg(format!(
                "{id}: the clip is {clip_secs:.3}s (effective {duration:.3}s) but MMAudio needs at \
                 least {MIN_DURATION_SECS:.2}s of video for one Synchformer segment (16 frames @ 25 fps)"
            )));
        }
        // Uniform frame geometry (the encoders resize per frame, but a zero-sized frame is malformed).
        if let Some(bad) = frames.iter().find(|f| {
            f.width == 0 || f.height == 0 || f.pixels.len() != (f.width * f.height * 3) as usize
        }) {
            return Err(gen_core::Error::Msg(format!(
                "{id}: a VideoSync frame is malformed ({}x{}, {} bytes; expected w*h*3)",
                bad.width,
                bad.height,
                bad.pixels.len()
            )));
        }
    }
    Ok(())
}

/// The effective render duration: `min(target ?? default, clip_length, trained 8 s window)` — the
/// port of `demo.py`'s `duration = min(--duration, video_length)` capped by the trained latent window.
pub(crate) fn effective_duration(req: &GenerationRequest, clip_secs: f32) -> f32 {
    let cap = req
        .audio
        .as_ref()
        .and_then(|a| a.target_duration)
        .unwrap_or(DEFAULT_DURATION_SECS);
    cap.min(clip_secs).min(MAX_DURATION_SECS)
}

/// The `(latent_seq_len, clip_seq_len, sync_seq_len)` for a duration — the port of `SequenceConfig`
/// (`sequence_config.py`) for the 16k config (`sampling_rate=16000`, `spectrogram_frame_rate=256`,
/// `latent_downsample_rate=2`, `clip_frame_rate=8`, `sync_frame_rate=25`, 16-frame / step-8 segments,
/// `sync_downsample_rate=2`).
pub(crate) fn seq_lengths(duration: f32) -> (usize, usize, usize) {
    let duration = duration as f64;
    let latent = (duration * 16000.0 / 256.0 / 2.0).ceil() as usize; // ceil(duration * 31.25)
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

/// Nearest-frame temporal resample of `frames` (captured at `src_fps`) to `count` frames at `dst_fps`,
/// sampling the first `count / dst_fps` seconds — the analogue of the reference's per-fps video decode.
pub(crate) fn resample_frames(
    frames: &[Image],
    src_fps: f32,
    dst_fps: f32,
    count: usize,
) -> Vec<image::RgbImage> {
    let n = frames.len();
    (0..count)
        .map(|i| {
            let t = i as f32 / dst_fps;
            let src = (t * src_fps).round() as usize;
            let src = src.min(n - 1);
            let f = &frames[src];
            image::RgbImage::from_raw(f.width, f.height, f.pixels.clone())
                .expect("validated frame pixel buffer is w*h*3")
        })
        .collect()
}

/// The assembled MMAudio synthesis pipeline: the two conditioners + the MM-DiT + the 16k decoder,
/// all resident on one device. The DiT is behind a `Mutex` because each request reconfigures its
/// sequence lengths ([`mmdit::MmAudioDit::update_seq_lengths`]) for the clip's duration.
pub struct MmAudioPipeline {
    clip: DfnClipEncoder,
    sync: SynchformerVisualEncoder,
    dit: Mutex<mmdit::MmAudioDit>,
    decoder: AudioDecoder16k,
    device: Device,
}

/// One synthesis progress event (mirrors the sibling providers' `PipelineProgress`).
pub enum PipelineProgress {
    /// Euler step `k` of `total` completed.
    Step(usize),
    /// The DiT finished; the 16k decoder (VAE + BigVGAN) is running.
    Decoding,
}

impl MmAudioPipeline {
    /// Load all five components from their individually-provisioned [`WeightsSource`]s (epic 13657) —
    /// `clip` / `synchformer` / `dit` / `vae` / `vocoder`, staged by the caller in
    /// [`LoadSpec::components`] and validated at [`load`]. Each component struct already loads from a
    /// single checkpoint file (or a per-repo snapshot dir it probes); there is no assembled snapshot
    /// directory anymore (sc-13666).
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
        let dit = mmdit::load(dit_src, device)?;
        let decoder = AudioDecoder16k::load(vae_src, vocoder_src, device)?;
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

    /// Encode the frames + prompt, reconfigure the DiT for the clip's duration, run the Euler /
    /// CFG sampler from a `seed`-seeded Gaussian prior, and decode to a 16 kHz waveform. `probe`
    /// returns `true` to cancel (checked before/each step/before decode); `progress` reports steps.
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

        // --- conditioners -------------------------------------------------------------------
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

        // --- prior noise, seeded deterministically ------------------------------------------
        let x0 = seeded_prior(
            seed,
            latent_seq_len,
            mmdit::Config::small_16k().latent_dim,
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

    /// The **injectable assembly core** (the sc-12843 wiring): from already-encoded conditioning
    /// features + a prior `x0`, reconfigure the DiT to the features' sequence lengths, run the
    /// Euler / CFG flow-matching sampler (reference `flow_matching.py` + `ode_wrapper`), and decode
    /// latent → mel → 16 kHz waveform. Split out so the end-to-end **reference-parity** harness can
    /// inject the reference's own dumped features + prior noise, isolating this assembly from the
    /// (separately parity-verified) encoders and from torch-vs-Rust RNG.
    #[allow(clippy::too_many_arguments)]
    pub fn synthesize_from_features(
        &self,
        clip_feat: &Tensor,     // (1, clip_seq_len, 1024)
        sync_feat: &Tensor,     // (1, sync_seq_len, 768)
        text_feat: &Tensor,     // (1, 77, 1024)
        neg_text_feat: &Tensor, // (1, 77, 1024)
        x0: &Tensor,            // (1, latent_seq_len, 20)
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

        // --- Euler / CFG flow-matching sampler (reference flow_matching.py, ode_wrapper) -----
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
        let latent = dit.unnormalize(&x).map_err(AudioError::from)?; // (1, N, 20)
        drop(dit);

        // --- decode: latent -> mel -> 16 kHz waveform ---------------------------------------
        if probe() {
            return Err(AudioError::Canceled);
        }
        progress(PipelineProgress::Decoding);
        // The VAE consumes (B, latent_dim, N); the DiT emits (B, N, latent_dim) (reference decode()
        // transposes before the VAE).
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

/// One CFG-combined flow at scalar timestep `t` (reference `ode_wrapper`): `cfg·v(cond) + (1−cfg)·v(empty)`.
pub(crate) fn cfg_flow(
    dit: &mmdit::MmAudioDit,
    latent: &Tensor,
    t: f64,
    cond: &mmdit::Conditions,
    empty: &mmdit::Conditions,
    cfg: f64,
) -> CResult<Tensor> {
    let bs = latent.dim(0)?;
    let tvec = Tensor::full(t as f32, (bs,), dit.device())?;
    if cfg < 1.0 {
        return dit.predict_flow(latent, &tvec, cond);
    }
    let vc = dit.predict_flow(latent, &tvec, cond)?;
    let ve = dit.predict_flow(latent, &tvec, empty)?;
    (vc * cfg)? + (ve * (1.0 - cfg))?
}

/// A `seed`-seeded standard-Gaussian prior `(1, latent_seq_len, latent_dim)` — deterministic
/// run-to-run (the reproducibility law). Not byte-identical to torch's RNG; the parity harness injects
/// torch's dumped prior for the tight comparison.
pub(crate) fn seeded_prior(
    seed: u64,
    latent_seq_len: usize,
    latent_dim: usize,
    dev: &Device,
) -> CResult<Tensor> {
    let mut rng = StdRng::seed_from_u64(seed);
    let n = latent_seq_len * latent_dim;
    let data: Vec<f32> = (0..n).map(|_| StandardNormal.sample(&mut rng)).collect();
    Tensor::from_vec(data, (1, latent_seq_len, latent_dim), dev)
}

pub(crate) fn check_seq(t: &Tensor, batch: usize, seq: usize, name: &str) -> AudioResult<()> {
    let d = t.dims();
    if d.first() != Some(&batch) || d.get(1) != Some(&seq) {
        return Err(AudioError::Msg(format!(
            "{MODEL_ID}: {name} feature shape {d:?} does not match derived ({batch}, {seq}, …)"
        )));
    }
    Ok(())
}

/// Recover a poisoned mutex (a prior panic mid-synthesis) — the audio twin of `candle_gen::lock_recover`.
pub(crate) fn lock_recover<T>(m: &Mutex<T>) -> std::sync::MutexGuard<'_, T> {
    match m.lock() {
        Ok(g) => g,
        Err(poisoned) => poisoned.into_inner(),
    }
}

/// The five caller-provisioned component sources resolved at [`load`] (epic 13657), held so the heavy
/// pipeline can be built lazily on first `generate`. Each is a [`LoadSpec::components`] entry validated
/// by [`require_component`] before this struct is constructed.
struct ComponentSources {
    clip: WeightsSource,
    sync: WeightsSource,
    dit: WeightsSource,
    vae: WeightsSource,
    vocoder: WeightsSource,
}

/// A loaded (lazy) MMAudio generator. The heavy pipeline (CLIP ViT-H + Synchformer + MM-DiT + VAE +
/// BigVGAN, several GB resident in f32) is built on first use and cached; `load` does no file I/O
/// beyond argument checks (the sibling providers' lazy-load discipline).
pub struct MmAudioGenerator {
    descriptor: ModelDescriptor,
    components: ComponentSources,
    pipeline: Mutex<Option<Arc<MmAudioPipeline>>>,
}

impl MmAudioGenerator {
    fn pipeline(&self) -> gen_core::Result<Arc<MmAudioPipeline>> {
        let mut guard = lock_recover(&self.pipeline);
        if let Some(p) = guard.as_ref() {
            return Ok(p.clone());
        }
        let device = candle_audio::default_device()?;
        let c = &self.components;
        let built = Arc::new(MmAudioPipeline::from_components(
            &c.clip, &c.sync, &c.dit, &c.vae, &c.vocoder, &device,
        )?);
        *guard = Some(built.clone());
        Ok(built)
    }
}

/// Construct the (lazy) generator from a [`LoadSpec`]. The five checkpoints are provisioned as named
/// [`LoadSpec::components`] — `clip` / `synchformer` / `dit` / `vae` / `vocoder`
/// ([`REQUIRED_COMPONENTS`]) — each required at load via [`require_component`], so a missing component
/// is a **load-time** contract error rather than a mid-render surprise (epic 13657, sc-13666). There
/// is no assembled snapshot directory: `spec.weights` is unused (mmaudio is a pure assembly of the
/// five named components). Unrecognized component keys, quantization, adapters, and control /
/// IP-adapter overlays are rejected — refusing is more honest than silently dropping.
pub fn load(spec: &LoadSpec) -> gen_core::Result<Box<dyn Generator>> {
    reject_unknown_components(spec, REQUIRED_COMPONENTS, MODEL_ID)?;
    let clip =
        require_component(spec, "clip", MODEL_ID, "DFN5B-CLIP ViT-H/14 conditioner")?.clone();
    let sync =
        require_component(spec, "synchformer", MODEL_ID, "Synchformer visual encoder")?.clone();
    let dit = require_component(spec, "dit", MODEL_ID, "small_16k MM-DiT network")?.clone();
    let vae = require_component(spec, "vae", MODEL_ID, "16k mel-VAE decoder")?.clone();
    let vocoder = require_component(spec, "vocoder", MODEL_ID, "16k BigVGAN vocoder")?.clone();
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
    Ok(Box::new(MmAudioGenerator {
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

impl Generator for MmAudioGenerator {
    fn descriptor(&self) -> &ModelDescriptor {
        &self.descriptor
    }

    fn validate(&self, req: &GenerationRequest) -> gen_core::Result<()> {
        validate_request(&self.descriptor, req)
    }

    fn generate(
        &self,
        req: &GenerationRequest,
        on_progress: &mut dyn FnMut(Progress),
    ) -> gen_core::Result<GenerationOutput> {
        self.validate(req)?;
        // Pre-generate cancellation seam: consult the flag before ANY heavy work (pipeline build,
        // encode, denoise all come after this).
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

// Explicit catalog registration for `mmaudio_small_16k` (composed by `candle-audio-catalog`).
candle_audio::register_generators! {
    pub const REGISTRATION = descriptor => load
}

#[cfg(test)]
mod tests {
    use super::*;
    use candle_audio::gen_core::{AudioParams, CancelFlag};

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
    fn descriptor_advertises_the_video_to_audio_surface() {
        let d = descriptor();
        assert_eq!(d.id, "mmaudio_small_16k");
        assert_eq!(d.family, "mmaudio");
        assert_eq!(d.backend, "candle");
        assert!(matches!(d.modality, Modality::Audio));
        assert_eq!(d.capabilities.audio_sample_rates, [16_000]);
        assert_eq!(d.capabilities.max_audio_duration_secs, Some(8.0));
        assert_eq!(
            d.capabilities.conditioning,
            vec![ConditioningKind::VideoSync]
        );
        assert!(d.capabilities.supports_guidance);
        assert!(d.capabilities.supports_negative_prompt);
        assert!(!d.capabilities.supports_streaming);
        assert!(!d.capabilities.supports_multi_speaker);
        assert_eq!(d.capabilities.max_count, 1);
        assert!(d.capabilities.audio_voices.is_empty());
        // The five named components (epic 13657, sc-13666) the caller must provision — pins the input
        // contract so a change to the component surface is deliberate (CONTRIBUTING.md).
        assert_eq!(
            d.required_components,
            ["clip", "synchformer", "dit", "vae", "vocoder"]
        );
    }

    #[test]
    fn seq_lengths_match_the_reference_config() {
        // 8 s → the CONFIG_16K assertions (latent 250, clip 64, sync 192).
        assert_eq!(seq_lengths(8.0), (250, 64, 192));
        // 1 s (the testkit clip) → latent 32, clip 8, sync 16 (2 segments × 8).
        assert_eq!(seq_lengths(1.0), (32, 8, 16));
        // A short-but-valid clip just above the 0.64 s floor → exactly one sync segment (8 tokens).
        let (_l, _c, sync) = seq_lengths(0.72);
        assert_eq!(sync, 8, "~0.7s is exactly one 16-frame segment");
    }

    #[test]
    fn validate_gates_the_conditioning_and_sampling_surface() {
        let d = descriptor();
        // An in-surface clip (8 frames @ 8 fps = 1 s) passes.
        let ok = foley_req(foley_frames(8, 16, 16, 0), 8);
        assert!(
            validate_request(&d, &ok).is_ok(),
            "valid Foley clip must pass"
        );

        // Un-advertised conditioning is the shared floor's job, but a too-short clip is ours: 8 frames
        // @ 25 fps = 0.32 s < 0.64 s (one segment) → typed Msg.
        let short = foley_req(foley_frames(8, 16, 16, 0), 25);
        assert!(
            validate_request(&d, &short).is_err(),
            "sub-segment clip rejected"
        );

        // Missing fps on a VideoSync clip → Msg.
        let mut no_fps = foley_req(foley_frames(8, 16, 16, 0), 8);
        no_fps.fps = None;
        assert!(
            validate_request(&d, &no_fps).is_err(),
            "VideoSync needs req.fps"
        );

        // Out-of-range sampling knobs.
        let mut bad_steps = foley_req(foley_frames(8, 16, 16, 0), 8);
        bad_steps.steps = Some(MAX_STEPS + 1);
        assert!(validate_request(&d, &bad_steps).is_err());
        let mut bad_cfg = foley_req(foley_frames(8, 16, 16, 0), 8);
        bad_cfg.guidance = Some(0.5);
        assert!(validate_request(&d, &bad_cfg).is_err());

        // Duration above the trained 8 s window → rejected by the shared floor.
        let mut long = foley_req(foley_frames(80, 16, 16, 0), 8);
        long.audio = Some(AudioParams {
            target_duration: Some(9.0),
            ..Default::default()
        });
        assert!(validate_request(&d, &long).is_err(), "target > 8s rejected");

        // Empty prompt is allowed (video-only Foley is first-class).
        let mut empty_prompt = foley_req(foley_frames(8, 16, 16, 0), 8);
        empty_prompt.prompt = String::new();
        assert!(validate_request(&d, &empty_prompt).is_ok());
    }

    #[test]
    fn effective_duration_caps_by_target_clip_and_window() {
        // No target → capped by clip length (2 s here) then the 8 s window.
        let r = foley_req(foley_frames(16, 16, 16, 0), 8);
        assert!((effective_duration(&r, 2.0) - 2.0).abs() < 1e-6);
        // A shorter target wins.
        let mut r = foley_req(foley_frames(80, 16, 16, 0), 8);
        r.audio = Some(AudioParams {
            target_duration: Some(3.0),
            ..Default::default()
        });
        assert!((effective_duration(&r, 8.0) - 3.0).abs() < 1e-6);
        // The 8 s window caps a longer clip with no target.
        let r = foley_req(foley_frames(800, 16, 16, 0), 8);
        assert!((effective_duration(&r, 100.0) - 8.0).abs() < 1e-6);
    }

    /// A weights-free [`LoadSpec`] that stages every required component (placeholder paths — `load`
    /// is lazy and reads no file, so the pipeline is not built here). `weights` is an ignored
    /// placeholder (mmaudio is a pure assembly of the five named components).
    fn staged_spec() -> LoadSpec {
        let dir = std::env::temp_dir().join("mmaudio-16k-staged");
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
        // Components point at nonexistent files, but a pre-tripped cancel must return `Canceled`
        // before the lazy pipeline build ever touches them.
        let g = load(&staged_spec()).unwrap();
        let flag = CancelFlag::new();
        flag.cancel();
        let mut req = foley_req(foley_frames(8, 16, 16, 0), 8);
        req.cancel = flag;
        let err = g.generate(&req, &mut |_| {}).unwrap_err();
        assert!(matches!(err, gen_core::Error::Canceled));
    }

    #[test]
    fn generate_on_missing_component_weights_fails_cleanly() {
        // Components staged but their files do not exist → the lazy pipeline build fails with a
        // non-`Canceled` error (not a mid-render surprise).
        let g = load(&staged_spec()).unwrap();
        let req = foley_req(foley_frames(8, 16, 16, 0), 8);
        let err = g.generate(&req, &mut |_| {}).unwrap_err();
        assert!(!matches!(err, gen_core::Error::Canceled));
    }
}
