//! # candle-gen-ltx
//!
//! The **LTX-2.3 (distilled 22B)** text-to-video provider for [`candle-gen`](candle_gen) — the candle
//! (Windows/CUDA) sibling of `mlx-gen-ltx`. LTX has **no** `candle-transformers` reference: the
//! `AVTransformer3DModel` video DiT ([`transformer`]), the `CausalVideoAutoencoder` temporal VAE
//! ([`vae`], on a from-scratch [`conv3d`]), the **Gemma-3-12B** text encoder ([`gemma`]) +
//! per-token-RMS aggregation + 8-layer learnable-register connector ([`text_encoder`], [`connector`])
//! are all ported here. The distilled rectified-flow denoise runs through the unified
//! `candle_gen::run_av_curated_sampler` over the fixed `STAGE1_SIGMAS`
//! schedule (epic 7114), so no per-crate scheduler module is needed.
//!
//! **video+audio (sc-3698 / sc-5495):** [`LtxGenerator::generate`] runs Gemma-3-12B → video +
//! audio text projections → connectors → the 48-layer dual-modal `AvDiT` (split
//! 3-D RoPE, per-head gated attention, adaLN-single, bidirectional cross-modal attention) joint
//! denoise → the temporal VAE decoder (frames) **plus** the `AudioDecoder`
//! → `LtxVocoder` → a synchronized 48 kHz stereo `AudioTrack`. Registered under
//! `"ltx_2_3_distilled"`; two-stage distilled denoise (no CFG). Reference I2V, FLF/keyframes,
//! extend/bridge IC-LoRA clips, and masked replace-person controls share the VAE encoder and per-token
//! timestep path. The learned 2-stage latent upsampler runs between half-resolution stage one and
//! full-resolution stage two; prompt-enhance and fp8/on-the-fly quant remain deferred. LTX AudioVideo
//! projection adapters are supported on both dense and packed tiers.
//!
//! **Dtypes:** the DiT, connector, text projection, and Gemma encoder run **bf16** (the checkpoint's
//! native dtype; 22B+12B does not fit f32 on a single 96 GB GPU); the VAE runs **f32**; attention and
//! norms upcast to f32. `backend = "candle"`, `mac_only = false`.
//!
//! **Weights:** `spec.weights` points at an LTX-2.3 snapshot dir (the
//! `ltx-2.3-22b-distilled.safetensors` single-file checkpoint bundling DiT + VAE + projection +
//! connector). The Gemma-3-12B encoder + its `tokenizer.json` live in a separate snapshot, provisioned
//! by the caller through the **`LoadSpec::text_encoder`** slot (or co-located at `<root>/text_encoder`).
//! As of sc-13749 there is no environment side-channel or HF-cache scan — an absent encoder is a
//! load-time, actionable error naming the slot (epic 13657; the candle sibling of sc-13664).

pub mod adapters;
pub mod audio_vae;
pub mod conditioning;
pub mod config;
pub mod connector;
pub mod conv3d;
pub mod dit_train;
pub mod gemma;
pub mod pipeline;
pub mod quant;
pub mod rope;
pub mod text_encoder;
pub mod tier;
pub mod training;
pub mod transformer;
pub mod upsampler;
pub mod vae;
pub mod vocoder;

use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex};

use candle_gen::candle_core::{DType, Device, Tensor};
#[cfg(test)]
use candle_gen::gen_core::AdapterKind;
use candle_gen::gen_core::{
    self, AdapterSpec, AudioTrack, Capabilities, Conditioning, ConditioningKind, GenerationOutput,
    GenerationRequest, Generator, Image, LoadSpec, Modality, ModelDescriptor, PerComponentBytes,
    Progress, Quant, StepSupport, WeightsSource,
};
use candle_gen::{run_av_curated_sampler, AvLatents, CandleError, Result as CResult};

use audio_vae::AudioDecoder;
use config::{
    compute_audio_frames, AudioVaeConfig, AvConfig, ConnectorConfig, GemmaConfig, VocoderConfig,
    DEFAULT_FPS, DEFAULT_FRAMES, MODEL_ID, NATIVE_STEPS, STAGE1_SIGMAS, STAGE2_SIGMAS,
    TEXT_MAX_LENGTH,
};
use text_encoder::LtxTextEncoder;
use transformer::AvDiT;
use upsampler::LatentUpsampler;
use vae::LtxVideoVae;
/// Provider-facing LTX geometry, derived from the decoder implementation.
pub const VAE_TILING: candle_gen::gen_core::tiling::VaeTiling = LtxVideoVae::VAE_TILING;
use vocoder::LtxVocoder;

const DIT_DTYPE: DType = DType::BF16;
const VAE_DTYPE: DType = DType::F32;

#[cfg(test)]
mod vae_tiling_assignment_tests {
    #[test]
    fn provider_id_resolves_to_the_concrete_decoder_geometry() {
        assert_eq!(
            super::VAE_TILING,
            candle_gen::gen_core::tiling::VaeTiling::LTX
        );
        assert_eq!(super::VAE_TILING, super::LtxVideoVae::VAE_TILING);
        assert_eq!(super::vae_tiling(super::MODEL_ID), Some(super::VAE_TILING));
        assert_eq!(super::vae_tiling("ltx_2_3"), None);
    }
}
/// The request width/height multiple `validate` enforces (= `2×config::SPATIAL_SCALE` = 64): both
/// LTX backends run stage one on the half-resolution VAE grid and stage two on the final grid.
/// Exposed as the pinned-engine stride SceneWorks ties `requiresDimensionsMultipleOf` to.
pub const SIZE_MULTIPLE: u32 = (config::SPATIAL_SCALE * 2) as u32;

#[derive(Clone)]
struct Components {
    te: Arc<LtxTextEncoder>,
    avdit: Arc<AvDiT>,
    vae: Arc<LtxVideoVae>,
    upsampler: Arc<LatentUpsampler>,
    vae_has_encoder: bool,
    /// Audio decode chain — `None` on the packed MLX tier path (sc-9545), which is **video-only**: the
    /// tier's audio-VAE + vocoder ship in a different key layout (channels-last convs, no `decoder.`/
    /// `vocoder.` prefix) that is a separate ingestion slice (follow-up), and the sc-9417 render AC is a
    /// video render. The audio latent stream still runs through the joint AvDiT (cross-modal coupling
    /// keeps the video coherent); only the audio VAE→vocoder decode is skipped.
    audio: Option<AudioChain>,
    tokenizer: Arc<tokenizers::Tokenizer>,
}

#[derive(Clone)]
struct AudioChain {
    decoder: Arc<AudioDecoder>,
    vocoder: Arc<LtxVocoder>,
    sample_rate: u32,
}

struct EncodedKeyframe {
    latent: Tensor,
    frame_idx: usize,
    strength: f32,
}

struct EncodedClip {
    latent: Tensor,
    /// Output-frame coordinate consumed by the appended-token RoPE path.
    frame_offset: i32,
    strength: f32,
}

struct Pipeline {
    av_cfg: AvConfig,
    gemma_cfg: GemmaConfig,
    conn_cfg: ConnectorConfig,
    audio_conn_cfg: ConnectorConfig,
    audio_vae_cfg: AudioVaeConfig,
    vocoder_cfg: VocoderConfig,
    root: PathBuf,
    device: Device,
    /// Gemma-encoder path from `LoadSpec::text_encoder` (sc-8827); see [`Pipeline::gemma_dir`].
    gemma_override: Option<PathBuf>,
    upsampler_override: Option<PathBuf>,
}

impl Pipeline {
    fn load(
        root: &Path,
        device: &Device,
        gemma_override: Option<PathBuf>,
        upsampler_override: Option<PathBuf>,
    ) -> Self {
        Self {
            av_cfg: AvConfig::ltx_2_3(),
            gemma_cfg: GemmaConfig::gemma_3_12b(),
            conn_cfg: ConnectorConfig::ltx_2_3(),
            audio_conn_cfg: ConnectorConfig::ltx_2_3_audio(),
            audio_vae_cfg: AudioVaeConfig::ltx_2_3(),
            vocoder_cfg: VocoderConfig::ltx_2_3(),
            root: root.to_path_buf(),
            device: device.clone(),
            gemma_override,
            upsampler_override,
        }
    }

    /// The single full **dense bf16** LTX-2.3 checkpoint in `root` — the 22B model bundling DiT + VAE +
    /// audio-VAE + vocoder + projection (not a LoRA / upscaler / fp8 variant). Handles both the base
    /// `Lightricks/LTX-2.3` (`ltx-2.3-22b-distilled*.safetensors`) and full-model fine-tunes whose file
    /// is named differently (e.g. the eros merge's `10Eros_v1_bf16.safetensors`, sc-5495): the snapshot
    /// may carry several `.safetensors` (bf16 + fp8 variants), so prefer `distilled`, then a `bf16`
    /// dense file, then the largest remaining — fp8/mixed are skipped (candle loads the bf16 weights).
    fn ltx_checkpoint(&self) -> CResult<PathBuf> {
        ltx_checkpoint_in(&self.root)
    }

    /// The Gemma-3-12B encoder snapshot dir. A `LoadSpec::text_encoder` path (sc-8827) wins; else the
    /// co-located `<root>/text_encoder` (sc-13749 — no environment / HF-cache scan any more).
    fn gemma_dir(&self) -> CResult<PathBuf> {
        gemma_dir_for(&self.root, self.gemma_override.as_deref())
    }

    fn upsampler_file(&self) -> CResult<PathBuf> {
        if let Some(path) = &self.upsampler_override {
            return canonical_upsampler_file(path);
        }
        canonical_upsampler_file(&self.root)
    }

    fn safetensors_in(dir: &Path) -> CResult<Vec<PathBuf>> {
        // Shared sorted-`.safetensors` resolver (sc-8999 / F-019).
        candle_gen::sorted_safetensors(dir, "ltx")
    }

    fn load_components(
        &self,
        adapters: &[AdapterSpec],
        with_vae_encoder: bool,
    ) -> CResult<Components> {
        // sc-9545: a packed MLX split-tier subdir (`.../q4` or `.../q8`) is ingested through the
        // remapping VarBuilders in `tier` so the sc-9417 packed-detect seam fires on the real tier
        // weights with no dense staging; the single-bundle dense checkpoint keeps the legacy path below.
        if let Some(paths) = tier::TierPaths::detect(&self.root, self.gemma_override.as_deref()) {
            return self.load_components_tier(&paths, adapters, with_vae_encoder);
        }

        let ltx_file = self.ltx_checkpoint()?;
        let gemma_dir = self.gemma_dir()?;
        let gemma_files = Self::safetensors_in(&gemma_dir)?;

        // Two builders over the single LTX file: bf16 (DiT + projection + connector), f32 (VAE).
        let ltx_files = [ltx_file];
        let vb_bf16 = candle_gen::mmap_var_builder(&ltx_files, DIT_DTYPE, &self.device)?;
        let vb_f32 = candle_gen::mmap_var_builder(&ltx_files, VAE_DTYPE, &self.device)?;
        let upsampler_vb =
            candle_gen::mmap_var_builder(&[self.upsampler_file()?], VAE_DTYPE, &self.device)?;
        let gemma_vb = candle_gen::mmap_var_builder(&gemma_files, DIT_DTYPE, &self.device)?
            .pp("language_model.model");

        let dit_vb = vb_bf16.pp("model.diffusion_model");
        let mut avdit = AvDiT::new(dit_vb.clone(), &self.av_cfg)?;
        adapters::install_ltx_adapters(&mut avdit, adapters)?;
        let te = LtxTextEncoder::new_av(
            gemma_vb,
            vb_bf16.clone(),
            dit_vb,
            &self.gemma_cfg,
            &self.conn_cfg,
            &self.audio_conn_cfg,
        )?;
        let vae = if with_vae_encoder {
            LtxVideoVae::new_with_encoder(
                vb_f32.pp("vae"),
                vb_f32.pp("vae"),
                config::LATENT_CHANNELS,
                4,
            )?
        } else {
            LtxVideoVae::new(vb_f32.pp("vae"), config::LATENT_CHANNELS, 4)?
        };
        let upsampler = LatentUpsampler::load(upsampler_vb)?;
        // The audio VAE decoder + vocoder run f32 (post-sampling quality islands).
        let audio_decoder = AudioDecoder::load(&vb_f32.pp("audio_vae"), &self.audio_vae_cfg)?;
        let vocoder = LtxVocoder::load(vb_f32, &self.device, &self.vocoder_cfg)?;
        let audio_sample_rate = self.vocoder_cfg.final_sample_rate() as u32;

        let tok_path = gemma_dir.join("tokenizer.json");
        let tokenizer = tokenizers::Tokenizer::from_file(&tok_path)
            .map_err(|e| CandleError::Msg(format!("ltx: load gemma tokenizer: {e}")))?;

        Ok(Components {
            te: Arc::new(te),
            avdit: Arc::new(avdit),
            vae: Arc::new(vae),
            upsampler: Arc::new(upsampler),
            vae_has_encoder: with_vae_encoder,
            audio: Some(AudioChain {
                decoder: Arc::new(audio_decoder),
                vocoder: Arc::new(vocoder),
                sample_rate: audio_sample_rate,
            }),
            tokenizer: Arc::new(tokenizer),
        })
    }

    /// Load the DiT (packed) + connectors/text-projection (dense) + video VAE (dense) + Gemma TE
    /// straight from the split MLX packed tier (sc-9545). The DiT builder applies the crate→tier key
    /// remap so [`crate::quant::qlinear`]'s packed-detect fires on the real `.scales` siblings; the
    /// group_size is read + validated from `quantize_config.json` (AC). **Video-only**: the tier's
    /// audio-VAE + vocoder are a separate ingestion slice (channels-last, differently-prefixed) tracked
    /// as a follow-up — the audio latent stream still flows through the joint AvDiT, only its final
    /// VAE→vocoder decode is skipped.
    fn load_components_tier(
        &self,
        paths: &tier::TierPaths,
        adapters: &[AdapterSpec],
        with_vae_encoder: bool,
    ) -> CResult<Components> {
        // Read + validate the tier's group_size (AC): errors loudly if a tier ever ships a group the
        // packed loaders don't repack at, rather than mis-aligning the MLX→GGML repack.
        let _group = paths.validate_group_size()?;

        let dit_vb = paths.dit_vb(DIT_DTYPE, &self.device)?;
        let conn_vb = paths.connector_vb(DIT_DTYPE, &self.device)?;
        let vae_vb = paths.vae_vb(VAE_DTYPE, &self.device)?;
        // Explicit component sources take precedence even for a split tier;
        // otherwise the canonical co-located tier file is used.
        let upsampler_vb = if self.upsampler_override.is_some() {
            candle_gen::mmap_var_builder(&[self.upsampler_file()?], VAE_DTYPE, &self.device)?
        } else {
            paths.upsampler_vb(VAE_DTYPE, &self.device)?
        };
        let gemma_vb = paths.gemma_vb(DIT_DTYPE, &self.device)?;

        // The DiT loader roots at `model.diffusion_model.` (the remap strips it); the connector loader
        // is handed a `model.diffusion_model.`-prefixed builder too (the remap strips it), and the text
        // projection sits at the connector-file root (also reached through that builder).
        let dit_root = dit_vb.pp("model.diffusion_model");
        let conn_root = conn_vb.pp("model.diffusion_model");
        let mut avdit = AvDiT::new(dit_root.clone(), &self.av_cfg)?;
        adapters::install_ltx_adapters(&mut avdit, adapters)?;
        let te = LtxTextEncoder::new_av(
            gemma_vb,
            conn_root.clone(),
            conn_root,
            &self.gemma_cfg,
            &self.conn_cfg,
            &self.audio_conn_cfg,
        )?;
        let vae = if with_vae_encoder {
            LtxVideoVae::new_with_encoder(
                vae_vb.pp("vae"),
                paths.vae_encoder_vb(VAE_DTYPE, &self.device)?.pp("vae"),
                config::LATENT_CHANNELS,
                4,
            )?
        } else {
            LtxVideoVae::new(vae_vb.pp("vae"), config::LATENT_CHANNELS, 4)?
        };
        let upsampler = LatentUpsampler::load(upsampler_vb)?;

        let tok_path = paths.tokenizer_path();
        let tokenizer = tokenizers::Tokenizer::from_file(&tok_path)
            .map_err(|e| CandleError::Msg(format!("ltx tier: load gemma tokenizer: {e}")))?;

        Ok(Components {
            te: Arc::new(te),
            avdit: Arc::new(avdit),
            vae: Arc::new(vae),
            upsampler: Arc::new(upsampler),
            vae_has_encoder: with_vae_encoder,
            audio: None,
            tokenizer: Arc::new(tokenizer),
        })
    }

    /// Tokenize `prompt` with the Gemma tokenizer (BOS, right-truncate then **left-pad** to
    /// `TEXT_MAX_LENGTH`), returning `(input_ids [1, 256] u32, mask01 [256])`.
    fn tokenize(&self, tok: &tokenizers::Tokenizer, prompt: &str) -> CResult<(Tensor, Vec<u32>)> {
        let enc = tok
            .encode(prompt, true)
            .map_err(|e| CandleError::Msg(format!("ltx: tokenize: {e}")))?;
        let mut ids: Vec<u32> = enc.get_ids().to_vec();
        let max = TEXT_MAX_LENGTH;
        if ids.len() > max {
            ids.truncate(max);
        }
        let nv = ids.len();
        let pad = max - nv;
        let mut padded = vec![0u32; pad];
        padded.extend_from_slice(&ids);
        let mut mask = vec![0u32; pad];
        mask.extend(std::iter::repeat_n(1u32, nv));
        let input_ids = Tensor::from_vec(padded, (1, max), &self.device)?;
        Ok((input_ids, mask))
    }

    fn latent_index(raw: i32, latent_frames: usize, label: &str) -> CResult<usize> {
        let resolved = if raw < 0 {
            latent_frames as i32 + raw
        } else {
            raw
        };
        if resolved < 0 || resolved >= latent_frames as i32 {
            return Err(CandleError::Msg(format!(
                "ltx: {label} latent frame index {raw} is out of bounds for {latent_frames} frames"
            )));
        }
        Ok(resolved as usize)
    }

    fn encode_image(
        &self,
        vae: &LtxVideoVae,
        image: &Image,
        width: u32,
        height: u32,
    ) -> CResult<Tensor> {
        let video =
            conditioning::preprocess_conditioning_image(image, width, height, &self.device)?;
        Ok(vae.encode(&video)?)
    }

    /// Resolve and VAE-encode replace-latent inputs: a `Reference` is I2V at frame zero; explicit
    /// keyframes cover FLF and arbitrary latent-frame placement.
    fn build_keyframes(
        &self,
        req: &GenerationRequest,
        vae: &LtxVideoVae,
        latent_frames: usize,
        width: u32,
        height: u32,
    ) -> CResult<Vec<EncodedKeyframe>> {
        let mut out = Vec::new();
        let mut reference_seen = false;
        for entry in &req.conditioning {
            match entry {
                Conditioning::Reference { image, strength } => {
                    if reference_seen {
                        return Err(CandleError::Msg(
                            "ltx: multiple Reference images are not supported; use Keyframe entries"
                                .into(),
                        ));
                    }
                    reference_seen = true;
                    out.push(EncodedKeyframe {
                        latent: self.encode_image(vae, image, width, height)?,
                        frame_idx: 0,
                        strength: strength.or(req.strength).unwrap_or(1.0),
                    });
                }
                Conditioning::Keyframe {
                    image,
                    frame_idx,
                    strength,
                } => out.push(EncodedKeyframe {
                    latent: self.encode_image(vae, image, width, height)?,
                    frame_idx: Self::latent_index(*frame_idx, latent_frames, "keyframe")?,
                    strength: *strength,
                }),
                _ => {}
            }
        }
        Ok(out)
    }

    /// Resolve and VAE-encode IC-LoRA clips for extend/bridge and masked replace-person control.
    fn build_clips(
        &self,
        req: &GenerationRequest,
        vae: &LtxVideoVae,
        latent_frames: usize,
        width: u32,
        height: u32,
    ) -> CResult<Vec<EncodedClip>> {
        let mut out = Vec::new();
        for clip in req.video_clips() {
            let idx = Self::latent_index(clip.frame_idx, latent_frames, "clip")?;
            let video = conditioning::preprocess_conditioning_clip(
                clip.frames,
                width,
                height,
                &self.device,
            )?;
            out.push(EncodedClip {
                latent: vae.encode(&video)?,
                frame_offset: conditioning::latent_frame_to_output_offset(idx)?,
                strength: clip.strength,
            });
        }
        if let Some(control) = req.control_clip() {
            if control.frames.len() != control.mask.len() {
                return Err(CandleError::Msg(format!(
                    "ltx: replace-person frame count {} does not match mask count {}",
                    control.frames.len(),
                    control.mask.len()
                )));
            }
            let idx = Self::latent_index(control.start_frame, latent_frames, "replace-person")?;
            let masked = control
                .frames
                .iter()
                .zip(control.mask)
                .map(|(frame, mask)| {
                    conditioning::apply_replacement_mask(frame, mask, control.masking_strength)
                })
                .collect::<candle_gen::candle_core::Result<Vec<_>>>()?;
            let video =
                conditioning::preprocess_conditioning_clip(&masked, width, height, &self.device)?;
            out.push(EncodedClip {
                latent: vae.encode(&video)?,
                frame_offset: conditioning::latent_frame_to_output_offset(idx)?,
                strength: control.masking_strength,
            });
        }
        Ok(out)
    }

    fn render(
        &self,
        req: &GenerationRequest,
        comps: &Components,
        on_progress: &mut dyn FnMut(Progress),
    ) -> CResult<(Vec<Image>, u32, Option<AudioTrack>)> {
        let frames = req.frames.unwrap_or(DEFAULT_FRAMES);
        let fps = req.fps.unwrap_or(DEFAULT_FPS);
        let seed = req.seed.unwrap_or_else(gen_core::default_seed);
        let mut orchestration = pipeline::TwoStageOrchestration::new(seed);
        // Every render begins at the first distilled stage. The adapter overlay retains its complete
        // per-pass vector, so a two-stage caller switches this selector before its stage-two denoise
        // instead of collapsing Eros's `[1.0, 0.4]` at load time.

        // Text encode → video (1,256,4096) + audio (1,256,2048) contexts (one Gemma pass).
        let (input_ids, mask01) = self.tokenize(&comps.tokenizer, &req.prompt)?;
        let (video_ctx, audio_ctx) = comps.te.encode_both(&input_ids, &mask01)?;

        // Stage one lives on the half-resolution grid; the learned upsampler is
        // the only bridge to the full-resolution stage-two grid.
        let geometry = pipeline::two_stage_geometry(frames, req.width, req.height);
        let (t_lat, h_lat, w_lat) = (geometry.t, geometry.h1, geometry.w1);
        let af = compute_audio_frames(frames as usize, fps as f64).max(1);
        let video_grid = rope::create_position_grid(t_lat, h_lat, w_lat, fps as f32, &self.device)?;
        let audio_grid = rope::create_audio_position_grid(af, &self.device)?;

        let keyframes =
            self.build_keyframes(req, &comps.vae, t_lat, req.width / 2, req.height / 2)?;
        let stage2_keyframes =
            self.build_keyframes(req, &comps.vae, geometry.t, req.width, req.height)?;
        let clips = self.build_clips(req, &comps.vae, t_lat, req.width / 2, req.height / 2)?;
        let stage_seeds =
            orchestration.stage1_setup(!keyframes.is_empty(), !clips.is_empty(), |pass| {
                comps.avdit.set_adapter_pass(pass)
            });
        let vnoise =
            pipeline::create_noise(stage_seeds.video_stage1, t_lat, h_lat, w_lat, &self.device)?;
        let anoise = pipeline::create_audio_noise(stage_seeds.audio_stage1, af, &self.device)?;
        let conditioned = !keyframes.is_empty() || !clips.is_empty();
        if conditioned
            && !matches!(
                req.sampler.as_deref(),
                None | Some("euler") | Some("rectified-flow")
            )
        {
            return Err(CandleError::Msg(
                "ltx: image/keyframe/clip conditioning uses the native distilled Euler sampler; \
                 choose `euler`/`rectified-flow` or leave sampler unset"
                    .into(),
            ));
        }

        // Unified curated sampling over the JOINT video+audio streams (epic 7114 P4, sc-7125). LTX is
        // distilled rectified-flow with the fixed `STAGE1_SIGMAS` schedule, so per decision 3b it exposes
        // the SAMPLER axis but NO scheduler axis (the baked σ schedule is the native default). The
        // default `euler` reproduces the legacy per-stream `to_denoised`→`euler_step` loop exactly (the
        // FLOW `x0 = x − σ·v` recombine + euler == the native scheduler), the N1 no-op. Both streams are
        // velocity-prediction (`Sigma` convention); the AvDiT couples them via cross-modal attention each
        // forward, so the per-step model eval (flatten → AvDiT → unflatten) lives inside the closure.
        let mut stage1_fold = pipeline::StageProgressFold::new(0, NATIVE_STEPS, 11);
        let mut stage1_progress = |event: Progress| {
            if let Some(event) = stage1_fold.fold(event) {
                on_progress(event);
            }
        };
        let (vlat, alat) = if conditioned {
            let mut state = if keyframes.is_empty() {
                conditioning::VideoTokenState::base(&vnoise, &video_grid)?
            } else {
                let zeros = Tensor::zeros_like(&vnoise)?;
                let borrowed = keyframes
                    .iter()
                    .map(|keyframe| conditioning::Keyframe {
                        latent: &keyframe.latent,
                        frame_idx: keyframe.frame_idx,
                        strength: keyframe.strength,
                    })
                    .collect::<Vec<_>>();
                let i2v = conditioning::apply_keyframes(&zeros, &borrowed)?
                    .noised(&vnoise, STAGE1_SIGMAS[0])?;
                conditioning::VideoTokenState::from_i2v(&i2v, &video_grid)?
            };
            for clip in &clips {
                state = conditioning::append_keyframe_clip(
                    &state,
                    &clip.latent,
                    clip.frame_offset,
                    clip.strength,
                    fps as f32,
                )?;
            }
            let mut stage1_forward = || Ok(());
            let (state, audio) = pipeline::denoise_av_conditioned(
                &comps.avdit,
                &state,
                &anoise,
                &video_ctx,
                &audio_ctx,
                af,
                &audio_grid,
                &STAGE1_SIGMAS,
                &req.cancel,
                &mut stage1_forward,
                &mut stage1_progress,
            )?;
            let generated = state.latent.narrow(1, 0, state.target_tokens)?;
            (
                pipeline::unflatten_latent(&generated, t_lat, h_lat, w_lat)?,
                audio,
            )
        } else {
            let out = run_av_curated_sampler(
                req.sampler.as_deref(),
                &STAGE1_SIGMAS[..],
                AvLatents {
                    video: vnoise,
                    audio: anoise,
                },
                seed,
                &req.cancel,
                &mut stage1_progress,
                |av, sigma| -> CResult<AvLatents> {
                    let vflat = pipeline::flatten_latent(&av.video)?;
                    let aflat = pipeline::flatten_audio_latent(&av.audio)?;
                    let (vvel, avel) = comps.avdit.forward(
                        &vflat,
                        &aflat,
                        sigma as f64,
                        &video_ctx,
                        &audio_ctx,
                        &video_grid,
                        &audio_grid,
                    )?;
                    Ok(AvLatents {
                        video: pipeline::unflatten_latent(
                            &vvel.to_dtype(DType::F32)?,
                            t_lat,
                            h_lat,
                            w_lat,
                        )?,
                        audio: pipeline::unflatten_audio_latent(&avel.to_dtype(DType::F32)?, af)?,
                    })
                },
            )?;
            (out.video, out.audio)
        };

        // The denoised half-resolution latent enters the learned upsampler in
        // VAE space, then returns to DiT-normalized space before fresh stage-two
        // video/audio re-noise. Never substitute interpolation or a second pass
        // on the stage-one model output.
        let upsampled = orchestration.learned_upsample(|| {
            Ok(comps.vae.normalize_latents(
                &comps
                    .upsampler
                    .forward(&comps.vae.denormalize_latents(&vlat)?)?,
            )?)
        })?;
        let stage2_video_noise = pipeline::create_noise(
            stage_seeds.video_stage2,
            geometry.t,
            geometry.h2,
            geometry.w2,
            &self.device,
        )?;
        let stage2_audio_noise =
            pipeline::create_audio_noise(stage_seeds.audio_stage2, af, &self.device)?;
        let stage2_grid = rope::create_position_grid(
            geometry.t,
            geometry.h2,
            geometry.w2,
            fps as f32,
            &self.device,
        )?;
        let stage2_initial = orchestration.stage2_renoise(
            STAGE2_SIGMAS[0],
            || {
                Ok(AvLatents {
                    video: pipeline::renoise(&upsampled, &stage2_video_noise, STAGE2_SIGMAS[0])?,
                    audio: pipeline::renoise(&alat, &stage2_audio_noise, STAGE2_SIGMAS[0])?,
                })
            },
            |pass| comps.avdit.set_adapter_pass(pass),
        )?;
        let mut stage2_fold = pipeline::StageProgressFold::new(NATIVE_STEPS, 3, 11);
        let mut stage2_progress = |event: Progress| {
            if let Some(event) = stage2_fold.fold(event) {
                on_progress(event);
            }
        };
        let stage2 = if stage2_keyframes.is_empty() {
            run_av_curated_sampler(
                req.sampler.as_deref(),
                &STAGE2_SIGMAS,
                stage2_initial,
                stage_seeds.video_stage2,
                &req.cancel,
                &mut stage2_progress,
                |av, sigma| -> CResult<AvLatents> {
                    orchestration.stage2_forward(|| {
                        let vflat = pipeline::flatten_latent(&av.video)?;
                        let aflat = pipeline::flatten_audio_latent(&av.audio)?;
                        let (vvel, avel) = comps.avdit.forward(
                            &vflat,
                            &aflat,
                            sigma as f64,
                            &video_ctx,
                            &audio_ctx,
                            &stage2_grid,
                            &audio_grid,
                        )?;
                        Ok(AvLatents {
                            video: pipeline::unflatten_latent(
                                &vvel.to_dtype(DType::F32)?,
                                geometry.t,
                                geometry.h2,
                                geometry.w2,
                            )?,
                            audio: pipeline::unflatten_audio_latent(
                                &avel.to_dtype(DType::F32)?,
                                af,
                            )?,
                        })
                    })
                },
            )?
        } else {
            // FLF/I2V keys are encoded at both grids. Clips deliberately stop at
            // stage one: their appended-token positions are half-resolution IC-LoRA
            // controls, while stage two conditions only its target video tokens.
            let borrowed = stage2_keyframes
                .iter()
                .map(|keyframe| conditioning::Keyframe {
                    latent: &keyframe.latent,
                    frame_idx: keyframe.frame_idx,
                    strength: keyframe.strength,
                })
                .collect::<Vec<_>>();
            let conditioned = orchestration.stage2_keyframes(|| {
                Ok(conditioning::apply_keyframes(&upsampled, &borrowed)?
                    .noised(&stage2_video_noise, STAGE2_SIGMAS[0])?)
            })?;
            let state = conditioning::VideoTokenState::from_i2v(&conditioned, &stage2_grid)?;
            let mut stage2_forward = || {
                orchestration
                    .stage2_forward(|| Ok(()))
                    .map_err(|error| candle_gen::candle_core::Error::Msg(error.to_string()))
            };
            let (state, audio) = pipeline::denoise_av_conditioned(
                &comps.avdit,
                &state,
                &stage2_initial.audio,
                &video_ctx,
                &audio_ctx,
                af,
                &audio_grid,
                &STAGE2_SIGMAS,
                &req.cancel,
                &mut stage2_forward,
                &mut stage2_progress,
            )?;
            let generated = state.latent.narrow(1, 0, state.target_tokens)?;
            AvLatents {
                video: pipeline::unflatten_latent(
                    &generated,
                    geometry.t,
                    geometry.h2,
                    geometry.w2,
                )?,
                audio,
            }
        };
        let (vlat, alat) = (stage2.video, stage2.audio);

        on_progress(Progress::Decoding);
        // sc-7076 — memory-bounded + catchable VAE decode (budgeted tiling), replacing the single-pass
        // full-video decode that OOMs the worker on large/long outputs.
        let decoded = comps.vae.decode_budgeted(&vlat)?;
        let images = pipeline::frames_to_images(&decoded)?;
        // Audio decode only when the audio chain is loaded (the dense bundle); the packed MLX tier is
        // video-only (sc-9545) — its audio VAE/vocoder are a separate ingestion slice.
        let audio = match &comps.audio {
            Some(chain) => Some(pipeline::decode_audio_track(
                &chain.decoder,
                &chain.vocoder,
                &alat,
                chain.sample_rate,
            )?),
            None => None,
        };
        Ok((images, fps, audio))
    }
}

pub struct LtxGenerator {
    descriptor: ModelDescriptor,
    root: PathBuf,
    device: Device,
    /// Optional Gemma-encoder snapshot dir from `LoadSpec::text_encoder` (sc-8827); wins over the
    /// co-located `<root>/text_encoder` fallback in [`Pipeline::gemma_dir`] (sc-13749 — no env / cache).
    gemma_override: Option<PathBuf>,
    upsampler_override: Option<PathBuf>,
    adapters: Vec<AdapterSpec>,
    components: Mutex<Option<Components>>,
}

impl LtxGenerator {
    #[allow(clippy::unnecessary_map_or)] // `Option::is_none_or` is newer than the repository MSRV.
    fn components(&self, pipe: &Pipeline, with_vae_encoder: bool) -> gen_core::Result<Components> {
        let mut slot = candle_gen::lock_recover(&self.components);
        if slot.as_ref().map_or(true, |components| {
            components.vae_has_encoder != with_vae_encoder
        }) {
            // Switching request modes must not retain both VAE variants at once.
            *slot = None;
            *slot = Some(pipe.load_components(&self.adapters, with_vae_encoder)?);
        }
        Ok(slot.as_ref().expect("component cache populated").clone())
    }
}

fn needs_ltx_vae_encoder(req: &GenerationRequest) -> bool {
    !req.conditioning.is_empty()
}

impl Generator for LtxGenerator {
    fn descriptor(&self) -> &ModelDescriptor {
        &self.descriptor
    }

    fn validate(&self, req: &GenerationRequest) -> gen_core::Result<()> {
        self.descriptor
            .capabilities
            .validate_request(MODEL_ID, req)?;
        if req.prompt.is_empty() {
            return Err(gen_core::Error::Msg("ltx: prompt must not be empty".into()));
        }
        if !req.width.is_multiple_of(SIZE_MULTIPLE) || !req.height.is_multiple_of(SIZE_MULTIPLE) {
            return Err(gen_core::Error::Msg(format!(
                "ltx: width/height must be multiples of {SIZE_MULTIPLE} (got {}x{})",
                req.width, req.height
            )));
        }
        if let Some(f) = req.frames {
            if f == 0 || f % config::TEMPORAL_SCALE as u32 != 1 {
                return Err(gen_core::Error::Msg(format!(
                    "ltx: frames must satisfy frames % {} == 1 (got {f})",
                    config::TEMPORAL_SCALE
                )));
            }
        }
        let check_strength = |label: &str, strength: f32| -> gen_core::Result<()> {
            if !strength.is_finite() || !(0.0..=1.0).contains(&strength) {
                return Err(gen_core::Error::Msg(format!(
                    "ltx: {label} strength must be finite and in [0,1] (got {strength})"
                )));
            }
            Ok(())
        };
        if let Some(strength) = req.strength {
            check_strength("image", strength)?;
        }
        for entry in &req.conditioning {
            match entry {
                Conditioning::Reference {
                    strength: Some(strength),
                    ..
                } => check_strength("reference", *strength)?,
                Conditioning::Keyframe { strength, .. } => check_strength("keyframe", *strength)?,
                Conditioning::VideoClip {
                    frames, strength, ..
                } => {
                    check_strength("clip", *strength)?;
                    if frames.is_empty() || (frames.len() - 1) % config::TEMPORAL_SCALE != 0 {
                        return Err(gen_core::Error::Msg(format!(
                            "ltx: conditioning clip frame count must equal 1 + k*{} (got {})",
                            config::TEMPORAL_SCALE,
                            frames.len()
                        )));
                    }
                }
                Conditioning::ControlClip {
                    frames,
                    mask,
                    masking_strength,
                    ..
                } => {
                    check_strength("replace-person masking", *masking_strength)?;
                    if frames.len() != mask.len() {
                        return Err(gen_core::Error::Msg(format!(
                            "ltx: replace-person frame count {} does not match mask count {}",
                            frames.len(),
                            mask.len()
                        )));
                    }
                    if frames.is_empty() || (frames.len() - 1) % config::TEMPORAL_SCALE != 0 {
                        return Err(gen_core::Error::Msg(format!(
                            "ltx: replace-person clip frame count must equal 1 + k*{} (got {})",
                            config::TEMPORAL_SCALE,
                            frames.len()
                        )));
                    }
                }
                _ => {}
            }
        }
        if !req.conditioning.is_empty()
            && !matches!(
                req.sampler.as_deref(),
                None | Some("euler") | Some("rectified-flow")
            )
        {
            return Err(gen_core::Error::Unsupported(
                "ltx conditioned video uses native distilled Euler; choose euler/rectified-flow or leave sampler unset"
                    .into(),
            ));
        }
        // Bound the AvDiT denoise sequence length (F-131, sc-11234). The checks above bound only the
        // frame *shape*, never its magnitude, so a huge frame count (e.g. `frames: 2001`, which
        // satisfies `% 8 == 1`) at a large resolution produced ~400k latent tokens and OOM'd deep in
        // the 22B denoise loop rather than failing catchably here. The video latent token count
        // `t_lat · h_lat · w_lat` is the memory driver (self-attn working set + per-token q/k/v across
        // 48 layers); cap it against the GPU envelope. Uses the effective frame count (the render
        // default when `None`) and the already-validated (mult-of-32) width/height.
        let eff_frames = req.frames.unwrap_or(DEFAULT_FRAMES);
        let (t_lat, h_lat, w_lat) = pipeline::latent_dims(eff_frames, req.width, req.height);
        let resolve_idx = |raw: i32, label: &str| -> gen_core::Result<()> {
            let resolved = if raw < 0 { t_lat as i32 + raw } else { raw };
            if resolved < 0 || resolved >= t_lat as i32 {
                return Err(gen_core::Error::Msg(format!(
                    "ltx: {label} latent frame index {raw} is out of bounds for {t_lat} frames"
                )));
            }
            Ok(())
        };
        let mut reference_count = 0usize;
        let mut control_clip_count = 0usize;
        let mut appended_frames = 0usize;
        for entry in &req.conditioning {
            match entry {
                Conditioning::Reference { .. } => reference_count += 1,
                Conditioning::Keyframe { frame_idx, .. } => resolve_idx(*frame_idx, "keyframe")?,
                Conditioning::VideoClip {
                    frames, frame_idx, ..
                } => {
                    resolve_idx(*frame_idx, "clip")?;
                    appended_frames += (frames.len() - 1) / config::TEMPORAL_SCALE + 1;
                }
                Conditioning::ControlClip {
                    frames,
                    start_frame,
                    ..
                } => {
                    control_clip_count += 1;
                    resolve_idx(*start_frame, "replace-person")?;
                    appended_frames += (frames.len() - 1) / config::TEMPORAL_SCALE + 1;
                }
                _ => {}
            }
        }
        if reference_count > 1 {
            return Err(gen_core::Error::Msg(
                "ltx: multiple Reference images are not supported; use Keyframe entries".into(),
            ));
        }
        if control_clip_count > 1 {
            return Err(gen_core::Error::Msg(
                "ltx: exactly one ControlClip can be applied per request".into(),
            ));
        }
        let tokens = (t_lat + appended_frames) * h_lat * w_lat;
        let max_tokens = config::max_latent_tokens();
        if tokens > max_tokens {
            return Err(gen_core::Error::Msg(format!(
                "ltx: request too large — {eff_frames} frames at {}x{} is {tokens} latent tokens, \
                 over the {max_tokens}-token cap (the 22B AvDiT denoise loop would exceed the GPU \
                 memory envelope). Reduce the frame count or resolution, or raise \
                 LTX_MAX_LATENT_TOKENS for a larger-VRAM device.",
                req.width, req.height
            )));
        }
        // `req.steps` (sc-9027 / F-043) is enforced by the shared floor above, from
        // `Capabilities::supported_steps` — NOT by an `if` here. It used to be one, and that is
        // exactly how the two lanes drifted: candle refused `steps: 30` while mlx never read the
        // field (sc-19502). One declaration, one enforcement site, both lanes.
        Ok(())
    }

    fn generate(
        &self,
        req: &GenerationRequest,
        on_progress: &mut dyn FnMut(Progress),
    ) -> gen_core::Result<GenerationOutput> {
        self.validate(req)?;
        let pipe = Pipeline::load(
            &self.root,
            &self.device,
            self.gemma_override.clone(),
            self.upsampler_override.clone(),
        );
        let components = self.components(&pipe, needs_ltx_vae_encoder(req))?;
        let (frames, fps, audio) = pipe.render(req, &components, on_progress)?;
        Ok(GenerationOutput::Video { frames, fps, audio })
    }
}

/// LTX-2.3 distilled video descriptor — two-stage rectified-flow (no CFG / negative prompt;
/// guidance is distilled in) with image/keyframe/IC-LoRA clip conditioning. The denoise step count is
/// FIXED at [`NATIVE_STEPS`] (the baked
/// `STAGE1_SIGMAS` schedule); stage two always runs its fixed three-step `STAGE2_SIGMAS` refinement.
/// An explicit non-native `req.steps` is rejected in `validate` rather than silently ignored (sc-9027 /
/// F-043). Synchronized audio is produced (sc-5495, the joint video+audio streams); on-the-fly quant
/// remains deferred. AudioVideo projection adapters are supported through the shared additive adapter core.
pub fn descriptor() -> ModelDescriptor {
    ModelDescriptor {
        encoder_contract: None,
        denoiser_output_latent_space: Some(&candle_gen::gen_core::LTX_VIDEO_LATENT_SPACE),
        control_kinds: None,
        required_components: &[],
        id: MODEL_ID,
        family: "ltx",
        backend: "candle",
        modality: Modality::Video,
        capabilities: Capabilities {
            conditioning: vec![
                ConditioningKind::Reference,
                ConditioningKind::Keyframe,
                ConditioningKind::VideoClip,
                ConditioningKind::ControlClip,
            ],
            supports_lora: true,
            supports_lokr: true,
            // Unified curated SAMPLER menu (epic 7114 P4, sc-7125) over the joint video+audio streams +
            // the legacy `rectified-flow` alias (falls back to euler). Per decision 3b: sampler-only, NO
            // scheduler axis — LTX is distilled with the fixed `STAGE1_SIGMAS` schedule; `euler` is the
            // recommended default (the byte-faithful N1 path). The rest are exposed for ComfyUI parity.
            samplers: candle_gen::menu_with_aliases(
                candle_gen::curated_sampler_names(),
                &["rectified-flow"],
            ),
            min_size: SIZE_MULTIPLE,
            max_size: 1280,
            max_count: 1,
            // The distilled σ waypoints are baked into training, so 8 is not a default — it is the
            // ONLY renderable count (sc-9027 / F-043). Advertised rather than re-checked in
            // `validate` (sc-19502): the shared floor now owns the rejection, so `mlx-gen-ltx`
            // enforces the identical constraint from the identical declaration instead of, as it
            // did, never reading `req.steps` and silently rendering this same schedule anyway.
            //
            // Derived from the σ table rather than written as `vec![8]`, so re-baking the schedule
            // moves the advertised surface with it instead of leaving a stale literal behind.
            supported_steps: StepSupport::Exact(vec![NATIVE_STEPS]),
            supported_quants: &[] as &[Quant],
            ..Default::default()
        },
    }
}

/// The single full **dense bf16** LTX-2.3 checkpoint in `root` — the 22B model bundling DiT + VAE +
/// audio-VAE + vocoder + projection (not a LoRA / upscaler / fp8 variant). Handles both the base
/// `Lightricks/LTX-2.3` (`ltx-2.3-22b-distilled*.safetensors`) and full-model fine-tunes whose file is
/// named differently (e.g. the eros merge's `10Eros_v1_bf16.safetensors`, sc-5495): the snapshot may
/// carry several `.safetensors` (bf16 + fp8 variants), so prefer `distilled`, then a `bf16` dense file,
/// then the largest remaining — fp8/mixed are skipped (candle loads the bf16 weights).
///
/// **The single source of truth for which file the dense path loads** — [`Pipeline::ltx_checkpoint`]
/// mmaps it and [`component_footprint`] sizes it (sc-12397). Keeping the selection in one free function
/// is the whole point: the hosted `Lightricks/LTX-2.3` snapshot is ~146 GiB on disk against a ONE-file
/// load, so a consumer that sums the directory would over-predict by ~7x and refuse LTX on every GPU in
/// existence. Only this crate knows which file wins.
fn ltx_checkpoint_in(root: &Path) -> CResult<PathBuf> {
    let lname = |p: &Path| {
        p.file_name()
            .and_then(|n| n.to_str())
            .unwrap_or("")
            .to_ascii_lowercase()
    };
    let mut cands: Vec<PathBuf> = std::fs::read_dir(root)
        .map_err(|e| CandleError::Msg(format!("ltx: read snapshot dir: {e}")))?
        .filter_map(|e| e.ok().map(|e| e.path()))
        .filter(|p| {
            let name = lname(p);
            name.ends_with(".safetensors")
                && !name.contains("lora")
                && !name.contains("upscaler")
                && !name.contains("upsampler")
                && !name.contains("fp8")
                && !name.contains("mixed")
        })
        .collect();
    cands.sort();
    if cands.is_empty() {
        return Err(CandleError::Msg(format!(
            "ltx: no dense LTX-2.3 `.safetensors` checkpoint in {} (expected e.g. \
             `ltx-2.3-22b-distilled.safetensors` or a `*_bf16.safetensors` full-model fine-tune)",
            root.display()
        )));
    }
    if let Some(p) = cands.iter().find(|p| lname(p).contains("distilled")) {
        return Ok(p.clone());
    }
    if let Some(p) = cands.iter().find(|p| lname(p).contains("bf16")) {
        return Ok(p.clone());
    }
    // No name hint — the full dense model dwarfs any aux file, so take the largest.
    Ok(cands
        .into_iter()
        .max_by_key(|p| std::fs::metadata(p).map(|m| m.len()).unwrap_or(0))
        .expect("cands non-empty"))
}

/// Resolve the published learned refinement component. A `File` source is
/// exact; a directory source and ordinary snapshot use the canonical filename.
fn canonical_upsampler_file(path: &Path) -> CResult<PathBuf> {
    if path.is_file() {
        return Ok(path.to_path_buf());
    }
    let canonical = path.join("upsampler.safetensors");
    if canonical.is_file() {
        return Ok(canonical);
    }
    Err(CandleError::Msg(format!(
        "ltx requires the learned spatial upscaler — provide LoadSpec::components[\"spatial_upscaler\"] \
         as upsampler.safetensors or a directory containing it (looked in {})",
        canonical.display()
    )))
}

fn spec_upsampler_file(spec: &LoadSpec, root: &Path) -> CResult<PathBuf> {
    match spec
        .components
        .get(gen_core::LTX_SPATIAL_UPSCALER_COMPONENT)
    {
        Some(WeightsSource::File(path)) | Some(WeightsSource::Dir(path)) => {
            canonical_upsampler_file(path)
        }
        None => canonical_upsampler_file(root),
    }
}

/// The Gemma-3-12B encoder snapshot dir for a `root` + the `LoadSpec::text_encoder` path (sc-8827):
/// the caller-supplied path wins; else the co-located `<root>/text_encoder`. Both are **passed-in**
/// paths (the override rides the spec; `root` is `LoadSpec::weights`) — as of sc-13749 there is no
/// environment side-channel and no HF-cache scan (epic 13657, the candle sibling of sc-13664): an
/// absent encoder is a load-time, actionable error naming the slot.
///
/// Shared by [`Pipeline::gemma_dir`] and [`component_footprint`] so the gate sizes the encoder the load
/// will actually read. Note this is the DENSE path's precedence; the packed tier resolves its Gemma via
/// [`tier::TierPaths::detect`] (the spec path, else the tier's sibling `gemma/`) — also passed-in paths
/// only. [`component_footprint`] mirrors that split rather than assuming one rule.
fn gemma_dir_for(root: &Path, gemma_override: Option<&Path>) -> CResult<PathBuf> {
    if let Some(p) = gemma_override {
        if !p.is_dir() {
            return Err(CandleError::Msg(format!(
                "ltx: LoadSpec text_encoder path is not a directory: {}",
                p.display()
            )));
        }
        return Ok(p.to_path_buf());
    }
    let colocated = root.join("text_encoder");
    if colocated.is_dir() {
        return Ok(colocated);
    }
    Err(CandleError::Msg(format!(
        "ltx requires the Gemma-3-12B text encoder — set LoadSpec::text_encoder to a \
         google/gemma-3-12b-it snapshot dir (or co-locate it at <root>/text_encoder, i.e. {}). It is \
         no longer auto-discovered from an environment variable or the HF cache.",
        colocated.display()
    )))
}

/// The snapshot root a `spec` loads from — a `Dir` as-is, a `File`'s parent (LTX is the one video
/// provider that accepts a single-file source). Mirrors [`load`].
fn spec_root(spec: &LoadSpec) -> PathBuf {
    match &spec.weights {
        WeightsSource::Dir(p) => p.clone(),
        WeightsSource::File(p) => p
            .parent()
            .map(Path::to_path_buf)
            .unwrap_or_else(|| p.clone()),
    }
}

/// The provider-owned per-component on-disk footprint (sc-12397, epic 1788) — the size of the exact
/// files a load will mmap, NOT a directory sum.
///
/// Lets a pre-load fit gate size an LTX job honestly. The consumer (`sceneworks-worker`'s candle video
/// VRAM gate) cannot compute this itself, and the gap is not marginal:
///  * **dense** — [`ltx_checkpoint_in`] picks ONE root file out of a snapshot that also ships
///    `fp8`/`mixed`/lora/upscaler siblings. Hosted `Lightricks/LTX-2.3` is ~146 GiB on disk against that
///    single-file load, so a directory sum refuses LTX on every GPU that exists.
///  * **packed tier** — the load reads 5 files (`transformer` + `connector` + `vae_decoder` +
///    `vae_encoder` + learned `upsampler`); the encoder is required by every advertised
///    video-conditioning lane and the upsampler by every render.
///
/// Mapping onto [`PerComponentBytes`]' three slots: `text_encoder` = the Gemma-3-12B encoder (a
/// SEPARATE ~24 GB snapshot that is not under the weights root — omitting it would under-count by more
/// than the DiT). `dit` = the transformer, plus the connector on the tier path. `vae` = the tier's
/// `vae_decoder`; on the dense path it is **0** because the VAE is bundled inside the one checkpoint
/// already counted in `dit` — the slots are a partition of the load, never double-counted.
///
/// A component that cannot be resolved contributes `0` rather than erroring: the footprint is a pre-load
/// ADMISSION signal, and reporting no signal (⇒ the caller admits) is safer than refusing a job over an
/// unreadable path. `load_components` reports the real error moments later. In particular a dense
/// snapshot with no resolvable checkpoint, or an absent Gemma dir (no `LoadSpec::text_encoder` and no
/// `<root>/text_encoder` — not an error at gate time), simply reads 0.
pub(crate) fn component_footprint(spec: &LoadSpec) -> gen_core::Result<PerComponentBytes> {
    let root = spec_root(spec);
    let gemma_override = spec.text_encoder.as_ref().map(|src| match src {
        WeightsSource::Dir(p) | WeightsSource::File(p) => p.clone(),
    });
    // The tier path resolves Gemma through `TierPaths` (spec path, else the sibling `gemma/`); the dense
    // path through `gemma_dir_for` (spec path, else `<root>/text_encoder`). Follow whichever applies.
    if let Some(paths) = tier::TierPaths::detect(&root, gemma_override.as_deref()) {
        let tier_file = |name: &str| gen_core::safetensors_path_bytes(paths.tier_dir.join(name));
        return Ok(PerComponentBytes {
            text_encoder: gen_core::safetensors_path_bytes(&paths.gemma_dir),
            dit: tier_file("transformer.safetensors")
                + tier_file("connector.safetensors")
                + spec_upsampler_file(spec, &paths.tier_dir)
                    .map(gen_core::safetensors_path_bytes)
                    .unwrap_or(0),
            vae: tier_file("vae_decoder.safetensors") + tier_file("vae_encoder.safetensors"),
        });
    }
    Ok(PerComponentBytes {
        text_encoder: gemma_dir_for(&root, gemma_override.as_deref())
            .map(gen_core::safetensors_path_bytes)
            .unwrap_or(0),
        // The one dense checkpoint bundles DiT + VAE + audio-VAE + vocoder + projection.
        dit: ltx_checkpoint_in(&root)
            .map(gen_core::safetensors_path_bytes)
            .unwrap_or(0)
            + spec_upsampler_file(spec, &root)
                .map(gen_core::safetensors_path_bytes)
                .unwrap_or(0),
        vae: 0,
    })
}

/// Construct a lazy candle LTX-2.3 generator. `spec.weights` is an LTX-2.3 snapshot dir (the
/// `ltx-2.3-22b-distilled.safetensors` checkpoint); the Gemma encoder is provisioned by the caller via
/// the `LoadSpec::text_encoder` slot (or co-located at `<root>/text_encoder`) — no env / HF-cache scan
/// (sc-13749). LoRA and stamped/third-party LoKr adapters apply to the AudioVideo projection surface;
/// on-the-fly quantization remains unsupported. Request-side image/keyframe/clip conditioning uses the VAE
/// encoder bundled in the dense checkpoint or the packed tier's `vae_encoder.safetensors`.
pub fn load(spec: &LoadSpec) -> gen_core::Result<Box<dyn Generator>> {
    let root = match &spec.weights {
        WeightsSource::Dir(p) => p.clone(),
        WeightsSource::File(p) => p
            .parent()
            .map(|d| d.to_path_buf())
            .unwrap_or_else(|| p.clone()),
    };
    // Named-component contract (sc-13658/sc-13749): `spatial_upscaler` is the sole optional LTX
    // `LoadSpec::components` key; its Gemma TE rides the typed `text_encoder` slot, and there is no
    // uncensored/amoral enhancer variant (the mlx-only `uncensored_enhancer`). Unknown component keys
    // are rejected up front as `Unsupported`.
    gen_core::reject_unknown_components(
        spec,
        &[gen_core::LTX_SPATIAL_UPSCALER_COMPONENT],
        MODEL_ID,
    )?;
    if spec.quantize.is_some() {
        return Err(gen_core::Error::Unsupported(
            "candle ltx does not support on-the-fly Q4/Q8 quantization yet".into(),
        ));
    }
    if spec.control.is_some() || !spec.extra_controls.is_empty() || spec.ip_adapter.is_some() {
        return Err(gen_core::Error::Unsupported(
            "candle ltx video conditioning is request-side and does not consume ControlNet/IP-Adapter weight slots"
                .into(),
        ));
    }
    // sc-8827/sc-13749: the Gemma encoder location rides the spec (`LoadSpec::text_encoder`); `None`
    // falls back to the co-located `<root>/text_encoder` in `gemma_dir` (no env / HF-cache scan).
    let gemma_override = spec.text_encoder.as_ref().map(|src| match src {
        WeightsSource::Dir(p) | WeightsSource::File(p) => p.clone(),
    });
    let upsampler_override = spec
        .components
        .get(gen_core::LTX_SPATIAL_UPSCALER_COMPONENT)
        .map(|src| match src {
            WeightsSource::Dir(p) | WeightsSource::File(p) => p.clone(),
        });
    let device = candle_gen::default_device()?;
    Ok(Box::new(LtxGenerator {
        descriptor: descriptor(),
        root,
        device,
        gemma_override,
        upsampler_override,
        adapters: spec.adapters.clone(),
        components: Mutex::new(None),
    }))
}

candle_gen::register_generators! {
    pub(crate) const REGISTRATION = descriptor => load;
    footprint = component_footprint
}

/// Add the Candle LTX generator and trainer to an explicit media registry builder.
pub fn register_providers(
    registry: candle_gen::gen_core::ProviderRegistryBuilder,
) -> candle_gen::gen_core::ProviderRegistryBuilder {
    registry
        .register_generator(REGISTRATION)
        .register_trainer(training::TRAINER_REGISTRATION)
}

/// Build the complete explicit Candle LTX provider catalog.
pub fn provider_registry() -> candle_gen::gen_core::Result<candle_gen::gen_core::ProviderRegistry> {
    register_providers(candle_gen::gen_core::ProviderRegistryBuilder::new()).build()
}

/// Resolve the load-bearing VAE geometry for a Candle LTX generator id.
pub fn vae_tiling(provider_id: &str) -> Option<candle_gen::gen_core::tiling::VaeTiling> {
    (provider_id == MODEL_ID).then_some(VAE_TILING)
}

/// Resolve the provider-owned conservative VAE decode working-set peak for an LTX generator id.
pub fn conservative_video_decode_memory_profile(
    provider_id: &str,
    width: u32,
    height: u32,
    frames: u32,
) -> Option<candle_gen::VideoDecodeMemoryProfile> {
    vae_tiling(provider_id)?;
    vae::conservative_video_decode_memory_profile(width, height, frames)
}

#[cfg(test)]
mod explicit_registry_tests {
    #[test]
    fn explicit_catalog_has_stable_surface() {
        let registry = super::provider_registry().unwrap();
        let generators: Vec<String> = registry
            .generators()
            .map(|registration| (registration.descriptor)().id.to_string())
            .collect();
        let trainers: Vec<String> = registry
            .trainers()
            .map(|registration| (registration.descriptor)().id.to_string())
            .collect();

        assert_eq!(generators, ["ltx_2_3_distilled"]);
        assert_eq!(trainers, ["ltx_2_3"]);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn missing_spatial_upscaler_is_actionable() {
        let root = tempfile::tempdir().unwrap();
        let error = canonical_upsampler_file(root.path())
            .unwrap_err()
            .to_string();
        assert!(error.contains("spatial_upscaler"), "got: {error}");
        assert!(error.contains("upsampler.safetensors"), "got: {error}");
    }

    #[test]
    fn registers_and_resolves_as_candle_video() {
        let spec = LoadSpec::new(WeightsSource::Dir("/nonexistent".into()));
        let g = crate::provider_registry()
            .unwrap()
            .load(MODEL_ID, &spec)
            .expect("ltx is registered");
        assert_eq!(g.descriptor().id, MODEL_ID);
        assert_eq!(g.descriptor().family, "ltx");
        assert_eq!(g.descriptor().backend, "candle");
        assert_eq!(g.descriptor().modality, Modality::Video);
    }

    #[test]
    fn descriptor_does_not_claim_staged_residency() {
        let caps = descriptor().capabilities;
        assert!(!caps.unconditionally_engages_staged_residency);
        assert!(!caps.supports_sequential_offload);
        assert_eq!(
            caps.staged_residency_availability(),
            candle_gen::gen_core::StagedResidencyAvailability::Absent
        );
    }

    #[test]
    fn gemma_dir_uses_spec_text_encoder_and_ignores_env() {
        // sc-8827/sc-13749: `LoadSpec::text_encoder` drives the Gemma-encoder location. An existing dir
        // is returned as-is; a nonexistent override errors with the spec-side message. The
        // `$LTX_GEMMA_DIR` env side-channel was DELETED — this also pins that it is no longer consulted.
        let real_tmp = tempfile::tempdir().unwrap();
        let real = real_tmp.path().to_path_buf();
        let pipe = Pipeline::load(
            Path::new("/nonexistent/root"),
            &Device::Cpu,
            Some(real.clone()),
            None,
        );
        assert_eq!(pipe.gemma_dir().unwrap(), real);

        // A nonexistent override errors with the spec-side (not env) message.
        let bad = Pipeline::load(
            Path::new("/nonexistent/root"),
            &Device::Cpu,
            Some(PathBuf::from("/nonexistent/ltx_gemma")),
            None,
        );
        let err = bad.gemma_dir().unwrap_err().to_string();
        assert!(err.contains("LoadSpec text_encoder"), "got: {err}");

        // Negative env guard (sc-13749): even with `$LTX_GEMMA_DIR` pointing at a REAL dir, a spec with
        // no text_encoder and no co-located `<root>/text_encoder` must ERROR — the env is never read.
        // (Tests run single-threaded here, `RUST_TEST_THREADS=1`, so mutating the process env is safe.)
        std::env::set_var("LTX_GEMMA_DIR", &real);
        let no_te = Pipeline::load(Path::new("/nonexistent/root"), &Device::Cpu, None, None);
        let err = no_te.gemma_dir().unwrap_err().to_string();
        assert!(
            err.contains("LoadSpec::text_encoder"),
            "env must be ignored, got: {err}"
        );
        assert!(
            !err.contains("LTX_GEMMA_DIR"),
            "error must not name the removed env var: {err}"
        );
        std::env::remove_var("LTX_GEMMA_DIR");
    }

    /// sc-13749 load gate: with no `LoadSpec::text_encoder` AND no co-located `<root>/text_encoder`, the
    /// Gemma encoder is absent → a load-time actionable error **naming the slot** (not a silent env /
    /// HF-cache fallback). A co-located `<root>/text_encoder` is still honored: it is a passed-in path
    /// (the weights root is `LoadSpec::weights`), the candle sibling of the tier's `gemma/` convention.
    #[test]
    fn gemma_dir_requires_slot_or_colocated() {
        // Absent everywhere → actionable error naming the slot, never the removed env var.
        let none = Pipeline::load(Path::new("/nonexistent/root"), &Device::Cpu, None, None);
        let err = none.gemma_dir().unwrap_err().to_string();
        assert!(err.contains("LoadSpec::text_encoder"), "got: {err}");
        assert!(
            !err.contains("LTX_GEMMA_DIR"),
            "must not name the removed env var: {err}"
        );

        // A co-located `<root>/text_encoder` (a passed-in path via the weights root) is honored.
        let root_tmp = tempfile::tempdir().unwrap();
        let root = root_tmp.path().to_path_buf();
        let te = root.join("text_encoder");
        std::fs::create_dir_all(&te).unwrap();
        let pipe = Pipeline::load(&root, &Device::Cpu, None, None);
        assert_eq!(pipe.gemma_dir().unwrap(), te);
    }

    /// sc-13749 load gate: `spatial_upscaler` is the only LTX named component; Gemma still rides the
    /// typed `text_encoder` slot and the uncensored/amoral enhancer remains mlx-only. Unknown component
    /// keys are rejected at load with a typed `Unsupported` error; a no-component spec still loads when
    /// its canonical `upsampler.safetensors` is co-located (lazy weight resolution).
    #[test]
    fn load_rejects_unknown_component() {
        let bogus = LoadSpec::new(WeightsSource::Dir("/nonexistent".into())).with_component(
            "uncensored_enhancer",
            WeightsSource::Dir("/nope/amoral".into()),
        );
        assert!(matches!(
            crate::load(&bogus).err().expect("err"),
            gen_core::Error::Unsupported(_)
        ));
        let ok = LoadSpec::new(WeightsSource::Dir("/nonexistent".into()));
        assert!(crate::load(&ok).is_ok());
    }

    #[test]
    fn descriptor_and_lazy_load_advertise_lora_and_lokr() {
        assert!(descriptor().capabilities.supports_lora);
        assert!(descriptor().capabilities.supports_lokr);

        let lora = LoadSpec::new(WeightsSource::Dir("/nonexistent".into())).with_adapters(vec![
            AdapterSpec::new(
                PathBuf::from("/nonexistent/adapter.safetensors"),
                1.0,
                AdapterKind::Lora,
            ),
        ]);
        assert!(
            crate::load(&lora).is_ok(),
            "LTX load is lazy and must accept a LoRA spec"
        );

        let lokr = LoadSpec::new(WeightsSource::Dir("/nonexistent".into())).with_adapters(vec![
            AdapterSpec::new(
                PathBuf::from("/nonexistent/adapter.safetensors"),
                1.0,
                AdapterKind::Lokr,
            ),
        ]);
        assert!(
            crate::load(&lokr).is_ok(),
            "LTX load is lazy and must admit a stamped LoKr for component-time validation"
        );
    }

    #[test]
    fn ltx_checkpoint_selects_base_distilled_and_eros_bf16() {
        // Helper: a temp dir seeded with `files`, then `ltx_checkpoint()`'s chosen file name.
        let pick = |files: &[&str]| -> String {
            let dir_tmp = tempfile::tempdir().unwrap();
            let dir = dir_tmp.path().to_path_buf();
            for f in files {
                std::fs::write(dir.join(f), b"x").unwrap();
            }
            let pipe = Pipeline::load(&dir, &Device::Cpu, None, None);
            let got = pipe.ltx_checkpoint().unwrap();
            let name = got.file_name().unwrap().to_str().unwrap().to_owned();
            name
        };
        // Base `Lightricks/LTX-2.3`: the distilled file wins over dev / lora / upscaler.
        assert_eq!(
            pick(&[
                "ltx-2.3-22b-dev.safetensors",
                "ltx-2.3-22b-distilled.safetensors",
                "ltx-2.3-22b-distilled-lora-384.safetensors",
                "ltx-2.3-spatial-upscaler-x2.safetensors",
            ],),
            "ltx-2.3-22b-distilled.safetensors"
        );
        // Eros merge: the dense `_bf16` file wins; the fp8 / mixed variants are skipped.
        assert_eq!(
            pick(&[
                "10Eros_v1_bf16.safetensors",
                "10Eros_v1-fp8mixed_learned.safetensors",
                "10Eros_v1_fp8_transformer.safetensors",
            ],),
            "10Eros_v1_bf16.safetensors"
        );
    }

    #[test]
    fn descriptor_surface() {
        let d = descriptor();
        assert!(!d.capabilities.supports_guidance);
        assert!(!d.capabilities.supports_negative_prompt);
        assert!(!d.capabilities.mac_only);
        assert_eq!(
            d.capabilities.conditioning,
            [
                ConditioningKind::Reference,
                ConditioningKind::Keyframe,
                ConditioningKind::VideoClip,
                ConditioningKind::ControlClip,
            ]
        );
        // sc-7125: curated sampler menu + the legacy `rectified-flow` alias; NO scheduler axis (3b).
        assert!(d.capabilities.samplers.contains(&"rectified-flow"));
        assert!(d.capabilities.samplers.contains(&"euler"));
        assert!(d.capabilities.samplers.contains(&"dpmpp_2m"));
        assert!(d.capabilities.schedulers.is_empty());
    }

    #[test]
    fn validate_admits_i2v_flf_extend_bridge_and_replace_person() {
        let generator = crate::provider_registry()
            .unwrap()
            .load(
                MODEL_ID,
                &LoadSpec::new(WeightsSource::Dir("/nonexistent".into())),
            )
            .unwrap();
        let image = Image {
            width: 32,
            height: 32,
            pixels: vec![127; 32 * 32 * 3],
        };
        let mask = Image {
            width: 32,
            height: 32,
            pixels: vec![255; 32 * 32 * 3],
        };
        let base = GenerationRequest {
            prompt: "a person crosses the room".into(),
            width: 704,
            height: 512,
            frames: Some(49),
            ..Default::default()
        };
        for conditioning in [
            vec![Conditioning::Reference {
                image: image.clone(),
                strength: Some(0.8),
            }],
            vec![
                Conditioning::Keyframe {
                    image: image.clone(),
                    frame_idx: 0,
                    strength: 1.0,
                },
                Conditioning::Keyframe {
                    image: image.clone(),
                    frame_idx: -1,
                    strength: 1.0,
                },
            ],
            vec![
                Conditioning::VideoClip {
                    frames: vec![image.clone()],
                    frame_idx: 0,
                    strength: 1.0,
                },
                Conditioning::VideoClip {
                    frames: vec![image.clone()],
                    frame_idx: -1,
                    strength: 0.75,
                },
            ],
            vec![Conditioning::ControlClip {
                frames: vec![image.clone()],
                mask: vec![mask.clone()],
                masking_strength: 0.9,
                start_frame: 0,
                mode: gen_core::ReplacementMode::FaceOnly,
            }],
        ] {
            assert!(generator
                .validate(&GenerationRequest {
                    conditioning,
                    ..base.clone()
                })
                .is_ok());
        }
    }

    #[test]
    fn validate_rejects_malformed_conditioning_without_loading_weights() {
        let generator = crate::provider_registry()
            .unwrap()
            .load(
                MODEL_ID,
                &LoadSpec::new(WeightsSource::Dir("/nonexistent".into())),
            )
            .unwrap();
        let image = Image {
            width: 32,
            height: 32,
            pixels: vec![127; 32 * 32 * 3],
        };
        let control = Conditioning::ControlClip {
            frames: vec![image.clone()],
            mask: vec![image.clone()],
            masking_strength: 1.0,
            start_frame: 0,
            mode: gen_core::ReplacementMode::FaceOnly,
        };
        let base = GenerationRequest {
            prompt: "x".into(),
            width: 704,
            height: 512,
            frames: Some(49),
            ..Default::default()
        };
        let cases = [
            GenerationRequest {
                conditioning: vec![Conditioning::Keyframe {
                    image: image.clone(),
                    frame_idx: 99,
                    strength: 1.0,
                }],
                ..base.clone()
            },
            GenerationRequest {
                conditioning: vec![Conditioning::VideoClip {
                    frames: vec![image.clone(), image.clone()],
                    frame_idx: 0,
                    strength: 1.0,
                }],
                ..base.clone()
            },
            GenerationRequest {
                conditioning: vec![control.clone(), control],
                ..base.clone()
            },
            GenerationRequest {
                sampler: Some("heun".into()),
                conditioning: vec![Conditioning::Reference {
                    image,
                    strength: Some(1.0),
                }],
                ..base
            },
        ];
        for request in cases {
            assert!(
                generator.validate(&request).is_err(),
                "must reject {request:?}"
            );
        }
    }

    #[test]
    fn vae_encoder_is_not_retained_for_unconditioned_t2v() {
        let base = GenerationRequest::default();
        assert!(!needs_ltx_vae_encoder(&base));
        assert!(needs_ltx_vae_encoder(&GenerationRequest {
            conditioning: vec![Conditioning::Reference {
                image: Image {
                    width: 1,
                    height: 1,
                    pixels: vec![0; 3],
                },
                strength: Some(0.5),
            }],
            ..base
        }));
    }

    #[test]
    fn validate_accepts_txt2video_and_rejects_unsupported() {
        let spec = LoadSpec::new(WeightsSource::Dir("/nonexistent".into()));
        let g = crate::provider_registry()
            .unwrap()
            .load(MODEL_ID, &spec)
            .unwrap();
        let ok = GenerationRequest {
            prompt: "a cat walking across a sunny garden".into(),
            width: 704,
            height: 512,
            frames: Some(49),
            sampler: Some("rectified-flow".into()),
            ..Default::default()
        };
        assert!(g.validate(&ok).is_ok());
        for bad in [
            GenerationRequest::default(),
            GenerationRequest {
                prompt: "x".into(),
                frames: Some(48), // not ≡ 1 (mod 8)
                ..Default::default()
            },
            GenerationRequest {
                prompt: "x".into(),
                width: 700, // not a multiple of 32
                ..Default::default()
            },
        ] {
            assert!(g.validate(&bad).is_err(), "should reject: {bad:?}");
        }
        // sc-12587: `SIZE_MULTIPLE` is the pinned stride SceneWorks ties `requiresDimensionsMultipleOf`
        // to — candle's distilled ltx refines on the 64× final-output grid. Pin the value and prove
        // a multiple of 16 that is not a multiple of SIZE_MULTIPLE is rejected with the stride error.
        assert_eq!(SIZE_MULTIPLE, (config::SPATIAL_SCALE * 2) as u32);
        assert_eq!(SIZE_MULTIPLE, 64);
        let off_stride = g
            .validate(&GenerationRequest {
                width: 672, // 21×32 — above the minimum but not SIZE_MULTIPLE
                ..ok.clone()
            })
            .unwrap_err()
            .to_string();
        assert!(
            off_stride.contains("multiples of 64"),
            "expected the stride error, got: {off_stride}"
        );
    }

    /// sc-9027 / F-043: the distilled schedule is fixed, so `render` runs exactly `NATIVE_STEPS`
    /// (`STAGE1_SIGMAS.len() − 1`) denoise steps and never resamples for an arbitrary `req.steps`.
    #[test]
    fn native_steps_matches_baked_schedule() {
        assert_eq!(NATIVE_STEPS as usize, STAGE1_SIGMAS.len() - 1);
        assert_eq!(NATIVE_STEPS, 8);
    }

    /// `req.steps` is no longer silently ignored: `None` (distilled default) and an explicit
    /// `Some(NATIVE_STEPS)` are accepted; any other override is rejected with a diagnostic rather than
    /// quietly running the baked 8-step schedule.
    #[test]
    fn validate_honors_or_rejects_req_steps() {
        let spec = LoadSpec::new(WeightsSource::Dir("/nonexistent".into()));
        let g = crate::provider_registry()
            .unwrap()
            .load(MODEL_ID, &spec)
            .unwrap();
        let base = GenerationRequest {
            prompt: "a cat walking across a sunny garden".into(),
            width: 704,
            height: 512,
            frames: Some(49),
            ..Default::default()
        };
        // Default (None) → distilled schedule.
        assert!(g.validate(&base).is_ok());
        // Explicit native step count is honored.
        assert!(g
            .validate(&GenerationRequest {
                steps: Some(NATIVE_STEPS),
                ..base.clone()
            })
            .is_ok());
        // A non-native override (the F-043 `steps: 30` case) is rejected, not silently ignored.
        for s in [1u32, 4, 7, 9, 30, 50] {
            assert!(
                g.validate(&GenerationRequest {
                    steps: Some(s),
                    ..base.clone()
                })
                .is_err(),
                "steps={s} must be rejected"
            );
        }
    }

    /// F-131 / sc-11234: `validate` bounds the video latent token count (`t_lat · h_lat · w_lat`),
    /// so a huge frame count that passes the `% 8 == 1` shape check but would OOM the 22B AvDiT
    /// denoise loop is rejected catchably up front instead of blowing up mid-render. An in-bounds
    /// long clip still passes.
    #[test]
    fn validate_rejects_unbounded_frame_count() {
        let spec = LoadSpec::new(WeightsSource::Dir("/nonexistent".into()));
        let g = crate::provider_registry()
            .unwrap()
            .load(MODEL_ID, &spec)
            .unwrap();
        let base = GenerationRequest {
            prompt: "a cat walking across a sunny garden".into(),
            width: 1280,
            height: 1280,
            ..Default::default()
        };
        // The finding's pathological case: 2001 frames satisfies `% 8 == 1` (shape-valid) but is
        // ~400k latent tokens at 1280² — far over the cap.
        assert_eq!(
            2001 % config::TEMPORAL_SCALE as u32,
            1,
            "shape-valid frame count"
        );
        let huge = GenerationRequest {
            frames: Some(2001),
            ..base.clone()
        };
        let err = g.validate(&huge).unwrap_err().to_string();
        assert!(
            err.contains("latent tokens") && err.contains("cap"),
            "over-cap request rejected with a clear message: {err}"
        );

        // The token count is the actual driver: computing it here mirrors `validate`.
        let (t, h, w) = pipeline::latent_dims(2001, 1280, 1280);
        assert!(
            t * h * w > config::max_latent_tokens(),
            "2001@1280² exceeds the cap"
        );

        // A generous but in-bounds clip still validates: 129 frames at 704×480 → t_lat 17 ·
        // (22·15) = 5610 latent tokens, comfortably under the 131072 cap.
        let ok = GenerationRequest {
            frames: Some(129),
            width: 704,
            height: 512,
            ..base
        };
        assert!(
            g.validate(&ok).is_ok(),
            "an in-bounds long clip must pass: {ok:?}"
        );
    }

    /// sc-12397 — the DENSE layout: the footprint must size the ONE checkpoint `ltx_checkpoint_in`
    /// picks, plus the Gemma encoder. NOT the directory.
    ///
    /// This is why LTX owns its own footprint. The hosted `Lightricks/LTX-2.3` is ~146 GiB on disk
    /// (`estimatedSizeBytes: 157004895813`) against a SINGLE-file load, because the snapshot also ships
    /// fp8/mixed/lora/upscaler siblings. A consumer summing the dir would over-predict by ~7x and refuse
    /// LTX on every GPU in existence — a wall-reject, the worst failure a fit gate has.
    ///
    /// Kills the mutation: swapping `ltx_checkpoint_in` for `safetensors_dir_bytes(root)` makes `dit`
    /// read 12_400 instead of 9_000.
    #[test]
    fn component_footprint_dense_sizes_one_checkpoint_plus_gemma() {
        let root_tmp = tempfile::tempdir().unwrap();
        let root = root_tmp.path().to_path_buf();
        for (name, len) in [
            ("ltx-2.3-22b-distilled.safetensors", 9_000_u64), // the one that loads
            ("ltx-2.3-22b-fp8.safetensors", 2_000),           // skipped: fp8
            ("ltx-2.3-22b-mixed.safetensors", 1_000),         // skipped: mixed
            ("some-upscaler.safetensors", 300),               // skipped: upscaler
            ("a-lora.safetensors", 100),                      // skipped: lora
        ] {
            std::fs::File::create(root.join(name))
                .unwrap()
                .set_len(len)
                .unwrap();
        }
        // The Gemma encoder is a SEPARATE snapshot threaded via `LoadSpec::text_encoder` — omitting it
        // would under-count by more than the DiT on the real model (~24 GB).
        let gemma = root.join("gemma-snapshot");
        std::fs::create_dir_all(&gemma).unwrap();
        std::fs::File::create(gemma.join("model.safetensors"))
            .unwrap()
            .set_len(4_000)
            .unwrap();

        let mut spec = LoadSpec::new(WeightsSource::Dir(root.clone()));
        spec.text_encoder = Some(WeightsSource::Dir(gemma.clone()));
        let fp = component_footprint(&spec).expect("footprint");

        assert_eq!(fp.dit, 9_000, "the distilled checkpoint alone, not the dir");
        assert_eq!(fp.text_encoder, 4_000, "the Gemma snapshot must be counted");
        assert_eq!(
            fp.vae, 0,
            "the dense checkpoint bundles the VAE — counting it again would double-count"
        );
        // The slots partition the load: 13_000, not the 12_400-in-root dir sum + gemma.
        assert_eq!(fp.text_encoder + fp.dit + fp.vae, 13_000);
    }

    /// sc-12397 — the PACKED TIER layout: the five files the video render loads, plus sibling Gemma.
    /// The VAE encoder is part of the truthful footprint because advertised conditioning consumes it.
    #[test]
    fn component_footprint_tier_sizes_conditioning_encoder_plus_gemma() {
        let snapshot_tmp = tempfile::tempdir().unwrap();
        let snapshot = snapshot_tmp.path().to_path_buf();
        let tier = snapshot.join("q4");
        std::fs::create_dir_all(&tier).unwrap();
        // `TierPaths::detect` needs BOTH markers: transformer.safetensors + quantize_config.json.
        std::fs::write(tier.join("quantize_config.json"), "{}").unwrap();
        for (name, len) in [
            ("transformer.safetensors", 5_000_u64), // loaded
            ("connector.safetensors", 700),         // loaded
            ("vae_decoder.safetensors", 300),       // loaded
            ("vae_encoder.safetensors", 9_000),     // loaded for video conditioning
            ("audio_vae.safetensors", 8_000),       // NOT loaded
            ("vocoder.safetensors", 7_000),         // NOT loaded
            ("upsampler.safetensors", 6_000),       // loaded for stage two
        ] {
            std::fs::File::create(tier.join(name))
                .unwrap()
                .set_len(len)
                .unwrap();
        }
        // The tier's Gemma is its SIBLING (`<snapshot>/gemma`), not an override.
        let gemma = snapshot.join("gemma");
        std::fs::create_dir_all(&gemma).unwrap();
        std::fs::File::create(gemma.join("model.safetensors"))
            .unwrap()
            .set_len(4_000)
            .unwrap();

        let spec = LoadSpec::new(WeightsSource::Dir(tier.clone()));
        let fp = component_footprint(&spec).expect("footprint");

        assert_eq!(fp.dit, 11_700, "transformer + connector + upsampler");
        assert_eq!(fp.vae, 9_300, "decoder + conditioning encoder");
        assert_eq!(fp.text_encoder, 4_000, "the sibling gemma/ dir");
        assert_eq!(fp.text_encoder + fp.dit + fp.vae, 25_000);
    }

    /// An unresolvable snapshot reports NO SIGNAL rather than erroring: the footprint is a pre-load
    /// ADMISSION signal, so "no signal" (⇒ the caller admits) beats refusing a job over an unreadable
    /// path. `load_components` surfaces the real error moments later.
    ///
    /// sc-13749: all three slots are pinned to 0. `text_encoder` can now be asserted deterministically —
    /// `gemma_dir_for` no longer consults any environment side-channel (deleted), so with no override and
    /// no `<root>/text_encoder` it resolves to nothing regardless of the runner's environment.
    #[test]
    fn component_footprint_reports_no_signal_rather_than_failing() {
        let spec = LoadSpec::new(WeightsSource::Dir("/nonexistent-ltx-snapshot".into()));
        let fp = component_footprint(&spec).expect("a missing snapshot is not a footprint error");
        assert_eq!(
            (fp.text_encoder, fp.dit, fp.vae),
            (0, 0, 0),
            "an unreadable snapshot must read as no signal, not an error"
        );
    }
}
