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
//!
//! **LTX-2.5 (sc-18767):** the `CausalDiffusionVAE` video decoder — five stages of 3-D
//! neighborhood attention + SwiGLU feeding an eight-block single-step diffusion stage — is
//! [`diff_vae`], the candle twin of `mlx_gen_ltx::diff_vae`. It reads the released
//! `vae/ltx-2.5-video-vae-bf16.safetensors` **verbatim** (no conversion: every tensor it wants is
//! already a PyTorch `[out, in]` matrix), implements the neighborhood-attention operator itself
//! rather than taking a NATTEN/CUTLASS dependency, and is asserted against the *same* committed
//! goldens the MLX port uses. The conv decoder ([`vae`]) is untouched and still the selectable
//! decode path for a conv-VAE checkpoint.

pub mod adapters;
pub mod audio_vae;
pub mod block_stream;
pub mod bundle;
pub mod conditioning;
pub mod config;
pub mod connector;
pub mod conv3d;
pub mod dev_sampler;
pub mod dfr;
pub mod diff_vae;
pub mod dit_train;
pub mod duration_head;
pub mod gemma;
pub mod gemma4_te;
pub mod image_crf;
pub mod memory_strategy;
pub mod memory_strategy_2_5;
pub mod params;
pub mod pipeline;
pub mod quant;
pub mod quant_eval;
pub mod rope;
pub mod text_encoder;
pub mod tier;
pub mod tokenizer;
pub mod training;
pub mod transformer;
pub mod upsampler;
pub mod vae;
pub mod vocoder;

use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex};

use candle_gen::candle_core::{DType, Device, Tensor};
#[cfg(test)]
use candle_gen::gen_core::ltx_checkpoint::LtxBundleBuilder;
use candle_gen::gen_core::ltx_checkpoint::{LtxBundle, LtxCheckpointLayout, LtxComponent};
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
use dev_sampler::{ExecutionPlan, TransformerVariant};
use diff_vae::{NaDiffusionDecoder, NaDiffusionDecoderConfig};
use duration_head::DurationHead;
use gemma4_te::Ltx25TextEncoder;
use quant_eval::{Ltx25GpuGeneration, Ltx25QuantAdmission, Ltx25QuantMode};
use text_encoder::LtxTextEncoder;
use tokenizer::{ensure_single_leading_bos_u32, gemma_bos_id, Ltx25Tokenizer};
use transformer::AvDiT;
use upsampler::LatentUpsampler;
use vae::LtxVideoVae;
/// Provider-facing LTX geometry, derived from the decoder implementation.
pub const VAE_TILING: candle_gen::gen_core::tiling::VaeTiling = LtxVideoVae::VAE_TILING;
use vocoder::LtxVocoder;

const DIT_DTYPE: DType = DType::BF16;
const VAE_DTYPE: DType = DType::F32;

/// Public provider id for the split-component Gemma-4 LTX-2.5 distilled route.
pub const MODEL_25_ID: &str = "ltx_2_5_distilled";

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
    te: Arc<TextEncoder>,
    avdit: Arc<AvDiT>,
    vae: Arc<LtxVideoVae>,
    /// The selected alternate decoder.  Its latent input is exactly the conv VAE's DiT-normalized
    /// space, so conditioning and the learned upsamplers remain shared while final decode changes.
    diffusion_decoder: Option<Arc<NaDiffusionDecoder>>,
    duration_head: Option<Arc<DurationHead>>,
    upsampler: Arc<LatentUpsampler>,
    temporal_upsampler: Option<Arc<LatentUpsampler>>,
    vae_has_encoder: bool,
    /// Audio decode chain — `None` on the packed MLX tier path (sc-9545), which is **video-only**: the
    /// tier's audio-VAE + vocoder ship in a different key layout (channels-last convs, no `decoder.`/
    /// `vocoder.` prefix) that is a separate ingestion slice (follow-up), and the sc-9417 render AC is a
    /// video render. The audio latent stream still runs through the joint AvDiT (cross-modal coupling
    /// keeps the video coherent); only the audio VAE→vocoder decode is skipped.
    audio: Option<AudioChain>,
    tokenizer: Arc<PromptTokenizer>,
}

enum TextEncoder {
    Gemma3(Box<LtxTextEncoder>),
    Gemma4(Box<Ltx25TextEncoder>),
}

impl TextEncoder {
    fn encode_both(&self, input_ids: &Tensor, mask01: &[u32]) -> CResult<(Tensor, Tensor)> {
        match self {
            Self::Gemma3(te) => Ok(te.encode_both(input_ids, mask01)?),
            Self::Gemma4(te) => Ok(te.encode_both(input_ids, mask01)?),
        }
    }
}

enum PromptTokenizer {
    Gemma3(tokenizers::Tokenizer),
    Gemma4(Ltx25Tokenizer),
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
    /// `Some` makes this the split LTX-2.5 route.  All config is read before weights are materialised.
    split_bundle: Option<LtxBundle>,
    /// Exact transformer identity read from the split safetensors metadata.  Dense LTX-2.3 is
    /// always the historical distilled route; split LTX-2.5 must provide this explicitly.
    transformer_variant: TransformerVariant,
    use_diffusion_decoder: bool,
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
            split_bundle: None,
            transformer_variant: TransformerVariant::Distilled,
            use_diffusion_decoder: false,
        }
    }

    fn load_split(
        bundle: LtxBundle,
        device: &Device,
        use_diffusion_decoder: bool,
        quant_mode: Ltx25QuantMode,
        transformer_variant: TransformerVariant,
    ) -> gen_core::Result<Self> {
        // Re-bind the selected precision when materialization begins. This is deliberately not just
        // a load-time descriptor check: replacing a staged component between construction and the
        // lazy request load must refuse rather than render its bf16/q4 numerics under another label.
        quant_mode.validate_bundle_source(&bundle)?;
        let av_cfg = AvConfig::from_bundle(&bundle)?;
        let conn_cfg = ConnectorConfig::from_bundle(&bundle)?;
        let audio_conn_cfg = ConnectorConfig::audio_from_bundle(&bundle)?;
        let audio_vae_cfg = AudioVaeConfig::from_bundle(&bundle)?;
        let vocoder_cfg = VocoderConfig::from_bundle(&bundle)?;
        // Make the selected decoder's declaration a load-path fact, rather than a descriptor claim.
        let selected = if use_diffusion_decoder {
            LtxComponent::DiffusionVideoVae
        } else {
            LtxComponent::ConvVideoVae
        };
        let declaration = config::VideoVaeDeclaration::from_bundle(&bundle, selected)?;
        if declaration.is_diffusion() != use_diffusion_decoder {
            return Err(gen_core::Error::Msg(
                "ltx_2_5: selected video decoder declaration disagrees with its component".into(),
            ));
        }
        Ok(Self {
            av_cfg,
            gemma_cfg: GemmaConfig::gemma_3_12b(),
            conn_cfg,
            audio_conn_cfg,
            audio_vae_cfg,
            vocoder_cfg,
            root: bundle
                .require(LtxComponent::Transformer)?
                .path()
                .parent()
                .unwrap_or(Path::new("."))
                .to_path_buf(),
            device: device.clone(),
            gemma_override: None,
            upsampler_override: None,
            split_bundle: Some(bundle),
            transformer_variant,
            use_diffusion_decoder,
        })
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
        if self.split_bundle.is_some() {
            return self.load_components_split(adapters, with_vae_encoder);
        }
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
        let upsampler_file = self.upsampler_file()?;
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
        // Loaded through the path constructor, so a stamped checkpoint's declared config is
        // cross-checked against the structure the weights imply instead of the rank silently
        // winning.
        let upsampler = LatentUpsampler::from_checkpoint(&upsampler_file, VAE_DTYPE, &self.device)?;
        // The audio VAE decoder + vocoder run f32 (post-sampling quality islands).
        let audio_decoder = AudioDecoder::load(&vb_f32.pp("audio_vae"), &self.audio_vae_cfg)?;
        let vocoder = LtxVocoder::load(vb_f32, &self.device, &self.vocoder_cfg)?;
        let audio_sample_rate = self.vocoder_cfg.final_sample_rate() as u32;

        let tok_path = gemma_dir.join("tokenizer.json");
        let tokenizer = tokenizers::Tokenizer::from_file(&tok_path)
            .map_err(|e| CandleError::Msg(format!("ltx: load gemma tokenizer: {e}")))?;

        Ok(Components {
            te: Arc::new(TextEncoder::Gemma3(Box::new(te))),
            avdit: Arc::new(avdit),
            vae: Arc::new(vae),
            diffusion_decoder: None,
            duration_head: None,
            upsampler: Arc::new(upsampler),
            temporal_upsampler: None,
            vae_has_encoder: with_vae_encoder,
            audio: Some(AudioChain {
                decoder: Arc::new(audio_decoder),
                vocoder: Arc::new(vocoder),
                sample_rate: audio_sample_rate,
            }),
            tokenizer: Arc::new(PromptTokenizer::Gemma3(tokenizer)),
        })
    }

    /// Materialise the actual split LTX-2.5 component files.  This deliberately shares the 2.3
    /// renderer rather than creating a provider-local sampler: both decoder choices consume the
    /// same DiT-normalized latent and use the same learned spatial upsampler.
    fn load_components_split(
        &self,
        adapters: &[AdapterSpec],
        with_vae_encoder: bool,
    ) -> CResult<Components> {
        let bundle = self.split_bundle.as_ref().expect("split route has bundle");
        let transformer = bundle
            .require(LtxComponent::Transformer)
            .map_err(|e| CandleError::Msg(e.to_string()))?
            .path();
        let conv = bundle
            .require(LtxComponent::ConvVideoVae)
            .map_err(|e| CandleError::Msg(e.to_string()))?
            .path();
        let audio = bundle
            .require(LtxComponent::AudioVae)
            .map_err(|e| CandleError::Msg(e.to_string()))?
            .path();
        let dit_all =
            candle_gen::mmap_var_builder(&[transformer.to_path_buf()], DIT_DTYPE, &self.device)?;
        let dit_vb = dit_all.pp("model.diffusion_model");
        let mut avdit = AvDiT::new(dit_vb.clone(), &self.av_cfg)?;
        adapters::install_ltx25_adapters(&mut avdit, adapters)?;
        let te = Ltx25TextEncoder::from_bundle_av(
            bundle,
            dit_all.clone(),
            dit_vb,
            &self.av_cfg,
            &self.conn_cfg,
            &self.audio_conn_cfg,
        )?;
        let conv_vb = candle_gen::mmap_var_builder(&[conv.to_path_buf()], VAE_DTYPE, &self.device)?;
        let vae = if with_vae_encoder {
            LtxVideoVae::new_with_encoder(
                conv_vb.pp("vae"),
                conv_vb.pp("vae"),
                config::LATENT_CHANNELS,
                4,
            )?
        } else {
            LtxVideoVae::new(conv_vb.pp("vae"), config::LATENT_CHANNELS, 4)?
        };
        let upsampler = LatentUpsampler::from_checkpoint(
            bundle
                .require(LtxComponent::SpatialUpsampler)
                .map_err(|e| CandleError::Msg(e.to_string()))?
                .path(),
            VAE_DTYPE,
            &self.device,
        )?;
        let audio_vb =
            candle_gen::mmap_var_builder(&[audio.to_path_buf()], VAE_DTYPE, &self.device)?;
        let audio_decoder = AudioDecoder::load(&audio_vb.pp("audio_vae"), &self.audio_vae_cfg)?;
        let vocoder = LtxVocoder::load(audio_vb, &self.device, &self.vocoder_cfg)?;
        let diffusion_decoder = if self.use_diffusion_decoder {
            let component = bundle
                .require(LtxComponent::DiffusionVideoVae)
                .map_err(|e| CandleError::Msg(e.to_string()))?;
            let cfg = NaDiffusionDecoderConfig::from_embedded_vae(
                component
                    .config()
                    .map_err(|e| CandleError::Msg(e.to_string()))?,
            )?;
            let diff_vb = candle_gen::mmap_var_builder(
                &[component.path().to_path_buf()],
                VAE_DTYPE,
                &self.device,
            )?;
            Some(Arc::new(NaDiffusionDecoder::load(
                diff_vb.pp("decoder"),
                diff_vb,
                &cfg,
            )?))
        } else {
            None
        };
        let duration_weights = candle_gen::Weights::from_file(
            bundle
                .require(LtxComponent::DurationHead)
                .map_err(|e| CandleError::Msg(e.to_string()))?
                .path(),
            &self.device,
            DType::F32,
        )?;
        let duration_head = Some(Arc::new(DurationHead::from_weights(
            &duration_weights,
            &self.device,
        )?));
        let temporal_upsampler = Some(Arc::new(LatentUpsampler::from_checkpoint(
            bundle
                .require(LtxComponent::TemporalUpsampler)
                .map_err(|e| CandleError::Msg(e.to_string()))?
                .path(),
            VAE_DTYPE,
            &self.device,
        )?));
        let tokenizer = Ltx25Tokenizer::from_packed_te_file(
            bundle
                .require(LtxComponent::TextEncoder)
                .map_err(|e| CandleError::Msg(e.to_string()))?
                .path(),
        )?;
        Ok(Components {
            te: Arc::new(TextEncoder::Gemma4(Box::new(te))),
            avdit: Arc::new(avdit),
            vae: Arc::new(vae),
            diffusion_decoder,
            duration_head,
            upsampler: Arc::new(upsampler),
            temporal_upsampler,
            vae_has_encoder: with_vae_encoder,
            audio: Some(AudioChain {
                decoder: Arc::new(audio_decoder),
                vocoder: Arc::new(vocoder),
                sample_rate: self.vocoder_cfg.final_sample_rate() as u32,
            }),
            tokenizer: Arc::new(PromptTokenizer::Gemma4(tokenizer)),
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
        let upsampler_file = if self.upsampler_override.is_some() {
            self.upsampler_file()?
        } else {
            paths.upsampler_file()?
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
        // Path constructor, same reason as the unified route above.
        let upsampler = LatentUpsampler::from_checkpoint(&upsampler_file, VAE_DTYPE, &self.device)?;

        let tok_path = paths.tokenizer_path();
        let tokenizer = tokenizers::Tokenizer::from_file(&tok_path)
            .map_err(|e| CandleError::Msg(format!("ltx tier: load gemma tokenizer: {e}")))?;

        Ok(Components {
            te: Arc::new(TextEncoder::Gemma3(Box::new(te))),
            avdit: Arc::new(avdit),
            vae: Arc::new(vae),
            diffusion_decoder: None,
            duration_head: None,
            upsampler: Arc::new(upsampler),
            temporal_upsampler: None,
            vae_has_encoder: with_vae_encoder,
            audio: None,
            tokenizer: Arc::new(PromptTokenizer::Gemma3(tokenizer)),
        })
    }

    /// Tokenize `prompt` with the Gemma tokenizer (exactly one leading BOS, right-truncate then
    /// **left-pad** to `TEXT_MAX_LENGTH`), returning `(input_ids [1, 256] u32, mask01 [256])`.
    ///
    /// Gemma-3's `tokenizer.json` post-processor already supplies the `<bos>`, so the
    /// [`ensure_single_leading_bos_u32`] call is normally a no-op — it is the explicit guard against
    /// the two ways this goes wrong (no BOS from a post-processor-less tokenizer, a duplicate one
    /// from an unconditional prepend), and it is the same policy the 2.5 path runs (sc-18762).
    fn tokenize(&self, tok: &PromptTokenizer, prompt: &str) -> CResult<(Tensor, Vec<u32>)> {
        if let PromptTokenizer::Gemma4(tok) = tok {
            return tok.encode(prompt, TEXT_MAX_LENGTH, &self.device);
        }
        let PromptTokenizer::Gemma3(tok) = tok else {
            unreachable!("Gemma tokenizer variants covered");
        };
        let bos_id = gemma_bos_id(tok)?;
        let enc = tok
            .encode(prompt, true)
            .map_err(|e| CandleError::Msg(format!("ltx: tokenize: {e}")))?;
        let mut ids: Vec<u32> = enc.get_ids().to_vec();
        let max = TEXT_MAX_LENGTH;
        if ids.len() > max {
            ids.truncate(max);
        }
        ensure_single_leading_bos_u32(&mut ids, bos_id, max);
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

    /// Re-compresses the image first at the checkpoint's resolved `default_image_crf` (sc-18759
    /// — [`config::CHECKPOINT_MODEL_VERSION`] resolves to [`params::LTX_2_3_PARAMS`]'s `crf: 33`)
    /// before the existing normalize/layout in `conditioning::preprocess_conditioning_image`.
    fn encode_image(
        &self,
        vae: &LtxVideoVae,
        image: &Image,
        width: u32,
        height: u32,
    ) -> CResult<Tensor> {
        let video = image_crf::condition_image_for_checkpoint(
            image,
            width,
            height,
            config::CHECKPOINT_MODEL_VERSION,
            None,
            &self.device,
            &mut image_crf::default_image_recompress,
        )?;
        Ok(vae.encode(&video)?)
    }

    /// Resolve and VAE-encode replace-latent inputs: a `Reference` is I2V at frame zero; the
    /// replace_person `MultiReference` carrier is an ordered 1–4 contact sheet at frame zero; and
    /// explicit keyframes cover FLF and arbitrary latent-frame placement.
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
                Conditioning::MultiReference { images } => {
                    // SC-20776: the LTX IC-LoRA has one image-latent identity carrier. Compose
                    // all ordered references before encoding so the public 1–4 surface does not
                    // collapse to the first character on Candle.
                    let composite = conditioning::compose_ordered_character_references(
                        images, req.width, req.height,
                    )
                    .map_err(|error| CandleError::Msg(error.to_string()))?;
                    out.push(EncodedKeyframe {
                        latent: self.encode_image(vae, &composite, width, height)?,
                        frame_idx: 0,
                        strength: 1.0,
                    });
                }
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

    /// The split LTX-2.5 DFR execution branch.  It deliberately consumes the ordinary provider's
    /// already-materialised Gemma contexts, VAE conditioning latents, DiT, and learned upsamplers;
    /// there is no second loader or helper-only sampler path.
    #[allow(clippy::too_many_arguments)]
    fn render_dfr(
        &self,
        req: &GenerationRequest,
        comps: &Components,
        frames: u32,
        fps: u32,
        seed: u64,
        video_ctx: &Tensor,
        audio_ctx: &Tensor,
        negative_context: Option<(&Tensor, &Tensor)>,
        on_progress: &mut dyn FnMut(Progress),
    ) -> CResult<(Vec<Image>, u32, Option<AudioTrack>)> {
        let (canvas_frames, _, mut keyframe_positions) =
            candle_gen::gen_core::ltx_dfr::resolve_canvas(
                frames as i64,
                config::TEMPORAL_SCALE as i64,
            )
            .map_err(|e| CandleError::Msg(e.to_string()))?;
        if let Some(count) = req.num_generated_keyframes.filter(|count| *count > 0) {
            keyframe_positions.extend(
                candle_gen::gen_core::ltx_dfr::evenly_spaced_keyframe_positions(
                    count,
                    canvas_frames,
                ),
            );
            keyframe_positions.sort_unstable();
            keyframe_positions.dedup();
        }
        let geometry = pipeline::two_stage_geometry(canvas_frames as u32, req.width, req.height);
        let audio_frames = compute_audio_frames(canvas_frames as usize, fps as f64).max(1);
        let audio_grid = rope::create_audio_position_grid(audio_frames, &self.device)?;
        let stage1 =
            self.build_keyframes(req, &comps.vae, geometry.t, req.width / 2, req.height / 2)?;
        let stage2 = self.build_keyframes(req, &comps.vae, geometry.t, req.width, req.height)?;
        if stage1.len() != stage2.len()
            || stage1
                .iter()
                .zip(&stage2)
                .any(|(left, right)| left.frame_idx != right.frame_idx)
        {
            return Err(CandleError::Msg(
                "ltx_2_5_distilled: DFR keyframe encodes disagree between stages".into(),
            ));
        }
        let image_keyframes: Vec<dfr::DfrStageKeyframe> = stage1
            .into_iter()
            .zip(stage2)
            .map(|(low, full)| dfr::DfrStageKeyframe {
                stage1: low.latent,
                stage2: full.latent,
                frame_idx: low.frame_idx,
                strength: low.strength,
            })
            .collect();
        let temporal = comps.temporal_upsampler.as_deref().ok_or_else(|| {
            CandleError::Msg("ltx_2_5_distilled: DFR requires the temporal latent upsampler".into())
        })?;
        let parts = dfr::DfrComponents {
            dit: &comps.avdit,
            vae: &comps.vae,
            spatial_upsampler: &comps.upsampler,
            temporal_upsampler: Some(temporal),
            video_ctx,
            audio_ctx,
            negative_video_ctx: negative_context.map(|(video, _)| video),
            negative_audio_ctx: negative_context.map(|(_, audio)| audio),
            transformer_variant: self.transformer_variant,
            audio_grid: &audio_grid,
            audio_frames,
        };
        let dfr_req = dfr::DfrRequest {
            canvas_frames,
            requested_frames: frames as i64,
            keyframe_positions: &keyframe_positions,
            geometry,
            fps: fps as f32,
            seed,
            temporal_upsample_rounds: req.temporal_upsample_rounds.unwrap_or(0),
            detailing_downscale: None,
            video_keyframes: &image_keyframes,
        };
        let output = dfr::generate_dfr_av_latents(&parts, &dfr_req, &req.cancel, on_progress)?;
        on_progress(Progress::Decoding);
        let decoded = match &comps.diffusion_decoder {
            Some(decoder) => {
                decoder.decode_seeded(&output.video_latent, seed.wrapping_add(4), None)?
            }
            None => comps.vae.decode_budgeted(&output.video_latent)?,
        };
        let images = pipeline::frames_to_images(&decoded)?;
        let audio = match &comps.audio {
            Some(chain) => Some(pipeline::decode_audio_track(
                &chain.decoder,
                &chain.vocoder,
                &output.audio_latent,
                chain.sample_rate,
            )?),
            None => None,
        };
        Ok((images, output.playback_fps.round() as u32, audio))
    }

    fn render(
        &self,
        req: &GenerationRequest,
        comps: &Components,
        on_progress: &mut dyn FnMut(Progress),
    ) -> CResult<(Vec<Image>, u32, Option<AudioTrack>)> {
        let fps = req.fps.unwrap_or(DEFAULT_FPS);
        let seed = req.seed.unwrap_or_else(gen_core::default_seed);
        let mut orchestration = pipeline::TwoStageOrchestration::new(seed);
        // Every render begins at stage one with adapter pass zero.  A dev transformer therefore
        // runs raw stage one; the incoming distilled rank-450 adapter contract remains selected at
        // pass one only when the shared stage-two refinement begins.

        // Text encode → video (1,256,4096) + audio (1,256,2048) contexts (one Gemma pass).
        let (input_ids, mask01) = self.tokenize(&comps.tokenizer, &req.prompt)?;
        let (video_ctx, audio_ctx) = comps.te.encode_both(&input_ids, &mask01)?;
        let negative_context = if self.transformer_variant.is_dev() {
            // Empty negative text is the official unconditional conditioning when a caller omits
            // `negative_prompt`; it still takes a real Gemma pass, never aliases the positive stack.
            let negative_prompt = req.negative_prompt.as_deref().unwrap_or("");
            let (negative_ids, negative_mask) = self.tokenize(&comps.tokenizer, negative_prompt)?;
            Some(comps.te.encode_both(&negative_ids, &negative_mask)?)
        } else {
            None
        };
        let mut predict = || -> gen_core::Result<f32> {
            let head = comps.duration_head.as_ref().ok_or_else(|| {
                gen_core::Error::Unsupported(
                    "ltx_2_5_distilled: auto_duration requires the DurationHead component".into(),
                )
            })?;
            head.predict_seconds(Some(&video_ctx), Some(&audio_ctx))
                .map_err(|e| gen_core::Error::Msg(e.to_string()))
        };
        let frames = candle_gen::gen_core::duration_head::resolve_request_num_frames(
            req.frames,
            req.auto_duration,
            fps as f32,
            config::TEMPORAL_SCALE as u32,
            &mut predict,
        )?
        .unwrap_or(DEFAULT_FRAMES);
        let uses_dfr = req.num_generated_keyframes.is_some_and(|count| count > 0)
            || req
                .temporal_upsample_rounds
                .is_some_and(|rounds| rounds > 0);
        if uses_dfr {
            return self.render_dfr(
                req,
                comps,
                frames,
                fps,
                seed,
                &video_ctx,
                &audio_ctx,
                negative_context
                    .as_ref()
                    .map(|(video, audio)| (video, audio)),
                on_progress,
            );
        }

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
        let stage1_plan = ExecutionPlan::for_variant(self.transformer_variant);
        let stage1_steps = stage1_plan.transitions() as u32;
        let total_steps = stage1_steps + STAGE2_SIGMAS.len() as u32 - 1;
        let mut stage1_fold = pipeline::StageProgressFold::new(0, stage1_steps, total_steps);
        let mut stage1_progress = |event: Progress| {
            if let Some(event) = stage1_fold.fold(event) {
                on_progress(event);
            }
        };
        let (vlat, alat) = if self.transformer_variant.is_dev() {
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
                    .noised(&vnoise, stage1_plan.sigmas[0])?;
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
            let (negative_video_ctx, negative_audio_ctx) =
                negative_context.as_ref().ok_or_else(|| {
                    CandleError::Msg(
                        "ltx_2_5: dev transformer is missing its negative-text conditioning".into(),
                    )
                })?;
            let mut stage1_forward = || Ok(());
            let stage1_stg_blocks: Vec<usize> = stage1_plan
                .stg_blocks
                .iter()
                .map(|&block| block as usize)
                .collect();
            let (state, audio) = pipeline::denoise_av_dev_conditioned(
                &comps.avdit,
                &state,
                &anoise,
                &video_ctx,
                &audio_ctx,
                negative_video_ctx,
                negative_audio_ctx,
                af,
                &audio_grid,
                &stage1_plan.sigmas,
                &stage1_stg_blocks,
                crate::params::LTX_2_5_PARAMS.video_guider,
                crate::params::LTX_2_5_PARAMS.audio_guider,
                &req.cancel,
                &mut stage1_forward,
                &mut stage1_progress,
            )?;
            let generated = state.latent.narrow(1, 0, state.target_tokens)?;
            (
                pipeline::unflatten_latent(&generated, t_lat, h_lat, w_lat)?,
                audio,
            )
        } else if conditioned {
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
            let stage1_video_request = pipeline::flatten_latent(&vnoise)?;
            let stage1_audio_request = pipeline::flatten_audio_latent(&anoise)?;
            let stage1_rope = comps.avdit.prepare_rope(
                &stage1_video_request,
                &stage1_audio_request,
                &video_grid,
                &audio_grid,
            )?;
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
                    let (vvel, avel) = comps.avdit.forward_prepared(
                        &vflat,
                        &aflat,
                        sigma as f64,
                        &video_ctx,
                        &audio_ctx,
                        &video_grid,
                        &audio_grid,
                        &stage1_rope,
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
        let mut stage2_fold = pipeline::StageProgressFold::new(stage1_steps, 3, total_steps);
        let mut stage2_progress = |event: Progress| {
            if let Some(event) = stage2_fold.fold(event) {
                on_progress(event);
            }
        };
        let stage2 = if stage2_keyframes.is_empty() {
            let stage2_video_request = pipeline::flatten_latent(&stage2_initial.video)?;
            let stage2_audio_request = pipeline::flatten_audio_latent(&stage2_initial.audio)?;
            let stage2_rope = comps.avdit.prepare_rope(
                &stage2_video_request,
                &stage2_audio_request,
                &stage2_grid,
                &audio_grid,
            )?;
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
                        let (vvel, avel) = comps.avdit.forward_prepared(
                            &vflat,
                            &aflat,
                            sigma as f64,
                            &video_ctx,
                            &audio_ctx,
                            &stage2_grid,
                            &audio_grid,
                            &stage2_rope,
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
        let decoded = match &comps.diffusion_decoder {
            // The explicit `diffusion_video_vae` LoadSpec component is an alternate decoder, not a
            // second denoise route.  It consumes the same final normalized latent as the conv VAE.
            Some(decoder) => decoder.decode_seeded(&vlat, seed.wrapping_add(4), None)?,
            None => match memory_strategy::selected_decode_cap(req)? {
                Some((edge, overlap)) => comps
                    .vae
                    .decode_budgeted_with_spatial_cap(&vlat, edge, overlap)?,
                None => comps.vae.decode_budgeted(&vlat)?,
            },
        };
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
    memory_strategy: Option<gen_core::MemoryProviderContract>,
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
        // DFR knobs (sc-18789): the 2.3 checkpoint has no learned keyframe-slot marker
        // (`use_keyframes_abs_pos_embedding: false`); refuse up front like the reference's
        // `assert_generated_keyframes_supported`, typed as `Unsupported` (the mlx twin refuses the
        // same way).
        //
        // These run BEFORE the shared floor on purpose (sc-18778) — see the matching comment in
        // `mlx-gen-ltx`'s `validate_request`. The floor refuses the same knobs from this
        // descriptor's (defaulted-off) `supports_generated_keyframes` /
        // `max_temporal_upsample_rounds`; going first only preserves the message that names the
        // checkpoint generation which does support them.
        if req.num_generated_keyframes.is_some_and(|n| n > 0) {
            return Err(gen_core::Error::Unsupported(
                "ltx: num_generated_keyframes requires a generated-keyframe checkpoint \
                 (use_keyframes_abs_pos_embedding, LTX >= 2.5)"
                    .into(),
            ));
        }
        if req.temporal_upsample_rounds.is_some_and(|r| r > 0) {
            return Err(gen_core::Error::Unsupported(
                "ltx: temporal_upsample_rounds requires the LTX-2.5 DFR pipeline (generated \
                 keyframe slots + the temporal latent upsampler)"
                    .into(),
            ));
        }
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
        let mut multi_references: Option<&[Image]> = None;
        let mut control_clip_count = 0usize;
        let mut appended_frames = 0usize;
        for entry in &req.conditioning {
            match entry {
                Conditioning::Reference { .. } => reference_count += 1,
                Conditioning::MultiReference { images } => {
                    if multi_references.replace(images.as_slice()).is_some() {
                        return Err(gen_core::Error::Msg(
                            "ltx: replace_person accepts exactly one ordered MultiReference carrier"
                                .into(),
                        ));
                    }
                }
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
        let replace_person = control_clip_count == 1 || multi_references.is_some();
        if replace_person {
            if control_clip_count != 1 {
                return Err(gen_core::Error::Msg(
                    "ltx: replace_person requires exactly one ControlClip".into(),
                ));
            }
            let Some(images) = multi_references else {
                return Err(gen_core::Error::Msg(
                    "ltx: replace_person requires exactly one ordered MultiReference carrier"
                        .into(),
                ));
            };
            if reference_count != 0
                || req.conditioning.iter().any(|entry| {
                    matches!(
                        entry,
                        Conditioning::Keyframe { .. } | Conditioning::VideoClip { .. }
                    )
                })
            {
                return Err(gen_core::Error::Msg(
                    "ltx: replace_person cannot be mixed with Reference, Keyframe, or VideoClip conditioning"
                        .into(),
                ));
            }
            conditioning::compose_ordered_character_references(images, req.width, req.height)
                .map_err(|error| gen_core::Error::Msg(error.to_string()))?;
            if let Some(control) = req.control_clip() {
                if control.start_frame != 0 {
                    return Err(gen_core::Error::Msg(format!(
                        "ltx: replace_person ControlClip must start at latent frame 0 (got {})",
                        control.start_frame
                    )));
                }
            }
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

    fn memory_strategy_contract(&self) -> Option<&gen_core::MemoryProviderContract> {
        self.memory_strategy.as_ref()
    }

    fn memory_strategy_safety_check(
        &self,
        context: &gen_core::MemoryRunContext,
    ) -> gen_core::MemorySafetyDecision {
        let Some(contract) = self.memory_strategy.as_ref() else {
            return if context.selection.strategy == gen_core::MemoryStrategy::Resident {
                gen_core::MemorySafetyDecision::Accept
            } else {
                gen_core::MemorySafetyDecision::Reject {
                    reason: format!(
                        "{MODEL_ID}: loaded route has no calibrated q4 I2V memory contract"
                    ),
                }
            };
        };
        memory_strategy::safety_check(contract, context)
    }

    fn begin_memory_strategy_request(
        &self,
        context: &gen_core::MemoryRunContext,
    ) -> gen_core::Result<Option<Box<dyn gen_core::MemoryRequestScope + '_>>> {
        let Some(contract) = self.memory_strategy.as_ref() else {
            return Ok(None);
        };
        memory_strategy::begin_request(contract, self.device.clone(), context)
    }
}

/// LTX-2.3 distilled video descriptor — two-stage rectified-flow (no CFG / negative prompt;
/// guidance is distilled in) with image/keyframe/ordered-replace-person/IC-LoRA clip conditioning. The denoise step count is
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
                ConditioningKind::MultiReference,
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
            // The practical CUDA route is the pre-packed split q4 tier. Dense and q8 remain
            // loadable compatibility paths but are deliberately not advertised as this provider's
            // request-scoped I2V memory surface.
            supported_quants: &[Quant::Q4],
            // sc-18764 / R2 (per-backend capability honesty). Stated EXPLICITLY rather than left to
            // `Default`, because it is a load-bearing negative: this crate has no `enhance` module
            // on either checkpoint generation, so `enhance_prompt` must be refused by the shared
            // floor, never silently ignored. `ltx_2_5_enhance_capability.rs` asserts both the flag
            // and the refusal it produces; flipping this literal to `true` is the mutation that
            // must make that test fail.
            supports_prompt_enhancement: false,
            ..Default::default()
        },
    }
}

/// The ordinary Candle provider descriptor for LTX-2.5's split Gemma-4 bundle.  Axes that are not
/// materialised by this route stay explicitly closed; they are never accepted as inert metadata.
pub fn descriptor_25() -> ModelDescriptor {
    descriptor_25_for_variant(TransformerVariant::Distilled)
}

/// The descriptor returned by an ordinary split-provider load after the transformer's safetensors
/// identity has been parsed.  Catalog discovery remains conservative (`descriptor_25` advertises
/// distilled), while a loaded dev checkpoint cannot accept the distilled 8-step / no-negative
/// request surface.
fn descriptor_25_for_variant(transformer_variant: TransformerVariant) -> ModelDescriptor {
    let mut out = descriptor();
    out.id = MODEL_25_ID;
    out.capabilities.supports_lora = true;
    out.capabilities.supports_lokr = false;
    out.capabilities.supports_prompt_enhancement = false;
    out.capabilities.supports_auto_duration = true;
    out.capabilities.supports_generated_keyframes = true;
    out.capabilities.max_temporal_upsample_rounds = 2;
    out.capabilities.supports_diffusion_decoder = true;
    match transformer_variant {
        TransformerVariant::Distilled => {
            out.capabilities.supported_steps = StepSupport::Exact(vec![NATIVE_STEPS]);
            out.capabilities.supports_negative_prompt = false;
        }
        TransformerVariant::Dev => {
            out.capabilities.supported_steps = StepSupport::Exact(vec![30]);
            out.capabilities.supports_negative_prompt = true;
            // The dev transformer has one official guided Euler trajectory.  Leaving the curated
            // distilled menu open would accept a knob that its variant cannot execute.
            out.capabilities.samplers.clear();
        }
    }
    out
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

/// Published filenames for the learned x2 LTX refinement checkpoint. The
/// converter stages the upstream name; the short name is the package-local
/// canonical form. A directory carrying both is ambiguous and rejected rather
/// than choosing one by incidental listing order.
const UPSAMPLER_FILENAMES: [&str; 2] = [
    "upsampler.safetensors",
    "ltx-2.3-spatial-upscaler-x2-1.1.safetensors",
];

/// Resolve the published learned refinement component. A `File` source is
/// exact; a directory source and ordinary snapshot accept either published
/// filename but fail closed when both are staged.
pub(crate) fn canonical_upsampler_file(path: &Path) -> CResult<PathBuf> {
    if path.is_file() {
        return Ok(path.to_path_buf());
    }
    let matches = UPSAMPLER_FILENAMES
        .iter()
        .map(|name| path.join(name))
        .filter(|candidate| candidate.is_file())
        .collect::<Vec<_>>();
    match matches.as_slice() {
        [single] => Ok(single.clone()),
        [] => Err(CandleError::Msg(format!(
            "ltx requires the learned spatial upscaler — provide LoadSpec::components[\"spatial_upscaler\"] \
             as one of {} in a directory (looked in {})",
            UPSAMPLER_FILENAMES.join(", "),
            path.display()
        ))),
        _ => Err(CandleError::Msg(format!(
            "ltx spatial upscaler directory {} is ambiguous: found multiple published files ({})",
            path.display(),
            matches
                .iter()
                .map(|candidate| candidate.display().to_string())
                .collect::<Vec<_>>()
                .join(", ")
        ))),
    }
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
    // Layout gate (sc-18757): which LTX generation this checkpoint holds is decided by its declared
    // `model_version` — NOT by which files are present and NOT by their names. This engine implements
    // the LTX-2.3 all-in-one component set; an LTX-2.5 split bundle carries a different DiT, a Gemma 4
    // text encoder and a diffusion VAE, so it is refused here, by version, with the version named.
    // Without this gate a 2.5 tree fell through to `ltx_checkpoint_in`, which picks a file by NAME and
    // would have handed the 2.5 transformer to the 2.3 loader. `crate::bundle` resolves the split
    // layout for the engine that implements it.
    let layout_probe = match &spec.weights {
        WeightsSource::File(p) => p.clone(),
        WeightsSource::Dir(_) => root.clone(),
    };
    let declared_version = bundle::declared_model_version(&layout_probe)?;
    if gen_core::ltx_checkpoint::layout_for_declared_version(declared_version.as_deref())
        == LtxCheckpointLayout::Split
    {
        return Err(gen_core::Error::Unsupported(format!(
            "ltx_2_3_distilled: {} declares model_version {:?}, which ships as a split-component \
             bundle (per-component transformer / text encoder / video VAE / audio VAE / duration \
             head / latent upsamplers, each with its own config). This engine loads the all-in-one \
             LTX-2.3 layout only.",
            layout_probe.display(),
            declared_version.unwrap_or_default(),
        )));
    }
    if spec.quantize.is_some() && spec.quantize != Some(Quant::Q4) {
        return Err(gen_core::Error::Unsupported(
            "candle ltx supports only the pre-packed q4 tier; q8/on-the-fly quantization are not a released route".into(),
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
    #[cfg(feature = "cuda")]
    let memory_strategy: Option<gen_core::MemoryProviderContract> =
        memory_strategy::contract_for_loaded(spec)?.map(|(contract, _tier)| contract);
    #[cfg(not(feature = "cuda"))]
    let memory_strategy = None;
    let device = candle_gen::default_device()?;
    Ok(Box::new(LtxGenerator {
        descriptor: descriptor(),
        root,
        device,
        gemma_override,
        upsampler_override,
        adapters: spec.adapters.clone(),
        memory_strategy,
        components: Mutex::new(None),
    }))
}

/// Lazy LTX-2.5 split-bundle provider.  Metadata/configuration is resolved at ordinary catalog
/// load time; multi-gigabyte tensors remain request-scoped in the existing LTX renderer.
pub struct Ltx25Generator {
    descriptor: ModelDescriptor,
    bundle: LtxBundle,
    device: Device,
    use_diffusion_decoder: bool,
    /// Parsed from the split transformer metadata at ordinary provider load time and re-threaded
    /// through the request-local pipeline materializer.
    transformer_variant: TransformerVariant,
    /// Exact selected source mode; retained by the ordinary provider so a later load cannot
    /// silently report q4/bf16 after a distinct selector was requested.
    quant_mode: Ltx25QuantMode,
    /// Selected LoRA stack retained through the lazy provider boundary and installed while the
    /// request-scoped split transformer is materialised.
    adapters: Vec<AdapterSpec>,
    components: Mutex<Option<Components>>,
}

/// Load the split LTX-2.5 route through the standard generator registry.  The selected decoder is
/// intentional: staging `diffusion_video_vae` selects DiffVAE; otherwise the conv decoder is used.
pub fn load_25(spec: &LoadSpec) -> gen_core::Result<Box<dyn Generator>> {
    let known = bundle::split_component_ids();
    gen_core::reject_unknown_components(spec, &known, MODEL_25_ID)?;
    let quant_mode = Ltx25QuantMode::from_load_spec(spec)?;
    let device = candle_gen::default_device()?;
    let gpu = Ltx25GpuGeneration::from_device(&device)?;
    match quant_eval::admit(quant_mode, gpu, quant_eval::ACCEPTED_MEASUREMENT_RECEIPTS) {
        Ltx25QuantAdmission::Admitted => {}
        Ltx25QuantAdmission::Refused { reason } => {
            return Err(gen_core::Error::Unsupported(reason));
        }
    }
    let resolved = bundle::resolve_split_bundle(spec)?;
    if resolved.layout() != LtxCheckpointLayout::Split {
        return Err(gen_core::Error::Msg(format!(
            "ltx_2_5_distilled: expected an LTX-2.5 split-component bundle, got {}",
            resolved.layout().id()
        )));
    }
    let transformer_variant = TransformerVariant::from_bundle(&resolved)
        .map_err(|error| gen_core::Error::Msg(error.to_string()))?;
    bundle::assert_gemma_version(&resolved)?;
    let use_diffusion_decoder = spec
        .components
        .contains_key(LtxComponent::DiffusionVideoVae.id());
    // Validate every production component/config now.  The actual constructors are deliberately
    // deferred until generation so normal provider/catalog construction is weights-free.
    let pipe = Pipeline::load_split(
        resolved.clone(),
        &Device::Cpu,
        use_diffusion_decoder,
        quant_mode,
        transformer_variant,
    )?;
    resolved.require(LtxComponent::DurationHead)?;
    resolved.require(LtxComponent::TemporalUpsampler)?;
    drop(pipe);
    Ok(Box::new(Ltx25Generator {
        descriptor: descriptor_25_for_variant(transformer_variant),
        bundle: resolved,
        device,
        use_diffusion_decoder,
        transformer_variant,
        quant_mode,
        adapters: spec.adapters.clone(),
        components: Mutex::new(None),
    }))
}

impl Generator for Ltx25Generator {
    fn descriptor(&self) -> &ModelDescriptor {
        &self.descriptor
    }

    fn validate(&self, req: &GenerationRequest) -> gen_core::Result<()> {
        self.descriptor
            .capabilities
            .validate_request(MODEL_25_ID, req)?;
        if req.prompt.is_empty() {
            return Err(gen_core::Error::Msg(
                "ltx_2_5_distilled: prompt must not be empty".into(),
            ));
        }
        if !req.width.is_multiple_of(SIZE_MULTIPLE) || !req.height.is_multiple_of(SIZE_MULTIPLE) {
            return Err(gen_core::Error::Msg(format!(
                "ltx_2_5_distilled: width/height must be multiples of {SIZE_MULTIPLE} (got {}x{})",
                req.width, req.height
            )));
        }
        if let Some(frames) = req.frames {
            if frames == 0 || frames % config::TEMPORAL_SCALE as u32 != 1 {
                return Err(gen_core::Error::Msg(format!(
                    "ltx_2_5_distilled: frames must satisfy frames % {} == 1 (got {frames})",
                    config::TEMPORAL_SCALE
                )));
            }
        }
        Ok(())
    }

    fn generate(
        &self,
        req: &GenerationRequest,
        on_progress: &mut dyn FnMut(Progress),
    ) -> gen_core::Result<GenerationOutput> {
        self.validate(req)?;
        let pipe = Pipeline::load_split(
            self.bundle.clone(),
            &self.device,
            self.use_diffusion_decoder,
            self.quant_mode,
            self.transformer_variant,
        )?;
        let mut slot = candle_gen::lock_recover(&self.components);
        let want_encoder = needs_ltx_vae_encoder(req);
        if slot
            .as_ref()
            .is_none_or(|components| components.vae_has_encoder != want_encoder)
        {
            *slot = None;
            *slot = Some(pipe.load_components(&self.adapters, want_encoder)?);
        }
        let components = slot.as_ref().expect("split components populated").clone();
        drop(slot);
        let (frames, fps, audio) = pipe.render(req, &components, on_progress)?;
        Ok(GenerationOutput::Video { frames, fps, audio })
    }
}

candle_gen::register_generators! {
    pub(crate) const REGISTRATION = descriptor => load;
    footprint = component_footprint
}

candle_gen::register_generators! {
    pub(crate) const REGISTRATION_25 = descriptor_25 => load_25
}

/// Add the Candle LTX generator and trainer to an explicit media registry builder.
pub fn register_providers(
    registry: candle_gen::gen_core::ProviderRegistryBuilder,
) -> candle_gen::gen_core::ProviderRegistryBuilder {
    let registry = registry
        .register_generator(REGISTRATION)
        .register_generator(REGISTRATION_25);
    #[cfg(feature = "cuda")]
    let registry = registry
        .register_memory_strategy(memory_strategy::MEMORY_REGISTRATION)
        .register_memory_contract_fixture(memory_strategy::MEMORY_FIXTURE)
        .register_memory_behavior(memory_strategy::MEMORY_BEHAVIOR);
    registry
        .register_trainer(training::TRAINER_REGISTRATION)
        .register_trainer(training::TRAINER_REGISTRATION_25)
}

/// Register the weights-free Candle/CUDA q4 I2V memory surface without requiring CUDA or weights.
pub fn register_memory_contract_surfaces(
    registry: candle_gen::gen_core::ProviderRegistryBuilder,
) -> candle_gen::gen_core::ProviderRegistryBuilder {
    registry
        .register_memory_strategy(memory_strategy::MEMORY_REGISTRATION)
        .register_memory_contract_fixture(memory_strategy::MEMORY_FIXTURE)
        .register_memory_behavior(memory_strategy::MEMORY_BEHAVIOR)
}

/// Build the complete explicit Candle LTX provider catalog.
pub fn provider_registry() -> candle_gen::gen_core::Result<candle_gen::gen_core::ProviderRegistry> {
    register_providers(candle_gen::gen_core::ProviderRegistryBuilder::new()).build()
}

/// Resolve the load-bearing VAE geometry for a Candle LTX generator id.
pub fn vae_tiling(provider_id: &str) -> Option<candle_gen::gen_core::tiling::VaeTiling> {
    (provider_id == MODEL_ID || provider_id == MODEL_25_ID).then_some(VAE_TILING)
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

        assert_eq!(generators, ["ltx_2_3_distilled", "ltx_2_5_distilled"]);
        assert_eq!(trainers, ["ltx_2_3", "ltx_2_5_distilled"]);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn ltx25_ordinary_provider_validation_enforces_declared_geometry_and_closed_axes() {
        let generator = Ltx25Generator {
            descriptor: descriptor_25(),
            // Validation is deliberately independent of tensor construction; this is the ordinary
            // provider object the catalog returns after its split-bundle load gate.
            bundle: LtxBundleBuilder::new()
                .build()
                .expect("empty validation-only bundle"),
            device: Device::Cpu,
            use_diffusion_decoder: false,
            transformer_variant: TransformerVariant::Distilled,
            quant_mode: Ltx25QuantMode::Bf16,
            adapters: Vec::new(),
            components: Mutex::new(None),
        };
        let request = GenerationRequest {
            prompt: "a quiet moonlit harbor".into(),
            width: 512,
            height: 512,
            frames: Some(17),
            ..Default::default()
        };
        Generator::validate(&generator, &request).expect("64px / 1+8*k must pass");

        let mut bad_width = request.clone();
        bad_width.width = 672; // divisible by 32 but not the two-stage 64px provider stride.
        assert!(Generator::validate(&generator, &bad_width)
            .unwrap_err()
            .to_string()
            .contains("multiples of 64"));

        let mut bad_frames = request.clone();
        bad_frames.frames = Some(16);
        assert!(Generator::validate(&generator, &bad_frames)
            .unwrap_err()
            .to_string()
            .contains("frames"));

        let mut dfr = request.clone();
        dfr.num_generated_keyframes = Some(1);
        Generator::validate(&generator, &dfr).expect("declared generated-keyframe axis must pass");
        let mut temporal = request;
        temporal.temporal_upsample_rounds = Some(2);
        Generator::validate(&generator, &temporal).expect("declared temporal-DFR axis must pass");

        assert!(generator.descriptor.capabilities.supports_diffusion_decoder);
        assert_eq!(
            generator.descriptor.capabilities.supported_quants,
            &[Quant::Q4],
            "unmeasured int8-convrot/nvfp4 must stay out of the catalog surface"
        );
    }

    #[test]
    fn ltx25_loaded_variant_changes_the_ordinary_provider_request_surface() {
        let distilled = descriptor_25_for_variant(TransformerVariant::Distilled);
        assert_eq!(
            distilled.capabilities.supported_steps,
            StepSupport::Exact(vec![NATIVE_STEPS])
        );
        assert!(!distilled.capabilities.supports_negative_prompt);

        let dev = descriptor_25_for_variant(TransformerVariant::Dev);
        assert_eq!(
            dev.capabilities.supported_steps,
            StepSupport::Exact(vec![30])
        );
        assert!(dev.capabilities.supports_negative_prompt);
        assert!(dev.capabilities.samplers.is_empty());

        let request = GenerationRequest {
            prompt: "a red kite over the sea".into(),
            width: 512,
            height: 512,
            frames: Some(17),
            steps: Some(30),
            negative_prompt: Some("blurred motion".into()),
            ..Default::default()
        };
        assert!(dev
            .capabilities
            .validate_request(MODEL_25_ID, &request)
            .is_ok());
        assert!(distilled
            .capabilities
            .validate_request(MODEL_25_ID, &request)
            .is_err());
    }

    #[test]
    fn ltx25_catalog_route_reaches_the_quant_policy_before_bundle_loading() {
        // This is intentionally an ordinary registry load, not a selector helper. The nonexistent
        // root proves that Q8 is parsed as the LTX ConvRot option at the provider boundary before
        // file discovery could accidentally select a different precision.
        let spec =
            LoadSpec::new(WeightsSource::Dir("/nonexistent/ltx25".into())).with_quant(Quant::Q8);
        let result = crate::provider_registry()
            .expect("provider registry")
            .load(MODEL_25_ID, &spec);
        let error = match result {
            Ok(_) => panic!("unmeasured ConvRot must fail closed"),
            Err(error) => error.to_string(),
        };
        assert!(error.contains("int8-convrot"), "got: {error}");
        assert!(
            error.contains("terminal measurement case") || error.contains("not catalog-adopted"),
            "quant policy must run before missing-bundle discovery, got: {error}"
        );
    }

    #[test]
    fn ltx25_provider_path_cannot_bypass_duration_or_dfr_execution() {
        // Mutation-sensitive production-path witness: deleting either call site makes the ordinary
        // `Ltx25Generator::generate → Pipeline::render` path fail, rather than leaving a helper test
        // green while a request silently ignores its advanced axis.
        let source = include_str!("lib.rs");
        let provider = source
            .split("impl Generator for Ltx25Generator")
            .nth(1)
            .expect("LTX-2.5 provider implementation exists");
        assert!(provider.contains("pipe.render(req, &components, on_progress)"));
        let renderer = source
            .split("fn render(")
            .nth(1)
            .expect("ordinary renderer exists");
        assert!(renderer.contains("resolve_request_num_frames("));
        assert!(renderer.contains("self.render_dfr("));
        assert!(renderer.contains("denoise_av_dev_conditioned("));
        assert!(renderer.contains("negative_context"));
        assert!(source.contains("dfr::generate_dfr_av_latents("));
    }

    #[test]
    fn ltx25_provider_path_cannot_bypass_selected_quant_mode() {
        // Mutation-sensitive companion to the real registry-load witness above: a future refactor
        // cannot remove the materialization-time binding and accidentally load bf16 components
        // after an advanced selection has passed the policy gate.
        let source = include_str!("lib.rs");
        let loader = source
            .split("pub fn load_25")
            .nth(1)
            .expect("LTX-2.5 registry loader exists");
        assert!(loader.contains("Ltx25QuantMode::from_load_spec(spec)"));
        assert!(loader.contains("quant_eval::admit(quant_mode"));
        let split_loader = source
            .split("fn load_split(")
            .nth(1)
            .expect("split materializer exists");
        assert!(split_loader.contains("quant_mode.validate_bundle_source(&bundle)"));
    }

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
    fn directory_component_accepts_the_official_published_upscaler_filename() {
        let root = tempfile::tempdir().unwrap();
        let staged = tempfile::tempdir().unwrap();
        let published = staged
            .path()
            .join("ltx-2.3-spatial-upscaler-x2-1.1.safetensors");
        std::fs::File::create(&published).unwrap();
        let spec = LoadSpec::new(WeightsSource::Dir(root.path().to_path_buf())).with_component(
            gen_core::LTX_SPATIAL_UPSCALER_COMPONENT,
            WeightsSource::Dir(staged.path().to_path_buf()),
        );
        assert_eq!(spec_upsampler_file(&spec, root.path()).unwrap(), published);
    }

    #[test]
    fn directory_component_rejects_ambiguous_published_upscalers() {
        let staged = tempfile::tempdir().unwrap();
        for name in UPSAMPLER_FILENAMES {
            std::fs::File::create(staged.path().join(name)).unwrap();
        }
        let error = canonical_upsampler_file(staged.path())
            .unwrap_err()
            .to_string();
        assert!(error.contains("ambiguous"), "got: {error}");
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
    /// either published learned-upscaler filename is co-located (lazy weight resolution).
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
                "ltx-2.3-spatial-upscaler-x2-1.1.safetensors",
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
                ConditioningKind::MultiReference,
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
            vec![
                Conditioning::ControlClip {
                    frames: vec![image.clone()],
                    mask: vec![mask.clone()],
                    masking_strength: 0.9,
                    start_frame: 0,
                    mode: gen_core::ReplacementMode::FaceOnly,
                },
                Conditioning::MultiReference {
                    images: vec![image.clone(), image.clone(), image.clone(), image.clone()],
                },
            ],
        ] {
            assert!(generator
                .validate(&GenerationRequest {
                    conditioning,
                    ..base.clone()
                })
                .is_ok());
        }
        for reference_count in 1..=4 {
            assert!(
                generator
                    .validate(&GenerationRequest {
                        conditioning: vec![
                            Conditioning::ControlClip {
                                frames: vec![image.clone()],
                                mask: vec![mask.clone()],
                                masking_strength: 0.9,
                                start_frame: 0,
                                mode: gen_core::ReplacementMode::FaceOnly,
                            },
                            Conditioning::MultiReference {
                                images: vec![image.clone(); reference_count],
                            },
                        ],
                        ..base.clone()
                    })
                    .is_ok(),
                "{reference_count} ordered references must be admitted"
            );
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
                conditioning: vec![control.clone(), control.clone()],
                ..base.clone()
            },
            GenerationRequest {
                sampler: Some("heun".into()),
                conditioning: vec![Conditioning::Reference {
                    image: image.clone(),
                    strength: Some(1.0),
                }],
                ..base.clone()
            },
        ];
        for request in cases {
            assert!(
                generator.validate(&request).is_err(),
                "must reject {request:?}"
            );
        }

        // SC-20776: all malformed/crossed replace-person requests fail in `validate`, before the
        // lazy provider ever constructs VAE/Gemma/AvDiT components.
        for conditioning in [
            vec![control.clone()],
            vec![Conditioning::MultiReference {
                images: vec![image.clone()],
            }],
            vec![
                control.clone(),
                Conditioning::Reference {
                    image: image.clone(),
                    strength: None,
                },
                Conditioning::MultiReference {
                    images: vec![image.clone()],
                },
            ],
            vec![
                control.clone(),
                Conditioning::MultiReference {
                    images: vec![image.clone(); 5],
                },
            ],
        ] {
            assert!(generator
                .validate(&GenerationRequest {
                    conditioning,
                    ..base.clone()
                })
                .is_err());
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

    /// sc-18789: the DFR knobs are refused on the 2.3 checkpoint with a TYPED `Unsupported`
    /// (candle twin of the mlx refusal); `0`/`None` stay accepted.
    #[test]
    fn validate_refuses_dfr_knobs_on_2_3_typed() {
        let spec = LoadSpec::new(WeightsSource::Dir("/nonexistent".into()));
        let model = crate::provider_registry()
            .unwrap()
            .load(MODEL_ID, &spec)
            .unwrap();
        let base = GenerationRequest {
            prompt: "p".into(),
            width: 640,
            height: 384,
            frames: Some(25),
            ..Default::default()
        };
        for (label, req) in [
            (
                "num_generated_keyframes",
                GenerationRequest {
                    num_generated_keyframes: Some(3),
                    ..base.clone()
                },
            ),
            (
                "temporal_upsample_rounds",
                GenerationRequest {
                    temporal_upsample_rounds: Some(1),
                    ..base.clone()
                },
            ),
        ] {
            let err = model.validate(&req).expect_err(label);
            assert!(
                matches!(err, gen_core::Error::Unsupported(_)),
                "{label}: {err}"
            );
            assert!(err.to_string().contains("2.5"), "{label}: {err}");
        }
        let off = GenerationRequest {
            num_generated_keyframes: Some(0),
            temporal_upsample_rounds: Some(0),
            ..base
        };
        model.validate(&off).expect("0 is off");
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
