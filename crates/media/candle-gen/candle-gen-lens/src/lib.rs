//! # candle-gen-lens
//!
//! The **Lens / Lens-Turbo** text-to-image provider for [`candle-gen`](candle_gen) — the candle
//! (Windows/CUDA) sibling of the `mlx-gen` Lens port (epic 3164). Lens is a three-component model:
//!
//! 1. a **gpt-oss-20b** MoE LLM used **encoder-only** ([`text_encoder`]) — 24-layer / 32-expert /
//!    top-4, attention sinks, alternating sliding/full attention, YaRN RoPE, clamped-SwiGLU experts,
//!    MXFP4-native expert weights; run forward capturing hidden states at `[5, 11, 17, 23]`;
//! 2. a **48-layer dual-stream MMDiT** ([`transformer`], `LensTransformer2DModel`, sc-5112) —
//!    fused-QKV joint attention over `[img, txt]`, complex axial RoPE ([`rope`]), AdaLN dual
//!    modulation, SwiGLU MLPs, multi-layer text front-end;
//! 3. the **Flux.2 VAE** ([`vae`], `AutoencoderKLFlux2`, sc-5113) — reused from `candle-gen-flux2`
//!    via a thin decode shim (reshape the DiT output into the packed NCHW grid → `decode_packed`).
//!
//! This crate is being built story-by-story under epic **5107**. The first landed piece is the
//! gpt-oss encoder decoder block ([`text_encoder`], sc-5108): a from-scratch port — candle-transformers
//! ships no `gpt_oss` model (the Gate-0 spike found upstream PRs #3129/#3581/#3391 all unmerged), so
//! the decoder is adapted from the verified-parity reference in candle PR #3581 onto `candle_nn`.
//!
//! **Dtype:** the encoder runs **bf16** (the checkpoint's native non-expert dtype); the MXFP4 expert
//! weights are dequantized to bf16 at load (sc-5108 bring-up). The eventual MXFP4 → GGUF Q4 `QMatMul`
//! transcode that keeps the ~12 GB footprint is sc-5111.

pub mod adapters;
pub mod dit_train;
pub mod preview;
pub mod quant;
pub mod reasoner;
pub mod resolution;
pub mod rope;
pub mod schedule;
pub mod text;
pub mod text_encoder;
pub mod training;
pub mod transformer;
pub mod vae;

pub use adapters::{install_additive, merge_adapters, AdditiveReport, MergeReport};
pub use quant::QLinear;
pub use reasoner::{LensReasoner, DEFAULT_MAX_NEW_TOKENS};

use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex};

use candle_gen::candle_core::safetensors::MmapedSafetensors;
use candle_gen::candle_core::{DType, Device, IndexOp, Tensor};
use candle_gen::candle_nn::VarBuilder;
use candle_gen::gen_core::{
    self, AdapterSpec, Capabilities, GenerationOutput, GenerationRequest, Generator, Image,
    LoadSpec, Modality, ModelDescriptor, OffloadPolicy, PidWeights, Precision, Progress, Quant,
    SizeFloor, WeightsSource,
};
use candle_gen::{CandleError, LatentDecoder, Result as CResult};
use candle_gen_pid::PidEngine;
use rand::{rngs::StdRng, SeedableRng};

/// The PiD backbone (latent-space) tag for Lens (epic 7840 / sc-7853). Lens reuses the FLUX.2 VAE, so
/// its latent space is `flux2` — the same packed 128-ch BN-normalized student FLUX.2 resolves.
const PID_BACKBONE: &str = "flux2";

use candle_gen::gen_core::sampling::TimestepConvention;
use schedule::{cfg_rescale, lens_mu, lens_sigmas, LensSamplingDefaults, BASE, TURBO};
use text::{LensTokenizer, TXT_OFFSET};
use text_encoder::{Config as EncoderConfig, GptOssTextEncoder, DEFAULT_SELECTED_LAYERS};
use transformer::{LensDitConfig, LensTransformer};
use vae::Flux2Vae;

/// Registry id — the distilled turbo variant (4-step / guidance 1.0).
pub const MODEL_ID_TURBO: &str = "lens_turbo";
/// Registry id — the base variant (20-step / CFG 5.0).
pub const MODEL_ID_BASE: &str = "lens";

/// The VAE downsample factor (`vae_scale_factor`): a Lens latent cell maps to a 16×16 pixel tile
/// (Flux.2's 8× conv VAE composed with the 2× DiT patchify). Image dims must be multiples of this.
pub const VAE_SCALE_FACTOR: u32 = 16;

/// The one production text-encoder window SC-15800 publishes.
///
/// Real packed weights, release build, CUDA device 0 (95.59 GiB RTX PRO 6000 Blackwell), driver
/// RESERVED high-water in GiB. Each cell is the maximum over actual 25/47/92-token Lens tokenizer
/// outputs; every captured conditioning tensor and mask was byte-identical to the resident path.
///
/// | tier | resident | w1 | w2 | w4 | w8 | all 24 |
/// |---|---:|---:|---:|---:|---:|---:|
/// | q4 | 13.562 | 1.719 | 2.219 | 3.281 | 5.312 | 13.531 |
/// | q8 | 24.625 | 2.094 | 2.906 | 4.594 | 7.906 | 21.344 |
///
/// Window 1 cuts the binding conditioning peak 87.3% at q4 and 91.5% at q8. The all-covering
/// mutation restores the resident live set (q8 RESERVED is lower because the sidecar path avoids the
/// resident loader's one-time conversion transient). Prompt length changes the minimum-window peak
/// by at most 64 MiB over this sweep, so it does not alter the selected window. No wider window buys
/// a memory advantage; publish the tightest one. An end-to-end q4 Lens-Turbo request (512x512, one
/// denoise step, seed 15800) produced byte-identical pixels while reducing RESERVED request peak from
/// 21.875 GiB resident to 8.281 GiB Sequential+Deferred/window-1 (62.1%). The dense/MXFP4 control
/// measured 38.781 GiB resident, 3.156 GiB at window 1, and 38.969 GiB for all 24 layers, with exact
/// conditioning bytes. It is deliberately ineligible because opening a dense layer performs
/// source-format conversion inside each window rather than the required post-SC-16096 device-format
/// transfer.
pub const DEFAULT_TEXT_ENCODER_WINDOW: usize = 1;
/// Shared Lens/Candle ladder candidates. These match the proved MLX geometry where the backend
/// primitives are equivalent; entry-specific calibration remains owned by the catalog stories.
pub const DECODE_TILE_EDGES: &[u32] = &[768, 640, 512, 448, 384, 320, 256];
pub const DECODE_TILE_EDGE: u32 = 512;
pub const DECODE_OVERLAP: u32 = 128;
pub const ATTENTION_CHUNK_SIZE: u32 = 16_777_216;
pub const TRANSFORMER_WINDOW_SIZES: &[u32] = &[1, 2, 4, 8, 12, 24];
pub const TRANSFORMER_BLOCK_COUNT: u32 = 48;
pub const MEMORY_CALIBRATION_FINGERPRINT: &str =
    "lens-candle-cuda-shared-ladder-device-format-blocks-v1";
/// Fixed harmony-preamble `Current date:`. The preamble is the first [`TXT_OFFSET`] tokens, which are
/// **sliced off** before the DiT conditioning, so the date never reaches the image path — a fixed
/// constant keeps generation deterministic regardless of wall-clock.
pub const DEFAULT_DATE: &str = "2025-01-01";

/// The encoder + DiT run **bf16** (the checkpoint dtype). By default the MXFP4 experts dequantize to
/// bf16 at load; with `spec.quantize` they transcode to GGUF Q4/Q8 instead (sc-5111, the quantized
/// experts then compute in f32). The VAE always runs **f32** (the shared Flux.2 decoder).
const ENC_DTYPE: DType = DType::BF16;
const DIT_DTYPE: DType = DType::BF16;
const VAE_DTYPE: DType = DType::F32;

/// The loaded four components, shared by both variants (cloneable `Arc` handles).
#[derive(Clone)]
struct Components {
    text: TextComponents,
    heavy: HeavyComponents,
}

#[derive(Clone)]
struct TextComponents {
    tokenizer: Arc<LensTokenizer>,
    encoder: Arc<GptOssTextEncoder>,
}

#[derive(Clone)]
struct HeavyComponents {
    transformer: Arc<LensTransformer>,
    vae: Arc<Flux2Vae>,
    /// Optional NVIDIA PiD super-resolving decoder (epic 7840 / sc-7853), loaded once when the model
    /// was loaded with `LoadSpec::pid`. `None` ⇒ the native `Flux2Vae` decode (the default path).
    pid: Option<Arc<PidEngine>>,
}

/// A loadable Lens pipeline (the snapshot root + device + any DiT LoRA/LoKr adapters + optional DiT
/// quant level); components are loaded lazily on first use.
struct Pipeline {
    root: PathBuf,
    device: Device,
    /// LoRA/LoKr adapters merged into the `transformer/` weights on load (sc-5116). Empty = the stock
    /// mmap path.
    adapters: Vec<AdapterSpec>,
    /// Q4/Q8 quantization requested at load (`None` = dense bf16). When set it transcodes **both** the
    /// gpt-oss encoder MoE experts to GGUF (sc-5111, the ~12 GB encoder footprint) and the DiT's
    /// compute-heavy linears (sc-5117) — the encoder is the memory hog, the DiT the compute. The VAE
    /// stays f32. One `Quant` drives both; each consumer maps it to the GGUF block dtype it needs.
    quant: Option<Quant>,
    /// The `LoadSpec::pid` component captured at load (epic 7840 / sc-7853), threaded into the lazy
    /// component build so the PiD engine loads once alongside the base model. `None` when not opted in.
    pid_spec: Option<PidWeights>,
}

impl Pipeline {
    fn load(
        root: &Path,
        device: &Device,
        adapters: Vec<AdapterSpec>,
        quant: Option<Quant>,
        pid_spec: Option<PidWeights>,
    ) -> Self {
        Self {
            root: root.to_path_buf(),
            device: device.clone(),
            adapters,
            quant,
            pid_spec,
        }
    }

    /// The sorted `.safetensors` files of a snapshot sub-dir (errors if the dir or its weights are
    /// missing).
    fn component_files(&self, sub: &str) -> CResult<Vec<PathBuf>> {
        let dir = self.root.join(sub);
        if !dir.is_dir() {
            return Err(CandleError::Msg(format!(
                "lens snapshot is missing the {sub}/ dir (expected a Lens diffusers snapshot at {})",
                self.root.display()
            )));
        }
        // Shared sorted-`.safetensors` resolver (sc-8999 / F-019); the crafted "missing dir" message
        // above stays local (it names the expected Lens snapshot).
        candle_gen::sorted_safetensors(&dir, "lens")
    }

    /// A `VarBuilder` over the `.safetensors` of a snapshot sub-dir, mmapped at `dtype`.
    fn component_vb(&self, sub: &str, dtype: DType) -> CResult<VarBuilder<'static>> {
        let files = self.component_files(sub)?;
        candle_gen::mmap_var_builder(&files, dtype, &self.device)
    }

    /// The parsed [`candle_gen::quant::PackedConfig`] of a snapshot sub-dir's `config.json`, when it is a
    /// **pre-quantized MLX-packed tier** (its `config.json` carries a `quantization` block). `None` when
    /// the config is absent/unreadable or dense. Used by [`load_components`](Self::load_components) to
    /// thread the parsed `group_size` into a LOUD guard (sc-9474): the shared packed loaders
    /// (`QLinear::linear_detect` for the DiT, `repack_packed_weight` for the encoder experts) repack at
    /// the MLX default group size 64 that every hosted `SceneWorks/lens-mlx` tier uses, so a hypothetical
    /// future group-32 tier fails at load rather than silently repacking u32 codes to garbage.
    fn packed_config(&self, sub: &str) -> Option<candle_gen::quant::PackedConfig> {
        let path = self.root.join(sub).join("config.json");
        std::fs::read_to_string(&path)
            .ok()
            .and_then(|s| serde_json::from_str::<serde_json::Value>(&s).ok())
            .and_then(|v| candle_gen::quant::PackedConfig::from_config(&v))
    }

    fn packed_group_size(&self, sub: &str) -> Option<i32> {
        self.packed_config(sub).map(|config| config.group_size)
    }

    /// Assert a packed component's declared `group_size` is the MLX default 64 the shared packed loaders
    /// assume (sc-9474). A dense/absent config is `None` and skips the guard. A future non-64 tier must
    /// thread the parsed group size through the shared `*_gs` entry points (as candle-gen-boogu, sc-9410)
    /// before it can load.
    fn guard_packed_group_size(&self, sub: &str) -> CResult<()> {
        if let Some(gs) = self.packed_group_size(sub) {
            let default = candle_gen::quant::MLX_GROUP_SIZE as i32;
            if gs != default {
                return Err(CandleError::Msg(format!(
                    "lens {sub}/ packed tier declares quantization.group_size = {gs} but the \
                     candle-gen-lens packed loaders assume the MLX default {default} (sc-9474). Thread \
                     the parsed group_size through the shared `*_gs` entry points (as candle-gen-boogu \
                     does) before loading this tier."
                )));
            }
        }
        Ok(())
    }

    fn packed_sidecars(
        &self,
        sub: &str,
        cancel: &gen_core::CancelFlag,
    ) -> CResult<Option<Arc<candle_gen::quant::PackedWeightSidecars>>> {
        candle_gen::check_cancel(cancel)?;
        let Some(packed) = self.packed_config(sub) else {
            return Ok(None);
        };
        let files = self.component_files(sub)?;
        let prepared = candle_gen::quant::PackedWeightSidecars::open_and_prepare_cancelable(
            &files,
            &self.root.join(sub),
            packed,
            &self.device,
            cancel,
        );
        if cancel.is_cancelled() {
            return Err(CandleError::Canceled);
        }
        let (_, sidecars) = prepared?;
        Ok(Some(Arc::new(sidecars)))
    }

    fn load_components(&self) -> CResult<Components> {
        // sc-9474: both already-quantize→packed conversions below (the encoder MoE experts via
        // `repack_packed_weight`, the DiT projections via `QLinear::linear_detect`) repack at the MLX
        // default group size 64. Assert the parsed `quantization.group_size` matches before loading, so a
        // future group-32 tier (as boogu's is) fails LOUD instead of silently repacking to garbage.
        self.guard_packed_group_size("text_encoder")?;
        self.guard_packed_group_size("transformer")?;
        let text = self.load_resident_text_components()?;
        // Adapters ride as **forward-time additive residuals** on the DiT's projections — on BOTH the
        // packed and the dense tier (sc-11105, additive-everywhere for epic 10765). The base weight is
        // never mutated: the packed base stays packed (no dense `W` to fold into anyway), and the dense
        // base stays an unmutated mmap — so the offload/eviction path can drop-and-restore it cheaply
        // (a folded `W += δ` pins an in-memory host copy). `install_additive` equals the old dense fold
        // to f32 tolerance (~1 ULP), so this trades a byte-exact adapter render for an evictable base.
        let mut transformer = LensTransformer::new(
            &LensDitConfig::lens(),
            self.component_vb("transformer", DIT_DTYPE)?,
        )?;
        if !self.adapters.is_empty() {
            adapters::install_additive(&mut transformer, &self.adapters)?;
        }
        // Q4/Q8 the DiT's compute-heavy linears. Two routes compose (sc-9413): a packed MLX tier
        // (`SceneWorks/lens-mlx`, `.scales` present) already loaded each projection straight from the
        // packed parts inside `LensTransformer::new` (no dense staging), so this pass is a **no-op**
        // over those; a dense bf16 tier loaded dense, so this pass folds it to `Q4_0`/`Q8_0` in place.
        // The `install_additive → quantize` ordering: `AdaptLinear::quantize` folds only the **base**
        // (dense→packed) and leaves any forward-time residual attached, so a dense-tier LoRA + Q4 request
        // keeps its residual; the per-`QLinear` `quantize` no-ops on an already-packed base.
        if let Some(quant) = self.quant {
            transformer.quantize(quant)?;
        }
        let vae = Flux2Vae::new(self.component_vb("vae", VAE_DTYPE)?)?;
        // Load the optional PiD super-resolving decoder once (epic 7840 / sc-7853) when the caller
        // opted in via `LoadSpec::pid`; Lens shares the FLUX.2 VAE latent space (`flux2` student).
        let pid = match self.pid_spec.as_ref() {
            Some(spec) => Some(Arc::new(PidEngine::from_spec(
                spec,
                PID_BACKBONE,
                &self.device,
            )?)),
            None => None,
        };
        Ok(Components {
            text,
            heavy: HeavyComponents {
                transformer: Arc::new(transformer),
                vae: Arc::new(vae),
                pid,
            },
        })
    }

    fn load_resident_text_components(&self) -> CResult<TextComponents> {
        self.guard_packed_group_size("text_encoder")?;
        let tokenizer =
            LensTokenizer::from_file(self.root.join("tokenizer").join("tokenizer.json"))?;
        let encoder = GptOssTextEncoder::new_quant(
            &EncoderConfig::gpt_oss_20b(),
            self.component_vb("text_encoder", ENC_DTYPE)?,
            // `ggml_dtype` is `Err` for `Quant::Nvfp4` (no GGUF block type — NVFP4 is served by
            // `Nvfp4Linear`, sc-11042); `transpose()?` surfaces that instead of the GGUF fold path.
            self.quant.map(quant::ggml_dtype).transpose()?,
        )?;
        Ok(TextComponents {
            tokenizer: Arc::new(tokenizer),
            encoder: Arc::new(encoder),
        })
    }

    /// Encode one prompt → its `num_text_layers` captured gpt-oss layers (sliced at [`TXT_OFFSET`]) +
    /// the valid mask `[1, S]` (all-1; a single prompt is unpadded). A prompt shorter than the offset
    /// (never, for real prompts) collapses to length-0 features.
    fn load_streamable_text_components(
        &self,
        cancel: &gen_core::CancelFlag,
    ) -> CResult<TextComponents> {
        candle_gen::check_cancel(cancel)?;
        self.guard_packed_group_size("text_encoder")?;
        let tokenizer =
            LensTokenizer::from_file(self.root.join("tokenizer").join("tokenizer.json"))?;
        let files = self.component_files("text_encoder")?;
        let vb = candle_gen::mmap_var_builder(&files, ENC_DTYPE, &self.device)?;
        let quant = self.quant.map(quant::ggml_dtype).transpose()?;
        let encoder = GptOssTextEncoder::new_quant_streamable(
            &EncoderConfig::gpt_oss_20b(),
            vb,
            files,
            quant,
            self.packed_sidecars("text_encoder", cancel)?,
        )?;
        Ok(TextComponents {
            tokenizer: Arc::new(tokenizer),
            encoder: Arc::new(encoder),
        })
    }

    fn load_heavy_components(
        &self,
        stream_transformer_blocks: bool,
        cancel: &gen_core::CancelFlag,
    ) -> CResult<HeavyComponents> {
        self.guard_packed_group_size("transformer")?;
        candle_gen::check_cancel(cancel)?;
        let mut transformer = if stream_transformer_blocks {
            if !self.adapters.is_empty() {
                return Err(CandleError::Msg(
                    "lens: streamed DiT residency is not calibrated with adapters".into(),
                ));
            }
            let sidecars = self
                .packed_sidecars("transformer", cancel)?
                .ok_or_else(|| {
                    CandleError::Msg(
                        "lens: streamed DiT residency requires an already-packed q4/q8 transformer"
                            .into(),
                    )
                })?;
            LensTransformer::new_block_streamed(
                &LensDitConfig::lens(),
                self.component_vb("transformer", DIT_DTYPE)?,
                sidecars,
            )?
        } else {
            LensTransformer::new(
                &LensDitConfig::lens(),
                self.component_vb("transformer", DIT_DTYPE)?,
            )?
        };
        if !self.adapters.is_empty() {
            adapters::install_additive(&mut transformer, &self.adapters)?;
        }
        if let Some(quant) = self.quant.filter(|_| !stream_transformer_blocks) {
            transformer.quantize(quant)?;
        }
        let vae = Flux2Vae::new(self.component_vb("vae", VAE_DTYPE)?)?;
        let pid = match self.pid_spec.as_ref() {
            Some(spec) => Some(Arc::new(PidEngine::from_spec(
                spec,
                PID_BACKBONE,
                &self.device,
            )?)),
            None => None,
        };
        Ok(HeavyComponents {
            transformer: Arc::new(transformer),
            vae: Arc::new(vae),
            pid,
        })
    }

    fn encode_one(
        &self,
        comps: &TextComponents,
        prompt: &str,
        date: &str,
        window: Option<usize>,
        cancel: &gen_core::CancelFlag,
    ) -> CResult<(Vec<Tensor>, Tensor)> {
        let ids = comps.tokenizer.encode(prompt, date)?;
        let l = ids.len();
        let input_ids = Tensor::from_vec(ids, (1, l), &self.device)?;
        let layers = comps.encoder.capture_with_window(
            &input_ids,
            &DEFAULT_SELECTED_LAYERS,
            window,
            cancel,
        )?;
        if l > TXT_OFFSET {
            let s = l - TXT_OFFSET;
            let features = layers
                .iter()
                .map(|f| f.narrow(1, TXT_OFFSET, s))
                .collect::<candle_gen::candle_core::Result<Vec<_>>>()?;
            let mask = Tensor::ones((1, s), DType::F32, &self.device)?;
            Ok((features, mask))
        } else {
            let dim = layers[0].dim(2)?;
            let features = (0..DEFAULT_SELECTED_LAYERS.len())
                .map(|_| Tensor::zeros((1, 0, dim), ENC_DTYPE, &self.device))
                .collect::<candle_gen::candle_core::Result<Vec<_>>>()?;
            let mask = Tensor::zeros((1, 0), DType::F32, &self.device)?;
            Ok((features, mask))
        }
    }

    /// Encode positives + negatives and assemble the joint CFG batch: each feature layer is
    /// `[2, S_txt, 2880]` (`[pos; neg]`) and the mask is `[2, S_txt]` (`1` = valid). An empty negative
    /// is the **unconditional branch**: zero text features + an all-zero mask (no text tokens), not a
    /// second encode.
    ///
    /// When `guided` is false (effective guidance `== 1.0`, the `lens_turbo` DEFAULT) the joint batch
    /// collapses to `cond` under [`cfg_rescale`], so the uncond half is neither encoded nor batched —
    /// each layer is `[1, S_txt, 2880]` and the mask `[1, S_txt]` (sc-8993). The denoise loop then runs
    /// a single (batch-1) DiT forward per step instead of two.
    #[allow(clippy::too_many_arguments)]
    fn encode_prompt(
        &self,
        comps: &TextComponents,
        prompt: &str,
        negative: &str,
        date: &str,
        guided: bool,
        window: Option<usize>,
        cancel: &gen_core::CancelFlag,
    ) -> CResult<(Vec<Tensor>, Tensor)> {
        let (pos_feats, pos_mask) = self.encode_one(comps, prompt, date, window, cancel)?;
        if !guided {
            // Guidance disabled: skip the uncond encode/batch entirely; cond-only conditioning.
            let features = pos_feats
                .iter()
                .map(|f| f.to_dtype(DIT_DTYPE))
                .collect::<candle_gen::candle_core::Result<Vec<_>>>()?;
            return Ok((features, pos_mask));
        }
        let s_pos = pos_feats[0].dim(1)?;
        let (neg_feats, neg_mask) = if negative.trim().is_empty() {
            let zeros = pos_feats
                .iter()
                .map(|f| f.zeros_like())
                .collect::<candle_gen::candle_core::Result<Vec<_>>>()?;
            (zeros, pos_mask.zeros_like()?)
        } else {
            self.encode_one(comps, negative, date, window, cancel)?
        };
        let s_neg = neg_feats[0].dim(1)?;

        let target = s_pos.max(s_neg);
        let pos_feats = pad_features(&pos_feats, s_pos, target, &self.device)?;
        let neg_feats = pad_features(&neg_feats, s_neg, target, &self.device)?;
        let pos_mask = pad_mask(&pos_mask, s_pos, target, &self.device)?;
        let neg_mask = pad_mask(&neg_mask, s_neg, target, &self.device)?;

        let mut features = Vec::with_capacity(pos_feats.len());
        for (pf, nf) in pos_feats.iter().zip(neg_feats.iter()) {
            features.push(Tensor::cat(&[pf, nf], 0)?.to_dtype(DIT_DTYPE)?);
        }
        let mask = Tensor::cat(&[&pos_mask, &neg_mask], 0)?;
        Ok((features, mask))
    }

    /// The denoising loop over the joint CFG conditioning + an initial latent
    /// (`[1, latent_h·latent_w, 128]`). Returns the final patch-space latents (feed to [`vae::decode`]).
    ///
    /// Routed through the unified curated sampler/scheduler framework (epic 7114 P4, sc-7123): the
    /// `scheduler` axis picks the σ schedule over the Lens empirical-μ shift (`native` = the legacy
    /// `flow_match` `build_flow_sigmas`), the `sampler` axis picks the integrator. The DEFAULT
    /// (`euler` over the native schedule) is the N1 no-op — algebraically the legacy `euler_step` loop
    /// `x + v·(σ_{i+1} − σ_i)` within the framework's `to_d` round-trip tolerance. Lens feeds the raw
    /// (shifted) sigma as the model timestep (`Sigma` convention) and is standard-guidance, so the CFG
    /// (`cfg_rescale`) lives inside the `predict` closure — a multi-eval solver re-runs the whole closure.
    ///
    /// `preview` is the optional per-step latent preview hook (epic 16948, sc-16955). Lens shares the
    /// FLUX.2 32-channel latent space and packed token layout, so it reuses
    /// [`candle_gen_flux2::preview::hook`] rather than owning a projector — see the module docs there
    /// for why the projection runs after the VAE-owned de-normalize + unpatchify. `None` is
    /// byte-identical to a run without it; the render lanes build a hook per image and the
    /// `denoise_for_parity` seam passes `None` (it has no request, and therefore no sink).
    ///
    /// The hook sees the sampler's running latent, which is the single **conditional** token stream:
    /// the joint `[cond, uncond]` batch is fused inside the predict closure and `cfg_rescale` blends
    /// it back to one velocity before returning, so no unconditional half ever becomes the latent.
    #[allow(clippy::too_many_arguments)]
    fn denoise(
        &self,
        comps: &HeavyComponents,
        features: &[Tensor],
        mask: &Tensor,
        init_latents: &Tensor,
        latent_h: usize,
        latent_w: usize,
        num_steps: usize,
        guidance: f32,
        guided: bool,
        sampler: Option<&str>,
        scheduler: Option<&str>,
        seed: u64,
        cancel: &gen_core::CancelFlag,
        memory: gen_core::GenerationMemory,
        preview: Option<&candle_gen::preview::PreviewHook<'_>>,
        on_progress: &mut dyn FnMut(Progress),
    ) -> CResult<Tensor> {
        let mu = lens_mu(num_steps, latent_h, latent_w);
        let native = lens_sigmas(num_steps, latent_h, latent_w);
        let sigmas = candle_gen::resolve_flow_schedule(scheduler, mu, num_steps, &native);
        let init = init_latents.to_dtype(DIT_DTYPE)?;
        let attention_budget = if memory.chunk_attention {
            let size = memory.attention_chunk_size.ok_or_else(|| {
                CandleError::Msg("lens: bounded attention is missing its chunk size".into())
            })?;
            gen_core::attention_budget::AttentionBudget::from_score_elements(size as u64, false)
        } else {
            gen_core::attention_budget::AttentionBudget::from_score_elements(
                candle_gen::ATTN_SCORES_BUDGET as u64,
                false,
            )
        };
        let attention_plan = gen_core::attention_budget::AttentionPlan::budgeted(attention_budget)
            .with_cancel(cancel);
        // sc-17719 — an all-valid mask makes `build_joint_mask` an all-zero additive term that is then
        // broadcast onto the FULL score matrix `[B, heads, q, k]` in every block, every step. At 2048²
        // that matrix is ~16.4k × 16.4k per head, so adding a known zero to it is the largest piece of
        // pure waste in the denoise. `forward` documents `text_valid: None` as the skip path. After
        // sc-8993's CFG-off gate the all-valid case is the common one (a single unpadded prompt, or a
        // cond-only encode); zeros appear only when the two prompts differ in length and the shorter
        // one is padded. Resolved once here — one host sync per render, never per step.
        let text_valid = if mask.min_all()?.to_scalar::<f32>()? == 1.0 {
            None
        } else {
            Some(mask)
        };
        let transformer_window = if memory.stream_transformer_blocks {
            memory.transformer_window_size.ok_or_else(|| {
                CandleError::Msg("lens: streamed DiT is missing its window size".into())
            })? as usize
        } else {
            TRANSFORMER_BLOCK_COUNT as usize
        };
        candle_gen::run_flow_sampler(
            sampler,
            TimestepConvention::Sigma,
            &sigmas,
            init,
            seed,
            cancel,
            on_progress,
            preview,
            |latents, sigma| -> CResult<Tensor> {
                if !guided {
                    // Guidance disabled: cfg_rescale(cond, ·, 1.0) == cond, so run a single
                    // cond-only (batch-1) forward and skip the wasted uncond half (sc-8993).
                    return comps.transformer.forward_with_memory(
                        latents,
                        features,
                        text_valid,
                        sigma,
                        1,
                        latent_h,
                        latent_w,
                        attention_plan,
                        transformer_window,
                        cancel,
                    );
                }
                // Joint CFG batch: duplicate the latent (cond/uncond share x_t), one DiT call.
                let hidden = Tensor::cat(&[latents, latents], 0)?; // [2, seq, 128]
                let noise = comps.transformer.forward_with_memory(
                    &hidden,
                    features,
                    text_valid,
                    sigma,
                    1,
                    latent_h,
                    latent_w,
                    attention_plan,
                    transformer_window,
                    cancel,
                )?;
                let cond = noise.narrow(0, 0, 1)?;
                let uncond = noise.narrow(0, 1, 1)?;
                Ok(cfg_rescale(&cond, &uncond, guidance)?)
            },
        )
    }

    fn render(
        &self,
        req: &GenerationRequest,
        comps: &Components,
        defaults: Defaults,
        on_progress: &mut dyn FnMut(Progress),
    ) -> CResult<Vec<Image>> {
        let steps = req
            .steps
            .map(|s| s as usize)
            .unwrap_or(defaults.steps as usize);
        let guidance = req.guidance.unwrap_or(defaults.guidance);
        // Standard CFG with the Lens `cfg_rescale`: at guidance == 1.0 the combine reduces exactly to
        // cond, so guidance is effectively off — skip the uncond encode/forward entirely (sc-8993).
        let guided = guidance != 1.0;
        let negative = req.negative_prompt.as_deref().unwrap_or("");
        let base_seed = req.seed.unwrap_or_else(gen_core::default_seed);
        let latent_h = (req.height / VAE_SCALE_FACTOR) as usize;
        let latent_w = (req.width / VAE_SCALE_FACTOR) as usize;

        let (features, mask) = self.encode_prompt(
            &comps.text,
            &req.prompt,
            negative,
            DEFAULT_DATE,
            guided,
            None,
            &req.cancel,
        )?;

        // Resolve the decode seam once for the whole batch (epic 7840 / sc-7853): a per-generation PiD
        // decoder bound to this prompt when `req.use_pid` is set (errors if requested but not loaded),
        // else `None` → the native Flux2Vae decode. Shared across `count` images (same prompt).
        let pid_decoder = candle_gen_pid::resolve_pid_decoder(
            comps.heavy.pid.as_deref(),
            req,
            base_seed,
            defaults.id,
        )?;

        let memory = req.memory.unwrap_or_default();
        candle_gen::for_each_image_seed(base_seed, req.count, |seed| {
            let init = create_noise(seed, latent_h, latent_w, &self.device)?;
            // Per-step latent preview (epic 16948, sc-16955), bound to the same `(latent_h, latent_w)`
            // the decode tail below builds its packed grid from. Built per image so each seed's
            // trajectory starts at frame 1.
            let preview = preview::hook(&req.preview, &comps.heavy.vae, latent_h, latent_w);
            let latents = self.denoise(
                &comps.heavy,
                &features,
                &mask,
                &init,
                latent_h,
                latent_w,
                steps,
                guidance,
                guided,
                req.sampler.as_deref(),
                req.scheduler.as_deref(),
                seed,
                &req.cancel,
                memory,
                Some(&preview),
                on_progress,
            )?;
            on_progress(Progress::Decoding);
            // PiD (super-resolving) decode when the toggle resolved one; else the native VAE. PiD
            // consumes the packed BN-normalized `[1,128,h,w]` latent directly — the *same* packed grid
            // `vae::decode` builds from the DiT output `[1, seq, 128]` (reshape → permute → contiguous),
            // then BN-de-normalizes; here PiD gets that grid before de-normalization. Returns `[1,3,4H,4W]`.
            let decoded = match &pid_decoder {
                Some(pid) => {
                    let (b, _seq, c) = latents.dims3()?;
                    let packed = latents
                        .reshape((b, latent_h, latent_w, c))?
                        .permute((0, 3, 1, 2))?
                        .contiguous()?;
                    pid.decode(&packed)?
                }
                None => vae::decode_with_tiling(
                    &comps.heavy.vae,
                    &latents,
                    latent_h,
                    latent_w,
                    decode_tile(memory, defaults.id)?,
                )?,
            };
            to_image(&decoded)
        })
    }

    fn render_sequential(
        &self,
        req: &GenerationRequest,
        defaults: Defaults,
        stream_text: bool,
        on_progress: &mut dyn FnMut(Progress),
    ) -> CResult<Vec<Image>> {
        // Conditioning is always streamed one GPT-OSS layer at a time on the deferred packed
        // route. The shared rung-4 window belongs to the DiT and is consumed by `denoise` below.
        let window = if stream_text {
            Some(DEFAULT_TEXT_ENCODER_WINDOW)
        } else {
            None
        };

        let steps = req
            .steps
            .map(|s| s as usize)
            .unwrap_or(defaults.steps as usize);
        let guidance = req.guidance.unwrap_or(defaults.guidance);
        let guided = guidance != 1.0;
        let negative = req.negative_prompt.as_deref().unwrap_or("");
        let base_seed = req.seed.unwrap_or_else(gen_core::default_seed);
        let latent_h = (req.height / VAE_SCALE_FACTOR) as usize;
        let latent_w = (req.width / VAE_SCALE_FACTOR) as usize;

        let text = if stream_text {
            self.load_streamable_text_components(&req.cancel)?
        } else {
            self.load_resident_text_components()?
        };
        let (features, mask) = self.encode_prompt(
            &text,
            &req.prompt,
            negative,
            DEFAULT_DATE,
            guided,
            window,
            &req.cancel,
        )?;
        drop(text);
        self.device.synchronize()?;

        let memory = req.memory.unwrap_or_default();
        let stream_transformer_blocks = memory.stream_transformer_blocks;
        let heavy = self.load_heavy_components(stream_transformer_blocks, &req.cancel)?;
        let pid_decoder =
            candle_gen_pid::resolve_pid_decoder(heavy.pid.as_deref(), req, base_seed, defaults.id)?;
        candle_gen::for_each_image_seed(base_seed, req.count, |seed| {
            let init = create_noise(seed, latent_h, latent_w, &self.device)?;
            // Per-step latent preview (epic 16948, sc-16955) — the sequential-residency twin of the
            // resident lane's hook, bound to the same grid this lane decodes against.
            let preview = preview::hook(&req.preview, &heavy.vae, latent_h, latent_w);
            let latents = self.denoise(
                &heavy,
                &features,
                &mask,
                &init,
                latent_h,
                latent_w,
                steps,
                guidance,
                guided,
                req.sampler.as_deref(),
                req.scheduler.as_deref(),
                seed,
                &req.cancel,
                memory,
                Some(&preview),
                on_progress,
            )?;
            on_progress(Progress::Decoding);
            let decoded = match &pid_decoder {
                Some(pid) => {
                    let (b, _seq, c) = latents.dims3()?;
                    let packed = latents
                        .reshape((b, latent_h, latent_w, c))?
                        .permute((0, 3, 1, 2))?
                        .contiguous()?;
                    pid.decode(&packed)?
                }
                None => vae::decode_with_tiling(
                    &heavy.vae,
                    &latents,
                    latent_h,
                    latent_w,
                    decode_tile(memory, defaults.id)?,
                )?,
            };
            to_image(&decoded)
        })
    }
}

fn decode_tile(
    memory: gen_core::GenerationMemory,
    provider_id: &str,
) -> CResult<Option<(u32, u32)>> {
    if !memory.tile_vae_decode {
        return Ok(None);
    }
    let edge = memory.decode_tile_edge.ok_or_else(|| {
        CandleError::Msg(format!(
            "{provider_id}: tiled decode is missing a tile edge"
        ))
    })?;
    let overlap = memory.decode_overlap.ok_or_else(|| {
        CandleError::Msg(format!("{provider_id}: tiled decode is missing an overlap"))
    })?;
    Ok(Some((edge, overlap)))
}

/// Zero-pad each `[B, cur, C]` feature layer along the sequence axis to length `target`.
fn pad_features(
    features: &[Tensor],
    cur: usize,
    target: usize,
    device: &Device,
) -> candle_gen::candle_core::Result<Vec<Tensor>> {
    if cur == target {
        return Ok(features.to_vec());
    }
    let pad = target - cur;
    features
        .iter()
        .map(|f| {
            let (b, _, c) = f.dims3()?;
            let z = Tensor::zeros((b, pad, c), f.dtype(), device)?;
            Tensor::cat(&[f, &z], 1)
        })
        .collect()
}

/// Zero-pad a `[B, cur]` mask along the sequence axis to length `target`.
fn pad_mask(
    mask: &Tensor,
    cur: usize,
    target: usize,
    device: &Device,
) -> candle_gen::candle_core::Result<Tensor> {
    if cur == target {
        return Ok(mask.clone());
    }
    let pad = target - cur;
    let b = mask.dim(0)?;
    let z = Tensor::zeros((b, pad), DType::F32, device)?;
    Tensor::cat(&[mask, &z], 1)
}

/// Deterministic packed initial noise `[1, latent_h·latent_w, 128]` (sc-3673 pattern): N(0,1) from a
/// fixed CPU RNG (NOT candle's CUDA `randn`), then moved to `device`.
fn create_noise(
    seed: u64,
    latent_h: usize,
    latent_w: usize,
    device: &Device,
) -> candle_gen::candle_core::Result<Tensor> {
    let seq = latent_h * latent_w;
    let n = seq * 128;
    let mut rng = StdRng::seed_from_u64(seed);
    let data = candle_gen::seeded_normal_vec(&mut rng, n);
    Tensor::from_vec(data, (1, seq, 128), &Device::Cpu)?.to_device(device)
}

/// Convert a decoded image `[1, 3, H, W]` (NCHW) in `[-1, 1]` to an RGB8 [`Image`].
fn to_image(decoded: &Tensor) -> CResult<Image> {
    let scaled = ((decoded.clamp(-1f32, 1f32)? + 1.0)? * 127.5)?;
    let img = candle_gen::round_rgb8(&scaled)?;
    let img = img.i(0)?.to_device(&Device::Cpu)?;
    let (c, h, w) = img.dims3()?;
    if c != 3 {
        return Err(CandleError::Msg(format!(
            "lens: expected 3 channels, got {c}"
        )));
    }
    let pixels = img.permute((1, 2, 0))?.flatten_all()?.to_vec1::<u8>()?;
    Ok(Image {
        width: w as u32,
        height: h as u32,
        pixels,
    })
}

/// Per-variant sampling defaults (`num_steps`, `guidance_scale`) baked into the loaded generator.
#[derive(Clone, Copy)]
struct Defaults {
    id: &'static str,
    steps: u32,
    guidance: f32,
}

impl Defaults {
    const fn from(id: &'static str, d: LensSamplingDefaults) -> Self {
        Self {
            id,
            steps: d.num_steps as u32,
            guidance: d.guidance_scale,
        }
    }
}

const TURBO_DEFAULTS: Defaults = Defaults::from(MODEL_ID_TURBO, TURBO);
const BASE_DEFAULTS: Defaults = Defaults::from(MODEL_ID_BASE, BASE);

#[cfg(any(feature = "cuda", test))]
fn build_lens_turbo_memory_strategy_contract(spec: &LoadSpec) -> gen_core::MemoryProviderContract {
    build_lens_memory_strategy_contract_with_eligibility(
        MODEL_ID_TURBO,
        spec,
        streams_dit_blocks(spec),
    )
}

#[cfg(test)]
fn build_lens_turbo_memory_strategy_contract_with_eligibility(
    spec: &LoadSpec,
    streamable: bool,
) -> gen_core::MemoryProviderContract {
    build_lens_memory_strategy_contract_with_eligibility(MODEL_ID_TURBO, spec, streamable)
}

#[cfg(any(feature = "cuda", test))]
fn build_lens_memory_strategy_contract(
    provider_id: &'static str,
    spec: &LoadSpec,
) -> gen_core::MemoryProviderContract {
    build_lens_memory_strategy_contract_with_eligibility(
        provider_id,
        spec,
        streams_dit_blocks(spec),
    )
}

#[cfg(any(feature = "cuda", test))]
fn build_lens_memory_strategy_contract_with_eligibility(
    provider_id: &'static str,
    spec: &LoadSpec,
    streamable: bool,
) -> gen_core::MemoryProviderContract {
    use gen_core::{
        MemoryBackendRealization, MemoryFormulaKind, MemoryFormulaVariable,
        MemoryLifecycleCapabilities, MemoryParameterRanges, MemoryPhase, MemoryPrerequisiteScope,
        MemoryProviderContract, MemoryRuntimeSemantics, MemoryStrategy, MemoryStrategyCapability,
        MemoryStrategyPrerequisite, MemoryStrategySupport, MemoryWindowMaterialization,
    };

    let components = gen_core::PerComponentBytes::from_spec_subdirs(
        spec,
        &["text_encoder"],
        &["transformer"],
        &["vae"],
    )
    .unwrap_or_default();
    let phases = vec![
        MemoryPhase::Conditioning,
        MemoryPhase::Denoise,
        MemoryPhase::Decode,
    ];
    let strategies = MemoryStrategy::ALL
        .into_iter()
        .map(|strategy| MemoryStrategyCapability {
            strategy,
            support: if strategy == MemoryStrategy::BoundedTransformerResidency && !streamable {
                MemoryStrategySupport::Missing
            } else {
                MemoryStrategySupport::Implemented
            },
            parameters: match strategy {
                MemoryStrategy::BoundedDecode => MemoryParameterRanges {
                    decode_tile_edges: DECODE_TILE_EDGES.to_vec(),
                    decode_overlaps: vec![DECODE_OVERLAP],
                    ..Default::default()
                },
                MemoryStrategy::BoundedAttention => MemoryParameterRanges {
                    attention_chunk_sizes: vec![ATTENTION_CHUNK_SIZE],
                    ..Default::default()
                },
                MemoryStrategy::BoundedTransformerResidency if streamable => {
                    MemoryParameterRanges {
                        transformer_window_sizes: TRANSFORMER_WINDOW_SIZES.to_vec(),
                        transformer_window_components: vec![gen_core::TransformerComponent::Dit],
                        ..Default::default()
                    }
                }
                _ => MemoryParameterRanges {
                    ..Default::default()
                },
            },
        })
        .collect();

    MemoryProviderContract {
        provider_id: provider_id.to_owned(),
        backend: MemoryBackendRealization::CandleCuda {
            device_residency: true,
            host_backed_weights: true,
            host_to_device_block_materialization: true,
            // Packed q4/q8 is the only load shape for which rung 4 is Implemented. Component-open
            // prepares content-addressed GGML sidecars; a window maps and transfers those exact
            // bytes, with no MLX-affine conversion or device-to-host round trip in the window.
            block_materialization: MemoryWindowMaterialization::DeviceFormatTransfer,
        },
        strategies,
        pid_decode_routes: None,
        load_shape: spec.load_shape,
        additional_prerequisites: [
            MemoryStrategy::BoundedDecode,
            MemoryStrategy::BoundedAttention,
            MemoryStrategy::BoundedTransformerResidency,
        ]
        .into_iter()
        .map(|strategy| {
            (
                strategy,
                MemoryStrategyPrerequisite::Rung {
                    rung: MemoryStrategy::StagedResidency,
                    scope: MemoryPrerequisiteScope::EngagedInSameRequest,
                },
            )
        })
        .collect(),
        default_engagement_exclusions: Vec::new(),
        resident_request_memory: gen_core::ResidentRequestMemory::PreserveLoadDefaults,
        lifecycle: MemoryLifecycleCapabilities {
            phases: phases.clone(),
            synchronized_phase_release: true,
            decode_tiling: true,
            attention_chunking: true,
            transformer_window_materialization: streamable,
        },
        formula: MemoryFormulaKind::PhaseEnvelope {
            phases,
            variables: vec![
                MemoryFormulaVariable::AssetBytes,
                MemoryFormulaVariable::ConditioningTokenCount,
                MemoryFormulaVariable::PixelCount,
                MemoryFormulaVariable::BatchCount,
                MemoryFormulaVariable::OverlayBytes,
                MemoryFormulaVariable::DecodeTileArea,
                MemoryFormulaVariable::AttentionChunkSize,
                MemoryFormulaVariable::TransformerWindowSize,
            ],
        },
        calibration: memory_calibration(spec, streamable),
        asset_facts: gen_core::MemoryAssetFacts {
            base_bytes: components
                .text_encoder
                .saturating_add(components.dit)
                .saturating_add(components.vae),
            conditioning_bytes: components.text_encoder,
            transformer_bytes: components.dit,
            decoder_bytes: components.vae,
            overlay_bytes: 0,
        },
        runtime: MemoryRuntimeSemantics::default(),
    }
}

fn lens_memory_strategy_safety_decision(
    loaded_precision: Precision,
    loaded_quant: Option<Quant>,
    component_precision_floors: &'static [gen_core::ComponentPrecisionFloor],
    contract: &gen_core::MemoryProviderContract,
    context: &gen_core::MemoryRunContext,
) -> gen_core::MemorySafetyDecision {
    gen_core::standard_memory_strategy_safety_check(
        contract,
        context,
        Some(gen_core::MemoryNumericTier {
            precision: loaded_precision,
            quant: loaded_quant,
            component_precision_floors,
        }),
        None,
    )
}

#[cfg(any(feature = "cuda", test))]
struct LensMemoryScope {
    provider_id: &'static str,
    device: Device,
    geometry: gen_core::MemoryGeometry,
    memory: Option<gen_core::GenerationMemory>,
    transformer_window: Option<u32>,
    use_pid: bool,
    finished: bool,
}

#[cfg(test)]
fn lens_generation_memory(
    contract: &gen_core::MemoryProviderContract,
    selection: gen_core::MemorySelection,
) -> Option<gen_core::GenerationMemory> {
    contract.generation_memory(&selection)
}

#[cfg(any(feature = "cuda", test))]
impl LensMemoryScope {
    fn new(
        provider_id: &'static str,
        device: Device,
        contract: &gen_core::MemoryProviderContract,
        context: &gen_core::MemoryRunContext,
    ) -> Self {
        Self {
            provider_id,
            device,
            geometry: context.geometry,
            memory: contract.generation_memory(&context.selection),
            transformer_window: contract
                .engages(
                    context.selection.strategy,
                    gen_core::MemoryStrategy::BoundedTransformerResidency,
                )
                .then_some(context.selection.parameters.transformer_window_size)
                .flatten(),
            use_pid: context.use_pid,
            finished: false,
        }
    }

    fn ensure_active(&self) -> gen_core::Result<()> {
        if self.finished {
            Err(gen_core::Error::Msg(format!(
                "{} memory-strategy request scope is already finished",
                self.provider_id
            )))
        } else {
            Ok(())
        }
    }

    fn validate_geometry(&self, geometry: gen_core::MemoryGeometry) -> gen_core::Result<()> {
        if geometry.width == self.geometry.width
            && geometry.height == self.geometry.height
            && geometry.frames == self.geometry.frames
            && geometry.reference_count == self.geometry.reference_count
            && geometry.batch > 0
            && geometry.batch <= self.geometry.batch
        {
            return Ok(());
        }
        Err(gen_core::Error::Unsupported(format!(
            "{}: hook geometry does not fit the admitted request geometry",
            self.provider_id
        )))
    }
}

#[cfg(any(feature = "cuda", test))]
impl gen_core::MemoryRequestScope for LensMemoryScope {
    fn configure_request(&mut self, request: &mut GenerationRequest) -> gen_core::Result<()> {
        self.ensure_active()?;
        if request.use_pid != self.use_pid
            || !request.conditioning.is_empty()
            || request.phases.is_some()
        {
            return Err(gen_core::Error::Unsupported(format!(
                "{}: optimized memory strategies cover ordinary text-to-image only",
                self.provider_id
            )));
        }
        if request.width != self.geometry.width
            || request.height != self.geometry.height
            || request.count != self.geometry.batch
            || request.image_reference_count() != self.geometry.reference_count
        {
            return Err(gen_core::Error::Unsupported(format!(
                "{}: request geometry changed after memory admission",
                self.provider_id
            )));
        }
        request.memory = self.memory;
        Ok(())
    }

    fn enter_phase(&mut self, _phase: gen_core::MemoryPhase) -> gen_core::Result<()> {
        self.ensure_active()
    }

    fn leave_phase(&mut self, _phase: gen_core::MemoryPhase) -> gen_core::Result<()> {
        self.ensure_active()
    }

    fn configure_decode(
        &mut self,
        tile_edge: u32,
        overlap: u32,
        geometry: gen_core::MemoryGeometry,
    ) -> gen_core::Result<()> {
        self.ensure_active()?;
        self.validate_geometry(geometry)?;
        if self.use_pid {
            return Err(gen_core::Error::Unsupported(format!(
                "{}: PiD does not consume the native VAE tile plan",
                self.provider_id
            )));
        }
        if DECODE_TILE_EDGES.contains(&tile_edge) && overlap == DECODE_OVERLAP {
            Ok(())
        } else {
            Err(gen_core::Error::Unsupported(format!(
                "{}: native decode tiling does not publish {tile_edge}/{overlap}",
                self.provider_id
            )))
        }
    }

    fn configure_attention(&mut self, chunk_size: u32) -> gen_core::Result<()> {
        self.ensure_active()?;
        if chunk_size == ATTENTION_CHUNK_SIZE {
            Ok(())
        } else {
            Err(gen_core::Error::Unsupported(format!(
                "{}: attention chunk size is fixed at {ATTENTION_CHUNK_SIZE}, got {chunk_size}",
                self.provider_id
            )))
        }
    }

    fn materialize_transformer_window(
        &mut self,
        first_block: u32,
        block_count: u32,
    ) -> gen_core::Result<()> {
        self.ensure_active()?;
        let Some(window) = self.transformer_window else {
            return Err(gen_core::Error::Unsupported(format!(
                "{}: bounded transformer residency was not selected",
                self.provider_id
            )));
        };
        if window == 0 || block_count == 0 || !first_block.is_multiple_of(window) {
            return Err(gen_core::Error::Unsupported(format!(
                "{}: invalid transformer window {block_count} at {first_block}",
                self.provider_id
            )));
        }
        if first_block >= TRANSFORMER_BLOCK_COUNT {
            return Err(gen_core::Error::Unsupported(format!(
                "{}: transformer window starts past the {TRANSFORMER_BLOCK_COUNT}-block stack",
                self.provider_id
            )));
        }
        let expected = window.min(TRANSFORMER_BLOCK_COUNT - first_block);
        if block_count == expected {
            Ok(())
        } else {
            Err(gen_core::Error::Unsupported(format!(
                "{}: admitted window {window} requires {expected} blocks at {first_block}, got {block_count}",
                self.provider_id
            )))
        }
    }

    fn finish(&mut self, _outcome: gen_core::MemoryRunOutcome) -> gen_core::Result<()> {
        self.ensure_active()?;
        self.device
            .synchronize()
            .map_err(gen_core::Error::backend)?;
        self.finished = true;
        Ok(())
    }
}

#[cfg(any(feature = "cuda", test))]
impl Drop for LensMemoryScope {
    fn drop(&mut self) {
        if !self.finished {
            let _ = self.device.synchronize();
            self.finished = true;
        }
    }
}

/// A loaded, dispatchable Lens generator: the pipeline + the variant's descriptor & sampling defaults.
/// Components are cached after the first `generate`.
pub struct LensGenerator {
    descriptor: ModelDescriptor,
    defaults: Defaults,
    pipeline: Pipeline,
    components: Mutex<Option<Components>>,
    /// Serializes the manual conditioning/denoise/decode lifecycle and makes cache eviction safe.
    lifecycle: Mutex<()>,
    sequential: bool,
    stream_text: bool,
    stream_dit: bool,
    loaded_precision: Precision,
    loaded_quant: Option<Quant>,
    memory_contract: Option<gen_core::MemoryProviderContract>,
}

impl LensGenerator {
    /// Test/parity constructor: a generator over a snapshot dir with the turbo defaults (lazy
    /// components). The sampling defaults are irrelevant to `denoise_for_parity` (which takes
    /// explicit `steps`/`guidance`); this just gives the e2e gate a concrete generator to drive.
    pub fn for_parity(root: impl AsRef<Path>) -> CResult<Self> {
        let device = candle_gen::default_device()?;
        Ok(Self {
            descriptor: descriptor_turbo(),
            defaults: TURBO_DEFAULTS,
            pipeline: Pipeline::load(root.as_ref(), &device, Vec::new(), None, None),
            components: Mutex::new(None),
            lifecycle: Mutex::new(()),
            sequential: false,
            stream_text: false,
            stream_dit: false,
            loaded_precision: Precision::Bf16,
            loaded_quant: None,
            memory_contract: None,
        })
    }

    fn components(&self) -> gen_core::Result<Components> {
        // `?` bridges the candle-side `load_components` error into `gen_core::Error`.
        Ok(candle_gen::cached(&self.components, || {
            self.pipeline.load_components()
        })?)
    }

    fn execution_mode(&self, req: &GenerationRequest) -> gen_core::Result<(bool, bool)> {
        let stage_residency = req
            .memory
            .as_ref()
            .map(|memory| memory.stage_residency)
            .unwrap_or(self.sequential);
        let stream_dit = req
            .memory
            .as_ref()
            .map(|memory| memory.stream_transformer_blocks)
            .unwrap_or(false);
        let has_bounded_work = req.memory.as_ref().is_some_and(|memory| {
            memory.tile_vae_decode || memory.chunk_attention || memory.stream_transformer_blocks
        });
        if has_bounded_work && !stage_residency {
            return Err(gen_core::Error::Unsupported(format!(
                "{}: bounded memory strategies require staged residency in the same request",
                self.defaults.id
            )));
        }
        if stream_dit && !self.stream_dit {
            return Err(gen_core::Error::Unsupported(format!(
                "{}: this load is not an eligible packed q4/q8 DiT stream",
                self.defaults.id
            )));
        }
        let stream_text = stage_residency && self.stream_text;
        Ok((stage_residency, stream_text))
    }

    fn cache_components_for_request(&self, stage_residency: bool) -> bool {
        !stage_residency && !self.sequential
    }

    /// e2e-parity hook (sc-5115): encode → denoise from **injected** latents → decode, factoring out
    /// the RNG so a cross-build comparison isolates the wiring. Returns the final patch latents
    /// `[1, seq, 128]` and the decoded image `[1, 3, H, W]` in `[-1, 1]`.
    #[allow(clippy::too_many_arguments)]
    pub fn denoise_for_parity(
        &self,
        prompt: &str,
        negative: &str,
        date: &str,
        init_latents: &Tensor,
        latent_h: usize,
        latent_w: usize,
        num_steps: usize,
        guidance: f32,
    ) -> CResult<(Tensor, Tensor)> {
        let comps = self
            .components()
            .map_err(|e| CandleError::Msg(e.to_string()))?;
        // Match render's guidance gate: at guidance == 1.0 the uncond branch is skipped (sc-8993).
        let guided = guidance != 1.0;
        let cancel = gen_core::CancelFlag::new();
        let (features, mask) = self.pipeline.encode_prompt(
            &comps.text,
            prompt,
            negative,
            date,
            guided,
            None,
            &cancel,
        )?;
        // Parity hook drives the default (euler over the native flow_match schedule), no cancel, and
        // no preview: this seam takes injected latents rather than a `GenerationRequest`, so there is
        // no `PreviewSink` to emit into and a frame here would have no consumer.
        let latents = self.pipeline.denoise(
            &comps.heavy,
            &features,
            &mask,
            init_latents,
            latent_h,
            latent_w,
            num_steps,
            guidance,
            guided,
            None,
            None,
            0,
            &cancel,
            gen_core::GenerationMemory::default(),
            None,
            &mut |_| {},
        )?;
        let decoded = vae::decode(&comps.heavy.vae, &latents, latent_h, latent_w)?;
        Ok((latents, decoded))
    }
}

impl Generator for LensGenerator {
    fn descriptor(&self) -> &ModelDescriptor {
        &self.descriptor
    }

    fn memory_strategy_contract(&self) -> Option<&gen_core::MemoryProviderContract> {
        self.memory_contract.as_ref()
    }

    fn memory_strategy_safety_check(
        &self,
        context: &gen_core::MemoryRunContext,
    ) -> gen_core::MemorySafetyDecision {
        let Some(contract) = self.memory_contract.as_ref() else {
            return gen_core::MemorySafetyDecision::Reject {
                reason: format!("{}: no memory-strategy contract", self.defaults.id),
            };
        };
        lens_memory_strategy_safety_decision(
            self.loaded_precision,
            self.loaded_quant,
            self.descriptor.capabilities.component_precision_floors,
            contract,
            context,
        )
    }

    fn begin_memory_strategy_request(
        &self,
        context: &gen_core::MemoryRunContext,
    ) -> gen_core::Result<Option<Box<dyn gen_core::MemoryRequestScope + '_>>> {
        #[cfg(feature = "cuda")]
        {
            let Some(contract) = self.memory_contract.as_ref() else {
                return Ok(None);
            };
            if context.mode != gen_core::MemoryMode::TextToImage
                || context.has_reference
                || context.use_pid
                || context.has_phases
            {
                return Err(gen_core::Error::Unsupported(format!(
                    "{}: optimized memory strategies cover ordinary text-to-image only",
                    self.defaults.id
                )));
            }
            if let gen_core::MemorySafetyDecision::Reject { reason } =
                self.memory_strategy_safety_check(context)
            {
                return Err(gen_core::Error::Unsupported(reason));
            }
            Ok(Some(Box::new(LensMemoryScope::new(
                self.defaults.id,
                self.pipeline.device.clone(),
                contract,
                context,
            ))))
        }
        #[cfg(not(feature = "cuda"))]
        {
            let _ = context;
            Ok(None)
        }
    }

    fn validate(&self, req: &GenerationRequest) -> gen_core::Result<()> {
        validate_request(self.defaults.id, &self.descriptor.capabilities, req)
    }

    fn generate(
        &self,
        req: &GenerationRequest,
        on_progress: &mut dyn FnMut(Progress),
    ) -> gen_core::Result<GenerationOutput> {
        self.validate(req)?;
        let _lifecycle = candle_gen::lock_recover(&self.lifecycle);
        let (stage_residency, stream_text) = self.execution_mode(req)?;
        let images = if stage_residency {
            // A prior resident request may have populated the lazy cache. Drop it before entering
            // the staged phase envelope; the lifecycle lock proves no concurrent request holds a
            // clone of those components while we synchronize and release them.
            let cached = candle_gen::lock_recover(&self.components).take();
            let had_cached = cached.is_some();
            drop(cached);
            if had_cached {
                self.pipeline
                    .device
                    .synchronize()
                    .map_err(CandleError::from)?;
            }
            self.pipeline
                .render_sequential(req, self.defaults, stream_text, on_progress)?
        } else if !self.cache_components_for_request(stage_residency) {
            // A Sequential-loaded generator may be asked for a resident baseline, but that resident
            // stack is request-local. Caching it would leave the full encoder/heavy bundle alive and
            // invalidate a later rung-4 request's cold calibrated bound on the same generator.
            let comps = self.pipeline.load_components()?;
            self.pipeline
                .render(req, &comps, self.defaults, on_progress)?
        } else {
            let comps = self.components()?;
            self.pipeline
                .render(req, &comps, self.defaults, on_progress)?
        };
        Ok(GenerationOutput::Images(images))
    }
}

/// Lens' identity + capabilities for `id` — constructible without loading weights. The norm-rescaled
/// CFG path is always present; turbo simply defaults guidance to 1.0. **Standard guidance, not
/// true-CFG.** LoRA/LoKr are wired (sc-5116, merged into the DiT on load); Q4/Q8 quant is wired for
/// **both** the gpt-oss encoder experts (sc-5111) and the DiT (sc-5117, GGUF `QMatMul` folded in after
/// the merge).
fn descriptor_for(id: &'static str) -> ModelDescriptor {
    ModelDescriptor {
        denoiser_output_latent_space: Some(&candle_gen::gen_core::FLUX2_PACKED_LATENT_SPACE),
        control_kinds: None,
        required_components: &[],
        id,
        family: "lens",
        backend: "candle",
        modality: Modality::Image,
        capabilities: Capabilities {
            supports_negative_prompt: true,
            supports_guidance: true,
            supports_true_cfg: false,
            conditioning: vec![], // pure T2I — no img2img / control / IP in the Lens port
            supports_lora: true,
            supports_lokr: true,
            // Unified curated sampler/scheduler menu (epic 7114 P4, sc-7123) + the legacy aliases
            // (`flow_match_euler`/`flow_match`), which fall back to euler / the native schedule (N3).
            samplers: candle_gen::menu_with_aliases(
                candle_gen::curated_sampler_names(),
                &["flow_match_euler"],
            ),
            schedulers: candle_gen::menu_with_aliases(
                candle_gen::curated_scheduler_names(),
                &["flow_match"],
            ),
            // Buckets span 736..2080 (all ÷16); allow any ÷16 size in a sane range.
            supported_guidance_methods: vec![],
            min_size: 256,
            max_size: 2080,
            max_count: 8,
            mac_only: false,
            supported_quants: &[Quant::Q4, Quant::Q8],
            component_precision_floors: &[],
            supports_kv_cache: false,
            // The Lens schedule computes its own empirical-μ shift internally (not a loader hint).
            requires_sigma_shift: false,
            supports_sequential_offload: true,
            // Per-step latent previews (epic 16948, sc-16955). Lens denoises the FLUX.2 32-channel
            // latent space in the same packed token layout, and loads a VAE whose 250 learned tensors
            // round exactly onto the fit donor's — so both render lanes hand the shared sampler a
            // `candle_gen_flux2::preview` hook and no fit of its own is introduced.
            supports_preview: true,
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

/// Public descriptor accessors (used by the registry submits + tests).
pub fn descriptor_turbo() -> ModelDescriptor {
    descriptor_for(MODEL_ID_TURBO)
}
pub fn descriptor_base() -> ModelDescriptor {
    descriptor_for(MODEL_ID_BASE)
}

/// Capability-driven request validation (unit-testable without loaded weights).
fn validate_request(
    id: &str,
    caps: &Capabilities,
    req: &GenerationRequest,
) -> gen_core::Result<()> {
    caps.validate_request(id, req)?;
    if req.prompt.is_empty() {
        return Err(gen_core::Error::Msg(format!(
            "{id}: prompt must not be empty"
        )));
    }
    if req.steps == Some(0) {
        return Err(gen_core::Error::Msg(format!("{id}: steps must be >= 1")));
    }
    if !req.width.is_multiple_of(VAE_SCALE_FACTOR) || !req.height.is_multiple_of(VAE_SCALE_FACTOR) {
        return Err(gen_core::Error::Msg(format!(
            "{id}: width/height must be multiples of {VAE_SCALE_FACTOR} (got {}x{})",
            req.width, req.height
        )));
    }
    Ok(())
}

fn packed_text_encoder_config(spec: &LoadSpec) -> Option<candle_gen::quant::PackedConfig> {
    let WeightsSource::Dir(root) = &spec.weights else {
        return None;
    };
    let component = root.join("text_encoder");
    let json = std::fs::read_to_string(component.join("config.json")).ok()?;
    let config = serde_json::from_str::<serde_json::Value>(&json).ok()?;
    let packed = candle_gen::quant::PackedConfig::from_config(&config)?;
    let files = candle_gen::sorted_safetensors(&component, "lens").ok()?;
    // SAFETY: read-only model artifacts, mapped only long enough to inspect the immutable headers.
    // This makes eligibility depend on an actual packed triple, not merely a possibly stale config.
    let source = unsafe { MmapedSafetensors::multi(&files).ok()? };
    packed_encoder_inventory_is_exact(
        &source,
        &EncoderConfig::gpt_oss_20b(),
        packed.bits,
        packed.group_size,
    )
    .then_some(packed)
}

fn packed_encoder_inventory_is_exact(
    source: &MmapedSafetensors,
    cfg: &EncoderConfig,
    bits: i32,
    group_size: i32,
) -> bool {
    let Ok(bits) = usize::try_from(bits) else {
        return false;
    };
    let Ok(group_size) = usize::try_from(group_size) else {
        return false;
    };
    if !matches!(bits, 4 | 8) || group_size == 0 || 32 % bits != 0 {
        return false;
    }
    let codes_per_word = 32 / bits;
    for layer in 0..cfg.num_hidden_layers {
        for (projection, out_dim, in_dim) in [
            ("gate_up_proj", 2 * cfg.intermediate_size, cfg.hidden_size),
            ("down_proj", cfg.hidden_size, cfg.intermediate_size),
        ] {
            if !in_dim.is_multiple_of(group_size) || !in_dim.is_multiple_of(codes_per_word) {
                return false;
            }
            let base = format!("model.layers.{layer}.mlp.experts.{projection}");
            let Ok(weight) = source.get(&format!("{base}.weight")) else {
                return false;
            };
            let Ok(scales) = source.get(&format!("{base}.scales")) else {
                return false;
            };
            let Ok(biases) = source.get(&format!("{base}.biases")) else {
                return false;
            };
            if weight.dtype() != safetensors::tensor::Dtype::U32
                || scales.dtype() != safetensors::tensor::Dtype::BF16
                || biases.dtype() != safetensors::tensor::Dtype::BF16
                || weight.shape() != [cfg.num_local_experts, out_dim, in_dim / codes_per_word]
                || scales.shape() != [cfg.num_local_experts, out_dim, in_dim / group_size]
                || biases.shape() != scales.shape()
            {
                return false;
            }
        }
    }
    true
}

fn transformer_numeric_tier_matches(spec: &LoadSpec, expected_bits: usize) -> bool {
    let WeightsSource::Dir(root) = &spec.weights else {
        return false;
    };
    let component = root.join("transformer");
    let Ok(json) = std::fs::read_to_string(component.join("config.json")) else {
        return false;
    };
    let Ok(config) = serde_json::from_str::<serde_json::Value>(&json) else {
        return false;
    };
    let declared = candle_gen::quant::PackedConfig::from_config(&config);
    let Ok(files) = candle_gen::sorted_safetensors(&component, "lens") else {
        return false;
    };
    // SAFETY: read-only model artifacts, mapped only for immutable header inspection.
    let Ok(source) = (unsafe { MmapedSafetensors::multi(&files) }) else {
        return false;
    };
    let mut packed_triples = 0usize;
    let mut u32_weights = 0usize;
    for (name, view) in source.tensors() {
        if name.ends_with(".weight") && view.dtype() == safetensors::tensor::Dtype::U32 {
            u32_weights += 1;
        }
        let Some(base) = name.strip_suffix(".scales") else {
            continue;
        };
        let (Ok(weight), Ok(biases)) = (
            source.get(&format!("{base}.weight")),
            source.get(&format!("{base}.biases")),
        ) else {
            return false;
        };
        packed_triples += 1;
        let Some(packed) = declared else {
            return false;
        };
        let Ok(bits) = usize::try_from(packed.bits) else {
            return false;
        };
        let Ok(group_size) = usize::try_from(packed.group_size) else {
            return false;
        };
        if bits != expected_bits
            || group_size != candle_gen::quant::MLX_GROUP_SIZE
            || weight.dtype() != safetensors::tensor::Dtype::U32
            || view.dtype() != safetensors::tensor::Dtype::BF16
            || biases.dtype() != safetensors::tensor::Dtype::BF16
            || view.shape() != biases.shape()
            || !matches!(weight.shape(), [_, _] | [_, _, _])
            || weight.shape().len() != view.shape().len()
            || weight.shape()[..weight.shape().len() - 1] != view.shape()[..view.shape().len() - 1]
            || weight.shape()[weight.shape().len() - 1] * (32 / bits)
                != view.shape()[view.shape().len() - 1] * group_size
        {
            return false;
        }
    }
    matches!(declared, Some(packed)
        if packed.bits == expected_bits as i32
            && packed.group_size == candle_gen::quant::MLX_GROUP_SIZE as i32
            && packed_triples > 0
            && packed_triples == u32_weights)
}

fn is_plain_measured_load(spec: &LoadSpec) -> bool {
    spec.adapters.is_empty()
        && spec.pid.is_none()
        && spec.control.is_none()
        && spec.extra_controls.is_empty()
        && spec.ip_adapter.is_none()
}

fn streams_text_encoder(spec: &LoadSpec) -> bool {
    let expected_bits = match spec.quantize {
        Some(Quant::Q4) => 4,
        Some(Quant::Q8) => 8,
        _ => return false,
    };
    let Some(packed) = packed_text_encoder_config(spec) else {
        return false;
    };
    matches!(spec.offload_policy, OffloadPolicy::Sequential)
        && matches!(
            spec.load_shape,
            gen_core::LoadShape::DeferredMaterialization
        )
        && spec.precision == Precision::Bf16
        && is_plain_measured_load(spec)
        && packed.bits == expected_bits
        && packed.group_size == candle_gen::quant::MLX_GROUP_SIZE as i32
}

fn streams_dit_blocks(spec: &LoadSpec) -> bool {
    let expected_bits = match spec.quantize {
        Some(Quant::Q4) => 4,
        Some(Quant::Q8) => 8,
        _ => return false,
    };
    streams_text_encoder(spec) && transformer_numeric_tier_matches(spec, expected_bits)
}

#[cfg(any(feature = "cuda", test))]
fn memory_calibration(
    spec: &LoadSpec,
    _streamable: bool,
) -> Option<gen_core::MemoryCalibrationIdentity> {
    // The base resident envelope does not carry typed adapter/PiD component bytes. Refuse calibrated
    // admission for those load shapes until they have their own measured component accounting.
    if !is_plain_measured_load(spec) {
        return None;
    }
    Some(gen_core::MemoryCalibrationIdentity::new(
        MEMORY_CALIBRATION_FINGERPRINT,
        spec.load_shape,
    ))
}

/// Construct a lazy candle Lens generator with the given per-variant defaults. `spec.weights` must be
/// a `microsoft/Lens` / `microsoft/Lens-Turbo` diffusers snapshot dir (`tokenizer/`, `text_encoder/`,
/// `transformer/`, `vae/`). DiT LoRA/LoKr adapters (`spec.adapters`) are merged into the transformer
/// weights on first use (sc-5116). `spec.quantize` (Q4/Q8) transcodes **both** the gpt-oss encoder
/// experts to GGUF `Q4_0`/`Q8_0` (sc-5111; ~13 GB at Q4 vs ~40 GB bf16, the encoder is the memory hog)
/// and the DiT's compute-heavy linears (sc-5117, folded in after the adapter merge). ControlNet /
/// IP-Adapter are not part of the Lens port and are rejected here.
fn load_with(spec: &LoadSpec, defaults: Defaults) -> gen_core::Result<Box<dyn Generator>> {
    let root = match &spec.weights {
        WeightsSource::Dir(p) => p.clone(),
        WeightsSource::File(_) => {
            return Err(gen_core::Error::Msg(format!(
                "{}: expects a Lens snapshot directory (tokenizer/ text_encoder/ transformer/ vae/), \
                 not a single .safetensors file",
                defaults.id
            )));
        }
    };
    // `spec.quantize` (encoder + DiT) and `spec.adapters` (DiT additive install, sc-11105) are both
    // applied downstream in `load_components`, so neither is rejected here.
    if spec.control.is_some() || !spec.extra_controls.is_empty() || spec.ip_adapter.is_some() {
        return Err(gen_core::Error::Unsupported(format!(
            "{}: ControlNet / IP-Adapter conditioning is not part of the Lens port",
            defaults.id
        )));
    }
    let device = candle_gen::default_device()?;
    #[cfg(feature = "cuda")]
    let memory_contract = Some(build_lens_memory_strategy_contract(defaults.id, spec));
    #[cfg(not(feature = "cuda"))]
    let memory_contract = None;
    Ok(Box::new(LensGenerator {
        descriptor: descriptor_for(defaults.id),
        defaults,
        // PiD is an optional aux decoder (epic 7840 / sc-7853): capture the load-spec component (if
        // any) so the lazy component build loads the engine once. Unlike control/IP above, it is not
        // rejected — `None` simply keeps the byte-exact native-VAE path.
        pipeline: Pipeline::load(
            &root,
            &device,
            spec.adapters.clone(),
            spec.quantize,
            spec.pid.clone(),
        ),
        components: Mutex::new(None),
        lifecycle: Mutex::new(()),
        sequential: matches!(spec.offload_policy, OffloadPolicy::Sequential),
        // This provider's text-window implementation is physically coupled to the staged lifecycle:
        // conditioning must finish and release before the heavy phase opens. It is also published
        // only for packed q4/q8, whose post-SC-16096 sidecars make each window a device-format
        // transfer. Sequential+Eager remains a valid rung-1-only path with a resident text phase.
        stream_text: streams_text_encoder(spec),
        stream_dit: streams_dit_blocks(spec),
        loaded_precision: spec.precision,
        loaded_quant: spec.quantize,
        memory_contract,
    }))
}

fn load_turbo(spec: &LoadSpec) -> gen_core::Result<Box<dyn Generator>> {
    load_with(spec, TURBO_DEFAULTS)
}
fn load_base(spec: &LoadSpec) -> gen_core::Result<Box<dyn Generator>> {
    load_with(spec, BASE_DEFAULTS)
}

candle_gen::register_generators! {
    pub(crate) const TURBO_REGISTRATION = descriptor_turbo => load_turbo
}
candle_gen::register_generators! {
    pub(crate) const BASE_REGISTRATION = descriptor_base => load_base
}

#[cfg(feature = "cuda")]
fn registered_lens_turbo_memory_strategy_contract(
    spec: &LoadSpec,
) -> gen_core::Result<gen_core::MemoryProviderContract> {
    Ok(build_lens_turbo_memory_strategy_contract(spec))
}

#[cfg(feature = "cuda")]
fn registered_lens_base_memory_strategy_contract(
    spec: &LoadSpec,
) -> gen_core::Result<gen_core::MemoryProviderContract> {
    Ok(build_lens_memory_strategy_contract(MODEL_ID_BASE, spec))
}

#[cfg(any(feature = "cuda", test))]
fn registered_lens_turbo_memory_strategy_safety_check(
    spec: &LoadSpec,
    contract: &gen_core::MemoryProviderContract,
    context: &gen_core::MemoryRunContext,
) -> gen_core::MemorySafetyDecision {
    lens_memory_strategy_safety_decision(
        spec.precision,
        spec.quantize,
        descriptor_turbo().capabilities.component_precision_floors,
        contract,
        context,
    )
}

#[cfg(any(feature = "cuda", test))]
fn registered_lens_valid_fixture(
    spec: &LoadSpec,
    contract: &gen_core::MemoryProviderContract,
    strategy: gen_core::MemoryStrategy,
) -> gen_core::Result<Vec<gen_core::MemoryBehaviorFixture>> {
    if !strategy.is_optimized() {
        return Ok(Vec::new());
    }
    let context = gen_core::standard_memory_behavior_context(
        contract,
        strategy,
        gen_core::MemoryNumericTier {
            precision: spec.precision,
            quant: spec.quantize,
            component_precision_floors: descriptor_turbo().capabilities.component_precision_floors,
        },
        gen_core::MemoryBehaviorRoute {
            mode: gen_core::MemoryMode::TextToImage,
            reference_count: 0,
            use_pid: false,
            has_phases: false,
            overlay: None,
        },
    )?;
    Ok(vec![gen_core::MemoryBehaviorFixture::new(context)])
}

#[cfg(any(feature = "cuda", test))]
fn registered_lens_begin_request(
    spec: &LoadSpec,
    contract: &gen_core::MemoryProviderContract,
    context: &gen_core::MemoryRunContext,
) -> gen_core::Result<Option<Box<dyn gen_core::MemoryRequestScope>>> {
    if let gen_core::MemorySafetyDecision::Reject { reason } =
        registered_lens_turbo_memory_strategy_safety_check(spec, contract, context)
    {
        return Err(gen_core::Error::Unsupported(reason));
    }
    let provider_id = if contract.provider_id == MODEL_ID_BASE {
        MODEL_ID_BASE
    } else {
        MODEL_ID_TURBO
    };
    Ok(Some(Box::new(LensMemoryScope::new(
        provider_id,
        Device::Cpu,
        contract,
        context,
    ))))
}

#[cfg(test)]
mod weights_free_behavior_tests {
    use super::*;

    #[test]
    fn cpu_scope_executes_the_registered_lens_behavior() {
        let spec = LoadSpec::new(WeightsSource::Dir("/nonexistent/lens".into()))
            .with_offload_policy(candle_gen::gen_core::OffloadPolicy::Sequential)
            .with_load_shape(candle_gen::gen_core::LoadShape::DeferredMaterialization);
        let contract = build_lens_turbo_memory_strategy_contract_with_eligibility(&spec, true);
        let mut fixture = registered_lens_valid_fixture(
            &spec,
            &contract,
            gen_core::MemoryStrategy::BoundedTransformerResidency,
        )
        .unwrap()
        .into_iter()
        .next()
        .unwrap();
        let mut scope = registered_lens_begin_request(&spec, &contract, &fixture.context)
            .unwrap()
            .unwrap();
        scope.configure_request(&mut fixture.request).unwrap();
        assert_eq!(
            fixture.request.memory,
            contract.generation_memory(&fixture.context.selection)
        );
        scope.finish(gen_core::MemoryRunOutcome::Complete).unwrap();
    }
}

#[cfg(feature = "cuda")]
const TURBO_MEMORY_REGISTRATION: gen_core::MemoryRegistration = gen_core::MemoryRegistration {
    provider_id: MODEL_ID_TURBO,
    contract: registered_lens_turbo_memory_strategy_contract,
    safety_check: registered_lens_turbo_memory_strategy_safety_check,
};
#[cfg(feature = "cuda")]
const BASE_MEMORY_REGISTRATION: gen_core::MemoryRegistration = gen_core::MemoryRegistration {
    provider_id: MODEL_ID_BASE,
    contract: registered_lens_base_memory_strategy_contract,
    safety_check: registered_lens_turbo_memory_strategy_safety_check,
};
#[cfg(feature = "cuda")]
const TURBO_MEMORY_BEHAVIOR: gen_core::MemoryBehaviorRegistration =
    gen_core::MemoryBehaviorRegistration {
        provider_id: MODEL_ID_TURBO,
        valid_fixtures: registered_lens_valid_fixture,
        begin_request: registered_lens_begin_request,
    };
#[cfg(feature = "cuda")]
const BASE_MEMORY_BEHAVIOR: gen_core::MemoryBehaviorRegistration =
    gen_core::MemoryBehaviorRegistration {
        provider_id: MODEL_ID_BASE,
        valid_fixtures: registered_lens_valid_fixture,
        begin_request: registered_lens_begin_request,
    };

/// Add all Candle Lens generators and trainers to an explicit media registry builder.
pub fn register_providers(
    registry: candle_gen::gen_core::ProviderRegistryBuilder,
) -> candle_gen::gen_core::ProviderRegistryBuilder {
    let registry = registry
        .register_generator(TURBO_REGISTRATION)
        .register_generator(BASE_REGISTRATION);
    #[cfg(feature = "cuda")]
    let registry = registry
        .register_memory_strategy(TURBO_MEMORY_REGISTRATION)
        .register_memory_behavior(TURBO_MEMORY_BEHAVIOR)
        .register_memory_strategy(BASE_MEMORY_REGISTRATION)
        .register_memory_behavior(BASE_MEMORY_BEHAVIOR);
    registry.register_trainer(training::TRAINER_REGISTRATION)
}

/// Build the complete explicit Candle Lens provider catalog.
pub fn provider_registry() -> candle_gen::gen_core::Result<candle_gen::gen_core::ProviderRegistry> {
    register_providers(candle_gen::gen_core::ProviderRegistryBuilder::new()).build()
}

#[cfg(test)]
mod explicit_registry_tests {
    #[test]
    fn explicit_catalog_has_stable_surface() {
        let registry = super::provider_registry().unwrap();
        let explicit_generators: Vec<String> = registry
            .generators()
            .map(|registration| (registration.descriptor)().id.to_string())
            .collect();
        let explicit_trainers: Vec<String> = registry
            .trainers()
            .map(|registration| (registration.descriptor)().id.to_string())
            .collect();

        assert_eq!(explicit_generators, ["lens_turbo", "lens"]);
        assert_eq!(explicit_trainers, ["lens"]);
    }
}

#[cfg(test)]
mod integration_tests {
    use super::*;

    fn packed_memory_spec(tmp: &tempfile::TempDir, quant: Quant) -> (PathBuf, LoadSpec) {
        let bits = match quant {
            Quant::Q4 => 4,
            Quant::Q8 => 8,
            other => panic!("packed memory fixture does not support {other:?}"),
        };
        let root = tmp.path().join(format!("sc15800_lens_contract_{bits}"));
        let text = root.join("text_encoder");
        std::fs::create_dir_all(&text).unwrap();
        std::fs::write(
            text.join("config.json"),
            format!(r#"{{"quantization": {{"bits": {bits}, "group_size": 64}}}}"#),
        )
        .unwrap();
        let base = "model.layers.0.mlp.experts.gate_up_proj";
        let tensors = std::collections::HashMap::from([
            (
                format!("{base}.weight"),
                Tensor::from_vec(vec![0u32], (1,), &Device::Cpu).unwrap(),
            ),
            (
                format!("{base}.scales"),
                Tensor::from_vec(vec![1f32], (1,), &Device::Cpu).unwrap(),
            ),
            (
                format!("{base}.biases"),
                Tensor::from_vec(vec![0f32], (1,), &Device::Cpu).unwrap(),
            ),
        ]);
        candle_gen::candle_core::safetensors::save(&tensors, text.join("model.safetensors"))
            .unwrap();
        let spec = LoadSpec::new(WeightsSource::Dir(root.clone()))
            .with_quant(quant)
            .with_offload_policy(OffloadPolicy::Sequential)
            .with_load_shape(gen_core::LoadShape::DeferredMaterialization);
        (root, spec)
    }

    fn mini_packed_inventory(tmp: &tempfile::TempDir, complete: bool) -> (PathBuf, EncoderConfig) {
        let root = tmp
            .path()
            .join(format!("sc15800_lens_inventory_{complete}"));
        std::fs::create_dir_all(&root).unwrap();
        let mut cfg = EncoderConfig::gpt_oss_20b();
        cfg.num_hidden_layers = 2;
        cfg.num_local_experts = 2;
        cfg.hidden_size = 64;
        cfg.intermediate_size = 64;
        let mut tensors = std::collections::HashMap::new();
        for layer in 0..cfg.num_hidden_layers {
            for (projection, out_dim, in_dim) in [
                ("gate_up_proj", 2 * cfg.intermediate_size, cfg.hidden_size),
                ("down_proj", cfg.hidden_size, cfg.intermediate_size),
            ] {
                if !complete && layer == cfg.num_hidden_layers - 1 && projection == "down_proj" {
                    continue;
                }
                let base = format!("model.layers.{layer}.mlp.experts.{projection}");
                tensors.insert(
                    format!("{base}.weight"),
                    Tensor::zeros(
                        (cfg.num_local_experts, out_dim, in_dim / 8),
                        DType::U32,
                        &Device::Cpu,
                    )
                    .unwrap(),
                );
                for suffix in ["scales", "biases"] {
                    tensors.insert(
                        format!("{base}.{suffix}"),
                        Tensor::zeros(
                            (cfg.num_local_experts, out_dim, in_dim / 64),
                            DType::BF16,
                            &Device::Cpu,
                        )
                        .unwrap(),
                    );
                }
            }
        }
        candle_gen::candle_core::safetensors::save(&tensors, root.join("model.safetensors"))
            .unwrap();
        (root, cfg)
    }

    #[test]
    fn packed_inventory_requires_both_expert_projections_in_every_layer() {
        let tmp = tempfile::tempdir().unwrap();
        for complete in [true, false] {
            let (root, cfg) = mini_packed_inventory(&tmp, complete);
            let files = vec![root.join("model.safetensors")];
            // SAFETY: immutable test fixture, alive for the duration of this assertion.
            let source = unsafe { MmapedSafetensors::multi(&files).unwrap() };
            assert_eq!(
                packed_encoder_inventory_is_exact(&source, &cfg, 4, 64),
                complete
            );
            std::fs::remove_dir_all(root).ok();
        }
    }

    #[test]
    fn transformer_artifact_must_match_the_selected_numeric_tier() {
        let root_tmp = tempfile::tempdir().unwrap();
        let root = root_tmp.path().to_path_buf();
        let component = root.join("transformer");
        std::fs::create_dir_all(&component).unwrap();
        std::fs::write(
            component.join("config.json"),
            r#"{"quantization": {"bits": 4, "group_size": 64}}"#,
        )
        .unwrap();
        let mut packed = std::collections::HashMap::from([
            (
                "proj.weight".to_owned(),
                Tensor::zeros((4, 8), DType::U32, &Device::Cpu).unwrap(),
            ),
            (
                "proj.scales".to_owned(),
                Tensor::zeros((4, 1), DType::BF16, &Device::Cpu).unwrap(),
            ),
            (
                "proj.biases".to_owned(),
                Tensor::zeros((4, 1), DType::BF16, &Device::Cpu).unwrap(),
            ),
        ]);
        candle_gen::candle_core::safetensors::save(&packed, component.join("model.safetensors"))
            .unwrap();
        let spec = LoadSpec::new(WeightsSource::Dir(root.clone()));
        assert!(transformer_numeric_tier_matches(&spec, 4));
        assert!(!transformer_numeric_tier_matches(&spec, 8));
        packed.insert(
            "orphan.weight".to_owned(),
            Tensor::zeros((4, 8), DType::U32, &Device::Cpu).unwrap(),
        );
        candle_gen::candle_core::safetensors::save(&packed, component.join("model.safetensors"))
            .unwrap();
        assert!(
            !transformer_numeric_tier_matches(&spec, 4),
            "every U32 packed weight must belong to a validated affine triple"
        );

        std::fs::write(component.join("config.json"), r#"{"dtype": "bfloat16"}"#).unwrap();
        assert!(
            !transformer_numeric_tier_matches(&spec, 4),
            "packed tensors with an absent tier declaration must fail closed"
        );
        let dense = std::collections::HashMap::from([(
            "proj.weight".to_owned(),
            Tensor::zeros((4, 64), DType::BF16, &Device::Cpu).unwrap(),
        )]);
        candle_gen::candle_core::safetensors::save(&dense, component.join("model.safetensors"))
            .unwrap();
        assert!(
            !transformer_numeric_tier_matches(&spec, 4),
            "rung 4 requires a packed transformer whose blocks are transfer-ready"
        );
    }

    #[test]
    fn pre_cancelled_streamable_component_open_is_typed_and_does_no_io() {
        let pipeline = Pipeline::load(
            Path::new("/nonexistent/lens"),
            &Device::Cpu,
            Vec::new(),
            Some(Quant::Q4),
            None,
        );
        let cancel = gen_core::CancelFlag::new();
        cancel.cancel();
        assert!(matches!(
            pipeline.load_streamable_text_components(&cancel),
            Err(CandleError::Canceled)
        ));
    }

    #[cfg(feature = "cuda")]
    fn sc15800_quiesce(device: &Device, pool: candle_gen::cuda_mempool::MemPool) -> CResult<()> {
        device.synchronize()?;
        assert!(pool.trim(), "cuMemPoolTrimTo failed");
        assert!(pool.reset_high_water(), "CUDA pool high-water reset failed");
        Ok(())
    }

    #[cfg(feature = "cuda")]
    fn sc15800_cpu_f32(tensors: &[Tensor]) -> CResult<Vec<Vec<f32>>> {
        tensors
            .iter()
            .map(|tensor| -> CResult<Vec<f32>> {
                Ok(tensor
                    .to_dtype(DType::F32)?
                    .flatten_all()?
                    .to_device(&Device::Cpu)?
                    .to_vec1::<f32>()?)
            })
            .collect()
    }

    /// SC-15800's real-weight Candle calibration. This deliberately lives beside the private phase
    /// loaders so it measures the production Lens encoder, tokenizer, sidecar path, and shared
    /// `block_window::run_windowed` driver rather than a look-alike loop.
    ///
    /// Run one tier per process (the CUDA allocator's high-water is process-global):
    ///
    /// ```text
    /// SC15800_LENS_ROOT=<snapshot/q4> SC15800_LENS_QUANT=q4 \
    /// cargo test -p candle-gen-lens --features cuda --release \
    ///   rung4_real_weights_conditioning_window_sweep -- --ignored --nocapture --test-threads=1
    /// ```
    ///
    /// Repeat with the q8 snapshot / `q8`, and the dense snapshot / `dense`. Packed q4/q8 use the
    /// post-SC-16096 device-format sidecars; dense is measured as a control but is not eligible for a
    /// streamed production contract because its MXFP4-to-bf16 conversion would occur per window.
    #[cfg(feature = "cuda")]
    #[test]
    #[ignore = "real Lens weights + CUDA; set SC15800_LENS_ROOT"]
    fn rung4_real_weights_conditioning_window_sweep() -> CResult<()> {
        let Ok(root) = std::env::var("SC15800_LENS_ROOT") else {
            println!("[sc-15800] SKIP: SC15800_LENS_ROOT not set");
            return Ok(());
        };
        let quant_tag = std::env::var("SC15800_LENS_QUANT").unwrap_or_else(|_| "q4".to_owned());
        let quant = match quant_tag.as_str() {
            "q4" => Some(Quant::Q4),
            "q8" => Some(Quant::Q8),
            "dense" => None,
            other => panic!("SC15800_LENS_QUANT must be q4, q8, or dense; got {other}"),
        };
        let device = Device::new_cuda(0)?;
        let pool = candle_gen::cuda_mempool::MemPool::device_default(0)
            .expect("CUDA device 0 default memory pool");
        let (_, total) = candle_gen::cuda_mempool::mem_info()
            .expect("cuMemGetInfo after creating a CUDA context");
        let gib = 1024.0 * 1024.0 * 1024.0;
        println!(
            "[sc-15800] Candle Lens {quant_tag}; host CUDA device 0 {:.2} GiB; root={root}",
            total as f64 / gib
        );

        let pipeline = Pipeline::load(Path::new(&root), &device, Vec::new(), quant, None);
        let prompts = [
            "a red fox",
            "a cinematic portrait of an astronaut botanist in a humid glasshouse, soft window light, detailed leaves, 85mm lens",
            "An intricate editorial photograph of a coastal research station at sunrise, with weathered timber, solar arrays, scientists carrying instrument cases, seabirds above the cliffs, sea mist catching warm light, layered foreground grasses, realistic materials, restrained colors, natural depth, and documentary composition. Preserve fine structural details, legible spatial relationships, and a calm atmospheric horizon.",
        ];

        // Resident references first, so no streamable embedding/norm allocation biases their floor.
        // References move to CPU before the resident encoder is released.
        let resident = pipeline.load_resident_text_components()?;
        assert!(!resident.encoder.is_streamable());
        let mut references = Vec::with_capacity(prompts.len());
        for prompt in prompts {
            sc15800_quiesce(&device, pool)?;
            let (features, mask) = pipeline.encode_prompt(
                &resident,
                prompt,
                "",
                DEFAULT_DATE,
                false,
                None,
                &gen_core::CancelFlag::default(),
            )?;
            device.synchronize()?;
            let live = pool.used_high().expect("USED_MEM_HIGH") as f64 / gib;
            let reserved = pool.reserved_high().expect("RESERVED_MEM_HIGH") as f64 / gib;
            let tokens = mask.dim(1)?;
            println!(
                "[sc-15800] prompt_tokens={tokens:>3} resident: live={live:.3} GiB reserved={reserved:.3} GiB"
            );
            references.push((
                tokens,
                sc15800_cpu_f32(&features)?,
                mask.to_device(&Device::Cpu)?
                    .flatten_all()?
                    .to_vec1::<f32>()?,
                reserved,
            ));
        }
        drop(resident);
        device.synchronize()?;
        assert!(pool.trim());

        // Component-open prepares/reuses content-addressed device-format sidecars once. No source
        // conversion occurs inside any measured window below.
        let streamed = pipeline.load_streamable_text_components(&gen_core::CancelFlag::new())?;
        assert!(streamed.encoder.is_streamable());
        // Dense is a measured control, not a publishable tier: it performs MXFP4-to-bf16 source
        // conversion while opening each window. Resident vs minimum/all-covering is sufficient to
        // show its bound and mutation; the production q4/q8 sidecar tiers receive the full curve.
        let windows: &[usize] = if quant_tag == "dense" {
            &[1, 24]
        } else {
            &[1, 2, 4, 8, 24]
        };
        for (prompt_index, prompt) in prompts.iter().enumerate() {
            let (tokens, ref_features, ref_mask, resident_reserved) = &references[prompt_index];
            let mut rows = Vec::new();
            for &window in windows {
                sc15800_quiesce(&device, pool)?;
                let (features, mask) = pipeline.encode_prompt(
                    &streamed,
                    prompt,
                    "",
                    DEFAULT_DATE,
                    false,
                    Some(window),
                    &gen_core::CancelFlag::default(),
                )?;
                device.synchronize()?;
                let live = pool.used_high().expect("USED_MEM_HIGH") as f64 / gib;
                let reserved = pool.reserved_high().expect("RESERVED_MEM_HIGH") as f64 / gib;
                let got = sc15800_cpu_f32(&features)?;
                let got_mask = mask
                    .to_device(&Device::Cpu)?
                    .flatten_all()?
                    .to_vec1::<f32>()?;
                assert_eq!(got_mask, *ref_mask, "window {window} changed the mask");
                assert_eq!(
                    got, *ref_features,
                    "window {window} changed conditioning bytes for {tokens} tokens"
                );
                println!(
                    "[sc-15800] prompt_tokens={tokens:>3} window={window:>2}: live={live:.3} GiB reserved={reserved:.3} GiB"
                );
                rows.push((window, reserved));
            }
            let w1 = rows[0].1;
            let w24 = rows.last().expect("window 24 row").1;
            assert!(
                w24 > w1 * 1.5,
                "mutation/control failed for {tokens} tokens: all-covering window {w24:.3} GiB did not restore materially more peak than window 1 {w1:.3} GiB"
            );
            assert!(
                resident_reserved > &(w1 * 1.5),
                "window 1 did not materially reduce conditioning peak for {tokens} tokens: resident {resident_reserved:.3} GiB vs {w1:.3} GiB"
            );
        }
        Ok(())
    }

    /// End-to-end companion to the conditioning sweep: the same packed Lens-Turbo request is run
    /// resident and Sequential+Deferred in one process after sidecar preparation. Pixel bytes must
    /// be identical, and the driver-reserved request peak reports whether staging changes the actual
    /// admission-gate envelope rather than merely the encoder sub-phase.
    #[cfg(feature = "cuda")]
    #[test]
    #[ignore = "real Lens weights + CUDA; set SC15800_LENS_ROOT"]
    fn rung4_real_weights_request_peak_and_pixels() -> CResult<()> {
        let Ok(root) = std::env::var("SC15800_LENS_ROOT") else {
            println!("[sc-15800] SKIP: SC15800_LENS_ROOT not set");
            return Ok(());
        };
        let quant_tag = std::env::var("SC15800_LENS_QUANT").unwrap_or_else(|_| "q4".to_owned());
        let quant = match quant_tag.as_str() {
            "q4" => Some(Quant::Q4),
            "q8" => Some(Quant::Q8),
            "dense" => None,
            other => panic!("SC15800_LENS_QUANT must be q4, q8, or dense; got {other}"),
        };
        let device = Device::new_cuda(0)?;
        let pool = candle_gen::cuda_mempool::MemPool::device_default(0)
            .expect("CUDA device 0 default memory pool");
        let gib = 1024.0 * 1024.0 * 1024.0;

        // Keep artifact creation outside both request peaks. Dense has no sidecars, but opening its
        // streamable encoder would perform no layer conversion and remains a valid setup step.
        let pipeline = Pipeline::load(Path::new(&root), &device, Vec::new(), quant, None);
        let prepared = pipeline.load_streamable_text_components(&gen_core::CancelFlag::new())?;
        drop(prepared);
        device.synchronize()?;
        assert!(pool.trim());

        let req = GenerationRequest {
            prompt: "a red fox in soft window light".to_owned(),
            width: 512,
            height: 512,
            count: 1,
            seed: Some(15_800),
            steps: Some(1),
            guidance: Some(1.0),
            ..Default::default()
        };
        let run = |policy: OffloadPolicy,
                   load_shape: gen_core::LoadShape|
         -> CResult<(Image, f64, f64)> {
            let mut spec = LoadSpec::new(WeightsSource::Dir(PathBuf::from(&root)))
                .with_offload_policy(policy)
                .with_load_shape(load_shape);
            if let Some(quant) = quant {
                spec = spec.with_quant(quant);
            }
            let generator = provider_registry()?.load(MODEL_ID_TURBO, &spec)?;
            sc15800_quiesce(&device, pool)?;
            let output = generator.generate(&req, &mut |_| {})?;
            device.synchronize()?;
            let live = pool.used_high().expect("USED_MEM_HIGH") as f64 / gib;
            let reserved = pool.reserved_high().expect("RESERVED_MEM_HIGH") as f64 / gib;
            let image = match output {
                GenerationOutput::Images(mut images) if images.len() == 1 => images.remove(0),
                GenerationOutput::Images(images) => {
                    return Err(CandleError::Msg(format!(
                        "expected one image, got {}",
                        images.len()
                    )))
                }
                GenerationOutput::Video { .. } | GenerationOutput::Audio(_) => {
                    return Err(CandleError::Msg("expected image output".to_owned()))
                }
            };
            drop(generator);
            device.synchronize()?;
            assert!(pool.trim());
            Ok((image, live, reserved))
        };

        let (resident, resident_live, resident_reserved) = run(
            OffloadPolicy::Resident,
            gen_core::LoadShape::EagerMaterialization,
        )?;
        let (staged, staged_live, staged_reserved) = run(
            OffloadPolicy::Sequential,
            gen_core::LoadShape::DeferredMaterialization,
        )?;
        assert_eq!(
            (staged.width, staged.height),
            (resident.width, resident.height)
        );
        assert_eq!(
            staged.pixels, resident.pixels,
            "Sequential+Deferred changed Lens-Turbo pixels"
        );
        println!(
            "[sc-15800] request {quant_tag} 512x512/1-step: resident live={resident_live:.3} GiB reserved={resident_reserved:.3} GiB; sequential-window1 live={staged_live:.3} GiB reserved={staged_reserved:.3} GiB; reserved change={:.1}%",
            100.0 * (staged_reserved / resident_reserved - 1.0)
        );
        Ok(())
    }

    /// SC-15819 authoritative serial smoke: one production request per cumulative ladder rung on a
    /// single q4 Lens-Turbo artifact. This is implementation evidence only; catalog calibration and
    /// promotion remain owned by the entry-level stories.
    #[cfg(feature = "cuda")]
    #[test]
    #[ignore = "real Lens q4 weights + CUDA; set SC15819_LENS_ROOT"]
    fn sc15819_real_weights_five_rung_sequence() -> CResult<()> {
        let Ok(root) = std::env::var("SC15819_LENS_ROOT") else {
            println!("[sc-15819] SKIP: SC15819_LENS_ROOT not set");
            return Ok(());
        };
        let device = Device::new_cuda(0)?;
        let pool = candle_gen::cuda_mempool::MemPool::device_default(0)
            .expect("CUDA device 0 default memory pool");
        let gib = 1024.0 * 1024.0 * 1024.0;
        let optimized_spec = || {
            LoadSpec::new(WeightsSource::Dir(PathBuf::from(&root)))
                .with_quant(Quant::Q4)
                .with_offload_policy(OffloadPolicy::Sequential)
                .with_load_shape(gen_core::LoadShape::DeferredMaterialization)
        };
        let contract = build_lens_memory_strategy_contract(MODEL_ID_TURBO, &optimized_spec());
        assert!(contract.conformance_errors().is_empty());
        assert_eq!(
            contract
                .calibration
                .as_ref()
                .map(|value| value.fingerprint.as_str()),
            Some(MEMORY_CALIBRATION_FINGERPRINT)
        );

        // Prepare content-addressed sidecars outside every measured request.
        let prep = Pipeline::load(Path::new(&root), &device, Vec::new(), Some(Quant::Q4), None);
        drop(prep.load_streamable_text_components(&gen_core::CancelFlag::new())?);
        drop(prep.load_heavy_components(true, &gen_core::CancelFlag::new())?);
        sc15800_quiesce(&device, pool)?;

        let staged = gen_core::GenerationMemory {
            stage_residency: true,
            ..Default::default()
        };
        let decode = gen_core::GenerationMemory {
            tile_vae_decode: true,
            decode_tile_edge: Some(DECODE_TILE_EDGE),
            decode_overlap: Some(DECODE_OVERLAP),
            ..staged
        };
        let attention = gen_core::GenerationMemory {
            chunk_attention: true,
            attention_chunk_size: Some(ATTENTION_CHUNK_SIZE),
            ..decode
        };
        let transformer = gen_core::GenerationMemory {
            stream_transformer_blocks: true,
            transformer_window_size: Some(4),
            transformer_window_component: Some(gen_core::TransformerComponent::Dit),
            ..attention
        };
        let rows = [
            (
                "resident",
                OffloadPolicy::Resident,
                gen_core::LoadShape::EagerMaterialization,
                None,
            ),
            (
                "staged",
                OffloadPolicy::Sequential,
                gen_core::LoadShape::DeferredMaterialization,
                Some(staged),
            ),
            (
                "decode",
                OffloadPolicy::Sequential,
                gen_core::LoadShape::DeferredMaterialization,
                Some(decode),
            ),
            (
                "attention",
                OffloadPolicy::Sequential,
                gen_core::LoadShape::DeferredMaterialization,
                Some(attention),
            ),
            (
                "transformer",
                OffloadPolicy::Sequential,
                gen_core::LoadShape::DeferredMaterialization,
                Some(transformer),
            ),
        ];
        let mut images = Vec::new();
        for (label, policy, shape, memory) in rows {
            let spec = LoadSpec::new(WeightsSource::Dir(PathBuf::from(&root)))
                .with_quant(Quant::Q4)
                .with_offload_policy(policy)
                .with_load_shape(shape);
            let generator = provider_registry()?.load(MODEL_ID_TURBO, &spec)?;
            let req = GenerationRequest {
                prompt: "a red fox in soft window light".to_owned(),
                width: 1024,
                height: 1024,
                count: 1,
                seed: Some(15_819),
                steps: Some(1),
                guidance: Some(1.0),
                memory,
                ..Default::default()
            };
            sc15800_quiesce(&device, pool)?;
            let output = generator.generate(&req, &mut |_| {})?;
            device.synchronize()?;
            let live = pool.used_high().expect("USED_MEM_HIGH") as f64 / gib;
            let reserved = pool.reserved_high().expect("RESERVED_MEM_HIGH") as f64 / gib;
            let image = match output {
                GenerationOutput::Images(mut values) if values.len() == 1 => values.remove(0),
                _ => return Err(CandleError::Msg("expected one image".to_owned())),
            };
            let checksum = image.pixels.iter().fold(0_u64, |sum, value| {
                sum.wrapping_mul(16777619) ^ u64::from(*value)
            });
            println!(
                "[sc-15819] {label}: live={live:.3} GiB reserved={reserved:.3} GiB checksum={checksum:016x}"
            );
            assert_eq!((image.width, image.height), (1024, 1024));
            assert!(!image.pixels.is_empty());
            images.push((label, image));
            drop(generator);
            sc15800_quiesce(&device, pool)?;
        }
        assert_eq!(
            images[0].1.pixels, images[1].1.pixels,
            "staging changed pixels"
        );
        assert_eq!(
            images[2].1.pixels, images[3].1.pixels,
            "attention changed tiled pixels"
        );
        assert_eq!(
            images[3].1.pixels, images[4].1.pixels,
            "DiT windows changed pixels"
        );

        let generator = provider_registry()?.load(MODEL_ID_TURBO, &optimized_spec())?;
        let canceled = GenerationRequest {
            prompt: "cancel cleanup".to_owned(),
            width: 1024,
            height: 1024,
            count: 1,
            memory: Some(transformer),
            ..Default::default()
        };
        canceled.cancel.cancel();
        assert!(matches!(
            generator.generate(&canceled, &mut |_| {}),
            Err(gen_core::Error::Canceled)
        ));
        let invalid = GenerationRequest {
            prompt: "invalid prerequisite cleanup".to_owned(),
            memory: Some(gen_core::GenerationMemory {
                chunk_attention: true,
                attention_chunk_size: Some(ATTENTION_CHUNK_SIZE),
                ..Default::default()
            }),
            ..Default::default()
        };
        assert!(generator.generate(&invalid, &mut |_| {}).is_err());
        drop(generator);
        sc15800_quiesce(&device, pool)?;
        Ok(())
    }

    #[test]
    fn descriptors_are_lens() {
        for (d, id, steps, g) in [
            (descriptor_turbo(), MODEL_ID_TURBO, 4u32, 1.0f32),
            (descriptor_base(), MODEL_ID_BASE, 20, 5.0),
        ] {
            assert_eq!(d.id, id);
            assert_eq!(d.family, "lens");
            assert_eq!(d.backend, "candle");
            assert_eq!(d.modality, Modality::Image);
            assert!(d.capabilities.supports_guidance);
            assert!(d.capabilities.supports_negative_prompt);
            assert!(!d.capabilities.supports_true_cfg);
            assert!(d.capabilities.conditioning.is_empty());
            assert!(!d.capabilities.mac_only);
            let def = if id == MODEL_ID_TURBO {
                TURBO_DEFAULTS
            } else {
                BASE_DEFAULTS
            };
            assert_eq!((def.steps, def.guidance), (steps, g));
        }
    }

    /// **The parsed packed `group_size` is threaded into a LOUD guard, not discarded** (sc-9474). A
    /// `transformer/` (or `text_encoder/`) `config.json` carrying `quantization: { bits, group_size }`
    /// parses to its on-disk group size; the guard passes for the MLX default 64 (every hosted
    /// `SceneWorks/lens-mlx` tier) and errors for a group-32 tier rather than silently repacking u32 codes
    /// to garbage through the group-64 shared loaders. A dense/absent config skips the guard.
    #[test]
    fn packed_group_size_guard_rejects_non_default() {
        let root_tmp = tempfile::tempdir().unwrap();
        let root = root_tmp.path().to_path_buf();
        let sub_dir = root.join("transformer");
        std::fs::create_dir_all(&sub_dir).unwrap();
        let write_cfg = |json: &str| std::fs::write(sub_dir.join("config.json"), json).unwrap();
        let pipe = || Pipeline::load(&root, &Device::Cpu, Vec::new(), Some(Quant::Q4), None);

        // group-64 (the MLX default): the parsed group size survives and the guard passes.
        write_cfg(r#"{"quantization": {"bits": 4, "group_size": 64}}"#);
        assert_eq!(
            pipe().packed_group_size("transformer"),
            Some(candle_gen::quant::MLX_GROUP_SIZE as i32),
            "parsed group_size must be threaded, not discarded"
        );
        assert!(
            pipe().guard_packed_group_size("transformer").is_ok(),
            "group-64 (the default) must pass the guard"
        );

        // group-32 (boogu's group size): the guard fails LOUD instead of silently repacking to garbage.
        write_cfg(r#"{"quantization": {"bits": 4, "group_size": 32}}"#);
        assert_eq!(pipe().packed_group_size("transformer"), Some(32));
        assert!(
            pipe().guard_packed_group_size("transformer").is_err(),
            "a group-32 tier must be rejected LOUD, not silently repacked (sc-9474)"
        );

        // A dense config (no `quantization`) ⇒ None ⇒ the guard is skipped.
        write_cfg(r#"{"in_channels": 128}"#);
        assert!(pipe().packed_group_size("transformer").is_none());
        assert!(pipe().guard_packed_group_size("transformer").is_ok());

        // An absent config dir ⇒ None ⇒ skipped (a dense snapshot with no packed config still loads).
        assert!(pipe().packed_group_size("text_encoder").is_none());
        assert!(pipe().guard_packed_group_size("text_encoder").is_ok());
    }

    #[test]
    fn both_ids_resolve_in_registry() {
        // The family catalog resolves both ids. Loading is **lazy** (weights are read on first
        // `generate`), so construction succeeds even with a bogus directory.
        for id in [MODEL_ID_TURBO, MODEL_ID_BASE] {
            let spec = LoadSpec::new(WeightsSource::Dir("/nonexistent/lens".into()));
            assert!(
                crate::provider_registry().unwrap().load(id, &spec).is_ok(),
                "{id} should resolve + lazily construct in the registry"
            );
        }
    }

    #[test]
    fn text_window_requires_sequential_deferred_packed_load() {
        let tmp = tempfile::tempdir().unwrap();
        let base = || LoadSpec::new(WeightsSource::Dir("/nonexistent/lens".into()));
        assert!(!streams_text_encoder(&base()));
        assert!(!streams_text_encoder(
            &base()
                .with_quant(Quant::Q4)
                .with_load_shape(gen_core::LoadShape::DeferredMaterialization)
        ));
        assert!(!streams_text_encoder(
            &base()
                .with_quant(Quant::Q4)
                .with_offload_policy(OffloadPolicy::Sequential)
        ));
        assert!(!streams_text_encoder(
            &base()
                .with_offload_policy(OffloadPolicy::Sequential)
                .with_load_shape(gen_core::LoadShape::DeferredMaterialization)
        ));
        for quant in [Quant::Q4, Quant::Q8] {
            let (root, spec) = packed_memory_spec(&tmp, quant);
            assert!(
                !streams_text_encoder(&spec),
                "a config plus one pseudo-triple must not advertise a 24-layer transfer-only encoder"
            );
            let wrong_quant = if quant == Quant::Q4 {
                Quant::Q8
            } else {
                Quant::Q4
            };
            assert!(
                !streams_text_encoder(&spec.clone().with_quant(wrong_quant)),
                "the requested quant must match the on-disk packed tier"
            );
            let mut adapted = spec.clone();
            adapted.adapters.push(AdapterSpec::new(
                root.join("adapter.safetensors"),
                1.0,
                gen_core::AdapterKind::Lora,
            ));
            assert!(!streams_text_encoder(&adapted));
            std::fs::remove_dir_all(root).ok();
        }
    }

    #[test]
    fn memory_contract_is_load_exact_and_dit_scoped() {
        let tmp = tempfile::tempdir().unwrap();
        use gen_core::{LoadShape, MemoryStrategy, MemoryStrategySupport, TransformerComponent};

        let base = || LoadSpec::new(WeightsSource::Dir("/nonexistent/lens".into()));
        let (eligible_root, eligible) = packed_memory_spec(&tmp, Quant::Q4);
        let contract = build_lens_turbo_memory_strategy_contract_with_eligibility(&eligible, true);
        let _detected_contract = build_lens_memory_strategy_contract(MODEL_ID_BASE, &eligible);
        let base_contract =
            build_lens_memory_strategy_contract_with_eligibility(MODEL_ID_BASE, &eligible, true);
        assert_eq!(base_contract.provider_id, MODEL_ID_BASE);
        assert!(base_contract.conformance_errors().is_empty());
        gen_core_testkit::check_memory_strategy_contract(&contract).unwrap();
        assert!(matches!(
            contract
                .capability(MemoryStrategy::StagedResidency)
                .unwrap()
                .support,
            MemoryStrategySupport::Implemented
        ));
        let bounded = contract
            .capability(MemoryStrategy::BoundedTransformerResidency)
            .unwrap();
        assert!(matches!(
            bounded.support,
            MemoryStrategySupport::Implemented
        ));
        assert_eq!(
            bounded.parameters.transformer_window_sizes,
            TRANSFORMER_WINDOW_SIZES
        );
        assert_eq!(
            bounded.parameters.transformer_window_components,
            [TransformerComponent::Dit]
        );
        assert_eq!(
            contract
                .capability(MemoryStrategy::BoundedDecode)
                .unwrap()
                .parameters
                .decode_tile_edges,
            DECODE_TILE_EDGES
        );
        assert_eq!(
            contract
                .capability(MemoryStrategy::BoundedAttention)
                .unwrap()
                .parameters
                .attention_chunk_sizes,
            [ATTENTION_CHUNK_SIZE]
        );
        assert!(contract.calibration.is_some());
        assert_eq!(
            contract.calibration.as_ref().unwrap().fingerprint,
            MEMORY_CALIBRATION_FINGERPRINT
        );
        let (q8_root, q8_spec) = packed_memory_spec(&tmp, Quant::Q8);
        let q8_contract =
            build_lens_turbo_memory_strategy_contract_with_eligibility(&q8_spec, true);
        assert_eq!(contract.calibration, q8_contract.calibration);
        let mut adapted = eligible.clone();
        adapted.adapters.push(AdapterSpec::new(
            eligible_root.join("adapter.safetensors"),
            1.0,
            gen_core::AdapterKind::Lora,
        ));
        assert!(
            build_lens_turbo_memory_strategy_contract(&adapted)
                .calibration
                .is_none(),
            "unaccounted auxiliary resident bytes must fail closed"
        );

        for ineligible in [
            base().with_quant(Quant::Q4),
            base()
                .with_quant(Quant::Q4)
                .with_offload_policy(OffloadPolicy::Sequential),
            base()
                .with_offload_policy(OffloadPolicy::Sequential)
                .with_load_shape(LoadShape::DeferredMaterialization),
        ] {
            let contract = build_lens_turbo_memory_strategy_contract(&ineligible);
            gen_core_testkit::check_memory_strategy_contract(&contract).unwrap();
            assert!(matches!(
                contract
                    .capability(MemoryStrategy::BoundedTransformerResidency)
                    .unwrap()
                    .support,
                MemoryStrategySupport::Missing
            ));
            assert_eq!(
                contract.calibration.as_ref().unwrap().fingerprint,
                MEMORY_CALIBRATION_FINGERPRINT
            );
        }
        std::fs::remove_dir_all(eligible_root).ok();
        std::fs::remove_dir_all(q8_root).ok();
    }

    #[test]
    fn safety_check_rejects_stale_calibration_and_wrong_numeric_tier() {
        let tmp = tempfile::tempdir().unwrap();
        let (root, spec) = packed_memory_spec(&tmp, Quant::Q4);
        let contract = build_lens_turbo_memory_strategy_contract_with_eligibility(&spec, true);
        let calibration = contract.calibration.clone().unwrap();
        let generator = LensGenerator {
            descriptor: descriptor_turbo(),
            defaults: TURBO_DEFAULTS,
            pipeline: Pipeline::load(&root, &Device::Cpu, Vec::new(), Some(Quant::Q4), None),
            components: Mutex::new(None),
            lifecycle: Mutex::new(()),
            sequential: true,
            stream_text: true,
            stream_dit: true,
            loaded_precision: Precision::Bf16,
            loaded_quant: Some(Quant::Q4),
            memory_contract: Some(contract),
        };
        let mut context = gen_core::MemoryRunContext {
            selection: gen_core::MemorySelection {
                strategy: gen_core::MemoryStrategy::BoundedTransformerResidency,
                parameters: gen_core::MemoryStrategyParameters {
                    decode_tile_edge: Some(DECODE_TILE_EDGE),
                    decode_overlap: Some(DECODE_OVERLAP),
                    attention_chunk_size: Some(ATTENTION_CHUNK_SIZE),
                    transformer_window_size: Some(1),
                    transformer_window_component: Some(gen_core::TransformerComponent::Dit),
                },
                tier: gen_core::MemoryNumericTier {
                    precision: Precision::Bf16,
                    quant: Some(Quant::Q4),
                    component_precision_floors: &[],
                },
            },
            calibration_abi: calibration.abi,
            calibration_fingerprint: calibration.fingerprint.clone(),
            load_shape: calibration.load_shape,
            mode: gen_core::MemoryMode::TextToImage,
            has_reference: false,
            use_pid: false,
            has_phases: false,
            geometry: gen_core::MemoryGeometry {
                width: 512,
                height: 512,
                batch: 1,
                frames: 1,
                reference_count: 0,
            },
            overlay: None,
            budget: gen_core::MemoryBudget {
                total_bytes: u64::MAX,
                committed_bytes: 0,
                reclaimable_bytes: 0,
                reserved_headroom_bytes: 0,
            },
            predicted_peak_bytes: 1,
            cache_state: gen_core::MemoryCacheState::Cold,
            evidence_revision: "test".to_owned(),
        };
        assert_eq!(
            generator.memory_strategy_safety_check(&context),
            gen_core::MemorySafetyDecision::Accept
        );
        context.calibration_fingerprint = "stale".to_owned();
        assert!(matches!(
            generator.memory_strategy_safety_check(&context),
            gen_core::MemorySafetyDecision::Reject { .. }
        ));
        context.calibration_fingerprint = calibration.fingerprint;
        context.selection.tier.quant = Some(Quant::Q8);
        assert!(matches!(
            generator.memory_strategy_safety_check(&context),
            gen_core::MemorySafetyDecision::Reject { .. }
        ));
        std::fs::remove_dir_all(root).ok();
    }

    #[test]
    fn selected_dit_window_reaches_the_request_scope() {
        let tmp = tempfile::tempdir().unwrap();
        use gen_core::MemoryRequestScope;

        let (root, spec) = packed_memory_spec(&tmp, Quant::Q4);
        let contract = build_lens_turbo_memory_strategy_contract_with_eligibility(&spec, true);
        let tier = gen_core::MemoryNumericTier {
            precision: gen_core::Precision::Bf16,
            quant: Some(Quant::Q4),
            component_precision_floors: &[],
        };
        let select = |strategy, parameters| gen_core::MemorySelection {
            strategy,
            parameters,
            tier,
        };
        assert_eq!(
            lens_generation_memory(
                &contract,
                select(
                    gen_core::MemoryStrategy::Resident,
                    gen_core::MemoryStrategyParameters::default()
                )
            ),
            None
        );
        assert_eq!(
            lens_generation_memory(
                &contract,
                select(
                    gen_core::MemoryStrategy::StagedResidency,
                    gen_core::MemoryStrategyParameters::default()
                )
            ),
            Some(gen_core::GenerationMemory {
                stage_residency: true,
                ..Default::default()
            })
        );
        let selection = select(
            gen_core::MemoryStrategy::BoundedTransformerResidency,
            gen_core::MemoryStrategyParameters {
                decode_tile_edge: Some(DECODE_TILE_EDGE),
                decode_overlap: Some(DECODE_OVERLAP),
                attention_chunk_size: Some(ATTENTION_CHUNK_SIZE),
                transformer_window_size: Some(4),
                transformer_window_component: Some(gen_core::TransformerComponent::Dit),
            },
        );
        let memory = lens_generation_memory(&contract, selection).unwrap();
        assert!(memory.stage_residency);
        assert!(memory.stream_transformer_blocks);
        assert_eq!(memory.decode_tile_edge, Some(DECODE_TILE_EDGE));
        assert_eq!(memory.attention_chunk_size, Some(ATTENTION_CHUNK_SIZE));
        assert_eq!(memory.transformer_window_size, Some(4));
        assert_eq!(
            memory.transformer_window_component,
            Some(gen_core::TransformerComponent::Dit)
        );

        let mut request = GenerationRequest::default();
        let mut scope = LensMemoryScope {
            provider_id: MODEL_ID_TURBO,
            device: Device::Cpu,
            geometry: gen_core::MemoryGeometry {
                width: request.width,
                height: request.height,
                batch: request.count,
                frames: 1,
                reference_count: 0,
            },
            memory: Some(memory),
            transformer_window: Some(4),
            use_pid: false,
            finished: false,
        };
        scope.configure_request(&mut request).unwrap();
        assert_eq!(request.memory, Some(memory));
        scope.materialize_transformer_window(0, 4).unwrap();
        scope.materialize_transformer_window(44, 4).unwrap();
        assert!(scope.materialize_transformer_window(0, 2).is_err());
        assert!(scope.materialize_transformer_window(48, 1).is_err());
        scope.finish(gen_core::MemoryRunOutcome::Complete).unwrap();
        std::fs::remove_dir_all(root).ok();
    }

    #[test]
    fn request_memory_overrides_load_defaults_without_cross_run_leakage() {
        let generator = LensGenerator {
            descriptor: descriptor_turbo(),
            defaults: TURBO_DEFAULTS,
            pipeline: Pipeline::load(
                Path::new("/nonexistent/lens"),
                &Device::Cpu,
                Vec::new(),
                Some(Quant::Q4),
                None,
            ),
            components: Mutex::new(None),
            lifecycle: Mutex::new(()),
            sequential: true,
            stream_text: true,
            stream_dit: true,
            loaded_precision: Precision::Bf16,
            loaded_quant: Some(Quant::Q4),
            memory_contract: None,
        };
        let resolved = |memory: Option<gen_core::GenerationMemory>| {
            generator
                .execution_mode(&GenerationRequest {
                    memory,
                    ..Default::default()
                })
                .unwrap()
        };
        assert_eq!(resolved(None), (true, true));
        assert_eq!(
            resolved(Some(gen_core::GenerationMemory::default())),
            (false, false)
        );
        assert_eq!(
            resolved(Some(gen_core::GenerationMemory {
                stage_residency: true,
                ..Default::default()
            })),
            (true, true)
        );
        for memory in [
            gen_core::GenerationMemory {
                tile_vae_decode: true,
                decode_tile_edge: Some(DECODE_TILE_EDGE),
                decode_overlap: Some(DECODE_OVERLAP),
                ..Default::default()
            },
            gen_core::GenerationMemory {
                chunk_attention: true,
                attention_chunk_size: Some(ATTENTION_CHUNK_SIZE),
                ..Default::default()
            },
        ] {
            assert!(generator
                .execution_mode(&GenerationRequest {
                    memory: Some(memory),
                    ..Default::default()
                })
                .is_err());
        }
        assert_eq!(resolved(None), (true, true));
        assert!(
            !generator.cache_components_for_request(false),
            "a resident baseline on a Sequential-loaded generator must stay request-local"
        );
        assert!(candle_gen::lock_recover(&generator.components).is_none());

        let ineligible = LensGenerator {
            stream_text: false,
            stream_dit: false,
            ..generator
        };
        assert!(ineligible
            .execution_mode(&GenerationRequest {
                memory: Some(gen_core::GenerationMemory {
                    stage_residency: true,
                    stream_transformer_blocks: true,
                    transformer_window_size: Some(1),
                    transformer_window_component: Some(gen_core::TransformerComponent::Dit),
                    ..Default::default()
                }),
                ..Default::default()
            })
            .is_err());
    }

    #[test]
    fn validate_rejects_bad_inputs() {
        let caps = descriptor_turbo().capabilities;
        let ok = GenerationRequest {
            prompt: "a fox".into(),
            width: 1024,
            height: 1024,
            ..Default::default()
        };
        assert!(validate_request(MODEL_ID_TURBO, &caps, &ok).is_ok());
        let empty = GenerationRequest {
            prompt: "".into(),
            ..ok.clone()
        };
        assert!(validate_request(MODEL_ID_TURBO, &caps, &empty).is_err());
        let bad_dims = GenerationRequest {
            width: 1000,
            ..ok.clone()
        };
        assert!(validate_request(MODEL_ID_TURBO, &caps, &bad_dims).is_err());

        // sc-12612: `VAE_SCALE_FACTOR` is the pinned stride SceneWorks ties every advertised Lens
        // image bucket to. Pin the value and mutation-check that a size which is a multiple of 8 (a
        // lower divisor) but not VAE_SCALE_FACTOR (16) is still rejected with the stride error, and
        // an on-stride in-range size passes.
        assert_eq!(VAE_SCALE_FACTOR, 16);
        let off_stride = validate_request(
            MODEL_ID_TURBO,
            &caps,
            &GenerationRequest {
                width: 1000, // 125×8 — a multiple of 8 but not VAE_SCALE_FACTOR
                ..ok.clone()
            },
        )
        .unwrap_err()
        .to_string();
        assert!(
            off_stride.contains("multiples of 16"),
            "expected the stride error, got: {off_stride}"
        );
        assert!(validate_request(
            MODEL_ID_TURBO,
            &caps,
            &GenerationRequest {
                width: 1024, // 64×16 — on-stride
                ..ok.clone()
            }
        )
        .is_ok());
    }
}
