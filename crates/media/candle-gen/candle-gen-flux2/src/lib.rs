//! # candle-gen-flux2
//!
//! The **FLUX.2** provider crate for [`candle-gen`](candle_gen) — the candle (Windows/CUDA) sibling of
//! `mlx-gen-flux2`. Unlike FLUX.1 (sc-3694), FLUX.2 has **no** `candle-transformers` reference: the
//! MMDiT transformer ([`transformer`]), the 32-channel 2×2-patchify VAE ([`vae`]), the decoder-LM
//! prompt-embeds text path ([`text_encoder`]), the 4-axis RoPE ([`pos_embed`]) and the flow-match
//! geometry ([`pipeline`]) are all ported here from the macOS provider.
//!
//! **Two txt2img variants** are registered, selected by [`config::Flux2Variant`]:
//! - **`flux2_klein_9b`** (sc-3695): Qwen3 TE (hidden states 9/18/27 → 12288-wide `prompt_embeds`) →
//!   the MMDiT (8 joint + 24 fused-single blocks) → the AutoencoderKL-Flux2 decoder. Distilled
//!   **4-step** flow-match Euler, CFG-free at guidance 1.0 (>1 runs a classifier-free negative pass).
//! - **`flux2_dev`** (epic 6564 sc-7457): the 32B flagship. **Mistral** TE (layers 10/20/30 →
//!   15360-wide `prompt_embeds`) → a wider/deeper MMDiT (8 joint + **48** single blocks, **48** heads,
//!   joint 15360). Guidance-**distilled** (embedded scalar, the FLUX.1-dev pattern): ~28 steps at
//!   guidance ~4 via a single forward feeding the DiT's guidance embedder — **not** true CFG.
//!
//! Same deterministic CPU-seeded-noise contract (sc-3673). Tokenization reuses gen-core's
//! [`TextTokenizer`]: klein with [`ChatTemplate::QwenInstructNoThink`], dev with
//! [`ChatTemplate::Flux2DevMistral`].
//!
//! **Sampling (epic 7114 P4, sc-7123):** both denoise loops (txt2img `Pipeline::render` and the edit
//! path [`Flux2Edit`]) route through the unified curated sampler/scheduler driver
//! (`candle_gen::run_flow_sampler` / `resolve_flow_schedule`). FLUX.2 is a rectified-flow engine using
//! the `Sigma` convention but embeds σ×1000, so the predict closure feeds `sigma * 1000.0` to the
//! transformer; the klein guidance>1 CFG blend / the dev embedded-guidance scalar (and, on the edit
//! path, the joint `[target, refs]` concat) live inside that closure. The descriptor advertises the
//! curated sampler/scheduler menus; the default (unset sampler/scheduler) path is the N1 no-op — euler
//! over the native empirical-mu flow-match schedule.
//!
//! **Surface:** txt2img for both variants (gen-core-registered). Conditioned dev surfaces are bespoke,
//! worker-invoked-by-name providers (the candle pattern, NOT registry entries): klein reference edit
//! [`Flux2Edit`] (sc-5487) — extended to **dev** multi-reference edit (sc-7460) via the DiT token
//! concat with the embedded-guidance forward — and dev strict-pose ControlNet [`Flux2Control`]
//! (sc-7460), the `FLUX.2-dev-Fun-Controlnet-Union` VACE branch. The dev conditioned paths run the
//! CPU-stage → quantize-onto-GPU loader ([`quant`]) so the 32B fits the memory ceiling. Still not
//! wired: the klein weight-variant edits (`flux2_klein_9b_kv_edit`) and LoRA/LoKr. `backend =
//! "candle"`, `mac_only = false`.

pub mod config;
pub mod control_provider;
pub mod convert;
pub mod edit_provider;
#[cfg_attr(not(any(feature = "cuda", test)), allow(dead_code))]
pub mod memory_strategy;
pub mod pipeline;
pub mod pos_embed;
pub mod preview;
pub mod quant;
pub mod text_encoder;
pub mod transformer;
pub mod vae;

/// Re-export the pinned width/height stride at the crate root so SceneWorks can tie each advertised
/// FLUX.2 image bucket to `candle_gen_flux2::SIZE_MULTIPLE` (sc-12612) instead of a hand-copied literal.
pub use config::SIZE_MULTIPLE;
pub use control_provider::{Flux2Control, Flux2ControlPaths, Flux2ControlRequest};
pub use convert::convert_and_assemble;
pub use edit_provider::{Flux2Edit, Flux2EditPaths, Flux2EditRequest};
pub use transformer::{
    Flux2ControlBranch, Flux2ControlTransformer, Flux2Transformer, CONTROL_IN_DIM,
};

/// Content identity for the CUDA resident/staged real-weight calibration harness.
pub const RESIDENCY_CALIBRATION_FINGERPRINT: &str = "flux2-cuda-residency-v1";

use std::path::{Path, PathBuf};
use std::sync::Arc;

use candle_gen::candle_core::{DType, Device, IndexOp, Tensor};
use candle_gen::candle_nn::VarBuilder;
use candle_gen::gen_core::attention_budget::{AttentionBudget, AttentionPlan};
use candle_gen::gen_core::sampling::TimestepConvention;
use candle_gen::gen_core::tokenizer::{ChatTemplate, TextTokenizer, TokenizerConfig};
use candle_gen::gen_core::{
    self, Capabilities, GenerationOutput, GenerationRequest, Generator, Image, LoadSpec, Modality,
    ModelDescriptor, PidWeights, Progress, Quant, SizeFloor, WeightsSource,
};
use candle_gen::{CandleError, LatentDecoder, Result as CResult};
use candle_gen_pid::{PidDecoder, PidEngine};

use config::{Flux2Config, Flux2Variant};
use text_encoder::Flux2PromptEncoder;
use vae::Flux2Vae;

/// The PiD backbone (latent-space) tag for FLUX.2 (epic 7840 / sc-7853): the `flux2` student consumes
/// the packed 128-channel BN-normalized latent at H/16 directly (the same tensor `decode_packed`
/// BN-de-normalizes). Lens reuses this same latent space (it shares the FLUX.2 VAE).
const PID_BACKBONE: &str = "flux2";

/// Qwen3 `<|endoftext|>` pad token id (klein FLUX.2 text encoder).
const QWEN_PAD_TOKEN_ID: i32 = 151643;
/// Mistral `<pad>` pad token id (dev FLUX.2 text encoder).
const MISTRAL_PAD_TOKEN_ID: i32 = 11;

/// The loaded FLUX.2 components, `Arc`-shared so the generator caches them across `generate` calls.
#[derive(Clone)]
struct Components {
    te: Arc<Flux2PromptEncoder>,
    transformer: Arc<Flux2Transformer>,
    vae: Arc<Flux2Vae>,
    /// Tokenizer (variant-specific pad token + chat template), loaded+parsed **once** at component load
    /// and reused across every prompt/branch encode (sc-8991 / F-011) instead of re-parsing
    /// `tokenizer.json` per request.
    tokenizer: Arc<TextTokenizer>,
    /// Optional NVIDIA PiD super-resolving decoder (epic 7840 / sc-7853), loaded once when the model
    /// was loaded with `LoadSpec::pid`. `None` ⇒ the native `Flux2Vae::decode_packed` (the default path).
    pid: Option<Arc<PidEngine>>,
}

/// The just-loaded heavy phase owned by the sequential path — the DiT + VAE + the optional PiD engine,
/// loaded together AFTER the text encoder was dropped so they reuse that freed pool. Bundled into one
/// value because it is the `Heavy` of [`candle_gen::run_sequential`] (sc-12089), which loads the phase
/// through a single closure. Not `Arc`-shared: the sequential path deliberately drops each component
/// after its phase rather than keeping the cross-request cache.
struct SeqHeavy {
    transformer: Flux2Transformer,
    vae: Flux2Vae,
    /// The optional PiD engine — `None` both when the caller never opted in via `LoadSpec::pid` and when
    /// THIS request will not decode through it (F-177, [`Pipeline::pid_to_load`]).
    pid: Option<Arc<PidEngine>>,
}

enum TextPhase {
    Resident(Components),
    Sequential(Box<(Flux2PromptEncoder, TextTokenizer)>),
}

enum HeavyPhase {
    Resident(Components),
    Sequential(Box<SeqHeavy>),
}

type Flux2Residency = candle_gen::Residency<TextPhase, HeavyPhase>;

/// A txt2img pipeline handle: snapshot root + device + the f32 compute dtype. `pub(crate)` so the
/// edit provider ([`edit_provider`]) reuses the snapshot mmap + prompt-encode scaffolding.
#[derive(Clone)]
pub(crate) struct Pipeline {
    pub(crate) variant: Flux2Variant,
    pub(crate) cfg: Flux2Config,
    pub(crate) root: PathBuf,
    pub(crate) device: Device,
    pub(crate) dtype: DType,
    /// When `Some`, the DiT (and, for dev, the TE) is staged dense in CPU RAM and quantized onto
    /// `device`. dev folds both TE + DiT; klein folds ONLY the DiT and keeps its Qwen3 TE dense
    /// (`te_quant`). `None` ⇒ everything dense (bf16 tier / fixtures).
    pub(crate) quant: Option<Quant>,
    /// The `LoadSpec::pid` component captured at load (epic 7840 / sc-7853), threaded into the lazy
    /// component build so the PiD engine loads once alongside the base model. `None` when not opted in.
    pub(crate) pid_spec: Option<PidWeights>,
    /// An in-place ComfyUI FLUX.2-dev fp8-mixed DiT single-file (epic 10451 Phase 2e, sc-10680). When
    /// `Some`, the transformer is built from this file (inline-scale fp8 dequant + BFL→diffusers remap,
    /// see [`convert::build_comfyui_dit_map`]) instead of the snapshot's `transformer/` dir; the text
    /// encoder / VAE / tokenizer still come from the resident snapshot `root`. `None` on every other
    /// path (registry txt2img, edit, control).
    pub(crate) comfyui_dit: Option<PathBuf>,
}

impl Pipeline {
    pub(crate) fn load(
        variant: Flux2Variant,
        quant: Option<Quant>,
        root: &Path,
        device: &Device,
        pid_spec: Option<PidWeights>,
    ) -> Self {
        Self {
            variant,
            cfg: variant.config(),
            root: root.to_path_buf(),
            device: device.clone(),
            // FLUX.2 runs the reference math in f32 (the TE + the MMDiT). The weights are large but
            // the math is parity-sensitive; a bf16 pass is a follow-up optimization.
            dtype: DType::F32,
            quant,
            pid_spec,
            comfyui_dit: None,
        }
    }

    /// Same as [`load`](Self::load) but sourcing the DiT from an in-place ComfyUI FLUX.2-dev fp8-mixed
    /// single-file (sc-10680). `root` is the resident FLUX.2-dev diffusers snapshot supplying the Mistral
    /// text encoder / VAE / tokenizer (the single DiT file carries none of those). `quant` (Q4/Q8) is
    /// honored for the DiT the same way the resident dev path is — the 32B does not fit the GPU dense
    /// after the fp8→f32 dequant, so each projection is folded onto the GPU. PiD is not wired here.
    pub(crate) fn load_comfyui(
        quant: Option<Quant>,
        root: &Path,
        device: &Device,
        comfyui_dit: PathBuf,
    ) -> Self {
        Self {
            variant: Flux2Variant::Dev,
            cfg: Flux2Variant::Dev.config(),
            root: root.to_path_buf(),
            device: device.clone(),
            dtype: DType::F32,
            quant,
            pid_spec: None,
            comfyui_dit: Some(comfyui_dit),
        }
    }

    /// mmap a VarBuilder over every `.safetensors` in the snapshot subdir `sub`, on `self.device`.
    pub(crate) fn component_vb(&self, sub: &str) -> CResult<VarBuilder<'static>> {
        self.component_vb_on(sub, &self.device)
    }

    /// [`Self::component_vb`] but on an explicit `device` — the quant path stages the TE + DiT on the
    /// CPU (system RAM) before quantizing onto the GPU, so the dense 32B never lands on the GPU.
    pub(crate) fn component_vb_on(
        &self,
        sub: &str,
        device: &Device,
    ) -> CResult<VarBuilder<'static>> {
        candle_gen::component_vb(&self.root, sub, self.dtype, device, "flux2")
    }

    /// Whether the snapshot component `sub/` is a **pre-quantized MLX-packed tier** — its `config.json`
    /// carries a `quantization` block (`candle_gen::quant::PackedConfig`), which an install-time convert
    /// job writes for a packed component. On a packed tier the loader builds each Linear/embedding
    /// **directly from the packed parts** on the GPU (sc-9087, no dense CPU staging); on a dense tier it
    /// falls back to the CPU-stage → quantize-onto-GPU path.
    ///
    /// A **genuinely-absent** `config.json` (file NotFound) is a legitimate dense/fixture snapshot shape
    /// → `Ok(false)` (dense path), so a single-file fixture with no `config.json` still loads. A config
    /// that **is present but corrupt** (I/O error, malformed JSON) is a damaged/partial download and
    /// errors loudly naming the file, rather than silently downgrading a packed component to the dense
    /// path (wrong tier / missing weights, no diagnostic). A well-formed config with no `quantization`
    /// block is simply a dense tier → `Ok(false)`. Mirrors the F-073 fix (sc-9010) in qwen-edit / krea.
    pub(crate) fn component_is_packed(&self, sub: &str) -> CResult<bool> {
        let path = self.root.join(sub).join("config.json");
        let text = match std::fs::read_to_string(&path) {
            Ok(t) => t,
            // No config.json at all → legitimate dense/fixture snapshot, not packed.
            Err(ref e) if e.kind() == std::io::ErrorKind::NotFound => return Ok(false),
            // Present but unreadable (permissions, partial download) → surface, don't swallow.
            Err(e) => {
                return Err(CandleError::Msg(format!(
                    "flux2: read {}: {e}",
                    path.display()
                )))
            }
        };
        // Present but malformed JSON → corrupt snapshot, error rather than fall to dense.
        let v: serde_json::Value = serde_json::from_str(&text).map_err(|e| {
            CandleError::Msg(format!(
                "flux2: parse {} (corrupt snapshot?): {e}",
                path.display()
            ))
        })?;
        Ok(candle_gen::quant::PackedConfig::from_config(&v).is_some())
    }

    /// The quant applied to the **text encoder** (the DiT always uses `self.quant`). dev folds its ~24B
    /// Mistral TE onto the GPU alongside the 32B DiT — the pair doesn't fit dense — so the TE quantizes
    /// with the DiT. klein keeps its 8B Qwen3 TE DENSE bf16 in EVERY tier (epic 8506 DENSE_TE: only the
    /// 9B DiT quantizes, for fidelity), so the TE is never quantized even at q4/q8 (sc-11031). Returns
    /// `None` for klein regardless of `self.quant`; `self.quant` for dev.
    fn te_quant(&self) -> Option<Quant> {
        if self.variant.is_dev() {
            self.quant
        } else {
            None
        }
    }

    /// Load the base **Mistral/Qwen3 TE + `Flux2Transformer` DiT** pair — the exact quantizable stack
    /// shared by every entry point (txt2img [`Self::load_components`], `Flux2Edit::load_variant`,
    /// `Flux2Control::load`). This is the single home for the "which builders + which tier/staging
    /// strategy" decision (F-024, sc-9004): it fixes the default builders (`Flux2PromptEncoder::new` /
    /// `Flux2Transformer::new`) and delegates the packed-vs-dense-vs-quant routing to
    /// [`Self::load_quantizable`]. Callers layer their extra components on top (the edit/control VAE
    /// *with encoder*, the control-branch overlay) — those are the genuine per-site differences and stay
    /// at the call site; only the copy-pasted TE+DiT loader moves here.
    ///
    /// A staging-strategy change (e.g. pre-quantized snapshot consumption) now lives in one place. Use
    /// [`Self::load_quantizable`] directly only if a future caller needs non-default module builders.
    pub(crate) fn load_te_and_dit(&self) -> CResult<(Flux2PromptEncoder, Flux2Transformer)> {
        self.load_quantizable(
            |cfg, vb| Ok(Flux2PromptEncoder::new(cfg, vb)?),
            |cfg, vb| Ok(Flux2Transformer::new(cfg, vb)?),
        )
    }

    /// Load ONLY the text encoder for the sequential-residency path (epic 10765 Phase 1c, sc-10868) —
    /// dropped right after the prompt encode so the decoder-LM TE (Mistral 24B on dev, Qwen3 on klein)
    /// frees before the DiT loads. Same per-tier routing (packed / dense+quant / dense) as the paired
    /// [`load_te_and_dit`](Self::load_te_and_dit) TE half; the DiT half is loaded separately (and later)
    /// via [`load_dit_seq`](Self::load_dit_seq).
    pub(crate) fn load_te_seq(&self) -> CResult<Flux2PromptEncoder> {
        self.load_one_quantizable(
            "text_encoder",
            self.te_quant(),
            |vb| Ok(Flux2PromptEncoder::new(&self.cfg, vb)?),
            |m, q, d| Ok(m.quantize(q, d)?),
        )
    }

    /// Load ONLY the DiT for the sequential path (sc-10868) — loaded after the text encoder was dropped,
    /// so it reuses the TE's freed allocator pool (capping peak at DiT+VAE, not TE+DiT+VAE). Same per-tier
    /// routing as the paired [`load_te_and_dit`](Self::load_te_and_dit) DiT half.
    pub(crate) fn load_dit_seq(&self) -> CResult<Flux2Transformer> {
        match &self.comfyui_dit {
            Some(dit_file) => self.load_comfyui_dit(dit_file),
            None => self.load_one_quantizable(
                "transformer",
                self.quant,
                |vb| Ok(Flux2Transformer::new(&self.cfg, vb)?),
                |m, q, d| Ok(m.quantize(q, d)?),
            ),
        }
    }

    /// Which PiD spec [`load_pid`](Self::load_pid) should actually load: the spec the caller opted into
    /// via `LoadSpec::pid`, but only when this load will use it (F-177).
    ///
    /// [`resolve_pid_decoder`](candle_gen_pid::resolve_pid_decoder) already gates the *decode* on
    /// `req.use_pid`, so an engine loaded for a request that did not ask for it is never read — under
    /// `Resident` that is a harmless one-time cost amortized across every later request, but under
    /// `Sequential` it is paid on EVERY generate and sits resident through the whole denoise, inside the
    /// very peak that path exists to bound.
    ///
    /// Pure, so the rule is unit-testable without weights or a GPU (krea's `pid_to_load` idiom).
    fn pid_to_load(&self, use_pid: bool) -> Option<&PidWeights> {
        self.pid_spec.as_ref().filter(|_| use_pid)
    }

    /// Load the optional PiD super-resolving decoder (epic 7840 / sc-7853) when the caller opted in via
    /// `LoadSpec::pid` AND this load will actually use it ([`pid_to_load`](Self::pid_to_load)); FLUX.2's
    /// `flux2` latent-space student. `None` ⇒ the native [`Flux2Vae`] decode.
    ///
    /// **`use_pid` (F-177).** [`load_components`](Self::load_components) passes `true` — the resident set
    /// is cached across requests, so the overlay must be there for whichever later request wants it. The
    /// `Sequential` path passes `req.use_pid`, because there this load runs on EVERY generate.
    fn load_pid(&self, use_pid: bool) -> CResult<Option<Arc<PidEngine>>> {
        Ok(match self.pid_to_load(use_pid) {
            Some(spec) => Some(Arc::new(PidEngine::from_spec(
                spec,
                PID_BACKBONE,
                &self.device,
            )?)),
            None => None,
        })
    }

    /// Load the whole heavy phase for the sequential path (sc-12089) — the DiT, then the VAE, then the
    /// optional PiD engine, in that order (the order the pre-seam code loaded them, kept so the tier
    /// routing and any load-time error surface identically). Runs AFTER the text encoder was dropped, so
    /// it reuses that freed allocator pool.
    ///
    /// **`use_pid` (F-177).** Threaded straight to [`load_pid`](Self::load_pid): this whole fn runs per
    /// generate, so a PiD engine loaded for a request that never asked for it would sit inside the peak
    /// this path exists to bound — while `resolve_pid_decoder` goes on to return `None` for it, so not a
    /// byte of it is read.
    fn load_heavy_seq(&self, use_pid: bool) -> CResult<SeqHeavy> {
        Ok(SeqHeavy {
            transformer: self.load_dit_seq()?,
            vae: Flux2Vae::new(self.component_vb("vae")?)?,
            pid: self.load_pid(use_pid)?,
        })
    }

    pub(crate) fn load_dit_seq_with_memory(
        &self,
        stream_transformer_blocks: bool,
    ) -> CResult<Flux2Transformer> {
        if !stream_transformer_blocks {
            return self.load_dit_seq();
        }
        if self.comfyui_dit.is_some() {
            return Err(CandleError::Msg(
                "flux2_dev: streamed blocks require a directory-backed transformer tier".to_owned(),
            ));
        }
        let packed = self.component_is_packed("transformer")?;
        let source_device = if self.quant.is_some() && !packed {
            Device::Cpu
        } else {
            self.device.clone()
        };
        Ok(Flux2Transformer::new_block_streamed(
            &self.cfg,
            self.component_vb_on("transformer", &source_device)?,
            self.quant,
            self.device.clone(),
        )?)
    }

    fn load_heavy_seq_with_memory(
        &self,
        use_pid: bool,
        stream_transformer_blocks: bool,
        bounded_host_decode: bool,
        cancel: &gen_core::CancelFlag,
    ) -> CResult<SeqHeavy> {
        if !stream_transformer_blocks && !bounded_host_decode {
            return self.load_heavy_seq(use_pid);
        }
        candle_gen::check_cancel(cancel)?;
        let transformer = self.load_dit_seq_with_memory(stream_transformer_blocks)?;
        candle_gen::check_cancel(cancel)?;
        let vae_device = if bounded_host_decode {
            Device::Cpu
        } else {
            self.device.clone()
        };
        let vae = Flux2Vae::new(self.component_vb_on("vae", &vae_device)?)?;
        candle_gen::check_cancel(cancel)?;
        Ok(SeqHeavy {
            transformer,
            vae,
            pid: self.load_pid(use_pid)?,
        })
    }

    /// Load the TE + DiT, routing each through the **packed** path (build straight from an MLX-packed
    /// tier on the GPU — sc-9087, no ~105 GB dense CPU staging) or the legacy **dense** path (stage
    /// dense in system RAM, then quantize each projection onto the GPU) per [`Self::component_is_packed`]
    /// and `self.quant`. Shared by txt2img, `Flux2Edit::load_dev`, and `Flux2Control` (they load the same
    /// quantizable pair; the callers add the VAE / control overlay) via [`Self::load_te_and_dit`], which
    /// fixes the default builders. `mk_te` / `mk_dit` build the module from a component VarBuilder
    /// (`Flux2PromptEncoder::new` / `Flux2Transformer::new`).
    pub(crate) fn load_quantizable(
        &self,
        mk_te: impl Fn(&Flux2Config, VarBuilder<'static>) -> CResult<Flux2PromptEncoder>,
        mk_dit: impl Fn(&Flux2Config, VarBuilder<'static>) -> CResult<Flux2Transformer>,
    ) -> CResult<(Flux2PromptEncoder, Flux2Transformer)> {
        let te = self.load_one_quantizable(
            "text_encoder",
            self.te_quant(),
            |vb| mk_te(&self.cfg, vb),
            |m, q, d| Ok(m.quantize(q, d)?),
        )?;
        let dit = self.load_one_quantizable(
            "transformer",
            self.quant,
            |vb| mk_dit(&self.cfg, vb),
            |m, q, d| Ok(m.quantize(q, d)?),
        )?;
        Ok((te, dit))
    }

    /// Load one quantizable component (`sub`) with its OWN `quant` (per-component: `self.quant` for the
    /// DiT, `self.te_quant()` for the TE — klein passes `None` there to keep the Qwen3 TE dense while its
    /// DiT quantizes). Three regimes:
    /// - **packed tier + quant**: build directly on the GPU from the packed parts (`.scales` detected
    ///   inside each `linear_detect`); no dense weight is ever materialized (sc-9087). The post-load
    ///   `quantize` pass is still called — it is a no-op on the already-packed projections and only
    ///   carries the dense leaves (RMSNorms, a dense token embedding) to the GPU.
    /// - **dense tier + quant**: stage dense in CPU RAM, then `quantize` folds each projection onto the
    ///   GPU (the legacy ~105 GB path, retained for dense tiers / large fixtures — and klein's on-the-fly
    ///   DiT quant off a dense BFL snapshot, sc-11031).
    /// - **no quant**: load dense on-device (klein's Qwen3 TE, small dev fixtures).
    fn load_one_quantizable<M>(
        &self,
        sub: &str,
        quant: Option<Quant>,
        build: impl FnOnce(VarBuilder<'static>) -> CResult<M>,
        quantize: impl FnOnce(&mut M, Quant, &Device) -> CResult<()>,
    ) -> CResult<M> {
        match quant {
            Some(q) if self.component_is_packed(sub)? => {
                // Build straight on the GPU from the packed tier — the packed footprint (≈ Q4: ¼ bf16)
                // lands directly; no dense staging.
                let mut m = build(self.component_vb_on(sub, &self.device)?)?;
                // No-op on the packed projections; moves the dense leaves to the GPU.
                quantize(&mut m, q, &self.device)?;
                Ok(m)
            }
            Some(q) => {
                // Dense tier: stage dense in CPU RAM, then quantize each projection onto the GPU.
                let mut m = build(self.component_vb_on(sub, &Device::Cpu)?)?;
                quantize(&mut m, q, &self.device)?;
                Ok(m)
            }
            None => build(self.component_vb(sub)?),
        }
    }

    /// Build the DiT from an in-place ComfyUI FLUX.2-dev fp8-mixed single-file (sc-10680): dequant the
    /// inline-scale fp8 MLPs + remap the BFL keys into an in-memory map ([`convert::build_comfyui_dit_map`])
    /// at the compute dtype (f32), then route by `self.quant` exactly as [`load_one_quantizable`]'s
    /// dense-tier regime does — the snapshot `transformer/` dir simply replaced by the single file:
    /// - **quant** (the 32B dev path): stage the dense f32 DiT in CPU RAM, then fold each projection onto
    ///   the GPU (`quantize`); the dense f32 32B never lands on the GPU (it would not fit).
    /// - **no quant** (small fixtures only): build dense on-device.
    fn load_comfyui_dit(&self, dit_file: &Path) -> CResult<Flux2Transformer> {
        // SAFETY: read-only mmap of a weight file; the standard candle loading path.
        let mmap =
            unsafe { candle_gen::candle_core::safetensors::MmapedSafetensors::new(dit_file) }
                .map_err(|e| {
                    CandleError::Msg(format!("flux2 comfyui: mmap {}: {e}", dit_file.display()))
                })?;
        let map = convert::build_comfyui_dit_map(&mmap, self.dtype)?;
        match self.quant {
            Some(q) => {
                let vb = VarBuilder::from_tensors(map, self.dtype, &Device::Cpu);
                let mut dit = Flux2Transformer::new(&self.cfg, vb)?;
                dit.quantize(q, &self.device)?;
                Ok(dit)
            }
            None => {
                let vb = VarBuilder::from_tensors(map, self.dtype, &self.device);
                Ok(Flux2Transformer::new(&self.cfg, vb)?)
            }
        }
    }

    fn load_components(&self) -> CResult<Components> {
        let (te, transformer) = match &self.comfyui_dit {
            // In-place ComfyUI DiT (sc-10680): the Mistral TE is NOT in the single DiT file, so it comes
            // from the snapshot through the same per-tier quant path (`load_te_seq` is the TE-only
            // quantizable loader); the DiT is dequanted + quantized from the in-place file.
            Some(_) => (self.load_te_seq()?, self.load_dit_seq()?),
            None => self.load_te_and_dit()?,
        };
        let vae = Flux2Vae::new(self.component_vb("vae")?)?;
        let tokenizer = self.build_tokenizer()?;
        // Load the optional PiD super-resolving decoder once (epic 7840 / sc-7853) when the caller
        // opted in via `LoadSpec::pid`; otherwise `None` and the render path uses the native Flux2Vae.
        // Resident: this set is cached across requests, so the overlay must be loaded for whichever later
        // request asks for it (F-177 — only the `Sequential` path gates this on `req.use_pid`).
        let pid = self.load_pid(true)?;
        Ok(Components {
            te: Arc::new(te),
            transformer: Arc::new(transformer),
            vae: Arc::new(vae),
            tokenizer: Arc::new(tokenizer),
            pid,
        })
    }

    /// Build the prompt tokenizer **once** (sc-8991 / F-011). The config (pad token + chat template) is
    /// variant-specific: klein uses the Qwen2 `<|endoftext|>` pad + the Qwen no-think chat template; dev
    /// uses the Mistral `<pad>` + the `[INST]…[/INST]` template. Callers cache the result on their
    /// `Components` / provider struct and reuse it across encodes rather than re-parsing per prompt.
    pub(crate) fn build_tokenizer(&self) -> CResult<TextTokenizer> {
        let (pad_token_id, chat_template) = if self.variant.is_dev() {
            (MISTRAL_PAD_TOKEN_ID, ChatTemplate::Flux2DevMistral)
        } else {
            (QWEN_PAD_TOKEN_ID, ChatTemplate::QwenInstructNoThink)
        };
        TextTokenizer::from_file(
            self.root.join("tokenizer/tokenizer.json"),
            TokenizerConfig {
                max_length: self.cfg.max_sequence_length,
                pad_token_id,
                chat_template,
                pad_to_max_length: true,
            },
        )
        .map_err(|e| CandleError::Msg(format!("flux2: load tokenizer: {e}")))
    }

    /// Tokenize + encode the prompt to `prompt_embeds` `[1, 512, 3·hidden]` (f32). `tok` is the cached
    /// tokenizer ([`Self::build_tokenizer`]) — parsed once, reused across encodes (sc-8991 / F-011).
    pub(crate) fn encode(
        &self,
        te: &Flux2PromptEncoder,
        tok: &TextTokenizer,
        prompt: &str,
    ) -> CResult<Tensor> {
        let out = tok
            .tokenize(prompt)
            .map_err(|e| CandleError::Msg(format!("flux2: tokenize: {e}")))?;
        let len = out.ids.len();
        let ids: Vec<u32> = out.ids.iter().map(|&i| i as u32).collect();
        let mask: Vec<i64> = out.mask.iter().map(|&m| m as i64).collect();
        let input_ids = Tensor::from_vec(ids, (1, len), &self.device)?;
        let attn_mask = Tensor::from_vec(mask, (1, len), &self.device)?;
        Ok(te.prompt_embeds(&input_ids, &attn_mask)?)
    }

    /// Encode the optional classifier-free **negative** prompt for the klein CFG blend: `Some` only on a
    /// non-embedded-guidance variant with `guidance > 1` (klein runs CFG-free at 1.0; dev is embedded-
    /// guidance, single-forward, so always `None`). Takes the TE + tokenizer directly so both the
    /// resident and sequential residency phases share the exact CFG condition.
    fn encode_negative(
        &self,
        te: &Flux2PromptEncoder,
        tok: &TextTokenizer,
        req: &GenerationRequest,
        guidance: f32,
    ) -> CResult<Option<Tensor>> {
        if !self.variant.uses_embedded_guidance() && guidance > 1.0 {
            let neg = req.negative_prompt.as_deref().unwrap_or(" ");
            Ok(Some(self.encode(te, tok, neg)?))
        } else {
            Ok(None)
        }
    }

    fn encode_phase(
        &self,
        phase: &TextPhase,
        req: &GenerationRequest,
    ) -> CResult<(Tensor, Option<Tensor>, f32)> {
        let guidance = req.guidance.unwrap_or(self.variant.default_guidance());
        let encode = |te: &Flux2PromptEncoder, tok: &TextTokenizer| -> CResult<_> {
            Ok((
                self.encode(te, tok, &req.prompt)?,
                self.encode_negative(te, tok, req, guidance)?,
                guidance,
            ))
        };
        let encoded = match phase {
            TextPhase::Resident(comps) => encode(&comps.te, &comps.tokenizer),
            TextPhase::Sequential(text) => {
                let (te, tokenizer) = text.as_ref();
                encode(te, tokenizer)
            }
        }?;
        // The sc-12195 post-encode boundary sync used to live here as a local `device.synchronize()`:
        // the sequential seam drops `TextPhase` as soon as this closure returns while candle's
        // Mistral/Qwen encode kernels are still in flight, and the heavy loader reuses the freed CUDA
        // allocations (deterministically corrupting FLUX.2-dev Q4 pixels). The sync now lives in the
        // shared seam — `candle_gen::run_sequential` synchronizes the device after the encode returns
        // and before the text phase drops, so every sequential consumer inherits it (sc-12453). Do
        // not re-add a local sync here; the seam is the single point of enforcement.
        Ok(encoded)
    }

    fn render_phase(
        &self,
        phase: &HeavyPhase,
        req: &GenerationRequest,
        encoded: (Tensor, Option<Tensor>, f32),
        on_progress: &mut dyn FnMut(Progress),
    ) -> CResult<Vec<Image>> {
        let steps = req
            .steps
            .map(|s| s as usize)
            .unwrap_or(self.variant.default_steps() as usize);
        let base_seed = req.seed.unwrap_or_else(gen_core::default_seed);
        let (prompt_embeds, negative, guidance) = encoded;
        let (transformer, vae, pid) = match phase {
            HeavyPhase::Resident(comps) => (
                comps.transformer.as_ref(),
                comps.vae.as_ref(),
                comps.pid.as_deref(),
            ),
            HeavyPhase::Sequential(heavy) => (&heavy.transformer, &heavy.vae, heavy.pid.as_deref()),
        };
        let pid_decoder =
            candle_gen_pid::resolve_pid_decoder(pid, req, base_seed, self.variant.id())?;
        self.sample(
            req,
            transformer,
            vae,
            &prompt_embeds,
            negative.as_ref(),
            pid_decoder.as_ref(),
            guidance,
            steps,
            base_seed,
            on_progress,
        )
    }

    /// The per-image denoise + decode loop shared by both residency modes. Given `prompt_embeds`
    /// (+ optional klein CFG `negative`), a borrowed DiT + VAE, and the resolved PiD seam, the sampled
    /// output is **byte-identical** across both residency modes — only the load/free schedule of the
    /// components handed in differs.
    #[allow(clippy::too_many_arguments)]
    fn sample(
        &self,
        req: &GenerationRequest,
        transformer: &Flux2Transformer,
        vae: &Flux2Vae,
        prompt_embeds: &Tensor,
        negative: Option<&Tensor>,
        pid_decoder: Option<&PidDecoder>,
        guidance: f32,
        steps: usize,
        base_seed: u64,
        on_progress: &mut dyn FnMut(Progress),
    ) -> CResult<Vec<Image>> {
        let embedded_guidance = self.variant.uses_embedded_guidance();
        let (lat_h, lat_w) = pipeline::latent_dims(req.width, req.height);
        let img_ids = pipeline::prepare_grid_ids(lat_h, lat_w);
        let txt_ids = pipeline::prepare_text_ids(self.cfg.max_sequence_length);
        let chunk_attention = req.memory.is_some_and(|memory| memory.chunk_attention);
        let attention_budget = if chunk_attention {
            req.memory
                .and_then(|memory| memory.attention_chunk_size)
                .unwrap_or(memory_strategy::ATTENTION_CHUNK_SIZE) as u64
        } else {
            candle_gen::ATTN_SCORES_BUDGET as u64
        };
        let attention_plan = AttentionPlan::budgeted(AttentionBudget::from_score_elements(
            attention_budget,
            false,
        ));
        let attention_plan = if chunk_attention {
            attention_plan.with_cancel(&req.cancel)
        } else {
            attention_plan
        };
        let transformer_window = req
            .memory
            .and_then(|memory| memory.transformer_window_size)
            .map(|window| window as usize)
            .unwrap_or(memory_strategy::DEFAULT_TRANSFORMER_WINDOW);

        // Curated sampler/scheduler routing (epic 7114 P4, sc-7123). The NATIVE schedule is the legacy
        // empirical-mu flow-match sigmas (descending, trailing 0.0); the same `mu` feeds the curated
        // scheduler axis so `normal`/`karras`/etc. honor the resolution-dependent shift. The default path
        // (sampler/scheduler unset) is the N1 no-op — euler over the native schedule reproduces the legacy
        // `euler_step` flow-match loop within tolerance.
        let mu = pipeline::compute_mu(pipeline::image_seq_len(req.width, req.height), steps);
        let native = pipeline::schedule(steps, req.width, req.height);
        let sigmas =
            candle_gen::resolve_flow_schedule(req.scheduler.as_deref(), mu, steps, &native);

        candle_gen::for_each_image_seed(base_seed, req.count, |seed| {
            let latents =
                pipeline::create_noise(&self.cfg, seed, req.width, req.height, &self.device)?;

            // Per-step latent preview (epic 16948, sc-16955). The sampler's running latent is the packed
            // 128-ch BN-normalized token sequence, so the hook unpacks it onto `(lat_h, lat_w)` — the same
            // grid the decode tail below resolves — and runs the VAE's own de-normalize + unpatchify before
            // projecting the raw 32-channel latent. Built per image so each seed's trajectory starts at
            // frame 1. An inert sink is byte-identical to no hook at all.
            let preview = preview::hook(&req.preview, vae, lat_h, lat_w);

            // The driver does cancel + progress + the euler/curated integrator step. The forward (and the
            // guidance>1 CFG blend) lives inside `predict` so a multi-eval solver re-runs it. FLUX.2 uses
            // the Sigma convention but the model embeds σ×1000, so feed `sigma * 1000.0` to the transformer.
            let latents = candle_gen::run_flow_sampler(
                req.sampler.as_deref(),
                TimestepConvention::Sigma,
                &sigmas,
                latents,
                seed,
                &req.cancel,
                on_progress,
                Some(&preview),
                |latents, sigma| -> CResult<Tensor> {
                    let ts = sigma * 1000.0;
                    let out = if embedded_guidance {
                        // dev: single forward feeding the embedded guidance scalar to the DiT.
                        transformer.forward_with_memory(
                            latents,
                            prompt_embeds,
                            &img_ids,
                            &txt_ids,
                            ts,
                            Some(guidance),
                            attention_plan,
                            transformer_window,
                            &req.cancel,
                        )?
                    } else {
                        let v = transformer.forward_with_memory(
                            latents,
                            prompt_embeds,
                            &img_ids,
                            &txt_ids,
                            ts,
                            None,
                            attention_plan,
                            transformer_window,
                            &req.cancel,
                        )?;
                        match negative {
                            Some(neg) => {
                                let vn = transformer.forward_with_memory(
                                    latents,
                                    neg,
                                    &img_ids,
                                    &txt_ids,
                                    ts,
                                    None,
                                    attention_plan,
                                    transformer_window,
                                    &req.cancel,
                                )?;
                                // vn + guidance·(v − vn)
                                (&vn + ((&v - &vn)? * guidance as f64)?)?
                            }
                            None => v,
                        }
                    };
                    Ok(out)
                },
            )?;

            on_progress(Progress::Decoding);
            let packed = pipeline::unpack_latents(&latents, req.width, req.height)?;
            let decoded = match pid_decoder {
                // PiD consumes the packed BN-normalized [1,128,H/16,W/16] latent directly (the same
                // tensor decode_packed BN-de-normalizes); returns [1,3,4H,4W].
                Some(pid) => pid.decode(&packed)?,
                None if req.memory.is_some_and(|memory| memory.tile_vae_decode) => {
                    let memory = req.memory.expect("guarded above");
                    vae.decode_packed_tiled(
                        &packed,
                        memory
                            .decode_tile_edge
                            .unwrap_or(memory_strategy::DECODE_TILE_EDGE),
                        memory
                            .decode_overlap
                            .unwrap_or(memory_strategy::DECODE_OVERLAP),
                    )?
                }
                None => vae.decode_packed(&packed)?, // [1,3,H,W] in [-1,1]
            };
            to_image(&decoded)
        })
    }
}

/// Map a decoded `[1, 3, H, W]` tensor in `[-1, 1]` to an RGB8 [`Image`].
pub(crate) fn to_image(decoded: &Tensor) -> CResult<Image> {
    let scaled = ((decoded.clamp(-1f32, 1f32)? + 1.0)? * 127.5)?;
    let img = candle_gen::round_rgb8(&scaled)?;
    let img = img.i(0)?.to_device(&Device::Cpu)?;
    let (c, h, w) = img.dims3()?;
    if c != 3 {
        return Err(CandleError::Msg(format!("expected 3 channels, got {c}")));
    }
    let pixels = img.permute((1, 2, 0))?.flatten_all()?.to_vec1::<u8>()?;
    Ok(Image {
        width: w as u32,
        height: h as u32,
        pixels,
    })
}

/// Serialize one bespoke-provider request and always synchronize its device before releasing the
/// lifecycle lock. The request error wins when both execution and synchronization fail: callers
/// must see cancellation/model failures, while a successful request still fails closed when its
/// final device fence does not complete.
pub(crate) fn run_bespoke_request<T>(
    lifecycle: &std::sync::Mutex<()>,
    run: impl FnOnce() -> CResult<T>,
    synchronize: impl FnOnce() -> candle_gen::candle_core::Result<()>,
) -> CResult<T> {
    let _lifecycle = candle_gen::lock_recover(lifecycle);
    let result = run();
    let synchronized = synchronize().map_err(CandleError::Candle);
    match (result, synchronized) {
        (Err(error), _) => Err(error),
        (Ok(_), Err(error)) => Err(error),
        (Ok(value), Ok(())) => Ok(value),
    }
}

/// A loaded candle FLUX.2 generator. The shared residency owner holds either the warm phase pair or
/// the deferred per-request loaders.
pub struct Flux2Generator {
    descriptor: ModelDescriptor,
    pipe: Pipeline,
    residency: Flux2Residency,
    lifecycle: std::sync::Mutex<()>,
    stream_cancel: Arc<std::sync::Mutex<gen_core::CancelFlag>>,
    bounded_host_decode: Arc<std::sync::Mutex<bool>>,
    loaded_quant: Option<Quant>,
    memory_strategy: Option<gen_core::MemoryProviderContract>,
    memory_admission: memory_strategy::Flux2AdmissionRegistry,
}

impl Generator for Flux2Generator {
    fn descriptor(&self) -> &ModelDescriptor {
        &self.descriptor
    }

    fn memory_strategy_contract(&self) -> Option<&gen_core::MemoryProviderContract> {
        self.memory_strategy.as_ref()
    }

    fn memory_strategy_safety_check(
        &self,
        context: &gen_core::MemoryRunContext,
    ) -> gen_core::MemorySafetyDecision {
        let Some(contract) = self.memory_strategy.as_ref() else {
            return gen_core::MemorySafetyDecision::Accept;
        };
        if let Err(error) = memory_strategy::validate_registered_generator_context(context) {
            self.memory_admission.clear_approval();
            return gen_core::MemorySafetyDecision::Reject {
                reason: error.to_string(),
            };
        }
        match memory_strategy::admission_safety_check(contract, context, self.loaded_quant) {
            gen_core::MemorySafetyDecision::Accept => {
                match self.memory_admission.approve(context) {
                    Ok(()) => gen_core::MemorySafetyDecision::Accept,
                    Err(error) => gen_core::MemorySafetyDecision::Reject {
                        reason: error.to_string(),
                    },
                }
            }
            rejected @ gen_core::MemorySafetyDecision::Reject { .. } => {
                self.memory_admission.clear_approval();
                rejected
            }
        }
    }

    fn begin_memory_strategy_request(
        &self,
        context: &gen_core::MemoryRunContext,
    ) -> gen_core::Result<Option<Box<dyn gen_core::MemoryRequestScope + '_>>> {
        let Some(contract) = self.memory_strategy.as_ref() else {
            return Ok(None);
        };
        memory_strategy::validate_context(contract, context, self.loaded_quant)?;
        memory_strategy::validate_registered_generator_context(context)?;
        Ok(Some(Box::new(
            memory_strategy::Flux2MemoryScope::new_bound(
                self.pipe.device.clone(),
                contract,
                context,
                self.memory_admission.clone(),
            )?,
        )))
    }

    fn validate(&self, req: &GenerationRequest) -> gen_core::Result<()> {
        let id = self.descriptor.id;
        self.descriptor.capabilities.validate_request(id, req)?;
        if req.prompt.is_empty() {
            return Err(gen_core::Error::Msg(format!(
                "{id}: prompt must not be empty"
            )));
        }
        if req.steps == Some(0) {
            return Err(gen_core::Error::Msg(format!(
                "{id}: steps must be >= 1 (an explicit 0 renders undenoised noise)"
            )));
        }
        if !req.width.is_multiple_of(SIZE_MULTIPLE) || !req.height.is_multiple_of(SIZE_MULTIPLE) {
            return Err(gen_core::Error::Msg(format!(
                "{id}: width/height must be multiples of {SIZE_MULTIPLE} (got {}x{})",
                req.width, req.height
            )));
        }
        Ok(())
    }

    fn generate(
        &self,
        req: &GenerationRequest,
        on_progress: &mut dyn FnMut(Progress),
    ) -> gen_core::Result<GenerationOutput> {
        self.validate(req)?;
        let _lifecycle = candle_gen::lock_recover(&self.lifecycle);
        self.memory_admission.consume_for_generate(req)?;
        let stage_residency = req
            .memory
            .as_ref()
            .is_some_and(|memory| memory.stage_residency);
        let stream_transformer_blocks = req
            .memory
            .as_ref()
            .is_some_and(|memory| memory.stream_transformer_blocks);
        if req.memory.as_ref().is_some_and(|memory| {
            memory.tile_vae_decode || memory.chunk_attention || memory.stream_transformer_blocks
        }) && !stage_residency
        {
            return Err(gen_core::Error::Unsupported(format!(
                "{}: constrained strategies require request-scoped staged residency",
                self.descriptor.id
            )));
        }
        if req.use_pid
            && req.memory.is_some_and(|memory| {
                memory.tile_vae_decode || memory.chunk_attention || memory.stream_transformer_blocks
            })
        {
            return Err(gen_core::Error::Unsupported(format!(
                "{}: optimized native-VAE strategies do not support PiD decode",
                self.descriptor.id
            )));
        }
        if stream_transformer_blocks && self.memory_strategy.is_none() {
            return Err(gen_core::Error::Unsupported(format!(
                "{}: streamed blocks require the CUDA memory contract",
                self.descriptor.id
            )));
        }
        *candle_gen::lock_recover(&self.stream_cancel) = req.cancel.clone();
        *candle_gen::lock_recover(&self.bounded_host_decode) =
            req.memory.is_some_and(|memory| memory.tile_vae_decode);
        let images = self.residency.run_request_scoped(
            stage_residency,
            stream_transformer_blocks,
            &req.cancel,
            req.use_pid,
            on_progress,
            |text| self.pipe.encode_phase(text, req),
            |_| Ok(self.pipe.device.synchronize()?),
            |heavy, encoded, on_progress| {
                let result = self.pipe.render_phase(heavy, req, encoded, on_progress);
                candle_gen::synchronize_result(&self.pipe.device, result)
            },
        );
        *candle_gen::lock_recover(&self.bounded_host_decode) = false;
        *candle_gen::lock_recover(&self.stream_cancel) = gen_core::CancelFlag::default();
        let images = images?;
        Ok(GenerationOutput::Images(images))
    }
}

/// The txt2img descriptor for `variant`. **klein**: guidance advertised (defaults to 1.0 / CFG-free,
/// but >1.0 runs a classifier-free negative pass), so `supports_negative_prompt`. **dev**: guidance is
/// the embedded scalar (single forward, no negative pass), so `supports_negative_prompt = false`.
/// Both: txt2img only (edit/Reference deferred to epic 6564 story 4), no LoRA, no on-the-fly quant.
fn descriptor(variant: Flux2Variant) -> ModelDescriptor {
    ModelDescriptor {
        control_kinds: None,
        required_components: &[],
        id: variant.id(),
        family: "flux2",
        backend: "candle",
        modality: Modality::Image,
        capabilities: Capabilities {
            // dev is guidance-distilled (embedded scalar, no negative pass); klein runs a
            // classifier-free negative pass when guidance > 1.
            supports_negative_prompt: !variant.uses_embedded_guidance(),
            supports_guidance: true,
            supports_true_cfg: false,
            // txt2img only in this slice — the mlx edit/Reference surface is deferred.
            conditioning: vec![],
            supports_lora: false,
            supports_lokr: false,
            // Curated sampler/scheduler menu (epic 7114 P4, sc-7123). The legacy `flow_match_euler`
            // scheduler alias is retained and falls back to the native schedule via the N3 path.
            samplers: candle_gen::curated_sampler_names(),
            schedulers: candle_gen::menu_with_aliases(
                candle_gen::curated_scheduler_names(),
                &["flow_match_euler"],
            ),
            supported_guidance_methods: vec![],
            min_size: 256,
            max_size: 2048,
            max_count: 8,
            // Not a distilled fixed-schedule model: any step count the shared sanity caps
            // admit is renderable (sc-19502).
            supported_steps: Vec::new(),
            mac_only: false,
            // Both quantize on-the-fly (CPU-stage → quantize-onto-GPU): dev folds the 32B DiT + Mistral
            // TE to fit the memory ceiling; klein (sc-11031) folds only the 9B DiT and keeps the Qwen3
            // TE dense bf16 (epic 8506 DENSE_TE, `Pipeline::te_quant`).
            supported_quants: &[Quant::Q4, Quant::Q8],
            component_precision_floors: &[],
            supports_kv_cache: false,
            // FLUX.2 uses the empirical-mu shifted flow-match schedule.
            requires_sigma_shift: true,
            supports_sequential_offload: true,
            // Per-step latent previews (epic 16948, sc-16955): every shipped FLUX.2 lane hands the
            // shared sampler a `crate::preview` hook that projects the raw 32-channel latent through
            // the epic-16624 fit. `candle-gen-catalog`'s `preview_advertising` guard derives this
            // flag from the sources, so it cannot be set ahead of the wiring or left behind it.
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

/// FLUX.2-klein-9b txt2img descriptor (the surface sc-3695 wired).
pub fn descriptor_klein() -> ModelDescriptor {
    descriptor(Flux2Variant::Klein9b)
}

/// FLUX.2-dev txt2img descriptor (epic 6564 story 1): the guidance-distilled 32B flagship.
pub fn descriptor_dev() -> ModelDescriptor {
    descriptor(Flux2Variant::Dev)
}

fn generator_from_pipeline(
    pipe: Pipeline,
    memory_spec: Option<&LoadSpec>,
) -> gen_core::Result<Flux2Generator> {
    let variant = pipe.variant;
    let resident_pipe = pipe.clone();
    let text_pipe = pipe.clone();
    let heavy_pipe = pipe.clone();
    let stream_cancel = Arc::new(std::sync::Mutex::new(gen_core::CancelFlag::default()));
    let heavy_cancel = stream_cancel.clone();
    let bounded_host_decode = Arc::new(std::sync::Mutex::new(false));
    let heavy_bounded_host_decode = bounded_host_decode.clone();
    let residency = Flux2Residency::request_scoped_with_resident(
        move |_| {
            let comps = resident_pipe.load_components()?;
            Ok((
                TextPhase::Resident(comps.clone()),
                HeavyPhase::Resident(comps),
            ))
        },
        move |_| {
            Ok(TextPhase::Sequential(Box::new((
                text_pipe.load_te_seq()?,
                text_pipe.build_tokenizer()?,
            ))))
        },
        move |use_pid, stream_transformer_blocks| {
            Ok(HeavyPhase::Sequential(Box::new(
                heavy_pipe.load_heavy_seq_with_memory(
                    use_pid,
                    stream_transformer_blocks,
                    *candle_gen::lock_recover(&heavy_bounded_host_decode),
                    &candle_gen::lock_recover(&heavy_cancel),
                )?,
            )))
        },
    );
    let loaded_quant = match memory_spec {
        Some(spec) => memory_strategy::resolved_quant(spec)?,
        None => pipe.quant,
    };
    #[cfg(any(feature = "cuda", test))]
    let memory_strategy = memory_spec
        .map(|spec| memory_strategy::contract_for_variant(variant, spec))
        .transpose()?;
    #[cfg(not(any(feature = "cuda", test)))]
    let memory_strategy = None;
    Ok(Flux2Generator {
        descriptor: descriptor(variant),
        pipe,
        residency,
        lifecycle: std::sync::Mutex::new(()),
        stream_cancel,
        bounded_host_decode,
        loaded_quant,
        memory_strategy,
        memory_admission: memory_strategy::Flux2AdmissionRegistry::new(variant.id()),
    })
}

/// Construct a lazy candle FLUX.2 generator for `variant`. `spec.weights` must be a
/// [`WeightsSource::Dir`] pointing at a diffusers snapshot (`text_encoder/`, `transformer/`, `vae/`,
/// `tokenizer/`) — klein at `black-forest-labs/FLUX.2-klein-9B`, dev at `black-forest-labs/FLUX.2-dev`
/// (whose `text_encoder/` is the Mistral3 checkpoint). Adapters / control overlays are rejected (not
/// wired). `spec.quantize` (Q4/Q8) is honored by BOTH variants — each component staged dense in CPU RAM
/// then folded onto the GPU: **dev** quantizes the 32B DiT + the ~24B Mistral TE (neither fits dense);
/// **klein** (sc-11031) quantizes ONLY the 9B DiT and keeps its 8B Qwen3 TE dense bf16 (epic 8506
/// DENSE_TE, `Pipeline::te_quant`). Without quant both load fully dense (klein's bf16 tier; dev is
/// fixture-only there — the full 32B needs the quant).
fn load_variant(variant: Flux2Variant, spec: &LoadSpec) -> gen_core::Result<Box<dyn Generator>> {
    Ok(Box::new(load_variant_concrete(variant, spec)?))
}

fn load_variant_concrete(
    variant: Flux2Variant,
    spec: &LoadSpec,
) -> gen_core::Result<Flux2Generator> {
    let id = variant.id();
    let root = match &spec.weights {
        WeightsSource::Dir(p) => p.clone(),
        WeightsSource::File(_) => {
            return Err(gen_core::Error::Msg(format!(
                "{id} expects a snapshot directory (text_encoder/ transformer/ vae/ tokenizer/), \
                 not a single .safetensors file"
            )));
        }
    };
    if !spec.adapters.is_empty() {
        return Err(gen_core::Error::Unsupported(format!(
            "candle {id} does not support LoRA/LoKr yet"
        )));
    }
    if spec.identity.is_some() || spec.text_encoder.is_some() || !spec.components.is_empty() {
        return Err(gen_core::Error::Unsupported(format!(
            "candle {id} does not support identity, external text-encoder, or named-component weights"
        )));
    }
    // Both variants honor Q4/Q8 on-the-fly (CPU-stage dense → quantize-onto-GPU): dev folds the 32B DiT
    // + the ~24B Mistral TE (neither fits the GPU dense), klein (sc-11031) folds ONLY the 9B DiT and
    // keeps the 8B Qwen3 TE DENSE bf16 in every tier (epic 8506 DENSE_TE — see `Pipeline::te_quant`).
    let quant = memory_strategy::resolved_quant(spec)?;
    if spec.control.is_some() || !spec.extra_controls.is_empty() || spec.ip_adapter.is_some() {
        return Err(gen_core::Error::Unsupported(format!(
            "candle {id} does not support control / IP-adapter / edit yet (txt2img only)"
        )));
    }
    let device = candle_gen::default_device()?;
    let pipe = Pipeline::load(variant, quant, &root, &device, spec.pid.clone());
    generator_from_pipeline(pipe, Some(spec))
}

/// Construct a lazy candle FLUX.2-**dev** generator that reads its **DiT** in place from an existing
/// ComfyUI fp8-mixed single-file (epic 10451 Phase 2e, sc-10680) — no copy, no re-download.
/// `transformer_file` is the user's `diffusion_models/flux2_dev_fp8mixed.safetensors` (BFL-native keys,
/// inline-scale fp8 MLPs); its keys are remapped + its fp8 weights dequanted (`w = w_fp8·weight_scale`)
/// in memory at component build (`convert::build_comfyui_dit_map`). `snapshot_dir` is a resident
/// FLUX.2-dev diffusers snapshot supplying the Mistral text encoder, VAE, and tokenizer (none of which
/// are in the single DiT file). `quant` (Q4/Q8) folds the dequanted DiT + the Mistral TE onto the GPU —
/// the 32B dev does not fit dense — matching the resident dev path; `None` is fixture-only. txt2img
/// only; no adapters / control / edit / PiD.
pub fn load_from_comfyui_dit(
    transformer_file: impl Into<PathBuf>,
    snapshot_dir: impl Into<PathBuf>,
    quant: Option<Quant>,
) -> gen_core::Result<Box<dyn Generator>> {
    let device = candle_gen::default_device()?;
    let root = snapshot_dir.into();
    let pipe = Pipeline::load_comfyui(quant, &root, &device, transformer_file.into());
    Ok(Box::new(generator_from_pipeline(pipe, None)?))
}

/// Registry load hook for `flux2_klein_9b`.
pub fn load_klein(spec: &LoadSpec) -> gen_core::Result<Box<dyn Generator>> {
    load_variant(Flux2Variant::Klein9b, spec)
}

/// Registry load hook for `flux2_dev`.
pub fn load_dev(spec: &LoadSpec) -> gen_core::Result<Box<dyn Generator>> {
    load_variant(Flux2Variant::Dev, spec)
}

// Link-time self-registration into gen-core's model registry — one per txt2img variant.
candle_gen::register_generators! {
    pub(crate) const KLEIN_REGISTRATION = descriptor_klein => load_klein
}
candle_gen::register_generators! {
    pub(crate) const DEV_REGISTRATION = descriptor_dev => load_dev
}

/// Add all Candle FLUX.2 providers to an explicit media registry builder.
pub fn register_providers(
    registry: candle_gen::gen_core::ProviderRegistryBuilder,
) -> candle_gen::gen_core::ProviderRegistryBuilder {
    let registry = registry
        .register_generator(KLEIN_REGISTRATION)
        .register_generator(DEV_REGISTRATION);
    #[cfg(feature = "cuda")]
    let registry = registry
        .register_memory_strategy(KLEIN_MEMORY_REGISTRATION)
        .register_memory_behavior(KLEIN_MEMORY_BEHAVIOR)
        .register_memory_strategy(DEV_MEMORY_REGISTRATION)
        .register_memory_behavior(DEV_MEMORY_BEHAVIOR);
    registry
}

#[cfg(feature = "cuda")]
const DEV_MEMORY_REGISTRATION: gen_core::MemoryRegistration = gen_core::MemoryRegistration {
    provider_id: config::FLUX2_DEV_ID,
    contract: memory_strategy::provider_contract,
    safety_check: memory_strategy::registered_safety_check,
};

#[cfg(feature = "cuda")]
const KLEIN_MEMORY_REGISTRATION: gen_core::MemoryRegistration = gen_core::MemoryRegistration {
    provider_id: config::FLUX2_KLEIN_9B_ID,
    contract: memory_strategy::klein_provider_contract,
    safety_check: memory_strategy::registered_safety_check,
};

#[cfg(feature = "cuda")]
const DEV_MEMORY_BEHAVIOR: gen_core::MemoryBehaviorRegistration =
    gen_core::MemoryBehaviorRegistration {
        provider_id: config::FLUX2_DEV_ID,
        valid_fixtures: memory_strategy::registered_valid_fixture,
        begin_request: memory_strategy::registered_begin_request,
    };

#[cfg(feature = "cuda")]
const KLEIN_MEMORY_BEHAVIOR: gen_core::MemoryBehaviorRegistration =
    gen_core::MemoryBehaviorRegistration {
        provider_id: config::FLUX2_KLEIN_9B_ID,
        valid_fixtures: memory_strategy::registered_valid_fixture,
        begin_request: memory_strategy::registered_begin_request,
    };

/// Build the complete explicit Candle FLUX.2 provider catalog.
pub fn provider_registry() -> candle_gen::gen_core::Result<candle_gen::gen_core::ProviderRegistry> {
    register_providers(candle_gen::gen_core::ProviderRegistryBuilder::new()).build()
}

#[cfg(test)]
mod explicit_registry_tests {
    #[test]
    fn explicit_catalog_has_stable_surface() {
        let registry = super::provider_registry().unwrap();
        let explicit: Vec<String> = registry
            .generators()
            .map(|registration| (registration.descriptor)().id.to_string())
            .collect();

        assert_eq!(explicit, ["flux2_klein_9b", "flux2_dev"]);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::{FLUX2_DEV_ID, FLUX2_KLEIN_9B_ID};
    use candle_gen::gen_core::ConditioningKind;

    #[test]
    fn bespoke_request_finalizes_success_cancellation_and_error() {
        use std::sync::atomic::{AtomicUsize, Ordering};

        let lifecycle = std::sync::Mutex::new(());
        let syncs = AtomicUsize::new(0);
        let success = run_bespoke_request(
            &lifecycle,
            || Ok(7),
            || {
                syncs.fetch_add(1, Ordering::SeqCst);
                Ok(())
            },
        )
        .unwrap();
        assert_eq!(success, 7);

        let canceled: CResult<()> = run_bespoke_request(
            &lifecycle,
            || Err(CandleError::Canceled),
            || {
                syncs.fetch_add(1, Ordering::SeqCst);
                Ok(())
            },
        );
        assert!(matches!(canceled, Err(CandleError::Canceled)));

        let failed: CResult<()> = run_bespoke_request(
            &lifecycle,
            || Err(CandleError::Msg("fixture failure".to_owned())),
            || {
                syncs.fetch_add(1, Ordering::SeqCst);
                Ok(())
            },
        );
        assert!(matches!(failed, Err(CandleError::Msg(_))));
        assert_eq!(syncs.load(Ordering::SeqCst), 3);
    }

    #[test]
    fn bespoke_request_lifecycle_serializes_concurrent_generate() {
        use std::sync::atomic::{AtomicUsize, Ordering};
        use std::sync::Arc;

        let lifecycle = Arc::new(std::sync::Mutex::new(()));
        let active = Arc::new(AtomicUsize::new(0));
        let max_active = Arc::new(AtomicUsize::new(0));
        let mut threads = Vec::new();
        for _ in 0..2 {
            let lifecycle = lifecycle.clone();
            let active = active.clone();
            let max_active = max_active.clone();
            threads.push(std::thread::spawn(move || {
                run_bespoke_request(
                    &lifecycle,
                    || {
                        let now = active.fetch_add(1, Ordering::SeqCst) + 1;
                        max_active.fetch_max(now, Ordering::SeqCst);
                        std::thread::sleep(std::time::Duration::from_millis(20));
                        active.fetch_sub(1, Ordering::SeqCst);
                        Ok(())
                    },
                    || Ok(()),
                )
                .unwrap();
            }));
        }
        for thread in threads {
            thread.join().unwrap();
        }
        assert_eq!(max_active.load(Ordering::SeqCst), 1);
    }

    #[test]
    fn auto_detected_packed_tier_is_the_generators_loaded_tier() {
        for (bits, expected) in [(4, Quant::Q4), (8, Quant::Q8)] {
            let root_tmp = tempfile::tempdir().unwrap();
            let root = root_tmp.path().to_path_buf();
            let transformer = root.join("transformer");
            std::fs::create_dir_all(&transformer).unwrap();
            std::fs::write(
                transformer.join("config.json"),
                format!(r#"{{"quantization":{{"bits":{bits},"group_size":64}}}}"#),
            )
            .unwrap();
            let mut spec = LoadSpec::new(WeightsSource::Dir(root.clone()));
            spec.load_shape = gen_core::LoadShape::DeferredMaterialization;
            assert_eq!(
                memory_strategy::resolved_quant(&spec).unwrap(),
                Some(expected)
            );
            for (label, generator) in [
                (
                    "dev",
                    load_variant_concrete(Flux2Variant::Dev, &spec).expect("lazy dev generator"),
                ),
                (
                    "klein",
                    load_variant_concrete(Flux2Variant::Klein9b, &spec)
                        .expect("lazy Klein generator"),
                ),
            ] {
                assert_eq!(
                    generator.pipe.quant,
                    Some(expected),
                    "{label} execution pipeline must carry the auto-detected tier"
                );
                assert_eq!(
                    generator.loaded_quant,
                    Some(expected),
                    "{label} admission identity must match execution"
                );
                let contract = generator.memory_strategy_contract().unwrap();
                let context = gen_core::standard_memory_behavior_context(
                    contract,
                    gen_core::MemoryStrategy::Resident,
                    memory_strategy::resolved_numeric_tier(&spec).unwrap(),
                    gen_core::MemoryBehaviorRoute {
                        mode: gen_core::MemoryMode::TextToImage,
                        reference_count: 0,
                        use_pid: false,
                        has_phases: false,
                        overlay: None,
                    },
                )
                .unwrap();
                assert!(
                    matches!(
                        generator.memory_strategy_safety_check(&context),
                        gen_core::MemorySafetyDecision::Accept
                    ),
                    "{label} must admit the auto-detected packed tier"
                );
            }
        }
    }

    #[test]
    fn registered_generator_requires_exact_safety_begin_configure_handshake() {
        let mut spec = LoadSpec::new(WeightsSource::Dir(PathBuf::from("/missing-flux2-dev")))
            .with_quant(Quant::Q4);
        spec.load_shape = gen_core::LoadShape::DeferredMaterialization;
        let generator = load_dev(&spec).expect("lazy dev generator");
        let contract = generator.memory_strategy_contract().unwrap().clone();
        let route = gen_core::MemoryBehaviorRoute {
            mode: gen_core::MemoryMode::TextToImage,
            reference_count: 0,
            use_pid: false,
            has_phases: false,
            overlay: None,
        };
        let context = gen_core::standard_memory_behavior_context(
            &contract,
            gen_core::MemoryStrategy::BoundedDecode,
            memory_strategy::resolved_numeric_tier(&spec).unwrap(),
            route,
        )
        .unwrap();
        let manual = GenerationRequest {
            prompt: "manual".to_owned(),
            memory: contract.generation_memory(&context.selection),
            ..Default::default()
        };
        assert!(generator.generate(&manual, &mut |_| {}).is_err());
        assert!(generator.begin_memory_strategy_request(&context).is_err());

        assert!(matches!(
            generator.memory_strategy_safety_check(&context),
            gen_core::MemorySafetyDecision::Accept
        ));
        let mut unconfigured = generator
            .begin_memory_strategy_request(&context)
            .unwrap()
            .unwrap();
        assert!(generator
            .generate(
                &GenerationRequest {
                    prompt: "unconfigured".to_owned(),
                    ..Default::default()
                },
                &mut |_| {},
            )
            .is_err());
        unconfigured
            .finish(gen_core::MemoryRunOutcome::Canceled)
            .unwrap();

        let mut mutations = Vec::new();
        let mut abi = context.clone();
        abi.calibration_abi += 1;
        mutations.push(abi);
        let mut fingerprint = context.clone();
        fingerprint.calibration_fingerprint.push_str("-stale");
        mutations.push(fingerprint);
        let mut phases = context.clone();
        phases.has_phases = true;
        mutations.push(phases);
        let mut mode = context.clone();
        mode.mode = gen_core::MemoryMode::Edit;
        mode.geometry.reference_count = 1;
        mode.has_reference = true;
        mutations.push(mode);
        let mut overlay = context.clone();
        overlay.overlay = Some(memory_strategy::CONTROL_OVERLAY.to_owned());
        mutations.push(overlay);
        for mutated in mutations {
            assert!(matches!(
                generator.memory_strategy_safety_check(&context),
                gen_core::MemorySafetyDecision::Accept
            ));
            assert!(generator.begin_memory_strategy_request(&mutated).is_err());
        }

        let alternate = gen_core::standard_memory_behavior_context(
            &contract,
            gen_core::MemoryStrategy::BoundedAttention,
            memory_strategy::resolved_numeric_tier(&spec).unwrap(),
            gen_core::MemoryBehaviorRoute {
                mode: gen_core::MemoryMode::TextToImage,
                reference_count: 0,
                use_pid: false,
                has_phases: false,
                overlay: None,
            },
        )
        .unwrap();
        assert!(matches!(
            generator.memory_strategy_safety_check(&context),
            gen_core::MemorySafetyDecision::Accept
        ));
        assert!(generator.begin_memory_strategy_request(&alternate).is_err());

        assert!(matches!(
            generator.memory_strategy_safety_check(&context),
            gen_core::MemorySafetyDecision::Accept
        ));
        let mut scope = generator
            .begin_memory_strategy_request(&context)
            .unwrap()
            .unwrap();
        let mut configured = GenerationRequest {
            prompt: "configured".to_owned(),
            ..Default::default()
        };
        scope.configure_request(&mut configured).unwrap();
        let copied = configured.clone();
        assert!(generator.generate(&copied, &mut |_| {}).is_err());
        configured.width /= 2;
        assert!(generator.generate(&configured, &mut |_| {}).is_err());
        scope
            .finish(gen_core::MemoryRunOutcome::Error {
                message: "adversarial rejection".to_owned(),
            })
            .unwrap();
    }

    /// F-177 (sc-12089): the PiD student is loaded only when the request will actually decode through it,
    /// so a `Sequential` generate that never asked for PiD does not pay for it — per generate, resident
    /// through the whole denoise, inside the peak the path exists to bound.
    ///
    /// `load_components` passes `use_pid = true` unconditionally and that is correct, not an oversight:
    /// it builds one cached set BEFORE any request exists, so the overlay has to be there for whichever
    /// later request wants it. GPU- and weights-free (`Pipeline::load` does no I/O).
    #[test]
    fn pid_loads_only_when_the_request_uses_it() {
        let spec = PidWeights {
            checkpoint: WeightsSource::File("/pid.safetensors".into()),
            gemma: WeightsSource::Dir("/gemma".into()),
        };
        let root = Path::new("/nonexistent");
        let with = Pipeline::load(Flux2Variant::Klein9b, None, root, &Device::Cpu, Some(spec));
        let without = Pipeline::load(Flux2Variant::Klein9b, None, root, &Device::Cpu, None);

        // Opted in at load AND wanted by this request → load it.
        assert!(with.pid_to_load(true).is_some());
        // Opted in at load but NOT wanted by this request → skip it. This is the F-177 arm: before the
        // fix the sequential path loaded the engine and `resolve_pid_decoder` then returned `None` for it,
        // so not a byte was ever read.
        assert!(with.pid_to_load(false).is_none());
        // Never opted in → nothing to load, whatever the request asked for. (`use_pid` with no `pid` spec
        // is `resolve_pid_decoder`'s error to report, not a reason to load anything here.)
        assert!(without.pid_to_load(true).is_none());
        assert!(without.pid_to_load(false).is_none());
    }

    #[test]
    fn registers_and_resolves_as_candle() {
        let spec = LoadSpec::new(WeightsSource::Dir("/nonexistent".into()));
        let g = crate::provider_registry()
            .unwrap()
            .load(FLUX2_KLEIN_9B_ID, &spec)
            .expect("flux2 is registered");
        assert_eq!(g.descriptor().id, FLUX2_KLEIN_9B_ID);
        assert_eq!(g.descriptor().family, "flux2");
        assert_eq!(g.descriptor().backend, "candle");
        assert_eq!(g.descriptor().modality, Modality::Image);
    }

    #[test]
    fn klein_descriptor_advertises_only_wired_txt2img_surface() {
        let d = descriptor_klein();
        assert_eq!(d.id, FLUX2_KLEIN_9B_ID);
        assert!(d.capabilities.supports_guidance);
        // klein runs a classifier-free negative pass when guidance > 1.
        assert!(d.capabilities.supports_negative_prompt);
        assert!(d.capabilities.requires_sigma_shift);
        assert!(!d.capabilities.mac_only);
        assert!(d.capabilities.conditioning.is_empty());
        assert!(!d.capabilities.supports_lora);
        assert!(!d.capabilities.supports_kv_cache);
        // klein now quantizes its DiT on-the-fly (sc-11031); the Qwen3 TE stays dense (`te_quant`).
        assert_eq!(d.capabilities.supported_quants, &[Quant::Q4, Quant::Q8]);
        assert!(!d.capabilities.accepts(ConditioningKind::Reference));
    }

    #[test]
    fn dev_registers_and_advertises_embedded_guidance_surface() {
        let spec = LoadSpec::new(WeightsSource::Dir("/nonexistent".into()));
        let g = crate::provider_registry()
            .unwrap()
            .load(FLUX2_DEV_ID, &spec)
            .expect("flux2_dev is registered");
        assert_eq!(g.descriptor().id, FLUX2_DEV_ID);
        assert_eq!(g.descriptor().family, "flux2");
        assert_eq!(g.descriptor().backend, "candle");
        assert_eq!(g.descriptor().modality, Modality::Image);
        let d = descriptor_dev();
        assert!(d.capabilities.supports_guidance);
        // dev is guidance-distilled (embedded scalar) — no negative pass, no true-CFG, not mac-only.
        assert!(!d.capabilities.supports_negative_prompt);
        assert!(!d.capabilities.supports_true_cfg);
        assert!(!d.capabilities.mac_only);
        assert!(d.capabilities.conditioning.is_empty());
        assert!(d.capabilities.requires_sigma_shift);
        // dev and klein both advertise Q4/Q8 now (CPU-stage → quantize-onto-GPU); klein keeps its Qwen3
        // TE dense (`te_quant`), dev folds the Mistral TE with the DiT (sc-11031).
        assert_eq!(d.capabilities.supported_quants, &[Quant::Q4, Quant::Q8]);
        assert_eq!(
            descriptor_klein().capabilities.supported_quants,
            &[Quant::Q4, Quant::Q8]
        );
    }

    #[test]
    fn validate_accepts_txt2img_and_rejects_unsupported() {
        let spec = LoadSpec::new(WeightsSource::Dir("/nonexistent".into()));
        let g = crate::provider_registry()
            .unwrap()
            .load(FLUX2_KLEIN_9B_ID, &spec)
            .unwrap();
        let ok = GenerationRequest {
            prompt: "a rusty robot holding a lit candle".into(),
            ..Default::default()
        };
        assert!(g.validate(&ok).is_ok());
        for bad in [
            GenerationRequest::default(),
            GenerationRequest {
                prompt: "x".into(),
                width: 1000,
                ..Default::default()
            },
            GenerationRequest {
                prompt: "x".into(),
                steps: Some(0),
                ..Default::default()
            },
        ] {
            assert!(g.validate(&bad).is_err(), "should reject: {bad:?}");
        }

        // sc-12612: `SIZE_MULTIPLE` is the pinned stride SceneWorks ties every advertised FLUX.2 bucket
        // to. Pin the value and mutation-check that a size which is a multiple of 8 (the VAE scale) but
        // not SIZE_MULTIPLE (16) is still rejected with the stride error, and an on-stride size passes.
        assert_eq!(SIZE_MULTIPLE, 16);
        let off_stride = g
            .validate(&GenerationRequest {
                prompt: "x".into(),
                width: 1000, // 125×8 — a multiple of 8 but not SIZE_MULTIPLE
                ..Default::default()
            })
            .unwrap_err()
            .to_string();
        assert!(
            off_stride.contains("multiples of 16"),
            "expected the stride error, got: {off_stride}"
        );
        assert!(g
            .validate(&GenerationRequest {
                prompt: "x".into(),
                width: 1024, // 64×16 — on-stride
                ..Default::default()
            })
            .is_ok());
    }

    #[test]
    fn load_rejects_unwired_surfaces() {
        use candle_gen::gen_core::{AdapterKind, AdapterSpec, IdentityWeights};
        let lora = LoadSpec::new(WeightsSource::Dir("/snap".into())).with_adapters(vec![
            AdapterSpec::new("/lora.safetensors".into(), 1.0, AdapterKind::Lora),
        ]);
        assert!(matches!(
            load_klein(&lora).err().expect("err"),
            gen_core::Error::Unsupported(_)
        ));
        let mut identity = LoadSpec::new(WeightsSource::Dir("/snap".into()));
        identity.identity = Some(IdentityWeights::default());
        let mut external_text_encoder = LoadSpec::new(WeightsSource::Dir("/snap".into()));
        external_text_encoder.text_encoder = Some(WeightsSource::Dir("/external-te".into()));
        let named_component = LoadSpec::new(WeightsSource::Dir("/snap".into())).with_component(
            "unwired_component",
            WeightsSource::File("/component.bin".into()),
        );
        for spec in [&identity, &external_text_encoder, &named_component] {
            assert!(matches!(
                load_klein(spec).err().expect("unwired field must reject"),
                gen_core::Error::Unsupported(_)
            ));
        }
        // klein (sc-11031) AND dev now accept Q4/Q8 on-the-fly (CPU-stage → quantize-onto-GPU): klein
        // folds only the 9B DiT (Qwen3 TE stays dense bf16, `te_quant`), dev folds the DiT + Mistral TE.
        // The generator builds lazily, so load succeeds without touching the (nonexistent) weights.
        let klein_q4 = LoadSpec::new(WeightsSource::Dir("/snap".into())).with_quant(Quant::Q4);
        assert!(load_klein(&klein_q4).is_ok());
        let klein_q8 = LoadSpec::new(WeightsSource::Dir("/snap".into())).with_quant(Quant::Q8);
        assert!(load_klein(&klein_q8).is_ok());
        let dev_quant = LoadSpec::new(WeightsSource::Dir("/snap".into())).with_quant(Quant::Q4);
        assert!(load_dev(&dev_quant).is_ok());
    }

    /// The loader's packed/dense routing decision (sc-9087): a component whose `config.json` carries a
    /// `quantization` block is a packed MLX tier (build directly on the GPU, no dense CPU staging); a
    /// component with a plain config, or none, is dense (the CPU-stage → quantize-onto path). Drives
    /// `Pipeline::load_one_quantizable`'s device choice.
    #[test]
    fn component_is_packed_reads_quantization_block() {
        let dir_tmp = tempfile::tempdir().unwrap();
        let dir = dir_tmp.path().to_path_buf();
        let pipe = Pipeline::load(Flux2Variant::Dev, Some(Quant::Q4), &dir, &Device::Cpu, None);

        let packed = dir.join("transformer");
        std::fs::create_dir_all(&packed).unwrap();
        std::fs::write(
            packed.join("config.json"),
            r#"{"quantization": {"bits": 4, "group_size": 64}}"#,
        )
        .unwrap();
        assert!(
            pipe.component_is_packed("transformer").unwrap(),
            "a `quantization` block ⇒ packed tier"
        );

        let dense = dir.join("text_encoder");
        std::fs::create_dir_all(&dense).unwrap();
        std::fs::write(dense.join("config.json"), r#"{"hidden_size": 5120}"#).unwrap();
        assert!(
            !pipe.component_is_packed("text_encoder").unwrap(),
            "no `quantization` block ⇒ dense tier"
        );
        // A component with no config.json at all → dense (fixtures still load).
        assert!(!pipe.component_is_packed("vae").unwrap());

        // A config.json that is *present but corrupt* (malformed JSON, e.g. a partial download) must
        // error loudly naming the file — NOT silently fall to the dense path (sc-9426 / F-073 sibling).
        let corrupt = dir.join("vae_bad");
        std::fs::create_dir_all(&corrupt).unwrap();
        std::fs::write(corrupt.join("config.json"), b"{ not json").unwrap();
        let err = pipe
            .component_is_packed("vae_bad")
            .expect_err("corrupt config.json must error, not fall to dense");
        assert!(
            format!("{err}").contains("config.json"),
            "the error should name the offending file, got: {err}"
        );
    }

    /// The shared quantizable-loader's three device/dtype-selection regimes (the F-024 de-dup home,
    /// sc-9004). `load_one_quantizable` is the single body behind `load_te_and_dit` (and thus behind
    /// txt2img, edit, and control): the same routing decision every entry point makes. Exercised on CPU
    /// with a stub module that records the device its VarBuilder was built on:
    /// - **no quant** → build on the configured device (`self.device`), no staging.
    /// - **dense tier + quant** → stage dense on the CPU, then quantize onto `self.device`.
    /// - **packed tier + quant** → build directly on `self.device` (no dense CPU staging, sc-9087).
    ///
    /// The dtype passed to the builder is always `self.dtype` (f32) regardless of regime — the loaded
    /// weights + dtype/device stay byte-identical per site (the invariant the de-dup must preserve).
    #[test]
    fn load_one_quantizable_selects_device_per_tier() {
        use candle_gen::candle_core::safetensors;
        use std::collections::HashMap;

        /// Records the device + dtype its VarBuilder was constructed on, and whether the post-build
        /// `quantize` hook ran (the CPU-stage → quantize-onto-GPU / packed handoff).
        struct Probe {
            device: Device,
            dtype: DType,
            quantized: std::cell::Cell<bool>,
        }

        let dir_tmp = tempfile::tempdir().unwrap();
        let dir = dir_tmp.path().to_path_buf();
        // A one-tensor safetensors shard so `component_vb_on` mmaps successfully for either component.
        let write_shard = |sub: &str, packed: bool| {
            let comp = dir.join(sub);
            std::fs::create_dir_all(&comp).unwrap();
            let mut map = HashMap::new();
            map.insert(
                "w".to_string(),
                Tensor::zeros((2, 2), DType::F32, &Device::Cpu).unwrap(),
            );
            safetensors::save(&map, comp.join("model.safetensors")).unwrap();
            if packed {
                std::fs::write(
                    comp.join("config.json"),
                    r#"{"quantization": {"bits": 4, "group_size": 64}}"#,
                )
                .unwrap();
            }
        };

        // The build closure just records what the loader handed it; the quantize closure records that it
        // ran and on which device (the CPU-stage → quantize-onto-GPU handoff).
        let build = |vb: VarBuilder| -> CResult<Probe> {
            Ok(Probe {
                device: vb.device().clone(),
                dtype: vb.dtype(),
                quantized: std::cell::Cell::new(false),
            })
        };
        let quantize = |m: &mut Probe, _q: Quant, _d: &Device| -> CResult<()> {
            m.quantized.set(true);
            Ok(())
        };

        // no quant → configured device, no staging call.
        write_shard("text_encoder", false);
        let pipe = Pipeline::load(Flux2Variant::Klein9b, None, &dir, &Device::Cpu, None);
        let p = pipe
            .load_one_quantizable("text_encoder", None, build, quantize)
            .unwrap();
        assert!(matches!(p.device, Device::Cpu));
        assert_eq!(p.dtype, DType::F32);
        assert!(!p.quantized.get(), "no-quant path must not quantize");

        // dense tier + quant → the builder sees the CPU (staging), then quantize runs onto the device.
        let dense = Pipeline::load(Flux2Variant::Dev, Some(Quant::Q4), &dir, &Device::Cpu, None);
        let p = dense
            .load_one_quantizable("text_encoder", Some(Quant::Q4), build, quantize)
            .unwrap();
        assert!(
            matches!(p.device, Device::Cpu),
            "dense-tier build stages on CPU"
        );
        assert!(
            p.quantized.get(),
            "dense-tier + quant must quantize onto the device"
        );

        // packed tier + quant → the builder sees the configured device directly (no dense staging).
        write_shard("transformer", true);
        let packed = Pipeline::load(Flux2Variant::Dev, Some(Quant::Q4), &dir, &Device::Cpu, None);
        let p = packed
            .load_one_quantizable("transformer", Some(Quant::Q4), build, quantize)
            .unwrap();
        assert!(
            matches!(p.device, Device::Cpu),
            "packed-tier build lands on the configured device (no CPU staging step)"
        );
        assert!(
            p.quantized.get(),
            "packed-tier still runs the (no-op on projections) quantize to carry dense leaves"
        );
    }

    /// DENSE_TE invariant (epic 8506, sc-11031): klein quantizes ONLY its DiT and keeps the 8B Qwen3 TE
    /// dense bf16 in every tier, so `te_quant` is `None` for klein at ANY DiT quant; dev folds its Mistral
    /// TE with the DiT, so `te_quant` tracks `self.quant`.
    #[test]
    fn te_quant_keeps_klein_text_encoder_dense() {
        let dir_tmp = tempfile::tempdir().unwrap();
        let dir = dir_tmp.path().to_path_buf();
        for q in [None, Some(Quant::Q4), Some(Quant::Q8)] {
            let klein = Pipeline::load(Flux2Variant::Klein9b, q, &dir, &Device::Cpu, None);
            assert_eq!(klein.te_quant(), None, "klein TE stays dense at {q:?}");
            let dev = Pipeline::load(Flux2Variant::Dev, q, &dir, &Device::Cpu, None);
            assert_eq!(dev.te_quant(), q, "dev folds its TE with the DiT at {q:?}");
        }
    }

    /// `load_te_and_dit` is a thin delegation to `load_quantizable` with the default TE+DiT builders —
    /// the single home the three entry points (txt2img/edit/control) now share (F-024, sc-9004). It
    /// surfaces the underlying loader error (here: a snapshot missing the `text_encoder/` component)
    /// unchanged, confirming the delegation is wired without needing real 32B weights.
    #[test]
    fn load_te_and_dit_surfaces_missing_component() {
        let dir_tmp = tempfile::tempdir().unwrap();
        let dir = dir_tmp.path().to_path_buf();
        // No component dirs written → the shared loader must error on the missing text_encoder/.
        let pipe = Pipeline::load(Flux2Variant::Klein9b, None, &dir, &Device::Cpu, None);
        let err = pipe
            .load_te_and_dit()
            .err()
            .expect("missing components")
            .to_string();
        assert!(
            err.contains("text_encoder"),
            "delegation surfaces the loader's missing-component error, got: {err}"
        );
    }

    /// The in-place ComfyUI DiT entry point (epic 10451 Phase 2e, sc-10680) builds a lazy dev generator
    /// without touching weights: it stamps the dev descriptor + carries the DiT file, and the resident
    /// snapshot dir is the root supplying the TE/VAE/tokenizer. Loading is lazy, so this asserts the
    /// plumbing on CPU with no weights (the render itself is GPU-validated separately).
    #[test]
    fn load_from_comfyui_dit_builds_lazy_dev_generator() {
        let g = load_from_comfyui_dit(
            "/tree/diffusion_models/flux2_dev_fp8mixed.safetensors",
            "/snap/flux2-dev",
            Some(Quant::Q8),
        )
        .expect("comfyui dev generator builds lazily");
        assert_eq!(g.descriptor().id, FLUX2_DEV_ID);
        assert_eq!(g.descriptor().family, "flux2");
        assert_eq!(g.descriptor().backend, "candle");
        assert_eq!(g.descriptor().modality, Modality::Image);
    }

    #[test]
    fn staged_comfyui_dit_loader_preserves_the_selected_single_file() {
        let selected = PathBuf::from("/selected/flux2-comfyui.safetensors");
        let pipe = Pipeline::load_comfyui(
            None,
            Path::new("/missing-snapshot"),
            &Device::Cpu,
            selected.clone(),
        );
        let error = pipe
            .load_dit_seq()
            .err()
            .expect("the selected fixture path is deliberately absent")
            .to_string();
        assert!(
            error.contains(&selected.display().to_string()),
            "request staging must read the selected ComfyUI DiT, got: {error}"
        );
    }

    #[test]
    fn load_rejects_single_file_source() {
        let spec = LoadSpec::new(WeightsSource::File("/tmp/flux2.safetensors".into()));
        let err = load_klein(&spec)
            .err()
            .expect("expected an error")
            .to_string();
        assert!(err.contains("snapshot directory"), "got: {err}");
    }

    /// Image construction is lazy and the legacy load policy no longer selects lifecycle behavior.
    #[test]
    fn load_policy_is_not_a_residency_authority() {
        let resident = LoadSpec::new(WeightsSource::Dir("/snap".into()));
        let legacy_staged = LoadSpec::new(WeightsSource::Dir("/snap".into()))
            .with_offload_policy(gen_core::OffloadPolicy::Sequential);
        for spec in [&resident, &legacy_staged] {
            assert!(load_dev(spec).is_ok());
            assert!(load_klein(spec).is_ok());
        }
    }

    const BOUNDED_DECODE_MAX_ABS_ERROR: f64 = 2.0;

    #[derive(Clone, Copy, Debug, PartialEq)]
    struct RgbParityMetrics {
        changed_fraction: f64,
        maximum_error: u8,
        mean_error: f64,
        root_mean_square_error: f64,
        psnr_db: f64,
    }

    fn rgb_parity_metrics(reference: &[u8], candidate: &[u8]) -> Result<RgbParityMetrics, String> {
        if reference.len() != candidate.len() {
            return Err(format!(
                "RGB length mismatch: resident={} candidate={}",
                reference.len(),
                candidate.len()
            ));
        }
        if reference.is_empty() {
            return Err("RGB parity requires a non-empty output".to_owned());
        }
        let mut changed = 0u64;
        let mut maximum = 0u8;
        let mut absolute_sum = 0u64;
        let mut square_sum = 0u64;
        for (&resident, &bounded) in reference.iter().zip(candidate) {
            let error = resident.abs_diff(bounded);
            changed += u64::from(error != 0);
            maximum = maximum.max(error);
            absolute_sum += u64::from(error);
            square_sum += u64::from(error) * u64::from(error);
        }
        let count = reference.len() as f64;
        let mean_error = absolute_sum as f64 / count;
        let root_mean_square_error = (square_sum as f64 / count).sqrt();
        let psnr_db = if root_mean_square_error == 0.0 {
            f64::INFINITY
        } else {
            20.0 * (255.0 / root_mean_square_error).log10()
        };
        Ok(RgbParityMetrics {
            changed_fraction: changed as f64 / count,
            maximum_error: maximum,
            mean_error,
            root_mean_square_error,
            psnr_db,
        })
    }

    fn output_parity_contract(
        strategy: gen_core::MemoryStrategy,
    ) -> gen_core::MemoryParityContract {
        if strategy >= gen_core::MemoryStrategy::BoundedDecode {
            gen_core::MemoryParityContract::Tolerance {
                metric: "rgb8_max_abs_error".to_owned(),
                maximum_error: BOUNDED_DECODE_MAX_ABS_ERROR,
            }
        } else {
            gen_core::MemoryParityContract::Exact
        }
    }

    fn assess_output_parity(
        strategy: gen_core::MemoryStrategy,
        reference: &[u8],
        output: &[u8],
    ) -> (
        gen_core::MemoryParityContract,
        gen_core::MemoryParityResult,
        Option<RgbParityMetrics>,
    ) {
        let bounded = strategy >= gen_core::MemoryStrategy::BoundedDecode;
        let contract = output_parity_contract(strategy);
        let metrics = match rgb_parity_metrics(reference, output) {
            Ok(metrics) => metrics,
            Err(reason) => {
                return (
                    contract,
                    gen_core::MemoryParityResult::Failed { reason },
                    None,
                );
            }
        };
        let passed = if bounded {
            f64::from(metrics.maximum_error) <= BOUNDED_DECODE_MAX_ABS_ERROR
        } else {
            metrics.maximum_error == 0
        };
        let result = if passed {
            gen_core::MemoryParityResult::Passed
        } else {
            gen_core::MemoryParityResult::Failed {
                reason: format!(
                    "{} parity failed: max_abs={} mean_abs={:.12} rmse={:.12}",
                    if bounded { "bounded decode" } else { "exact" },
                    metrics.maximum_error,
                    metrics.mean_error,
                    metrics.root_mean_square_error,
                ),
            }
        };
        (contract, result, Some(metrics))
    }

    #[cfg(feature = "cuda")]
    fn measured_output_parity(
        strategy: gen_core::MemoryStrategy,
        output: &[u8],
    ) -> (gen_core::MemoryParityContract, gen_core::MemoryParityResult) {
        let contract = output_parity_contract(strategy);
        let Ok(reference_path) = std::env::var("FLUX2_PARITY_REFERENCE") else {
            return (contract, gen_core::MemoryParityResult::NotRun);
        };
        let reference = std::fs::read(&reference_path).unwrap_or_else(|error| {
            panic!("read FLUX2_PARITY_REFERENCE={reference_path}: {error}")
        });
        let (contract, result, metrics) = assess_output_parity(strategy, &reference, output);
        if let Some(metrics) = metrics {
            eprintln!(
                "MEMORY_PARITY_DIAGNOSTIC strategy={strategy:?} reference={} changed_fraction={:.12} max_abs={} mean_abs={:.12} rmse={:.12} psnr_db={:.12}",
                reference_path,
                metrics.changed_fraction,
                metrics.maximum_error,
                metrics.mean_error,
                metrics.root_mean_square_error,
                metrics.psnr_db,
            );
        }
        (contract, result)
    }

    #[test]
    fn rgb_parity_metrics_cover_exact_bounded_and_shape_failure() {
        let exact = rgb_parity_metrics(&[0, 1, 255], &[0, 1, 255]).unwrap();
        assert_eq!(exact.maximum_error, 0);
        assert_eq!(exact.changed_fraction, 0.0);
        assert!(exact.psnr_db.is_infinite());

        let bounded = rgb_parity_metrics(&[0, 10, 255, 100], &[2, 9, 254, 100]).unwrap();
        assert_eq!(bounded.maximum_error, 2);
        assert_eq!(bounded.changed_fraction, 0.75);
        assert_eq!(bounded.mean_error, 1.0);
        assert!((bounded.root_mean_square_error - (1.5f64).sqrt()).abs() < 1e-12);
        assert!(rgb_parity_metrics(&[0], &[0, 1]).is_err());

        for strategy in [
            gen_core::MemoryStrategy::Resident,
            gen_core::MemoryStrategy::StagedResidency,
        ] {
            let (contract, result, _) = assess_output_parity(strategy, &[1, 2], &[1, 2]);
            assert_eq!(contract, gen_core::MemoryParityContract::Exact);
            assert_eq!(result, gen_core::MemoryParityResult::Passed);
        }
        let (contract, result, _) =
            assess_output_parity(gen_core::MemoryStrategy::BoundedDecode, &[0, 10], &[2, 9]);
        assert_eq!(
            contract,
            gen_core::MemoryParityContract::Tolerance {
                metric: "rgb8_max_abs_error".to_owned(),
                maximum_error: 2.0,
            }
        );
        assert_eq!(result, gen_core::MemoryParityResult::Passed);
        let (_, result, _) = assess_output_parity(
            gen_core::MemoryStrategy::BoundedTransformerResidency,
            &[0],
            &[3],
        );
        assert!(matches!(
            result,
            gen_core::MemoryParityResult::Failed { .. }
        ));
        let (_, result, metrics) =
            assess_output_parity(gen_core::MemoryStrategy::Resident, &[0], &[0, 1]);
        assert!(matches!(
            result,
            gen_core::MemoryParityResult::Failed { .. }
        ));
        assert!(metrics.is_none());
    }

    /// Shared body for the FLUX.2 offload A/B harnesses (epic 10765 Phase 1c, sc-10868 dev / sc-11008
    /// klein). Loads `label`'s snapshot from the `dir_env` env var and runs ONE probed 1024²
    /// generation whose residency mode is carried by `GenerationMemory::stage_residency`, calibrated
    /// with `FLUX2_OFFLOAD_MODE=request-staged`; it prints one strict `MEMORY_EVIDENCE_V1` record and
    /// writes the raw RGB pixels to `FLUX2_OUT`. Run each rung in a SEPARATE process, setting
    /// `FLUX2_PARITY_REFERENCE` to the resident raw RGB after that first run. Staged residency must be
    /// byte-exact; decode-composed rungs use the provider-owned `rgb8_max_abs_error <= 2` contract and
    /// also print changed fraction, mean absolute error, RMSE, and PSNR. The staged peak must be
    /// materially lower because the dense text encoder is dropped before the DiT loads. Separate
    /// processes are REQUIRED — candle's cudarc caching
    /// allocator never returns pages to the driver, so a second in-process run reuses the first run's
    /// pool and reads the same peak. `honor_quant` reads `FLUX2_QUANT` (q4/q8) — set for both dev (folds
    /// the 32B DiT + Mistral TE) and klein (sc-11031: folds only the 9B DiT, Qwen3 TE stays dense);
    /// unset `FLUX2_QUANT` loads the dense tier. Needs a real-file (hardlink-staged, not raw-HF-symlink)
    /// snapshot in `dir_env` and a CUDA device.
    #[cfg(feature = "cuda")]
    fn run_probed_offload_ab(
        label: &str,
        dir_env: &str,
        load: fn(&LoadSpec) -> gen_core::Result<Box<dyn Generator>>,
        honor_quant: bool,
        steps: u32,
    ) {
        let dir = std::env::var(dir_env).unwrap_or_else(|_| {
            panic!("set {dir_env} to a real-file (hardlink-staged) {label} snapshot")
        });
        let out = std::env::var("FLUX2_OUT").expect("set FLUX2_OUT to the pixel-dump path");
        let mut spec = LoadSpec::new(WeightsSource::Dir(dir.into()));
        // honor FLUX2_QUANT (q4/q8) when set, else load dense. dev fits only quantized (bf16 is a
        // fixture); klein (sc-11031) quantizes its DiT and keeps the Qwen3 TE dense, or loads bf16 dense.
        if honor_quant {
            spec = match std::env::var("FLUX2_QUANT")
                .unwrap_or_default()
                .to_lowercase()
                .as_str()
            {
                "q4" => spec.with_quant(Quant::Q4),
                "q8" => spec.with_quant(Quant::Q8),
                _ => spec,
            };
        }
        spec.load_shape = gen_core::LoadShape::DeferredMaterialization;
        let rung = std::env::var("FLUX2_MEMORY_RUNG").unwrap_or_else(|_| {
            if std::env::var("FLUX2_OFFLOAD_MODE").is_ok_and(|mode| mode == "request-staged") {
                "staged".to_owned()
            } else {
                "resident".to_owned()
            }
        });
        let strategy = match rung.as_str() {
            "resident" => gen_core::MemoryStrategy::Resident,
            "staged" => gen_core::MemoryStrategy::StagedResidency,
            "decode" => gen_core::MemoryStrategy::BoundedDecode,
            "attention" => gen_core::MemoryStrategy::BoundedAttention,
            "blocks" => gen_core::MemoryStrategy::BoundedTransformerResidency,
            value => panic!("unsupported FLUX2_MEMORY_RUNG={value}"),
        };
        let mut req = GenerationRequest {
            prompt: "a rusty robot holding a lit candle, studio lighting".into(),
            width: 1024,
            height: 1024,
            steps: Some(steps),
            seed: Some(42),
            count: 1,
            ..Default::default()
        };
        assert!(
            candle_gen::testkit::reset_cuda_mempool_high_water(0),
            "reset CUDA live-allocation high-water"
        );
        let mut probe = candle_gen::testkit::VramProbe::start_rendered();
        let load_phase = probe.phase();
        let g = load(&spec).unwrap_or_else(|e| panic!("load {label}: {e}"));
        probe.end_load(load_phase);
        let contract = g
            .memory_strategy_contract()
            .expect("FLUX.2-dev memory contract");
        let tier = memory_strategy::resolved_numeric_tier(&spec).expect("numeric tier");
        let context = gen_core::standard_memory_behavior_context(
            contract,
            strategy,
            tier,
            gen_core::MemoryBehaviorRoute {
                mode: gen_core::MemoryMode::TextToImage,
                reference_count: 0,
                use_pid: false,
                has_phases: false,
                overlay: None,
            },
        )
        .expect("memory context");
        assert!(matches!(
            g.memory_strategy_safety_check(&context),
            gen_core::MemorySafetyDecision::Accept
        ));
        let mut scope = g
            .begin_memory_strategy_request(&context)
            .expect("begin memory request")
            .expect("memory request scope");
        scope
            .configure_request(&mut req)
            .expect("configure memory request");
        let generate_phase = probe.phase();
        let output = g.generate(&req, &mut |_| {}).expect("generate");
        probe.end_gen(generate_phase);
        scope
            .finish(gen_core::MemoryRunOutcome::Complete)
            .expect("finish memory request");
        let report = probe.report().assert_trustworthy(1.0);
        let live_peak_bytes = candle_gen::testkit::cuda_mempool_used_high_bytes(0)
            .expect("read CUDA live-allocation high-water");
        assert!(
            live_peak_bytes > 0,
            "CUDA live-allocation peak must be positive"
        );
        let img = match output {
            GenerationOutput::Images(mut v) => v.remove(0),
            other => panic!("expected images, got {other:?}"),
        };
        std::fs::write(&out, &img.pixels).expect("write pixels");
        let (parity, parity_result) = measured_output_parity(strategy, &img.pixels);
        let parity_failure = match &parity_result {
            gen_core::MemoryParityResult::Failed { reason } => Some(reason.clone()),
            _ => None,
        };
        eprintln!(
            "{}",
            candle_gen::testkit::memory_evidence_v1_line_with_parity(
                candle_gen::testkit::MemoryEvidenceProbe {
                    resolved_route: label,
                    declared_calibration: candle_gen::testkit::expected_memory_calibration(
                        spec.load_shape,
                    ),
                    observed_calibration: contract.calibration.clone().expect("calibration"),
                    tier: memory_strategy::resolved_numeric_tier(&spec)
                        .expect("resolved numeric tier"),
                    load_shape: spec.load_shape,
                    mode: gen_core::MemoryMode::TextToImage,
                    overlay: None,
                    geometry: gen_core::MemoryGeometry {
                        width: req.width,
                        height: req.height,
                        batch: req.count,
                        frames: 1,
                        reference_count: 0,
                    },
                    strategy,
                    engaged_composition: contract.engaged_composition(strategy),
                    parameters: context.selection.parameters,
                    observed_peak_bytes: live_peak_bytes,
                    harness_version: "candle-flux2-memory-ladder-v1",
                    output_bytes: &img.pixels,
                },
                parity,
                parity_result,
            )
        );
        eprintln!(
            "MEMORY_EVIDENCE_DIAGNOSTIC gpu={} {report} bytes={} {}x{} out={out}",
            candle_gen::testkit::probe_gpu(),
            img.pixels.len(),
            img.width,
            img.height
        );
        if let Some(reason) = parity_failure {
            panic!("FLUX.2 memory-ladder output parity failed: {reason}");
        }
    }

    /// Sequential-residency GPU validation (epic 10765 Phase 1c, sc-10868) for FLUX.2-**dev** (Mistral
    /// TE, guidance-distilled 32B). See [`run_probed_offload_ab`] for the A/B protocol. Ignored by
    /// default; needs a hardlink-staged FLUX.2-dev snapshot in `FLUX2_DEV_DIR`, a `FLUX2_QUANT` of
    /// `q4`/`q8` (the 32B needs it — omit only for a dense fixture), and a CUDA device.
    #[cfg(feature = "cuda")]
    #[test]
    #[ignore]
    fn flux2_dev_probed_generate_for_offload_ab() {
        run_probed_offload_ab("flux2_dev", "FLUX2_DEV_DIR", load_dev, true, 8);
    }

    /// klein sibling of [`flux2_dev_probed_generate_for_offload_ab`] (epic 10765 Phase 1c follow-up,
    /// sc-11008 bf16 / sc-11031 q4/q8). Same A/B protocol, but loads FLUX.2-**klein**-9B (Qwen3 TE + 9B
    /// DiT) from `FLUX2_KLEIN_DIR` at 4 steps CFG-free. `honor_quant = true` reads `FLUX2_QUANT` (q4/q8)
    /// now that klein quantizes its DiT on-the-fly off the dense BFL snapshot (sc-11031) — the Qwen3 TE
    /// stays dense bf16 in every tier; omit `FLUX2_QUANT` for the bf16-dense tier. Directly captures the
    /// klein resident-vs-sequential peak per tier that sc-10920 could only arch-scale, validating that
    /// dropping the DENSE ~16 GB Qwen3 TE before the 9B DiT loads is the sequential floor. Ignored by
    /// default; needs a hardlink-staged klein diffusers snapshot (`text_encoder/` = Qwen3,
    /// `transformer/`, `vae/`, `tokenizer/`) in `FLUX2_KLEIN_DIR` and a CUDA device.
    #[cfg(feature = "cuda")]
    #[test]
    #[ignore]
    fn flux2_klein_probed_generate_for_offload_ab() {
        run_probed_offload_ab("flux2_klein_9b", "FLUX2_KLEIN_DIR", load_klein, true, 4);
    }

    /// Reference-bearing companion to [`flux2_klein_probed_generate_for_offload_ab`]. One process
    /// exercises one exact route/rung coordinate through the bespoke context-bearing loader. Set
    /// `FLUX2_KLEIN_ROUTE` to `edit`, `reference`, `character`, or `style`; set
    /// `FLUX2_MEMORY_RUNG` to `resident`, `staged`, `decode`, `attention`, or `blocks`. The reference
    /// is any image format accepted by the `image` crate at `FLUX2_KLEIN_REF`; output/parity use the same `FLUX2_OUT` and
    /// `FLUX2_PARITY_REFERENCE` protocol as the registered text route. Invoke the coordinates in
    /// separate serial GPU0 processes because the CUDA allocator retains its pool.
    #[cfg(feature = "cuda")]
    #[test]
    #[ignore = "needs FLUX2_KLEIN_DIR + FLUX2_KLEIN_REF + FLUX2_OUT and a CUDA GPU"]
    fn flux2_klein_reference_routes_probed_memory_ladder() {
        use candle_gen::testkit::env_path;

        let root = env_path("FLUX2_KLEIN_DIR");
        let reference_path = env_path("FLUX2_KLEIN_REF");
        let reference_rgb = image::open(&reference_path)
            .unwrap_or_else(|error| panic!("decode {}: {error}", reference_path.display()))
            .to_rgb8();
        let reference = Image {
            width: reference_rgb.width(),
            height: reference_rgb.height(),
            pixels: reference_rgb.into_raw(),
        };
        let out = std::env::var("FLUX2_OUT").expect("set FLUX2_OUT to the pixel-dump path");
        let route = std::env::var("FLUX2_KLEIN_ROUTE").unwrap_or_else(|_| "edit".to_owned());
        let (mode, route_label) = match route.as_str() {
            "edit" => (gen_core::MemoryMode::Edit, "flux2_klein_9b_edit"),
            "reference" => (gen_core::MemoryMode::Edit, "flux2_klein_9b_reference"),
            "character" => (
                gen_core::MemoryMode::Other("character_image".to_owned()),
                "flux2_klein_9b_character",
            ),
            "style" => (
                gen_core::MemoryMode::Other("style_variations".to_owned()),
                "flux2_klein_9b_style",
            ),
            value => panic!("unsupported FLUX2_KLEIN_ROUTE={value}"),
        };
        let rung = std::env::var("FLUX2_MEMORY_RUNG").unwrap_or_else(|_| "resident".to_owned());
        let strategy = match rung.as_str() {
            "resident" => gen_core::MemoryStrategy::Resident,
            "staged" => gen_core::MemoryStrategy::StagedResidency,
            "decode" => gen_core::MemoryStrategy::BoundedDecode,
            "attention" => gen_core::MemoryStrategy::BoundedAttention,
            "blocks" => gen_core::MemoryStrategy::BoundedTransformerResidency,
            value => panic!("unsupported FLUX2_MEMORY_RUNG={value}"),
        };
        let mut spec = LoadSpec::new(WeightsSource::Dir(root.clone()));
        spec = match std::env::var("FLUX2_QUANT")
            .unwrap_or_default()
            .to_lowercase()
            .as_str()
        {
            "q4" => spec.with_quant(Quant::Q4),
            "q8" => spec.with_quant(Quant::Q8),
            _ => spec,
        };
        spec.load_shape = gen_core::LoadShape::DeferredMaterialization;
        let contract = memory_strategy::klein_provider_contract(&spec).expect("Klein contract");
        let tier = memory_strategy::resolved_numeric_tier(&spec).expect("numeric tier");
        let context = gen_core::standard_memory_behavior_context(
            &contract,
            strategy,
            tier,
            gen_core::MemoryBehaviorRoute {
                mode: mode.clone(),
                reference_count: 1,
                use_pid: false,
                has_phases: false,
                overlay: None,
            },
        )
        .expect("reference-route memory context");
        memory_strategy::validate_context(&contract, &context, tier.quant)
            .expect("reference-route admission");
        let req = Flux2EditRequest {
            prompt: "turn the reference into a cinematic portrait with warm studio lighting".into(),
            width: 1024,
            height: 1024,
            steps: 4,
            guidance: 1.0,
            seed: 42,
            ..Default::default()
        };

        assert!(
            candle_gen::testkit::reset_cuda_mempool_high_water(0),
            "reset CUDA live-allocation high-water"
        );
        let mut probe = candle_gen::testkit::VramProbe::start_rendered();
        let load_phase = probe.phase();
        let model =
            Flux2Edit::load_klein_with_memory_context(&Flux2EditPaths { root }, &spec, &context)
                .expect("load context-bound Klein edit");
        probe.end_load(load_phase);

        let mut stale = context.clone();
        stale.calibration_fingerprint.push_str("-stale");
        assert!(
            model
                .generate_with_memory_context(
                    &stale,
                    &req,
                    std::slice::from_ref(&reference),
                    &mut |_| {}
                )
                .is_err(),
            "fingerprint/context mutation must fail before generation"
        );

        let generate_phase = probe.phase();
        let img = model
            .generate_with_memory_context(
                &context,
                &req,
                std::slice::from_ref(&reference),
                &mut |_| {},
            )
            .expect("generate context-bound Klein reference route");
        probe.end_gen(generate_phase);
        let report = probe.report().assert_trustworthy(1.0);
        let live_peak_bytes = candle_gen::testkit::cuda_mempool_used_high_bytes(0)
            .expect("read CUDA live-allocation high-water");
        assert!(
            live_peak_bytes > 0,
            "CUDA live-allocation peak must be positive"
        );
        std::fs::write(&out, &img.pixels).expect("write pixels");
        let (parity, parity_result) = measured_output_parity(strategy, &img.pixels);
        let parity_failure = match &parity_result {
            gen_core::MemoryParityResult::Failed { reason } => Some(reason.clone()),
            _ => None,
        };
        eprintln!(
            "{}",
            candle_gen::testkit::memory_evidence_v1_line_with_parity(
                candle_gen::testkit::MemoryEvidenceProbe {
                    resolved_route: "flux2_klein_9b",
                    declared_calibration: candle_gen::testkit::expected_memory_calibration(
                        spec.load_shape,
                    ),
                    observed_calibration: contract.calibration.clone().expect("calibration"),
                    tier,
                    load_shape: spec.load_shape,
                    mode,
                    overlay: None,
                    geometry: context.geometry,
                    strategy,
                    engaged_composition: contract.engaged_composition(strategy),
                    parameters: context.selection.parameters,
                    observed_peak_bytes: live_peak_bytes,
                    harness_version: "candle-flux2-klein-reference-memory-ladder-v1",
                    output_bytes: &img.pixels,
                },
                parity,
                parity_result,
            )
        );
        eprintln!(
            "MEMORY_EVIDENCE_DIAGNOSTIC route={route_label} gpu={} {report} bytes={} {}x{} out={out}",
            candle_gen::testkit::probe_gpu(),
            img.pixels.len(),
            img.width,
            img.height,
        );
        if let Some(reason) = parity_failure {
            panic!("FLUX.2 reference memory-ladder output parity failed: {reason}");
        }
    }
}
