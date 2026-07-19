//! Krea 2 text-to-image pipeline (sc-7580/sc-7582) — tokenize → Qwen3-VL-4B condition-encode (the
//! 12-layer select stack) → DiT (text_fusion aggregator + single-stream denoise) → Qwen-Image VAE
//! decode. Port of `mlx-gen-krea`'s `pipeline.rs` (the reference `sampling.py::sample`). Two render
//! surfaces share this one pipeline:
//! - **Turbo** ([`render`]) — the distilled few-step **CFG-free** path (one DiT forward/step).
//! - **Raw** ([`render_base`], sc-9994 / epic 9992) — the undistilled 12B DiT with **true
//!   classifier-free guidance** (two DiT forwards/step: cond vs uncond) + optional user negative
//!   prompt at 52 steps, resolution-dynamic mu ([`base_schedule`]). The Boogu base/turbo precedent.
//!
//! **CFG-free (Turbo).** The TDM distillation baked the guided velocity into the weights, so there is
//! no unconditional branch (`guidance == 0` in the reference) — one DiT forward per step. Per-sample
//! `B = 1`: one prompt → no padding → the DiT runs the full valid context.
//!
//! **Rectified-flow v-param Euler.** The DiT consumes the raw sigma as its timestep
//! ([`TimestepConvention::Sigma`]; it scales ×1000 internally) and predicts the flow velocity
//! directly, so the core [`candle_gen::run_flow_sampler`] Euler step `x + v·(σ_{i+1} − σ_i)` is exactly
//! the reference `img += (tprev − tcurr)·v`. The native exponential-mu schedule
//! ([`crate::schedule::turbo_sigmas`]) is the byte-exact default; a per-generation curated
//! sampler/scheduler (epic 7114) reshapes over the same mu.

use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex};

use candle_gen::candle_core::{DType, Device, IndexOp, Tensor};
use candle_gen::gen_core::imageops::{resize_bicubic_u8, resize_lanczos_u8};
use candle_gen::gen_core::sampling::TimestepConvention;
use candle_gen::gen_core::{self, AdapterSpec, GenerationRequest, Image, PidWeights, Progress};
use candle_gen::quant::Int8Context;
use candle_gen::{CandleError, LatentDecoder, Result};
use candle_gen_pid::PidEngine;
use candle_gen_qwen_image::vae::QwenVae;
use rand::{rngs::StdRng, SeedableRng};

/// The PiD backbone (latent-space) tag for Krea (epic 7840 / sc-7853). Krea reuses the Qwen-Image VAE,
/// so its latent space is `qwenimage` — the same `2kto4k` 4× student Qwen-Image resolves.
const PID_BACKBONE: &str = "qwenimage";

use crate::config::Krea2Config;
use crate::loader::Weights;
use crate::schedule::{dynamic_mu, krea_sigmas, turbo_sigmas, TURBO_MU, TURBO_STEPS};
use crate::text_encoder::{KreaTeConfig, KreaTextEncoder};
use crate::transformer::Krea2Transformer;
use crate::vae::{load_vae, load_vae_encoder, QwenVaeEncoder};

/// Component dtypes. The Qwen3-VL TE **computes in f32** (parity-grade for this encoder, shared with
/// the ideogram/boogu ports — the select-layer taps feeding the DiT's `TextFusionTransformer` are
/// accumulation-sensitive) but its weights are **stored bf16** ([`TE_STORE_DTYPE`], sc-12828): the
/// hosted TE ships bf16 on disk, so an f32 *store* only widens those exact bf16 values — it carries
/// no extra precision, yet holds ~15.6 GB resident (the single biggest non-DiT resident item, paid on
/// every Krea tier). Storing bf16 and upcasting each projection to f32 per matmul
/// ([`candle_gen::QLinear::forward_upcast`], with the RMSNorm weights loaded f32) is **bit-identical**
/// to the f32-store forward (bf16→f32 widening is exact and every matmul still runs f32) at half the
/// footprint (~8 GB). The 12B DiT runs **bf16** (native on candle's CUDA backend); the Qwen-Image VAE
/// runs **f32** (decode-precision-sensitive).
const DIT_DTYPE: DType = DType::BF16;

/// The TE weight **storage** dtype (sc-12828). Distinct from the encoder's **compute** dtype (f32,
/// enforced inside [`KreaTextEncoder`] by upcasting the embedding to f32 and each projection via
/// `forward_upcast`): the encoder holds bf16 weights and upcasts to f32 at each op, byte-identical to
/// an f32 store but at half the resident footprint. Applied by [`load_te_weights`] **only when the
/// snapshot's TE actually ships bf16 on disk** — the premise the bit-identity rests on — else that
/// loader keeps an f32 store (no silent truncation). The training / ControlNet paths deliberately keep
/// their own f32 TE load (they are not the resident-during-render path this halves).
const TE_STORE_DTYPE: DType = DType::BF16;

/// VAE spatial downscale (the latent is image/8 per side) and latent channel count.
const SPATIAL_SCALE: u32 = 8;
const LATENT_CHANNELS: usize = 16;

/// Raw (undistilled, full-CFG) generation defaults — the reference `sampling.py` Raw preset (sc-7566
/// spike), mirroring mlx-gen-krea `DEFAULT_RAW_STEPS` / `DEFAULT_RAW_GUIDANCE` (sc-9994). 52 steps,
/// guidance 3.5, resolution-dynamic mu ([`base_schedule`]); the SceneWorks manifest `default_steps` /
/// `defaults.guidanceScale` mirror these.
pub const RAW_STEPS: usize = 52;
pub const RAW_GUIDANCE: f32 = 3.5;

/// The image-edit reference cap (epic 10871 / sc-10878). The edit LoRA was trained on **one or two**
/// references in a **fixed order** — image 1 (required) + image 2 (optional), either can be a person —
/// and the authors note swapping
/// the order degrades results. The engine's `forward_edit` is generic over N references (each at its own
/// RoPE frame), but the trained contract is 1..=2, so the pipeline caps here.
pub const MAX_EDIT_REFERENCES: usize = 2;

/// Max prompt tokens the Qwen3-VL RoPE table is sized for (generous; Krea prompts + the 34-token
/// template prefix are short). Enforced up front by [`crate::tokenizer::KreaTokenizer::encode_prompt`]
/// so an over-length prompt returns a clear length error instead of an opaque tensor-shape error deep
/// in the condition encoder (sc-9047).
///
/// The **single** canonical cap for the whole crate (sc-11205 / F-120): the inference pipeline, the LoRA
/// trainer (`crate::training`), the ControlNet provider/trainer, and the `krea-control-*` example
/// binaries all import THIS constant. Keeping one definition means raising the cap can never leave the
/// inference and training/control lanes sized differently — a mismatch that surfaced only as the opaque
/// `narrow` error sc-9047 eliminated. `pub` (not `pub(crate)`) so the example crates — compiled as
/// separate crates against this library — can import it too.
pub const MAX_TEXT_TOKENS: usize = 1024;

/// Max tokens for the **image-grounded edit** conditioning (epic 10871 / sc-10880). Far larger than
/// [`MAX_TEXT_TOKENS`] because the edit template embeds one `<|image_pad|>` per merged vision token —
/// a single ~1 MP reference is ~1000 tokens, and two references push past 2000. Unlike the t2i path,
/// this is NOT bounded by the encoder's RoPE-table size: the grounded [`KreaTextEncoder::forward_with_images`]
/// builds a fresh interleaved-MRoPE table sized to the actual sequence, so the only cap is this guard
/// against a pathologically large reference set. (The t2i cap stays 1024 — short prompts only.)
pub(crate) const MAX_EDIT_TOKENS: usize = 8192;

/// The **text-phase** components (epic 10765 Phase 1c, sc-12089): the tokenizer + the Qwen3-VL-4B text
/// encoder — everything [`encode_prompt_context`] reads, and nothing the denoise/decode tail touches.
/// Split out of [`Components`] so the `Sequential` residency path can load→encode→**drop** it before the
/// 12B DiT materializes. The MLX twin is mlx-gen-krea's `KreaText` (sc-11101).
///
/// Krea's TE (Qwen3-VL-4B) is *smaller* than its DiT — the qwen-image pattern (TE < DiT), not the lens
/// pattern — so dropping it is a real but modest weight-side win (~2.9 GB at Q4); the larger effect is
/// that the denoise activations no longer stack on top of a resident encoder.
///
/// `pub(crate)`, not `pub`: every operation on it ([`encode_prompt_context`], [`encode_base_contexts`])
/// is crate-private, so an exported `KreaText` would be an opaque value a caller could obtain and do
/// nothing with — public surface this crate would then owe compatibility on (`CONTRIBUTING.md`) for a
/// seam the module docs call internal. The mlx-gen twin exports its `KreaText` because that one carries
/// public `from_snapshot` / `quantize` methods; this one carries none.
pub(crate) struct KreaText {
    tok: crate::tokenizer::KreaTokenizer,
    te: KreaTextEncoder,
    root: PathBuf,
    device: Device,
    /// Qwen3-VL vision tower for grounded edit conditioning (sc-12129). Kept lazy so ordinary
    /// txt2img/img2img/control requests neither load nor require the `visual.*` subtree.
    vision: Mutex<Option<Arc<crate::vision::VisionTower>>>,
}

impl KreaText {
    fn vision(&self) -> Result<Arc<crate::vision::VisionTower>> {
        candle_gen::cached(&self.vision, || {
            Ok(Arc::new(crate::vision::load_vision_tower(
                &self.root,
                &self.device,
            )?))
        })
    }
}

/// The **heavy-phase** components (epic 10765 Phase 1c, sc-12089): the single-stream DiT + the
/// Qwen-Image VAE (+ the optional PiD decoder) — everything downstream of the encoded context. Loaded
/// after [`KreaText`] is dropped on the `Sequential` path, so it reuses the encoder's freed allocator
/// pool. The MLX twin is mlx-gen-krea's `KreaHeavy` (sc-11101).
///
/// The VAE stays co-resident with the DiT through decode (it is small relative to the 12B DiT, so
/// splitting them further buys ~nothing) — the qwen-image `load_vae_seq` precedent (sc-10867).
///
/// `pub(crate)` for the same reason as [`KreaText`]: [`render_from_context`] and its siblings are all
/// crate-private, so there is nothing an external holder of this type could do with it.
pub(crate) struct KreaHeavy {
    dit: Krea2Transformer,
    vae: Arc<QwenVae>,
    /// Optional NVIDIA PiD super-resolving decoder (epic 7840 / sc-7853). Under `Resident` it is loaded
    /// once with the cached components whenever `LoadSpec::pid` was set. Under `Sequential` it is loaded
    /// per generate, so [`load_heavy`] takes `use_pid` and leaves this `None` for a request that never
    /// asked for it (F-177). `None` ⇒ the native `QwenVae` decode (the default path).
    pid: Option<Arc<PidEngine>>,
}

pub(crate) enum ResidencyContext {
    Turbo(Tensor),
    Raw {
        context: Tensor,
        negative: Option<Tensor>,
        guidance: f32,
    },
}

pub(crate) struct ResidencyHeavy {
    heavy: KreaHeavy,
    vae_encoder: QwenVaeEncoder,
}

pub(crate) fn load_residency_heavy(
    root: &Path,
    device: &Device,
    adapters: &[AdapterSpec],
    pid_spec: Option<&PidWeights>,
    use_pid: bool,
) -> Result<ResidencyHeavy> {
    Ok(ResidencyHeavy {
        heavy: load_heavy(root, device, adapters, pid_spec, use_pid)?,
        vae_encoder: load_vae_encoder(root, device)?,
    })
}

pub(crate) fn encode_residency(
    text: &KreaText,
    raw: bool,
    req: &GenerationRequest,
) -> Result<ResidencyContext> {
    if raw {
        let guidance = req.guidance.unwrap_or(RAW_GUIDANCE);
        let (context, negative) = encode_base_contexts(text, req, guidance)?;
        Ok(ResidencyContext::Raw {
            context,
            negative,
            guidance,
        })
    } else {
        Ok(ResidencyContext::Turbo(encode_prompt_context(text, req)?))
    }
}

pub(crate) fn render_residency(
    heavy: &ResidencyHeavy,
    context: ResidencyContext,
    req: &GenerationRequest,
    reference: Option<(&Image, Option<f32>)>,
    device: &Device,
    on_progress: &mut dyn FnMut(Progress),
) -> Result<Vec<Image>> {
    match (context, reference) {
        (ResidencyContext::Turbo(context), Some((reference, strength))) => {
            render_img2img_from_context(
                &heavy.heavy,
                &heavy.vae_encoder,
                req,
                reference,
                strength,
                device,
                &context,
                on_progress,
            )
        }
        (ResidencyContext::Turbo(context), None) => {
            render_from_context(&heavy.heavy, req, device, &context, on_progress)
        }
        (
            ResidencyContext::Raw {
                context,
                negative,
                guidance,
            },
            Some((reference, strength)),
        ) => render_base_img2img_from_contexts(
            &heavy.heavy,
            &heavy.vae_encoder,
            req,
            reference,
            strength,
            device,
            &context,
            negative.as_ref(),
            guidance,
            on_progress,
        ),
        (
            ResidencyContext::Raw {
                context,
                negative,
                guidance,
            },
            None,
        ) => render_base_from_contexts(
            &heavy.heavy,
            req,
            device,
            &context,
            negative.as_ref(),
            guidance,
            on_progress,
        ),
    }
}

/// The loaded Krea 2 Turbo components, `Arc`-shared so the generator caches them across `generate`.
///
/// The **`Resident`** aggregate of both phases (`KreaText` + `KreaHeavy`), held co-resident for the
/// whole job and across jobs via the generator's cache — exactly as before the sc-12089 phase split. The
/// split is an internal seam: every public `render*` entry point still takes `&Components` and runs the
/// same encode→denoise→decode body, so the resident path is byte-untouched (zero parity risk).
pub struct Components {
    pub(crate) text: KreaText,
    pub(crate) heavy: KreaHeavy,
}

/// Load all Turbo components from a Krea 2 snapshot (`tokenizer/ text_encoder/ transformer/ vae/`).
///
/// `adapters` (when non-empty) are trained `krea_2_raw` LoRA/LoKr `.safetensors` applied as **forward-
/// time additive residuals** on the DiT's projections (sc-11105 / sc-11720, [`crate::adapters::install_
/// additive`]) — the base (dense mmap or packed) is never folded, so it stays evictable and the adapter
/// equals the old fold to f32 tolerance. The surface spans attention + SwiGLU FFN + the front-end leaves.
/// Empty ⇒ the stock unadapted build.
pub fn load_components(
    root: &Path,
    device: &Device,
    adapters: &[AdapterSpec],
    pid_spec: Option<&PidWeights>,
) -> Result<Components> {
    // Both phases, in the SAME order the pre-split loader used (tokenizer + TE, then DiT + VAE + PiD),
    // so the resident load is byte-identical — the phase fns below are just the two halves named.
    //
    // `use_pid = true`: the resident set is built ONCE and cached across every request, before any
    // `GenerationRequest` exists, so the PiD overlay must be there for whichever later request asks for
    // it. That is the opposite of the `Sequential` path's calculus (see [`load_heavy`]), where the load
    // is per-generate and the request IS in hand.
    let text = load_text(root, device)?;
    let heavy = load_heavy(root, device, adapters, pid_spec, true)?;
    Ok(Components { text, heavy })
}

/// Which PiD spec [`load_heavy`] should actually load: the opted-in spec, but only when this load will
/// use it (F-177).
///
/// `resolve_pid_decoder` already gates the *decode* on `req.use_pid`, so a PiD engine loaded for a
/// request that did not ask for it is never read — under `Resident` that is a harmless one-time cost
/// amortized across every later request, but under `Sequential` it is paid on EVERY generate and the
/// student plus its multi-GB gemma-2-2b caption encoder sit resident through the whole denoise, inside
/// the very peak the path exists to bound.
///
/// Pure so the rule is unit-testable without weights or a GPU (the `img2img_reference` idiom).
fn pid_to_load(pid_spec: Option<&PidWeights>, use_pid: bool) -> Option<&PidWeights> {
    pid_spec.filter(|_| use_pid)
}

/// Load ONLY the text phase — the tokenizer + the Qwen3-VL-4B text encoder (epic 10765 Phase 1c,
/// sc-12089). The same two loads [`load_components`] runs first; factored out so the `Sequential` path
/// can scope them to the encode and drop them before the DiT loads. Mirrors qwen-image's `load_te_seq`
/// (sc-10867).
/// Load the Qwen3-VL text-encoder weights at a **bf16 store** (sc-12828), the resident-TE-halving win —
/// but only when the snapshot's TE actually ships **bf16 on disk** (the hosted tiers: probed here from
/// one tiny norm weight). The bit-identity of bf16-store vs f32-store holds *only* because the disk
/// weights are already bf16, so f32 merely widened them; a snapshot whose TE is a wider dtype keeps its
/// **f32** store rather than being silently truncated to bf16 (the compute is f32 either way — the
/// encoder upcasts each projection — so both render correctly; only the resident footprint differs).
fn load_te_weights(root: &Path, device: &Device) -> Result<Weights> {
    let dir = root.join("text_encoder");
    let w = Weights::from_dir(&dir, device, TE_STORE_DTYPE)?;
    if w.get_native("language_model.layers.0.input_layernorm.weight")?
        .dtype()
        == TE_STORE_DTYPE
    {
        Ok(w)
    } else {
        Ok(Weights::from_dir(&dir, device, DType::F32)?)
    }
}

pub(crate) fn load_text(root: &Path, device: &Device) -> Result<KreaText> {
    let tok = crate::tokenizer::KreaTokenizer::from_snapshot(root, device)?;

    let te_cfg = KreaTeConfig::from_snapshot(root)?;
    let te_w = load_te_weights(root, device)?;
    let te = KreaTextEncoder::load(&te_w, "language_model", &te_cfg, MAX_TEXT_TOKENS)?;

    Ok(KreaText {
        tok,
        te,
        root: root.to_path_buf(),
        device: device.clone(),
        vision: Mutex::new(None),
    })
}

/// Load ONLY the heavy phase — the single-stream DiT (+ additive adapters) + the Qwen-Image VAE + the
/// optional PiD decoder (epic 10765 Phase 1c, sc-12089). The same loads [`load_components`] runs after
/// the text phase, with the identical adapter/PiD handling; factored out so the `Sequential` path can
/// load it AFTER the text phase was dropped, reusing that freed pool. Mirrors qwen-image's
/// `load_transformer_seq` / `load_vae_seq` / `load_pid_seq` (sc-10867).
///
/// **`use_pid` (F-177).** Whether to load the optional PiD student. `load_components` passes `true` (the
/// resident set is cached across requests, so the overlay must be there for whichever request wants it);
/// the `Sequential` paths pass `req.use_pid`, because there the load runs on EVERY generate and the
/// engine stays resident through the whole denoise. Loading it for a request that never asked would add
/// the student **and** its multi-GB gemma-2-2b caption encoder to exactly the peak this path exists to
/// bound — while `resolve_pid_decoder` goes on to return `None` for it, so not a byte of it is read.
/// The mlx-gen twin threads the same flag (`load_krea_heavy(spec, root, id, load_pid)`).
pub(crate) fn load_heavy(
    root: &Path,
    device: &Device,
    adapters: &[AdapterSpec],
    pid_spec: Option<&PidWeights>,
    use_pid: bool,
) -> Result<KreaHeavy> {
    let cfg = Krea2Config::from_snapshot(root)?;
    let dit_w = Weights::from_dir(&root.join("transformer"), device, DIT_DTYPE)?;
    crate::convert::validate_transformer(&dit_w, &cfg)?;
    // Adapters ride as **forward-time additive residuals** on the DiT's projections — on BOTH the packed
    // and the dense tier (sc-11105, additive-everywhere for epic 10765). The base weight is never mutated:
    // instead of reconstructing each adapted projection's dense weight (packed tier) or folding `W += δ`
    // into the mmap (dense tier) — either of which pins an un-evictable in-memory copy — `install_additive`
    // keeps the base an unmutated mmap/packed base and pushes the LoRA/LoKr delta as an unmerged residual,
    // so the offload/eviction path can drop-and-restore it cheaply. It equals the old fold to f32 tolerance
    // (~1 ULP). A non-empty spec that matches no target is a hard error (the worker then falls back rather
    // than silently rendering unadapted).
    let mut dit = Krea2Transformer::load(&dit_w, &cfg)?;
    if !adapters.is_empty() {
        crate::adapters::install_additive(&mut dit, adapters)?;
    }

    let vae = load_vae(root, device)?;

    // The optional PiD super-resolving decoder (epic 7840 / sc-7853), loaded when the caller opted in via
    // `LoadSpec::pid` AND this load will actually use it (F-177 — see the `use_pid` doc above; under
    // `Sequential` this whole fn runs per generate). Krea shares the Qwen-Image VAE latent space
    // (`qwenimage` student).
    let pid = match pid_to_load(pid_spec, use_pid) {
        Some(spec) => Some(Arc::new(PidEngine::from_spec(spec, PID_BACKBONE, device)?)),
        None => None,
    };

    Ok(KreaHeavy {
        dit,
        vae: Arc::new(vae),
        pid,
    })
}

/// Load Turbo components with the DiT taken from a **single-file INT8-ConvRot checkpoint** (sc-9300)
/// instead of the snapshot's `transformer/` dir. The tokenizer / Qwen3-VL TE / Qwen-Image VAE still come
/// from the canonical `root` snapshot (the ConvRot artifact quantizes only the DiT). `convrot_dit` is
/// the native-mmdit-keyed `.safetensors` file; the DiT's 28 blocks' attn+mlp load as per-output-channel
/// int8 (cuBLASLt IGEMM on CUDA), everything else dense bf16.
///
/// **Coherent as of sc-9601.** The checkpoint's int8 weights are the *rotated* `W·R` (regular-Hadamard,
/// group 256); each ConvRot projection now applies the matching online `RHT(x)` activation rotation
/// ([`candle_gen::quant::convrot`]) before the int8 IGEMM, so `RHT(x)·(W·R)ᵀ = x·Wᵀ` and the render is
/// coherent (the sc-9300 A/B NO-GO was the missing online leg — arXiv 2512.03673 / ComfyUI ConvRot,
/// clean-room from the paper + the checkpoint format). The per-channel dequant fold runs on-device
/// (sc-9601 perf). Worker wiring as a shipping generator variant stays deferred (sc-9092 pattern).
///
/// **sm_89 floor (locked decision 7 / sc-9300).** The int8 IGEMM tier is only offered on compute
/// capability ≥ 8.9 (RTX 40-series and up). On CUDA, this errors up front if the device is below the
/// floor rather than rendering on a card the marketing contract excludes; on non-CUDA it is a no-op
/// (the CPU dequant-dense fallback is for tests, not a shipping path).
pub fn load_components_convrot(
    root: &Path,
    convrot_dit: &Path,
    device: &Device,
) -> Result<Components> {
    // The floor probe needs a cuBLASLt handle to read the device's compute capability — so it KEEPS it
    // and hands it to the DiT weight set as the trunk's one shared handle (sc-12301 scope 5), instead of
    // building 32 MiB of workspace, reading two integers off it, and dropping it.
    let text = load_text(root, device)?;
    let heavy = load_heavy_convrot(root, convrot_dit, device)?;
    Ok(Components { text, heavy })
}

/// The heavy half of an INT8-ConvRot load — the int8 DiT (from `convrot_dit`) + the Qwen-Image VAE (from
/// `root`). Shared by the **resident** path ([`load_components_convrot`], which pairs it with a co-loaded
/// TE) and the **sequential** path ([`load_residency_heavy_convrot`], which loads it only after the TE
/// has been encoded and dropped — sc-12089 / epic 10765). Both must build the DiT identically, so the
/// int8-vs-bf16 choice lives in exactly one place.
///
/// PiD is always `None`: the ConvRot lane does not combine with the super-resolving decoder (sc-9300).
pub(crate) fn load_heavy_convrot(
    root: &Path,
    convrot_dit: &Path,
    device: &Device,
) -> Result<KreaHeavy> {
    // The floor probe needs a cuBLASLt handle to read the device's compute capability — so it KEEPS it
    // and hands it to the DiT weight set as the trunk's one shared handle (sc-12301 scope 5), instead of
    // building 32 MiB of workspace, reading two integers off it, and dropping it.
    let int8 = ensure_int8_floor(device)?;

    let cfg = Krea2Config::from_snapshot(root)?;
    // Seeded with the floor probe's handle: every int8 projection `Krea2Transformer::load` detects below
    // shares this ONE handle rather than building its own (the sc-12301 defect).
    let dit_w = Weights::from_convrot_file(convrot_dit, device, DIT_DTYPE)?.with_int8_context(int8);
    crate::convert::validate_transformer(&dit_w, &cfg)?;
    let dit = Krea2Transformer::load(&dit_w, &cfg)?;

    let vae = load_vae(root, device)?;

    Ok(KreaHeavy {
        dit,
        vae: Arc::new(vae),
        pid: None,
    })
}

/// The **sequential** twin of [`load_heavy_convrot`] (sc-12089 / epic 10765 Phase 1c): the int8 DiT +
/// VAE **plus** the VAE encoder, loaded only after [`KreaText`] has been dropped so it reuses the
/// encoder's freed pool. The int8-ConvRot sibling of [`load_residency_heavy`]; the difference is only
/// which checkpoint the DiT comes from (the int8 single-file vs `root/transformer`).
pub(crate) fn load_residency_heavy_convrot(
    root: &Path,
    convrot_dit: &Path,
    device: &Device,
) -> Result<ResidencyHeavy> {
    Ok(ResidencyHeavy {
        heavy: load_heavy_convrot(root, convrot_dit, device)?,
        vae_encoder: load_vae_encoder(root, device)?,
    })
}

/// Enforce the INT8-ConvRot sm_89 compute-capability floor (locked decision 7), **returning the handle
/// the probe had to build** as the trunk's shared [`Int8Context`] (sc-12301 scope 5).
///
/// Reuses the sc-9299 cuBLASLt compute-cap probe (`meets_fp8_floor` ⇔ capability ≥ 8.9). A non-CUDA
/// device is allowed and yields an empty context (the CPU dequant path is test-only). On CUDA below the
/// floor this errors with the marketing contract — and the handle is dropped with the context, since no
/// projection will be built.
///
/// The floor stays **sm_89 and stays here**: `Int8Context::new` deliberately does not gate on it, so the
/// context type is reusable by any int8 caller, and reusing NVFP4's sm_120 `Nvfp4Context` would wrongly
/// deny int8 on sm_89..sm_120 cards.
#[cfg(feature = "cuda")]
fn ensure_int8_floor(device: &Device) -> Result<Int8Context> {
    let ctx = Int8Context::new(device)
        .map_err(|e| CandleError::Msg(format!("krea convrot: cublasLt probe: {e}")))?;
    if device.is_cuda() {
        let lt = ctx.handle_for(device)?;
        if !lt
            .meets_fp8_floor()
            .map_err(|e| CandleError::Msg(format!("krea convrot: compute-cap probe: {e}")))?
        {
            let cap = lt.compute_cap().unwrap_or((0, 0));
            return Err(CandleError::Msg(format!(
                "krea INT8-ConvRot requires compute capability >= 8.9 (RTX 40-series+); this device is \
                 sm_{}{} — the ConvRot variant is not offered on older cards",
                cap.0, cap.1
            )));
        }
    }
    Ok(ctx)
}

/// Non-CUDA build: the int8 floor is vacuous (the CPU dequant-dense fallback is test-only), and the
/// shared context is empty — there is no handle to share.
#[cfg(not(feature = "cuda"))]
fn ensure_int8_floor(_device: &Device) -> Result<Int8Context> {
    Ok(Int8Context::none())
}

/// Render the **Turbo** (CFG-free, few-step rectified-flow Euler) text-to-image path for `req`.
pub fn render(
    comps: &Components,
    req: &GenerationRequest,
    device: &Device,
    on_progress: &mut dyn FnMut(Progress),
) -> Result<Vec<Image>> {
    // Condition encoding (seed-independent): the 12 selected Qwen3-VL hidden layers, stacked +
    // prefix-dropped → the DiT's text_fusion context [1, n_tok, 12, 2560]. CFG-free, B=1.
    let context = encode_prompt_context(&comps.text, req)?;
    render_from_context(&comps.heavy, req, device, &context, on_progress)
}

/// SPIKE (sc-8596, HELD) — the ComfyUI-Conditioning-Rebalance trick ported to candle: reweight the 12
/// stacked Qwen3-VL select-layer taps by a per-layer scalar **before** the DiT's `TextFusionTransformer`
/// aggregates them (layerwise-attn → `projector` num_layers→1 → refiner). This is an "IP-Adapter-LIKE"
/// steering knob on the *text-only* weights — no new model weights. `weights.len()` must equal the layer
/// axis (`context.dim(2)` = 12); all-ones is a byte-exact no-op. Krea/Qwen-Image-family specific (depends
/// on the multi-tap structure); it does NOT generalize to CLIP/T5 encoders. See [`apply_tap_weights`].
pub fn render_tap_reweight(
    comps: &Components,
    req: &GenerationRequest,
    device: &Device,
    weights: &[f32],
    on_progress: &mut dyn FnMut(Progress),
) -> Result<Vec<Image>> {
    let context = encode_prompt_context(&comps.text, req)?;
    let context = apply_tap_weights(&context, weights)?;
    render_from_context(&comps.heavy, req, device, &context, on_progress)
}

/// SPIKE (sc-8596) — scale each of the stacked select-layer taps of a Krea text context
/// `[b, n_tok, num_layers, hidden]` by `weights[i]` along the layer axis (axis 2). A `[1,1,num_layers,1]`
/// broadcast multiply; all-ones reproduces the input bit-for-bit. Errors if `weights.len()` ≠ the layer
/// axis so a mis-sized sweep vector fails loudly rather than silently truncating.
pub fn apply_tap_weights(context: &Tensor, weights: &[f32]) -> Result<Tensor> {
    let n = context.dim(2)?;
    if weights.len() != n {
        return Err(CandleError::Msg(format!(
            "krea tap reweight: {} weights but the context has {n} select layers",
            weights.len()
        )));
    }
    let w = Tensor::from_vec(weights.to_vec(), (1, 1, n, 1), context.device())?
        .to_dtype(context.dtype())?;
    Ok(context.broadcast_mul(&w)?)
}

/// The GPU-validated-safe range for [`GenerationRequest::text_style_gain`] — the sc-8596 A/B swept
/// `[0.25, 1.75]` and every point stayed coherent. The engine clamps to this rather than trusting a
/// caller/UI to bound it.
const TEXT_STYLE_GAIN_RANGE: (f32, f32) = (0.25, 1.75);

/// Map the single "text style" gain scalar `g` to the per-layer tap ramp `w[i] = g + (2−2g)·i/(n−1)`
/// (sc-11878, the shipped control over the sc-8596 spike mechanism). `g` is clamped to
/// [`TEXT_STYLE_GAIN_RANGE`]. `g = 1` yields all-ones (a no-op); `g > 1` emphasizes the early
/// (low-level) taps (`w[0] = g`, tapering to `w[n−1] = 2−g`), `g < 1` biases the late (semantic) taps.
/// At `g = 1.75` this reproduces the spike's `early_ramp` (1.75→0.25). `n = 1` degenerates to `[g]`.
fn tap_gain_weights(gain: f32, n: usize) -> Vec<f32> {
    let g = gain.clamp(TEXT_STYLE_GAIN_RANGE.0, TEXT_STYLE_GAIN_RANGE.1);
    if n <= 1 {
        return vec![g; n];
    }
    (0..n)
        .map(|i| g + (2.0 - 2.0 * g) * (i as f32) / ((n - 1) as f32))
        .collect()
}

/// Apply the optional "text style" tap-reweight `gain` to a freshly-encoded positive context
/// (sc-11878; extended to control/edit in sc-12009). `None` — or a gain within 1e-4 of 1.0 — returns
/// the context untouched (the no-op fast path), so plain requests pay nothing. Takes the raw
/// `Option<f32>` (not a `GenerationRequest`) so the bespoke pose-control request can share it. Krea-
/// only: reached only from the candle-gen-krea render paths.
pub(crate) fn maybe_apply_style_gain(context: Tensor, gain: Option<f32>) -> Result<Tensor> {
    match gain {
        Some(g) if (g - 1.0).abs() > 1e-4 => {
            let n = context.dim(2)?;
            apply_tap_weights(&context, &tap_gain_weights(g, n))
        }
        _ => Ok(context),
    }
}

/// Encode the positive prompt to the DiT context and apply the optional text-style gain — the single
/// seam every txt2img / img2img entry point shares (sc-11878). The negative (CFG-uncond) context is
/// encoded without the gain, so the knob steers only the conditional prediction (matching the spike).
fn encode_prompt_context(text: &KreaText, req: &GenerationRequest) -> Result<Tensor> {
    let context = text
        .te
        .forward(&text.tok.encode_prompt(&req.prompt, MAX_TEXT_TOKENS)?)?;
    maybe_apply_style_gain(context, req.text_style_gain)
}

/// Encode the **unconditional** (CFG negative) context for the Raw/Edit full-CFG paths — the plain
/// encode with NO style gain, so the knob steers only the conditional prediction (sc-11878). An absent /
/// empty negative prompt is the reference's `""`. Split out alongside [`encode_prompt_context`] so both
/// the resident and `Sequential` Raw paths encode both branches from the SAME text phase before it is
/// dropped (sc-12089).
fn encode_negative_context(text: &KreaText, negative: &str) -> Result<Tensor> {
    Ok(text
        .te
        .forward(&text.tok.encode_prompt(negative, MAX_TEXT_TOKENS)?)?)
}

/// The seed-loop + schedule + decode tail shared by [`render`] and the unified residency path.
/// [`render_tap_reweight`] spike: everything downstream of the (possibly reweighted) `context`
/// `[1, n_tok, 12, 2560]`.
///
/// Takes [`KreaHeavy`] rather than [`Components`] (sc-12089) — the whole point of the phase split is
/// that this body cannot observe whether the text phase is still resident or was already dropped, so
/// the `Resident` and `Sequential` paths run the SAME code and produce byte-identical output.
fn render_from_context(
    comps: &KreaHeavy,
    req: &GenerationRequest,
    device: &Device,
    context: &Tensor,
    on_progress: &mut dyn FnMut(Progress),
) -> Result<Vec<Image>> {
    let steps = req.steps.map(|s| s as usize).unwrap_or(TURBO_STEPS);
    let base_seed = req.seed.unwrap_or_else(gen_core::default_seed);

    // Native exponential-mu Turbo sigmas are the byte-exact default; a curated scheduler reshapes over
    // the same mu. Raw sigma → DiT timestep, raw velocity → Euler `x + v·(σ_{i+1} − σ_i)`.
    let native = turbo_sigmas(steps);
    let sigmas = candle_gen::resolve_flow_schedule(
        req.scheduler.as_deref(),
        TURBO_MU as f32,
        steps,
        &native,
    );

    // Resolve the decode seam once for the whole batch (epic 7840 / sc-7853): a per-generation PiD
    // decoder bound to this prompt when `req.use_pid` is set (errors if requested but not loaded), else
    // `None` → the native QwenVae decode. Shared across `count` images (same prompt).
    let pid_decoder = candle_gen_pid::resolve_pid_decoder(
        comps.pid.as_deref(),
        req,
        base_seed,
        crate::KREA_2_TURBO_ID,
    )?;

    candle_gen::for_each_image_seed(base_seed, req.count, |seed| {
        let noise = init_noise(req.height, req.width, seed, device)?;
        let lat = candle_gen::run_flow_sampler(
            req.sampler.as_deref(),
            TimestepConvention::Sigma,
            &sigmas,
            noise,
            seed,
            &req.cancel,
            on_progress,
            |x, timestep| -> Result<Tensor> {
                let t = Tensor::from_vec(vec![timestep], (1,), device)?;
                let v = comps.dit.forward(x, &t, context)?;
                Ok(v.to_dtype(DType::F32)?)
            },
        )?;
        on_progress(Progress::Decoding);
        // PiD (super-resolving) decode when the toggle resolved one; else the native VAE. Both consume
        // the same normalized `[1,16,H/8,W/8]` latent (a zero-transform seam); PiD returns a larger
        // `[1,3,4H,4W]` tensor and `to_image` reads the size from it.
        let decoded = match &pid_decoder {
            Some(pid) => pid.decode(&lat)?,
            None => comps.vae.decode(&lat)?.to_dtype(DType::F32)?,
        };
        to_image(&decoded)
    })
}

/// Render the **Turbo img2img** (reference-guided latent-init) path (sc-10134, epic 8588) — the CFG-free
/// sibling of [`render`] seeded from a VAE-encoded reference instead of pure noise. The candle/CUDA twin
/// of mlx-gen-krea's `generate_turbo_img2img` (mlx A1, sc-8590). The mechanism (the fork's `LatentCreator`
/// img2img leaves, ported byte-for-byte from `mlx_gen::img2img`):
/// 1. LANCZOS-resize the reference to the target resolution (`preprocess_img2img_init`), VAE-encode it
///    to the normalized `[1,16,H/8,W/8]` **clean** latent — the same space as the init noise.
/// 2. `start = init_time_step(steps, strength)` = `max(1, floor(steps·strength))` (reference fidelity:
///    higher strength → later start → closer to the reference — the fork's convention, NOT SDXL's).
/// 3. Blend `x_start = (1−σ_start)·clean + σ_start·noise` at `σ_start = sigmas[start]`
///    (`add_noise_by_interpolation`).
/// 4. Run the CFG-free rectified-flow Euler loop over `sigmas[start..]` from `x_start` (one DiT
///    forward/step, exactly as [`render`]).
///
/// The condition encode, schedule, and PiD/native decode seam are identical to [`render`] — img2img
/// changes only the initial latent and the schedule start. `strength == 1.0` (start == steps) leaves a
/// single trailing `σ = 0.0`, i.e. the clean reference decoded verbatim (full fidelity, no denoise), the
/// SDXL-edit "empty schedule is a no-op" behaviour.
#[allow(clippy::too_many_arguments)]
pub fn render_img2img(
    comps: &Components,
    vae_encoder: &QwenVaeEncoder,
    req: &GenerationRequest,
    reference: &Image,
    strength: Option<f32>,
    device: &Device,
    on_progress: &mut dyn FnMut(Progress),
) -> Result<Vec<Image>> {
    // Condition encoding (seed-independent), identical to `render`: img2img changes only the init latent.
    let context = encode_prompt_context(&comps.text, req)?;
    render_img2img_from_context(
        &comps.heavy,
        vae_encoder,
        req,
        reference,
        strength,
        device,
        &context,
        on_progress,
    )
}

/// The Turbo img2img reference-encode + seed-loop + decode tail (sc-12089): everything downstream of the
/// encoded context, shared by resident and sequential img2img. Takes [`KreaHeavy`],
/// so it cannot observe whether the text phase is still resident — both residency paths are
/// byte-identical.
#[allow(clippy::too_many_arguments)]
fn render_img2img_from_context(
    comps: &KreaHeavy,
    vae_encoder: &QwenVaeEncoder,
    req: &GenerationRequest,
    reference: &Image,
    strength: Option<f32>,
    device: &Device,
    context: &Tensor,
    on_progress: &mut dyn FnMut(Progress),
) -> Result<Vec<Image>> {
    let steps = req.steps.map(|s| s as usize).unwrap_or(TURBO_STEPS);
    let base_seed = req.seed.unwrap_or_else(gen_core::default_seed);

    let native = turbo_sigmas(steps);
    let sigmas = candle_gen::resolve_flow_schedule(
        req.scheduler.as_deref(),
        TURBO_MU as f32,
        steps,
        &native,
    );

    // The img2img start step + the σ that seeds the blend (the fork's leaves). Clamp `start` to the last
    // schedule index defensively so `sigmas[start]` is always valid (a curated scheduler may re-stride the
    // schedule to a length other than `steps + 1`).
    let start = init_time_step(steps, strength).min(sigmas.len().saturating_sub(1));
    let sigma_start = sigmas[start];

    // The clean reference latent (seed-independent): LANCZOS-resize → `[-1,1]` pixels → the normalized
    // 16-ch latent (f32, the noise's dtype for the blend + sampler).
    let init_pixels = preprocess_img2img_init(reference, req.width, req.height, device)?;
    let clean = vae_encoder.encode(&init_pixels)?.to_dtype(DType::F32)?;

    let pid_decoder = candle_gen_pid::resolve_pid_decoder(
        comps.pid.as_deref(),
        req,
        base_seed,
        crate::KREA_2_TURBO_ID,
    )?;

    candle_gen::for_each_image_seed(base_seed, req.count, |seed| {
        let noise = init_noise(req.height, req.width, seed, device)?;
        let x_start = add_noise_by_interpolation(&clean, &noise, sigma_start)?;
        // `sigmas[start..]` seeds the sampler at `sigma_start` (= its first element, the state x_start is
        // at). A single-element tail (strength == 1.0) has no Euler step — decode x_start (= clean) as-is.
        let sub = &sigmas[start..];
        let lat = if sub.len() < 2 {
            x_start
        } else {
            candle_gen::run_flow_sampler(
                req.sampler.as_deref(),
                TimestepConvention::Sigma,
                sub,
                x_start,
                seed,
                &req.cancel,
                on_progress,
                |x, timestep| -> Result<Tensor> {
                    let t = Tensor::from_vec(vec![timestep], (1,), device)?;
                    let v = comps.dit.forward(x, &t, context)?;
                    Ok(v.to_dtype(DType::F32)?)
                },
            )?
        };
        on_progress(Progress::Decoding);
        let decoded = match &pid_decoder {
            Some(pid) => pid.decode(&lat)?,
            None => comps.vae.decode(&lat)?.to_dtype(DType::F32)?,
        };
        to_image(&decoded)
    })
}

/// Render the **Raw** (undistilled, full classifier-free-guidance) rectified-flow text-to-image path
/// for `req` (`krea_2_raw`, epic 9992 / sc-9994) — the CFG sibling of [`render`]. Two DiT forwards per
/// step, the conditional (positive prompt) and the unconditional (the user negative prompt, or `""`
/// when none), combined by the **reference** `sampling.py:129` formula via `krea_cfg_combine`
/// (`v = cond + guidance·(cond − uncond)`, NOT the textbook `uncond + g·Δ`: Krea's guidance is offset by
/// one). `guidance ≤ 0` short-circuits to a single conditional forward (the uncond context is never
/// encoded), matching the reference `cfg = guidance > 0`. Unlike Turbo's fixed `mu = 1.15`, the schedule
/// is resolution-**dynamic** ([`base_schedule`]). Everything else — the Qwen3-VL condition encode, the
/// PiD/native decode seam, and the per-seed batch loop — is identical to [`render`].
pub fn render_base(
    comps: &Components,
    req: &GenerationRequest,
    device: &Device,
    on_progress: &mut dyn FnMut(Progress),
) -> Result<Vec<Image>> {
    let guidance = req.guidance.unwrap_or(RAW_GUIDANCE);
    let (context, neg_context) = encode_base_contexts(&comps.text, req, guidance)?;
    render_base_from_contexts(
        &comps.heavy,
        req,
        device,
        &context,
        neg_context.as_ref(),
        guidance,
        on_progress,
    )
}

/// Encode the Raw path's conditional + unconditional contexts from one text phase (sc-12089) — the
/// single seam both residency modes share, so the `Sequential` path
/// cannot drift from the resident encode.
///
/// The unconditional branch is built ONLY when CFG is active (reference `cfg = guidance > 0`); an
/// absent / empty negative prompt defaults to `""` (reference `negative_prompts = [""] * n`). `guidance
/// ≤ 0` returns `None` and the uncond context is never encoded.
fn encode_base_contexts(
    text: &KreaText,
    req: &GenerationRequest,
    guidance: f32,
) -> Result<(Tensor, Option<Tensor>)> {
    // Positive (conditional) condition encoding (seed-independent): the 12 selected Qwen3-VL hidden
    // layers → the DiT's text_fusion context [1, n_tok, 12, 2560].
    let context = encode_prompt_context(text, req)?;

    let neg_context = if guidance > 0.0 {
        let negative = req.negative_prompt.as_deref().unwrap_or_default();
        Some(encode_negative_context(text, negative)?)
    } else {
        None
    };

    Ok((context, neg_context))
}

/// The Raw seed-loop + dynamic schedule + two-forward CFG denoise + decode tail (sc-12089): everything
/// downstream of the encoded contexts, shared by both residency modes.
///
/// Takes [`KreaHeavy`] rather than [`Components`] — like [`render_from_context`], this body cannot
/// observe whether the text phase is still resident, so both residency paths produce byte-identical
/// output.
#[allow(clippy::too_many_arguments)]
fn render_base_from_contexts(
    comps: &KreaHeavy,
    req: &GenerationRequest,
    device: &Device,
    context: &Tensor,
    neg_context: Option<&Tensor>,
    guidance: f32,
    on_progress: &mut dyn FnMut(Progress),
) -> Result<Vec<Image>> {
    let steps = req.steps.map(|s| s as usize).unwrap_or(RAW_STEPS);
    let base_seed = req.seed.unwrap_or_else(gen_core::default_seed);

    // Resolution-dynamic Raw sigma schedule (mu from the image-token count); a curated scheduler
    // reshapes over the same dynamic mu. Raw sigma → DiT timestep, raw velocity → Euler
    // `x + v·(σ_{i+1} − σ_i)`.
    let sigmas = base_schedule(steps, req.width, req.height, req.scheduler.as_deref());

    // Resolve the decode seam once for the whole batch (epic 7840 / sc-7853): a per-generation PiD
    // decoder bound to this prompt when `req.use_pid` is set (errors if requested but not loaded), else
    // `None` → the native QwenVae decode.
    let pid_decoder = candle_gen_pid::resolve_pid_decoder(
        comps.pid.as_deref(),
        req,
        base_seed,
        crate::KREA_2_RAW_ID,
    )?;

    candle_gen::for_each_image_seed(base_seed, req.count, |seed| {
        let noise = init_noise(req.height, req.width, seed, device)?;
        let lat = candle_gen::run_flow_sampler(
            req.sampler.as_deref(),
            TimestepConvention::Sigma,
            &sigmas,
            noise,
            seed,
            &req.cancel,
            on_progress,
            |x, timestep| -> Result<Tensor> {
                let t = Tensor::from_vec(vec![timestep], (1,), device)?;
                let cond = comps.dit.forward(x, &t, context)?;
                // Two-forward CFG when a negative context was prepared (guidance > 0); else the bare
                // conditional velocity. Combined by the shared reference formula (`krea_cfg_combine`).
                let v = match neg_context {
                    Some(nc) => {
                        let uncond = comps.dit.forward(x, &t, nc)?;
                        krea_cfg_combine(&cond, &uncond, guidance)?
                    }
                    None => cond,
                };
                Ok(v.to_dtype(DType::F32)?)
            },
        )?;
        on_progress(Progress::Decoding);
        let decoded = match &pid_decoder {
            Some(pid) => pid.decode(&lat)?,
            None => comps.vae.decode(&lat)?.to_dtype(DType::F32)?,
        };
        to_image(&decoded)
    })
}

/// Render the **Raw img2img** (reference-guided latent-init under full classifier-free guidance) path
/// (sc-10226, epic 8588) — the img2img sibling of [`render_base`] seeded from a VAE-encoded reference
/// instead of pure noise, and the full-CFG sibling of the CFG-free Turbo [`render_img2img`]. The
/// candle/CUDA twin of mlx-gen-krea's `generate_base_img2img_with_progress` (mlx A5a, sc-10224). It is
/// exactly [`render_base`]'s undistilled two-forward CFG denoise (resolution-dynamic [`base_schedule`],
/// the reference `krea_cfg_combine` combine, an optional user negative prompt) run over the
/// reference-seeded init latent + reduced schedule of [`render_img2img`]:
/// 1. LANCZOS-resize the reference to the target resolution (`preprocess_img2img_init`), VAE-encode it
///    to the normalized `[1,16,H/8,W/8]` **clean** latent (the same space as the init noise).
/// 2. `start = init_time_step(steps, strength)` (the fork's reference-fidelity convention: higher
///    strength → later start → closer to the reference — NOT SDXL's).
/// 3. Blend `x_start = (1−σ_start)·clean + σ_start·noise` at `σ_start = sigmas[start]`
///    (`add_noise_by_interpolation`).
/// 4. Run the Raw CFG rectified-flow Euler loop over `sigmas[start..]` from `x_start` — two DiT forwards
///    per step when `guidance > 0` (cond vs uncond, combined by `krea_cfg_combine`), one otherwise,
///    exactly as [`render_base`].
///
/// The condition/uncondition encode, dynamic schedule, and PiD/native decode seam are identical to
/// [`render_base`]; img2img changes only the initial latent and the schedule start. `strength == 1.0`
/// (start == steps) leaves a single trailing `σ = 0.0` — the clean reference decoded verbatim (no
/// denoise), matching [`render_img2img`].
#[allow(clippy::too_many_arguments)]
pub fn render_base_img2img(
    comps: &Components,
    vae_encoder: &QwenVaeEncoder,
    req: &GenerationRequest,
    reference: &Image,
    strength: Option<f32>,
    device: &Device,
    on_progress: &mut dyn FnMut(Progress),
) -> Result<Vec<Image>> {
    let guidance = req.guidance.unwrap_or(RAW_GUIDANCE);

    // The conditional + unconditional condition encoding (seed-independent), identical to `render_base`
    // (the shared `encode_base_contexts` seam): img2img changes only the init latent + schedule start,
    // not the conditioning.
    let (context, neg_context) = encode_base_contexts(&comps.text, req, guidance)?;
    render_base_img2img_from_contexts(
        &comps.heavy,
        vae_encoder,
        req,
        reference,
        strength,
        device,
        &context,
        neg_context.as_ref(),
        guidance,
        on_progress,
    )
}

/// The Raw img2img reference-encode + seed-loop + CFG denoise + decode tail (sc-12089): everything
/// downstream of the encoded contexts, shared by both residency modes. Takes [`KreaHeavy`], so both
/// residency paths are byte-identical.
#[allow(clippy::too_many_arguments)]
fn render_base_img2img_from_contexts(
    comps: &KreaHeavy,
    vae_encoder: &QwenVaeEncoder,
    req: &GenerationRequest,
    reference: &Image,
    strength: Option<f32>,
    device: &Device,
    context: &Tensor,
    neg_context: Option<&Tensor>,
    guidance: f32,
    on_progress: &mut dyn FnMut(Progress),
) -> Result<Vec<Image>> {
    let steps = req.steps.map(|s| s as usize).unwrap_or(RAW_STEPS);
    let base_seed = req.seed.unwrap_or_else(gen_core::default_seed);

    // Resolution-dynamic Raw sigma schedule (mu from the image-token count), identical to `render_base`.
    let sigmas = base_schedule(steps, req.width, req.height, req.scheduler.as_deref());

    // The img2img start step + the σ that seeds the blend (the fork's leaves, shared with the Turbo
    // `render_img2img`). Clamp `start` to the last schedule index defensively so `sigmas[start]` is always
    // valid (a curated scheduler may re-stride the schedule to a length other than `steps + 1`).
    let start = init_time_step(steps, strength).min(sigmas.len().saturating_sub(1));
    let sigma_start = sigmas[start];

    // The clean reference latent (seed-independent): LANCZOS-resize → `[-1,1]` pixels → the normalized
    // 16-ch latent (f32, the noise's dtype for the blend + sampler).
    let init_pixels = preprocess_img2img_init(reference, req.width, req.height, device)?;
    let clean = vae_encoder.encode(&init_pixels)?.to_dtype(DType::F32)?;

    let pid_decoder = candle_gen_pid::resolve_pid_decoder(
        comps.pid.as_deref(),
        req,
        base_seed,
        crate::KREA_2_RAW_ID,
    )?;

    candle_gen::for_each_image_seed(base_seed, req.count, |seed| {
        let noise = init_noise(req.height, req.width, seed, device)?;
        let x_start = add_noise_by_interpolation(&clean, &noise, sigma_start)?;
        // `sigmas[start..]` seeds the sampler at `sigma_start` (= its first element, the state x_start is
        // at). A single-element tail (strength == 1.0) has no Euler step — decode x_start (= clean) as-is.
        let sub = &sigmas[start..];
        let lat = if sub.len() < 2 {
            x_start
        } else {
            candle_gen::run_flow_sampler(
                req.sampler.as_deref(),
                TimestepConvention::Sigma,
                sub,
                x_start,
                seed,
                &req.cancel,
                on_progress,
                |x, timestep| -> Result<Tensor> {
                    let t = Tensor::from_vec(vec![timestep], (1,), device)?;
                    let cond = comps.dit.forward(x, &t, context)?;
                    // Two-forward CFG when a negative context was prepared (guidance > 0); else the bare
                    // conditional velocity. Combined by the shared reference formula (`krea_cfg_combine`),
                    // exactly as `render_base`.
                    let v = match neg_context {
                        Some(nc) => {
                            let uncond = comps.dit.forward(x, &t, nc)?;
                            krea_cfg_combine(&cond, &uncond, guidance)?
                        }
                        None => cond,
                    };
                    Ok(v.to_dtype(DType::F32)?)
                },
            )?
        };
        on_progress(Progress::Decoding);
        let decoded = match &pid_decoder {
            Some(pid) => pid.decode(&lat)?,
            None => comps.vae.decode(&lat)?.to_dtype(DType::F32)?,
        };
        to_image(&decoded)
    })
}

/// The resident image-edit-only component (epic 10871), loaded lazily on the first edit so the txt2img
/// paths keep their footprint. The Qwen3-VL vision tower moved into the `KreaText` phase in sc-12129 so grounded
/// conditioning completes inside the droppable text phase; only the Qwen-Image VAE encoder remains
/// here for the resident path.
pub struct EditComponents {
    vae_encoder: QwenVaeEncoder,
}

/// Load the resident image-edit-only VAE encoder from a Krea 2 snapshot. The vision tower is lazy on
/// `KreaText` and loads only when grounded conditioning is first requested.
pub fn load_edit_components(root: &Path, device: &Device) -> Result<EditComponents> {
    Ok(EditComponents {
        vae_encoder: load_vae_encoder(root, device)?,
    })
}

/// Render the **image-edit** path (epic 10871 / sc-10877, sc-10878) — a source reference (or two:
/// image 1, then image 2) plus an instruction produce an edited image with identity/subject
/// preserved. Kontext-style: each reference is VAE-encoded at the **target** resolution and prepended to
/// the DiT sequence at its own RoPE frame ([`Krea2Transformer::forward_edit`]); the denoise runs the Raw
/// true-CFG loop **from pure noise** (the source is in-context conditioning, NOT a noised init — the key
/// difference from an img2img latent-blend). `references` is the fixed-order source set (1 or 2).
///
/// **Dual conditioning (R2).** BOTH wires the edit LoRA was trained against are supplied: (a) the
/// in-context VAE tokens (references VAE-encoded at target res, seq-concat in `forward_edit`), and (b)
/// the image-grounded Qwen3-VL encoding (references run through the vision tower, spliced into the
/// condition encoder — [`KreaTextEncoder::forward_with_images`]). The vision tower runs once per
/// reference and is reused across the positive/negative instruction encodings.
pub fn render_edit(
    comps: &Components,
    edit: &EditComponents,
    req: &GenerationRequest,
    references: &[Image],
    // `true` = the CFG-free distilled Turbo edit (`krea_2_turbo_edit`, sc-11640): few-step
    // `turbo_schedule` (fixed mu), guidance forced to 0 (a single conditional forward). `false` = the
    // undistilled Raw edit (`krea_2_edit`, epic 10871): the resolution-dynamic `base_schedule` under
    // full CFG. Mirrors mlx-gen-krea `render_edit(distilled)`.
    distilled: bool,
    device: &Device,
    on_progress: &mut dyn FnMut(Progress),
) -> Result<Vec<Image>> {
    let encoded = encode_edit_context(&comps.text, req, references, distilled, device)?;
    render_edit_from_context(
        &comps.heavy,
        &edit.vae_encoder,
        encoded,
        req,
        references,
        device,
        on_progress,
    )
}

/// The complete Qwen3-VL-grounded edit conditioning produced while [`KreaText`] is resident. It owns
/// its tensors so the text encoder and lazily loaded vision tower can drop before the DiT/VAE bundle
/// loads under sequential residency.
pub(crate) struct EditContext {
    context: Tensor,
    neg_context: Option<Tensor>,
    steps: usize,
    base_seed: u64,
    guidance: f32,
    distilled: bool,
}

/// Run the full grounded Qwen3-VL encode inside the text phase (sc-12129). The vision tower is loaded
/// lazily by [`KreaText::vision`], used for every reference, and retained only as part of the text
/// phase. The returned tensors do not borrow it, so sequential residency can drop the entire phase
/// before loading the DiT and VAE.
pub(crate) fn encode_edit_context(
    text: &KreaText,
    req: &GenerationRequest,
    references: &[Image],
    distilled: bool,
    device: &Device,
) -> Result<EditContext> {
    check_reference_count(references.len())?;
    let steps =
        req.steps
            .map(|s| s as usize)
            .unwrap_or(if distilled { TURBO_STEPS } else { RAW_STEPS });
    let base_seed = req.seed.unwrap_or_else(gen_core::default_seed);
    // Turbo edit is CFG-free: force guidance 0 so the unconditional grounded encode is skipped and the
    // sampler runs a single conditional forward. Raw edit honors the request guidance (default 3.5).
    let guidance = if distilled {
        0.0
    } else {
        req.guidance.unwrap_or(RAW_GUIDANCE)
    };

    // Wire (b): run the vision tower ONCE per reference (shared across the pos/neg instruction encodings).
    let vision = text.vision()?;
    let grounding = encode_vision(&vision, references, device)?;

    // Positive (+ optional unconditional) image-grounded conditioning: the instruction is tokenized with
    // the vision blocks, and the tower's merged embeds + deepstack features are spliced in. The optional
    // "text style" tap-reweight gain (sc-12009) applies to the POSITIVE grounded context — it carries the
    // same 12-tap structure the plain encode does, so `apply_tap_weights` is shape-safe. The negative
    // (CFG-uncond) grounded context is left untouched so the knob steers only the conditional prediction.
    let context = maybe_apply_style_gain(
        grounding.condition(&text.tok, &text.te, &req.prompt)?,
        req.text_style_gain,
    )?;
    let neg_context = if guidance > 0.0 {
        let negative = req.negative_prompt.as_deref().unwrap_or_default();
        Some(grounding.condition(&text.tok, &text.te, negative)?)
    } else {
        None
    };

    Ok(EditContext {
        context,
        neg_context,
        steps,
        base_seed,
        guidance,
        distilled,
    })
}

/// Render a sequential edit after its grounded context escaped the text phase. [`ResidencyHeavy`]
/// supplies the same DiT/VAE/PiD bundle and VAE encoder used by the existing img2img residency path.
pub(crate) fn render_edit_residency(
    heavy: &ResidencyHeavy,
    encoded: EditContext,
    req: &GenerationRequest,
    references: &[Image],
    device: &Device,
    on_progress: &mut dyn FnMut(Progress),
) -> Result<Vec<Image>> {
    render_edit_from_context(
        &heavy.heavy,
        &heavy.vae_encoder,
        encoded,
        req,
        references,
        device,
        on_progress,
    )
}

fn render_edit_from_context(
    heavy: &KreaHeavy,
    vae_encoder: &QwenVaeEncoder,
    encoded: EditContext,
    req: &GenerationRequest,
    references: &[Image],
    device: &Device,
    on_progress: &mut dyn FnMut(Progress),
) -> Result<Vec<Image>> {
    let EditContext {
        context,
        neg_context,
        steps,
        base_seed,
        guidance,
        distilled,
    } = encoded;

    // Wire (a): VAE-encode each reference at the TARGET resolution → the normalized 16-ch latent (static
    // across steps). Fixed order preserved (image 1, then image 2 — sc-10878).
    let ref_latents = encode_references(vae_encoder, references, req.width, req.height, device)?;

    // Turbo edit runs the distilled few-step `turbo_schedule` (fixed mu) the CFG-free student expects;
    // Raw edit runs the resolution-dynamic `base_schedule` (undistilled, like `render_base`).
    let sigmas = if distilled {
        turbo_schedule(steps, req.scheduler.as_deref())
    } else {
        base_schedule(steps, req.width, req.height, req.scheduler.as_deref())
    };

    // Resolve the decode seam once for the whole batch (epic 7840 / sc-7853, sc-11197): a per-generation
    // PiD decoder bound to this request when `req.use_pid` is set (errors if requested but not loaded),
    // else `None` → the native QwenVae decode. The edit id honors `use_pid` exactly like its `render` /
    // `render_base` txt2img siblings — an edit generator loaded with a PiD engine (`spec.pid`) is no
    // longer a dead wire. `model_id` names the surface for the error message (Raw vs distilled Turbo edit).
    let pid_decoder = candle_gen_pid::resolve_pid_decoder(
        heavy.pid.as_deref(),
        req,
        base_seed,
        edit_pid_model_id(distilled),
    )?;

    candle_gen::for_each_image_seed(base_seed, req.count, |seed| {
        let noise = init_noise(req.height, req.width, seed, device)?;
        let lat = candle_gen::run_flow_sampler(
            req.sampler.as_deref(),
            TimestepConvention::Sigma,
            &sigmas,
            noise,
            seed,
            &req.cancel,
            on_progress,
            |x, timestep| -> Result<Tensor> {
                let t = Tensor::from_vec(vec![timestep], (1,), device)?;
                let cond = heavy.dit.forward_edit(x, &t, &context, &ref_latents)?;
                let v = match &neg_context {
                    Some(nc) => {
                        let uncond = heavy.dit.forward_edit(x, &t, nc, &ref_latents)?;
                        krea_cfg_combine(&cond, &uncond, guidance)?
                    }
                    None => cond,
                };
                Ok(v.to_dtype(DType::F32)?)
            },
        )?;
        on_progress(Progress::Decoding);
        // PiD (super-resolving) decode when the toggle resolved one; else the native VAE. Both consume
        // the same normalized `[1,16,H/8,W/8]` latent (a zero-transform seam); PiD returns a larger
        // `[1,3,4H,4W]` tensor and `to_image` reads the size from it — matching `render` / `render_base`.
        let decoded = match &pid_decoder {
            Some(pid) => pid.decode(&lat)?,
            None => heavy.vae.decode(&lat)?.to_dtype(DType::F32)?,
        };
        to_image(&decoded)
    })
}

/// VAE-encode each source `references` image at the target `(width, height)` → the normalized 16-ch
/// latent `[1, 16, H/8, W/8]` the DiT's `img_in` consumes (same space as the noise). Fixed order is
/// preserved (the returned vec is in `references` order — image 1, then image 2; sc-10878).
fn encode_references(
    vae_encoder: &QwenVaeEncoder,
    references: &[Image],
    width: u32,
    height: u32,
    device: &Device,
) -> Result<Vec<Tensor>> {
    let mut out = Vec::with_capacity(references.len());
    for im in references {
        let pixels = image_to_pixels(im, width, height, device)?;
        out.push(vae_encoder.encode(&pixels)?);
    }
    Ok(out)
}

/// Validate the source-reference count against the fixed-order [`MAX_EDIT_REFERENCES`] cap (sc-10878):
/// at least one (an edit needs a source) and at most two (image 1, then image 2). Pure so it is unit-testable.
fn check_reference_count(n: usize) -> Result<()> {
    if n == 0 {
        return Err(CandleError::Msg(
            "krea edit: at least one source reference image is required".into(),
        ));
    }
    if n > MAX_EDIT_REFERENCES {
        return Err(CandleError::Msg(format!(
            "krea edit: at most {MAX_EDIT_REFERENCES} references are supported (image 1, \
             then image 2); got {n}"
        )));
    }
    Ok(())
}

/// The vision-tower output for the source references (epic 10871 / sc-10880), shared across the
/// positive/negative instruction encodings: each reference's merged image embeds + 3 deepstack features
/// + patch grid, and the per-reference `<|image_pad|>` placeholder count for the tokenizer.
struct Grounding {
    embeds: Vec<Tensor>,
    deepstack: Vec<Vec<Tensor>>,
    grids: Vec<[i32; 3]>,
    n_per_ref: Vec<usize>,
}

impl Grounding {
    /// Image-grounded condition encoding for one instruction: tokenize `prompt` with the reference
    /// vision blocks, then splice the tower's embeds/deepstack into the condition encoder.
    fn condition(
        &self,
        tok: &crate::tokenizer::KreaTokenizer,
        te: &KreaTextEncoder,
        prompt: &str,
    ) -> Result<Tensor> {
        let ids = tok.encode_with_images(prompt, &self.n_per_ref, MAX_EDIT_TOKENS)?;
        Ok(te.forward_with_images(&ids, &self.embeds, &self.deepstack, &self.grids)?)
    }
}

/// Run the Qwen3-VL vision tower over each source reference (in fixed order — sc-10878) → the shared
/// [`Grounding`]. Preprocess + tower `forward` per reference (epic 10871 / sc-10880).
fn encode_vision(
    vision: &crate::vision::VisionTower,
    references: &[Image],
    device: &Device,
) -> Result<Grounding> {
    let merge = vision.config().spatial_merge_size;
    let mut g = Grounding {
        embeds: Vec::with_capacity(references.len()),
        deepstack: Vec::with_capacity(references.len()),
        grids: Vec::with_capacity(references.len()),
        n_per_ref: Vec::with_capacity(references.len()),
    };
    for im in references {
        let (embeds, deepstack, grid) = crate::vision::encode_image(
            vision,
            &im.pixels,
            im.height as usize,
            im.width as usize,
            device,
        )?;
        g.n_per_ref
            .push(crate::vision::merged_token_count(grid, merge));
        g.embeds.push(embeds);
        g.deepstack.push(deepstack);
        g.grids.push(grid);
    }
    Ok(g)
}

/// One RGB8 [`Image`] (HWC, `[0, 255]`) resized to `(target_w, target_h)` → the VAE input tensor
/// `[1, 3, H, W]` in `[-1, 1]` (f32, on `device`). BICUBIC resample (gen-core's PIL-exact
/// [`resize_bicubic_u8`], the same used by the vision preprocess), then `(x/255)·2 − 1`.
fn image_to_pixels(im: &Image, target_w: u32, target_h: u32, device: &Device) -> Result<Tensor> {
    let (w, h) = (target_w as usize, target_h as usize);
    if im.pixels.len()
        != candle_gen::gen_core::imageops::checked_image_buffer_len(
            im.width as usize,
            im.height as usize,
            3,
        )
        .unwrap_or(usize::MAX)
    {
        return Err(CandleError::Msg(format!(
            "krea edit: reference pixel buffer {} != {}x{}x3",
            im.pixels.len(),
            im.width,
            im.height
        )));
    }
    let resized: Vec<f32> = if (im.width, im.height) == (target_w, target_h) {
        im.pixels.iter().map(|&p| p as f32).collect()
    } else {
        resize_bicubic_u8(&im.pixels, im.height as usize, im.width as usize, h, w)?
    };
    // HWC [0,255] → CHW [-1,1].
    let mut chw = vec![0f32; 3 * h * w];
    for c in 0..3 {
        for y in 0..h {
            for x in 0..w {
                chw[c * h * w + y * w + x] = resized[(y * w + x) * 3 + c] / 127.5 - 1.0;
            }
        }
    }
    Ok(Tensor::from_vec(chw, (1, 3, h, w), device)?)
}

/// The **Raw** flow-match sigma schedule for `steps` at a given resolution: the exponential-mu shift
/// with a resolution-**dynamic** `mu` interpolated in image-token count ([`dynamic_mu`]), unlike the
/// Turbo fixed `mu = 1.15`. Length `steps + 1`, descending with a trailing `0.0`; a curated scheduler
/// (epic 7114) reshapes over the same dynamic mu. Mirrors mlx-gen-krea `base_schedule`.
pub fn base_schedule(steps: usize, width: u32, height: u32, scheduler: Option<&str>) -> Vec<f32> {
    // Image token count = (W/16)·(H/16) (latent /8 then patch /2) — the reference `x.shape[1]`.
    let seq_len = (width as f64 / 16.0) * (height as f64 / 16.0);
    let mu = dynamic_mu(seq_len);
    let native = krea_sigmas(steps, mu);
    candle_gen::resolve_flow_schedule(scheduler, mu as f32, steps, &native)
}

/// The **Turbo** flow-match sigma schedule for `steps`: the TDM-distilled fixed-mu (`TURBO_MU = 1.15`)
/// exponential-shift `turbo_sigmas`, optionally reshaped by a curated scheduler over the same mu. Length
/// `steps + 1`, descending with a trailing `0.0`. The distilled CFG-free Turbo edit (`krea_2_turbo_edit`,
/// sc-11640) denoises on this trajectory the few-step student was trained on, NOT the resolution-dynamic
/// `base_schedule`. Mirrors mlx-gen-krea `turbo_schedule` + the `render_turbo` inline build.
pub fn turbo_schedule(steps: usize, scheduler: Option<&str>) -> Vec<f32> {
    let native = turbo_sigmas(steps);
    candle_gen::resolve_flow_schedule(scheduler, TURBO_MU as f32, steps, &native)
}

/// The surface id for the image-edit PiD decode-seam error message (sc-11197): the registered Raw edit
/// ([`crate::KREA_2_EDIT_ID`]) vs the distilled Turbo edit ([`crate::KREA_2_TURBO_EDIT_ID`], driven
/// through the worker's bespoke lane with `distilled = true`). Pure so the routing is unit-testable.
fn edit_pid_model_id(distilled: bool) -> &'static str {
    if distilled {
        crate::KREA_2_TURBO_EDIT_ID
    } else {
        crate::KREA_2_EDIT_ID
    }
}

/// Krea's classifier-free-guidance velocity combine — the reference `sampling.py:129`
/// `v = v_cond + guidance·(v_cond − v_uncond)`, **NOT** the standard `v_uncond + g·Δ`. Krea's guidance
/// is offset by one: the standard form applies one full step LESS guidance, and at `guidance = 1.0`
/// collapses to exactly `v_cond` (zero effective CFG — the washed-out-render trap). Single source of
/// truth so the Raw inference path (sc-9994) and the trainer preview (`training::render_sample`) can
/// never drift again (mirrors the mlx-gen sc-10009 dedupe). The caller runs this only for
/// `guidance > 0` (a single conditional forward otherwise, matching `cfg = guidance > 0`).
pub(crate) fn krea_cfg_combine(
    v_cond: &Tensor,
    v_uncond: &Tensor,
    guidance: f32,
) -> Result<Tensor> {
    let guided = ((v_cond - v_uncond)? * guidance as f64)?;
    Ok((v_cond + guided)?)
}

/// Resolve the img2img start step — the candle port of `mlx_gen::img2img::init_time_step` (the fork's
/// `Config.init_time_step`): for `strength` in `(0, 1]`, `max(1, floor(num_steps · strength))`; otherwise
/// `0` (pure txt2img — the blend at `σ = sigmas[0] = 1.0` is exactly the noise). Higher strength → later
/// start → fewer denoise steps → output stays closer to the reference (the fork's reference-fidelity
/// convention, the INVERSE of SDXL's "higher strength = more change"). Pure, so it is unit-testable.
pub(crate) fn init_time_step(num_steps: usize, strength: Option<f32>) -> usize {
    match strength {
        Some(s) if s > 0.0 => {
            let s = s.clamp(0.0, 1.0);
            // Python `int(num_steps * strength)` truncates toward zero == floor for s >= 0.
            ((num_steps as f32 * s) as usize).max(1)
        }
        _ => 0,
    }
}

/// The img2img noise-interpolation blend — the candle port of `mlx_gen::img2img::add_noise_by_interpolation`
/// (the fork's `LatentCreator.add_noise_by_interpolation`): `(1 − σ)·clean + σ·noise`, seeding the denoise
/// loop at `σ = sigmas[init_time_step]`. `clean` and `noise` share the normalized `[1,16,H/8,W/8]` latent
/// space + f32 dtype.
fn add_noise_by_interpolation(clean: &Tensor, noise: &Tensor, sigma: f32) -> Result<Tensor> {
    let clean_part = (clean * (1.0 - sigma) as f64)?;
    let noise_part = (noise * sigma as f64)?;
    Ok((clean_part + noise_part)?)
}

/// Preprocess an RGB8 reference [`Image`] to the VAE-encoder input `[1, 3, H, W]` in `[-1, 1]` (f32) at the
/// target `(width, height)` — the candle port of `mlx_gen::img2img::preprocess_init_image`. PIL **LANCZOS**
/// resample (`resize_lanczos_u8`, the fork's `scale_to_dimensions`; a no-op when already sized), then
/// `(v/255)·2 − 1`. Distinct from [`image_to_pixels`] (the edit path's BICUBIC vision-preprocess mirror) —
/// the img2img init mirrors the mlx img2img leaf's LANCZOS filter exactly.
fn preprocess_img2img_init(
    im: &Image,
    target_w: u32,
    target_h: u32,
    device: &Device,
) -> Result<Tensor> {
    let (w, h) = (target_w as usize, target_h as usize);
    if im.pixels.len()
        != candle_gen::gen_core::imageops::checked_image_buffer_len(
            im.width as usize,
            im.height as usize,
            3,
        )
        .unwrap_or(usize::MAX)
    {
        return Err(CandleError::Msg(format!(
            "krea img2img: reference pixel buffer {} != {}x{}x3",
            im.pixels.len(),
            im.width,
            im.height
        )));
    }
    let resized: Vec<f32> = if (im.width, im.height) == (target_w, target_h) {
        im.pixels.iter().map(|&p| p as f32).collect()
    } else {
        resize_lanczos_u8(&im.pixels, im.height as usize, im.width as usize, h, w)?
    };
    // HWC [0,255] → CHW [-1,1] (`(v/255)·2 − 1` == `v/127.5 − 1`), matching the fork's `to_array`.
    let mut chw = vec![0f32; 3 * h * w];
    for c in 0..3 {
        for y in 0..h {
            for x in 0..w {
                chw[c * h * w + y * w + x] = resized[(y * w + x) * 3 + c] / 127.5 - 1.0;
            }
        }
    }
    Ok(Tensor::from_vec(chw, (1, 3, h, w), device)?)
}

/// Seeded initial Gaussian latent noise `[1, 16, H/8, W/8]` (f32; the VAE's 8× spatial compression).
/// Deterministic, launch-portable CPU RNG (sc-3673 parity), exactly as the z-image/ideogram/boogu
/// providers. The model layer offsets `seed` per image in a batch (reference `seed + i`).
fn init_noise(height: u32, width: u32, seed: u64, device: &Device) -> Result<Tensor> {
    let (lat_h, lat_w) = (
        (height / SPATIAL_SCALE) as usize,
        (width / SPATIAL_SCALE) as usize,
    );
    let n = LATENT_CHANNELS * lat_h * lat_w;
    let mut rng = StdRng::seed_from_u64(seed);
    let noise = candle_gen::seeded_normal_vec(&mut rng, n);
    Ok(
        Tensor::from_vec(noise, (1, LATENT_CHANNELS, lat_h, lat_w), &Device::Cpu)?
            .to_device(device)?,
    )
}

/// Convert a decoded pixel tensor `[1, 3, H, W]` in `[-1, 1]` (f32) → RGB8 [`Image`]. Shared by the
/// native VAE decode (`QwenVae::decode` applies the per-channel `z·std + mean` de-normalize internally)
/// and the PiD super-resolving decode (which already emits `[-1, 1]` pixels, possibly at 4× the size).
/// The reference `clamp(-1,1)·0.5 + 0.5` denormalize is the `(x+1)·127.5` below; the output size is read
/// from the tensor, never assumed (PiD may be larger than VAE-native).
pub(crate) fn to_image(decoded: &Tensor) -> Result<Image> {
    let scaled = ((decoded.clamp(-1f32, 1f32)? + 1.0)? * 127.5)?;
    let img = candle_gen::round_rgb8(&scaled)?;
    let img = img.i(0)?.to_device(&Device::Cpu)?;
    let (c, h, w) = img.dims3()?;
    if c != 3 {
        return Err(CandleError::Msg(format!(
            "krea: expected 3 channels, got {c}"
        )));
    }
    let pixels = img.permute((1, 2, 0))?.flatten_all()?.to_vec1::<u8>()?;
    Ok(Image {
        width: w as u32,
        height: h as u32,
        pixels,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    /// sc-12828: `load_text` stores the Qwen3-VL TE at **bf16**, not f32 — the deliberate, non-default
    /// choice the ~7.6 GB/tier resident saving rides on (the encoder still computes f32 via
    /// `forward_upcast`, bit-identical — see `text_encoder::tests`). Pinned so a revert to an f32 store
    /// (which would silently restore the tax) fails here rather than only showing up in a VRAM probe.
    #[test]
    fn te_is_stored_bf16() {
        assert_eq!(TE_STORE_DTYPE, DType::BF16);
        assert_eq!(DIT_DTYPE, DType::BF16);
    }

    /// sc-12828 H1: `load_te_weights` stores bf16 **only** when the snapshot's TE ships bf16 on disk
    /// (probed from a norm weight); a wider snapshot keeps its f32 store rather than being silently
    /// truncated to bf16. Guards the premise the bit-identity claim rests on.
    #[test]
    fn te_store_falls_back_to_f32_off_bf16_disk() {
        fn write_te(dir: &std::path::Path, dtype: DType) {
            let te = dir.join("text_encoder");
            std::fs::create_dir_all(&te).unwrap();
            let w = Tensor::ones(4usize, DType::F32, &Device::Cpu)
                .unwrap()
                .to_dtype(dtype)
                .unwrap();
            let mut m = std::collections::HashMap::new();
            m.insert(
                "language_model.layers.0.input_layernorm.weight".to_string(),
                w,
            );
            candle_gen::candle_core::safetensors::save(&m, te.join("model.safetensors")).unwrap();
            std::fs::write(te.join("config.json"), b"{}").unwrap();
        }
        let base = std::env::temp_dir().join(format!(
            "krea_te_store_{}_{:?}",
            std::process::id(),
            std::thread::current().id()
        ));

        let bf = base.join("bf16");
        write_te(&bf, DType::BF16);
        assert_eq!(
            load_te_weights(&bf, &Device::Cpu).unwrap().dtype(),
            DType::BF16,
            "bf16-on-disk TE → bf16 store (the win)"
        );

        let f = base.join("f32");
        write_te(&f, DType::F32);
        assert_eq!(
            load_te_weights(&f, &Device::Cpu).unwrap().dtype(),
            DType::F32,
            "non-bf16 TE → f32 store (never silently truncated)"
        );

        let _ = std::fs::remove_dir_all(&base);
    }

    /// F-177 (sc-12089): the PiD student is loaded only when the request will actually decode through
    /// it, so a `Sequential` generate that never asked for PiD does not pay for the student + its
    /// gemma-2-2b caption encoder — per generate, resident through the whole denoise, inside the peak
    /// the path exists to bound.
    ///
    /// The resident loader passes `use_pid = true` unconditionally and that is correct, not an
    /// oversight: it builds one cached set BEFORE any request exists, so the overlay has to be there for
    /// whichever later request wants it.
    #[test]
    fn pid_loads_only_when_the_request_uses_it() {
        let spec = PidWeights {
            checkpoint: gen_core::WeightsSource::File("/pid.safetensors".into()),
            gemma: gen_core::WeightsSource::Dir("/gemma".into()),
        };

        // Opted in at load AND wanted by this request → load it.
        assert!(pid_to_load(Some(&spec), true).is_some());
        // Opted in at load but NOT wanted by this request → skip it. This is the F-177 arm: before the
        // fix this loaded the engine and `resolve_pid_decoder` then returned `None` for it, so not a byte
        // was ever read.
        assert!(pid_to_load(Some(&spec), false).is_none());
        // Never opted in → nothing to load, whatever the request asked for. (`use_pid` with no `pid`
        // spec is `resolve_pid_decoder`'s error to report, not a reason to load anything here.)
        assert!(pid_to_load(None, true).is_none());
        assert!(pid_to_load(None, false).is_none());
    }

    /// SPIKE (sc-8596): all-ones tap weights are an identity reweight (byte-exact), and a per-layer
    /// scalar scales exactly its select-layer slice along axis 2.
    #[test]
    fn apply_tap_weights_scales_layer_axis() {
        // context [b=1, n_tok=2, num_layers=3, hidden=2], all ones.
        let ctx = Tensor::ones((1, 2, 3, 2), DType::F32, &Device::Cpu).unwrap();

        // Identity: all-ones weights reproduce the input.
        let id = apply_tap_weights(&ctx, &[1.0, 1.0, 1.0]).unwrap();
        assert_eq!(
            id.flatten_all().unwrap().to_vec1::<f32>().unwrap(),
            ctx.flatten_all().unwrap().to_vec1::<f32>().unwrap()
        );

        // Per-layer scale: layer 0 → 2, layer 1 → 0, layer 2 → 0.5, along axis 2 only.
        let out = apply_tap_weights(&ctx, &[2.0, 0.0, 0.5]).unwrap();
        // out[.., layer, ..] equals the weight for that layer (input was all ones).
        let l0 = out
            .narrow(2, 0, 1)
            .unwrap()
            .flatten_all()
            .unwrap()
            .to_vec1::<f32>()
            .unwrap();
        let l1 = out
            .narrow(2, 1, 1)
            .unwrap()
            .flatten_all()
            .unwrap()
            .to_vec1::<f32>()
            .unwrap();
        let l2 = out
            .narrow(2, 2, 1)
            .unwrap()
            .flatten_all()
            .unwrap()
            .to_vec1::<f32>()
            .unwrap();
        assert!(l0.iter().all(|&x| (x - 2.0).abs() < 1e-6));
        assert!(l1.iter().all(|&x| x.abs() < 1e-6));
        assert!(l2.iter().all(|&x| (x - 0.5).abs() < 1e-6));
    }

    /// sc-11878: the single "text style" gain scalar maps to the validated tap ramp — g=1 is a no-op
    /// (all ones), g=1.75 reproduces the spike's early_ramp (1.75→0.25), and out-of-range clamps.
    #[test]
    fn tap_gain_weights_maps_scalar_to_ramp() {
        // g = 1 → all ones (no-op).
        assert_eq!(tap_gain_weights(1.0, 12), vec![1.0; 12]);

        // g = 1.75 → linear ramp 1.75 → 0.25 across 12 taps (the spike's early_ramp).
        let w = tap_gain_weights(1.75, 12);
        assert!((w[0] - 1.75).abs() < 1e-6);
        assert!((w[11] - 0.25).abs() < 1e-6);
        // Monotonically decreasing, centered on 1.0 at the midpoint.
        assert!(w.windows(2).all(|p| p[0] > p[1]));
        assert!((w[0] + w[11] - 2.0).abs() < 1e-6);

        // g = 0.5 → mirror ramp 0.5 → 1.5 (late emphasis).
        let lo = tap_gain_weights(0.5, 12);
        assert!((lo[0] - 0.5).abs() < 1e-6);
        assert!((lo[11] - 1.5).abs() < 1e-6);

        // Clamp: 3.0 → 1.75, 0.0 → 0.25.
        assert!((tap_gain_weights(3.0, 12)[0] - 1.75).abs() < 1e-6);
        assert!((tap_gain_weights(0.0, 12)[0] - 0.25).abs() < 1e-6);

        // Degenerate n.
        assert_eq!(tap_gain_weights(1.5, 1), vec![1.5]);
    }

    /// `maybe_apply_style_gain` is a no-op for `None` and for a gain within 1e-4 of 1.0, and reweights
    /// otherwise (byte-identical to the explicit `apply_tap_weights` with the mapped ramp).
    #[test]
    fn maybe_apply_style_gain_noop_and_apply() {
        let ctx = Tensor::ones((1, 2, 12, 4), DType::F32, &Device::Cpu).unwrap();
        let flat = |t: &Tensor| t.flatten_all().unwrap().to_vec1::<f32>().unwrap();

        // None → untouched.
        assert_eq!(
            flat(&maybe_apply_style_gain(ctx.clone(), None).unwrap()),
            flat(&ctx)
        );

        // g ≈ 1 → untouched (no-op fast path).
        assert_eq!(
            flat(&maybe_apply_style_gain(ctx.clone(), Some(1.00005)).unwrap()),
            flat(&ctx)
        );

        // g = 1.5 → equals the explicit ramp apply.
        let got = maybe_apply_style_gain(ctx.clone(), Some(1.5)).unwrap();
        let want = apply_tap_weights(&ctx, &tap_gain_weights(1.5, 12)).unwrap();
        assert_eq!(flat(&got), flat(&want));
    }

    /// A mis-sized weight vector must fail loudly (not silently truncate/broadcast).
    #[test]
    fn apply_tap_weights_rejects_wrong_len() {
        let ctx = Tensor::ones((1, 2, 3, 2), DType::F32, &Device::Cpu).unwrap();
        assert!(apply_tap_weights(&ctx, &[1.0, 1.0]).is_err());
        assert!(apply_tap_weights(&ctx, &[1.0; 4]).is_err());
    }

    /// `image_to_pixels` normalizes an RGB8 image to `[1, 3, H, W]` in `[-1, 1]` and, when the source is
    /// already at the target size, maps pixel `v` → `v/127.5 − 1` per channel with no resample (a mid-gray
    /// 128 → ~0.0039, black 0 → −1, white 255 → +1).
    #[test]
    fn image_to_pixels_normalizes_and_shapes() {
        // 2×2 image, channel-constant: R=0, G=128, B=255.
        let mut px = Vec::new();
        for _ in 0..4 {
            px.extend_from_slice(&[0u8, 128, 255]);
        }
        let im = Image {
            width: 2,
            height: 2,
            pixels: px,
        };
        let t = image_to_pixels(&im, 2, 2, &Device::Cpu).unwrap();
        assert_eq!(t.dims(), &[1, 3, 2, 2]);
        let v = t.flatten_all().unwrap().to_vec1::<f32>().unwrap();
        // CHW: first 4 = R (−1), next 4 = G (~0), last 4 = B (+1).
        assert!(v[0..4].iter().all(|&x| (x - (-1.0)).abs() < 1e-6));
        assert!(v[4..8].iter().all(|&x| x.abs() < 0.01));
        assert!(v[8..12].iter().all(|&x| (x - 1.0).abs() < 1e-6));
    }

    #[test]
    fn image_to_pixels_rejects_bad_buffer() {
        let im = Image {
            width: 4,
            height: 4,
            pixels: vec![0u8; 10],
        };
        assert!(image_to_pixels(&im, 4, 4, &Device::Cpu).is_err());
    }

    #[test]
    fn reference_count_enforces_fixed_order_cap() {
        // 0 → error (an edit needs a source); 1 and 2 ok; 3 → error (past the image 1 / image 2 cap).
        assert!(check_reference_count(0).is_err());
        assert!(check_reference_count(1).is_ok());
        assert!(check_reference_count(2).is_ok());
        let err = check_reference_count(3).unwrap_err().to_string();
        assert!(err.contains('2'), "names the cap: {err}");
        assert!(
            err.contains("image 1") && err.contains("image 2"),
            "documents order: {err}"
        );
    }

    /// The Krea CFG combine is the reference `cond + g·(cond − uncond)`, not the standard
    /// `uncond + g·Δ`. With cond = 2, uncond = 1 (Δ = 1): g = 1 → 3 (the standard form would give 2 —
    /// exactly `cond` — which is why the shared default `sample_guidance_scale = 1.0` washed previews
    /// out, sc-10009); a larger g pushes further from cond, away from uncond; g = 0 → cond.
    #[test]
    fn cfg_combine_is_reference_offset_by_one() {
        let cond = Tensor::from_vec(vec![2.0f32], (1,), &Device::Cpu).unwrap();
        let uncond = Tensor::from_vec(vec![1.0f32], (1,), &Device::Cpu).unwrap();
        for (g, want) in [(1.0f32, 3.0f32), (3.5, 5.5), (0.0, 2.0)] {
            let v = krea_cfg_combine(&cond, &uncond, g).unwrap();
            let got = v.to_vec1::<f32>().unwrap()[0];
            assert!((got - want).abs() < 1e-5, "g={g}: got {got}, want {want}");
        }
    }

    /// The edit path names the right surface for the PiD decode-seam error (sc-11197): the registered
    /// Raw edit id when undistilled, the distilled Turbo-edit tag when `distilled = true`.
    #[test]
    fn edit_pid_model_id_names_the_surface() {
        assert_eq!(edit_pid_model_id(false), crate::KREA_2_EDIT_ID);
        assert_eq!(edit_pid_model_id(true), crate::KREA_2_TURBO_EDIT_ID);
    }

    /// The image-edit decode seam honors `req.use_pid` exactly like `render` / `render_base` (sc-11197,
    /// F-088). Before the fix `render_edit` never consulted `use_pid` and always decoded with the native
    /// VAE — the descriptor-accepts / render-drops trap. This asserts the shared seam `render_edit` now
    /// calls (`resolve_pid_decoder` with the edit id): `use_pid = true` with no loaded engine is a hard
    /// error (no longer silently dropped), and `use_pid = false` resolves to the native path (`None`).
    #[test]
    fn edit_decode_seam_honors_use_pid() {
        // use_pid requested but no PiD engine loaded → the same hard error the txt2img siblings raise
        // (previously `render_edit` swallowed the flag and returned a native-resolution image).
        let want_pid = GenerationRequest {
            prompt: "make it autumn".into(),
            width: 512,
            height: 512,
            use_pid: true,
            ..Default::default()
        };
        // (`PidDecoder` is not `Debug`, so match the seam result rather than `unwrap_err`.)
        let err =
            match candle_gen_pid::resolve_pid_decoder(None, &want_pid, 0, edit_pid_model_id(false))
            {
                Ok(_) => panic!("use_pid = true with no loaded PiD engine must be a hard error"),
                Err(e) => e.to_string(),
            };
        assert!(
            err.contains(crate::KREA_2_EDIT_ID) && err.contains("use_pid"),
            "use_pid on the edit id must error when unloaded, naming the surface: {err}"
        );

        // use_pid unset → the native QwenVae decode (no decoder resolved), decoupled from any engine.
        let no_pid = GenerationRequest {
            prompt: "make it autumn".into(),
            width: 512,
            height: 512,
            use_pid: false,
            ..Default::default()
        };
        let resolved =
            candle_gen_pid::resolve_pid_decoder(None, &no_pid, 0, edit_pid_model_id(false));
        assert!(
            matches!(resolved, Ok(None)),
            "use_pid = false must resolve to the native decode path (None)"
        );
    }

    /// The Raw schedule uses the resolution-dynamic mu (vs Turbo's fixed 1.15): at 1024² the image-token
    /// count is `(1024/16)² = 4096`, so `mu = dynamic_mu(4096) = 0.90625`, and the native (unscheduled)
    /// sigmas match the reference `timesteps(seq_len=4096)` — a descending `[1.0 … 0.0]` of length
    /// `steps + 1`. Distinct from `turbo_sigmas`, confirming Raw is not on the distilled fixed-mu curve.
    #[test]
    fn base_schedule_is_resolution_dynamic() {
        let sig = base_schedule(4, 1024, 1024, None);
        assert_eq!(sig.len(), 5);
        assert_eq!(sig.first().copied(), Some(1.0));
        assert_eq!(sig.last().copied(), Some(0.0));
        // Reference `timesteps(seq_len=4096, steps=4)` at f64 precision (narrowed to the f32 the sampler
        // stores) — the same values schedule.rs asserts for the dynamic-mu path.
        let want = [1.0f64, 0.88130659, 0.71223223, 0.45205718, 0.0];
        for (i, (&g, w)) in sig.iter().zip(want).enumerate() {
            assert!((g as f64 - w).abs() < 1e-5, "sigma[{i}] = {g}, want {w}");
        }
        assert_ne!(
            sig,
            turbo_sigmas(4),
            "Raw dynamic-mu differs from Turbo fixed-mu"
        );
    }

    /// `init_time_step` is the fork's `max(1, floor(steps·strength))` for strength in (0,1], else 0.
    /// Higher strength → later start (closer to the reference); a tiny positive strength still starts at
    /// 1 (never 0, which is txt2img); >1 clamps to full; None / 0 is pure txt2img (start 0).
    #[test]
    fn init_time_step_is_the_fork_convention() {
        assert_eq!(init_time_step(8, None), 0);
        assert_eq!(init_time_step(8, Some(0.0)), 0);
        assert_eq!(init_time_step(8, Some(-0.5)), 0);
        assert_eq!(init_time_step(8, Some(0.5)), 4);
        assert_eq!(init_time_step(8, Some(1.0)), 8);
        // floor(8·0.01) = 0 → clamped up to the minimum denoise start of 1.
        assert_eq!(init_time_step(8, Some(0.01)), 1);
        // strength clamps to 1.0 before the multiply.
        assert_eq!(init_time_step(8, Some(2.0)), 8);
        assert_eq!(init_time_step(52, Some(0.5)), 26);
    }

    /// `add_noise_by_interpolation` is the `(1−σ)·clean + σ·noise` blend: σ=0 → clean, σ=1 → noise,
    /// σ=0.5 → the midpoint.
    #[test]
    fn add_noise_by_interpolation_blends_clean_and_noise() {
        let clean = Tensor::from_vec(vec![2.0f32, 4.0], (2,), &Device::Cpu).unwrap();
        let noise = Tensor::from_vec(vec![0.0f32, 0.0], (2,), &Device::Cpu).unwrap();
        for (sigma, want) in [
            (0.0f32, [2.0f32, 4.0]),
            (1.0, [0.0, 0.0]),
            (0.5, [1.0, 2.0]),
        ] {
            let out = add_noise_by_interpolation(&clean, &noise, sigma).unwrap();
            let got = out.to_vec1::<f32>().unwrap();
            assert!(
                (got[0] - want[0]).abs() < 1e-6 && (got[1] - want[1]).abs() < 1e-6,
                "σ={sigma}: got {got:?}, want {want:?}"
            );
        }
    }

    /// `preprocess_img2img_init` normalizes an RGB8 reference to `[1, 3, H, W]` in `[-1, 1]` (the same
    /// `v/127.5 − 1` mapping as the edit `image_to_pixels`, but the img2img LANCZOS filter), and rejects a
    /// malformed pixel buffer. At the source size it is a no-op resize (R=0 → −1, G=128 → ~0, B=255 → +1).
    #[test]
    fn preprocess_img2img_init_normalizes_and_shapes() {
        let mut px = Vec::new();
        for _ in 0..4 {
            px.extend_from_slice(&[0u8, 128, 255]);
        }
        let im = Image {
            width: 2,
            height: 2,
            pixels: px,
        };
        let t = preprocess_img2img_init(&im, 2, 2, &Device::Cpu).unwrap();
        assert_eq!(t.dims(), &[1, 3, 2, 2]);
        let v = t.flatten_all().unwrap().to_vec1::<f32>().unwrap();
        assert!(v[0..4].iter().all(|&x| (x - (-1.0)).abs() < 1e-6));
        assert!(v[4..8].iter().all(|&x| x.abs() < 0.01));
        assert!(v[8..12].iter().all(|&x| (x - 1.0).abs() < 1e-6));

        let bad = Image {
            width: 4,
            height: 4,
            pixels: vec![0u8; 10],
        };
        assert!(preprocess_img2img_init(&bad, 4, 4, &Device::Cpu).is_err());
    }
}
