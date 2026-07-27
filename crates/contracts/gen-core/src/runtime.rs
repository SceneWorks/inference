//! Shared load/exec types used by both [`Generator`](crate::generator::Generator) and
//! [`Transform`](crate::transform::Transform): where weights come from, quantization +
//! precision knobs, adapter specs, cooperative cancellation, and progress events.

use std::collections::BTreeMap;
use std::path::PathBuf;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;

/// Where a model's weights come from — **always a local, already-provisioned path**. There is
/// deliberately **no** hub-fetch variant: inference never self-fetches weights and has no knowledge
/// of any download cache (epic 13657). A consumer resolves and stages every path — the base
/// `weights`, each typed overlay (control / ip_adapter / …), and every [`LoadSpec::components`]
/// entry — before calling `load`, and a missing component is a load-time contract error
/// ([`crate::control::require_component`]), never a mid-render fetch. (The previously-reserved
/// sc-2340 hub-fetch direction is permanently rejected.)
#[derive(Clone, Debug)]
pub enum WeightsSource {
    /// A directory of (possibly sharded) `.safetensors`.
    Dir(PathBuf),
    /// A single `.safetensors` file.
    File(PathBuf),
}

/// Quantization tier a load may request. [`Q4`](Self::Q4)/[`Q8`](Self::Q8) are the group-wise
/// affine int tiers; [`Nvfp4`](Self::Nvfp4) is the NVFP4 FP4 tensor-core tier
/// (epic 11037).
///
/// **A quant tier is a creative choice — a distinct, additive tier, never a silent numerics swap
/// (epic 11037 SC#5).** [`Nvfp4`](Self::Nvfp4) was added under the **Option A** packaging decision of
/// sc-11042: NVFP4 is exposed as its *own* user-selectable tier, **not** a Blackwell execution backend
/// auto-substituted for [`Q4`](Self::Q4). NVFP4's numerics differ from int4-affine `q4` (E2M1 4-bit
/// elements + FP8-E4M3 block scales, W4A4 regime), so auto-swapping `q4` → NVFP4 on `sm_120` would
/// silently change a picked tier's output — the SC#5 violation Option A avoids. Adding this variant
/// changes **no** existing tier's numerics or behavior; each of `Q4`/`Q8` maps exactly as before.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Quant {
    Q4,
    Q8,
    /// The **NVFP4 FP4** tier (epic 11037, sc-11042 Option A) — E2M1 4-bit elements over 16-element
    /// blocks with FP8-E4M3 micro-scales + an FP32 per-tensor scale (~4.5 effective bits/weight). A
    /// *distinct* creative-choice tier, not an int4-affine equivalent. Served **natively packed** by
    /// candle-gen's `Nvfp4Linear` (the packed-forward path, resident at the NVFP4 footprint — never a
    /// dequant→bf16 dense expansion, epic 11037 SC#6) through the sc-11039 cuBLASLt FP4 GEMM on
    /// consumer Blackwell `sm_120`; on other hardware it falls back cleanly. Surfaced through the
    /// candle-gen catalog only under the `cuda` feature — the MLX/macOS runtime (no FP4 hardware) and
    /// the CPU candle bundle (no FP4 compute) do not offer it.
    Nvfp4,
}

impl Quant {
    /// Element bit-width of the tier. For [`Q4`](Self::Q4)/[`Q8`](Self::Q8) this is the width passed to
    /// the MLX affine quantizer. [`Nvfp4`](Self::Nvfp4) reports `4` (its E2M1 elements are 4-bit) but is
    /// **not** an MLX-quantizer tier — it carries FP8 block scales + an FP32 per-tensor scale
    /// (~4.5 *effective* bits/weight) and is served by candle-gen's NVFP4 packed path, not the MLX
    /// affine quantizer; do not route `Nvfp4` through an MLX `quantize(bits)` call on this width alone.
    pub fn bits(self) -> i32 {
        match self {
            Quant::Q4 => 4,
            Quant::Q8 => 8,
            Quant::Nvfp4 => 4,
        }
    }
}

/// Compute precision for dense (non-quantized) weights.
///
/// [`Bf16`](Self::Bf16) doubles as the registry's **"dense default / no precision override"
/// sentinel**, not a literal request for bf16 tensors: each provider maps it to its own native
/// dense dtype. Most providers do run bf16 under it (e.g. sensenova), but the SDXL-family loaders
/// (kolors, instantid) run **fp16** — they still gate on `Bf16` and reject `Fp32` because a
/// precision override is not wired, then load at `Dtype::Float16`. So an audit of dtype behavior
/// through `LoadSpec` must read `Bf16` as "the provider's default dense dtype", which is not
/// universally bf16. (A distinct `Precision::Default`/`Dense` sentinel would make this explicit but
/// would touch every provider's match arm — deferred; this note is the documented contract.)
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum Precision {
    /// Dense default — the provider's native dense dtype (bf16 for most, fp16 for the SDXL family).
    /// See the type-level note: this is the "no override" sentinel, not a literal bf16 request.
    #[default]
    Bf16,
    /// Full-precision override, honored only by providers that wire it (others reject it at `load`).
    Fp32,
}

/// Component-residency strategy for a load (epic 10765 Phase 1, sc-10769/sc-10821). The default keeps
/// every model component resident for the whole generation (fast, cross-request cached). `Sequential`
/// asks a provider that supports it to load→use→DROP each heavy component in phase order (text encoder →
/// transformer/UNet → VAE) so peak VRAM is bounded to the largest single working set instead of the sum,
/// letting a small card run a model that would OOM resident — at the cost of the cross-request weight
/// cache. Advisory, never an error: a provider that has not wired it treats `Sequential` as `Resident`.
/// Whether a given engine actually honors it is not FLUX/backend-specific — it is advertised per model
/// via [`Capabilities::supports_sequential_offload`](crate::generator::Capabilities::supports_sequential_offload),
/// which a consumer reads to tell "bounds peak memory here" from "no-op fallback" (sc-11126).
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum OffloadPolicy {
    /// All components co-resident for the whole generation (today's behavior). Fast; keeps the cache.
    #[default]
    Resident,
    /// Load→use→drop each heavy component in phase order to minimize peak VRAM. Advisory: a provider
    /// that has not wired it falls back to `Resident`.
    Sequential,
}

/// How to load a model. `weights` is required; everything else defaults to dense bf16. The
/// device is the process-default Metal GPU — the crate runs single-device (the MLX default
/// device is not thread-safe; the worker serializes jobs per thread).
#[derive(Clone, Debug)]
pub struct LoadSpec {
    pub weights: WeightsSource,
    pub quantize: Option<Quant>,
    pub precision: Precision,
    /// Auxiliary control-branch weights overlaid onto the base model at load time — a ControlNet
    /// checkpoint applied on top of `weights` (e.g. Z-Image's Fun-Controlnet-Union safetensors).
    /// `None` for the plain base model; a control-variant loader requires it. A load-time model
    /// *component* (it alters the graph), distinct from [`adapters`](Self::adapters) below, which
    /// are forward-time residual overlays on existing linears.
    pub control: Option<WeightsSource>,
    /// **Additional** ControlNet checkpoints for MultiControlNet (sc-3378) — used by providers that
    /// sum several control branches (the SDXL provider). These are loaded *after* [`control`](Self::control)
    /// and paired, in order, with the request's `Conditioning::Control` images (the diffusers
    /// `MultiControlNetModel` order semantics: branch *i* ← the *i*-th `Control`). Empty for the
    /// single-branch case (then only `control` is used); providers that do not support multi-control
    /// (Z-Image / Qwen union checkpoints) ignore this field.
    pub extra_controls: Vec<WeightsSource>,
    /// Auxiliary **IP-Adapter** weights overlaid at load time (sc-3059) — the image-prompt
    /// conditioning checkpoint (image encoder + Resampler + decoupled cross-attn K/V), e.g. an
    /// `h94/IP-Adapter`-layout snapshot dir. `None` for the plain base model. Like
    /// [`control`](Self::control), a load-time graph *component* (it adds K/V projections to the
    /// cross-attention), distinct from forward-time [`adapters`](Self::adapters).
    pub ip_adapter: Option<WeightsSource>,
    /// LoRA/LoKr adapters baked onto the model at load time. Multiples + mixed LoRA/LoKr stack by
    /// construction (see the provider `adapters` modules). Applied during `load` on the still-mutable
    /// model — the seam, since `Generator::generate`/`Transform::apply` take `&self` and the frozen
    /// fork likewise applies adapters in its initializer. Changing the adapter set means reloading.
    pub adapters: Vec<AdapterSpec>,
    /// Auxiliary **PiD** (NVIDIA Pixel-Diffusion) decoder weights overlaid at load time (epic 7840) —
    /// the optional super-resolving replacement for the model's VAE decode step. `None` for the plain
    /// VAE-decoding model; `Some` makes the PiD decoder *available*, after which the per-generation
    /// [`crate::GenerationRequest::use_pid`] flag selects it at the decode call site. Like
    /// [`control`](Self::control)/[`ip_adapter`](Self::ip_adapter) it is a load-time component (PiD's
    /// net + Gemma-2 caption encoder are heavy, so they load once and the toggle rides each request);
    /// only providers whose latent space has a PiD backbone read it (Qwen-Image / Krea today —
    /// sc-7845), and they ignore it when the request does not request PiD.
    pub pid: Option<PidWeights>,
    /// Auxiliary **identity-conditioning** sub-model weights (PuLID / InstantID family, sc-8827) — the
    /// EVA-CLIP tower, the identity encoder checkpoint, and the native face-analysis weight dir that a
    /// face-ID provider needs on top of its diffusion backbone. `None` for a plain base model. A
    /// face-ID provider that reads this slot **requires** it: the caller drives every identity path
    /// through the spec (backend-neutral — just paths), and an absent slot (or an absent sub-field) is a
    /// **load-time** error, never a fetch from an env var or a derived on-disk cache (epic 13657, sc-13664 —
    /// the PuLID-FLUX loader dropped its historical `PULID_*` env / HF-cache-derived fallbacks).
    pub identity: Option<IdentityWeights>,
    /// Auxiliary **external text-encoder** snapshot directory (sc-8827) — a separate TE snapshot a
    /// provider loads alongside its main checkpoint, e.g. LTX-2.3's Gemma-3-12B encoder (which is not
    /// bundled in the checkpoint dir). A provider that reads this slot **requires** it: the caller
    /// drives the TE location through the spec (backend-neutral — just a path), and an absent slot is a
    /// **load-time** error, never a process-global env var or a derived on-disk cache scan (epic 13657,
    /// sc-13664 — LTX-2.3 dropped its historical `$LTX_GEMMA_DIR` / HF-cache-derived fallbacks).
    pub text_encoder: Option<WeightsSource>,
    /// Component-residency strategy (epic 10765, sc-10821). [`OffloadPolicy::Resident`] (default) keeps
    /// every component resident for the whole generation; [`OffloadPolicy::Sequential`] asks a supporting
    /// provider to load→use→drop each heavy component after its phase so peak VRAM is the largest single
    /// working set, not the sum. Advisory — a provider that has not wired the residency lifecycle
    /// ignores it and stays `Resident`; [`Capabilities::supports_sequential_offload`](crate::generator::Capabilities::supports_sequential_offload)
    /// advertises which engines honor it (sc-11126). Backend-neutral.
    pub offload_policy: OffloadPolicy,
    /// **Named, caller-provisioned model components** (epic 13657) — the generic, additive home for
    /// the extra weight artifacts a model needs beyond its base `weights` and the typed overlays
    /// above, keyed by a stable component id. The complement of
    /// [`ModelDescriptor::required_components`](crate::generator::ModelDescriptor::required_components):
    /// the descriptor *advertises* which ids a model requires (weights-free, so a consumer knows what
    /// to stage), and this map *carries* the resolved local path for each. A provider reads each id at
    /// load time via [`require_component`](crate::control::require_component); a required id absent
    /// here is a **load-time** contract error, not a mid-render fetch (the whole point of the seam —
    /// it converts e.g. perth's mid-render watermark-weight fetch into a load contract error), and an
    /// unrecognized id is rejected via
    /// [`reject_unknown_components`](crate::control::reject_unknown_components). Default empty; set
    /// with [`with_component`](Self::with_component), mirroring [`with_control`](Self::with_control).
    ///
    /// This is deliberately a `BTreeMap<String, WeightsSource>` (not a typed slot per component and
    /// not a new [`WeightsSource`] hub-fetch variant — both alternatives were rejected in the sc-13591
    /// research): components are model-specific and open-ended, so a generic keyed map lets a new
    /// model declare new ids without a contract edit, while the descriptor's `required_components`
    /// keeps the set discoverable and conformance-checked.
    ///
    /// ## Provider → component-id registry (the reserved ids downstream stories consume)
    ///
    /// This map is the registry of record for component ids. The provisional set (epic 13657):
    ///
    /// | Model | Component ids |
    /// |-------|---------------|
    /// | chatterbox (TTS) | `perth`, `voice_embedding` |
    /// | MOSS tts / tts-realtime | `codec` |
    /// | SDXL | `tokenizer_clip_l`, `tokenizer_clip_bigg`, `vae_fp16_fix` |
    /// | mmaudio | `clip`, `synchformer`, `dit`, `vae`, `vocoder` |
    /// | sensenova (fast) | `distill_lora` |
    /// | LTX-2.3 | `uncensored_enhancer` |
    /// | acestep (Cover) | `sft_cover` |
    ///
    /// sc-13664 wired sensenova's `distill_lora` (the 8-step distill LoRA for `sensenova_u1_8b_fast`,
    /// with a co-located-in-snapshot fallback; **not** a universally-`required_components` id, because a
    /// pre-merged turnkey tier bakes the merge in and needs no LoRA) and LTX-2.3's optional
    /// `uncensored_enhancer` (the amoral 4-bit Gemma enhancer, read on demand when a request sets
    /// `use_uncensored_enhancer`). acestep's `sft_cover` follows the same on-demand shape: the ~7.8 GB
    /// sft Cover snapshot dir, read only for a `Cover` audio-edit request, so it is likewise **not** a
    /// `required_components` id — text2music and the region edit modes load without it. LTX-2.3's
    /// *main* Gemma text encoder rides the typed [`text_encoder`](Self::text_encoder) slot (now
    /// required), not this map. Ids are lowercase
    /// `snake_case` registry identifiers (same shape as a descriptor `id`); a model's declared
    /// `required_components` ids are validated non-empty and unique by the descriptor conformance sweep
    /// ([`model_descriptor_errors`](crate::registry::model_descriptor_errors)).
    pub components: BTreeMap<String, WeightsSource>,
}

/// Where the optional PiD decoder's weights come from (epic 7840). A PiD decoder is tied to a
/// *latent space*, not a model, so a provider in an eligible space points at the converted
/// per-latent-space checkpoint plus the shared Gemma-2-2b caption encoder. Backend-neutral (just
/// paths); the tensor load lives in `mlx-gen-pid`.
#[derive(Clone, Debug)]
pub struct PidWeights {
    /// The converted PiD student checkpoint — a single `.safetensors`
    /// ([`WeightsSource::File`]; `tools/convert_pid.py` output for this latent space).
    pub checkpoint: WeightsSource,
    /// The `gemma-2-2b-it` snapshot **directory** ([`WeightsSource::Dir`]) — the caption encoder PiD
    /// conditions on (must contain the weights + `tokenizer.json`).
    pub gemma: WeightsSource,
}

/// The identity-conditioning sub-model weights a face-ID provider (PuLID / InstantID family) needs on
/// top of its diffusion backbone (F-114). Backend-neutral paths; the tensor load lives in the provider
/// crate.
///
/// Each field is `Option` only so the struct can be built incrementally / defaulted; a provider that
/// reads a field **requires** it — the caller supplies every path through this struct, and an absent
/// field is a **load-time** error (epic 13657, sc-13664). There is no env-var or cache fallback:
/// the old "optional field ⇒ provider `PULID_*` env / HF-cache-derived default" convention was
/// deleted, so a `None` a provider needs fails fast at load rather than silently scanning the disk.
#[derive(Clone, Debug, Default)]
pub struct IdentityWeights {
    /// The identity-encoder checkpoint — a single `.safetensors` (PuLID's
    /// `pulid_flux_v0.9.1.safetensors`). Required by the PuLID-FLUX loader (`None` ⇒ load-time error).
    pub encoder: Option<WeightsSource>,
    /// The converted EVA-CLIP vision tower — a single `.safetensors`. Required by the PuLID-FLUX loader
    /// (`None` ⇒ load-time error).
    pub eva: Option<WeightsSource>,
    /// The native face-analysis weight **directory** ([`WeightsSource::Dir`]) — must contain
    /// `scrfd_10g` / `arcface_iresnet100` / `bisenet_parsing` safetensors. Required by the PuLID-FLUX
    /// loader (`None` ⇒ load-time error).
    pub face_dir: Option<WeightsSource>,
}

impl LoadSpec {
    /// Dense bf16 load from the given source.
    pub fn new(weights: WeightsSource) -> Self {
        Self {
            weights,
            quantize: None,
            precision: Precision::Bf16,
            control: None,
            extra_controls: Vec::new(),
            ip_adapter: None,
            adapters: Vec::new(),
            pid: None,
            identity: None,
            text_encoder: None,
            offload_policy: OffloadPolicy::Resident,
            components: BTreeMap::new(),
        }
    }

    /// Builder-style quantization override.
    pub fn with_quant(mut self, quant: Quant) -> Self {
        self.quantize = Some(quant);
        self
    }

    /// Builder-style component-residency override (epic 10765, sc-10821). [`OffloadPolicy::Sequential`]
    /// asks a supporting provider to load→use→drop each heavy component to cap peak VRAM; the default
    /// [`OffloadPolicy::Resident`] keeps everything co-resident. Which engines honor it is advertised by
    /// [`Capabilities::supports_sequential_offload`](crate::generator::Capabilities::supports_sequential_offload).
    pub fn with_offload_policy(mut self, offload_policy: OffloadPolicy) -> Self {
        self.offload_policy = offload_policy;
        self
    }

    /// Builder-style control-branch overlay (the ControlNet checkpoint over the base `weights`).
    pub fn with_control(mut self, control: WeightsSource) -> Self {
        self.control = Some(control);
        self
    }

    /// Builder-style named component (epic 13657) — stage the caller-provisioned local path for the
    /// component `id` into [`components`](Self::components). Mirrors [`with_control`](Self::with_control);
    /// the id is the stable key a provider reads at load via
    /// [`require_component`](crate::control::require_component). Re-inserting the same id replaces the
    /// prior path (last write wins). See [`components`](Self::components) for the id registry.
    pub fn with_component(mut self, id: impl Into<String>, src: WeightsSource) -> Self {
        self.components.insert(id.into(), src);
        self
    }

    /// Builder-style **additional** ControlNet checkpoint for MultiControlNet (sc-3378) — appends to
    /// [`extra_controls`](Self::extra_controls). Call after [`with_control`](Self::with_control); each
    /// extra branch pairs, in order, with the request's `Conditioning::Control` images. Supported by
    /// the SDXL provider.
    pub fn with_extra_control(mut self, control: WeightsSource) -> Self {
        self.extra_controls.push(control);
        self
    }

    /// Builder-style IP-Adapter overlay (the image-prompt checkpoint dir over the base `weights`).
    pub fn with_ip_adapter(mut self, ip_adapter: WeightsSource) -> Self {
        self.ip_adapter = Some(ip_adapter);
        self
    }

    /// Builder-style LoRA/LoKr adapters to bake on at load time (replaces any already set).
    pub fn with_adapters(mut self, adapters: Vec<AdapterSpec>) -> Self {
        self.adapters = adapters;
        self
    }

    /// Builder-style optional PiD decoder overlay (epic 7840) — the converted per-latent-space PiD
    /// checkpoint + the Gemma-2 caption-encoder snapshot dir. Makes PiD *available*; the per-request
    /// [`crate::GenerationRequest::use_pid`] flag then selects it at decode.
    pub fn with_pid(mut self, checkpoint: WeightsSource, gemma: WeightsSource) -> Self {
        self.pid = Some(PidWeights { checkpoint, gemma });
        self
    }
}

/// A single adapter to stack at load time. Multiples + mixed LoRA/LoKr are supported by
/// construction — see the provider `adapters` modules. Carried by [`LoadSpec::adapters`].
#[derive(Clone, Debug)]
pub struct AdapterSpec {
    pub path: PathBuf,
    pub scale: f32,
    pub kind: AdapterKind,
    /// Per-denoise-pass strength override (LTX-2.3 only). When `Some`, the slice gives this
    /// adapter's strength for each distilled stage (LTX runs a 2-stage denoise, so a length-2
    /// `[stage1, stage2]`); when `None`, [`scale`](Self::scale) is applied uniformly to every pass.
    /// This is the LTX "per-pass strength" feature (sc-2687) — the reference has no per-stage
    /// schedule, so it is net-new. Like [`LoadSpec::control`], it is a model-specific knob on the
    /// shared spec: **only LTX reads it**; every other model ignores it (its denoise is single-pass).
    pub pass_scales: Option<Vec<f32>>,
    /// Which expert of a dual-expert MoE model (Wan2.2 A14B) this adapter targets (sc-2683).
    /// `None` = shared: merged onto **both** the high- and low-noise experts (the reference
    /// `--lora` file → `(loras)+(loras_high/low)`); `Some(High)`/`Some(Low)` = one expert only
    /// (`--lora-high` / `--lora-low`). Like [`pass_scales`](Self::pass_scales), this is a
    /// model-specific knob on the shared spec: **only the Wan MoE models read it**; every
    /// single-stream model ignores it (a `Some(_)` there is surfaced, not silently honored).
    pub moe_expert: Option<MoeExpert>,
}

/// One adapter file's actual provider-side install outcome.
///
/// Providers that can partially accept an adapter publish these reports through
/// [`Generator::adapter_apply_reports`](crate::Generator::adapter_apply_reports) after generation.
/// The report is deliberately tensor-free and additive to the existing generator contract: providers
/// that do not opt in return an empty list, preserving their current behavior.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct AdapterApplyReport {
    /// The adapter file the report describes. This is the resolved local path from
    /// [`AdapterSpec::path`], allowing a consumer to correlate reports with load-order specs without
    /// guessing from file headers.
    pub adapter_path: PathBuf,
    /// Target deltas/factors the provider accepted for this adapter.
    pub applied: usize,
    /// Target stems the provider did not apply. Empty means the provider accepted the whole adapter
    /// surface it recognized.
    pub skipped: Vec<String>,
}

impl AdapterSpec {
    /// A uniform-strength adapter (the common case): [`scale`](Self::scale) on every denoise pass,
    /// no per-pass override, shared across both MoE experts. Equivalent to a literal with
    /// `pass_scales: None, moe_expert: None`.
    pub fn new(path: PathBuf, scale: f32, kind: AdapterKind) -> Self {
        Self {
            path,
            scale,
            kind,
            pass_scales: None,
            moe_expert: None,
        }
    }

    /// Builder-style per-pass strength override (LTX only — see [`pass_scales`](Self::pass_scales)).
    pub fn with_pass_scales(mut self, pass_scales: Vec<f32>) -> Self {
        self.pass_scales = Some(pass_scales);
        self
    }

    /// Builder-style MoE expert target (Wan2.2 A14B only — see [`moe_expert`](Self::moe_expert)).
    pub fn with_moe_expert(mut self, expert: MoeExpert) -> Self {
        self.moe_expert = Some(expert);
        self
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum AdapterKind {
    Lora,
    Lokr,
}

/// One expert of a dual-expert MoE denoiser (Wan2.2 A14B), naming which checkpoint an adapter
/// merges onto. The A14B splits denoising at a noise `boundary` between a **high**-noise expert
/// (early, noisy steps) and a **low**-noise expert (late steps); a trained Wan MoE LoRA ships as a
/// high/low pair (e.g. `*_wan22_high` + `*_wan22_low`). See [`AdapterSpec::moe_expert`].
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum MoeExpert {
    High,
    Low,
}

/// Cooperative cancellation handle threaded into a request; a model checks it between steps
/// and bails early. Cloneable — the caller keeps a handle to cancel an in-flight job.
#[derive(Clone, Default)]
pub struct CancelFlag(Arc<AtomicBool>);

impl CancelFlag {
    pub fn new() -> Self {
        Self::default()
    }

    /// Request cancellation of the in-flight generation.
    pub fn cancel(&self) {
        self.0.store(true, Ordering::Relaxed);
    }

    /// Whether cancellation has been requested.
    pub fn is_cancelled(&self) -> bool {
        self.0.load(Ordering::Relaxed)
    }
}

impl std::fmt::Debug for CancelFlag {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_tuple("CancelFlag")
            .field(&self.is_cancelled())
            .finish()
    }
}

/// One frame of the developing image, handed to a [`PreviewSink`] during denoise.
///
/// The image is a **linear latent→RGB approximation at latent resolution** — for a VAE with an 8×
/// spatial scale that is `width/8 × height/8` (128×128 for a 1024² render), RGB8, three bytes per
/// pixel. It is explicitly **not** a VAE decode: producing it costs one small matmul instead of a
/// full decoder forward, which is what makes a per-step preview affordable. Consumers should
/// upscale it for display and must not treat it as the render's output.
///
/// `current` / `total` are 1-based and mirror [`Progress::Step`], so a consumer can drive one
/// progress indicator from either. `current` advances monotonically and never exceeds `total`; a
/// solver that evaluates the model more than once per step (Heun-family) emits at most one frame
/// per schedule position rather than overrunning the count.
#[derive(Clone, Debug)]
pub struct PreviewFrame {
    /// 1-based schedule position this frame was projected from.
    pub current: u32,
    /// Total denoise steps in this trajectory.
    pub total: u32,
    /// Latent-resolution RGB8 approximation of the developing image.
    pub image: crate::media::Image,
}

/// Per-step preview sink threaded onto a request ([`GenerationRequest::preview`]).
///
/// The [`CancelFlag`] pattern: a cheap cloneable handle carried as a request **field**, not a
/// [`Progress`] variant — `Progress` stays `Copy` and no exhaustive match downstream changes. The
/// inert [`default`](Default::default) is free: a supporting engine gates the projection on
/// [`is_active`](Self::is_active) and does no work at all when nobody is listening.
///
/// **Emission is synchronous and on the denoise thread**, so the closure must return promptly —
/// forward the frame to a channel rather than encoding or rendering inside it. Support is
/// per-engine and opt-in; an engine that never emits is indistinguishable from an inert sink.
///
/// [`GenerationRequest::preview`]: crate::GenerationRequest::preview
#[derive(Clone, Default)]
pub struct PreviewSink(Option<Arc<dyn Fn(PreviewFrame) + Send + Sync>>);

impl PreviewSink {
    /// Build an active sink from a callback. The callback runs on the denoise thread, once per
    /// emitted frame.
    pub fn new(sink: impl Fn(PreviewFrame) + Send + Sync + 'static) -> Self {
        Self(Some(Arc::new(sink)))
    }

    /// Whether anyone is listening. Engines gate the projection work on this so an inert sink
    /// costs one branch per denoise evaluation.
    pub fn is_active(&self) -> bool {
        self.0.is_some()
    }

    /// Deliver one frame. A no-op on an inert sink.
    pub fn emit(&self, frame: PreviewFrame) {
        if let Some(sink) = &self.0 {
            sink(frame);
        }
    }
}

impl std::fmt::Debug for PreviewSink {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_tuple("PreviewSink")
            .field(&self.is_active())
            .finish()
    }
}

/// A progress event streamed to the caller during a long `generate` / `apply`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Progress {
    /// Denoising step `current` of `total` (1-based).
    Step { current: u32, total: u32 },
    /// VAE decode underway (post-denoise).
    Decoding,
    /// A heavy model component is (re)loading (epic 10765, sc-11126). Emitted only under
    /// [`OffloadPolicy::Sequential`], where the residency seam load→use→drops each component *inside*
    /// `generate` — a multi-second, multi-GB step during which no `Step`/`Decoding` event fires, so
    /// without this the UI would freeze silently while a component streams from disk (F-179). The
    /// [`Resident`](OffloadPolicy::Resident) path loads everything before `generate` and never emits it.
    Loading(LoadPhase),
}

/// Which component the residency seam is loading when it emits [`Progress::Loading`] (sc-11126). The
/// `Sequential` lifecycle has two in-`generate` load phases: the phase-A text/vision encoder, then the
/// heavy render bundle (transformer/U-Net + VAE + any control/PiD overlay).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum LoadPhase {
    /// The phase-A prompt encoder (text or vision-language), loaded first and dropped before the
    /// render bundle materializes.
    TextEncoder,
    /// The heavy render bundle — the transformer/U-Net, the VAE, and any control/PiD overlay.
    Renderer,
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::Mutex;

    fn frame(current: u32, total: u32) -> PreviewFrame {
        PreviewFrame {
            current,
            total,
            image: crate::media::Image {
                width: 2,
                height: 1,
                pixels: vec![0, 0, 0, 255, 255, 255],
            },
        }
    }

    /// The inert default is the zero-cost path: engines gate their projection on `is_active`, and
    /// `emit` on an inert sink must not panic (a provider may emit unconditionally).
    #[test]
    fn default_preview_sink_is_inert() {
        let sink = PreviewSink::default();
        assert!(!sink.is_active());
        sink.emit(frame(1, 8)); // must be a no-op, not a panic
    }

    /// An active sink receives every emitted frame, in order, through a clone of the handle — the
    /// `CancelFlag` pattern: cloning the request must not detach the sink.
    #[test]
    fn active_preview_sink_receives_frames_through_a_clone() {
        let seen = Arc::new(Mutex::new(Vec::new()));
        let recorder = Arc::clone(&seen);
        let sink = PreviewSink::new(move |f: PreviewFrame| {
            recorder
                .lock()
                .unwrap()
                .push((f.current, f.total, f.image.pixels.len()));
        });
        assert!(sink.is_active());

        let cloned = sink.clone();
        sink.emit(frame(1, 8));
        cloned.emit(frame(2, 8));

        assert_eq!(*seen.lock().unwrap(), vec![(1, 8, 6), (2, 8, 6)]);
    }

    /// `Debug` reports liveness, never the closure — matching `CancelFlag`'s shape so a request
    /// stays printable.
    #[test]
    fn preview_sink_debug_reports_liveness() {
        assert_eq!(
            format!("{:?}", PreviewSink::default()),
            "PreviewSink(false)"
        );
        assert_eq!(
            format!("{:?}", PreviewSink::new(|_| {})),
            "PreviewSink(true)"
        );
    }
}
