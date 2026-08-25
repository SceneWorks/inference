//! Krea 2 Turbo **pose-ControlNet inference** provider (sc-8464, epic 8459) — candle (Windows/CUDA).
//!
//! The deployable sibling of the sc-8460 spike harness (`examples/krea-control-infer.rs`): loads the
//! frozen Krea 2 Turbo base (through the composable [`KreaTrainDit`] — the same forward the branch
//! trains against) plus a trained [`ControlBranch`] overlay, and renders
//! the standard 8-step CFG-free Turbo denoise conditioned on a rendered OpenPose skeleton.
//!
//! **How it conditions:** the pose skeleton is VAE-encoded (Qwen-Image VAE) into a control latent, then
//! [`forward_with_control`] — a drop-in for the base
//! `dit.forward` — adds the branch residual into the frozen main stream after each of the first N
//! single-stream blocks, scaled by `control_scale` and RMS-clamped at τ (the S0 recipe: τ = 0.15,
//! applied identically train/infer). `control_scale = 0` is engine-proven **byte-identical** to the
//! un-branched base generation at the same seed (the spike's identity contract).
//!
//! Bespoke provider (NOT gen-core-registered), worker-invoked by name — the candle pattern for
//! conditioned surfaces (mirrors [`crate::control_train`]'s trainer and the FLUX.2 control provider).
//! Krea 2 Turbo is CFG-free + distilled few-step: a single guidance-inert forward per step, no
//! negative pass. The base DiT keeps a packed q4/q8 tier packed in VRAM (dequant-on-forward, sc-11727)
//! and the control-branch overlay is packed to the tier the base tier implies (`branch_tier`, sc-11743
//! mechanism / sc-15799 policy) — the studio-trained overlay is published bf16, so a packed base always
//! repacks it at load rather than carrying precision the tier choice did not ask for. `generate` takes
//! `&self` so one load serves many poses; the residual clamp is a fixed recipe constant set at load, not
//! a knob.

use std::path::{Path, PathBuf};

use candle_gen::candle_core::{DType, Device, Tensor};
use candle_gen::gen_core::runtime::CancelFlag;
use candle_gen::gen_core::sampling::TimestepConvention;
use candle_gen::gen_core::{
    AdapterSpec, Image, OffloadPolicy, PreviewSink, Progress, Quant, WeightsSource,
};
use candle_gen::train::flow_match::component_vb;
use candle_gen::{CandleError, Result};
use candle_gen_qwen_image::vae::{QwenVae, QwenVaeEncoder};
use rand::{rngs::StdRng, SeedableRng};

use crate::config::Krea2Config;
use crate::control::{forward_with_control, ControlBranch, DEFAULT_RESIDUAL_CLAMP};
use crate::loader::Weights;
use crate::pipeline::maybe_apply_style_gain;
use crate::pipeline::to_image;
use crate::pipeline::MAX_TEXT_TOKENS;
use crate::text_encoder::{KreaTeConfig, KreaTextEncoder};
use crate::tokenizer::KreaTokenizer;
use crate::train_dit::{KreaTrainDit, KREA_ATTN_CHUNK_BUDGET};
use crate::{load_vae, turbo_sigmas, TURBO_STEPS};

/// Qwen-Image VAE 8× spatial compression (latent side = pixels / 8).
const SPATIAL_SCALE: u32 = 8;
/// Latent channel count (Qwen-Image VAE).
const LATENT_CHANNELS: usize = 16;
/// Width/height must be a multiple of this (VAE 8× × 2×2 patchify), matching the base txt2img guard.
/// Single source of truth = the crate-root [`crate::SIZE_MULTIPLE`] (sc-12612).
use crate::SIZE_MULTIPLE;

/// Default `control_scale` for the distilled CFG-free Turbo base. The S0 spike found the usable band
/// ~0.5–0.75 (widening to ~0.7–0.9 with more data); ship a comfortable mid default and hard-cap the
/// exposed range ≤ 0.85 (over-drive haloes to halftone above that). The worker applies the cap.
pub const DEFAULT_CONTROL_SCALE: f32 = 0.6;

/// Paths to the Krea 2 control checkpoints: the Krea 2 Turbo diffusers snapshot dir (`text_encoder/`,
/// `transformer/`, `vae/`, `tokenizer/`) + the trained control-branch overlay (a single `.safetensors`).
pub struct Krea2ControlPaths {
    /// Krea 2 Turbo diffusers snapshot dir (the deployed base the overlay applies on).
    pub root: PathBuf,
    /// Optional INT8-ConvRot DiT single-file. When present, it replaces `root/transformer` while the
    /// tokenizer, Qwen3-VL text encoder, and Qwen-Image VAE continue to load from `root`, matching the
    /// registered Krea provider's validated ConvRot route. User LoRA/LoKr residuals apply over the
    /// frozen int8 projections before the control branch is composed.
    pub convrot_dit: Option<PathBuf>,
    /// Optional caller-owned native-mmdit Krea DiT. This is mutually exclusive with
    /// [`convrot_dit`](Self::convrot_dit) and composes with the same pose branch and adapter stack.
    pub native_dit: Option<PathBuf>,
    /// The trained control-branch overlay checkpoint (`.safetensors`, e.g. `control_step5000.safetensors`).
    pub control: PathBuf,
    /// User LoRA/LoKr adapters applied **additively** to the frozen base DiT (sc-11720) — a character /
    /// style adapter reshapes the generated subject while the control branch keeps the pose lock. The
    /// control branch is never adapted. Empty ⇒ the stock control build.
    pub adapters: Vec<AdapterSpec>,
    /// The tier the control-branch overlay is packed to and held at in VRAM (dequant-on-forward,
    /// sc-11743 mechanism; sc-15799 policy).
    ///
    /// **Not a memory lever, and not the caller's free choice.** It MUST be
    /// [`gen_core::tier_integrity::control_branch_tier`](candle_gen::gen_core::tier_integrity::control_branch_tier)
    /// of the base tier `root` resolves to: a q8 base carries a q8 branch, a q4 base carries a q8 branch
    /// (the declared, measured floor — a q4 control residual measures "pose-locked; non-pose details
    /// drift"), and a dense base carries a dense branch. `None` therefore means *the base is dense*, not
    /// "the default".
    ///
    /// It was previously the last-resort rung of the worker's control fit ladder, which left a bf16
    /// branch resident on every card with headroom — the branch's projections are 3.30 B params ≈
    /// **6.6 GB** bf16 against ~3.3 GB at q8 (see [`ControlBranch::from_checkpoint_quantized`]),
    /// so that is **~3.3 GB** of precision a q8 render never asked for. (Not the 8.4 GB the catalog's
    /// `branchPackSaveGb` once claimed for it: 8.4 exceeds the whole branch, so it was never a
    /// weight-side quantity. sc-16013 owns the re-measure.)
    /// This provider cannot derive the value itself (it is never told a tier name; the base tier is
    /// implied by `root` and auto-detected from the packed weights), so the worker owns the derivation
    /// and this field is how it is delivered.
    pub branch_tier: Option<Quant>,
    /// Engage sc-6217-style **query-row attention chunking** on the composable base stack + the control
    /// branch (sc-11745) — the fit-ladder rung **between** VAE-decode tiling (sc-11744) and branch-quant
    /// (sc-11743). `false` (the default and the norm) runs each single-stream block's joint `[ctx; img]`
    /// attention unchunked at the i32-guard budget — full speed, the ~11 GB-of-activations 1024² denoise
    /// peak. The worker's Krea control fit-gate (sc-11754) flips this to `true` **only** when the
    /// predicted *denoise*-phase peak exceeds free VRAM, lowering the scores budget to
    /// [`KREA_ATTN_CHUNK_BUDGET`] so each per-block attention block is bounded (a small speed cost, no
    /// quality cost — the chunked result is numerically identical). A **resolution cap** is a separate,
    /// sharper lever the worker owns by choosing smaller render dims; it needs no knob here. On a card
    /// with headroom this stays `false`.
    pub chunk_attention: bool,
    /// Legacy load-time policy retained for source compatibility. The request field is authoritative.
    pub offload_policy: OffloadPolicy,
}

/// One Krea 2 strict-pose control request. Krea 2 Turbo is CFG-free (no guidance / negative pass) —
/// the conditioning knobs beyond the prompt are `control_scale` and the optional `text_style_gain`.
#[derive(Clone)]
pub struct Krea2ControlRequest {
    pub prompt: String,
    pub width: u32,
    pub height: u32,
    pub steps: usize,
    /// How strongly the control branch locks the base (S0 usable ~0.5–0.85). `0.0` ⇒ base passthrough
    /// (byte-identical to un-branched generation at the same seed).
    pub control_scale: f32,
    /// Optional "text style" tap-reweight gain (sc-12009): reweights the 12 stacked Qwen3-VL taps of the
    /// single CFG-free conditional context before the DiT's TextFusion. `None`/g≈1 is a byte-exact no-op;
    /// the worker clamps to the GPU-validated `[0.25, 1.75]`. Mirrors the txt2img/img2img knob.
    pub text_style_gain: Option<f32>,
    pub seed: u64,
    /// Route the final latent→pixel VAE decode through the seam-free **tiled tail** even below the
    /// im2col-overflow threshold (sc-11744). `false` (the default) is the monolithic decode — full speed,
    /// the ~30 GB end-of-render spike. The worker's Krea control fit-ladder (sc-11754) flips this to
    /// `true` **only** when the predicted decode-phase peak exceeds free VRAM — the cheapest rung (a speed
    /// cost, no quality cost) ahead of branch-quant. On a card with headroom it stays `false`.
    pub tile_vae_decode: bool,
    /// Release the prompt encoder before loading the render bundle for this request.
    pub stage_residency: bool,
    /// Cooperative cancellation, checked before each denoise step (the engine contract).
    pub cancel: CancelFlag,
    /// Per-step latent-preview sink (epic 16948, sc-16950) — the bespoke-request twin of
    /// [`gen_core::GenerationRequest::preview`](candle_gen::gen_core::GenerationRequest::preview).
    /// This provider is invoked by name rather than through the registry, so it carries its own field;
    /// the semantics are identical, including that the [`PreviewSink::default`] inert sink is
    /// byte-identical to a render with no preview at all.
    pub preview: PreviewSink,
}

impl Default for Krea2ControlRequest {
    fn default() -> Self {
        Self {
            prompt: String::new(),
            width: 1024,
            height: 1024,
            steps: TURBO_STEPS,
            control_scale: DEFAULT_CONTROL_SCALE,
            text_style_gain: None,
            seed: 0,
            tile_vae_decode: false,
            stage_residency: false,
            cancel: CancelFlag::default(),
            preview: PreviewSink::default(),
        }
    }
}

/// The phase-A Qwen3-VL prompt encoder. Under sequential residency this value drops before any heavy
/// component is loaded.
struct Krea2ControlText {
    tokenizer: KreaTokenizer,
    te: KreaTextEncoder,
}

impl Krea2ControlText {
    fn encode(&self, req: &Krea2ControlRequest) -> Result<Tensor> {
        maybe_apply_style_gain(
            self.te
                .forward(&self.tokenizer.encode_prompt(&req.prompt, MAX_TEXT_TOKENS)?)?,
            req.text_style_gain,
        )
    }
}

/// The heavy render phase: composable Turbo DiT + pose-control branch + both Qwen-Image VAE halves.
/// The control branch deliberately stays beside the DiT rather than spanning the two phases.
struct Krea2ControlHeavy {
    dit: KreaTrainDit,
    branch: ControlBranch,
    vae: QwenVae,
    vae_encoder: QwenVaeEncoder,
}

/// A loaded Krea 2 control model whose residency value exclusively owns either the warm text/heavy pair
/// or the deferred phase loaders. Sequential bounds peak at `max(Qwen3-VL, DiT + branch + VAE)`.
pub struct Krea2Control {
    device: Device,
    residency: candle_gen::Residency<Krea2ControlText, Krea2ControlHeavy>,
    /// Complete caller-prepared identity retained for deferred warm/staged materialization.
    prepared_spec: Option<candle_gen::gen_core::LoadSpec>,
}

impl Krea2Control {
    /// Retain both phase loaders. The first warm request populates the shared pair; a staged request
    /// loads and releases each phase within `generate`.
    pub fn load(paths: &Krea2ControlPaths) -> Result<Self> {
        Self::load_with_text_encoder(paths, None)
    }

    pub fn load_with_text_encoder(
        paths: &Krea2ControlPaths,
        text_encoder: Option<WeightsSource>,
    ) -> Result<Self> {
        // NOTE: the former "INT8-ConvRot does not support LoRA/LoKr" rejection is gone on purpose —
        // sc-18477 wired `install_additive` into the ConvRot arm of `load_control_heavy`, so the
        // combination is now implemented rather than merely accepted. The remaining guard is the
        // one that still holds: at most ONE replacement DiT may be selected.
        if paths.convrot_dit.is_some() && paths.native_dit.is_some() {
            return Err(CandleError::Msg(
                "krea control: select at most one replacement DiT (ConvRot or native)".into(),
            ));
        }
        let text_root = paths.root.clone();
        let source =
            text_encoder.unwrap_or_else(|| WeightsSource::Dir(text_root.join("text_encoder")));
        let text_encoder = resolve_control_text_encoder_source(&text_root, &source)?;
        let device = candle_gen::default_device()?;
        let text_device = device.clone();
        let heavy_root = paths.root.clone();
        let heavy_convrot_dit = paths.convrot_dit.clone();
        let heavy_native_dit = paths.native_dit.clone();
        let heavy_control = paths.control.clone();
        let heavy_adapters = paths.adapters.clone();
        let heavy_branch_tier = paths.branch_tier;
        let heavy_chunk_attention = paths.chunk_attention;
        let heavy_device = device.clone();
        let residency = candle_gen::Residency::request_scoped(
            move |_| load_control_text(&text_root, &text_encoder, &text_device),
            move |_use_pid, _| {
                load_control_heavy(
                    &heavy_root,
                    heavy_convrot_dit.as_deref(),
                    heavy_native_dit.as_deref(),
                    &heavy_control,
                    &heavy_adapters,
                    heavy_branch_tier,
                    heavy_chunk_attention,
                    &heavy_device,
                )
            },
        );
        Ok(Self {
            device,
            residency,
            prepared_spec: None,
        })
    }

    /// Load from the exact caller-prepared specification.
    ///
    /// The compatibility [`Self::load_with_text_encoder`] entry point validates a selected path at
    /// provider construction. Request authors that already prepared an encoder contract must retain
    /// that complete receipt instead: it also pins the selected config, tokenizer, and complete shard
    /// inventory. Keep the full provider construction inside the prepared-file bracket so a compatible
    /// replacement between admission and this bespoke load cannot be silently revalidated.
    pub fn load_with_spec(
        paths: &Krea2ControlPaths,
        spec: &candle_gen::gen_core::LoadSpec,
    ) -> Result<Self> {
        validate_spec_root(&paths.root, spec, "krea control")?;
        let mut model = spec.read_prepared_files_unchanged(|| {
            let control = required_source_path(spec.control.as_ref(), "krea control overlay")?;
            let convrot_dit = spec
                .components
                .get(candle_gen::gen_core::KREA_CONVROT_DIT_COMPONENT)
                .map(|source| required_file_path(source, "krea control ConvRot DiT"))
                .transpose()?;
            Self::load_with_text_encoder(
                &Krea2ControlPaths {
                    root: paths.root.clone(),
                    convrot_dit,
                    // `validate_spec_root` above rejects a `File` base, so a native DiT can only
                    // arrive through the runtime paths — carry the caller's selection rather than
                    // silently dropping it to `None`.
                    native_dit: paths.native_dit.clone(),
                    control,
                    adapters: spec.adapters.clone(),
                    branch_tier: paths.branch_tier,
                    chunk_attention: paths.chunk_attention,
                    offload_policy: paths.offload_policy,
                },
                spec.text_encoder.clone(),
            )
        })?;
        model.prepared_spec = Some(spec.clone());
        Ok(model)
    }

    /// Generate one strict-pose-conditioned image from a rendered OpenPose skeleton. The control image
    /// must already match the request dimensions; the worker renders it at those exact dimensions.
    pub fn generate(
        &self,
        req: &Krea2ControlRequest,
        control_image: &Image,
        on_progress: &mut dyn FnMut(Progress),
    ) -> Result<Image> {
        validate_request(req)?;
        read_with_prepared_spec(self.prepared_spec.as_ref(), || {
            self.residency.run_request_scoped(
                req.stage_residency,
                false,
                &req.cancel,
                false,
                on_progress,
                |text| text.encode(req),
                |_| Ok(self.device.synchronize()?),
                |heavy, context, on_progress| {
                    let result =
                        heavy.render(&self.device, req, control_image, context, on_progress);
                    candle_gen::synchronize_result(&self.device, result)
                },
            )
        })
    }
}

fn read_with_prepared_spec<T>(
    spec: Option<&candle_gen::gen_core::LoadSpec>,
    read: impl FnOnce() -> Result<T>,
) -> Result<T> {
    match spec {
        Some(spec) => spec.read_prepared_files_unchanged(read),
        None => read(),
    }
}

fn required_source_path(source: Option<&WeightsSource>, label: &str) -> Result<PathBuf> {
    match source {
        Some(WeightsSource::Dir(path) | WeightsSource::File(path)) => Ok(path.clone()),
        None => Err(CandleError::Msg(format!(
            "{label}: prepared load spec is missing the required source"
        ))),
    }
}

fn required_file_path(source: &WeightsSource, label: &str) -> Result<PathBuf> {
    match source {
        WeightsSource::File(path) => Ok(path.clone()),
        WeightsSource::Dir(path) => Err(CandleError::Msg(format!(
            "{label}: expected a file source, got directory {}",
            path.display()
        ))),
    }
}

fn validate_spec_root(
    runtime_root: &Path,
    spec: &candle_gen::gen_core::LoadSpec,
    label: &str,
) -> Result<()> {
    match &spec.weights {
        WeightsSource::Dir(admitted_root) if admitted_root == runtime_root => Ok(()),
        WeightsSource::Dir(admitted_root) => Err(CandleError::Msg(format!(
            "{label}: runtime base {} differs from admitted base {}",
            runtime_root.display(),
            admitted_root.display()
        ))),
        WeightsSource::File(_) => Err(CandleError::Msg(format!(
            "{label}: admitted base must be the runtime snapshot directory"
        ))),
    }
}

/// Load strict-pose control around a caller-owned native-mmdit Krea DiT. The imported transformer,
/// selected adapter stack, and control branch are all retained by one resident provider; every
/// selected adapter must apply to the imported base or loading the heavy phase fails loudly.
pub fn load_control_from_native_dit_file(
    dit_file: impl AsRef<Path>,
    base_snapshot_dir: impl AsRef<Path>,
    control: impl AsRef<Path>,
    adapters: &[AdapterSpec],
) -> Result<Krea2Control> {
    Krea2Control::load(&Krea2ControlPaths {
        root: base_snapshot_dir.as_ref().to_path_buf(),
        convrot_dit: None,
        native_dit: Some(dit_file.as_ref().to_path_buf()),
        control: control.as_ref().to_path_buf(),
        adapters: adapters.to_vec(),
        branch_tier: None,
        chunk_attention: false,
        offload_policy: OffloadPolicy::Resident,
    })
}

/// Load the Qwen3-VL text phase exactly once per resident model or once per sequential generation.
fn load_control_text(
    _root: &Path,
    selected_source: &candle_gen::gen_core::ValidatedEncoderSource,
    device: &Device,
) -> Result<Krea2ControlText> {
    let tokenizer = KreaTokenizer::from_validated_source(selected_source, device)?;
    let te_cfg = KreaTeConfig::qwen3_vl_4b();
    let te_w = selected_source.read_unchanged(|source| -> Result<Weights> {
        Ok(match source {
            WeightsSource::Dir(path) => Weights::from_dir(path, device, DType::F32)?,
            WeightsSource::File(path) => Weights::from_file(path, device, DType::F32)?,
        })
    })?;
    let te = KreaTextEncoder::load(&te_w, "language_model", &te_cfg, MAX_TEXT_TOKENS)?;
    drop(te_w);
    Ok(Krea2ControlText { tokenizer, te })
}

fn resolve_control_text_encoder_source(
    root: &Path,
    selected_source: &WeightsSource,
) -> Result<candle_gen::gen_core::ValidatedEncoderSource> {
    let selected = crate::ENCODER_CONTRACT
        .validate_source_against_base(selected_source, root)
        .map_err(CandleError::from)?;
    let builtin = WeightsSource::Dir(root.join("text_encoder"));
    let expected_bits = candle_gen::gen_core::text_encoder_packed_quant_bits(&builtin)?;
    if let Some(bits) = selected.load_time_quant_bits(expected_bits, "krea_2_turbo_control")? {
        return Err(CandleError::Msg(format!(
            "krea_2_turbo_control requires a selected text encoder already packed at Q{bits}; this provider does not repack a dense Krea encoder on the fly"
        )));
    }
    Ok(selected)
}

/// Load the render phase after the text value has dropped on the sequential path.
#[allow(clippy::too_many_arguments)]
fn load_control_heavy(
    root: &Path,
    convrot_dit: Option<&Path>,
    native_dit: Option<&Path>,
    control: &Path,
    adapters: &[AdapterSpec],
    branch_tier: Option<Quant>,
    chunk_attention: bool,
    device: &Device,
) -> Result<Krea2ControlHeavy> {
    let cfg = Krea2Config::from_snapshot(root)?;
    let mut dit = match control_dit_source(convrot_dit, native_dit)? {
        ControlDitSource::Snapshot => {
            let mut dit_w = Weights::from_dir(&root.join("transformer"), device, DType::BF16)?;
            // Diff-patch (`.diff`/`.diff_b`) deltas fold into the dense baseline weights before the DiT
            // builds (the projector filter-bypass is outside the additive residual surface); low-rank
            // user adapters then ride as residuals. The pose control branch is never adapted either way.
            let diff = crate::adapters::fold_diff_patch(&mut dit_w, adapters)?;
            let mut dit = KreaTrainDit::load_inference(&dit_w, &cfg)?;
            drop(dit_w);
            if !adapters.is_empty() {
                crate::adapters::install_additive_with_diff(
                    &mut dit,
                    adapters,
                    &diff.applied_by_spec,
                )?;
            }
            dit
        }
        ControlDitSource::ConvRot(convrot_dit) => {
            // Reuse the registered provider's exact descriptor validation, native-key remap, online
            // Hadamard rotation, shared cuBLASLt context, and sm_89 floor. Only the composable DiT type
            // differs because the pose branch injects residuals between main blocks.
            let int8 = crate::pipeline::ensure_int8_floor(device)?;
            let dit_w = Weights::from_convrot_file(convrot_dit, device, DType::BF16)?
                .with_int8_context(int8);
            crate::convert::validate_transformer(&dit_w, &cfg)?;
            let mut dit = KreaTrainDit::load_inference(&dit_w, &cfg)?;
            if !adapters.is_empty() {
                crate::adapters::install_additive(&mut dit, adapters, 0)?;
            }
            dit
        }
        ControlDitSource::Native(native_dit) => {
            // sc-20651: config in scope so the compiled plan unpads a block-padded import.
            let mut dit_w = Weights::from_native_file_for(
                native_dit,
                device,
                DType::BF16,
                crate::native_mapping::DeclaredLogicalShapes::FromConfig(&cfg),
            )?;
            crate::convert::validate_native_transformer(&dit_w, &cfg)?;
            let diff = crate::adapters::fold_diff_patch(&mut dit_w, adapters)?;
            let mut dit = KreaTrainDit::load_inference(&dit_w, &cfg)?;
            if !adapters.is_empty() {
                crate::adapters::install_additive_with_diff(
                    &mut dit,
                    adapters,
                    &diff.applied_by_spec,
                )?;
            }
            dit
        }
    };

    let mut branch = match branch_tier {
        Some(quant) => ControlBranch::from_checkpoint_quantized(control, &cfg, device, quant)?,
        None => ControlBranch::from_checkpoint(control, &cfg, device)?,
    };
    branch.freeze();
    branch.set_residual_clamp(Some(DEFAULT_RESIDUAL_CLAMP));
    if chunk_attention {
        dit.set_attention_budget(KREA_ATTN_CHUNK_BUDGET);
        branch.set_attention_budget(KREA_ATTN_CHUNK_BUDGET);
    }

    let vae = load_vae(root, device)?;
    let vae_encoder = QwenVaeEncoder::new(component_vb(
        root,
        "vae",
        device,
        DType::F32,
        "krea control infer",
    )?)?;
    Ok(Krea2ControlHeavy {
        dit,
        branch,
        vae,
        vae_encoder,
    })
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum ControlDitSource<'a> {
    Snapshot,
    ConvRot(&'a Path),
    Native(&'a Path),
}

fn control_dit_source<'a>(
    convrot_dit: Option<&'a Path>,
    native_dit: Option<&'a Path>,
) -> Result<ControlDitSource<'a>> {
    match (convrot_dit, native_dit) {
        (Some(_), Some(_)) => Err(CandleError::Msg(
            "krea control: select at most one replacement DiT (ConvRot or native)".into(),
        )),
        (Some(path), None) => Ok(ControlDitSource::ConvRot(path)),
        (None, Some(path)) => Ok(ControlDitSource::Native(path)),
        (None, None) => Ok(ControlDitSource::Snapshot),
    }
}

impl Krea2ControlHeavy {
    fn render(
        &self,
        device: &Device,
        req: &Krea2ControlRequest,
        control_image: &Image,
        context: Tensor,
        on_progress: &mut dyn FnMut(Progress),
    ) -> Result<Image> {
        let ctrl_nchw = control_image_to_nchw(control_image, req.width, req.height, device)?;
        let ctrl_latent = self.vae_encoder.encode(&ctrl_nchw)?;
        let scale = req.control_scale as f64;

        let (lat_h, lat_w) = (
            (req.height / SPATIAL_SCALE) as usize,
            (req.width / SPATIAL_SCALE) as usize,
        );
        let mut rng = StdRng::seed_from_u64(req.seed);
        let noise = candle_gen::seeded_normal_vec(&mut rng, LATENT_CHANNELS * lat_h * lat_w);
        let noise = Tensor::from_vec(noise, (1, LATENT_CHANNELS, lat_h, lat_w), &Device::Cpu)?
            .to_device(device)?;

        let sigmas = turbo_sigmas(req.steps);
        // The control branch injects its residual inside `forward_with_control`, so the running latent
        // the preview sees is the ordinary `[1, 16, H/8, W/8]` Krea trajectory — the same latent space
        // the reused Qwen fit was measured in, packed base tier or not (the branch dequants on load).
        let preview = crate::preview::hook(&req.preview);
        let latent = candle_gen::run_flow_sampler(
            None,
            TimestepConvention::Sigma,
            &sigmas,
            noise,
            req.seed,
            &req.cancel,
            on_progress,
            Some(&preview),
            |x, timestep| -> Result<Tensor> {
                let t = Tensor::from_vec(vec![timestep], (1,), device)?;
                let v = forward_with_control(
                    &self.dit,
                    &self.branch,
                    x,
                    &t,
                    &context,
                    &ctrl_latent,
                    scale,
                )?;
                Ok(v.to_dtype(DType::F32)?)
            },
        )?;

        on_progress(Progress::Decoding);
        let decoded = self
            .vae
            .decode_with(&latent, req.tile_vae_decode)?
            .to_dtype(DType::F32)?;
        to_image(&decoded)
    }
}

/// Validate the seed-independent request knobs before any tensor work. The empty-prompt guard mirrors
/// the registered txt2img `validate` (an empty prompt reaches the TE as a zero-length sequence and
/// surfaces as a deep tensor-shape error instead of a clean validation error).
fn validate_request(req: &Krea2ControlRequest) -> Result<()> {
    if req.prompt.trim().is_empty() {
        return Err(CandleError::Msg("krea control: prompt is required".into()));
    }
    if !req.width.is_multiple_of(SIZE_MULTIPLE) || !req.height.is_multiple_of(SIZE_MULTIPLE) {
        return Err(CandleError::Msg(format!(
            "krea control: width/height must be multiples of {SIZE_MULTIPLE} (got {}x{})",
            req.width, req.height
        )));
    }
    if req.steps == 0 {
        return Err(CandleError::Msg("krea control: steps must be >= 1".into()));
    }
    Ok(())
}

/// The rendered OpenPose skeleton (HWC RGB u8, already at `width`×`height`) → `[1, 3, H, W]` f32 in
/// `[-1, 1]`, channel-first — the exact normalization `candle_gen::train::dataset::load_image_tensor`
/// produces at train time, so the VAE-encoded control latent is identical to what the branch was
/// trained on. The worker driver renders the control map at the provider's output dims, so a size
/// mismatch is a wiring bug, not a resize case (the lib carries no image codec) — it errors loudly.
fn control_image_to_nchw(
    image: &Image,
    width: u32,
    height: u32,
    device: &Device,
) -> Result<Tensor> {
    let (iw, ih) = (image.width, image.height);
    if (iw, ih) != (width, height) {
        return Err(CandleError::Msg(format!(
            "krea control: control image {iw}x{ih} must match the render size {width}x{height}"
        )));
    }
    let (rw, rh) = (width as usize, height as usize);
    if image.pixels.len()
        != candle_gen::gen_core::imageops::checked_image_buffer_len(rw, rh, 3).unwrap_or(usize::MAX)
    {
        return Err(CandleError::Msg(format!(
            "krea control: control pixel buffer {} != {width}x{height}x3",
            image.pixels.len()
        )));
    }
    let mut data = vec![0f32; 3 * rh * rw];
    for y in 0..rh {
        for x in 0..rw {
            let base = (y * rw + x) * 3;
            for c in 0..3 {
                // HWC u8 [0,255] → channel-first [3, H, W]; [-1, 1].
                data[c * rh * rw + y * rw + x] = image.pixels[base + c] as f32 / 127.5 - 1.0;
            }
        }
    }
    Ok(Tensor::from_vec(data, (1, 3, rh, rw), &Device::Cpu)?.to_device(device)?)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn validation_complete_paths(
        offload_policy: OffloadPolicy,
    ) -> (tempfile::TempDir, Krea2ControlPaths) {
        let fixture = tempfile::tempdir().unwrap();
        gen_core_testkit::write_encoder_contract_fixture(
            &fixture.path().join("text_encoder"),
            crate::ENCODER_CONTRACT,
        )
        .unwrap();
        let paths = Krea2ControlPaths {
            root: fixture.path().to_path_buf(),
            convrot_dit: None,
            native_dit: None,
            control: PathBuf::from("/nonexistent/krea-control-residency-test-overlay.safetensors"),
            adapters: Vec::new(),
            branch_tier: None,
            chunk_attention: false,
            offload_policy,
        };
        (fixture, paths)
    }

    /// Weight-free proof that construction is lazy and the legacy path policy is not an authority.
    #[test]
    fn legacy_policy_does_not_eagerly_load_components() {
        let (_fixture, paths) = validation_complete_paths(OffloadPolicy::Sequential);
        let model = Krea2Control::load(&paths)
            .expect("construction must validate but not materialize the snapshot");
        assert!(model
            .residency
            .with_resident_parts(|_, _| ())
            .unwrap()
            .is_none());
    }

    /// The resident legacy value is equally lazy; neither load-time value can choose the request route.
    #[test]
    fn resident_legacy_policy_is_also_lazy() {
        let (_fixture, paths) = validation_complete_paths(OffloadPolicy::Resident);
        let model = Krea2Control::load(&paths)
            .expect("construction must validate but not materialize the snapshot");
        assert!(model
            .residency
            .with_resident_parts(|_, _| ())
            .unwrap()
            .is_none());
    }

    #[test]
    fn post_construction_control_mutation_fails_before_deferred_materialization() {
        let (fixture, mut paths) = validation_complete_paths(OffloadPolicy::Resident);
        let control = fixture.path().join("control.safetensors");
        std::fs::write(&control, b"before").unwrap();
        paths.control = control.clone();

        let selected = WeightsSource::Dir(fixture.path().join("text_encoder"));
        let validated = crate::ENCODER_CONTRACT
            .validate_source_against_base(&selected, fixture.path())
            .unwrap();
        let mut spec =
            candle_gen::gen_core::LoadSpec::new(WeightsSource::Dir(fixture.path().to_path_buf()))
                .with_control(WeightsSource::File(control.clone()));
        validated.prepare_load_spec(&mut spec).unwrap();
        let model = Krea2Control::load_with_spec(&paths, &spec)
            .expect("sparse fixture construction must retain deferred loaders");

        std::fs::write(&control, b"after!").unwrap();
        let request = Krea2ControlRequest {
            prompt: "a dancer".into(),
            width: SIZE_MULTIPLE,
            height: SIZE_MULTIPLE,
            steps: 1,
            ..Default::default()
        };
        let control_image = Image {
            width: SIZE_MULTIPLE,
            height: SIZE_MULTIPLE,
            pixels: vec![0; (SIZE_MULTIPLE * SIZE_MULTIPLE * 3) as usize],
        };
        let progress_called = std::cell::Cell::new(false);
        let error = model
            .generate(&request, &control_image, &mut |_| progress_called.set(true))
            .expect_err("mutated control must fail before the first deferred materializer")
            .to_string();
        assert!(
            error.contains("receipt changed") || error.contains("pinned weights"),
            "unexpected mutation error: {error}"
        );
        assert!(!progress_called.get(), "materialization emitted progress");
        assert!(
            model
                .residency
                .with_resident_parts(|_, _| ())
                .unwrap()
                .is_none(),
            "the warm deferred loader ran before the retained receipt rejected mutation"
        );
    }

    /// SC-16453: the immutable ConvRot file must survive the provider-path boundary and select the
    /// ConvRot loader rather than falling back to `root/transformer`.
    #[test]
    fn convrot_identity_selects_the_convrot_dit_loader() {
        let path = Path::new("/models/krea2_turbo_int8_convrot.safetensors");
        assert_eq!(
            control_dit_source(Some(path), None).unwrap(),
            ControlDitSource::ConvRot(path)
        );
        assert_eq!(
            control_dit_source(None, None).unwrap(),
            ControlDitSource::Snapshot
        );
    }

    #[test]
    fn native_identity_selects_native_and_conflicts_fail_closed() {
        let native = Path::new("/models/community-krea.safetensors");
        let convrot = Path::new("/models/krea-convrot.safetensors");
        assert_eq!(
            control_dit_source(None, Some(native)).unwrap(),
            ControlDitSource::Native(native)
        );
        assert!(control_dit_source(Some(convrot), Some(native)).is_err());

        // Lazy means the DiT, VAE, and overlay stay unread — the base snapshot's text encoder is
        // admitted against `ENCODER_CONTRACT` at construction, so the base needs the same fixture
        // every other construction test in this module uses. The overlay and adapter paths stay
        // nonexistent, which is what keeps the laziness claim honest.
        let (_base, base_paths) = validation_complete_paths(OffloadPolicy::Resident);
        let model = load_control_from_native_dit_file(
            native,
            &base_paths.root,
            "/nonexistent/control.safetensors",
            &[AdapterSpec::new(
                "/nonexistent/user.safetensors".into(),
                0.75,
                candle_gen::gen_core::AdapterKind::Lora,
            )],
        )
        .expect("native control construction stays lazy while retaining its complete route");
        assert!(model
            .residency
            .with_resident_parts(|_, _| ())
            .unwrap()
            .is_none());
    }

    /// ConvRot + adapters is admitted and retained for the lazy heavy-phase loader.
    #[test]
    fn convrot_retains_adapters_for_lazy_heavy_load() {
        // `validation_complete_paths` (not the old `missing_paths`): the loader now validates the
        // selected text encoder up front, so the root must carry a real encoder fixture.
        let (_fixture, mut paths) = validation_complete_paths(OffloadPolicy::Resident);
        paths.convrot_dit = Some(PathBuf::from(
            "/nonexistent/krea2_turbo_int8_convrot.safetensors",
        ));
        paths.adapters.push(AdapterSpec::new(
            PathBuf::from("/nonexistent/adapter.safetensors"),
            1.0,
            candle_gen::gen_core::AdapterKind::Lora,
        ));
        Krea2Control::load(&paths).expect("ConvRot plus an adapter must remain lazily loadable");
    }

    /// The request defaults match the Turbo control production knobs (1024², 8 CFG-free steps,
    /// control scale 0.6).
    #[test]
    fn request_defaults() {
        let r = Krea2ControlRequest::default();
        assert_eq!((r.width, r.height), (1024, 1024));
        assert_eq!(r.steps, TURBO_STEPS);
        assert_eq!(r.control_scale, DEFAULT_CONTROL_SCALE);
        // Untiled by default (sc-11744): the monolithic full-speed decode — the fit-ladder flips it on
        // only when the decode-phase peak won't fit.
        assert!(!r.tile_vae_decode);
        assert!(!r.cancel.is_cancelled());
    }

    /// The empty-prompt guard: an empty or whitespace-only prompt is a clean validation error; a real
    /// prompt with valid size passes.
    #[test]
    fn validate_request_rejects_empty_prompt() {
        let empty = Krea2ControlRequest::default();
        assert!(validate_request(&empty)
            .unwrap_err()
            .to_string()
            .contains("prompt is required"));

        let whitespace = Krea2ControlRequest {
            prompt: " \t\n".into(),
            ..Default::default()
        };
        assert!(validate_request(&whitespace)
            .unwrap_err()
            .to_string()
            .contains("prompt is required"));

        let ok = Krea2ControlRequest {
            prompt: "a dancer mid-leap".into(),
            ..Default::default()
        };
        assert!(validate_request(&ok).is_ok());
    }

    /// The size/steps guards fire.
    #[test]
    fn validate_request_keeps_size_and_steps_guards() {
        let odd = Krea2ControlRequest {
            prompt: "a dancer".into(),
            height: 1000,
            ..Default::default()
        };
        assert!(validate_request(&odd)
            .unwrap_err()
            .to_string()
            .contains("multiples"));

        let zero_steps = Krea2ControlRequest {
            prompt: "a dancer".into(),
            steps: 0,
            ..Default::default()
        };
        assert!(validate_request(&zero_steps)
            .unwrap_err()
            .to_string()
            .contains("steps"));
    }

    /// Real-weight two-process resident/sequential parity + peak harness for the bespoke control lane.
    /// Run once per mode in separate processes because candle's CUDA allocator retains its pool:
    ///
    /// ```text
    /// KREA_TURBO_DIR=<tier> KREA_CONTROL_CKPT=<overlay> KREA_CONTROL_POSE=<png> \
    /// KREA_OUT=resident.rgb cargo test -p candle-gen-krea --features cuda \
    ///   control_probed_generate_for_offload_ab -- --ignored --nocapture
    /// KREA_OFFLOAD_MODE=request-staged KREA_TURBO_DIR=<tier> KREA_CONTROL_CKPT=<overlay> \
    /// KREA_CONTROL_POSE=<png> KREA_OUT=sequential.rgb cargo test -p candle-gen-krea \
    ///   --features cuda control_probed_generate_for_offload_ab -- --ignored --nocapture
    /// ```
    ///
    /// Compare the raw pixel files byte-for-byte and use the printed rendered-device `overall-peak`
    /// deltas as the resident/sequential calibration. `KREA_CONTROL_BRANCH_QUANT=q8|q4` selects the
    /// branch tier; omitted means bf16. `KREA_CONTROL_CONVROT_DIT=<file>` replaces the standard DiT,
    /// while `KREA_TILE_VAE=1` and `KREA_CHUNK_ATTN=1` isolate the two fit-ladder savings.
    /// `KREA_AB_RES` defaults to 768 and `KREA_AB_STEPS` defaults to eight. One step is sufficient for
    /// packed-tier peak calibration because the same denoise working set is reused at every step, but
    /// both processes in a parity pair must use the same value.
    #[cfg(feature = "cuda")]
    #[test]
    #[ignore]
    fn control_probed_generate_for_offload_ab() {
        let root = PathBuf::from(std::env::var("KREA_TURBO_DIR").expect("set KREA_TURBO_DIR"));
        let control =
            PathBuf::from(std::env::var("KREA_CONTROL_CKPT").expect("set KREA_CONTROL_CKPT"));
        let pose_path =
            std::env::var("KREA_CONTROL_POSE").expect("set KREA_CONTROL_POSE to a pose PNG");
        let out = std::env::var("KREA_OUT").expect("set KREA_OUT to the raw pixel-dump path");
        let res = std::env::var("KREA_AB_RES")
            .ok()
            .and_then(|value| value.trim().parse().ok())
            .unwrap_or(768u32);
        let steps = std::env::var("KREA_AB_STEPS")
            .ok()
            .and_then(|value| value.trim().parse().ok())
            .unwrap_or(TURBO_STEPS);
        let branch_tier = match std::env::var("KREA_CONTROL_BRANCH_QUANT")
            .unwrap_or_default()
            .trim()
            .to_ascii_lowercase()
            .as_str()
        {
            "" | "bf16" | "none" => None,
            "q8" => Some(Quant::Q8),
            "q4" => Some(Quant::Q4),
            other => panic!("KREA_CONTROL_BRANCH_QUANT must be bf16|q8|q4, got {other}"),
        };
        let stage_residency =
            std::env::var("KREA_OFFLOAD_MODE").is_ok_and(|mode| mode == "request-staged");
        let convrot_dit = std::env::var("KREA_CONTROL_CONVROT_DIT")
            .ok()
            .filter(|value| !value.trim().is_empty())
            .map(PathBuf::from);
        let tile_vae_decode = std::env::var("KREA_TILE_VAE")
            .is_ok_and(|value| matches!(value.trim(), "1" | "true" | "yes"));
        let chunk_attention = std::env::var("KREA_CHUNK_ATTN")
            .is_ok_and(|value| matches!(value.trim(), "1" | "true" | "yes"));
        let mut tier_spec = candle_gen::gen_core::LoadSpec::new(
            candle_gen::gen_core::WeightsSource::Dir(root.clone()),
        );
        if let Some(path) = convrot_dit.as_ref() {
            tier_spec.text_encoder = Some(candle_gen::gen_core::WeightsSource::File(path.clone()));
        }
        let base_quant = crate::actual_quant_tier(&tier_spec, "krea_2_turbo_control")
            .expect("resolve measured Krea control base tier");

        let pose = image::open(pose_path).expect("decode pose PNG").to_rgb8();
        let pose = image::imageops::resize(&pose, res, res, image::imageops::FilterType::Lanczos3);
        let pose = Image {
            width: res,
            height: res,
            pixels: pose.into_raw(),
        };
        let paths = Krea2ControlPaths {
            root,
            convrot_dit,
            native_dit: None,
            control,
            adapters: Vec::new(),
            branch_tier,
            chunk_attention,
            offload_policy: OffloadPolicy::Resident,
        };
        let request = Krea2ControlRequest {
            prompt: "a dancer in a colorful studio, cinematic lighting".into(),
            width: res,
            height: res,
            steps,
            control_scale: DEFAULT_CONTROL_SCALE,
            seed: 42,
            tile_vae_decode,
            stage_residency,
            ..Default::default()
        };

        assert!(
            candle_gen::testkit::reset_cuda_mempool_high_water(0),
            "reset CUDA live-allocation high-water"
        );
        let mut probe = candle_gen::testkit::VramProbe::start_rendered();
        let load_phase = probe.phase();
        let model = Krea2Control::load(&paths).expect("load Krea control provider");
        probe.end_load(load_phase);
        let gen_phase = probe.phase();
        let image = model
            .generate(&request, &pose, &mut |_| {})
            .expect("generate Krea control image");
        probe.end_gen(gen_phase);
        let report = probe.report().assert_trustworthy(1.0);
        let live_peak_bytes = candle_gen::testkit::cuda_mempool_used_high_bytes(0)
            .expect("read CUDA live-allocation high-water");
        assert!(
            live_peak_bytes > 0,
            "CUDA live-allocation peak must be positive"
        );
        std::fs::write(&out, &image.pixels).expect("write raw pixels");

        let strategy = if chunk_attention {
            candle_gen::gen_core::MemoryStrategy::BoundedAttention
        } else if tile_vae_decode {
            candle_gen::gen_core::MemoryStrategy::BoundedDecode
        } else if stage_residency {
            candle_gen::gen_core::MemoryStrategy::StagedResidency
        } else {
            candle_gen::gen_core::MemoryStrategy::Resident
        };
        let memory_contract = crate::build_krea_control_memory_strategy_contract(&tier_spec)
            .expect("build Krea control memory contract");
        let engaged_composition = memory_contract.engaged_composition(strategy);
        assert_eq!(
            stage_residency,
            engaged_composition.contains(&candle_gen::gen_core::MemoryStrategy::StagedResidency),
            "Krea control probe flags must execute the contract's staged-residency composition"
        );
        assert_eq!(
            tile_vae_decode,
            engaged_composition.contains(&candle_gen::gen_core::MemoryStrategy::BoundedDecode),
            "Krea control probe flags must execute the contract's bounded-decode composition"
        );
        assert_eq!(
            chunk_attention,
            engaged_composition.contains(&candle_gen::gen_core::MemoryStrategy::BoundedAttention),
            "Krea control probe flags must execute the contract's bounded-attention composition"
        );
        let parameters = candle_gen::gen_core::MemoryStrategyParameters {
            decode_tile_edge: engaged_composition
                .contains(&candle_gen::gen_core::MemoryStrategy::BoundedDecode)
                .then_some(512),
            decode_overlap: engaged_composition
                .contains(&candle_gen::gen_core::MemoryStrategy::BoundedDecode)
                .then_some(128),
            attention_chunk_size: engaged_composition
                .contains(&candle_gen::gen_core::MemoryStrategy::BoundedAttention)
                .then_some(KREA_ATTN_CHUNK_BUDGET as u32),
            ..Default::default()
        };
        let overlay = match branch_tier {
            Some(Quant::Q8) => "pose-control-q8",
            Some(Quant::Q4) => "pose-control-q4",
            Some(Quant::Nvfp4) => "pose-control-nvfp4",
            None => "pose-control-bf16",
        };
        let observed_calibration = memory_contract
            .calibration
            .clone()
            .expect("Krea control contract must export its calibration identity");
        eprintln!(
            "{}",
            candle_gen::testkit::memory_evidence_v1_line(
                candle_gen::testkit::MemoryEvidenceProbe {
                    resolved_route: "krea_2_turbo_control",
                    declared_calibration: candle_gen::testkit::expected_memory_calibration(
                        observed_calibration.load_shape,
                    ),
                    observed_calibration: observed_calibration.clone(),
                    tier: candle_gen::gen_core::MemoryNumericTier {
                        precision: candle_gen::gen_core::Precision::Bf16,
                        quant: base_quant,
                        component_precision_floors: &[],
                    },
                    load_shape: observed_calibration.load_shape,
                    mode: candle_gen::gen_core::MemoryMode::ImageToImage,
                    overlay: Some(overlay.to_owned()),
                    geometry: candle_gen::gen_core::MemoryGeometry {
                        width: request.width,
                        height: request.height,
                        batch: 1,
                        frames: 1,
                        reference_count: 1,
                    },
                    strategy,
                    engaged_composition,
                    parameters,
                    observed_peak_bytes: live_peak_bytes,
                    harness_version: "candle-krea-control-residency-v1",
                    output_bytes: &image.pixels,
                }
            )
        );
        eprintln!(
            "MEMORY_EVIDENCE_DIAGNOSTIC id=krea_2_turbo_control gpu={} {}x{} steps={} branch_tier={branch_tier:?} convrot={} tile_vae_decode={tile_vae_decode} chunk_attention={chunk_attention} | {report} | bytes={} out={out}",
            candle_gen::testkit::probe_gpu(),
            image.width,
            image.height,
            request.steps,
            paths.convrot_dit.is_some(),
            image.pixels.len(),
        );
    }
}
