//! The Qwen-Image-**Edit** provider (sc-5487, epic 5480) — the candle (Windows/CUDA) sibling of
//! `mlx-gen-qwen-image`'s `QwenImageEdit`. Reference-conditioned image editing on `qwen_image_edit`:
//!
//! 1. **VL conditioning** — the reference + edit prompt go through the [`QwenVisionLanguageEncoder`]
//!    (vision tower + LM splice, Slice A) to `[1, S−64, 3584]` prompt embeds (the vision tower runs
//!    once, reused across the positive/negative prompts).
//! 2. **Dual-latent** — each reference is VAE-encoded + packed and concatenated **after** the noise
//!    over the sequence axis; the transformer's 3-axis RoPE spans `[noise] + references`
//!    ([`QwenTransformer::forward_edit`]). `zero_cond_t` (Edit-2511) modulates the conditioning
//!    tokens as clean; the original Edit / 2509 runs a single timestep (auto-detected from the
//!    transformer config).
//! 3. flow-match Euler denoise (true CFG with norm-rescale) → slice the noise prefix → VAE decode.
//!
//! A bespoke provider driven **directly** by the worker (like [`crate::control_fun::QwenFunControl`]
//! and `candle_gen_sdxl::SdxlEdit`) — the registered `qwen_image` descriptor stays txt2img-only.
//!
//! NB: candle's CUDA attention indexes scores with i32, so a joint sequence whose scores tensor
//! exceeds `i32::MAX` elements (~2.1B) would silently corrupt — the shared `JointAttention` guards
//! this by chunking over query rows once the scores exceed `ATTN_SCORES_BUDGET` (sc-6217), and the
//! `edit_validate` high-res run confirms a coherent 1536² edit through that chunked path.

use std::collections::HashMap;
use std::path::{Path, PathBuf};

use candle_gen::candle_core::{DType, Device, Tensor};
use candle_gen::candle_nn::VarBuilder;
use candle_gen::gen_core::runtime::CancelFlag;
use candle_gen::gen_core::tokenizer::TextTokenizer;
use candle_gen::gen_core::{AdapterSpec, Image, OffloadPolicy, Progress};
use candle_gen::{CandleError, Result};

use crate::config::{TextEncoderConfig, TransformerConfig, NEGATIVE_FALLBACK};
use crate::image_processor::{ImageInput, QwenImageProcessor};
use crate::pipeline;
use crate::transformer::QwenTransformer;
use crate::vae::{QwenVae, QwenVaeEncoder};
use crate::vision_language::{load_vision_language_encoder, QwenVisionLanguageEncoder};
use crate::vl_tokenizer::{
    condition_resize_dims, encode_reference_latents, preprocess_edit_image, tokenize_edit_text,
};

/// The transformer runs bf16 (native dtype); the VL encoder + VAE run f32.
const DIT_DTYPE: DType = DType::BF16;
const ENC_DTYPE: DType = DType::F32;

/// The conditioning produced by [`QwenEdit::encode_conditioning`] and consumed by
/// [`QwenEdit::denoise_and_decode`]: `(pos_embeds, neg_embeds, static_reference_latents, cond_grids)`.
/// The pieces that must survive the VL-encoder drop in the sequential path (all small — no model weights).
type EditConditioning = (Tensor, Option<Tensor>, Tensor, Vec<(usize, usize)>);

/// Paths to the Qwen-Image-Edit checkpoint.
pub struct QwenEditPaths {
    /// The `Qwen/Qwen-Image-Edit` diffusers snapshot dir (`text_encoder/` [LM + vision], `transformer/`,
    /// `vae/`, `tokenizer/`). The validated reference is `-2511`.
    pub root: PathBuf,
    /// LoRA/LoKr adapters folded into the MMDiT at load (sc-6220) — e.g. the Qwen-Image-Edit-2511
    /// Lightning distill, stacked ahead of any user adapters. **Empty** = the production (non-distilled)
    /// edit path: the transformer loads via the mmap fast path, byte-identical to before.
    pub adapters: Vec<AdapterSpec>,
    /// Component-residency policy (epic 10765 Phase 1c follow-up, sc-10968). `Sequential` routes
    /// [`QwenEdit::generate`] through the phased load→encode→DROP→load path (load the VL encoder + VAE
    /// encoder, VL-encode the prompt + VAE-encode the references, then DROP the VL encoder before the DiT
    /// loads), capping peak VRAM at the cost of a per-request reload; `Resident` (default) loads all four
    /// components once at [`QwenEdit::load`] and keeps them, like the pre-sc-10968 behavior. The worker's
    /// edit fit-gate sets this when it predicts the resident sum won't fit but the sequential working set
    /// will (mirrors the txt2img `LoadSpec::offload_policy`, sc-10867).
    pub offload_policy: OffloadPolicy,
}

/// One Qwen-Image-Edit generation request.
#[derive(Clone)]
pub struct QwenEditRequest {
    pub prompt: String,
    pub negative: String,
    pub width: u32,
    pub height: u32,
    pub steps: usize,
    /// True-CFG guidance scale. Ignored (CFG forced off) on the [`lightning`](Self::lightning) path.
    pub guidance: f32,
    pub seed: u64,
    /// The Qwen-Image-Edit-2511-Lightning few-step distill path (sc-6220): use the static-shift
    /// [`pipeline::lightning_sigmas`] schedule and run **CFG-off** (a single forward per step, no
    /// negative branch — the distill LoRA is CFG-distilled). The matching distill LoRA must be supplied
    /// via [`QwenEditPaths::adapters`]. `false` = the production multi-step true-CFG path.
    pub lightning: bool,
    pub cancel: CancelFlag,
}

impl Default for QwenEditRequest {
    fn default() -> Self {
        Self {
            prompt: String::new(),
            negative: String::new(),
            width: 1024,
            height: 1024,
            steps: 30,
            guidance: 4.0,
            seed: 0,
            lightning: false,
            cancel: CancelFlag::default(),
        }
    }
}

/// mmap a [`VarBuilder`] over every `.safetensors` in `root/sub` at `dtype`.
fn component_vb(
    root: &Path,
    sub: &str,
    dtype: DType,
    device: &Device,
) -> Result<VarBuilder<'static>> {
    candle_gen::component_vb(root, sub, dtype, device, "qwen edit")
}

/// Load every `.safetensors` in `root/transformer` into one CPU tensor map (native dtype). The eager
/// load (vs the mmap [`component_vb`] fast path) is what lets the adapter deltas fold into the dense
/// weights before the MMDiT is built (sc-6220).
fn load_transformer_tensors(root: &Path) -> Result<HashMap<String, Tensor>> {
    let dir = root.join("transformer");
    if !dir.is_dir() {
        return Err(CandleError::Msg(format!(
            "qwen edit: snapshot is missing the transformer/ dir (at {})",
            root.display()
        )));
    }
    // Shared sorted-`.safetensors` resolver (sc-8999 / F-019); this path loads into a CPU map for
    // adapter merging (not the mmap fast path), so it keeps its own loop.
    let files = candle_gen::sorted_safetensors(&dir, "qwen edit")?;
    let mut map = HashMap::new();
    for f in &files {
        let part = candle_gen::candle_core::safetensors::load(f, &Device::Cpu)?;
        map.extend(part);
    }
    Ok(map)
}

/// Build the MMDiT, applying LoRA/LoKr `adapters` by the route the base tier + adapter type allow
/// (sc-6220, sc-11091, sc-11684):
///
/// * **No adapters** — the mmap fast path (byte-identical to before), serving a dense *or* packed base.
/// * **Additive residual** ([`crate::adapters::install_additive`]) — the DEFAULT whenever the adapters
///   have a deferred form (plain LoRA / structured LoKr), on a **packed q4/q8 OR dense bf16** base. Build
///   the DiT via the mmap fast path (base kept as-is — q4/q8 codes or dense weights, never
///   dequantized/folded) then push each adapter as `y = base(x) + Σ scale·((x·A)·B)`. So the
///   Qwen-Image-Edit-2511-Lightning distill (all 720 attn+MLP Linears) applies at the base's footprint and
///   the adapted DiT stays streamable under sequential residency ([`QwenEdit::load_transformer_seq`]) —
///   instead of the eager fold's whole-DiT CPU load. Costs ~1 ULP vs the fold (`W·x + δ·x ≠ (W+δ)·x`),
///   accepted uniformly across tiers (sc-11684).
/// * **Dense fold fallback** ([`crate::adapters::merge_adapters`], `W += δ` in f32) — ONLY for adapter
///   types with no deferred form (**LoHa**'s Hadamard, **untagged third-party LyCORIS LoKr**) on a dense
///   base. Bit-exact but not streamable; these types are rare and dense-only (on a packed base
///   `install_additive` errors — there is no dense `W` to fold into).
///
/// A non-empty `adapters` slice that matches no MMDiT module errors on either route (it never renders an
/// unadapted image silently).
fn load_transformer(
    root: &Path,
    adapters: &[AdapterSpec],
    dtype: DType,
    device: &Device,
) -> Result<QwenTransformer> {
    let cfg = TransformerConfig::qwen_image();
    let dit_dir = root.join("transformer");
    // The DiT packed-detects each `Linear`: an MLX-packed edit tier (`SceneWorks/qwen-image-edit-2511
    // -mlx` q4/q8) loads straight from the packed parts at the `group_size` read from
    // `transformer/config.json` (64); a dense Edit snapshot loads unchanged (the group size is inert on
    // the dense path). See `crate::transformer_group_size`.
    let gs = crate::transformer_group_size(&dit_dir);
    if adapters.is_empty() {
        return Ok(QwenTransformer::new_gs(
            &cfg,
            component_vb(root, "transformer", dtype, device)?,
            gs,
        )?);
    }
    // Additive residual for anything with a deferred form — REQUIRED on a packed base (no dense `W` to
    // fold), and now the default on a DENSE base too (sc-11684) so the adapted DiT loads at the base's
    // footprint and streams under sequential residency instead of the eager whole-DiT fold. LoHa /
    // untagged-LyCORIS-LoKr on a dense base have no deferred form → fall through to the fold below.
    if crate::transformer_is_packed(&dit_dir)
        || crate::adapters::adapters_additive_capable(adapters)?
    {
        // Base kept as-is (packed q4/q8 codes or dense weights) via the mmap fast path, then push the
        // LoRA/LoKr as forward-time residuals — never folding a delta into the base.
        let mut dit =
            QwenTransformer::new_gs(&cfg, component_vb(root, "transformer", dtype, device)?, gs)?;
        // Discard the report — like the fold path below, library code stays quiet on stderr; a
        // non-matching adapter surface already errors inside `install_additive` (sc-9035 / F-051).
        let _ = crate::adapters::install_additive(&mut dit, adapters)?;
        return Ok(dit);
    }
    // Dense fold FALLBACK (sc-11684): LoHa / untagged-LyCORIS-LoKr on a dense base — no deferred additive
    // form, so fold the delta into the weight before the MMDiT is built (each merged tensor cast to
    // `dtype` + moved to `device` as the VarBuilder serves it, so peak GPU is unchanged vs the mmap path).
    // Bit-exact but not streamable; these adapter types are rare and dense-only.
    let mut tensors = load_transformer_tensors(root)?;
    crate::adapters::merge_adapters(&mut tensors, adapters)?;
    let vb = VarBuilder::from_tensors(tensors, dtype, device);
    Ok(QwenTransformer::new_gs(&cfg, vb, gs)?)
}

/// `transformer/config.json` `zero_cond_t` (Edit-2511 = true; the original Edit / 2509 omit it).
///
/// A genuinely-absent `transformer/config.json` (the original Edit / 2509 snapshots don't gate on it)
/// or an absent `zero_cond_t` key defaults to `false`. But a *present-but-corrupt* config — I/O error,
/// malformed JSON, or a `zero_cond_t` of the wrong type — errors loudly rather than silently switching
/// an Edit-2511 render to the 2509 single-timestep modulation on a damaged snapshot (sc-9010 / F-073).
fn read_zero_cond_t(root: &Path) -> Result<bool> {
    let path = root.join("transformer/config.json");
    let text = match std::fs::read_to_string(&path) {
        Ok(t) => t,
        // Absent config ⇒ documented default (2509 / original Edit).
        Err(ref e) if e.kind() == std::io::ErrorKind::NotFound => return Ok(false),
        Err(e) => {
            return Err(CandleError::Msg(format!(
                "qwen edit: read {}: {e}",
                path.display()
            )))
        }
    };
    let v: serde_json::Value = serde_json::from_str(&text).map_err(|e| {
        CandleError::Msg(format!(
            "qwen edit: parse {} (corrupt snapshot?): {e}",
            path.display()
        ))
    })?;
    match v.get("zero_cond_t") {
        // Key absent ⇒ documented default.
        None | Some(serde_json::Value::Null) => Ok(false),
        Some(b) => b.as_bool().ok_or_else(|| {
            CandleError::Msg(format!(
                "qwen edit: `zero_cond_t` in {} must be a bool, got {b}",
                path.display()
            ))
        }),
    }
}

/// Locate the assembled HF `tokenizer.json` (sc-6294). The original `Qwen-Image-Edit` ships it under
/// `tokenizer/`, but `Qwen-Image-Edit-2511` ships the assembled file only inside the Qwen2.5-VL
/// processor bundle (`processor/tokenizer.json`) — the `tokenizer/` dir there carries just the BPE
/// source (`merges.txt`/`vocab.json`). The two locations are byte-identical (same SHA256), so prefer
/// `tokenizer/`, then fall back to `processor/`, so a whole-repo -2511 download loads without a
/// hand-staged tokenizer.json.
fn tokenizer_json_path(root: &Path) -> Result<PathBuf> {
    for rel in ["tokenizer/tokenizer.json", "processor/tokenizer.json"] {
        let p = root.join(rel);
        if p.is_file() {
            return Ok(p);
        }
    }
    Err(CandleError::Msg(format!(
        "qwen edit: no tokenizer.json under tokenizer/ or processor/ (at {})",
        root.display()
    )))
}

/// The loaded Qwen-Image-Edit model. [`candle_gen::Residency`] exclusively owns either the warm
/// component pair or the two deferred phase loaders. The image processor, tokenizer, and
/// `zero_cond_t` flag are cheap and always resident.
pub struct QwenEdit {
    device: Device,
    residency: candle_gen::Residency<EditText, EditHeavy>,
    processor: QwenImageProcessor,
    tokenizer: TextTokenizer,
    zero_cond_t: bool,
}

struct EditText {
    vl_encoder: QwenVisionLanguageEncoder,
    vae_encoder: QwenVaeEncoder,
}

struct EditHeavy {
    transformer: QwenTransformer,
    vae: QwenVae,
}

impl QwenEdit {
    /// Load the Qwen-Image-Edit components from a snapshot dir. Under the default `Resident` policy all
    /// four heavy components load now; under `Sequential` (sc-10968) they are deferred to the per-phase
    /// residency loaders, and only the cheap tokenizer / processor / `zero_cond_t` load here.
    pub fn load(paths: &QwenEditPaths) -> Result<Self> {
        let device = candle_gen::default_device()?;
        let root = paths.root.clone();
        let te_cfg = TextEncoderConfig::qwen_image();

        // Shared tokenizer policy (F-134 / sc-11190) with the edit lane's own `-2511` processor-bundle
        // path resolution — one `tokenizer_config()` home keeps edit's caption tokenization identical to
        // the txt2img lane's.
        let tokenizer = TextTokenizer::from_file(
            tokenizer_json_path(&root)?,
            crate::control_common::tokenizer_config(&te_cfg),
        )
        .map_err(|e| CandleError::Msg(format!("qwen edit: load tokenizer: {e}")))?;

        let policy = candle_gen::effective_offload_policy(paths.offload_policy);
        let resident_root = root.clone();
        let resident_device = device.clone();
        let resident_adapters = paths.adapters.clone();
        let text_root = root.clone();
        let text_device = device.clone();
        let heavy_root = root.clone();
        let heavy_device = device.clone();
        let heavy_adapters = paths.adapters.clone();
        let residency = candle_gen::Residency::from_policy_with_resident(
            policy,
            move || {
                Ok((
                    EditText {
                        vl_encoder: load_vision_language_encoder(&resident_root, &resident_device)?,
                        vae_encoder: QwenVaeEncoder::new(component_vb(
                            &resident_root,
                            "vae",
                            ENC_DTYPE,
                            &resident_device,
                        )?)?,
                    },
                    EditHeavy {
                        transformer: load_transformer(
                            &resident_root,
                            &resident_adapters,
                            DIT_DTYPE,
                            &resident_device,
                        )?,
                        vae: QwenVae::new(component_vb(
                            &resident_root,
                            "vae",
                            ENC_DTYPE,
                            &resident_device,
                        )?)?,
                    },
                ))
            },
            move || {
                Ok(EditText {
                    vl_encoder: load_vision_language_encoder(&text_root, &text_device)?,
                    vae_encoder: QwenVaeEncoder::new(component_vb(
                        &text_root,
                        "vae",
                        ENC_DTYPE,
                        &text_device,
                    )?)?,
                })
            },
            move |_| {
                Ok(EditHeavy {
                    transformer: load_transformer(
                        &heavy_root,
                        &heavy_adapters,
                        DIT_DTYPE,
                        &heavy_device,
                    )?,
                    vae: QwenVae::new(component_vb(&heavy_root, "vae", ENC_DTYPE, &heavy_device)?)?,
                })
            },
        )?;

        Ok(Self {
            zero_cond_t: read_zero_cond_t(&root)?,
            device,
            residency,
            processor: QwenImageProcessor::default(),
            tokenizer,
        })
    }

    /// VL-encode one prompt against the precomputed `vision` embeds → `[1, S−64, 3584]` at the DiT
    /// dtype. `n_image_tokens` is the shared `<|image_pad|>` run length (from the image preprocess).
    /// Takes `vl_encoder` by ref so the resident and sequential paths encode identically.
    fn encode_prompt(
        &self,
        vl_encoder: &QwenVisionLanguageEncoder,
        prompt: &str,
        n_image_tokens: usize,
        vision: &Tensor,
    ) -> Result<Tensor> {
        let ids = tokenize_edit_text(&self.tokenizer, prompt, n_image_tokens)?;
        let len = ids.len();
        let input_ids = Tensor::from_vec(ids, (1, len), &self.device)?;
        let embeds = vl_encoder.encode_with_vision(&input_ids, vision)?;
        Ok(embeds.to_dtype(DIT_DTYPE)?)
    }

    /// The shared conditioning head (sc-10968): VL-encode the vision tower + prompt(s) and VAE-encode the
    /// reference dual-latent, borrowing the VL + VAE encoders so the resident and sequential paths produce
    /// byte-identical `(pos, neg, static_latents, cond_grids)`. The **first** reference drives the VL
    /// prompt embeds, **all** are VAE-encoded into the dual-latent sequence, and the **last** sets the
    /// condition resolution — the exact semantics of the pre-sc-10968 monolithic `generate`.
    fn encode_conditioning(
        &self,
        vl_encoder: &QwenVisionLanguageEncoder,
        vae_encoder: &QwenVaeEncoder,
        req: &QwenEditRequest,
        references: &[Image],
    ) -> Result<EditConditioning> {
        let first = references.first().ok_or_else(|| {
            CandleError::Msg("qwen edit: at least one reference image is required".into())
        })?;
        let last = references.last().expect("non-empty checked");

        // VL conditioning: preprocess the first reference once (image-only), run the vision tower once,
        // then encode the positive (+ negative for CFG) prompts reusing the vision embeds.
        let edit_img = preprocess_edit_image(&self.processor, image_input(first), &self.device)?;
        let vision = vl_encoder.encode_vision(&edit_img.pixel_values, &[edit_img.grid])?;
        let pos = self.encode_prompt(vl_encoder, &req.prompt, edit_img.n_image_tokens, &vision)?;
        // CFG-off on the lightning path: the distill LoRA is CFG-distilled, so a single forward per
        // step (no negative branch) — matching the MLX lightning recipe (sc-6220).
        let neg = if req.guidance > 1.0 && !req.lightning {
            let n = if req.negative.trim().is_empty() {
                NEGATIVE_FALLBACK
            } else {
                req.negative.as_str()
            };
            Some(self.encode_prompt(vl_encoder, n, edit_img.n_image_tokens, &vision)?)
        } else {
            None
        };

        // Dual-latent references (static across steps): VAE-encode each reference at the VL condition
        // resolution (from the last reference's aspect), pack, and concatenate over the sequence axis.
        let (vl_w, vl_h) = condition_resize_dims(last.width as usize, last.height as usize);
        let mut packed = Vec::with_capacity(references.len());
        let mut cond_grids = Vec::with_capacity(references.len());
        for im in references {
            let (latents, grid) = encode_reference_latents(
                vae_encoder,
                image_input(im),
                vl_w as u32,
                vl_h as u32,
                &self.device,
            )?;
            packed.push(latents.to_dtype(DIT_DTYPE)?);
            cond_grids.push(grid);
        }
        let static_latents = if packed.len() == 1 {
            packed.pop().expect("len checked")
        } else {
            Tensor::cat(&packed.iter().collect::<Vec<_>>(), 1)?
        };
        Ok((pos, neg, static_latents, cond_grids))
    }

    /// The shared denoise + decode tail (sc-10968): given already-encoded `(pos, neg, static_latents,
    /// cond_grids)` and the just-resident DiT + VAE decoder, run the flow sampler (dual-latent concat +
    /// true-CFG blend inside the `predict` closure) and decode. Borrows the DiT / VAE so BOTH the resident
    /// and sequential paths run this identical loop — only the load/free schedule differs, not this code.
    ///
    /// Lightning uses the static-shift schedule (resolution-independent); production uses the dynamic-μ
    /// schedule (sc-6220). Routed through the unified curated sampler/scheduler framework (epic 7114 P4,
    /// sc-7123): the bespoke edit provider has no `req.sampler`/`req.scheduler` surface yet, so both stay
    /// `None` (the N1 default: `euler` over the native schedule). The model is fed the raw sigma (`Sigma`
    /// convention); Qwen-Image-Edit is **true CFG**, and the dual-latent concat/slice (concatenate the
    /// updating noise with the static reference latents over the sequence axis, then slice the noise
    /// prefix post-forward) lives — with the pos/neg/blend — inside the `predict` closure.
    #[allow(clippy::too_many_arguments)]
    fn denoise_and_decode(
        &self,
        transformer: &QwenTransformer,
        vae: &QwenVae,
        req: &QwenEditRequest,
        pos: &Tensor,
        neg: Option<&Tensor>,
        static_latents: &Tensor,
        cond_grids: &[(usize, usize)],
        on_progress: &mut dyn FnMut(Progress),
    ) -> Result<Image> {
        let (lat_h, lat_w) = pipeline::latent_dims(req.width, req.height);
        let noise_seq = lat_h * lat_w;

        let (native, mu) = if req.lightning {
            (
                pipeline::lightning_sigmas(req.steps),
                pipeline::lightning_mu(),
            )
        } else {
            (
                pipeline::qwen_sigmas(req.steps, req.width, req.height),
                pipeline::qwen_mu(req.width, req.height),
            )
        };
        let sigmas = candle_gen::resolve_flow_schedule(None, mu, req.steps, &native);
        let latents = pipeline::create_noise(req.seed, req.width, req.height, &self.device)?
            .to_dtype(DIT_DTYPE)?;

        let latents = candle_gen::run_flow_sampler(
            None,
            candle_gen::gen_core::sampling::TimestepConvention::Sigma,
            &sigmas,
            latents,
            req.seed,
            &req.cancel,
            on_progress,
            |latents, sigma| -> Result<Tensor> {
                // Concatenate the (updating) noise with the (static) reference latents over the sequence.
                let joint = Tensor::cat(&[latents, static_latents], 1)?;
                let pos_v = transformer
                    .forward_edit(
                        &joint,
                        pos,
                        sigma,
                        lat_h,
                        lat_w,
                        cond_grids,
                        self.zero_cond_t,
                    )?
                    .narrow(1, 0, noise_seq)?;
                match neg {
                    Some(neg) => {
                        let neg_v = transformer
                            .forward_edit(
                                &joint,
                                neg,
                                sigma,
                                lat_h,
                                lat_w,
                                cond_grids,
                                self.zero_cond_t,
                            )?
                            .narrow(1, 0, noise_seq)?;
                        Ok(pipeline::compute_guided_noise(
                            &pos_v,
                            &neg_v,
                            req.guidance,
                        )?)
                    }
                    None => Ok(pos_v),
                }
            },
        )?;

        on_progress(Progress::Decoding);
        let lat = pipeline::unpack_latents(&latents, req.width, req.height)?;
        let decoded = vae.decode(&lat)?;
        crate::control_common::to_image(&decoded)
    }

    /// Reference-conditioned edit. `references` is the (validated non-empty) reference image set: the
    /// **first** drives the VL prompt embeds, **all** are VAE-encoded into the dual-latent sequence,
    /// and the **last** sets the condition resolution (the fork's `_compute_dimensions`). The residency
    /// owner supplies either warm components or phased loads to these same encode/render bodies.
    pub fn generate(
        &self,
        req: &QwenEditRequest,
        references: &[Image],
        on_progress: &mut dyn FnMut(Progress),
    ) -> Result<Image> {
        self.residency.run(
            &req.cancel,
            false,
            on_progress,
            |text| self.encode_conditioning(&text.vl_encoder, &text.vae_encoder, req, references),
            |heavy, (pos, neg, static_latents, cond_grids), on_progress| {
                self.denoise_and_decode(
                    &heavy.transformer,
                    &heavy.vae,
                    req,
                    &pos,
                    neg.as_ref(),
                    &static_latents,
                    &cond_grids,
                    on_progress,
                )
            },
        )
    }
}

/// Borrow an [`Image`] as an [`ImageInput`] (RGB uint8 HWC).
fn image_input(im: &Image) -> ImageInput<'_> {
    ImageInput {
        data: &im.pixels,
        height: im.height as usize,
        width: im.width as usize,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn request_defaults() {
        let r = QwenEditRequest::default();
        assert_eq!((r.width, r.height), (1024, 1024));
        assert_eq!(r.steps, 30);
        assert!(!r.cancel.is_cancelled());
    }

    fn zero_cond_t_tmp(name: &str) -> PathBuf {
        let tmp = std::env::temp_dir().join(format!(
            "qwen_edit_zct_{name}_{}_{:?}",
            std::process::id(),
            std::thread::current().id()
        ));
        let _ = std::fs::remove_dir_all(&tmp);
        std::fs::create_dir_all(tmp.join("transformer")).unwrap();
        tmp
    }

    #[test]
    fn zero_cond_t_defaults_false_when_config_absent() {
        // A nonexistent config (dir/file) → false, the original Qwen-Image-Edit / 2509 path.
        assert!(!read_zero_cond_t(Path::new("/nonexistent")).unwrap());
    }

    #[test]
    fn zero_cond_t_defaults_false_when_key_absent() {
        // Config present but the key genuinely absent (a valid 2509 config.json) → documented default.
        let tmp = zero_cond_t_tmp("keyabsent");
        std::fs::write(
            tmp.join("transformer/config.json"),
            br#"{"num_layers": 60}"#,
        )
        .unwrap();
        assert!(!read_zero_cond_t(&tmp).unwrap());
        let _ = std::fs::remove_dir_all(&tmp);
    }

    #[test]
    fn zero_cond_t_reads_present_value() {
        // Edit-2511 config with the key set true → true.
        let tmp = zero_cond_t_tmp("present");
        std::fs::write(
            tmp.join("transformer/config.json"),
            br#"{"zero_cond_t": true}"#,
        )
        .unwrap();
        assert!(read_zero_cond_t(&tmp).unwrap());
        let _ = std::fs::remove_dir_all(&tmp);
    }

    #[test]
    fn zero_cond_t_errors_on_corrupt_json() {
        // A present-but-malformed config (partial download) must error, NOT silently downgrade to 2509.
        let tmp = zero_cond_t_tmp("corrupt");
        std::fs::write(tmp.join("transformer/config.json"), b"{ this is not json").unwrap();
        assert!(read_zero_cond_t(&tmp).is_err());
        let _ = std::fs::remove_dir_all(&tmp);
    }

    #[test]
    fn zero_cond_t_errors_on_wrong_type() {
        // `zero_cond_t` present but the wrong type → error naming the field, not a silent false.
        let tmp = zero_cond_t_tmp("wrongtype");
        std::fs::write(
            tmp.join("transformer/config.json"),
            br#"{"zero_cond_t": "yes"}"#,
        )
        .unwrap();
        let err = read_zero_cond_t(&tmp).unwrap_err().to_string();
        assert!(
            err.contains("zero_cond_t"),
            "error should name the field: {err}"
        );
        let _ = std::fs::remove_dir_all(&tmp);
    }

    #[test]
    fn tokenizer_json_path_prefers_tokenizer_then_processor() {
        // -2511 ships the assembled tokenizer.json only under processor/ (sc-6294).
        let tmp = std::env::temp_dir().join(format!("qwen_edit_tok_{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&tmp);
        std::fs::create_dir_all(tmp.join("processor")).unwrap();
        std::fs::write(tmp.join("processor/tokenizer.json"), b"{}").unwrap();
        assert!(tokenizer_json_path(&tmp)
            .unwrap()
            .ends_with("processor/tokenizer.json"));

        // When tokenizer/ also has it (the original Edit), that location wins.
        std::fs::create_dir_all(tmp.join("tokenizer")).unwrap();
        std::fs::write(tmp.join("tokenizer/tokenizer.json"), b"{}").unwrap();
        assert!(tokenizer_json_path(&tmp)
            .unwrap()
            .ends_with("tokenizer/tokenizer.json"));

        // Neither present → a descriptive error rather than a silent panic.
        let _ = std::fs::remove_dir_all(&tmp);
        assert!(tokenizer_json_path(&tmp).is_err());
    }

    /// Sequential-residency GPU validation (epic 10765 Phase 1c follow-up, sc-10968) — the edit sibling of
    /// `qwen_image_probed_generate_for_offload_ab`. ONE probed reference edit whose mode is either the
    /// `CANDLE_GEN_OFFLOAD` env (the override) or `QWEN_OFFLOAD_MODE=spec-sequential` →
    /// `QwenEditPaths::offload_policy` (the worker-facing contract); prints the device peak VRAM and writes
    /// the raw RGB pixels to `QWEN_OUT`. Run it TWICE in SEPARATE processes (resident vs sequential) and
    /// compare: the pixel files must be byte-identical (parity) and the sequential peak materially lower
    /// (the Qwen2.5-VL encoder + VAE encoder dropped before the DiT loads). Two processes are REQUIRED —
    /// cudarc's caching allocator never returns pages, so a second in-process run reads the first's peak.
    /// Ignored by default; needs a real-file (hardlink-staged) Qwen-Image-Edit snapshot in
    /// `QWEN_EDIT_SNAPSHOT`, a reference PPM in `QWEN_EDIT_REF`, and a CUDA device.
    ///
    /// Setting `QWEN_EDIT_LIGHTNING=1` re-points the same probe at the **Qwen-Image-Edit-2511-Lightning**
    /// few-step distill (sc-11066): the lightx2v 4-step LoRA at `QWEN_EDIT_LIGHTNING_LORA` folds into the
    /// MMDiT at load ([`QwenEditPaths::adapters`]) and the request runs 4-step **CFG-OFF** (`lightning:true`,
    /// `guidance:1.0` → a single MMDiT forward per step, no cond/uncond doubling). Same device-level peak
    /// protocol, so the resident/sequential peaks the runner prints are the true Lightning CFG-off numbers
    /// that replace the conservative base-CFG estimate carried in the manifest.
    #[cfg(feature = "cuda")]
    #[test]
    #[ignore = "needs QWEN_EDIT_SNAPSHOT + QWEN_EDIT_REF (a reference PPM) + a CUDA GPU"]
    fn qwen_edit_probed_generate_for_offload_ab() {
        use candle_gen::gen_core::AdapterKind;
        use candle_gen::testkit::{env_path, probe_gpu, read_ppm, VramProbe};

        let root = env_path("QWEN_EDIT_SNAPSHOT");
        let out = std::env::var("QWEN_OUT").expect("set QWEN_OUT to the pixel-dump path");
        let reference = read_ppm(&env_path("QWEN_EDIT_REF"));

        // Two ways to select sequential residency, both exercised by the A/B runner: the env override, or
        // `QWEN_OFFLOAD_MODE=spec-sequential` → `QwenEditPaths::offload_policy` (the worker contract).
        let spec_mode = std::env::var("QWEN_OFFLOAD_MODE").unwrap_or_default();
        let offload_policy = if spec_mode == "spec-sequential" {
            OffloadPolicy::Sequential
        } else {
            OffloadPolicy::Resident
        };

        // `QWEN_EDIT_LIGHTNING=1` → the CFG-off 4-step distill path (sc-11066): fold the lightx2v LoRA and
        // run `lightning:true` at guidance 1.0. Otherwise the base true-CFG 8-step path (the sc-11019
        // conservative upper bound). The base runs guidance 4.0 (a cond+uncond MMDiT batch); Lightning is a
        // single forward, which is exactly the peak delta this measure captures.
        let lightning = std::env::var("QWEN_EDIT_LIGHTNING")
            .map(|v| v == "1" || v.eq_ignore_ascii_case("true"))
            .unwrap_or(false);
        let adapters = if lightning {
            vec![AdapterSpec::new(
                env_path("QWEN_EDIT_LIGHTNING_LORA"),
                1.0,
                AdapterKind::Lora,
            )]
        } else {
            vec![]
        };
        let req = QwenEditRequest {
            prompt: "make the background a snowy mountain at sunset".into(),
            width: 1024,
            height: 1024,
            steps: if lightning { 4 } else { 8 },
            guidance: if lightning { 1.0 } else { 4.0 },
            seed: 42,
            lightning,
            ..Default::default()
        };

        let mut probe = VramProbe::start_rendered();
        let load_phase = probe.phase();
        let model = QwenEdit::load(&QwenEditPaths {
            root,
            adapters,
            offload_policy,
        })
        .expect("load QwenEdit");
        probe.end_load(load_phase);
        let generate_phase = probe.phase();
        let img = model
            .generate(&req, &[reference], &mut |_| {})
            .expect("generate");
        probe.end_gen(generate_phase);
        let report = probe.report().assert_trustworthy(1.0);
        let peak_mib = (report.peak_gb * 1.0e9 / (1024.0 * 1024.0)).round() as u64;
        std::fs::write(&out, &img.pixels).expect("write pixels");

        let env_mode = std::env::var("CANDLE_GEN_OFFLOAD").unwrap_or_default();
        let mode = if spec_mode == "spec-sequential" {
            "spec-sequential"
        } else if env_mode.eq_ignore_ascii_case("sequential") {
            "env-sequential"
        } else {
            "resident"
        };
        let path = if lightning { "lightning" } else { "base" };
        eprintln!(
            "SEQ_AB path={path} mode={mode} gpu={} peak_mib={peak_mib} | {report} | bytes={} {}x{} out={out}",
            probe_gpu(),
            img.pixels.len(),
            img.width,
            img.height
        );
    }
}
