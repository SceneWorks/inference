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
use std::sync::{Arc, Mutex};

use candle_gen::candle_core::{DType, Device, Tensor};
use candle_gen::candle_nn::VarBuilder;
use candle_gen::gen_core::runtime::CancelFlag;
use candle_gen::gen_core::tokenizer::TextTokenizer;
use candle_gen::gen_core::{
    AdapterSpec, GenerationMemory, Image, OffloadPolicy, PreviewSink, Progress, WeightsSource,
};
use candle_gen::{CandleError, Result};

use crate::config::{TextEncoderConfig, TransformerConfig, NEGATIVE_FALLBACK};
use crate::image_processor::{ImageInput, QwenImageProcessor};
use crate::pipeline;
use crate::transformer::QwenTransformer;
use crate::vae::{QwenVae, QwenVaeEncoder};
use crate::vision_language::{
    load_vision_language_encoder_with_text_encoder, validate_builtin_vision_encoder_source,
    QwenVisionLanguageEncoder,
};
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
    /// Optional decoder-LM substitution. The provider validates the complete Qwen2.5-VL text
    /// contract at construction; the image-conditioning `visual.*` tower remains sourced from
    /// `root/text_encoder` because it is outside the decoder contract.
    pub text_encoder: Option<WeightsSource>,
    /// LoRA/LoKr adapters folded into the MMDiT at load (sc-6220) — e.g. the Qwen-Image-Edit-2511
    /// Lightning distill, stacked ahead of any user adapters. **Empty** = the production (non-distilled)
    /// edit path: the transformer loads via the mmap fast path, byte-identical to before.
    pub adapters: Vec<AdapterSpec>,
    /// Legacy load-time policy retained for source compatibility. Image residency is request-scoped;
    /// [`QwenEditRequest::stage_residency`] is the sole lifecycle authority.
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
    /// Release the text/VAE-encode phase before loading the DiT/VAE-decode phase for this request.
    pub stage_residency: bool,
    /// Shared memory-ladder selection configured by the worker's request scope.
    pub memory: Option<GenerationMemory>,
    pub cancel: CancelFlag,
    /// Per-step latent-preview sink (epic 16948, sc-16952) — the bespoke-request twin of
    /// [`gen_core::GenerationRequest::preview`](candle_gen::gen_core::GenerationRequest::preview),
    /// carried as a field because the edit lane is a bespoke provider the worker drives by name
    /// rather than through the registry. The [`Default`] is inert, and an inert sink is
    /// seeded-byte-identical to a render with no preview at all.
    ///
    /// The frames are of the **target** image only. The reference latents are concatenated onto the
    /// DiT sequence inside the predict closure and narrowed straight back off, so the sampler's
    /// running latent — the only thing the hook is ever handed — never carries a reference token.
    pub preview: PreviewSink,
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
            stage_residency: false,
            memory: None,
            cancel: CancelFlag::default(),
            preview: PreviewSink::default(),
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
    stream_transformer_blocks: bool,
    cancel: &CancelFlag,
) -> Result<QwenTransformer> {
    let cfg = TransformerConfig::qwen_image();
    let dit_dir = root.join("transformer");
    // The DiT packed-detects each `Linear`: an MLX-packed edit tier (`SceneWorks/qwen-image-edit-2511
    // -mlx` q4/q8) loads straight from the packed parts at the `group_size` read from
    // `transformer/config.json` (64); a dense Edit snapshot loads unchanged (the group size is inert on
    // the dense path). See `crate::transformer_group_size`.
    let gs = crate::transformer_group_size(&dit_dir);
    if adapters.is_empty() {
        if stream_transformer_blocks {
            let files = candle_gen::sorted_safetensors(&dit_dir, "qwen edit")?;
            if let Some(packed) = crate::transformer_packed_config(&dit_dir) {
                use candle_gen::quant::PackedWeightSidecars;

                let prepared = PackedWeightSidecars::open_and_prepare_prefix_cancelable(
                    &files,
                    &dit_dir,
                    packed,
                    device,
                    cancel,
                    "transformer_blocks.",
                );
                if cancel.is_cancelled() {
                    return Err(CandleError::Canceled);
                }
                let (source, sidecars) = prepared?;
                let sidecars = Arc::new(sidecars);
                let vb = VarBuilder::from_backend(Box::new(source), dtype, device.clone());
                return Ok(QwenTransformer::new_block_streamed_with_sidecars_gs(
                    &cfg, vb, gs, sidecars,
                )?);
            }
            return Ok(QwenTransformer::new_block_streamed_gs(
                &cfg,
                component_vb(root, "transformer", dtype, device)?,
                gs,
            )?);
        }
        return Ok(QwenTransformer::new_gs(
            &cfg,
            component_vb(root, "transformer", dtype, device)?,
            gs,
        )?);
    }
    if stream_transformer_blocks {
        return Err(CandleError::Msg(
            "qwen edit: bounded transformer residency is unavailable when adapters are attached"
                .into(),
        ));
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
#[cfg(test)]
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
    lifecycle: Mutex<()>,
    stream_cancel: Arc<Mutex<CancelFlag>>,
    processor: QwenImageProcessor,
    tokenizer: TextTokenizer,
    zero_cond_t: bool,
    /// Complete caller-prepared identity retained across request-scoped deferred loads.
    prepared_spec: Option<candle_gen::gen_core::LoadSpec>,
}

struct EditText {
    vl_encoder: QwenVisionLanguageEncoder,
    vae_encoder: QwenVaeEncoder,
}

struct EditHeavy {
    transformer: QwenTransformer,
    vae: QwenVae,
}

fn resolve_edit_text_encoder_source(
    root: &Path,
    selected: Option<&WeightsSource>,
) -> Result<candle_gen::gen_core::ValidatedEncoderSource> {
    let selected = selected
        .cloned()
        .unwrap_or_else(|| WeightsSource::Dir(root.join("text_encoder")));
    let selected = crate::ENCODER_CONTRACT.validate_source_against_base(&selected, root)?;
    selected.load_time_quant_bits(None, "qwen_image_edit")?;
    Ok(selected)
}

impl QwenEdit {
    /// Load the cheap tokenizer / processor / `zero_cond_t` and retain request-scoped component
    /// loaders. The first warm request caches all four components; a staged request loads the
    /// vision/VAE encoders and render bundle in separate phases.
    pub fn load(paths: &QwenEditPaths) -> Result<Self> {
        let root = paths.root.clone();
        let te_cfg = TextEncoderConfig::qwen_image();
        let text_encoder_source =
            resolve_edit_text_encoder_source(&root, paths.text_encoder.as_ref())?;
        // Qwen Edit always consumes the built-in visual tower, including when request residency is
        // staged. Admit its exact config + header surface before device creation, tokenizer parsing,
        // or retention of any deferred payload loader, then carry this pin into both load closures.
        let vision_encoder_source = validate_builtin_vision_encoder_source(&root)?;
        let device = candle_gen::default_device()?;

        // Shared tokenizer policy (F-134 / sc-11190) with the edit lane's own `-2511` processor-bundle
        // path resolution — one `tokenizer_config()` home keeps edit's caption tokenization identical to
        // the txt2img lane's.
        let tokenizer = text_encoder_source.read_tokenizer_unchanged(|path| {
            TextTokenizer::from_file(path, crate::control_common::tokenizer_config(&te_cfg))
                .map_err(|e| CandleError::Msg(format!("qwen edit: load tokenizer: {e}")))
        })?;

        let resident_root = root.clone();
        let resident_device = device.clone();
        let resident_adapters = paths.adapters.clone();
        let resident_text_encoder = text_encoder_source.clone();
        let resident_vision_encoder = vision_encoder_source.clone();
        let text_root = root.clone();
        let text_device = device.clone();
        let request_text_encoder = text_encoder_source;
        let request_vision_encoder = vision_encoder_source;
        let heavy_root = root.clone();
        let heavy_device = device.clone();
        let heavy_adapters = paths.adapters.clone();
        let stream_cancel = Arc::new(Mutex::new(CancelFlag::default()));
        let heavy_cancel = stream_cancel.clone();
        let residency = candle_gen::Residency::request_scoped_with_resident(
            move |_| {
                Ok((
                    EditText {
                        vl_encoder: load_vision_language_encoder_with_text_encoder(
                            &resident_text_encoder,
                            &resident_vision_encoder,
                            &resident_device,
                        )?,
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
                            false,
                            &CancelFlag::default(),
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
            move |_| {
                Ok(EditText {
                    vl_encoder: load_vision_language_encoder_with_text_encoder(
                        &request_text_encoder,
                        &request_vision_encoder,
                        &text_device,
                    )?,
                    vae_encoder: QwenVaeEncoder::new(component_vb(
                        &text_root,
                        "vae",
                        ENC_DTYPE,
                        &text_device,
                    )?)?,
                })
            },
            move |_, stream_transformer_blocks| {
                Ok(EditHeavy {
                    transformer: load_transformer(
                        &heavy_root,
                        &heavy_adapters,
                        DIT_DTYPE,
                        &heavy_device,
                        stream_transformer_blocks,
                        &candle_gen::lock_recover(&heavy_cancel),
                    )?,
                    vae: QwenVae::new(component_vb(&heavy_root, "vae", ENC_DTYPE, &heavy_device)?)?,
                })
            },
        );

        Ok(Self {
            zero_cond_t: read_zero_cond_t(&root)?,
            device,
            residency,
            lifecycle: Mutex::new(()),
            stream_cancel,
            processor: QwenImageProcessor::default(),
            tokenizer,
            prepared_spec: None,
        })
    }

    /// Load through the exact prepared decoder-LM receipt retained by the caller.
    pub fn load_with_spec(
        paths: &QwenEditPaths,
        spec: &candle_gen::gen_core::LoadSpec,
    ) -> Result<Self> {
        match &spec.weights {
            WeightsSource::Dir(admitted_root) if admitted_root == &paths.root => {}
            WeightsSource::Dir(admitted_root) => {
                return Err(CandleError::Msg(format!(
                    "qwen edit: runtime base {} differs from admitted base {}",
                    paths.root.display(),
                    admitted_root.display()
                )));
            }
            WeightsSource::File(_) => {
                return Err(CandleError::Msg(
                    "qwen edit: admitted base must be the runtime snapshot directory".to_owned(),
                ));
            }
        }
        let mut model = spec.read_prepared_files_unchanged(|| {
            Self::load(&QwenEditPaths {
                root: paths.root.clone(),
                text_encoder: spec.text_encoder.clone(),
                adapters: spec.adapters.clone(),
                offload_policy: paths.offload_policy,
            })
        })?;
        model.prepared_spec = Some(spec.clone());
        Ok(model)
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
        let memory = req.memory.unwrap_or_default();
        let attention_budget = if memory.chunk_attention {
            candle_gen::gen_core::attention_budget::AttentionBudget::CONSTRAINED
        } else {
            candle_gen::gen_core::attention_budget::AttentionBudget::from_score_elements(
                candle_gen::ATTN_SCORES_BUDGET as u64,
                false,
            )
        };
        let attention_plan =
            candle_gen::gen_core::attention_budget::AttentionPlan::budgeted(attention_budget)
                .with_cancel(&req.cancel);
        let transformer_window = memory
            .transformer_window_size
            .map(|value| value as usize)
            .unwrap_or(crate::memory_strategy::TRANSFORMER_BLOCKS as usize);

        // Per-step latent preview (epic 16948, sc-16952). The sampler's running latent is the packed
        // NOISE prefix `[1, (H/16)·(W/16), 64]` alone — the reference concatenation and the true-CFG
        // pos/neg blend both happen inside the predict closure below, and the closure narrows its
        // result back to `noise_seq` — so the projector is handed target tokens only and unpacks them
        // to `[1, 16, H/8, W/8]` before applying the QwenVae fit.
        let preview = crate::preview::hook(&req.preview, req.width, req.height);

        let latents = candle_gen::run_flow_sampler(
            None,
            candle_gen::gen_core::sampling::TimestepConvention::Sigma,
            &sigmas,
            latents,
            req.seed,
            &req.cancel,
            on_progress,
            Some(&preview),
            |latents, sigma| -> Result<Tensor> {
                // Concatenate the (updating) noise with the (static) reference latents over the sequence.
                let joint = Tensor::cat(&[latents, static_latents], 1)?;
                let pos_v = transformer
                    .forward_edit_with_memory(
                        &joint,
                        pos,
                        sigma,
                        lat_h,
                        lat_w,
                        cond_grids,
                        self.zero_cond_t,
                        attention_plan,
                        transformer_window,
                        &req.cancel,
                    )?
                    .narrow(1, 0, noise_seq)?;
                match neg {
                    Some(neg) => {
                        let neg_v = transformer
                            .forward_edit_with_memory(
                                &joint,
                                neg,
                                sigma,
                                lat_h,
                                lat_w,
                                cond_grids,
                                self.zero_cond_t,
                                attention_plan,
                                transformer_window,
                                &req.cancel,
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
        let decoded = vae.decode_with_tile(
            &lat,
            memory.tile_vae_decode.then(|| {
                (
                    memory
                        .decode_tile_edge
                        .unwrap_or(crate::memory_strategy::DECODE_TILE_EDGE),
                    memory
                        .decode_overlap
                        .unwrap_or(crate::memory_strategy::DECODE_OVERLAP),
                )
            }),
        )?;
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
        let _lifecycle = candle_gen::lock_recover(&self.lifecycle);
        read_with_prepared_spec(self.prepared_spec.as_ref(), || {
            *candle_gen::lock_recover(&self.stream_cancel) = req.cancel.clone();
            let memory = req.memory.unwrap_or_default();
            let stage_residency = req.stage_residency || memory.stage_residency;
            if (memory.tile_vae_decode
                || memory.chunk_attention
                || memory.stream_transformer_blocks)
                && !stage_residency
            {
                return Err(CandleError::Msg(
                    "qwen edit: bounded decode, attention, and transformer residency require request-scoped staged residency"
                        .into(),
                ));
            }
            self.residency.run_request_scoped(
                stage_residency,
                memory.stream_transformer_blocks,
                &req.cancel,
                false,
                on_progress,
                |text| {
                    self.encode_conditioning(&text.vl_encoder, &text.vae_encoder, req, references)
                },
                |_| Ok(self.device.synchronize()?),
                |heavy, (pos, neg, static_latents, cond_grids), on_progress| {
                    let result = self.denoise_and_decode(
                        &heavy.transformer,
                        &heavy.vae,
                        req,
                        &pos,
                        neg.as_ref(),
                        &static_latents,
                        &cond_grids,
                        on_progress,
                    );
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

    fn edit_paths(root: &Path, offload_policy: OffloadPolicy) -> QwenEditPaths {
        QwenEditPaths {
            root: root.to_path_buf(),
            text_encoder: None,
            adapters: Vec::new(),
            offload_policy,
        }
    }

    fn edit_load_error(root: &Path, offload_policy: OffloadPolicy) -> String {
        match QwenEdit::load(&edit_paths(root, offload_policy)) {
            Ok(_) => panic!("invalid built-in vision source reached deferred residency"),
            Err(error) => error.to_string(),
        }
    }

    fn rename_safetensors_header_key(path: &Path, from: &str, to: &str) {
        use std::io::{Read, Seek, SeekFrom, Write};

        assert_eq!(from.len(), to.len());
        let mut file = std::fs::OpenOptions::new()
            .read(true)
            .write(true)
            .open(path)
            .unwrap();
        let mut encoded_len = [0_u8; 8];
        file.read_exact(&mut encoded_len).unwrap();
        let mut header = vec![0_u8; u64::from_le_bytes(encoded_len) as usize];
        file.read_exact(&mut header).unwrap();
        let matches = header
            .windows(from.len())
            .enumerate()
            .filter_map(|(offset, bytes)| (bytes == from.as_bytes()).then_some(offset))
            .collect::<Vec<_>>();
        assert_eq!(matches.len(), 1, "fixture must contain one {from} header");
        let start = matches[0];
        header[start..start + to.len()].copy_from_slice(to.as_bytes());
        file.seek(SeekFrom::Start(8)).unwrap();
        file.write_all(&header).unwrap();
    }

    #[test]
    fn request_defaults() {
        let r = QwenEditRequest::default();
        assert_eq!((r.width, r.height), (1024, 1024));
        assert_eq!(r.steps, 30);
        assert!(!r.cancel.is_cancelled());
    }

    #[test]
    fn selected_decoder_contract_is_validated_separately_from_the_builtin_vision_tower() {
        let fixture = tempfile::tempdir().unwrap();
        gen_core_testkit::write_encoder_contract_tokenizer_fixture(
            fixture.path(),
            crate::ENCODER_CONTRACT,
        )
        .unwrap();
        let selected = fixture.path().join("selected-decoder");
        gen_core_testkit::write_encoder_contract_fixture(&selected, crate::ENCODER_CONTRACT)
            .unwrap();
        resolve_edit_text_encoder_source(fixture.path(), Some(&WeightsSource::Dir(selected)))
            .expect("exact selected decoder contract");

        let wrong = fixture.path().join("wrong-kv-heads");
        gen_core_testkit::write_encoder_contract_fixture(&wrong, crate::ENCODER_CONTRACT).unwrap();
        let config_path = wrong.join("config.json");
        let mut config: serde_json::Value =
            serde_json::from_slice(&std::fs::read(&config_path).unwrap()).unwrap();
        config["num_key_value_heads"] =
            serde_json::json!(crate::ENCODER_CONTRACT.num_key_value_heads + 1);
        std::fs::write(&config_path, serde_json::to_vec(&config).unwrap()).unwrap();
        let error =
            resolve_edit_text_encoder_source(fixture.path(), Some(&WeightsSource::Dir(wrong)))
                .expect_err("wrong GQA shape must reject")
                .to_string();
        assert!(error.contains("num_key_value_heads"), "unexpected: {error}");
    }

    #[test]
    fn public_load_admits_builtin_vision_before_retaining_deferred_loaders() {
        let valid = tempfile::tempdir().unwrap();
        gen_core_testkit::write_multimodal_encoder_contract_fixture(
            &valid.path().join("text_encoder"),
            crate::ENCODER_CONTRACT,
            crate::VISION_ENCODER_CONTRACT,
        )
        .unwrap();
        QwenEdit::load(&edit_paths(valid.path(), OffloadPolicy::Sequential))
            .expect("validation-complete sparse metadata must reach deferred residency");

        let missing = tempfile::tempdir().unwrap();
        gen_core_testkit::write_encoder_contract_fixture(
            &missing.path().join("text_encoder"),
            crate::ENCODER_CONTRACT,
        )
        .unwrap();

        let wrong_config = tempfile::tempdir().unwrap();
        gen_core_testkit::write_multimodal_encoder_contract_fixture(
            &wrong_config.path().join("text_encoder"),
            crate::ENCODER_CONTRACT,
            crate::VISION_ENCODER_CONTRACT,
        )
        .unwrap();
        let config_path = wrong_config.path().join("text_encoder/config.json");
        let mut config: serde_json::Value =
            serde_json::from_slice(&std::fs::read(&config_path).unwrap()).unwrap();
        config["vision_config"]["depth"] =
            serde_json::json!(crate::VISION_ENCODER_CONTRACT.num_hidden_layers + 1);
        std::fs::write(&config_path, serde_json::to_vec(&config).unwrap()).unwrap();

        let wrong_header = tempfile::tempdir().unwrap();
        gen_core_testkit::write_multimodal_encoder_contract_fixture(
            &wrong_header.path().join("text_encoder"),
            crate::ENCODER_CONTRACT,
            crate::VISION_ENCODER_CONTRACT,
        )
        .unwrap();
        rename_safetensors_header_key(
            &wrong_header.path().join("text_encoder/model.safetensors"),
            "visual.patch_embed.proj.weight",
            "visual.patch_embed.proj.weighx",
        );

        for policy in [OffloadPolicy::Resident, OffloadPolicy::Sequential] {
            let error = edit_load_error(missing.path(), policy);
            assert!(error.contains("vision_config"), "unexpected: {error}");

            let error = edit_load_error(wrong_config.path(), policy);
            assert!(error.contains("depth"), "unexpected: {error}");

            let error = edit_load_error(wrong_header.path(), policy);
            assert!(
                error.contains("visual.patch_embed.proj.weight") && error.contains("missing"),
                "unexpected: {error}"
            );
        }
    }

    fn zero_cond_t_tmp(tmp: &tempfile::TempDir, name: &str) -> PathBuf {
        let tmp = tmp.path().join(format!(
            "qwen_edit_zct_{name}_{:?}",
            std::thread::current().id()
        ));
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
        let tmp = tempfile::tempdir().unwrap();
        // Config present but the key genuinely absent (a valid 2509 config.json) → documented default.
        let tmp = zero_cond_t_tmp(&tmp, "keyabsent");
        std::fs::write(
            tmp.join("transformer/config.json"),
            br#"{"num_layers": 60}"#,
        )
        .unwrap();
        assert!(!read_zero_cond_t(&tmp).unwrap());
    }

    #[test]
    fn zero_cond_t_reads_present_value() {
        let tmp = tempfile::tempdir().unwrap();
        // Edit-2511 config with the key set true → true.
        let tmp = zero_cond_t_tmp(&tmp, "present");
        std::fs::write(
            tmp.join("transformer/config.json"),
            br#"{"zero_cond_t": true}"#,
        )
        .unwrap();
        assert!(read_zero_cond_t(&tmp).unwrap());
    }

    #[test]
    fn zero_cond_t_errors_on_corrupt_json() {
        let tmp = tempfile::tempdir().unwrap();
        // A present-but-malformed config (partial download) must error, NOT silently downgrade to 2509.
        let tmp = zero_cond_t_tmp(&tmp, "corrupt");
        std::fs::write(tmp.join("transformer/config.json"), b"{ this is not json").unwrap();
        assert!(read_zero_cond_t(&tmp).is_err());
    }

    #[test]
    fn zero_cond_t_errors_on_wrong_type() {
        let tmp = tempfile::tempdir().unwrap();
        // `zero_cond_t` present but the wrong type → error naming the field, not a silent false.
        let tmp = zero_cond_t_tmp(&tmp, "wrongtype");
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
    }

    #[test]
    fn tokenizer_json_path_prefers_tokenizer_then_processor() {
        // -2511 ships the assembled tokenizer.json only under processor/ (sc-6294).
        let tmp_guard = tempfile::tempdir().unwrap();
        let tmp = tmp_guard.path().to_path_buf();
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

        // Neither present → a descriptive error rather than a silent panic. Emptying the
        // guarded root is the point of this leg, not cleanup.
        std::fs::remove_dir_all(tmp.join("processor")).unwrap();
        std::fs::remove_dir_all(tmp.join("tokenizer")).unwrap();
        assert!(tokenizer_json_path(&tmp).is_err());
    }

    /// Sequential-residency GPU validation (epic 10765 Phase 1c follow-up, sc-10968) — the edit sibling of
    /// `qwen_image_probed_generate_for_offload_ab`. ONE probed reference edit whose residency is carried
    /// by `QwenEditRequest::stage_residency`; prints the device peak VRAM and writes
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
        use candle_gen::gen_core::{AdapterKind, LoadSpec, WeightsSource};
        use candle_gen::testkit::{env_path, probe_gpu, read_ppm, VramProbe};

        let root = env_path("QWEN_EDIT_SNAPSHOT");
        let out = std::env::var("QWEN_OUT").expect("set QWEN_OUT to the pixel-dump path");
        let reference = read_ppm(&env_path("QWEN_EDIT_REF"));

        let stage_residency =
            std::env::var("QWEN_OFFLOAD_MODE").is_ok_and(|mode| mode == "request-staged");

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
        let mut memory_spec = LoadSpec::new(WeightsSource::Dir(root.clone()));
        memory_spec.adapters = adapters.clone();
        let (observed_calibration, tier) =
            crate::memory_strategy::evidence_identity_and_tier("qwen_image_edit", &memory_spec)
                .expect("resolve qwen_image_edit executable evidence identity and tier");
        let evidence_load_shape = observed_calibration.load_shape;
        let req = QwenEditRequest {
            prompt: "make the background a snowy mountain at sunset".into(),
            width: 1024,
            height: 1024,
            steps: if lightning { 4 } else { 8 },
            guidance: if lightning { 1.0 } else { 4.0 },
            seed: 42,
            lightning,
            stage_residency,
            ..Default::default()
        };

        assert!(
            candle_gen::testkit::reset_cuda_mempool_high_water(0),
            "reset CUDA live-allocation high-water"
        );
        let mut probe = VramProbe::start_rendered();
        let load_phase = probe.phase();
        let model = QwenEdit::load(&QwenEditPaths {
            root,
            text_encoder: None,
            adapters,
            offload_policy: OffloadPolicy::Resident,
        })
        .expect("load QwenEdit");
        probe.end_load(load_phase);
        let generate_phase = probe.phase();
        let img = model
            .generate(&req, &[reference], &mut |_| {})
            .expect("generate");
        probe.end_gen(generate_phase);
        let report = probe.report().assert_trustworthy(1.0);
        let live_peak_bytes = candle_gen::testkit::cuda_mempool_used_high_bytes(0)
            .expect("read CUDA live-allocation high-water");
        assert!(
            live_peak_bytes > 0,
            "CUDA live-allocation peak must be positive"
        );
        std::fs::write(&out, &img.pixels).expect("write pixels");

        let strategy = if stage_residency {
            candle_gen::gen_core::MemoryStrategy::StagedResidency
        } else {
            candle_gen::gen_core::MemoryStrategy::Resident
        };
        let path = if lightning { "lightning" } else { "base" };
        eprintln!(
            "{}",
            candle_gen::testkit::memory_evidence_v1_line(
                candle_gen::testkit::MemoryEvidenceProbe {
                    resolved_route: if lightning {
                        "qwen_image_edit_lightning"
                    } else {
                        "qwen_image_edit"
                    },
                    declared_calibration: candle_gen::testkit::expected_memory_calibration(
                        evidence_load_shape,
                    ),
                    load_shape: evidence_load_shape,
                    observed_calibration,
                    tier,
                    mode: candle_gen::gen_core::MemoryMode::Edit,
                    overlay: lightning.then(|| "lightning".to_owned()),
                    geometry: candle_gen::gen_core::MemoryGeometry {
                        width: req.width,
                        height: req.height,
                        batch: 1,
                        frames: 1,
                        reference_count: 1,
                    },
                    strategy,
                    engaged_composition: if stage_residency {
                        vec![
                            candle_gen::gen_core::MemoryStrategy::Resident,
                            candle_gen::gen_core::MemoryStrategy::StagedResidency,
                        ]
                    } else {
                        vec![candle_gen::gen_core::MemoryStrategy::Resident]
                    },
                    parameters: candle_gen::gen_core::MemoryStrategyParameters::default(),
                    observed_peak_bytes: live_peak_bytes,
                    harness_version: "candle-qwen-image-edit-residency-v1",
                    output_bytes: &img.pixels,
                }
            )
        );
        eprintln!(
            "MEMORY_EVIDENCE_DIAGNOSTIC path={path} gpu={} {report} bytes={} {}x{} out={out}",
            probe_gpu(),
            img.pixels.len(),
            img.width,
            img.height
        );
    }
}
