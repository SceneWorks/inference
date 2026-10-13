//! The Qwen-Image 2.1 text-to-image pipeline body — the port of `QwenImage21Pipeline.__call__`
//! for the text-only path: packed noise → resolution-shifted flow-match Euler denoise (optional
//! true CFG) → unpack → latent denormalisation → RGBA decode → RGB emission.
//!
//! Progress, cancellation, the per-step boundary and the curated-sampler axis all come from the
//! shared [`run_flow_sampler`] contract; this module only supplies the velocity closure and the
//! Qwen-Image 2.1 latent layout (unpatched `[1, h·w, 64]`, `TimestepConvention::Sigma`).

use candle_core::{DType, Device, IndexOp, Tensor};
use candle_gen::gen_core::sampling::TimestepConvention;
use candle_gen::gen_core::tiling::TilingConfig;
use candle_gen::gen_core::tokenizer::TextTokenizer;
use candle_gen::gen_core::{CancelFlag, GenerationRequest, Image, Progress, RgbaImage};
use candle_gen::run_flow_sampler;
use candle_gen::{CandleError as Error, Result};

use crate::config::VAE_SCALE_FACTOR;
use crate::reference::{prepare_references, PreparedReference};
use crate::text_encoder::{QwenImage21TextEncoder, TextConditioning};
use crate::transformer::{JointLayout, PrefixKvCache, QwenImage21Transformer, Segment};
use crate::vae::QwenImage21Vae;

/// `_pack_latents`: 2.1 consumes latents unpatched, so packing is a plain spatial flatten —
/// `[B, C, h, w]` → `[B, h·w, C]`.
pub fn pack_latents(latents: &Tensor) -> Result<Tensor> {
    let dims = latents.dims().to_vec();
    if dims.len() != 4 {
        return Err(Error::Msg(format!(
            "qwen_image_2_1: pack_latents expects [B, C, h, w], got {dims:?}"
        )));
    }
    let (b, c, h, w) = (dims[0], dims[1], dims[2], dims[3]);
    Ok(latents
        .reshape((b, c, h * w))?
        .transpose(1, 2)?
        .contiguous()?)
}

/// `_unpack_latents`: `[B, h·w, C]` → `[B, C, h, w]` for a `width × height` image.
pub fn unpack_latents(latents: &Tensor, width: u32, height: u32) -> Result<Tensor> {
    let (h, w) = latent_grid(width, height);
    let dims = latents.dims().to_vec();
    if dims.len() != 3 || dims[1] != h * w {
        return Err(Error::Msg(format!(
            "qwen_image_2_1: unpack_latents expects [B, {}, C] for {width}x{height}, got {dims:?}",
            h * w
        )));
    }
    let (b, c) = (dims[0], dims[2]);
    Ok(latents
        .transpose(1, 2)?
        .reshape((b, c, h, w))?
        .contiguous()?)
}

/// The latent grid `(h, w)` of a `width × height` request: `2 · (size // 32)` per side.
pub fn latent_grid(width: u32, height: u32) -> (usize, usize) {
    let multiple = VAE_SCALE_FACTOR * 2;
    (
        2 * (height / multiple) as usize,
        2 * (width / multiple) as usize,
    )
}

/// Seeded packed Gaussian noise `[1, h·w, channels]` (f32) for a `width × height` request.
/// Drawn on the CPU through the shared launch-portable seeded normal draw, so a seed reproduces
/// bit-for-bit on every candle backend (and differs from upstream's torch stream — seed parity
/// across frameworks is not a goal, exactly as on the MLX twin).
pub fn create_noise(
    seed: u64,
    width: u32,
    height: u32,
    channels: usize,
    device: &Device,
) -> Result<Tensor> {
    let (h, w) = latent_grid(width, height);
    let mut rng = <rand::rngs::StdRng as rand::SeedableRng>::seed_from_u64(seed);
    let values = candle_gen::seeded_normal_vec(&mut rng, h * w * channels);
    Ok(Tensor::from_vec(values, (1, h * w, channels), &Device::Cpu)?.to_device(device)?)
}

/// Prompt → conditioning `[1, L, hidden]` (f32) through the loaded tower; `drop` is
/// [`crate::system_prompt_drop_count`] for the loaded tokenizer.
pub fn encode_prompt(
    text_encoder: &QwenImage21TextEncoder,
    tokenizer: &TextTokenizer,
    prompt: &str,
    drop: usize,
) -> Result<Tensor> {
    text_encoder.encode_prompt(tokenizer, prompt, drop)
}

/// VAE-encode every prepared reference, in order, into the denoiser-space packed latents the
/// joint sequence prepends to the noise — `prepare_latents`' `images is not None` branch:
/// `_encode_vae_image` with `sample_mode="argmax"` (the posterior **mode**, not a sample), the
/// same `(z − mean) / std` normalisation the target latents use, then the unpatched
/// `[1, (h/16)·(w/16), 64]` flatten.
pub fn encode_references(
    vae: &QwenImage21Vae,
    references: &[PreparedReference],
) -> Result<Vec<Tensor>> {
    references
        .iter()
        .map(|reference| {
            let mode = vae.encode_mode(&reference.vae_input)?;
            pack_latents(&vae.normalize(&mode)?)
        })
        .collect()
}

/// The joint text/image layout for a conditioned request — the Rust form of upstream's
/// `image_pad_mask` expansion (`repeat_interleave(img_mask, where(img_mask, 4, 1))`) plus the
/// appended target block.
///
/// `image_pad_mask` marks the vision slots in the **VLM** sequence; each slot stands for a 2×2
/// group of latent tokens, so a run of `n` slots belonging to reference `k` becomes one
/// [`Segment::Image`] of that reference's `(h/16, w/16)` grid. Block boundaries come from the
/// per-reference slot counts, not from runs of `true`: two adjacent references with no text
/// between them stay two blocks, exactly as `build_token_metadata` insists.
pub fn joint_layout(
    image_pad_mask: &[bool],
    references: &[PreparedReference],
    width: u32,
    height: u32,
) -> Result<JointLayout> {
    let slots = image_pad_mask.iter().filter(|m| **m).count();
    let expected: usize = references.iter().map(PreparedReference::vision_slots).sum();
    if slots != expected {
        return Err(Error::Msg(format!(
            "qwen_image_2_1: the prompt reserved {slots} vision slots but the {} reference images \
             need {expected}",
            references.len()
        )));
    }
    let mut segments: Vec<Segment> = Vec::with_capacity(2 * references.len() + 2);
    let mut text_run = 0usize;
    let mut cursor = 0usize;
    let mut next_reference = 0usize;
    while cursor < image_pad_mask.len() {
        if !image_pad_mask[cursor] {
            text_run += 1;
            cursor += 1;
            continue;
        }
        if text_run > 0 {
            segments.push(Segment::Text { len: text_run });
            text_run = 0;
        }
        let reference = references.get(next_reference).ok_or_else(|| {
            Error::Msg(
                "qwen_image_2_1: the prompt carries more vision-slot runs than reference images"
                    .into(),
            )
        })?;
        let take = reference.vision_slots();
        for offset in 0..take {
            if !image_pad_mask
                .get(cursor + offset)
                .copied()
                .unwrap_or(false)
            {
                return Err(Error::Msg(format!(
                    "qwen_image_2_1: reference image {next_reference} needs {take} contiguous \
                     vision slots but the prompt breaks the run after {offset}"
                )));
            }
        }
        let (h, w) = reference.latent_grid();
        segments.push(Segment::Image {
            height: h,
            width: w,
        });
        cursor += take;
        next_reference += 1;
    }
    if next_reference != references.len() {
        return Err(Error::Msg(format!(
            "qwen_image_2_1: only {next_reference} of {} reference images found a vision-slot run \
             in the prompt",
            references.len()
        )));
    }
    if text_run > 0 {
        segments.push(Segment::Text { len: text_run });
    }
    let (h, w) = latent_grid(width, height);
    segments.push(Segment::Image {
        height: h,
        width: w,
    });
    Ok(JointLayout { segments })
}

/// The text rows of a conditioning tensor — everything the DiT does **not** overwrite with a
/// condition latent. `txt_in` is row-wise, so selecting the rows before the projection is exactly
/// upstream's "project everything, then scatter the latents over the image rows".
pub fn text_rows(hidden: &Tensor, image_pad_mask: &[bool]) -> Result<Tensor> {
    if image_pad_mask.iter().all(|m| !*m) {
        return Ok(hidden.clone());
    }
    let keep: Vec<u32> = image_pad_mask
        .iter()
        .enumerate()
        .filter(|(_, m)| !**m)
        .map(|(i, _)| i as u32)
        .collect();
    let index = Tensor::from_vec(keep.clone(), keep.len(), hidden.device())?;
    Ok(hidden.index_select(&index, 1)?.contiguous()?)
}

/// One prompt branch as the DiT consumes it: the conditioning's text rows and the joint layout
/// (condition blocks interleaved with the text, target block last). The render path builds one per
/// CFG branch; the edit trainer (sc-24162) builds one per dataset item through the same
/// [`joint_branch`], so the two cannot drift.
pub struct JointBranch {
    /// `[1, text_len, hidden]` — the conditioning with the vision-slot rows removed.
    pub text: Tensor,
    /// The joint text/image layout, target block last.
    pub layout: JointLayout,
}

/// Assemble one prompt branch ([`JointBranch`]) from its encoded conditioning, the ordered
/// prepared references it was encoded with, and the target's `width × height`:
/// [`joint_layout`] over the image-pad mask and [`text_rows`] of the hidden states.
pub fn joint_branch(
    conditioning: &TextConditioning,
    references: &[PreparedReference],
    width: u32,
    height: u32,
) -> Result<JointBranch> {
    let layout = joint_layout(&conditioning.image_pad_mask, references, width, height)?;
    let text = text_rows(&conditioning.hidden, &conditioning.image_pad_mask)?;
    Ok(JointBranch { text, layout })
}

/// The ordered image stream [`QwenImage21Transformer::forward_joint`] consumes for a layout built
/// by [`joint_branch`]: every condition latent in reference order, then the target — upstream's
/// `latent_model_input = cat([*reference_latents, latents], dim=1)`. Shared by the denoise loop and
/// the edit trainer's forward (sc-24162).
pub fn joint_images<'a>(reference_latents: &'a [Tensor], target: &'a Tensor) -> Vec<&'a Tensor> {
    let mut images: Vec<&'a Tensor> = Vec::with_capacity(reference_latents.len() + 1);
    images.extend(reference_latents.iter());
    images.push(target);
    images
}

/// Host-preprocess an ordered reference list against the loaded tower's own Qwen3-VL processor
/// geometry ([`prepare_references`]). Empty ⇒ the text-to-image route (no vision tower needed); a
/// non-empty list on a snapshot that ships no vision tower is a typed refusal
/// ([`missing_vision_tower`]). Shared by the render path and the edit trainer (sc-24162).
pub fn prepare_conditioning_references(
    text_encoder: &QwenImage21TextEncoder,
    images: &[RgbaImage],
) -> Result<Vec<PreparedReference>> {
    if images.is_empty() {
        return Ok(Vec::new());
    }
    let vision = text_encoder
        .vision_config()
        .ok_or_else(missing_vision_tower)?;
    prepare_references(images, vision, text_encoder.device())
}

/// The typed refusal for reference conditioning on a snapshot without a Qwen3-VL vision tower.
pub fn missing_vision_tower() -> Error {
    Error::Unsupported(
        "qwen_image_2_1: reference conditioning needs the snapshot's Qwen3-VL vision tower \
         (`text_encoder/config.json` `vision_config` + `model.visual.*`), which this snapshot \
         does not carry"
            .to_string(),
    )
}

/// The reference conditioning one denoise run carries: the packed condition latents (in order)
/// and the joint layout of each branch.
pub struct ReferenceConditioning<'a> {
    /// Packed `[1, tokens, 64]` latents, condition images first, in request order.
    pub latents: &'a [Tensor],
    /// Layout of the positive branch.
    pub layout: &'a JointLayout,
    /// Layout of the negative branch — a different prompt is a different text length, so true CFG
    /// needs its own.
    pub negative_layout: Option<&'a JointLayout>,
}

/// The prefix-KV-cache switch of one denoise run — upstream's `use_kv_cache` (default `True`).
///
/// The cache trades memory for time: it holds `2 · blocks · prefix_tokens · inner_dim` values per
/// prompt branch ([`PrefixKvCache::nbytes`]) for the length of the denoise loop, and in exchange
/// every model evaluation after the first runs the blocks over the target tokens only instead of
/// the whole joint sequence. Shedding it ([`Self::Off`]) is exactly the prefill path, unchanged.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum KvCacheMode {
    /// Prefill once, decode from the cache afterwards (wherever the model allows it — see
    /// [`KvCachePath`]).
    #[default]
    On,
    /// Evaluate the full block-causal joint sequence at every model evaluation.
    Off,
}

/// Which path a denoise run took — the answer to "did the cache run?".
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum KvCachePath {
    /// Every model evaluation ran the full joint sequence: the cache was [`KvCacheMode::Off`], or
    /// the model is not `causal_condition` (the only configuration the cache is valid under — as
    /// upstream's `use_kv_cache and transformer.config.causal_condition`).
    #[default]
    Prefill,
    /// The first model evaluation of each branch prefilled its cache; every later one decoded the
    /// target tokens from it.
    Cached,
}

/// The prompt branch of one model evaluation.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum PromptBranch {
    Positive,
    /// The true-CFG unconditional branch.
    Negative,
}

/// What one transformer forward did.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum EvaluationPath {
    /// Full joint sequence, no cache involved.
    Prefill,
    /// Full joint sequence, and the branch's prefix K/V were captured into its cache.
    Extract,
    /// Target tokens only, against the branch's cached prefix K/V.
    Cached,
}

/// One transformer forward of a denoise run, in the order they ran.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct ModelEvaluation {
    pub branch: PromptBranch,
    pub path: EvaluationPath,
    /// Tokens the transformer blocks ran over. For [`EvaluationPath::Extract`] and
    /// [`EvaluationPath::Cached`] it is read off the block stack's own output; for
    /// [`EvaluationPath::Prefill`] it is the joint layout's length.
    pub block_tokens: usize,
}

/// What a denoise run did, filled in as it runs — so it is complete up to the point of a
/// cancellation or an error, not only on success.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct DenoiseReport {
    /// The path the run took.
    pub kv_cache: KvCachePath,
    /// Bytes the run's prefix caches held (both branches), `0` on the prefill path.
    pub kv_cache_bytes: usize,
    /// Every transformer forward, in order. A solver that evaluates the model more than once per
    /// step contributes one entry per evaluation and branch.
    pub evaluations: Vec<ModelEvaluation>,
}

/// Everything one denoise run needs.
pub struct DenoiseInputs<'a> {
    pub transformer: &'a QwenImage21Transformer,
    /// Descending schedule, trailing `0.0` (see [`crate::scheduler`]).
    pub sigmas: &'a [f32],
    /// Initial packed latents `[1, h·w, C]`.
    pub latents: Tensor,
    pub prompt_embeds: &'a Tensor,
    /// The negative branch; `Some` enables true CFG with `true_cfg_scale`.
    pub negative_embeds: Option<&'a Tensor>,
    pub true_cfg_scale: f32,
    pub width: u32,
    pub height: u32,
    /// Curated solver name (`None` = Euler, the native path).
    pub sampler: Option<&'a str>,
    pub seed: u64,
    pub cancel: &'a CancelFlag,
    /// Ordered reference conditioning, or `None` for the text-to-image route.
    pub references: Option<ReferenceConditioning<'a>>,
    /// The prefix KV cache switch. [`KvCacheMode::Off`] is the prefill path, byte-for-byte.
    pub kv_cache: KvCacheMode,
}

/// One prompt branch of the velocity closure: its text rows, its reference layout (if any) and —
/// on the cached path — the prefix cache its first evaluation fills.
struct BranchRun<'a> {
    kind: PromptBranch,
    text: &'a Tensor,
    /// `Some` on the reference route: the condition latents and this branch's joint layout.
    conditioned: Option<(&'a [Tensor], &'a JointLayout)>,
    /// Filled by this branch's first evaluation and by nothing else; a cache never crosses
    /// branches (a different prompt is a different prefix).
    cache: Option<PrefixKvCache>,
}

impl BranchRun<'_> {
    /// One model evaluation of this branch at `sigma`.
    ///
    /// On the cached path the rule is per **model evaluation**, not per sampler step: whichever
    /// evaluation comes first prefills, every other one decodes. The cache is step-independent, so
    /// a solver that evaluates twice per step or off the schedule (Heun, DPM++ SDE) needs nothing
    /// more. The slot is filled only by a completed extract — an evaluation that fails or is
    /// cancelled leaves it empty, so no later evaluation can decode from a missing or partial
    /// cache.
    fn evaluate(
        &mut self,
        transformer: &QwenImage21Transformer,
        cached: bool,
        latents: &Tensor,
        sigma: f32,
        grid: (usize, usize),
        report: &mut DenoiseReport,
    ) -> Result<Tensor> {
        let (h, w) = grid;
        let (velocity, path, block_tokens) = if !cached {
            match self.conditioned {
                Some((reference_latents, layout)) => {
                    let images = joint_images(reference_latents, latents);
                    (
                        transformer.forward_joint(self.text, &images, sigma, layout)?,
                        EvaluationPath::Prefill,
                        layout.total_len(),
                    )
                }
                None => (
                    transformer.forward(latents, self.text, sigma, h, w)?,
                    EvaluationPath::Prefill,
                    self.text.dim(1)? + h * w,
                ),
            }
        } else if let Some(cache) = &self.cache {
            let (velocity, block_tokens) = transformer.cached_counted(latents, sigma, cache)?;
            (velocity, EvaluationPath::Cached, block_tokens)
        } else {
            let (velocity, cache, block_tokens) = match self.conditioned {
                Some((reference_latents, layout)) => transformer.extract_counted(
                    self.text,
                    &joint_images(reference_latents, latents),
                    sigma,
                    layout,
                )?,
                None => transformer.extract_counted(
                    self.text,
                    &[latents],
                    sigma,
                    &JointLayout::text_to_image(self.text.dim(1)?, h, w),
                )?,
            };
            report.kv_cache_bytes += cache.nbytes();
            self.cache = Some(cache);
            (velocity, EvaluationPath::Extract, block_tokens)
        };
        report.evaluations.push(ModelEvaluation {
            branch: self.kind,
            path,
            block_tokens,
        });
        Ok(velocity)
    }
}

/// The denoise loop: with a negative branch each model evaluation applies `neg + s·(pos − neg)`.
/// Returns the final packed latents (f32). [`denoise_reported`] without the report.
///
/// With [`DenoiseInputs::references`] set, the condition latents are prepended to the target
/// latents (`latent_model_input = cat([*reference_latents, latents], dim=1)`) and the
/// transformer attends over the full interleaved layout; the velocity it returns is already
/// sliced to the target block.
pub fn denoise(inputs: DenoiseInputs<'_>, on_progress: &mut dyn FnMut(Progress)) -> Result<Tensor> {
    denoise_reported(inputs, &mut DenoiseReport::default(), on_progress)
}

/// [`denoise`], recording into `report` which path ran and every transformer forward it made.
///
/// With [`KvCacheMode::Off`] every evaluation runs the transformer over the full joint sequence —
/// the pre-cache loop, unchanged. With [`KvCacheMode::On`] (and a `causal_condition` model) the
/// first evaluation of each branch prefills that branch's [`PrefixKvCache`] and every later one
/// feeds the blocks the target tokens only. The caches live exactly as long as this call: they are
/// owned by the velocity closure the sampler consumes, so they are dropped before it returns — on
/// success, cancellation and error alike — and nothing of them reaches the VAE decode.
///
/// `report` is reset on entry and written as the loop runs, so after a cancellation it still says
/// what ran.
pub fn denoise_reported(
    inputs: DenoiseInputs<'_>,
    report: &mut DenoiseReport,
    on_progress: &mut dyn FnMut(Progress),
) -> Result<Tensor> {
    let grid = latent_grid(inputs.width, inputs.height);
    let transformer = inputs.transformer;
    let scale = inputs.true_cfg_scale as f64;
    let references = inputs.references.as_ref();
    // `cache_enabled = use_kv_cache and self.transformer.config.causal_condition`.
    let cached = inputs.kv_cache == KvCacheMode::On && transformer.config().causal_condition;
    *report = DenoiseReport {
        kv_cache: if cached {
            KvCachePath::Cached
        } else {
            KvCachePath::Prefill
        },
        ..DenoiseReport::default()
    };
    let mut positive = BranchRun {
        kind: PromptBranch::Positive,
        text: inputs.prompt_embeds,
        conditioned: references.map(|c| (c.latents, c.layout)),
        cache: None,
    };
    let mut negative = inputs.negative_embeds.map(|text| BranchRun {
        kind: PromptBranch::Negative,
        text,
        conditioned: references.and_then(|c| c.negative_layout.map(|layout| (c.latents, layout))),
        cache: None,
    });
    // `move`: the closure owns both branches — and with them their caches — and the sampler
    // consumes it, so the caches are dropped when the loop returns, whatever the outcome.
    let predict = move |latents: &Tensor, sigma: f32| -> Result<Tensor> {
        let velocity = positive.evaluate(transformer, cached, latents, sigma, grid, report)?;
        match negative.as_mut() {
            Some(negative) => {
                let uncond =
                    negative.evaluate(transformer, cached, latents, sigma, grid, report)?;
                // `neg_noise_pred + true_cfg_scale * (noise_pred - neg_noise_pred)`
                let diff = ((&velocity - &uncond)? * scale)?;
                Ok((&uncond + diff)?)
            }
            None => Ok(velocity),
        }
    };
    run_flow_sampler(
        inputs.sampler,
        TimestepConvention::Sigma,
        inputs.sigmas,
        inputs.latents,
        inputs.seed,
        inputs.cancel,
        on_progress,
        None,
        predict,
    )
}

/// Alpha-composite an RGBA NCHW tensor in `[-1, 1]` over white → RGB NCHW in `[-1, 1]` (f32).
/// The chosen RGB emission for the current gen-core image surface: a transparent region reads as
/// white, exactly as a viewer would show the RGBA result on a white page, rather than exposing
/// whatever the decoder painted behind the alpha.
pub fn rgba_to_rgb_over_white(rgba: &Tensor) -> Result<Tensor> {
    let dims = rgba.dims().to_vec();
    if dims.len() != 4 || dims[1] != 4 {
        return Err(Error::Msg(format!(
            "qwen_image_2_1: rgba_to_rgb_over_white expects [B, 4, H, W], got {dims:?}"
        )));
    }
    let x = rgba.to_dtype(DType::F32)?;
    let rgb01 = ((x.i((.., 0..3, .., ..))? * 0.5)? + 0.5)?;
    let a01 = ((x.i((.., 3..4, .., ..))? * 0.5)? + 0.5)?;
    // The `1 − α` term is `[B, 1, H, W]` against `[B, 3, H, W]`: it must BROADCAST over the three
    // colour channels, exactly as the alpha multiply above does.
    let out01 = rgb01.broadcast_mul(&a01)?.broadcast_add(&(1.0 - &a01)?)?;
    Ok(((out01 * 2.0)? - 1.0)?)
}

/// Default bounded-decode tile geometry (output pixels) when a request asks for tiling without
/// naming one — the sibling image VAEs' 512 px / 64 px.
pub const DECODE_TILE_EDGE: u32 = 512;
/// Default bounded-decode tile overlap (output pixels).
pub const DECODE_OVERLAP: u32 = 64;

/// The bounded-decode tiling a request selects: `GenerationMemory::tile_vae_decode` with its
/// optional edge/overlap (output pixels), else `None` (single-pass decode).
pub fn decode_tiling(req: &GenerationRequest) -> Option<TilingConfig> {
    req.memory
        .filter(|memory| memory.tile_vae_decode)
        .map(|memory| {
            TilingConfig::spatial_only(
                memory.decode_tile_edge.unwrap_or(DECODE_TILE_EDGE) as i32,
                memory.decode_overlap.unwrap_or(DECODE_OVERLAP) as i32,
            )
        })
}

/// Final packed latents → the VAE's four-channel decode, NCHW `[1, 4, H, W]` in `[-1, 1]`:
/// unpack, denormalise (`z·std + mean`), decode RGBA (tiled when `tiling` is given).
///
/// The single decode both emissions are built from. There is only ever ONE denoise and ONE decode
/// per image: upstream has no transparency flag and always decodes four channels (see
/// `UPSTREAM.md`), so [`decode_rgb`] and [`decode_rgba`] differ **only** in what they do with the
/// alpha the decoder already produced.
fn decode_to_rgba_tensor(
    vae: &QwenImage21Vae,
    latents: &Tensor,
    width: u32,
    height: u32,
    tiling: Option<&TilingConfig>,
    cancel: Option<&CancelFlag>,
) -> Result<Tensor> {
    let unpacked = unpack_latents(latents, width, height)?;
    let vae_space = vae.denormalize(&unpacked)?;
    match tiling {
        Some(cfg) => vae.decode_rgba_tiled(&vae_space, cfg, cancel),
        None => vae.decode_rgba(&vae_space),
    }
}

/// Final packed latents → RGB8 [`Image`]: `decode_to_rgba_tensor`, composite over white,
/// quantise. The `OutputChannels::Rgb` emission (the default), byte-for-byte unchanged by
/// sc-24111.
pub fn decode_rgb(
    vae: &QwenImage21Vae,
    latents: &Tensor,
    width: u32,
    height: u32,
    tiling: Option<&TilingConfig>,
    cancel: Option<&CancelFlag>,
) -> Result<Image> {
    let rgba = decode_to_rgba_tensor(vae, latents, width, height, tiling, cancel)?;
    decoded_to_image(&rgba_to_rgb_over_white(&rgba)?)
}

/// Final packed latents → RGBA8 [`RgbaImage`] with **straight (un-premultiplied)** alpha
/// (sc-24111): `decode_to_rgba_tensor`, quantise. The `OutputChannels::Rgba` emission.
///
/// No compositing anywhere on this path — the alpha the VAE decoded reaches the caller, which is
/// the whole point of the opt-in. Quantisation is upstream's `VaeImageProcessor.postprocess`:
/// `clip(x·0.5 + 0.5, 0, 1)` per channel, `(v·255).round()`, `uint8`, interleaved RGBA (see
/// [`decoded_to_rgba_image`]).
pub fn decode_rgba(
    vae: &QwenImage21Vae,
    latents: &Tensor,
    width: u32,
    height: u32,
    tiling: Option<&TilingConfig>,
    cancel: Option<&CancelFlag>,
) -> Result<RgbaImage> {
    let rgba = decode_to_rgba_tensor(vae, latents, width, height, tiling, cancel)?;
    decoded_to_rgba_image(&rgba)
}

/// RGB NCHW in `[-1, 1]` → the gen-core RGB8 [`Image`], with diffusers-compatible
/// nearest-even rounding ([`candle_gen::round_rgb8`]).
pub fn decoded_to_image(rgb: &Tensor) -> Result<Image> {
    let dims = rgb.dims().to_vec();
    if dims.len() != 4 || dims[1] != 3 {
        return Err(Error::Msg(format!(
            "qwen_image_2_1: decoded_to_image expects [1, 3, H, W], got {dims:?}"
        )));
    }
    let (height, width) = (dims[2], dims[3]);
    let scaled = ((rgb.clamp(-1f32, 1f32)? + 1.0)? * 127.5)?;
    let pixels = candle_gen::round_rgb8(&scaled)?
        .i(0)?
        .to_device(&Device::Cpu)?
        .permute((1, 2, 0))?
        .flatten_all()?
        .to_vec1::<u8>()?;
    Ok(Image {
        width: width as u32,
        height: height as u32,
        pixels,
    })
}

/// RGBA NCHW in `[-1, 1]` → the gen-core RGBA8 [`RgbaImage`] with **straight (un-premultiplied)**
/// alpha (sc-24111) — the four-channel sibling of [`decoded_to_image`], sharing its exact
/// quantisation chain ([`candle_gen::round_rgb8`]'s diffusers-compatible nearest-even rounding).
///
/// This is upstream's `VaeImageProcessor.postprocess` verbatim: `(x / 2 + 0.5).clamp(0, 1)` per
/// channel **independently**, then `(v · 255).round().astype(uint8)` and
/// `PIL.Image.fromarray(..., mode="RGBA")`. Nothing premultiplies, and the alpha is clamped by the
/// same clamp as the colour — so the RGB and the RGBA emission of one decode differ only in
/// whether the alpha is applied (over white) or carried.
pub fn decoded_to_rgba_image(rgba: &Tensor) -> Result<RgbaImage> {
    let dims = rgba.dims().to_vec();
    if dims.len() != 4 || dims[1] != 4 {
        return Err(Error::Msg(format!(
            "qwen_image_2_1: decoded_to_rgba_image expects [1, 4, H, W], got {dims:?}"
        )));
    }
    let (height, width) = (dims[2], dims[3]);
    let scaled = ((rgba.clamp(-1f32, 1f32)? + 1.0)? * 127.5)?;
    let pixels = candle_gen::round_rgb8(&scaled)?
        .i(0)?
        .to_device(&Device::Cpu)?
        .permute((1, 2, 0))?
        .flatten_all()?
        .to_vec1::<u8>()?;
    Ok(RgbaImage {
        width: width as u32,
        height: height as u32,
        pixels,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn pack_unpack_round_trip() {
        let dev = Device::Cpu;
        // A 32x64 request is a 2x4 latent grid; two channels.
        let x = Tensor::from_vec(
            (0..2 * 2 * 4).map(|v| v as f32).collect::<Vec<_>>(),
            (1, 2, 2, 4),
            &dev,
        )
        .unwrap();
        let packed = pack_latents(&x).unwrap();
        assert_eq!(packed.dims(), &[1, 8, 2]);
        // token (row 1, col 2) = index 6, channel 1 → original x[0, 1, 1, 2] = 8 + 4 + 2 = 14.
        assert_eq!(
            packed.i((0, 6, 1)).unwrap().to_scalar::<f32>().unwrap(),
            14.0
        );
        let back = unpack_latents(&packed, 64, 32).unwrap();
        assert_eq!(back.dims(), &[1, 2, 2, 4]);
        assert_eq!(
            back.i((0, 1, 1, 2)).unwrap().to_scalar::<f32>().unwrap(),
            14.0
        );
        assert!(
            unpack_latents(&packed, 64, 64).is_err(),
            "token count must match the grid"
        );
    }

    /// The two production callers of the loop pick opposite sides of the switch: the generator
    /// runs upstream's default (cache on), the trainer's preview render keeps the prefill path its
    /// memory envelope prices. Neither caller surfaces the path it ran, and on this backend the
    /// two paths are bit-identical, so nothing downstream could tell a flipped call site — the
    /// source is the only place to hold it.
    #[test]
    fn the_generator_runs_the_cache_and_the_trainer_preview_sheds_it() {
        let switches = |source: &str| -> Vec<String> {
            source
                .lines()
                .map(str::trim)
                .filter(|line| line.starts_with("kv_cache:"))
                .map(str::to_owned)
                .collect()
        };
        assert_eq!(
            switches(include_str!("lib.rs")),
            ["kv_cache: KvCacheMode::On,"],
            "the generator's one denoise call"
        );
        assert_eq!(
            switches(include_str!("training.rs")),
            ["kv_cache: crate::pipeline::KvCacheMode::Off,"],
            "the trainer's one preview denoise call"
        );
        assert_eq!(KvCacheMode::default(), KvCacheMode::On);
    }

    #[test]
    fn noise_is_seeded_and_shaped() {
        let dev = Device::Cpu;
        let a = create_noise(7, 64, 32, 8, &dev).unwrap();
        let b = create_noise(7, 64, 32, 8, &dev).unwrap();
        let c = create_noise(8, 64, 32, 8, &dev).unwrap();
        assert_eq!(a.dims(), &[1, 8, 8]);
        let host = |t: &Tensor| t.flatten_all().unwrap().to_vec1::<f32>().unwrap();
        assert_eq!(host(&a), host(&b));
        assert_ne!(host(&a), host(&c));
    }

    #[test]
    fn transparent_pixels_composite_to_white() {
        let dev = Device::Cpu;
        // Two pixels: an opaque red and a fully transparent black.
        let rgba = Tensor::from_vec(
            vec![1.0f32, -1.0, -1.0, -1.0, -1.0, -1.0, 1.0, -1.0],
            (1, 4, 1, 2),
            &dev,
        )
        .unwrap();
        let rgb = rgba_to_rgb_over_white(&rgba).unwrap();
        let v = rgb.flatten_all().unwrap().to_vec1::<f32>().unwrap();
        assert_eq!(v, vec![1.0, 1.0, -1.0, 1.0, -1.0, 1.0]);
    }
}
