//! The candle **Lens LoRA/LoKr trainer** (sc-5147) — the candle twin of the worker's Python torch
//! `lens_train_runner.py`, implementing the backend-neutral
//! [`gen_core::Trainer`](candle_gen::gen_core::train::Trainer) with `backend = "candle"`. Together with
//! the inference cutover it retires `/opt/lens-venv` + `INCLUDE_LENS` — the last Python holdout for Lens
//! (epic 3482 / 5164). It reuses the shared [`candle_gen::train`] harness the SDXL/Z-Image/Wan stories
//! established, building on [`crate::dit_train`]'s vendored trainable DiT and [`crate::vae`]'s encode
//! shim.
//!
//! Since sc-7787 the cache → loop → save scaffolding lives in the shared single-model flow-match
//! driver ([`candle_gen::train::flow_match`]); this module supplies the Lens-specific hooks via
//! [`FlowMatchTrainer`] — caching, DiT construction, and the parity-critical `compute_loss_grads`.
//!
//! Registered under `"lens"` — the **non-distilled** `microsoft/Lens` base (the de-distill lesson,
//! sc-1583; a LoRA trained here applies cleanly to `lens_turbo`, same architecture).
//!
//! ## The Lens recipe (from `lens_train_runner.py`)
//!
//! Cache → loop → save, on the **flow-match** objective:
//!  - **Flow-match, no negation.** `x_t = (1−t)·x0 + t·noise`, `target = noise − x0`; the DiT's **raw**
//!    velocity is regressed toward it (Lens feeds the transformer output to the scheduler *without*
//!    negation — opposite of Z-Image). The timestep `t ∈ (0, 1)` is fed to the DiT **directly** (no
//!    `1 − σ`, no `·1000`) — which (with the gradient-checkpoint split below) is why
//!    `compute_loss_grads` stays per-crate rather than collapsing into the shared driver.
//!  - **gpt-oss text front-end, cached + frozen.** Each caption is gpt-oss-encoded and its 4 selected
//!    layers ([`DEFAULT_SELECTED_LAYERS`] = 5/11/17/23) captured + cropped at [`TXT_OFFSET`] (the
//!    harmony-preamble offset) — exactly the inference `encode_one`. Cached once; the encoder is dropped
//!    before the DiT loads.
//!  - **Latents from a neural VAE encode.** Each image is `Flux2Vae`-encoded to the packed DiT latent
//!    `[1, S, 128]` ([`crate::vae::encode`], posterior mean) and cached.
//!  - **Targets:** the fused dual-stream attention projections [`LENS_ATTN_TARGETS`]
//!    (`img_qkv`/`txt_qkv`/`to_out.0`/`to_add_out`); train only the adapter, freeze the gpt-oss encoder
//!    + VAE + DiT base.
//!  - **Save** a diffusers-format `.safetensors` (bare dotted PEFT keys for LoRA / `lokr_w*` + metadata
//!    for LoKr) that the inference merge ([`crate::adapters`]) loads unchanged.
//!
//! The 48-block backward always runs **gradient-checkpointed** (candle's matmul backward materializes a
//! grad for the frozen base weight too, so a dense 48-block backward holds ~48 layers of weight-grads at
//! once — the Wan lesson). Adapter factors / loss / grads / optimizer state stay f32 (master weights);
//! the frozen base + activation stream follow `train_dtype` (bf16 default).

use std::sync::Arc;

use candle_gen::candle_core::backprop::GradStore;
use candle_gen::candle_core::{DType, Device, IndexOp, Tensor, Var};

use candle_gen::gen_core::sampling::TimestepConvention;
use candle_gen::gen_core::train::subject_mask::{CropBox, PreparedSubjectMask};
use candle_gen::gen_core::train::{
    Trainer, TrainerDescriptor, TrainingConfig, TrainingOutput, TrainingProgress, TrainingRequest,
};
use candle_gen::gen_core::{self, CancelFlag, Image, LoadSpec, Modality, Progress, WeightsSource};
use candle_gen::train::dataset::{bucket_edges, decode_square, square_image_tensor};
use candle_gen::train::flow_match::{
    self, combine_terms, prepared_subject_mask_weight, run_flow_match_training, step_terms,
    validate_flow_match_request, weighted_velocity_loss, AuxStep, FlowMatchTrainer, SamplePlan,
    StepLosses, StepSample,
};
use candle_gen::train::gradient_checkpoint::checkpointed_backward;
use candle_gen::train::perceptual::{Parameterization, PerceptualPath};
use candle_gen::train::tae::TinyDecoderSpec;
use candle_gen::{CandleError, Result};
use candle_gen_perceptual::{AuxGeometry, AuxLossContext, DecoderSpec};
use rand::{rngs::StdRng, SeedableRng};

use crate::dit_train::{LensTransformerTrain, LENS_ATTN_TARGETS};
use crate::schedule::{cfg_rescale, lens_mu, lens_sigmas};
use crate::text::{LensTokenizer, TXT_OFFSET};
use crate::text_encoder::{Config as EncoderConfig, GptOssTextEncoder, DEFAULT_SELECTED_LAYERS};
use crate::transformer::LensDitConfig;
use crate::vae::{decode as vae_decode, encode as vae_encode, pack_unpacked_latent, Flux2Vae};
use crate::{DEFAULT_DATE, MODEL_ID_BASE, VAE_SCALE_FACTOR};

/// Per-cadence preview-prompt cap (the shared `SAMPLE_PROMPT_CAP` the sc-8650 contract documents) — at
/// most this many of `cfg.sample_prompts` are pre-encoded + rendered each sample cadence.
const SAMPLE_PROMPT_CAP: usize = 4;

/// Error-message prefix shared by [`validate_flow_match_request`] and the driver's `no usable dataset
/// items` guard.
const LABEL: &str = "lens trainer";

/// gpt-oss is encoded at bf16 for caching (it only produces the cached, frozen features; kept f32 in
/// the cache and dropped before the DiT loads).
const ENC_DTYPE: DType = DType::BF16;

/// One micro-step's forward+backward over the installed adapter `Var`s: build the noised latent at `t`,
/// predict the **raw** velocity through the (LoRA-adapted) DiT, regress it toward `noise − x0` (plus,
/// epic 2123 E8, the planned perceptual term — [`step_loss`]), and return the step's
/// [`StepLosses`] + grads keyed by `lora_vars`. `(h, w)` is the sample's latent grid (per bucket edge);
/// `text_feats` are the cached, frozen gpt-oss features (any dtype — cast to `compute_dtype` here). A
/// free function so the tests can drive it against a tiny DiT.
///
/// `use_checkpoint` selects the **gradient-checkpointed** backward — required at scale, not just a memory
/// lever: candle's matmul backward materializes a gradient for the *frozen* base weight too, so a dense
/// 48-block backward holds ~48 layers of base-weight grads at once. The checkpointed path runs the
/// adapter-free pre-main forward detached, then segments the per-block stack so only one block's
/// transient weight-grads are live at a time (see [`LensTransformerTrain::main_block_segments`]). Both
/// paths yield the same adapter grads (the `dense_and_checkpoint_grads_match` test pins this).
#[allow(clippy::too_many_arguments)]
fn compute_loss_grads(
    dit: &LensTransformerTrain,
    lora_vars: &[Var],
    x0: &Tensor,
    text_feats: &[Tensor],
    h: usize,
    w: usize,
    t: f64,
    noise: &Tensor,
    mae: bool,
    mask_weight: Option<&Tensor>,
    compute_dtype: DType,
    use_checkpoint: bool,
    aux: Option<&AuxStep<'_>>,
) -> Result<(StepLosses, GradStore)> {
    let (x_t_f32, target) = flow_match::build_batch(x0, noise, t)?;
    let x_t = x_t_f32.to_dtype(compute_dtype)?;
    let feats: Vec<Tensor> = text_feats
        .iter()
        .map(|f| f.to_dtype(compute_dtype))
        .collect::<candle_gen::candle_core::Result<_>>()?;
    let timestep = t as f32; // fed to the DiT directly (no 1−σ, no ·1000)

    if use_checkpoint {
        // Pre-main (img/txt embeds, frozen) has no adapters → its `(hidden, encoder)` boundary is a
        // detached constant; the input cotangent is discarded.
        let (hidden, encoder, ctx) = dit.forward_pre_main(&x_t, &feats, None, timestep, 1, h, w)?;
        let hidden_d = hidden.detach();
        let encoder_d = encoder.detach();
        // The final segment's loss breakdown is read back through `parts` (declared before the
        // segments, which borrow it).
        let parts = std::cell::Cell::new(None);
        let mut segs = dit.main_block_segments(&ctx);
        // Final segment: head → raw velocity (NO negation) → flow-match regression (+ the planned
        // perceptual term) → [loss].
        let target_owned = target.clone();
        let ctx_ref = &ctx;
        let (x_t_ref, parts_ref) = (&x_t_f32, &parts);
        segs.push(Box::new(move |st: &[Tensor]| {
            let v = dit.velocity_out(&st[0], ctx_ref)?;
            let (loss, losses) =
                step_loss(&v, &target_owned, x_t_ref, t, h, w, mask_weight, mae, aux)
                    .map_err(|e| candle_gen::candle_core::Error::Msg(e.to_string()))?;
            parts_ref.set(Some(losses));
            Ok(vec![loss])
        }));
        let (_, grads) = checkpointed_backward(&segs, &[hidden_d, encoder_d], lora_vars)?;
        drop(segs);
        let losses = parts.take().ok_or_else(|| {
            CandleError::Msg(format!("{LABEL}: the checkpointed loss segment never ran"))
        })?;
        Ok((losses, grads))
    } else {
        // Dense backward (tiny models / tests only — see the `use_checkpoint` note re: OOM at scale).
        let v = dit.forward(&x_t, &feats, None, timestep, 1, h, w)?;
        let (loss, losses) = step_loss(&v, &target, &x_t_f32, t, h, w, mask_weight, mae, aux)?;
        let grads = loss.backward()?;
        Ok((losses, grads))
    }
}

/// The step's loss on the raw velocity `v` (`[1, S, 128]`): the flow-match regression toward
/// `target` and — epic 2123 E8, when `aux` plans one — the weighted perceptual term on the model's
/// x0 estimate `x_t − t·v` (Lens regresses `noise − x0`), unpacked to TAEF2's
/// `[1, 32, 2h, 2w]` batch-normalized latent ([`unpack_to_decoder_latent`]). On an aux-only step
/// the diffusion term is not computed. `aux = None` is exactly the pre-epic-2123 loss.
#[allow(clippy::too_many_arguments)]
fn step_loss(
    v: &Tensor,
    target: &Tensor,
    x_t_f32: &Tensor,
    t: f64,
    h: usize,
    w: usize,
    mask_weight: Option<&Tensor>,
    mae: bool,
    aux: Option<&AuxStep<'_>>,
) -> Result<(Tensor, StepLosses)> {
    let (diffusion_on, aux_on) = step_terms(aux);
    let diffusion = if diffusion_on {
        Some(weighted_velocity_loss(v, target, mask_weight, mae)?)
    } else {
        None
    };
    let aux_term = match aux {
        Some(a) if aux_on => {
            let x0_hat = Parameterization::FlowNoiseMinusX0 { sigma: t as f32 }
                .recover_x0(x_t_f32, &v.to_dtype(DType::F32)?)?;
            a.aux_loss(&unpack_to_decoder_latent(&x0_hat, h, w)?)?
        }
        _ => None,
    };
    combine_terms(diffusion, aux_term)
}

/// A packed Lens latent `[B, h·w, 128]` (the cache / DiT layout, batch-normalized in the 128-ch
/// patch space) → the `[B, 32, 2h, 2w]` grid TAEF2 decodes: the inverse of
/// [`pack_unpacked_latent`] (un-flatten, then undo the FLUX.2 2×2 patchify — channel
/// `c·4 + ph·2 + pw`). The normalization is kept: TAEF2 consumes the batch-normalized latent
/// unpatchified (its wrapper's latent batch-norm is the identity).
fn unpack_to_decoder_latent(packed: &Tensor, h: usize, w: usize) -> Result<Tensor> {
    let (b, _, c) = packed.dims3()?;
    let c4 = c / 4;
    Ok(packed
        .reshape((b, h, w, c))?
        .permute((0, 3, 1, 2))? // [B, 128, h, w]
        .reshape((b, c4, 2, 2, h, w))?
        .permute((0, 1, 4, 2, 5, 3))? // [B, 32, h, 2, w, 2]
        .reshape((b, c4, h * 2, w * 2))?)
}

/// The subject-mask loss weight for one cached bucket latent (sc-24828 × sc-2127), `None` when the
/// technique is off. The item's mask (loaded once) takes [`decode_square`]'s centre-square crop and
/// is area-averaged onto this bucket's **unpacked** 32-ch
/// VAE latent grid `[1, C, 2·lat_h, 2·lat_w]` (C = 32), then packed through the very 2×2 patchify + flatten
/// the cached latent went through ([`pack_unpacked_latent`]) — so each of the 128 packed channels
/// carries the weight of its own 2×2 sub-position, element-for-element with `x0` `[1, S, 128]`.
fn bucket_mask_weight(
    mask: Option<&PreparedSubjectMask>,
    x0: &Tensor,
    lat_h: usize,
    lat_w: usize,
    device: &Device,
) -> Result<Option<Tensor>> {
    // `x0` is `[1, S, 4·C]` (2×2 patchify folds 4 sub-positions into each channel group).
    let unpacked = [1usize, x0.dim(2)? / 4, 2 * lat_h, 2 * lat_w];
    prepared_subject_mask_weight(LABEL, mask, CropBox::center_square, &unpacked, device)?
        .map(|w| pack_unpacked_latent(&w).map_err(Into::into))
        .transpose()
}

/// gpt-oss-encode `caption` → its 4 captured layers cropped at [`TXT_OFFSET`], each `[1, s, 2880]`
/// (f32, cached). Mirrors the inference `encode_one` (single prompt, unpadded). A caption whose token
/// length is `≤ TXT_OFFSET` (the harmony preamble alone) yields length-0 features — surfaced as an error
/// (an empty caption is a dataset bug, not silently trained on zero text).
fn encode_caption(
    tokenizer: &LensTokenizer,
    encoder: &GptOssTextEncoder,
    caption: &str,
    device: &Device,
) -> Result<Vec<Tensor>> {
    let ids = tokenizer
        .encode(caption, DEFAULT_DATE)
        .map_err(|e| CandleError::Msg(format!("lens trainer: tokenize caption: {e}")))?;
    let l = ids.len();
    if l <= TXT_OFFSET {
        return Err(CandleError::Msg(format!(
            "lens trainer: caption {caption:?} tokenizes to {l} tokens (≤ the {TXT_OFFSET}-token \
             harmony preamble) — it carries no text features"
        )));
    }
    let input_ids = Tensor::from_vec(ids, (1, l), device)?;
    let layers = encoder.capture(&input_ids, &DEFAULT_SELECTED_LAYERS)?;
    let s = l - TXT_OFFSET;
    layers
        .iter()
        .map(|f| {
            Ok(f.narrow(1, TXT_OFFSET, s)?
                .to_dtype(DType::F32)?
                .contiguous()?)
        })
        .collect()
}

/// The joint **CFG** conditioning for one preview prompt (sc-8650), assembled in `LensTrainer::cache`
/// while the gpt-oss encoder is still resident: each feature layer is `[2, S, 2880]` (`[pos; neg]`) and
/// the shared mask is `[2, S]` (`1` = valid). This is the *exact* shape the inference `Pipeline::denoise`
/// feeds the DiT — the trainable [`LensTransformerTrain::forward`] takes the same
/// `(&[Tensor] feats, Some(&mask))` pair, so the preview path is byte-for-byte the inference CFG batch.
struct LensPromptCond {
    /// Per-layer joint features `[2, S, 2880]` (`[pos; neg]`), `num_text_layers` of them.
    features: Vec<Tensor>,
    /// The joint valid mask `[2, S]` (`1` = valid token).
    mask: Tensor,
}

/// The Lens preview-sample render state (sc-8650) — everything the trainer's `render_sample` needs to
/// run the family's **CFG** denoise on the **in-progress** trainable DiT, built once in the trainer's
/// `cache` while the gpt-oss text encoder is still resident:
///
///  * `conds` — the per-prompt joint CFG conditioning (positive + the shared empty-negative
///    unconditional branch), 1:1 with [`SamplePlan::prompts`]. Encoded with the same
///    `encode_caption` the cache loop uses (train/infer conditioning parity), then assembled into the
///    `[2, S, 2880]` / `[2, S]` joint batch exactly like the inference `Pipeline::encode_prompt`.
///  * `vae` — the resident [`Flux2Vae`] **decoder** (`Arc` as inference holds it); the cache pass loads
///    only the encoder, so the decoder is loaded here for the preview path.
///  * `latent_h` / `latent_w` — the packed latent grid (`edge / 16`) the seeded preview noise + the DiT
///    forward are shaped at — the largest training-bucket edge ([`bucket_edges`], just
///    `bucket_resolution(cfg.resolution)` with buckets off — the edge the cached latents use).
pub struct LensSampleState {
    conds: Vec<LensPromptCond>,
    vae: Arc<Flux2Vae>,
    latent_h: usize,
    latent_w: usize,
}

/// Build the joint CFG conditioning for one preview `prompt` from the resident encoder (sc-8650): encode
/// the positive features with [`encode_caption`], take an **empty negative** (the unconditional branch =
/// zero features + an all-zero mask, never a second encode — exactly the inference
/// `Pipeline::encode_prompt` empty-negative path), and `cat([pos; neg], 0)` into the
/// `[2, S, 2880]` / `[2, S]` joint batch.
fn encode_prompt_cond(
    tokenizer: &LensTokenizer,
    encoder: &GptOssTextEncoder,
    prompt: &str,
    device: &Device,
) -> Result<LensPromptCond> {
    let pos_feats = encode_caption(tokenizer, encoder, prompt, device)?;
    let s = pos_feats[0].dim(1)?;
    // Empty negative = the unconditional branch: zero text features + an all-zero mask (no text
    // tokens), padded to the positive length (a single preview prompt is unpadded, so pos == neg == s).
    let pos_mask = Tensor::ones((1, s), DType::F32, device)?;
    let neg_mask = Tensor::zeros((1, s), DType::F32, device)?;
    let mut features = Vec::with_capacity(pos_feats.len());
    for pf in &pos_feats {
        let nf = pf.zeros_like()?;
        features.push(Tensor::cat(&[pf, &nf], 0)?);
    }
    let mask = Tensor::cat(&[&pos_mask, &neg_mask], 0)?;
    Ok(LensPromptCond { features, mask })
}

/// Deterministic packed initial noise `[1, latent_h·latent_w, 128]` for a preview render (sc-8650) — the
/// training-side twin of the inference `create_noise`: N(0,1) from a fixed CPU RNG (launch-portable, NOT
/// candle's CUDA `randn`, sc-3673), then moved to `device`.
fn sample_noise_latent(
    latent_h: usize,
    latent_w: usize,
    seed: u64,
    device: &Device,
) -> Result<Tensor> {
    let seq = latent_h * latent_w;
    let n = seq * 128;
    let mut rng = StdRng::seed_from_u64(seed);
    let data = candle_gen::seeded_normal_vec(&mut rng, n);
    Ok(Tensor::from_vec(data, (1, seq, 128), &Device::Cpu)?.to_device(device)?)
}

/// VAE-decode a final preview latent `[1, h·w, 128]` → RGB8 [`Image`] (sc-8650) — the training-side twin
/// of the inference `to_image` composed with [`crate::vae::decode`] (`(x.clamp(-1,1)+1)·127.5`).
fn decode_preview(vae: &Flux2Vae, lat: &Tensor, latent_h: usize, latent_w: usize) -> Result<Image> {
    let decoded = vae_decode(vae, lat, latent_h, latent_w)?; // [1, 3, H, W] in [-1, 1]
    let scaled = ((decoded.clamp(-1f32, 1f32)? + 1.0)? * 127.5)?;
    let img = candle_gen::round_rgb8(&scaled)?;
    let img = img.i(0)?.to_device(&Device::Cpu)?;
    let (c, h, w) = img.dims3()?;
    if c != 3 {
        return Err(CandleError::Msg(format!(
            "lens: preview decode expected 3 channels, got {c}"
        )));
    }
    let pixels = img.permute((1, 2, 0))?.flatten_all()?.to_vec1::<u8>()?;
    Ok(Image {
        width: w as u32,
        height: h as u32,
        pixels,
    })
}

/// Lens's latent family for the shared aux-loss builder (epic 2123 E8): the FLUX.2 32-channel VAE
/// latent (batch-normalized, unpatchified), decoded by TAEF2. No latent-LPIPS family.
fn aux_loss_context(device: &Device) -> AuxLossContext<'_> {
    AuxLossContext {
        label: LABEL,
        decoder: taef2_decoder(),
        latent_lpips: None,
        device,
    }
}

fn taef2_decoder() -> DecoderSpec {
    DecoderSpec::Tiny {
        name: "TAEF2",
        config: TinyDecoderSpec::taef2(),
    }
}

/// The epic-2123 perceptual path for `cfg`: `None` when no aux loss is enabled (nothing loads).
fn load_perceptual_path(cfg: &TrainingConfig, device: &Device) -> Result<Option<PerceptualPath>> {
    candle_gen_perceptual::build_perceptual_path(cfg, &aux_loss_context(device))
}

/// Extra training memory (bytes) the enabled perceptual losses add for `items` dataset items: TAEF2
/// + each loss at the largest bucket edge, plus one reference per (item, bucket) entry (E7).
fn perceptual_footprint_bytes(cfg: &TrainingConfig, items: usize) -> u64 {
    let edges = bucket_edges(cfg);
    let edge = edges.iter().copied().max().unwrap_or(0);
    candle_gen_perceptual::perceptual_footprint(
        cfg,
        &taef2_decoder(),
        AuxGeometry::image(edge, items * edges.len()),
    )
}

/// Epic 2123 E7 preflight: with a perceptual loss on, the DiT's resident weights (`transformer/`
/// safetensors — the lower bound; no fitted activation model exists for this trainer, so the same
/// base applies checkpointed or dense) plus the aux footprint must fit `budget_bytes`. Depth off ⇒
/// no check.
fn aux_memory_preflight(
    root: &std::path::Path,
    cfg: &TrainingConfig,
    items: usize,
    budget_bytes: u64,
) -> Result<()> {
    let aux = perceptual_footprint_bytes(cfg, items);
    if aux == 0 {
        return Ok(());
    }
    let base = flow_match::component_bytes(root, "transformer", LABEL)?;
    flow_match::check_aux_memory(LABEL, base, aux, budget_bytes)
}

/// Identity + capabilities of the candle Lens trainer: LoRA + LoKr, `backend = "candle"`.
pub fn trainer_descriptor() -> TrainerDescriptor {
    TrainerDescriptor {
        id: MODEL_ID_BASE,
        family: "lens",
        backend: "candle",
        modality: Modality::Image,
        supports_lora: true,
        supports_lokr: true,
        // sc-10894 lockstep catch-up: gen-core gained `TrainerDescriptor.supports_control` (mirrors
        // mlx-gen-lens's `false`).
        supports_control: false,
        // Adapter-only: no full base fine-tune path (sc-14056). The shared
        // `validate_full_finetune_request` floor makes a `full_finetune` request a typed reject.
        supports_full_finetune: false,
        max_reference_images: 0,
        // Epic 2123 S2 (sc-24827): weight noise + gradient noise at the adapter optimizer
        // update.
        // sc-2127 (epic 2123): multi-resolution buckets — one cached latent (+ its own latent grid)
        // per bucket edge, walked by the shared driver's `BucketSchedule`.
        // sc-24828 (epic 2123): subject-masked loss — a per-bucket weight map, packed like that
        // bucket's latent, cached next to it.
        // sc-24830 (epic 2123): depth anchoring through the shared perceptual path (TAEF2 +
        // Depth-Anything-V2) on the checkpointed (production) and dense backwards.
        techniques: gen_core::train::TrainingTechniques {
            resolution_buckets: true,
            subject_mask_loss: true,
            depth_anchoring: true,
            ..gen_core::train::TrainingTechniques::ADAPTER_NOISE
        },
    }
}

/// A loaded candle Lens trainer. Loading is **lazy** — the gpt-oss encoder / VAE / DiT are built inside
/// [`train`](Trainer::train) at the request's compute dtype.
pub struct LensTrainer {
    descriptor: TrainerDescriptor,
    root: std::path::PathBuf,
    device: Device,
}

/// Construct the (lazy) candle Lens trainer from a [`LoadSpec`] whose `weights` is the `microsoft/Lens`
/// snapshot directory (`tokenizer/ text_encoder/ transformer/ vae/`).
pub fn load_trainer(spec: &LoadSpec) -> Result<Box<dyn Trainer>> {
    let root =
        match &spec.weights {
            WeightsSource::Dir(p) => p.clone(),
            WeightsSource::File(_) => return Err(CandleError::Msg(
                "lens trainer expects a snapshot directory (tokenizer/ text_encoder/ transformer/ \
                 vae/), not a single .safetensors file"
                    .into(),
            )),
        };
    Ok(Box::new(LensTrainer {
        descriptor: trainer_descriptor(),
        root,
        device: candle_gen::default_device()?,
    }))
}

// `register_trainer!` defines the explicit trainer constant and bridges the crate's rich `Result`
// into `gen_core::Result` via `Into::into`.
candle_gen::register_trainer! {
    pub(crate) const TRAINER_REGISTRATION = trainer_descriptor => load_trainer
}

impl Trainer for LensTrainer {
    fn descriptor(&self) -> &TrainerDescriptor {
        &self.descriptor
    }

    fn validate(&self, req: &TrainingRequest) -> gen_core::Result<()> {
        // Shared full-base-fine-tune floor (sc-14056): an adapter-only trainer must reject a
        // `full_finetune` request (typed `Unsupported`) rather than silently training a LoRA
        // adapter the caller did not ask for (F-006/F-055).
        gen_core::train::validate_full_finetune_request(self.descriptor(), req)?;
        // Shared training-technique floor (epic 2123 E3): a technique this trainer does not
        // declare (e.g. `weight_noise_sigma > 0`) is a typed refusal, never silently ignored.
        gen_core::train::validate_training_techniques(self.descriptor(), req)?;
        gen_core::train::validate_edit_request(self.descriptor(), req)?;
        validate_flow_match_request(req, LABEL).map_err(Into::into)
    }

    fn train(
        &mut self,
        req: &TrainingRequest,
        on_progress: &mut dyn FnMut(TrainingProgress),
    ) -> gen_core::Result<TrainingOutput> {
        // Epic 2123 E3: refuse an unsupported technique at the `train` entry point too, before
        // any loading/caching — a caller that skips `validate` must not get it silently ignored.
        gen_core::train::validate_training_techniques(self.descriptor(), req)?;
        validate_flow_match_request(req, LABEL)?;
        run_flow_match_training(self, req, on_progress).map_err(Into::into)
    }
}

impl FlowMatchTrainer for LensTrainer {
    type Dit = LensTransformerTrain;
    /// `(x0 packed latent [1, S, 128], the 4 cached gpt-oss feature layers, its latent grid
    /// (lat_h, lat_w), subject-mask loss weight)`, tensors f32. The grid is per entry: with
    /// resolution buckets (sc-2127) each cached latent carries the grid of the bucket edge it was
    /// encoded at. The weight (packed exactly like that entry's `x0`) is `None` unless
    /// subject-masked loss is on (sc-24828).
    type Cached = (Tensor, Vec<Tensor>, (usize, usize), Option<Tensor>);
    type Aux = ();
    /// Preview-sample render state: per-prompt joint CFG conditioning + resident VAE decoder + the
    /// preview latent grid (sc-8650).
    type SampleState = LensSampleState;
    const LABEL: &'static str = LABEL;

    fn device(&self) -> &Device {
        &self.device
    }

    fn default_targets(&self) -> &'static [&'static str] {
        &LENS_ATTN_TARGETS
    }

    /// Epic 2123 E7: with a perceptual loss on, refuse a run whose DiT weights + TAEF2/DA2
    /// footprint exceed the device budget.
    fn preflight(&self, req: &TrainingRequest) -> Result<()> {
        aux_memory_preflight(
            &self.root,
            &req.config,
            req.items.len(),
            flow_match::device_training_budget_bytes(&self.device, LABEL),
        )
    }

    fn perceptual_path(
        &self,
        req: &TrainingRequest,
        device: &Device,
    ) -> Result<Option<PerceptualPath>> {
        load_perceptual_path(&req.config, device)
    }

    /// The cached packed `[1, S, 128]` latent, unpacked to TAEF2's `[1, 32, 2h, 2w]` grid.
    fn reference_latent(&self, cached: &Self::Cached, _aux: &()) -> Result<Tensor> {
        let (x0, _, (h, w), _) = cached;
        unpack_to_decoder_latent(x0, *h, *w)
    }

    fn cache(
        &self,
        req: &TrainingRequest,
        device: &Device,
        on_progress: &mut dyn FnMut(TrainingProgress),
    ) -> Result<(Vec<Self::Cached>, (), SamplePlan<LensSampleState>)> {
        // sc-2127: one training edge per resolution bucket (just `[resolution]` when buckets are off);
        // previews render at the largest (epic 2123 E7).
        let edges = bucket_edges(&req.config);
        let edge = edges.iter().copied().max().unwrap_or(0);
        let tokenizer =
            LensTokenizer::from_file(self.root.join("tokenizer").join("tokenizer.json"))?;
        // gpt-oss is the caching workhorse (dense bf16, ~40 GB transient) — built then dropped.
        let encoder = GptOssTextEncoder::new(
            &EncoderConfig::gpt_oss_20b(),
            flow_match::component_vb(&self.root, "text_encoder", device, ENC_DTYPE, LABEL)?,
        )?;
        let vae = Flux2Vae::new_with_encoder(flow_match::component_vb(
            &self.root,
            "vae",
            device,
            DType::F32,
            LABEL,
        )?)?;

        let total = req.items.len() as u32;
        // Item-major over the bucket edges: `cache[item * edges.len() + bucket]` — the layout the
        // driver's `BucketSchedule` indexes (sc-2127).
        let mut cache: Vec<Self::Cached> = Vec::with_capacity(req.items.len() * edges.len());
        for (i, item) in req.items.iter().enumerate() {
            if req.cancel.is_cancelled() {
                break;
            }
            on_progress(TrainingProgress::Caching {
                current: i as u32 + 1,
                total,
            });
            let feats = encode_caption(&tokenizer, &encoder, &item.caption, device)?;
            // sc-24828: the item's subject mask is read + checked once, then resampled per bucket
            // onto that bucket's latent grid (`None` when masked loss is off).
            let mask = PreparedSubjectMask::load_if_enabled(
                LABEL,
                item,
                req.config.subject_mask_loss.as_ref(),
            )?;
            let square = decode_square(&item.image_path)?; // decoded once, resized per bucket edge
            for &edge in &edges {
                let img = square_image_tensor(&square, edge, device)?; // [1,3,edge,edge] in [-1,1]
                let (x0, lh, lw) = vae_encode(&vae, &img)?; // [1, S, 128] packed latent (mean), f32
                let mask_weight = bucket_mask_weight(mask.as_ref(), &x0, lh, lw, device)?;
                cache.push((x0, feats.clone(), (lh, lw), mask_weight));
            }
        }

        // Preview samples (sc-8650) — while the gpt-oss encoder is STILL resident, pre-encode up to
        // `SAMPLE_PROMPT_CAP` of the configured prompts into the joint CFG batch (positive + the shared
        // empty-negative unconditional branch), using the same `encode_caption` the cache loop uses
        // (train/infer conditioning parity), and load a resident `Flux2Vae` *decoder* (the cache pass
        // built an encoder-only VAE). The previews render at the largest bucket edge (`edge / 16`
        // packed grid; the single cached edge when buckets are off). The driver renders these from the
        // in-progress adapter each cadence.
        let sample_plan = if req.config.sample_every > 0 && !req.config.sample_prompts.is_empty() {
            let lat = (edge / VAE_SCALE_FACTOR) as usize;
            let prompts: Vec<String> = req
                .config
                .sample_prompts
                .iter()
                .take(SAMPLE_PROMPT_CAP)
                .cloned()
                .collect();
            let conds = prompts
                .iter()
                .map(|p| encode_prompt_cond(&tokenizer, &encoder, p, device))
                .collect::<Result<Vec<LensPromptCond>>>()?;
            // The cache VAE is encoder-only; load a fresh decoder-capable `Flux2Vae` for the preview
            // decode path (f32, matching inference).
            let vae_decoder = Flux2Vae::new(flow_match::component_vb(
                &self.root,
                "vae",
                device,
                DType::F32,
                LABEL,
            )?)?;
            SamplePlan {
                prompts,
                state: Some(LensSampleState {
                    conds,
                    vae: Arc::new(vae_decoder),
                    latent_h: lat,
                    latent_w: lat,
                }),
            }
        } else {
            SamplePlan::disabled()
        };

        // The encoders are dead weight once cached + previews pre-encoded — drop them before the DiT
        // (working set) loads. The resident VAE *decoder* lives on in the sample plan's state.
        drop(encoder);
        drop(vae);
        Ok((cache, (), sample_plan))
    }

    fn build_dit(&self, req: &TrainingRequest, device: &Device) -> Result<LensTransformerTrain> {
        let compute_dtype = flow_match::parse_compute_dtype(&req.config.train_dtype);
        Ok(LensTransformerTrain::new(
            &LensDitConfig::lens(),
            flow_match::component_vb(&self.root, "transformer", device, compute_dtype, LABEL)?,
        )?)
    }

    fn micro_step(
        &self,
        dit: &LensTransformerTrain,
        vars: &[Var],
        cached: &Self::Cached,
        _aux: &(),
        cfg: &TrainingConfig,
        step: u32,
        sample: StepSample<'_>,
        device: &Device,
    ) -> Result<(f32, GradStore)> {
        // The grid of the bucket this entry was encoded at (sc-2127).
        let (x0, feats, (lat_h, lat_w), mask_weight) = cached;
        let (lat_h, lat_w) = (*lat_h, *lat_w);
        // Lens feeds `t` to the DiT directly (cast to f64), and the 48-block backward always uses the
        // gradient-checkpointed path.
        let mut t = flow_match::sample_unit_timestep(
            &cfg.timestep_type,
            &cfg.timestep_bias,
            flow_match::timestep_seed(cfg.seed, step),
        ) as f64;
        let noise =
            flow_match::sample_noise(x0.dims(), flow_match::noise_seed(cfg.seed, step), device)?;
        // Epic 2123 E8: plan on the sampled `t` (Lens's flow-match noise level, 1 = pure noise); an
        // aux-only step trains at `t` remapped into the loss window.
        let aux = sample.plan(t as f32)?;
        if let Some(a) = &aux {
            t = a.noise_level() as f64;
        }
        let (losses, grads) = compute_loss_grads(
            dit,
            vars,
            x0,
            feats,
            lat_h,
            lat_w,
            t,
            &noise,
            flow_match::is_mae(cfg),
            mask_weight.as_ref(),
            flow_match::parse_compute_dtype(&cfg.train_dtype),
            true,
            aux.as_ref(),
        )?;
        Ok((losses.total, grads))
    }

    /// Render preview prompt `index` from the **in-progress** trainable DiT (sc-8650) — the training-side
    /// mirror of the inference `Pipeline::render`/`Pipeline::denoise`. Lens is a **standard-guidance
    /// (CFG) family**, so the denoise closure runs the joint `[2, …]` batch (`cat([x, x], 0)`, the
    /// cond/uncond branches share `x_t`), one [`LensTransformerTrain::forward`] over the pre-encoded
    /// `[2, S, 2880]` features + `[2, S]` mask, and norm-rescaled [`cfg_rescale`] at
    /// `cfg.sample_guidance_scale`. Lens consumes the **raw** velocity at the *shifted sigma* timestep
    /// ([`TimestepConvention::Sigma`] — the σ is fed to the DiT directly), so the closure passes the bare
    /// σ as `timestep` and returns the guided velocity (no negation). `TrainingConfig` carries no per-run
    /// sampler/scheduler knob, so the native empirical-μ flow-match schedule is used (the byte-exact
    /// inference default — `None` sampler/scheduler resolve to euler over the native sigmas). Best-effort:
    /// any error here is logged + skipped by the driver, never aborting the run.
    fn render_sample(
        &self,
        dit: &LensTransformerTrain,
        state: &LensSampleState,
        index: usize,
        cfg: &TrainingConfig,
        seed: u64,
    ) -> Result<Image> {
        let device = &self.device;
        let steps = (cfg.sample_steps.max(1)) as usize;
        let guidance = cfg.sample_guidance_scale;
        let (latent_h, latent_w) = (state.latent_h, state.latent_w);
        let cond = &state.conds[index];

        // Native empirical-μ flow-match sigmas — the byte-exact inference default; `None` scheduler (no
        // per-run knob on `TrainingConfig`) resolves to the native schedule, `None` sampler to euler.
        let mu = lens_mu(steps, latent_h, latent_w);
        let native = lens_sigmas(steps, latent_h, latent_w);
        let sigmas = candle_gen::resolve_flow_schedule(None, mu, steps, &native);

        // Run the denoise loop in the DiT's compute dtype, exactly like inference (`init.to_dtype(
        // DIT_DTYPE)` then a bf16 loop) — the sampler's `x + v·dt` update needs the latent + the
        // closure's velocity to share a dtype, so the closure returns the velocity in the loop dtype
        // (no F32 cast) just as `Pipeline::denoise` does.
        let loop_dtype = flow_match::parse_compute_dtype(&cfg.train_dtype);
        let noise = sample_noise_latent(latent_h, latent_w, seed, device)?.to_dtype(loop_dtype)?;
        // The joint CFG features are cached f32 (portable); the DiT forward (like the train micro-step,
        // which casts feats to `compute_dtype`) needs them in the loop dtype — cast once up front.
        let feats: Vec<Tensor> = cond
            .features
            .iter()
            .map(|f| f.to_dtype(loop_dtype))
            .collect::<candle_gen::candle_core::Result<_>>()?;
        let mask = &cond.mask;

        // A preview need not honor cancel mid-denoise — a fresh never-cancel flag (the trainer's
        // `req.cancel` is only available in `cache`, not here).
        let cancel = CancelFlag::new();
        let mut on_progress = |_: Progress| {};
        // Deliberately NO per-step latent preview hook (epic 16948, sc-16955): this is the trainer's
        // periodic sample render, driven from a synthetic request that carries no PreviewSink, and its
        // result is delivered as a finished `TrainingProgress::Sample` image rather than as a live
        // denoise stream. `candle-gen-catalog`'s route inventory pins this exact site as dark with
        // that reason — the same decision sc-16950 recorded for Krea's trainer and sc-16954 for SDXL's.
        let lat = candle_gen::run_flow_sampler(
            None,
            TimestepConvention::Sigma,
            &sigmas,
            noise,
            seed,
            &cancel,
            &mut on_progress,
            None,
            |latents, sigma| -> Result<Tensor> {
                // Joint CFG batch: duplicate the latent (cond/uncond share x_t), one DiT call over the
                // pre-encoded `[2, S, 2880]` features + `[2, S]` mask (frame = 1).
                let hidden = Tensor::cat(&[latents, latents], 0)?; // [2, seq, 128]
                let velocity =
                    dit.forward(&hidden, &feats, Some(mask), sigma, 1, latent_h, latent_w)?;
                let pos = velocity.narrow(0, 0, 1)?;
                let neg = velocity.narrow(0, 1, 1)?;
                // `cfg_rescale` preserves the input dtype, so the guided velocity matches the loop
                // latent — no cast (mirrors `Pipeline::denoise`, which returns it directly).
                Ok(cfg_rescale(&pos, &neg, guidance)?)
            },
        )?;
        // The denoise ran in the bf16 loop dtype; the resident VAE is F32 → cast back before decode.
        decode_preview(&state.vae, &lat.to_dtype(DType::F32)?, latent_h, latent_w)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use candle_gen::candle_nn::{VarBuilder, VarMap};
    use candle_gen::train::lora::build_lora_targets;
    use candle_gen::train::optim::{clip_grad_norm, TrainOptimizer};

    /// The pre-epic-2123 call shape (no perceptual step) for the legacy tests.
    #[allow(clippy::too_many_arguments)]
    fn compute_loss_grads(
        dit: &LensTransformerTrain,
        lora_vars: &[Var],
        x0: &Tensor,
        text_feats: &[Tensor],
        h: usize,
        w: usize,
        t: f64,
        noise: &Tensor,
        mae: bool,
        mask_weight: Option<&Tensor>,
        compute_dtype: DType,
        use_checkpoint: bool,
    ) -> Result<(f32, GradStore)> {
        super::compute_loss_grads(
            dit,
            lora_vars,
            x0,
            text_feats,
            h,
            w,
            t,
            noise,
            mae,
            mask_weight,
            compute_dtype,
            use_checkpoint,
            None,
        )
        .map(|(l, g)| (l.total, g))
    }

    /// A tiny Lens-shaped DiT config (2 layers, 2 heads × 8, 1 text layer) — exercises the real
    /// flow-match forward+backward on CPU. Mirrors `dit_train`'s tiny cfg (Σ axes = head_dim).
    fn tiny_cfg() -> LensDitConfig {
        LensDitConfig {
            patch_size: 2,
            in_channels: 32,
            out_channels: 8,
            num_layers: 2,
            num_heads: 2,
            head_dim: 8,
            inner_dim: 16,
            enc_hidden_dim: 12,
            num_text_layers: 1,
            timestep_channels: 16,
            axes_dims_rope: [2, 2, 4],
            rope_theta: 10_000.0,
        }
    }

    /// Randomize every var in a fresh `VarMap` — a zero patch/img_in weight makes `hidden ≡ 0` and the
    /// adapter grads vacuously zero; real training loads nonzero weights, so the tiny tests must too.
    fn randomize_base(vm: &VarMap, dev: &Device) {
        for v in vm.all_vars() {
            v.set(&Tensor::randn(0f32, 0.1f32, v.dims(), dev).unwrap())
                .unwrap();
        }
    }

    /// Tiny synthetic inputs: a packed latent `[1, h·w, in_channels]`, one text-feature layer, noise,
    /// and the latent grid `(h, w)`.
    fn tiny_inputs(
        cfg: &LensDitConfig,
        dev: &Device,
    ) -> (Tensor, Vec<Tensor>, Tensor, usize, usize) {
        let (h, w) = (2usize, 2usize);
        let img_len = h * w;
        let x0 = Tensor::randn(0f32, 1f32, (1, img_len, cfg.in_channels), dev).unwrap();
        let feat = Tensor::randn(0f32, 1f32, (1, 3, cfg.enc_hidden_dim), dev).unwrap();
        let noise = Tensor::randn(0f32, 1f32, (1, img_len, cfg.in_channels), dev).unwrap();
        (x0, vec![feat], noise, h, w)
    }

    /// The keystone training gate: a real flow-match forward+backward over the tiny DiT with nonzero
    /// LoRA factors yields a finite loss and a gradient on **every** adapter `Var` (save the last block's
    /// `to_add_out`, whose text-stream output the image-velocity head discards — see `dit_train`).
    #[test]
    fn backward_reaches_lora_factors() {
        let dev = Device::Cpu;
        let cfg = tiny_cfg();
        let vm = VarMap::new();
        let vb = VarBuilder::from_varmap(&vm, DType::F32, &dev);
        let mut dit = LensTransformerTrain::new(&cfg, vb).unwrap();
        randomize_base(&vm, &dev);
        let suffixes: Vec<String> = LENS_ATTN_TARGETS.iter().map(|s| s.to_string()).collect();
        let set = build_lora_targets(&mut dit, &suffixes, 4, 8.0, 7, &dev).unwrap();
        // Move B off its zero-init so both A and B grads are nonzero.
        for v in &set.vars {
            v.set(&Tensor::randn(0f32, 0.02f32, v.as_tensor().dims(), &dev).unwrap())
                .unwrap();
        }
        let (x0, feats, noise, h, w) = tiny_inputs(&cfg, &dev);
        let (loss, grads) = compute_loss_grads(
            &dit,
            &set.vars,
            &x0,
            &feats,
            h,
            w,
            0.5,
            &noise,
            false,
            None,
            DType::F32,
            false,
        )
        .unwrap();
        assert!(loss.is_finite(), "loss must be finite, got {loss}");
        let mut saw_nonzero = false;
        for v in &set.vars {
            if let Some(g) = grads.get(v.as_tensor()) {
                let gv = g.flatten_all().unwrap().to_vec1::<f32>().unwrap();
                assert!(gv.iter().all(|x| x.is_finite()), "non-finite gradient");
                if gv.iter().any(|x| x.abs() > 1e-9) {
                    saw_nonzero = true;
                }
            }
        }
        assert!(saw_nonzero, "backprop is not reaching the adapter factors");
        assert_eq!(set.vars.len(), 4 * 2 * cfg.num_layers); // 4 projections × 2 factors × layers
    }

    /// The correctness gate for the gradient-checkpointed backward (the path real training always uses):
    /// it must reproduce the dense `loss.backward()` grads (mod float reassociation) on the tiny DiT.
    #[test]
    fn dense_and_checkpoint_grads_match() {
        let dev = Device::Cpu;
        let cfg = tiny_cfg();
        let vm = VarMap::new();
        let vb = VarBuilder::from_varmap(&vm, DType::F32, &dev);
        let mut dit = LensTransformerTrain::new(&cfg, vb).unwrap();
        randomize_base(&vm, &dev);
        let suffixes: Vec<String> = LENS_ATTN_TARGETS.iter().map(|s| s.to_string()).collect();
        let set = build_lora_targets(&mut dit, &suffixes, 4, 8.0, 7, &dev).unwrap();
        for v in &set.vars {
            v.set(&Tensor::randn(0f32, 0.02f32, v.as_tensor().dims(), &dev).unwrap())
                .unwrap();
        }
        let (x0, feats, noise, h, w) = tiny_inputs(&cfg, &dev);
        let (loss_d, g_d) = compute_loss_grads(
            &dit,
            &set.vars,
            &x0,
            &feats,
            h,
            w,
            0.5,
            &noise,
            false,
            None,
            DType::F32,
            false,
        )
        .unwrap();
        let (loss_c, g_c) = compute_loss_grads(
            &dit,
            &set.vars,
            &x0,
            &feats,
            h,
            w,
            0.5,
            &noise,
            false,
            None,
            DType::F32,
            true,
        )
        .unwrap();
        assert!(
            (loss_d - loss_c).abs() < 1e-4,
            "loss: dense {loss_d} vs checkpoint {loss_c}"
        );
        let mut saw_nonzero = false;
        for (i, v) in set.vars.iter().enumerate() {
            // A var with no dense grad (the discarded last-block to_add_out) is skipped in both paths.
            let (Some(a), Some(b)) = (g_d.get(v.as_tensor()), g_c.get(v.as_tensor())) else {
                continue;
            };
            let a = a.flatten_all().unwrap().to_vec1::<f32>().unwrap();
            let b = b.flatten_all().unwrap().to_vec1::<f32>().unwrap();
            assert_eq!(a.len(), b.len());
            for (x, y) in a.iter().zip(b.iter()) {
                assert!(
                    (x - y).abs() < 1e-4,
                    "grad mismatch for var {i} (dense {x} vs checkpoint {y})"
                );
                if x.abs() > 1e-6 {
                    saw_nonzero = true;
                }
            }
        }
        assert!(saw_nonzero, "expected nonzero adapter grads to compare");
    }

    /// sc-24828: subject-masked loss on both backward paths. The weight is built on the unpacked
    /// `[1, 8, 4, 4]` latent grid and packed to `[1, 4, 32]` exactly like `x0`. An all-ones map is the
    /// unweighted loss; an all-zero map zeroes the loss AND every adapter gradient (dense and
    /// checkpointed); a half map lands between and matches across paths.
    #[test]
    fn subject_mask_weight_reaches_both_backward_paths() {
        let dev = Device::Cpu;
        let cfg = tiny_cfg();
        let vm = VarMap::new();
        let vb = VarBuilder::from_varmap(&vm, DType::F32, &dev);
        let mut dit = LensTransformerTrain::new(&cfg, vb).unwrap();
        randomize_base(&vm, &dev);
        let suffixes: Vec<String> = LENS_ATTN_TARGETS.iter().map(|s| s.to_string()).collect();
        let set = build_lora_targets(&mut dit, &suffixes, 4, 8.0, 7, &dev).unwrap();
        for v in &set.vars {
            v.set(&Tensor::randn(0f32, 0.02f32, v.as_tensor().dims(), &dev).unwrap())
                .unwrap();
        }
        let (x0, feats, noise, h, w) = tiny_inputs(&cfg, &dev);
        let unpacked = [1usize, cfg.in_channels / 4, 2 * h, 2 * w];
        let map = |m: &[f32]| {
            let u = flow_match::subject_mask_weight(m, 2 * h, 2 * w, &unpacked, &dev).unwrap();
            let packed = pack_unpacked_latent(&u).unwrap();
            assert_eq!(packed.dims(), x0.dims());
            packed
        };
        let run = |weight: Option<&Tensor>, ckpt: bool| {
            compute_loss_grads(
                &dit,
                &set.vars,
                &x0,
                &feats,
                h,
                w,
                0.5,
                &noise,
                false,
                weight,
                DType::F32,
                ckpt,
            )
            .unwrap()
        };
        let (plain, _) = run(None, false);
        let ones = map(&[1.0; 16]);
        assert!((run(Some(&ones), false).0 - plain).abs() < 1e-6);
        let zeros = map(&[0.0; 16]);
        for ckpt in [false, true] {
            let (loss, grads) = run(Some(&zeros), ckpt);
            assert_eq!(
                loss, 0.0,
                "ckpt={ckpt}: an all-background map must zero the loss"
            );
            for v in &set.vars {
                if let Some(g) = grads.get(v.as_tensor()) {
                    let g = g.flatten_all().unwrap().to_vec1::<f32>().unwrap();
                    assert!(
                        g.iter().all(|x| *x == 0.0),
                        "ckpt={ckpt}: nonzero adapter grad"
                    );
                }
            }
        }
        let half: Vec<f32> = (0..16).map(|i| if i % 4 < 2 { 1.0 } else { 0.0 }).collect();
        let half = map(&half);
        let (dense, _) = run(Some(&half), false);
        let (ckpt, _) = run(Some(&half), true);
        assert!(dense > 0.0 && dense < plain, "{dense} vs {plain}");
        assert!(
            (dense - ckpt).abs() < 1e-4,
            "dense {dense} vs checkpoint {ckpt}"
        );
    }

    /// sc-24828: the packed weight lines up with the packed latent. A weight whose value encodes its
    /// unpacked `(y, x)` position lands at token `(y/2)·w + x/2`, channel `c·4 + (y%2)·2 + x%2` — the
    /// FLUX.2 2×2 patchify + flatten order `vae::encode` gives `x0`.
    #[test]
    fn packed_subject_mask_weight_lines_up_with_packed_latent() {
        let dev = Device::Cpu;
        let (h, w, c) = (2usize, 3usize, 2usize); // packed grid h×w; unpacked 4×6, c channels
        let vals: Vec<f32> = (0..2 * h * 2 * w)
            .map(|i| ((i / (2 * w)) * 10 + i % (2 * w)) as f32)
            .collect();
        let u = flow_match::subject_mask_weight(&vals, 2 * h, 2 * w, &[1, c, 2 * h, 2 * w], &dev)
            .unwrap();
        let p = pack_unpacked_latent(&u).unwrap();
        assert_eq!(p.dims(), &[1, h * w, 4 * c]);
        let p = p.squeeze(0).unwrap().to_vec2::<f32>().unwrap();
        for (tok, row) in p.iter().enumerate() {
            let (i, j) = (tok / w, tok % w);
            for (ch, v) in row.iter().enumerate() {
                let (ph, pw) = ((ch % 4) / 2, ch % 2);
                let (y, x) = (2 * i + ph, 2 * j + pw);
                assert_eq!(*v, (y * 10 + x) as f32, "token {tok} channel {ch}");
            }
        }
    }

    /// sc-24828 × sc-2127: with two resolution buckets and masked loss on, the item's mask (loaded
    /// once) yields a weight per bucket that is packed exactly like that bucket's latent — same
    /// `[1, S, 4·C]` shape — and the masked-out region (the right half of the centre crop) is zero.
    #[test]
    fn bucket_mask_weight_follows_each_bucket_latent() {
        use candle_gen::gen_core::train::TrainingItem;
        let dev = Device::Cpu;
        let dir = tempfile::tempdir().unwrap();
        // A 48x32 landscape image: centre crop x in [8, 40); subject = crop's left half [8, 24).
        let img = dir.path().join("img.png");
        image::RgbImage::new(48, 32).save(&img).unwrap();
        let mask_path = dir.path().join("mask.png");
        image::GrayImage::from_fn(48, 32, |x, _| {
            image::Luma([if (8..24).contains(&x) { 255 } else { 0 }])
        })
        .save(&mask_path)
        .unwrap();
        let mut item = TrainingItem::captioned(img, "c".into());
        item.subject_mask_path = Some(mask_path);
        let on = candle_gen::gen_core::SubjectMaskLoss {
            background_weight: 0.0,
            subject_weight: 1.0,
        };
        let mask = PreparedSubjectMask::load_if_enabled(LABEL, &item, Some(&on))
            .unwrap()
            .expect("masked loss on");
        let c = 2usize; // unpacked channels; packed = 4·c
        for (lh, lw) in [(2usize, 2usize), (4, 4)] {
            let x0 = Tensor::zeros((1, lh * lw, 4 * c), DType::F32, &dev).unwrap();
            let w = bucket_mask_weight(Some(&mask), &x0, lh, lw, &dev)
                .unwrap()
                .expect("weight");
            assert_eq!(w.dims(), x0.dims(), "bucket {lh}x{lw}");
            let rows = w.squeeze(0).unwrap().to_vec2::<f32>().unwrap();
            for (tok, row) in rows.iter().enumerate() {
                let j = tok % lw;
                for (ch, v) in row.iter().enumerate() {
                    let x = 2 * j + ch % 2; // unpacked column of this packed element
                    let expected = if x < lw { 1.0 } else { 0.0 };
                    assert_eq!(*v, expected, "bucket {lh}x{lw} token {tok} channel {ch}");
                }
            }
        }
        let x0 = Tensor::zeros((1, 4, 4 * c), DType::F32, &dev).unwrap();
        assert!(bucket_mask_weight(None, &x0, 2, 2, &dev).unwrap().is_none());
    }

    /// The candle Lens trainer declares subject-masked loss (sc-24828).
    #[test]
    fn descriptor_declares_subject_mask_loss() {
        assert!(trainer_descriptor().techniques.subject_mask_loss);
    }

    /// A few optimizer steps on a fixed batch lower the loss — the step descends the flow-match
    /// objective end to end through the harness.
    #[test]
    fn one_optimizer_step_descends() {
        let dev = Device::Cpu;
        let cfg = tiny_cfg();
        let vm = VarMap::new();
        let vb = VarBuilder::from_varmap(&vm, DType::F32, &dev);
        let mut dit = LensTransformerTrain::new(&cfg, vb).unwrap();
        randomize_base(&vm, &dev);
        let suffixes: Vec<String> = LENS_ATTN_TARGETS.iter().map(|s| s.to_string()).collect();
        let set = build_lora_targets(&mut dit, &suffixes, 4, 8.0, 7, &dev).unwrap();
        for v in &set.vars {
            v.set(&Tensor::randn(0f32, 0.02f32, v.as_tensor().dims(), &dev).unwrap())
                .unwrap();
        }
        let (x0, feats, noise, h, w) = tiny_inputs(&cfg, &dev);
        let mut opt = TrainOptimizer::from_config("adamw", set.vars.clone(), 1e-2, 0.0).unwrap();
        let loss_at = |dit: &LensTransformerTrain| {
            compute_loss_grads(
                dit,
                &set.vars,
                &x0,
                &feats,
                h,
                w,
                0.5,
                &noise,
                false,
                None,
                DType::F32,
                false,
            )
            .unwrap()
        };
        let (loss0, mut grads) = loss_at(&dit);
        for _ in 0..5 {
            clip_grad_norm(&mut grads, &set.vars, 1.0).unwrap();
            opt.step(&grads).unwrap();
            grads = loss_at(&dit).1;
        }
        let (loss1, _) = loss_at(&dit);
        assert!(
            loss1 < loss0,
            "5 steps on a fixed batch should lower the loss: {loss0} -> {loss1}"
        );
    }

    /// The trainer resolves through the explicit family registry as the candle Lens
    /// trainer; `load_trainer` is lazy, so a nonexistent weights dir still resolves.
    #[test]
    fn trainer_registers_and_resolves_as_candle() {
        let spec = LoadSpec::new(WeightsSource::Dir("/nonexistent".into()));
        let t = crate::provider_registry()
            .unwrap()
            .load_trainer(MODEL_ID_BASE, &spec)
            .expect("candle lens trainer is registered");
        assert_eq!(t.descriptor().id, MODEL_ID_BASE);
        assert_eq!(t.descriptor().backend, "candle");
        assert_eq!(t.descriptor().modality, Modality::Image);
        assert!(t.descriptor().supports_lora && t.descriptor().supports_lokr);
        assert!(t.descriptor().techniques.resolution_buckets);
    }

    /// sc-2127: the micro-step reads each cached entry's OWN latent grid, so one run trains across
    /// bucket entries of different grids (here a 2×2 and a 4×2 packed grid) through the same DiT.
    #[test]
    fn micro_step_uses_each_entrys_grid() {
        let dev = Device::Cpu;
        let cfg = tiny_cfg();
        let vm = VarMap::new();
        let vb = VarBuilder::from_varmap(&vm, DType::F32, &dev);
        let mut dit = LensTransformerTrain::new(&cfg, vb).unwrap();
        randomize_base(&vm, &dev);
        let suffixes: Vec<String> = LENS_ATTN_TARGETS.iter().map(|s| s.to_string()).collect();
        let set = build_lora_targets(&mut dit, &suffixes, 4, 8.0, 7, &dev).unwrap();
        let trainer = LensTrainer {
            descriptor: trainer_descriptor(),
            root: "/nonexistent".into(),
            device: dev.clone(),
        };
        let feat = Tensor::randn(0f32, 1f32, (1, 3, cfg.enc_hidden_dim), &dev).unwrap();
        let train_cfg = TrainingConfig {
            train_dtype: "f32".into(),
            ..TrainingConfig::default()
        };
        for (step, (h, w)) in [(2usize, 2usize), (4, 2)].into_iter().enumerate() {
            let x0 = Tensor::randn(0f32, 1f32, (1, h * w, cfg.in_channels), &dev).unwrap();
            let cached = (x0, vec![feat.clone()], (h, w), None);
            let (loss, _) = trainer
                .micro_step(
                    &dit,
                    &set.vars,
                    &cached,
                    &(),
                    &train_cfg,
                    step as u32 + 1,
                    flow_match::StepSample::plain(0, 0),
                    &dev,
                )
                .unwrap();
            assert!(
                loss.is_finite(),
                "grid {h}x{w}: loss must be finite, got {loss}"
            );
        }
    }

    /// `validate` rejects an empty dataset, zero rank/steps, an unsupported optimizer, and unrecognized
    /// timestep/loss knobs — before any load (now via the shared `flow_match::validate_flow_match_request`).
    #[test]
    fn validate_rejects_bad_requests() {
        use candle_gen::gen_core::runtime::CancelFlag;
        use candle_gen::gen_core::train::TrainingItem;
        let spec = LoadSpec::new(WeightsSource::Dir("/nonexistent".into()));
        let t = crate::provider_registry()
            .unwrap()
            .load_trainer(MODEL_ID_BASE, &spec)
            .unwrap();
        let base = TrainingRequest {
            items: vec![TrainingItem {
                image_path: "/img.png".into(),
                caption: "x".into(),
                control_image_path: None,
                model_options: Default::default(),
                reference_image_paths: Vec::new(),
                subject_mask_path: None,
            }],
            config: TrainingConfig::default(),
            output_dir: "/out".into(),
            file_name: "a.safetensors".into(),
            trigger_words: vec![],
            cancel: CancelFlag::new(),
        };
        assert!(t.validate(&base).is_ok());
        let bad = |mutate: &dyn Fn(&mut TrainingRequest)| {
            let mut r = base.clone();
            mutate(&mut r);
            assert!(t.validate(&r).is_err());
        };
        bad(&|r| r.items.clear());
        bad(&|r| r.config.rank = 0);
        bad(&|r| r.config.steps = 0);
        bad(&|r| r.config.optimizer = "lion".into());
        bad(&|r| r.config.timestep_type = "bogus".into());
        bad(&|r| r.config.loss_type = "huber".into());
    }

    /// Epic 2123 depth anchoring (sc-24830) on the candle Lens trainer: the tiny DiT (packed 32 =
    /// 4 × 8-channel latent) + a random-init tiny 8-channel decoder + tiny DA2, through the real step.
    mod depth_anchoring {
        use super::*;
        use candle_gen::gen_core::train::{AuxLossSchedule, DepthModelSize};
        use candle_gen::train::flow_match::AuxDriver;

        fn schedule() -> AuxLossSchedule {
            AuxLossSchedule {
                weight: 0.5,
                t_min: 0.6,
                t_max: 0.9,
                every_n: 2,
            }
        }

        struct Fixture {
            dit: LensTransformerTrain,
            vars: Vec<Var>,
            x0: Tensor,
            feats: Vec<Tensor>,
            noise: Tensor,
            h: usize,
            w: usize,
        }

        fn fixture() -> Fixture {
            let dev = Device::Cpu;
            let cfg = tiny_cfg();
            let vm = VarMap::new();
            let vb = VarBuilder::from_varmap(&vm, DType::F32, &dev);
            let mut dit = LensTransformerTrain::new(&cfg, vb).unwrap();
            randomize_base(&vm, &dev);
            let suffixes: Vec<String> = LENS_ATTN_TARGETS.iter().map(|s| s.to_string()).collect();
            let set = build_lora_targets(&mut dit, &suffixes, 4, 8.0, 7, &dev).unwrap();
            for v in &set.vars {
                v.set(&Tensor::randn(0f32, 0.02f32, v.as_tensor().dims(), &dev).unwrap())
                    .unwrap();
            }
            let (x0, feats, noise, h, w) = tiny_inputs(&cfg, &dev);
            Fixture {
                dit,
                vars: set.vars,
                x0,
                feats,
                noise,
                h,
                w,
            }
        }

        /// Step 1 = diffusion step, step 2 = the item's 2nd window ⇒ aux-only.
        fn driver(f: &Fixture) -> (AuxDriver, gen_core::BucketSchedule) {
            let path = candle_gen_perceptual::testing::tiny_depth_path(8, schedule(), &Device::Cpu)
                .unwrap();
            let sched = gen_core::BucketSchedule::new(1, &[], 3);
            let clean = unpack_to_decoder_latent(&f.x0, f.h, f.w).unwrap();
            let d = AuxDriver::prepare(path, 1, |_| Ok(clean.clone()), &sched, 1, 0).unwrap();
            (d, sched)
        }

        fn run(
            f: &Fixture,
            t: f64,
            ckpt: bool,
            aux: Option<&AuxStep<'_>>,
        ) -> (StepLosses, GradStore) {
            super::super::compute_loss_grads(
                &f.dit,
                &f.vars,
                &f.x0,
                &f.feats,
                f.h,
                f.w,
                t,
                &f.noise,
                false,
                None,
                DType::F32,
                ckpt,
                aux,
            )
            .unwrap()
        }

        fn grad_bits(g: &GradStore, v: &Var) -> Vec<u32> {
            g.get(v.as_tensor())
                .map(|t| {
                    t.flatten_all()
                        .unwrap()
                        .to_vec1::<f32>()
                        .unwrap()
                        .iter()
                        .map(|x| x.to_bits())
                        .collect()
                })
                .unwrap_or_default()
        }

        /// The decoder layout is the exact inverse of the cache packing (FLUX.2 2×2 patchify +
        /// flatten). Mutation: swap the `(ph, pw)` axes in the unpatchify permute ⇒ red.
        #[test]
        fn unpack_inverts_the_cache_packing() {
            let dev = Device::Cpu;
            let n = 8 * 6 * 4;
            let unpacked = Tensor::arange(0f32, n as f32, &dev)
                .unwrap()
                .reshape((1, 8, 6, 4))
                .unwrap();
            let packed = pack_unpacked_latent(&unpacked).unwrap(); // [1, 3·2, 32]
            let back = unpack_to_decoder_latent(&packed, 3, 2).unwrap();
            assert_eq!(
                back.flatten_all().unwrap().to_vec1::<f32>().unwrap(),
                unpacked.flatten_all().unwrap().to_vec1::<f32>().unwrap()
            );
        }

        /// (a)/(b) on both backward paths: the diffusion step has no aux term; the aux-only step has
        /// no diffusion term, total == aux, and a nonzero finite LoRA B gradient. Mutations:
        /// compute the diffusion term unconditionally ⇒ red; drop `aux` in the checkpointed
        /// segment ⇒ red.
        #[test]
        fn aux_only_step_trains_the_adapter_through_depth_alone() {
            let f = fixture();
            let (mut d, sched) = driver(&f);
            let s1 = d.sample(1, &sched).plan(0.5).unwrap().unwrap();
            assert!(s1.diffusion() && !s1.has_aux());
            for ckpt in [false, true] {
                let (l, _) = run(&f, 0.5, ckpt, Some(&s1));
                assert!(
                    l.diffusion.is_some() && l.aux.is_none(),
                    "ckpt={ckpt}: {l:?}"
                );
                assert_eq!(l.total, l.diffusion.unwrap());
            }
            let s2 = d.sample(2, &sched).plan(0.5).unwrap().unwrap();
            assert!(!s2.diffusion() && s2.has_aux());
            let t = s2.noise_level() as f64;
            for ckpt in [false, true] {
                let (l, g) = run(&f, t, ckpt, Some(&s2));
                assert!(l.diffusion.is_none(), "ckpt={ckpt}: {l:?}");
                let aux = l.aux.expect("aux term");
                assert!(aux.is_finite() && aux > 0.0, "ckpt={ckpt}: {aux}");
                assert_eq!(l.total, aux);
                let mut nonzero_b = false;
                for (i, v) in f.vars.iter().enumerate() {
                    let gv: Vec<f32> = grad_bits(&g, v).into_iter().map(f32::from_bits).collect();
                    assert!(gv.iter().all(|x| x.is_finite()), "ckpt={ckpt} var {i}");
                    nonzero_b |= i % 2 == 1 && gv.iter().any(|x| *x != 0.0);
                }
                assert!(
                    nonzero_b,
                    "ckpt={ckpt}: no LoRA B gradient from the depth term"
                );
            }
        }

        /// The aux term is the depth loss of THE trainer's x0 estimate: recomputed independently from
        /// the model output with the trainer's parameterisation, it matches the step's aux bits.
        /// Mutation: swap the parameterisation in `step_loss` (e.g. `FlowX0MinusNoise`) ⇒ red.
        #[test]
        fn aux_term_is_the_depth_loss_of_the_recovered_x0() {
            let f = fixture();
            let (mut d, sched) = driver(&f);
            let _ = d.sample(1, &sched);
            let s2 = d.sample(2, &sched).plan(0.5).unwrap().unwrap();
            let t = s2.noise_level() as f64;
            let (l, _) = run(&f, t, false, Some(&s2));
            let (x_t, _) = flow_match::build_batch(&f.x0, &f.noise, t).unwrap();
            let v = f
                .dit
                .forward(&x_t, &f.feats, None, t as f32, 1, f.h, f.w)
                .unwrap();
            let x0 = (&x_t - (v * t as f32 as f64).unwrap()).unwrap();
            let x0 = unpack_to_decoder_latent(&x0, f.h, f.w).unwrap();
            let want = s2
                .aux_loss(&x0)
                .unwrap()
                .unwrap()
                .to_scalar::<f32>()
                .unwrap();
            assert_eq!(l.aux.unwrap().to_bits(), want.to_bits());
        }

        /// (c) Depth off is bit-identical to the pre-epic-2123 dense step (reproduced verbatim).
        /// Mutation: perturb the `None` combine ⇒ red.
        #[test]
        fn depth_off_is_bit_identical_to_the_legacy_step() {
            let f = fixture();
            let (l, g) = run(&f, 0.5, false, None);
            assert_eq!((l.diffusion, l.aux), (Some(l.total), None));
            let (x_t, target) = flow_match::build_batch(&f.x0, &f.noise, 0.5).unwrap();
            let v = f
                .dit
                .forward(&x_t, &f.feats, None, 0.5, 1, f.h, f.w)
                .unwrap();
            let loss = weighted_velocity_loss(&v, &target, None, false).unwrap();
            let legacy = loss.to_scalar::<f32>().unwrap();
            let lg = loss.backward().unwrap();
            assert_eq!(l.total.to_bits(), legacy.to_bits());
            for v in &f.vars {
                assert_eq!(grad_bits(&g, v), grad_bits(&lg, v));
            }
        }

        /// `micro_step` (the checkpointed production path) trains an aux-only step at the plan's
        /// remapped `t`. Mutation: drop `t = a.noise_level()` ⇒ red.
        #[test]
        fn micro_step_trains_at_the_planned_noise_level() {
            let f = fixture();
            let (mut d, sched) = driver(&f);
            let trainer = LensTrainer {
                descriptor: trainer_descriptor(),
                root: "/nonexistent".into(),
                device: Device::Cpu,
            };
            let cfg = TrainingConfig {
                train_dtype: "f32".into(),
                ..TrainingConfig::default()
            };
            let cached = (f.x0.clone(), f.feats.clone(), (f.h, f.w), None);
            let _ = d.sample(1, &sched);
            let sample = d.sample(2, &sched);
            let raw = flow_match::sample_unit_timestep(
                &cfg.timestep_type,
                &cfg.timestep_bias,
                flow_match::timestep_seed(cfg.seed, 2),
            );
            let planned = sample.plan(raw).unwrap().unwrap();
            assert!(!planned.diffusion());
            let noise = flow_match::sample_noise(
                f.x0.dims(),
                flow_match::noise_seed(cfg.seed, 2),
                &Device::Cpu,
            )
            .unwrap();
            let expect = super::super::compute_loss_grads(
                &f.dit,
                &f.vars,
                &f.x0,
                &f.feats,
                f.h,
                f.w,
                planned.noise_level() as f64,
                &noise,
                false,
                None,
                DType::F32,
                true,
                Some(&planned),
            )
            .unwrap()
            .0
            .total;
            let (got, _) = trainer
                .micro_step(&f.dit, &f.vars, &cached, &(), &cfg, 2, sample, &Device::Cpu)
                .unwrap();
            assert_eq!(got.to_bits(), expect.to_bits());
        }

        fn depth_on() -> TrainingConfig {
            let mut cfg = TrainingConfig::default();
            cfg.depth_anchoring.schedule = schedule();
            cfg
        }

        /// (d) E7: footprint 0 off, larger for Large DA2; the preflight refuses between base and
        /// base + aux, checkpointed or dense. Mutation: skip the guard ⇒ red.
        #[test]
        fn preflight_counts_the_perceptual_models() {
            assert_eq!(perceptual_footprint_bytes(&TrainingConfig::default(), 4), 0);
            let mut cfg = depth_on();
            let small = perceptual_footprint_bytes(&cfg, 4);
            assert!(small > 0);
            cfg.depth_anchoring.model_size = DepthModelSize::Large;
            assert!(perceptual_footprint_bytes(&cfg, 4) > small);
            cfg.depth_anchoring.model_size = DepthModelSize::Small;
            let tmp = tempfile::tempdir().unwrap();
            let tdir = tmp.path().join("transformer");
            std::fs::create_dir_all(&tdir).unwrap();
            std::fs::write(tdir.join("m.safetensors"), vec![0u8; 4096]).unwrap();
            let base = 4096u64;
            assert!(aux_memory_preflight(tmp.path(), &TrainingConfig::default(), 4, 1).is_ok());
            for ckpt in [false, true] {
                cfg.gradient_checkpointing = ckpt;
                assert!(aux_memory_preflight(tmp.path(), &cfg, 4, base + small).is_ok());
                let e = aux_memory_preflight(tmp.path(), &cfg, 4, base + small - 1)
                    .unwrap_err()
                    .to_string();
                assert!(e.contains("perceptual"), "ckpt={ckpt}: {e}");
            }
        }

        /// (e) The descriptor declares depth anchoring; depth off builds no path; a missing decoder
        /// dir names TAEF2. Mutation: `depth_anchoring: false` ⇒ red.
        #[test]
        fn descriptor_and_loader_errors() {
            assert!(trainer_descriptor().techniques.depth_anchoring);
            let dev = Device::Cpu;
            assert!(load_perceptual_path(&TrainingConfig::default(), &dev)
                .unwrap()
                .is_none());
            let tmp = tempfile::tempdir().unwrap();
            let mut cfg = depth_on();
            cfg.depth_anchoring.model_dir = Some(tmp.path().join("da2"));
            let e = load_perceptual_path(&cfg, &dev).err().unwrap().to_string();
            assert!(e.contains("TAEF2"), "{e}");
            cfg.perceptual_decoder_dir = Some(tmp.path().join("no-taef2"));
            let e = load_perceptual_path(&cfg, &dev).err().unwrap().to_string();
            assert!(e.contains("TAEF2"), "{e}");
        }
    }
}
