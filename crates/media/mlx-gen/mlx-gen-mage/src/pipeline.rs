//! Rectified-flow sampler, native-resolution packing, and the prompt→image / edit paths —
//! **owned by sc-14041** (native-resolution work continues in sc-14043, edit in sc-14048).
//!
//! Port of `_vendor/mage_flow/pipeline.py`. The pieces, with their pinned answers:
//!
//! - **Schedule.** `FlowMatchEulerDiscreteScheduler(num_train_timesteps=1000, shift=6.0,
//!   use_dynamic_shifting=false)` with `set_timesteps(sigmas=linspace(1, 1/N, N))` (`:37-50`) —
//!   static shift `6σ/(1+5σ)` plus a terminal 0. **Turbo is the same ladder at N = 4**, not a
//!   distilled timestep table: N=4 gives sigmas `[1.0, 0.94736844, 0.85714287, 0.66666669, 0.0]`.
//!   The step is plain Euler, `x += (σ_next − σ_cur) · v` (`:343`).
//! - **Initial latent.** [`crate::latent`], never plain `randn`.
//! - **Packing.** Latents flatten to a variable-length token sequence (`patch_size == 1`) and are
//!   packed under a fixed budget with per-sample cumulative offsets (`cu_seqlens`) instead of
//!   block-diagonal masks. Sides must be multiples of
//!   [`SIZE_MULTIPLE`]; the native range is
//!   [`MIN_SIZE`]–[`MAX_SIZE`] per side.
//! - **CFG.** `use_neg = cfg != 1.0` (`:326`, `:535`): at cfg ≤ 1 the reference builds **no**
//!   unconditional branch at all — one segment, one `cu_seqlens` pair, positive conditioning only.
//!   Both Turbo variants default there, so the CFG-off path is a first-class case, not an edge one.
//!   Under `batch_cfg` the duplicated uncond branch rotates at msrope frame index 1 — see
//!   [`crate::rope_embedder`].
//! - **Edit sequence.** `[noisy_target, ref_1, …, ref_N]` — **target first** (`:552-555`), which
//!   corrects the epic's `[τ, z_src, noisy z_tgt]` on both ordering and τ-placement: τ is the
//!   *separate text stream*, not part of the image sequence. Refs are clean latents,
//!   re-concatenated every step, and only the target tokens are stepped (`:557-565`). Frame index:
//!   target 0, ref_j → j. Refs are VAE-encoded at *target* resolution; the copy fed to the VL
//!   vision tower is long-edge capped at
//!   [`VL_COND_LONG_EDGE`](crate::config::VL_COND_LONG_EDGE).
//! - **Decode.** `vae.decode(unpack(tokens.float(), h, w))` (`:121-127`); `unpack` reshapes at
//!   `ceil(height/16) × ceil(width/16)` (`models/utils.py:36`).
//!
//! Boundary goldens for every one of these stages, and the hardened checker that verifies them
//! (76 invariants at cfg > 1, 71 at cfg ≤ 1, with `--self-test`), live in
//! `crates/media/mlx-gen/tools/`.

use image::{imageops::FilterType, RgbImage};
use mlx_gen::attention::{AttentionBudget, AttentionPlan};
use mlx_gen::driver::{
    run_mlx_driver_slices, DriverEvaluationKind, DriverHistoryPolicy, DriverRefusalReasons,
    DriverRunState, DriverStepInfo, DriverSupport, FamilyDriver, SliceInput, SliceSpec,
    SliceTransition, StepHooks,
};
use mlx_gen::gen_core::{BoundaryState, GenerationMemory};
use mlx_gen::tiling::{SpatialTiling, TilingConfig};
use mlx_gen::{CancelFlag, Error, Progress, Result};
use mlx_rs::ops::{concatenate_axis, maximum, minimum};
use mlx_rs::{Array, Dtype};

use crate::config::{LATENT_CHANNELS, MAX_SIZE, MIN_SIZE, SIZE_MULTIPLE, VAE_DOWNSAMPLE_FACTOR};
use crate::latent::{encode_noise, GsKey};
use crate::rope_embedder::{ImgShape, PackContext, PackLayout};
use crate::text_encoder::{MageTextEncoder, PromptKind};
use crate::transformer::MageTransformer;
use crate::vae::MageVae;
use std::path::Path;

/// The published Mage-Flow scheduler shift.
pub const STATIC_SHIFT: f32 = 6.0;

/// The frozen pipeline constructs an unconditional branch only above guidance scale 1.
pub const fn uses_cfg(cfg: f32) -> bool {
    cfg != 1.0
}

/// Upstream's native-resolution image-token budget for one packed DiT invocation.
///
/// This budget is applied before the optional fused-CFG duplication, exactly like the reference's
/// sample packer. It partitions samples; it never rounds or bucket-quantizes their dimensions.
pub const MAX_PACKED_IMAGE_TOKENS: i32 = 50_000;

/// One independently seeded, native-resolution generation request.
#[derive(Debug, Clone, Copy)]
pub struct GenerationSample<'a> {
    pub prompt: &'a str,
    pub negative_prompt: &'a str,
    pub height: u32,
    pub width: u32,
    pub seed: i64,
}

/// A consecutive group of samples that fits one native-resolution DiT pack.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct GenerationPack {
    pub sample_range: std::ops::Range<usize>,
    pub grids: Vec<(i32, i32)>,
    pub image_tokens: i32,
}

/// Per-sample observable outputs from variable-geometry generation.
pub struct BatchGenerationTrace {
    pub samples: Vec<GenerationTrace>,
    pub packs: Vec<GenerationPack>,
}

/// The four loaded components needed for the Mage-Flow generation path.
pub struct MageFlowPipeline {
    pub text_encoder: MageTextEncoder,
    pub transformer: MageTransformer,
    pub vae: MageVae,
}

/// Observable boundary tensors retained by the real-weight parity gate.
pub struct GenerationTrace {
    pub final_tokens: Array,
    pub final_latent: Array,
    pub trajectories: Vec<Array>,
    /// Exact reference byte conversion, HWC `Uint8`.
    pub image_u8: Array,
}

pub struct EditTrace {
    pub final_tokens: Array,
    pub reference_tokens: Array,
    pub trajectories: Vec<Array>,
    pub image_u8: Array,
}

/// Conditioning retained after the request-scoped text phase has been synchronized and released.
pub(crate) struct EncodedGenerationPack {
    pub(crate) pack: GenerationPack,
    cond: Array,
    positive_lens: Vec<i32>,
    negative: Option<(Array, Vec<i32>)>,
}

pub(crate) struct DenoisedGenerationSample {
    tokens: Array,
    gh: i32,
    gw: i32,
    height: u32,
    width: u32,
    trajectories: Vec<Array>,
}
pub(crate) struct DenoisedGenerationBatch {
    samples: Vec<DenoisedGenerationSample>,
    packs: Vec<GenerationPack>,
}

pub(crate) struct EncodedEdit {
    cond: Array,
    cond_len: i32,
    negative: Option<(Array, i32)>,
}

pub(crate) struct DenoisedEdit {
    target: Array,
    reference_tokens: Array,
    trajectories: Vec<Array>,
    gh: i32,
    gw: i32,
}

pub(crate) fn encode_generation_phase(
    text: &MageTextEncoder,
    samples: &[GenerationSample<'_>],
    cfg: f32,
    cancel: &CancelFlag,
) -> Result<Vec<EncodedGenerationPack>> {
    let packs = plan_generation_packs(samples)?;
    packs
        .into_iter()
        .map(|pack| {
            let pack_samples = &samples[pack.sample_range.clone()];
            let mut texts = pack_samples
                .iter()
                .map(|sample| sample.prompt)
                .collect::<Vec<_>>();
            if uses_cfg(cfg) {
                texts.extend(pack_samples.iter().map(|sample| sample.negative_prompt));
            }
            let conditioning = text.encode_with_cancel(&texts, PromptKind::Gen, Some(cancel))?;
            let count = pack_samples.len();
            let positive_lens = conditioning.seq_lens[..count]
                .iter()
                .map(|&len| len as i32)
                .collect::<Vec<_>>();
            let positive_tokens: i32 = positive_lens.iter().sum();
            let parts = if uses_cfg(cfg) {
                conditioning.txt.split_axis(&[positive_tokens], 0)?
            } else {
                vec![conditioning.txt]
            };
            let cond = parts[0].reshape(&[1, positive_tokens, -1])?;
            let negative = if uses_cfg(cfg) {
                let lens = conditioning.seq_lens[count..]
                    .iter()
                    .map(|&len| len as i32)
                    .collect::<Vec<_>>();
                Some((parts[1].reshape(&[1, lens.iter().sum(), -1])?, lens))
            } else {
                None
            };
            Ok(EncodedGenerationPack {
                pack,
                cond,
                positive_lens,
                negative,
            })
        })
        .collect()
}

pub(crate) fn materialize_generation_encoded(encoded: &Vec<EncodedGenerationPack>) -> Result<()> {
    for pack in encoded {
        mlx_rs::transforms::eval(
            std::iter::once(&pack.cond).chain(pack.negative.as_ref().map(|(txt, _)| txt)),
        )?;
    }
    Ok(())
}

#[allow(clippy::too_many_arguments)]
pub(crate) fn denoise_generation_phase(
    transformer: &MageTransformer,
    samples: &[GenerationSample<'_>],
    encoded: Vec<EncodedGenerationPack>,
    steps: usize,
    cfg: f32,
    gs_key: &GsKey,
    scheduler_shift: Option<f32>,
    memory: Option<GenerationMemory>,
    cancel: &CancelFlag,
    on_progress: &mut dyn FnMut(Progress),
) -> Result<DenoisedGenerationBatch> {
    let native = mage_flow_sigmas(steps)?;
    let sigmas = mlx_gen::resolve_flow_schedule_with_shift(
        None,
        scheduler_shift,
        STATIC_SHIFT.ln(),
        steps,
        &native,
    )?;
    let total_steps = steps * encoded.len();
    let mut out = Vec::with_capacity(samples.len());
    let mut packs = Vec::with_capacity(encoded.len());
    for (pack_index, encoded) in encoded.into_iter().enumerate() {
        let pack_samples = &samples[encoded.pack.sample_range.clone()];
        packs.push(encoded.pack.clone());
        let initial = pack_samples
            .iter()
            .map(|sample| {
                initial_tokens(
                    sample.height,
                    sample.width,
                    sample.seed,
                    gs_key,
                    Dtype::Bfloat16,
                )
            })
            .collect::<Result<Vec<_>>>()?;
        let refs = initial.iter().collect::<Vec<_>>();
        let tokens = concatenate_axis(&refs, 1)?;
        let layout = generation_layout(&encoded.pack.grids, encoded.positive_lens)?;
        let mut trajectories = Vec::new();
        let tokens = denoise_capture_with_progress(
            transformer,
            tokens,
            &encoded.cond,
            layout,
            encoded
                .negative
                .as_ref()
                .map(|(txt, lens)| (txt, lens.clone())),
            cfg,
            false,
            &sigmas,
            memory,
            cancel,
            &mut |step, latent| {
                if step < 2 {
                    trajectories.push(latent.clone());
                }
            },
            &mut |progress_event| match progress_event {
                Progress::Step { current, .. } => on_progress(Progress::Step {
                    current: (pack_index * steps) as u32 + current,
                    total: total_steps as u32,
                }),
                other => on_progress(other),
            },
        )?;
        mlx_gen::image_execution::policy_debug("mage.memory", &memory);
        mlx_gen::image_execution::scalar("mage.cfg", cfg);
        for sample in pack_samples {
            mlx_gen::image_execution::sampled_native_with_rng(
                "mage-euler-v1",
                "noise-fraction",
                tokens.dtype(),
                &sigmas,
                sample.seed as u64,
                "mage-gaussian-shading-v1;seed=wrapping-i64;gs-key=provider-resolved",
            );
        }
        let boundaries = encoded
            .pack
            .grids
            .iter()
            .scan(0, |at, &(gh, gw)| {
                *at += gh * gw;
                Some(*at)
            })
            .collect::<Vec<_>>();
        let split = &boundaries[..boundaries.len().saturating_sub(1)];
        let pieces = tokens.split_axis(split, 1)?;
        for ((tokens, &(gh, gw)), sample) in pieces
            .into_iter()
            .zip(&encoded.pack.grids)
            .zip(pack_samples)
        {
            out.push(DenoisedGenerationSample {
                tokens,
                gh,
                gw,
                height: sample.height,
                width: sample.width,
                trajectories: if pack_samples.len() == 1 {
                    std::mem::take(&mut trajectories)
                } else {
                    Vec::new()
                },
            });
        }
    }
    Ok(DenoisedGenerationBatch {
        samples: out,
        packs,
    })
}

/// Decode every sample in a batch, opening the request's decode phase first.
///
/// **The `Progress::Decoding` boundary is emitted exactly once, before the loop** (sc-22738). The
/// production `Generator::generate` path used to discard the sink here entirely, so the decode
/// phase the memory contract declares was invisible to every consumer; emitting per sample instead
/// would restart the bar per output, which the gen-core progress contract rejects (F-136/F-162).
pub(crate) fn decode_generation_phase(
    vae: &MageVae,
    denoised: DenoisedGenerationBatch,
    memory: Option<GenerationMemory>,
    cancel: &CancelFlag,
    on_progress: &mut dyn FnMut(Progress),
) -> Result<BatchGenerationTrace> {
    on_progress(Progress::Decoding);
    let mut samples = Vec::with_capacity(denoised.samples.len());
    for item in denoised.samples {
        let image_u8 = decode_with_memory(vae, &item.tokens, item.gh, item.gw, memory, cancel)?;
        samples.push(GenerationTrace {
            final_latent: unpack_tokens(&item.tokens, item.gh, item.gw)?,
            final_tokens: item.tokens,
            trajectories: item.trajectories,
            image_u8,
        });
        debug_assert_eq!(
            samples.last().unwrap().image_u8.shape(),
            [item.height as i32, item.width as i32, 3]
        );
    }
    Ok(BatchGenerationTrace {
        samples,
        packs: denoised.packs,
    })
}

pub(crate) fn materialize_generation_denoised(denoised: &DenoisedGenerationBatch) -> Result<()> {
    mlx_rs::transforms::eval(denoised.samples.iter().map(|sample| &sample.tokens))?;
    Ok(())
}

pub(crate) fn encode_edit_phase(
    text: &MageTextEncoder,
    instruction: &str,
    negative_instruction: &str,
    references: &[RgbImage],
    cfg: f32,
    cancel: &CancelFlag,
) -> Result<EncodedEdit> {
    let positive = text.encode_edit_with_cancel(instruction, references, Some(cancel))?;
    let cond_len = positive.seq_lens[0] as i32;
    let cond = positive.txt.reshape(&[1, cond_len, -1])?;
    let negative = if uses_cfg(cfg) {
        let encoded =
            text.encode_edit_with_cancel(negative_instruction, references, Some(cancel))?;
        let len = encoded.seq_lens[0] as i32;
        Some((encoded.txt.reshape(&[1, len, -1])?, len))
    } else {
        None
    };
    Ok(EncodedEdit {
        cond,
        cond_len,
        negative,
    })
}

pub(crate) fn materialize_edit_encoded(encoded: &EncodedEdit) -> Result<()> {
    mlx_rs::transforms::eval(
        std::iter::once(&encoded.cond).chain(encoded.negative.as_ref().map(|(txt, _)| txt)),
    )?;
    Ok(())
}

#[allow(clippy::too_many_arguments)]
pub(crate) fn denoise_edit_phase(
    transformer: &MageTransformer,
    vae: &MageVae,
    encoded: EncodedEdit,
    references: &[RgbImage],
    height: u32,
    width: u32,
    steps: usize,
    cfg: f32,
    seed: i64,
    gs_key: &GsKey,
    scheduler_shift: Option<f32>,
    memory: Option<GenerationMemory>,
    cancel: &CancelFlag,
    on_progress: &mut dyn FnMut(Progress),
) -> Result<DenoisedEdit> {
    let (gh, gw) = edit_latent_hw(height, width)?;
    let resized = references
        .iter()
        .map(|image| reference_nchw(image, height, width))
        .collect::<Result<Vec<_>>>()?;
    let refs = resized.iter().collect::<Vec<_>>();
    let ref_batch = concatenate_axis(&refs, 0)?;
    let reference_tokens = vae
        .encode_sample(&ref_batch, seed as u64)?
        .transpose_axes(&[0, 2, 3, 1])?
        .reshape(&[1, references.len() as i32 * gh * gw, LATENT_CHANNELS])?
        .as_dtype(Dtype::Bfloat16)?;
    let target = initial_tokens_for_grid(gh, gw, seed, gs_key, Dtype::Bfloat16)?;
    let shapes = std::iter::repeat_n(ImgShape::latent(gh, gw), references.len() + 1).collect();
    let total_tokens = (references.len() as i32 + 1) * gh * gw;
    let layout = PackLayout::new(shapes, vec![total_tokens], vec![encoded.cond_len])?;
    let native = mage_flow_sigmas(steps)?;
    let sigmas = mlx_gen::resolve_flow_schedule_with_shift(
        None,
        scheduler_shift,
        STATIC_SHIFT.ln(),
        steps,
        &native,
    )?;
    let mut trajectories = Vec::with_capacity(2);
    let target = run_mage_mlx_driver(
        "mage_mlx_edit_driver",
        transformer,
        target,
        gh * gw,
        Some(reference_tokens.clone()),
        &encoded.cond,
        layout,
        encoded
            .negative
            .as_ref()
            .map(|(txt, len)| (txt, vec![*len])),
        cfg,
        false,
        &sigmas,
        memory,
        cancel,
        Some(&mut trajectories),
        on_progress,
    )?;
    mlx_gen::image_execution::policy_debug("mage.memory", &memory);
    mlx_gen::image_execution::scalar("mage.cfg", cfg);
    mlx_gen::image_execution::sampled_native_with_rng("mage-edit-euler-v1", "noise-fraction", target.dtype(), &sigmas, seed as u64,
        "mage-gaussian-shading-v1;seed=wrapping-i64;reference-vae-posterior=seed;gs-key=provider-resolved");
    Ok(DenoisedEdit {
        target,
        reference_tokens,
        trajectories,
        gh,
        gw,
    })
}

/// Decode one edited image, opening the request's decode phase first (sc-22738 — see
/// [`decode_generation_phase`]).
pub(crate) fn decode_edit_phase(
    vae: &MageVae,
    denoised: DenoisedEdit,
    memory: Option<GenerationMemory>,
    cancel: &CancelFlag,
    on_progress: &mut dyn FnMut(Progress),
) -> Result<EditTrace> {
    on_progress(Progress::Decoding);
    let image_u8 = decode_with_memory(
        vae,
        &denoised.target,
        denoised.gh,
        denoised.gw,
        memory,
        cancel,
    )?;
    Ok(EditTrace {
        final_tokens: denoised.target,
        reference_tokens: denoised.reference_tokens,
        trajectories: denoised.trajectories,
        image_u8,
    })
}

pub(crate) fn materialize_edit_denoised(denoised: &DenoisedEdit) -> Result<()> {
    mlx_rs::transforms::eval([&denoised.target, &denoised.reference_tokens])?;
    Ok(())
}

/// Where each Mage-Flow component's weights live.
///
/// The published upstream snapshot is FLAT — one directory holding `transformer/`, `text_encoder/`
/// and `vae/` — and [`MageComponentDirs::flat`] preserves exactly that. The SceneWorks mirrors add a
/// SPLIT layout (sc-14980 / sc-14979): the per-tier DiT lives in the variant mirror's
/// `<tier>/transformer/`, while the text encoder and VAE — **bit-identical across all six
/// variants** — are hosted once in a shared components mirror and staged as caller-provisioned
/// co-requisite paths. Carrying the three dirs explicitly is what lets one loader serve both.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct MageComponentDirs {
    /// The `transformer/` dir (its `config.json` + `diffusion_pytorch_model.safetensors`).
    pub transformer: std::path::PathBuf,
    /// The `text_encoder/` dir (Qwen3-VL weights, config, tokenizer/processor assets).
    pub text_encoder: std::path::PathBuf,
    /// The `vae/` dir.
    pub vae: std::path::PathBuf,
}

impl MageComponentDirs {
    /// The flat published layout: every component directly under `root`.
    pub fn flat(root: impl AsRef<Path>) -> Self {
        let root = root.as_ref();
        Self {
            transformer: root.join("transformer"),
            text_encoder: root.join("text_encoder"),
            vae: root.join("vae"),
        }
    }
}

/// Read a component dir's packed tier marker (`config.json` → `quantization.bits`).
///
/// [`mlx_gen::quant::packed_quant_bits`] is keyed by `(root, component)`; these paths are already
/// full component dirs, so split the last segment back off rather than re-implementing the parse.
fn component_packed_bits(dir: &Path) -> Result<Option<i32>> {
    let (Some(parent), Some(name)) = (dir.parent(), dir.file_name().and_then(|n| n.to_str()))
    else {
        return Err(Error::Msg(format!(
            "mage_flow: component path {} has no directory name",
            dir.display()
        )));
    };
    mlx_gen::quant::packed_quant_bits(parent, name)
}

/// Decide whether `dir` still needs load-time quantization to serve `requested` bits.
///
/// `Some(bits)` ⇒ run the load-time `quantize(bits)`; `None` ⇒ skip it, because the artifact is
/// already packed at the requested tier (or a dense tier was requested and the artifact is dense).
/// A tier mismatch in either direction is a hard error, never a silent downgrade: `quantize` is a
/// no-op over packed weights, so serving a Q4 request from a Q8 artifact would quietly hand back Q8.
pub(crate) fn load_time_quant_bits(
    dir: &Path,
    requested: Option<i32>,
    model_id: &str,
) -> Result<Option<i32>> {
    let (Some(parent), Some(name)) = (dir.parent(), dir.file_name().and_then(|n| n.to_str()))
    else {
        return Err(Error::Msg(format!(
            "mage_flow: component path {} has no directory name",
            dir.display()
        )));
    };
    match requested {
        // A bf16 request must be served by dense weights: `quantize` is never called, so a packed
        // artifact here would silently serve Q4/Q8 under a bf16 label.
        None => match component_packed_bits(dir)? {
            Some(packed) => Err(Error::Msg(format!(
                "{model_id}: {} is a pre-quantized Q{packed} artifact but the bf16 (dense) tier was \
                 requested; point at the bf16 tier or request Q{packed}",
                dir.display()
            ))),
            None => Ok(None),
        },
        Some(bits) => Ok(mlx_gen::quant::needs_load_time_quant(parent, name, bits, model_id)?
            .then_some(bits)),
    }
}

pub(crate) struct MageHeavyComponents {
    pub(crate) transformer: MageTransformer,
    pub(crate) vae: MageVae,
}

pub(crate) fn load_text_component(
    dirs: &MageComponentDirs,
    quant_bits: Option<i32>,
    multimodal: bool,
    stream_lm: bool,
) -> Result<MageTextEncoder> {
    let bits = load_time_quant_bits(&dirs.text_encoder, quant_bits, "mage_flow")?;
    if stream_lm && bits.is_some() {
        return Err(Error::Unsupported(
            "mage_flow: deferred text-layer residency requires a dense bf16 snapshot or a prepacked matching tier; load-time layer quantization is not streamable".into(),
        ));
    }
    let mut text = if multimodal {
        crate::text_encoder::load::load_multimodal_dir_with(&dirs.text_encoder, stream_lm)?
    } else {
        crate::text_encoder::load::load_dir_with(&dirs.text_encoder, stream_lm)?
    };
    if let Some(bits) = bits {
        text.quantize(bits)?;
    }
    if quant_bits.is_some() {
        verify_quantized_counts("text encoder", text.quantized_linear_count(), 253)?;
    }
    if stream_lm {
        text.arm_lm_block_stream(mlx_gen::WeightsSource::Dir(dirs.text_encoder.clone()), None)?;
        if text.resident_lm_layer_count() != 0 {
            return Err(Error::Msg(
                "mage_flow: deferred text encoder retained resident language-model layers".into(),
            ));
        }
    }
    Ok(text)
}

pub(crate) fn load_heavy_components(
    dirs: &MageComponentDirs,
    quant_bits: Option<i32>,
    part: crate::vae::VaePart,
    stream_transformer: bool,
    adapters: &[mlx_gen::AdapterSpec],
) -> Result<MageHeavyComponents> {
    let dit_bits = load_time_quant_bits(&dirs.transformer, quant_bits, "mage_flow")?;
    let vae_bits = load_time_quant_bits(&dirs.vae, quant_bits, "mage_flow")?;
    if stream_transformer && dit_bits.is_some() {
        return Err(Error::Unsupported(
            "mage_flow: deferred DiT residency requires a dense bf16 snapshot or a prepacked matching tier; load-time block quantization is not streamable".into(),
        ));
    }
    let mut transformer = if stream_transformer {
        MageTransformer::load_streamed(&dirs.transformer)?
    } else {
        MageTransformer::load(&dirs.transformer)?
    };
    if stream_transformer {
        transformer = transformer.with_block_stream(
            mlx_gen::WeightsSource::Dir(dirs.transformer.clone()),
            dit_bits,
        );
    }
    let mut vae = crate::vae::load(&dirs.vae, part, Dtype::Bfloat16)?;
    if let Some(bits) = dit_bits {
        transformer.quantize(bits)?;
    }
    if let Some(bits) = vae_bits {
        vae.quantize(bits)?;
    }
    crate::adapters::apply_mage_adapters(&mut transformer, adapters)?;
    if quant_bits.is_some() {
        verify_quantized_counts("DiT", transformer.quantized_linear_count(), 174)?;
        let (expected, packed) = vae.quantization_count();
        verify_quantized_counts("VAE", packed, expected)?;
    }
    transformer.finalize_block_stream()?;
    if stream_transformer && transformer.resident_block_count() != 0 {
        return Err(Error::Msg(
            "mage_flow: deferred DiT retained resident transformer blocks".into(),
        ));
    }
    Ok(MageHeavyComponents { transformer, vae })
}

impl MageFlowPipeline {
    /// Load a published diffusers snapshot (`text_encoder/`, `transformer/`, `vae/`).
    pub fn load(root: impl AsRef<Path>) -> Result<Self> {
        Self::load_with_quant(root, None)
    }

    /// Load and optionally quantize all live Linears in the TE, DiT, and VAE, from a flat snapshot.
    pub fn load_with_quant(root: impl AsRef<Path>, quant_bits: Option<i32>) -> Result<Self> {
        Self::load_components(
            &MageComponentDirs::flat(root),
            quant_bits,
            crate::vae::VaePart::Decode,
        )
    }

    /// Load the full edit pipeline, including the Qwen3-VL vision tower and Mage-VAE encoder.
    pub fn load_edit(root: impl AsRef<Path>, quant_bits: Option<i32>) -> Result<Self> {
        Self::load_components(
            &MageComponentDirs::flat(root),
            quant_bits,
            crate::vae::VaePart::Both,
        )
    }

    /// Load from explicitly-addressed component dirs, quantizing **per component** only where the
    /// artifact on disk is not already packed at the requested tier (sc-14980).
    ///
    /// This is the seam that makes a pre-quantized tier a drop-in for load-time quant: a packed
    /// `transformer/` and `text_encoder/` skip the dense read + quantize entirely, while the VAE —
    /// which ships dense in every tier because its quantizable weights are folded away at load (see
    /// [`crate::convert`]) — still takes the load-time path. The `quantized_linear_count`
    /// assertions run against the FINAL state either way, so a packed artifact that failed to
    /// auto-detect is caught exactly as a failed load-time pack would be.
    pub fn load_components(
        dirs: &MageComponentDirs,
        quant_bits: Option<i32>,
        part: crate::vae::VaePart,
    ) -> Result<Self> {
        Self::load_components_with_shape(dirs, quant_bits, part, false)
    }

    pub(crate) fn load_components_with_shape(
        dirs: &MageComponentDirs,
        quant_bits: Option<i32>,
        part: crate::vae::VaePart,
        stream_transformer: bool,
    ) -> Result<Self> {
        const MODEL_ID: &str = "mage_flow";
        // Resolve every tier decision BEFORE loading, so a mismatched tier fails fast instead of
        // after an 8 GB read.
        let te_bits = load_time_quant_bits(&dirs.text_encoder, quant_bits, MODEL_ID)?;
        let dit_bits = load_time_quant_bits(&dirs.transformer, quant_bits, MODEL_ID)?;
        let vae_bits = load_time_quant_bits(&dirs.vae, quant_bits, MODEL_ID)?;
        if stream_transformer && dit_bits.is_some() {
            return Err(Error::Unsupported(
                "mage_flow: deferred DiT residency requires a dense bf16 snapshot or a prepacked matching tier; load-time block quantization is not streamable".into(),
            ));
        }

        let multimodal = matches!(part, crate::vae::VaePart::Both);
        let mut transformer = if stream_transformer {
            MageTransformer::load_streamed(&dirs.transformer)?
        } else {
            MageTransformer::load(&dirs.transformer)?
        };
        if stream_transformer {
            transformer = transformer.with_block_stream(
                mlx_gen::WeightsSource::Dir(dirs.transformer.clone()),
                dit_bits,
            );
        }
        let mut pipeline = Self {
            text_encoder: if multimodal {
                crate::text_encoder::load_multimodal_dir(&dirs.text_encoder)?
            } else {
                crate::text_encoder::load_dir(&dirs.text_encoder)?
            },
            transformer,
            vae: crate::vae::load(&dirs.vae, part, Dtype::Bfloat16)?,
        };
        if let Some(bits) = te_bits {
            pipeline.text_encoder.quantize(bits)?;
        }
        if let Some(bits) = dit_bits {
            pipeline.transformer.quantize(bits)?;
        }
        if let Some(bits) = vae_bits {
            pipeline.vae.quantize(bits)?;
        }
        if quant_bits.is_some() && !multimodal {
            verify_quantized_counts(
                "text encoder",
                pipeline.text_encoder.quantized_linear_count(),
                253,
            )?;
            verify_quantized_counts("DiT", pipeline.transformer.quantized_linear_count(), 174)?;
            let (expected_vae, packed_vae) = pipeline.vae.quantization_count();
            verify_quantized_counts("VAE", packed_vae, expected_vae)?;
        }
        pipeline.transformer.finalize_block_stream()?;
        Ok(pipeline)
    }

    /// Generate one decoded NCHW image in the reference's `[0,255]` float range.
    #[allow(clippy::too_many_arguments)]
    pub fn generate(
        &self,
        prompt: &str,
        negative_prompt: &str,
        height: u32,
        width: u32,
        steps: usize,
        cfg: f32,
        seed: i64,
        gs_key: &GsKey,
        renormalize: bool,
    ) -> Result<Array> {
        Ok(self
            .generate_trace(
                prompt,
                negative_prompt,
                height,
                width,
                steps,
                cfg,
                seed,
                gs_key,
                renormalize,
                &mut |_| {},
            )?
            .image_u8)
    }

    /// Generate while retaining parity boundaries and reporting normal platform progress.
    #[allow(clippy::too_many_arguments)]
    pub fn generate_trace(
        &self,
        prompt: &str,
        negative_prompt: &str,
        height: u32,
        width: u32,
        steps: usize,
        cfg: f32,
        seed: i64,
        gs_key: &GsKey,
        renormalize: bool,
        on_progress: &mut dyn FnMut(Progress),
    ) -> Result<GenerationTrace> {
        let mut batch = self.generate_batch_trace(
            &[GenerationSample {
                prompt,
                negative_prompt,
                height,
                width,
                seed,
            }],
            steps,
            cfg,
            gs_key,
            renormalize,
            on_progress,
        )?;
        Ok(batch.samples.swap_remove(0))
    }

    /// Instruction-edit one target from one or more references.
    #[allow(clippy::too_many_arguments)]
    pub fn edit_trace(
        &self,
        instruction: &str,
        negative_instruction: &str,
        references: &[RgbImage],
        height: u32,
        width: u32,
        steps: usize,
        cfg: f32,
        seed: i64,
        gs_key: &GsKey,
        renormalize: bool,
        on_progress: &mut dyn FnMut(Progress),
    ) -> Result<EditTrace> {
        self.edit_trace_with_memory(
            instruction,
            negative_instruction,
            references,
            height,
            width,
            steps,
            cfg,
            seed,
            gs_key,
            renormalize,
            None,
            &CancelFlag::default(),
            on_progress,
        )
    }

    #[allow(clippy::too_many_arguments)]
    pub fn edit_trace_with_memory(
        &self,
        instruction: &str,
        negative_instruction: &str,
        references: &[RgbImage],
        height: u32,
        width: u32,
        steps: usize,
        cfg: f32,
        seed: i64,
        gs_key: &GsKey,
        renormalize: bool,
        memory: Option<GenerationMemory>,
        cancel: &CancelFlag,
        on_progress: &mut dyn FnMut(Progress),
    ) -> Result<EditTrace> {
        if references.is_empty() {
            return Err(Error::Msg(
                "mage_flow edit: at least one reference image is required".into(),
            ));
        }
        let reference_tokens =
            self.sample_reference_tokens(references, height, width, seed as u64)?;
        self.edit_trace_from_reference_tokens_with_memory(
            instruction,
            negative_instruction,
            references,
            &reference_tokens,
            height,
            width,
            steps,
            cfg,
            seed,
            gs_key,
            renormalize,
            memory,
            cancel,
            on_progress,
        )
    }

    /// Resize all references to the target and sample their VAE posteriors in one seeded batch.
    pub fn sample_reference_tokens(
        &self,
        references: &[RgbImage],
        height: u32,
        width: u32,
        seed: u64,
    ) -> Result<Array> {
        if references.is_empty() {
            return Err(Error::Msg(
                "mage_flow edit: at least one reference image is required".into(),
            ));
        }
        let (gh, gw) = edit_latent_hw(height, width)?;
        let resized = references
            .iter()
            .map(|image| reference_nchw(image, height, width))
            .collect::<Result<Vec<_>>>()?;
        let resized_refs = resized.iter().collect::<Vec<_>>();
        let ref_batch = concatenate_axis(&resized_refs, 0)?;
        // The reference seeds once before encoding the complete reference batch, then samples one
        // posterior tensor. Keeping this as one call preserves that multi-reference RNG shape.
        self.vae
            .encode_sample(&ref_batch, seed)?
            .transpose_axes(&[0, 2, 3, 1])?
            .reshape(&[1, references.len() as i32 * gh * gw, LATENT_CHANNELS])?
            .as_dtype(Dtype::Bfloat16)
            .map_err(Into::into)
    }

    /// Run the instruction-edit algorithm from already sampled reference tokens.
    ///
    /// This boundary keeps the stochastic VAE posterior separate from denoise parity: a Torch
    /// oracle can replay its recorded posterior through MLX without pretending the two runtimes'
    /// seeded normal generators produce the same samples.
    #[allow(clippy::too_many_arguments)]
    pub fn edit_trace_from_reference_tokens(
        &self,
        instruction: &str,
        negative_instruction: &str,
        references: &[RgbImage],
        reference_tokens: &Array,
        height: u32,
        width: u32,
        steps: usize,
        cfg: f32,
        seed: i64,
        gs_key: &GsKey,
        renormalize: bool,
        on_progress: &mut dyn FnMut(Progress),
    ) -> Result<EditTrace> {
        self.edit_trace_from_reference_tokens_with_memory(
            instruction,
            negative_instruction,
            references,
            reference_tokens,
            height,
            width,
            steps,
            cfg,
            seed,
            gs_key,
            renormalize,
            None,
            &CancelFlag::default(),
            on_progress,
        )
    }

    #[allow(clippy::too_many_arguments)]
    pub fn edit_trace_from_reference_tokens_with_memory(
        &self,
        instruction: &str,
        negative_instruction: &str,
        references: &[RgbImage],
        reference_tokens: &Array,
        height: u32,
        width: u32,
        steps: usize,
        cfg: f32,
        seed: i64,
        gs_key: &GsKey,
        renormalize: bool,
        memory: Option<GenerationMemory>,
        cancel: &CancelFlag,
        on_progress: &mut dyn FnMut(Progress),
    ) -> Result<EditTrace> {
        if references.is_empty() {
            return Err(Error::Msg(
                "mage_flow edit: at least one reference image is required".into(),
            ));
        }
        let (gh, gw) = edit_latent_hw(height, width)?;
        let expected = [1, references.len() as i32 * gh * gw, LATENT_CHANNELS];
        if reference_tokens.shape() != expected {
            return Err(Error::Msg(format!(
                "mage_flow edit: reference tokens must be {expected:?}, got {:?}",
                reference_tokens.shape()
            )));
        }
        let reference_tokens = reference_tokens.as_dtype(Dtype::Bfloat16)?;
        let target = initial_tokens_for_grid(gh, gw, seed, gs_key, Dtype::Bfloat16)?;

        let positive = self.text_encoder.encode_edit(instruction, references)?;
        let cond = positive
            .txt
            .reshape(&[1, positive.seq_lens[0] as i32, -1])?;
        let negative = if uses_cfg(cfg) {
            let encoded = self
                .text_encoder
                .encode_edit(negative_instruction, references)?;
            Some((
                encoded.txt.reshape(&[1, encoded.seq_lens[0] as i32, -1])?,
                encoded.seq_lens[0] as i32,
            ))
        } else {
            None
        };
        let shapes =
            std::iter::repeat_n(ImgShape::latent(gh, gw), references.len() + 1).collect::<Vec<_>>();
        let total_tokens = (references.len() as i32 + 1) * gh * gw;
        let layout = PackLayout::new(
            shapes,
            vec![total_tokens],
            vec![positive.seq_lens[0] as i32],
        )?;
        let sigmas = mage_flow_sigmas(steps)?;
        let mut trajectories = Vec::with_capacity(2);
        let target = run_mage_mlx_driver(
            "mage_mlx_edit_driver",
            &self.transformer,
            target,
            gh * gw,
            Some(reference_tokens.clone()),
            &cond,
            layout,
            negative.as_ref().map(|(txt, len)| (txt, vec![*len])),
            cfg,
            renormalize,
            &sigmas,
            memory,
            cancel,
            Some(&mut trajectories),
            on_progress,
        )?;
        let image_u8 = decode_with_memory(&self.vae, &target, gh, gw, memory, cancel)?;
        Ok(EditTrace {
            final_tokens: target,
            reference_tokens,
            trajectories,
            image_u8,
        })
    }

    /// Generate a variable-geometry batch without resizing or bucket quantization.
    #[allow(clippy::too_many_arguments)]
    pub fn generate_batch(
        &self,
        samples: &[GenerationSample<'_>],
        steps: usize,
        cfg: f32,
        gs_key: &GsKey,
        renormalize: bool,
    ) -> Result<Vec<Array>> {
        Ok(self
            .generate_batch_trace(samples, steps, cfg, gs_key, renormalize, &mut |_| {})?
            .samples
            .into_iter()
            .map(|trace| trace.image_u8)
            .collect())
    }

    /// Generate variable-length native-resolution packs while retaining per-sample boundaries.
    #[allow(clippy::too_many_arguments)]
    pub fn generate_batch_trace(
        &self,
        samples: &[GenerationSample<'_>],
        steps: usize,
        cfg: f32,
        gs_key: &GsKey,
        renormalize: bool,
        on_progress: &mut dyn FnMut(Progress),
    ) -> Result<BatchGenerationTrace> {
        self.generate_batch_trace_with_memory(
            samples,
            steps,
            cfg,
            gs_key,
            renormalize,
            None,
            &CancelFlag::default(),
            on_progress,
        )
    }

    #[allow(clippy::too_many_arguments)]
    pub fn generate_batch_trace_with_memory(
        &self,
        samples: &[GenerationSample<'_>],
        steps: usize,
        cfg: f32,
        gs_key: &GsKey,
        renormalize: bool,
        memory: Option<GenerationMemory>,
        cancel: &CancelFlag,
        on_progress: &mut dyn FnMut(Progress),
    ) -> Result<BatchGenerationTrace> {
        let packs = plan_generation_packs(samples)?;
        let sigmas = mage_flow_sigmas(steps)?;
        let total_steps = steps
            .checked_mul(packs.len())
            .ok_or_else(|| Error::Msg("mage_flow: progress step count overflow".into()))?;
        let mut traces = Vec::with_capacity(samples.len());

        for (pack_index, pack) in packs.iter().enumerate() {
            let pack_samples = &samples[pack.sample_range.clone()];
            let mut texts = pack_samples
                .iter()
                .map(|sample| sample.prompt)
                .collect::<Vec<_>>();
            if uses_cfg(cfg) {
                texts.extend(pack_samples.iter().map(|sample| sample.negative_prompt));
            }
            let conditioning = self.text_encoder.encode(&texts, PromptKind::Gen)?;
            let sample_count = pack_samples.len();
            let positive_lens = conditioning.seq_lens[..sample_count]
                .iter()
                .map(|&len| len as i32)
                .collect::<Vec<_>>();
            let positive_tokens: i32 = positive_lens.iter().sum();
            let hidden = self.transformer.config().context_in_dim;
            let (cond_flat, neg_flat) = if uses_cfg(cfg) {
                let parts = conditioning.txt.split_axis(&[positive_tokens], 0)?;
                (parts[0].clone(), Some(parts[1].clone()))
            } else {
                (conditioning.txt.clone(), None)
            };
            let cond = cond_flat.reshape(&[1, positive_tokens, hidden])?;
            let negative_lens = if uses_cfg(cfg) {
                Some(
                    conditioning.seq_lens[sample_count..]
                        .iter()
                        .map(|&len| len as i32)
                        .collect::<Vec<_>>(),
                )
            } else {
                None
            };
            let negative = match (neg_flat, negative_lens) {
                (Some(txt), Some(lens)) => {
                    let tokens = lens.iter().sum();
                    Some((txt.reshape(&[1, tokens, hidden])?, lens))
                }
                _ => None,
            };

            let initial = pack_samples
                .iter()
                .map(|sample| {
                    initial_tokens(
                        sample.height,
                        sample.width,
                        sample.seed,
                        gs_key,
                        Dtype::Bfloat16,
                    )
                })
                .collect::<Result<Vec<_>>>()?;
            let initial_refs = initial.iter().collect::<Vec<_>>();
            let tokens = concatenate_axis(&initial_refs, 1)?;
            let layout = generation_layout(&pack.grids, positive_lens)?;
            let mut trajectories = Vec::with_capacity(2);
            let out = denoise_capture_with_progress(
                &self.transformer,
                tokens,
                &cond,
                layout,
                negative.as_ref().map(|(txt, lens)| (txt, lens.clone())),
                cfg,
                renormalize,
                &sigmas,
                memory,
                cancel,
                &mut |step, latent| {
                    if sample_count == 1 && step < 2 {
                        trajectories.push(latent.clone());
                    }
                },
                &mut |progress_event| match progress_event {
                    Progress::Step { current, .. } => on_progress(Progress::Step {
                        current: (pack_index * steps) as u32 + current,
                        total: total_steps as u32,
                    }),
                    other => on_progress(other),
                },
            )?;

            let boundaries = pack
                .grids
                .iter()
                .scan(0, |offset, &(gh, gw)| {
                    *offset += gh * gw;
                    Some(*offset)
                })
                .collect::<Vec<_>>();
            let split_points = &boundaries[..boundaries.len().saturating_sub(1)];
            let outputs = out.split_axis(split_points, 1)?;
            for ((tokens, &(gh, gw)), sample) in
                outputs.into_iter().zip(pack.grids.iter()).zip(pack_samples)
            {
                let final_latent = unpack_tokens(&tokens, gh, gw)?;
                on_progress(Progress::Decoding);
                let image_u8 = decode_with_memory(&self.vae, &tokens, gh, gw, memory, cancel)?;
                debug_assert_eq!(
                    image_u8.shape(),
                    [sample.height as i32, sample.width as i32, 3]
                );
                traces.push(GenerationTrace {
                    final_tokens: tokens,
                    final_latent,
                    trajectories: if sample_count == 1 {
                        std::mem::take(&mut trajectories)
                    } else {
                        Vec::new()
                    },
                    image_u8,
                });
            }
        }
        Ok(BatchGenerationTrace {
            samples: traces,
            packs,
        })
    }
}

pub fn assemble_edit_sequence(target: &Array, references: &Array) -> Result<Array> {
    if target.ndim() != 3
        || references.ndim() != 3
        || target.shape()[0] != 1
        || references.shape()[0] != 1
        || target.shape()[2] != references.shape()[2]
    {
        return Err(Error::Msg(format!(
            "mage_flow edit: target/reference token shapes are incompatible: {:?} / {:?}",
            target.shape(),
            references.shape()
        )));
    }
    concatenate_axis(&[target, references], 1).map_err(Into::into)
}

fn verify_quantized_counts(component: &str, packed: usize, expected: usize) -> Result<()> {
    if packed != expected {
        return Err(Error::Msg(format!(
            "mage_flow: {component} quantization packed {packed}/{expected} required projections"
        )));
    }
    Ok(())
}

/// Validate exact requested geometry and greedily preserve input order under the 50k-token budget.
pub fn plan_generation_packs(samples: &[GenerationSample<'_>]) -> Result<Vec<GenerationPack>> {
    if samples.is_empty() {
        return Err(Error::Msg(
            "mage_flow: generation batch must not be empty".into(),
        ));
    }
    let mut packs = Vec::new();
    let mut start = 0;
    let mut grids = Vec::new();
    let mut tokens = 0;
    for (index, sample) in samples.iter().enumerate() {
        let grid = latent_hw(sample.height, sample.width)?;
        let sample_tokens = grid.0 * grid.1;
        if !grids.is_empty() && tokens + sample_tokens > MAX_PACKED_IMAGE_TOKENS {
            packs.push(GenerationPack {
                sample_range: start..index,
                grids: std::mem::take(&mut grids),
                image_tokens: tokens,
            });
            start = index;
            tokens = 0;
        }
        grids.push(grid);
        tokens += sample_tokens;
    }
    packs.push(GenerationPack {
        sample_range: start..samples.len(),
        grids,
        image_tokens: tokens,
    });
    Ok(packs)
}

/// Build the exact diffusers static-shift schedule used by Mage-Flow.
///
/// The input ladder is `linspace(1, 1/N, N)`, each value is mapped through
/// `shift*s/(1+(shift-1)*s)`, and the terminal zero is appended.
pub fn mage_flow_sigmas(steps: usize) -> Result<Vec<f32>> {
    if steps == 0 {
        return Err(Error::Msg(
            "mage_flow: steps must be greater than zero".into(),
        ));
    }
    // Match ATen's endpoint-aware f32 linspace fill before applying Diffusers' f32 static shift.
    // A one-direction `start + i*step` accumulates several one-ULP errors late in the 20-step
    // ladder, which feed back through every subsequent nonlinear DiT call.
    let n = steps as f32;
    let start = 1.0f32;
    let end = 1.0 / n;
    let step = if steps > 1 {
        (end - start) / (steps - 1) as f32
    } else {
        0.0
    };
    let mut sigmas = (0..steps)
        .map(|i| {
            // ATen fills from the nearer endpoint on each half, avoiding the extra endpoint error
            // a one-direction `start + i*step` accumulates.
            let s = if i < steps / 2 {
                step.mul_add(i as f32, start)
            } else {
                (-step).mul_add((steps - i - 1) as f32, end)
            };
            STATIC_SHIFT * s / (1.0 + (STATIC_SHIFT - 1.0) * s)
        })
        .collect::<Vec<_>>();
    sigmas.push(0.0);
    Ok(sigmas)
}

/// Validate and convert output pixels to Mage's latent grid.
pub fn latent_hw(height: u32, width: u32) -> Result<(i32, i32)> {
    if !(MIN_SIZE..=MAX_SIZE).contains(&height) || !(MIN_SIZE..=MAX_SIZE).contains(&width) {
        return Err(Error::Msg(format!(
            "mage_flow: {width}x{height} must be within {MIN_SIZE}..={MAX_SIZE} per side"
        )));
    }
    if !height.is_multiple_of(SIZE_MULTIPLE) || !width.is_multiple_of(SIZE_MULTIPLE) {
        return Err(Error::Msg(format!(
            "mage_flow: {width}x{height} must be divisible by {SIZE_MULTIPLE}"
        )));
    }
    Ok((
        (height / VAE_DOWNSAMPLE_FACTOR) as i32,
        (width / VAE_DOWNSAMPLE_FACTOR) as i32,
    ))
}

/// Convert edit pixels to the latent grid.
///
/// The public model descriptor still enforces the production 512..=2048 range. The reference
/// parity oracle deliberately uses 256² to keep its 4.1B-parameter CPU run tractable, so the
/// lower-level edit trace accepts that frozen test geometry as well.
fn edit_latent_hw(height: u32, width: u32) -> Result<(i32, i32)> {
    if height > MAX_SIZE || width > MAX_SIZE || height < 256 || width < 256 {
        return Err(Error::Msg(format!(
            "mage_flow edit: {width}x{height} must be within 256..={MAX_SIZE} per side"
        )));
    }
    if !height.is_multiple_of(SIZE_MULTIPLE) || !width.is_multiple_of(SIZE_MULTIPLE) {
        return Err(Error::Msg(format!(
            "mage_flow edit: {width}x{height} must be divisible by {SIZE_MULTIPLE}"
        )));
    }
    Ok((
        (height / VAE_DOWNSAMPLE_FACTOR) as i32,
        (width / VAE_DOWNSAMPLE_FACTOR) as i32,
    ))
}

fn reference_nchw(image: &RgbImage, height: u32, width: u32) -> Result<Array> {
    let resized = if image.dimensions() == (width, height) {
        image.clone()
    } else {
        image::imageops::resize(image, width, height, FilterType::CatmullRom)
    };
    let mut values = vec![0f32; 3 * height as usize * width as usize];
    for channel in 0..3usize {
        for y in 0..height as usize {
            for x in 0..width as usize {
                let pixel = resized.get_pixel(x as u32, y as u32)[channel] as f32;
                values[(channel * height as usize + y) * width as usize + x] = pixel / 127.5 - 1.0;
            }
        }
    }
    Ok(Array::from_slice(
        &values,
        &[1, 3, height as i32, width as i32],
    ))
}

/// Create the required Gaussian-Shading initial latent and flatten it to image tokens.
pub fn initial_tokens(
    height: u32,
    width: u32,
    seed: i64,
    key: &GsKey,
    dtype: Dtype,
) -> Result<Array> {
    let (gh, gw) = latent_hw(height, width)?;
    initial_tokens_for_grid(gh, gw, seed, key, dtype)
}

fn initial_tokens_for_grid(
    gh: i32,
    gw: i32,
    seed: i64,
    key: &GsKey,
    dtype: Dtype,
) -> Result<Array> {
    mlx_gen::image_execution::policy_debug("mage.resolved_gs_key", key);
    encode_noise(
        (LATENT_CHANNELS as usize, gh as usize, gw as usize),
        key,
        seed,
        dtype,
    )?
    .transpose_axes(&[0, 2, 3, 1])?
    .reshape(&[1, gh * gw, LATENT_CHANNELS])
    .map_err(Into::into)
}

/// Gaussian-Shading initial-latent tokens for an explicit latent grid, bypassing the production
/// size-range check [`latent_hw`] enforces. Used by the LoRA trainer's preview render (sc-14055),
/// which samples at the (possibly sub-512) training resolution rather than a production geometry.
pub fn encode_noise_tokens(
    gh: i32,
    gw: i32,
    seed: i64,
    key: &GsKey,
    dtype: Dtype,
) -> Result<Array> {
    initial_tokens_for_grid(gh, gw, seed, key, dtype)
}

/// Convert one packed token stream back to the NCHW latent consumed by Mage-VAE.
pub fn unpack_tokens(tokens: &Array, gh: i32, gw: i32) -> Result<Array> {
    if tokens.shape() != [1, gh * gw, LATENT_CHANNELS] {
        return Err(Error::Msg(format!(
            "mage_flow: expected tokens [1, {}, {}], got {:?}",
            gh * gw,
            LATENT_CHANNELS,
            tokens.shape()
        )));
    }
    Ok(tokens
        .reshape(&[1, gh, gw, LATENT_CHANNELS])?
        .transpose_axes(&[0, 3, 1, 2])?)
}

/// Combine conditional and unconditional velocity exactly as the reference.
pub fn cfg_velocity(cond: &Array, unc: &Array, cfg: f32, renormalize: bool) -> Result<Array> {
    if cond.shape() != unc.shape() {
        return Err(Error::Msg("mage_flow: CFG velocity shapes differ".into()));
    }
    // Torch keeps tensor/scalar CFG arithmetic at the velocity tensor's dtype. An f32 MLX
    // scalar would promote the guided velocity, causing the scheduler to retain f32 latents
    // after its documented cast back to `model_output.dtype`.
    let scale = Array::from_slice(&[cfg], &[1]).as_dtype(cond.dtype())?;
    let guided = unc.add(&cond.subtract(unc)?.multiply(&scale)?)?;
    if !renormalize {
        return guided.as_dtype(cond.dtype()).map_err(Into::into);
    }
    // Per-token L2 norm over channels, matching torch.norm(..., dim=-1, keepdim=True).
    let cond_norm = cond.multiply(cond)?.sum_axis(-1, true)?.sqrt()?;
    let guided_norm = guided
        .multiply(&guided)?
        .sum_axis(-1, true)?
        .sqrt()?
        .add(Array::from_slice(&[1e-6f32], &[1]))?;
    Ok(guided
        .multiply(&cond_norm.divide(&guided_norm)?)?
        .as_dtype(cond.dtype())?)
}

/// One deterministic Diffusers `FlowMatchEulerDiscreteScheduler.step`.
fn flow_euler_step(sample: &Array, model_output: &Array, delta: f32) -> Result<Array> {
    let model_dtype = model_output.dtype();
    Ok(sample
        .as_dtype(Dtype::Float32)?
        .add(
            &model_output
                .as_dtype(Dtype::Float32)?
                .multiply(Array::from_slice(&[delta], &[1]))?,
        )?
        .as_dtype(model_dtype)?)
}

/// Forward-evaluation seam for [`MageMlxDriver`] (`PIPE-7`, `DEC-13`).
pub trait MageMlxForward {
    /// Whether this forward evaluates fused `[cond, uncond]` CFG branches along axis 1.
    fn uses_cfg(&self) -> bool;

    /// Predict transformer velocity for `fused_or_single_sequence` at noise level `sigma` and
    /// outer step `step`.
    fn forward_sequence(
        &self,
        fused_or_single_sequence: &Array,
        sigma: f32,
        step: usize,
        cancel: &CancelFlag,
    ) -> Result<Array>;
}

struct TransformerMageMlxForward<'a> {
    transformer: &'a MageTransformer,
    cond_txt: &'a Array,
    cond_ctx: PackContext,
    negative: Option<(Array, PackContext)>,
    memory: Option<GenerationMemory>,
}

impl MageMlxForward for TransformerMageMlxForward<'_> {
    fn uses_cfg(&self) -> bool {
        self.negative.is_some()
    }

    fn forward_sequence(
        &self,
        sequence: &Array,
        sigma: f32,
        _step: usize,
        cancel: &CancelFlag,
    ) -> Result<Array> {
        if let Some((fused_txt, fused_ctx)) = &self.negative {
            let sigma_arr = Array::from_slice(
                &vec![sigma; fused_ctx.segments()],
                &[fused_ctx.segments() as i32],
            );
            transformer_forward(
                self.transformer,
                sequence,
                fused_txt,
                &sigma_arr,
                fused_ctx,
                self.memory,
                cancel,
            )
        } else {
            let sigma_arr = Array::from_slice(
                &vec![sigma; self.cond_ctx.segments()],
                &[self.cond_ctx.segments() as i32],
            );
            transformer_forward(
                self.transformer,
                sequence,
                self.cond_txt,
                &sigma_arr,
                &self.cond_ctx,
                self.memory,
                cancel,
            )
        }
    }
}

struct MageMlxDriverHistory {
    total_steps: usize,
    sigmas_bits: Vec<u32>,
    expected_next_step: usize,
}

/// Typed [`FamilyDriver`] for MLX Mage-Flow rectified-flow Euler denoising across packed T2I
/// batches and target-first clean-reference Edit sequences (`PIPE-7`, `DEC-13`).
pub struct MageMlxDriver<F> {
    sigmas: Vec<f32>,
    sigmas_bits: Vec<u32>,
    cfg: f32,
    renormalize: bool,
    target_tokens: i32,
    reference_tokens: Option<Array>,
    forward: F,
    support: DriverSupport,
}

impl<F: MageMlxForward> MageMlxDriver<F> {
    pub fn build_support(driver_id: &'static str) -> DriverSupport {
        DriverSupport {
            driver_id,
            arbitrary_slice: true,
            carry_history: true,
            reset_history: true,
            renoise_from_clean: false,
            geometry_change: false,
            read_only_boundary_estimate: false,
            previous_estimate_capture: false,
            uses_global_rng: false,
            refusal_reasons: DriverRefusalReasons {
                renoise_from_clean:
                    "native MLX Mage flow driver does not re-noise from clean estimates",
                geometry_change:
                    "native MLX Mage flow driver does not support geometry-changing phase boundaries",
                read_only_boundary_estimate:
                    "native MLX Mage flow driver does not expose read-only boundary clean estimates",
                ..DriverRefusalReasons::default()
            },
        }
    }

    pub fn new(
        driver_id: &'static str,
        sigmas: Vec<f32>,
        cfg: f32,
        renormalize: bool,
        target_tokens: i32,
        reference_tokens: Option<Array>,
        forward: F,
    ) -> Result<Self> {
        if sigmas.len() < 2 {
            return Err(Error::Msg(
                "mage_flow: scheduler needs a terminal sigma".into(),
            ));
        }
        if target_tokens <= 0 {
            return Err(Error::Msg(
                "mage_flow: target token count must be positive".into(),
            ));
        }
        let sigmas_bits = sigmas.iter().map(|s| s.to_bits()).collect();
        Ok(Self {
            sigmas,
            sigmas_bits,
            cfg,
            renormalize,
            target_tokens,
            reference_tokens,
            forward,
            support: Self::build_support(driver_id),
        })
    }
}

impl<F: MageMlxForward> FamilyDriver<Array> for MageMlxDriver<F> {
    fn support(&self) -> &DriverSupport {
        &self.support
    }

    fn run_slice(
        &self,
        input: SliceInput<'_, Array>,
        state: &mut DriverRunState,
        hooks: &mut StepHooks<'_, Array>,
    ) -> mlx_gen::gen_core::Result<BoundaryState<Array>> {
        self.support.validate_slice(&input, state)?;
        let total_steps = self.sigmas.len() - 1;
        if input.total_steps != total_steps || input.span.end > total_steps {
            return Err(mlx_gen::gen_core::Error::Unsupported(format!(
                "{}: slice {:?} / total_steps={} disagrees with Mage schedule length {total_steps}",
                self.support.driver_id, input.span, input.total_steps
            )));
        }

        let carrying = input.transition == SliceTransition::Continue
            && input.history == DriverHistoryPolicy::Carry;
        if carrying {
            let Some(existing) = state.history_ref::<MageMlxDriverHistory>() else {
                return Err(mlx_gen::gen_core::Error::Unsupported(format!(
                    "{}: Carry requested without prior Mage driver history in DriverRunState",
                    self.support.driver_id
                )));
            };
            if existing.total_steps != input.total_steps || existing.sigmas_bits != self.sigmas_bits
            {
                return Err(mlx_gen::gen_core::Error::Unsupported(format!(
                    "{}: Carry refused across incompatible Mage sigma schedule",
                    self.support.driver_id
                )));
            }
            if existing.expected_next_step != input.span.start {
                return Err(mlx_gen::gen_core::Error::Unsupported(format!(
                    "{}: Carry refused at non-contiguous step {} (expected {})",
                    self.support.driver_id, input.span.start, existing.expected_next_step
                )));
            }
        }

        state.enter_slice(&input)?;
        if !carrying {
            state.set_history(MageMlxDriverHistory {
                total_steps,
                sigmas_bits: self.sigmas_bits.clone(),
                expected_next_step: input.span.start,
            });
        }

        let mut target = match input.state {
            BoundaryState::XSigma(x)
            | BoundaryState::CleanEstimate { latent: x, .. }
            | BoundaryState::RouteSpecific { state: x, .. } => x,
        };
        if target.shape().get(1).copied() != Some(self.target_tokens) {
            return Err(mlx_gen::gen_core::Error::Msg(format!(
                "{}: expected target tokens length {}, got {:?}",
                self.support.driver_id,
                self.target_tokens,
                target.shape()
            )));
        }

        let use_cfg = self.forward.uses_cfg();
        for step in input.span.clone() {
            if input.cancel.is_cancelled() {
                return Err(mlx_gen::gen_core::Error::Canceled);
            }
            let info = DriverStepInfo {
                pass: input.pass,
                phase: input.phase,
                outer_step: step,
                total_outer_steps: input.total_steps,
                eval_in_step: 0,
                kind: DriverEvaluationKind::Trajectory,
                sigma: self.sigmas[step],
                next_sigma: self.sigmas[step + 1],
            };
            hooks.observe(&info, &target)?;

            let sequence = if let Some(refs) = &self.reference_tokens {
                assemble_edit_sequence(&target, refs).map_err(mlx_gen::gen_core::Error::from)?
            } else {
                target.clone()
            };
            let seq_tokens = sequence.shape()[1];
            let model_input = if use_cfg {
                concatenate_axis(&[&sequence, &sequence], 1)
                    .map_err(|e| mlx_gen::gen_core::Error::Msg(e.to_string()))?
            } else {
                sequence
            };

            let raw_out = self
                .forward
                .forward_sequence(&model_input, self.sigmas[step], step, input.cancel)
                .map_err(mlx_gen::gen_core::Error::from)?;
            state.record_evaluation(DriverEvaluationKind::Trajectory);

            let velocity = if use_cfg {
                let parts = raw_out
                    .split_axis(&[seq_tokens], 1)
                    .map_err(|e| mlx_gen::gen_core::Error::Msg(e.to_string()))?;
                cfg_velocity(&parts[0], &parts[1], self.cfg, self.renormalize)
                    .map_err(mlx_gen::gen_core::Error::from)?
            } else {
                raw_out
            };

            let target_velocity = if self.reference_tokens.is_some() {
                velocity
                    .split_axis(&[self.target_tokens], 1)
                    .map_err(|e| mlx_gen::gen_core::Error::Msg(e.to_string()))?
                    .swap_remove(0)
            } else {
                velocity
            };

            // Diffusers' FlowMatchEulerDiscreteScheduler upcasts the sample, performs the complete
            // Euler addition in f32, then casts the result back to model_output.dtype.
            target = flow_euler_step(
                &target,
                &target_velocity,
                self.sigmas[step + 1] - self.sigmas[step],
            )
            .map_err(mlx_gen::gen_core::Error::from)?;
            target = hooks.post_step(&info, target)?;
            mlx_rs::transforms::eval([&target])
                .map_err(|e| mlx_gen::gen_core::Error::Msg(e.to_string()))?;
            hooks.materialize(&info, &target)?;
            if let Some(history) = state.history_mut::<MageMlxDriverHistory>() {
                history.expected_next_step = step + 1;
            }
            state.record_outer_step(step);
            hooks.emit_progress(&info);
        }

        state.finish_slice(input.span.end);
        Ok(BoundaryState::XSigma(target))
    }
}

#[allow(clippy::too_many_arguments)]
fn run_mage_mlx_driver(
    driver_id: &'static str,
    transformer: &MageTransformer,
    initial_target: Array,
    target_tokens: i32,
    reference_tokens: Option<Array>,
    cond_txt: &Array,
    cond_layout: PackLayout,
    negative: Option<(&Array, Vec<i32>)>,
    cfg: f32,
    renormalize: bool,
    sigmas: &[f32],
    memory: Option<GenerationMemory>,
    cancel: &CancelFlag,
    mut capture_trajectories: Option<&mut Vec<Array>>,
    on_progress: &mut dyn FnMut(Progress),
) -> Result<Array> {
    if sigmas.len() < 2 {
        return Err(Error::Msg(
            "mage_flow: scheduler needs a terminal sigma".into(),
        ));
    }
    let use_cfg = uses_cfg(cfg);
    if use_cfg && negative.is_none() {
        return Err(Error::Msg(
            "mage_flow: cfg > 1 requires negative conditioning".into(),
        ));
    }
    let cond_ctx = transformer.pack_context(cond_layout.clone())?;
    let neg_packed = if use_cfg {
        let (neg_txt, neg_lens) = negative.as_ref().unwrap();
        let fused_ctx = transformer.pack_context(cond_layout.fused_cfg(neg_lens)?)?;
        let fused_txt = concatenate_axis(&[cond_txt, neg_txt], 1)?;
        Some((fused_txt, fused_ctx))
    } else {
        None
    };
    let forward = TransformerMageMlxForward {
        transformer,
        cond_txt,
        cond_ctx,
        negative: neg_packed,
        memory,
    };
    let driver = MageMlxDriver::new(
        driver_id,
        sigmas.to_vec(),
        cfg,
        renormalize,
        target_tokens,
        reference_tokens.clone(),
        forward,
    )?;
    let total_steps = sigmas.len() - 1;
    let mut state = DriverRunState::new();
    let mut observe = |info: &DriverStepInfo, target: &Array| -> mlx_gen::gen_core::Result<()> {
        if let Some(trajectories) = capture_trajectories.as_deref_mut() {
            if info.outer_step < 2 {
                let seq = if let Some(refs) = &reference_tokens {
                    assemble_edit_sequence(target, refs).map_err(mlx_gen::gen_core::Error::from)?
                } else {
                    target.clone()
                };
                let observed = if use_cfg {
                    concatenate_axis(&[&seq, &seq], 1)
                        .map_err(|e| mlx_gen::gen_core::Error::Msg(e.to_string()))?
                } else {
                    seq
                };
                trajectories.push(observed);
            }
        }
        Ok(())
    };
    let mut hooks = StepHooks::new(on_progress);
    hooks.on_observe = Some(&mut observe);
    let out = run_mlx_driver_slices(
        &driver,
        [SliceSpec::full(total_steps)],
        BoundaryState::XSigma(initial_target),
        total_steps,
        0,
        cancel,
        None,
        &mut state,
        &mut hooks,
        |_, out, _| Ok(out),
    )?;
    let BoundaryState::XSigma(final_target) = out else {
        unreachable!("MageMlxDriver always emits BoundaryState::XSigma");
    };
    Ok(final_target)
}

/// Run Mage's rectified-flow Euler loop over already-packed text and image streams.
///
/// At `cfg > 1`, conditional and unconditional branches are packed into one transformer
/// forward. [`PackLayout::fused_cfg`] deliberately preserves the reference's shifted frame index
/// for the appended unconditional image shapes.
#[allow(clippy::too_many_arguments)]
pub fn denoise(
    transformer: &MageTransformer,
    img: Array,
    cond_txt: &Array,
    cond_layout: PackLayout,
    negative: Option<(&Array, Vec<i32>)>,
    cfg: f32,
    renormalize: bool,
    sigmas: &[f32],
) -> Result<Array> {
    denoise_capture_with_progress(
        transformer,
        img,
        cond_txt,
        cond_layout,
        negative,
        cfg,
        renormalize,
        sigmas,
        None,
        &CancelFlag::default(),
        &mut |_, _| {},
        &mut |_| {},
    )
}

#[allow(clippy::too_many_arguments)]
fn denoise_capture_with_progress(
    transformer: &MageTransformer,
    img: Array,
    cond_txt: &Array,
    cond_layout: PackLayout,
    negative: Option<(&Array, Vec<i32>)>,
    cfg: f32,
    renormalize: bool,
    sigmas: &[f32],
    memory: Option<GenerationMemory>,
    cancel: &CancelFlag,
    on_step: &mut dyn FnMut(usize, &Array),
    on_progress: &mut dyn FnMut(Progress),
) -> Result<Array> {
    let target_tokens = img.shape()[1];
    let mut trajectories = Vec::new();
    let out = run_mage_mlx_driver(
        "mage_mlx_generation_driver",
        transformer,
        img,
        target_tokens,
        None,
        cond_txt,
        cond_layout,
        negative,
        cfg,
        renormalize,
        sigmas,
        memory,
        cancel,
        Some(&mut trajectories),
        on_progress,
    )?;
    for (step, latent) in trajectories.iter().enumerate() {
        on_step(step, latent);
    }
    Ok(out)
}

/// Decode a final token stream and apply the reference's clamp/range conversion.
pub fn decode(vae: &MageVae, tokens: &Array, gh: i32, gw: i32) -> Result<Array> {
    decode_with_memory(vae, tokens, gh, gw, None, &CancelFlag::default())
}

fn decode_with_memory(
    vae: &MageVae,
    tokens: &Array,
    gh: i32,
    gw: i32,
    memory: Option<GenerationMemory>,
    cancel: &CancelFlag,
) -> Result<Array> {
    let latent = unpack_tokens(tokens, gh, gw)?;
    let pixels = match memory.filter(|m| m.tile_vae_decode) {
        Some(memory) => vae.decode_tiled(
            &latent,
            &TilingConfig {
                spatial: Some(SpatialTiling {
                    tile_px: memory.decode_tile_edge.unwrap_or(512) as i32,
                    overlap_px: memory.decode_overlap.unwrap_or(256) as i32,
                }),
                temporal: None,
            },
            Some(cancel),
        )?,
        None => vae.decode(&latent)?,
    };
    let pixels = maximum(&pixels, Array::from_slice(&[-1.0f32], &[1]))?;
    let pixels = minimum(&pixels, Array::from_slice(&[1.0f32], &[1]))?;
    let scaled = pixels
        .add(Array::from_slice(&[1.0f32], &[1]))?
        .multiply(Array::from_slice(&[127.5f32], &[1]))?;
    Ok(scaled
        .as_dtype(Dtype::Uint8)?
        .transpose_axes(&[0, 2, 3, 1])?
        .reshape(&[
            gh * VAE_DOWNSAMPLE_FACTOR as i32,
            gw * VAE_DOWNSAMPLE_FACTOR as i32,
            3,
        ])?)
}

fn transformer_forward(
    transformer: &MageTransformer,
    img: &Array,
    txt: &Array,
    sigma: &Array,
    ctx: &PackContext,
    memory: Option<GenerationMemory>,
    cancel: &CancelFlag,
) -> Result<Array> {
    let attention = match memory.filter(|m| m.chunk_attention) {
        Some(memory) => AttentionPlan::budgeted(AttentionBudget::from_score_elements(
            memory.attention_chunk_size.unwrap_or(16_777_216) as u64,
            true,
        ))
        .with_cancel(cancel),
        None => AttentionPlan::UNBOUNDED,
    };
    let window = transformer.block_window(
        memory
            .filter(|m| m.stream_transformer_blocks)
            .and_then(|m| m.transformer_window_size)
            .map(|size| size as usize),
        cancel,
    )?;
    transformer.forward_with_rungs(img, txt, sigma, ctx, attention, window)
}

/// Generation layout for a list of latent grids and encoded prompt lengths.
pub fn generation_layout(grids: &[(i32, i32)], txt_lens: Vec<i32>) -> Result<PackLayout> {
    PackLayout::generation(
        grids.iter().map(|&(h, w)| ImgShape::latent(h, w)).collect(),
        txt_lens,
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn rgb8_conversion_truncates_like_torch_byte() {
        let pixels = Array::from_slice(&[-1.0f32, -0.5, 0.0, 0.5, 1.0], &[1, 1, 1, 5]);
        let scaled = pixels
            .add(Array::from_slice(&[1.0f32], &[1]))
            .unwrap()
            .multiply(Array::from_slice(&[127.5f32], &[1]))
            .unwrap()
            .as_dtype(Dtype::Uint8)
            .unwrap();
        mlx_rs::transforms::eval([&scaled]).unwrap();
        assert_eq!(scaled.as_slice::<u8>(), &[0, 63, 127, 191, 255]);
    }

    #[test]
    fn edit_stream_is_target_first_and_references_remain_clean() {
        let target = Array::from_slice(&[1f32, 2.0, 3.0, 4.0], &[1, 2, 2]);
        let refs = Array::from_slice(&[10f32, 11.0, 20.0, 21.0], &[1, 2, 2]);
        let first = assemble_edit_sequence(&target, &refs).unwrap();
        assert_eq!(
            first.as_slice::<f32>(),
            &[1.0, 2.0, 3.0, 4.0, 10.0, 11.0, 20.0, 21.0]
        );
        let stepped_target = Array::from_slice(&[5f32, 6.0, 7.0, 8.0], &[1, 2, 2]);
        let second = assemble_edit_sequence(&stepped_target, &refs).unwrap();
        assert_eq!(
            &second.as_slice::<f32>()[4..],
            &first.as_slice::<f32>()[4..],
            "clean references must be re-concatenated unchanged at every step"
        );
        assert_ne!(
            &second.as_slice::<f32>()[..4],
            &first.as_slice::<f32>()[..4],
            "only target tokens advance"
        );
        assert!(
            assemble_edit_sequence(&target, &Array::from_slice(&[0f32; 3], &[1, 1, 3])).is_err()
        );
    }

    #[test]
    fn edit_layout_has_n_plus_one_shapes_in_one_attention_window() {
        let shape = ImgShape::latent(4, 3);
        let layout = PackLayout::new(vec![shape, shape, shape], vec![36], vec![9]).unwrap();
        assert_eq!(layout.segments(), 1);
        assert_eq!(layout.img_shapes().len(), 3);
        assert_eq!(layout.img_lens(), &[36]);
        assert!(
            PackLayout::new(vec![ImgShape::latent(4, 3); 3], vec![12, 24], vec![9],).is_err(),
            "splitting one edit sample into per-image attention windows must fail"
        );
    }

    #[test]
    fn exact_twenty_step_static_shift_schedule() {
        let s = mage_flow_sigmas(20).unwrap();
        assert_eq!(
            s,
            vec![
                1.0, 0.99130434, 0.98181814, 0.97142863, 0.96000004, 0.94736844, 0.9333333,
                0.917647, 0.90000004, 0.88000005, 0.85714287, 0.83076924, 0.8, 0.76363635, 0.72,
                0.6666667, 0.6, 0.51428574, 0.4, 0.24000001, 0.0,
            ],
            "every sigma must match the torch/diffusers CPU oracle bit-for-bit"
        );
        assert!(s.windows(2).all(|w| w[0] > w[1]));
    }

    #[test]
    fn exact_thirty_step_base_static_shift_schedule() {
        let s = mage_flow_sigmas(30).unwrap();
        assert_eq!(
            s,
            vec![
                1.0, 0.9942857, 0.9882353, 0.98181814, 0.97499996, 0.96774185, 0.96000004,
                0.9517242, 0.9428571, 0.9333334, 0.92307687, 0.912, 0.90000004, 0.8869566,
                0.87272733, 0.85714287, 0.8399999, 0.8210527, 0.79999995, 0.77647054, 0.75,
                0.71999997, 0.68571424, 0.6461538, 0.59999996, 0.54545456, 0.48, 0.39999998, 0.3,
                0.17142859, 0.0,
            ],
            "all 31 Base sigmas must match the frozen Torch/Diffusers CPU oracle bit-for-bit"
        );
        assert!(s.windows(2).all(|w| w[0] > w[1]));
    }

    #[test]
    fn turbo_uses_the_same_shifted_four_step_ladder_and_no_cfg_branch() {
        let sigmas = mage_flow_sigmas(4).unwrap();
        assert_eq!(sigmas.len(), 5);
        let expected = [1.0, 0.94736844, 0.85714287, 0.6666667, 0.0];
        for (index, (got, want)) in sigmas.iter().zip(expected).enumerate() {
            assert!(
                (got - want).abs() < 1e-6,
                "Turbo sigma {index}: got {got}, want {want}"
            );
        }
        assert!(!uses_cfg(1.0));
        assert!(uses_cfg(1.0001));
    }

    #[test]
    fn cfg_and_optional_norm_match_reference_formula() {
        let cond = Array::from_slice(&[3.0f32, 4.0], &[1, 1, 2]);
        let unc = Array::from_slice(&[1.0f32, 2.0], &[1, 1, 2]);
        let plain = cfg_velocity(&cond, &unc, 5.0, false).unwrap();
        assert_eq!(plain.as_slice::<f32>(), &[11.0, 12.0]);
        let normed = cfg_velocity(&cond, &unc, 5.0, true).unwrap();
        let norm =
            (normed.as_slice::<f32>()[0].powi(2) + normed.as_slice::<f32>()[1].powi(2)).sqrt();
        assert!((norm - 5.0).abs() < 1e-5);

        let cond_bf16 = cond.as_dtype(Dtype::Bfloat16).unwrap();
        let unc_bf16 = unc.as_dtype(Dtype::Bfloat16).unwrap();
        assert_eq!(
            cfg_velocity(&cond_bf16, &unc_bf16, 5.0, false)
                .unwrap()
                .dtype(),
            Dtype::Bfloat16,
            "Torch scalar CFG arithmetic preserves the model-output dtype"
        );
    }

    #[test]
    fn euler_add_rounds_only_after_the_f32_add() {
        let sample = Array::from_slice(&[1.390625f32], &[1])
            .as_dtype(Dtype::Bfloat16)
            .unwrap();
        let velocity = Array::from_slice(&[1.859375f32], &[1])
            .as_dtype(Dtype::Bfloat16)
            .unwrap();
        let delta = -0.05263156f32;
        let got = flow_euler_step(&sample, &velocity, delta).unwrap();
        let pre_add_rounded = sample
            .add(
                velocity
                    .as_dtype(Dtype::Float32)
                    .unwrap()
                    .multiply(Array::from_slice(&[delta], &[1]))
                    .unwrap()
                    .as_dtype(Dtype::Bfloat16)
                    .unwrap(),
            )
            .unwrap();
        mlx_rs::transforms::eval([&got, &pre_add_rounded]).unwrap();
        let got_f32 = got.as_dtype(Dtype::Float32).unwrap();
        let wrong_f32 = pre_add_rounded.as_dtype(Dtype::Float32).unwrap();
        mlx_rs::transforms::eval([&got_f32, &wrong_f32]).unwrap();
        assert_ne!(
            got_f32.as_slice::<f32>(),
            wrong_f32.as_slice::<f32>(),
            "fixture must discriminate Diffusers' post-add cast from pre-add rounding"
        );
    }

    #[test]
    fn size_and_pack_round_trip_are_explicit() {
        assert_eq!(latent_hw(1024, 1024).unwrap(), (64, 64));
        assert_eq!(latent_hw(512, 2048).unwrap(), (32, 128));
        assert_eq!(latent_hw(2048, 512).unwrap(), (128, 32));
        assert_eq!(latent_hw(688, 1232).unwrap(), (43, 77));
        assert!(latent_hw(496, 512).is_err());
        assert!(latent_hw(512, 2064).is_err());
        assert!(latent_hw(1023, 1024).is_err());
        let x = Array::from_slice(
            &vec![0.0f32; 2 * 3 * LATENT_CHANNELS as usize],
            &[1, 2 * 3, LATENT_CHANNELS],
        );
        assert_eq!(unpack_tokens(&x, 2, 3).unwrap().shape(), [1, 128, 2, 3]);
    }

    #[test]
    fn native_geometry_packs_are_exact_and_budgeted() {
        let samples = [
            GenerationSample {
                prompt: "a",
                negative_prompt: "",
                height: 2048,
                width: 2048,
                seed: 1,
            },
            GenerationSample {
                prompt: "b",
                negative_prompt: "",
                height: 2048,
                width: 2048,
                seed: 2,
            },
            GenerationSample {
                prompt: "c",
                negative_prompt: "",
                height: 2048,
                width: 2048,
                seed: 3,
            },
            GenerationSample {
                prompt: "d",
                negative_prompt: "",
                height: 512,
                width: 2048,
                seed: 4,
            },
            GenerationSample {
                prompt: "e",
                negative_prompt: "",
                height: 688,
                width: 1232,
                seed: 5,
            },
        ];
        let packs = plan_generation_packs(&samples).unwrap();
        assert_eq!(packs.len(), 2);
        assert_eq!(packs[0].sample_range, 0..3);
        assert_eq!(packs[0].image_tokens, 49_152);
        assert_eq!(packs[0].grids, vec![(128, 128); 3]);
        assert_eq!(packs[1].sample_range, 3..5);
        assert_eq!(packs[1].grids, vec![(32, 128), (43, 77)]);
        assert!(packs
            .iter()
            .all(|pack| pack.image_tokens <= MAX_PACKED_IMAGE_TOKENS));
    }

    #[test]
    fn mixed_layout_carries_axes_boundaries_and_fused_cfg_segments() {
        let layout = generation_layout(&[(32, 128), (128, 32), (43, 77)], vec![7, 11, 13]).unwrap();
        // These image/text lens and cumulative boundaries are the frozen reference's concrete
        // representation of the story's "axes_lens"; MSRoPE additionally consumes img_shapes.
        assert_eq!(layout.img_lens(), &[4096, 4096, 3311]);
        assert_eq!(layout.img_cu(), vec![0, 4096, 8192, 11503]);
        assert_eq!(layout.txt_cu(), vec![0, 7, 18, 31]);
        assert_eq!(layout.img_shapes().len(), 3);

        let fused = layout.fused_cfg(&[5, 9, 15]).unwrap();
        assert_eq!(fused.segments(), 6);
        assert_eq!(fused.img_lens(), &[4096, 4096, 3311, 4096, 4096, 3311]);
        assert_eq!(fused.txt_lens(), &[7, 11, 13, 5, 9, 15]);
        assert_eq!(fused.img_tokens(), 23_006);
    }

    #[test]
    fn quant_audit_rejects_one_omitted_small_projection() {
        // Mutation discriminator: skipping even one small projection (for example a timestep
        // `linear_2`) changes 253→252 and must fail, independent of total memory movement.
        assert!(verify_quantized_counts("text encoder", 252, 253).is_err());
        assert!(verify_quantized_counts("DiT", 173, 174).is_err());
        assert!(verify_quantized_counts("VAE", 4, 5).is_err());
        assert!(verify_quantized_counts("VAE", 5, 5).is_ok());
    }

    /// The tiny committed VAE (`tests/fixtures/mage_vae_tiny.safetensors`, sc-14039) — built here
    /// only so the decode-phase signatures can be driven weights-free. Every assertion below fails
    /// or returns before a single tensor reaches it.
    fn tiny_vae() -> MageVae {
        let w = mlx_gen::weights::Weights::from_file(concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/tests/fixtures/mage_vae_tiny.safetensors"
        ))
        .expect("tiny Mage-VAE fixture");
        let raw = w
            .require("fixture.shape")
            .unwrap()
            .as_dtype(Dtype::Int32)
            .unwrap();
        let v: Vec<i32> = raw.as_slice::<i32>().to_vec();
        let shape = crate::vae::denoiser::MageVaeShape {
            patch: v[0],
            hidden: v[1],
            hidden_x: v[2],
            in_channels: v[3],
            bottleneck: v[4],
            num_cond_blocks: v[5] as usize,
            num_mlp_blocks: v[6] as usize,
            max_freqs: v[7] as usize,
            attn_tile: v[8],
        };
        MageVae::from_weights_with_shape(&w, Dtype::Float32, true, shape, "pipeline")
            .expect("build tiny Mage-VAE")
    }

    /// Tokens the decode cannot consume, so the phase's work aborts immediately after the boundary
    /// and no Metal decode runs in a unit test.
    fn undecodable_tokens() -> Array {
        Array::from_slice(&[0f32], &[1, 1, 1])
    }

    fn generation_sample() -> DenoisedGenerationSample {
        DenoisedGenerationSample {
            tokens: undecodable_tokens(),
            gh: 2,
            gw: 2,
            height: 32,
            width: 32,
            trajectories: Vec::new(),
        }
    }

    /// sc-22738: `Progress::Decoding` reaches the caller **once per request**, before any sample is
    /// decoded — the production `Generator::generate` closure discarded the sink entirely, so the
    /// decode phase was invisible on every shipped Mage-Flow member.
    #[test]
    #[ignore = "model inference; explicit model-proof workflow"]
    fn the_generation_decode_phase_opens_once_before_any_sample() {
        let vae = tiny_vae();
        let cancel = CancelFlag::default();

        // Zero samples: a per-sample emit would report nothing at all here.
        let mut events = Vec::new();
        decode_generation_phase(
            &vae,
            DenoisedGenerationBatch {
                samples: Vec::new(),
                packs: Vec::new(),
            },
            None,
            &cancel,
            &mut |p| events.push(p),
        )
        .expect("an empty batch decodes to an empty trace");
        assert_eq!(
            events.len(),
            1,
            "expected exactly one event, got {events:?}"
        );
        assert!(matches!(events[0], Progress::Decoding));

        // Two samples: a per-sample emit would report twice. The boundary precedes the decode, so
        // it is recorded even though the first sample's tokens are rejected.
        let mut events = Vec::new();
        let result = decode_generation_phase(
            &vae,
            DenoisedGenerationBatch {
                samples: vec![generation_sample(), generation_sample()],
                packs: Vec::new(),
            },
            None,
            &cancel,
            &mut |p| events.push(p),
        );
        assert!(
            result.is_err(),
            "the fixture tokens are deliberately undecodable"
        );
        assert_eq!(
            events.len(),
            1,
            "expected exactly one event, got {events:?}"
        );
        assert!(matches!(events[0], Progress::Decoding));
        crate::model_inference_device::record_model_inference_device();
        println!("MODEL_INFERENCE_BODY_COMPLETED pipeline::tests::the_generation_decode_phase_opens_once_before_any_sample");
    }

    /// The edit route reaches decode through its own phase function (sc-22738).
    #[test]
    #[ignore = "model inference; explicit model-proof workflow"]
    fn the_edit_decode_phase_opens_before_the_decode() {
        let vae = tiny_vae();
        let mut events = Vec::new();
        let result = decode_edit_phase(
            &vae,
            DenoisedEdit {
                target: undecodable_tokens(),
                reference_tokens: undecodable_tokens(),
                trajectories: Vec::new(),
                gh: 2,
                gw: 2,
            },
            None,
            &CancelFlag::default(),
            &mut |p| events.push(p),
        );
        assert!(
            result.is_err(),
            "the fixture tokens are deliberately undecodable"
        );
        assert_eq!(
            events.len(),
            1,
            "expected exactly one event, got {events:?}"
        );
        assert!(matches!(events[0], Progress::Decoding));
        crate::model_inference_device::record_model_inference_device();
        println!("MODEL_INFERENCE_BODY_COMPLETED pipeline::tests::the_edit_decode_phase_opens_before_the_decode");
    }
}

#[cfg(test)]
mod driver_tests {
    use super::*;
    use mlx_gen::driver::conformance::{verify_driver_conformance, ConformanceCase};
    use std::cell::RefCell;
    use std::rc::Rc;

    type SeenLog = Rc<RefCell<Vec<(usize, Vec<i32>, f32, Array)>>>;

    struct MockMageMlxForward {
        use_cfg: bool,
        target_tokens: i32,
        seen: SeenLog,
    }

    impl MageMlxForward for MockMageMlxForward {
        fn uses_cfg(&self) -> bool {
            self.use_cfg
        }

        fn forward_sequence(
            &self,
            sequence: &Array,
            sigma: f32,
            step: usize,
            _cancel: &CancelFlag,
        ) -> Result<Array> {
            mlx_rs::transforms::eval([sequence])?;
            self.seen
                .borrow_mut()
                .push((step, sequence.shape().to_vec(), sigma, sequence.clone()));
            let seq_f32 = sequence.as_dtype(Dtype::Float32)?;
            let scaled = if self.use_cfg {
                let half = sequence.shape()[1] / 2;
                let parts = seq_f32.split_axis(&[half], 1)?;
                let cond = parts[0]
                    .multiply(Array::from_f32(-0.5 - 0.1 * sigma))?
                    .add(Array::from_f32(0.05 * (step as f32 + 1.0)))?;
                let unc = parts[1]
                    .multiply(Array::from_f32(-0.25))?
                    .add(Array::from_f32(-0.02 * self.target_tokens as f32))?;
                concatenate_axis(&[&cond, &unc], 1)?
            } else {
                seq_f32
                    .multiply(Array::from_f32(-0.4 - 0.1 * sigma))?
                    .add(Array::from_f32(0.03 * (step as f32 + 1.0)))?
            };
            Ok(scaled.as_dtype(sequence.dtype())?)
        }
    }

    fn mlx_bf16_bit_equal(a: &Array, b: &Array) -> bool {
        let af = a.as_dtype(Dtype::Float32).unwrap();
        let bf = b.as_dtype(Dtype::Float32).unwrap();
        mlx_rs::transforms::eval([&af, &bf]).unwrap();
        if af.shape() != bf.shape() {
            return false;
        }
        let sa = af.as_slice::<f32>();
        let sb = bf.as_slice::<f32>();
        sa.len() == sb.len()
            && sa
                .iter()
                .zip(sb.iter())
                .all(|(x, y)| x.to_bits() == y.to_bits())
    }

    #[test]
    fn mage_mlx_driver_passes_conformance_t2i_turbo_and_edit() {
        let target_tokens = 4i32;
        let ref_tokens =
            Array::from_slice(&[9.0f32, -8.0, 7.0, -6.0, 5.0, -4.0, 3.0, -2.0], &[1, 4, 2])
                .as_dtype(Dtype::Bfloat16)
                .unwrap();
        let init_target = Array::from_slice(
            &[0.5f32, -0.25, 1.0, -1.5, 0.25, 0.75, -0.5, 0.125],
            &[1, 4, 2],
        )
        .as_dtype(Dtype::Bfloat16)
        .unwrap();

        for (name, cfg, refs) in [
            ("mage_mlx_t2i_cfg", 5.0f32, None),
            ("mage_mlx_t2i_turbo_nocfg", 1.0f32, None),
            ("mage_mlx_edit_cfg", 5.0f32, Some(ref_tokens.clone())),
        ] {
            let sigmas = mage_flow_sigmas(4).unwrap();
            let report = verify_driver_conformance(ConformanceCase {
                total_steps: 4,
                split_at: 2,
                seed: 77,
                base_geometry: (64, 64),
                alt_geometry: (128, 128),
                make_driver: || {
                    let forward = MockMageMlxForward {
                        use_cfg: uses_cfg(cfg),
                        target_tokens,
                        seen: Rc::new(RefCell::new(Vec::new())),
                    };
                    Box::new(
                        MageMlxDriver::new(
                            name,
                            sigmas.clone(),
                            cfg,
                            false,
                            target_tokens,
                            refs.clone(),
                            forward,
                        )
                        .unwrap(),
                    )
                },
                make_initial: || init_target.clone(),
                latents_equal: mlx_bf16_bit_equal,
            })
            .unwrap_or_else(|err| panic!("{name} conformance failed: {err}"));
            assert_eq!(report.total_steps, 4);
            assert!(report.verified_inert_split);
            assert!(report.verified_cancellation);
        }
    }

    #[test]
    fn mage_mlx_edit_driver_reconcatenates_clean_references_and_switches_cfg() {
        let target_tokens = 3i32;
        let init = Array::from_slice(&[0.5f32, -0.5, 0.25, -0.25, 1.0, -1.0], &[1, 3, 2])
            .as_dtype(Dtype::Bfloat16)
            .unwrap();
        let refs = Array::from_slice(&[10.0f32, 11.0, 20.0, 21.0], &[1, 2, 2])
            .as_dtype(Dtype::Bfloat16)
            .unwrap();
        let sigmas = mage_flow_sigmas(4).unwrap();
        let cancel = CancelFlag::default();

        // CFG=5.0: fused [target, refs, target, refs], references clean on every step.
        let seen_cfg: SeenLog = Rc::new(RefCell::new(Vec::new()));
        let drv_cfg = MageMlxDriver::new(
            "mage_mlx_edit_invariant",
            sigmas.clone(),
            5.0,
            false,
            target_tokens,
            Some(refs.clone()),
            MockMageMlxForward {
                use_cfg: true,
                target_tokens,
                seen: seen_cfg.clone(),
            },
        )
        .unwrap();
        let mut st = DriverRunState::new();
        let mut progress = Vec::new();
        let mut on_progress = |p: Progress| progress.push(p);
        let mut hooks = StepHooks::new(&mut on_progress);
        let out = run_mlx_driver_slices(
            &drv_cfg,
            [SliceSpec::full(4)],
            BoundaryState::XSigma(init.clone()),
            4,
            0,
            &cancel,
            None,
            &mut st,
            &mut hooks,
            |_, out, _| Ok(out),
        )
        .unwrap();
        let BoundaryState::XSigma(final_target) = out else {
            panic!("expected XSigma");
        };
        assert_eq!(final_target.shape(), [1, 3, 2]);
        assert_eq!(final_target.dtype(), Dtype::Bfloat16);
        assert_eq!(progress.len(), 4);

        let log = seen_cfg.borrow();
        assert_eq!(log.len(), 4);
        for (step, shape, _, seq) in log.iter() {
            assert_eq!(shape.as_slice(), &[1, 10, 2], "step {step} fused CFG edit");
            let parts = seq.split_axis(&[3, 5, 8], 1).unwrap();
            assert!(
                mlx_bf16_bit_equal(&parts[1], &refs),
                "step {step} cond clean refs modified"
            );
            assert!(
                mlx_bf16_bit_equal(&parts[3], &refs),
                "step {step} uncond clean refs modified"
            );
            assert!(
                mlx_bf16_bit_equal(&parts[0], &parts[2]),
                "step {step} cond/uncond targets must match before forward"
            );
        }

        // Turbo CFG=1.0: single branch [seq] without duplication.
        let seen_turbo: SeenLog = Rc::new(RefCell::new(Vec::new()));
        let drv_turbo = MageMlxDriver::new(
            "mage_mlx_turbo_invariant",
            sigmas,
            1.0,
            false,
            target_tokens,
            Some(refs),
            MockMageMlxForward {
                use_cfg: false,
                target_tokens,
                seen: seen_turbo.clone(),
            },
        )
        .unwrap();
        let mut st_t = DriverRunState::new();
        let mut noop = |_: Progress| {};
        let mut hooks_t = StepHooks::new(&mut noop);
        let _ = run_mlx_driver_slices(
            &drv_turbo,
            [SliceSpec::full(4)],
            BoundaryState::XSigma(init),
            4,
            0,
            &cancel,
            None,
            &mut st_t,
            &mut hooks_t,
            |_, out, _| Ok(out),
        )
        .unwrap();
        for (step, shape, _, _) in seen_turbo.borrow().iter() {
            assert_eq!(
                shape.as_slice(),
                &[1, 5, 2],
                "step {step} cfg==1.0 must not duplicate uncond branch"
            );
        }
    }

    #[test]
    fn mage_mlx_initial_tokens_preserve_gaussian_shading_watermark() {
        let key = GsKey::default();
        let a = initial_tokens_for_grid(2, 2, 42, &key, Dtype::Bfloat16).unwrap();
        let b = initial_tokens_for_grid(2, 2, 42, &key, Dtype::Bfloat16).unwrap();
        let c = initial_tokens_for_grid(2, 2, 43, &key, Dtype::Bfloat16).unwrap();
        assert_eq!(a.shape(), [1, 4, LATENT_CHANNELS]);
        assert_eq!(a.dtype(), Dtype::Bfloat16);
        assert!(mlx_bf16_bit_equal(&a, &b));
        assert!(!mlx_bf16_bit_equal(&a, &c));

        // Direct match against `encode_noise` (never plain `randn` or zeros).
        let encoded = encode_noise((LATENT_CHANNELS as usize, 2, 2), &key, 42, Dtype::Bfloat16)
            .unwrap()
            .transpose_axes(&[0, 2, 3, 1])
            .unwrap()
            .reshape(&[1, 4, LATENT_CHANNELS])
            .unwrap();
        assert!(mlx_bf16_bit_equal(&a, &encoded));
    }
}
