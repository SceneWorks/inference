//! Snapshot loaders — the `Qwen/Qwen-Image-2.1` diffusers layout, received as a caller-provisioned
//! local directory (`WeightsSource::Dir`; inference never self-fetches):
//!
//! ```text
//! <root>/
//!   model_index.json
//!   processor/tokenizer.json            (+ tokenizer_config.json, chat_template.jinja, …)
//!   scheduler/scheduler_config.json
//!   text_encoder/config.json + model-*.safetensors   (Qwen3VLForConditionalGeneration)
//!   transformer/config.json + diffusion_pytorch_model-*.safetensors
//!   vae/config.json + diffusion_pytorch_model.safetensors
//! ```
//!
//! Every geometry comes from the component's own `config.json`, so the miniature parity snapshot
//! and the production snapshot go through one code path. Unlike the MLX twin — which loads dense at
//! the on-disk dtype — candle's `VarBuilder` casts on read, so every component is materialized at
//! ONE dtype, [`compute_dtype`]: f32 on CPU (the parity lane) and bf16 on CUDA/Metal for the DiT,
//! the VAE **and the Qwen3-VL text encoder** (language tower + ViT), matching the released
//! checkpoint's own dtype and how upstream runs it. Before sc-24114 the text encoder was read at
//! f32 on every backend, which made the tower's resident footprint 4 B/param on CUDA — ~28 GiB
//! for a component upstream holds at ~14 GiB.

use std::collections::HashMap;
use std::path::{Path, PathBuf};

use candle_core::{DType, Device};
use candle_gen::gen_core::tokenizer::{ChatTemplate, TextTokenizer, TokenizerConfig};
use candle_gen::gen_core::WeightsSource;
use candle_gen::{CandleError as Error, Result};

use crate::config::{
    SchedulerConfig, TextEncoderConfig, TransformerConfig, VaeConfig, VisionConfig,
};
use crate::text_encoder::QwenImage21TextEncoder;
use crate::transformer::QwenImage21Transformer;
use crate::vae::QwenImage21Vae;

/// Prefix of the language tower inside `Qwen3VLForConditionalGeneration` checkpoints.
pub const TEXT_ENCODER_PREFIX: &str = "model.language_model";
/// Prefix of the ViT tower inside `Qwen3VLForConditionalGeneration` checkpoints.
pub const VISION_TOWER_PREFIX: &str = "model.visual";
/// Longest prompt (template included) the tokenizer keeps; upstream never truncates, and the
/// released tokenizer's `model_max_length` is 262144 — this only bounds a pathological prompt.
pub const MAX_PROMPT_TOKENS: usize = 4096;
/// `<|endoftext|>`, the released tokenizer's pad token (unused: prompts are never padded).
const PAD_TOKEN_ID: i32 = 151_643;
/// Error label every loader message carries.
pub(crate) const LABEL: &str = "qwen_image_2_1";

/// The dtype this backend materializes and computes **every** component in: the checkpoint's own
/// bf16 on a GPU backend (CUDA/Metal) — the text encoder included, exactly as upstream's
/// `QwenImage21Pipeline` runs `Qwen3VLForConditionalGeneration` in `torch_dtype=bfloat16` — and f32
/// on CPU (candle's CPU half-precision kernels are slow, and the parity lane wants f32 anyway).
pub fn compute_dtype() -> DType {
    #[cfg(any(feature = "cuda", feature = "metal"))]
    {
        DType::BF16
    }
    #[cfg(not(any(feature = "cuda", feature = "metal")))]
    {
        DType::F32
    }
}

/// The snapshot directory behind a spec, or an actionable error for a single-file source.
pub fn snapshot_root(source: &WeightsSource) -> Result<&Path> {
    match source {
        WeightsSource::Dir(p) => Ok(p),
        WeightsSource::File(_) => Err(Error::Msg(
            "qwen_image_2_1 expects a snapshot directory (processor/ scheduler/ text_encoder/ \
             transformer/ vae/), not a single .safetensors file"
                .into(),
        )),
    }
}

/// `processor/tokenizer.json` (falling back to `tokenizer/tokenizer.json`).
pub fn tokenizer_path(root: &Path) -> Result<PathBuf> {
    for candidate in ["processor/tokenizer.json", "tokenizer/tokenizer.json"] {
        let path = root.join(candidate);
        if path.is_file() {
            return Ok(path);
        }
    }
    Err(Error::Msg(format!(
        "qwen_image_2_1: no processor/tokenizer.json under {}",
        root.display()
    )))
}

/// The prompt tokenizer. The template is rendered by [`crate::prompt_template`] and tokenized
/// preformatted, so the chat-template axis is `None` here.
pub fn load_tokenizer(root: &Path) -> Result<TextTokenizer> {
    Ok(TextTokenizer::from_file(
        tokenizer_path(root)?,
        TokenizerConfig {
            max_length: MAX_PROMPT_TOKENS,
            pad_token_id: PAD_TOKEN_ID,
            chat_template: ChatTemplate::None,
            pad_to_max_length: false,
        },
    )?)
}

/// The Qwen3-VL **vision** geometry for a snapshot: `text_encoder/config.json`'s `vision_config`
/// plus the image-processor block of `processor/preprocessor_config.json` (falling back to
/// `processor/processor_config.json`, which is what `save_pretrained` writes for a
/// `Qwen3VLProcessor`).
///
/// `Ok(None)` when the snapshot declares no `vision_config` — the reference route then refuses
/// with a typed error instead of rendering something wrongly conditioned.
pub fn load_vision_config(root: &Path) -> Result<Option<VisionConfig>> {
    let text_encoder = root.join("text_encoder").join("config.json");
    if !text_encoder.is_file() {
        return Ok(None);
    }
    let bytes = std::fs::read(&text_encoder).map_err(|e| {
        Error::Msg(format!(
            "qwen_image_2_1: read {}: {e}",
            text_encoder.display()
        ))
    })?;
    let value: serde_json::Value = serde_json::from_slice(&bytes)
        .map_err(|e| Error::Msg(format!("qwen_image_2_1: text_encoder/config.json: {e}")))?;
    if value.get("vision_config").is_none() {
        return Ok(None);
    }
    let mut processor = None;
    for candidate in [
        "processor/preprocessor_config.json",
        "processor/processor_config.json",
    ] {
        let path = root.join(candidate);
        if path.is_file() {
            let bytes = std::fs::read(&path)
                .map_err(|e| Error::Msg(format!("qwen_image_2_1: read {}: {e}", path.display())))?;
            processor = Some(
                serde_json::from_slice::<serde_json::Value>(&bytes)
                    .map_err(|e| Error::Msg(format!("qwen_image_2_1: {candidate}: {e}")))?,
            );
            break;
        }
    }
    VisionConfig::from_value(&value, processor.as_ref()).map(Some)
}

/// The ViT tower's own disjoint `model.visual.*` slice of the `text_encoder/` shards, materialised
/// on `device` as the flat weight map `candle-llm` loads from.
///
/// Only the visual keys are read: the shard headers are mmapped and each wanted tensor is pulled
/// individually, so attaching the tower never stages a second copy of the language tower (the MLX
/// twin filters an already-resident `Weights` map instead; candle's loader hands out a
/// `VarBuilder`, which cannot be enumerated).
fn load_vision_tower_weights(
    dir: &Path,
    device: &Device,
    dtype: DType,
) -> Result<candle_llm::primitives::Weights> {
    let files = candle_gen::loader::sorted_safetensors(dir, LABEL)?;
    let mut tensors: HashMap<String, candle_core::Tensor> = HashMap::new();
    for file in &files {
        // SAFETY: the same invariant `candle_gen::loader`'s mmap surface documents — a read-only,
        // process-owned weight file, mapped only for the duration of this call.
        let st = unsafe { candle_core::safetensors::MmapedSafetensors::new(file)? };
        let names: Vec<String> = st
            .tensors()
            .into_iter()
            .map(|(name, _)| name)
            .filter(|name| name.starts_with(VISION_TOWER_PREFIX))
            .collect();
        for name in names {
            // Cast to the language tower's `dtype` (`compute_dtype` on the production path), so
            // the two halves of ONE encoder run at one dtype: bf16 end to end on CUDA/Metal, as
            // upstream; f32 end to end on the CPU parity lane. (The released ViT ships bf16, so
            // on a GPU build this is the `Arc`-clone no-op cast.)
            let tensor = st.load(&name, device)?.to_dtype(dtype)?;
            tensors.insert(name, tensor);
        }
    }
    if tensors.is_empty() {
        return Err(Error::Msg(format!(
            "qwen_image_2_1: no `{VISION_TOWER_PREFIX}.*` tensors in {}",
            dir.display()
        )));
    }
    Ok(candle_llm::primitives::Weights::from_map(
        tensors,
        device.clone(),
    ))
}

/// The Qwen3-VL language tower from `<dir>/` (config + shards) on `device`, with the ViT tower
/// attached when `vision` is given. Pass `None` to load the language tower alone (the
/// text-to-image path).
pub fn load_text_encoder_from(
    dir: &Path,
    device: &Device,
    vision: Option<&VisionConfig>,
) -> Result<QwenImage21TextEncoder> {
    // The tower is materialized at the backend's compute dtype like every other component:
    // bf16 on CUDA/Metal (upstream runs the encoder in bf16, and a bf16-resident tower is what
    // the SceneWorks memory floors were derived at), f32 on the CPU parity lane (sc-24114).
    load_text_encoder_from_at(dir, device, vision, compute_dtype())
}

/// [`load_text_encoder_from`] at an explicit `dtype` — the seam the tests use to hold the tower
/// (language + ViT) to ONE materialization dtype on the CPU lane, where [`compute_dtype`] happens
/// to be the f32 the tower used to be hard-wired to. Production goes through
/// [`load_text_encoder_from`] and never names a dtype.
pub fn load_text_encoder_from_at(
    dir: &Path,
    device: &Device,
    vision: Option<&VisionConfig>,
    dtype: DType,
) -> Result<QwenImage21TextEncoder> {
    let cfg = TextEncoderConfig::from_json_file(&dir.join("config.json"))?;
    let vb = candle_gen::loader::load_sorted_mmap(dir, dtype, device, LABEL)?;
    let encoder = QwenImage21TextEncoder::new(&cfg, vb, TEXT_ENCODER_PREFIX)?;
    let Some(vision) = vision else {
        return Ok(encoder);
    };
    let weights = load_vision_tower_weights(dir, device, dtype)?;
    let tower = candle_llm::models::Qwen35VisionModel::from_weights(
        &weights,
        VISION_TOWER_PREFIX,
        vision.tower.clone(),
    )
    .map_err(|e| Error::Msg(format!("qwen_image_2_1: Qwen3-VL vision tower: {e}")))?;
    Ok(encoder.with_vision(tower, vision.clone()))
}

/// The Qwen3-VL language tower from `<root>/text_encoder/`, plus the ViT tower when the snapshot
/// declares one.
pub fn load_text_encoder(root: &Path, device: &Device) -> Result<QwenImage21TextEncoder> {
    let vision = load_vision_config(root)?;
    load_text_encoder_from(&root.join("text_encoder"), device, vision.as_ref())
}

/// The DiT from `<root>/transformer/`.
pub fn load_transformer(root: &Path, device: &Device) -> Result<QwenImage21Transformer> {
    let cfg = TransformerConfig::from_json_file(&root.join("transformer").join("config.json"))?;
    let vb = candle_gen::loader::component_vb(root, "transformer", compute_dtype(), device, LABEL)?;
    QwenImage21Transformer::new(&cfg, vb)
}

/// The RGBA VAE from `<root>/vae/`.
///
/// The autoencoder is all-conv and stays **dense at every installable tier** (`crate::quant`), so
/// its `VarBuilder` reads floats. [`crate::quant::guard_dense`] over the decode tail's outermost
/// convolution turns a future tier that packed the VAE into a hard, named load error instead of a
/// u32 code stream silently reinterpreted as bf16.
pub fn load_vae(root: &Path, device: &Device) -> Result<QwenImage21Vae> {
    let cfg = VaeConfig::from_json_file(&root.join("vae").join("config.json"))?;
    let vb = candle_gen::loader::component_vb(root, "vae", compute_dtype(), device, LABEL)?;
    for base in ["decoder.conv_in", "decoder.conv_out", "conv2"] {
        crate::quant::guard_dense(&vb, base)?;
    }
    QwenImage21Vae::new(&cfg, vb)
}

/// `<root>/scheduler/scheduler_config.json`, or the frozen production values when the snapshot
/// ships none.
pub fn load_scheduler_config(root: &Path) -> Result<SchedulerConfig> {
    let path = root.join("scheduler").join("scheduler_config.json");
    if path.is_file() {
        SchedulerConfig::from_json_file(&path)
    } else {
        Ok(SchedulerConfig::production())
    }
}
