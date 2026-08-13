//! Real-checkpoint loading for FLUX.2-klein from a `black-forest-labs/FLUX.2-klein-9B` snapshot
//! directory (standard diffusers multi-component tree):
//! ```text
//!   <root>/tokenizer/tokenizer.json
//!   <root>/text_encoder/*.safetensors   (Qwen3, `model.*` keys)
//!   <root>/transformer/*.safetensors    (S3)
//!   <root>/vae/*.safetensors            (S2)
//! ```
//! The Qwen3 `text_encoder` layout maps directly onto the encoder under the `"model"` prefix
//! (the fork's mapping only strips `model.`), so it needs no remap.

use std::path::Path;

use mlx_gen::tokenizer::{ChatTemplate, TextTokenizer, TokenizerConfig};
use mlx_gen::weights::Weights;
use mlx_gen::{Result, WeightsSource};

use crate::config::{Flux2Config, Flux2Quant};
use crate::text_encoder::{Qwen3TextEncoder, Qwen3TextEncoderConfig};
use crate::transformer::{Flux2ControlBranch, Flux2ControlTransformer, Flux2Transformer};
use crate::vae::Flux2Vae;
use crate::vision::{Mistral3Projector, PixtralVisionConfig, PixtralVisionTower};

/// Read a component's pre-quantized-snapshot manifest (sc-5917): the `quantization` block
/// (`{ "bits", "group_size" }`) [`crate::convert`] writes into `{dir}/config.json`. `None` for a
/// dense snapshot (block absent / no config) ⇒ the dense load path. Defaults mirror the convert's
/// (`bits 4`, `group_size 64`) if a field is somehow missing from an otherwise-present block.
pub(crate) fn read_component_quant(dir: &Path) -> Result<Option<Flux2Quant>> {
    let path = dir.join("config.json");
    if !path.exists() {
        return Ok(None);
    }
    let v: serde_json::Value = serde_json::from_str(&std::fs::read_to_string(&path)?)
        .map_err(|e| mlx_gen::Error::Msg(format!("flux2: parse {}: {e}", path.display())))?;
    Ok(v.get("quantization")
        .filter(|q| q.is_object())
        .map(|q| Flux2Quant {
            bits: q
                .get("bits")
                .and_then(serde_json::Value::as_i64)
                .unwrap_or(4) as i32,
            group_size: q
                .get("group_size")
                .and_then(serde_json::Value::as_i64)
                .unwrap_or(64) as i32,
        }))
}

pub(crate) fn alias_transformer_double_block(w: &mut Weights, index: usize) {
    let from = format!("transformer_blocks.{index}.attn.to_out.0");
    let to = format!("transformer_blocks.{index}.attn.to_out");
    for suffix in ["weight", "scales", "biases"] {
        w.alias(&format!("{from}.{suffix}"), &format!("{to}.{suffix}"));
    }
}

/// Qwen2 pad token id (`<|endoftext|>`).
pub const PAD_TOKEN_ID: i32 = 151643;
/// The fork's `LanguageTokenizer` max_length for the FLUX.2 `qwen3` tokenizer.
pub const MAX_LENGTH: usize = 512;

/// Load the Qwen2 tokenizer with FLUX.2's chat template (`enable_thinking=False`) and the fork's
/// padding policy (`padding="max_length"` → every prompt padded to 512).
pub fn load_tokenizer(root: &Path) -> Result<TextTokenizer> {
    let source = crate::config::KLEIN_ENCODER_CONTRACT
        .validate_source_against_base(&WeightsSource::Dir(root.join("text_encoder")), root)?;
    load_validated_tokenizer(&source)
}

pub(crate) fn load_validated_tokenizer(
    source: &mlx_gen::gen_core::ValidatedEncoderSource,
) -> Result<TextTokenizer> {
    source.read_tokenizer_unchanged(|path| {
        TextTokenizer::from_file(
            path,
            TokenizerConfig {
                max_length: MAX_LENGTH,
                pad_token_id: PAD_TOKEN_ID,
                chat_template: ChatTemplate::QwenInstructNoThink,
                pad_to_max_length: true,
            },
        )
        .map_err(Into::into)
    })
}

/// Load the Qwen3 text encoder. The on-disk `model.*` keys map directly onto the encoder tree
/// under the `"model"` prefix — no remap needed. Manifest-aware: a pre-quantized klein snapshot
/// (sc-5917 convert) loads packed; a stock dense snapshot loads dense (no `quantization` block).
pub fn load_text_encoder(root: &Path) -> Result<Qwen3TextEncoder> {
    let selected = crate::config::KLEIN_ENCODER_CONTRACT
        .validate_source_against_base(&WeightsSource::Dir(root.join("text_encoder")), root)?;
    selected.read_unchanged(load_text_encoder_from_source)
}

pub(crate) fn load_text_encoder_from_source(source: &WeightsSource) -> Result<Qwen3TextEncoder> {
    let quant = read_source_quant(source)?;
    let w = weights_from_source(source)?;
    let encoder = Qwen3TextEncoder::from_weights_quant(
        &w,
        "model",
        &Qwen3TextEncoderConfig::klein_9b(),
        quant,
    )?;
    w.materialize_accessed()?;
    Ok(encoder)
}

/// `<pad>` token id for the FLUX.2-dev Mistral tokenizer (vs klein's Qwen2 `<|endoftext|>` 151643).
pub const DEV_PAD_TOKEN_ID: i32 = 11;

/// Load the FLUX.2-dev tokenizer: the Mistral/Pixtral `tokenizer.json` with the dev chat template
/// (`[SYSTEM_PROMPT]…[/SYSTEM_PROMPT][INST]…[/INST]`, BOS auto-prepended by the post-processor) and
/// the fork's `padding="max_length"` (every prompt padded to 512 with `<pad>`). The `PixtralProcessor`
/// image path is not part of the T2I tokenization (sc-5918).
pub fn load_tokenizer_dev(root: &Path) -> Result<TextTokenizer> {
    let source = crate::config::DEV_ENCODER_CONTRACT
        .validate_source_against_base(&WeightsSource::Dir(root.join("text_encoder")), root)?;
    load_validated_tokenizer_dev(&source)
}

pub(crate) fn load_validated_tokenizer_dev(
    source: &mlx_gen::gen_core::ValidatedEncoderSource,
) -> Result<TextTokenizer> {
    source.read_tokenizer_unchanged(|path| {
        TextTokenizer::from_file(
            path,
            TokenizerConfig {
                max_length: MAX_LENGTH,
                pad_token_id: DEV_PAD_TOKEN_ID,
                chat_template: ChatTemplate::Flux2DevMistral,
                pad_to_max_length: true,
            },
        )
        .map_err(Into::into)
    })
}

/// Load the **FLUX.2-dev Mistral** text encoder (sc-5915). The dev `text_encoder` is a
/// `Mistral3ForConditionalGeneration`; the T2I path consumes only its language tower, whose weights
/// live under the `language_model.model.*` prefix (the vision tower + projector are unused here,
/// sc-5918). Same decoder-LM graph as klein's Qwen3 minus the per-head q/k-norm (`qk_norm: false`).
pub fn load_text_encoder_dev(root: &Path) -> Result<Qwen3TextEncoder> {
    let selected = crate::config::DEV_ENCODER_CONTRACT
        .validate_source_against_base(&WeightsSource::Dir(root.join("text_encoder")), root)?;
    selected.read_unchanged(load_text_encoder_dev_from_source)
}

pub(crate) fn load_text_encoder_dev_from_source(
    source: &WeightsSource,
) -> Result<Qwen3TextEncoder> {
    let quant = read_source_quant(source)?;
    let w = weights_from_source(source)?;
    let encoder = load_text_encoder_dev_from(&w, quant)?;
    w.materialize_accessed()?;
    Ok(encoder)
}

/// [`load_text_encoder_dev`] from an already-parsed `text_encoder/` [`Weights`] (+ its component
/// quant manifest). Lets the dev edit load parse the ~45 GB shard set ONCE and share it across the
/// Mistral language tower, the Pixtral vision tower, and the multimodal projector (F-112).
pub(crate) fn load_text_encoder_dev_from(
    w: &Weights,
    quant: Option<Flux2Quant>,
) -> Result<Qwen3TextEncoder> {
    let mut encoder = Qwen3TextEncoder::from_weights_quant(
        w,
        "language_model.model",
        &Qwen3TextEncoderConfig::mistral_dev(),
        quant,
    )?;
    // Also load the final norm + LM head so this encoder can drive the caption-upsampling
    // `generate()` loop (sc-6030) — the dev `Mistral3ForConditionalGeneration` snapshot carries
    // `language_model.lm_head.weight` (dense bf16) + `language_model.model.norm.weight`, both
    // retained by the pre-quant convert. The T2I/edit prompt-embeds path never touches them.
    encoder.load_generation_head(w, "language_model", quant)?;
    Ok(encoder)
}

/// Load the whole dev `text_encoder/` group — the Mistral language tower, the Pixtral vision tower,
/// and the multimodal projector — parsing the ~45 GB shard set ONCE and sharing it across all three
/// (F-112), instead of three independent `Weights::from_dir` reads. The vision tower + projector are
/// full precision; only the language tower honours the component quant manifest.
pub fn load_dev_text_encoder_group(
    root: &Path,
) -> Result<(Qwen3TextEncoder, PixtralVisionTower, Mistral3Projector)> {
    let selected = crate::config::DEV_ENCODER_CONTRACT
        .validate_source_against_base(&WeightsSource::Dir(root.join("text_encoder")), root)?;
    selected.validate_vision(
        &crate::config::DEV_VISION_ENCODER_CONTRACT,
        &crate::config::DEV_ENCODER_CONTRACT,
    )?;
    selected.read_unchanged(load_dev_text_encoder_group_from_source)
}

/// Load a substitutable Mistral language tower while keeping the checkpoint-coupled Pixtral vision
/// tower and projector on the builtin source. If both sources are identical the legacy one-parse
/// path is preserved exactly.
pub(crate) fn load_dev_text_encoder_group_from_sources(
    language_source: &WeightsSource,
    multimodal_source: &WeightsSource,
) -> Result<(Qwen3TextEncoder, PixtralVisionTower, Mistral3Projector)> {
    if same_source(language_source, multimodal_source) {
        return load_dev_text_encoder_group_from_source(language_source);
    }
    let language_quant = read_source_quant(language_source)?;
    let language_weights = weights_from_source(language_source)?;
    let encoder = load_text_encoder_dev_from(&language_weights, language_quant)?;
    language_weights.materialize_accessed()?;
    let multimodal_weights = weights_from_source(multimodal_source)?;
    let vision_tower = load_vision_tower_dev_from(&multimodal_weights)?;
    let projector = load_multimodal_projector_dev_from(&multimodal_weights)?;
    multimodal_weights.materialize_accessed()?;
    Ok((encoder, vision_tower, projector))
}

pub(crate) fn load_dev_text_encoder_group_from_source(
    source: &WeightsSource,
) -> Result<(Qwen3TextEncoder, PixtralVisionTower, Mistral3Projector)> {
    let quant = read_source_quant(source)?;
    let w = weights_from_source(source)?;
    let encoder = load_text_encoder_dev_from(&w, quant)?;
    let vision_tower = load_vision_tower_dev_from(&w)?;
    let projector = load_multimodal_projector_dev_from(&w)?;
    w.materialize_accessed()?;
    Ok((encoder, vision_tower, projector))
}

fn same_source(left: &WeightsSource, right: &WeightsSource) -> bool {
    match (left, right) {
        (WeightsSource::Dir(left), WeightsSource::Dir(right))
        | (WeightsSource::File(left), WeightsSource::File(right)) => left == right,
        _ => false,
    }
}

fn read_source_quant(source: &WeightsSource) -> Result<Option<Flux2Quant>> {
    let config = match source {
        WeightsSource::Dir(path) => path.join("config.json"),
        WeightsSource::File(path) => path
            .parent()
            .unwrap_or_else(|| Path::new("."))
            .join("config.json"),
    };
    read_component_quant(config.parent().unwrap_or_else(|| Path::new(".")))
}

fn weights_from_source(source: &WeightsSource) -> Result<Weights> {
    match source {
        WeightsSource::Dir(path) => Weights::from_dir(path),
        WeightsSource::File(path) => Weights::from_file(path),
    }
}

/// Load the FLUX.2 VAE. The on-disk diffusers keys (`encoder.*`/`decoder.*`/`quant_conv.*`/
/// `bn.*`) map directly onto the module; conv weights are transposed `[O,I,H,W]→[O,H,W,I]` at
/// construction.
pub fn load_vae(root: &Path) -> Result<Flux2Vae> {
    let w = Weights::from_dir(root.join("vae"))?;
    Flux2Vae::from_weights(&w)
}

/// Load the MMDiT transformer for `cfg`, applying the diffusers→internal renames (the fork's
/// `Flux2WeightMapping`): the time embedding `time_guidance_embed.timestep_embedder.linear_{1,2}`
/// → `time_guidance_embed.linear_{1,2}`, and each double block's Sequential
/// `transformer_blocks.{i}.attn.to_out.0` → `to_out`. Everything else matches 1:1. The renames are
/// arch-general (klein and dev are the same `Flux2Transformer2DModel`); only `cfg` differs.
fn load_transformer_with(root: &Path, cfg: &Flux2Config) -> Result<Flux2Transformer> {
    let dir = root.join("transformer");
    let quant = read_component_quant(&dir)?;
    let mut w = Weights::from_dir(dir)?;
    // Rename the diffusers→internal keys. For a pre-quantized snapshot (sc-5917) the renamed
    // Linear's value lives in the packed triple `{weight (u32 codes), scales, biases}`, so alias
    // all three (each `alias` is a no-op when its source is absent — a dense snapshot just renames
    // `.weight`).
    let alias_lin = |w: &mut Weights, from: &str, to: &str| {
        for suffix in ["weight", "scales", "biases"] {
            w.alias(&format!("{from}.{suffix}"), &format!("{to}.{suffix}"));
        }
    };
    for n in ["linear_1", "linear_2"] {
        alias_lin(
            &mut w,
            &format!("time_guidance_embed.timestep_embedder.{n}"),
            &format!("time_guidance_embed.{n}"),
        );
    }
    for i in 0..cfg.num_double_layers {
        alias_transformer_double_block(&mut w, i);
    }
    Flux2Transformer::from_weights_quant(&w, cfg, quant)
}

/// Load the FLUX.2-klein MMDiT transformer.
pub fn load_transformer(root: &Path) -> Result<Flux2Transformer> {
    load_transformer_with(root, &Flux2Config::klein_9b())
}

/// Load the FLUX.2-dev MMDiT transformer (sc-5916): the same parametric module tree as klein at the
/// dev dims (48 single blocks / 48 heads / joint 15360), via `Flux2Config::dev()`.
pub fn load_transformer_dev(root: &Path) -> Result<Flux2Transformer> {
    load_transformer_with(root, &Flux2Config::dev())
}

/// Load the FLUX.2-dev base MMDiT **plus** the Fun-Controlnet-Union control branch (sc-2292) from
/// the dev `transformer/` snapshot (`root`) and the control checkpoint (`control` — a single
/// `.safetensors` `File`, or a `Dir` of them; the HF cache stores it as one file in a snapshot dir).
/// The base loads manifest-aware (a pre-quantized dev snapshot loads packed, sc-5917); the control
/// keys (`control_img_in.*`, `control_transformer_blocks.{i}.*`) map 1:1 onto the branch and load
/// dense (un-prefixed). `load_dev_control` then quantizes the whole [`Flux2ControlTransformer`] —
/// a no-op on the already-packed base, packing the dense control overlay.
pub fn load_control_transformer_dev(
    root: &Path,
    control: &WeightsSource,
) -> Result<Flux2ControlTransformer> {
    let cfg = Flux2Config::dev();
    let base = load_transformer_dev(root)?;
    let mut control_weights = match control {
        WeightsSource::File(p) => Weights::from_file(p)?,
        WeightsSource::Dir(p) => Weights::from_dir(p)?,
    };
    // Each control block's attention output projection ships as the diffusers Sequential
    // `attn.to_out.0` (`[Linear, Dropout]`), exactly like the base `transformer_blocks.{i}.attn.to_out.0`
    // — rename it to the internal `attn.to_out` the shared `DoubleBlock` loader reads. (`alias` of
    // `.scales`/`.biases` is a no-op for the dense bf16 control checkpoint.)
    let n_control = cfg.control_layer_places().len();
    for i in 0..n_control {
        for suffix in ["weight", "scales", "biases"] {
            control_weights.alias(
                &format!("control_transformer_blocks.{i}.attn.to_out.0.{suffix}"),
                &format!("control_transformer_blocks.{i}.attn.to_out.{suffix}"),
            );
        }
    }
    let branch = Flux2ControlBranch::from_weights(&control_weights, "", &cfg)?;
    Ok(Flux2ControlTransformer::new(base, branch))
}

/// Load the FLUX.2-dev **Pixtral vision tower** (sc-5918) from the `text_encoder/` snapshot — the
/// `vision_tower.*` keys of the `Mistral3ForConditionalGeneration` checkpoint. Used for
/// edit/reference image conditioning (sc-5919), not the T2I path. The tower stays full precision
/// (only the MMDiT + Mistral language tower quantize), so it loads dense regardless of the
/// pre-quantized-snapshot manifest.
pub fn load_vision_tower_dev(root: &Path) -> Result<PixtralVisionTower> {
    let selected = crate::config::DEV_ENCODER_CONTRACT
        .validate_source_against_base(&WeightsSource::Dir(root.join("text_encoder")), root)?;
    selected.validate_vision(
        &crate::config::DEV_VISION_ENCODER_CONTRACT,
        &crate::config::DEV_ENCODER_CONTRACT,
    )?;
    selected.read_unchanged(|source| {
        let w = weights_from_source(source)?;
        let tower = load_vision_tower_dev_from(&w)?;
        w.materialize_accessed()?;
        Ok(tower)
    })
}

/// [`load_vision_tower_dev`] from an already-parsed `text_encoder/` [`Weights`] (F-112).
pub(crate) fn load_vision_tower_dev_from(w: &Weights) -> Result<PixtralVisionTower> {
    PixtralVisionTower::from_weights(w, "vision_tower", PixtralVisionConfig::dev())
}

/// Load the FLUX.2-dev **Mistral3 multimodal projector** (sc-5918) from the `text_encoder/`
/// snapshot (`multi_modal_projector.*` keys). `spatial_merge_size = 2`; the projector's RMSNorm
/// uses the Mistral **text** `rms_norm_eps` (1e-5), per the reference. Full precision, like the
/// vision tower.
pub fn load_multimodal_projector_dev(root: &Path) -> Result<Mistral3Projector> {
    let selected = crate::config::DEV_ENCODER_CONTRACT
        .validate_source_against_base(&WeightsSource::Dir(root.join("text_encoder")), root)?;
    selected.validate_vision(
        &crate::config::DEV_VISION_ENCODER_CONTRACT,
        &crate::config::DEV_ENCODER_CONTRACT,
    )?;
    selected.read_unchanged(|source| {
        let w = weights_from_source(source)?;
        let projector = load_multimodal_projector_dev_from(&w)?;
        w.materialize_accessed()?;
        Ok(projector)
    })
}

/// [`load_multimodal_projector_dev`] from an already-parsed `text_encoder/` [`Weights`] (F-112).
pub(crate) fn load_multimodal_projector_dev_from(w: &Weights) -> Result<Mistral3Projector> {
    Mistral3Projector::from_weights(w, "multi_modal_projector", 2, 1e-5)
}
