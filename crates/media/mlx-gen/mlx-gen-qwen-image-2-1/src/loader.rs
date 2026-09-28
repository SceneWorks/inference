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
//! Weights load dense at their on-disk dtype (bf16 released; f32 for the tiny parity snapshot) and
//! every geometry comes from the component's own `config.json`, so one code path serves both.
//!
//! # Every loader materializes at load (sc-24114) — do not remove as "unnecessary"
//!
//! Each loader calls `Weights::materialize_accessed` before returning. MLX runs a safetensors
//! `Load` on its **CPU stream**; when a GPU kernel consumes a not-yet-read `Load`, `eval` encodes a
//! GPU-timeline wait on that `pread` inside the Metal command buffer (`Event::wait(stream)` →
//! `encodeWait`). Left lazy, the first forward *was* the load: a 14 GB DiT streamed from a slow
//! (~300 MiB/s) volume held command buffers on the Metal timeline past the **GPU watchdog** —
//! `kIOGPUCommandBufferCallbackErrorTimeout`, then `SubmissionsIgnored` for the rest of the process
//! (the sc-24110/24111 real-weight smokes). Materializing here reads on the CPU stream with no GPU
//! consumer waiting, and is also the sc-22414 GPU-view coherence seam every other provider's loader
//! sits behind. It does not defeat staged residency: `Residency` calls these loaders at the start
//! of the phase that needs the component (tower for the encode, DiT + VAE after the tower is
//! dropped), so the staged peak stays `max(tower, DiT + VAE)`. `materialize_accessed` (not
//! `materialize`) so the untied `lm_head` and, on the text-only route, `model.visual.*` are never
//! read.

use std::collections::HashMap;
use std::path::{Path, PathBuf};

use mlx_gen::tokenizer::{ChatTemplate, TextTokenizer, TokenizerConfig};
use mlx_gen::weights::Weights;
use mlx_gen::{Error, Result, WeightsSource};

use crate::config::{
    SchedulerConfig, TextEncoderConfig, TransformerConfig, VaeConfig, VisionConfig,
};
use crate::text_encoder::QwenImage21TextEncoder;
use crate::transformer::QwenImage21Transformer;
use crate::vae::QwenImage21Vae;

/// Prefix of the language tower inside `Qwen3VLForConditionalGeneration` checkpoints.
pub const TEXT_ENCODER_PREFIX: &str = "model.language_model";
/// Longest prompt (template included) the tokenizer keeps; upstream never truncates, and the
/// released tokenizer's `model_max_length` is 262144 — this only bounds a pathological prompt.
pub const MAX_PROMPT_TOKENS: usize = 4096;
/// `<|endoftext|>`, the released tokenizer's pad token (unused: prompts are never padded).
const PAD_TOKEN_ID: i32 = 151_643;

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

/// Prefix of the ViT tower inside `Qwen3VLForConditionalGeneration` checkpoints.
pub const VISION_TOWER_PREFIX: &str = "model.visual";

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
    let value: serde_json::Value = serde_json::from_slice(&std::fs::read(&text_encoder)?)
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
            processor = Some(
                serde_json::from_slice::<serde_json::Value>(&std::fs::read(&path)?)
                    .map_err(|e| Error::Msg(format!("qwen_image_2_1: {candidate}: {e}")))?,
            );
            break;
        }
    }
    VisionConfig::from_value(&value, processor.as_ref()).map(Some)
}

/// The Qwen3-VL language tower from `<text_encoder>/` (config + shards), with the ViT tower
/// attached when the snapshot ships one. `root` supplies the `processor/` geometry; pass `None`
/// to load the language tower alone (the text-to-image path).
pub fn load_text_encoder_from(
    dir: &Path,
    vision: Option<&VisionConfig>,
) -> Result<QwenImage21TextEncoder> {
    let cfg = TextEncoderConfig::from_json_file(&dir.join("config.json"))?;
    let w = Weights::from_dir(dir)?;
    let encoder = QwenImage21TextEncoder::from_weights(&w, TEXT_ENCODER_PREFIX, &cfg)?;
    let Some(vision) = vision else {
        w.materialize_accessed()?;
        return Ok(encoder);
    };
    // The ViT tower reads its own disjoint `model.visual.*` slice of the same shards. `mlx-llm`
    // carries the shared Qwen3-VL tower, so the weights are handed across on its own `Weights`.
    let visual: HashMap<String, mlx_rs::Array> = w
        .keys()
        .filter(|key| key.starts_with(VISION_TOWER_PREFIX))
        .map(str::to_string)
        .collect::<Vec<_>>()
        .into_iter()
        .filter_map(|key| w.get(&key).cloned().map(|array| (key, array)))
        .collect();
    if visual.is_empty() {
        return Err(Error::MissingTensor(format!("{VISION_TOWER_PREFIX}.*")));
    }
    let tower = mlx_llm::models::Qwen3VLVisionModel::from_weights(
        &mlx_llm::primitives::Weights::from_map(visual),
        VISION_TOWER_PREFIX,
        vision.tower.clone(),
    )
    .map_err(|e| Error::Msg(format!("qwen_image_2_1: Qwen3-VL vision tower: {e}")))?;
    // The `visual` slice was read through `get`, so it is in the accessed set too.
    w.materialize_accessed()?;
    Ok(encoder.with_vision(tower, vision.clone()))
}

/// The Qwen3-VL language tower from `<root>/text_encoder/`, plus the ViT tower when the snapshot
/// declares one.
pub fn load_text_encoder(root: &Path) -> Result<QwenImage21TextEncoder> {
    let vision = load_vision_config(root)?;
    load_text_encoder_from(&root.join("text_encoder"), vision.as_ref())
}

/// The DiT from `<root>/transformer/`.
pub fn load_transformer(root: &Path) -> Result<QwenImage21Transformer> {
    let dir = root.join("transformer");
    let cfg = TransformerConfig::from_json_file(&dir.join("config.json"))?;
    let w = Weights::from_dir(&dir)?;
    let transformer = QwenImage21Transformer::from_weights(&w, &cfg)?;
    w.materialize_accessed()?;
    Ok(transformer)
}

/// The RGBA VAE from `<root>/vae/`.
pub fn load_vae(root: &Path) -> Result<QwenImage21Vae> {
    let dir = root.join("vae");
    let cfg = VaeConfig::from_json_file(&dir.join("config.json"))?;
    let w = Weights::from_dir(&dir)?;
    let vae = QwenImage21Vae::from_weights(&w, &cfg)?;
    w.materialize_accessed()?;
    Ok(vae)
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

#[cfg(test)]
mod tests {
    use super::*;
    use mlx_rs::Array;

    fn tiny_snapshot() -> PathBuf {
        Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/tiny-snapshot")
    }

    fn truncate(path: &Path) {
        std::fs::OpenOptions::new()
            .write(true)
            .open(path)
            .unwrap_or_else(|e| panic!("{}: {e}", path.display()))
            .set_len(0)
            .unwrap();
    }

    fn byte_equal(a: &Array, b: &Array) -> bool {
        a.shape() == b.shape()
            && mlx_rs::ops::eq(a, b)
                .and_then(|m| m.all(None))
                .and_then(|m| m.try_item::<bool>())
                .unwrap_or(false)
    }

    /// sc-24114: every loader consumes its snapshot bytes **at load**, not inside the first
    /// forward. The pre-fix loaders handed out lazy `Load`s, so step 1 of a render was the read —
    /// and each GPU kernel consuming an unread weight put a GPU-timeline wait on a `pread` inside
    /// a Metal command buffer (`kIOGPUCommandBufferCallbackErrorTimeout` from a slow volume).
    ///
    /// Mechanical pin: load all three components from a private copy of the tiny snapshot, then
    /// **truncate every shard to zero bytes**, then run a forward through each and compare it with
    /// the same forward from an untouched load. If the reads were deferred, the forward reads an
    /// empty file (an error, or different bytes); if they were consumed at load, the outputs are
    /// identical.
    ///
    /// *Mutation that reds this:* dropping any of the three `w.materialize_accessed()` calls.
    #[test]
    fn loaders_consume_their_snapshot_bytes_at_load() {
        let tmp = tempfile::tempdir().unwrap();
        let root = tmp.path().join("snapshot");
        mlx_gen::quant::copy_dir(&tiny_snapshot(), &root).unwrap();

        let reference_root = tiny_snapshot();
        let (ref_te, ref_dit, ref_vae) = (
            load_text_encoder(&reference_root).unwrap(),
            load_transformer(&reference_root).unwrap(),
            load_vae(&reference_root).unwrap(),
        );
        let (te, dit, vae) = (
            load_text_encoder(&root).unwrap(),
            load_transformer(&root).unwrap(),
            load_vae(&root).unwrap(),
        );
        for shard in [
            "text_encoder/model.safetensors",
            "transformer/diffusion_pytorch_model.safetensors",
            "vae/diffusion_pytorch_model.safetensors",
        ] {
            truncate(&root.join(shard));
        }

        // Text encoder: four tokens through the tower.
        let ids = Array::from_slice(&[5i32, 17, 29, 41], &[1, 4]);
        let mask = Array::ones::<i32>(&[1, 4]).unwrap();
        let want = ref_te.forward(&ids, &mask).unwrap();
        let got = te
            .forward(&ids, &mask)
            .expect("text encoder forward after truncation");
        mlx_rs::transforms::eval([&want, &got]).unwrap();
        assert!(
            byte_equal(&want, &got),
            "text encoder output differs after truncation"
        );

        // DiT: a 2x2 latent grid conditioned on four text rows, at the fixture's geometry.
        let cfg = dit.config();
        let (h, w) = (2usize, 2usize);
        let latents = Array::from_slice(
            &(0..(h * w * cfg.in_channels))
                .map(|i| (i as f32 * 0.11).sin())
                .collect::<Vec<_>>(),
            &[1, (h * w) as i32, cfg.in_channels as i32],
        );
        let text = Array::from_slice(
            &(0..4 * cfg.context_in_dim)
                .map(|i| (i as f32 * 0.07).cos())
                .collect::<Vec<_>>(),
            &[1, 4, cfg.context_in_dim as i32],
        );
        let want = ref_dit.forward(&latents, &text, 0.5, h, w).unwrap();
        let got = dit
            .forward(&latents, &text, 0.5, h, w)
            .expect("transformer forward after truncation");
        mlx_rs::transforms::eval([&want, &got]).unwrap();
        assert!(
            byte_equal(&want, &got),
            "transformer output differs after truncation"
        );

        // VAE: decode one latent frame.
        let z_dim = vae.config().z_dim as i32;
        let z = Array::from_slice(
            &(0..(z_dim * 4))
                .map(|i| (i as f32 * 0.19).sin())
                .collect::<Vec<_>>(),
            &[1, z_dim, 2, 2],
        );
        let want = ref_vae.decode_rgba(&z).unwrap();
        let got = vae.decode_rgba(&z).expect("vae decode after truncation");
        mlx_rs::transforms::eval([&want, &got]).unwrap();
        assert!(
            byte_equal(&want, &got),
            "vae output differs after truncation"
        );
    }
}
