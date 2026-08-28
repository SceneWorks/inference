//! Snapshot-layout loader for SD3.5 (E5, sc-7864). Assembles the full text-to-image stack from a
//! `stabilityai/stable-diffusion-3.5-large` diffusers snapshot directory:
//!
//! ```text
//! <root>/transformer/      diffusion_pytorch_model{,-00001-of-00002}.safetensors  (MMDiT)
//! <root>/text_encoder/     model.safetensors                                       (CLIP-L)
//! <root>/text_encoder_2/   model.safetensors                                       (CLIP-G / bigG)
//! <root>/text_encoder_3/   model-0000{1,2}-of-00002.safetensors                    (T5-XXL)
//! <root>/tokenizer{,_2}/   vocab.json + merges.txt                                 (CLIP BPE)
//! <root>/tokenizer_3/      tokenizer.json                                          (T5)
//! <root>/vae/              diffusion_pytorch_model.safetensors                     (16-ch VAE)
//! ```
//!
//! This crate ships NO net-new encoder/VAE: it REUSES the SDXL CLIP encoder (loaded twice), the FLUX
//! T5 encoder, and the Z-Image 16-ch VAE (via E4's [`crate::vae::load_sd3_vae`]). The loader's job is
//! the snapshot-layout glue + the diffusers→MLX weight-key remap each reused module expects.

use std::collections::BTreeSet;
use std::path::Path;

use mlx_gen::tokenizer::{ChatTemplate, TextTokenizer, TokenizerConfig};
use mlx_gen::weights::Weights;
use mlx_gen::{Error, Result};
use mlx_rs::Dtype;

use mlx_gen_sdxl::tokenizer::ClipBpeTokenizer;
use mlx_gen_sdxl::ClipTextEncoder;

use crate::config::Sd3Arch;
use crate::text::{sd3_clip_g_config, sd3_clip_l_config, Sd3TextEncoders};
use crate::transformer::Sd3Transformer;
use crate::vae::load_sd3_vae;

/// CLIP context window (diffusers `tokenizer(..., max_length=77, padding="max_length")`).
pub const CLIP_MAX_LENGTH: usize = 77;
/// CLIP eos / `<|endoftext|>` id (49407) — the canonical CLIP-BPE vocab maximum, used as the
/// per-encoder pad-id fallback when a `tokenizer_config.json` `pad_token` cannot be resolved.
pub const CLIP_EOS_ID: i32 = 49407;
/// CLIP pad token id — DEPRECATED alias of [`CLIP_EOS_ID`]. **Not** the correct pad for CLIP-bigG,
/// which pads with `!` (0); see [`resolve_clip_pad_id`] (sc-9581). Kept only so no external caller
/// breaks; the pipeline resolves the per-encoder pad token instead.
pub const CLIP_PAD_ID: i32 = CLIP_EOS_ID;

/// The per-encoder CLIP pad token ids for SD3.5 (sc-9581). SD3.5's two CLIP tokenizers pad with
/// DIFFERENT tokens: CLIP-L (`tokenizer/`) pads with `<|endoftext|>` (49407) but OpenCLIP-bigG
/// (`tokenizer_2/`) pads with `!` (id 0). Padding bigG with eos instead of `!` corrupts its
/// penultimate hidden on every pad position (and thus the joint MMDiT context) for any prompt
/// shorter than 77 tokens — the diffusers-parity harness in the candle sibling measured the joint
/// context cosine dropping to ~0.98 (bigG penultimate ~0.40) under the wrong pad. Resolved from each
/// `tokenizer_config.json` `pad_token` at load, mirroring candle-gen-sd3's `resolve_clip_pad_id`.
#[derive(Debug, Clone, Copy)]
pub struct Sd3ClipPad {
    /// CLIP-L (`tokenizer/`) pad id — `<|endoftext|>` (49407).
    pub pad_l: i32,
    /// CLIP-bigG (`tokenizer_2/`) pad id — `!` (0). Differs from L.
    pub pad_g: i32,
}

/// Resolve one CLIP tokenizer directory's configured pad token id (sc-9581): read
/// `<dir>/tokenizer_config.json` `pad_token` (a bare string like `"!"`/`"<|endoftext|>"`, or an
/// `AddedToken`-shaped object with a `content` field) and look that string up in `<dir>/vocab.json`.
/// Falls back to [`CLIP_EOS_ID`] if the config or vocab entry is absent (diffusers pads bigG with `!`
/// and L with eos; the fallback keeps L correct even without a config). A present but unreadable or
/// malformed config is an error: silently substituting eos would corrupt bigG conditioning.
pub fn resolve_clip_pad_id(dir: &Path) -> Result<i32> {
    let cfg = dir.join("tokenizer_config.json");
    let config = match std::fs::read_to_string(&cfg) {
        Ok(config) => config,
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => return Ok(CLIP_EOS_ID),
        Err(error) => return Err(error.into()),
    };
    let value = serde_json::from_str::<serde_json::Value>(&config)
        .map_err(|error| Error::Msg(format!("sd3: parse {}: {error}", cfg.display())))?;
    let pad_str = match &value["pad_token"] {
        serde_json::Value::String(s) => Some(s.clone()),
        serde_json::Value::Object(o) => o.get("content").and_then(|c| c.as_str().map(String::from)),
        _ => None,
    };
    let Some(pad_str) = pad_str else {
        return Ok(CLIP_EOS_ID);
    };
    Ok(std::fs::read_to_string(dir.join("vocab.json"))
        .ok()
        .and_then(|s| serde_json::from_str::<std::collections::HashMap<String, i32>>(&s).ok())
        .and_then(|vocab| vocab.get(&pad_str).copied())
        .unwrap_or(CLIP_EOS_ID))
}

/// Resolve BOTH CLIP encoders' pad ids for a snapshot (sc-9581): `tokenizer/` (L, eos=49407) and
/// `tokenizer_2/` (bigG, `!`=0). See [`Sd3ClipPad`] / [`resolve_clip_pad_id`].
pub fn load_clip_pad_ids(root: &Path) -> Result<Sd3ClipPad> {
    Ok(Sd3ClipPad {
        pad_l: resolve_clip_pad_id(&root.join("tokenizer"))?,
        pad_g: resolve_clip_pad_id(&root.join("tokenizer_2"))?,
    })
}
/// T5 sequence length for SD3 (diffusers `max_sequence_length=256`).
pub const T5_MAX_LENGTH: usize = 256;
/// T5 pad token id — `<pad>` (0).
pub const T5_PAD_ID: i32 = 0;

/// Load CLIP-L (`text_encoder`) at f32 — the SD3 CLIP-L config (768-wide, with a 768 text projection).
fn load_clip_l(file: &Path) -> Result<ClipTextEncoder> {
    let mut w = Weights::from_file(file)?;
    w.cast_all(Dtype::Float32)?;
    ClipTextEncoder::from_weights(&w, "text_model", &sd3_clip_l_config())
}

/// Load CLIP-G / OpenCLIP-bigG (`text_encoder_2`) at f32 — 1280-wide with the 1280 pooled projection.
fn load_clip_g(file: &Path) -> Result<ClipTextEncoder> {
    let mut w = Weights::from_file(file)?;
    w.cast_all(Dtype::Float32)?;
    ClipTextEncoder::from_weights(&w, "text_model", &sd3_clip_g_config())
}

/// Load the three text encoders. CLIP-L + CLIP-G via the SDXL encoder at the `text_model` prefix; the
/// T5-XXL via the FLUX encoder at the empty prefix (sharded `text_encoder_3/` loaded as a dir). Loaded
/// dense at f32 (the CLIP path runs f32; the T5 promotes internally) — `quantize` is applied after.
pub fn load_text_encoders(root: &Path) -> Result<Sd3TextEncoders> {
    let artifacts = mlx_gen::gen_core::resolve_sd3_text_encoder_artifacts(root)
        .map_err(|error| Error::Msg(error.to_string()))?;
    let clip_l = load_clip_l(&artifacts.clip_l)?;
    let clip_g = load_clip_g(&artifacts.clip_g)?;
    let t5_w = load_t5_weights(&artifacts.t5_shards)?;
    let t5 = mlx_gen_flux::T5TextEncoder::from_weights(&t5_w, "")?;
    Ok(Sd3TextEncoders { clip_l, clip_g, t5 })
}

/// Load exactly the master T5 shards selected by the shared identity resolver. Reject duplicate
/// tensor keys across shards instead of letting later directory order silently overwrite them.
fn load_t5_weights(shards: &[std::path::PathBuf]) -> Result<Weights> {
    let mut merged = Weights::empty();
    let mut keys = BTreeSet::new();
    for shard in shards {
        let w = Weights::from_file(shard)?;
        for k in w.keys().map(String::from).collect::<Vec<_>>() {
            if !keys.insert(k.clone()) {
                return Err(Error::Msg(format!(
                    "sd3 t5: duplicate tensor key `{k}` in resolved master shard {}",
                    shard.display()
                )));
            }
            merged.insert(k.clone(), w.require(&k)?.clone());
        }
    }
    Ok(merged)
}

/// Load the CLIP BPE tokenizer (one instance serves both CLIP encoders — `tokenizer/` and
/// `tokenizer_2/` ship byte-identical `vocab.json` + `merges.txt`).
pub fn load_clip_tokenizer(root: &Path) -> Result<ClipBpeTokenizer> {
    ClipBpeTokenizer::from_dir(root.join("tokenizer"))
}

/// Load the T5 tokenizer from `tokenizer_3/tokenizer.json`, configured to pad to SD3's 256-token T5
/// window with the `<pad>` (0) token — diffusers `padding="max_length", max_length=256`.
pub fn load_t5_tokenizer(root: &Path) -> Result<TextTokenizer> {
    let config = TokenizerConfig {
        max_length: T5_MAX_LENGTH,
        pad_token_id: T5_PAD_ID,
        chat_template: ChatTemplate::None,
        pad_to_max_length: true,
    };
    Ok(TextTokenizer::from_file(
        root.join("tokenizer_3").join("tokenizer.json"),
        config,
    )?)
}

/// Load the MMDiT transformer from `transformer/` (sharded; auto dense-vs-prequantized per Linear).
pub fn load_transformer(root: &Path, arch: &Sd3Arch) -> Result<Sd3Transformer> {
    Sd3Transformer::from_dir(&root.join("transformer"), arch)
}

/// Load the 16-channel VAE (decoder + encoder) from `vae/` via the E4 reuse path.
pub fn load_vae(root: &Path) -> Result<mlx_gen_z_image::vae::Vae> {
    load_sd3_vae(&root.join("vae"))
}
