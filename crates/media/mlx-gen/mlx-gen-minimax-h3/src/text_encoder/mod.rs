//! MiniMax-H3's **Qwen3-VL-32B** condition encoder (sc-17143) — the `context` the H3 DiT consumes.
//!
//! A 64-layer decoder-only LM whose hidden state after **50** layers is handed to the
//! H3-Omni-Transformer. Structurally this is the same trick `mlx-gen-krea` already ships for
//! Qwen3-VL-4B (sc-7569), at 32B and with a **single** select layer instead of a 12-layer stack —
//! so the DiT context is `[B, L, 5120]`, not the rank-4 stack Krea produces.
//!
//! **Unlike Krea, H3 has no template-prefix slice.** Krea's conditioner does render a template and
//! drop its prefix; H3's does not, and copying that half of the pattern is what sc-18741 corrects.
//!
//! # Provenance: the weights are upstream Qwen, unmodified
//!
//! All 14 text-encoder shards are **byte-identical** (LFS SHA-256) to
//! `Qwen/Qwen3-VL-32B-Instruct` — 66,714,912,872 bytes, 0 differing files. `config.json`,
//! `chat_template.json`, `tokenizer.json` and `vocab.json` are identical too. The *only* file
//! MiniMax changed in the whole component is `tokenizer_config.json`. The model card agrees: "The
//! H3-Encoder uses the full pretrained weights of Qwen3-VL-32B and provides the hidden states from
//! its 50th layer". [`MiniMaxH3TeConfig::qwen3_vl_32b`] therefore describes an unmodified upstream
//! checkpoint, and the TE may be sourced from Qwen's repo with a `tokenizer_config.json` overlay
//! rather than rehosting 66.73 GB.
//!
//! # The three knobs that no tensor can show you
//!
//! sc-17140 and sc-17141 both lost time to configuration that leaves no trace in the tensor index.
//! Three such knobs exist here, and each carries its own test:
//!
//! 1. **[`SELECT_HIDDEN`] = 50** lives only in the model card — no config file names it. It is an
//!    HF `output_hidden_states` index, so it captures the output of 0-indexed layer **49** and
//!    layers 50-63 are never run. `tests/te_parity.rs`'s
//!    `layer_selection_is_off_by_one_safe_in_both_directions` pins it against the reference in both
//!    directions.
//! 2. **[`APPLIES_CHAT_TEMPLATE`] = false** — the conditioner tokenizes the presentation verbatim
//!    with `add_special_tokens=False`. No config declares that either; it is a property of the
//!    reference conditioner, and sc-17143 derived the opposite before that reference existed.
//! 3. **The seven MiniMax special-token ids are assigned by list ORDER at load time** and appear in
//!    no vocabulary file. See [`tokenizer`].

pub mod attention;
pub mod encoder;
pub mod layer;
pub mod mlp;
pub mod tokenizer;

pub use attention::Qwen3Attention;
pub use encoder::MiniMaxH3TextEncoder;
pub use layer::Qwen3DecoderLayer;
pub use mlp::Qwen3Mlp;
pub use tokenizer::{
    MiniMaxH3Tokenizer, SpecialTokens, APPLIES_CHAT_TEMPLATE, MINIMAX_ADDED_SPECIALS,
};

// The HF half-split text RoPE is identical across families and lives in core.
pub use mlx_gen::nn::TextRope;

use std::path::Path;

use image::RgbImage;
use mlx_rs::Array;

use mlx_gen::adapters::AdaptableLinear;
use mlx_gen::media::Image;
use mlx_gen::nn::TokenEmbedding;
use mlx_gen::weights::Weights;
use mlx_gen::{Error, Result};
use mlx_gen_boogu::{vision::preprocess::preprocess_image, VisionConfig, VisionTower};

/// Weight-key prefix of the text tower inside `text_encoder/`. The published checkpoint nests the
/// Qwen3-VL submodules under `model.` (`model.language_model.*`, `model.visual.*`, plus a top-level
/// `lm_head.weight`) — unlike Krea's diffusers repackaging, which drops the `model.` level.
pub const LM_PREFIX: &str = "model.language_model";

/// Weight-key prefix of the vision tower inside `text_encoder/` (351 of the checkpoint's 1058
/// tensors).
pub const VISION_PREFIX: &str = "model.visual";

/// Quantization group size for this crate's Qwen3 projections (the `mlx-gen` default).
pub const GROUP_SIZE: i32 = 64;

/// The HF `output_hidden_states` index the H3 DiT consumes.
///
/// **This number exists only in the model card** ("the hidden states from its 50th layer"); no
/// shipped config declares it. HF indexing means `hidden_states[50]` is the state after running 50
/// decoder layers, i.e. the output of **0-indexed layer 49** — so the encoder loads and runs layers
/// `0..=49` and never touches layers 50-63 or the final `norm`.
///
/// Two consequences worth stating plainly, because sc-17139 depends on them:
/// - decoder layers **50-63 and `lm_head` are dead weight** for generation, and can be trimmed;
/// - a one-off in either direction silently changes the conditioning, so
///   `encoder::tests` asserts inequality against `hidden_states[49]` and `[51]`, not just equality
///   against `[50]`.
pub const SELECT_HIDDEN: usize = 50;

/// Total decoder layers in the published checkpoint. Only [`SELECT_HIDDEN`] of them are ever run.
pub const NUM_HIDDEN_LAYERS: i32 = 64;

/// Qwen3-VL spatial merge factor (`spatial_merge_size`); a `merge×merge` block of ViT patches
/// collapses to one LM image token. Fixed at 2 across the family.
pub(crate) const SPATIAL_MERGE: i32 = 2;

/// Qwen3-VL-32B text-tower architecture (verified against the published `text_encoder/config.json`
/// `text_config`: `qwen3_vl_text`, hidden 5120, 64 layers, GQA 64/8, `head_dim` 128, FFN 25600,
/// eps 1e-6, θ 5e6) plus the H3 conditioning policy (which hidden state to take).
#[derive(Debug, Clone, PartialEq)]
pub struct MiniMaxH3TeConfig {
    /// `text_config.hidden_size` — 5120, and equal to the DiT's `text_dim`, so the context needs no
    /// projection.
    pub hidden_size: i32,
    /// `text_config.num_hidden_layers` — 64 in the checkpoint. Only `select_hidden` are run.
    pub num_layers: i32,
    /// `text_config.num_attention_heads` — 64.
    pub num_heads: i32,
    /// `text_config.num_key_value_heads` — 8 (GQA, 8× repeat).
    pub num_kv_heads: i32,
    /// `text_config.head_dim` — 128. NB `64 × 128 = 8192 ≠ hidden_size`, so the q projection is not
    /// square; the block must not infer `head_dim` from `hidden_size / num_heads`.
    pub head_dim: i32,
    /// `text_config.intermediate_size` — 25600.
    pub intermediate_size: i32,
    /// `text_config.rms_norm_eps` — 1e-6.
    pub rms_norm_eps: f32,
    /// `text_config.rope_theta` — 5e6.
    pub rope_theta: f32,
    /// `text_config.vocab_size` — 151936. Larger than the 151676 ids the tokenizer can produce; the
    /// tail is padding, which is why MiniMax's added tokens land on untrained rows (see
    /// [`tokenizer`]).
    pub vocab_size: i32,
    /// The HF `output_hidden_states` index handed to the DiT — [`SELECT_HIDDEN`].
    pub select_hidden: usize,
    /// `image_token_id` (top-level, not under `text_config`) — 151655.
    pub image_token_id: i32,
    /// `video_token_id` — 151656. Ref2VA's video references occupy `<|video_pad|>` runs.
    pub video_token_id: i32,
    /// `vision_start_token_id` — 151652.
    pub vision_start_token_id: i32,
    /// `vision_end_token_id` — 151653.
    pub vision_end_token_id: i32,
    /// `text_config.rope_scaling.mrope_section` `[T, H, W]` over `head_dim/2`, summing to 64. A
    /// text-only prompt reduces to plain 1-D RoPE (every section indexes the same sequential
    /// position); the vision path uses all three.
    pub mrope_section: [i32; 3],
    /// `text_config.rope_scaling.mrope_interleaved` — `true` for H3.
    ///
    /// **Load-bearing and invisible in the tensor index**, exactly the sc-17140 `qk_norm_type`
    /// hazard: it selects *interleaved* section assignment (frequency `j` belongs to axis `j % 3`
    /// within the section span) over the *contiguous* blocking older Qwen-VL used. It leaves no
    /// tensor behind, so `tests::mrope_interleaved_is_load_bearing` pins the parse and
    /// `encoder::tests::interleaved_mrope_routes_h_and_w_to_distinct_frequencies` pins the effect.
    pub mrope_interleaved: bool,
}

impl Default for MiniMaxH3TeConfig {
    fn default() -> Self {
        Self::qwen3_vl_32b()
    }
}

impl MiniMaxH3TeConfig {
    /// The shipped `text_encoder/config.json` values.
    pub fn qwen3_vl_32b() -> Self {
        Self {
            hidden_size: 5120,
            num_layers: NUM_HIDDEN_LAYERS,
            num_heads: 64,
            num_kv_heads: 8,
            head_dim: 128,
            intermediate_size: 25600,
            rms_norm_eps: 1e-6,
            rope_theta: 5_000_000.0,
            vocab_size: 151936,
            select_hidden: SELECT_HIDDEN,
            image_token_id: 151655,
            video_token_id: 151656,
            vision_start_token_id: 151652,
            vision_end_token_id: 151653,
            mrope_section: [24, 20, 20],
            mrope_interleaved: true,
        }
    }

    /// 0-indexed decoder layer whose OUTPUT is the DiT context — `select_hidden - 1`.
    ///
    /// This is the whole off-by-one surface, in one place: HF `hidden_states[k]` is the state after
    /// `k` layers, so index 50 is layer 49's output.
    pub fn out_layer(&self) -> Result<usize> {
        self.select_hidden.checked_sub(1).ok_or_else(|| {
            Error::Msg("minimax-h3 te: select_hidden 0 is the raw embedding, not a layer".into())
        })
    }

    /// Number of decoder layers that must be loaded and run — `select_hidden`. The remaining
    /// `num_layers - select_hidden` (14 of 64) are never executed and need not be resident.
    pub fn layers_to_run(&self) -> usize {
        self.select_hidden
    }

    /// Parse `<root>/text_encoder/config.json`. Missing scalars fall back to
    /// [`Self::qwen3_vl_32b`]; `select_hidden` has no config home at all (see the module docs) and
    /// always comes from the constant.
    pub fn from_snapshot(root: impl AsRef<Path>) -> Result<Self> {
        let path = root.as_ref().join("text_encoder").join("config.json");
        let text = std::fs::read_to_string(&path)
            .map_err(|e| Error::Msg(format!("minimax-h3 te: read {}: {e}", path.display())))?;
        Self::from_config_json(&text)
    }

    /// Parse a `text_encoder/config.json` document. Split out from [`Self::from_snapshot`] so the
    /// parse is testable without a 66.7 GB checkout.
    pub fn from_config_json(text: &str) -> Result<Self> {
        let v: serde_json::Value = serde_json::from_str(text)
            .map_err(|e| Error::Msg(format!("minimax-h3 te: parse config.json: {e}")))?;
        let tc = v.get("text_config").unwrap_or(&v);
        let d = Self::qwen3_vl_32b();
        let i32_of = |o: &serde_json::Value, k: &str, dflt: i32| {
            o.get(k)
                .and_then(serde_json::Value::as_i64)
                .map(|n| n as i32)
                .unwrap_or(dflt)
        };
        // `rope_scaling` is the shipped spelling; `rope_parameters` is what newer transformers
        // renames it to. Honor either.
        let rope = tc
            .get("rope_scaling")
            .or_else(|| tc.get("rope_parameters"))
            .cloned()
            .unwrap_or(serde_json::Value::Null);

        Ok(Self {
            hidden_size: i32_of(tc, "hidden_size", d.hidden_size),
            num_layers: i32_of(tc, "num_hidden_layers", d.num_layers),
            num_heads: i32_of(tc, "num_attention_heads", d.num_heads),
            num_kv_heads: i32_of(tc, "num_key_value_heads", d.num_kv_heads),
            head_dim: i32_of(tc, "head_dim", d.head_dim),
            intermediate_size: i32_of(tc, "intermediate_size", d.intermediate_size),
            rms_norm_eps: tc
                .get("rms_norm_eps")
                .and_then(serde_json::Value::as_f64)
                .map(|n| n as f32)
                .unwrap_or(d.rms_norm_eps),
            rope_theta: rope
                .get("rope_theta")
                .or_else(|| tc.get("rope_theta"))
                .and_then(serde_json::Value::as_f64)
                .map(|n| n as f32)
                .unwrap_or(d.rope_theta),
            vocab_size: i32_of(tc, "vocab_size", d.vocab_size),
            // No config file declares either of these. See the module docs.
            select_hidden: d.select_hidden,
            image_token_id: i32_of(&v, "image_token_id", d.image_token_id),
            video_token_id: i32_of(&v, "video_token_id", d.video_token_id),
            vision_start_token_id: i32_of(&v, "vision_start_token_id", d.vision_start_token_id),
            vision_end_token_id: i32_of(&v, "vision_end_token_id", d.vision_end_token_id),
            mrope_section: rope
                .get("mrope_section")
                .and_then(|a| a.as_array())
                .and_then(|a| {
                    let sec: Vec<i32> = a
                        .iter()
                        .filter_map(|x| x.as_i64().map(|n| n as i32))
                        .collect();
                    <[i32; 3]>::try_from(sec).ok()
                })
                .unwrap_or(d.mrope_section),
            mrope_interleaved: rope
                .get("mrope_interleaved")
                .and_then(serde_json::Value::as_bool)
                .unwrap_or(d.mrope_interleaved),
        })
    }
}

/// H3's Qwen3-VL-32B **vision tower** config, read from the shipped `text_encoder/config.json`
/// `vision_config`.
///
/// Identical to [`VisionConfig::qwen3_vl`] (boogu's 8B tower) in every field except
/// `out_hidden_size`, which must equal the LM width so merged image embeds splice straight into the
/// token stream: **5120** here vs boogu's 4096. Feeds the shared [`VisionTower`] so the
/// parity-critical tower is not duplicated a fifth time.
pub fn minimax_h3_vision_config() -> VisionConfig {
    VisionConfig {
        hidden_size: 1152,
        num_heads: 16,
        intermediate_size: 4304,
        depth: 27,
        out_hidden_size: 5120,
        patch_size: 16,
        temporal_patch_size: 2,
        spatial_merge_size: SPATIAL_MERGE,
        in_channels: 3,
        num_position_embeddings: 2304,
        deepstack_visual_indexes: vec![8, 16, 24],
    }
}

/// The vision-tower output for a set of references, computed once and reusable across CFG branches:
/// per-reference merged image embeds `[nⱼ, hidden]`, the deepstack features (one stack per
/// reference), the `[t, h, w]` patch grids, and the merged-token counts that drive the template.
pub struct GroundedVision {
    /// Merged image embeds per reference, `[nⱼ, out_hidden]`.
    pub embeds: Vec<Array>,
    /// Deepstack features per reference — one `[nⱼ, out_hidden]` per `deepstack_visual_indexes`.
    pub deepstack: Vec<Vec<Array>>,
    /// `[t, h, w]` patch grid per reference.
    pub grids: Vec<[i32; 3]>,
    /// Merged vision-token count per reference — the number of `<|image_pad|>` placeholders.
    pub counts: Vec<usize>,
}

/// Run the shared Qwen3-VL [`VisionTower`] over every reference image, collecting the per-image
/// merged embeds / deepstack / grid + merged-token count.
///
/// Split from [`encode_grounded_from_vision`] so a CFG pass runs the tower ONCE and grounds both the
/// positive and negative prompt on the shared output — the tower forward is the expensive,
/// text-independent half. At least one source is required.
pub fn run_vision(vision: &VisionTower, sources: &[&Image]) -> Result<GroundedVision> {
    if sources.is_empty() {
        return Err(Error::Msg(
            "minimax-h3 te (grounded): at least one reference image is required".into(),
        ));
    }
    let mut embeds = Vec::with_capacity(sources.len());
    let mut deepstack = Vec::with_capacity(sources.len());
    let mut grids = Vec::with_capacity(sources.len());
    let mut counts = Vec::with_capacity(sources.len());
    for &src in sources {
        let rgb =
            RgbImage::from_raw(src.width, src.height, src.pixels.clone()).ok_or_else(|| {
                Error::Msg("minimax-h3 te (grounded): reference image is not valid RGB8".into())
            })?;
        let (pixels, grid) = preprocess_image(&rgb)?;
        let (emb, ds) = vision.forward(&pixels, &[grid])?;
        counts.push(emb.shape()[0] as usize);
        embeds.push(emb);
        deepstack.push(ds);
        grids.push(grid);
    }
    Ok(GroundedVision {
        embeds,
        deepstack,
        grids,
        counts,
    })
}

/// Build the DiT-consumable grounded context `[1, s, hidden]` for one prompt, reusing a
/// pre-computed [`GroundedVision`]: build the presentation — a `"<Picture i>: "` label and an
/// `<|image_pad|>` block per reference, then the prompt verbatim — and run
/// [`MiniMaxH3TextEncoder::forward_with_images`].
pub fn encode_grounded_from_vision(
    gv: &GroundedVision,
    tok: &MiniMaxH3Tokenizer,
    te: &MiniMaxH3TextEncoder,
    prompt: &str,
) -> Result<Array> {
    let (ids, mask) = tok.encode_with_images(prompt, &gv.counts)?;
    te.forward_with_images(&ids, &mask, &gv.embeds, &gv.deepstack, &gv.grids)
}

/// Image-grounded condition encoding for one or more references: run the [`VisionTower`] over every
/// source, build the grounded template ids, and run the encoder. Callers building both a positive
/// and a negative context should instead call [`run_vision`] once and
/// [`encode_grounded_from_vision`] per prompt, to avoid a redundant tower forward.
pub fn encode_grounded(
    vision: &VisionTower,
    tok: &MiniMaxH3Tokenizer,
    te: &MiniMaxH3TextEncoder,
    sources: &[&Image],
    prompt: &str,
) -> Result<Array> {
    let gv = run_vision(vision, sources)?;
    encode_grounded_from_vision(&gv, tok, te, prompt)
}

/// Text-encoder shards the layer-50 tap needs. Shard 13 holds only the never-executed tail (layers
/// 58-63), so mapping the whole 66.7 GB component would read 4.9 GB for nothing — and the shipped
/// manifest does not even download it.
pub const TE_SHARDS: std::ops::RangeInclusive<u32> = 1..=12;

/// The shard holding the Qwen3-VL **vision tower**. All 351 `model.visual.*` tensors live in shard
/// 14 and nowhere else, so [`TE_SHARDS`] excludes it for `t2va` and `fl2va` must add it back.
pub const VISION_SHARD: u32 = 14;

/// Map the text-encoder shards a presentation needs, out of `dir`.
///
/// `t2va` reads shards 1-12 ([`TE_SHARDS`]). `fl2va` / `ref2va` additionally need
/// [`VISION_SHARD`], from which only `model.visual.*` is kept — shard 14 also holds layers 62-63,
/// the final norm and `lm_head`, none of which is ever executed, so the grounded path would
/// otherwise carry them for nothing.
///
/// # The shard file names are the addressing scheme, not a convention
///
/// This function does **not** glob: it names `model-{i:05}-of-00014.safetensors` for a fixed set of
/// `i`. A packed tier
/// ([`quantize_minimax_h3_text_encoder`](crate::convert::quantize_minimax_h3_text_encoder))
/// therefore preserves the source shard names exactly — a re-sharded tier would silently map a
/// different set of layers, and every downstream shape check would still pass.
///
/// Public so the sc-19120 tier measurement can exercise **this** mapping rather than a copy of it:
/// a per-stage memory figure taken from a re-implementation measures the re-implementation.
pub fn map_shards(dir: &Path, with_vision: bool) -> Result<Weights> {
    let mut w = Weights::empty();
    let shards: Vec<u32> = if with_vision {
        TE_SHARDS.chain(std::iter::once(VISION_SHARD)).collect()
    } else {
        TE_SHARDS.collect()
    };
    for i in shards {
        let shard = format!("model-{i:05}-of-00014.safetensors");
        let part = Weights::from_file(dir.join(&shard))?;
        let keys: Vec<String> = part.keys().map(str::to_owned).collect();
        for k in keys {
            if with_vision && i == VISION_SHARD && !k.starts_with(VISION_PREFIX) {
                continue;
            }
            let t = part.require(&k)?.clone();
            w.insert(k, t);
        }
    }
    Ok(w)
}

/// Load a bias-less Qwen3 projection from its `{base}.weight` key, auto-detecting a pre-quantized
/// snapshot (`{base}.scales` present) via the core loader.
pub(crate) fn lin(w: &Weights, key: &str) -> Result<AdaptableLinear> {
    let base = key.strip_suffix(".weight").unwrap_or(key);
    mlx_gen::quant::lin(w, base, false, GROUP_SIZE)
}

/// Load a token embedding, auto-detecting a pre-quantized snapshot.
pub(crate) fn embedding(w: &Weights, base: &str) -> Result<TokenEmbedding> {
    mlx_gen::quant::embedding(w, base, GROUP_SIZE)
}

/// The single packed width `group` shares, or `None` if any member is dense or they disagree.
///
/// Disagreement collapses to `None` rather than to the first member's width. A sub-module packed at
/// two widths is a mis-built tier, and reporting *a* width would let it read as a valid one — the
/// encoder-level [`MiniMaxH3TextEncoder::packed_bits`] turns the resulting `None`-among-`Some`s
/// into a named error.
pub(crate) fn uniform_packed_bits<'a>(
    group: impl IntoIterator<Item = &'a AdaptableLinear>,
) -> Option<i32> {
    let mut seen: Option<i32> = None;
    for l in group {
        let (.., bits) = l.quantized_params()?;
        match seen {
            None => seen = Some(bits),
            Some(b) if b != bits => return None,
            Some(_) => {}
        }
    }
    seen
}

/// The core key-join helper, re-exported so the block modules can spell `join_key(prefix, leaf)`
/// without each importing core directly (`mlx-gen-krea` / `-boogu` / `-ideogram` each re-spell an
/// identical private copy instead; this uses the one in core).
pub(crate) use mlx_gen::weights::join as join_key;

#[cfg(test)]
mod tests {
    use super::*;

    /// The published `text_encoder/config.json`, verbatim (the `text_config` + `vision_config`
    /// scalars this crate reads). Committed rather than read from a snapshot so the parse is
    /// covered without weights.
    const SHIPPED_CONFIG: &str = r#"{
      "architectures": ["Qwen3VLForConditionalGeneration"],
      "image_token_id": 151655,
      "model_type": "qwen3_vl",
      "text_config": {
        "attention_bias": false, "head_dim": 128, "hidden_act": "silu",
        "hidden_size": 5120, "intermediate_size": 25600,
        "max_position_embeddings": 262144, "model_type": "qwen3_vl_text",
        "num_attention_heads": 64, "num_hidden_layers": 64, "num_key_value_heads": 8,
        "rms_norm_eps": 1e-06,
        "rope_scaling": { "mrope_interleaved": true, "mrope_section": [24, 20, 20], "rope_type": "default" },
        "rope_theta": 5000000, "vocab_size": 151936
      },
      "video_token_id": 151656,
      "vision_config": { "deepstack_visual_indexes": [8, 16, 24], "depth": 27, "hidden_size": 1152,
        "intermediate_size": 4304, "num_heads": 16, "out_hidden_size": 5120, "patch_size": 16,
        "spatial_merge_size": 2, "temporal_patch_size": 2, "num_position_embeddings": 2304 },
      "vision_end_token_id": 151653,
      "vision_start_token_id": 151652
    }"#;

    /// Parsing the shipped config must reproduce the hardcoded defaults exactly. If upstream ever
    /// edits a scalar, this fails rather than silently drifting.
    #[test]
    fn shipped_config_parses_to_the_declared_defaults() {
        let parsed = MiniMaxH3TeConfig::from_config_json(SHIPPED_CONFIG).unwrap();
        assert_eq!(parsed, MiniMaxH3TeConfig::qwen3_vl_32b());
    }

    /// `head_dim` must come from the config, never from `hidden_size / num_heads` — for this model
    /// those differ (128 vs 80), so an inferred `head_dim` would be silently wrong.
    #[test]
    fn head_dim_is_not_derivable_from_hidden_size() {
        let c = MiniMaxH3TeConfig::qwen3_vl_32b();
        assert_eq!(c.head_dim, 128);
        assert_ne!(c.head_dim, c.hidden_size / c.num_heads);
        assert_eq!(c.hidden_size / c.num_heads, 80);
        // The q projection is therefore NOT square: 64 heads × 128 = 8192 out, 5120 in.
        assert_eq!(c.num_heads * c.head_dim, 8192);
    }

    /// The mrope sections must tile exactly `head_dim / 2`; a mis-parsed section silently
    /// re-assigns frequencies to the wrong axis.
    #[test]
    fn mrope_sections_tile_half_the_head_dim() {
        let c = MiniMaxH3TeConfig::qwen3_vl_32b();
        assert_eq!(c.mrope_section.iter().sum::<i32>(), c.head_dim / 2);
        assert_eq!(c.mrope_section, [24, 20, 20]);
    }

    /// `mrope_interleaved` leaves NO tensor behind, so nothing but an explicit assertion can catch
    /// it flipping. It is `true` for H3 — the interleaved section assignment, not contiguous
    /// blocking.
    #[test]
    fn mrope_interleaved_is_load_bearing() {
        assert!(MiniMaxH3TeConfig::qwen3_vl_32b().mrope_interleaved);
        assert!(
            MiniMaxH3TeConfig::from_config_json(SHIPPED_CONFIG)
                .unwrap()
                .mrope_interleaved
        );
        // A config that omits it must not silently read as `false`.
        let without = SHIPPED_CONFIG.replace("\"mrope_interleaved\": true,", "");
        assert!(
            MiniMaxH3TeConfig::from_config_json(&without)
                .unwrap()
                .mrope_interleaved
        );
    }

    /// The layer-50 arithmetic, stated once so the encoder cannot drift from it: HF index 50 is
    /// layer 49's output, 50 layers run, 14 unused.
    #[test]
    fn select_hidden_50_means_50_layers_run_and_14_unused() {
        let c = MiniMaxH3TeConfig::qwen3_vl_32b();
        assert_eq!(c.select_hidden, 50);
        assert_eq!(c.out_layer().unwrap(), 49);
        assert_eq!(c.layers_to_run(), 50);
        assert_eq!(c.num_layers as usize - c.layers_to_run(), 14);
    }

    /// `select_hidden = 0` is the raw embedding, which is not a layer output — a typed error, not a
    /// panicking underflow.
    #[test]
    fn select_hidden_zero_is_a_typed_error() {
        let mut c = MiniMaxH3TeConfig::qwen3_vl_32b();
        c.select_hidden = 0;
        let e = c.out_layer().unwrap_err().to_string();
        assert!(e.contains("select_hidden"), "unexpected error: {e}");
    }

    /// The context width must equal the DiT's `text_dim` (5120) — if it did not, the port would
    /// need a projection the reference does not have.
    #[test]
    fn context_width_matches_the_dit_text_dim() {
        assert_eq!(MiniMaxH3TeConfig::qwen3_vl_32b().hidden_size, 5120);
    }

    /// H3's vision tower differs from boogu's ONLY in `out_hidden_size`, and that field must track
    /// the LM width or the merged embeds cannot splice into the token stream.
    #[test]
    fn vision_config_matches_boogu_except_the_lm_width() {
        let h3 = minimax_h3_vision_config();
        let boogu = VisionConfig::qwen3_vl();
        assert_eq!(
            h3.out_hidden_size,
            MiniMaxH3TeConfig::qwen3_vl_32b().hidden_size
        );
        assert_eq!(h3.out_hidden_size, 5120);
        assert_ne!(h3.out_hidden_size, boogu.out_hidden_size);
        // Everything else is the shared tower.
        assert_eq!(h3.hidden_size, boogu.hidden_size);
        assert_eq!(h3.depth, boogu.depth);
        assert_eq!(h3.num_heads, boogu.num_heads);
        assert_eq!(h3.intermediate_size, boogu.intermediate_size);
        assert_eq!(h3.patch_size, boogu.patch_size);
        assert_eq!(h3.spatial_merge_size, boogu.spatial_merge_size);
        assert_eq!(h3.temporal_patch_size, boogu.temporal_patch_size);
        assert_eq!(h3.num_position_embeddings, boogu.num_position_embeddings);
        assert_eq!(h3.deepstack_visual_indexes, boogu.deepstack_visual_indexes);
    }
}
