//! MiniMax-H3's **Qwen3-VL-32B** condition encoder (sc-17156, candle) — the `context` the H3 DiT
//! consumes.
//!
//! A 64-layer decoder-only LM whose hidden state after **50** layers is handed to the
//! H3-Omni-Transformer. The candle sibling of `mlx_gen_minimax_h3::text_encoder`, and structurally
//! the same tower `candle-gen-boogu` already ships for Qwen3-VL-8B — at 32 B, with a **mid-stack
//! tap** instead of `last_hidden_state`, and therefore **no final norm**.
//!
//! # Provenance: the weights are upstream Qwen, unmodified
//!
//! All 14 text-encoder shards are **byte-identical** (LFS SHA-256) to
//! `Qwen/Qwen3-VL-32B-Instruct` — 66,714,912,872 bytes, 0 differing files. The *only* file MiniMax
//! changed in the whole component is `tokenizer_config.json`. The model card agrees: "The H3-Encoder
//! uses the full pretrained weights of Qwen3-VL-32B and provides the hidden states from its 50th
//! layer". [`MiniMaxH3TeConfig::qwen3_vl_32b`] therefore describes an unmodified upstream
//! checkpoint.
//!
//! # The three knobs that no tensor can show you
//!
//! sc-17140 and sc-17141 both lost time to configuration that leaves no trace in the tensor index.
//! Three such knobs exist here, and each carries its own test:
//!
//! 1. **[`SELECT_HIDDEN`] = 50** lives only in the model card — no config file names it. It is an
//!    HF `output_hidden_states` index, so it captures the output of 0-indexed layer **49** and
//!    layers 50-63 are never run.
//! 2. **`APPLIES_CHAT_TEMPLATE` = false** — the conditioner tokenizes the presentation verbatim
//!    with `add_special_tokens=False`. See [`tokenizer`].
//! 3. **The seven MiniMax special-token ids are assigned by list ORDER at load time** and appear in
//!    no vocabulary file. See [`tokenizer`].
//!
//! # What this lane loads, and what it deliberately does not
//!
//! The MLX sibling maps whole shards eagerly, so it enumerates the shard *files* it needs (1-12 for
//! `t2va`, plus 14 for the vision tower). candle reads through a header-only mmap, so the trim is
//! expressed as a **key prefix set** instead: [`lm_prefixes`] names `embed_tokens` and layers
//! `0..select_hidden` and nothing else, so `{prefix}.norm.weight`, layers 50-63 and `lm_head.weight`
//! are never materialized on any device. The two backends therefore arrive at the same ~53 GB
//! conditioning working set by different routes, and `tests::lm_prefixes_stop_at_the_selected_layer`
//! pins that the prefix list cannot silently widen.
//!
//! The **vision tower** is consumed from `candle-gen-boogu` rather than duplicated — H3's
//! `vision_config` is identical to boogu's in every field but `out_hidden_size` (5120 vs 4096),
//! exactly as on the MLX lane.

pub mod encoder;
pub mod tokenizer;

pub use encoder::MiniMaxH3TextEncoder;
pub use tokenizer::{
    MiniMaxH3Tokenizer, SpecialTokens, APPLIES_CHAT_TEMPLATE, MINIMAX_ADDED_SPECIALS,
};

use std::path::Path;

use candle_gen::candle_core::{DType, Device, Tensor};
use candle_gen::gen_core::Image;
use candle_gen::{CandleError, Result};
use candle_gen_boogu::vision::preprocess::preprocess_image;
use candle_gen_boogu::vision::{VisionConfig, VisionTower};

/// Weight-key prefix of the text tower inside `text_encoder/`. The published checkpoint nests the
/// Qwen3-VL submodules under `model.` (`model.language_model.*`, `model.visual.*`, plus a top-level
/// `lm_head.weight`).
pub const LM_PREFIX: &str = "model.language_model";

/// Weight-key prefix of the vision tower inside `text_encoder/` (351 of the checkpoint's 1058
/// tensors, all in shard 14).
pub const VISION_PREFIX: &str = "model.visual";

/// The HF `output_hidden_states` index the H3 DiT consumes.
///
/// **This number exists only in the model card** ("the hidden states from its 50th layer"); no
/// shipped config declares it. HF indexing means `hidden_states[50]` is the state after running 50
/// decoder layers, i.e. the output of **0-indexed layer 49** — so the encoder loads and runs layers
/// `0..=49` and never touches layers 50-63 or the final `norm`.
pub const SELECT_HIDDEN: usize = 50;

/// Total decoder layers in the published checkpoint. Only [`SELECT_HIDDEN`] of them are ever run.
pub const NUM_HIDDEN_LAYERS: usize = 64;

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
    pub hidden_size: usize,
    /// `text_config.num_hidden_layers` — 64 in the checkpoint. Only `select_hidden` are run.
    pub num_layers: usize,
    /// `text_config.num_attention_heads` — 64.
    pub num_heads: usize,
    /// `text_config.num_key_value_heads` — 8 (GQA, 8× repeat).
    pub num_kv_heads: usize,
    /// `text_config.head_dim` — 128. NB `64 × 128 = 8192 ≠ hidden_size`, so the q projection is not
    /// square; the block must not infer `head_dim` from `hidden_size / num_heads`.
    pub head_dim: usize,
    /// `text_config.intermediate_size` — 25600.
    pub intermediate_size: usize,
    /// `text_config.rms_norm_eps` — 1e-6.
    pub rms_norm_eps: f64,
    /// `text_config.rope_theta` — 5e6.
    pub rope_theta: f32,
    /// `text_config.vocab_size` — 151936. Larger than the 151676 ids the tokenizer can produce; the
    /// tail is padding, which is why MiniMax's added tokens land on untrained rows (see
    /// [`tokenizer`]).
    pub vocab_size: usize,
    /// The HF `output_hidden_states` index handed to the DiT — [`SELECT_HIDDEN`].
    pub select_hidden: usize,
    /// `image_token_id` (top-level, not under `text_config`) — 151655.
    pub image_token_id: u32,
    /// `video_token_id` — 151656. Ref2VA's video references occupy `<|video_pad|>` runs (sc-17157).
    pub video_token_id: u32,
    /// `vision_start_token_id` — 151652.
    pub vision_start_token_id: u32,
    /// `vision_end_token_id` — 151653.
    pub vision_end_token_id: u32,
    /// `text_config.rope_scaling.mrope_section` `[T, H, W]` over `head_dim/2`, summing to 64. A
    /// text-only prompt reduces to plain 1-D RoPE (every section indexes the same sequential
    /// position); the vision path uses all three.
    pub mrope_section: [usize; 3],
    /// `text_config.rope_scaling.mrope_interleaved` — `true` for H3.
    ///
    /// **Load-bearing and invisible in the tensor index**: it selects *interleaved* section
    /// assignment (frequency `j` belongs to axis `j % 3` within the section span) over the
    /// *contiguous* blocking older Qwen-VL used. `candle_gen::grounding::mrope_cos_sin` implements
    /// the interleaved form and only that form, so this field is a **parse assertion** here rather
    /// than a dispatch: `tests::mrope_interleaved_is_load_bearing` pins the parse and
    /// [`encoder::MiniMaxH3TextEncoder::from_weights`] refuses a config that declares `false`,
    /// because silently running the interleaved kernel under a contiguous declaration is exactly the
    /// class of divergence nothing downstream can see.
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
            vocab_size: 151_936,
            select_hidden: SELECT_HIDDEN,
            image_token_id: 151_655,
            video_token_id: 151_656,
            vision_start_token_id: 151_652,
            vision_end_token_id: 151_653,
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
            CandleError::Msg(
                "minimax-h3 te: select_hidden 0 is the raw embedding, not a layer".into(),
            )
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
        let text = std::fs::read_to_string(&path).map_err(|e| {
            CandleError::Msg(format!("minimax-h3 te: read {}: {e}", path.display()))
        })?;
        Self::from_config_json(&text)
    }

    /// Parse a `text_encoder/config.json` document. Split out from [`Self::from_snapshot`] so the
    /// parse is testable without a 66.7 GB checkout.
    pub fn from_config_json(text: &str) -> Result<Self> {
        let v: serde_json::Value = serde_json::from_str(text)
            .map_err(|e| CandleError::Msg(format!("minimax-h3 te: parse config.json: {e}")))?;
        let tc = v.get("text_config").unwrap_or(&v);
        let d = Self::qwen3_vl_32b();
        let usize_of = |o: &serde_json::Value, k: &str, dflt: usize| {
            o.get(k)
                .and_then(serde_json::Value::as_u64)
                .map(|n| n as usize)
                .unwrap_or(dflt)
        };
        let u32_of = |o: &serde_json::Value, k: &str, dflt: u32| {
            o.get(k)
                .and_then(serde_json::Value::as_u64)
                .map(|n| n as u32)
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
            hidden_size: usize_of(tc, "hidden_size", d.hidden_size),
            num_layers: usize_of(tc, "num_hidden_layers", d.num_layers),
            num_heads: usize_of(tc, "num_attention_heads", d.num_heads),
            num_kv_heads: usize_of(tc, "num_key_value_heads", d.num_kv_heads),
            head_dim: usize_of(tc, "head_dim", d.head_dim),
            intermediate_size: usize_of(tc, "intermediate_size", d.intermediate_size),
            rms_norm_eps: tc
                .get("rms_norm_eps")
                .and_then(serde_json::Value::as_f64)
                .unwrap_or(d.rms_norm_eps),
            rope_theta: rope
                .get("rope_theta")
                .or_else(|| tc.get("rope_theta"))
                .and_then(serde_json::Value::as_f64)
                .map(|n| n as f32)
                .unwrap_or(d.rope_theta),
            vocab_size: usize_of(tc, "vocab_size", d.vocab_size),
            // No config file declares either of these. See the module docs.
            select_hidden: d.select_hidden,
            image_token_id: u32_of(&v, "image_token_id", d.image_token_id),
            video_token_id: u32_of(&v, "video_token_id", d.video_token_id),
            vision_start_token_id: u32_of(&v, "vision_start_token_id", d.vision_start_token_id),
            vision_end_token_id: u32_of(&v, "vision_end_token_id", d.vision_end_token_id),
            mrope_section: rope
                .get("mrope_section")
                .and_then(|a| a.as_array())
                .and_then(|a| {
                    let sec: Vec<usize> = a
                        .iter()
                        .filter_map(|x| x.as_u64().map(|n| n as usize))
                        .collect();
                    <[usize; 3]>::try_from(sec).ok()
                })
                .unwrap_or(d.mrope_section),
            mrope_interleaved: rope
                .get("mrope_interleaved")
                .and_then(serde_json::Value::as_bool)
                .unwrap_or(d.mrope_interleaved),
        })
    }
}

/// The exact weight-key prefixes the LM tower reads out of `text_encoder/`, and **nothing else**.
///
/// This is the candle expression of the MLX lane's shard trim. `{prefix}.embed_tokens.` plus one
/// entry per layer that will actually run; the trailing dot is load-bearing (`layers.5.` must not
/// match `layers.50.`). `{prefix}.norm.weight` and `lm_head.weight` are absent by construction, so
/// the never-executed tail is never read off disk, let alone put on a device.
pub fn lm_prefixes(prefix: &str, cfg: &MiniMaxH3TeConfig) -> Vec<String> {
    let mut out = Vec::with_capacity(cfg.layers_to_run() + 1);
    out.push(format!("{prefix}.embed_tokens."));
    for i in 0..cfg.layers_to_run() {
        out.push(format!("{prefix}.layers.{i}."));
    }
    out
}

/// H3's Qwen3-VL-32B **vision tower** config, read from the shipped `text_encoder/config.json`
/// `vision_config`.
///
/// Identical to [`VisionConfig::qwen3_vl`] (boogu's 8B tower) in every field except
/// `out_hidden_size`, which must equal the LM width so merged image embeds splice straight into the
/// token stream: **5120** here vs boogu's 4096. Feeds the shared [`VisionTower`] so the
/// parity-critical tower is not duplicated.
///
/// The MLX sibling's config additionally carries `intermediate_size: 4304`; candle's `VisionConfig`
/// has no such field because its tower infers the MLP width from the `linear_fc1` / `linear_fc2`
/// weight shapes. That is a loader difference, not a geometry one — the same 4304 is read off the
/// checkpoint either way — so there is nothing to declare here, and
/// `tests::vision_config_matches_boogu_except_the_lm_width` compares only fields that exist on
/// both.
pub fn minimax_h3_vision_config() -> VisionConfig {
    VisionConfig {
        hidden_size: 1152,
        num_heads: 16,
        depth: 27,
        out_hidden_size: 5120,
        patch_size: 16,
        temporal_patch_size: 2,
        spatial_merge_size: SPATIAL_MERGE as usize,
        in_channels: 3,
        num_position_embeddings: 2304,
        deepstack_visual_indexes: vec![8, 16, 24],
    }
}

/// Load the shared Qwen3-VL vision tower out of a snapshot's `text_encoder/` component.
///
/// The tower lives entirely in shard 14 (all 351 `model.visual.*` tensors), alongside the
/// never-executed decoder tail — so a *directory* read that materialized everything would carry
/// ~15 GB for nothing. `candle_gen_boogu::loader::Weights` mmaps headers and materializes only the
/// tensors the tower asks for by name, so the prefix does the trim without a shard allowlist.
///
/// Loaded **f32**, which is what boogu's tower requires (`Weights::get_f32` throughout): the tower
/// is 0.9 GB, so widening it is a rounding error against the 53 GB conditioning stage.
pub fn load_vision_tower(root: impl AsRef<Path>, device: &Device) -> Result<VisionTower> {
    let dir = root.as_ref().join("text_encoder");
    let w = candle_gen_boogu::loader::Weights::from_dir(&dir, device, DType::F32)?;
    Ok(VisionTower::load(
        &w,
        minimax_h3_vision_config(),
        VISION_PREFIX,
    )?)
}

/// The vision-tower output for a set of references: per-reference merged image embeds
/// `[nⱼ, hidden]`, the deepstack features (one stack per reference), the `[t, h, w]` patch grids,
/// and the merged-token counts that drive the presentation.
pub struct GroundedVision {
    /// Merged image embeds per reference, `[nⱼ, out_hidden]`.
    pub embeds: Vec<Tensor>,
    /// Deepstack features per reference — one `[nⱼ, out_hidden]` per `deepstack_visual_indexes`.
    pub deepstack: Vec<Vec<Tensor>>,
    /// `[t, h, w]` patch grid per reference.
    pub grids: Vec<[i32; 3]>,
    /// Merged vision-token count per reference — the number of `<|image_pad|>` placeholders.
    pub counts: Vec<usize>,
}

/// Run the shared Qwen3-VL [`VisionTower`] over every reference image, collecting the per-image
/// merged embeds / deepstack / grid + merged-token count. At least one source is required.
///
/// `device` must be the device the tower was loaded onto — candle tensors are device-bound (unlike
/// MLX arrays), and a preprocess on the wrong device surfaces as an opaque matmul mismatch deep
/// inside the ViT rather than at the call site. The pipeline builds tower and inputs from one
/// `Device` handle for exactly that reason.
pub fn run_vision(
    vision: &VisionTower,
    sources: &[&Image],
    device: &Device,
) -> Result<GroundedVision> {
    if sources.is_empty() {
        return Err(CandleError::Msg(
            "minimax-h3 te (grounded): at least one reference image is required".into(),
        ));
    }
    let mut embeds = Vec::with_capacity(sources.len());
    let mut deepstack = Vec::with_capacity(sources.len());
    let mut grids = Vec::with_capacity(sources.len());
    let mut counts = Vec::with_capacity(sources.len());
    for &src in sources {
        let (pixels, grid) =
            preprocess_image(&src.pixels, src.height as usize, src.width as usize, device)?;
        let (emb, ds) = vision.forward(&pixels, &[grid])?;
        counts.push(emb.dim(0)?);
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
    device: &Device,
) -> Result<Tensor> {
    let (ids, mask) = tok.encode_with_images(prompt, &gv.counts, device)?;
    te.forward_with_images(&ids, &mask, &gv.embeds, &gv.deepstack, &gv.grids)
}

/// Image-grounded condition encoding for one or more references: run the [`VisionTower`] over every
/// source, build the grounded presentation ids, and run the encoder.
pub fn encode_grounded(
    vision: &VisionTower,
    tok: &MiniMaxH3Tokenizer,
    te: &MiniMaxH3TextEncoder,
    sources: &[&Image],
    prompt: &str,
    device: &Device,
) -> Result<Tensor> {
    let gv = run_vision(vision, sources, device)?;
    encode_grounded_from_vision(&gv, tok, te, prompt, device)
}

/// The dtype the LM tower's projections are stored at.
///
/// The published shards are bf16, so a bf16 **store** only re-reads what is on disk while an f32
/// store would double the 53 GB conditioning stage to carry no extra precision. Compute is still
/// f32 — [`crate::nn::linear_nb`] casts the weight up to the activation dtype per matmul, and the
/// embedding is explicitly widened — which is the same store/compute split `candle-gen-boogu`
/// proved bit-identical to an f32 store for this exact tower (sc-12828).
pub const TE_STORE_DTYPE: DType = DType::BF16;

/// Compute dtype of the condition encoder — f32, the parity-grade precision for this tower.
pub const TE_COMPUTE_DTYPE: DType = DType::F32;

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
        assert_eq!(c.mrope_section.iter().sum::<usize>(), c.head_dim / 2);
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
        assert_eq!(c.num_layers - c.layers_to_run(), 14);
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
        assert_eq!(
            MiniMaxH3TeConfig::qwen3_vl_32b().hidden_size,
            crate::dit::MiniMaxH3DitConfig::default().text_dim
        );
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
        assert_eq!(h3.patch_size, boogu.patch_size);
        assert_eq!(h3.spatial_merge_size, boogu.spatial_merge_size);
        assert_eq!(h3.temporal_patch_size, boogu.temporal_patch_size);
        assert_eq!(h3.num_position_embeddings, boogu.num_position_embeddings);
        assert_eq!(h3.deepstack_visual_indexes, boogu.deepstack_visual_indexes);
    }

    /// **The trim is the prefix list.** It must name `embed_tokens` and exactly the layers that run,
    /// and it must not name the final norm, `lm_head`, or any layer at or past `select_hidden`.
    ///
    /// The trailing dot on each layer prefix is what stops `layers.5.` from also admitting
    /// `layers.50.` — a `starts_with` filter written without it would quietly load 10 extra layers,
    /// which is ~10 GB of weights and no test failure anywhere else.
    #[test]
    fn lm_prefixes_stop_at_the_selected_layer() {
        let cfg = MiniMaxH3TeConfig::qwen3_vl_32b();
        let p = lm_prefixes(LM_PREFIX, &cfg);
        assert_eq!(p.len(), cfg.layers_to_run() + 1);
        assert_eq!(p[0], "model.language_model.embed_tokens.");
        assert_eq!(p[1], "model.language_model.layers.0.");
        assert_eq!(p[50], "model.language_model.layers.49.");
        assert!(!p.iter().any(|x| x == "model.language_model.norm."));

        let admits = |key: &str| p.iter().any(|x| key.starts_with(x.as_str()));
        assert!(admits(
            "model.language_model.layers.49.self_attn.q_proj.weight"
        ));
        assert!(!admits(
            "model.language_model.layers.50.self_attn.q_proj.weight"
        ));
        // The dot guard: `layers.5.` must not swallow `layers.50.` … `layers.59.`.
        assert!(!admits("model.language_model.layers.55.mlp.up_proj.weight"));
        assert!(!admits("model.language_model.norm.weight"));
        assert!(!admits("lm_head.weight"));
        assert!(!admits("model.visual.blocks.0.attn.qkv.weight"));
    }

    /// The store/compute split is a declaration two other modules read, so it is pinned rather than
    /// left implicit: bf16 on disk and on the device, f32 through every matmul.
    #[test]
    fn the_encoder_stores_bf16_and_computes_f32() {
        assert_eq!(TE_STORE_DTYPE, DType::BF16);
        assert_eq!(TE_COMPUTE_DTYPE, DType::F32);
        assert_ne!(TE_STORE_DTYPE, TE_COMPUTE_DTYPE);
    }
}
