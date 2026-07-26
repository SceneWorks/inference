//! Offline pre-quantization: read the dense published snapshot and write the physically distinct
//! `q4/` / `q8/` / `bf16/` tier artifacts the SceneWorks mirrors host (sc-14980).
//!
//! Mirrors `mlx_gen_z_image::convert` / `mlx_gen_flux2::convert` (same `mlx_rs::ops::quantize`,
//! byte-equal to the load-time [`AdaptableLinear::quantize`](mlx_gen::adapters::AdaptableLinear)),
//! differing only in Mage's key layout and the per-component predicates below.
//!
//! # Why Mage's tier layout is SPLIT, unlike every sibling
//!
//! `z-image` writes one self-contained `<tier>/` holding transformer + text encoder + VAE, because
//! it is a single model. Mage ships **six** variants whose text encoder (8.875 GB) and VAE
//! (0.345 GB) are **bit-identical** across all six — only the 8.232 GB DiT differs (epic 14034;
//! re-confirmed in sc-14059 via matching VAE LFS OIDs across `Mage-Flow-Base`/`Mage-Flow-Edit`).
//! Duplicating a self-contained tier per variant would cost 105 GB for a full install. So:
//!
//! - [`prequantize_variant_tier`] writes a **DiT-only** tier into each variant mirror.
//! - [`prequantize_shared_components`] writes the **shared** text encoder + VAE tier ONCE, into a
//!   single components mirror that all six variants resolve as a co-requisite.
//!
//! # What is packed, and what deliberately is not
//!
//! - **DiT — packed.** All 174 Linears ([`is_dit_target`]); the qk-RMSNorm scales stay dense.
//! - **Text encoder — packed.** The 253 LM projections + token embedding, and the Qwen3-VL vision
//!   tower's attention/MLP projections (the edit path loads it; `mlx_gen_boogu`'s vision loader is
//!   packed-aware). `pos_embed` and `patch_embed.proj` are excluded — see [`is_te_target`].
//! - **VAE — DENSE, never packed.** This is not an omission. 59 of the VAE's 60 quantizable 2-D
//!   weights are `pipeline.blocks.{i}.adaLN_modulation.1.weight`, which `MageVae::from_weights`
//!   **folds away at load** (`fold_decode`: adaLN depends only on `t`, and decode always runs
//!   `t = 0`, so the reference precomputes and frees those MLPs). The load-time `vae.quantize`
//!   therefore packs only the *live* post-fold projections — a set that does not exist on disk, so
//!   an on-disk pack of those keys would be folded as quantized codes and silently corrupt the
//!   decode. The disk cost of that correctness is ~0: the sibling z-image VAE shrinks 0.1677 →
//!   0.1647 GB (1.8%) from q4 packing, because the VAE is conv-dominated. Shipping it dense also
//!   keeps the sc-14046 memory envelope exactly as measured, since `vae.quantize` still runs at
//!   load.
//!
//! # Precision floors — why the Q4 tier is not uniformly Q4 (sc-15071)
//!
//! Two projections tolerate 8 bits and not 4. Packed uniformly at Q4 the tier did not render the
//! prompt at all — it produced a repeating tiled texture — so these are correctness floors, not
//! quality preferences. [`quant_floor_bits`] is the single seam both the offline converter here and
//! the load-time `quantize` methods call, so an artifact and a load-time tier cannot disagree.
//!
//! | base | floor | why |
//! |---|---|---|
//! | `norm_out.linear` | 8 | The output head's `AdaLayerNormContinuous` modulation. Its `scale`/`shift` come from `temb` — one vector per pack segment — so its error is a coherent per-channel distortion applied identically to every token, immediately after a non-affine LayerNorm has erased each token's own scale and with nothing downstream to dilute it. At Q4, on 0.5% of the DiT's parameters, it does 4× the damage of the next-worst group. |
//! | `model.language_model.layers.*` | 8 | The 36 Qwen3-VL decoder layers; the SwiGLU MLP specifically. The Wan/UMT5 text-encoder-floor precedent. Q4 text and Q4 image weights are not independently tolerable errors — each alone is recoverable, together they are not. |
//!
//! The token embedding, the vision tower and every other DiT projection keep the tier's own width;
//! flooring them buys nothing measurable and costs real bytes. Full per-group measurements are on
//! the two constants in `crate::quant`, and the executable proof is the mutation probe in
//! `tests/quant_real_weights.rs`.

use std::path::Path;

use mlx_gen::quant::{
    copy_asset, copy_dir, load_dir_map, quantize_map, save_map, write_quantized_config,
};
use mlx_gen::{Error, Result};

use crate::quant::{floor_bits, FINAL_MOD_BASE, GROUP_SIZE, LM_LAYER_PREFIX};
use crate::transformer::{TRANSFORMER_CONFIG_FILE, TRANSFORMER_WEIGHTS_FILE};

/// The tier subdirectory names the SceneWorks mirrors publish, in quality order.
///
/// `bf16` is a dense passthrough (no `quantization` marker); `q8`/`q4` are packed.
pub const TIERS: &[&str] = &["bf16", "q8", "q4"];

/// Map a tier name to the pack bit-width, or `None` for the dense `bf16` tier.
pub fn tier_bits(tier: &str) -> Result<Option<i32>> {
    match tier {
        "bf16" => Ok(None),
        "q8" => Ok(Some(8)),
        "q4" => Ok(Some(4)),
        other => Err(Error::Msg(format!(
            "mage_flow convert: unknown tier {other:?}; expected one of {TIERS:?}"
        ))),
    }
}

// ============================================================================================
// Per-component pack predicates (operate on the **base** = the on-disk key minus its `.weight`).
// The shared `quantize_map` shape guard (2-D, `in % group_size == 0`, `in >= group_size`) is the
// backstop, so these are faithfulness + documentation — EXCEPT where noted, where the predicate is
// the ONLY thing standing between a correct tier and a silently broken one.
// ============================================================================================

/// The bit-width a packable `base` is written at when tier `requested` was asked for.
///
/// `requested` for almost everything; **8** for the two projections that do not survive 4 bits (see
/// the "Precision floors" section of this module's docs). This is the only place either floor is
/// decided — the converters below and the load-time `quantize` methods all route through it, so a
/// pre-quantized artifact and load-time quantization cannot drift apart. `a_packed_tier_renders_\
/// identically_to_load_time_quantization` asserts that equality end to end (max_abs 0).
pub fn quant_floor_bits(base: &str, requested: i32) -> i32 {
    floor_bits(base, requested)
}

/// DiT dense-passthrough suffixes — the qk-RMSNorm scales, all 1-D.
const DIT_DENSE_NORM_SUFFIXES: &[&str] = &[".norm_q", ".norm_k", ".norm_added_q", ".norm_added_k"];

/// `true` iff a DiT base names one of the 174 quantizable Linears.
///
/// Packed: `img_in`, `txt_in`, `time_text_embed.timestep_embedder.linear_{1,2}`, `norm_out.linear`,
/// `proj_out`, and per block the 8 joint-attention projections
/// (`to_{q,k,v}`, `to_out.0`, `add_{q,k,v}_proj`, `to_add_out`), the 4 gelu-approximate FFN
/// projections (`{img,txt}_mlp.net.0.proj`, `{img,txt}_mlp.net.2`) and the 2 adaLN modulations
/// (`{img,txt}_mod.1`) — 2 + 2 + 12·14 + 2 = 174, the count `MageTransformer::quantize` asserts.
///
/// Dense: `txt_norm` and the four per-head qk-RMSNorms (all 1-D, also shape-guarded).
pub fn is_dit_target(base: &str) -> bool {
    base != "txt_norm" && !DIT_DENSE_NORM_SUFFIXES.iter().any(|s| base.ends_with(s))
}

/// `true` iff a text-encoder base names a quantizable tensor.
///
/// Packed — LM (`model.language_model.*`): the token embedding plus every GQA / SwiGLU `*_proj`
/// (`q,k,v,o,gate,up,down`) = 1 + 36·7 = 253, the count `Qwen3VlTextEncoder::quantize` asserts.
/// Packed — vision tower (`model.visual.*`, loaded by `mlx_gen_boogu::VisionTower`, whose loader is
/// packed-aware): the fused `attn.qkv`, the `attn.proj` out-projection, and the `linear_fc1` /
/// `linear_fc2` pairs of the block MLPs, the merger, and the three deepstack mergers.
///
/// **Two exclusions are load-bearing, not cosmetic:**
/// - `model.visual.pos_embed.weight` is `[2304, 1024]` — 2-D and group-aligned, so the
///   `quantize_map` shape guard would NOT save it — but `VisionTower::from_weights` reads it with a
///   raw `w.require(...)`, not through `quant::lin`. Packing it produces a `u32` code tensor where
///   a bf16 embedding table is expected: a wrong-dtype position embedding, not a load error.
/// - `model.visual.patch_embed.proj.weight` is a 5-D conv the loader reshapes into a dense
///   `AdaptableLinear`. It is shape-guarded too, but is named here so the intent is explicit.
///   (It ends `.proj`, not `_proj`, so the LM rule would not have caught it either way.)
pub fn is_te_target(base: &str) -> bool {
    if base.ends_with(".pos_embed") || base.ends_with(".patch_embed.proj") {
        return false;
    }
    base.ends_with(".embed_tokens")
        || base.ends_with("_proj")
        || base.ends_with(".attn.qkv")
        || base.ends_with(".attn.proj")
        || base.ends_with(".linear_fc1")
        || base.ends_with(".linear_fc2")
}

// ============================================================================================
// Per-component dir converters.
// ============================================================================================

/// Pre-quantize the DiT `transformer` dir → a packed `diffusion_pytorch_model.safetensors` +
/// annotated `config.json` in `dst`, or a byte-exact dense copy when `bits` is `None`.
///
/// The output keeps Mage's own [`TRANSFORMER_WEIGHTS_FILE`] name (not the siblings'
/// `model.safetensors`) so [`crate::transformer::MageTransformer::load`] and the checkpoint-identity
/// fingerprint in [`crate::model`] read a tier exactly as they read the flat root — no path
/// branching on the load side.
pub fn quantize_mage_transformer(src: &Path, dst: &Path, bits: Option<i32>) -> Result<()> {
    std::fs::create_dir_all(dst)?;
    let Some(bits) = bits else {
        // bf16 tier: byte-exact passthrough. Copying rather than round-tripping through
        // load/save keeps the identity fingerprint's source bytes intact.
        copy_asset(src, dst, TRANSFORMER_WEIGHTS_FILE)?;
        copy_asset(src, dst, TRANSFORMER_CONFIG_FILE)?;
        return Ok(());
    };
    // Two passes, because `norm_out.linear` has an 8-bit floor (sc-15071 — a uniformly-Q4 DiT
    // renders a tiled texture instead of the prompt; see `crate::quant::FINAL_MOD_MIN_BITS`).
    // Pass 2 re-reads pass 1's output, but its predicate matches only the still-dense floor base,
    // so already-packed codes are passed straight through rather than re-quantized.
    let map = quantize_map(load_dir_map(src)?, bits, GROUP_SIZE, |base| {
        is_dit_target(base) && base != FINAL_MOD_BASE
    })?;
    let map = quantize_map(
        map,
        quant_floor_bits(FINAL_MOD_BASE, bits),
        GROUP_SIZE,
        |base| base == FINAL_MOD_BASE,
    )?;
    save_map(&dst.join(TRANSFORMER_WEIGHTS_FILE), &map)?;
    // The marker records the TIER, not a per-tensor width: `packed_quant_bits` uses it only to
    // decide whether load-time quantization still has to run, while the loaders derive each
    // tensor's actual bit-width from its packed shapes (`mlx_gen::quant::packed_bits`). A Q4 tier
    // whose head modulation is packed at 8 is therefore still, correctly, `bits: 4`.
    write_quantized_config(src, dst, bits, GROUP_SIZE)
}

/// The non-weight files the `text_encoder/` dir carries (tokenizer, processor, chat template).
/// Copied verbatim into every tier — they are tier-independent and tiny.
const TEXT_ENCODER_ASSETS: &[&str] = &[
    "chat_template.json",
    "generation_config.json",
    "merges.txt",
    "preprocessor_config.json",
    "tokenizer.json",
    "tokenizer_config.json",
    "video_preprocessor_config.json",
    "vocab.json",
];

/// Pre-quantize the `text_encoder` dir → a packed, **consolidated** `model.safetensors` + annotated
/// `config.json` + the tokenizer/processor tail in `dst`; a dense consolidation when `bits` is
/// `None`.
///
/// The source ships two shards plus a `model.safetensors.index.json`. The index is deliberately NOT
/// carried over: `Weights::from_dir` globs `*.safetensors` and rejects duplicate keys, so a stale
/// two-shard index alongside a consolidated file would be a confusing no-op at best. The loaders
/// read the dir, never the index.
pub fn quantize_mage_text_encoder(src: &Path, dst: &Path, bits: Option<i32>) -> Result<()> {
    std::fs::create_dir_all(dst)?;
    let map = load_dir_map(src)?;
    let map = match bits {
        // Two passes for the same reason the DiT needs them: the 36 LM decoder layers have an
        // 8-bit floor (sc-15071 — see `crate::quant::LM_LAYER_MIN_BITS`), while the token embedding
        // and the vision tower take the tier's own width. Pass 2's predicate matches only keys pass
        // 1 left dense, so packed codes are never re-quantized.
        Some(bits) => {
            let map = quantize_map(map, bits, GROUP_SIZE, |base| {
                is_te_target(base) && !base.starts_with(LM_LAYER_PREFIX)
            })?;
            quantize_map(
                map,
                quant_floor_bits(LM_LAYER_PREFIX, bits),
                GROUP_SIZE,
                |base| is_te_target(base) && base.starts_with(LM_LAYER_PREFIX),
            )?
        }
        None => map,
    };
    save_map(&dst.join("model.safetensors"), &map)?;
    match bits {
        Some(bits) => write_quantized_config(src, dst, bits, GROUP_SIZE)?,
        None => {
            copy_asset(src, dst, "config.json")?;
        }
    }
    for name in TEXT_ENCODER_ASSETS {
        copy_asset(src, dst, name)?;
    }
    Ok(())
}

// ============================================================================================
// Tier assembly.
// ============================================================================================

/// The per-tier `model_index.json` a variant tier ships.
///
/// It declares ONLY the components physically present in the tier dir. That is what makes the
/// SceneWorks worker's `tier_components_present` completeness probe (which reads a tier's own
/// `model_index.json` and requires every declared component dir to exist) pass for a DiT-only tier
/// instead of demoting it as torn. The shared components are recorded as a JSON **object** — not an
/// array — because that probe treats any two-string array value as a component name.
fn variant_tier_model_index(tier: &str, components_repo: &str) -> serde_json::Value {
    serde_json::json!({
        "_class_name": "MageFlowPipeline",
        "_mage_flow_version": "0.1.0",
        "transformer": ["mage_flow", "MageFlow"],
        "scheduler": ["diffusers", "FlowMatchEulerDiscreteScheduler"],
        "_sceneworks_tier": tier,
        "_sceneworks_shared_components": {
            "text_encoder": components_repo,
            "vae": components_repo,
        },
    })
}

/// Assemble one **variant** tier (`<dst_root>/`): the DiT plus the scheduler config and a tier
/// `model_index.json`. The text encoder and VAE are NOT written here — they live once in the shared
/// components mirror ([`prequantize_shared_components`]).
pub fn prequantize_variant_tier(
    src_root: &Path,
    dst_root: &Path,
    tier: &str,
    components_repo: &str,
) -> Result<()> {
    let bits = tier_bits(tier)?;
    std::fs::create_dir_all(dst_root)?;
    quantize_mage_transformer(
        &src_root.join("transformer"),
        &dst_root.join("transformer"),
        bits,
    )?;
    let scheduler = src_root.join("scheduler");
    if scheduler.exists() {
        copy_dir(&scheduler, &dst_root.join("scheduler"))?;
    }
    let index = serde_json::to_string_pretty(&variant_tier_model_index(tier, components_repo))
        .map_err(|e| {
            Error::Msg(format!(
                "mage_flow convert: serialize model_index.json: {e}"
            ))
        })?;
    std::fs::write(dst_root.join("model_index.json"), index)?;
    for name in ["LICENSE", "LICENSE.md", "LICENSE.txt", "README.md"] {
        copy_asset(src_root, dst_root, name)?;
    }
    Ok(())
}

/// Assemble one **shared components** tier (`<dst_root>/`): the packed text encoder plus the dense
/// VAE. Written once for the whole family; all six variants resolve it as a co-requisite.
pub fn prequantize_shared_components(src_root: &Path, dst_root: &Path, tier: &str) -> Result<()> {
    let bits = tier_bits(tier)?;
    std::fs::create_dir_all(dst_root)?;
    quantize_mage_text_encoder(
        &src_root.join("text_encoder"),
        &dst_root.join("text_encoder"),
        bits,
    )?;
    // The VAE is a byte-exact dense copy in every tier — see the module note on `fold_decode`.
    copy_dir(&src_root.join("vae"), &dst_root.join("vae"))?;
    for name in ["LICENSE", "LICENSE.md", "LICENSE.txt", "README.md"] {
        copy_asset(src_root, dst_root, name)?;
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use mlx_rs::ops::{eq, quantize};
    use mlx_rs::{Array, Dtype};
    use std::collections::HashMap;

    #[test]
    fn dit_predicate_packs_every_linear_not_the_norms() {
        for base in [
            "img_in",
            "txt_in",
            "proj_out",
            "norm_out.linear",
            "time_text_embed.timestep_embedder.linear_1",
            "time_text_embed.timestep_embedder.linear_2",
            "transformer_blocks.0.attn.to_q",
            "transformer_blocks.11.attn.to_out.0",
            "transformer_blocks.3.attn.add_k_proj",
            "transformer_blocks.3.attn.to_add_out",
            "transformer_blocks.7.img_mlp.net.0.proj",
            "transformer_blocks.7.img_mlp.net.2",
            "transformer_blocks.7.txt_mlp.net.0.proj",
            "transformer_blocks.7.txt_mlp.net.2",
            "transformer_blocks.7.img_mod.1",
            "transformer_blocks.7.txt_mod.1",
        ] {
            assert!(is_dit_target(base), "{base} should be packed");
        }
        for base in [
            "txt_norm",
            "transformer_blocks.0.attn.norm_q",
            "transformer_blocks.0.attn.norm_k",
            "transformer_blocks.0.attn.norm_added_q",
            "transformer_blocks.0.attn.norm_added_k",
        ] {
            assert!(!is_dit_target(base), "{base} should stay dense");
        }
    }

    /// The predicate must select EXACTLY the 174 Linears `MageTransformer::quantize` counts, over
    /// the real published key set. A predicate that packed one norm too many (or one Linear too
    /// few) would load fine and only fail the count assertion at generation time.
    #[test]
    fn dit_predicate_selects_exactly_the_174_asserted_linears() {
        let mut bases: Vec<String> = vec![
            "img_in".into(),
            "txt_in".into(),
            "proj_out".into(),
            "norm_out.linear".into(),
            "time_text_embed.timestep_embedder.linear_1".into(),
            "time_text_embed.timestep_embedder.linear_2".into(),
            "txt_norm".into(),
        ];
        for i in 0..12 {
            for suffix in [
                "attn.to_q",
                "attn.to_k",
                "attn.to_v",
                "attn.to_out.0",
                "attn.add_q_proj",
                "attn.add_k_proj",
                "attn.add_v_proj",
                "attn.to_add_out",
                "img_mlp.net.0.proj",
                "img_mlp.net.2",
                "txt_mlp.net.0.proj",
                "txt_mlp.net.2",
                "img_mod.1",
                "txt_mod.1",
                // the four dense per-head norms
                "attn.norm_q",
                "attn.norm_k",
                "attn.norm_added_q",
                "attn.norm_added_k",
            ] {
                bases.push(format!("transformer_blocks.{i}.{suffix}"));
            }
        }
        let packed = bases.iter().filter(|b| is_dit_target(b)).count();
        assert_eq!(
            packed, 174,
            "the DiT predicate must select exactly the 174 Linears MageTransformer::quantize asserts"
        );
    }

    /// sc-15071: the head's adaLN modulation is still one of the 174 packed Linears, but never at
    /// fewer than 8 bits. A regression that dropped the floor would reproduce the shipped defect
    /// (the Q4 tier rendering a tiled texture), and the count assertions would NOT notice, because
    /// the tensor stays packed either way.
    #[test]
    fn the_head_modulation_is_packed_but_never_below_eight_bits() {
        assert!(is_dit_target(FINAL_MOD_BASE), "must still be packed");
        assert_eq!(quant_floor_bits(FINAL_MOD_BASE, 4), 8, "Q4 must be floored");
        assert_eq!(quant_floor_bits(FINAL_MOD_BASE, 8), 8);
        // Every other projection takes the requested width unchanged — the floor is one tensor,
        // not a blanket upgrade that would silently turn the Q4 tier into Q8.
        for base in [
            "img_in",
            "txt_in",
            "proj_out",
            "time_text_embed.timestep_embedder.linear_1",
            "transformer_blocks.0.img_mod.1",
            "transformer_blocks.11.txt_mod.1",
            "transformer_blocks.0.attn.to_q",
        ] {
            assert_eq!(
                quant_floor_bits(base, 4),
                4,
                "{base} must stay at the tier width"
            );
            assert_eq!(quant_floor_bits(base, 8), 8, "{base}");
        }
    }

    /// The converter's two-pass write must produce exactly the widths the load path produces:
    /// Q8 shapes for the head modulation, Q4 shapes for everything else, in one Q4 artifact.
    #[test]
    fn the_q4_converter_writes_a_mixed_width_map_matching_the_load_path() {
        let mut map = HashMap::new();
        for base in [FINAL_MOD_BASE, "proj_out"] {
            map.insert(
                format!("{base}.weight"),
                Array::zeros::<f32>(&[256, 128]).unwrap(),
            );
        }
        let out = quantize_map(map, 4, GROUP_SIZE, |base| {
            is_dit_target(base) && base != FINAL_MOD_BASE
        })
        .unwrap();
        let out = quantize_map(
            out,
            quant_floor_bits(FINAL_MOD_BASE, 4),
            GROUP_SIZE,
            |base| base == FINAL_MOD_BASE,
        )
        .unwrap();
        // scales are [out, in/gs] either way; the u32 code tensor is [out, in·bits/32], so the
        // packed width is visible in the shape — which is exactly how the loader recovers it.
        let bits_of = |base: &str| {
            mlx_gen::quant::packed_bits(
                out.get(&format!("{base}.weight")).unwrap(),
                out.get(&format!("{base}.scales")).unwrap(),
                GROUP_SIZE,
            )
            .unwrap()
        };
        assert_eq!(
            bits_of(FINAL_MOD_BASE),
            8,
            "head modulation must land at Q8"
        );
        assert_eq!(bits_of("proj_out"), 4, "the rest of the tier stays Q4");
    }

    /// sc-15071: the LM decoder layers keep their 8-bit floor, while the token embedding and the
    /// vision tower stay at the tier's own width. Getting this wrong in either direction is silent
    /// — every projection stays packed and every count assertion still passes.
    #[test]
    fn the_lm_decoder_layers_are_floored_but_the_embedding_and_vision_tower_are_not() {
        for base in [
            "model.language_model.layers.0.self_attn.q_proj",
            "model.language_model.layers.35.self_attn.o_proj",
            "model.language_model.layers.7.mlp.gate_proj",
            "model.language_model.layers.7.mlp.down_proj",
        ] {
            assert!(is_te_target(base), "{base} must still be packed");
            assert_eq!(quant_floor_bits(base, 4), 8, "{base} must be floored at Q8");
            assert_eq!(quant_floor_bits(base, 8), 8, "{base}");
        }
        for base in [
            "model.language_model.embed_tokens",
            "model.visual.blocks.0.attn.qkv",
            "model.visual.blocks.5.mlp.linear_fc1",
            "model.visual.merger.linear_fc1",
        ] {
            assert_eq!(
                quant_floor_bits(base, 4),
                4,
                "{base} must keep the tier width — flooring it would inflate the Q4 tier for no \
                 measured quality gain"
            );
        }
    }

    #[test]
    fn te_predicate_packs_lm_projections_and_the_vision_tower() {
        for base in [
            "model.language_model.embed_tokens",
            "model.language_model.layers.0.self_attn.q_proj",
            "model.language_model.layers.35.self_attn.o_proj",
            "model.language_model.layers.7.mlp.gate_proj",
            "model.language_model.layers.7.mlp.down_proj",
            "model.visual.blocks.0.attn.qkv",
            "model.visual.blocks.23.attn.proj",
            "model.visual.blocks.5.mlp.linear_fc1",
            "model.visual.blocks.5.mlp.linear_fc2",
            "model.visual.merger.linear_fc1",
            "model.visual.deepstack_merger_list.2.linear_fc2",
        ] {
            assert!(is_te_target(base), "{base} should be packed");
        }
        for base in [
            "model.language_model.layers.0.self_attn.q_norm",
            "model.language_model.layers.0.self_attn.k_norm",
            "model.language_model.layers.0.input_layernorm",
            "model.language_model.layers.0.post_attention_layernorm",
            "model.language_model.norm",
            "model.visual.blocks.0.norm1",
            "model.visual.merger.norm",
        ] {
            assert!(!is_te_target(base), "{base} should stay dense");
        }
    }

    /// The two exclusions the shape guard does NOT cover. `pos_embed` is 2-D and group-aligned, so
    /// only this predicate keeps it dense — and the vision tower reads it as a raw bf16 table, so a
    /// pack would be silently wrong rather than a load error. This is the discriminating case: a
    /// naive `ends_with(".proj") || ends_with("_proj")` predicate passes every other assertion in
    /// this module and fails here.
    #[test]
    fn te_predicate_excludes_the_raw_loaded_vision_embeddings() {
        assert!(
            !is_te_target("model.visual.pos_embed"),
            "pos_embed is [2304,1024] — 2-D and group-aligned, so the shape guard will NOT keep it \
             dense; packing it hands VisionTower::from_weights u32 codes where it requires a bf16 table"
        );
        assert!(
            !is_te_target("model.visual.patch_embed.proj"),
            "patch_embed.proj is a 5-D conv the loader reshapes into a dense AdaptableLinear"
        );
    }

    #[test]
    fn tier_bits_maps_the_published_tiers_and_rejects_others() {
        assert_eq!(tier_bits("bf16").unwrap(), None);
        assert_eq!(tier_bits("q8").unwrap(), Some(8));
        assert_eq!(tier_bits("q4").unwrap(), Some(4));
        assert!(tier_bits("q6").is_err());
        assert!(tier_bits("").is_err());
    }

    /// A DiT-only tier's `model_index.json` must declare exactly the components that are physically
    /// present, or the worker's completeness probe demotes the tier as torn. In particular the
    /// shared-components record must NOT be a two-element string array — that probe would read it
    /// as a component named `_sceneworks_shared_components` and require a directory of that name.
    #[test]
    fn variant_tier_index_declares_only_the_components_present_in_the_tier() {
        let index = variant_tier_model_index("q4", "SceneWorks/Mage-Flow-Components-mlx");
        let is_component = |v: &serde_json::Value| {
            v.as_array()
                .is_some_and(|p| p.len() == 2 && p.iter().all(serde_json::Value::is_string))
        };
        let mut components: Vec<&str> = index
            .as_object()
            .unwrap()
            .iter()
            .filter(|(_, v)| is_component(v))
            .map(|(k, _)| k.as_str())
            .collect();
        components.sort_unstable();
        assert_eq!(
            components,
            ["scheduler", "transformer"],
            "a variant tier ships the DiT + scheduler only; text_encoder/vae are shared co-requisites"
        );
        assert_eq!(index["_sceneworks_tier"], "q4");
    }

    fn byte_equal(a: &Array, b: &Array) -> bool {
        a.shape() == b.shape()
            && a.dtype() == b.dtype()
            && eq(a, b).unwrap().all(None).unwrap().item::<bool>()
    }

    /// The packed triple a DiT Linear becomes is byte-identical to the op the load-time
    /// `AdaptableLinear::quantize` runs (bf16 cast, group 64) — the pre-quantize-on-disk ==
    /// quantize-at-load guarantee that lets a tier artifact replace load-time quant with no
    /// numerical change.
    #[test]
    fn quantize_map_packs_targets_byte_identical_to_load_time_quantize() {
        let w = Array::from_slice(
            &(0..64 * 128).map(|i| (i as f32).sin()).collect::<Vec<_>>(),
            &[64, 128],
        );
        let mut map: HashMap<String, Array> = HashMap::new();
        map.insert("transformer_blocks.0.attn.to_q.weight".into(), w.clone());
        map.insert(
            "transformer_blocks.0.attn.norm_q.weight".into(),
            Array::ones::<f32>(&[128]).unwrap(),
        );

        let out = quantize_map(map, 4, GROUP_SIZE, is_dit_target).unwrap();

        let wq = out
            .get("transformer_blocks.0.attn.to_q.weight")
            .expect("packed");
        assert_eq!(wq.dtype(), Dtype::Uint32, "Q4 codes are u32-packed");
        let scales = out.get("transformer_blocks.0.attn.to_q.scales").unwrap();
        let biases = out.get("transformer_blocks.0.attn.to_q.biases").unwrap();
        let (ewq, esc, ebi) =
            quantize(w.as_dtype(Dtype::Bfloat16).unwrap(), GROUP_SIZE, 4).unwrap();
        assert!(byte_equal(wq, &ewq), "packed weight != load-time quantize");
        assert!(byte_equal(scales, &esc), "scales != load-time quantize");
        assert!(byte_equal(biases, &ebi), "biases != load-time quantize");

        let n = out
            .get("transformer_blocks.0.attn.norm_q.weight")
            .expect("dense norm");
        assert_eq!(n.dtype(), Dtype::Float32, "norm unchanged");
        assert!(!out.contains_key("transformer_blocks.0.attn.norm_q.scales"));
    }
}
