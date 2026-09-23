//! Offline pre-quantization: read the dense `Qwen/Qwen-Image-2.1` snapshot and write a packed
//! per-tier snapshot that [`crate::load`] consumes with no dense transient (sc-24112).
//!
//! This is the producer side of [`crate::quant`]; read that module's table first — it owns the
//! per-component tiering decision and the reasons it is **not** the 2512 `mlx_gen_qwen_image` table.
//!
//! # Output layout
//!
//! [`prequantize_turnkey`] assembles a complete standalone snapshot, so a tier is loadable by
//! pointing `WeightsSource::Dir` at it and nothing else:
//!
//! ```text
//! <dst_root>/
//!   model_index.json            copied
//!   LICENSE / README.md         copied (Qwen Research License — F-045)
//!   CHANGES.md                  the licence's §3(b) change record: what was re-packed, at what
//!                               width, from which upstream revision, by which converter
//!   SHA256SUMS                  `sha256sum` manifest over every other file (the binding digest)
//!   transformer/
//!     config.json               source config + {"quantization": {"bits", "group_size": 64}}
//!     model.safetensors         packed: every Linear as {base}.weight|.scales|.biases
//!   text_encoder/
//!     config.json               source config + {"quantization": {"bits", "group_size": 64}}
//!     model.safetensors         packed language-tower decoder Linears; everything else verbatim
//!   vae/ processor/ scheduler/  copied verbatim (dense)
//! ```
//!
//! Both packed components carry the **same** `bits`: a tier is a whole-pipeline contract, so
//! `text_encoder/`'s marker is the tier the caller selected. Writing a different width in either
//! marker would mislabel the artefact for [`mlx_gen::quant::packed_quant_bits`], and therefore for
//! every fit estimate that reads it.
//!
//! `CHANGES.md` and `SHA256SUMS` are written by [`prequantize_turnkey`] **itself**
//! ([`write_change_record`], [`write_sha256sums`]), not by the example driver around it, so no
//! tier can be assembled without the change record the Qwen Research License requires of a
//! modified redistribution or without the manifest a published artefact is bound to (sc-24114).
//!
//! # Reproducibility
//!
//! [`mlx_gen::quant::save_map`] sorts keys before serialization,
//! [`mlx_gen::quant::write_quantized_config`] writes `serde_json::to_string_pretty` of a
//! deterministically-merged value, and the change record carries no timestamp or host name, so
//! converting the same source twice produces **byte-identical** files — `SHA256SUMS` included.
//! `tests/tiers.rs` pins that by SHA-256 over the converted tiny snapshot, which is what lets a
//! published tier be bound to a hash.
//!
//! # Equivalence to load-time quantization
//!
//! [`mlx_gen::quant::quantize_map`] casts each target to bf16 and runs the same `mlx_rs::ops::quantize`
//! that [`crate::transformer::QwenImage21Transformer::quantize`] runs at load, so a packed tier is
//! **byte-identical** to quantizing the dense snapshot in memory — on every width that is a multiple
//! of [`crate::quant::GROUP_SIZE`]. The released DiT and language tower have no other width
//! ([`transformer_widths_are_group_aligned`]); the miniature parity snapshot does, and there the
//! converter deliberately leaves them dense rather than packing at a second group size the loader
//! could not read back. See the [`crate::quant`] module docs.

use std::collections::BTreeMap;
use std::path::Path;

use mlx_gen::quant::{
    copy_dir, copy_turnkey_assets, load_dir_map, quantize_map, save_map, write_quantized_config,
};
use mlx_gen::{Error, Result};

use crate::config::{TextEncoderConfig, TransformerConfig};
use crate::quant::{Tier, DENSE_COMPONENTS, GROUP_SIZE};

// ============================================================================================
// Pack predicates. Both operate on the **raw on-disk base** (the key minus its `.weight` suffix);
// the shared `quantize_map` shape guard (2-D, `in % group_size == 0`, `in >= group_size`) is the
// backstop, so these are faithfulness + documentation rather than the only safety net.
// ============================================================================================

/// Transformer suffixes that are RMSNorm scales, not `Linear` weights: the per-head query/key norms
/// of every block. Together with the top-level `txt_in.text_norm` these are the only `…​.weight`
/// entries under `transformer/` that are not a projection.
const DIT_DENSE_NORM_SUFFIXES: &[&str] = &[".norm_q", ".norm_k"];

/// The top-level text RMSNorm the DiT applies before `txt_in.in_layer`.
const DIT_TEXT_NORM: &str = "txt_in.text_norm";

/// `true` iff a `transformer/` base names a quantizable `Linear`: the image/text embedders
/// (`img_in`, `txt_in.in_layer`, `txt_in.out_layer`), the timestep MLP, the shared adaLN
/// `modulation.1`, every block's joint-attention `to_q`/`to_k`/`to_v`/`to_out.0` and gated FFN
/// (`img_mlp.gate_layer`/`proj`/`out`), and the final `norm_out.linear` / `proj_out`.
pub fn is_transformer_target(base: &str) -> bool {
    base != DIT_TEXT_NORM && !DIT_DENSE_NORM_SUFFIXES.iter().any(|s| base.ends_with(s))
}

/// The seven decoder projections of one Qwen3 block, relative to `…​.layers.{i}`.
const LM_PROJECTIONS: &[&str] = &[
    "self_attn.q_proj",
    "self_attn.k_proj",
    "self_attn.v_proj",
    "self_attn.o_proj",
    "mlp.gate_proj",
    "mlp.up_proj",
    "mlp.down_proj",
];

/// The decoder-layer prefix inside a `Qwen3VLForConditionalGeneration` checkpoint.
const LM_LAYER_PREFIX: &str = "model.language_model.layers.";

/// `true` iff a `text_encoder/` base names a decoder `Linear` the loaded language tower builds
/// through the packed-detect seam.
///
/// This is deliberately an allow-list rather than a norm-exclusion list, because the checkpoint
/// carries three things the tower never loads and must not silently reinterpret: the token
/// embedding (`model.language_model.embed_tokens` — [`crate::text_encoder`] packs decoder layers
/// only, so keeping it dense is what preserves pre-quantize ≡ quantize-at-load), the untied
/// `lm_head`, and the whole `model.visual.*` vision tower. All three ride through **dense**: the
/// vision tower has no ported quantize path and is the reference/edit route's input, not ours to
/// re-encode here.
pub fn is_text_encoder_target(base: &str) -> bool {
    let Some(rest) = base.strip_prefix(LM_LAYER_PREFIX) else {
        return false;
    };
    let Some((index, projection)) = rest.split_once('.') else {
        return false;
    };
    index.chars().all(|c| c.is_ascii_digit())
        && !index.is_empty()
        && LM_PROJECTIONS.contains(&projection)
}

// ============================================================================================
// Per-component converters.
// ============================================================================================

/// Pre-quantize the `transformer/` dir (sharded `*.safetensors` + index + `config.json`) into a
/// packed `model.safetensors` + annotated `config.json` in `dst`. `bits` is 4 or 8.
pub fn quantize_transformer(src: &Path, dst: &Path, bits: i32) -> Result<()> {
    pack_component(src, dst, bits, is_transformer_target)
}

/// Pre-quantize the `text_encoder/` dir into a packed `model.safetensors` + annotated
/// `config.json` in `dst`. `bits` is the tier's own width — a tier is a whole-pipeline contract, so
/// this is always [`crate::quant::Tier::transformer_bits`] — and the parameter stays explicit so the
/// converter has no hidden policy of its own.
pub fn quantize_text_encoder(src: &Path, dst: &Path, bits: i32) -> Result<()> {
    pack_component(src, dst, bits, is_text_encoder_target)
}

fn pack_component(
    src: &Path,
    dst: &Path,
    bits: i32,
    is_target: impl Fn(&str) -> bool,
) -> Result<()> {
    if !matches!(bits, 4 | 8) {
        return Err(Error::Msg(format!(
            "qwen_image_2_1 convert: unsupported bit-width Q{bits}; expected 4 or 8"
        )));
    }
    std::fs::create_dir_all(dst)?;
    let map = quantize_map(load_dir_map(src)?, bits, GROUP_SIZE, is_target)?;
    save_map(&dst.join("model.safetensors"), &map)?;
    write_quantized_config(src, dst, bits, GROUP_SIZE)
}

// ============================================================================================
// Turnkey assembly.
// ============================================================================================

/// Assemble the complete `tier` snapshot for `src_root` (the dense released snapshot) under
/// `dst_root`.
///
/// [`Tier::Bf16`] is refused: the dense tier **is** the source snapshot, and copying 31 GB to call
/// it a conversion would produce a second artefact to keep in sync for no benefit. Mirror the source
/// instead.
pub fn prequantize_turnkey(src_root: &Path, dst_root: &Path, tier: Tier) -> Result<()> {
    let (Some(dit_bits), Some(te_bits)) = (tier.transformer_bits(), tier.text_encoder_bits())
    else {
        return Err(Error::Msg(format!(
            "qwen_image_2_1 convert: the {} tier is the dense released snapshot itself; mirror it \
             rather than converting it",
            tier.dir_name()
        )));
    };
    std::fs::create_dir_all(dst_root)?;
    quantize_transformer(
        &src_root.join("transformer"),
        &dst_root.join("transformer"),
        dit_bits,
    )?;
    quantize_text_encoder(
        &src_root.join("text_encoder"),
        &dst_root.join("text_encoder"),
        te_bits,
    )?;
    for rel in DENSE_COMPONENTS {
        let src = src_root.join(rel);
        if src.exists() {
            copy_dir(&src, &dst_root.join(rel))?;
        }
    }
    copy_turnkey_assets(src_root, dst_root)?;
    // The two files a *distributed* tier must carry beyond the weights (sc-24114): the licence's
    // change record, and the manifest that binds the artefact to its bytes. Written by the
    // converter itself — not by a driver around it — so no tier can leave here without them.
    write_change_record(dst_root, tier)?;
    write_sha256sums(dst_root)
}

// ============================================================================================
// The distributed tier's change record and manifest.
// ============================================================================================

/// File name of the change record every converted tier carries.
pub const CHANGES_FILE: &str = "CHANGES.md";
/// File name of the SHA-256 manifest every converted tier carries.
pub const SHA256SUMS_FILE: &str = "SHA256SUMS";

/// The Qwen Research License §3(b) change record for one converted tier: which components were
/// re-packed and at what width, which ride through verbatim, the upstream snapshot and revision
/// the tier was derived from, and the converter that produced it.
///
/// Deterministic by construction — no timestamp, no host name — so the record is part of the
/// byte-reproducible artefact rather than the one file that makes two conversions differ.
pub fn change_record(tier: Tier) -> Result<String> {
    let (Some(dit_bits), Some(te_bits)) = (tier.transformer_bits(), tier.text_encoder_bits())
    else {
        return Err(Error::Msg(format!(
            "qwen_image_2_1 convert: the {} tier is the dense released snapshot itself and \
             carries no change record",
            tier.dir_name()
        )));
    };
    let dense: Vec<String> = DENSE_COMPONENTS
        .iter()
        .map(|component| format!("`{component}/`"))
        .collect();
    Ok(format!(
        "# Qwen-Image 2.1 — `{tier}` tier: change record\n\
         \n\
         This directory is a **modified copy** of the `{repo}` snapshot at revision\n\
         `{revision}`, re-packed offline into the `{tier}` installable tier by\n\
         `{converter}` {version} (`mlx_gen_qwen_image_2_1::convert::prequantize_turnkey`).\n\
         This file is the notice of those changes the Qwen Research License Agreement §3(b)\n\
         requires; the licence itself (`LICENSE`) and the upstream `README.md` are carried\n\
         verbatim, and the §3(c) attribution is in `NOTICE` of the producing crate.\n\
         \n\
         ## Files that were changed\n\
         \n\
         | component | change |\n\
         |---|---|\n\
         | `transformer/model.safetensors` | every 2-D `Linear` weight affine-quantized to \
         **Q{dit_bits}, group size {group}** (`weight` = packed u32 codes, plus `scales` and \
         `biases` per group); RMSNorm scales dense |\n\
         | `transformer/config.json` | upstream config plus `\"quantization\": {{\"bits\": \
         {dit_bits}, \"group_size\": {group}}}` |\n\
         | `text_encoder/model.safetensors` | the `model.language_model.layers.*` decoder \
         projections (q/k/v/o, gate/up/down) affine-quantized to **Q{te_bits}, group size \
         {group}**; token embedding, norms, `lm_head` and the whole `model.visual.*` vision tower \
         dense and unchanged; the sharded upstream files are consolidated into one |\n\
         | `text_encoder/config.json` | upstream config plus `\"quantization\": {{\"bits\": \
         {te_bits}, \"group_size\": {group}}}` |\n\
         \n\
         A tier is a whole-pipeline contract: both packed components carry the same width.\n\
         \n\
         ## Files that were not changed\n\
         \n\
         {dense}, `model_index.json`, `LICENSE` and `README.md` are byte-for-byte copies of \
         the upstream snapshot.\n\
         \n\
         ## Integrity\n\
         \n\
         `{sums}` beside this file lists the SHA-256 of every file in this directory. The \
         conversion is byte-reproducible, so a re-run from the same upstream revision \
         reproduces every digest.\n",
        tier = tier.dir_name(),
        repo = crate::UPSTREAM_HF_REPO,
        revision = crate::UPSTREAM_HF_REVISION,
        converter = env!("CARGO_PKG_NAME"),
        version = env!("CARGO_PKG_VERSION"),
        group = GROUP_SIZE,
        dense = dense.join(", "),
        sums = SHA256SUMS_FILE,
    ))
}

/// Write [`change_record`] as `<dst_root>/CHANGES.md`.
pub fn write_change_record(dst_root: &Path, tier: Tier) -> Result<()> {
    std::fs::write(dst_root.join(CHANGES_FILE), change_record(tier)?)?;
    Ok(())
}

/// Every regular file under `root` (hidden entries skipped, like the copy that produced them),
/// keyed by its `/`-separated path relative to `root`, valued by its lowercase hex SHA-256 —
/// sorted, so the manifest built from it is deterministic. [`SHA256SUMS_FILE`] itself is
/// excluded, since it cannot list its own digest.
pub fn digest_tree(root: &Path) -> Result<BTreeMap<String, String>> {
    use sha2::{Digest, Sha256};
    fn walk(root: &Path, dir: &Path, out: &mut BTreeMap<String, String>) -> Result<()> {
        for entry in std::fs::read_dir(dir)? {
            let path = entry?.path();
            if mlx_gen::gen_core::weightsmeta::is_hidden_file(&path) {
                continue;
            }
            if path.is_dir() {
                walk(root, &path, out)?;
                continue;
            }
            let rel = path
                .strip_prefix(root)
                .map_err(|e| Error::Msg(format!("qwen_image_2_1 convert: {e}")))?
                .to_string_lossy()
                .replace('\\', "/");
            if rel == SHA256SUMS_FILE {
                continue;
            }
            let bytes = std::fs::read(&path)?;
            out.insert(rel, format!("{:x}", Sha256::digest(&bytes)));
        }
        Ok(())
    }
    let mut out = BTreeMap::new();
    walk(root, root, &mut out)?;
    Ok(out)
}

/// Write `<dst_root>/SHA256SUMS` in `sha256sum` format (`<hex>  <relative path>`, one per line,
/// sorted) over every other file in the tier — the manifest that binds a published artefact to
/// its bytes. `sha256sum -c SHA256SUMS` from inside the tier verifies it.
pub fn write_sha256sums(dst_root: &Path) -> Result<()> {
    let mut manifest = String::new();
    for (rel, digest) in digest_tree(dst_root)? {
        manifest.push_str(&format!("{digest}  {rel}\n"));
    }
    std::fs::write(dst_root.join(SHA256SUMS_FILE), manifest)?;
    Ok(())
}

// ============================================================================================
// The released-geometry invariant the single group size rests on.
// ============================================================================================

/// Every `Linear` input width the released DiT geometry implies, so the group-alignment invariant
/// can be checked from `transformer/config.json` arithmetic rather than from 14 GB of weights.
///
/// `inner = num_attention_heads · attention_head_dim`. The widths are: `in_channels` (`img_in`),
/// `context_in_dim` (`txt_in.in_layer`), `inner` (every block projection, `txt_in.out_layer`,
/// `modulation.1`, `norm_out.linear`, `proj_out`), `inner · mlp_ratio` (the FFN's `out`), and the
/// timestep embedder's sinusoidal width — which this port fixes at `inner` for `linear_1` and
/// `inner` for `linear_2`, both already covered.
pub fn transformer_linear_widths(cfg: &TransformerConfig) -> Vec<usize> {
    let inner = cfg.num_attention_heads * cfg.attention_head_dim;
    vec![
        cfg.in_channels,
        cfg.context_in_dim,
        inner,
        inner * cfg.mlp_ratio,
    ]
}

/// `true` iff every width in [`transformer_linear_widths`] is a multiple of [`GROUP_SIZE`] — i.e.
/// this geometry packs **completely** at the one group size a tier may declare.
pub fn transformer_widths_are_group_aligned(cfg: &TransformerConfig) -> bool {
    transformer_linear_widths(cfg)
        .into_iter()
        .all(|w| w.is_multiple_of(GROUP_SIZE as usize))
}

/// The language tower's `Linear` input widths: `hidden_size` (QKV and the FFN's gate/up) and
/// `intermediate_size` (the FFN's down projection). `o_proj` reads
/// `num_attention_heads · head_dim`, which equals `hidden_size` on every Qwen3 geometry but is
/// listed separately rather than assumed.
pub fn text_encoder_linear_widths(cfg: &TextEncoderConfig) -> Vec<usize> {
    vec![
        cfg.hidden_size,
        cfg.intermediate_size,
        cfg.num_attention_heads * cfg.head_dim,
    ]
}

/// `true` iff every width in [`text_encoder_linear_widths`] is a multiple of [`GROUP_SIZE`].
pub fn text_encoder_widths_are_group_aligned(cfg: &TextEncoderConfig) -> bool {
    text_encoder_linear_widths(cfg)
        .into_iter()
        .all(|w| w.is_multiple_of(GROUP_SIZE as usize))
}

#[cfg(test)]
mod tests {
    use super::*;
    use mlx_rs::ops::{eq, quantize};
    use mlx_rs::{Array, Dtype};
    use std::collections::HashMap;

    #[test]
    fn transformer_predicate_packs_every_projection_and_no_norm() {
        for base in [
            "img_in",
            "proj_out",
            "norm_out.linear",
            "modulation.1",
            "txt_in.in_layer",
            "txt_in.out_layer",
            "time_text_embed.timestep_embedder.linear_1",
            "time_text_embed.timestep_embedder.linear_2",
            "transformer_blocks.0.attn.to_q",
            "transformer_blocks.31.attn.to_out.0",
            "transformer_blocks.7.img_mlp.gate_layer",
            "transformer_blocks.7.img_mlp.proj",
            "transformer_blocks.7.img_mlp.out",
        ] {
            assert!(is_transformer_target(base), "{base} must pack");
        }
        for base in [
            "txt_in.text_norm",
            "transformer_blocks.0.attn.norm_q",
            "transformer_blocks.31.attn.norm_k",
        ] {
            assert!(!is_transformer_target(base), "{base} must stay dense");
        }
    }

    #[test]
    fn text_encoder_predicate_is_an_allow_list_over_the_language_tower_only() {
        for base in [
            "model.language_model.layers.0.self_attn.q_proj",
            "model.language_model.layers.0.self_attn.k_proj",
            "model.language_model.layers.0.self_attn.v_proj",
            "model.language_model.layers.35.self_attn.o_proj",
            "model.language_model.layers.35.mlp.gate_proj",
            "model.language_model.layers.35.mlp.up_proj",
            "model.language_model.layers.35.mlp.down_proj",
        ] {
            assert!(is_text_encoder_target(base), "{base} must pack");
        }
        for base in [
            // Loaded but deliberately dense: packing it would break pre-quantize ≡ quantize-at-load.
            "model.language_model.embed_tokens",
            "model.language_model.norm",
            "model.language_model.layers.0.input_layernorm",
            "model.language_model.layers.0.post_attention_layernorm",
            "model.language_model.layers.0.self_attn.q_norm",
            "model.language_model.layers.0.self_attn.k_norm",
            // Never loaded by this route: they ride through dense.
            "lm_head",
            "model.visual.blocks.0.attn.qkv",
            "model.visual.merger.linear_fc1",
            // Near-misses that must not be admitted.
            "model.language_model.layers.x.mlp.up_proj",
            "model.language_model.layers.0.mlp.up_proj.extra",
            "layers.0.mlp.up_proj",
        ] {
            assert!(!is_text_encoder_target(base), "{base} must stay dense");
        }
    }

    /// The released geometries pack completely at the one group size a tier may declare — which is
    /// what makes "a packed tier is byte-identical to quantizing at load" exact in production. Read
    /// from the frozen `config.json` numbers, not from the weights.
    #[test]
    fn released_geometries_are_fully_group_aligned() {
        let dit = TransformerConfig::production();
        assert_eq!(
            transformer_linear_widths(&dit),
            vec![64, 4096, 4096, 12288],
            "the released DiT's four distinct Linear input widths"
        );
        assert!(transformer_widths_are_group_aligned(&dit));

        let te = TextEncoderConfig::production();
        assert_eq!(text_encoder_linear_widths(&te), vec![4096, 12288, 4096]);
        assert!(text_encoder_widths_are_group_aligned(&te));
    }

    fn byte_equal(a: &Array, b: &Array) -> bool {
        a.shape() == b.shape()
            && a.dtype() == b.dtype()
            && eq(a, b).unwrap().all(None).unwrap().item::<bool>()
    }

    /// The packed triple is byte-identical to the op `AdaptableLinear::quantize` runs at load
    /// (bf16 cast, group 64) — the pre-quantize-on-disk ≡ quantize-at-load guarantee — and an
    /// excluded norm passes through untouched.
    #[test]
    fn packed_targets_are_byte_identical_to_load_time_quantize() {
        for bits in [4, 8] {
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

            let out = quantize_map(map, bits, GROUP_SIZE, is_transformer_target).unwrap();
            let wq = out
                .get("transformer_blocks.0.attn.to_q.weight")
                .expect("packed codes");
            assert_eq!(wq.dtype(), Dtype::Uint32, "Q{bits} codes are u32-packed");
            let (ewq, esc, ebi) =
                quantize(w.as_dtype(Dtype::Bfloat16).unwrap(), GROUP_SIZE, bits).unwrap();
            assert!(byte_equal(wq, &ewq), "Q{bits} weight");
            assert!(byte_equal(
                out.get("transformer_blocks.0.attn.to_q.scales").unwrap(),
                &esc
            ));
            assert!(byte_equal(
                out.get("transformer_blocks.0.attn.to_q.biases").unwrap(),
                &ebi
            ));
            let n = out
                .get("transformer_blocks.0.attn.norm_q.weight")
                .expect("dense norm");
            assert_eq!(n.dtype(), Dtype::Float32);
            assert!(!out.contains_key("transformer_blocks.0.attn.norm_q.scales"));
        }
    }

    /// A width the single declared group size cannot cover stays dense rather than being packed at
    /// a second group size the loader would decode at the wrong bit-width.
    #[test]
    fn a_width_below_the_group_size_stays_dense() {
        let mut map: HashMap<String, Array> = HashMap::new();
        map.insert(
            "img_in.weight".into(),
            Array::ones::<f32>(&[16, 32]).unwrap(),
        );
        let out = quantize_map(map, 4, GROUP_SIZE, is_transformer_target).unwrap();
        assert!(
            !out.contains_key("img_in.scales"),
            "a 32-wide Linear cannot be packed at group 64 and must pass through dense"
        );
        assert_eq!(out.get("img_in.weight").unwrap().dtype(), Dtype::Float32);
    }

    #[test]
    fn the_dense_tier_is_refused_rather_than_copied() {
        let tmp = tempfile::tempdir().unwrap();
        let err = prequantize_turnkey(tmp.path(), &tmp.path().join("out"), Tier::Bf16)
            .unwrap_err()
            .to_string();
        assert!(err.contains("mirror it"), "{err}");
        assert!(change_record(Tier::Bf16).is_err());
    }

    fn tiny_snapshot() -> std::path::PathBuf {
        Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/tiny-snapshot")
    }

    /// **The converter itself writes the change record and the manifest** (sc-24114): every
    /// converted tier carries a `CHANGES.md` naming what was re-packed, at what width, from which
    /// upstream revision and by which converter, and a `SHA256SUMS` whose digests match every
    /// other file in the tier. Neither is the example driver's to add.
    ///
    /// *Mutation that reds this:* dropping the `write_change_record` / `write_sha256sums` calls
    /// from the tail of `prequantize_turnkey` (the driver used to write `SHA256SUMS` on its own).
    #[test]
    fn a_converted_tier_carries_its_change_record_and_manifest() {
        use sha2::{Digest, Sha256};

        let tmp = tempfile::tempdir().unwrap();
        for tier in [Tier::Q8, Tier::Q4] {
            let out = tmp.path().join(tier.dir_name());
            prequantize_turnkey(&tiny_snapshot(), &out, tier).unwrap();
            let bits = tier.transformer_bits().unwrap();

            // The change record: the licence's notice of what was changed, and nothing vague.
            let changes = std::fs::read_to_string(out.join(CHANGES_FILE)).unwrap();
            assert_eq!(changes, change_record(tier).unwrap());
            for needle in [
                &format!("`{}` tier", tier.dir_name()),
                &format!("Q{bits}, group size {GROUP_SIZE}"),
                "`transformer/model.safetensors`",
                "`text_encoder/model.safetensors`",
                "`transformer/config.json`",
                "`text_encoder/config.json`",
                "`vae/`",
                crate::UPSTREAM_HF_REPO,
                crate::UPSTREAM_HF_REVISION,
                env!("CARGO_PKG_NAME"),
                env!("CARGO_PKG_VERSION"),
                "§3(b)",
                SHA256SUMS_FILE,
            ] {
                assert!(
                    changes.contains(needle),
                    "{}: missing {needle:?}",
                    tier.dir_name()
                );
            }
            // Both packed components are named at the SAME width — a tier is one width.
            assert!(
                !changes.contains(&format!("Q{}", if bits == 4 { 8 } else { 4 })),
                "{}: the record names a second width:\n{changes}",
                tier.dir_name()
            );

            // The manifest: every other file, each digest right, itself excluded, sorted.
            let manifest = std::fs::read_to_string(out.join(SHA256SUMS_FILE)).unwrap();
            let mut listed = Vec::new();
            for line in manifest.lines() {
                let (digest, rel) = line
                    .split_once("  ")
                    .unwrap_or_else(|| panic!("malformed manifest line {line:?}"));
                assert_ne!(rel, SHA256SUMS_FILE, "the manifest must not list itself");
                let bytes = std::fs::read(out.join(rel)).unwrap();
                assert_eq!(
                    digest,
                    format!("{:x}", Sha256::digest(&bytes)),
                    "{}: {rel} digest",
                    tier.dir_name()
                );
                listed.push(rel.to_owned());
            }
            let mut sorted = listed.clone();
            sorted.sort();
            assert_eq!(listed, sorted, "the manifest is sorted");
            for rel in [
                CHANGES_FILE,
                "transformer/model.safetensors",
                "transformer/config.json",
                "text_encoder/model.safetensors",
                "text_encoder/config.json",
                "vae/config.json",
            ] {
                assert!(listed.iter().any(|l| l == rel), "{rel} must be listed");
            }
            assert_eq!(
                listed.len(),
                digest_tree(&out).unwrap().len(),
                "every regular file but the manifest is listed"
            );
        }
    }
}
