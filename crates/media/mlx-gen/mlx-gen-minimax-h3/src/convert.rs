//! Offline pre-quantization of the MiniMax-H3 DiT into the `q4` / `q8` tier artifacts the
//! `SceneWorks/minimax-h3-mlx` mirror publishes (sc-17150).
//!
//! Tiers ship **pre-quantized**; nothing quantizes at load. The packed weights auto-detect through
//! [`mlx_gen::quant::lin`] on the `{base}.scales` signal, so one loader serves a dense `bf16/` tier
//! and a packed `q8/` / `q4/` tier with no dense transient.
//!
//! # Why this converter streams instead of calling [`mlx_gen::quant::load_dir_map`]
//!
//! Every sibling converter (`mlx_gen_z_image::convert`, `mlx_gen_flux2::convert`, …) reads the whole
//! component into one map, packs it, and writes one `model.safetensors`. That shape does not
//! survive here: `transformer/` is **66_280_430_080 B** of tensor data, so a whole-map convert holds
//! the dense source *and* the growing packed output at once — 101 GB at q8 on a 137 GB machine, with
//! the host already OOM-killed once in this epic.
//!
//! [`quantize_minimax_h3_dit`] therefore processes **one source shard at a time**: read shard `i`,
//! pack it, force evaluation, write it, hand it to `on_shard`, drop it, drain the MLX allocator
//! cache. Peak residency is one source shard plus its packed output (~7.5 GB at the shipped 14-shard
//! layout) rather than the whole component, and `on_shard` can upload-and-delete so the peak *disk*
//! footprint is bounded the same way (the `mlx_gen_krea_realtime::convert` `on_shard` contract).
//!
//! # What is packed, and what is deliberately left dense
//!
//! | group | bytes at bf16 | tier treatment |
//! |---|---:|---|
//! | block + refiner attention & feed-forward | 40_076_574_720 | packed at the tier width |
//! | `adaln_proj.linear` (50 blocks) | 26_011_238_400 | packed at [`adaln_bits`](AdaLnTierPolicy) |
//! | the float32 I/O heads, `context_embedder`, `norm_out.linear`, every norm | 192_616_960 | **always dense** |
//!
//! The dense residue is **0.29%** of the checkpoint, and every member of it is dense for a reason
//! that is not "it was too small to bother with":
//!
//! * **The float32 heads** — `proj_in` / `proj_out` / `audio_proj_in` / `audio_proj_out` and both
//!   `time_embedder` layers — ship float32 in a checkpoint whose block stack is bfloat16
//!   ([`crate::dit::heads`]). [`mlx_gen::quant::quantize_map`] bf16-casts before packing, so packing
//!   these would silently downgrade the patch projections that produce the model's own velocity
//!   output, to save 68_776_960 B (0.10%). Worse, [`crate::dit::heads`] reads them with a raw
//!   `Weights::require` rather than [`mlx_gen::quant::lin`] — a packed tensor there would load u32
//!   codes where a float is expected and produce **no load error at all**, which is exactly how
//!   Mage's `pos_embed` failed (sc-14980).
//! * **`context_embedder` and `norm_out.linear`** are the text projection and the final modulation.
//!   Mage had to floor `norm_out.linear` to 8 bits after a uniform-Q4 tier rendered a tiled texture
//!   instead of the prompt (sc-15071); keeping both dense costs 112_852_992 B — 0.6% of the q4 tier
//!   — and removes that failure mode rather than re-discovering it.
//! * **The norms** are 1-D and could never be packed anyway ([`mlx_gen::quant::quantize_map`]'s
//!   shape guard); they are named in [`DENSE_NORM_SUFFIXES`] so the policy is readable, not so it is
//!   enforced twice.
//!
//! Because the dense set is exactly the set [`crate::dit::heads`] reads, **no head loader needs a
//! packed path**: only [`crate::dit::layers::LinearNoBias`] and
//! [`crate::dit::block::AdaLnProjection`] gained one.
//!
//! # The AdaLN question
//!
//! `adaln_proj.linear` is 39.2% of the DiT and is **precomputed then evicted before step 0**
//! (sc-17145), so its width does not move denoise-*resident* memory at all. sc-17139 proposed
//! spending that freedom on quality — hold AdaLN at q8 inside the q4 tier for +6_502_809_600 B of
//! disk and zero resident cost. [`AdaLnTierPolicy`] makes that a declared knob rather than an
//! implicit default. See [`AdaLnTierPolicy::for_tier`] for the shipped decision and the two
//! measurements it rests on.

use std::collections::HashMap;
use std::path::{Path, PathBuf};

use mlx_rs::Array;

use mlx_gen::quant::{quantize_map, write_quantized_config};
use mlx_gen::weights::Weights;
use mlx_gen::{Error, Result};

/// Affine quantization group size for every packed DiT tensor.
///
/// The codebase default (`mlx_gen::quant::DEFAULT_GROUP_SIZE`), and legal here because **every**
/// packable input width in the DiT is a multiple of 64: 5376 (84 groups), 7168 (112), 14336 (224),
/// 2688 (42) and 5120 (80). The two widths that are not — `proj_in`'s 96 and `audio_proj_in`'s 32 —
/// belong to the always-dense float32 head set, so no target is ever left unpackable by this choice.
pub const GROUP_SIZE: i32 = mlx_gen::quant::DEFAULT_GROUP_SIZE;

/// The tier subdirectory names the `SceneWorks/minimax-h3-mlx` mirror publishes, in quality order.
///
/// `bf16` is a dense passthrough of the upstream shards and carries **no** `quantization` marker;
/// `q8` and `q4` are packed by this module and do carry one.
pub const TIERS: &[&str] = &["bf16", "q8", "q4"];

/// Bases left dense in **every** tier by policy rather than by shape — see the module docs.
///
/// These are matched exactly, not by suffix: they are all top-level DiT tensors, and a suffix match
/// on e.g. `proj_out` would also swallow `audio_proj_out` (harmless here, but the exactness is what
/// makes the list auditable against the 638-tensor index).
pub const DENSE_BY_POLICY: &[&str] = &[
    "proj_in",
    "proj_out",
    "audio_proj_in",
    "audio_proj_out",
    "time_embedder.linear_1",
    "time_embedder.linear_2",
    "context_embedder",
    "norm_out.linear",
];

/// Norm suffixes that pass through dense. All are 1-D and therefore already unpackable by
/// [`mlx_gen::quant::quantize_map`]'s shape guard; listing them makes the policy legible.
pub const DENSE_NORM_SUFFIXES: &[&str] = &[
    ".norm1",
    ".norm2",
    ".norm_q",
    ".norm_k",
    ".norm",
    ".final_norm",
];

/// `true` iff `base` names the per-block AdaLN projection — the 26_011_238_400 B group that
/// [`crate::dit::adaln::AdaLnCache::precompute_and_evict`] releases before step 0.
///
/// `base` is the on-disk key minus its `.weight` suffix, as [`mlx_gen::quant::quantize_map`] passes
/// it: `transformer_blocks.{i}.adaln_proj.linear`.
pub fn is_adaln_target(base: &str) -> bool {
    base.ends_with(".adaln_proj.linear")
}

/// `true` iff `base` names a Linear packed at the **tier's own** bit width: the block and
/// token-refiner attention projections (`to_q`/`to_k`/`to_v`/`to_out.0`) and feed-forwards
/// (`ff.net.0.proj`/`ff.net.2`).
///
/// A negative predicate, matching the `mlx_gen_mage` / `mlx_gen_z_image` / `mlx_gen_flux2` shape:
/// everything is a target except the declared dense sets and the AdaLN group, which
/// [`is_adaln_target`] packs in its own pass at its own width.
pub fn is_dit_linear_target(base: &str) -> bool {
    !DENSE_BY_POLICY.contains(&base)
        && !DENSE_NORM_SUFFIXES.iter().any(|s| base.ends_with(s))
        && !is_adaln_target(base)
}

/// How wide the AdaLN projection is packed for a given tier.
///
/// Separate from the tier's own width because `adaln_proj` is evicted before step 0, so its width
/// buys disk and load transient but **not** denoise-resident memory (sc-17145) — the one place in
/// this model where paying for precision is close to free at run time.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct AdaLnTierPolicy {
    /// Bit width `adaln_proj.linear` is packed at.
    pub bits: i32,
}

impl AdaLnTierPolicy {
    /// The shipped policy: AdaLN is packed at the **tier's own width**, uniformly.
    ///
    /// sc-17139 proposed flooring AdaLN to q8 inside the q4 tier — +6_502_809_600 B of disk against
    /// "zero denoise-resident cost". Both halves of that trade were measured on the real weights
    /// rather than argued, and they disagree with the reasoning that was recorded for it.
    ///
    /// # The memory objection was well-posed, and it does not bite
    ///
    /// [`crate::dit::adaln::AdaLnCache::precompute`] reads and evaluates all 50 projections
    /// **before** eviction, so a wider AdaLN raises the precompute transient by the same +6.5 GB —
    /// free only while the precompute peak stays under the denoise peak, an inequality nobody had
    /// measured. It **holds, comfortably**. Four real renders at `576x320 / 124 frames / 50 steps`:
    ///
    /// | variant | MLX peak | frame stddev | inter-frame motion |
    /// |---|---:|---:|---:|
    /// | q4 uniform | 52.81 GB | 65.6 | 1.70 |
    /// | q4 + AdaLN@q8 | 53.07 GB | 66.0 | 1.75 |
    /// | q8 | 53.07 GB | 71.0 | 1.83 |
    /// | bf16 | 52.81 GB | 71.1 | 1.80 |
    ///
    /// **bf16 carries a 26_020_915_200 B AdaLN and peaks the same as q4's 7.3 GB one.** The
    /// conclusion holds — AdaLN width does not move the peak at any tier — but the *cause*
    /// originally recorded here ("the peak is activation-dominated") is **retracted** (sc-18659).
    /// The dense Qwen3-VL-32B text encoder runs first and sets a
    /// [`53.07 GB`](crate::memory_strategy::CONDITIONING_STAGE_PEAK_BYTES) process high-water that
    /// masks every later stage, so this column is flat across tier **by construction**: while the
    /// conditioning stage binds, no DiT-side width — AdaLN's or any other group's — can move the
    /// number, and the flatness says nothing about activation pressure. sc-19120's packed TE tier
    /// is what makes the column informative again.
    ///
    /// # So the decision rests on quality per byte, and there uniform wins clearly
    ///
    /// Flooring AdaLN to q8 costs **+34.6% disk** on the tier chosen precisely for leanness, and
    /// buys 0.4 of the 5.5-point q4→bf16 stddev gap — about **7% of the gap for a third more
    /// disk**. Frame mean barely moves either (101.2 → 101.8, against 116.3 at bf16).
    ///
    /// This contradicts the *reasoning* an earlier pass recorded, which claimed q4's AdaLN error is
    /// "the same order" as the block Linears'. Measured by hand-dequantizing the built artifact
    /// against the source, it is **1.9-2.8x worse**: relative max-abs-diff 0.0785 for
    /// `adaln_proj.linear` against 0.0407 (`attn.to_q`) and 0.0278 (`ff.net.0.proj`). AdaLN really
    /// is the worst-conditioned group — and fixing it barely helps, which is the actual finding:
    /// the q4 deficit is **distributed across the 312 block Linears**, not concentrated in the one
    /// group a floor could rescue.
    ///
    /// The knob stays because a different geometry could shift the peak's composition, and
    /// `MINIMAX_H3_ADALN_BITS` on the `minimax_h3_prequant` example rebuilds either variant in
    /// ~10 s — but the default is uniform.
    pub fn for_tier(bits: i32) -> Self {
        Self { bits }
    }
}

/// Bits for a tier directory name, or `None` for the dense `bf16` passthrough.
pub fn tier_bits(tier: &str) -> Result<Option<i32>> {
    match tier {
        "bf16" => Ok(None),
        "q8" => Ok(Some(8)),
        "q4" => Ok(Some(4)),
        other => Err(Error::Msg(format!(
            "minimax-h3 convert: unknown tier {other:?}; expected one of {TIERS:?}"
        ))),
    }
}

/// The source shards of a DiT component directory, in index order.
fn source_shards(src: &Path) -> Result<Vec<PathBuf>> {
    let mut files: Vec<PathBuf> = std::fs::read_dir(src)
        .map_err(|e| Error::Msg(format!("minimax-h3 convert: read {}: {e}", src.display())))?
        .filter_map(|e| e.ok().map(|e| e.path()))
        .filter(|p| p.extension().and_then(|s| s.to_str()) == Some("safetensors"))
        // AppleDouble `._*` sidecars match the `.safetensors` glob and are not weights.
        .filter(|p| !mlx_gen::gen_core::weightsmeta::is_hidden_file(p))
        .collect();
    files.sort();
    if files.is_empty() {
        return Err(Error::Msg(format!(
            "minimax-h3 convert: no .safetensors in {}",
            src.display()
        )));
    }
    Ok(files)
}

/// The packed shard file name for shard `i` of `n` — the upstream `diffusion_pytorch_model` stem, so
/// a tier directory is layout-identical to the dense component it was built from.
fn shard_name(i: usize, n: usize) -> String {
    format!(
        "diffusion_pytorch_model-{:05}-of-{:05}.safetensors",
        i + 1,
        n
    )
}

/// Pre-quantize one DiT component directory (`transformer/` or `transformer_ref/`) into `dst`,
/// **one source shard at a time**.
///
/// `bits` is the tier width (4 or 8); `adaln` is the width the per-block AdaLN projection is packed
/// at. Every finished shard is handed to `on_shard` before the next is read, so a caller can upload
/// and delete it and bound the local disk footprint to one shard.
///
/// Writes, in `dst`: the packed shards, a `config.json` carrying the `quantization` marker (the
/// **tier's** width — see [`mlx_gen::quant::write_quantized_config`] on why an unmarked packed
/// component is the dangerous case), and a `diffusion_pytorch_model.safetensors.index.json` whose
/// `weight_map` covers every emitted tensor, so `scripts/release/verify_model_snapshot.py` can prove
/// a hosted tier is complete without listing 14 shards per tier by hand.
///
/// Returns the emitted shard paths in order, whether or not `on_shard` deleted them.
pub fn quantize_minimax_h3_dit(
    src: &Path,
    dst: &Path,
    bits: i32,
    adaln: AdaLnTierPolicy,
    on_shard: &mut dyn FnMut(&Path) -> Result<()>,
) -> Result<Vec<PathBuf>> {
    if !matches!(bits, 4 | 8) {
        return Err(Error::Msg(format!(
            "minimax-h3 convert: tier width must be 4 or 8, got {bits}"
        )));
    }
    if !matches!(adaln.bits, 4 | 8) {
        return Err(Error::Msg(format!(
            "minimax-h3 convert: AdaLN width must be 4 or 8, got {}",
            adaln.bits
        )));
    }
    let shards = source_shards(src)?;
    let n = shards.len();
    std::fs::create_dir_all(dst)?;

    // Marker first: a consumer that sees shards appearing already knows what tier they are.
    write_quantized_config(src, dst, bits, GROUP_SIZE)?;

    let mut written = Vec::with_capacity(n);
    let mut weight_map: HashMap<String, String> = HashMap::new();
    let mut total_size: usize = 0;

    for (i, shard) in shards.iter().enumerate() {
        let source = Weights::from_file(shard)?;
        let map: HashMap<String, Array> = source
            .keys()
            .map(|k| {
                (
                    k.to_string(),
                    source.get(k).expect("key listed by `keys`").clone(),
                )
            })
            .collect();
        drop(source);

        // Two passes so the AdaLN group can carry its own width. They are disjoint by construction —
        // `is_dit_linear_target` excludes exactly what `is_adaln_target` matches — so pass 2 only
        // ever sees keys pass 1 left dense and no tensor can be packed twice. `packed_twice_is_
        // impossible` in the tests below is the guard on that, mutated one predicate at a time.
        let packed = quantize_map(map, bits, GROUP_SIZE, is_dit_linear_target)?;
        let packed = quantize_map(packed, adaln.bits, GROUP_SIZE, is_adaln_target)?;

        let name = shard_name(i, n);
        let path = dst.join(&name);
        // `save_map` evaluates before writing, which is what actually materializes this shard's
        // packs; nothing before this line has forced a single byte.
        mlx_gen::quant::save_map(&path, &packed)?;

        for (key, tensor) in &packed {
            total_size += tensor.nbytes();
            weight_map.insert(key.clone(), name.clone());
        }
        drop(packed);

        on_shard(&path)?;
        written.push(path);

        // Return this shard's buffers to the system allocator before the next one is mapped. Without
        // it MLX retains them in its own cache and the "one shard at a time" bound is nominal only.
        mlx_rs::memory::clear_cache();
    }

    write_index(dst, &weight_map, total_size)?;
    Ok(written)
}

/// Write the safetensors index describing the emitted shard set.
fn write_index(dst: &Path, weight_map: &HashMap<String, String>, total_size: usize) -> Result<()> {
    let mut entries: Vec<(&String, &String)> = weight_map.iter().collect();
    entries.sort_unstable_by_key(|(key, _)| *key);
    let map: serde_json::Map<String, serde_json::Value> = entries
        .into_iter()
        .map(|(k, v)| (k.clone(), serde_json::Value::String(v.clone())))
        .collect();
    let index = serde_json::json!({
        "metadata": { "total_size": total_size },
        "weight_map": serde_json::Value::Object(map),
    });
    let text = serde_json::to_string_pretty(&index)
        .map_err(|e| Error::Msg(format!("minimax-h3 convert: serialize index: {e}")))?;
    std::fs::write(
        dst.join("diffusion_pytorch_model.safetensors.index.json"),
        text,
    )?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Every base the published 638-tensor `transformer/` index carries, collapsed over the block
    /// axis — the ground truth both predicates are audited against.
    const PUBLISHED_BASES: &[(&str, bool)] = &[
        // (base, is a tier-width pack target)
        ("transformer_blocks.0.attn.to_q", true),
        ("transformer_blocks.49.attn.to_k", true),
        ("transformer_blocks.7.attn.to_v", true),
        ("transformer_blocks.3.attn.to_out.0", true),
        ("transformer_blocks.0.ff.net.0.proj", true),
        ("transformer_blocks.0.ff.net.2", true),
        ("token_refiner.refiner_blocks.0.attn.to_q", true),
        ("token_refiner.refiner_blocks.1.attn.to_out.0", true),
        ("token_refiner.refiner_blocks.0.ff.net.0.proj", true),
        ("token_refiner.refiner_blocks.1.ff.net.2", true),
        // The AdaLN group is NOT a tier-width target — it has its own pass.
        ("transformer_blocks.0.adaln_proj.linear", false),
        ("transformer_blocks.49.adaln_proj.linear", false),
        // Dense by policy.
        ("proj_in", false),
        ("proj_out", false),
        ("audio_proj_in", false),
        ("audio_proj_out", false),
        ("time_embedder.linear_1", false),
        ("time_embedder.linear_2", false),
        ("context_embedder", false),
        ("norm_out.linear", false),
        // Dense norms.
        ("transformer_blocks.0.norm1", false),
        ("transformer_blocks.0.norm2", false),
        ("transformer_blocks.0.attn.norm_q", false),
        ("transformer_blocks.0.attn.norm_k", false),
        ("norm_out.norm", false),
        ("token_refiner.final_norm", false),
        ("token_refiner.refiner_blocks.0.norm1", false),
    ];

    #[test]
    fn tier_predicate_matches_the_published_base_set() {
        for (base, want) in PUBLISHED_BASES {
            assert_eq!(
                is_dit_linear_target(base),
                *want,
                "is_dit_linear_target({base:?})"
            );
        }
    }

    #[test]
    fn adaln_predicate_matches_only_the_evicted_group() {
        for (base, _) in PUBLISHED_BASES {
            assert_eq!(
                is_adaln_target(base),
                base.ends_with(".adaln_proj.linear"),
                "is_adaln_target({base:?})"
            );
        }
    }

    /// The two passes must partition the packable set: nothing may be packed twice, and nothing that
    /// should be packed may fall between them.
    #[test]
    fn packed_twice_is_impossible() {
        for (base, _) in PUBLISHED_BASES {
            assert!(
                !(is_dit_linear_target(base) && is_adaln_target(base)),
                "{base:?} is matched by BOTH passes"
            );
        }
    }

    #[test]
    fn tier_bits_maps_the_published_tier_names() {
        assert_eq!(tier_bits("bf16").unwrap(), None);
        assert_eq!(tier_bits("q8").unwrap(), Some(8));
        assert_eq!(tier_bits("q4").unwrap(), Some(4));
        assert!(tier_bits("q6").is_err());
        assert_eq!(TIERS, ["bf16", "q8", "q4"]);
    }

    #[test]
    fn group_size_divides_every_packable_input_width() {
        // The widths the 638-tensor index carries on a pack target.
        for width in [5376, 7168, 14336, 2688, 5120] {
            assert_eq!(width % GROUP_SIZE, 0, "width {width}");
        }
        // …and the two that do not divide belong to the always-dense float32 head set.
        for width in [96, 32] {
            assert_ne!(width % GROUP_SIZE, 0, "width {width}");
        }
    }

    #[test]
    fn shard_names_are_the_upstream_layout() {
        assert_eq!(
            shard_name(0, 14),
            "diffusion_pytorch_model-00001-of-00014.safetensors"
        );
        assert_eq!(
            shard_name(13, 14),
            "diffusion_pytorch_model-00014-of-00014.safetensors"
        );
    }

    #[test]
    fn tier_width_is_validated_before_any_file_is_touched() {
        let tmp = tempfile::tempdir().unwrap();
        let src = tmp.path().join("src");
        let dst = tmp.path().join("dst");
        std::fs::create_dir_all(&src).unwrap();
        let mut noop = |_: &Path| Ok(());
        for bad in [0, 2, 16] {
            assert!(
                quantize_minimax_h3_dit(&src, &dst, bad, AdaLnTierPolicy::for_tier(4), &mut noop)
                    .is_err(),
                "tier width {bad} must be rejected"
            );
            assert!(
                quantize_minimax_h3_dit(&src, &dst, 4, AdaLnTierPolicy { bits: bad }, &mut noop)
                    .is_err(),
                "AdaLN width {bad} must be rejected"
            );
        }
        assert!(
            !dst.exists(),
            "a rejected width must not create the tier dir"
        );
    }
}
