//! Krea Realtime 14B transformer weight converter (sc-8435 S2).
//!
//! Krea Realtime 14B is Wan 2.1 T2V 14B weight-for-weight, so its native transformer tensor **set** is
//! identical to what [`mlx_gen_wan::convert::sanitize_wan_transformer`] already normalizes. The only
//! on-disk delta is layout:
//!
//!   1. **single-file** `krea-realtime-video-14b.safetensors` — every key prefixed **`model.`**
//!      (e.g. `model.blocks.0.self_attn.q.weight`, `model.patch_embedding.weight`), and
//!   2. **sharded** `transformer/` (3 shards + index) — the **same** key names **without** the
//!      `model.` prefix (the plain Wan native layout).
//!
//! Both on-disk layouts are the reference `krea-ai/realtime-video` checkpoint's shipped formats, so the
//! key normalization here is **adapted from** the reference checkpoint layout: [`normalize_krea_keys`]
//! strips a leading `model.` from every key so both layouts collapse to the plain Wan native names, then
//! the shared Wan sanitizer maps them onto the internal [`mlx_gen_wan::WanTransformer`] key layout. The
//! reference runs the DiT in **bf16**, so — like every Wan converter
//! ([`mlx_gen_wan::convert::convert_ti2v_5b`] etc.) — we cast the (F16) checkpoint to
//! [`TRANSFORMER_DTYPE`] (`Bfloat16`) on the way out.
//!
//! This is the **non-gated** S2 converter: it is validated against the S1 tensor inventory with
//! synthesized fixtures (`tests/`) — never the real 28.58 GB checkpoint. Real-weight byte parity + the
//! MLX rehost are the gated S2 remainder tracked on sc-8435.

use std::collections::HashMap;
use std::path::{Path, PathBuf};

use mlx_gen::quant::save_map;
use mlx_gen::weights::Weights;
use mlx_gen::{Error, Result};
use mlx_gen_wan::config::WanQuant;
use mlx_gen_wan::convert::sanitize_wan_transformer;
use mlx_rs::{Array, Dtype};

use crate::config::KreaRealtimeConfig;
use crate::load::verify_transformer_tensors;

/// The prefix the **single-file** Krea Realtime checkpoint layout adds to every tensor name. The
/// sharded `transformer/` layout omits it, so stripping it (when present) normalizes both to the plain
/// Wan native names.
pub const KREA_MODEL_PREFIX: &str = "model.";

/// The converted transformer's filename inside a snapshot directory — what
/// [`crate::t2v`]'s load path reads.
pub const DIT_FILE: &str = "dit.safetensors";

/// The dtype the converted transformer ships in. The Krea Realtime reference runs the DiT in bf16
/// (the F16 checkpoint is cast on load), matching every other Wan converter's bf16 transformer.
pub const TRANSFORMER_DTYPE: Dtype = Dtype::Bfloat16;

/// Strip a single leading [`KREA_MODEL_PREFIX`] from `name` if present (else return it unchanged). The
/// prefix is only ever the outermost `model.` the single-file layout adds — the sharded layout's keys
/// (and every inner `.self_attn`/`.ffn` segment) are untouched.
pub fn strip_model_prefix(name: &str) -> &str {
    name.strip_prefix(KREA_MODEL_PREFIX).unwrap_or(name)
}

/// Normalize either on-disk layout to the plain Wan native key names by stripping a leading `model.`
/// from every key. The sharded `transformer/` layout has no prefix, so this is a no-op there; the
/// single-file layout's `model.` prefix is removed so both feed [`sanitize_wan_transformer`]
/// identically.
pub fn normalize_krea_keys(raw: HashMap<String, Array>) -> HashMap<String, Array> {
    raw.into_iter()
        .map(|(k, v)| (strip_model_prefix(&k).to_string(), v))
        .collect()
}

/// Cast every **floating-point** tensor in `map` to `dtype` in place (skipping any already at `dtype`).
/// Mirrors the private `cast_map` the Wan converters use for their F16/F32→bf16 transformer cast.
///
/// Integer tensors are left alone (sc-15203): on a pre-quantized Q4/Q8 tier the predicate Linears'
/// `{base}.weight` holds **u32 packed codes**, and casting those to bf16 would reinterpret 8 (Q4) or 4
/// (Q8) packed 4/8-bit weights as one float — silently destroying the tier rather than failing. The
/// `.scales`/`.biases` companions are floats and are cast (a no-op: MLX packs them at the source
/// weight's bf16).
fn cast_floats(map: &mut HashMap<String, Array>, dtype: Dtype) -> Result<()> {
    for v in map.values_mut() {
        if v.dtype().is_float() && v.dtype() != dtype {
            *v = v.as_dtype(dtype)?;
        }
    }
    Ok(())
}

/// Full Krea Realtime transformer sanitize: normalize either layout's keys ([`normalize_krea_keys`]),
/// map them onto the internal Wan DiT layout ([`sanitize_wan_transformer`] — `patch_embedding` conv →
/// Linear reshape, `text/time_embedding` Sequential rename, `ffn.0/2` → `fc1/fc2`, `freqs` dropped),
/// and cast to [`TRANSFORMER_DTYPE`] (bf16). The result is exactly the key layout
/// [`mlx_gen_wan::WanTransformer::from_weights`] consumes.
pub fn sanitize_krea_realtime_transformer(
    raw: HashMap<String, Array>,
) -> Result<HashMap<String, Array>> {
    let normalized = normalize_krea_keys(raw);
    let mut sanitized = sanitize_wan_transformer(&normalized)?;
    cast_floats(&mut sanitized, TRANSFORMER_DTYPE)?;
    Ok(sanitized)
}

/// Pack a **sanitized** Krea Realtime transformer map into a pre-quantized Q4/Q8 tier (sc-15203, S19),
/// reusing [`mlx_gen_wan::convert::quantize_wan_transformer`] verbatim — Krea Realtime is Wan-2.1-14B
/// weight-for-weight, so its `_quantize_predicate` surface (per-block self/cross-attention `q/k/v/o` +
/// `ffn.fc1`/`fc2`) is exactly Wan's, and reusing the Wan packer is what makes the packed tier loadable
/// by the reused [`mlx_gen_wan::WanTransformer::from_weights`] with no Krea-specific consume path.
///
/// Each matched `{base}.weight` (bf16) becomes the MLX affine triple `{base}.weight` (u32 codes) +
/// `{base}.scales` + `{base}.biases`; the Linear's dense `.bias`, the norms, the modulation tables, the
/// embeddings, `time_projection` and the head all pass through untouched.
pub fn quantize_krea_realtime_transformer(
    map: HashMap<String, Array>,
    bits: i32,
    group_size: i32,
) -> Result<HashMap<String, Array>> {
    if !matches!(bits, 4 | 8) {
        return Err(Error::Msg(format!(
            "krea-realtime: unsupported quantization width Q{bits}; this engine ships Q4 / Q8 / bf16"
        )));
    }
    mlx_gen_wan::convert::quantize_wan_transformer(map, bits, group_size)
}

/// Convert a native Krea Realtime 14B transformer into a complete **tier snapshot directory**: the
/// sanitized `dit.safetensors` (bf16, or packed Q4/Q8 when `quantize` is `Some((bits, group_size))`)
/// plus the `config.json` the load path reads back ([`KreaRealtimeConfig::to_json`], carrying the
/// `quantization` block on a packed tier). This is the rehost-side half of the three-tier surface
/// (sc-15203): the bf16 / Q8 / Q4 snapshots published to `SceneWorks/krea-realtime-14b-mlx` are three
/// calls to this with `None` / `Some((8, 64))` / `Some((4, 64))`.
///
/// The TE / VAE / tokenizer are stock Wan and are assembled alongside by the rehost flow (Krea Realtime
/// ships transformer-only) — this writes the DiT + config only.
///
/// The packed map is [`verify_transformer_tensors`]-checked against the emitted config's geometry
/// **before** anything is written: the `config.json` asserts the shipped 14B preset's dimensions, so a
/// source checkpoint that is not that geometry would otherwise produce a snapshot that is internally
/// inconsistent and only fails at *load* — on the real rehost, after a 28.58 GB conversion and an HF
/// upload. The check is shape-only (metadata, no materialization), so it costs nothing.
pub fn convert_krea_realtime_tier(
    src: impl AsRef<Path>,
    out_dir: impl AsRef<Path>,
    quantize: Option<(i32, i32)>,
) -> Result<PathBuf> {
    convert_krea_realtime_tier_with_config(
        src,
        out_dir,
        quantize,
        &KreaRealtimeConfig::krea_realtime_14b(),
    )
}

/// [`convert_krea_realtime_tier`] against an explicit geometry instead of the shipped
/// [`KreaRealtimeConfig::krea_realtime_14b`] preset: `cfg` is both what the emitted `config.json`
/// declares **and** what the converted tensors are verified against, so the two can never disagree.
/// The preset entry point is what the rehost drives; this seam exists so the emitter is exercised over
/// a tiny fixture geometry (`tests/quant_tiers.rs`) without the real 28.58 GB checkpoint.
pub fn convert_krea_realtime_tier_with_config(
    src: impl AsRef<Path>,
    out_dir: impl AsRef<Path>,
    quantize: Option<(i32, i32)>,
    cfg: &KreaRealtimeConfig,
) -> Result<PathBuf> {
    let out_dir = out_dir.as_ref();

    let map = read_native_map(src.as_ref())?;
    let sanitized = sanitize_krea_realtime_transformer(map)?;
    let dit = match quantize {
        Some((bits, group_size)) => {
            quantize_krea_realtime_transformer(sanitized, bits, group_size)?
        }
        None => sanitized,
    };

    let mut cfg = cfg.clone();
    cfg.wan.quantization = quantize.map(|(bits, group_size)| WanQuant { bits, group_size });
    // Fail here, before the (multi-GB) write, rather than at load: the emitted config declares this
    // geometry, so the tensors must actually be it — at the tier that was just packed.
    verify_transformer_tensors(&dit, &cfg.wan).map_err(|e| {
        Error::Msg(format!(
            "krea-realtime: refusing to write a tier snapshot whose transformer does not match the \
             config it would ship with (source {}): {e}",
            src.as_ref().display()
        ))
    })?;

    std::fs::create_dir_all(out_dir)?;
    save_map(&out_dir.join(DIT_FILE), &dit)?;
    let text = serde_json::to_string_pretty(&cfg.to_json())
        .map_err(|e| Error::Msg(format!("krea-realtime: serialize config.json: {e}")))?;
    std::fs::write(out_dir.join("config.json"), text)?;

    Ok(out_dir.to_path_buf())
}

/// The subdirectory a **sharded** tier snapshot puts its transformer shards in — the layout
/// [`crate::t2v`]'s load path falls back to when a tier ships no single-file [`DIT_FILE`]
/// (`Weights::from_dir` globs `*.safetensors` and requires the shard set to be disjoint).
pub const TRANSFORMER_DIR: &str = "transformer";

/// Default shard size for [`convert_krea_realtime_tier_sharded`] (4 GiB) — comfortably under the
/// single-file sizes HF discourages, and small enough that a disk-bounded rehost only ever holds one
/// shard locally.
pub const DEFAULT_SHARD_BYTES: u64 = 4 * 1024 * 1024 * 1024;

/// Assign `map`'s tensors to shards of at most `max_shard_bytes` (a tensor larger than the budget gets
/// its own shard). Keys are sorted first, so the plan is a pure function of the tensor set and the
/// budget — the same source always shards identically.
fn plan_shards(map: &HashMap<String, Array>, max_shard_bytes: u64) -> Vec<Vec<String>> {
    let mut names: Vec<&String> = map.keys().collect();
    names.sort();

    let mut shards: Vec<Vec<String>> = Vec::new();
    let mut current: Vec<String> = Vec::new();
    let mut used: u64 = 0;
    for name in names {
        let bytes = map[name].nbytes() as u64;
        if !current.is_empty() && used + bytes > max_shard_bytes {
            shards.push(std::mem::take(&mut current));
            used = 0;
        }
        current.push(name.clone());
        used += bytes;
    }
    if !current.is_empty() {
        shards.push(current);
    }
    shards
}

/// Does `name` look like a shard this emitter wrote (`dit-{i}-of-{n}.safetensors`)?
fn is_shard_file(name: &str) -> bool {
    name.strip_prefix("dit-")
        .and_then(|r| r.strip_suffix(".safetensors"))
        .and_then(|r| r.split_once("-of-"))
        .is_some_and(|(i, n)| {
            !i.is_empty()
                && !n.is_empty()
                && i.bytes().all(|b| b.is_ascii_digit())
                && n.bytes().all(|b| b.is_ascii_digit())
        })
}

/// Delete any shard files left in `shard_dir` by a previous emit.
///
/// The shard *count* is a function of the byte budget, so re-emitting the same tier at a different
/// `max_shard_bytes` produces a differently-named set (`…-of-00007` vs `…-of-00004`) and would
/// otherwise leave the old set sitting beside the new one. `Weights::from_dir` globs every
/// `*.safetensors` in the directory and rejects duplicate keys, so that mixture fails loudly rather
/// than loading something wrong — but on a multi-hour rehost it fails *after* the work, and a
/// delete-as-you-go publisher would already have pushed the stale names to the remote. Clearing up
/// front removes the foot-gun.
///
/// Only files matching this emitter's own `dit-{i}-of-{n}.safetensors` naming are removed; anything
/// else the caller put in the directory is left alone.
fn clear_stale_shards(shard_dir: &Path) -> Result<()> {
    for entry in std::fs::read_dir(shard_dir)? {
        let entry = entry?;
        if entry.file_name().to_str().is_some_and(is_shard_file) {
            std::fs::remove_file(entry.path())?;
        }
    }
    Ok(())
}

/// [`convert_krea_realtime_tier`] emitted as a **sharded** `transformer/` directory instead of one
/// [`DIT_FILE`], writing one shard at a time and handing each finished shard to `on_shard` before the
/// next is built.
///
/// This exists for the dense **bf16** tier of the rehost (sc-8435 S2b). That tier is a single ~28.6 GB
/// tensor set, and materializing it as one file requires that much free disk *plus* whatever the
/// upload needs — which the conversion host does not always have. Sharding bounds the requirement to
/// one shard: `on_shard` can upload the shard and delete it, so the peak local footprint is
/// `max_shard_bytes` rather than the whole tier. It is also the friendlier hosted layout for a tier
/// this size (resumable per-shard downloads instead of one ~28.6 GB object).
///
/// The emitted directory is `<out_dir>/transformer/dit-{i}-of-{n}.safetensors` plus the same
/// `<out_dir>/config.json` the single-file emitter writes, which is exactly the sharded layout
/// [`crate::t2v`]'s loader already accepts — so a sharded tier and a single-file tier are
/// interchangeable to every consumer.
///
/// Like the single-file emitter, the **whole** tensor set is [`verify_transformer_tensors`]-checked
/// against the emitted config *before* any shard is written, so a geometry mismatch fails up front
/// rather than after some shards have already been uploaded. Each shard's tensors are dropped from the
/// working map once written, so peak memory is bounded by the shard rather than the tier.
///
/// Shard files from a previous emit are cleared first, so re-emitting at a different
/// `max_shard_bytes` cannot leave a stale, differently-named set beside the new one (the shard count
/// is a function of the budget, so the filenames change with it).
///
/// Returns the shard paths in order (whether or not `on_shard` deleted them).
pub fn convert_krea_realtime_tier_sharded(
    src: impl AsRef<Path>,
    out_dir: impl AsRef<Path>,
    quantize: Option<(i32, i32)>,
    cfg: &KreaRealtimeConfig,
    max_shard_bytes: u64,
    on_shard: &mut dyn FnMut(&Path) -> Result<()>,
) -> Result<Vec<PathBuf>> {
    let out_dir = out_dir.as_ref();
    let src = src.as_ref();

    let map = read_native_map(src)?;
    let sanitized = sanitize_krea_realtime_transformer(map)?;
    let mut dit = match quantize {
        Some((bits, group_size)) => {
            quantize_krea_realtime_transformer(sanitized, bits, group_size)?
        }
        None => sanitized,
    };

    let mut cfg = cfg.clone();
    cfg.wan.quantization = quantize.map(|(bits, group_size)| WanQuant { bits, group_size });
    verify_transformer_tensors(&dit, &cfg.wan).map_err(|e| {
        Error::Msg(format!(
            "krea-realtime: refusing to write a tier snapshot whose transformer does not match the \
             config it would ship with (source {}): {e}",
            src.display()
        ))
    })?;

    let shard_dir = out_dir.join(TRANSFORMER_DIR);
    std::fs::create_dir_all(&shard_dir)?;
    clear_stale_shards(&shard_dir)?;

    // Write config.json first: a consumer that sees shards appearing has the geometry already.
    let text = serde_json::to_string_pretty(&cfg.to_json())
        .map_err(|e| Error::Msg(format!("krea-realtime: serialize config.json: {e}")))?;
    std::fs::write(out_dir.join("config.json"), text)?;

    let plan = plan_shards(&dit, max_shard_bytes);
    let total = plan.len();
    let mut written = Vec::with_capacity(total);
    for (i, names) in plan.into_iter().enumerate() {
        let path = shard_dir.join(format!("dit-{:05}-of-{:05}.safetensors", i + 1, total));
        // Move this shard's tensors out of the working map so they are released after the write.
        let shard: HashMap<String, Array> = names
            .into_iter()
            .map(|n| {
                let a = dit.remove(&n).expect("planned key is in the map");
                (n, a)
            })
            .collect();
        save_map(&path, &shard)?;
        drop(shard);
        on_shard(&path)?;
        written.push(path);
    }

    Ok(written)
}

/// Read either on-disk transformer layout into an owned key→`Array` map: a **directory** (the sharded
/// `transformer/` layout — merged via its safetensors index) or a single **file** (the `model.`-
/// prefixed single-file layout). MLX arrays are ref-counted, so the clones are handle copies.
fn read_native_map(src: &Path) -> Result<HashMap<String, Array>> {
    let w = if src.is_dir() {
        Weights::from_dir(src)?
    } else if src.is_file() {
        Weights::from_file(src)?
    } else {
        return Err(Error::Msg(format!(
            "krea-realtime: transformer source not found: {}",
            src.display()
        )));
    };
    Ok(w.keys()
        .map(|k| {
            (
                k.to_string(),
                w.require(k).expect("key from keys()").clone(),
            )
        })
        .collect())
}

/// Convert a native Krea Realtime 14B transformer (single-file `*.safetensors` **or** a sharded
/// `transformer/` directory) into a sanitized, bf16 `out_file` that
/// [`mlx_gen_wan::WanTransformer::from_weights`] loads directly. The TE / VAE / tokenizer are stock Wan
/// and are provisioned separately (Krea Realtime ships transformer-only) — this converts the DiT only,
/// mirroring the transformer step of [`mlx_gen_wan::convert::convert_ti2v_5b`].
pub fn convert_krea_realtime_transformer(
    src: impl AsRef<Path>,
    out_file: impl AsRef<Path>,
) -> Result<()> {
    let map = read_native_map(src.as_ref())?;
    let sanitized = sanitize_krea_realtime_transformer(map)?;
    let out = out_file.as_ref();
    if let Some(parent) = out.parent() {
        std::fs::create_dir_all(parent)?;
    }
    save_map(out, &sanitized)?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn strips_only_the_outer_model_prefix() {
        assert_eq!(
            strip_model_prefix("model.blocks.0.self_attn.q.weight"),
            "blocks.0.self_attn.q.weight"
        );
        // Sharded layout (no prefix) is untouched.
        assert_eq!(
            strip_model_prefix("blocks.0.self_attn.q.weight"),
            "blocks.0.self_attn.q.weight"
        );
        // Only the OUTER `model.` is removed — an inner segment is never a false match.
        assert_eq!(
            strip_model_prefix("blocks.0.model.weight"),
            "blocks.0.model.weight"
        );
    }

    #[test]
    fn transformer_dtype_is_bf16() {
        assert_eq!(TRANSFORMER_DTYPE, Dtype::Bfloat16);
    }

    /// The bf16 cast must skip **integer** tensors (sc-15203): a pre-quantized tier's `{base}.weight`
    /// holds u32 packed codes, and casting those to bf16 would reinterpret each u32 as one float —
    /// silently destroying 8 (Q4) / 4 (Q8) packed weights per element instead of failing. The float
    /// companions (`.scales`/`.biases`) still cast. Discriminating: it asserts the u32 dtype AND the
    /// exact code values survive, so a cast that happened to preserve the shape would still fail.
    #[test]
    fn cast_skips_the_u32_packed_codes_and_still_casts_the_float_companions() {
        let codes: Vec<u32> = vec![0xDEAD_BEEF, 0x0123_4567, 1, u32::MAX];
        let mut map = HashMap::from([
            (
                "blocks.0.self_attn.q.weight".to_string(),
                Array::from_slice(&codes, &[2, 2]),
            ),
            (
                "blocks.0.self_attn.q.scales".to_string(),
                Array::from_slice(&[0.5f32, 0.25, 0.125, 1.0], &[2, 2]),
            ),
        ]);
        cast_floats(&mut map, TRANSFORMER_DTYPE).unwrap();

        let w = &map["blocks.0.self_attn.q.weight"];
        assert_eq!(w.dtype(), Dtype::Uint32, "packed codes must stay u32");
        assert_eq!(w.shape(), &[2, 2]);
        assert_eq!(
            w.as_slice::<u32>(),
            codes.as_slice(),
            "the packed code words must be preserved bit-for-bit"
        );
        assert_eq!(
            map["blocks.0.self_attn.q.scales"].dtype(),
            Dtype::Bfloat16,
            "float companions still cast to the transformer dtype"
        );
    }

    /// The tier packer is the Wan packer: it produces the MLX affine triple for exactly the
    /// `_quantize_predicate` Linears and leaves everything else (dense biases, norms, the head) alone.
    #[test]
    fn tier_packer_packs_only_the_quantize_predicate() {
        let map = HashMap::from([
            (
                "blocks.0.self_attn.q.weight".to_string(),
                Array::zeros::<f32>(&[64, 64])
                    .unwrap()
                    .as_dtype(Dtype::Bfloat16)
                    .unwrap(),
            ),
            (
                "blocks.0.self_attn.q.bias".to_string(),
                Array::zeros::<f32>(&[64])
                    .unwrap()
                    .as_dtype(Dtype::Bfloat16)
                    .unwrap(),
            ),
            (
                "head.head.weight".to_string(),
                Array::zeros::<f32>(&[64, 64])
                    .unwrap()
                    .as_dtype(Dtype::Bfloat16)
                    .unwrap(),
            ),
        ]);
        let packed = quantize_krea_realtime_transformer(map, 4, 64).unwrap();

        // Packed: u32 codes `[64, 64·4/32 = 8]` + `[64, 64/64 = 1]` scales/biases.
        assert_eq!(packed["blocks.0.self_attn.q.weight"].dtype(), Dtype::Uint32);
        assert_eq!(packed["blocks.0.self_attn.q.weight"].shape(), &[64, 8]);
        assert_eq!(packed["blocks.0.self_attn.q.scales"].shape(), &[64, 1]);
        assert_eq!(packed["blocks.0.self_attn.q.biases"].shape(), &[64, 1]);
        // The Linear's dense bias is untouched.
        assert_eq!(packed["blocks.0.self_attn.q.bias"].shape(), &[64]);
        assert!(!packed.contains_key("blocks.0.self_attn.q.bias.scales"));
        // The head is outside the predicate → still a dense bf16 `[64, 64]`.
        assert_eq!(packed["head.head.weight"].dtype(), Dtype::Bfloat16);
        assert_eq!(packed["head.head.weight"].shape(), &[64, 64]);
        assert!(!packed.contains_key("head.head.scales"));

        // An unsupported width is rejected up front (not silently routed through MLX quantize).
        let err = quantize_krea_realtime_transformer(HashMap::new(), 6, 64)
            .expect_err("Q6 is not a shipped tier");
        assert!(err.to_string().contains("Q4 / Q8 / bf16"), "got: {err}");
    }
}
