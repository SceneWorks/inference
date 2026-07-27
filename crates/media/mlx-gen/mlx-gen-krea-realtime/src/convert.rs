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
pub fn convert_krea_realtime_tier(
    src: impl AsRef<Path>,
    out_dir: impl AsRef<Path>,
    quantize: Option<(i32, i32)>,
) -> Result<PathBuf> {
    let out_dir = out_dir.as_ref();
    std::fs::create_dir_all(out_dir)?;

    let map = read_native_map(src.as_ref())?;
    let sanitized = sanitize_krea_realtime_transformer(map)?;
    let dit = match quantize {
        Some((bits, group_size)) => {
            quantize_krea_realtime_transformer(sanitized, bits, group_size)?
        }
        None => sanitized,
    };
    save_map(&out_dir.join(DIT_FILE), &dit)?;

    let mut cfg = KreaRealtimeConfig::krea_realtime_14b();
    cfg.wan.quantization = quantize.map(|(bits, group_size)| WanQuant { bits, group_size });
    let text = serde_json::to_string_pretty(&cfg.to_json())
        .map_err(|e| Error::Msg(format!("krea-realtime: serialize config.json: {e}")))?;
    std::fs::write(out_dir.join("config.json"), text)?;

    Ok(out_dir.to_path_buf())
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
