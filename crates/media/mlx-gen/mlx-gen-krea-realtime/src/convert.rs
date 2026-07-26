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
//! [`normalize_krea_keys`] strips a leading `model.` from every key so both layouts collapse to the
//! plain Wan native names, then the shared Wan sanitizer maps them onto the internal
//! [`mlx_gen_wan::WanTransformer`] key layout. The reference runs the DiT in **bf16**, so — like every
//! Wan converter ([`mlx_gen_wan::convert::convert_ti2v_5b`] etc.) — we cast the (F16) checkpoint to
//! [`TRANSFORMER_DTYPE`] (`Bfloat16`) on the way out.
//!
//! This is the **non-gated** S2 converter: it is validated against the S1 tensor inventory with
//! synthesized fixtures (`tests/`) — never the real 28.58 GB checkpoint. Real-weight byte parity + the
//! MLX rehost are the gated S2 remainder tracked on sc-8435.

use std::collections::HashMap;
use std::path::Path;

use mlx_gen::quant::save_map;
use mlx_gen::weights::Weights;
use mlx_gen::{Error, Result};
use mlx_gen_wan::convert::sanitize_wan_transformer;
use mlx_rs::{Array, Dtype};

/// The prefix the **single-file** Krea Realtime checkpoint layout adds to every tensor name. The
/// sharded `transformer/` layout omits it, so stripping it (when present) normalizes both to the plain
/// Wan native names.
pub const KREA_MODEL_PREFIX: &str = "model.";

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

/// Cast every tensor in `map` to `dtype` in place (skipping any already at `dtype`). Mirrors the
/// private `cast_map` the Wan converters use for their F16/F32→bf16 transformer cast.
fn cast_all(map: &mut HashMap<String, Array>, dtype: Dtype) -> Result<()> {
    for v in map.values_mut() {
        if v.dtype() != dtype {
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
    cast_all(&mut sanitized, TRANSFORMER_DTYPE)?;
    Ok(sanitized)
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
}
