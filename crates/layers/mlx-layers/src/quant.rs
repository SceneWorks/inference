//! Group-wise affine Q4/Q8 quantization metadata, packed-weight loaders, and [`TokenEmbedding`]
//! primitives shared across MLX media and LLM engines (`SW-6955`).

use std::io::ErrorKind;
use std::path::{Component, Path};

use crate::quantization_cache::quantize_cached as quantize;
use mlx_rs::ops::dequantize;
use mlx_rs::{Array, Dtype};

use crate::adapters::AdaptableLinear;
use crate::{MlxWeightSource, Result};

/// A token-embedding table (`[vocab, hidden]`), dense or group-quantized (Q4/Q8) — the shared
/// implementation behind the FLUX.2 and Z-Image text encoders' `embed_tokens` (F-083). `forward` is a
/// row-gather (dequantizing the packed form on the fly), mirroring mlx's `QuantizedEmbedding`, and
/// returns **f32** (both encoders run f32 internally). FLUX.1's CLIP/T5 embedding intentionally
/// returns native bf16, so it keeps its own variant rather than using this.
pub enum TokenEmbedding {
    Dense(Array),
    Native(crate::NativeProjection, Dtype),
    Quantized {
        wq: Array,
        scales: Array,
        biases: Array,
        group_size: i32,
        bits: i32,
    },
}

impl TokenEmbedding {
    /// Gather rows for `ids` (`[b, s]`) → `[b, s, hidden]`, cast to f32.
    pub fn forward(&self, ids: &Array) -> Result<Array> {
        Ok(self.rows(ids)?.as_dtype(Dtype::Float32)?)
    }
    pub fn rows(&self, ids: &Array) -> Result<Array> {
        let out = match self {
            Self::Dense(w) => w.take_axis(ids, 0)?,
            Self::Native(p, dtype) => p.rows_as(ids, *dtype)?,
            Self::Quantized {
                wq,
                scales,
                biases,
                group_size,
                bits,
            } => {
                let pw = wq.take_axis(ids, 0)?;
                let sc = scales.take_axis(ids, 0)?;
                let bi = biases.take_axis(ids, 0)?;
                dequantize(&pw, &sc, &bi, *group_size, *bits)?
            }
        };
        Ok(out)
    }

    /// Build a quantized table directly from **already-packed** parts read off disk — the
    /// consume-side counterpart to [`quantize`](Self::quantize), for a *pre-quantized* checkpoint
    /// (the FLUX.2-dev path, sc-5917). No re-quantization happens; the on-disk `scales`/`biases`
    /// are used as-is. Mirrors [`AdaptableLinear::from_quantized_parts`] for the embedding case.
    ///
    /// [`AdaptableLinear::from_quantized_parts`]: crate::adapters::AdaptableLinear::from_quantized_parts
    pub fn from_quantized_parts(
        wq: Array,
        scales: Array,
        biases: Array,
        group_size: i32,
        bits: i32,
    ) -> Self {
        Self::Quantized {
            wq,
            scales,
            biases,
            group_size,
            bits,
        }
    }

    /// Quantize a dense table in place (group_size 64), the mlx-rs equivalent of
    /// `nn.Embedding.to_quantized`. No-op if already quantized. `cast_to_bf16` packs the **bf16**
    /// weight (FLUX.2's path, to byte-match the fork's bf16 `nn.quantize`); when false the weight is
    /// packed as-is (Z-Image's path). The two are equivalent for a bf16-native checkpoint — the flag
    /// preserves each caller's exact prior behaviour.
    pub fn quantize(&mut self, bits: i32, cast_to_bf16: bool) -> Result<()> {
        if let Self::Dense(w) = self {
            let (wq, scales, biases) = if cast_to_bf16 {
                quantize(&w.as_dtype(Dtype::Bfloat16)?, 64, bits)?
            } else {
                quantize(w, 64, bits)?
            };
            *self = Self::Quantized {
                wq,
                scales,
                biases,
                group_size: 64,
                bits,
            };
        }
        Ok(())
    }

    pub fn is_quantized(&self) -> bool {
        matches!(self, Self::Quantized { .. })
    }

    /// Evaluate the retained embedding representation without evaluating any other model weight.
    /// Used by bounded load-time quantization walks to keep the dense-to-packed transient local to
    /// this table.
    pub fn materialize_weights(&self) -> Result<()> {
        match self {
            Self::Dense(weight) => mlx_rs::transforms::eval([weight])?,
            Self::Native(p, _) => p.materialize()?,
            Self::Quantized {
                wq, scales, biases, ..
            } => mlx_rs::transforms::eval([wq, scales, biases])?,
        }
        Ok(())
    }
}

/// MLX's default quantization group size; mflux relies on it (never overrides). Used by the real
/// quantization seams ([`AdaptableLinear::quantize`](crate::adapters::AdaptableLinear::quantize) and
/// the Kolors ChatGLM3 quantizer).
pub const DEFAULT_GROUP_SIZE: i32 = 64;

/// Whether an `in_features`-wide Linear can be group-wise affine-quantized at `group_size`: MLX
/// `quantize` requires the contraction dim to be a multiple of the group, so the per-model quantize
/// cascades leave a misaligned projection (tiny embedders, odd patch projections) dense. The one
/// spelling of that predicate (sc-27320) — see
/// [`AdaptableLinear::quantize_if_aligned`](crate::adapters::AdaptableLinear::quantize_if_aligned).
pub fn is_quant_aligned(in_features: i32, group_size: i32) -> bool {
    in_features % group_size == 0
}

/// Read the packed Q4/Q8 tier declared by `<root>/<component>/config.json`.
///
/// A missing config, or a valid config with no `quantization` member, denotes a dense component.
/// Every other read/JSON/marker failure is an error: a present but damaged marker must never be
/// mistaken for dense weights and silently load-time quantized. `component` is a provider-owned,
/// non-empty relative path; absolute paths and `..` are rejected.
pub fn packed_quant_bits(root: &Path, component: &str) -> Result<Option<i32>> {
    let component_path = Path::new(component);
    if component_path.as_os_str().is_empty()
        || component_path
            .components()
            .any(|part| !matches!(part, Component::Normal(_)))
    {
        return Err(crate::Error::Msg(format!(
            "packed quant: component path must be a non-empty relative path without `..`, got \
             {component:?}"
        )));
    }

    packed_quant_bits_at(&root.join(component_path))
}

/// [`packed_quant_bits`] against a component directory that is already resolved — for a caller
/// holding the component path itself rather than a snapshot root plus a component name (the
/// offline pre-quantize converters, which are handed `…/transformer` directly).
pub fn packed_quant_bits_at(component_dir: &Path) -> Result<Option<i32>> {
    let config_path = component_dir.join("config.json");
    let bytes = match gen_core::__model_input_reader!(
        std::fs::read(&config_path),
        "crates/media/mlx-gen/src/quant.rs::packed_quant_bits_at::std::fs::read@3186",
        ArtifactInput,
        &config_path,
        |__model_input_path| std::fs::read(__model_input_path)
    ) {
        Ok(bytes) => bytes,
        Err(err) if err.kind() == ErrorKind::NotFound => {
            return packed_bits_from_headers(component_dir)
        }
        Err(err) => {
            return Err(crate::Error::Msg(format!(
                "packed quant: read {}: {err}",
                config_path.display()
            )))
        }
    };
    let config: serde_json::Value = serde_json::from_slice(&bytes).map_err(|err| {
        crate::Error::Msg(format!(
            "packed quant: parse {}: {err}",
            config_path.display()
        ))
    })?;
    let Some(marker) = config.get("quantization") else {
        // No config marker: the packing is read from the safetensors HEADERS (a packed snapshot
        // published without the marker — e.g. older SceneWorks SDXL-family q8 revisions — is still
        // packed, and must never be mistaken for dense weights and load-time quantized).
        return packed_bits_from_headers(component_dir);
    };
    let marker = marker.as_object().ok_or_else(|| {
        crate::Error::Msg(format!(
            "packed quant: {} `quantization` must be an object",
            config_path.display()
        ))
    })?;
    let bits_i64 = marker
        .get("bits")
        .and_then(serde_json::Value::as_i64)
        .ok_or_else(|| {
            crate::Error::Msg(format!(
                "packed quant: {} `quantization.bits` must be an integer",
                config_path.display()
            ))
        })?;
    let bits = i32::try_from(bits_i64).map_err(|_| {
        crate::Error::Msg(format!(
            "packed quant: {} `quantization.bits` is out of range: {bits_i64}",
            config_path.display()
        ))
    })?;
    if !matches!(bits, 4 | 8) {
        return Err(crate::Error::Msg(format!(
            "packed quant: {} declares unsupported `quantization.bits` {bits}; expected 4 or 8",
            config_path.display()
        )));
    }
    Ok(Some(bits))
}

/// The group size of a packed component that carries no config marker: the SceneWorks converters
/// pack at 64 (`quantize_map`), and [`packed_bits_from_headers`] only accepts a pack whose shapes
/// are exactly a Q4 or Q8 pack at that group size.
pub const HEADER_PACKED_GROUP_SIZE: i32 = 64;

/// The packed bit-width of a component read from its safetensors HEADERS alone (no payload): every
/// `X.weight` with a sibling `X.scales` must be a 2-D u32 code tensor `[out, in·bits/32]` beside
/// `[out, in/64]` scales, with one bit-width (4 or 8) across the component. `None` when no tensor is
/// packed (dense; a non-u32 `.weight` beside a `.scales` is not a pack). Any inconsistent or
/// non-Q4/Q8 u32 pack is an error, never "dense".
pub fn packed_bits_from_headers(component_dir: &Path) -> Result<Option<i32>> {
    let Ok(entries) = std::fs::read_dir(component_dir) else {
        return Ok(None);
    };
    let mut files: Vec<_> = entries
        .filter_map(|e| e.ok().map(|e| e.path()))
        .filter(|p| p.extension().and_then(|e| e.to_str()) == Some("safetensors"))
        .collect();
    files.sort();
    let mut headers = std::collections::BTreeMap::new();
    for file in files {
        for h in crate::gen_core::weightsmeta::safetensors_path_tensor_headers(&file)
            .map_err(|e| crate::Error::Msg(format!("packed quant: {}: {e}", file.display())))?
        {
            headers.insert(h.name.clone(), h);
        }
    }
    let mut bits: Option<i32> = None;
    for (name, scales) in &headers {
        let Some(base) = name.strip_suffix(".scales") else {
            continue;
        };
        let Some(weight) = headers.get(&format!("{base}.weight")) else {
            continue;
        };
        let bad = || {
            crate::Error::Msg(format!(
                "packed quant: {} {base}: weight {:?} {:?} / scales {:?} is not a Q4/Q8 pack at group \
                 size {HEADER_PACKED_GROUP_SIZE}",
                component_dir.display(),
                weight.dtype.as_str(),
                weight.shape,
                scales.shape
            ))
        };
        // Only u32 codes are a pack; a dense tensor that happens to have a `.scales` sibling is not.
        if weight.dtype.as_str() != "U32" {
            continue;
        }
        if weight.shape.len() != 2 || scales.shape.len() != 2 {
            return Err(bad());
        }
        let input = (scales.shape[1] as u64) * HEADER_PACKED_GROUP_SIZE as u64;
        let packed_bits = (weight.shape[1] as u64) * 32;
        if input == 0 || weight.shape[0] != scales.shape[0] || !packed_bits.is_multiple_of(input) {
            return Err(bad());
        }
        let b = i32::try_from(packed_bits / input).map_err(|_| bad())?;
        if !matches!(b, 4 | 8) || bits.is_some_and(|seen| seen != b) {
            return Err(bad());
        }
        bits = Some(b);
    }
    Ok(bits)
}

/// Read the affine quantization group size declared by a packed component's `config.json`.
///
/// Dense components return `None`. A packed marker must declare one of the group sizes implemented
/// by MLX (`32`, `64`, or `128`); rejecting any other value here keeps corrupt or hand-edited
/// manifests from failing later inside `quantized_matmul` with an opaque backend error.
pub fn packed_quant_group_size_at(component_dir: &Path) -> Result<Option<i32>> {
    let config_path = component_dir.join("config.json");
    let bytes = match gen_core::__model_input_reader!(
        std::fs::read(&config_path),
        "crates/media/mlx-gen/src/quant.rs::packed_quant_group_size_at::std::fs::read@8611",
        ArtifactInput,
        &config_path,
        |__model_input_path| std::fs::read(__model_input_path)
    ) {
        Ok(bytes) => bytes,
        Err(err) if err.kind() == ErrorKind::NotFound => return Ok(None),
        Err(err) => {
            return Err(crate::Error::Msg(format!(
                "packed quant: read {}: {err}",
                config_path.display()
            )))
        }
    };
    let config: serde_json::Value = serde_json::from_slice(&bytes).map_err(|err| {
        crate::Error::Msg(format!(
            "packed quant: parse {}: {err}",
            config_path.display()
        ))
    })?;
    let Some(marker) = config.get("quantization") else {
        // Packed per the headers but unmarked: the converter's group size.
        return Ok(packed_bits_from_headers(component_dir)?.map(|_| HEADER_PACKED_GROUP_SIZE));
    };
    let marker = marker.as_object().ok_or_else(|| {
        crate::Error::Msg(format!(
            "packed quant: {} `quantization` must be an object",
            config_path.display()
        ))
    })?;
    let group_size_i64 = marker
        .get("group_size")
        .and_then(serde_json::Value::as_i64)
        .ok_or_else(|| {
            crate::Error::Msg(format!(
                "packed quant: {} `quantization.group_size` must be an integer",
                config_path.display()
            ))
        })?;
    let group_size = i32::try_from(group_size_i64).map_err(|_| {
        crate::Error::Msg(format!(
            "packed quant: {} `quantization.group_size` is out of range: {group_size_i64}",
            config_path.display()
        ))
    })?;
    if !matches!(group_size, 32 | 64 | 128) {
        return Err(crate::Error::Msg(format!(
            "packed quant: {} declares unsupported `quantization.group_size` {group_size}; expected 32, 64, or 128",
            config_path.display()
        )));
    }
    Ok(Some(group_size))
}

/// Decide whether a requested Q4/Q8 tier must be produced by load-time quantization.
///
/// Packed turnkeys must match the request exactly; a mismatch is a hard error because provider
/// `quantize` methods intentionally no-op over packed weights. Direct provider struct loaders that
/// bypass the registry/load seam also bypass this guard by design and must pin their own source tier.
pub fn needs_load_time_quant(
    root: &Path,
    component: &str,
    requested_bits: i32,
    model_id: &str,
) -> Result<bool> {
    if !matches!(requested_bits, 4 | 8) {
        return Err(crate::Error::Msg(format!(
            "{model_id}: unsupported requested quantization tier Q{requested_bits}; expected Q4 or Q8"
        )));
    }
    match packed_quant_bits(root, component)? {
        Some(packed) if packed != requested_bits => Err(crate::Error::Msg(format!(
            "{model_id}: {component}/ is a pre-quantized Q{packed} turnkey but Q{requested_bits} \
             was requested; quantize is a no-op on packed weights so the request would silently \
             serve Q{packed}. Point at a Q{requested_bits} snapshot (or a dense one)."
        ))),
        Some(_) => Ok(false),
        None => Ok(true),
    }
}

/// Derive the quant bit-width from the packed shapes at group size `group_size`: `scales` is
/// `[out, in/gs]` ⇒ `in = scales.cols·gs`; the u32-packed `weight` is `[out, in·bits/32]` ⇒
/// `bits = wq.cols·32/in`. Exact for any group-aligned Q4/Q8 pack, so the bit-width need not be
/// carried in a side manifest.
///
/// F-011: returns `Result` and validates the dtype and exact geometry a corrupt/mis-converted
/// pre-quantized snapshot would otherwise mishandle: codes must be u32, scales floating point, rows
/// must agree, and packed columns must divide exactly rather than truncating into an apparently valid
/// Q4/Q8 width. The shared load seam for every Group-B packed snapshot feeds straight off external
/// `.safetensors`, so these tensors are untrusted.
pub fn packed_bits(wq: &Array, scales: &Array, group_size: i32) -> Result<i32> {
    packed_bits_from_metadata(
        wq.shape(),
        wq.dtype(),
        scales.shape(),
        scales.dtype(),
        group_size,
    )
}

/// The exact packed-load geometry validator without constructing or reading a tensor. Both the
/// executable loader and provider-owned header normalization use this seam; metadata supplies no
/// allocator/backing authority. Accepted layouts and refusals are those of [`packed_bits`].
pub fn packed_bits_from_metadata(
    wshape: &[i32],
    weight_dtype: Dtype,
    sshape: &[i32],
    scales_dtype: Dtype,
    group_size: i32,
) -> Result<i32> {
    if sshape.len() != 2 || wshape.len() != 2 {
        return Err(crate::Error::Msg(format!(
            "packed quant: scales and weight must be 2-D, got scales {:?} / weight {:?}",
            sshape, wshape
        )));
    }
    if weight_dtype != Dtype::Uint32 {
        return Err(crate::Error::Msg(format!(
            "packed quant: weight codes must be u32, got {:?}",
            weight_dtype
        )));
    }
    if !matches!(
        scales_dtype,
        Dtype::Float32 | Dtype::Float16 | Dtype::Bfloat16
    ) {
        return Err(crate::Error::Msg(format!(
            "packed quant: scales must be floating point, got {:?}",
            scales_dtype
        )));
    }
    if sshape[0] != wshape[0] {
        return Err(crate::Error::Msg(format!(
            "packed quant: scales and weight rows differ ({} != {})",
            sshape[0], wshape[0]
        )));
    }
    let in_dim = sshape[1].checked_mul(group_size).ok_or_else(|| {
        crate::Error::Msg(format!(
            "packed quant: input dimension overflow (scales cols {} × group_size {group_size})",
            sshape[1]
        ))
    })?;
    if in_dim == 0 {
        return Err(crate::Error::Msg(format!(
            "packed quant: zero input dim (scales cols {} × group_size {})",
            sshape[1], group_size
        )));
    }
    let packed_width = wshape[1].checked_mul(32).ok_or_else(|| {
        crate::Error::Msg(format!(
            "packed quant: packed width overflow (weight cols {} × 32)",
            wshape[1]
        ))
    })?;
    if packed_width % in_dim != 0 {
        return Err(crate::Error::Msg(format!(
            "packed quant: weight cols {} do not exactly encode scales cols {} × group_size {group_size}",
            wshape[1], sshape[1]
        )));
    }
    let bits = packed_width / in_dim;
    if !matches!(bits, 4 | 8) {
        // Name the assumed `group_size` (sc-15154). Every term here is derived FROM it, so a caller
        // that passes the wrong one gets an illegal width from a perfectly good artifact — Mage's q8
        // vision tower (packed at 64, read at 32) reported "bits 16" and was diagnosed as a bad
        // upload. A reader who can see the assumption can check it before re-hosting anything.
        return Err(crate::Error::Msg(format!(
            "packed quant: inferred bit-width {bits} ∉ {{4, 8}} \
             (weight cols {}, scales cols {} × group_size {group_size} ⇒ in_dim {in_dim}); either \
             the snapshot is corrupt/mis-converted, or it was packed at a different group_size than \
             the {group_size} this loader assumed",
            wshape[1], sshape[1]
        )));
    }
    Ok(bits)
}

/// Resolve native codec operands before family-specific casts or affine handling.
pub fn native_lin(
    w: &(impl MlxWeightSource + ?Sized),
    base: &str,
    bias: bool,
) -> Result<Option<AdaptableLinear>> {
    if !crate::NativeProjection::is_declared(w, base, &format!("{base}.weight")) {
        return Ok(None);
    }
    crate::NativeProjection::from_weights(w, base, bias)?
        .map(|native| bind_native_linear(w, base, native))
        .transpose()
}

fn bind_native_linear(
    w: &(impl MlxWeightSource + ?Sized),
    base: &str,
    native: crate::NativeProjection,
) -> Result<AdaptableLinear> {
    let mut layer = AdaptableLinear::from_native(native);
    for delta in w.native_projection_deltas(base) {
        layer.merge_dense_delta(delta)?;
    }
    Ok(layer)
}

/// Load `{base}` as an [`AdaptableLinear`] — **packed** (Q4/Q8) when `{base}.scales` is present (a
/// pre-quantized snapshot, bit-width inferred from the shapes at `group_size`), else **dense**. `bias`
/// additionally loads the dense `{base}.bias` (the quantization's own `{base}.biases` is distinct and
/// always loaded on the packed path). Auto-detection means one loader serves both a dense bf16 and a
/// pre-quantized snapshot with no `quantization` manifest to read. Shared by the provider crates'
/// packed loaders — each passes its own `group_size`.
pub fn lin<W: MlxWeightSource + ?Sized>(
    w: &W,
    base: &str,
    bias: bool,
    group_size: i32,
) -> Result<AdaptableLinear> {
    let bias = bias
        .then(|| w.require(&format!("{base}.bias")).cloned())
        .transpose()?;
    lin_with_bias(w, base, bias, group_size)
}

/// A projection whose architecture supplies the bias under a separate key (e.g. EVA q_bias).
pub fn lin_with_bias(
    w: &(impl MlxWeightSource + ?Sized),
    base: &str,
    bias: Option<Array>,
    group_size: i32,
) -> Result<AdaptableLinear> {
    if crate::NativeProjection::is_declared(w, base, &format!("{base}.weight")) {
        if let Some(mut native) = crate::NativeProjection::from_weights(w, base, false)? {
            if let Some(b) = bias {
                native.set_bias(b)?;
            }
            return bind_native_linear(w, base, native);
        }
    }
    if let Some(scales) = w.get(&format!("{base}.scales")) {
        let wq = w.require(&format!("{base}.weight"))?;
        let bits = packed_bits(wq, scales, group_size)?;
        return Ok(AdaptableLinear::from_quantized_parts(
            wq.clone(),
            scales.clone(),
            w.require(&format!("{base}.biases"))?.clone(),
            bias,
            group_size,
            bits,
        ));
    }
    Ok(AdaptableLinear::dense(
        w.require(&format!("{base}.weight"))?.clone(),
        bias,
    ))
}

/// Load `{base}` as a [`TokenEmbedding`] — packed when `{base}.scales` is present, else dense (the
/// embedding analogue of [`lin`]).
pub fn embedding<W: MlxWeightSource + ?Sized>(
    w: &W,
    base: &str,
    group_size: i32,
) -> Result<TokenEmbedding> {
    if let Some(native) = crate::NativeProjection::from_weights(w, base, false)? {
        return Ok(TokenEmbedding::Native(native, Dtype::Bfloat16));
    }
    if let Some(scales) = w.get(&format!("{base}.scales")) {
        let wq = w.require(&format!("{base}.weight"))?;
        let bits = packed_bits(wq, scales, group_size)?;
        return Ok(TokenEmbedding::from_quantized_parts(
            wq.clone(),
            scales.clone(),
            w.require(&format!("{base}.biases"))?.clone(),
            group_size,
            bits,
        ));
    }
    Ok(TokenEmbedding::Dense(
        w.require(&format!("{base}.weight"))?.clone(),
    ))
}

/// Embedding-table coefficients in an architecture's chosen dense dtype. Encoded tables keep their
/// source representation; only gathered rows cross the floating-point boundary.
pub fn embedding_compute(
    w: &(impl MlxWeightSource + ?Sized),
    base: &str,
    dtype: Dtype,
) -> Result<TokenEmbedding> {
    if let Some(p) = crate::NativeProjection::from_weights(w, base, false)? {
        return Ok(TokenEmbedding::Native(p, dtype));
    }
    Ok(TokenEmbedding::Dense(
        w.require(&format!("{base}.weight"))?.as_dtype(dtype)?,
    ))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn quant_alignment_and_packed_metadata_validation() {
        assert!(is_quant_aligned(128, DEFAULT_GROUP_SIZE));
        assert!(!is_quant_aligned(96, DEFAULT_GROUP_SIZE));
        assert!(is_quant_aligned(96, 32));

        assert_eq!(
            packed_bits_from_metadata(&[64, 8], Dtype::Uint32, &[64, 1], Dtype::Bfloat16, 64)
                .unwrap(),
            4
        );
        assert_eq!(
            packed_bits_from_metadata(&[64, 16], Dtype::Uint32, &[64, 1], Dtype::Float16, 64)
                .unwrap(),
            8
        );
        assert!(
            packed_bits_from_metadata(&[64, 8], Dtype::Float32, &[64, 1], Dtype::Bfloat16, 64)
                .is_err()
        );
        assert!(
            packed_bits_from_metadata(&[64, 12], Dtype::Uint32, &[64, 1], Dtype::Bfloat16, 64)
                .is_err()
        );
    }
}
