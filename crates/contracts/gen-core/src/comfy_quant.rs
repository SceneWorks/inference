//! ComfyUI **`.comfy_quant` per-layer quantization descriptors** and the **reference decode math**
//! of the formats they name (epic 20398, sc-20385).
//!
//! Tensor-free, backend-neutral. Every backend reader validates descriptors through
//! [`parse_comfy_quant_descriptor`] and prices / decodes through the geometry and element decoders
//! here; the codec implementations in `mlx-gen` / `candle-gen` are thin wrappers (MLX decodes fp8 on
//! device with the same bit semantics; Candle's dense fallback decodes on the host through these
//! exact functions). The golden tests in this module are written against the format specifications
//! (OCP FP8 E4M3FN / E5M2, OCP MX E8M0 + block-32) and ComfyUI's own `comfy_kitchen` eager reference,
//! never against a measured device value.
//!
//! # The on-disk convention (ComfyUI `comfy/ops.py::MixedPrecisionOps`, `comfy_kitchen` ≥ 0.2)
//!
//! A quantized `Linear` at state-dict prefix `{layer}.` stores:
//!
//! * `{layer}.weight` — the quantized values, dtype = the format's storage dtype;
//! * `{layer}.comfy_quant` — a rank-1 `U8` tensor holding UTF-8 JSON:
//!   `{"format": "<name>", "full_precision_matrix_mult": true?, ...}` (int8 files also carry
//!   `"per_row": true`);
//! * `{layer}.weight_scale` — the scale companion (shape/dtype per format below);
//! * `{layer}.input_scale` — optional `F32` scalar activation scale (fp8 formats only; consumed
//!   but irrelevant to weight decode).
//!
//! | `format`          | storage            | `weight_scale`                                        | decode                                       |
//! |-------------------|--------------------|-------------------------------------------------------|----------------------------------------------|
//! | `float8_e4m3fn`   | `F8_E4M3` `[out,in]` | `F32` scalar (`[]` or `[1]`)                          | `fp8_e4m3fn_to_f32(v) · scale`               |
//! | `float8_e5m2`     | `F8_E5M2` `[out,in]` | `F32` scalar                                          | `fp8_e5m2_to_f32(v) · scale`                 |
//! | `mxfp8`           | `F8_E4M3` `[⌈out/32⌉·32, ⌈in/32⌉·32]` | `F8_E8M0`/`U8` `[⌈rows/128⌉·128, ⌈(cols/32)/4⌉·4]` in cuBLAS 128×4 swizzle | per 32-block `e4m3 · 2^(e8m0−127)`, unpadded |
//! | `int8_tensorwise` | `I8` `[out,in]`    | `F32` `[out]` / `[out,1]` (scalar when `out == 1`)    | `code · scale[row]` (the MLX/Candle arm)     |
//! | `nvfp4`           | `U8` `[out, in/2]` (E2M1 nibbles, **even element in the high nibble**) | `F8_E4M3` `[⌈out/128⌉·128, ⌈(in/16)/4⌉·4]` in the same cuBLAS 128×4 swizzle, **plus** a scalar `F32` `{layer}.weight_scale_2` | per 16-block `e2m1 · e4m3(block) · global`, unpadded |
//!
//! # Header-declared quantization (sc-20641)
//!
//! Not every ComfyUI-lineage checkpoint carries per-layer `.comfy_quant` tensors. The NVFP4
//! converters write a **file-level** `__metadata__._quantization_metadata` JSON object instead —
//! `{"format_version": "1.0", "layers": {"<layer>": {"format": "nvfp4"}, …}}` — whose layer names
//! are often *relative* to the state-dict prefix the tensors carry (`blocks.0.attn.wq` for
//! `model.diffusion_model.blocks.0.attn.wq.weight`). [`parse_quantization_metadata`] validates that
//! object into a table of [`PartialComfyQuantDescriptor`], and the plan compiler resolves the prefix
//! exactly once and fails closed when it is not unique.
//!
//! ## The two routes are NOT interchangeable (sc-20651)
//!
//! An earlier revision of this module claimed both routes "carry the same per-layer objects". Real
//! ComfyUI output disproves it: `kreamania_variant1.safetensors` declares each of its 264 int8
//! projections **twice** — the per-tensor blob as `{"format": "int8_tensorwise", "per_row": true}`
//! and the file-level table as a bare `{"format": "int8_tensorwise"}`. Parsing the file-level entry
//! under the standalone rules refused the whole checkpoint on `Int8NotPerRow`, even though the
//! authoritative blob right next to the weight said `per_row: true`.
//!
//! The precedence rule the compiler applies, and the reason this module has two parse entry points:
//!
//! * the per-tensor `.comfy_quant` blob is **authoritative** — it sits with the weight it describes;
//! * a file-level entry for the same layer is **corroborating**: a strict subset of the blob's
//!   fields (keys it does not declare) must not refuse, while a declared value that *contradicts*
//!   the blob still refuses as `LogicalWeightPlanError::DescriptorConflict`;
//! * a layer the file-level table declares **alone** must still stand on its own, and is completed
//!   through [`PartialComfyQuantDescriptor::into_complete`] with the same refusals as before.

use std::collections::BTreeMap;
use std::fmt;

/// Block length of one MXFP8 shared exponent (OCP MX spec; `comfy_kitchen.MXFP8_BLOCK_SIZE`).
pub const MXFP8_BLOCK: usize = 32;
/// ComfyUI pads MXFP8 storage to multiples of 32 on both axes (`TensorCoreMXFP8Layout.get_padded_shape`).
pub const MXFP8_PAD: usize = 32;
/// cuBLAS block-scale swizzle tile: 128 rows × 4 scale columns (`comfy_kitchen.float_utils.to_blocked`).
pub const MXFP8_SCALE_ROW_TILE: usize = 128;
pub const MXFP8_SCALE_COL_TILE: usize = 4;

/// Block length of one NVFP4 FP8 micro-scale (`comfy.quant_ops.QUANT_ALGOS["nvfp4"].group_size`,
/// `comfy.float.stochastic_round_quantize_nvfp4_block`'s `block_size`).
pub const NVFP4_BLOCK: usize = 16;
/// ComfyUI pads NVFP4 storage to multiples of 16 on both axes
/// (`TensorCoreNVFP4Layout.get_padded_shape` / `pad_16x`).
pub const NVFP4_PAD: usize = 16;

/// The `.comfy_quant` `format` names this workspace can plan.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum ComfyQuantFormat {
    /// `int8_tensorwise` with `per_row: true` — int8 codes + per-output-row f32 scale.
    Int8TensorwisePerRow,
    /// `float8_e4m3fn` — fp8 E4M3FN values + one per-tensor f32 scale.
    Float8E4M3Fn,
    /// `float8_e5m2` — fp8 E5M2 values + one per-tensor f32 scale.
    Float8E5M2,
    /// `mxfp8` — fp8 E4M3FN values + E8M0 shared exponents per 32-element block along the last axis.
    Mxfp8,
    /// `nvfp4` — E2M1 nibbles (two per `U8` byte) + one FP8 E4M3 micro-scale per 16-element block
    /// **and** one `F32` per-tensor `weight_scale_2` (the two-level NVFP4 scaling).
    Nvfp4,
}

impl ComfyQuantFormat {
    /// The exact `format` string ComfyUI writes.
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Int8TensorwisePerRow => "int8_tensorwise",
            Self::Float8E4M3Fn => "float8_e4m3fn",
            Self::Float8E5M2 => "float8_e5m2",
            Self::Mxfp8 => "mxfp8",
            Self::Nvfp4 => "nvfp4",
        }
    }

    pub fn parse(name: &str) -> Option<Self> {
        Some(match name {
            "int8_tensorwise" => Self::Int8TensorwisePerRow,
            "float8_e4m3fn" => Self::Float8E4M3Fn,
            "float8_e5m2" => Self::Float8E5M2,
            "mxfp8" => Self::Mxfp8,
            "nvfp4" => Self::Nvfp4,
            _ => return None,
        })
    }
}

impl fmt::Display for ComfyQuantFormat {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.as_str())
    }
}

/// A validated `.comfy_quant` descriptor.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct ComfyQuantDescriptor {
    pub format: ComfyQuantFormat,
    /// ComfyUI's per-layer "dequantize and multiply in full precision" flag: the layer keeps its
    /// packed storage but never takes a quantized matmul. A codec plan treats it as "dense fallback
    /// residency, even where a native path exists".
    pub full_precision_matrix_mult: bool,
}

/// Why a `.comfy_quant` blob is not a descriptor this workspace accepts. Every variant is a
/// per-layer fact; the plan compiler attaches the layer name.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum ComfyQuantDescriptorError {
    NotUtf8,
    NotJson {
        detail: String,
    },
    NotAnObject,
    MissingFormat,
    /// A `format` this workspace has no codec for (a future/unknown ComfyUI format fails closed,
    /// never best-effort).
    UnsupportedFormat {
        format: String,
    },
    /// `convrot` present: the rotated INT8 convention belongs to the Candle ConvRot loader, not the
    /// plain per-row arm, and must not be dequantized as if unrotated.
    ConvRot,
    /// `int8_tensorwise` without `per_row: true` (the only int8 layout either backend decodes).
    Int8NotPerRow,
    /// A flag that must be a JSON boolean is not one.
    NonBooleanField {
        field: &'static str,
    },
    /// A key this workspace does not model. ComfyUI writes exactly `format`,
    /// `full_precision_matrix_mult`, and (int8) `per_row`; anything else could redefine the layout
    /// (`group_size`, a future `layout`) and is refused rather than ignored.
    UnknownField {
        field: String,
    },
}

impl fmt::Display for ComfyQuantDescriptorError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::NotUtf8 => write!(f, "descriptor bytes are not UTF-8"),
            Self::NotJson { detail } => write!(f, "descriptor is not valid JSON: {detail}"),
            Self::NotAnObject => write!(f, "descriptor JSON is not an object"),
            Self::MissingFormat => write!(f, "descriptor has no string `format`"),
            Self::UnsupportedFormat { format } => write!(
                f,
                "descriptor format {format:?} has no registered codec on this workspace (known: \
                 int8_tensorwise, float8_e4m3fn, float8_e5m2, mxfp8, nvfp4)"
            ),
            Self::ConvRot => write!(
                f,
                "descriptor carries `convrot`; rotated int8 checkpoints are not the plain per-row \
                 format"
            ),
            Self::Int8NotPerRow => write!(
                f,
                "descriptor format `int8_tensorwise` must declare `per_row: true`"
            ),
            Self::NonBooleanField { field } => {
                write!(f, "descriptor field `{field}` must be a JSON boolean")
            }
            Self::UnknownField { field } => write!(
                f,
                "descriptor field {field:?} is not part of the ComfyUI `.comfy_quant` convention \
                 this workspace models; refusing rather than guessing the layout"
            ),
        }
    }
}

impl std::error::Error for ComfyQuantDescriptorError {}

/// Parse and validate one `.comfy_quant` payload (the rank-1 `U8` tensor's bytes).
pub fn parse_comfy_quant_descriptor(
    bytes: &[u8],
) -> Result<ComfyQuantDescriptor, ComfyQuantDescriptorError> {
    let text = std::str::from_utf8(bytes).map_err(|_| ComfyQuantDescriptorError::NotUtf8)?;
    let json: serde_json::Value =
        serde_json::from_str(text).map_err(|error| ComfyQuantDescriptorError::NotJson {
            detail: error.to_string(),
        })?;
    descriptor_from_json(&json)
}

/// One layer descriptor as *declared*, before the completeness rules that only a standalone
/// declaration has to satisfy — every modelled field kept as `Option` so "absent" and "declared
/// `false`" stay distinguishable (sc-20651).
///
/// # Why this type exists
///
/// The two declaration routes are **not** interchangeable on real ComfyUI output. The published
/// KreaMania int8 artifacts write the authoritative per-tensor `.comfy_quant` blob as
/// `{"format": "int8_tensorwise", "per_row": true}` while the file-level
/// `__metadata__._quantization_metadata` table writes the *same* layer as a bare
/// `{"format": "int8_tensorwise"}`. An earlier revision of this module's docs asserted both routes
/// "carry the same per-layer objects"; reading `kreamania_variant1.safetensors` (264 int8
/// projections, 264 blobs, 264 file-level entries) disproved it.
///
/// So the file-level table parses into *this* type — declared fields only, no completeness check —
/// and the plan compiler decides what a given layer's declaration set means:
///
/// * blob present ⇒ the blob is authoritative and this entry is corroborating. A strict **subset**
///   (fields this entry does not declare) must not refuse; a declared field whose **value**
///   conflicts still refuses (see `LogicalWeightPlanError::DescriptorConflict`).
/// * blob absent ⇒ this entry is the only declaration, so it must stand on its own and goes through
///   [`Self::into_complete`] — which is exactly the pre-sc-20651 behaviour for a metadata-only file
///   (the ComfyUI Kitchen NVFP4 converters' form, sc-20641).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct PartialComfyQuantDescriptor {
    pub format: ComfyQuantFormat,
    pub full_precision_matrix_mult: Option<bool>,
    /// Only meaningful for `int8_tensorwise`; `None` everywhere else (the key is refused as an
    /// unknown field on any other format).
    pub per_row: Option<bool>,
}

impl PartialComfyQuantDescriptor {
    /// Apply the rules a **standalone** declaration must satisfy and produce the validated
    /// descriptor: today that is int8's `per_row: true`, the only int8 layout either backend
    /// decodes. Absent optional fields take their ComfyUI defaults.
    pub fn into_complete(self) -> Result<ComfyQuantDescriptor, ComfyQuantDescriptorError> {
        if self.format == ComfyQuantFormat::Int8TensorwisePerRow && self.per_row != Some(true) {
            return Err(ComfyQuantDescriptorError::Int8NotPerRow);
        }
        Ok(ComfyQuantDescriptor {
            format: self.format,
            full_precision_matrix_mult: self.full_precision_matrix_mult.unwrap_or(false),
        })
    }

    /// The field names on which this (corroborating) declaration **contradicts** an authoritative
    /// per-tensor descriptor, in declaration order. Presence-aware: a field this declaration omits
    /// is not a disagreement, only a declared value that differs is. Empty ⇒ the two declarations
    /// are consistent and the authoritative one stands.
    pub fn disagreement_with(&self, tensor: &ComfyQuantDescriptor) -> Vec<&'static str> {
        let mut disagreement = Vec::new();
        if self.format != tensor.format {
            disagreement.push("format");
        }
        if self
            .full_precision_matrix_mult
            .is_some_and(|declared| declared != tensor.full_precision_matrix_mult)
        {
            disagreement.push("full_precision_matrix_mult");
        }
        // `per_row` does not survive into `ComfyQuantDescriptor` — the int8 format variant *is* the
        // per-row layout — so a declared `per_row: false` on an int8 layer contradicts the
        // authoritative blob just as surely as a format mismatch would.
        if self.per_row == Some(false) && tensor.format == ComfyQuantFormat::Int8TensorwisePerRow {
            disagreement.push("per_row");
        }
        disagreement
    }
}

impl fmt::Display for PartialComfyQuantDescriptor {
    /// Renders only what the declaration actually carries, so a diagnostic never invents a default
    /// the producer never wrote.
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "`{}`", self.format)?;
        let mut declared = Vec::new();
        if let Some(flag) = self.full_precision_matrix_mult {
            declared.push(format!("full_precision_matrix_mult={flag}"));
        }
        if let Some(flag) = self.per_row {
            declared.push(format!("per_row={flag}"));
        }
        if declared.is_empty() {
            f.write_str(" (no other field declared)")
        } else {
            write!(f, " ({})", declared.join(", "))
        }
    }
}

/// The shared validation half of [`parse_comfy_quant_descriptor`]: one already-parsed JSON value
/// describing a single layer, validated as a **standalone** declaration.
///
/// The file-level `_quantization_metadata` form ([`parse_quantization_metadata`]) parses through
/// [`partial_descriptor_from_json`] instead, because a real ComfyUI file's two routes do *not*
/// carry the same per-layer objects — see [`PartialComfyQuantDescriptor`].
pub fn descriptor_from_json(
    json: &serde_json::Value,
) -> Result<ComfyQuantDescriptor, ComfyQuantDescriptorError> {
    partial_descriptor_from_json(json)?.into_complete()
}

/// Parse one layer descriptor object into its declared fields, applying every rule that does not
/// depend on the declaration standing alone: JSON shape, `format` support, the ConvRot refusal,
/// field types, and the unknown-key refusal.
pub fn partial_descriptor_from_json(
    json: &serde_json::Value,
) -> Result<PartialComfyQuantDescriptor, ComfyQuantDescriptorError> {
    let object = json
        .as_object()
        .ok_or(ComfyQuantDescriptorError::NotAnObject)?;
    let format = object
        .get("format")
        .and_then(serde_json::Value::as_str)
        .ok_or(ComfyQuantDescriptorError::MissingFormat)?;
    if object.contains_key("convrot") || object.contains_key("convrot_groupsize") {
        return Err(ComfyQuantDescriptorError::ConvRot);
    }
    let format = ComfyQuantFormat::parse(format).ok_or_else(|| {
        ComfyQuantDescriptorError::UnsupportedFormat {
            format: format.to_owned(),
        }
    })?;
    let full_precision_matrix_mult = match object.get("full_precision_matrix_mult") {
        None => None,
        Some(serde_json::Value::Bool(flag)) => Some(*flag),
        Some(_) => {
            return Err(ComfyQuantDescriptorError::NonBooleanField {
                field: "full_precision_matrix_mult",
            })
        }
    };
    let per_row = match object.get("per_row") {
        None => None,
        Some(serde_json::Value::Bool(flag)) => Some(*flag),
        Some(_) => return Err(ComfyQuantDescriptorError::NonBooleanField { field: "per_row" }),
    };
    for key in object.keys() {
        let known = matches!(key.as_str(), "format" | "full_precision_matrix_mult")
            || (key == "per_row" && format == ComfyQuantFormat::Int8TensorwisePerRow);
        if !known {
            return Err(ComfyQuantDescriptorError::UnknownField { field: key.clone() });
        }
    }
    Ok(PartialComfyQuantDescriptor {
        format,
        full_precision_matrix_mult,
        per_row,
    })
}

// =================================================================================================
// File-level `__metadata__._quantization_metadata` descriptors (sc-20641).
// =================================================================================================

/// Why a `__metadata__._quantization_metadata` blob is not a descriptor table this workspace
/// accepts. Layer-level defects carry the layer name the fault appears on.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum QuantizationMetadataError {
    NotJson {
        detail: String,
    },
    NotAnObject,
    /// `format_version` present but not the `"1.0"` this workspace models. An absent key is
    /// accepted as v1 (sc-21482): the real pinned `Comfy-Org/Krea-2` NVFP4 export writes no
    /// version key at all.
    FormatVersion {
        found: Option<String>,
    },
    /// `layers` absent or not a JSON object.
    Layers,
    /// The table declares no layer at all — an empty declaration is a producer defect, not a
    /// "nothing is quantized" statement (an unquantized file carries no metadata key).
    NoLayers,
    /// One layer's descriptor object is malformed or names a format without a codec.
    Layer {
        layer: String,
        defect: ComfyQuantDescriptorError,
    },
}

impl fmt::Display for QuantizationMetadataError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::NotJson { detail } => write!(
                f,
                "`__metadata__._quantization_metadata` is not valid JSON: {detail}"
            ),
            Self::NotAnObject => write!(
                f,
                "`__metadata__._quantization_metadata` is not a JSON object"
            ),
            Self::FormatVersion { found } => write!(
                f,
                "`_quantization_metadata.format_version` must be the string \"1.0\", got {found:?}"
            ),
            Self::Layers => write!(
                f,
                "`_quantization_metadata.layers` must be a JSON object of layer → descriptor"
            ),
            Self::NoLayers => write!(
                f,
                "`_quantization_metadata.layers` declares no layer; refusing a quantization \
                 declaration that governs nothing"
            ),
            Self::Layer { layer, defect } => write!(f, "layer {layer:?}: {defect}"),
        }
    }
}

impl std::error::Error for QuantizationMetadataError {}

/// Parse a file-level `__metadata__._quantization_metadata` payload into the per-layer descriptor
/// table the plan compiler consumes, keyed by the metadata's own layer names (which may be relative
/// to the tensors' state-dict prefix — the compiler resolves that).
///
/// Every layer object goes through [`partial_descriptor_from_json`], so an `nvfp4` layer that also
/// declares an unmodelled key is refused here exactly as a `.comfy_quant` tensor would be.
///
/// The table's entries stay **partial** (sc-20651): a real ComfyUI int8 file declares the layer here
/// as a bare `{"format": "int8_tensorwise"}` while its authoritative per-tensor blob carries
/// `per_row: true`, so the completeness rules can only be applied once the compiler knows whether a
/// blob corroborates the entry. See [`PartialComfyQuantDescriptor`]; a layer this table declares
/// *alone* is completed with [`PartialComfyQuantDescriptor::into_complete`] by the compiler and
/// refuses exactly as before.
pub fn parse_quantization_metadata(
    payload: &str,
) -> Result<BTreeMap<String, PartialComfyQuantDescriptor>, QuantizationMetadataError> {
    let json: serde_json::Value =
        serde_json::from_str(payload).map_err(|error| QuantizationMetadataError::NotJson {
            detail: error.to_string(),
        })?;
    let object = json
        .as_object()
        .ok_or(QuantizationMetadataError::NotAnObject)?;
    // `format_version` is refused when PRESENT and not the "1.0" this workspace models. An absent
    // key is accepted as v1: the real pinned `Comfy-Org/Krea-2` NVFP4 export (sc-21482) writes
    // `{"layers": {…}}` with no version key at all, and ComfyUI's own reader
    // (`comfy.sd.load_diffusion_model_state_dict`) never consults one — so "absent" is the
    // format's ground truth, not a producer defect.
    if let Some(version) = object.get("format_version") {
        if version.as_str() != Some("1.0") {
            return Err(QuantizationMetadataError::FormatVersion {
                found: Some(
                    version
                        .as_str()
                        .map(str::to_owned)
                        .unwrap_or_else(|| version.to_string()),
                ),
            });
        }
    }
    let layers = object
        .get("layers")
        .and_then(|v| v.as_object())
        .ok_or(QuantizationMetadataError::Layers)?;
    if layers.is_empty() {
        return Err(QuantizationMetadataError::NoLayers);
    }
    let mut table = BTreeMap::new();
    for (layer, value) in layers {
        let descriptor = partial_descriptor_from_json(value).map_err(|defect| {
            QuantizationMetadataError::Layer {
                layer: layer.clone(),
                defect,
            }
        })?;
        table.insert(layer.clone(), descriptor);
    }
    Ok(table)
}

// =================================================================================================
// Element decoders — the reference semantics, in plain f32.
// =================================================================================================

/// Decode one OCP FP8 **E4M3FN** byte (`torch.float8_e4m3fn`): sign · 4 exponent bits (bias 7) ·
/// 3 mantissa bits; no infinities; the all-ones exponent+mantissa patterns (`0x7F`, `0xFF`) are NaN;
/// exponent 0 is subnormal (`mantissa/8 · 2^-6`). Range ±448.
pub fn fp8_e4m3fn_to_f32(byte: u8) -> f32 {
    let sign = if byte & 0x80 != 0 { -1.0_f32 } else { 1.0_f32 };
    let exponent = (byte >> 3) & 0x0F;
    let mantissa = byte & 0x07;
    if exponent == 0x0F && mantissa == 0x07 {
        return f32::NAN;
    }
    let magnitude = if exponent == 0 {
        (mantissa as f32 / 8.0) * 2f32.powi(-6)
    } else {
        (1.0 + mantissa as f32 / 8.0) * 2f32.powi(exponent as i32 - 7)
    };
    sign * magnitude
}

/// Decode one OCP FP8 **E5M2** byte (`torch.float8_e5m2`): sign · 5 exponent bits (bias 15) ·
/// 2 mantissa bits; IEEE-like — exponent 31 is ±inf (mantissa 0) or NaN; exponent 0 is subnormal
/// (`mantissa/4 · 2^-14`). Range ±57344. Bit-identical to the top byte of an IEEE binary16.
pub fn fp8_e5m2_to_f32(byte: u8) -> f32 {
    let sign = if byte & 0x80 != 0 { -1.0_f32 } else { 1.0_f32 };
    let exponent = (byte >> 2) & 0x1F;
    let mantissa = byte & 0x03;
    if exponent == 0x1F {
        return if mantissa == 0 {
            sign * f32::INFINITY
        } else {
            f32::NAN
        };
    }
    let magnitude = if exponent == 0 {
        (mantissa as f32 / 4.0) * 2f32.powi(-14)
    } else {
        (1.0 + mantissa as f32 / 4.0) * 2f32.powi(exponent as i32 - 15)
    };
    sign * magnitude
}

/// Decode one OCP MX **E8M0** shared-exponent byte (`torch.float8_e8m0fnu`): `2^(byte − 127)` for
/// `0..=254` (so `0x00` is `2^-127`, a representable f32 subnormal, and `0x7F` is `1.0`); `0xFF` is
/// NaN. No sign, no mantissa, no zero.
///
/// Note: `comfy_kitchen`'s *eager* dequantizer maps `0x00` to `0.0` instead of `2^-127`; its
/// quantizer only ever emits `0x00` for an all-zero block, where both readings give `0`, so the two
/// agree on every block a ComfyUI file can contain. The spec value is used here.
pub fn e8m0_to_f32(byte: u8) -> f32 {
    if byte == 0xFF {
        return f32::NAN;
    }
    if byte == 0 {
        return f32::from_bits(0x0040_0000); // 2^-127, subnormal
    }
    f32::from_bits((byte as u32) << 23)
}

/// Decode a whole E4M3FN payload through a per-tensor scalar scale: `out[i] = decode(v[i]) · scale`.
pub fn decode_fp8_e4m3fn_scalar(values: &[u8], scale: f32, out: &mut Vec<f32>) {
    out.clear();
    out.extend(values.iter().map(|&byte| fp8_e4m3fn_to_f32(byte) * scale));
}

/// Decode a whole E5M2 payload through a per-tensor scalar scale.
pub fn decode_fp8_e5m2_scalar(values: &[u8], scale: f32, out: &mut Vec<f32>) {
    out.clear();
    out.extend(values.iter().map(|&byte| fp8_e5m2_to_f32(byte) * scale));
}

/// OCP **E2M1** (FP4) code → value: sign in bit 3, 2 exponent bits (bias 1), 1 mantissa bit; no
/// infinities and no NaN — all 16 codes are finite. `exponent == 0` is subnormal (`mantissa · 2^-1`).
/// This is the grid ComfyUI's `stochastic_float_to_fp4_e2m1` emits as `(sign << 3) | (exp << 1) | m`.
pub const E2M1_LUT: [f32; 16] = [
    0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0, -0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0,
];

/// Decode one E2M1 nibble (`0..16`; the high bits of a larger value are ignored by the mask).
pub fn e2m1_to_f32(code: u8) -> f32 {
    E2M1_LUT[(code & 0x0F) as usize]
}

// =================================================================================================
// MXFP8 geometry.
// =================================================================================================

/// ComfyUI's stored (32-padded) shape of a logical `[rows, cols]` MXFP8 weight.
pub fn mxfp8_padded_shape(logical: [usize; 2]) -> [usize; 2] {
    [
        logical[0].div_ceil(MXFP8_PAD) * MXFP8_PAD,
        logical[1].div_ceil(MXFP8_PAD) * MXFP8_PAD,
    ]
}

/// The shape `comfy.float.to_blocked` gives a `[rows, blocks]` block-scale matrix:
/// `[⌈rows/128⌉·128, ⌈blocks/4⌉·4]`. The 128×4 tiling is a property of the cuBLAS block-scale
/// layout, not of the element format, so MXFP8 (block 32) and NVFP4 (block 16) share it.
pub fn blocked_scale_shape(rows: usize, blocks: usize) -> [usize; 2] {
    [
        rows.div_ceil(MXFP8_SCALE_ROW_TILE) * MXFP8_SCALE_ROW_TILE,
        blocks.div_ceil(MXFP8_SCALE_COL_TILE) * MXFP8_SCALE_COL_TILE,
    ]
}

/// Index into a `to_blocked` scale buffer of shape [`blocked_scale_shape`]`(rows, blocks)` (read
/// row-major / flat) of the scale belonging to logical `(row, block)`.
///
/// `to_blocked` views the padded `[R, B]` scale matrix as `(R/128, 128, B/4, 4)`, permutes to
/// `(R/128, B/4, 128, 4)` — so the **atom grid is walked row-major**, `atom = (row/128)·(B/4) +
/// block/4` — splits the 128 rows into `(4, 32)`, transposes those two, and flattens; the intra-atom
/// slot is `(row%32)·16 + ((row%128)/32)·4 + block%4`.
///
/// This is the single derivation of the swizzle; every format-specific helper delegates here rather
/// than re-spelling the arithmetic.
pub fn blocked_scale_index(rows: usize, blocks: usize, row: usize, block: usize) -> usize {
    let col_tiles = blocked_scale_shape(rows, blocks)[1] / MXFP8_SCALE_COL_TILE;
    let row_tile = row / MXFP8_SCALE_ROW_TILE;
    let row_in_tile = row % MXFP8_SCALE_ROW_TILE;
    let atom = row_tile * col_tiles + block / MXFP8_SCALE_COL_TILE;
    (atom * 32 + row_in_tile % 32) * 16 + (row_in_tile / 32) * 4 + block % MXFP8_SCALE_COL_TILE
}

/// The swizzled block-scale tensor shape for a stored (padded) `[rows, cols]` MXFP8 weight:
/// `[⌈rows/128⌉·128, ⌈(cols/32)/4⌉·4]`. `cols` must already be a multiple of 32.
pub fn mxfp8_scale_shape(stored: [usize; 2]) -> [usize; 2] {
    blocked_scale_shape(stored[0], stored[1] / MXFP8_BLOCK)
}

/// [`blocked_scale_index`] for an MXFP8 weight whose stored shape is `stored` (`blocks =
/// stored_cols / 32`).
pub fn mxfp8_swizzled_scale_index(stored: [usize; 2], row: usize, block: usize) -> usize {
    blocked_scale_index(stored[0], stored[1] / MXFP8_BLOCK, row, block)
}

/// Why an MXFP8 layer's geometry is not decodable.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Mxfp8GeometryError {
    /// Stored weight is not rank-2.
    StoredRank { rank: usize },
    /// A stored axis is not a multiple of 32 (ComfyUI pads both axes; an unpadded file is not this
    /// format).
    StoredNotPadded { stored: [usize; 2] },
    /// The declared logical shape does not pad to the stored shape.
    LogicalDoesNotPadToStored {
        logical: [usize; 2],
        stored: [usize; 2],
        expected_stored: [usize; 2],
    },
    /// The scale tensor's shape is not the swizzled shape the stored weight needs.
    ScaleShape {
        stored: [usize; 2],
        scale: Vec<usize>,
        expected: [usize; 2],
    },
    /// Payload byte counts disagree with the shapes (defensive: the header reader already checks).
    PayloadLength {
        what: &'static str,
        expected: usize,
        actual: usize,
    },
}

impl fmt::Display for Mxfp8GeometryError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::StoredRank { rank } => write!(f, "mxfp8 weight must be rank-2, got rank {rank}"),
            Self::StoredNotPadded { stored } => write!(
                f,
                "mxfp8 weight stored shape {stored:?} is not a multiple of {MXFP8_PAD} on both axes"
            ),
            Self::LogicalDoesNotPadToStored {
                logical,
                stored,
                expected_stored,
            } => write!(
                f,
                "mxfp8 logical shape {logical:?} pads to {expected_stored:?} but the stored weight is {stored:?}"
            ),
            Self::ScaleShape {
                stored,
                scale,
                expected,
            } => write!(
                f,
                "mxfp8 weight_scale shape {scale:?} is not the swizzled {expected:?} a stored {stored:?} weight needs"
            ),
            Self::PayloadLength {
                what,
                expected,
                actual,
            } => write!(f, "mxfp8 {what} payload is {actual} bytes, expected {expected}"),
        }
    }
}

impl std::error::Error for Mxfp8GeometryError {}

/// Validate MXFP8 per-layer geometry from the header: stored weight shape, the scale tensor shape,
/// and (when the adapter declares it) the logical shape. Returns the logical shape the decode
/// unpads to — the declared one, or the stored one when no logical shape is declared (stored ⊇
/// logical; the model's own shape validation is then the backstop for a layer whose true shape is
/// not 32-aligned).
pub fn validate_mxfp8_geometry(
    stored: &[usize],
    scale_shape: &[usize],
    logical: Option<&[usize]>,
) -> Result<[usize; 2], Mxfp8GeometryError> {
    let [rows, cols] = stored else {
        return Err(Mxfp8GeometryError::StoredRank { rank: stored.len() });
    };
    let stored = [*rows, *cols];
    if !rows.is_multiple_of(MXFP8_PAD) || !cols.is_multiple_of(MXFP8_PAD) || *cols == 0 {
        return Err(Mxfp8GeometryError::StoredNotPadded { stored });
    }
    let expected_scale = mxfp8_scale_shape(stored);
    if scale_shape != expected_scale.as_slice() {
        return Err(Mxfp8GeometryError::ScaleShape {
            stored,
            scale: scale_shape.to_vec(),
            expected: expected_scale,
        });
    }
    match logical {
        None => Ok(stored),
        Some(logical) => {
            let [l_rows, l_cols] = logical else {
                return Err(Mxfp8GeometryError::LogicalDoesNotPadToStored {
                    logical: [0, 0],
                    stored,
                    expected_stored: [0, 0],
                });
            };
            let logical = [*l_rows, *l_cols];
            let expected_stored = mxfp8_padded_shape(logical);
            if expected_stored != stored {
                return Err(Mxfp8GeometryError::LogicalDoesNotPadToStored {
                    logical,
                    stored,
                    expected_stored,
                });
            }
            Ok(logical)
        }
    }
}

/// Reference MXFP8 dequantization: `values` is the stored `[rows, cols]` E4M3FN payload (row-major,
/// 32-padded), `scales` the swizzled E8M0 payload of shape [`mxfp8_scale_shape`]`(stored)`, and the
/// result is the **logical** `[logical_rows, logical_cols]` matrix (the padding rows/columns are
/// dropped), row-major f32: `out[r][c] = e4m3(values[r][c]) · 2^(e8m0(scale[r][c/32]) − 127)`.
pub fn decode_mxfp8(
    values: &[u8],
    scales: &[u8],
    stored: [usize; 2],
    logical: [usize; 2],
    out: &mut Vec<f32>,
) -> Result<(), Mxfp8GeometryError> {
    let expected_values = stored[0] * stored[1];
    if values.len() != expected_values {
        return Err(Mxfp8GeometryError::PayloadLength {
            what: "weight",
            expected: expected_values,
            actual: values.len(),
        });
    }
    let scale_shape = mxfp8_scale_shape(stored);
    let expected_scales = scale_shape[0] * scale_shape[1];
    if scales.len() != expected_scales {
        return Err(Mxfp8GeometryError::PayloadLength {
            what: "weight_scale",
            expected: expected_scales,
            actual: scales.len(),
        });
    }
    if logical[0] > stored[0] || logical[1] > stored[1] {
        return Err(Mxfp8GeometryError::LogicalDoesNotPadToStored {
            logical,
            stored,
            expected_stored: mxfp8_padded_shape(logical),
        });
    }
    out.clear();
    out.reserve(logical[0] * logical[1]);
    let blocks = stored[1] / MXFP8_BLOCK;
    for row in 0..logical[0] {
        let row_values = &values[row * stored[1]..(row + 1) * stored[1]];
        for block in 0..blocks {
            let block_start = block * MXFP8_BLOCK;
            if block_start >= logical[1] {
                break;
            }
            let scale = e8m0_to_f32(scales[mxfp8_swizzled_scale_index(stored, row, block)]);
            let block_end = (block_start + MXFP8_BLOCK).min(logical[1]);
            out.extend(
                row_values[block_start..block_end]
                    .iter()
                    .map(|&byte| fp8_e4m3fn_to_f32(byte) * scale),
            );
        }
    }
    Ok(())
}

// =================================================================================================
// NVFP4 geometry + reference decode (sc-20641).
// =================================================================================================

/// ComfyUI's stored (16-padded) logical shape of an NVFP4 weight
/// (`TensorCoreNVFP4Layout.get_padded_shape` / `pad_16x`). The **byte** shape on disk is
/// `[rows, cols / 2]` — two E2M1 codes per `U8`.
pub fn nvfp4_padded_shape(logical: [usize; 2]) -> [usize; 2] {
    [
        logical[0].div_ceil(NVFP4_PAD) * NVFP4_PAD,
        logical[1].div_ceil(NVFP4_PAD) * NVFP4_PAD,
    ]
}

/// The `to_blocked` block-scale shape for a stored (16-padded) `[rows, cols]` NVFP4 weight:
/// `[⌈rows/128⌉·128, ⌈(cols/16)/4⌉·4]`. `cols` must already be a multiple of 16.
pub fn nvfp4_scale_shape(stored: [usize; 2]) -> [usize; 2] {
    blocked_scale_shape(stored[0], stored[1] / NVFP4_BLOCK)
}

/// [`blocked_scale_index`] for an NVFP4 weight whose stored shape is `stored` (`blocks =
/// stored_cols / 16`).
pub fn nvfp4_swizzled_scale_index(stored: [usize; 2], row: usize, block: usize) -> usize {
    blocked_scale_index(stored[0], stored[1] / NVFP4_BLOCK, row, block)
}

/// Why an NVFP4 layer's geometry is not decodable. Every variant is a per-layer fact; the plan
/// compiler attaches the tensor name.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Nvfp4GeometryError {
    /// The stored `U8` weight is not rank-2 `[rows, cols/2]`.
    StoredRank { rank: usize },
    /// A stored axis is not a multiple of 16 (ComfyUI pads both axes; an unpadded file is not this
    /// format). `stored` is the **logical** `[rows, cols]` the byte shape implies.
    StoredNotPadded { stored: [usize; 2] },
    /// The declared logical shape does not pad to the stored shape.
    LogicalDoesNotPadToStored {
        logical: [usize; 2],
        stored: [usize; 2],
        expected_stored: [usize; 2],
    },
    /// The block-scale tensor's shape is not the swizzled shape the stored weight needs.
    ScaleShape {
        stored: [usize; 2],
        scale: Vec<usize>,
        expected: [usize; 2],
    },
    /// Payload byte counts disagree with the shapes — a truncated nibble payload lands here.
    PayloadLength {
        what: &'static str,
        expected: usize,
        actual: usize,
    },
    /// The `weight_scale_2` per-tensor scale is not a usable multiplier. Carried as raw bits so the
    /// error type stays `Eq` alongside its siblings (a NaN scale is exactly one of the cases).
    GlobalScale { bits: u32 },
    /// A block scale that governs real (non-padding) elements is the E4M3 NaN code. ComfyUI clamps
    /// block scales to 448, so this is corruption, and multiplying through it would quietly poison
    /// 16 weights instead of failing.
    BlockScaleNaN { row: usize, block: usize },
    /// A block scale that governs real (non-padding) elements carries the E4M3 sign bit. ComfyUI's
    /// quantizer clamps block scales to `[0, 448]`, so a negative scale is corruption; multiplying
    /// through it would silently negate 16 weights instead of failing (sc-21482 — the check the
    /// provider-owned Krea payload scan used to make, now owned by the codec).
    BlockScaleNegative { row: usize, block: usize },
    /// A padding element (row ≥ logical rows, or column ≥ logical cols) is not the E2M1 zero code
    /// ComfyUI's `F.pad` writes. A non-zero pad means the shapes are being read wrong — the values
    /// would be dropped silently otherwise.
    PaddingNotZero { row: usize, col: usize, code: u8 },
}

impl fmt::Display for Nvfp4GeometryError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::StoredRank { rank } => write!(
                f,
                "nvfp4 weight must be a rank-2 U8 [out, in/2] matrix, got rank {rank}"
            ),
            Self::StoredNotPadded { stored } => write!(
                f,
                "nvfp4 weight logical shape {stored:?} is not a multiple of {NVFP4_PAD} on both axes"
            ),
            Self::LogicalDoesNotPadToStored {
                logical,
                stored,
                expected_stored,
            } => write!(
                f,
                "nvfp4 logical shape {logical:?} pads to {expected_stored:?} but the stored weight is {stored:?}"
            ),
            Self::ScaleShape {
                stored,
                scale,
                expected,
            } => write!(
                f,
                "nvfp4 weight_scale shape {scale:?} is not the swizzled {expected:?} a stored {stored:?} weight needs"
            ),
            Self::PayloadLength {
                what,
                expected,
                actual,
            } => write!(f, "nvfp4 {what} payload is {actual} bytes, expected {expected}"),
            Self::GlobalScale { bits } => write!(
                f,
                "nvfp4 `weight_scale_2` must be finite and non-negative, got {}",
                f32::from_bits(*bits)
            ),
            Self::BlockScaleNaN { row, block } => write!(
                f,
                "nvfp4 block scale at (row {row}, block {block}) is the E4M3 NaN code 0x7F"
            ),
            Self::BlockScaleNegative { row, block } => write!(
                f,
                "nvfp4 block scale at (row {row}, block {block}) carries the E4M3 sign bit; block \
                 scales are clamped to [0, 448] at quantization, so a negative scale is corruption"
            ),
            Self::PaddingNotZero { row, col, code } => write!(
                f,
                "nvfp4 padding element at (row {row}, col {col}) holds E2M1 code {code:#x}, not zero"
            ),
        }
    }
}

impl std::error::Error for Nvfp4GeometryError {}

/// Validate NVFP4 per-layer geometry from the header: the stored `U8` `[rows, cols/2]` byte shape,
/// the block-scale tensor shape, and (when the adapter declares it) the logical shape. Returns
/// `(stored_logical, logical)` — the 16-padded `[rows, cols]` the payload holds, and the shape the
/// decode unpads to (the declared one, or the stored one when no logical shape is declared).
pub fn validate_nvfp4_geometry(
    packed: &[usize],
    scale_shape: &[usize],
    logical: Option<&[usize]>,
) -> Result<([usize; 2], [usize; 2]), Nvfp4GeometryError> {
    let [rows, packed_cols] = packed else {
        return Err(Nvfp4GeometryError::StoredRank { rank: packed.len() });
    };
    let stored = [*rows, packed_cols.saturating_mul(2)];
    if stored[0] == 0
        || stored[1] == 0
        || !stored[0].is_multiple_of(NVFP4_PAD)
        || !stored[1].is_multiple_of(NVFP4_PAD)
    {
        return Err(Nvfp4GeometryError::StoredNotPadded { stored });
    }
    let expected_scale = nvfp4_scale_shape(stored);
    if scale_shape != expected_scale.as_slice() {
        return Err(Nvfp4GeometryError::ScaleShape {
            stored,
            scale: scale_shape.to_vec(),
            expected: expected_scale,
        });
    }
    match logical {
        None => Ok((stored, stored)),
        Some(logical) => {
            let [l_rows, l_cols] = logical else {
                return Err(Nvfp4GeometryError::LogicalDoesNotPadToStored {
                    logical: [0, 0],
                    stored,
                    expected_stored: [0, 0],
                });
            };
            let logical = [*l_rows, *l_cols];
            let expected_stored = nvfp4_padded_shape(logical);
            if expected_stored != stored {
                return Err(Nvfp4GeometryError::LogicalDoesNotPadToStored {
                    logical,
                    stored,
                    expected_stored,
                });
            }
            Ok((stored, logical))
        }
    }
}

/// Validate the NVFP4 block-scale **payload** for one layer: every scale byte that governs a real
/// (non-padding) element of the `logical` matrix must be a valid non-negative, non-NaN UE4M3
/// magnitude. `scales` is the `to_blocked` swizzled buffer for the `stored` grid, `logical` the
/// shape the layer decodes/unpads to.
///
/// This is a value check the plan compiler's header-level geometry checks cannot make (it never
/// touches payload bytes), and it is codec-owned rather than provider-owned (sc-21482): both the
/// dense decode ([`decode_nvfp4`] calls it) and a backend's packed-native repack must refuse the
/// same corrupted scale surface the same way, before the bad multiplier reaches any weight or any
/// GEMM.
pub fn validate_nvfp4_block_scale_payload(
    scales: &[u8],
    stored: [usize; 2],
    logical: [usize; 2],
) -> Result<(), Nvfp4GeometryError> {
    let scale_shape = nvfp4_scale_shape(stored);
    let expected_scales = scale_shape[0] * scale_shape[1];
    if scales.len() != expected_scales {
        return Err(Nvfp4GeometryError::PayloadLength {
            what: "weight_scale",
            expected: expected_scales,
            actual: scales.len(),
        });
    }
    let blocks = stored[1] / NVFP4_BLOCK;
    for row in 0..logical[0].min(stored[0]) {
        for block in 0..blocks {
            if block * NVFP4_BLOCK >= logical[1] {
                break;
            }
            let scale_byte = scales[nvfp4_swizzled_scale_index(stored, row, block)];
            if scale_byte & 0x7F == 0x7F {
                return Err(Nvfp4GeometryError::BlockScaleNaN { row, block });
            }
            if scale_byte & 0x80 != 0 {
                return Err(Nvfp4GeometryError::BlockScaleNegative { row, block });
            }
        }
    }
    Ok(())
}

/// Reference NVFP4 dequantization — the two-level decode, in plain f32.
///
/// `packed` is the stored `[rows, cols/2]` byte payload (row-major, **even column in the high
/// nibble**, per `comfy.float.stochastic_float_to_fp4_e2m1`'s `(even << 4) | odd`), `scales` the
/// `to_blocked` E4M3 payload of shape [`nvfp4_scale_shape`]`(stored)`, `global_scale` the layer's
/// `weight_scale_2`. The result is the **logical** matrix, row-major:
///
/// ```text
/// out[r][c] = e2m1(code[r][c]) · e4m3(scale[r][c / 16]) · global_scale
/// ```
///
/// Padding elements outside the logical shape are validated to be zero rather than merely dropped.
pub fn decode_nvfp4(
    packed: &[u8],
    scales: &[u8],
    global_scale: f32,
    stored: [usize; 2],
    logical: [usize; 2],
    out: &mut Vec<f32>,
) -> Result<(), Nvfp4GeometryError> {
    let row_bytes = stored[1] / 2;
    let expected_packed = stored[0] * row_bytes;
    if packed.len() != expected_packed {
        return Err(Nvfp4GeometryError::PayloadLength {
            what: "weight",
            expected: expected_packed,
            actual: packed.len(),
        });
    }
    if !global_scale.is_finite() || global_scale < 0.0 {
        return Err(Nvfp4GeometryError::GlobalScale {
            bits: global_scale.to_bits(),
        });
    }
    if logical[0] > stored[0] || logical[1] > stored[1] {
        return Err(Nvfp4GeometryError::LogicalDoesNotPadToStored {
            logical,
            stored,
            expected_stored: nvfp4_padded_shape(logical),
        });
    }
    validate_nvfp4_block_scale_payload(scales, stored, logical)?;

    let code_at = |row: usize, col: usize| -> u8 {
        let byte = packed[row * row_bytes + col / 2];
        if col.is_multiple_of(2) {
            byte >> 4
        } else {
            byte & 0x0F
        }
    };
    // Padding must be the zero code ComfyUI's `F.pad` writes — checked before anything is emitted so
    // a mis-read shape cannot be silently trimmed away.
    for row in 0..stored[0] {
        let col_start = if row < logical[0] { logical[1] } else { 0 };
        for col in col_start..stored[1] {
            let code = code_at(row, col);
            if code != 0 {
                return Err(Nvfp4GeometryError::PaddingNotZero { row, col, code });
            }
        }
    }

    let blocks = stored[1] / NVFP4_BLOCK;
    out.clear();
    out.reserve(logical[0] * logical[1]);
    for row in 0..logical[0] {
        for block in 0..blocks {
            let block_start = block * NVFP4_BLOCK;
            if block_start >= logical[1] {
                break;
            }
            // NaN/negative scale bytes were refused by `validate_nvfp4_block_scale_payload` above.
            let scale_byte = scales[nvfp4_swizzled_scale_index(stored, row, block)];
            let element_scale = fp8_e4m3fn_to_f32(scale_byte) * global_scale;
            let block_end = (block_start + NVFP4_BLOCK).min(logical[1]);
            for col in block_start..block_end {
                out.push(e2m1_to_f32(code_at(row, col)) * element_scale);
            }
        }
    }
    Ok(())
}

/// Reference int8 per-row dequantization: `codes` is `[rows, cols]` row-major, `scales` has one
/// entry per row. Shared by the backend arms' tests.
pub fn decode_int8_per_row(codes: &[i8], scales: &[f32], cols: usize, out: &mut Vec<f32>) {
    out.clear();
    out.extend(
        codes
            .iter()
            .enumerate()
            .map(|(index, &code)| code as f32 * scales[index / cols]),
    );
}

#[cfg(test)]
mod tests {
    use super::*;

    // --- independent spec references: bit-field arithmetic spelled out the long way -------------

    /// E4M3FN per the OCP FP8 spec, computed from the bit fields with integer arithmetic only.
    fn spec_e4m3fn(byte: u8) -> f32 {
        let s = (byte >> 7) as i32;
        let e = ((byte >> 3) & 0xF) as i32;
        let m = (byte & 0x7) as i32;
        if e == 15 && m == 7 {
            return f32::NAN;
        }
        let value = if e == 0 {
            // subnormal: 0.m × 2^(1-7)
            (m as f64) / 8.0 * 2f64.powi(1 - 7)
        } else {
            (1.0 + (m as f64) / 8.0) * 2f64.powi(e - 7)
        };
        (if s == 1 { -value } else { value }) as f32
    }

    /// E5M2 per the OCP FP8 spec.
    fn spec_e5m2(byte: u8) -> f32 {
        let s = (byte >> 7) as i32;
        let e = ((byte >> 2) & 0x1F) as i32;
        let m = (byte & 0x3) as i32;
        if e == 31 {
            return if m == 0 {
                if s == 1 {
                    f32::NEG_INFINITY
                } else {
                    f32::INFINITY
                }
            } else {
                f32::NAN
            };
        }
        let value = if e == 0 {
            (m as f64) / 4.0 * 2f64.powi(1 - 15)
        } else {
            (1.0 + (m as f64) / 4.0) * 2f64.powi(e - 15)
        };
        (if s == 1 { -value } else { value }) as f32
    }

    fn same_value(left: f32, right: f32) -> bool {
        (left.is_nan() && right.is_nan()) || left.to_bits() == right.to_bits()
    }

    #[test]
    fn e4m3fn_decoder_matches_the_spec_for_all_256_codes_and_the_canonical_table() {
        for code in 0..=255_u8 {
            assert!(
                same_value(fp8_e4m3fn_to_f32(code), spec_e4m3fn(code)),
                "code {code:#04x}: got {} want {}",
                fp8_e4m3fn_to_f32(code),
                spec_e4m3fn(code)
            );
        }
        // Hand-derived anchors (OCP FP8 spec, Table 1): 1.0, max 448, min normal 2^-6, min
        // subnormal 2^-9, -1.125, 4.5, zero, negative zero, NaN.
        assert_eq!(fp8_e4m3fn_to_f32(0x38), 1.0);
        assert_eq!(fp8_e4m3fn_to_f32(0x7E), 448.0);
        assert_eq!(fp8_e4m3fn_to_f32(0x08), 2f32.powi(-6));
        assert_eq!(fp8_e4m3fn_to_f32(0x01), 2f32.powi(-9));
        assert_eq!(fp8_e4m3fn_to_f32(0xB9), -1.125);
        assert_eq!(fp8_e4m3fn_to_f32(0x49), 4.5);
        assert_eq!(fp8_e4m3fn_to_f32(0x00).to_bits(), 0.0_f32.to_bits());
        assert_eq!(fp8_e4m3fn_to_f32(0x80).to_bits(), (-0.0_f32).to_bits());
        assert!(fp8_e4m3fn_to_f32(0x7F).is_nan() && fp8_e4m3fn_to_f32(0xFF).is_nan());
        // No infinities in E4M3FN: exponent 15 with a non-7 mantissa is finite.
        assert_eq!(fp8_e4m3fn_to_f32(0x78), 256.0);
    }

    #[test]
    fn e5m2_decoder_matches_the_spec_for_all_256_codes_and_equals_the_binary16_top_byte() {
        for code in 0..=255_u8 {
            assert!(
                same_value(fp8_e5m2_to_f32(code), spec_e5m2(code)),
                "code {code:#04x}"
            );
            // E5M2 is binary16 truncated to its top byte: decode via an IEEE binary16 whose low
            // byte is 0 (third independent reference).
            let bits = (code as u16) << 8;
            let (sign, exponent, mantissa) = (
                (bits >> 15) as i32,
                ((bits >> 10) & 0x1F) as i32,
                (bits & 0x3FF) as i32,
            );
            let binary16 = if exponent == 31 {
                if mantissa == 0 {
                    f32::INFINITY * if sign == 1 { -1.0 } else { 1.0 }
                } else {
                    f32::NAN
                }
            } else if exponent == 0 {
                (if sign == 1 { -1.0 } else { 1.0 }) * (mantissa as f32 / 1024.0) * 2f32.powi(-14)
            } else {
                (if sign == 1 { -1.0 } else { 1.0 })
                    * (1.0 + mantissa as f32 / 1024.0)
                    * 2f32.powi(exponent - 15)
            };
            assert!(
                same_value(fp8_e5m2_to_f32(code), binary16),
                "code {code:#04x}"
            );
        }
        assert_eq!(fp8_e5m2_to_f32(0x3C), 1.0);
        assert_eq!(fp8_e5m2_to_f32(0x7B), 57344.0);
        assert_eq!(fp8_e5m2_to_f32(0x04), 2f32.powi(-14));
        assert_eq!(fp8_e5m2_to_f32(0x01), 2f32.powi(-16));
        assert_eq!(fp8_e5m2_to_f32(0x7C), f32::INFINITY);
        assert_eq!(fp8_e5m2_to_f32(0xFC), f32::NEG_INFINITY);
        assert!(fp8_e5m2_to_f32(0x7D).is_nan());
    }

    #[test]
    fn e8m0_decoder_is_two_to_the_biased_exponent_with_nan_at_255() {
        assert_eq!(e8m0_to_f32(127), 1.0);
        assert_eq!(e8m0_to_f32(128), 2.0);
        assert_eq!(e8m0_to_f32(126), 0.5);
        assert_eq!(e8m0_to_f32(254), 2f32.powi(127));
        assert_eq!(e8m0_to_f32(1), 2f32.powi(-126));
        assert_eq!(e8m0_to_f32(0), 2f32.powi(-127));
        assert!(e8m0_to_f32(0).is_subnormal());
        assert!(e8m0_to_f32(255).is_nan());
        for code in 0..=254_u8 {
            assert_eq!(e8m0_to_f32(code), 2f32.powi(code as i32 - 127), "{code}");
        }
    }

    #[test]
    fn descriptor_parser_accepts_the_five_formats_and_refuses_every_defect_by_name() {
        assert_eq!(
            parse_comfy_quant_descriptor(br#"{"format": "nvfp4"}"#).map(|d| d.format),
            Ok(ComfyQuantFormat::Nvfp4)
        );
        assert_eq!(
            parse_comfy_quant_descriptor(br#"{"format": "float8_e4m3fn"}"#),
            Ok(ComfyQuantDescriptor {
                format: ComfyQuantFormat::Float8E4M3Fn,
                full_precision_matrix_mult: false
            })
        );
        assert_eq!(
            parse_comfy_quant_descriptor(
                br#"{"format": "float8_e5m2", "full_precision_matrix_mult": true}"#
            ),
            Ok(ComfyQuantDescriptor {
                format: ComfyQuantFormat::Float8E5M2,
                full_precision_matrix_mult: true
            })
        );
        assert_eq!(
            parse_comfy_quant_descriptor(br#"{"format": "mxfp8"}"#).map(|d| d.format),
            Ok(ComfyQuantFormat::Mxfp8)
        );
        assert_eq!(
            parse_comfy_quant_descriptor(br#"{"format": "int8_tensorwise", "per_row": true}"#)
                .map(|d| d.format),
            Ok(ComfyQuantFormat::Int8TensorwisePerRow)
        );
        let cases: &[(&[u8], ComfyQuantDescriptorError)] = &[
            (b"\xff\xfe", ComfyQuantDescriptorError::NotUtf8),
            (
                b"{",
                ComfyQuantDescriptorError::NotJson {
                    detail: String::new(),
                },
            ),
            (b"[1]", ComfyQuantDescriptorError::NotAnObject),
            (
                br#"{"per_row": true}"#,
                ComfyQuantDescriptorError::MissingFormat,
            ),
            (
                br#"{"format": "int4_awq"}"#,
                ComfyQuantDescriptorError::UnsupportedFormat {
                    format: "int4_awq".to_owned(),
                },
            ),
            (
                br#"{"format": "int8_tensorwise", "per_row": true, "convrot": true}"#,
                ComfyQuantDescriptorError::ConvRot,
            ),
            (
                br#"{"format": "int8_tensorwise", "per_row": true, "convrot_groupsize": 256}"#,
                ComfyQuantDescriptorError::ConvRot,
            ),
            (
                br#"{"format": "int8_tensorwise"}"#,
                ComfyQuantDescriptorError::Int8NotPerRow,
            ),
            (
                br#"{"format": "int8_tensorwise", "per_row": false}"#,
                ComfyQuantDescriptorError::Int8NotPerRow,
            ),
            (
                br#"{"format": "float8_e4m3fn", "full_precision_matrix_mult": "yes"}"#,
                ComfyQuantDescriptorError::NonBooleanField {
                    field: "full_precision_matrix_mult",
                },
            ),
            (
                br#"{"format": "float8_e4m3fn", "group_size": 32}"#,
                ComfyQuantDescriptorError::UnknownField {
                    field: "group_size".to_owned(),
                },
            ),
            (
                br#"{"format": "float8_e4m3fn", "per_row": true}"#,
                ComfyQuantDescriptorError::UnknownField {
                    field: "per_row".to_owned(),
                },
            ),
        ];
        for (bytes, expected) in cases {
            let got = parse_comfy_quant_descriptor(bytes).expect_err("must refuse");
            match (&got, expected) {
                (
                    ComfyQuantDescriptorError::NotJson { .. },
                    ComfyQuantDescriptorError::NotJson { .. },
                ) => {}
                _ => assert_eq!(&got, expected, "{:?}", String::from_utf8_lossy(bytes)),
            }
            assert!(!got.to_string().is_empty());
        }
    }

    /// E2M1 per the OCP MX spec, computed from the bit fields — sign bit 3, 2 exponent bits at
    /// bias 1, 1 mantissa bit — with no LUT in sight. This is the second derivation the shipped
    /// [`E2M1_LUT`] is checked against; it is also exactly the inverse of the encoder ComfyUI
    /// writes (`(sign << 3) | (exp << 1) | mantissa`).
    fn spec_e2m1(code: u8) -> f32 {
        let s = ((code >> 3) & 1) as i32;
        let e = ((code >> 1) & 0x3) as i32;
        let m = (code & 0x1) as i32;
        let value = if e == 0 {
            (m as f64) / 2.0 // subnormal: 0.m × 2^(1-1)
        } else {
            (1.0 + (m as f64) / 2.0) * 2f64.powi(e - 1)
        };
        (if s == 1 { -value } else { value }) as f32
    }

    #[test]
    fn e2m1_decoder_matches_the_spec_for_all_16_codes_and_the_canonical_grid() {
        for code in 0..16_u8 {
            assert!(
                same_value(e2m1_to_f32(code), spec_e2m1(code)),
                "code {code:#x}: got {} want {}",
                e2m1_to_f32(code),
                spec_e2m1(code)
            );
        }
        // The canonical NVFP4 grid, hand-written (NVIDIA NVFP4 / OCP MX E2M1 Table): the eight
        // positive magnitudes and their negations, in code order.
        assert_eq!(
            E2M1_LUT,
            [
                0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0, -0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0,
                -6.0
            ]
        );
        // E2M1 has no NaN and no infinity: every code is finite, and 6.0 is the max magnitude.
        assert!(E2M1_LUT.iter().all(|value| value.is_finite()));
        assert_eq!(
            E2M1_LUT
                .iter()
                .fold(0.0_f32, |max, value| max.max(value.abs())),
            6.0
        );
        // Sign bit is bit 3 and nothing else: code and code|8 differ only in sign.
        for code in 0..8_u8 {
            assert_eq!(
                e2m1_to_f32(code | 0x08),
                -e2m1_to_f32(code),
                "code {code:#x}"
            );
        }
        // High bits above the nibble are masked off (the packer hands whole bytes through).
        assert_eq!(e2m1_to_f32(0xF7), e2m1_to_f32(0x07));
    }

    #[test]
    fn nvfp4_geometry_pads_both_axes_to_16_and_sizes_the_swizzled_scale_tile() {
        assert_eq!(nvfp4_padded_shape([40, 70]), [48, 80]);
        assert_eq!(nvfp4_padded_shape([16, 32]), [16, 32]);
        // The shipped kreamania_variant7 shapes: U8 [6144, 3072] → logical [6144, 6144], scale
        // [6144, 384]; U8 [16384, 3072] → scale [16384, 384]; U8 [6144, 8192] → scale [6144, 1024].
        assert_eq!(nvfp4_scale_shape([6144, 6144]), [6144, 384]);
        assert_eq!(nvfp4_scale_shape([16384, 6144]), [16384, 384]);
        assert_eq!(nvfp4_scale_shape([6144, 16384]), [6144, 1024]);
        assert_eq!(nvfp4_scale_shape([1536, 6144]), [1536, 384]);
        // Padding of the scale tile itself: 48 rows → 128, 5 blocks → 8.
        assert_eq!(nvfp4_scale_shape([48, 80]), [128, 8]);

        assert_eq!(
            validate_nvfp4_geometry(&[6144, 3072], &[6144, 384], None),
            Ok(([6144, 6144], [6144, 6144]))
        );
        assert_eq!(
            validate_nvfp4_geometry(&[48, 40], &[128, 8], Some(&[40, 70])),
            Ok(([48, 80], [40, 70]))
        );
        assert_eq!(
            validate_nvfp4_geometry(&[48, 40], &[128, 8], Some(&[40, 90])),
            Err(Nvfp4GeometryError::LogicalDoesNotPadToStored {
                logical: [40, 90],
                stored: [48, 80],
                expected_stored: [48, 96]
            })
        );
        // Byte shape whose doubled columns are not 16-aligned, and a non-16 row count.
        assert_eq!(
            validate_nvfp4_geometry(&[32, 20], &[128, 4], None),
            Err(Nvfp4GeometryError::StoredNotPadded { stored: [32, 40] })
        );
        assert_eq!(
            validate_nvfp4_geometry(&[20, 32], &[128, 4], None),
            Err(Nvfp4GeometryError::StoredNotPadded { stored: [20, 64] })
        );
        assert_eq!(
            validate_nvfp4_geometry(&[48, 40], &[48, 5], None),
            Err(Nvfp4GeometryError::ScaleShape {
                stored: [48, 80],
                scale: vec![48, 5],
                expected: [128, 8]
            })
        );
        assert_eq!(
            validate_nvfp4_geometry(&[6144], &[6144, 384], None),
            Err(Nvfp4GeometryError::StoredRank { rank: 1 })
        );
    }

    /// The NVFP4 swizzle is the *same* cuBLAS 128×4 `to_blocked` layout MXFP8 uses — only the block
    /// width differs — so both delegate to one derivation and the pinned `to_blocked` table pins
    /// both.
    #[test]
    fn nvfp4_and_mxfp8_share_one_blocked_scale_derivation() {
        // A [256, 256] MXFP8 weight and a [256, 128] NVFP4 weight both have 8 blocks per row, so
        // every (row, block) must land at the same flat slot under either helper.
        for row in 0..256 {
            for block in 0..8 {
                let shared = blocked_scale_index(256, 8, row, block);
                assert_eq!(mxfp8_swizzled_scale_index([256, 256], row, block), shared);
                assert_eq!(nvfp4_swizzled_scale_index([256, 128], row, block), shared);
            }
        }
        assert_eq!(blocked_scale_shape(256, 8), [256, 8]);
        assert_eq!(mxfp8_scale_shape([256, 256]), blocked_scale_shape(256, 8));
        assert_eq!(nvfp4_scale_shape([256, 128]), blocked_scale_shape(256, 8));
        // Bijection over a multi-atom grid (2 row atoms × 2 block atoms) — the case where a
        // row-major vs column-major atom walk would differ.
        let (rows, blocks) = (256, 8);
        let shape = blocked_scale_shape(rows, blocks);
        let mut seen = vec![false; shape[0] * shape[1]];
        for row in 0..rows {
            for block in 0..blocks {
                let index = blocked_scale_index(rows, blocks, row, block);
                assert!(!seen[index], "({row},{block}) collides at {index}");
                seen[index] = true;
            }
        }
        assert!(seen.iter().all(|slot| *slot));
        // The atom grid is walked ROW-major (`permute(0, 2, 1, 3)` in `to_blocked`): logical
        // (row 0, block 4) is in atom 1, i.e. flat 512 — not atom 2 (1024), which a column-major
        // atom walk would give for this 2×2 grid.
        assert_eq!(blocked_scale_index(256, 8, 0, 4), 512);
        assert_eq!(blocked_scale_index(256, 8, 128, 0), 1024);
    }

    #[test]
    fn nvfp4_decode_applies_both_scale_levels_unpads_and_refuses_corruption() {
        // Stored [16, 64] (4 blocks per row), logical [10, 40]: a non-block-aligned tail (cols
        // 32..40 share block 1's scale, 40..64 are padding) and padded rows 10..16.
        let stored = [16_usize, 64];
        let logical = [10_usize, 40];
        let row_bytes = stored[1] / 2;
        let mut packed = vec![0_u8; stored[0] * row_bytes];
        // Real region: column c holds E2M1 code (c % 8) — walks the whole positive grid.
        for row in 0..logical[0] {
            for col in 0..logical[1] {
                let code = (col % 8) as u8;
                let index = row * row_bytes + col / 2;
                if col.is_multiple_of(2) {
                    packed[index] = (packed[index] & 0x0F) | (code << 4);
                } else {
                    packed[index] = (packed[index] & 0xF0) | code;
                }
            }
        }
        let scale_shape = nvfp4_scale_shape(stored);
        let mut scales = vec![0x7F_u8; scale_shape[0] * scale_shape[1]]; // NaN poison everywhere
        let exponents = [0x38_u8, 0x40, 0x30, 0x3C]; // 1.0, 2.0, 0.5, 1.5
                                                     // Blocks 0..3 govern real columns (block 2 covers the partial tail 32..40); block 3 is pure
                                                     // padding and keeps its NaN poison, which a correct decode never reads.
        for row in 0..logical[0] {
            for block in 0..3 {
                scales[nvfp4_swizzled_scale_index(stored, row, block)] =
                    exponents[(row + block) % 4];
            }
        }
        let global = 0.25_f32;

        let mut out = Vec::new();
        decode_nvfp4(&packed, &scales, global, stored, logical, &mut out).unwrap();
        assert_eq!(out.len(), logical[0] * logical[1]);
        for row in 0..logical[0] {
            for col in 0..logical[1] {
                let block = col / NVFP4_BLOCK;
                let expected =
                    E2M1_LUT[col % 8] * fp8_e4m3fn_to_f32(exponents[(row + block) % 4]) * global;
                assert_eq!(out[row * logical[1] + col], expected, "row {row} col {col}");
            }
        }

        // Truncated nibble payload / truncated scales.
        assert!(matches!(
            decode_nvfp4(&packed[..10], &scales, global, stored, logical, &mut out),
            Err(Nvfp4GeometryError::PayloadLength { what: "weight", .. })
        ));
        assert!(matches!(
            decode_nvfp4(&packed, &scales[..10], global, stored, logical, &mut out),
            Err(Nvfp4GeometryError::PayloadLength {
                what: "weight_scale",
                ..
            })
        ));
        // A non-finite / negative global scale.
        for bad in [f32::NAN, f32::INFINITY, -1.0] {
            assert!(matches!(
                decode_nvfp4(&packed, &scales, bad, stored, logical, &mut out),
                Err(Nvfp4GeometryError::GlobalScale { .. })
            ));
        }
        // A logical shape wider than storage.
        assert!(matches!(
            decode_nvfp4(&packed, &scales, global, stored, [10, 65], &mut out),
            Err(Nvfp4GeometryError::LogicalDoesNotPadToStored { .. })
        ));
        // A NaN block scale over real elements refuses instead of poisoning 16 weights.
        let mut nan_scales = scales.clone();
        nan_scales[nvfp4_swizzled_scale_index(stored, 3, 1)] = 0x7F;
        assert_eq!(
            decode_nvfp4(&packed, &nan_scales, global, stored, logical, &mut out),
            Err(Nvfp4GeometryError::BlockScaleNaN { row: 3, block: 1 })
        );
        // A non-zero padding element refuses rather than being trimmed away.
        let mut padded = packed.clone();
        // Row 0, column 41 — an odd column, so the low nibble of byte 20.
        padded[41 / 2] |= 0x07;
        assert_eq!(
            decode_nvfp4(&padded, &scales, global, stored, logical, &mut out),
            Err(Nvfp4GeometryError::PaddingNotZero {
                row: 0,
                col: 41,
                code: 7
            })
        );
        let mut padded_row = packed.clone();
        padded_row[12 * row_bytes] |= 0x50;
        assert_eq!(
            decode_nvfp4(&padded_row, &scales, global, stored, logical, &mut out),
            Err(Nvfp4GeometryError::PaddingNotZero {
                row: 12,
                col: 0,
                code: 5
            })
        );
    }

    #[test]
    fn quantization_metadata_parses_the_layer_table_and_refuses_every_defect() {
        let table = parse_quantization_metadata(
            r#"{"format_version": "1.0", "layers": {"blocks.0.attn.wq": {"format": "nvfp4"},
                "blocks.0.mlp.up": {"format": "float8_e4m3fn", "full_precision_matrix_mult": true}}}"#,
        )
        .expect("valid table");
        assert_eq!(table.len(), 2);
        assert_eq!(
            table["blocks.0.attn.wq"],
            // Declared fields only (sc-20651): an entry that says nothing about
            // `full_precision_matrix_mult` must stay distinguishable from one that declares `false`.
            PartialComfyQuantDescriptor {
                format: ComfyQuantFormat::Nvfp4,
                full_precision_matrix_mult: None,
                per_row: None,
            }
        );
        assert_eq!(
            table["blocks.0.mlp.up"].full_precision_matrix_mult,
            Some(true)
        );
        // Completing a standalone entry fills the ComfyUI defaults.
        assert_eq!(
            table["blocks.0.attn.wq"].into_complete(),
            Ok(ComfyQuantDescriptor {
                format: ComfyQuantFormat::Nvfp4,
                full_precision_matrix_mult: false
            })
        );
        // …and a bare int8 entry, which every real ComfyUI file writes, is only refusable as a
        // STANDALONE declaration — the table itself parses it (sc-20651).
        let int8 = parse_quantization_metadata(
            r#"{"format_version": "1.0", "layers": {"q": {"format": "int8_tensorwise"}}}"#,
        )
        .expect("a bare int8 entry is a well-formed table entry");
        assert_eq!(
            int8["q"],
            PartialComfyQuantDescriptor {
                format: ComfyQuantFormat::Int8TensorwisePerRow,
                full_precision_matrix_mult: None,
                per_row: None,
            }
        );
        assert_eq!(
            int8["q"].into_complete(),
            Err(ComfyQuantDescriptorError::Int8NotPerRow)
        );

        assert!(matches!(
            parse_quantization_metadata("{"),
            Err(QuantizationMetadataError::NotJson { .. })
        ));
        assert_eq!(
            parse_quantization_metadata("[1]"),
            Err(QuantizationMetadataError::NotAnObject)
        );
        // An ABSENT `format_version` is accepted as v1 (sc-21482): the real pinned
        // `Comfy-Org/Krea-2` NVFP4 export writes `{"layers": {…}}` with no version key, and
        // ComfyUI's own reader never consults one. Only a present-but-different version refuses.
        assert!(parse_quantization_metadata(r#"{"layers": {"a": {"format": "nvfp4"}}}"#).is_ok());
        assert_eq!(
            parse_quantization_metadata(r#"{"format_version": 1.0, "layers": {}}"#),
            Err(QuantizationMetadataError::FormatVersion {
                found: Some("1.0".to_owned())
            })
        );
        assert_eq!(
            parse_quantization_metadata(
                r#"{"format_version": "2.0", "layers": {"a": {"format": "nvfp4"}}}"#
            ),
            Err(QuantizationMetadataError::FormatVersion {
                found: Some("2.0".to_owned())
            })
        );
        assert_eq!(
            parse_quantization_metadata(r#"{"format_version": "1.0"}"#),
            Err(QuantizationMetadataError::Layers)
        );
        assert_eq!(
            parse_quantization_metadata(r#"{"format_version": "1.0", "layers": {}}"#),
            Err(QuantizationMetadataError::NoLayers)
        );
        // A layer defect is named by layer and carries the descriptor-level reason, unchanged.
        assert_eq!(
            parse_quantization_metadata(
                r#"{"format_version": "1.0", "layers": {"blocks.3.attn.wq": {"format": "fp6"}}}"#
            ),
            Err(QuantizationMetadataError::Layer {
                layer: "blocks.3.attn.wq".to_owned(),
                defect: ComfyQuantDescriptorError::UnsupportedFormat {
                    format: "fp6".to_owned()
                }
            })
        );
        assert_eq!(
            parse_quantization_metadata(
                r#"{"format_version": "1.0", "layers": {"blocks.3.attn.wq": {"format": "nvfp4", "group_size": 32}}}"#
            ),
            Err(QuantizationMetadataError::Layer {
                layer: "blocks.3.attn.wq".to_owned(),
                defect: ComfyQuantDescriptorError::UnknownField {
                    field: "group_size".to_owned()
                }
            })
        );
        for error in [
            QuantizationMetadataError::NotAnObject,
            QuantizationMetadataError::Layers,
            QuantizationMetadataError::NoLayers,
            QuantizationMetadataError::FormatVersion { found: None },
        ] {
            assert!(!error.to_string().is_empty());
        }
    }

    #[test]
    fn mxfp8_geometry_pads_both_axes_and_sizes_the_swizzled_scale_tile() {
        assert_eq!(mxfp8_padded_shape([40, 70]), [64, 96]);
        assert_eq!(mxfp8_padded_shape([32, 64]), [32, 64]);
        assert_eq!(mxfp8_padded_shape([1, 36]), [32, 64]);
        // comfy_kitchen eager: quantize_mxfp8(randn(40,70), pad_32x=True) → qdata [64,96], scale [128,4].
        assert_eq!(mxfp8_scale_shape([64, 96]), [128, 4]);
        assert_eq!(mxfp8_scale_shape([6144, 6144]), [6144, 192]);
        assert_eq!(mxfp8_scale_shape([256, 160]), [256, 8]);
        assert_eq!(
            validate_mxfp8_geometry(&[64, 96], &[128, 4], Some(&[40, 70])),
            Ok([40, 70])
        );
        assert_eq!(
            validate_mxfp8_geometry(&[64, 96], &[128, 4], None),
            Ok([64, 96])
        );
        assert_eq!(
            validate_mxfp8_geometry(&[64, 96], &[128, 4], Some(&[40, 100])),
            Err(Mxfp8GeometryError::LogicalDoesNotPadToStored {
                logical: [40, 100],
                stored: [64, 96],
                expected_stored: [64, 128]
            })
        );
        assert_eq!(
            validate_mxfp8_geometry(&[40, 70], &[128, 4], None),
            Err(Mxfp8GeometryError::StoredNotPadded { stored: [40, 70] })
        );
        assert_eq!(
            validate_mxfp8_geometry(&[64, 96], &[64, 3], None),
            Err(Mxfp8GeometryError::ScaleShape {
                stored: [64, 96],
                scale: vec![64, 3],
                expected: [128, 4]
            })
        );
        assert_eq!(
            validate_mxfp8_geometry(&[64], &[128, 4], None),
            Err(Mxfp8GeometryError::StoredRank { rank: 1 })
        );
    }

    /// The swizzle index reproduces comfy_kitchen `float_utils.to_blocked` exactly. The expected
    /// table below is the flat position each `(row, block)` lands at for a `[256, 8]` padded scale
    /// matrix (stored weight `[256, 256]`), computed by running `to_blocked` in the ComfyUI venv
    /// (torch 2.10, comfy_kitchen 0.2.8) on `arange(256*8).reshape(256, 8)` and reading where each
    /// value went; spot values cover every tile transition (row 31→32, 127→128, block 3→4).
    #[test]
    fn mxfp8_swizzle_index_matches_comfy_kitchen_to_blocked() {
        let stored = [256, 256];
        // (row, block) → flat index in the swizzled buffer.
        let expected: &[((usize, usize), usize)] = &[
            ((0, 0), 0),
            ((0, 1), 1),
            ((0, 3), 3),
            ((0, 4), 512),
            ((0, 7), 515),
            ((1, 0), 16),
            ((31, 0), 496),
            ((32, 0), 4),
            ((32, 3), 7),
            ((33, 1), 21),
            ((64, 0), 8),
            ((96, 0), 12),
            ((127, 3), 511),
            ((127, 7), 1023),
            ((128, 0), 1024),
            ((128, 4), 1536),
            ((255, 7), 2047),
        ];
        for &((row, block), index) in expected {
            assert_eq!(
                mxfp8_swizzled_scale_index(stored, row, block),
                index,
                "(row {row}, block {block})"
            );
        }
        // Bijection over the whole buffer: every (row, block) maps to a distinct in-range slot.
        let scale_shape = mxfp8_scale_shape(stored);
        let mut seen = vec![false; scale_shape[0] * scale_shape[1]];
        for row in 0..stored[0] {
            for block in 0..stored[1] / MXFP8_BLOCK {
                let index = mxfp8_swizzled_scale_index(stored, row, block);
                assert!(!seen[index], "({row},{block}) collides at {index}");
                seen[index] = true;
            }
        }
        assert!(seen.iter().all(|slot| *slot));
    }

    #[test]
    fn mxfp8_decode_unpads_and_applies_each_blocks_shared_exponent() {
        // Stored [32, 64] (2 blocks per row), logical [3, 40] — a non-block-aligned tail (cols 32..40
        // use block 1's exponent, cols 40..64 are padding) and padded rows 3..32.
        let stored = [32, 64];
        let logical = [3, 40];
        let mut values = vec![0_u8; stored[0] * stored[1]];
        // Row r, column c holds code 0x38 (1.0) + (c % 4) → 1.0, 1.125, 1.25, 1.375.
        for row in 0..stored[0] {
            for col in 0..stored[1] {
                values[row * stored[1] + col] = 0x38 + (col % 4) as u8;
            }
        }
        // Padding columns and rows carry poison so a wrong unpad shows up.
        for row in 0..stored[0] {
            for col in 40..64 {
                values[row * stored[1] + col] = 0x7E; // 448
            }
        }
        let scale_shape = mxfp8_scale_shape(stored);
        let mut scales = vec![0xFF_u8; scale_shape[0] * scale_shape[1]]; // NaN poison everywhere
                                                                         // Row 0: block 0 → 2^0, block 1 → 2^1. Row 1: 2^-1, 2^2. Row 2: 2^3, 2^-2.
        let exps = [[127_u8, 128], [126, 129], [130, 125]];
        for (row, row_exps) in exps.iter().enumerate() {
            for (block, exp) in row_exps.iter().enumerate() {
                scales[mxfp8_swizzled_scale_index(stored, row, block)] = *exp;
            }
        }
        let mut out = Vec::new();
        decode_mxfp8(&values, &scales, stored, logical, &mut out).unwrap();
        assert_eq!(out.len(), 3 * 40);
        let base = [1.0_f32, 1.125, 1.25, 1.375];
        for row in 0..3 {
            for col in 0..40 {
                let block = col / 32;
                let expected = base[col % 4] * 2f32.powi(exps[row][block] as i32 - 127);
                assert_eq!(out[row * 40 + col], expected, "row {row} col {col}");
            }
        }
        // Adversarial: a logical shape wider than storage refuses; wrong payload lengths refuse.
        assert!(matches!(
            decode_mxfp8(&values, &scales, stored, [3, 65], &mut out),
            Err(Mxfp8GeometryError::LogicalDoesNotPadToStored { .. })
        ));
        assert!(matches!(
            decode_mxfp8(&values[..10], &scales, stored, logical, &mut out),
            Err(Mxfp8GeometryError::PayloadLength { what: "weight", .. })
        ));
        assert!(matches!(
            decode_mxfp8(&values, &scales[..10], stored, logical, &mut out),
            Err(Mxfp8GeometryError::PayloadLength {
                what: "weight_scale",
                ..
            })
        ));
    }
}
