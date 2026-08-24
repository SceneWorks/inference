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
//!
//! `nvfp4` (`{layer}.weight_scale_2` + E2M1 nibbles) is named and refused here — sc-20641.

use std::fmt;

/// Block length of one MXFP8 shared exponent (OCP MX spec; `comfy_kitchen.MXFP8_BLOCK_SIZE`).
pub const MXFP8_BLOCK: usize = 32;
/// ComfyUI pads MXFP8 storage to multiples of 32 on both axes (`TensorCoreMXFP8Layout.get_padded_shape`).
pub const MXFP8_PAD: usize = 32;
/// cuBLAS block-scale swizzle tile: 128 rows × 4 scale columns (`comfy_kitchen.float_utils.to_blocked`).
pub const MXFP8_SCALE_ROW_TILE: usize = 128;
pub const MXFP8_SCALE_COL_TILE: usize = 4;

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
}

impl ComfyQuantFormat {
    /// The exact `format` string ComfyUI writes.
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Int8TensorwisePerRow => "int8_tensorwise",
            Self::Float8E4M3Fn => "float8_e4m3fn",
            Self::Float8E5M2 => "float8_e5m2",
            Self::Mxfp8 => "mxfp8",
        }
    }

    pub fn parse(name: &str) -> Option<Self> {
        Some(match name {
            "int8_tensorwise" => Self::Int8TensorwisePerRow,
            "float8_e4m3fn" => Self::Float8E4M3Fn,
            "float8_e5m2" => Self::Float8E5M2,
            "mxfp8" => Self::Mxfp8,
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
    /// never best-effort: `nvfp4` lands here until sc-20641 registers it).
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
                 int8_tensorwise, float8_e4m3fn, float8_e5m2, mxfp8)"
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
        None => false,
        Some(serde_json::Value::Bool(flag)) => *flag,
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
    if format == ComfyQuantFormat::Int8TensorwisePerRow && per_row != Some(true) {
        return Err(ComfyQuantDescriptorError::Int8NotPerRow);
    }
    for key in object.keys() {
        let known = matches!(key.as_str(), "format" | "full_precision_matrix_mult")
            || (key == "per_row" && format == ComfyQuantFormat::Int8TensorwisePerRow);
        if !known {
            return Err(ComfyQuantDescriptorError::UnknownField { field: key.clone() });
        }
    }
    Ok(ComfyQuantDescriptor {
        format,
        full_precision_matrix_mult,
    })
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

/// The swizzled block-scale tensor shape for a stored (padded) `[rows, cols]` MXFP8 weight:
/// `[⌈rows/128⌉·128, ⌈(cols/32)/4⌉·4]`. `cols` must already be a multiple of 32.
pub fn mxfp8_scale_shape(stored: [usize; 2]) -> [usize; 2] {
    let blocks = stored[1] / MXFP8_BLOCK;
    [
        stored[0].div_ceil(MXFP8_SCALE_ROW_TILE) * MXFP8_SCALE_ROW_TILE,
        blocks.div_ceil(MXFP8_SCALE_COL_TILE) * MXFP8_SCALE_COL_TILE,
    ]
}

/// Index into the cuBLAS-swizzled (`to_blocked`) scale buffer of the scale for `(row, block)`, where
/// the swizzled buffer is `mxfp8_scale_shape(stored)` row-major and `blocks = stored_cols / 32`.
///
/// `to_blocked` views the padded `[R, B]` scale matrix as `(R/128, 128, B/4, 4)`, permutes to
/// `(R/128, B/4, 128, 4)`, splits the 128 rows into `(4, 32)`, transposes those two, and flattens —
/// so element `(row, block)` lands at
/// `(((row/128)·(B/4) + block/4)·32 + row%32)·16 + ((row%128)/32)·4 + block%4`.
pub fn mxfp8_swizzled_scale_index(stored: [usize; 2], row: usize, block: usize) -> usize {
    let scale_shape = mxfp8_scale_shape(stored);
    let col_tiles = scale_shape[1] / MXFP8_SCALE_COL_TILE;
    let row_tile = row / MXFP8_SCALE_ROW_TILE;
    let row_in_tile = row % MXFP8_SCALE_ROW_TILE;
    (((row_tile * col_tiles + block / MXFP8_SCALE_COL_TILE) * 32) + row_in_tile % 32) * 16
        + (row_in_tile / 32) * 4
        + block % MXFP8_SCALE_COL_TILE
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
    fn descriptor_parser_accepts_the_four_formats_and_refuses_every_defect_by_name() {
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
                br#"{"format": "nvfp4"}"#,
                ComfyQuantDescriptorError::UnsupportedFormat {
                    format: "nvfp4".to_owned(),
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
