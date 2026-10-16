//! FLUX3-specific artifact preflight and independent CPU format oracles.
//!
//! Frozen BFL 45e57c9 `fp8.py`, `nvfp4.py`, `checkpoint.py` and pinned distilled
//! manifests define these layouts. FP8R is per-channel/per-token W8A8, NOT scalar
//! FP8. NVFP4 consumes a STATIC calibrated input scale, NOT a live tensor amax.
//! These functions do not open a device or claim a measured allocation bound.

use std::collections::{BTreeMap, BTreeSet};

use crate::weightsmeta::{Dtype, SafetensorsTensorHeader};
use crate::{Error, Result};

use super::DitConfig;

fn invalid(detail: impl std::fmt::Display) -> Error {
    Error::Unsupported(format!("flux3 quantization: {detail}"))
}

/// A source format, not a promise that every component of the model is quantized.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum DitFormat {
    Dense,
    Fp8Rowwise,
    Nvfp4,
}

/// Device facts supplied without opening a device during artifact inspection.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum QuantDevice {
    Cpu,
    Mlx,
    Cuda { major: u32, minor: u32 },
}

impl DitFormat {
    /// Parse the source `config.json.quantization` marker. Never infer a format from
    /// a filename or scale dtype: FP8R and NVFP4 both carry E4M3 scales/payloads.
    pub fn from_marker(marker: &serde_json::Value) -> Result<Self> {
        match marker {
            serde_json::Value::Null => Ok(Self::Dense),
            serde_json::Value::String(value) if value == "fp8r" => Ok(Self::Fp8Rowwise),
            serde_json::Value::String(value) if value == "nvfp4" => Ok(Self::Nvfp4),
            _ => Err(invalid(format!("unknown source format {marker}; install a pinned native package or explicitly convert the original dense artifact to a supported native tier"))),
        }
    }

    /// Necessary device gate only; the loader must ALSO prove kernel availability
    /// before staging weights. There is no dequantization or torch fallback.
    pub fn check_device(self, device: QuantDevice) -> Result<()> {
        let supported = match (self, device) {
            (Self::Dense, _) => true,
            (Self::Fp8Rowwise, QuantDevice::Cuda { major, minor }) => (major, minor) >= (8, 9),
            (Self::Nvfp4, QuantDevice::Cuda { major, .. }) => major >= 12,
            _ => false,
        };
        if supported {
            Ok(())
        } else {
            Err(invalid(format!("{self:?} has no source-equivalent packed kernel on {device:?}; use supported CUDA hardware or explicitly convert the ORIGINAL dense package to a native affine tier; no silent dense fallback")))
        }
    }
}

/// One required tensor, with the stored dtype (not its compute dtype).
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct TensorSpec {
    pub dtype: Dtype,
    pub shape: Vec<usize>,
}

impl TensorSpec {
    pub fn bytes(&self) -> Result<u64> {
        self.shape
            .iter()
            .try_fold(self.dtype.size() as u64, |n, &d| {
                n.checked_mul(d as u64)
                    .ok_or_else(|| invalid("tensor byte size overflow"))
            })
    }
}

/// Exact native key/shape inventory, including dense exclusions and q/k scales.
/// This uses FLUX3 names and graph structure, never a FLUX2 rename heuristic.
pub fn expected_tensors(
    cfg: &DitConfig,
    format: DitFormat,
) -> Result<BTreeMap<String, TensorSpec>> {
    if cfg.hidden == 0
        || cfg.heads == 0
        || !cfg.hidden.is_multiple_of(cfg.heads)
        || !cfg.mlp_ratio.is_finite()
        || cfg.mlp_ratio <= 0.
        || cfg.depth > 1024
        || cfg.depth_single > 1024
    {
        return Err(invalid("invalid DiT geometry"));
    }
    let h = cfg.hidden;
    let mh = cfg.mlp_hidden();
    let twice_mh = mh
        .checked_mul(2)
        .ok_or_else(|| invalid("MLP width overflow"))?;
    let mut matrices = BTreeMap::new();
    let mut insert = |name: String, out, input| {
        matrices.insert(name, vec![out, input]);
    };
    insert("emb_in.image".into(), h, cfg.image_in_channels);
    insert("emb_in.image_cond".into(), h, cfg.image_cond_in_channels);
    insert("txt_in".into(), h, cfg.text_in_channels);
    for prefix in ["time_in", "ref_time_in"] {
        if prefix == "ref_time_in" && !cfg.flow_map {
            continue;
        }
        insert(format!("{prefix}.in_layer"), h, 256);
        insert(format!("{prefix}.out_layer"), h, h);
    }
    for prefix in ["early_stream_modulations", "single_stream_modulations"] {
        for mode in ["image", "image_cond", "txt"] {
            insert(
                format!("{prefix}.{mode}.lin"),
                h.checked_mul(3)
                    .ok_or_else(|| invalid("modulation width overflow"))?,
                h,
            );
        }
    }
    let mut block_names = Vec::new();
    for (prefix, count) in [
        ("content_mode_blocks.image", cfg.depth),
        ("content_mode_blocks.image_cond", cfg.depth),
        ("txt_mode_blocks", cfg.depth),
        ("single_blocks", cfg.depth_single),
    ] {
        for i in 0..count {
            let prefix = format!("{prefix}.{i}");
            for leaf in ["q_proj", "k_proj", "v_proj", "attn_out"] {
                insert(format!("{prefix}.{leaf}"), h, h);
            }
            insert(format!("{prefix}.mlp_in"), twice_mh, h);
            insert(format!("{prefix}.mlp_out"), h, mh);
            block_names.push(prefix);
        }
    }
    insert("final_layer.image.linear".into(), cfg.image_in_channels, h);
    insert(
        "final_layer.image.adaLN_modulation.1".into(),
        h.checked_mul(2)
            .ok_or_else(|| invalid("final width overflow"))?,
        h,
    );
    let mut result = BTreeMap::new();
    for (name, shape) in matrices {
        let packed = is_source_quantized(&name, cfg, format);
        let n = shape[0];
        let k = shape[1];
        let dtype = if !packed {
            Dtype::BF16
        } else if format == DitFormat::Fp8Rowwise {
            Dtype::F8_E4M3
        } else {
            Dtype::U8
        };
        if packed && format == DitFormat::Nvfp4 && !k.is_multiple_of(16) {
            return Err(invalid(format!(
                "{name}: NVFP4 contraction width {k} is not divisible by 16"
            )));
        }
        result.insert(
            format!("{name}.weight"),
            TensorSpec {
                dtype,
                shape: if dtype == Dtype::U8 {
                    vec![n, k / 2]
                } else {
                    shape
                },
            },
        );
        if packed {
            let (dtype, shape) = if format == DitFormat::Fp8Rowwise {
                (Dtype::F32, vec![n])
            } else {
                (Dtype::F8_E4M3, vec![n, k / 16])
            };
            result.insert(format!("{name}.weight_scale"), TensorSpec { dtype, shape });
            if format == DitFormat::Nvfp4 {
                for suffix in ["weight_scale_2", "input_scale"] {
                    result.insert(
                        format!("{name}.{suffix}"),
                        TensorSpec {
                            dtype: Dtype::F32,
                            shape: vec![],
                        },
                    );
                }
            }
        }
    }
    for name in block_names {
        for norm in ["query_norm", "key_norm"] {
            result.insert(
                format!("{name}.norm.{norm}.scale"),
                TensorSpec {
                    dtype: Dtype::BF16,
                    shape: vec![cfg.head_dim()],
                },
            );
        }
    }
    Ok(result)
}

/// Source manifests' exact target/exclusion rule, including the asymmetric first
/// image block and last four shared blocks that NVFP4 intentionally leaves dense.
fn is_source_quantized(name: &str, cfg: &DitConfig, format: DitFormat) -> bool {
    if format == DitFormat::Dense {
        return false;
    }
    if name.starts_with("txt_mode_blocks.")
        || name.starts_with("emb_in.")
        || name.starts_with("final_layer.")
        || name.starts_with("time_in.")
        || name.starts_with("ref_time_in.")
        || name == "txt_in"
    {
        return false;
    }
    if format == DitFormat::Nvfp4 {
        if name.starts_with("early_stream_modulations.")
            || name.starts_with("single_stream_modulations.")
            || name.starts_with("content_mode_blocks.image.0.")
        {
            return false;
        }
        if let Some(rest) = name.strip_prefix("single_blocks.") {
            if let Some((index, _)) = rest.split_once('.') {
                if index
                    .parse::<usize>()
                    .is_ok_and(|i| i >= cfg.depth_single.saturating_sub(4))
                {
                    return false;
                }
            }
        }
    }
    true
}

/// All tensors must match exactly before payload allocation: missing scales,
/// duplicate keys, dense/base identity mismatches and extra tensors are errors.
pub fn validate_headers(
    cfg: &DitConfig,
    format: DitFormat,
    headers: &[SafetensorsTensorHeader],
) -> Result<()> {
    let expected = expected_tensors(cfg, format)?;
    let mut seen = BTreeSet::new();
    for header in headers {
        if !seen.insert(&header.name) {
            return Err(invalid(format!("duplicate tensor {}", header.name)));
        }
        let want = expected
            .get(&header.name)
            .ok_or_else(|| invalid(format!("unexpected tensor {}", header.name)))?;
        if want.dtype != header.dtype
            || want.shape != header.shape
            || want.bytes()? != header.data_bytes
        {
            return Err(invalid(format!(
                "{}: expected {:?} {:?} ({} bytes), got {:?} {:?} ({} bytes)",
                header.name,
                want.dtype,
                want.shape,
                want.bytes()?,
                header.dtype,
                header.shape,
                header.data_bytes
            )));
        }
    }
    if let Some(missing) = expected.keys().find(|name| !seen.contains(name)) {
        return Err(invalid(format!("missing tensor {missing}")));
    }
    Ok(())
}

/// E4M3FN finite decoding (0x7f/0xff are NaN). Signed zero is retained.
pub fn decode_e4m3(code: u8) -> f32 {
    let mag = code & 127;
    let exponent = mag >> 3;
    let mantissa = mag & 7;
    let value = if mag == 127 {
        f32::NAN
    } else if exponent == 0 {
        (mantissa as f32) * 2f32.powi(-9)
    } else {
        (1. + mantissa as f32 / 8.) * 2f32.powi(exponent as i32 - 7)
    };
    if code & 128 != 0 {
        -value
    } else {
        value
    }
}

/// Saturating round-to-nearest-even E4M3FN, independent of a tensor backend.
pub fn encode_e4m3(value: f32) -> Result<u8> {
    if !value.is_finite() {
        return Err(invalid("non-finite E4M3 input"));
    }
    let sign = if value.is_sign_negative() { 128 } else { 0 };
    let value = value.abs().min(448.);
    let mut best = 0u8;
    let mut distance = f32::INFINITY;
    for code in 0..=126u8 {
        let d = (decode_e4m3(code) - value).abs();
        if d < distance || (d == distance && code & 1 == 0) {
            best = code;
            distance = d;
        }
    }
    Ok(sign | best)
}

/// Source `quantize_rows`: max(abs(row))/448, clamped at 1e-12.
pub fn quantize_fp8_rows(values: &[f32], width: usize) -> Result<(Vec<u8>, Vec<f32>)> {
    if width == 0 || !values.len().is_multiple_of(width) {
        return Err(invalid("invalid FP8 row geometry"));
    }
    let mut codes = Vec::with_capacity(values.len());
    let mut scales = Vec::with_capacity(values.len() / width);
    for row in values.chunks_exact(width) {
        if row.iter().any(|v| !v.is_finite()) {
            return Err(invalid("non-finite FP8 input"));
        }
        let scale = (row.iter().fold(0f32, |m, v| m.max(v.abs())) / 448.).max(1e-12);
        scales.push(scale);
        for &v in row {
            codes.push(encode_e4m3(v / scale)?);
        }
    }
    Ok((codes, scales))
}

pub fn dequantize_fp8_rows(codes: &[u8], scales: &[f32], width: usize) -> Result<Vec<f32>> {
    if width == 0 || scales.len().checked_mul(width) != Some(codes.len()) {
        return Err(invalid("invalid FP8 codes/scales geometry"));
    }
    let mut out = Vec::with_capacity(codes.len());
    for (row, &scale) in codes.chunks_exact(width).zip(scales) {
        if !scale.is_finite() || scale <= 0. {
            return Err(invalid("invalid FP8 row scale"));
        }
        for &code in row {
            let value = decode_e4m3(code);
            if !value.is_finite() {
                return Err(invalid("non-finite FP8 code"));
            }
            out.push(value * scale);
        }
    }
    Ok(out)
}

fn round_up(value: usize, alignment: usize) -> Result<usize> {
    value
        .checked_add(alignment - 1)
        .map(|n| n / alignment * alignment)
        .ok_or_else(|| invalid("padded geometry overflow"))
}

/// CUTLASS/FlashInfer 128x4 block-scale atom layout; M-atoms are fastest.
pub fn scale_offset(row: usize, block: usize, padded_rows: usize) -> Result<usize> {
    if padded_rows == 0 || !padded_rows.is_multiple_of(128) || row >= padded_rows {
        return Err(invalid("invalid scale row"));
    }
    let atom = (block / 4)
        .checked_mul(padded_rows / 128)
        .and_then(|a| a.checked_add(row / 128))
        .ok_or_else(|| invalid("scale atom overflow"))?;
    atom.checked_mul(512)
        .and_then(|a| a.checked_add((row % 32) * 16 + (row % 128 / 32) * 4 + block % 4))
        .ok_or_else(|| invalid("scale offset overflow"))
}

pub fn swizzle_scales(values: &[u8], rows: usize, blocks: usize) -> Result<Vec<u8>> {
    if rows.checked_mul(blocks) != Some(values.len()) {
        return Err(invalid("invalid block-scale inventory"));
    }
    let padded_rows = round_up(rows, 128)?;
    let len = padded_rows
        .checked_mul(round_up(blocks, 4)?)
        .ok_or_else(|| invalid("scale allocation overflow"))?;
    let mut result = vec![0; len];
    for r in 0..rows {
        for b in 0..blocks {
            result[scale_offset(r, b, padded_rows)?] = values[r * blocks + b];
        }
    }
    Ok(result)
}

/// CPU oracle output; scales are padded/swizzled, global_scale is STATIC.
#[derive(Clone, Debug)]
pub struct StaticNvfp4 {
    pub packed: Vec<u8>,
    pub scales: Vec<u8>,
    pub global_scale: f32,
}

const E2M1: [f32; 8] = [0., 0.5, 1., 1.5, 2., 3., 4., 6.];
fn encode_e2m1(value: f32) -> u8 {
    let mut best = 0usize;
    let mut distance = f32::INFINITY;
    for (i, &candidate) in E2M1.iter().enumerate() {
        let d = (candidate - value.abs()).abs();
        if d < distance || (d == distance && i % 2 == 0) {
            best = i;
            distance = d;
        }
    }
    best as u8 | if value.is_sign_negative() { 8 } else { 0 }
}

/// Static-scale activation oracle. The *weight* converter's MSE scale sweep is a
/// different algorithm; never use this function to regenerate published weights.
pub fn quantize_nvfp4_static(
    values: &[f32],
    width: usize,
    input_scale: f32,
) -> Result<StaticNvfp4> {
    if width == 0
        || !width.is_multiple_of(16)
        || !values.len().is_multiple_of(width)
        || !input_scale.is_finite()
        || input_scale <= 0.
        || values.iter().any(|v| !v.is_finite())
    {
        return Err(invalid(
            "invalid static NVFP4 geometry, input scale or value",
        ));
    }
    let global_scale = input_scale.max(1e-8);
    let rows = values.len() / width;
    let mut packed = vec![0; values.len() / 2];
    let mut scales = Vec::with_capacity(values.len() / 16);
    for (block_index, block) in values.chunks_exact(16).enumerate() {
        let amax = block.iter().fold(0f32, |m, v| m.max(v.abs()));
        let code = encode_e4m3(amax / (6. * global_scale))?;
        scales.push(code);
        let divisor = decode_e4m3(code) * global_scale;
        for (i, &v) in block.iter().enumerate() {
            let q = if divisor == 0. {
                0
            } else {
                encode_e2m1(v / divisor)
            };
            packed[block_index * 8 + i / 2] |= q << ((i % 2) * 4);
        }
    }
    Ok(StaticNvfp4 {
        packed,
        scales: swizzle_scales(&scales, rows, width / 16)?,
        global_scale,
    })
}

#[cfg(test)]
#[path = "quantization_tests.rs"]
mod tests;
