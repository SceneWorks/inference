//! Offline **W4A8 packer** and **CPU dequant reference** (SW-6134, feature SW-F287).
//!
//! Ported from comfy-kitchen's eager W4A8 weight preparation (Apache-2.0, Copyright (c) 2025
//! Comfy Org): `comfy_kitchen/backends/eager/w4a8_int8.py`
//! (`_quantize_rotated_w4a8_int8_weight`, `_assign_codes`, `_assign_grid`, `_grid_levels`,
//! `_dequant_int4_grouped_to_int8`) and `comfy_kitchen/tensor/w4a8_packed.py` (`_FIXED_LUT`,
//! `_pack_codes`). Only the frozen dialect this workspace executes is ported — 4-bit, symmetric,
//! the fixed Lloyd-Max codebook, group 16, ConvRot block 256, OCP E4M3FN relative scales — the same
//! dialect [`super::W4A8Int8::parse`] admits. See the crate `NOTICE`.
//!
//! # The format, as packed here
//!
//! For a floating weight `W [out, in]`:
//!
//! 1. **Rotate.** Every 256-column block of every row is multiplied by the regular (non-Walsh)
//!    Hadamard `R = H₄^{⊗4} / 16` ([`super::regular_hadamard`]) — computed as the pinned radix-4
//!    butterfly (strides `1, 4, 16, 64`, `0.5` per stage) in F32 by [`rotate_rows`]. `R` is a
//!    symmetric orthogonal involution, so the same transform un-rotates ([`decode_dense`]).
//! 2. **Group-quantize.** Each 16-column group `g` of the rotated row gets a float scale `s_g`:
//!    `s_g = max|g|`, codes = nearest codebook level of `g / s_g`, then two alternating
//!    least-squares refinements `s_g = Σ g·cb[q] / Σ cb[q]²` with re-assignment (upstream's
//!    `_ALS_ITERS = 2`). Both clamp at `1e-8` exactly as upstream.
//! 3. **Channel scale.** `s_channel[n] = max(max_c |cb[q]·s_g| / 127, 1e-8)` (F32).
//! 4. **Relative scale.** `s_rel = e4m3fn(s_g / s_channel)` — round-to-nearest-even to OCP E4M3FN
//!    ([`f32_to_e4m3fn`]). Upstream relies on `torch`'s cast, which produces NaN past ±448; this
//!    packer saturates to `448` instead so it can never emit a NaN scale (which the loader refuses).
//! 5. **Grid assignment.** The codes are re-chosen against the int8 values the GEMM will actually
//!    multiply, `levels = clamp(round_ties_even(cb · s_rel), ±127)`, nearest to `W / s_channel`
//!    (upstream's `_assign_grid`; ties resolve to the lower index).
//! 6. **Pack.** Two codes per byte, **even column in the low nibble** ([`pack_nibbles`]).
//!
//! Rows are independent (the codebook is fixed, not fitted per tensor), so a caller packing a
//! large tensor may split it by rows across threads and concatenate the results in row order.
//!
//! Deliberate differences from upstream, all outside the stored-format contract:
//! * The rotation runs in F32 for every input dtype (upstream rotates in the weight's own dtype,
//!   so a BF16 source is rotated in BF16 there). F32 is the more accurate choice; codes may
//!   differ from an upstream export of the same BF16 tensor where a value sits on a rounding edge.
//! * Upstream switches to a k-means-fitted codebook when the group-normalized rotated weights are
//!   heavy tailed (`_W4A8_GATE_KURTOSIS`, "real models never do"). This packer always uses the
//!   frozen Lloyd-Max table ([`DEFAULT_CODEBOOK`]) unless the caller supplies another sorted table,
//!   and the CUDA small-M decode path in upstream requires the frozen table anyway.
//! * Sums run sequentially in F32; upstream's reduction order is torch's. Neither is part of the
//!   format (any `s_g` decodes the same way).

use super::{CONVROT_GROUP, GROUP_SIZE};
use std::borrow::Cow;
use std::fmt;

/// The frozen 16-level **Lloyd-Max** codebook for a group-normalized Gaussian
/// (comfy-kitchen `tensor/w4a8_packed.py::_FIXED_LUT`). ConvRot makes every layer's rotated groups
/// near-Gaussian, which is why one table serves every projection. Sorted ascending — the nearest
/// level search ([`nearest_level`]) relies on it.
pub const DEFAULT_CODEBOOK: [f32; 16] = [
    -0.980602, -0.794529, -0.638165, -0.500986, -0.377321, -0.263187, -0.155210, -0.050720,
    0.052541, 0.156985, 0.265284, 0.379533, 0.502636, 0.638953, 0.794876, 0.980671,
];

/// ALS group-scale refinement passes (upstream `_ALS_ITERS`).
pub const ALS_ITERS: usize = 2;

/// Lower clamp on every float scale (upstream `clamp(min=1e-8)`).
pub const SCALE_FLOOR: f32 = 1e-8;

/// The canonical `.comfy_quant` payload of the frozen dialect — the exact bytes the published H3
/// W4A8 export carries (keys sorted, no whitespace).
pub const DESCRIPTOR_PAYLOAD: &[u8] =
    br#"{"convrot_groupsize":256,"format":"asym_w4a8_int8","group_size":16}"#;

/// Largest finite OCP E4M3FN magnitude.
const E4M3_MAX: f32 = 448.0;

/// Why a weight cannot be packed.
#[derive(Clone, Debug, PartialEq)]
pub enum W4A8PackError {
    /// `out_features` or `in_features` is zero, or `weight.len()` is not `out · in`.
    Shape {
        out_features: usize,
        in_features: usize,
        len: usize,
    },
    /// `in_features` is not a multiple of the 256-column rotation block.
    InFeaturesNotRotationMultiple { in_features: usize },
    /// The weight holds a NaN or an infinity.
    NonFinite,
    /// The codebook is not 16 finite, strictly ascending levels.
    Codebook,
}

impl fmt::Display for W4A8PackError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Shape {
                out_features,
                in_features,
                len,
            } => write!(
                f,
                "W4A8 pack: weight of {len} elements is not a nonempty [{out_features}, {in_features}]"
            ),
            Self::InFeaturesNotRotationMultiple { in_features } => write!(
                f,
                "W4A8 pack: in_features {in_features} is not a multiple of the ConvRot block {CONVROT_GROUP}"
            ),
            Self::NonFinite => write!(f, "W4A8 pack: the weight holds a non-finite value"),
            Self::Codebook => write!(
                f,
                "W4A8 pack: the codebook must be 16 finite, strictly ascending levels"
            ),
        }
    }
}

impl std::error::Error for W4A8PackError {}

/// One packed W4A8 projection: exactly the operand set the checkpoint stores.
#[derive(Clone, Debug, PartialEq)]
pub struct W4A8Packed {
    pub out_features: usize,
    pub in_features: usize,
    /// `[out, in/2]` two's-complement-agnostic bytes (stored as safetensors `I8`); even column in
    /// the low nibble.
    pub packed: Vec<u8>,
    /// `[out, in/16]` raw OCP E4M3FN bytes (stored as `F8_E4M3`).
    pub s_rel: Vec<u8>,
    /// `[out]` F32 per-output-channel scale.
    pub s_channel: Vec<f32>,
    /// The 16 F32 levels the codes index.
    pub codebook: [f32; 16],
}

/// One tensor of a packed projection in its exact checkpoint dtype and shape, ready for
/// `safetensors::serialize` / `serialize_to_file` (it implements [`safetensors::View`]).
#[derive(Clone, Debug, PartialEq)]
pub struct W4A8CheckpointTensor {
    pub name: String,
    pub dtype: safetensors::Dtype,
    pub shape: Vec<usize>,
    pub data: Vec<u8>,
}

impl W4A8CheckpointTensor {
    /// A tensor written **unchanged** beside the packed projections (a dense projection kept at
    /// higher precision, a bias, a norm scale, a modulation table), in its stored dtype. A tier
    /// builder reads it through the gen-core header layer, so the dtype arrives as gen-core's own
    /// [`crate::weightsmeta::Dtype`]; this converts it to the serializer's. `data` must be exactly
    /// `shape`'s element count at that width.
    pub fn passthrough(
        name: impl Into<String>,
        dtype: crate::weightsmeta::Dtype,
        shape: Vec<usize>,
        data: Vec<u8>,
    ) -> Result<Self, String> {
        let name = name.into();
        let dtype_name = dtype.as_str();
        let st_dtype: safetensors::Dtype = serde_json::from_value(serde_json::json!(dtype_name))
            .map_err(|_| format!("{name}: dtype {dtype_name} has no safetensors serializer"))?;
        let expected = shape
            .iter()
            .try_fold(dtype.size(), |acc, &dim| acc.checked_mul(dim));
        if expected != Some(data.len()) {
            return Err(format!(
                "{name}: {} payload bytes do not match {dtype_name} {shape:?}",
                data.len()
            ));
        }
        Ok(Self {
            name,
            dtype: st_dtype,
            shape,
            data,
        })
    }
}

impl safetensors::View for W4A8CheckpointTensor {
    fn dtype(&self) -> safetensors::Dtype {
        self.dtype
    }
    fn shape(&self) -> &[usize] {
        &self.shape
    }
    fn data(&self) -> Cow<'_, [u8]> {
        Cow::Borrowed(&self.data)
    }
    fn data_len(&self) -> usize {
        self.data.len()
    }
}

impl safetensors::View for &W4A8CheckpointTensor {
    fn dtype(&self) -> safetensors::Dtype {
        self.dtype
    }
    fn shape(&self) -> &[usize] {
        &self.shape
    }
    fn data(&self) -> Cow<'_, [u8]> {
        Cow::Borrowed(&self.data)
    }
    fn data_len(&self) -> usize {
        self.data.len()
    }
}

impl W4A8Packed {
    /// The validated descriptor geometry of this projection.
    pub fn descriptor(&self) -> super::W4A8Int8 {
        super::W4A8Int8 {
            out_features: self.out_features,
            in_features: self.in_features,
            group_size: GROUP_SIZE,
            rotation_group: CONVROT_GROUP,
        }
    }

    /// The projection's tensors under the ComfyUI on-disk names (`{base}.comfy_quant`,
    /// `{base}.weight`, `{base}.weight_codebook`, `{base}.weight_s_channel`,
    /// `{base}.weight_s_rel`) with their exact dtypes. A bias, when the layer has one, is the
    /// caller's to add as `{base}.bias` in its source dtype.
    pub fn checkpoint_tensors(&self, base: &str) -> Vec<W4A8CheckpointTensor> {
        use safetensors::Dtype;
        let f32_bytes = |v: &[f32]| v.iter().flat_map(|x| x.to_le_bytes()).collect::<Vec<u8>>();
        vec![
            W4A8CheckpointTensor {
                name: format!("{base}.comfy_quant"),
                dtype: Dtype::U8,
                shape: vec![DESCRIPTOR_PAYLOAD.len()],
                data: DESCRIPTOR_PAYLOAD.to_vec(),
            },
            W4A8CheckpointTensor {
                name: format!("{base}.weight"),
                dtype: Dtype::I8,
                shape: vec![self.out_features, self.in_features / 2],
                data: self.packed.clone(),
            },
            W4A8CheckpointTensor {
                name: format!("{base}.weight_codebook"),
                dtype: Dtype::F32,
                shape: vec![16],
                data: f32_bytes(&self.codebook),
            },
            W4A8CheckpointTensor {
                name: format!("{base}.weight_s_channel"),
                dtype: Dtype::F32,
                shape: vec![self.out_features],
                data: f32_bytes(&self.s_channel),
            },
            W4A8CheckpointTensor {
                name: format!("{base}.weight_s_rel"),
                dtype: Dtype::F8_E4M3,
                shape: vec![self.out_features, self.in_features / GROUP_SIZE],
                data: self.s_rel.clone(),
            },
        ]
    }

    /// Read one projection back from a safetensors file, validating it through the same
    /// [`super::W4A8Int8::parse`] / `validate_*` contract the loaders use. `Ok(None)` when `base`
    /// carries no W4A8 descriptor.
    pub fn from_safetensors(
        st: &safetensors::SafeTensors<'_>,
        base: &str,
    ) -> Result<Option<Self>, String> {
        use safetensors::Dtype;
        let get = |suffix: &str| {
            st.tensor(&format!("{base}.{suffix}"))
                .map_err(|e| format!("{base}.{suffix}: {e}"))
        };
        let Ok(payload) = st.tensor(&format!("{base}.comfy_quant")) else {
            return Ok(None);
        };
        let weight = get("weight")?;
        let codebook = get("weight_codebook")?;
        let s_channel = get("weight_s_channel")?;
        let s_rel = get("weight_s_rel")?;
        if payload.dtype() != Dtype::U8
            || weight.dtype() != Dtype::I8
            || codebook.dtype() != Dtype::F32
            || s_channel.dtype() != Dtype::F32
            || s_rel.dtype() != Dtype::F8_E4M3
        {
            return Err(format!(
                "{base}: W4A8 requires U8 descriptor, I8 weight, F32 codebook/s_channel and \
                 F8_E4M3 s_rel"
            ));
        }
        let d = super::W4A8Int8::parse(
            payload.data(),
            weight.shape(),
            s_rel.shape(),
            s_channel.shape(),
            codebook.shape(),
        )?;
        let f32s = |b: &[u8]| {
            b.chunks_exact(4)
                .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
                .collect::<Vec<f32>>()
        };
        let levels = f32s(codebook.data());
        let channels = f32s(s_channel.data());
        d.validate_s_rel(s_rel.data())?;
        d.validate_codebook(&levels)?;
        d.validate_s_channel(&channels)?;
        let mut book = [0f32; 16];
        book.copy_from_slice(&levels);
        Ok(Some(Self {
            out_features: d.out_features,
            in_features: d.in_features,
            packed: weight.data().to_vec(),
            s_rel: s_rel.data().to_vec(),
            s_channel: channels,
            codebook: book,
        }))
    }

    /// The grouped int8 weight the GEMM multiplies (`[out, in]`, still in the rotated basis).
    pub fn dequant_int8(&self) -> Vec<i8> {
        let packed: Vec<i8> = self.packed.iter().map(|&b| b as i8).collect();
        let mut out = Vec::new();
        super::dequant_packed_to_int8(
            &packed,
            &self.s_rel,
            &self.codebook,
            self.out_features,
            self.in_features,
            &mut out,
        )
        .expect("a W4A8Packed always has a consistent geometry");
        out
    }
}

/// Serialize checkpoint tensors to safetensors bytes (gen-core's `safetensors` version, so callers
/// on another version of the crate need not bridge `View` impls). Entries other than W4A8 operands
/// — a bias, dense tensors — may be included as [`W4A8CheckpointTensor`] values too.
pub fn serialize_checkpoint(tensors: &[W4A8CheckpointTensor]) -> Result<Vec<u8>, String> {
    safetensors::serialize(tensors.iter().map(|t| (t.name.clone(), t)), &None)
        .map_err(|e| format!("W4A8 checkpoint serialize: {e}"))
}

/// [`serialize_checkpoint`] straight to `path`.
pub fn write_checkpoint(
    tensors: &[W4A8CheckpointTensor],
    path: &std::path::Path,
) -> Result<(), String> {
    safetensors::serialize_to_file(tensors.iter().map(|t| (t.name.clone(), t)), &None, path)
        .map_err(|e| format!("{}: W4A8 checkpoint write: {e}", path.display()))
}

/// Encode an F32 to OCP **E4M3FN** with round-to-nearest-even, saturating at `±448` (never NaN;
/// NaN input maps to `0x7F`). The exact inverse of [`crate::comfy_quant::fp8_e4m3fn_to_f32`] on
/// every finite code.
pub fn f32_to_e4m3fn(x: f32) -> u8 {
    if x.is_nan() {
        return 0x7F;
    }
    let sign = if x.is_sign_negative() { 0x80u8 } else { 0 };
    let a = x.abs();
    if a >= E4M3_MAX {
        return sign | 0x7E;
    }
    // Subnormal range: steps of 2^-9 below the smallest normal 2^-6. Rounding up to 8 steps lands
    // exactly on code 0x08 (= 2^-6), so the integer step count IS the code.
    let min_normal = 2f32.powi(-6);
    if a < min_normal {
        let steps = (a * 512.0).round_ties_even() as u8; // exact: power-of-two scale
        return sign | steps;
    }
    // Normal: a = 2^e · (1 + m/8). `(e + 7) << 3 | q` is monotone in the value, so rounding the
    // mantissa up to q = 8 carries into the exponent field by plain integer addition.
    let e = ((a.to_bits() >> 23) & 0xFF) as i32 - 127;
    let q = ((a / 2f32.powi(e) - 1.0) * 8.0).round_ties_even() as i32; // exact scalings
    let code = (((e + 7) << 3) + q).min(0x7E);
    sign | code as u8
}

/// Rotate every `group`-column block of every `k`-wide row by the regular Hadamard `R`, in place,
/// as the pinned F32 radix-4 butterfly (strides `1, 4, 16, …`, each stage `0.5 ·`), the same
/// addition order as `candle_quant_kernels::convrot::convrot_fht_f32` and the activation kernel.
/// `R` is an involution: applying this twice restores the input (to rounding).
pub fn rotate_rows(data: &mut [f32], k: usize, group: usize) {
    debug_assert!(group.is_power_of_two() && group.trailing_zeros().is_multiple_of(2));
    debug_assert!(k.is_multiple_of(group) && data.len().is_multiple_of(k));
    let mut stride = 1usize;
    while stride < group {
        for blk in data.chunks_exact_mut(4 * stride) {
            for s in 0..stride {
                let (a, b, c, d) = (
                    blk[s],
                    blk[stride + s],
                    blk[2 * stride + s],
                    blk[3 * stride + s],
                );
                let ab = a + b;
                blk[s] = ((ab + c) - d) * 0.5;
                blk[stride + s] = ((ab - c) + d) * 0.5;
                blk[2 * stride + s] = (((a - b) + c) + d) * 0.5;
                blk[3 * stride + s] = (((-a + b) + c) + d) * 0.5;
            }
        }
        stride *= 4;
    }
}

/// Index of the level nearest `v` in an ascending table, with upstream's `searchsorted` (left)
/// neighbour rule: of the insertion point's two neighbours the upper wins only when strictly
/// closer, so ties resolve to the lower index.
pub fn nearest_level(levels: &[f32], v: f32) -> usize {
    let last = levels.len() - 1;
    let pos = levels.partition_point(|&l| l < v); // first index with level >= v
    let lo = pos.saturating_sub(1).min(last);
    let hi = pos.min(last);
    if (v - levels[hi]).abs() < (v - levels[lo]).abs() {
        hi
    } else {
        lo
    }
}

/// Pack `[n, k]` 4-bit codes (`0..=15`) two per byte, even column in the **low** nibble.
pub fn pack_nibbles(codes: &[u8]) -> Vec<u8> {
    debug_assert!(codes.len().is_multiple_of(2));
    codes
        .chunks_exact(2)
        .map(|p| (p[0] & 0xF) | ((p[1] & 0xF) << 4))
        .collect()
}

/// Inverse of [`pack_nibbles`].
pub fn unpack_nibbles(packed: &[u8]) -> Vec<u8> {
    packed.iter().flat_map(|&b| [b & 0xF, b >> 4]).collect()
}

/// Pack a row-major F32 weight `[out_features, in_features]` (the *original*, unrotated basis) with
/// the frozen [`DEFAULT_CODEBOOK`].
pub fn quantize_weight(
    weight: &[f32],
    out_features: usize,
    in_features: usize,
) -> Result<W4A8Packed, W4A8PackError> {
    quantize_weight_with_codebook(weight, out_features, in_features, &DEFAULT_CODEBOOK)
}

/// [`quantize_weight`] against a caller-chosen codebook (16 finite, strictly ascending levels).
pub fn quantize_weight_with_codebook(
    weight: &[f32],
    out_features: usize,
    in_features: usize,
    codebook: &[f32; 16],
) -> Result<W4A8Packed, W4A8PackError> {
    let (n, k) = (out_features, in_features);
    if n == 0 || k == 0 || n.checked_mul(k) != Some(weight.len()) {
        return Err(W4A8PackError::Shape {
            out_features,
            in_features,
            len: weight.len(),
        });
    }
    if !k.is_multiple_of(CONVROT_GROUP) {
        return Err(W4A8PackError::InFeaturesNotRotationMultiple { in_features });
    }
    if weight.iter().any(|v| !v.is_finite()) {
        return Err(W4A8PackError::NonFinite);
    }
    if codebook.iter().any(|l| !l.is_finite()) || codebook.windows(2).any(|w| w[0] >= w[1]) {
        return Err(W4A8PackError::Codebook);
    }
    let groups = k / GROUP_SIZE;
    let mut packed = Vec::with_capacity(n * k / 2);
    let mut s_rel = Vec::with_capacity(n * groups);
    let mut s_channel = Vec::with_capacity(n);
    let mut row = vec![0f32; k];
    let mut codes = vec![0u8; k];
    let mut scales = vec![0f32; groups];
    for r in 0..n {
        row.copy_from_slice(&weight[r * k..(r + 1) * k]);
        rotate_rows(&mut row, k, CONVROT_GROUP);
        // Steps 2-3: float group scales by max-abs init + ALS, then the channel scale.
        let mut peak = 0f32;
        for (g, (grp, scale)) in row
            .chunks_exact(GROUP_SIZE)
            .zip(scales.iter_mut())
            .enumerate()
        {
            let q = &mut codes[g * GROUP_SIZE..(g + 1) * GROUP_SIZE];
            let mut s = grp
                .iter()
                .fold(0f32, |m, v| m.max(v.abs()))
                .max(SCALE_FLOOR);
            assign(grp, s, codebook, q);
            for _ in 0..ALS_ITERS {
                let (mut num, mut den) = (0f32, 0f32);
                for (&w, &c) in grp.iter().zip(q.iter()) {
                    let l = codebook[c as usize];
                    num += w * l;
                    den += l * l;
                }
                s = (num / den.max(SCALE_FLOOR)).max(SCALE_FLOOR);
                assign(grp, s, codebook, q);
            }
            *scale = s;
            for &c in q.iter() {
                peak = peak.max((codebook[c as usize] * s).abs());
            }
        }
        let channel = (peak / 127.0).max(SCALE_FLOOR);
        s_channel.push(channel);
        // Steps 4-5: E4M3 relative scales, then codes re-chosen on the decoded int8 grid.
        for (g, &s) in scales.iter().enumerate() {
            let rel = f32_to_e4m3fn(s / channel);
            s_rel.push(rel);
            let rel = crate::comfy_quant::fp8_e4m3fn_to_f32(rel);
            let mut levels = [0f32; 16];
            for (l, &c) in levels.iter_mut().zip(codebook.iter()) {
                *l = (c * rel).round_ties_even().clamp(-127.0, 127.0);
            }
            for c in g * GROUP_SIZE..(g + 1) * GROUP_SIZE {
                codes[c] = nearest_level(&levels, row[c] / channel) as u8;
            }
        }
        packed.extend(pack_nibbles(&codes));
    }
    Ok(W4A8Packed {
        out_features: n,
        in_features: k,
        packed,
        s_rel,
        s_channel,
        codebook: *codebook,
    })
}

fn assign(group: &[f32], scale: f32, codebook: &[f32; 16], out: &mut [u8]) {
    for (q, &w) in out.iter_mut().zip(group) {
        *q = nearest_level(codebook, w / scale) as u8;
    }
}

/// The **CPU dequant reference**: the dense F32 weight `[out, in]` in the original basis that a
/// packed projection represents — the int8 grid value times `s_channel`, un-rotated by `R`
/// (upstream `dequantize_w4a8_int8_weight`). `x · decode_dense(p)ᵀ` is the reference linear the
/// native W4A8 forward is measured against.
pub fn decode_dense(p: &W4A8Packed) -> Vec<f32> {
    let k = p.in_features;
    let mut dense: Vec<f32> = p
        .dequant_int8()
        .chunks_exact(k)
        .zip(&p.s_channel)
        .flat_map(|(row, &s)| row.iter().map(move |&q| q as f32 * s))
        .collect();
    rotate_rows(&mut dense, k, CONVROT_GROUP);
    dense
}

#[cfg(test)]
mod tests;
