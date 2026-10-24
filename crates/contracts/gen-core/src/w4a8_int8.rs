//! Native packed **W4A8 (asym_w4a8_int8) projection contract and bit-accurate host reference**
//! (sc-26685).
//!
//! This is deliberately separate from both the plain `int8-per-row-v1` dense decoder and the
//! `convrot-int8-native` codec (`crate::convrot_int8`): the stored weight is **two int4 codes per
//! I8 byte**, indexed through a 16-level Lloyd-Max codebook, a per-group fp8 (E4M3FN) scale, and an
//! F32 per-output-channel scale left to the int8 GEMM epilogue. The activation side is the same
//! fused ConvRot (regular Hadamard, block 256) + row-wise int8 quantization as the ConvRot INT8
//! codec; only the weight half — packed codes, codebook, `s_rel`/`s_channel` — is new.
//!
//! # Authority
//!
//! Comfy Kitchen PR#90, tree `06a261f8` (head `d908d589`) — pinned in the epic source-of-truth:
//!
//! * `comfy_kitchen/tensor/w4a8_int8.py` — the export dialect and the eager reference;
//! * `comfy_kitchen/backends/cuda/ops/w4a8_gemm.cu` — `dequant_int4_grouped_to_int8_kernel`, the
//!   memory-bound int4→int8 weight dequant: `int8_w = __float2int_rn(cb[q] · s_rel) clamp [-127,127]`
//!   (codebook) / `(q−8)·s_rel` (uniform), fp8 scale decoded with `__NV_E4M3`;
//! * `comfy_kitchen/backends/cuda/ops/cutlass_gemm_int8.cu` — the fused int32→float epilogue
//!   `((acc · xs) · s_channel) + bias`, each float op round-to-nearest, store round-to-nearest;
//! * `comfy_kitchen/backends/cuda/ops/int8_linear.cu` — `quantize_int8_rowwise_convrot_kernel`, the
//!   f32 Fast-Hadamard rotation + row-wise int8 activation quantizer with the bf16 cast chain, and
//!   `dequantize_int8_linear_kernel`, the same epilogue as `cutlass_gemm_int8.cu`.
//!
//! The reference functions in this module encode those kernels' **exact** rounding and fold order,
//! in plain `f32`/`i8` — no tensor dependency — so both native backends' fixture tests compare
//! against the same bit-accurate oracle rather than a re-derived approximation.

use serde::Deserialize;

/// Stable source/storage and native execution identity of this validated operand format.
///
/// Consumers may persist this identity only after validating the descriptor and operands with
/// [`W4A8Int8`]. It is distinct from `convrot-int8-native` (S4) and `int8-per-row-v1`: reporting
/// either for these packed codebook codes would describe the wrong storage and weight arithmetic.
pub const CODEC_ID: &str = "w4a8-int8-native";

/// Maximum descriptor payload; checked before copying a descriptor off a device.
pub const MAX_DESCRIPTOR_BYTES: usize = 4096;

/// The frozen dialect's fixed weight-group length. Two int4 codes share one stored byte, eight
/// groups span the 16-column vectorized dequant width, and this exact value is a property of the
/// published format the exported checkpoint declares (`{"group_size":16}`).
pub const GROUP_SIZE: usize = 16;

/// The frozen dialect's fixed ConvRot rotation group (`{"convrot_groupsize":256}`): a regular
/// (non-Walsh) Hadamard block, `H₄` Kronecker-power of order 256, normalized by `1/16`.
pub const CONVROT_GROUP: usize = 256;

/// The regular `H₄` Hadamard block. **Sign order is load-bearing**: this is the PR#90 "regular"
/// order, not the Sylvester/Walsh order, and the published checkpoint was quantized against exactly
/// these signs.
pub const H4: [[f32; 4]; 4] = [
    [1.0, 1.0, 1.0, -1.0],
    [1.0, 1.0, -1.0, 1.0],
    [1.0, -1.0, 1.0, 1.0],
    [-1.0, 1.0, 1.0, 1.0],
];

/// The largest finite bf16 value (`3.38953139e38`), bit-exact — the `finite_max_for_dtype<bf16>`
/// clamp the pinned activation quantizer applies to the row absmax before the `1/127` scale. This
/// is the H3 DiT's activation dtype, so it is the `dtype_max` the W4A8 oracle and fixtures pass to
/// [`quantize_rotated_row`] / [`reference_forward`].
pub const BF16_MAX: f32 = f32::from_bits(0x7F7F_0000);

/// Validated logical geometry. The stored weight is `I8 [out, in/2]` (two uint4 codes per byte,
/// even column in the **low** nibble); `in_features` is the logical (dequantized-int8) width.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct W4A8Int8 {
    pub out_features: usize,
    pub in_features: usize,
    pub group_size: usize,
    pub rotation_group: usize,
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct Descriptor {
    format: String,
    group_size: usize,
    convrot_groupsize: usize,
}

impl W4A8Int8 {
    /// Parse the authoritative U8 `.comfy_quant` payload and validate its operand shapes.
    ///
    /// * `weight` — the packed `I8` weight shape `[out, in/2]`;
    /// * `s_rel` — the `F8_E4M3` per-group relative scale shape `[out, in/GROUP_SIZE]`;
    /// * `s_channel` — the `F32` per-output-channel scale shape `[out]`;
    /// * `codebook` — the `F32` `[16]` level table (required: the frozen dialect is symmetric +
    ///   codebook; a uniform `(q−8)` dialect is a different codec and is refused here rather than
    ///   decoded under the wrong table).
    ///
    /// Only the actual frozen dialect is admitted: `{"format":"asym_w4a8_int8","group_size":16,
    /// "convrot_groupsize":256}`. Any other group/rotation size, a `convrot` bool, an absent
    /// codebook, or an unknown field refuses through `serde(deny_unknown_fields)`.
    pub fn parse(
        payload: &[u8],
        weight: &[usize],
        s_rel: &[usize],
        s_channel: &[usize],
        codebook: &[usize],
    ) -> Result<Self, String> {
        if payload.is_empty() || payload.len() > MAX_DESCRIPTOR_BYTES {
            return Err("W4A8 descriptor length is invalid".into());
        }
        let d: Descriptor =
            serde_json::from_slice(payload).map_err(|e| format!("W4A8 descriptor: {e}"))?;
        if d.format != "asym_w4a8_int8" {
            return Err("expected format asym_w4a8_int8".into());
        }
        if d.group_size != GROUP_SIZE {
            return Err(format!(
                "W4A8 group_size must be {GROUP_SIZE}, got {}",
                d.group_size
            ));
        }
        if d.convrot_groupsize != CONVROT_GROUP {
            return Err(format!(
                "W4A8 convrot_groupsize must be {CONVROT_GROUP}, got {}",
                d.convrot_groupsize
            ));
        }
        if weight.len() != 2 || weight.contains(&0) || !weight[1].is_multiple_of(2) {
            return Err("W4A8 weight must be nonempty [out, in/2] with an even byte width".into());
        }
        let (n, k_half) = (weight[0], weight[1]);
        let k = k_half * 2;
        if !k.is_multiple_of(CONVROT_GROUP) {
            return Err("W4A8 logical input must be a multiple of the rotation group".into());
        }
        if k % GROUP_SIZE != 0 {
            return Err("W4A8 logical input must be a multiple of the group size".into());
        }
        if s_rel != [n, k / GROUP_SIZE] {
            return Err("W4A8 weight_s_rel must be F8_E4M3 [out, in/group]".into());
        }
        if s_channel != [n] {
            return Err("W4A8 weight_s_channel must be F32 [out]".into());
        }
        if codebook != [16] {
            return Err("W4A8 codebook must be F32 [16] (the symmetric codebook dialect)".into());
        }
        if n.checked_mul(k_half).is_none() || n > i32::MAX as usize || k > i32::MAX as usize {
            return Err("W4A8 integer GEMM geometry overflow".into());
        }
        Ok(Self {
            out_features: n,
            in_features: k,
            group_size: d.group_size,
            rotation_group: d.convrot_groupsize,
        })
    }

    /// Value checks shared by both native backends, before a projection is executable.
    ///
    /// Unlike the ConvRot INT8 codec's strict-positive row-scale validator, the per-**group**
    /// F8 scale is allowed to be **zero**: the export clamps the source `s_g`/`s_channel` to
    /// `1e-8`, but the e4m3 cast of a vanishingly small quotient underflows to a subnormal or zero
    /// scale, which a zero-weight group carries legitimately. NaN and negative scales are still
    /// refused — the converter only ever emits a non-negative relative scale.
    pub fn validate_s_rel(&self, bytes: &[u8]) -> Result<(), String> {
        if bytes.len() != self.out_features * (self.in_features / GROUP_SIZE) {
            return Err("W4A8 s_rel byte count does not match the geometry".into());
        }
        for &b in bytes {
            let v = crate::comfy_quant::fp8_e4m3fn_to_f32(b);
            if v.is_nan() || v < 0.0 {
                return Err("W4A8 weight_s_rel must be non-negative finite E4M3FN".into());
            }
        }
        Ok(())
    }

    /// Per-output-channel scales must be finite and strictly positive.
    pub fn validate_s_channel(&self, scales: &[f32]) -> Result<(), String> {
        if scales.len() != self.out_features {
            return Err("W4A8 weight_s_channel length does not match the geometry".into());
        }
        for s in scales {
            if !s.is_finite() || *s <= 0.0 {
                return Err("W4A8 weight_s_channel must be finite positive F32".into());
            }
        }
        Ok(())
    }

    /// The 16 codebook levels must all be finite (they are signed Lloyd-Max levels; zero is a
    /// legitimate middle entry, and no positivity constraint applies).
    pub fn validate_codebook(&self, levels: &[f32]) -> Result<(), String> {
        if levels.len() != 16 {
            return Err("W4A8 codebook must have 16 levels".into());
        }
        if levels.iter().any(|l| !l.is_finite()) {
            return Err("W4A8 codebook levels must be finite F32".into());
        }
        Ok(())
    }

    /// Resident bytes: the packed `I8` codes plus the three scale/codebook companions. **No dense
    /// int8 or float weight is resident** — the int4→int8 dequant target is a transient per-call
    /// (or per-chunk) scratch buffer, which is the whole point of the W4A8 residency claim.
    pub fn resident_bytes(&self) -> usize {
        self.out_features * (self.in_features / 2)                             // packed I8
            + self.out_features * (self.in_features / GROUP_SIZE)              // s_rel F8
            + self.out_features * 4                                            // s_channel F32
            + 16 * 4 // codebook F32
    }
}

// =================================================================================================
// Bit-accurate host reference (the fixture oracle).
// =================================================================================================

/// Round `v` (finite) to the nearest `f32` representable as bf16, ties **to even** — the exact
/// semantics of CUDA `__float2bfloat16_rn` and MLX `as_dtype(BF16)`, and therefore of every
/// bf16 cast in the pinned kernels' activation epilogue.
pub fn bf16_rn(v: f32) -> f32 {
    if v.is_nan() || v.is_infinite() {
        return v;
    }
    let bits = v.to_bits();
    // bf16 keeps the top 16 bits (1 sign + 8 exp + 7 mantissa), dropping the low 16 mantissa bits.
    // Round-half-to-even on the dropped bits: add half-ulp (0x7FFF) plus the kept mantissa's LSB,
    // then truncate the low 16 bits.
    let rounded = bits.wrapping_add(0x0000_7FFF + ((bits >> 16) & 1)) & 0xFFFF_0000;
    f32::from_bits(rounded)
}

/// The dense regular-Hadamard rotation of `order` (`order` must be a power of 4), normalized by
/// `1/√order`. Produced exactly as the native loaders build it: a Kronecker power of [`H4`] with
/// per-element sign `∏ H4[i_d][j_d]`, scaled by `1/√order`.
///
/// This is the *reference* matrix; the fused CUDA activation kernel factors it as a radix-4 Fast
/// Hadamard Transform ([`fht_rotate_row`]) that computes the same product in a different summation
/// order.
pub fn regular_hadamard(order: usize) -> Vec<f32> {
    debug_assert!(order == 1 || (order.is_power_of_two() && order.is_multiple_of(4)));
    let mut out = Vec::with_capacity(order * order);
    for idx in 0..order * order {
        let (mut r, mut c) = (idx / order, idx % order);
        let mut s = 1.0f32;
        while r != 0 || c != 0 {
            s *= H4[r % 4][c % 4];
            r /= 4;
            c /= 4;
        }
        out.push(s / (order as f32).sqrt());
    }
    out
}

/// The four rows of [`H4`] dotted with `(x0, x1, x2, x3)` — the exact radix-4 butterfly used by the
/// pinned `quantize_int8_rowwise_convrot_kernel`, with C's left-to-right float addition order.
fn h4_row_dot(d: usize, x0: f32, x1: f32, x2: f32, x3: f32) -> f32 {
    match d {
        0 => x0 + x1 + x2 - x3,
        1 => x0 + x1 - x2 + x3,
        2 => x0 - x1 + x2 + x3,
        _ => -x0 + x1 + x2 + x3,
    }
}

/// The fused ConvRot/quantizer's f32 Fast-Hadamard butterfly over one `CONVROT_GROUP` block.
/// Four stages (strides 1, 4, 16, 64) each scaled by `1/2`, ping-ponging between `a` and `b`; an
/// even number of stages leaves the result in `a`. Matches
/// `quantize_int8_rowwise_convrot_kernel`'s arithmetic order bit-for-bit.
fn fht_block(a: &mut [f32; CONVROT_GROUP], b: &mut [f32; CONVROT_GROUP]) {
    let mut read = 0usize; // 0 = `a`, 1 = `b`
    let mut write = 1usize;
    for stage in 0..4usize {
        let s = [1usize, 4, 16, 64][stage];
        let (src, dst): (&[f32], &mut [f32]) = if read == 0 {
            (&a[..], &mut b[..])
        } else {
            (&b[..], &mut a[..])
        };
        for (i, dst_i) in dst.iter_mut().enumerate() {
            let d = (i / s) & 3;
            let base = i - d * s;
            *dst_i = 0.5
                * h4_row_dot(
                    d,
                    src[base],
                    src[base + s],
                    src[base + 2 * s],
                    src[base + 3 * s],
                );
        }
        std::mem::swap(&mut read, &mut write);
    }
    debug_assert_eq!(read, 0, "four stages must land back in the first buffer");
}

/// Rotate one row (`k` elements, `k` a multiple of `CONVROT_GROUP`) by the block-diagonal regular
/// Hadamard, in f32, using the FHT factorization above — the CUDA activation kernel's rotation.
pub fn fht_rotate_row(x: &[f32], k: usize, out: &mut Vec<f32>) {
    out.clear();
    out.resize(k, 0.0);
    let mut a = [0.0f32; CONVROT_GROUP];
    let mut b = [0.0f32; CONVROT_GROUP];
    for block in (0..k).step_by(CONVROT_GROUP) {
        a.copy_from_slice(&x[block..block + CONVROT_GROUP]);
        fht_block(&mut a, &mut b);
        out[block..block + CONVROT_GROUP].copy_from_slice(&a);
    }
}

/// Quantize one already-rotated f32 row to int8 codes + a per-row scale, replicating the pinned
/// `quantize_int8_rowwise_convrot_kernel`'s post-FHT tail exactly:
///
/// ```text
/// scale   = max(min(absmax(rot), bf16_max) / 127, 1e-30)
/// scaled  = f32( bf16( f32(bf16(rot)) / f32(bf16(scale)) ) )
/// code    = clamp( round_ties_even(scaled), -128, 127 )
/// ```
///
/// `dtype_max` is [`BF16_MAX`] for a bf16 activation (the H3 DiT's compute dtype); pass
/// `f32::MAX`/`65504.0` for the other two activation dtypes the codec admits.
pub fn quantize_rotated_row(
    rotated: &[f32],
    dtype_max: f32,
    out_codes: &mut Vec<i8>,
    out_scale: &mut f32,
) {
    let mut abs_max = 0.0f32;
    for &v in rotated {
        let a = v.abs();
        if !a.is_nan() && a > abs_max {
            abs_max = a;
        }
    }
    let scale = (abs_max.min(dtype_max) * (1.0 / 127.0)).max(1.0e-30);
    let scale_bf16 = bf16_rn(scale);
    out_codes.clear();
    out_codes.reserve(rotated.len());
    for &v in rotated {
        let t = bf16_rn(v) / scale_bf16;
        let scaled = bf16_rn(t);
        let q = if scaled.is_nan() {
            -128.0f32
        } else {
            scaled.round_ties_even().clamp(-128.0, 127.0)
        };
        out_codes.push(q as i8);
    }
    *out_scale = scale;
}

/// Decode the packed int4 weight into the grouped int8 the GEMM actually multiplies, replicating
/// `dequant_int4_grouped_to_int8_kernel`:
///
/// ```text
/// byte  = packed[n, c/2]            (even col -> low nibble, odd -> high nibble)
/// lvl   = codebook[nibble]          (the frozen codebook dialect)
/// s     = e4m3_to_f32(s_rel[n, c/16])
/// int8  = round_ties_even(lvl · s)  clamped to [-127, 127]
/// ```
///
/// The group/convrot rotation is **not** applied here — the reference forward rotates the
/// activation ([`fht_rotate_row`]), matching the native path, and this returns the rotated-basis
/// int8 weight the GEMM consumes. `packed` is `[out, in/2]`, `s_rel` `[out, in/16]` raw E4M3 bytes.
pub fn dequant_packed_to_int8(
    packed: &[i8],
    s_rel: &[u8],
    codebook: &[f32],
    out_features: usize,
    in_features: usize,
    out: &mut Vec<i8>,
) -> Result<(), String> {
    if codebook.len() != 16 {
        return Err("W4A8 codebook must have 16 levels".into());
    }
    if packed.len() != out_features * (in_features / 2) {
        return Err("W4A8 packed byte count does not match the geometry".into());
    }
    if s_rel.len() != out_features * (in_features / GROUP_SIZE) {
        return Err("W4A8 s_rel byte count does not match the geometry".into());
    }
    out.clear();
    out.reserve(out_features * in_features);
    let k_half = in_features / 2;
    let groups = in_features / GROUP_SIZE;
    for n in 0..out_features {
        let srow = n * groups;
        for c in 0..in_features {
            let byte = packed[n * k_half + c / 2] as u8;
            let nibble = if c % 2 == 0 {
                byte & 0xF
            } else {
                (byte >> 4) & 0xF
            };
            let lvl = codebook[nibble as usize];
            let s = crate::comfy_quant::fp8_e4m3fn_to_f32(s_rel[srow + c / GROUP_SIZE]);
            let v = (lvl * s).round_ties_even().clamp(-127.0, 127.0);
            out.push(v as i8);
        }
    }
    Ok(())
}

/// The complete bit-accurate reference forward for one packed W4A8 projection — the frozen
/// symmetric+codebook dialect — matching the pinned kernels end to end:
///
/// ```text
/// xr[m,:]    = FHT_rotate(x[m,:])                       f32 butterfly
/// xq[m,:],xs = quantize_rotated_row(xr)                 round_ties_even, clamp [-128,127]
/// wq[n,:]    = dequant_packed_to_int8(weight)           round_ties_even, clamp [-127,127]
/// acc        = Σ xq[m,k]·wq[n,k]                        exact int32
/// out[m,n]   = bf16_rn( (f32(acc)·xs[m])·s_channel[n] + bias[n] )
/// ```
///
/// `x` is the bf16 activation widened to f32 (each value already exactly representable), row-major
/// `[m, in_features]`; `dtype_max` is the activation dtype's finite maximum. The result is
/// row-major `[m, out_features]` f32, **in the same bf16 precision the native path stores** — this
/// function returns the bf16-rounded values, which is what the fixtures compare element-for-element.
/// The 10-argument shape is intentional: this is a tensor-free oracle consumed by both backend
/// test crates, and every operand is a raw primitive slice rather than a backend tensor.
#[allow(clippy::too_many_arguments)]
pub fn reference_forward(
    x: &[f32],
    packed: &[i8],
    s_rel: &[u8],
    s_channel: &[f32],
    codebook: &[f32],
    bias: Option<&[f32]>,
    m: usize,
    out_features: usize,
    in_features: usize,
    dtype_max: f32,
) -> Result<Vec<f32>, String> {
    if x.len() != m * in_features {
        return Err("W4A8 reference: activation length mismatch".into());
    }
    if s_channel.len() != out_features {
        return Err("W4A8 reference: s_channel length mismatch".into());
    }
    if let Some(b) = bias {
        if b.len() != out_features {
            return Err("W4A8 reference: bias length mismatch".into());
        }
    }
    let mut wq = Vec::new();
    dequant_packed_to_int8(packed, s_rel, codebook, out_features, in_features, &mut wq)?;

    let mut out = vec![0.0f32; m * out_features];
    let mut xq = Vec::new();
    let mut rotated = Vec::new();
    for row in 0..m {
        fht_rotate_row(
            &x[row * in_features..(row + 1) * in_features],
            in_features,
            &mut rotated,
        );
        let mut scale = 0.0f32;
        quantize_rotated_row(&rotated, dtype_max, &mut xq, &mut scale);
        for n in 0..out_features {
            let mut acc = 0i32;
            for k in 0..in_features {
                acc += i32::from(xq[k]) * i32::from(wq[n * in_features + k]);
            }
            let mut v = acc as f32 * scale * s_channel[n];
            if let Some(b) = bias {
                v += b[n];
            }
            out[row * out_features + n] = bf16_rn(v);
        }
    }
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn actual_h3_w4a8_descriptor_and_residency() {
        let payload = r#"{"convrot_groupsize":256,"format":"asym_w4a8_int8","group_size":16}"#;
        // out=5376, in=7168 (the published attn to_out / qkv width): packed [5376, 3584].
        let d = W4A8Int8::parse(
            payload.as_bytes(),
            &[5376, 3584],
            &[5376, 448],
            &[5376],
            &[16],
        )
        .unwrap();
        assert_eq!(d.in_features, 7168);
        assert_eq!(d.group_size, 16);
        assert_eq!(d.rotation_group, 256);
        assert_eq!(d.resident_bytes(), 5376 * 3584 + 5376 * 448 + 5376 * 4 + 64);
        d.validate_s_channel(&[1e-30; 5376]).unwrap();
        assert!(d.validate_s_rel(&[0u8, 0x24, 0x3C, 0x7E]).is_err());
    }

    #[test]
    fn refuses_malformed_contradictory_and_partial_metadata() {
        let good = r#"{"convrot_groupsize":256,"format":"asym_w4a8_int8","group_size":16}"#;
        for bad in [
            good.replace("256", "64"),
            good.replace("\"group_size\":16", "\"group_size\":8"),
            good.replace("asym_w4a8_int8", "int8_tensorwise"),
            good.replace("asym_w4a8_int8", "asym_w4a8_int8x"),
            // a `convrot` bool is not part of the W4A8 dialect and must refuse.
            good.replace("256,", "256,\"convrot\":true,"),
            "{}".to_string(),
            "[]".to_string(),
        ] {
            assert!(
                W4A8Int8::parse(bad.as_bytes(), &[16, 128], &[16, 16], &[16], &[16]).is_err(),
                "must refuse {bad}"
            );
        }
        // Shapes that agree on `format` but not the geometry.
        for (w, s, ch, cb) in [
            (&[16, 64][..], &[16, 16][..], &[16][..], &[16][..]), // in = 128, s_rel [16,8]
            (&[16, 128][..], &[16, 8][..], &[16][..], &[16][..]),
            (&[16, 128][..], &[16, 16][..], &[16, 1][..], &[16][..]),
            (&[16, 128][..], &[16, 16][..], &[16][..], &[17][..]),
            (&[16, 128][..], &[16, 16][..], &[16][..], &[15][..]),
        ] {
            assert!(W4A8Int8::parse(good.as_bytes(), w, s, ch, cb).is_err());
        }
        assert!(W4A8Int8::parse(
            &vec![b' '; MAX_DESCRIPTOR_BYTES + 1],
            &[1, 128],
            &[1, 16],
            &[1],
            &[16]
        )
        .is_err());
    }

    #[test]
    fn scale_validators_admit_zero_group_scale_but_refuse_nan_and_negative() {
        let d = W4A8Int8::parse(
            br#"{"convrot_groupsize":256,"format":"asym_w4a8_int8","group_size":16}"#,
            &[2, 128],
            &[2, 16],
            &[2],
            &[16],
        )
        .unwrap();
        // E4M3 zero (0x00) and +1.0 (0x38) are valid; NaN (0x7F/0xFF) and a negative scale
        // (0xA8 = -0.25) refuse. 32 group scales for out=2, in=256.
        let mut ok = vec![0x38u8; 32];
        ok[0] = 0x00;
        assert!(d.validate_s_rel(&ok).is_ok());
        let mut nan = vec![0x38u8; 32];
        nan[0] = 0x7F;
        assert!(d.validate_s_rel(&nan).is_err());
        let mut neg = vec![0x38u8; 32];
        neg[0] = 0xA8;
        assert!(d.validate_s_rel(&neg).is_err());
        assert!(d.validate_s_channel(&[1.0, 1.0]).is_ok());
        assert!(d.validate_s_channel(&[0.0, 1.0]).is_err());
        assert!(d.validate_s_channel(&[f32::NAN, 1.0]).is_err());
        assert!(d.validate_codebook(&[0.0; 16]).is_ok());
        assert!(d.validate_codebook(&[f32::NAN; 16]).is_err());
    }

    #[test]
    fn bf16_rounding_is_ties_to_even() {
        assert_eq!(bf16_rn(1.0), 1.0);
        // The halfway between two adjacent bf16 values rounds to the even mantissa (0.5).
        let lo = 0.5f32;
        let hi = f32::from_bits(0x3F01_0000); // 0.50390625
        let mid = (lo + hi) / 2.0; // 0.501953125
        assert_eq!(bf16_rn(mid), 0.5);
    }

    #[test]
    fn regular_hadamard_order_4_is_regular_not_sylvester() {
        let r = regular_hadamard(4);
        let want: Vec<f32> = H4.iter().flatten().map(|&v| 0.5 * v).collect();
        assert_eq!(r, want);
    }

    #[test]
    fn packed_dequant_honors_nibble_codebook_fp8_ties_and_saturation() {
        // --- ties to even ---
        // cb[0] = 0.5, cb[1] = 1.5. packed[0] = 0x10 -> col0 nibble 0, col1 nibble 1.
        let mut cb = vec![0.0f32; 16];
        cb[0] = 0.5;
        cb[1] = 1.5;
        let mut packed = vec![0i8; 16];
        packed[0] = 0x10;
        let s_rel = vec![0x38u8; 2]; // 1.0 per group
        let mut out = Vec::new();
        dequant_packed_to_int8(&packed, &s_rel, &cb, 1, 32, &mut out).unwrap();
        assert_eq!(out[0], 0, "0.5 rounds ties-to-even to 0");
        assert_eq!(out[1], 2, "1.5 rounds ties-to-even to 2");

        // --- saturation and fp8 decode ---
        // cb all 1.0; group0 scale 128 (0x70), so 1.0*128 saturates at 127.
        let cb_ones = vec![1.0f32; 16];
        let mut packed_sat = vec![0i8; 16];
        packed_sat[0] = 0x00;
        let s_rel_sat = vec![0x70u8, 0x38]; // 128, 1.0
        let mut out_sat = Vec::new();
        dequant_packed_to_int8(&packed_sat, &s_rel_sat, &cb_ones, 1, 32, &mut out_sat).unwrap();
        assert_eq!(out_sat[0], 127); // saturated high
        assert_eq!(out_sat[16], 1); // group1 scale 1.0 -> 1
    }

    #[test]
    fn reference_forward_is_pinned_elementwise_arithmetic() {
        // in=256 (one rotation group), out=2, m=1, no bias. A codebook with cb[8] = 0 makes every
        // packed nibble decode to int8 weight 0, so the int32 accumulation is exactly 0 and the
        // epilogue yields exactly 0 — a bit-exact, non-fudged oracle check.
        let m = 1usize;
        let out = 2usize;
        let k = 256usize;
        let x: Vec<f32> = vec![1.0; m * k];
        let cb: Vec<f32> = (0..16).map(|j| j as f32 / 8.0 - 1.0).collect();
        let mut packed = vec![0i8; out * (k / 2)];
        packed.fill(0x88u8 as i8); // both nibbles code 8 -> cb[8] = 0.0
        let s_rel = vec![0x38u8; out * (k / 16)]; // 1.0
        let s_channel = vec![1.0f32; out];
        let r = reference_forward(
            &x, &packed, &s_rel, &s_channel, &cb, None, m, out, k, BF16_MAX,
        )
        .unwrap();
        assert_eq!(r.len(), m * out);
        assert_eq!(r, vec![0.0f32; out]);
    }
}
