//! Native packed **W4A8 (asym_w4a8_int8) projection contract and reference packed decode**
//! (sc-26685).
//!
//! This is deliberately separate from both the plain `int8-per-row-v1` dense decoder and the
//! `convrot-int8-native` codec (`crate::convrot_int8`): the stored weight is **two int4 codes per
//! I8 byte**, indexed through a 16-level Lloyd-Max codebook, a per-group fp8 (E4M3FN) scale, and an
//! F32 per-output-channel scale left to the int8 GEMM epilogue. The activation side is the same
//! fused ConvRot (regular Hadamard, block 256) + row-wise int8 quantization as the ConvRot INT8
//! codec; only the weight half — packed codes, codebook, `s_rel`/`s_channel` — is new.
//!
//! This module owns the **contract** ([`W4A8Int8`] descriptor/geometry/value validation and
//! residency) and the **host packed→int8 decode** ([`dequant_packed_to_int8`]) that the Candle
//! CPU/Metal dequant-dense fallback consumes — the same production-decoder role
//! `decode_int8_per_row` / `decode_nvfp4` / `decode_mxfp8` play in [`crate::comfy_quant`]. The
//! bit-accurate *forward* reference (activation FHT + quantizer + int32 GEMM + epilogue) is
//! **deliberately not here**: the fixture oracle is the pinned external executable below, not a
//! host re-derivation of it. The ConvRot activation rotation itself is S4's proven
//! `candle_gen::quant::regular_hadamard` / the MLX `convrot_rotation` builder — never re-derived.
//!
//! # Authority (exact pinned blob SHAs at comfy-kitchen PR#90, tree `06a261f8`, head `d908d589`)
//!
//! * `comfy_kitchen/backends/cuda/ops/w4a8_gemm.cu` — blob `21002ea10bf0d05d945083c61f79098324b76bb1`.
//!   `dequant_int4_grouped_to_int8_kernel`: nibble unpack `c0 = byte & 0xF, c1 = (byte >> 4) & 0xF`
//!   (even column low nibble, odd high); scale decode `__half2float(__nv_cvt_fp8_to_halfraw(v,
//!   __NV_E4M3))`; level `v = codebook ? cb[c] : (float(c) - 8.0f)`; output
//!   `static_cast<int8_t>(max(-127, min(127, __float2int_rn(v * s))))` — **`__float2int_rn` is
//!   round-half-to-even, and the weight clamp is `[-127, 127]`**.
//! * `comfy_kitchen/backends/cuda/ops/cutlass_gemm_int8.cu` — blob `f7142fa41973a808cd6a5fe4b0b1cd063a7b2a3b`.
//!   Epilogue `D = (sum_k A·B) * x_scale[m] * w_scale[n] + bias[n]`: `ElementAcc = int32_t`,
//!   `ElementCompute = float`, `XScale`/`WScale`/`Mul0`/`Mul1`/`Add2` all `round_to_nearest`
//!   (`FloatRoundStyle::round_to_nearest`), store `round_to_nearest`. Fold order is
//!   `((acc · xs) · ws) + bias`, each float op round-to-nearest, store round-to-nearest.
//! * `comfy_kitchen/backends/cuda/ops/int8_linear.cu` — blob `4c7d49e72425aa868d388a5b73b98c585e1fbded`.
//!   `quantize_int8_rowwise_convrot_kernel` (activation): f32 Fast-Hadamard butterfly of the
//!   *regular* (non-Walsh) Hadamard block `H4 = [[1,1,1,-1],[1,1,-1,1],[1,-1,1,1],[-1,1,1,1]]`
//!   (`h4_row_dot`), four stages stride `1,4,16,64` each `* 0.5`; row scale
//!   `max(min(absmax, 3.38953139e38f) / 127, 1e-30)`; per-element `scaled = f32(bf16(f32(bf16(rot))
//!   / f32(bf16(scale))))` (`quant_div_float_to_float`), then `nearbyintf(scaled)` (round-half-even)
//!   clamp **`[-128, 127]`**. `dequantize_int8_linear_kernel` is the 2-pass twin of the epilogue
//!   above (`value = float(acc) * x_scales[row] * weight_scale; value += bias; from_float(value)`).
//!
//! Note the two clamps differ by design: the **weight** dequant clamps `[-127, 127]`, the
//! **activation** quantizer clamps `[-128, 127]` — matching the two kernels verbatim.

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
/// (non-Walsh) Hadamard block of order 256, normalized by `1/16`.
pub const CONVROT_GROUP: usize = 256;

/// The regular (non-Walsh) `H₄` Hadamard block. **Sign order is load-bearing** — this is the PR#90
/// "regular" order (rows as `h4_row_dot` in `int8_linear.cu`), not the Sylvester/Walsh order, and it
/// is the exact basis S4's `candle_gen::quant::regular_hadamard` already proves
/// (`order4_is_regular_not_sylvester`). The MLX native primitive builds its rotation from
/// [`regular_hadamard`]; the Candle native primitive reuses S4's builder — the two must never drift.
pub const H4: [[f32; 4]; 4] = [
    [1.0, 1.0, 1.0, -1.0],
    [1.0, 1.0, -1.0, 1.0],
    [1.0, -1.0, 1.0, 1.0],
    [-1.0, 1.0, 1.0, 1.0],
];

/// The dense regular-Hadamard rotation of `order` (a power of four), normalized by `1/√order` —
/// the Kronecker power of [`H4`] with per-element sign `∏ H4[i_d][j_d]`. This is the **production**
/// rotation basis: the MLX `W4A8Linear` builds its resident rotation from it (cached through
/// `Weights::convrot_rotation`), so the packed W4A8 weight and the ConvRot activation leg share one
/// authority. The activation-side fused CUDA path factors the same product as a radix-4 FHT.
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
        for &s in scales {
            if !s.is_finite() || s <= 0.0 {
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
            + 16 * 4                                                           // codebook F32
    }
}

/// Decode the packed int4 weight into the grouped int8 the GEMM actually multiplies, replicating
/// `dequant_int4_grouped_to_int8_kernel` (`w4a8_gemm.cu`, blob `21002ea10`, above):
///
/// ```text
/// byte  = packed[n, c/2]            (even col -> low nibble, odd -> high nibble)
/// lvl   = codebook[nibble]          (the frozen codebook dialect)
/// s     = e4m3_to_f32(s_rel[n, c/16])
/// int8  = round_ties_even(lvl · s)  clamped to [-127, 127]
/// ```
///
/// This is the production host decoder for the Candle CPU/Metal dequant-dense fallback; the
/// rotation is **not** applied here (the stored weight is already in the rotated basis, and the
/// activation rotation is S4's existing path). `packed` is `[out, in/2]`, `s_rel` `[out, in/16]`
/// raw E4M3 bytes, `codebook` the 16 F32 levels.
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
}