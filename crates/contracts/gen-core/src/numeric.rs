//! Shared scalar numeric helpers (sc-27419): one definition each of the f32 ↔ bfloat16 conversions
//! and the IEEE half decode that contract modules (FLUX3 affine packing, Veda key/query staging,
//! distributed reduce precision, pickle and depth-map f16 decode) previously each hand-rolled, so
//! they share one NaN policy and one rounding rule.

/// f32 → bf16 bits, round-to-nearest-even; a NaN stays a (quiet) NaN. Without the NaN guard the
/// rounding add carries a low-payload NaN (e.g. `0x7f80_0001`) into +inf and `0x7fff_ffff` into -0.
pub fn bf16_bits(x: f32) -> u16 {
    let bits = x.to_bits();
    if x.is_nan() {
        return ((bits >> 16) | 0x0040) as u16;
    }
    let round = 0x7fff + ((bits >> 16) & 1);
    (bits.wrapping_add(round) >> 16) as u16
}

/// bf16 bits → the exact f32 value.
pub fn bf16_value(bits: u16) -> f32 {
    f32::from_bits((bits as u32) << 16)
}

/// Round an f32 to the nearest bfloat16 (ties to even), returned as f32. A NaN is returned as is.
pub fn round_bf16(x: f32) -> f32 {
    if x.is_nan() {
        return x;
    }
    bf16_value(bf16_bits(x))
}

/// IEEE-754 half bits → the exact f32 value (inf/NaN payload kept; no `half` crate dependency).
pub fn f16_bits_to_f32(h: u16) -> f32 {
    let sign = (h & 0x8000) as u32;
    let exp = (h >> 10) & 0x1f;
    let mant = (h & 0x3ff) as u32;
    let bits = if exp == 0 {
        // Zero or subnormal: exactly `mant × 2^-24` (sc-27419 — the old pickle decoder normalised
        // with an off-by-one exponent and returned half of every subnormal).
        (sign << 16) | (mant as f32 * f32::from_bits(0x3380_0000)).to_bits()
    } else if exp == 0x1f {
        (sign << 16) | 0x7f80_0000 | (mant << 13)
    } else {
        (sign << 16) | (((exp as i32 - 15 + 127) as u32) << 23) | (mant << 13)
    };
    f32::from_bits(bits)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn bf16_rounds_to_nearest_even_and_keeps_nan_a_nan() {
        assert_eq!(bf16_bits(1.0), 0x3f80);
        // 1 + 2^-8 is a tie between 1.0 (even) and 1 + 2^-7: rounds to even.
        assert_eq!(bf16_bits(1.0 + 2f32.powi(-8)), 0x3f80);
        assert_eq!(bf16_bits(1.0 + 3.0 * 2f32.powi(-8)), 0x3f82);
        assert_eq!(bf16_bits(f32::INFINITY), 0x7f80);
        assert_eq!(bf16_bits(f32::NEG_INFINITY), 0xff80);
        // Low-payload NaNs stay NaN instead of carrying into +inf / -0.
        for nan in [0x7f80_0001u32, 0x7fff_ffff, 0xffc0_0000, 0x7fc0_0000] {
            assert!(
                bf16_value(bf16_bits(f32::from_bits(nan))).is_nan(),
                "{nan:#x}"
            );
        }
        assert_eq!(round_bf16(1.2), 1.203125);
        assert!(round_bf16(f32::NAN).is_nan());
        assert_eq!(round_bf16(f32::NEG_INFINITY), f32::NEG_INFINITY);
    }

    /// Every one of the 65,536 half bit patterns decodes to the value of the IEEE formula (the
    /// arithmetic form `iris::train::depth` used to inline), NaNs to NaN.
    #[test]
    fn f16_decode_is_exact_for_every_bit_pattern() {
        for h in 0..=u16::MAX {
            let sign = if h & 0x8000 != 0 { -1.0 } else { 1.0 };
            let exp = ((h >> 10) & 0x1f) as i32;
            let frac = (h & 0x3ff) as f64;
            let want = match exp {
                0 => sign * frac * 2f64.powi(-24),
                31 if frac == 0.0 => sign * f64::INFINITY,
                31 => f64::NAN,
                e => sign * (1.0 + frac / 1024.0) * 2f64.powi(e - 15),
            };
            let got = f64::from(f16_bits_to_f32(h));
            if want.is_nan() {
                assert!(got.is_nan(), "{h:#06x}");
            } else {
                assert_eq!(got.to_bits(), want.to_bits(), "{h:#06x}");
            }
        }
    }

    /// The shared `round_bf16` is bit-identical to the masked-add form `dist.rs` used to inline.
    #[test]
    fn round_bf16_matches_the_masked_add_form() {
        let legacy = |x: f32| {
            let b = x.to_bits();
            f32::from_bits(b.wrapping_add(0x7FFF + ((b >> 16) & 1)) & 0xFFFF_0000)
        };
        let mut s = 0x1234_5678u32;
        for _ in 0..100_000 {
            s ^= s << 13;
            s ^= s >> 17;
            s ^= s << 5;
            let x = f32::from_bits(s);
            if !x.is_nan() {
                assert_eq!(round_bf16(x).to_bits(), legacy(x).to_bits(), "{s:#x}");
            }
        }
    }
}
