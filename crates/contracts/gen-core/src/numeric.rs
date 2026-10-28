//! Shared scalar numeric helpers (sc-27419): one definition of the f32 ↔ bfloat16 conversions the
//! contract modules (FLUX3 affine packing, Veda key/query staging, distributed reduce precision)
//! previously each hand-rolled, so they share one NaN policy and one rounding rule.

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
