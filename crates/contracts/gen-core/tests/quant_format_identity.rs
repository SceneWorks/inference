//! Distinct packed-float formats must never masquerade as affine integer tiers.
use gen_core::{ComponentPrecisionFloor, PrecisionFloorComponent, Quant};

#[test]
fn fp8r_round_trips_without_aliasing_q8_or_dense() {
    for (quant, key, bits) in [
        (Quant::Q4, "q4", 4),
        (Quant::Q8, "q8", 8),
        (Quant::Nvfp4, "nvfp4", 4),
        (Quant::Fp8Rowwise, "fp8r", 8),
    ] {
        assert_eq!(quant.key(), key);
        assert_eq!(Quant::from_key(key).unwrap(), quant);
        assert_eq!(quant.bits(), bits);
    }
    assert_ne!(Quant::Fp8Rowwise, Quant::Q8);
    assert_ne!(Quant::Fp8Rowwise.key(), Quant::Q8.key());
    assert!(Quant::from_key("fp8").is_err());
    assert!(Quant::from_key("bf16").is_err());
    assert!(Quant::Fp8Rowwise.affine_bits().is_err());
    assert!(Quant::Nvfp4.affine_bits().is_err());
    assert_eq!(Quant::Q8.affine_bits().unwrap(), 8);
}

#[test]
fn precision_floor_matches_exact_format_not_element_width() {
    let floor = ComponentPrecisionFloor {
        component: PrecisionFloorComponent::TextEncoder,
        selected_tier: Quant::Fp8Rowwise,
        resident_tier: Quant::Q8,
    };
    assert!(floor.applies_to(Quant::Fp8Rowwise));
    assert!(!floor.applies_to(Quant::Q8));
    assert!(!floor.applies_to(Quant::Nvfp4));
}
