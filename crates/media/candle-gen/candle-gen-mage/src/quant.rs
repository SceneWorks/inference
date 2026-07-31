//! Mage load-time Q4/Q8 seam.
//!
//! The published Mage snapshots are dense BF16. For a quantized load the DiT is built on CPU and
//! every one of its 174 live `Linear` projections is folded directly onto the target device as a
//! GGUF Q4_0/Q8_0 tensor. This avoids ever materializing the dense DiT on CUDA. BF16 loads retain the
//! original direct-to-device path unchanged.

use candle_core::{Device, Result, Tensor};
use candle_gen::gen_core::{
    effective_component_quant, ComponentPrecisionFloor, PrecisionFloorComponent, Quant,
};
use candle_gen::quant::{DenseLinear, QLinear};
use candle_gen_boogu::loader::Weights;
use candle_nn::Linear;

/// Candle's binding declaration of the same two Mage q4 floors used by the MLX provider.
pub const COMPONENT_PRECISION_FLOORS: &[ComponentPrecisionFloor] = &[
    ComponentPrecisionFloor {
        component: PrecisionFloorComponent::TextEncoder,
        selected_tier: Quant::Q4,
        resident_tier: Quant::Q8,
    },
    ComponentPrecisionFloor {
        component: PrecisionFloorComponent::TransformerHead,
        selected_tier: Quant::Q4,
        resident_tier: Quant::Q8,
    },
];

pub(crate) fn component_quant(component: PrecisionFloorComponent, selected: Quant) -> Quant {
    effective_component_quant(COMPONENT_PRECISION_FLOORS, component, selected)
}

/// Shape-inferred dense loader used by the Mage transformer before its optional fold.
pub(crate) fn linear(weights: &Weights, base: &str, bias: bool) -> Result<QLinear> {
    let weight = weights.get(&format!("{base}.weight"))?;
    let bias = if bias {
        Some(weights.get(&format!("{base}.bias"))?)
    } else {
        None
    };
    Ok(QLinear::from_dense(DenseLinear::Linear(Linear::new(
        weight, bias,
    ))))
}

/// Fold one projection to the requested tier on `device`.
pub(crate) fn quantize_onto(linear: &mut QLinear, quant: Quant, device: &Device) -> Result<()> {
    linear.quantize_dequant_onto(quant, device)
}

/// Move a still-dense projection onto the target device.
pub(crate) fn move_onto(linear: &mut QLinear, device: &Device) -> Result<()> {
    linear.to_device(device)
}

/// Move a non-linear leaf tensor alongside the folded projections.
pub(crate) fn tensor_onto(tensor: &mut Tensor, device: &Device) -> Result<()> {
    *tensor = tensor.to_device(device)?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use candle_core::{DType, Tensor};
    use candle_gen::quant::MatmulStrategy;

    fn fixture() -> QLinear {
        let weight = Tensor::from_vec(
            (0..64 * 32)
                .map(|index| ((index as f32) * 0.017).sin())
                .collect::<Vec<_>>(),
            (64, 32),
            &Device::Cpu,
        )
        .unwrap();
        QLinear::from_dense(DenseLinear::Linear(Linear::new(weight, None)))
    }

    #[test]
    fn q4_and_q8_take_distinct_production_fold_path() {
        for quant in [Quant::Q4, Quant::Q8] {
            let mut linear = fixture();
            quantize_onto(&mut linear, quant, &Device::Cpu).unwrap();
            assert!(linear.is_quantized(), "{quant:?} stayed dense");
            assert_eq!(linear.matmul_strategy(), Some(MatmulStrategy::DequantDense));
        }
    }

    #[test]
    fn q4_component_floors_are_descriptor_visible_and_shared_by_both_load_seams() {
        let caps = crate::descriptor().capabilities;
        assert_eq!(caps.component_precision_floors, COMPONENT_PRECISION_FLOORS);
        assert_eq!(
            component_quant(PrecisionFloorComponent::TextEncoder, Quant::Q4),
            Quant::Q8
        );
        assert_eq!(
            component_quant(PrecisionFloorComponent::TransformerHead, Quant::Q4),
            Quant::Q8
        );
        for component in [
            PrecisionFloorComponent::TextEncoder,
            PrecisionFloorComponent::TransformerHead,
        ] {
            assert_eq!(component_quant(component, Quant::Q8), Quant::Q8);
        }
    }

    #[test]
    fn q8_is_closer_than_q4_and_both_are_nonconstant() {
        let input = Tensor::from_vec(
            (0..3 * 32)
                .map(|index| ((index as f32) * 0.031).cos())
                .collect::<Vec<_>>(),
            (3, 32),
            &Device::Cpu,
        )
        .unwrap();
        let dense = fixture().forward(&input).unwrap();
        let error = |quant| {
            let mut linear = fixture();
            quantize_onto(&mut linear, quant, &Device::Cpu).unwrap();
            let output = linear.forward(&input).unwrap();
            let std = output
                .to_dtype(DType::F32)
                .unwrap()
                .flatten_all()
                .unwrap()
                .to_vec1::<f32>()
                .unwrap();
            assert!(std.windows(2).any(|pair| pair[0] != pair[1]));
            (output - &dense)
                .unwrap()
                .abs()
                .unwrap()
                .mean_all()
                .unwrap()
                .to_scalar::<f32>()
                .unwrap()
        };
        let q8 = error(Quant::Q8);
        let q4 = error(Quant::Q4);
        assert!(q8 < q4, "q8 error {q8} must be below q4 error {q4}");
    }
}
