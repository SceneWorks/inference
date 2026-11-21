//! Native packed W4A8 projections, re-exported from [`candle_layers::w4a8_int8`] (`SW-6955`).

pub use candle_layers::w4a8_int8::*;

#[cfg(test)]
mod tests {
    use super::*;
    use crate::Weights;
    use candle_core::{DType, Device, Tensor};
    use std::collections::HashMap;

    fn operands() -> HashMap<String, Tensor> {
        let device = &Device::Cpu;
        let mut map = HashMap::new();
        let payload = br#"{"convrot_groupsize":256,"format":"asym_w4a8_int8","group_size":16}"#;
        map.insert(
            "p.comfy_quant".into(),
            Tensor::from_slice(payload, payload.len(), device).unwrap(),
        );
        map.insert(
            "p.weight".into(),
            Tensor::zeros((16, 128), DType::U8, device).unwrap(),
        );
        map.insert(
            "p.weight_s_rel".into(),
            Tensor::from_raw_buffer(&[0x38; 256], DType::F8E4M3, &[16, 16], device).unwrap(),
        );
        map.insert(
            "p.weight_s_channel".into(),
            Tensor::ones(16, DType::F32, device).unwrap(),
        );
        map.insert(
            "p.weight_codebook".into(),
            Tensor::from_vec((0..16).map(|v| v as f32 - 8.).collect(), 16, device).unwrap(),
        );
        map
    }

    #[test]
    fn refuses_orphaned_and_contradictory_operands() {
        let mut map = operands();
        map.remove("p.comfy_quant");
        let err = W4A8Linear::from_weights(&Weights::from_map(map), "p", false).unwrap_err();
        assert!(err.to_string().contains("authoritative"));
        for suffix in ["weight_scale", "scales", "biases", "weight_correction"] {
            let mut map = operands();
            map.insert(
                format!("p.{suffix}"),
                Tensor::ones(16, DType::F32, &Device::Cpu).unwrap(),
            );
            let err = W4A8Linear::from_weights(&Weights::from_map(map), "p", false).unwrap_err();
            assert!(err.to_string().contains("forbids"));
        }
    }

    #[test]
    fn refuses_noncanonical_descriptor_and_invalid_values_before_cuda() {
        let mut map = operands();
        let payload = br#"{"convrot_groupsize":256,"format":"asym_w4a8_int8","group_size":16,"convrot":true}"#;
        map.insert(
            "p.comfy_quant".into(),
            Tensor::from_slice(payload, payload.len(), &Device::Cpu).unwrap(),
        );
        let err = W4A8Linear::from_weights(&Weights::from_map(map), "p", false).unwrap_err();
        assert!(err.to_string().contains("descriptor"));
        let mut map = operands();
        map.insert(
            "p.weight_s_channel".into(),
            Tensor::zeros(16, DType::F32, &Device::Cpu).unwrap(),
        );
        let err = W4A8Linear::from_weights(&Weights::from_map(map), "p", false).unwrap_err();
        assert!(err.to_string().contains("finite positive F32"));
        let mut map = operands();
        map.insert(
            "p.weight_s_rel".into(),
            Tensor::from_raw_buffer(&[0x7f; 256], DType::F8E4M3, &[16, 16], &Device::Cpu).unwrap(),
        );
        let err = W4A8Linear::from_weights(&Weights::from_map(map), "p", false).unwrap_err();
        assert!(err.to_string().contains("non-negative finite E4M3FN"));
    }

    #[test]
    fn dense_projection_is_not_claimed_as_w4a8() {
        let mut map = HashMap::new();
        map.insert(
            "p.weight".into(),
            Tensor::zeros((16, 256), DType::F32, &Device::Cpu).unwrap(),
        );
        assert!(
            W4A8Linear::from_weights(&Weights::from_map(map), "p", false)
                .unwrap()
                .is_none()
        );
    }

    #[cfg(not(feature = "cuda"))]
    #[test]
    fn validated_w4a8_refuses_a_floating_fallback() {
        let err = W4A8Linear::from_weights(&Weights::from_map(operands()), "p", false).unwrap_err();
        assert!(err
            .to_string()
            .contains("requires native CUDA; no floating fallback"));
    }
}
