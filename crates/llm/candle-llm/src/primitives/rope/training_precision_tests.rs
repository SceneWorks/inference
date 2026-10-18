//! Frozen HF5.20 explicit Module.to(dtype) buffer semantics; CPU-only constants.
use super::*;
#[test]
fn selected_storage_rounds_frequency_values_then_keeps_f32_phase_math() {
    let original = Rope::standard(16, 5_000_000.0);
    let original_values = original.inv_freq.clone();
    for (dtype, expected) in [
        (DType::BF16, 149.0f32 / 1024.0),
        (DType::F16, 1191.0 / 8192.0),
    ] {
        let rounded = original.with_training_frequency_storage(dtype).unwrap();
        assert_eq!(
            rounded.inv_freq[1], expected,
            "actual tinyHFModule.to observation"
        );
        assert_eq!(
            original.inv_freq, original_values,
            "inference schedule must not mutate"
        );
        let positions = [0, 1, 127, 511];
        let (cos, sin) = rounded
            .cos_sin_at(&positions, DType::F32, &Device::Cpu)
            .unwrap();
        let actual_cos = cos.flatten_all().unwrap().to_vec1::<f32>().unwrap();
        let actual_sin = sin.flatten_all().unwrap().to_vec1::<f32>().unwrap();
        let phases = Tensor::new(&positions.map(|p| p as f32 * expected), &Device::Cpu).unwrap();
        let expected_cos = phases.cos().unwrap().to_vec1::<f32>().unwrap();
        let expected_sin = phases.sin().unwrap().to_vec1::<f32>().unwrap();
        for row in 0..positions.len() {
            assert_eq!(actual_cos[row * 16 + 1], expected_cos[row]);
            assert_eq!(actual_sin[row * 16 + 1], expected_sin[row]);
        }
    }
    assert_eq!(
        original
            .with_training_frequency_storage(DType::F32)
            .unwrap()
            .inv_freq,
        original_values
    );
    assert!(original
        .with_training_frequency_storage(DType::U32)
        .is_err());
}
