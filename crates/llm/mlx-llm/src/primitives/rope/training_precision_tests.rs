//! Supported-host tests for source training buffer rounding. Not run on Windows.
use super::*;
#[test]
fn selected_storage_rebuilds_both_host_schedule_and_device_frequency_row() {
    let original = Rope::standard(16, 5_000_000.0);
    let original_values = original.inv_freq.clone();
    for (dtype, expected) in [
        (Dtype::Bfloat16, 149.0f32 / 1024.0),
        (Dtype::Float16, 1191.0 / 8192.0),
    ] {
        let rounded = original.with_training_frequency_storage(dtype).unwrap();
        assert_eq!(
            rounded.inv_freq[1], expected,
            "actual tinyHFModule.to observation"
        );
        assert_eq!(original.inv_freq, original_values);
        mlx_rs::transforms::eval([&rounded.freq_row]).unwrap();
        assert_eq!(
            rounded.freq_row.dtype(),
            Dtype::Float32,
            "rounded values are widened for source F32 phase math"
        );
        let row = rounded.freq_row.as_slice::<f32>();
        assert_eq!(row[1], expected);
        assert_eq!(row[9], expected);
        let positions = [0, 1, 127, 511];
        let (cos, sin) = rounded.cos_sin_at(&positions, Dtype::Float32).unwrap();
        mlx_rs::transforms::eval([&cos, &sin]).unwrap();
        let phases = Array::from_slice(&positions.map(|p| p as f32 * expected), &[4]);
        let expected_cos = phases.cos().unwrap();
        let expected_sin = phases.sin().unwrap();
        mlx_rs::transforms::eval([&expected_cos, &expected_sin]).unwrap();
        let cos = cos.reshape(&[-1]).unwrap();
        let sin = sin.reshape(&[-1]).unwrap();
        mlx_rs::transforms::eval([&cos, &sin]).unwrap();
        for i in 0..4 {
            assert!(
                (cos.as_slice::<f32>()[i * 16 + 1] - expected_cos.as_slice::<f32>()[i]).abs()
                    < 1e-6
            );
            assert!(
                (sin.as_slice::<f32>()[i * 16 + 1] - expected_sin.as_slice::<f32>()[i]).abs()
                    < 1e-6
            );
        }
    }
    assert_eq!(
        original
            .with_training_frequency_storage(Dtype::Float32)
            .unwrap()
            .inv_freq,
        original_values
    );
    assert!(original
        .with_training_frequency_storage(Dtype::Uint32)
        .is_err());
}
