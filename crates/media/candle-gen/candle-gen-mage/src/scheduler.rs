//! Mage's static-shift FlowMatch Euler schedule.

use candle_core::{DType, Error, Result};

use crate::config::STATIC_SHIFT;

pub fn sigmas(steps: usize) -> Result<Vec<f32>> {
    if steps == 0 {
        return Err(Error::Msg("mage: steps must be >= 1".into()));
    }
    // Match ATen's endpoint-aware f32 linspace fill. A one-direction fill differs by ULPs late
    // in the production ladders and those errors feed back through every subsequent DiT call.
    let n = steps as f32;
    let start = 1.0f32;
    let end = 1.0 / n;
    let step = if steps > 1 {
        (end - start) / (steps - 1) as f32
    } else {
        0.0
    };
    let shift = STATIC_SHIFT as f32;
    let mut out = (0..steps)
        .map(|i| {
            let raw = if i < steps / 2 {
                step.mul_add(i as f32, start)
            } else {
                (-step).mul_add((steps - i - 1) as f32, end)
            };
            shift * raw / (1.0 + (shift - 1.0) * raw)
        })
        .collect::<Vec<_>>();
    out.push(0.0);
    Ok(out)
}

pub fn euler_step(
    x: &candle_core::Tensor,
    velocity: &candle_core::Tensor,
    sigma: f32,
    next: f32,
) -> Result<candle_core::Tensor> {
    let output_dtype = velocity.dtype();
    (x.to_dtype(DType::F32)? + (velocity.to_dtype(DType::F32)? * (next - sigma) as f64)?)?
        .to_dtype(output_dtype)
}

/// Flow matching regresses velocity `z - epsilon`, not epsilon/noise prediction.
pub fn velocity_target(
    z: &candle_core::Tensor,
    epsilon: &candle_core::Tensor,
) -> Result<candle_core::Tensor> {
    z - epsilon
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn turbo_uses_same_shift_six_ladder() {
        let got = sigmas(4).unwrap();
        let want = [1.0, 0.947_368_44, 0.857_142_87, 0.666_666_7, 0.0];
        for (a, b) in got.iter().zip(want) {
            assert!((a - b).abs() < 2e-7, "{a} != {b}");
        }
    }

    #[test]
    fn production_ladders_match_torch_bit_for_bit() {
        assert_eq!(
            sigmas(20).unwrap(),
            vec![
                1.0, 0.99130434, 0.98181814, 0.97142863, 0.96000004, 0.94736844, 0.9333333,
                0.917647, 0.90000004, 0.88000005, 0.85714287, 0.83076924, 0.8, 0.76363635, 0.72,
                0.6666667, 0.6, 0.51428574, 0.4, 0.24000001, 0.0,
            ]
        );
        assert_eq!(
            sigmas(30).unwrap(),
            vec![
                1.0, 0.9942857, 0.9882353, 0.98181814, 0.97499996, 0.96774185, 0.96000004,
                0.9517242, 0.9428571, 0.9333334, 0.92307687, 0.912, 0.90000004, 0.8869566,
                0.87272733, 0.85714287, 0.8399999, 0.8210527, 0.79999995, 0.77647054, 0.75,
                0.71999997, 0.68571424, 0.6461538, 0.59999996, 0.54545456, 0.48, 0.39999998, 0.3,
                0.17142859, 0.0,
            ]
        );
    }

    #[test]
    fn euler_upcasts_bf16_math_before_rounding_once() {
        let d = candle_core::Device::Cpu;
        let x = candle_core::Tensor::new(&[1.0f32], &d)
            .unwrap()
            .to_dtype(DType::BF16)
            .unwrap();
        let velocity = candle_core::Tensor::new(&[0.003_906_25f32], &d)
            .unwrap()
            .to_dtype(DType::BF16)
            .unwrap();
        let got = euler_step(&x, &velocity, 1.0, 0.0)
            .unwrap()
            .to_dtype(DType::F32)
            .unwrap()
            .to_vec1::<f32>()
            .unwrap();
        assert_eq!(got, [0.996_093_75]);
    }

    #[test]
    fn velocity_is_z_minus_epsilon_mutation_guard() {
        let d = candle_core::Device::Cpu;
        let z = candle_core::Tensor::new(&[3f32, -1.], &d).unwrap();
        let e = candle_core::Tensor::new(&[1f32, 2.], &d).unwrap();
        assert_eq!(
            velocity_target(&z, &e).unwrap().to_vec1::<f32>().unwrap(),
            [2., -3.]
        );
    }
}
