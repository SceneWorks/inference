//! FP32 host noise and source-faithful ordinary/DMD sampling over the native graph.
use crate::precision::linear_fp32;
use crate::transformer::Transformer;
use mlx_gen::array::scalar;
use mlx_gen::gen_core::moss_image::{
    moss_image_schedule, MossImageAcceleration, MossImageResolved,
};
use mlx_gen::{CancelFlag, Error, Progress, Result};
use mlx_rs::ops::{add, multiply, subtract};
use mlx_rs::{Array, Dtype};

/// Read logical C-order values, rather than the physical storage of an MLX strided view.
pub(crate) fn logical_f32_readback(array: &Array) -> Result<Vec<f32>> {
    let contiguous = mlx_gen::array::contiguous(&array.as_dtype(Dtype::Float32)?)?;
    mlx_rs::transforms::eval([&contiguous])?;
    Ok(contiguous
        .try_as_slice::<f32>()
        .map_err(|error| Error::Msg(error.to_string()))?
        .to_vec())
}

pub(crate) fn check(cancel: &CancelFlag) -> Result<()> {
    if cancel.is_cancelled() {
        Err(Error::Canceled)
    } else {
        Ok(())
    }
}
pub(crate) fn digest_f32(values: &[f32]) -> String {
    use sha2::{Digest, Sha256};
    let mut h = Sha256::new();
    for v in values {
        h.update(v.to_le_bytes());
    }
    format!("{:x}", h.finalize())
}
pub(crate) fn normal(seed: u64, n: usize) -> Vec<f32> {
    let mut state = seed;
    let mut uniform = || {
        state = state.wrapping_add(0x9e3779b97f4a7c15);
        let mut z = state;
        z = (z ^ (z >> 30)).wrapping_mul(0xbf58476d1ce4e5b9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94d049bb133111eb);
        ((z ^ (z >> 31)) >> 11) as f64 / (1u64 << 53) as f64
    };
    let mut out = Vec::with_capacity(n);
    while out.len() < n {
        let r = (-2.0 * uniform().max(1e-12).ln()).sqrt();
        let t = std::f64::consts::TAU * uniform();
        out.push((r * t.cos()) as f32);
        if out.len() < n {
            out.push((r * t.sin()) as f32)
        }
    }
    out
}
pub(crate) struct Sampling<'a> {
    pub transformer: &'a Transformer,
    pub resolved: &'a MossImageResolved,
    pub positive: &'a [Array],
    pub negative: &'a [Array],
    pub dataspace: Option<(&'a Array, &'a [f32])>,
}
impl Sampling<'_> {
    pub(crate) fn run(
        &self,
        noise: &Array,
        cancel: &CancelFlag,
        progress: &mut dyn FnMut(Progress),
    ) -> Result<Array> {
        let r = self.resolved;
        check(cancel)?;
        if !matches!(r.acceleration, MossImageAcceleration::Off) {
            let (q, codes) = self
                .dataspace
                .ok_or_else(|| Error::Unsupported("MOSS DMD dataspace missing".into()))?;
            let mut clean = multiply(noise, scalar(0.0))?;
            let shape = noise.shape();
            for (i, code) in codes.iter().take(r.steps as usize).enumerate() {
                check(cancel)?;
                let flat = clean
                    .transpose_axes(&[0, 2, 3, 1])?
                    .reshape(&[-1, shape[1]])?;
                let mixed = linear_fp32(&flat, q)?
                    .reshape(&[shape[0], shape[2], shape[3], shape[1]])?
                    .transpose_axes(&[0, 3, 1, 2])?;
                let input = add(noise, mixed)?;
                let velocity = self
                    .transformer
                    .forward(&input, &[*code], self.positive, &[], cancel)?
                    .as_dtype(Dtype::Float32)?;
                clean = subtract(noise, velocity)?;
                mlx_rs::transforms::eval([&clean])?;
                check(cancel)?;
                progress(Progress::Step {
                    current: i as u32 + 1,
                    total: r.steps,
                });
            }
            check(cancel)?;
            return Ok(clean);
        }
        let sigmas = moss_image_schedule(r.steps, r.shift, r.training_shift)?;
        let actual = sigmas.len() - 1;
        let mut latent = noise.clone();
        for step in 0..actual {
            check(cancel)?;
            let time = [sigmas[step] * 1000.0];
            let conditional = self
                .transformer
                .forward(&latent, &time, self.positive, &[], cancel)?
                .as_dtype(Dtype::Float32)?;
            let mut prediction = conditional.clone();
            let active = r.skipping_active(step, actual);
            if r.guidance != 1.0 {
                let skip = if active {
                    r.uncond_skip_layers.as_slice()
                } else {
                    &[]
                };
                let uncond = self
                    .transformer
                    .forward(&latent, &time, self.negative, skip, cancel)?
                    .as_dtype(Dtype::Float32)?;
                prediction = add(
                    &uncond,
                    multiply(subtract(&conditional, &uncond)?, scalar(r.guidance))?,
                )?;
            }
            if let Some(slg) = r
                .slg
                .as_ref()
                .filter(|s| active && s.scale > 0.0 && !s.layers.is_empty())
            {
                let skipped = self
                    .transformer
                    .forward(&latent, &time, self.positive, &slg.layers, cancel)?
                    .as_dtype(Dtype::Float32)?;
                if r.guidance == 1.0 {
                    prediction = skipped.clone();
                }
                prediction = add(
                    prediction,
                    multiply(subtract(&conditional, &skipped)?, scalar(slg.scale))?,
                )?;
            }
            latent = add(
                latent,
                multiply(prediction, scalar(sigmas[step + 1] - sigmas[step]))?,
            )?;
            mlx_rs::transforms::eval([&latent])?;
            check(cancel)?;
            progress(Progress::Step {
                current: step as u32 + 1,
                total: actual as u32,
            });
        }
        check(cancel)?;
        Ok(latent)
    }
}
#[cfg(test)]
mod cpu_tests {
    use super::*;
    #[test]
    fn noise_is_fp32_finite_seeded_and_digest_sensitive() {
        let a = normal(19, 4097);
        assert_eq!(a, normal(19, 4097));
        assert_ne!(digest_f32(&a), digest_f32(&normal(20, 4097)));
        assert!(a.iter().all(|v| v.is_finite()));
        let mean = a.iter().sum::<f32>() / a.len() as f32;
        assert!(mean.abs() < 0.08);
        assert!(a.iter().any(|v| *v != ((*v * 128.0).round() / 128.0)));
    }
    #[test]
    fn cancellation_is_typed() {
        let c = CancelFlag::new();
        c.cancel();
        assert!(matches!(check(&c), Err(Error::Canceled)));
    }
}

#[cfg(test)]
mod host_readback_cpu_tests {
    use super::*;
    #[test]
    fn noncontiguous_b1_and_b2_latents_read_in_logical_nchw_order() {
        const CHILD: &str = "MOSS_TEST_LOGICAL_READBACK_CPU_CHILD";
        if std::env::var(CHILD).as_deref() != Ok("1") {
            // Process-default devices are global. Isolate CPU selection so this regression can
            // run beside the ordinary Metal fixtures even under a parallel libtest harness.
            let output = std::process::Command::new(std::env::current_exe().unwrap())
                .args(["--exact", "pipeline::host_readback_cpu_tests::noncontiguous_b1_and_b2_latents_read_in_logical_nchw_order", "--nocapture", "--test-threads=1"])
                .env(CHILD, "1")
                .output().unwrap();
            assert!(
                output.status.success(),
                "CPU readback child failed: {}\n{}",
                String::from_utf8_lossy(&output.stdout),
                String::from_utf8_lossy(&output.stderr)
            );
            return;
        }
        // This isolated child runs exactly this test. Both defaults are CPU, including C++
        // operations that omit their explicit stream; no parent or sibling device is changed.
        mlx_rs::Device::set_default(&mlx_rs::Device::new(mlx_rs::DeviceType::Cpu, 0));
        mlx_rs::with_new_default_stream(mlx_rs::Stream::cpu(), || {
            for batch in [1usize, 2] {
                let source: Vec<f32> = (0..batch * 2 * 3 * 4)
                    .map(|i| (i as f32 - 24.0) / 32.0)
                    .collect();
                let view = Array::from_slice(&source, &[batch as i32, 2, 3, 4])
                    .as_dtype(Dtype::Bfloat16)
                    .unwrap()
                    .transpose_axes(&[0, 3, 1, 2])
                    .unwrap();
                let mut expected = Vec::new();
                for b in 0..batch {
                    for c in 0..4 {
                        for y in 0..2 {
                            for x in 0..3 {
                                expected.push(source[((b * 2 + y) * 3 + x) * 4 + c]);
                            }
                        }
                    }
                }
                let physical = view.as_dtype(Dtype::Float32).unwrap();
                mlx_rs::transforms::eval([&physical]).unwrap();
                assert_ne!(
                    physical.try_as_slice::<f32>().unwrap(),
                    expected.as_slice(),
                    "fixture must discriminate physical NHWC storage from logical NCHW"
                );
                assert_eq!(logical_f32_readback(&view).unwrap(), expected);
                let samples = (0..batch)
                    .map(|b| mlx_gen::attention::slice_axis(&view, 0, b as i32, 1).unwrap())
                    .collect::<Vec<_>>();
                let concatenated =
                    mlx_rs::ops::concatenate_axis(&samples.iter().collect::<Vec<_>>(), 0).unwrap();
                assert_eq!(concatenated.shape(), [batch as i32, 4, 2, 3]);
                assert_eq!(logical_f32_readback(&concatenated).unwrap(), expected);
                if batch == 1 {
                    // MOSS's decoded NHWC image is handed to the shared converter as NCHW.
                    // Its explicit flatten must retain logical RGB order even for this strided crop.
                    let rgb = mlx_gen::attention::slice_axis(&view, 1, 0, 3).unwrap();
                    let image = mlx_gen::image::decoded_to_image(&rgb).unwrap();
                    let expected_rgb: Vec<u8> = source
                        .chunks_exact(4)
                        .flat_map(|pixel| pixel[..3].iter())
                        .map(|v| {
                            (((v * 0.5 + 0.5).clamp(0.0, 1.0)) * 255.0).round_ties_even() as u8
                        })
                        .collect();
                    assert_eq!((image.width, image.height), (3, 2));
                    assert_eq!(image.pixels, expected_rgb);
                }
            }
        });
    }
}
