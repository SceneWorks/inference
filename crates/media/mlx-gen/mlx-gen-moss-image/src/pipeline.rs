//! FP32 host noise and source-faithful ordinary/DMD sampling over the native graph.
use crate::precision::linear_fp32;
use crate::transformer::Transformer;
use mlx_gen::array::scalar;
use mlx_gen::driver::{
    run_mlx_driver_slices, DriverEvaluationKind, DriverRefusalReasons, DriverRunState,
    DriverStepInfo, DriverSupport, FamilyDriver, SliceInput, SliceSpec, StepHooks,
};
use mlx_gen::gen_core::guidance::{cfg as guidance_cfg, pag_apply};
use mlx_gen::gen_core::moss_image::{
    moss_image_schedule, MossImageAcceleration, MossImageResolved,
};
use mlx_gen::gen_core::{BoundaryState, EstimateSource};
use mlx_gen::{CancelFlag, Error, MlxLatentOps, Progress, Result};
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
/// Compute the MOSS DMD step input `noise + linear_fp32(clean, q)` in NCHW layout.
pub(crate) fn dmd_input(noise: &Array, q: &Array, clean: &Array) -> Result<Array> {
    let shape = noise.shape();
    let flat = clean
        .transpose_axes(&[0, 2, 3, 1])?
        .reshape(&[-1, shape[1]])?;
    let mixed = linear_fp32(&flat, q)?
        .reshape(&[shape[0], shape[2], shape[3], shape[1]])?
        .transpose_axes(&[0, 3, 1, 2])?;
    Ok(add(noise, mixed)?)
}

/// Typed `FamilyDriver` for MOSS-Image's fixed-noise DMD recurrence (`SW-6969`).
///
/// Carries the clean estimate `clean` across slices `[start, end)` (`clean_0 = 0`,
/// `input = noise + linear_fp32(clean, q)`, `clean = noise - velocity`), generic over its forward
/// seam for weights-free Tier 0 conformance testing.
pub struct MossDmdDriver<F> {
    support: DriverSupport,
    noise: Array,
    q: Array,
    codes: Vec<f32>,
    forward: F,
}

impl<F> MossDmdDriver<F>
where
    F: Fn(&Array, f32, usize, &CancelFlag) -> Result<Array>,
{
    pub fn new(noise: Array, q: Array, codes: Vec<f32>, forward: F) -> Self {
        Self {
            support: DriverSupport {
                driver_id: "moss_image_dmd",
                arbitrary_slice: true,
                carry_history: true,
                reset_history: true,
                renoise_from_clean: false,
                geometry_change: false,
                read_only_boundary_estimate: true,
                previous_estimate_capture: false,
                uses_global_rng: false,
                refusal_reasons: DriverRefusalReasons {
                    renoise_from_clean: "MOSS-Image DMD uses a fixed-noise linear mix and refuses external Renoise transitions",
                    geometry_change: "MOSS-Image DMD does not support mid-trajectory spatial resampling",
                    ..DriverRefusalReasons::default()
                },
            },
            noise,
            q,
            codes,
            forward,
        }
    }
}

impl<F> FamilyDriver<Array> for MossDmdDriver<F>
where
    F: Fn(&Array, f32, usize, &CancelFlag) -> Result<Array>,
{
    fn support(&self) -> &DriverSupport {
        &self.support
    }

    fn run_slice(
        &self,
        input: SliceInput<'_, Array>,
        state: &mut DriverRunState,
        hooks: &mut StepHooks<'_, Array>,
    ) -> mlx_gen::gen_core::Result<BoundaryState<Array>> {
        self.support.validate_slice(&input, state)?;
        if input.total_steps > self.codes.len() {
            return Err(mlx_gen::gen_core::Error::Unsupported(format!(
                "{}: total_steps={} exceeds DMD step_codes len={}",
                self.support.driver_id,
                input.total_steps,
                self.codes.len()
            )));
        }
        state.enter_slice(&input)?;

        let mut clean = match input.state {
            BoundaryState::XSigma(x)
            | BoundaryState::CleanEstimate { latent: x, .. }
            | BoundaryState::RouteSpecific { state: x, .. } => x,
        };

        for step in input.span.clone() {
            if input.cancel.is_cancelled() {
                return Err(mlx_gen::gen_core::Error::Canceled);
            }
            let code = self.codes[step];
            let next_code = self.codes.get(step + 1).copied().unwrap_or(0.0);
            let info = DriverStepInfo {
                pass: input.pass,
                phase: input.phase,
                outer_step: step,
                total_outer_steps: input.total_steps,
                eval_in_step: 0,
                kind: DriverEvaluationKind::Trajectory,
                sigma: code,
                next_sigma: next_code,
            };
            hooks.observe(&info, &clean)?;
            let mixed_input = dmd_input(&self.noise, &self.q, &clean)?;
            let velocity = (self.forward)(&mixed_input, code, step, input.cancel)?;
            state.record_evaluation(DriverEvaluationKind::Trajectory);
            clean = subtract(&self.noise, velocity)
                .map_err(|e| mlx_gen::gen_core::Error::Msg(e.to_string()))?;
            clean = hooks.post_step(&info, clean)?;
            hooks.materialize(&info, &clean)?;
            if input.cancel.is_cancelled() {
                return Err(mlx_gen::gen_core::Error::Canceled);
            }
            state.record_outer_step(step);
            hooks.emit_progress(&info);
        }
        state.finish_slice(input.span.end);

        if input.request_boundary_estimate {
            if input.cancel.is_cancelled() {
                return Err(mlx_gen::gen_core::Error::Canceled);
            }
            let est = if input.span.end < input.total_steps {
                let code = self.codes[input.span.end];
                let mixed_input = dmd_input(&self.noise, &self.q, &clean)?;
                let velocity = (self.forward)(&mixed_input, code, input.span.end, input.cancel)?;
                state.record_evaluation(DriverEvaluationKind::BoundaryEstimate);
                subtract(&self.noise, velocity)
                    .map_err(|e| mlx_gen::gen_core::Error::Msg(e.to_string()))?
            } else {
                state.record_evaluation(DriverEvaluationKind::BoundaryEstimate);
                clean.clone()
            };
            return Ok(BoundaryState::CleanEstimate {
                latent: est,
                source: EstimateSource::EndpointEvaluation,
            });
        }

        if input.span.end == input.total_steps {
            Ok(BoundaryState::CleanEstimate {
                latent: clean,
                source: EstimateSource::EndpointEvaluation,
            })
        } else {
            Ok(BoundaryState::XSigma(clean))
        }
    }
}

/// Typed `FamilyDriver` for MOSS-Image's ordinary FP32 flow-Euler loop (`SW-6969`).
pub struct MossEulerDriver<F> {
    support: DriverSupport,
    sigmas: Vec<f32>,
    forward: F,
}

impl<F> MossEulerDriver<F>
where
    F: Fn(&Array, usize, f32, &CancelFlag) -> Result<Array>,
{
    pub fn new(sigmas: Vec<f32>, forward: F) -> Self {
        Self {
            support: DriverSupport {
                driver_id: "moss_image_euler",
                arbitrary_slice: true,
                carry_history: true,
                reset_history: true,
                renoise_from_clean: false,
                geometry_change: false,
                read_only_boundary_estimate: true,
                previous_estimate_capture: false,
                uses_global_rng: false,
                refusal_reasons: DriverRefusalReasons {
                    renoise_from_clean:
                        "MOSS-Image Euler driver refuses external Renoise transitions",
                    geometry_change:
                        "MOSS-Image Euler driver does not support mid-trajectory spatial resampling",
                    ..DriverRefusalReasons::default()
                },
            },
            sigmas,
            forward,
        }
    }
}

impl<F> FamilyDriver<Array> for MossEulerDriver<F>
where
    F: Fn(&Array, usize, f32, &CancelFlag) -> Result<Array>,
{
    fn support(&self) -> &DriverSupport {
        &self.support
    }

    fn run_slice(
        &self,
        input: SliceInput<'_, Array>,
        state: &mut DriverRunState,
        hooks: &mut StepHooks<'_, Array>,
    ) -> mlx_gen::gen_core::Result<BoundaryState<Array>> {
        self.support.validate_slice(&input, state)?;
        let expected_steps = self.sigmas.len().saturating_sub(1);
        if input.total_steps != expected_steps {
            return Err(mlx_gen::gen_core::Error::Unsupported(format!(
                "{}: total_steps={} does not match sigma schedule steps={expected_steps}",
                self.support.driver_id, input.total_steps
            )));
        }
        state.enter_slice(&input)?;

        let mut latent = match input.state {
            BoundaryState::XSigma(x)
            | BoundaryState::CleanEstimate { latent: x, .. }
            | BoundaryState::RouteSpecific { state: x, .. } => x,
        };

        for step in input.span.clone() {
            if input.cancel.is_cancelled() {
                return Err(mlx_gen::gen_core::Error::Canceled);
            }
            let sigma = self.sigmas[step];
            let next_sigma = self.sigmas[step + 1];
            let info = DriverStepInfo {
                pass: input.pass,
                phase: input.phase,
                outer_step: step,
                total_outer_steps: input.total_steps,
                eval_in_step: 0,
                kind: DriverEvaluationKind::Trajectory,
                sigma,
                next_sigma,
            };
            hooks.observe(&info, &latent)?;
            let prediction = (self.forward)(&latent, step, sigma, input.cancel)?;
            state.record_evaluation(DriverEvaluationKind::Trajectory);
            latent = add(
                latent,
                multiply(prediction, scalar(next_sigma - sigma))
                    .map_err(|e| mlx_gen::gen_core::Error::Msg(e.to_string()))?,
            )
            .map_err(|e| mlx_gen::gen_core::Error::Msg(e.to_string()))?;
            latent = hooks.post_step(&info, latent)?;
            hooks.materialize(&info, &latent)?;
            if input.cancel.is_cancelled() {
                return Err(mlx_gen::gen_core::Error::Canceled);
            }
            state.record_outer_step(step);
            hooks.emit_progress(&info);
        }
        state.finish_slice(input.span.end);

        if input.request_boundary_estimate {
            if input.cancel.is_cancelled() {
                return Err(mlx_gen::gen_core::Error::Canceled);
            }
            let est = if input.span.end < input.total_steps {
                let sigma = self.sigmas[input.span.end];
                let prediction = (self.forward)(&latent, input.span.end, sigma, input.cancel)?;
                state.record_evaluation(DriverEvaluationKind::BoundaryEstimate);
                subtract(
                    &latent,
                    multiply(prediction, scalar(sigma))
                        .map_err(|e| mlx_gen::gen_core::Error::Msg(e.to_string()))?,
                )
                .map_err(|e| mlx_gen::gen_core::Error::Msg(e.to_string()))?
            } else {
                state.record_evaluation(DriverEvaluationKind::BoundaryEstimate);
                latent.clone()
            };
            return Ok(BoundaryState::CleanEstimate {
                latent: est,
                source: EstimateSource::EndpointEvaluation,
            });
        }

        Ok(BoundaryState::XSigma(latent))
    }
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
        let ops = MlxLatentOps;
        if !matches!(r.acceleration, MossImageAcceleration::Off) {
            let (q, codes) = self
                .dataspace
                .ok_or_else(|| Error::Unsupported("MOSS DMD dataspace missing".into()))?;
            let clean = multiply(noise, scalar(0.0))?;
            let total_steps = r.steps as usize;
            let step_codes: Vec<f32> = codes.iter().take(total_steps).copied().collect();
            let driver = MossDmdDriver::new(
                noise.clone(),
                q.clone(),
                step_codes,
                |input, code, i, cancel| {
                    let cond = self
                        .transformer
                        .forward(input, &[code], self.positive, &[], cancel)?
                        .as_dtype(Dtype::Float32)?;
                    let pag_scale = r.pag_step_scale(i, total_steps);
                    if pag_scale > 0.0 {
                        let pag_layers = r
                            .pag
                            .as_ref()
                            .map_or(&[][..], |pag| pag.step_layers(i, total_steps));
                        let perturbed = self
                            .transformer
                            .forward_with_pag(
                                input,
                                &[code],
                                self.positive,
                                &[],
                                pag_layers,
                                cancel,
                            )?
                            .as_dtype(Dtype::Float32)?;
                        Ok(pag_apply(&ops, &cond, &cond, &perturbed, pag_scale)?)
                    } else {
                        Ok(cond)
                    }
                },
            );
            let mut run_state = DriverRunState::new();
            let mut on_materialize = |_info: &DriverStepInfo, lat: &Array| {
                mlx_rs::transforms::eval([lat])
                    .map_err(|e| mlx_gen::gen_core::Error::Msg(e.to_string()))
            };
            let mut hooks = StepHooks::new(progress);
            hooks.on_materialize = Some(&mut on_materialize);
            let out = run_mlx_driver_slices(
                &driver,
                [SliceSpec::full(total_steps)],
                BoundaryState::XSigma(clean),
                total_steps,
                0,
                cancel,
                None,
                &mut run_state,
                &mut hooks,
                |_, state, _| Ok(state),
            )?;
            check(cancel)?;
            return Ok(match out {
                BoundaryState::XSigma(x)
                | BoundaryState::CleanEstimate { latent: x, .. }
                | BoundaryState::RouteSpecific { state: x, .. } => x,
            });
        }
        let sigmas = moss_image_schedule(r.steps, r.shift, r.training_shift)?;
        let actual = sigmas.len() - 1;
        let driver = MossEulerDriver::new(sigmas, |latent, step, sigma, cancel| {
            let time = [sigma * 1000.0];
            let conditional = self
                .transformer
                .forward(latent, &time, self.positive, &[], cancel)?
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
                    .forward(latent, &time, self.negative, skip, cancel)?
                    .as_dtype(Dtype::Float32)?;
                prediction = guidance_cfg(&ops, &conditional, &uncond, r.guidance)?;
            }
            if let Some(slg) = r
                .slg
                .as_ref()
                .filter(|s| active && s.scale > 0.0 && !s.layers.is_empty())
            {
                let skipped = self
                    .transformer
                    .forward(latent, &time, self.positive, &slg.layers, cancel)?
                    .as_dtype(Dtype::Float32)?;
                if r.guidance == 1.0 {
                    prediction = skipped.clone();
                }
                prediction = pag_apply(&ops, &prediction, &conditional, &skipped, slg.scale)?;
            }
            let pag_scale = r.pag_step_scale(step, actual);
            if pag_scale > 0.0 {
                let pag_layers = r
                    .pag
                    .as_ref()
                    .map_or(&[][..], |pag| pag.step_layers(step, actual));
                let perturbed = self
                    .transformer
                    .forward_with_pag(latent, &time, self.positive, &[], pag_layers, cancel)?
                    .as_dtype(Dtype::Float32)?;
                prediction = pag_apply(&ops, &prediction, &conditional, &perturbed, pag_scale)?;
            }
            Ok(prediction)
        });
        let mut run_state = DriverRunState::new();
        let mut on_materialize = |_info: &DriverStepInfo, lat: &Array| {
            mlx_rs::transforms::eval([lat])
                .map_err(|e| mlx_gen::gen_core::Error::Msg(e.to_string()))
        };
        let mut hooks = StepHooks::new(progress);
        hooks.on_materialize = Some(&mut on_materialize);
        let out = run_mlx_driver_slices(
            &driver,
            [SliceSpec::full(actual)],
            BoundaryState::XSigma(noise.clone()),
            actual,
            0,
            cancel,
            None,
            &mut run_state,
            &mut hooks,
            |_, state, _| Ok(state),
        )?;
        check(cancel)?;
        Ok(match out {
            BoundaryState::XSigma(x)
            | BoundaryState::CleanEstimate { latent: x, .. }
            | BoundaryState::RouteSpecific { state: x, .. } => x,
        })
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

    #[test]
    fn moss_dmd_and_euler_drivers_pass_tier0_conformance() {
        use mlx_gen::driver::conformance::{verify_driver_conformance, ConformanceCase};

        let q = Array::from_slice(&[1.0f32, 0.25, -0.1, 0.9], &[2, 2]);
        let noise = Array::from_slice(
            &[0.5f32, -0.25, 0.75, 0.1, -0.3, 0.2, 0.4, -0.6],
            &[1, 2, 2, 2],
        );
        let codes = vec![1000.0f32, 750.0, 500.0, 250.0];
        let dmd_report = verify_driver_conformance(ConformanceCase {
            total_steps: 4,
            split_at: 2,
            seed: 13,
            base_geometry: (512, 512),
            alt_geometry: (1024, 1024),
            make_driver: {
                let noise = noise.clone();
                let q = q.clone();
                let codes = codes.clone();
                move || {
                    Box::new(MossDmdDriver::new(
                        noise.clone(),
                        q.clone(),
                        codes.clone(),
                        |input, code, step, _cancel| {
                            let scale = scalar(0.1 + 0.0001 * code + 0.02 * step as f32);
                            Ok(multiply(input, scale)?)
                        },
                    ))
                }
            },
            make_initial: || Array::from_slice(&[0.0f32; 8], &[1, 2, 2, 2]),
            latents_equal: |a, b| {
                logical_f32_readback(a)
                    .unwrap()
                    .iter()
                    .zip(logical_f32_readback(b).unwrap().iter())
                    .all(|(x, y)| x.to_bits() == y.to_bits())
            },
        })
        .expect("MossDmdDriver must pass Tier 0 conformance");
        assert!(dmd_report.verified_inert_split);
        assert!(dmd_report.verified_cancellation);
        assert_eq!(dmd_report.unsplit_evaluations, 4);

        let sigmas = moss_image_schedule(4, 1.0, 1.0).unwrap();
        let euler_report = verify_driver_conformance(ConformanceCase {
            total_steps: 4,
            split_at: 2,
            seed: 13,
            base_geometry: (512, 512),
            alt_geometry: (1024, 1024),
            make_driver: move || {
                Box::new(MossEulerDriver::new(
                    sigmas.clone(),
                    |latent, step, sigma, _cancel| {
                        let scale = scalar(0.2 + 0.1 * sigma + 0.03 * step as f32);
                        Ok(multiply(latent, scale)?)
                    },
                ))
            },
            make_initial: move || noise.clone(),
            latents_equal: |a, b| {
                logical_f32_readback(a)
                    .unwrap()
                    .iter()
                    .zip(logical_f32_readback(b).unwrap().iter())
                    .all(|(x, y)| x.to_bits() == y.to_bits())
            },
        })
        .expect("MossEulerDriver must pass Tier 0 conformance");
        assert!(euler_report.verified_inert_split);
        assert!(euler_report.verified_cancellation);
        assert_eq!(euler_report.unsplit_evaluations, 4);
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
