//! Corrected Stable Audio 3 sampler math and mutation gates.

use std::cell::RefCell;
use std::path::{Path, PathBuf};
use std::rc::Rc;
use std::time::Instant;

use candle_audio_stable_audio_3::candle_audio::candle_core::{DType, Device, Tensor};
use candle_audio_stable_audio_3::config::DiffusionObjective;
use candle_audio_stable_audio_3::dit::{Guidance, StableAudio3Dit};
use candle_audio_stable_audio_3::sampler::{
    adapt_sample_size_for_max, build_schedule, effective_schedule_lengths, initialize_latents,
    padding_mask, resource_estimate, sample, sample_dit, sample_initialized, sample_with_callback,
    sample_with_host_timestep, validate_guidance, DistributionShift, GuidanceInterval,
    InjectedNoise, NoiseSource, SamplerKind, Schedule, SeededNoise,
};
use candle_audio_stable_audio_3::weights::SnapshotLayout;
use candle_nn::VarBuilder;

/// The crate's three-way real-weight device selector, identical to `same_oracle.rs` and
/// `chunked_oracle.rs`.
///
/// The three real-weight cases below each branched on `SA3_TEST_METAL` alone, so on the CUDA lanes
/// — which set `SA3_TEST_CUDA` — they silently executed on `Device::Cpu`. A requested backend that
/// is unavailable is a hard failure, never a fallback.
fn test_device() -> Device {
    if std::env::var_os("SA3_TEST_METAL").is_some() {
        Device::new_metal(0).expect("SA3_TEST_METAL requested but Metal is unavailable")
    } else if std::env::var_os("SA3_TEST_CUDA").is_some() {
        #[cfg(feature = "cuda")]
        {
            Device::new_cuda(0).expect("SA3_TEST_CUDA requested but CUDA is unavailable")
        }
        #[cfg(not(feature = "cuda"))]
        {
            panic!("SA3_TEST_CUDA requires --features cuda")
        }
    } else {
        Device::Cpu
    }
}

fn values(tensor: &Tensor) -> Vec<f32> {
    tensor
        .to_dtype(DType::F32)
        .unwrap()
        .flatten_all()
        .unwrap()
        .to_vec1()
        .unwrap()
}

fn assert_close(actual: &[f32], expected: &[f32], tolerance: f32) {
    assert_eq!(actual.len(), expected.len());
    for (index, (&actual, &expected)) in actual.iter().zip(expected).enumerate() {
        assert!(
            (actual - expected).abs() <= tolerance,
            "value {index}: {actual} != {expected}"
        );
    }
}

fn no_noise() -> InjectedNoise {
    InjectedNoise::new(Vec::new())
}

#[test]
fn frozen_torch_euler_dpmpp_and_scalar_rk4_match_every_step() {
    let path = Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("../../../docs/migration/sa3-sampler-reference/sampler.json");
    let oracle: serde_json::Value = serde_json::from_slice(&std::fs::read(path).unwrap()).unwrap();
    let schedule = oracle["schedules"]["shared"]
        .as_array()
        .unwrap()
        .iter()
        .map(|value| value.as_f64().unwrap() as f32)
        .collect::<Vec<_>>();
    let device = Device::Cpu;
    let initial = Tensor::from_vec(vec![0.75f32], (1, 1, 1), &device).unwrap();
    let cases = [
        (SamplerKind::Euler, "euler"),
        (SamplerKind::Dpmpp, "dpmpp"),
        (SamplerKind::Rk4, "rk4FrozenScalar"),
    ];
    for (kind, key) in cases {
        let mut unused_noise = no_noise();
        let output = sample(
            kind,
            &initial,
            &Schedule::Shared(schedule.clone()),
            None,
            &mut unused_noise,
            true,
            |x, t| Ok((((x * 0.25)? + (t.unsqueeze(1)?.unsqueeze(2)? * 0.5)?)? - 0.125)?),
        )
        .unwrap();
        let expected = oracle["trajectories"]["frozenTorch"][key]
            .as_array()
            .unwrap();
        assert_eq!(expected.len(), output.trajectory.len() + 1);
        for (index, step) in output.trajectory.iter().enumerate() {
            for (field, actual) in [
                ("x", &step.x),
                ("t", &step.timestep),
                ("denoised", &step.denoised),
            ] {
                let expected_values = expected[index][field]
                    .as_array()
                    .unwrap()
                    .iter()
                    .map(|value| value.as_f64().unwrap() as f32)
                    .collect::<Vec<_>>();
                assert_close(&values(actual), &expected_values, 2e-6);
            }
        }
        let expected_final = expected.last().unwrap()["final"]
            .as_array()
            .unwrap()
            .iter()
            .map(|value| value.as_f64().unwrap() as f32)
            .collect::<Vec<_>>();
        assert_close(&values(&output.latents), &expected_final, 2e-6);
    }
}

#[test]
fn default_logsnr_schedule_matches_frozen_p0_and_strength_is_monotonic() {
    let shift = DistributionShift::LogSnr {
        anchor_length: 2000,
        anchor_logsnr: -6.2,
        rate: 0.0,
        logsnr_end: 2.0,
    };
    let Schedule::Shared(full) = build_schedule(8, 1.0, &shift, None, 16).unwrap() else {
        panic!("expected shared schedule")
    };
    assert_close(
        &full,
        &[
            1.0, 0.9943756, 0.98448026, 0.95791227, 0.8909032, 0.74554664, 0.5124974, 0.273885, 0.0,
        ],
        2e-7,
    );
    for strength in [0.5, 0.9] {
        let Schedule::Shared(partial) = build_schedule(8, strength, &shift, None, 16).unwrap()
        else {
            panic!("expected shared schedule")
        };
        assert_eq!(partial[0], strength);
        assert_eq!(*partial.last().unwrap(), 0.0);
        assert!(partial.windows(2).all(|pair| pair[1] <= pair[0]));
        for (&actual, &normalized) in partial.iter().zip(&full) {
            assert!((actual - normalized * strength).abs() < 2e-7);
        }
        // Frozen Python's shift-after-scaling mutation increases on its first step.
        let broken_first_after = shift_value_for_mutation(strength * 0.875);
        assert!(broken_first_after > strength);
    }
}

fn shift_value_for_mutation(t: f32) -> f32 {
    let start = -6.2f32;
    let end = 2.0f32;
    1.0 / (1.0 + (end - t * (end - start)).exp())
}

#[test]
fn all_shift_families_and_typed_b_equals_steps_plus_one_are_finite() {
    let shifts = [
        DistributionShift::Identity,
        DistributionShift::Flux {
            min_length: 256,
            max_length: 4096,
            alpha_min: 1.0,
            alpha_max: 6.93,
        },
        DistributionShift::Full {
            min_length: 256,
            max_length: 4096,
            base_shift: 0.5,
            max_shift: 1.15,
            use_sine: false,
        },
        DistributionShift::Full {
            min_length: 256,
            max_length: 4096,
            base_shift: 0.5,
            max_shift: 1.15,
            use_sine: true,
        },
        DistributionShift::LogSnr {
            anchor_length: 2000,
            anchor_logsnr: -6.2,
            rate: 1.0,
            logsnr_end: 2.0,
        },
    ];
    for shift in shifts {
        let schedule = build_schedule(3, 1.0, &shift, Some(&[256, 512, 1024, 4096]), 16).unwrap();
        assert_eq!(schedule.batch(), Some(4)); // B == S+1, still unambiguously batched.
        assert_eq!(schedule.steps(), 3);
        let Schedule::PerExample(rows) = schedule else {
            unreachable!()
        };
        for row in rows {
            assert!(row.iter().all(|value| value.is_finite()));
            assert!(row.windows(2).all(|pair| pair[1] <= pair[0]));
        }
    }

    let exact_cases = [
        (
            DistributionShift::Flux {
                min_length: 256,
                max_length: 4096,
                alpha_min: 1.0,
                alpha_max: 6.93,
            },
            1024,
            vec![1.0, 0.88760847, 0.7247067, 0.46737581, 0.0],
        ),
        (
            DistributionShift::Full {
                min_length: 256,
                max_length: 4096,
                base_shift: 0.5,
                max_shift: 1.15,
                use_sine: false,
            },
            1024,
            vec![1.0, 0.8492348, 0.6524894, 0.3849448, 0.0],
        ),
        (
            DistributionShift::Full {
                min_length: 256,
                max_length: 4096,
                base_shift: 0.5,
                max_shift: 1.15,
                use_sine: true,
            },
            1024,
            vec![1.0, 0.97208863, 0.8546768, 0.5684905, 0.0],
        ),
    ];
    for (shift, length, expected) in exact_cases {
        let Schedule::Shared(actual) = build_schedule(4, 1.0, &shift, None, length).unwrap() else {
            unreachable!()
        };
        assert_close(&actual, &expected, 2e-6);
    }
}

#[test]
fn adaptation_matches_short_long_boundaries_and_all_six_maxima() {
    let maxima = [
        5_292_032, 5_292_032, 5_324_800, 5_324_800, 16_777_216, 16_777_216,
    ];
    for maximum in maxima {
        let short = adapt_sample_size_for_max(maximum, &[Some(0.25)], 6.0).unwrap();
        assert_eq!(short.sample_size, 278_528);
        assert_eq!(short.latent_length, 68);
        assert_eq!(short.effective_lengths, Some(vec![3]));
        assert_eq!(short.valid_lengths, vec![67]);

        let long = adapt_sample_size_for_max(maximum, &[Some(30.0)], 6.0).unwrap();
        assert_eq!(long.sample_size, 1_589_248);
        assert_eq!(long.latent_length, 388);
        assert_eq!(long.valid_lengths, vec![387]);

        let clamped = adapt_sample_size_for_max(maximum, &[Some(1_000.0)], 6.0).unwrap();
        assert_eq!(clamped.sample_size, maximum);
    }
    let exact = adapt_sample_size_for_max(5_292_032, &[Some(2_880.0 / 44_100.0)], 0.0).unwrap();
    assert_eq!(exact.sample_size, 8_192);
    let mixed = adapt_sample_size_for_max(5_324_800, &[Some(1.0), None], 6.0).unwrap();
    assert_eq!(mixed.sample_size, 311_296);
    assert_eq!(mixed.effective_lengths, None);
    let zero = adapt_sample_size_for_max(5_292_032, &[Some(0.0)], 6.0).unwrap();
    assert_eq!(zero.sample_size, 5_292_032);
    let unaligned_clamp = adapt_sample_size_for_max(10_001, &[Some(100.0)], 6.0).unwrap();
    assert_eq!(unaligned_clamp.sample_size, 10_001);
    assert_eq!(unaligned_clamp.latent_length, 2);
    for (samples, expected) in [
        (4_095usize, 8_192usize),
        (4_096, 8_192),
        (4_097, 8_192),
        (8_191, 8_192),
        (8_192, 8_192),
        (8_193, 16_384),
    ] {
        let geometry =
            adapt_sample_size_for_max(5_292_032, &[Some(samples as f64 / 44_100.0)], 0.0).unwrap();
        assert_eq!(geometry.sample_size, expected, "samples={samples}");
    }
    assert!(adapt_sample_size_for_max(100, &[Some(f64::NAN)], 6.0).is_err());
    assert!(adapt_sample_size_for_max(100, &[Some(1.0)], f64::INFINITY).is_err());
}

#[test]
fn padding_mask_values_are_exact() {
    let mask = padding_mask(&[1, 3], 4, &Device::Cpu).unwrap();
    assert_eq!(
        mask.flatten_all().unwrap().to_vec1::<u8>().unwrap(),
        vec![1, 0, 0, 0, 1, 1, 1, 0]
    );
}

#[test]
fn raw_effective_length_excludes_headroom_and_missing_falls_back_globally() {
    assert_eq!(
        effective_schedule_lengths(&[Some(0.25), Some(30.0)]).unwrap(),
        Some(vec![3, 323])
    );
    assert_eq!(
        effective_schedule_lengths(&[Some(0.25), None]).unwrap(),
        None
    );
    assert!(effective_schedule_lengths(&[Some(-1.0)]).is_err());
}

#[test]
fn init_mix_bounds_shapes_and_zero_short_circuit_are_explicit() {
    let device = Device::Cpu;
    let init = Tensor::from_vec(vec![2f32, 4.0], (1, 1, 2), &device).unwrap();
    let noise = Tensor::from_vec(vec![10f32, 20.0], (1, 1, 2), &device).unwrap();
    assert_close(
        &values(&initialize_latents(&noise, Some(&init), 0.0).unwrap()),
        &[2.0, 4.0],
        0.0,
    );
    assert_close(
        &values(&initialize_latents(&noise, Some(&init), 0.25).unwrap()),
        &[4.0, 8.0],
        0.0,
    );
    assert_close(
        &values(&initialize_latents(&noise, Some(&init), 1.0).unwrap()),
        &[10.0, 20.0],
        0.0,
    );
    assert!(initialize_latents(&noise, None, 0.5).is_err());
    assert!(initialize_latents(&noise, Some(&init), 1.01).is_err());
    assert!(initialize_latents(&noise, Some(&init), f32::NAN).is_err());

    let calls = Rc::new(RefCell::new(0usize));
    let observed = Rc::clone(&calls);
    let mut zero_noise = no_noise();
    let output = sample_initialized(
        SamplerKind::Euler,
        &noise,
        Some(&init),
        0.0,
        &Schedule::Shared(vec![0.0, 0.0]),
        None,
        &mut zero_noise,
        true,
        move |x, _| {
            *observed.borrow_mut() += 1;
            Ok(x.clone())
        },
    )
    .unwrap();
    assert_eq!(output.model_calls, 0);
    assert_eq!(*calls.borrow(), 0);
    assert_close(&values(&output.latents), &[2.0, 4.0], 0.0);
    let mut mismatched_noise = no_noise();
    assert!(sample_initialized(
        SamplerKind::Euler,
        &noise,
        Some(&init),
        0.5,
        &Schedule::Shared(vec![1.0, 0.0]),
        None,
        &mut mismatched_noise,
        false,
        |x, _| Ok(x.clone()),
    )
    .is_err());
}

#[test]
fn live_callback_is_pre_update_and_error_cancels_before_mutation() {
    let device = Device::Cpu;
    let initial = Tensor::from_vec(vec![1f32], (1, 1, 1), &device).unwrap();
    let schedule = Schedule::Shared(vec![1.0, 0.5, 0.0]);
    let seen = Rc::new(RefCell::new(Vec::new()));
    let seen_callback = Rc::clone(&seen);
    let calls = Rc::new(RefCell::new(0usize));
    let model_calls = Rc::clone(&calls);
    let mut callback = move |step: &candle_audio_stable_audio_3::sampler::SampleStep| {
        seen_callback
            .borrow_mut()
            .push((step.index, values(&step.x), values(&step.denoised)));
        if step.index == 1 {
            Err(candle_audio_stable_audio_3::candle_audio::AudioError::Msg(
                "cancelled".into(),
            ))
        } else {
            Ok(())
        }
    };
    let result = sample_with_callback(
        SamplerKind::Euler,
        &initial,
        &schedule,
        None,
        &mut no_noise(),
        false,
        Some(&mut callback),
        move |x, _| {
            *model_calls.borrow_mut() += 1;
            Ok(x.clone())
        },
    );
    assert!(result.is_err());
    assert_eq!(*calls.borrow(), 2);
    assert_eq!(seen.borrow()[0], (0, vec![1.0], vec![0.0]));
    assert_eq!(seen.borrow()[1].0, 1);
    assert_close(&seen.borrow()[1].1, &[0.5], 0.0);
}

#[test]
fn guidance_interval_is_exact_and_fails_closed_for_mixed_batches() {
    let configured = Guidance {
        cfg_scale: 3.0,
        apg_scale: 0.5,
        cfg_norm_threshold: 0.25,
        scale_phi: 0.3,
    };
    let interval = GuidanceInterval {
        min_sigma: 0.2,
        max_sigma: 0.8,
    };
    assert_eq!(
        interval
            .guidance_for_values(&[0.5, 0.7], configured)
            .unwrap()
            .cfg_scale,
        3.0
    );
    assert_eq!(
        interval
            .guidance_for_values(&[0.9, 0.95], configured)
            .unwrap()
            .cfg_scale,
        1.0
    );
    assert!(interval
        .guidance_for_values(&[0.5, 0.9], configured)
        .is_err());
    assert!(GuidanceInterval {
        min_sigma: -0.1,
        max_sigma: 1.0,
    }
    .guidance_for_values(&[0.5], configured)
    .is_err());
    for invalid in [
        Guidance {
            cfg_scale: f64::NAN,
            ..configured
        },
        Guidance {
            cfg_scale: -1.0,
            ..configured
        },
        Guidance {
            apg_scale: 1.1,
            ..configured
        },
        Guidance {
            scale_phi: -0.1,
            ..configured
        },
        Guidance {
            cfg_norm_threshold: f64::INFINITY,
            ..configured
        },
    ] {
        assert!(validate_guidance(invalid).is_err());
    }
}

#[test]
fn rk4_host_timestep_surface_covers_every_stage_and_terminal_clamp() {
    let device = Device::Cpu;
    let initial = Tensor::from_vec(vec![1f32, 2.0], (2, 1, 1), &device).unwrap();
    let schedule = Schedule::PerExample(vec![vec![1.0, 0.5, 0.0], vec![1.0, 0.25, 0.0]]);
    let stages = Rc::new(RefCell::new(Vec::<Vec<f32>>::new()));
    let captured = Rc::clone(&stages);
    sample_with_host_timestep(
        SamplerKind::Rk4,
        &initial,
        &schedule,
        None,
        &mut no_noise(),
        false,
        None,
        move |x, _, host_timestep| {
            captured.borrow_mut().push(host_timestep.to_vec());
            Tensor::zeros_like(x).map_err(Into::into)
        },
    )
    .unwrap();
    let expected = [
        vec![1.0, 1.0],
        vec![0.75, 0.625],
        vec![0.75, 0.625],
        vec![0.5, 0.25],
        vec![0.5, 0.25],
        vec![0.25, 0.125],
        vec![0.25, 0.125],
        vec![1e-5, 1e-5],
    ];
    assert_eq!(*stages.borrow(), expected);
}

#[test]
fn nonzero_per_example_euler_dpmpp_and_pingpong_are_locked() {
    let device = Device::Cpu;
    let initial = Tensor::from_vec(vec![0.75f32, -0.4], (2, 1, 1), &device).unwrap();
    let schedule = Schedule::PerExample(vec![vec![1.0, 0.7, 0.2, 0.0], vec![1.0, 0.55, 0.1, 0.0]]);
    let field = |x: &Tensor, t: &Tensor| {
        Ok((((x * 0.25)? + (t.unsqueeze(1)?.unsqueeze(2)? * 0.5)?)? - 0.125)?)
    };
    for (kind, expected) in [
        (SamplerKind::Euler, vec![0.38128906, -0.5115199]),
        (SamplerKind::Dpmpp, vec![0.3858859, -0.5056215]),
    ] {
        let output = sample(
            kind,
            &initial,
            &schedule,
            None,
            &mut no_noise(),
            false,
            field,
        )
        .unwrap();
        assert_close(&values(&output.latents), &expected, 3e-6);
    }
    let draws = vec![
        Tensor::from_vec(vec![0.5f32, -0.25], (2, 1, 1), &device).unwrap(),
        Tensor::from_vec(vec![1f32, 0.75], (2, 1, 1), &device).unwrap(),
        Tensor::from_vec(vec![99f32, 99.0], (2, 1, 1), &device).unwrap(),
    ];
    let output = sample(
        SamplerKind::Pingpong,
        &initial,
        &schedule,
        None,
        &mut InjectedNoise::new(draws),
        false,
        field,
    )
    .unwrap();
    assert_close(&values(&output.latents), &[0.33001876, -0.32572606], 3e-6);
    assert_eq!(output.noise_draws, 3);
}

#[test]
fn euler_sign_callbacks_and_mask_shape_are_exact() {
    let device = Device::Cpu;
    let initial = Tensor::from_vec(vec![1f32, 9.0], (1, 1, 2), &device).unwrap();
    let schedule = Schedule::Shared(vec![1.0, 0.5, 0.0]);
    let mask = padding_mask(&[1], 2, &device).unwrap();
    let mut noise = no_noise();
    let output = sample(
        SamplerKind::Euler,
        &initial,
        &schedule,
        Some(&mask),
        &mut noise,
        true,
        |x, _| Ok((x * 2f64)?),
    )
    .unwrap();
    assert_eq!(output.model_calls, 2);
    assert_eq!(output.trajectory.len(), 2);
    // The sampler validates/forwards the mask but does not duplicate DiT's internal V-zero rule.
    assert_close(&values(&output.latents), &[0.0, 0.0], 0.0);
    assert_close(&values(&output.trajectory[0].denoised), &[-1.0, -9.0], 0.0);
}

#[test]
fn corrected_rk4_supports_per_example_schedule_and_terminal_clamp() {
    let device = Device::Cpu;
    let initial = Tensor::from_vec(vec![1f32, 2.0], (2, 1, 1), &device).unwrap();
    let schedule = Schedule::PerExample(vec![vec![1.0, 0.5, 0.0], vec![1.0, 0.25, 0.0]]);
    let seen = Rc::new(RefCell::new(Vec::<Vec<f32>>::new()));
    let capture = Rc::clone(&seen);
    let mut noise = no_noise();
    let output = sample(
        SamplerKind::Rk4,
        &initial,
        &schedule,
        None,
        &mut noise,
        true,
        move |x, t| {
            capture.borrow_mut().push(values(t));
            Ok(x.clone()) // dx/dt=x; each element has its own dt.
        },
    )
    .unwrap();
    assert_eq!(output.model_calls, 8);
    assert_eq!(seen.borrow().len(), 8);
    assert_close(seen.borrow().last().unwrap(), &[1e-5, 1e-5], 1e-8);
    // Frozen scalar RK4 formula, independently evaluated for each schedule row.
    let scalar_rk4 = |mut x: f32, row: &[f32]| {
        for pair in row.windows(2) {
            let dt = pair[1] - pair[0];
            let k1 = x;
            let k2 = x + dt * k1 / 2.0;
            let k3 = x + dt * k2 / 2.0;
            let k4 = x + dt * k3;
            x += dt * (k1 + 2.0 * k2 + 2.0 * k3 + k4) / 6.0;
        }
        x
    };
    assert_close(
        &values(&output.latents),
        &[
            scalar_rk4(1.0, &[1.0, 0.5, 0.0]),
            scalar_rk4(2.0, &[1.0, 0.25, 0.0]),
        ],
        2e-6,
    );
}

#[test]
fn rf_dpmpp_zero_velocity_is_identity_for_shared_and_batched_schedules() {
    let device = Device::Cpu;
    for schedule in [
        Schedule::Shared(vec![1.0, 0.8, 0.3, 0.0]),
        Schedule::PerExample(vec![vec![1.0, 0.8, 0.3, 0.0], vec![1.0, 0.6, 0.2, 0.0]]),
    ] {
        let batch = schedule.batch().unwrap_or(2);
        let initial = Tensor::from_vec(
            (1..=batch).map(|v| v as f32).collect::<Vec<_>>(),
            (batch, 1, 1),
            &device,
        )
        .unwrap();
        let mut noise = no_noise();
        let output = sample(
            SamplerKind::Dpmpp,
            &initial,
            &schedule,
            None,
            &mut noise,
            true,
            |x, _| Tensor::zeros_like(x).map_err(Into::into),
        )
        .unwrap();
        assert_close(&values(&output.latents), &values(&initial), 2e-6);
        assert_eq!(output.model_calls, 3);
    }
}

#[test]
fn pingpong_consumes_terminal_draw_and_seeded_requests_are_reproducible() {
    let device = Device::Cpu;
    let initial = Tensor::zeros((1, 1, 2), DType::F32, &device).unwrap();
    let schedule = Schedule::Shared(vec![1.0, 0.5, 0.0]);
    let draws = vec![
        Tensor::from_vec(vec![2f32, 4.0], (1, 1, 2), &device).unwrap(),
        Tensor::from_vec(vec![99f32, 99.0], (1, 1, 2), &device).unwrap(),
    ];
    let mut injected = InjectedNoise::new(draws);
    let output = sample(
        SamplerKind::Pingpong,
        &initial,
        &schedule,
        None,
        &mut injected,
        true,
        |x, _| Tensor::zeros_like(x).map_err(Into::into),
    )
    .unwrap();
    assert_eq!(output.noise_draws, 2);
    assert_close(&values(&output.latents), &[1.0, 2.0], 0.0);

    let run = |seed| {
        let mut seeded = SeededNoise::new(seed);
        // One request-local stream owns the initial noise, each step draw, and terminal eager draw.
        let request_initial = seeded.standard_normal_like(&initial).unwrap();
        let result = sample(
            SamplerKind::Pingpong,
            &request_initial,
            &schedule,
            None,
            &mut seeded,
            false,
            |x, _| Tensor::zeros_like(x).map_err(Into::into),
        )
        .unwrap();
        (
            values(&request_initial),
            values(&result.latents),
            seeded.draws(),
        )
    };
    let first = run(14542);
    let second = run(14542);
    assert_eq!(first, second);
    assert_eq!(first.2, 3);
}

#[test]
fn truthful_defaults_and_fail_closed_objective_are_locked() {
    assert_eq!(
        SamplerKind::recommended(DiffusionObjective::RfDenoiser).unwrap(),
        SamplerKind::Pingpong
    );
    assert_eq!(
        SamplerKind::recommended(DiffusionObjective::RectifiedFlow).unwrap(),
        SamplerKind::Euler
    );
    assert!(SamplerKind::recommended(DiffusionObjective::V).is_err());
    assert!(build_schedule(0, 1.0, &DistributionShift::Identity, None, 16).is_err());
    let device = Device::Cpu;
    let initial = Tensor::zeros((1, 1, 1), DType::F32, &device).unwrap();
    for invalid in [
        Schedule::Shared(vec![1.0, 0.0, 0.0]),
        Schedule::Shared(vec![1.0, 0.5, 0.5, 0.0]),
    ] {
        assert!(sample(
            SamplerKind::Euler,
            &initial,
            &invalid,
            None,
            &mut no_noise(),
            false,
            |x, _| Tensor::zeros_like(x).map_err(Into::into),
        )
        .is_err());
    }
}

#[test]
fn call_draw_and_transfer_costs_cover_short_long_and_max_geometry() {
    let lengths = [
        68usize,
        388,
        5_292_032 / 4096,
        5_324_800 / 4096,
        16_777_216 / 4096,
    ];
    for latent_length in lengths {
        let elements = 256 * latent_length;
        assert_eq!(
            resource_estimate(SamplerKind::Euler, 8, 1, elements)
                .unwrap()
                .model_calls,
            8
        );
        assert_eq!(
            resource_estimate(SamplerKind::Dpmpp, 8, 1, elements)
                .unwrap()
                .model_calls,
            8
        );
        assert_eq!(
            resource_estimate(SamplerKind::Rk4, 8, 1, elements)
                .unwrap()
                .model_calls,
            32
        );
        let pingpong = resource_estimate(SamplerKind::Pingpong, 8, 1, elements).unwrap();
        assert_eq!(pingpong.model_calls, 8);
        assert_eq!(pingpong.full_latent_noise_draws, 8);
        assert_eq!(
            pingpong.seeded_noise_device_elements,
            elements.saturating_mul(8)
        );
        assert_eq!(pingpong.schedule_device_elements, 9);
        assert_eq!(
            pingpong.total_host_to_device_elements,
            elements.saturating_mul(8) + 9
        );
    }
    assert!(resource_estimate(SamplerKind::Euler, 0, 1, 1).is_err());
}

struct RealCase {
    env: &'static str,
    artifact: &'static str,
    max_samples: usize,
    objective: DiffusionObjective,
}

const REAL_CASES: &[RealCase] = &[
    RealCase {
        env: "SA3_SMALL_MUSIC_SNAPSHOT",
        artifact: "small-music-reference.safetensors",
        max_samples: 5_292_032,
        objective: DiffusionObjective::RfDenoiser,
    },
    RealCase {
        env: "SA3_SMALL_SFX_SNAPSHOT",
        artifact: "small-sfx-reference.safetensors",
        max_samples: 5_292_032,
        objective: DiffusionObjective::RfDenoiser,
    },
    RealCase {
        env: "SA3_SMALL_MUSIC_BASE_SNAPSHOT",
        artifact: "small-music-base-reference.safetensors",
        max_samples: 5_324_800,
        objective: DiffusionObjective::RectifiedFlow,
    },
    RealCase {
        env: "SA3_SMALL_SFX_BASE_SNAPSHOT",
        artifact: "small-sfx-base-reference.safetensors",
        max_samples: 5_324_800,
        objective: DiffusionObjective::RectifiedFlow,
    },
    RealCase {
        env: "SA3_MEDIUM_SNAPSHOT",
        artifact: "medium-reference.safetensors",
        max_samples: 16_777_216,
        objective: DiffusionObjective::RfDenoiser,
    },
    RealCase {
        env: "SA3_MEDIUM_BASE_SNAPSHOT",
        artifact: "medium-base-reference.safetensors",
        max_samples: 16_777_216,
        objective: DiffusionObjective::RectifiedFlow,
    },
];

fn snapshot(name: &str) -> PathBuf {
    std::env::var_os(name)
        .map(PathBuf::from)
        .unwrap_or_else(|| panic!("{name} must point to its pinned immutable snapshot"))
}

fn cosine(actual: &Tensor, expected: &Tensor) -> f64 {
    let actual = values(actual);
    let expected = values(expected);
    let (mut dot, mut aa, mut bb) = (0.0f64, 0.0f64, 0.0f64);
    for (&a, &b) in actual.iter().zip(&expected) {
        dot += a as f64 * b as f64;
        aa += (a as f64).powi(2);
        bb += (b as f64).powi(2);
    }
    dot / (aa.sqrt() * bb.sqrt()).max(f64::MIN_POSITIVE)
}

fn metrics(actual: &Tensor, expected: &Tensor) -> (f64, f32) {
    let actual_values = values(actual);
    let expected_values = values(expected);
    let max_abs = actual_values
        .iter()
        .zip(&expected_values)
        .map(|(&actual, &expected)| (actual - expected).abs())
        .fold(0.0f32, f32::max);
    (cosine(actual, expected), max_abs)
}

/// All six P0 artifacts forced Pingpong, including the base models whose truthful default is
/// Euler. Explicit oracle draws are reconstructed from adjacent frozen states; runtime RNG is not
/// involved. The final eager draw is an injected zero because its coefficient is exactly zero.
#[test]
#[ignore = "requires all six pinned immutable snapshots; set SA3_*_SNAPSHOT"]
fn all_six_real_p0_pingpong_trajectories_match_stepwise() {
    let device = test_device();
    for case in REAL_CASES {
        let layout = SnapshotLayout::from_dir(&snapshot(case.env)).unwrap();
        assert_eq!(layout.config.sample_size, case.max_samples, "{}", case.env);
        let model_config = match &layout.config.model {
            candle_audio_stable_audio_3::config::ModelConfig::Diffusion(model) => model,
            _ => panic!("{} must be a diffusion snapshot", case.env),
        };
        assert_eq!(
            model_config.diffusion.diffusion_objective, case.objective,
            "{}",
            case.env
        );
        assert_eq!(
            SamplerKind::recommended(case.objective).unwrap(),
            if case.objective == DiffusionObjective::RfDenoiser {
                SamplerKind::Pingpong
            } else {
                SamplerKind::Euler
            }
        );
        let model = StableAudio3Dit::from_layout(&layout, &device).unwrap();
        let artifact = Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("../../../docs/migration/sa3-reference")
            .join(case.artifact);
        let oracle = unsafe {
            VarBuilder::from_mmaped_safetensors(&[artifact], DType::F32, &device).unwrap()
        };
        let get = |name: &str| oracle.get_unchecked(name).unwrap();
        let initial = get("sampler_initial_noise");
        let prompt = get("t5_projected_padded");
        let seconds = Tensor::from_vec(vec![0.25f32], 1, &device).unwrap();
        let local = Tensor::zeros((1, 257, 16), DType::F32, &device).unwrap();
        let mut sigmas = (0..8)
            .map(|index| values(&get(&format!("step_{index:02}_sigma")))[0])
            .collect::<Vec<_>>();
        sigmas.push(0.0);
        let schedule = Schedule::Shared(sigmas.clone());
        let mut draws = Vec::with_capacity(8);
        for index in 0..7 {
            let next = sigmas[index + 1];
            let next_x = get(&format!("step_{:02}_x", index + 1));
            let clean = get(&format!("step_{index:02}_denoised"));
            draws.push(
                ((&next_x - (&clean * (1.0 - next) as f64).unwrap()).unwrap() / next as f64)
                    .unwrap(),
            );
        }
        draws.push(Tensor::zeros_like(&initial).unwrap());
        let mut injected = InjectedNoise::new(draws);
        let output = sample_dit(
            &model,
            SamplerKind::Pingpong,
            &initial,
            &schedule,
            &prompt,
            None,
            None,
            &seconds,
            &local,
            None,
            Guidance::default(),
            &mut injected,
            true,
        )
        .unwrap();
        assert_eq!(output.model_calls, 8, "{}", case.env);
        assert_eq!(output.noise_draws, 8, "{}", case.env);
        let mut min_cosine = 1.0f64;
        let mut max_abs = 0.0f32;
        let mut step_max_abs = Vec::with_capacity(8);
        for (index, step) in output.trajectory.iter().enumerate() {
            assert_close(&values(&step.timestep), &[sigmas[index]], 0.0);
            let expected_x = get(&format!("step_{index:02}_x"));
            let expected_clean = get(&format!("step_{index:02}_denoised"));
            let (x_cosine, x_abs) = metrics(&step.x, &expected_x);
            let (clean_cosine, clean_abs) = metrics(&step.denoised, &expected_clean);
            min_cosine = min_cosine.min(x_cosine).min(clean_cosine);
            max_abs = max_abs.max(x_abs).max(clean_abs);
            step_max_abs.push(x_abs.max(clean_abs));
            assert!(x_cosine >= 0.999, "{} step {index} x", case.env);
            assert!(clean_cosine >= 0.999, "{} step {index} denoised", case.env);
        }
        let (final_cosine, final_abs) = metrics(&output.latents, &get("sampler_final"));
        min_cosine = min_cosine.min(final_cosine);
        max_abs = max_abs.max(final_abs);
        assert!(final_cosine >= 0.999, "{} final", case.env);
        eprintln!(
            "{}: min_cosine={min_cosine:.9} max_abs={max_abs:.9} \
             step_max_abs={step_max_abs:?} final_abs={final_abs:.9}",
            case.env,
        );
    }
}

fn max_abs_diff(left: &Tensor, right: &Tensor) -> f32 {
    values(&left.broadcast_sub(right).unwrap().abs().unwrap())
        .into_iter()
        .fold(0.0, f32::max)
}

#[test]
#[ignore = "requires pinned small-music post/base snapshots and real 2B CFG forwards"]
fn real_sampler_cfg_apg_scale_phi_matches_frozen_upstream() {
    let device = test_device();
    let guidance_artifact = Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("../../../docs/migration/sa3-sampler-reference/guidance.safetensors");
    let guidance_oracle = unsafe {
        VarBuilder::from_mmaped_safetensors(&[guidance_artifact], DType::F32, &device).unwrap()
    };
    for (case, oracle_prefix) in [
        (&REAL_CASES[0], "small-music"),
        (&REAL_CASES[2], "small-music-base"),
    ] {
        let layout = SnapshotLayout::from_dir(&snapshot(case.env)).unwrap();
        let model = StableAudio3Dit::from_layout(&layout, &device).unwrap();
        let p0_artifact = Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("../../../docs/migration/sa3-reference")
            .join(case.artifact);
        let oracle = unsafe {
            VarBuilder::from_mmaped_safetensors(&[p0_artifact], DType::F32, &device).unwrap()
        };
        let initial = oracle.get_unchecked("sampler_initial_noise").unwrap();
        let prompt = oracle.get_unchecked("t5_projected_padded").unwrap();
        let seconds = Tensor::from_vec(vec![0.25f32], 1, &device).unwrap();
        let local = Tensor::zeros((1, 257, 16), DType::F32, &device).unwrap();
        let mask = padding_mask(&[12], 16, &device).unwrap();
        let schedule = Schedule::Shared(vec![0.5, 0.0]);
        let variants = [
            (
                "vanilla",
                Guidance {
                    cfg_scale: 2.5,
                    apg_scale: 0.0,
                    ..Guidance::default()
                },
            ),
            (
                "apg",
                Guidance {
                    cfg_scale: 2.5,
                    apg_scale: 1.0,
                    ..Guidance::default()
                },
            ),
            (
                "blended_rescaled",
                Guidance {
                    cfg_scale: 2.5,
                    apg_scale: 0.5,
                    cfg_norm_threshold: 0.25,
                    scale_phi: 0.3,
                },
            ),
        ];
        let mut outputs = Vec::new();
        for (variant, guidance) in variants {
            let mut unused_noise = no_noise();
            let output = sample_dit(
                &model,
                SamplerKind::Euler,
                &initial,
                &schedule,
                &prompt,
                None,
                None,
                &seconds,
                &local,
                Some(&mask),
                guidance,
                &mut unused_noise,
                false,
            )
            .unwrap();
            assert_eq!(output.model_calls, 1);
            assert_eq!(output.noise_draws, 0);
            assert!(values(&output.latents).into_iter().all(f32::is_finite));
            let expected = guidance_oracle
                .get_unchecked(&format!("{oracle_prefix}.{variant}.final"))
                .unwrap();
            let (cosine, max_abs) = metrics(&output.latents, &expected);
            eprintln!("{oracle_prefix}.{variant}: cosine={cosine:.9} max_abs={max_abs:.9}");
            assert!(cosine >= 0.999);
            outputs.push(output.latents);
        }
        assert!(max_abs_diff(&outputs[0], &outputs[1]) > 1e-5);
        assert!(max_abs_diff(&outputs[1], &outputs[2]) > 1e-5);
    }
}

/// One-case resource probe selected by explicit immutable snapshot/P0 paths. Run it in a fresh
/// process for each duration so `/usr/bin/time -l` reports a meaningful per-case peak RSS.
#[test]
#[ignore = "set SA3_RESOURCE_SNAPSHOT, SA3_RESOURCE_P0, and SA3_RESOURCE_SECONDS"]
fn real_default_sampler_resource_probe() {
    let snapshot_path = std::env::var_os("SA3_RESOURCE_SNAPSHOT")
        .map(PathBuf::from)
        .expect("SA3_RESOURCE_SNAPSHOT is required");
    let p0_path = std::env::var_os("SA3_RESOURCE_P0")
        .map(PathBuf::from)
        .expect("SA3_RESOURCE_P0 is required");
    let seconds_value: f64 = std::env::var("SA3_RESOURCE_SECONDS")
        .expect("SA3_RESOURCE_SECONDS is required")
        .parse()
        .expect("SA3_RESOURCE_SECONDS must be numeric");
    let device = test_device();
    let layout = SnapshotLayout::from_dir(&snapshot_path).unwrap();
    let model_config = match &layout.config.model {
        candle_audio_stable_audio_3::config::ModelConfig::Diffusion(model) => model,
        _ => panic!("resource snapshot must be a diffusion model"),
    };
    let kind = SamplerKind::recommended(model_config.diffusion.diffusion_objective).unwrap();
    let geometry = candle_audio_stable_audio_3::sampler::default_sample_geometry(
        &layout.config,
        &[Some(seconds_value)],
    )
    .unwrap();
    let effective = effective_schedule_lengths(&[Some(seconds_value)]).unwrap();
    let schedule = build_schedule(
        8,
        1.0,
        &candle_audio_stable_audio_3::sampler::inference_shift(&model_config.diffusion),
        effective.as_deref(),
        geometry.latent_length,
    )
    .unwrap();
    let model = StableAudio3Dit::from_layout(&layout, &device).unwrap();
    let p0 =
        unsafe { VarBuilder::from_mmaped_safetensors(&[p0_path], DType::F32, &device).unwrap() };
    let prompt = p0.get_unchecked("t5_projected_padded").unwrap();
    let initial = Tensor::zeros((1, 256, geometry.latent_length), DType::F32, &device).unwrap();
    let local = Tensor::zeros((1, 257, geometry.latent_length), DType::F32, &device).unwrap();
    let seconds = Tensor::from_vec(vec![seconds_value as f32], 1, &device).unwrap();
    let mask = padding_mask(&geometry.valid_lengths, geometry.latent_length, &device).unwrap();
    let mut noise = SeededNoise::new(14542);
    let started = Instant::now();
    let output = sample_dit(
        &model,
        kind,
        &initial,
        &schedule,
        &prompt,
        None,
        None,
        &seconds,
        &local,
        Some(&mask),
        Guidance::default(),
        &mut noise,
        false,
    )
    .unwrap();
    let checksum = output
        .latents
        .to_dtype(DType::F32)
        .unwrap()
        .sum_all()
        .unwrap()
        .to_scalar::<f32>()
        .unwrap();
    // `to_scalar` is the synchronization boundary for queued Metal/CUDA work.
    let elapsed = started.elapsed().as_secs_f64();
    let estimate = resource_estimate(kind, 8, 1, 256 * geometry.latent_length).unwrap();
    eprintln!(
        "resource kind={kind:?} seconds={seconds_value} samples={} latent={} valid={} \
         calls={} draws={} schedule_device_elements={} host_to_device_elements={} host_to_device_bytes={} \
         elapsed={elapsed:.6} checksum={checksum:.9}",
        geometry.sample_size,
        geometry.latent_length,
        geometry.valid_lengths[0],
        output.model_calls,
        output.noise_draws,
        estimate.schedule_device_elements,
        estimate.total_host_to_device_elements,
        estimate.total_host_to_device_elements * std::mem::size_of::<f32>(),
    );
}
