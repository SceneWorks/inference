//! Weights-free checks of the Candle sampling loop ([`integrate`]) on the tiny fixture DiT:
//! upstream's Euler/CFG arithmetic step by step, CFG off, per-image seed independence of the
//! batch, progress and cancellation — on the CPU and, under `--features cuda`, on `cuda:0` (where
//! the CUDA trajectory is additionally held to the CPU one).

use candle_gen::candle_core::{Device, Tensor};
use candle_gen::gen_core::supra2::{self as s2, image_seed, initial_noise};
use candle_gen::gen_core::{CancelFlag, Progress};
use candle_gen_supra2::{integrate, SupraDit, TextConditioning};

use crate::support::{devices, host, report, tiny, tiny_fixture, tol};

/// The tiny DiT plus a conditional (fixture sample 0) and unconditional (sample 1, padded) context.
fn setup(device: &Device) -> (SupraDit, TextConditioning, TextConditioning) {
    let (weights, io) = tiny_fixture(device);
    let cfg = tiny();
    let dit = SupraDit::from_tensors(&weights, &cfg).unwrap();
    let mask = host(&io["in.mask"]);
    let ctx = &io["in.ctx"];
    let cond = TextConditioning {
        context: ctx.narrow(0, 0, 1).unwrap(),
        mask: mask[..cfg.ctx_len].to_vec(),
    };
    let uncond = TextConditioning {
        context: ctx.narrow(0, 1, 1).unwrap(),
        mask: mask[cfg.ctx_len..].to_vec(),
    };
    (dit, cond, uncond)
}

fn noise(images: u32, seed: u64) -> Vec<f32> {
    let per = tiny().latent_numel();
    (0..images)
        .flat_map(|i| initial_noise(image_seed(seed, i), per))
        .collect()
}

/// Upstream's loop restated on the host from single (unbatched) DiT evaluations:
/// `z += dt · (v_u + g (v_c − v_u))`, or `z += dt · v_c` without CFG.
fn host_reference(
    dit: &SupraDit,
    cond: &TextConditioning,
    uncond: Option<&TextConditioning>,
    z0: &[f32],
    steps: u32,
    guidance: f32,
) -> Vec<f32> {
    let cfg = *dit.config();
    let n = z0.len() / cfg.latent_numel();
    let s = cfg.latent_size;
    let eval = |c: &TextConditioning, z: &[f32], t: f32| -> Vec<f32> {
        let zt = Tensor::from_vec(z.to_vec(), (n, cfg.latent_channels, s, s), dit.device()).unwrap();
        let ctx = c.context.repeat((n, 1, 1)).unwrap();
        let mask = dit.additive_mask(&c.mask.repeat(n), n, c.mask.len()).unwrap();
        let tf = dit.timestep_features(&vec![t; n]).unwrap();
        host(&dit.forward(&zt, &tf, &ctx, &mask).unwrap())
    };
    let dt = s2::euler_dt(steps);
    let mut z = z0.to_vec();
    for t in s2::euler_timesteps(steps) {
        let vc = eval(cond, &z, t);
        let v: Vec<f32> = match uncond {
            Some(u) => {
                let vu = eval(u, &z, t);
                vc.iter()
                    .zip(&vu)
                    .map(|(&c, &u)| s2::cfg_mix(c, u, guidance))
                    .collect()
            }
            None => vc,
        };
        for (zi, vi) in z.iter_mut().zip(&v) {
            *zi += dt * vi;
        }
    }
    z
}

#[test]
fn cfg_and_plain_trajectories_follow_upstreams_euler_loop() {
    let mut cpu_latents: Option<Vec<f32>> = None;
    for (name, device) in devices() {
        let (dit, cond, uncond) = setup(&device);
        let z0 = noise(2, 7);
        let cancel = CancelFlag::new();
        let mut steps_seen = Vec::new();
        let cfg_run = integrate(
            &dit,
            &cond,
            Some(&uncond),
            &z0,
            4,
            3.0,
            &cancel,
            &mut |p| {
                if let Progress::Step { current, total } = p {
                    steps_seen.push((current, total));
                }
            },
        )
        .unwrap();
        assert_eq!(steps_seen, [(1, 4), (2, 4), (3, 4), (4, 4)]);
        let got = host(&cfg_run);
        // Batching cond+uncond (and the two images) into one forward only reorders reductions.
        let bound = tol(&device, (1e-5, 1e-6), (1e-4, 1e-5));
        report(
            &format!("sampler.cfg3.{name}"),
            &got,
            &host_reference(&dit, &cond, Some(&uncond), &z0, 4, 3.0),
            bound,
        );
        let plain = integrate(&dit, &cond, None, &z0, 3, 1.0, &cancel, &mut |_| {}).unwrap();
        report(
            &format!("sampler.plain.{name}"),
            &host(&plain),
            &host_reference(&dit, &cond, None, &z0, 3, 1.0),
            bound,
        );
        match &cpu_latents {
            None => cpu_latents = Some(got),
            // Cross-device: the accelerator integrates the same z0 to the CPU's latents.
            Some(cpu) => report(
                &format!("sampler.cfg3.{name}_vs_cpu"),
                &got,
                cpu,
                (1e-4, 1e-5),
            ),
        }
    }
}

/// Image `i` of a request is the single-image request seeded `seed + i`: batching never couples
/// images, so each output asset is reproducible on its own.
#[test]
fn each_image_is_reproducible_from_its_own_seed() {
    for (name, device) in devices() {
        let (dit, cond, uncond) = setup(&device);
        let cancel = CancelFlag::new();
        let pair = host(
            &integrate(
                &dit,
                &cond,
                Some(&uncond),
                &noise(2, 40),
                3,
                2.5,
                &cancel,
                &mut |_| {},
            )
            .unwrap(),
        );
        let second = host(
            &integrate(
                &dit,
                &cond,
                Some(&uncond),
                &noise(1, 41),
                3,
                2.5,
                &cancel,
                &mut |_| {},
            )
            .unwrap(),
        );
        let per = tiny().latent_numel();
        report(
            &format!("sampler.seed_split.{name}"),
            &pair[per..],
            &second,
            tol(&device, (1e-5, 1e-6), (1e-4, 1e-5)),
        );
    }
}

#[test]
fn cancellation_stops_before_the_next_step() {
    let (dit, cond, uncond) = setup(&Device::Cpu);
    let cancel = CancelFlag::new();
    let mut seen = 0;
    let err = integrate(
        &dit,
        &cond,
        Some(&uncond),
        &noise(1, 0),
        10,
        3.0,
        &cancel,
        &mut |p| {
            if matches!(p, Progress::Step { .. }) {
                seen += 1;
                if seen == 2 {
                    cancel.cancel();
                }
            }
        },
    )
    .err()
    .expect("a cancelled run must fail");
    assert!(matches!(err, candle_gen::CandleError::Canceled), "{err}");
    assert_eq!(seen, 2);
}

#[test]
fn malformed_noise_is_refused() {
    let (dit, cond, _) = setup(&Device::Cpu);
    let cancel = CancelFlag::new();
    for bad in [vec![], vec![0.0f32; tiny().latent_numel() + 1]] {
        assert!(integrate(&dit, &cond, None, &bad, 2, 1.0, &cancel, &mut |_| {}).is_err());
    }
}
