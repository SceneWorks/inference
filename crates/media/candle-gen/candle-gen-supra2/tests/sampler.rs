//! The provider's sampling loop ([`sampling::denoise`], the code `Supra2Image::generate_with` runs)
//! on the MLX twin's tiny fixture: 3 Euler steps with CFG on (stored uncond, g = 3) and off (g = 1)
//! against upstream's verbatim `generate()` loop (`traj.*`), per-image seed independence of the
//! batch, progress and cancellation — on the CPU and, under `--features cuda`, on `cuda:0` (where
//! the trajectory is additionally held to the CPU one).

use candle_gen::candle_core::{Device, Tensor};
use candle_gen::gen_core::supra2::{image_seed, initial_noise};
use candle_gen::gen_core::{CancelFlag, Progress};
use candle_gen_supra2::sampling::{self, ContextBatch};
use candle_gen_supra2::SupraDit;

use crate::support::{devices, host, report, tiny, tiny_fixture, tol};

#[test]
fn tiny_denoise_loop_matches_upstream_trajectories() {
    let mut cpu: Option<(Vec<f32>, Vec<f32>)> = None;
    for (name, device) in devices() {
        let (weights, io) = tiny_fixture(&device);
        let cfg = tiny();
        let dit = SupraDit::from_tensors(&weights, &cfg).unwrap();
        let mask = SupraDit::additive_mask(&host(&io["in.mask"]), 2, cfg.ctx_len, &device).unwrap();
        let cond = ContextBatch {
            context: &io["in.ctx"],
            mask: &mask,
        };
        let (uctx, umask) = sampling::broadcast(
            &io["traj.uncond_ctx"].unsqueeze(0).unwrap(),
            &host(&io["traj.uncond_mask"]),
            2,
        )
        .unwrap();
        let uncond = ContextBatch {
            context: &uctx,
            mask: &umask,
        };
        let cancel = CancelFlag::new();
        let mut steps_seen = Vec::new();
        let on = sampling::denoise(
            &dit,
            io["traj.z0"].clone(),
            3,
            3.0,
            cond,
            Some(uncond),
            &cancel,
            &mut |p| {
                if let Progress::Step { current, total } = p {
                    steps_seen.push((current, total));
                }
            },
        )
        .unwrap();
        assert_eq!(steps_seen, [(1, 3), (2, 3), (3, 3)]);
        // Declared (max_rel, mean_rel): the MLX twin's CPU trajectory bound, 5e-5 / 5e-6; the
        // accelerator bound is 10x (reduction order only).
        let bound = tol(&device, (5e-5, 5e-6), (5e-4, 5e-5));
        let on = host(&on);
        report(
            &format!("tiny_traj.cfg3.{name}"),
            &on,
            &host(&io["traj.cfg3"]),
            bound,
        );
        let off = host(
            &sampling::denoise(
                &dit,
                io["traj.z0"].clone(),
                3,
                1.0,
                cond,
                None,
                &cancel,
                &mut |_| {},
            )
            .unwrap(),
        );
        report(
            &format!("tiny_traj.cfg1.{name}"),
            &off,
            &host(&io["traj.cfg1"]),
            bound,
        );
        match &cpu {
            None => cpu = Some((on, off)),
            // Cross-device: the accelerator integrates the same z0 to the CPU's latents.
            Some((cpu_on, cpu_off)) => {
                report(
                    &format!("tiny_traj.cfg3.{name}_vs_cpu"),
                    &on,
                    cpu_on,
                    (1e-4, 1e-5),
                );
                report(
                    &format!("tiny_traj.cfg1.{name}_vs_cpu"),
                    &off,
                    cpu_off,
                    (1e-4, 1e-5),
                );
            }
        }
    }
}

/// Image `i` of a request is the single-image request seeded `seed + i`: batching never couples
/// images, so each output asset is reproducible on its own.
#[test]
fn each_image_is_reproducible_from_its_own_seed() {
    for (name, device) in devices() {
        let (weights, io) = tiny_fixture(&device);
        let cfg = tiny();
        let dit = SupraDit::from_tensors(&weights, &cfg).unwrap();
        let ctx = io["in.ctx"].narrow(0, 0, 1).unwrap();
        let mask01 = host(&io["in.mask"])[..cfg.ctx_len].to_vec();
        let run = |seeds: &[u64]| -> Vec<f32> {
            let n = seeds.len();
            let (c, m) = sampling::broadcast(&ctx, &mask01, n).unwrap();
            let (u, um) = sampling::broadcast(
                &io["traj.uncond_ctx"].unsqueeze(0).unwrap(),
                &host(&io["traj.uncond_mask"]),
                n,
            )
            .unwrap();
            let per = cfg.latent_numel();
            let z0: Vec<f32> = seeds.iter().flat_map(|&s| initial_noise(s, per)).collect();
            let s = cfg.latent_size;
            let z0 = Tensor::from_vec(z0, (n, cfg.latent_channels, s, s), &device).unwrap();
            host(
                &sampling::denoise(
                    &dit,
                    z0,
                    3,
                    2.5,
                    ContextBatch {
                        context: &c,
                        mask: &m,
                    },
                    Some(ContextBatch {
                        context: &u,
                        mask: &um,
                    }),
                    &CancelFlag::new(),
                    &mut |_| {},
                )
                .unwrap(),
            )
        };
        let pair = run(&[image_seed(40, 0), image_seed(40, 1)]);
        let second = run(&[41]);
        report(
            &format!("sampler.seed_split.{name}"),
            &pair[cfg.latent_numel()..],
            &second,
            tol(&device, (1e-5, 1e-6), (1e-4, 1e-5)),
        );
    }
}

#[test]
fn cancellation_stops_before_the_next_step_and_zero_steps_is_refused() {
    let device = Device::Cpu;
    let (weights, io) = tiny_fixture(&device);
    let cfg = tiny();
    let dit = SupraDit::from_tensors(&weights, &cfg).unwrap();
    let mask = SupraDit::additive_mask(&host(&io["in.mask"]), 2, cfg.ctx_len, &device).unwrap();
    let cond = ContextBatch {
        context: &io["in.ctx"],
        mask: &mask,
    };
    let cancel = CancelFlag::new();
    let mut seen = 0;
    let err = sampling::denoise(
        &dit,
        io["traj.z0"].clone(),
        10,
        1.0,
        cond,
        None,
        &cancel,
        &mut |p| {
            if matches!(p, Progress::Step { .. }) {
                seen += 1;
                if seen == 2 {
                    cancel.cancel();
                }
            }
        },
    );
    assert!(
        matches!(err, Err(candle_gen::CandleError::Canceled)),
        "a cancelled run must fail with the typed cancellation"
    );
    assert_eq!(seen, 2);
    assert!(sampling::denoise(
        &dit,
        io["traj.z0"].clone(),
        0,
        1.0,
        cond,
        None,
        &CancelFlag::new(),
        &mut |_| {}
    )
    .is_err());
}
