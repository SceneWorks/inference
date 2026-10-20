//! Weights-free parity of [`SupraDit`] against the frozen upstream `SupraDiT` module on the MLX
//! twin's tiny random-weight fixture (two samples at t = 0 and t = 0.62, one with a padded text
//! mask): patchify/unpatchify, the host timestep features, AdaLN-Zero modulation, the masked
//! unmodulated cross-attention and the final layer. Runs on the CPU and, under `--features cuda`,
//! on `cuda:0` (the manual `windows-cuda` CI lane).

use candle_gen::candle_core::Tensor;
use candle_gen_supra2::SupraDit;

use crate::support::{devices, diff_stats, host, report, tiny, tiny_fixture, tol};

#[test]
fn tiny_dit_matches_upstream_module() {
    for (name, device) in devices() {
        let (weights, io) = tiny_fixture(&device);
        let cfg = tiny();
        let dit = SupraDit::from_tensors(&weights, &cfg).unwrap();

        let t = host(&io["in.t"]);
        let features = dit.timestep_features(&t).unwrap();
        let (max_abs, _, _) = diff_stats(&host(&features), &host(&io["out.t_freq"]));
        assert!(max_abs < 1e-4, "{name}: timestep features max_abs {max_abs}");

        let mask = dit
            .additive_mask(&host(&io["in.mask"]), 2, cfg.ctx_len)
            .unwrap();
        let v = dit
            .forward(&io["in.z"], &features, &io["in.ctx"], &mask)
            .unwrap();
        assert_eq!(v.dims(), [2, 4, 16, 16]);
        // Declared (max_rel, mean_rel): CPU f32 vs torch f32 CPU is the MLX twin's structural
        // gate, 1e-5 / 1e-6. The accelerator bound is 10x that (reduction order only).
        report(
            &format!("tiny_dit.{name}"),
            &host(&v),
            &host(&io["out.v"]),
            tol(&device, (1e-5, 1e-6), (1e-4, 1e-5)),
        );
    }
}

/// Padded context rows are inert: the cross-attention masks their keys with `-inf`, so whatever
/// those rows hold (upstream's encoder output, Candle's zeros, or garbage) cannot move the output.
/// This is what lets the Candle text path encode only the real tokens (see `encode_tokens`).
#[test]
fn masked_context_rows_cannot_move_the_velocity() {
    for (name, device) in devices() {
        let (weights, io) = tiny_fixture(&device);
        let cfg = tiny();
        let dit = SupraDit::from_tensors(&weights, &cfg).unwrap();
        let mask01 = host(&io["in.mask"]);
        assert!(
            mask01.contains(&0.0),
            "the fixture must carry a padded sample"
        );
        let mask = dit.additive_mask(&mask01, 2, cfg.ctx_len).unwrap();
        let features = dit.timestep_features(&host(&io["in.t"])).unwrap();
        let base = host(
            &dit.forward(&io["in.z"], &features, &io["in.ctx"], &mask)
                .unwrap(),
        );

        // Overwrite every masked row with large values.
        let ctx = host(&io["in.ctx"]);
        let garbage: Vec<f32> = ctx
            .iter()
            .enumerate()
            .map(|(i, &v)| {
                if mask01[i / cfg.ctx_dim] == 0.0 {
                    1e3 * ((i % 7) as f32 - 3.0)
                } else {
                    v
                }
            })
            .collect();
        let garbage = Tensor::from_vec(garbage, (2, cfg.ctx_len, cfg.ctx_dim), &device).unwrap();
        let moved = host(&dit.forward(&io["in.z"], &features, &garbage, &mask).unwrap());
        let (max_abs, _, ref_max) = diff_stats(&moved, &base);
        eprintln!("{name}: masked-row perturbation moved the velocity by {max_abs:.3e}");
        if device.is_cpu() {
            assert_eq!(max_abs, 0.0, "{name}: masked rows leaked into the output");
        } else {
            assert!(max_abs <= 1e-6 * ref_max, "{name}: {max_abs}");
        }
    }
}

#[test]
fn tiny_dit_rejects_extra_or_misshapen_tensors() {
    let device = candle_gen::candle_core::Device::Cpu;
    let (weights, _) = tiny_fixture(&device);
    let mut extra = weights.clone();
    extra.insert(
        "blocks.9.mlp.0.weight".into(),
        Tensor::zeros(1, candle_gen::candle_core::DType::F32, &device).unwrap(),
    );
    assert!(SupraDit::from_tensors(&extra, &tiny()).is_err());
    let mut missing = weights.clone();
    missing.remove("final.linear.bias");
    assert!(SupraDit::from_tensors(&missing, &tiny()).is_err());
    let wider = candle_gen::gen_core::supra2::Supra2Config {
        d_model: 64,
        ..tiny()
    };
    assert!(SupraDit::from_tensors(&weights, &wider).is_err());
    let mut half = weights.clone();
    let w = half["x_embed.weight"]
        .to_dtype(candle_gen::candle_core::DType::F16)
        .unwrap();
    half.insert("x_embed.weight".into(), w);
    assert!(SupraDit::from_tensors(&half, &tiny()).is_err());
}
