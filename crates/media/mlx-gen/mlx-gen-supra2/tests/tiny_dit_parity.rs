//! Weights-free parity against the frozen upstream modules on tiny random-weight geometries
//! (`tests/fixtures/tiny_{dit,t5}_parity.safetensors`, dumped by `tools/dump_supra2_reference.py
//! tiny`):
//!
//! * one `SupraDiT` forward — two samples at t = 0 and t = 0.62, one with a padded text mask
//!   (patchify/unpatchify, host timestep features, AdaLN-Zero, masked unmodulated cross-attention,
//!   final layer);
//! * the provider's own [`sampling::denoise`] loop over 3 Euler steps, CFG on (stored uncond,
//!   g = 3) and off (g = 1), against upstream's verbatim `generate()` loop body;
//! * the 12-head-style non-XXL T5 geometry path (2 blocks × 4 heads) with a padded mask.
//!
//! All run on the MLX **CPU** device: Metal's f32 GEMM accumulates in reduced precision (measured
//! 3e-3 relative on the tiny DiT), which would hide a small structural error. Declared bounds are
//! relative to the reference abs-max: max ≤ 1e-5, mean ≤ 1e-6 (T5/DiT forward) and max ≤ 5e-5,
//! mean ≤ 5e-6 (3-step trajectories).

use std::collections::HashMap;
use std::path::Path;

use gen_core_checkpoint::supra2::Supra2Config;
use mlx_gen::weights::Weights;
use mlx_gen::CancelFlag;
use mlx_gen_flux::text_encoder::{T5Geometry, T5TextEncoder};
use mlx_gen_supra2::sampling::{self, ContextBatch};
use mlx_gen_supra2::SupraDit;
use mlx_rs::Array;

use crate::common::{diff_stats, host};

fn tiny() -> Supra2Config {
    Supra2Config {
        latent_channels: 4,
        latent_size: 16,
        patch: 2,
        d_model: 32,
        depth: 2,
        num_heads: 2,
        mlp_hidden: 128,
        ctx_dim: 24,
        ctx_len: 8,
        freq_dim: 256,
    }
}

/// Selects the MLX CPU device for one test and restores the GPU default afterwards.
struct CpuDevice;
impl CpuDevice {
    fn select() -> Self {
        mlx_rs::Device::set_default(&mlx_rs::Device::cpu());
        CpuDevice
    }
}
impl Drop for CpuDevice {
    fn drop(&mut self) {
        mlx_rs::Device::set_default(&mlx_rs::Device::gpu());
    }
}

fn fixture(name: &str) -> Weights {
    Weights::from_file(
        Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("tests/fixtures")
            .join(name),
    )
    .unwrap()
}

fn prefixed(all: &Weights, prefix: &str) -> HashMap<String, Array> {
    all.keys()
        .filter_map(|key| {
            key.strip_prefix(prefix)
                .map(|name| (name.to_string(), all.require(key).unwrap().clone()))
        })
        .collect()
}

fn assert_close(stage: &str, got: &Array, want: &Array, max_rel: f32, mean_rel: f32) {
    let (max_abs, mean_abs, ref_max) = diff_stats(&host(got), &host(want));
    eprintln!("{stage}: max_abs {max_abs:.3e} mean_abs {mean_abs:.3e} ref_absmax {ref_max:.3}");
    assert!(
        max_abs <= max_rel * ref_max,
        "{stage}: max_abs {max_abs} vs ref {ref_max}"
    );
    assert!(
        mean_abs <= mean_rel * ref_max,
        "{stage}: mean_abs {mean_abs} vs ref {ref_max}"
    );
}

#[test]
fn tiny_dit_matches_upstream_module() {
    let _cpu = CpuDevice::select();
    let all = fixture("tiny_dit_parity.safetensors");
    let cfg = tiny();
    let dit = SupraDit::from_weights(&Weights::from_map(prefixed(&all, "w.")), &cfg).unwrap();

    let t = host(all.require("in.t").unwrap());
    let features = dit.timestep_features(&t);
    let (max_abs, _, _) = diff_stats(&host(&features), &host(all.require("out.t_freq").unwrap()));
    assert!(max_abs < 1e-4, "timestep features max_abs {max_abs}");

    let mask01 = host(all.require("in.mask").unwrap());
    let mask = SupraDit::additive_mask(&mask01, 2, cfg.ctx_len).unwrap();
    let v = dit
        .forward(
            all.require("in.z").unwrap(),
            &features,
            all.require("in.ctx").unwrap(),
            &mask,
        )
        .unwrap();
    assert_eq!(v.shape(), &[2, 4, 16, 16]);
    // Measured 4.1e-5 / 4.3e-6 absolute at ref_absmax 9.05.
    assert_close("tiny DiT", &v, all.require("out.v").unwrap(), 1e-5, 1e-6);
}

/// The provider's sampling loop (`sampling::denoise`, the code `Supra2Image::generate_with` runs)
/// against upstream's loop: CFG on with the stored unconditional context, and CFG off.
#[test]
fn tiny_denoise_loop_matches_upstream_trajectories() {
    let _cpu = CpuDevice::select();
    let all = fixture("tiny_dit_parity.safetensors");
    let cfg = tiny();
    let dit = SupraDit::from_weights(&Weights::from_map(prefixed(&all, "w.")), &cfg).unwrap();
    let mask =
        SupraDit::additive_mask(&host(all.require("in.mask").unwrap()), 2, cfg.ctx_len).unwrap();
    let cond = ContextBatch {
        context: all.require("in.ctx").unwrap(),
        mask: &mask,
    };
    let uncond_ctx = all
        .require("traj.uncond_ctx")
        .unwrap()
        .expand_dims(0)
        .unwrap();
    let (uctx, umask) = sampling::broadcast(
        &uncond_ctx,
        &host(all.require("traj.uncond_mask").unwrap()),
        2,
    )
    .unwrap();
    let z0 = all.require("traj.z0").unwrap().clone();
    let cancel = CancelFlag::new();

    let mut steps_seen = Vec::new();
    let on = sampling::denoise(
        &dit,
        z0.clone(),
        3,
        3.0,
        cond,
        Some(ContextBatch {
            context: &uctx,
            mask: &umask,
        }),
        &cancel,
        &mut |p| steps_seen.push(p),
    )
    .unwrap();
    assert_eq!(steps_seen.len(), 3);
    assert_close(
        "tiny CFG 3 trajectory",
        &on,
        all.require("traj.cfg3").unwrap(),
        5e-5,
        5e-6,
    );

    let off = sampling::denoise(&dit, z0, 3, 1.0, cond, None, &cancel, &mut |_| {}).unwrap();
    assert_close(
        "tiny CFG off trajectory",
        &off,
        all.require("traj.cfg1").unwrap(),
        5e-5,
        5e-6,
    );

    cancel.cancel();
    let cancelled = sampling::denoise(
        &dit,
        all.require("traj.z0").unwrap().clone(),
        3,
        1.0,
        cond,
        None,
        &cancel,
        &mut |_| {},
    );
    assert!(matches!(cancelled, Err(mlx_gen::Error::Canceled)));
}

/// The non-XXL T5 geometry path Flan-T5-Base takes (`T5Geometry { blocks, heads }`) — here 2 blocks
/// × 4 heads — with a padded mask, vs HF `T5EncoderModel`.
#[test]
fn tiny_t5_geometry_matches_hf_encoder() {
    let _cpu = CpuDevice::select();
    let all = fixture("tiny_t5_parity.safetensors");
    let w = Weights::from_map(prefixed(&all, "w."));
    let t5 = T5TextEncoder::from_weights_with_geometry(
        &w,
        "",
        32,
        T5Geometry {
            blocks: 2,
            heads: 4,
        },
    )
    .unwrap();
    let ids: Vec<i32> = host(all.require("in.ids").unwrap())
        .iter()
        .map(|&v| v as i32)
        .collect();
    let additive: Vec<f32> = host(all.require("in.mask").unwrap())
        .iter()
        .map(|&m| if m != 0.0 { 0.0 } else { f32::NEG_INFINITY })
        .collect();
    let len = ids.len() as i32;
    let hidden = t5
        .forward_masked(
            &Array::from_slice(&ids, &[1, len]),
            Some(&Array::from_slice(&additive, &[1, 1, 1, len])),
        )
        .unwrap();
    assert_close(
        "tiny T5",
        &hidden,
        all.require("out.hidden").unwrap(),
        1e-5,
        1e-6,
    );
}

#[test]
fn tiny_dit_rejects_extra_or_misshapen_tensors() {
    let all = fixture("tiny_dit_parity.safetensors");
    let weights = prefixed(&all, "w.");
    let mut extra = weights.clone();
    extra.insert(
        "blocks.9.mlp.0.weight".into(),
        Array::from_slice(&[0f32], &[1]),
    );
    assert!(SupraDit::from_weights(&Weights::from_map(extra), &tiny()).is_err());
    let wider = Supra2Config {
        d_model: 64,
        ..tiny()
    };
    assert!(SupraDit::from_weights(&Weights::from_map(weights), &wider).is_err());
}
