//! Weights-free parity of [`SupraDit`] against the frozen upstream `SupraDiT` module, on a tiny
//! random-weight geometry (`tests/fixtures/tiny_dit_parity.safetensors`, dumped by
//! `tools/dump_supra2_reference.py tiny`): two samples at t = 0 and t = 0.62, one with a padded text
//! mask. Covers patchify/unpatchify, the host timestep features, AdaLN-Zero modulation, the masked
//! unmodulated cross-attention and the final layer.

use std::path::Path;

use mlx_gen::gen_core::supra2::Supra2Config;
use mlx_gen::weights::Weights;
use mlx_gen_supra2::SupraDit;
use mlx_rs::Array;

use crate::{diff_stats, host};

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

#[test]
fn tiny_dit_matches_upstream_module() {
    // Structural parity runs on the MLX CPU device: Metal's f32 GEMM accumulates in reduced
    // precision (measured 3e-3 relative here), which would hide a small structural error. The
    // real-weight tests cover the Metal path with their own declared tolerances.
    struct RestoreGpu;
    impl Drop for RestoreGpu {
        fn drop(&mut self) {
            mlx_rs::Device::set_default(&mlx_rs::Device::gpu());
        }
    }
    let _restore = RestoreGpu;
    mlx_rs::Device::set_default(&mlx_rs::Device::cpu());
    let path =
        Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/tiny_dit_parity.safetensors");
    let all = Weights::from_file(&path).unwrap();
    let mut weights = std::collections::HashMap::new();
    for key in all.keys() {
        if let Some(name) = key.strip_prefix("w.") {
            weights.insert(name.to_string(), all.require(key).unwrap().clone());
        }
    }
    let w = Weights::from_map(weights);
    let cfg = tiny();
    let dit = SupraDit::from_weights(&w, &cfg).unwrap();

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
    let (max_abs, mean_abs, ref_max) = diff_stats(&host(&v), &host(all.require("out.v").unwrap()));
    eprintln!("tiny DiT: max_abs {max_abs:.3e} mean_abs {mean_abs:.3e} ref_absmax {ref_max:.3}");
    // Declared tolerance (f32 CPU vs torch f32 CPU): max_abs ≤ 1e-5 · ref_absmax and
    // mean_abs ≤ 1e-6 · ref_absmax; measured 4.1e-5 / 4.3e-6 absolute at ref_absmax 9.05.
    assert!(
        max_abs <= 1e-5 * ref_max,
        "max_abs {max_abs} vs ref {ref_max}"
    );
    assert!(
        mean_abs <= 1e-6 * ref_max,
        "mean_abs {mean_abs} vs ref {ref_max}"
    );
}

#[test]
fn tiny_dit_rejects_extra_or_misshapen_tensors() {
    let path =
        Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/tiny_dit_parity.safetensors");
    let all = Weights::from_file(&path).unwrap();
    let mut weights = std::collections::HashMap::new();
    for key in all.keys() {
        if let Some(name) = key.strip_prefix("w.") {
            weights.insert(name.to_string(), all.require(key).unwrap().clone());
        }
    }
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
