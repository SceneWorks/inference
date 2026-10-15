//! Flow-map DiT forward vs upstream `DiT.forward` on the miniature checkpoint (per-mode early
//! blocks, the shared single stream over `[image, text]`, 4-axis RoPE, q/k RMSNorm, the parallel
//! attention + SwiGLU block, target-time conditioning, `LastLayer`), single-sample and
//! two-sample packs.
//!
//! Tolerances:
//! * FP32 (`Precision::Fp32` compute, MLX CPU stream) vs the oracle's FP32 CPU forward over the
//!   same bf16-rounded parameters and inputs: only summation order differs — 1e-4 of peak.
//! * bf16 compute (the release's autocast policy) vs the oracle's bf16-autocast CPU forward: bf16
//!   matmul / attention rounding on both sides — 5e-2 of peak.
//!
//! Every test runs in a CPU-default child (`cpu_process`).

use mlx_gen::gen_core::flux3::text_ids;
use mlx_gen_flux3::{load_dit, SampleTokens};
use mlx_rs::{Array, Dtype};

use crate::common::{
    assert_close, cpu_process, fixture, host_f32, on_cpu, tiny_config, tiny_package,
};

fn ids_of(a: &Array) -> Vec<[f32; 4]> {
    host_f32(a)
        .chunks(4)
        .map(|c| [c[0], c[1], c[2], c[3]])
        .collect()
}

fn scalar(a: &Array) -> f32 {
    host_f32(a)[0]
}

#[test]
fn single_sample_forward_matches_upstream_in_fp32() {
    cpu_process(
        "dit_parity::single_sample_forward_matches_upstream_in_fp32",
        single_sample_forward_matches_upstream_in_fp32_body,
    );
}

fn single_sample_forward_matches_upstream_in_fp32_body() {
    on_cpu(|| {
        let config = tiny_config();
        let dit = load_dit(
            &tiny_package().join("model.safetensors"),
            &config,
            Dtype::Float32,
        )
        .unwrap();
        assert!(dit.is_flow_map());
        let golden = fixture("flux3_dit_golden.safetensors");
        let text = fixture("flux3_text_golden.safetensors");
        let tokens = text.require("short/tokens").unwrap();
        let text_ids = text_ids(tokens.shape()[0] as usize);
        let image_ids = ids_of(golden.require("single/image_ids").unwrap());
        let out = dit
            .forward(&[SampleTokens {
                text: tokens,
                text_ids: &text_ids,
                image: golden.require("single/x").unwrap(),
                image_ids: &image_ids,
                image_cond: None,
                timestep: scalar(golden.require("single/t").unwrap()),
                target_timestep: Some(scalar(golden.require("single/t_ref").unwrap())),
            }])
            .unwrap();
        assert_eq!(out.dtype(), Dtype::Float32);
        assert_close(
            "velocity",
            &out,
            golden.require("single/velocity").unwrap(),
            1e-4,
        );
    });
}

/// Two samples with different prompts, grids and times, packed mode-major: each sample's image
/// tokens attend only their own sample (the per-sample bounds), and sample 0's velocity is the
/// single-sample velocity.
///
/// *Mutation that reds this:* attending over the whole packed stream instead of per sample, or
/// applying one sample's modulation rows to the other's tokens.
#[test]
fn two_sample_pack_matches_upstream_in_fp32() {
    cpu_process(
        "dit_parity::two_sample_pack_matches_upstream_in_fp32",
        two_sample_pack_matches_upstream_in_fp32_body,
    );
}

fn two_sample_pack_matches_upstream_in_fp32_body() {
    on_cpu(|| {
        let config = tiny_config();
        let dit = load_dit(
            &tiny_package().join("model.safetensors"),
            &config,
            Dtype::Float32,
        )
        .unwrap();
        let golden = fixture("flux3_dit_golden.safetensors");
        let text = fixture("flux3_text_golden.safetensors");
        let tokens0 = text.require("short/tokens").unwrap();
        let tokens1 = text.require("empty/tokens").unwrap();
        let text_ids0 = text_ids(tokens0.shape()[0] as usize);
        let text_ids1 = text_ids(tokens1.shape()[0] as usize);
        let image_ids0 = ids_of(golden.require("single/image_ids").unwrap());
        let image_ids1 = ids_of(golden.require("pair/image_ids2").unwrap());
        let out = dit
            .forward(&[
                SampleTokens {
                    text: tokens0,
                    text_ids: &text_ids0,
                    image: golden.require("single/x").unwrap(),
                    image_ids: &image_ids0,
                    image_cond: None,
                    timestep: scalar(golden.require("single/t").unwrap()),
                    target_timestep: Some(scalar(golden.require("single/t_ref").unwrap())),
                },
                SampleTokens {
                    text: tokens1,
                    text_ids: &text_ids1,
                    image: golden.require("pair/x2").unwrap(),
                    image_ids: &image_ids1,
                    image_cond: None,
                    timestep: scalar(golden.require("pair/t2").unwrap()),
                    target_timestep: Some(scalar(golden.require("pair/t2_ref").unwrap())),
                },
            ])
            .unwrap();
        let want = golden.require("pair/velocity").unwrap();
        assert_eq!(out.shape(), want.shape());
        assert_close("pair velocity", &out, want, 1e-4);
        let n0 = golden.require("single/x").unwrap().shape()[0];
        let first = out.split_axis(&[n0], 0).unwrap().swap_remove(0);
        assert_close(
            "pair sample 0 == single",
            &first,
            golden.require("single/velocity").unwrap(),
            1e-4,
        );
    });
}

#[test]
fn single_sample_forward_matches_upstream_in_bf16() {
    cpu_process(
        "dit_parity::single_sample_forward_matches_upstream_in_bf16",
        single_sample_forward_matches_upstream_in_bf16_body,
    );
}

fn single_sample_forward_matches_upstream_in_bf16_body() {
    on_cpu(|| {
        let config = tiny_config();
        let dit = load_dit(
            &tiny_package().join("model.safetensors"),
            &config,
            Dtype::Bfloat16,
        )
        .unwrap();
        let golden = fixture("flux3_dit_golden.safetensors");
        let text = fixture("flux3_text_golden.safetensors");
        let tokens = text.require("short/tokens").unwrap();
        let text_ids = text_ids(tokens.shape()[0] as usize);
        let image_ids = ids_of(golden.require("single/image_ids").unwrap());
        let out = dit
            .forward(&[SampleTokens {
                text: tokens,
                text_ids: &text_ids,
                image: golden.require("single/x").unwrap(),
                image_ids: &image_ids,
                image_cond: None,
                timestep: scalar(golden.require("single/t").unwrap()),
                target_timestep: Some(scalar(golden.require("single/t_ref").unwrap())),
            }])
            .unwrap();
        assert_eq!(out.dtype(), Dtype::Bfloat16);
        assert_close(
            "velocity (bf16)",
            &out,
            golden.require("single/velocity_bf16").unwrap(),
            5e-2,
        );
        // The bf16 policy is a different computation from f32, visibly: the two goldens differ.
        let (max_abs, _, _) = crate::common::errors(
            golden.require("single/velocity_bf16").unwrap(),
            golden.require("single/velocity").unwrap(),
        );
        assert!(max_abs > 1e-4, "the bf16 oracle differs from the f32 one");
    });
}

/// The timestep embedding forms `1000·t` in the compute dtype: under bf16 the DiT sees the
/// rounded time (`0.9163 → 0.9921875·1000 → 992`? no: `bf16(0.9163) = 0.91796875`, `×1000` in
/// bf16 = 916), under f32 the exact one.
#[test]
fn timestep_embedding_follows_the_compute_dtype() {
    cpu_process(
        "dit_parity::timestep_embedding_follows_the_compute_dtype",
        timestep_embedding_follows_the_compute_dtype_body,
    );
}

fn timestep_embedding_follows_the_compute_dtype_body() {
    on_cpu(|| {
        let config = tiny_config();
        let f32_dit = load_dit(
            &tiny_package().join("model.safetensors"),
            &config,
            Dtype::Float32,
        )
        .unwrap();
        let bf16_dit = load_dit(
            &tiny_package().join("model.safetensors"),
            &config,
            Dtype::Bfloat16,
        )
        .unwrap();
        // The first interior point of the 16-step release schedule (≈ 0.99047), read off the
        // contract rather than spelled as a literal f32 cannot hold exactly.
        let t = mlx_gen::gen_core::flux3::flow_map_schedule(16, 6.93).unwrap()[1];
        assert!((0.9904..0.9905).contains(&t), "schedule[1] = {t}");
        let f = host_f32(&f32_dit.timestep_embedding(t).unwrap());
        let b = host_f32(&bf16_dit.timestep_embedding(t).unwrap());
        assert_eq!(f.len(), 256);
        // cos(1000·t · freq_0 = 990.47…) vs cos(992): different phases.
        assert_eq!(f[0], (1000.0f32 * t).cos());
        assert!(
            (b[0] - (992.0f32).cos()).abs() < 2e-2,
            "bf16 time 992: {}",
            b[0]
        );
        assert!(
            (f[0] - b[0]).abs() > 1e-2,
            "the two policies differ at the first frequency"
        );
    });
}
