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

/// The timestep embedding forms `1000·t` in the compute dtype: the first interior release
/// timestep becomes 992 under bf16, versus about 990.47 under f32. Evaluate independent phase
/// references with the same backend transcendental kernel, not Rust libm's different rounding.
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
        let f_embedding = f32_dit.timestep_embedding(t).unwrap();
        let b_embedding = bf16_dit.timestep_embedding(t).unwrap();
        assert_eq!(f_embedding.dtype(), Dtype::Float32);
        assert_eq!(b_embedding.dtype(), Dtype::Bfloat16);
        let f = host_f32(&f_embedding);
        let b = host_f32(&b_embedding);
        assert_eq!(f.len(), 256);
        assert_eq!(b.len(), 256);
        // Pin frequencies 0 and 1 independently of timestep_freqs/DiT. Keep the same 128-wide
        // backend unary kernel as the cosine/sine halves; Rust f32::cos differs by one ulp on
        // macOS, which is not a phase/dtype defect. No numeric tolerance is relaxed here.
        let second_frequency = (-(10000.0f64.ln() as f32) / 128.0).exp();
        let reference = |phase: f32, dtype: Dtype| {
            let mut phases = [phase; 128];
            phases[1] = phase * second_frequency;
            let args = mlx_rs::Array::from_slice(&phases, &[1, 128]);
            let mut values = host_f32(&args.cos().unwrap().as_dtype(dtype).unwrap());
            values.extend(host_f32(&args.sin().unwrap().as_dtype(dtype).unwrap()));
            values
        };
        let expected_f = reference(1000.0f32 * t, Dtype::Float32);
        let expected_b = reference(992.0, Dtype::Bfloat16);
        for index in [0, 1, 128, 129] {
            assert_eq!(
                f[index], expected_f[index],
                "f32 phase/frequency at {index}"
            );
            assert_eq!(
                b[index], expected_b[index],
                "bf16 phase/frequency at {index}"
            );
        }
        // Negative controls: omitting the 1000 scaling or using the f32 phase for bf16 must
        // disagree with these exact references (the latter still has the right output dtype).
        assert_ne!(f[0], reference(t, Dtype::Float32)[0]);
        assert_ne!(b[0], reference(1000.0f32 * t, Dtype::Bfloat16)[0]);
        assert!(
            (f[0] - b[0]).abs() > 1e-2,
            "the two policies differ at the first frequency"
        );
    });
}

/// Same independent frozen-source fixture as Candle; hosted macOS verifies this body.
#[test]
fn base_cfg_and_explicit_adapter_identity_match_frozen_source() {
    cpu_process(
        "dit_parity::base_cfg_and_explicit_adapter_identity_match_frozen_source",
        || {
            on_cpu(|| {
                use mlx_gen::{
                    gen_core::{flux3::assemble_window, AdapterKind, AdapterSpec, CancelFlag},
                    weights::Weights,
                };
                use mlx_gen_flux3::pipeline::denoise_cfg;
                use mlx_gen_flux3::{Flux3Dit, TextConditioning};
                let mut config = tiny_config();
                config.dit.flow_map = false;
                let mut weights =
                    Weights::from_file(&tiny_package().join("model.safetensors")).unwrap();
                let remove: Vec<_> = weights
                    .keys()
                    .filter(|key| key.starts_with("ref_time_in."))
                    .map(str::to_owned)
                    .collect();
                for key in remove {
                    weights.remove(&key);
                }
                let mut dit =
                    Flux3Dit::from_weights(&weights, &config.dit, Dtype::Float32).unwrap();
                let text_fixture = fixture("flux3_text_golden.safetensors");
                let conditioning = |name: &str| {
                    let tokens = text_fixture
                        .require(&format!("{name}/tokens"))
                        .unwrap()
                        .clone();
                    let ids = ids_of(text_fixture.require(&format!("{name}/ids")).unwrap());
                    let window = assemble_window(&vec![1; ids.len()], 0, 1, 4096).unwrap();
                    TextConditioning {
                        tokens,
                        ids,
                        window,
                    }
                };
                let positive = conditioning("short");
                let negative = conditioning("empty");
                let alt = conditioning("long");
                let golden = fixture("flux3_base_golden.safetensors");
                let x = golden.require("noise").unwrap();
                let ids = ids_of(golden.require("image_ids").unwrap());
                let schedule = host_f32(golden.require("schedule").unwrap());
                for guidance in [0.0f32, 1.0, 3.5] {
                    let mut steps = Vec::new();
                    let output = denoise_cfg(
                        &dit,
                        &positive,
                        &negative,
                        x,
                        &ids,
                        &schedule,
                        guidance,
                        &CancelFlag::new(),
                        |step| steps.push(step),
                    )
                    .unwrap();
                    assert_eq!(steps, [1, 2, 3, 4]);
                    assert_close(
                        "base CFG",
                        &output,
                        golden.require(&format!("cfg_{guidance:.1}")).unwrap(),
                        1e-4,
                    );
                }
                let out = denoise_cfg(
                    &dit,
                    &positive,
                    &alt,
                    x,
                    &ids,
                    &schedule,
                    3.5,
                    &CancelFlag::new(),
                    |_| {},
                )
                .unwrap();
                assert_close(
                    "explicit negative row",
                    &out,
                    golden.require("negative_alt").unwrap(),
                    1e-4,
                );
                let spec = AdapterSpec::new(
                    crate::common::fixtures().join("flux3_base_lora.safetensors"),
                    -0.25,
                    AdapterKind::Lora,
                );
                mlx_gen_flux3::adapters::install(&mut dit, &[spec]).unwrap();
                let out = denoise_cfg(
                    &dit,
                    &positive,
                    &negative,
                    x,
                    &ids,
                    &schedule,
                    3.5,
                    &CancelFlag::new(),
                    |_| {},
                )
                .unwrap();
                assert_close(
                    "six target LoRA",
                    &out,
                    golden.require("adapter_on").unwrap(),
                    1e-4,
                );
                assert!(crate::common::errors(&out, golden.require("cfg_3.5").unwrap()).0 > 1e-5);
            });
        },
    );
}
