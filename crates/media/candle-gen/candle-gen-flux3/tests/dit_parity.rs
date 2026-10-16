//! Flow-map DiT forward vs upstream `DiT.forward` on the miniature checkpoint (per-mode early
//! blocks, the shared single stream over `[image, text]`, 4-axis RoPE, q/k RMSNorm, the parallel
//! attention + SwiGLU block, target-time conditioning, `LastLayer`), single-sample and
//! two-sample packs — the same fixture the MLX twin's `dit_parity` reads.
//!
//! Tolerances:
//! * FP32 (`Precision::Fp32` compute, Candle CPU) vs the oracle's FP32 CPU forward over the same
//!   bf16-rounded parameters and inputs: only summation order differs — 1e-4 of peak.
//! * bf16 compute (the release's autocast policy) vs the oracle's bf16-autocast CPU forward: bf16
//!   matmul / attention rounding on both sides — 5e-2 of peak. Runs on the build's device, so the
//!   CUDA lane holds the real bf16 kernels to the same fixture; on a CPU build the bf16 policy is
//!   the f32-accumulate / round-to-bf16 emulation.

use candle_gen::candle_core::{DType, Device};
use candle_gen::gen_core::flux3::text_ids;
use candle_gen_flux3::{load_dit, SampleTokens};

use crate::common::{
    assert_close, cpu, device, errors, fixture, host_f32, ids_of, scalar, tiny_config, tiny_package,
};

fn dit(compute: DType, device: &Device) -> candle_gen_flux3::Flux3Dit {
    load_dit(
        &tiny_package().join("model.safetensors"),
        &tiny_config(),
        compute,
        device,
    )
    .unwrap()
}

#[test]
fn single_sample_forward_matches_upstream_in_fp32() {
    let dit = dit(DType::F32, &cpu());
    assert!(dit.is_flow_map());
    let golden = fixture("flux3_dit_golden.safetensors");
    let text = fixture("flux3_text_golden.safetensors");
    let tokens = text.require("short/tokens");
    let text_ids = text_ids(tokens.dim(0).unwrap());
    let image_ids = ids_of(golden.require("single/image_ids"));
    let out = dit
        .forward(&[SampleTokens {
            text: tokens,
            text_ids: &text_ids,
            image: golden.require("single/x"),
            image_ids: &image_ids,
            image_cond: None,
            timestep: scalar(golden.require("single/t")),
            target_timestep: Some(scalar(golden.require("single/t_ref"))),
        }])
        .unwrap();
    assert_eq!(out.dtype(), DType::F32);
    assert_close("velocity", &out, golden.require("single/velocity"), 1e-4);
}

/// Two samples with different prompts, grids and times, packed mode-major: each sample's image
/// tokens attend only their own sample (the per-sample bounds), and sample 0's velocity is the
/// single-sample velocity.
///
/// *Mutation that reds this:* attending over the whole packed stream instead of per sample, or
/// applying one sample's modulation rows to the other's tokens.
#[test]
fn two_sample_pack_matches_upstream_in_fp32() {
    let dit = dit(DType::F32, &cpu());
    let golden = fixture("flux3_dit_golden.safetensors");
    let text = fixture("flux3_text_golden.safetensors");
    let tokens0 = text.require("short/tokens");
    let tokens1 = text.require("empty/tokens");
    let text_ids0 = text_ids(tokens0.dim(0).unwrap());
    let text_ids1 = text_ids(tokens1.dim(0).unwrap());
    let image_ids0 = ids_of(golden.require("single/image_ids"));
    let image_ids1 = ids_of(golden.require("pair/image_ids2"));
    let out = dit
        .forward(&[
            SampleTokens {
                text: tokens0,
                text_ids: &text_ids0,
                image: golden.require("single/x"),
                image_ids: &image_ids0,
                image_cond: None,
                timestep: scalar(golden.require("single/t")),
                target_timestep: Some(scalar(golden.require("single/t_ref"))),
            },
            SampleTokens {
                text: tokens1,
                text_ids: &text_ids1,
                image: golden.require("pair/x2"),
                image_ids: &image_ids1,
                image_cond: None,
                timestep: scalar(golden.require("pair/t2")),
                target_timestep: Some(scalar(golden.require("pair/t2_ref"))),
            },
        ])
        .unwrap();
    let want = golden.require("pair/velocity");
    assert_eq!(out.dims(), want.dims());
    assert_close("pair velocity", &out, want, 1e-4);
    let n0 = golden.require("single/x").dim(0).unwrap();
    assert_close(
        "pair sample 0 == single",
        &out.narrow(0, 0, n0).unwrap(),
        golden.require("single/velocity"),
        1e-4,
    );
}

/// The bf16 policy on the build's device vs the oracle's bf16-autocast forward.
#[test]
fn single_sample_forward_matches_upstream_in_bf16() {
    let device = device();
    let dit = dit(DType::BF16, &device);
    let golden = fixture("flux3_dit_golden.safetensors");
    let text = fixture("flux3_text_golden.safetensors");
    let on = |key: &str| golden.require(key).to_device(&device).unwrap();
    let tokens = text.require("short/tokens").to_device(&device).unwrap();
    let text_ids = text_ids(tokens.dim(0).unwrap());
    let image_ids = ids_of(golden.require("single/image_ids"));
    let x = on("single/x");
    let out = dit
        .forward(&[SampleTokens {
            text: &tokens,
            text_ids: &text_ids,
            image: &x,
            image_ids: &image_ids,
            image_cond: None,
            timestep: scalar(golden.require("single/t")),
            target_timestep: Some(scalar(golden.require("single/t_ref"))),
        }])
        .unwrap();
    assert_eq!(out.dtype(), DType::BF16);
    assert_close(
        "velocity (bf16)",
        &out,
        golden.require("single/velocity_bf16"),
        5e-2,
    );
    // The bf16 policy is a different computation from f32, visibly: the two goldens differ.
    let (max_abs, _, _) = errors(
        golden.require("single/velocity_bf16"),
        golden.require("single/velocity"),
    );
    assert!(max_abs > 1e-4, "the bf16 oracle differs from the f32 one");
}

/// The timestep embedding forms `1000·t` in the compute dtype: under bf16 the DiT sees the
/// rounded time (`bf16(0.9163) = 0.91796875`, `×1000` in bf16 = 916), under f32 the exact one.
#[test]
fn timestep_embedding_follows_the_compute_dtype() {
    let f32_dit = dit(DType::F32, &cpu());
    let bf16_dit = dit(DType::BF16, &cpu());
    let t = 0.990_471_7f32; // the first interior point of the 16-step release schedule
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
}

/// A flow-map DiT refuses a sample without a target time; the checkpoint/config `flow_map`
/// disagreement is a typed refusal at load.
#[test]
fn flow_map_inputs_and_config_are_checked() {
    let dit = dit(DType::F32, &cpu());
    let golden = fixture("flux3_dit_golden.safetensors");
    let text = fixture("flux3_text_golden.safetensors");
    let tokens = text.require("short/tokens");
    let text_ids = text_ids(tokens.dim(0).unwrap());
    let image_ids = ids_of(golden.require("single/image_ids"));
    let err = dit
        .forward(&[SampleTokens {
            text: tokens,
            text_ids: &text_ids,
            image: golden.require("single/x"),
            image_ids: &image_ids,
            image_cond: None,
            timestep: 0.5,
            target_timestep: None,
        }])
        .expect_err("a flow map needs t_ref");
    assert!(err.to_string().contains("target timestep"), "{err}");

    let mut config = tiny_config();
    config.dit.flow_map = false;
    match load_dit(
        &tiny_package().join("model.safetensors"),
        &config,
        DType::F32,
        &cpu(),
    ) {
        Err(candle_gen::CandleError::Unsupported(m)) => assert!(m.contains("flow_map"), "{m}"),
        other => panic!("expected a typed refusal, got {:?}", other.err()),
    }
}

/// Independent frozen-source CFG and additive adapter goldens; CPU only, no real weights.
#[test]
fn base_cfg_and_explicit_adapter_identity_match_frozen_source() {
    use candle_gen::candle_core::safetensors;
    use candle_gen::candle_nn::VarBuilder;
    use candle_gen::gen_core::{
        flux3::{adapters, assemble_window},
        AdapterKind, AdapterSpec, CancelFlag,
    };
    use candle_gen_flux3::pipeline::denoise_cfg;
    use candle_gen_flux3::{Flux3Dit, TextConditioning};
    let device = Device::Cpu;
    let mut config = tiny_config();
    config.dit.flow_map = false;
    let weights = safetensors::load(tiny_package().join("model.safetensors"), &device)
        .unwrap()
        .into_iter()
        .filter(|(key, _)| !key.starts_with("ref_time_in."))
        .collect();
    let vb = VarBuilder::from_tensors(weights, DType::F32, &device);
    let mut dit = Flux3Dit::from_var_builder(&vb, &config.dit, DType::F32, &device).unwrap();
    let text_fixture = fixture("flux3_text_golden.safetensors");
    let conditioning = |name: &str| {
        let tokens = text_fixture.require(&format!("{name}/tokens")).clone();
        let ids = ids_of(text_fixture.require(&format!("{name}/ids")));
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
    let x = golden.require("noise");
    let ids = ids_of(golden.require("image_ids"));
    let schedule = host_f32(golden.require("schedule"));
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
            golden.require(&format!("cfg_{guidance:.1}")),
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
        golden.require("negative_alt"),
        1e-4,
    );
    let spec = AdapterSpec::new(
        crate::common::fixtures().join("flux3_base_lora.safetensors"),
        -0.25,
        AdapterKind::Lora,
    );
    let plan = adapters::inspect(&spec, &config.dit).unwrap();
    assert_eq!(plan.len(), 6);
    candle_gen_flux3::adapters::install(&mut dit, std::slice::from_ref(&spec)).unwrap();
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
    assert_close("six target LoRA", &out, golden.require("adapter_on"), 1e-4);
    let max = errors(&out, golden.require("cfg_3.5")).0;
    assert!(max > 1e-5, "adapter must change actual sampled tokens");
    let mut wrong = spec.clone();
    wrong.scale = f32::NAN;
    assert!(matches!(
        candle_gen_flux3::adapters::validate(&[wrong], &config.dit),
        Err(candle_gen::CandleError::Unsupported(_))
    ));
    let mut zero = spec;
    zero.scale = 0.0;
    let mut zero_dit = Flux3Dit::from_var_builder(&vb, &config.dit, DType::F32, &device).unwrap();
    let bare = denoise_cfg(
        &zero_dit,
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
    candle_gen_flux3::adapters::install(&mut zero_dit, &[zero]).unwrap();
    let disabled = denoise_cfg(
        &zero_dit,
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
    assert_eq!(
        host_f32(&bare),
        host_f32(&disabled),
        "zero adapter scale is a bitwise no-op"
    );
}
