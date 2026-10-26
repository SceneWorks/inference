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

// AE computes in f32 before the source explicitly stores conditioning in BF16. CPU summation
// order can flip a rounding midpoint without violating the pre-cast gate. Keep the source f32
// max bound, add a tighter mean guard, then grade the stored dtype in ULPs rather than raising
// the absolute bound for one reference count. A sparse-rounding guard rejects systematic bias.
fn assert_reference_f32(got: &[f32], want: &[f32]) {
    assert_eq!(got.len(), want.len());
    assert!(!got.is_empty());
    let peak = want
        .iter()
        .fold(1.0_f32, |peak, value| peak.max(value.abs()));
    let mut max = 0.0_f32;
    let mut sum = 0.0_f64;
    for (&a, &b) in got.iter().zip(want) {
        let delta = (a - b).abs();
        assert!(delta.is_finite());
        max = max.max(delta);
        sum += f64::from(delta);
    }
    let mean = sum / got.len() as f64;
    eprintln!("AE pre-cast f32: max={max:.9e} mean={mean:.9e} peak={peak:.9e}");
    assert!(max <= 1e-4 * peak, "AE pre-cast f32 max drift: {max}");
    assert!(
        mean <= 1e-6 * f64::from(peak),
        "AE pre-cast f32 mean drift: {mean}"
    );
}

fn assert_reference_bf16(got: &[f32], want: &[f32]) {
    assert_eq!(got.len(), want.len());
    assert!(!got.is_empty());
    let ordered = |value: f32| -> u32 {
        assert!(value.is_finite());
        assert_eq!(
            value.to_bits() & 0xffff,
            0,
            "reference token omitted its BF16 cast"
        );
        let bits = value.to_bits() >> 16;
        if bits & 0x8000 != 0 {
            0x8000 - (bits & 0x7fff)
        } else {
            0x8000 + bits
        }
    };
    let peak = want
        .iter()
        .fold(1.0_f32, |peak, value| peak.max(value.abs()));
    let mut changed = 0_usize;
    let mut sum = 0.0_f64;
    for (&a, &b) in got.iter().zip(want) {
        let ulp = ordered(a).abs_diff(ordered(b));
        assert!(ulp <= 1, "AE stored BF16 drift: {a} vs {b}, {ulp} ULPs");
        changed += usize::from(ulp != 0);
        sum += f64::from((a - b).abs());
    }
    let mean = sum / got.len() as f64;
    eprintln!(
        "AE stored BF16: changed={changed}/{} mean={mean:.9e} peak={peak:.9e}",
        got.len()
    );
    assert!(
        mean <= 1e-4 * f64::from(peak),
        "AE stored BF16 mean drift: {mean}"
    );
    assert!(
        changed * 100 <= got.len(),
        "AE stored BF16 rounding differences exceed 1%: {changed}/{}",
        got.len()
    );
}

#[test]
fn reference_precision_guards_reject_bias_two_ulps_and_missing_cast() {
    let want = vec![0.125_f32; 100];
    let next = f32::from_bits(want[0].to_bits() + (1 << 16));
    let mut sparse = want.clone();
    sparse[0] = next;
    assert_reference_bf16(&sparse, &want);
    let mut biased = sparse.clone();
    biased[1] = next;
    assert!(std::panic::catch_unwind(|| assert_reference_bf16(&biased, &want)).is_err());
    let mut two_ulps = want.clone();
    two_ulps[0] = f32::from_bits(want[0].to_bits() + (2 << 16));
    assert!(std::panic::catch_unwind(|| assert_reference_bf16(&two_ulps, &want)).is_err());
    let mut uncast = want.clone();
    uncast[0] = f32::from_bits(want[0].to_bits() + 1);
    assert!(std::panic::catch_unwind(|| assert_reference_bf16(&uncast, &want)).is_err());
    let f32_bias = vec![0.12505_f32; 100];
    assert!(std::panic::catch_unwind(|| assert_reference_f32(&f32_bias, &want)).is_err());
}

fn dit(compute: DType, device: &Device) -> candle_gen_flux3::Flux3Dit {
    load_dit(
        &tiny_package().join("model.safetensors"),
        &tiny_config(),
        compute,
        device,
    )
    .unwrap()
}

/// Frozen source independently encodes ordered non-square RGB references and holds their AE
/// tokens fixed across every step, including BOTH base CFG branches at guidance zero and one.
#[test]
fn ordered_reference_edits_match_frozen_source() {
    use candle_gen::gen_core::{
        flux3::{
            assemble_window,
            references::{prepare, ReferenceOutputSize},
        },
        CancelFlag, Image,
    };
    use candle_gen_flux3::pipeline::{
        decode_tokens, denoise_cfg_with_references, denoise_flow_map_with_references,
    };
    use candle_gen_flux3::{load_autoencoder, TextConditioning};
    let dev = Device::Cpu;
    let golden = fixture("flux3_reference_edit_golden.safetensors");
    assert_eq!(
        golden.meta("upstream_code_revision"),
        candle_gen::gen_core::flux3::UPSTREAM_CODE_REVISION
    );
    let ae = load_autoencoder(&tiny_package().join("ae.safetensors"), &dev).unwrap();
    let flow = dit(DType::F32, &dev);
    let mut cfg = tiny_config();
    cfg.dit.flow_map = false;
    let base = candle_gen_flux3::load_dit(
        &crate::common::fixtures().join("tiny-base-snapshot/model.safetensors"),
        &cfg,
        DType::F32,
        &dev,
    )
    .unwrap();
    let text = fixture("flux3_text_golden.safetensors");
    let conditioning = |name: &str| {
        let tokens = text.require(&format!("{name}/tokens")).clone();
        let ids = ids_of(text.require(&format!("{name}/ids")));
        let window = assemble_window(&vec![1; ids.len()], 0, 1, 4096).unwrap();
        TextConditioning {
            tokens,
            ids,
            window,
        }
    };
    let positive = conditioning("short");
    let negative = conditioning("empty");
    let originals: Vec<Image> = (0..10)
        .map(|i| {
            let t = golden.require(&format!("input/{i}"));
            let dims = t.dims();
            Image {
                width: dims[1] as u32,
                height: dims[0] as u32,
                pixels: t.flatten_all().unwrap().to_vec1::<u8>().unwrap(),
            }
        })
        .collect();
    let ids = ids_of(golden.require("image_ids"));
    let schedule = host_f32(golden.require("schedule"));
    for (name, order) in [
        ("one", vec![0]),
        ("two", vec![0, 1]),
        ("ten", (0..10).collect()),
        ("reversed", (0..10).rev().collect()),
    ] {
        let input = order
            .iter()
            .map(|&i| originals[i].clone())
            .collect::<Vec<_>>();
        let prepared = prepare(&input, ReferenceOutputSize::default()).unwrap();
        // Same source colors with alpha0/128/255: AE drops alpha; PUP independently composites.
        let mixed = candle_gen::gen_core::GenerationRequest {
            flux3_reference_output: Some(ReferenceOutputSize::default()),
            conditioning: order
                .iter()
                .map(|&i| {
                    if i % 2 == 0 {
                        candle_gen::gen_core::Conditioning::ReferenceRgba {
                            image: candle_gen::gen_core::RgbaImage {
                                width: originals[i].width,
                                height: originals[i].height,
                                pixels: crate::common::host_f32(
                                    golden.require(&format!("rgba/{i}")),
                                )
                                .into_iter()
                                .map(|v| v as u8)
                                .collect(),
                            },
                            strength: None,
                        }
                    } else {
                        candle_gen::gen_core::Conditioning::Reference {
                            image: originals[i].clone(),
                            strength: None,
                        }
                    }
                })
                .collect(),
            ..Default::default()
        };
        let ae_input = candle_gen::gen_core::flux3::references::request_images(&mixed).unwrap();
        let mixed_prepared = candle_gen::gen_core::flux3::references::prepare_borrowed(
            &ae_input.iter().map(|i| i.as_ref()).collect::<Vec<_>>(),
            ReferenceOutputSize::default(),
        )
        .unwrap();
        assert_eq!(
            mixed_prepared.images, prepared.images,
            "source RGB-drop-alpha is exact, not white-premultiplied"
        );
        let pup = candle_gen::gen_core::flux3::references::upsampler_input(&mixed).unwrap();
        for (position, &i) in order.iter().enumerate() {
            if i % 2 == 0 {
                let (_, _, rgb) = pup.references[position].to_rgb8().unwrap();
                let expected = crate::common::host_f32(golden.require(&format!("pup_rgb/{i}")))
                    .into_iter()
                    .map(|v| v as u8)
                    .collect::<Vec<_>>();
                assert_eq!(rgb, expected, "Pillow source alpha-over-white fork");
            }
        }
        let prepared = mixed_prepared;
        let pre_cast = prepared
            .images
            .iter()
            .map(|image| {
                let values = image
                    .pixels
                    .iter()
                    .map(|&v| v as f32 / 127.5 - 1.0)
                    .collect::<Vec<_>>();
                let input = candle_gen::candle_core::Tensor::from_vec(
                    values,
                    (image.height as usize, image.width as usize, 3),
                    &dev,
                )
                .unwrap()
                .permute((2, 0, 1))
                .unwrap()
                .unsqueeze(0)
                .unwrap()
                .contiguous()
                .unwrap();
                ae.encode(&input).unwrap().0
            })
            .collect::<Vec<_>>();
        let pre_cast = candle_gen::candle_core::Tensor::cat(&pre_cast, 0).unwrap();
        assert_reference_f32(
            &host_f32(&pre_cast),
            &host_f32(golden.require(&format!("{name}/reference_tokens_f32"))),
        );
        let references =
            candle_gen_flux3::references::encode(&ae, &prepared, &dev, &CancelFlag::new()).unwrap();
        assert_eq!(references.tokens.dtype(), DType::BF16);
        assert_eq!(
            references.ids,
            ids_of(golden.require(&format!("{name}/reference_ids")))
        );
        assert_eq!(
            host_f32(&pre_cast.to_dtype(DType::BF16).unwrap()),
            host_f32(&references.tokens),
            "production reference encode must retain exactly the AE's BF16 cast",
        );
        assert_reference_bf16(
            &host_f32(&references.tokens),
            &host_f32(golden.require(&format!("{name}/reference_tokens"))),
        );
        let out = denoise_flow_map_with_references(
            &flow,
            &positive,
            golden.require("noise"),
            &ids,
            &schedule,
            Some(&references),
            &CancelFlag::new(),
            |_| {},
        )
        .unwrap();
        assert_close(
            name,
            &out,
            golden.require(&format!("{name}/flow_tokens")),
            1e-4,
        );
        let decoded = decode_tokens(&ae, &out, 2, 3).unwrap();
        let expected = golden
            .require(&format!("{name}/rgb8"))
            .flatten_all()
            .unwrap()
            .to_vec1::<u8>()
            .unwrap();
        assert_eq!((decoded.width, decoded.height), (48, 32));
        let delta = decoded
            .pixels
            .iter()
            .zip(&expected)
            .map(|(&a, &b)| a.abs_diff(b))
            .max()
            .unwrap();
        assert!(delta <= 1, "decoded RGB rounding, max byte delta {delta}");
        if name == "two" || name == "ten" {
            for g in [0.0f32, 1.0, 3.5] {
                let result = denoise_cfg_with_references(
                    &base,
                    &positive,
                    &negative,
                    golden.require("noise"),
                    &ids,
                    &schedule,
                    g,
                    Some(&references),
                    &CancelFlag::new(),
                    |_| {},
                )
                .unwrap();
                assert_close(
                    "base ordered reference CFG",
                    &result,
                    golden.require(&format!("{name}/cfg_{g:.1}")),
                    1e-4,
                );
            }
        }
    }
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
