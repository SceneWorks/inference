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
fn ordered_reference_edits_match_frozen_source() {
    cpu_process(
        "dit_parity::ordered_reference_edits_match_frozen_source",
        ordered_reference_edits_match_frozen_source_body,
    );
}

/// Frozen source independently encodes ordered non-square RGB references and holds their AE
/// tokens fixed across every step, including BOTH base CFG branches at guidance zero and one.
fn ordered_reference_edits_match_frozen_source_body() {
    on_cpu(|| {
        use mlx_gen::gen_core::{
            flux3::{
                assemble_window,
                references::{prepare, ReferenceOutputSize},
            },
            CancelFlag, Image,
        };
        use mlx_gen_flux3::pipeline::{
            decode_tokens, denoise_cfg_with_references, denoise_flow_map_with_references,
        };
        use mlx_gen_flux3::{load_autoencoder, TextConditioning};
        let golden = fixture("flux3_reference_edit_golden.safetensors");
        assert_eq!(
            golden.metadata("upstream_code_revision").unwrap(),
            mlx_gen::gen_core::flux3::UPSTREAM_CODE_REVISION
        );
        let ae = load_autoencoder(&tiny_package().join("ae.safetensors")).unwrap();
        let flow = mlx_gen_flux3::load_dit(
            &tiny_package().join("model.safetensors"),
            &tiny_config(),
            Dtype::Float32,
        )
        .unwrap();
        let mut cfg = tiny_config();
        cfg.dit.flow_map = false;
        let base = mlx_gen_flux3::load_dit(
            &crate::common::fixtures().join("tiny-base-snapshot/model.safetensors"),
            &cfg,
            Dtype::Float32,
        )
        .unwrap();
        let text = fixture("flux3_text_golden.safetensors");
        let conditioning = |name: &str| {
            let tokens = text.require(&format!("{name}/tokens")).unwrap().clone();
            let ids = ids_of(text.require(&format!("{name}/ids")).unwrap());
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
                let t = golden.require(&format!("input/{i}")).unwrap();
                let dims = t.shape();
                Image {
                    width: dims[1] as u32,
                    height: dims[0] as u32,
                    pixels: host_f32(t).into_iter().map(|v| v as u8).collect(),
                }
            })
            .collect();
        let ids = ids_of(golden.require("image_ids").unwrap());
        let schedule = host_f32(golden.require("schedule").unwrap());
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
            let mixed = mlx_gen::gen_core::GenerationRequest {
                flux3_reference_output: Some(ReferenceOutputSize::default()),
                conditioning: order
                    .iter()
                    .map(|&i| {
                        if i.is_multiple_of(2) {
                            mlx_gen::gen_core::Conditioning::ReferenceRgba {
                                image: mlx_gen::gen_core::RgbaImage {
                                    width: originals[i].width,
                                    height: originals[i].height,
                                    pixels: crate::common::host_f32(
                                        golden.require(&format!("rgba/{i}")).unwrap(),
                                    )
                                    .into_iter()
                                    .map(|v| v as u8)
                                    .collect(),
                                },
                                strength: None,
                            }
                        } else {
                            mlx_gen::gen_core::Conditioning::Reference {
                                image: originals[i].clone(),
                                strength: None,
                            }
                        }
                    })
                    .collect(),
                ..Default::default()
            };
            let ae_input = mlx_gen::gen_core::flux3::references::request_images(&mixed).unwrap();
            let mixed_prepared = mlx_gen::gen_core::flux3::references::prepare_borrowed(
                &ae_input.iter().map(|i| i.as_ref()).collect::<Vec<_>>(),
                ReferenceOutputSize::default(),
            )
            .unwrap();
            assert_eq!(
                mixed_prepared.images, prepared.images,
                "source RGB-drop-alpha is exact, not white-premultiplied"
            );
            let pup = mlx_gen::gen_core::flux3::references::upsampler_input(&mixed).unwrap();
            for (position, &i) in order.iter().enumerate() {
                if i.is_multiple_of(2) {
                    let (_, _, rgb) = pup.references[position].to_rgb8().unwrap();
                    let expected =
                        crate::common::host_f32(golden.require(&format!("pup_rgb/{i}")).unwrap())
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
                    let input = Array::from_slice(
                        &values,
                        &[1, image.height as i32, image.width as i32, 3],
                    );
                    let tokens = ae.encode(&input).unwrap().0;
                    mlx_rs::transforms::eval([&tokens]).unwrap();
                    tokens
                })
                .collect::<Vec<_>>();
            let pre_cast_refs = pre_cast.iter().collect::<Vec<_>>();
            let pre_cast = mlx_rs::ops::concatenate_axis(&pre_cast_refs, 0).unwrap();
            assert_reference_f32(
                &host_f32(&pre_cast),
                &host_f32(
                    golden
                        .require(&format!("{name}/reference_tokens_f32"))
                        .unwrap(),
                ),
            );
            let references =
                mlx_gen_flux3::references::encode(&ae, &prepared, &CancelFlag::new()).unwrap();
            assert_eq!(references.tokens.dtype(), Dtype::Bfloat16);
            assert_eq!(
                references.ids,
                ids_of(golden.require(&format!("{name}/reference_ids")).unwrap())
            );
            assert_eq!(
                host_f32(&pre_cast.as_dtype(Dtype::Bfloat16).unwrap()),
                host_f32(&references.tokens),
                "production reference encode must retain exactly the AE's BF16 cast",
            );
            assert_reference_bf16(
                &host_f32(&references.tokens),
                &host_f32(golden.require(&format!("{name}/reference_tokens")).unwrap()),
            );
            let out = denoise_flow_map_with_references(
                &flow,
                &positive,
                golden.require("noise").unwrap(),
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
                golden.require(&format!("{name}/flow_tokens")).unwrap(),
                1e-4,
            );
            let decoded = decode_tokens(&ae, &out, 2, 3).unwrap();
            let expected = host_f32(golden.require(&format!("{name}/rgb8")).unwrap())
                .into_iter()
                .map(|v| v as u8)
                .collect::<Vec<_>>();
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
                        golden.require("noise").unwrap(),
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
                        golden.require(&format!("{name}/cfg_{g:.1}")).unwrap(),
                        1e-4,
                    );
                }
            }
        }
    });
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
                    Weights::from_file(tiny_package().join("model.safetensors")).unwrap();
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
