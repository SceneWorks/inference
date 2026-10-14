//! sc-25909: Candle-on-Metal reference renders. The VAE encoder's `AvgDown` shortcut took a 5-D
//! strided mean, which candle-metal-kernels' `reduce.metal` `get_strided_idx_fallback` mis-indexes
//! (out-of-bounds reads), so reference latents were wrong and pool-dependent (NaN after the
//! vision tower ran). Meaningful on `--features metal`; on CPU these compare CPU with CPU.

use candle_core::{Device, Tensor};
use candle_gen::gen_core::{Image, RgbaImage};
use candle_gen_qwen_image_2_1::{
    encode_references, load_text_encoder, load_tokenizer, load_vae,
    prepare_conditioning_references, system_prompt_drop_count,
};

use crate::common::{host_f32, turbo_snapshot};

fn reference(kind: u32) -> RgbaImage {
    let edge = 64u32;
    let mut pixels = Vec::with_capacity((edge * edge * 3) as usize);
    for y in 0..edge {
        for x in 0..edge {
            if kind == 0 {
                pixels.extend([(x * 4) as u8, (y * 4) as u8, 32]);
            } else {
                let on = (x / 8 + y / 8) % 2 == 0;
                pixels.extend([if on { 230 } else { 20 }, 128, (255 - x * 3) as u8]);
            }
        }
    }
    RgbaImage::from_rgb(&Image {
        width: edge,
        height: edge,
        pixels,
    })
    .unwrap()
}

fn poison(device: &Device) {
    // Fill and drop pool buffers of many sizes with NaN so an op that does not write every
    // element of its output exposes it.
    for n in [
        64usize, 128, 256, 512, 1024, 1156, 2048, 4096, 8192, 16384, 65536,
    ] {
        for dt in [candle_core::DType::F32, candle_core::DType::BF16] {
            let t = Tensor::full(f32::NAN, (n,), device)
                .unwrap()
                .to_dtype(dt)
                .unwrap();
            let _ = t
                .sum_all()
                .unwrap()
                .to_dtype(candle_core::DType::F32)
                .unwrap()
                .to_vec0::<f32>();
        }
    }
}

fn det(shape: &[usize], seed: usize, device: &Device) -> Tensor {
    let n: usize = shape.iter().product();
    let v: Vec<f32> = (0..n)
        .map(|i| (((i * 37 + seed * 11) % 97) as f32 / 97.0) - 0.5)
        .collect();
    Tensor::from_vec(v, shape, device).unwrap()
}

/// sc-25909 root cause: a reduction over a non-trailing dim of a >4-D tensor takes candle-metal's
/// `get_strided_idx_fallback`, which decodes the wrong dims (reads out of bounds).
#[test]
#[ignore = "documents the unfixed candle-metal-kernels rank>4 strided-reduce bug (1e6aa85); red on Metal"]
fn strided_reduce_rank_vs_cpu() {
    let gpu = candle_gen::default_device().unwrap();
    let cases: Vec<(&str, Vec<usize>, Vec<usize>, usize)> = vec![
        (
            "4d mean(2) [1,2,2,256]",
            vec![1, 2, 2, 16, 16],
            vec![1, 2, 2, 256],
            2,
        ),
        (
            "5d mean(2) [1,2,2,16,16]",
            vec![1, 2, 2, 16, 16],
            vec![1, 2, 2, 16, 16],
            2,
        ),
        (
            "5d mean(2) [1,8,2,4,4] (AvgDown)",
            vec![1, 8, 2, 4, 4],
            vec![1, 8, 2, 4, 4],
            2,
        ),
        (
            "6d sum(3)",
            vec![1, 2, 3, 2, 4, 4],
            vec![1, 2, 3, 2, 4, 4],
            3,
        ),
    ];
    let mut bad = Vec::new();
    for (name, src, view, dim) in cases {
        let g = det(&src, 5, &gpu)
            .reshape(view.as_slice())
            .unwrap()
            .mean(dim)
            .unwrap();
        let c = det(&src, 5, &Device::Cpu)
            .reshape(view.as_slice())
            .unwrap()
            .mean(dim)
            .unwrap();
        let (g, c) = (host_f32(&g), host_f32(&c));
        let d = g
            .iter()
            .zip(&c)
            .fold(0f32, |m, (x, y)| m.max((x - y).abs()));
        let nf = g.iter().filter(|v| !v.is_finite()).count();
        eprintln!("{name}: max|gpu-cpu|={d:.3e} nonfinite={nf}");
        if d > 1e-5 || nf > 0 {
            bad.push(name);
        }
    }
    assert!(
        bad.is_empty(),
        "Metal reduction disagrees with CPU: {bad:?}"
    );
}

/// The reported symptom at the generator level: one loaded generator, a 2-reference render, a
/// 10-reference render, then the same 2-reference render again (prefix cache unset = prefill).
/// The two 2-reference renders must be byte-identical.
#[test]
fn two_reference_render_is_reproducible_after_a_ten_reference_render() {
    use candle_gen::gen_core::{
        Conditioning, GenerationOutput, GenerationRequest, LoadSpec, WeightsSource,
    };
    let root = crate::common::tiny_snapshot();
    let generator = candle_gen_qwen_image_2_1::provider_registry()
        .unwrap()
        .load("qwen_image_2_1", &LoadSpec::new(WeightsSource::Dir(root)))
        .unwrap();
    let img = |kind: u32| {
        let r = reference(kind);
        Image {
            width: r.width,
            height: r.height,
            pixels: r
                .pixels
                .chunks_exact(4)
                .flat_map(|p| [p[0], p[1], p[2]])
                .collect(),
        }
    };
    let req = |n: usize| GenerationRequest {
        prompt: "a red fox in the forest".to_owned(),
        width: 32,
        height: 32,
        seed: Some(42),
        steps: Some(4),
        conditioning: vec![Conditioning::MultiReference {
            images: (0..n).map(|i| img((i % 2) as u32)).collect(),
        }],
        ..Default::default()
    };
    let render = |n: usize| -> Vec<u8> {
        let GenerationOutput::Images(images) = generator.generate(&req(n), &mut |_| {}).unwrap()
        else {
            panic!("images");
        };
        images[0].pixels.clone()
    };
    let first = render(2);
    let _ = render(10);
    let again = render(2);
    poison(&candle_gen::default_device().unwrap());
    let poisoned = render(2);
    let pm = first
        .iter()
        .zip(&poisoned)
        .map(|(a, b)| (*a as f64 - *b as f64).abs())
        .sum::<f64>()
        / first.len() as f64;
    eprintln!("2-ref before vs after NaN-poisoning the device pool: mean|D|={pm:.2}/255");
    assert_eq!(
        first, poisoned,
        "a 2-reference render depends on the device pool contents"
    );
    let mean = first
        .iter()
        .zip(&again)
        .map(|(a, b)| (*a as f64 - *b as f64).abs())
        .sum::<f64>()
        / first.len() as f64;
    eprintln!("2-ref before vs after a 10-ref render: mean|Δ|={mean:.2}/255");
    assert_eq!(
        first, again,
        "a 2-reference render depends on the previous render"
    );
}

/// The proof test for the in-crate fix: after the reference text-encoder route has run on the
/// device (the state the bug showed NaN in), the device's packed reference latents are finite and
/// agree with the CPU's to bf16 rounding. Before the fix: 128/128 non-finite, or off by ~0.3.
#[test]
fn reference_vae_latents_on_the_device_match_cpu_after_the_text_encoder() {
    let encode = |device: &Device| -> Vec<f32> {
        let fixture = turbo_snapshot();
        let root = fixture.path();
        let tokenizer = load_tokenizer(root).unwrap();
        let encoder = load_text_encoder(root, device).unwrap();
        let vae = load_vae(root, device).unwrap();
        let drop = system_prompt_drop_count(&tokenizer).unwrap();
        let prepared = prepare_conditioning_references(&encoder, &[reference(0)]).unwrap();
        encoder
            .encode_conditioning(&tokenizer, "a red fox in the forest", drop, &prepared)
            .unwrap();
        host_f32(&encode_references(&vae, &prepared).unwrap()[0])
    };
    let device = encode(&candle_gen::default_device().unwrap());
    let cpu = encode(&Device::Cpu);
    let nonfinite = device.iter().filter(|v| !v.is_finite()).count();
    let worst = device
        .iter()
        .zip(&cpu)
        .fold(0f32, |m, (a, b)| m.max((a - b).abs()));
    eprintln!("device vs cpu reference latents: nonfinite={nonfinite} max|D|={worst:.3e}");
    assert_eq!(nonfinite, 0, "non-finite reference latents on the device");
    // bf16 VAE on Metal vs f32 on CPU: measured 3.2e-3 after the fix, 3.2e-1 before.
    assert!(
        worst < 3e-2,
        "device reference latents disagree with CPU: {worst:.3e}"
    );
}
