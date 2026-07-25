//! Actual BF16/Q8/Q4 generation, text-fidelity, and VAE round-trip acceptance.
//!
//! Text-tier decision (sc-14046): Q8 is the quality floor for fidelity-sensitive use
//! (four-prompt pooled-vector relative error `<= 0.04`, cosine `>= 0.995`). Q4 remains a valid
//! low-memory, explicitly low-fidelity tier only while it preserves prompt identity/ranking and
//! separates matched prompts from a rotated wrong-prompt mutation; callers should prefer Q8 whenever
//! its fit gate admits it.

use mlx_gen::{GenerationOutput, GenerationRequest, LoadSpec, Quant, WeightsSource};
use mlx_gen_mage::model::REGISTRATION_BASE;
use mlx_gen_mage::text_encoder::PromptKind;
use mlx_gen_mage::vae::{self, VaePart};
use mlx_gen_mage::MageFlowPipeline;
use mlx_rs::Dtype;

fn snapshot() -> String {
    std::env::var("MAGE_SNAPSHOT").expect("set MAGE_SNAPSHOT to a complete Mage-Flow snapshot")
}

fn generate_pixels(generator: &dyn mlx_gen::Generator, prompt: &str, seed: u64) -> Vec<u8> {
    let request = GenerationRequest {
        prompt: prompt.into(),
        width: 512,
        height: 512,
        steps: Some(2),
        guidance: Some(5.0),
        seed: Some(seed),
        ..Default::default()
    };
    let GenerationOutput::Images(images) = generator.generate(&request, &mut |_| {}).unwrap()
    else {
        panic!("returned non-image output");
    };
    images.into_iter().next().unwrap().pixels
}

fn pixel_mae(a: &[u8], b: &[u8]) -> f64 {
    a.iter()
        .zip(b)
        .map(|(&x, &y)| (x as f64 - y as f64).abs())
        .sum::<f64>()
        / a.len() as f64
}

fn image_structure(pixels: &[u8]) -> (u8, u8, f64, usize) {
    let min = *pixels.iter().min().unwrap();
    let max = *pixels.iter().max().unwrap();
    let mean = pixels.iter().map(|&x| x as f64).sum::<f64>() / pixels.len() as f64;
    let variance = pixels
        .iter()
        .map(|&x| {
            let d = x as f64 - mean;
            d * d
        })
        .sum::<f64>()
        / pixels.len() as f64;
    let row_bytes = 512 * 3;
    let repeated_rows = pixels
        .chunks_exact(row_bytes)
        .collect::<Vec<_>>()
        .windows(2)
        .filter(|pair| pair[0] == pair[1])
        .count();
    (min, max, variance.sqrt(), repeated_rows)
}

fn cosine(a: &[f32], b: &[f32]) -> f64 {
    let dot = a
        .iter()
        .zip(b)
        .map(|(&x, &y)| x as f64 * y as f64)
        .sum::<f64>();
    let an = a.iter().map(|&x| (x as f64).powi(2)).sum::<f64>().sqrt();
    let bn = b.iter().map(|&x| (x as f64).powi(2)).sum::<f64>().sqrt();
    dot / (an * bn).max(1e-12)
}

fn pairwise_similarities(vectors: &[Vec<f32>]) -> Vec<f64> {
    let mut out = Vec::new();
    for i in 0..vectors.len() {
        for j in i + 1..vectors.len() {
            out.push(cosine(&vectors[i], &vectors[j]));
        }
    }
    out
}

fn ranking_agreement(a: &[f64], b: &[f64]) -> f64 {
    let mut agree = 0;
    let mut total = 0;
    for i in 0..a.len() {
        for j in i + 1..a.len() {
            agree += usize::from((a[i] - a[j]).signum() == (b[i] - b[j]).signum());
            total += 1;
        }
    }
    agree as f64 / total as f64
}

#[test]
#[ignore = "needs full real weights and an authorized Metal device"]
fn qwen_q8_beats_q4_and_both_preserve_prompt_fidelity() {
    let prompts = [
        "a precise technical illustration of a glass greenhouse at dusk",
        "a watercolor portrait of an astronaut holding yellow flowers",
        "an aerial photograph of a snowy mountain village",
        "a minimalist red chair on a white studio background",
    ];
    let encode = |bits| {
        let pipeline = MageFlowPipeline::load_with_quant(snapshot(), bits).unwrap();
        prompts
            .iter()
            .map(|prompt| {
                let txt = pipeline
                    .text_encoder
                    .encode(&[prompt], PromptKind::Gen)
                    .unwrap()
                    .txt
                    .as_dtype(Dtype::Float32)
                    .unwrap();
                mlx_rs::transforms::eval([&txt]).unwrap();
                let shape = txt.shape();
                let hidden = shape[1] as usize;
                let rows = shape[0] as usize;
                let values = txt.as_slice::<f32>();
                (0..hidden)
                    .map(|column| {
                        (0..rows)
                            .map(|row| values[row * hidden + column])
                            .sum::<f32>()
                            / rows as f32
                    })
                    .collect::<Vec<_>>()
            })
            .collect::<Vec<_>>()
    };
    let dense = encode(None);
    mlx_rs::memory::clear_cache();
    let q8 = encode(Some(8));
    mlx_rs::memory::clear_cache();
    let q4 = encode(Some(4));
    let flatten = |vectors: &[Vec<f32>]| vectors.concat();
    let dense_flat = flatten(&dense);
    let q8_flat = flatten(&q8);
    let q4_flat = flatten(&q4);
    let rel = |got: &[f32], want: &[f32]| {
        let diff = got
            .iter()
            .zip(want)
            .map(|(&x, &y)| (x - y).abs())
            .sum::<f32>()
            / got.len() as f32;
        let reference = want.iter().map(|x| x.abs()).sum::<f32>() / want.len() as f32;
        diff / reference.max(1e-12)
    };
    let q8_error = rel(&q8_flat, &dense_flat);
    let q4_error = rel(&q4_flat, &dense_flat);
    let q8_cos = q8
        .iter()
        .zip(&dense)
        .map(|(a, b)| cosine(a, b))
        .sum::<f64>()
        / prompts.len() as f64;
    let q4_cos = q4
        .iter()
        .zip(&dense)
        .map(|(a, b)| cosine(a, b))
        .sum::<f64>()
        / prompts.len() as f64;
    let wrong_cos = q4
        .iter()
        .enumerate()
        .map(|(i, vector)| cosine(vector, &dense[(i + 1) % dense.len()]))
        .sum::<f64>()
        / prompts.len() as f64;
    let dense_ranks = pairwise_similarities(&dense);
    let q8_rank = ranking_agreement(&dense_ranks, &pairwise_similarities(&q8));
    let q4_rank = ranking_agreement(&dense_ranks, &pairwise_similarities(&q4));
    println!(
        "Qwen3-VL multi-prompt fidelity: q8_rel={q8_error:.6}, q4_rel={q4_error:.6}, \
         q8_cos={q8_cos:.6}, q4_cos={q4_cos:.6}, wrong_prompt_cos={wrong_cos:.6}, \
         q8_rank={q8_rank:.3}, q4_rank={q4_rank:.3}"
    );
    assert!(
        q8_error < q4_error,
        "Q8 must retain more text fidelity than Q4"
    );
    assert!(q8_error <= 0.04, "Q8 quality-floor text fidelity exceeded");
    assert!(q4_error <= 0.35, "Q4 low-memory usability ceiling exceeded");
    assert!(q8_cos >= 0.995, "Q8 cosine quality floor regressed");
    assert!(q4_cos >= 0.96, "Q4 lost prompt-vector identity");
    assert!(
        q4_cos >= wrong_cos + 0.05,
        "Q4 no longer separates matched prompts from the wrong-prompt mutation"
    );
    assert!(q8_rank >= 0.90, "Q8 prompt-similarity ranking regressed");
    assert!(q4_rank >= 0.90, "Q4 prompt-similarity ranking is unusable");
}

#[test]
#[ignore = "needs full real weights and an authorized Metal device"]
fn all_three_tiers_generate_and_round_trip_mean_latents() {
    let mut tier_results = Vec::new();
    for tier in [None, Some(Quant::Q8), Some(Quant::Q4)] {
        mlx_rs::memory::clear_cache();
        mlx_rs::memory::reset_peak_memory();
        let mut spec = LoadSpec::new(WeightsSource::Dir(snapshot().into()));
        spec.quantize = tier;
        let generator = (REGISTRATION_BASE.load)(&spec).unwrap();
        let primary = generate_pixels(generator.as_ref(), "a red fox beneath a pine tree", 14046);
        let prompt_changed =
            generate_pixels(generator.as_ref(), "a blue sailboat on a stormy sea", 14046);
        let seed_changed =
            generate_pixels(generator.as_ref(), "a red fox beneath a pine tree", 14047);
        assert_eq!(primary.len(), 512 * 512 * 3);
        let (min, max, stddev, repeated_rows) = image_structure(&primary);
        let prompt_mae = pixel_mae(&primary, &prompt_changed);
        let seed_mae = pixel_mae(&primary, &seed_changed);
        println!(
            "tier {tier:?}: range={min}..{max}, stddev={stddev:.3}, repeated_rows={repeated_rows}, \
             prompt_mae={prompt_mae:.3}, seed_mae={seed_mae:.3}"
        );
        assert!(
            max.saturating_sub(min) >= 128,
            "tier {tier:?} collapsed dynamic range"
        );
        assert!(stddev >= 20.0, "tier {tier:?} image contrast collapsed");
        assert_eq!(repeated_rows, 0, "tier {tier:?} repeated adjacent rows");
        assert!(prompt_mae >= 8.0, "tier {tier:?} ignores prompt changes");
        assert!(seed_mae >= 8.0, "tier {tier:?} ignores seed changes");

        let mut codec = vae::load(snapshot(), VaePart::Both, Dtype::Bfloat16).unwrap();
        if let Some(quant) = tier {
            codec.quantize(quant.bits()).unwrap();
        }
        let pixels = mlx_rs::Array::from_slice(
            &primary
                .iter()
                .map(|&value| value as f32 / 127.5 - 1.0)
                .collect::<Vec<_>>(),
            &[1, 512, 512, 3],
        )
        .transpose_axes(&[0, 3, 1, 2])
        .unwrap();
        let mean = codec.encode_mean(&pixels).unwrap();
        let decoded = codec.decode(&mean).unwrap();
        mlx_rs::transforms::eval([&mean, &decoded]).unwrap();
        assert_eq!(mean.shape(), [1, 128, 32, 32]);
        assert!(
            decoded
                .as_dtype(Dtype::Float32)
                .unwrap()
                .as_slice::<f32>()
                .iter()
                .all(|value| value.is_finite()),
            "tier {tier:?} round trip produced non-finite pixels"
        );
        let peak_gb = mlx_rs::memory::get_peak_memory() as f64 / 1e9;
        println!("tier {tier:?}: generation + mean-latent roundtrip MLX peak {peak_gb:.3} GB");
        let (low, high) = match tier {
            None => (16.0, 20.0),
            Some(Quant::Q8) => (8.5, 12.0),
            Some(Quant::Q4) => (5.0, 7.0),
            Some(Quant::Nvfp4) => unreachable!(),
        };
        assert!(
            (low..=high).contains(&peak_gb),
            "tier {tier:?} peak {peak_gb:.3} GB escaped calibrated band {low}..={high}"
        );
        tier_results.push((tier, primary, prompt_mae, seed_mae, peak_gb));
    }
    let dense = &tier_results[0];
    let q8 = &tier_results[1];
    let q4 = &tier_results[2];
    assert!(q4.4 + 2.0 <= q8.4 && q8.4 + 4.0 <= dense.4);
    assert!(
        q4.2 >= dense.2 * 0.25 && q4.3 >= dense.3 * 0.25,
        "Q4 must retain at least 25% of BF16 prompt/seed discrimination"
    );
    let q8_bf16_mae = pixel_mae(&q8.1, &dense.1);
    let q4_bf16_mae = pixel_mae(&q4.1, &dense.1);
    println!("tier similarity to BF16: q8_mae={q8_bf16_mae:.3}, q4_mae={q4_bf16_mae:.3}");
    assert!(
        q8_bf16_mae < q4_bf16_mae,
        "Q8 output must remain closer to BF16 than Q4"
    );
    assert!(
        q8_bf16_mae <= 15.0,
        "Q8 output escaped the BF16 quality-floor oracle"
    );
    assert!(
        q4_bf16_mae <= 85.0,
        "Q4 output departed too far from the BF16 quality oracle"
    );
}
