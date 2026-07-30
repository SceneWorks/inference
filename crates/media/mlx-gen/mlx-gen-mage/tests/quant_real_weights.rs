//! Actual BF16/Q8/Q4 generation, text-fidelity, and VAE round-trip acceptance.
//!
//! Text-tier decision (sc-14046, **revised by sc-15071**): Q8 is the quality floor for
//! fidelity-sensitive use (four-prompt pooled-vector relative error `<= 0.04`, cosine `>= 0.995`).
//! The original conclusion — that Q4 text was a usable "explicitly low-fidelity" tier because it
//! preserved prompt identity and ranking — was drawn from prompt vectors alone, against a dense
//! DiT. It does not survive rendering: a Q4 text encoder paired with a Q4 DiT does not produce the
//! prompt at all. The Q4 tier therefore now shares Q8's LM decoder layers
//! (`crate::quant::LM_LAYER_MIN_BITS`), and the four-prompt fidelity numbers below improved
//! accordingly (Q4 relative error 0.35-ceiling → 0.035). Only the token embedding and the vision
//! tower still take Q4's own width.

use mlx_gen::quant::{load_dir_map, quantize_map};
use mlx_gen::weights::Weights;
use mlx_gen::{GenerationOutput, GenerationRequest, LoadSpec, Quant, WeightsSource};
use mlx_gen_mage::config::{TE_RMS_NORM_EPS, TE_ROPE_THETA};
use mlx_gen_mage::convert::{is_dit_target, is_te_target};
use mlx_gen_mage::model::REGISTRATION_BASE;
use mlx_gen_mage::text_encoder::{
    load_tokenizer_dir, verify_text_config, PromptKind, Qwen3VlTextEncoder, LM_PREFIX,
};
use mlx_gen_mage::vae::{self, VaePart};
use mlx_gen_mage::{GsKey, MageFlowConfig, MageFlowPipeline, MageTextEncoder, MageTransformer};
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

// =================================================================================================
// sc-15071 — the STRUCTURAL tier gate.
//
// The numeric bands below (`all_three_tiers_generate_and_round_trip_mean_latents`) are a liveness
// probe, not a quality gate: the Q4 tier that shipped rendered a repeating blue tiled texture with
// no cube and no table, and it passed every one of them — std ~64, full dynamic range, zero
// repeated rows, prompt/seed MAE well clear of their floors, and a BF16 MAE inside the 85.0
// tolerance. Numbers moving is not the same as the image being the prompt.
//
// This gate compares each tier against the BF16 render **of the same prompt at the same seed**.
// Fixed seed makes the tiers spatially aligned — they integrate the same trajectory from the same
// noise — so a plain correlation of the two images is a direct measure of "is this the same
// scene", and it collapses for any semantically wrong output, tiled or otherwise. It is not a
// tiling detector; a detector for the one failure mode already observed would be the same mistake
// one level up.
//
// The gate is proven by MUTATION, not by assertion: the test rebuilds each component the way it was
// packed before sc-15071 (uniform Q4 through `quantize_map` + the real packed loader, i.e. the
// precision floor defeated) and requires each to FAIL the same floors the real tiers pass. Both
// floors are covered, because Q4 needed both to be correct.
// =================================================================================================

/// The story's repro prompt/geometry, verbatim.
const SCENE_PROMPT: &str = "a red cube on a white table, studio lighting";
const SCENE_SEED: i64 = 42;
const SCENE_STEPS: usize = 30;
const SCENE_CFG: f32 = 5.0;
const SCENE_SIZE: u32 = 512;

/// Luma correlation each tier must reach against the BF16 render.
///
/// **Measured** at the constants above: BF16 1.000 (identity), Q8 **0.9973**, Q4 **0.8770** —
/// against **0.3205** with the DiT head-modulation floor defeated, **0.1628** with the text-encoder
/// floor defeated, and **0.0543** for the uniformly-Q4 pipeline that shipped. The floor sits in the
/// wide empty gap between 0.877 and 0.320, fitted tightly to neither side.
const SCENE_CORR_FLOOR: f64 = 0.70;

/// The same floor for the coarse 16×16 block-mean image — scene *layout* agreement, insensitive to
/// the fine-texture differences a lower-bit tier legitimately has. **Measured**: Q8 0.9982,
/// Q4 0.8897, against 0.4193 / 0.2363 for the two mutations.
const SCENE_BLOCK_CORR_FLOOR: f64 = 0.75;

fn render_scene(pipeline: &MageFlowPipeline) -> Vec<u8> {
    let image = pipeline
        .generate(
            SCENE_PROMPT,
            " ",
            SCENE_SIZE,
            SCENE_SIZE,
            SCENE_STEPS,
            SCENE_CFG,
            SCENE_SEED,
            &GsKey::default(),
            false,
        )
        .unwrap();
    mlx_rs::transforms::eval([&image]).unwrap();
    image.as_slice::<u8>().to_vec()
}

fn luma(pixels: &[u8]) -> Vec<f64> {
    pixels
        .chunks_exact(3)
        .map(|c| 0.299 * c[0] as f64 + 0.587 * c[1] as f64 + 0.114 * c[2] as f64)
        .collect()
}

/// Pearson correlation. `1.0` is the same image; `~0` is an unrelated one. Scale/offset invariant,
/// so a tier that is merely darker or lower-contrast than BF16 is not punished — only one that
/// puts different content in different places.
fn correlation(a: &[f64], b: &[f64]) -> f64 {
    let mean = |v: &[f64]| v.iter().sum::<f64>() / v.len() as f64;
    let (ma, mb) = (mean(a), mean(b));
    let (mut num, mut da, mut db) = (0.0, 0.0, 0.0);
    for (x, y) in a.iter().zip(b) {
        let (u, v) = (x - ma, y - mb);
        num += u * v;
        da += u * u;
        db += v * v;
    }
    num / (da.sqrt() * db.sqrt()).max(1e-12)
}

/// Mean luma over 16×16 pixel blocks — the scene's coarse layout.
fn block_means(pixels: &[u8]) -> Vec<f64> {
    const BLOCK: usize = 16;
    let n = SCENE_SIZE as usize;
    let l = luma(pixels);
    let mut out = Vec::with_capacity((n / BLOCK) * (n / BLOCK));
    for by in 0..n / BLOCK {
        for bx in 0..n / BLOCK {
            let mut sum = 0.0;
            for y in 0..BLOCK {
                for x in 0..BLOCK {
                    sum += l[(by * BLOCK + y) * n + bx * BLOCK + x];
                }
            }
            out.push(sum / (BLOCK * BLOCK) as f64);
        }
    }
    out
}

/// `(luma_corr, block_corr, mae)` of `got` against the BF16 reference.
fn scene_scores(got: &[u8], reference: &[u8]) -> (f64, f64, f64) {
    (
        correlation(&luma(got), &luma(reference)),
        correlation(&block_means(got), &block_means(reference)),
        pixel_mae(got, reference),
    )
}

/// Rebuild the DiT exactly as it was packed before sc-15071: `quantize_map` at a uniform width over
/// all 174 targets — the 8-bit floor on `norm_out.linear` defeated — loaded back through the real
/// packed loader (`{base}.scales` auto-detection). This is the shipped artifact, not a simulation
/// of it, which is what makes it usable as proof that the gate catches the real defect.
fn dit_packed_at_uniform_bits(bits: i32) -> MageTransformer {
    let dir = std::path::Path::new(&snapshot()).join("transformer");
    let cfg = MageFlowConfig::from_transformer_config_json(
        &std::fs::read_to_string(dir.join("config.json")).unwrap(),
    )
    .unwrap();
    let map = quantize_map(load_dir_map(&dir).unwrap(), bits, 64, is_dit_target).unwrap();
    let mut w = Weights::empty();
    for (key, value) in map {
        w.insert(&key, value);
    }
    MageTransformer::from_weights(&w, cfg).unwrap()
}

/// The text-encoder counterpart: every one of the 253 targets packed at a uniform width, i.e. the
/// `LM_LAYER_MIN_BITS` floor defeated, loaded back through the real packed loader.
fn text_encoder_packed_at_uniform_bits(bits: i32) -> MageTextEncoder {
    let dir = std::path::Path::new(&snapshot()).join("text_encoder");
    let cfg =
        verify_text_config(&std::fs::read_to_string(dir.join("config.json")).unwrap()).unwrap();
    let map = quantize_map(load_dir_map(&dir).unwrap(), bits, 64, is_te_target).unwrap();
    let mut w = Weights::empty();
    for (key, value) in map {
        w.insert(&key, value);
    }
    let lm = Qwen3VlTextEncoder::from_weights(&w, LM_PREFIX, &cfg, TE_RMS_NORM_EPS, TE_ROPE_THETA)
        .unwrap();
    MageTextEncoder::new(load_tokenizer_dir(&dir).unwrap(), lm)
}

#[test]
#[ignore = "needs full real weights and an authorized Metal device"]
fn every_tier_renders_the_reference_scene_and_a_uniform_q4_head_fails_the_gate() {
    let reference = {
        let pipeline = MageFlowPipeline::load_with_quant(snapshot(), None).unwrap();
        render_scene(&pipeline)
    };
    mlx_rs::memory::clear_cache();

    let mut scores = Vec::new();
    for bits in [8, 4] {
        let pipeline = MageFlowPipeline::load_with_quant(snapshot(), Some(bits)).unwrap();
        let (corr, block, mae) = scene_scores(&render_scene(&pipeline), &reference);
        println!("Q{bits}: luma_corr={corr:.4} block_corr={block:.4} mae_vs_bf16={mae:.3}");
        assert!(
            corr >= SCENE_CORR_FLOOR,
            "Q{bits} did not render the reference scene: luma correlation {corr:.4} < \
             {SCENE_CORR_FLOOR}. The tier is producing a different image, not a lower-fidelity \
             version of the same one."
        );
        assert!(
            block >= SCENE_BLOCK_CORR_FLOOR,
            "Q{bits} scene layout diverged: block correlation {block:.4} < \
             {SCENE_BLOCK_CORR_FLOOR}"
        );
        scores.push((bits, corr, block));
        mlx_rs::memory::clear_cache();
    }
    assert!(
        scores[0].1 > scores[1].1,
        "Q8 must stay closer to BF16 than Q4 does"
    );

    // --- the mutations: each precision floor defeated in turn, through the real packer -----------
    // Both floors are load-bearing, and a floor nobody has tried to remove is not evidence that it
    // was needed. Each mutation rebuilds the component the way it was packed before sc-15071.
    let mut mutants = Vec::new();
    for (what, mutate) in [
        (
            "DiT head modulation (norm_out.linear) at Q4",
            0, // 0 = mutate the transformer
        ),
        ("text-encoder LM decoder layers at Q4", 1),
    ] {
        let mut pipeline = MageFlowPipeline::load_with_quant(snapshot(), Some(4)).unwrap();
        if mutate == 0 {
            pipeline.transformer = dit_packed_at_uniform_bits(4);
        } else {
            pipeline.text_encoder = text_encoder_packed_at_uniform_bits(4);
        }
        let mutant = render_scene(&pipeline);
        let (corr, block, mae) = scene_scores(&mutant, &reference);
        println!("mutation [{what}]: luma_corr={corr:.4} block_corr={block:.4} mae={mae:.3}");
        assert!(
            corr < SCENE_CORR_FLOOR && block < SCENE_BLOCK_CORR_FLOOR,
            "MUTATION SURVIVED: {what} scored luma {corr:.4} / block {block:.4}, which still \
             passes this gate. That is the defect sc-15071 reported, so a gate it passes is not a \
             gate. Retune SCENE_CORR_FLOOR / SCENE_BLOCK_CORR_FLOOR against the measurements in \
             the module comment — do not relax them."
        );
        mutants.push((what, mutant, mae));
        drop(pipeline);
        mlx_rs::memory::clear_cache();
    }

    // The mutations must also stay invisible to the *liveness* bands below, or this gate would be
    // claiming credit for a catch the cheaper assertions already made. A wrong image with full
    // dynamic range, healthy contrast and no repeated rows is exactly what shipped.
    //
    // The BF16-MAE band is deliberately NOT asserted here: it is regime-dependent, not a property
    // of the image being right. At these 30 steps the two mutations land at 78.9 and 88.7 — one
    // inside the old 85.0 tolerance and one just outside it — while the tier that renders correctly
    // sits at 28.9. A threshold that a correct render and a tiled render straddle by 10 units in
    // either direction is a coin flip, which is why a distance-to-BF16 scalar was never going to be
    // the gate and correlation is.
    for (what, mutant, mae) in &mutants {
        let (min, max, stddev, repeated_rows) = image_structure(mutant);
        println!(
            "[{what}] under the OLD liveness bands: range={min}..{max} stddev={stddev:.3} \
             repeated_rows={repeated_rows} (mae {mae:.3}, unasserted)"
        );
        assert!(
            max.saturating_sub(min) >= 128 && stddev >= 20.0 && repeated_rows == 0,
            "[{what}] no longer passes the old dynamic-range / stddev / repeated-row bands, so it \
             no longer demonstrates why those bands were insufficient. Re-derive the claim in the \
             module comment rather than deleting this assertion."
        );
    }
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
        // sc-15071 moved the Q4 band up from 5.0..=7.0 (measured 7.868, was 5.940): the precision
        // floors that make Q4 actually render the prompt put the Qwen3-VL LM's 36 decoder layers
        // at 8 bits. The old band described a tier that produced a tiled texture, so it was never
        // the Q4 tier's real cost. `crate::memory::calibration_peak_gb` carries the same anchor and
        // has to move with it, or the fit gate under-reports and MLX exits the process.
        let (low, high) = match tier {
            None => (16.0, 20.0),
            Some(Quant::Q8) => (8.5, 12.0),
            Some(Quant::Q4) => (7.0, 9.0),
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
    // The Q4↔Q8 gap narrowed from ~4.1 GB to ~2.2 GB in sc-15071: Q4 now shares Q8's LM decoder
    // layers, so only the DiT, the vision-tower projections and the token embedding still differ.
    // The tiers must stay ordered and materially apart — 1.5 GB, measured 2.25 — but the old 2.0 GB
    // separation is no longer a property of a *correct* Q4 tier.
    assert!(
        q4.4 + 1.5 <= q8.4 && q8.4 + 4.0 <= dense.4,
        "tier peaks lost their separation: q4={:.3} q8={:.3} bf16={:.3} GB",
        q4.4,
        q8.4,
        dense.4
    );
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
    // Tightened from 85.0 (sc-15071). 85.0 was wide enough to admit the tiled render this whole
    // file failed to catch; the corrected tier measures 34.98 here, so 50.0 keeps ~40% headroom
    // while no longer being a tolerance a semantically wrong image can sit inside. It is still only
    // a coarse liveness bound — `every_tier_renders_the_reference_scene_and_a_uniform_q4_head_fails_the_gate`
    // is the probe that actually checks the image is the prompt.
    assert!(
        q4_bf16_mae <= 50.0,
        "Q4 output departed too far from the BF16 quality oracle"
    );
}
