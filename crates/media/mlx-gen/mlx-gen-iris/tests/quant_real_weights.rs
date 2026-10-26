//! sc-26266 real-weight gates for the `q8` / `q4` tiers against the pinned `speridlabs/iris-3b` +
//! `Qwen/Qwen3-VL-4B-Instruct` snapshots. `#[ignore]`d: they need ~21 GB of weights and the GPU.
//! Inputs come from the environment, like `real_weights`:
//!
//! * `IRIS_WEIGHTS_DIR` — the `speridlabs/iris-3b` snapshot root (generation backbone).
//! * `IRIS_TEXT_ENCODER_DIR` — the `Qwen/Qwen3-VL-4B-Instruct` snapshot.
//! * `IRIS_OUT` — a directory the tier renders write their PNGs into (render gate only).
//! * `IRIS_QUANT_STEPS` — solver steps of the render gate (default 30).
//!
//! Run each under an external memory guard (one at a time, `--test-threads 1`):
//! `cargo test -p mlx-gen-iris --release --test integration -- --ignored quant_real_weights:: --test-threads 1`

use std::path::PathBuf;

use mlx_gen::gen_core::iris::{IrisConfig, IrisTask, TEXT_ENCODER_COMPONENT};
use mlx_gen::{GenerationOutput, GenerationRequest, LoadSpec, Progress, Quant, WeightsSource};
use mlx_gen_iris::{load_backbone_quantized, IrisTextEncoder, MODEL_ID};
use mlx_rs::{Array, Dtype};

fn env_dir(name: &str) -> PathBuf {
    PathBuf::from(std::env::var(name).unwrap_or_else(|_| panic!("{name} is required")))
}

const TIERS: [Option<Quant>; 3] = [None, Some(Quant::Q8), Some(Quant::Q4)];

/// `measured / dense` must fall in `[lo, hi]`.
fn assert_ratio(what: &str, measured: u64, dense: u64, lo: f64, hi: f64) {
    let ratio = measured as f64 / dense as f64;
    eprintln!(
        "[[IRIS_QUANT]] {what}: {:.3} GiB = {ratio:.3} x bf16",
        measured as f64 / f64::from(1 << 30)
    );
    assert!(
        (lo..=hi).contains(&ratio),
        "{what}: {ratio:.3} x bf16 is outside [{lo}, {hi}]"
    );
}

/// The resident weight bytes of each component at each tier, measured from the LOADED model
/// (every parameter array's `nbytes`), one component resident at a time.
///
/// Declared bounds, from the release's tensor headers (sc-26266): the backbone's projections are
/// ~all of its bytes, so it lands at 0.532 x (q8) / 0.282 x (q4) of bf16 as priced; the tower keeps
/// its 0.78 GB tied embedding bf16, so it lands at 0.577 x / 0.351 x; the pair at 0.558 x /
/// 0.322 x. The loader holds biases at bf16 where the pricing charges f32, so the measured dense
/// denominator is a little smaller — the windows below allow for that, and still separate the
/// tiers from each other and from a dense load.
#[test]
#[ignore = "needs the real weights (IRIS_WEIGHTS_DIR, IRIS_TEXT_ENCODER_DIR) and the GPU"]
fn real_tiers_shrink_the_resident_weights() {
    let backbone = env_dir("IRIS_WEIGHTS_DIR");
    let config = IrisConfig::from_dir(&backbone).unwrap();
    let tower = env_dir("IRIS_TEXT_ENCODER_DIR");

    let mut dit_bytes = Vec::new();
    for quant in TIERS {
        let start = std::time::Instant::now();
        let (dit, _) = load_backbone_quantized(
            &backbone,
            &config,
            Dtype::Bfloat16,
            &[],
            IrisTask::Generation,
            MODEL_ID,
            quant,
        )
        .unwrap();
        dit_bytes.push(dit.resident_weight_bytes());
        #[cfg(target_os = "macos")]
        crate::common::metal_guard::record_model_inference_device();
        eprintln!(
            "[[IRIS_QUANT]] backbone {quant:?} load {:.1}s",
            start.elapsed().as_secs_f32()
        );
        drop(dit);
        mlx_rs::memory::clear_cache();
    }
    let mut te_bytes = Vec::new();
    for quant in TIERS {
        let start = std::time::Instant::now();
        let te = IrisTextEncoder::load_quantized(&tower, &config.text_encoder, quant).unwrap();
        assert_eq!(te.is_quantized(), quant.is_some());
        te_bytes.push(te.resident_weight_bytes());
        #[cfg(target_os = "macos")]
        crate::common::metal_guard::record_model_inference_device();
        eprintln!(
            "[[IRIS_QUANT]] text encoder {quant:?} load {:.1}s",
            start.elapsed().as_secs_f32()
        );
        drop(te);
        mlx_rs::memory::clear_cache();
    }
    assert_ratio("backbone q8", dit_bytes[1], dit_bytes[0], 0.45, 0.57);
    assert_ratio("backbone q4", dit_bytes[2], dit_bytes[0], 0.22, 0.31);
    assert_ratio("text encoder q8", te_bytes[1], te_bytes[0], 0.50, 0.60);
    assert_ratio("text encoder q4", te_bytes[2], te_bytes[0], 0.28, 0.37);
    let pair = |i: usize| dit_bytes[i] + te_bytes[i];
    assert_ratio("backbone + text encoder q8", pair(1), pair(0), 0.48, 0.58);
    assert_ratio("backbone + text encoder q4", pair(2), pair(0), 0.25, 0.34);

    #[cfg(feature = "model-inference-proof")]
    eprintln!("\nMODEL_INFERENCE_BODY_COMPLETED quant_real_weights::real_tiers_shrink_the_resident_weights");
}

/// Luminance of an RGB8 image box-averaged to a `cells x cells` grid.
fn thumbnail(img: &mlx_gen::Image, cells: u32) -> Vec<f64> {
    let (cw, ch) = (img.width / cells, img.height / cells);
    let mut out = vec![0f64; (cells * cells) as usize];
    for y in 0..cells * ch {
        for x in 0..cells * cw {
            let i = ((y * img.width + x) * 3) as usize;
            let p = &img.pixels[i..i + 3];
            let luma = 0.299 * f64::from(p[0]) + 0.587 * f64::from(p[1]) + 0.114 * f64::from(p[2]);
            out[((y / ch) * cells + x / cw) as usize] += luma / f64::from(cw * ch);
        }
    }
    out
}

fn pearson(a: &[f64], b: &[f64]) -> f64 {
    let n = a.len() as f64;
    let (ma, mb) = (a.iter().sum::<f64>() / n, b.iter().sum::<f64>() / n);
    let cov: f64 = a.iter().zip(b).map(|(x, y)| (x - ma) * (y - mb)).sum();
    let va: f64 = a.iter().map(|x| (x - ma).powi(2)).sum();
    let vb: f64 = b.iter().map(|y| (y - mb).powi(2)).sum();
    cov / (va * vb).sqrt()
}

/// The production path at 512² from a fixed seed at bf16, q8 and q4 (catalog load, default
/// residency, CFG 3, `IRIS_QUANT_STEPS` steps): each render is coherent and keeps the bf16
/// render's composition. The three PNGs land in `IRIS_OUT` for a visual check. This is the coarse
/// image-level sanity gate; what the tiers' quantization controls is gated directly by
/// [`real_packed_projections_sit_on_their_stored_grid`] (weights) and
/// [`real_quantized_tiers_track_the_bf16_denoiser_at_512`] (one denoiser step, conditioning).
///
/// * coherent — mean RGB8 value in (5, 250) and luminance standard deviation ≥ 12 (not flat);
/// * composition — Pearson correlation of the 32 x 32 box-averaged luminance against bf16 ≥ 0.85
///   at q8 and ≥ 0.70 at q4 (first GPU run, 30 steps: 0.953 / 0.864).
///
/// **No pixel-distance bound.** The pre-measurement declaration also bounded mean |Δ| against
/// bf16 (≤ 12 / 255 at q8, ≤ 28 at q4); the first real-weight run measured 16.4 / 33.4 with both
/// renders coherent, and the bound measured the sampler, not the tier. On the release (MLX CPU,
/// 512², CFG 3, one solver step at s = 1.0 and s = 0.8): a 0.086 % relative Gaussian
/// perturbation of every backbone weight — a tenth of the q8 rounding — already moves the
/// CFG-combined velocity by 7.3 / 6.6 %, and bf16-vs-f32 compute of the dense tier by 7.6 / 5.8 %,
/// while the whole q8 backbone moves it 7.6 / 8.9 %. Thirty such steps of a deterministic solver
/// amplify any perturbation at the bf16 floor to a different fine-detail trajectory, so mean
/// |Δ| after 30 steps cannot separate a correct q8 tier from the dense tier's own precision
/// noise; the measured mean |Δ| is reported for the record only.
#[test]
#[ignore = "needs the real weights (IRIS_WEIGHTS_DIR, IRIS_TEXT_ENCODER_DIR), IRIS_OUT and the GPU"]
fn real_quantized_renders_track_the_bf16_render_at_512() {
    let steps: u32 = std::env::var("IRIS_QUANT_STEPS")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(30);
    let out = env_dir("IRIS_OUT");
    let registry = mlx_gen_iris::provider_registry().unwrap();
    let req = GenerationRequest {
        prompt: "a red fox sleeping in fresh snow, golden hour".into(),
        width: 512,
        height: 512,
        steps: Some(steps),
        seed: Some(26266),
        ..Default::default()
    };
    let mut renders = Vec::new();
    for quant in TIERS {
        let mut spec = LoadSpec::new(WeightsSource::Dir(env_dir("IRIS_WEIGHTS_DIR")));
        spec.quantize = quant;
        spec.components.insert(
            TEXT_ENCODER_COMPONENT.into(),
            WeightsSource::Dir(env_dir("IRIS_TEXT_ENCODER_DIR")),
        );
        let start = std::time::Instant::now();
        let generator = registry.load(MODEL_ID, &spec).unwrap();
        eprintln!(
            "[[IRIS_QUANT]] {quant:?} load {:.1}s",
            start.elapsed().as_secs_f32()
        );
        let start = std::time::Instant::now();
        let mut last = (0, 0);
        let GenerationOutput::Images(mut images) = generator
            .generate(&req, &mut |p| {
                if let Progress::Step { current, total } = p {
                    last = (current, total);
                }
            })
            .unwrap()
        else {
            panic!("expected images")
        };
        eprintln!(
            "[[IRIS_QUANT]] {quant:?} 512² x {steps} steps: {:.1}s",
            start.elapsed().as_secs_f32()
        );
        assert_eq!(last, (steps, steps));
        drop(generator);
        mlx_rs::memory::clear_cache();
        let img = images.remove(0);
        let tier = quant.map_or("bf16".to_owned(), |q| format!("{q:?}").to_lowercase());
        let path = out.join(format!("iris_3b_{tier}_512.png"));
        image::RgbImage::from_raw(img.width, img.height, img.pixels.clone())
            .unwrap()
            .save(&path)
            .unwrap();
        let mean = img.pixels.iter().map(|&p| f64::from(p)).sum::<f64>() / img.pixels.len() as f64;
        let luma = thumbnail(&img, 512);
        let lm = luma.iter().sum::<f64>() / luma.len() as f64;
        let std = (luma.iter().map(|v| (v - lm).powi(2)).sum::<f64>() / luma.len() as f64).sqrt();
        eprintln!(
            "[[IRIS_QUANT]] {tier}: wrote {} (mean {mean:.1}, luma std {std:.1})",
            path.display()
        );
        assert!(
            mean > 5.0 && mean < 250.0,
            "{tier}: degenerate image (mean {mean})"
        );
        assert!(std >= 12.0, "{tier}: a flat image (luma std {std})");
        renders.push(img);
    }
    let reference = thumbnail(&renders[0], 32);
    for (tier, img, min_corr) in [("q8", &renders[1], 0.85), ("q4", &renders[2], 0.70)] {
        let corr = pearson(&thumbnail(img, 32), &reference);
        let mean = img
            .pixels
            .iter()
            .zip(&renders[0].pixels)
            .map(|(a, b)| (f64::from(*a) - f64::from(*b)).abs())
            .sum::<f64>()
            / img.pixels.len() as f64;
        eprintln!(
            "[[IRIS_QUANT]] {tier} vs bf16: thumbnail corr {corr:.4}, mean |Δ| {mean:.2} (reported)"
        );
        assert!(
            corr >= min_corr,
            "{tier}: composition corr {corr} < {min_corr}"
        );
    }

    #[cfg(target_os = "macos")]
    crate::common::metal_guard::record_model_inference_device();
    eprintln!("\nMODEL_INFERENCE_BODY_COMPLETED quant_real_weights::real_quantized_renders_track_the_bf16_render_at_512");
}

/// Every projection the release's q8 / q4 tiers pack, through the production quantizers — the
/// backbone's `nn::Loader::linear` and the tower's `Projection::load` under
/// `text_encoder_quant_spec` — is packed (the set the loaders pack: every backbone projection
/// at least one 64-wide group wide, every tower attention / MLP projection — the candle twin's
/// `ProjectionCensus` set) at the tier's width and group, and its dequantized weight is within
/// 1.05 x of the f32-grid optimum for that width and group. CPU only (a CPU child with Metal
/// trapped); no forward runs, so nothing here goes through the sampler's trajectory.
///
/// Measured (sc-26266, MLX CPU stream, relative Frobenius error against the bf16 weight): MLX's
/// own `quantize` packs the q8 projections at 0.83–1.0 % (backbone) / 0.85–1.0 % (tower), ~1.5 x
/// the f32-grid optimum, because it rounds the codes to the f32 grid and then stores bf16 scales
/// and biases; the stored-grid packing the loaders use sits at the optimum (0.55 % on a
/// 2560-wide projection). At q4 both are ~9–10 % (the modulation cores 12–13 %), within noise of
/// the optimum. Through the loaders (304 backbone and 252 tower projections packed at each
/// tier), worst against the optimum: q8 1.040 x (backbone, the 16 x 64 `pixel_blocks.0.mlp.fc2`)
/// / 1.007 x (tower); q4 1.001 x / 1.000 x.
///
/// *Mutations that red this:* packing with MLX's `quantize` (q8 ratio ~1.5); a wrong bit width or
/// group size on any projection (the ratio against the tier's optimum); a projection left dense
/// (`is_quantized`) or a tower projection the spec does not reach.
#[test]
#[ignore = "needs the real weights (IRIS_WEIGHTS_DIR, IRIS_TEXT_ENCODER_DIR); CPU only, ~10 min"]
fn real_packed_projections_sit_on_their_stored_grid() {
    crate::common::cpu_process(
        "quant_real_weights::real_packed_projections_sit_on_their_stored_grid",
        || {
            use mlx_gen_iris::nn::{BackboneQuant, Loader};
            use mlx_llm::primitives::projection::Projection;
            let backbone = env_dir("IRIS_WEIGHTS_DIR").join("model.safetensors");
            let tower = env_dir("IRIS_TEXT_ENCODER_DIR");
            let tower_cfg = mlx_llm::ModelConfig::from_dir(&tower).unwrap();
            for (quant, bits) in [(Quant::Q8, 8), (Quant::Q4, 4)] {
                // Backbone: one projection at a time through the production loader, released as
                // it goes (the f32 checkpoint is ~11 GB).
                let weights = mlx_gen::weights::Weights::from_file(&backbone).unwrap();
                let mut keys: Vec<String> = weights
                    .keys()
                    .filter(|k| k.ends_with(".weight"))
                    .map(str::to_owned)
                    .collect();
                keys.sort();
                let loader = Loader {
                    weights: &weights,
                    compute: Dtype::Bfloat16,
                    adapted: None,
                    quant: Some(BackboneQuant { bits }),
                };
                let (mut packed, mut worst, mut worst_at) = (0, 0f64, String::new());
                for key in &keys {
                    let w = weights.get(key).unwrap();
                    if w.ndim() != 2 || w.shape()[1] < 64 {
                        continue;
                    }
                    let prefix = key.trim_end_matches(".weight");
                    let bias = weights.get(&format!("{prefix}.bias")).is_some();
                    let linear = loader.linear(prefix, bias).unwrap();
                    assert!(linear.is_quantized(), "q{bits}: {prefix} is dense");
                    let arrays = linear.arrays();
                    let bf16 = w.as_dtype(Dtype::Bfloat16).unwrap();
                    let ratio =
                        crate::common::packing_error(&bf16, arrays[0], arrays[1], arrays[2], bits)
                            / crate::common::f32_grid_error(&bf16, bits);
                    if ratio > worst {
                        (worst, worst_at) = (ratio, prefix.to_owned());
                    }
                    packed += 1;
                    drop(linear);
                    mlx_rs::memory::clear_cache();
                }
                eprintln!(
                    "[[IRIS_QUANT]] backbone q{bits}: {packed} projections packed, worst {worst:.4} x \
                     the f32-grid optimum ({worst_at})"
                );
                assert!(packed > 0);
                assert!(
                    worst <= 1.05,
                    "backbone q{bits}: {worst_at} at {worst} x the optimum"
                );
                drop(weights);

                // Tower: every attention / MLP projection through the tier's spec.
                let spec =
                    mlx_gen_iris::text_encoder::text_encoder_quant_spec(&tower_cfg, Some(quant))
                        .unwrap()
                        .unwrap();
                assert_eq!(spec.bits, bits);
                let (mut packed, mut worst, mut worst_at) = (0usize, 0f64, String::new());
                let mut shards: Vec<_> = std::fs::read_dir(&tower)
                    .unwrap()
                    .map(|e| e.unwrap().path())
                    .filter(|p| p.extension().is_some_and(|x| x == "safetensors"))
                    .collect();
                shards.sort();
                for shard in shards {
                    let mut tensors = Array::load_safetensors(&shard).unwrap();
                    let mut names: Vec<String> = tensors.keys().cloned().collect();
                    names.sort();
                    for name in names {
                        let is_projection = name.starts_with("model.language_model.layers.")
                            && name.ends_with("_proj.weight")
                            && (name.contains(".self_attn.") || name.contains(".mlp."));
                        if !is_projection {
                            continue;
                        }
                        let w = tensors
                            .remove(&name)
                            .unwrap()
                            .as_dtype(Dtype::Bfloat16)
                            .unwrap();
                        let Projection::Quantized(q) =
                            Projection::load(w.clone(), Some(spec)).unwrap()
                        else {
                            panic!("q{bits}: {name} is dense under the tier's spec")
                        };
                        assert_eq!((q.bits, q.group_size), (bits, 64), "{name}");
                        let ratio =
                            crate::common::packing_error(&w, &q.weight, &q.scales, &q.biases, bits)
                                / crate::common::f32_grid_error(&w, bits);
                        if ratio > worst {
                            (worst, worst_at) = (ratio, name.clone());
                        }
                        packed += 1;
                        mlx_rs::memory::clear_cache();
                    }
                }
                eprintln!(
                    "[[IRIS_QUANT]] tower q{bits}: {packed} projections packed, worst {worst:.4} x \
                     the f32-grid optimum ({worst_at})"
                );
                assert_eq!(packed, tower_cfg.num_layers * 7, "tower q{bits}");
                assert!(
                    worst <= 1.05,
                    "tower q{bits}: {worst_at} at {worst} x the optimum"
                );
            }
        },
    );
}

/// What the q8 / q4 tiers' quantization controls, measured without the sampler's trajectory: the
/// conditioning the quantized tower hands the backbone, and ONE CFG-combined denoiser evaluation
/// at a fixed 512² noise and timestep — production loads (`IrisTextEncoder::load_quantized`,
/// `load_backbone_quantized`), solver steps 0 (s = 1.0) and 15 (s = 0.8) of the 30-step plan,
/// CFG 3, the empty negative prompt.
///
/// **The floor.** Every velocity error is judged against the dense tier's own precision noise
/// `F`, measured in the same run: the bf16 backbone against the same backbone at f32 compute,
/// same conditioning. A tier cannot be held closer to bf16 than bf16 is to its own f32 policy.
///
/// **Bounds, from the release** (sc-26266; MLX CPU stream, 512², production packing reproduced
/// as dequantized weights in dense matmuls — fake-quant, so the numbers isolate the weight error
/// from the GPU kernels):
///
/// | measured (step 0 / step 15) | velocity rel | / F |
/// |---|---|---|
/// | F: bf16 vs f32 compute | 7.64 % / 5.84 % | 1 / 1 |
/// | backbone q8 (bf16 tower) | 7.63 % / 8.86 % | 1.00 / 1.52 |
/// | backbone q4 (bf16 tower) | 12.4 % / 26.5 % | 1.62 / 4.53 |
/// | tower q8 (bf16 backbone) | 7.81 % / 6.48 % | 1.02 / 1.11 |
/// | tower q4 (bf16 backbone) | 9.46 % / 7.01 % | 1.24 / 1.20 |
/// | control: 0.086 % Gaussian on every backbone weight | 7.25 % / 6.57 % | 0.95 / 1.12 |
/// | control: 0.86 % Gaussian (q8's error, unstructured) | 7.62 % / 8.68 % | 1.00 / 1.49 |
///
/// Conditioning (relative L2 over the selected hidden states, prompt / empty negative): q8 2.3 %
/// / 4.3 %, q4 18.2 % / 36.1 %; the 0.086 % Gaussian control on every tower projection 1.7 % /
/// 3.4 %.
///
/// The q8 backbone moves one step exactly as far as unstructured noise of its own weight error
/// does (1.52 F against 1.49 F) — no structured defect — and the q4 backbone 3x further. Gated
/// per step, the backbone alone (bf16 conditioning): q8 ≤ 2 F, q4 ≤ 7 F; the whole tier (both
/// components quantized, not measured on CPU — the tower alone adds ~0.1–0.2 F): q8 ≤ 2.5 F,
/// q4 ≤ 8 F. Conditioning: q8 ≤ 6e-2 and cos ≥
/// 0.998, q4 ≤ 0.5 and cos ≥ 0.90, for both prompts.
///
/// *Mutations that red this:* a q8 tier packed at 4 bits (backbone ~4.5 F > 2 F; tower
/// conditioning ~0.2–0.36 > 6e-2); any defect that moves one denoiser step beyond the precision
/// floor's multiple — a mis-padded or mis-grouped projection, a packed base the forward reads
/// wrongly on the GPU kernels.
#[test]
#[ignore = "needs the real weights (IRIS_WEIGHTS_DIR, IRIS_TEXT_ENCODER_DIR) and the GPU"]
fn real_quantized_tiers_track_the_bf16_denoiser_at_512() {
    use mlx_gen::gen_core::iris::GenerationParams;
    use mlx_gen_iris::pipeline::{encode, noise, Conditioning};
    use mlx_gen_iris::solver::cfg_combine;
    use mlx_gen_iris::{IrisDiT, TextBatch};
    use mlx_rs::ops::concatenate_axis;

    let backbone = env_dir("IRIS_WEIGHTS_DIR");
    let config = IrisConfig::from_dir(&backbone).unwrap();
    let tower = env_dir("IRIS_TEXT_ENCODER_DIR");
    let req = GenerationRequest {
        prompt: "a red fox sleeping in fresh snow, golden hour".into(),
        width: 512,
        height: 512,
        steps: Some(30),
        seed: Some(26266),
        ..Default::default()
    };
    let params = GenerationParams::resolve(&req, 0, &config).unwrap();
    let plan = params.plan().unwrap();
    let z = noise(26266, config.model.in_channels, 512, 512).unwrap();
    let steps = [0usize, 15];
    let rel = |a: &Array, b: &Array| -> (f64, f64) {
        let (a, b) = (
            a.as_dtype(Dtype::Float32).unwrap(),
            b.as_dtype(Dtype::Float32).unwrap(),
        );
        let sum = |x: Array| f64::from(x.sum(None).unwrap().item::<f32>());
        let d = sum(a.subtract(&b).unwrap().square().unwrap());
        let (na, nb) = (sum(a.square().unwrap()), sum(b.square().unwrap()));
        let dot = sum(a.multiply(&b).unwrap());
        ((d / nb).sqrt(), dot / (na * nb).sqrt())
    };

    let mut conditioning: Vec<Conditioning> = Vec::new();
    for quant in TIERS {
        let te = IrisTextEncoder::load_quantized(&tower, &config.text_encoder, quant).unwrap();
        assert_eq!(te.is_quantized(), quant.is_some());
        let c = encode(&te, &params).unwrap();
        mlx_rs::transforms::eval([&c.cond[0].states, &c.uncond.as_ref().unwrap().states]).unwrap();
        drop(te);
        mlx_rs::memory::clear_cache();
        conditioning.push(c);
    }
    for (i, tier, bound, min_cos) in [(1, "q8", 6e-2, 0.998), (2, "q4", 0.5, 0.90)] {
        for (which, got, want) in [
            ("prompt", &conditioning[i].cond[0], &conditioning[0].cond[0]),
            (
                "negative",
                conditioning[i].uncond.as_ref().unwrap(),
                conditioning[0].uncond.as_ref().unwrap(),
            ),
        ] {
            let (r, c) = rel(&got.states, &want.states);
            eprintln!("[[IRIS_QUANT]] tower {tier} {which} conditioning: rel {r:.4e} cos {c:.6}");
            assert!(
                r <= bound && c >= min_cos,
                "tower {tier} {which}: rel {r}, cos {c}"
            );
        }
    }

    let velocity = |dit: &IrisDiT, cond: &Conditioning, step: usize| -> Array {
        let t = plan[step].model_time(config.flow.num_train_timesteps);
        let (u, c) = (cond.uncond.as_ref().unwrap(), &cond.cond[0]);
        let states = concatenate_axis(&[&u.states, &c.states], 0).unwrap();
        let mask = vec![u.mask.clone(), c.mask.clone()];
        let out = dit
            .forward(
                &concatenate_axis(&[&z, &z], 0).unwrap(),
                &Array::from_slice(&[t, t], &[2]),
                &TextBatch {
                    states: &states,
                    mask: &mask,
                },
            )
            .unwrap();
        let halves = out.split(2, 0).unwrap();
        let v = cfg_combine(&halves[0], &halves[1], params.cfg_scale).unwrap();
        v.eval().unwrap();
        v
    };
    let load = |quant: Option<Quant>, compute: Dtype| {
        load_backbone_quantized(
            &backbone,
            &config,
            compute,
            &[],
            IrisTask::Generation,
            MODEL_ID,
            quant,
        )
        .unwrap()
        .0
    };
    let dense = load(None, Dtype::Bfloat16);
    let reference: Vec<Array> = steps
        .iter()
        .map(|&s| velocity(&dense, &conditioning[0], s))
        .collect();
    drop(dense);
    mlx_rs::memory::clear_cache();
    let f32_dense = load(None, Dtype::Float32);
    let floor: Vec<f64> = steps
        .iter()
        .zip(&reference)
        .map(|(&s, r)| rel(r, &velocity(&f32_dense, &conditioning[0], s)).0)
        .collect();
    drop(f32_dense);
    mlx_rs::memory::clear_cache();
    eprintln!("[[IRIS_QUANT]] floor F (bf16 vs f32 compute) at steps {steps:?}: {floor:?}");

    for (quant, i, tier_bound, backbone_bound) in
        [(Quant::Q8, 1, 2.5, 2.0), (Quant::Q4, 2, 8.0, 7.0)]
    {
        let dit = load(Some(quant), Dtype::Bfloat16);
        for (k, &step) in steps.iter().enumerate() {
            let (alone, _) = rel(&velocity(&dit, &conditioning[0], step), &reference[k]);
            let (tier, cos) = rel(&velocity(&dit, &conditioning[i], step), &reference[k]);
            eprintln!(
                "[[IRIS_QUANT]] {quant:?} step {step} (s = {:.3}): backbone alone {alone:.4e} \
                 ({:.2} F), whole tier {tier:.4e} ({:.2} F, cos {cos:.6})",
                plan[step].s,
                alone / floor[k],
                tier / floor[k]
            );
            assert!(
                alone <= backbone_bound * floor[k],
                "{quant:?} step {step}: backbone alone {alone} > {backbone_bound} F ({})",
                floor[k]
            );
            assert!(
                tier <= tier_bound * floor[k],
                "{quant:?} step {step}: whole tier {tier} > {tier_bound} F ({})",
                floor[k]
            );
        }
        drop(dit);
        mlx_rs::memory::clear_cache();
    }

    #[cfg(target_os = "macos")]
    crate::common::metal_guard::record_model_inference_device();
    eprintln!("\nMODEL_INFERENCE_BODY_COMPLETED quant_real_weights::real_quantized_tiers_track_the_bf16_denoiser_at_512");
}
