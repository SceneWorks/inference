//! Dedicated Metal proof for the released checkpoint and both stored MLX tiers.
use std::path::{Path, PathBuf};

use core_llm::{
    Constraint, Content, FinishReason, ImageRef, JsonSchema, LoadSpec, Message, Sampling,
    StreamEvent, TextLlm, TextLlmRequest, ThinkingMode,
};
use mlx_llm::{
    image::{Qwen35ImageProcessor, Qwen35Preprocess},
    models::{Qwen35Cache, Qwen35Config, Qwen35Model, Qwen35VisionConfig, Qwen35VisionModel},
    primitives::{kv_cache::KvCache, Weights},
    LlamaProvider,
};
use mlx_rs::{Array, Dtype};
use serde::Deserialize;
use serde_json::{json, Value};
use sha2::{Digest, Sha256};

#[derive(Deserialize)]
struct Oracle {
    schema_version: u32,
    hf_revision: String,
    source_inventory_sha256: String,
    cases: Vec<Case>,
}
#[derive(Deserialize)]
struct Case {
    id: String,
    prompt: String,
    thinking: bool,
    input_ids: Vec<i32>,
    image: Option<Image>,
    image_grid_thw: Option<Vec<[i32; 3]>>,
    image_processor_samples: Vec<PatchSample>,
    image_processor_sample_abs_tolerance: Option<f32>,
    steps: Vec<Step>,
}
#[derive(Deserialize)]
struct PatchSample {
    patch: usize,
    offset: usize,
    value: f32,
}
#[derive(Deserialize, Default)]
struct Image {
    width: u32,
    height: u32,
    rgb: [u8; 3],
    split_x: Option<u32>,
    split_y: Option<u32>,
    quadrants: Option<[[u8; 3]; 4]>,
}
impl Image {
    fn pixels(&self) -> ImageRef {
        let pixels = if let Some(colors) = self.quadrants {
            let (sx, sy) = (self.split_x.unwrap(), self.split_y.unwrap());
            (0..self.height)
                .flat_map(|y| {
                    (0..self.width)
                        .flat_map(move |x| colors[2 * usize::from(y >= sy) + usize::from(x >= sx)])
                })
                .collect()
        } else {
            self.rgb.repeat(self.width as usize * self.height as usize)
        };
        ImageRef::new(self.width, self.height, pixels).unwrap()
    }
}
#[derive(Deserialize)]
struct Step {
    token_id: i32,
    token_ids: Vec<i32>,
    logits: Vec<f32>,
    max_abs_error: f32,
    semantic_score: Option<SemanticScore>,
}

#[derive(Deserialize)]
struct SemanticScore {
    positive_token_ids: Vec<i32>,
    negative_token_ids: Vec<i32>,
    source_positive_margin: f32,
}

impl SemanticScore {
    fn margin(&self, ids: &[i32], logits: &[f32]) -> f32 {
        let maximum = |group: &[i32]| {
            assert!(!group.is_empty());
            group
                .iter()
                .map(|token| {
                    let index = ids
                        .iter()
                        .position(|id| id == token)
                        .expect("semantic token absent from reference");
                    let value = logits[index];
                    assert!(value.is_finite());
                    value
                })
                .fold(f32::NEG_INFINITY, f32::max)
        };
        maximum(&self.positive_token_ids) - maximum(&self.negative_token_ids)
    }
}

fn request(prompt: &str, image: Option<&Image>) -> TextLlmRequest {
    let mut user = Message::user(prompt);
    if let Some(image) = image {
        user.content.insert(0, Content::Image(image.pixels()));
    }
    TextLlmRequest {
        messages: vec![user],
        sampling: Sampling::greedy(),
        max_new_tokens: 32,
        thinking: ThinkingMode::Disabled,
        seed: Some(0),
        ..Default::default()
    }
}

/// Sample the process ledger across both load and request. MLX's resettable peak
/// counts array buffers, while the process ledger also covers the tokenizer,
/// image preprocessing, queued Metal allocations and released driver buffers.
fn observe_peak<T>(run: impl FnOnce() -> T) -> (T, u64) {
    use mlx_llm::primitives::weights::process_footprint_bytes;
    use std::{sync::mpsc, thread, time::Duration};
    let (stop, samples) = mpsc::channel::<()>();
    thread::scope(|scope| {
        let reader = scope.spawn(move || {
            let mut peak = process_footprint_bytes();
            while matches!(
                samples.recv_timeout(Duration::from_millis(50)),
                Err(mpsc::RecvTimeoutError::Timeout)
            ) {
                peak = peak.max(process_footprint_bytes());
            }
            peak.max(process_footprint_bytes())
        });
        let value = run();
        drop(stop);
        (value, reader.join().expect("physical footprint sampler"))
    })
}

fn load_and_measure(dir: &Path, tier: &str) -> LlamaProvider {
    use mlx_llm::primitives::weights::process_footprint_bytes;
    let spec = LoadSpec::dense(dir.display().to_string());
    if tier == "Q8" {
        return LlamaProvider::load(&spec).unwrap();
    }
    mlx_rs::memory::clear_cache();
    mlx_rs::memory::reset_peak_memory();
    let baseline = process_footprint_bytes();
    let ((provider, record), observed_peak) = observe_peak(|| {
        let provider = LlamaProvider::load(&spec).unwrap();
        let load_peak = mlx_rs::memory::get_peak_memory();
        let load_footprint = process_footprint_bytes();
        mlx_rs::memory::reset_peak_memory();
        let mut req = request(
            "Name the dominant color of this image.",
            Some(&Image {
                width: 1536,
                height: 1024,
                rgb: [35, 70, 200],
                ..Default::default()
            }),
        );
        req.max_new_tokens = 1;
        let out = provider.complete(&req).unwrap();
        assert_eq!(out.usage.generated_tokens, 1);
        assert!(out.usage.prompt_tokens >= 1536);
        let context = provider.descriptor().capabilities.max_context_tokens;
        assert_eq!(context, 524288);
        let record = json!({"tier":tier,"image":[1536,1024],"context_cap":context,"prefill_tokens":out.usage.prompt_tokens,"generated_tokens":out.usage.generated_tokens,"output":out.text,
            "loaded_process_bytes":load_footprint,"load_metal_peak_bytes":load_peak,"request_metal_peak_bytes":mlx_rs::memory::get_peak_memory(),
            "retained_metal_bytes":mlx_rs::memory::get_active_memory(),"metal_allocator_cached_bytes":mlx_rs::memory::get_cache_memory()});
        (provider, record)
    });
    assert!(baseline > 0 && observed_peak >= baseline);
    println!(
        "\nSENSENOVA_LOOP_MEMORY {}",
        json!({"product":record,"process_baseline_bytes":baseline,"sampled_process_peak_bytes":observed_peak,"sample_interval_ms":50})
    );
    provider
}

fn production(dir: &Path, tier: &str, cases: &[Case]) {
    mlx_rs::memory::reset_peak_memory();
    let provider = load_and_measure(dir, tier);
    assert_eq!(provider.descriptor().family, "sensenova_loop");
    assert_eq!(provider.is_quantized(), tier != "BF16");
    let caps = &provider.descriptor().capabilities;
    let template = core_llm::JinjaChatTemplate::new(
        std::fs::read_to_string(dir.join("chat_template.jinja")).unwrap(),
    );
    let tokenizer = core_llm::Tokenizer::from_file(dir.join("tokenizer.json")).unwrap();
    assert!(caps.supports_vision && caps.speculative_proposers().is_empty());
    for case in cases {
        use core_llm::ChatTemplate;
        let user = if let Some(image) = &case.image {
            let p = Qwen35ImageProcessor::default();
            let (h, w) = p
                .smart_resize(image.height as usize, image.width as usize)
                .unwrap();
            let n = (h / p.patch_size / p.merge_size) * (w / p.patch_size / p.merge_size);
            format!(
                "<|vision_start|>{}<|vision_end|>{}",
                "<|image_pad|>".repeat(n),
                case.prompt
            )
        } else {
            case.prompt.clone()
        };
        let prompt = template
            .render_with(
                &[Message::user(user)],
                &core_llm::RenderOptions::generation().with_enable_thinking(Some(case.thinking)),
            )
            .unwrap();
        let native_ids = tokenizer.encode(&prompt, false).unwrap();
        assert!(
            native_ids
                .iter()
                .map(|&id| id as i32)
                .eq(case.input_ids.iter().copied()),
            "prompt differs from upstream: {tier}/{}",
            case.id
        );
        let r = request(&case.prompt, case.image.as_ref());
        let out = provider.complete(&r).unwrap();
        let expected = match case.id.as_str() {
            "text" => "91",
            "quadrants" => "red",
            color => color,
        };
        assert!(
            out.text
                .trim()
                .trim_end_matches(['.', '!'])
                .eq_ignore_ascii_case(expected),
            "{tier}/{}: {:?}",
            case.id,
            out.text
        );
        assert!(out.thinking.as_deref().unwrap_or("").is_empty());
        assert_eq!(
            out.finish_reason,
            Some(FinishReason::Stop),
            "{tier}/{}",
            case.id
        );
        assert_eq!(out.usage.prompt_tokens as usize, case.input_ids.len());
        println!(
            "SENSENOVA_LOOP_PRODUCT {}",
            json!({"tier":tier,"case":case.id,"answer":out.text,"prompt_tokens":out.usage.prompt_tokens,"generated_tokens":out.usage.generated_tokens,"metal_peak_bytes":mlx_rs::memory::get_peak_memory()})
        );
    }
    let schema = JsonSchema::compile(&json!({
        "type":"object","properties":{"color":{"type":"string","enum":["red","blue","green","yellow"]}},
        "required":["color"],"additionalProperties":false
    })).unwrap();
    for case in cases
        .iter()
        .filter(|c| matches!(c.id.as_str(), "blue" | "red"))
    {
        let mut r = request(
            "What is the dominant color of the image? Return a JSON object with a color field.",
            case.image.as_ref(),
        );
        r.constraint = Some(Constraint::JsonSchema(schema.clone()));
        let out = provider.complete(&r).unwrap();
        assert_eq!(
            serde_json::from_str::<Value>(&out.text).unwrap(),
            json!({"color":case.id})
        );
    }
    let mut r = request("What is 13 times 7? Explain your reasoning.", None);
    r.thinking = ThinkingMode::Enabled;
    r.max_new_tokens = 8;
    let out = provider.complete(&r).unwrap();
    assert!(!out.thinking.as_deref().unwrap_or("").trim().is_empty());
    let r = request("Count from one to ten.", None);
    let mut emitted = 0;
    let out = provider
        .generate(&r, &mut |event| {
            if matches!(event, StreamEvent::Token { .. }) {
                emitted += 1;
                r.cancel.cancel();
            }
        })
        .unwrap();
    assert_eq!(emitted, 1);
    assert_eq!(out.finish_reason, Some(FinishReason::Cancelled));
    println!(
        "SENSENOVA_LOOP_PRODUCT {}",
        json!({"tier":tier,"strict_json":true,"thinking":true,"cancelled_after_tokens":emitted})
    );
}

struct Visual {
    features: Array,
    grid: Vec<[i32; 3]>,
}

fn prefill(
    model: &Qwen35Model,
    ids: &[i32],
    image: Option<&Visual>,
    cache: &mut Qwen35Cache,
) -> (Array, i32) {
    let tensor = Array::from_slice(ids, &[1, ids.len() as i32]);
    if let Some(visual) = image {
        let embeds = model.embed_input_ids(&tensor).unwrap();
        let merged = model
            .splice_image_features(&embeds, ids, &visual.features, 248056)
            .unwrap();
        let (t, h, w, delta) = model.mrope_positions(ids, &visual.grid, 248056, 2).unwrap();
        (
            model
                .decode_logits_from_embeds(&merged, [&t, &h, &w], cache)
                .unwrap(),
            delta,
        )
    } else {
        (model.decode_logits(&tensor, cache, 0).unwrap(), 0)
    }
}

fn selected(logits: &Array, ids: &[i32]) -> Vec<f32> {
    logits
        .reshape(&[-1])
        .unwrap()
        .take_axis(Array::from_slice(ids, &[ids.len() as i32]), 0)
        .unwrap()
        .as_dtype(Dtype::Float32)
        .unwrap()
        .as_slice::<f32>()
        .to_vec()
}

fn centered_error(lhs: &[f32], rhs: &[f32]) -> f32 {
    assert!(!lhs.is_empty() && lhs.len() == rhs.len());
    lhs.iter()
        .zip(rhs)
        .map(|(a, b)| {
            assert!(a.is_finite() && b.is_finite());
            ((a - lhs[0]) - (b - rhs[0])).abs()
        })
        .fold(0., f32::max)
}

/// Compare native packed matmul to CPU FP32 arithmetic over independently unpacked
/// checkpoint words. Basis inputs cover word/group boundaries. The mixed input
/// exercises signed reduction across those boundaries and both ends of each row.
fn check_packed_projection(weights: &Weights, spec: mlx_llm::primitives::QuantSpec) {
    use mlx_llm::primitives::projection::Projection;
    let key = "model.language_model.layers.3.self_attn.k_proj";
    let weight = weights.require(&format!("{key}.weight")).unwrap();
    let rows = weight.shape()[0];
    let take = |tensor: &Array| {
        tensor
            .take_axis(Array::from_slice(&[0, rows / 2, rows - 1], &[3]), 0)
            .unwrap()
    };
    let packed = take(weight);
    let scales = take(weights.require(&format!("{key}.scales")).unwrap());
    let biases = take(weights.require(&format!("{key}.biases")).unwrap());
    let words = packed.as_slice::<u32>();
    let scales_f32 = scales.as_dtype(Dtype::Float32).unwrap();
    let biases_f32 = biases.as_dtype(Dtype::Float32).unwrap();
    let scales_fp32 = scales_f32.as_slice::<f32>();
    let biases_fp32 = biases_f32.as_slice::<f32>();
    let bits = spec.bits as usize;
    let per_word = 32 / bits;
    let group = spec.group_size as usize;
    let width = scales.shape()[1] as usize * group;
    let columns = [
        0,
        1,
        per_word - 1,
        per_word,
        group - 1,
        group,
        width / 2 + 1,
        width - 1,
    ];
    let mut inputs = vec![0f32; (columns.len() + 1) * width];
    for (i, &column) in columns.iter().enumerate() {
        inputs[i * width + column] = 1.;
        inputs[columns.len() * width + column] = if i % 2 == 0 { 0.5 } else { -1. };
    }
    let projection =
        Projection::from_quantized(packed.clone(), scales, biases, spec, Dtype::Bfloat16).unwrap();
    let x = Array::from_slice(&inputs, &[(columns.len() + 1) as i32, width as i32])
        .as_dtype(Dtype::Bfloat16)
        .unwrap();
    let native = projection
        .forward(&x)
        .unwrap()
        .as_dtype(Dtype::Float32)
        .unwrap();
    let outputs = native.as_slice::<f32>();
    let mut max_error = 0f32;
    for (i, input) in inputs.chunks_exact(width).enumerate() {
        for row in 0..3 {
            let mut reference = 0.;
            let mut magnitude = 0.;
            for &column in &columns {
                let word = words[(row * width + column) / per_word];
                let code = ((word >> ((column % per_word) * bits)) & ((1 << bits) - 1)) as f32;
                let g = (row * width + column) / group;
                let scaled = code * scales_fp32[g];
                let bias = biases_fp32[g];
                reference += input[column] * (scaled + bias);
                magnitude += input[column].abs() * (scaled.abs() + bias.abs());
            }
            // Two BF16 ULPs on the contributing terms allow native dequantization
            // and the BF16 result cast, including cancellation near zero.
            let tolerance = magnitude * (2.0 / 128.0) + f32::EPSILON;
            let error = (outputs[3 * i + row] - reference).abs();
            assert!(
                error <= tolerance,
                "Q{bits} {key} input {i} row {row}: {error} > {tolerance}"
            );
            max_error = max_error.max(error);
        }
    }
    println!(
        "\nSENSENOVA_LOOP_PROJECTION {}",
        json!({"bits":bits,"projection":key,"rows":3,"inputs":columns.len()+1,"max_abs_error":max_error})
    );
}

fn forced_scores(dir: &Path, tier: &str, cases: &[Case]) {
    let config: Value =
        serde_json::from_slice(&std::fs::read(dir.join("config.json")).unwrap()).unwrap();
    let mut weights = Weights::from_configured_dir(dir, &config).unwrap();
    let cfg = Qwen35Config::from_json(&config).unwrap();
    assert_eq!(
        cfg.quantization.map(|q| q.bits),
        match tier {
            "Q4" => Some(4),
            "Q8" => Some(8),
            _ => None,
        }
    );
    if let Some(spec) = cfg.quantization {
        check_packed_projection(&weights, spec);
    }
    let model = Qwen35Model::build_lazy(&weights, "model.language_model", cfg, None).unwrap();
    weights.materialize_groups(&model.param_groups()).unwrap();
    let tower = Qwen35VisionModel::from_sensenova_weights(
        &weights,
        "model.visual",
        Qwen35VisionConfig::from_json(&config).unwrap(),
    )
    .unwrap();
    drop(weights);
    for case in cases {
        assert!(!case.thinking && case.steps.len() >= 3 && !case.input_ids.is_empty());
        let image = case.image.as_ref().map(|image| {
            let pixels = image.pixels();
            let (patches, grid) = Qwen35ImageProcessor::default()
                .preprocess(
                    &pixels.pixels,
                    pixels.width as usize,
                    pixels.height as usize,
                )
                .unwrap();
            assert_eq!(case.image_grid_thw.as_deref(), Some(grid.as_slice()));
            let tolerance = if case.id == "quadrants" { 0.02 } else { 1e-5 };
            assert_eq!(case.image_processor_sample_abs_tolerance, Some(tolerance));
            assert!(!case.image_processor_samples.is_empty());
            let [rows, width] = patches.shape() else {
                panic!("expected a matrix of image patches")
            };
            let actual = patches.as_slice::<f32>();
            for sample in &case.image_processor_samples {
                assert!(sample.patch < *rows as usize && sample.offset < *width as usize);
                let value = actual[sample.patch * *width as usize + sample.offset];
                assert!(
                    (value - sample.value).abs() <= tolerance,
                    "{tier}/{} patch {} offset {}: {value} vs {}",
                    case.id,
                    sample.patch,
                    sample.offset,
                    sample.value
                );
            }
            let features = tower.forward(&patches, &grid).unwrap();
            Visual { features, grid }
        });
        let mut ids = case.input_ids.clone();
        let mut cache = model.new_cache();
        let (mut logits, delta) = prefill(&model, &ids, image.as_ref(), &mut cache);
        assert_eq!(cache.num_layers(), 56);
        for (i, step) in case.steps.iter().enumerate() {
            let actual = selected(&logits, &step.token_ids);
            assert_eq!(cache.offset(), ids.len() as i32);
            let kv = cache.attention_kv_dtypes().unwrap();
            assert_eq!(kv, vec![(Dtype::Bfloat16, Dtype::Bfloat16); 14]);
            assert!(cache.recurrent_bytes() > 0);
            let reference_error = centered_error(&actual, &step.logits);
            if let Some(score) = &step.semantic_score {
                let source = score.margin(&step.token_ids, &step.logits);
                assert!(source > 0.);
                assert_eq!(source, score.source_positive_margin);
                let margin = score.margin(&step.token_ids, &actual);
                assert!(
                    margin > 0.,
                    "{tier}/{} step {i}: semantic margin {margin}",
                    case.id
                );
                println!(
                    "\nSENSENOVA_LOOP_SEMANTIC {}",
                    json!({"tier":tier.to_ascii_lowercase(),"case_id":case.id,"step":i,"margin":margin})
                );
            }
            let mut cold = model.new_cache();
            let cold_logits = prefill(&model, &ids, image.as_ref(), &mut cold).0;
            let cold_error = centered_error(&actual, &selected(&cold_logits, &step.token_ids));
            println!(
                "SENSENOVA_LOOP_SCORES {}",
                json!({"tier":tier,"case":case.id,"step":i,"position":ids.len(),"full_kv":kv.len(),"linear_states":cache.num_layers()-kv.len(),"cache_bytes":cache.bytes(),"oracle_max_centered_error":reference_error,"cold_max_centered_error":cold_error,"upstream_envelope":step.max_abs_error,"token_ids":step.token_ids,"native_logits":actual})
            );
            assert!(step.max_abs_error.is_finite() && step.max_abs_error > 0.);
            if tier == "BF16" {
                assert!(
                    reference_error <= step.max_abs_error,
                    "BF16 {}/step {i}: {reference_error} > {}",
                    case.id,
                    step.max_abs_error
                );
            }
            assert!(
                cold_error <= step.max_abs_error,
                "{tier} {}/step {i} cached vs cold: {cold_error} > {}",
                case.id,
                step.max_abs_error
            );
            if i + 1 < case.steps.len() {
                let next = Array::from_slice(&[step.token_id], &[1, 1]);
                logits = model
                    .decode_logits(&next, &mut cache, ids.len() as i32 + delta)
                    .unwrap();
                ids.push(step.token_id);
            }
        }
    }
}

#[test]
#[ignore = "model inference; explicit model-proof workflow"]
fn sensenova_loop_release_proof() {
    let path = std::env::var_os("SENSENOVA_LOOP_ORACLE").expect("SENSENOVA_LOOP_ORACLE");
    let bytes = std::fs::read(path).unwrap();
    println!(
        "\nSENSENOVA_LOOP_ORACLE_SHA256 {:x}",
        Sha256::digest(&bytes)
    );
    let oracle: Oracle = serde_json::from_slice(&bytes).unwrap();
    assert_eq!(oracle.schema_version, 1);
    assert_eq!(oracle.hf_revision, core_llm::sensenova_loop::MODEL_REVISION);
    assert_eq!(oracle.source_inventory_sha256.len(), 64);
    let mut cases: Vec<_> = oracle.cases.iter().map(|c| c.id.as_str()).collect();
    cases.sort_unstable();
    assert_eq!(cases, ["blue", "quadrants", "red", "text"]);
    let image = oracle
        .cases
        .iter()
        .find(|c| c.id == "quadrants")
        .unwrap()
        .image
        .as_ref()
        .unwrap();
    assert_eq!(
        (image.width, image.height, image.split_x, image.split_y),
        (353, 249, Some(209), Some(155))
    );
    assert_eq!(image.rgb, [205, 35, 35]);
    assert_eq!(
        image.quadrants.unwrap(),
        [[205, 35, 35], [35, 70, 200], [36, 174, 47], [232, 190, 40]]
    );
    assert!(matches!(
        mlx_rs::Stream::task_local_or_default()
            .get_device()
            .unwrap()
            .get_type()
            .unwrap(),
        mlx_rs::DeviceType::Gpu
    ));
    for tier in ["BF16", "Q8", "Q4"] {
        let dir = PathBuf::from(
            std::env::var_os(format!("SENSENOVA_LOOP_{tier}_SNAPSHOT")).expect("tier snapshot"),
        );
        production(&dir, tier, &oracle.cases);
        mlx_rs::memory::clear_cache();
        forced_scores(&dir, tier, &oracle.cases);
        mlx_rs::memory::clear_cache();
        crate::model_inference_device::record_model_inference_device();
    }
    println!(
        "MODEL_INFERENCE_BODY_COMPLETED sensenova_loop_real_weights::sensenova_loop_release_proof"
    );
}
