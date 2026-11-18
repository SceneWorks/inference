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
    steps: Vec<Step>,
}
#[derive(Deserialize)]
struct Image {
    width: u32,
    height: u32,
    rgb: [u8; 3],
}
impl Image {
    fn pixels(&self) -> ImageRef {
        ImageRef::new(
            self.width,
            self.height,
            self.rgb.repeat(self.width as usize * self.height as usize),
        )
        .unwrap()
    }
}
#[derive(Deserialize)]
struct Step {
    token_id: i32,
    token_ids: Vec<i32>,
    logits: Vec<f32>,
    max_abs_error: f32,
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
        "SENSENOVA_LOOP_MEMORY {}",
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
        let expected = if case.id == "text" {
            "91"
        } else {
            case.id.as_str()
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
    let mut r = request("Return a JSON object with status set to ok.", None);
    r.constraint = Some(Constraint::JsonSchema(
        JsonSchema::compile(&json!({
        "type":"object","properties":{"status":{"type":"string","enum":["ok"]}},
        "required":["status"],"additionalProperties":false}))
        .unwrap(),
    ));
    let out = provider.complete(&r).unwrap();
    assert_eq!(
        serde_json::from_str::<Value>(&out.text).unwrap(),
        json!({"status":"ok"})
    );
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
    println!("SENSENOVA_LOOP_ORACLE_SHA256 {:x}", Sha256::digest(&bytes));
    let oracle: Oracle = serde_json::from_slice(&bytes).unwrap();
    assert_eq!(oracle.schema_version, 1);
    assert_eq!(oracle.hf_revision, core_llm::sensenova_loop::MODEL_REVISION);
    assert_eq!(oracle.source_inventory_sha256.len(), 64);
    let mut cases: Vec<_> = oracle.cases.iter().map(|c| c.id.as_str()).collect();
    cases.sort_unstable();
    assert_eq!(cases, ["blue", "red", "text"]);
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
