//! Dedicated CUDA proof for the pinned SenseNova-Loop-9B and its two native
//! GGML tiers. The HF oracle is produced independently from the frozen wrapper.
#[cfg(feature = "cuda")]
#[path = "common/cuda_pool.rs"]
mod cuda_pool;
use candle_llm::LlamaProvider;
use core_llm::{
    Constraint, Content, FinishReason, ImageRef, JsonSchema, LoadSpec, Message, Quantize, Role,
    Sampling, StreamEvent, TextLlm, TextLlmRequest, ThinkingMode,
};
use serde_json::{json, Value};
use std::{collections::BTreeSet, path::PathBuf};

fn request(case: &Value) -> TextLlmRequest {
    let mut content = Vec::new();
    if let Some(image) = case["image"].as_object() {
        let w = image["width"].as_u64().unwrap() as u32;
        let h = image["height"].as_u64().unwrap() as u32;
        let rgb: [u8; 3] = serde_json::from_value(image["rgb"].clone()).unwrap();
        content.push(Content::Image(
            ImageRef::new(w, h, rgb.repeat((w * h) as usize)).unwrap(),
        ));
    }
    content.push(Content::text(case["prompt"].as_str().unwrap()));
    assert_eq!(case["thinking"], false);
    TextLlmRequest {
        messages: vec![Message {
            role: Role::User,
            content,
            thinking: None,
            tool_calls: vec![],
        }],
        sampling: Sampling::greedy(),
        max_new_tokens: 32,
        seed: Some(0),
        thinking: ThinkingMode::Disabled,
        ..Default::default()
    }
}

fn centered_error(actual: &[f32], expected: &[f32]) -> f32 {
    assert_eq!(actual.len(), expected.len());
    assert!(actual.iter().chain(expected).all(|v| v.is_finite()));
    actual
        .iter()
        .zip(expected)
        .map(|(x, y)| ((x - actual[0]) - (y - expected[0])).abs())
        .fold(0., f32::max)
}

#[cfg(feature = "cuda")]
fn production_request(
    model: &LlamaProvider,
    tier: &str,
    ctx: &candle_core::cuda_backend::cudarc::driver::CudaContext,
    before: &Value,
) {
    let after_load = cuda_pool::sample(ctx);
    cuda_pool::reset_pool_peaks(ctx);
    let mut req = request(
        &json!({"prompt":"What is the dominant color of this image? Answer with one word.",
        "thinking":false,"image":{"width":1536,"height":1024,"rgb":[205,35,35]}}),
    );
    req.max_new_tokens = 1;
    let mut ids = Vec::new();
    let out = model
        .generate(&req, &mut |event| {
            if let StreamEvent::Token { id, .. } = event {
                ids.push(id)
            }
        })
        .unwrap();
    let after_request = cuda_pool::sample(ctx);
    assert_eq!(out.usage.generated_tokens, 1);
    assert!(out.usage.prompt_tokens > 1536);
    assert_eq!(ids.len(), 1);
    println!(
        "SENSENOVA_LOOP_MEMORY_ENDPOINT {}",
        json!({"tier":tier,"device":"candle-cuda",
        "memory_scope":"synchronized device-wide usage and CUDA primary-context pool high watermarks, reset separately before cold load and first request",
        "width":1536,"height":1024,"generated_tokens":out.usage.generated_tokens,"prompt_tokens":out.usage.prompt_tokens,
        "token_id":ids[0],"before_load":before,"after_load":after_load,"after_request":after_request})
    );
}

#[test]
#[ignore = "model inference; explicit model-proof workflow"]
fn sensenova_loop_release_proof() {
    let path = PathBuf::from(
        std::env::var_os("SENSENOVA_LOOP_ORACLE").expect("upstream BF16 oracle JSON"),
    );
    let oracle: Value = serde_json::from_reader(std::fs::File::open(&path).unwrap()).unwrap();
    assert_eq!(oracle["schema_version"], 1);
    assert_eq!(
        oracle["hf_revision"],
        core_llm::sensenova_loop::MODEL_REVISION
    );
    assert_eq!(
        oracle["source_inventory_sha256"].as_str().unwrap().len(),
        64
    );
    #[cfg(feature = "model-inference-proof")]
    candle_llm::model_input_proof::record_reader_paths(
        "candle-llm::sensenova_loop_proof::read_oracle",
        candle_llm::model_input_proof::InputRole::Reference,
        &[&path],
    );
    let cases = oracle["cases"].as_array().unwrap();
    assert_eq!(cases.len(), 3);
    assert_eq!(
        cases
            .iter()
            .map(|c| c["id"].as_str().unwrap())
            .collect::<BTreeSet<_>>(),
        BTreeSet::from(["text", "red", "blue"])
    );
    for (tier, quant) in [
        ("BF16", None),
        ("Q8", Some(Quantize::Q8)),
        ("Q4", Some(Quantize::Q4)),
    ] {
        let dir =
            std::env::var(format!("SENSENOVA_LOOP_{tier}_SNAPSHOT")).expect("required snapshot");
        let mut load = LoadSpec::dense(dir.clone());
        load.quantize = quant;
        #[cfg(feature = "cuda")]
        let measurement = (tier != "Q8").then(|| {
            let device = candle_llm::device::select_device().unwrap();
            let ctx = device
                .as_cuda_device()
                .unwrap()
                .cuda_stream()
                .context()
                .clone();
            let before = cuda_pool::sample(&ctx);
            cuda_pool::reset_pool_peaks(&ctx);
            (ctx, before)
        });
        let model = LlamaProvider::load(&load).unwrap();
        #[cfg(feature = "cuda")]
        if let Some((ctx, before)) = measurement {
            production_request(&model, tier, &ctx, &before);
        }
        let desc = model.descriptor();
        assert_eq!(desc.family, "sensenova_loop");
        assert!(desc.capabilities.supports_vision && desc.capabilities.supports_thinking);
        assert_eq!(desc.capabilities.max_context_tokens, 524288);
        let census = model.load_record().census.unwrap();
        assert_eq!(
            census.projections.ggml.count,
            if quant.is_some() { 201 } else { 0 }
        );
        assert_eq!(
            candle_llm::provider::eos_token_ids(std::path::Path::new(&dir)),
            vec![248044, 248046]
        );
        for case in cases {
            let req = request(case);
            let id = case["id"].as_str().unwrap();
            let steps = case["steps"].as_array().unwrap();
            assert_eq!(steps.len(), 3);
            let forced: Vec<_> = steps[..2]
                .iter()
                .map(|s| s["token_id"].as_i64().unwrap() as i32)
                .collect();
            let (ids, rows) = model.sensenova_proof_logits(&req, &forced).unwrap();
            assert_eq!(
                ids,
                serde_json::from_value::<Vec<i32>>(case["input_ids"].clone()).unwrap()
            );
            for (step, ((native, cold), reference)) in rows.iter().zip(steps).enumerate() {
                let selected: Vec<usize> =
                    serde_json::from_value(reference["token_ids"].clone()).unwrap();
                assert!(selected.len() >= 2);
                let expected: Vec<f32> =
                    serde_json::from_value(reference["logits"].clone()).unwrap();
                let subset =
                    |scores: &[f32]| selected.iter().map(|&i| scores[i]).collect::<Vec<_>>();
                let actual = subset(native);
                let hf_error = centered_error(&actual, &expected);
                let cache_error = centered_error(&actual, &subset(cold));
                let tolerance = reference["max_abs_error"].as_f64().unwrap() as f32;
                assert!(tolerance.is_finite() && tolerance > 0.);
                println!("SenseNova tier={tier} case={id} step={step} cuda=true full_cache=14 recurrent_cache=42 hf_centered_error={hf_error} cold_centered_error={cache_error} oracle_tolerance={tolerance}");
                if quant.is_none() {
                    assert!(
                        hf_error <= tolerance,
                        "{tier} {id} step {step}: HF {hf_error} > {tolerance}"
                    );
                }
                assert!(
                    cache_error <= tolerance,
                    "{tier} {id} step {step}: native cache {cache_error} > {tolerance}"
                );
            }
            let result = model.generate(&req, &mut |_| {}).unwrap();
            assert_eq!(
                result.finish_reason,
                Some(FinishReason::Stop),
                "{tier}/{id} must end at a release EOS"
            );
            let expected = if id == "text" { "91" } else { id };
            assert!(
                result.text.to_lowercase().contains(expected),
                "{tier}/{id}: {:?}",
                result.text
            );
            println!("SenseNova tier={tier} case={id} answer={:?}", result.text);
        }
        let mut structured = request(cases.iter().find(|c| c["id"] == "text").unwrap());
        structured.messages[0].content = vec![Content::text(
            "Output a JSON object with answer set to the string 91.",
        )];
        let schema = JsonSchema::compile(&json!({"type":"object","properties":{"answer":{"type":"string","const":"91"}},"required":["answer"],"additionalProperties":false})).unwrap();
        structured.constraint = Some(Constraint::JsonSchema(schema.clone()));
        let out = model.generate(&structured, &mut |_| {}).unwrap();
        assert!(schema.accepts(out.text.trim()));
        assert_eq!(
            serde_json::from_str::<Value>(out.text.trim()).unwrap(),
            json!({"answer":"91"})
        );
        let mut cancel = structured;
        cancel.constraint = None;
        let flag = cancel.cancel.clone();
        let mut emitted = 0;
        let cancelled = model.generate(&cancel, &mut |e| {
            if matches!(e, StreamEvent::Token { .. }) {
                emitted += 1;
                flag.cancel();
            }
        });
        assert!(emitted > 0);
        assert!(
            matches!(cancelled, Err(core_llm::Error::Canceled))
                || cancelled.unwrap().finish_reason == Some(FinishReason::Cancelled)
        );
        println!("SenseNova tier={tier} strict_json={:?} cancelled_after_tokens={emitted} ggml_projections={}",out.text,census.projections.ggml.count);
    }
    #[cfg(feature = "model-inference-proof")]
    println!("\nMODEL_INFERENCE_BODY_COMPLETED sensenova_loop_release_proof");
}
