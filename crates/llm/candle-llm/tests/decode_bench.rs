//! `decode_bench` — the decode-perf suite for the Blackwell fast-decode epic (sc-24128 / sc-24129).
//!
//! One `#[ignore]`d test that loads a Qwen3.6/3.8 snapshot, decodes a fixed greedy fixture through
//! each decode path, and writes one JSON document the `scripts/release/decode_bench.py` harness
//! seals with git identity, hardware and per-process memory samples. It is driven like the
//! sc-23942 native-comparison test: an already-built test executable, `--exact --ignored`, with
//! its inputs in the environment (every path is passed in; nothing is derived from a cache):
//!
//! | variable                   | meaning                                                      |
//! |----------------------------|--------------------------------------------------------------|
//! | `DECODE_BENCH_SNAPSHOT`    | snapshot directory (config.json, tokenizer*.json, shards)    |
//! | `DECODE_BENCH_OUTPUT`      | JSON path to write (must not exist)                          |
//! | `DECODE_BENCH_ROWS`        | comma list of `reference`, `step_model`, `mtp` (default all) |
//! | `DECODE_BENCH_DRAFTS`      | MTP draft widths, comma list (default `1,2,3,4,5`)           |
//! | `DECODE_BENCH_NEW_TOKENS`  | tokens generated per row (default 256)                       |
//! | `DECODE_BENCH_WARMUP_TOKENS` | tokens of the untimed warm-up run (default 16)             |
//! | `DECODE_BENCH_PROMPT`      | user message (default: a long-answer explanation request)    |
//! | `DECODE_BENCH_LABEL`       | hardware label recorded verbatim (default `RTX Pro 6000 / sm_120`) |
//!
//! Rows are greedy (`temperature = 0`), no stop tokens, so every row emits exactly `NEW_TOKENS`
//! and the token sequences are comparable: each row records whether it matched the reference row
//! token-for-token and, if not, the first divergence. Timing brackets the *decode* phase only
//! (prefill is reported separately) with a device synchronize on both sides.
//!
//! The block between the `head-only` markers uses seams that do not exist on the pre-epic
//! baseline (`StepModel`, host-sync accounting). `decode_bench.py baseline-source` rewrites this
//! file into a copy that compiles against `d2b8cb335` by replacing that block with the stub in
//! `BASELINE_STUB`; the harness is what keeps the two in step.

use std::collections::BTreeMap;
use std::path::{Path, PathBuf};
use std::time::Instant;

use candle_core::Device;
use candle_llm::decode::{
    generate_from_prefill, generate_qwen35_mtp_timed, CancelFlag, GenerationConfig,
    GenerationOutput, SpeculativeStats,
};
use candle_llm::device::select_device;
use candle_llm::models::{Qwen35Config, Qwen35Model, Qwen35Mtp};
use candle_llm::primitives::{input_ids, Weights};
use core_llm::{ChatTemplate, JinjaChatTemplate, Message, RenderOptions, Tokenizer};
use serde_json::{json, Value};

// >>> head-only
use candle_llm::decode::generate_step_timed;
use candle_llm::decode::StepModel;
use candle_llm::primitives::host_sync_count;

fn host_syncs_now() -> Option<u64> {
    Some(host_sync_count())
}

/// The `StepModel` row: the same greedy loop as the reference, driven through the seam.
fn step_model_row(
    model: &Qwen35Model,
    prompt: &[i32],
    config: &GenerationConfig,
    device: &Device,
) -> (GenerationOutput, f64, f64, Option<(u64, u64)>, Option<u64>) {
    let mut prefill_secs = 0.0;
    let started = Instant::now();
    let mut decode_started = None;
    let mut boundary = || {
        device.synchronize()?;
        prefill_secs = started.elapsed().as_secs_f64();
        decode_started = Some(Instant::now());
        Ok(())
    };
    let (out, record) = generate_step_timed(
        model,
        prompt,
        config,
        &CancelFlag::new(),
        &mut |_| {},
        None,
        Some(&mut boundary),
    )
    .expect("step_model generation");
    device.synchronize().unwrap();
    let decode_secs = decode_started.unwrap().elapsed().as_secs_f64();
    // The cache's own accounting for a full-length request, from a fresh replay of the prompt.
    let mut cache = StepModel::new_cache(model);
    model
        .step(&mut cache, candle_llm::decode::StepRequest::last(prompt))
        .unwrap();
    let cache_bytes = cache.memory().total_bytes() as u64;
    (
        out,
        prefill_secs,
        decode_secs,
        Some((record.target_forwards, record.host_syncs)),
        Some(cache_bytes),
    )
}
// <<< head-only

/// What `decode_bench.py baseline-source` substitutes for the head-only block so the file compiles
/// against the pre-epic baseline. Kept here (next to the block it replaces) so a reviewer sees both.
#[allow(dead_code)]
const BASELINE_STUB: &str = r#"
fn host_syncs_now() -> Option<u64> {
    None
}

fn step_model_row(
    _model: &Qwen35Model,
    _prompt: &[i32],
    _config: &GenerationConfig,
    _device: &Device,
) -> (GenerationOutput, f64, f64, Option<(u64, u64)>, Option<u64>) {
    unreachable!("the step_model row is not available on the pre-epic baseline")
}
"#;

const DEFAULT_LABEL: &str = "RTX Pro 6000 / sm_120";
const DEFAULT_PROMPT: &str = "Write a detailed, multi-paragraph explanation of how transformer \
    language models generate text. Cover tokenization, self-attention, the key/value cache, and \
    greedy versus sampled decoding, and finish with the trade-offs of speculative decoding.";

fn env_or(name: &str, default: &str) -> String {
    std::env::var(name)
        .ok()
        .filter(|v| !v.trim().is_empty())
        .unwrap_or_else(|| default.to_string())
}

fn required(name: &str) -> String {
    std::env::var(name)
        .ok()
        .filter(|v| !v.trim().is_empty())
        .unwrap_or_else(|| panic!("{name} must be set"))
}

fn device_used_bytes(device: &Device) -> Option<u64> {
    #[cfg(feature = "cuda")]
    {
        device.synchronize().ok()?;
        let cuda = device.as_cuda_device().ok()?;
        let (free, total) = cuda.cuda_stream().context().mem_get_info().ok()?;
        return Some((total - free) as u64);
    }
    #[allow(unreachable_code)]
    {
        let _ = device;
        None
    }
}

fn greedy_config(new_tokens: usize) -> GenerationConfig {
    let mut config = GenerationConfig {
        max_new_tokens: new_tokens,
        seed: Some(0),
        stop_tokens: Vec::new(),
        ..Default::default()
    };
    config.sampling.temperature = 0.0;
    config
}

/// Render one user turn through the snapshot's chat template (the sidecar `chat_template.jinja`
/// modern HF layouts ship, else the key embedded in `tokenizer_config.json`) and tokenize it.
fn render_prompt(snapshot: &Path, user: &str) -> Vec<i32> {
    let tokenizer = Tokenizer::from_file(snapshot.join("tokenizer.json")).expect("tokenizer.json");
    let template = match std::fs::read_to_string(snapshot.join("chat_template.jinja")) {
        Ok(source) if !source.trim().is_empty() => JinjaChatTemplate::new(source),
        _ => JinjaChatTemplate::from_tokenizer_config_file(snapshot.join("tokenizer_config.json"))
            .expect("tokenizer_config.json chat template"),
    };
    let rendered = template
        .render_with(&[Message::user(user)], &RenderOptions::generation())
        .expect("render prompt");
    tokenizer
        .encode(&rendered, false)
        .expect("encode prompt")
        .into_iter()
        .map(|id| id as i32)
        .collect()
}

fn load(snapshot: &Path, device: &Device) -> (Qwen35Model, Option<Qwen35Mtp>) {
    let config: Value =
        serde_json::from_str(&std::fs::read_to_string(snapshot.join("config.json")).unwrap())
            .unwrap();
    let cfg = Qwen35Config::from_json(&config).expect("qwen3_5 config");
    let weights = Weights::from_dir(snapshot, device).expect("load weights");
    let prefix = if weights.contains("model.language_model.embed_tokens.weight") {
        "model.language_model"
    } else {
        "model"
    };
    let model = Qwen35Model::from_weights(&weights, prefix, cfg.clone()).expect("build model");
    let mtp = (cfg.mtp_num_hidden_layers > 0 && Qwen35Mtp::complete_in(&weights, &cfg))
        .then(|| Qwen35Mtp::from_weights_with(&weights, &model, None).expect("build mtp"));
    (model, mtp)
}

/// The pre-epic reference path: `decode_logits` prefill + the shared token-at-a-time loop.
fn reference_row(
    model: &Qwen35Model,
    prompt: &[i32],
    config: &GenerationConfig,
    device: &Device,
) -> (GenerationOutput, f64, f64) {
    device.synchronize().unwrap();
    let started = Instant::now();
    let mut cache = model.new_cache();
    let first = model
        .decode_logits(&input_ids(prompt, device).unwrap(), &mut cache, 0)
        .expect("prefill");
    device.synchronize().unwrap();
    let prefill_secs = started.elapsed().as_secs_f64();
    let decode_started = Instant::now();
    let out = generate_from_prefill(
        model,
        &mut cache,
        first,
        prompt.to_vec(),
        config,
        &CancelFlag::new(),
        &mut |_| {},
        None,
    )
    .expect("reference generation");
    device.synchronize().unwrap();
    (out, prefill_secs, decode_started.elapsed().as_secs_f64())
}

fn mtp_row(
    model: &Qwen35Model,
    mtp: &Qwen35Mtp,
    prompt: &[i32],
    config: &GenerationConfig,
    drafts: u32,
    device: &Device,
) -> (GenerationOutput, SpeculativeStats, f64, f64) {
    device.synchronize().unwrap();
    let started = Instant::now();
    let mut prefill_secs = 0.0;
    let mut decode_started = None;
    let mut boundary = || {
        device.synchronize()?;
        prefill_secs = started.elapsed().as_secs_f64();
        decode_started = Some(Instant::now());
        Ok(())
    };
    let (out, stats) = generate_qwen35_mtp_timed(
        model,
        mtp,
        prompt,
        config,
        drafts,
        &CancelFlag::new(),
        &mut |_| {},
        None,
        &mut boundary,
    )
    .expect("mtp generation");
    device.synchronize().unwrap();
    let decode_secs = decode_started.unwrap().elapsed().as_secs_f64();
    (out, stats, prefill_secs, decode_secs)
}

fn divergence(reference: &[i32], other: &[i32]) -> Option<usize> {
    reference
        .iter()
        .zip(other)
        .position(|(a, b)| a != b)
        .or_else(|| (reference.len() != other.len()).then_some(reference.len().min(other.len())))
}

#[allow(clippy::too_many_arguments)]
fn row_json(
    path: &str,
    drafts: Option<u32>,
    out: &GenerationOutput,
    reference: Option<&[i32]>,
    prefill_secs: f64,
    decode_secs: f64,
    forwards: Option<u64>,
    proposed: Option<u64>,
    accepted: Option<u64>,
    host_syncs: Option<u64>,
    device_used_after: Option<u64>,
    cache_logical_bytes: Option<u64>,
) -> Value {
    let generated = out.tokens.len() as u64;
    let ratio = |num: Option<u64>, den: u64| -> Value {
        match num {
            Some(n) if den > 0 => json!(n as f64 / den as f64),
            _ => Value::Null,
        }
    };
    let acceptance = match (accepted, proposed) {
        (Some(a), Some(p)) if p > 0 => json!(a as f64 / p as f64),
        _ => Value::Null,
    };
    let diverged = reference.map(|r| divergence(r, &out.tokens));
    json!({
        "path": path,
        "mtp_drafts": drafts,
        "generated_tokens": generated,
        "prefill_seconds": prefill_secs,
        "decode_seconds": decode_secs,
        "decode_tokens_per_second": if decode_secs > 0.0 { json!(generated as f64 / decode_secs) } else { Value::Null },
        "target_forwards": forwards,
        "proposed_tokens": proposed,
        "accepted_tokens": accepted,
        "acceptance_rate": acceptance,
        "target_forwards_per_generated_token": ratio(forwards, generated),
        "host_syncs": host_syncs,
        "host_syncs_per_token": ratio(host_syncs, generated),
        "device_used_bytes_after": device_used_after,
        "cache_logical_bytes": cache_logical_bytes,
        "tokens_match_reference": reference.map(|_| diverged.is_none()),
        "first_divergence": diverged.flatten(),
        "tokens": out.tokens,
    })
}

#[test]
#[ignore = "decode-perf suite: needs DECODE_BENCH_SNAPSHOT / DECODE_BENCH_OUTPUT and a GPU"]
fn decode_bench() {
    let snapshot = PathBuf::from(required("DECODE_BENCH_SNAPSHOT"));
    let output = PathBuf::from(required("DECODE_BENCH_OUTPUT"));
    assert!(!output.exists(), "{} already exists", output.display());
    let rows: Vec<String> = env_or("DECODE_BENCH_ROWS", "reference,step_model,mtp")
        .split(',')
        .map(|s| s.trim().to_string())
        .filter(|s| !s.is_empty())
        .collect();
    let drafts: Vec<u32> = env_or("DECODE_BENCH_DRAFTS", "1,2,3,4,5")
        .split(',')
        .map(|s| s.trim().parse().expect("draft width"))
        .collect();
    let new_tokens: usize = env_or("DECODE_BENCH_NEW_TOKENS", "256").parse().unwrap();
    let warmup_tokens: usize = env_or("DECODE_BENCH_WARMUP_TOKENS", "16").parse().unwrap();
    let label = env_or("DECODE_BENCH_LABEL", DEFAULT_LABEL);
    let prompt_text = env_or("DECODE_BENCH_PROMPT", DEFAULT_PROMPT);

    let device = select_device().expect("device");
    let device_name = if device.is_cuda() { "cuda" } else { "cpu" };
    let load_started = Instant::now();
    let (model, mtp) = load(&snapshot, &device);
    let load_secs = load_started.elapsed().as_secs_f64();
    let used_after_load = device_used_bytes(&device);
    let prompt = render_prompt(&snapshot, &prompt_text);
    let config = greedy_config(new_tokens);
    let warm = greedy_config(warmup_tokens);
    let mut peak_used = used_after_load;
    let mut note_peak = |used: Option<u64>| {
        if let (Some(p), Some(u)) = (peak_used, used) {
            peak_used = Some(p.max(u));
        } else if peak_used.is_none() {
            peak_used = used;
        }
        used
    };

    let mut rows_json = Vec::new();
    let mut reference_tokens: Option<Vec<i32>> = None;

    if rows.iter().any(|r| r == "reference") {
        reference_row(&model, &prompt, &warm, &device);
        let syncs0 = host_syncs_now();
        let (out, prefill, decode) = reference_row(&model, &prompt, &config, &device);
        let syncs = host_syncs_now().zip(syncs0).map(|(a, b)| a - b);
        let used = note_peak(device_used_bytes(&device));
        eprintln!(
            "[decode_bench] reference        {:>7.2} tok/s  prefill {:.3}s",
            out.tokens.len() as f64 / decode,
            prefill
        );
        rows_json.push(row_json(
            "reference",
            None,
            &out,
            None,
            prefill,
            decode,
            Some(out.tokens.len() as u64),
            None,
            None,
            syncs,
            used,
            None,
        ));
        reference_tokens = Some(out.tokens);
    }

    if rows.iter().any(|r| r == "step_model") {
        step_model_row(&model, &prompt, &warm, &device);
        let (out, prefill, decode, record, cache_bytes) =
            step_model_row(&model, &prompt, &config, &device);
        let used = note_peak(device_used_bytes(&device));
        eprintln!(
            "[decode_bench] step_model       {:>7.2} tok/s  prefill {:.3}s",
            out.tokens.len() as f64 / decode,
            prefill
        );
        let (forwards, syncs) = match record {
            Some((forwards, syncs)) => (Some(forwards), Some(syncs)),
            None => (None, None),
        };
        rows_json.push(row_json(
            "step_model",
            None,
            &out,
            reference_tokens.as_deref(),
            prefill,
            decode,
            forwards,
            None,
            None,
            syncs,
            used,
            cache_bytes,
        ));
        if reference_tokens.is_none() {
            reference_tokens = Some(out.tokens);
        }
    }

    if rows.iter().any(|r| r == "mtp") {
        let mtp = mtp.as_ref().expect("snapshot carries a complete MTP head");
        for &k in &drafts {
            mtp_row(&model, mtp, &prompt, &warm, k, &device);
            let syncs0 = host_syncs_now();
            let (out, stats, prefill, decode) = mtp_row(&model, mtp, &prompt, &config, k, &device);
            let syncs = host_syncs_now().zip(syncs0).map(|(a, b)| a - b);
            let used = note_peak(device_used_bytes(&device));
            eprintln!(
                "[decode_bench] mtp K={k}          {:>7.2} tok/s  accept {:.3}  fwd/tok {:.3}",
                out.tokens.len() as f64 / decode,
                if stats.proposed > 0 {
                    stats.accepted as f64 / stats.proposed as f64
                } else {
                    0.0
                },
                stats.forwards as f64 / out.tokens.len().max(1) as f64
            );
            rows_json.push(row_json(
                "mtp",
                Some(k),
                &out,
                reference_tokens.as_deref(),
                prefill,
                decode,
                Some(stats.forwards as u64),
                Some(stats.proposed as u64),
                Some(stats.accepted as u64),
                syncs,
                used,
                None,
            ));
        }
    }

    let mut doc = BTreeMap::new();
    doc.insert("schema_version", json!(1));
    doc.insert("suite", json!("decode_bench"));
    doc.insert("label", json!(label));
    doc.insert("snapshot", json!(snapshot.display().to_string()));
    doc.insert("device", json!(device_name));
    doc.insert(
        "compute_dtype",
        json!(format!("{:?}", model.compute_dtype())),
    );
    doc.insert("load_seconds", json!(load_secs));
    doc.insert("prompt_tokens", json!(prompt.len()));
    doc.insert("new_tokens", json!(new_tokens));
    doc.insert("warmup_tokens", json!(warmup_tokens));
    doc.insert("device_used_bytes_after_load", json!(used_after_load));
    doc.insert("peak_device_used_bytes", json!(peak_used));
    doc.insert(
        "device_memory_scope",
        json!("cuMemGetInfo total-free on the selected device: device-wide, includes co-tenants"),
    );
    doc.insert("rows", json!(rows_json));
    if let Some(parent) = output.parent() {
        std::fs::create_dir_all(parent).unwrap();
    }
    std::fs::write(&output, serde_json::to_string_pretty(&doc).unwrap()).unwrap();
    eprintln!("[decode_bench] wrote {}", output.display());
}
