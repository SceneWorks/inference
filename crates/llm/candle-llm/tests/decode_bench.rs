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
//! | `DECODE_BENCH_ROWS`        | comma list of `reference`, `reference_unfused`, `reference_cublaslt`, `step_model`, `mtp` (default `reference,step_model,mtp`) |
//! | `DECODE_BENCH_FORMAT`      | projection weight format: `bf16` (default) or `nvfp4` (quantized at load, sc-24135) |
//! | `DECODE_BENCH_DRAFTS`      | MTP draft widths, comma list (default `1,2,3,4,5`)           |
//! | `DECODE_BENCH_NEW_TOKENS`  | tokens generated per row (default 256)                       |
//! | `DECODE_BENCH_WARMUP_TOKENS` | tokens of the untimed warm-up run (default 16)             |
//! | `DECODE_BENCH_PROMPT`      | user message (default: a long-answer explanation request)    |
//! | `DECODE_BENCH_LABEL`       | hardware label recorded verbatim (default `RTX Pro 6000 / sm_120`) |
//! | `DECODE_BENCH_KV_CACHE`    | `static` (default) or `growing`: the `step_model` row's KV cache  |
//! | `DECODE_BENCH_ATTN`        | `gqa` (default) or `expanded`: the growing slots' attention       |
//!
//! Rows are greedy (`temperature = 0`), no stop tokens, so every row emits exactly `NEW_TOKENS`
//! and the token sequences are comparable: each row records whether it matched the reference row
//! token-for-token and, if not, the first divergence. The `reference_unfused` row (sc-24137) is
//! the reference loop with the fused decode primitives switched **off** for that row only, so one
//! document holds the fused-on vs fused-off token identity and tok/s; every row also records its
//! fused-vs-reference primitive tally (`fused_primitives`) and the document records the switch.
//! With `DECODE_BENCH_FORMAT=nvfp4` every row also records which NVFP4 projection path ran
//! (`nvfp4_projections`: fused decode GEMV vs cuBLASLt W4A4 calls and the last cuBLASLt reason,
//! sc-24136) and the document records the GEMV switch (`nvfp4_gemv`); the `reference_cublaslt`
//! row is the reference loop with the GEMV switched **off** for that row only, so one document
//! holds the GEMV-on vs cuBLASLt tok/s. Timing brackets the *decode* phase only
//! (prefill is reported separately) with a device synchronize on both sides.
//!
//! Memory: `device_used_bytes_at_last_token` is `cuMemGetInfo` total-free sampled from the row's
//! stream callback at its last generated token, while that row's cache is still alive (device-wide,
//! so it includes the weights and any co-tenant); `peak_device_used_bytes` is the maximum over the
//! rows and the post-load sample. The `step_model` row also reports its **final** cache's own
//! accounting (`cache_live_bytes`, `cache_checkpoint_bytes`) - the rollback checkpoints included -
//! returned by the step driver.
//!
//! Every row on head says which KV cache and which attention formulation produced it
//! (sc-24132): `kv_cache` is `static` for the preallocated cache (the `step_model` row's default;
//! `DECODE_BENCH_KV_CACHE=growing` selects the `AttnKv` reference slots through the same driver)
//! or `growing` (the reference and MTP rows always); `attn_formulation` is `gqa` (the un-expanded
//! `sdpa_gqa_causal` every path runs since S4) or `expanded` (`DECODE_BENCH_ATTN=expanded`: the
//! pre-S4 `repeat_kv` arithmetic on the growing slots, the labelled comparison row that reproduces
//! the sealed pre-epic baseline's bits). The pre-epic baseline binary reports neither (`null`).
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
    generate_from_prefill, generate_qwen35_mtp_timed, CancelFlag, Decode, GenerationConfig,
    GenerationOutput, SpeculativeStats, StreamEvent,
};
use candle_llm::device::select_device;
use candle_llm::models::{Qwen35Config, Qwen35Model, Qwen35Mtp};
use candle_llm::primitives::{input_ids, Weights};
use core_llm::{ChatTemplate, JinjaChatTemplate, Message, RenderOptions, Tokenizer};
use serde_json::{json, Value};

// >>> head-only
use candle_llm::decode::generate_step_timed;
use candle_llm::decode::CountingDecode;
use candle_llm::primitives::{
    fused_kernels_enabled, fused_tally, host_sync_count, nvfp4_gemv_enabled,
    nvfp4_gemv_policy_guard, nvfp4_path_tally, set_fused_kernels, ProjectionFormat,
};

fn host_syncs_now() -> Option<u64> {
    Some(host_sync_count())
}

/// The fused-primitive switch state (`on` / `off`), or `None` on a binary without the switch.
fn fused_switch() -> Option<&'static str> {
    Some(if fused_kernels_enabled() { "on" } else { "off" })
}

/// A snapshot of the thread's fused-vs-reference primitive tally (sc-24137), or `None` on a
/// binary without the fused primitives; `fused_delta` turns two snapshots into a row's JSON.
fn fused_tally_now() -> Option<candle_llm::primitives::FusedTally> {
    Some(fused_tally())
}

fn fused_delta(
    before: Option<candle_llm::primitives::FusedTally>,
    switch: Option<&'static str>,
) -> Option<Value> {
    let d = fused_tally_now()?.since(&before?);
    Some(json!({
        "switch": switch,
        "fused": d.fused,
        "reference": d.reference,
        "reference_reason": d.reference_reason,
        "path": d.label(),
    }))
}

/// Run `f` with the fused primitives switched off, restoring the previous policy afterwards.
fn with_fused_off<T>(f: impl FnOnce() -> T) -> T {
    let was = fused_kernels_enabled();
    set_fused_kernels(Some(false));
    let out = f();
    set_fused_kernels(Some(was));
    out
}

/// The NVFP4 decode-GEMV switch state (`on` / `off`), or `None` on a binary without it.
fn nvfp4_gemv_switch() -> Option<&'static str> {
    Some(if nvfp4_gemv_enabled() { "on" } else { "off" })
}

/// A snapshot of the thread's NVFP4 projection path tally (sc-24136), or `None` on a binary
/// without it; `nvfp4_delta` turns two snapshots into a row's JSON.
fn nvfp4_tally_now() -> Option<candle_llm::primitives::Nvfp4PathTally> {
    Some(nvfp4_path_tally())
}

fn nvfp4_delta(
    before: Option<candle_llm::primitives::Nvfp4PathTally>,
    switch: Option<&'static str>,
) -> Option<Value> {
    let d = nvfp4_tally_now()?.since(&before?);
    Some(json!({
        "switch": switch,
        "gemv": d.gemv,
        "cublaslt": d.cublaslt,
        "cublaslt_reason": d.cublaslt_reason,
        "path": d.label(),
    }))
}

/// Run `f` with the NVFP4 decode GEMV switched off (every NVFP4 projection on cuBLASLt),
/// restoring the previous policy afterwards. Uses the process-wide policy guard so a prior
/// env-derived (`None`) policy comes back as `None`, not pinned to whatever `Some(bool)` state
/// this call happened to observe.
fn with_nvfp4_gemv_off<T>(f: impl FnOnce() -> T) -> T {
    let _guard = nvfp4_gemv_policy_guard(Some(false));
    f()
}

/// Build the target (and the MTP head, when the snapshot carries one) in the requested
/// projection format: `bf16` keeps the checkpoint's dense projections, `nvfp4` quantizes them at
/// load (sc-24135).
fn build_model(
    weights: &Weights,
    prefix: &str,
    cfg: Qwen35Config,
    format: &str,
    device: &Device,
) -> (Qwen35Model, Option<Qwen35Mtp>) {
    let format = match format {
        "bf16" => None,
        "nvfp4" => Some(ProjectionFormat::nvfp4(device).expect("NVFP4 capability")),
        other => panic!("DECODE_BENCH_FORMAT must be bf16 or nvfp4, got {other}"),
    };
    let model = Qwen35Model::from_weights_format(weights, prefix, cfg.clone(), format.as_ref())
        .expect("build model");
    let mtp = (cfg.mtp_num_hidden_layers > 0 && Qwen35Mtp::complete_in(weights, &cfg)).then(|| {
        Qwen35Mtp::from_weights_format(weights, &model, format.as_ref()).expect("build mtp")
    });
    (model, mtp)
}

/// The reference row with its target forwards **measured**: the loop runs through a
/// `CountingDecode` and the direct `decode_logits` prefill is noted as one external forward.
fn reference_row(
    model: &Qwen35Model,
    prompt: &[i32],
    config: &GenerationConfig,
    device: &Device,
    on_event: &mut dyn FnMut(StreamEvent),
) -> (GenerationOutput, f64, f64, Option<u64>) {
    let counted = CountingDecode::new(model);
    let (out, prefill, decode) = run_reference(model, &counted, prompt, config, device, on_event);
    counted.note_external_forward(); // the `decode_logits` prefill `run_reference` issues directly
    (out, prefill, decode, Some(counted.forwards()))
}

/// The `StepModel` row: the same greedy loop as the reference, driven through the seam. Returns
/// the record's `(target_forwards, host_syncs)` and the **final** cache's `(live, checkpoint)`
/// bytes - the state the timed request held at its last step.
#[allow(clippy::type_complexity)]
fn step_model_row(
    model: &Qwen35Model,
    prompt: &[i32],
    config: &GenerationConfig,
    device: &Device,
    on_event: &mut dyn FnMut(StreamEvent),
) -> (
    GenerationOutput,
    f64,
    f64,
    Option<(u64, u64)>,
    Option<(u64, u64)>,
    Option<(&'static str, &'static str)>,
) {
    let mut prefill_secs = 0.0;
    let started = Instant::now();
    let mut decode_started = None;
    let mut boundary = || {
        device.synchronize()?;
        prefill_secs = started.elapsed().as_secs_f64();
        decode_started = Some(Instant::now());
        Ok(())
    };
    let (out, record, memory) = generate_step_timed(
        model,
        prompt,
        config,
        &CancelFlag::new(),
        on_event,
        None,
        Some(&mut boundary),
    )
    .expect("step_model generation");
    device.synchronize().unwrap();
    let decode_secs = decode_started.unwrap().elapsed().as_secs_f64();
    (
        out,
        prefill_secs,
        decode_secs,
        Some((record.target_forwards, record.host_syncs)),
        Some((memory.live_bytes as u64, memory.checkpoint_bytes as u64)),
        Some((record.kv_cache.label(), record.attn_formulation.label())),
    )
}

/// `DECODE_BENCH_KV_CACHE` (`static` by default; `growing` runs the `AttnKv` reference slots
/// through the same driver) selects which KV cache the `step_model` row builds (sc-24132).
fn select_step_kv_cache(model: &mut Qwen35Model) {
    use candle_llm::primitives::KvCacheKind;
    let kind = match env_or("DECODE_BENCH_KV_CACHE", "static").as_str() {
        "static" => KvCacheKind::Static,
        "growing" => KvCacheKind::Growing,
        other => panic!("DECODE_BENCH_KV_CACHE must be `static` or `growing`, got {other:?}"),
    };
    model.set_step_kv_cache(kind);
}

/// `DECODE_BENCH_ATTN` (`gqa` by default; `expanded` selects the pre-S4 `repeat_kv` + `sdpa`
/// arithmetic on the growing `AttnKv` slots) selects how the reference / growing paths attend
/// (sc-24132). The static cache always attends un-expanded.
fn select_attn_formulation(model: &mut Qwen35Model) {
    use candle_llm::primitives::AttnFormulation;
    let formulation = match env_or("DECODE_BENCH_ATTN", "gqa").as_str() {
        "gqa" => AttnFormulation::Gqa,
        "expanded" => AttnFormulation::Expanded,
        other => panic!("DECODE_BENCH_ATTN must be `gqa` or `expanded`, got {other:?}"),
    };
    model.set_attn_formulation(formulation);
}

/// The `(kv_cache, attn_formulation)` labels of a row decoded on the model's own growing cache —
/// the reference loop and the MTP loop (`new_cache` / `make_cache`, always the `AttnKv` slots).
fn growing_row_kinds(model: &Qwen35Model) -> Option<(&'static str, &'static str)> {
    Some(("growing", model.attn_formulation().label()))
}
// <<< head-only

/// What `decode_bench.py baseline-source` substitutes for the head-only block so the file compiles
/// against the pre-epic baseline. Kept here (next to the block it replaces) so a reviewer sees both.
#[allow(dead_code)]
const BASELINE_STUB: &str = r#"
fn host_syncs_now() -> Option<u64> {
    None
}

fn fused_switch() -> Option<&'static str> {
    None
}

fn fused_tally_now() -> Option<()> {
    None
}

fn fused_delta(_before: Option<()>, _switch: Option<&'static str>) -> Option<Value> {
    None
}

fn with_fused_off<T>(f: impl FnOnce() -> T) -> T {
    f()
}

fn nvfp4_gemv_switch() -> Option<&'static str> {
    None
}

fn nvfp4_tally_now() -> Option<()> {
    None
}

fn nvfp4_delta(_before: Option<()>, _switch: Option<&'static str>) -> Option<Value> {
    None
}

fn with_nvfp4_gemv_off<T>(f: impl FnOnce() -> T) -> T {
    f()
}

fn build_model(
    weights: &Weights,
    prefix: &str,
    cfg: Qwen35Config,
    format: &str,
    _device: &Device,
) -> (Qwen35Model, Option<Qwen35Mtp>) {
    assert_eq!(format, "bf16", "the pre-epic baseline has no NVFP4 projections");
    let model = Qwen35Model::from_weights(weights, prefix, cfg.clone()).expect("build model");
    let mtp = (cfg.mtp_num_hidden_layers > 0 && Qwen35Mtp::complete_in(weights, &cfg))
        .then(|| Qwen35Mtp::from_weights_with(weights, &model, None).expect("build mtp"));
    (model, mtp)
}

fn reference_row(
    model: &Qwen35Model,
    prompt: &[i32],
    config: &GenerationConfig,
    device: &Device,
    on_event: &mut dyn FnMut(StreamEvent),
) -> (GenerationOutput, f64, f64, Option<u64>) {
    let (out, prefill, decode) = run_reference(model, model, prompt, config, device, on_event);
    (out, prefill, decode, None)
}

#[allow(clippy::type_complexity)]
fn step_model_row(
    _model: &Qwen35Model,
    _prompt: &[i32],
    _config: &GenerationConfig,
    _device: &Device,
    _on_event: &mut dyn FnMut(StreamEvent),
) -> (
    GenerationOutput,
    f64,
    f64,
    Option<(u64, u64)>,
    Option<(u64, u64)>,
    Option<(&'static str, &'static str)>,
) {
    unreachable!("the step_model row is not available on the pre-epic baseline")
}

fn select_step_kv_cache(_model: &mut Qwen35Model) {}

fn select_attn_formulation(_model: &mut Qwen35Model) {}

fn growing_row_kinds(_model: &Qwen35Model) -> Option<(&'static str, &'static str)> {
    None
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

fn load(snapshot: &Path, device: &Device, format: &str) -> (Qwen35Model, Option<Qwen35Mtp>) {
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
    build_model(&weights, prefix, cfg, format, device)
    // `weights` drops here: tensors the model did not keep (the bf16 NVFP4 sources) are freed.
}

/// The pre-epic reference path: `decode_logits` prefill + the shared token-at-a-time loop, driven
/// through `decoder` (the model itself, or a counting wrapper around it on head).
fn run_reference(
    model: &Qwen35Model,
    decoder: &dyn Decode,
    prompt: &[i32],
    config: &GenerationConfig,
    device: &Device,
    on_event: &mut dyn FnMut(StreamEvent),
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
        decoder,
        &mut cache,
        first,
        prompt.to_vec(),
        config,
        &CancelFlag::new(),
        on_event,
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
    on_event: &mut dyn FnMut(StreamEvent),
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
        on_event,
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
    device_used_at_last_token: Option<u64>,
    cache: Option<(u64, u64)>,
    kinds: Option<(&str, &str)>,
    fused_primitives: Option<Value>,
    nvfp4_projections: Option<Value>,
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
    let matches = diverged.map(|d| d.is_none());
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
        "device_used_bytes_at_last_token": device_used_at_last_token,
        "cache_live_bytes": cache.map(|(live, _)| live),
        "cache_checkpoint_bytes": cache.map(|(_, checkpoint)| checkpoint),
        "kv_cache": kinds.map(|(kv_cache, _)| kv_cache),
        "attn_formulation": kinds.map(|(_, attn)| attn),
        "fused_primitives": fused_primitives,
        "nvfp4_projections": nvfp4_projections,
        "tokens_match_reference": matches,
        "first_divergence": diverged.flatten(),
        "tokens": out.tokens,
    })
}

/// Checks the `step_model` row's final-cache accounting: once a request has taken a single-token
/// step after the prefill (`new_tokens >= 2`), the step-seam cache holds a real rollback
/// checkpoint, so `checkpoint_bytes == 0` would mean the bench is reading a cache that never
/// decoded (the E6 bytes it exists to report would be missing).
fn checked_step_cache(new_tokens: usize, cache: Option<(u64, u64)>) -> Option<(u64, u64)> {
    if let Some((live, checkpoint)) = cache {
        assert!(live > 0, "step_model cache reports no live bytes");
        if new_tokens >= 2 {
            assert!(
                checkpoint > 0,
                "step_model cache reports no checkpoint bytes after {new_tokens} tokens"
            );
        }
    }
    cache
}

/// A stream callback that samples device memory at the row's last generated token, while the row's
/// cache is still alive (the device synchronize it implies lands at the end of the timed decode).
fn last_token_sampler<'a>(
    device: &'a Device,
    new_tokens: usize,
    slot: &'a mut Option<u64>,
) -> impl FnMut(StreamEvent) + 'a {
    move |event| {
        if let StreamEvent::Token { step, .. } = event {
            if step + 1 == new_tokens {
                *slot = device_used_bytes(device);
            }
        }
    }
}

#[test]
fn step_cache_accepts_checkpoint_bytes_and_single_token_runs() {
    assert_eq!(checked_step_cache(2, Some((10, 5))), Some((10, 5)));
    assert_eq!(checked_step_cache(1, Some((10, 0))), Some((10, 0)));
    assert_eq!(checked_step_cache(256, None), None);
}

#[test]
#[should_panic(expected = "no checkpoint bytes after 2 tokens")]
fn step_cache_without_checkpoint_bytes_after_two_tokens_is_refused() {
    checked_step_cache(2, Some((10, 0)));
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
    let format = env_or("DECODE_BENCH_FORMAT", "bf16");

    let device = select_device().expect("device");
    let device_name = if device.is_cuda() { "cuda" } else { "cpu" };
    let load_started = Instant::now();
    let (mut model, mut mtp) = load(&snapshot, &device, &format);
    select_step_kv_cache(&mut model);
    select_attn_formulation(&mut model);
    if let Some(mtp) = mtp.as_mut() {
        mtp.set_attn_formulation(model.attn_formulation());
    }
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
    let switch = fused_switch();
    let nv_switch = nvfp4_gemv_switch();

    if rows.iter().any(|r| r == "reference") {
        reference_row(&model, &prompt, &warm, &device, &mut |_| {});
        let syncs0 = host_syncs_now();
        let fused0 = fused_tally_now();
        let nv0 = nvfp4_tally_now();
        let mut at_last = None;
        let (out, prefill, decode, forwards) = reference_row(
            &model,
            &prompt,
            &config,
            &device,
            &mut last_token_sampler(&device, new_tokens, &mut at_last),
        );
        let syncs = host_syncs_now().zip(syncs0).map(|(a, b)| a - b);
        let fused = fused_delta(fused0, switch);
        let nvfp4 = nvfp4_delta(nv0, nv_switch);
        let used = note_peak(at_last);
        eprintln!(
            "[decode_bench] reference        {:>7.2} tok/s  prefill {:.3}s  fused {}  nvfp4 {}",
            out.tokens.len() as f64 / decode,
            prefill,
            fused.as_ref().map_or("n/a".to_string(), |f| f.to_string()),
            nvfp4.as_ref().map_or("n/a".to_string(), |f| f.to_string())
        );
        rows_json.push(row_json(
            "reference",
            None,
            &out,
            None,
            prefill,
            decode,
            forwards,
            None,
            None,
            syncs,
            used,
            None,
            growing_row_kinds(&model),
            fused,
            nvfp4,
        ));
        reference_tokens = Some(out.tokens);
    }

    if rows.iter().any(|r| r == "reference_unfused") {
        with_fused_off(|| reference_row(&model, &prompt, &warm, &device, &mut |_| {}));
        let syncs0 = host_syncs_now();
        let fused0 = fused_tally_now();
        let nv0 = nvfp4_tally_now();
        let mut at_last = None;
        let (out, prefill, decode, forwards) = with_fused_off(|| {
            reference_row(
                &model,
                &prompt,
                &config,
                &device,
                &mut last_token_sampler(&device, new_tokens, &mut at_last),
            )
        });
        let syncs = host_syncs_now().zip(syncs0).map(|(a, b)| a - b);
        let fused = fused_delta(fused0, Some("off"));
        let nvfp4 = nvfp4_delta(nv0, nv_switch);
        let used = note_peak(at_last);
        eprintln!(
            "[decode_bench] reference_unfused {:>6.2} tok/s  prefill {:.3}s  fused {}",
            out.tokens.len() as f64 / decode,
            prefill,
            fused.as_ref().map_or("n/a".to_string(), |f| f.to_string())
        );
        rows_json.push(row_json(
            "reference_unfused",
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
            None,
            growing_row_kinds(&model),
            fused,
            nvfp4,
        ));
        if reference_tokens.is_none() {
            reference_tokens = Some(out.tokens);
        }
    }

    if rows.iter().any(|r| r == "reference_cublaslt") {
        with_nvfp4_gemv_off(|| reference_row(&model, &prompt, &warm, &device, &mut |_| {}));
        let syncs0 = host_syncs_now();
        let fused0 = fused_tally_now();
        let nv0 = nvfp4_tally_now();
        let mut at_last = None;
        let (out, prefill, decode, forwards) = with_nvfp4_gemv_off(|| {
            reference_row(
                &model,
                &prompt,
                &config,
                &device,
                &mut last_token_sampler(&device, new_tokens, &mut at_last),
            )
        });
        let syncs = host_syncs_now().zip(syncs0).map(|(a, b)| a - b);
        let fused = fused_delta(fused0, switch);
        let nvfp4 = nvfp4_delta(nv0, nv_switch.map(|_| "off"));
        let used = note_peak(at_last);
        eprintln!(
            "[decode_bench] reference_cublaslt {:>5.2} tok/s  prefill {:.3}s  nvfp4 {}",
            out.tokens.len() as f64 / decode,
            prefill,
            nvfp4.as_ref().map_or("n/a".to_string(), |f| f.to_string())
        );
        rows_json.push(row_json(
            "reference_cublaslt",
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
            None,
            growing_row_kinds(&model),
            fused,
            nvfp4,
        ));
        if reference_tokens.is_none() {
            reference_tokens = Some(out.tokens);
        }
    }

    if rows.iter().any(|r| r == "step_model") {
        step_model_row(&model, &prompt, &warm, &device, &mut |_| {});
        let fused0 = fused_tally_now();
        let nv0 = nvfp4_tally_now();
        let mut at_last = None;
        let (out, prefill, decode, record, cache, kinds) = step_model_row(
            &model,
            &prompt,
            &config,
            &device,
            &mut last_token_sampler(&device, new_tokens, &mut at_last),
        );
        let cache = checked_step_cache(new_tokens, cache);
        let fused = fused_delta(fused0, switch);
        let nvfp4 = nvfp4_delta(nv0, nv_switch);
        let used = note_peak(at_last);
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
            cache,
            kinds,
            fused,
            nvfp4,
        ));
        if reference_tokens.is_none() {
            reference_tokens = Some(out.tokens);
        }
    }

    if rows.iter().any(|r| r == "mtp") {
        let mtp = mtp.as_ref().expect("snapshot carries a complete MTP head");
        for &k in &drafts {
            mtp_row(&model, mtp, &prompt, &warm, k, &device, &mut |_| {});
            let syncs0 = host_syncs_now();
            let fused0 = fused_tally_now();
            let nv0 = nvfp4_tally_now();
            let mut at_last = None;
            let (out, stats, prefill, decode) = mtp_row(
                &model,
                mtp,
                &prompt,
                &config,
                k,
                &device,
                &mut last_token_sampler(&device, new_tokens, &mut at_last),
            );
            let syncs = host_syncs_now().zip(syncs0).map(|(a, b)| a - b);
            let fused = fused_delta(fused0, switch);
            let nvfp4 = nvfp4_delta(nv0, nv_switch);
            let used = note_peak(at_last);
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
                growing_row_kinds(&model),
                fused,
                nvfp4,
            ));
        }
    }

    let mut doc = BTreeMap::new();
    doc.insert("schema_version", json!(2));
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
    doc.insert("fused_kernels", json!(switch));
    doc.insert("weight_format", json!(format));
    doc.insert("nvfp4_gemv", json!(nv_switch));
    doc.insert("device_used_bytes_after_load", json!(used_after_load));
    doc.insert("peak_device_used_bytes", json!(peak_used));
    doc.insert(
        "device_memory_scope",
        json!(
            "cuMemGetInfo total-free on the selected device, sampled at each row's last generated \
             token while its cache is alive: device-wide, includes weights and co-tenants"
        ),
    );
    doc.insert(
        "cache_memory_scope",
        json!(
            "step_model row: the final cache's own logical accounting after the timed run \
             (live state; rollback checkpoints counted separately)"
        ),
    );
    doc.insert("rows", json!(rows_json));
    if let Some(parent) = output.parent() {
        std::fs::create_dir_all(parent).unwrap();
    }
    std::fs::write(&output, serde_json::to_string_pretty(&doc).unwrap()).unwrap();
    eprintln!("[decode_bench] wrote {}", output.display());
}
