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
//! | `DECODE_BENCH_ROWS`        | comma list of `reference`, `reference_unfused`, `reference_cublaslt`, `step_model`, `mtp`, `ngram`, `sampled`, `sampled_step_model` (default `reference,step_model,mtp`) |
//! | `DECODE_BENCH_FORMAT`      | projection weight format, quantized at load for both families: `bf16` (default, dense), `q8` / `q4` (GGML Q8_0 / Q4_K, the `Quantize::Q8` / `Q4` load path), `nvfp4` (sc-24135; the llama family since sc-24140) |
//! | `DECODE_BENCH_SAMPLING`    | the stochastic rows' `temperature,top_p,seed` (default `0.7,0.9,0`)  |
//! | `DECODE_BENCH_DRAFTS`      | MTP draft widths, comma list (default `1,2,3,4,5`)           |
//! | `DECODE_BENCH_NGRAM_DRAFTS` | n-gram draft widths, comma list (default `3`)               |
//! | `DECODE_BENCH_NEW_TOKENS`  | tokens generated per row (default 256)                       |
//! | `DECODE_BENCH_WARMUP_TOKENS` | tokens of the untimed warm-up run (default 16)             |
//! | `DECODE_BENCH_PROMPT`      | user message (default: a long-answer explanation request)    |
//! | `DECODE_BENCH_LABEL`       | hardware label recorded verbatim (default `RTX Pro 6000 / sm_120`); the harness checks it against the probed device |
//! | `DECODE_BENCH_KV_CACHE`    | `static` (default) or `growing`: the `step_model` row's KV cache  |
//! | `DECODE_BENCH_ATTN`        | `gqa` (default) or `expanded`: the growing slots' attention       |
//! | `CANDLE_LLM_CUDA_GRAPHS`   | `1` runs the `step_model` / `mtp` / `ngram` rows through the CUDA-graph runner (sc-24134); every row records `cuda_graphs` |
//! | `CANDLE_LLM_CUDA_STREAM`   | `legacy` or `own`: the CUDA stream the model runs on (sc-24134; unset = `own` with `CANDLE_LLM_CUDA_GRAPHS` on, else `legacy`), recorded as `cuda_stream` |
//!
//! Rows are greedy (`temperature = 0`), no stop tokens, so every row emits exactly `NEW_TOKENS`
//! and the token sequences are comparable: each row records whether it matched the reference row
//! token-for-token and, if not, the first divergence. The **stochastic** rows (sc-24140) are the
//! same loops under a seeded temperature + top-p sampler (`DECODE_BENCH_SAMPLING`, recorded as the
//! row's `sampling`): `sampled` is the reference loop, `sampled_step_model` the step seam, and a
//! `sampled_step_model` row compares against the run's `sampled` row (the same seed, so the same
//! draws wherever the logits agree). Every row records its sampler telemetry (`sampler`: the path
//! — `device`, `host:<reason>` or `none` —, device / host draws and whole logits rows copied to
//! the host, per token too), the S5 measurement inside the one decode-perf home. The `reference_unfused` row (sc-24137) is
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
//! (sc-24132): `kv_cache` is `static` for the preallocated cache (the `step_model`, `mtp` and
//! `ngram` rows' default; `DECODE_BENCH_KV_CACHE=growing` selects the `AttnKv` reference slots
//! through the same seam) or `growing` (the reference row always); `attn_formulation` is `gqa`
//! (the un-expanded `sdpa_gqa_causal` every path runs since S4) or `expanded`
//! (`DECODE_BENCH_ATTN=expanded`: the pre-S4 `repeat_kv` arithmetic on the growing slots, the
//! labelled comparison row that reproduces the sealed pre-epic baseline's bits). The pre-epic
//! baseline binary reports neither (`null`).
//!
//! The speculative rows (sc-24130) run the unified engine over the step seam — the `mtp` rows
//! with the native MTP proposer, the `ngram` rows with prompt lookup — and report the proposer
//! the record says ran (`proposer`), the device->host syncs per verify step
//! (`host_syncs_per_verify_step`, the AC2 figure: exactly 1 on head, `K + 1` on the pre-epic loop,
//! which the baseline binary reports as `null` because it predates the counter), the replay
//! forwards the engine spent on the `RollbackUnavailable` fallback (`replay_forwards`, E2) and the
//! recovery split behind them (`verify_steps`, `direct_rollbacks`,
//! `target_forwards_per_verify_step` — the measured target forwards net of the prefill per verify
//! step, exactly 1.0 on the per-token DeltaNet checkpoint ring, sc-24131 AC2).
//!
//! **Run identity (sc-24140 feature-end review).** Both families record the device the rows ran
//! on as probed from the driver (`device`, `device_name`, `compute_capability` — `12.0` on
//! sm_120), the exact prompt token ids (`prompt_token_ids`) and the build's provenance
//! (`build.git_sha` / `build.git_dirty`, embedded when the binary was built with
//! `CANDLE_LLM_BUILD_PROVENANCE=1`; `null` otherwise). `decode_bench.py` refuses a document whose
//! probed device does not match the hardware label, and a head run whose binary does not name
//! the runtime SHA and a clean tree; the prompt hash and the sampling seed join the table's
//! comparison key. These helpers sit outside the head-only block, so a baseline binary records
//! them too (its build provenance is `null`: its commit predates the build script).
//!
//! **Model selection (sc-24138).** The snapshot's `config.json` picks the family: a Qwen3.5 /
//! 3.6 / 3.8 hybrid runs the rows above; a llama-family checkpoint (a `CausalLm` — Qwen3-8B, the
//! epic's second real-weight model) runs `reference` / `reference_unfused` / `reference_cublaslt`
//! / `sampled` (the `CausalLm` reference loop, E2), `step_model` / `sampled_step_model` (the step
//! seam, static KV by default) and `ngram` (prompt lookup through the unified engine), in any
//! `DECODE_BENCH_FORMAT` (sc-24140: NVFP4 included — the decode GEMV serves its ≤ 8-row
//! forwards, as for the hybrid). `mtp` is refused: the family has no MTP head. For it `DECODE_BENCH_ATTN` defaults to `gqa` as for the
//! hybrid — the model default since sc-24138, the static cache's arithmetic, so the reference rows
//! and the `step_model` row are token-identical by construction; `DECODE_BENCH_ATTN=expanded`
//! selects the pre-migration `repeat_kv` arithmetic on the reference rows (and a growing
//! `step_model` cache), the labelled "expanded attn" comparison row. The static cache attends
//! un-expanded (`gqa`) either way. The document then carries `model_family: "llama"` and its
//! `architecture`.
//!
//! The block between the `head-only` markers uses seams that do not exist on the pre-epic
//! baseline (`StepModel`, host-sync accounting). `decode_bench.py baseline-source` rewrites this
//! file into a copy that compiles against `d2b8cb335` by replacing that block with the stub in
//! `BASELINE_STUB`; the harness is what keeps the two in step. The stub carries the pre-epic
//! reference loops of **both** families (sc-24140: the `CausalLm` `decode_logits` +
//! `generate_from_prefill` reference too, so a Qwen3-8B baseline exists) in `bf16`, `q8` and `q4`
//! — the formats the pre-epic loaders already had; NVFP4 postdates it.

use std::collections::BTreeMap;
use std::path::{Path, PathBuf};
use std::time::Instant;

use candle_core::Device;
use candle_llm::decode::{
    generate_from_prefill, CancelFlag, Decode, GenerationConfig, GenerationOutput,
    SpeculativeStats, StreamEvent,
};
use candle_llm::device::select_device;
use candle_llm::models::{Qwen35Config, Qwen35Model, Qwen35Mtp};
use candle_llm::primitives::{input_ids, Weights};
use core_llm::{ChatTemplate, JinjaChatTemplate, Message, RenderOptions, Tokenizer};
use serde_json::{json, Value};

// >>> head-only
use candle_llm::decode::generate_step_timed;
use candle_llm::decode::CountingDecode;
use candle_llm::decode::{
    cuda_graphs_enabled, generate_speculative_with, graph_tally, GraphRunner, GraphTally,
    MtpProposer, NgramProposer, Proposer, RequestSpan, SpeculativePrompt, StepModel,
};
use candle_llm::models::Qwen35Cache;
use candle_llm::primitives::{
    fused_kernels_enabled, fused_policy_guard, fused_tally, host_sync_count, nvfp4_gemv_enabled,
    nvfp4_gemv_policy_guard, nvfp4_path_tally, ProjectionFormat,
};

/// The CUDA-graph switch state (`on` / `off`), or `None` on a binary without the runner.
fn cuda_graphs_switch() -> Option<&'static str> {
    Some(if cuda_graphs_enabled() { "on" } else { "off" })
}

/// The CUDA stream the model runs on (`own` / `legacy`, sc-24134), or `None` on a binary
/// without the selector.
fn cuda_stream_label() -> Option<&'static str> {
    candle_llm::device::CudaStreamKind::from_env()
        .ok()
        .map(|k| k.label())
}

/// A snapshot of the thread's CUDA-graph tally (sc-24134); `cuda_graphs_delta` turns two
/// snapshots into a row's JSON.
fn cuda_graphs_now() -> Option<GraphTally> {
    Some(graph_tally())
}

fn cuda_graphs_delta(before: Option<GraphTally>, switch: Option<&'static str>) -> Option<Value> {
    let d = cuda_graphs_now()?.since(&before?);
    Some(json!({
        "switch": switch,
        "replayed": d.replayed,
        "eager": d.eager,
        "captured": d.captured,
        "fallback_reason": d.fallback_reason,
        "path": d.label(),
    }))
}

/// The step model a row drives: the graph runner over the target when the switch is on
/// (sc-24134), the bare target otherwise.
fn stepper<'a>(
    model: &'a Qwen35Model,
    runner: &'a GraphRunner<'a, Qwen35Model>,
) -> &'a dyn StepModel<Cache = Qwen35Cache> {
    if cuda_graphs_enabled() {
        runner
    } else {
        model
    }
}

/// A speculative row through the unified engine: the output, the raw counters, the prefill and
/// decode seconds, the `(kv_cache, attn_formulation)` labels of the cache it ran on, the
/// engine's host syncs per verify step, the proposer the record says ran, and its
/// [`Recovery`] counters (sc-24131).
type SpeculativeRow = (
    GenerationOutput,
    SpeculativeStats,
    f64,
    f64,
    Option<(&'static str, &'static str)>,
    Option<f64>,
    &'static str,
    Option<Recovery>,
);

fn engine_row<P: Proposer>(
    model: &Qwen35Model,
    proposer: &mut P,
    prompt: &[i32],
    config: &GenerationConfig,
    drafts: usize,
    device: &Device,
    on_event: &mut dyn FnMut(StreamEvent),
) -> SpeculativeRow {
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
    let runner = GraphRunner::new(model);
    let run = generate_speculative_with(
        stepper(model, &runner),
        proposer,
        SpeculativePrompt::Tokens(prompt),
        config,
        drafts,
        &CancelFlag::new(),
        on_event,
        None,
        None,
        Some(&mut boundary),
    )
    .expect("speculative generation");
    device.synchronize().unwrap();
    let decode_secs = decode_started.unwrap().elapsed().as_secs_f64();
    (
        run.output,
        run.stats,
        prefill_secs,
        decode_secs,
        Some((
            run.record.kv_cache.label(),
            run.record.attn_formulation.label(),
        )),
        run.record.host_syncs_per_verify_step(),
        run.record.proposer.label(),
        Some((
            run.record.verify_steps,
            run.record.direct_rollbacks,
            run.record.replay_forwards,
            run.record.prefill_forwards,
        )),
    )
}

/// The `mtp` row: the engine with the native MTP proposer, `drafts` per verify step.
fn mtp_row(
    model: &Qwen35Model,
    mtp: &Qwen35Mtp,
    prompt: &[i32],
    config: &GenerationConfig,
    drafts: u32,
    device: &Device,
    on_event: &mut dyn FnMut(StreamEvent),
) -> SpeculativeRow {
    let mut proposer = MtpProposer::new(mtp);
    engine_row(
        model,
        &mut proposer,
        prompt,
        config,
        drafts as usize,
        device,
        on_event,
    )
}

/// The `ngram` row: the engine with prompt lookup (trailing n-grams up to 3), `drafts` per step.
fn ngram_row(
    model: &Qwen35Model,
    prompt: &[i32],
    config: &GenerationConfig,
    drafts: u32,
    device: &Device,
    on_event: &mut dyn FnMut(StreamEvent),
) -> SpeculativeRow {
    let mut proposer = NgramProposer { max_ngram: 3 };
    engine_row(
        model,
        &mut proposer,
        prompt,
        config,
        drafts as usize,
        device,
        on_event,
    )
}

fn host_syncs_now() -> Option<u64> {
    Some(host_sync_count())
}

/// The engine's replay forwards (the `RollbackUnavailable` -> replay fallback, E2), `None` on a
/// binary whose stats predate the counter.
fn replay_forwards(stats: &SpeculativeStats) -> Option<u64> {
    Some(stats.replays as u64)
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
/// Uses the process-wide policy guard so a prior env-derived (`None`) policy comes back as
/// `None`, not pinned to whatever `Some(bool)` state this call happened to observe.
fn with_fused_off<T>(f: impl FnOnce() -> T) -> T {
    let _guard = fused_policy_guard(Some(false));
    f()
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

/// A snapshot of this thread's sampler counters (sc-24133), or `None` on a binary without them;
/// `sampler_delta` turns it into a row's `sampler` JSON.
fn sampler_now() -> Option<RequestSpan> {
    Some(RequestSpan::begin())
}

/// The row's sampler telemetry since `before`: the path (`device`, `host:<reason>`, `none`), the
/// device and host draws, and the whole logits rows copied to the host — also per generated token.
fn sampler_delta(before: Option<RequestSpan>, generated: usize) -> Option<Value> {
    let t = before?.sampler();
    let per_token = |n: u64| (generated > 0).then(|| n as f64 / generated as f64);
    Some(json!({
        "path": t.label(),
        "device_draws": t.device_draws,
        "host_draws": t.host_draws,
        "logits_to_host": t.logits_to_host,
        "logits_to_host_per_token": per_token(t.logits_to_host),
        "host_draws_per_token": per_token(t.host_draws),
    }))
}

/// `DECODE_BENCH_FORMAT` as the load-time projection format both families' loaders take:
/// `bf16` keeps the checkpoint's dense projections; `q8` / `q4` are GGML Q8_0 / Q4_K (the
/// `Quantize::Q8` / `Q4` load path); `nvfp4` quantizes on-device at load (sc-24135 / sc-24140).
fn projection_format(format: &str, device: &Device) -> Option<ProjectionFormat> {
    use candle_llm::primitives::QuantSpec;
    match format {
        "bf16" => None,
        "q8" => Some(ProjectionFormat::from(QuantSpec::q8())),
        "q4" => Some(ProjectionFormat::from(QuantSpec::q4())),
        "nvfp4" => Some(ProjectionFormat::nvfp4(device).expect("NVFP4 capability")),
        other => panic!("DECODE_BENCH_FORMAT must be bf16, q8, q4 or nvfp4, got {other}"),
    }
}

/// The loaded model's weight census (bits/param by projection kind), `None` on a binary without it.
fn census_json(census: candle_llm::primitives::WeightCensus) -> Option<Value> {
    let tally = |t: &candle_llm::primitives::ProjectionTally| {
        json!({
            "count": t.count,
            "params": t.params,
            "resident_bytes": t.resident_bytes,
            "bits_per_param": t.bits_per_param(),
        })
    };
    let p = &census.projections;
    Some(json!({
        "projections": {
            "dense": tally(&p.dense),
            "ggml": tally(&p.ggml),
            "nvfp4": tally(&p.nvfp4),
            "total": tally(&p.total()),
        },
        "other_tensors": tally(&census.other),
        "total": tally(&census.total()),
    }))
}

/// The hybrid's census for the document (target + MTP head).
fn hybrid_census(model: &Qwen35Model, mtp: Option<&Qwen35Mtp>) -> Option<Value> {
    let mut census = model.weight_census();
    if let Some(mtp) = mtp {
        census.merge(&mtp.weight_census());
    }
    census_json(census)
}

/// Build the target (and the MTP head, when the snapshot carries one) in the requested
/// projection format ([`projection_format`]).
fn build_model(
    weights: &Weights,
    prefix: &str,
    cfg: Qwen35Config,
    format: &str,
    device: &Device,
) -> (Qwen35Model, Option<Qwen35Mtp>) {
    let format = projection_format(format, device);
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
    &'static str,
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
    let runner = GraphRunner::new(model);
    let (out, record, memory) = generate_step_timed(
        stepper(model, &runner),
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
        record.proposer.label(),
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

/// The MTP head attends in the target's selected formulation (sc-24132). Head-only: the pre-epic
/// baseline has neither selector (sc-24140 moved this out of the shared body, which the baseline
/// rewrite could not compile).
fn select_mtp_attn_formulation(model: &Qwen35Model, mtp: Option<&mut Qwen35Mtp>) {
    if let Some(mtp) = mtp {
        mtp.set_attn_formulation(model.attn_formulation());
    }
}
/// sc-24140: `DECODE_BENCH_FORMAT` names the four load-time formats both families take.
#[test]
fn projection_format_names_bf16_q8_q4_and_nvfp4() {
    use candle_llm::primitives::QuantSpec;
    let cpu = Device::Cpu;
    assert!(projection_format("bf16", &cpu).is_none());
    assert_eq!(
        projection_format("q8", &cpu).and_then(|f| f.ggml()),
        Some(QuantSpec::q8())
    );
    assert_eq!(
        projection_format("q4", &cpu).and_then(|f| f.ggml()),
        Some(QuantSpec::q4())
    );
    // NVFP4 is a capability: on the CPU it is the typed refusal, never a silent downgrade.
    let nvfp4 = std::panic::catch_unwind(|| projection_format("nvfp4", &Device::Cpu));
    assert!(nvfp4.is_err(), "NVFP4 on the CPU must refuse");
    let other = std::panic::catch_unwind(|| projection_format("fp8", &Device::Cpu));
    assert!(other.is_err(), "an unknown format must refuse");
}

// ---- The llama family (sc-24138): a `CausalLm` snapshot, e.g. Qwen3-8B ----------------------

/// Load a llama-family snapshot in `DECODE_BENCH_FORMAT` (every [`projection_format`], NVFP4
/// included since sc-24140 — the shared loader, `CausalLm::from_weights_format`) and apply the
/// selectors: `DECODE_BENCH_KV_CACHE` picks the step seam's cache (static by default) and
/// `DECODE_BENCH_ATTN` the reference loop's formulation — `gqa` unless the labelled pre-migration
/// comparison (`expanded`) is asked for, exactly as `select_attn_formulation` does for the hybrid.
fn causal_load(
    snapshot: &Path,
    cfg: candle_llm::config::ModelConfig,
    device: &Device,
    format: &str,
) -> candle_llm::models::CausalLm {
    use candle_llm::primitives::{AttnFormulation, KvCacheKind};
    let format = projection_format(format, device);
    let weights = Weights::from_dir(snapshot, device).expect("load weights");
    let mut model =
        candle_llm::models::CausalLm::from_weights_format(&weights, "", cfg, format.as_ref())
            .expect("build model");
    // `weights` drops here: the tensors the model did not keep (the NVFP4 sources) are freed.
    drop(weights);
    match env_or("DECODE_BENCH_KV_CACHE", "static").as_str() {
        "static" => model.set_step_kv_cache(KvCacheKind::Static),
        "growing" => model.set_step_kv_cache(KvCacheKind::Growing),
        other => panic!("DECODE_BENCH_KV_CACHE must be `static` or `growing`, got {other:?}"),
    }
    match env_or("DECODE_BENCH_ATTN", "gqa").as_str() {
        "gqa" => model.set_attn_formulation(AttnFormulation::Gqa),
        "expanded" => model.set_attn_formulation(AttnFormulation::Expanded),
        other => panic!("DECODE_BENCH_ATTN must be `gqa` or `expanded`, got {other:?}"),
    }
    model
}

/// The llama-family model's weight census for the document.
fn causal_census(model: &candle_llm::models::CausalLm) -> Option<Value> {
    census_json(model.weight_census())
}

/// The `(kv_cache, attn_formulation)` labels of a `CausalLm` reference row (its own growing
/// cache, in the selector's effective formulation).
fn causal_reference_kinds(
    model: &candle_llm::models::CausalLm,
) -> Option<(&'static str, &'static str)> {
    Some((
        "growing",
        model
            .effective_attn_formulation(model.attn_formulation())
            .label(),
    ))
}

/// The `CausalLm` reference row with its target forwards **measured**: the shared reference loop
/// through a `CountingDecode`, the direct `decode_logits` prefill noted as one external forward.
fn causal_reference_row(
    model: &candle_llm::models::CausalLm,
    prompt: &[i32],
    config: &GenerationConfig,
    device: &Device,
    on_event: &mut dyn FnMut(StreamEvent),
) -> (GenerationOutput, f64, f64, Option<u64>) {
    let counted = CountingDecode::new(model);
    let (out, prefill, decode) =
        run_causal_reference(model, &counted, prompt, config, device, on_event);
    counted.note_external_forward();
    (out, prefill, decode, Some(counted.forwards()))
}

/// One row through the step seam: the engine with `proposer` (`NoProposer` is the token-at-a-time
/// `step_model` row; `NgramProposer` the `ngram` rows). Returns the output, the counters, the
/// prefill / decode seconds, the record and the final cache's accounting.
#[allow(clippy::type_complexity)]
fn causal_engine_row<P: Proposer>(
    model: &candle_llm::models::CausalLm,
    proposer: &mut P,
    prompt: &[i32],
    config: &GenerationConfig,
    drafts: usize,
    device: &Device,
    on_event: &mut dyn FnMut(StreamEvent),
) -> (
    GenerationOutput,
    SpeculativeStats,
    f64,
    f64,
    candle_llm::decode::DecodeRecord,
    (u64, u64),
) {
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
    // The graph runner over the model when the switch is on (sc-24134), as for the hybrid rows.
    let runner = GraphRunner::new(model);
    let stepper: &dyn StepModel<Cache = candle_llm::primitives::StepKvCache> =
        if cuda_graphs_enabled() {
            &runner
        } else {
            model
        };
    let run = generate_speculative_with(
        stepper,
        proposer,
        SpeculativePrompt::Tokens(prompt),
        config,
        drafts,
        &CancelFlag::new(),
        on_event,
        None,
        None,
        Some(&mut boundary),
    )
    .expect("step-seam generation");
    device.synchronize().unwrap();
    let decode_secs = decode_started.unwrap().elapsed().as_secs_f64();
    (
        run.output,
        run.stats,
        prefill_secs,
        decode_secs,
        run.record,
        (
            run.memory.live_bytes as u64,
            run.memory.checkpoint_bytes as u64,
        ),
    )
}

/// The llama family's step-seam rows (sc-24138): `step_model` (the step seam, token at a time,
/// on `DECODE_BENCH_KV_CACHE`), `ngram` (prompt lookup through the unified engine, one row per
/// `DECODE_BENCH_NGRAM_DRAFTS` width) and `sampled_step_model` (the step seam under the seeded
/// sampler, sc-24140), each with the row's device-memory sample at its last token. A greedy row
/// compares against `reference_tokens`, a sampled one against `sampled_reference`; the first row
/// of a kind becomes that kind's reference when the reference loop was not requested.
#[allow(clippy::too_many_arguments)]
fn causal_engine_rows(
    model: &candle_llm::models::CausalLm,
    rows: &[String],
    prompt: &[i32],
    greedy: (&GenerationConfig, &GenerationConfig),
    sampled: (&GenerationConfig, &GenerationConfig),
    device: &Device,
    new_tokens: usize,
    reference_tokens: &mut Option<Vec<i32>>,
    sampled_reference: &mut Option<Vec<i32>>,
) -> Vec<(Value, Option<u64>)> {
    use candle_llm::decode::NoProposer;
    let switch = fused_switch();
    let graphs_switch = cuda_graphs_switch();
    let nv_switch = nvfp4_gemv_switch();
    let mut engine_rows: Vec<(&str, usize, bool)> = Vec::new();
    if rows.iter().any(|r| r == "step_model") {
        engine_rows.push(("step_model", 0, false));
    }
    if rows.iter().any(|r| r == "ngram") {
        for k in env_or("DECODE_BENCH_NGRAM_DRAFTS", "3").split(',') {
            engine_rows.push((
                "ngram",
                k.trim().parse().expect("n-gram draft width"),
                false,
            ));
        }
    }
    if rows.iter().any(|r| r == "sampled_step_model") {
        engine_rows.push(("sampled_step_model", 0, true));
    }
    let mut out_rows = Vec::new();
    for (row, k, sampling) in engine_rows {
        let (config, warm) = if sampling { sampled } else { greedy };
        let run = |cfg: &GenerationConfig, sink: &mut dyn FnMut(StreamEvent)| {
            if row == "ngram" {
                let mut proposer = NgramProposer { max_ngram: 3 };
                causal_engine_row(model, &mut proposer, prompt, cfg, k, device, sink)
            } else {
                causal_engine_row(model, &mut NoProposer, prompt, cfg, 0, device, sink)
            }
        };
        run(warm, &mut |_| {});
        let syncs0 = host_syncs_now();
        let fused0 = fused_tally_now();
        let graphs0 = cuda_graphs_now();
        let nv0 = nvfp4_tally_now();
        let sampler0 = sampler_now();
        let mut at_last = None;
        let (out, stats, prefill, decode, record, cache) = run(
            config,
            &mut last_token_sampler(device, new_tokens, &mut at_last),
        );
        let sampler = sampler_delta(sampler0, out.tokens.len());
        let syncs = host_syncs_now().zip(syncs0).map(|(a, b)| a - b);
        let fused = fused_delta(fused0, switch);
        let graphs = cuda_graphs_delta(graphs0, graphs_switch);
        let nvfp4 = nvfp4_delta(nv0, nv_switch);
        let per_verify = record.host_syncs_per_verify_step();
        let recovery = Some((
            record.verify_steps,
            record.direct_rollbacks,
            record.replay_forwards,
            record.prefill_forwards,
        ));
        eprintln!(
            "[decode_bench] {row} K={k:<2}       {:>7.2} tok/s  accept {:.3}  fwd/tok {:.3}  syncs/verify {}  {}  kv {}  attn {}  graphs {}  sampler {}",
            out.tokens.len() as f64 / decode,
            if stats.proposed > 0 {
                stats.accepted as f64 / stats.proposed as f64
            } else {
                0.0
            },
            stats.forwards as f64 / out.tokens.len().max(1) as f64,
            per_verify.map_or("n/a".to_string(), |s| format!("{s:.2}")),
            recovery_text(Some(stats.forwards as u64), recovery),
            record.kv_cache.label(),
            record.attn_formulation.label(),
            graphs.as_ref().map_or("n/a".to_string(), |g| g.to_string()),
            sampler.as_ref().map_or("n/a".to_string(), |s| s.to_string()),
        );
        let speculative = row == "ngram";
        let basis = if sampling {
            sampled_reference.as_deref()
        } else {
            reference_tokens.as_deref()
        };
        let value = annotate_row(
            row_json(
                row,
                speculative.then_some(k as u32),
                &out,
                basis,
                prefill,
                decode,
                Some(stats.forwards as u64),
                speculative.then_some(stats.proposed as u64),
                speculative.then_some(stats.accepted as u64),
                syncs,
                at_last,
                Some(cache),
                Some((record.kv_cache.label(), record.attn_formulation.label())),
                per_verify,
                recovery,
                fused,
                graphs,
                nvfp4,
                Some(record.proposer.label()),
                Some(stats.replays as u64),
            ),
            sampling.then_some(config),
            sampler,
        );
        out_rows.push((value, at_last));
        let slot = if sampling {
            &mut *sampled_reference
        } else {
            &mut *reference_tokens
        };
        if slot.is_none() {
            *slot = Some(out.tokens);
        }
    }
    out_rows
}
// <<< head-only

/// What `decode_bench.py baseline-source` substitutes for the head-only block so the file compiles
/// against the pre-epic baseline. Kept here (next to the block it replaces) so a reviewer sees both.
#[allow(dead_code)]
const BASELINE_STUB: &str = r#"
fn host_syncs_now() -> Option<u64> {
    None
}

fn replay_forwards(_stats: &SpeculativeStats) -> Option<u64> {
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

fn cuda_graphs_switch() -> Option<&'static str> {
    None
}

fn cuda_stream_label() -> Option<&'static str> {
    None
}

fn cuda_graphs_now() -> Option<()> {
    None
}

fn cuda_graphs_delta(_before: Option<()>, _switch: Option<&'static str>) -> Option<Value> {
    None
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

fn sampler_now() -> Option<()> {
    None
}

fn sampler_delta(_before: Option<()>, _generated: usize) -> Option<Value> {
    None
}

/// The pre-epic loaders' formats: dense or GGML Q8_0 / Q4_K quantized at load. NVFP4 postdates
/// the baseline.
fn baseline_quant(format: &str) -> Option<candle_llm::primitives::QuantSpec> {
    use candle_llm::primitives::QuantSpec;
    match format {
        "bf16" => None,
        "q8" => Some(QuantSpec::q8()),
        "q4" => Some(QuantSpec::q4()),
        other => panic!("the pre-epic baseline loads bf16, q8 or q4, not {other}"),
    }
}

fn hybrid_census(_model: &Qwen35Model, _mtp: Option<&Qwen35Mtp>) -> Option<Value> {
    None
}

fn build_model(
    weights: &Weights,
    prefix: &str,
    cfg: Qwen35Config,
    format: &str,
    _device: &Device,
) -> (Qwen35Model, Option<Qwen35Mtp>) {
    let quant = baseline_quant(format);
    let model = Qwen35Model::from_weights_with(weights, prefix, cfg.clone(), quant)
        .expect("build model");
    let mtp = (cfg.mtp_num_hidden_layers > 0 && Qwen35Mtp::complete_in(weights, &cfg))
        .then(|| Qwen35Mtp::from_weights_with(weights, &model, quant).expect("build mtp"));
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
    &'static str,
) {
    unreachable!("the step_model row is not available on the pre-epic baseline")
}

fn select_step_kv_cache(_model: &mut Qwen35Model) {}

fn select_attn_formulation(_model: &mut Qwen35Model) {}

fn growing_row_kinds(_model: &Qwen35Model) -> Option<(&'static str, &'static str)> {
    None
}

fn select_mtp_attn_formulation(_model: &Qwen35Model, _mtp: Option<&mut Qwen35Mtp>) {}

/// The pre-epic `CausalLm` load: `from_weights_with` in bf16 / q8 / q4, the model's own
/// (pre-migration) attention arithmetic, no step seam.
fn causal_load(
    snapshot: &Path,
    cfg: candle_llm::config::ModelConfig,
    device: &Device,
    format: &str,
) -> candle_llm::models::CausalLm {
    let weights = Weights::from_dir(snapshot, device).expect("load weights");
    candle_llm::models::CausalLm::from_weights_with(&weights, "", cfg, baseline_quant(format))
        .expect("build model")
}

fn causal_census(_model: &candle_llm::models::CausalLm) -> Option<Value> {
    None
}

fn causal_reference_kinds(
    _model: &candle_llm::models::CausalLm,
) -> Option<(&'static str, &'static str)> {
    None
}

/// The pre-epic `CausalLm` reference: `decode_logits` + `generate_from_prefill`, uncounted.
fn causal_reference_row(
    model: &candle_llm::models::CausalLm,
    prompt: &[i32],
    config: &GenerationConfig,
    device: &Device,
    on_event: &mut dyn FnMut(StreamEvent),
) -> (GenerationOutput, f64, f64, Option<u64>) {
    let (out, prefill, decode) =
        run_causal_reference(model, model, prompt, config, device, on_event);
    (out, prefill, decode, None)
}

#[allow(clippy::too_many_arguments)]
fn causal_engine_rows(
    _model: &candle_llm::models::CausalLm,
    rows: &[String],
    _prompt: &[i32],
    _greedy: (&GenerationConfig, &GenerationConfig),
    _sampled: (&GenerationConfig, &GenerationConfig),
    _device: &Device,
    _new_tokens: usize,
    _reference_tokens: &mut Option<Vec<i32>>,
    _sampled_reference: &mut Option<Vec<i32>>,
) -> Vec<(Value, Option<u64>)> {
    assert!(
        !rows
            .iter()
            .any(|r| matches!(r.as_str(), "step_model" | "ngram" | "sampled_step_model")),
        "the step-seam rows are not available on the pre-epic baseline"
    );
    Vec::new()
}

type SpeculativeRow = (
    GenerationOutput,
    SpeculativeStats,
    f64,
    f64,
    Option<(&'static str, &'static str)>,
    Option<f64>,
    &'static str,
    Option<Recovery>,
);

/// The pre-epic MTP loop (`generate_qwen35_mtp_timed`), which the baseline binary still carries.
fn mtp_row(
    model: &Qwen35Model,
    mtp: &Qwen35Mtp,
    prompt: &[i32],
    config: &GenerationConfig,
    drafts: u32,
    device: &Device,
    on_event: &mut dyn FnMut(StreamEvent),
) -> SpeculativeRow {
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
    let (out, stats) = candle_llm::decode::generate_qwen35_mtp_timed(
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
    // No record on the baseline: the loop is the MTP loop by construction.
    (out, stats, prefill_secs, decode_secs, None, None, "mtp", None)
}

fn ngram_row(
    _model: &Qwen35Model,
    _prompt: &[i32],
    _config: &GenerationConfig,
    _drafts: u32,
    _device: &Device,
    _on_event: &mut dyn FnMut(StreamEvent),
) -> SpeculativeRow {
    unreachable!("the ngram row is not available on the pre-epic baseline")
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

/// The device the rows ran on, probed from the driver (sc-24140 feature-end review): the CUDA
/// device's name and compute capability (`major.minor`), so the harness checks the operator's
/// hardware label against the hardware; `(None, None)` on a CPU device. Outside the head-only
/// block — it uses only what the pre-epic baseline already links — so both binaries record it.
fn device_probe(device: &Device) -> (Option<String>, Option<String>) {
    #[cfg(feature = "cuda")]
    if let Ok(cuda) = device.as_cuda_device() {
        use candle_core::cuda_backend::cudarc::driver::sys::CUdevice_attribute as Attribute;
        let stream = cuda.cuda_stream();
        let context = stream.context();
        let major = context.attribute(Attribute::CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR);
        let minor = context.attribute(Attribute::CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR);
        let capability = major
            .ok()
            .zip(minor.ok())
            .map(|(major, minor)| format!("{major}.{minor}"));
        return (context.name().ok(), capability);
    }
    #[allow(unreachable_code)]
    {
        let _ = device;
        (None, None)
    }
}

/// The source state this binary was built from (sc-24140 feature-end review): the checkout's
/// `HEAD` and whether its tree was dirty, embedded by `candle-llm`'s build script when the build
/// ran with `CANDLE_LLM_BUILD_PROVENANCE=1`; `null` without it (and on the pre-epic baseline,
/// whose commit has no build script). The harness requires a head run's to name the runtime SHA
/// and a clean tree.
fn build_provenance() -> Value {
    let non_empty = |value: Option<&'static str>| value.filter(|v| !v.is_empty());
    json!({
        "git_sha": non_empty(option_env!("CANDLE_LLM_BUILD_GIT_SHA")),
        "git_dirty": non_empty(option_env!("CANDLE_LLM_BUILD_GIT_DIRTY")).map(|v| v == "1"),
    })
}

/// The document fields both families record about where and from what the rows ran (sc-24140
/// feature-end review): the probed device, the build provenance and the exact prompt token ids —
/// which the harness hashes into the table's comparison key beside the sampling seed.
fn insert_run_identity(doc: &mut BTreeMap<&'static str, Value>, device: &Device, prompt: &[i32]) {
    let (device_name, compute_capability) = device_probe(device);
    doc.insert("device_name", json!(device_name));
    doc.insert("compute_capability", json!(compute_capability));
    doc.insert("build", build_provenance());
    doc.insert("prompt_token_ids", json!(prompt));
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

/// The pre-epic `CausalLm` reference path (E2): a `decode_logits` prefill on the model's own
/// growing cache, then the shared token-at-a-time loop driven through `decoder` (the model itself,
/// or a counting wrapper around it on head) — the same arithmetic on both binaries.
fn run_causal_reference(
    model: &candle_llm::models::CausalLm,
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

/// Whether the snapshot is a llama-family (`CausalLm`) checkpoint rather than a Qwen3.5/3.6/3.8
/// hybrid — decided from `config.json` by the same dispatch the provider uses.
fn is_causal_snapshot(snapshot: &Path) -> bool {
    let config: Value =
        serde_json::from_str(&std::fs::read_to_string(snapshot.join("config.json")).unwrap())
            .unwrap();
    !matches!(
        candle_llm::config::Architecture::from_config(&config),
        Ok(candle_llm::config::Architecture::Qwen35)
    )
}

/// The stochastic rows' sampler (sc-24140): `DECODE_BENCH_SAMPLING` = `temperature,top_p,seed`
/// (default `0.7,0.9,0`), no stop tokens, so every row still emits exactly `new_tokens`.
fn sampled_config(new_tokens: usize) -> GenerationConfig {
    let spec = env_or("DECODE_BENCH_SAMPLING", "0.7,0.9,0");
    let parts: Vec<&str> = spec.split(',').map(str::trim).collect();
    let [temperature, top_p, seed] = parts.as_slice() else {
        panic!("DECODE_BENCH_SAMPLING must be `temperature,top_p,seed`, got {spec:?}");
    };
    let mut config = greedy_config(new_tokens);
    config.sampling.temperature = temperature.parse().expect("sampling temperature");
    config.sampling.top_p = top_p.parse().expect("sampling top_p");
    config.seed = Some(seed.parse().expect("sampling seed"));
    assert!(
        config.sampling.temperature > 0.0,
        "a stochastic row needs a positive temperature"
    );
    config
}

/// A row's `sampling` knobs (`null` for a greedy row).
fn sampling_json(config: &GenerationConfig) -> Value {
    json!({
        "temperature": config.sampling.temperature,
        "top_p": config.sampling.top_p,
        "top_k": config.sampling.top_k,
        "seed": config.seed,
    })
}

/// Add the sc-24140 row dimensions to a row: its `sampling` (`null` = greedy) and its
/// `sampler` telemetry (`null` on a binary without the counters).
fn annotate_row(
    mut row: Value,
    sampling: Option<&GenerationConfig>,
    sampler: Option<Value>,
) -> Value {
    row["sampling"] = sampling.map_or(Value::Null, sampling_json);
    row["sampler"] = sampler.unwrap_or(Value::Null);
    row
}

/// The decode bench over a llama-family snapshot (sc-24138 AC2; formats, stochastic rows and the
/// pre-epic baseline since sc-24140). Rows: `reference` (the `CausalLm` reference loop, growing
/// cache, `DECODE_BENCH_ATTN` selecting its formulation on head), `reference_unfused` (fused
/// primitives off for the row), `reference_cublaslt` (NVFP4 decode GEMV off for the row),
/// `sampled` (the reference loop under the seeded sampler), and the step-seam rows of
/// `causal_engine_rows`. `mtp` is refused: the llama family has no MTP head.
#[allow(clippy::too_many_arguments)]
fn causal_decode_bench(
    snapshot: &Path,
    output: &Path,
    rows: &[String],
    new_tokens: usize,
    warmup_tokens: usize,
    label: &str,
    prompt_text: &str,
    format: &str,
) {
    assert!(
        !rows.iter().any(|r| r == "mtp"),
        "the llama family has no MTP head; use reference / reference_unfused / \
         reference_cublaslt / sampled / step_model / sampled_step_model / ngram"
    );
    let device = select_device().expect("device");
    let device_name = if device.is_cuda() { "cuda" } else { "cpu" };
    let load_started = Instant::now();
    let cfg = candle_llm::config::ModelConfig::from_dir(snapshot).expect("config.json");
    let architecture = format!("{:?}", cfg.architecture);
    let model = causal_load(snapshot, cfg, &device, format);
    let load_secs = load_started.elapsed().as_secs_f64();
    let used_after_load = device_used_bytes(&device);
    let prompt = render_prompt(snapshot, prompt_text);
    let config = greedy_config(new_tokens);
    let warm = greedy_config(warmup_tokens);
    let sampled = sampled_config(new_tokens);
    let sampled_warm = sampled_config(warmup_tokens);
    let mut peak_used = used_after_load;
    let mut note_peak = |used: Option<u64>| {
        if let (Some(p), Some(u)) = (peak_used, used) {
            peak_used = Some(p.max(u));
        } else if peak_used.is_none() {
            peak_used = used;
        }
        used
    };
    let switch = fused_switch();
    let graphs_switch = cuda_graphs_switch();
    let nv_switch = nvfp4_gemv_switch();
    let mut rows_json = Vec::new();
    let mut reference_tokens: Option<Vec<i32>> = None;
    let mut sampled_reference: Option<Vec<i32>> = None;

    // (row, fused off, NVFP4 GEMV off, sampled)
    for (row, fused_off, gemv_off, sampling) in [
        ("reference", false, false, false),
        ("reference_unfused", true, false, false),
        ("reference_cublaslt", false, true, false),
        ("sampled", false, false, true),
    ] {
        if !rows.iter().any(|r| r == row) {
            continue;
        }
        let (row_config, row_warm) = if sampling {
            (&sampled, &sampled_warm)
        } else {
            (&config, &warm)
        };
        let run = |cfg: &GenerationConfig, sink: &mut dyn FnMut(StreamEvent)| {
            if fused_off {
                with_fused_off(|| causal_reference_row(&model, &prompt, cfg, &device, sink))
            } else if gemv_off {
                with_nvfp4_gemv_off(|| causal_reference_row(&model, &prompt, cfg, &device, sink))
            } else {
                causal_reference_row(&model, &prompt, cfg, &device, sink)
            }
        };
        run(row_warm, &mut |_| {});
        let syncs0 = host_syncs_now();
        let fused0 = fused_tally_now();
        let graphs0 = cuda_graphs_now();
        let nv0 = nvfp4_tally_now();
        let sampler0 = sampler_now();
        let mut at_last = None;
        let (out, prefill, decode, forwards) = run(
            row_config,
            &mut last_token_sampler(&device, new_tokens, &mut at_last),
        );
        let sampler = sampler_delta(sampler0, out.tokens.len());
        let syncs = host_syncs_now().zip(syncs0).map(|(a, b)| a - b);
        let fused = fused_delta(fused0, if fused_off { Some("off") } else { switch });
        let graphs = cuda_graphs_delta(graphs0, graphs_switch);
        let nvfp4 = nvfp4_delta(
            nv0,
            if gemv_off {
                nv_switch.map(|_| "off")
            } else {
                nv_switch
            },
        );
        let used = note_peak(at_last);
        eprintln!(
            "[decode_bench] {row:<18} {:>7.2} tok/s  prefill {:.3}s  nvfp4 {}  sampler {}",
            out.tokens.len() as f64 / decode,
            prefill,
            nvfp4.as_ref().map_or("n/a".to_string(), |f| f.to_string()),
            sampler
                .as_ref()
                .map_or("n/a".to_string(), |s| s.to_string()),
        );
        let basis = if sampling {
            sampled_reference.as_deref()
        } else {
            reference_tokens.as_deref()
        };
        rows_json.push(annotate_row(
            row_json(
                row,
                None,
                &out,
                basis,
                prefill,
                decode,
                forwards,
                None,
                None,
                syncs,
                used,
                None,
                causal_reference_kinds(&model),
                None,
                None,
                fused,
                graphs,
                nvfp4,
                Some("none"),
                None,
            ),
            sampling.then_some(row_config),
            sampler,
        ));
        let slot = if sampling {
            &mut sampled_reference
        } else {
            &mut reference_tokens
        };
        if slot.is_none() {
            *slot = Some(out.tokens);
        }
    }

    for (value, used) in causal_engine_rows(
        &model,
        rows,
        &prompt,
        (&config, &warm),
        (&sampled, &sampled_warm),
        &device,
        new_tokens,
        &mut reference_tokens,
        &mut sampled_reference,
    ) {
        note_peak(used);
        rows_json.push(value);
    }

    let mut doc = BTreeMap::new();
    doc.insert("schema_version", json!(2));
    doc.insert("suite", json!("decode_bench"));
    doc.insert("model_family", json!("llama"));
    doc.insert("weight_format", json!(format));
    doc.insert("weight_census", json!(causal_census(&model)));
    doc.insert("architecture", json!(architecture));
    doc.insert("label", json!(label));
    doc.insert("snapshot", json!(snapshot.display().to_string()));
    doc.insert("device", json!(device_name));
    insert_run_identity(&mut doc, &device, &prompt);
    doc.insert(
        "compute_dtype",
        json!(format!("{:?}", model.compute_dtype())),
    );
    doc.insert("load_seconds", json!(load_secs));
    doc.insert("prompt_tokens", json!(prompt.len()));
    doc.insert("new_tokens", json!(new_tokens));
    doc.insert("warmup_tokens", json!(warmup_tokens));
    doc.insert("sampling", sampling_json(&sampled));
    doc.insert("fused_kernels", json!(switch));
    doc.insert("cuda_graphs", json!(graphs_switch));
    doc.insert("cuda_stream", json!(cuda_stream_label()));
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
            "step-seam rows: the final cache's own logical accounting after the timed run (a \
             static cache is its whole preallocation; the llama family keeps no checkpoints)"
        ),
    );
    doc.insert("rows", json!(rows_json));
    if let Some(parent) = output.parent() {
        std::fs::create_dir_all(parent).unwrap();
    }
    std::fs::write(output, serde_json::to_string_pretty(&doc).unwrap()).unwrap();
    eprintln!("[decode_bench] wrote {}", output.display());
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
    syncs_per_verify_step: Option<f64>,
    recovery: Option<Recovery>,
    fused_primitives: Option<Value>,
    cuda_graphs: Option<Value>,
    nvfp4_projections: Option<Value>,
    proposer: Option<&str>,
    replay_forwards: Option<u64>,
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
        "mtp_drafts": if path == "mtp" { drafts } else { None },
        "drafts": drafts,
        "proposer": proposer,
        "host_syncs_per_verify_step": syncs_per_verify_step,
        "verify_steps": recovery.map(|(v, _, _, _)| v),
        "direct_rollbacks": recovery.map(|(_, d, _, _)| d),
        "target_forwards_per_verify_step": forwards_per_verify(forwards, recovery),
        "generated_tokens": generated,
        "prefill_seconds": prefill_secs,
        "decode_seconds": decode_secs,
        "decode_tokens_per_second": if decode_secs > 0.0 { json!(generated as f64 / decode_secs) } else { Value::Null },
        "target_forwards": forwards,
        "proposed_tokens": proposed,
        "accepted_tokens": accepted,
        "acceptance_rate": acceptance,
        "replay_forwards": replay_forwards,
        "target_forwards_per_generated_token": ratio(forwards, generated),
        "host_syncs": host_syncs,
        "host_syncs_per_token": ratio(host_syncs, generated),
        "device_used_bytes_at_last_token": device_used_at_last_token,
        "cache_live_bytes": cache.map(|(live, _)| live),
        "cache_checkpoint_bytes": cache.map(|(_, checkpoint)| checkpoint),
        "kv_cache": kinds.map(|(kv_cache, _)| kv_cache),
        "attn_formulation": kinds.map(|(_, attn)| attn),
        "fused_primitives": fused_primitives,
        "cuda_graphs": cuda_graphs,
        "nvfp4_projections": nvfp4_projections,
        "tokens_match_reference": matches,
        "first_divergence": diverged.flatten(),
        "tokens": out.tokens,
    })
}

/// A speculative row's recovery counters, from the engine's record (sc-24131):
/// `(verify_steps, direct_rollbacks, replay_forwards, prefill_forwards)`.
type Recovery = (u64, u64, u64, u64);

/// Target forwards per verify step from the row's **measured** target forwards:
/// `(target_forwards - prefill_forwards) / verify_steps` — the formula of
/// `DecodeRecord::target_forwards_per_verify_step`, so a forward that is neither a verify step nor
/// a counted replay raises it (sc-24131 AC2: exactly 1.0 on the per-token checkpoint ring).
/// `None` without a verify step or a forward count.
fn forwards_per_verify(forwards: Option<u64>, recovery: Option<Recovery>) -> Option<f64> {
    match (forwards, recovery) {
        (Some(forwards), Some((verify_steps, _, _, prefill))) if verify_steps > 0 => {
            Some(forwards.saturating_sub(prefill) as f64 / verify_steps as f64)
        }
        _ => None,
    }
}

/// The recovery counters for the log line: `fwd/verify` (the AC2 figure of sc-24131) and how
/// many verify steps were recovered by a direct rollback vs a replay fallback.
fn recovery_text(forwards: Option<u64>, recovery: Option<Recovery>) -> String {
    match (forwards_per_verify(forwards, recovery), recovery) {
        (Some(per_verify), Some((_, direct, replays, _))) => {
            format!("fwd/verify {per_verify:.2}  rollbacks {direct} direct / {replays} replay")
        }
        _ => "fwd/verify n/a".to_string(),
    }
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
fn forwards_per_verify_is_measured_net_of_the_prefill() {
    // 6 forwards = 1 prefill + 4 verify steps + 1 replay.
    assert_eq!(forwards_per_verify(Some(6), Some((4, 3, 1, 1))), Some(1.25));
    // Three forwards that are neither verify steps nor counted replays raise it; the counters
    // alone (`(verify_steps + replays) / verify_steps`) would still say 1.25.
    assert_eq!(forwards_per_verify(Some(9), Some((4, 3, 1, 1))), Some(2.0));
    assert_eq!(forwards_per_verify(Some(9), Some((0, 0, 0, 1))), None);
    assert_eq!(
        recovery_text(Some(9), Some((4, 3, 1, 1))),
        "fwd/verify 2.00  rollbacks 3 direct / 1 replay"
    );
}

#[test]
fn sampled_config_is_the_seeded_temperature_top_p_triple() {
    let config = sampled_config(8);
    assert_eq!(config.max_new_tokens, 8);
    assert!(
        config.stop_tokens.is_empty(),
        "every row emits exactly new_tokens"
    );
    // The default `0.7,0.9,0` unless the environment overrides it.
    if std::env::var("DECODE_BENCH_SAMPLING").is_err() {
        assert_eq!(config.sampling.temperature, 0.7);
        assert_eq!(config.sampling.top_p, 0.9);
        assert_eq!(config.seed, Some(0));
    }
    let row = annotate_row(
        json!({"path": "sampled"}),
        Some(&config),
        Some(json!({"path": "device"})),
    );
    assert_eq!(
        row["sampling"]["temperature"],
        json!(config.sampling.temperature)
    );
    assert_eq!(row["sampling"]["seed"], json!(config.seed));
    assert_eq!(row["sampler"]["path"], "device");
    let greedy = annotate_row(json!({"path": "reference"}), None, None);
    assert!(greedy["sampling"].is_null() && greedy["sampler"].is_null());
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

/// sc-24140 feature-end review: the probe reads the device the rows run on — nothing on a CPU
/// device; a CUDA device's driver name and `major.minor` compute capability, `12.0` on the sm_120
/// lane (which `REQUIRE_SM120=1` demands).
#[test]
fn device_probe_reads_the_cuda_device_and_nothing_on_cpu() {
    assert_eq!(device_probe(&Device::Cpu), (None, None));
    let Ok(device) = select_device() else {
        return;
    };
    let (name, capability) = device_probe(&device);
    if !device.is_cuda() {
        assert_eq!((name, capability), (None, None));
        return;
    }
    let name = name.expect("a CUDA device has a name");
    assert!(!name.trim().is_empty());
    let capability = capability.expect("a CUDA device has a compute capability");
    let (major, minor) = capability.split_once('.').expect("major.minor");
    assert!(
        major.parse::<u32>().is_ok() && minor.parse::<u32>().is_ok(),
        "{capability}"
    );
    let require_sm120 = std::env::var("REQUIRE_SM120").unwrap_or_default();
    if !matches!(require_sm120.trim(), "" | "0") {
        assert_eq!(capability, "12.0", "{name}");
    }
}

/// sc-24140 feature-end review: the embedded build provenance is absent (a build without
/// `CANDLE_LLM_BUILD_PROVENANCE=1`: both fields `null`) or names this checkout's `HEAD` and
/// whether the tree was dirty.
#[test]
fn build_provenance_is_absent_or_names_the_checkout_head() {
    let build = build_provenance();
    let Some(sha) = build["git_sha"].as_str() else {
        assert_eq!(build, json!({"git_sha": null, "git_dirty": null}));
        return;
    };
    let head = std::process::Command::new("git")
        .args(["-C", env!("CARGO_MANIFEST_DIR"), "rev-parse", "HEAD"])
        .output()
        .expect("git");
    assert!(head.status.success());
    assert_eq!(sha, String::from_utf8_lossy(&head.stdout).trim());
    assert!(build["git_dirty"].is_boolean(), "{build}");
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
    if is_causal_snapshot(&snapshot) {
        causal_decode_bench(
            &snapshot,
            &output,
            &rows,
            new_tokens,
            warmup_tokens,
            &label,
            &prompt_text,
            &format,
        );
        return;
    }

    let device = select_device().expect("device");
    let device_name = if device.is_cuda() { "cuda" } else { "cpu" };
    let load_started = Instant::now();
    let (mut model, mut mtp) = load(&snapshot, &device, &format);
    select_step_kv_cache(&mut model);
    select_attn_formulation(&mut model);
    select_mtp_attn_formulation(&model, mtp.as_mut());
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
    let graphs_switch = cuda_graphs_switch();
    let nv_switch = nvfp4_gemv_switch();

    if rows.iter().any(|r| r == "reference") {
        reference_row(&model, &prompt, &warm, &device, &mut |_| {});
        let syncs0 = host_syncs_now();
        let fused0 = fused_tally_now();
        let graphs0 = cuda_graphs_now();
        let nv0 = nvfp4_tally_now();
        let sampler0 = sampler_now();
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
        let graphs = cuda_graphs_delta(graphs0, graphs_switch);
        let nvfp4 = nvfp4_delta(nv0, nv_switch);
        let used = note_peak(at_last);
        let sampler = sampler_delta(sampler0, out.tokens.len());
        eprintln!(
            "[decode_bench] reference        {:>7.2} tok/s  prefill {:.3}s  fused {}  nvfp4 {}",
            out.tokens.len() as f64 / decode,
            prefill,
            fused.as_ref().map_or("n/a".to_string(), |f| f.to_string()),
            nvfp4.as_ref().map_or("n/a".to_string(), |f| f.to_string())
        );
        rows_json.push(annotate_row(
            row_json(
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
                None,
                None,
                fused,
                graphs,
                nvfp4,
                Some("none"),
                None,
            ),
            None,
            sampler,
        ));
        reference_tokens = Some(out.tokens);
    }

    if rows.iter().any(|r| r == "reference_unfused") {
        with_fused_off(|| reference_row(&model, &prompt, &warm, &device, &mut |_| {}));
        let syncs0 = host_syncs_now();
        let fused0 = fused_tally_now();
        let graphs0 = cuda_graphs_now();
        let nv0 = nvfp4_tally_now();
        let sampler0 = sampler_now();
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
        let graphs = cuda_graphs_delta(graphs0, graphs_switch);
        let nvfp4 = nvfp4_delta(nv0, nv_switch);
        let used = note_peak(at_last);
        let sampler = sampler_delta(sampler0, out.tokens.len());
        eprintln!(
            "[decode_bench] reference_unfused {:>6.2} tok/s  prefill {:.3}s  fused {}",
            out.tokens.len() as f64 / decode,
            prefill,
            fused.as_ref().map_or("n/a".to_string(), |f| f.to_string())
        );
        rows_json.push(annotate_row(
            row_json(
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
                None,
                None,
                fused,
                graphs,
                nvfp4,
                Some("none"),
                None,
            ),
            None,
            sampler,
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
        let graphs0 = cuda_graphs_now();
        let sampler0 = sampler_now();
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
        let graphs = cuda_graphs_delta(graphs0, graphs_switch);
        let used = note_peak(at_last);
        let sampler = sampler_delta(sampler0, out.tokens.len());
        eprintln!(
            "[decode_bench] reference_cublaslt {:>5.2} tok/s  prefill {:.3}s  nvfp4 {}",
            out.tokens.len() as f64 / decode,
            prefill,
            nvfp4.as_ref().map_or("n/a".to_string(), |f| f.to_string())
        );
        rows_json.push(annotate_row(
            row_json(
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
                None,
                None,
                fused,
                graphs,
                nvfp4,
                Some("none"),
                None,
            ),
            None,
            sampler,
        ));
        if reference_tokens.is_none() {
            reference_tokens = Some(out.tokens);
        }
    }

    if rows.iter().any(|r| r == "step_model") {
        step_model_row(&model, &prompt, &warm, &device, &mut |_| {});
        let fused0 = fused_tally_now();
        let graphs0 = cuda_graphs_now();
        let nv0 = nvfp4_tally_now();
        let sampler0 = sampler_now();
        let mut at_last = None;
        let (out, prefill, decode, record, cache, kinds, proposer) = step_model_row(
            &model,
            &prompt,
            &config,
            &device,
            &mut last_token_sampler(&device, new_tokens, &mut at_last),
        );
        let cache = checked_step_cache(new_tokens, cache);
        let fused = fused_delta(fused0, switch);
        let graphs = cuda_graphs_delta(graphs0, graphs_switch);
        let nvfp4 = nvfp4_delta(nv0, nv_switch);
        let used = note_peak(at_last);
        let sampler = sampler_delta(sampler0, out.tokens.len());
        eprintln!(
            "[decode_bench] step_model       {:>7.2} tok/s  prefill {:.3}s  graphs {}",
            out.tokens.len() as f64 / decode,
            prefill,
            graphs.as_ref().map_or("n/a".to_string(), |g| g.to_string())
        );
        let (forwards, syncs) = match record {
            Some((forwards, syncs)) => (Some(forwards), Some(syncs)),
            None => (None, None),
        };
        rows_json.push(annotate_row(
            row_json(
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
                None,
                None,
                fused,
                graphs,
                nvfp4,
                Some(proposer),
                None,
            ),
            None,
            sampler,
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
            let graphs0 = cuda_graphs_now();
            let nv0 = nvfp4_tally_now();
            let sampler0 = sampler_now();
            let mut at_last = None;
            let (out, stats, prefill, decode, kinds, per_verify, proposer, recovery) = mtp_row(
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
            let graphs = cuda_graphs_delta(graphs0, graphs_switch);
            let nvfp4 = nvfp4_delta(nv0, nv_switch);
            let used = note_peak(at_last);
            let sampler = sampler_delta(sampler0, out.tokens.len());
            eprintln!(
                "[decode_bench] mtp K={k}          {:>7.2} tok/s  accept {:.3}  fwd/tok {:.3}  syncs/verify {}  {}  graphs {}",
                out.tokens.len() as f64 / decode,
                if stats.proposed > 0 {
                    stats.accepted as f64 / stats.proposed as f64
                } else {
                    0.0
                },
                stats.forwards as f64 / out.tokens.len().max(1) as f64,
                per_verify.map_or("n/a".to_string(), |s| format!("{s:.2}")),
                recovery_text(Some(stats.forwards as u64), recovery),
                graphs.as_ref().map_or("n/a".to_string(), |g| g.to_string())
            );
            rows_json.push(annotate_row(
                row_json(
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
                    kinds.or_else(|| growing_row_kinds(&model)),
                    per_verify,
                    recovery,
                    fused,
                    graphs,
                    nvfp4,
                    Some(proposer),
                    replay_forwards(&stats),
                ),
                None,
                sampler,
            ));
        }
    }

    if rows.iter().any(|r| r == "ngram") {
        let ngram_drafts: Vec<u32> = env_or("DECODE_BENCH_NGRAM_DRAFTS", "3")
            .split(',')
            .map(|s| s.trim().parse().expect("n-gram draft width"))
            .collect();
        for &k in &ngram_drafts {
            ngram_row(&model, &prompt, &warm, k, &device, &mut |_| {});
            let syncs0 = host_syncs_now();
            let fused0 = fused_tally_now();
            let graphs0 = cuda_graphs_now();
            let nv0 = nvfp4_tally_now();
            let sampler0 = sampler_now();
            let mut at_last = None;
            let (out, stats, prefill, decode, kinds, per_verify, proposer, recovery) = ngram_row(
                &model,
                &prompt,
                &config,
                k,
                &device,
                &mut last_token_sampler(&device, new_tokens, &mut at_last),
            );
            let syncs = host_syncs_now().zip(syncs0).map(|(a, b)| a - b);
            let fused = fused_delta(fused0, switch);
            let graphs = cuda_graphs_delta(graphs0, graphs_switch);
            let nvfp4 = nvfp4_delta(nv0, nv_switch);
            let used = note_peak(at_last);
            let sampler = sampler_delta(sampler0, out.tokens.len());
            eprintln!(
                "[decode_bench] ngram K={k}        {:>7.2} tok/s  accept {:.3}  fwd/tok {:.3}  syncs/verify {}  {}",
                out.tokens.len() as f64 / decode,
                if stats.proposed > 0 {
                    stats.accepted as f64 / stats.proposed as f64
                } else {
                    0.0
                },
                stats.forwards as f64 / out.tokens.len().max(1) as f64,
                per_verify.map_or("n/a".to_string(), |s| format!("{s:.2}")),
                recovery_text(Some(stats.forwards as u64), recovery)
            );
            rows_json.push(annotate_row(
                row_json(
                    "ngram",
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
                    kinds,
                    per_verify,
                    recovery,
                    fused,
                    graphs,
                    nvfp4,
                    Some(proposer),
                    replay_forwards(&stats),
                ),
                None,
                sampler,
            ));
        }
    }

    // The stochastic rows (sc-24140): the reference loop and the step seam under the seeded
    // sampler; `sampled_step_model` compares against `sampled`.
    let sampled = sampled_config(new_tokens);
    let sampled_warm = sampled_config(warmup_tokens);
    let mut sampled_reference: Option<Vec<i32>> = None;
    if rows.iter().any(|r| r == "sampled") {
        reference_row(&model, &prompt, &sampled_warm, &device, &mut |_| {});
        let syncs0 = host_syncs_now();
        let fused0 = fused_tally_now();
        let graphs0 = cuda_graphs_now();
        let nv0 = nvfp4_tally_now();
        let sampler0 = sampler_now();
        let mut at_last = None;
        let (out, prefill, decode, forwards) = reference_row(
            &model,
            &prompt,
            &sampled,
            &device,
            &mut last_token_sampler(&device, new_tokens, &mut at_last),
        );
        let sampler = sampler_delta(sampler0, out.tokens.len());
        let syncs = host_syncs_now().zip(syncs0).map(|(a, b)| a - b);
        let fused = fused_delta(fused0, switch);
        let graphs = cuda_graphs_delta(graphs0, graphs_switch);
        let nvfp4 = nvfp4_delta(nv0, nv_switch);
        let used = note_peak(at_last);
        eprintln!(
            "[decode_bench] sampled          {:>7.2} tok/s  prefill {:.3}s  sampler {}",
            out.tokens.len() as f64 / decode,
            prefill,
            sampler
                .as_ref()
                .map_or("n/a".to_string(), |s| s.to_string())
        );
        rows_json.push(annotate_row(
            row_json(
                "sampled",
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
                None,
                None,
                fused,
                graphs,
                nvfp4,
                Some("none"),
                None,
            ),
            Some(&sampled),
            sampler,
        ));
        sampled_reference = Some(out.tokens);
    }

    if rows.iter().any(|r| r == "sampled_step_model") {
        step_model_row(&model, &prompt, &sampled_warm, &device, &mut |_| {});
        let fused0 = fused_tally_now();
        let graphs0 = cuda_graphs_now();
        let nv0 = nvfp4_tally_now();
        let sampler0 = sampler_now();
        let mut at_last = None;
        let (out, prefill, decode, record, cache, kinds, proposer) = step_model_row(
            &model,
            &prompt,
            &sampled,
            &device,
            &mut last_token_sampler(&device, new_tokens, &mut at_last),
        );
        let sampler = sampler_delta(sampler0, out.tokens.len());
        let cache = checked_step_cache(new_tokens, cache);
        let fused = fused_delta(fused0, switch);
        let graphs = cuda_graphs_delta(graphs0, graphs_switch);
        let nvfp4 = nvfp4_delta(nv0, nv_switch);
        let used = note_peak(at_last);
        eprintln!(
            "[decode_bench] sampled_step_model {:>5.2} tok/s  prefill {:.3}s  sampler {}",
            out.tokens.len() as f64 / decode,
            prefill,
            sampler
                .as_ref()
                .map_or("n/a".to_string(), |s| s.to_string())
        );
        let (forwards, syncs) = match record {
            Some((forwards, syncs)) => (Some(forwards), Some(syncs)),
            None => (None, None),
        };
        rows_json.push(annotate_row(
            row_json(
                "sampled_step_model",
                None,
                &out,
                sampled_reference.as_deref(),
                prefill,
                decode,
                forwards,
                None,
                None,
                syncs,
                used,
                cache,
                kinds,
                None,
                None,
                fused,
                graphs,
                nvfp4,
                Some(proposer),
                None,
            ),
            Some(&sampled),
            sampler,
        ));
    }

    let mut doc = BTreeMap::new();
    doc.insert("schema_version", json!(2));
    doc.insert("suite", json!("decode_bench"));
    doc.insert("label", json!(label));
    doc.insert("snapshot", json!(snapshot.display().to_string()));
    doc.insert("device", json!(device_name));
    insert_run_identity(&mut doc, &device, &prompt);
    doc.insert(
        "compute_dtype",
        json!(format!("{:?}", model.compute_dtype())),
    );
    doc.insert("load_seconds", json!(load_secs));
    doc.insert("prompt_tokens", json!(prompt.len()));
    doc.insert("new_tokens", json!(new_tokens));
    doc.insert("warmup_tokens", json!(warmup_tokens));
    doc.insert("fused_kernels", json!(switch));
    doc.insert("cuda_graphs", json!(graphs_switch));
    doc.insert("cuda_stream", json!(cuda_stream_label()));
    doc.insert("weight_format", json!(format));
    doc.insert("weight_census", json!(hybrid_census(&model, mtp.as_ref())));
    doc.insert("sampling", sampling_json(&sampled));
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
