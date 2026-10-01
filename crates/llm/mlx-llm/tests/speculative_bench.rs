//! The speculative-decoding benchmark (epic sc-24432 acceptance test 3 / E6) on MLX: the
//! `#[ignore]`d real-weight entry point the terminal campaign drives, and the weights-free run of
//! the same harness on the shared fixture. Both are Candle's (`candle-llm`'s `speculative_bench`
//! test) over an `mlx-llama` provider: [`core_llm_testkit::run_speculative_bench_from_env`] lists
//! every `SPECULATIVE_BENCH_*` knob and [`core_llm_testkit::BenchRow`] documents the JSON schema.
//! The pre-epic baseline driver is `core-llm-testkit/baseline/speculative_bench_baseline.rs`.
//!
//! `--release` links the Release libmlx cell, so fetch that one (the script's default is Debug):
//!
//! ```text
//! eval "$(scripts/fetch-prebuilt-mlx.sh --build-type Release)" \
//!   && export PMETAL_MLX_PREBUILT_DIR PMETAL_METALLIB_PATH
//! SPECULATIVE_BENCH_SNAPSHOT=/path/to/snapshot SPECULATIVE_BENCH_OUTPUT=/tmp/bench.json \
//!   cargo test --release -p mlx-llm --test integration -- \
//!   speculative_bench::speculative_bench_writes_the_baseline_document --ignored --nocapture
//! ```

use core_llm::{LoadSpec, TextLlm};

use crate::common::Fixture;

fn load(spec: &LoadSpec) -> Result<Box<dyn TextLlm>, String> {
    mlx_llm::LlamaProvider::load(spec)
        .map(|p| Box::new(p) as Box<dyn TextLlm>)
        .map_err(|e| e.to_string())
}

/// The effective state of every MLX runtime switch (`core_llm_testkit::BenchSwitches`), read from
/// the switch objects on this (the generating) thread.
fn switches() -> Vec<(&'static str, serde_json::Value)> {
    use mlx_llm::switches::{DEVICE_SAMPLER, FUSED_ROTATION, GDN_KERNEL, PIPELINING};
    [&PIPELINING, &DEVICE_SAMPLER, &FUSED_ROTATION, &GDN_KERNEL]
        .into_iter()
        .map(|switch| (switch.process().env(), serde_json::json!(switch.enabled())))
        .collect()
}

#[test]
#[ignore = "needs a snapshot via SPECULATIVE_BENCH_SNAPSHOT and an output path via SPECULATIVE_BENCH_OUTPUT"]
fn speculative_bench_writes_the_baseline_document() {
    let (output, doc) = core_llm_testkit::run_speculative_bench_from_env("mlx", &load, &switches)
        .unwrap_or_else(|e| panic!("{e}"));
    println!("wrote {} ({} rows)", output.display(), doc.rows.len());
}

/// The entry point records every MLX switch.
#[test]
fn the_entry_reads_every_mlx_switch() {
    let names: Vec<_> = switches().into_iter().map(|(name, _)| name).collect();
    assert_eq!(
        names,
        [
            "MLX_LLM_PIPELINING",
            "MLX_LLM_DEVICE_SAMPLER",
            "MLX_LLM_FUSED_ROTATION",
            "MLX_LLM_GDN_KERNEL"
        ]
    );
}

/// The harness end to end on the shared fixture: every schema field over two repeats with the
/// prefix cache off by default, and a warm-up that never lends the measured prompt to its row.
#[test]
fn the_bench_runs_on_the_fixture_with_isolated_warmups() {
    let root = Fixture::new("mlx-llm-speculative-bench-", None);
    core_llm_testkit::check_speculative_bench_on_fixture(root.root(), "mlx", &load, &switches)
        .unwrap_or_else(|e| panic!("{e}"));
}
