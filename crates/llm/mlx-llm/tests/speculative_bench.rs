//! The speculative-decoding benchmark (epic sc-24432 acceptance test 3 / E6) on MLX: the
//! `#[ignore]`d real-weight entry point the terminal campaign drives, and the weights-free run of
//! the same harness on the shared fixture. Both are Candle's (`candle-llm`'s `speculative_bench`
//! test) over an `mlx-llama` provider: [`core_llm_testkit::run_speculative_bench_from_env`] lists
//! every `SPECULATIVE_BENCH_*` knob and [`core_llm_testkit::BenchRow`] documents the JSON schema.
//! The pre-epic baseline driver is `core-llm-testkit/baseline/speculative_bench_baseline.rs`.
//!
//! ```text
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

#[test]
#[ignore = "needs a snapshot via SPECULATIVE_BENCH_SNAPSHOT and an output path via SPECULATIVE_BENCH_OUTPUT"]
fn speculative_bench_writes_the_baseline_document() {
    let (output, doc) = core_llm_testkit::run_speculative_bench_from_env("mlx", &load)
        .unwrap_or_else(|e| panic!("{e}"));
    println!("wrote {} ({} rows)", output.display(), doc.rows.len());
}

/// The harness end to end on the shared fixture: every schema field over two repeats with the
/// prefix cache off by default, and a warm-up that never lends the measured prompt to its row.
#[test]
fn the_bench_runs_on_the_fixture_with_isolated_warmups() {
    let root = Fixture::new("mlx-llm-speculative-bench-", None);
    core_llm_testkit::check_speculative_bench_on_fixture(root.root(), "mlx", &load)
        .unwrap_or_else(|e| panic!("{e}"));
}
