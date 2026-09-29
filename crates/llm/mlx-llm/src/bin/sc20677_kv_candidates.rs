//! SC-20677 matched-budget KV candidate comparison (group-affine, packed RVQ, RaBitQ).
//!
//! Synthetic: `cargo run --release -p mlx-llm --bin sc20677_kv_candidates -- --out report.json`.
//! Captured real K/V (files written by `sc20677_capture_kv parent ...`, see `mlx_llm::kv_capture`):
//! `cargo run --release -p mlx-llm --bin sc20677_kv_candidates -- --kv layer-000.safetensors --out report.json`.
//! See `--help` and `mlx_llm::primitives::kv_candidates::compare` for the file format and the
//! one-command capture+compare invocation.

fn main() -> std::process::ExitCode {
    let args = std::env::args().skip(1).collect::<Vec<_>>();
    match mlx_llm::primitives::kv_candidates::compare::cli(&args) {
        Ok(()) => std::process::ExitCode::SUCCESS,
        Err(error) => {
            eprintln!("sc20677_kv_candidates: {error}");
            std::process::ExitCode::FAILURE
        }
    }
}
