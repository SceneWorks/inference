//! SC-20677 matched-budget KV candidate comparison (group-affine, packed RVQ, RaBitQ).
//!
//! Synthetic: `cargo run --release -p mlx-llm --bin sc20677_kv_candidates -- --out report.json`.
//! Captured real K/V (one command once captures exist):
//! `cargo run --release -p mlx-llm --bin sc20677_kv_candidates -- --kv layer.safetensors --out report.json`.
//! See `--help` and `mlx_llm::primitives::kv_candidates::compare` for the file format.

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
