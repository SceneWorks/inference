//! SC-20677 real K/V capture: prefill + one decode step through the campaign's product decoder,
//! writing per-layer `q`/`k`/`v` safetensors for `sc20677_kv_candidates --kv`. The `parent` runs
//! the model `worker` only under the campaign supervisor's footprint cap, host reserve, and
//! deadline. See `mlx_llm::kv_capture` for the file format and the one-command capture+compare.

fn main() -> std::process::ExitCode {
    let args = std::env::args().skip(1).collect::<Vec<_>>();
    match mlx_llm::kv_capture::cli(&args) {
        Ok(()) => std::process::ExitCode::SUCCESS,
        Err(error) => {
            eprintln!("sc20677_capture_kv: {error}");
            std::process::ExitCode::FAILURE
        }
    }
}
