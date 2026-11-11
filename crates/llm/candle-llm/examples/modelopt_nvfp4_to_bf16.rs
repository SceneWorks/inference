//! Dequantize a ModelOpt NVFP4 snapshot (e.g. `MiniMaxAI/MiniMax-H3-Image-PE`) into the dense
//! bf16 snapshot every non-Blackwell runtime starts from (SW-7099), through
//! [`candle_llm::modelopt::convert_snapshot_to_bf16`].
//!
//! ```text
//! cargo run --release -p candle-llm --example modelopt_nvfp4_to_bf16 -- <source> <out_dir>
//! ```
//!
//! Runs on the CPU, one shard at a time. The Q8 / Q4 tiers are then written from `<out_dir>` by
//! each backend's `prepare_snapshot` example (`-p candle-llm` for Candle GGML blocks, `-p mlx-llm`
//! for MLX affine). Prints the report as one JSON line on success.

use std::path::PathBuf;
use std::process::ExitCode;

fn main() -> ExitCode {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let [source, out_dir] = args.as_slice() else {
        eprintln!("usage: modelopt_nvfp4_to_bf16 <source> <out_dir>");
        return ExitCode::from(2);
    };
    match candle_llm::modelopt::convert_snapshot_to_bf16(
        &PathBuf::from(source),
        &PathBuf::from(out_dir),
    ) {
        Ok(report) => {
            let line = serde_json::json!({
                "out_dir": out_dir,
                "dequantized": report.dequantized,
                "passthrough": report.passthrough,
                "shards": report.shards,
            });
            println!("{line}");
            ExitCode::SUCCESS
        }
        Err(e) => {
            eprintln!("modelopt_nvfp4_to_bf16 failed: {e}");
            ExitCode::FAILURE
        }
    }
}
