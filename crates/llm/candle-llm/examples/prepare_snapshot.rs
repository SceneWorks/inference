//! Persist a Q4 / Q8 tier of an HF-safetensors (or GGUF) snapshot through candle-llm's registered
//! [`prepare_snapshot`](candle_llm::prepare_snapshot) — the CLI the YuE rehost tiers (sc-19375) are
//! produced with (`scripts/audio/prepare_yue_assets.py` drives it).
//!
//! ```text
//! cargo run --release -p candle-llm --example prepare_snapshot -- <source> <out_dir> <q4|q8|dense>
//! ```
//!
//! Runs on the CPU (the preparer never opens an accelerator). Prints the [`PrepareReport`] as one
//! JSON line on success.

use std::path::PathBuf;
use std::process::ExitCode;

use core_llm::{PrepareSpec, Quantize};

fn main() -> ExitCode {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let [source, out_dir, tier] = args.as_slice() else {
        eprintln!("usage: prepare_snapshot <source> <out_dir> <q4|q8|dense>");
        return ExitCode::from(2);
    };
    let quantize = match tier.as_str() {
        "q4" => Some(Quantize::Q4),
        "q8" => Some(Quantize::Q8),
        "dense" => None,
        other => {
            eprintln!("unknown tier {other:?}; expected q4, q8 or dense");
            return ExitCode::from(2);
        }
    };
    let spec = PrepareSpec {
        source: PathBuf::from(source),
        out_dir: PathBuf::from(out_dir),
        quantize,
    };
    match candle_llm::prepare_snapshot(&spec) {
        Ok(report) => {
            let line = serde_json::json!({
                "out_dir": report.out_dir.display().to_string(),
                "num_tensors": report.num_tensors,
                "quantized": report.quantized.map(|q| format!("{q:?}")),
                "passthrough": report.passthrough,
            });
            println!("{line}");
            ExitCode::SUCCESS
        }
        Err(e) => {
            eprintln!("prepare_snapshot failed: {e}");
            ExitCode::FAILURE
        }
    }
}
