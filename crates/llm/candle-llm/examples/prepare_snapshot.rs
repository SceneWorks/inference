//! Persist a Q4 / Q8 tier of an HF-safetensors (or GGUF) snapshot through candle-llm's registered
//! [`prepare_snapshot`](candle_llm::prepare_snapshot) — the CLI the YuE rehost tiers (sc-19375) are
//! produced with (`scripts/audio/prepare_yue_assets.py` drives it).
//!
//! ```text
//! cargo run --release -p candle-llm --example prepare_snapshot -- \
//!   <source> <out_dir> <q4|q8|q4-full|q8-full|dense>
//! ```
//!
//! `q8-full` and `q4-full` are the fully quantized qwen3_5 tiers (SW-7201,
//! [`candle_llm::prepare::prepare_fully_quantized`]): the Q8_0 / Q4_K body plus `lm_head`,
//! `embed_tokens` and every Gated DeltaNet `in_proj_a` / `in_proj_b` stored as Q8_0 blocks, so
//! `q8-full` keeps no language-model matrix bf16 and `q4-full` is a 4-bit body with a Q8_0 head and
//! embedding. The vision tower stays dense and the MTP head is stored as in the plain tier.
//!
//! Runs on the CPU (the preparer never opens an accelerator). Prints the [`PrepareReport`] as one
//! JSON line on success.

use std::path::PathBuf;
use std::process::ExitCode;

use core_llm::{PrepareSpec, Quantize};

fn main() -> ExitCode {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let [source, out_dir, tier] = args.as_slice() else {
        eprintln!("usage: prepare_snapshot <source> <out_dir> <q4|q8|q4-full|q8-full|dense>");
        return ExitCode::from(2);
    };
    let (quantize, full) = match tier.as_str() {
        "q4" => (Some(Quantize::Q4), false),
        "q8" => (Some(Quantize::Q8), false),
        "q4-full" => (Some(Quantize::Q4), true),
        "q8-full" => (Some(Quantize::Q8), true),
        "dense" => (None, false),
        other => {
            eprintln!("unknown tier {other:?}; expected q4, q8, q4-full, q8-full or dense");
            return ExitCode::from(2);
        }
    };
    let spec = PrepareSpec {
        source: PathBuf::from(source),
        out_dir: PathBuf::from(out_dir),
        quantize,
    };
    let prepared = if full {
        candle_llm::prepare::prepare_fully_quantized(&spec)
    } else {
        candle_llm::prepare_snapshot(&spec)
    };
    match prepared {
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
