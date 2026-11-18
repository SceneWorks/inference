//! Persist a Q4 / Q8 tier of an HF-safetensors (or GGUF) snapshot through mlx-llm's registered
//! [`prepare_snapshot`](mlx_llm::prepare_snapshot) — the MLX counterpart of candle-llm's
//! `prepare_snapshot` example.
//!
//! ```text
//! cargo run --release -p mlx-llm --example prepare_snapshot -- \
//!   <source> <out_dir> <q4|q8|q4-full|q8-full|dense> [--without-mtp]
//! ```
//!
//! `q8-full` and `q4-full` are the fully quantized Qwen3.5 tiers (SW-7201,
//! [`mlx_llm::write_hf_snapshot_tier`]): the q8 / q4 body plus `lm_head`, `embed_tokens` and every
//! Gated DeltaNet `in_proj_a` / `in_proj_b` stored 8-bit affine (group size 64), so `q8-full` keeps
//! no language-model matrix bf16 and `q4-full` is a 4-bit body with an 8-bit head and embedding.
//! The vision tower stays dense and the MTP head is quantized as in the plain tier.
//!
//! `--without-mtp` writes the snapshot without its native MTP head
//! ([`mlx_llm::write_hf_snapshot_without_native_mtp`]; HF safetensors sources only).
//!
//! The decode-speedups benchmark (epic sc-24432, sc-24446) loads its large Q4 MLX rows from a
//! snapshot prepared here rather than quantizing at load: MLX load admission prices a load-time
//! quantization at twice the BF16 source (`mlx_llm::load_memory`), which refuses Qwen3.8-27B and
//! Qwen3.6-35B-A3B on a 128 GiB host at both measured revisions, while a prepared snapshot is
//! priced at its own (Q4) payload and holds the same quantized projections. The Qwen3.6-35B-A3B
//! epic/baseline rows use `--without-mtp`: the pre-epic MLX loader refuses its MoE MTP head.
//!
//! Prints the [`PrepareReport`](core_llm::PrepareReport) as one JSON line on success.

use std::path::PathBuf;
use std::process::ExitCode;

use core_llm::{PrepareSpec, Quantize};
use mlx_llm::primitives::projection::QuantSpec;

fn main() -> ExitCode {
    let mut args: Vec<String> = std::env::args().skip(1).collect();
    let without_mtp = match args.iter().position(|a| a == "--without-mtp") {
        Some(i) => {
            args.remove(i);
            true
        }
        None => false,
    };
    let [source, out_dir, tier] = args.as_slice() else {
        eprintln!(
            "usage: prepare_snapshot <source> <out_dir> <q4|q8|q4-full|q8-full|dense> \
             [--without-mtp]"
        );
        return ExitCode::from(2);
    };
    let (quantize, spec, full) = match tier.as_str() {
        "q4" => (Some(Quantize::Q4), Some(QuantSpec::q4()), false),
        "q8" => (Some(Quantize::Q8), Some(QuantSpec::q8()), false),
        "q4-full" => (Some(Quantize::Q4), Some(QuantSpec::q4()), true),
        "q8-full" => (Some(Quantize::Q8), Some(QuantSpec::q8()), true),
        "dense" => (None, None, false),
        other => {
            eprintln!("unknown tier {other:?}; expected q4, q8, q4-full, q8-full or dense");
            return ExitCode::from(2);
        }
    };
    let line = if let (true, Some(spec)) = (full, spec) {
        mlx_llm::write_hf_snapshot_tier(source, out_dir, spec, without_mtp)
            .map(|report| {
                serde_json::json!({
                    "out_dir": report.out_dir.display().to_string(),
                    "num_tensors": report.num_tensors,
                    "quantized": quantize.map(|q| format!("{q:?}")),
                    "quantized_projections": report.quantized_projections,
                    "quantized_remaining": report.quantized_remaining,
                    "without_mtp": without_mtp,
                })
            })
            .map_err(|e| e.to_string())
    } else if without_mtp {
        mlx_llm::write_hf_snapshot_without_native_mtp(source, out_dir, spec)
            .map(|report| {
                serde_json::json!({
                    "out_dir": report.out_dir.display().to_string(),
                    "num_tensors": report.num_tensors,
                    "quantized": quantize.map(|q| format!("{q:?}")),
                    "quantized_projections": report.quantized_projections,
                    "without_mtp": true,
                })
            })
            .map_err(|e| e.to_string())
    } else {
        let spec = PrepareSpec {
            source: PathBuf::from(source),
            out_dir: PathBuf::from(out_dir),
            quantize,
        };
        mlx_llm::prepare_snapshot(&spec)
            .map(|report| {
                serde_json::json!({
                    "out_dir": report.out_dir.display().to_string(),
                    "num_tensors": report.num_tensors,
                    "quantized": report.quantized.map(|q| format!("{q:?}")),
                    "passthrough": report.passthrough,
                })
            })
            .map_err(|e| e.to_string())
    };
    match line {
        Ok(line) => {
            println!("{line}");
            ExitCode::SUCCESS
        }
        Err(e) => {
            eprintln!("prepare_snapshot failed: {e}");
            ExitCode::FAILURE
        }
    }
}
