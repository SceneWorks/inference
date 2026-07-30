//! Prepare a persisted, quantized snapshot from a dense Hugging Face model directory.
//!
//! ```text
//! cargo run --release --example prepare_snapshot -- <source_dir> <out_dir> [q4|q8|dense]
//! ```
//!
//! This is the consumer-side ingest step: `<source_dir>` is a dense snapshot the caller already
//! provisioned (this workspace never fetches), and `<out_dir>` receives a loadable snapshot with
//! the attention/MLP **projections** quantized. Embeddings, the LM head, and norms stay dense —
//! the engine's quantization invariant.
//!
//! It drives the backend-neutral `core_llm::SnapshotPreparer` contract rather than calling
//! `mlx_llm::write_hf_snapshot` directly, so it exercises the same path a product would: the
//! registry probes the source, selects the backend, and reports what it actually did.
//!
//! Quantizing on ingest is what makes a 4B model fit a phone: dense bf16 is ~7.5 GB, Q4 is
//! ~2.2 GB.

use std::path::Path;
use std::time::Instant;

use core_llm::{PrepareSpec, Quantize};
use mlx_llm::prepare_snapshot;

fn main() {
    if let Err(e) = run() {
        eprintln!("error: {e}");
        std::process::exit(1);
    }
}

fn dir_size_bytes(dir: &Path) -> u64 {
    let Ok(entries) = std::fs::read_dir(dir) else {
        return 0;
    };
    entries
        .flatten()
        .filter_map(|e| e.metadata().ok())
        .map(|m| if m.is_file() { m.len() } else { 0 })
        .sum()
}

fn gib(bytes: u64) -> f64 {
    bytes as f64 / (1024.0 * 1024.0 * 1024.0)
}

fn run() -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
    let mut args = std::env::args().skip(1);
    let source = args
        .next()
        .ok_or("usage: prepare_snapshot <source_dir> <out_dir> [q4|q8|dense]")?;
    let out = args.next().ok_or("missing <out_dir> argument")?;
    let quantize = match args.next().as_deref() {
        None | Some("q4") => Some(Quantize::Q4),
        Some("q8") => Some(Quantize::Q8),
        Some("dense") => None,
        Some(other) => {
            return Err(format!("unknown quantization {other:?}; use q4|q8|dense").into())
        }
    };

    let (source, out) = (Path::new(&source), Path::new(&out));
    let source_bytes = dir_size_bytes(source);
    eprintln!(
        "preparing {} ({:.2} GiB) -> {} [{}]",
        source.display(),
        gib(source_bytes),
        out.display(),
        match quantize {
            Some(Quantize::Q4) => "q4",
            Some(Quantize::Q8) => "q8",
            None => "dense",
        }
    );

    let started = Instant::now();
    let report = prepare_snapshot(&PrepareSpec {
        source: source.to_path_buf(),
        out_dir: out.to_path_buf(),
        quantize,
    })?;
    let elapsed = started.elapsed();

    let out_bytes = dir_size_bytes(out);
    eprintln!(
        "\ndone in {:.1}s\n  input format : {}\n  quantized    : {:?}\n  tensors      : {}\n  passthrough  : {}\n  out dir      : {}\n  size         : {:.2} GiB ({:.1}% of source)",
        elapsed.as_secs_f64(),
        report.input_format.as_str(),
        report.quantized,
        report.num_tensors,
        report.passthrough,
        report.out_dir.display(),
        gib(out_bytes),
        if source_bytes > 0 {
            100.0 * out_bytes as f64 / source_bytes as f64
        } else {
            0.0
        }
    );
    Ok(())
}
