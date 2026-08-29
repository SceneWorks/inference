//! SC-18777 terminal-only advanced-quant evidence producer.
//!
//! The binary is available only behind the non-runtime `terminal-quant-measurement` feature; an
//! actual run additionally refuses a build without `cuda` or a live CUDA device. Run the bf16 case
//! for a GPU first, then pass its generated output and receipt to each candidate case. Every
//! invocation requires a new output directory and a clean committed inference worktree.
//!
//! ```text
//! cargo run --release -p candle-gen-ltx \
//!   --features cuda,terminal-quant-measurement \
//!   --example ltx25-quant-measure -- \
//!   --acknowledgement I_ACKNOWLEDGE_SC18777_TERMINAL_MEASUREMENT_ONLY \
//!   --case ltx25-bf16-blackwell-v1 \
//!   --snapshot D:\models\ltx-2.5\distilled\bf16 \
//!   --model-revision 791ef61731ad067bd13ebff8cc0f07532476d9ef \
//!   --output-dir D:\evidence\ltx25-bf16-blackwell-v1
//! ```

use std::path::PathBuf;

use candle_gen_ltx::quant_measurement::{run, TerminalMeasurementConfig};

type Result<T> = std::result::Result<T, Box<dyn std::error::Error + Send + Sync>>;

fn value(args: &[String], key: &str) -> Result<String> {
    let index = args
        .iter()
        .position(|arg| arg == key)
        .ok_or_else(|| format!("missing required {key} <value>"))?;
    args.get(index + 1)
        .filter(|value| !value.starts_with("--"))
        .cloned()
        .ok_or_else(|| format!("missing value after {key}").into())
}

fn optional_value(args: &[String], key: &str) -> Option<String> {
    args.iter()
        .position(|arg| arg == key)
        .and_then(|index| args.get(index + 1))
        .filter(|value| !value.starts_with("--"))
        .cloned()
}

fn main() -> Result<()> {
    let args = std::env::args().collect::<Vec<_>>();
    if args.iter().any(|arg| arg == "--list-cases") {
        for case in candle_gen_ltx::quant_eval::TERMINAL_MEASUREMENT_CASES {
            println!(
                "{} mode={} gpu={} {}x{}x{}@{} seed={}",
                case.id,
                case.mode.id(),
                case.gpu.id(),
                case.width,
                case.height,
                case.frames,
                case.fps,
                case.seed,
            );
        }
        return Ok(());
    }
    let receipt = run(TerminalMeasurementConfig {
        acknowledgement: value(&args, "--acknowledgement")?,
        case_id: value(&args, "--case")?,
        snapshot: PathBuf::from(value(&args, "--snapshot")?),
        model_revision: value(&args, "--model-revision")?,
        output_dir: PathBuf::from(value(&args, "--output-dir")?),
        reference_output: optional_value(&args, "--reference-output").map(PathBuf::from),
        reference_receipt: optional_value(&args, "--reference-receipt").map(PathBuf::from),
    })?;
    println!(
        "sealed {} receipt={} output={} peak={} bytes wall={} ms",
        receipt.case_id,
        receipt.receipt_sha256,
        receipt.output_sha256,
        receipt.peak_vram_bytes,
        receipt.wall_clock_ms,
    );
    Ok(())
}
