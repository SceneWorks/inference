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
//!   --snapshot-root D:\hf\models--Lightricks--LTX-Video\snapshots\791ef61731ad067bd13ebff8cc0f07532476d9ef \
//!   --bundle-subdir bundles\distilled\bf16 \
//!   --model-revision 791ef61731ad067bd13ebff8cc0f07532476d9ef \
//!   --output-dir D:\evidence\ltx25-bf16-blackwell-v1
//! ```
//!
//! Candidate runs additionally require the explicit bf16 snapshot/revision and the canonical
//! `generated-output.bin`/`receipt.json` pair from that snapshot's evidence directory. Candidate
//! and reference inventories are intentionally distinct and independently sealed.

use std::path::PathBuf;

use candle_gen_ltx::quant_measurement::{
    materialize_campaign_promotion, run, run_campaign, TerminalMeasurementConfig,
};

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
                "{} mode={} variant={} gpu={} {}x{}x{}@{} seed={}",
                case.id,
                case.mode.id(),
                case.transformer_variant.id(),
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
    let acknowledgement = value(&args, "--acknowledgement")?;
    if args.iter().any(|arg| arg == "--materialize-promotion") {
        materialize_campaign_promotion(
            &acknowledgement,
            PathBuf::from(value(&args, "--campaign-manifest")?).as_path(),
            PathBuf::from(value(&args, "--promotion-input")?).as_path(),
            PathBuf::from(value(&args, "--evidence-root")?).as_path(),
            PathBuf::from(value(&args, "--output-dir")?).as_path(),
        )?;
        println!("materialized verified LTX-2.5 promotion artifacts");
        return Ok(());
    }
    if let Some(manifest) = optional_value(&args, "--campaign-manifest") {
        let physical_gpu = value(&args, "--physical-gpu")?
            .parse::<usize>()
            .map_err(|_| "--physical-gpu must be one numeric physical ordinal")?;
        let receipts = run_campaign(
            &acknowledgement,
            PathBuf::from(manifest).as_path(),
            PathBuf::from(value(&args, "--output-root")?).as_path(),
            physical_gpu,
        )?;
        println!("sealed {} serial LTX-2.5 campaign receipts", receipts.len());
        return Ok(());
    }
    let receipt = run(TerminalMeasurementConfig {
        acknowledgement,
        case_id: value(&args, "--case")?,
        snapshot: PathBuf::from(value(&args, "--snapshot-root")?),
        bundle_subdir: PathBuf::from(value(&args, "--bundle-subdir")?),
        model_revision: value(&args, "--model-revision")?,
        output_dir: PathBuf::from(value(&args, "--output-dir")?),
        reference_snapshot: optional_value(&args, "--reference-snapshot-root").map(PathBuf::from),
        reference_bundle_subdir: optional_value(&args, "--reference-bundle-subdir")
            .map(PathBuf::from),
        reference_model_revision: optional_value(&args, "--reference-model-revision"),
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
