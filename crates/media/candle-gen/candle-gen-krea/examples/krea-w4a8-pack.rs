//! SW-6135: build the Krea 2 **W4A8 tier** — rotate and pack the DiT projections of a dense bf16
//! master into one native-keyed `asym_w4a8_int8` safetensors file
//! ([`candle_gen_krea::w4a8_tier::build_w4a8_tier`]; the format and the projection policy are
//! documented there).
//!
//! ```text
//! cargo run -p candle-gen-krea --release --example krea-w4a8-pack -- \
//!   <bf16 snapshot>/transformer <out>/krea2_turbo_w4a8.safetensors [--policy mixed|blanket] [--threads N]
//! ```
//!
//! CPU only (no `cuda` feature needed): packing is the offline packer, not a forward. Prints the
//! packed/kept census and writes `<out>.report.json` beside the file with the census, the byte
//! counts and the output's SHA-256 (the pin a published repo records). The report is not part of
//! the tier; publish the `.safetensors` alone.

use std::collections::BTreeMap;
use std::io::Read;
use std::path::PathBuf;
use std::time::Instant;

use candle_gen_krea::w4a8_tier::{build_w4a8_tier, KeepReason, W4A8Policy};
use sha2::Digest;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let mut args = std::env::args().skip(1);
    let usage = "usage: krea-w4a8-pack <transformer_dir> <out.safetensors> [--policy mixed|blanket] [--threads N]";
    let transformer = PathBuf::from(args.next().ok_or(usage)?);
    let out = PathBuf::from(args.next().ok_or(usage)?);
    let mut policy = W4A8Policy::Mixed;
    let mut threads = std::thread::available_parallelism().map_or(8, |n| n.get());
    while let Some(flag) = args.next() {
        match flag.as_str() {
            "--policy" => {
                policy = match args.next().as_deref() {
                    Some("mixed") => W4A8Policy::Mixed,
                    Some("blanket") => W4A8Policy::Blanket,
                    _ => return Err(usage.into()),
                }
            }
            "--threads" => threads = args.next().ok_or(usage)?.parse()?,
            _ => return Err(usage.into()),
        }
    }
    let config: serde_json::Value =
        serde_json::from_slice(&std::fs::read(transformer.join("config.json"))?)?;
    let cfg = candle_gen_krea::Krea2Config::from_json(&config)?;

    let started = Instant::now();
    let report = build_w4a8_tier(&transformer, cfg.num_layers, policy, &out, threads)?;
    let pack_s = started.elapsed().as_secs_f64();

    let mut kept: BTreeMap<String, usize> = BTreeMap::new();
    for (_, reason) in &report.kept {
        let label = match reason {
            KeepReason::Role(reason) => format!("{reason:?}"),
            KeepReason::Geometry => "Geometry".to_owned(),
        };
        *kept.entry(label).or_default() += 1;
    }
    let packed_params: usize = report.packed.iter().map(|(_, o, i)| o * i).sum();

    let mut hasher = sha2::Sha256::new();
    let mut file = std::fs::File::open(&out)?;
    let mut buf = vec![0_u8; 8 << 20];
    loop {
        let n = file.read(&mut buf)?;
        if n == 0 {
            break;
        }
        hasher.update(&buf[..n]);
    }
    let sha256 = format!("{:x}", hasher.finalize());
    let size = std::fs::metadata(&out)?.len();

    println!(
        "[sw-6135] {policy:?}: packed {} projections ({:.2}B params), kept {} at source precision {kept:?}",
        report.packed.len(),
        packed_params as f64 / 1e9,
        report.kept.len(),
    );
    println!(
        "[sw-6135] {} tensors; source {:.2} GB -> payload {:.2} GB; file {size} B sha256 {sha256}; packed in {pack_s:.1}s on {threads} threads",
        report.tensors,
        report.source_bytes as f64 / 1e9,
        report.output_bytes as f64 / 1e9,
    );
    let json = serde_json::json!({
        "policy": format!("{policy:?}"),
        "source": transformer,
        "file": out.file_name().map(|n| n.to_string_lossy().into_owned()),
        "size_bytes": size,
        "sha256": sha256,
        "tensors": report.tensors,
        "source_bytes": report.source_bytes,
        "payload_bytes": report.output_bytes,
        "packed_projections": report.packed.len(),
        "packed_params": packed_params,
        "kept_by_reason": kept,
        "packed": report.packed.iter().map(|(b, _, _)| b).collect::<Vec<_>>(),
        "pack_seconds": pack_s,
        "threads": threads,
    });
    let mut report_path = out.clone().into_os_string();
    report_path.push(".report.json");
    std::fs::write(&report_path, serde_json::to_vec_pretty(&json)?)?;
    Ok(())
}
