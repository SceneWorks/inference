//! CPU-only offline conversion of an Iris artifact to its packed q8 / q4 tier
//! (`mlx_gen_iris::packed`).
//!
//! `iris_prequant_cpu KIND SOURCE DESTINATION BITS TENSOR_BUDGET_MIB`, KIND one of `generation`,
//! `restoration`, `depth` (a backbone export directory) or `text-encoder` (the Qwen3-VL snapshot;
//! only its language tower is written). Use an external process RSS guard in addition to the
//! header-derived tensor envelope. No downloads and no GPU submission: the Metal trap aborts if a
//! dispatch escapes the CPU scope.
#[cfg(target_os = "macos")]
#[path = "../tests/common/metal_guard.rs"]
mod metal_guard;

use mlx_gen_iris::packed::{cpu_conversion_tensor_bytes, prequantize_cpu, ConvertKind};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args = std::env::args().skip(1).collect::<Vec<_>>();
    if args.len() != 5 {
        return Err(
            "usage: iris_prequant_cpu generation|restoration|depth|text-encoder SOURCE \
                    DESTINATION BITS TENSOR_BUDGET_MIB (external RSS guard required)"
                .into(),
        );
    }
    #[cfg(target_os = "macos")]
    {
        metal_guard::forbid_metal();
        eprintln!(
            "Metal submission trap armed: {}",
            metal_guard::metal_forbidden()
        );
    }
    let kind = match args[0].as_str() {
        "generation" | "restoration" => ConvertKind::Export { depth: false },
        "depth" => ConvertKind::Export { depth: true },
        "text-encoder" => ConvertKind::TextEncoder,
        other => return Err(format!("unknown kind {other}").into()),
    };
    let src = std::path::Path::new(&args[1]);
    let dst = std::path::Path::new(&args[2]);
    let bits: i32 = args[3].parse()?;
    let budget = args[4]
        .parse::<u64>()?
        .checked_mul(1024 * 1024)
        .ok_or("tensor budget overflow")?;
    let required = cpu_conversion_tensor_bytes(kind, src, bits)?;
    eprintln!(
        "CPU tensor envelope: {required} bytes; operator budget: {budget} bytes. This is NOT a \
         total process RSS bound."
    );
    prequantize_cpu(kind, src, dst, bits, budget)?;
    Ok(())
}
