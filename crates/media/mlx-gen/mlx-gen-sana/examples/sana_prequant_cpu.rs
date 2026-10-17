//! CPU-only offline conversion. Example: sana_prequant_cpu SRC DST 4 TENSOR_BUDGET_MIB
//! Use an external process RSS guard in addition to the header-derived tensor envelope.
//! No downloads and no GPU submission; the Metal trap aborts if dispatch escapes CPU scope.
#[cfg(target_os = "macos")]
#[path = "../../mlx-gen-iris/tests/common/metal_guard.rs"]
mod metal_guard;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args = std::env::args().skip(1).collect::<Vec<_>>();
    if args.len() != 4 {
        return Err("usage: sana_prequant_cpu SOURCE DESTINATION BITS TENSOR_BUDGET_MIB (external RSS guard required)".into());
    }
    #[cfg(target_os = "macos")]
    {
        metal_guard::forbid_metal();
        eprintln!(
            "Metal submission trap armed: {}",
            metal_guard::metal_forbidden()
        );
    }
    let src = std::path::Path::new(&args[0]);
    let dst = std::path::Path::new(&args[1]);
    let bits = args[2].parse()?;
    let budget = args[3]
        .parse::<u64>()?
        .checked_mul(1024 * 1024)
        .ok_or("tensor budget overflow")?;
    let required = mlx_gen_sana::convert::cpu_conversion_tensor_bytes(src)?;
    eprintln!("CPU tensor envelope: {required} bytes; operator budget: {budget} bytes. This is NOT a total process RSS bound.");
    mlx_gen_sana::convert::prequantize_turnkey_cpu(src, dst, bits, budget)?;
    Ok(())
}
