//! One-call Q8 pre-quant of the ideogram-4-mlx snapshot so it fits 64GB (bf16 is ~53GB — the DiT is
//! doubled for asymmetric CFG). Run from the workspace root:
//!   cargo run --release --example ideogram_prequant -p mlx-gen-ideogram
use std::path::PathBuf;
use std::time::Instant;

fn env_or(k: &str, d: &str) -> String {
    std::env::var(k).unwrap_or_else(|_| d.to_string())
}

fn main() {
    let src = PathBuf::from(env_or(
        "IDEOGRAM_SRC",
        "/Users/zakkeown/Models/aether/ideogram-4-mlx",
    ));
    let dst = PathBuf::from(env_or(
        "IDEOGRAM_Q8",
        "/Users/zakkeown/Models/aether/ideogram-4-q8",
    ));
    let bits: i32 = env_or("IDEOGRAM_BITS", "8").parse().expect("bits");
    eprintln!("[prequant] {} -> Q{bits} {}", src.display(), dst.display());
    let t = Instant::now();
    mlx_gen_ideogram::convert::prequantize_turnkey(&src, &dst, bits).expect("prequantize_turnkey");
    let du = std::fs::read_dir(&dst)
        .map(|rd| rd.filter_map(|e| e.ok()).count())
        .unwrap_or(0);
    eprintln!(
        "[prequant] done in {:.0}s — {} entries in {}",
        t.elapsed().as_secs_f32(),
        du,
        dst.display()
    );
}
