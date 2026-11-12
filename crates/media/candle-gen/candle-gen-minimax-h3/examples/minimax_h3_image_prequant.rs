//! Produce one MiniMax-H3-Image DiT tier on the candle lane (SW-7093). NOT part of CI: the
//! real-weights `minimax-h3-image` profile runs it on the CUDA runner.
//!
//! ```text
//! MINIMAX_H3_IMAGE_SRC=<MiniMax-H3-Image snapshot root> \
//! MINIMAX_H3_IMAGE_OUT=<dir for this tier> MINIMAX_H3_TIER=q4 \
//!   cargo run --release --example minimax_h3_image_prequant -p candle-gen-minimax-h3
//! ```
//!
//! Writes `<out>/transformer_diffusers/` (packed shards, marked `config.json`, index) and
//! `<out>/pdd_fused_heads.safetensors`. An output directory that already exists is a hard error.
//! `MINIMAX_H3_ADALN_BITS` overrides the AdaLN width independently of the tier.
use std::path::PathBuf;

fn env(key: &str) -> Option<String> {
    std::env::var(key).ok().filter(|v| !v.trim().is_empty())
}

fn main() {
    let src = PathBuf::from(env("MINIMAX_H3_IMAGE_SRC").expect("MINIMAX_H3_IMAGE_SRC"));
    let dst = PathBuf::from(env("MINIMAX_H3_IMAGE_OUT").expect("MINIMAX_H3_IMAGE_OUT"));
    let bits: i32 = match env("MINIMAX_H3_TIER")
        .expect("MINIMAX_H3_TIER=q4|q8")
        .as_str()
    {
        "q4" => 4,
        "q8" => 8,
        other => panic!("MINIMAX_H3_TIER must be q4 or q8, got {other:?}"),
    };
    let adaln: i32 = env("MINIMAX_H3_ADALN_BITS").map_or(bits, |v| v.parse().expect("4 or 8"));
    assert!(
        !dst.exists(),
        "{} already exists -- refusing to overwrite a tier",
        dst.display()
    );
    let started = std::time::Instant::now();
    let mut total = 0u64;
    let shards =
        candle_gen_minimax_h3::convert::quantize_image_dit(&src, &dst, bits, adaln, &mut |path| {
            total += std::fs::metadata(path).map(|m| m.len()).unwrap_or(0);
            println!(
                "  [shard] {} (running total {total} B, {:.1}s)",
                path.file_name().unwrap_or_default().to_string_lossy(),
                started.elapsed().as_secs_f64()
            );
            Ok(())
        })
        .expect("quantize_image_dit");
    println!("[done] tier q{bits}: {} shards, {total} B", shards.len());
}
