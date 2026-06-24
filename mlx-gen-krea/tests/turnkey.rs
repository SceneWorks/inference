//! sc-7573 — assemble the pre-quantized **turnkey** Krea 2 Turbo snapshot and prove it loads + renders
//! through the exact `KreaPipeline::from_snapshot` path (the published `SceneWorks/krea-2-turbo-mlx`).
//!
//! The published repo carries one complete, `from_snapshot`-loadable root per quant in a `q{bits}`
//! subdir (`q8/`, `q4/` — the Ideogram pattern the worker's `krea_model_subdir` resolves), so each
//! invocation assembles into `<KREA_ASM_OUT>/q{bits}/…`. Run twice to build both.
//!
//! `#[ignore]` (needs the real ~32 GB dense snapshot + disk + Metal). Drive per quant via env:
//!   KREA_ASM_SRC=<dense snapshot root>   (defaults to KREA_TURBO_DIR)
//!   KREA_ASM_OUT=<turnkey repo root>     (defaults to <tmp>/krea-2-turbo-mlx)
//!   KREA_ASM_BITS=8|4                    (defaults to 8 — the ship default)
//! e.g. assemble Q8 + verify:
//!   KREA_TURBO_DIR=<snapshot> KREA_ASM_OUT=~/krea-2-turbo-mlx KREA_ASM_BITS=8 \
//!     cargo test -p mlx-gen-krea --release --test turnkey assemble_turnkey_loads -- --ignored --nocapture

use std::path::{Path, PathBuf};

use mlx_gen_krea::convert::assemble_quantized_snapshot;
use mlx_gen_krea::{KreaPipeline, TurboOptions};

fn env_or(key: &str, fallback: &str) -> String {
    std::env::var(key).unwrap_or_else(|_| {
        std::env::var(fallback).unwrap_or_else(|_| panic!("set {key} or {fallback}"))
    })
}

/// Total bytes of a directory tree (for the manifest `estimatedSizeBytes`).
fn dir_size(dir: &Path) -> u64 {
    let mut total = 0;
    if let Ok(rd) = std::fs::read_dir(dir) {
        for e in rd.flatten() {
            let p = e.path();
            if p.is_dir() {
                total += dir_size(&p);
            } else if let Ok(m) = p.metadata() {
                total += m.len();
            }
        }
    }
    total
}

#[test]
#[ignore = "needs real weights (~32 GB dense snapshot): set KREA_ASM_SRC/KREA_TURBO_DIR"]
fn assemble_turnkey_loads() {
    let src = PathBuf::from(env_or("KREA_ASM_SRC", "KREA_TURBO_DIR"));
    let root = std::env::var("KREA_ASM_OUT")
        .map(PathBuf::from)
        .unwrap_or_else(|_| std::env::temp_dir().join("krea-2-turbo-mlx"));
    let bits: i32 = std::env::var("KREA_ASM_BITS")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(8);
    let dst = root.join(format!("q{bits}"));

    println!(
        "[7573] assembling Q{bits} turnkey  {}  →  {}",
        src.display(),
        dst.display()
    );
    let _ = std::fs::remove_dir_all(&dst); // start clean so no stale source shards linger
    assemble_quantized_snapshot(&src, &dst, bits).expect("assemble turnkey");
    println!(
        "[7573] q{bits} turnkey size = {:.2} GB",
        dir_size(&dst) as f64 / 1e9
    );

    // Load through the SHIPPED path + render a small image to prove the packed snapshot runs e2e.
    let pipe = KreaPipeline::from_snapshot(&dst).expect("from_snapshot on the turnkey");
    let img = pipe
        .generate_turbo(
            "a red apple on a wooden table",
            &TurboOptions {
                width: 512,
                height: 512,
                steps: 8,
                seed: 0,
                sampler: None,
                scheduler: None,
            },
        )
        .expect("generate_turbo");

    assert_eq!((img.width, img.height), (512, 512));
    let (mn, mx) = img
        .pixels
        .iter()
        .fold((255u8, 0u8), |(mn, mx), &p| (mn.min(p), mx.max(p)));
    println!("[7573] render stats: min={mn} max={mx}");
    assert!(
        mx - mn > 32,
        "turnkey render looks degenerate (min={mn} max={mx})"
    );
    println!(
        "[7573] OK — Q{bits} turnkey at {} loads + renders",
        dst.display()
    );
}
