//! Compare two rendered PNGs pixel-for-pixel.
//!
//! ```text
//! cargo run --release -p mlx-gen-ios-catalog --example compare_renders -- <a.png> <b.png>
//! ```
//!
//! # Why this exists
//!
//! "The device render looks the same as the Mac's" is an eyeball claim, and eyeballs cannot tell
//! *identical* from *very close* — which is exactly the distinction that matters here. Two questions
//! need separating and only numbers separate them:
//!
//! 1. **Is generation deterministic across platforms?** Same prompt, seed, steps, residency and
//!    decode strategy on macOS and iOS should give the *same bytes* if MLX is deterministic across
//!    two Metal devices. A non-zero difference means it is not, and every cross-platform "identical
//!    output" claim in this repo would need qualifying.
//! 2. **What does the tiled decode cost against the canonical path?** That difference is expected
//!    and bounded (DC-AE's attention normalizer is global — see `mlx-gen-sana`'s
//!    `decode_tiling_parity`), and worth quantifying rather than asserting.
//!
//! Reports max, mean, and the fraction of channel samples past two visibility thresholds, because a
//! handful of outliers on high-contrast edges is a very different result from a broad tonal shift
//! and `max` alone cannot tell them apart.

use std::path::Path;

fn main() {
    if let Err(e) = run() {
        eprintln!("error: {e}");
        std::process::exit(1);
    }
}

fn run() -> Result<(), Box<dyn std::error::Error>> {
    let mut args = std::env::args().skip(1);
    let a_path = args
        .next()
        .ok_or("usage: compare_renders <a.png> <b.png>")?;
    let b_path = args
        .next()
        .ok_or("usage: compare_renders <a.png> <b.png>")?;

    let a = image::open(Path::new(&a_path))?.to_rgb8();
    let b = image::open(Path::new(&b_path))?.to_rgb8();

    if a.dimensions() != b.dimensions() {
        return Err(format!(
            "dimensions differ: {:?} vs {:?}",
            a.dimensions(),
            b.dimensions()
        )
        .into());
    }

    let (mut max, mut sum, mut n8, mut n32, mut differing) = (0u8, 0u64, 0u64, 0u64, 0u64);
    for (pa, pb) in a.as_raw().iter().zip(b.as_raw()) {
        let d = pa.abs_diff(*pb);
        if d > 0 {
            differing += 1;
        }
        max = max.max(d);
        sum += d as u64;
        if d > 8 {
            n8 += 1;
        }
        if d > 32 {
            n32 += 1;
        }
    }
    let total = a.as_raw().len() as f64;

    println!("{}\n  vs\n{}\n", a_path, b_path);
    println!("  dimensions      {:?}", a.dimensions());
    println!(
        "  identical       {}",
        if differing == 0 {
            "YES — byte-for-byte"
        } else {
            "no"
        }
    );
    println!(
        "  differing       {differing} / {} samples ({:.2}%)",
        total as u64,
        100.0 * differing as f64 / total
    );
    println!("  max |Δ|         {max} / 255");
    println!("  mean |Δ|        {:.4}", sum as f64 / total);
    println!(
        "  >8/255          {:.2}%  (roughly the threshold of visibility)",
        100.0 * n8 as f64 / total
    );
    println!(
        "  >32/255         {:.2}%  (plainly visible)",
        100.0 * n32 as f64 / total
    );

    Ok(())
}
