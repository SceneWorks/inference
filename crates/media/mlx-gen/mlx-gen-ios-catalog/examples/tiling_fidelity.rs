//! Does DC-AE decode tiling change the *generated* image? (E5, iOS memory work.)
//!
//! ```text
//! cargo run --release -p mlx-gen-ios-catalog --example tiling_fidelity -- <snapshot_dir>
//!     [--size N] [--steps N] [--tiles 512,384,256,192] [--overlap PX] [--out DIR]
//! ```
//!
//! # Why this exists separately from the parity test
//!
//! `mlx-gen-sana`'s `decode_tiling_parity` compares tiled and whole-image decodes of a **random
//! normal** latent. That isolates the layout bridge perfectly (a single tile reproduces the whole
//! decode bit-for-bit) but overstates the artifact: white noise is out-of-distribution for the
//! decoder and maximally sensitive to anything that changes spatial scope.
//!
//! A real denoised latent is spatially correlated, and it is the only thing a user will ever see. So
//! this runs the actual generator at one seed, decodes it every way, and reports the pixel
//! distribution — not just max and mean, which a handful of edge pixels can dominate.
//!
//! # What it is testing for
//!
//! DC-AE's decoder attention (`SanaMultiscaleLinearAttention`) normalizes by `1/(Σ + eps)` **summed
//! over every spatial position**, so tiling it would truncate that sum to the tile. Since sc-19753
//! the attention runs once over the whole image in the dense head and only the attention-free tail
//! is tiled, so what remains is a convolutional seam effect. This quantifies it on real content
//! against SANA's published decode ladder (`mlx_gen_sana::memory_strategy::DECODE_TILE_EDGES`).

use std::path::Path;

use mlx_gen::gen_core::{
    GenerationMemory, GenerationOutput, GenerationRequest, Image, Progress, WeightsSource,
};

fn main() {
    if let Err(e) = run() {
        eprintln!("error: {e}");
        std::process::exit(1);
    }
}

/// Pixel-difference distribution between two same-size images.
struct Diff {
    max: u8,
    mean: f64,
    /// Fraction of channel samples differing by more than 8/255 — the rough threshold below which a
    /// difference is invisible on a photographic image.
    over_8: f64,
    /// Fraction differing by more than 32/255, which is plainly visible.
    over_32: f64,
}

fn diff(a: &Image, b: &Image) -> Diff {
    let (mut max, mut sum, mut n8, mut n32) = (0u8, 0u64, 0u64, 0u64);
    for (x, y) in a.pixels.iter().zip(&b.pixels) {
        let d = x.abs_diff(*y);
        max = max.max(d);
        sum += d as u64;
        if d > 8 {
            n8 += 1;
        }
        if d > 32 {
            n32 += 1;
        }
    }
    let total = a.pixels.len() as f64;
    Diff {
        max,
        mean: sum as f64 / total,
        over_8: 100.0 * n8 as f64 / total,
        over_32: 100.0 * n32 as f64 / total,
    }
}

/// Render once. The decode geometry always rides the **request**: `Some((edge, overlap))` tiles,
/// `None` is an explicit whole-image decode (an all-disabled memory block). A request block beats
/// both `MLX_GEN_SANA_DECODE_TILE` and the `Sequential` load default, so no ambient setting can
/// change what a row of the table measured.
fn generate(
    dir: &Path,
    size: u32,
    steps: u32,
    tile: Option<(u32, u32)>,
) -> Result<Image, Box<dyn std::error::Error>> {
    // The iOS load defaults (Sequential), the policy whose peak this whole effort is bounding.
    let spec = mlx_gen_ios_catalog::sana_load_spec(WeightsSource::Dir(dir.to_path_buf()));
    let generator = mlx_gen_sana::load_sana(&spec)?;
    let request = GenerationRequest {
        prompt: "a lighthouse on a rocky coast at dawn".to_string(),
        width: size,
        height: size,
        count: 1,
        steps: Some(steps),
        // Fixed seed: the tiled and whole-image runs must denoise to the SAME latent, or the
        // comparison measures sampling variance instead of the decode.
        seed: Some(0),
        memory: Some(match tile {
            Some((edge, overlap)) => GenerationMemory {
                tile_vae_decode: true,
                decode_tile_edge: Some(edge),
                decode_overlap: Some(overlap),
                ..Default::default()
            },
            None => GenerationMemory::default(),
        }),
        ..Default::default()
    };
    // Read the geometry back rather than trusting the argument: this example has already shipped one
    // wrong table because the knob it set was not the knob the run used.
    match mlx_gen_sana::pipeline::resolved_decode_plan(request.memory, true) {
        Some(plan) => println!(
            "    decode: TILED edge={} overlap={} (chosen by {:?})",
            plan.edge, plan.overlap, plan.source
        ),
        None => println!("    decode: WHOLE-IMAGE"),
    }
    let mut noop = |_: Progress| {};
    match generator.generate(&request, &mut noop)? {
        GenerationOutput::Images(mut images) if !images.is_empty() => Ok(images.remove(0)),
        _ => Err("generator returned no image".into()),
    }
}

fn write_png(image: &Image, path: &Path) -> Result<(), Box<dyn std::error::Error>> {
    let buf: image::RgbImage =
        image::ImageBuffer::from_raw(image.width, image.height, image.pixels.clone())
            .ok_or("pixel buffer does not match dimensions")?;
    buf.save(path)?;
    Ok(())
}

fn run() -> Result<(), Box<dyn std::error::Error>> {
    let mut args = std::env::args().skip(1);
    let dir = args.next().ok_or(
        "usage: tiling_fidelity <snapshot_dir> [--size N] [--steps N] [--tiles a,b,c] [--out DIR]",
    )?;

    let mut size = 1024u32;
    let mut steps = 4u32;
    // The published ladder: the production path refuses an edge outside it.
    let mut tiles: Vec<i32> = mlx_gen_sana::memory_strategy::DECODE_TILE_EDGES
        .iter()
        .map(|&edge| edge as i32)
        .collect();
    // ONE overlap across the whole sweep, matching what the contract ladder publishes. Held fixed
    // so the edge is the only variable moving down the table.
    let mut overlap = mlx_gen_sana::pipeline::DECODE_OVERLAP as u32;
    let mut out_dir: Option<String> = None;
    while let Some(flag) = args.next() {
        match flag.as_str() {
            "--size" => size = args.next().ok_or("--size needs a value")?.parse()?,
            "--steps" => steps = args.next().ok_or("--steps needs a value")?.parse()?,
            "--tiles" => {
                tiles = args
                    .next()
                    .ok_or("--tiles needs a value")?
                    .split(',')
                    .map(str::parse)
                    .collect::<Result<_, _>>()?
            }
            "--overlap" => overlap = args.next().ok_or("--overlap needs a value")?.parse()?,
            "--out" => out_dir = Some(args.next().ok_or("--out needs a value")?),
            other => return Err(format!("unknown argument {other:?}").into()),
        }
    }
    let dir = Path::new(&dir);

    println!("DC-AE decode tiling fidelity\n  {size}px, {steps} steps, seed 0, overlap {overlap} (fixed)");

    // Baseline: an explicit whole-image request (see `generate`).
    let baseline = generate(dir, size, steps, None)?;
    println!("  baseline (whole-image decode, explicit request) rendered");
    if let Some(out) = &out_dir {
        write_png(&baseline, &Path::new(out).join("baseline.png"))?;
    }

    println!(
        "\n  {:>6}  {:>7}  {:>8}  {:>9}  {:>9}",
        "tile", "max |Δ|", "mean |Δ|", ">8/255", ">32/255"
    );
    for tile in &tiles {
        let tiled = generate(dir, size, steps, Some((*tile as u32, overlap)))?;
        let d = diff(&baseline, &tiled);
        println!(
            "  {tile:>6}  {:>7}  {:>8.3}  {:>8.2}%  {:>8.2}%",
            d.max, d.mean, d.over_8, d.over_32
        );
        if let Some(out) = &out_dir {
            write_png(&tiled, &Path::new(out).join(format!("tile_{tile}.png")))?;
        }
    }

    println!(
        "\n  A convolutional boundary artifact concentrates at seams and shrinks with overlap;\n  \
         a broad `>8/255` shift instead would mean a global reduction is being tiled."
    );
    Ok(())
}
