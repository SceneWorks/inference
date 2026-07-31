//! Does DC-AE decode tiling change the *generated* image? (E5, iOS memory work.)
//!
//! ```text
//! cargo run --release -p mlx-gen-ios-catalog --example tiling_fidelity -- <snapshot_dir>
//!     [--size N] [--steps N] [--tiles 512,256,192,128] [--overlap PX] [--out DIR]
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
//! DC-AE's decoder is `EfficientViTBlock` = `SanaMultiscaleLinearAttention → GLUMBConv`, and that
//! attention normalizes by `1/(Σ + eps)` **summed over every spatial position**. Tiling truncates
//! that sum to the tile. Unlike a convolution's boundary effect, no amount of overlap repairs it —
//! which is the hypothesis this quantifies on real content.

use std::path::Path;

use mlx_gen::gen_core::{
    GenerationMemory, GenerationOutput, GenerationRequest, Image, LoadSpec, OffloadPolicy,
    Progress, WeightsSource,
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

/// Render once. `tile` names the decode geometry through the **request**; `None` leaves whatever the
/// env override / residency default selects (which is how the whole-image baseline is taken).
///
/// The request path rather than `MLX_GEN_SANA_DECODE_TILE`, because the env override derives its
/// overlap as `edge / 4` with no way to say otherwise — and this example's whole job is to attribute a
/// pixel difference to one variable at a time. A sweep that moved the edge and the overlap together
/// cannot say which of them the difference came from.
fn generate(
    dir: &Path,
    size: u32,
    steps: u32,
    tile: Option<(u32, u32)>,
) -> Result<Image, Box<dyn std::error::Error>> {
    let spec = LoadSpec {
        // Sequential is the iOS policy, and the one whose peak this whole effort is bounding.
        offload_policy: OffloadPolicy::Sequential,
        ..LoadSpec::new(WeightsSource::Dir(dir.to_path_buf()))
    };
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
        memory: tile.map(|(edge, overlap)| GenerationMemory {
            tile_vae_decode: true,
            decode_tile_edge: Some(edge),
            decode_overlap: Some(overlap),
            ..Default::default()
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
    let mut tiles = vec![512i32, 384, 256, 192, 128];
    // ONE overlap across the whole sweep, matching what the contract ladder publishes
    // (`memory_strategy::DECODE_OVERLAP`, carried by `mlx_gen_pid::DecodeRoutes` as a single
    // `native_overlap`). Held fixed so the edge is the only variable moving down the table.
    let mut overlap = 48u32;
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

    // Baseline: whole-image, forced with `0`.
    //
    // This used to `remove_var`, on the reasoning that an unset variable selects the untiled path.
    // That stopped being true when SANA made tiling the default under `Sequential` — which is the
    // policy this example loads under (see `generate`). An unset variable now selects the provider
    // default of 128 px, so the "whole-image baseline" was a 128-tiled render and every row below
    // was reporting its difference from *that*.
    //
    // It failed loudly once you look: the 128 row came back max |Δ| 0, mean 0.000 — a tiled decode
    // cannot be bit-identical to an untiled one, and it was identical because it was being compared
    // against itself. Only `0` expresses whole-image now.
    std::env::set_var("MLX_GEN_SANA_DECODE_TILE", "0");
    let baseline = generate(dir, size, steps, None)?;
    println!("  baseline (whole-image decode, forced with MLX_GEN_SANA_DECODE_TILE=0) rendered");
    // The override has to come back OFF before the tiled rows, or it would beat every request block
    // below and silently re-render the baseline seven times. `remove_var` is correct here and was
    // wrong for the baseline — the same call means different things on the two sides of this line,
    // which is exactly why the baseline is now forced with an explicit `0`.
    std::env::remove_var("MLX_GEN_SANA_DECODE_TILE");
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
        "\n  A convolutional boundary artifact concentrates at seams and shrinks with overlap.\n  \
         DC-AE's per-tile attention normalizer does not: it shifts the whole tile's tone, so the\n  \
         `>8/255` column stays large however the tiles are drawn."
    );
    Ok(())
}
