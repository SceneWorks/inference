//! Measure an image generator's working set against an iOS per-app budget, **on macOS**.
//!
//! ```text
//! cargo run --release -p mlx-gen-ios-catalog --example image_budget -- <snapshot_dir>
//!     [--budget-mib N] [--steps N] [--size N] [--count N] [--resident-only]
//!     [--tile EDGE] [--overlap PX]
//! ```
//!
//! The media counterpart to `mlx-llm`'s `memory_budget`, and the reason it exists separately: a
//! diffusion model is not one weight set but three (text encoder, transformer, VAE decoder), used
//! in phases. That structure is what makes it *fit* a phone — or not.
//!
//! # The question this answers
//!
//! SANA's Q4 tier is ~5.6 GB on disk (Gemma-2 encoder 2.3 + DiT 2.0 + DC-AE 1.25). Held
//! co-resident that exceeds an 8 GB device's ~4 GB cap outright. But the phases are sequential —
//! encode, then denoise, then decode — so the encoder can be dropped before the transformer is
//! needed, bounding peak memory to `max(encoder, DiT + VAE)` rather than the sum.
//!
//! `gen_core::OffloadPolicy::Sequential` already implements that. This measures **both** policies
//! against a budget so the difference is a number rather than an assumption, which is what decides
//! whether SANA ships on iOS unchanged.
//!
//! # What this does and does not simulate
//!
//! Same as `memory_budget`: MLX's allocator accounting carries over to iOS because it is the same
//! code, weights, and Metal allocator. Jetsam does not — `set_memory_limit` is backpressure, so
//! staying under budget proves the working set fits, not that iOS will let the app live. Confirm
//! on device.

use std::path::Path;
use std::time::Instant;

use mlx_gen::gen_core::{
    GenerationMemory, GenerationOutput, GenerationRequest, LoadSpec, OffloadPolicy, Progress,
    WeightsSource,
};
use mlx_rs::memory;

/// An 8 GB iPhone's approximate per-app cap — the line worth generalizing to.
const DEFAULT_BUDGET_MIB: usize = 4096;

fn mib(bytes: usize) -> f64 {
    bytes as f64 / (1024.0 * 1024.0)
}

fn main() {
    if let Err(e) = run() {
        eprintln!("error: {e}");
        std::process::exit(1);
    }
}

/// Generate one image under `policy`, returning (peak bytes, seconds).
fn measure(
    dir: &Path,
    policy: OffloadPolicy,
    size: u32,
    steps: u32,
    count: u32,
    tile: Option<(u32, u32)>,
    label: &str,
) -> Result<(usize, f64), Box<dyn std::error::Error>> {
    println!("\n{label}");

    memory::clear_cache();
    memory::reset_peak_memory();
    // The number this harness exists to produce, and the one it was not producing.
    //
    // Every verdict below was decided on `get_peak_memory`, which is the high-water mark of LIVE
    // allocation and excludes MLX's reuse cache. A memory kill does not make that distinction —
    // Darwin's `phys_footprint`, which iOS jetsam reads, counts both. The two disagree in ORDER, not
    // just in magnitude: of three configurations measured here, the one with the lowest
    // `get_peak_memory` (Z-Image 1024/tile256, 3102 MiB) had the highest footprint (6488 MiB) and
    // was the only one the device refused. A budget ladder built on the peak alone can therefore
    // rank a fatal configuration as the safest of the set, which is exactly what happened.
    let probe = mlx_gen::memory_probe::FootprintProbe::start_default();

    let spec = LoadSpec {
        offload_policy: policy,
        ..LoadSpec::new(WeightsSource::Dir(dir.to_path_buf()))
    };

    let started = Instant::now();
    let generator = mlx_gen_sana::load_sana(&spec)?;
    println!(
        "  after load        active {:>7.0} MiB   peak {:>7.0} MiB",
        mib(memory::get_active_memory()),
        mib(memory::get_peak_memory())
    );

    let request = GenerationRequest {
        prompt: "a lighthouse on a rocky coast at dawn".to_string(),
        width: size,
        height: size,
        count,
        steps: Some(steps),
        seed: Some(0),
        // The REQUEST path, deliberately, and not `MLX_GEN_SANA_DECODE_TILE`. The env override
        // derives its overlap as `edge / 4` with no way to say otherwise, so every sweep driven
        // through it measures a different overlap at every edge. A contract ladder publishes ONE
        // overlap (`mlx_gen_pid::DecodeRoutes` carries a single `native_overlap`), so the sweep that
        // calibrates it has to be able to hold the overlap fixed while the edge moves.
        memory: tile.map(|(edge, overlap)| GenerationMemory {
            tile_vae_decode: true,
            decode_tile_edge: Some(edge),
            decode_overlap: Some(overlap),
            ..Default::default()
        }),
        ..Default::default()
    };

    // Read the geometry back off the resolver instead of asserting it. `MLX_GEN_SANA_DECODE_TILE`
    // beats the request by design, so a stale env var from an earlier run in the same shell would
    // silently re-measure `edge / 4` under a row labelled with the overlap asked for here. Printing
    // the winning SOURCE is what makes that visible rather than plausible.
    match mlx_gen_sana::pipeline::resolved_decode_plan(
        request.memory,
        policy == OffloadPolicy::Sequential,
    ) {
        Some(plan) => println!(
            "  decode geometry   TILED edge={} overlap={} (chosen by {:?})",
            plan.edge, plan.overlap, plan.source
        ),
        None => println!("  decode geometry   WHOLE-IMAGE"),
    }

    // Weights fault in lazily, so peak is only real after an actual generation — measuring at load
    // time reports a reassuring and useless number.
    // Cancellation rides `GenerationRequest::cancel`, not a separate argument.
    // Sample allocation at every progress callback. `Progress` names the phase, so this
    // localizes the peak to encode / denoise-step / decode rather than reporting one number for
    // the whole generation — which is what two failed weight-reduction attempts needed and did
    // not have.
    let mut trace: Vec<(String, f64, f64)> = Vec::new();
    let mut on_progress = |p: Progress| {
        // Collapse Step{current,total} to just "Step" so consecutive denoise steps merge below.
        let label = match p {
            Progress::Step { .. } => "Step".to_string(),
            other => format!("{other:?}"),
        };
        trace.push((
            label,
            mib(memory::get_active_memory()),
            mib(memory::get_peak_memory()),
        ));
    };
    let out = generator.generate(&request, &mut on_progress)?;

    // Collapse consecutive samples of the same phase to first/last, so a 4-step denoise is two
    // lines rather than four.
    let mut collapsed: Vec<(String, f64, f64)> = Vec::new();
    for sample in &trace {
        match collapsed.last_mut() {
            Some(prev) if prev.0 == sample.0 => {
                prev.1 = sample.1;
                prev.2 = sample.2;
            }
            _ => collapsed.push(sample.clone()),
        }
    }
    for (phase, active, peak) in &collapsed {
        println!("    {phase:<28} active {active:>7.0} MiB   peak {peak:>7.0} MiB");
    }
    let secs = started.elapsed().as_secs_f64();
    let peak = memory::get_peak_memory();
    let footprint = probe.finish() as usize;

    println!(
        "  after generation  active {:>7.0} MiB   peak {:>7.0} MiB   ({} image(s) in {secs:.1}s)",
        mib(memory::get_active_memory()),
        mib(peak),
        match &out {
            GenerationOutput::Images(images) => images.len(),
            _ => 0,
        }
    );
    println!(
        "  PEAK FOOTPRINT    {:>7.0} MiB   (active+cache; +{:.0} MiB over the reported peak)",
        mib(footprint),
        mib(footprint.saturating_sub(peak))
    );

    drop(generator);
    memory::clear_cache();
    // The FOOTPRINT is returned as the verdict quantity, not the peak. A budget is a statement about
    // what the OS will tolerate, and the OS tolerates footprint. Returning the peak here is what let
    // every verdict below be decided on a number no allocator enforces.
    Ok((footprint, secs))
}

fn run() -> Result<(), Box<dyn std::error::Error>> {
    let mut args = std::env::args().skip(1);
    let dir = args
        .next()
        .ok_or("usage: image_budget <snapshot_dir> [--budget-mib N] [--steps N] [--size N] [--count N] [--resident-only] [--tile EDGE] [--overlap PX]")?;

    let mut budget_mib = DEFAULT_BUDGET_MIB;
    let mut steps = 4u32; // SANA is few-step; 4 exercises the loop without dominating the runtime.
    let mut size = 1024u32;
    // Not cosmetic. Since the staged-decode change the count loop denoises EVERY seed before
    // anything decodes (so the trunk is shed once for the batch), which means phase C now runs N
    // decodes back-to-back inside one scope. A decode transient is the largest allocation in the
    // request, so whether two of them can be live at once is the difference between the published
    // peak holding for `count > 1` and not.
    let mut count = 1u32;
    let mut resident_only = false;
    let mut sequential_only = false;
    let mut no_cache = false;
    // The decode geometry to sweep, driven through the REQUEST rather than the env override so the
    // overlap is expressible independently of the edge (see `measure`). Two separate `Option`s so
    // `--overlap 48 --tile 512` and the reverse mean the same thing — a sweep loop sets one of them
    // per iteration, and an order-dependent parse would make half the rows measure something else.
    let mut tile_edge: Option<u32> = None;
    let mut tile_overlap: Option<u32> = None;
    while let Some(flag) = args.next() {
        match flag.as_str() {
            "--tile" => tile_edge = Some(args.next().ok_or("--tile needs a value")?.parse()?),
            "--overlap" => {
                tile_overlap = Some(args.next().ok_or("--overlap needs a value")?.parse()?)
            }
            "--budget-mib" => {
                budget_mib = args.next().ok_or("--budget-mib needs a value")?.parse()?
            }
            "--steps" => steps = args.next().ok_or("--steps needs a value")?.parse()?,
            "--size" => size = args.next().ok_or("--size needs a value")?.parse()?,
            "--count" => count = args.next().ok_or("--count needs a value")?.parse()?,
            "--resident-only" => resident_only = true,
            // Skip the resident pass. The two runs share a process, and MLX's peak is a
            // high-water mark, so a resident peak measured first masks any sequential improvement.
            "--sequential-only" => sequential_only = true,
            // Disable MLX's buffer cache entirely (freed blocks return to the system at once).
            // Diagnostic: a large drop here means the peak is retained-but-free memory, not live
            // allocation — a very different problem from "the model is too big".
            "--no-cache" => no_cache = true,
            // Diagnostic: force an eval between decoder stages. If the peak drops, the decode's
            // monolithic lazy graph is the cause and stage-wise eval is the fix.
            "--eval-stages" => std::env::set_var("MLX_GEN_DCAE_EVAL_STAGES", "1"),
            other => return Err(format!("unknown argument {other:?}").into()),
        }
    }

    let dir = Path::new(&dir);
    // `--overlap` without `--tile` is a mistake worth naming: it looks like it set the geometry and
    // sets nothing, because the request block only exists when an edge is named.
    if tile_overlap.is_some() && tile_edge.is_none() {
        return Err("--overlap needs a --tile to apply to".into());
    }
    let tile = tile_edge.map(|edge| (edge, tile_overlap.unwrap_or(edge / 4)));
    let budget_bytes = budget_mib * 1024 * 1024;
    println!(
        "iOS image-generation budget\n  budget {budget_mib} MiB ({:.1} GiB) -- an 8 GB device's \
         cap unless overridden\n  {size}px, {steps} steps, count {count}",
        budget_mib as f64 / 1024.0
    );
    memory::set_memory_limit(budget_bytes);
    if no_cache {
        memory::set_cache_limit(0);
        println!("  cache limit 0 (buffer cache disabled)");
    }

    // `Option`, not a `(0, 0.0)` sentinel. Under `--sequential-only` the zero flowed into the
    // verdict table and printed `resident peak 0 MiB (0%) FITS` — a PASS verdict for a
    // configuration that was never run, and the strongest-looking row in the output. It also made
    // the savings line divide by the zero, reporting "131461021% more time".
    let resident = if sequential_only {
        None
    } else {
        Some(measure(
            dir,
            OffloadPolicy::Resident,
            size,
            steps,
            count,
            tile,
            "RESIDENT (all components co-resident)",
        )?)
    };

    let sequential = if resident_only {
        None
    } else {
        Some(measure(
            dir,
            OffloadPolicy::Sequential,
            size,
            steps,
            count,
            tile,
            "SEQUENTIAL (encode -> drop encoder -> denoise -> shed DiT -> decode)",
        )?)
    };

    println!("\nVERDICT (budget {budget_mib} MiB)");
    let verdict = |label: &str, peak: usize, secs: f64| {
        let pct = 100.0 * peak as f64 / budget_bytes as f64;
        let status = if peak > budget_bytes {
            "OVER"
        } else if pct > 80.0 {
            "TIGHT"
        } else {
            "FITS"
        };
        println!(
            "  {label:<12} peak {:>7.0} MiB ({pct:>3.0}%)  {secs:>5.1}s  {status}",
            mib(peak)
        );
        peak <= budget_bytes
    };

    let resident_ok = resident.map(|(peak, secs)| verdict("resident", peak, secs));
    let sequential_ok = match sequential {
        Some((peak, secs)) => {
            let ok = verdict("sequential", peak, secs);
            // Only when both were actually measured. A comparison against a configuration that did
            // not run is not a weaker claim, it is a fabricated one.
            if let Some((resident_peak, resident_secs)) = resident {
                let saved = resident_peak.saturating_sub(peak);
                println!(
                    "  sequential saves {:.0} MiB ({:.0}%) for {:.0}% more time",
                    mib(saved),
                    100.0 * saved as f64 / resident_peak.max(1) as f64,
                    100.0 * (secs / resident_secs.max(1e-6) - 1.0),
                );
            }
            ok
        }
        // Nothing to say about a lane that was not run; fall back to whatever WAS measured.
        None => resident_ok.unwrap_or(false),
    };

    if !sequential_ok {
        println!(
            "\n  Over budget even sequentially. Options, in order: a smaller text encoder (the \
             2-bit Gemma-2 the crate docs mention), DC-AE tiling, or a smaller image size."
        );
        std::process::exit(1);
    }
    // `Some(false)` — measured and does not fit. `None` means resident was never run, and silence is
    // the honest output for a lane with no measurement behind it.
    if resident_ok == Some(false) {
        println!("\n  Requires OffloadPolicy::Sequential on iOS -- resident does not fit.");
    }
    Ok(())
}
