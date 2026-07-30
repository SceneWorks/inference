//! Measure an image generator's working set against an iOS per-app budget, **on macOS**.
//!
//! ```text
//! cargo run --release -p mlx-gen-ios-catalog --example image_budget -- <snapshot_dir>
//!     [--budget-mib N] [--steps N] [--size N] [--resident-only]
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
    GenerationOutput, GenerationRequest, LoadSpec, OffloadPolicy, Progress, WeightsSource,
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
    label: &str,
) -> Result<(usize, f64), Box<dyn std::error::Error>> {
    println!("\n{label}");

    memory::clear_cache();
    memory::reset_peak_memory();

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
        count: 1,
        steps: Some(steps),
        seed: Some(0),
        ..Default::default()
    };

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

    println!(
        "  after generation  active {:>7.0} MiB   peak {:>7.0} MiB   ({} image(s) in {secs:.1}s)",
        mib(memory::get_active_memory()),
        mib(peak),
        match &out {
            GenerationOutput::Images(images) => images.len(),
            _ => 0,
        }
    );

    drop(generator);
    memory::clear_cache();
    Ok((peak, secs))
}

fn run() -> Result<(), Box<dyn std::error::Error>> {
    let mut args = std::env::args().skip(1);
    let dir = args
        .next()
        .ok_or("usage: image_budget <snapshot_dir> [--budget-mib N] [--steps N] [--size N] [--resident-only]")?;

    let mut budget_mib = DEFAULT_BUDGET_MIB;
    let mut steps = 4u32; // SANA is few-step; 4 exercises the loop without dominating the runtime.
    let mut size = 1024u32;
    let mut resident_only = false;
    let mut sequential_only = false;
    let mut no_cache = false;
    while let Some(flag) = args.next() {
        match flag.as_str() {
            "--budget-mib" => {
                budget_mib = args.next().ok_or("--budget-mib needs a value")?.parse()?
            }
            "--steps" => steps = args.next().ok_or("--steps needs a value")?.parse()?,
            "--size" => size = args.next().ok_or("--size needs a value")?.parse()?,
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
    let budget_bytes = budget_mib * 1024 * 1024;
    println!(
        "iOS image-generation budget\n  budget {budget_mib} MiB ({:.1} GiB) -- an 8 GB device's \
         cap unless overridden\n  {size}px, {steps} steps",
        budget_mib as f64 / 1024.0
    );
    memory::set_memory_limit(budget_bytes);
    if no_cache {
        memory::set_cache_limit(0);
        println!("  cache limit 0 (buffer cache disabled)");
    }

    let (resident_peak, resident_secs) = if sequential_only {
        (0usize, 0.0f64)
    } else {
        measure(
            dir,
            OffloadPolicy::Resident,
            size,
            steps,
            "RESIDENT (all components co-resident)",
        )?
    };

    let sequential = if resident_only {
        None
    } else {
        Some(measure(
            dir,
            OffloadPolicy::Sequential,
            size,
            steps,
            "SEQUENTIAL (encode -> drop encoder -> denoise -> decode)",
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

    let resident_ok = verdict("resident", resident_peak, resident_secs);
    let sequential_ok = match sequential {
        Some((peak, secs)) => {
            let ok = verdict("sequential", peak, secs);
            let saved = resident_peak.saturating_sub(peak);
            println!(
                "  sequential saves {:.0} MiB ({:.0}%) for {:.0}% more time",
                mib(saved),
                100.0 * saved as f64 / resident_peak.max(1) as f64,
                100.0 * (secs / resident_secs.max(1e-6) - 1.0),
            );
            ok
        }
        None => resident_ok,
    };

    if !sequential_ok {
        println!(
            "\n  Over budget even sequentially. Options, in order: a smaller text encoder (the \
             2-bit Gemma-2 the crate docs mention), DC-AE tiling, or a smaller image size."
        );
        std::process::exit(1);
    }
    if !resident_ok {
        println!("\n  Requires OffloadPolicy::Sequential on iOS -- resident does not fit.");
    }
    Ok(())
}
