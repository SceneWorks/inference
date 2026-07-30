//! Measure a model's memory working set against an iOS per-app budget, **on macOS**.
//!
//! ```text
//! cargo run --release --example memory_budget -- <snapshot_dir> [--budget-mib N] [--tokens N]
//! cargo run --release --example memory_budget -- <dir_a> --also <dir_b>   # co-residency
//! ```
//!
//! # Why this exists
//!
//! iPhone enforces a per-app memory cap that does not scale with installed RAM (~6 GB on a 12 GB
//! device, ~4 GB on an 8 GB one). Exceeding it is a jetsam kill: the app "just closes", with no
//! crash log tying it to inference. Finding that boundary by trial on a phone means an
//! edit-build-install-launch cycle per attempt, on hardware that is somebody's daily driver.
//!
//! Almost none of that search needs the device. What a phone uniquely tells you is the *policy*
//! (the cap, jetsam, thermals); what a Mac tells you perfectly well is the *working set*, because
//! it is the same code, the same weights, and the same Metal allocator. So: find the number here,
//! confirm it there.
//!
//! # What this does and does not simulate
//!
//! **Measured faithfully:** MLX's active and peak allocation, the buffer cache, and how much of a
//! given budget the model actually occupies. These are allocator facts, not host-OS facts, so they
//! carry over to iOS.
//!
//! **Not simulated:** jetsam itself. `mlx::set_memory_limit` is *backpressure* — when active
//! memory exceeds it MLX blocks and waits for in-flight work rather than failing — so a run that
//! "stays under budget" here proves the working set fits, not that iOS will let it live. It also
//! says nothing about the host app's own footprint (UI, framework overhead), which shares the same
//! cap. Treat a pass as necessary, not sufficient, and confirm on device.
//!
//! The default budget is **4096 MiB — an 8 GB device's cap, not a 12 GB one's** — so the answer
//! generalizes downward rather than only describing the roomiest phone
//! (`docs/architecture/ios-project-spec.md` §0.1).

use std::path::Path;
use std::time::Instant;

use core_llm::{LoadSpec, Message, Sampling, TextLlmRequest};
use mlx_llm::load_for_model;
use mlx_rs::memory;

/// An 8 GB iPhone's approximate per-app cap. See the module docs for why this is the default.
const DEFAULT_BUDGET_MIB: usize = 4096;
const DEFAULT_TOKENS: u32 = 64;

fn mib(bytes: usize) -> f64 {
    bytes as f64 / (1024.0 * 1024.0)
}

/// MLX's allocator state at one instant.
struct Snapshot {
    active: usize,
    peak: usize,
    cache: usize,
}

impl Snapshot {
    fn take() -> Self {
        Self {
            active: memory::get_active_memory(),
            peak: memory::get_peak_memory(),
            cache: memory::get_cache_memory(),
        }
    }
}

fn report(label: &str, s: &Snapshot) {
    println!(
        "  {label:<22} active {:>8.0} MiB   peak {:>8.0} MiB   cache {:>7.0} MiB",
        mib(s.active),
        mib(s.peak),
        mib(s.cache)
    );
}

fn main() {
    if let Err(e) = run() {
        eprintln!("error: {e}");
        std::process::exit(1);
    }
}

/// Load a model, generate, and report the allocator's high-water mark. Returns peak bytes.
fn exercise(dir: &Path, tokens: u32, label: &str) -> Result<usize, Box<dyn std::error::Error>> {
    println!("\n{label}: {}", dir.display());

    memory::clear_cache();
    memory::reset_peak_memory();
    report("baseline", &Snapshot::take());

    let started = Instant::now();
    let llm = load_for_model(&LoadSpec::dense(dir.to_string_lossy().to_string()))?;
    let load_secs = started.elapsed().as_secs_f64();
    report("after load", &Snapshot::take());

    // Weights fault in lazily, so a load alone under-reports. Generating is what makes the working
    // set real — measuring before this would produce a reassuring and useless number.
    let request = TextLlmRequest {
        messages: vec![Message::user(
            "Write a detailed account of a long sea voyage.",
        )],
        sampling: Sampling::greedy(),
        max_new_tokens: tokens,
        seed: Some(0),
        ..Default::default()
    };
    let gen_started = Instant::now();
    let out = llm.complete(&request)?;
    let gen_secs = gen_started.elapsed().as_secs_f64();

    let after = Snapshot::take();
    report("after generation", &after);
    println!(
        "  {:<22} load {load_secs:.1}s   {} tok in {gen_secs:.1}s ({:.1} tok/s)",
        "timing",
        out.usage.generated_tokens,
        out.usage.generated_tokens as f64 / gen_secs.max(1e-6)
    );

    drop(llm);
    memory::clear_cache();
    report("after unload", &Snapshot::take());

    Ok(after.peak)
}

fn run() -> Result<(), Box<dyn std::error::Error>> {
    let mut args = std::env::args().skip(1);
    let primary = args.next().ok_or(
        "usage: memory_budget <snapshot_dir> [--also <dir>] [--budget-mib N] [--tokens N]",
    )?;

    let mut also: Option<String> = None;
    let mut budget_mib = DEFAULT_BUDGET_MIB;
    let mut tokens = DEFAULT_TOKENS;
    while let Some(flag) = args.next() {
        match flag.as_str() {
            "--also" => also = Some(args.next().ok_or("--also needs a directory")?),
            "--budget-mib" => {
                budget_mib = args.next().ok_or("--budget-mib needs a value")?.parse()?
            }
            "--tokens" => tokens = args.next().ok_or("--tokens needs a value")?.parse()?,
            other => return Err(format!("unknown argument {other:?}").into()),
        }
    }

    let budget_bytes = budget_mib * 1024 * 1024;
    println!(
        "iOS memory budget simulation\n  budget {budget_mib} MiB ({:.1} GiB) -- an 8 GB device's \
         cap unless overridden\n  host   macOS (allocator facts carry over; jetsam does not -- see \
         the module docs)",
        budget_mib as f64 / 1024.0
    );

    // Backpressure, not a hard cap: MLX blocks rather than failing when this is exceeded. It keeps
    // the run honest about pressure without turning an over-budget model into an opaque crash.
    let previous = memory::set_memory_limit(budget_bytes);
    println!("  (MLX limit {:.0} MiB -> {budget_mib} MiB)", mib(previous));

    let peak_primary = exercise(Path::new(&primary), tokens, "PRIMARY")?;

    let combined_peak = match &also {
        None => peak_primary,
        Some(second) => {
            // The co-residency question: after unloading the first model, does the second fit?
            // This is the sequential (staged) case — the one an 8 GB device needs when two models
            // cannot be held at once. True simultaneous co-residency is a device-only measurement.
            let peak_second = exercise(Path::new(second), tokens, "SECONDARY (after unload)")?;
            println!(
                "\n  staged handoff: primary peaked {:.0} MiB, secondary {:.0} MiB",
                mib(peak_primary),
                mib(peak_second)
            );
            peak_primary.max(peak_second)
        }
    };

    let headroom = budget_bytes as i64 - combined_peak as i64;
    let used_pct = 100.0 * combined_peak as f64 / budget_bytes as f64;
    println!(
        "\nVERDICT\n  peak working set {:.0} MiB of {budget_mib} MiB budget ({used_pct:.0}%)\n  \
         headroom {:.0} MiB",
        mib(combined_peak),
        headroom as f64 / (1024.0 * 1024.0)
    );

    // The host app's own footprint shares the cap, so "just fits" is not a pass. 20% is a
    // deliberately blunt stand-in for UI, framework overhead, and OS variance -- the point is to
    // refuse a verdict that would only hold on an idle device.
    if headroom < 0 {
        println!("  OVER BUDGET -- would not fit an 8 GB device; expect a jetsam kill");
        std::process::exit(1);
    } else if used_pct > 80.0 {
        println!("  TIGHT -- under budget, but <20% spare for the host app. Confirm on device.");
    } else {
        println!("  FITS -- with room for the host app. Confirm on device before relying on it.");
    }
    Ok(())
}
