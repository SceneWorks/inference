//! The bounded real-weight validation render (sc-24108) — `#[ignore]`d, needs the pinned
//! `Qwen/Qwen-Image-2.1` snapshot at `MLX_GEN_QWEN_IMAGE_2_1_SNAPSHOT` (inference never
//! self-fetches or derives a cache location, epic 13657) and the Metal GPU.
//!
//! Loads the released bf16 weights through the explicit catalog's production load path and
//! renders one image at the upstream default preset (1:1 2048×2048, 40 steps, seed 42, no
//! guidance), writing a PNG to `QWEN_IMAGE_2_1_RENDER_OUT` (default: the current directory). It
//! also pins the released tokenizer's system-prefix drop count (14).
//!
//! Run detached with an external RSS guard (the host has been kernel-panicked by an unguarded
//! MLX run before):
//!
//! ```sh
//! MLX_GEN_QWEN_IMAGE_2_1_SNAPSHOT=…/models--Qwen--Qwen-Image-2.1/snapshots/790c9263… \
//! QWEN_IMAGE_2_1_RENDER_OUT=~/SceneWorks/render-validation-sc-24108 \
//!   cargo test --locked --release -p mlx-gen-qwen-image-2-1 --test integration \
//!   e2e_real_weights:: -- --ignored --nocapture
//! ```
//!
//! `QWEN_IMAGE_2_1_RENDER_SIZE=WxH` and `QWEN_IMAGE_2_1_RENDER_STEPS=N` override the preset for a
//! quicker smoke.
//!
//! The `validation_render_turbo_*` tests are the distilled checkpoint's twins (sc-25900,
//! sc-25902): they read the unmodified `Qwen/Qwen-Image-2.1-Turbo` snapshot at
//! `MLX_GEN_QWEN_IMAGE_2_1_TURBO_SNAPSHOT`, load it as `qwen_image_2_1_turbo`, and leave the step
//! count to the snapshot's saved sigmas unless a test says otherwise. They need only that
//! snapshot, and honour the same `QWEN_IMAGE_2_1_*` switches as the base test they mirror —
//! except `validation_render_turbo_installed_tiers` (AT1, sc-25909), which reads the converted
//! q8/q4 Turbo tiers at `MLX_GEN_QWEN_IMAGE_2_1_TURBO_TIER_SNAPSHOT` instead.
//!
//! With `QWEN_IMAGE_2_1_RENDER_GOLDEN=<dir>` the two base render tests are also AT4's pre-epic
//! comparison (byte-equal against the PNGs the same tests wrote at `c1f594203`; *The pre-epic
//! golden (AT4)* in `UPSTREAM.md`).
//!
//! # The prefix KV cache (sc-25905)
//!
//! The cache is a request selection and an unset request runs prefill, so every render here
//! **states its mode** and asserts, on the generator's own report, that that is the path that
//! ran. Each test documents its default; `QWEN_IMAGE_2_1_PREFIX_KV_CACHE=cached|prefill` forces
//! one mode on every render of the run. A `[kv]` line beside the `[mem]` lines prints the path,
//! the bytes the caches held, the budget the request was priced at **on the tier it loaded**
//! (cache-shed transient + `prefix_kv_cache_demand_bytes`), and MLX's observed peak active memory
//! over the render — unconditionally, with or without `QWEN_IMAGE_2_1_MEMORY_TRACE`. Each
//! forced-shed twin pair turns two such peaks into AT3's `observed_cache_delta` and writes
//! `<stem>_prefix_kv_cache_memory.csv` into `QWEN_IMAGE_2_1_RENDER_OUT`. Defaults:
//!
//! | render | default | why |
//! |---|---|---|
//! | base text-to-image, base ten-reference boundary | prefill | the pre-epic path the published floors were derived for |
//! | base two-reference edit | cached | the caller-shaped request |
//! | every Turbo render, incl. the ten-reference edit | cached | what the epic ships |
//! | Turbo two- and ten-reference **forced-shed twins** (each right after its cached render) | prefill | the epic's "one run forced to shed that produces the prefill output" (AT3) |

use std::path::{Path, PathBuf};
use std::time::Instant;

use mlx_gen::gen_core::{GenerationMemory, PrefixKvCacheMode, Progress};
use mlx_gen::{GenerationOutput, GenerationRequest, Generator, LoadSpec, Quant, WeightsSource};
use mlx_gen_qwen_image_2_1::memory_strategy::derived;
use mlx_gen_qwen_image_2_1::{load_tokenizer, system_prompt_drop_count, Tier, PRESETS};
use mlx_rs::ops::indexing::IndexOp;

fn snapshot() -> PathBuf {
    let p = std::env::var("MLX_GEN_QWEN_IMAGE_2_1_SNAPSHOT").unwrap_or_else(|_| {
        panic!("set MLX_GEN_QWEN_IMAGE_2_1_SNAPSHOT to the pinned snapshot dir; inference never self-fetches (epic 13657)")
    });
    PathBuf::from(p)
}

fn turbo_snapshot() -> PathBuf {
    let p = std::env::var("MLX_GEN_QWEN_IMAGE_2_1_TURBO_SNAPSHOT").unwrap_or_else(|_| {
        panic!("set MLX_GEN_QWEN_IMAGE_2_1_TURBO_SNAPSHOT to the pinned Qwen-Image-2.1-Turbo snapshot dir; inference never self-fetches (epic 13657)")
    });
    PathBuf::from(p)
}

fn out_dir() -> PathBuf {
    let dir = std::env::var("QWEN_IMAGE_2_1_RENDER_OUT")
        .map(PathBuf::from)
        .unwrap_or_else(|_| PathBuf::from("."));
    std::fs::create_dir_all(&dir).unwrap();
    dir
}

/// `QWEN_IMAGE_2_1_PREFIX_KV_CACHE=cached|prefill`: force one prefix-KV-cache mode on every
/// render of the run. Unset, each render runs its own documented default.
const PREFIX_KV_CACHE_ENV: &str = "QWEN_IMAGE_2_1_PREFIX_KV_CACHE";

fn prefix_kv_cache_override() -> Option<PrefixKvCacheMode> {
    let value = std::env::var(PREFIX_KV_CACHE_ENV).ok()?;
    Some(match value.as_str() {
        "cached" => PrefixKvCacheMode::Cached,
        "prefill" => PrefixKvCacheMode::Prefill,
        other => panic!("{PREFIX_KV_CACHE_ENV} must be `cached` or `prefill`, got `{other}`"),
    })
}

/// `req` selecting the prefix-KV-cache `mode`, keeping whatever else its memory block carries.
fn selecting(mut req: GenerationRequest, mode: PrefixKvCacheMode) -> GenerationRequest {
    req.memory = Some(GenerationMemory {
        prefix_kv_cache: Some(mode),
        ..req.memory.unwrap_or_default()
    });
    req
}

/// What one [`render_reporting`] call observed beside its output.
#[derive(Clone, Copy, Debug)]
struct Observed {
    /// MLX's **peak active memory** over the render, in bytes — `get_peak_memory` reset before
    /// `generate_with_report` and read after it (and before every progress callback, so a
    /// `QWEN_IMAGE_2_1_MEMORY_TRACE` line resetting the counter mid-render cannot hide a peak).
    /// Observed, never committed.
    peak_active: u64,
    /// The bytes the caches held (`PrefixKvCacheReport::cache_bytes`).
    cache_bytes: u64,
    /// The cache surcharge the request is priced at on its tier (0 for prefill) — derived.
    priced_cache: u64,
}

/// One render of `req` (which must select `mode`) through `generate_with_report`, asserting that
/// the denoise loop ran the path the request selected, and printing the `[kv]` evidence line:
/// the path, the bytes the caches held, what the request is priced at **on `tier`** — its
/// cache-shed transient, plus `prefix_kv_cache_demand_bytes` when cached — and MLX's observed
/// peak active memory over the render. The priced figures are **derived**; the peak is the
/// observation [`observed_cache_delta_failures`] compares them with across a forced-shed twin.
fn render_reporting(
    generator: &dyn Generator,
    req: &GenerationRequest,
    mode: PrefixKvCacheMode,
    tier: Tier,
    label: &str,
    on_progress: &mut dyn FnMut(Progress),
) -> (GenerationOutput, Observed) {
    assert_eq!(
        req.memory.and_then(|memory| memory.prefix_kv_cache),
        Some(mode),
        "{label}: the request states its mode"
    );
    let active_before = mlx_rs::memory::get_active_memory() as u64;
    mlx_rs::memory::reset_peak_memory();
    let mut peak_active = 0_u64;
    let report = generator
        .generate_with_report(req, &mut |p| {
            peak_active = peak_active.max(mlx_rs::memory::get_peak_memory() as u64);
            on_progress(p)
        })
        .unwrap_or_else(|e| panic!("{label}: the render runs: {e}"));
    let peak_active = peak_active.max(mlx_rs::memory::get_peak_memory() as u64);
    let ran = report
        .prefix_kv_cache
        .expect("the path is reported on every render");
    let references = req.memory_reference_count();
    let negative = req.negative_prompt.is_some() && req.true_cfg.is_some_and(|scale| scale > 1.0);
    let shed = derived::request_transient_budget_bytes(
        tier,
        req.width,
        req.height,
        derived::TABLE_CONDITIONING_TOKENS,
        references,
        negative,
        None,
    );
    let surcharge = match mode {
        PrefixKvCacheMode::Cached => {
            derived::prefix_kv_cache_demand_bytes(tier, req.width, req.height, references, negative)
        }
        PrefixKvCacheMode::Prefill => 0,
    };
    eprintln!(
        "[kv] {label}: tier={} selected={} ran={} cache_bytes={} ({:.2} GiB) overflow={:?}; priced \
         transient = shed {:.2} GiB + cache {:.2} GiB = {:.2} GiB above the resident set \
         (derived); observed peak_active={:.2} GiB (active {:.2} GiB before the render)",
        tier.dir_name(),
        mode.label(),
        ran.path.label(),
        ran.cache_bytes,
        ran.cache_bytes as f64 / GIB,
        ran.overflow,
        shed as f64 / GIB,
        surcharge as f64 / GIB,
        (shed + surcharge) as f64 / GIB,
        peak_active as f64 / GIB,
        active_before as f64 / GIB,
    );
    assert_eq!(
        (ran.path, ran.overflow),
        (mode, None),
        "{label}: the render ran the path it selected — a `cached` request that reports an \
         overflow exceeded its priced prefix bound and is evidence, not noise"
    );
    assert_eq!(
        ran.cache_bytes > 0,
        mode.is_cached(),
        "{label}: a cached render holds a cache and a prefill render holds none"
    );
    assert!(
        ran.cache_bytes <= surcharge,
        "{label}: the caches hold {} bytes against {surcharge} priced",
        ran.cache_bytes
    );
    (
        report.output.expect("a render has an output"),
        Observed {
            peak_active,
            cache_bytes: ran.cache_bytes,
            priced_cache: surcharge,
        },
    )
}

const GIB: f64 = (1_u64 << 30) as f64;

/// The headroom the observed cache delta may show above the priced cache: 10 % of it plus 1 GiB.
///
/// `peak(cached) − peak(shed)` is what holding the cache adds to the render's high-water mark.
/// The priced figure is an upper bound on the cache itself (1024 text rows and 4920 tokens per
/// reference, against the ~4096 a square reference fits to), so the delta is expected **below**
/// it; the headroom is for what the two paths allocate differently besides the cache: the
/// extract evaluation's per-block contiguous K/V copies before the previous block's are released
/// (one block's worth, ~1/32 = 3 % of the cache), MLX's allocator rounding of the cached buffers
/// to its size classes (≤ a few % of a large buffer), and the eval pipeline's in-flight outputs —
/// the 1 GiB this crate's allocator bounds already budget as `MLX_EVAL_SLACK_BYTES` for the same
/// slack.
fn cache_delta_bound(priced: u64) -> u64 {
    priced + priced / 10 + (1 << 30)
}

/// The one memory claim a cached render and its forced-shed twin make together: holding the cache
/// raised the peak (the delta is positive) and by no more than its priced bytes plus
/// [`cache_delta_bound`]'s headroom. Prints the observed delta beside the priced bytes either way,
/// and **returns** what failed rather than panicking, so a render body collects it with every
/// other case's checks and asserts once, after its last case has printed its numbers.
/// `require_positive` is false only where a dry run's miniature cache can sit under another
/// phase's peak.
fn observed_cache_delta_failures(
    label: &str,
    cached_peak: u64,
    shed_peak: u64,
    priced: u64,
    require_positive: bool,
) -> Vec<String> {
    let delta = cached_peak as i128 - shed_peak as i128;
    let bound = cache_delta_bound(priced);
    eprintln!(
        "[kv] {label}: observed_cache_delta = peak(cached) {cached_peak} − peak(shed) {shed_peak} \
         = {delta} bytes ({:.3} GiB); priced cache bytes = {priced} ({:.3} GiB); bound = {bound} \
         ({:.3} GiB, priced · 1.10 + 1 GiB)",
        delta as f64 / GIB,
        priced as f64 / GIB,
        bound as f64 / GIB,
    );
    let mut failures = Vec::new();
    if delta > bound as i128 {
        failures.push(format!(
            "{label}: holding the cache raised the peak by {delta} bytes, above the {bound} its priced \
             {priced} bytes allow"
        ));
    }
    if require_positive && delta <= 0 {
        failures.push(format!(
            "{label}: the cached render peaked {delta} bytes against its shed twin — the cache \
             added nothing to the high-water mark"
        ));
    }
    failures
}

/// The bound and the sign check bite where they should, on synthetic peaks (no device).
///
/// *Mutations that red this:* dropping either check of [`observed_cache_delta_failures`];
/// a bound without the priced term or without the headroom.
#[test]
fn the_observed_cache_delta_is_bounded_by_the_priced_cache() {
    const G: u64 = 1 << 30;
    let priced = 24 * G;
    let check = |label, cached: u64, shed: u64, require_positive| {
        observed_cache_delta_failures(label, cached, shed, priced, require_positive)
    };
    // At the bound, and a little below it, pass; one byte over fails.
    assert!(check("at bound", 30 * G + cache_delta_bound(priced), 30 * G, true).is_empty());
    assert!(check("typical", 50 * G, 30 * G, true).is_empty());
    assert_eq!(
        check("over", 30 * G + cache_delta_bound(priced) + 1, 30 * G, true).len(),
        1
    );
    // No rise is refused only when a rise is required.
    assert_eq!(check("flat", 30 * G, 30 * G, true).len(), 1);
    assert!(check("flat dry run", 30 * G, 30 * G, false).is_empty());
    assert_eq!(cache_delta_bound(10 * G), 11 * G + G);
}

/// `QWEN_IMAGE_2_1_RENDER_GOLDEN=<dir>`: PNGs rendered by the **same named test at the pre-epic
/// revision** (`c1f594203`) into its `QWEN_IMAGE_2_1_RENDER_OUT`. Read only by the base-id tests
/// ([`validation_render_default_preset`], [`validation_render_reference_edit`]); see *The pre-epic
/// golden (AT4)* in this crate's `UPSTREAM.md` for the procedure.
const RENDER_GOLDEN_ENV: &str = "QWEN_IMAGE_2_1_RENDER_GOLDEN";

fn render_golden() -> Option<PathBuf> {
    std::env::var_os(RENDER_GOLDEN_ENV).map(PathBuf::from)
}

/// Compare the render just written as `file_name` with its pre-epic golden, **byte for byte** on
/// this backend (the MLX base forward was refactored by epic 25899 to be bit-identical; nothing
/// else moved), print the `[golden]` line, append it to `<out_dir>/golden_report.txt`, and
/// **return** the failure (a mismatch, a geometry change or a missing golden file) instead of
/// panicking — so one case's mismatch cannot cost the comparison of the cases after it; the body
/// asserts every result at once with [`assert_golden_clean`] after its last render. Only a
/// cache-shed render is comparable — the pre-epic revision had no cache — so a golden with a
/// cached render is a usage error and panics at once.
fn compare_with_golden(
    golden: Option<&Path>,
    out_dir: &Path,
    file_name: &str,
    image: &mlx_gen::gen_core::Image,
    mode: PrefixKvCacheMode,
) -> Option<String> {
    let golden = golden?;
    assert_eq!(
        mode,
        PrefixKvCacheMode::Prefill,
        "{file_name}: {RENDER_GOLDEN_ENV} compares the cache-shed path with the pre-epic render; \
         run with QWEN_IMAGE_2_1_PREFIX_KV_CACHE=prefill"
    );
    let path = golden.join(file_name);
    let (line, failure) = match image::open(&path) {
        Err(e) => {
            let failure = format!(
                "{file_name}: {RENDER_GOLDEN_ENV}: {} is the golden for this render: {e}",
                path.display()
            );
            (format!("[golden] {failure}"), Some(failure))
        }
        Ok(want) => {
            let want = want.to_rgb8();
            if (want.width(), want.height()) != (image.width, image.height) {
                let failure = format!(
                    "{file_name}: golden geometry {}x{} against the render's {}x{}",
                    want.width(),
                    want.height(),
                    image.width,
                    image.height
                );
                (format!("[golden] {failure}"), Some(failure))
            } else {
                let (mean, max) = pixel_difference(&image.pixels, want.as_raw());
                let equal = image.pixels == *want.as_raw();
                (
                    format!(
                        "[golden] {file_name}: mean |Δ| = {mean:.4}, max |Δ| = {max} (of 255) \
                         against {}; bar: byte-equal; {}",
                        path.display(),
                        if equal { "PASS" } else { "FAIL" }
                    ),
                    (!equal).then(|| {
                        format!(
                            "{file_name}: the cache-shed render is not the pre-epic render byte \
                             for byte (mean |Δ| {mean:.4}, max {max})"
                        )
                    }),
                )
            }
        }
    };
    eprintln!("{line}");
    record_golden_line(out_dir, &line);
    failure
}

/// Append one `[golden]` line to `<out_dir>/golden_report.txt` — the evidence a reviewer reads
/// for every compared render, written before any assertion.
fn record_golden_line(out_dir: &Path, line: &str) {
    use std::io::Write;
    let mut report = std::fs::OpenOptions::new()
        .create(true)
        .append(true)
        .open(out_dir.join("golden_report.txt"))
        .unwrap();
    writeln!(report, "{line}").unwrap();
}

/// The one AT4 assertion of a render body: every [`compare_with_golden`] result it collected,
/// after its last render.
fn assert_golden_clean(failures: &[String]) {
    assert!(
        failures.is_empty(),
        "{} render(s) differ from their pre-epic golden:\n{}",
        failures.len(),
        failures.join("\n")
    );
}

/// The `(current, total)` of every `Step` event a default-sampler render of `total` steps
/// reports: one per step, in order.
fn every_step(total: u32) -> Vec<(u32, u32)> {
    (1..=total).map(|current| (current, total)).collect()
}

/// Reject a render that is not a picture: a constant or near-constant RGB8 buffer (a black, white
/// or flat-grey frame). The floor is deliberately low — sixteen luma levels of spread and more
/// than sixteen distinct byte values — so it cannot reject a legitimately dark or soft image, only
/// a degenerate one.
fn assert_not_degenerate(label: &str, pixels: &[u8]) {
    let luma = |px: &[u8]| {
        (299 * u32::from(px[0]) + 587 * u32::from(px[1]) + 114 * u32::from(px[2])) / 1000
    };
    let (mut lo, mut hi) = (u32::MAX, 0u32);
    let mut seen = [false; 256];
    for px in pixels.chunks_exact(3) {
        let l = luma(px);
        lo = lo.min(l);
        hi = hi.max(l);
        for byte in px {
            seen[usize::from(*byte)] = true;
        }
    }
    let distinct = seen.iter().filter(|s| **s).count();
    eprintln!("{label}: luma range [{lo}, {hi}], {distinct} distinct byte values");
    assert!(
        hi >= lo + 16 && distinct > 16,
        "{label}: the render is a near-constant field (luma [{lo}, {hi}], {distinct} distinct \
         byte values), not a picture"
    );
}

/// A progress printer that reports the render clock **and** the interval since the previous
/// event (`+Δs`). The shared sampler evaluates step `k−1`'s graph only after it has reported
/// `Step { k }`, so the `+Δ` on the line *after* `step k` is the cost of step `k−1`'s forward,
/// `step 1` and `step 2` always land together, and the bare clock reads like an accelerating
/// per-step cost when it is a cumulative timestamp (that misread cost a GPU investigation on
/// sc-24114). The `+Δ` column is the per-step figure to quote.
fn progress_logger(label: String) -> impl FnMut(Progress) {
    let render_started = Instant::now();
    let mut previous = render_started;
    move |p| {
        let now = Instant::now();
        let clock = now.duration_since(render_started).as_secs_f32();
        let delta = now.duration_since(previous).as_secs_f32();
        previous = now;
        match p {
            Progress::Step { current, total } => {
                eprintln!("{label}step {current}/{total} ({clock:.1}s, +{delta:.1}s)")
            }
            other => eprintln!("{label}{other:?} ({clock:.1}s, +{delta:.1}s)"),
        }
    }
}

/// `phys_footprint` of this process — the counter the SceneWorks memory campaign's ceiling reads
/// (`physical_footprint_at_or_above_…`) — and its lifetime maximum, in bytes, via
/// `proc_pid_rusage(RUSAGE_INFO_V4)`. RSS is meaningless for Metal buffers; this is not.
#[path = "support/physical_footprint.rs"]
mod physical_footprint;
pub(crate) use physical_footprint::phys_footprint;

/// One `[mem]` line per phase boundary: MLX's active / cache / peak-active counters, the process
/// footprint, and the footprint high-water mark the sampler thread saw since the previous line.
/// Resets MLX's peak counter so the next line's `peak_active` is that phase's own high-water mark.
fn memory_line(label: &str, footprint_max: &std::sync::atomic::AtomicU64) {
    const GIB: f64 = (1u64 << 30) as f64;
    let (fp, fp_lifetime) = phys_footprint();
    let fp_phase = footprint_max.swap(0, std::sync::atomic::Ordering::Relaxed);
    eprintln!(
        "[mem] {label}: active={:.2} GiB cache={:.2} GiB peak_active={:.2} GiB footprint={:.2} GB \
         footprint_phase_max={:.2} GB footprint_lifetime_max={:.2} GB mlx_memory_limit={:.2} GB \
         mlx_cache_limit={:.2} GB",
        mlx_rs::memory::get_active_memory() as f64 / GIB,
        mlx_rs::memory::get_cache_memory() as f64 / GIB,
        mlx_rs::memory::get_peak_memory() as f64 / GIB,
        fp as f64 / 1e9,
        fp_phase.max(fp) as f64 / 1e9,
        fp_lifetime as f64 / 1e9,
        mlx_rs::memory::get_memory_limit() as f64 / 1e9,
        {
            // `get_cache_limit` is not bound; read it by setting and restoring.
            let current = mlx_rs::memory::set_cache_limit(0);
            mlx_rs::memory::set_cache_limit(current);
            current as f64 / 1e9
        },
    );
    mlx_rs::memory::reset_peak_memory();
}

/// A 50 ms footprint sampler so the per-phase footprint peak is observed, not just its value at
/// the boundary. Returns the shared high-water cell; the thread runs until the process exits.
fn footprint_sampler() -> std::sync::Arc<std::sync::atomic::AtomicU64> {
    let cell = std::sync::Arc::new(std::sync::atomic::AtomicU64::new(0));
    let shared = cell.clone();
    std::thread::spawn(move || loop {
        let (fp, _) = phys_footprint();
        shared.fetch_max(fp, std::sync::atomic::Ordering::Relaxed);
        std::thread::sleep(std::time::Duration::from_millis(50));
    });
    cell
}

/// Tokenizer only — no weights are opened. The released `processor/chat_template.jinja` renders a
/// lone system message as exactly the literal prefix the port tokenizes, so the derived drop count
/// is upstream's `_drop_idx` (14); `tools/_qwen21_common.py` re-proves the template/literal
/// agreement token-for-token through `Qwen3VLProcessor.apply_chat_template` whenever the snapshot
/// is present.
#[test]
#[ignore]
fn released_tokenizer_drops_fourteen_system_tokens() {
    let tokenizer = load_tokenizer(&snapshot()).unwrap();
    let count = system_prompt_drop_count(&tokenizer).unwrap();
    let ids = tokenizer
        .encode_ids(&mlx_gen_qwen_image_2_1::system_prefix(), true)
        .unwrap();
    eprintln!("released tokenizer: system prefix = {count} tokens {ids:?}");
    assert_eq!(count, 14);
    assert_eq!(
        ids,
        [151644, 8948, 198, 1092, 30782, 408, 323, 23643, 279, 3897, 9934, 13, 151645, 198]
    );
}

/// Config only — no weights are opened. The 128 transcribed `QWEN_IMAGE_2_1_Z64_MEAN` / `_STD`
/// floats that identify the latent space must be the released `vae/config.json`'s
/// `latents_mean` / `latents_std`, bit for bit after the f32 round both sides make.
#[test]
#[ignore]
fn latent_space_statistics_match_the_released_vae_config() {
    use mlx_gen::gen_core::{QWEN_IMAGE_2_1_Z64_MEAN, QWEN_IMAGE_2_1_Z64_STD};
    let cfg =
        mlx_gen_qwen_image_2_1::VaeConfig::from_json_file(&snapshot().join("vae/config.json"))
            .unwrap();
    assert_eq!(cfg.z_dim, 64);
    assert_eq!(cfg.scale_factor_spatial, 16);
    assert_eq!(cfg.latents_mean, QWEN_IMAGE_2_1_Z64_MEAN.to_vec());
    assert_eq!(cfg.latents_std, QWEN_IMAGE_2_1_Z64_STD.to_vec());
    eprintln!("released vae/config.json latents_mean/std == QWEN_IMAGE_2_1_Z64_MEAN/STD (64 + 64)");
}

/// The base checkpoint's default-preset render. With `QWEN_IMAGE_2_1_RENDER_GOLDEN=<dir>` it is
/// also AT4's pre-epic comparison: the PNG it writes must equal, byte for byte, the PNG of the
/// same name the same test wrote at `c1f594203` (see *The pre-epic golden (AT4)* in `UPSTREAM.md`).
#[test]
#[ignore]
fn validation_render_default_preset() {
    let steps: u32 = std::env::var("QWEN_IMAGE_2_1_RENDER_STEPS")
        .ok()
        .map(|s| s.parse().unwrap())
        .unwrap_or(mlx_gen_qwen_image_2_1::DEFAULT_STEPS);
    // Prefill by default: this is the pre-epic render the published floors were derived for.
    validation_render(
        &Target::dense(mlx_gen_qwen_image_2_1::MODEL_ID, snapshot()),
        &[Some(steps)],
        PrefixKvCacheMode::Prefill,
        render_golden().as_deref(),
    );
}

/// The distilled Turbo checkpoint's validation render (sc-25900): the same 1:1 2048×2048 preset,
/// seed and prompt as [`validation_render_default_preset`], through the `qwen_image_2_1_turbo`
/// id, with **no step count on the request** — so it runs the snapshot's own saved sigmas (eight
/// on the pinned revision) and asserts the sampler reported exactly that many steps. Needs the
/// unmodified `Qwen/Qwen-Image-2.1-Turbo` snapshot at `MLX_GEN_QWEN_IMAGE_2_1_TURBO_SNAPSHOT`.
/// `QWEN_IMAGE_2_1_RENDER_STEPS=N` names a step count instead (the resampled schedule), and every
/// other `QWEN_IMAGE_2_1_RENDER_*` / `QWEN_IMAGE_2_1_MEMORY_TRACE` switch applies unchanged.
///
/// ```sh
/// MLX_GEN_QWEN_IMAGE_2_1_TURBO_SNAPSHOT=…/models--Qwen--Qwen-Image-2.1-Turbo/snapshots/fc13ba19… \
/// QWEN_IMAGE_2_1_RENDER_OUT=~/SceneWorks/render-validation-sc-25900 \
///   cargo test --locked --release -p mlx-gen-qwen-image-2-1 --test integration \
///   e2e_real_weights::validation_render_turbo_default -- --ignored --nocapture
/// ```
#[test]
#[ignore]
fn validation_render_turbo_default() {
    let root = turbo_snapshot();
    let saved = mlx_gen_qwen_image_2_1::load_saved_sigma_schedule(&root)
        .unwrap()
        .default_steps();
    let steps: Option<u32> = std::env::var("QWEN_IMAGE_2_1_RENDER_STEPS")
        .ok()
        .map(|s| s.parse().unwrap());
    // Cached by default, like every Turbo render here.
    let reported = validation_render(
        &Target::dense(mlx_gen_qwen_image_2_1::TURBO_MODEL_ID, root),
        &[steps],
        PrefixKvCacheMode::Cached,
        None,
    );
    assert_eq!(
        reported,
        [steps.unwrap_or(saved)],
        "a request with no step count runs one step per saved sigma"
    );
}

/// The Turbo **step sweep** (sc-25902): the default 1:1 2048×2048 preset, same prompt and seed as
/// [`validation_render_turbo_default`], rendered at 4, 6, 12 and 16 steps — the resampled saved
/// schedule on both sides of the checkpoint's own eight — from one loaded generator, one PNG per
/// step count (`qwen_image_2_1_turbo_2048x2048_<N>steps_seed42.png`). It asserts only that every
/// count ran as asked; the PNGs are what the epic's terminal story reads to choose the manifest's
/// minimum step count. `QWEN_IMAGE_2_1_SWEEP_STEPS=4,6,12,16` names other counts, and every
/// `QWEN_IMAGE_2_1_RENDER_*` / `QWEN_IMAGE_2_1_MEMORY_TRACE` switch applies unchanged.
///
/// ```sh
/// MLX_GEN_QWEN_IMAGE_2_1_TURBO_SNAPSHOT=…/models--Qwen--Qwen-Image-2.1-Turbo/snapshots/fc13ba19… \
/// QWEN_IMAGE_2_1_RENDER_OUT=~/SceneWorks/render-validation-sc-25902 \
///   cargo test --locked --release -p mlx-gen-qwen-image-2-1 --test integration \
///   e2e_real_weights::validation_render_turbo_step_sweep -- --ignored --nocapture
/// ```
#[test]
#[ignore]
fn validation_render_turbo_step_sweep() {
    let steps: Vec<u32> = match std::env::var("QWEN_IMAGE_2_1_SWEEP_STEPS") {
        Ok(list) => list
            .split(',')
            .map(|steps| steps.trim().parse().expect("a step count"))
            .collect(),
        Err(_) => vec![4, 6, 12, 16],
    };
    let asked: Vec<Option<u32>> = steps.iter().copied().map(Some).collect();
    let reported = validation_render(
        &Target::dense(mlx_gen_qwen_image_2_1::TURBO_MODEL_ID, turbo_snapshot()),
        &asked,
        PrefixKvCacheMode::Cached,
        None,
    );
    assert_eq!(reported, steps, "every swept step count ran as asked");
}

/// What a render body renders with: the engine id, how the generator is loaded, the tier its
/// `[kv]` pricing is read at, and the stem its evidence files are named with.
struct Target {
    id: &'static str,
    spec: LoadSpec,
    tier: Tier,
    stem: String,
}

impl Target {
    /// The dense (bf16) snapshot at `root` under `id`; evidence named after the id, as every
    /// render here has always been.
    fn dense(id: &'static str, root: PathBuf) -> Self {
        Self {
            id,
            spec: LoadSpec::new(WeightsSource::Dir(root)),
            tier: Tier::Bf16,
            stem: id.to_owned(),
        }
    }

    /// The installed packed `tier` of `id` at `<tiers>/<q8|q4>/`, loaded with the matching
    /// quantize selector; evidence named `<id>_<tier>_…`.
    fn installed(id: &'static str, tiers: &Path, tier: Tier) -> Self {
        let quant = match tier {
            Tier::Q8 => Quant::Q8,
            Tier::Q4 => Quant::Q4,
            Tier::Bf16 => panic!("the dense tier is the snapshot itself, not an installed tier"),
        };
        Self {
            id,
            spec: LoadSpec::new(WeightsSource::Dir(tiers.join(tier.dir_name()))).with_quant(quant),
            tier,
            stem: format!("{id}_{}", tier.dir_name()),
        }
    }

    fn load(&self) -> Box<dyn Generator> {
        mlx_gen_qwen_image_2_1::provider_registry()
            .unwrap()
            .load(self.id, &self.spec)
            .unwrap_or_else(|e| panic!("{}: loads: {e}", self.stem))
    }
}

/// One bounded validation render of `target` per entry of `steps`, all from one loaded generator;
/// a `None` entry leaves the step count to the generator. Returns the step total the sampler
/// reported for each. Every render runs the prefix-KV-cache mode `default_cache` unless
/// `QWEN_IMAGE_2_1_PREFIX_KV_CACHE` forces one; `golden` is [`compare_with_golden`]'s.
fn validation_render(
    target: &Target,
    steps: &[Option<u32>],
    default_cache: PrefixKvCacheMode,
    golden: Option<&Path>,
) -> Vec<u32> {
    let (mut width, mut height) = (PRESETS[0].width, PRESETS[0].height);
    if let Ok(size) = std::env::var("QWEN_IMAGE_2_1_RENDER_SIZE") {
        let (w, h) = size.split_once('x').expect("WxH");
        width = w.parse().unwrap();
        height = h.parse().unwrap();
    }
    validation_render_at(
        target,
        &out_dir(),
        (width, height),
        steps,
        prefix_kv_cache_override().unwrap_or(default_cache),
        golden,
    )
}

/// [`validation_render`] at an explicit size into an explicit directory — the body, free of the
/// `QWEN_IMAGE_2_1_RENDER_SIZE` / `_RENDER_OUT` environment so the default-run dry run at the
/// bottom of this file can drive it on the miniature fixture.
fn validation_render_at(
    target: &Target,
    out_dir: &std::path::Path,
    (width, height): (u32, u32),
    steps: &[Option<u32>],
    cache: PrefixKvCacheMode,
    golden: Option<&Path>,
) -> Vec<u32> {
    let id = target.id;
    // `QWEN_IMAGE_2_1_MEMORY_TRACE=1` prints a `[mem]` line at every phase boundary (sc-24114's
    // footprint investigation); `QWEN_IMAGE_2_1_RENDER_TILED=1` asks for the bounded 512/64 decode
    // explicitly (the default already bounds every area above 512², sc-24114), and
    // `QWEN_IMAGE_2_1_CLEAR_CACHE_BEFORE_DECODE=1` sheds MLX's buffer cache at the `Decoding` event.
    let trace = std::env::var_os("QWEN_IMAGE_2_1_MEMORY_TRACE").is_some();
    let footprint_max = footprint_sampler();
    if trace {
        memory_line("before load", &footprint_max);
    }

    let started = Instant::now();
    let generator = target.load();
    eprintln!("loaded in {:.1}s", started.elapsed().as_secs_f32());
    if trace {
        memory_line("after load (lazy)", &footprint_max);
    }

    let memory = if std::env::var_os("QWEN_IMAGE_2_1_RENDER_TILED").is_some() {
        Some(mlx_gen::gen_core::GenerationMemory {
            tile_vae_decode: true,
            decode_tile_edge: Some(mlx_gen_qwen_image_2_1::pipeline::DECODE_TILE_EDGE),
            decode_overlap: Some(mlx_gen_qwen_image_2_1::pipeline::DECODE_OVERLAP),
            ..Default::default()
        })
    } else {
        None
    };
    let clear_before_decode =
        std::env::var_os("QWEN_IMAGE_2_1_CLEAR_CACHE_BEFORE_DECODE").is_some();
    let mut reported = Vec::with_capacity(steps.len());
    let mut renders: Vec<(u32, Vec<u8>)> = Vec::with_capacity(steps.len());
    let mut golden_failures = Vec::new();
    for steps in steps {
        let req = GenerationRequest {
            prompt: "A neon shop sign that reads \"QWEN IMAGE 2.1\", rainy night, reflections on wet pavement"
                .to_owned(),
            width,
            height,
            steps: *steps,
            seed: Some(42),
            memory,
            ..Default::default()
        };
        let req = selecting(req, cache);
        eprintln!(
            "decode tiling: {:?}",
            mlx_gen_qwen_image_2_1::pipeline::decode_tiling(&req)
        );
        let render_started = Instant::now();
        let mut log = progress_logger(String::new());
        let footprint_for_progress = footprint_max.clone();
        let mut reported_steps = Vec::new();
        let out = render_reporting(
            generator.as_ref(),
            &req,
            cache,
            target.tier,
            &format!("{} {width}x{height}", target.stem),
            &mut |p| {
                log(p);
                if let Progress::Step { current, total } = p {
                    reported_steps.push((current, total));
                }
                if trace {
                    let label = match &p {
                        Progress::Loading(phase) => format!("loading {phase:?}"),
                        Progress::Step { current, .. } => format!("step {current} reported"),
                        Progress::Decoding => "before decode".to_owned(),
                        Progress::Upsampling { tokens, max_tokens } => {
                            format!("upsampling {tokens}/{max_tokens}")
                        }
                    };
                    memory_line(&label, &footprint_for_progress);
                }
                if clear_before_decode && matches!(p, Progress::Decoding) {
                    mlx_rs::memory::clear_cache();
                    if trace {
                        memory_line("before decode, after clear_cache", &footprint_for_progress);
                    }
                }
            },
        );
        if trace {
            memory_line("after decode (generate returned)", &footprint_max);
            mlx_rs::memory::clear_cache();
            memory_line("after clear_cache", &footprint_max);
        }
        let GenerationOutput::Images(images) = out.0 else {
            panic!("images expected");
        };
        let image = &images[0];
        assert_eq!((image.width, image.height), (width, height));
        let ran = reported_steps
            .last()
            .expect("the sampler reports every step")
            .1;
        assert_eq!(
            reported_steps,
            every_step(ran),
            "one event per step, in order"
        );
        let file_name = format!("{}_{width}x{height}_{ran}steps_seed42.png", target.stem);
        let path = out_dir.join(&file_name);
        image::save_buffer(
            &path,
            &image.pixels,
            image.width,
            image.height,
            image::ColorType::Rgb8,
        )
        .unwrap();
        eprintln!(
            "wrote {} after {:.1}s total ({:.1}s render)",
            path.display(),
            started.elapsed().as_secs_f32(),
            render_started.elapsed().as_secs_f32()
        );
        golden_failures.extend(compare_with_golden(
            golden, out_dir, &file_name, image, cache,
        ));
        // Evidence someone reads to pick a step count must not be a blank frame, and two step
        // counts must not have produced the same picture (a dead `steps` knob).
        assert_not_degenerate(&format!("{} {ran} steps", target.stem), &image.pixels);
        for (other, pixels) in &renders {
            assert!(
                *other == ran || *pixels != image.pixels,
                "{id}: {other} and {ran} steps rendered identical pixels"
            );
        }
        renders.push((ran, image.pixels.clone()));
        reported.push(ran);
    }
    assert_golden_clean(&golden_failures);
    reported
}

/// A deterministic synthetic reference: three independent colour sawtooths. It is **not a
/// picture** — a real-weight edit over it faithfully reproduces the sawtooth as a colour mosaic (the
/// candle twin's CUDA evidence, run 38664658356: chroma busyness 32–41, [`assert_not_mosaic`]), so
/// it proves nothing about image correctness. It stays for the miniature-fixture dry runs, the
/// eleven-reference refusal (refused before any image is read), the base edit's AT4 golden
/// comparison (the pre-epic goldens were rendered over exactly these inputs), and as the negative
/// calibration of [`assert_not_mosaic`].
fn synthetic_reference(seed: u32, width: u32, height: u32) -> mlx_gen::gen_core::Image {
    let mut pixels = Vec::with_capacity((width * height * 3) as usize);
    for y in 0..height {
        for x in 0..width {
            pixels.push(((x * 3 + y * 5 + seed * 17) % 256) as u8);
            pixels.push(((x * 7 + seed * 29) % 256) as u8);
            pixels.push(((y * 11 + seed * 41) % 256) as u8);
        }
    }
    mlx_gen::gen_core::Image {
        width,
        height,
        pixels,
    }
}

/// Edge of every reference the real-weight edit smokes send (each is fitted to
/// `OUTPUT_RESOLUTION` by the processor whatever its size).
const REFERENCE_EDGE: u32 = 768;

/// The edit prompt every reference-edit render sends.
const EDIT_PROMPT: &str =
    "Combine the subjects of the reference images into one scene, evening light";

/// One subject per reference, in reference order — distinct objects, so a correct edit is a scene
/// visibly built from them. The candle twin renders the same list.
const REFERENCE_SUBJECTS: [&str; MAX_REFERENCES] = [
    "A red vintage bicycle leaning against a white brick wall, daylight photograph",
    "A golden retriever sitting on green grass, portrait photograph",
    "A blue ceramic teapot on a wooden table, studio photograph",
    "A yellow taxi cab parked on a city street, photograph",
    "A potted cactus with pink flowers on a sunny windowsill, photograph",
    "A white lighthouse on a rocky coast under a clear sky, photograph",
    "An orange tabby cat curled up on a knitted blanket, photograph",
    "A plate of colourful macarons on a marble counter, photograph",
    "A green velvet armchair in an empty bright room, photograph",
    "A striped hot air balloon over rolling hills, photograph",
];

/// `QWEN_IMAGE_2_1_EDIT_REFS`: a directory of real images (any format `image` reads) to use as
/// the references instead, in natural name order (operator-supplied).
const EDIT_REFS_ENV: &str = "QWEN_IMAGE_2_1_EDIT_REFS";

/// Where an edit smoke's references come from when `QWEN_IMAGE_2_1_EDIT_REFS` names no directory.
#[derive(Clone, Copy, Debug)]
enum ReferenceSource {
    /// [`synthetic_reference`] — for the dry runs and the AT4 golden comparison only.
    Synthetic,
    /// **Real pictures, rendered in the same run** by the generator under test
    /// ([`rendered_references`]): no reference images are provisioned with the snapshots, and the
    /// Qwen licence keeps none in this repository, so the run makes its own — `edge`²
    /// text-to-image renders of [`REFERENCE_SUBJECTS`], written beside the evidence.
    Rendered { edge: u32 },
}

impl ReferenceSource {
    /// Whether the edit output over these references must be a picture ([`assert_not_mosaic`]).
    fn is_real(self) -> bool {
        matches!(self, Self::Rendered { .. }) || std::env::var_os(EDIT_REFS_ENV).is_some()
    }
}

/// `count` text-to-image renders of [`REFERENCE_SUBJECTS`] by `generator`, `edge`² each, at the
/// generator's own step count and seeds `101..`, prefill (no cache selected), each written as
/// `<stem>_reference_<n>_<edge>x<edge>.png`. With `checks` (every real-weight caller; a dry run on
/// the miniature fixture renders noise by construction and passes `None`) each is also held to
/// [`band_failure`] and [`mosaic_failure`], and what fails is pushed there — so the edit cases over
/// a broken reference still render and print before the caller's one assertion.
fn rendered_references(
    generator: &dyn Generator,
    count: usize,
    edge: u32,
    out_dir: &Path,
    stem: &str,
    mut checks: Option<&mut Vec<String>>,
) -> Vec<mlx_gen::gen_core::Image> {
    assert!(count <= REFERENCE_SUBJECTS.len(), "{count} references");
    REFERENCE_SUBJECTS[..count]
        .iter()
        .enumerate()
        .map(|(i, subject)| {
            let label = format!("{stem} reference {}", i + 1);
            let started = Instant::now();
            let out = generator
                .generate(
                    &GenerationRequest {
                        prompt: (*subject).to_owned(),
                        width: edge,
                        height: edge,
                        seed: Some(101 + i as u64),
                        ..Default::default()
                    },
                    &mut |_| {},
                )
                .unwrap_or_else(|e| panic!("{label}: renders: {e}"));
            let GenerationOutput::Images(mut images) = out else {
                panic!("{label}: images expected");
            };
            let image = images.remove(0);
            assert_eq!((image.width, image.height), (edge, edge), "{label}");
            image::save_buffer(
                out_dir.join(format!("{stem}_reference_{}_{edge}x{edge}.png", i + 1)),
                &image.pixels,
                edge,
                edge,
                image::ColorType::Rgb8,
            )
            .unwrap();
            eprintln!(
                "{label}: {subject:?} in {:.1}s",
                started.elapsed().as_secs_f32()
            );
            if let Some(failures) = checks.as_deref_mut() {
                failures.extend(band_failure(&label, &image.pixels, edge, edge));
                failures.extend(mosaic_failure(&label, &image.pixels, edge, edge));
            }
            image
        })
        .collect()
}

/// Reject a render whose rows are stale GPU memory rather than a picture — the candle twin's
/// check (its sc-24114 CUDA evidence: 430 correct rows above 1,600 rows of noise). Per row, the
/// mean absolute second difference of the luma along the columns is ~3 for a rendered scene and
/// 20–35 for uninitialised memory; a row over 20 is static, and more than a quarter of them fails
/// the render. Prints the fraction and returns the failure rather than panicking.
fn band_failure(label: &str, pixels: &[u8], width: u32, height: u32) -> Option<String> {
    let (w, h) = (width as usize, height as usize);
    assert_eq!(pixels.len(), w * h * 3, "{label}: RGB8 geometry");
    assert!(w >= 3, "{label}: width {w} is too narrow to measure");
    let luma =
        |px: &[u8]| 0.299 * f32::from(px[0]) + 0.587 * f32::from(px[1]) + 0.114 * f32::from(px[2]);
    let static_rows = pixels
        .chunks_exact(w * 3)
        .filter(|row| {
            let l: Vec<f32> = row.chunks_exact(3).map(luma).collect();
            let hf = l
                .windows(3)
                .map(|t| (t[2] - 2.0 * t[1] + t[0]).abs())
                .sum::<f32>()
                / (w - 2) as f32;
            hf > 20.0
        })
        .count();
    let fraction = static_rows as f32 / h as f32;
    eprintln!(
        "{label}: static rows {static_rows}/{h} ({:.1}%)",
        100.0 * fraction
    );
    (fraction > 0.25).then(|| {
        format!(
            "{label}: {static_rows} of {h} rows are noise — the render is band-corrupted (a stale \
             or truncated device buffer reached the output)"
        )
    })
}

/// **Chroma busyness**: the RGB8 image box-averaged onto a 64×64 grid, then the mean absolute
/// difference between neighbouring cells of its two chroma channels (`B − Y`, `R − Y`, in 0..=255
/// levels), averaged over both axes and both channels. A photograph or a rendered scene changes
/// colour slowly at that scale; a colour mosaic changes it every few pixels. The grid makes it
/// independent of the render size (cells are `width/64 × height/64` pixels; a remainder is
/// ignored). The candle twin computes the same figure.
fn chroma_busyness(pixels: &[u8], width: u32, height: u32) -> f64 {
    const GRID: usize = 64;
    let (w, h) = (width as usize, height as usize);
    assert_eq!(pixels.len(), w * h * 3, "RGB8 geometry");
    assert!(
        w >= GRID && h >= GRID,
        "{w}x{h} is below the {GRID}-cell grid"
    );
    let (cw, ch) = (w / GRID, h / GRID);
    // One plane per chroma channel: `B − Y`, then `R − Y`.
    let mut planes = [vec![0f64; GRID * GRID], vec![0f64; GRID * GRID]];
    for gy in 0..GRID {
        for gx in 0..GRID {
            let mut sum = [0f64; 2];
            for y in gy * ch..(gy + 1) * ch {
                for x in gx * cw..(gx + 1) * cw {
                    let px = &pixels[(y * w + x) * 3..][..3];
                    let (r, g, b) = (f64::from(px[0]), f64::from(px[1]), f64::from(px[2]));
                    let luma = 0.299 * r + 0.587 * g + 0.114 * b;
                    sum[0] += b - luma;
                    sum[1] += r - luma;
                }
            }
            let n = (cw * ch) as f64;
            planes[0][gy * GRID + gx] = sum[0] / n;
            planes[1][gy * GRID + gx] = sum[1] / n;
        }
    }
    let mut total = 0f64;
    for plane in &planes {
        let (mut across, mut down) = (0f64, 0f64);
        for gy in 0..GRID {
            for gx in 0..GRID {
                let here = plane[gy * GRID + gx];
                if gx + 1 < GRID {
                    across += (plane[gy * GRID + gx + 1] - here).abs();
                }
                if gy + 1 < GRID {
                    down += (plane[(gy + 1) * GRID + gx] - here).abs();
                }
            }
        }
        let pairs = (GRID * (GRID - 1)) as f64;
        total += (across / pairs + down / pairs) / 2.0;
    }
    total / 2.0
}

/// The chroma busyness ([`chroma_busyness`]) above which a render is a colour mosaic, not a
/// picture — the candle twin's bar, calibrated once, offline, on images that exist: the CUDA edit
/// renders over synthetic references (run 38664658356) score 31.9 and 40.9; that run's Turbo
/// text-to-image renders 3.1–3.4 and every final render committed under `docs/migration/evidence/`
/// 3.0–19.3. 25 sits between the busiest real picture and the mildest mosaic.
const MOSAIC_CHROMA_BAR: f64 = 25.0;

/// Reject a render that is a colour mosaic rather than a picture — what an edit over the
/// synthetic sawtooth produces, and what no shape or blank-frame check catches.
fn assert_not_mosaic(label: &str, pixels: &[u8], width: u32, height: u32) {
    if let Some(failure) = mosaic_failure(label, pixels, width, height) {
        panic!("{failure}");
    }
}

thread_local! {
    /// The bar [`mosaic_failure`] applies on this thread: [`MOSAIC_CHROMA_BAR`], except inside
    /// [`the_edit_body_collects_every_check_and_asserts_after_its_last_case`], which lowers it to
    /// zero so every picture check of a miniature-fixture render fails on purpose.
    static MOSAIC_BAR: std::cell::Cell<f64> = const { std::cell::Cell::new(MOSAIC_CHROMA_BAR) };
}

/// [`assert_not_mosaic`]'s check, printed, returning the failure instead of panicking — for a
/// render body that collects every case's checks and asserts once, after its last case.
fn mosaic_failure(label: &str, pixels: &[u8], width: u32, height: u32) -> Option<String> {
    let busyness = chroma_busyness(pixels, width, height);
    let bar = MOSAIC_BAR.with(std::cell::Cell::get);
    eprintln!("{label}: chroma busyness {busyness:.2} (mosaic above {bar})");
    (busyness > bar).then(|| {
        format!(
            "{label}: chroma busyness {busyness:.2} > {bar} — a colour mosaic, not a \
             picture derived from its references"
        )
    })
}

/// [`assert_not_mosaic`] rejects the synthetic sawtooth references at the sizes an edit renders
/// and accepts a smooth synthetic scene with hard-edged shapes. Device-free.
#[test]
fn the_mosaic_check_rejects_the_synthetic_references_only() {
    for edge in [512, 768, 1024] {
        let mosaic = synthetic_reference(1, edge, edge);
        assert!(
            std::panic::catch_unwind(|| assert_not_mosaic("sawtooth", &mosaic.pixels, edge, edge))
                .is_err(),
            "the {edge}² sawtooth must be rejected"
        );
    }
    // A sky-to-ground gradient with three saturated discs: hard colour edges, but few of them.
    let edge = 512u32;
    let mut scene = Vec::with_capacity((edge * edge * 3) as usize);
    for y in 0..edge {
        for x in 0..edge {
            let disc = |cx: f32, cy: f32, r: f32| {
                ((x as f32 - cx).powi(2) + (y as f32 - cy).powi(2)).sqrt() < r
            };
            let px = if disc(128.0, 300.0, 70.0) {
                [220, 30, 30]
            } else if disc(300.0, 340.0, 60.0) {
                [30, 60, 220]
            } else if disc(420.0, 120.0, 50.0) {
                [250, 210, 40]
            } else if y < edge / 2 {
                [90, 150, 230 - (y / 4) as u8]
            } else {
                [60, 140 - (y / 8) as u8, 50]
            };
            scene.extend_from_slice(&px);
        }
    }
    assert_not_mosaic("scene", &scene, edge, edge);
}

/// A file name split into text and numeric runs (`ref10_x` → `["ref", 10, "_x"]`), so numbers
/// compare by value.
fn natural_key(name: &str) -> Vec<(String, u64)> {
    let mut key = Vec::new();
    let mut text = String::new();
    let mut rest = name;
    while !rest.is_empty() {
        let digits = rest.chars().take_while(char::is_ascii_digit).count();
        if digits > 0 {
            key.push((
                std::mem::take(&mut text),
                rest[..digits].parse().unwrap_or(u64::MAX),
            ));
            rest = &rest[digits..];
        } else {
            let c = rest.chars().next().unwrap();
            text.push(c);
            rest = &rest[c.len_utf8()..];
        }
    }
    if !text.is_empty() {
        key.push((text, 0));
    }
    key
}

/// The `count` ordered references of an edit smoke: the operator's `QWEN_IMAGE_2_1_EDIT_REFS`
/// directory when set, else `source`'s. `checks` is [`rendered_references`]'s.
fn references(
    generator: &dyn Generator,
    source: ReferenceSource,
    count: usize,
    out_dir: &Path,
    stem: &str,
    checks: Option<&mut Vec<String>>,
) -> Vec<mlx_gen::gen_core::Image> {
    let Ok(dir) = std::env::var(EDIT_REFS_ENV) else {
        return match source {
            ReferenceSource::Synthetic => (0..count)
                .map(|i| synthetic_reference(i as u32 + 1, REFERENCE_EDGE, REFERENCE_EDGE))
                .collect(),
            ReferenceSource::Rendered { edge } => {
                rendered_references(generator, count, edge, out_dir, stem, checks)
            }
        };
    };
    let mut paths: Vec<_> = std::fs::read_dir(&dir)
        .unwrap_or_else(|e| panic!("QWEN_IMAGE_2_1_EDIT_REFS={dir}: {e}"))
        .filter_map(|entry| entry.ok().map(|entry| entry.path()))
        .filter(|path| path.is_file())
        .collect();
    // Natural order, so `ref10_*` follows `ref9_*` rather than `ref1_*`: the two-reference case
    // takes the first two files a caller laid out, not a lexical accident.
    paths.sort_by_key(|path| natural_key(&path.file_name().unwrap_or_default().to_string_lossy()));
    assert!(
        paths.len() >= count,
        "QWEN_IMAGE_2_1_EDIT_REFS={dir} holds {} files, {count} needed",
        paths.len()
    );
    paths
        .into_iter()
        .take(count)
        .map(|path| {
            let rgb = image::open(&path)
                .unwrap_or_else(|e| panic!("{}: {e}", path.display()))
                .to_rgb8();
            mlx_gen::gen_core::Image {
                width: rgb.width(),
                height: rgb.height(),
                pixels: rgb.into_raw(),
            }
        })
        .collect()
}

/// One reference-edit render: how many references, the target size, the step count (`None`
/// leaves it to the generator) and the prefix-KV-cache mode it runs.
#[derive(Clone, Copy, Debug)]
struct EditCase {
    references: usize,
    width: u32,
    height: u32,
    steps: Option<u32>,
    cache: PrefixKvCacheMode,
}

/// One bounded real-weight **edit** render per reference count (sc-24110): the two-reference
/// 1024x1024 case a caller actually sends, and the ten-reference boundary at a small target so
/// the longest joint sequence is exercised without a long render. Finishes by proving eleven
/// references are refused at `validate`, before any weight is touched.
///
/// **Cache modes and the budgets they expect** (derived; bf16 tier; transient above the 28.6 GiB
/// resident set — printed on each render's `[kv]` line):
///
/// * two references, 1024² — **cached**: 10.38 GiB cache-shed transient + 5.30 GiB priced cache
///   (`prefix_kv_cache_demand_bytes`; ~4.0 GiB actually held for two square-fitted references and
///   a short prompt);
/// * ten references, 512² — **prefill**, deliberately: this is the pre-epic boundary render the
///   published floors were derived for, and it stays the evidence for the cache-shed budget
///   (10.38 GiB). It recomputes the ~41k-token prefix at each of its two steps, which is why it
///   runs the fewest steps the base sampler accepts.
///
/// `QWEN_IMAGE_2_1_PREFIX_KV_CACHE=cached|prefill` forces one mode on both — and
/// `QWEN_IMAGE_2_1_PREFIX_KV_CACHE=prefill QWEN_IMAGE_2_1_RENDER_GOLDEN=<dir>` is AT4's pre-epic
/// comparison: both PNGs must equal, byte for byte, the PNGs of the same names this test wrote at
/// `c1f594203` (see *The pre-epic golden (AT4)* in `UPSTREAM.md`).
///
/// Run detached under the RSS guard, exactly like [`validation_render_default_preset`]:
///
/// ```sh
/// MLX_GEN_QWEN_IMAGE_2_1_SNAPSHOT=.../snapshots/790c9263... \
/// QWEN_IMAGE_2_1_RENDER_OUT=~/SceneWorks/render-validation-sc-24110 \
///   cargo test --locked --release -p mlx-gen-qwen-image-2-1 --test integration \
///   e2e_real_weights::validation_render_reference_edit -- --ignored --nocapture --test-threads 1
/// ```
#[test]
#[ignore]
fn validation_render_reference_edit() {
    // Without a golden the references are real pictures rendered in the run and each edit output
    // must be a picture ([`assert_not_mosaic`]); with one they are the synthetic references the
    // pre-epic goldens were rendered over — the same inputs, which proves the port unchanged and
    // nothing about the image.
    let golden = render_golden();
    let source = if golden.is_some() {
        ReferenceSource::Synthetic
    } else {
        ReferenceSource::Rendered {
            edge: REFERENCE_EDGE,
        }
    };
    reference_edit_renders(
        &Target::dense(mlx_gen_qwen_image_2_1::MODEL_ID, snapshot()),
        source,
        mlx_gen_qwen_image_2_1::DEFAULT_STEPS,
        vec![
            EditCase {
                references: 2,
                width: 1024,
                height: 1024,
                steps: Some(8),
                cache: PrefixKvCacheMode::Cached,
            },
            EditCase {
                references: 10,
                width: 512,
                height: 512,
                steps: Some(2),
                cache: PrefixKvCacheMode::Prefill,
            },
        ],
        false,
        golden.as_deref(),
    );
}

/// The Turbo twin of [`validation_render_reference_edit`] (sc-25902), through the
/// `qwen_image_2_1_turbo` id on the unmodified Turbo snapshot: the two-reference 1024x1024 case
/// and the ten-reference boundary, **both with no step count on the request** — so each runs the
/// snapshot's saved sigmas (eight on the pinned revision), asserted from the steps the sampler
/// reported — and **both cached** (sc-25905): the ten-reference case is the epic's cached
/// ten-reference render, paying for its ~41k-token prefix once instead of eight times. Finishes
/// with the eleven-reference refusal.
///
/// Then the **forced-shed twins** (AT3): **each** cached case — the ten-reference one, which is
/// AT3's cached ten-reference run, and the two-reference one — rendered again right after it with
/// the cache shed, and compared with its cached render — the epic's "one run forced to shed that
/// produces the prefill output". The two paths are the same mathematics; each twin is held to a
/// full-resolution mean difference of at most [`SHED_TWIN_MEAN_BAR`] (2 of 255 — kept until an
/// MLX run measures otherwise; its doc says why CUDA's larger twin gap does not move it) **and**
/// to the same picture ([`TWIN_SAME_PICTURE_BAR`] on a 16×16 grid). Where the paths part, if they
/// do, is [`validation_render_turbo_prefix_kv_cache_velocity`]'s question. The references are real
/// pictures the run renders ([`ReferenceSource::Rendered`]), and every edit output must be a
/// picture ([`mosaic_failure`]).
///
/// Each twin pair is also AT3's memory comparison, printed **unconditionally**: MLX's peak active
/// memory over the cached render and over its shed twin, their difference
/// (`observed_cache_delta`) beside the priced cache bytes, held to `0 < delta ≤ priced · 1.10 +
/// 1 GiB` ([`observed_cache_delta_failures`]), and written to
/// `<stem>_prefix_kv_cache_memory.csv` in `QWEN_IMAGE_2_1_RENDER_OUT`.
///
/// **Budgets these renders expect** (derived; bf16 tier; transient above the Turbo checkpoint's
/// 28.0 GiB resident set — printed on each `[kv]` line):
///
/// * two references, 1024², cached: 10.38 GiB shed + 5.30 GiB priced cache (~4.0 GiB held);
/// * ten references, 512², cached: 10.38 GiB shed + 24.5 GiB priced cache (~20.0 GiB held for ten
///   square-fitted references) — so about **58 GiB active at peak** (28.0 + ~10 + ~20), under a
///   62.9 GiB memory limit. Size the RSS guard for it;
/// * the two twins, prefill: 10.38 GiB each.
///
/// `QWEN_IMAGE_2_1_PREFIX_KV_CACHE=cached|prefill` forces one mode on every render, and then the
/// twin is skipped (there is nothing to compare it with).
///
/// ```sh
/// MLX_GEN_QWEN_IMAGE_2_1_TURBO_SNAPSHOT=…/models--Qwen--Qwen-Image-2.1-Turbo/snapshots/fc13ba19… \
/// QWEN_IMAGE_2_1_RENDER_OUT=~/SceneWorks/render-validation-sc-25902 \
///   cargo test --locked --release -p mlx-gen-qwen-image-2-1 --test integration \
///   e2e_real_weights::validation_render_turbo_reference_edit -- --ignored --nocapture --test-threads 1
/// ```
#[test]
#[ignore]
fn validation_render_turbo_reference_edit() {
    let root = turbo_snapshot();
    let saved = mlx_gen_qwen_image_2_1::load_saved_sigma_schedule(&root)
        .unwrap()
        .default_steps();
    assert_eq!(saved, 8, "the pinned Turbo revision saves eight sigmas");
    reference_edit_renders(
        &Target::dense(mlx_gen_qwen_image_2_1::TURBO_MODEL_ID, root),
        ReferenceSource::Rendered {
            edge: REFERENCE_EDGE,
        },
        saved,
        vec![
            EditCase {
                references: 2,
                width: 1024,
                height: 1024,
                steps: None,
                cache: PrefixKvCacheMode::Cached,
            },
            EditCase {
                references: 10,
                width: 512,
                height: 512,
                steps: None,
                cache: PrefixKvCacheMode::Cached,
            },
        ],
        true,
        None,
    );
}

/// **Where the cached and prefill paths part, per model evaluation** — the MLX port of the candle
/// twin's `turbo_validation_prefix_kv_cache_velocity` (whose CUDA run root-caused the candle
/// forced-shed twin's divergence as benign bf16 rounding), so the MLX cached path carries the same
/// strict, device-independent bar. On the released Turbo weights, over the same rendered
/// references and the same two cases (two references at 1024², ten at 512²), it drives the DiT
/// directly along the **prefill** trajectory and, at every evaluation, also runs the other path on
/// the **same** latent and timestep:
///
/// * **evaluation 1** — `forward_joint_extract` against `forward_joint`: the same forward over the
///   same rows, so the velocity must be **bit-identical** (asserted);
/// * **the cache itself** — re-extracted at the last step, from a different latent and timestep:
///   the prefix modulates from `t = 0` and is causal, so every cached K/V row must be
///   **bit-identical** to the first extract's (asserted on blocks 0, middle and last);
/// * **every later evaluation** — `forward_joint_cached` against `forward_joint` on the same input:
///   the relative L2 gap `‖v_cached − v_prefill‖ / ‖v_prefill‖`, beside a **control** — the
///   prefill's own gap under a one-bf16-ulp perturbation of its input ([`ulp_nudged`]) — and held
///   to [`CACHED_GAP_CONTROL_RATIO_BAR`] times that control. Mathematically the gap is zero; in
///   bf16 the per-token projections run on `S` joint rows in the prefill and on the target rows
///   alone when cached, and a GEMM library may pick its kernel by the row count `M`. The `[gemm]`
///   lines measure that on this device for MLX's own matmul, with random weights at the DiT's
///   projection shapes and this case's `(S, target)` row counts (printed, never asserted);
/// * **block 0, every later evaluation** — the cached evaluation is traced
///   (`forward_joint_cached_traced`) and its `block_0_attn` / `_mlp` / `_out` held against the
///   **target rows** of the prefill's own block 0 on the same input
///   (`forward_joint_traced_through`, which traces the prefill only that far): at most
///   [`CACHED_BLOCK0_REL_BAR`] relative L2;
/// * every evaluation is judged on its own ([`cached_evaluation_failures`]): a non-finite gap or
///   control fails it, so a NaN cannot hide behind a `max`;
/// * **the trajectories** — the cached path's own trajectory is run beside the prefill one and the
///   relative latent gap printed per step, and both final latents are decoded and compared in
///   pixels and on the same-picture grid ([`grid_difference`]).
///
/// The bars are the candle twin's, calibrated on CUDA (their docs carry the runs). Both are
/// ratios or ulp-scale relative gaps measured against the same device's own numbers, so they carry
/// no device's absolute values — but **this test has never run on Metal**: it runs at the epic-end
/// MLX real-weight campaign, and its first run's numbers are the MLX measurement. Run it detached
/// with an external RSS guard, like every real-weight MLX run on the shared Mac.
///
/// ```sh
/// MLX_GEN_QWEN_IMAGE_2_1_TURBO_SNAPSHOT=…/models--Qwen--Qwen-Image-2.1-Turbo/snapshots/fc13ba19… \
/// QWEN_IMAGE_2_1_RENDER_OUT=…/render-validation \
///   cargo test --locked --release -p mlx-gen-qwen-image-2-1 --test integration \
///   e2e_real_weights::validation_render_turbo_prefix_kv_cache_velocity \
///   -- --ignored --exact --nocapture
/// ```
#[test]
#[ignore]
fn validation_render_turbo_prefix_kv_cache_velocity() {
    use mlx_gen::gen_core::Conditioning;
    use mlx_gen_qwen_image_2_1::{
        collect_references, create_noise, decode_rgb, encode_references, joint_branch,
        joint_images, load_saved_sigma_schedule, load_text_encoder, load_transformer, load_vae,
        prepare_conditioning_references, scheduler::resample_saved_sigmas,
    };
    use mlx_rs::ops::{add, multiply};
    use mlx_rs::Array;

    let root = turbo_snapshot();
    let out_dir = out_dir();
    let id = mlx_gen_qwen_image_2_1::TURBO_MODEL_ID;
    let saved = load_saved_sigma_schedule(&root).unwrap();
    let steps = saved.default_steps() as usize;
    assert_eq!(steps, 8, "the pinned Turbo revision saves eight sigmas");
    let sigmas = resample_saved_sigmas(&saved.sample_sigmas, steps).unwrap();

    let mut failures = Vec::new();
    // The references the edit smoke renders (same subjects, seeds and edge), made by the
    // generator, which is then dropped before the components load.
    let pool = {
        let generator = Target::dense(id, root.clone()).load();
        let source = ReferenceSource::Rendered {
            edge: REFERENCE_EDGE,
        };
        references(
            generator.as_ref(),
            source,
            MAX_REFERENCES,
            &out_dir,
            id,
            Some(&mut failures),
        )
    };

    let tokenizer = load_tokenizer(&root).unwrap();
    let drop_count = system_prompt_drop_count(&tokenizer).unwrap();
    let cases = [(2usize, 1024u32, 1024u32), (MAX_REFERENCES, 512, 512)];
    // Both cases' text side first, evaluated, so the tower is dropped before the DiT loads.
    let branches: Vec<_> = {
        let te = load_text_encoder(&root).unwrap();
        cases
            .iter()
            .map(|&(count, width, height)| {
                let req = GenerationRequest {
                    prompt: EDIT_PROMPT.to_owned(),
                    width,
                    height,
                    conditioning: vec![Conditioning::MultiReference {
                        images: pool[..count].to_vec(),
                    }],
                    ..Default::default()
                };
                let images = collect_references(&req).unwrap();
                let prepared = prepare_conditioning_references(&te, &images).unwrap();
                let conditioning = te
                    .encode_conditioning(&tokenizer, EDIT_PROMPT, drop_count, &prepared)
                    .unwrap();
                let branch = joint_branch(&conditioning, &prepared, width, height).unwrap();
                branch.text.eval().unwrap();
                (prepared, branch)
            })
            .collect()
    };
    let transformer = load_transformer(&root).unwrap();
    let vae = load_vae(&root).unwrap();
    let cfg = transformer.config().clone();
    let inner = cfg.inner_dim();

    for (&(count, width, height), (prepared, branch)) in cases.iter().zip(&branches) {
        let label = format!("{id} {count} refs {width}x{height}");
        let (total, target) = (branch.layout.total_len(), branch.layout.target_tokens());
        eprintln!(
            "[kv-velocity] {label}: joint S = {total} rows (prefix {}), target {target} rows",
            branch.layout.prefix_len()
        );
        gemm_row_probe(&label, total, target, inner, inner * cfg.mlp_ratio);

        let reference_latents = encode_references(&vae, prepared).unwrap();
        let text = &branch.text;
        let layout = &branch.layout;
        let prefill = |x: &Array, sigma: f32| {
            transformer
                .forward_joint(text, &joint_images(&reference_latents, x), sigma, layout)
                .unwrap()
        };
        let step = |x: &Array, v: &Array, dt: f32| {
            let next = add(x, multiply(v, Array::from_f32(dt)).unwrap()).unwrap();
            next.eval().unwrap();
            next
        };
        let mut x_prefill = create_noise(42, width, height, cfg.in_channels).unwrap();
        let mut x_cached = x_prefill.clone();
        let mut cache = None;
        let (mut worst_gap, mut worst_ratio, mut worst_block0) = (0f64, 0f64, 0f64);
        for i in 0..steps {
            let (sigma, dt) = (sigmas[i], sigmas[i + 1] - sigmas[i]);
            let v_prefill = prefill(&x_prefill, sigma);
            let v_cached = match &cache {
                None => {
                    let (v_extract, extracted) = transformer
                        .forward_joint_extract(
                            text,
                            &joint_images(&reference_latents, &x_prefill),
                            sigma,
                            layout,
                        )
                        .unwrap();
                    let (gap, max_abs, differing) = velocity_gap(&v_extract, &v_prefill);
                    eprintln!(
                        "[kv-velocity] {label}: step 1 extract vs prefill: rel {gap:.3e}, max \
                         |Δ| {max_abs:.3e}, {differing} elements differ"
                    );
                    if differing != 0 {
                        failures.push(format!(
                            "{label}: the extract evaluation is not bit-identical to the prefill \
                             ({differing} elements differ, rel {gap:.3e})"
                        ));
                    }
                    cache = Some(extracted);
                    v_extract
                }
                Some(cache) => {
                    // The cached evaluation on the prefill's own input, traced, so its block 0
                    // can be held against the prefill's block 0 below.
                    let (same_input, cached_trace) = transformer
                        .forward_joint_cached_traced(&x_prefill, sigma, cache)
                        .unwrap();
                    let (gap, max_abs, differing) = velocity_gap(&same_input, &v_prefill);
                    // The control: the prefill path's own response to an input perturbation of
                    // bf16-rounding size — what any one-ulp difference does to this velocity.
                    let (control, _, _) =
                        velocity_gap(&prefill(&ulp_nudged(&x_prefill), sigma), &v_prefill);
                    let ratio = gap / control.max(f64::MIN_POSITIVE);
                    let prefill_trace = transformer
                        .forward_joint_traced_through(
                            text,
                            &joint_images(&reference_latents, &x_prefill),
                            sigma,
                            layout,
                            0,
                        )
                        .unwrap();
                    let block0 = block0_target_gaps(&prefill_trace, &cached_trace, target);
                    drop(cached_trace);
                    // Block 0's own one-ulp control, printed beside it: the prefill's block 0 on
                    // the nudged input against the prefill's block 0 — the size of any one-ulp
                    // difference at this depth, on this device.
                    let nudged_trace = transformer
                        .forward_joint_traced_through(
                            text,
                            &joint_images(&reference_latents, &ulp_nudged(&x_prefill)),
                            sigma,
                            layout,
                            0,
                        )
                        .unwrap();
                    let nudged_rows: Vec<_> = prefill_trace
                        .iter()
                        .filter(|(name, _)| name.starts_with("block_0_"))
                        .map(|(name, full)| (name.clone(), target_rows(full, target)))
                        .collect();
                    let block0_control = block0_target_gaps(&nudged_trace, &nudged_rows, target);
                    drop((prefill_trace, nudged_trace, nudged_rows));
                    let rms = rms(&v_prefill);
                    eprintln!(
                        "[kv-velocity] {label}: step {} cached vs prefill on the same input: rel \
                         {gap:.3e}, max |Δ| {max_abs:.3e} (velocity rms {rms:.3e}), {differing} \
                         elements differ; one-ulp input control rel {control:.3e}; ratio \
                         {ratio:.2}",
                        i + 1
                    );
                    for ((stage, rel, max_abs, differing), (_, control_rel, _, _)) in
                        block0.iter().zip(&block0_control)
                    {
                        eprintln!(
                            "[kv-velocity] {label}: step {} {stage} target rows cached vs \
                             prefill: rel {rel:.3e}, max |Δ| {max_abs:.3e}, {differing} elements \
                             differ (bar rel {CACHED_BLOCK0_REL_BAR:.1e}); one-ulp input control \
                             rel {control_rel:.3e}",
                            i + 1
                        );
                    }
                    let block0: Vec<(String, f64)> = block0
                        .into_iter()
                        .map(|(stage, rel, _, _)| (stage, rel))
                        .collect();
                    failures.extend(cached_evaluation_failures(
                        &label,
                        i + 1,
                        gap,
                        control,
                        &block0,
                    ));
                    raise_worst(&mut worst_gap, gap);
                    raise_worst(&mut worst_ratio, ratio);
                    for (_, rel) in &block0 {
                        raise_worst(&mut worst_block0, *rel);
                    }
                    transformer
                        .forward_joint_cached(&x_cached, sigma, cache)
                        .unwrap()
                }
            };
            if i + 1 == steps {
                // The cache's step-independence, on this step's (different) latent and timestep.
                let first = cache.take().unwrap();
                let probe = [0, first.num_blocks() / 2, first.num_blocks() - 1];
                let host: Vec<_> = probe
                    .iter()
                    .map(|&b| {
                        let (k, v) = first.layer(b).unwrap();
                        (host_bits(k), host_bits(v))
                    })
                    .collect();
                drop(first);
                let (_, again) = transformer
                    .forward_joint_extract(
                        text,
                        &joint_images(&reference_latents, &x_prefill),
                        sigma,
                        layout,
                    )
                    .unwrap();
                for (&b, (k, v)) in probe.iter().zip(&host) {
                    let (k2, v2) = again.layer(b).unwrap();
                    let same = *k == host_bits(k2) && *v == host_bits(v2);
                    eprintln!(
                        "[kv-velocity] {label}: block {b} prefix K/V re-extracted at step {}: {}",
                        i + 1,
                        if same { "bit-identical" } else { "DIFFERENT" }
                    );
                    if !same {
                        failures.push(format!(
                            "{label}: block {b}'s prefix K/V changed between extracts at step 1 \
                             and step {}",
                            i + 1
                        ));
                    }
                }
            }
            x_prefill = step(&x_prefill, &v_prefill, dt);
            x_cached = step(&x_cached, &v_cached, dt);
            let (traj, _, _) = velocity_gap(&x_cached, &x_prefill);
            eprintln!(
                "[kv-velocity] {label}: after step {}: trajectory latent gap rel {traj:.3e}",
                i + 1
            );
        }
        let decode = |x: &Array| {
            decode_rgb(&vae, x, width, height, None, None)
                .unwrap()
                .pixels
        };
        let (pixels_prefill, pixels_cached) = (decode(&x_prefill), decode(&x_cached));
        let (mean, max) = pixel_difference(&pixels_cached, &pixels_prefill);
        eprintln!(
            "[kv-velocity] {label}: pixels cached vs prefill trajectory: mean |Δ| {mean:.4}, max \
             {max}, 16-grid {:.2}; worst per-evaluation velocity gap rel {worst_gap:.3e}, \
             {worst_ratio:.2}× the one-ulp control (bar {CACHED_GAP_CONTROL_RATIO_BAR}); worst \
             block-0 target-row gap rel {worst_block0:.3e} (bar {CACHED_BLOCK0_REL_BAR:.1e})",
            grid_difference(&pixels_cached, &pixels_prefill, width, height)
        );
        image::save_buffer(
            out_dir.join(format!(
                "{id}_velocity_{count}refs_{width}x{height}_{steps}steps_prefill.png"
            )),
            &pixels_prefill,
            width,
            height,
            image::ColorType::Rgb8,
        )
        .unwrap();
    }
    assert!(failures.is_empty(), "{}", failures.join("\n"));
}

/// How many times the one-ulp control's gap a cached evaluation's velocity may sit from the
/// prefill one on the same latent and timestep ([`validation_render_turbo_prefix_kv_cache_velocity`])
/// — the candle twin's bar, calibrated on CUDA: the real cached path 1.44× at its worst step (ten
/// references; two references 0×, bit-identical), reproduced bit for bit across runs 38668492569
/// and 38671361727; a one-position RoPE shift of the cached target rows 4.52× and 4.82× (mutation
/// run 38674397097). 2.5 is the geometric middle. A ratio to a control measured on the same
/// device, step and input, so it carries no device's absolute numbers; **not yet measured on
/// MLX**.
const CACHED_GAP_CONTROL_RATIO_BAR: f64 = 2.5;

/// The relative L2 gap block 0's target rows may show, cached against prefill on the same input
/// ([`block0_target_gaps`]; worst of `block_0_attn`, `_mlp`, `_out`, at any one evaluation).
/// Measured on CUDA (RTX PRO 6000 Blackwell, inference run 38683212858):
///
/// * the real cached path — two references: every stage of every evaluation **bit-identical**;
///   ten references: `block_0_attn` bit-identical, `block_0_mlp` 1.99e-3–2.16e-3 (its
///   `hidden → inner` projection is the one row-dependent GEMM, measured beside it at 2.15e-3),
///   `block_0_out` 3.3e-4–7.1e-4. Worst **2.16e-3**;
/// * the cached-decode RoPE-shift mutant (same run, a temporary reverted commit): worst
///   **2.15e-2** (two references) and **3.01e-2** (ten), both `block_0_mlp` at step 2; the gap
///   shrinks with the noise level (3.2e-3 / 3.8e-3 at step 8), but every evaluation is judged on
///   its own and steps 2–6 are over the bar in both cases.
///
/// 7e-3 is near the geometric middle (6.8e-3) of the real path's worst and the mutant's: 3.2×
/// headroom over the real path, 3.1× under the mutant. The block-0 one-ulp control is printed
/// beside each reading (1.3e-3–1.5e-2 by stage and step).
///
/// The candle twin's bar; **not yet measured on MLX** — the first MLX run at the epic-end campaign
/// is where it is checked.
const CACHED_BLOCK0_REL_BAR: f64 = 7e-3;

/// `x` with every element scaled by `1 ± 2⁻⁸` (a fixed pseudo-random sign per element): about
/// half a bf16 ulp on average, so once the DiT casts its input to bf16 roughly half the elements
/// move by one ulp — a perturbation the size of one bf16 rounding. The candle twin's control.
fn ulp_nudged(x: &mlx_rs::Array) -> mlx_rs::Array {
    let n: i32 = x.shape().iter().product();
    let factors: Vec<f32> = (0..n as u64)
        .map(|i| {
            let hash = i.wrapping_mul(0x9E37_79B9_7F4A_7C15) >> 63;
            if hash == 0 {
                1.0 + 1.0 / 256.0
            } else {
                1.0 - 1.0 / 256.0
            }
        })
        .collect();
    let factors = mlx_rs::Array::from_slice(&factors, x.shape());
    mlx_rs::ops::multiply(x, &factors).unwrap()
}

/// `(relative L2, max |Δ|, elements that differ)` of `a` against `b` (same shape, any float dtype).
fn velocity_gap(a: &mlx_rs::Array, b: &mlx_rs::Array) -> (f64, f32, usize) {
    assert_eq!(a.shape(), b.shape(), "same shape");
    let (a, b) = (crate::common::host_f32(a), crate::common::host_f32(b));
    let (mut diff2, mut ref2, mut max_abs, mut differing) = (0f64, 0f64, 0f32, 0usize);
    for (x, y) in a.iter().zip(&b) {
        let d = x - y;
        diff2 += f64::from(d) * f64::from(d);
        ref2 += f64::from(*y) * f64::from(*y);
        max_abs = max_abs.max(d.abs());
        differing += usize::from(x.to_bits() != y.to_bits());
    }
    (
        (diff2 / ref2.max(f64::MIN_POSITIVE)).sqrt(),
        max_abs,
        differing,
    )
}

/// Root mean square of an array's elements.
fn rms(t: &mlx_rs::Array) -> f64 {
    let v = crate::common::host_f32(t);
    (v.iter().map(|x| f64::from(*x).powi(2)).sum::<f64>() / v.len() as f64).sqrt()
}

/// The raw bits of an array's elements, on the host, for an exact comparison.
fn host_bits(t: &mlx_rs::Array) -> Vec<u32> {
    crate::common::host_f32(t)
        .into_iter()
        .map(f32::to_bits)
        .collect()
}

/// **The row-count dependence of a projection, measured on this device** — the candle twin's
/// probe, for MLX's matmul: for each of the DiT's per-token projection shapes (`inner → inner`,
/// `inner → hidden`, `hidden → inner`), a bf16 `x · Wᵀ` with random weights over `total` random
/// rows, against the same product over the last `target` of those rows alone — the prefill's and
/// the cached decode's `M` for the target tokens. Prints how many output elements of the target
/// rows differ; never asserted (which kernel a library picks for an `M` is the library's).
fn gemm_row_probe(label: &str, total: usize, target: usize, inner: usize, hidden: usize) {
    use mlx_rs::{random, Dtype};
    let (total, target) = (total as i32, target as i32);
    for (k, n) in [(inner, inner), (inner, hidden), (hidden, inner)] {
        let (k, n) = (k as i32, n as i32);
        let x = random::normal::<f32>(&[1, total, k][..], None, None, None)
            .unwrap()
            .as_dtype(Dtype::Bfloat16)
            .unwrap();
        let w = mlx_rs::ops::multiply(
            random::normal::<f32>(&[n, k][..], None, None, None).unwrap(),
            mlx_rs::Array::from_f32(1.0 / (k as f32).sqrt()),
        )
        .unwrap()
        .as_dtype(Dtype::Bfloat16)
        .unwrap();
        let wt = w.t();
        let full = mlx_rs::ops::matmul(&x, &wt).unwrap();
        let full_rows = full.index((.., total - target..total, ..));
        let alone = mlx_rs::ops::matmul(x.index((.., total - target..total, ..)), &wt).unwrap();
        let (gap, max_abs, differing) = velocity_gap(&alone, &full_rows);
        eprintln!(
            "[gemm] {label}: {k}->{n} over M={total} vs M={target}: {differing} of {} target \
             outputs differ (rel {gap:.3e}, max |Δ| {max_abs:.3e})",
            target * n
        );
    }
}

/// The last `target` rows of a `[1, S, C]` joint-sequence stage — the target block's.
fn target_rows(full: &mlx_rs::Array, target: usize) -> mlx_rs::Array {
    let rows = full.shape()[1];
    full.index((.., rows - target as i32..rows, ..))
}

/// The rows a block-0 comparison reads: each of `block_0_attn`, `block_0_mlp` and `block_0_out`,
/// the prefill trace's **target** rows (the last `target` of its `[1, S, inner]`) against the
/// cached trace's `[1, target, inner]` — `(stage, relative L2, max |Δ|, elements that differ)`.
fn block0_target_gaps(
    prefill: &[(String, mlx_rs::Array)],
    cached: &[(String, mlx_rs::Array)],
    target: usize,
) -> Vec<(String, f64, f32, usize)> {
    let find = |trace: &'_ [(String, mlx_rs::Array)], stage: &str| -> mlx_rs::Array {
        trace
            .iter()
            .find(|(name, _)| name == stage)
            .unwrap_or_else(|| panic!("the trace holds `{stage}`"))
            .1
            .clone()
    };
    ["block_0_attn", "block_0_mlp", "block_0_out"]
        .into_iter()
        .map(|stage| {
            let prefill_rows = target_rows(&find(prefill, stage), target);
            let (rel, max_abs, differing) = velocity_gap(&find(cached, stage), &prefill_rows);
            (stage.to_owned(), rel, max_abs, differing)
        })
        .collect()
}

/// `worst = max(worst, x)`, except that a NaN `x` replaces it (`f64::max` would drop it).
fn raise_worst(worst: &mut f64, x: f64) {
    if x.is_nan() || x > *worst {
        *worst = x;
    }
}

/// The verdict on one cached evaluation of [`validation_render_turbo_prefix_kv_cache_velocity`]:
/// its velocity gap to the prefill on the same input, the one-ulp `control`, and block 0's
/// target-row gaps `(stage, relative L2)`. **Per evaluation**, so no aggregate can hide one: a
/// non-finite gap or control (a NaN would otherwise sail through any `max` and every `>`), a
/// `gap / control` above [`CACHED_GAP_CONTROL_RATIO_BAR`], and a non-finite or above-bar block-0
/// gap ([`CACHED_BLOCK0_REL_BAR`]) each fail. The candle twin's verdict.
fn cached_evaluation_failures(
    label: &str,
    step: usize,
    gap: f64,
    control: f64,
    block0: &[(String, f64)],
) -> Vec<String> {
    let mut failures = Vec::new();
    if !gap.is_finite() || !control.is_finite() {
        failures.push(format!(
            "{label}: step {step}'s cached-vs-prefill velocity gap ({gap}) or its one-ulp control \
             ({control}) is not finite"
        ));
    } else {
        let ratio = gap / control.max(f64::MIN_POSITIVE);
        if ratio > CACHED_GAP_CONTROL_RATIO_BAR {
            failures.push(format!(
                "{label}: step {step}'s cached velocity is {ratio:.2}× further from the prefill \
                 one on the same input than the one-ulp control, above \
                 {CACHED_GAP_CONTROL_RATIO_BAR}"
            ));
        }
    }
    for (stage, rel) in block0 {
        if !(rel.is_finite() && *rel <= CACHED_BLOCK0_REL_BAR) {
            failures.push(format!(
                "{label}: step {step}'s {stage} target rows sit rel {rel:.3e} from the prefill's, \
                 above {CACHED_BLOCK0_REL_BAR:.1e} (or not finite)"
            ));
        }
    }
    failures
}

/// [`cached_evaluation_failures`] passes the real path's numbers and fails a NaN or infinite gap,
/// a NaN control, a ratio over the bar, and a NaN or over-bar block-0 gap — device-free.
///
/// *Mutations that red this:* dropping the finiteness check (a NaN gap passes every `>`); the
/// block-0 check comparing with `>` alone (a NaN passes); dropping the block-0 check.
#[test]
fn the_cached_evaluation_verdict_rejects_non_finite_and_over_bar_gaps() {
    let block0 = |rel: f64| vec![("block_0_attn".to_owned(), rel)];
    let verdict =
        |gap, control, rel| cached_evaluation_failures("t", 2, gap, control, &block0(rel));
    let control = 4e-2;
    let ok = CACHED_BLOCK0_REL_BAR / 2.0;
    assert!(verdict(0.0, control, 0.0).is_empty(), "bit-identical");
    assert!(
        verdict(1.44 * control, control, ok).is_empty(),
        "the measured real path"
    );
    assert!(
        verdict(
            CACHED_GAP_CONTROL_RATIO_BAR * control,
            control,
            CACHED_BLOCK0_REL_BAR
        )
        .is_empty(),
        "at both bars"
    );
    for (what, gap, control, rel) in [
        ("NaN gap", f64::NAN, control, ok),
        ("infinite gap", f64::INFINITY, control, ok),
        ("NaN control", 1e-2, f64::NAN, ok),
        ("ratio over the bar", 4.5 * control, control, ok),
        ("NaN block-0 gap", 0.0, control, f64::NAN),
        (
            "block-0 gap over the bar",
            0.0,
            control,
            CACHED_BLOCK0_REL_BAR * 1.01,
        ),
    ] {
        assert_eq!(verdict(gap, control, rel).len(), 1, "{what}");
    }
    let mut worst = 0.0;
    raise_worst(&mut worst, 1.0);
    raise_worst(&mut worst, 0.5);
    assert_eq!(worst, 1.0);
    raise_worst(&mut worst, f64::NAN);
    assert!(worst.is_nan(), "a NaN reaches the summary line");
}

/// The block-0 comparison's plumbing on the committed miniature snapshot:
/// `forward_joint_traced_through(.., 0)` traces block 0 and nothing after it,
/// [`block0_target_gaps`] reads the prefill's **target** rows against the cached decode's, and on
/// the same input they agree far inside [`CACHED_BLOCK0_REL_BAR`] — while against the prefill on
/// another latent, or against the prefill's leading (prefix) rows, they do not.
///
/// *Mutations that red this:* [`block0_target_gaps`] reading the first `target` rows; the seam
/// tracing past block 0 or stopping before it.
#[test]
fn the_block0_comparison_reads_the_target_rows_on_the_miniature_fixture() {
    use mlx_gen_qwen_image_2_1::{joint_images, load_transformer, JointLayout, Segment};
    use mlx_rs::Array;

    let model = load_transformer(&crate::common::tiny_snapshot()).unwrap();
    let cfg = model.config().clone();
    let seeded = |shape: [i32; 3], seed: f32| {
        let n: i32 = shape.iter().product();
        let values: Vec<f32> = (0..n).map(|i| (i as f32 * 0.37 + seed).sin()).collect();
        Array::from_slice(&values, &shape)
    };
    let layout = JointLayout {
        segments: vec![
            Segment::Text { len: 6 },
            Segment::Image {
                height: 2,
                width: 4,
            },
            Segment::Image {
                height: 4,
                width: 4,
            },
        ],
    };
    let target = layout.target_tokens();
    let (ctx, ch) = (cfg.context_in_dim as i32, cfg.in_channels as i32);
    let text = seeded([1, 6, ctx], 0.1);
    let refs = [seeded([1, 8, ch], 0.2)];
    let (_, cache) = model
        .forward_joint_extract(
            &text,
            &joint_images(&refs, &seeded([1, target as i32, ch], 0.3)),
            0.9,
            &layout,
        )
        .unwrap();
    let x = seeded([1, target as i32, ch], 0.4);
    let prefill = |x: &Array| {
        model
            .forward_joint_traced_through(&text, &joint_images(&refs, x), 0.5, &layout, 0)
            .unwrap()
    };
    let traced = prefill(&x);
    let stages: Vec<&str> = traced.iter().map(|(name, _)| name.as_str()).collect();
    assert!(stages.contains(&"block_0_out"), "{stages:?}");
    assert!(
        !stages.iter().any(|name| name.starts_with("block_1")),
        "{stages:?}"
    );
    let (_, cached) = model.forward_joint_cached_traced(&x, 0.5, &cache).unwrap();
    let same = block0_target_gaps(&traced, &cached, target);
    assert_eq!(same.len(), 3);
    for (stage, rel, _, _) in &same {
        assert!(
            *rel <= CACHED_BLOCK0_REL_BAR / 100.0,
            "{stage}: rel {rel:.3e}"
        );
    }
    let other = block0_target_gaps(
        &prefill(&seeded([1, target as i32, ch], 0.6)),
        &cached,
        target,
    );
    let leading: Vec<(String, Array)> = traced
        .iter()
        .filter(|(name, _)| name.starts_with("block_0_"))
        .map(|(name, t)| (name.clone(), t.index((.., 0..target as i32, ..))))
        .collect();
    let misread = block0_target_gaps(&leading, &cached, target);
    for gaps in [other, misread] {
        let worst = gaps.iter().map(|(_, rel, _, _)| *rel).fold(0.0, f64::max);
        assert!(worst > CACHED_BLOCK0_REL_BAR, "worst {worst:.3e}");
    }
}

/// The reference-edit smoke of `target`: one render per [`EditCase`] — `steps: None` leaves the
/// count to the generator, which must then run `default_steps` — each cached case followed, with
/// `forced_shed_twin`, by its forced-shed twin, then the eleven-reference refusal.
///
/// `QWEN_IMAGE_2_1_EDIT_CASES="refs:WxH:steps[:cached|prefill],…"` replaces the cases (sc-24114's
/// encode transient measurement: `2:2048x2048:2`; a case that names no mode runs **prefill**, the
/// path that measurement was made on); `QWEN_IMAGE_2_1_PREFIX_KV_CACHE` forces one mode on every
/// case and skips the twin; `QWEN_IMAGE_2_1_MEMORY_TRACE=1` prints the same `[mem]` phase lines
/// as the T2I render. The reference encodes are lazy, so they materialize inside the first
/// denoise step's eval: the phase that ends at `step 2 reported` is the one whose `peak_active`
/// carries the encoder's transient — and, on a cached render, the prefix caches being built.
fn reference_edit_renders(
    target: &Target,
    references: ReferenceSource,
    default_steps: u32,
    default_cases: Vec<EditCase>,
    forced_shed_twin: bool,
    golden: Option<&Path>,
) {
    let mut cases: Vec<EditCase> = match std::env::var("QWEN_IMAGE_2_1_EDIT_CASES") {
        Ok(spec) => spec
            .split(',')
            .map(|case| {
                let mut parts = case.split(':');
                let references = parts.next().unwrap().parse().unwrap();
                let (w, h) = parts.next().unwrap().split_once('x').expect("WxH");
                let steps = parts.next().unwrap().parse().unwrap();
                let cache = match parts.next() {
                    None | Some("prefill") => PrefixKvCacheMode::Prefill,
                    Some("cached") => PrefixKvCacheMode::Cached,
                    Some(other) => panic!("an edit case's mode is `cached` or `prefill`: {other}"),
                };
                EditCase {
                    references,
                    width: w.parse().unwrap(),
                    height: h.parse().unwrap(),
                    steps: Some(steps),
                    cache,
                }
            })
            .collect(),
        Err(_) => default_cases,
    };
    let forced = prefix_kv_cache_override();
    if let Some(mode) = forced {
        for case in &mut cases {
            case.cache = mode;
        }
    }
    reference_edit_renders_in(
        target,
        &out_dir(),
        references,
        default_steps,
        cases,
        forced_shed_twin && forced.is_none(),
        golden,
    );
}

/// `(mean, max)` absolute difference of two RGB8 buffers, in 0..=255 levels.
fn pixel_difference(a: &[u8], b: &[u8]) -> (f64, u8) {
    assert_eq!(a.len(), b.len(), "the two renders are the same size");
    let mut sum = 0_u64;
    let mut max = 0_u8;
    for (x, y) in a.iter().zip(b) {
        let d = x.abs_diff(*y);
        sum += u64::from(d);
        max = max.max(d);
    }
    (sum as f64 / a.len() as f64, max)
}

/// The reference count whose forced-shed twin must show a positive peak difference.
const MAX_REFERENCES: usize = mlx_gen_qwen_image_2_1::MAX_REFERENCE_IMAGES;

/// The **same-picture** difference: both RGB8 images box-averaged onto a 16×16 grid, then the
/// mean absolute difference of the cells over the three channels (0..=255 levels) — the candle
/// twin's figure. It ignores where an edge or a texture lands to within a cell and compares what
/// the picture is. Cells are `width/16 × height/16` pixels; a remainder is ignored.
fn grid_difference(a: &[u8], b: &[u8], width: u32, height: u32) -> f64 {
    const GRID: usize = 16;
    let (w, h) = (width as usize, height as usize);
    assert_eq!(a.len(), w * h * 3, "RGB8 geometry");
    assert_eq!(b.len(), a.len(), "the two renders are the same size");
    assert!(
        w >= GRID && h >= GRID,
        "{w}x{h} is below the {GRID}-cell grid"
    );
    let (cw, ch) = (w / GRID, h / GRID);
    let mut total = 0f64;
    for gy in 0..GRID {
        for gx in 0..GRID {
            let mut sums = [0f64; 6];
            for y in gy * ch..(gy + 1) * ch {
                for x in gx * cw..(gx + 1) * cw {
                    let at = (y * w + x) * 3;
                    for c in 0..3 {
                        sums[c] += f64::from(a[at + c]);
                        sums[3 + c] += f64::from(b[at + c]);
                    }
                }
            }
            let n = (cw * ch) as f64;
            total += (0..3)
                .map(|c| (sums[c] - sums[3 + c]).abs() / n)
                .sum::<f64>();
        }
    }
    total / (GRID * GRID * 3) as f64
}

/// The same-picture difference ([`grid_difference`]) a forced-shed twin may show against its
/// cached render — the candle twin's bar, derived from its CUDA measurement (inference run
/// 38668492569, real references): an eight-step flow over the ~42k-token ten-reference sequence
/// amplifies the ulp-level difference between the paths' bf16 arithmetic into a shifted but
/// identical scene (full-resolution mean 7.64, max 242; **3.41** on the grid), where the same
/// prompt at different step counts measures 4.50–5.08 and two different pictures 58.6–60.6.
/// **Not measured on MLX**: the first real MLX run prints its own numbers beside this bar. It is
/// held **in addition to** [`SHED_TWIN_MEAN_BAR`], not instead of it.
const TWIN_SAME_PICTURE_BAR: f64 = 15.0;

/// The mean absolute pixel difference (of 255) a forced-shed twin may show against its cached
/// render — this crate's tiling-parity bar, the twin bar MLX has carried since the cache landed.
/// **Not measured on released weights.** CUDA's ten-reference twin sits far above it (mean 7.64:
/// cuBLAS picks a different GEMM kernel for the cached decode's row count, and eight Turbo steps
/// amplify that bf16 rounding), but that is a measurement of cuBLAS, not of Metal: MLX's own
/// kernels choose by their own rules, and nothing yet shows the Metal paths part at all. So the
/// ≤ 2 bar **stays until an MLX run measures otherwise** — a real MLX twin above it is a finding to
/// root-cause (as the CUDA one was, with [`validation_render_turbo_prefix_kv_cache_velocity`]), not
/// a number to loosen the bar to. [`TWIN_SAME_PICTURE_BAR`] is held beside it, and the velocity
/// test is the cached path's per-evaluation check either way.
const SHED_TWIN_MEAN_BAR: f64 = 2.0;

/// What a forced-shed twin of the `references`-reference case fails: a full-resolution mean
/// difference above [`SHED_TWIN_MEAN_BAR`], and a same-picture difference above
/// [`TWIN_SAME_PICTURE_BAR`] — each its own failure, collected with the body's other checks.
fn twin_failures(references: usize, mean: f64, max: u8, same_picture: f64) -> Vec<String> {
    let mut failures = Vec::new();
    if mean.is_nan() || mean > SHED_TWIN_MEAN_BAR {
        failures.push(format!(
            "the {references}-reference forced-shed render differs from the cached one by a mean \
             of {mean:.4} levels (max {max}), above {SHED_TWIN_MEAN_BAR}; the two paths are the \
             same mathematics"
        ));
    }
    if same_picture.is_nan() || same_picture > TWIN_SAME_PICTURE_BAR {
        failures.push(format!(
            "the {references}-reference forced-shed render is a different picture from the cached \
             one: 16-grid difference {same_picture:.2} (full-resolution mean {mean:.4}, max \
             {max}), above {TWIN_SAME_PICTURE_BAR}"
        ));
    }
    failures
}

/// Both twin bars bite on their own — device-free.
///
/// *Mutations that red this:* dropping the mean bar (the CUDA-sized shift passes); dropping the
/// same-picture bar.
#[test]
fn the_twin_bars_hold_the_mean_and_the_picture_separately() {
    assert!(twin_failures(10, 0.0, 0, 0.0).is_empty(), "bit-identical");
    assert!(
        twin_failures(10, SHED_TWIN_MEAN_BAR, 40, 3.0).is_empty(),
        "at the mean bar"
    );
    // CUDA's measured ten-reference twin: the same picture, but over the MLX mean bar.
    let cuda_shift = twin_failures(10, 7.64, 242, 3.41);
    assert_eq!(cuda_shift.len(), 1, "{cuda_shift:?}");
    assert!(cuda_shift[0].contains("mean"), "{cuda_shift:?}");
    assert_eq!(twin_failures(2, 1.0, 9, 20.0).len(), 1, "another picture");
    assert_eq!(
        twin_failures(2, f64::NAN, 0, f64::NAN).len(),
        2,
        "NaN fails both"
    );
}

/// [`grid_difference`] passes a picture against itself shifted by a few pixels and fails it
/// against a different one — device-free.
#[test]
fn the_same_picture_difference_tolerates_a_shift_and_rejects_another_picture() {
    let (w, h) = (256u32, 256u32);
    let scene = |shift: u32, invert: bool| -> Vec<u8> {
        let mut px = Vec::with_capacity((w * h * 3) as usize);
        for y in 0..h {
            for x in 0..w {
                let x = x.saturating_sub(shift);
                let disc = ((x as f32 - 90.0).powi(2) + (y as f32 - 140.0).powi(2)).sqrt() < 50.0;
                let rgb = if disc {
                    [230u8, 40, 40]
                } else if y < h / 2 {
                    [80, 140, 220]
                } else {
                    [50, 120, 40]
                };
                px.extend(rgb.map(|v| if invert { 255 - v } else { v }));
            }
        }
        px
    };
    let base = scene(0, false);
    assert_eq!(grid_difference(&base, &base, w, h), 0.0);
    let shifted = grid_difference(&base, &scene(4, false), w, h);
    let other = grid_difference(&base, &scene(0, true), w, h);
    eprintln!("shifted {shifted:.2}, other {other:.2}");
    assert!(
        shifted <= TWIN_SAME_PICTURE_BAR,
        "a 4-px shift: {shifted:.2}"
    );
    assert!(other > TWIN_SAME_PICTURE_BAR, "another picture: {other:.2}");
}

/// [`reference_edit_renders`] over explicit cases into an explicit directory — the body, free of
/// the `QWEN_IMAGE_2_1_*` environment so the default-run dry run at the bottom of this file can
/// drive it on the miniature fixture.
///
/// `real_weights` (set by [`reference_edit_renders_in`], i.e. every real-weight caller; cleared
/// by the dry run) arms
///
/// * the sign half of [`observed_cache_delta_failures`] for the **ten-reference** twin — the one
///   where the cache dominates every other phase (24.5 GiB priced at bf16 against a ~10.6 GiB
///   denoise and a ≤ 9.4 GiB reference encode), so holding it must raise the peak. With fewer
///   references the reference encode can set both renders' peak and the difference is
///   legitimately near zero, so those twins are printed and held to the upper bound only (the
///   miniature cache is a few hundred kB under a VAE encode that is not);
/// * over real `references` ([`ReferenceSource::is_real`]), [`assert_not_mosaic`] on every edit
///   output — the miniature fixture's outputs are noise by construction.
///
/// The references are made (or read) once, after the generator loads, and every case takes the
/// first `count` of them in order.
#[allow(clippy::too_many_arguments)]
fn reference_edit_renders_in(
    target: &Target,
    out_dir: &std::path::Path,
    references: ReferenceSource,
    default_steps: u32,
    cases: Vec<EditCase>,
    forced_shed_twin: bool,
    golden: Option<&Path>,
) {
    reference_edit_renders_with(
        target,
        out_dir,
        references,
        default_steps,
        cases,
        forced_shed_twin,
        golden,
        true,
    )
}

/// [`reference_edit_renders_in`] with `real_weights` explicit — see there.
#[allow(clippy::too_many_arguments)]
fn reference_edit_renders_with(
    target: &Target,
    out_dir: &std::path::Path,
    references: ReferenceSource,
    default_steps: u32,
    cases: Vec<EditCase>,
    forced_shed_twin: bool,
    golden: Option<&Path>,
    real_weights: bool,
) {
    use mlx_gen::gen_core::Conditioning;

    let stem = target.stem.as_str();
    let started = Instant::now();
    let generator = target.load();
    eprintln!("loaded in {:.1}s", started.elapsed().as_secs_f32());
    // Every check of the body but the golden comparison (AT4, [`assert_golden_clean`]) — the
    // references', each edit output's, each twin's — is collected here and asserted once, after
    // the last case has rendered and printed its numbers.
    let mut failures: Vec<String> = Vec::new();
    let most = cases.iter().map(|case| case.references).max().unwrap_or(0);
    let pool = self::references(
        generator.as_ref(),
        references,
        most,
        out_dir,
        stem,
        real_weights.then_some(&mut failures),
    );
    let check_picture = real_weights && references.is_real();

    let trace = std::env::var_os("QWEN_IMAGE_2_1_MEMORY_TRACE").is_some();
    let footprint_max = footprint_sampler();
    // One render of `case`, with the cache mode `cache` (the case's own, or the twin's): its
    // pixels, its observation, its golden failure and its picture-check failure.
    let render = |case: EditCase,
                  cache: PrefixKvCacheMode,
                  suffix: &str|
     -> (Vec<u8>, Observed, Option<String>, Option<String>) {
        let EditCase {
            references: count,
            width,
            height,
            steps,
            ..
        } = case;
        let req = selecting(
            GenerationRequest {
                prompt: EDIT_PROMPT.to_owned(),
                width,
                height,
                steps,
                seed: Some(42),
                conditioning: vec![Conditioning::MultiReference {
                    images: pool[..count].to_vec(),
                }],
                ..Default::default()
            },
            cache,
        );
        let label = format!("{stem} {count} refs {}", cache.label());
        if trace {
            memory_line(&format!("{label}: before generate"), &footprint_max);
        }
        let render_started = Instant::now();
        let mut log = progress_logger(format!("{label}: "));
        let mut reported_steps = Vec::new();
        let (out, observed) = render_reporting(
            generator.as_ref(),
            &req,
            cache,
            target.tier,
            &label,
            &mut |p| {
                log(p);
                if let Progress::Step { current, total } = p {
                    reported_steps.push((current, total));
                }
                if trace {
                    memory_line(&format!("{label}: {p:?}"), &footprint_max);
                }
            },
        );
        if trace {
            memory_line(
                &format!("{label}: after decode (generate returned)"),
                &footprint_max,
            );
        }
        let GenerationOutput::Images(mut images) = out else {
            panic!("images expected");
        };
        let image = images.remove(0);
        assert_eq!((image.width, image.height), (width, height));
        let ran = steps.unwrap_or(default_steps);
        assert_eq!(
            reported_steps,
            every_step(ran),
            "{label}: the requested step count, or the checkpoint's own — one event per step"
        );
        let file_name =
            format!("{stem}_edit_{count}refs_{width}x{height}_{ran}steps_seed42{suffix}.png");
        let path = out_dir.join(&file_name);
        image::save_buffer(
            &path,
            &image.pixels,
            image.width,
            image.height,
            image::ColorType::Rgb8,
        )
        .unwrap();
        eprintln!(
            "wrote {} after {:.1}s render",
            path.display(),
            render_started.elapsed().as_secs_f32()
        );
        let golden_failure = compare_with_golden(golden, out_dir, &file_name, &image, cache);
        let picture_failure = check_picture
            .then(|| mosaic_failure(&label, &image.pixels, width, height))
            .flatten();
        (image.pixels, observed, golden_failure, picture_failure)
    };

    // Each twin is rendered **immediately after** the cached render it is compared with, so no
    // other request runs on the generator between the two sides of the comparison — the pixels
    // and the peaks are both read across that one pair.
    let mut memory_rows = vec![
        "case,references,mode,peak_active_bytes,cache_bytes,priced_cache_bytes,observed_cache_delta_bytes"
            .to_owned(),
    ];
    let mut twins = 0;
    let mut golden_failures = Vec::new();
    for case in cases {
        let (cached, cached_report, failure, picture) = render(case, case.cache, "");
        golden_failures.extend(failure);
        failures.extend(picture);
        if !forced_shed_twin || !case.cache.is_cached() {
            continue;
        }
        twins += 1;
        let (shed, shed_report, failure, picture) =
            render(case, PrefixKvCacheMode::Prefill, "_prefill_twin");
        golden_failures.extend(failure);
        failures.extend(picture);
        let (mean, max) = pixel_difference(&shed, &cached);
        let same_picture = grid_difference(&shed, &cached, case.width, case.height);
        eprintln!(
            "[kv] {stem} forced-shed twin of the {}-reference case: mean |Δ| = {mean:.4}, max |Δ| = \
             {max} (of 255), 16-grid {same_picture:.2} against its cached render; bars: mean <= \
             {SHED_TWIN_MEAN_BAR}, 16-grid <= {TWIN_SAME_PICTURE_BAR}",
            case.references
        );
        let delta = cached_report.peak_active as i128 - shed_report.peak_active as i128;
        for (mode, report, delta) in [
            ("cached", &cached_report, delta.to_string()),
            ("prefill_twin", &shed_report, String::new()),
        ] {
            memory_rows.push(format!(
                "{}refs_{}x{},{},{mode},{},{},{},{delta}",
                case.references,
                case.width,
                case.height,
                case.references,
                report.peak_active,
                report.cache_bytes,
                report.priced_cache,
            ));
        }
        // Written before the assertions, so a red run still leaves the numbers.
        std::fs::write(
            out_dir.join(format!("{stem}_prefix_kv_cache_memory.csv")),
            memory_rows.join("\n") + "\n",
        )
        .unwrap();
        failures.extend(observed_cache_delta_failures(
            &format!("{stem} {}-reference twin", case.references),
            cached_report.peak_active,
            shed_report.peak_active,
            cached_report.priced_cache,
            real_weights && case.references == MAX_REFERENCES,
        ));
        failures.extend(twin_failures(case.references, mean, max, same_picture));
    }
    assert!(
        !forced_shed_twin || twins > 0,
        "a forced-shed twin needs a cached case"
    );
    // After every case has rendered, written its PNG and its `[golden]` / `[kv]` lines — the
    // collected failures printed first, so a golden miss does not hide them.
    for failure in &failures {
        eprintln!("[check] FAILED: {failure}");
    }
    assert_golden_clean(&golden_failures);
    assert!(
        failures.is_empty(),
        "{} check(s) failed:\n{}",
        failures.len(),
        failures.join("\n")
    );

    let err = generator
        .validate(&GenerationRequest {
            prompt: "too many".to_owned(),
            width: 512,
            height: 512,
            // Refused at `validate`, before any image is looked at, so the eleven are synthetic
            // whatever `QWEN_IMAGE_2_1_EDIT_REFS` holds (a ten-file dir is the boundary itself).
            conditioning: vec![Conditioning::MultiReference {
                images: (1..=11).map(|i| synthetic_reference(i, 256, 256)).collect(),
            }],
            ..Default::default()
        })
        .err()
        .map(|e| e.to_string())
        .expect("eleven references must be refused");
    eprintln!("eleven references: {err}");
    assert!(err.contains("at most 10"), "{err}");
}

/// A **transparent** RGBA reference: a real 1024² picture `generator` renders in the run (the
/// first of [`rendered_references`]) matted onto a soft-edged disc so the alpha is a real ramp
/// rather than a binary cut-out (a hard matte would survive any resampling order and so could not
/// exercise the premultiplied resize).
///
/// Set `QWEN_IMAGE_2_1_RGBA_REF` to a PNG with an alpha channel to extract from that image
/// instead (operator-supplied; nothing is rendered then).
fn transparent_reference(
    generator: &dyn Generator,
    out_dir: &Path,
    stem: &str,
) -> mlx_gen::RgbaImage {
    if let Ok(path) = std::env::var("QWEN_IMAGE_2_1_RGBA_REF") {
        let rgba = image::open(&path)
            .unwrap_or_else(|e| panic!("QWEN_IMAGE_2_1_RGBA_REF={path}: {e}"))
            .to_rgba8();
        return mlx_gen::RgbaImage {
            width: rgba.width(),
            height: rgba.height(),
            pixels: rgba.into_raw(),
        };
    }
    let mut failures = Vec::new();
    let picture =
        rendered_references(generator, 1, 1024, out_dir, stem, Some(&mut failures)).remove(0);
    assert!(failures.is_empty(), "{}", failures.join("\n"));
    let (width, height) = (picture.width, picture.height);
    let mut out = mlx_gen::RgbaImage::from_rgb(&picture).unwrap();
    let (cx, cy) = ((width - 1) as f32 / 2.0, (height - 1) as f32 / 2.0);
    let radius = width.min(height) as f32 * 0.36;
    for (i, px) in out.pixels.chunks_exact_mut(4).enumerate() {
        let (x, y) = ((i as u32 % width) as f32, (i as u32 / width) as f32);
        let dist = ((x - cx).powi(2) + (y - cy).powi(2)).sqrt();
        // A 3-px linear ramp at the disc boundary: opaque inside, transparent outside.
        px[3] = (((radius + 1.5 - dist) / 3.0).clamp(0.0, 1.0) * 255.0).round() as u8;
    }
    out
}

/// The alpha histogram claim this smoke exists to make: the emitted alpha must be **non-trivial**.
///
/// "Four channels came back" is not evidence of transparency — a port that widened an RGB decode
/// with a constant `A = 255` would satisfy it. So this asserts three things that a constant or
/// near-constant plane cannot satisfy together, prints the full 16-bucket histogram, and writes it
/// beside the PNG:
///
/// * the alpha is not constant;
/// * a real share of the image is substantially transparent (`A < 128` on at least 2 % of pixels)
///   **and** a real share is substantially opaque (`A > 200` on at least 2 %) — i.e. the matte
///   separates figure from ground rather than dimming everything uniformly;
/// * compositing over white actually changes the picture (the RGB a flattening consumer would see
///   differs from the raw colour on at least 2 % of pixels), which is what makes the alpha
///   load-bearing rather than decorative.
fn assert_nontrivial_alpha(label: &str, image: &mlx_gen::RgbaImage, out_dir: &std::path::Path) {
    let alpha: Vec<u8> = image.pixels.chunks_exact(4).map(|px| px[3]).collect();
    let total = alpha.len();
    assert!(total > 0, "{label}: empty image");

    let mut histogram = [0usize; 16];
    for &a in &alpha {
        histogram[(a as usize) / 16] += 1;
    }
    let (min, max) = (*alpha.iter().min().unwrap(), *alpha.iter().max().unwrap());
    let transparent = alpha.iter().filter(|&&a| a < 128).count();
    let opaque = alpha.iter().filter(|&&a| a > 200).count();

    // How much the white composite moves the colour.
    let flattened = image.to_rgb_over_white().unwrap();
    let raw: Vec<u8> = image
        .pixels
        .chunks_exact(4)
        .flat_map(|px| px[..3].to_vec())
        .collect();
    let moved = flattened
        .pixels
        .chunks_exact(3)
        .zip(raw.chunks_exact(3))
        .filter(|(a, b)| a != b)
        .count();

    let report = format!(
        "{label}: alpha min={min} max={max} \
         transparent(<128)={transparent}/{total} ({:.1}%) \
         opaque(>200)={opaque}/{total} ({:.1}%) \
         composite-moved={moved}/{total} ({:.1}%)\nhistogram(16 buckets)={histogram:?}\n",
        100.0 * transparent as f32 / total as f32,
        100.0 * opaque as f32 / total as f32,
        100.0 * moved as f32 / total as f32,
    );
    eprint!("{report}");
    std::fs::write(out_dir.join(format!("{label}_alpha.txt")), &report).unwrap();

    assert!(min != max, "{label}: the alpha channel is constant ({min})");
    let floor = total / 50; // 2 %
    assert!(
        transparent >= floor,
        "{label}: only {transparent}/{total} pixels are substantially transparent; the render \
         carries no usable matte"
    );
    assert!(
        opaque >= floor,
        "{label}: only {opaque}/{total} pixels are substantially opaque; the render is a uniform \
         wash rather than a subject on transparency"
    );
    assert!(
        moved >= floor,
        "{label}: compositing over white changed only {moved}/{total} pixels — the alpha is not \
         load-bearing"
    );
}

/// The bounded real-weight **transparency** smoke (sc-24111): one transparent text-to-image and
/// one RGBA-reference extraction, both at 1024×1024 / 8 steps, each asserting a non-trivial alpha
/// histogram on the emitted `RgbaImage`.
///
/// Transparency is requested by **prompt** (upstream ships no flag — see `UPSTREAM.md`);
/// `output_channels: Rgba` only decides that the decoder's alpha reaches the caller instead of
/// being composited over white. Both PNGs are written as RGBA8 so the alpha survives to disk.
///
/// Run detached under the RSS guard, exactly like the sibling smokes
/// (`scratchpad/render_guard_rgba.sh`):
///
/// ```sh
/// MLX_GEN_QWEN_IMAGE_2_1_SNAPSHOT=…/snapshots/790c9263… \
/// QWEN_IMAGE_2_1_RENDER_OUT=~/SceneWorks/render-validation-sc-24111 \
///   cargo test --locked --release -p mlx-gen-qwen-image-2-1 --test integration \
///   e2e_real_weights::validation_render_transparency -- --ignored --nocapture --test-threads 1
/// ```
#[test]
#[ignore]
fn validation_render_transparency() {
    transparency_renders(mlx_gen_qwen_image_2_1::MODEL_ID, snapshot(), "", Some(8));
}

/// The Turbo twin of [`validation_render_transparency`] (sc-25902), through the
/// `qwen_image_2_1_turbo` id on the unmodified Turbo snapshot: the transparent text-to-image and
/// the RGBA-reference extraction at 1024×1024 with **no step count on the request** (the saved
/// eight sigmas), each asserting the same non-trivial alpha histogram.
///
/// ```sh
/// MLX_GEN_QWEN_IMAGE_2_1_TURBO_SNAPSHOT=…/models--Qwen--Qwen-Image-2.1-Turbo/snapshots/fc13ba19… \
/// QWEN_IMAGE_2_1_RENDER_OUT=~/SceneWorks/render-validation-sc-25902 \
///   cargo test --locked --release -p mlx-gen-qwen-image-2-1 --test integration \
///   e2e_real_weights::validation_render_turbo_transparency -- --ignored --nocapture --test-threads 1
/// ```
#[test]
#[ignore]
fn validation_render_turbo_transparency() {
    let root = turbo_snapshot();
    let saved = mlx_gen_qwen_image_2_1::load_saved_sigma_schedule(&root)
        .unwrap()
        .default_steps();
    let reported =
        transparency_renders(mlx_gen_qwen_image_2_1::TURBO_MODEL_ID, root, "turbo_", None);
    assert_eq!(
        reported, [saved; 2],
        "no step count runs one step per saved sigma"
    );
}

/// The transparency smoke of engine `id` over the snapshot at `root`; `steps: None` leaves the
/// step count to the generator. `report_prefix` keeps one id's `<label>_alpha.txt` reports from
/// overwriting the other's in a shared output directory. Returns the step total each of the two
/// renders reported.
fn transparency_renders(
    id: &str,
    root: PathBuf,
    report_prefix: &str,
    steps: Option<u32>,
) -> Vec<u32> {
    use mlx_gen::gen_core::{Conditioning, OutputChannels};

    let out_dir = out_dir();
    let started = Instant::now();
    let registry = mlx_gen_qwen_image_2_1::provider_registry().unwrap();
    let generator = registry
        .load(id, &LoadSpec::new(WeightsSource::Dir(root)))
        .unwrap();
    eprintln!("loaded in {:.1}s", started.elapsed().as_secs_f32());

    // The model card's transparent-image prompt form, and an extraction prompt over a reference
    // that already carries alpha (transparent-layer editing).
    let cases: [(&str, String, Vec<Conditioning>); 2] = [
        (
            "t2i_transparent",
            "This is an RGBA image with transparency. A cute cartoon fox sticker, bold clean \
             outline. The image has an alpha channel and the background is transparent."
                .to_owned(),
            Vec::new(),
        ),
        (
            "rgba_reference_extraction",
            "Extract the subject onto a transparent background. This is an RGBA image with \
             transparency; the background is fully transparent."
                .to_owned(),
            vec![Conditioning::ReferenceRgba {
                image: transparent_reference(generator.as_ref(), &out_dir, id),
                strength: None,
            }],
        ),
    ];

    let mut reported = Vec::new();
    for (label, prompt, conditioning) in cases {
        let req = GenerationRequest {
            prompt,
            width: 1024,
            height: 1024,
            steps,
            seed: Some(42),
            conditioning,
            output_channels: OutputChannels::Rgba,
            ..Default::default()
        };
        generator
            .validate(&req)
            .unwrap_or_else(|e| panic!("{label}: the RGBA request was refused at validate: {e}"));

        let render_started = Instant::now();
        let mut log = progress_logger(format!("{label}: "));
        let mut reported_steps = Vec::new();
        let out = generator
            .generate(&req, &mut |p| {
                log(p);
                if let Progress::Step { current, total } = p {
                    reported_steps.push((current, total));
                }
            })
            .unwrap();
        let GenerationOutput::ImagesRgba(images) = out else {
            panic!("{label}: an `output_channels: Rgba` request must emit ImagesRgba");
        };
        let image = &images[0];
        image.validate().unwrap();
        assert_eq!((image.width, image.height), (1024, 1024));
        assert_eq!(image.channels(), 4);
        let ran = reported_steps
            .last()
            .unwrap_or_else(|| panic!("{label}: the sampler reports every step"))
            .1;
        assert_eq!(
            reported_steps,
            every_step(ran),
            "{label}: one event per step, in order"
        );
        if let Some(steps) = steps {
            assert_eq!(ran, steps, "{label}: the requested step count ran");
        }

        let path = out_dir.join(format!("{id}_{label}_1024x1024_{ran}steps_seed42.png"));
        image::save_buffer(
            &path,
            &image.pixels,
            image.width,
            image.height,
            image::ColorType::Rgba8,
        )
        .unwrap();
        assert_nontrivial_alpha(&format!("{report_prefix}{label}"), image, &out_dir);
        eprintln!(
            "wrote {} after {:.1}s render",
            path.display(),
            render_started.elapsed().as_secs_f32()
        );
        reported.push(ran);
    }
    reported
}

fn turbo_tier_snapshot() -> PathBuf {
    let p = std::env::var("MLX_GEN_QWEN_IMAGE_2_1_TURBO_TIER_SNAPSHOT").unwrap_or_else(|_| {
        panic!(
            "set MLX_GEN_QWEN_IMAGE_2_1_TURBO_TIER_SNAPSHOT to a directory holding the converted \
             Qwen-Image-2.1-Turbo q8/ and q4/ tiers (`qwen_image_2_1_prequant`; default output \
             ~/SceneWorks/qwen-image-2-1-turbo-tiers/); inference never self-fetches (epic 13657)"
        )
    });
    let path = PathBuf::from(p);
    assert!(path.is_dir(), "{} is not a directory", path.display());
    path
}

/// The **installed Turbo tiers** on real weights (AT1, sc-25909) — the Turbo twin of the base
/// tiers' real-weight renders (the candle `validation_render_installed_tiers`, the
/// `lora_real_weights` `*_every_tier` tests over `MLX_GEN_QWEN_IMAGE_2_1_TIER_SNAPSHOT`).
/// `MLX_GEN_QWEN_IMAGE_2_1_TURBO_TIER_SNAPSHOT` names one directory holding a complete `q8/` and
/// `q4/` Turbo tier, the same layout as the base tier snapshot, which is what
/// `qwen_image_2_1_prequant` writes:
///
/// ```sh
/// for t in q8 q4; do
///   QWEN21_SRC=…/models--Qwen--Qwen-Image-2.1-Turbo/snapshots/fc13ba19… QWEN21_TIER=$t \
///     cargo run --locked --release --example qwen_image_2_1_prequant -p mlx-gen-qwen-image-2-1
/// done   # → ~/SceneWorks/qwen-image-2-1-turbo-tiers/{q8,q4}/
/// MLX_GEN_QWEN_IMAGE_2_1_TURBO_TIER_SNAPSHOT=~/SceneWorks/qwen-image-2-1-turbo-tiers \
/// QWEN_IMAGE_2_1_RENDER_OUT=~/SceneWorks/render-validation-sc-25909 \
///   cargo test --locked --release -p mlx-gen-qwen-image-2-1 --test integration \
///   e2e_real_weights::validation_render_turbo_installed_tiers -- --ignored --exact --nocapture
/// ```
///
/// Per tier: the directory must self-report as the tier its name claims (read from the packed
/// shapes), load packed under `qwen_image_2_1_turbo` with the matching quantize selector, and
/// render (1) a cached 1024² text-to-image over the tier's own saved sigmas and (2) a cached
/// two-reference 1024² edit, followed by its forced-shed twin. Every `[kv]` line is priced at
/// **the tier** — the packed tiers compute in f32, so the priced cache and denoise are twice the
/// bf16 tier's — and the twin's observed peak delta is held to that tier's priced cache.
/// Evidence: `qwen_image_2_1_turbo_<q8|q4>_…png` and
/// `qwen_image_2_1_turbo_<q8|q4>_prefix_kv_cache_memory.csv`.
#[test]
#[ignore]
fn validation_render_turbo_installed_tiers() {
    let tiers = turbo_tier_snapshot();
    let saved = mlx_gen_qwen_image_2_1::load_saved_sigma_schedule(&tiers.join("q8"))
        .unwrap()
        .default_steps();
    assert_eq!(saved, 8, "the pinned Turbo revision saves eight sigmas");
    turbo_installed_tier_renders_in(
        &tiers,
        &out_dir(),
        (1024, 1024),
        saved,
        ReferenceSource::Rendered {
            edge: REFERENCE_EDGE,
        },
    );
}

/// [`validation_render_turbo_installed_tiers`]'s body over an explicit tier directory, size,
/// output directory and reference source, so the dry run below drives it on tiers converted from
/// the miniature fixture. Its twin is a two-reference one, so its peak difference is printed and
/// held to the upper bound only (see [`reference_edit_renders_with`]).
fn turbo_installed_tier_renders_in(
    tiers: &Path,
    out_dir: &Path,
    (width, height): (u32, u32),
    saved_steps: u32,
    references: ReferenceSource,
) {
    use mlx_gen_qwen_image_2_1::quant::installed_tier;
    let id = mlx_gen_qwen_image_2_1::TURBO_MODEL_ID;
    for tier in [Tier::Q8, Tier::Q4] {
        let dir = tiers.join(tier.dir_name());
        assert_eq!(
            installed_tier(&dir).unwrap(),
            tier,
            "{}: {} does not hold the tier its name claims",
            tier.dir_name(),
            dir.display()
        );
        let target = Target::installed(id, tiers, tier);
        let reported = validation_render_at(
            &target,
            out_dir,
            (width, height),
            &[None],
            PrefixKvCacheMode::Cached,
            None,
        );
        assert_eq!(
            reported,
            [saved_steps],
            "{}: one step per saved sigma",
            target.stem
        );
        reference_edit_renders_in(
            &target,
            out_dir,
            references,
            saved_steps,
            vec![EditCase {
                references: 2,
                width,
                height,
                steps: None,
                cache: PrefixKvCacheMode::Cached,
            }],
            true,
            None,
        );
    }
}

/// **The smoke bodies, dry-run on the miniature fixtures — not ignored** (sc-25902, sc-25909).
/// The `#[ignore]`d tests above compile on every lane and run on none, so a wrong step assertion,
/// a mis-built request or a mis-wired comparison in them would first surface in the middle of a
/// real-weight run. This drives the same bodies end to end at 64×64 into a temp dir — proving
/// their wiring and assertions, nothing about the released weights:
///
/// * the step sweep ([`validation_render_at`]) through the `qwen_image_2_1_turbo` id, cached and
///   shed;
/// * the reference edit ([`reference_edit_renders_in`]) with its two- and ten-reference cases,
///   **both cached and each followed by its forced-shed twin** (AT3), the observed-peak
///   comparison of every twin ([`observed_cache_delta_failures`]) and its memory CSV, and the
///   eleven-reference refusal; then the base test's cached-beside-prefill shape;
/// * the installed-tier body ([`turbo_installed_tier_renders_in`], AT1) over `q8/` and `q4/`
///   tiers converted here from the Turbo fixture by the production converter — the directory
///   layout `MLX_GEN_QWEN_IMAGE_2_1_TURBO_TIER_SNAPSHOT` names;
/// * the pre-epic golden comparison ([`compare_with_golden`], AT4) on the **base** id: the base
///   T2I and edit renders written once, then rendered again against that directory (byte-equal,
///   passes), against a golden with one byte changed (fails), with a cached render (fails) and
///   with a file missing (fails).
///
/// (The transparency body is not dry-run: its alpha-histogram claim is about the released
/// checkpoint and the miniature decode does not meet it.)
///
/// *Mutations that red this:* the Turbo generator ignoring `req.steps` (the sweep then reports
/// eight steps for every count, and renders identical pixels); a step event dropped or repeated;
/// a body dropping its cache selection from the request (`render_reporting` refuses a request
/// that does not state its mode, and a cached case then reports `prefill`); the twin rendering
/// cached again (it asserts the `prefill` path); only the first cached case twinned (the
/// ten-reference twin's PNG is missing); [`compare_with_golden`] returning early (the corrupted
/// golden passes); the installed-tier body loading without the quantize selector (refused).
/// The edit body asserts **once**, after its last case: with its real-weight checks armed
/// (`real_weights`) on the miniature fixture and the mosaic bar lowered to zero on this thread
/// ([`MOSAIC_BAR`]) so that every picture check — each in-run reference's and each edit output's —
/// fails, every case and twin still renders and writes its files, and the one failure names every
/// case's check. It needs no snapshot and runs wherever the dry run does.
///
/// *Mutations that red this:* any collected check of [`reference_edit_renders_with`] (a
/// reference's, an edit output's [`mosaic_failure`], a twin's [`observed_cache_delta_failures`]
/// or [`twin_failures`]) asserting in place — the ten-reference files are then never written.
#[test]
fn the_edit_body_collects_every_check_and_asserts_after_its_last_case() {
    use std::panic::{catch_unwind, AssertUnwindSafe};

    let tmp = tempfile::tempdir().unwrap();
    let root = crate::common::turbo_snapshot(tmp.path());
    let out = tmp.path().join("evidence");
    std::fs::create_dir_all(&out).unwrap();
    let id = mlx_gen_qwen_image_2_1::TURBO_MODEL_ID;
    let turbo = Target::dense(id, root);
    let case = |references| EditCase {
        references,
        width: 64,
        height: 64,
        steps: None,
        cache: PrefixKvCacheMode::Cached,
    };
    MOSAIC_BAR.with(|bar| bar.set(0.0));
    let err = catch_unwind(AssertUnwindSafe(|| {
        reference_edit_renders_with(
            &turbo,
            &out,
            ReferenceSource::Rendered { edge: 64 },
            8,
            vec![case(2), case(10)],
            true,
            None,
            true,
        );
    }))
    .expect_err("every picture check fails at a zero bar");
    MOSAIC_BAR.with(|bar| bar.set(MOSAIC_CHROMA_BAR));
    let message = err.downcast_ref::<String>().cloned().unwrap_or_default();
    eprintln!("{message}");
    for count in [2, 10] {
        for suffix in ["", "_prefill_twin"] {
            let name = format!("{id}_edit_{count}refs_64x64_8steps_seed42{suffix}.png");
            assert!(
                out.join(&name).is_file(),
                "{name}: every case still rendered"
            );
        }
        assert!(
            message.contains(&format!("{id} {count} refs cached: chroma busyness")),
            "the one failure carries the {count}-reference case's check:\n{message}"
        );
    }
}

#[test]
fn the_turbo_smoke_bodies_run_end_to_end_on_the_miniature_fixture() {
    use std::panic::{catch_unwind, AssertUnwindSafe};

    let tmp = tempfile::tempdir().unwrap();
    let root = crate::common::turbo_snapshot(tmp.path());
    let out = tmp.path().join("evidence");
    std::fs::create_dir_all(&out).unwrap();
    let id = mlx_gen_qwen_image_2_1::TURBO_MODEL_ID;
    let turbo = Target::dense(id, root.clone());

    let asked = [4, 6, 12, 16];
    let cached = PrefixKvCacheMode::Cached;
    let reported = validation_render_at(&turbo, &out, (64, 64), &asked.map(Some), cached, None);
    assert_eq!(reported, asked);
    for steps in asked {
        assert!(out
            .join(format!("{id}_64x64_{steps}steps_seed42.png"))
            .is_file());
    }
    // No step count: the fixture's eight saved sigmas.
    assert_eq!(
        validation_render_at(&turbo, &out, (64, 64), &[None], cached, None),
        [8]
    );
    // …and the same body with the cache shed, as the base renders run it.
    assert_eq!(
        validation_render_at(
            &turbo,
            &out,
            (64, 64),
            &[Some(5)],
            PrefixKvCacheMode::Prefill,
            None
        ),
        [5]
    );

    // The Turbo reference edit as the real-weight test runs it: two and ten references, both
    // cached, each followed by its forced-shed twin.
    let case = |references| EditCase {
        references,
        width: 64,
        height: 64,
        steps: None,
        cache: cached,
    };
    // The miniature cache sits under the fixture's reference encode, so the twins' sign check is
    // not armed here; the bound is, and the check itself is pinned by
    // `the_observed_cache_delta_is_bounded_by_the_priced_cache`.
    // Rendered references, so the in-run reference renders are driven too (64² on the fixture).
    reference_edit_renders_with(
        &turbo,
        &out,
        ReferenceSource::Rendered { edge: 64 },
        8,
        vec![case(2), case(10)],
        true,
        None,
        false,
    );
    for count in [2, 10] {
        for suffix in ["", "_prefill_twin"] {
            let name = format!("{id}_edit_{count}refs_64x64_8steps_seed42{suffix}.png");
            assert!(out.join(&name).is_file(), "{name}");
        }
    }
    let csv = std::fs::read_to_string(out.join(format!("{id}_prefix_kv_cache_memory.csv")))
        .expect("the twins' memory evidence is written");
    assert_eq!(
        csv.lines().count(),
        1 + 2 * 2,
        "a header and two rows per twin:\n{csv}"
    );
    // …and the base test's shape: a cached case beside a prefill boundary, no twin.
    reference_edit_renders_in(
        &turbo,
        &out,
        ReferenceSource::Synthetic,
        8,
        vec![
            case(2),
            EditCase {
                cache: PrefixKvCacheMode::Prefill,
                ..case(10)
            },
        ],
        false,
        None,
    );

    // AT1: the installed-tier body over tiers the production converter writes from the fixture.
    let tiers = tmp.path().join("turbo-tiers");
    for tier in [Tier::Q8, Tier::Q4] {
        mlx_gen_qwen_image_2_1::convert::prequantize_turnkey(
            &root,
            &tiers.join(tier.dir_name()),
            tier,
        )
        .unwrap();
    }
    turbo_installed_tier_renders_in(&tiers, &out, (64, 64), 8, ReferenceSource::Synthetic);
    for tier in ["q8", "q4"] {
        for name in [
            format!("{id}_{tier}_64x64_8steps_seed42.png"),
            format!("{id}_{tier}_edit_2refs_64x64_8steps_seed42.png"),
            format!("{id}_{tier}_edit_2refs_64x64_8steps_seed42_prefill_twin.png"),
            format!("{id}_{tier}_prefix_kv_cache_memory.csv"),
        ] {
            assert!(out.join(&name).is_file(), "{name}");
        }
    }

    // AT4: the golden comparison, on the base id's prefill renders.
    let base = Target::dense(
        mlx_gen_qwen_image_2_1::MODEL_ID,
        crate::common::tiny_snapshot(),
    );
    let golden = tmp.path().join("golden");
    std::fs::create_dir_all(&golden).unwrap();
    let prefill = PrefixKvCacheMode::Prefill;
    let edit_cases = || {
        vec![
            EditCase {
                references: 2,
                width: 64,
                height: 64,
                steps: Some(2),
                cache: prefill,
            },
            EditCase {
                references: 10,
                width: 64,
                height: 64,
                steps: Some(2),
                cache: prefill,
            },
        ]
    };
    validation_render_at(&base, &golden, (64, 64), &[Some(3)], prefill, None);
    reference_edit_renders_in(
        &base,
        &golden,
        ReferenceSource::Synthetic,
        40,
        edit_cases(),
        false,
        None,
    );
    let checked = tmp.path().join("checked");
    std::fs::create_dir_all(&checked).unwrap();
    validation_render_at(
        &base,
        &checked,
        (64, 64),
        &[Some(3)],
        prefill,
        Some(&golden),
    );
    reference_edit_renders_in(
        &base,
        &checked,
        ReferenceSource::Synthetic,
        40,
        edit_cases(),
        false,
        Some(&golden),
    );
    // One byte off in the golden is a failure, a cached render against a golden is a usage
    // error, and a golden directory without the file is a failure — none of them a skip.
    let t2i = golden.join("qwen_image_2_1_64x64_3steps_seed42.png");
    let mut changed = image::open(&t2i).unwrap().to_rgb8();
    changed.as_mut()[0] ^= 1;
    let corrupted = tmp.path().join("corrupted");
    std::fs::create_dir_all(&corrupted).unwrap();
    changed
        .save(corrupted.join("qwen_image_2_1_64x64_3steps_seed42.png"))
        .unwrap();
    let render_against = |dir: &Path, mode| {
        catch_unwind(AssertUnwindSafe(|| {
            validation_render_at(&base, &checked, (64, 64), &[Some(3)], mode, Some(dir));
        }))
    };
    assert!(
        render_against(&corrupted, prefill).is_err(),
        "a changed byte"
    );
    assert!(render_against(&golden, cached).is_err(), "a cached render");
    assert!(
        render_against(&tmp.path().join("evidence"), prefill).is_err(),
        "a golden directory without the file"
    );
    // A mismatch on the FIRST edit case still lets the second render, write its PNG and be
    // compared: the body asserts once, after its last case.
    let first_off = tmp.path().join("first-case-off");
    mlx_gen::quant::copy_dir(&golden, &first_off).unwrap();
    let two_refs = "qwen_image_2_1_edit_2refs_64x64_2steps_seed42.png";
    let mut changed = image::open(first_off.join(two_refs)).unwrap().to_rgb8();
    changed.as_mut()[0] ^= 1;
    changed.save(first_off.join(two_refs)).unwrap();
    let rechecked = tmp.path().join("rechecked");
    std::fs::create_dir_all(&rechecked).unwrap();
    assert!(
        catch_unwind(AssertUnwindSafe(|| {
            reference_edit_renders_in(
                &base,
                &rechecked,
                ReferenceSource::Synthetic,
                40,
                edit_cases(),
                false,
                Some(&first_off),
            );
        }))
        .is_err(),
        "a first-case mismatch fails the body"
    );
    let report = std::fs::read_to_string(rechecked.join("golden_report.txt")).unwrap();
    let ten_refs = "qwen_image_2_1_edit_10refs_64x64_2steps_seed42.png";
    assert!(
        report
            .lines()
            .any(|line| line.contains(two_refs) && line.ends_with("FAIL")),
        "{report}"
    );
    assert!(
        report
            .lines()
            .any(|line| line.contains(ten_refs) && line.ends_with("PASS")),
        "the second case was still compared:\n{report}"
    );
    assert!(rechecked.join(ten_refs).is_file(), "and its PNG written");
}

/// [`assert_not_degenerate`] rejects a blank frame and accepts a dark, low-contrast picture.
#[test]
fn the_degenerate_render_check_rejects_a_flat_frame_only() {
    let flat = vec![7u8; 64 * 64 * 3];
    assert!(std::panic::catch_unwind(|| assert_not_degenerate("flat", &flat)).is_err());
    // Two byte values only: spread without content.
    let two: Vec<u8> = (0..64 * 64 * 3)
        .map(|i| if (i / 3) % 2 == 0 { 0 } else { 255 })
        .collect();
    assert!(std::panic::catch_unwind(|| assert_not_degenerate("two-valued", &two)).is_err());
    // A dim ramp: 24 luma levels of spread, 24 distinct values.
    let dim: Vec<u8> = (0..64 * 64)
        .flat_map(|i| [(i % 24) as u8 + 10; 3])
        .collect();
    assert_not_degenerate("dim ramp", &dim);
}
