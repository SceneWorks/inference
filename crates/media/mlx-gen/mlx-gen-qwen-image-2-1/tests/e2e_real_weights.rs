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
/// observation [`assert_observed_cache_delta`] compares them with across a forced-shed twin.
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
/// [`cache_delta_bound`]'s headroom. Prints the observed delta beside the priced bytes either way.
/// `require_positive` is false only where a dry run's miniature cache can sit under another
/// phase's peak.
fn assert_observed_cache_delta(
    label: &str,
    cached_peak: u64,
    shed_peak: u64,
    priced: u64,
    require_positive: bool,
) {
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
    assert!(
        delta <= bound as i128,
        "{label}: holding the cache raised the peak by {delta} bytes, above the {bound} its priced \
         {priced} bytes allow"
    );
    if require_positive {
        assert!(
            delta > 0,
            "{label}: the cached render peaked {delta} bytes against its shed twin — the cache \
             added nothing to the high-water mark"
        );
    }
}

/// The bound and the sign check bite where they should, on synthetic peaks (no device).
///
/// *Mutations that red this:* dropping either assertion of [`assert_observed_cache_delta`];
/// a bound without the priced term or without the headroom.
#[test]
fn the_observed_cache_delta_is_bounded_by_the_priced_cache() {
    const G: u64 = 1 << 30;
    let priced = 24 * G;
    // At the bound, and a little below it, pass; one byte over fails.
    assert_observed_cache_delta(
        "at bound",
        30 * G + cache_delta_bound(priced),
        30 * G,
        priced,
        true,
    );
    assert_observed_cache_delta("typical", 50 * G, 30 * G, priced, true);
    assert!(std::panic::catch_unwind(|| {
        assert_observed_cache_delta(
            "over",
            30 * G + cache_delta_bound(priced) + 1,
            30 * G,
            priced,
            true,
        )
    })
    .is_err());
    // No rise is refused only when a rise is required.
    assert!(std::panic::catch_unwind(|| {
        assert_observed_cache_delta("flat", 30 * G, 30 * G, priced, true)
    })
    .is_err());
    assert_observed_cache_delta("flat dry run", 30 * G, 30 * G, priced, false);
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

/// Hold the render just written as `file_name` to its pre-epic golden, **byte for byte** on this
/// backend (the MLX base forward was refactored by epic 25899 to be bit-identical; nothing else
/// moved). Only a cache-shed render is comparable — the pre-epic revision had no cache — so a
/// golden with a cached render is a usage error, not a skip. A golden directory missing the file
/// is a failure too.
fn compare_with_golden(
    golden: Option<&Path>,
    file_name: &str,
    image: &mlx_gen::gen_core::Image,
    mode: PrefixKvCacheMode,
) {
    let Some(golden) = golden else {
        return;
    };
    assert_eq!(
        mode,
        PrefixKvCacheMode::Prefill,
        "{file_name}: {RENDER_GOLDEN_ENV} compares the cache-shed path with the pre-epic render; \
         run with QWEN_IMAGE_2_1_PREFIX_KV_CACHE=prefill"
    );
    let path = golden.join(file_name);
    let want = image::open(&path)
        .unwrap_or_else(|e| {
            panic!(
                "{RENDER_GOLDEN_ENV}: {} is the golden for this render: {e}",
                path.display()
            )
        })
        .to_rgb8();
    assert_eq!(
        (want.width(), want.height()),
        (image.width, image.height),
        "{file_name}: golden geometry"
    );
    let (mean, max) = pixel_difference(&image.pixels, want.as_raw());
    eprintln!(
        "[golden] {file_name}: mean |Δ| = {mean:.4}, max |Δ| = {max} (of 255) against {}; bar: \
         byte-equal",
        path.display()
    );
    assert!(
        image.pixels == *want.as_raw(),
        "{file_name}: the cache-shed render is not the pre-epic render byte for byte (mean |Δ| \
         {mean:.4}, max {max})"
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
        compare_with_golden(golden, &file_name, image, cache);
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
    reported
}

/// A deterministic synthetic reference, so the edit smoke needs no checked-in photographs.
/// `QWEN_IMAGE_2_1_EDIT_REFS` points at a directory of real PNGs to use instead, in name order —
/// reference images for a real evaluation are user-provided, never synthesised or fetched here.
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

fn references(count: usize, width: u32, height: u32) -> Vec<mlx_gen::gen_core::Image> {
    let Ok(dir) = std::env::var("QWEN_IMAGE_2_1_EDIT_REFS") else {
        return (0..count)
            .map(|i| synthetic_reference(i as u32 + 1, width, height))
            .collect();
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
    reference_edit_renders(
        &Target::dense(mlx_gen_qwen_image_2_1::MODEL_ID, snapshot()),
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
        render_golden().as_deref(),
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
/// produces the prefill output". The two paths are the same mathematics; the bar is a mean
/// absolute pixel difference of at most 2 of 255 (the bar this crate's tiling parity uses), and
/// the mean and maximum are printed. **That bar has never been run on released weights**: if the
/// first real run exceeds it, the printed numbers are the finding.
///
/// Each twin pair is also AT3's memory comparison, printed **unconditionally**: MLX's peak active
/// memory over the cached render and over its shed twin, their difference
/// (`observed_cache_delta`) beside the priced cache bytes, held to `0 < delta ≤ priced · 1.10 +
/// 1 GiB` ([`assert_observed_cache_delta`]), and written to
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

/// The mean absolute pixel difference (of 255) a forced-shed twin may show against its cached
/// render — this crate's tiling-parity bar. Not measured on released weights.
const SHED_TWIN_MEAN_BAR: f64 = 2.0;

/// [`reference_edit_renders`] over explicit cases into an explicit directory — the body, free of
/// the `QWEN_IMAGE_2_1_*` environment so the default-run dry run at the bottom of this file can
/// drive it on the miniature fixture.
///
/// `require_positive_delta` arms the sign half of [`assert_observed_cache_delta`] for the
/// **ten-reference** twin — the one where the cache dominates every other phase (24.5 GiB priced
/// at bf16 against a ~10.6 GiB denoise and a ≤ 9.4 GiB reference encode), so holding it must raise
/// the peak. With fewer references the reference encode can set both renders' peak and the
/// difference is legitimately near zero, so those twins are printed and held to the upper bound
/// only. Every real-weight caller sets it; the dry run cannot (the miniature cache is a few
/// hundred kB under a VAE encode that is not).
fn reference_edit_renders_in(
    target: &Target,
    out_dir: &std::path::Path,
    default_steps: u32,
    cases: Vec<EditCase>,
    forced_shed_twin: bool,
    golden: Option<&Path>,
) {
    reference_edit_renders_with(
        target,
        out_dir,
        default_steps,
        cases,
        forced_shed_twin,
        golden,
        true,
    )
}

/// [`reference_edit_renders_in`] with the sign check's arming explicit — see there.
#[allow(clippy::too_many_arguments)]
fn reference_edit_renders_with(
    target: &Target,
    out_dir: &std::path::Path,
    default_steps: u32,
    cases: Vec<EditCase>,
    forced_shed_twin: bool,
    golden: Option<&Path>,
    require_positive_delta: bool,
) {
    use mlx_gen::gen_core::Conditioning;

    let stem = target.stem.as_str();
    let started = Instant::now();
    let generator = target.load();
    eprintln!("loaded in {:.1}s", started.elapsed().as_secs_f32());

    let trace = std::env::var_os("QWEN_IMAGE_2_1_MEMORY_TRACE").is_some();
    let footprint_max = footprint_sampler();
    // One render of `case`, with the cache mode `cache` (the case's own, or the twin's).
    let render = |case: EditCase, cache: PrefixKvCacheMode, suffix: &str| -> (Vec<u8>, Observed) {
        let EditCase {
            references: count,
            width,
            height,
            steps,
            ..
        } = case;
        let req = selecting(
            GenerationRequest {
                prompt:
                    "Combine the subjects of the reference images into one scene, evening light"
                        .to_owned(),
                width,
                height,
                steps,
                seed: Some(42),
                conditioning: vec![Conditioning::MultiReference {
                    images: references(count, 768, 768),
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
        compare_with_golden(golden, &file_name, &image, cache);
        (image.pixels, observed)
    };

    // Each twin is rendered **immediately after** the cached render it is compared with, so no
    // other request runs on the generator between the two sides of the comparison — the pixels
    // and the peaks are both read across that one pair.
    let mut memory_rows = vec![
        "case,references,mode,peak_active_bytes,cache_bytes,priced_cache_bytes,observed_cache_delta_bytes"
            .to_owned(),
    ];
    let mut twins = 0;
    for case in cases {
        let (cached, cached_report) = render(case, case.cache, "");
        if !forced_shed_twin || !case.cache.is_cached() {
            continue;
        }
        twins += 1;
        let (shed, shed_report) = render(case, PrefixKvCacheMode::Prefill, "_prefill_twin");
        let (mean, max) = pixel_difference(&shed, &cached);
        eprintln!(
            "[kv] {stem} forced-shed twin of the {}-reference case: mean |Δ| = {mean:.4}, max |Δ| = \
             {max} (of 255) against its cached render; bar: mean <= {SHED_TWIN_MEAN_BAR}",
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
        assert_observed_cache_delta(
            &format!("{stem} {}-reference twin", case.references),
            cached_report.peak_active,
            shed_report.peak_active,
            cached_report.priced_cache,
            require_positive_delta && case.references == MAX_REFERENCES,
        );
        assert!(
            mean <= SHED_TWIN_MEAN_BAR,
            "the forced-shed render differs from the cached one by a mean of {mean:.4} levels \
             (max {max}); the two paths are the same mathematics"
        );
    }
    assert!(
        !forced_shed_twin || twins > 0,
        "a forced-shed twin needs a cached case"
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

/// A **transparent** RGBA reference of `width`×`height`: the same deterministic picture
/// [`synthetic_reference`] produces, matted onto a soft-edged disc so the alpha is a real ramp
/// rather than a binary cut-out (a hard matte would survive any resampling order and so could not
/// exercise the premultiplied resize).
///
/// Set `QWEN_IMAGE_2_1_RGBA_REF` to a PNG with an alpha channel to extract from a real image
/// instead; reference images for a real evaluation are user-provided.
fn transparent_reference(width: u32, height: u32) -> mlx_gen::RgbaImage {
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
    let rgb = synthetic_reference(1, width, height);
    let mut out = mlx_gen::RgbaImage::from_rgb(&rgb).unwrap();
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
                image: transparent_reference(1024, 1024),
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
    turbo_installed_tier_renders_in(&tiers, &out_dir(), (1024, 1024), saved);
}

/// [`validation_render_turbo_installed_tiers`]'s body over an explicit tier directory, size and
/// output directory, so the dry run below drives it on tiers converted from the miniature
/// fixture. Its twin is a two-reference one, so its peak difference is printed and held to the
/// upper bound only (see [`reference_edit_renders_with`]).
fn turbo_installed_tier_renders_in(
    tiers: &Path,
    out_dir: &Path,
    (width, height): (u32, u32),
    saved_steps: u32,
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
///   comparison of every twin ([`assert_observed_cache_delta`]) and its memory CSV, and the
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
    reference_edit_renders_with(&turbo, &out, 8, vec![case(2), case(10)], true, None, false);
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
    turbo_installed_tier_renders_in(&tiers, &out, (64, 64), 8);
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
    reference_edit_renders_in(&base, &golden, 40, edit_cases(), false, None);
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
    reference_edit_renders_in(&base, &checked, 40, edit_cases(), false, Some(&golden));
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
