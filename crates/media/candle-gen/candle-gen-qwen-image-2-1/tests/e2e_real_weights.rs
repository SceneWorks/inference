//! The bounded real-weight validation render (sc-24109) — `#[ignore]`d **and** gated to a GPU
//! backend: it needs the pinned `Qwen/Qwen-Image-2.1` snapshot at
//! `CANDLE_GEN_QWEN_IMAGE_2_1_SNAPSHOT` (inference never self-fetches or derives a cache location,
//! epic 13657) and an accelerator, because the released weights are bf16 and candle's plain CPU
//! backend has no bf16 matmul — the same reason `candle-gen-ltx`'s conformance suite is gated.
//!
//! The gate is `any(cuda, metal)` rather than `cuda` alone so the macOS candle lane type-checks
//! this file too (`cargo check --features metal --all-targets`); CUDA is the lane the render is
//! meant for. Ordinary CI never reaches the `#[ignore]`d tests: the dispatch-only `qwen-image-2-1`
//! profile of `.github/workflows/real-weights.yml` (sc-24114) provisions both pinned base
//! snapshots on the Windows CUDA runner, runs every `#[ignore]`d test here that is **not**
//! `turbo_`-prefixed and uploads the PNGs, alpha reports and logs as the
//! `qwen-image-2-1-cuda-evidence` artifact; the `turbo_` tests run on the `qwen-image-2-1-turbo`
//! profile (below). The non-ignored tests here (the dry run of the render bodies and the
//! device-free checks) run on every `cuda` / `metal` test lane.
//!
//! Loads the released bf16 weights through the explicit catalog's production load path and renders
//! one image at the upstream default preset (1:1 2048×2048, 40 steps, seed 42, no guidance),
//! writing the raw RGB8 bytes and a PNG of them to `QWEN_IMAGE_2_1_RENDER_OUT` (default: the
//! current directory). It
//! also pins the released tokenizer's system-prefix drop count (14) — the one claim that needs only
//! the snapshot's `processor/`, not the GPU.
//!
//! ```sh
//! CANDLE_GEN_QWEN_IMAGE_2_1_SNAPSHOT=…/models--Qwen--Qwen-Image-2.1/snapshots/790c9263… \
//! QWEN_IMAGE_2_1_RENDER_OUT=…/render-validation-sc-24109 \
//!   cargo test --locked --release -p candle-gen-qwen-image-2-1 --features cuda \
//!   --test integration e2e_real_weights:: -- --ignored --nocapture
//! ```
//!
//! `QWEN_IMAGE_2_1_RENDER_SIZE=WxH` and `QWEN_IMAGE_2_1_RENDER_STEPS=N` override the preset for a
//! quicker smoke.
//!
//! The `turbo_`-prefixed tests (sc-25901, sc-25902) are the Turbo twins: the default-preset
//! render, the reference edit, the transparency smoke, and a step sweep. They read
//! `CANDLE_GEN_QWEN_IMAGE_2_1_TURBO_SNAPSHOT` instead — and nothing of the base snapshot — and are
//! run by their own dispatch-only profile, `qwen-image-2-1-turbo`, which provisions
//! `Qwen/Qwen-Image-2.1-Turbo`; the `qwen-image-2-1` profile above runs every test here that is
//! **not** `turbo_`-prefixed. The prefix is what partitions the two lanes
//! (`scripts/tests/test_ci_workflow_policy.py`), so a Turbo test must carry it.
//!
//! # The prefix KV cache (sc-25905)
//!
//! The cache is a request selection and an unset request runs prefill, so the default-preset and
//! reference-edit renders here **state their mode** and assert, on the generator's own report,
//! that that is the path that ran. `QWEN_IMAGE_2_1_PREFIX_KV_CACHE=cached|prefill` forces one
//! mode on every such render of the run. A `[kv]` line prints the path, the bytes the caches
//! held, and the surcharge the request is priced at (`prefix_kv_cache_demand_bytes`; this crate
//! publishes no cache-shed figure). Defaults:
//!
//! | render | default | why |
//! |---|---|---|
//! | base default preset, base ten-reference boundary | prefill | the pre-epic path the published floors were derived for |
//! | base two-reference edit | cached | the caller-shaped request |
//! | every Turbo default-preset / sweep / reference-edit render, incl. ten references | cached | what the epic ships |
//! | Turbo two- and ten-reference **forced-shed twins** (each right after its cached render) | prefill | the epic's "one run forced to shed that produces the prefill output" (AT3) |
//!
//! The transparency and base installed-tier renders select nothing and so run prefill; the Turbo
//! installed-tier renders (`turbo_validation_render_installed_tiers`) are cached.
//!
//! # Device memory (AT5, sc-25909)
//!
//! On a `--features cuda` build every mode-stating render also reads the CUDA pool's live bytes
//! before it and their high-water over it (`device_memory`), prints them on its `[kv]` line, and
//! each forced-shed twin pair writes `<stem>_prefix_kv_cache_memory.csv` into
//! `QWEN_IMAGE_2_1_RENDER_OUT` and holds `peak(cached) − peak(shed)` to the priced cache. A Metal
//! build says, loudly, that it measured nothing.

#![cfg(any(feature = "cuda", feature = "metal"))]

use std::path::{Path, PathBuf};
use std::time::Instant;

use candle_core::Tensor;

use candle_gen::gen_core::{
    GenerationMemory, GenerationOutput, GenerationRequest, Generator, LoadSpec, PrefixKvCacheMode,
    Progress, Quant, WeightsSource,
};
use candle_gen_qwen_image_2_1::memory_strategy::prefix_kv_cache_demand_bytes;
use candle_gen_qwen_image_2_1::{load_tokenizer, system_prompt_drop_count, PRESETS};

/// `QWEN_IMAGE_2_1_PREFIX_KV_CACHE=cached|prefill`: force one prefix-KV-cache mode on every
/// mode-stating render of the run. Unset, each render runs its own documented default.
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
    /// The device memory window around the render ([`device_memory`]): `(baseline, peak)` live
    /// pool bytes — `None` where this backend has no sampler. Observed, never committed.
    device: Option<(u64, u64)>,
    /// The bytes the caches held (`PrefixKvCacheReport::cache_bytes`).
    cache_bytes: u64,
    /// The cache surcharge the request is priced at (0 for prefill) — derived.
    priced_cache: u64,
}

/// One render of `req` (which must select `mode`) through `generate_with_report`, asserting that
/// the denoise loop ran the path the request selected, and printing the `[kv]` evidence line:
/// the path, the bytes the caches held, the surcharge a cached request is priced at on this
/// build's device (`prefix_kv_cache_demand_bytes` — **derived**), and the device memory the
/// render was observed to use ([`device_memory`]: the CUDA pool's live bytes before it and their
/// high-water over it).
fn render_reporting(
    generator: &dyn Generator,
    req: &GenerationRequest,
    mode: PrefixKvCacheMode,
    label: &str,
    on_progress: &mut dyn FnMut(Progress),
) -> (GenerationOutput, Observed) {
    assert_eq!(
        req.memory.and_then(|memory| memory.prefix_kv_cache),
        Some(mode),
        "{label}: the request states its mode"
    );
    let window = device_memory::Window::begin();
    let report = generator
        .generate_with_report(req, on_progress)
        .unwrap_or_else(|e| panic!("{label}: the render runs: {e}"));
    let device = window.map(device_memory::Window::end);
    let ran = report
        .prefix_kv_cache
        .expect("the path is reported on every render");
    let negative = req.negative_prompt.is_some() && req.true_cfg.is_some_and(|scale| scale > 1.0);
    let surcharge = match mode {
        PrefixKvCacheMode::Cached => prefix_kv_cache_demand_bytes(
            req.width,
            req.height,
            req.memory_reference_count(),
            negative,
        ),
        PrefixKvCacheMode::Prefill => 0,
    };
    let observed = match device {
        Some((baseline, peak)) => format!(
            "observed device used: {:.2} GiB before, peak {:.2} GiB (+{:.2} GiB)",
            baseline as f64 / GIB,
            peak as f64 / GIB,
            peak.saturating_sub(baseline) as f64 / GIB
        ),
        None => "no device memory sampler on this backend".to_owned(),
    };
    eprintln!(
        "[kv] {label}: selected={} ran={} cache_bytes={} ({:.2} GiB) overflow={:?}; priced cache \
         surcharge = {:.2} GiB above the request's cache-shed demand (derived); {observed}",
        mode.label(),
        ran.path.label(),
        ran.cache_bytes,
        ran.cache_bytes as f64 / GIB,
        ran.overflow,
        surcharge as f64 / GIB,
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
            device,
            cache_bytes: ran.cache_bytes,
            priced_cache: surcharge,
        },
    )
}

const GIB: f64 = (1_u64 << 30) as f64;

/// The device memory one render uses (AT5, sc-25909) — **in the test, independent of any lane
/// script**: on CUDA the driver's stream-ordered pool counters candle allocates from
/// (`candle_gen::cuda_mempool`: `USED_MEM_CURRENT` before the render as the baseline, and the
/// `USED_MEM_HIGH` high-water over it, reset once the whole context has been synchronized so a
/// previous render's pending frees are not inherited). A continuous high-water, so no sub-poll
/// transient is missed (the module doc there says why `nvidia-smi` sampling under-reads). On a
/// Metal or CPU build there is no such counter: [`Window::begin`] says so, loudly, once per
/// render, and returns `None`.
mod device_memory {
    #[cfg(feature = "cuda")]
    pub struct Window {
        pool: candle_gen::cuda_mempool::MemPool,
        baseline: u64,
    }

    #[cfg(feature = "cuda")]
    fn synchronize_device() {
        candle_core::cuda_backend::cudarc::driver::CudaContext::new(0)
            .expect("the CUDA primary context")
            .synchronize()
            .expect("cuCtxSynchronize");
    }

    #[cfg(feature = "cuda")]
    impl Window {
        pub fn begin() -> Option<Self> {
            synchronize_device();
            let pool = candle_gen::cuda_mempool::MemPool::device_default(0)
                .expect("the CUDA default memory pool");
            let baseline = pool.used().expect("USED_MEM_CURRENT");
            assert!(pool.reset_used_high_water(), "reset USED_MEM_HIGH");
            Some(Self { pool, baseline })
        }

        /// `(baseline, peak)` live pool bytes.
        pub fn end(self) -> (u64, u64) {
            synchronize_device();
            let peak = self.pool.used_high().expect("USED_MEM_HIGH");
            (self.baseline, peak.max(self.baseline))
        }
    }

    #[cfg(not(feature = "cuda"))]
    pub struct Window;

    #[cfg(not(feature = "cuda"))]
    impl Window {
        pub fn begin() -> Option<Self> {
            eprintln!(
                "[mem] NO DEVICE MEMORY SAMPLER on this build (not `--features cuda`): the observed \
                 cache delta and the memory CSV are CUDA-only; nothing is measured here"
            );
            None
        }

        pub fn end(self) -> (u64, u64) {
            unreachable!("no window is opened without a sampler")
        }
    }
}

/// The headroom the observed cache delta may show above the priced cache: 10 % of it plus 1 GiB —
/// the MLX twin's bound, for the same reasons: the priced figure is an upper bound on the cache
/// itself, and the headroom covers what the two paths allocate differently besides it — the
/// extract step's per-block contiguous K/V copy before its source is released (one block's
/// worth, ~1/32 = 3 % of the cache), the pool's allocation granularity, and kernel workspace.
fn cache_delta_bound(priced: u64) -> u64 {
    priced + priced / 10 + (1 << 30)
}

/// A cached render and its forced-shed twin together: holding the cache raised the device peak
/// (`require_positive`: the delta is positive) and by no more than [`cache_delta_bound`] of its
/// priced bytes. Prints the delta beside the priced bytes either way.
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
        "{label}: holding the cache raised the device peak by {delta} bytes, above the {bound} \
         its priced {priced} bytes allow"
    );
    if require_positive {
        assert!(
            delta > 0,
            "{label}: the cached render peaked {delta} bytes against its shed twin — the cache \
             added nothing to the device high-water mark"
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
    assert!(std::panic::catch_unwind(|| {
        assert_observed_cache_delta("flat", 30 * G, 30 * G, priced, true)
    })
    .is_err());
    assert_observed_cache_delta("flat dry run", 30 * G, 30 * G, priced, false);
    assert_eq!(cache_delta_bound(10 * G), 11 * G + G);
}

/// `QWEN_IMAGE_2_1_RENDER_GOLDEN=<dir>`: the raw `.rgb` renders the **same named test wrote at
/// the pre-epic revision** (`c1f594203`) into its `QWEN_IMAGE_2_1_RENDER_OUT`. Read only by the
/// base-id tests ([`validation_render_default_preset`], [`validation_render_reference_edit`]); see
/// *The pre-epic golden (AT4)* in this crate's `UPSTREAM.md`.
const RENDER_GOLDEN_ENV: &str = "QWEN_IMAGE_2_1_RENDER_GOLDEN";

fn render_golden() -> Option<PathBuf> {
    std::env::var_os(RENDER_GOLDEN_ENV).map(PathBuf::from)
}

/// The mean absolute difference (of 255) a base cache-shed render may show against its pre-epic
/// golden on a GPU lane.
///
/// Not zero, by design: the one approved base change on this backend is the prefill's exact
/// modulation select (sc-25904, *History — the prefill's modulation select* in `UPSTREAM.md`),
/// which moves bf16 output. Its size: the old blend `zero + (real − zero)·mask` was inexact on 36 %
/// of target-row modulation entries in bf16, median relative error 0.6 % (about one bf16 ULP),
/// p99 9 %, max abs 0.031 — i.e. per step, a perturbation of the order of one bf16 rounding of
/// the scale/gate rows, compounded over the denoise. A same-seed render perturbed by roundings of
/// that order stays the same picture and moves its pixels by a few levels; a regression of the
/// forward (a wrong row, a dropped term) makes a different picture, tens of levels off on
/// average. 4 of 255 separates the two with margin on both sides. **Never run on released
/// weights**: if the first comparison exceeds it, the printed numbers are the finding.
const GOLDEN_MEAN_BAR: f64 = 4.0;
/// …and at most this share of its pixels may move by more than [`GOLDEN_FAR_LEVELS`]: a localized
/// structural change (an edge moved, a glyph redrawn) that a mean over a 2048² frame would dilute.
/// Max |Δ| itself is printed, not bounded — one shifted edge pixel saturates it.
const GOLDEN_FAR_FRACTION: f64 = 0.01;
const GOLDEN_FAR_LEVELS: u8 = 32;

/// Compare the raw RGB8 render just written as `file_name` (`….rgb`) with its pre-epic golden
/// within [`GOLDEN_MEAN_BAR`] / [`GOLDEN_FAR_FRACTION`], print the `[golden]` line (mean, max,
/// far share), append it to `<out_dir>/golden_report.txt`, and **return** the failure (a miss,
/// a size change or a missing golden file) instead of panicking — so one case's mismatch cannot
/// cost the comparison of the cases after it; the body asserts every result at once with
/// [`assert_golden_clean`] after its last render. Only a cache-shed render is comparable — the
/// pre-epic revision had no cache — so a golden with a cached render is a usage error and panics
/// at once.
fn compare_with_golden(
    golden: Option<&Path>,
    out_dir: &Path,
    file_name: &str,
    pixels: &[u8],
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
    let (line, failure) = match std::fs::read(&path) {
        Err(e) => {
            let failure = format!(
                "{file_name}: {RENDER_GOLDEN_ENV}: {} is the golden for this render: {e}",
                path.display()
            );
            (format!("[golden] {failure}"), Some(failure))
        }
        Ok(want) if want.len() != pixels.len() => {
            let failure = format!(
                "{file_name}: the golden holds {} bytes against the render's {}",
                want.len(),
                pixels.len()
            );
            (format!("[golden] {failure}"), Some(failure))
        }
        Ok(want) => {
            let (mean, max) = pixel_difference(pixels, &want);
            let far = pixels
                .iter()
                .zip(&want)
                .filter(|(a, b)| a.abs_diff(**b) > GOLDEN_FAR_LEVELS)
                .count() as f64
                / pixels.len() as f64;
            let pass = mean <= GOLDEN_MEAN_BAR && far <= GOLDEN_FAR_FRACTION;
            (
                format!(
                    "[golden] {file_name}: mean |Δ| = {mean:.4}, max |Δ| = {max} (of 255), {:.3} % \
                     of bytes moved by more than {GOLDEN_FAR_LEVELS} against {}; bars: mean <= \
                     {GOLDEN_MEAN_BAR}, far share <= {:.1} %; {}",
                    100.0 * far,
                    path.display(),
                    100.0 * GOLDEN_FAR_FRACTION,
                    if pass { "PASS" } else { "FAIL" }
                ),
                (!pass).then(|| {
                    format!(
                        "{file_name}: the cache-shed render is not the pre-epic render within the \
                         approved select's bound (mean |Δ| {mean:.4}, far share {:.3} %, max \
                         {max})",
                        100.0 * far
                    )
                }),
            )
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

/// Write `pixels` (RGB8 or RGBA8, per `color`) as a PNG beside the raw bytes, so the evidence
/// artifact carries something a reviewer can open without knowing the geometry (sc-24114).
fn save_png(
    path: &std::path::Path,
    pixels: &[u8],
    width: u32,
    height: u32,
    color: image::ColorType,
) {
    image::save_buffer(path, pixels, width, height, color)
        .unwrap_or_else(|e| panic!("{}: {e}", path.display()));
    eprintln!("wrote {}", path.display());
}

/// Reject a render whose rows are stale GPU memory rather than a picture (sc-24114): the 2048²
/// evidence render passed every shape assertion with 430 correct rows above 1,600 rows of noise
/// — candle's CUDA im2col launch had truncated the full-resolution conv's element count to u32.
/// Per row, the mean absolute second difference of the luma along the columns is ~3 for a
/// rendered scene (the evidence's intact strip peaks at 9.2, the sane 1024² renders at 5.8) and
/// 20–35 for uninitialised memory; a row over 20 is static, and more than a quarter of them fails
/// the render. The fraction is printed so a borderline pass is still readable in the log.
fn assert_not_band_corrupted(label: &str, pixels: &[u8], width: u32, height: u32) {
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
    assert!(
        fraction <= 0.25,
        "{label}: {static_rows} of {h} rows are noise — the render is band-corrupted (a stale \
         or truncated device buffer reached the output)"
    );
}

fn snapshot() -> PathBuf {
    let p = std::env::var("CANDLE_GEN_QWEN_IMAGE_2_1_SNAPSHOT").unwrap_or_else(|_| {
        panic!("set CANDLE_GEN_QWEN_IMAGE_2_1_SNAPSHOT to the pinned snapshot dir; inference never self-fetches (epic 13657)")
    });
    PathBuf::from(p)
}

#[test]
#[ignore]
fn released_tokenizer_drops_fourteen_system_tokens() {
    let tokenizer = load_tokenizer(&snapshot()).unwrap();
    assert_eq!(system_prompt_drop_count(&tokenizer).unwrap(), 14);
}

/// The base checkpoint's default-preset render. With `QWEN_IMAGE_2_1_RENDER_GOLDEN=<dir>` it is
/// also AT4's pre-epic comparison: the `.rgb` it writes is held to the file of the same name the
/// same test wrote at `c1f594203` ([`compare_with_golden`]; *The pre-epic golden (AT4)* in
/// `UPSTREAM.md`).
#[test]
#[ignore]
fn validation_render_default_preset() {
    let (mut width, mut height) = (PRESETS[0].width, PRESETS[0].height);
    if let Ok(size) = std::env::var("QWEN_IMAGE_2_1_RENDER_SIZE") {
        let (w, h) = size.split_once('x').expect("WxH");
        width = w.parse().unwrap();
        height = h.parse().unwrap();
    }
    let steps: u32 = std::env::var("QWEN_IMAGE_2_1_RENDER_STEPS")
        .ok()
        .map(|s| s.parse().unwrap())
        .unwrap_or(candle_gen_qwen_image_2_1::DEFAULT_STEPS);
    // Prefill by default: this is the pre-epic render the published floors were derived for.
    let cache = prefix_kv_cache_override().unwrap_or(PrefixKvCacheMode::Prefill);
    validation_render_at(
        snapshot(),
        &out_dir(),
        (width, height),
        steps,
        cache,
        render_golden().as_deref(),
    );
}

/// [`validation_render_default_preset`]'s body at an explicit snapshot, output directory, size,
/// step count and cache mode — free of the environment, so the dry run below drives it.
fn validation_render_at(
    root: PathBuf,
    out_dir: &Path,
    (width, height): (u32, u32),
    steps: u32,
    cache: PrefixKvCacheMode,
    golden: Option<&Path>,
) {
    let started = Instant::now();
    let registry = candle_gen_qwen_image_2_1::provider_registry().unwrap();
    let generator = registry
        .load("qwen_image_2_1", &LoadSpec::new(WeightsSource::Dir(root)))
        .unwrap();
    eprintln!("loaded in {:.1}s", started.elapsed().as_secs_f32());

    let req = GenerationRequest {
        prompt: "A neon shop sign that reads \"QWEN IMAGE 2.1\", rainy night, reflections on wet pavement"
            .to_owned(),
        width,
        height,
        steps: Some(steps),
        seed: Some(42),
        ..Default::default()
    };
    let req = selecting(req, cache);
    let render_started = Instant::now();
    let (out, _) = render_reporting(
        generator.as_ref(),
        &req,
        cache,
        "default_preset",
        &mut |p| {
            if let Progress::Step { current, total } = p {
                eprintln!(
                    "step {current}/{total} ({:.1}s)",
                    render_started.elapsed().as_secs_f32()
                );
            } else {
                eprintln!("{p:?} ({:.1}s)", render_started.elapsed().as_secs_f32());
            }
        },
    );
    let GenerationOutput::Images(images) = out else {
        panic!("images expected");
    };
    let image = &images[0];
    assert_eq!((image.width, image.height), (width, height));
    assert_eq!(image.pixels.len(), (width as usize) * (height as usize) * 3);
    let file_name = format!("qwen_image_2_1_candle_{width}x{height}_{steps}steps_seed42.rgb");
    let path = out_dir.join(&file_name);
    std::fs::write(&path, &image.pixels).unwrap();
    save_png(
        &path.with_extension("png"),
        &image.pixels,
        width,
        height,
        image::ColorType::Rgb8,
    );
    let golden_failure = compare_with_golden(golden, out_dir, &file_name, &image.pixels, cache);
    assert_not_band_corrupted("default_preset", &image.pixels, width, height);
    assert_golden_clean(golden_failure.as_slice());
    eprintln!(
        "wrote {} ({width}x{height} RGB8) after {:.1}s total ({:.1}s render)",
        path.display(),
        started.elapsed().as_secs_f32(),
        render_started.elapsed().as_secs_f32()
    );
}

/// Reject a render that is not a picture: a constant or near-constant RGB8 buffer (a black, white
/// or flat-grey frame). The floor is deliberately low — sixteen luma levels of spread and more
/// than sixteen distinct byte values — so it cannot reject a legitimately dark or soft image, only
/// a degenerate one. ([`assert_not_band_corrupted`] catches the opposite failure, noise.)
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

/// [`assert_not_degenerate`] rejects a blank frame and accepts a dark, low-contrast picture. Not
/// ignored and device-free: it is the one part of the Turbo step sweep that can be exercised
/// without the released weights.
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

fn out_dir() -> PathBuf {
    let dir = std::env::var("QWEN_IMAGE_2_1_RENDER_OUT")
        .map(PathBuf::from)
        .unwrap_or_else(|_| PathBuf::from("."));
    std::fs::create_dir_all(&dir).unwrap();
    dir
}

fn turbo_snapshot() -> PathBuf {
    let p = std::env::var("CANDLE_GEN_QWEN_IMAGE_2_1_TURBO_SNAPSHOT").unwrap_or_else(|_| {
        panic!("set CANDLE_GEN_QWEN_IMAGE_2_1_TURBO_SNAPSHOT to the pinned Qwen/Qwen-Image-2.1-Turbo snapshot dir; inference never self-fetches (epic 13657)")
    });
    PathBuf::from(p)
}

/// The Turbo twin of [`validation_render_default_preset`] (sc-25901): the released
/// `Qwen/Qwen-Image-2.1-Turbo` snapshot, **unmodified**, loaded as `qwen_image_2_1_turbo` through
/// the catalog path and rendered at the default 1:1 2048×2048 preset with `steps` left unset — so
/// the step count is the one the snapshot's own `sample_sigmas` gives (eight for the pinned
/// revision, asserted from the progress events), true CFG off, seed 42.
///
/// It also proves, on the real bytes, the two storage differences the tiny fixture only imitates:
/// the bf16 `vae/` and the single-file `text_encoder/model.safetensors` both load.
///
/// Run by the dispatch-only `qwen-image-2-1-turbo` profile of `real-weights.yml`; the `turbo_`
/// prefix is what that lane selects on (and what the base `qwen-image-2-1` lane leaves out — it
/// does not provision the Turbo snapshot).
///
/// ```sh
/// CANDLE_GEN_QWEN_IMAGE_2_1_TURBO_SNAPSHOT=…/models--Qwen--Qwen-Image-2.1-Turbo/snapshots/fc13ba19… \
/// QWEN_IMAGE_2_1_RENDER_OUT=…/render-validation-sc-25901 \
///   cargo test --locked --release -p candle-gen-qwen-image-2-1 --features cuda \
///   --test integration e2e_real_weights::turbo_validation_render_default_preset \
///   -- --ignored --exact --nocapture
/// ```
#[test]
#[ignore]
fn turbo_validation_render_default_preset() {
    // Unset by default: the count then comes from the snapshot, which is the claim.
    let steps: Option<u32> = std::env::var("QWEN_IMAGE_2_1_RENDER_STEPS")
        .ok()
        .map(|s| s.parse().unwrap());
    turbo_validation_renders(&[steps]);
}

/// The Turbo **step sweep** (sc-25902): the default 1:1 2048×2048 preset, the prompt and seed of
/// [`turbo_validation_render_default_preset`], rendered at 4, 6, 12 and 16 steps — the resampled
/// saved schedule on both sides of the checkpoint's own eight — from one loaded generator, one PNG
/// per step count (`qwen_image_2_1_turbo_candle_2048x2048_<N>steps_seed42.png`). It asserts that
/// every count ran as asked and that no render is band-corrupted; the PNGs are what the epic's
/// terminal story reads to choose the manifest's minimum step count.
/// `QWEN_IMAGE_2_1_SWEEP_STEPS=4,6,12,16` names other counts; `QWEN_IMAGE_2_1_RENDER_SIZE` applies.
///
/// ```sh
/// CANDLE_GEN_QWEN_IMAGE_2_1_TURBO_SNAPSHOT=…/models--Qwen--Qwen-Image-2.1-Turbo/snapshots/fc13ba19… \
/// QWEN_IMAGE_2_1_RENDER_OUT=…/render-validation-sc-25902 \
///   cargo test --locked --release -p candle-gen-qwen-image-2-1 --features cuda \
///   --test integration e2e_real_weights::turbo_validation_render_step_sweep \
///   -- --ignored --exact --nocapture
/// ```
#[test]
#[ignore]
fn turbo_validation_render_step_sweep() {
    let steps: Vec<Option<u32>> = match std::env::var("QWEN_IMAGE_2_1_SWEEP_STEPS") {
        Ok(list) => list
            .split(',')
            .map(|steps| Some(steps.trim().parse().expect("a step count")))
            .collect(),
        Err(_) => vec![Some(4), Some(6), Some(12), Some(16)],
    };
    turbo_validation_renders(&steps);
}

/// One default-preset Turbo render per entry of `steps`, all from one loaded generator; a `None`
/// entry leaves the step count to the snapshot's saved sigmas. Each asserts the step total the
/// sampler reported.
fn turbo_validation_renders(steps: &[Option<u32>]) {
    let root = turbo_snapshot();
    let (mut width, mut height) = (PRESETS[0].width, PRESETS[0].height);
    if let Ok(size) = std::env::var("QWEN_IMAGE_2_1_RENDER_SIZE") {
        let (w, h) = size.split_once('x').expect("WxH");
        width = w.parse().unwrap();
        height = h.parse().unwrap();
    }
    let out_dir = out_dir();

    let saved = candle_gen_qwen_image_2_1::load_saved_sigma_schedule(&root).unwrap();
    eprintln!("snapshot sample_sigmas: {:?}", saved.sample_sigmas);
    assert_eq!(
        saved.default_steps(),
        8,
        "the pinned Turbo revision saves eight sigmas"
    );

    let started = Instant::now();
    let registry = candle_gen_qwen_image_2_1::provider_registry().unwrap();
    let generator = registry
        .load(
            candle_gen_qwen_image_2_1::TURBO_MODEL_ID,
            &LoadSpec::new(WeightsSource::Dir(root)),
        )
        .unwrap();
    eprintln!("loaded in {:.1}s", started.elapsed().as_secs_f32());

    let mut renders: Vec<(u32, Vec<u8>)> = Vec::with_capacity(steps.len());
    for steps in steps {
        let req = GenerationRequest {
            prompt: "A neon shop sign that reads \"QWEN IMAGE 2.1\", rainy night, reflections on wet pavement"
                .to_owned(),
            width,
            height,
            steps: *steps,
            seed: Some(42),
            ..Default::default()
        };
        // Cached by default, like every Turbo render here.
        let cache = prefix_kv_cache_override().unwrap_or(PrefixKvCacheMode::Cached);
        let req = selecting(req, cache);
        let render_started = Instant::now();
        let mut totals = Vec::new();
        let (out, _) = render_reporting(
            generator.as_ref(),
            &req,
            cache,
            "turbo_default_preset",
            &mut |p| {
                if let Progress::Step { current, total } = p {
                    totals.push(total);
                    eprintln!(
                        "step {current}/{total} ({:.1}s)",
                        render_started.elapsed().as_secs_f32()
                    );
                } else {
                    eprintln!("{p:?} ({:.1}s)", render_started.elapsed().as_secs_f32());
                }
            },
        );
        let ran = steps.unwrap_or(saved.default_steps() as u32);
        assert_eq!(
            totals,
            vec![ran; ran as usize],
            "the requested step count, or one step per saved sigma"
        );
        let GenerationOutput::Images(images) = out else {
            panic!("images expected");
        };
        let image = &images[0];
        assert_eq!((image.width, image.height), (width, height));
        assert_eq!(image.pixels.len(), (width as usize) * (height as usize) * 3);
        let path = out_dir.join(format!(
            "qwen_image_2_1_turbo_candle_{width}x{height}_{ran}steps_seed42.rgb"
        ));
        std::fs::write(&path, &image.pixels).unwrap();
        save_png(
            &path.with_extension("png"),
            &image.pixels,
            width,
            height,
            image::ColorType::Rgb8,
        );
        assert_not_band_corrupted(
            &format!("turbo_default_preset_{ran}steps"),
            &image.pixels,
            width,
            height,
        );
        // Evidence someone reads to pick a step count must not be a blank frame, and two step
        // counts must not have produced the same picture (a dead `steps` knob).
        assert_not_degenerate(&format!("turbo_default_preset_{ran}steps"), &image.pixels);
        for (other, pixels) in &renders {
            assert!(
                *other == ran || *pixels != image.pixels,
                "{other} and {ran} steps rendered identical pixels"
            );
        }
        renders.push((ran, image.pixels.clone()));
        eprintln!(
            "wrote {} ({width}x{height} RGB8) after {:.1}s total ({:.1}s render)",
            path.display(),
            started.elapsed().as_secs_f32(),
            render_started.elapsed().as_secs_f32()
        );
    }
}

/// A deterministic synthetic reference: three independent colour sawtooths. It is **not a
/// picture** — a real-weight edit over it faithfully reproduces the sawtooth as a noise mosaic
/// (sc-25909's CUDA evidence, run 38664658356: chroma busyness 32–41, [`assert_not_mosaic`]), so it
/// proves nothing about image correctness. It stays for the miniature-fixture dry runs, the
/// eleven-reference refusal (refused before any image is read), the base edit's AT4 golden
/// comparison (the pre-epic goldens were rendered over exactly these inputs), and as the negative
/// calibration of [`assert_not_mosaic`].
fn synthetic_reference(seed: u32, width: u32, height: u32) -> candle_gen::gen_core::Image {
    let mut pixels = Vec::with_capacity((width * height * 3) as usize);
    for y in 0..height {
        for x in 0..width {
            pixels.push(((x * 3 + y * 5 + seed * 17) % 256) as u8);
            pixels.push(((x * 7 + seed * 29) % 256) as u8);
            pixels.push(((y * 11 + seed * 41) % 256) as u8);
        }
    }
    candle_gen::gen_core::Image {
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
/// visibly built from them.
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

/// Where an edit smoke's references come from when `QWEN_IMAGE_2_1_EDIT_REFS` names no directory.
#[derive(Clone, Copy, Debug)]
enum ReferenceSource {
    /// [`synthetic_reference`] — for the dry runs and the AT4 golden comparison only.
    Synthetic,
    /// **Real pictures, rendered in the same run** by the generator under test
    /// ([`rendered_references`]): the CUDA lane provisions no reference images, and the Qwen
    /// licence keeps none in this repository, so the run makes its own — `edge`² text-to-image
    /// renders of [`REFERENCE_SUBJECTS`], written beside the evidence.
    Rendered { edge: u32 },
}

impl ReferenceSource {
    /// Whether the edit output over these references must be a picture ([`assert_not_mosaic`]).
    fn is_real(self) -> bool {
        matches!(self, Self::Rendered { .. }) || std::env::var_os(EDIT_REFS_ENV).is_some()
    }
}

/// `QWEN_IMAGE_2_1_EDIT_REFS`: a directory of raw `WxH.rgb` files to use as the references
/// instead, in name order (operator-supplied; each file's dimensions come from its stem).
const EDIT_REFS_ENV: &str = "QWEN_IMAGE_2_1_EDIT_REFS";

/// `count` text-to-image renders of [`REFERENCE_SUBJECTS`] by `generator`, `edge`² each, at the
/// generator's own step count and seeds `101..`, prefill (no cache selected). Each is held to the
/// same checks as any evidence render and written as `<stem>_reference_<n>_<edge>x<edge>.png`.
fn rendered_references(
    generator: &dyn Generator,
    count: usize,
    edge: u32,
    out_dir: &Path,
    stem: &str,
) -> Vec<candle_gen::gen_core::Image> {
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
            save_png(
                &out_dir.join(format!("{stem}_reference_{}_{edge}x{edge}.png", i + 1)),
                &image.pixels,
                edge,
                edge,
                image::ColorType::Rgb8,
            );
            eprintln!(
                "{label}: {subject:?} in {:.1}s",
                started.elapsed().as_secs_f32()
            );
            image
        })
        .collect()
}

/// The `count` ordered references of an edit smoke: the operator's `QWEN_IMAGE_2_1_EDIT_REFS`
/// directory when set, else `source`'s.
fn references(
    generator: &dyn Generator,
    source: ReferenceSource,
    count: usize,
    out_dir: &Path,
    stem: &str,
) -> Vec<candle_gen::gen_core::Image> {
    let Ok(dir) = std::env::var(EDIT_REFS_ENV) else {
        return match source {
            ReferenceSource::Synthetic => (0..count)
                .map(|i| synthetic_reference(i as u32 + 1, REFERENCE_EDGE, REFERENCE_EDGE))
                .collect(),
            ReferenceSource::Rendered { edge } => {
                rendered_references(generator, count, edge, out_dir, stem)
            }
        };
    };
    let mut paths: Vec<_> = std::fs::read_dir(&dir)
        .unwrap_or_else(|e| panic!("{EDIT_REFS_ENV}={dir}: {e}"))
        .filter_map(|entry| entry.ok().map(|entry| entry.path()))
        .filter(|path| path.extension().is_some_and(|ext| ext == "rgb"))
        .collect();
    paths.sort();
    assert!(
        paths.len() >= count,
        "{EDIT_REFS_ENV}={dir} holds {} .rgb files, {count} needed",
        paths.len()
    );
    paths
        .into_iter()
        .take(count)
        .map(|path| {
            let stem = path
                .file_stem()
                .and_then(|s| s.to_str())
                .unwrap_or_else(|| panic!("{}: a WxH.rgb name", path.display()));
            let (w, h) = stem
                .split_once('x')
                .unwrap_or_else(|| panic!("{}: a WxH.rgb name", path.display()));
            let (w, h): (u32, u32) = (w.parse().unwrap(), h.parse().unwrap());
            let pixels = std::fs::read(&path).unwrap();
            assert_eq!(
                pixels.len(),
                (w as usize) * (h as usize) * 3,
                "{}: {w}x{h} RGB8",
                path.display()
            );
            candle_gen::gen_core::Image {
                width: w,
                height: h,
                pixels,
            }
        })
        .collect()
}

/// **Chroma busyness**: the RGB8 image box-averaged onto a 64×64 grid, then the mean absolute
/// difference between neighbouring cells of its two chroma channels (`B − Y`, `R − Y`, in 0..=255
/// levels), averaged over both axes and both channels. A photograph or a rendered scene changes
/// colour slowly at that scale; a colour mosaic changes it every few pixels. The grid makes it
/// independent of the render size (cells are `width/64 × height/64` pixels; a remainder is
/// ignored).
fn chroma_busyness(pixels: &[u8], width: u32, height: u32) -> f64 {
    const GRID: usize = 64;
    let (w, h) = (width as usize, height as usize);
    assert_eq!(pixels.len(), w * h * 3, "RGB8 geometry");
    assert!(
        w >= GRID && h >= GRID,
        "{w}x{h} is below the {GRID}-cell grid"
    );
    let (cw, ch) = (w / GRID, h / GRID);
    let mut cells = vec![[0f64; 2]; GRID * GRID];
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
            cells[gy * GRID + gx] = [sum[0] / n, sum[1] / n];
        }
    }
    let mut total = 0f64;
    for channel in 0..2 {
        let (mut across, mut down) = (0f64, 0f64);
        for gy in 0..GRID {
            for gx in 0..GRID {
                let here = cells[gy * GRID + gx][channel];
                if gx + 1 < GRID {
                    across += (cells[gy * GRID + gx + 1][channel] - here).abs();
                }
                if gy + 1 < GRID {
                    down += (cells[(gy + 1) * GRID + gx][channel] - here).abs();
                }
            }
        }
        let pairs = (GRID * (GRID - 1)) as f64;
        total += (across / pairs + down / pairs) / 2.0;
    }
    total / 2.0
}

/// The chroma busyness ([`chroma_busyness`]) above which a render is a colour mosaic, not a
/// picture. Calibrated once, offline, on images that exist (not a machine-dependent value):
///
/// * known-bad — the CUDA edit renders over synthetic references (run 38664658356): 31.9 (ten
///   references, 512²) and 40.9 (two references, 1024²); the synthetic reference itself scores
///   higher still ([`the_mosaic_check_rejects_the_synthetic_references_only`]);
/// * known-good — that run's 2048² Turbo text-to-image renders 3.1–3.4, and every final render
///   committed under `docs/migration/evidence/` 3.0–19.3 (the highest: a saturated, busy Sana Heun
///   render at 19.3 and a Kolors Heun lantern scene at 16.0).
///
/// 25 sits between the busiest real picture (19.3) and the mildest mosaic (31.9).
const MOSAIC_CHROMA_BAR: f64 = 25.0;

/// Reject a render that is a colour mosaic rather than a picture — the failure the band and
/// degenerate checks both pass (a sawtooth reference reproduced by a correct edit).
fn assert_not_mosaic(label: &str, pixels: &[u8], width: u32, height: u32) {
    let busyness = chroma_busyness(pixels, width, height);
    eprintln!("{label}: chroma busyness {busyness:.2} (mosaic above {MOSAIC_CHROMA_BAR})");
    assert!(
        busyness <= MOSAIC_CHROMA_BAR,
        "{label}: chroma busyness {busyness:.2} > {MOSAIC_CHROMA_BAR} — a colour mosaic, not a \
         picture derived from its references"
    );
}

/// [`assert_not_mosaic`] rejects the synthetic sawtooth references at the sizes an edit renders
/// and accepts a smooth synthetic scene with hard-edged shapes — device-free, so it runs on every
/// `cuda` / `metal` test lane.
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

/// One bounded real-weight **edit** render per reference count (sc-24110) — the candle twin of the
/// MLX smoke, aimed at the CUDA lane: the two-reference 1024x1024 case a caller actually sends,
/// and the ten-reference boundary at a small target so the longest joint sequence is exercised
/// without a long render. Finishes by proving eleven references are refused at `validate`, before
/// any weight is touched.
///
/// Every condition image is fitted to `output_resolution` (1024 px) whatever the target size, so
/// ten references are ~41k prefix tokens on their own.
///
/// **Cache modes and what they are priced at** (derived, bf16; the surcharge above the request's
/// cache-shed demand, printed on each render's `[kv]` line):
///
/// * two references, 1024² — **cached**: 5.30 GiB priced (`prefix_kv_cache_demand_bytes`; ~4.0 GiB
///   actually held for two square-fitted references and a short prompt);
/// * ten references, 512² — **prefill**, deliberately: this is the pre-epic boundary render, and
///   it stays the evidence for the cache-shed demand. It recomputes the ~41k-token prefix at each
///   of its two steps, which is why it runs the fewest steps the base sampler accepts.
///
/// `QWEN_IMAGE_2_1_PREFIX_KV_CACHE=cached|prefill` forces one mode on both — and
/// `QWEN_IMAGE_2_1_PREFIX_KV_CACHE=prefill QWEN_IMAGE_2_1_RENDER_GOLDEN=<dir>` is AT4's pre-epic
/// comparison of both renders ([`compare_with_golden`]; *The pre-epic golden (AT4)* in
/// `UPSTREAM.md`).
///
/// **References.** Without a golden the references are real pictures rendered in the same run
/// ([`ReferenceSource::Rendered`]) and each edit output must be a picture ([`assert_not_mosaic`]);
/// with a golden they are the [`synthetic_reference`]s the pre-epic goldens were rendered over —
/// a comparison of the same inputs, which proves the port unchanged and nothing about the image.
///
/// ```sh
/// CANDLE_GEN_QWEN_IMAGE_2_1_SNAPSHOT=.../snapshots/790c9263... \
/// QWEN_IMAGE_2_1_RENDER_OUT=.../render-validation-sc-24110 \
///   cargo test --locked --release -p candle-gen-qwen-image-2-1 --features cuda \
///   --test integration e2e_real_weights::validation_render_reference_edit \
///   -- --ignored --nocapture --test-threads 1
/// ```
#[test]
#[ignore]
fn validation_render_reference_edit() {
    let golden = render_golden();
    let source = if golden.is_some() {
        ReferenceSource::Synthetic
    } else {
        ReferenceSource::Rendered {
            edge: REFERENCE_EDGE,
        }
    };
    reference_edit_renders(
        &Target::dense("qwen_image_2_1", snapshot(), "qwen_image_2_1_candle"),
        source,
        candle_gen_qwen_image_2_1::DEFAULT_STEPS,
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
/// snapshot's saved sigmas (eight on the pinned revision), asserted from the progress events —
/// and **both cached** (sc-25905): the ten-reference case is the epic's cached ten-reference
/// render, paying for its ~41k-token prefix once instead of eight times. Finishes with the
/// eleven-reference refusal.
///
/// Then the **forced-shed twins** (AT3): **each** cached case — the ten-reference one, which is
/// AT3's cached ten-reference run, and the two-reference one — rendered again right after it with
/// the cache shed, and compared with its cached render — the epic's "one run forced to shed that
/// produces the prefill output". The two paths are the same mathematics and agree to rounding,
/// not bit for bit; the bar is a mean absolute pixel difference of at most 2 of 255, and the mean
/// and maximum are printed. **That bar has never been run on released weights or in bf16**: if
/// the first real run exceeds it, the printed numbers are the finding.
///
/// Each twin pair is also AT3's / AT5's memory comparison on CUDA ([`device_memory`]): the pool's
/// live-byte high-water over the cached render and over its shed twin, their difference
/// (`observed_cache_delta`) printed beside the priced cache bytes and held to `delta ≤ priced ·
/// 1.10 + 1 GiB` (and, for the ten-reference twin, `> 0`), with every render's baseline and peak
/// written to `qwen_image_2_1_turbo_candle_prefix_kv_cache_memory.csv` in
/// `QWEN_IMAGE_2_1_RENDER_OUT` — the directory the lane uploads as
/// `qwen-image-2-1-turbo-cuda-evidence`.
///
/// **What the cached ten-reference render needs on the CUDA lane** (derived; bf16; nothing here
/// is measured, and this lane has never executed): 28.0 GiB of resident weights (the default
/// `Resident` load keeps the tower beside the DiT and VAE), **~20.0 GiB of prefix cache** held for
/// the eight steps (24.5 GiB priced), and the prefill forward's own transient over the 42k-token
/// joint sequence on top — a structural count of about 4 GiB, plus the ten reference encodes.
/// Roughly **53–58 GiB against the 95.6 GiB of one card of the lane's box** (2× RTX PRO 6000
/// Blackwell): it is expected to fit with tens of GiB to spare; the observed peaks above are the
/// measurement, and a co-tenant on the same card is the one thing that could change that.
///
/// `QWEN_IMAGE_2_1_PREFIX_KV_CACHE=cached|prefill` forces one mode on every render, and then the
/// twin is skipped (there is nothing to compare it with).
///
/// ```sh
/// CANDLE_GEN_QWEN_IMAGE_2_1_TURBO_SNAPSHOT=…/models--Qwen--Qwen-Image-2.1-Turbo/snapshots/fc13ba19… \
/// QWEN_IMAGE_2_1_RENDER_OUT=…/render-validation-sc-25902 \
///   cargo test --locked --release -p candle-gen-qwen-image-2-1 --features cuda \
///   --test integration e2e_real_weights::turbo_validation_render_reference_edit \
///   -- --ignored --exact --nocapture
/// ```
#[test]
#[ignore]
fn turbo_validation_render_reference_edit() {
    let root = turbo_snapshot();
    let saved = candle_gen_qwen_image_2_1::load_saved_sigma_schedule(&root)
        .unwrap()
        .default_steps() as u32;
    assert_eq!(saved, 8, "the pinned Turbo revision saves eight sigmas");
    reference_edit_renders(
        &Target::dense(
            candle_gen_qwen_image_2_1::TURBO_MODEL_ID,
            root,
            "qwen_image_2_1_turbo_candle",
        ),
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

/// **Where the cached and prefill paths part, per model evaluation** (sc-25909 follow-up) — the
/// component-level diagnostic behind the forced-shed twins of
/// [`turbo_validation_render_reference_edit`]. On the released Turbo weights, over the same rendered
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
///   the relative L2 gap `‖v_cached − v_prefill‖ / ‖v_prefill‖` is printed per step and held to
///   [`CACHED_VELOCITY_GAP_BAR`]. Mathematically zero; in bf16 it is the rounding of the
///   per-token projections, which run on `S` joint rows in the prefill and on the target rows
///   alone when cached — cuBLAS chooses its GEMM kernel (tiling, split-K) by the row count `M`, so
///   the target rows' reduction order can differ between the two. The `[gemm]` lines measure that
///   directly on this device, with random weights at the DiT's own projection shapes and this
///   case's `(S, target)` row counts;
/// * **the trajectories** — the cached path's own trajectory is run beside the prefill one and the
///   relative latent gap printed per step, and both final latents are decoded and compared in
///   pixels (`[kv-velocity] … pixels`), the number the forced-shed twin bar is about.
///
/// ```sh
/// CANDLE_GEN_QWEN_IMAGE_2_1_TURBO_SNAPSHOT=…/models--Qwen--Qwen-Image-2.1-Turbo/snapshots/fc13ba19… \
/// QWEN_IMAGE_2_1_RENDER_OUT=…/render-validation \
///   cargo test --locked --release -p candle-gen-qwen-image-2-1 --features cuda \
///   --test integration e2e_real_weights::turbo_validation_prefix_kv_cache_velocity \
///   -- --ignored --exact --nocapture
/// ```
#[test]
#[ignore]
fn turbo_validation_prefix_kv_cache_velocity() {
    use candle_gen::gen_core::Conditioning;
    use candle_gen_qwen_image_2_1::{
        collect_references, create_noise, decode_rgb, encode_references, joint_branch,
        joint_images, load_text_encoder, load_tokenizer, load_transformer, load_vae,
        prepare_conditioning_references, scheduler::resample_saved_sigmas,
    };

    let root = turbo_snapshot();
    let out_dir = out_dir();
    let stem = "qwen_image_2_1_turbo_candle";
    let saved = candle_gen_qwen_image_2_1::load_saved_sigma_schedule(&root).unwrap();
    let steps = saved.default_steps();
    assert_eq!(steps, 8, "the pinned Turbo revision saves eight sigmas");
    let sigmas = resample_saved_sigmas(&saved.sample_sigmas, steps).unwrap();

    // The references the edit smoke renders (same subjects, seeds and edge), made by the
    // generator, which is then dropped before the components load.
    let pool = {
        let generator = Target::dense(
            candle_gen_qwen_image_2_1::TURBO_MODEL_ID,
            root.clone(),
            stem,
        )
        .load();
        let source = ReferenceSource::Rendered {
            edge: REFERENCE_EDGE,
        };
        references(generator.as_ref(), source, MAX_REFERENCES, &out_dir, stem)
    };

    let device = candle_gen::default_device().unwrap();
    let tokenizer = load_tokenizer(&root).unwrap();
    let drop_count = system_prompt_drop_count(&tokenizer).unwrap();
    let cases = [(2usize, 1024u32, 1024u32), (MAX_REFERENCES, 512, 512)];
    // Both cases' text side first, so the tower is dropped before the DiT loads.
    let branches: Vec<_> = {
        let te = load_text_encoder(&root, &device).unwrap();
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
                (prepared, branch)
            })
            .collect()
    };
    let transformer = load_transformer(&root, &device).unwrap();
    let vae = load_vae(&root, &device).unwrap();
    let cfg = transformer.config().clone();
    let inner = cfg.inner_dim();

    let mut failures = Vec::new();
    for (&(count, width, height), (prepared, branch)) in cases.iter().zip(&branches) {
        let label = format!("{stem} {count} refs {width}x{height}");
        let (total, target) = (branch.layout.total_len(), branch.layout.target_tokens());
        eprintln!(
            "[kv-velocity] {label}: joint S = {total} rows (prefix {}), target {target} rows",
            branch.layout.prefix_len()
        );
        gemm_row_probe(&label, &device, total, target, inner, inner * cfg.mlp_ratio);
        if strict_cublas_reductions(&device, true) {
            gemm_row_probe(
                &format!("{label} [no reduced-precision reductions]"),
                &device,
                total,
                target,
                inner,
                inner * cfg.mlp_ratio,
            );
            strict_cublas_reductions(&device, false);
        }

        let reference_latents = encode_references(&vae, prepared).unwrap();
        let text = &branch.text;
        let layout = &branch.layout;
        let prefill = |x: &Tensor, sigma: f32| {
            transformer
                .forward_joint(text, &joint_images(&reference_latents, x), sigma, layout)
                .unwrap()
        };
        let mut x_prefill = create_noise(42, width, height, cfg.in_channels, &device).unwrap();
        let mut x_cached = x_prefill.clone();
        let mut cache = None;
        let (mut worst_gap, mut worst_ratio) = (0f64, 0f64);
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
                    let same_input = transformer
                        .forward_joint_cached(&x_prefill, sigma, cache)
                        .unwrap();
                    let (gap, max_abs, differing) = velocity_gap(&same_input, &v_prefill);
                    // The control: the prefill path's own response to an input perturbation of
                    // bf16-rounding size — what any one-ulp difference does to this velocity.
                    let (control, _, _) =
                        velocity_gap(&prefill(&ulp_nudged(&x_prefill), sigma), &v_prefill);
                    let ratio = gap / control.max(f64::MIN_POSITIVE);
                    worst_gap = worst_gap.max(gap);
                    worst_ratio = worst_ratio.max(ratio);
                    let rms = rms(&v_prefill);
                    eprintln!(
                        "[kv-velocity] {label}: step {} cached vs prefill on the same input: rel \
                         {gap:.3e}, max |Δ| {max_abs:.3e} (velocity rms {rms:.3e}), {differing} \
                         elements differ; one-ulp input control rel {control:.3e}; ratio \
                         {ratio:.2}",
                        i + 1
                    );
                    // The mechanism, at the first cached evaluation and the last: the same pair
                    // with cuBLAS forbidden to reduce split-K partials in bf16.
                    if (i == 1 || i + 1 == steps) && strict_cublas_reductions(&device, true) {
                        let strict_prefill = prefill(&x_prefill, sigma);
                        let strict_cached = transformer
                            .forward_joint_cached(&x_prefill, sigma, cache)
                            .unwrap();
                        strict_cublas_reductions(&device, false);
                        let (strict_gap, strict_max, strict_differing) =
                            velocity_gap(&strict_cached, &strict_prefill);
                        let (_, _, prefill_moved) = velocity_gap(&strict_prefill, &v_prefill);
                        eprintln!(
                            "[kv-velocity] {label}: step {} [no reduced-precision reductions] \
                             cached vs prefill: rel {strict_gap:.3e}, max |Δ| {strict_max:.3e}, \
                             {strict_differing} elements differ (the prefill itself moved in \
                             {prefill_moved} elements)",
                            i + 1
                        );
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
            x_prefill = (&x_prefill + (&v_prefill * f64::from(dt)).unwrap()).unwrap();
            x_cached = (&x_cached + (&v_cached * f64::from(dt)).unwrap()).unwrap();
            let (traj, _, _) = velocity_gap(&x_cached, &x_prefill);
            eprintln!(
                "[kv-velocity] {label}: after step {}: trajectory latent gap rel {traj:.3e}",
                i + 1
            );
        }
        let decode = |x: &Tensor| {
            decode_rgb(&vae, x, width, height, None, None)
                .unwrap()
                .pixels
        };
        let (pixels_prefill, pixels_cached) = (decode(&x_prefill), decode(&x_cached));
        let (mean, max) = pixel_difference(&pixels_cached, &pixels_prefill);
        eprintln!(
            "[kv-velocity] {label}: pixels cached vs prefill trajectory: mean |Δ| {mean:.4}, max \
             {max}, 16-grid {:.2}; worst per-evaluation velocity gap rel {worst_gap:.3e}, \
             {worst_ratio:.2}× the one-ulp control (bar {CACHED_GAP_CONTROL_RATIO_BAR})",
            grid_difference(&pixels_cached, &pixels_prefill, width, height)
        );
        save_png(
            &out_dir.join(format!(
                "{stem}_velocity_{count}refs_{width}x{height}_{steps}steps_prefill.png"
            )),
            &pixels_prefill,
            width,
            height,
            image::ColorType::Rgb8,
        );
        if worst_ratio > CACHED_GAP_CONTROL_RATIO_BAR {
            failures.push(format!(
                "{label}: a cached evaluation's velocity is {worst_ratio:.2}× further from the \
                 prefill one on the same input than the one-ulp control, above \
                 {CACHED_GAP_CONTROL_RATIO_BAR}"
            ));
        }
    }
    assert!(failures.is_empty(), "{}", failures.join("\n"));
}

/// How many times the one-ulp control's gap a cached evaluation's velocity may sit from the
/// prefill one on the same latent and timestep ([`turbo_validation_prefix_kv_cache_velocity`]).
/// Provisional until the second CUDA measurement.
const CACHED_GAP_CONTROL_RATIO_BAR: f64 = 10.0;

/// `x` with every element scaled by `1 ± 2⁻⁸` (a fixed pseudo-random sign per element): about
/// half a bf16 ulp on average, so once the DiT casts its input to bf16 roughly half the elements
/// move by one ulp — a perturbation the size of one bf16 rounding.
fn ulp_nudged(x: &Tensor) -> Tensor {
    let n = x.elem_count();
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
    let factors = Tensor::from_vec(factors, x.shape(), x.device()).unwrap();
    (x * factors).unwrap()
}

/// Switch this device's cuBLAS handle between its default math mode (`strict = false`, which lets
/// a split-K GEMM reduce its partial sums in the bf16 output type) and
/// `CUBLAS_MATH_DISALLOW_REDUCED_PRECISION_REDUCTION` (`strict = true`, partials reduced in the f32
/// compute type) — the switch `candle-audio-yue2`'s VAE sets on its own handle. Returns whether a
/// switch happened: `false` on a build without CUDA, where there is no cuBLAS.
#[cfg(feature = "cuda")]
fn strict_cublas_reductions(device: &candle_core::Device, strict: bool) -> bool {
    use candle_core::cuda::cudarc::cublas::sys;
    let cuda = device.as_cuda_device().unwrap();
    cuda.cuda_stream().context().bind_to_thread().unwrap();
    let blas = cuda.cublas_handle();
    let mode = if strict {
        sys::cublasMath_t::CUBLAS_MATH_DISALLOW_REDUCED_PRECISION_REDUCTION
    } else {
        sys::cublasMath_t::CUBLAS_DEFAULT_MATH
    };
    unsafe { sys::cublasSetMathMode(*blas.handle(), mode) }
        .result()
        .unwrap();
    true
}

#[cfg(not(feature = "cuda"))]
fn strict_cublas_reductions(_device: &candle_core::Device, _strict: bool) -> bool {
    false
}

/// `(relative L2, max |Δ|, elements that differ)` of `a` against `b` (same shape, any float dtype).
fn velocity_gap(a: &Tensor, b: &Tensor) -> (f64, f32, usize) {
    let a: Vec<f32> = a
        .to_dtype(candle_core::DType::F32)
        .unwrap()
        .flatten_all()
        .unwrap()
        .to_vec1()
        .unwrap();
    let b: Vec<f32> = b
        .to_dtype(candle_core::DType::F32)
        .unwrap()
        .flatten_all()
        .unwrap()
        .to_vec1()
        .unwrap();
    assert_eq!(a.len(), b.len(), "same shape");
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

/// Root mean square of a tensor's elements.
fn rms(t: &Tensor) -> f64 {
    let v: Vec<f32> = t
        .to_dtype(candle_core::DType::F32)
        .unwrap()
        .flatten_all()
        .unwrap()
        .to_vec1()
        .unwrap();
    (v.iter().map(|x| f64::from(*x).powi(2)).sum::<f64>() / v.len() as f64).sqrt()
}

/// The raw bits of a tensor's elements, on the host, for an exact comparison.
fn host_bits(t: &Tensor) -> Vec<u32> {
    t.to_dtype(candle_core::DType::F32)
        .unwrap()
        .flatten_all()
        .unwrap()
        .to_vec1::<f32>()
        .unwrap()
        .into_iter()
        .map(f32::to_bits)
        .collect()
}

/// **The row-count dependence of a projection, measured on this device**: for each of the DiT's
/// per-token projection shapes (`inner → inner`, `inner → hidden`, `hidden → inner`), a bf16
/// `candle_nn::Linear` with random weights over `total` random rows, against the same Linear over
/// the last `target` of those rows alone — the prefill's and the cached decode's `M` for the target
/// tokens. Prints how many output elements of the target rows differ. Weights are random because
/// the kernel choice depends on the shape and dtype, not the values. Printed, never asserted: which
/// kernel a vendor library picks for an `M` is the device's and the library's, not this crate's.
fn gemm_row_probe(
    label: &str,
    device: &candle_core::Device,
    total: usize,
    target: usize,
    inner: usize,
    hidden: usize,
) {
    use candle_core::{DType, Module};
    for (k, n) in [(inner, inner), (inner, hidden), (hidden, inner)] {
        let x = Tensor::randn(0f32, 1.0, (1, total, k), device)
            .unwrap()
            .to_dtype(DType::BF16)
            .unwrap();
        let w = (Tensor::randn(0f32, 1.0, (n, k), device).unwrap() / (k as f64).sqrt())
            .unwrap()
            .to_dtype(DType::BF16)
            .unwrap();
        let linear = candle_nn::Linear::new(w, None);
        let full = linear.forward(&x).unwrap();
        let full_rows = full.narrow(1, total - target, target).unwrap();
        let alone = linear
            .forward(
                &x.narrow(1, total - target, target)
                    .unwrap()
                    .contiguous()
                    .unwrap(),
            )
            .unwrap();
        let (gap, max_abs, differing) = velocity_gap(&alone, &full_rows);
        eprintln!(
            "[gemm] {label}: {k}->{n} over M={total} vs M={target}: {differing} of {} target \
             outputs differ (rel {gap:.3e}, max |Δ| {max_abs:.3e})",
            target * n
        );
    }
}

/// What a render body renders with: the engine id, how the generator is loaded, and the stem its
/// evidence files are named with.
struct Target {
    id: &'static str,
    spec: LoadSpec,
    stem: String,
}

impl Target {
    /// The dense snapshot at `root` under `id`.
    fn dense(id: &'static str, root: PathBuf, stem: &str) -> Self {
        Self {
            id,
            spec: LoadSpec::new(WeightsSource::Dir(root)),
            stem: stem.to_owned(),
        }
    }

    /// The installed packed tier `label` (`q8` / `q4`) of `id` at `<tiers>/<label>/`, loaded
    /// with the matching quantize selector; evidence named `<stem>_<label>_…`.
    fn installed(id: &'static str, tiers: &Path, label: &str, quant: Quant, stem: &str) -> Self {
        Self {
            id,
            spec: LoadSpec::new(WeightsSource::Dir(tiers.join(label))).with_quant(quant),
            stem: format!("{stem}_{label}"),
        }
    }

    fn load(&self) -> Box<dyn Generator> {
        candle_gen_qwen_image_2_1::provider_registry()
            .unwrap()
            .load(self.id, &self.spec)
            .unwrap_or_else(|e| panic!("{}: loads: {e}", self.stem))
    }
}

/// The reference-edit smoke of `target` over `references`: one render per [`EditCase`] — `steps:
/// None` leaves the count to the generator, which must then run `default_steps` — each cached case
/// followed, with `forced_shed_twin`, by its forced-shed twin, then the eleven-reference refusal.
/// `QWEN_IMAGE_2_1_PREFIX_KV_CACHE` forces one mode on every case and skips the twins; `golden`
/// is [`compare_with_golden`]'s.
fn reference_edit_renders(
    target: &Target,
    references: ReferenceSource,
    default_steps: u32,
    mut cases: Vec<EditCase>,
    forced_shed_twin: bool,
    golden: Option<&Path>,
) {
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
        true,
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

/// The **same-picture** difference: both RGB8 images box-averaged onto a 16×16 grid, then the
/// mean absolute difference of the cells over the three channels (0..=255 levels). It ignores
/// where an edge or a texture lands to within a cell and compares what the picture is — layout,
/// colour and tone. Cells are `width/16 × height/16` pixels; a remainder is ignored.
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
/// cached render. **Not a pixel bar**: the two paths are the same mathematics but not the same
/// bf16 arithmetic — cuBLAS reduces the target rows' `hidden → inner` projection differently at
/// the cached decode's row count (see [`turbo_validation_prefix_kv_cache_velocity`]) — and an
/// eight-step flow over a ~42k-token joint sequence amplifies that ulp-level difference into a
/// visibly shifted but identical scene. Measured on CUDA (run 38668492569, real references):
///
/// * the two-reference twin: bit-identical (0.00);
/// * the ten-reference twin: full-resolution mean 7.64 (max 242), but **3.41** on the grid;
/// * the same prompt and seed at different step counts (4/6, 8/12, 12/16 steps, 2048²) —
///   the same scene by construction — 4.50–5.08;
/// * two different pictures (the ten- against the two-reference edit; this run's ten-reference
///   edit against the synthetic-reference one of run 38664658356) 58.6–60.6.
///
/// 15 is three times the same-scene spread and four times under a different picture.
const TWIN_SAME_PICTURE_BAR: f64 = 15.0;

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
/// the `QWEN_IMAGE_2_1_*` environment so the dry run at the bottom of this file can drive it on
/// the miniature fixture.
///
/// `real_weights` is set by every real-weight caller and by no dry run. It arms
///
/// * the sign half of [`assert_observed_cache_delta`] for the **ten-reference** twin — the one
///   where the cache (24.5 GiB priced in bf16) dominates every other phase, so holding it must
///   raise the device peak. With fewer references the reference encode or the decode can set both
///   renders' peak, so those twins are printed and held to the upper bound only (the miniature
///   cache is a few hundred kB under phases that are not);
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
    real_weights: bool,
) {
    use candle_gen::gen_core::Conditioning;

    let stem = target.stem.as_str();
    let started = Instant::now();
    let generator = target.load();
    eprintln!("loaded in {:.1}s", started.elapsed().as_secs_f32());
    let most = cases.iter().map(|case| case.references).max().unwrap_or(0);
    let pool = self::references(generator.as_ref(), references, most, out_dir, stem);
    let check_picture = real_weights && references.is_real();

    // One render of `case`, with the cache mode `cache` (the case's own, or the twin's).
    let render = |case: EditCase,
                  cache: PrefixKvCacheMode,
                  suffix: &str|
     -> (Vec<u8>, Observed, Option<String>) {
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
        let render_started = Instant::now();
        let mut totals = Vec::new();
        let (out, observed) = render_reporting(generator.as_ref(), &req, cache, &label, &mut |p| {
            if let Progress::Step { current, total } = p {
                totals.push(total);
                eprintln!(
                    "{label}: step {current}/{total} ({:.1}s)",
                    render_started.elapsed().as_secs_f32()
                );
            } else {
                eprintln!(
                    "{label}: {p:?} ({:.1}s)",
                    render_started.elapsed().as_secs_f32()
                );
            }
        });
        let ran = steps.unwrap_or(default_steps);
        assert_eq!(
            totals,
            vec![ran; ran as usize],
            "{label}: the requested step count, or the checkpoint's own"
        );
        let GenerationOutput::Images(mut images) = out else {
            panic!("images expected");
        };
        let image = images.remove(0);
        assert_eq!((image.width, image.height), (width, height));
        assert_eq!(image.pixels.len(), (width as usize) * (height as usize) * 3);
        let file_name =
            format!("{stem}_edit_{count}refs_{width}x{height}_{ran}steps_seed42{suffix}.rgb");
        let path = out_dir.join(&file_name);
        std::fs::write(&path, &image.pixels).unwrap();
        save_png(
            &path.with_extension("png"),
            &image.pixels,
            width,
            height,
            image::ColorType::Rgb8,
        );
        let golden_failure = compare_with_golden(golden, out_dir, &file_name, &image.pixels, cache);
        assert_not_band_corrupted(&label, &image.pixels, width, height);
        if check_picture {
            assert_not_mosaic(&label, &image.pixels, width, height);
        }
        eprintln!(
            "wrote {} ({width}x{height} RGB8) after {:.1}s render",
            path.display(),
            render_started.elapsed().as_secs_f32()
        );
        (image.pixels, observed, golden_failure)
    };

    // Each twin is rendered **immediately after** the cached render it is compared with, so no
    // other request runs on the generator between the two sides of the comparison — the pixels
    // and the device peaks are both read across that one pair.
    let mut memory_rows = vec![
        "case,mode,baseline_used_bytes,peak_used_bytes,cache_bytes,priced_cache_bytes,observed_cache_delta_bytes"
            .to_owned(),
    ];
    let mut twins = 0;
    let mut golden_failures = Vec::new();
    let mut twin_failures: Vec<String> = Vec::new();
    for case in cases {
        let (cached, cached_observed, failure) = render(case, case.cache, "");
        golden_failures.extend(failure);
        if !forced_shed_twin || !case.cache.is_cached() {
            continue;
        }
        twins += 1;
        let (shed, shed_observed, failure) =
            render(case, PrefixKvCacheMode::Prefill, "_prefill_twin");
        golden_failures.extend(failure);
        let (mean, max) = pixel_difference(&shed, &cached);
        let same_picture = grid_difference(&shed, &cached, case.width, case.height);
        eprintln!(
            "[kv] {stem} forced-shed twin of the {}-reference case: mean |Δ| = {mean:.4}, max |Δ| = \
             {max} (of 255), 16-grid {same_picture:.2} against its cached render; bar: 16-grid <= \
             {TWIN_SAME_PICTURE_BAR}",
            case.references
        );
        let twin_label = format!("{stem} {}-reference twin", case.references);
        match (cached_observed.device, shed_observed.device) {
            (Some((cached_base, cached_peak)), Some((shed_base, shed_peak))) => {
                let delta = cached_peak as i128 - shed_peak as i128;
                let case_label = format!("{}refs_{}x{}", case.references, case.width, case.height);
                for (mode, base, peak, observed, delta) in [
                    (
                        "cached",
                        cached_base,
                        cached_peak,
                        cached_observed,
                        delta.to_string(),
                    ),
                    (
                        "prefill_twin",
                        shed_base,
                        shed_peak,
                        shed_observed,
                        String::new(),
                    ),
                ] {
                    memory_rows.push(format!(
                        "{case_label},{mode},{base},{peak},{},{},{delta}",
                        observed.cache_bytes, observed.priced_cache
                    ));
                }
                // Written before the assertions, so a red run still leaves the numbers.
                std::fs::write(
                    out_dir.join(format!("{stem}_prefix_kv_cache_memory.csv")),
                    memory_rows.join("\n") + "\n",
                )
                .unwrap();
                assert_observed_cache_delta(
                    &twin_label,
                    cached_peak,
                    shed_peak,
                    cached_observed.priced_cache,
                    real_weights && case.references == MAX_REFERENCES,
                );
            }
            _ => eprintln!(
                "[kv] {twin_label}: NOT MEASURED — no device memory sampler on this build (the \
                 observed cache delta is a `--features cuda` measurement)"
            ),
        }
        if same_picture > TWIN_SAME_PICTURE_BAR {
            twin_failures.push(format!(
                "the {}-reference forced-shed render is a different picture from the cached one: \
                 16-grid difference {same_picture:.2} (full-resolution mean {mean:.4}, max {max}), \
                 above {TWIN_SAME_PICTURE_BAR}",
                case.references
            ));
        }
    }
    assert!(
        !forced_shed_twin || twins > 0,
        "a forced-shed twin needs a cached case"
    );
    // After every case has rendered, written its files and its `[golden]` / `[kv]` lines.
    assert_golden_clean(&golden_failures);
    assert!(twin_failures.is_empty(), "{}", twin_failures.join("; "));

    let err = generator
        .validate(&GenerationRequest {
            prompt: "too many".to_owned(),
            width: 512,
            height: 512,
            // Refused at `validate`, before any image is looked at, so the eleven are synthetic
            // whatever `QWEN_IMAGE_2_1_EDIT_REFS` holds (a ten-file dir is the boundary itself,
            // and `references(11, …)` would panic on it instead of reaching the refusal).
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

/// The reference count whose forced-shed twin must show a positive device-peak difference.
const MAX_REFERENCES: usize = candle_gen_qwen_image_2_1::MAX_REFERENCE_IMAGES;

/// **A dry run of the render bodies** on the miniature fixtures — not `#[ignore]`d, so it runs
/// wherever this file compiles (a `cuda` / `metal` build) with no snapshot. The real-weight tests
/// above run on no pull-request lane; this drives their bodies at 64×64 into temp dirs, proving
/// their wiring and assertions and nothing about the released weights:
///
/// * the reference edit ([`reference_edit_renders_in`]) through the `qwen_image_2_1_turbo` id —
///   two and ten references, both cached, **each followed by its forced-shed twin** (AT3) and its
///   device-memory comparison ([`device_memory`]: measured on a CUDA build, a loud "not
///   measured" on Metal), then the base test's cached-beside-prefill shape;
/// * the installed Turbo tiers ([`turbo_installed_tier_renders_in`], AT1) over `q8/` and `q4/`
///   composed the way the MLX converter writes them (`turbo_tiers::compose_tier`) — the layout
///   `CANDLE_GEN_QWEN_IMAGE_2_1_TURBO_TIER_SNAPSHOT` names;
/// * the pre-epic golden comparison ([`compare_with_golden`], AT4) on the **base** id's default
///   preset and edit bodies: written once, then rendered against that directory (passes), against
///   an inverted golden (fails), with a cached render (fails) and with a file missing (fails).
///
/// *Mutations that red this:* the body dropping its cache selection from the request
/// (`render_reporting` refuses a request that does not state its mode, and a cached case then
/// reports `prefill`); the twin rendering cached again (it asserts the `prefill` path); only the
/// first cached case twinned (the ten-reference twin's file is missing); `compare_with_golden`
/// returning early (the inverted golden passes); the installed-tier body loading without the
/// quantize selector (refused).
#[test]
fn the_turbo_reference_edit_body_runs_end_to_end_on_the_miniature_fixture() {
    use std::panic::{catch_unwind, AssertUnwindSafe};

    let turbo = crate::common::turbo_snapshot();
    let out = tempfile::tempdir().unwrap();
    let id = candle_gen_qwen_image_2_1::TURBO_MODEL_ID;
    let target = Target::dense(id, turbo.path().to_path_buf(), "dry_run");
    let case = |references| EditCase {
        references,
        width: 64,
        height: 64,
        steps: None,
        cache: PrefixKvCacheMode::Cached,
    };
    // The miniature cache sits under the fixture's other phases, so the twins' sign check is not
    // armed here; the bound is, and the check itself is pinned by
    // `the_observed_cache_delta_is_bounded_by_the_priced_cache`.
    // Rendered references, so the in-run reference renders are driven too (64² on the fixture).
    reference_edit_renders_in(
        &target,
        out.path(),
        ReferenceSource::Rendered { edge: 64 },
        8,
        vec![case(2), case(10)],
        true,
        None,
        false,
    );
    for count in [2, 10] {
        for suffix in ["", "_prefill_twin"] {
            let name = format!("dry_run_edit_{count}refs_64x64_8steps_seed42{suffix}.png");
            assert!(out.path().join(&name).is_file(), "{name}");
        }
    }
    #[cfg(feature = "cuda")]
    {
        let csv = std::fs::read_to_string(out.path().join("dry_run_prefix_kv_cache_memory.csv"))
            .expect("the twins' device-memory evidence is written on CUDA");
        assert_eq!(
            csv.lines().count(),
            1 + 2 * 2,
            "a header and two rows per twin:\n{csv}"
        );
    }
    let mixed = Target::dense(id, turbo.path().to_path_buf(), "dry_run_mixed");
    reference_edit_renders_in(
        &mixed,
        out.path(),
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
        false,
    );

    // AT1: the installed-tier body over tiers composed as the converter writes them.
    let tiers = tempfile::tempdir().unwrap();
    for tier in [
        candle_gen_qwen_image_2_1::quant::Tier::Q8,
        candle_gen_qwen_image_2_1::quant::Tier::Q4,
    ] {
        crate::turbo_tiers::compose_tier(turbo.path(), tiers.path(), tier);
    }
    turbo_installed_tier_renders_in(
        tiers.path(),
        out.path(),
        (64, 64),
        8,
        ReferenceSource::Synthetic,
    );
    for tier in ["q8", "q4"] {
        for name in [
            format!("qwen_image_2_1_turbo_candle_{tier}_64x64_8steps_seed42.png"),
            format!("qwen_image_2_1_turbo_candle_{tier}_edit_2refs_64x64_8steps_seed42.png"),
            format!(
                "qwen_image_2_1_turbo_candle_{tier}_edit_2refs_64x64_8steps_seed42_prefill_twin.png"
            ),
        ] {
            assert!(out.path().join(&name).is_file(), "{name}");
        }
    }

    // AT4: the golden comparison, on the base id's cache-shed renders.
    let base = crate::common::tiny_snapshot();
    let prefill = PrefixKvCacheMode::Prefill;
    let golden = tempfile::tempdir().unwrap();
    let checked = tempfile::tempdir().unwrap();
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
    let base_edit = Target::dense("qwen_image_2_1", base.clone(), "qwen_image_2_1_candle");
    validation_render_at(base.clone(), golden.path(), (64, 64), 3, prefill, None);
    reference_edit_renders_in(
        &base_edit,
        golden.path(),
        ReferenceSource::Synthetic,
        40,
        edit_cases(),
        false,
        None,
        false,
    );
    validation_render_at(
        base.clone(),
        checked.path(),
        (64, 64),
        3,
        prefill,
        Some(golden.path()),
    );
    reference_edit_renders_in(
        &base_edit,
        checked.path(),
        ReferenceSource::Synthetic,
        40,
        edit_cases(),
        false,
        Some(golden.path()),
        false,
    );
    let name = "qwen_image_2_1_candle_64x64_3steps_seed42.rgb";
    let inverted = tempfile::tempdir().unwrap();
    let bytes: Vec<u8> = std::fs::read(golden.path().join(name))
        .unwrap()
        .into_iter()
        .map(|b| 255 - b)
        .collect();
    std::fs::write(inverted.path().join(name), bytes).unwrap();
    let render_against = |dir: &Path, mode| {
        catch_unwind(AssertUnwindSafe(|| {
            validation_render_at(base.clone(), checked.path(), (64, 64), 3, mode, Some(dir));
        }))
    };
    assert!(
        render_against(inverted.path(), prefill).is_err(),
        "an inverted golden"
    );
    assert!(
        render_against(golden.path(), PrefixKvCacheMode::Cached).is_err(),
        "a cached render"
    );
    assert!(
        render_against(out.path(), prefill).is_err(),
        "a golden directory without the file"
    );
    // A miss on the FIRST edit case still lets the second render, write its files and be
    // compared: the body asserts once, after its last case.
    let first_off = tempfile::tempdir().unwrap();
    for entry in std::fs::read_dir(golden.path()).unwrap() {
        let entry = entry.unwrap();
        std::fs::copy(entry.path(), first_off.path().join(entry.file_name())).unwrap();
    }
    let two_refs = "qwen_image_2_1_candle_edit_2refs_64x64_2steps_seed42.rgb";
    let inverted_first: Vec<u8> = std::fs::read(first_off.path().join(two_refs))
        .unwrap()
        .into_iter()
        .map(|b| 255 - b)
        .collect();
    std::fs::write(first_off.path().join(two_refs), inverted_first).unwrap();
    let rechecked = tempfile::tempdir().unwrap();
    assert!(
        catch_unwind(AssertUnwindSafe(|| {
            reference_edit_renders_in(
                &base_edit,
                rechecked.path(),
                ReferenceSource::Synthetic,
                40,
                edit_cases(),
                false,
                Some(first_off.path()),
                false,
            );
        }))
        .is_err(),
        "a first-case miss fails the body"
    );
    let report = std::fs::read_to_string(rechecked.path().join("golden_report.txt")).unwrap();
    let ten_refs = "qwen_image_2_1_candle_edit_10refs_64x64_2steps_seed42.rgb";
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
    assert!(
        rechecked.path().join(ten_refs).is_file(),
        "and its render written"
    );
}

fn turbo_tier_snapshot() -> PathBuf {
    let p = std::env::var("CANDLE_GEN_QWEN_IMAGE_2_1_TURBO_TIER_SNAPSHOT").unwrap_or_else(|_| {
        panic!(
            "set CANDLE_GEN_QWEN_IMAGE_2_1_TURBO_TIER_SNAPSHOT to a directory holding the converted \
             Qwen-Image-2.1-Turbo q8/ and q4/ tiers (the MLX `qwen_image_2_1_prequant` output); \
             inference never self-fetches (epic 13657)"
        )
    });
    let path = PathBuf::from(p);
    assert!(path.is_dir(), "{} is not a directory", path.display());
    path
}

/// The **installed Turbo tiers** on real weights (AT1, sc-25909) — the Turbo twin of
/// [`validation_render_installed_tiers`]. `CANDLE_GEN_QWEN_IMAGE_2_1_TURBO_TIER_SNAPSHOT` names
/// one directory holding a complete `q8/` and `q4/` Turbo tier — the layout the MLX converter
/// writes (`QWEN21_SRC=<turbo snapshot> QWEN21_TIER=q8|q4 cargo run --release --example
/// qwen_image_2_1_prequant -p mlx-gen-qwen-image-2-1`, default output
/// `~/SceneWorks/qwen-image-2-1-turbo-tiers/<tier>/`) and the base tier snapshot has.
///
/// **The Turbo tiers are not published yet**, so no manifest row provisions them and the
/// `candle-qwen-image-2-1-turbo` lane runs this test **only when the dispatch names a path**
/// (`qwen_image_2_1_turbo_tier_snapshot`, an absolute directory on the CUDA runner holding the
/// copied `q8/` and `q4/`); without one the lane prints a `::warning::` and writes
/// `turbo_validation_render_installed_tiers.NOT-RUN.txt` into its evidence instead — it never
/// reports the test as passed. Once the tiers are published as a pinned repository revision, add
/// its `[[models]]` row (profile `qwen-image-2-1-turbo`, this variable) to
/// `release/real-weight-models.toml`, point this variable at the materialized path in the job's
/// `env`, and select this test unconditionally.
///
/// Per tier: the directory must self-report as the tier its name claims (read from the packed
/// shapes), load packed under `qwen_image_2_1_turbo` with the matching quantize selector, and
/// render (1) a cached 1024² text-to-image over the tier's own saved sigmas — not a flat or
/// band-corrupted field — and (2) a cached two-reference 1024² edit followed by its forced-shed
/// twin (with its device-memory comparison).
///
/// ```sh
/// CANDLE_GEN_QWEN_IMAGE_2_1_TURBO_TIER_SNAPSHOT=…/qwen-image-2-1-turbo-tiers \
/// QWEN_IMAGE_2_1_RENDER_OUT=…/render-validation-sc-25909 \
///   cargo test --locked --release -p candle-gen-qwen-image-2-1 --features cuda \
///   --test integration e2e_real_weights::turbo_validation_render_installed_tiers \
///   -- --ignored --exact --nocapture
/// ```
#[test]
#[ignore]
fn turbo_validation_render_installed_tiers() {
    let tiers = turbo_tier_snapshot();
    let saved = candle_gen_qwen_image_2_1::load_saved_sigma_schedule(&tiers.join("q8"))
        .unwrap()
        .default_steps() as u32;
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

/// [`turbo_validation_render_installed_tiers`]'s body over an explicit tier directory, size,
/// output directory and reference source, so the dry run drives it on tiers composed from the
/// miniature fixture.
fn turbo_installed_tier_renders_in(
    tiers: &Path,
    out_dir: &Path,
    (width, height): (u32, u32),
    saved_steps: u32,
    references: ReferenceSource,
) {
    use candle_gen_qwen_image_2_1::quant::{installed_tier, Tier};
    let id = candle_gen_qwen_image_2_1::TURBO_MODEL_ID;
    for (label, tier, quant) in [("q8", Tier::Q8, Quant::Q8), ("q4", Tier::Q4, Quant::Q4)] {
        let dir = tiers.join(label);
        assert_eq!(
            installed_tier(&dir).unwrap(),
            tier,
            "{label}: {} does not hold the tier its name claims",
            dir.display()
        );
        let target = Target::installed(id, tiers, label, quant, "qwen_image_2_1_turbo_candle");
        let generator = target.load();
        let req = selecting(
            GenerationRequest {
                prompt: "A neon shop sign that reads \"QWEN IMAGE 2.1\", rainy night, reflections \
                         on wet pavement"
                    .to_owned(),
                width,
                height,
                steps: None,
                seed: Some(42),
                ..Default::default()
            },
            PrefixKvCacheMode::Cached,
        );
        let mut totals = Vec::new();
        let (out, _) = render_reporting(
            generator.as_ref(),
            &req,
            PrefixKvCacheMode::Cached,
            &format!("{} t2i", target.stem),
            &mut |p| {
                if let Progress::Step { total, .. } = p {
                    totals.push(total);
                }
            },
        );
        assert_eq!(
            totals,
            vec![saved_steps; saved_steps as usize],
            "{label}: one step per saved sigma"
        );
        let GenerationOutput::Images(images) = out else {
            panic!("{label}: images expected");
        };
        let image = &images[0];
        assert_eq!((image.width, image.height), (width, height));
        let path = out_dir.join(format!(
            "{}_{width}x{height}_{saved_steps}steps_seed42.rgb",
            target.stem
        ));
        std::fs::write(&path, &image.pixels).unwrap();
        save_png(
            &path.with_extension("png"),
            &image.pixels,
            width,
            height,
            image::ColorType::Rgb8,
        );
        assert_not_band_corrupted(&target.stem, &image.pixels, width, height);
        assert_not_degenerate(&target.stem, &image.pixels);
        drop(generator);
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
            true,
        );
    }
}

/// A **transparent** RGBA reference: a real 1024² picture `generator` renders in the run (the
/// first of [`rendered_references`]; the CUDA lane provisions no reference images) matted onto a
/// soft-edged disc so the alpha is a real ramp rather than a binary cut-out (a hard matte would
/// survive any resampling order and so could not exercise the premultiplied resize). The candle
/// twin of the MLX smoke's reference (sc-24111).
///
/// Set `QWEN_IMAGE_2_1_RGBA_REF` to a PNG with an alpha channel to extract from that image
/// instead (operator-supplied; nothing is rendered then).
fn transparent_reference(
    generator: &dyn Generator,
    out_dir: &Path,
    stem: &str,
) -> candle_gen::gen_core::RgbaImage {
    if let Ok(path) = std::env::var("QWEN_IMAGE_2_1_RGBA_REF") {
        let rgba = image::open(&path)
            .unwrap_or_else(|e| panic!("QWEN_IMAGE_2_1_RGBA_REF={path}: {e}"))
            .to_rgba8();
        return candle_gen::gen_core::RgbaImage {
            width: rgba.width(),
            height: rgba.height(),
            pixels: rgba.into_raw(),
        };
    }
    let picture = rendered_references(generator, 1, 1024, out_dir, stem).remove(0);
    let (width, height) = (picture.width, picture.height);
    let mut out = candle_gen::gen_core::RgbaImage::from_rgb(&picture).unwrap();
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

/// The alpha histogram claim the transparency smoke exists to make — the same three conditions as
/// the MLX twin's: the alpha is not constant; at least 2 % of pixels are substantially transparent
/// (`A < 128`) **and** at least 2 % substantially opaque (`A > 200`); and compositing over white
/// moves at least 2 % of pixels. "Four channels came back" is not evidence of transparency — a
/// port that widened an RGB decode with a constant `A = 255` would satisfy it. The full 16-bucket
/// histogram is printed and written beside the PNG.
fn assert_nontrivial_alpha(
    label: &str,
    image: &candle_gen::gen_core::RgbaImage,
    out_dir: &std::path::Path,
) {
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

/// The bounded real-weight **transparency** smoke on candle (sc-24114, the twin of the MLX
/// sc-24111 smoke): one transparent text-to-image and one RGBA-reference extraction, both at
/// 1024×1024 / 8 steps, each asserting a non-trivial alpha histogram on the emitted `RgbaImage`.
///
/// Transparency is requested by **prompt** (upstream ships no flag — see `UPSTREAM.md`);
/// `output_channels: Rgba` only decides that the decoder's alpha reaches the caller instead of
/// being composited over white. Both PNGs are written as RGBA8 so the alpha survives to disk.
///
/// ```sh
/// CANDLE_GEN_QWEN_IMAGE_2_1_SNAPSHOT=.../snapshots/790c9263... \
/// QWEN_IMAGE_2_1_RENDER_OUT=.../render-validation-sc-24114 \
///   cargo test --locked --release -p candle-gen-qwen-image-2-1 --features cuda \
///   --test integration e2e_real_weights::validation_render_transparency \
///   -- --ignored --exact --nocapture --test-threads 1
/// ```
#[test]
#[ignore]
fn validation_render_transparency() {
    transparency_renders(
        "qwen_image_2_1",
        snapshot(),
        "qwen_image_2_1_candle",
        "",
        candle_gen_qwen_image_2_1::DEFAULT_STEPS,
        Some(8),
    );
}

/// The Turbo twin of [`validation_render_transparency`] (sc-25902), through the
/// `qwen_image_2_1_turbo` id on the unmodified Turbo snapshot: the transparent text-to-image and
/// the RGBA-reference extraction at 1024×1024 with **no step count on the request** (the saved
/// eight sigmas, asserted from the progress events), each asserting the same non-trivial alpha
/// histogram.
///
/// ```sh
/// CANDLE_GEN_QWEN_IMAGE_2_1_TURBO_SNAPSHOT=…/models--Qwen--Qwen-Image-2.1-Turbo/snapshots/fc13ba19… \
/// QWEN_IMAGE_2_1_RENDER_OUT=…/render-validation-sc-25902 \
///   cargo test --locked --release -p candle-gen-qwen-image-2-1 --features cuda \
///   --test integration e2e_real_weights::turbo_validation_render_transparency \
///   -- --ignored --exact --nocapture
/// ```
#[test]
#[ignore]
fn turbo_validation_render_transparency() {
    let root = turbo_snapshot();
    let saved = candle_gen_qwen_image_2_1::load_saved_sigma_schedule(&root)
        .unwrap()
        .default_steps() as u32;
    assert_eq!(saved, 8, "the pinned Turbo revision saves eight sigmas");
    transparency_renders(
        candle_gen_qwen_image_2_1::TURBO_MODEL_ID,
        root,
        "qwen_image_2_1_turbo_candle",
        "turbo_",
        saved,
        None,
    );
}

/// The transparency smoke of engine `id` over the snapshot at `root`; `steps: None` leaves the
/// step count to the generator, which must then run `default_steps`. `stem` prefixes the PNG names
/// and `report_prefix` the `<label>_alpha.txt` reports.
fn transparency_renders(
    id: &str,
    root: PathBuf,
    stem: &str,
    report_prefix: &str,
    default_steps: u32,
    steps: Option<u32>,
) {
    use candle_gen::gen_core::{Conditioning, OutputChannels};

    let out_dir = out_dir();
    let started = Instant::now();
    let registry = candle_gen_qwen_image_2_1::provider_registry().unwrap();
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
                image: transparent_reference(generator.as_ref(), &out_dir, stem),
                strength: None,
            }],
        ),
    ];

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
        let mut totals = Vec::new();
        let out = generator
            .generate(&req, &mut |p| {
                if let Progress::Step { current, total } = p {
                    totals.push(total);
                    eprintln!(
                        "{label}: step {current}/{total} ({:.1}s)",
                        render_started.elapsed().as_secs_f32()
                    );
                } else {
                    eprintln!(
                        "{label}: {p:?} ({:.1}s)",
                        render_started.elapsed().as_secs_f32()
                    );
                }
            })
            .unwrap();
        let ran = steps.unwrap_or(default_steps);
        assert_eq!(
            totals,
            vec![ran; ran as usize],
            "{label}: the requested step count, or the checkpoint's own"
        );
        let GenerationOutput::ImagesRgba(images) = out else {
            panic!("{label}: an `output_channels: Rgba` request must emit ImagesRgba");
        };
        let image = &images[0];
        image.validate().unwrap();
        assert_eq!((image.width, image.height), (1024, 1024));
        assert_eq!(image.channels(), 4);

        save_png(
            &out_dir.join(format!("{stem}_{label}_1024x1024_{ran}steps_seed42.png")),
            &image.pixels,
            image.width,
            image.height,
            image::ColorType::Rgba8,
        );
        assert_nontrivial_alpha(&format!("{report_prefix}{label}"), image, &out_dir);
        eprintln!(
            "{label}: done after {:.1}s render",
            render_started.elapsed().as_secs_f32()
        );
    }
}

/// The published Q8/Q4 tier snapshot (`SceneWorks/qwen-image-2-1-mlx`): one directory holding a
/// complete `q8/` and `q4/` snapshot, each in the packed `model.safetensors` layout the candle
/// loader reads natively (sc-24112).
fn tier_snapshot() -> PathBuf {
    let p = std::env::var("CANDLE_GEN_QWEN_IMAGE_2_1_TIER_SNAPSHOT").unwrap_or_else(|_| {
        panic!("set CANDLE_GEN_QWEN_IMAGE_2_1_TIER_SNAPSHOT to the pinned SceneWorks/qwen-image-2-1-mlx snapshot dir (holding q8/ and q4/); inference never self-fetches (epic 13657)")
    });
    PathBuf::from(p)
}

/// The **published** tiers load and render on real weights (sc-24114). `tests/tiers.rs` proves the
/// packed loader against the committed miniature fixture; this is the one test that opens the
/// artefacts a user actually installs. Each tier must self-report as what its directory claims
/// (read from the packed shapes, not the label), load packed through the production catalog path,
/// and render a 1024×1024 / 8-step image that is not degenerate — a missed packed site reads codes
/// as floats and collapses the image to a flat field, which the spread floor rejects.
///
/// ```sh
/// CANDLE_GEN_QWEN_IMAGE_2_1_TIER_SNAPSHOT=.../snapshots/1691de01... \
/// QWEN_IMAGE_2_1_RENDER_OUT=.../render-validation-sc-24114 \
///   cargo test --locked --release -p candle-gen-qwen-image-2-1 --features cuda \
///   --test integration e2e_real_weights::validation_render_installed_tiers \
///   -- --ignored --exact --nocapture --test-threads 1
/// ```
#[test]
#[ignore]
fn validation_render_installed_tiers() {
    use candle_gen::gen_core::Quant;
    use candle_gen_qwen_image_2_1::quant::{installed_tier, Tier};

    let root = tier_snapshot();
    let out_dir = std::env::var("QWEN_IMAGE_2_1_RENDER_OUT")
        .map(PathBuf::from)
        .unwrap_or_else(|_| PathBuf::from("."));
    std::fs::create_dir_all(&out_dir).unwrap();
    let (width, height, steps) = (1024u32, 1024u32, 8u32);

    for (label, tier, quant) in [("q8", Tier::Q8, Quant::Q8), ("q4", Tier::Q4, Quant::Q4)] {
        let dir = root.join(label);
        assert_eq!(
            installed_tier(&dir).unwrap(),
            tier,
            "{label}: {} does not hold the tier its name claims",
            dir.display()
        );

        let started = Instant::now();
        let registry = candle_gen_qwen_image_2_1::provider_registry().unwrap();
        let generator = registry
            .load(
                "qwen_image_2_1",
                &LoadSpec::new(WeightsSource::Dir(dir)).with_quant(quant),
            )
            .unwrap();
        eprintln!("{label}: loaded in {:.1}s", started.elapsed().as_secs_f32());

        let req = GenerationRequest {
            prompt:
                "A neon shop sign that reads \"QWEN IMAGE 2.1\", rainy night, reflections on wet pavement"
                    .to_owned(),
            width,
            height,
            steps: Some(steps),
            seed: Some(42),
            ..Default::default()
        };
        let render_started = Instant::now();
        let out = generator
            .generate(&req, &mut |p| {
                if let Progress::Step { current, total } = p {
                    eprintln!(
                        "{label}: step {current}/{total} ({:.1}s)",
                        render_started.elapsed().as_secs_f32()
                    );
                }
            })
            .unwrap();
        let GenerationOutput::Images(images) = out else {
            panic!("{label}: images expected");
        };
        let image = &images[0];
        assert_eq!((image.width, image.height), (width, height));
        assert_eq!(image.pixels.len(), (width as usize) * (height as usize) * 3);

        let n = image.pixels.len() as f64;
        let mean = image.pixels.iter().map(|&v| v as f64).sum::<f64>() / n;
        let variance = image
            .pixels
            .iter()
            .map(|&v| (v as f64 - mean).powi(2))
            .sum::<f64>()
            / n;
        let std = variance.sqrt();
        eprintln!("{label}: pixel mean={mean:.2} std={std:.2}");
        assert!(
            std > 8.0,
            "{label}: the render is a near-flat field (std {std:.2}) — a packed site was read as \
             dense"
        );

        let path = out_dir.join(format!(
            "qwen_image_2_1_candle_{label}_{width}x{height}_{steps}steps_seed42.rgb"
        ));
        std::fs::write(&path, &image.pixels).unwrap();
        save_png(
            &path.with_extension("png"),
            &image.pixels,
            width,
            height,
            image::ColorType::Rgb8,
        );
        assert_not_band_corrupted(label, &image.pixels, width, height);
        eprintln!(
            "{label}: {:.1}s total ({:.1}s render)",
            started.elapsed().as_secs_f32(),
            render_started.elapsed().as_secs_f32()
        );
    }
}
