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

/// Hold the raw RGB8 render just written as `file_name` (`….rgb`) to its pre-epic golden within
/// [`GOLDEN_MEAN_BAR`] / [`GOLDEN_FAR_FRACTION`], printing mean, max and the far share. Only a
/// cache-shed render is comparable — the pre-epic revision had no cache — so a golden with a
/// cached render is a usage error; a golden directory missing the file is a failure.
fn compare_with_golden(
    golden: Option<&Path>,
    file_name: &str,
    pixels: &[u8],
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
    let want = std::fs::read(&path).unwrap_or_else(|e| {
        panic!(
            "{RENDER_GOLDEN_ENV}: {} is the golden for this render: {e}",
            path.display()
        )
    });
    let (mean, max) = pixel_difference(pixels, &want);
    let far = pixels
        .iter()
        .zip(&want)
        .filter(|(a, b)| a.abs_diff(**b) > GOLDEN_FAR_LEVELS)
        .count() as f64
        / pixels.len() as f64;
    eprintln!(
        "[golden] {file_name}: mean |Δ| = {mean:.4}, max |Δ| = {max} (of 255), {:.3} % of bytes \
         moved by more than {GOLDEN_FAR_LEVELS} against {}; bars: mean <= {GOLDEN_MEAN_BAR}, far \
         share <= {:.1} %",
        100.0 * far,
        path.display(),
        100.0 * GOLDEN_FAR_FRACTION
    );
    assert!(
        mean <= GOLDEN_MEAN_BAR && far <= GOLDEN_FAR_FRACTION,
        "{file_name}: the cache-shed render is not the pre-epic render within the approved \
         select's bound (mean |Δ| {mean:.4}, far share {:.3} %, max {max})",
        100.0 * far
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
    assert_not_band_corrupted("default_preset", &image.pixels, width, height);
    compare_with_golden(golden, &file_name, &image.pixels, cache);
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

/// A deterministic synthetic reference, so the edit smoke needs no checked-in photographs.
/// `QWEN_IMAGE_2_1_EDIT_REFS` points at a directory of raw `WxH.rgb` files to use instead, in
/// name order — reference images for a real evaluation are user-provided, never synthesised or
/// fetched here. Each file's dimensions come from its stem (`768x768.rgb`).
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

fn references(count: usize, width: u32, height: u32) -> Vec<candle_gen::gen_core::Image> {
    let Ok(dir) = std::env::var("QWEN_IMAGE_2_1_EDIT_REFS") else {
        return (0..count)
            .map(|i| synthetic_reference(i as u32 + 1, width, height))
            .collect();
    };
    let mut paths: Vec<_> = std::fs::read_dir(&dir)
        .unwrap_or_else(|e| panic!("QWEN_IMAGE_2_1_EDIT_REFS={dir}: {e}"))
        .filter_map(|entry| entry.ok().map(|entry| entry.path()))
        .filter(|path| path.extension().is_some_and(|ext| ext == "rgb"))
        .collect();
    paths.sort();
    assert!(
        paths.len() >= count,
        "QWEN_IMAGE_2_1_EDIT_REFS={dir} holds {} .rgb files, {count} needed",
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
    reference_edit_renders(
        &Target::dense("qwen_image_2_1", snapshot(), "qwen_image_2_1_candle"),
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
        render_golden().as_deref(),
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

/// The reference-edit smoke of `target`: one render per [`EditCase`] — `steps: None` leaves the
/// count to the generator, which must then run `default_steps` — each cached case followed, with
/// `forced_shed_twin`, by its forced-shed twin, then the eleven-reference refusal.
/// `QWEN_IMAGE_2_1_PREFIX_KV_CACHE` forces one mode on every case and skips the twins; `golden`
/// is [`compare_with_golden`]'s.
fn reference_edit_renders(
    target: &Target,
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

/// The mean absolute pixel difference (of 255) a forced-shed twin may show against its cached
/// render. Not measured on released weights.
const SHED_TWIN_MEAN_BAR: f64 = 2.0;

/// [`reference_edit_renders`] over explicit cases into an explicit directory — the body, free of
/// the `QWEN_IMAGE_2_1_*` environment so the dry run at the bottom of this file can drive it on
/// the miniature fixture.
///
/// `require_positive_delta` arms the sign half of [`assert_observed_cache_delta`] for the
/// **ten-reference** twin — the one where the cache (24.5 GiB priced in bf16) dominates every
/// other phase, so holding it must raise the device peak. With fewer references the reference
/// encode or the decode can set both renders' peak, so those twins are printed and held to the
/// upper bound only. Every real-weight caller sets it; the dry run cannot (the miniature cache is
/// a few hundred kB under phases that are not).
#[allow(clippy::too_many_arguments)]
fn reference_edit_renders_in(
    target: &Target,
    out_dir: &std::path::Path,
    default_steps: u32,
    cases: Vec<EditCase>,
    forced_shed_twin: bool,
    golden: Option<&Path>,
    require_positive_delta: bool,
) {
    use candle_gen::gen_core::Conditioning;

    let stem = target.stem.as_str();
    let started = Instant::now();
    let generator = target.load();
    eprintln!("loaded in {:.1}s", started.elapsed().as_secs_f32());

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
        assert_not_band_corrupted(&label, &image.pixels, width, height);
        compare_with_golden(golden, &file_name, &image.pixels, cache);
        eprintln!(
            "wrote {} ({width}x{height} RGB8) after {:.1}s render",
            path.display(),
            render_started.elapsed().as_secs_f32()
        );
        (image.pixels, observed)
    };

    // Each twin is rendered **immediately after** the cached render it is compared with, so no
    // other request runs on the generator between the two sides of the comparison — the pixels
    // and the device peaks are both read across that one pair.
    let mut memory_rows = vec![
        "case,mode,baseline_used_bytes,peak_used_bytes,cache_bytes,priced_cache_bytes,observed_cache_delta_bytes"
            .to_owned(),
    ];
    let mut twins = 0;
    for case in cases {
        let (cached, cached_observed) = render(case, case.cache, "");
        if !forced_shed_twin || !case.cache.is_cached() {
            continue;
        }
        twins += 1;
        let (shed, shed_observed) = render(case, PrefixKvCacheMode::Prefill, "_prefill_twin");
        let (mean, max) = pixel_difference(&shed, &cached);
        eprintln!(
            "[kv] {stem} forced-shed twin of the {}-reference case: mean |Δ| = {mean:.4}, max |Δ| = \
             {max} (of 255) against its cached render; bar: mean <= {SHED_TWIN_MEAN_BAR}",
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
                    require_positive_delta && case.references == MAX_REFERENCES,
                );
            }
            _ => eprintln!(
                "[kv] {twin_label}: NOT MEASURED — no device memory sampler on this build (the \
                 observed cache delta is a `--features cuda` measurement)"
            ),
        }
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
    reference_edit_renders_in(
        &target,
        out.path(),
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
    turbo_installed_tier_renders_in(tiers.path(), out.path(), (64, 64), 8);
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
    turbo_installed_tier_renders_in(&tiers, &out_dir(), (1024, 1024), saved);
}

/// [`turbo_validation_render_installed_tiers`]'s body over an explicit tier directory, size and
/// output directory, so the dry run drives it on tiers composed from the miniature fixture.
fn turbo_installed_tier_renders_in(
    tiers: &Path,
    out_dir: &Path,
    (width, height): (u32, u32),
    saved_steps: u32,
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

/// A **transparent** RGBA reference of `width`×`height`: the same deterministic picture
/// [`synthetic_reference`] produces, matted onto a soft-edged disc so the alpha is a real ramp
/// rather than a binary cut-out (a hard matte would survive any resampling order and so could not
/// exercise the premultiplied resize). The candle twin of the MLX smoke's reference (sc-24111).
///
/// Set `QWEN_IMAGE_2_1_RGBA_REF` to a PNG with an alpha channel to extract from a real image
/// instead; reference images for a real evaluation are user-provided.
fn transparent_reference(width: u32, height: u32) -> candle_gen::gen_core::RgbaImage {
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
    let rgb = synthetic_reference(1, width, height);
    let mut out = candle_gen::gen_core::RgbaImage::from_rgb(&rgb).unwrap();
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
                image: transparent_reference(1024, 1024),
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
