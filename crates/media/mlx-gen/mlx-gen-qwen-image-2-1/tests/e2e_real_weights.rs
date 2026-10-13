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
//! snapshot, and honour the same `QWEN_IMAGE_2_1_*` switches as the base test they mirror.

use std::path::PathBuf;
use std::time::Instant;

use mlx_gen::gen_core::Progress;
use mlx_gen::{GenerationOutput, GenerationRequest, LoadSpec, WeightsSource};
use mlx_gen_qwen_image_2_1::{load_tokenizer, system_prompt_drop_count, PRESETS};

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

#[test]
#[ignore]
fn validation_render_default_preset() {
    let steps: u32 = std::env::var("QWEN_IMAGE_2_1_RENDER_STEPS")
        .ok()
        .map(|s| s.parse().unwrap())
        .unwrap_or(mlx_gen_qwen_image_2_1::DEFAULT_STEPS);
    validation_render(mlx_gen_qwen_image_2_1::MODEL_ID, snapshot(), &[Some(steps)]);
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
    let reported = validation_render(mlx_gen_qwen_image_2_1::TURBO_MODEL_ID, root, &[steps]);
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
        mlx_gen_qwen_image_2_1::TURBO_MODEL_ID,
        turbo_snapshot(),
        &asked,
    );
    assert_eq!(reported, steps, "every swept step count ran as asked");
}

/// One bounded validation render of engine `id` from the snapshot at `root` per entry of `steps`,
/// all from one loaded generator; a `None` entry leaves the step count to the generator. Returns
/// the step total the sampler reported for each.
fn validation_render(id: &str, root: PathBuf, steps: &[Option<u32>]) -> Vec<u32> {
    let (mut width, mut height) = (PRESETS[0].width, PRESETS[0].height);
    if let Ok(size) = std::env::var("QWEN_IMAGE_2_1_RENDER_SIZE") {
        let (w, h) = size.split_once('x').expect("WxH");
        width = w.parse().unwrap();
        height = h.parse().unwrap();
    }
    validation_render_at(id, root, &out_dir(), (width, height), steps)
}

/// [`validation_render`] at an explicit size into an explicit directory — the body, free of the
/// `QWEN_IMAGE_2_1_RENDER_SIZE` / `_RENDER_OUT` environment so the default-run dry run at the
/// bottom of this file can drive it on the miniature fixture.
fn validation_render_at(
    id: &str,
    root: PathBuf,
    out_dir: &std::path::Path,
    (width, height): (u32, u32),
    steps: &[Option<u32>],
) -> Vec<u32> {
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
    let registry = mlx_gen_qwen_image_2_1::provider_registry().unwrap();
    let generator = registry
        .load(id, &LoadSpec::new(WeightsSource::Dir(root)))
        .unwrap();
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
        eprintln!(
            "decode tiling: {:?}",
            mlx_gen_qwen_image_2_1::pipeline::decode_tiling(&req)
        );
        let render_started = Instant::now();
        let mut log = progress_logger(String::new());
        let footprint_for_progress = footprint_max.clone();
        let mut reported_steps = Vec::new();
        let out = generator
            .generate(&req, &mut |p| {
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
            })
            .unwrap();
        if trace {
            memory_line("after decode (generate returned)", &footprint_max);
            mlx_rs::memory::clear_cache();
            memory_line("after clear_cache", &footprint_max);
        }
        let GenerationOutput::Images(images) = out else {
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
        let path = out_dir.join(format!("{id}_{width}x{height}_{ran}steps_seed42.png"));
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
        // Evidence someone reads to pick a step count must not be a blank frame, and two step
        // counts must not have produced the same picture (a dead `steps` knob).
        assert_not_degenerate(&format!("{id} {ran} steps"), &image.pixels);
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

/// One bounded real-weight **edit** render per reference count (sc-24110): the two-reference
/// 1024x1024 case a caller actually sends, and the ten-reference boundary at a small target so
/// the longest joint sequence is exercised without a long render. Finishes by proving eleven
/// references are refused at `validate`, before any weight is touched.
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
    // (references, width, height, steps) — the caller-shaped case, then the boundary.
    //
    // Every reference is fitted to `output_resolution` (1024 px) whatever the target size, so ten
    // references are ~41k prefix tokens on their own. The boundary case's step count was set to
    // the fewest the sampler accepts when this port recomputed the whole block-causal prefix
    // every step; the render path now runs upstream's prefix KV cache (sc-25903, see
    // UPSTREAM.md), under which only the first step pays for the prefix and later steps cost
    // what a text-to-image step does. The step count is left as it was.
    reference_edit_renders(
        mlx_gen_qwen_image_2_1::MODEL_ID,
        snapshot(),
        mlx_gen_qwen_image_2_1::DEFAULT_STEPS,
        vec![(2, 1024, 1024, Some(8)), (10, 512, 512, Some(2))],
    );
}

/// The Turbo twin of [`validation_render_reference_edit`] (sc-25902), through the
/// `qwen_image_2_1_turbo` id on the unmodified Turbo snapshot: the two-reference 1024x1024 case
/// and the ten-reference boundary, **both with no step count on the request** — so each runs the
/// snapshot's saved sigmas (eight on the pinned revision), asserted from the steps the sampler
/// reported. The boundary case runs the full schedule here because the prefix KV cache makes only
/// the first evaluation pay for the ~41k-token prefix. Finishes with the eleven-reference refusal.
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
        mlx_gen_qwen_image_2_1::TURBO_MODEL_ID,
        root,
        saved,
        vec![(2, 1024, 1024, None), (10, 512, 512, None)],
    );
}

/// The reference-edit smoke of engine `id` over the snapshot at `root`: one render per
/// `(references, width, height, steps)` case — `steps: None` leaves the count to the generator,
/// which must then run `default_steps` — then the eleven-reference refusal.
///
/// `QWEN_IMAGE_2_1_EDIT_CASES="refs:WxH:steps,…"` replaces the cases (sc-24114's encode transient
/// measurement: `2:2048x2048:2`); `QWEN_IMAGE_2_1_MEMORY_TRACE=1` prints the same `[mem]` phase
/// lines as the T2I render. The reference encodes are lazy, so they materialize inside the first
/// denoise step's eval: the phase that ends at `step 2 reported` is the one whose `peak_active`
/// carries the encoder's transient.
fn reference_edit_renders(
    id: &str,
    root: PathBuf,
    default_steps: u32,
    default_cases: Vec<(usize, u32, u32, Option<u32>)>,
) {
    let cases: Vec<(usize, u32, u32, Option<u32>)> =
        match std::env::var("QWEN_IMAGE_2_1_EDIT_CASES") {
            Ok(spec) => spec
                .split(',')
                .map(|case| {
                    let mut parts = case.split(':');
                    let count = parts.next().unwrap().parse().unwrap();
                    let (w, h) = parts.next().unwrap().split_once('x').expect("WxH");
                    let steps = parts.next().unwrap().parse().unwrap();
                    (count, w.parse().unwrap(), h.parse().unwrap(), Some(steps))
                })
                .collect(),
            Err(_) => default_cases,
        };
    reference_edit_renders_in(id, root, &out_dir(), default_steps, cases);
}

/// [`reference_edit_renders`] over explicit cases into an explicit directory — the body, free of
/// the `QWEN_IMAGE_2_1_EDIT_CASES` / `_RENDER_OUT` environment so the default-run dry run at the
/// bottom of this file can drive it on the miniature fixture.
fn reference_edit_renders_in(
    id: &str,
    root: PathBuf,
    out_dir: &std::path::Path,
    default_steps: u32,
    cases: Vec<(usize, u32, u32, Option<u32>)>,
) {
    use mlx_gen::gen_core::Conditioning;

    let started = Instant::now();
    let registry = mlx_gen_qwen_image_2_1::provider_registry().unwrap();
    let generator = registry
        .load(id, &LoadSpec::new(WeightsSource::Dir(root)))
        .unwrap();
    eprintln!("loaded in {:.1}s", started.elapsed().as_secs_f32());

    let trace = std::env::var_os("QWEN_IMAGE_2_1_MEMORY_TRACE").is_some();
    let footprint_max = footprint_sampler();
    for (count, width, height, steps) in cases {
        let refs = references(count, 768, 768);
        let req = GenerationRequest {
            prompt: "Combine the subjects of the reference images into one scene, evening light"
                .to_owned(),
            width,
            height,
            steps,
            seed: Some(42),
            conditioning: vec![Conditioning::MultiReference { images: refs }],
            ..Default::default()
        };
        if trace {
            memory_line("before generate", &footprint_max);
        }
        let render_started = Instant::now();
        let mut log = progress_logger(format!("{count} refs: "));
        let mut reported_steps = Vec::new();
        let out = generator
            .generate(&req, &mut |p| {
                log(p);
                if let Progress::Step { current, total } = p {
                    reported_steps.push((current, total));
                }
                if trace {
                    memory_line(&format!("{count} refs: {p:?}"), &footprint_max);
                }
            })
            .unwrap();
        if trace {
            memory_line("after decode (generate returned)", &footprint_max);
        }
        let GenerationOutput::Images(images) = out else {
            panic!("images expected");
        };
        let image = &images[0];
        assert_eq!((image.width, image.height), (width, height));
        let ran = steps.unwrap_or(default_steps);
        assert_eq!(
            reported_steps,
            every_step(ran),
            "{count} refs: the requested step count, or the checkpoint's own — one event per step"
        );
        let path = out_dir.join(format!(
            "{id}_edit_{count}refs_{width}x{height}_{ran}steps_seed42.png"
        ));
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
    }

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

/// **The Turbo smoke bodies, dry-run on the miniature Turbo fixture — not ignored** (sc-25902).
/// The `#[ignore]`d Turbo tests above compile on every lane and run on none, so a wrong step
/// assertion or a mis-built request in them would first surface in the middle of a real-weight
/// run. This drives the same two bodies — the step sweep ([`validation_render_at`]) and the
/// reference edit with its ten-reference boundary and eleven-reference refusal
/// ([`reference_edit_renders_in`]) — end to end through the `qwen_image_2_1_turbo` id at 64×64,
/// writing into a temp dir. It proves the bodies' wiring and assertions, nothing about the released
/// weights. (The transparency body is not dry-run: its alpha-histogram claim is about the released
/// checkpoint and the miniature decode does not meet it.)
///
/// *Mutations that red this:* the Turbo generator ignoring `req.steps` (the sweep then reports
/// eight steps for every count, and renders identical pixels); a step event dropped or repeated.
#[test]
fn the_turbo_smoke_bodies_run_end_to_end_on_the_miniature_fixture() {
    let tmp = tempfile::tempdir().unwrap();
    let root = crate::common::turbo_snapshot(tmp.path());
    let out = tmp.path().join("evidence");
    std::fs::create_dir_all(&out).unwrap();
    let id = mlx_gen_qwen_image_2_1::TURBO_MODEL_ID;

    let asked = [4, 6, 12, 16];
    let reported = validation_render_at(id, root.clone(), &out, (64, 64), &asked.map(Some));
    assert_eq!(reported, asked);
    for steps in asked {
        assert!(out
            .join(format!("{id}_64x64_{steps}steps_seed42.png"))
            .is_file());
    }
    // No step count: the fixture's eight saved sigmas.
    assert_eq!(
        validation_render_at(id, root.clone(), &out, (64, 64), &[None]),
        [8]
    );

    reference_edit_renders_in(
        id,
        root,
        &out,
        8,
        vec![(2, 64, 64, None), (10, 64, 64, None)],
    );
    for count in [2, 10] {
        assert!(out
            .join(format!("{id}_edit_{count}refs_64x64_8steps_seed42.png"))
            .is_file());
    }
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
