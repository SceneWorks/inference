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
fn phys_footprint() -> (u64, u64) {
    extern "C" {
        fn proc_pid_rusage(pid: i32, flavor: i32, buffer: *mut u8) -> i32;
    }
    // `rusage_info_v4` is 296 bytes: `ri_phys_footprint` at byte 72 and
    // `ri_lifetime_max_phys_footprint` at byte 240 (see `<sys/resource.h>`).
    let mut buffer = [0u8; 512];
    // SAFETY: the buffer is larger than `rusage_info_v4` and the pid is our own.
    let rc = unsafe { proc_pid_rusage(std::process::id() as i32, 4, buffer.as_mut_ptr()) };
    assert_eq!(rc, 0, "proc_pid_rusage failed");
    let read = |offset: usize| u64::from_ne_bytes(buffer[offset..offset + 8].try_into().unwrap());
    (read(72), read(240))
}

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
    let root = snapshot();
    let (mut width, mut height) = (PRESETS[0].width, PRESETS[0].height);
    if let Ok(size) = std::env::var("QWEN_IMAGE_2_1_RENDER_SIZE") {
        let (w, h) = size.split_once('x').expect("WxH");
        width = w.parse().unwrap();
        height = h.parse().unwrap();
    }
    let steps: u32 = std::env::var("QWEN_IMAGE_2_1_RENDER_STEPS")
        .ok()
        .map(|s| s.parse().unwrap())
        .unwrap_or(mlx_gen_qwen_image_2_1::DEFAULT_STEPS);
    let out_dir = std::env::var("QWEN_IMAGE_2_1_RENDER_OUT")
        .map(PathBuf::from)
        .unwrap_or_else(|_| PathBuf::from("."));
    std::fs::create_dir_all(&out_dir).unwrap();

    // `QWEN_IMAGE_2_1_MEMORY_TRACE=1` prints a `[mem]` line at every phase boundary (sc-24114's
    // footprint investigation); `QWEN_IMAGE_2_1_RENDER_TILED=1` asks for the bounded 512/64 decode
    // explicitly (the default already bounds every preset, sc-24114), and
    // `QWEN_IMAGE_2_1_CLEAR_CACHE_BEFORE_DECODE=1` sheds MLX's buffer cache at the `Decoding` event.
    let trace = std::env::var_os("QWEN_IMAGE_2_1_MEMORY_TRACE").is_some();
    let footprint_max = footprint_sampler();
    if trace {
        memory_line("before load", &footprint_max);
    }

    let started = Instant::now();
    let registry = mlx_gen_qwen_image_2_1::provider_registry().unwrap();
    let generator = registry
        .load("qwen_image_2_1", &LoadSpec::new(WeightsSource::Dir(root)))
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
    let req = GenerationRequest {
        prompt: "A neon shop sign that reads \"QWEN IMAGE 2.1\", rainy night, reflections on wet pavement"
            .to_owned(),
        width,
        height,
        steps: Some(steps),
        seed: Some(42),
        memory,
        ..Default::default()
    };
    eprintln!(
        "decode tiling: {:?}",
        mlx_gen_qwen_image_2_1::pipeline::decode_tiling(&req)
    );
    let clear_before_decode =
        std::env::var_os("QWEN_IMAGE_2_1_CLEAR_CACHE_BEFORE_DECODE").is_some();
    let render_started = Instant::now();
    let mut log = progress_logger(String::new());
    let footprint_for_progress = footprint_max.clone();
    let out = generator
        .generate(&req, &mut |p| {
            log(p);
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
    let path = out_dir.join(format!(
        "qwen_image_2_1_{width}x{height}_{steps}steps_seed42.png"
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
        "wrote {} after {:.1}s total ({:.1}s render)",
        path.display(),
        started.elapsed().as_secs_f32(),
        render_started.elapsed().as_secs_f32()
    );
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
    use mlx_gen::gen_core::Conditioning;

    let root = snapshot();
    let out_dir = std::env::var("QWEN_IMAGE_2_1_RENDER_OUT")
        .map(PathBuf::from)
        .unwrap_or_else(|_| PathBuf::from("."));
    std::fs::create_dir_all(&out_dir).unwrap();

    let started = Instant::now();
    let registry = mlx_gen_qwen_image_2_1::provider_registry().unwrap();
    let generator = registry
        .load("qwen_image_2_1", &LoadSpec::new(WeightsSource::Dir(root)))
        .unwrap();
    eprintln!("loaded in {:.1}s", started.elapsed().as_secs_f32());

    // (references, width, height, steps) — the caller-shaped case, then the boundary.
    //
    // Every reference is fitted to `output_resolution` (1024 px) whatever the target size, so ten
    // references are ~41k prefix tokens on their own, and this port recomputes the whole
    // block-causal prefix every step (no KV cache — see UPSTREAM.md). The boundary case therefore
    // runs the fewest steps the sampler accepts rather than a smaller target: only the step count
    // moves its cost.
    //
    // `QWEN_IMAGE_2_1_EDIT_CASES="refs:WxH:steps,…"` replaces the two cases (sc-24114's encode
    // transient measurement: `2:2048x2048:2`); `QWEN_IMAGE_2_1_MEMORY_TRACE=1` prints the same
    // `[mem]` phase lines as the T2I render. The reference encodes are lazy, so they materialize
    // inside the first denoise step's eval: the phase that ends at `step 2 reported` is the one
    // whose `peak_active` carries the encoder's transient.
    let cases: Vec<(usize, u32, u32, u32)> = match std::env::var("QWEN_IMAGE_2_1_EDIT_CASES") {
        Ok(spec) => spec
            .split(',')
            .map(|case| {
                let mut parts = case.split(':');
                let count = parts.next().unwrap().parse().unwrap();
                let (w, h) = parts.next().unwrap().split_once('x').expect("WxH");
                let steps = parts.next().unwrap().parse().unwrap();
                (count, w.parse().unwrap(), h.parse().unwrap(), steps)
            })
            .collect(),
        Err(_) => vec![(2, 1024, 1024, 8), (10, 512, 512, 2)],
    };
    let trace = std::env::var_os("QWEN_IMAGE_2_1_MEMORY_TRACE").is_some();
    let footprint_max = footprint_sampler();
    for (count, width, height, steps) in cases {
        let refs = references(count, 768, 768);
        let req = GenerationRequest {
            prompt: "Combine the subjects of the reference images into one scene, evening light"
                .to_owned(),
            width,
            height,
            steps: Some(steps),
            seed: Some(42),
            conditioning: vec![Conditioning::MultiReference { images: refs }],
            ..Default::default()
        };
        if trace {
            memory_line("before generate", &footprint_max);
        }
        let render_started = Instant::now();
        let mut log = progress_logger(format!("{count} refs: "));
        let out = generator
            .generate(&req, &mut |p| {
                log(p);
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
        let path = out_dir.join(format!(
            "qwen_image_2_1_edit_{count}refs_{width}x{height}_{steps}steps_seed42.png"
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
    use mlx_gen::gen_core::{Conditioning, OutputChannels};

    let root = snapshot();
    let out_dir = std::env::var("QWEN_IMAGE_2_1_RENDER_OUT")
        .map(PathBuf::from)
        .unwrap_or_else(|_| PathBuf::from("."));
    std::fs::create_dir_all(&out_dir).unwrap();

    let started = Instant::now();
    let registry = mlx_gen_qwen_image_2_1::provider_registry().unwrap();
    let generator = registry
        .load("qwen_image_2_1", &LoadSpec::new(WeightsSource::Dir(root)))
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
            steps: Some(8),
            seed: Some(42),
            conditioning,
            output_channels: OutputChannels::Rgba,
            ..Default::default()
        };
        generator
            .validate(&req)
            .unwrap_or_else(|e| panic!("{label}: the RGBA request was refused at validate: {e}"));

        let render_started = Instant::now();
        let out = generator
            .generate(&req, &mut progress_logger(format!("{label}: ")))
            .unwrap();
        let GenerationOutput::ImagesRgba(images) = out else {
            panic!("{label}: an `output_channels: Rgba` request must emit ImagesRgba");
        };
        let image = &images[0];
        image.validate().unwrap();
        assert_eq!((image.width, image.height), (1024, 1024));
        assert_eq!(image.channels(), 4);

        let path = out_dir.join(format!(
            "qwen_image_2_1_{label}_1024x1024_8steps_seed42.png"
        ));
        image::save_buffer(
            &path,
            &image.pixels,
            image.width,
            image.height,
            image::ColorType::Rgba8,
        )
        .unwrap();
        assert_nontrivial_alpha(label, image, &out_dir);
        eprintln!(
            "wrote {} after {:.1}s render",
            path.display(),
            render_started.elapsed().as_secs_f32()
        );
    }
}
