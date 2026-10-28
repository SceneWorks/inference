//! **Veda dense-vs-sparse A/B, one real-weight render per invocation** (epic sc-25140, sc-25777).
//!
//! The candle/CUDA twin of `mlx-gen-minimax-h3/tests/veda_ab.rs`. Each invocation renders ONE
//! MiniMax-H3 clip with the request described by the `VEDA_AB_*` environment and writes the decoded
//! clip plus a `result.json` receipt into `VEDA_AB_OUT`. An A/B pair is two invocations that differ
//! only in `VEDA_AB_VEDA` (`off` then `on`); `.github/workflows/veda-ab-cuda.yml` drives the pairs.
//! One render per process is deliberate: cudarc's caching allocator never returns pages to the
//! driver, so a second render in the same process would re-report the first one's device high-water.
//!
//! ```sh
//! MINIMAX_H3_SNAPSHOT=<staged real-file root holding vae/, audio_vae/, tokenizer/, transformer/, text_encoder/> \
//! VEDA_AB_OUT=<dir> VEDA_AB_MODE=t2va VEDA_AB_TIER=q4 \
//! VEDA_AB_DIT_DIR=<staged q4 transformer/> VEDA_AB_TE_DIR=<staged q4 text_encoder/> \
//! VEDA_AB_RECIPE=minimax_h3_turbo_8step VEDA_AB_LORAS=<turbo-8step.safetensors>:1.0 \
//! VEDA_AB_VEDA=on VEDA_AB_PREDICTOR=<veda predictor .safetensors> \
//!   cargo test -p candle-gen-minimax-h3 --features cuda --release --test integration \
//!     veda_ab::veda_ab_render -- --ignored --exact --nocapture
//! ```
//!
//! # The request contract (shared with the MLX twin)
//!
//! | variable | meaning |
//! |---|---|
//! | `VEDA_AB_OUT` | output directory (created) |
//! | `VEDA_AB_MODE` | `t2va` / `fl2va` / `r2va`; checked against the supplied images |
//! | `VEDA_AB_FIRST_IMAGE` / `VEDA_AB_LAST_IMAGE` | `fl2va` keyframes (frame 0 / frame -1) |
//! | `VEDA_AB_REF_IMAGES` | comma-separated `r2va` reference images, in order |
//! | `VEDA_AB_TIER` | `q4` / `q8` / `bf16` — the three tiers candle serves for H3 |
//! | `VEDA_AB_LORAS` | comma-separated `path:strength` (strength optional, default 1.0) |
//! | `VEDA_AB_RECIPE` | `base` or a SceneWorks turbo catalog id → steps + video/audio shift |
//! | `VEDA_AB_STEPS` | overrides the recipe's step count (the product's `advanced.steps` rule) |
//! | `VEDA_AB_WIDTH` / `VEDA_AB_HEIGHT` / `VEDA_AB_FRAMES` / `VEDA_AB_SEED` / `VEDA_AB_PROMPT` | request |
//! | `VEDA_AB_VEDA` | `off` → `veda: None`; `on` → `VedaRequest { enabled: true, ..Default }` |
//! | `VEDA_AB_PREDICTOR` | predictor file, staged as `LoadSpec.components["veda_predictor"]` when `on` |
//!
//! # Where the weights come from (candle only)
//!
//! Mirrors the SceneWorks worker's candle route (`resolve_candle_minimax_h3_load` +
//! `video_load_spec`): `spec.weights` is the base snapshot root (`MINIMAX_H3_SNAPSHOT`); a packed
//! tier stages its `transformer/` and `text_encoder/` as the `"transformer"` / `"text_encoder"`
//! components (`VEDA_AB_DIT_DIR` / `VEDA_AB_TE_DIR`) and asserts `spec.quantize`; `bf16` stages
//! nothing and asserts nothing, so the provider reads the root's own dense `transformer/` and
//! `text_encoder/`. `r2va` never stages `transformer_ref/` — the provider derives it as the staged
//! DiT's sibling, exactly as in the product. Every path must hold REAL files: candle's memmap cannot
//! traverse Windows reparse-point links (see `scripts/ci/validate_h3_vram_staging.py`).
//!
//! The one deliberate difference from the product: for `r2va` at `bf16` the product stages the
//! SceneWorks rehost's `bf16/transformer/`, which is a macOS-only download; here the dense upstream
//! root (whose `transformer_ref/` sibling ships in the same snapshot) is used instead. `result.json`
//! records the directories that were actually staged.

use std::io::Write as _;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::Arc;
use std::time::{Duration, Instant};

use candle_gen::gen_core::{
    AdapterKind, AdapterSpec, AudioTrack, Conditioning, GenerationOutput, GenerationRequest, Image,
    LoadSpec, OffloadPolicy, Precision, Progress, Quant, VedaRequest, WeightsSource,
};
use candle_gen_minimax_h3::DEFAULT_STEPS;
use candle_gen_minimax_h3::MODEL_ID;
use candle_gen_minimax_h3::SMALLEST_LEGAL_FRAMES;
use candle_gen_minimax_h3::{AUDIO_SIGMA_SHIFT, VIDEO_SIGMA_SHIFT};

/// The logical CUDA device candle renders on (`default_device()` pins `cuda:0`).
#[cfg(feature = "cuda")]
const CANDLE_LOGICAL_DEVICE: i32 = 0;

/// Default canvas: 16:9 at the 544 short edge the turbo variants train at, 32-aligned.
const DEFAULT_WIDTH: u32 = 960;
const DEFAULT_HEIGHT: u32 = 544;
const DEFAULT_SEED: u64 = 25_140;
const DEFAULT_PROMPT: &str = "a lighthouse on a rocky coast at dusk, waves breaking against the \
                              rocks, seagulls calling";

/// A sampling recipe: `(steps, video shift, audio shift)` plus which partition it was trained for.
#[derive(Clone, Copy, Debug)]
struct Recipe {
    id: &'static str,
    steps: Option<u32>,
    video_shift: Option<f32>,
    audio_shift: f32,
    reference_partition: bool,
}

/// The turbo recipes, read from SceneWorks `config/manifests/builtin.loras.jsonc`
/// (`feature/sc-25140-minimax-h3-veda`, `sampling` of each `minimax_h3_*turbo*` entry; all four
/// files at `lightx2v/Minimax-h3-Turbo` revision `5d1d4829fe614c1b93fcfd9cc7718e9ba71f73e1`).
/// `base` is the engine's own default regime (`DEFAULT_STEPS`, `VIDEO_SIGMA_SHIFT`), passed as
/// `None` exactly as the product does, so the receipt records `null` rather than a copied constant.
const RECIPES: &[Recipe] = &[
    Recipe {
        id: "base",
        steps: None,
        video_shift: None,
        audio_shift: AUDIO_SIGMA_SHIFT,
        reference_partition: false,
    },
    Recipe {
        id: "minimax_h3_turbo_8step",
        steps: Some(8),
        video_shift: Some(12.0),
        audio_shift: 3.0,
        reference_partition: false,
    },
    Recipe {
        id: "minimax_h3_turbo_4step_768p",
        steps: Some(4),
        video_shift: Some(6.0),
        audio_shift: 3.0,
        reference_partition: false,
    },
    Recipe {
        id: "minimax_h3_turbo_4step_v01",
        steps: Some(4),
        video_shift: Some(12.0),
        audio_shift: 3.0,
        reference_partition: false,
    },
    Recipe {
        id: "minimax_h3_ref2v_turbo_4step",
        steps: Some(4),
        video_shift: Some(12.0),
        audio_shift: 3.0,
        reference_partition: true,
    },
];

fn env(key: &str) -> Option<String> {
    std::env::var(key)
        .ok()
        .map(|v| v.trim().to_owned())
        .filter(|v| !v.is_empty())
}

fn env_parse<T: std::str::FromStr>(key: &str) -> Option<T> {
    env(key).map(|v| {
        v.parse()
            .unwrap_or_else(|_| panic!("{key}={v:?} does not parse"))
    })
}

fn list(key: &str) -> Vec<String> {
    env(key)
        .map(|v| {
            v.split(',')
                .map(|s| s.trim().to_owned())
                .filter(|s| !s.is_empty())
                .collect()
        })
        .unwrap_or_default()
}

/// `path[:strength]`. Split on the LAST colon so a Windows drive letter survives; a suffix that is
/// not a number means the whole entry is the path.
fn parse_lora(entry: &str) -> (PathBuf, f32) {
    match entry.rsplit_once(':') {
        Some((path, strength)) if !path.is_empty() => match strength.trim().parse::<f32>() {
            Ok(s) => (PathBuf::from(path), s),
            Err(_) => (PathBuf::from(entry), 1.0),
        },
        _ => (PathBuf::from(entry), 1.0),
    }
}

fn load_image(path: &str) -> Image {
    let rgb = candle_gen::gen_core::__model_input_reader!(
image::open(path),
"crates/media/candle-gen/candle-gen-minimax-h3/tests/veda_ab.rs::load_image::image::open@7446",
Reference,
path,
|__model_input_path| image::open(__model_input_path)
,
origin = candle_gen::gen_core::model_input_proof::SourceOrigin::ExternalSelected
)
    .unwrap_or_else(|e| panic!("open image {path}: {e}"))
    .to_rgb8();
    Image {
        width: rgb.width(),
        height: rgb.height(),
        pixels: rgb.into_raw(),
    }
}

fn require_dir(path: &Path, what: &str) {
    assert!(
        path.is_dir(),
        "{what}: {} is not a directory",
        path.display()
    );
}

/// Polls this process's resident set on a side thread; `max_rss_bytes` is its sampled maximum.
struct RssSampler {
    max: Arc<AtomicU64>,
    stop: Arc<AtomicBool>,
    thread: Option<std::thread::JoinHandle<()>>,
}

impl RssSampler {
    fn start() -> Self {
        let max = Arc::new(AtomicU64::new(0));
        let stop = Arc::new(AtomicBool::new(false));
        let (m, s) = (max.clone(), stop.clone());
        let thread = std::thread::spawn(move || {
            while !s.load(Ordering::Relaxed) {
                if let Some(bytes) =
                    candle_gen::gen_core::h3_terminal_telemetry::host_process_rss_bytes()
                {
                    m.fetch_max(bytes, Ordering::Relaxed);
                }
                std::thread::sleep(Duration::from_millis(50));
            }
        });
        Self {
            max,
            stop,
            thread: Some(thread),
        }
    }

    fn finish(mut self) -> Option<u64> {
        self.stop.store(true, Ordering::Relaxed);
        if let Some(t) = self.thread.take() {
            let _ = t.join();
        }
        let max = self.max.load(Ordering::Relaxed);
        (max > 0).then_some(max)
    }
}

/// Reset the device pool high-water so the receipt describes THIS render.
#[cfg(feature = "cuda")]
fn reset_device_peak() {
    if let Some(pool) = candle_gen::cuda_mempool::MemPool::device_current(CANDLE_LOGICAL_DEVICE) {
        pool.reset_high_water();
    }
}

#[cfg(not(feature = "cuda"))]
fn reset_device_peak() {}

/// `(device_bytes, detail)`: the CUDA pool's concurrent-live high-water (`USED_MEM_HIGH`) plus the
/// reserved high-water in the detail block; `None` off CUDA.
#[cfg(feature = "cuda")]
fn device_peak() -> (Option<u64>, serde_json::Value) {
    let pool = candle_gen::cuda_mempool::MemPool::device_current(CANDLE_LOGICAL_DEVICE);
    let used_high = pool.as_ref().and_then(|p| p.used_high());
    (
        used_high,
        serde_json::json!({
            "method": "cuda_current_pool_high_water_reset_before_load",
            "usedHighBytes": used_high,
            "reservedHighBytes": pool.as_ref().and_then(|p| p.reserved_high()),
        }),
    )
}

#[cfg(not(feature = "cuda"))]
fn device_peak() -> (Option<u64>, serde_json::Value) {
    (
        None,
        serde_json::json!({ "method": "unavailable_without_cuda" }),
    )
}

/// 16-bit PCM WAV of an interleaved track.
fn write_wav(path: &Path, audio: &AudioTrack) -> std::io::Result<()> {
    let channels = u32::from(audio.channels.max(1));
    let data_len = (audio.samples.len() * 2) as u32;
    let mut f = std::io::BufWriter::new(std::fs::File::create(path)?);
    f.write_all(b"RIFF")?;
    f.write_all(&(36 + data_len).to_le_bytes())?;
    f.write_all(b"WAVEfmt ")?;
    f.write_all(&16u32.to_le_bytes())?;
    f.write_all(&1u16.to_le_bytes())?;
    f.write_all(&(channels as u16).to_le_bytes())?;
    f.write_all(&audio.sample_rate.to_le_bytes())?;
    f.write_all(&(audio.sample_rate * channels * 2).to_le_bytes())?;
    f.write_all(&((channels * 2) as u16).to_le_bytes())?;
    f.write_all(&16u16.to_le_bytes())?;
    f.write_all(b"data")?;
    f.write_all(&data_len.to_le_bytes())?;
    for s in &audio.samples {
        let v = (s.clamp(-1.0, 1.0) * f32::from(i16::MAX)).round() as i16;
        f.write_all(&v.to_le_bytes())?;
    }
    f.flush()
}

fn write_png(path: &Path, frame: &Image) {
    image::RgbImage::from_raw(frame.width, frame.height, frame.pixels.clone())
        .expect("RGB8 frame")
        .save(path)
        .unwrap_or_else(|e| panic!("write {}: {e}", path.display()));
}

/// Mux PNG frames (+ WAV) into `clip.mp4` with ffmpeg. `Ok(false)` when ffmpeg is not installed, so
/// the caller keeps the PNG frames instead.
fn mux_mp4(frames_dir: &Path, wav: Option<&Path>, fps: u32, out: &Path) -> Result<bool, String> {
    let mut cmd = std::process::Command::new("ffmpeg");
    cmd.args(["-nostdin", "-y", "-loglevel", "error", "-framerate"])
        .arg(fps.to_string())
        .arg("-i")
        .arg(frames_dir.join("frame_%05d.png"));
    if let Some(wav) = wav {
        cmd.arg("-i").arg(wav);
    }
    cmd.args(["-c:v", "libx264", "-crf", "12", "-pix_fmt", "yuv420p"]);
    if wav.is_some() {
        cmd.args(["-c:a", "aac", "-b:a", "192k", "-shortest"]);
    }
    cmd.arg(out);
    match cmd.output() {
        Err(e) if e.kind() == std::io::ErrorKind::NotFound => Ok(false),
        Err(e) => Err(format!("spawn ffmpeg: {e}")),
        Ok(o) if o.status.success() => Ok(true),
        Ok(o) => Err(format!(
            "ffmpeg failed ({}): {}",
            o.status,
            String::from_utf8_lossy(&o.stderr)
        )),
    }
}

#[test]
#[ignore = "sc-25777 Veda A/B; one real-weight MiniMax-H3 render per process, configured by VEDA_AB_* (see module docs)"]
fn veda_ab_render() {
    // ---- request ------------------------------------------------------------------------------
    let out: PathBuf = env("VEDA_AB_OUT").expect("VEDA_AB_OUT").into();
    std::fs::create_dir_all(&out).expect("create VEDA_AB_OUT");
    let mode = env("VEDA_AB_MODE").unwrap_or_else(|| "t2va".into());
    let first = env("VEDA_AB_FIRST_IMAGE");
    let last = env("VEDA_AB_LAST_IMAGE");
    let refs = list("VEDA_AB_REF_IMAGES");
    match mode.as_str() {
        "t2va" => assert!(
            first.is_none() && last.is_none() && refs.is_empty(),
            "t2va takes no images"
        ),
        "fl2va" => assert!(
            (first.is_some() || last.is_some()) && refs.is_empty(),
            "fl2va needs VEDA_AB_FIRST_IMAGE and/or VEDA_AB_LAST_IMAGE and no references"
        ),
        "r2va" => assert!(
            !refs.is_empty() && first.is_none() && last.is_none(),
            "r2va needs VEDA_AB_REF_IMAGES and no keyframes"
        ),
        other => panic!("VEDA_AB_MODE={other:?}: expected t2va | fl2va | r2va"),
    }
    let tier = env("VEDA_AB_TIER").unwrap_or_else(|| "q4".into());
    let quant = match tier.as_str() {
        "q4" => Some(Quant::Q4),
        "q8" => Some(Quant::Q8),
        "bf16" => None,
        other => panic!("VEDA_AB_TIER={other:?}: candle MiniMax-H3 serves q4 | q8 | bf16"),
    };
    let recipe_id = env("VEDA_AB_RECIPE").unwrap_or_else(|| "base".into());
    let recipe = *RECIPES
        .iter()
        .find(|r| r.id == recipe_id)
        .unwrap_or_else(|| panic!("VEDA_AB_RECIPE={recipe_id:?} is not a known recipe"));
    if recipe.id != "base" {
        assert_eq!(
            recipe.reference_partition,
            mode == "r2va",
            "{} is trained for the {} partition, not {mode}",
            recipe.id,
            if recipe.reference_partition {
                "transformer_ref (r2va)"
            } else {
                "transformer (t2va/fl2va)"
            }
        );
    }
    // The engine holds audio at its own constant on every path (there is no request field), so a
    // recipe declaring another audio shift could not be honoured — refuse rather than mislabel.
    assert_eq!(
        recipe.audio_shift, AUDIO_SIGMA_SHIFT,
        "recipe {} declares audio shift {} but the engine renders audio at {AUDIO_SIGMA_SHIFT}",
        recipe.id, recipe.audio_shift
    );
    let steps: Option<u32> = env_parse("VEDA_AB_STEPS").or(recipe.steps);
    let width: u32 = env_parse("VEDA_AB_WIDTH").unwrap_or(DEFAULT_WIDTH);
    let height: u32 = env_parse("VEDA_AB_HEIGHT").unwrap_or(DEFAULT_HEIGHT);
    let frames: u32 = env_parse("VEDA_AB_FRAMES").unwrap_or(SMALLEST_LEGAL_FRAMES as u32);
    let seed: u64 = env_parse("VEDA_AB_SEED").unwrap_or(DEFAULT_SEED);
    let prompt = env("VEDA_AB_PROMPT").unwrap_or_else(|| DEFAULT_PROMPT.to_owned());
    let veda_on = match env("VEDA_AB_VEDA").as_deref().unwrap_or("off") {
        "on" => true,
        "off" => false,
        other => panic!("VEDA_AB_VEDA={other:?}: expected off | on"),
    };
    let predictor = env("VEDA_AB_PREDICTOR").map(PathBuf::from);
    let loras: Vec<(PathBuf, f32)> = list("VEDA_AB_LORAS")
        .iter()
        .map(|e| parse_lora(e))
        .collect();

    // ---- load spec (the product's candle wiring) ----------------------------------------------
    let root: PathBuf = env("MINIMAX_H3_SNAPSHOT")
        .expect("MINIMAX_H3_SNAPSHOT must name the staged MiniMax-H3 base root (real files)")
        .into();
    require_dir(&root, "MINIMAX_H3_SNAPSHOT");
    let mut spec = LoadSpec::new(WeightsSource::Dir(root.clone()));
    spec.quantize = quant;
    spec.precision = Precision::Bf16;
    spec.offload_policy = OffloadPolicy::Sequential;
    let (dit_dir, te_dir) = match quant {
        Some(_) => {
            let dit: PathBuf = env("VEDA_AB_DIT_DIR")
                .unwrap_or_else(|| panic!("{tier} needs VEDA_AB_DIT_DIR (the tier's transformer/)"))
                .into();
            let te: PathBuf = env("VEDA_AB_TE_DIR")
                .unwrap_or_else(|| panic!("{tier} needs VEDA_AB_TE_DIR (the tier's text_encoder/)"))
                .into();
            require_dir(&dit, "VEDA_AB_DIT_DIR");
            require_dir(&te, "VEDA_AB_TE_DIR");
            spec.components
                .insert("transformer".into(), WeightsSource::Dir(dit.clone()));
            spec.components
                .insert("text_encoder".into(), WeightsSource::Dir(te.clone()));
            (dit, te)
        }
        None => (root.join("transformer"), root.join("text_encoder")),
    };
    if mode == "r2va" {
        let sibling = dit_dir.parent().map(|p| p.join("transformer_ref"));
        assert!(
            sibling.as_deref().is_some_and(Path::is_dir),
            "r2va denoises on the staged DiT's sibling transformer_ref/, which is missing beside {}",
            dit_dir.display()
        );
    }
    spec.adapters = loras
        .iter()
        .map(|(path, scale)| {
            assert!(path.is_file(), "LoRA {} does not exist", path.display());
            AdapterSpec::new(path.clone(), *scale, AdapterKind::Lora)
        })
        .collect();
    let veda = if veda_on {
        let predictor = predictor
            .clone()
            .expect("VEDA_AB_VEDA=on needs VEDA_AB_PREDICTOR");
        assert!(
            predictor.is_file(),
            "predictor {} does not exist",
            predictor.display()
        );
        spec.components
            .insert("veda_predictor".into(), WeightsSource::File(predictor));
        Some(VedaRequest {
            enabled: true,
            ..Default::default()
        })
    } else {
        None
    };

    let mut conditioning = Vec::new();
    if let Some(p) = &first {
        conditioning.push(Conditioning::Keyframe {
            image: load_image(p),
            frame_idx: 0,
            strength: 1.0,
        });
    }
    if let Some(p) = &last {
        conditioning.push(Conditioning::Keyframe {
            image: load_image(p),
            frame_idx: -1,
            strength: 1.0,
        });
    }
    for p in &refs {
        conditioning.push(Conditioning::Reference {
            image: load_image(p),
            strength: None,
        });
    }
    let req = GenerationRequest {
        prompt: prompt.clone(),
        width,
        height,
        frames: Some(frames),
        steps,
        seed: Some(seed),
        scheduler_shift: recipe.video_shift,
        conditioning,
        veda,
        ..Default::default()
    };

    let config = serde_json::json!({
        "backend": if cfg!(feature = "cuda") { "candle_cuda" } else { "candle_cpu" },
        "model": MODEL_ID,
        "mode": mode,
        "tier": tier,
        "recipe": recipe.id,
        "steps": steps,
        "effectiveSteps": steps.unwrap_or(DEFAULT_STEPS),
        "videoShift": recipe.video_shift,
        "effectiveVideoShift": recipe.video_shift.unwrap_or(VIDEO_SIGMA_SHIFT),
        "audioShift": recipe.audio_shift,
        "width": width,
        "height": height,
        "frames": frames,
        "seed": seed,
        "prompt": prompt,
        "veda": if veda_on { "on" } else { "off" },
        "predictor": if veda_on { predictor.as_ref().map(|p| p.display().to_string()) } else { None },
        "loras": loras.iter().map(|(p, s)| serde_json::json!({"path": p.display().to_string(), "strength": s})).collect::<Vec<_>>(),
        "firstImage": first,
        "lastImage": last,
        "refImages": refs,
        "weights": {
            "root": root.display().to_string(),
            "transformer": dit_dir.display().to_string(),
            "textEncoder": te_dir.display().to_string(),
            "quantizeAsserted": quant.map(|q| format!("{q:?}")),
        },
    });
    eprintln!(
        "[veda-ab] config {}",
        serde_json::to_string(&config).unwrap()
    );

    // ---- render -------------------------------------------------------------------------------
    reset_device_peak();
    let rss = RssSampler::start();
    let started = Instant::now();
    let registry = candle_gen_minimax_h3::provider_registry().expect("minimax-h3 registry");
    let generator = registry
        .load(MODEL_ID, &spec)
        .unwrap_or_else(|e| panic!("load {MODEL_ID} ({tier}): {e}"));
    let provider_load = started.elapsed();

    let gen_started = Instant::now();
    let mut step_ticks: Vec<Duration> = Vec::new();
    let mut decode_tick: Option<Duration> = None;
    let out_video = generator
        .generate(&req, &mut |p| match p {
            Progress::Step { .. } => step_ticks.push(gen_started.elapsed()),
            Progress::Decoding => decode_tick = Some(gen_started.elapsed()),
            Progress::Loading(_)
            | Progress::PromptUpsamplingComplete
            | Progress::ConditioningComplete
            | Progress::Upsampling { .. } => {}
        })
        .unwrap_or_else(|e| panic!("generate {MODEL_ID} ({tier}, veda {veda_on}): {e}"));
    let gen_total = gen_started.elapsed();
    let max_rss = rss.finish();
    let (device_bytes, device_detail) = device_peak();
    let report = generator.veda_execution_report();

    // ---- timings ------------------------------------------------------------------------------
    // Step ticks fire after each evaluation's device sync, so the n-1 inter-tick gaps are whole
    // steps. Step 1 has no preceding tick, so its duration is estimated as the mean and everything
    // before it (conditioning, staging, DiT map) is `preDenoiseS`.
    let n = step_ticks.len();
    assert!(n > 0, "no denoise step was reported");
    let step_mean = if n > 1 {
        (step_ticks[n - 1] - step_ticks[0]).as_secs_f64() / (n - 1) as f64
    } else {
        step_ticks[0].as_secs_f64()
    };
    let denoise_s = (step_ticks[n - 1] - step_ticks[0]).as_secs_f64() + step_mean;
    let pre_denoise_s = (step_ticks[0].as_secs_f64() - step_mean).max(0.0);
    let decode_start = decode_tick.expect("the decode phase must be reported");
    let decode_s = (gen_total - decode_start).as_secs_f64();

    // ---- outputs ------------------------------------------------------------------------------
    let (frames_out, fps, audio) = match out_video {
        GenerationOutput::Video { frames, fps, audio } => (frames, fps, audio),
        other => panic!("{MODEL_ID} must produce Video, got {other:?}"),
    };
    assert_eq!(frames_out.len(), frames as usize, "decoded frame count");
    let frames_dir = out.join("frames");
    std::fs::create_dir_all(&frames_dir).expect("frames dir");
    for (i, f) in frames_out.iter().enumerate() {
        write_png(&frames_dir.join(format!("frame_{i:05}.png")), f);
    }
    write_png(
        &out.join("frame_mid.png"),
        &frames_out[frames_out.len() / 2],
    );
    let wav = audio.as_ref().map(|a| {
        let path = out.join("audio.wav");
        write_wav(&path, a).expect("write audio.wav");
        path
    });
    let mp4 = out.join("clip.mp4");
    let muxed = mux_mp4(&frames_dir, wav.as_deref(), fps, &mp4).expect("mux clip.mp4");
    if muxed {
        // The mp4 is the deliverable; the PNG sequence stays only when ffmpeg is unavailable.
        std::fs::remove_dir_all(&frames_dir).expect("remove PNG frames after mux");
    }

    let degenerate = degeneracy(&frames_out, audio.as_ref().map(|a| a.samples.as_slice()));
    let veda_json = if veda_on {
        let report = report.expect("MiniMax-H3 publishes a Veda execution report");
        serde_json::json!({
            "outcome": report.outcome_json(),
            "total_attention_fraction": report.total_attention_fraction(),
        })
    } else {
        serde_json::Value::Null
    };
    let result = serde_json::json!({
        "config": config,
        "degenerate": degenerate,
        "timings": {
            "load_s": provider_load.as_secs_f64() + pre_denoise_s,
            "denoise_s": denoise_s,
            "decode_s": decode_s,
            "step_mean_s": step_mean,
            "providerLoadS": provider_load.as_secs_f64(),
            "preDenoiseS": pre_denoise_s,
            "generateS": gen_total.as_secs_f64(),
            "stepTicksS": step_ticks.iter().map(Duration::as_secs_f64).collect::<Vec<_>>(),
            "method": "load_s = provider load + generate start until the first step tick less one mean step (conditioning, staging, DiT map); steps from Progress::Step tick gaps; decode_s from Progress::Decoding to generate return",
        },
        "peak": {
            "device_bytes": device_bytes,
            "device": device_detail,
            "max_rss_bytes": max_rss,
            "rssMethod": "current-process resident set polled every 50 ms from before load to generate return",
        },
        "veda": veda_json,
        "output": {
            "frames": frames_out.len(),
            "fps": fps,
            "width": frames_out[0].width,
            "height": frames_out[0].height,
            "mp4": muxed.then_some("clip.mp4"),
            "pngFrames": (!muxed).then_some("frames/frame_%05d.png"),
            "wav": wav.as_ref().map(|_| "audio.wav"),
            "audioSampleRate": audio.as_ref().map(|a| a.sample_rate),
            "audioChannels": audio.as_ref().map(|a| a.channels),
        },
    });
    let text = serde_json::to_string_pretty(&result).unwrap();
    std::fs::write(out.join("result.json"), &text).expect("write result.json");
    println!(
        "[veda-ab] result {}",
        serde_json::to_string(&result).unwrap()
    );
    assert!(
        degenerate.is_empty(),
        "degenerate render (veda {}): {}",
        if veda_on { "on" } else { "off" },
        degenerate.join("; ")
    );

    #[cfg(feature = "model-inference-proof")]
    println!("\nMODEL_INFERENCE_BODY_COMPLETED veda_ab::veda_ab_render");
}

#[test]
fn lora_entries_keep_windows_drive_letters() {
    assert_eq!(
        parse_lora(r"E:\loras\turbo.safetensors:0.8"),
        (PathBuf::from(r"E:\loras\turbo.safetensors"), 0.8)
    );
    assert_eq!(
        parse_lora(r"E:\loras\turbo.safetensors"),
        (PathBuf::from(r"E:\loras\turbo.safetensors"), 1.0)
    );
    assert_eq!(
        parse_lora("/w/clay.safetensors:1.25"),
        (PathBuf::from("/w/clay.safetensors"), 1.25)
    );
    assert_eq!(
        parse_lora("/w/clay.safetensors"),
        (PathBuf::from("/w/clay.safetensors"), 1.0)
    );
}

#[test]
fn every_recipe_is_unique_and_renders_audio_at_the_engine_shift() {
    let mut ids: Vec<&str> = RECIPES.iter().map(|r| r.id).collect();
    ids.sort_unstable();
    ids.dedup();
    assert_eq!(ids.len(), RECIPES.len());
    assert!(RECIPES.iter().all(|r| r.audio_shift == AUDIO_SIGMA_SHIFT));
}

/// Why a decoded render is degenerate — the signature of NaN/Inf poisoning that a VAE decode
/// clamps into a flat (all-black) clip and a silent or non-finite soundtrack. Empty means the render
/// is usable. Checked AFTER the outputs and `result.json` are written, so a failing cell still
/// leaves its evidence behind, but the test fails instead of reporting a poisoned render as passed.
fn degeneracy(frames: &[Image], audio: Option<&[f32]>) -> Vec<String> {
    /// Pixel standard deviation (0–255 scale) below which the whole clip is one flat colour.
    const MIN_CLIP_PIXEL_STD: f64 = 1.0;
    let mut why = Vec::new();
    let (mut n, mut sum, mut sum_sq) = (0f64, 0f64, 0f64);
    for px in frames.iter().flat_map(|f| f.pixels.iter()) {
        let v = f64::from(*px);
        n += 1.0;
        sum += v;
        sum_sq += v * v;
    }
    let std = if n > 0.0 {
        (sum_sq / n - (sum / n).powi(2)).max(0.0).sqrt()
    } else {
        0.0
    };
    if std < MIN_CLIP_PIXEL_STD {
        why.push(format!(
            "decoded clip pixel std {std:.3} < {MIN_CLIP_PIXEL_STD} (flat clip, mean {:.1})",
            if n > 0.0 { sum / n } else { 0.0 }
        ));
    }
    if let Some(samples) = audio {
        let non_finite = samples.iter().filter(|s| !s.is_finite()).count();
        if non_finite > 0 {
            why.push(format!(
                "{non_finite} of {} audio samples are non-finite",
                samples.len()
            ));
        }
        let rms = (samples
            .iter()
            .filter(|s| s.is_finite())
            .map(|s| f64::from(*s).powi(2))
            .sum::<f64>()
            / samples.len().max(1) as f64)
            .sqrt();
        if rms == 0.0 {
            why.push(format!("audio RMS is 0 over {} samples", samples.len()));
        }
    }
    why
}

#[test]
fn degeneracy_flags_flat_clips_and_silent_or_non_finite_audio() {
    let frame = |pixels: Vec<u8>| Image {
        width: 2,
        height: 1,
        pixels,
    };
    let black = vec![frame(vec![0; 6]), frame(vec![0; 6])];
    let live = vec![frame(vec![0, 40, 80, 120, 160, 200]), frame(vec![9; 6])];
    assert!(degeneracy(&live, Some(&[0.1, -0.2, 0.05])).is_empty());
    assert!(degeneracy(&live, None).is_empty());
    assert_eq!(degeneracy(&black, Some(&[0.1, -0.2])).len(), 1);
    assert_eq!(degeneracy(&live, Some(&[0.0; 8])).len(), 1);
    assert_eq!(degeneracy(&live, Some(&[0.1, f32::NAN])).len(), 1);
    assert_eq!(degeneracy(&black, Some(&[f32::NAN; 4])).len(), 3);
}
