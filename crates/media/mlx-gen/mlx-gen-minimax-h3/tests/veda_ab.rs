//! **Veda dense-vs-sparse A/B** (epic sc-25140 / sc-25777): ONE real-weight MiniMax-H3 render per
//! invocation, configured entirely by env vars, writing the decoded clip plus a measured
//! `result.json`. Run it twice with only `VEDA_AB_VEDA` changed and compare the two output dirs with
//! `scripts/veda_ab_compare.py`.
//!
//! `#[ignore]`d — it needs real weights and Metal. Each cell is its own process on purpose: MLX's
//! peak counter is process-wide, so two renders in one process would share a high-water mark.
//!
//! # Load wiring (mirrors the SceneWorks product, not a test-only path)
//!
//! Replicates `sceneworks-worker` `video_jobs/minimax_h3.rs::resolve_minimax_h3_load` +
//! `wan.rs::video_load_spec` on `feature/sc-25140-minimax-h3-veda`:
//!
//! * `spec.weights` = the upstream base root (`MiniMaxAI/MiniMax-H3`: tokenizer, VAEs, FL2VA docs,
//!   and the dense bf16 text encoder);
//! * `components["transformer"]` = `<tier root>/<tier>/transformer` — always the BASE partition;
//!   the engine resolves `transformer_ref/` as its sibling for r2va;
//! * `components["text_encoder"]` = `<tier root>/<tier>/text_encoder` at q4/q8, absent at bf16
//!   (engine falls back to `<base>/text_encoder`);
//! * `spec.quantize` = the tier's quant (`None` at bf16), `spec.precision` = bf16;
//! * `spec.adapters` = one `AdapterKind::Lora` per `VEDA_AB_LORAS` entry, `scale` = the strength
//!   (the product's `resolve_dense_adapters`; alpha/rank fold is resolved from the file);
//! * `components["veda_predictor"]` = the predictor FILE, staged only when Veda is enabled;
//! * request: `veda`, `steps`, `scheduler_shift` (video only — the engine pins the audio shift),
//!   `fps`, `seed`, conditioning built like `minimax_h3_conditioning` (first keyframe at index 0,
//!   last at `-1`, then reference images in order).
//!
//! # Env vars (shared `VEDA_AB_*` contract — a candle/CUDA twin reads the same names)
//!
//! | var | meaning | default |
//! |---|---|---|
//! | `SCENEWORKS_MLX_MINIMAX_H3_TIER_DIR` | dir holding `q4/` `q8/` `bf16/` (`SceneWorks/minimax-h3-mlx` snapshot) — the product's override name | required |
//! | `SCENEWORKS_MLX_MINIMAX_H3_BASE_DIR` | `MiniMaxAI/MiniMax-H3` snapshot root — the product's override name | required |
//! | `VEDA_AB_OUT` | output dir (created) | required |
//! | `VEDA_AB_MODE` | `t2va` \| `fl2va` \| `r2va` | `t2va` |
//! | `VEDA_AB_FIRST_IMAGE` / `VEDA_AB_LAST_IMAGE` | fl2va keyframes (one or both) | — |
//! | `VEDA_AB_REF_IMAGES` | r2va reference images, comma-separated, in order | — |
//! | `VEDA_AB_TIER` | `q4` \| `q8` \| `bf16` | `q4` |
//! | `VEDA_AB_LORAS` | `path[:strength],...` | none |
//! | `VEDA_AB_RECIPE` | catalog id: `base` \| `minimax_h3_turbo_8step` \| `minimax_h3_turbo_4step_768p` \| `minimax_h3_turbo_4step_v01` \| `minimax_h3_ref2v_turbo_4step` | `base` |
//! | `VEDA_AB_STEPS` | override of the recipe's NFE | recipe |
//! | `VEDA_AB_WIDTH` / `VEDA_AB_HEIGHT` / `VEDA_AB_FRAMES` | geometry (frames on the `17n+5` lattice; fps fixed 24) | 576 / 320 / 124 |
//! | `VEDA_AB_SEED` / `VEDA_AB_PROMPT` | | 17147 / a lighthouse prompt |
//! | `VEDA_AB_VEDA` | `off` (request `veda: None`) \| `on` (`VedaRequest { enabled: true, ..defaults }`) | `off` |
//! | `VEDA_AB_PREDICTOR` | predictor `.safetensors` file, required when on | — |
//!
//! Recipes are `builtin.loras.jsonc` `sampling` on the SceneWorks feature branch: base = engine
//! defaults (50 NFE, video shift 12), `turbo_4step_768p` 4/6, `turbo_8step` 8/12,
//! `turbo_4step_v01` 4/12, `ref2v_turbo_4step` 4/12 (audio shift 3 everywhere, engine-pinned). A
//! turbo recipe REQUIRES its LoRA file in `VEDA_AB_LORAS` — 4 steps without the distilled residual
//! is the undercooked render sc-18729 measured.
//!
//! # Outputs (in `VEDA_AB_OUT`)
//!
//! `frames.rgb` (raw rgb24, `width x height x frames`), `audio.f32le` (interleaved stereo f32,
//! 32 kHz), `video.mp4` (h264 crf 12 + aac, when `ffmpeg` is on PATH), `result.json`:
//! `config`, `timings{load_s, denoise_s, decode_s, step_mean_s, ...}`,
//! `peak{mlx_or_device_bytes, max_rss_bytes, ...}`, `veda{outcome, total_attention_fraction}` or
//! `null` when off. The engine maps TE/DiT/LoRA lazily inside `generate`, so `load_s` is small and
//! `denoise_s` (generate entry -> last step) includes them; `step_mean_s` is over the `steps - 1`
//! intervals between step ticks, so it excludes that setup.
//!
//! # Examples (`--release`: the Veda selector runs host-side loops that debug would penalise)
//!
//! ```sh
//! export SCENEWORKS_MLX_MINIMAX_H3_TIER_DIR=<minimax-h3-mlx snapshot> SCENEWORKS_MLX_MINIMAX_H3_BASE_DIR=<MiniMax-H3 snapshot>
//! T=<lightx2v Minimax-h3-Turbo snapshot>
//! RUN='cargo test --release -p mlx-gen-minimax-h3 --test integration veda_ab::veda_ab_render_one_cell -- --ignored --nocapture'
//! # t2va, turbo 8-step, Veda on
//! VEDA_AB_MODE=t2va VEDA_AB_RECIPE=minimax_h3_turbo_8step VEDA_AB_LORAS=$T/minimax_h3_fl2v_turbo_8step_v1.0_bf16.safetensors \
//!   VEDA_AB_VEDA=on VEDA_AB_PREDICTOR=<predictor.safetensors> VEDA_AB_OUT=/tmp/ab/t2va_t8_on $RUN
//! # fl2va first+last, turbo 4-step 768p
//! VEDA_AB_MODE=fl2va VEDA_AB_FIRST_IMAGE=a.png VEDA_AB_LAST_IMAGE=b.png VEDA_AB_RECIPE=minimax_h3_turbo_4step_768p \
//!   VEDA_AB_LORAS=$T/minimax_h3_fl2v_turbo_4step_v1.0_768p_bf16.safetensors VEDA_AB_OUT=/tmp/ab/fl2va_off $RUN
//! # r2va, ref2v turbo 4-step
//! VEDA_AB_MODE=r2va VEDA_AB_REF_IMAGES=r1.png,r2.png VEDA_AB_RECIPE=minimax_h3_ref2v_turbo_4step \
//!   VEDA_AB_LORAS=$T/minimax_h3_ref2v_turbo_4step_v0.1_bf16.safetensors VEDA_AB_OUT=/tmp/ab/r2va_off $RUN
//! # t2va base 50 NFE
//! VEDA_AB_MODE=t2va VEDA_AB_OUT=/tmp/ab/t2va_base_off $RUN
//! ```

use std::path::{Path, PathBuf};
use std::time::Instant;

use mlx_gen::gen_core::runtime::{AdapterKind, AdapterSpec};
use mlx_gen::gen_core::{
    CancelFlag, Conditioning, GenerationOutput, GenerationRequest, Image, LoadSpec, Precision,
    Progress, Quant, VedaRequest, WeightsSource,
};
use mlx_gen_minimax_h3::model::{load, DIT_COMPONENT, TEXT_ENCODER_COMPONENT};
use mlx_gen_minimax_h3::SMALLEST_LEGAL_FRAMES;
use serde_json::json;

/// The predictor component key the engine reads (`veda::contract::COMPONENT`).
const VEDA_PREDICTOR_COMPONENT: &str = "veda_predictor";

fn env(key: &str) -> Option<String> {
    std::env::var(key)
        .ok()
        .map(|v| v.trim().to_owned())
        .filter(|v| !v.is_empty())
}

fn env_req(key: &str) -> String {
    env(key).unwrap_or_else(|| panic!("{key} is required; see the module docs"))
}

fn env_parse<T: std::str::FromStr>(key: &str) -> Option<T>
where
    T::Err: std::fmt::Display,
{
    env(key).map(|v| {
        v.parse()
            .unwrap_or_else(|e| panic!("{key}={v} does not parse: {e}"))
    })
}

/// `(catalog id, steps, video shift, required LoRA file name)` — `builtin.loras.jsonc` `sampling`.
fn recipe(name: &str) -> (Option<u32>, Option<f32>, Option<&'static str>) {
    match name {
        // Base regime: `None` ⇒ the engine's DEFAULT_STEPS (50) and VIDEO_SIGMA_SHIFT (12.0),
        // exactly what the product passes when no turbo LoRA is selected.
        "base" => (None, None, None),
        "minimax_h3_turbo_4step_768p" => (
            Some(4),
            Some(6.0),
            Some("minimax_h3_fl2v_turbo_4step_v1.0_768p_bf16.safetensors"),
        ),
        "minimax_h3_turbo_8step" => (
            Some(8),
            Some(12.0),
            Some("minimax_h3_fl2v_turbo_8step_v1.0_bf16.safetensors"),
        ),
        "minimax_h3_turbo_4step_v01" => (
            Some(4),
            Some(12.0),
            Some("minimax_h3_fl2v_turbo_4step_v0.1.safetensors"),
        ),
        "minimax_h3_ref2v_turbo_4step" => (
            Some(4),
            Some(12.0),
            Some("minimax_h3_ref2v_turbo_4step_v0.1_bf16.safetensors"),
        ),
        other => panic!("VEDA_AB_RECIPE={other} is not a known recipe"),
    }
}

fn load_rgb(path: &str) -> Image {
    let rgb = image::open(path)
        .unwrap_or_else(|e| panic!("open image {path}: {e}"))
        .to_rgb8();
    Image {
        width: rgb.width(),
        height: rgb.height(),
        pixels: rgb.into_raw(),
    }
}

fn parse_loras(raw: Option<String>) -> Vec<AdapterSpec> {
    let Some(raw) = raw else {
        return Vec::new();
    };
    raw.split(',')
        .map(str::trim)
        .filter(|s| !s.is_empty())
        .map(|entry| {
            let (path, scale) = match entry.rsplit_once(':') {
                Some((p, s)) => (
                    p,
                    s.parse::<f32>()
                        .unwrap_or_else(|e| panic!("LoRA strength in {entry}: {e}")),
                ),
                None => (entry, 1.0),
            };
            let path = PathBuf::from(path);
            assert!(path.is_file(), "LoRA {} does not exist", path.display());
            AdapterSpec::new(path, scale, AdapterKind::Lora)
        })
        .collect()
}

/// `getrusage(RUSAGE_SELF).ru_maxrss` — bytes on macOS. `struct rusage` is two `timeval`s then
/// fourteen `long`s.
fn max_rss_bytes() -> u64 {
    extern "C" {
        fn getrusage(who: i32, usage: *mut i64) -> i32;
    }
    let mut usage = [0i64; 18];
    // SAFETY: `usage` is exactly `sizeof(struct rusage)` on 64-bit Darwin and outlives the call.
    let rc = unsafe { getrusage(0, usage.as_mut_ptr()) };
    assert_eq!(rc, 0, "getrusage failed");
    usage[4] as u64
}

/// `ri_lifetime_max_phys_footprint` (`rusage_info_v4`) — what macOS charged this process at its
/// peak, Metal buffers included (the same read `mlx-llm`'s `tests/common/footprint.rs` uses).
fn lifetime_peak_footprint_bytes() -> u64 {
    extern "C" {
        fn proc_pid_rusage(pid: i32, flavor: i32, buffer: *mut u64) -> i32;
    }
    let mut info = [0u64; 64];
    // SAFETY: `info` outlives the call and is larger than `rusage_info_v4`.
    let rc = unsafe { proc_pid_rusage(std::process::id() as i32, 4, info.as_mut_ptr()) };
    assert_eq!(rc, 0, "proc_pid_rusage failed");
    info[30]
}

fn write_outputs(out: &Path, frames: &[Image], audio: &[f32], fps: u32, w: u32, h: u32) -> bool {
    let mut rgb = Vec::with_capacity(frames.len() * (w * h * 3) as usize);
    for f in frames {
        rgb.extend_from_slice(&f.pixels);
    }
    std::fs::write(out.join("frames.rgb"), rgb).expect("write frames.rgb");
    let pcm: Vec<u8> = audio.iter().flat_map(|s| s.to_le_bytes()).collect();
    std::fs::write(out.join("audio.f32le"), pcm).expect("write audio.f32le");
    let status = std::process::Command::new("ffmpeg")
        .args([
            "-y",
            "-loglevel",
            "error",
            "-f",
            "rawvideo",
            "-pix_fmt",
            "rgb24",
        ])
        .args(["-s", &format!("{w}x{h}"), "-r", &fps.to_string(), "-i"])
        .arg(out.join("frames.rgb"))
        .args(["-f", "f32le", "-ar", "32000", "-ac", "2", "-i"])
        .arg(out.join("audio.f32le"))
        .args(["-c:v", "libx264", "-crf", "12", "-pix_fmt", "yuv420p"])
        .args(["-c:a", "aac", "-b:a", "256k", "-shortest"])
        .arg(out.join("video.mp4"))
        .status();
    match status {
        Ok(s) if s.success() => true,
        other => {
            eprintln!("[veda_ab] ffmpeg mux skipped/failed ({other:?}); raw outputs kept");
            false
        }
    }
}

/// **One A/B cell.** See the module docs for every knob.
#[test]
#[ignore = "sc-25777: real-weight MiniMax-H3 render + Metal; one cell per invocation, env-configured"]
fn veda_ab_render_one_cell() {
    let tier_root = PathBuf::from(env_req("SCENEWORKS_MLX_MINIMAX_H3_TIER_DIR"));
    let base_root = PathBuf::from(env_req("SCENEWORKS_MLX_MINIMAX_H3_BASE_DIR"));
    let out = PathBuf::from(env_req("VEDA_AB_OUT"));
    std::fs::create_dir_all(&out).expect("create VEDA_AB_OUT");

    let tier = env("VEDA_AB_TIER").unwrap_or_else(|| "q4".into());
    let quant = match tier.as_str() {
        "q4" => Some(Quant::Q4),
        "q8" => Some(Quant::Q8),
        "bf16" => None,
        other => panic!("VEDA_AB_TIER={other}: expected q4 | q8 | bf16"),
    };
    let mode = env("VEDA_AB_MODE").unwrap_or_else(|| "t2va".into());
    let recipe_name = env("VEDA_AB_RECIPE").unwrap_or_else(|| "base".into());
    let (recipe_steps, recipe_shift, recipe_lora) = recipe(&recipe_name);
    let steps = env_parse::<u32>("VEDA_AB_STEPS").or(recipe_steps);
    let shift = recipe_shift;
    let width = env_parse("VEDA_AB_WIDTH").unwrap_or(576u32);
    let height = env_parse("VEDA_AB_HEIGHT").unwrap_or(320u32);
    let frames = env_parse("VEDA_AB_FRAMES").unwrap_or(SMALLEST_LEGAL_FRAMES as u32);
    let fps = 24u32;
    let seed = env_parse("VEDA_AB_SEED").unwrap_or(17_147u64);
    let prompt = env("VEDA_AB_PROMPT").unwrap_or_else(|| {
        "a lighthouse on a rocky coast at dusk, waves breaking against the rocks, seagulls calling"
            .into()
    });

    let adapters = parse_loras(env("VEDA_AB_LORAS"));
    if let Some(file) = recipe_lora {
        assert!(
            adapters
                .iter()
                .any(|a| a.path.file_name().is_some_and(|n| n == file)),
            "VEDA_AB_RECIPE={recipe_name} needs {file} in VEDA_AB_LORAS"
        );
    }

    // --- conditioning: the product's `minimax_h3_conditioning` order -------------------------
    let first = env("VEDA_AB_FIRST_IMAGE");
    let last = env("VEDA_AB_LAST_IMAGE");
    let refs: Vec<String> = env("VEDA_AB_REF_IMAGES")
        .map(|r| {
            r.split(',')
                .map(|s| s.trim().to_owned())
                .filter(|s| !s.is_empty())
                .collect()
        })
        .unwrap_or_default();
    let mut conditioning = Vec::new();
    match mode.as_str() {
        "t2va" => assert!(
            first.is_none() && last.is_none() && refs.is_empty(),
            "t2va takes no images"
        ),
        "fl2va" => {
            assert!(
                first.is_some() || last.is_some(),
                "fl2va needs VEDA_AB_FIRST_IMAGE and/or VEDA_AB_LAST_IMAGE"
            );
            assert!(refs.is_empty(), "fl2va takes no references");
            if let Some(p) = &first {
                conditioning.push(Conditioning::Keyframe {
                    image: load_rgb(p),
                    frame_idx: 0,
                    strength: 1.0,
                });
            }
            if let Some(p) = &last {
                conditioning.push(Conditioning::Keyframe {
                    image: load_rgb(p),
                    frame_idx: -1,
                    strength: 1.0,
                });
            }
        }
        "r2va" => {
            assert!(!refs.is_empty(), "r2va needs VEDA_AB_REF_IMAGES");
            assert!(first.is_none() && last.is_none(), "r2va takes no keyframes");
            for p in &refs {
                conditioning.push(Conditioning::Reference {
                    image: load_rgb(p),
                    strength: None,
                });
            }
        }
        other => panic!("VEDA_AB_MODE={other}: expected t2va | fl2va | r2va"),
    }

    // --- veda ---------------------------------------------------------------------------------
    let veda_on = match env("VEDA_AB_VEDA").as_deref().unwrap_or("off") {
        "on" => true,
        "off" => false,
        other => panic!("VEDA_AB_VEDA={other}: expected on | off"),
    };
    let effective_steps = steps.unwrap_or(mlx_gen_minimax_h3::model::DEFAULT_STEPS) as usize;
    let veda = veda_on.then(|| VedaRequest {
        enabled: true,
        ..VedaRequest::default()
    });
    let predictor = veda_on.then(|| {
        let p = PathBuf::from(env_req("VEDA_AB_PREDICTOR"));
        assert!(p.is_file(), "predictor {} does not exist", p.display());
        p
    });

    // --- the LoadSpec: `resolve_minimax_h3_load` + `video_load_spec` -------------------------
    let tier_dir = tier_root.join(&tier);
    let mut spec = LoadSpec::new(WeightsSource::Dir(base_root.clone()))
        .with_component(
            DIT_COMPONENT,
            WeightsSource::Dir(tier_dir.join(DIT_COMPONENT)),
        )
        .with_adapters(adapters.clone());
    if tier != "bf16" {
        spec = spec.with_component(
            TEXT_ENCODER_COMPONENT,
            WeightsSource::Dir(tier_dir.join(TEXT_ENCODER_COMPONENT)),
        );
    }
    if let Some(p) = &predictor {
        spec = spec.with_component(VEDA_PREDICTOR_COMPONENT, WeightsSource::File(p.clone()));
    }
    spec.quantize = quant;
    spec.precision = Precision::Bf16;

    let config = json!({
        "mode": mode, "tier": tier, "tierRoot": tier_root, "baseRoot": base_root,
        "recipe": recipe_name, "steps": steps, "effectiveSteps": effective_steps,
        "videoShift": shift, "audioShift": "engine AUDIO_SIGMA_SHIFT (3.0)",
        "width": width, "height": height, "frames": frames, "fps": fps, "seed": seed,
        "prompt": prompt, "firstImage": first, "lastImage": last, "referenceImages": refs,
        "loras": adapters.iter().map(|a| json!({"path": a.path, "strength": a.scale})).collect::<Vec<_>>(),
        "veda": veda.as_ref().map(|v| format!("{v:?}")),
        "vedaPredictor": predictor,
        "components": spec.components.iter().map(|(k, v)| (k.clone(), json!(format!("{v:?}")))).collect::<serde_json::Map<_, _>>(),
    });
    println!(
        "[veda_ab] config {}",
        serde_json::to_string_pretty(&config).unwrap()
    );

    let t_load = Instant::now();
    let model = load(&spec).unwrap_or_else(|e| panic!("load: {e}"));
    let load_s = t_load.elapsed().as_secs_f64();

    let req = GenerationRequest {
        veda,
        prompt,
        width,
        height,
        frames: Some(frames),
        fps: Some(fps),
        steps,
        scheduler_shift: shift,
        seed: Some(seed),
        conditioning,
        cancel: CancelFlag::default(),
        ..Default::default()
    };
    model.validate(&req).expect("the request must validate");

    mlx_rs::memory::clear_cache();
    mlx_rs::memory::reset_peak_memory();
    let started = Instant::now();
    let mut ticks: Vec<Instant> = Vec::new();
    let mut decoding_at: Option<Instant> = None;
    let output = model
        .generate(&req, &mut |p| match p {
            Progress::Step { current, total } => {
                let now = Instant::now();
                let since = ticks.last().copied().unwrap_or(started);
                println!(
                    "[veda_ab] step {current}/{total} (+{:.1} s)",
                    now.duration_since(since).as_secs_f64()
                );
                ticks.push(now);
            }
            Progress::Decoding => decoding_at = Some(Instant::now()),
            Progress::Loading(_) | Progress::ConditioningComplete | Progress::Upsampling { .. } => {
            }
        })
        .unwrap_or_else(|e| panic!("generate: {e}"));
    let wall_s = started.elapsed().as_secs_f64();
    let mlx_peak = mlx_rs::memory::get_peak_memory() as u64;

    let (frames_out, fps_out, audio) = match output {
        GenerationOutput::Video { frames, fps, audio } => (frames, fps, audio),
        other => panic!("expected a Video output, got {other:?}"),
    };
    let audio = audio.expect("MiniMax-H3 always produces a soundtrack");
    assert!(!ticks.is_empty(), "the denoise reported no steps");

    let first_tick = ticks[0];
    let last_tick = *ticks.last().unwrap();
    let intervals: Vec<f64> = ticks
        .windows(2)
        .map(|w| w[1].duration_since(w[0]).as_secs_f64())
        .collect();
    let decode_s = decoding_at.map(|d| wall_s - d.duration_since(started).as_secs_f64());
    let report = model.veda_execution_report();

    let (out_w, out_h) = (frames_out[0].width, frames_out[0].height);
    let mp4 = write_outputs(&out, &frames_out, &audio.samples, fps_out, out_w, out_h);
    let degenerate = degeneracy(&frames_out, Some(&audio.samples));

    let result = json!({
        "config": config,
        "degenerate": degenerate,
        "output": {
            "frames": frames_out.len(), "width": out_w, "height": out_h, "fps": fps_out,
            "audioSampleRate": audio.sample_rate, "audioChannels": audio.channels,
            "audioSamples": audio.samples.len(), "mp4": mp4,
        },
        "timings": {
            "load_s": load_s,
            "denoise_s": last_tick.duration_since(started).as_secs_f64(),
            "decode_s": decode_s,
            "step_mean_s": (!intervals.is_empty())
                .then(|| intervals.iter().sum::<f64>() / intervals.len() as f64),
            "setup_and_first_eval_s": first_tick.duration_since(started).as_secs_f64(),
            "step_intervals_s": intervals,
            "generate_wall_s": wall_s,
        },
        "peak": {
            "mlx_or_device_bytes": mlx_peak,
            "max_rss_bytes": max_rss_bytes(),
            "lifetime_phys_footprint_bytes": lifetime_peak_footprint_bytes(),
        },
        // `null` when Veda was not requested: the dense arm carries no sparse outcome.
        "veda": report.as_ref().filter(|_| veda_on).map(|r| json!({
            "outcome": r.outcome_json(),
            "total_attention_fraction": r.total_attention_fraction(),
        })),
    });
    let text = serde_json::to_string_pretty(&result).unwrap();
    std::fs::write(out.join("result.json"), &text).expect("write result.json");
    println!("[veda_ab] result {text}");
    assert!(
        degenerate.is_empty(),
        "degenerate render (veda {}): {}",
        if veda_on { "on" } else { "off" },
        degenerate.join("; ")
    );
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
