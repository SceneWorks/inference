//! Render one short SCAIL-2 character-animation clip from a staged tier directory and report whether
//! any projection went through the saved-quantization cache. NOT part of CI: it needs real weights
//! and a CUDA GPU. Modeled on `candle-gen-minimax-h3/examples/minimax_h3_tier_render.rs`.
//!
//! ```text
//! SCAIL2_TIER_DIR=<flat tier (dit/t5_encoder/vae/clip .safetensors, config.json, tokenizer.json)> \
//! SCAIL2_TIER=q4|q8|bf16 SCAIL2_TIER_OUT=<dir for frames + report> \
//! SCAIL2_TIER_REF=<reference character png> SCAIL2_TIER_REF_MASK=<its color-coded mask png> \
//! SCAIL2_TIER_DRIVING_DIR=<driving pose frames *.png> SCAIL2_TIER_DRIVING_MASK_DIR=<per-frame masks *.png> \
//! [SCAIL2_TIER_FRAMES=9 SCAIL2_TIER_W=384 SCAIL2_TIER_H=640 SCAIL2_TIER_STEPS=<default 40>] \
//!   cargo run --release --features cuda --example scail2_tier_render -p candle-gen-scail2
//! ```
//!
//! Driving frames and masks are read in file-name order and truncated to `SCAIL2_TIER_FRAMES`. A
//! counting `QuantizationArtifactCache` is installed for the whole load + render. A Candle-native
//! (GGML) tier must leave it at **zero** materializations in every phase; an MLX-affine tier routes
//! every packed projection through it. Writes the first, middle and last frames as PNG plus
//! `report.json`.
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Arc;

use candle_gen::gen_core::{
    Conditioning, GenerationOutput, GenerationRequest, Image, LoadSpec, Progress, Quant,
    QuantizationArtifact, QuantizationArtifactCache, QuantizationArtifactKey,
    QuantizationCacheContext, ReplacementMode, VideoMode, WeightsSource,
};

fn env(key: &str) -> Option<String> {
    std::env::var(key).ok().filter(|v| !v.trim().is_empty())
}

fn env_or<T: std::str::FromStr>(key: &str, default: T) -> T {
    env(key).map_or(default, |v| {
        v.parse()
            .unwrap_or_else(|_| panic!("{key}={v:?} does not parse"))
    })
}

fn load_rgb(path: &Path) -> Image {
    let rgb = image::open(path)
        .unwrap_or_else(|e| panic!("open {}: {e}", path.display()))
        .to_rgb8();
    Image {
        width: rgb.width(),
        height: rgb.height(),
        pixels: rgb.into_raw(),
    }
}

/// Every `*.png` in `dir`, in file-name order, truncated to `limit`.
fn load_frames(dir: &Path, limit: usize) -> Vec<Image> {
    let mut paths: Vec<PathBuf> = std::fs::read_dir(dir)
        .unwrap_or_else(|e| panic!("read {}: {e}", dir.display()))
        .filter_map(|entry| entry.ok().map(|entry| entry.path()))
        .filter(|path| path.extension().is_some_and(|ext| ext == "png"))
        .collect();
    paths.sort();
    paths.truncate(limit);
    assert!(!paths.is_empty(), "no .png frames in {}", dir.display());
    paths.iter().map(|path| load_rgb(path)).collect()
}

fn save_png(image: &Image, path: &Path) {
    image::RgbImage::from_raw(image.width, image.height, image.pixels.clone())
        .expect("frame buffer matches its geometry")
        .save(path)
        .unwrap_or_else(|e| panic!("write {}: {e}", path.display()));
}

/// Counts every artifact a load asks the saved-quantization cache for; produced artifacts live in a
/// scratch folder under the run's output dir and are deleted when their lease ends.
struct CountingCache {
    requests: AtomicUsize,
    dir: PathBuf,
}
struct ScratchArtifact(PathBuf);
impl QuantizationArtifact for ScratchArtifact {
    fn path(&self) -> &Path {
        &self.0
    }
}
impl Drop for ScratchArtifact {
    fn drop(&mut self) {
        let _ = std::fs::remove_file(&self.0);
    }
}
struct ScratchDir(PathBuf);
impl Drop for ScratchDir {
    fn drop(&mut self) {
        let _ = std::fs::remove_dir_all(&self.0);
    }
}
impl QuantizationArtifactCache for CountingCache {
    fn materialize(
        &self,
        key: &QuantizationArtifactKey,
        produce: &mut dyn FnMut(&Path) -> candle_gen::gen_core::Result<()>,
    ) -> candle_gen::gen_core::Result<Box<dyn QuantizationArtifact>> {
        let n = self.requests.fetch_add(1, Ordering::SeqCst) + 1;
        eprintln!(
            "[scail2-tier-render] saved-quantization request #{n}: {}.{}",
            key.digest(),
            key.extension()
        );
        std::fs::create_dir_all(&self.dir).map_err(|e| {
            candle_gen::gen_core::Error::Msg(format!("create {}: {e}", self.dir.display()))
        })?;
        let artifact = ScratchArtifact(self.dir.join(format!("{n}.{}", key.extension())));
        produce(artifact.path())?;
        Ok(Box::new(artifact))
    }
}

fn main() {
    let tier_dir = PathBuf::from(env("SCAIL2_TIER_DIR").expect("SCAIL2_TIER_DIR"));
    let out = PathBuf::from(env("SCAIL2_TIER_OUT").expect("SCAIL2_TIER_OUT"));
    let tier = env("SCAIL2_TIER").unwrap_or_else(|| "q4".into());
    let quant = match tier.as_str() {
        "q4" => Some(Quant::Q4),
        "q8" => Some(Quant::Q8),
        "bf16" => None,
        other => panic!("SCAIL2_TIER must be q4, q8 or bf16, got {other:?}"),
    };
    let reference = load_rgb(&PathBuf::from(
        env("SCAIL2_TIER_REF").expect("SCAIL2_TIER_REF"),
    ));
    let reference_mask = load_rgb(&PathBuf::from(
        env("SCAIL2_TIER_REF_MASK").expect("SCAIL2_TIER_REF_MASK"),
    ));
    let frame_limit: usize = env_or("SCAIL2_TIER_FRAMES", 9);
    let driving = load_frames(
        &PathBuf::from(env("SCAIL2_TIER_DRIVING_DIR").expect("SCAIL2_TIER_DRIVING_DIR")),
        frame_limit,
    );
    let driving_masks = load_frames(
        &PathBuf::from(env("SCAIL2_TIER_DRIVING_MASK_DIR").expect("SCAIL2_TIER_DRIVING_MASK_DIR")),
        frame_limit,
    );
    assert_eq!(
        driving.len(),
        driving_masks.len(),
        "one mask per driving frame"
    );
    let width: u32 = env_or("SCAIL2_TIER_W", 384);
    let height: u32 = env_or("SCAIL2_TIER_H", 640);
    let seed: u64 = env_or("SCAIL2_TIER_SEED", 42);
    let steps: Option<u32> =
        env("SCAIL2_TIER_STEPS").map(|s| s.parse().expect("SCAIL2_TIER_STEPS"));
    let prompt = env("SCAIL2_TIER_PROMPT")
        .unwrap_or_else(|| "a character waving its arms, following the driving motion".into());
    std::fs::create_dir_all(&out).expect("create SCAIL2_TIER_OUT");

    let scratch = out.join("saved-quantization");
    let _scratch_guard = ScratchDir(scratch.clone());
    let cache = Arc::new(CountingCache {
        requests: AtomicUsize::new(0),
        dir: scratch.clone(),
    });
    let _scope = QuantizationCacheContext::new(cache.clone()).install();
    let _runtime = candle_gen::quantization_cache::install_runtime_cache_context();
    let mut probe = candle_gen::testkit::VramProbe::start_rendered();

    let frames_in = driving.len();
    eprintln!(
        "[scail2-tier-render] tier={tier} dir={} {width}x{height} frames={frames_in} steps={steps:?} seed={seed}",
        tier_dir.display()
    );
    let started = std::time::Instant::now();
    let load_phase = probe.phase();
    let mut spec = LoadSpec::new(WeightsSource::Dir(tier_dir.clone()));
    if let Some(quant) = quant {
        spec = spec.with_quant(quant);
    }
    let generator = candle_gen_scail2::provider_registry()
        .expect("provider registry")
        .load(candle_gen_scail2::MODEL_ID, &spec)
        .unwrap_or_else(|e| panic!("load: {e}"));
    probe.end_load(load_phase);
    let load_secs = started.elapsed().as_secs_f64();

    // Saved-quantization request count at each progress milestone. SCAIL-2's `load` is lazy, so its
    // components (T5, CLIP, VAE, DiT) materialize inside `generate`; the milestones attribute any
    // request to the phase that asked for it.
    let mut milestones: Vec<(String, usize)> =
        vec![("afterLoad".into(), cache.requests.load(Ordering::SeqCst))];
    let mut first_step_secs: Option<f64> = None;
    let mut steps_total: Option<u32> = None;
    let mut on_progress = |p: Progress| {
        let label = match &p {
            Progress::Step { current: 1, .. } => Some("firstStep".to_string()),
            Progress::Step { .. } => None,
            other => Some(format!("{other:?}")),
        };
        if let Some(label) = label {
            milestones.push((label, cache.requests.load(Ordering::SeqCst)));
        }
        if let Progress::Step { current, total } = p {
            steps_total = Some(total);
            if first_step_secs.is_none() {
                first_step_secs = Some(started.elapsed().as_secs_f64());
            }
            eprint!("\r[scail2-tier-render] step {current}/{total}   ");
        }
    };
    let req = GenerationRequest {
        prompt: prompt.clone(),
        width,
        height,
        count: 1,
        frames: Some(frames_in as u32),
        fps: Some(16),
        seed: Some(seed),
        steps,
        video_mode: Some(VideoMode::Animation),
        conditioning: vec![
            Conditioning::Reference {
                image: reference,
                strength: None,
            },
            Conditioning::Mask {
                image: reference_mask,
            },
            Conditioning::ControlClip {
                frames: driving,
                mask: driving_masks,
                masking_strength: 1.0,
                start_frame: 0,
                mode: ReplacementMode::default(),
            },
        ],
        ..Default::default()
    };
    let gen_phase = probe.phase();
    let output = generator
        .generate(&req, &mut on_progress)
        .unwrap_or_else(|e| panic!("generate: {e}"));
    probe.end_gen(gen_phase);
    let total_secs = started.elapsed().as_secs_f64();
    eprintln!();
    milestones.push((
        "afterGenerate".into(),
        cache.requests.load(Ordering::SeqCst),
    ));
    let GenerationOutput::Video { frames, fps, .. } = output else {
        panic!("SCAIL-2 must produce video");
    };
    assert!(!frames.is_empty(), "SCAIL-2 returned no frames");
    let picks = [0, frames.len() / 2, frames.len() - 1];
    let mut pngs = Vec::new();
    for i in picks {
        let png = out.join(format!("frame_{i:04}.png"));
        save_png(&frames[i], &png);
        pngs.push(png);
    }
    let requests = cache.requests.load(Ordering::SeqCst);
    let vram = probe.report();
    let report = serde_json::json!({
        "id": candle_gen_scail2::MODEL_ID,
        "tier": tier,
        "tierDir": tier_dir,
        "prompt": prompt,
        "width": frames[0].width, "height": frames[0].height,
        "framesIn": frames_in, "framesOut": frames.len(), "fps": fps,
        "steps": steps_total.or(steps),
        "seed": seed,
        "savedQuantizationRequests": requests,
        "requestMilestones": milestones,
        "loadSecs": load_secs,
        "firstStepSecs": first_step_secs,
        "renderSecs": total_secs - load_secs,
        "totalSecs": total_secs,
        "vram": vram.to_string(),
        "savedFrames": picks,
        "pngs": pngs,
    });
    std::fs::write(
        out.join("report.json"),
        serde_json::to_string_pretty(&report).unwrap(),
    )
    .expect("write report.json");
    println!("{report}");
    println!("[scail2-tier-render] savedQuantizationRequests={requests}");
}
