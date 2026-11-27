//! Render one MiniMax-H3 clip from a staged pre-quantized tier and report whether any projection
//! went through the saved-quantization cache. NOT part of CI: it needs real weights and a CUDA GPU.
//!
//! ```text
//! MINIMAX_H3_ROOT=<upstream MiniMaxAI/MiniMax-H3 snapshot root (vae/, audio_vae/, tokenizer/, FL2VA/)> \
//! MINIMAX_H3_DIT_DIR=<tier>/transformer MINIMAX_H3_TE_DIR=<tier>/text_encoder MINIMAX_H3_TIER=q4 \
//! MINIMAX_H3_OUT=<dir for frames + report> \
//!   cargo run --release --features cuda --example minimax_h3_tier_render -p candle-gen-minimax-h3
//! ```
//!
//! A counting `QuantizationArtifactCache` is installed for the whole load + render. A Candle-native
//! (GGML) tier must leave it at **zero** materializations; an MLX-affine tier routes every packed
//! projection through it. Writes the first, middle and last frames as PNG plus `report.json`.
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Arc;

use candle_gen::gen_core::{
    GenerationOutput, GenerationRequest, LoadSpec, Progress, Quant, QuantizationArtifact,
    QuantizationArtifactCache, QuantizationArtifactKey, QuantizationCacheContext, WeightsSource,
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

/// Counts every artifact a load asks the saved-quantization cache for. Requested artifacts are
/// produced under `dir` (a `saved-quantization/` folder inside the run's own `MINIMAX_H3_OUT`) so an
/// MLX-tier control run still completes; the count is the evidence, and every produced file is
/// disposable — see [`ScratchArtifact`].
struct CountingCache {
    requests: AtomicUsize,
    dir: PathBuf,
}

/// One produced control-run artifact, deleted when its lease ends. An MLX-tier load requests one
/// per packed projection (tens of GB in total), so keeping them would fill the run's storage with
/// copies nothing reads again.
struct ScratchArtifact(PathBuf);
impl QuantizationArtifact for ScratchArtifact {
    fn path(&self) -> &Path {
        &self.0
    }
}
/// The control run's artifact folder, removed when the run ends however it ends.
struct ScratchDir(PathBuf);
impl Drop for ScratchDir {
    fn drop(&mut self) {
        let _ = std::fs::remove_dir_all(&self.0);
    }
}

impl Drop for ScratchArtifact {
    fn drop(&mut self) {
        // A consumer still mapping the file (Windows refuses the delete) leaves it for the
        // end-of-run sweep of `dir` in `main`.
        let _ = std::fs::remove_file(&self.0);
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
            "[h3-tier-render] saved-quantization request #{n}: {}.{}",
            key.digest(),
            key.extension()
        );
        std::fs::create_dir_all(&self.dir).map_err(|e| {
            candle_gen::gen_core::Error::Msg(format!("create {}: {e}", self.dir.display()))
        })?;
        // Owned from before `produce` runs, so a failed or partial write is removed too.
        let artifact = ScratchArtifact(self.dir.join(format!("{n}.{}", key.extension())));
        produce(artifact.path())?;
        Ok(Box::new(artifact))
    }
}

fn save_png(img: &candle_gen::gen_core::Image, path: &Path) {
    image::RgbImage::from_raw(img.width, img.height, img.pixels.clone())
        .expect("frame buffer matches its geometry")
        .save(path)
        .unwrap_or_else(|e| panic!("write {}: {e}", path.display()));
}

fn main() {
    let root = PathBuf::from(env("MINIMAX_H3_ROOT").expect("MINIMAX_H3_ROOT"));
    let dit = PathBuf::from(env("MINIMAX_H3_DIT_DIR").expect("MINIMAX_H3_DIT_DIR"));
    let te = PathBuf::from(env("MINIMAX_H3_TE_DIR").expect("MINIMAX_H3_TE_DIR"));
    let out = PathBuf::from(env("MINIMAX_H3_OUT").expect("MINIMAX_H3_OUT"));
    let tier = env("MINIMAX_H3_TIER").unwrap_or_else(|| "q4".into());
    let quant = match tier.as_str() {
        "q4" => Quant::Q4,
        "q8" => Quant::Q8,
        other => panic!("MINIMAX_H3_TIER must be q4 or q8, got {other:?}"),
    };
    std::fs::create_dir_all(&out).expect("create MINIMAX_H3_OUT");

    let mut spec = LoadSpec::new(WeightsSource::Dir(root.clone())).with_quant(quant);
    spec.components
        .insert("transformer".into(), WeightsSource::Dir(dit.clone()));
    spec.components
        .insert("text_encoder".into(), WeightsSource::Dir(te.clone()));
    let req = GenerationRequest {
        prompt: env("MINIMAX_H3_PROMPT").unwrap_or_else(|| {
            "a red fox trotting through fresh snow in a birch forest, soft morning light".into()
        }),
        width: env_or("MINIMAX_H3_W", 576),
        height: env_or("MINIMAX_H3_H", 320),
        frames: Some(env_or("MINIMAX_H3_FRAMES", 124)),
        steps: Some(env_or("MINIMAX_H3_STEPS", 20)),
        seed: Some(env_or("MINIMAX_H3_SEED", 7)),
        ..Default::default()
    };

    let scratch = out.join("saved-quantization");
    // Declared before the generator, so it drops after it (and on a panic): whatever a still-mapped
    // artifact left behind goes with the run.
    let _scratch_guard = ScratchDir(scratch.clone());
    let cache = Arc::new(CountingCache {
        requests: AtomicUsize::new(0),
        dir: scratch.clone(),
    });
    let _scope = QuantizationCacheContext::new(cache.clone()).install();
    let _runtime = candle_gen::quantization_cache::install_runtime_cache_context();

    eprintln!(
        "[h3-tier-render] tier={tier} root={} dit={} te={} {}x{} frames={:?} steps={:?}",
        root.display(),
        dit.display(),
        te.display(),
        req.width,
        req.height,
        req.frames,
        req.steps
    );
    let started = std::time::Instant::now();
    let registry = candle_gen_minimax_h3::provider_registry().expect("provider registry");
    let generator = registry
        .load(candle_gen_minimax_h3::MODEL_ID, &spec)
        .unwrap_or_else(|e| panic!("load: {e}"));
    let load_secs = started.elapsed().as_secs_f64();
    let mut first_step_secs = None;
    let mut on_progress = |p: Progress| {
        if first_step_secs.is_none() && matches!(p, Progress::Step { .. }) {
            first_step_secs = Some(started.elapsed().as_secs_f64());
            eprintln!(
                "[h3-tier-render] first denoise step at {:.1}s (load + conditioning + weights)",
                first_step_secs.unwrap()
            );
        }
    };
    let output = generator
        .generate(&req, &mut on_progress)
        .unwrap_or_else(|e| panic!("generate: {e}"));
    let total_secs = started.elapsed().as_secs_f64();
    let GenerationOutput::Video { frames, fps, audio } = output else {
        panic!("MiniMax-H3 must produce video");
    };
    let picks = [0, frames.len() / 2, frames.len() - 1];
    for i in picks {
        save_png(&frames[i], &out.join(format!("frame_{i:04}.png")));
    }
    let requests = cache.requests.load(Ordering::SeqCst);
    let report = serde_json::json!({
        "tier": tier,
        "ditDir": dit,
        "textEncoderDir": te,
        "width": req.width, "height": req.height, "frames": frames.len(), "fps": fps,
        "steps": req.steps,
        "audioSamples": audio.as_ref().map(|a| a.samples.len()),
        "savedQuantizationRequests": requests,
        "loadSecs": load_secs,
        "firstStepSecs": first_step_secs,
        "totalSecs": total_secs,
        "savedFrames": picks,
    });
    std::fs::write(
        out.join("report.json"),
        serde_json::to_string_pretty(&report).unwrap(),
    )
    .expect("write report.json");
    println!("{report}");
    println!("[h3-tier-render] savedQuantizationRequests={requests}");
}
