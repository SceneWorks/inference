//! Render one Qwen-Image (txt2img) or Qwen-Image-Edit image from a staged tier directory and report
//! whether any projection went through the saved-quantization cache. NOT part of CI: it needs real
//! weights and a CUDA GPU. Modeled on `candle-gen-minimax-h3/examples/minimax_h3_tier_render.rs`.
//!
//! ```text
//! QWEN_TIER_DIR=<tier root (transformer/, text_encoder/, vae/, tokenizer/, ...)> QWEN_TIER=q4 \
//! QWEN_TIER_OUT=<dir for image + report> [QWEN_TIER_MODE=txt2img|edit QWEN_TIER_INPUT=<png>] \
//!   cargo run --release --features cuda --example qwen_image_tier_render -p candle-gen-qwen-image
//! ```
//!
//! A counting `QuantizationArtifactCache` is installed for the whole load + render. A Candle-native
//! (GGML) tier must leave it at **zero** materializations; an MLX-affine tier routes every packed
//! projection through it. Writes `image.png` plus `report.json`.
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Arc;

use candle_gen::gen_core::{
    GenerationOutput, GenerationRequest, Image, LoadSpec, OffloadPolicy, Progress,
    QuantizationArtifact, QuantizationArtifactCache, QuantizationArtifactKey,
    QuantizationCacheContext, WeightsSource,
};
use candle_gen_qwen_image::{QwenEdit, QwenEditPaths, QwenEditRequest};

fn env(key: &str) -> Option<String> {
    std::env::var(key).ok().filter(|v| !v.trim().is_empty())
}

fn env_or<T: std::str::FromStr>(key: &str, default: T) -> T {
    env(key).map_or(default, |v| {
        v.parse()
            .unwrap_or_else(|_| panic!("{key}={v:?} does not parse"))
    })
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
            "[qwen-tier-render] saved-quantization request #{n}: {}.{}",
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
    let tier_dir = PathBuf::from(env("QWEN_TIER_DIR").expect("QWEN_TIER_DIR"));
    let out = PathBuf::from(env("QWEN_TIER_OUT").expect("QWEN_TIER_OUT"));
    let tier = env("QWEN_TIER").unwrap_or_else(|| "q4".into());
    let mode = env("QWEN_TIER_MODE").unwrap_or_else(|| "txt2img".into());
    let width: u32 = env_or("QWEN_TIER_W", 1024);
    let height: u32 = env_or("QWEN_TIER_H", 1024);
    let seed: u64 = env_or("QWEN_TIER_SEED", 42);
    let steps: Option<u32> = env("QWEN_TIER_STEPS").map(|s| s.parse().expect("QWEN_TIER_STEPS"));
    let gpu: usize = env_or("QWEN_TIER_VRAM_GPU", 0);
    std::fs::create_dir_all(&out).expect("create QWEN_TIER_OUT");

    let scratch = out.join("saved-quantization");
    let _scratch_guard = ScratchDir(scratch.clone());
    let cache = Arc::new(CountingCache {
        requests: AtomicUsize::new(0),
        dir: scratch.clone(),
    });
    let _scope = QuantizationCacheContext::new(cache.clone()).install();
    let _runtime = candle_gen::quantization_cache::install_runtime_cache_context();
    let mut probe = candle_gen::testkit::VramProbe::start(gpu);

    let started = std::time::Instant::now();
    let mut first_step_secs: Option<f64> = None;
    let mut steps_total: Option<u32> = None;
    let mut on_progress = |p: Progress| {
        if let Progress::Step { current, total } = p {
            steps_total = Some(total);
            if first_step_secs.is_none() {
                first_step_secs = Some(started.elapsed().as_secs_f64());
            }
            eprint!("\r[qwen-tier-render] step {current}/{total}   ");
        }
    };

    let (image, prompt, load_secs, effective_steps) = match mode.as_str() {
        "txt2img" => {
            let prompt = env("QWEN_TIER_PROMPT")
                .unwrap_or_else(|| "a red fox sitting in a snowy forest, photorealistic".into());
            eprintln!(
                "[qwen-tier-render] mode=txt2img tier={tier} dir={} {width}x{height} steps={steps:?} seed={seed}",
                tier_dir.display()
            );
            let load_phase = probe.phase();
            let spec = LoadSpec::new(WeightsSource::Dir(tier_dir.clone()));
            let generator = candle_gen_qwen_image::provider_registry()
                .expect("provider registry")
                .load("qwen_image", &spec)
                .unwrap_or_else(|e| panic!("load: {e}"));
            probe.end_load(load_phase);
            let load_secs = started.elapsed().as_secs_f64();
            let req = GenerationRequest {
                prompt: prompt.clone(),
                width,
                height,
                count: 1,
                seed: Some(seed),
                steps,
                ..Default::default()
            };
            let gen_phase = probe.phase();
            let output = generator
                .generate(&req, &mut on_progress)
                .unwrap_or_else(|e| panic!("generate: {e}"));
            probe.end_gen(gen_phase);
            let GenerationOutput::Images(mut images) = output else {
                panic!("qwen_image must produce images");
            };
            (images.remove(0), prompt, load_secs, steps)
        }
        "edit" => {
            let prompt = env("QWEN_TIER_PROMPT").unwrap_or_else(|| "make the sky purple".into());
            let input = PathBuf::from(env("QWEN_TIER_INPUT").expect("QWEN_TIER_INPUT"));
            let rgb = image::open(&input)
                .unwrap_or_else(|e| panic!("open {}: {e}", input.display()))
                .to_rgb8();
            let reference = Image {
                width: rgb.width(),
                height: rgb.height(),
                pixels: rgb.into_raw(),
            };
            eprintln!(
                "[qwen-tier-render] mode=edit tier={tier} dir={} input={} {width}x{height} steps={steps:?} seed={seed}",
                tier_dir.display(),
                input.display()
            );
            let load_phase = probe.phase();
            let model = QwenEdit::load(&QwenEditPaths {
                root: tier_dir.clone(),
                text_encoder: None,
                adapters: Vec::new(),
                offload_policy: OffloadPolicy::Resident,
            })
            .unwrap_or_else(|e| panic!("load: {e}"));
            probe.end_load(load_phase);
            let load_secs = started.elapsed().as_secs_f64();
            let mut req = QwenEditRequest {
                prompt: prompt.clone(),
                width,
                height,
                seed,
                ..Default::default()
            };
            if let Some(s) = steps {
                req.steps = s as usize;
            }
            let effective = Some(req.steps as u32);
            let gen_phase = probe.phase();
            let image = model
                .generate(&req, &[reference], &mut on_progress)
                .unwrap_or_else(|e| panic!("generate: {e}"));
            probe.end_gen(gen_phase);
            (image, prompt, load_secs, effective)
        }
        other => panic!("QWEN_TIER_MODE must be txt2img or edit, got {other:?}"),
    };
    let total_secs = started.elapsed().as_secs_f64();
    eprintln!();

    let png = out.join("image.png");
    image::RgbImage::from_raw(image.width, image.height, image.pixels.clone())
        .expect("image buffer matches its geometry")
        .save(&png)
        .unwrap_or_else(|e| panic!("write {}: {e}", png.display()));
    let requests = cache.requests.load(Ordering::SeqCst);
    let vram = probe.report();
    let report = serde_json::json!({
        "mode": mode,
        "tier": tier,
        "tierDir": tier_dir,
        "prompt": prompt,
        "width": image.width, "height": image.height,
        "steps": steps_total.or(effective_steps),
        "seed": seed,
        "savedQuantizationRequests": requests,
        "loadSecs": load_secs,
        "firstStepSecs": first_step_secs,
        "renderSecs": total_secs - load_secs,
        "totalSecs": total_secs,
        "vram": vram.to_string(),
        "png": png,
    });
    std::fs::write(
        out.join("report.json"),
        serde_json::to_string_pretty(&report).unwrap(),
    )
    .expect("write report.json");
    println!("{report}");
    println!("[qwen-tier-render] savedQuantizationRequests={requests}");
}
