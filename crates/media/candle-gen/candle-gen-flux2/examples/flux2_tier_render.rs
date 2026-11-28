//! Render one FLUX.2 (dev or klein) txt2img image from a staged tier directory and report
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
            "[flux2-tier-render] saved-quantization request #{n}: {}.{}",
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
    let tier_dir = PathBuf::from(env("FLUX2_TIER_DIR").expect("FLUX2_TIER_DIR"));
    let out = PathBuf::from(env("FLUX2_TIER_OUT").expect("FLUX2_TIER_OUT"));
    let tier = env("FLUX2_TIER").unwrap_or_else(|| "q4".into());
    let id = env("FLUX2_TIER_ID").unwrap_or_else(|| "flux2_dev".into());
    let quant = match tier.as_str() {
        "q4" => Quant::Q4,
        "q8" => Quant::Q8,
        other => panic!("FLUX2_TIER must be q4 or q8, got {other:?}"),
    };
    let width: u32 = env_or("FLUX2_TIER_W", 1024);
    let height: u32 = env_or("FLUX2_TIER_H", 1024);
    let seed: u64 = env_or("FLUX2_TIER_SEED", 42);
    let steps: Option<u32> = env("FLUX2_TIER_STEPS").map(|s| s.parse().expect("FLUX2_TIER_STEPS"));
    let gpu: usize = env_or("FLUX2_TIER_VRAM_GPU", 0);
    let prompt = env("FLUX2_TIER_PROMPT")
        .unwrap_or_else(|| "a red fox sitting in a snowy forest, photorealistic".into());
    std::fs::create_dir_all(&out).expect("create FLUX2_TIER_OUT");

    let scratch = out.join("saved-quantization");
    let _scratch_guard = ScratchDir(scratch.clone());
    let cache = Arc::new(CountingCache {
        requests: AtomicUsize::new(0),
        dir: scratch.clone(),
    });
    let _scope = QuantizationCacheContext::new(cache.clone()).install();
    let _runtime = candle_gen::quantization_cache::install_runtime_cache_context();
    let mut probe = candle_gen::testkit::VramProbe::start(gpu);

    eprintln!(
        "[flux2-tier-render] id={id} tier={tier} dir={} {width}x{height} steps={steps:?} seed={seed}",
        tier_dir.display()
    );
    let started = std::time::Instant::now();
    let load_phase = probe.phase();
    let mut spec = LoadSpec::new(WeightsSource::Dir(tier_dir.clone())).with_quant(quant);
    // Diagnostic: substitute another text-encoder component (e.g. the MLX tier's) for the tier's own.
    let te_override = env("FLUX2_TIER_TE_DIR").map(PathBuf::from);
    if let Some(te) = &te_override {
        spec.text_encoder = Some(WeightsSource::Dir(te.clone()));
    }
    let generator = candle_gen_flux2::provider_registry()
        .expect("provider registry")
        .load(&id, &spec)
        .unwrap_or_else(|e| panic!("load: {e}"));
    probe.end_load(load_phase);
    let load_secs = started.elapsed().as_secs_f64();

    // Saved-quantization request count at each progress milestone, so a mixed run can attribute
    // materializations to the phase (text encoder vs DiT) that asked for them.
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
            eprint!("\r[flux2-tier-render] step {current}/{total}   ");
        }
    };
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
    let total_secs = started.elapsed().as_secs_f64();
    eprintln!();
    let GenerationOutput::Images(images) = output else {
        panic!("{id} must produce images");
    };
    let image = &images[0];

    let png = out.join("image.png");
    image::RgbImage::from_raw(image.width, image.height, image.pixels.clone())
        .expect("image buffer matches its geometry")
        .save(&png)
        .unwrap_or_else(|e| panic!("write {}: {e}", png.display()));
    let requests = cache.requests.load(Ordering::SeqCst);
    let vram = probe.report();
    let report = serde_json::json!({
        "id": id,
        "tier": tier,
        "tierDir": tier_dir,
        "prompt": prompt,
        "width": image.width, "height": image.height,
        "steps": steps_total.or(steps),
        "seed": seed,
        "savedQuantizationRequests": requests,
        "requestMilestones": milestones,
        "textEncoderOverride": te_override,
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
    println!("[flux2-tier-render] savedQuantizationRequests={requests}");
}
