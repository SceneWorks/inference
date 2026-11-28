//! Render one short LTX clip (LTX-2.3 or LTX-2.5 distilled) from a staged split tier directory and
//! report whether any projection went through the saved-quantization cache. NOT part of CI: it
//! needs real weights and a CUDA GPU. Modeled on `candle-gen-flux2/examples/flux2_tier_render.rs`.
//!
//! ```text
//! LTX_TIER_DIR=<split tier (transformer.safetensors, ..., split_model.json)> \
//! LTX_TIER_ID=ltx_2_3_distilled|ltx_2_5_distilled LTX_TIER=q4|q8|bf16 \
//! LTX_TIER_OUT=<dir for frames + report> [LTX_TIER_TE=<separate Gemma text-encoder dir>] \
//! [LTX_TIER_INPUT=<png reference> LTX_TIER_W=704 LTX_TIER_H=480 LTX_TIER_FRAMES=33] \
//!   cargo run --release --features cuda --example ltx_tier_render -p candle-gen-ltx
//! ```
//!
//! The load follows the production shape: `resolved_route` is the engine id and `quantize` names
//! the tier. LTX-2.3 tiers carry no Gemma encoder; `LTX_TIER_TE` supplies it through
//! `LoadSpec::text_encoder`. A counting `QuantizationArtifactCache` is installed for the whole
//! load + render. A Candle-native (GGML) tier must leave it at **zero** materializations in every
//! phase. Writes the first, middle and last frames as PNG plus `report.json`.
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Arc;

use candle_gen::gen_core::{
    Conditioning, GenerationOutput, GenerationRequest, Image, LoadSpec, Progress, Quant,
    QuantizationArtifact, QuantizationArtifactCache, QuantizationArtifactKey,
    QuantizationCacheContext, WeightsSource,
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
            "[ltx-tier-render] saved-quantization request #{n}: {}.{}",
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
    let tier_dir = PathBuf::from(env("LTX_TIER_DIR").expect("LTX_TIER_DIR"));
    let out = PathBuf::from(env("LTX_TIER_OUT").expect("LTX_TIER_OUT"));
    let tier = env("LTX_TIER").unwrap_or_else(|| "q4".into());
    let id = env("LTX_TIER_ID").unwrap_or_else(|| "ltx_2_5_distilled".into());
    let quant = match tier.as_str() {
        "q4" => Some(Quant::Q4),
        "q8" => Some(Quant::Q8),
        "bf16" => None,
        other => panic!("LTX_TIER must be q4, q8 or bf16, got {other:?}"),
    };
    let width: u32 = env_or("LTX_TIER_W", 704);
    let height: u32 = env_or("LTX_TIER_H", 480);
    let frames: u32 = env_or("LTX_TIER_FRAMES", 33);
    let seed: u64 = env_or("LTX_TIER_SEED", 42);
    let steps: Option<u32> = env("LTX_TIER_STEPS").map(|s| s.parse().expect("LTX_TIER_STEPS"));
    let prompt = env("LTX_TIER_PROMPT")
        .unwrap_or_else(|| "a red fox walking through a snowy forest, photorealistic".into());
    let reference = env("LTX_TIER_INPUT").map(|path| {
        let rgb = image::open(&path)
            .unwrap_or_else(|e| panic!("open {path}: {e}"))
            .to_rgb8();
        Image {
            width: rgb.width(),
            height: rgb.height(),
            pixels: rgb.into_raw(),
        }
    });
    let load_only = env("LTX_TIER_LOAD_ONLY").is_some();
    std::fs::create_dir_all(&out).expect("create LTX_TIER_OUT");

    let scratch = out.join("saved-quantization");
    let _scratch_guard = ScratchDir(scratch.clone());
    let cache = Arc::new(CountingCache {
        requests: AtomicUsize::new(0),
        dir: scratch.clone(),
    });
    let _scope = QuantizationCacheContext::new(cache.clone()).install();
    let _runtime = candle_gen::quantization_cache::install_runtime_cache_context();
    let mut probe = candle_gen::testkit::VramProbe::start_rendered();

    eprintln!(
        "[ltx-tier-render] id={id} tier={tier} dir={} {width}x{height} frames={frames} steps={steps:?} seed={seed} reference={}",
        tier_dir.display(),
        reference.is_some()
    );
    let started = std::time::Instant::now();
    let load_phase = probe.phase();
    let mut spec = LoadSpec::new(WeightsSource::Dir(tier_dir.clone())).with_resolved_route(&id);
    spec.quantize = quant;
    let text_encoder = env("LTX_TIER_TE").map(PathBuf::from);
    if let Some(te) = &text_encoder {
        spec = spec.with_text_encoder(WeightsSource::Dir(te.clone()));
    }
    let generator = candle_gen_ltx::provider_registry()
        .expect("provider registry")
        .load(&id, &spec)
        .unwrap_or_else(|e| panic!("load: {e}"));
    probe.end_load(load_phase);
    let load_secs = started.elapsed().as_secs_f64();
    if load_only {
        println!(
            "[ltx-tier-render] load-only: loaded in {load_secs:.1}s, savedQuantizationRequests={}",
            cache.requests.load(Ordering::SeqCst)
        );
        return;
    }

    // Saved-quantization request count at each progress milestone, so a run can attribute any
    // materialization to the phase (text encoder vs DiT vs VAE / upsampler) that asked for it.
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
            eprint!("\r[ltx-tier-render] step {current}/{total}   ");
        }
    };
    let req = GenerationRequest {
        prompt: prompt.clone(),
        width,
        height,
        count: 1,
        frames: Some(frames),
        seed: Some(seed),
        steps,
        conditioning: reference
            .into_iter()
            .map(|image| Conditioning::Reference {
                image,
                strength: None,
            })
            .collect(),
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
        panic!("{id} must produce video");
    };
    assert!(!frames.is_empty(), "{id} returned no frames");
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
        "id": id,
        "tier": tier,
        "tierDir": tier_dir,
        "textEncoder": text_encoder,
        "prompt": prompt,
        "width": frames[0].width, "height": frames[0].height,
        "framesOut": frames.len(), "fps": fps,
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
    println!("[ltx-tier-render] savedQuantizationRequests={requests}");
}
