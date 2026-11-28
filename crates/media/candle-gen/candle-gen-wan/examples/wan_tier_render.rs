//! Render one short Wan2.2 clip (TI2V-5B, T2V-A14B or I2V-A14B) from a staged tier directory and
//! report whether any projection went through the saved-quantization cache. NOT part of CI: it
//! needs real weights and a CUDA GPU. Modeled on `candle-gen-flux2/examples/flux2_tier_render.rs`.
//!
//! ```text
//! WAN_TIER_DIR=<tier root (text_encoder/, transformer/[, transformer_2/], vae/, tokenizer/)> \
//! WAN_TIER_ID=wan2_2_ti2v_5b|wan2_2_t2v_14b|wan2_2_i2v_14b WAN_TIER=q4|q8|bf16 \
//! WAN_TIER_OUT=<dir for frames + report> [WAN_TIER_INPUT=<png, required for I2V>] \
//! [WAN_TIER_W=832 WAN_TIER_H=480 WAN_TIER_FRAMES=33 WAN_TIER_STEPS=<default>] \
//!   cargo run --release --features cuda --example wan_tier_render -p candle-gen-wan
//! ```
//!
//! The load follows the production shape: `resolved_route` is the engine id, `quantize` names the
//! tier, and `i2v_memory_strategy::prepare_load_spec` binds the snapshot identity and pins its
//! inventory before the registry load. A counting `QuantizationArtifactCache` is installed for the
//! whole load + render. A Candle-native (GGML) tier must leave it at **zero** materializations in
//! every phase. Writes the first, middle and last frames as PNG plus `report.json`.
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
            "[wan-tier-render] saved-quantization request #{n}: {}.{}",
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
    let tier_dir = PathBuf::from(env("WAN_TIER_DIR").expect("WAN_TIER_DIR"));
    let out = PathBuf::from(env("WAN_TIER_OUT").expect("WAN_TIER_OUT"));
    let tier = env("WAN_TIER").unwrap_or_else(|| "q4".into());
    let id = env("WAN_TIER_ID").unwrap_or_else(|| "wan2_2_t2v_14b".into());
    let quant = match tier.as_str() {
        "q4" => Some(Quant::Q4),
        "q8" => Some(Quant::Q8),
        "bf16" => None,
        other => panic!("WAN_TIER must be q4, q8 or bf16, got {other:?}"),
    };
    let width: u32 = env_or("WAN_TIER_W", 832);
    let height: u32 = env_or("WAN_TIER_H", 480);
    let frames: u32 = env_or("WAN_TIER_FRAMES", 33);
    let seed: u64 = env_or("WAN_TIER_SEED", 42);
    let steps: Option<u32> = env("WAN_TIER_STEPS").map(|s| s.parse().expect("WAN_TIER_STEPS"));
    let prompt = env("WAN_TIER_PROMPT")
        .unwrap_or_else(|| "a red fox walking through a snowy forest, photorealistic".into());
    let reference = env("WAN_TIER_INPUT").map(|path| {
        let rgb = image::open(&path)
            .unwrap_or_else(|e| panic!("open {path}: {e}"))
            .to_rgb8();
        Image {
            width: rgb.width(),
            height: rgb.height(),
            pixels: rgb.into_raw(),
        }
    });
    let load_only = env("WAN_TIER_LOAD_ONLY").is_some();
    std::fs::create_dir_all(&out).expect("create WAN_TIER_OUT");

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
        "[wan-tier-render] id={id} tier={tier} dir={} {width}x{height} frames={frames} steps={steps:?} seed={seed} reference={}",
        tier_dir.display(),
        reference.is_some()
    );
    let started = std::time::Instant::now();
    let load_phase = probe.phase();
    let mut spec = LoadSpec::new(WeightsSource::Dir(tier_dir.clone())).with_resolved_route(&id);
    spec.quantize = quant;
    candle_gen_wan::i2v_memory_strategy::prepare_load_spec(&mut spec, &id)
        .unwrap_or_else(|e| panic!("prepare_load_spec: {e}"));
    let generator = candle_gen_wan::provider_registry()
        .expect("provider registry")
        .load(&id, &spec)
        .unwrap_or_else(|e| panic!("load: {e}"));
    probe.end_load(load_phase);
    let load_secs = started.elapsed().as_secs_f64();
    if load_only {
        println!(
            "[wan-tier-render] load-only: loaded in {load_secs:.1}s, savedQuantizationRequests={}",
            cache.requests.load(Ordering::SeqCst)
        );
        return;
    }

    // Saved-quantization request count at each progress milestone, so a run can attribute any
    // materialization to the phase (text encoder vs DiT experts vs VAE) that asked for it.
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
            eprint!("\r[wan-tier-render] step {current}/{total}   ");
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
        // The worker spells the mode the prepared memory receipt selects its row by.
        video_mode: Some(
            if reference.is_some() {
                "image_to_video"
            } else {
                "text_to_video"
            }
            .to_owned(),
        ),
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
    println!("[wan-tier-render] savedQuantizationRequests={requests}");
}
