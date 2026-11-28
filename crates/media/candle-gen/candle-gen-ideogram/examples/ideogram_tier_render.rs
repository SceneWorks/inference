//! Render one Ideogram 4 (base or turbo) txt2img image from a staged tier directory and report
//! whether any projection went through the saved-quantization cache. NOT part of CI: it needs real
//! weights and a CUDA GPU. Modeled on `candle-gen-flux2/examples/flux2_tier_render.rs`.
//!
//! ```text
//! IDEOGRAM_TIER_DIR=<tier root (transformer/, unconditional_transformer/, text_encoder/, vae/, ...)> \
//! IDEOGRAM_TIER_ID=ideogram_4|ideogram_4_turbo IDEOGRAM_TIER=q4|q8|bf16 \
//! IDEOGRAM_TIER_OUT=<dir for image + report> \
//!   cargo run --release --features cuda --example ideogram_tier_render -p candle-gen-ideogram
//! ```
//!
//! A counting `QuantizationArtifactCache` is installed for the whole load + render. A Candle-native
//! (GGML) tier must leave it at **zero** materializations in every phase; an MLX-affine tier routes
//! every packed projection through it. Writes `image.png` plus `report.json`.
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Arc;

use candle_gen::gen_core::{
    GenerationOutput, GenerationRequest, LoadSpec, Progress, QuantizationArtifact,
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
            "[ideogram-tier-render] saved-quantization request #{n}: {}.{}",
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

/// Wrap a plain prompt into Ideogram's minimal JSON caption (as `examples/ideogram-render.rs` and
/// the worker's `ideogram_caption::ensure_caption_prompt` do); a caption passes through unchanged.
fn to_caption(prompt: &str) -> String {
    let p = prompt.trim();
    if p.starts_with('{') && p.contains("compositional_deconstruction") {
        return p.to_string();
    }
    let q: String = p
        .chars()
        .flat_map(|c| match c {
            '"' => vec!['\\', '"'],
            '\\' => vec!['\\', '\\'],
            _ => vec![c],
        })
        .collect();
    format!(
        "{{\"high_level_description\": \"{q}\", \"compositional_deconstruction\": \
         {{\"background\": \"{q}\", \"elements\": []}}}}"
    )
}

fn main() {
    let tier_dir = PathBuf::from(env("IDEOGRAM_TIER_DIR").expect("IDEOGRAM_TIER_DIR"));
    let out = PathBuf::from(env("IDEOGRAM_TIER_OUT").expect("IDEOGRAM_TIER_OUT"));
    let tier = env("IDEOGRAM_TIER").unwrap_or_else(|| "q4".into());
    let id = env("IDEOGRAM_TIER_ID").unwrap_or_else(|| "ideogram_4".into());
    // `IDEOGRAM_TIER` only labels the report: an Ideogram turnkey reads its physical tier from the
    // snapshot and refuses `spec.quantize`; the load instead requires the exact `resolved_route`.
    assert!(
        matches!(tier.as_str(), "q4" | "q8" | "bf16"),
        "IDEOGRAM_TIER must be q4, q8 or bf16, got {tier:?}"
    );
    let width: u32 = env_or("IDEOGRAM_TIER_W", 1024);
    let height: u32 = env_or("IDEOGRAM_TIER_H", 1024);
    let seed: u64 = env_or("IDEOGRAM_TIER_SEED", 42);
    let steps: Option<u32> =
        env("IDEOGRAM_TIER_STEPS").map(|s| s.parse().expect("IDEOGRAM_TIER_STEPS"));
    // Ideogram 4 expects a JSON caption; the worker wraps a plain prompt the same way.
    let prompt = to_caption(
        &env("IDEOGRAM_TIER_PROMPT")
            .unwrap_or_else(|| "a red fox sitting in a snowy forest, photorealistic".into()),
    );
    std::fs::create_dir_all(&out).expect("create IDEOGRAM_TIER_OUT");

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
        "[ideogram-tier-render] id={id} tier={tier} dir={} {width}x{height} steps={steps:?} seed={seed}",
        tier_dir.display()
    );
    let started = std::time::Instant::now();
    let load_phase = probe.phase();
    let mut spec = LoadSpec::new(WeightsSource::Dir(tier_dir.clone()));
    spec.resolved_route = Some(id.clone());
    let generator = candle_gen_ideogram::provider_registry()
        .expect("provider registry")
        .load(&id, &spec)
        .unwrap_or_else(|e| panic!("load: {e}"));
    probe.end_load(load_phase);
    let load_secs = started.elapsed().as_secs_f64();
    if env("IDEOGRAM_TIER_LOAD_ONLY").is_some() {
        println!(
            "[ideogram-tier-render] load-only: loaded in {load_secs:.1}s, savedQuantizationRequests={}",
            cache.requests.load(Ordering::SeqCst)
        );
        return;
    }

    // Saved-quantization request count at each progress milestone, so a run can attribute any
    // materialization to the phase (text encoders vs MMDiT vs VAE) that asked for it.
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
            eprint!("\r[ideogram-tier-render] step {current}/{total}   ");
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
    milestones.push((
        "afterGenerate".into(),
        cache.requests.load(Ordering::SeqCst),
    ));
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
    println!("[ideogram-tier-render] savedQuantizationRequests={requests}");
}
