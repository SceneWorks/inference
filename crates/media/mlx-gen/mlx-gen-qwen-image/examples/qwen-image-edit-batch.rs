//! Run a receipt-driven batch of local Qwen-Image-Edit jobs.
//!
//! This example intentionally performs no model download and loads the local snapshot once for the
//! entire batch. It is a thin executable boundary for callers that need exact prompts, seeds, and
//! output paths without depending on the SceneWorks worker protocol.

use std::env;
use std::error::Error;
use std::fs;
use std::path::{Path, PathBuf};
use std::time::Instant;

use image::{ImageBuffer, RgbImage};
use mlx_gen::{
    Conditioning, GenerationOutput, GenerationRequest, Image, LoadPhase, LoadSpec, OffloadPolicy,
    Progress, Quant, WeightsSource,
};
use serde_json::{json, Map, Value};

const PLAN_KIND: &str = "qwen-image-edit-batch-plan";
const RESULT_KIND: &str = "qwen-image-edit-batch-result";

struct Arguments {
    snapshot: PathBuf,
    plan: PathBuf,
    quant: Quant,
    offload: OffloadPolicy,
}

fn usage() -> &'static str {
    "usage: qwen-image-edit-batch --snapshot DIR --plan FILE [--quant q4|q8] \
     [--offload resident|sequential]"
}

fn arguments() -> Result<Arguments, Box<dyn Error>> {
    let mut snapshot = None;
    let mut plan = None;
    let mut quant = Quant::Q8;
    let mut offload = OffloadPolicy::Sequential;
    let mut args = env::args().skip(1);
    while let Some(flag) = args.next() {
        let value = args
            .next()
            .ok_or_else(|| format!("missing value after {flag}; {}", usage()))?;
        match flag.as_str() {
            "--snapshot" => snapshot = Some(PathBuf::from(value)),
            "--plan" => plan = Some(PathBuf::from(value)),
            "--quant" => {
                quant = match value.as_str() {
                    "q4" => Quant::Q4,
                    "q8" => Quant::Q8,
                    _ => return Err(format!("unsupported quant tier: {value}").into()),
                }
            }
            "--offload" => {
                offload = match value.as_str() {
                    "resident" => OffloadPolicy::Resident,
                    "sequential" => OffloadPolicy::Sequential,
                    _ => return Err(format!("unsupported offload policy: {value}").into()),
                }
            }
            _ => return Err(format!("unknown argument: {flag}; {}", usage()).into()),
        }
    }
    Ok(Arguments {
        snapshot: snapshot.ok_or_else(|| format!("--snapshot is required; {}", usage()))?,
        plan: plan.ok_or_else(|| format!("--plan is required; {}", usage()))?,
        quant,
        offload,
    })
}

fn object<'a>(value: &'a Value, label: &str) -> Result<&'a Map<String, Value>, Box<dyn Error>> {
    value
        .as_object()
        .ok_or_else(|| format!("{label} must be an object").into())
}

fn text<'a>(object: &'a Map<String, Value>, key: &str) -> Result<&'a str, Box<dyn Error>> {
    object
        .get(key)
        .and_then(Value::as_str)
        .filter(|value| !value.is_empty())
        .ok_or_else(|| format!("{key} must be non-empty text").into())
}

fn unsigned(object: &Map<String, Value>, key: &str) -> Result<u64, Box<dyn Error>> {
    object
        .get(key)
        .and_then(Value::as_u64)
        .ok_or_else(|| format!("{key} must be a non-negative integer").into())
}

fn image_from_path(path: &Path) -> Result<Image, Box<dyn Error>> {
    let rgb = image::open(path)?.into_rgb8();
    Ok(Image {
        width: rgb.width(),
        height: rgb.height(),
        pixels: rgb.into_raw(),
    })
}

fn save_image(image: Image, path: &Path) -> Result<(), Box<dyn Error>> {
    if let Some(parent) = path.parent() {
        fs::create_dir_all(parent)?;
    }
    let encoded: RgbImage = ImageBuffer::from_raw(image.width, image.height, image.pixels)
        .ok_or("generator returned an invalid RGB pixel buffer")?;
    let file_name = path
        .file_name()
        .and_then(|value| value.to_str())
        .ok_or("output path must end in a UTF-8 file name")?;
    let temporary = path.with_file_name(format!(".{file_name}.tmp.png"));
    encoded.save(&temporary)?;
    fs::rename(temporary, path)?;
    Ok(())
}

fn progress(job_id: &str, event: Progress) {
    match event {
        Progress::Step { current, total } => eprintln!("[{job_id}] step {current}/{total}"),
        Progress::Decoding => eprintln!("[{job_id}] decoding"),
        Progress::Loading(LoadPhase::TextEncoder) => {
            eprintln!("[{job_id}] loading text/vision encoder")
        }
        Progress::Loading(LoadPhase::Renderer) => eprintln!("[{job_id}] loading renderer"),
    }
}

fn main() -> Result<(), Box<dyn Error>> {
    let args = arguments()?;
    if !args.snapshot.is_dir() {
        return Err(format!("snapshot does not exist: {}", args.snapshot.display()).into());
    }
    let plan_value: Value = serde_json::from_slice(&fs::read(&args.plan)?)?;
    let plan = object(&plan_value, "plan")?;
    if plan.get("schemaVersion").and_then(Value::as_u64) != Some(1)
        || plan.get("kind").and_then(Value::as_str) != Some(PLAN_KIND)
    {
        return Err("unsupported Qwen edit batch plan".into());
    }
    let jobs = plan
        .get("jobs")
        .and_then(Value::as_array)
        .filter(|jobs| !jobs.is_empty())
        .ok_or("plan.jobs must be a non-empty array")?;

    eprintln!("loading Qwen-Image-Edit from {}", args.snapshot.display());
    let load_started = Instant::now();
    let spec = LoadSpec::new(WeightsSource::Dir(args.snapshot.clone()))
        .with_quant(args.quant)
        .with_offload_policy(args.offload);
    let generator = mlx_gen_qwen_image::model_edit::load(&spec)?;
    let load_seconds = load_started.elapsed().as_secs_f64();
    eprintln!("model boundary ready in {load_seconds:.1}s");

    let batch_started = Instant::now();
    let mut results = Vec::with_capacity(jobs.len());
    for (index, value) in jobs.iter().enumerate() {
        let job = object(value, &format!("jobs[{index}]"))?;
        let job_id = text(job, "jobId")?;
        let reference_path = PathBuf::from(text(job, "referencePath")?);
        let output_path = PathBuf::from(text(job, "outputPath")?);
        let prompt = text(job, "prompt")?.to_owned();
        let negative_prompt = match job.get("negativePrompt") {
            None | Some(Value::Null) => None,
            Some(Value::String(value)) => Some(value.clone()),
            _ => return Err(format!("jobs[{index}].negativePrompt must be null or text").into()),
        };
        let width = u32::try_from(unsigned(job, "width")?)?;
        let height = u32::try_from(unsigned(job, "height")?)?;
        let steps = u32::try_from(unsigned(job, "steps")?)?;
        let seed = unsigned(job, "seed")?;
        let guidance = job
            .get("guidance")
            .and_then(Value::as_f64)
            .filter(|value| value.is_finite())
            .ok_or_else(|| format!("jobs[{index}].guidance must be finite"))?
            as f32;
        let reference = image_from_path(&reference_path)?;
        if reference.width != width || reference.height != height {
            return Err(format!(
                "job {job_id}: requested {width}x{height} must equal the reference dimensions {}x{}",
                reference.width, reference.height
            )
            .into());
        }

        eprintln!("[{job_id}] generating {}", output_path.display());
        let started = Instant::now();
        let request = GenerationRequest {
            prompt,
            negative_prompt,
            width,
            height,
            count: 1,
            seed: Some(seed),
            steps: Some(steps),
            guidance: Some(guidance),
            conditioning: vec![Conditioning::Reference {
                image: reference,
                strength: None,
            }],
            ..Default::default()
        };
        let output = generator.generate(&request, &mut |event| progress(job_id, event))?;
        let image = match output {
            GenerationOutput::Images(mut images) if images.len() == 1 => images.swap_remove(0),
            GenerationOutput::Images(images) => {
                return Err(
                    format!("job {job_id}: expected one image, got {}", images.len()).into(),
                )
            }
            _ => return Err(format!("job {job_id}: generator returned non-image output").into()),
        };
        if image.width != width || image.height != height {
            return Err(format!(
                "job {job_id}: output dimensions {}x{} differ from requested {width}x{height}",
                image.width, image.height
            )
            .into());
        }
        save_image(image, &output_path)?;
        results.push(json!({
            "jobId": job_id,
            "outputPath": output_path,
            "width": width,
            "height": height,
            "elapsedSeconds": started.elapsed().as_secs_f64(),
        }));
    }

    println!(
        "{}",
        serde_json::to_string_pretty(&json!({
            "schemaVersion": 1,
            "kind": RESULT_KIND,
            "modelLoadSeconds": load_seconds,
            "batchElapsedSeconds": batch_started.elapsed().as_secs_f64(),
            "jobs": results,
        }))?
    );
    Ok(())
}
