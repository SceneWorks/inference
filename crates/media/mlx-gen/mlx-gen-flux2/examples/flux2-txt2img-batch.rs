//! Resumable receipt-driven FLUX.2 text-to-image batch worker for local evaluation campaigns.
//!
//! This is an accelerator/manual example, not a CI test. It downloads nothing, loads one local
//! snapshot once, writes each PNG atomically, and persists the batch result after every image.

use std::env;
use std::error::Error;
use std::fs;
use std::path::{Path, PathBuf};
use std::time::Instant;

use image::{ImageBuffer, RgbImage};
use mlx_gen::{
    GenerationOutput, GenerationRequest, Image, LoadPhase, LoadSpec, OffloadPolicy, Progress,
    Quant, WeightsSource,
};
use serde_json::{json, Map, Value};

struct Arguments {
    provider_id: String,
    plan: PathBuf,
    quant: Quant,
    offload: OffloadPolicy,
}

fn usage() -> &'static str {
    "usage: flux2-txt2img-batch --provider flux2_klein_9b --plan FILE \
     [--quant q4|q8] [--offload resident|sequential]"
}

fn arguments() -> Result<Arguments, Box<dyn Error>> {
    let mut provider_id = None;
    let mut plan = None;
    let mut quant = Quant::Q8;
    let mut offload = OffloadPolicy::Sequential;
    let mut args = env::args().skip(1);
    while let Some(flag) = args.next() {
        let value = args
            .next()
            .ok_or_else(|| format!("missing value after {flag}; {}", usage()))?;
        match flag.as_str() {
            "--provider" => provider_id = Some(value),
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
    let provider_id = provider_id.ok_or_else(|| format!("--provider is required; {}", usage()))?;
    if provider_id != "flux2_klein_9b" {
        return Err(format!("unsupported batch provider: {provider_id}").into());
    }
    Ok(Arguments {
        provider_id,
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

fn optional_number(object: &Map<String, Value>, key: &str) -> Result<Option<f32>, Box<dyn Error>> {
    match object.get(key) {
        None | Some(Value::Null) => Ok(None),
        Some(value) => value
            .as_f64()
            .filter(|number| number.is_finite())
            .map(|number| Some(number as f32))
            .ok_or_else(|| format!("{key} must be null or finite numeric").into()),
    }
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

fn existing_dimensions(path: &Path) -> Option<(u32, u32)> {
    image::image_dimensions(path).ok()
}

fn progress(job_id: &str, event: Progress) {
    match event {
        Progress::Step { current, total } => eprintln!("[{job_id}] step {current}/{total}"),
        Progress::Decoding => eprintln!("[{job_id}] decoding"),
        Progress::Loading(LoadPhase::TextEncoder) => {
            eprintln!("[{job_id}] loading text encoder")
        }
        Progress::Loading(LoadPhase::Renderer) => eprintln!("[{job_id}] loading renderer"),
    }
}

fn persist_result(
    path: &Path,
    provider_id: &str,
    model_path: &Path,
    quant: Quant,
    offload: OffloadPolicy,
    model_load_seconds: f64,
    results: &[Value],
) -> Result<(), Box<dyn Error>> {
    if let Some(parent) = path.parent() {
        fs::create_dir_all(parent)?;
    }
    let document = json!({
        "schemaVersion": 1,
        "kind": "flux2-txt2img-batch-result",
        "providerId": provider_id,
        "modelPath": model_path,
        "quantization": format!("{quant:?}").to_lowercase(),
        "offloadPolicy": format!("{offload:?}").to_lowercase(),
        "modelLoadSeconds": model_load_seconds,
        "completed": results.len(),
        "results": results,
    });
    let file_name = path
        .file_name()
        .and_then(|value| value.to_str())
        .ok_or("result path must end in a UTF-8 file name")?;
    let temporary = path.with_file_name(format!(".{file_name}.tmp"));
    fs::write(&temporary, serde_json::to_vec_pretty(&document)?)?;
    fs::rename(temporary, path)?;
    Ok(())
}

fn main() -> Result<(), Box<dyn Error>> {
    let args = arguments()?;
    let plan_value: Value = serde_json::from_slice(&fs::read(&args.plan)?)?;
    let plan = object(&plan_value, "plan")?;
    let model_path = PathBuf::from(text(plan, "modelPath")?);
    if !model_path.is_dir() {
        return Err(format!("model snapshot does not exist: {}", model_path.display()).into());
    }
    if !matches!(plan.get("adapter"), None | Some(Value::Null)) {
        return Err("FLUX.2 acquisition batches do not accept an adapter".into());
    }
    let result_path = PathBuf::from(text(plan, "resultPath")?);
    let jobs = plan
        .get("requests")
        .and_then(Value::as_array)
        .filter(|jobs| !jobs.is_empty())
        .ok_or("plan.requests must be a non-empty array")?;

    eprintln!(
        "loading {} from {} ({:?}, {:?})",
        args.provider_id,
        model_path.display(),
        args.quant,
        args.offload
    );
    let load_started = Instant::now();
    let spec = LoadSpec::new(WeightsSource::Dir(model_path.clone()))
        .with_quant(args.quant)
        .with_offload_policy(args.offload);
    let generator = mlx_gen_flux2::provider_registry()?.load(&args.provider_id, &spec)?;
    let model_load_seconds = load_started.elapsed().as_secs_f64();
    eprintln!("model boundary ready in {model_load_seconds:.1}s");

    let mut results = Vec::with_capacity(jobs.len());
    persist_result(
        &result_path,
        &args.provider_id,
        &model_path,
        args.quant,
        args.offload,
        model_load_seconds,
        &results,
    )?;
    for (index, value) in jobs.iter().enumerate() {
        let job = object(value, &format!("requests[{index}]"))?;
        let job_id = text(job, "id")?;
        let output_path = PathBuf::from(text(job, "outputPath")?);
        let width = u32::try_from(unsigned(job, "width")?)?;
        let height = u32::try_from(unsigned(job, "height")?)?;
        let steps = u32::try_from(unsigned(job, "steps")?)?;
        let seed = unsigned(job, "seed")?;
        let prompt = text(job, "prompt")?.to_owned();
        let guidance = optional_number(job, "guidance")?;
        if existing_dimensions(&output_path) == Some((width, height)) {
            eprintln!("[{job_id}] skip existing {}", output_path.display());
            results.push(json!({
                "jobId": job_id,
                "outputPath": output_path,
                "width": width,
                "height": height,
                "seed": seed,
                "steps": steps,
                "status": "reused_existing",
                "elapsedSeconds": 0.0,
            }));
            persist_result(
                &result_path,
                &args.provider_id,
                &model_path,
                args.quant,
                args.offload,
                model_load_seconds,
                &results,
            )?;
            continue;
        }

        eprintln!("[{job_id}] generating {}", output_path.display());
        let started = Instant::now();
        let request = GenerationRequest {
            prompt,
            width,
            height,
            count: 1,
            seed: Some(seed),
            steps: Some(steps),
            guidance,
            ..Default::default()
        };
        generator.validate(&request)?;
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
                "job {job_id}: output dimensions {}x{} differ from {width}x{height}",
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
            "seed": seed,
            "steps": steps,
            "status": "generated",
            "elapsedSeconds": started.elapsed().as_secs_f64(),
        }));
        persist_result(
            &result_path,
            &args.provider_id,
            &model_path,
            args.quant,
            args.offload,
            model_load_seconds,
            &results,
        )?;
    }
    println!(
        "completed {} FLUX.2 renders; result={}",
        results.len(),
        result_path.display()
    );
    Ok(())
}
