//! Real-weight throughput bench for the MiniMax-H3-Image LoRA trainer (SW-F369). NOT part of CI and
//! never run by a test: a manual workload on a CUDA box.
//!
//! Trains `H3_BENCH_STEPS` steps on a directory of `image + .txt caption` pairs at
//! `H3_BENCH_RESOLUTION` (rank 16, lr 1e-4, sigmoid/balanced timesteps, bf16 compute, gradient
//! checkpointing on — the trainer's defaults), prints every step's wall seconds (and, with
//! `H3_BENCH_SPANS=1` on a `model-inference-proof` build, the synchronized profile spans of each
//! step), saves the LoRA, then renders two six-step samples through the registered generator with
//! the saved adapter. Everything is written under `H3_BENCH_OUT`; nothing else is written.
//!
//! With `H3_BENCH_TRAINING_ADAPTER` the run trains against a frozen de-distill training adapter
//! (`TrainingConfig::training_adapter`, e.g. `ShootTheSound/Fizgig-MiniMax-H3-Image-Training-Adapter`'s
//! `fizgig_minimax_h3_image_training_adapter.safetensors`): applied to the DiT for every training
//! forward/backward, absent from the samples (they load the saved LoRA alone) and never saved.
//!
//! ```text
//! CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=0 \
//! H3_BENCH_SNAPSHOT=<MiniMax-H3-Image snapshot> \
//! H3_BENCH_TEXT_ENCODER=<MiniMax-H3 snapshot>/text_encoder \
//! H3_BENCH_DATA=<dir of a.png + a.txt ...> H3_BENCH_OUT=<new output dir> \
//!   cargo run --release -p candle-gen-minimax-h3 --example h3_image_train_bench \
//!   --features cuda,model-inference-proof
//! ```
//!
//! | variable | meaning (default) |
//! | --- | --- |
//! | `H3_BENCH_SNAPSHOT` | the `MiniMaxAI/MiniMax-H3-Image` snapshot root (required) |
//! | `H3_BENCH_TEXT_ENCODER` | the `text_encoder/` dir when the snapshot lacks one |
//! | `H3_BENCH_DIT` + `H3_BENCH_QUANT` | a staged packed DiT tier's `transformer_diffusers/` and its `q4` / `q8` (bf16 base when unset) |
//! | `H3_BENCH_DATA` | the dataset directory (required) |
//! | `H3_BENCH_OUT` | the output directory (required; created) |
//! | `H3_BENCH_STEPS` | optimizer steps (10) |
//! | `H3_BENCH_RESOLUTION` | training edge (2048) |
//! | `H3_BENCH_TIMESTEP_TYPE` / `H3_BENCH_TIMESTEP_BIAS` | (`sigmoid` / `balanced`) |
//! | `H3_BENCH_TRAINING_ADAPTER` | a frozen training adapter `.safetensors` applied while training (none) |
//! | `H3_BENCH_TRAINING_ADAPTER_STRENGTH` | the strength it is applied at (1.0, the production value) |
//! | `H3_BENCH_BUDGET_GB` | override the device training budget (GB) — e.g. 40 forces the streamed residency for an A/B |
//! | `H3_BENCH_SPANS` | `1`: record synchronized profile spans (needs `model-inference-proof`; perturbs step times) |
//! | `H3_BENCH_SAMPLE_PROMPTS` | `|`-separated sample prompts (the first two captions) |
//! | `H3_BENCH_SAMPLE_SIZE` | sample edge (the training edge) |
//! | `H3_BENCH_NO_SAMPLES` | `1`: skip the samples |

use std::path::{Path, PathBuf};
use std::time::Instant;

use candle_gen::gen_core::train::{
    Trainer, TrainingConfig, TrainingItem, TrainingProgress, TrainingRequest,
};
use candle_gen::gen_core::{
    AdapterKind, AdapterSpec, GenerationOutput, LoadShape, LoadSpec, Quant, WeightsSource,
};
use candle_gen_minimax_h3::training::{preview_request, MiniMaxH3ImageTrainer};

const LORA_FILE: &str = "h3_image_bench_lora.safetensors";

fn env(name: &str) -> Option<String> {
    std::env::var(name).ok().filter(|v| !v.trim().is_empty())
}

fn required_dir(name: &str) -> PathBuf {
    PathBuf::from(env(name).unwrap_or_else(|| panic!("{name} must be set")))
}

fn parsed<T: std::str::FromStr>(name: &str, default: T) -> T {
    env(name).map_or(default, |v| {
        v.trim()
            .parse()
            .unwrap_or_else(|_| panic!("{name}={v} does not parse"))
    })
}

fn spec() -> LoadSpec {
    let root = required_dir("H3_BENCH_SNAPSHOT");
    let mut spec = LoadSpec::new(WeightsSource::Dir(root));
    if let Some(te) = env("H3_BENCH_TEXT_ENCODER") {
        spec.components
            .insert("text_encoder".into(), WeightsSource::Dir(te.into()));
    }
    if let Some(dit) = env("H3_BENCH_DIT") {
        spec.components.insert(
            "transformer_diffusers".into(),
            WeightsSource::Dir(dit.into()),
        );
        spec.quantize = match env("H3_BENCH_QUANT").as_deref() {
            Some("q4") => Some(Quant::Q4),
            Some("q8") => Some(Quant::Q8),
            other => panic!("H3_BENCH_DIT needs H3_BENCH_QUANT=q4|q8, got {other:?}"),
        };
    }
    spec
}

/// `image + same-stem .txt` pairs in `dir`, sorted by name.
fn dataset(dir: &Path) -> Vec<TrainingItem> {
    let mut images: Vec<PathBuf> = std::fs::read_dir(dir)
        .unwrap_or_else(|e| panic!("reading {}: {e}", dir.display()))
        .flatten()
        .map(|e| e.path())
        .filter(|p| {
            p.extension().and_then(|e| e.to_str()).is_some_and(|e| {
                matches!(
                    e.to_ascii_lowercase().as_str(),
                    "png" | "jpg" | "jpeg" | "webp"
                )
            })
        })
        .collect();
    images.sort();
    let items: Vec<TrainingItem> = images
        .into_iter()
        .filter_map(|image| {
            let caption = std::fs::read_to_string(image.with_extension("txt")).ok()?;
            Some(TrainingItem::captioned(image, caption.trim().to_owned()))
        })
        .collect();
    assert!(
        !items.is_empty(),
        "{} holds no image + .txt caption pairs",
        dir.display()
    );
    items
}

/// Print and keep the spans recorded during `step` (a `model-inference-proof` build only).
fn record_spans(spans: bool, step: u32, rows: &mut Vec<serde_json::Value>) {
    #[cfg(feature = "model-inference-proof")]
    if spans {
        let totals = candle_gen::train::profile::take_span_totals();
        for (name, (calls, secs)) in &totals {
            eprintln!("[h3-bench]   span {name:<22} {calls:>4} calls {secs:>8.3} s");
        }
        rows.push(serde_json::json!({"step": step, "spans": totals}));
    }
    #[cfg(not(feature = "model-inference-proof"))]
    let _ = (spans, step, rows);
}

fn main() {
    let spec = spec();
    let data = required_dir("H3_BENCH_DATA");
    let out = required_dir("H3_BENCH_OUT");
    std::fs::create_dir_all(&out).expect("create H3_BENCH_OUT");
    let steps: u32 = parsed("H3_BENCH_STEPS", 10);
    let resolution: u32 = parsed("H3_BENCH_RESOLUTION", 2048);
    let items = dataset(&data);
    let captions: Vec<String> = items.iter().map(|i| i.caption.clone()).collect();
    let cfg = TrainingConfig {
        rank: 16,
        alpha: 16.0,
        learning_rate: 1e-4,
        steps,
        resolution,
        save_every: 0,
        seed: 369,
        train_dtype: "bf16".into(),
        timestep_type: env("H3_BENCH_TIMESTEP_TYPE").unwrap_or_else(|| "sigmoid".into()),
        timestep_bias: env("H3_BENCH_TIMESTEP_BIAS").unwrap_or_else(|| "balanced".into()),
        training_adapter: env("H3_BENCH_TRAINING_ADAPTER").map(PathBuf::from),
        ..gen_core_train::h3_image::default_training_config()
    };
    let adapter_strength: f32 = parsed(
        "H3_BENCH_TRAINING_ADAPTER_STRENGTH",
        candle_gen_minimax_h3::training::training_adapter::TRAINING_ADAPTER_STRENGTH,
    );
    if let Some(adapter) = &cfg.training_adapter {
        assert!(
            adapter.is_file(),
            "H3_BENCH_TRAINING_ADAPTER={} is not a file",
            adapter.display()
        );
    }
    assert!(cfg.gradient_checkpointing);
    eprintln!(
        "[h3-bench] {} items from {}, {steps} steps at {resolution}, rank {} lr {} {}/{} \
         timesteps, training adapter {}, out {}",
        items.len(),
        data.display(),
        cfg.rank,
        cfg.learning_rate,
        cfg.timestep_type,
        cfg.timestep_bias,
        cfg.training_adapter.as_ref().map_or_else(
            || "none".to_owned(),
            |p| format!("{} at {adapter_strength}", p.display())
        ),
        out.display()
    );
    let req = TrainingRequest {
        items,
        config: cfg.clone(),
        output_dir: out.clone(),
        file_name: LORA_FILE.into(),
        trigger_words: vec![],
        cancel: Default::default(),
    };

    let started = Instant::now();
    let mut trainer = MiniMaxH3ImageTrainer::load(&spec).expect("load the trainer");
    if let Some(gb) = env("H3_BENCH_BUDGET_GB") {
        let gb: f64 = gb.parse().expect("H3_BENCH_BUDGET_GB is a number");
        trainer.set_memory_budget_override(Some((gb * 1e9) as u64));
    }
    trainer.set_training_adapter_strength(adapter_strength);
    trainer.validate(&req).expect("the request validates");
    let spans = env("H3_BENCH_SPANS").as_deref() == Some("1");
    let mut step_seconds: Vec<f64> = Vec::new();
    let mut span_rows: Vec<serde_json::Value> = Vec::new();
    let mut last = Instant::now();
    let mut on_progress = |p: TrainingProgress| match p {
        TrainingProgress::Training { step, loss, .. } => {
            let secs = last.elapsed().as_secs_f64();
            last = Instant::now();
            step_seconds.push(secs);
            eprintln!("[h3-bench] step {step}/{steps} loss {loss:.5} {secs:.2} s");
            record_spans(spans, step, &mut span_rows);
        }
        TrainingProgress::Sample { .. } => {}
        other => {
            // Everything before step 1 (captions, latents, the DiT load) is preparation.
            last = Instant::now();
            eprintln!(
                "[h3-bench] {:?} ({:.0} s)",
                other,
                started.elapsed().as_secs_f64()
            );
        }
    };
    let mut run = |on_progress: &mut dyn FnMut(TrainingProgress)| trainer.train(&req, on_progress);
    #[cfg(feature = "model-inference-proof")]
    let output = if spans {
        candle_gen::train::profile::with_span_profile(|| run(&mut on_progress))
    } else {
        run(&mut on_progress)
    };
    #[cfg(not(feature = "model-inference-proof"))]
    let output = {
        assert!(
            !spans,
            "H3_BENCH_SPANS=1 needs a --features model-inference-proof build"
        );
        run(&mut on_progress)
    };
    let output = output.expect("training completes");
    drop(trainer);
    // Step 1 includes the DiT build (load, AdaLN precompute, block residency) — no progress event
    // separates them — so it is reported apart from the steady steps.
    let steady: Vec<f64> = step_seconds.iter().skip(2).copied().collect();
    let mean = steady.iter().sum::<f64>() / steady.len().max(1) as f64;
    eprintln!(
        "[h3-bench] trained {} steps, final loss {:.5}; step 1 (with the DiT build) {:.1} s, steady \
         mean over steps 3..{} {:.2} s/step; adapter {}",
        output.steps,
        output.final_loss,
        step_seconds.first().copied().unwrap_or(0.0),
        step_seconds.len(),
        mean,
        output.adapter_path.display()
    );

    let mut samples = Vec::new();
    if env("H3_BENCH_NO_SAMPLES").as_deref() != Some("1") {
        let prompts: Vec<String> = match env("H3_BENCH_SAMPLE_PROMPTS") {
            Some(p) => p.split('|').map(|s| s.trim().to_owned()).collect(),
            None => captions.iter().take(2).cloned().collect(),
        };
        let edge: u32 = parsed("H3_BENCH_SAMPLE_SIZE", resolution);
        let mut render_spec = spec
            .clone()
            .with_load_shape(LoadShape::DeferredMaterialization);
        render_spec.adapters = vec![AdapterSpec::new(
            output.adapter_path.clone(),
            1.0,
            AdapterKind::Lora,
        )];
        for (index, prompt) in prompts.iter().take(2).enumerate() {
            let t = Instant::now();
            // A fresh generator per sample: the registered load + adapter install a user's render
            // runs (six distilled steps).
            let generator =
                candle_gen_minimax_h3::image_model::load_image(&render_spec).expect("load");
            let req = preview_request(
                prompt,
                369 + index as u64,
                (edge, edge),
                &[],
                Default::default(),
            );
            generator.validate(&req).expect("sample request validates");
            let image = match generator
                .generate(&req, &mut |_| {})
                .expect("sample renders")
            {
                GenerationOutput::Images(mut images) => images.remove(0),
                _ => panic!("minimax_h3_image returns images"),
            };
            drop(generator);
            let path = out.join(format!("sample_{index}.png"));
            image::save_buffer(
                &path,
                &image.pixels,
                image.width,
                image.height,
                image::ColorType::Rgb8,
            )
            .expect("save sample");
            eprintln!(
                "[h3-bench] sample {index} ({prompt:?}) -> {} in {:.0} s",
                path.display(),
                t.elapsed().as_secs_f64()
            );
            samples.push(serde_json::json!({"prompt": prompt, "path": path, "seconds": t.elapsed().as_secs_f64()}));
        }
    }

    let report = serde_json::json!({
        "steps": output.steps,
        "resolution": resolution,
        "training_adapter": cfg.training_adapter,
        "training_adapter_strength": cfg.training_adapter.as_ref().map(|_| adapter_strength),
        "final_loss": output.final_loss,
        "step_seconds": step_seconds,
        "steady_mean_seconds_from_step_3": mean,
        "spans_enabled": spans,
        "spans": span_rows,
        "adapter": output.adapter_path,
        "samples": samples,
    });
    std::fs::write(
        out.join("bench_report.json"),
        serde_json::to_vec_pretty(&report).unwrap(),
    )
    .expect("write the report");
}
