//! Wan2.2 **T2V-A14B** (dual-expert MoE) txt2video smoke driver — resolves THIS crate's
//! explicitly registered generator through `provider_registry().load("wan2_2_t2v_14b", …)`, runs a real
//! `generate` against a local Wan2.2-T2V-A14B diffusers snapshot, and writes each decoded frame to PNG.
//! The human-eyeball check behind sc-5174.
//!
//! ```text
//! cargo run --release --example wan14b-txt2video --features cuda -- \
//!   --snapshot "C:\Users\…\models--Wan-AI--Wan2.2-T2V-A14B-Diffusers\snapshots\<hash>" \
//!   --prompt "a fluffy cat walking across a sunny garden, cinematic" \
//!   --width 320 --height 320 --frames 17 --steps 20 --seed 42 --out wan14b_smoke
//! ```
//!
//! With `--comfyui-high <file> --comfyui-low <file>` it instead loads those two in-place ComfyUI Wan2.2
//! experts (native-Wan keys, companion scaled-fp8) via
//! [`candle_gen_wan::wan14b::load_from_comfyui_experts_with_options`], sourcing the UMT5 TE / VAE /
//! tokenizer from `--snapshot`. The sc-10671 GPU-val path. Adding `--comfyui-te <file>`
//! (`umt5_xxl_fp8_e4m3fn_scaled`) and/or `--comfyui-vae <file>` (`wan_2.1_vae.safetensors`) reads those
//! components in place too (sc-10909); whichever is omitted falls back to `--snapshot`.
//!
//! SW-6137 (opt-in Sol-Attn sparse self-attention, lossy): `--sol-topk <ratio>` loads the generator
//! with the `sol_attention` engine option (top-k selection at `ratio`), `--sol-sinks <start>:<end>`
//! adds forced sink key blocks (on the snapshot and the `--comfyui-high/--comfyui-low` lanes alike).
//! The run prints the attention path counts, the mean s/step over the denoise steps and the CUDA
//! pool's peak (`USED_MEM_HIGH`). `--attn-capture <skip>:<file>` saves the Q/K/V of the
//! opt-in-routed self-attention call after `skip` others to a safetensors file (evidence tooling for
//! the locality investigation; it needs `--sol-topk`, the only opt-in this example sets).

use std::path::PathBuf;

use candle_gen::gen_core::{
    AdapterKind, AdapterSpec, GenerationOutput, GenerationRequest, LoadSpec, MoeExpert,
    OffloadPolicy, Progress, WeightsSource,
};

type Result<T> = std::result::Result<T, Box<dyn std::error::Error>>;

fn arg(args: &[String], key: &str) -> Option<String> {
    args.iter()
        .position(|a| a == key)
        .and_then(|i| args.get(i + 1).cloned())
}

fn campaign_contract(args: &[String]) -> Result<Option<(String, String, OffloadPolicy)>> {
    if !args.iter().any(|arg| arg == "--sc20686-campaign") {
        return Ok(None);
    }
    let source_ref = arg(args, "--sc20686-source-ref")
        .ok_or("SC-20686 campaign requires --sc20686-source-ref <inference-commit>")?;
    let residency = arg(args, "--sc20686-residency")
        .ok_or("SC-20686 campaign requires --sc20686-residency sequential")?;
    if residency != "sequential" {
        return Err("wan2_2_t2v_14b campaign residency must be sequential".into());
    }
    Ok(Some((source_ref, residency, OffloadPolicy::Sequential)))
}

fn main() -> Result<()> {
    let args: Vec<String> = std::env::args().collect();
    let snapshot = arg(&args, "--snapshot")
        .or_else(|| std::env::var("WAN14B_SNAPSHOT").ok())
        .ok_or("pass --snapshot <dir> (or set WAN14B_SNAPSHOT)")?;
    // Campaign observation is armed before loading but activated only by the producer after the
    // registry has resolved the real snapshot and request geometry.  This keeps caller flags from
    // becoming evidence and ensures the scope covers generation and release.
    let campaign_contract = campaign_contract(&args)?;
    let _campaign_request = if let Some((source_ref, residency, _)) = &campaign_contract {
        {
            let event_path = arg(&args, "--sc20686-events")
                .filter(|path| path != "-")
                .ok_or("SC-20686 campaign requires a dedicated --sc20686-events <file>")?;
            let request = candle_gen_wan::sc20686_observer::request_output(
                event_path, source_ref, residency,
            )?;
            Some(if args.iter().any(|arg| arg == "--sc20686-cancel") {
                request.arm_cancellation()
            } else {
                request.arm()
            })
        }
    } else {
        None
    };
    let prompt = arg(&args, "--prompt").unwrap_or_else(|| {
        "a fluffy cat walking across a sunny garden, gentle camera pan, cinematic, highly detailed"
            .into()
    });
    let negative = arg(&args, "--negative");
    let steps: Option<u32> = arg(&args, "--steps").and_then(|s| s.parse().ok());
    let guidance: Option<f32> = arg(&args, "--guidance").and_then(|s| s.parse().ok());
    let seed: u64 = arg(&args, "--seed")
        .and_then(|s| s.parse().ok())
        .unwrap_or(42);
    let width: u32 = arg(&args, "--width")
        .and_then(|s| s.parse().ok())
        .unwrap_or(320);
    let height: u32 = arg(&args, "--height")
        .and_then(|s| s.parse().ok())
        .unwrap_or(320);
    let frames: Option<u32> = arg(&args, "--frames").and_then(|s| s.parse().ok());
    let fps: Option<u32> = arg(&args, "--fps").and_then(|s| s.parse().ok());
    let sampler = arg(&args, "--sampler");
    let out = arg(&args, "--out").unwrap_or_else(|| "wan14b_smoke".into());
    // Optional A14B MoE LoRA pair (e.g. the lightx2v Lightning distill, sc-10026): the high-noise file
    // tags the high expert, the low-noise file the low. Applied additively on a packed q4/q8 tier
    // (sc-10094/10095) or folded on a dense snapshot. `--lora-scale` scales both (default 1.0).
    let lora_high = arg(&args, "--lora-high");
    let lora_low = arg(&args, "--lora-low");
    let lora_scale: f32 = arg(&args, "--lora-scale")
        .and_then(|s| s.parse().ok())
        .unwrap_or(1.0);
    let mut adapters: Vec<AdapterSpec> = Vec::new();
    if let Some(high) = &lora_high {
        adapters.push(
            AdapterSpec::new(PathBuf::from(high), lora_scale, AdapterKind::Lora)
                .with_moe_expert(MoeExpert::High),
        );
    }
    if let Some(low) = &lora_low {
        adapters.push(
            AdapterSpec::new(PathBuf::from(low), lora_scale, AdapterKind::Lora)
                .with_moe_expert(MoeExpert::Low),
        );
    }

    println!(
        "[smoke] snapshot={snapshot}\n[smoke] {width}x{height} frames={frames:?} steps={steps:?} \
         guidance={guidance:?} sampler={sampler:?} seed={seed}\n[smoke] prompt={prompt:?}\n\
         [smoke] adapters: high={lora_high:?} low={lora_low:?} scale={lora_scale}"
    );

    // sc-10671: `--comfyui-high/--comfyui-low` read the two ComfyUI experts in place (scaled-fp8 dequant
    // + native→diffusers remap), sourcing TE/VAE/tokenizer from `--snapshot`; else the registry loads
    // the whole snapshot.
    let offload = campaign_contract
        .as_ref()
        .map(|(_, _, policy)| *policy)
        .unwrap_or_default();
    // SW-6137: the opt-in Sol-Attn engine option.
    let sol_attention = arg(&args, "--sol-topk")
        .map(|ratio| -> Result<candle_gen::SolConfig> {
            let mut cfg = candle_gen::SolConfig::top_k(ratio.parse()?);
            if let Some(sinks) = arg(&args, "--sol-sinks") {
                let (a, b) = sinks.split_once(':').ok_or("--sol-sinks <start>:<end>")?;
                cfg.sink_blocks = (a.parse()?, b.parse()?);
            }
            Ok(cfg)
        })
        .transpose()?;
    println!("[smoke] sol_attention={sol_attention:?}");
    // `--sol-sinks` only modifies `--sol-topk`, and `--attn-capture` captures an opt-in-routed call
    // (this example's only opt-in is `--sol-topk`): without it either flag would do nothing.
    for dependent in ["--sol-sinks", "--attn-capture"] {
        if sol_attention.is_none() && arg(&args, dependent).is_some() {
            return Err(format!("{dependent} needs --sol-topk <ratio>").into());
        }
    }
    if let Some(capture) = arg(&args, "--attn-capture") {
        let (skip, _) = capture
            .split_once(':')
            .ok_or("--attn-capture <skip>:<file>")?;
        candle_gen::fused_attention::arm_int8_operand_capture(skip.parse()?);
    }
    let gen = match (arg(&args, "--comfyui-high"), arg(&args, "--comfyui-low")) {
        (Some(high), Some(low)) => {
            // `load_from_comfyui_experts` takes no adapters, so `--lora-high/--lora-low` would be
            // silently dropped in ComfyUI mode — invalidating any sc-10671-in-place + Lightning-distill
            // A/B (F-127). Reject the combination rather than rendering unadapted.
            if !adapters.is_empty() {
                return Err(
                    "--lora-high/--lora-low are not supported alongside --comfyui-high/--comfyui-low \
                     (the ComfyUI in-place loader takes no adapters); drop one flag set"
                        .into(),
                );
            }
            // sc-10909: optional in-place UMT5 TE / Wan VAE; whichever is omitted falls back to snapshot.
            let te_file = arg(&args, "--comfyui-te").map(PathBuf::from);
            let vae_file = arg(&args, "--comfyui-vae").map(PathBuf::from);
            println!(
                "[smoke] comfyui experts: high={high} low={low} (in place, scaled-fp8→bf16)\n\
                 [smoke] comfyui te={te_file:?} vae={vae_file:?} (in place when Some, else snapshot)"
            );
            // The engine options reach this lane too (`--sol-topk`), as on the snapshot lane.
            candle_gen_wan::wan14b::load_from_comfyui_experts_with_options(
                PathBuf::from(&high),
                PathBuf::from(&low),
                te_file,
                vae_file,
                PathBuf::from(&snapshot),
                false,
                offload,
                candle_gen_wan::wan14b::Wan14bEngineOptions { sol_attention },
            )?
        }
        _ => {
            let spec = LoadSpec::new(WeightsSource::Dir(PathBuf::from(&snapshot)))
                .with_adapters(adapters)
                .with_offload_policy(offload);
            match sol_attention {
                None => candle_gen_wan::provider_registry()?.load("wan2_2_t2v_14b", &spec)?,
                Some(cfg) => candle_gen_wan::wan14b::load_t2v_14b_with_options(
                    &spec,
                    candle_gen_wan::wan14b::Wan14bEngineOptions {
                        sol_attention: Some(cfg),
                    },
                )?,
            }
        }
    };
    println!(
        "[smoke] resolved engine id={} backend={} modality={:?}",
        gen.descriptor().id,
        gen.descriptor().backend,
        gen.descriptor().modality
    );

    let req = GenerationRequest {
        prompt,
        negative_prompt: negative,
        width,
        height,
        count: 1,
        seed: Some(seed),
        steps,
        guidance,
        frames,
        fps,
        sampler,
        ..Default::default()
    };

    let mut step_marks: Vec<std::time::Instant> = Vec::new();
    let mut on_progress = |p: Progress| match p {
        Progress::Step { current, total } => {
            step_marks.push(std::time::Instant::now());
            print!("\r[smoke] step {current}/{total}   ");
            use std::io::Write;
            let _ = std::io::stdout().flush();
        }
        Progress::Decoding => println!("\n[smoke] decoding"),
        Progress::Loading(phase) => println!("\n[smoke] loading {phase:?}"),
        Progress::PromptUpsamplingComplete
        | Progress::PromptEncodingComplete
        | Progress::PostProcessing
        | Progress::ConditioningComplete
        | Progress::Upsampling { .. } => {}
    };
    #[cfg(feature = "cuda")]
    let pool = candle_gen::cuda_mempool::MemPool::device_current(0);
    #[cfg(feature = "cuda")]
    if let Some(pool) = &pool {
        pool.reset_used_high_water();
    }
    let paths_before = candle_gen::attention_path_counts();
    let t0 = std::time::Instant::now();
    let output = match gen.generate(&req, &mut on_progress) {
        Ok(output) => output,
        Err(_error)
            if args.iter().any(|arg| arg == "--sc20686-cancel")
                && candle_gen_wan::sc20686_observer::campaign_cancelled() =>
        {
            println!("[smoke] expected SC-20686 campaign cancellation");
            return Ok(());
        }
        Err(error) => return Err(error.into()),
    };
    let secs = t0.elapsed().as_secs_f32();
    let paths = candle_gen::attention_path_counts().since(&paths_before);
    println!(
        "\n[smoke] attention paths: flash={} sol={} int8={} budgeted={} sol_refusals={:?}",
        paths.flash,
        paths.sol,
        paths.int8,
        paths.budgeted(),
        paths.sol_refusals
    );
    // Step k's mark is when step k finished; the mean interval over steps 2..n excludes the
    // first step's lazy load / conditioning.
    if step_marks.len() >= 2 {
        let span = step_marks[step_marks.len() - 1]
            .duration_since(step_marks[0])
            .as_secs_f64();
        println!(
            "[smoke] denoise: {:.3} s/step (mean of {} step intervals)",
            span / (step_marks.len() - 1) as f64,
            step_marks.len() - 1
        );
    }
    #[cfg(feature = "cuda")]
    if let Some(pool) = &pool {
        println!(
            "[smoke] CUDA pool peak (USED_MEM_HIGH): {:.2} GiB",
            pool.used_high().unwrap_or(0) as f64 / (1u64 << 30) as f64
        );
    }
    if let Some(capture) = arg(&args, "--attn-capture") {
        let (_, file) = capture.split_once(':').ok_or("--attn-capture")?;
        match candle_gen::fused_attention::take_int8_operand_capture() {
            Some(c) => {
                let scale = candle_gen::candle_core::Tensor::new(&[c.scale as f32], c.q.device())?;
                candle_gen::candle_core::safetensors::save(
                    &std::collections::HashMap::from([
                        ("q".to_string(), c.q.contiguous()?),
                        ("k".to_string(), c.k.contiguous()?),
                        ("v".to_string(), c.v.contiguous()?),
                        ("scale".to_string(), scale),
                    ]),
                    file,
                )?;
                println!("[smoke] captured attention operands -> {file}");
            }
            None => println!("[smoke] no opt-in attention call was captured"),
        }
    }
    let (frames, fps) = match output {
        GenerationOutput::Video { frames, fps, .. } => (frames, fps),
        _ => return Err("expected video, got images".into()),
    };
    println!("[smoke] {} frame(s) @ {fps}fps in {secs:.1}s", frames.len());

    std::fs::create_dir_all(&out)?;
    for (i, f) in frames.iter().enumerate() {
        let buf = image::RgbImage::from_raw(f.width, f.height, f.pixels.clone())
            .ok_or("invalid RGB buffer dimensions")?;
        buf.save(PathBuf::from(&out).join(format!("frame_{i:03}.png")))?;
    }
    println!(
        "[smoke] wrote {} frames to {}/ ({}x{})",
        frames.len(),
        out,
        frames[0].width,
        frames[0].height
    );
    Ok(())
}
