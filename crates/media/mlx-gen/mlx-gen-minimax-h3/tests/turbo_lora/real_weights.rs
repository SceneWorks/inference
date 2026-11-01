//! Real-weight support: the published turbo set and the measured-render harness.

use super::*;

// ─── real weights ──────────────────────────────────────────────────────────────────────────────

/// The published `lightx2v/Minimax-h3-Turbo` set, from `MINIMAX_H3_TURBO_LORA`.
pub(super) fn turbo_dir() -> PathBuf {
    let raw = std::env::var("MINIMAX_H3_TURBO_LORA").unwrap_or_default();
    assert!(
        !raw.is_empty(),
        "MINIMAX_H3_TURBO_LORA must point at a downloaded `lightx2v/Minimax-h3-Turbo` snapshot dir"
    );
    let dir = PathBuf::from(raw);
    assert!(
        dir.is_dir(),
        "MINIMAX_H3_TURBO_LORA={} is not a dir",
        dir.display()
    );
    dir
}

// ─── sc-18729: the measured render ─────────────────────────────────────────────────────────────
//
// The fold gates above prove the LoRA lands on the right 312 modules at the right strength. They
// say nothing about whether a 4-step schedule actually produces a clip, what it costs, or what it
// costs in quality — which is the whole reason the toggle exists. This section renders.
//
// **The sampling recipe is documented, not guessed.** `lightx2v/Minimax-h3-Turbo`'s README points
// at `github.com/ModelTC/Minimax-H3-Turbo`, whose model-specs table lists per-variant training
// shifts (video / audio) and recommended inference steps, and whose
// `DIFFUSERS_SETUP_AND_INFERENCE.md` gives the invocation verbatim:
//
// ```text
// python inference_minimax_h3.py --jobs-json examples/prompts_t2va_test.json \
//   --lora-path minimax_h3_fl2v_turbo_4step_v1.0_768p_bf16.safetensors \
//   --inference-steps 4 --video-shift 6 --lora-alpha 128 \
//   --megapixels 1.0 --aspect-ratio 16:9 --output-dir outputs/lora_4nfe_768p
// ```
//
// | variant | steps | video / audio shift |
// |---|---|---|
// | base | 50 | 12 / 3 |
// | `4step_v0.1`, `8step_v1.0` (544p) | 4 / 8 | 12 / 3 |
// | `4step_v1.0_768p` | 4 | **6** / 3 |
//
// `--lora-alpha 128` is the reference script working around its own hardcoded default of 8; this
// engine resolves alpha from the file's `__metadata__`, so it is not a knob here and must not
// become one (sc-18724). The audio shift never moves across the published set, which is why
// `scheduler_shift` overrides the video half only.

/// Env var, empty-as-absent.
pub(super) fn render_env(key: &str) -> Option<String> {
    std::env::var(key).ok().filter(|v| !v.trim().is_empty())
}

pub(super) fn render_env_f32(key: &str) -> Option<f32> {
    render_env(key).map(|v| {
        v.parse()
            .unwrap_or_else(|e| panic!("{key}={v} is not a float: {e}"))
    })
}

pub(super) fn render_env_u32(key: &str) -> Option<u32> {
    render_env(key).map(|v| {
        v.parse()
            .unwrap_or_else(|e| panic!("{key}={v} is not an integer: {e}"))
    })
}

/// What one render cost and produced.
pub(super) struct RenderReceipt {
    pub(super) frames: Vec<mlx_gen::gen_core::Image>,
    pub(super) audio: mlx_gen::gen_core::AudioTrack,
    pub(super) fps: u32,
    /// `generate` entry → **first** `Progress::Step`: the text-encoder map, the prompt encode, the
    /// TE release, the DiT map, the **LoRA fold**, the AdaLN precompute-and-evict — **and the first
    /// model evaluation**, because `render_latents` reports a step only once that evaluation has
    /// finished. One bucket because that is the granularity `Progress` exposes on the `Resident`
    /// path; H3 emits no `Loading` phases.
    pub(super) setup_and_first_eval_s: f64,
    /// First → last `Progress::Step`. This spans `steps − 1` evaluations, **not** `steps` — the
    /// off-by-one that would otherwise understate `s_per_step` by 25 % on a 4-step render.
    pub(super) denoise_tail_s: f64,
    /// `Progress::Decoding` → return: video VAE + audio VAE + the A/V fit.
    pub(super) decode_s: f64,
    /// Seconds per model evaluation, over the `steps − 1` intervals actually bounded by two ticks.
    /// `None` for a single-step render, where no interval exists and any figure would be invented.
    pub(super) s_per_step: Option<f64>,
    /// Wall-clock of each interval, in order — so a warm-up outlier is visible rather than averaged
    /// into a headline number.
    pub(super) step_intervals_s: Vec<f64>,
    pub(super) wall_s: f64,
    pub(super) peak_bytes: u64,
    pub(super) steps_seen: u32,
}

/// Everything one arm of the comparison varies. A struct rather than a parameter list because the
/// arms differ in one or two fields at a time, and a positional call site of eight values is how a
/// canvas ends up transposed against a frame count with nothing to catch it.
pub(super) struct RenderRecipe<'a> {
    /// The upstream snapshot root — text encoder, tokenizer, both VAEs.
    pub(super) root: &'a Path,
    /// The staged tier's `transformer/`, or `None` for a flat `root/transformer` install.
    pub(super) dit: Option<&'a Path>,
    /// The turbo file, or `None` for the base arm.
    pub(super) lora: Option<&'a Path>,
    pub(super) steps: u32,
    /// The **video** sigma shift. `None` keeps the base checkpoint's published 12.0.
    pub(super) shift: Option<f32>,
    pub(super) width: u32,
    pub(super) height: u32,
    pub(super) frames: u32,
    pub(super) seed: u64,
    pub(super) prompt: &'a str,
}

/// Render once at an explicit recipe, staging the tier exactly as a split install does.
///
/// `recipe.lora` is `None` for the base arm. When `Some`, the alpha is resolved from the file header
/// here too — printed, not passed — so the receipt records the strength the engine independently
/// resolved rather than one this harness supplied.
pub(super) fn render_once(recipe: &RenderRecipe<'_>) -> RenderReceipt {
    let &RenderRecipe {
        root,
        dit,
        lora,
        steps,
        shift,
        width,
        height,
        frames,
        seed,
        prompt,
    } = recipe;
    use mlx_gen::gen_core::{
        CancelFlag, GenerationOutput, GenerationRequest, LoadSpec, Progress, WeightsSource,
    };
    use mlx_gen_minimax_h3::model::{load, DIT_COMPONENT};

    let mut load_spec = LoadSpec::new(WeightsSource::Dir(root.to_path_buf()));
    if let Some(d) = dit {
        load_spec = load_spec.with_component(DIT_COMPONENT, WeightsSource::Dir(d.to_path_buf()));
    }
    // `spec.quantize` is deliberately left unset: `reconcile_tier` treats the on-disk marker as
    // authoritative, and asserting `Q4` would additionally demand a q4-marked `transformer_ref`,
    // which the `t2va` path never reads. The tier that loads is still whatever is staged.
    if let Some(p) = lora {
        let w = Weights::from_file(p).unwrap_or_else(|e| panic!("read {}: {e}", p.display()));
        let alpha = resolve_alpha(&w).unwrap_or_else(|e| panic!("alpha: {e}"));
        println!(
            "[render] adapter {} — {} tensors, resolved alpha {alpha}, fold {} at rank 128",
            p.file_name().unwrap_or_default().to_string_lossy(),
            w.len(),
            alpha_rank_fold(alpha, 128.0)
        );
        assert!(
            !is_comfyui_key_space(&w),
            "the ComfyUI twin fuses qkv and swaps the SwiGLU halves; use the diffusers file"
        );
        load_spec = load_spec.with_adapters(vec![spec(p.to_path_buf(), 1.0)]);
    }
    let model = load(&load_spec).unwrap_or_else(|e| panic!("load: {e}"));

    let req = GenerationRequest {
        prompt: prompt.into(),
        width,
        height,
        frames: Some(frames),
        steps: Some(steps),
        scheduler_shift: shift,
        seed: Some(seed),
        cancel: CancelFlag::default(),
        ..Default::default()
    };
    model.validate(&req).expect("the recipe must validate");

    mlx_rs::memory::clear_cache();
    mlx_rs::memory::reset_peak_memory();
    let started = Instant::now();
    // Every tick, not just the first and last: `s_per_step` is then an average over intervals that
    // really exist, and a warm-up outlier stays visible instead of being smeared into the headline.
    let mut ticks: Vec<Instant> = Vec::new();
    let mut decoding_at: Option<Instant> = None;
    let mut steps_seen = 0u32;
    let out = model
        .generate(&req, &mut |p| match p {
            Progress::Step { current, total } => {
                steps_seen += 1;
                let now = Instant::now();
                if let Some(prev) = ticks.last() {
                    println!(
                        "[render]   step {current}/{total} (+{:.1} s)",
                        now.duration_since(*prev).as_secs_f64()
                    );
                } else {
                    println!(
                        "[render]   step {current}/{total} — setup + first eval took {:.1} s",
                        now.duration_since(started).as_secs_f64()
                    );
                }
                ticks.push(now);
            }
            Progress::Decoding => decoding_at = Some(Instant::now()),
            Progress::Loading(_)
            | Progress::PromptUpsamplingComplete
            | Progress::PromptEncodingComplete
            | Progress::PostProcessing
            | Progress::ConditioningComplete
            | Progress::Upsampling { .. } => {}
        })
        .unwrap_or_else(|e| panic!("generate: {e}"));
    let wall_s = started.elapsed().as_secs_f64();
    // Read the peak only after `generate` has returned. Every product below is a **host** buffer —
    // `Vec<u8>` pixels and `Vec<f32>` samples — which cannot exist unless the whole graph was
    // materialized inside the timed region. That is what makes this peak meaningful: the epic's
    // recorded trap is a bare `MiniMaxH3Dit::load` reporting 33 KB because MLX mmaps lazily and
    // nothing forced the tensors. Nothing here is unforced.
    let peak_bytes = mlx_rs::memory::get_peak_memory() as u64;

    let (frames, fps, audio) = match out {
        GenerationOutput::Video { frames, fps, audio } => (frames, fps, audio),
        other => panic!("expected a Video output, got {other:?}"),
    };
    let audio = audio.expect("MiniMax-H3 always produces a soundtrack");

    let first = *ticks
        .first()
        .expect("the denoise must report at least one step");
    let last = *ticks
        .last()
        .expect("the denoise must report at least one step");
    let dec = decoding_at.expect("the decode phase must be reported");
    let step_intervals_s: Vec<f64> = ticks
        .windows(2)
        .map(|w| w[1].duration_since(w[0]).as_secs_f64())
        .collect();
    let denoise_tail_s = last.duration_since(first).as_secs_f64();
    RenderReceipt {
        setup_and_first_eval_s: first.duration_since(started).as_secs_f64(),
        denoise_tail_s,
        decode_s: wall_s - dec.duration_since(started).as_secs_f64(),
        // Divided by the interval count, which is `steps − 1`. Dividing by `steps` is the
        // 25 %-at-4-steps error this harness must not make.
        s_per_step: (!step_intervals_s.is_empty())
            .then(|| denoise_tail_s / step_intervals_s.len() as f64),
        step_intervals_s,
        wall_s,
        peak_bytes,
        steps_seen,
        frames,
        audio,
        fps,
    }
}

/// Per-frame stddev and inter-frame motion, on the 0-255 scale — the "is this a picture, and is it
/// one scene" evidence. Same statistic `quant_tiers_real` gates tiers on, so the rows compare.
pub(super) fn clip_stats(frames: &[mlx_gen::gen_core::Image]) -> (f64, f64) {
    let mut sd_sum = 0.0;
    let mut motion_sum = 0.0;
    for (i, f) in frames.iter().enumerate() {
        let n = f.pixels.len() as f64;
        let mean = f.pixels.iter().map(|&p| f64::from(p)).sum::<f64>() / n;
        sd_sum += (f
            .pixels
            .iter()
            .map(|&p| (f64::from(p) - mean).powi(2))
            .sum::<f64>()
            / n)
            .sqrt();
        if i > 0 {
            let prev = &frames[i - 1].pixels;
            motion_sum += f
                .pixels
                .iter()
                .zip(prev.iter())
                .map(|(&a, &b)| (f64::from(a) - f64::from(b)).abs())
                .sum::<f64>()
                / n;
        }
    }
    (
        sd_sum / frames.len() as f64,
        motion_sum / (frames.len() - 1) as f64,
    )
}
