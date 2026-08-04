//! sc-16959 — the SANA preview seam driven through **both** shared samplers on real (tiny) trunks,
//! on CPU (epic 16948).
//!
//! `src/preview.rs`'s unit tests exercise the projectors and hooks directly, and
//! `tests/preview_real_weights.rs` renders the two shipped routes on CUDA. This file covers the part
//! neither can: the wiring *through the drivers*, with a real trunk in the loop, in the lane that runs
//! on every PR.
//!
//! It is the only place four of this story's acceptance criteria are checked without a GPU:
//!
//! * **Exactly one frame per outer step on a multi-eval solver** — the base lane under `heun`, with the
//!   evaluation count established as genuinely greater than the step count *first*, so the dedup
//!   assertion cannot pass vacuously.
//! * **The 1-step and 4-step Sprint schedules** — the SCM driver has no σ array and keys frames on the
//!   step index, and `num_steps == 1` is a real Sprint request rather than an edge case.
//! * **The SCM driver hands the hook a `σ_data`-scaled latent** — asserted against the seed latent the
//!   caller passed in, which is what makes the `1/σ_data` correction in `crate::preview` necessary
//!   rather than decorative.
//! * **CFG never reaches the preview** — every latent the base hook sees is batch-1 in the trunk's own
//!   channel width, never a fused `[2, …]` pair.
//!
//! Both trunks are the committed tiny goldens `transformer_parity.rs` numerically validates, so the
//! denoise in the loop is the shipped one rather than a stub.

use std::collections::HashMap;
use std::sync::{Arc, Mutex};

use candle_gen::candle_core::{DType, Device, Tensor};
use candle_gen::gen_core::{CancelFlag, Image, PreviewFrame, PreviewSink, Progress};
use candle_gen::preview::PreviewHook;
use candle_gen::{ScmScheduler, Weights, SCM_SIGMA_DATA};
use candle_gen_sana::pipeline::{denoise_cfg, sana_sigmas};
use candle_gen_sana::{denoise_sprint, SanaTransformer, SanaTransformerConfig};

/// The tiny base config `transformer_parity.rs` validates its golden against.
fn tiny_base_config() -> SanaTransformerConfig {
    SanaTransformerConfig {
        in_channels: 4,
        out_channels: 4,
        num_attention_heads: 2,
        attention_head_dim: 8,
        num_layers: 2,
        num_cross_attention_heads: 2,
        cross_attention_head_dim: 8,
        caption_channels: 24,
        mlp_ratio: 2.5,
        patch_size: 1,
        norm_eps: 1e-6,
        caption_norm_eps: 1e-5,
        attn_qk_norm_eps: 1e-5,
        attn_eps: 1e-15,
        guidance_embeds: false,
        guidance_embeds_scale: 0.1,
        qk_norm: false,
    }
}

/// The tiny Sprint config: the base plus the guidance embedder and qk-norm.
fn tiny_sprint_config() -> SanaTransformerConfig {
    SanaTransformerConfig {
        guidance_embeds: true,
        qk_norm: true,
        ..tiny_base_config()
    }
}

/// Build a trunk from a committed golden's `w.`-prefixed weights.
fn trunk_from_golden(file: &str, config: SanaTransformerConfig) -> SanaTransformer {
    let path = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("tests")
        .join("fixtures")
        .join(file);
    let golden = Weights::from_file(&path, &Device::Cpu, DType::F32).expect("load the tiny golden");
    let mut map = HashMap::new();
    for key in golden.keys() {
        if let Some(rest) = key.strip_prefix("w.") {
            map.insert(
                rest.to_string(),
                golden.require(key).expect("golden tensor"),
            );
        }
    }
    SanaTransformer::from_weights(&Weights::from_map(map), config).expect("build the tiny trunk")
}

fn base_trunk() -> SanaTransformer {
    trunk_from_golden("sana_transformer_golden.safetensors", tiny_base_config())
}

fn sprint_trunk() -> SanaTransformer {
    trunk_from_golden("sana_sprint_trunk_golden.safetensors", tiny_sprint_config())
}

/// Deterministic pseudo-random fill (LCG) — reproducible, no rand dep.
fn det(shape: &[usize], seed: u64) -> Tensor {
    let n: usize = shape.iter().product();
    let mut s = seed.wrapping_mul(2_654_435_761).wrapping_add(1);
    let mut v = Vec::with_capacity(n);
    for _ in 0..n {
        s = s
            .wrapping_mul(6_364_136_223_846_793_005)
            .wrapping_add(1_442_695_040_888_963_407);
        v.push((((s >> 33) as f64) / ((1u64 << 31) as f64) - 1.0) as f32);
    }
    Tensor::from_vec(v, shape, &Device::Cpu).expect("deterministic tensor")
}

fn values(t: &Tensor) -> Vec<f32> {
    t.flatten_all()
        .expect("flatten")
        .to_vec1::<f32>()
        .expect("f32 values")
}

fn collecting_sink() -> (PreviewSink, Arc<Mutex<Vec<PreviewFrame>>>) {
    let frames = Arc::new(Mutex::new(Vec::new()));
    let captured = Arc::clone(&frames);
    let sink = PreviewSink::new(move |frame| candle_gen::lock_recover(&captured).push(frame));
    (sink, frames)
}

/// One latent as the driver handed it to the hook: its shape and its values.
type SeenLatent = (Vec<usize>, Vec<f32>);
/// The shared record every row's projector appends to.
type SeenLatents = Arc<Mutex<Vec<SeenLatent>>>;

/// A projector that records the exact tensor the driver handed it and emits a 1×1 frame.
///
/// The tiny trunks are 4-channel, so this crate's own 32-channel projectors would (correctly) refuse
/// every latent and the frames would all be swallowed. Recording the *shapes and values* the driver
/// passes is what this file is measuring, and it is the same tensor the shipped projector would see.
fn recording_projector(seen: &SeenLatents) -> impl Fn(&Tensor) -> candle_gen::Result<Image> + '_ {
    let recorded = Arc::clone(seen);
    move |latents: &Tensor| {
        candle_gen::lock_recover(&recorded).push((latents.dims().to_vec(), values(latents)));
        Ok(Image {
            width: 1,
            height: 1,
            pixels: vec![0, 0, 0],
        })
    }
}

// ── The base lane: `denoise_cfg` → `run_flow_sampler` ─────────────────────────────────────────────

/// Drive the base lane once and return `(evaluation count, denoised latent)`.
///
/// The caller owns the sink its `preview` hook was built over, so the frames come back through that
/// rather than through this helper.
fn run_base(sampler: Option<&str>, steps: usize, preview: &PreviewHook<'_>) -> (usize, Tensor) {
    let config = tiny_base_config();
    let trunk = base_trunk();
    let latents = det(&[1, config.out_channels as usize, 6, 4], 0);
    let cond = det(&[1, 7, config.caption_channels as usize], 100);
    let uncond = det(&[1, 7, config.caption_channels as usize], 101);
    let cancel = CancelFlag::default();
    let mut evaluations = 0usize;
    let out = denoise_cfg(
        &trunk,
        &sana_sigmas(None, steps),
        sampler,
        7,
        latents,
        &cond,
        Some(&uncond),
        4.5,
        &Device::Cpu,
        &cancel,
        &mut |p: Progress| {
            if matches!(p, Progress::Step { .. }) {
                evaluations += 1;
            }
        },
        preview,
    )
    .expect("base flow denoise");
    (evaluations, out)
}

/// Euler evaluates once per outer step, so the base lane emits exactly one frame per step, numbered
/// `1..=steps` with `total == steps`.
#[test]
fn the_base_lane_emits_one_frame_per_euler_step() {
    let steps = 6usize;
    let (sink, frames) = collecting_sink();
    let seen = Arc::new(Mutex::new(Vec::new()));
    let hook = PreviewHook::new(&sink, recording_projector(&seen));

    let (evaluations, _) = run_base(None, steps, &hook);

    let frames = candle_gen::lock_recover(&frames);
    assert_eq!(evaluations, steps, "Euler evaluates once per outer step");
    assert_eq!(
        frames
            .iter()
            .map(|f| (f.current, f.total))
            .collect::<Vec<_>>(),
        (1..=steps as u32)
            .map(|n| (n, steps as u32))
            .collect::<Vec<_>>()
    );
}

/// **One frame per outer step on a multi-eval solver**, proven non-vacuous first.
///
/// `heun` evaluates the model twice per outer step. The shared driver calls `on_progress` once per
/// *evaluation* and deliberately repeats the step number, so counting `Progress::Step` events IS
/// counting evaluations. This row asserts there are **more** evaluations than outer steps before
/// asserting the frames collapsed to exactly the outer steps — a solver that turned out to evaluate
/// once per step would make the frame-count assertion prove nothing about dedup.
#[test]
fn a_multi_eval_solver_still_emits_one_frame_per_outer_step() {
    let steps = 5usize;
    let (sink, frames) = collecting_sink();
    let seen = Arc::new(Mutex::new(Vec::new()));
    let hook = PreviewHook::new(&sink, recording_projector(&seen));

    let (evaluations, _) = run_base(Some("heun"), steps, &hook);

    assert!(
        evaluations > steps,
        "heun must evaluate more than once per outer step or this row proves nothing about dedup \
         ({evaluations} evaluations for {steps} steps)"
    );
    let frames = candle_gen::lock_recover(&frames);
    assert_eq!(
        frames.len(),
        steps,
        "a multi-eval solver must still emit exactly one frame per outer step ({evaluations} \
         evaluations)"
    );
    assert_eq!(
        frames
            .iter()
            .map(|f| (f.current, f.total))
            .collect::<Vec<_>>(),
        (1..=steps as u32)
            .map(|n| (n, steps as u32))
            .collect::<Vec<_>>()
    );
}

/// **CFG never reaches the preview.** Base SANA runs the cond and uncond trunk forwards *inside*
/// `denoise_cfg`'s predict closure and blends them before the solver advances, so every latent the
/// hook sees is one batch-1 tensor in the trunk's own channel width — never a fused `[2, …]` pair.
#[test]
fn the_base_preview_never_sees_a_fused_unconditional_half() {
    let config = tiny_base_config();
    let (sink, _frames) = collecting_sink();
    let seen = Arc::new(Mutex::new(Vec::new()));
    let hook = PreviewHook::new(&sink, recording_projector(&seen));

    run_base(None, 4, &hook);

    let seen = candle_gen::lock_recover(&seen);
    assert_eq!(seen.len(), 4);
    for (dims, _) in seen.iter() {
        assert_eq!(
            dims,
            &vec![1, config.out_channels as usize, 6, 4],
            "the base hook must see one batch-1 latent, not a fused cond/uncond batch"
        );
    }
}

/// N1 for the base seam: an inert sink leaves the denoised latent byte-identical to an active one at
/// the same seed, and does no projection at all.
#[test]
fn the_base_lane_is_byte_identical_under_an_inert_sink() {
    let seen = Arc::new(Mutex::new(Vec::new()));

    let inert = PreviewSink::default();
    let inert_hook = PreviewHook::new(&inert, recording_projector(&seen));
    let (_, quiet) = run_base(None, 4, &inert_hook);
    assert!(
        candle_gen::lock_recover(&seen).is_empty(),
        "an inert sink must not invoke the projector"
    );

    let (active, _frames) = collecting_sink();
    let active_hook = PreviewHook::new(&active, recording_projector(&seen));
    let (_, watched) = run_base(None, 4, &active_hook);
    assert_eq!(candle_gen::lock_recover(&seen).len(), 4);

    assert_eq!(
        values(&quiet),
        values(&watched),
        "an active preview sink must not change a single denoised value"
    );
}

// ── The Sprint lane: `denoise_sprint` → `run_scm_sampler` ─────────────────────────────────────────

/// Drive the Sprint lane once and return the denoised latent.
fn run_sprint(steps: usize, preview: &PreviewHook<'_>) -> (Tensor, Tensor) {
    let config = tiny_sprint_config();
    let trunk = sprint_trunk();
    let seed_latents = det(&[1, config.out_channels as usize, 6, 4], 3);
    let cond = det(&[1, 5, config.caption_channels as usize], 11);
    let cancel = CancelFlag::default();
    let out = denoise_sprint(
        &trunk,
        &ScmScheduler::new(steps),
        7,
        seed_latents.clone(),
        &cond,
        4.5,
        config.guidance_embeds_scale,
        &Device::Cpu,
        &cancel,
        &mut |_| {},
        preview,
    )
    .expect("Sprint SCM denoise");
    (out, seed_latents)
}

/// The Sprint lane emits exactly one frame per SCM step across the whole 1–4 operating band, keyed on
/// the **step index** — the SCM driver has no σ array for the sigma-keyed counter to search.
///
/// The `steps == 1` row is the one the story names explicitly: one frame, `current == 1`, `total == 1`,
/// no division by zero and no stall.
#[test]
fn the_sprint_lane_emits_one_frame_per_scm_step_including_the_single_step_schedule() {
    for steps in [1usize, 2, 3, 4] {
        let (sink, frames) = collecting_sink();
        let seen = Arc::new(Mutex::new(Vec::new()));
        let hook = PreviewHook::new(&sink, recording_projector(&seen));

        run_sprint(steps, &hook);

        let frames = candle_gen::lock_recover(&frames);
        assert_eq!(
            frames
                .iter()
                .map(|f| (f.current, f.total))
                .collect::<Vec<_>>(),
            (1..=steps as u32)
                .map(|n| (n, steps as u32))
                .collect::<Vec<_>>(),
            "{steps}-step Sprint numbering"
        );
    }
}

/// The single-step schedule, called out on its own because it is the degenerate case: `ScmScheduler`
/// reports `is_single_step()`, the driver skips the renoise, and the counter's `total` is 1.
#[test]
fn a_single_step_sprint_render_emits_exactly_one_frame_and_does_not_stall() {
    let scheduler = ScmScheduler::new(1);
    assert!(scheduler.is_single_step());
    assert_eq!(scheduler.num_steps(), 1);

    let (sink, frames) = collecting_sink();
    let seen = Arc::new(Mutex::new(Vec::new()));
    let hook = PreviewHook::new(&sink, recording_projector(&seen));
    let (out, _) = run_sprint(1, &hook);

    let frames = candle_gen::lock_recover(&frames);
    assert_eq!(frames.len(), 1);
    assert_eq!((frames[0].current, frames[0].total), (1, 1));
    assert!(
        values(&out).iter().all(|v| v.is_finite()),
        "a single-step Sprint render must produce a finite latent"
    );
}

/// **The `σ_data` scale the Sprint projector has to undo**, measured against the caller's own seed
/// latent rather than restated from the driver's docs.
///
/// `run_scm_sampler` multiplies the seed latent by `σ_data` on entry and hands the hook that scaled
/// tensor, which is why `crate::preview::sprint_hook` carries `1/σ_data`. If this row ever fails
/// because the driver stopped pre-scaling, the correction must go with it.
#[test]
fn the_scm_driver_hands_the_hook_the_sigma_data_scaled_latent() {
    let (sink, _frames) = collecting_sink();
    let seen = Arc::new(Mutex::new(Vec::new()));
    let hook = PreviewHook::new(&sink, recording_projector(&seen));

    let (_, seed_latents) = run_sprint(4, &hook);

    let seen = candle_gen::lock_recover(&seen);
    assert_eq!(seen.len(), 4);
    let expected = values(
        &seed_latents
            .affine(SCM_SIGMA_DATA as f64, 0.0)
            .expect("sigma_data pre-scale"),
    );
    assert_eq!(
        seen[0].1, expected,
        "step 0 must see the seed latent scaled by sigma_data ({SCM_SIGMA_DATA})"
    );
    // And it is genuinely different from the unscaled seed, or the correction would be a no-op.
    assert_ne!(seen[0].1, values(&seed_latents));
}

/// N1 for the SCM seam: an inert sink leaves the denoised latent byte-identical and projects nothing.
#[test]
fn the_sprint_lane_is_byte_identical_under_an_inert_sink() {
    let seen = Arc::new(Mutex::new(Vec::new()));

    let inert = PreviewSink::default();
    let (quiet, _) = run_sprint(4, &PreviewHook::new(&inert, recording_projector(&seen)));
    assert!(candle_gen::lock_recover(&seen).is_empty());

    let (active, _frames) = collecting_sink();
    let (watched, _) = run_sprint(4, &PreviewHook::new(&active, recording_projector(&seen)));
    assert_eq!(candle_gen::lock_recover(&seen).len(), 4);

    assert_eq!(values(&quiet), values(&watched));
}

/// A projector that always fails costs neither render anything — previews are decorative on both
/// drivers, and a projection error is swallowed rather than surfaced.
#[test]
fn a_failing_projector_never_fails_either_render() {
    let (base_sink, base_frames) = collecting_sink();
    let base_hook = PreviewHook::new(&base_sink, |_: &Tensor| {
        Err(candle_gen::CandleError::Msg("synthetic failure".into()))
    });
    let (_, base_out) = run_base(None, 4, &base_hook);
    assert!(values(&base_out).iter().all(|v| v.is_finite()));
    assert!(candle_gen::lock_recover(&base_frames).is_empty());

    let (sprint_sink, sprint_frames) = collecting_sink();
    let sprint_hook = PreviewHook::new(&sprint_sink, |_: &Tensor| {
        Err(candle_gen::CandleError::Msg("synthetic failure".into()))
    });
    let (sprint_out, _) = run_sprint(4, &sprint_hook);
    assert!(values(&sprint_out).iter().all(|v| v.is_finite()));
    assert!(candle_gen::lock_recover(&sprint_frames).is_empty());
}
