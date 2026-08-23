//! gen-core contract conformance for the candle SDXL provider (sc-4481, epic 3720).
//!
//! Runs the backend-neutral [`gen_core_testkit`] suite — validate-honesty, progress monotonicity,
//! typed cancellation, seed-determinism — against the real candle generator.
//! This is the suite whose **seed-determinism** check is the regression guard for the spike's
//! repro defect (sc-3498) that sc-3673 fixed.
//!
//! It drives a real `generate`, so it needs the CUDA backend + a local SDXL snapshot and is
//! `#[ignore]`d by default. On the Windows/Blackwell box (v143 vcvars + CUDA on PATH):
//!
//! ```text
//! set SDXL_SNAPSHOT=C:\Users\…\models--stabilityai--stable-diffusion-xl-base-1.0\snapshots\<hash>
//! cargo test -p candle-gen-sdxl --features cuda --release --test integration conformance:: -- --ignored
//! ```
//!
//! ## SDXL parity (sc-3677)
//!
//! Both [`sdxl_conformance`] and [`realvisxl_conformance`] run the SAME suite against the SAME
//! `candle_gen_sdxl::load` — the worker maps both ids onto one engine and RealVisXL_V5.0 shares the
//! SDXL architecture + diffusers component layout, so the only input that varies is the snapshot dir.
//! Passing the suite *is* the parity evidence the story asks for (AC: realvisxl generates a correct
//! image on the Candle lane; parity tests pass): it locks **output dims** (the request's WxH is the
//! emitted image size), **seed semantics** (same request+seed ⇒ byte-identical output;
//! [`check_seed_determinism`]), **scheduler/steps/guidance defaults** (`Step.total` == resolved
//! steps; [`check_progress`]), **contract/sidecar field shape** (validate-honesty +
//! [`gen_core::GenerationOutput::Images`]), and **cancellation/progress** (typed `Canceled`, monotone
//! `Step`).
//!
//! **Accepted differences vs the Python `SdxlDiffusersAdapter` (documented, not bugs):**
//! - **Sampler:** the candle lane runs **DDIM (eta=0)** and advertises only `ddim`; the Python/MLX
//!   default is `euler_ancestral`. sc-3673 chose DDIM for launch-portable determinism (the spike's
//!   ancestral path was non-reproducible across launches, sc-3498). Both are SDXL-correct solvers;
//!   cross-backend *pixel* equality is explicitly NOT a goal (RNG algorithms differ).
//! - **Surface:** txt2img only — conditioning and the lcm/hyper accel samplers are not advertised, so
//!   the worker keeps those shapes on the Python fallback (sc-3678) rather than the backend silently
//!   dropping a control. (sc-6128 DID wire the few-step `lightning` sampler; the testkit's
//!   validate-honesty check therefore now also exercises `validate(sampler="lightning")`, and
//!   [`realvisxl_lightning_render`] is the real-weight non-degeneracy guard for the render itself.)
//! - **dtype:** CLIP + UNet + VAE load f16 with the `madebyollin/sdxl-vae-fp16-fix` VAE; the VAE
//!   un-scale is the diffusers-correct 0.13025. These match diffusers' fp16 path, not a deviation.
#![cfg(feature = "cuda")]

use std::path::PathBuf;

use candle_gen::gen_core::{
    GenerationOutput, GenerationRequest, LoadSpec, Progress, WeightsSource,
};
use gen_core_testkit::{conformance, Profile};

/// Stage the three passed-in SDXL components (epic 13657 / sc-13663) onto a spec from explicit,
/// env-pointed local dirs (no hub fetch): the CLIP-L/bigG tokenizer dirs (`tokenizer.json`) and the
/// fp16-fix VAE dir (`diffusion_pytorch_model.safetensors`). Real-weight, so `#[ignore]`d.
fn with_sdxl_components(spec: LoadSpec) -> LoadSpec {
    let dir = |k: &str| {
        WeightsSource::Dir(PathBuf::from(
            std::env::var(k).unwrap_or_else(|_| panic!("set {k} to the component's local dir")),
        ))
    };
    spec.with_component("tokenizer_clip_l", dir("SDXL_TOKENIZER_CLIP_L_DIR"))
        .with_component("tokenizer_clip_bigg", dir("SDXL_TOKENIZER_CLIP_BIGG_DIR"))
        .with_component("vae_fp16_fix", dir("SDXL_VAE_FP16_FIX_DIR"))
}

#[test]
#[ignore = "needs SDXL_SNAPSHOT (a diffusers snapshot dir) + a CUDA GPU; run with --features cuda --ignored"]
fn sdxl_conformance() {
    let snap = std::env::var("SDXL_SNAPSHOT")
        .expect("set SDXL_SNAPSHOT to a stabilityai/stable-diffusion-xl-base-1.0 snapshot dir");
    let spec = with_sdxl_components(LoadSpec::new(WeightsSource::Dir(PathBuf::from(snap))));

    // 512² (≥ the descriptor's min_size 512) at a small step count keeps the suite's ~4 generate()
    // calls cheap — it verifies contract behavior, not image quality. `steps` must equal what the
    // model resolves req.steps to (check_progress asserts Step.total == profile.steps); the pipeline
    // uses req.steps verbatim, so 4 → 4.
    let profile = Profile {
        width: 512,
        height: 512,
        steps: 4,
        ..Profile::cheap()
    };

    // Resolve through this provider's direct loader. Panics with aggregated failures.
    conformance(|| candle_gen_sdxl::load(&spec).unwrap(), &profile);
}

/// sc-3677: the same conformance suite against a **RealVisXL_V5.0** snapshot — the parity evidence
/// that `realvisxl` generates a correct image on the Candle lane. RealVisXL ships the standard
/// diffusers tree with the SAME `.fp16.safetensors` component filenames this pipeline loads
/// (`unet/diffusion_pytorch_model.fp16.safetensors`, `text_encoder{,_2}/model.fp16.safetensors`), so
/// it resolves through the identical `candle_gen_sdxl::load` path — no single-file loader is needed
/// (the diffusers component layout is present, not absent). Only the snapshot env var differs from
/// [`sdxl_conformance`]. See the module header for the accepted differences vs the Python adapter.
///
/// ```text
/// set REALVISXL_SNAPSHOT=C:\Users\…\models--SG161222--RealVisXL_V5.0\snapshots\<hash>
/// cargo test -p candle-gen-sdxl --features cuda --release --test integration conformance:: -- --ignored
/// ```
#[test]
#[ignore = "needs REALVISXL_SNAPSHOT (a RealVisXL_V5.0 diffusers snapshot dir) + a CUDA GPU; run with --features cuda --ignored"]
fn realvisxl_conformance() {
    let snap = std::env::var("REALVISXL_SNAPSHOT")
        .expect("set REALVISXL_SNAPSHOT to an SG161222/RealVisXL_V5.0 snapshot dir");
    let spec = with_sdxl_components(LoadSpec::new(WeightsSource::Dir(PathBuf::from(snap))));

    // Same cheap 512²/4-step profile as sdxl_conformance — this verifies contract parity, not image
    // quality; the human-eyeball check is the txt2img example pointed at a RealVisXL snapshot.
    let profile = Profile {
        width: 512,
        height: 512,
        steps: 4,
        ..Profile::cheap()
    };

    conformance(|| candle_gen_sdxl::load(&spec).unwrap(), &profile);
}

/// sc-6128 acceptance: the candle SDXL lightning path renders a **non-degenerate** image at ~5 steps
/// via `sampler="lightning"` — the automatable half of "RealVisXL Lightning renders correctly on
/// Windows" (image *quality* is the human eyeball via `examples/sdxl-txt2img.rs --sampler lightning`).
///
/// Gated on `REALVISXL_LIGHTNING_SNAPSHOT` (a distilled RealVisXL Lightning / SDXL-Lightning diffusers
/// snapshot dir) — base SDXL through this sampler at 5 steps would render undertrained mush, which is
/// exactly the failure this story guards against, so the test demands a Lightning checkpoint. It
/// asserts the output dims, that progress reaches the 5th step, and that the pixels are not a flat
/// constant (a collapsed/blank decode), i.e. the few-step schedule actually produced structure.
///
/// ```text
/// set REALVISXL_LIGHTNING_SNAPSHOT=C:\Users\…\models--…--RealVisXL-Lightning\snapshots\<hash>
/// cargo test -p candle-gen-sdxl --features cuda --release --test integration -- --ignored conformance::realvisxl_lightning_render
/// ```
#[test]
#[ignore = "needs REALVISXL_LIGHTNING_SNAPSHOT (a distilled Lightning snapshot dir) + a CUDA GPU; run with --features cuda --ignored"]
fn realvisxl_lightning_render() {
    let snap = std::env::var("REALVISXL_LIGHTNING_SNAPSHOT").expect(
        "set REALVISXL_LIGHTNING_SNAPSHOT to a distilled RealVisXL Lightning / SDXL-Lightning \
         diffusers snapshot dir",
    );
    let spec = with_sdxl_components(LoadSpec::new(WeightsSource::Dir(PathBuf::from(snap))));
    let gen = candle_gen_sdxl::load(&spec).unwrap();

    let req = GenerationRequest {
        prompt: "a photo of a rusty robot holding a lit candle, cinematic lighting".into(),
        width: 512,
        height: 512,
        count: 1,
        seed: Some(42),
        steps: Some(5),
        // The worker forces this for `realvisxl_lightning`; CFG is off in the engine regardless.
        sampler: Some("lightning".into()),
        guidance: Some(1.0),
        ..Default::default()
    };

    let mut last_step = (0u32, 0u32);
    let mut on_progress = |p: Progress| {
        if let Progress::Step { current, total } = p {
            last_step = (current, total);
        }
    };
    let out = gen
        .generate(&req, &mut on_progress)
        .expect("lightning render");

    let images = match out {
        GenerationOutput::Images(imgs) => imgs,
        _ => panic!("expected images, got video"),
    };
    assert_eq!(images.len(), 1, "count=1 ⇒ one image");
    let img = &images[0];
    assert_eq!((img.width, img.height), (512, 512), "output dims = request");
    assert_eq!(img.pixels.len(), 512 * 512 * 3, "RGB8 buffer = W·H·3");
    // Progress reached the final (5th) step — the 5-step schedule actually ran.
    assert_eq!(last_step, (5, 5), "Step progress should end at 5/5");
    // Non-degenerate: a collapsed/blank decode is a flat constant. Real structure has spread.
    let min = *img.pixels.iter().min().unwrap();
    let max = *img.pixels.iter().max().unwrap();
    assert!(
        max - min > 16,
        "lightning render looks degenerate (flat): min={min} max={max}"
    );
}

/// sc-14195 acceptance, real weights: `guidanceScale = 1.0` on the **default** (curated `ddim`)
/// sampler renders a non-degenerate image instead of dying in the UNet's cross-attention matmul
/// (`shape mismatch in matmul, lhs: [10, 4096, 64], rhs: [20, 64, 77]`).
///
/// This is the story's own repro on a real checkpoint. It differs from
/// [`realvisxl_lightning_render`] in the axis that actually broke: Lightning has always taken its
/// own CFG-off path (it narrows the conditioning to the cond row itself), so it never exercised the
/// curated sampler's CFG-off fork — which is the one every non-`lightning` request at guidance ≤ 1
/// lands on, and the one that had no matching narrow.
///
/// It renders the CFG-off case on **both** routes the story's job could have taken — the explicit
/// `euler` the request named and the omitted-sampler default — plus a CFG-on arm, asserting the
/// default path still works and yields a *different* image ("default CFG behaviour unchanged").
///
/// Note this is the real-hardware **liveness** gate; it deliberately does not try to prove which
/// conditioning row was consumed, because a 4-step render gives no cheap oracle for that. The
/// row-identity pin (cond row, not the negative) is the CPU test
/// `pipeline::tests::render_at_guidance_one_runs_cfg_off_without_batch_mismatch`, which can compare
/// pixels against a controlled `[B, B]` / `[A, A]` conditioning stack.
///
/// `vars.SDXL_SNAPSHOT` in CI points at the **dense** `sdxl-base-1.0` tier; the story hit the packed
/// q4 `SceneWorks/sdxl-base-mlx` tier. The batch contract is tier-independent (the packed/dense fork
/// is inside the UNet's Linear surface, not its batch axis) and this was validated locally against
/// the q4 tier, so either snapshot exercises the regression:
///
/// ```text
/// set SDXL_SNAPSHOT=E:\huggingface\hub\models--SceneWorks--sdxl-base-mlx\snapshots\<hash>\q4
/// cargo test -p candle-gen-sdxl --features cuda --release --test integration -- --ignored conformance::sdxl_cfg_off
/// ```
#[test]
#[ignore = "needs SDXL_SNAPSHOT (a diffusers snapshot dir) + a CUDA GPU; run with --features cuda --ignored"]
fn sdxl_cfg_off_guidance_one_render() {
    let snap = std::env::var("SDXL_SNAPSHOT")
        .expect("set SDXL_SNAPSHOT to an SDXL-family diffusers snapshot dir");
    let spec = with_sdxl_components(LoadSpec::new(WeightsSource::Dir(PathBuf::from(snap))));
    let gen = candle_gen_sdxl::load(&spec).unwrap();

    // The story's request shape, minus its 1024² (512²/4 steps keeps the GPU cost down; the batch
    // contract is resolution-independent).
    let render = |guidance: f32, sampler: Option<&str>| {
        let req = GenerationRequest {
            prompt: "a photo of a rusty robot holding a lit candle, cinematic lighting".into(),
            width: 512,
            height: 512,
            count: 1,
            seed: Some(124),
            steps: Some(4),
            guidance: Some(guidance),
            sampler: sampler.map(str::to_string),
            ..Default::default()
        };
        let mut last_step = (0u32, 0u32);
        let mut on_progress = |p: Progress| {
            if let Progress::Step { current, total } = p {
                last_step = (current, total);
            }
        };
        let out = gen.generate(&req, &mut on_progress).unwrap_or_else(|e| {
            panic!("render at guidance {guidance} (sampler {sampler:?}) failed: {e}")
        });
        let images = match out {
            GenerationOutput::Images(imgs) => imgs,
            _ => panic!("expected images, got video"),
        };
        assert_eq!(images.len(), 1, "count=1 ⇒ one image");
        assert_eq!(last_step, (4, 4), "Step progress should end at 4/4");
        images.into_iter().next().unwrap()
    };

    // CFG OFF via the **omitted** sampler (resolves to the curated `ddim`) — the broadest route,
    // and the one every request that names no sampler takes.
    let off = render(1.0, None);
    assert_eq!((off.width, off.height), (512, 512), "output dims = request");
    assert_eq!(off.pixels.len(), 512 * 512 * 3, "RGB8 buffer = W·H·3");
    let (min, max) = (
        *off.pixels.iter().min().unwrap(),
        *off.pixels.iter().max().unwrap(),
    );
    assert!(
        max - min > 16,
        "CFG-off render looks degenerate (flat): min={min} max={max}"
    );

    // CFG OFF via the explicit `euler` the failing job actually sent (`advanced.sampler=euler`).
    // Both names land in `denoise_curated`, but this is the literal repro.
    let off_euler = render(1.0, Some("euler"));
    let (emin, emax) = (
        *off_euler.pixels.iter().min().unwrap(),
        *off_euler.pixels.iter().max().unwrap(),
    );
    assert!(
        emax - emin > 16,
        "CFG-off euler render looks degenerate (flat): min={emin} max={emax}"
    );

    // CFG ON — the default path, unchanged, and genuinely a different denoise.
    let on = render(7.0, None);
    assert_ne!(
        off.pixels, on.pixels,
        "CFG-off must not silently render the CFG-on image"
    );
}

/// A ~150-CLIP-token prompt (the story's 106-word repro shape) — comfortably past CLIP's 77-token
/// window, so the chunker must produce two windows.
const LONG_PROMPT: &str = "a highly detailed cinematic portrait of a weathered brass automaton \
    seated at a cluttered workbench in a Victorian clockmaker's attic, wearing a patched leather \
    apron and cracked goggles pushed up onto its forehead, surrounded by half-assembled pocket \
    watches, brass gears, coiled springs, oil cans and yellowed technical drawings pinned to the \
    sloping wooden walls, warm amber lamplight raking across the scene from a single dusty window \
    on the left, fine motes of dust suspended in the beam, shallow depth of field, 85mm lens, \
    volumetric light, intricate metal textures with patina and verdigris, soft rim light along the \
    automaton's shoulder, muted teal shadows balancing the warm highlights, photorealistic render, \
    ultra sharp focus on the hands, film grain";

/// The prefix of [`LONG_PROMPT`] that fits inside a single CLIP window — the conditioning the
/// pre-sc-20528 code could have produced if it had truncated instead of erroring.
const SHORT_PREFIX: &str = "a highly detailed cinematic portrait of a weathered brass automaton \
    seated at a cluttered workbench in a Victorian clockmaker's attic, wearing a patched leather \
    apron and cracked goggles";

/// sc-20528 acceptance, real weights: a prompt past CLIP's architectural 77-token window **renders**
/// on the SDXL family instead of failing `prompt too long: 146 tokens > 77`.
///
/// This is the story's own repro on a real checkpoint, and it is deliberately not just a liveness
/// check. Three arms:
///
/// 1. **AC1** — [`LONG_PROMPT`] renders a non-degenerate image at the requested dims.
/// 2. **AC4/no-silent-truncation** — the long render differs pixel-wise from a render of
///    [`SHORT_PREFIX`] at the same seed. If the engine had quietly dropped the tail (the one outcome
///    the story forbids), the two would be byte-identical: the prefix IS what a truncating engine
///    would have encoded. A difference is positive evidence that the second CLIP window reached the
///    UNet's cross-attention.
/// 3. **AC3** — an over-long *negative* prompt also renders, exercising the uncond row through the
///    same chunker (the `[uncond, cond]` batch stack fails outright if the two rows disagree on
///    sequence length).
///
/// Both CLIP encoders (AC2) are covered implicitly and unavoidably: SDXL concatenates CLIP-L and
/// CLIP-bigG on the feature axis, so a render only completes when both took the same chunked path.
///
/// ```text
/// set SDXL_SNAPSHOT=E:\huggingface\hub\models--stabilityai--stable-diffusion-xl-base-1.0\snapshots\<hash>
/// cargo test -p candle-gen-sdxl --features cuda --release --test integration -- --ignored conformance::sdxl_long_prompt
/// ```
#[test]
#[ignore = "needs SDXL_SNAPSHOT (a diffusers snapshot dir) + a CUDA GPU; run with --features cuda --ignored"]
fn sdxl_long_prompt_render() {
    let snap = std::env::var("SDXL_SNAPSHOT")
        .expect("set SDXL_SNAPSHOT to an SDXL-family diffusers snapshot dir");
    let spec = with_sdxl_components(LoadSpec::new(WeightsSource::Dir(PathBuf::from(snap))));
    let gen = candle_gen_sdxl::load(&spec).unwrap();

    // 512²/4 steps keeps the GPU cost down — the conditioning length is resolution-independent.
    let render = |prompt: &str, negative: Option<&str>| {
        let req = GenerationRequest {
            prompt: prompt.into(),
            negative_prompt: negative.map(str::to_string),
            width: 512,
            height: 512,
            count: 1,
            seed: Some(20528),
            steps: Some(4),
            ..Default::default()
        };
        let mut on_progress = |_: Progress| {};
        let out = gen
            .generate(&req, &mut on_progress)
            .unwrap_or_else(|e| panic!("render failed: {e}"));
        match out {
            GenerationOutput::Images(imgs) => imgs.into_iter().next().unwrap(),
            _ => panic!("expected images, got video"),
        }
    };

    // (1) The over-long prompt renders at all — the regression this story exists for.
    let long = render(LONG_PROMPT, None);
    assert_eq!(
        (long.width, long.height),
        (512, 512),
        "output dims = request"
    );
    assert_eq!(long.pixels.len(), 512 * 512 * 3, "RGB8 buffer = W·H·3");
    let (min, max) = (
        *long.pixels.iter().min().unwrap(),
        *long.pixels.iter().max().unwrap(),
    );
    assert!(
        max - min > 16,
        "long-prompt render looks degenerate (flat): min={min} max={max}"
    );

    // (2) The dropped tail would have been invisible — prove it was not dropped.
    let truncated = render(SHORT_PREFIX, None);
    assert_ne!(
        long.pixels, truncated.pixels,
        "a long prompt must not render identically to its first CLIP window — that is exactly the \
         silent truncation sc-20528 forbids"
    );

    // (3) The negative prompt takes the same path.
    let long_negative = render(SHORT_PREFIX, Some(LONG_PROMPT));
    assert_eq!(long_negative.pixels.len(), 512 * 512 * 3);
    assert_ne!(
        long_negative.pixels, truncated.pixels,
        "an over-long negative prompt must actually condition the render"
    );
}
