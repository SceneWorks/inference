//! sc-10840 (epic 10834): the `Sequential` component-residency A/B on real SANA weights.
//!
//! `#[ignore]`d — needs a real `Sana_1600M_1024px_diffusers`-shaped snapshot (`SANA_PIPELINE_WEIGHTS`).
//! Run:
//!   SANA_PIPELINE_WEIGHTS=/path/Sana_1600M_1024px_diffusers \
//!     cargo test -p mlx-gen-sana --release --test sequential_residency_real_weights -- --ignored --nocapture
//!
//! Same two claims as the SDXL / Z-Image A/Bs: (1) `Sequential` peaks LOWER than `Resident` because the
//! Gemma-2 CHI text encoder is dropped (+ `clear_cache()`) before the Linear-DiT trunk + DC-AE
//! materialize, and (2) the output is BYTE-IDENTICAL. SANA's Gemma encoder is comparable to (often ≥)
//! the DiT, so the saving is proportionally large. A repeat-job check confirms nothing stays resident
//! across jobs. Set `SANA_SEQ_STEPS` / `SANA_SEQ_SIZE` to tune; `SANA_SPRINT=1` drives the Sprint id.

use std::path::PathBuf;

use mlx_gen::{
    Conditioning, GenerationOutput, GenerationRequest, Image, LoadSpec, OffloadPolicy,
    WeightsSource,
};
use mlx_rs::memory::{clear_cache, get_peak_memory, reset_peak_memory};

const GIB: f64 = 1024.0 * 1024.0 * 1024.0;

fn env_u32(key: &str, default: u32) -> u32 {
    std::env::var(key)
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(default)
}

fn snapshot() -> Option<PathBuf> {
    std::env::var("SANA_PIPELINE_WEIGHTS")
        .ok()
        .map(PathBuf::from)
}

fn is_sprint() -> bool {
    std::env::var("SANA_SPRINT").is_ok()
}

fn model_id() -> &'static str {
    if is_sprint() {
        "sana_sprint_1600m"
    } else {
        "sana_1600m"
    }
}

fn probe_request() -> GenerationRequest {
    // Base SANA is true-CFG (pos + neg encode) — exercises the seam's cond+uncond materialize/drop path.
    // Sprint is CFG-free (cond only). A fixed seed makes the byte-identity assertion meaningful.
    let size = env_u32("SANA_SEQ_SIZE", 1024);
    let (guidance, negative, steps) = if is_sprint() {
        (None, None, env_u32("SANA_SEQ_STEPS", 2))
    } else {
        (
            Some(4.5),
            Some("blurry, low quality".to_string()),
            env_u32("SANA_SEQ_STEPS", 12),
        )
    };
    GenerationRequest {
        prompt: "a red panda on a mossy log in a misty forest, photograph".into(),
        negative_prompt: negative,
        guidance,
        width: size,
        height: size,
        seed: Some(1234),
        steps: Some(steps),
        ..Default::default()
    }
}

/// Like [`render_measured`] but returns EVERY image, for `count > 1`.
fn render_all(
    policy: OffloadPolicy,
    snap: PathBuf,
    req: &GenerationRequest,
) -> (Vec<Vec<u8>>, usize) {
    let spec = LoadSpec::new(WeightsSource::Dir(snap)).with_offload_policy(policy);
    let model = mlx_gen_sana::provider_registry()
        .expect("build provider registry")
        .load(model_id(), &spec)
        .expect("load sana");
    reset_peak_memory();
    let out = model.generate(req, &mut |_| {}).expect("generate");
    let peak = get_peak_memory();
    let images = match out {
        GenerationOutput::Images(v) => v.into_iter().map(|i| i.pixels).collect(),
        other => panic!("expected Images, got {other:?}"),
    };
    drop(model);
    clear_cache();
    (images, peak)
}

fn render_measured(
    policy: OffloadPolicy,
    snap: PathBuf,
    req: &GenerationRequest,
) -> (Vec<u8>, usize) {
    let spec = LoadSpec::new(WeightsSource::Dir(snap)).with_offload_policy(policy);
    let model = mlx_gen_sana::provider_registry()
        .expect("build provider registry")
        .load(model_id(), &spec)
        .expect("load sana");
    reset_peak_memory();
    let out = model.generate(req, &mut |_| {}).expect("generate");
    let peak = get_peak_memory();
    let img = match out {
        GenerationOutput::Images(mut v) => {
            assert_eq!(v.len(), 1, "expected a single image");
            v.pop().unwrap()
        }
        other => panic!("expected Images, got {other:?}"),
    };
    let Image { pixels, .. } = img;
    drop(model);
    clear_cache();
    (pixels, peak)
}

/// The residency A/B, with the DECODE STRATEGY held constant.
///
/// # Why this test pins the env var, and why that is not a dodge
///
/// The sibling suites (SDXL, Z-Image) assert that `Sequential` output is byte-identical to
/// `Resident`. The claim under test is that the residency *mechanism* — dropping components and
/// re-loading them per generation — is transparent. That claim is true for SANA and is what this
/// asserts.
///
/// What is **not** true for SANA is that its bounded decode is free. `Sequential` now tiles the
/// DC-AE decode by default (untiled was measured at 9177 MiB and killed the app on device), and
/// DC-AE tiling is **not output-preserving**: its `SanaMultiscaleLinearAttention` normalizes by
/// `1/(Σ+eps)` over every spatial position it is given, so a tile sees a different denominator than
/// the whole image. Z-Image's VAE is convolutional and reconstructs exactly under overlapping
/// tiles; SANA's cannot, by construction rather than by tuning.
///
/// So comparing a tiled `Sequential` against an untiled `Resident` measures the decode strategy and
/// tells you nothing about residency. `MLX_GEN_SANA_DECODE_TILE=0` equalizes the two arms, and the
/// byte-identity assertion then means what it means everywhere else in the workspace.
///
/// The divergence the default introduces is not swept up — it is asserted directly in
/// [`sequential_default_tiles_and_therefore_diverges`] below, so the surprising behaviour is pinned
/// rather than merely absent.
#[test]
#[ignore = "needs a Sana_1600M_1024px_diffusers snapshot; set SANA_PIPELINE_WEIGHTS"]
fn sequential_bounds_peak_and_is_byte_identical() {
    let Some(snap) = snapshot() else {
        eprintln!("skipping: set SANA_PIPELINE_WEIGHTS to run the SANA residency A/B");
        return;
    };
    // Equalize the decode across both arms; see the doc comment.
    std::env::set_var("MLX_GEN_SANA_DECODE_TILE", "0");

    let req = probe_request();
    let (pixels_resident, peak_resident) =
        render_measured(OffloadPolicy::Resident, snap.clone(), &req);
    let (pixels_sequential, peak_sequential) =
        render_measured(OffloadPolicy::Sequential, snap, &req);
    std::env::remove_var("MLX_GEN_SANA_DECODE_TILE");

    println!(
        "SANA ({}) {}x{} @ {} steps:\n  Resident   peak = {:.3} GiB\n  Sequential peak = {:.3} GiB\n  saved = {:.3} GiB ({:.1}%)",
        model_id(),
        req.width,
        req.height,
        req.steps.unwrap(),
        peak_resident as f64 / GIB,
        peak_sequential as f64 / GIB,
        (peak_resident.saturating_sub(peak_sequential)) as f64 / GIB,
        100.0 * (peak_resident.saturating_sub(peak_sequential)) as f64 / peak_resident as f64,
    );

    let diff = pixels_resident
        .iter()
        .zip(&pixels_sequential)
        .filter(|(a, b)| a != b)
        .count();
    assert_eq!(
        diff,
        0,
        "Sequential residency changed the output: {diff}/{} bytes differ (must be byte-identical)",
        pixels_resident.len()
    );
    assert!(
        peak_sequential < peak_resident,
        "Sequential peak {:.3} GiB was not below Resident {:.3} GiB — the Gemma-TE drop did not \
         reduce peak",
        peak_sequential as f64 / GIB,
        peak_resident as f64 / GIB,
    );
}

/// The default `Sequential` render DIVERGES from `Resident`, and that is the intended behaviour.
///
/// Pinned explicitly because it is surprising and because the test above deliberately equalizes it
/// away. A consumer that switches `OffloadPolicy` for memory reasons gets a *different image* from
/// SANA — not a worse one, and not a seamed one, but not the same bytes.
///
/// The trade is forced, not chosen: an untiled DC-AE decode was measured at 9177 MiB and killed the
/// app on an iPhone with a 6135 MiB cap, while the tiled path completed at 2751 MiB. Exact-but-dead
/// is not a policy. `Resident` keeps the exact decode, so a host with memory to spare pays nothing.
///
/// If this test ever starts failing because the two agree, something re-enabled the untiled decode
/// under `Sequential` — which is the shipping defect this whole change exists to close, and it must
/// not come back silently.
#[test]
#[ignore = "needs a Sana_1600M_1024px_diffusers snapshot; set SANA_PIPELINE_WEIGHTS"]
fn sequential_default_tiles_and_therefore_diverges() {
    let Some(snap) = snapshot() else {
        eprintln!("skipping: set SANA_PIPELINE_WEIGHTS to run the SANA residency A/B");
        return;
    };
    // No override: this is exactly what a product gets.
    std::env::remove_var("MLX_GEN_SANA_DECODE_TILE");

    let req = probe_request();
    let (resident, _) = render_measured(OffloadPolicy::Resident, snap.clone(), &req);
    let (sequential, _) = render_measured(OffloadPolicy::Sequential, snap, &req);

    let diff = resident
        .iter()
        .zip(&sequential)
        .filter(|(a, b)| a != b)
        .count();
    assert!(
        diff > 0,
        "Sequential produced byte-identical output to Resident, which means it did NOT tile. \
         Untiled DC-AE decode is the configuration that was killed on device — see \
         `pipeline::decode_tiling`."
    );

    // Different, but not broken: a layout error or a lost tile moves most of the frame. DC-AE's
    // attention-scope shift moves a minority of bytes and none of them far (measured mean |Δ| ~2.6
    // of 255 — see `decode_tiling_parity`).
    let fraction = diff as f64 / resident.len() as f64;
    println!(
        "default Sequential vs Resident: {diff}/{} bytes differ ({:.1}%)",
        resident.len(),
        100.0 * fraction
    );
    assert!(
        fraction < 0.95,
        "nearly every byte differs ({:.1}%) — that is a layout or geometry error, not the \
         attention-scope shift tiling is expected to cause",
        100.0 * fraction
    );
}

/// img2img and `count > 1` through the **staged** seam (sc-13571 adoption).
///
/// These are the two request shapes the staged decode restructured, and neither was covered.
///
/// **img2img** is the path where the DC-AE *encoder*'s lifetime matters. `StagedHeavy::shed_dit`
/// drops the encoder along with the trunk, so an encoder still needed after the shed would be gone.
/// It is not — `encode_init_latents` runs inside the denoise phase — but that is an argument, and
/// the argument is exactly what a restructure invalidates.
///
/// **count > 1** is the reordering itself: every seed now denoises before anything decodes, so the
/// trunk can be shed once for the batch instead of held across N decodes. That changes what is
/// live when, and it changes `Mid` from one latent to a `Vec`.
///
/// Both are asserted against the **resident** path rather than against a golden, so this stays a
/// statement about the seam (staging must not change pixels) and not about SANA's sampler.
#[test]
#[ignore = "needs a Sana_1600M_1024px_diffusers snapshot; set SANA_PIPELINE_WEIGHTS"]
fn staged_seam_preserves_img2img_and_multi_image_output() {
    let Some(snap) = snapshot() else {
        eprintln!("skipping: set SANA_PIPELINE_WEIGHTS to run the SANA staged-seam checks");
        return;
    };
    // Equalize the decode, for the same reason as `sequential_bounds_peak_and_is_byte_identical`:
    // the subject here is the staged SEAM (the encoder's lifetime across `shed_dit`, and the
    // denoise-all-then-decode-all reordering). Leaving the default in place would compare a tiled
    // Sequential against an untiled Resident and fail on the decode strategy, which this test does
    // not exercise and cannot diagnose.
    std::env::set_var("MLX_GEN_SANA_DECODE_TILE", "0");
    struct RestoreEnv;
    impl Drop for RestoreEnv {
        fn drop(&mut self) {
            std::env::remove_var("MLX_GEN_SANA_DECODE_TILE");
        }
    }
    // Guard, not a trailing call: an assertion failure below must not leak the override into
    // whatever test runs next.
    let _restore = RestoreEnv;

    // ── count > 1.
    let mut batch = probe_request();
    batch.count = 2;
    let (batch_resident, _) = render_all(OffloadPolicy::Resident, snap.clone(), &batch);
    let (batch_sequential, _) = render_all(OffloadPolicy::Sequential, snap.clone(), &batch);
    assert_eq!(batch_resident.len(), 2, "count=2 must return two images");
    assert_eq!(
        batch_sequential.len(),
        2,
        "count=2 must return two images under Sequential too"
    );
    assert_ne!(
        batch_resident[0], batch_resident[1],
        "the two seeds produced identical images — the per-image seed offset was lost, which the \
         batch reordering could silently do by hoisting the seed out of the loop"
    );
    for (i, (r, s)) in batch_resident.iter().zip(&batch_sequential).enumerate() {
        let diff = r.iter().zip(s).filter(|(a, b)| a != b).count();
        assert_eq!(
            diff, 0,
            "image {i} of a count=2 batch differs between residencies: {diff} bytes"
        );
    }

    // ── img2img: use the first batch image as the reference, so no fixture is needed and the
    // init latents are in-distribution.
    let init = Image {
        width: batch.width,
        height: batch.height,
        pixels: batch_resident[0].clone(),
    };
    let mut i2i = probe_request();
    i2i.conditioning = vec![Conditioning::Reference {
        image: init,
        strength: Some(0.6),
    }];
    let (i2i_resident, _) = render_all(OffloadPolicy::Resident, snap.clone(), &i2i);
    let (i2i_sequential, _) = render_all(OffloadPolicy::Sequential, snap, &i2i);

    let diff = i2i_resident[0]
        .iter()
        .zip(&i2i_sequential[0])
        .filter(|(a, b)| a != b)
        .count();
    assert_eq!(
        diff, 0,
        "img2img differs between residencies: {diff} bytes. Under Sequential the DC-AE encoder is \
         built only when the request needs it and is dropped by `shed_dit` — a lifetime error there \
         shows up here and nowhere else."
    );
    // A strength-0.6 img2img must still resemble neither pure noise nor the input exactly.
    let unchanged = i2i_resident[0]
        .iter()
        .zip(&batch_resident[0])
        .filter(|(a, b)| a == b)
        .count();
    assert!(
        unchanged < i2i_resident[0].len(),
        "img2img returned the reference image unchanged — the denoise did not run"
    );
}

#[test]
#[ignore = "needs a Sana_1600M_1024px_diffusers snapshot; set SANA_PIPELINE_WEIGHTS"]
fn sequential_repeat_job_stays_bounded() {
    let Some(snap) = snapshot() else {
        eprintln!("skipping: set SANA_PIPELINE_WEIGHTS to run the SANA residency A/B");
        return;
    };
    let req = probe_request();
    let (_p1, peak1) = render_measured(OffloadPolicy::Sequential, snap.clone(), &req);
    let (_p2, peak2) = render_measured(OffloadPolicy::Sequential, snap, &req);
    println!(
        "SANA Sequential repeat-job peaks: job1 = {:.3} GiB, job2 = {:.3} GiB",
        peak1 as f64 / GIB,
        peak2 as f64 / GIB,
    );
    let slop = peak1 / 10;
    assert!(
        peak2 <= peak1 + slop,
        "repeat Sequential job peaked higher ({:.3} vs {:.3} GiB) — a component stayed resident",
        peak2 as f64 / GIB,
        peak1 as f64 / GIB,
    );
}
