//! sc-16955 — candle **Lens** per-step latent preview real-weight validation (epic 16948).
//!
//! Lens contributes no fit and no projector: it denoises the FLUX.2 32-channel latent space in the
//! same packed token layout, through the same `Flux2Vae`, so `crate::preview` re-exports
//! `candle_gen_flux2::preview`. Two things still have to be established *here*, and neither transfers
//! from the FLUX.2 rows:
//!
//! 1. **The VAE Lens loads is the one the fit was measured over.**
//!    [`the_lens_vae_rounds_onto_the_flux2_fit_donor`] is the reuse gate. Lens publishes an **f32**
//!    container while the epic-16624 fit was measured on FLUX.2-klein's **bf16** one, so a hash
//!    equality is unavailable and a matching Rust type proves nothing: every one of the 250 learned
//!    tensors is compared, and each must round — round-to-nearest-even, the rounding a bf16 cast
//!    performs — exactly onto the donor's bits.
//! 2. **The frames actually develop on a Lens render.**
//!    [`lens_preview_frames_evolve_toward_the_final_image`] drives the registered route through the
//!    `Generator` seam with a live sink, checks numbering, checks seeded byte-identity against an inert
//!    render, and measures that every frame is closer to — and more like — the finished image than the
//!    one before it. The strip is written out for review.
//!
//! ```sh
//! LENS_PREVIEW_SNAPSHOT=E:\huggingface\hub\models--SceneWorks--Lens\snapshots\<rev> \
//! LENS_PREVIEW_VAE=...\models--SceneWorks--Lens\snapshots\<rev>\vae\diffusion_pytorch_model.safetensors \
//! LENS_FLUX2_FIT_VAE=...\models--black-forest-labs--FLUX.2-klein-9B\snapshots\<rev>\vae\diffusion_pytorch_model.safetensors \
//! LENS_PREVIEW_ARTIFACT_DIR=E:\out\sc-16955 \
//!   cargo test -p candle-gen-lens --release --features cuda --test integration preview_real_weights:: \
//!     -- --ignored --nocapture
//! ```
//!
//! Every input is **required** by the row that uses it: a row that early-returns on an unset variable
//! still reports SUCCESS, and in a run log a skipped gate is indistinguishable from one that ran and
//! proved something. Asking for `--ignored` is already the opt-in.

use std::collections::BTreeMap;
use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex};

use candle_gen::candle_core::{DType, Device, Tensor};
use candle_gen::gen_core::{
    GenerationOutput, GenerationRequest, Image, LoadSpec, PreviewFrame, PreviewSink, Progress,
    WeightsSource,
};

const PROMPT: &str =
    "A weathered lighthouse on a rocky headland at golden hour, warm sunlight, dramatic clouds, \
     highly detailed photograph.";
const SEED: u64 = 16955;

/// The SHA-256 of the **f32** `AutoencoderKLFlux2` container every Lens snapshot publishes —
/// 336,213,556 bytes. Byte-identical across `SceneWorks/Lens`
/// @ `5c5521d4417a3cae55816929ece69319d1e7712a`, `Comfy-Org/Lens`
/// @ `198d6ddf4d9fac0d8b0548dc9be4310452f5c146` (as `vae/flux2-vae.safetensors`), every tier of
/// `SceneWorks/lens-mlx` @ `4e1349c1962950eee328c69537904631ebc64283` and `SceneWorks/lens-turbo-mlx`
/// @ `d3f485c320039595cff16d4f686a5f9378714f25`, and `black-forest-labs/FLUX.2-dev`
/// @ `26afe3a78bb242c0a8bb181dcc8937bb16e5c66c` — so `lens`, `lens_turbo` and `flux2_dev` all decode
/// through one file.
const LENS_VAE_SHA256: &str = "d64f3a68e1cc4f9f4e29b6e0da38a0204fe9a49f2d4053f0ec1fa1ca02f9c4b5";

/// The SHA-256 of the **bf16** container the epic-16624 32-channel fit was measured against —
/// 168,120,878 bytes, `black-forest-labs/FLUX.2-klein-9B`
/// @ `92196c8e11f7b6cf2b7493e037d8c5345c559216`. A different file, which is why the row below is a
/// tensor comparison rather than a hash equality.
const FLUX2_FIT_VAE_SHA256: &str =
    "ca70d2202afe6415bdbcb8793ba8cd99fd159cfe6192381504d6c4d3036e0f04";

/// The measured extent of the transfer, pinned so a partial comparison cannot pass as a full one.
const VAE_LEARNED_TENSORS: usize = 250;
const VAE_LEARNED_VALUES: usize = 84_046_371;

/// BatchNorm's forward-pass counter — the one tensor that is not part of the learned map, read by
/// nothing (`Flux2Vae::build` loads `bn.running_mean` and `bn.running_var` and never this).
const VAE_UNUSED_COUNTER: &str = "bn.num_batches_tracked";

fn env_path(name: &str) -> Option<PathBuf> {
    std::env::var(name).ok().map(PathBuf::from)
}

/// An input a row cannot run without. Missing means **fail**, not skip.
fn required_path(name: &str) -> PathBuf {
    env_path(name).unwrap_or_else(|| {
        panic!(
            "{name} must be set for this row — skipping it would report success while proving \
             nothing"
        )
    })
}

fn artifact_dir() -> PathBuf {
    env_path("LENS_PREVIEW_ARTIFACT_DIR")
        .unwrap_or_else(|| std::env::temp_dir().join("lens_preview_sc16955"))
}

fn env_u32(name: &str, fallback: u32) -> u32 {
    std::env::var(name)
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(fallback)
}

fn sha256_of(path: &Path) -> String {
    use sha2::{Digest, Sha256};
    let mut file = std::fs::File::open(path).unwrap_or_else(|e| panic!("open {path:?}: {e}"));
    let mut hasher = Sha256::new();
    std::io::copy(&mut file, &mut hasher).expect("hash the file");
    format!("{:x}", hasher.finalize())
}

fn tensors_of(path: &Path) -> BTreeMap<String, Tensor> {
    candle_gen::candle_core::safetensors::load(path, &Device::Cpu)
        .unwrap_or_else(|e| panic!("parse {path:?}: {e}"))
        .into_iter()
        .collect()
}

/// A bf16 tensor widened to f32, exactly. Comparing widened values is equivalent to comparing the
/// 16-bit patterns: bf16 → f32 is lossless and injective, so two bf16 tensors widen to equal f32
/// vectors iff their bit patterns match (weights carry no NaN, the only leak).
fn widened_bf16(tensor: &Tensor) -> Vec<f32> {
    tensor
        .to_dtype(DType::F32)
        .and_then(|t| t.flatten_all())
        .and_then(|t| t.to_vec1::<f32>())
        .expect("widen a bf16 tensor to f32")
}

// ── Provenance: the reuse is grounded in tensor bytes ─────────────────────────────────────────────

/// **The reuse gate.** Lens's f32 VAE and the FLUX.2-klein bf16 VAE the fit was measured on are two
/// container widths of one learned checkpoint.
///
/// Both inputs are required. There is no configuration of this row that passes without performing the
/// comparison.
#[test]
#[ignore = "needs LENS_PREVIEW_VAE + LENS_FLUX2_FIT_VAE; run with --ignored"]
fn the_lens_vae_rounds_onto_the_flux2_fit_donor() {
    let lens_path = required_path("LENS_PREVIEW_VAE");
    let donor_path = required_path("LENS_FLUX2_FIT_VAE");

    let (lens_sha, donor_sha) = (sha256_of(&lens_path), sha256_of(&donor_path));
    eprintln!("  lens  vae/ {lens_sha}  {}", lens_path.display());
    eprintln!("  donor vae/ {donor_sha}  {}", donor_path.display());
    assert_eq!(
        lens_sha, LENS_VAE_SHA256,
        "the VAE this Lens snapshot publishes is not the file the reuse was grounded against"
    );
    assert_eq!(
        donor_sha, FLUX2_FIT_VAE_SHA256,
        "LENS_FLUX2_FIT_VAE must be the snapshot the epic-16624 32-channel fit was measured on"
    );
    assert_ne!(
        lens_sha, donor_sha,
        "the two files are deliberately different container widths of the same tensors — if they \
         ever became byte-identical this row's rounding argument would be dead code"
    );

    let (wide, narrow) = (tensors_of(&lens_path), tensors_of(&donor_path));
    assert_eq!(
        wide.keys().collect::<Vec<_>>(),
        narrow.keys().collect::<Vec<_>>(),
        "the two containers must hold the same key set"
    );
    assert_eq!(wide.len(), VAE_LEARNED_TENSORS + 1);

    let mut values = 0usize;
    for (key, w) in &wide {
        let n = &narrow[key];
        assert_eq!(w.dims(), n.dims(), "{key}: shapes must match");
        if key == VAE_UNUSED_COUNTER {
            continue;
        }
        assert_eq!(
            w.dtype(),
            DType::F32,
            "{key}: Lens publishes an f32 container"
        );
        assert_eq!(n.dtype(), DType::BF16, "{key}: the donor is bf16");
        // The comparison IS the bf16 cast: widening the donor instead would silently accept an f32
        // value that merely rounds close, rather than one that rounds exactly onto these bits.
        let cast = w.to_dtype(DType::BF16).expect("cast to bf16");
        let (a, b) = (widened_bf16(&cast), widened_bf16(n));
        assert_eq!(
            a, b,
            "{key}: the Lens tensor does not round onto the donor's"
        );
        values += a.len();
    }
    assert_eq!(
        values, VAE_LEARNED_VALUES,
        "the comparison must cover every learned value"
    );
    eprintln!("  {VAE_LEARNED_TENSORS} learned tensors, {values} values: bf16-round-identical");

    // The fit Lens projects with is the FLUX.2 one, over 32 channels — re-exported, never copied.
    assert_eq!(candle_gen_lens::preview::PREVIEW_LATENT_CHANNELS, 32);
}

// ── Frame analysis helpers ────────────────────────────────────────────────────────────────────────

fn mean_abs_delta(a: &[u8], b: &[u8]) -> f64 {
    assert_eq!(a.len(), b.len(), "compared buffers must match in length");
    a.iter()
        .zip(b)
        .map(|(&x, &y)| (x as f64 - y as f64).abs())
        .sum::<f64>()
        / a.len() as f64
}

fn correlation(a: &[u8], b: &[u8]) -> f64 {
    assert_eq!(a.len(), b.len());
    let n = a.len() as f64;
    let (ma, mb) = (
        a.iter().map(|&v| v as f64).sum::<f64>() / n,
        b.iter().map(|&v| v as f64).sum::<f64>() / n,
    );
    let mut num = 0.0;
    let (mut da, mut db) = (0.0, 0.0);
    for (&x, &y) in a.iter().zip(b) {
        let (dx, dy) = (x as f64 - ma, y as f64 - mb);
        num += dx * dy;
        da += dx * dx;
        db += dy * dy;
    }
    if da == 0.0 || db == 0.0 {
        return 0.0;
    }
    num / (da.sqrt() * db.sqrt())
}

fn downsample_raw(pixels: &[u8], src_w: u32, src_h: u32, w: u32, h: u32) -> Vec<u8> {
    let mut out = Vec::with_capacity((w * h * 3) as usize);
    for y in 0..h {
        for x in 0..w {
            let sx = (x as u64 * src_w as u64 / w as u64) as u32;
            let sy = (y as u64 * src_h as u64 / h as u64) as u32;
            let idx = ((sy * src_w + sx) * 3) as usize;
            out.extend_from_slice(&pixels[idx..idx + 3]);
        }
    }
    out
}

fn downsample(img: &Image, w: u32, h: u32) -> Vec<u8> {
    downsample_raw(&img.pixels, img.width, img.height, w, h)
}

fn save_png(pixels: &[u8], width: u32, height: u32, name: &str) {
    let dir = artifact_dir();
    std::fs::create_dir_all(&dir).expect("create the artifact dir");
    let path = dir.join(name);
    let buf: image::RgbImage = image::ImageBuffer::from_raw(width, height, pixels.to_vec())
        .expect("frame buffer matches its dimensions");
    buf.save(&path)
        .unwrap_or_else(|e| panic!("save {path:?}: {e}"));
    eprintln!("  wrote {}", path.display());
}

fn save_strip(frames: &[PreviewFrame], name: &str) {
    assert!(!frames.is_empty(), "an empty strip cannot be written");
    let (fw, fh) = (frames[0].image.width, frames[0].image.height);
    let total_w = fw * frames.len() as u32;
    let mut sheet = vec![0u8; (total_w * fh * 3) as usize];
    for (i, frame) in frames.iter().enumerate() {
        let x0 = i as u32 * fw;
        for y in 0..fh {
            for x in 0..fw {
                let src = ((y * fw + x) * 3) as usize;
                let dst = (((y * total_w) + x0 + x) * 3) as usize;
                sheet[dst..dst + 3].copy_from_slice(&frame.image.pixels[src..src + 3]);
            }
        }
    }
    save_png(&sheet, total_w, fh, name);
}

fn one_image(out: GenerationOutput) -> Image {
    let GenerationOutput::Images(mut images) = out else {
        panic!("expected GenerationOutput::Images");
    };
    assert_eq!(images.len(), 1, "these rows render a single image");
    images.pop().expect("one image")
}

fn collecting_sink() -> (PreviewSink, Arc<Mutex<Vec<PreviewFrame>>>) {
    let frames = Arc::new(Mutex::new(Vec::new()));
    let captured = Arc::clone(&frames);
    let sink = PreviewSink::new(move |frame| candle_gen::lock_recover(&captured).push(frame));
    (sink, frames)
}

/// Consecutive frames must not be the same picture. A floor rather than a ratio, and a low one: on an
/// 8-bit scale 0.1 mean |Δ| is unmistakably "not identical" while leaving room for the FIRST pair of a
/// flow strip, which is the smallest step of the whole trajectory (measured 0.448 on the 12-step klein
/// lane). The strong statement about movement is the monotone acceleration beside it, not this floor —
/// sc-16954's flat `> 0.5` was calibrated on a 4-channel fit whose coefficients are ~2.4× larger, and
/// it does not transfer.
const MIN_FRAME_MOVEMENT: f64 = 0.1;

/// The strip must close a meaningful share of its distance to the finished image. Expressed as a
/// fraction of the distance travelled rather than as a ratio of the endpoints, because the endpoints
/// carry the fit's irreducible residual — see the comment at the assertion.
const MIN_DISTANCE_FALL: f64 = 0.25;

/// The same strip analysis the FLUX.2 rows apply, so the family's lanes are all closed with one
/// measurement. See `candle-gen-flux2/tests/preview_real_weights.rs` for where the `0.75` correlation
/// floor is derived — it is 86.8% of the FLUX.2 **fit** R²'s correlation ceiling (√0.76409 ≈ 0.874),
/// the same fraction of its own ceiling the earlier families were held to, compared like-for-like on
/// an in-sample statistic rather than against a holdout one.
///
/// ## Why `min_acceleration` is a parameter too
///
/// "Movement accelerates into the terminal step" is a property of the **schedule**, exactly as
/// `min_r_last` is, and the two explain each other. FLUX.2's empirical-μ flow schedule is strongly
/// back-loaded: its terminal previewed step moves 23x the opening one (Lens, the same schedule family,
/// 15.7x), and its last frame correspondingly falls short of the fit-derived 0.75. Ideogram's
/// `LogitNormalSchedule` is not: its steps are far more evenly spread (1.9x), and its last frame
/// reaches +0.828 — 95% of the 0.874 ceiling. Hard-coding either number as a shared constant would
/// assert one family's schedule about another's.
fn assert_the_strip_converges(
    label: &str,
    frames: &[PreviewFrame],
    final_image: &Image,
    steps: u32,
    min_r_last: f64,
    min_acceleration: f64,
) {
    assert_eq!(
        frames
            .iter()
            .map(|f| (f.current, f.total))
            .collect::<Vec<_>>(),
        (1..=steps).map(|n| (n, steps)).collect::<Vec<_>>(),
        "{label}: a {steps}-step render must emit exactly {steps} frames numbered 1..={steps}"
    );

    // Raw-latent resolution: the finished image is `latent · 8` on each axis, because the unpatchify
    // doubles the token grid and the VAE upsamples 8×. Derived from the render rather than restated,
    // since Lens resolves its own aspect bucket. A CFG-fused `[2, …]` latent fails the packed-layout
    // contract outright, so a strip that exists at all is proof no unconditional half was projected.
    let (lw, lh) = (final_image.width / 8, final_image.height / 8);
    for frame in frames {
        assert_eq!(
            (frame.image.width, frame.image.height),
            (lw, lh),
            "{label}: frames must be raw-VAE-latent resolution"
        );
    }

    // Every frame must differ from its predecessor — N copies of one image would satisfy a naive
    // "N frames arrived" check while showing nothing developing. The whole vector is computed and
    // printed BEFORE anything is asserted, so one run reports the entire strip rather than stopping at
    // the first pair.
    let movement: Vec<f64> = frames
        .windows(2)
        .map(|p| mean_abs_delta(&p[0].image.pixels, &p[1].image.pixels))
        .collect();
    for (pair, delta) in frames.windows(2).zip(&movement) {
        eprintln!(
            "  {label} frame {:>2} → {:>2}: mean |Δ| {delta:.3}",
            pair[0].current, pair[1].current
        );
    }

    let target = downsample(final_image, lw, lh);
    let distances: Vec<f64> = frames
        .iter()
        .map(|f| mean_abs_delta(&f.image.pixels, &target))
        .collect();
    for (frame, distance) in frames.iter().zip(&distances) {
        eprintln!(
            "  {label} frame {:>2}: mean |Δ| to final {distance:.2}",
            frame.current
        );
    }

    // Absolute distance can only ever say "closer", never "resembles": the projection is a global
    // linear approximation of the decode, so even a perfectly converged latent keeps an offset and
    // gain error against the true pixels. Correlation over a coarse thumbnail, which averages the
    // residual away and leaves subject placement and colour masses, is what "the preview looks like
    // the image" actually means for a decorative frame.
    let coarse = 16u32;
    let coarse_target = downsample(final_image, coarse, coarse);
    let correlations: Vec<f64> = frames
        .iter()
        .map(|f| {
            correlation(
                &downsample_raw(
                    &f.image.pixels,
                    f.image.width,
                    f.image.height,
                    coarse,
                    coarse,
                ),
                &coarse_target,
            )
        })
        .collect();
    for (frame, r) in frames.iter().zip(&correlations) {
        eprintln!(
            "  {label} frame {:>2}: coarse correlation with final {r:+.3}",
            frame.current
        );
    }

    // ---- Nothing is asserted until every metric above has been printed. --------------------------
    //
    // sc-16954 shipped a row that stopped at the first failing pair, so a threshold discussion needed
    // a second run to see the rest of the strip. One run now reports the whole thing.

    // 1. No two consecutive frames are the same picture.
    assert!(
        movement.iter().all(|d| *d > MIN_FRAME_MOVEMENT),
        "{label}: some consecutive frames are effectively identical: {movement:?}"
    );
    // 2. Frame-to-frame movement ACCELERATES into the terminal step. This is the flow-match
    //    empirical-μ schedule's signature — its σ steps grow toward the terminal node — and it is a
    //    far stronger statement than a flat floor: a hook reading a stale, duplicated or wrongly
    //    scaled latent would not reproduce it. It is also why a flat floor ported from the ε cohort
    //    does not transfer: the FIRST pair of a flow strip is the smallest step of the trajectory.
    //
    //    Asserted over the **second half** plus an end-to-end ratio, not as strict monotonicity across
    //    the whole strip. The opening frames are near-pure noise projected through a global linear map,
    //    so the mean |Δ| between two of them carries sampling noise comparable to the σ step itself —
    //    Lens's measured strip dips once there (`0.636, 0.751, 0.683, 0.827, …`) while rising cleanly
    //    everywhere after. Demanding strict monotonicity would be asserting that noise, not the
    //    schedule.
    let back_half = &movement[movement.len() / 2..];
    assert!(
        back_half.windows(2).all(|p| p[1] > p[0]),
        "{label}: movement must rise monotonically over the second half of the strip: {movement:?}"
    );
    let (opening, closing) = (movement[0], movement[movement.len() - 1]);
    eprintln!(
        "  {label}: movement {opening:.3} → {closing:.3} ({:.1}x)",
        closing / opening
    );
    assert!(
        closing > opening * min_acceleration,
        "{label}: the terminal step must dominate the opening one by at least {min_acceleration}x          ({opening:.3} → {closing:.3})"
    );
    assert!(
        movement.iter().all(|d| *d <= closing + f64::EPSILON),
        "{label}: the last previewed step must be the largest: {movement:?}"
    );

    // 3. The strip approaches the finished image, at every step and by a meaningful margin.
    //
    //    The margin is expressed as a **fraction of the distance travelled**, not as sc-16954's
    //    `last < first * 0.6` ratio, because that ratio measures the fit's irreducible residual as
    //    much as the trajectory: a projection explaining R² 0.764 of the decode leaves a large
    //    constant offset that neither the first nor the last frame can cross, so a coarser fit shows a
    //    smaller fractional fall for the very same convergence. The scale-free statement about
    //    resemblance is the correlation below; this one is the "it is actually moving toward it" half.
    let (first, last) = (distances[0], distances[distances.len() - 1]);
    let fall = (first - last) / first;
    eprintln!(
        "  {label}: distance fell {:.1}% ({first:.2} → {last:.2})",
        fall * 100.0
    );
    assert!(
        fall > MIN_DISTANCE_FALL,
        "{label}: the strip must converge on the final image (first {first:.2} → last {last:.2}, \
         fall {:.3}, floor {MIN_DISTANCE_FALL})",
        fall
    );
    assert!(
        distances.windows(2).all(|p| p[1] < p[0]),
        "{label}: distance to the finished image must fall at every step: {distances:?}"
    );

    // 4. The strip actually comes to resemble the render, monotonically.
    let (r_first, r_last) = (correlations[0], correlations[correlations.len() - 1]);
    assert!(
        r_last > min_r_last,
        "{label}: the last preview frame must resemble the finished render \
         (r {r_last:+.3}, floor {min_r_last:+.3})"
    );
    assert!(
        correlations.windows(2).all(|p| p[1] > p[0]),
        "{label}: resemblance must increase at every step: {correlations:?}"
    );
    // "The strip develops" is asserted as a **rise**, not as an absolute floor on the first frame.
    // Correlation is taken over flattened RGB triplets, so it carries channel-mean structure as well
    // as spatial structure, and this fit's intercept is a near-neutral grey — so a frame of
    // pre-denoise noise starts at a non-zero, scene-dependent floor. sc-16950's `r_first < 0.35`
    // ceiling is deliberately not ported; the rise plus a loose ceiling is what cannot be faked, since
    // a strip that opened on the finished image would have nowhere to rise to.
    assert!(
        r_first < 0.75,
        "{label}: the first frame is pre-denoise noise and must not already BE the render \
         (r {r_first:+.3})"
    );
    assert!(
        r_last - r_first > 0.30,
        "{label}: resemblance must actually develop across the strip \
         (first {r_first:+.3} → last {r_last:+.3})"
    );
}

// ── Driving the registered route// ── Driving the registered route ──────────────────────────────────────────────────────────────────

/// Render the registered `lens` route twice on one warmed generator at the same seed — once inert,
/// once live — and hold the strip to [`assert_the_strip_converges`].
#[test]
#[ignore = "needs LENS_PREVIEW_SNAPSHOT + a CUDA GPU; run with --features cuda --ignored"]
fn lens_preview_frames_evolve_toward_the_final_image() {
    let label = "lens-euler";
    let steps = env_u32("LENS_PREVIEW_STEPS", 12);
    let size = env_u32("LENS_PREVIEW_SIZE", 1024);
    eprintln!("── {label}: {size}² × {steps} steps");

    let spec = LoadSpec::new(WeightsSource::Dir(required_path("LENS_PREVIEW_SNAPSHOT")));
    let generator = candle_gen_lens::provider_registry()
        .expect("lens registry")
        .load(candle_gen_lens::MODEL_ID_BASE, &spec)
        .unwrap_or_else(|e| panic!("load lens: {e}"));

    let request = |sink: Option<PreviewSink>| GenerationRequest {
        prompt: PROMPT.into(),
        width: size,
        height: size,
        count: 1,
        seed: Some(SEED),
        steps: Some(steps),
        preview: sink.unwrap_or_default(),
        ..GenerationRequest::default()
    };

    // N1: the inert baseline. Same generator, same seed, no sink.
    let mut noop = |_: Progress| {};
    let inert = one_image(
        generator
            .generate(&request(None), &mut noop)
            .unwrap_or_else(|e| panic!("{label}: inert render: {e}")),
    );

    let (sink, frames) = collecting_sink();
    let live = one_image(
        generator
            .generate(&request(Some(sink)), &mut noop)
            .unwrap_or_else(|e| panic!("{label}: live render: {e}")),
    );

    assert_eq!(
        inert.pixels, live.pixels,
        "{label}: attaching a live preview sink changed the seeded render"
    );

    let frames = candle_gen::lock_recover(&frames).clone();
    // 0.55 against a measured +0.625 on this lane; see `assert_the_strip_converges` for why the
    // floor is a schedule-dependent backstop rather than a fit-derived number.
    // 0.55 against a measured +0.625, and 5.0x against a measured 15.7x: Lens rides the FLUX.2
    // empirical-μ schedule, so both numbers track the FLUX.2 klein lane's.
    assert_the_strip_converges(label, &frames, &live, steps, 0.55, 5.0);
    save_strip(&frames, &format!("{label}-strip.png"));
    save_png(
        &live.pixels,
        live.width,
        live.height,
        &format!("{label}-final.png"),
    );
}
