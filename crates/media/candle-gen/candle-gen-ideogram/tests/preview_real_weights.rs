//! sc-16955 — candle **Ideogram 4** per-step latent preview real-weight validation (epic 16948).
//!
//! Ideogram is the epic's first genuine **bespoke** wired crate: it drives no shared sampler at all, so
//! it emits through `candle_gen::preview::emit_preview_at` from inside `pipeline::denoise`. Three
//! things have to be established here:
//!
//! 1. **The VAE Ideogram loads is the one the fit was measured over.**
//!    [`the_ideogram_vae_is_the_flux2_fit_donor_tensor_for_tensor`] is the reuse gate. Ideogram
//!    publishes its own `vae/model.safetensors`, so a hash equality is unavailable: every one of the
//!    250 learned tensors is compared against the FLUX.2-klein donor and must be **byte-identical**.
//!    Only the unused `bn.num_batches_tracked` counter differs (in value, and on the packed re-host in
//!    integer dtype) — asserted as an exception rather than skipped.
//! 2. **The frames actually develop on a bespoke lane.**
//!    [`ideogram_preview_frames_evolve_toward_the_final_image`] drives the registered route through
//!    the `Generator` seam with a live sink, checks numbering, checks seeded byte-identity against an
//!    inert render, and measures convergence. Wiring only shared-driver families would have left this
//!    whole crate dark on an advertised route.
//! 3. **The `(ph,pw,c)` recovery is the one the decode uses.** The frames and the finished image are
//!    recovered by one function (`pipeline::raw_latent`), so a frame that resembles the render is also
//!    evidence the patch-major unpack is right — the trap `mlx-gen-flux2/src/preview.rs` names.
//!
//! ```sh
//! IDEOGRAM_PREVIEW_SNAPSHOT=E:\huggingface\hub\models--SceneWorks--ideogram-4\snapshots\<rev>\bf16
//! IDEOGRAM_PREVIEW_VAE=...\bf16\vae\model.safetensors
//! IDEOGRAM_FLUX2_FIT_VAE=...\models--black-forest-labs--FLUX.2-klein-9B\snapshots\<rev>\vae\diffusion_pytorch_model.safetensors
//! IDEOGRAM_PREVIEW_ARTIFACT_DIR=E:\out\sc-16955
//!   cargo test -p candle-gen-ideogram --release --features cuda --test preview_real_weights
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

/// The SHA-256 of the `vae/model.safetensors` `SceneWorks/ideogram-4`
/// @ `2e8fb610109bf0db195344cc424df98b301d3cad` publishes under `bf16/` — 168,120,846 bytes, 251
/// tensors (250 learned bf16 + the `bn.num_batches_tracked` I64 counter).
///
/// A **different file** from the FLUX.2 fit donor, by exactly 32 bytes: the donor's safetensors header
/// carries a `__metadata__` block and this one does not. The tensor data is the same bytes, which is
/// what the row below establishes and what a hash equality could not have.
const IDEOGRAM_VAE_SHA256: &str =
    "00089549a43994958293780eaecf43ed44c4e2680b2241a8a7d3578cc2ae409b";

/// The SHA-256 of the packed re-host's VAE, `SceneWorks/ideogram-4-mlx`
/// @ `a3095855b8819dc0d6b067cb1354aaa7da189ff8`, identical across its `q4/` and `q8/` tiers —
/// 168,120,870 bytes. A **third** container of the same 250 learned tensors; it differs from the donor
/// only in the unused counter's integer dtype (I32 rather than I64), which is the difference
/// `mlx-gen-flux2/src/preview.rs` records. Pinned so the packed tiers are covered by name even when
/// the dense snapshot is the one staged for a render.
const IDEOGRAM_PACKED_VAE_SHA256: &str =
    "bb9ba30dec375f7fef52a4e47cda26e9354082710849d531df69eca724ce3bc9";

/// The SHA-256 of the **bf16** container the epic-16624 32-channel fit was measured against —
/// 168,120,878 bytes, `black-forest-labs/FLUX.2-klein-9B`
/// @ `92196c8e11f7b6cf2b7493e037d8c5345c559216`.
const FLUX2_FIT_VAE_SHA256: &str =
    "ca70d2202afe6415bdbcb8793ba8cd99fd159cfe6192381504d6c4d3036e0f04";

/// The measured extent of the transfer, pinned so a partial comparison cannot pass as a full one.
const VAE_LEARNED_TENSORS: usize = 250;
const VAE_LEARNED_VALUES: usize = 84_046_371;

/// BatchNorm's forward-pass counter — the one tensor that is not part of the learned map, read by
/// nothing (`Flux2Vae::build` loads `bn.running_mean` and `bn.running_var` and never this), and the
/// only tensor on which the Ideogram containers differ from the donor.
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
    env_path("IDEOGRAM_PREVIEW_ARTIFACT_DIR")
        .unwrap_or_else(|| std::env::temp_dir().join("ideogram_preview_sc16955"))
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

/// **The reuse gate.** The VAE Ideogram loads and the VAE the epic-16624 32-channel fit was measured
/// on hold the same 250 learned tensors, byte for byte.
///
/// Ideogram publishes its own file, so this is a tensor comparison rather than a hash equality — and
/// it lands on the strongest available result: not "rounds onto", not merely "value-identical", but
/// the same bf16 bit patterns in every one of the 250 learned tensors. The single exception is the
/// unused `bn.num_batches_tracked` counter, which differs in value (and on the packed re-host in
/// integer dtype) and is asserted as an exception rather than skipped, so a future container that
/// differed somewhere *else* could not hide behind it.
///
/// Both inputs are required. There is no configuration of this row that passes without performing the
/// comparison.
#[test]
#[ignore = "needs IDEOGRAM_PREVIEW_VAE + IDEOGRAM_FLUX2_FIT_VAE; run with --ignored"]
fn the_ideogram_vae_is_the_flux2_fit_donor_tensor_for_tensor() {
    let ideogram_path = required_path("IDEOGRAM_PREVIEW_VAE");
    let donor_path = required_path("IDEOGRAM_FLUX2_FIT_VAE");

    let (ideogram_sha, donor_sha) = (sha256_of(&ideogram_path), sha256_of(&donor_path));
    eprintln!(
        "  ideogram vae/ {ideogram_sha}  {}",
        ideogram_path.display()
    );
    eprintln!("  donor    vae/ {donor_sha}  {}", donor_path.display());
    assert!(
        ideogram_sha == IDEOGRAM_VAE_SHA256 || ideogram_sha == IDEOGRAM_PACKED_VAE_SHA256,
        "the VAE this Ideogram snapshot publishes is neither of the two files the reuse was grounded \
         against (dense {IDEOGRAM_VAE_SHA256}, packed {IDEOGRAM_PACKED_VAE_SHA256})"
    );
    assert_eq!(
        donor_sha, FLUX2_FIT_VAE_SHA256,
        "IDEOGRAM_FLUX2_FIT_VAE must be the snapshot the epic-16624 32-channel fit was measured on"
    );
    assert_ne!(
        ideogram_sha, donor_sha,
        "the two files are deliberately different containers of the same tensors — if they ever \
         became byte-identical this row's tensor argument would be dead code"
    );

    let (ideogram, donor) = (tensors_of(&ideogram_path), tensors_of(&donor_path));
    assert_eq!(
        ideogram.keys().collect::<Vec<_>>(),
        donor.keys().collect::<Vec<_>>(),
        "the two containers must hold the same key set"
    );
    assert_eq!(ideogram.len(), VAE_LEARNED_TENSORS + 1);

    let mut values = 0usize;
    for (key, a) in &ideogram {
        let b = &donor[key];
        assert_eq!(a.dims(), b.dims(), "{key}: shapes must match");
        if key == VAE_UNUSED_COUNTER {
            // The documented exception, and the ONLY one permitted. The dense container matches the
            // donor's I64 dtype and the packed re-host narrows it to I32; neither is ever read.
            assert!(
                matches!(a.dtype(), DType::I64 | DType::I32),
                "{key}: expected an integer counter, got {:?}",
                a.dtype()
            );
            continue;
        }
        assert_eq!(a.dtype(), DType::BF16, "{key}: Ideogram publishes bf16");
        assert_eq!(b.dtype(), DType::BF16, "{key}: the donor is bf16");
        let (a, b) = (widened_bf16(a), widened_bf16(b));
        assert_eq!(
            a, b,
            "{key}: the Ideogram tensor is not byte-identical to the donor's"
        );
        values += a.len();
    }
    assert_eq!(
        values, VAE_LEARNED_VALUES,
        "the comparison must cover every learned value"
    );
    eprintln!("  {VAE_LEARNED_TENSORS} learned tensors, {values} values: bit-identical");

    // The fit Ideogram projects with is the FLUX.2 one, over 32 channels — re-exported, never copied.
    assert_eq!(candle_gen_ideogram::preview::PREVIEW_LATENT_CHANNELS, 32);
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

/// The last-frame resemblance floor for the Ideogram lane, and the one place in this epic where the
/// **fit-derived** number is actually reachable.
///
/// Ideogram's `LogitNormalSchedule` spreads its steps far more evenly than the FLUX.2 empirical-μ one,
/// so the never-previewed terminal advancement is a small share of the trajectory: the measured last
/// frame reaches r `+0.828`, which is 95% of the fit's `√0.76409 ≈ 0.874` ceiling and comfortably
/// above the 0.75 that 86.8%-of-ceiling derivation gives. The FLUX.2 and Lens lanes sit below it for a
/// schedule reason their own acceleration measurements make visible (23x and 15.7x against Ideogram's
/// 1.9x), which is exactly why neither floor is shared.
const IDEOGRAM_MIN_R_LAST: f64 = 0.75;

/// Ideogram's schedule is NOT back-loaded the way FLUX.2's is — measured 1.9x, against 23x for klein.
/// Asserting the FLUX.2 shape here would be asserting one family's schedule about another's.
const IDEOGRAM_MIN_ACCELERATION: f64 = 1.5;

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

/// Render the registered `ideogram_4` route twice on one warmed generator at the same seed — once
/// inert, once live — and hold the strip to [`assert_the_strip_converges`].
///
/// This is the row that proves the **bespoke** wiring: there is no shared-driver call site in this
/// crate, so nothing in the catalog's hooked-site scan could have covered it, and every frame here
/// came out of `pipeline::denoise`'s own `emit_preview_at`.
#[test]
#[ignore = "needs IDEOGRAM_PREVIEW_SNAPSHOT + a CUDA GPU; run with --features cuda --ignored"]
fn ideogram_preview_frames_evolve_toward_the_final_image() {
    let label = "ideogram-4";
    let steps = env_u32("IDEOGRAM_PREVIEW_STEPS", 12);
    let size = env_u32("IDEOGRAM_PREVIEW_SIZE", 1024);
    eprintln!("── {label}: {size}² × {steps} steps");

    let spec = LoadSpec::new(WeightsSource::Dir(required_path(
        "IDEOGRAM_PREVIEW_SNAPSHOT",
    )));
    let id = std::env::var("IDEOGRAM_PREVIEW_ROUTE")
        .unwrap_or_else(|_| candle_gen_ideogram::config::MODEL_ID.to_string());
    let generator = candle_gen_ideogram::provider_registry()
        .expect("ideogram registry")
        .load(&id, &spec)
        .unwrap_or_else(|e| panic!("load {id}: {e}"));

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
    assert_the_strip_converges(
        label,
        &frames,
        &live,
        steps,
        IDEOGRAM_MIN_R_LAST,
        IDEOGRAM_MIN_ACCELERATION,
    );
    save_strip(&frames, &format!("{label}-strip.png"));
    save_png(
        &live.pixels,
        live.width,
        live.height,
        &format!("{label}-final.png"),
    );
}
