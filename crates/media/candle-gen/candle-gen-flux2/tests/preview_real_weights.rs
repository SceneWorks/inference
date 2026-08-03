//! sc-16955 — candle FLUX.2 per-step latent **preview** real-weight validation (epic 16948).
//!
//! Three things a shape-only smoke cannot establish, and which this epic requires of every wiring
//! story:
//!
//! 1. **The reused fit belongs to this latent space.** FLUX.2 adds no fit; `crate::preview` reuses the
//!    epic-16624 thirty-two-channel constants `mlx-gen-flux2` committed.
//!    [`the_flux2_family_ships_one_learned_vae_in_two_container_widths`] is the reuse gate: the bf16
//!    container the fit was measured against and the f32 container `flux2_dev` (and Lens) load are two
//!    widths of **one** checkpoint — all 250 learned tensors of the f32 file round, round-to-nearest-even,
//!    exactly onto the bf16 file's bits.
//!    [`the_boogu_vae_is_not_the_flux2_one_and_that_is_why_boogu_is_unwired`] is the negative half, and
//!    the executable form of this story's Boogu adjudication.
//! 2. **The frames actually develop.** [`flux2_klein_preview_frames_evolve_toward_the_final_image`]
//!    drives the registered route through the `Generator` seam with a live sink, checks numbering,
//!    checks seeded byte-identity against an inert render, and measures that every frame is closer to —
//!    and more like — the finished image than the one before it. The strip is written out for review.
//! 3. **One frame per OUTER step on a multi-eval solver.**
//!    [`a_multi_eval_solver_emits_one_frame_per_outer_step`] runs `heun` and proves the guard is
//!    non-vacuous first: the shared driver calls `on_progress` once per *evaluation*, so a two-eval
//!    solver must produce strictly more progress events than steps before "frames == steps" means
//!    anything.
//!
//! [`the_flow_cohort_needs_no_sigma_correction`] is the row for this family's σ convention. It is the
//! **only non-`#[ignore]`d row in this file** and runs on the committed constants alone — deliberately,
//! because sc-16954 shipped a red row that hid behind `-- --ignored` excluding it. Run this file both
//! ways.
//!
//! ```sh
//! FLUX2_PREVIEW_SNAPSHOT=E:\huggingface\hub\models--black-forest-labs--FLUX.2-klein-9B\snapshots\<rev> \
//! FLUX2_FIT_VAE=...\models--black-forest-labs--FLUX.2-klein-9B\snapshots\<rev>\vae\diffusion_pytorch_model.safetensors \
//! FLUX2_F32_VAE=...\models--black-forest-labs--FLUX.2-dev\snapshots\<rev>\vae\diffusion_pytorch_model.safetensors \
//! FLUX2_BOOGU_VAE=...\models--Boogu--Boogu-Image-0.1-Turbo\snapshots\<rev>\vae\diffusion_pytorch_model.safetensors \
//! FLUX2_PREVIEW_ARTIFACT_DIR=E:\out\sc-16955 \
//!   cargo test -p candle-gen-flux2 --release --features cuda --test preview_real_weights \
//!     -- --ignored --nocapture
//! ```
//!
//! Every input is **required** by the row that uses it: a row that early-returns on an unset variable
//! still reports SUCCESS, and in a run log a skipped gate is indistinguishable from one that ran and
//! proved something. Asking for `--ignored` is already the opt-in.

use std::collections::BTreeMap;
use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex};

use candle_gen::gen_core::{
    GenerationOutput, GenerationRequest, Image, LoadSpec, PreviewFrame, PreviewSink, Progress,
    WeightsSource,
};

const PROMPT: &str =
    "A weathered lighthouse on a rocky headland at golden hour, warm sunlight, dramatic clouds, \
     highly detailed photograph.";
const SEED: u64 = 16955;

/// The SHA-256 of the **bf16** container of the FLUX.2 `AutoencoderKLFlux2` — 168,120,878 bytes, 251
/// tensors (250 learned bf16 + the unused `bn.num_batches_tracked` I64 counter).
///
/// Published byte-identically by `black-forest-labs/FLUX.2-klein-9B`
/// @ `92196c8e11f7b6cf2b7493e037d8c5345c559216` and by **every** tier (`bf16`/`q4`/`q8`) of the
/// `SceneWorks/flux2-klein-9b-mlx` @ `1d36c68041725a14c76566cdf6cea4270b264b03` and
/// `SceneWorks/flux2-klein-9b-kv-mlx` @ `fc6579b25dcb7e1bce85dd27cb9b901312110bab` re-hosts — the MLX
/// packer mirrors the VAE dense rather than packing it. This is the file the epic-16624 fit was
/// measured against (`mlx-gen-flux2/src/preview.rs`: eight FLUX.2 **Klein** renders).
pub const FLUX2_BF16_VAE_SHA256: &str =
    "ca70d2202afe6415bdbcb8793ba8cd99fd159cfe6192381504d6c4d3036e0f04";

/// The SHA-256 of the **f32** container of the same VAE — 336,213,556 bytes, the same 251 tensors at
/// double the width.
///
/// Published byte-identically by `black-forest-labs/FLUX.2-dev` @ `26afe3a78bb242c0a8bb181dcc8937bb16e5c66c`,
/// `SceneWorks/flux2-dev-mlx` @ `0c9b86f4d91eeaec3db11bcc9cc0e4c006faed74` (q4 + q8),
/// `SceneWorks/Lens` @ `5c5521d4417a3cae55816929ece69319d1e7712a`, `Comfy-Org/Lens`
/// @ `198d6ddf4d9fac0d8b0548dc9be4310452f5c146` (as `vae/flux2-vae.safetensors`), and every tier of
/// `SceneWorks/lens-mlx` @ `4e1349c1962950eee328c69537904631ebc64283` and `SceneWorks/lens-turbo-mlx`
/// @ `d3f485c320039595cff16d4f686a5f9378714f25`.
///
/// A **different file** from the bf16 one, which is exactly why the tensor-level row below exists and
/// a hash equality would not have done.
pub const FLUX2_F32_VAE_SHA256: &str =
    "d64f3a68e1cc4f9f4e29b6e0da38a0204fe9a49f2d4053f0ec1fa1ca02f9c4b5";

/// The SHA-256 of `Boogu/Boogu-Image-0.1-Turbo` @ `7c475e94ddb10529daa9142942d297675dde1acc`'s
/// `vae/diffusion_pytorch_model.safetensors` — 335,306,212 bytes, **244** f32 tensors and **no**
/// `bn.*` stats.
///
/// A plain 16-channel `AutoencoderKL` (the FLUX.1 / Z-Image lineage; `crate`'s sibling
/// `candle-gen-boogu` loads it through `candle_transformers::models::z_image::vae::AutoEncoderKL`),
/// not an `AutoencoderKLFlux2`. This is why sc-16955 leaves Boogu unwired.
const BOOGU_VAE_SHA256: &str = "8c717328c8ad41faab2ccfd52ae17332505c6833cf176aad56e7b58f2c4d4c94";

/// The measured extent of the transfer, pinned so a partial comparison cannot pass as a full one.
pub const VAE_LEARNED_TENSORS: usize = 250;
pub const VAE_LEARNED_VALUES: usize = 84_046_371;

/// The one tensor that is not part of the learned map: BatchNorm's forward-pass counter. It is read by
/// nothing — `Flux2Vae::build` loads `bn.running_mean` and `bn.running_var` and never this — and it is
/// where the Ideogram containers differ from these two.
pub const VAE_UNUSED_COUNTER: &str = "bn.num_batches_tracked";

fn env_path(name: &str) -> Option<PathBuf> {
    std::env::var(name).ok().map(PathBuf::from)
}

/// An input a row cannot run without. Missing means **fail**, not skip.
pub fn required_path(name: &str) -> PathBuf {
    env_path(name).unwrap_or_else(|| {
        panic!(
            "{name} must be set for this row — skipping it would report success while proving \
             nothing"
        )
    })
}

fn env_u32(name: &str, fallback: u32) -> u32 {
    std::env::var(name)
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(fallback)
}

fn artifact_dir() -> PathBuf {
    env_path("FLUX2_PREVIEW_ARTIFACT_DIR")
        .unwrap_or_else(|| std::env::temp_dir().join("flux2_preview_sc16955"))
}

pub fn sha256_of(path: &Path) -> String {
    use sha2::{Digest, Sha256};
    let mut file = std::fs::File::open(path).unwrap_or_else(|e| panic!("open {path:?}: {e}"));
    let mut hasher = Sha256::new();
    std::io::copy(&mut file, &mut hasher).expect("hash the file");
    format!("{:x}", hasher.finalize())
}

/// Load a `.safetensors` file's tensors, keyed and ordered by name.
pub fn tensors_of(path: &Path) -> BTreeMap<String, candle_gen::candle_core::Tensor> {
    candle_gen::candle_core::safetensors::load(path, &candle_gen::candle_core::Device::Cpu)
        .unwrap_or_else(|e| panic!("parse {path:?}: {e}"))
        .into_iter()
        .collect()
}

// ── Provenance: the reuse is grounded in tensor bytes ─────────────────────────────────────────────

/// **The reuse gate.** The FLUX.2 family ships one learned VAE in two container widths, and the fit
/// donor is the bf16 one.
///
/// This is the row a hash equality could not have replaced: `flux2_klein_9b` loads the bf16 file and
/// `flux2_dev` loads the f32 file, so "both crates name `Flux2Vae`" proves nothing. What is proven
/// here is that every one of the 250 learned tensors in the f32 file rounds — round-to-nearest-even,
/// the same rounding a bf16 cast performs — exactly onto the bf16 file's bits. Two widths of one
/// checkpoint, not two fine-tunes, so the fit's input domain is the same in both.
#[test]
#[ignore = "needs FLUX2_FIT_VAE + FLUX2_F32_VAE; run with --ignored"]
fn the_flux2_family_ships_one_learned_vae_in_two_container_widths() {
    use candle_gen::candle_core::DType;

    let bf16_path = required_path("FLUX2_FIT_VAE");
    let f32_path = required_path("FLUX2_F32_VAE");

    let (bf16_sha, f32_sha) = (sha256_of(&bf16_path), sha256_of(&f32_path));
    eprintln!("  bf16 (fit donor): {bf16_sha}  {}", bf16_path.display());
    eprintln!("  f32  (dev/lens) : {f32_sha}  {}", f32_path.display());
    assert_eq!(
        bf16_sha, FLUX2_BF16_VAE_SHA256,
        "FLUX2_FIT_VAE is not the file the epic-16624 32-channel fit was measured against; \
         re-derive the fit with mlx-gen-flux2/tests/fit_preview_rgb.rs before reusing it"
    );
    assert_eq!(f32_sha, FLUX2_F32_VAE_SHA256, "FLUX2_F32_VAE moved");
    assert_eq!(
        std::fs::metadata(&bf16_path).expect("stat").len(),
        168_120_878
    );
    assert_eq!(
        std::fs::metadata(&f32_path).expect("stat").len(),
        336_213_556
    );
    assert_ne!(
        bf16_sha, f32_sha,
        "if these ever became one file this row's rounding argument would be dead code"
    );

    let (wide, narrow) = (tensors_of(&f32_path), tensors_of(&bf16_path));
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
            // The one tensor that is not part of the learned map, and the one the Ideogram containers
            // differ on. Asserted as an exception rather than skipped silently.
            assert_eq!(w.dtype(), DType::I64);
            assert_eq!(n.dtype(), DType::I64);
            continue;
        }
        assert_eq!(
            w.dtype(),
            DType::F32,
            "{key}: the wide container must be f32"
        );
        assert_eq!(
            n.dtype(),
            DType::BF16,
            "{key}: the narrow container must be bf16"
        );
        // The comparison IS the bf16 cast: widening the bf16 side instead would silently accept an f32
        // value that merely rounds close, rather than one that rounds exactly onto these bits.
        let cast = w
            .to_dtype(DType::BF16)
            .expect("cast the f32 tensor to bf16");
        let (a, b) = (widened_bf16(&cast), widened_bf16(n));
        assert_eq!(
            a, b,
            "{key}: the f32 tensor does not round onto the bf16 one"
        );
        values += a.len();
    }
    assert_eq!(
        values, VAE_LEARNED_VALUES,
        "the comparison must cover every learned value"
    );
    eprintln!("  {VAE_LEARNED_TENSORS} learned tensors, {values} values: bf16-round-identical");

    assert_eq!(candle_gen_flux2::preview::PREVIEW_LATENT_CHANNELS, 32);
    assert_eq!(candle_gen_flux2::preview::PACKED_LATENT_CHANNELS, 128);
}

/// A bf16 tensor widened to f32, exactly.
///
/// Comparing widened values is equivalent to comparing the 16-bit patterns: bf16 → f32 is lossless
/// and injective, so two bf16 tensors widen to equal f32 vectors **iff** their bit patterns match
/// (weights carry no NaN, the only case where that equivalence would leak). Doing it this way keeps
/// the comparison inside candle rather than adding a `half` dependency for a test.
pub fn widened_bf16(tensor: &candle_gen::candle_core::Tensor) -> Vec<f32> {
    tensor
        .to_dtype(candle_gen::candle_core::DType::F32)
        .and_then(|t| t.flatten_all())
        .and_then(|t| t.to_vec1::<f32>())
        .expect("widen a bf16 tensor to f32")
}

/// The negative half of the adjudication, and the executable form of this story's Boogu decision.
///
/// The epic groups Boogu with FLUX.2 and sc-16955's acceptance criterion is that it emits **only** if
/// its VAE is proven to be the FLUX.2 one. It is not: a different file, a different tensor count, a
/// different architecture (no `bn.*` stats at all — the BatchNorm normalization is the defining
/// feature of `AutoencoderKLFlux2`), and a 16-channel latent rather than 32. Left as a row rather than
/// as prose so that a future snapshot swap which *did* make them the same file would be noticed.
#[test]
#[ignore = "needs FLUX2_BOOGU_VAE; run with --ignored"]
fn the_boogu_vae_is_not_the_flux2_one_and_that_is_why_boogu_is_unwired() {
    let boogu = required_path("FLUX2_BOOGU_VAE");
    let sha = sha256_of(&boogu);
    eprintln!("  boogu vae: {sha}  {}", boogu.display());
    assert_eq!(sha, BOOGU_VAE_SHA256, "the staged Boogu VAE moved");
    assert_ne!(sha, FLUX2_BF16_VAE_SHA256);
    assert_ne!(sha, FLUX2_F32_VAE_SHA256);

    let tensors = tensors_of(&boogu);
    assert_eq!(
        tensors.len(),
        244,
        "Boogu ships a plain AutoencoderKL, not the 251-tensor AutoencoderKLFlux2"
    );
    assert!(
        !tensors.keys().any(|k| k.starts_with("bn.")),
        "Boogu's VAE has no BatchNorm stats — the packed-space normalization that DEFINES the FLUX.2 \
         latent space — so the 32-channel fit does not describe it"
    );
    // The channel count is the decisive number, read off the decoder's input conv.
    let conv_in = tensors
        .get("decoder.conv_in.weight")
        .expect("a decoder input conv");
    assert_eq!(
        conv_in.dims()[1],
        16,
        "Boogu decodes a 16-channel latent; the committed fit is over 32 channels ({}), so wiring it \
         here would ship a borrowed fit",
        candle_gen_flux2::preview::PREVIEW_LATENT_CHANNELS
    );
}

// ── Frame analysis helpers ────────────────────────────────────────────────────────────────────────

pub fn mean_abs_delta(a: &[u8], b: &[u8]) -> f64 {
    assert_eq!(a.len(), b.len(), "compared buffers must match in length");
    a.iter()
        .zip(b)
        .map(|(&x, &y)| (x as f64 - y as f64).abs())
        .sum::<f64>()
        / a.len() as f64
}

pub fn correlation(a: &[u8], b: &[u8]) -> f64 {
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

/// Nearest-neighbour box resample of an RGB8 buffer.
pub fn downsample_raw(pixels: &[u8], src_w: u32, src_h: u32, w: u32, h: u32) -> Vec<u8> {
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

pub fn downsample(img: &Image, w: u32, h: u32) -> Vec<u8> {
    downsample_raw(&img.pixels, img.width, img.height, w, h)
}

pub fn save_png(dir: &Path, pixels: &[u8], width: u32, height: u32, name: &str) {
    std::fs::create_dir_all(dir).expect("create the artifact dir");
    let path = dir.join(name);
    let buf: image::RgbImage = image::ImageBuffer::from_raw(width, height, pixels.to_vec())
        .expect("frame buffer matches its dimensions");
    buf.save(&path)
        .unwrap_or_else(|e| panic!("save {path:?}: {e}"));
    eprintln!("  wrote {}", path.display());
}

/// Lay the strip out as one horizontal contact sheet so a reviewer sees the progression at a glance.
pub fn save_strip(dir: &Path, frames: &[PreviewFrame], name: &str) {
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
    save_png(dir, &sheet, total_w, fh, name);
}

pub fn one_image(out: GenerationOutput) -> Image {
    let GenerationOutput::Images(mut images) = out else {
        panic!("expected GenerationOutput::Images");
    };
    assert_eq!(images.len(), 1, "these rows render a single image");
    images.pop().expect("one image")
}

pub fn collecting_sink() -> (PreviewSink, Arc<Mutex<Vec<PreviewFrame>>>) {
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

/// The shared strip analysis: numbering, latent resolution, per-frame movement, falling distance to
/// the finished image, and rising resemblance to it. Applied identically to every lane in the FLUX.2
/// family (this crate, `candle-gen-lens`, `candle-gen-ideogram`) so none can be closed with a weaker
/// measurement than the others.
///
/// ## What the correlation floor is, and what it is NOT
///
/// A projection cannot correlate with the decode better than the fit does, so the fit fixes a
/// **ceiling**: the FLUX.2 fit R² is `0.76409` (`mlx-gen-flux2/src/preview.rs`), a correlation ceiling
/// of √0.76409 ≈ `0.874`. The in-sample R² is the like-for-like statistic — the 16-channel QwenVae
/// families were held against an in-sample 0.9586 and sc-16954 matched that with SDXL's in-sample
/// 0.91849 — so the holdout 0.69504 is deliberately not used here.
///
/// What the ceiling does **not** fix is the floor, and conflating the two is the trap this comment
/// exists for. `min_r_last` also measures *how far the trajectory has travelled one step from the end*,
/// which is a property of the **schedule**: the hook emits BEFORE each solver step (sc-16949), so the
/// final advancement is never previewed. sc-16954 recorded exactly this, and set per-lane floors 0.83
/// (12-step ancestral) and 0.55 (few-step Euler-trailing) for the same fit.
///
/// FLUX.2's empirical-μ flow schedule is strongly **back-loaded**, so its unpreviewed terminal step is
/// a large share of the trajectory — and the measurement says so directly. On the 12-step klein lane
/// the last frame reaches r `+0.556`, and frame-to-frame movement over the strip runs
/// `0.448 → 10.321`: the final previewed step alone moves more than the first nine combined.
/// Lengthening the schedule moves the last frame toward the ceiling exactly as that explanation
/// predicts — at 28 steps the same render reaches r `+0.663` and its distance fall rises from 32.7% to
/// 40.3% — which is why the shortfall is read as the schedule rather than as the wiring.
///
/// So `min_r_last` is a per-lane "the strip never got close" backstop and nothing more. The
/// load-bearing assertions are the three strict monotonicities (movement accelerating, distance
/// falling, resemblance rising) and the ≥ 0.30 total rise, none of which a stale, duplicated or
/// wrongly-scaled latent could reproduce.
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
#[allow(clippy::too_many_arguments)]
pub fn assert_the_strip_converges(
    label: &str,
    frames: &[PreviewFrame],
    final_image: &Image,
    steps: u32,
    latent_w: u32,
    latent_h: u32,
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

    // Raw-latent resolution, and batch 1. A CFG-fused `[2, …]` latent fails the packed-layout contract
    // outright, so a strip that exists at all is already proof the preview never saw a fused
    // unconditional half — there would be no frames if it had.
    for frame in frames {
        assert_eq!(
            (frame.image.width, frame.image.height),
            (latent_w, latent_h),
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

    let target = downsample(final_image, latent_w, latent_h);
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

fn base_request(steps: u32, size: u32, sampler: Option<&str>) -> GenerationRequest {
    GenerationRequest {
        prompt: PROMPT.into(),
        width: size,
        height: size,
        count: 1,
        seed: Some(SEED),
        steps: Some(steps),
        sampler: sampler.map(str::to_string),
        ..GenerationRequest::default()
    }
}

/// Render the registered klein route twice on one warmed generator at the same seed — once inert, once
/// live — and hold the strip to [`assert_the_strip_converges`]. Returns the live run's
/// `Progress::Step` count (which IS its evaluation count) and its frames.
fn assert_klein_previews_converge(
    label: &str,
    sampler: Option<&str>,
    steps: u32,
    size: u32,
    min_r_last: f64,
    min_acceleration: f64,
) -> (usize, Vec<PreviewFrame>) {
    eprintln!("── {label}: {size}² × {steps} steps, sampler {sampler:?}");
    let spec = LoadSpec::new(WeightsSource::Dir(required_path("FLUX2_PREVIEW_SNAPSHOT")));
    let generator =
        candle_gen_flux2::load_klein(&spec).unwrap_or_else(|e| panic!("load flux2: {e}"));

    // N1: the inert baseline. Same generator, same seed, no sink.
    let mut noop = |_: Progress| {};
    let inert = one_image(
        generator
            .generate(&base_request(steps, size, sampler), &mut noop)
            .unwrap_or_else(|e| panic!("{label}: inert render: {e}")),
    );

    let (sink, frames) = collecting_sink();
    let mut request = base_request(steps, size, sampler);
    request.preview = sink;
    let mut events = 0usize;
    let mut count_progress = |p: Progress| {
        if matches!(p, Progress::Step { .. }) {
            events += 1;
        }
    };
    let live = one_image(
        generator
            .generate(&request, &mut count_progress)
            .unwrap_or_else(|e| panic!("{label}: live render: {e}")),
    );

    // An active sink must not move a single bit of the render.
    assert_eq!(
        inert.pixels, live.pixels,
        "{label}: attaching a live preview sink changed the seeded render"
    );

    let frames = candle_gen::lock_recover(&frames).clone();
    let dir = artifact_dir();
    assert_the_strip_converges(
        label,
        &frames,
        &live,
        steps,
        size / 8,
        size / 8,
        min_r_last,
        min_acceleration,
    );
    save_strip(&dir, &frames, &format!("{label}-strip.png"));
    save_png(
        &dir,
        &live.pixels,
        live.width,
        live.height,
        &format!("{label}-final.png"),
    );
    (events, frames)
}

/// The registered route's shipped lane: `run_flow_sampler` over `FlowModelSampling`, which is what
/// every FLUX.2 request takes, including one that names no sampler at all.
#[test]
#[ignore = "needs FLUX2_PREVIEW_SNAPSHOT + a CUDA GPU; run with --features cuda --ignored"]
fn flux2_klein_preview_frames_evolve_toward_the_final_image() {
    let steps = env_u32("FLUX2_PREVIEW_STEPS", 12);
    let size = env_u32("FLUX2_PREVIEW_SIZE", 1024);
    // 0.50 against a measured +0.556 on this lane. See `assert_the_strip_converges` for why this is a
    // backstop rather than a fit-derived number: the empirical-μ flow schedule's unpreviewed terminal
    // step is a large share of the trajectory.
    assert_klein_previews_converge("flux2-klein-euler", None, steps, size, 0.50, 5.0);
}

/// Exactly one frame per **outer** solver step on a multi-eval solver.
///
/// The guard is made non-vacuous first, and in the strongest available way: the shared driver calls
/// `on_progress` once per *evaluation* (`sampler.rs` computes the step count on every eval and
/// deliberately repeats it), so counting `Progress::Step` events IS counting evaluations. If `heun`
/// did not evaluate twice per step the event count would equal the step count and "frames == steps"
/// would prove nothing — so that inequality is asserted before the frame count is.
#[test]
#[ignore = "needs FLUX2_PREVIEW_SNAPSHOT + a CUDA GPU; run with --features cuda --ignored"]
fn a_multi_eval_solver_emits_one_frame_per_outer_step() {
    let steps = 8u32;
    let (events, _) =
        assert_klein_previews_converge("flux2-klein-heun", Some("heun"), steps, 768, 0.50, 3.0);
    eprintln!("  heun: {events} evaluations for {steps} outer steps");
    assert!(
        events > steps as usize,
        "heun must evaluate more than once per outer step or this row proves nothing about the \
         preview counter's dedup ({events} events for {steps} steps)"
    );
    // `assert_klein_previews_converge` already required exactly `steps` frames numbered 1..=steps, so
    // the dedup collapsed the extra evaluations. Stated here because that is the point of the row.
}

/// This family's σ-convention finding, measured rather than argued — and the counterpart of sc-16954's
/// VE-correction row, which found the **opposite** for the discrete ε cohort.
///
/// `run_flow_sampler` integrates a `FlowModelSampling` whose `input_scale` is exactly `1.0` at every σ,
/// so the running latent already *is* the tensor the fit was measured against and no `with_sigma`
/// correction is needed. The cheap decisive signal sc-16954 named is the first frame's rail-clipped
/// fraction: SDXL's uncorrected projection clipped 89.4% of pixels to 0/255, which is what a missing
/// input scaling looks like. Here the same measurement is taken on the latent this family's first
/// emission actually sees — flow priors are unit-normal, `σ_max = 1.0` — and it must come out readable.
///
/// Runs on the committed constants alone, no weights, and is deliberately **not** `#[ignore]`d: it is
/// the row that must appear in a plain `cargo test` of this file. sc-16954 shipped a red row that hid
/// because the only non-ignored row in its file was excluded by `-- --ignored`.
#[test]
fn the_flow_cohort_needs_no_sigma_correction() {
    use candle_gen::candle_core::{Device, Tensor};
    use candle_gen::gen_core::sampling::{FlowModelSampling, ModelSampling, TimestepConvention};

    // The convention, first: the claim is about `input_scale`, so it is read off the very
    // `ModelSampling` the driver integrates rather than asserted about the family in prose.
    let ms = FlowModelSampling::new(TimestepConvention::Sigma);
    for sigma in [0.0f32, 0.05, 0.25, 0.5, 0.75, 1.0] {
        assert_eq!(
            ms.input_scale(sigma),
            1.0,
            "FlowModelSampling::input_scale must be identically 1.0; at {sigma} it is not, and this \
             family would need PreviewHook::with_sigma"
        );
    }

    // The consequence, measured. A unit-normal packed latent at σ_max = 1.0 is what the first emission
    // sees. `bn_std`/`bn_mean` are the identity here so the measurement isolates the projection: a
    // real VAE's stats rescale the latent, and the row would then be measuring the checkpoint.
    let (lat_h, lat_w) = (32usize, 32usize);
    let mut rng = <rand::rngs::StdRng as rand::SeedableRng>::seed_from_u64(SEED);
    let noise = candle_gen::seeded_normal_vec(&mut rng, lat_h * lat_w * 128);
    let tokens = Tensor::from_vec(noise, (1, lat_h * lat_w, 128), &Device::Cpu).expect("latent");
    let std = Tensor::ones(
        (1, 128, 1, 1),
        candle_gen::candle_core::DType::F32,
        &Device::Cpu,
    )
    .expect("std");
    let mean = Tensor::zeros(
        (1, 128, 1, 1),
        candle_gen::candle_core::DType::F32,
        &Device::Cpu,
    )
    .expect("mean");

    let frame =
        candle_gen_flux2::preview::project_packed_tokens(&tokens, &std, &mean, lat_h, lat_w)
            .expect("project the first-emission latent");
    let rails = frame.pixels.iter().filter(|&&v| v == 0 || v == 255).count() as f64
        / frame.pixels.len() as f64;
    eprintln!("  flow prior at sigma_max: rail-clipped fraction {rails:.4}");
    // Measured on this seeded latent: 0.0000. The bound is loose enough that a rounding change cannot
    // flip it and far below sc-16954's uncorrected SDXL 0.894, which is the number it is being
    // contrasted with.
    assert!(
        rails < 0.05,
        "an uncorrected flow-space projection must already be a readable noise field, not a clipped \
         one ({rails:.4}) — if this ever fails, the family needs PreviewHook::with_sigma"
    );
}
