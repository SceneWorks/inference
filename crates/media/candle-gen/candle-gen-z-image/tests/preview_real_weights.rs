//! sc-16957 — candle Z-Image per-step latent **preview** real-weight validation (epic 16948).
//!
//! Four things a shape-only smoke cannot establish, and which this epic requires of every wiring
//! story:
//!
//! 1. **The reused fit belongs to this latent space.** Z-Image adds no fit; `crate::preview` reuses
//!    the epic-16624 sixteen-channel constants `mlx-gen-z-image` committed.
//!    [`the_committed_fit_donor_is_the_shipped_z_image_vae`] is the reuse gate.
//! 2. **Which 16-channel space this actually is.** [`the_z_image_vae_is_the_flux1_one`] is the row the
//!    epic asked this story to settle: `Tongyi-MAI/Z-Image-Turbo`'s VAE is **byte-identical** to
//!    `black-forest-labs/FLUX.1-dev`'s, so epic 16624 committed **two** fits over **one** latent space.
//!    sc-16955 established that "16 channels" alone does not make two spaces the same and sc-16956
//!    answered it for Boogu; this answers it for Z-Image, in the opposite direction from what "its own
//!    committed fit" suggests.
//! 3. **The frames actually develop, on every wired lane.**
//!    [`z_image_preview_frames_evolve_toward_the_final_image`] covers the four registered-descriptor
//!    lanes (Turbo and base, resident and staged-residency);
//!    [`the_control_routes_preview_their_target_latent`] covers the name-driven Fun-ControlNet
//!    provider's two modes — one hooked, one a bespoke loop — and
//!    [`the_edit_route_previews_its_reduced_schedule`] covers the img2img provider's bespoke loop over
//!    the strength-reduced schedule. Every row renders twice at one seed, inert and live, checks
//!    seeded byte-identity, and measures monotone convergence. The strips are written out for review.
//! 4. **One frame per OUTER step on a multi-eval solver.**
//!    [`a_multi_eval_solver_emits_one_frame_per_outer_step`] runs `heun` and proves the guard is
//!    non-vacuous first: the shared driver calls `on_progress` once per *evaluation*, so a two-eval
//!    solver must produce strictly more progress events than steps before "frames == steps" means
//!    anything.
//!
//! [`the_flow_cohort_needs_no_sigma_correction`] is the row for this family's σ convention. It is the
//! **only non-`#[ignore]`d row in this file** and runs on the committed constants alone —
//! deliberately, because sc-16954 shipped a red row that hid behind `-- --ignored` excluding it. Run
//! this file both ways.
//!
//! ```sh
//! ZIMAGE_TURBO_SNAPSHOT=E:\huggingface\hub\models--Tongyi-MAI--Z-Image-Turbo\snapshots\<rev> \
//! ZIMAGE_BASE_SNAPSHOT=E:\huggingface\hub\models--Tongyi-MAI--Z-Image\snapshots\<rev> \
//! ZIMAGE_FIT_VAE=E:\huggingface\hub\models--SceneWorks--z-image-turbo-mlx\snapshots\<rev>\bf16\vae\model.safetensors \
//! ZIMAGE_DIFFUSERS_VAE=...\models--Tongyi-MAI--Z-Image-Turbo\snapshots\<rev>\vae\diffusion_pytorch_model.safetensors \
//! ZIMAGE_BASE_DIFFUSERS_VAE=...\models--Tongyi-MAI--Z-Image\snapshots\<rev>\vae\diffusion_pytorch_model.safetensors \
//! ZIMAGE_FLUX1_VAE=...\models--black-forest-labs--FLUX.1-dev\snapshots\<rev>\vae\diffusion_pytorch_model.safetensors \
//! ZIMAGE_TURBO_CONTROL_NET=...\models--alibaba-pai--Z-Image-Turbo-Fun-Controlnet-Union-2.1\snapshots\<rev> \
//! ZIMAGE_BASE_CONTROL_NET=...\models--alibaba-pai--Z-Image-Fun-Controlnet-Union-2.1\snapshots\<rev> \
//! ZIMAGE_PREVIEW_POSE=E:\out\sc-16957\pose.ppm \
//! ZIMAGE_PREVIEW_EDIT_SOURCE=E:\out\sc-16957\source.ppm \
//! ZIMAGE_PREVIEW_ARTIFACT_DIR=E:\out\sc-16957 \
//!   cargo test -p candle-gen-z-image --release --features cuda --test preview_real_weights \
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
    GenerationMemory, GenerationOutput, GenerationRequest, Image, LoadSpec, PreviewFrame,
    PreviewSink, Progress, WeightsSource,
};

const PROMPT: &str =
    "A weathered lighthouse on a rocky headland at golden hour, warm sunlight, dramatic clouds, \
     highly detailed photograph.";
const SEED: u64 = 16957;

/// The SHA-256 of the container the epic-16624 Z-Image sixteen-channel fit was measured against —
/// `SceneWorks/z-image-turbo-mlx` @ `bb2bc9893b3c49ae96c813350775f791a2e8bc80`, **bf16** tier,
/// `vae/model.safetensors`, 167,666,968 bytes, 244 tensors.
///
/// `mlx-gen-z-image/src/preview.rs` names exactly this file, so it is the anchor every other container
/// is measured against.
const FIT_VAE_SHA256: &str = "0fbab8b661f6ee6af81c88a6eb1501ec1f7b4b8fe4ad29803507ebe0cf863810";

/// The SHA-256 of the plain **bf16 diffusers** container candle actually loads — 167,666,902 bytes,
/// 244 tensors.
///
/// Published byte-identically by `Tongyi-MAI/Z-Image-Turbo`, `Tongyi-MAI/Z-Image` **and**
/// `black-forest-labs/FLUX.1-dev`. That last identity is the finding
/// [`the_z_image_vae_is_the_flux1_one`] exists for, and this is the same constant sc-16956 pinned as
/// `DIFFUSERS_VAE_SHA256` in `candle-gen-flux/tests/preview_real_weights.rs`.
const DIFFUSERS_VAE_SHA256: &str =
    "f5b59a26851551b67ae1fe58d32e76486e1e812def4696a4bea97f16604d40a3";

/// The measured extent of each comparison, pinned so a partial one cannot pass as a full one.
const VAE_TENSORS: usize = 244;
const VAE_VALUES: usize = 83_819_683;

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

fn env_u32(name: &str, fallback: u32) -> u32 {
    std::env::var(name)
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(fallback)
}

fn artifact_dir() -> PathBuf {
    env_path("ZIMAGE_PREVIEW_ARTIFACT_DIR")
        .unwrap_or_else(|| std::env::temp_dir().join("z_image_preview_sc16957"))
}

fn sha256_of(path: &Path) -> String {
    use sha2::{Digest, Sha256};
    let mut file = std::fs::File::open(path).unwrap_or_else(|e| panic!("open {path:?}: {e}"));
    let mut hasher = Sha256::new();
    std::io::copy(&mut file, &mut hasher).expect("hash the file");
    format!("{:x}", hasher.finalize())
}

/// Load a `.safetensors` file's tensors, keyed and ordered by name.
fn tensors_of(path: &Path) -> BTreeMap<String, Tensor> {
    candle_gen::candle_core::safetensors::load(path, &Device::Cpu)
        .unwrap_or_else(|e| panic!("parse {path:?}: {e}"))
        .into_iter()
        .collect()
}

/// A tensor widened to f32, exactly.
///
/// Comparing widened values is equivalent to comparing the 16-bit patterns: bf16 → f32 is lossless and
/// injective, so two bf16 tensors widen to equal f32 vectors **iff** their bit patterns match (weights
/// carry no NaN, the only case where that equivalence would leak).
fn widened(tensor: &Tensor) -> Vec<f32> {
    tensor
        .to_dtype(DType::F32)
        .and_then(|t| t.flatten_all())
        .and_then(|t| t.to_vec1::<f32>())
        .expect("widen a tensor to f32")
}

/// Assert two containers hold the same 244 learned tensors, bit for bit, and return the value count.
fn assert_same_learned_tensors(label: &str, a: &Path, b: &Path) -> usize {
    let (left, right) = (tensors_of(a), tensors_of(b));
    assert_eq!(left.len(), VAE_TENSORS, "{label}: {a:?} tensor count");
    assert_eq!(right.len(), VAE_TENSORS, "{label}: {b:?} tensor count");
    assert_eq!(
        left.keys().collect::<Vec<_>>(),
        right.keys().collect::<Vec<_>>(),
        "{label}: the two containers must hold the same key set"
    );
    let mut values = 0usize;
    for (key, one) in &left {
        let other = &right[key];
        assert_eq!(one.dims(), other.dims(), "{label}: {key} shapes must match");
        assert_eq!(one.dtype(), other.dtype(), "{label}: {key} dtypes");
        assert_eq!(
            widened(one),
            widened(other),
            "{label}: {key} is not bit-identical across the two containers"
        );
        values += one.elem_count();
    }
    assert_eq!(
        values, VAE_VALUES,
        "{label}: the comparison must cover every learned value"
    );
    values
}

// ── Provenance: the reuse is grounded in tensor bytes ─────────────────────────────────────────────

/// **The reuse gate.** The container the epic-16624 Z-Image fit was measured against and the container
/// candle actually loads are the same 244 learned tensors.
///
/// A hash equality could not have replaced this row: the fit donor is the MLX packer's re-container of
/// the diffusers file, so its SHA-256 and its length differ (167,666,968 vs 167,666,902). The whole
/// 66-byte gap is in the JSON **header**, not in the weights — the two writers disagree on per-tensor
/// key order, on how wide the `data_offsets` integers print, on the `__metadata__` value and on
/// trailing padding, and the tensor payload is the same 167,639,366 bytes on both sides. Every learned
/// tensor underneath is byte-identical, at both tiers Z-Image ships (Turbo and base), so the fit's
/// input domain is the same whichever snapshot a render loads.
#[test]
#[ignore = "needs ZIMAGE_FIT_VAE + ZIMAGE_DIFFUSERS_VAE + ZIMAGE_BASE_DIFFUSERS_VAE; run with --ignored"]
fn the_committed_fit_donor_is_the_shipped_z_image_vae() {
    let fit_path = required_path("ZIMAGE_FIT_VAE");
    let turbo_path = required_path("ZIMAGE_DIFFUSERS_VAE");
    let base_path = required_path("ZIMAGE_BASE_DIFFUSERS_VAE");

    for (label, path, expected, bytes) in [
        (
            "fit donor (mlx bf16)",
            &fit_path,
            FIT_VAE_SHA256,
            167_666_968u64,
        ),
        (
            "Z-Image-Turbo diffusers",
            &turbo_path,
            DIFFUSERS_VAE_SHA256,
            167_666_902,
        ),
        (
            "Z-Image (base) diffusers",
            &base_path,
            DIFFUSERS_VAE_SHA256,
            167_666_902,
        ),
    ] {
        let sha = sha256_of(path);
        eprintln!("  {label:<25}: {sha}  {}", path.display());
        assert_eq!(
            sha, expected,
            "{label} moved — re-derive the fit with mlx-gen-z-image/tests/fit_preview_rgb.rs before \
             reusing it"
        );
        assert_eq!(std::fs::metadata(path).expect("stat").len(), bytes);
    }
    assert_ne!(
        FIT_VAE_SHA256, DIFFUSERS_VAE_SHA256,
        "a different container — which is why this row compares tensors, not hashes"
    );

    let values = assert_same_learned_tensors("fit donor vs Z-Image-Turbo", &fit_path, &turbo_path);
    eprintln!(
        "  fit donor vs Z-Image-Turbo: {VAE_TENSORS} tensors, {values} values, bit-identical"
    );

    // The base tier is the same file as the Turbo one, so its identity to the donor follows — asserted
    // rather than argued, because "the base snapshot ships a different VAE" is precisely the change
    // that would invalidate reusing one fit for both registered ids.
    assert_same_learned_tensors("fit donor vs Z-Image base", &fit_path, &base_path);
    eprintln!("  fit donor vs Z-Image base : {VAE_TENSORS} tensors, bit-identical");

    assert_eq!(candle_gen_z_image::preview::PREVIEW_LATENT_CHANNELS, 16);
}

/// **The finding this story was asked to settle.** Z-Image's 16-channel latent space is not merely
/// *a* 16-channel space — it is FLUX.1-dev's, byte for byte.
///
/// `Tongyi-MAI/Z-Image-Turbo`, `Tongyi-MAI/Z-Image` and `black-forest-labs/FLUX.1-dev` all publish the
/// **same** `vae/diffusion_pytorch_model.safetensors`: one SHA-256, one length, 244 identical tensors,
/// the same `latent_channels: 16` / `scaling_factor: 0.3611` / `shift_factor: 0.1159`. The Z-Image
/// `vae/config.json` even records where it came from — `"_name_or_path": "flux-dev"`.
///
/// The consequence is stated plainly because it affects two stories' constants: epic 16624 committed
/// **two** fits over **one** latent space, `mlx-gen-flux/src/preview.rs`'s and
/// `mlx-gen-z-image/src/preview.rs`'s, measured on different render sets. Neither is wrong; they are
/// duplicates. `candle-gen-z-image` keeps the Z-Image-measured one (MLX parity, and it was measured on
/// Z-Image renders), and collapsing them is a cross-engine decision recorded as a follow-up rather
/// than taken here.
///
/// Left as a row rather than as prose so that a snapshot swap which made them *stop* matching — a
/// genuinely new Z-Image latent space — is noticed before the wrong fit ships.
#[test]
#[ignore = "needs ZIMAGE_DIFFUSERS_VAE + ZIMAGE_FLUX1_VAE; run with --ignored"]
fn the_z_image_vae_is_the_flux1_one() {
    let z_path = required_path("ZIMAGE_DIFFUSERS_VAE");
    let flux_path = required_path("ZIMAGE_FLUX1_VAE");

    let (z_sha, flux_sha) = (sha256_of(&z_path), sha256_of(&flux_path));
    eprintln!("  z-image vae: {z_sha}  {}", z_path.display());
    eprintln!("  flux1   vae: {flux_sha}  {}", flux_path.display());
    assert_eq!(z_sha, DIFFUSERS_VAE_SHA256, "the staged Z-Image VAE moved");
    assert_eq!(
        flux_sha, DIFFUSERS_VAE_SHA256,
        "the staged FLUX.1-dev VAE is not the container sc-16956 pinned"
    );
    assert_eq!(
        z_sha, flux_sha,
        "Z-Image and FLUX.1-dev must publish the SAME VAE container — if they ever stop, Z-Image has \
         its own latent space and the two committed fits are no longer duplicates"
    );

    // The hashes already settle it; the tensor walk is what makes the claim specific — same key set,
    // same shapes, same bits, 244 tensors and 83,819,683 values.
    let values = assert_same_learned_tensors("z-image vs flux1", &z_path, &flux_path);
    eprintln!("  z-image vs flux1: {VAE_TENSORS} tensors, {values} values, bit-identical");

    // And the scale constants that define the space, read off the config candle's `VaeConfig::z_image`
    // mirrors. A shared weight file with a different scaling factor would still be a different space.
    let config = z_path.parent().expect("vae/ directory").join("config.json");
    let text = std::fs::read_to_string(&config).unwrap_or_else(|e| panic!("read {config:?}: {e}"));
    for needle in [
        "\"latent_channels\": 16",
        "\"scaling_factor\": 0.3611",
        "\"shift_factor\": 0.1159",
    ] {
        assert!(
            text.contains(needle),
            "{config:?} must record {needle} — the fit is defined over that scaling"
        );
    }
    assert!(
        text.contains("flux-dev"),
        "the Z-Image vae/config.json records `_name_or_path: flux-dev`; if that ever changes, \
         re-measure rather than assuming the lineage"
    );
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

/// Nearest-neighbour box resample of an RGB8 buffer.
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

fn save_png(dir: &Path, pixels: &[u8], width: u32, height: u32, name: &str) {
    std::fs::create_dir_all(dir).expect("create the artifact dir");
    let path = dir.join(name);
    let buf: image::RgbImage = image::ImageBuffer::from_raw(width, height, pixels.to_vec())
        .expect("frame buffer matches its dimensions");
    buf.save(&path)
        .unwrap_or_else(|e| panic!("save {path:?}: {e}"));
    eprintln!("  wrote {}", path.display());
}

/// Lay the strip out as one horizontal contact sheet so a reviewer sees the progression at a glance.
fn save_strip(dir: &Path, frames: &[PreviewFrame], name: &str) {
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
/// 8-bit scale 0.1 mean |Δ| is unmistakably "not identical" while leaving room for the smallest step
/// of a strip.
const MIN_FRAME_MOVEMENT: f64 = 0.1;

/// The strip must close a meaningful share of its distance to the finished image. Expressed as a
/// fraction of the distance travelled rather than as a ratio of the endpoints, because the endpoints
/// carry the fit's irreducible residual.
const MIN_DISTANCE_FALL: f64 = 0.25;

/// The shared strip analysis: numbering, latent resolution, per-frame movement, falling distance to
/// the finished image, and rising resemblance to it.
///
/// ## What the correlation floor is, and what it is NOT
///
/// A projection cannot correlate with the decode better than the fit does, so the fit fixes a
/// **ceiling**: the Z-Image fit's in-sample R² is `0.98133` (`mlx-gen-z-image/src/preview.rs`), a
/// correlation ceiling of √0.98133 ≈ `0.9906`. The in-sample R² is the like-for-like statistic — the
/// QwenVae families were held against an in-sample 0.9586 and sc-16954 matched that with SDXL's
/// in-sample 0.91849 — so the holdout 0.92827 is deliberately not used here. (sc-16954 was caught
/// comparing an in-sample number against a holdout one, which produced a floor that was too loose;
/// this is the same statistic in both places.)
///
/// What the ceiling does **not** fix is the floor. `min_r_last` also measures *how far the trajectory
/// has travelled one step from the end*, which is a property of the **schedule**: the hook emits
/// BEFORE each solver step (sc-16949), so the final advancement is never previewed. Each lane's floor
/// is therefore passed in, carries the number actually measured on that lane, and is held to a
/// comparable fraction of this fit's ceiling.
///
/// ## Why per-frame movement is measured but its SHAPE is not asserted
///
/// sc-16956 could assert that FLUX.1's frame-to-frame movement accelerates, because FLUX.1-dev's
/// time-shifted flow schedule is back-loaded. Z-Image-Turbo's is **linear** by construction —
/// `set_timesteps(steps, Some(mu))` applies no shift under `use_dynamic_shifting=false`, which
/// `crate::pipeline::render` documents as correctness-critical — so equal σ intervals give a roughly
/// flat movement profile, and the base path's static shift=6.0 gives yet another. Asserting a shape
/// here would be asserting the schedule, not the wiring. The profile is printed for review, a floor
/// keeps any pair from being identical, and the load-bearing statements are the two *monotonicities*
/// (distance falls at every step, resemblance rises at every step) plus the total rise — none of which
/// a stale, duplicated or wrongly-scaled latent could reproduce.
///
/// ## Why "the strip develops" is a per-lane pair, not two constants
///
/// A txt2img strip starts on pre-denoise noise, so it must open **well below** the finished render and
/// rise a long way: `max_r_first = 0.75`, `min_rise = 0.30`, exactly the numbers sc-16956 used and this
/// epic asks for.
///
/// A **strength-reduced img2img** strip cannot satisfy those, and not because of any wiring defect: its
/// first emission is `x_t = (1 − σ_start)·source + σ_start·noise` at `σ_start = sigmas[start]`, so it
/// legitimately opens *partly converged* — that is what a structure-preserving edit means. Measured on
/// the `z_image_edit` lane at strength 0.5, the first frame already correlates +0.764 with the finished
/// render. Applying the txt2img ceiling there would not be a stricter test; it would be a **wrong** one,
/// asserting that an edit starts from noise when the whole point is that it does not. That lane
/// therefore passes its own pair, stated and justified at the call site, while the two monotonicities
/// and the ≥ 25 % distance fall apply unchanged to every lane.
#[allow(clippy::too_many_arguments)]
fn assert_the_strip_converges(
    label: &str,
    frames: &[PreviewFrame],
    final_image: &Image,
    steps: u32,
    latent_w: u32,
    latent_h: u32,
    min_r_last: f64,
    develops: Develops,
) {
    assert_eq!(
        frames
            .iter()
            .map(|f| (f.current, f.total))
            .collect::<Vec<_>>(),
        (1..=steps).map(|n| (n, steps)).collect::<Vec<_>>(),
        "{label}: a {steps}-step render must emit exactly {steps} frames numbered 1..={steps}"
    );

    // Native-latent resolution, and batch 1. A CFG-fused `[2, …]` latent fails the layout contract
    // outright, so a strip that exists at all is already proof the preview never saw a fused
    // unconditional half — there would be no frames if it had.
    for frame in frames {
        assert_eq!(
            (frame.image.width, frame.image.height),
            (latent_w, latent_h),
            "{label}: frames must be VAE-latent resolution"
        );
    }

    // Every metric is computed and printed BEFORE anything is asserted, so one run reports the entire
    // strip rather than stopping at the first failing pair.
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

    // 1. No two consecutive frames are the same picture.
    assert!(
        movement.iter().all(|d| *d > MIN_FRAME_MOVEMENT),
        "{label}: some consecutive frames are effectively identical: {movement:?}"
    );

    // 2. The strip approaches the finished image, at every step and by a meaningful margin.
    let (first, last) = (distances[0], distances[distances.len() - 1]);
    let fall = (first - last) / first;
    eprintln!(
        "  {label}: distance fell {:.1}% ({first:.2} → {last:.2})",
        fall * 100.0
    );
    assert!(
        fall > MIN_DISTANCE_FALL,
        "{label}: the strip must converge on the final image (first {first:.2} → last {last:.2}, \
         fall {fall:.3}, floor {MIN_DISTANCE_FALL})"
    );
    assert!(
        distances.windows(2).all(|p| p[1] < p[0]),
        "{label}: distance to the finished image must fall at every step: {distances:?}"
    );

    // 3. The strip actually comes to resemble the render, monotonically.
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
    // "The strip develops" is asserted as a **rise** plus a ceiling on where it may open, not as an
    // absolute floor on the first frame: correlation is taken over flattened RGB triplets, so it
    // carries channel-mean structure as well as spatial structure, and this fit's intercept is a
    // near-neutral grey — a frame of pre-denoise noise starts at a non-zero, scene-dependent floor.
    // sc-16950's `r_first < 0.35` ceiling is deliberately not ported; the rise plus a loose ceiling is
    // what cannot be faked, since a strip that opened on the finished image would have nowhere to rise
    // to. The pair is per-lane — see this function's docs for why an img2img strip needs its own.
    assert!(
        r_first < develops.max_r_first,
        "{label}: the first frame must not already BE the render \
         (r {r_first:+.3}, ceiling {:+.3})",
        develops.max_r_first
    );
    assert!(
        r_last - r_first > develops.min_rise,
        "{label}: resemblance must actually develop across the strip \
         (first {r_first:+.3} → last {r_last:+.3}, floor {:+.3})",
        develops.min_rise
    );
}

/// How far a lane's strip is allowed to open, and how far it must then travel.
#[derive(Clone, Copy)]
struct Develops {
    /// The first frame must correlate **below** this with the finished render.
    max_r_first: f64,
    /// `r_last − r_first` must exceed this.
    min_rise: f64,
}

/// A strip that starts on pre-denoise noise — every txt2img and control lane. The epic's numbers.
const FROM_NOISE: Develops = Develops {
    max_r_first: 0.75,
    min_rise: 0.30,
};

/// A **strength-reduced img2img** strip, which opens partly converged by construction (§ the function
/// docs). Measured on `z_image_edit` at strength 0.5: opens +0.764, closes +0.892, a rise of +0.128.
/// The ceiling still says "visibly not the render yet" and the floor still says "it moved a real
/// distance"; what it does not do is pretend a structure-preserving edit begins in noise.
const FROM_A_PARTIAL_LATENT: Develops = Develops {
    max_r_first: 0.85,
    min_rise: 0.08,
};

// ── Driving the registered routes ─────────────────────────────────────────────────────────────────

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

/// The staged-residency knob that routes a request away from `pipeline::render` / `render_base` and
/// into `render_sequential` / `render_base_sequential` — the second sampler site each registered
/// descriptor owns, and the one a memory-constrained SceneWorks run actually takes.
fn staged_memory() -> GenerationMemory {
    GenerationMemory {
        stage_residency: true,
        ..GenerationMemory::default()
    }
}

/// Which registered descriptor a row drives.
#[derive(Clone, Copy)]
enum Route {
    Turbo,
    Base,
}

impl Route {
    fn snapshot_var(self) -> &'static str {
        match self {
            Route::Turbo => "ZIMAGE_TURBO_SNAPSHOT",
            Route::Base => "ZIMAGE_BASE_SNAPSHOT",
        }
    }

    fn load(self, spec: &LoadSpec) -> Box<dyn candle_gen::gen_core::Generator> {
        match self {
            Route::Turbo => candle_gen_z_image::load(spec),
            Route::Base => candle_gen_z_image::base::load(spec),
        }
        .unwrap_or_else(|e| panic!("load z-image: {e}"))
    }
}

/// Render one registered route twice on one warmed generator at the same seed — once inert, once live
/// — and hold the strip to [`assert_the_strip_converges`]. Returns the live run's `Progress::Step`
/// count (which IS its evaluation count).
#[allow(clippy::too_many_arguments)]
fn assert_registered_route_previews_converge(
    label: &str,
    route: Route,
    sampler: Option<&str>,
    steps: u32,
    size: u32,
    memory: Option<GenerationMemory>,
    min_r_last: f64,
) -> usize {
    eprintln!("── {label}: {size}² × {steps} steps, sampler {sampler:?}, staged {memory:?}");
    let spec = LoadSpec::new(WeightsSource::Dir(required_path(route.snapshot_var())));
    let generator = route.load(&spec);

    let build = || {
        let mut request = base_request(steps, size, sampler);
        request.memory = memory;
        request
    };

    // N1: the inert baseline. Same generator, same seed, no sink.
    let mut noop = |_: Progress| {};
    let inert = one_image(
        generator
            .generate(&build(), &mut noop)
            .unwrap_or_else(|e| panic!("{label}: inert render: {e}")),
    );

    let (sink, frames) = collecting_sink();
    let mut request = build();
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
        FROM_NOISE,
    );
    save_strip(&dir, &frames, &format!("{label}-strip.png"));
    save_png(
        &dir,
        &live.pixels,
        live.width,
        live.height,
        &format!("{label}-final.png"),
    );
    events
}

/// The four lanes the two registered descriptors own: `z_image_turbo` and `z_image`, each in its
/// resident form (`pipeline::render` / `render_base`) and its staged-residency form
/// (`denoise_sequential` / `denoise_base_sequential`, which a `stage_residency` request takes
/// instead). All four are `run_flow_sampler` sites and all four hand it a projector hook.
///
/// The staged rows are not a formality: they are a *different* sampler call site with its own hook
/// construction, and a SceneWorks run on a memory-constrained GPU takes them rather than the resident
/// ones.
#[test]
#[ignore = "needs ZIMAGE_TURBO_SNAPSHOT + ZIMAGE_BASE_SNAPSHOT + a CUDA GPU; run with --features cuda --ignored"]
fn z_image_preview_frames_evolve_toward_the_final_image() {
    let size = env_u32("ZIMAGE_PREVIEW_SIZE", 512);
    let turbo_steps = env_u32("ZIMAGE_PREVIEW_TURBO_STEPS", 8);
    let base_steps = env_u32("ZIMAGE_PREVIEW_BASE_STEPS", 20);

    // Floors carry the number measured on each lane; see `assert_the_strip_converges` for why they are
    // per-lane and why the fit's in-sample ceiling is √0.98133 ≈ 0.991.
    assert_registered_route_previews_converge(
        "z_image_turbo-resident",
        Route::Turbo,
        None,
        turbo_steps,
        size,
        None,
        MIN_R_LAST_TURBO,
    );
    assert_registered_route_previews_converge(
        "z_image_turbo-staged",
        Route::Turbo,
        None,
        turbo_steps,
        size,
        Some(staged_memory()),
        MIN_R_LAST_TURBO,
    );
    assert_registered_route_previews_converge(
        "z_image-resident",
        Route::Base,
        None,
        base_steps,
        size,
        None,
        MIN_R_LAST_BASE,
    );
    assert_registered_route_previews_converge(
        "z_image-staged",
        Route::Base,
        None,
        base_steps,
        size,
        Some(staged_memory()),
        MIN_R_LAST_BASE,
    );
}

/// Exactly one frame per **outer** solver step on a multi-eval solver.
///
/// The guard is made non-vacuous first, and in the strongest available way: the shared driver calls
/// `on_progress` once per *evaluation* (`sampler.rs` computes the step count on every eval and
/// deliberately repeats it), so counting `Progress::Step` events IS counting evaluations. If `heun`
/// did not evaluate twice per step the event count would equal the step count and "frames == steps"
/// would prove nothing — so that inequality is asserted before the frame count is.
#[test]
#[ignore = "needs ZIMAGE_TURBO_SNAPSHOT + a CUDA GPU; run with --features cuda --ignored"]
fn a_multi_eval_solver_emits_one_frame_per_outer_step() {
    let steps = 8u32;
    let size = env_u32("ZIMAGE_PREVIEW_SIZE", 512);
    let events = assert_registered_route_previews_converge(
        "z_image_turbo-heun",
        Route::Turbo,
        Some("heun"),
        steps,
        size,
        None,
        MIN_R_LAST_TURBO_HEUN,
    );
    eprintln!("  heun: {events} evaluations for {steps} outer steps");
    assert!(
        events > steps as usize,
        "heun must evaluate more than once per outer step or this row proves nothing about the \
         preview counter's dedup ({events} events for {steps} steps)"
    );
    // `assert_registered_route_previews_converge` already required exactly `steps` frames numbered
    // 1..=steps, so the dedup collapsed the extra evaluations. Stated here because that is the point.
}

// ── Driving the name-driven providers ─────────────────────────────────────────────────────────────

/// Both Fun-ControlNet modes, which is where this crate's two wiring layers meet on one provider.
///
/// * **Turbo** (`control::generate_turbo`) owns a bespoke flow-Euler loop and emits by calling the
///   shared `emit_preview_at` directly. This is the row that proves the bespoke half works end to end
///   — the catalog's tally can see that a direct emission call *exists*, not that it produces frames.
/// * **Base** (`control::generate_base`) drives `run_flow_sampler` with a hook, under real CFG.
///
/// Both project the target latent only: the 33-channel control context is injected inside
/// `forward_control` and never becomes part of the tensor the loop integrates. A strip at latent
/// resolution with 16-channel-fit colours is what that looks like from outside — a leaked context
/// would fail the layout gate and produce no frames at all.
#[test]
#[ignore = "needs ZIMAGE_*_SNAPSHOT + ZIMAGE_*_CONTROL_NET + ZIMAGE_PREVIEW_POSE + a CUDA GPU; run with --features cuda --ignored"]
fn the_control_routes_preview_their_target_latent() {
    use candle_gen_z_image::control::{ZImageControl, ZImageControlPaths, ZImageControlRequest};

    let pose = candle_gen::testkit::read_ppm(&required_path("ZIMAGE_PREVIEW_POSE"));
    let dir = artifact_dir();

    for (mode, base, snapshot_var, net_var, steps, guidance, negative, min_r_last) in [
        (
            "z_image_turbo_control",
            false,
            "ZIMAGE_TURBO_SNAPSHOT",
            "ZIMAGE_TURBO_CONTROL_NET",
            8u32,
            None,
            None,
            MIN_R_LAST_TURBO_CONTROL,
        ),
        (
            "z_image_control",
            true,
            "ZIMAGE_BASE_SNAPSHOT",
            "ZIMAGE_BASE_CONTROL_NET",
            20,
            Some(4.0f32),
            Some("blurry, low quality, deformed"),
            MIN_R_LAST_BASE_CONTROL,
        ),
    ] {
        // Each mode has TWO lanes, and they are different functions, not a flag on one: a resident model
        // takes `generate_turbo` / `generate_base`, while a request-scoped `stage_residency` load leaves
        // the transformer absent and routes through `generate_staged` → `denoise_turbo_with` /
        // `denoise_base_with`. Both are shipped, both are user-reachable from the worker, and each carries
        // its own emission — so all four are rendered rather than the two default ones.
        for staged in [false, true] {
            let label = &format!("{mode}{}", if staged { "-staged" } else { "-resident" });
            eprintln!("── {label}: {}² × {steps} steps", pose.width);
            let paths = ZImageControlPaths {
                snapshot: required_path(snapshot_var),
                control: required_path(net_var),
                base,
            };
            let memory = if staged {
                staged_memory()
            } else {
                GenerationMemory::default()
            };
            let model = if staged {
                ZImageControl::load_with_memory(&paths, memory)
            } else {
                ZImageControl::load(&paths)
            }
            .unwrap_or_else(|e| panic!("{label}: load: {e}"));

            let build = || ZImageControlRequest {
                prompt: PROMPT.into(),
                width: pose.width,
                height: pose.height,
                steps: steps as usize,
                seed: SEED,
                guidance,
                negative_prompt: negative.map(str::to_string),
                memory,
                ..ZImageControlRequest::default()
            };

            let mut noop = |_: Progress| {};
            let inert = model
                .generate(&build(), &pose, &mut noop)
                .unwrap_or_else(|e| panic!("{label}: inert render: {e}"));

            let (sink, frames) = collecting_sink();
            let request = ZImageControlRequest {
                preview: sink,
                ..build()
            };
            let live = model
                .generate(&request, &pose, &mut noop)
                .unwrap_or_else(|e| panic!("{label}: live render: {e}"));
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
                pose.width / 8,
                pose.height / 8,
                min_r_last,
                // A control render is still txt2img: the pose skeleton reaches the DiT as an encoded
                // context, never as the initial latent, so the strip starts on pure noise like any other.
                FROM_NOISE,
            );
            save_strip(&dir, &frames, &format!("{label}-strip.png"));
            save_png(
                &dir,
                &live.pixels,
                live.width,
                live.height,
                &format!("{label}-final.png"),
            );
        }
    }
}

/// The img2img / masked-edit provider's bespoke loop, over the **reduced** schedule its strength
/// selects.
///
/// This is the lane whose counter is the easiest to get wrong: `edit::generate` runs
/// `for step_i in start..steps` and reports `total = steps - start`, so the counter is built over the
/// difference and fed the loop-local index. A counter fed the absolute `step_i` would number the first
/// frame `start + 1` — or, once `start >= total`, emit nothing at all. The row therefore asserts the
/// frame count against the REDUCED total, and drives a strength that makes `start > 0` so the
/// distinction is live rather than degenerate.
///
/// It also closes the "edit routes project target tokens only" criterion: the VAE-encoded source is
/// folded into `x_t` before the loop, so there is one trajectory and it is the target's.
#[test]
#[ignore = "needs ZIMAGE_TURBO_SNAPSHOT + ZIMAGE_PREVIEW_EDIT_SOURCE + a CUDA GPU; run with --features cuda --ignored"]
fn the_edit_route_previews_its_reduced_schedule() {
    use candle_gen_z_image::edit::{ZImageEdit, ZImageEditPaths, ZImageEditRequest};

    let source = candle_gen::testkit::read_ppm(&required_path("ZIMAGE_PREVIEW_EDIT_SOURCE"));
    let (steps, strength) = (12usize, 0.5f32);
    // `init_time_step`: max(1, floor(steps·strength)) — 6 here, so the reduced schedule is 6 steps and
    // an absolute-index counter would be visibly wrong.
    let start = (steps as f32 * strength).floor().max(1.0) as usize;
    let reduced = (steps - start) as u32;
    assert!(
        start > 0 && reduced > 1,
        "the strength must actually shorten the schedule for this row to mean anything \
         (start {start}, reduced {reduced})"
    );
    eprintln!(
        "── z_image_edit: {}x{} × {steps} steps @ strength {strength} → {reduced} previewed steps",
        source.width, source.height
    );

    let paths = ZImageEditPaths {
        base: required_path("ZIMAGE_TURBO_SNAPSHOT"),
    };
    let model = ZImageEdit::load(&paths).unwrap_or_else(|e| panic!("load z-image edit: {e}"));

    let build = || ZImageEditRequest {
        prompt: PROMPT.into(),
        width: source.width - (source.width % 16),
        height: source.height - (source.height % 16),
        steps,
        strength,
        seed: SEED,
        ..ZImageEditRequest::default()
    };

    let mut noop = |_: Progress| {};
    let inert = model
        .generate(&build(), &source, &mut noop)
        .unwrap_or_else(|e| panic!("edit inert render: {e}"));

    let (sink, frames) = collecting_sink();
    let request = ZImageEditRequest {
        preview: sink,
        ..build()
    };
    let live = model
        .generate(&request, &source, &mut noop)
        .unwrap_or_else(|e| panic!("edit live render: {e}"));
    assert_eq!(
        inert.pixels, live.pixels,
        "attaching a live preview sink changed the seeded edit"
    );

    let frames = candle_gen::lock_recover(&frames).clone();
    let request_size = build();
    assert_the_strip_converges(
        "z_image_edit",
        &frames,
        &live,
        reduced,
        request_size.width / 8,
        request_size.height / 8,
        MIN_R_LAST_EDIT,
        // The one lane that does NOT start on noise: its first emission is the strength blend of the
        // VAE-encoded source with seeded noise, so it opens partly converged on purpose.
        FROM_A_PARTIAL_LATENT,
    );
    let dir = artifact_dir();
    save_strip(&dir, &frames, "z_image_edit-strip.png");
    save_png(
        &dir,
        &live.pixels,
        live.width,
        live.height,
        "z_image_edit-final.png",
    );
}

// ── The measured per-lane correlation floors ──────────────────────────────────────────────────────
//
// Each carries the number measured on that lane at 512², rounded down by roughly two points of slack:
// the six floors sit 0.020–0.026 under their measurements, none of them given a wider margin than the
// rest. They are
// deliberately NOT one constant: `min_r_last` is a joint statement about the fit's ceiling
// (√0.98133 ≈ 0.9906) and about how much of the trajectory the lane's SCHEDULE leaves to the one step
// the hook never previews (it emits before each solver step, sc-16949). Turbo's linear 8-step schedule
// leaves ~1/8 of the run unpreviewed and reaches +0.921; the base path's static shift=6.0 is heavily
// back-loaded and reaches only +0.836 over 20 steps for exactly that reason — the same effect sc-16955
// measured on FLUX.2. A single floor would either be vacuous for Turbo or unreachable for base.
// See `assert_the_strip_converges` for the full statement.

/// Measured +0.921 on the 8-step 512² Turbo lane; resident and staged are the same trajectory and both
/// reported it. 93 % of the fit's ceiling.
const MIN_R_LAST_TURBO: f64 = 0.90;
/// Measured +0.913 on the 8-step 512² Turbo `heun` lane (15 evaluations, 8 frames).
const MIN_R_LAST_TURBO_HEUN: f64 = 0.89;
/// Measured +0.836 on the 20-step 512² base CFG lane, resident and staged alike. The lowest of the six
/// and the most schedule-bound: shift=6.0 back-loads the trajectory, so the unpreviewed terminal step
/// carries a large share — the strip's own distance-to-final falls **5.38** in the last previewed step
/// alone (50.81 → 22.26 over the whole 20-frame strip), and the 20-step base-control lane back-loads
/// the same way at 7.34 (60.53 → 20.49).
const MIN_R_LAST_BASE: f64 = 0.81;
/// Measured +0.942 on the 8-step 512² Turbo control lane — the BESPOKE-loop lane, and the highest of
/// the six, because a pose-locked composition resolves earlier than a free one.
const MIN_R_LAST_TURBO_CONTROL: f64 = 0.92;
/// Measured +0.920 on the 20-step 512² base control lane (real CFG, guidance 4.0).
const MIN_R_LAST_BASE_CONTROL: f64 = 0.90;
/// Measured +0.892 on the 6-previewed-step img2img lane at strength 0.5. The strip starts partly
/// converged (the VAE-encoded source is already in `x_t`), so it both opens and closes higher than a
/// txt2img strip — see [`FROM_A_PARTIAL_LATENT`].
const MIN_R_LAST_EDIT: f64 = 0.87;

// ── The σ convention ──────────────────────────────────────────────────────────────────────────────

/// This family's σ-convention finding, measured rather than argued — and the counterpart of
/// sc-16954's VE-correction row, which found the **opposite** for the discrete ε cohort.
///
/// `run_flow_sampler` integrates a `FlowModelSampling` whose `input_scale` is exactly `1.0` at every σ,
/// so the running latent already *is* the tensor the fit was measured against and no `with_sigma`
/// correction is needed. The three bespoke Turbo loops scale nothing either — they hand `latents`
/// straight to the DiT — so the same conclusion covers every Z-Image lane.
///
/// The cheap decisive signal sc-16954 named is the first frame's rail-clipped fraction: SDXL's
/// uncorrected projection clipped 89.4% of pixels to 0/255, which is what a missing input scaling
/// looks like. Here the same measurement is taken on the latent this family's first emission actually
/// sees — `common::seed_noise` is unit-normal and Z-Image's `σ_max = 1.0` — and it must come out
/// readable.
///
/// Runs on the committed constants alone, no weights, and is deliberately **not** `#[ignore]`d: it is
/// the row that must appear in a plain `cargo test` of this file. sc-16954 shipped a red row that hid
/// because the only non-ignored row in its file was excluded by `-- --ignored`.
#[test]
fn the_flow_cohort_needs_no_sigma_correction() {
    use candle_gen::gen_core::sampling::{FlowModelSampling, ModelSampling, TimestepConvention};

    // The convention, first: the claim is about `input_scale`, so it is read off the very
    // `ModelSampling` the driver integrates rather than asserted about the family in prose. Both
    // conventions are checked because Z-Image drives `OneMinusSigma`, not the default `Sigma`.
    for conv in [TimestepConvention::Sigma, TimestepConvention::OneMinusSigma] {
        let ms = FlowModelSampling::new(conv);
        for sigma in [0.0f32, 0.05, 0.25, 0.5, 0.75, 1.0] {
            assert_eq!(
                ms.input_scale(sigma),
                1.0,
                "FlowModelSampling::input_scale must be identically 1.0; at {sigma} under {conv:?} \
                 it is not, and this family would need PreviewHook::with_sigma"
            );
        }
    }

    // The consequence, measured. A unit-normal 5-D latent at σ_max = 1.0 is what the first emission
    // sees.
    let (lat_h, lat_w) = (32usize, 32usize);
    let mut rng = <rand::rngs::StdRng as rand::SeedableRng>::seed_from_u64(SEED);
    let noise = candle_gen::seeded_normal_vec(&mut rng, 16 * lat_h * lat_w);
    let latents = Tensor::from_vec(noise, (1, 16, 1, lat_h, lat_w), &Device::Cpu).expect("latent");

    let frame = candle_gen_z_image::preview::project_frame_latents(&latents)
        .expect("project the first-emission latent");
    let rails = frame.pixels.iter().filter(|&&v| v == 0 || v == 255).count() as f64
        / frame.pixels.len() as f64;
    eprintln!("  flow prior at sigma_max: rail-clipped fraction {rails:.4}");
    // The bound is loose enough that a rounding change cannot flip it and far below sc-16954's
    // uncorrected SDXL 0.894, which is the number it is being contrasted with.
    assert!(
        rails < 0.05,
        "an uncorrected flow-space projection must already be a readable noise field, not a clipped \
         one ({rails:.4}) — if this ever fails, the family needs PreviewHook::with_sigma"
    );
}
