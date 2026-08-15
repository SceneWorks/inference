//! sc-16952 — candle Qwen-Image per-step latent **preview** real-weight validation (epic 16948).
//!
//! Two independent things a shape-only smoke cannot establish, and which this epic requires of every
//! wiring story:
//!
//! 1. **The reused fit belongs to this latent space.** [`t2i_vae_is_the_pinned_fit_donor`] and
//!    [`edit_vae_is_the_pinned_fit_donor`] hash the `vae/` shard each lane actually loads and require
//!    it to be the *identical file* the epic-16624 QwenVae fit was measured against — not merely a
//!    container of the same values, which is as far as sc-16950 could get for Krea. Two rows rather
//!    than one because two different snapshots are involved and whichever half did not run has to be
//!    visible in the log.
//! 2. **The frames actually develop.** [`t2i_preview_frames_evolve_toward_the_final_image`] and
//!    [`edit_preview_frames_evolve_toward_the_final_image`] render through the real stacks with a live
//!    sink, check the numbering contract, check seeded byte-identity against an inert render, and
//!    measure that each frame is closer to the finished image than the one before it. Both strips are
//!    written out for direct review.
//!
//! ```sh
//! QWEN_PREVIEW_T2I_DIR=E:\models\qwen-image-mlx\q4 \
//! QWEN_PREVIEW_EDIT_DIR=E:\models\qwen-image-edit-2511-mlx\q4 \
//! QWEN_PREVIEW_EDIT_REFERENCE=E:\out\sc-16952\t2i_final.png \
//! QWEN_PREVIEW_ARTIFACT_DIR=E:\out\sc-16952 \
//!   cargo test -p candle-gen-qwen-image --release --features cuda --test preview_real_weights \
//!     -- --ignored --nocapture
//! ```
//!
//! Every input is **required** by the row that uses it: a row that early-returns on an unset variable
//! still reports SUCCESS, and in a run log a skipped gate is indistinguishable from one that ran and
//! proved something. Asking for `--ignored` is already the opt-in.

use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex};

use candle_gen::gen_core::{
    GenerationOutput, GenerationRequest, Image, LoadSpec, OffloadPolicy, PreviewFrame, PreviewSink,
    WeightsSource,
};
use candle_gen_qwen_image::{
    QwenEdit, QwenEditPaths, QwenEditRequest, QwenFunControl, QwenFunControlPaths,
    QwenFunControlRequest,
};

const PROMPT: &str =
    "A medium-shot photograph of a red fox sitting in a snowy forest at golden hour.";
const EDIT_PROMPT: &str = "make it a bright summer meadow with wildflowers, warm midday light";

/// The SHA-256 of `vae/diffusion_pytorch_model.safetensors` in **every** snapshot a candle
/// Qwen-Image route loads a VAE from, and in the `SceneWorks/qwen-image-mlx`
/// @ `8080a4171f1c8b7fca6c30491eafbe6ffab754bf` snapshot the epic-16624 fit was measured against.
///
/// One file, 253,806,966 bytes, republished unmodified across `Qwen/Qwen-Image-2512`
/// (@ `25468b98e3276ca6700de15c6628e51b7de54a26`), `Qwen/Qwen-Image-Edit-2511`
/// (@ `6f3ccc0b56e431dc6a0c2b2039706d7d26f22cb9`), and the packed q4 / q8 tiers of
/// `SceneWorks/qwen-image-mlx` and `SceneWorks/qwen-image-edit-2511-mlx`
/// (@ `0dfbf3a018bcee42d77de14494c35f97a7531def`) — every tier keeps the VAE dense.
///
/// Byte identity is a strictly stronger reuse ground than sc-16950's tensor-by-tensor comparison, and
/// it is why this crate needs no such comparison: there is no container difference to argue past.
const QWEN_VAE_SHA256: &str = "0c8bc8b758c649abef9ea407b95408389a3b2f610d0d10fcb054fe171d0a8344";

/// The SHA-256 of the `vae/config.json` published beside it — identical across the same snapshots,
/// and therefore carrying the same `latents_mean` / `latents_std`.
const QWEN_VAE_CONFIG_SHA256: &str =
    "c448160dba5ce79c965cb075ee02e18d1c42eb6424f787e5869790d577b56a65";

/// The per-channel de-normalization that *defines* the 16-channel latent space the fit lives in.
/// Asserted from the parsed config as well as from its hash: the hash proves the file is unchanged,
/// these prove the file says what the fit assumed.
const LATENTS_MEAN: [f64; 16] = [
    -0.7571, -0.7089, -0.9113, 0.1075, -0.1745, 0.9653, -0.1517, 1.5508, 0.4134, -0.0715, 0.5517,
    -0.3632, -0.1922, -0.9497, 0.2503, -0.2921,
];
const LATENTS_STD: [f64; 16] = [
    2.8184, 1.4541, 2.3275, 2.6558, 1.2196, 1.7708, 2.6052, 2.0743, 3.2687, 2.1526, 2.8652, 1.5579,
    1.6382, 1.1253, 2.8251, 1.916,
];

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
    env_path("QWEN_PREVIEW_ARTIFACT_DIR")
        .unwrap_or_else(|| std::env::temp_dir().join("qwen_preview_sc16952"))
}

// ── Provenance: the reuse is grounded in tensor bytes ─────────────────────────────────────────────

fn sha256_of(path: &Path) -> String {
    use sha2::{Digest, Sha256};
    let mut file = std::fs::File::open(path).unwrap_or_else(|e| panic!("open {path:?}: {e}"));
    let mut hasher = Sha256::new();
    std::io::copy(&mut file, &mut hasher).expect("hash the file");
    format!("{:x}", hasher.finalize())
}

/// The single `.safetensors` under a snapshot's `vae/` dir — the file
/// `crate::vae::QwenVae::new` is handed, on every lane.
fn vae_file(root: &Path) -> PathBuf {
    let dir = root.join("vae");
    let mut files: Vec<PathBuf> = std::fs::read_dir(&dir)
        .unwrap_or_else(|e| panic!("read {dir:?}: {e}"))
        .filter_map(|entry| entry.ok().map(|e| e.path()))
        .filter(|p| p.extension().is_some_and(|e| e == "safetensors"))
        .collect();
    files.sort();
    assert_eq!(files.len(), 1, "expected exactly one VAE shard in {dir:?}");
    files.pop().expect("one shard")
}

fn read_latent_stats(root: &Path) -> (Vec<f64>, Vec<f64>) {
    let path = root.join("vae").join("config.json");
    let text = std::fs::read_to_string(&path).unwrap_or_else(|e| panic!("read {path:?}: {e}"));
    let json: serde_json::Value = serde_json::from_str(&text).expect("parse vae/config.json");
    let read = |key: &str| -> Vec<f64> {
        json[key]
            .as_array()
            .unwrap_or_else(|| panic!("{key} missing from {path:?}"))
            .iter()
            .map(|v| v.as_f64().expect("numeric latent stat"))
            .collect()
    };
    (read("latents_mean"), read("latents_std"))
}

/// Assert that `root` publishes the pinned VAE — the same bytes, and the same `vae/config.json`
/// defining the same normalized latent space the fit was measured in.
fn assert_is_the_fit_donor_vae(label: &str, root: &Path) {
    let shard = vae_file(root);
    let shard_sha = sha256_of(&shard);
    let config_sha = sha256_of(&root.join("vae").join("config.json"));
    eprintln!("{label} vae/         {shard_sha}  {}", shard.display());
    eprintln!("{label} vae/config   {config_sha}");
    assert_eq!(
        shard_sha, QWEN_VAE_SHA256,
        "{label}: the VAE this lane loads is not the file the reused fit was measured against"
    );
    assert_eq!(
        config_sha, QWEN_VAE_CONFIG_SHA256,
        "{label}: the VAE config differs, so the normalized latent space may not be the fitted one"
    );
    let (mean, std) = read_latent_stats(root);
    assert_eq!(mean, LATENTS_MEAN, "{label}: latents_mean defines the fit");
    assert_eq!(std, LATENTS_STD, "{label}: latents_std defines the fit");
}

/// The base txt2img (and, through the same snapshot, the 2512-Fun ControlNet lane, which loads its
/// `QwenVae` and `QwenVaeEncoder` from `QwenFunControlPaths::qwen_base`) VAE is the fit donor.
#[test]
#[ignore = "needs a real Qwen-Image snapshot (set QWEN_PREVIEW_T2I_DIR)"]
fn t2i_vae_is_the_pinned_fit_donor() {
    assert_is_the_fit_donor_vae("qwen-image  ", &required_path("QWEN_PREVIEW_T2I_DIR"));
}

/// The edit lane loads its VAE from its own snapshot root, so it is pinned separately rather than
/// assumed to follow the base.
#[test]
#[ignore = "needs a real Qwen-Image-Edit snapshot (set QWEN_PREVIEW_EDIT_DIR)"]
fn edit_vae_is_the_pinned_fit_donor() {
    assert_is_the_fit_donor_vae("qwen-edit   ", &required_path("QWEN_PREVIEW_EDIT_DIR"));
}

// ── Runtime: the frames actually develop ──────────────────────────────────────────────────────────

/// Mean absolute per-channel distance between two equal-length RGB8 buffers.
fn mean_abs_delta(a: &[u8], b: &[u8]) -> f64 {
    assert_eq!(a.len(), b.len());
    a.iter()
        .zip(b)
        .map(|(x, y)| (*x as i32 - *y as i32).unsigned_abs() as f64)
        .sum::<f64>()
        / a.len() as f64
}

/// Pearson correlation between two equal-length RGB8 buffers, over all channels.
fn correlation(a: &[u8], b: &[u8]) -> f64 {
    assert_eq!(a.len(), b.len());
    let n = a.len() as f64;
    let (mean_a, mean_b) = (
        a.iter().map(|&v| v as f64).sum::<f64>() / n,
        b.iter().map(|&v| v as f64).sum::<f64>() / n,
    );
    let (mut cov, mut var_a, mut var_b) = (0.0, 0.0, 0.0);
    for (&x, &y) in a.iter().zip(b) {
        let (dx, dy) = (x as f64 - mean_a, y as f64 - mean_b);
        cov += dx * dy;
        var_a += dx * dx;
        var_b += dy * dy;
    }
    let denominator = (var_a * var_b).sqrt();
    if denominator == 0.0 {
        return 0.0;
    }
    cov / denominator
}

/// Box-downsample a raw RGB8 buffer to `(w, h)`.
fn downsample_raw(pixels: &[u8], src_w: u32, src_h: u32, w: u32, h: u32) -> Vec<u8> {
    let (sw, sh) = (src_w as usize, src_h as usize);
    let (tw, th) = (w as usize, h as usize);
    let mut out = vec![0u8; tw * th * 3];
    for ty in 0..th {
        for tx in 0..tw {
            let (x0, x1) = (tx * sw / tw, ((tx + 1) * sw / tw).max(tx * sw / tw + 1));
            let (y0, y1) = (ty * sh / th, ((ty + 1) * sh / th).max(ty * sh / th + 1));
            for c in 0..3 {
                let mut sum = 0u32;
                let mut n = 0u32;
                for y in y0..y1.min(sh) {
                    for x in x0..x1.min(sw) {
                        sum += pixels[(y * sw + x) * 3 + c] as u32;
                        n += 1;
                    }
                }
                out[(ty * tw + tx) * 3 + c] = (sum / n.max(1)) as u8;
            }
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
    let path = dir.join(format!("{name}.png"));
    image::save_buffer(&path, pixels, width, height, image::ExtendedColorType::Rgb8)
        .expect("save a PNG");
    eprintln!("  saved {}", path.display());
}

/// Write the frames side by side as one strip, plus each frame individually — the artifact the epic
/// asks to be reviewed directly.
fn save_strip(frames: &[PreviewFrame], name: &str) {
    let (w, h) = (
        frames[0].image.width as usize,
        frames[0].image.height as usize,
    );
    let strip_w = w * frames.len();
    let mut strip = vec![0u8; strip_w * h * 3];
    for (i, frame) in frames.iter().enumerate() {
        for y in 0..h {
            let src = &frame.image.pixels[y * w * 3..(y + 1) * w * 3];
            let x0 = (y * strip_w + i * w) * 3;
            strip[x0..x0 + w * 3].copy_from_slice(src);
        }
        save_png(
            &frame.image.pixels,
            frame.image.width,
            frame.image.height,
            &format!("{name}_frame{:02}", frame.current),
        );
    }
    save_png(&strip, strip_w as u32, h as u32, &format!("{name}_strip"));
}

fn one_image(out: GenerationOutput) -> Image {
    let GenerationOutput::Images(mut images) = out else {
        panic!("expected GenerationOutput::Images");
    };
    assert_eq!(images.len(), 1);
    images.pop().expect("one image")
}

fn collecting_sink() -> (PreviewSink, Arc<Mutex<Vec<PreviewFrame>>>) {
    let frames = Arc::new(Mutex::new(Vec::new()));
    let captured = Arc::clone(&frames);
    let sink = PreviewSink::new(move |frame| candle_gen::lock_recover(&captured).push(frame));
    (sink, frames)
}

/// The shared strip analysis: numbering, latent resolution, per-frame movement, falling distance to
/// the finished image, and rising correlation with it. Applied identically to t2i and edit so neither
/// lane can be closed with a weaker measurement than the other.
fn assert_the_strip_converges(
    label: &str,
    frames: &[PreviewFrame],
    final_image: &Image,
    steps: u32,
    width: u32,
    height: u32,
) {
    assert_eq!(
        frames
            .iter()
            .map(|f| (f.current, f.total))
            .collect::<Vec<_>>(),
        (1..=steps).map(|n| (n, steps)).collect::<Vec<_>>(),
        "{label}: a {steps}-step render must emit exactly {steps} frames numbered 1..={steps}"
    );

    // The frames are VAE-latent resolution `H/8 × W/8`, which is also the proof that projection ran
    // AFTER `unpack_latents`: the packed token grid is `H/16 × W/16`, and a projection of the packed
    // sequence could not have produced this size (it could not have produced anything — the packed
    // latent is rank 3 and fails the `[1, C, h, w]` contract outright, so the strip would be empty).
    for frame in frames {
        assert_eq!(
            (frame.image.width, frame.image.height),
            (width / 8, height / 8),
            "{label}: frames must be VAE-latent resolution, not the packed token grid"
        );
    }

    // Every frame must differ from its predecessor — N copies of one image would satisfy a naive
    // "N frames arrived" check while showing nothing developing.
    for pair in frames.windows(2) {
        let delta = mean_abs_delta(&pair[0].image.pixels, &pair[1].image.pixels);
        eprintln!(
            "  {label} frame {:>2} → {:>2}: mean |Δ| {delta:.2}",
            pair[0].current, pair[1].current
        );
        assert!(
            delta > 0.5,
            "{label}: frames {} and {} are effectively identical (mean |Δ| {delta:.3})",
            pair[0].current,
            pair[1].current
        );
    }

    let (lw, lh) = (width / 8, height / 8);
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
    let (first, last) = (distances[0], distances[distances.len() - 1]);
    assert!(
        last < first * 0.6,
        "{label}: the strip must converge on the final image (first {first:.2} → last {last:.2})"
    );
    assert!(
        distances.windows(2).all(|p| p[1] < p[0]),
        "{label}: distance to the finished image must fall at every step: {distances:?}"
    );

    // Absolute distance can only ever say "closer", never "resembles": the projection is a global
    // linear approximation of the decode (R² 0.9586), so even a perfectly converged latent keeps an
    // offset and gain error against the true pixels. A hook also emits *before* each step, so the last
    // frame is one solver advancement short of the render. Correlation over a coarse thumbnail — which
    // averages the residual noise away and leaves subject placement and colour masses — is what "the
    // preview looks like the image" actually means for a decorative frame.
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
    let (r_first, r_last) = (correlations[0], correlations[correlations.len() - 1]);
    assert!(
        r_last > 0.85,
        "{label}: the last preview frame must resemble the finished render (r {r_last:+.3})"
    );
    assert!(
        correlations.windows(2).all(|p| p[1] > p[0]),
        "{label}: resemblance must increase at every step: {correlations:?}"
    );
    // "The strip develops" is asserted as a **rise**, not as an absolute floor on the first frame.
    //
    // Correlation is taken over the flattened RGB triplets, so it carries the channel-mean structure
    // as well as the spatial structure — and the fit's intercept (0.406, 0.386, 0.287) is itself
    // R > G > B, which every warm-lit render also is. A frame of pure pre-denoise noise therefore
    // starts at a non-zero, *scene-dependent* correlation floor rather than at zero: the measured
    // 1024² t2i strip opened at +0.292 and the 768² edit strip at +0.553, both on genuine noise. A
    // fixed low ceiling on `r_first` would be reading that floor as if it were resemblance and would
    // fail an honest lane for the colour of its prompt.
    //
    // The rise is what cannot be faked: a strip that opened on the finished image — the failure this
    // guards, and the one a naive "N frames arrived" check misses — has nowhere to rise to. It is
    // layered with the strictly monotone rise above, the falling mean |Δ|, and the per-frame
    // movement floor.
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

/// The base txt2img runtime gate: a real render emits one numbered frame per step, is
/// seeded-byte-identical to the same render with an inert sink, and the frames march monotonically
/// toward the finished image instead of being N views of noise.
#[test]
#[ignore = "needs a real Qwen-Image snapshot on a CUDA box (set QWEN_PREVIEW_T2I_DIR)"]
fn t2i_preview_frames_evolve_toward_the_final_image() {
    let root = required_path("QWEN_PREVIEW_T2I_DIR");
    let steps = env_u32("QWEN_PREVIEW_STEPS", 12);
    let size = env_u32("QWEN_PREVIEW_SIZE", 1024);

    let spec = LoadSpec::new(WeightsSource::Dir(root));
    let generator = candle_gen_qwen_image::provider_registry()
        .unwrap()
        .load("qwen_image", &spec)
        .expect("load qwen_image");

    let base = GenerationRequest {
        prompt: PROMPT.into(),
        width: size,
        height: size,
        count: 1,
        seed: Some(0),
        steps: Some(steps),
        ..Default::default()
    };

    // Inert first: the byte-identity baseline, on the same warmed generator.
    let inert = one_image(
        generator
            .generate(&base, &mut |_| {})
            .expect("inert-sink render"),
    );

    let (sink, frames) = collecting_sink();
    let active_request = GenerationRequest {
        preview: sink,
        ..base
    };
    let active = one_image(
        generator
            .generate(&active_request, &mut |_| {})
            .expect("active-sink render"),
    );
    assert_eq!(
        inert.pixels, active.pixels,
        "an active preview sink must not change a single output byte at the same seed"
    );

    let frames = candle_gen::lock_recover(&frames).clone();
    save_strip(&frames, &format!("t2i_{size}_s{steps}"));
    save_png(
        &active.pixels,
        active.width,
        active.height,
        &format!("t2i_{size}_s{steps}_final"),
    );
    assert_the_strip_converges("t2i ", &frames, &active, steps, size, size);
}

/// The **edit** runtime gate. Same measurements as the base lane, plus the thing that is specific to
/// this route: the frames are of the target image, never of a reference.
///
/// That property is load-bearing and is asserted three ways here, on top of the structural argument
/// in `candle_gen_qwen_image::preview`'s module doc. First, the reference is deliberately loaded at a
/// **different** resolution from the request, so a reference-derived frame could not carry the
/// asserted `width/8 × height/8` size. Second, a projection over the joint noise+reference sequence
/// could not have produced a frame at all — `unpack_latents` would reject the element count — so an
/// N-frame strip is itself evidence the hook was handed the narrowed target. Third, the strip is
/// required to converge on the **edited output**, which the shared analysis measures by correlation.
#[test]
#[ignore = "needs a real Qwen-Image-Edit snapshot on a CUDA box (set QWEN_PREVIEW_EDIT_DIR, \
            QWEN_PREVIEW_EDIT_REFERENCE)"]
fn edit_preview_frames_evolve_toward_the_final_image() {
    let root = required_path("QWEN_PREVIEW_EDIT_DIR");
    let reference_path = required_path("QWEN_PREVIEW_EDIT_REFERENCE");
    let steps = env_u32("QWEN_PREVIEW_EDIT_STEPS", 12) as usize;
    let size = env_u32("QWEN_PREVIEW_EDIT_SIZE", 768);

    let decoded = image::open(&reference_path)
        .unwrap_or_else(|e| panic!("open {reference_path:?}: {e}"))
        .to_rgb8();
    let reference = Image {
        width: decoded.width(),
        height: decoded.height(),
        pixels: decoded.into_raw(),
    };
    eprintln!(
        "reference {}x{} → editing at {size}x{size}",
        reference.width, reference.height
    );
    // A checked precondition, not an operator's choice: the reference must NOT be the request's own
    // size, so that "the frames are the target's latent size" discriminates between the target and
    // the reference rather than being satisfied by the two happening to coincide.
    assert!(
        (reference.width, reference.height) != (size, size),
        "this row needs a reference whose resolution differs from the edit request's ({size}²), \
         or the target-only frame-size assertion below proves nothing"
    );

    let model = QwenEdit::load(&QwenEditPaths {
        root,
        text_encoder: None,
        adapters: vec![],
        offload_policy: OffloadPolicy::Resident,
    })
    .expect("load QwenEdit");

    let base = QwenEditRequest {
        prompt: EDIT_PROMPT.into(),
        negative: "blurry, lowres, artifacts, watermark".into(),
        width: size,
        height: size,
        steps,
        guidance: 4.0,
        seed: 12345,
        ..Default::default()
    };

    let inert = model
        .generate(&base, std::slice::from_ref(&reference), &mut |_| {})
        .expect("inert-sink edit");

    let (sink, frames) = collecting_sink();
    let active_request = QwenEditRequest {
        preview: sink,
        ..base.clone()
    };
    let active = model
        .generate(
            &active_request,
            std::slice::from_ref(&reference),
            &mut |_| {},
        )
        .expect("active-sink edit");
    assert_eq!(
        inert.pixels, active.pixels,
        "an active preview sink must not change a single output byte at the same seed"
    );

    let frames = candle_gen::lock_recover(&frames).clone();
    save_strip(&frames, &format!("edit_{size}_s{steps}"));
    save_png(
        &active.pixels,
        active.width,
        active.height,
        &format!("edit_{size}_s{steps}_final"),
    );
    assert_the_strip_converges("edit", &frames, &active, steps as u32, size, size);

    // The target-only check stated as a measurement rather than only as a shape: the last frame must
    // track the EDITED output more closely than it tracks the reference it was conditioned on. A
    // preview that had projected reference tokens would invert this.
    let coarse = 16u32;
    let to_final = correlation(
        &downsample_raw(
            &frames[frames.len() - 1].image.pixels,
            frames[frames.len() - 1].image.width,
            frames[frames.len() - 1].image.height,
            coarse,
            coarse,
        ),
        &downsample(&active, coarse, coarse),
    );
    let to_reference = correlation(
        &downsample_raw(
            &frames[frames.len() - 1].image.pixels,
            frames[frames.len() - 1].image.width,
            frames[frames.len() - 1].image.height,
            coarse,
            coarse,
        ),
        &downsample(&reference, coarse, coarse),
    );
    eprintln!("  edit last frame: r(edited) {to_final:+.3}  r(reference) {to_reference:+.3}");
    assert!(
        to_final > to_reference,
        "the edit preview must track the image being generated, not the reference \
         (r(edited) {to_final:+.3} vs r(reference) {to_reference:+.3})"
    );
}

/// The **ControlNet/Fun** runtime gate. The third shipped lane, held to the same measurement as the
/// other two rather than closed on its route-inventory row alone — leaving the least-exercised route
/// to a source scan is exactly the scope narrowing this epic's reviews hunt for.
///
/// The hint is an ordinary RGB image, which is all this lane wants: the 2512-Fun branch is
/// deliberately input-agnostic (no mode index — pose, canny and depth share one path), and it
/// VAE-encodes whatever it is handed into the packed 132-channel control context. What is being
/// measured here is preview convergence, not control fidelity, so a hint that is not a real pose map
/// is a legitimate input and the render it steers is not judged.
#[test]
#[ignore = "needs a real Qwen-Image base + 2512-Fun control checkpoint on a CUDA box (set             QWEN_PREVIEW_CONTROL_BASE_DIR, QWEN_PREVIEW_CONTROL_NET, QWEN_PREVIEW_CONTROL_HINT)"]
fn control_fun_preview_frames_evolve_toward_the_final_image() {
    let qwen_base = required_path("QWEN_PREVIEW_CONTROL_BASE_DIR");
    let controlnet = required_path("QWEN_PREVIEW_CONTROL_NET");
    let hint_path = required_path("QWEN_PREVIEW_CONTROL_HINT");
    let steps = env_u32("QWEN_PREVIEW_CONTROL_STEPS", 12) as usize;
    let size = env_u32("QWEN_PREVIEW_CONTROL_SIZE", 768);

    // This lane loads its VAE from the BASE snapshot, so its provenance is the base's — pinned here
    // rather than inferred from the t2i row, because `QwenFunControlPaths` can name a different base.
    assert_is_the_fit_donor_vae("qwen-control", &qwen_base);

    let decoded = image::open(&hint_path)
        .unwrap_or_else(|e| panic!("open {hint_path:?}: {e}"))
        .to_rgb8();
    let (source_w, source_h) = (decoded.width(), decoded.height());
    // This lane requires the hint to arrive at the request's exact size (`preprocess_control_image`
    // rejects a mismatch rather than resampling), so the harness resamples instead of constraining
    // which image can be supplied.
    let hint = Image {
        width: size,
        height: size,
        pixels: downsample_raw(&decoded.into_raw(), source_w, source_h, size, size),
    };
    eprintln!("control hint {source_w}x{source_h} → resampled to {size}x{size}");

    let model = QwenFunControl::load(&QwenFunControlPaths {
        qwen_base,
        text_encoder: None,
        controlnet,
        adapters: Vec::new(),
    })
    .expect("load QwenFunControl");

    let base = QwenFunControlRequest {
        prompt: PROMPT.into(),
        negative: "blurry, lowres, artifacts, watermark".into(),
        width: size,
        height: size,
        steps,
        guidance: 4.0,
        seed: 12345,
        ..Default::default()
    };

    let inert = model
        .generate(&base, &hint, &mut |_| {})
        .expect("inert-sink control render");

    let (sink, frames) = collecting_sink();
    let active_request = QwenFunControlRequest {
        preview: sink,
        ..base.clone()
    };
    let active = model
        .generate(&active_request, &hint, &mut |_| {})
        .expect("active-sink control render");
    assert_eq!(
        inert.pixels, active.pixels,
        "an active preview sink must not change a single output byte at the same seed"
    );

    let frames = candle_gen::lock_recover(&frames).clone();
    save_strip(&frames, &format!("control_{size}_s{steps}"));
    save_png(
        &active.pixels,
        active.width,
        active.height,
        &format!("control_{size}_s{steps}_final"),
    );
    assert_the_strip_converges("ctrl", &frames, &active, steps as u32, size, size);
}
