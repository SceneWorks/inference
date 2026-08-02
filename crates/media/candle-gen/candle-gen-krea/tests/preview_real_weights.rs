//! sc-16950 — candle Krea 2 per-step latent **preview** real-weight validation (epic 16948).
//!
//! Two independent things a shape-only smoke cannot establish, and which this epic requires of every
//! wiring story:
//!
//! 1. **The reused fit belongs to this latent space.** Two rows, deliberately separate:
//!    [`krea_vae_bytes_are_the_pinned_shared_snapshot`] pins the Krea side (the `vae/` SHA-256 both
//!    Krea snapshots publish, and the `latents_mean`/`latents_std` that define the normalized space),
//!    and [`krea_vae_matches_the_qwen_fit_vae_tensor_for_tensor`] is the reuse gate proper — a
//!    tensor-by-tensor comparison against the Qwen-Image VAE the epic-16624 fit was measured on.
//! 2. **The frames actually develop.** [`turbo_preview_frames_evolve_toward_the_final_image`] renders
//!    through the registered `krea_2_turbo` engine with a live sink, checks the numbering contract,
//!    checks seeded byte-identity against an inert render, and measures that each frame is closer to
//!    the finished image than the one before it. The strip is written out for direct review.
//!
//! ```sh
//! KREA_TURBO_DIR=D:\models\Krea-2-Turbo \
//! QWEN_IMAGE_VAE_FILE=D:\models\qwen-image-mlx\q8\vae\diffusion_pytorch_model.safetensors \
//! KREA_PREVIEW_ARTIFACT_DIR=D:\out\sc-16950 \
//!   cargo test -p candle-gen-krea --release --features cuda --test preview_real_weights \
//!     -- --ignored --nocapture
//! ```
//!
//! `QWEN_IMAGE_VAE_FILE` is **required** by the comparison row, which fails rather than skips when it
//! is unset. That row is the story's central claim — that the epic-16624 fit is legitimately reusable
//! rather than a guess from a matching Rust type name — and a gate that early-returns reports SUCCESS
//! having proven nothing, which in a run log is indistinguishable from a gate that ran. The two
//! halves are separate rows for the same reason: whichever half did not run has to be visible.
//! `KREA_RAW_DIR` stays genuinely optional; it strengthens the Krea-side row without gating it.

use std::collections::BTreeMap;
use std::path::{Path, PathBuf};

use candle_gen::gen_core::{
    GenerationOutput, GenerationRequest, Image, LoadSpec, PreviewFrame, PreviewSink, WeightsSource,
};

const PROMPT: &str =
    "A medium-shot photograph of a red fox sitting in a snowy forest at golden hour.";

/// The `vae/diffusion_pytorch_model.safetensors` SHA-256 published by BOTH `krea/Krea-2-Turbo`
/// @ `1161245028ef398cd0a951101b2bbf486464f841` and `krea/Krea-2-Raw`
/// @ `4ad9f4b627a647fad78b3dfeebb09f2654aeb494` — one file, two snapshots, and the file every candle
/// Krea route loads (`crate::vae::load_vae` reads `<root>/vae`).
const KREA_VAE_SHA256: &str = "ab1b61103959913d6c7e628cf793dbb2ca4726a40a3b3ae206c52b8e75bf6f08";

/// The `q4|q8/vae/diffusion_pytorch_model.safetensors` SHA-256 of `SceneWorks/qwen-image-mlx`
/// @ `8080a4171f1c8b7fca6c30491eafbe6ffab754bf` — the snapshot the epic-16624 QwenVae fit was measured
/// against. A bf16 container of the same values Krea publishes as f32.
const QWEN_IMAGE_MLX_VAE_SHA256: &str =
    "0c8bc8b758c649abef9ea407b95408389a3b2f610d0d10fcb054fe171d0a8344";

/// The per-channel de-normalization that *defines* the 16-channel latent space the fit lives in. If
/// these ever diverge between the two snapshots the weights being identical would not be enough.
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

fn artifact_dir() -> PathBuf {
    env_path("KREA_PREVIEW_ARTIFACT_DIR")
        .unwrap_or_else(|| std::env::temp_dir().join("krea_preview_sc16950"))
}

// ── Provenance: the reuse is grounded in tensor bytes ─────────────────────────────────────────────

fn sha256_of(path: &Path) -> String {
    use sha2::{Digest, Sha256};
    let mut file = std::fs::File::open(path).unwrap_or_else(|e| panic!("open {path:?}: {e}"));
    let mut hasher = Sha256::new();
    std::io::copy(&mut file, &mut hasher).expect("hash the VAE file");
    format!("{:x}", hasher.finalize())
}

/// The single `.safetensors` under a snapshot's `vae/` dir.
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

/// Widen a raw safetensors tensor to `f64` values, so an f32 container and a bf16 container of the
/// same numbers compare equal. bf16 is the top 16 bits of the f32 bit pattern.
fn tensor_values(view: &safetensors::tensor::TensorView<'_>) -> Vec<f64> {
    let raw = view.data();
    match view.dtype() {
        safetensors::Dtype::F32 => raw
            .chunks_exact(4)
            .map(|b| f32::from_le_bytes([b[0], b[1], b[2], b[3]]) as f64)
            .collect(),
        safetensors::Dtype::BF16 => raw
            .chunks_exact(2)
            .map(|b| f32::from_bits(u32::from(u16::from_le_bytes([b[0], b[1]])) << 16) as f64)
            .collect(),
        safetensors::Dtype::F64 => raw
            .chunks_exact(8)
            .map(|b| f64::from_le_bytes(b.try_into().expect("8 bytes")))
            .collect(),
        other => panic!("unexpected VAE tensor dtype {other:?}"),
    }
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

/// An input this row cannot run without. Missing means **fail**, not skip: a row that early-returns
/// on an unset variable still reports SUCCESS, and a skipped gate is indistinguishable in the run
/// output from one that ran and proved something.
fn required_path(name: &str) -> PathBuf {
    env_path(name).unwrap_or_else(|| {
        panic!(
            "{name} must be set for this row — it is a provenance gate, and skipping it would \
             report success while proving nothing"
        )
    })
}

/// The Krea half: the `vae/` bytes every candle Krea route loads are the pinned file both published
/// snapshots share, and they define the 16-channel latent space the reused fit lives in.
///
/// This is the *weaker* half on its own — it says nothing about Qwen-Image. The comparison that makes
/// the reuse legitimate is [`krea_vae_matches_the_qwen_fit_vae_tensor_for_tensor`], and it is a
/// separate row precisely so running only this one cannot read as having proven that.
#[test]
#[ignore = "needs a real Krea snapshot (set KREA_TURBO_DIR; optionally KREA_RAW_DIR)"]
fn krea_vae_bytes_are_the_pinned_shared_snapshot() {
    let turbo = required_path("KREA_TURBO_DIR");

    let turbo_vae = vae_file(&turbo);
    let turbo_sha = sha256_of(&turbo_vae);
    eprintln!("Krea Turbo vae/  {turbo_sha}  {}", turbo_vae.display());
    assert_eq!(
        turbo_sha, KREA_VAE_SHA256,
        "the Krea Turbo VAE is not the file the reused fit was grounded against"
    );

    // Raw publishes the same file. Checking it explicitly is what makes "every Krea route shares one
    // latent space" a measurement rather than an assumption. Genuinely optional: the Turbo snapshot
    // alone already carries the file every route loads.
    if let Some(raw) = env_path("KREA_RAW_DIR") {
        let raw_vae = vae_file(&raw);
        let raw_sha = sha256_of(&raw_vae);
        eprintln!("Krea Raw   vae/  {raw_sha}  {}", raw_vae.display());
        assert_eq!(
            raw_sha, KREA_VAE_SHA256,
            "Krea Raw must load the same VAE bytes as Krea Turbo"
        );
    } else {
        eprintln!("note: set KREA_RAW_DIR to also pin the Raw snapshot's vae/ against Turbo's");
    }

    let (mean, std) = read_latent_stats(&turbo);
    assert_eq!(mean, LATENTS_MEAN, "latents_mean defines the fitted space");
    assert_eq!(std, LATENTS_STD, "latents_std defines the fitted space");
}

/// **The reuse gate.** `candle-gen-qwen-image::preview` ships the epic-16624 fit unchanged; a
/// tensor-by-tensor identity between the VAE Krea loads and the VAE that fit was measured against is
/// what makes that legitimate rather than a guess from a matching Rust type name.
///
/// Both inputs are required. There is no configuration of this row that passes without performing the
/// comparison.
#[test]
#[ignore = "needs both snapshots (set KREA_TURBO_DIR and QWEN_IMAGE_VAE_FILE)"]
fn krea_vae_matches_the_qwen_fit_vae_tensor_for_tensor() {
    let turbo = required_path("KREA_TURBO_DIR");
    let qwen_vae = required_path("QWEN_IMAGE_VAE_FILE");

    // Re-pinned here rather than borrowed from the row above: this row must establish for itself that
    // the file it compared is the file Krea loads.
    let turbo_vae = vae_file(&turbo);
    let turbo_sha = sha256_of(&turbo_vae);
    eprintln!("Krea Turbo vae/  {turbo_sha}  {}", turbo_vae.display());
    assert_eq!(
        turbo_sha, KREA_VAE_SHA256,
        "the Krea Turbo VAE is not the file the reused fit was grounded against"
    );

    let qwen_sha = sha256_of(&qwen_vae);
    eprintln!("Qwen-Image vae/  {qwen_sha}  {}", qwen_vae.display());
    if qwen_sha != QWEN_IMAGE_MLX_VAE_SHA256 {
        // Not a failure: the comparison below is dtype-agnostic by design, so a different container
        // of the same values is a legitimate input. Said out loud so the log records which it was.
        eprintln!(
            "  note: not the pinned SceneWorks/qwen-image-mlx container \
             ({QWEN_IMAGE_MLX_VAE_SHA256}) — the tensor comparison below is what decides"
        );
    }

    let krea_bytes = std::fs::read(&turbo_vae).expect("read the Krea VAE");
    let qwen_bytes = std::fs::read(&qwen_vae).expect("read the Qwen-Image VAE");
    let krea = safetensors::SafeTensors::deserialize(&krea_bytes).expect("parse the Krea VAE");
    let qwen = safetensors::SafeTensors::deserialize(&qwen_bytes).expect("parse the Qwen VAE");

    let krea_names: BTreeMap<_, _> = krea.tensors().into_iter().collect();
    let qwen_names: BTreeMap<_, _> = qwen.tensors().into_iter().collect();
    assert_eq!(
        krea_names.keys().collect::<Vec<_>>(),
        qwen_names.keys().collect::<Vec<_>>(),
        "the two VAEs must expose the same tensor set"
    );

    let mut values = 0usize;
    for (name, krea_view) in &krea_names {
        let qwen_view = &qwen_names[name];
        assert_eq!(
            krea_view.shape(),
            qwen_view.shape(),
            "{name}: shapes must match"
        );
        let a = tensor_values(krea_view);
        let b = tensor_values(qwen_view);
        assert_eq!(a, b, "{name}: tensor values differ between the two VAEs");
        values += a.len();
    }
    eprintln!(
        "VAE transfer is exact: {} tensors / {values} values identical (f32 container vs bf16 container)",
        krea_names.len()
    );
    assert_eq!(krea_names.len(), 194);
    assert_eq!(values, 126_892_531);
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

/// Box-downsample an RGB8 image to `(w, h)`. The preview is a latent-resolution projection of the
/// trajectory, so the finished image has to come down to that size to be comparable.
fn downsample(img: &Image, w: u32, h: u32) -> Vec<u8> {
    downsample_raw(&img.pixels, img.width, img.height, w, h)
}

/// [`downsample`] over a raw RGB8 buffer.
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

/// Write the frames side by side as one strip, plus each frame individually — the artifact the epic
/// asks to be reviewed directly.
fn save_strip(frames: &[PreviewFrame], name: &str) {
    let dir = artifact_dir();
    std::fs::create_dir_all(&dir).expect("create the artifact dir");
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
        let path = dir.join(format!("{name}_frame{:02}.png", frame.current));
        image::save_buffer(
            &path,
            &frame.image.pixels,
            frame.image.width,
            frame.image.height,
            image::ExtendedColorType::Rgb8,
        )
        .expect("save a frame");
    }
    let path = dir.join(format!("{name}_strip.png"));
    image::save_buffer(
        &path,
        &strip,
        strip_w as u32,
        h as u32,
        image::ExtendedColorType::Rgb8,
    )
    .expect("save the strip");
    eprintln!("  saved {}", path.display());
}

fn save_image(img: &Image, name: &str) {
    let dir = artifact_dir();
    std::fs::create_dir_all(&dir).expect("create the artifact dir");
    let path = dir.join(format!("{name}.png"));
    image::save_buffer(
        &path,
        &img.pixels,
        img.width,
        img.height,
        image::ExtendedColorType::Rgb8,
    )
    .expect("save an image");
    eprintln!("  saved {}", path.display());
}

fn one_image(out: GenerationOutput) -> Image {
    let GenerationOutput::Images(mut images) = out else {
        panic!("expected GenerationOutput::Images");
    };
    assert_eq!(images.len(), 1);
    images.pop().expect("one image")
}

/// The story's runtime gate: a real 8-step Turbo render emits exactly 8 frames numbered 1..=8 with
/// `total == 8`, is byte-identical to the same seeded render with an inert sink, and the frames march
/// monotonically closer to the finished image instead of being eight views of noise.
#[test]
#[ignore = "needs the real Krea 2 Turbo snapshot on a CUDA box (set KREA_TURBO_DIR)"]
fn turbo_preview_frames_evolve_toward_the_final_image() {
    // Required, not skipped, for the same reason the provenance rows are: asking for `--ignored` is
    // already the opt-in, and past that point a green row must mean the render happened.
    let root = required_path("KREA_TURBO_DIR");
    let steps: u32 = std::env::var("KREA_PREVIEW_STEPS")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(8);
    let size: u32 = std::env::var("KREA_PREVIEW_SIZE")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(1024);

    let spec = LoadSpec::new(WeightsSource::Dir(root));
    let generator = candle_gen_krea::provider_registry()
        .unwrap()
        .load("krea_2_turbo", &spec)
        .expect("load krea_2_turbo");

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

    let frames = std::sync::Arc::new(std::sync::Mutex::new(Vec::<PreviewFrame>::new()));
    let captured = std::sync::Arc::clone(&frames);
    let active_request = GenerationRequest {
        preview: PreviewSink::new(move |frame| candle_gen::lock_recover(&captured).push(frame)),
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
    assert_eq!(
        frames
            .iter()
            .map(|f| (f.current, f.total))
            .collect::<Vec<_>>(),
        (1..=steps).map(|n| (n, steps)).collect::<Vec<_>>(),
        "a {steps}-step render must emit exactly {steps} frames numbered 1..={steps}"
    );

    let latent_side = size / 8;
    for frame in &frames {
        assert_eq!(
            (frame.image.width, frame.image.height),
            (latent_side, latent_side),
            "frames are latent-resolution projections"
        );
    }

    save_strip(&frames, &format!("turbo_{size}_s{steps}"));
    save_image(&active, &format!("turbo_{size}_s{steps}_final"));

    // Every frame must differ from its predecessor — eight copies of one image would satisfy a naive
    // "N frames arrived" check while showing nothing developing.
    for pair in frames.windows(2) {
        let delta = mean_abs_delta(&pair[0].image.pixels, &pair[1].image.pixels);
        eprintln!(
            "  frame {:>2} → {:>2}: mean |Δ| {delta:.2}",
            pair[0].current, pair[1].current
        );
        assert!(
            delta > 0.5,
            "frames {} and {} are effectively identical (mean |Δ| {delta:.3})",
            pair[0].current,
            pair[1].current
        );
    }

    // And they must converge: distance to the finished image, at preview resolution, falls from the
    // first frame to the last, and the last frame resembles the render rather than noise.
    let target = downsample(&active, latent_side, latent_side);
    let distances: Vec<f64> = frames
        .iter()
        .map(|f| mean_abs_delta(&f.image.pixels, &target))
        .collect();
    for (frame, distance) in frames.iter().zip(&distances) {
        eprintln!(
            "  frame {:>2}: mean |Δ| to final {distance:.2}",
            frame.current
        );
    }
    let (first, last) = (distances[0], distances[distances.len() - 1]);
    assert!(
        last < first * 0.6,
        "the strip must converge on the final image (first {first:.2} → last {last:.2})"
    );
    assert!(
        distances.windows(2).all(|p| p[1] < p[0]),
        "distance to the finished image must fall at every step: {distances:?}"
    );

    // The per-pixel metric above is dominated by the residual noise still in the trajectory — a hook
    // emits *before* each step, so the last frame is one solver advancement short of the render and
    // will never be pixel-close. What "resembles the final image" actually means for a decorative
    // preview is compositional: subject placement and colour masses. Measure that by taking both
    // sides down to a 16² thumbnail, which averages the residual noise away and leaves the layout.
    let coarse = 16u32;
    let coarse_target = downsample(&active, coarse, coarse);
    let coarse_distances: Vec<f64> = frames
        .iter()
        .map(|f| {
            mean_abs_delta(
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
    for (frame, distance) in frames.iter().zip(&coarse_distances) {
        eprintln!(
            "  frame {:>2}: coarse {coarse}² mean |Δ| to final {distance:.2}",
            frame.current
        );
    }
    let (coarse_first, coarse_last) = (
        coarse_distances[0],
        coarse_distances[coarse_distances.len() - 1],
    );
    assert!(
        coarse_last < coarse_first * 0.35,
        "the composition must resolve across the strip (first {coarse_first:.2} → last {coarse_last:.2})"
    );

    // Absolute distance can only ever say "closer", never "resembles": the projection is a global
    // linear approximation of the decode (R² 0.9586), so even a perfectly converged latent keeps an
    // offset and gain error against the true pixels. Correlation is the metric that survives both that
    // and the residual noise, and it is what "the preview looks like the image" actually means.
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
            "  frame {:>2}: coarse correlation with final {r:+.3}",
            frame.current
        );
    }
    let (r_first, r_last) = (correlations[0], correlations[correlations.len() - 1]);
    assert!(
        r_first < 0.35,
        "the first frame is pre-denoise noise and must NOT already resemble the render (r {r_first:+.3})"
    );
    // The retained 8-step 1024² run measured +0.144 → +0.987, rising at every step. The floor is set
    // below that with headroom for a different prompt; a very low step count legitimately lands lower
    // (the sc-16630 Lens-Turbo four-step observation), which is why the step count is not overridden
    // by default.
    assert!(
        r_last > 0.85,
        "the last preview frame must resemble the finished render (r {r_last:+.3})"
    );
    assert!(
        correlations.windows(2).all(|p| p[1] > p[0]),
        "resemblance must increase at every step: {correlations:?}"
    );
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
