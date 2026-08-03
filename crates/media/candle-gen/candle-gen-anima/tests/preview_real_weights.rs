//! sc-16953 — candle Anima per-step latent **preview** real-weight validation (epic 16948).
//!
//! Two independent things a shape-only smoke cannot establish, and which this epic requires of every
//! wiring story:
//!
//! 1. **The reused fit belongs to this latent space.** Anima adds no fit; it reuses the epic-16624
//!    QwenVae constants `candle-gen-qwen-image` ships. [`anima_vae_bytes_are_the_pinned_snapshot`]
//!    pins the Anima side (the `vae/` shard every Anima route loads), and
//!    [`anima_vae_matches_the_qwen_fit_vae_tensor_for_tensor`] is the reuse gate proper — a
//!    tensor-by-tensor comparison against the Qwen-Image VAE the fit was measured on, through
//!    `crate::vae::convert_vae_key`, because Anima publishes the same weights under the **original**
//!    Qwen naming and therefore in a different file.
//! 2. **The frames actually develop.** [`base_preview_frames_evolve_toward_the_final_image`],
//!    [`aesthetic_preview_frames_evolve_toward_the_final_image`] and
//!    [`turbo_preview_frames_evolve_toward_the_final_image`] render each shipped variant through the
//!    registered `Generator` seam with a live sink, check the numbering contract, check seeded
//!    byte-identity against an inert render, and measure that each frame is closer to the finished
//!    image than the one before it. All three strips are written out for direct review.
//!
//! All three variants get a runtime row rather than one standing in for the others: `anima_base` and
//! `anima_aesthetic` run **true CFG** (two DiT forwards per evaluation) while `anima_turbo` is the
//! merged CFG-free student, and "the preview never projects the unconditional half" is only worth
//! measuring on the lanes that have one.
//!
//! ```sh
//! ANIMA_PREVIEW_DIR=E:\huggingface\hub\models--circlestone-labs--Anima\snapshots\<rev>\split_files \
//! ANIMA_QWEN_FIT_VAE=E:\huggingface\hub\models--SceneWorks--qwen-image-mlx\snapshots\<rev>\q8\vae\diffusion_pytorch_model.safetensors \
//! ANIMA_PREVIEW_ARTIFACT_DIR=E:\out\sc-16953 \
//!   cargo test -p candle-gen-anima --release --features cuda --test preview_real_weights \
//!     -- --ignored --nocapture
//! ```
//!
//! Every input is **required** by the row that uses it: a row that early-returns on an unset variable
//! still reports SUCCESS, and in a run log a skipped gate is indistinguishable from one that ran and
//! proved something. Asking for `--ignored` is already the opt-in.

use std::collections::BTreeMap;
use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex};

use candle_gen::candle_core::{DType, Device};
use candle_gen::gen_core::{
    GenerationOutput, GenerationRequest, Image, LoadSpec, PreviewFrame, PreviewSink, WeightsSource,
};
use candle_gen_anima::Variant;

const PROMPT: &str =
    "Anime illustration of a silver-haired traveler beneath cherry blossoms at sunset, detailed, \
     cinematic lighting.";
const NEGATIVE: &str = "blurry, lowres, artifacts, watermark, text";

/// The SHA-256 of `split_files/vae/qwen_image_vae.safetensors` in `circlestone-labs/Anima`
/// @ `53eec3898af698b2cf2a11379021fc9c5465d228` — 253,806,246 bytes, the single VAE shard **every**
/// Anima variant loads (`crate::loader` reads one `vae/` file for all three DiT checkpoints).
const ANIMA_VAE_SHA256: &str = "a70580f0213e67967ee9c95f05bb400e8fb08307e017a924bf3441223e023d1f";

/// The SHA-256 of `q4|q8/vae/diffusion_pytorch_model.safetensors` in `SceneWorks/qwen-image-mlx`
/// @ `8080a4171f1c8b7fca6c30491eafbe6ffab754bf` — 253,806,966 bytes, the snapshot the epic-16624
/// QwenVae fit was measured against and the file sc-16952 pinned for every candle Qwen-Image lane.
///
/// A **different file** from Anima's: same tensors, published under the original rather than the
/// diffusers naming, so the 720-byte difference is the safetensors header alone. Which is exactly why
/// the tensor-by-tensor row below exists and a hash equality would not have done.
const QWEN_FIT_VAE_SHA256: &str =
    "0c8bc8b758c649abef9ea407b95408389a3b2f610d0d10fcb054fe171d0a8344";

/// The measured extent of the transfer, pinned so a partial comparison cannot pass as a full one.
const VAE_TENSORS: usize = 194;
const VAE_VALUES: usize = 126_892_531;

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
    env_path("ANIMA_PREVIEW_ARTIFACT_DIR")
        .unwrap_or_else(|| std::env::temp_dir().join("anima_preview_sc16953"))
}

// ── Provenance: the reuse is grounded in tensor bytes ─────────────────────────────────────────────

fn sha256_of(path: &Path) -> String {
    use sha2::{Digest, Sha256};
    let mut file = std::fs::File::open(path).unwrap_or_else(|e| panic!("open {path:?}: {e}"));
    let mut hasher = Sha256::new();
    std::io::copy(&mut file, &mut hasher).expect("hash the file");
    format!("{:x}", hasher.finalize())
}

/// The single `.safetensors` under an Anima snapshot's `split_files/vae/` — the file
/// `crate::vae::load_vae` is handed, for every variant.
fn anima_vae_file(root: &Path) -> PathBuf {
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

/// The Anima half: the `vae/` bytes every Anima variant loads are the pinned published shard.
///
/// This is the *weaker* half on its own — it says nothing about Qwen-Image. The comparison that makes
/// the reuse legitimate is [`anima_vae_matches_the_qwen_fit_vae_tensor_for_tensor`], and it is a
/// separate row precisely so running only this one cannot read as having proven that.
#[test]
#[ignore = "needs a real Anima snapshot (set ANIMA_PREVIEW_DIR)"]
fn anima_vae_bytes_are_the_pinned_snapshot() {
    let root = required_path("ANIMA_PREVIEW_DIR");
    let vae = anima_vae_file(&root);
    let sha = sha256_of(&vae);
    let size = std::fs::metadata(&vae).expect("stat the VAE").len();
    eprintln!("anima vae/  {sha}  {size} bytes  {}", vae.display());
    assert_eq!(
        sha, ANIMA_VAE_SHA256,
        "the VAE this snapshot publishes is not the file the reuse was grounded against"
    );
    assert_eq!(size, 253_806_246);

    // Anima publishes no `vae/config.json`, and needs none: candle's `QwenVae` carries
    // `latents_mean` / `latents_std` — the per-channel de-normalization that *defines* the fitted
    // space — as Rust constants, and this crate reuses that very type. Asserted as an absence so the
    // reasoning is pinned rather than implied: if a config ever appears beside the shard, whoever
    // adds it has to decide here whether it is authoritative.
    assert!(
        !root.join("vae").join("config.json").exists(),
        "a vae/config.json appeared beside the Anima shard — the normalized latent space is \
         currently defined by candle_gen_qwen_image::vae's Rust constants, so decide which is \
         authoritative before trusting the reused fit"
    );
}

/// **The reuse gate.** `candle-gen-qwen-image::preview` ships the epic-16624 fit unchanged; a
/// tensor-by-tensor identity between the VAE Anima loads and the VAE that fit was measured against is
/// what makes reusing it legitimate rather than a guess from a matching Rust type name.
///
/// The comparison runs through [`candle_gen_anima::vae::convert_vae_key`] — the production rename —
/// so it also proves that rename is a total bijection onto the diffusers key set, which is the only
/// reason `QwenVae` can read Anima's file at all.
///
/// Both inputs are required. There is no configuration of this row that passes without performing the
/// comparison.
#[test]
#[ignore = "needs both snapshots (set ANIMA_PREVIEW_DIR and ANIMA_QWEN_FIT_VAE)"]
fn anima_vae_matches_the_qwen_fit_vae_tensor_for_tensor() {
    let root = required_path("ANIMA_PREVIEW_DIR");
    let qwen_vae = required_path("ANIMA_QWEN_FIT_VAE");

    // Re-pinned here rather than borrowed from the row above: this row must establish for itself that
    // the file it compared is the file Anima loads.
    let anima_vae = anima_vae_file(&root);
    let anima_sha = sha256_of(&anima_vae);
    let qwen_sha = sha256_of(&qwen_vae);
    eprintln!("anima     vae/  {anima_sha}  {}", anima_vae.display());
    eprintln!("qwen-fit  vae/  {qwen_sha}  {}", qwen_vae.display());
    assert_eq!(
        anima_sha, ANIMA_VAE_SHA256,
        "the Anima VAE is not the file the reuse was grounded against"
    );
    assert_eq!(
        qwen_sha, QWEN_FIT_VAE_SHA256,
        "ANIMA_QWEN_FIT_VAE must be the snapshot the epic-16624 fit was measured on"
    );
    assert_ne!(
        anima_sha, qwen_sha,
        "the two files are deliberately different containers of the same tensors — if they ever \
         became byte-identical this row's rename argument would be dead code"
    );

    let load = |path: &Path| {
        candle_gen::candle_core::safetensors::load(path, &Device::Cpu)
            .unwrap_or_else(|e| panic!("parse {path:?}: {e}"))
    };
    let anima: BTreeMap<String, _> = load(&anima_vae).into_iter().collect();
    let qwen: BTreeMap<String, _> = load(&qwen_vae).into_iter().collect();

    // The rename must be a bijection onto the fit donor's key set: no collision, no orphan either way.
    let renamed: BTreeMap<String, String> = anima
        .keys()
        .map(|key| (candle_gen_anima::vae::convert_vae_key(key), key.clone()))
        .collect();
    assert_eq!(
        renamed.len(),
        anima.len(),
        "convert_vae_key collapsed two Anima keys onto one diffusers name"
    );
    assert_eq!(
        renamed.keys().collect::<Vec<_>>(),
        qwen.keys().collect::<Vec<_>>(),
        "the renamed Anima key set must be exactly the fit donor's"
    );

    let values_of = |tensor: &candle_gen::candle_core::Tensor| -> Vec<f32> {
        tensor
            .to_dtype(DType::F32)
            .and_then(|t| t.flatten_all())
            .and_then(|t| t.to_vec1::<f32>())
            .expect("widen a VAE tensor to f32")
    };

    let mut values = 0usize;
    for (diffusers_key, original_key) in &renamed {
        let a = &anima[original_key];
        let q = &qwen[diffusers_key];
        assert_eq!(
            a.dims(),
            q.dims(),
            "{original_key} → {diffusers_key}: shapes must match"
        );
        assert_eq!(
            a.dtype(),
            q.dtype(),
            "{original_key} → {diffusers_key}: both snapshots publish bf16 containers, so a dtype \
             difference means one of them was re-quantized"
        );
        let (a, q) = (values_of(a), values_of(q));
        assert_eq!(
            a, q,
            "{original_key} → {diffusers_key}: tensor values differ between the two VAEs"
        );
        values += a.len();
    }
    eprintln!(
        "VAE transfer is exact: {} tensors / {values} values bit-identical (original vs diffusers \
         naming, both bf16)",
        renamed.len()
    );
    assert_eq!(renamed.len(), VAE_TENSORS);
    assert_eq!(values, VAE_VALUES);
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
/// the finished image, and rising correlation with it. Applied identically to all three variants so
/// none can be closed with a weaker measurement than the others.
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

    // The frames are VAE-latent resolution `H/8 × W/8`. That is also the proof the projection ran
    // after the temporal squeeze: the running latent is `[1, 16, 1, H/8, W/8]`, which fails the
    // `[1, C, h, w]` contract outright, so an un-squeezed projection could not have produced a frame
    // at all and the strip would be empty.
    for frame in frames {
        assert_eq!(
            (frame.image.width, frame.image.height),
            (width / 8, height / 8),
            "{label}: frames must be VAE-latent resolution"
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
    // offset and gain error against the true pixels. A hook also emits *before* each step, so the
    // last frame is one solver advancement short of the render. Correlation over a coarse thumbnail —
    // which averages the residual noise away and leaves subject placement and colour masses — is what
    // "the preview looks like the image" actually means for a decorative frame.
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
    // starts at a non-zero, *scene-dependent* correlation floor rather than at zero. A fixed low
    // ceiling on `r_first` would be reading that floor as if it were resemblance and would fail an
    // honest lane for the colour of its prompt (sc-16952's finding; sc-16950's `r_first < 0.35`
    // deliberately not ported).
    //
    // The rise is what cannot be faked: a strip that opened on the finished image — the failure this
    // guards, and the one a naive "N frames arrived" check misses — has nowhere to rise to. It is
    // layered with the strictly monotone rise above, the falling mean |Δ|, and the per-frame movement
    // floor.
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

/// Render one variant twice on one warmed generator at the same seed — once with an inert sink, once
/// with a live one — and hold the strip to [`assert_the_strip_converges`].
fn assert_variant_previews_converge(variant: Variant, steps: u32, size: u32) {
    let root = required_path("ANIMA_PREVIEW_DIR");
    let id = variant.id();
    eprintln!(
        "── {id}: {size}² × {steps} steps, CFG {}",
        variant.uses_cfg()
    );

    let generator = candle_gen_anima::provider_registry()
        .expect("anima registry")
        .load(id, &LoadSpec::new(WeightsSource::Dir(root)))
        .unwrap_or_else(|e| panic!("load {id}: {e}"));

    let base = GenerationRequest {
        prompt: PROMPT.into(),
        negative_prompt: variant.uses_cfg().then(|| NEGATIVE.into()),
        width: size,
        height: size,
        count: 1,
        seed: Some(16953),
        steps: Some(steps),
        ..Default::default()
    };

    // Inert first: the byte-identity baseline, on the same warmed generator.
    let inert = one_image(
        generator
            .generate(&base, &mut |_| {})
            .unwrap_or_else(|e| panic!("{id} inert-sink render: {e}")),
    );

    let (sink, frames) = collecting_sink();
    let active_request = GenerationRequest {
        preview: sink,
        ..base
    };
    let active = one_image(
        generator
            .generate(&active_request, &mut |_| {})
            .unwrap_or_else(|e| panic!("{id} active-sink render: {e}")),
    );
    assert_eq!(
        inert.pixels, active.pixels,
        "{id}: an active preview sink must not change a single output byte at the same seed"
    );

    let frames = candle_gen::lock_recover(&frames).clone();
    let name = format!("{id}_{size}_s{steps}");
    save_strip(&frames, &name);
    save_png(
        &active.pixels,
        active.width,
        active.height,
        &format!("{name}_final"),
    );
    assert_the_strip_converges(id, &frames, &active, steps, size, size);
}

/// `anima_base` — the CFG lane, and the variant the descriptor defaults describe.
#[test]
#[ignore = "needs a real Anima snapshot on a CUDA box (set ANIMA_PREVIEW_DIR)"]
fn base_preview_frames_evolve_toward_the_final_image() {
    assert_variant_previews_converge(
        Variant::Base,
        env_u32("ANIMA_PREVIEW_STEPS", 12),
        env_u32("ANIMA_PREVIEW_SIZE", 1024),
    );
}

/// `anima_aesthetic` — the second CFG lane. A separate row rather than an assumption that it follows
/// the base: it is a different DiT checkpoint, and it is a shipped route.
#[test]
#[ignore = "needs a real Anima snapshot on a CUDA box (set ANIMA_PREVIEW_DIR)"]
fn aesthetic_preview_frames_evolve_toward_the_final_image() {
    assert_variant_previews_converge(
        Variant::Aesthetic,
        env_u32("ANIMA_PREVIEW_STEPS", 12),
        env_u32("ANIMA_PREVIEW_SIZE", 1024),
    );
}

/// `anima_turbo` — the merged CFG-free few-step student, so the predict closure runs a **single** DiT
/// forward. Held to the same measurement as the CFG lanes on a shorter schedule.
#[test]
#[ignore = "needs a real Anima snapshot on a CUDA box (set ANIMA_PREVIEW_DIR)"]
fn turbo_preview_frames_evolve_toward_the_final_image() {
    assert_variant_previews_converge(
        Variant::Turbo,
        env_u32("ANIMA_PREVIEW_TURBO_STEPS", 10),
        env_u32("ANIMA_PREVIEW_SIZE", 1024),
    );
}
