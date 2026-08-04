//! sc-16958 — candle SD3.5 per-step latent **preview** real-weight validation (epic 16948).
//!
//! Three things a shape-only smoke cannot establish, and which this epic requires of every wiring
//! story:
//!
//! 1. **The reused fit belongs to this latent space, and this latent space is SD3.5's own.** SD3.5
//!    adds no fit; it reuses the epic-16624 constants `mlx-gen-sd3` measured.
//!    [`the_three_sd3_snapshots_ship_one_identical_vae`] is what lets one fit cover three registered
//!    routes, and [`the_sd3_vae_is_not_the_flux1_latent_space`] is the row this epic specifically
//!    asked for: sc-16956 (Boogu) and sc-16957 (Z-Image) each found their "16 channels" to be
//!    FLUX.1-dev's space, so SD3.5 must be settled rather than assumed either way. It is settled by a
//!    tensor walk, because the two containers are the same architecture at the same byte size and a
//!    size comparison would say nothing.
//! 2. **The frames actually develop, on every shipped route and both of its lanes.**
//!    [`large_preview_frames_evolve_toward_the_final_image`],
//!    [`turbo_preview_frames_evolve_toward_the_final_image`],
//!    [`medium_preview_frames_evolve_toward_the_final_image`] and
//!    [`img2img_preview_frames_evolve_from_the_forked_latent`] render through the registered
//!    `Generator` seam with a live sink, check the numbering contract, check seeded byte-identity
//!    against an inert render, and measure that each frame is closer to the finished image than the
//!    one before it. Every strip is written out for direct review.
//! 3. **Exactly one frame per outer step on a multi-eval solver.**
//!    [`a_multi_eval_solver_emits_one_frame_per_outer_step`] runs `heun` and proves the guard is
//!    non-vacuous *first*: the shared driver calls `on_progress` once per **evaluation**, so counting
//!    `Progress::Step` events is counting evaluations, and the row asserts there are more of them than
//!    outer steps before it asserts the frame count collapsed to the outer steps.
//!
//! All three variants get a runtime row rather than one standing in for the others: `sd3_5_large` and
//! `sd3_5_medium` run **true CFG** (two MMDiT forwards per evaluation, over different transformers)
//! while `sd3_5_large_turbo` is the guidance-distilled single-forward student, and "the preview never
//! projects the unconditional half" is only worth measuring on the lanes that have one.
//!
//! ```sh
//! SD3_LARGE_SNAPSHOT=E:\huggingface\hub\models--stabilityai--stable-diffusion-3.5-large\snapshots\<rev> \
//! SD3_TURBO_SNAPSHOT=E:\huggingface\hub\models--stabilityai--stable-diffusion-3.5-large-turbo\snapshots\<rev> \
//! SD3_MEDIUM_SNAPSHOT=E:\huggingface\hub\models--stabilityai--stable-diffusion-3.5-medium\snapshots\<rev> \
//! SD3_FLUX1_VAE=E:\huggingface\hub\models--black-forest-labs--FLUX.1-dev\snapshots\<rev>\vae\diffusion_pytorch_model.safetensors \
//! SD3_PREVIEW_ARTIFACT_DIR=E:\out\sc-16958 \
//!   cargo test -p candle-gen-sd3 --release --features cuda --test preview_real_weights \
//!     -- --ignored --nocapture
//! ```
//!
//! Every input is **required** by the row that uses it: a row that early-returns on an unset variable
//! still reports SUCCESS, and in a run log a skipped gate is indistinguishable from one that ran and
//! proved something. Asking for `--ignored` is already the opt-in.
//!
//! [`the_flow_cohort_needs_no_sigma_correction`] is the **only non-`#[ignore]`d row in this file** and
//! runs on the committed constants alone — it is the row that must appear in a plain `cargo test` of
//! this file. sc-16954 shipped a red row that hid because the only non-ignored row in its file was
//! excluded by `-- --ignored`.

use std::collections::BTreeMap;
use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex};

use candle_gen::candle_core::{DType, Device, Tensor};
use candle_gen::gen_core::{
    Conditioning, GenerationOutput, GenerationRequest, Image, LoadSpec, PreviewFrame, PreviewSink,
    Progress, WeightsSource,
};

const PROMPT: &str =
    "A weathered brass astrolabe on a navigator's desk beside a cracked leather map, warm lamplight, \
     deep shadows, photographic detail.";
const NEGATIVE: &str = "blurry, lowres, artifacts, watermark, text";
const SEED: u64 = 16958;

/// The img2img / `Reference` strength the fork row uses. Named because two things must agree on it —
/// the request that is rendered and the expected emitted-frame count derived through
/// `pipeline::init_time_step` — and a literal repeated in both places is exactly how they drift.
const IMG2IMG_STRENGTH: f32 = 0.6;

/// The SHA-256 of `vae/diffusion_pytorch_model.safetensors` — 167,666,902 bytes, 244 bf16 tensors —
/// which **all three** registered SD3.5 snapshots ship identically:
/// `stabilityai/stable-diffusion-3.5-large` @ `ceddf0a7fdf2064ea28e2213e3b84e4afa170a0f` (the
/// revision the epic-16624 fit was measured on), `…-large-turbo` @
/// `ec07796fc06b096cc56de9762974a28f4c632eda`, and `…-medium` @
/// `b940f670f0eda2d07fbb75229e779da1ad11eb80`.
const SD3_VAE_SHA256: &str = "8f53304a79335b55e13ec50f63e5157fee4deb2f30d5fae0654e2b2653c109dc";
const SD3_VAE_BYTES: u64 = 167_666_902;

/// The SHA-256 of the SD3.5 `vae/config.json` (809 bytes) the fit's provenance record names — the
/// file that carries the `1.5305` / `0.0609` normalization defining the fitted space.
const SD3_VAE_CONFIG_SHA256: &str =
    "58557f2439dfa867450caef425b5d11160be8aa9c34d60dbf23a94a6a94cb060";

/// The SHA-256 of `vae/diffusion_pytorch_model.safetensors` in `black-forest-labs/FLUX.1-dev` @
/// `3de623fc3c33e44ffbe2bad470d0f45bccf2eb21` — the container sc-16956 pinned for FLUX.1 and
/// sc-16957 proved Z-Image also ships. **The same 167,666,902 bytes as SD3.5's**, which is exactly
/// why the row below walks tensors instead of comparing sizes.
const FLUX1_VAE_SHA256: &str = "f5b59a26851551b67ae1fe58d32e76486e1e812def4696a4bea97f16604d40a3";

/// The measured extent of the comparison, pinned so a partial walk cannot pass as a full one.
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
    env_path("SD3_PREVIEW_ARTIFACT_DIR")
        .unwrap_or_else(|| std::env::temp_dir().join("sd3_preview_sc16958"))
}

// ── Provenance: the reuse is grounded in tensor bytes ─────────────────────────────────────────────

fn sha256_of(path: &Path) -> String {
    use sha2::{Digest, Sha256};
    let mut file = std::fs::File::open(path).unwrap_or_else(|e| panic!("open {path:?}: {e}"));
    let mut hasher = Sha256::new();
    std::io::copy(&mut file, &mut hasher).expect("hash the file");
    format!("{:x}", hasher.finalize())
}

/// The `vae/diffusion_pytorch_model.safetensors` under a diffusers snapshot — the file
/// `crate::vae::load_vae` is handed, for every variant.
fn snapshot_vae(root: &Path) -> PathBuf {
    let vae = root.join("vae").join("diffusion_pytorch_model.safetensors");
    assert!(
        vae.is_file(),
        "{vae:?} is missing — this row needs the VAE the pipeline actually loads"
    );
    vae
}

/// The three registered routes and the env var naming each one's snapshot.
const SNAPSHOT_VARS: [(&str, &str); 3] = [
    ("sd3_5_large", "SD3_LARGE_SNAPSHOT"),
    ("sd3_5_large_turbo", "SD3_TURBO_SNAPSHOT"),
    ("sd3_5_medium", "SD3_MEDIUM_SNAPSHOT"),
];

/// One fit covers three registered routes because the three snapshots publish the **same VAE file**,
/// not because they share a channel count.
///
/// All three are required. There is no configuration of this row that passes having checked one
/// snapshot and assumed the others — which is the specific mistake this epic asked SD3.5 not to make,
/// since `sd3_5_large_turbo` and `sd3_5_medium` are separate repositories with separate revisions.
#[test]
#[ignore = "needs all three SD3.5 snapshots (set SD3_LARGE_SNAPSHOT, SD3_TURBO_SNAPSHOT, SD3_MEDIUM_SNAPSHOT)"]
fn the_three_sd3_snapshots_ship_one_identical_vae() {
    for (id, var) in SNAPSHOT_VARS {
        let root = required_path(var);
        let vae = snapshot_vae(&root);
        let sha = sha256_of(&vae);
        let size = std::fs::metadata(&vae).expect("stat the VAE").len();
        eprintln!("  {id:<18} vae/  {sha}  {size} bytes");
        assert_eq!(
            sha, SD3_VAE_SHA256,
            "{id}: the VAE this snapshot publishes is not the file the reused fit was measured on"
        );
        assert_eq!(size, SD3_VAE_BYTES, "{id}: unexpected VAE container size");

        // The normalization is half the definition of the fitted space, so the config is pinned too —
        // a snapshot that kept the weights but re-scaled them would project wrong while passing a
        // weights-only check.
        let config = root.join("vae").join("config.json");
        let config_sha = sha256_of(&config);
        eprintln!("  {id:<18} vae/config.json  {config_sha}");
        assert_eq!(
            config_sha, SD3_VAE_CONFIG_SHA256,
            "{id}: the VAE config — and therefore the scale/shift defining the fitted latent \
             space — is not the one the fit was measured under"
        );
    }

    // The engine's own constants must be the ones that config carries, or the space the sampler
    // integrates is not the space the config pins.
    assert_eq!(candle_gen_sd3::vae::SCALING_FACTOR, 1.5305);
    assert_eq!(candle_gen_sd3::vae::SHIFT_FACTOR, 0.0609);
}

/// **The which-16-channel-space gate.** SD3.5's VAE is *not* FLUX.1-dev's.
///
/// Epic 16948 asked every 16-channel story to settle which space it occupies, because sc-16956 and
/// sc-16957 each found theirs to be FLUX.1-dev's — sc-16957 finding the *same container hash*, which
/// meant epic 16624 had committed two fits over one space. SD3.5 is the opposite finding, and it can
/// only be reached by walking tensors: the two files are the same architecture, the same 244 keys with
/// the same shapes and the same bf16 dtype, and the **same 167,666,902-byte container size**. A size
/// or shape comparison would say "identical" about two different VAEs.
///
/// So this row asserts the strong form: every one of the 244 tensors **differs**. A single matching
/// tensor would mean the two lineages overlap and the reasoning above would need revisiting.
///
/// Both inputs are required.
#[test]
#[ignore = "needs SD3_LARGE_SNAPSHOT + SD3_FLUX1_VAE"]
fn the_sd3_vae_is_not_the_flux1_latent_space() {
    let sd3_vae = snapshot_vae(&required_path("SD3_LARGE_SNAPSHOT"));
    let flux_vae = required_path("SD3_FLUX1_VAE");

    // Re-pinned here rather than borrowed from the row above: this row must establish for itself that
    // the files it compared are the two it names.
    let (sd3_sha, flux_sha) = (sha256_of(&sd3_vae), sha256_of(&flux_vae));
    eprintln!("  sd3.5     vae/  {sd3_sha}");
    eprintln!("  flux1-dev vae/  {flux_sha}");
    assert_eq!(sd3_sha, SD3_VAE_SHA256, "not the SD3.5 VAE");
    assert_eq!(
        flux_sha, FLUX1_VAE_SHA256,
        "SD3_FLUX1_VAE must be the FLUX.1-dev container sc-16956 pinned and sc-16957 found Z-Image \
         also ships"
    );
    assert_eq!(
        std::fs::metadata(&sd3_vae).expect("stat").len(),
        std::fs::metadata(&flux_vae).expect("stat").len(),
        "the two containers are deliberately the same size — if that ever stops being true, the \
         tensor walk below is no longer the only way to tell them apart and this row's reasoning \
         should be revisited rather than silently kept"
    );

    let load = |path: &Path| -> BTreeMap<String, Tensor> {
        candle_gen::candle_core::safetensors::load(path, &Device::Cpu)
            .unwrap_or_else(|e| panic!("parse {path:?}: {e}"))
            .into_iter()
            .collect()
    };
    let sd3 = load(&sd3_vae);
    let flux = load(&flux_vae);
    assert_eq!(
        sd3.keys().collect::<Vec<_>>(),
        flux.keys().collect::<Vec<_>>(),
        "the two VAEs must have the same key set — that shared architecture is the whole reason a \
         channel count could not settle this"
    );

    let values_of = |tensor: &Tensor| -> Vec<f32> {
        tensor
            .to_dtype(DType::F32)
            .and_then(|t| t.flatten_all())
            .and_then(|t| t.to_vec1::<f32>())
            .expect("widen a VAE tensor to f32")
    };

    let (mut values, mut identical) = (0usize, 0usize);
    for (key, sd3_tensor) in &sd3 {
        let flux_tensor = &flux[key];
        assert_eq!(sd3_tensor.dims(), flux_tensor.dims(), "{key}: shapes");
        assert_eq!(sd3_tensor.dtype(), flux_tensor.dtype(), "{key}: dtypes");
        let (a, b) = (values_of(sd3_tensor), values_of(flux_tensor));
        if a == b {
            identical += 1;
            eprintln!("  !! {key} is byte-identical between the two VAEs");
        }
        values += a.len();
    }
    eprintln!(
        "  walked {} tensors / {values} values: {identical} identical, {} differing",
        sd3.len(),
        sd3.len() - identical
    );
    assert_eq!(sd3.len(), VAE_TENSORS);
    assert_eq!(values, VAE_VALUES);
    assert_eq!(
        identical, 0,
        "SD3.5's VAE shares its architecture with FLUX.1-dev's but must share none of its trained \
         weights — {identical} tensor(s) matched, so the two lineages overlap and the claim that \
         SD3.5 occupies its own 16-channel space (and that sc-17309 needs no SD3.5 row) is wrong"
    );
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
                let (mut sum, mut n) = (0u32, 0u32);
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

/// Per-lane development criteria, each carrying **its own** measured numbers.
///
/// Every bound below is derived from that exact lane's own run — sc-16957 shipped a floor justified by
/// a different lane's measurement and sc-16956 shipped 0.25 of unexplained slack — and the headroom is
/// uniform and stated: **0.03 under a measured correlation, 0.06 over a measured distance ratio.**
///
/// Four bounds rather than one, because the lanes do not all develop from the same starting point:
///
/// * `max_r_first` / `min_rise` encode "the strip did not open on the finished image". For a txt2img
///   lane that starts from pure noise they are the shared loose `0.75` / `0.30` — loose because a
///   tight `r_first` bound would read the fit's own warm intercept (0.646, 0.626, 0.615 — R > G > B,
///   as most warm-lit renders are) as if it were resemblance, which is why sc-16950's `r_first < 0.35`
///   is deliberately not ported.
/// * They are the **wrong measurement for a forked lane**, and the run proved it: img2img at strength
///   0.6 starts at r **+0.954** because it forks from a VAE-encoded source rather than from noise, so
///   the shared bound's own message ("the first frame is pre-denoise noise") is simply false there.
///   Rather than loosen the bound for everyone, that lane declares its own and shifts the weight onto
///   `max_distance_ratio`, which is the statistic that still discriminates: a strip that had opened on
///   the finished image would sit at a ratio near 1.0, not 0.34.
///
/// Both the falling distance and the rising resemblance are additionally asserted **strictly monotone**
/// for every lane, which no set of endpoint bounds implies.
struct Develops {
    /// Floor under the measured final-frame correlation with the finished render.
    min_r_last: f64,
    /// Ceiling over the measured first-frame correlation — "it did not start as the render".
    max_r_first: f64,
    /// Floor under the measured `r_last − r_first` rise.
    min_rise: f64,
    /// Ceiling over the measured `last / first` mean-|Δ|-to-final ratio — "it converged".
    max_distance_ratio: f64,
}

/// The shared "develops from pure noise" bounds, used by every txt2img lane below.
const FROM_NOISE: (f64, f64) = (0.75, 0.30);

/// `sd3_5_large` txt2img, 12 steps at 1024², CFG 3.5 — measured r **+0.348 → +0.979**, mean |Δ| to
/// final **80.09 → 19.00** (ratio 0.237).
const LARGE: Develops = Develops {
    min_r_last: 0.949,
    max_r_first: FROM_NOISE.0,
    min_rise: FROM_NOISE.1,
    max_distance_ratio: 0.30,
};

/// `sd3_5_large_turbo` txt2img, 8 steps at 1024², distilled (no CFG) — measured r **+0.250 → +0.987**,
/// mean |Δ| to final **78.53 → 16.24** (ratio 0.207). The *highest* final resemblance of the three
/// routes and the *lowest* first frame: a distilled 8-step schedule moves further per step, so it both
/// starts further from the render and finishes closer to it. Its own numbers, not Large's.
const TURBO: Develops = Develops {
    min_r_last: 0.957,
    max_r_first: FROM_NOISE.0,
    min_rise: FROM_NOISE.1,
    max_distance_ratio: 0.27,
};

/// `sd3_5_medium` txt2img, 12 steps at 1024², CFG 3.5 — measured r **+0.459 → +0.985**, mean |Δ| to
/// final **95.59 → 12.11** (ratio 0.127, the tightest of the five lanes).
const MEDIUM: Develops = Develops {
    min_r_last: 0.955,
    max_r_first: FROM_NOISE.0,
    min_rise: FROM_NOISE.1,
    max_distance_ratio: 0.18,
};

/// `sd3_5_large` img2img / `Reference` at strength 0.6, 12 requested steps at 1024² — which forks at
/// `init_time_step(12, 0.6) = 7` and therefore emits **5** frames over the reduced σ tail. Measured
/// r **+0.954 → +0.979**, mean |Δ| to final **56.30 → 18.91** (ratio 0.336).
///
/// The only lane that does not start from noise, and the one that made the shared bounds' assumption
/// visible: its first frame resembles the render at +0.954 because the trajectory begins at a
/// VAE-encoded source. `max_r_first` and `min_rise` are therefore its own, and the claim that the
/// strip develops rests on `max_distance_ratio` plus the strict monotonicity of both series — the
/// distance to the finished image still falls by a factor of three across five frames.
const IMG2IMG: Develops = Develops {
    min_r_last: 0.949,
    max_r_first: 0.97,
    min_rise: 0.015,
    max_distance_ratio: 0.42,
};

/// `sd3_5_large` txt2img under `heun`, 8 steps at 768² — measured r **+0.377 → +0.984**, mean |Δ| to
/// final **85.36 → 19.04** (ratio 0.223), over **15** model evaluations deduped to 8 frames.
const LARGE_HEUN: Develops = Develops {
    min_r_last: 0.954,
    max_r_first: FROM_NOISE.0,
    min_rise: FROM_NOISE.1,
    max_distance_ratio: 0.28,
};

/// The shared strip analysis: numbering, latent resolution, per-frame movement, falling distance to
/// the finished image, and rising correlation with it. Applied identically to every lane so none can
/// be closed with a weaker measurement than the others.
///
/// `emitted` is the number of frames the lane is expected to produce, which is **not** always the
/// requested step count: the img2img fork hands the driver only the reduced `sigmas[start..]` tail, so
/// that lane emits `steps − init_time_step(steps, strength)`. Passing it in — derived from the
/// pipeline's own fork function — is what keeps this assertion exact on both lanes instead of being
/// loosened to accommodate one of them.
fn assert_the_strip_converges(
    label: &str,
    frames: &[PreviewFrame],
    final_image: &Image,
    emitted: u32,
    size: u32,
    develops: &Develops,
) {
    assert_eq!(
        frames
            .iter()
            .map(|f| (f.current, f.total))
            .collect::<Vec<_>>(),
        (1..=emitted).map(|n| (n, emitted)).collect::<Vec<_>>(),
        "{label}: this lane must emit exactly {emitted} frames numbered 1..={emitted}"
    );

    // The frames are VAE-latent resolution `H/8 × W/8`, which is also the proof that no unpack or
    // squeeze was needed: the running latent is `[1, 16, H/8, W/8]` and projects at exactly that size.
    let (lw, lh) = (size / 8, size / 8);
    for frame in frames {
        assert_eq!(
            (frame.image.width, frame.image.height),
            (lw, lh),
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
    let ratio = last / first;
    assert!(
        ratio < develops.max_distance_ratio,
        "{label}: the strip must converge on the final image \
         (first {first:.2} → last {last:.2}, ratio {ratio:.3}, ceiling {:.3})",
        develops.max_distance_ratio
    );
    assert!(
        distances.windows(2).all(|p| p[1] < p[0]),
        "{label}: distance to the finished image must fall at every step: {distances:?}"
    );

    // Absolute distance can only ever say "closer", never "resembles": the projection is a global
    // linear approximation of the decode (holdout R² 0.9146), so even a perfectly converged latent
    // keeps an offset and gain error against the true pixels. A hook also emits *before* each step, so
    // the last frame is one solver advancement short of the render. Correlation over a coarse
    // thumbnail — which averages the residual noise away and leaves subject placement and colour
    // masses — is what "the preview looks like the image" actually means for a decorative frame.
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
        r_last > develops.min_r_last,
        "{label}: the last preview frame must resemble the finished render \
         (r {r_last:+.3}, floor {:+.3})",
        develops.min_r_last
    );
    assert!(
        r_first < develops.max_r_first,
        "{label}: the strip must not open on something that already IS the render \
         (r {r_first:+.3}, ceiling {:+.3})",
        develops.max_r_first
    );
    assert!(
        r_last - r_first > develops.min_rise,
        "{label}: resemblance must actually develop across the strip \
         (first {r_first:+.3} → last {r_last:+.3}, rise {:+.3}, floor {:+.3})",
        r_last - r_first,
        develops.min_rise
    );
    // Monotonicity is asserted separately because no pair of endpoint bounds implies it: a strip that
    // wandered and happened to end well would satisfy every bound above.
    assert!(
        correlations.windows(2).all(|p| p[1] > p[0]),
        "{label}: resemblance must increase at every step: {correlations:?}"
    );
}

fn base_request(id: &str, steps: u32, size: u32, sampler: Option<&str>) -> GenerationRequest {
    let cfg = id != candle_gen_sd3::MODEL_ID_TURBO;
    GenerationRequest {
        prompt: PROMPT.into(),
        negative_prompt: cfg.then(|| NEGATIVE.into()),
        guidance: cfg.then_some(3.5),
        width: size,
        height: size,
        count: 1,
        seed: Some(SEED),
        steps: Some(steps),
        sampler: sampler.map(str::to_string),
        ..Default::default()
    }
}

/// Render one lane twice on one warmed generator at the same seed — once with an inert sink, once with
/// a live one — and hold the strip to [`assert_the_strip_converges`]. Returns the collected frames and
/// the number of `Progress::Step` events the live render reported, which IS its evaluation count.
#[allow(clippy::too_many_arguments)]
fn render_and_assert(
    label: &str,
    id: &str,
    var: &str,
    steps: u32,
    size: u32,
    sampler: Option<&str>,
    reference: Option<Image>,
    develops: &Develops,
) -> (Vec<PreviewFrame>, usize) {
    let root = required_path(var);
    eprintln!("── {label}: {size}² × {steps} steps, sampler {sampler:?}");

    let generator = candle_gen_sd3::provider_registry()
        .expect("sd3 registry")
        .load(id, &LoadSpec::new(WeightsSource::Dir(root)))
        .unwrap_or_else(|e| panic!("load {id}: {e}"));

    // The img2img fork hands the driver only `sigmas[start..]`, so this lane emits fewer frames than
    // it requests steps. The expectation is derived from the pipeline's OWN fork function rather than
    // restated, so the two cannot drift — and it stays an exact equality instead of being loosened to
    // a range that would accept a genuinely wrong count.
    let mut base = base_request(id, steps, size, sampler);
    let emitted = match reference {
        Some(image) => {
            base.conditioning = vec![Conditioning::Reference {
                image,
                strength: Some(IMG2IMG_STRENGTH),
            }];
            steps
                - candle_gen_sd3::pipeline::init_time_step(steps as usize, Some(IMG2IMG_STRENGTH))
                    as u32
        }
        None => steps,
    };

    // Inert first: the byte-identity baseline, on the same warmed generator.
    let inert = one_image(
        generator
            .generate(&base, &mut |_| {})
            .unwrap_or_else(|e| panic!("{label} inert-sink render: {e}")),
    );

    let (sink, frames) = collecting_sink();
    let active_request = GenerationRequest {
        preview: sink,
        ..base
    };
    let mut evaluations = 0usize;
    let active = one_image(
        generator
            .generate(&active_request, &mut |p| {
                if matches!(p, Progress::Step { .. }) {
                    evaluations += 1;
                }
            })
            .unwrap_or_else(|e| panic!("{label} active-sink render: {e}")),
    );
    assert_eq!(
        inert.pixels, active.pixels,
        "{label}: an active preview sink must not change a single output byte at the same seed"
    );

    let frames = candle_gen::lock_recover(&frames).clone();
    let name = format!("{label}_{size}_s{steps}");
    save_strip(&frames, &name);
    save_png(
        &active.pixels,
        active.width,
        active.height,
        &format!("{name}_final"),
    );
    assert_the_strip_converges(label, &frames, &active, emitted, size, develops);
    (frames, evaluations)
}

/// `sd3_5_large` — the true-CFG lane, and the route the descriptor defaults describe.
#[test]
#[ignore = "needs SD3_LARGE_SNAPSHOT + a CUDA GPU; run with --features cuda --ignored"]
fn large_preview_frames_evolve_toward_the_final_image() {
    render_and_assert(
        "sd3_5_large",
        candle_gen_sd3::MODEL_ID,
        "SD3_LARGE_SNAPSHOT",
        env_u32("SD3_PREVIEW_STEPS", 12),
        env_u32("SD3_PREVIEW_SIZE", 1024),
        None,
        None,
        &LARGE,
    );
}

/// `sd3_5_large_turbo` — the guidance-distilled few-step student, so the predict closure runs a
/// **single** MMDiT forward. Held to the same measurement on a shorter schedule.
#[test]
#[ignore = "needs SD3_TURBO_SNAPSHOT + a CUDA GPU; run with --features cuda --ignored"]
fn turbo_preview_frames_evolve_toward_the_final_image() {
    render_and_assert(
        "sd3_5_large_turbo",
        candle_gen_sd3::MODEL_ID_TURBO,
        "SD3_TURBO_SNAPSHOT",
        env_u32("SD3_PREVIEW_TURBO_STEPS", 8),
        env_u32("SD3_PREVIEW_SIZE", 1024),
        None,
        None,
        &TURBO,
    );
}

/// `sd3_5_medium` — the second true-CFG lane. A separate row rather than an assumption that it follows
/// Large: it is a different transformer entirely (24×1536 MMDiT-X with 13 dual-attention blocks), and
/// it is a shipped route.
#[test]
#[ignore = "needs SD3_MEDIUM_SNAPSHOT + a CUDA GPU; run with --features cuda --ignored"]
fn medium_preview_frames_evolve_toward_the_final_image() {
    render_and_assert(
        "sd3_5_medium",
        candle_gen_sd3::MODEL_ID_MEDIUM,
        "SD3_MEDIUM_SNAPSHOT",
        env_u32("SD3_PREVIEW_STEPS", 12),
        env_u32("SD3_PREVIEW_SIZE", 1024),
        None,
        None,
        &MEDIUM,
    );
}

/// The **second lane** every route has: img2img / `Reference`, which forks the denoise at a later σ
/// node and runs the reduced `start..` schedule tail.
///
/// It reaches the same single hooked site, but it is a genuinely different trajectory — one that
/// starts from a VAE-encoded source rather than pure noise — so it gets its own row and its own floor
/// rather than being assumed to follow txt2img. It is also where "the preview projects the target
/// only" is observable: the source is blended into `x_t` before the driver call, so from the first
/// emission there is one trajectory and it is the target's.
///
/// The reference is a txt2img render from this same generator, so the row needs no extra input and
/// cannot skip for a missing image.
#[test]
#[ignore = "needs SD3_LARGE_SNAPSHOT + a CUDA GPU; run with --features cuda --ignored"]
fn img2img_preview_frames_evolve_from_the_forked_latent() {
    let size = env_u32("SD3_PREVIEW_SIZE", 1024);
    let steps = env_u32("SD3_PREVIEW_STEPS", 12);
    let root = required_path("SD3_LARGE_SNAPSHOT");
    let generator = candle_gen_sd3::provider_registry()
        .expect("sd3 registry")
        .load(
            candle_gen_sd3::MODEL_ID,
            &LoadSpec::new(WeightsSource::Dir(root)),
        )
        .expect("load sd3_5_large");
    let source = one_image(
        generator
            .generate(
                &base_request(candle_gen_sd3::MODEL_ID, steps, size, None),
                &mut |_| {},
            )
            .expect("render the img2img source"),
    );
    save_png(
        &source.pixels,
        source.width,
        source.height,
        "img2img_source",
    );

    render_and_assert(
        "sd3_5_large_img2img",
        candle_gen_sd3::MODEL_ID,
        "SD3_LARGE_SNAPSHOT",
        steps,
        size,
        None,
        Some(source),
        &IMG2IMG,
    );
}

/// **One frame per outer step on a multi-eval solver**, proven non-vacuous first.
///
/// `heun` evaluates the model twice per outer step. The shared driver calls `on_progress` once per
/// *evaluation* (`sampler.rs` recomputes the step count on every eval and deliberately repeats it), so
/// counting `Progress::Step` events IS counting evaluations. The row asserts there are **more**
/// evaluations than outer steps before asserting the frames collapsed to exactly the outer steps — a
/// solver that turned out to evaluate once per step would make the frame-count assertion prove nothing
/// about dedup, and that is the failure this ordering rules out.
#[test]
#[ignore = "needs SD3_LARGE_SNAPSHOT + a CUDA GPU; run with --features cuda --ignored"]
fn a_multi_eval_solver_emits_one_frame_per_outer_step() {
    let steps = env_u32("SD3_PREVIEW_HEUN_STEPS", 8);
    let (frames, evaluations) = render_and_assert(
        "sd3_5_large_heun",
        candle_gen_sd3::MODEL_ID,
        "SD3_LARGE_SNAPSHOT",
        steps,
        env_u32("SD3_PREVIEW_HEUN_SIZE", 768),
        Some("heun"),
        None,
        &LARGE_HEUN,
    );
    eprintln!("  heun: {evaluations} evaluations for {steps} outer steps");
    assert!(
        evaluations > steps as usize,
        "heun must evaluate more than once per outer step or this row proves nothing about dedup \
         ({evaluations} evaluations for {steps} steps)"
    );
    // `assert_the_strip_converges` already pinned the numbering to exactly 1..=steps, so the dedup
    // collapsed the extra evaluations. Restated here because that is the point of this row.
    assert_eq!(
        frames.len(),
        steps as usize,
        "a multi-eval solver must still emit exactly one frame per outer step"
    );
}

// ── The σ convention, measured on the committed constants alone ───────────────────────────────────

/// SD3.5 needs no `input_scale` correction, and the consequence is measured rather than asserted in
/// prose.
///
/// The cheap decisive signal sc-16954 named is the first frame's rail-clipped fraction: SDXL's
/// uncorrected projection clipped 89.4% of pixels to 0/255, which is what a missing input scaling
/// looks like. Here the same measurement is taken on the latent this family's first emission actually
/// sees — `render_core`'s seeded noise is unit-normal and SD3.5's shifted σ schedule starts at
/// `σ_max = 1.0` — and it must come out readable.
///
/// Runs on the committed constants alone, no weights, and is deliberately **not** `#[ignore]`d: it is
/// the row that must appear in a plain `cargo test` of this file. sc-16954 shipped a red row that hid
/// because the only non-ignored row in its file was excluded by `-- --ignored`.
#[test]
fn the_flow_cohort_needs_no_sigma_correction() {
    use candle_gen::gen_core::sampling::{FlowModelSampling, ModelSampling, TimestepConvention};

    // The convention first: the claim is about `input_scale`, so it is read off the very
    // `ModelSampling` the driver integrates rather than asserted about the family in prose. SD3.5
    // drives `TimestepConvention::Sigma`.
    let ms = FlowModelSampling::new(TimestepConvention::Sigma);
    for sigma in [0.0f32, 0.05, 0.25, 0.5, 0.75, 1.0] {
        assert_eq!(
            ms.input_scale(sigma),
            1.0,
            "FlowModelSampling::input_scale must be identically 1.0; at {sigma} it is not, and \
             SD3.5 would need PreviewHook::with_sigma"
        );
    }

    // The schedule the pipeline actually builds must start at the σ this measurement assumes.
    let sigmas = candle_gen_sd3::pipeline::sd3_sigmas(12, 3.0);
    assert_eq!(
        sigmas[0], 1.0,
        "the SD3.5 flow schedule starts at σ_max = 1.0"
    );

    // The consequence, measured. A unit-normal `[1, 16, h, w]` latent at σ_max is what the first
    // emission sees — the same shape and distribution `render_core` seeds.
    let (lat_h, lat_w) = (32usize, 32usize);
    let mut rng = <rand::rngs::StdRng as rand::SeedableRng>::seed_from_u64(SEED);
    let noise = candle_gen::seeded_normal_vec(&mut rng, 16 * lat_h * lat_w);
    let latents = Tensor::from_vec(noise, (1, 16, lat_h, lat_w), &Device::Cpu).expect("latent");

    let frame = candle_gen_sd3::preview::project_latents(&latents)
        .expect("project the first-emission latent");
    let rails = frame.pixels.iter().filter(|&&v| v == 0 || v == 255).count() as f64
        / frame.pixels.len() as f64;
    eprintln!("  flow prior at sigma_max: rail-clipped fraction {rails:.4}");
    // The bound is loose enough that a rounding change cannot flip it and far below sc-16954's
    // uncorrected SDXL 0.894, which is the number it is being contrasted with.
    assert!(
        rails < 0.05,
        "an uncorrected flow-space projection must already be a readable noise field, not a clipped \
         one ({rails:.4}) — if this ever fails, SD3.5 needs PreviewHook::with_sigma"
    );
}
