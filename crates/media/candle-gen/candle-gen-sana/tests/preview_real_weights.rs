//! sc-16959 — candle SANA per-step latent **preview** real-weight validation (epic 16948).
//!
//! Three things a shape-only smoke cannot establish, and which this epic requires of every wiring
//! story — with a fourth this story alone has, because SANA is the only candle family here that
//! drives two sampler drivers and carries two committed fits.
//!
//! 1. **Each reused fit belongs to the route it is wired to.** SANA adds no fit; it reuses the two
//!    epic-16624 constant sets `mlx-gen-sana` measured. [`each_route_loads_the_vae_its_fit_was_measured_on`]
//!    proves each candle snapshot publishes the *byte-identical* file the corresponding MLX fit's
//!    provenance record names, and
//!    [`the_two_dc_aes_share_one_encoder_and_differ_only_in_the_decoder_tail`] measures exactly how
//!    the two autoencoders relate — which is what makes two fits necessary rather than duplicative.
//!    That second row walks tensors, because the two containers are the same architecture at exactly
//!    the same byte size, and a size comparison would say "identical" about two DC-AEs that decode
//!    differently.
//! 2. **The frames actually develop, on every shipped route.**
//!    [`base_preview_frames_evolve_toward_the_final_image`],
//!    [`base_heun_preview_frames_evolve_toward_the_final_image`] and
//!    [`sprint_preview_frames_evolve_toward_the_final_image`] render through the registered
//!    `Generator` seam with a live sink, check the numbering contract, check seeded byte-identity
//!    against an inert render, and measure that each frame is closer to the finished image than the
//!    one before it. Every strip is written out for direct review.
//! 3. **Exactly one frame per outer step on a multi-eval solver.**
//!    [`base_heun_preview_frames_evolve_toward_the_final_image`] runs `heun` and proves the guard is
//!    non-vacuous *first*: the shared driver calls `on_progress` once per **evaluation**, so counting
//!    `Progress::Step` events is counting evaluations, and the row asserts there are more of them than
//!    outer steps before it asserts the frame count collapsed to the outer steps.
//! 4. **The two routes are measured separately, never with one strip standing in for both.** A shared
//!    strip is exactly what would hide the bug this story exists to avoid, so base and Sprint each get
//!    their own snapshot, their own render, their own artifacts and their own floors — every bound
//!    below carries the number from that lane's own run.
//!
//! ```sh
//! SANA_BASE_SNAPSHOT=E:\huggingface\hub\models--Efficient-Large-Model--Sana_1600M_1024px_diffusers\snapshots\<rev> \
//! SANA_SPRINT_SNAPSHOT=E:\huggingface\hub\models--Efficient-Large-Model--Sana_Sprint_1.6B_1024px_diffusers\snapshots\<rev> \
//! SANA_PREVIEW_ARTIFACT_DIR=E:\out\sc-16959 \
//!   cargo test -p candle-gen-sana --release --features cuda --test preview_real_weights \
//!     -- --ignored --nocapture
//! ```
//!
//! Every input is **required** by the row that uses it: a row that early-returns on an unset variable
//! still reports SUCCESS, and in a run log a skipped gate is indistinguishable from one that ran and
//! proved something. Asking for `--ignored` is already the opt-in.
//!
//! [`the_two_routes_sigma_conventions_are_what_the_projectors_assume`] is the **only non-`#[ignore]`d
//! row in this file** and runs on the committed constants alone — it is the row that must appear in a
//! plain `cargo test` of this file. sc-16954 shipped a red row that hid because the only non-ignored
//! row in its file was excluded by `-- --ignored`.

use std::collections::BTreeMap;
use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex};

use candle_gen::candle_core::{Device, Tensor};
use candle_gen::gen_core::{
    GenerationOutput, GenerationRequest, Image, LoadSpec, PreviewFrame, PreviewSink, Progress,
    WeightsSource,
};
use safetensors::{Dtype, SafeTensors};

const PROMPT: &str =
    "A weathered brass astrolabe on a navigator's desk beside a cracked leather map, warm lamplight, \
     deep shadows, photographic detail.";
const NEGATIVE: &str = "blurry, lowres, artifacts, watermark, text";
const SEED: u64 = 16959;

/// The SHA-256 of the **base** `vae/diffusion_pytorch_model.safetensors` — 1,249,044,836 bytes — that
/// `Efficient-Large-Model/Sana_1600M_1024px_diffusers` @ `d1b54936033cd7d45410ecadd692c5c502a19a38`
/// publishes, and which `SceneWorks/Sana_1600M_1024px_mlx` published for the epic-16624 base fit.
const BASE_VAE_SHA256: &str = "15a4b09e56d95b768a0ec9da50b702e21d920333fc9b3480d66bb5c7fad9d87f";

/// The SHA-256 of the **Sprint** `vae/diffusion_pytorch_model.safetensors` that
/// `Efficient-Large-Model/Sana_Sprint_1.6B_1024px_diffusers` @
/// `b3c9ce6f29ad4161a00fa58a62e476b9c75ca934` publishes, and which
/// `SceneWorks/Sana_Sprint_1.6B_1024px_mlx` published for the epic-16624 Sprint fit.
const SPRINT_VAE_SHA256: &str = "dfd991d1b54ffabf22745c5885589d8f2a7bc59930d95d92bd741c4fc64454bb";

/// Both containers are exactly this size. Pinned because it is the reason the tensor walk below
/// exists: a size comparison cannot distinguish the two DC-AEs.
const VAE_BYTES: u64 = 1_249_044_836;

/// The measured extent of the base-vs-Sprint tensor walk, pinned so a partial walk cannot pass as a
/// full one.
const VAE_TENSORS: usize = 375;
const VAE_VALUES: usize = 312_250_275;
const VAE_PAYLOAD_BYTES: usize = 1_249_001_100;
/// The measured base/Sprint overlap: 320 of the 375 tensors are byte-identical and 55 differ. Pinned
/// as exact counts because **both** endpoints would falsify this story's reasoning — see
/// [`the_two_dc_aes_share_one_encoder_and_differ_only_in_the_decoder_tail`].
const VAE_IDENTICAL_TENSORS: usize = 320;
const VAE_DIFFERING_TENSORS: usize = 55;
/// All 179 encoder tensors are byte-identical, which is why this is one latent space with two
/// decoders rather than two latent spaces.
const VAE_ENCODER_TENSORS: usize = 179;

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
    env_path("SANA_PREVIEW_ARTIFACT_DIR")
        .unwrap_or_else(|| std::env::temp_dir().join("sana_preview_sc16959"))
}

// ── Provenance: each reuse is grounded in tensor bytes, per route ─────────────────────────────────

fn sha256_of(path: &Path) -> String {
    use sha2::{Digest, Sha256};
    let mut file = std::fs::File::open(path).unwrap_or_else(|e| panic!("open {path:?}: {e}"));
    let mut hasher = Sha256::new();
    std::io::copy(&mut file, &mut hasher).expect("hash the file");
    format!("{:x}", hasher.finalize())
}

/// The `vae/diffusion_pytorch_model.safetensors` under a diffusers snapshot — the file
/// `pipeline::resolve_component_files` selects and `DcAeDecoder::from_weights` is handed.
///
/// The selection is not assumed: the resolver is asked, so this row reads whatever the engine would
/// actually load rather than a filename this test believes in. Both SANA `vae/` dirs resolve to
/// exactly one file.
fn snapshot_vae(root: &Path) -> PathBuf {
    let chosen = candle_gen_sana::pipeline::resolve_component_files(&root.join("vae"))
        .unwrap_or_else(|e| panic!("resolve the vae component under {root:?}: {e}"));
    assert_eq!(
        chosen.len(),
        1,
        "a SANA vae/ dir must resolve to exactly one file, got {chosen:?}"
    );
    chosen.into_iter().next().expect("one file")
}

/// The two registered routes and the env var naming each one's snapshot.
const SNAPSHOT_VARS: [(&str, &str, &str); 2] = [
    ("sana_1600m", "SANA_BASE_SNAPSHOT", BASE_VAE_SHA256),
    (
        "sana_sprint_1600m",
        "SANA_SPRINT_SNAPSHOT",
        SPRINT_VAE_SHA256,
    ),
];

/// Each route loads the **exact file** its own reused fit was measured on.
///
/// This is the strongest form of the epic's "ground reuse in tensor bytes" requirement: not "the same
/// architecture", not "the same channel count", but the same container hash. Both routes are checked,
/// against **different** expected hashes, and the row additionally asserts the two differ from each
/// other — the property that makes two fits correct rather than redundant.
#[test]
#[ignore = "needs both SANA snapshots (set SANA_BASE_SNAPSHOT, SANA_SPRINT_SNAPSHOT)"]
fn each_route_loads_the_vae_its_fit_was_measured_on() {
    let mut seen = Vec::new();
    for (id, var, expected) in SNAPSHOT_VARS {
        let vae = snapshot_vae(&required_path(var));
        let sha = sha256_of(&vae);
        let size = std::fs::metadata(&vae).expect("stat the VAE").len();
        eprintln!("  {id:<18} {}  {sha}  {size} bytes", vae.display());
        assert_eq!(
            sha, expected,
            "{id}: the VAE this snapshot publishes is not the file its reused fit was measured on"
        );
        assert_eq!(size, VAE_BYTES, "{id}: unexpected VAE container size");
        seen.push(sha);
    }
    assert_ne!(
        seen[0], seen[1],
        "base and Sprint must publish DIFFERENT autoencoders — if they ever stop doing so, the two \
         committed fits become one question rather than two answers"
    );

    // The scaling factor is half the definition of the fitted space, and both routes build the same
    // `DcAeConfig::sana_f32c32()`, so it is pinned once here.
    assert_eq!(
        candle_gen_sana::DcAeConfig::sana_f32c32().scaling_factor,
        0.41407
    );
}

/// **The two-fits gate**, and the measurement that says *why* two fits exist.
///
/// This epic's recurring surprise is that container size, key set and channel count all fail to
/// distinguish autoencoders: sc-16956 found Boogu's "16 channels" to be FLUX.1's, sc-16957 found
/// Z-Image's to be literally the same file, sc-16958 found SD3.5's to be a wholly different VAE at an
/// identical size. SANA is a **fourth** case none of those would have predicted, and it is only
/// visible by walking tensors: the two DC-AEs **partially overlap**.
///
/// Measured, over the same 1,249,044,836-byte container size and the same 375 keys, shapes and dtype:
///
/// * **320 of 375 tensors are byte-identical**, including the **entire encoder** — all 179 of its
///   tensors — plus `decoder.conv_in` and the whole of `decoder.up_blocks.3`, the deepest decoder
///   stage;
/// * **55 differ, and every one of them is in the `decoder.` subtree** — `up_blocks.0`, `up_blocks.1`,
///   `up_blocks.2`, `norm_out` and `conv_out`: the last three upsampling stages and the output head.
///
/// So DC-AE 1.1 (Sprint) is a **decoder-tail fine-tune** of DC-AE 1.0 (base). The encoder is what
/// defines the latent space, and it is unchanged — but an RGB preview fit is a least-squares map from
/// a latent to that autoencoder's **decoded pixels**, and the decode is exactly what was retrained.
/// One fit therefore cannot serve both routes, and the reason is sharper than "two latent spaces":
/// it is one latent space with two decoders.
///
/// The row asserts all of that by exact count and by subtree, so neither a wholesale divergence nor a
/// collapse to one file could pass. The comparison is over the tensors' **raw payload bytes**, read
/// straight out of the two containers rather than through candle: widening to `Vec<f32>` and comparing
/// values would report a genuinely identical pair as *differing* if either held a NaN, which is the
/// one way an overlap count could come out wrong.
#[test]
#[ignore = "needs both SANA snapshots (set SANA_BASE_SNAPSHOT, SANA_SPRINT_SNAPSHOT)"]
fn the_two_dc_aes_share_one_encoder_and_differ_only_in_the_decoder_tail() {
    let base_vae = snapshot_vae(&required_path("SANA_BASE_SNAPSHOT"));
    let sprint_vae = snapshot_vae(&required_path("SANA_SPRINT_SNAPSHOT"));

    // Re-pinned here rather than borrowed from the row above: this row must establish for itself that
    // the files it compared are the two it names.
    let (base_sha, sprint_sha) = (sha256_of(&base_vae), sha256_of(&sprint_vae));
    eprintln!("  base   vae/  {base_sha}");
    eprintln!("  sprint vae/  {sprint_sha}");
    assert_eq!(base_sha, BASE_VAE_SHA256, "not the base SANA DC-AE");
    assert_eq!(sprint_sha, SPRINT_VAE_SHA256, "not the Sprint DC-AE");
    assert_eq!(
        std::fs::metadata(&base_vae).expect("stat").len(),
        std::fs::metadata(&sprint_vae).expect("stat").len(),
        "the two containers are deliberately the same size — if that ever stops being true, the \
         tensor walk below is no longer the only way to tell them apart and this row's reasoning \
         should be revisited rather than silently kept"
    );

    let (base_bytes, sprint_bytes) = (
        std::fs::read(&base_vae).unwrap_or_else(|e| panic!("read {base_vae:?}: {e}")),
        std::fs::read(&sprint_vae).unwrap_or_else(|e| panic!("read {sprint_vae:?}: {e}")),
    );
    let parse = |name: &str, raw: &[u8]| -> BTreeMap<String, (Vec<usize>, Dtype, Vec<u8>)> {
        SafeTensors::deserialize(raw)
            .unwrap_or_else(|e| panic!("parse the {name} container: {e}"))
            .tensors()
            .into_iter()
            .map(|(key, view)| {
                (
                    key,
                    (view.shape().to_vec(), view.dtype(), view.data().to_vec()),
                )
            })
            .collect()
    };
    let base = parse("base", &base_bytes);
    let sprint = parse("Sprint", &sprint_bytes);
    assert_eq!(
        base.keys().collect::<Vec<_>>(),
        sprint.keys().collect::<Vec<_>>(),
        "the two DC-AEs must have the same key set — that shared architecture is the whole reason a \
         channel count could not settle this"
    );

    let (mut values, mut payload) = (0usize, 0usize);
    let (mut identical, mut differing): (Vec<&str>, Vec<&str>) = (Vec::new(), Vec::new());
    for (key, (shape, dtype, base_payload)) in &base {
        let (sprint_shape, sprint_dtype, sprint_payload) = &sprint[key];
        assert_eq!(shape, sprint_shape, "{key}: shapes");
        assert_eq!(dtype, sprint_dtype, "{key}: dtypes");
        assert_eq!(
            base_payload.len(),
            sprint_payload.len(),
            "{key}: payload sizes"
        );
        if base_payload == sprint_payload {
            identical.push(key.as_str());
        } else {
            differing.push(key.as_str());
            eprintln!("  differs: {key} {shape:?}");
        }
        values += shape.iter().product::<usize>();
        payload += base_payload.len();
    }
    eprintln!(
        "  walked {} tensors / {values} values: {} identical, {} differing",
        base.len(),
        identical.len(),
        differing.len()
    );

    // The walk covered both containers whole.
    assert_eq!(base.len(), VAE_TENSORS);
    assert_eq!(values, VAE_VALUES);
    assert_eq!(
        payload, VAE_PAYLOAD_BYTES,
        "the walk must cover the whole tensor region of both containers"
    );

    // The overlap, by exact count. Neither endpoint is acceptable: `differing == 0` would mean one fit
    // could serve both routes, and `differing == VAE_TENSORS` would contradict the shared-encoder
    // finding this story records.
    assert_eq!(
        (identical.len(), differing.len()),
        (VAE_IDENTICAL_TENSORS, VAE_DIFFERING_TENSORS),
        "the base/Sprint DC-AE overlap has moved; re-derive the two-fits reasoning rather than \
         updating these numbers on their own"
    );

    // And where the difference lives: entirely in the decoder tail. This is the load-bearing half —
    // it is what turns "the files differ" into "the decode differs, so the fits must".
    let encoder_total = base
        .keys()
        .filter(|key| key.starts_with("encoder."))
        .count();
    assert_eq!(encoder_total, VAE_ENCODER_TENSORS);
    assert!(
        differing.iter().all(|key| key.starts_with("decoder.")),
        "every differing tensor must be decoder-side — a differing ENCODER tensor would mean the two \
         routes' latent spaces are genuinely different, which is a stronger claim than this story \
         makes: {:?}",
        differing
            .iter()
            .filter(|key| !key.starts_with("decoder."))
            .collect::<Vec<_>>()
    );
    assert_eq!(
        identical
            .iter()
            .filter(|key| key.starts_with("encoder."))
            .count(),
        encoder_total,
        "the whole encoder must be byte-identical — that is what makes this one latent space with \
         two decoders rather than two latent spaces"
    );

    // The retrained region, named rather than merely counted, so a fine-tune that moved elsewhere in
    // the decoder is a diff here.
    let mut parts: Vec<&str> = differing
        .iter()
        .map(|key| {
            key.strip_prefix("decoder.")
                .and_then(|rest| rest.split('.').next())
                .expect("a decoder key")
        })
        .collect();
    parts.sort_unstable();
    parts.dedup();
    assert_eq!(
        parts,
        vec!["conv_out", "norm_out", "up_blocks"],
        "the Sprint fine-tune touches the decoder's output head and upsampling stages"
    );
    let mut stages: Vec<&str> = differing
        .iter()
        .filter_map(|key| key.strip_prefix("decoder.up_blocks."))
        .map(|rest| rest.split('.').next().expect("a stage index"))
        .collect();
    stages.sort_unstable();
    stages.dedup();
    assert_eq!(
        stages,
        vec!["0", "1", "2"],
        "only the last three upsampling stages differ; up_blocks.3 — the stage closest to the latent \
         — is byte-identical, as is decoder.conv_in"
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
/// asks to be reviewed directly. One strip **per route**: a shared strip is exactly what would hide
/// the mistake this story exists to avoid.
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
/// Every bound below is derived from that exact lane's own run, and the headroom is uniform and
/// stated: **0.03 under a measured correlation, 0.06 under a measured rise** (a rise differences two
/// correlations, so it carries the 0.03 allowance of each), **0.06 over a measured distance ratio.**
/// No bound is justified by a neighbouring lane's measurement, and there is no unexplained slack —
/// this is the story where a base bound carried over onto Sprint would be the very confusion it
/// exists to avoid.
///
/// `min_rise` used to be the one exception: a single shared `0.30` for all three lanes. It was never
/// unsound — every lane clears it — but it hid how differently the three lanes are placed against it.
/// The `heun` lane rises **+0.360**, against **+0.481** on Euler and **+0.695** on Sprint, so one
/// shared floor gives the three of them 0.06, 0.181 and 0.395 of margin respectively, none of it
/// stated. Each lane now derives its own from its own rise, so the margin is a uniform 0.06 across
/// all three, and the number that used to be shared falls out as `heun`'s: **0.360 − 0.06 = 0.300**.
/// `heun`'s rise is the shallowest of the three by construction — a second-order solver's first frame
/// is already further along — and that is now visible in its bound rather than being an accident of a
/// shared constant. The change is strictly tightening: Euler 0.30 → 0.421, Sprint 0.30 → 0.635,
/// `heun` unchanged.
///
/// `max_r_first` genuinely is shared, and stays shared: both measured preview lanes are txt2img cases,
/// so every captured strip starts at the flow / SCM prior. It is deliberately loose because a tight
/// `r_first` bound would read a fit's own warm intercept as if it were resemblance.
struct Develops {
    /// Floor under the measured final-frame correlation with the finished render.
    min_r_last: f64,
    /// Ceiling over the measured first-frame correlation — "it did not start as the render".
    max_r_first: f64,
    /// Floor under the measured `r_last − r_first` rise, **per lane**.
    min_rise: f64,
    /// Ceiling over the measured `last / first` mean-|Δ|-to-final ratio — "it converged".
    max_distance_ratio: f64,
}

/// The shared "did not start as the render" ceiling for these two measured txt2img preview cases.
const MAX_R_FIRST: f64 = 0.75;

/// `sana_1600m` txt2img, 12 steps at 1024², true CFG 4.5, native flow-Euler over the static shift-3.0
/// schedule — measured r **+0.477 → +0.958** (rise +0.481), mean |Δ| to final **49.76 → 19.81**
/// (ratio 0.398).
const BASE: Develops = Develops {
    // 0.958 − 0.03.
    min_r_last: 0.928,
    max_r_first: MAX_R_FIRST,
    // 0.958 − 0.477 = 0.481; 0.481 − 0.06 = 0.421.
    min_rise: 0.421,
    // 0.398 + 0.06 = 0.458, rounded up to two decimals.
    max_distance_ratio: 0.46,
};

/// `sana_1600m` txt2img under `heun`, 8 steps at 512² — measured r **+0.578 → +0.938** (rise +0.360,
/// the shallowest of the three), mean |Δ| to final **48.20 → 21.54** (ratio 0.447), over **15** model
/// evaluations deduped to 8 frames.
///
/// Its own numbers, not the Euler lane's: `heun`'s second-order step lands the strip in a different
/// place at both ends, and it runs at a different resolution. The shallow rise is that same fact read
/// from the other end — its *first* frame is already at +0.578, well ahead of Euler's +0.477, so it
/// has less distance left to travel even though it finishes lower.
const BASE_HEUN: Develops = Develops {
    // 0.938 − 0.03.
    min_r_last: 0.908,
    max_r_first: MAX_R_FIRST,
    // 0.938 − 0.578 = 0.360; 0.360 − 0.06 = 0.300.
    min_rise: 0.300,
    // 0.447 + 0.06 = 0.507, rounded up to two decimals.
    max_distance_ratio: 0.51,
};

/// `sana_sprint_1600m` txt2img, 4 SCM steps at 1024², embedded guidance 4.5 — measured
/// r **+0.254 → +0.949** (rise +0.695, the steepest of the three), mean |Δ| to final
/// **61.64 → 14.42** (ratio 0.234, the tightest of the three lanes).
///
/// Its own numbers, over its own decoder and its own fit; nothing here is inherited from the base
/// lane. The shape is what a four-step consistency schedule should look like beside a twelve-step
/// flow one — it starts *further* from the render (r +0.254 against the base lane's +0.477, because
/// each SCM step moves much further) and finishes closer in absolute distance (14.42 against 19.81).
const SPRINT: Develops = Develops {
    // 0.949 − 0.03.
    min_r_last: 0.919,
    max_r_first: MAX_R_FIRST,
    // 0.949 − 0.254 = 0.695; 0.695 − 0.06 = 0.635.
    min_rise: 0.635,
    // 0.234 + 0.06 = 0.294, rounded up to two decimals.
    max_distance_ratio: 0.30,
};

/// The shared strip analysis: numbering, latent resolution, per-frame movement, falling distance to
/// the finished image, and rising correlation with it. Applied identically to both routes so neither
/// can be closed with a weaker measurement than the other.
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

    // The frames are DC-AE latent resolution `H/32 × W/32`, which is also the proof that no unpack or
    // squeeze was needed: the running latent is `[1, 32, H/32, W/32]` and projects at exactly that
    // size. The divisor is the engine's own `SPATIAL_SCALE`, not a number restated here.
    let scale = candle_gen_sana::pipeline::SPATIAL_SCALE;
    let (lw, lh) = (size / scale, size / scale);
    for frame in frames {
        assert_eq!(
            (frame.image.width, frame.image.height),
            (lw, lh),
            "{label}: frames must be DC-AE latent resolution"
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
    // linear approximation of the decode, so even a perfectly converged latent keeps an offset and
    // gain error against the true pixels. A hook also emits *before* each step, so the last frame is
    // one solver advancement short of the render. Correlation over a coarse thumbnail — which averages
    // the residual noise away and leaves subject placement and colour masses — is what "the preview
    // looks like the image" actually means for a decorative frame.
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

/// A request for one route. Sprint is CFG-free — it takes no negative prompt and advertises no
/// curated sampler — so its request genuinely differs rather than being the base one with a flag off.
fn base_request(id: &str, steps: u32, size: u32, sampler: Option<&str>) -> GenerationRequest {
    let sprint = id == candle_gen_sana::SPRINT_MODEL_ID;
    GenerationRequest {
        prompt: PROMPT.into(),
        negative_prompt: (!sprint).then(|| NEGATIVE.into()),
        guidance: Some(4.5),
        width: size,
        height: size,
        count: 1,
        seed: Some(SEED),
        steps: Some(steps),
        sampler: sampler.map(str::to_string),
        ..Default::default()
    }
}

/// Render one route twice on one warmed generator at the same seed — once with an inert sink, once
/// with a live one — and hold the strip to [`assert_the_strip_converges`]. Returns the collected
/// frames and the number of `Progress::Step` events the live render reported, which IS its evaluation
/// count.
///
/// `non_vacuity` runs on that evaluation count **before any other assertion in this function**, which
/// is what lets the `heun` row establish that its solver really does evaluate more than once per outer
/// step ahead of the frame numbering that fact is what makes meaningful. Rows with nothing to
/// establish first pass `&|_| {}`.
#[allow(clippy::too_many_arguments)]
fn render_and_assert(
    label: &str,
    id: &str,
    var: &str,
    steps: u32,
    size: u32,
    sampler: Option<&str>,
    develops: &Develops,
    non_vacuity: &dyn Fn(usize),
) -> (Vec<PreviewFrame>, usize) {
    let root = required_path(var);
    eprintln!("── {label}: {size}² × {steps} steps, sampler {sampler:?}");

    let generator = candle_gen_sana::provider_registry()
        .expect("sana registry")
        .load(id, &LoadSpec::new(WeightsSource::Dir(root)))
        .unwrap_or_else(|e| panic!("load {id}: {e}"));

    let base = base_request(id, steps, size, sampler);

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

    // First, ahead of every other assertion here — see this function's docs.
    non_vacuity(evaluations);

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
    assert_the_strip_converges(label, &frames, &active, steps, size, develops);
    (frames, evaluations)
}

/// `sana_1600m` — the true-CFG flow-match lane, through `run_flow_sampler`.
#[test]
#[ignore = "needs SANA_BASE_SNAPSHOT + a CUDA GPU; run with --features cuda --ignored"]
fn base_preview_frames_evolve_toward_the_final_image() {
    render_and_assert(
        "sana_1600m",
        candle_gen_sana::MODEL_ID,
        "SANA_BASE_SNAPSHOT",
        env_u32("SANA_PREVIEW_STEPS", 12),
        env_u32("SANA_PREVIEW_SIZE", 1024),
        None,
        &BASE,
        &|_| {},
    );
}

/// **One frame per outer step on a multi-eval solver**, proven non-vacuous first.
///
/// `heun` evaluates the model twice per outer step. The shared driver calls `on_progress` once per
/// *evaluation* and deliberately repeats the step number, so counting `Progress::Step` events IS
/// counting evaluations. The row asserts there are **more** evaluations than outer steps before
/// asserting the frames collapsed to exactly the outer steps.
///
/// Only the base route has this axis: Sprint advertises the `"default"` sentinel alone, because the
/// SCM consistency loop is not a curated `Solver`. Sprint's equivalent risk — a short schedule — is
/// covered by its own row's 4-step and by the 1-step rows in `tests/preview_wiring.rs`.
#[test]
#[ignore = "needs SANA_BASE_SNAPSHOT + a CUDA GPU; run with --features cuda --ignored"]
fn base_heun_preview_frames_evolve_toward_the_final_image() {
    let steps = env_u32("SANA_PREVIEW_HEUN_STEPS", 8);
    let (frames, evaluations) = render_and_assert(
        "sana_1600m_heun",
        candle_gen_sana::MODEL_ID,
        "SANA_BASE_SNAPSHOT",
        steps,
        env_u32("SANA_PREVIEW_HEUN_SIZE", 512),
        Some("heun"),
        &BASE_HEUN,
        &|evaluations| {
            eprintln!("  heun: {evaluations} evaluations for {steps} outer steps");
            assert!(
                evaluations > steps as usize,
                "heun must evaluate more than once per outer step or this row proves nothing about \
                 dedup ({evaluations} evaluations for {steps} steps)"
            );
        },
    );
    assert_eq!(
        frames.len(),
        steps as usize,
        "a multi-eval solver must still emit exactly one frame per outer step ({evaluations} \
         evaluations)"
    );
}

/// `sana_sprint_1600m` — the CFG-free SCM / TrigFlow lane, through `run_scm_sampler`.
///
/// A separate route with a separate snapshot, a separate autoencoder and a separate fit, so it gets
/// its own render and its own artifacts rather than being assumed to follow the base lane. It is also
/// the only lane in this epic whose projector applies a `1/σ_data` correction, and the strip is where
/// that shows: an uncorrected Sprint preview would project half the latent the fit was measured on.
#[test]
#[ignore = "needs SANA_SPRINT_SNAPSHOT + a CUDA GPU; run with --features cuda --ignored"]
fn sprint_preview_frames_evolve_toward_the_final_image() {
    render_and_assert(
        "sana_sprint_1600m",
        candle_gen_sana::SPRINT_MODEL_ID,
        "SANA_SPRINT_SNAPSHOT",
        env_u32("SANA_PREVIEW_SPRINT_STEPS", 4),
        env_u32("SANA_PREVIEW_SIZE", 1024),
        None,
        &SPRINT,
        &|_| {},
    );
}

// ── The σ conventions, measured on the committed constants alone ──────────────────────────────────

/// The two routes' σ conventions differ, and each projector matches its own — measured rather than
/// asserted in prose.
///
/// The cheap decisive signal sc-16954 named is the first frame's rail-clipped fraction: SDXL's
/// uncorrected projection clipped 89.4% of pixels to 0/255, which is what a missing input scaling
/// looks like. Here the same measurement is taken on the latent each route's first emission actually
/// sees, and both must come out readable:
///
/// * **base** — `run_flow_sampler` integrates a `FlowModelSampling` whose `input_scale` is identically
///   1.0, so the first emission is the unit-normal seed latent at `σ_max = 1.0`, unscaled.
/// * **Sprint** — `run_scm_sampler` hands the hook `x · σ_data`, and the projector multiplies by
///   `1/σ_data` first. The rail fraction is measured on the corrected projection, which is what
///   ships.
///
/// The uncorrected Sprint fraction is printed alongside for contrast, because on this family the
/// missing correction does **not** show up as clipping: `σ_data = 0.5` shrinks the latent rather than
/// growing it, so an uncorrected Sprint frame collapses toward the fit's intercept instead of toward
/// the rails. The statistic that discriminates there is contrast, and it is asserted too.
///
/// Runs on the committed constants alone, no weights, and is deliberately **not** `#[ignore]`d.
#[test]
fn the_two_routes_sigma_conventions_are_what_the_projectors_assume() {
    use candle_gen::gen_core::sampling::{FlowModelSampling, ModelSampling, TimestepConvention};

    // The base convention first: the claim is about `input_scale`, so it is read off the very
    // `ModelSampling` the driver integrates rather than asserted about the family in prose.
    let ms = FlowModelSampling::new(TimestepConvention::Sigma);
    for sigma in [0.0f32, 0.05, 0.25, 0.5, 0.75, 1.0] {
        assert_eq!(
            ms.input_scale(sigma),
            1.0,
            "FlowModelSampling::input_scale must be identically 1.0; at {sigma} it is not, and base \
             SANA would need PreviewHook::with_sigma"
        );
    }
    // The schedule the pipeline actually builds must start at the σ this measurement assumes.
    assert_eq!(
        candle_gen_sana::pipeline::sana_sigmas(None, 12)[0],
        1.0,
        "the SANA flow schedule starts at σ_max = 1.0"
    );

    // The consequence, measured. A unit-normal `[1, 32, h, w]` latent is what the base route's first
    // emission sees — the same shape and distribution `pipeline::create_noise` seeds.
    let (lat_h, lat_w) = (32usize, 32usize);
    let mut rng = <rand::rngs::StdRng as rand::SeedableRng>::seed_from_u64(SEED);
    let noise = candle_gen::seeded_normal_vec(&mut rng, 32 * lat_h * lat_w);
    let prior =
        Tensor::from_vec(noise, (1, 32, lat_h, lat_w), &Device::Cpu).expect("the flow/SCM prior");

    let rails = |image: &Image| -> f64 {
        image.pixels.iter().filter(|&&v| v == 0 || v == 255).count() as f64
            / image.pixels.len() as f64
    };
    let spread = |image: &Image, flat: &Image| -> f64 {
        image
            .pixels
            .iter()
            .zip(&flat.pixels)
            .map(|(&v, &g)| (v as i32 - g as i32).unsigned_abs() as f64)
            .sum::<f64>()
            / image.pixels.len() as f64
    };

    let base_frame =
        candle_gen_sana::preview::project_base_latents(&prior).expect("project the base prior");
    eprintln!(
        "  base   flow prior at sigma_max: rail-clipped fraction {:.4}",
        rails(&base_frame)
    );
    assert!(
        rails(&base_frame) < 0.05,
        "an uncorrected flow-space projection must already be a readable noise field, not a clipped \
         one ({:.4}) — if this ever fails, base SANA needs PreviewHook::with_sigma",
        rails(&base_frame)
    );

    // Sprint: the driver hands the hook `x · σ_data`, and the shipped projector corrects it.
    let scm_running = prior
        .affine(candle_gen::SCM_SIGMA_DATA as f64, 0.0)
        .expect("the SCM loop's sigma_data pre-scale");
    let sprint_frame = candle_gen_sana::preview::project_sprint_latents(
        &scm_running,
        candle_gen_sana::preview::SPRINT_INVERSE_SIGMA_DATA,
    )
    .expect("project the corrected SCM prior");
    let sprint_uncorrected = candle_gen_sana::preview::project_sprint_latents(&scm_running, 1.0)
        .expect("project the uncorrected SCM prior");
    eprintln!(
        "  sprint SCM prior (corrected):   rail-clipped fraction {:.4}",
        rails(&sprint_frame)
    );
    eprintln!(
        "  sprint SCM prior (uncorrected): rail-clipped fraction {:.4}",
        rails(&sprint_uncorrected)
    );
    assert!(
        rails(&sprint_frame) < 0.05,
        "the corrected SCM projection must be a readable noise field, not a clipped one ({:.4})",
        rails(&sprint_frame)
    );

    // On this family the missing correction shows as lost contrast rather than clipping, so that is
    // what is asserted: `σ_data = 0.5` shrinks the latent, collapsing the frame toward the intercept.
    let flat = candle_gen_sana::preview::project_sprint_latents(
        &prior.zeros_like().expect("zero latent"),
        1.0,
    )
    .expect("the intercept frame");
    let (corrected, uncorrected) = (
        spread(&sprint_frame, &flat),
        spread(&sprint_uncorrected, &flat),
    );
    eprintln!("  sprint spread about the intercept: corrected {corrected:.2}, uncorrected {uncorrected:.2}");
    assert!(
        corrected > uncorrected * 1.5,
        "an uncorrected Sprint preview must be visibly flatter than the corrected one it ships \
         (corrected {corrected:.2}, uncorrected {uncorrected:.2})"
    );
}
