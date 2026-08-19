//! sc-10976 (epic 10975 — MLX **video**-lane sequential component residency): the residency proof for
//! LTX-2.3 on real weights. LTX now stages the ~24 GB Gemma-3-12B text encoder load → `encode_av` →
//! **drop + `clear_cache()`** BEFORE the AvDiT materializes (`mlx-gen-ltx/src/model.rs`), mirroring
//! Wan's `encode_text_staged`. The provider source owns that phase-order guarantee. This harness
//! guards it numerically at the historical small-q4 baseline and records capture-only measurements at
//! larger/tier-varied geometries where activation growth makes the component-only bound inapplicable.
//!
//! **Why this is NOT the image-lane `OffloadPolicy::Resident`↔`Sequential` A/B.** Per epic 10975 the
//! video lane stages **unconditionally** (Wan-style: always load → use → drop, no `offload_policy` /
//! fit-gate branch — video is slow enough that a cross-job warm cache is worth ~nothing against the
//! encoder's memory pressure). So there is no production "Resident" mode to flip. Instead we bound the
//! staged `generate` peak below a **co-residence estimate** = (measured TE resident peak) + (the AvDiT's
//! on-disk `transformer.safetensors` bytes). The pre-sc-10976 `load()` held BOTH giants resident for the
//! whole job, so its peak was ≥ that estimate; the staged path holds at most one giant at a time.
//! Only the default q4/no-audio `256×256×9` row asserts this component-only comparison. Other rows
//! still run the genuine production `generate`, report the comparison for context, and validate their
//! output, but are capture-only: they make no residency-saving or fit/working-set claim because their
//! denoise/decode activations can exceed the component-only estimate. Quantizing Gemma is an
//! orthogonal lever for the text-phase floor and is not inferred from this probe.
//!
//! **Output correctness** is covered by the existing parity gates (`te_parity`, `pipeline_parity`,
//! `i2v_parity`, `s0_parity`): staging changes only WHEN each component is built/freed, not the encode /
//! denoise / decode math, so those bit-exact/parity tests remain the authority. This test owns the
//! q4-baseline MEMORY regression gate + a non-degenerate-output sanity check for every capture.
//!
//! `#[ignore]`d — needs the real snapshot. Defaults to the HF cache `SceneWorks/ltx-2.3-mlx` (model dir
//! = its `q4` subdir; Gemma = its bundled `gemma/`); override with `LTX_MODEL_DIR` / `LTX_GEMMA_DIR`.
//! Geometry, transformer tier, FPS, and video mode are process inputs so every capture is reproducible
//! without editing this file:
//!
//! ```text
//! # Pin the exact immutable snapshot paths used for a published capture. The historical default is
//! # q4, 256x256x9, 24 fps, with audio decode skipped.
//! LTX_MODEL_DIR=/path/to/models--SceneWorks--ltx-2.3-mlx/snapshots/01df27d308466533aa09d251e3aebdcc627d07eb/q4 \
//! LTX_GEMMA_DIR=/path/to/models--SceneWorks--ltx-2.3-mlx/snapshots/01df27d308466533aa09d251e3aebdcc627d07eb/gemma \
//!   cargo test -p mlx-gen-ltx --release --test sequential_residency_real_weights -- --ignored --nocapture
//!
//! # Reproduce the historical maximum-envelope bf16/no-audio row.
//! LTX_MODEL_DIR=/path/to/models--SceneWorks--ltx-2.3-mlx/snapshots/01df27d308466533aa09d251e3aebdcc627d07eb/bf16 \
//! LTX_GEMMA_DIR=/path/to/models--SceneWorks--ltx-2.3-mlx/snapshots/01df27d308466533aa09d251e3aebdcc627d07eb/gemma \
//! LTX_TIER=bf16 LTX_W=1280 LTX_H=704 LTX_FRAMES=449 LTX_FPS=30 \
//!   cargo test -p mlx-gen-ltx --release --test sequential_residency_real_weights -- --ignored --nocapture
//!
//! # Measure the production full-A/V route instead (`default` means no `video_mode` override).
//! LTX_MODEL_DIR=/path/to/models--SceneWorks--ltx-2.3-mlx/snapshots/01df27d308466533aa09d251e3aebdcc627d07eb/q8 \
//! LTX_GEMMA_DIR=/path/to/models--SceneWorks--ltx-2.3-mlx/snapshots/01df27d308466533aa09d251e3aebdcc627d07eb/gemma \
//! LTX_TIER=q8 LTX_W=768 LTX_H=512 LTX_FRAMES=145 LTX_FPS=24 LTX_VIDEO_MODE=default \
//!   cargo test -p mlx-gen-ltx --release --test sequential_residency_real_weights -- --ignored --nocapture
//! ```
//!
//! For an exploratory local run only, `MLX_GEN_MODELS_ROOT` may point at a `models--*/snapshots`
//! root; the harness resolves `refs/main` first and otherwise chooses a tier-bearing snapshot
//! deterministically. Published rows must use explicit paths so their revision identity is not
//! implicit in mutable cache state. Any missing explicit path is a hard failure, never a skipped test.
//!
//! `get_peak_memory()` is MLX's peak **active** allocation only. It excludes the allocator cache, while
//! macOS `recommendedMaxWorkingSetSize` constrains the process footprint containing both. The probe
//! therefore also prints `get_cache_memory()` sampled at the end of the staged-generate bracket. The
//! two observations are intentionally reported separately: peak-active and end-of-bracket cache are
//! not guaranteed to be simultaneous and must not be presented as an exact summed peak.

use mlx_gen::weights::Weights;
use mlx_gen::{GenerationOutput, GenerationRequest, Image, LoadSpec, Quant, WeightsSource};
use mlx_gen_ltx::config::SplitModel;
use mlx_gen_ltx::gemma::GemmaConfig;
use mlx_gen_ltx::{LtxConfig, LtxTextEncoder, LtxTokenizer};
use mlx_rs::memory::{clear_cache, get_cache_memory, get_peak_memory, reset_peak_memory};
use mlx_rs::Dtype;
use std::path::{Path, PathBuf};

const GIB: f64 = 1024.0 * 1024.0 * 1024.0;

const DEFAULT_WIDTH: u32 = 256;
const DEFAULT_HEIGHT: u32 = 256;
const DEFAULT_FRAMES: u32 = 9;
const DEFAULT_FPS: u32 = 24;
const CAPTURE_ENV_PROBE: &str = "LTX_CAPTURE_ENV_PROBE";

static CAPTURE_ENV_LOCK: std::sync::Mutex<()> = std::sync::Mutex::new(());

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Tier {
    Q4,
    Q8,
    Bf16,
}

impl Tier {
    fn parse(raw: &str) -> Self {
        match raw.trim().to_ascii_lowercase().as_str() {
            "q4" => Self::Q4,
            "q8" => Self::Q8,
            "bf16" => Self::Bf16,
            other => panic!("LTX_TIER must be q4, q8, or bf16 (got {other:?})"),
        }
    }

    fn from_model_dir(model: &std::path::Path) -> Self {
        let split =
            SplitModel::from_model_dir(model).expect("read split_model.json for capture tier");
        match (split.quantized, split.bits) {
            (false, _) => Self::Bf16,
            (true, 4) => Self::Q4,
            (true, 8) => Self::Q8,
            (true, bits) => panic!(
                "unsupported quantized LTX capture tier in {}: split_model.json bits={bits}",
                model.display()
            ),
        }
    }

    fn as_str(self) -> &'static str {
        match self {
            Self::Q4 => "q4",
            Self::Q8 => "q8",
            Self::Bf16 => "bf16",
        }
    }

    fn load_quant(self) -> Option<Quant> {
        match self {
            Self::Q4 => Some(Quant::Q4),
            Self::Q8 => Some(Quant::Q8),
            Self::Bf16 => None,
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum VideoMode {
    Default,
    NoAudio,
    VideoOnly,
}

impl VideoMode {
    fn parse(raw: &str) -> Self {
        match raw.trim().to_ascii_lowercase().as_str() {
            "default" => Self::Default,
            "no_audio" => Self::NoAudio,
            "video_only" => Self::VideoOnly,
            other => {
                panic!("LTX_VIDEO_MODE must be default, no_audio, or video_only (got {other:?})")
            }
        }
    }

    fn request_value(self) -> Option<String> {
        match self {
            Self::Default => None,
            Self::NoAudio => Some("no_audio".into()),
            Self::VideoOnly => Some("video_only".into()),
        }
    }

    fn as_str(self) -> &'static str {
        match self {
            Self::Default => "default",
            Self::NoAudio => "no_audio",
            Self::VideoOnly => "video_only",
        }
    }

    fn expects_audio(self) -> bool {
        matches!(self, Self::Default)
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct CaptureConfig {
    width: u32,
    height: u32,
    frames: u32,
    fps: u32,
    requested_tier: Option<Tier>,
    video_mode: VideoMode,
}

impl CaptureConfig {
    fn from_env() -> Self {
        // The ignored real-weight test and the env-contract test can share this binary under
        // `--include-ignored`. Take one process-local snapshot under the same lock so a future
        // in-process env fixture cannot splice values from two configurations. The current fixture
        // runs in a subprocess (see below), which also isolates mutation from the expensive capture.
        let _lock = CAPTURE_ENV_LOCK
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        Self::from_lookup(|name| std::env::var(name).ok())
    }

    fn from_lookup(mut lookup: impl FnMut(&str) -> Option<String>) -> Self {
        let config = Self {
            width: lookup_u32(&mut lookup, "LTX_W", DEFAULT_WIDTH),
            height: lookup_u32(&mut lookup, "LTX_H", DEFAULT_HEIGHT),
            frames: lookup_u32(&mut lookup, "LTX_FRAMES", DEFAULT_FRAMES),
            fps: lookup_u32(&mut lookup, "LTX_FPS", DEFAULT_FPS),
            requested_tier: lookup("LTX_TIER").map(|raw| Tier::parse(&raw)),
            video_mode: lookup("LTX_VIDEO_MODE")
                .map(|raw| VideoMode::parse(&raw))
                .unwrap_or(VideoMode::NoAudio),
        };
        config.validate();
        config
    }

    fn validate(self) {
        assert!(self.width > 0, "LTX_W must be greater than zero");
        assert!(self.height > 0, "LTX_H must be greater than zero");
        assert!(self.frames > 0, "LTX_FRAMES must be greater than zero");
        assert!(self.fps > 0, "LTX_FPS must be greater than zero");
        assert_eq!(
            self.width % mlx_gen_ltx::SIZE_MULTIPLE,
            0,
            "LTX_W must be a multiple of {}",
            mlx_gen_ltx::SIZE_MULTIPLE
        );
        assert_eq!(
            self.height % mlx_gen_ltx::SIZE_MULTIPLE,
            0,
            "LTX_H must be a multiple of {}",
            mlx_gen_ltx::SIZE_MULTIPLE
        );
        assert_eq!((self.frames - 1) % 8, 0, "LTX_FRAMES must be 1 + 8*k");
    }

    fn request(self) -> GenerationRequest {
        GenerationRequest {
            prompt: "a red fox trotting through a snowy forest, cinematic".into(),
            width: self.width,
            height: self.height,
            frames: Some(self.frames),
            fps: Some(self.fps),
            video_mode: self.video_mode.request_value(),
            seed: Some(1234),
            ..Default::default()
        }
    }

    fn is_historical_default(self, tier: Tier) -> bool {
        self.width == DEFAULT_WIDTH
            && self.height == DEFAULT_HEIGHT
            && self.frames == DEFAULT_FRAMES
            && self.fps == DEFAULT_FPS
            && tier == Tier::Q4
            && self.video_mode == VideoMode::NoAudio
    }
}

fn lookup_u32(lookup: &mut impl FnMut(&str) -> Option<String>, name: &str, default: u32) -> u32 {
    match lookup(name) {
        Some(raw) => raw
            .parse::<u32>()
            .unwrap_or_else(|_| panic!("{name} must be an unsigned integer (got {raw:?})")),
        None => default,
    }
}

/// Resolve a tier-bearing snapshot deterministically for exploratory local runs. Prefer the cached
/// `refs/main` target; if that ref is absent, sort all matching immutable snapshot ids. Published
/// captures bypass this helper by supplying exact `LTX_MODEL_DIR` / `LTX_GEMMA_DIR` paths.
fn hf_snapshot_for_tier(models_root: &Path, model: &str, tier: Tier) -> Option<PathBuf> {
    let repo = models_root.join(model);
    let snapshots = repo.join("snapshots");
    if let Ok(revision) = std::fs::read_to_string(repo.join("refs/main")) {
        let candidate = snapshots.join(revision.trim());
        if candidate.join(tier.as_str()).is_dir() {
            return Some(candidate);
        }
    }
    let mut candidates: Vec<_> = std::fs::read_dir(&snapshots)
        .ok()?
        .filter_map(Result::ok)
        .map(|entry| entry.path())
        .filter(|path| path.is_dir() && path.join(tier.as_str()).is_dir())
        .collect();
    candidates.sort();
    candidates.into_iter().next()
}

/// The LTX model dir (split-weight snapshot) — `LTX_MODEL_DIR`, else the HF-cache
/// `SceneWorks/ltx-2.3-mlx/<LTX_TIER>`. An explicit model dir keeps its historical behavior and may
/// point at any tier; setting `LTX_TIER` alongside it turns the tier into a checked assertion.
fn model_dir_from_lookup(
    requested_tier: Option<Tier>,
    mut lookup: impl FnMut(&str) -> Option<String>,
) -> PathBuf {
    if let Some(path) = lookup("LTX_MODEL_DIR") {
        assert!(!path.is_empty(), "LTX_MODEL_DIR must not be empty");
        return PathBuf::from(path);
    }
    let tier = requested_tier.unwrap_or(Tier::Q4);
    let models_root = lookup("MLX_GEN_MODELS_ROOT").unwrap_or_else(|| {
        panic!(
            "set exact LTX_MODEL_DIR and LTX_GEMMA_DIR paths for a reproducible capture, or set \
             MLX_GEN_MODELS_ROOT for exploratory cache discovery"
        )
    });
    let snapshot = hf_snapshot_for_tier(
        Path::new(&models_root),
        "models--SceneWorks--ltx-2.3-mlx",
        tier,
    )
    .unwrap_or_else(|| {
        panic!(
            "no {} tier exists under {}/models--SceneWorks--ltx-2.3-mlx/snapshots",
            tier.as_str(),
            models_root
        )
    });
    snapshot.join(tier.as_str())
}

fn model_dir(requested_tier: Option<Tier>) -> PathBuf {
    model_dir_from_lookup(requested_tier, |name| std::env::var(name).ok())
}

/// The bundled Gemma-3-12B TE dir — `LTX_GEMMA_DIR`, else the snapshot's `gemma/` (the sibling of the
/// model `q4` dir).
fn gemma_dir_from_lookup(model: &Path, mut lookup: impl FnMut(&str) -> Option<String>) -> PathBuf {
    if let Some(path) = lookup("LTX_GEMMA_DIR") {
        assert!(!path.is_empty(), "LTX_GEMMA_DIR must not be empty");
        return PathBuf::from(path);
    }
    model
        .parent()
        .expect("model dir has a parent")
        .join("gemma")
}

fn gemma_dir(model: &Path) -> PathBuf {
    gemma_dir_from_lookup(model, |name| std::env::var(name).ok())
}

fn checked_capture_tier(model: &Path, requested: Option<Tier>) -> Tier {
    let detected = Tier::from_model_dir(model);
    if let Some(requested) = requested {
        assert_eq!(
            detected,
            requested,
            "LTX_TIER={} does not match the checkpoint at {} (detected {})",
            requested.as_str(),
            model.display(),
            detected.as_str()
        );
    }
    detected
}

/// Measure the resident footprint of the Gemma text phase ALONE: build the AudioVideo TE exactly as
/// `load()` does (bf16; the bundled `…/gemma` is dense bf16, so `gemma_quant = None`), run a real
/// `encode_av`, and `eval` so every layer's weights are forced resident. Returns the peak bytes.
fn te_resident_peak(model: &std::path::Path, gemma: &std::path::Path) -> usize {
    let cfg = LtxConfig::from_model_dir(model).expect("LtxConfig::from_model_dir");
    let gemma_w = Weights::from_dir(gemma).expect("gemma weights");
    let connector_w =
        Weights::from_file(model.join("connector.safetensors")).expect("connector weights");
    reset_peak_memory();
    let te = LtxTextEncoder::from_weights_av(
        &gemma_w,
        &connector_w,
        GemmaConfig::gemma_3_12b(),
        None, // the bundled `gemma/` is dense bf16 (no `config.json` quantization block)
        &cfg,
        Dtype::Bfloat16,
    )
    .expect("build TE");
    let tok = LtxTokenizer::from_dir(gemma).expect("tokenizer");
    // Pad to the production prompt length (`MAX_PROMPT_TOKENS` = 1024) so this baseline's Gemma encode
    // footprint matches the real text phase — and clears the connector's 128-register minimum.
    let (ids, mask) = tok
        .encode("a red fox trotting through a snowy forest", 1024)
        .expect("tokenize");
    let (video_ctx, audio_ctx) = te.encode_av(&ids, &mask).expect("encode_av");
    mlx_rs::transforms::eval([&video_ctx, &audio_ctx]).expect("eval");
    let peak = get_peak_memory();
    drop(te);
    clear_cache();
    peak
}

/// Run the real staged `generate`, returning the video frames, whether audio was returned, peak active
/// memory, and the allocator cache sampled at the end of the generate bracket.
fn staged_generate(
    model: &std::path::Path,
    gemma: &std::path::Path,
    tier: Tier,
    request: &GenerationRequest,
) -> (Vec<Image>, bool, usize, usize) {
    // Builder-style, not struct-update: `LoadSpec` carries crate-private prepared-pin and
    // receipt fields, so `..LoadSpec::new(..)` no longer compiles from outside `gen-core`.
    let mut spec = LoadSpec::new(WeightsSource::Dir(model.to_path_buf()))
        .with_text_encoder(WeightsSource::Dir(gemma.to_path_buf()));
    if let Some(quant) = tier.load_quant() {
        spec = spec.with_quant(quant);
    }
    let m = mlx_gen_ltx::provider_registry()
        .expect("build explicit LTX provider registry")
        .load("ltx_2_3", &spec)
        .expect("load ltx_2_3");
    reset_peak_memory();
    let out = m.generate(request, &mut |_| {}).expect("generate");
    let peak_active = get_peak_memory();
    let end_cache = get_cache_memory();
    let (frames, has_audio) = match out {
        GenerationOutput::Video { frames, audio, .. } => (frames, audio.is_some()),
        other => panic!("expected Video, got {other:?}"),
    };
    drop(m);
    clear_cache();
    (frames, has_audio, peak_active, end_cache)
}

fn capture_identity_summary(
    capture: CaptureConfig,
    tier: Tier,
    peak_active: usize,
    end_cache: usize,
) -> String {
    format!(
        "tier = {}\n  geometry = {}×{}×{}\n  fps = {}\n  video mode = {}\n  \
         staged generate peak active memory = {:.2} GiB\n  \
         staged generate end-of-bracket cache memory = {:.2} GiB",
        tier.as_str(),
        capture.width,
        capture.height,
        capture.frames,
        capture.fps,
        capture.video_mode.as_str(),
        peak_active as f64 / GIB,
        end_cache as f64 / GIB,
    )
}

#[test]
#[ignore = "needs SceneWorks/ltx-2.3-mlx (LTX_MODEL_DIR/LTX_GEMMA_DIR or the HF cache); ~25 GB+ RAM"]
fn ltx_staged_peak_stays_below_te_plus_dit_coresidence() {
    let capture = CaptureConfig::from_env();
    let model = model_dir(capture.requested_tier);
    let gemma = gemma_dir(&model);
    assert!(
        model.join("transformer.safetensors").exists(),
        "capture model is missing transformer.safetensors: {}",
        model.display()
    );
    assert!(
        gemma.is_dir(),
        "capture Gemma directory does not exist: {}",
        gemma.display()
    );
    let tier = checked_capture_tier(&model, capture.requested_tier);
    let request = capture.request();

    // The AvDiT's resident weight proxy: the on-disk `transformer.safetensors` bytes (follow symlink).
    let dit_bytes = std::fs::metadata(model.join("transformer.safetensors"))
        .expect("stat transformer.safetensors")
        .len() as usize;

    // Staged production path first, then the TE-alone resident peak (each brackets its own
    // reset/clear so neither inflates the other).
    let (frames, has_audio, staged_peak, staged_cache) =
        staged_generate(&model, &gemma, tier, &request);
    let te_peak = te_resident_peak(&model, &gemma);

    // The pre-sc-10976 `load()` held BOTH giants resident for the whole job, so its peak was AT LEAST
    // this (it also carried the small components + activations, which this estimate omits — i.e. the
    // bound is conservative).
    let coresident_estimate = te_peak + dit_bytes;

    let identity = capture_identity_summary(capture, tier, staged_peak, staged_cache);
    println!(
        "\nLTX sequential residency capture:\n  {identity}\n  \
         TE resident peak (Gemma text phase) = {:.2} GiB\n  \
         AvDiT weights (transformer.safetensors) = {:.2} GiB\n  \
         co-resident estimate (TE + DiT, pre-sc-10976 floor) = {:.2} GiB",
        te_peak as f64 / GIB,
        dit_bytes as f64 / GIB,
        coresident_estimate as f64 / GIB,
    );

    // (1) Non-degenerate output: the right number of frames at the requested size, and frame 0 is not a
    // flat single-color buffer (a smoke test that the staged denoise + decode actually produced pixels).
    assert_eq!(
        frames.len(),
        capture.frames as usize,
        "expected {} video frames",
        capture.frames
    );
    assert_eq!(
        has_audio,
        capture.video_mode.expects_audio(),
        "video mode {} returned an unexpected audio-track state",
        capture.video_mode.as_str()
    );
    let f0 = &frames[0];
    assert_eq!(
        f0.pixels.len(),
        (capture.width * capture.height * 3) as usize,
        "frame 0 is {}×{} — wrong pixel count",
        f0.width,
        f0.height
    );
    assert!(
        f0.pixels.iter().any(|&p| p != f0.pixels[0]),
        "frame 0 is a flat single-color buffer — the staged denoise/decode produced no image"
    );

    // (2) The historical small-q4 default is also the staging regression gate: at that geometry,
    // activations are small enough that TE+DiT is a valid upper bound. At larger configurable capture
    // geometries, denoise/decode activations can legitimately exceed that component-only estimate, so
    // applying this assertion would make the published maximum-envelope row unreproducible.
    if capture.is_historical_default(tier) {
        let saved = coresident_estimate.saturating_sub(staged_peak);
        println!(
            "  baseline delta below co-residence estimate = {:.2} GiB ({:.1}%)",
            saved as f64 / GIB,
            100.0 * saved as f64 / coresident_estimate as f64,
        );
        assert!(
            staged_peak < coresident_estimate,
            "staged peak {:.2} GiB was NOT below the TE+DiT co-residence estimate {:.2} GiB — the \
             Gemma drop before the DiT did not bound peak (staging regressed?)",
            staged_peak as f64 / GIB,
            coresident_estimate as f64 / GIB,
        );
        // (3) Tripwire: the DiT really left co-residence — the win should be multiple GiB (q4 DiT ≈
        // 10 GiB), well above measurement noise.
        assert!(
            saved as f64 / GIB > 2.0,
            "saved only {:.2} GiB — expected several GiB (≈ the DiT dropped out of co-residence); \
             staging may not be freeing the AvDiT / TE",
            saved as f64 / GIB,
        );
    } else {
        println!(
            "  note = non-baseline capture; the component-only co-residence comparison is reported \
             but not asserted (the regression threshold is calibrated only for the historical \
             q4/no-audio baseline, and activation memory scales with geometry)"
        );
    }
}

#[test]
fn capture_lookup_preserves_defaults_and_propagates_every_request_input() {
    let empty = std::collections::HashMap::<String, String>::new();
    let defaults = CaptureConfig::from_lookup(|name| empty.get(name).cloned());
    assert_eq!(
        defaults,
        CaptureConfig {
            width: DEFAULT_WIDTH,
            height: DEFAULT_HEIGHT,
            frames: DEFAULT_FRAMES,
            fps: DEFAULT_FPS,
            requested_tier: None,
            video_mode: VideoMode::NoAudio,
        }
    );
    assert!(defaults.is_historical_default(Tier::Q4));
    let request = defaults.request();
    assert_eq!(request.width, DEFAULT_WIDTH);
    assert_eq!(request.height, DEFAULT_HEIGHT);
    assert_eq!(request.frames, Some(DEFAULT_FRAMES));
    assert_eq!(request.fps, Some(DEFAULT_FPS));
    assert_eq!(request.video_mode.as_deref(), Some("no_audio"));

    let custom: std::collections::HashMap<String, String> = std::collections::HashMap::from([
        ("LTX_W".into(), "768".into()),
        ("LTX_H".into(), "512".into()),
        ("LTX_FRAMES".into(), "145".into()),
        ("LTX_FPS".into(), "30".into()),
        ("LTX_TIER".into(), "Q8".into()),
        ("LTX_VIDEO_MODE".into(), "default".into()),
    ]);
    let capture = CaptureConfig::from_lookup(|name| custom.get(name).cloned());
    assert_eq!(capture.requested_tier, Some(Tier::Q8));
    assert_eq!(capture.video_mode, VideoMode::Default);
    let request = capture.request();
    assert_eq!((request.width, request.height), (768, 512));
    assert_eq!(request.frames, Some(145));
    assert_eq!(request.fps, Some(30));
    assert_eq!(request.video_mode, None);
    assert_eq!(request.seed, Some(1234));
}

const CAPTURE_ENV_KEYS: [&str; 6] = [
    "LTX_W",
    "LTX_H",
    "LTX_FRAMES",
    "LTX_FPS",
    "LTX_TIER",
    "LTX_VIDEO_MODE",
];

#[test]
fn capture_from_env_subprocess_probe() {
    if std::env::var(CAPTURE_ENV_PROBE).as_deref() != Ok("1") {
        return;
    }
    let capture = CaptureConfig::from_env();
    assert_eq!(capture.requested_tier, Some(Tier::Bf16));
    assert_eq!(capture.video_mode, VideoMode::VideoOnly);
    let request = capture.request();
    assert_eq!((request.width, request.height), (640, 384));
    assert_eq!(request.frames, Some(17));
    assert_eq!(request.fps, Some(25));
    assert_eq!(request.video_mode.as_deref(), Some("video_only"));
    assert!(!capture.video_mode.expects_audio());
}

#[test]
fn capture_from_env_maps_video_modes_and_rejects_invalid_geometry() {
    // Exercise the real `from_env` path without mutating this test process. That keeps an
    // `--include-ignored` real capture isolated from the fixture by construction.
    let mut child =
        std::process::Command::new(std::env::current_exe().expect("current test binary"));
    child
        .arg("--exact")
        .arg("capture_from_env_subprocess_probe")
        .arg("--nocapture")
        .env(CAPTURE_ENV_PROBE, "1");
    for name in CAPTURE_ENV_KEYS {
        child.env_remove(name);
    }
    let output = child
        .env("LTX_W", "640")
        .env("LTX_H", "384")
        .env("LTX_FRAMES", "17")
        .env("LTX_FPS", "25")
        .env("LTX_TIER", "bf16")
        .env("LTX_VIDEO_MODE", "video_only")
        .output()
        .expect("run isolated from_env probe");
    assert!(
        output.status.success(),
        "isolated from_env probe failed:\nstdout:\n{}\nstderr:\n{}",
        String::from_utf8_lossy(&output.stdout),
        String::from_utf8_lossy(&output.stderr)
    );

    for (name, value) in [
        ("LTX_W", "480"),
        ("LTX_H", "352"),
        ("LTX_FRAMES", "16"),
        ("LTX_FPS", "0"),
        ("LTX_W", "not-a-number"),
        ("LTX_TIER", "q3"),
        ("LTX_VIDEO_MODE", "silent"),
    ] {
        let result = std::panic::catch_unwind(|| {
            let vars = std::collections::HashMap::from([(name.to_string(), value.to_string())]);
            CaptureConfig::from_lookup(|key| vars.get(key).cloned())
        });
        assert!(result.is_err(), "{name}={value:?} unexpectedly passed");
    }

    assert_eq!(
        VideoMode::NoAudio.request_value().as_deref(),
        Some("no_audio")
    );
    assert_eq!(
        VideoMode::VideoOnly.request_value().as_deref(),
        Some("video_only")
    );
    assert!(VideoMode::Default.expects_audio());
}

#[test]
fn model_paths_are_explicit_or_resolve_a_deterministic_tier_bearing_snapshot() {
    let explicit = model_dir_from_lookup(Some(Tier::Q8), |name| {
        (name == "LTX_MODEL_DIR").then(|| "/immutable/snapshot/q8".into())
    });
    assert_eq!(explicit, PathBuf::from("/immutable/snapshot/q8"));
    assert_eq!(
        gemma_dir_from_lookup(&explicit, |name| {
            (name == "LTX_GEMMA_DIR").then(|| "/immutable/snapshot/gemma".into())
        }),
        PathBuf::from("/immutable/snapshot/gemma")
    );

    let temp = tempfile::tempdir().expect("temp root");
    let repo = temp.path().join("models--SceneWorks--ltx-2.3-mlx");
    let snapshots = repo.join("snapshots");
    for revision in ["aaa-old", "bbb-main", "ccc-other"] {
        std::fs::create_dir_all(snapshots.join(revision)).expect("create snapshot");
    }
    std::fs::create_dir_all(snapshots.join("aaa-old/q8")).expect("old q8");
    std::fs::create_dir_all(snapshots.join("bbb-main/q4")).expect("main q4");
    std::fs::create_dir_all(snapshots.join("bbb-main/q8")).expect("main q8");
    std::fs::create_dir_all(snapshots.join("ccc-other/q8")).expect("other q8");
    std::fs::create_dir_all(repo.join("refs")).expect("refs");
    std::fs::write(repo.join("refs/main"), "bbb-main\n").expect("main ref");

    assert_eq!(
        hf_snapshot_for_tier(temp.path(), "models--SceneWorks--ltx-2.3-mlx", Tier::Q8),
        Some(snapshots.join("bbb-main"))
    );
    std::fs::remove_file(repo.join("refs/main")).expect("remove main ref");
    assert_eq!(
        hf_snapshot_for_tier(temp.path(), "models--SceneWorks--ltx-2.3-mlx", Tier::Q8),
        Some(snapshots.join("aaa-old"))
    );

    let root = temp.path().display().to_string();
    let resolved = model_dir_from_lookup(Some(Tier::Q8), |name| {
        (name == "MLX_GEN_MODELS_ROOT").then(|| root.clone())
    });
    assert_eq!(resolved, snapshots.join("aaa-old/q8"));
    assert_eq!(
        gemma_dir_from_lookup(&resolved, |_| None),
        snapshots.join("aaa-old/gemma")
    );

    let missing = std::panic::catch_unwind(|| {
        model_dir_from_lookup(Some(Tier::Bf16), |name| {
            (name == "MLX_GEN_MODELS_ROOT").then(|| root.clone())
        })
    });
    assert!(
        missing.is_err(),
        "missing explicitly requested tier must fail"
    );
}

#[test]
fn checkpoint_tier_detection_and_capture_identity_are_load_bearing() {
    let temp = tempfile::tempdir().expect("temp root");
    let q4 = temp.path().join("q4");
    let q8 = temp.path().join("q8");
    let bf16 = temp.path().join("bf16");
    for dir in [&q4, &q8, &bf16] {
        std::fs::create_dir_all(dir).expect("tier dir");
    }
    std::fs::write(
        q4.join("split_model.json"),
        r#"{"quantized":true,"quantization_bits":4}"#,
    )
    .expect("q4 split manifest");
    std::fs::write(
        q8.join("split_model.json"),
        r#"{"quantized":true,"quantization_bits":8}"#,
    )
    .expect("q8 split manifest");

    assert_eq!(checked_capture_tier(&q4, Some(Tier::Q4)), Tier::Q4);
    assert_eq!(checked_capture_tier(&q8, Some(Tier::Q8)), Tier::Q8);
    assert_eq!(checked_capture_tier(&bf16, Some(Tier::Bf16)), Tier::Bf16);
    let mismatch = std::panic::catch_unwind(|| checked_capture_tier(&q8, Some(Tier::Q4)));
    assert!(mismatch.is_err(), "requested and loaded tier must agree");

    let capture = CaptureConfig {
        width: 768,
        height: 512,
        frames: 145,
        fps: 24,
        requested_tier: Some(Tier::Q8),
        video_mode: VideoMode::Default,
    };
    let summary = capture_identity_summary(
        capture,
        Tier::Q8,
        5 * 1024_usize.pow(3),
        2 * 1024_usize.pow(3),
    );
    for expected in [
        "tier = q8",
        "geometry = 768×512×145",
        "fps = 24",
        "video mode = default",
        "peak active memory = 5.00 GiB",
        "end-of-bracket cache memory = 2.00 GiB",
    ] {
        assert!(
            summary.contains(expected),
            "missing {expected:?} in {summary:?}"
        );
    }
}
