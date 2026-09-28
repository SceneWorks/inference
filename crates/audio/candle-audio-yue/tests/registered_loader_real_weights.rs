//! **Real-weight end-to-end smoke through the registered loader** (epic sc-19373 R9): a
//! `LoadSpec` over the staged SceneWorks YuE assets goes through the provider registry's `load`
//! (the same registrations the audio catalog composes and the worker reaches), never through a
//! component loader, and one short song segment is rendered through every production stage —
//! mm tokenizer → 7B stage 1 → 1B stage 2 → xcodec → Vocos → splice.
//!
//! Cases, each over the **worker's `LoadSpec` shape** — weights = the selected stage-1 tier
//! directory (`<stage-1 repo>/<tier>`), `stage2` = the same tier of the stage-2 repo
//! (`<stage-2 repo>/<tier>`), `xcodec` = the xcodec repo root (not tiered), `quantize` = the tier:
//!
//! - `yue_en_cot` (no reference);
//! - `yue_en_icl` with a short single-track reference (upstream's `prompt_egs/pop.00001.mp3`,
//!   decoded to f32 by `scripts/reference/yue_icl_reference.py` into
//!   `$YUE_REF_DIR/sceneworks-derived/`);
//! - `yue_en_cot` over the tiered repo **roots** — the other layout
//!   [`candle_audio_yue::snapshot::resolve_tier_dir`] publicly accepts (the worker does not send
//!   it; a direct engine caller may).
//!
//! Gated like the other real-weight tests: `#[ignore]`d in ordinary runs; under `--ignored` the
//! variables are **required** (unset panics — the test never passes silently).
//!
//! ```text
//! YUE_SNAPSHOT_ROOT=~/.cache/sceneworks-yue-assets YUE_REF_DIR=~/.cache/sceneworks-yue-ref \
//!   [YUE_TIER=q4|q8|bf16] \
//!   cargo test --release -p candle-audio-yue --test registered_loader_real_weights \
//!   -- --ignored --nocapture --test-threads 1
//! ```
//!
//! `YUE_SNAPSHOT_ROOT` holds the tiered rehost roots `yue-s1-7b-anneal-en-{cot,icl}-candle/`,
//! `yue-s2-1b-general-candle/` and `xcodec-mini-infer/`. `YUE_TIER` (default `q4`) selects the
//! tier directory of both LMs. Add `--features metal` / `--features cuda` to run it on the
//! accelerator.
//!
//! Measured on CPU (M-series Mac, q4, release build, 2026-09-25, one case per process under an
//! external RSS watchdog): worker-shape `yue_en_cot` 78 s wall, peak RSS 9.7 GB; worker-shape
//! `yue_en_icl` (4 s stereo clip, 0–3 s window) 102 s, 6.8 GB; repo-root `yue_en_cot` 75 s,
//! 9.7 GB. Peaks vary run to run (7.4–9.7 GB observed for the same CoT case).

use std::path::PathBuf;
use std::time::Instant;

use candle_audio_yue::gen_core::{
    AudioParams, AudioTrack, Conditioning, GenerationOutput, GenerationRequest, LoadPhase,
    LoadSpec, Progress, Quant, TimeRegion, WeightsSource,
};
use candle_audio_yue::model::{SAMPLE_RATE, STAGE2_COMPONENT_ID, XCODEC_COMPONENT_ID};

fn required_env(name: &str, what: &str) -> PathBuf {
    std::env::var_os(name)
        .unwrap_or_else(|| panic!("set {name} to {what}"))
        .into()
}

fn snapshot_root() -> PathBuf {
    required_env(
        "YUE_SNAPSHOT_ROOT",
        "the directory holding the staged yue-s1-7b-anneal-*-candle, yue-s2-1b-general-candle and \
         xcodec-mini-infer snapshot roots",
    )
}

/// The tier directory name and the `quantize` it asserts.
fn tier() -> (&'static str, Option<Quant>) {
    match std::env::var("YUE_TIER").as_deref() {
        Err(_) | Ok("q4") => ("q4", Some(Quant::Q4)),
        Ok("q8") => ("q8", Some(Quant::Q8)),
        Ok("bf16") => ("bf16", None),
        Ok(other) => panic!("YUE_TIER must be q4, q8 or bf16, got {other}"),
    }
}

/// How the LM snapshots are handed to the loader.
#[derive(Clone, Copy, Debug)]
enum Layout {
    /// What the SceneWorks worker sends: each LM's selected tier directory.
    WorkerTierDirs,
    /// The tiered repo roots (the loader picks the tier directory).
    RepoRoots,
}

/// The `LoadSpec` for `stage1_repo` in `layout`; xcodec is always its (untiered) repo root.
fn load_spec(stage1_repo: &str, layout: Layout) -> LoadSpec {
    let root = snapshot_root();
    let (tier_dir, quantize) = tier();
    let dir = |repo: &str, tiered: bool| {
        let mut p = root.join(repo);
        if tiered && matches!(layout, Layout::WorkerTierDirs) {
            p = p.join(tier_dir);
        }
        assert!(p.is_dir(), "{} is not staged", p.display());
        WeightsSource::Dir(p)
    };
    let mut spec = LoadSpec::new(dir(stage1_repo, true))
        .with_component(STAGE2_COMPONENT_ID, dir("yue-s2-1b-general-candle", true))
        .with_component(XCODEC_COMPONENT_ID, dir("xcodec-mini-infer", false));
    spec.quantize = quantize;
    spec
}

/// The smallest valid song: one lyric segment, the 100-token stage-1 budget (the default
/// `min_new_tokens` floor, so `<EOA>` cannot end it early), a fixed seed.
fn one_segment_request() -> GenerationRequest {
    GenerationRequest {
        prompt: "pop upbeat female vocal bright".into(),
        seed: Some(7),
        audio: Some(AudioParams {
            lyrics: Some("[verse]\nHello sunshine on my face\n".into()),
            segments: Some(1),
            max_new_tokens_per_segment: Some(100),
            ..Default::default()
        }),
        ..Default::default()
    }
}

/// Load `id` through the registry, render `req`, and hold the output and progress to the
/// generator contract.
fn render_and_check(id: &str, stage1_repo: &str, layout: Layout, req: &GenerationRequest) {
    let spec = load_spec(stage1_repo, layout);
    let registry = candle_audio_yue::provider_registry().expect("registry builds");
    let t0 = Instant::now();
    let generator = registry
        .load(id, &spec)
        .unwrap_or_else(|e| panic!("{id}: the registered loader refused: {e}"));
    assert_eq!(generator.descriptor().id, id);
    generator.validate(req).expect("the request is valid");

    let mut events = Vec::new();
    let out = generator
        .generate(req, &mut |p| events.push(p))
        .unwrap_or_else(|e| panic!("{id}: render failed: {e}"));
    let elapsed = t0.elapsed();

    let GenerationOutput::Audio(track) = out else {
        panic!("{id}: expected audio output");
    };
    let AudioTrack {
        samples,
        sample_rate,
        channels,
        stems,
    } = track;
    assert_eq!(sample_rate, SAMPLE_RATE, "{id}: 44.1 kHz mix");
    assert_eq!(sample_rate, 44_100);
    assert_eq!(channels, 1);
    assert!(!samples.is_empty(), "{id}: empty mix");
    let names: Vec<&str> = stems.iter().map(|s| s.name.as_str()).collect();
    assert_eq!(names, ["vocals", "instrumental"], "{id}: stems");
    for stem in &stems {
        assert_eq!(
            stem.samples.len(),
            samples.len(),
            "{id}: `{}` stem length differs from the mix",
            stem.name
        );
    }
    for (name, s) in std::iter::once(("mix", &samples))
        .chain(stems.iter().map(|s| (s.name.as_str(), &s.samples)))
    {
        assert!(
            s.iter().all(|x| x.is_finite()),
            "{id}: non-finite {name} sample"
        );
        assert!(
            s.iter().any(|&x| x != 0.0),
            "{id}: the {name} track is all zeros"
        );
    }

    // Progress: both LMs report loading, one step per segment plus one per stage-2 track
    // (total = segments + 2), then the codec + vocoder pass.
    let loads = events
        .iter()
        .filter(|p| matches!(p, Progress::Loading(LoadPhase::Renderer)))
        .count();
    assert_eq!(
        loads, 2,
        "{id}: stage-1 and stage-2 load events: {events:?}"
    );
    let steps: Vec<(u32, u32)> = events
        .iter()
        .filter_map(|p| match p {
            Progress::Step { current, total } => Some((*current, *total)),
            _ => None,
        })
        .collect();
    assert_eq!(steps, [(1, 3), (2, 3), (3, 3)], "{id}: step events");
    assert!(
        events.iter().any(|p| matches!(p, Progress::Decoding)),
        "{id}: no Decoding event: {events:?}"
    );
    println!(
        "{id}: {:.2} s of audio ({} samples @ {sample_rate} Hz) in {:.1} s (load + render, tier {:?})",
        samples.len() as f32 / sample_rate as f32,
        samples.len(),
        elapsed.as_secs_f32(),
        spec.quantize
    );
}

#[test]
#[ignore = "real weights: set YUE_SNAPSHOT_ROOT (see the module docs)"]
fn en_cot_renders_one_segment_through_the_registered_loader() {
    render_and_check(
        "yue_en_cot",
        "yue-s1-7b-anneal-en-cot-candle",
        Layout::WorkerTierDirs,
        &one_segment_request(),
    );
}

#[test]
#[ignore = "real weights: set YUE_SNAPSHOT_ROOT (see the module docs)"]
fn en_cot_renders_one_segment_from_the_tiered_repo_roots() {
    render_and_check(
        "yue_en_cot",
        "yue-s1-7b-anneal-en-cot-candle",
        Layout::RepoRoots,
        &one_segment_request(),
    );
}

#[test]
#[ignore = "real weights: set YUE_SNAPSHOT_ROOT and YUE_REF_DIR (see the module docs)"]
fn en_icl_renders_one_segment_with_a_single_track_reference() {
    let ref_dir = required_env(
        "YUE_REF_DIR",
        "the YuE reference environment (holds sceneworks-derived/pop.00001.f32le, written by \
         scripts/reference/yue_icl_reference.py)",
    );
    let path = ref_dir.join("sceneworks-derived").join("pop.00001.f32le");
    let raw = std::fs::read(&path).unwrap_or_else(|e| {
        panic!(
            "{}: {e} — run scripts/reference/yue_icl_reference.py to decode it",
            path.display()
        )
    });
    // 44.1 kHz interleaved stereo; keep the first 4 s and prompt with its 0–3 s window.
    let (rate, channels, secs) = (44_100usize, 2usize, 4usize);
    let samples: Vec<f32> = raw
        .chunks_exact(4)
        .take(rate * channels * secs)
        .map(|b| f32::from_le_bytes([b[0], b[1], b[2], b[3]]))
        .collect();
    assert_eq!(
        samples.len(),
        rate * channels * secs,
        "clip shorter than 4 s"
    );
    let mut req = one_segment_request();
    req.conditioning = vec![Conditioning::ReferenceAudio {
        audio: AudioTrack {
            samples,
            sample_rate: rate as u32,
            channels: channels as u16,
            stems: Vec::new(),
        },
        strength: None,
    }];
    req.audio.as_mut().unwrap().reference_region = Some(TimeRegion {
        start_secs: 0.0,
        end_secs: Some(3.0),
    });
    render_and_check(
        "yue_en_icl",
        "yue-s1-7b-anneal-en-icl-candle",
        Layout::WorkerTierDirs,
        &req,
    );
}
