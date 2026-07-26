//! Mis-wiring gates for the registered Stable Audio 3 checkpoints (`sc-14544`, `sc-14545`).
//!
//! `small-music` and `small-sfx` are architecturally identical: same tensor inventory, same DiT
//! geometry, same SAME-S pretransform, same bundled T5Gemma stack, and root checkpoints of exactly
//! the same 2,270,384,940 bytes. `gen_core::ModelRegistration::load` receives no provider id, so
//! nothing about the `LoadSpec` says which registration the caller reached. Every test here mutates
//! a real snapshot and asserts the load fails — a test that passes with swapped weights is a false
//! green, and would be the only thing standing between a consumer and silently receiving music
//! from `stable_audio_3_small_sfx`.
//!
//! `medium` (sc-14545) adds a second, sharper case. Against the smalls it is separated on shape
//! alone (997 root tensors against 685, `1536x24` differential against `1024x20` ordinary, a
//! `16,777,216`-frame ceiling against `5,292,032`), so those swaps never reach the hash pin. Against
//! its own **base** sibling it is separated by almost nothing:
//! `stabilityai/stable-audio-3-medium-base` ships the *same* 997/522/472 inventory, the *same*
//! `16,777,216` `sample_size`, a root safetensors of the *same* 9,222,116,660 bytes, and — unlike
//! the music/SFX pair — the *same* conditioner `repo_id` (`stabilityai/stable-audio-3-medium`).
//! The only config-level difference in the entire file is `diffusion_objective`
//! (`rectified_flow` against `rf_denoiser`). So for medium the `repo_id` check discriminates
//! nothing, and the objective check plus the SHA-256 pin are the whole gate.

use std::path::{Path, PathBuf};

use candle_audio_stable_audio_3::gen_core::{
    self, AudioParams, GenerationOutput, GenerationRequest, Generator, LoadSpec, WeightsSource,
};
use candle_audio_stable_audio_3::pipeline::conditioner_repo_id;
use candle_audio_stable_audio_3::weights::SnapshotLayout;
use candle_audio_stable_audio_3::{load_variant, Variant};

/// The cheapest request that still exercises the whole lazy load and synthesis path.
fn short_request() -> GenerationRequest {
    GenerationRequest {
        prompt: "Futuristic laser blast, sharp energy pulse, stereo movement, arcade style".into(),
        seed: Some(42),
        steps: Some(1),
        sampler: Some("pingpong".into()),
        audio: Some(AudioParams {
            target_duration: Some(0.25),
            sample_rate: Some(44_100),
            ..Default::default()
        }),
        ..Default::default()
    }
}

const ENTRIES: &[&str] = &[
    "model_config.json",
    "model.safetensors",
    "t5gemma-b-b-ul2/config.json",
    "t5gemma-b-b-ul2/model.safetensors",
    "t5gemma-b-b-ul2/tokenizer.json",
    "t5gemma-b-b-ul2/tokenizer.model",
];

fn snapshot(env: &str) -> PathBuf {
    PathBuf::from(
        std::env::var(env).unwrap_or_else(|_| panic!("set {env} to the pinned immutable snapshot")),
    )
}

fn music() -> PathBuf {
    snapshot("SA3_SMALL_MUSIC_SNAPSHOT")
}

fn sfx() -> PathBuf {
    snapshot("SA3_SMALL_SFX_SNAPSHOT")
}

fn medium() -> PathBuf {
    snapshot("SA3_MEDIUM_SNAPSHOT")
}

/// A snapshot that is not part of any shipped registration, so a developer running this suite by
/// hand can skip the mutations that need it.
///
/// A skip that CI also takes is a gate nobody runs, so the real-weight jobs set
/// `SA3_REQUIRE_ALL_SNAPSHOTS=1`, which turns every skip below into a failure. Without that, adding
/// a job and forgetting the environment variable would look identical to a passing run.
fn optional_snapshot(env: &str) -> Option<PathBuf> {
    match std::env::var(env) {
        Ok(value) => Some(PathBuf::from(value)),
        Err(_) if std::env::var("SA3_REQUIRE_ALL_SNAPSHOTS").is_ok() => panic!(
            "{env} is unset while SA3_REQUIRE_ALL_SNAPSHOTS is set — this gate would have been \
             skipped silently"
        ),
        Err(_) => None,
    }
}

/// `stabilityai/stable-audio-3-medium-base` — same inventory, same `sample_size`, same root byte
/// length, same conditioner `repo_id`.
fn medium_base() -> Option<PathBuf> {
    optional_snapshot("SA3_MEDIUM_BASE_SNAPSHOT")
}

/// `stabilityai/SAME-L` — the standalone autoencoder whose 472 tensors medium embeds verbatim.
fn standalone_same_l() -> Option<PathBuf> {
    optional_snapshot("SA3_SAME_L_SNAPSHOT")
}

/// Materialize one snapshot entry without copying gigabytes where the platform allows it.
///
/// The source is canonicalized first: a provisioned snapshot may itself be a tree of relative
/// symlinks into a content-addressed store, and linking such an entry into a different directory
/// would carry a relative target that no longer resolves. The destination keeps the snapshot-
/// relative name regardless of what the canonical source file is called.
fn link(source: &Path, destination: &Path) {
    let source = std::fs::canonicalize(source)
        .unwrap_or_else(|error| panic!("resolve {}: {error}", source.display()));
    if std::fs::hard_link(&source, destination).is_ok() {
        return;
    }
    #[cfg(unix)]
    if std::os::unix::fs::symlink(&source, destination).is_ok() {
        return;
    }
    #[cfg(windows)]
    if std::os::windows::fs::symlink_file(&source, destination).is_ok() {
        return;
    }
    std::fs::copy(&source, destination).expect("materialize snapshot entry");
}

/// A snapshot directory assembled from `base`, with each `(relative, source_root)` override taken
/// from a different checkpoint.
struct Assembled {
    root: PathBuf,
}

impl Assembled {
    fn new(label: &str, base: &Path, overrides: &[(&str, PathBuf)]) -> Self {
        let root = std::env::temp_dir().join(format!(
            "sa3-variant-binding-{}-{label}",
            std::process::id()
        ));
        let _ = std::fs::remove_dir_all(&root);
        std::fs::create_dir_all(root.join("t5gemma-b-b-ul2")).unwrap();
        for entry in ENTRIES {
            let source_root = overrides
                .iter()
                .find(|(relative, _)| relative == entry)
                .map(|(_, source)| source.clone())
                .unwrap_or_else(|| base.to_path_buf());
            link(&source_root.join(entry), &root.join(entry));
        }
        Self { root }
    }

    fn spec(&self) -> LoadSpec {
        LoadSpec::new(WeightsSource::Dir(self.root.clone()))
    }

    /// Re-point one entry at a different checkpoint *after* the directory has been handed to a
    /// loader, without disturbing any other entry.
    fn replace(&self, relative: &str, source_root: &Path) {
        let destination = self.root.join(relative);
        std::fs::remove_file(&destination)
            .unwrap_or_else(|error| panic!("unlink {}: {error}", destination.display()));
        link(&source_root.join(relative), &destination);
    }
}

/// The assembly mechanism's own control.
///
/// Reassembling a snapshot with **no** overrides must still load. Without this, a broken
/// materialization (a dangling link, a missing entry) would make every mutation assertion in this
/// file pass for the wrong reason — the loader would be rejecting a broken directory, not a swapped
/// checkpoint.
fn assert_unmutated_reassembly_loads(label: &str, variant: Variant, base: &Path) {
    let assembled = Assembled::new(label, base, &[]);
    load_variant(variant, &assembled.spec()).unwrap_or_else(|error| {
        panic!(
            "{label}: an unmutated reassembly of {} must load — the mutation gates in this file \
             are meaningless otherwise ({error})",
            variant.model_id()
        )
    });
}

impl Drop for Assembled {
    fn drop(&mut self) {
        let _ = std::fs::remove_dir_all(&self.root);
    }
}

fn expect_rejected(label: &str, variant: Variant, spec: &LoadSpec) {
    match load_variant(variant, spec) {
        Ok(_) => panic!(
            "{label}: {} accepted a mutated snapshot — the identity gate is a false green",
            variant.model_id()
        ),
        Err(gen_core::Error::Msg(message)) => {
            eprintln!("{label}: rejected with `{message}`");
            assert!(
                message.contains(variant.model_id()) || message.contains(variant.hub_repo()),
                "{label}: rejection must name the registration it protects: {message}"
            );
        }
        Err(other) => panic!("{label}: expected an identity rejection, got {other:?}"),
    }
}

#[test]
#[ignore = "requires both pinned 3.45 GB small snapshots"]
fn each_variant_accepts_only_its_own_snapshot() {
    // The discriminating control: unmutated snapshots must load under their own registration.
    // Without this, every assertion below would pass on a loader that rejects everything.
    load_variant(
        Variant::SmallMusic,
        &LoadSpec::new(WeightsSource::Dir(music())),
    )
    .expect("the real small-music snapshot must load under stable_audio_3_small_music");
    load_variant(Variant::SmallSfx, &LoadSpec::new(WeightsSource::Dir(sfx())))
        .expect("the real small-sfx snapshot must load under stable_audio_3_small_sfx");

    expect_rejected(
        "music snapshot under the SFX registration",
        Variant::SmallSfx,
        &LoadSpec::new(WeightsSource::Dir(music())),
    );
    expect_rejected(
        "SFX snapshot under the music registration",
        Variant::SmallMusic,
        &LoadSpec::new(WeightsSource::Dir(sfx())),
    );
}

#[test]
#[ignore = "requires both pinned 3.45 GB small snapshots"]
fn swapping_the_root_checkpoint_under_a_registration_fails_the_identity_gate() {
    assert_unmutated_reassembly_loads("sfx-reassembled", Variant::SmallSfx, &sfx());
    assert_unmutated_reassembly_loads("music-reassembled", Variant::SmallMusic, &music());

    // The SFX config and the music DiT/SAME-S/conditioner weights. The config-level `repo_id`
    // check passes here; only the root SHA-256 pin catches it, and the safetensors header carries
    // no identity metadata of its own.
    let music_root_under_sfx = Assembled::new(
        "music-root-under-sfx",
        &sfx(),
        &[("model.safetensors", music())],
    );
    expect_rejected(
        "music root safetensors under the SFX config",
        Variant::SmallSfx,
        &music_root_under_sfx.spec(),
    );

    let sfx_root_under_music = Assembled::new(
        "sfx-root-under-music",
        &music(),
        &[("model.safetensors", sfx())],
    );
    expect_rejected(
        "SFX root safetensors under the music config",
        Variant::SmallMusic,
        &sfx_root_under_music.spec(),
    );
}

#[test]
#[ignore = "requires both pinned 3.45 GB small snapshots"]
fn mixing_one_variants_conditioner_config_with_the_others_dit_fails() {
    assert_unmutated_reassembly_loads("mix-control-sfx", Variant::SmallSfx, &sfx());

    // The music `model_config.json` — and therefore the music conditioner `repo_id` — bolted onto
    // the SFX root checkpoint. The registration-bound `repo_id` check rejects this before any
    // tensor is read.
    let music_config_on_sfx_dit = Assembled::new(
        "music-config-on-sfx-dit",
        &sfx(),
        &[("model_config.json", music())],
    );
    expect_rejected(
        "music conditioner config with the SFX DiT, under the SFX id",
        Variant::SmallSfx,
        &music_config_on_sfx_dit.spec(),
    );
    expect_rejected(
        "music conditioner config with the SFX DiT, under the music id",
        Variant::SmallMusic,
        &music_config_on_sfx_dit.spec(),
    );

    let sfx_config_on_music_dit = Assembled::new(
        "sfx-config-on-music-dit",
        &music(),
        &[("model_config.json", sfx())],
    );
    expect_rejected(
        "SFX conditioner config with the music DiT, under the music id",
        Variant::SmallMusic,
        &sfx_config_on_music_dit.spec(),
    );
    expect_rejected(
        "SFX conditioner config with the music DiT, under the SFX id",
        Variant::SmallSfx,
        &sfx_config_on_music_dit.spec(),
    );
}

/// The snapshot must still be authentic when the tensors are actually read, not only when the
/// generator was constructed.
///
/// `load_variant` verifies the pins, but the pipeline is built lazily on first generate, so there
/// is a window between the two. Swapping `model.safetensors` inside that window and leaving
/// `model_config.json` in place keeps the conditioner `repo_id` check passing, and both roots are
/// exactly 2,270,384,940 bytes — the SHA-256 pin is the only thing that can tell them apart, and it
/// has to run on the lazy path for that to matter.
///
/// The discriminating control is the **same generator instance** serving real audio once the
/// authentic root is restored: this test cannot pass on a generator that simply never works, and
/// it would fail outright if the re-verification in `pipeline()` were deleted.
#[test]
#[ignore = "requires both pinned 3.45 GB small snapshots"]
fn swapping_the_root_after_load_is_rejected_before_any_tensor_is_read() {
    let assembled = Assembled::new("post-load-swap", &sfx(), &[]);
    let generator = load_variant(Variant::SmallSfx, &assembled.spec())
        .expect("the authentic SFX reassembly must load");

    assembled.replace("model.safetensors", &music());
    match generator.generate(&short_request(), &mut |_| {}) {
        Ok(_) => panic!(
            "stable_audio_3_small_sfx served music weights swapped in after load — the pinned \
             identity is not re-checked at tensor-open time"
        ),
        Err(gen_core::Error::Msg(message)) => {
            eprintln!("post-load root swap rejected with `{message}`");
            assert!(
                message.contains(Variant::SmallSfx.model_id())
                    && message.contains("model.safetensors"),
                "rejection must name the registration and the swapped file: {message}"
            );
        }
        Err(other) => panic!("expected an identity rejection, got {other:?}"),
    }

    assembled.replace("model.safetensors", &sfx());
    match generator
        .generate(&short_request(), &mut |_| {})
        .expect("the same generator must serve once the authentic root is restored")
    {
        GenerationOutput::Audio(track) => {
            assert_eq!(track.sample_rate, 44_100);
            assert!(!track.samples.is_empty());
        }
        other => panic!("expected audio, got {other:?}"),
    }
}

#[test]
#[ignore = "requires both pinned 3.45 GB small snapshots"]
fn shipped_configs_declare_the_conditioner_repo_that_identifies_them() {
    for (root, expected) in [
        (music(), Variant::SmallMusic),
        (sfx(), Variant::SmallSfx),
        (medium(), Variant::Medium),
    ] {
        let layout = SnapshotLayout::from_dir(&root).unwrap();
        assert_eq!(
            conditioner_repo_id(&layout),
            Some(expected.hub_repo()),
            "{} conditioner repo_id",
            expected.model_id()
        );
    }
}

/// Medium against the two smalls, in both directions.
///
/// These are shape rejections rather than identity rejections — the wrapper never gets as far as
/// hashing — which is exactly the property sc-14545 had to preserve when the hard-coded small
/// geometry became a per-variant record. Before that change the validator pinned `685` tensors and
/// `1024x20` for every registration; loosening it to admit medium would have silently opened the
/// small ids to medium weights.
#[test]
#[ignore = "requires the pinned small and medium snapshots"]
fn medium_and_the_smalls_reject_each_others_snapshots() {
    // Controls first: each real snapshot must load under its own registration. Without these the
    // rejections below would pass on a loader that rejects everything.
    load_variant(
        Variant::Medium,
        &LoadSpec::new(WeightsSource::Dir(medium())),
    )
    .expect("the real medium snapshot must load under stable_audio_3_medium");
    load_variant(
        Variant::SmallMusic,
        &LoadSpec::new(WeightsSource::Dir(music())),
    )
    .expect("the real small-music snapshot must load under stable_audio_3_small_music");
    load_variant(Variant::SmallSfx, &LoadSpec::new(WeightsSource::Dir(sfx())))
        .expect("the real small-sfx snapshot must load under stable_audio_3_small_sfx");

    for (label, variant) in [
        (
            "music snapshot under the medium registration",
            Variant::SmallMusic,
        ),
        (
            "SFX snapshot under the medium registration",
            Variant::SmallSfx,
        ),
    ] {
        let root = if variant == Variant::SmallMusic {
            music()
        } else {
            sfx()
        };
        expect_rejected(
            label,
            Variant::Medium,
            &LoadSpec::new(WeightsSource::Dir(root)),
        );
    }
    expect_rejected(
        "medium snapshot under the music registration",
        Variant::SmallMusic,
        &LoadSpec::new(WeightsSource::Dir(medium())),
    );
    expect_rejected(
        "medium snapshot under the SFX registration",
        Variant::SmallSfx,
        &LoadSpec::new(WeightsSource::Dir(medium())),
    );
}

/// The medium mutation the `repo_id` check cannot see.
///
/// `medium-base` matches medium on every quantity the wrapper checks except `diffusion_objective`,
/// and matches it byte-for-byte on root length. If this were admitted, `stable_audio_3_medium` would
/// serve a base checkpoint whose 8-step Pingpong default and `cfg_scale=1` descriptor are wrong for
/// it — a silent quality collapse rather than an error.
#[test]
#[ignore = "requires the pinned medium and medium-base snapshots"]
fn the_medium_base_sibling_is_rejected_under_the_post_trained_medium_id() {
    let Some(base) = medium_base() else {
        eprintln!("SA3_MEDIUM_BASE_SNAPSHOT unset; skipping the sharpest medium mutation");
        return;
    };
    // Control: the post-trained snapshot loads.
    load_variant(
        Variant::Medium,
        &LoadSpec::new(WeightsSource::Dir(medium())),
    )
    .expect("the real medium snapshot must load under stable_audio_3_medium");

    expect_rejected(
        "medium-base snapshot under the post-trained medium registration",
        Variant::Medium,
        &LoadSpec::new(WeightsSource::Dir(base.clone())),
    );

    // And the mutation that isolates the hash pin from the objective check: the post-trained config
    // (so `diffusion_objective` reads `rf_denoiser` and `repo_id` matches) over the base root
    // weights, which are the same 9,222,116,660 bytes.
    assert_unmutated_reassembly_loads("medium-reassembled", Variant::Medium, &medium());
    let base_root_under_medium = Assembled::new(
        "medium-base-root-under-medium",
        &medium(),
        &[("model.safetensors", base)],
    );
    expect_rejected(
        "medium-base root safetensors under the post-trained medium config",
        Variant::Medium,
        &base_root_under_medium.spec(),
    );
}

/// A standalone SAME-L snapshot is an autoencoder, not a provider checkpoint.
#[test]
#[ignore = "requires the pinned SAME-L snapshot"]
fn the_standalone_same_l_autoencoder_is_rejected_under_every_registration() {
    let Some(same_l) = standalone_same_l() else {
        eprintln!("SA3_SAME_L_SNAPSHOT unset; skipping the standalone-autoencoder mutation");
        return;
    };
    for variant in [Variant::Medium, Variant::SmallMusic, Variant::SmallSfx] {
        match load_variant(variant, &LoadSpec::new(WeightsSource::Dir(same_l.clone()))) {
            Ok(_) => panic!(
                "{} accepted a standalone SAME-L autoencoder snapshot",
                variant.model_id()
            ),
            Err(error) => eprintln!("{}: rejected with `{error}`", variant.model_id()),
        }
    }
}

/// Medium's own load-to-use window, with the same discriminating restore control the SFX case uses.
#[test]
#[ignore = "requires the pinned medium and medium-base snapshots"]
fn swapping_the_medium_root_after_load_is_rejected_before_any_tensor_is_read() {
    let Some(base) = medium_base() else {
        eprintln!("SA3_MEDIUM_BASE_SNAPSHOT unset; skipping the medium post-load swap");
        return;
    };
    let assembled = Assembled::new("medium-post-load-swap", &medium(), &[]);
    let generator = load_variant(Variant::Medium, &assembled.spec())
        .expect("the authentic medium reassembly must load");

    assembled.replace("model.safetensors", &base);
    match generator.generate(&short_request(), &mut |_| {}) {
        Ok(_) => panic!(
            "stable_audio_3_medium served base weights swapped in after load — the pinned identity \
             is not re-checked at tensor-open time"
        ),
        Err(gen_core::Error::Msg(message)) => {
            eprintln!("medium post-load root swap rejected with `{message}`");
            assert!(
                message.contains(Variant::Medium.model_id())
                    && message.contains("model.safetensors"),
                "rejection must name the registration and the swapped file: {message}"
            );
        }
        Err(other) => panic!("expected an identity rejection, got {other:?}"),
    }

    assembled.replace("model.safetensors", &medium());
    match generator
        .generate(&short_request(), &mut |_| {})
        .expect("the same generator must serve once the authentic root is restored")
    {
        GenerationOutput::Audio(track) => {
            assert_eq!(track.sample_rate, 44_100);
            assert!(!track.samples.is_empty());
        }
        other => panic!("expected audio, got {other:?}"),
    }
}
