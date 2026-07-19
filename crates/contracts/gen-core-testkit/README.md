# gen-core-testkit

> Package: `sceneworks-gen-core-testkit` · library: `gen_core_testkit`

A **contract conformance suite** for [`gen-core`](../gen-core/README.md) providers. Given any
boxed provider — an MLX family from `mlx-gen`, a Candle family from `candle-gen` — it
exercises the behavioral guarantees the contract *promises but cannot express in the type
system*:

- **typed cancellation** — a tripped `CancelFlag` actually stops the work;
- **progress monotonicity** — `Progress` events advance and never regress;
- **seed determinism** — the same seed reproduces output; a fresh seed does not;
- **capability honesty** — a provider serves exactly what its `Capabilities` advertise, and
  rejects the rest in `validate`.

Like `gen-core`, the testkit has **zero tensor dependencies** — it drives the provider purely
through the public contract, so it runs on the Linux `gen-core` lane against an in-crate stub
exactly as it does on a backend lane against a real family. Both backends run it in CI, so a
provider that silently ignores `CancelFlag` or reports no progress becomes a CI failure
instead of a field report.

## Contracts covered

One shared conformance suite per contract, each with an in-crate stub that self-tests it (a
compliant stub passes; a deliberately-broken stub — one that ignores cancel, returns a
wrong-dimension embedding, or produces the wrong stem count — fails):

- **`Generator`** (image/video) — `conformance` / `Profile`;
- **`Generator` under `Modality::Audio`** (TTS / SFX / music) — `audio_conformance` /
  `AudioProfile` (validates through the size-skipping audio floor; asserts a well-formed
  `AudioTrack`, audio-surface capability gaps as typed errors, cancel/progress/seed);
- **`Trainer`** — `trainer_conformance` / `TrainerProfile`;
- **`Captioner`** (image→text) — `captioner_conformance` / `CaptionerProfile`;
- **`Transcriber`** (audio→text ASR) — `transcriber_conformance` / `TranscriberProfile`;
- **`VoiceEmbedder`** (speaker identity) — `voice_embedder_conformance` / `VoiceEmbedderProfile`;
- **`AudioTransform`** (voice conversion / stem separation / super-resolution) —
  `audio_transform_conformance` / `AudioTransformProfile` (output cardinality by kind);
- **`AudioEmbedder`** (CLAP-style joint audio↔text) — `audio_embedder_conformance` /
  `AudioEmbedderProfile` (same-dim, L2-normalized, finite vectors).

## Usage

Family crates dev-depend on it and run their real model through the suite:

```rust
// generator, trainer, and captioner conformance:
gen_core_testkit::conformance(
    || registry.load("z_image_turbo", &spec).unwrap(),
    &gen_core_testkit::Profile::cheap(),
);
gen_core_testkit::trainer_conformance(
    || registry.load_trainer("z_image_turbo", &spec).unwrap(),
    &gen_core_testkit::TrainerProfile::cheap(items, out_dir),
);
gen_core_testkit::captioner_conformance(
    || registry.load_captioner("<captioner-id>", &spec).unwrap(),
    &gen_core_testkit::CaptionerProfile::cheap(),
);

// audio contracts — a text→audio generator and the four audio-trait providers:
gen_core_testkit::audio_conformance(
    || registry.load("<audio-generator-id>", &spec).unwrap(),
    &gen_core_testkit::AudioProfile::cheap(),
);
gen_core_testkit::transcriber_conformance(
    || registry.load_transcriber("whisper", &spec).unwrap(),
    &gen_core_testkit::TranscriberProfile::cheap(),
);
gen_core_testkit::voice_embedder_conformance(
    || registry.load_voice_embedder("chatterbox", &spec).unwrap(),
    &gen_core_testkit::VoiceEmbedderProfile::cheap(),
);
gen_core_testkit::audio_transform_conformance(
    || registry.load_audio_transform("openvoice", &spec).unwrap(),
    &gen_core_testkit::AudioTransformProfile::cheap(),
);
gen_core_testkit::audio_embedder_conformance(
    || registry.load_audio_embedder("clap", &spec).unwrap(),
    &gen_core_testkit::AudioEmbedderProfile::cheap(),
);
```

The `*_conformance` entry points run every check and panic with the aggregated failures; the
individual `check_*` functions are public so a provider's own test can target one guarantee at
a time.

## License

Apache-2.0.
