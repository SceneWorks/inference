# Audio Backend Strategy

> **Status:** Accepted
> **Decision date:** 2026-07-18
> **Scope:** The tensor backend for the audio-generation modality (epic sc-12833),
> and the bundle/catalog mechanics that carry it (sc-12901)

## Decision summary

Audio generation is **Candle-native on every platform**. One Candle implementation
of each audio model serves `runtime-cpu`, `runtime-cuda`, **and** `runtime-macos`;
no ONNX/onnxruntime lane and no third tensor backend is introduced. MLX ports of
individual audio models remain sanctioned later, per model, where Apple-silicon
performance demands them — they register natively in the mlx bundle with no extra
mechanics.

Because the macOS bundle's media backend is `mlx`, audio providers ride in a
**dedicated audio section** of `RuntimeCatalog` with its own single declared
backend (`candle` on all three bundles). The audio section is the one sanctioned
cross-backend seam; it is generators-only and validated as strictly as the media
registry. The single-backend invariant on the media, LLM, and snapshot-preparer
registries is unchanged.

## Context

The repository composes two tensor backends: MLX (`runtime-macos` via
`mlx-gen-catalog`) and Candle (`runtime-cuda` / `runtime-cpu` via
`candle-gen-catalog`). `runtime-catalog` enforces that every provider descriptor
in a bundle belongs to the bundle's single declared backend.

The original audio plan assumed an ONNX/ort lane only because the discarded
SoundWorks prototype used ONNX for Kokoro TTS. That was a default inherited from
a throwaway repo, not a decision. Audio is planned to grow to roughly eight
modalities (TTS, music, SFX, voice conversion, ASR-adjacent tasks, …) totalling
on the order of thirty model tasks, and both macOS and Windows/CUDA products must
ship audio, so the backend choice compounds across every future audio story.

Evidence gathered for this decision:

- **The pinned Candle revision already implements audio models.** The workspace
  pins `candle-core` / `candle-nn` / `candle-transformers` at revision
  `1e6aa85e` (enforced by `scripts/check-workspace.py`), and that exact revision
  ships `whisper`, `encodec`, `metavoice`, `quantized_metavoice`, `parler_tts`,
  `mimi`, `snac`, `dac`, `csm`, and `voxtral` in `candle-transformers`. Audio in
  Candle is upstream-supported prior art, not a greenfield bet.
- **Candle covers all three platforms from one implementation.** Candle runs
  CPU, CUDA, and Metal; `runtime-cpu` already lists `aarch64-apple-darwin` as a
  supported target triple. The walking-skeleton model, Kokoro (82 M parameters,
  StyleTTS2 + iSTFT-Net vocoder), synthesizes in real time on CPU in existing
  Rust implementations, so even the CPU path is sufficient for first audio on
  every platform.
- **Kokoro has Rust precedent.** Community Rust ports exist both over ONNX
  (`Kokoros`, `kokoro-en` + misaki-style G2P) and over Candle (the `any-tts`
  crate ships a Candle-native Kokoro backend). The G2P/phonemization front-end
  and voice-style vectors are backend-neutral and reusable regardless of tensor
  backend.
- **ort has a supply-chain problem in this workspace.** The `ort` crate's
  default `download-binaries` feature makes `ort-sys`'s build script download a
  prebuilt `onnxruntime` shared library from pyke's CDN at build time. That
  binary is invisible to `cargo deny check sources` (deny.toml pins
  `unknown-registry`/`unknown-git` to `deny` and allows exactly crates.io plus
  the two pinned backend forks), breaks hermetic `--locked` builds, and would
  add an unpinned native artifact to every release. Avoiding the download means
  vendoring or source-building onnxruntime (a large CMake/C++ dependency) per
  platform — a bigger maintenance surface than the model ports it would save.
- **There is no usable MLX audio path in Rust today.** `mlx-audio` (Kokoro et
  al. on MLX) is Python/Swift; nothing consumable from `mlx-rs` exists. An MLX
  audio lane means hand-porting every model onto the pinned `mlx-rs` fork —
  proven feasible in this repo (Wan, Mochi, Flux, …) but roughly doubling the
  per-model cost if required for all of them up front.

## Chosen strategy

1. **Backend:** every shipped audio provider is implemented on the pinned Candle
   revision. On `runtime-cpu`/`runtime-cuda` this is the bundle's own backend;
   on `runtime-macos` the audio lane runs Candle (CPU today; Candle-Metal is an
   implementation option per model) alongside the mlx media graph.
2. **Composition:** audio providers register through the **existing** generator
   contract (`ModelRegistration` / `register_generators!` — no new trait) into a
   **separate audio registry** assembled by an audio composition root owned by
   the audio lane (sc-12835). The audio backend never leaks into
   `mlx-gen-catalog` or `candle-gen-catalog`; bundle inclusion remains a
   deliberate per-bundle edit.
3. **Validation:** `RuntimeCatalog::try_new_with_audio` accepts the audio
   registry plus its declared backend and enforces: non-empty audio backend,
   every audio generator on exactly that backend, generators-only in the audio
   section, no id collisions with media generators, and the same weights-free
   descriptor conformance sweep as media.
4. **MLX escape hatch:** if a specific audio model later justifies an MLX port
   for Mac performance, it ships as an ordinary `mlx` provider — the macOS
   bundle can then either carry it in the media registry (backend matches) or
   the audio lane's backend for that bundle flips once the whole lane migrates.
   Either move is an explicit, test-pinned catalog edit.

### Platform-coverage matrix

| Bundle          | Media backend | Audio lane backend | Audio ships how                                            |
| --------------- | ------------- | ------------------ | ---------------------------------------------------------- |
| `runtime-macos` | `mlx`         | `candle`           | Candle audio catalog in the dedicated audio section        |
| `runtime-cuda`  | `candle`      | `candle`           | Same audio catalog, CUDA device selection per provider     |
| `runtime-cpu`   | `candle`      | `candle`           | Same audio catalog, CPU device                             |
| LLM-only profile (`--no-default-features`) | per bundle | *none* | Audio is part of the media composition profile |

Every bundle gets the **same** audio provider surface from one composition root
— platform differences (device selection, quant tiers) stay inside providers or
are pinned explicitly in bundle surface tests, exactly like the existing NVFP4
difference.

## Rejected alternatives

### ONNX/ort as a single cross-platform audio lane

Fewest ports, but it introduces a **third backend** into a deliberately
two-backend architecture, and its runtime is a prebuilt binary downloaded by a
build script from a third-party CDN — outside `cargo deny [sources]`, outside
`--locked` hermeticity, and outside the pinned-revision discipline that
`check-workspace.py` enforces for MLX and Candle. Escaping the download means
vendoring/building onnxruntime from source on three platforms, which costs more
than it saves. It would also make audio the only modality whose weights, graph
format (protobuf `.onnx`), and execution semantics differ from the rest of the
repo, splitting the snapshot-preparation and conformance story.

### Native two-backend audio (mlx-audio + candle-audio ports per model)

Most symmetric with image/video/LLM, and it keeps the single-backend invariant
untouched. Rejected as the **default** because it roughly doubles the port work
across ~30 planned audio model tasks for little user-visible gain: unlike the
image/video families, the walking-skeleton audio models are small (Kokoro is
82 M parameters) and run in real time on CPU, so MLX's Apple-silicon advantage
is not load-bearing for audio today, and there is no mlx-rs audio ecosystem to
lean on. The strategy keeps per-model MLX ports sanctioned where measurement
shows Mac performance demands them — this alternative is deferred per model,
not forbidden.

### Hybrid — ONNX now, native later

Speed-to-first-audio without the ONNX costs being temporary: the third backend,
the ort provenance problem, and the throwaway integration work all land
immediately, and "native later" means re-porting every shipped model and
re-validating outputs against new goldens. Candle-native is nearly as fast to
first WAV (Kokoro's Candle precedent exists) with none of the debt.

## Consequences and accepted tradeoffs

- `runtime-macos` (with default `media` feature) will additionally compile the
  Candle audio provider graph once sc-12835 lands. Candle CPU compiles cleanly
  on `aarch64-apple-darwin` (it is already a supported `runtime-cpu` triple);
  binary-size and build-time cost is accepted for one shipped audio surface.
- The "one backend per bundle" invariant is refined to "one backend **per
  registry section**": media/LLM/preparers remain single-backend per bundle;
  the audio section carries exactly one declared backend of its own. The
  refinement is enforced in `runtime-catalog`, not waived.
- `RuntimeCatalogSnapshot` gains two additive fields (`audio_backend`,
  `audio_generator_ids`); existing serialized fields are unchanged
  (CONTRIBUTING: serialized compatibility surface, additive only).
- Snapshot preparation for audio weights reuses the Candle preparer path; on
  `runtime-macos` the preparer registry is mlx-only today, so audio snapshot
  preparation on macOS is an explicit integration point for sc-12835 (the
  Candle preparer is in-repo; carrying it in the mac bundle mirrors the audio
  section decision).

## Recommendation for sc-12835 (audio lane scaffold)

Build the Candle audio lane as a sibling of the existing media families:

1. `crates/audio/candle-audio/` — shared audio engine primitives (mel/STFT,
   iSTFT vocoder helpers, WAV encode via `gen_core::AudioTrack`), plus
   `crates/audio/candle-audio-catalog/` — the audio composition root exposing
   `register_providers(builder) -> builder` and `provider_registry()`, mirroring
   `candle-gen-catalog`'s shape. Update `EXPECTED_MEMBER_COUNT` in
   `scripts/check-workspace.py`.
2. Wire that one `provider_registry()` into all three bundles' existing
   `audio_registry()` seams (`runtime-{macos,cpu,cuda}/src/lib.rs`) and pin the
   ordered audio id surface in each bundle's smoke test.
3. Integrate `Modality::Audio` (sc-12834) when it lands: audio providers declare
   it, and `runtime-catalog::validate_audio` tightens to require the audio
   modality inside the audio section and forbid it in the media registry.
4. Keep first-model scope (sc-12836, Kokoro) to: safetensors weights + voice
   style vectors, backend-neutral G2P front-end, Candle StyleTTS2 decoder +
   iSTFT-Net vocoder, CPU device first (real-time at 82 M), CUDA/Metal device
   selection as follow-ups measured per platform.
