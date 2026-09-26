# SheetSage2 utility assessment (sc-23003)

Bounded exploration for epic [sc-22988](https://app.shortcut.com/trefry/epic/22988) (YuE2,
noncommercial music experimentation), story
[sc-23003](https://app.shortcut.com/trefry/story/23003). It answers three questions: what the released
SheetSage2 utility actually does, what a native Candle version would take, and whether to adopt it
as a reusable utility or only as the YuE2 cover dependency already planned in
[sc-22996](https://app.shortcut.com/trefry/story/22996).

Written 2026-09-26 by an agent (claude). The adoption decision belongs to the owner. This document
recommends; it does not commit anyone to building a standalone utility.

## How to read the evidence tags

Every capability claim carries one of three tags:

- **OBSERVED**: this exploration ran it. The artifact is cited under
  [`scripts/reference/sheetsage2/artifacts/`](../../scripts/reference/sheetsage2/artifacts), produced
  by [`run_experiment.py`](../../scripts/reference/sheetsage2/run_experiment.py). The
  [README](../../scripts/reference/sheetsage2/README.md) has the commands and environment.
- **DOCUMENTED**: read from upstream code or docs at a pinned revision, cited as `file@rev`.
- **UNTESTED**: not exercised here. The tag gives the exact blocker and the next test.

Revision shorthands:

- `SS2@eab522a` = `m-a-p/SheetSage2@eab522a8168e8b8b8c4856bf8609cd86198f01fe` (release, 2026-09-09)
- `SS2@4f89269` = upstream `main` on 2026-09-26
- `MERT@d8ba1c7` = `m-a-p/MERT-v2-FullSong@d8ba1c745e733b3908ce6ad16ebeb17ac7600a42`
- `YuE@92a73cc` = `github.com/multimodal-art-projection/YuE@92a73cc7652fcc1f937855e4b765e0a0edd7ff2e`

Absolute timings and RSS values are observations on one machine: Apple M5 Max (Mac17,6), 18
logical CPUs, 128 GiB, macOS 26.6.2, torch 2.8.0 on the **CPU** device in fp32 with 8 threads. They are not
portable requirements. They also predict nothing about native Metal or CUDA cost.

## Recommendation: **revise**

**Revise.** Port SheetSage2 and MERT-v2-FullSong natively once, as a reusable
music-transcription provider. Do that inside the already-planned sc-22996 work, and re-slice
sc-22996 so the port stands alone. Do **not** build the standalone utility's product surface (job
API, UI, piano and score rendering, tensor exports) until the owner decides to.

Why this recommendation:

1. **The capability is real and the port is tractable.**
   - Quality on clean, controlled audio is high. On the ground-truth synthetic clip, OBSERVED:
     - melody note F1 0.95 at 50 ms onset tolerance, exact pitch (0.99 at 100 ms, pitch class);
     - chord major/minor overlap 0.99;
     - beat F1 0.98, downbeat F1 0.96, tempo exact;
     - key correct.
     Source: [`artifacts/evaluation.json`](../../scripts/reference/sheetsage2/artifacts/evaluation.json).
   - The model closure is two safetensors files, about 2.76 GB in fp32. It runs well within a laptop
     budget on CPU: about 15–17 s per 300 s window and a 4.7–5.4 GiB process-tree peak (OBSERVED,
     [`artifacts/measurements.json`](../../scripts/reference/sheetsage2/artifacts/measurements.json)).
   - Every tensor op maps to Candle primitives the repo already has or has close templates for (see
     [native portability](#native-portability)).
2. **The port is needed anyway.** Epic requirement E5 and sc-22996 already require native
   SheetSage2 + MERT2 for covers. A standalone utility adds almost nothing on the model side. What it
   adds is product surface: a job type, a review/preview UI, and exports. The only question is whether
   that surface is worth building, and that is a product decision.
3. **Hold the standalone surface for now:**
   - **License.** Both weight sets are CC BY-NC 4.0 (see [licensing](#licensing)). A standalone
     utility could only ship behind the same experimental/noncommercial route as YuE2 (E2). It cannot
     be a general-purpose "audio to sheet music" feature.
   - **Real-recording quality needs review tooling.** On the public-domain vocal-plus-band recording
     (OBSERVED, [`artifacts/real_full/`](../../scripts/reference/sheetsage2/artifacts/real_full)):
     - meter, key, tempo feel and melodic contour were right;
     - harmony collapsed to C major throughout;
     - the vocal line was probably written an octave high.
     Any user-facing utility therefore needs the ABC review and editing surface that sc-22997/sc-23000
     are building for YuE2. It should reuse that surface, not duplicate it.
   - **Upstream's renderer is not shippable as is.** It drives headless Chromium through Playwright
     (DOCUMENTED, `rendering_sheetsage2.py@SS2@eab522a`). Nothing in the native runtime can take that
     dependency. If the owner adopts the utility, render in the SceneWorks web UI with abcjs (MIT)
     instead (see [previews](#score-and-audio-preview-and-export)).
4. **Stop** pursuing these as product features, because no workflow uses them:
   - the tensor/embedding/logit exports (`export_embeddings/logits/scores`);
   - the `preset="paper"` benchmark path.

   They stay useful as parity-fixture tooling.

The executable slices are in [proposed slices](#proposed-implementation-slices-owner-decision), split
into what sc-22996 should absorb and what only a standalone decision would add.

## Pinned identities and upstream drift

| Asset | Pin | Contents | Evidence |
|---|---|---|---|
| SheetSage2 | `SS2@eab522a` | `model.safetensors` 228,738,564 B, LFS sha256 `b235f68091a5f5b644000f2b5acb57d1e70432aca2b34ab1b9cf27236e1f4274`, 355 F32 tensors (LoRA adapters + head); remote code (`*.py`); `render_assets/` (abcjs, font, 88 piano MP3s) | DOCUMENTED (Hub tree API at the pin) |
| MERT-v2-FullSong | `MERT@d8ba1c7` | `model.safetensors` 2,529,812,848 B, sha256 `e6dd2ab187d6dd62b6521cd7d8f932e237acf0c5757745a7232082e28391350d`, 876 F32 tensors, 632,295,808 parameters (`weights_manifest.json`) | DOCUMENTED; the SheetSage2 loader checks this sha256 and revision from its own `config.json` (`base_model_revision`, `base_model_sha256`), and this run passed it (OBSERVED, every model case) |
| YuE cover workflow | `YuE@92a73cc` | `docs/covers.md`, `skills/yue2-music/` | DOCUMENTED |

Neither Hub repository is gated. The API reports `gated: false` for both on 2026-09-26. The model
card tells users to sign in anyway, but no gate acceptance was needed (OBSERVED: this machine's
existing Hub token downloaded both pinned snapshots).

**Upstream has moved since the planning snapshot.** Nine SheetSage2 commits followed the release,
and MERT has one README-only commit. The weight LFS objects are unchanged in both repos (DOCUMENTED,
tree diff between `eab522a` and `4f89269`, and between `d8ba1c7` and `f58c87c`). The code
difference is:

- a new `chord_spelling_sheetsage2.py`;
- changed `exports_sheetsage2.py` and `tokenization_sheetsage2.py` ("Correct chord spelling with
  local keys and canonicalize key labels", "Fix chord spelling for non-C candidate roots").

OBSERVED effect:

- The model's token output is identical under both code revisions:
  `head_code_reproduces_release_tokens` in `evaluation.json`.
- In a C-major recording the outputs are byte-identical.
- In a flat key they are not. On the Eb-major synthetic clip the release writes `K:Eb` with chord
  symbols `D#`, `G#` and `A#`, and key label `D#:major`. The head writes `Eb`, `Ab`, `Bb` and
  `Eb:major` ([`artifacts/synth_eb_release/`](../../scripts/reference/sheetsage2/artifacts/synth_eb_release),
  [`artifacts/synth_eb_head/`](../../scripts/reference/sheetsage2/artifacts/synth_eb_head)).
- That release ABC is internally inconsistent: a flat key signature with sharp chord names. It is
  exactly what a full-score (`cot=full`) cover would feed to YuE2.

**Consequence for sc-22996:** port the head spelling logic, or pin a revision at or after
`55bfe14e`, and state which. The weights pin can stay at the release.

## What the utility is

### Model

DOCUMENTED from `modeling_mert2.py`, `modeling_sheetsage2.py`, `config.json`@`SS2@eab522a` and
`config.json`@`MERT@d8ba1c7`. The shapes were confirmed from the safetensors headers.

1. **Front end.** The input is 24 kHz mono float with no amplitude normalisation. It goes through a
   torchaudio `Spectrogram` with n_fft 2048, win 2048 (Hann, stored in the checkpoint as
   `feature_extractor.spectrogram.window`), hop 240 and power 2. Then `MelScale` with 128 bins,
   whose filterbank is stored as `feature_extractor.mel_scale.fb [1025,128]`. Then
   `AmplitudeToDB(power, top_db=None)`. The last frame is dropped, and each bin is normalised with
   the checkpoint's `mel_mean`/`mel_std`. The front end is forced to fp32 even when the model is cast
   to a lower precision. This gives 100 frames/s.
2. **MERT-v2-FullSong encoder** (632.3M parameters, hidden 1024):
   - Three ConvNeXt-v2 1-D stages with GlobalResponseNorm. They use LayerNorm plus a Conv1d with
     kernel 2 as resampler at strides 1/2/2, depths 3/4/5, and channels 128→512→1024. The output is
     25 Hz frames.
   - Then 24 Conformer blocks. Each has a macaron half-step FFN (4096) and 16-head non-causal
     self-attention with rotate-half RoPE (base 10000). The conv module is pointwise ×2, GLU,
     depthwise k31, LayerNorm, GELU, pointwise. The block ends with a final LayerNorm.
   - This is **not** a HuBERT/wav2vec2-style encoder. It has no raw-waveform conv feature extractor
     and no positional conv.
3. **SheetSage2 head** (44.6M parameters of its own; 676,887,449 in the merged model, OBSERVED
   `case.json` `parameters`):
   - Rank-64 LoRA adapters (alpha 128) on each Conformer's q/k/v/out projections. They are merged
     into the MERT weights at load: on CPU, in fp32, before any cast.
   - A softmax-weighted mix of the 25 hidden states (input plus 24 blocks, `layer_weight [25]`).
   - `encoder_projection` Linear 1024→512.
   - A Hugging Face **BART decoder**: 6 layers, d_model 512, 8 heads, FFN 2048, GELU, post-LN, with
     `layernorm_embedding`. It has learned positions `embed_positions [5122,512]` (5120 plus BART's
     offset of 2) and biased projections.
   - The embedding and output projection are tied (`token_embedding [31678,512]`).
   - Decoding is **greedy argmax under a hand-written grammar mask** (`PromptGrammarState`) with a
     KV cache (`generation_sheetsage2.py`). No sampling or RNG is involved, so native output can
     be compared token for token.
4. **Tokenizer.** The vocabulary is computed, not loaded: prompts, sub-beat shifts (257), 30,000 time
   tokens at 100 Hz over 300 s, meters (192), eighth positions (256), structure labels (23), 24
   keys, 25 maj/min and 361 full chord labels (15 qualities × 12 roots plus inversions, and `N`),
   pitches (256) and durations (24). Block sizes were OBSERVED by instantiating the tokenizer. It
   totals 31,678 ids and is pinned by `tokenizer_fingerprint 5ba3325af0344c7f`
   (`tokenization_sheetsage2.py`). It uses `mir_eval.chord` for chord parsing.

### Inference pipeline

DOCUMENTED from `pipeline_sheetsage2.py` and `audio_sheetsage2.py`@`SS2@eab522a`.

- **Decode.** Every window is zero-padded to the **fixed 300 s** context, so a 12 s clip costs about
  the same encoder time as a 300 s clip (OBSERVED: 14 s per window for 12 s of silence and for a
  39.4 s clip, [`failures/case.json`](../../scripts/reference/sheetsage2/artifacts/failures/case.json)
  and [`synth_full/case.json`](../../scripts/reference/sheetsage2/artifacts/synth_full/case.json)).
- **Longer songs.** Songs longer than 300 s use overlapping windows. The default preset uses 200 s
  overlap and 100 s look-ahead. Each window after the first is conditioned on a token prefix rebuilt
  from the events already accepted, then stitched. **UNTESTED**: this lane bounds clips to ≤60 s,
  so the multi-window path never ran. Next test: a >300 s public-domain recording, checking stitched
  event continuity and the `Overlap prefix fills the context` error.
- **Audio input.** Files are decoded with the `ffmpeg` CLI (`-ac 1 -ar 24000 -f f32le`, using
  FFmpeg's resampler). Arrays are channel-averaged and resampled with
  `torchaudio.functional.resample`. `preset="paper"` decodes through torchaudio's FFmpeg-library
  backend instead.
- **Outputs.** Events go to `export_result`, which writes:
  - `events.json` / `events.tsv`;
  - `*.lab` (beat, downbeat, chord, key, structure, melody_{full,vocal,instrumental}, rhythm_events);
  - MIDI (`transcription.mid` = melody plus chord accompaniment, `melody*.mid`, `chords.mid`,
    `notation/song_melody.mid`);
  - `playback.json`;
  - `score.abc`, built by `notation_sheetsage2.py`.

  `score.abc` is an independently validated two-voice ABC dialect: `V: Vocal` / `V: Ins`,
  `L:1/16`, chord symbols in the Vocal voice. If ABC cannot be built, the MIDI and annotations are
  still written and `abc_error` is set.

### Upstream code size

`SS2@eab522a` has 5,458 lines of Python. The model side is 809 lines (MERT2 361, SheetSage2 448).
Decode and grammar are 634 lines, the tokenizer 732. The symbolic/notation side is 1,570 lines for
`notation_sheetsage2.py` plus exports and MIDI at 292. At `SS2@4f89269` the chord spelling adds 196
lines. The post-processing is the larger half of a faithful port.

## Inputs and outputs mapped to product workflows

| Released surface | Tag | Product workflow it serves | Recommendation |
|---|---|---|---|
| Audio (path/bytes/stream, or waveform + rate) → melody (vocal + instrumental voices), chords (full vocabulary), beats/downbeats/meter, key, structure | OBSERVED (all model cases) | YuE2 cover source transcription (sc-22996, E5). Standalone: "analyze a reference track" (tempo/key/chords) for any audio-lane user | Port natively (sc-22996) |
| `score.abc` full (with chord symbols) | OBSERVED [`real_full/score.abc`](../../scripts/reference/sheetsage2/artifacts/real_full/score.abc) | YuE2 `cot=full` input; ABC review/edit (sc-22997) | Port; must stay compatible with YuE2's two-voice dialect |
| `score.abc` melody-only (`melody_only=True`) | OBSERVED [`real_melody/score.abc`](../../scripts/reference/sheetsage2/artifacts/real_melody/score.abc) | YuE2 `cot=melody` covers (the upstream-recommended cover route) | Port. It is a pure post-processing switch: the model tokens are identical with and without it (OBSERVED `melody_only_changes_tokens: false`) |
| MIDI (combined / per part) | OBSERVED (`*.mid` in each case) | Export to a DAW; source for piano preview | Port (a small SMF writer) |
| Timed annotations (`events.json`, `*.lab`) | OBSERVED | Provenance of a cover, UI timelines, invariant checks for edits | Port (JSON/LAB are lossless raw event views) |
| Piano WAV preview (mix/melody/vocal/instrumental/chords) | OBSERVED [`real_full/case.json`](../../scripts/reference/sheetsage2/artifacts/real_full/case.json), [`render_only/case.json`](../../scripts/reference/sheetsage2/artifacts/render_only/case.json) | "Listen to the transcription" during review | Do not port the renderer; if adopted, do it app-side (see below) |
| Score PDF/SVG/PNG | OBSERVED ([`real_full/score_001.png`](../../scripts/reference/sheetsage2/artifacts/real_full/score_001.png)) | Printable or shareable lead sheet | Do not port; if adopted, render ABC with abcjs in the web UI |
| Tensor exports: MERT frame features, mixed/encoder memory, decoder states, per-step logits/scores | OBSERVED shapes in [`synth_full/case.json`](../../scripts/reference/sheetsage2/artifacts/synth_full/case.json) (`[1,7500,1024]`, `[1,7500,512]`, `[1,64,512]` batches) | None found in the product. Useful for native parity fixtures | Stop as a product feature; keep as reference tooling |
| `save_pretrained` merged standalone snapshot | OBSERVED: the merged snapshot reproduces the adapter-load tokens exactly (`merged_snapshot_reproduces_adapter_load_tokens: true`); 2,708,204,284 B `model.safetensors` | Offline packaging option | Not needed natively: merge at load (see model management) |
| `preset="paper"` (benchmark windowing, torchaudio decode) | OBSERVED failure (see failure cases) | Benchmark reproduction only | Stop |

## Experiment results

The details are in the [experiment README](../../scripts/reference/sheetsage2/README.md). Fixtures:

- `nav_ssb`: the first 60 s of a public-domain U.S. Navy Band recording of "The Star-Spangled
  Banner", solo soprano with band. Provenance and digests are in
  [`artifacts/fixtures.json`](../../scripts/reference/sheetsage2/artifacts/fixtures.json).
- `synth`: a deterministic 39.4 s lead sheet with exact ground truth
  ([`synth_truth.json`](../../scripts/reference/sheetsage2/artifacts/synth_truth.json)), plus
  `synth_eb`, the same clip transposed to Eb major.

### Per-case observations (OBSERVED)

| Case | Result | Wall s (tree) | Transcribe s | Peak tree RSS GiB |
|---|---|---:|---:|---:|
| `real_full` (first online load; full prompts; piano WAV + PDF/SVG/PNG) | 27 bars, 3/4, `K:C`, 76 vocal notes, 0 instrumental, ABC valid, no warnings; rendering succeeded | 19.3 | 16.9 | 5.36 |
| `real_melody` (offline, `melody_only`) | Same notes and bars; no chord symbols | 18.2 | 15.8 | 4.76 |
| `synth_full` (offline; tensor exports; encoder timing) | Scores below; encoder alone 13.9–14.0 s per 300 s window; 340 tokens | 45.9 | 15.6 | 4.74 |
| `synth_merged` (`save_pretrained` + reload merged, offline) | Tokens identical to `synth_full`; save 0.6 s, merged reload 0.25 s | 18.8 | 15.3 | 5.39 |
| `synth_eb_release` / `synth_eb_head` | Tokens identical; chord spelling differs (see drift above) | 18.3 / 17.9 | — | 4.73 |
| `real_full_head` (head code, online fetch of the code) | Byte-identical ABC and LAB to `real_full` | 21.5 | 16.2 | 4.76 |
| `failures` (see below) | All probes behaved as listed | 31.3 | — | 4.72 |
| `render_only` (upstream `render.py`, no model) | Vocal + chords piano WAVs, 2 SVG pages; "No instrumental track is available; skipped." | 2.1 | — | 0.88 |

Notes on the table:

- Model load from a warm cache was 1.6 s. The very first online load, which included a one-time
  transformers cache migration, was 15.4 s.
- About 14 s of each ~15–17 s transcription is the MERT2 encoder over the padded 300 s window. The
  remaining roughly 1.5–2 s is grammar-masked greedy decoding of 340–456 tokens.
- The torch CPU peak stayed at about 4.7–5.4 GiB. The fp32 weights alone are about 2.7 GB.

### Accuracy on the ground-truth clip (OBSERVED)

Source: [`evaluation.json`](../../scripts/reference/sheetsage2/artifacts/evaluation.json), scored
with mir_eval 0.8.2.

| Measure | C major (`synth_full`) | Eb major (`synth_eb_*`) |
|---|---:|---:|
| Melody note F1, onset ±50 ms, exact pitch | 0.953 | 0.935 |
| Melody note F1, onset ±50 ms, pitch class | 0.953 (no octave errors) | 0.935 |
| Melody note F1, onset ±100 ms, pitch class | 0.991 | — |
| Chords, maj/min weighted overlap | 0.990 | 0.990 |
| Beat F1 / downbeat F1 | 0.982 / 0.963 | 0.982 / 0.963 |
| Tempo / meter / key | 100 BPM / 4/4 / C major, all correct | 100 / 4/4 / Eb (release labels it `D#:major`) |

The synthetic melody was voiced as `Ins` (instrumental), not `Vocal`. That is reasonable for a
synthetic timbre. It matters for YuE2 because the two ABC voices are conditioned separately.

### Real recording (OBSERVED; no ground truth, so these are qualitative)

What SheetSage2 got right:

- Meter 3/4 (correct for the anthem).
- Key C major.
- The opening contour G–E–C–E–G–C′–E′–D′–C′, including the chromatic F♯ on "twilight's last
  gleaming". This is the anthem melody in C.
- It rendered to a readable two-staff score
  ([`score_001.png`](../../scripts/reference/sheetsage2/artifacts/real_full/score_001.png)).

What it got wrong:

- **Harmony collapsed.** Every chord is `C`, `C:maj7/5` or `C:maj7/7`
  ([`chord.lab`](../../scripts/reference/sheetsage2/artifacts/real_full/chord.lab)). The IV/V
  harmony of the anthem is missing. A full-score (`cot=full`) cover built from this would lock the
  accompaniment to one chord. This is consistent with the upstream recommendation to prefer
  melody-only covers.
- **The octave is probably wrong (+12).** The transcribed vocal notes are G5–E5–C5… A spectral check
  of the first five notes found most energy one octave **below** each transcribed pitch. Confidence
  in this finding is medium: the band accompaniment also contributes energy. The synthetic clip had
  no octave errors.
  - Next test: a public-domain a cappella or vocal-stem recording with a known score (for example
    `File:EternalFather USNavyBand acapella.ogg` on Commons), scored with exact-pitch versus
    pitch-class F1.
  - Why it matters: the register of the ABC melody reaches YuE2 directly.
- **The pickup was not recognised.** The anacrusis is written as a rest-filled first bar
  (`z8g2e2`).

### Failure and edge cases (OBSERVED)

Source: [`failures/case.json`](../../scripts/reference/sheetsage2/artifacts/failures/case.json).

- An input shorter than 1,025 samples, or a non-finite one, raises
  `ValueError: Audio must contain at least 1025 finite samples at 24 kHz`.
- An array without `sampling_rate` raises `ValueError`.
- Undecodable bytes raise a `ValueError` carrying FFmpeg's stderr.
- **12 s of digital silence succeeds silently.** It returns a 7-bar ABC with beats, key and
  structure hallucinated, zero melody notes, no warning and no error. This happens in both full and
  `melody_only` mode. `melody_only` only raises when no ABC can be built at all.
  - Consequence for sc-22996: the native path must add its own "no or too few melody notes"
    refusal. Upstream will happily hand YuE2 an empty melody score.
- **`preset="paper"` failed:** `ValueError: Unsupported backend 'ffmpeg' specified; please select
  one of [] instead.`
  - Cause: torchaudio 2.8's FFmpeg backend loads FFmpeg's shared libraries, and this machine has
    FFmpeg 9.0.1. Upstream specifies 6.1.
  - The default preset uses the `ffmpeg` CLI and worked with 9.0.1.
  - **UNTESTED**: the paper preset itself. Blocker: no FFmpeg 6.x shared libraries installed. Next
    test: `brew install ffmpeg@6`, put its `lib` on `DYLD_FALLBACK_LIBRARY_PATH`, rerun the
    `failures` case. This is low value, because the preset only serves benchmark reproduction.
- **The YuE2 ABC-dialect validator was not run.**
  - What the validator is: `skills/yue2-music/scripts/abc_tools.py`@`YuE@92a73cc`,
    standard-library-only, sha256 `ea04b922dacebec7ad257a2f8d83bdb5dfecb7a23110c1a3121c5c41c313930e`.
    It is the hand-off contract check upstream's cover skill applies to SheetSage2 output. It
    requires, for example, the exact `V: Vocal …` / `V: Ins …` lines, chord symbols only in Vocal,
    and identical bar grids.
  - What happened: running it was **denied by this session's permission policy** (fetch and execute
    an external script).
  - **UNTESTED**. Next test: with owner approval, run
    `python abc_tools.py inspect <case>/score.abc` on each committed `score.abc`. Alternatively,
    port its rules as the native validator in sc-22996 and assert them on these fixtures.
  - The committed ABC files do carry the required header lines (OBSERVED by inspection).
- **`melody_only` and full ABC differ in more than chord symbols.** With chord symbols removed, the
  two are not byte-identical. Where a chord change split a rest, `z2z2` becomes `z4`
  (`melody_only_equals_full_without_chord_symbols: false`). The sounding content is the same. A
  native serializer must reproduce this rest merging to be byte-exact.

### Offline and determinism (OBSERVED)

- Every case after the first ran with `HF_HUB_OFFLINE=1` and `local_files_only=True` and succeeded.
- Rendering uses only bundled assets. The browser context is created offline.
- The merged snapshot reloads with `local_files_only`.
- On CPU fp32, repeated runs of the same input produced identical tokens across processes and across
  adapter-load versus merged-load.

**UNTESTED: bf16.** CPU autocast is disabled upstream (`inference_autocast` is CUDA-only), so every
run here was fp32. The upstream default is bf16 autocast on CUDA, and its benchmarks were measured
that way.
- Blocker: the lane is CPU only.
- Next test: the same fixtures on a CUDA runner with `dtype="bf16"`, comparing token identity with
  the fp32 artifacts. Expect divergence. The native precision policy (E8) needs the measured drift.

**UNTESTED: Metal and CUDA memory and throughput.** Blocker: CPU-only lane, and the GPU is reserved.
Next test: the native port's parity harness on Metal and CUDA. Upstream torch MPS is not a proxy for
the native cost.

## Native portability

This maps each stage to what already exists in the inference repo. The survey was of branch
`feature/sc-22988-yue2` @ `d5b18019b` with Candle pinned at `1e6aa85`. Nothing for YuE2,
SheetSage2 or MERT exists in any crate yet.

| Stage | Reusable today | Gap |
|---|---|---|
| Resample and downmix to 24 kHz mono | `candle_audio::dsp::resample_sinc_hann`, a port of the `torchaudio.functional.resample` defaults already used by YuE1 `icl.rs`. Container decoding happens in the SceneWorks host (`gen_core::AudioTrack`); inference has no FFmpeg | Upstream's file path resamples with FFmpeg's resampler, so parity fixtures should inject 24 kHz mono arrays, not files |
| STFT, power, mel, dB | `candle_audio::dsp::{hann_window, stft}` (`center=True` reflect, matching torchaudio framing); `mel.rs::MelFilterbank` is HTK with `norm=None`; the CLAP `10·log10(max(x,1e-10))` matches `AmplitudeToDB(top_db=None)`; the chatterbox S3 tokenizer already drops the last frame | Load `window`, `fb`, `mel_mean` and `mel_std` straight from the MERT checkpoint instead of rebuilding them. `dsp::stft` is host-side and radix-2 with no cached twiddles, which is slow for 300 s (30,000 frames). Precompute twiddles or move the STFT to the device |
| ConvNeXt-v2 1-D subsampler | Private `vocos.rs::ConvNeXtBlock` (depthwise k7, LN, MLP) is the template; the depthwise conv fast path covers k7 and k31 on all backends | GlobalResponseNorm (1-D, L2 over time) is new; so is the kernel-2 resampling conv |
| Conformer ×24 | Chatterbox S3 tokenizer `Block` (non-causal MHSA with rotate-half RoPE, theta 10000) is the closest attention; `candle_nn::rotary_emb::rope`; `flow_encoder.rs` gives the block skeleton | The macaron FFN and conv module (GLU + depthwise k31 + LN) are new. **Attention over 7,500 frames must be chunked or fused.** A materialised 16×7500² score tensor is about 3.6 GB fp32 per layer. Use `candle-gen` `sdpa_budgeted_bhsd`, `candle-llm` `sdpa`, or the query-chunked loop in `candle-audio-yue/src/hubert.rs` |
| LoRA merge at load | `candle-audio-stable-audio-3/src/adapters.rs` (`SimpleBackend` wrapper folding `(α/r)·B@A`) | Must merge in fp32 before any cast (upstream enforces this) |
| Layer mix + projection | Trivial | — |
| BART decoder, greedy + grammar | Templates only: `candle_transformers` `trocr.rs` (learned positions with offset 2, post-LN, self-KV cache, but no biases, no `layernorm_embedding`, and cross-attention K/V recomputed every step) and `marian.rs` (biases, both caches, but sinusoidal positions); the `candle-audio-whisper` decode loop gives the cancel/mask/argmax pattern | A BART decoder with biases, `layernorm_embedding`, learned offset-2 positions and cached cross-attention K/V. The grammar state machine (`PromptGrammarState`, about 110 lines) is new |
| Tokenizer, event decode, window plan and stitching | None | Pure Rust port, pinned by `tokenizer_fingerprint` |
| LAB/JSON/MIDI export | None (no MIDI writer in the repo) | A small SMF writer. Upstream uses pretty_midi at resolution 960 |
| ABC notation builder and validator | None | The largest symbolic piece: `notation_sheetsage2.py` 1,570 lines, plus the head chord spelling. Byte-exact parity against the committed `score.abc` fixtures is achievable because the input tokens are deterministic |

Architecture notes:

- The brief's hope that YuE1's HuBERT branch or a "MERT/HuBERT/wav2vec-style" encoder would carry
  over is **mostly wrong** for MERT-v2. Only the chunked non-causal attention loop and the idea of
  mixing all hidden states transfer. MERT v1 was HuBERT-style; MERT-v2 is mel + ConvNeXt +
  Conformer (DOCUMENTED, `modeling_mert2.py`@`MERT@d8ba1c7`).
- MERT2 is a **bidirectional** encoder returning continuous features. It is not YuE2's causal
  semantic tokenizer. The YuE skill warns against feeding MERT features to YuE2 as tokens
  (DOCUMENTED, `skills/yue2-music/SKILL.md`@`YuE@92a73cc`). So this port shares nothing with YuE2's
  own codec path.

## Offline use and model management

- **Closure: two repositories, both conditional cover dependencies (E7).** For SheetSage2 the
  closure is `model.safetensors`, `config.json` and `processor_config.json`; the Python code is not
  needed natively. For MERT the closure is `model.safetensors` and `config.json`.
  `render_assets/` is only needed if upstream rendering is ever used. Total weights are about
  2.76 GB fp32 (DOCUMENTED sizes).
- **Integrity.** SheetSage2's `config.json` names the MERT parent's revision and sha256. The native
  loader should enforce the same pin, using the `candle-audio-stable-audio-3` `verify_file_pin`
  pattern. That is the only existing audio-lane integrity check.
- **Adapter versus merged.** Keep the upstream adapter layout and merge at load. The merge is 96
  rank-64 products at 1024², trivial next to encoder inference. The merged snapshot is
  token-identical (OBSERVED), but it is a derived 2.7 GB artifact. Distributing it would be
  rehosting a modified NC work, which E7 allows only under a recorded distribution basis.
- **Downloads.** Inference never self-fetches (epic 13657 policy). The SceneWorks host downloads
  both pinned snapshots as components of the cover provider through `LoadSpec` components. That is
  sc-22989 (assets) and sc-22998 (catalog) work.
- **Offline behaviour matches upstream.** No network access is needed once snapshots are cached
  (OBSERVED for the reference path).
- **Precision tiers.** Upstream ships fp32 weights only. The bf16/q8/q4 behaviour of a native port
  is E8 engineering work with no upstream reference (**UNTESTED**; the blocker is the same as bf16
  above).

## Licensing

These claims were read from the files at the pins. **PROVISIONAL**: an agent read them and no human
has signed them off, the same status as `docs/licensing/`.

- **SheetSage2 weights: CC BY-NC 4.0.**
  - The model card front matter reads `license: cc-by-nc-4.0`, and the card says "Weights:
    CC BY-NC 4.0", linking to `LICENSE` (`README.md`@`SS2@eab522a`).
  - Caveat: the repository's `LICENSE` is **byte-identical to MERT's**. Its operative text names
    only "The MERT2-30s and MERT2-FS checkpoint weights in model.safetensors". It does not name
    SheetSage2's own adapter and decoder weights. The card declaration is the operative identifier.
    A human should confirm that reading. Declare `cc-by-nc-4.0` against the existing
    `CC_BY_NC_4_0` family (`crates/contracts/gen-core/src/license/families.rs`).
- **MERT-v2-FullSong weights: CC BY-NC 4.0.** The card front matter reads `license: cc-by-nc-4.0`,
  and `LICENSE`@`MERT@d8ba1c7` reads "licensed under Creative Commons Attribution-NonCommercial 4.0
  International (CC BY-NC 4.0)". Attribution must identify MERT2, the model name and the repository.
- **SheetSage2 and MERT code: no code license is stated.** The `*.py` files have no license header,
  and neither repository has a code license. The THIRD_PARTY notices list dependency licences only.
  The only repository-level declaration is the card's `cc-by-nc-4.0`. A native Rust port is a
  reimplementation, but porting the notation logic closely enough for byte-exact ABC could count as
  a derivative of that code. That would change nothing in practice: the weights are already
  noncommercial and the provider can only run under E2. Record this as an open item on the
  component license rows. It is not a blocker.
- **YuE source (covers guide, skill, `abc_tools.py`): Apache-2.0** (`LICENSE`@`YuE@92a73cc`). YuE's
  `MODEL_LICENSE` covers YuE2-3B, YuE2-Vae and YuE2-Vae-legacy only, not SheetSage2 or MERT.
- **Bundled third-party material in `render_assets/`.** This only matters if upstream rendering
  assets are ever shipped:
  - abcjs 6.6.3 is MIT, "Copyright (c) 2009-2024 Paul Rosen and Gregory Dyke" (`LICENSE.abcjs`).
  - The DejaVu Sans font is under the Bitstream Vera license, with DejaVu changes in the public
    domain (`LICENSE.font`).
  - The FluidR3 GM piano samples by Frank Wen are **CC BY 3.0 US** (`soundfonts/ATTRIBUTION.md`).
  - Playwright is Apache-2.0 and Chromium is under its own notices (`THIRD_PARTY_NOTICES.md`).
- **Python dependencies:**
  - Transformers (the BART decoder implementation) is Apache-2.0 and PyTorch is BSD-style, as
    stated in `THIRD_PARTY_NOTICES.md`.
  - mir_eval, pretty_midi and mido were not license-checked. They are reference-only and not needed
    natively.
- **Outputs.** CC BY-NC 4.0 is silent on outputs. The repo's licence table therefore records no
  `NonCommercialOutputs` term for this family (`families.rs` module docs). E2 still keeps the whole
  V2 cover route experimental and noncommercial.

## Overlap with sc-22996, sc-22989 and sc-22998

- **sc-22996 already owns the whole native model port.** Its AC asks for vocal/instrumental melody,
  chord, beat, key and structure, full and melody-only ABC, MIDI and annotation exports, offline
  running, unloading before generation, and persisted provenance. That is the entire model and
  symbolic surface above. A standalone utility needs **no second port**.

  Recommended changes to sc-22996 (for the owner or planner to apply):
  1. **Slice it.** A single story that ports two models, a 1.6k-line notation engine and the cover
     workflow does not match the one-review/one-fix cost model. Split it into the native port
     slices below, plus the cover-integration story that remains.
  2. **Add a reusable provider contract.** Register the transcriber as its own provider, not as a
     private stage of the YuE2 engine. A standalone utility would then be a thin SceneWorks-side
     surface later. Cost now is one descriptor and one registration.
     - Nothing in the existing gen-core contracts fits exactly. `Transcriber` is speech-to-text
       with language and timestamp options; `AudioTransform` is audio-to-audio.
     - A `MusicTranscriber` sibling contract is the likely shape: `AudioTrack` → ABC text + events +
       MIDI parts + LAB + warnings/diagnostics.
  3. **Add AC the experiment showed are needed:**
     - an explicit refusal or warning for empty or too-sparse melodies (silence passes upstream);
     - key-aware chord spelling, meaning the head code revision or later;
     - a statement of the voice assignment (Vocal/Ins) and octave handling for the ABC handed to
       YuE2;
     - token-exact parity on the committed CPU fp32 fixtures, then measured bf16 drift.
  4. **Keep MERT2 embeddings out of scope.** No workflow needs them. The epic already excludes
     standalone MERT2 understanding products.
- **sc-22989 (assets)** should add the two conditional components with their pins and sha256 values
  from the [identity table](#pinned-identities-and-upstream-drift): `SS2@eab522a` weights, or a
  later revision if the head code is ported (the weights are identical), and `MERT@d8ba1c7`. They
  are marked cover-only, not generation-mandatory (E7), with no rehosting.
- **sc-22998 (catalog)** should add CC BY-NC 4.0 component license rows for both, including the
  SheetSage2 `LICENSE`-scope caveat and the undeclared-code-license note. The cover provider's
  disclosure must derive `NonCommercialWeights` from them.

## Score and audio preview and export

- **ABC, MIDI, LAB and JSON export** are cheap once the symbolic port exists (OBSERVED outputs above)
  and belong to sc-22996's persisted artifacts.
- **Score preview.** Upstream renders ABC with abcjs 6.6.3 inside headless Chromium
  (OBSERVED rendering works). SceneWorks already has a web UI, so if the owner adopts the utility,
  the natural implementation is abcjs (MIT) in the app's own webview. That needs no Playwright, no
  second browser, and no rendering in inference. PDF/PNG become browser print or export.
- **Piano audio preview.** Upstream synthesises in an `OfflineAudioContext` from CC BY 3.0 US
  FluidR3 MP3 samples. App-side abcjs synth playback is the equivalent. It carries a CC BY
  attribution obligation if those samples are bundled.
- **UNTESTED: long scores and multi-page layout at song length.** The 60 s clip produced 2 pages.
  Blocker: the ≤60 s lane bound. Next test: the >300 s multi-window test above, with
  `--render-score`.

## Proposed implementation slices (owner decision)

**A. Native port, in place of the monolithic sc-22996 AC.** This is the recommended "revise". It is
required for YuE2 covers regardless of the standalone decision.

1. **MERT2 encoder (Candle).**
   - Mel front end from the checkpoint buffers, ConvNeXt-v2 + GRN subsampler, 24 Conformer blocks
     with chunked or fused attention, and fp32 LoRA merge at load with MERT pin verification.
   - AC: per-layer hidden-state parity against reference dumps (`export_embeddings`/`--all-layers`)
     on CPU fp32, then Metal/CUDA with measured tolerances.
2. **SheetSage2 decode.**
   - Layer mix, projection, BART decoder with self and cross KV cache, `PromptGrammarState`, and the
     computed tokenizer with fingerprint check.
   - AC: token-exact on the committed fixtures, where `tokens.txt` in `synth_full`, `real_full` and
     `synth_eb_*` are the oracles.
3. **Symbolic post-processing.**
   - Event decode, window plan and stitching, LAB/JSON/MIDI, the ABC builder and validator, and
     key-aware chord spelling (head revision), plus an explicit empty-melody refusal.
   - AC: byte-identical `score.abc`/`chord.lab`/`key.lab` against the head-revision fixtures, and
     the silence case refused.
4. **Provider registration.** The `MusicTranscriber` contract (or an agreed alternative), component
   and license rows, and unload before generation. This slice feeds sc-22989/sc-22998.
5. **Cover integration** (what remains of sc-22996): reviewed melody/full ABC into YuE2
   `cot=melody/full`, provenance, and the two cover smokes.

**B. Standalone utility, only if the owner adopts it.** Each item is SceneWorks-side and
experimental/noncommercial only.

1. A worker job and API: "Transcribe to score" (audio → ABC/MIDI/LAB/events), reusing slice A.4.
2. A review UI with an abcjs score view, optional abcjs piano playback, and ABC/MIDI/PDF download,
   reusing the sc-22997/sc-23000 ABC review components.

**C. Stop.** Tensor and embedding export as a product feature, the paper preset, and bundling
Playwright or Chromium.

## Brief premises checked

- "Pinned identities (verify)" — **verified**: both pins exist and are the release commits. The
  MERT sha256 matches. However, **neither pin is upstream HEAD any more.** SheetSage2 `main` has
  post-release chord-spelling fixes (weights unchanged), and MERT has a README-only change.
- "Existing MERT/HuBERT/wav2vec-style encoders or YuE1's HuBERT branch" may fit — **mostly wrong**
  for MERT-v2 (see [native portability](#native-portability)).
- "If a repo is gated…" — neither repository is gated.
- SheetSage2 "requires FFmpeg 6.1" — the default path works with FFmpeg 9.0.1. Only the torchaudio
  FFmpeg-library path (`preset="paper"`) failed.
- "sc-22989 is creating a new inference crate right now" — on this base branch no YuE2 crate
  exists yet. The sibling worktree held only an untracked reference-environment script at survey
  time. That is in flight, not wrong.
