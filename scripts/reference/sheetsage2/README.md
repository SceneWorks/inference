# SheetSage2 reference experiment (sc-23003)

Bounded, reproducible run of the released SheetSage2 utility (audio -> ABC / MIDI / timed
annotations, plus its optional score/piano renderers) on the **torch CPU device**, used as evidence
for [the SheetSage2 utility assessment](../../../docs/reference/sheetsage2-utility-assessment.md).

**Reference tooling only.** The Python here runs upstream's `trust_remote_code` implementation to
observe what the utility does. It is not a shipped path and must never become one: epic sc-22988
requirement E3 requires production transcription to be native Rust/Candle.

## Pinned inputs

| Item | Identity |
|---|---|
| SheetSage2 weights + remote code | `m-a-p/SheetSage2` @ `eab522a8168e8b8b8c4856bf8609cd86198f01fe` (release commit, 2026-09-09) |
| MERT-v2-FullSong (loaded by SheetSage2 itself) | `m-a-p/MERT-v2-FullSong` @ `d8ba1c745e733b3908ce6ad16ebeb17ac7600a42`, `model.safetensors` sha256 `e6dd2ab187d6dd62b6521cd7d8f932e237acf0c5757745a7232082e28391350d` (checked by SheetSage2's loader against its `config.json`) |
| YuE2 cover workflow reference | `github.com/multimodal-art-projection/YuE` @ `92a73cc7652fcc1f937855e4b765e0a0edd7ff2e`, `docs/covers.md` |
| Real recording | "The Star-Spangled Banner", soprano MU1 Amy Broadbent with the U.S. Navy Band — U.S. federal government work, **public domain**. [Commons page](https://commons.wikimedia.org/wiki/File:%22The_Star-Spangled_Banner%22_-_Solo_vocalist_-_U.S._Navy_Band.oga); 3,516,534 bytes, SHA-1 `a423e4a4932db7dd332ecbf6dcfd78fd18545731` (Commons-published), SHA-256 `054e03b423228c698ee67dfbf73e8209286d54004db2035710b5126eda0a2e91`. Fetched and verified by the script; not committed. The first 60 s are transcribed (SheetSage2's own `max_seconds`). |
| Synthetic recordings | Generated deterministically by `run_experiment.py` (seed 23003): 39.4 s, 100 BPM 4/4, C major, 16 bars of melody + triads + bass + drums, with exact ground truth ([`artifacts/synth_truth.json`](artifacts/synth_truth.json)); and the same clip transposed +3 semitones to Eb major ([`artifacts/synth_eb_truth.json`](artifacts/synth_eb_truth.json)). |

## Environment

Upstream asks for Python 3.10/3.11 and FFmpeg 6.1 with shared libraries. This run used:

```bash
uv venv --python 3.11 ../sc-23003-sheetsage2-venv          # anywhere OUTSIDE the repository
SNAP=$(hf download m-a-p/SheetSage2 --revision eab522a8168e8b8b8c4856bf8609cd86198f01fe)
hf download m-a-p/MERT-v2-FullSong --revision d8ba1c745e733b3908ce6ad16ebeb17ac7600a42
uv pip install --python ../sc-23003-sheetsage2-venv/bin/python \
    -r "$SNAP/requirements.txt" -r "$SNAP/requirements-render.txt"
../sc-23003-sheetsage2-venv/bin/python -m playwright install --only-shell chromium   # renderer only
```

`requirements.txt` pins torch 2.8.0, torchaudio 2.8.0, transformers 4.45.2, huggingface-hub
0.36.0, safetensors 0.5.3, numpy 1.24.3, scipy 1.13.1, mir_eval 0.8.2, pretty_midi 0.2.10, mido
1.3.3; `requirements-render.txt` adds playwright 1.58.0. On macOS arm64 the PyPI torch wheel is the
CPU/MPS build; the script never selects MPS or CUDA and asserts every parameter is on `cpu`.
FFmpeg on the recorded machine was Homebrew 9.0.1, not 6.1 (see the `failures` case).

## Run

```bash
cd scripts/reference/sheetsage2
../../../../sc-23003-sheetsage2-venv/bin/python run_experiment.py all --work /path/outside/repo
```

Sub-commands: `fixtures` (fetch + verify the recording, synthesize the ground-truth clip), `run
[case]` (each case in a child process, sampled every 0.5 s with `ps`; the process tree is killed
above `--rss-cap-gb`, default 24), `evaluate` (mir_eval scores for the synthetic clip), `collect`
(copies small outputs into `artifacts/`, scrubbing local paths). `--threads` defaults to 8.

Cases:

| Case | What it exercises |
|---|---|
| `real_full` | First (online) load through the Hub with pinned revisions; full prompts on the 60 s recording; piano WAV (`mix`) + score PDF/SVG/PNG rendering through Playwright/abcjs |
| `real_melody` | Offline (`HF_HUB_OFFLINE=1`) reload; `melody_only=True` cover score |
| `synth_full` | Offline; full prompts on the ground-truth clip; `export_embeddings/logits/scores` tensor exports (shapes recorded, tensors not committed) |
| `synth_merged` | `save_pretrained` merged standalone snapshot, reload it with `local_files_only`, transcribe the same clip; token identity with `synth_full` is checked |
| `failures` | Short/non-finite/undecodable/unlabelled inputs, digital silence (full and melody-only), and the `preset="paper"` torchaudio-FFmpeg decode path |
| `render_only` | Upstream `render.py` on `real_full`'s outputs with no model loaded (vocal/instrumental/chords piano parts + SVG) |
| `real_full_head` | Same as `real_full` without rendering, but with upstream `main` code (`4f89269db831bdc1880124164a00d4f9385cd129`; weights unchanged) to measure post-release code drift |
| `synth_eb_release` / `synth_eb_head` | The synthetic clip transposed to Eb major, under release and head code: the case where key-aware chord spelling differs |

## Committed outputs

`artifacts/` holds only small text/MIDI outputs: `score.abc`, `*.mid`, `*.lab`, `events.json`,
`events.tsv`, `tokens.txt`/`tokens.json`, `result.json`, `playback.json`, the first PNG score page,
each case's `case.json` (timings, environment, warnings) and the tail of its log, plus
`evaluation.json`, `fixtures.json` and `measurements.json`. Cases that repeat another case's
transcription (`synth_merged`, `real_full_head`, `synth_eb_*`) keep only `score.abc`, `chord.lab`,
`key.lab`, `result.json` and `tokens.txt`. Weights, tensors, recordings and
rendered WAV/PDF files stay in `--work`; each case's `not_committed.json` lists what was omitted
and its size.

Timings and RSS in `measurements.json` are observations on one machine (named in the file) with
the torch CPU device in fp32. They are not portable requirements and say nothing about native
Metal/CUDA cost.
