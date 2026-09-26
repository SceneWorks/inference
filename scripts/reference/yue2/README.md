# YuE2 reference environment (epic sc-22988)

Every YuE2 fixture and parity comparison in this repository is produced against **one** pinned
upstream, so the stories of the epic that run in parallel share a single reference:

| What | Pin |
| --- | --- |
| Upstream source | `github.com/multimodal-art-projection/YuE` @ `92a73cc7652fcc1f937855e4b765e0a0edd7ff2e` (Apache-2.0) |
| Python stack | exactly the pins in that commit's `pyproject.toml` (torch 2.10.0, transformers 4.57.6, safetensors 0.7.0, tiktoken 0.12.0, numpy 2.2.6, …), Python 3.12 |
| Weights | the five revisions in `crates/audio/candle-audio-yue2/src/inventory.rs` (`REPOS`) |

The environment lives **outside** the repository and is never committed. Python is reference and
build-time tooling only; production YuE2 runs natively (epic E3).

## Set up (or re-verify)

```sh
scripts/reference/yue2/setup_reference_env.sh
```

- Clones the upstream into `$YUE2_REF_DIR/YuE` (default `~/.cache/sceneworks-yue2-ref`), checks
  out the pinned commit detached, and refuses to continue unless `HEAD` is that commit and the tree
  is clean.
- Creates `$YUE2_REF_DIR/venv` with `python3.12` (override with `YUE2_PYTHON`) and installs the
  pinned checkout non-editably with its own dependency pins, then runs `pip check`.
- Writes `$YUE2_REF_DIR/ENVIRONMENT.json` (commit, Python, platform, every installed package
  version) so a fixture can name the exact stack that produced it.
- Idempotent: a re-run only re-verifies.

## Weights

Downloads are an operator step, never part of a fixture script (and never part of the Rust
crates). Fetch the pinned revisions into a Hugging Face hub directory:

```sh
hf download m-a-p/YuE2-3B --revision 1a96eca688d6ae5d7f0feb88573fec89920fcd19
hf download m-a-p/YuE2-Vae --revision 95535e72a97bc0f09b8ada125d26b4009428c0e8
hf download m-a-p/YuE2-Vae-legacy --revision b54118f0fc462f08999d1ec07e88817f4ee3f770
# cover closure only (SheetSage2 + MERT-v2-FullSong), not needed for generation:
hf download m-a-p/SheetSage2 --revision eab522a8168e8b8b8c4856bf8609cd86198f01fe
hf download m-a-p/MERT-v2-FullSong --revision d8ba1c745e733b3908ce6ad16ebeb17ac7600a42
```

All five repositories were ungated and declared `cc-by-nc-4.0` on 2026-09-26. The weights are
noncommercial: see the `license` module of `candle-audio-yue2` before sharing anything derived
from them.

Then point the tooling at that hub directory (the one holding `models--m-a-p--*/`):

```sh
export YUE2_HF_HUB=/path/to/huggingface/hub
```

## Running a fixture generator

Run generators with the reference interpreter, offline, against the pinned snapshots:

```sh
HF_HUB_OFFLINE=1 YUE2_HF_HUB=... ~/.cache/sceneworks-yue2-ref/venv/bin/python scripts/reference/yue2/<generator>.py
```

Conventions for YuE2 generators (mirroring the YuE1 ones under `scripts/reference/`):

- Import upstream code from the installed pinned package (`import yue2`), never a copy or a
  different checkout. Resolve weights with `local_files_only=True` / `HF_HUB_OFFLINE=1` at the
  pinned revision, and verify them first (see below).
- Write fixtures under the consuming crate's `tests/fixtures/`, with a README naming the generator,
  the upstream commit, the weight revisions and `ENVIRONMENT.json`'s package versions. Never commit
  weights.
- Anything that loads full weights states its expected peak RSS and runs under an external RSS
  guard; the YuE2-3B checkpoint alone is 7.3 GB of BF16.

## Generators here

| Script | Produces |
| --- | --- |
| `asset_manifest.py` | `crates/audio/candle-audio-yue2/manifests/*.json` — the conversion manifests: every closure file's size and SHA-256 (checked against upstream's own `weights_manifest.json` via `yue2.storage.model_identity`), the identity conversion, and every tensor's name, dtype, shape and value digest as PyTorch loads it. Peak RSS measured at 7.6 GB (mostly the mmapped YuE2-3B file). |
| `protocol_fixtures.py` | `crates/audio/candle-audio-yue2/tests/fixtures/protocol/` — text-tokenizer, request-protocol and symbolic-plan fixtures, all evaluated by upstream code (see that directory's README). `--hub` adds the pinned-`qwen.tiktoken` cases. No weights loaded; peak RSS ~0.4 GB. |
| `vae_reference.py` | `tiny`: the committed CI fixture for the native VAE (`crates/audio/candle-audio-yue2/tests/fixtures/vae_tiny*`) — two released-topology VAEs at toy widths with randomized weights (standard/legacy), upstream `decode`, `YuE2Pipeline.decode` output and encoder posterior for one shared latent. `real`: the pinned-weight reference (both decoders on the same encoded latents); the waveforms are derived from CC BY-NC weights so they are written outside the repository (`~/.cache/sceneworks-yue2-fixtures/vae`) and only their SHA-256 and statistics are committed (`tests/fixtures/vae_real_reference.json`). Peak RSS measured at 0.34 GB (tiny) / 5.9 GB (real). |

The Rust side re-derives the tensor digests natively and compares
(`crates/audio/candle-audio-yue2/tests/real_weights.rs`, `#[ignore]`d; set `YUE2_HF_HUB`). The VAE
parity runs in the crate's lib tests (tiny fixture) and in `tests/vae_real_weights.rs`
(`#[ignore]`d; needs `YUE2_HF_HUB` and the `real` reference).
