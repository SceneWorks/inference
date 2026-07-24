# Vendored `mage_flow` — the frozen Mage-Flow parity oracle (sc-14036, epic 14034)

`mage_flow/` is a **verbatim copy** of the reference PyTorch inference implementation from

    https://github.com/microsoft/Mage @ df7f84d9f8fc991d189d929f03cff623b430a4a2
    (2026-07-23T03:15:46Z — "Point tech-report links back to arXiv (2607.19064)")
    subtree `mage_flow/`, git tree object 6b3e1ad42ca8b229176ceb58b153a3cc96341f0c

It is the ground truth that `tools/dump_mage_flow_golden.py` runs to produce the boundary
goldens the native Mage-Flow port (`mlx-gen-mage`, `candle-gen-mage`) is gated against. It is
**read-only**: nothing in this directory is imported by any Rust crate, shipped in any bundle,
or on any product path — it exists so the oracle survives.

## Why it is committed (this crate normally does NOT commit `_vendor/`)

The crate `.gitignore` ignores `_vendor/` because the usual vendored reference (e.g. the
SenseNova-U1 checkout, epic 3180) is cloned on demand and cannot be redistributed. **Vendoring
a parity golden is a license decision, and this one clears it:** `microsoft/Mage` is **MIT**
(`mage_flow/LICENSE`, copied verbatim from the upstream repo root), which explicitly permits
redistribution provided the copyright and permission notice travel with the copy — they do.

The reason to take that option here is **pull risk**. Mage-Flow was published 2026-07-22; the
Lens precedent in this repo is that a freshly released research drop can disappear (repo made
private, model cards pulled) between planning and porting. Every parity test in P1-P4 of epic
14034 depends on this code, so it is preserved in-tree rather than depended on remotely.

`crates/media/mlx-gen/NOTICE` records the attribution.

## What was and was not copied

| path | vendored? | note |
| --- | --- | --- |
| `pipeline.py`, `inference.py`, `app.py`, `__init__.py` | yes | verbatim |
| `models/` (`mage_flow.py`, `utils.py`, `modules/*`) | yes | verbatim — DiT, Mage-VAE, TE, RoPE/blocks, Gaussian-Shading, attention shim |
| `requirements.txt`, `pyproject.toml`, `README.md` | yes | verbatim — the env pins ARE part of the oracle |
| `LICENSE` | yes | upstream repo-root MIT, copied in beside the code it licenses |
| `assets/dog.jpg` | yes | the reference's own edit example (`app.py:161`, `README.md:270`); the edit golden's source image |
| `assets/*` (20 other files, ~40 MB) | **no** | README gallery/figure images only — no code path reads them. The README's `<img src="assets/…">` links therefore do not resolve locally; they resolve on GitHub. `scripts/check_docs.py` only walks `README.md`, `docs/**`, `release/**`, so this does not break the docs gate. |
| `mage_vl/` (upstream sibling package) | **no** | out of scope for epic 14034 |

Everything vendored is **byte-for-byte upstream** — no local patches. Verify with:

```sh
git -C /path/to/Mage checkout df7f84d9f8fc991d189d929f03cff623b430a4a2
diff -r --exclude=assets --exclude=__pycache__ \
  /path/to/Mage/mage_flow crates/media/mlx-gen/_vendor/mage_flow
```

The harness deliberately does **not** edit the vendored source to run off CUDA. It rebinds
`mage_flow.pipeline.ModelConfig` to a subclass whose `attn_type` defaults to `sdpa`
(`dump_mage_flow_golden.py::_load_model`) — the reference already supports that value in both
`_attn_backend.set_attn_backend` and `text_encoder._resolve_hf_attn_impl`. Keep it that way: any
future adaptation belongs in the harness, not here, so the `diff -r` above stays empty.

### SHA-256 of the vendored files

```
275b4dd619de4e16a017b10d0beec72abbbbf14ee8a2fc68f8bdb398e821f623  mage_flow/LICENSE
3415638dd4674b0a570f7db2b417efa6235a3c7553180652e5c2b5cc1eb13d58  mage_flow/README.md
0709764f182b55fdec6b8195a4d18640fe0ffef3785d3977d8ebc29905de7489  mage_flow/__init__.py
ac0597feddf1b6c5aaa2f18cc3ebb3e690dc42232462ac778af45044df41435e  mage_flow/app.py
164d8dfe707fb854e288ad2eea65c2db87e90af11f689c85502860eeaf3f4794  mage_flow/assets/dog.jpg
0a1e196f784f4daf4b4d1607cade1d066908341e0e444aa29fcea3967a8c1a3f  mage_flow/inference.py
7e5ba07fdb01f4a5912eaeb3afbe9d4ccc4e391e7de4cdc11865e5aa911ddfd5  mage_flow/models/__init__.py
59b6e1bee7f95a7fd2fa7bd9e765966832951e2bb184b5ae27283884997b845f  mage_flow/models/mage_flow.py
e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855  mage_flow/models/modules/__init__.py
7d652eeb2ab39e37554e3d62859c74edeffc8dce70b2c15687060c259647b559  mage_flow/models/modules/_attn_backend.py
9ffd7c68b6053ad37f6bf0925d8d94dcd72af751448ee0228ec13018fedf54e5  mage_flow/models/modules/mage_latent.py
4b198343b8929f48a0a14d388502c81f54be17b222f6831303d2da6a91f33a62  mage_flow/models/modules/mage_layers.py
eed7846d02bee28ebf7a1fb45db7d9f621dd66e702e761848d67bed426c1c4a7  mage_flow/models/modules/mage_text.py
64f4d7041003e416bc2f4fac5bbf8aabf2e7c798ad106682c34332ba347b0ef9  mage_flow/models/modules/mage_vae.py
65e490ce35fbe4d4057f115be2ab8731e01ea44b618c10aabac80d51fc38ef81  mage_flow/models/modules/text_encoder.py
0f242ca7a77e0f85b5985b5299304aa0786737de1ba414c7f8f47b8d664dbca8  mage_flow/models/utils.py
b9fc57018570372dd3404a733e19b918a359188dd9f5ef7817c6b30969fc13db  mage_flow/pipeline.py
d04bc0c4d884bbca77a79711ed38fbddcf4d8b642b3f2deb1a34b2b01d879f04  mage_flow/pyproject.toml
ba35ca260f3b1ad62b7550bec4397c6666a5dc0278c2e8f65fb6c31e0dfb3ccc  mage_flow/requirements.txt
```

## Pinned reference environment (verbatim from `mage_flow/requirements.txt`)

> Python 3.11, CUDA 12.6 per the upstream header. Python 3.12 also resolves every pin and is
> what the goldens in `tools/golden/` were dumped with.

```
torch==2.13.0
torchvision==0.28.0         # reference-image resize in the edit path
numpy==2.4.3                # used by the test/benchmark scripts
diffusers==0.38.0
transformers==5.5.0
accelerate==1.13.0          # used by transformers `from_pretrained` loading path
safetensors==0.8.0          # diffusers 0.38.0 requires safetensors>=0.8.0
huggingface_hub>=0.20       # snapshot_download for MageFlowPipeline.from_pretrained("<hf repo id>")
einops==0.8.2
pydantic==2.12.5
pillow==12.3.0
loguru==0.7.3
typing_extensions==4.15.0   # only needed on Python < 3.11 (fallback for typing.Unpack)
gradio>=6.0                 # Gradio web app (mage-flow-app / mage_flow/app.py)
```

plus, installed **separately with build isolation OFF** (it compiles a CUDA extension against
the installed torch ABI, so `-r requirements.txt` cannot carry it):

```
flash-attn==2.8.3
```

Two pins are load-bearing and easy to get wrong:

* **`transformers` must stay `>=5.3.0,<5.6`** (`pyproject.toml`). 5.6 removed the `input_embeds`
  kwarg of `create_causal_mask` that the vendored TE patch calls
  (`models/modules/text_encoder.py:255`). On 5.5.0 that call already emits a `FutureWarning`.
* **The versions stamped in the published weight configs are NOT the runtime pins.**
  `text_encoder/config.json` says `transformers_version: 4.57.0.dev0` and
  `scheduler/scheduler_config.json` says `_diffusers_version: 0.37.0`; those are provenance
  strings from whatever env serialized the weights. The runtime env is the table above.

`gradio` and `flash-attn` are **not** needed to produce the goldens (no web app; the attention
shim routes to `sdpa`); everything else is.

### Building the env

```sh
uv venv --python 3.12 /tmp/mageflow-ref-venv
uv pip install --python /tmp/mageflow-ref-venv/bin/python \
  torch==2.13.0 torchvision==0.28.0 numpy==2.4.3 \
  diffusers==0.38.0 transformers==5.5.0 accelerate==1.13.0 safetensors==0.8.0 \
  einops==0.8.2 pydantic==2.12.5 pillow==12.3.0 loguru==0.7.3 "huggingface_hub>=0.20"
# CUDA hosts only, after the above:
# uv pip install --python /tmp/mageflow-ref-venv/bin/python --no-build-isolation flash-attn==2.8.3
```

## Running the oracle

The dump harness puts this directory on `sys.path` itself, so no `PYTHONPATH` is needed:

```sh
cd crates/media/mlx-gen
/tmp/mageflow-ref-venv/bin/python tools/dump_mage_flow_golden.py --stage all
```

Weights come from the standard HF cache (`microsoft/Mage-Flow`, `microsoft/Mage-Flow-Edit`) or
from `$MAGE_SNAPSHOT` / `$MAGE_EDIT_SNAPSHOT`. See the harness docstring for the full env-var
surface and `tools/golden/README.md` for the golden manifest.

### Device notes (this is not a CUDA-only oracle)

* **CPU (`MAGE_DEVICE=cpu`) is the blessed path on macOS** and is what the goldens in
  `tools/golden/` were dumped with (the reference runs bf16 end to end; `load_from_repo`
  hard-casts and offers no dtype knob). Budget, at the 256²/4-step defaults on an M-series host:
  `noise`/`vae` are seconds, `te` ~15 s, the gen stack (`te` + `e2e` + `dit` + `dit_block`)
  ~8 min, and `edit` ~25 min. Edit is the slow one for two reasons — its packed image stream is
  twice as long (target + reference) and its content gate is a *multimodal* `.generate()`. Note
  the harness pre-screens before calling the pipeline (which screens again internally), so the
  moderation pass runs twice per stage. That is deliberate: it turns a screening failure into a
  loud error instead of a blank-white refusal-image golden that every parity test would happily
  match.
* **MPS runs the VAE and the text encoder fine but NOT the DiT.** `torch 2.13.0` MPS
  mis-handles `Tensor.repeat_interleave(repeats=<int32 tensor>)`, which the adaLN modulation
  broadcast uses (`models/modules/mage_layers.py:566`, fed by the int32 `cu_seqlens` the
  reference builds in `pipeline._lens_to_cu`). It fails loudly (`Invalid buffer size: 6528 GiB`),
  not silently. This is a torch/MPS bug, not a reference bug — do not "fix" it by editing the
  vendored source.
* **CUDA** should use `MAGE_ATTN=flash2` (the default there) with `flash-attn==2.8.3` installed,
  which is the exact configuration upstream ships.

## Maintenance

Re-vendor only on a deliberate upstream bump:

1. Re-copy the `mage_flow/` subtree from the new upstream rev (assets excluded except `dog.jpg`).
2. Update the commit SHA, tree SHA, and the SHA-256 table above.
3. Re-read `MAGE_FLOW_GAPS.md` — its answers carry `file:line` citations that a bump can shift,
   and a semantic change there (drop_idx, RoPE centering, edit ordering) invalidates the port.
4. Re-dump every golden (`--stage all`) and re-bless `tools/golden/CHECKSUMS.txt`.
