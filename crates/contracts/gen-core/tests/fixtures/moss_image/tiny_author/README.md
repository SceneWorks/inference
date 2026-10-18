# Tiny independent MOSS graph fixtures (sc-26323)

`fixture.json` uses the unchanged frozen author MOSSImageTransformer and all its
numerical helper classes/functions, extracted as AST definitions from
`frozen_author_pipeline.py`. Real Torch 2.9.1 and Diffusers mixins execute on the
CPU. Unused package-loading imports are omitted because the preparation runtime
has no Transformers installation. No model graph or numerical helper is replaced.
`LICENSE.author` is the exact Apache-2.0 LICENSE shipped with the frozen SFT bundle.
Source: https://huggingface.co/OpenMOSS-Team/MOSS-Image-SFT/blob/fc5de0d56fc910cbca6f6625a70f5f582b26deb1/pipeline_moss_image.py

`dmd_fixture.json` adds the unchanged `dmd_loop` and nested `Unmerged` wrapper AST
from the evaluated SceneWorks reference (`scripts/evaluation/moss/reference.py`
at evaluation source `ed011b85027c237865623f4b4aba478cfcb060a7`). That repository is
AGPL-3.0; its source is intentionally not copied into this Apache-2.0 repository.
Regeneration requires an explicitly supplied external `--reference-source` whose
exact SHA-256 is checked before AST extraction. The JSON contains synthetic data
and numerical outputs rather than copied reference source. Production adapter validation fixes rank 128 and 42
blocks, so the tiny fixture installs synthetic rank-3 projections with exactly the
same FP32 time/modulation promotion and six replacement parameter copy operations.
The tiny dimensions, rank, codes, noise, Q and replacement contents are synthetic
fixture data, not published author hyperparameters or trained codes. The fixture
records every step's input, velocity and clean estimate, three FP32 module hooks,
and deliberately incorrect cast-down/merged-BF16 results. The projection probe
independently discriminates residual dtype and rounding.

All tensors are JSON objects with shape, dtype and row-major values. BF16 values
are serialized after exact conversion to FP32, preserving their represented values.
Base weights are FP32 synthetic parameters; cast them to the case's backbone dtype
before execution. In mixed DMD, promote time/shared/head modules after the backbone
cast, then copy the six FP32 replacements and install FP32 unmerged adapters.

Regenerate with installed Torch and Diffusers, without downloads or accelerator use:

```sh
PYTHONDONTWRITEBYTECODE=1 python3 -B generate_fixture.py --author-source frozen_author_pipeline.py
PYTHONDONTWRITEBYTECODE=1 python3 -B generate_dmd_fixture.py --author-source frozen_author_pipeline.py --reference-source /path/to/evaluation/reference.py --base-fixture fixture.json
```

Both generators accept `--output-dir`. They enforce CPU, one Torch thread, no CUDA,
source hashes and a 120-second process alarm. Receipts bind source/script/data hashes.
`dmd_attempts.json` retains the first failed diagnostic (a probe dtype mismatch,
corrected before success). These fixtures prove neither real-weight execution nor
image quality. Native backends consume these independent expected results rather
than manufacturing expected outputs from their own implementations.
