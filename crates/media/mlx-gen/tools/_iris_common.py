"""Shared setup for the Iris-3B fixture generators (`dump_iris_*.py`, sc-25679).

Python is used ONLY here, offline, as the reference oracle: the frozen upstream modules are imported
from a source checkout and run on CPU in fp32. Nothing at runtime depends on Python.

Frozen upstream (recorded in `mlx-gen-iris/src/lib.rs` as constants — keep the two in lockstep):
  * code    `speridlabs/iris-3b` @ a8d15239dea469aba042cfa56ca3bb4e450d5ebc
  * weights HF `speridlabs/iris-3b` @ 7445443349bc9abe3c96f01ff793e2098ca012b3
  * text    HF `Qwen/Qwen3-VL-4B-Instruct` @ ebb281ec70b05090aa6165b016eac8ec08e71b17

Inputs (no machine-specific defaults are baked in):
  IRIS_SRC  path to a git checkout of `speridlabs/iris-3b` at the pinned commit. The checkout's HEAD
            and the sha256 of every upstream module the oracle imports are verified before use, so a
            drifted checkout fails loudly instead of producing a silently different golden.

Run inside an isolated venv (torch 2.8 CPU, transformers 4.57.1, safetensors, omegaconf) with
`python -I`, e.g.:
    IRIS_SRC=/path/to/iris-3b python -I tools/dump_iris_golden.py
"""

from __future__ import annotations

import hashlib
import json
import subprocess
import sys
from pathlib import Path

from _paths import fixture, require_env

IRIS_COMMIT = "a8d15239dea469aba042cfa56ca3bb4e450d5ebc"
IRIS_WEIGHTS_REVISION = "7445443349bc9abe3c96f01ff793e2098ca012b3"
QWEN3_VL_REPO = "Qwen/Qwen3-VL-4B-Instruct"
QWEN3_VL_REVISION = "ebb281ec70b05090aa6165b016eac8ec08e71b17"

FIXTURE_DIR = Path(fixture("mlx-gen-iris/tests/fixtures"))

# sha256 of every upstream module the oracle executes, at IRIS_COMMIT. A mismatch means the
# checkout is not the frozen source.
UPSTREAM_SHA256 = {
    "src/iris3b/config.py": "e29767fb65ed32c4fe2e443ee9cb2e503a913fd4cdfa4d1df54fa60f84c96ecb",
    "src/iris3b/models/dit.py": "c7ffdf46a466765995ffdd057f491bb5c52055d6ebc5e1b287db73f654a467fd",
    "src/iris3b/models/blocks/mmdit.py": "0e136ae070b701718f6907e70e06c20fbcf243ddfbe87ae6f4e1d30e9bd49111",
    "src/iris3b/models/blocks/single_stream.py": "f7cf776f4ba6ce28d7785714d9ff2652e513d7d4bad2df3d0101722258792ed5",
    "src/iris3b/models/blocks/pit.py": "d31811f5c494d0c934edea578fe2175485558a4bce54668ee098a87d72343225",
    "src/iris3b/nn/attention.py": "786ed95bc61c3390926e872b9f407e1f1b42ae62df2d89964a5b3803eed68131",
    "src/iris3b/nn/embeddings.py": "b06188352fc5217f83bf01d7cf3a6255c7c19c6a8fdc97c89fb342c8a51b6aaf",
    "src/iris3b/nn/mlp.py": "608ac04ca768b98cec02429a010f2e8ca42d44056c84ec36ac6ca5e3c9c8af8f",
    "src/iris3b/nn/modulation.py": "e20c45eb343e2d05a4ec9147e771beab071111c6bcbccb7c014c6405a801afe7",
    "src/iris3b/nn/norms.py": "9656faabf360bab3ed64b98f9a8f34f26886af4306f6e02a60fd76968d9a867d",
    "src/iris3b/nn/rope.py": "7ece6b626486b35786ab7068411d8e36476507042e666874a776f2ffa8740c69",
    "src/iris3b/flow/schedule.py": "1598da48a0df1b1ae9a2ffdcec08749a17779f9f4261f7d7dcd82e6580cccf74",
    "src/iris3b/flow/solver.py": "99bb92da3efddb10b61335edee978ad5575d2d8364064ebcf1c0cd5048d84ae6",
    "src/iris3b/sampling.py": "cc90c056c08e5b2b3458d32a07d57dbfce291cc8d5c067f31da238871254bb04",
    "src/iris3b/text/qwen3_vl.py": "5b3cbbd8fee2f7563897fc0be5a5b7655ec9edaf3d1d71e199ca4ead93dd70aa",
}


def upstream_root() -> Path:
    root = Path(require_env("IRIS_SRC", "a checkout of speridlabs/iris-3b at " + IRIS_COMMIT))
    head = subprocess.run(
        ["git", "-C", str(root), "rev-parse", "HEAD"], check=True, capture_output=True, text=True
    ).stdout.strip()
    if head != IRIS_COMMIT:
        raise SystemExit(f"IRIS_SRC HEAD is {head}, expected the frozen {IRIS_COMMIT}")
    for rel, want in UPSTREAM_SHA256.items():
        got = hashlib.sha256((root / rel).read_bytes()).hexdigest()
        if got != want:
            raise SystemExit(f"{rel}: sha256 {got} != pinned {want}")
    return root


def import_upstream() -> Path:
    root = upstream_root()
    sys.path.insert(0, str(root / "src"))
    return root


def save_safetensors(path: Path, tensors: dict, metadata: dict | None = None) -> None:
    import torch
    from safetensors.torch import save_file

    path.parent.mkdir(parents=True, exist_ok=True)
    flat = {
        k: v.detach().contiguous().to(torch.float32) if v.is_floating_point() else v.detach().contiguous()
        for k, v in tensors.items()
    }
    save_file(flat, str(path), metadata={k: str(v) for k, v in (metadata or {}).items()})
    print(f"wrote {path} ({len(flat)} tensors, {path.stat().st_size} bytes)")


def write_json(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    print(f"wrote {path}")
