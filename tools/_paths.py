"""Shared path helpers for the dev-only golden-dump scripts.

Keeps the scripts portable across machines/users: output fixtures are derived from the repo
root (this file's location), never a hardcoded ``/Users/<name>`` path, and the model snapshot
honors the standard Hugging Face cache (``HF_HUB_CACHE`` / ``HF_HOME``, else
``~/.cache/huggingface/hub``).

The scripts are run directly (``python tools/dump_*.py``), so ``tools/`` is on ``sys.path`` and
``from _paths import fixture`` resolves.
"""

from __future__ import annotations

import os
from pathlib import Path

# tools/_paths.py -> the repo root is one directory up.
REPO_ROOT = Path(__file__).resolve().parents[1]


def fixture(rel: str) -> str:
    """Absolute path to a repo-relative output/fixture file.

    e.g. ``fixture("mlx-gen-z-image/tests/fixtures/z_latents.safetensors")``.
    """
    return str(REPO_ROOT / rel)


def hf_hub_cache() -> Path:
    """The Hugging Face hub cache dir, honoring ``HF_HUB_CACHE`` / ``HF_HOME``.

    Falls back to the default ``~/.cache/huggingface/hub``.
    """
    if cache := os.environ.get("HF_HUB_CACHE"):
        return Path(cache)
    if home := os.environ.get("HF_HOME"):
        return Path(home) / "hub"
    return Path.home() / ".cache" / "huggingface" / "hub"
