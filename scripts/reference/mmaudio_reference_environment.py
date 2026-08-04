"""Pinned producer environment for the MMAudio parity-fixture generator (sc-17285).

The committed fixtures under `crates/audio/candle-audio-mmaudio/tests/fixtures/` record the
PyTorch reference's *numbers*, and those move with the stack that produced them — torch most of
all. So the stack is pinned rather than merely reported: `mmaudio_reference.py dump` validates the
running interpreter against this module and refuses to regenerate anything from an environment that
does not match, and the fixture metadata carries the validated versions so a future reader can tell
what produced them.

Keep this aligned with `crates/audio/candle-audio-mmaudio/_vendor/requirements-oracles.in`, which is
the hash-locked install side of the same contract;
`scripts/tests/test_mmaudio_reference.py` fails in ordinary CI if the two drift.
"""

from __future__ import annotations

import importlib.metadata
import sys

#: Distribution name -> exact version. Distribution names, not import names: `open_clip_torch`
#: imports as `open_clip` and `huggingface-hub` as `huggingface_hub`, and `importlib.metadata`
#: keys on the former.
REFERENCE_PACKAGES = {
    "einops": "0.8.2",
    "huggingface-hub": "1.26.0",
    "librosa": "0.11.0",
    "numpy": "2.4.6",
    "omegaconf": "2.3.1",
    "open_clip_torch": "3.3.0",
    "requests": "2.34.2",
    "safetensors": "0.8.0",
    "timm": "1.0.28",
    "torch": "2.13.0",
    "torchdiffeq": "0.2.5",
    "torchvision": "0.28.0",
    "tqdm": "4.70.0",
}

#: Major/minor only, deliberately unlike the Mage oracle's exact-patch pin. That lane installs one
#: exact standalone interpreter via uv in runner temp; this producer is an operator command on a dev
#: box with ~8 GB of weights staged, and a CPython patch release does not move torch's numerics —
#: the package pins above are what do. Pinning the patch here would reject a legitimate box for a
#: difference that cannot reach the fixtures.
REFERENCE_PYTHON = (3, 12)


def validate_python_version(
    version: tuple[int, ...], error_type: type[RuntimeError] = RuntimeError
) -> None:
    if tuple(version[:2]) != REFERENCE_PYTHON:
        raise error_type(
            f"reference Python is {'.'.join(map(str, version))}, "
            f"expected {'.'.join(map(str, REFERENCE_PYTHON))}.x"
        )


def validate_reference_environment(
    error_type: type[RuntimeError] = RuntimeError,
) -> dict[str, str]:
    """Validate and return all producer pins without changing filesystem state."""
    validate_python_version(sys.version_info[:3], error_type)
    actual = {}
    for package, expected in REFERENCE_PACKAGES.items():
        try:
            actual[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError as error:
            raise error_type(
                f"pinned reference package is missing: {package}=={expected}"
            ) from error
        if actual[package] != expected:
            raise error_type(
                f"pinned reference package mismatch: {package}=={actual[package]}, "
                f"expected {expected}"
            )
    return actual
