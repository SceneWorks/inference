"""Pinned producer environment shared by the Mage oracle provisioners."""

from __future__ import annotations

import importlib.metadata
import sys


REFERENCE_PACKAGES = {
    "accelerate": "1.13.0",
    "diffusers": "0.38.0",
    "einops": "0.8.2",
    "loguru": "0.7.3",
    "numpy": "2.4.3",
    "pillow": "12.3.0",
    "pydantic": "2.12.5",
    "safetensors": "0.8.0",
    "torch": "2.13.0",
    "torchvision": "0.28.0",
    "transformers": "5.5.0",
    "typing_extensions": "4.15.0",
}
# Keep this aligned with real-weights.yml. The self-hosted macOS oracle runner installs this exact
# standalone interpreter in runner temp via uv, avoiding system Python and global tool caches.
REFERENCE_PYTHON = (3, 12, 10)


def validate_python_version(
    version: tuple[int, int, int], error_type: type[RuntimeError] = RuntimeError
) -> None:
    if version != REFERENCE_PYTHON:
        raise error_type(
            f"reference Python is {'.'.join(map(str, version))}, "
            f"expected {'.'.join(map(str, REFERENCE_PYTHON))}"
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
