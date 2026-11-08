"""Frozen conditional FLUX.1 pair cases and exact external-asset metadata.

The base FLUX weights are the existing shape-faithful Tier 1 random fixture. Overlay and
face-analysis weights stay real: synthetic face-detector weights cannot produce a valid
reference identity. Both isolated arms receive the same hashed files and source image.
"""
import hashlib
import os
from pathlib import Path, PurePosixPath, PureWindowsPath
import re
import shutil

from scripts.ci.flux1_pair.fixture import sha

REFERENCE = Path("docs/migration/evidence/sc-16956/pulid-reference.png")
REFERENCE_SHA256 = "3995f2e856346748588e76a5557516d0218f44f5663701c5a50139d50c86a7be"

# Every native branch that actually ships is a separate bounded dispatch. Candle XLabs has
# no true-CFG request; Candle PuLID has no delayed/real-CFG request. Do not fabricate one.
CASES = {
    "control-pose": ("dev", ("mlx", "candle-cuda"), ("control",)),
    "control-canny": ("dev", ("mlx", "candle-cuda"), ("control",)),
    "control-depth": ("dev", ("mlx", "candle-cuda"), ("control",)),
    "control-zero": ("dev", ("mlx", "candle-cuda"), ("control",)),
    "ip-dev": ("dev", ("mlx", "candle-cuda"), ("ip_adapter", "image_encoder")),
    "ip-schnell": ("schnell", ("mlx", "candle-cuda"), ("ip_adapter", "image_encoder")),
    "ip-dev-cfg": ("dev", ("mlx",), ("ip_adapter", "image_encoder")),
    "pulid-fake": ("dev", ("mlx", "candle-cuda"), ("pulid", "eva", "scrfd", "arcface", "bisenet")),
    "pulid-real": ("dev", ("mlx",), ("pulid", "eva", "scrfd", "arcface", "bisenet")),
    "pulid-delayed": ("dev", ("mlx",), ("pulid", "eva", "scrfd", "arcface", "bisenet")),
    "pulid-pid": ("dev", ("mlx", "candle-cuda"),
                  ("pulid", "eva", "scrfd", "arcface", "bisenet", "pid_decoder", "pid_gemma")),
}


def case_metadata(case, backend, assets):
    if case not in CASES or backend not in CASES[case][1]:
        raise ValueError("conditional FLUX.1 case is unavailable on this backend")
    variant, _, roles = CASES[case]
    if type(assets) is not dict or set(assets) != set(roles):
        raise ValueError("conditional asset roles differ from the fixed case")
    normalized = {}
    for role in roles:
        row = assets[role]
        fields = {"path", "files", "revision"} if role == "pid_gemma" else {"path", "sha256", "revision"}
        if type(row) is not dict or set(row) != fields:
            raise ValueError("conditional asset needs path, sha256 and revision: " + role)
        if not isinstance(row["path"], str):
            raise ValueError("conditional asset path must be a string: " + role)
        path = PureWindowsPath(row["path"]) if backend == "candle-cuda" else PurePosixPath(row["path"])
        if not path.is_absolute() or ".." in path.parts or not row["path"].strip():
            raise ValueError("conditional asset path must be absolute and normalized: " + role)
        if role == "pid_gemma":
            files = row["files"]
            if (type(files) is not dict or not files or any(
                    not isinstance(name, str) or not name or "\\" in name
                    or PurePosixPath(name).is_absolute() or ".." in PurePosixPath(name).parts
                    or not isinstance(digest, str)
                    or not re.fullmatch(r"[0-9a-f]{64}", digest)
                    for name, digest in files.items())):
                raise ValueError("PiD Gemma needs a complete relative-file digest map")
        elif not isinstance(row["sha256"], str) or not re.fullmatch(r"[0-9a-f]{64}", row["sha256"]):
            raise ValueError("conditional asset needs a lowercase SHA-256: " + role)
        if not isinstance(row["revision"], str) or not 1 <= len(row["revision"]) <= 128:
            raise ValueError("conditional asset needs a source revision: " + role)
        normalized[role] = dict(row)
    return variant, normalized


def prepare_assets(work, source, assets):
    """Verify every external byte and copy it into the owned proof directory."""
    hub = Path(os.environ["FLUX1_HUB"]).resolve(strict=True)
    staged = work / "conditional-assets"
    staged.mkdir()
    rows = {}
    for role, row in assets.items():
        original = Path(row["path"])
        resolved = original.resolve(strict=True)
        if not resolved.is_relative_to(hub):
            raise ValueError("conditional asset lies outside the configured model hub: " + role)
        if role == "pid_gemma":
            if not resolved.is_dir():
                raise ValueError("PiD Gemma asset must be a directory")
            actual_files = {p.relative_to(resolved).as_posix() for p in resolved.rglob("*") if p.is_file()}
            if actual_files != set(row["files"]):
                raise ValueError("PiD Gemma file inventory changed")
            for name, digest in row["files"].items():
                path = (resolved / name).resolve(strict=True)
                if not path.is_relative_to(resolved) or sha(path) != digest:
                    raise ValueError("PiD Gemma file changed: " + name)
                destination = staged / "pid_gemma" / name
                destination.parent.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(path, destination)
                if sha(destination) != digest:
                    raise ValueError("staged PiD Gemma file changed: " + name)
            rows[role] = {**row, "resolved": str(resolved), "staged": str(staged / "pid_gemma"),
                          "staging": "copy", "bytes": sum((staged / "pid_gemma" / name).stat().st_size
                                                    for name in row["files"])}
            continue
        if not resolved.is_file() or resolved.stat().st_nlink < 1:
            raise ValueError("conditional asset is not a file: " + role)
        actual = sha(resolved)
        if actual != row["sha256"]:
            raise ValueError("conditional asset digest changed: " + role)
        destination = staged / (role + ".safetensors")
        shutil.copyfile(resolved, destination)
        if sha(destination) != actual:
            raise ValueError("staged conditional asset digest changed: " + role)
        rows[role] = {**row, "resolved": str(resolved), "staged": str(destination),
                      "staging": "copy", "bytes": destination.stat().st_size}
    if "ip_adapter" in rows:
        ip = staged / "ip"
        (ip / "image_encoder").mkdir(parents=True)
        os.link(staged / "ip_adapter.safetensors", ip / "ip_adapter.safetensors")
        os.link(staged / "image_encoder.safetensors", ip / "image_encoder" / "model.safetensors")
    if "scrfd" in rows:
        face = staged / "face"
        face.mkdir()
        for role, name in (("scrfd", "scrfd_10g"), ("arcface", "arcface_iresnet100"),
                           ("bisenet", "bisenet_parsing")):
            os.link(staged / (role + ".safetensors"), face / (name + ".safetensors"))
    reference = source / REFERENCE
    if sha(reference) != REFERENCE_SHA256:
        raise ValueError("repository PuLID reference image changed")
    staged_reference = staged / "reference.png"
    shutil.copyfile(reference, staged_reference)
    if sha(staged_reference) != REFERENCE_SHA256:
        raise ValueError("staged PuLID reference image changed")
    rows["reference_image"] = {"path": str(reference), "staged": str(staged_reference),
                               "staging": "copy", "sha256": REFERENCE_SHA256,
                               "revision": "repository-evidence/sc-16956", "bytes": staged_reference.stat().st_size}
    return staged, rows


def variant(case):
    return CASES[case][0]


def family(case, backend):
    pulid = case.startswith("pulid-")
    if backend == "mlx":
        return "mlx-gen-pulid" if pulid else "mlx-gen-flux"
    return "candle-gen-pulid" if pulid else "candle-gen-flux"


def write_control_source(work, case):
    """A fixed RGB hint; kinds change source bytes, while zero uses the pose source."""
    if not case.startswith("control-"):
        raise ValueError("control source requested for an identity route")
    kind = "pose" if case == "control-zero" else case.split("-", 1)[1]
    size = 256
    pixels = bytearray(size * size * 3)

    def put(x, y, value):
        if 0 <= x < size and 0 <= y < size:
            offset = (y * size + x) * 3
            pixels[offset:offset + 3] = bytes([value]) * 3

    if kind == "pose":
        pixels[:] = bytes([12]) * len(pixels)
        for y in range(size // 6, 5 * size // 6):
            for dx in range(-2, 3):
                put(size // 2 + dx, y, 235)
        for t in range(size // 4):
            for delta in (-1, 0, 1):
                put(size // 2 - t, size // 3 + t + delta, 235)
                put(size // 2 + t, size // 3 + t + delta, 235)
                put(size // 2 - t, 5 * size // 6 + delta, 235)
                put(size // 2 + t, 5 * size // 6 + delta, 235)
    elif kind == "canny":
        lo, hi = size // 5, 4 * size // 5
        for x in range(lo, hi + 1):
            put(x, lo, 255)
            put(x, hi, 255)
            put(lo + x - lo, lo + x - lo, 255)
            put(hi - (x - lo), lo + x - lo, 255)
        for y in range(lo, hi + 1):
            put(lo, y, 255)
            put(hi, y, 255)
    elif kind == "depth":
        for y in range(size):
            value = 255 * y // (size - 1)
            for x in range(size):
                put(x, y, value)
    else:
        raise ValueError("unknown control source kind")
    path = work / "control.rgb"
    path.write_bytes(pixels)
    return path, hashlib.sha256(pixels).hexdigest()
