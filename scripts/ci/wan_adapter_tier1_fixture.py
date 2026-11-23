"""Pinned-header synthetic fixture for the MLX Wan A14B dense adapter fold."""

import hashlib
import json
import math
from pathlib import Path
import re
import struct
from urllib.parse import urlsplit
from urllib.request import HTTPRedirectHandler, Request, build_opener

from scripts.ci.flux1_pair import fixture
from scripts.ci.model_inference_proofs import unlinked_path

PROFILE = "wan-a14b-t2v-dense-fold"
PACKAGE = "mlx-gen-wan"
REPOSITORY = "SceneWorks/wan2.2-t2v-a14b-mlx"
REVISION = "991eb255c544bbb2e1f1e07da4355c2f0a5337b7"
HEADERS = {
    "bf16/high_noise_model.safetensors": "6b279f0a1165e5df52e589de3ca2f25090e2ff64695ebde083d8a6d395a92a17",
    "bf16/low_noise_model.safetensors": "4516f62b00702cf4a491da8f4bc56c939474f6443ebef9d222410b48666f0ef8",
}
ADAPTERS = (("high", 0.75), ("shared-a", 0.5), ("low", -0.625), ("shared-b", 1.25))
ORDER = {"high": [1, 3, 0], "low": [1, 3, 2]}
SHAPE = [16, 2, 4, 6]
TARGET = "blocks.0.self_attn.q"
PROBE = Path(__file__).with_name("wan_adapter_tier1_probe.rs.in")
FIXTURE_FILES = {"transformer.safetensors", "inputs.safetensors",
                 *(name + ".safetensors" for name, _ in ADAPTERS)}
MAX_HEADER_BYTES = 2**20
ASSEMBLED_WEIGHT_BYTES = 1_168_227_456


def package(_backend):
    return PACKAGE


def overlay(_read, _arm, backend, binary):
    if backend != "mlx":
        raise ValueError("Wan fixture requires MLX")
    return {f"crates/media/mlx-gen/{PACKAGE}/src/bin/{binary}.rs": PROBE.read_bytes()}


def contract(_backend="mlx"):
    return {
        "compared": "bare and per-expert component velocities; timing and peaks are diagnostic only",
        "source_snapshot": {"repository": REPOSITORY, "revision": REVISION, "headers_sha256": HEADERS},
        "geometry": {"model_width": 5120, "heads": 40, "depth": 1, "latent_shape": SHAPE,
                     "text_shape": [64, 4096], "retained_bf16_bytes": ASSEMBLED_WEIGHT_BYTES},
        "settings": {"stage": "dense fold and one velocity per bare/high/low arm", "timestep": 900,
                     "fixture_seed": fixture.SEED, "base_dtype": "BF16", "factor_dtype": "BF16",
                     "device": {"kind": "mlx-metal", "index": 0}, "latent_dtype": "F32",
                     "rank": 4, "target": TARGET, "adapters_in_load_order":
                         [{"id": name, "scale": scale} for name, scale in ADAPTERS],
                     "expected_expert_order": ORDER},
        "tolerance": {"class": "AP13(a)", "output": "bit-exact finite F32 for all three velocities",
                      "receipt": "exact ordering and target counts", "provenance": "SGA §7.3; routing-only migration"},
        "probe_sha256": fixture.sha(PROBE),
    }


def validate_hub_url(url):
    target = urlsplit(url)
    host = target.hostname or ""
    if (target.scheme != "https" or target.username is not None or target.password is not None
            or target.port not in (443, None)
            or not (host == "huggingface.co" or host.endswith((".huggingface.co", ".hf.co")))):
        raise ValueError("unexpected Hugging Face header download origin")


class HubRedirects(HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        validate_hub_url(newurl)
        return super().redirect_request(req, fp, code, msg, headers, newurl)


def fetch_range(url, start, end):
    """Require exact range support; never start reading an unbounded remote checkpoint body."""
    validate_hub_url(url)
    request = Request(url, headers={"Range": f"bytes={start}-{end}", "Accept-Encoding": "identity"})
    length = end - start + 1
    open_range = build_opener(HubRedirects()).open
    with open_range(request, timeout=30) as response:
        interval = re.fullmatch(r"bytes (\d+)-(\d+)/(\d+)", response.headers.get("Content-Range", ""))
        if (response.status != 206 or interval is None
                or tuple(map(int, interval.groups()[:2])) != (start, end)
                or int(interval.group(3)) <= end
                or response.headers.get("Content-Encoding", "identity") != "identity"):
            raise ValueError("Hub did not serve the exact bounded tensor-header byte range")
        data = response.read(length + 1)
    if len(data) != length:
        raise ValueError("truncated or oversized tensor-header response")
    return data


def unique_object(pairs):
    result = dict(pairs)
    if len(result) != len(pairs):
        raise ValueError("duplicate key in proof input")
    return result


def retained_geometry(raw):
    header = json.loads(raw, object_pairs_hook=unique_object)
    if not isinstance(header, dict):
        raise ValueError("invalid tensor header")
    geometry = {}
    for name, row in header.items():
        if name == "__metadata__" or re.match(r"blocks\.[1-9]\d*\.", name):
            continue
        if not isinstance(row, dict) or row.get("dtype") != "BF16":
            raise ValueError("retained Wan layers must be BF16")
        shape = row.get("shape")
        if not isinstance(shape, list) or not 1 <= len(shape) <= 4 or any(
            type(axis) is not int or not 1 <= axis <= 32768 for axis in shape
        ):
            raise ValueError("malformed Wan tensor dimensions")
        geometry[name] = shape
    if (len(geometry) != 42 or geometry.get(TARGET + ".weight") != [5120, 5120]
            or geometry.get("patch_embedding_proj.weight") != [5120, 64]
            or geometry.get("text_embedding_0.weight") != [5120, 4096]
            or geometry.get("time_projection.weight") != [30720, 5120]
            or sum(math.prod(shape) * 2 for shape in geometry.values()) != ASSEMBLED_WEIGHT_BYTES):
        raise ValueError("Wan production header or depth truncation differs from frozen dimensions")
    return dict(sorted(geometry.items()))


def fetch_pinned_header(repository, revision, path, expected, fetch=fetch_range):
    url = f"https://huggingface.co/{repository}/resolve/{revision}/{path}"
    length, = struct.unpack("<Q", fetch(url, 0, 7))
    if not 0 < length <= MAX_HEADER_BYTES:
        raise ValueError("tensor header exceeds the bounded metadata read")
    data = fetch(url, 8, 7 + length)
    if len(data) != length or hashlib.sha256(data).hexdigest() != expected:
        raise ValueError("tensor header differs from the product snapshot pin")
    return data


def fetch_geometry(fetch=fetch_range):
    layouts = [retained_geometry(fetch_pinned_header(REPOSITORY, REVISION, path, expected, fetch))
               for path, expected in HEADERS.items()]
    if layouts[0] != layouts[1]:
        raise ValueError("Wan expert layouts disagree")
    return layouts[0]


def materialize(root):
    geometry = fetch_geometry()
    root.mkdir()
    # Identical base matrices make the differing outputs depend on the routed adapters alone.
    fixture.write_component(root / "transformer.safetensors", geometry)
    fixture.write_component(root / "inputs.safetensors", {"latent": SHAPE, "text": [64, 4096]})
    for name, _ in ADAPTERS:
        # The public normalization path strips each recognized namespace. Distinct namespaces seed
        # the factors separately while targeting the same projection in all four files.
        prefix = {"high": "model.", "shared-a": "base_model.model.",
                  "low": "diffusion_model.", "shared-b": "model.diffusion_model."}[name]
        make = {prefix + TARGET + ".lora_A.weight": [4, 5120],
                prefix + TARGET + ".lora_B.weight": [5120, 4]}
        fixture.write_component(root / (name + ".safetensors"), make)
    return identify(root)


def identify(root, files=FIXTURE_FILES):
    if {file.name for file in root.iterdir()} != files:
        raise ValueError("fixture contains an unexpected file")
    for name in files:
        path = root / name
        unlinked_path(path, root)
        if not path.is_file() or path.lstat().st_nlink != 1:
            raise ValueError("fixture must contain only regular, unlinked files")
    return {name: fixture.sha(root / name) for name in sorted(files)}


def observe(stdout, out, _backend="mlx"):
    if stdout.splitlines().count("SGA_ADAPTER_TIER1_BODY " + PROFILE) != 1:
        raise ValueError("missing unique completed Wan Metal fixture body")
    row = json.loads(out.read_text(encoding="utf-8"), object_pairs_hook=unique_object)
    if (not isinstance(row, dict) or set(row) != {"device", "order", "applied", "shape", "velocity_bits"}
            or row["device"] != {"kind": "mlx-metal", "index": 0} or row["order"] != ORDER
            or row["applied"] != {"high": [1, 1, 1], "low": [1, 1, 1]} or row["shape"] != SHAPE):
        raise ValueError("Wan fixture device, shape or expert route differed from its contract")
    velocities = row["velocity_bits"]
    if not isinstance(velocities, dict) or set(velocities) != {"bare", "high", "low"}:
        raise ValueError("missing Wan negative or expert velocity")
    for bits in velocities.values():
        if (not isinstance(bits, list) or len(bits) != math.prod(SHAPE) or not all(
                type(bit) is int and 0 <= bit < 2**32 and bit & 0x7f800000 != 0x7f800000 for bit in bits)):
            raise ValueError("invalid or nonfinite Wan velocity")
    if len({tuple(bits) for bits in velocities.values()}) != 3:
        raise ValueError("Wan adapters did not each change the shared base velocity")
    return row
