#!/usr/bin/env python3
"""Produce the pinned CPU oracle for an ACTUAL H3 checkpoint probe (sc-26679 terminal repair).

This is an **adapter, not a second oracle**. It implements no projection, interpolation,
quantization or dequantization arithmetic of its own; every numeric value comes from a frozen
producer, imported read-only and never executed through ``main``:

* W4A8 and ConvRot INT8 expectations use the frozen W4A8 producer's existing pinned replays:
  ``quantize_int8_rowwise_convrot_replay`` (``int8_linear.cu`` quantize kernel),
  ``decode_weight_replay`` (``w4a8_gemm.cu`` dequant kernel),
  ``int_accumulate_replay`` (``cutlass_gemm_int8.cu`` int32 accumulation) and
  ``epilogue_replay`` (F32 scale epilogue). ConvRot INT8 weights are already int8 in the artifact,
  so they feed ``int_accumulate_replay`` directly; the same BF16 quantizer replay applies.
* Pruned curve expectations use the frozen pruned producer's ``pinned_definitions`` to exec the
  hash-verified WanGP ``_interpolated_table`` / ``AdalnProj`` / ``FinalLayer`` definitions from the
  caller's local ``--reference-dir``. The adapter mirrors the producer's exact positional parent
  calls on the torch ``meta`` device (no block-body allocation), passes the ACTUAL reduced rank as
  the AdaLN input width, materializes only the extracted ``adaln_proj.linear`` on CPU, and calls
  its pinned forward with the actual operands.

Provenance: the W4A8 producer is the arithmetic authority for the packed replays; the ConvRot
producer and its pinned upstream sources are the format/reference authority. Both script sha256
values (and the pruned script) are recorded exactly.

Activation dtype: BF16 for packed projections (the pinned replay is ``InputType = nv_bfloat16``)
and F32 for the pruned curve.

No GPU, no model load, no real-weight inference, no network. CPU torch only.

CLI:
    python3 scripts/h3_actual_checkpoint_oracle.py \
      --spec /abs/outside/actual-probe-spec.json \
      --selected-tensors /abs/outside/selected-tensors.safetensors \
      --reference-dir /abs/outside/pinned-reference \
      --output /abs/outside/cpu-oracle.json \
      --receipt /abs/outside/producer-receipt.json

Exit 0 only when every case produced a finite f32 expectation. On refusal this writes the receipt
and exits 2; it never writes a partial oracle.
"""
from __future__ import annotations

import argparse
import ast
import hashlib
import importlib.util
import json
import math
import struct
import sys
from pathlib import Path
from typing import Any

MAX_SPEC_BYTES = 8 * 1024 * 1024
MAX_SELECTED_BYTES = 512 * 1024 * 1024
MAX_CASES = 64
MAX_SELECTED_TENSORS = 8192
MAX_TENSOR_NAME = 512

MANIFEST_SCHEMA = "sceneworks.h3.actual_checkpoint_probe"
SPEC_SCHEMA = MANIFEST_SCHEMA  # the bridge consumes the SAME shared manifest
ORACLE_SCHEMA = "sceneworks.h3.actual_checkpoint_oracle"
RECEIPT_SCHEMA = "sceneworks.h3.actual_checkpoint_oracle_producer_receipt"
SCHEMA_VERSION = 1

REPO_ROOT = Path(__file__).resolve().parents[1]
FROZEN_W4A8 = "scripts/ci/generate_minimax_h3_w4a8_fixture.py"
FROZEN_CONVROT = "scripts/h3_convrot_int8_oracle.py"
FROZEN_PRUNED = "scripts/h3_pruned_adaln_oracle.py"

GROUP_SIZE = 16
CONVROT_GROUP = 256

KIND_W4A8 = "w4a8_projection"
KIND_CONVROT = "convrot_projection"
KIND_MIXED = "mixed_w4a8_convrot"
KIND_INTERPOLATION = "pruned_timestep_interpolation"
KIND_REDUCED = "pruned_reduced_projection"
KINDS = (KIND_W4A8, KIND_CONVROT, KIND_MIXED, KIND_INTERPOLATION, KIND_REDUCED)

W4A8_SUFFIXES = ("weight", "weight_codebook", "weight_s_rel", "weight_s_channel", "comfy_quant")
CONVROT_SUFFIXES = ("weight", "weight_scale", "comfy_quant")

# Canonical per-kind comparison bounds. The packed native outputs are BF16-rounded like the pinned
# epilogue, so they must agree exactly; the pruned bounds are the pinned fixture's own bounds.
KIND_TOLERANCE = {
    KIND_W4A8: 0.0,
    KIND_CONVROT: 0.0,
    KIND_MIXED: 0.0,
    KIND_INTERPOLATION: 2e-6,
    KIND_REDUCED: 2e-5,
}


class Refusal(Exception):
    def __init__(self, code: str, detail: str):
        super().__init__(detail)
        self.code = code
        self.detail = detail


def sha256_hex(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def read_bounded(path: Path, cap: int, what: str) -> bytes:
    if not path.is_absolute():
        raise Refusal("bad_path", f"{what} must be an absolute path: {path}")
    try:
        handle = path.open("rb")
    except OSError as error:
        raise Refusal("missing_input", f"{what} is unreadable: {error}") from error
    with handle:
        size = handle.seek(0, 2)
        handle.seek(0)
        if size > cap:
            raise Refusal("metadata_too_large", f"{what} is {size} bytes, cap is {cap}")
        # Read at most cap+1: a file that grows after stat cannot make this unbounded.
        data = handle.read(cap + 1)
    if len(data) > cap:
        raise Refusal("metadata_too_large", f"{what} exceeded the {cap}-byte cap while reading")
    return data


def obj(value: Any, ctx: str) -> dict:
    if not isinstance(value, dict):
        raise Refusal("spec_malformed", f"{ctx} must be an object")
    return value


def require_str(raw: dict, key: str, ctx: str) -> str:
    if key not in raw or not isinstance(raw[key], str):
        raise Refusal("spec_malformed", f"{ctx}.{key} must be a string")
    return raw[key]


def require_sha(value: Any, ctx: str) -> str:
    if not isinstance(value, str) or len(value) != 64 or any(c not in "0123456789abcdefABCDEF" for c in value):
        raise Refusal("spec_malformed", f"{ctx} must be 64 hex characters")
    return value.lower()


def require_positive_shape(shape: Any, ctx: str) -> list[int]:
    if not isinstance(shape, list) or not shape or len(shape) > 8:
        raise Refusal("spec_malformed", f"{ctx} must be a non-empty 1..8-dimensional list")
    out = []
    for value in shape:
        if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
            raise Refusal("spec_malformed", f"{ctx} entries must be positive integers")
        out.append(value)
    return out


def require_finite_values(values: Any, count: int, ctx: str) -> list[float]:
    if not isinstance(values, list) or len(values) != count:
        raise Refusal("spec_malformed", f"{ctx} must contain {count} values")
    out = []
    for value in values:
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(float(value)):
            raise Refusal("spec_malformed", f"{ctx} entries must be finite numbers")
        out.append(float(value))
    return out


def validate_case(case: Any, index: int) -> None:
    ctx = f"cases[{index}]"
    case = obj(case, ctx)
    for key in ("id", "kind", "select", "tolerance"):
        if key not in case:
            raise Refusal("spec_malformed", f"{ctx}.{key} is required")
    case_id = case["id"]
    if not isinstance(case_id, str) or not case_id or len(case_id) > 128:
        raise Refusal("spec_malformed", f"{ctx}.id is invalid")
    if case["kind"] not in KINDS:
        raise Refusal("spec_malformed", f"{ctx}.kind must be one of {KINDS}")
    select = obj(case["select"], f"{ctx}.select")
    for key in ("base", "convrot_base", "table", "prefix", "weight", "bias_tensor"):
        if key in select and not isinstance(select[key], str):
            raise Refusal("spec_malformed", f"{ctx}.select.{key} must be a string")
    aliases = select.get("aliases") or {}
    if not isinstance(aliases, dict):
        raise Refusal("spec_malformed", f"{ctx}.select.aliases must be an object")
    for raw, canonical in aliases.items():
        if not isinstance(raw, str) or not isinstance(canonical, str):
            raise Refusal("spec_malformed", f"{ctx}.select.aliases must map strings to strings")
        for name in (raw, canonical):
            if len(name) > MAX_TENSOR_NAME or "/" in name or "\\" in name:
                raise Refusal("spec_malformed", f"{ctx}: invalid tensor name {name!r}")
    tolerance = obj(case["tolerance"], f"{ctx}.tolerance")
    if not isinstance(tolerance.get("max_abs"), (int, float)) or tolerance["max_abs"] < 0:
        raise Refusal("spec_malformed", f"{ctx}.tolerance.max_abs must be non-negative")
    inp = obj(case.get("input"), f"{ctx}.input")
    if case["kind"] in (KIND_W4A8, KIND_CONVROT, KIND_MIXED):
        if inp.get("dtype") != "BF16":
            raise Refusal("input_dtype", f"{ctx}: packed projections require input.dtype=BF16")
        shape = require_positive_shape(inp.get("shape"), f"{ctx}.input.shape")
        if len(shape) != 2:
            raise Refusal("input_shape", f"{ctx}: input must be [rows, k]")
        total = shape[0] * shape[1]
        inp["shape"] = shape
        inp["values"] = require_finite_values(inp.get("values"), total, f"{ctx}.input.values")
    elif case["kind"] == KIND_INTERPOLATION:
        times = inp.get("timesteps")
        if not isinstance(times, list) or not times:
            raise Refusal("input_shape", f"{ctx}: input.timesteps are required")
        inp["timesteps"] = require_finite_values(times, len(times), f"{ctx}.input.timesteps")
    else:
        if inp.get("dtype", "F32") != "F32":
            raise Refusal("input_dtype", f"{ctx}: pruned projection requires input.dtype=F32")
        shape = require_positive_shape(inp.get("shape"), f"{ctx}.input.shape")
        if len(shape) != 2:
            raise Refusal("input_shape", f"{ctx}: input must be [rows, rank]")
        total = shape[0] * shape[1]
        inp["shape"] = shape
        inp["values"] = require_finite_values(inp.get("values"), total, f"{ctx}.input.values")


def parse_spec(path: Path) -> dict:
    """Parse the shared probe MANIFEST (the bridge consumes the same artifact/staging/cases shape
    the native consumer reads; only the `oracle` block may still be pending until this bridge emits
    it). Validation is structural here; artifact stat/header and the SHA'd staging chain are
    verified in ``run`` after parsing, before any selected payload is read."""
    data = read_bounded(path, MAX_SPEC_BYTES, "spec")
    try:
        spec = json.loads(data)
    except json.JSONDecodeError as error:
        raise Refusal("spec_malformed", f"spec is not JSON: {error}") from error
    spec = obj(spec, "spec")
    if spec.get("schema") != MANIFEST_SCHEMA or spec.get("schema_version") != SCHEMA_VERSION:
        raise Refusal("spec_malformed", f"spec must be {MANIFEST_SCHEMA} v{SCHEMA_VERSION}")
    backend = spec.get("backend", "both")
    if backend not in ("candle", "mlx", "both"):
        raise Refusal("spec_malformed", f"spec.backend must be candle|mlx|both, got {backend!r}")
    artifact = obj(spec.get("artifact"), "spec.artifact")
    require_str(artifact, "path", "spec.artifact")
    artifact_sha = require_sha(artifact.get("sha256"), "spec.artifact.sha256")
    if not isinstance(artifact.get("bytes"), int) or isinstance(artifact.get("bytes"), bool) or artifact["bytes"] <= 0:
        raise Refusal("spec_malformed", "spec.artifact.bytes must be a positive integer")
    artifact["path"] = artifact["path"]
    artifact["sha256"] = artifact_sha
    staging = obj(spec.get("staging"), "spec.staging")
    require_str(staging, "receipt_path", "spec.staging")
    require_str(staging, "receipt_sha256", "spec.staging")
    require_sha(staging.get("verified_artifact_sha256"), "spec.staging.verified_artifact_sha256")
    cases = spec.get("cases")
    if not isinstance(cases, list) or not cases:
        raise Refusal("spec_malformed", "spec.cases must be a non-empty list")
    if len(cases) > MAX_CASES:
        raise Refusal("limits_exceeded", f"{len(cases)} cases exceeds {MAX_CASES}")
    for index, case in enumerate(cases):
        validate_case(case, index)
    return spec


# --------------------------------------------------------------------------------------
# Bounded mini-safetensors reader (consumer of the extractor output; no tensor math)
# --------------------------------------------------------------------------------------


def read_selected_safetensors(path: Path) -> dict[str, dict[str, Any]]:
    data = read_bounded(path, MAX_SELECTED_BYTES, "selected tensors")
    if len(data) < 8:
        raise Refusal("selected_malformed", "selected tensors file is truncated")
    (header_len,) = struct.unpack("<Q", data[:8])
    if header_len > 100_000_000:
        raise Refusal("selected_malformed", "selected tensors header is too large")
    data_start = 8 + header_len
    if data_start > len(data):
        raise Refusal("selected_malformed", "selected tensors header extends past the file")
    try:
        header = json.loads(data[8:data_start])
    except json.JSONDecodeError as error:
        raise Refusal("selected_malformed", f"selected tensors header: {error}") from error
    header = obj(header, "selected tensors header")
    tensors: dict[str, dict[str, Any]] = {}
    spans: list[tuple[int, int]] = []
    for name, entry in header.items():
        if name == "__metadata__":
            continue
        if not isinstance(entry, dict):
            raise Refusal("selected_malformed", f"tensor {name!r} header is not an object")
        dtype = entry.get("dtype")
        shape = entry.get("shape")
        offsets = entry.get("data_offsets")
        if not isinstance(dtype, str) or not isinstance(shape, list) or not isinstance(offsets, list) or len(offsets) != 2:
            raise Refusal("selected_malformed", f"tensor {name!r} header is incomplete")
        start, end = int(offsets[0]), int(offsets[1])
        if start < 0 or end < start:
            raise Refusal("selected_malformed", f"tensor {name!r} has invalid offsets")
        begin, stop = data_start + start, data_start + end
        if stop > len(data):
            raise Refusal("selected_malformed", f"tensor {name!r} extends past the file")
        spans.append((start, end))
        tensors[name] = {
            "dtype": dtype,
            "shape": [int(v) for v in shape],
            "bytes": bytes(data[begin:stop]),
        }
    expected = 0
    for start, end in sorted(spans):
        if start != expected:
            raise Refusal("selected_malformed", "selected tensors payload is not contiguous")
        expected = end
    if expected != len(data) - data_start:
        raise Refusal("selected_malformed", "selected tensors payload does not cover the file")
    return tensors


# --------------------------------------------------------------------------------------
# Shared-manifest artifact / staging validation (metadata only; no full-file rehash)
# --------------------------------------------------------------------------------------


def read_artifact_header(path: Path) -> dict[str, dict[str, Any]]:
    """Read ONLY the safetensors header of the actual artifact (bounded), validating the declared
    tensor ranges. The whole-file digest is never recomputed here."""
    if not path.is_absolute():
        raise Refusal("bad_path", f"artifact path must be absolute: {path}")
    try:
        handle = path.open("rb")
    except OSError as error:
        raise Refusal("artifact_missing", f"artifact is unreadable: {error}") from error
    with handle:
        size = handle.seek(0, 2)
        handle.seek(0)
        prefix = handle.read(8)
        if len(prefix) != 8:
            raise Refusal("artifact_header_invalid", "artifact is truncated")
        (header_len,) = struct.unpack("<Q", prefix)
        if header_len > 100_000_000:
            raise Refusal("artifact_header_invalid", "artifact header is too large")
        data_start = 8 + header_len
        if data_start > size:
            raise Refusal("artifact_header_invalid", "artifact header extends past the file")
        raw = handle.read(header_len)
    if len(raw) != header_len:
        raise Refusal("artifact_header_invalid", "artifact header is truncated")
    try:
        header = json.loads(raw)
    except json.JSONDecodeError as error:
        raise Refusal("artifact_header_invalid", f"artifact header: {error}") from error
    header = obj(header, "artifact header")
    tensors: dict[str, dict[str, Any]] = {}
    spans: list[tuple[int, int]] = []
    for name, entry in header.items():
        if name == "__metadata__":
            continue
        if not isinstance(entry, dict):
            raise Refusal("artifact_header_invalid", f"tensor {name!r} header is not an object")
        dtype, shape, offsets = entry.get("dtype"), entry.get("shape"), entry.get("data_offsets")
        if (
            not isinstance(dtype, str)
            or not isinstance(shape, list)
            or not isinstance(offsets, list)
            or len(offsets) != 2
        ):
            raise Refusal("artifact_header_invalid", f"tensor {name!r} header is incomplete")
        start, end = int(offsets[0]), int(offsets[1])
        if start < 0 or end < start:
            raise Refusal("artifact_header_invalid", f"tensor {name!r} has invalid offsets")
        spans.append((start, end))
        tensors[name] = {"dtype": dtype, "shape": [int(v) for v in shape], "data_bytes": end - start}
    expected = 0
    for start, end in sorted(spans):
        if start != expected:
            raise Refusal("artifact_header_invalid", "artifact payload is not contiguous")
        expected = end
    if expected != size - data_start:
        raise Refusal("artifact_header_invalid", "artifact payload does not cover the file")
    return tensors


def validate_artifact_chain(spec: dict) -> tuple[Path, dict[str, dict[str, Any]]]:
    artifact = obj(spec["artifact"], "artifact")
    path = Path(artifact["path"])
    if not path.is_absolute():
        raise Refusal("bad_path", f"artifact path must be absolute: {path}")
    try:
        size = path.stat().st_size
    except OSError as error:
        raise Refusal("artifact_missing", f"artifact is unreadable: {error}") from error
    if not path.is_file():
        raise Refusal("artifact_not_file", f"artifact is not a file: {path}")
    if size != artifact["bytes"]:
        raise Refusal(
            "artifact_size_mismatch",
            f"artifact declares {artifact['bytes']} bytes, disk has {size}",
        )
    return path, read_artifact_header(path)


def validate_staging_chain(spec: dict, artifact_path: Path) -> dict[str, Any]:
    staging = obj(spec["staging"], "staging")
    receipt_path = Path(staging["receipt_path"])
    receipt_bytes = read_bounded(receipt_path, 4 * 1024 * 1024, "staging projection")
    if sha256_hex(receipt_bytes) != staging["receipt_sha256"]:
        raise Refusal("staging_receipt_digest_mismatch", "staging projection digest mismatch")
    try:
        projection = json.loads(receipt_bytes)
    except json.JSONDecodeError as error:
        raise Refusal("staging_binding_mismatch", f"staging projection is not JSON: {error}") from error
    projection = obj(projection, "staging projection")
    if projection.get("schema") != "sceneworks.h3.staging_receipt" or projection.get("schema_version") != 1:
        raise Refusal("staging_binding_mismatch", "staging projection schema/version mismatch")
    if not isinstance(projection.get("producer"), str) or not projection["producer"].strip():
        raise Refusal("staging_binding_mismatch", "staging projection must name its producer")
    if projection.get("full_file_digest_verified") is not True:
        raise Refusal("staging_binding_mismatch", "staging must declare full_file_digest_verified")
    if staging.get("verified_artifact_sha256") != spec["artifact"]["sha256"]:
        raise Refusal("staging_binding_mismatch", "manifest verified artifact sha256 mismatch")
    if projection.get("artifact_path") != str(artifact_path):
        raise Refusal("staging_binding_mismatch", "staging artifact path does not match the manifest")
    if projection.get("artifact_sha256") != spec["artifact"]["sha256"]:
        raise Refusal("staging_binding_mismatch", "staging artifact sha256 does not match the manifest")
    if projection.get("artifact_bytes") != spec["artifact"]["bytes"]:
        raise Refusal("staging_binding_mismatch", "staging artifact byte count does not match the manifest")
    parent_path = Path(projection.get("parent_receipt_path") or "")
    parent_bytes = read_bounded(parent_path, 64 * 1024 * 1024, "parent staging receipt")
    if sha256_hex(parent_bytes) != projection.get("parent_receipt_sha256"):
        raise Refusal("staging_receipt_digest_mismatch", "parent staging receipt digest mismatch")
    return {
        "receipt_path": str(receipt_path),
        "receipt_sha256": staging["receipt_sha256"],
        "producer": projection["producer"],
        "parent_receipt_path": str(parent_path),
        "parent_receipt_sha256": projection["parent_receipt_sha256"],
    }


def validate_selection_binding(entries: list[dict], header: dict, case_id: str) -> None:
    for entry in entries:
        tensor = header.get(entry["name"])
        if tensor is None:
            raise Refusal(
                "selection_missing_actual_artifact",
                f"{case_id}: {entry['name']!r} is not in the actual artifact header",
            )
        if (
            tensor["dtype"] != entry["dtype"]
            or tensor["shape"] != entry["shape"]
            or tensor["data_bytes"] != entry["bytes"]
        ):
            raise Refusal(
                "selection_binding_mismatch",
                f"{case_id}: {entry['name']!r} selected {entry['dtype']}/{entry['shape']}/"
                f"{entry['bytes']} bytes != actual artifact {tensor['dtype']}/"
                f"{tensor['shape']}/{tensor['data_bytes']} bytes",
            )


# --------------------------------------------------------------------------------------
# Frozen producers (imported read-only; every script guards `main`)
# --------------------------------------------------------------------------------------


def load_frozen(relative: str):
    path = REPO_ROOT / relative
    if not path.is_file():
        raise Refusal("frozen_producer_missing", f"frozen producer not found: {relative}")
    spec = importlib.util.spec_from_file_location(f"frozen_{path.stem}", path)
    if spec is None or spec.loader is None:
        raise Refusal("frozen_producer_missing", f"cannot import {relative}")
    module = importlib.util.module_from_spec(spec)
    # Dataclasses resolve their module through `sys.modules`; register before executing.
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def frozen_provenance() -> list[dict[str, Any]]:
    # Read literal upstream pins without importing a producer (and never calling its main).
    # The producer's content revision is its SHA256; no unobserved git revision is invented.
    pin_names = {
        FROZEN_W4A8: {"LOADER_REVISION", "KITCHEN_MAIN", "KITCHEN_PR_HEAD", "KITCHEN_MERGED_TREE", "FROZEN_ARTIFACT_TREE"},
        FROZEN_CONVROT: {"KITCHEN", "PRODUCER", "LOADER", "HF"},
        FROZEN_PRUNED: {"REVISION"},
    }
    records = []
    for relative, required in pin_names.items():
        data = (REPO_ROOT / relative).read_bytes()
        pins = {}
        for node in ast.parse(data).body:
            if isinstance(node, ast.Assign) and isinstance(node.value, ast.Constant) and isinstance(node.value.value, str):
                for target in node.targets:
                    if isinstance(target, ast.Name) and target.id in required:
                        pins[target.id] = node.value.value
        if set(pins) != required:
            raise Refusal("frozen_producer_provenance", f"missing literal upstream pins in {relative}")
        digest = sha256_hex(data)
        records.append({"script": relative, "sha256": digest, "source_revision_sha256": digest, "upstream_revisions": pins})
    return records


# --------------------------------------------------------------------------------------
# Byte decoding only (every numeric value is computed by a frozen function)
# --------------------------------------------------------------------------------------


def f32_values(tensor: dict, name: str) -> list[float]:
    dtype, data = tensor["dtype"], tensor["bytes"]
    if dtype == "F32":
        return list(struct.unpack(f"<{len(data) // 4}f", data))
    if dtype == "F16":
        return [float(value) for value in struct.unpack(f"<{len(data) // 2}e", data)]
    if dtype == "BF16":
        bits = struct.unpack(f"<{len(data) // 2}H", data)
        return [struct.unpack("<f", struct.pack("<I", value << 16))[0] for value in bits]
    raise Refusal("selected_dtype", f"{name}: cannot widen {dtype} to F32")


def f32_tensor(tensor: dict, name: str):
    return _torch().tensor(f32_values(tensor, name), dtype=_torch().float32)


def rows_i8(tensor: dict, name: str) -> list[list[int]]:
    if tensor["dtype"] not in ("I8", "U8") or len(tensor["shape"]) != 2:
        raise Refusal("selected_dtype", f"{name}: expected I8 [n, k]")
    width = tensor["shape"][1]
    values = list(struct.unpack(f"<{len(tensor['bytes'])}b", tensor["bytes"]))
    return [values[r * width:(r + 1) * width] for r in range(tensor["shape"][0])]


def rows_u8(tensor: dict, name: str) -> list[list[int]]:
    if tensor["dtype"] not in ("U8", "F8_E4M3") or len(tensor["shape"]) != 2:
        raise Refusal("selected_dtype", f"{name}: expected U8/F8_E4M3 [n, k]")
    width = tensor["shape"][1]
    values = list(tensor["bytes"])
    return [values[r * width:(r + 1) * width] for r in range(tensor["shape"][0])]


def vector_f32(tensor: dict, name: str) -> list[float]:
    if tensor["dtype"] != "F32" or len(tensor["shape"]) != 1:
        raise Refusal("selected_dtype", f"{name}: expected F32 [n]")
    return f32_values(tensor, name)


def input_rows(case: dict, width: int) -> list[list[float]]:
    inp = obj(case.get("input"), f"{case['id']}.input")
    shape, values = inp.get("shape"), inp.get("values")
    if not isinstance(shape, list) or len(shape) != 2 or shape[1] != width:
        raise Refusal("input_shape", f"{case['id']}: input shape must be [rows, {width}]")
    if not isinstance(values, list) or len(values) != shape[0] * width:
        raise Refusal("input_shape", f"{case['id']}: input values do not match the shape")
    return [[float(values[r * width + c]) for c in range(width)] for r in range(shape[0])]


def flat(rows: list[list[float]]) -> list[float]:
    return [value for row in rows for value in row]


def input_binding(case: dict) -> dict:
    inp = obj(case.get("input"), f"{case['id']}.input")
    if "timesteps" in inp:
        values = [float(v) for v in inp["timesteps"]]
        dtype = "TIMESTEPS"
        shape = [len(values)]
    else:
        values = [float(v) for v in inp.get("values") or []]
        dtype = inp.get("dtype", "F32")
        shape = [int(v) for v in inp.get("shape") or []]
    digest = sha256_hex(b"".join(struct.pack("<f", value) for value in values))
    return {"dtype": dtype, "shape": shape, "values_sha256": digest}


# --------------------------------------------------------------------------------------
# Kinds
# --------------------------------------------------------------------------------------


def raw_for(select: dict, canonical: str) -> str:
    for raw, mapped in (select.get("aliases") or {}).items():
        if mapped == canonical:
            return raw
    return canonical


def tensor_at(tensors: dict, select: dict, canonical: str) -> dict:
    raw = raw_for(select, canonical)
    if raw not in tensors:
        raise Refusal("selection_missing", f"selected tensors have no {raw!r}")
    return tensors[raw]


def w4a8_rows(case: dict, tensors: dict, w4) -> list[list[float]]:
    select = case["select"]
    base = select["base"]
    weight = rows_i8(tensor_at(tensors, select, f"{base}.weight"), f"{base}.weight")
    n, packed_cols = len(weight), len(weight[0])
    k = packed_cols * 2
    if k % CONVROT_GROUP:
        raise Refusal("convrot_group_unsupported", f"{case['id']}: k={k} is not a multiple of {CONVROT_GROUP}")
    srel = rows_u8(tensor_at(tensors, select, f"{base}.weight_s_rel"), f"{base}.weight_s_rel")
    codebook = vector_f32(tensor_at(tensors, select, f"{base}.weight_codebook"), f"{base}.weight_codebook")
    s_channel = vector_f32(tensor_at(tensors, select, f"{base}.weight_s_channel"), f"{base}.weight_s_channel")
    if len(codebook) != 16 or len(srel) != n or len(s_channel) != n:
        raise Refusal("selected_dtype", f"{case['id']}: inconsistent W4A8 companion shapes")
    bias = vector_f32(tensor_at(tensors, select, f"{base}.bias"), f"{base}.bias") if select.get("bias") else None
    if select.get("bias") and len(bias) != n:
        raise Refusal("selected_dtype", f"{case['id']}: W4A8 bias length is not {n}")
    packed = [[byte & 0xFF for byte in row] for row in weight]
    decoded = w4.decode_weight_replay(packed, srel, codebook, k, group=GROUP_SIZE)
    rows = w4.to_bf16_rows(input_rows(case, k))
    _, xq, xs, _ = w4.quantize_int8_rowwise_convrot_replay(rows)
    acc, _, _ = w4.int_accumulate_replay(xq, decoded, k_tile=CONVROT_GROUP)
    return [
        [w4.epilogue_replay(acc[r][c], xs[r], s_channel[c], bias[c] if bias else None) for c in range(n)]
        for r in range(len(rows))
    ]


def convrot_rows(case: dict, tensors: dict, w4) -> list[list[float]]:
    select = case["select"]
    base = select["base"]
    descriptor = tensor_at(tensors, select, f"{base}.comfy_quant")
    try:
        declared = json.loads(descriptor["bytes"])
    except json.JSONDecodeError as error:
        raise Refusal("selected_dtype", f"{case['id']}: comfy_quant is not JSON: {error}") from error
    if descriptor["dtype"] != "U8" or int(declared.get("convrot_groupsize", 0)) != CONVROT_GROUP:
        raise Refusal(
            "convrot_group_unsupported",
            f"{case['id']}: convrot_groupsize={declared.get('convrot_groupsize')!r}; the pinned "
            f"BF16 replay covers group {CONVROT_GROUP}",
        )
    weight = rows_i8(tensor_at(tensors, select, f"{base}.weight"), f"{base}.weight")
    n, k = len(weight), len(weight[0])
    if k % CONVROT_GROUP:
        raise Refusal("convrot_group_unsupported", f"{case['id']}: k={k} is not a multiple of {CONVROT_GROUP}")
    scale = tensor_at(tensors, select, f"{base}.weight_scale")
    if scale["dtype"] != "F32" or scale["shape"] != [n, 1]:
        raise Refusal("selected_dtype", f"{case['id']}: ConvRot weight_scale must be F32 [{n}, 1]")
    # Preserve the stored [N,1] identity. The pinned rowwise epilogue consumes its N values.
    s_channel = f32_values(scale, f"{base}.weight_scale")
    bias = vector_f32(tensor_at(tensors, select, f"{base}.bias"), f"{base}.bias") if select.get("bias") else None
    if select.get("bias") and len(bias) != n:
        raise Refusal("selected_dtype", f"{case['id']}: ConvRot bias length is not {n}")
    rows = w4.to_bf16_rows(input_rows(case, k))
    _, xq, xs, _ = w4.quantize_int8_rowwise_convrot_replay(rows)
    acc, _, _ = w4.int_accumulate_replay(xq, weight, k_tile=CONVROT_GROUP)
    return [
        [w4.epilogue_replay(acc[r][c], xs[r], s_channel[c], bias[c] if bias else None) for c in range(n)]
        for r in range(len(rows))
    ]


def mixed_rows(case: dict, tensors: dict, w4) -> list[list[float]]:
    select = case["select"]
    primary = dict(case)
    primary["select"] = dict(select)
    primary["select"]["bias"] = bool(select.get("w4a8_bias"))
    companion = dict(case)
    companion["select"] = dict(select)
    companion["select"]["base"] = select["convrot_base"]
    companion["select"]["bias"] = bool(select.get("convrot_bias"))
    left, right = w4a8_rows(primary, tensors, w4), convrot_rows(companion, tensors, w4)
    if not left or not right or len(left) != len(right):
        raise Refusal("shape_mismatch", f"{case['id']}: mixed legs are incompatible")
    return [a + b for a, b in zip(left, right)]


def pruned_namespace(reference_dir: Path, pruned) -> dict:
    namespace = {
        "torch": _torch(),
        "nn": _nn(),
        "F": _F(),
        "math": math,
        "pay_attention": pruned.attention,
    }
    pruned.pinned_definitions(
        reference_dir,
        "prune_checkpoint.py",
        ["_factor_curve", "_compress_projection", "_interpolated_table"],
        namespace,
    )
    pruned.pinned_definitions(
        reference_dir,
        "transformer.py",
        ["_take", "_modulate", "_gated_residual", "MLP", "Attention", "AdalnProj", "DiTBlock", "FinalLayer"],
        namespace,
    )
    pruned.pinned_definitions(
        reference_dir,
        "components/packing.py",
        ["build_row_timesteps"],
        namespace,
    )
    return namespace


def projection_linear(ns, config: dict, prefix: str, actual_weight, actual_bias, rank: int):
    """Build ONLY the pinned AdaLN projection, never the block body.

    The producer's exact positional parent calls (``scripts/h3_pruned_adaln_oracle.py`` lines
    78-81) are mirrored, but on the torch ``meta`` device, so no body storage is allocated. The
    AdaLN input width is the ACTUAL selected rank (the reduced table rank), not
    ``config["time_embed_dim"]`` (the original unpruned embedding width) — the producer passes the
    rank there. Only the extracted ``adaln_proj.linear`` (out x rank) is materialized on CPU, then
    its pinned forward is called with the actual operands.
    """
    torch = _torch()
    hidden = int(config["hidden_size"])
    if prefix == "norm_out.linear":
        patch = int(config["in_channels"])
        for dim in config.get("patch_size", [1, 1, 1]):
            patch *= int(dim)
        parent = ns["FinalLayer"](
            hidden,
            int(rank),
            patch,
            int(config["audio_in_channels"]),
            float(config["final_norm_eps"]),
            apply_silu=False,
            adaln_dtype=torch.float32,
            dtype=torch.float32,
            device="meta",
        )
    else:
        parent = ns["DiTBlock"](
            hidden,
            int(config["num_attention_heads"]),
            int(config["attention_head_dim"]),
            int(config["ffn_dim"]),
            int(rank),
            float(config["norm_eps"]),
            float(config["qk_norm_eps"]),
            apply_silu=False,
            adaln_dtype=torch.float32,
            ffn_chunk_size=0,
            dtype=torch.float32,
            device="meta",
        )
    linear = getattr(getattr(parent, "adaln_proj", None), "linear", None)
    if linear is None:
        raise Refusal("adaln_adapter_unsupported", "pinned module has no adaln_proj.linear")
    # Materialize only this submodule; the parent body stays on meta. Bounded to out x rank.
    linear.to_empty(device="cpu")
    if tuple(linear.weight.shape) != tuple(actual_weight.shape) or tuple(linear.bias.shape) != tuple(actual_bias.shape):
        raise Refusal(
            "shape_mismatch",
            f"reduced operand shapes {tuple(actual_weight.shape)}/{tuple(actual_bias.shape)} do not "
            f"match the pinned projection [{tuple(linear.weight.shape)}]/[{tuple(linear.bias.shape)}]",
        )
    with torch.no_grad():
        linear.weight.copy_(actual_weight)
        linear.bias.copy_(actual_bias)
    if hasattr(linear, "device"):
        # A custom pinned Linear may consult `self.device`; the operands it now holds are CPU.
        try:
            linear.device = "cpu"
        except (AttributeError, TypeError):
            pass
    return linear


def pruned_output(case: dict, tensors: dict, ns, config: dict):
    torch = _torch()
    select = case["select"]
    if case["kind"] == KIND_INTERPOLATION:
        table_tensor = tensor_at(tensors, select, "adaln_t_table")
        if table_tensor["dtype"] != "F32" or len(table_tensor["shape"]) != 2:
            raise Refusal("selected_dtype", f"{case['id']}: pruned table must be F32 [grid, rank]")
        table = f32_tensor(table_tensor, "adaln_t_table").reshape(table_tensor["shape"][0], table_tensor["shape"][1])
        times = [float(v) for v in obj(case.get("input"), f"{case['id']}.input").get("timesteps") or []]
        if not times:
            raise Refusal("input_shape", f"{case['id']}: timesteps are required")
        return ns["_interpolated_table"](table, torch.tensor(times, dtype=torch.float32))
    prefix = select.get("prefix")
    weight_name = select.get("weight") or (f"{prefix}.weight" if prefix else None)
    bias_name = select.get("bias_tensor") or (f"{prefix}.bias" if prefix else None)
    if not weight_name or not bias_name:
        raise Refusal("selection_missing", f"{case['id']}: reduced projection needs weight and bias")
    if prefix is None:
        prefix = weight_name[: -len(".weight")] if weight_name.endswith(".weight") else weight_name
    actual_weight = f32_tensor(tensor_at(tensors, select, weight_name), weight_name)
    actual_bias = f32_tensor(tensor_at(tensors, select, bias_name), bias_name)
    if actual_weight.dim() != 2 or actual_bias.dim() != 1:
        raise Refusal("selected_dtype", f"{case['id']}: reduced operands must be [out, rank] and [out]")
    rank = int(actual_weight.shape[1])
    linear = projection_linear(ns, config, prefix, actual_weight, actual_bias, rank)
    inp = obj(case.get("input"), f"{case['id']}.input")
    coords = torch.tensor([float(v) for v in inp["values"]], dtype=torch.float32).reshape(inp["shape"][0], inp["shape"][1])
    return linear(coords)


def _torch():
    import torch

    return torch


def _nn():
    import torch.nn

    return torch.nn


def _F():
    import torch.nn.functional

    return torch.nn.functional


# --------------------------------------------------------------------------------------
# Selected-set enumeration (must equal the native resolver's raw set per backend)
# --------------------------------------------------------------------------------------


def selected_raw_names(case: dict, backend: str, config: dict) -> list[str]:
    select = case["select"]
    kind = case["kind"]
    names: list[str] = []
    if kind in (KIND_W4A8, KIND_CONVROT):
        suffixes = W4A8_SUFFIXES if kind == KIND_W4A8 else CONVROT_SUFFIXES
        names.extend(raw_for(select, f"{select['base']}.{suffix}") for suffix in suffixes)
        if select.get("bias"):
            names.append(raw_for(select, f"{select['base']}.bias"))
    elif kind == KIND_MIXED:
        for base, suffixes, bias in (
            (select["base"], W4A8_SUFFIXES, select.get("w4a8_bias")),
            (select["convrot_base"], CONVROT_SUFFIXES, select.get("convrot_bias")),
        ):
            names.extend(raw_for(select, f"{base}.{suffix}") for suffix in suffixes)
            if bias:
                names.append(raw_for(select, f"{base}.bias"))
    elif backend == "candle" or select.get("topology"):
        # Candle assembles and authenticates the complete reduced basis even when the numeric
        # operation is interpolation only. This must precede the MLX table-only selection.
        names.append(raw_for(select, "adaln_t_table"))
        for layer in range(int(config["num_layers"])):
            for suffix in ("weight", "bias"):
                names.append(raw_for(select, f"transformer_blocks.{layer}.adaln_proj.linear.{suffix}"))
        for suffix in ("weight", "bias"):
            names.append(raw_for(select, f"norm_out.linear.{suffix}"))
    elif kind == KIND_INTERPOLATION:
        names.append(raw_for(select, select.get("table", "adaln_t_table")))
    else:
        prefix = select.get("prefix")
        weight_name = select.get("weight") or (f"{prefix}.weight" if prefix else None)
        bias_name = select.get("bias_tensor") or (f"{prefix}.bias" if prefix else None)
        if not weight_name or not bias_name:
            raise Refusal("selection_missing", f"{case['id']}: reduced projection needs weight and bias")
        names.extend([raw_for(select, weight_name), raw_for(select, bias_name)])
    if len(names) > MAX_SELECTED_TENSORS:
        raise Refusal("limits_exceeded", f"{case['id']} selects {len(names)} tensors")
    return names


def selected_entries(names: list[str], tensors: dict) -> list[dict]:
    entries = []
    for name in names:
        if name not in tensors:
            raise Refusal("selection_missing", f"selected tensors have no {name!r}")
        tensor = tensors[name]
        entries.append(
            {
                "name": name,
                "sha256": sha256_hex(tensor["bytes"]),
                "dtype": tensor["dtype"],
                "shape": tensor["shape"],
                "bytes": len(tensor["bytes"]),
            }
        )
    return entries


# --------------------------------------------------------------------------------------
# Main
# --------------------------------------------------------------------------------------


def run(args: argparse.Namespace) -> int:
    receipt: dict[str, Any] = {
        "schema": RECEIPT_SCHEMA,
        "schema_version": SCHEMA_VERSION,
        "status": "refused",
        "code": None,
        "detail": None,
        "spec_path": str(args.spec),
        "selected_tensors_path": str(args.selected_tensors),
        "reference_dir": str(args.reference_dir) if args.reference_dir else None,
        "producer_provenance": None,
        "device": "cpu",
        "network_used": False,
        "gpu_used": False,
        "precision": {
            "packed_activation_dtype": "BF16",
            "pruned_activation_dtype": "F32",
            "arithmetic_authority": FROZEN_W4A8,
            "format_reference_authority": FROZEN_CONVROT,
            "pinned_functions": [
                "decode_weight_replay",
                "quantize_int8_rowwise_convrot_replay",
                "int_accumulate_replay",
                "epilogue_replay",
                "pinned_definitions/_interpolated_table/AdalnProj/FinalLayer",
            ],
        },
        "cases": [],
    }
    try:
        spec_bytes = read_bounded(args.spec, MAX_SPEC_BYTES, "spec")
        receipt["spec_sha256"] = sha256_hex(spec_bytes)
        spec = parse_spec(args.spec)
        artifact_path, artifact_header = validate_artifact_chain(spec)
        staging_summary = validate_staging_chain(spec, artifact_path)
        receipt["artifact"] = {
            "path": str(artifact_path),
            "sha256": spec["artifact"]["sha256"],
            "bytes": spec["artifact"]["bytes"],
            "tensor_count": len(artifact_header),
            "whole_file_digest_recomputed": False,
        }
        receipt["staging"] = staging_summary
        selected_bytes = read_bounded(args.selected_tensors, MAX_SELECTED_BYTES, "selected tensors")
        receipt["selected_tensors_sha256"] = sha256_hex(selected_bytes)
        tensors = read_selected_safetensors(args.selected_tensors)
        receipt["producer_provenance"] = frozen_provenance()
        config = spec.get("dit")
        w4 = load_frozen(FROZEN_W4A8)
        pruned = None
        pruned_ns = None
        oracle_cases = []
        for case in spec["cases"]:
            backend = spec.get("backend", "both")
            if case["kind"] in (KIND_INTERPOLATION, KIND_REDUCED) and not isinstance(config, dict):
                raise Refusal("spec_malformed", "pruned cases require spec.dit config")
            names = selected_raw_names(case, backend, config or {})
            entries = selected_entries(names, tensors)
            validate_selection_binding(entries, artifact_header, case["id"])
            kind = case["kind"]
            if kind == KIND_W4A8:
                rows = w4a8_rows(case, tensors, w4)
            elif kind == KIND_CONVROT:
                rows = convrot_rows(case, tensors, w4)
            elif kind == KIND_MIXED:
                rows = mixed_rows(case, tensors, w4)
            else:
                if args.reference_dir is None:
                    raise Refusal("reference_dir_missing", "pruned cases require --reference-dir")
                if pruned is None:
                    pruned = load_frozen(FROZEN_PRUNED)
                    pruned_ns = pruned_namespace(args.reference_dir, pruned)
                rows = pruned_output(case, tensors, pruned_ns, config).detach().float().tolist()
            if not rows or any(len(row) != len(rows[0]) for row in rows):
                raise Refusal("shape_mismatch", f"{case['id']}: ragged or empty output")
            values = flat(rows)
            if not all(math.isfinite(value) for value in values):
                raise Refusal("non_finite", f"{case['id']}: oracle output is not finite")
            binding = input_binding(case)
            oracle_cases.append(
                {
                    "id": case["id"],
                    "selected": entries,
                    "input": binding,
                    "tolerance": {"max_abs": KIND_TOLERANCE[kind]},
                    "output": {"shape": [len(rows), len(rows[0])], "values": values},
                }
            )
            receipt["cases"].append(
                {
                    "id": case["id"],
                    "status": "ok",
                    "code": None,
                    "selected": entries,
                    "input": binding,
                    "tolerance_max_abs": KIND_TOLERANCE[kind],
                    "output_shape": [len(rows), len(rows[0])],
                    "output_sha256": sha256_hex(json.dumps(values, separators=(",", ":")).encode()),
                }
            )
        oracle = {
            "schema": ORACLE_SCHEMA,
            "schema_version": SCHEMA_VERSION,
            "producer": "h3-actual-checkpoint-oracle-bridge/1",
            "producer_version": 1,
            "artifact_sha256": spec["artifact"]["sha256"],
            "producer_provenance": receipt["producer_provenance"],
            "device": "cpu",
            "activation_dtype": {"packed": "BF16", "pruned": "F32"},
            "cases": oracle_cases,
        }
        args.output.write_text(json.dumps(oracle, separators=(",", ":")) + "\n", encoding="utf-8")
        receipt["status"] = "ok"
        args.receipt.parent.mkdir(parents=True, exist_ok=True)
        args.receipt.write_text(json.dumps(receipt, separators=(",", ":")) + "\n", encoding="utf-8")
        return 0
    except Refusal as error:
        receipt["code"] = error.code
        receipt["detail"] = error.detail
        args.receipt.parent.mkdir(parents=True, exist_ok=True)
        args.receipt.write_text(json.dumps(receipt, separators=(",", ":")) + "\n", encoding="utf-8")
        print(f"refused: {error.code}: {error.detail}", file=sys.stderr)
        return 2
    except Exception as error:  # noqa: BLE001 - every failure must produce a receipt
        receipt["code"] = "producer_error"
        receipt["detail"] = f"{type(error).__name__}: {error}"
        args.receipt.parent.mkdir(parents=True, exist_ok=True)
        args.receipt.write_text(json.dumps(receipt, separators=(",", ":")) + "\n", encoding="utf-8")
        print(f"refused: {receipt['code']}: {receipt['detail']}", file=sys.stderr)
        return 2


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--spec", type=Path, required=True)
    parser.add_argument("--selected-tensors", type=Path, required=True)
    parser.add_argument("--reference-dir", type=Path, default=None)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--receipt", type=Path, required=True)
    return run(parser.parse_args(argv))


if __name__ == "__main__":
    raise SystemExit(main())
