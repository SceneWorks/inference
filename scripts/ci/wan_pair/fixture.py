"""Wan Tier 1 denoiser inputs from pinned production tensor headers. No learned execution."""
import hashlib
import json
import math
from pathlib import Path
import random
import re
import struct

from scripts.ci.flux1_pair.fixture import bf16_blocks, sha

HERE = Path(__file__).resolve().parent
SEED = 104729
CASES = {
    "mlx-a14b-t2v-resident-heun": "mlx-a14b-t2v",
    "mlx-a14b-i2v-resident-heun": "mlx-a14b-i2v",
    "mlx-a14b-i2v-swapped-heun": "mlx-a14b-i2v",
    "mlx-ti2v5b-t2v-heun": "mlx-ti2v5b",
    "cuda-ti2v5b-t2v-heun": "cuda-ti2v5b",
}


def case_spec(case):
    variant = CASES[case]
    mlx = variant.startswith("mlx")
    dense = "ti2v5b" in variant
    shape = ([48] if dense else [16]) + [2, 4, 6]
    if not mlx:
        shape = [1] + shape
    shift = 12 if variant == "mlx-a14b-t2v" else 5
    # MLX Wan scheduler.rs::compute_sigmas. Candle exposes progress, not evaluated sigma.
    sigma_max = 1 - 1 / 1000
    nodes = [sigma_max + (-sigma_max / 4) * i for i in range(4)]
    nodes = [shift * x / (1 + (shift - 1) * x) for x in nodes]
    bits = [struct.unpack("<I", struct.pack("<f", x))[0] for x in nodes]
    swapped = "swapped" in case
    trace = [{"event": "load_expert", "high": True}] if swapped else []
    high = True
    for evaluation, i in enumerate([0, 1, 1, 2, 2, 3, 3]):
        trace.append({"event": "step", "current": i + 1, "total": 4})
        if mlx:
            trace.append({"event": "evaluate", "evaluation": evaluation, "sigma_bits": bits[i]})
        if swapped and high and int(nodes[i] * 1000) < 900:
            trace.append({"event": "load_expert", "high": False})
            high = False
    stride = 16 if dense else 8
    guidance = [5.0] if dense else ([3.0, 4.0] if shift == 12 else [3.5, 3.5])
    settings = {"sampler": "heun", "seed": SEED, "steps": 4, "shift": shift,
                "guidance": guidance, "denoiser_dtype": "BF16", "latent_dtype": "F32",
                "latent_shape": shape, "patch": [1, 2, 2], "model_width": 3072 if dense else 5120,
                "heads": 24 if dense else 40, "depth": 1,
                "reference_channels": 20 if variant == "mlx-a14b-i2v" else 0,
                "boundary_timestep": None if dense else (875 if shift == 12 else 900),
                "render_width": 6 * stride, "render_height": 4 * stride, "frames": 5,
                "text_shape": [512, 4096], "residency": "swapped" if swapped else "resident"}
    return {"tensor_shape": shape, "outer_steps": 4, "trace": trace, "settings": settings}


def _header(variant):
    return json.loads((HERE / "headers.json").read_text(encoding="utf-8"))[variant]


def verify_source_headers(hub):
    """Re-derive every row from local pinned snapshots without reading weight payloads."""
    for variant in set(CASES.values()):
        entry = _header(variant)
        source = entry["source"]
        base = Path(hub) / ("models--" + source["repo"].replace("/", "--")) / "snapshots" / source["snapshot"]
        observed = {}
        for relative, expected in source["header_sha256"].items():
            with (base / relative).open("rb") as handle:
                size, = struct.unpack("<Q", handle.read(8))
                if not 0 < size < 16 * 1024**2:
                    raise ValueError(f"invalid header length for {relative}")
                header = handle.read(size)
            if hashlib.sha256(header).hexdigest() != expected:
                raise ValueError(f"source header changed for {relative}")
            filename = (relative.removeprefix("bf16/") if variant.startswith("mlx")
                        else "transformer/model.safetensors")
            retained = observed.setdefault(filename, {})
            for key, meta in json.loads(header).items():
                if key == "__metadata__" or re.search(r"(?:^|\.)blocks\.[1-9]\d*\.", key):
                    continue
                if key in retained:
                    raise ValueError(f"duplicate source tensor {key}")
                retained[key] = {field: meta[field] for field in ("dtype", "shape")}
        if observed != entry["geometry"]:
            raise ValueError(f"frozen rows differ from source for {variant}")


def _weight_blocks(key, meta):
    for block in bf16_blocks(key, meta["shape"]):
        if meta["dtype"] == "BF16":
            yield block
        elif meta["dtype"] == "F32":
            # The Diffusers checkpoint stores F32. Candle's component_vb maps it to
            # DIT_DTYPE (BF16) at load. Preserve that conversion in both proof arms.
            expanded = bytearray(2 * len(block))
            expanded[2::4], expanded[3::4] = block[0::2], block[1::2]
            yield expanded
        else:
            raise ValueError(f"unsupported source dtype {meta['dtype']}")


def _write_tensors(path, tensors):
    """Write one bounded-memory safetensors stream. Keys, dimensions and order are fixed."""
    header = {}
    offset = 0
    for name, (dtype, shape, _) in tensors.items():
        size = math.prod(shape) * {"BF16": 2, "F32": 4}[dtype]
        header[name] = {"dtype": dtype, "shape": shape, "data_offsets": [offset, offset + size]}
        offset += size
    raw = json.dumps(header, separators=(",", ":")).encode()
    raw += b" " * (-len(raw) % 8)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("xb") as target:
        target.write(struct.pack("<Q", len(raw)))
        target.write(raw)
        for key, (_, _, blocks) in tensors.items():
            start = target.tell()
            for block in blocks:
                target.write(block)
            left, right = header[key]["data_offsets"]
            if target.tell() - start != right - left:
                raise ValueError(f"tensor byte count differs for {key}")


def _inputs(variant):
    # The pos/neg inputs enter the per-expert text projection. Each raw context
    # has the production 512 tokens × 4096 channels with distinct seeded values.
    shape = [512, 4096]
    inputs = {k: ("BF16", shape, bf16_blocks(f"wan/raw-text/{k}", shape))
              for k in ("positive", "negative")}
    if variant == "mlx-a14b-i2v":
        # Same layout as build_i2v_y. The fixture starts after the VAE encoder
        # with the first-frame mask and a deterministic, nonzero z16 condition.
        rng = random.Random(SEED)
        mask = ([1.0] * 24 + [0.0] * 24) * 4
        planes = [rng.uniform(-1, 1) for _ in range(16 * 2 * 4 * 6)]
        values = mask + planes
        inputs["y"] = ("F32", [20, 2, 4, 6], [struct.pack(f"<{len(values)}f", *values)])
    return inputs


def make_fixture(root, case):
    root = Path(root)
    variant = CASES[case]
    header = _header(variant)
    for filename, geometry in header["geometry"].items():
        # Each expert has independent weights. Both I2V residency modes share
        # this variant string, filenames, raw context and channel-concat bytes.
        tensors = {key: (meta["dtype"], meta["shape"], _weight_blocks(f"wan/{variant}/{filename}/{key}", meta))
                   for key, meta in sorted(geometry.items())}
        _write_tensors(root / filename, tensors)
    _write_tensors(root / "inputs.safetensors", _inputs(variant))


def fixture_identity(root, case):
    variant = CASES[case]
    header = _header(variant)
    root = Path(root)
    files = sorted([*header["geometry"], "inputs.safetensors"])
    # Candle enumerates the whole transformer directory. Reject any extra weights
    # that could participate in the load without appearing in the paired receipt.
    observed = sorted(p.relative_to(root).as_posix() for p in root.rglob("*.safetensors"))
    if files != observed:
        raise ValueError("unexpected fixture tensor files")
    return {"fixture_schema": 1, "variant": variant, "seed": SEED, "depth": 1,
            "source": header["source"], "files": {f: sha(root / f) for f in files}}
