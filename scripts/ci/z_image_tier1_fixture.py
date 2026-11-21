"""Paired Z-Image Q4 DiT witness over the pinned shipping tensor geometry."""

import hashlib
import json
import math
from pathlib import Path
import random
import re
import struct

from scripts.ci.flux1_pair import fixture
from scripts.ci.wan_adapter_tier1_fixture import (
    fetch_pinned_header,
    identify as identify_files,
    unique_object,
)

PROFILE = "z-image-q4-dit"
PACKAGE = "candle-gen-z-image"
REPOSITORY = "SceneWorks/z-image-turbo-mlx"
REVISION = "bb2bc9893b3c49ae96c813350775f791a2e8bc80"
CHECKPOINT = "q4/transformer/model.safetensors"
HEADER_SHA256 = "613ed8d63d13349a83b08e8f61f7d820101807160d3075a7b5be50aa2948ce93"
WEIGHT_BYTES = 310_078_720
PROBE = Path(__file__).with_name("z_image_tier1_candle.rs.in")
GROUP = 64
WIDTH = 3840
MAIN_PACKED = 8
LATENT = [1, 16, 1, 4, 4]
CAPTION = [1, 32, 2560]
TIMESTEPS = [0.25, 0.75]
TARGETS = {"cap_embedder.1": [WIDTH, 2560], "layers.0.attention.to_q": [WIDTH, WIDTH]}
FIXTURE_FILES = {"transformer.safetensors", "inputs.safetensors", "adapter.safetensors"}
BYTE_WIDTH = {"U32": 4, "F32": 4, "BF16": 2}
STACKS = {"layers": 30, "noise_refiner": 2, "context_refiner": 2}
BLOCK = re.compile(r"^(layers|noise_refiner|context_refiner)\.(\d+)\.")
DEVICE = {"kind": "candle-cuda", "index": 0}


def package(backend):
    if backend != "candle-cuda":
        raise ValueError("Z-Image Q4 fixture requires Candle CUDA")
    return PACKAGE


def contract(backend):
    package(backend)
    return {
        "compared": "Z-Image Q4 DiT prepared-caption velocities, paired per resident or sidecar route; timing and peaks are diagnostic",
        "source_snapshot": {"repository": REPOSITORY, "revision": REVISION,
                            "file": CHECKPOINT, "header_sha256": HEADER_SHA256},
        "geometry": {"variant": "Turbo", "block_counts": {name: 1 for name in STACKS},
                     "width": WIDTH, "head_count": 30, "caption": CAPTION, "latent": LATENT,
                     "retained_weight_bytes": WEIGHT_BYTES},
        "settings": {"fixture_seed": fixture.SEED, "group_size": GROUP, "bits": 4,
                     "activation_dtype": "BF16", "compute_dtype": "BF16", "device": DEVICE,
                     "sidecar_window": 1, "timesteps": TIMESTEPS, "valid_caption_tokens": 5,
                     "packed_sidecar_count": MAIN_PACKED, "accelerated_attention": False,
                     "adapter_rank": 4, "adapter_alpha": 4, "adapter_strength": 1.25,
                     "adapter_targets": TARGETS, "cuda_pool_cap_bytes": 10 * 2**30},
        "tolerance": {"class": "AP13(a)",
                      "output": "bit-exact finite F32 readback per matching route, adapter state and timestep",
                      "provenance": "SGA §7.3; unchanged Q4 storage, operation order and attention policy"},
        "closure": "Diffusers Q4 snapshot loader. Combined ComfyUI FP8 imports require a separate native-source baseline pair.",
    }


def replace_once(data, old, new):
    if data.count(old) != 1:
        raise ValueError(f"expected exactly one Z-Image proof anchor: {old!r}")
    return data.replace(old, new, 1)


def expose(data, declaration):
    return replace_once(data, b"pub(crate) " + declaration, b"pub " + declaration)


def overlay(read, arm, backend, binary):
    package(backend)
    modules = {"base": ("packed_dit", "new"), "head": ("dit", "new_inference")}
    module, constructor = modules[arm]
    root = f"crates/media/candle-gen/{PACKAGE}/src"
    lib = f"{root}/lib.rs"
    source = f"{root}/{module}.rs"
    declaration = f"mod {module};".encode()
    graph = read(source)
    for anchor in (b"struct PreparedConditioning {", b"fn prepare_conditioning(",
                   b"fn forward_prepared_with_memory("):
        graph = expose(graph, anchor)
    if arm == "head":
        for anchor in (b"fn new_inference(cfg: &Config, vb:", b"fn new_block_streamed_with_sidecars("):
            graph = expose(graph, anchor)
    elif (graph.count(b"pub fn new(cfg: &Config,") != 1
          or graph.count(b"pub fn new_block_streamed_with_sidecars(") != 1):
        raise ValueError("legacy packed DiT constructors did not match the source-owned routes")
    probe = PROBE.read_bytes()
    probe = replace_once(probe, b"__z_image_module__", module.encode())
    probe = replace_once(probe, b"__z_image_resident__", constructor.encode())
    probe = replace_once(probe, b"// __SGA_CUDA_RECEIPT__", cuda_receipt())
    return {
        lib: replace_once(read(lib), declaration, b"pub " + declaration),
        source: graph,
        f"{root}/bin/{binary}.rs": probe,
    }


def cuda_receipt():
    # Reuse the controller's validated Candle CUDA child receipt and physical UUID check.
    path = PROBE.with_name("ltx_adapter_tier1_candle.rs.in")
    source = path.read_bytes()
    start = b'#[cfg(feature = "cuda")]\nfn retain_cuda_pool('
    end = b"\nfn main() ->"
    if source.count(start) != 1 or source.count(end) != 1:
        raise ValueError("shared Candle proof pool receipt anchor changed")
    a, b = source.index(start), source.index(end)
    if not a < b:
        raise ValueError("shared Candle proof pool receipt has no complete implementation")
    return source[a:b]


def retained_geometry(raw):
    data = json.loads(raw, object_pairs_hook=unique_object)
    if not isinstance(data, dict) or len(data) != 1074 or data.pop("__metadata__", False) is not None:
        raise ValueError("Z-Image Q4 header differs from its shipping geometry")
    blocks = {name: set() for name in STACKS}
    retained = {}
    for name, row in data.items():
        block = BLOCK.match(name)
        if block:
            stack, layer = block.groups()
            blocks[stack].add(int(layer))
            if layer != "0":
                continue
        if not isinstance(row, dict):
            raise ValueError(f"invalid retained Z-Image tensor {name}")
        dtype = row.get("dtype")
        shape = row.get("shape")
        if (dtype not in BYTE_WIDTH or not isinstance(shape, list) or not 1 <= len(shape) <= 2
                or any(type(n) is not int or not 1 <= n <= 16384 for n in shape)):
            raise ValueError(f"invalid retained Z-Image tensor {name}")
        retained[name] = dtype, shape
    if (any(blocks[name] != set(range(depth)) for name, depth in STACKS.items())
            or len(retained) != 116
            or sum(math.prod(shape) * BYTE_WIDTH[dtype]
                   for dtype, shape in retained.values()) != WEIGHT_BYTES):
        raise ValueError("Z-Image Q4 fixture depth or footprint changed")
    for name, (out_dim, in_dim) in TARGETS.items():
        if retained.get(name + ".weight") != ("U32", [out_dim, in_dim // 8]):
            raise ValueError("Q4 adapter projection geometry changed")
    packed = []
    for name, (dtype, shape) in retained.items():
        if dtype == "U32":
            if not name.endswith(".weight") or len(shape) != 2 or shape[1] % 8:
                raise ValueError(f"invalid group-64 Q4 layout for {name}")
            projection = name.removesuffix(".weight")
            affine = ("BF16", [shape[0], shape[1] // 8])
            if any(retained.get(projection + suffix) != affine for suffix in (".scales", ".biases")):
                raise ValueError(f"missing Q4 affine companions for {projection}")
            packed.append(projection)
    if len(packed) != 29 or sum(name.startswith("layers.0.") for name in packed) != MAIN_PACKED:
        raise ValueError("unexpected number of retained Q4 projections")
    return dict(sorted(retained.items()))


def tensor_data(name, dtype, shape):
    # Packed codes are reproducible uniform nibbles. Power-of-two scales and an exact
    # -7.5× bias center each group and avoid introducing an extra repack rounding input.
    count = math.prod(shape)
    if dtype == "U32":
        seed = hashlib.sha256(f"{fixture.SEED}/{name}".encode()).digest()
        rng = random.Random(int.from_bytes(seed, "little"))
        while count:
            take = min(count, 2**16)
            yield rng.randbytes(take * 4)
            count -= take
    elif name.endswith((".scales", ".biases")):
        in_dim = shape[1] * GROUP
        scale = 2.0 ** (-math.ceil(math.log2(math.sqrt(in_dim))) - 3)
        value = -7.5 * scale if name.endswith(".biases") else scale
        raw = struct.pack("<f", value)[2:]
        while count:
            take = min(count, 2**16)
            yield raw * take
            count -= take
    elif dtype == "F32":
        # The shipping header stores these as FP32. Generate the same values as the
        # deterministic BF16 palette, expanded exactly rather than using a new PRNG.
        for block in fixture.bf16_blocks(name, shape):
            out = bytearray(len(block) * 2)
            out[2::4], out[3::4] = block[::2], block[1::2]
            yield out
    else:
        yield from fixture.bf16_blocks(name, shape)


def write_weights(path, geometry, metadata=None):
    header = {} if metadata is None else {"__metadata__": metadata}
    offset = 0
    for name, (dtype, shape) in geometry.items():
        end = offset + math.prod(shape) * BYTE_WIDTH[dtype]
        header[name] = {"dtype": dtype, "shape": shape, "data_offsets": [offset, end]}
        offset = end
    raw = json.dumps(header, separators=(",", ":")).encode()
    raw += b" " * (-len(raw) % 8)
    with path.open("xb") as out:
        out.write(struct.pack("<Q", len(raw)))
        out.write(raw)
        for name, (dtype, shape) in geometry.items():
            for block in tensor_data(name, dtype, shape):
                out.write(block)
        if out.tell() != 8 + len(raw) + offset:
            raise ValueError("fixture payload size differs from its safetensors header")


def materialize(root):
    raw = fetch_pinned_header(REPOSITORY, REVISION, CHECKPOINT, HEADER_SHA256)
    geometry = retained_geometry(raw)
    root.mkdir()
    write_weights(root / "transformer.safetensors", geometry)
    fixture.write_component(root / "inputs.safetensors", {"latent": LATENT, "caption": CAPTION})
    factors = {}
    for name, (out_dim, in_dim) in TARGETS.items():
        factors[name + ".lora_A.weight"] = "BF16", [4, in_dim]
        factors[name + ".lora_B.weight"] = "BF16", [out_dim, 4]
    write_weights(root / "adapter.safetensors", factors, {"lora_rank": "4", "lora_alpha": "4"})
    return identify(root)


def identify(root):
    return identify_files(root, FIXTURE_FILES)


def observe(stdout, output, backend):
    package(backend)
    if stdout.splitlines().count("SGA_ADAPTER_TIER1_BODY " + PROFILE) != 1:
        raise ValueError("missing unique completed Z-Image Q4 fixture body")
    row = json.loads(output.read_text(encoding="utf-8"), object_pairs_hook=unique_object)
    expected = {"device": DEVICE, "shape": LATENT, "timesteps": TIMESTEPS,
                "applied": {"resident": len(TARGETS), "sidecar": len(TARGETS)}}
    if (not isinstance(row, dict) or set(row) != expected.keys() | {"velocity_bits"}
            or any(row[key] != value for key, value in expected.items())):
        raise ValueError("Z-Image fixture device, shape, time or adapter receipt differs from its plan")
    outputs = row["velocity_bits"]
    if not isinstance(outputs, dict) or set(outputs) != {"resident", "sidecar"}:
        raise ValueError("missing matching resident or sidecar velocity route")
    for route in outputs.values():
        if not isinstance(route, dict) or set(route) != {"bare", "adapted"}:
            raise ValueError("missing bare or adapted Z-Image output")
        for stages in route.values():
            if not isinstance(stages, list) or len(stages) != len(TIMESTEPS):
                raise ValueError("missing reused Z-Image preparation output")
            for bits in stages:
                if not isinstance(bits, list) or len(bits) != math.prod(LATENT) or not all(
                        type(bit) is int and 0 <= bit < 2**32 and bit & 0x7f800000 != 0x7f800000 for bit in bits):
                    raise ValueError("invalid or nonfinite Z-Image velocity")
        if route["bare"] == route["adapted"]:
            raise ValueError("Z-Image LoRA did not change the Q4 velocity")
    return row
