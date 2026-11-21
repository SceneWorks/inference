"""LTX 2.5 joint video/audio pass-scale witness with a pinned one-block fixture."""

import json
import math
from pathlib import Path
import re
import struct

from scripts.ci.flux1_pair import fixture
from scripts.ci.wan_adapter_tier1_fixture import fetch_pinned_header, identify as identify_files, unique_object

PROFILE = "ltx25-av-pass-scales"
REPOSITORY = "SceneWorks/ltx-2.5-mlx"
REVISION = "081658ce6886cacba20817ce0359bbefef706ff2"
CHECKPOINT = "dev/bf16/transformer.safetensors"
HEADER_SHA256 = "4167e35028851c220d7f11786ac0e4adeaa14b2aa0000828fefd2437f83183d6"
# Exact generated header after the pinned topology is trimmed to block 0, before its BF16 payload.
CANDLE_HEADER_SHA256 = "c9e3440c0fa10327332acc398319600f98226be0a750050b93a4acfdd815873b"
CANDLE_HEADER_PLACEHOLDER = "__SGA_LTX_CONSTRUCTOR_HEADER_SHA256__"
CANDLE_TENSOR_COUNT = 143
WEIGHT_BYTES = 1_628_279_680
PACKAGES = {"mlx": "mlx-gen-ltx", "candle-cuda": "candle-gen-ltx"}
PROBES = {"mlx": "mlx", "candle-cuda": "candle"}
DEVICES = {"mlx": "mlx-metal", "candle-cuda": "candle-cuda"}
SHAPES = {"video": [1, 2, 128], "audio": [1, 3, 128]}
TARGETS = {"patchify_proj": [4096, 128], "audio_patchify_proj": [2048, 128],
           "transformer_blocks.0.attn1.to_q": [4096, 4096],
           "transformer_blocks.0.audio_attn1.to_q": [2048, 2048],
           "proj_out": [128, 4096], "audio_proj_out": [128, 2048]}
FIXTURE_FILES = {"transformer.safetensors", "inputs.safetensors", "adapter.safetensors", "config.json"}
PLACEHOLDER = 'unreachable!("install the source-bound pass bridge")'

MLX_ARGS = "v, vt, vc, None, vp, a, at, ac, None, ap, None, Some(epoch)"
CANDLE_ARGS = "v, a, vt, 0.5, vc, ac, vp, ap, None, rope"
BRIDGES = {
    ("mlx", "base"): f"dit.set_lora_pass(pass);\n    dit.forward({MLX_ARGS})",
    ("mlx", "head"): f"dit.forward_controlled_with_scales({MLX_ARGS}, AvPerturbation::NONE, "
                     "Some(&mlx_gen::gen_core::phase::ScaleContext::new().with_pass(pass)))",
    ("candle-cuda", "base"): f"dit.set_adapter_pass(pass);\n    dit.forward_conditioned_prepared({CANDLE_ARGS})",
    ("candle-cuda", "head"): f"dit.forward_conditioned_prepared({CANDLE_ARGS}, "
                            "&candle_gen::gen_core::phase::ScaleContext::new().with_pass(pass))",
}


def package(backend):
    return PACKAGES[backend]


def contract(backend):
    geometry = {"depth": 1, "video_width": 4096, "audio_width": 2048, "shapes": SHAPES,
                "retained_bf16_bytes": WEIGHT_BYTES}
    if backend == "candle-cuda":
        geometry.update(constructor_header_sha256=CANDLE_HEADER_SHA256, constructor_tensor_count=CANDLE_TENSOR_COUNT)
    return {
        "compared": "LTX 2.5 joint AV component, one DiT block, fixed sigma and geometry; timing and peaks are diagnostic",
        "source_snapshot": {"repository": REPOSITORY, "revision": REVISION, "file": CHECKPOINT,
                            "header_sha256": HEADER_SHA256},
        "geometry": geometry,
        "settings": {"fixture_seed": fixture.SEED, "pass_scales": [0.0, 0.75], "timeline": [0, 1, 0],
                     "sigma": 0.5, "rank": 4, "alpha": 4, "targets": TARGETS,
                     "device": {"kind": DEVICES[backend], "index": 0}, "weights": "dense BF16",
                     "context": "same text, latent and prepared position inputs across all passes",
                     "cuda_pool_cap_bytes": 10 * 2**30 if backend == "candle-cuda" else None},
        "tolerance": {"class": "AP13(a)", "output": "bit-exact finite F32 readback in both streams",
                      "provenance": "SGA §7.3; per-pass argument delivery preserves operation order"},
    }


def expose(data, kind, symbol):
    old = f"pub(crate) {kind} {symbol}".encode()
    if data.count(old) != 1:
        raise ValueError(f"expected one source-owned LTX proof seam for {symbol}")
    return data.replace(old, f"pub {kind} {symbol}".encode())


def overlay(read, arm, backend, binary):
    family = "mlx-gen" if backend == "mlx" else "candle-gen"
    root = f"crates/media/{family}/{package(backend)}/src"
    probe = Path(__file__).with_name(f"ltx_adapter_tier1_{PROBES[backend]}.rs.in").read_text(encoding="utf-8")
    if probe.count(PLACEHOLDER) != 1:
        raise ValueError("LTX probe's one source-owned forward bridge is missing")
    # Import the perturbation only where the old public method cannot accept it.
    if backend == "mlx" and arm == "head":
        probe = "use mlx_gen_ltx::transformer::AvPerturbation;\n" + probe
    if backend == "candle-cuda":
        if probe.count(CANDLE_HEADER_PLACEHOLDER) != 1:
            raise ValueError("missing unique LTX Candle constructor header pin")
        probe = probe.replace(CANDLE_HEADER_PLACEHOLDER, CANDLE_HEADER_SHA256)
    files = {f"{root}/bin/{binary}.rs": probe.replace(PLACEHOLDER, BRIDGES[backend, arm]).encode()}
    if backend == "mlx" and arm == "head":
        path = f"{root}/transformer/av_dit.rs"
        files[path] = expose(read(path), "fn", "forward_controlled_with_scales(")
    if backend == "candle-cuda":
        path = f"{root}/transformer.rs"
        source = read(path)
        signature = b"pub(crate) fn forward_conditioned_prepared("
        if source.count(signature) != 1:
            raise ValueError("expected one source-owned Candle prepared AV forward")
        params = source.split(signature, 1)[1].split(b") -> Result", 1)[0]
        if (b"scales: &ScaleContext<'_>" in params) != (arm == "head"):
            raise ValueError("Candle AV forward does not match the requested baseline or explicit scale arm")
        source = expose(source, "struct", "PreparedAvRope {")
        source = expose(source, "fn", "prepare_rope(")
        source = expose(source, "fn", "forward_conditioned_prepared(")
        if arm == "base":
            source = expose(source, "fn", "set_adapter_pass(")
        files[path] = source
        path = f"{root}/tier.rs"
        source = expose(read(path), "fn", "ltx25_transformer_vb(")
        private_builder = b"\nfn rename_vb("
        if source.count(private_builder) != 1:
            raise ValueError("expected one source-owned LTX converted-tier key bridge")
        files[path] = source.replace(private_builder, b"\npub fn rename_vb(")
    return files


def parsed_header(raw):
    data = json.loads(raw, object_pairs_hook=unique_object)
    meta = data.pop("__metadata__")
    if (meta.get("model_version") != "2.5.0" or meta.get("sceneworks_tier") != "bf16"
            or meta.get("variant") != "dev"):
        raise ValueError("LTX geometry source differs from the pinned 2.5 Dev BF16 profile")
    transformer = json.loads(meta["config"], object_pairs_hook=unique_object)["transformer"]
    if transformer.get("num_layers") != 48:
        raise ValueError("LTX source layer count is not the shipping topology")
    transformer["num_layers"] = 1
    geometry = {}
    for name, row in data.items():
        if re.match(r"transformer_blocks\.[1-9]\d*\.", name):
            continue
        dims = row.get("shape")
        if (row.get("dtype") != "BF16" or not isinstance(dims, list) or not 1 <= len(dims) <= 4
                or any(type(n) is not int or not 1 <= n <= 65536 for n in dims)):
            raise ValueError("invalid LTX BF16 tensor or geometry")
        geometry[name] = dims
    if (len(geometry) != 143 or sum(math.prod(shape) * 2 for shape in geometry.values()) != WEIGHT_BYTES
            or any(geometry.get(name + ".weight") != shape for name, shape in TARGETS.items())):
        raise ValueError("LTX truncated geometry or adapter dimensions did not match the frozen profile")
    return dict(sorted(geometry.items())), transformer


def write_adapter(path):
    factors = {}
    for name, (out_dim, in_dim) in TARGETS.items():
        factors[name + ".lora_A.weight"] = [4, in_dim]
        factors[name + ".lora_B.weight"] = [out_dim, 4]
    header = {"__metadata__": {"lora_rank": "4", "lora_alpha": "4"}}
    offset = 0
    for name, shape in factors.items():
        end = offset + 2 * math.prod(shape)
        header[name] = {"dtype": "BF16", "shape": shape, "data_offsets": [offset, end]}
        offset = end
    raw = json.dumps(header, separators=(",", ":")).encode()
    raw += b" " * (-len(raw) % 8)
    with path.open("xb") as out:
        out.write(struct.pack("<Q", len(raw)))
        out.write(raw)
        for name, shape in factors.items():
            for block in fixture.bf16_blocks(name, shape):
                out.write(block)


def materialize(root):
    raw = fetch_pinned_header(REPOSITORY, REVISION, CHECKPOINT, HEADER_SHA256)
    geometry, config = parsed_header(raw)
    root.mkdir()
    fixture.write_component(root / "transformer.safetensors", geometry)
    fixture.write_component(root / "inputs.safetensors", {**SHAPES, "video_text": [1, 3, 4096],
                                                            "audio_text": [1, 3, 2048]})
    write_adapter(root / "adapter.safetensors")
    with (root / "config.json").open("x", encoding="utf-8") as out:
        json.dump(config, out)
    return identify(root)


def identify(root):
    return identify_files(root, FIXTURE_FILES)


def observe(stdout, out, backend):
    lines = stdout.splitlines()
    if backend == "candle-cuda" and lines.count(
        f"SGA_LTX_CONSTRUCTOR_OK {CANDLE_TENSOR_COUNT} {CANDLE_HEADER_SHA256}"
    ) != 1:
        raise ValueError("missing unique LTX Candle constructor metadata check")
    if lines.count("SGA_ADAPTER_TIER1_BODY " + PROFILE) != 1:
        raise ValueError("missing unique completed LTX AV fixture body")
    result = json.loads(out.read_text(encoding="utf-8"), object_pairs_hook=unique_object)
    expected = {"device": {"kind": DEVICES[backend], "index": 0}, "applied": len(TARGETS),
                "shapes": SHAPES, "passes": [0, 1, 0]}
    if (not isinstance(result, dict) or set(result) != expected.keys() | {"velocity_bits"}
            or any(result[key] != value for key, value in expected.items())):
        raise ValueError("LTX device, adapter receipt, shape or pass order differed from the contract")
    velocities = result["velocity_bits"]
    if not isinstance(velocities, dict) or set(velocities) != {"bare", "base", "refine", "reused"}:
        raise ValueError("missing LTX bare, base, refinement or reused pass velocity")
    for stage in velocities.values():
        if not isinstance(stage, dict) or set(stage) != SHAPES.keys():
            raise ValueError("missing paired LTX velocity streams")
        for stream, bits in stage.items():
            if (not isinstance(bits, list) or len(bits) != math.prod(SHAPES[stream]) or not all(
                    type(bit) is int and 0 <= bit < 2**32 and bit & 0x7f800000 != 0x7f800000 for bit in bits)):
                raise ValueError("invalid or nonfinite LTX velocity")
    if velocities["bare"] != velocities["base"] or velocities["base"] != velocities["reused"]:
        raise ValueError("LTX base/reused context differs from the adapter-free path")
    if any(velocities["refine"][name] == velocities["base"][name] for name in SHAPES):
        raise ValueError("the later LTX adapter pass did not change both stream velocities")
    return result
