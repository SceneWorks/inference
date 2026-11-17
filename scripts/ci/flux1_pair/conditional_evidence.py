"""Class-(a) conditional FLUX.1 same-backend evidence; no adjustable tolerance."""
import hashlib
import json
import math
import re
import struct
import zlib

from scripts.ci.flux1_pair import conditional, evidence
from scripts.ci.flux1_pair.fixture import sha

ARMS = evidence.ARMS
SHAPE = evidence.SHAPE
SETTINGS = {
    "base": "shape-faithful random BF16 FLUX.1, production widths and one DiT block",
    "overlays": "sha256-pinned real conditional and face-analysis assets shared by both arms",
    "prompt": "red kite portrait", "count": 1, "seed": 104729,
    "width": 256, "height": 256, "steps": 3, "sampler": "euler", "scheduler": "native",
    "adapters": [], "offload": "resident", "allocator": "same physical device and process-fresh arm",
    "class": "AP13(a): exact sigma bits, all outer states, event sequence and RGB8; zero tolerance",
}


def png(rgb, side):
    def chunk(tag, value):
        return (struct.pack(">I", len(value)) + tag + value
                + struct.pack(">I", zlib.crc32(tag + value)))
    stride = side * 3
    rows = b"".join(b"\0" + rgb[i:i + stride] for i in range(0, len(rgb), stride))
    return (b"\x89PNG\r\n\x1a\n"
            + chunk(b"IHDR", struct.pack(">IIBBBBB", side, side, 8, 2, 0, 0, 0))
            + chunk(b"IDAT", zlib.compress(rows)) + chunk(b"IEND", b""))


def validate_plan(plan, case, backend):
    variant = conditional.variant(case)
    provider = ("flux1_" + variant if backend == "mlx" and case.startswith("ip-")
                else "flux1_" + variant + "_ip_adapter" if case.startswith("ip-")
                else "flux1_dev_control" if case.startswith("control-") else "pulid_flux")
    phase = plan["pass"]["phase"]
    sigmas = plan["pass"]["sigmas"]
    stream = phase["stream"]
    inputs = plan["inputs"]
    prepared = plan["preparation"]
    cfg = case in ("ip-dev-cfg", "pulid-real", "pulid-delayed")
    bits = lambda value: struct.unpack("<I", struct.pack("<f", value))[0]
    if (plan["provider"] != provider
            or plan["backend"] != ("mlx" if backend == "mlx" else "candle")
            or phase["first_step"] != 0 or phase["end_step"] != 3
            or phase["sampler"] != "euler" or phase["guidance"] != (3.5 if variant == "dev" else 0.0)
            or stream != {"width": 256, "height": 256, "patch_stride": 16,
                          "tokens": 256, "token_width": 64}
            or any(inputs[name] != expected for name, expected in
                   (("width", 256), ("height", 256), ("count", 1), ("steps", 3),
                    ("sampler", "euler"), ("requested_seed", SETTINGS["seed"]),
                    ("negative_prompt", "blurred" if cfg else None),
                    ("true_cfg_bits", bits(2.0) if cfg else None),
                    ("timestep_to_start_cfg", 2 if case == "pulid-delayed" else None),
                    ("use_pid", case == "pulid-pid")))
            or not isinstance(prepared, dict) or prepared["adapters"] != []
            or prepared["sequential_offload"] or prepared["deferred_materialization"]
            or bool(prepared["control"]) != case.startswith("control-")
            or bool(prepared["ip_adapter"]) != case.startswith("ip-")
            or bool(prepared["identity"]) != case.startswith("pulid-")
            or bool(prepared["pid"]) != (case == "pulid-pid")
            or len(sigmas) != 4 or sigmas[0] != 1 or sigmas[-1] != 0
            or not all(a > b for a, b in zip(sigmas, sigmas[1:]))
            or not all(math.isfinite(value) for value in sigmas)):
        raise ValueError("conditional pre-load route or trajectory differs from frozen case")
    for index, actual in enumerate(sigmas[:-1]):
        t = (3 - index) / 3
        mu = math.exp(0.5) if variant == "dev" else 1.0
        expected = mu / (mu + 1 / t - 1)
        if abs(actual - expected) > 2e-7:
            raise ValueError("conditional pre-load schedule is not the native variant schedule")
    conditioning = inputs["conditioning"]
    if case.startswith("control-"):
        kind = "pose" if case == "control-zero" else case.split("-", 1)[1]
        expected = {"kind": kind, "width": 256, "height": 256,
                    "scale_bits": bits(0.0 if case == "control-zero" else 0.7)}
        if not (isinstance(conditioning, dict) and isinstance(conditioning.get("Control"), dict)
                and all(conditioning["Control"].get(key) == value for key, value in expected.items())):
            raise ValueError("ControlNet plan lost its fixed source or controls")
    else:
        expected = {"strength_bits": bits(0.7 if case.startswith("ip-") else 1.0)}
        if not (isinstance(conditioning, dict) and isinstance(conditioning.get("Reference"), dict)
                and all(conditioning["Reference"].get(key) == value for key, value in expected.items())):
            raise ValueError("identity plan lost its fixed reference controls")


def read_arm(output, stdout, arm, case, backend, artifact):
    if stdout.splitlines().count("FLUX1_CONDITIONAL_COMPLETE " + case) != 1:
        raise ValueError("missing completed conditional provider body")
    rgb = (output / "output.rgb").read_bytes()
    side = 1024 if case == "pulid-pid" else 256
    if len(rgb) != side * side * 3 or min(rgb) == max(rgb):
        raise ValueError("empty or degenerate conditional output")
    expected_device = "mlx-metal" if backend == "mlx" else "candle-cuda"
    states = re.findall(r"^FLUX1_PAIR_STATE ([0-9a-f]{8}) (\w+) (\S+)$", stdout, re.M)
    if len(states) != 4 or any(row[2] != expected_device for row in states):
        raise ValueError("wrong conditional tensor device or evaluation count")
    sigmas = [struct.unpack(">f", bytes.fromhex(row[0]))[0] for row in states]
    if sigmas[0] != 1 or sigmas[-1] != 0 or not all(a > b for a, b in zip(sigmas, sigmas[1:])):
        raise ValueError("invalid conditional schedule")
    progress = json.loads((output / "progress.json").read_text(encoding="utf-8"))
    events = json.loads((output / "events.json").read_text(encoding="utf-8"))
    if progress != [[1, 3], [2, 3], [3, 3]] or not events:
        raise ValueError("conditional progress or stage events are missing")
    observed = []
    for sigma, dtype, device in states:
        tensor = output / ("state-" + sigma + ".bin")
        if tensor.stat().st_size != 4 * 256 * 64:
            raise ValueError("wrong conditional state geometry")
        observed.append(dict(sigma_bits=sigma, dtype=dtype, device=device, sha256=sha(tensor)))
    if len({row["sha256"] for row in observed}) != 4:
        raise ValueError("conditional trajectory did not advance")
    plan = None
    if arm == "head":
        plan = json.loads((output / "plan.json").read_text(encoding="utf-8"))
        validate_plan(plan, case, backend)
        if [struct.pack(">f", value).hex() for value in plan["pass"]["sigmas"]] != [row[0] for row in states]:
            raise ValueError("observed conditional sigmas differ from pre-load plan")
    name = f"{arm}-{case}.png"
    (artifact / name).write_bytes(png(rgb, side))
    return {"states": observed, "events": events, "steps": progress,
            "rgb_sha256": hashlib.sha256(rgb).hexdigest(), "png": name,
            "png_sha256": sha(artifact / name), "planned_route": plan}


def compare(base, head):
    for name in ("states", "events", "steps", "rgb_sha256"):
        if base[name] != head[name]:
            raise ValueError("AP13 class (a) conditional mismatch in " + name)
