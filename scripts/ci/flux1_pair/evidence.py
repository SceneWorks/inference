"""Read the native probe's fixed outputs and evaluate AP13 class (a)."""
import hashlib
import json
import math
import re
import struct
import zlib

from scripts.ci.flux1_pair.fixture import sha

VARIANTS = ("dev", "schnell")
ARMS = ("base", "head")
SHAPE = [1, 256, 64]
SETTINGS = {"prompt": "red kite", "count": 1, "seed": 104729, "width": 256, "height": 256,
            "steps": 3, "sampler": "euler", "scheduler": "native", "negative_prompt": None,
            "conditioning": [], "adapters": [], "pid": False, "offload": "resident",
            "base_activation_precision": "BF16 load with native sampler dtype",
            "guidance": {"dev": 3.5, "schnell": None},
            "scope": "fresh process per arm and variant; fingerprint comparison only",
            "acceptance": "all three outer input tensors, terminal tensor, sigma bit patterns, step events and RGB8 bit-exact"}


def png(rgb):
    def chunk(tag, value):
        return struct.pack(">I", len(value)) + tag + value + struct.pack(">I", zlib.crc32(tag + value))
    rows = b"".join(b"\0" + rgb[i:i + 768] for i in range(0, len(rgb), 768))
    return (b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", struct.pack(">IIBBBBB", 256, 256, 8, 2, 0, 0, 0))
            + chunk(b"IDAT", zlib.compress(rows)) + chunk(b"IEND", b""))


def validate_plan(plan, variant, backend):
    phase = plan["pass"]["phase"]
    sigmas = plan["pass"]["sigmas"]
    if (plan["provider"] != "flux1_" + variant or plan["backend"] != ("mlx" if backend == "mlx" else "candle")
            or phase["first_step"] != 0 or phase["end_step"] != 3 or phase["sampler"] != "euler"
            or phase["guidance"] != (3.5 if variant == "dev" else 0)
            or phase["stream"] != {"width": 256, "height": 256, "patch_stride": 16, "tokens": 256, "token_width": 64}
            or plan["inputs"]["requested_seed"] != SETTINGS["seed"]
            or len(sigmas) != 4 or sigmas[0] != 1 or sigmas[-1] != 0
            or not all(a > b for a, b in zip(sigmas, sigmas[1:]))
            or not all(math.isfinite(s) for s in sigmas)):
        raise ValueError("preload plan disagrees with frozen route, seed, schedule or geometry")
    # Resolution-dependent dev schedule versus linear schnell, allowing native
    # f32 operation rounding ONLY in this setup check. A/B bits must be exact.
    for index, actual in enumerate(sigmas[:-1]):
        t = (3 - index) / 3
        exp_mu = math.exp(0.5) if variant == "dev" else 1.0
        ref = exp_mu / (exp_mu + 1 / t - 1)
        if abs(actual - ref) > 2e-7:
            raise ValueError("preload schedule is not the native variant's schedule")


def read_arm(output, stdout, arm, variant, backend, artifact):
    model = "flux1_" + variant
    if stdout.splitlines().count("FLUX1_PAIR_COMPLETE " + model) != 1:
        raise ValueError("missing completed provider body")
    rgb = (output / "output.rgb").read_bytes()
    if len(rgb) != 256 * 256 * 3 or min(rgb) == max(rgb):
        raise ValueError("empty/degenerate output")
    expected_device = "mlx-metal" if backend == "mlx" else "candle-cuda"
    rows = re.findall(r"^FLUX1_PAIR_STATE ([0-9a-f]{8}) (\w+) (\S+)$", stdout, re.M)
    if len(rows) != 4 or any(row[2] != expected_device for row in rows):
        raise ValueError("wrong in-scope device or evaluation count")
    sigmas = [struct.unpack(">f", bytes.fromhex(row[0]))[0] for row in rows]
    if sigmas[0] != 1 or sigmas[-1] != 0 or not all(a > b for a, b in zip(sigmas, sigmas[1:])):
        raise ValueError("invalid native schedule")
    progress = json.loads((output / "progress.json").read_text(encoding="utf-8"))
    if progress != [[1, 3], [2, 3], [3, 3]]:
        raise ValueError("missing/extra outer progress")
    states = []
    for sigma, dtype, device in rows:
        tensor = output / ("state-" + sigma + ".bin")
        if tensor.stat().st_size != 4 * 256 * 64:
            raise ValueError("wrong state geometry")
        states.append(dict(sigma_bits=sigma, dtype=dtype, device=device, sha256=sha(tensor)))
    if len({s["sha256"] for s in states}) != 4:
        raise ValueError("trajectory did not advance at every outer step")
    plan = None
    if arm == "head":
        plan = json.loads((output / "plan.json").read_text(encoding="utf-8"))
        validate_plan(plan, variant, backend)
        if [struct.pack(">f", f).hex() for f in plan["pass"]["sigmas"]] != [x[0] for x in rows]:
            raise ValueError("observed sampler sigmas differ from preload")
    name = f"{arm}-{variant}.png"
    (artifact / name).write_bytes(png(rgb))
    return {"states": states, "rgb_sha256": hashlib.sha256(rgb).hexdigest(), "steps": progress,
            "png": name, "png_sha256": sha(artifact / name), "planned_route": plan}


def compare(base, head):
    for name in ("states", "steps", "rgb_sha256"):
        if base[name] != head[name]:
            raise ValueError("AP13 class (a) mismatch in " + name)
