"""SCAIL-2 denoiser comparison at the post-encoder boundary."""
import json
from pathlib import Path
import struct

from scripts.ci.flux1_pair.fixture import bf16_blocks, sha
from scripts.ci.wan_pair.fixture import SEED, _write_tensors

HERE = Path(__file__).resolve().parent
VARIANT = "scail2-bf16"
CASES = {f"{lane}-scail2-{route}": VARIANT for lane in ("mlx", "cuda")
         for route in ("animation-unipc", "replacement-dpmpp")}


def _header():
    return json.loads((HERE / "headers.json").read_text())[VARIANT]


def _f32(key, shape):
    # Promote the deterministic BF16 palette to exactly representable F32 input data.
    for block in bf16_blocks("scail2/input/" + key, shape):
        out = bytearray(len(block) * 2)
        out[2::4], out[3::4] = block[::2], block[1::2]
        yield out


def _mask(t, h, w, start, ref=False):
    # Seven exclusive colors in each of four temporally packed channel groups.
    # Each source 8x8 color tile is constant, so its area average is exact.
    vals = []
    for channel in range(28):
        group, color = divmod(channel, 7)
        for cell in range(t):
            frame = start if cell == 0 else start + (cell - 1) * 4 + group + 1
            for y in range(h):
                for x in range(w):
                    vals.append(float(color == (x + y * w + frame) % 7 and (not ref or cell == 0)))
    return [struct.pack(f"<{len(vals)}f", *vals)]


def _inputs():
    shapes = {"positive": [512, 4096], "negative": [512, 4096], "clip": [1, 257, 1280],
              "reference": [16, 1, 4, 8], "extra": [16, 2, 4, 8],
              "pose0": [16, 2, 2, 4], "pose1": [16, 2, 2, 4]}
    inputs = {key: ("F32", shape, _f32(key, shape)) for key, shape in shapes.items()}
    inputs["ref_masks"] = ("F32", [28, 3, 4, 8], _mask(3, 4, 8, 0, ref=True))
    # Two independent still-image masks, each with its first frame repeated four times.
    extra = [b"".join(_mask(1, 4, 8, frame)) for frame in (1, 2)]
    inputs["extra_masks"] = ("F32", [28, 2, 4, 8], [
        b"".join(mask[channel*128:(channel+1)*128] for channel in range(28) for mask in extra)])
    for segment in range(2):
        inputs[f"driving{segment}"] = ("F32", [28, 2, 2, 4], _mask(2, 2, 4, segment * 4))
    history = [float(t == 0) for _ in range(4) for t in range(2) for _ in range(32)]
    inputs["history_mask"] = ("F32", [4, 2, 4, 8], [struct.pack("<256f", *history)])
    return inputs


def make_fixture(root, case):
    CASES[case]
    geometry = _header()["geometry"]["dit.safetensors"]
    weights = {k: (m["dtype"], m["shape"], bf16_blocks("scail2/dit/" + k, m["shape"]))
               for k, m in sorted(geometry.items())}
    _write_tensors(Path(root) / "dit.safetensors", weights)
    _write_tensors(Path(root) / "inputs.safetensors", _inputs())


def fixture_identity(root, case):
    header = _header()
    files = sorted([*header["geometry"], "inputs.safetensors"])
    if sorted(p.relative_to(root).as_posix() for p in Path(root).rglob("*.safetensors")) != files:
        raise ValueError("unexpected SCAIL-2 fixture file set")
    return {"fixture_schema": 1, "variant": CASES[case], "depth": 1, "seed": SEED,
            "source": header["source"], "files": {f: sha(Path(root) / f) for f in files}}


def case_spec(case):
    CASES[case]
    mlx = case.startswith("mlx-")
    repl = "replacement" in case
    count = 2 if repl else 1
    n = 2 if repl else 0
    solver = ("mlx_gen_wan::scheduler::" + ("FlowDpmpp2m" if repl else "FlowUniPC")
              if mlx else "FlowScheduler::UniPC")
    settings = {"seed": SEED, "steps": 4, "shift": 5.0, "guidance": 5.0,
                "requested_sampler": "dpm++" if repl else "unipc",
                "effective_sampler": solver,
                "replace": repl, "extra_references": n, "width": 64, "height": 32,
                "frames": 9 if repl else 5, "segment_len": 5, "segment_overlap": 1,
                "weight_dtype": "BF16", "compute_dtype": "F32", "depth": 1, "width_channels": 5120,
                "heads": 40, "patch": [1, 2, 2], "clip_shape": [1, 257, 1280], "text_shape": [512, 4096]}
    def f32(x):
        return struct.unpack("<f", struct.pack("<f", x))[0]
    timesteps = []
    for i in range(4):
        base = .999 + (-.999/4)*i if mlx else 1.0 + (.001-1.0)*i/4
        sigma = 5.0*base/(1.0+4.0*base)
        if mlx:
            timestep = float(int(f32(f32(sigma)*1000.0)))
            timesteps.append(struct.unpack("<I", struct.pack("<f", timestep))[0])
        else:
            timestep = (sigma - (1e-6 if i == 0 else 0.0))*1000
            timesteps.append(struct.unpack("<Q", struct.pack("<d", timestep))[0])
    trace = []
    chunks = ([([2, 2, 4], [0, 120, 0], False)] if repl else []) + [
        ([1, 2, 4], [n, 120 if repl else 0, 0], False),
        ([2, 2, 4], [n+1, 0, 0], False), ([2, 2, 4], [n+1, 0, 120], True)]
    for seg in range(count):
        trace.append({"event": "window", "start": 4*seg, "end": 4*seg+5, "seed": SEED+seg, "scheduler": solver})
        for i in range(4):
            for arm in ("positive", "negative"):
                trace.append({"event": "forward", "context": arm, "history": bool(seg), "t_bits": timesteps[i]})
                if seg == 0 and i == 0 and arm == "positive":
                    trace.append({"event": "packing", "tokens":
                                  (["additional_ref"] if repl else []) + ["ref_video", "pose"],
                                  "ref_video": ["reference", "video"],
                                  "rope": [{"grid": grid, "shift": shift, "pool": pool}
                                           for grid, shift, pool in chunks]})
            trace.append({"event": "step", "current": 4*seg+i+1, "total": 4*count})
    return {"settings": settings, "trace": trace, "outer_steps": 4*count,
            "tensor_shape": [count, 16, 2, 4, 8]}


def verify_source_headers(hub):
    """Verify the pinned source rows using the safetensors header alone."""
    import hashlib
    import re
    header = _header()
    source = header["source"]
    snapshot = Path(hub) / ("models--" + source["repo"].replace("/", "--")) / "snapshots" / source["snapshot"]
    name, expected = next(iter(source["header_sha256"].items()))
    with (snapshot / name).open("rb") as f:
        size, = struct.unpack("<Q", f.read(8))
        if not 0 < size < 16 * 1024**2:
            raise ValueError("invalid SCAIL-2 source header size")
        raw = f.read(size)
    if len(raw) != size or hashlib.sha256(raw).hexdigest() != expected:
        raise ValueError("wrong pinned SCAIL-2 source header")
    geometry = {key: {field: value[field] for field in ("dtype", "shape")}
                for key, value in json.loads(raw).items()
                if key != "__metadata__" and not re.match(r"blocks\.[1-9]\d*\.", key)}
    if header["geometry"] != {"dit.safetensors": geometry}:
        raise ValueError("truncated SCAIL-2 geometry differs from source")
