"""Deterministic, depth-reduced Mochi and SVD-XT paired denoiser fixtures."""
import hashlib
import json
import math
from pathlib import Path
import random
import re
import struct

from scripts.ci.flux1_pair.fixture import SEED, sha, weight_blocks
from scripts.ci.wan_pair.fixture import _write_tensors

HERE = Path(__file__).resolve().parent
MOCHI = "SceneWorks/mochi-1-mlx", "90a87786d9b5b592c3b3c083e5fcdf130007b1de"
SVD = "stabilityai/stable-video-diffusion-img2vid-xt", "9e43909513c6714f1bc78bcb44d96e733cd242aa"
MOCHI_DIT = "bf16/transformer/model.safetensors"
UNET = "unet/diffusion_pytorch_model.safetensors"
UNET_HALF = UNET.replace(".safetensors", ".fp16.safetensors")
AUX = ("image_encoder/model.fp16.safetensors", "vae/diffusion_pytorch_model.safetensors")
CASES = {
    "mlx-mochi-masked-euler": "mochi",
    "cuda-mochi-masked-euler": "mochi",
    "mlx-svd-native-euler": "mlx-svd",
    "mlx-svd-heun": "mlx-svd",
    "cuda-svd-euler-cfg": "cuda-svd",
    "cuda-svd-euler-single": "cuda-svd",
    "cuda-svd-heun-cfg": "cuda-svd",
}


def _retained(name, mochi):
    if mochi:
        match = re.match(r"transformer_blocks\.(\d+)\.", name)
        if not match:
            return name
        if match[1] == "47":
            return name.replace("transformer_blocks.47.", "transformer_blocks.1.", 1)
        return name if match[1] == "0" else None
    # The last up-layer consumes the skip from the adjacent resolution. Preserve
    # its real weights/width, dropping the interior up layer at the same resolution.
    match = re.match(r"(down|up)_blocks\.\d+\.(resnets|attentions)\.(\d+)\.", name)
    if not match:
        return name
    if match[3] == "0":
        return name
    if match[1] == "up" and match[3] == "2":
        return name[:match.start(3)] + "1" + name[match.end(3):]
    return None


def read_headers(hub):
    """Reconstruct retained rows from upstream length prefixes and headers alone."""
    observed = {}
    aux_geometry = {}
    for variant in sorted(set(CASES.values())):
        mochi = variant == "mochi"
        repo, revision = MOCHI if mochi else SVD
        snapshot = Path(hub) / ("models--" + repo.replace("/", "--")) / "snapshots" / revision
        files = [MOCHI_DIT] if mochi else ([UNET_HALF, *AUX] if variant == "mlx-svd" else [UNET])
        geometry, digests = {}, {}
        for file in files:
            with (snapshot / file).open("rb") as handle:
                size, = struct.unpack("<Q", handle.read(8))
                if not 0 < size < 16 * 1024**2:
                    raise ValueError("invalid safetensors header length")
                raw = handle.read(size)
                if len(raw) != size:
                    raise ValueError("short source header")
            digests[file] = hashlib.sha256(raw).hexdigest()
            header = json.loads(raw)
            header.pop("__metadata__", None)
            if file in AUX:
                aux_geometry[file.split("/")[0]] = {k: v["shape"] for k, v in header.items()}
                continue
            rows = {}
            for name, meta in header.items():
                key = _retained(name, mochi)
                if key is None:
                    continue
                if key in rows:
                    raise ValueError(f"ambiguous retained key {key}")
                rows[key] = {k: meta[k] for k in ("dtype", "shape")}
            geometry[file.removeprefix("bf16/")] = rows
        observed[variant] = {"source": {"repo": repo, "snapshot": revision, "header_sha256": digests},
                             "geometry": geometry}
    return observed, aux_geometry


def verify_source_headers(hub):
    headers, aux = read_headers(hub)
    if headers != json.loads((HERE / "headers.json").read_bytes()):
        raise ValueError("denoiser source headers differ from frozen rows")
    if aux != json.loads((HERE / "aux_shapes.json").read_bytes()):
        raise ValueError("SVD pipeline auxiliary geometry differs from source")


def _float_tensor(shape, values):
    return "F32", shape, [struct.pack(f"<{math.prod(shape)}f", *values)]


def _inputs(variant):
    rng = random.Random(SEED)
    def normal(n):
        return [rng.gauss(0, 1) for _ in range(n)]

    def context_blocks():
        for _ in range(2 * 256):
            yield struct.pack("<4096f", *normal(4096))
    if variant == "mochi":
        shape = [1, 12, 2, 4, 6]
        # Paired rows have different context and real-token positions; padded
        # embeddings stay nonzero so ignoring a mask changes the predictions.
        return {"noise": _float_tensor(shape, normal(math.prod(shape))),
                "context": ("F32", [2, 256, 4096], context_blocks()),
                "mask": _float_tensor([2, 256], [float(i < n) for n in (13, 37) for i in range(256)])}
    frames, channels, height, width = 3, 4, 8, 16
    shape = [1, frames, channels, height, width]
    noise = normal(math.prod(shape))
    image = normal(channels * height * width) * frames
    if variant == "mlx-svd":
        order = [((f * channels + c) * height + h) * width + w
                 for f in range(frames) for h in range(height) for w in range(width) for c in range(channels)]
        noise, image = ([v[i] for i in order] for v in (noise, image))
        shape = [1, frames, height, width, channels]
    return {"noise": _float_tensor(shape, noise), "image_latents": _float_tensor(shape, image),
            "image_embeds": _float_tensor([1, 1, 1024], normal(1024))}


def _header(variant):
    return json.loads((HERE / "headers.json").read_bytes())[variant]


def make_fixture(root, case):
    root = Path(root)
    variant = CASES[case]
    for file, rows in _header(variant)["geometry"].items():
        tensors = {key: (m["dtype"], m["shape"], weight_blocks(f"mochi-svd/{key}", m["shape"], m["dtype"]))
                   for key, m in sorted(rows.items())}
        _write_tensors(root / file, tensors)
    _write_tensors(root / "inputs.safetensors", _inputs(variant))


def fixture_identity(root, case):
    root = Path(root)
    variant = CASES[case]
    header = _header(variant)
    files = sorted([*header["geometry"], "inputs.safetensors"])
    observed = sorted(p.relative_to(root).as_posix() for p in root.rglob("*.safetensors"))
    if files != observed:
        raise ValueError("unexpected fixture tensor files")
    return {"fixture_schema": 1, "variant": variant, "seed": SEED, "depth": 2 if variant == "mochi" else 1,
            "source": header["source"], "files": {name: sha(root / name) for name in files}}


def _f32(value):
    return struct.unpack("<f", struct.pack("<f", value))[0]


def _bits(value):
    return struct.unpack("<I", struct.pack("<f", value))[0]


def case_spec(case):
    variant = CASES[case]
    mochi, mlx = variant == "mochi", case.startswith("mlx-")
    heun = "heun" in case
    if mochi:
        shape = [1, 12, 2, 4, 6]
        # Invert after float32 rounding, matching the public Mochi schedule.
        sigmas = [_f32(1 - _f32(v)) for v in [1, 0.9875, 0.975, 0.725]] + [1.0]
        settings = {"sampler": "flow_euler", "shift": 1.0, "guidance": 4.5, "frames": 7,
                    "denoiser_dtype": "BF16", "heads": 24, "model_width": 3072, "depth": 2,
                    "context_shape": [2, 256, 4096], "mask_tokens": [13, 37]}
    else:
        shape = [1, 3, 8, 16, 4] if mlx else [1, 3, 4, 8, 16]
        lo, hi = _f32(0.002) ** (1 / 7), 700 ** (1 / 7)
        sigmas = [_f32((hi + (i / 3) * (lo - hi)) ** 7) for i in range(4)] + [0.0]
        single = case.endswith("-single")
        settings = {"sampler": "heun" if heun else "euler", "sampler_request": "heun" if heun else None,
                    "guidance": [1.0, 1.0 if single else 2.0, 1.0 if single else 3.0],
                    "frames": 3, "denoiser_dtype": "F16" if mlx else "F32",
                    "channels": [320, 640, 1280, 1280], "depth": 1,
                    "image_embeds_shape": [1, 1, 1024],
                    "time_id_bits": [_bits(x) for x in [6, 127, 0.02]]}
    settings.update(seed=SEED, steps=4, latent_shape=shape, sigma_bits=[_bits(x) for x in sigmas])
    trace = []
    for j, i in enumerate([0, 1, 1, 2, 2, 3, 3] if heun else range(4)):
        trace.append({"event": "step", "current": i + 1, "total": 4})
        if heun and mlx:
            trace.append({"event": "evaluate", "evaluation": j, "sigma_bits": _bits(sigmas[i])})
    return {"tensor_shape": shape, "outer_steps": 4, "trace": trace, "settings": settings}


def overlay_source(head, backend, model, arm, tracked):
    """Read complete probe source from the verified candidate's Git objects for either revision."""
    if backend not in ("mlx", "candle-cuda") or model not in ("mochi", "svd") or arm not in ("base", "head"):
        raise ValueError("unregistered video paired probe")
    prefix = "mlx" if backend == "mlx" else "cuda"
    def read(name):
        return tracked(head, "scripts/ci/ms_pair/" + name)
    probe = read(f"{prefix}_{model}_probe.rs.in")
    shared = tracked(head, "scripts/ci/wan_pair/common.rs.in") + b"\n" + read(f"{prefix}_common.rs.in")
    if backend == "candle-cuda":
        shared += b"\n" + tracked(head, "scripts/ci/wan_pair/cuda_pool.rs.in")
    shared += b"\n" + read(f"{model}_settings.rs.in")
    if model == "svd" and prefix == "mlx":
        token = b"@SVD_AUX_SHAPES@"
        if probe.count(token) != 1:
            raise ValueError("expected one auxiliary geometry declaration")
        probe = probe.replace(token, read("aux_shapes.json"))
    return shared + b"\n" + probe
