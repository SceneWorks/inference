"""Dump the Iris-3B restoration goldens for the native MLX/Candle restorers (sc-25683).

Everything runs the FROZEN upstream restoration code (`iris3b/downstream/restoration.py`,
`iris3b/downstream/__init__.py`, `scripts/export_downstream.py`, `scripts/upscale.py`; sha256-pinned
below on top of `_iris_common.py`'s pins) on CPU in fp32. Nothing at runtime depends on Python.

Writes `mlx-gen-iris/tests/fixtures/`:

* `tiny-snapshot/upscaler/{config.yaml, model.safetensors, empty_prompt.safetensors}` — a miniature
  restoration export written by upstream's OWN `scripts/export_downstream.py restoration` from a
  synthetic training payload: the miniature generation backbone (`tiny-snapshot/iris`) with every
  parameter perturbed (a distinct "fine-tune"), a 3-token empty-prompt cache, `task.tile` = 32
  (a multiple of the miniature patch 4, so the multi-tile paths run in milliseconds) and
  `task.sigma` = 0.5 (the release's `model_t = 500`).
* `iris_restoration_golden.safetensors`:
  - `ops/bicubic/<case>/{in,out}` — `F.interpolate(scale_factor=…, bicubic)` incl. a fractional
    scale on odd sides, a downscale and a non-dyadic scale (1.3);
  - `ops/aa/<case>/{in,out}` — `F.interpolate(size=…, bicubic, antialias=True)` up, down and with
    one axis unchanged;
  - `ops/window/32`, `ops/window/1024/{centre_row,centre_col,sub}` — `gaussian_window`;
  - `ops/wavelet/{content,reference,out}` — `wavelet_color_fix` on a non-square image smaller than
    the largest dilation (replicate-padding boundary);
  - `e2e/<case>/{input,output,tiled[,fixed]}` — upstream `Restorer.__call__` end to end on the
    miniature export (`input`/`output` RGB8; `tiled` = the fused `tiled()` result, `fixed` = the
    colour-fixed float image, captured by wrapping the module functions);
  - metadata `tile_positions` (JSON), `fit_budget` (JSON: per case the formula input size, the
    output size and the sha256 of upstream's PIL-LANCZOS output bytes), `cases` (JSON).

Run: `IRIS_SRC=/path/to/iris-3b python -I tools/dump_iris_restoration.py` (see `_iris_common.py`).
"""

from __future__ import annotations

import hashlib
import json
import shutil
import sys
import tempfile
from pathlib import Path

# `python -I` does not put the script's directory on sys.path; the shared helpers live beside it.
sys.path.insert(0, str(Path(__file__).resolve().parent))

from _iris_common import FIXTURE_DIR, IRIS_COMMIT, import_upstream, save_safetensors

ROOT = import_upstream()

# sha256 of the upstream restoration modules this oracle executes, at IRIS_COMMIT.
RESTORATION_SHA256 = {
    "src/iris3b/downstream/__init__.py": "94034cdb0da5189b07f01b5f57f2a0cd04cda14cfb7cf9df1bc77e375cd02cec",
    "src/iris3b/downstream/restoration.py": "b523fac5d53705a8a1c1575b7cdf9ea8c75396ddeb0a5101e54abd662e072b61",
    "scripts/export_downstream.py": "be14ab9cfc443b8bd348a6ccde19843ae6806125620d1e2a090ee90dfbcc3afa",
    "scripts/upscale.py": "310be22e60038593d7d31db337c2979f7b32b273fcd715eb5951c6b0e195e721",
}

for rel, want in RESTORATION_SHA256.items():
    got = hashlib.sha256((ROOT / rel).read_bytes()).hexdigest()
    if got != want:
        raise SystemExit(f"{rel}: sha256 {got} != pinned {want!r}")
sys.path.insert(0, str(ROOT / "scripts"))

import numpy as np  # noqa: E402
import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402
from PIL import Image  # noqa: E402
from safetensors.torch import load_file  # noqa: E402

import export_downstream  # noqa: E402
from iris3b.downstream import restoration  # noqa: E402
from iris3b.downstream.restoration import Restorer, fit_budget, gaussian_window, tile_positions  # noqa: E402

torch.manual_seed(0)

SNAPSHOT = FIXTURE_DIR / "tiny-snapshot"
PARENT = SNAPSHOT / "iris"
EXPORT = SNAPSHOT / "upscaler"
TILE = 32
MODEL_T = 500


def formula_rgb(width: int, height: int) -> np.ndarray:
    """Deterministic RGB8 test image the Rust side regenerates bit for bit."""
    y, x = np.mgrid[0:height, 0:width].astype(np.int64)
    channels = [(x * 7 + y * 13 + c * 101 + (x * y) % 97) % 256 for c in range(3)]
    return np.stack(channels, axis=-1).astype(np.uint8)


def write_export() -> None:
    parent = load_file(str(PARENT / "model.safetensors"))
    g = torch.Generator().manual_seed(25683)
    weights = {
        f"model.{k}": v + 0.05 * v.abs().mean().clamp(min=1e-3) * torch.randn(v.shape, generator=g)
        for k, v in parent.items()
    }
    text_len, layers, dim = 10, 12, 32
    mask = torch.zeros(1, text_len, dtype=torch.bool)
    mask[0, :3] = True
    embeddings = torch.randn(1, text_len, layers, dim, generator=g) * mask[..., None, None]
    payload = {
        "config": {
            "generator": {"mode": "full", "model_t": MODEL_T, "coeff_t": MODEL_T},
            "data": {"crop_size": TILE},
        },
        "generator": weights,
    }
    if EXPORT.exists():
        shutil.rmtree(EXPORT)
    with tempfile.TemporaryDirectory() as tmp:
        ckpt, cond = Path(tmp) / "inference.pth", Path(tmp) / "empty-conditioning.pt"
        torch.save(payload, ckpt)
        torch.save({"embeddings": embeddings, "mask": mask}, cond)
        export_downstream.main([
            "restoration", str(ckpt), str(EXPORT), "--parent-config", str(PARENT / "config.yaml"),
            "--conditioning", str(cond),
        ])


def dump_ops(out: dict, meta: dict) -> None:
    g = torch.Generator().manual_seed(1)
    for name, (h, w), scale in [
        ("x4", (5, 7), 4.0),
        ("x2_5_odd", (9, 13), 2.5),
        ("x1", (6, 10), 1.0),
        ("x0_75", (12, 16), 0.75),
        ("x3", (11, 9), 3.0),
        # non-dyadic: floor(20 * 1.3) = 26 only with the Python float (f32 1.3 would give 25), and
        # the coordinate scale is f32(1 / 1.3)
        ("x1_3_nondyadic", (10, 20), 1.3),
    ]:
        x = torch.rand(1, 3, h, w, generator=g) * 1.2 - 0.1
        out[f"ops/bicubic/{name}/in"] = x
        out[f"ops/bicubic/{name}/out"] = F.interpolate(x, scale_factor=scale, mode="bicubic", align_corners=False)
        meta.setdefault("bicubic", {})[name] = scale
    for name, (h, w), size in [
        ("up", (12, 20), (32, 53)),
        ("down", (32, 56), (24, 40)),
        ("width_only", (32, 32), (32, 50)),
        ("down_odd", (45, 33), (17, 29)),
    ]:
        x = torch.rand(1, 3, h, w, generator=g) * 1.2 - 0.1
        out[f"ops/aa/{name}/in"] = x
        out[f"ops/aa/{name}/out"] = F.interpolate(x, size=size, mode="bicubic", align_corners=False, antialias=True)
    out["ops/window/32"] = gaussian_window(TILE, torch.device("cpu"))[0, 0]
    big = gaussian_window(1024, torch.device("cpu"))[0, 0]
    out["ops/window/1024/centre_row"] = big[512]
    out["ops/window/1024/centre_col"] = big[:, 511].contiguous()
    out["ops/window/1024/sub"] = big[::37, ::41].contiguous()
    content = torch.rand(1, 3, 29, 37, generator=g)
    reference = torch.rand(1, 3, 29, 37, generator=g)
    out["ops/wavelet/content"] = content
    out["ops/wavelet/reference"] = reference
    out["ops/wavelet/out"] = restoration.wavelet_color_fix(content, reference)
    meta["tile_positions"] = [
        [size, tile, stride, tile_positions(size, tile, stride)]
        for size, tile, stride in [
            (1024, 1024, 512), (1025, 1024, 512), (1536, 1024, 512), (2048, 1024, 512),
            (2064, 1024, 512), (4096, 1024, 512), (32, 32, 16), (112, 32, 16), (96, 32, 16),
        ]
    ]
    budget = []
    for w, h in [(1100, 700), (700, 1100), (2050, 300), (513, 513), (512, 1024), (1030, 20), (300, 200)]:
        img = fit_budget(Image.fromarray(formula_rgb(w, h), "RGB"))
        data = np.asarray(img.convert("RGB")).tobytes()
        budget.append({"input": [w, h], "output": list(img.size), "sha256": hashlib.sha256(data).hexdigest()})
    meta["fit_budget"] = budget


def dump_e2e(out: dict, meta: dict) -> None:
    restorer = Restorer(str(EXPORT), device="cpu")
    assert restorer.tile == TILE and restorer.sigma == MODEL_T / 1000
    captured: dict = {}
    tiled, fix = restoration.tiled, restoration.wavelet_color_fix

    def tiled_spy(*a, **k):
        captured["tiled"] = tiled(*a, **k).clone()
        return captured["tiled"]

    def fix_spy(*a, **k):
        captured["fixed"] = fix(*a, **k).clone()
        return captured["fixed"]

    restoration.tiled, restoration.wavelet_color_fix = tiled_spy, fix_spy
    g = torch.Generator().manual_seed(2)
    cases = [
        # name, (w, h), scale, colour fix, budgeted
        ("small_enlarge", (10, 6), 4.0, True, False),  # 40x24 output: short side <= tile
        ("tiled_nofix", (40, 24), 4.0, False, False),
        ("restore_1x", (70, 50), 1.0, True, False),
        ("fractional", (13, 9), 2.5, True, False),
        ("single_tile", (8, 8), 4.0, True, False),
        ("portrait_3x", (11, 23), 3.0, True, False),
        ("budgeted", (1030, 20), 1.0, True, True),
        # non-dyadic scale through the request path: 26x13 (f32-widened 1.3 would plan 25x12)
        ("non_dyadic_1_3", (20, 10), 1.3, True, False),
    ]
    for name, (w, h), scale, color_fix, budgeted in cases:
        captured.clear()
        if budgeted:
            pixels = formula_rgb(w, h)
        else:
            pixels = torch.randint(0, 256, (h, w, 3), generator=g, dtype=torch.uint8).numpy()
        image = Image.fromarray(pixels, "RGB")
        if budgeted:
            image = fit_budget(image)  # scripts/upscale.py's default path
        result = restorer(image, scale=scale, color_fix=color_fix)
        out[f"e2e/{name}/input"] = torch.from_numpy(pixels.copy())
        out[f"e2e/{name}/output"] = torch.from_numpy(np.asarray(result).copy())
        out[f"e2e/{name}/tiled"] = captured["tiled"][0]
        if color_fix:
            out[f"e2e/{name}/fixed"] = captured["fixed"][0]
        meta.setdefault("cases", []).append(
            {"name": name, "input": [w, h], "scale": scale, "color_fix": color_fix,
             "budgeted": budgeted, "output": list(result.size),
             "enlarged": min(result.size) <= TILE}
        )
        print(f"{name}: {w}x{h} x{scale} -> {result.size}", flush=True)
    restoration.tiled, restoration.wavelet_color_fix = tiled, fix


def main() -> None:
    write_export()
    out: dict = {}
    meta: dict = {}
    dump_ops(out, meta)
    dump_e2e(out, meta)
    save_safetensors(
        FIXTURE_DIR / "iris_restoration_golden.safetensors",
        out,
        {"iris_commit": IRIS_COMMIT, **{k: json.dumps(v) for k, v in meta.items()}},
    )


if __name__ == "__main__":
    main()
