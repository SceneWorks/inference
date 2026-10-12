"""Dump the Iris-3B generation-controls and adapter goldens (sc-25681), extending `dump_iris_golden.py`.

Runs the FROZEN upstream `iris3b.sampling.generate` (see `_iris_common.py`) on the COMMITTED miniature
snapshot (`tests/fixtures/tiny-snapshot/`, written by `dump_iris_golden.py` — this script never
rewrites it, and first re-derives the committed `iris_e2e_golden` image from it to prove it is the
same snapshot), once per non-default upstream control:

* `base`         — the release controls (order 2, shift 4, CFG 3 over (0, 1), empty negative).
* `order1`       — `order=1` (the native `euler` sampler).
* `shift2`       — `shift=2.0` (the native `scheduler_shift`).
* `interval`     — `cfg_interval=(0.3, 0.8)`: guidance on the last two of five evaluations only.
* `negative`     — `negative_prompt="golden hour"` at `cfg_scale=4`.
* `cfg_off`      — `cfg_scale=1` (no unconditional branch at all).
* `portrait`     — a 12x8 (portrait) canvas instead of 8x12 (landscape).
* `batch`        — `prompts=[short, "golden hour snow"]`, one batched solve; row 0 shares `base`'s noise.
* `prediction_x` — the checkpoint read as `flow.prediction: x` (`x0 = out`).

Every case shares one injected noise per canvas shape, so each control's golden differs from `base`
only by that control. Written to `mlx-gen-iris/tests/fixtures/iris_controls_golden.safetensors`
(`<case>/noise`, `<case>/image`; the case table travels as JSON in the `cases` metadata). Each case is
also rendered with the text tower in fp32 (`<case>/image_f32_tower`, everything else identical) — the
reference for a native lane whose tower computes in f32 (Candle CPU), and the measured size of
upstream's own bf16-vs-fp32 tower distance that bounds the bf16 comparison.

Adapters (upstream ships none; the format is the repo's trainer output): a PEFT LoRA
(`transformer.<path>.lora_A/B.weight` + `<path>.alpha`, `networkType=lora`) and a PEFT-stamped LoKr
(`<path>.lokr_w1` full + `<path>.lokr_w2_a/_b` low-rank, `networkType=lokr`, `rank`/`alpha`) over four
projections spanning the text adapter, a dual-stream block, a single-stream block and the pixel head,
stamped `family=iris`, `baseModel=iris_3b`, `irisTask=generation`; plus the same LoRA stamped
`irisTask=depth` (must be refused). The expected outcome is upstream's own forward with the delta
folded into the weights (`W += δ`, f32), on the `iris_dit_golden` inputs:
`iris_adapter_golden.safetensors` holds `delta/{lora,lokr}/<path>` and `out/{lora,lokr,both}`.

Run: `IRIS_SRC=/path/to/iris-3b python -I tools/dump_iris_controls.py` (see `_iris_common.py`).
"""

from __future__ import annotations

import json
import tempfile

import dump_iris_golden as base_dump  # noqa: E402  (imports the frozen upstream; main() is guarded)
from _iris_common import FIXTURE_DIR, IRIS_COMMIT, save_safetensors  # noqa: E402

import torch  # noqa: E402
from safetensors.torch import load_file, save_file  # noqa: E402

from iris3b.config import TextEncoderConfig  # noqa: E402
from iris3b.models.dit import IrisDiT  # noqa: E402
from iris3b.sampling import generate  # noqa: E402
from iris3b.text.qwen3_vl import Qwen3VLTextEncoder  # noqa: E402

SHORT = base_dump.PROMPTS["short"]
BASE = dict(prompts=[SHORT], steps=5, order=2, cfg_scale=3.0, cfg_interval=[0.0, 1.0], shift=4.0,
            negative_prompt="", height=8, width=12, prediction="v")
CASES = [
    dict(BASE, name="base"),
    dict(BASE, name="order1", order=1),
    dict(BASE, name="shift2", shift=2.0),
    dict(BASE, name="interval", cfg_interval=[0.3, 0.8]),
    dict(BASE, name="negative", negative_prompt="golden hour", cfg_scale=4.0),
    dict(BASE, name="cfg_off", cfg_scale=1.0),
    dict(BASE, name="portrait", height=12, width=8),
    dict(BASE, name="batch", prompts=[SHORT, "golden hour snow"]),
    dict(BASE, name="prediction_x", prediction="x"),
]

STAMPS = {"family": "iris", "baseModel": "iris_3b", "irisTask": "generation"}
LORA_RANK, LORA_ALPHA, LORA_SCALE = 2, 4.0, 0.75
LOKR_RANK, LOKR_ALPHA, LOKR_SCALE = 2, 1.0, 1.0


def load_snapshot() -> IrisDiT:
    model = IrisDiT(base_dump.TINY_DIT).eval()
    model.load_state_dict(load_file(str(base_dump.BACKBONE_DIR / "model.safetensors")), strict=True)
    return model


def noise_for(shape: tuple[int, int], rows: int) -> torch.Tensor:
    g = torch.Generator().manual_seed(11 + 100 * shape[0] + shape[1])
    one = torch.randn(1, 3, *shape, generator=g)
    extra = torch.randn(rows - 1, 3, *shape, generator=g) if rows > 1 else one[:0]
    return torch.cat([one, extra], 0)


def render(model: IrisDiT, enc: Qwen3VLTextEncoder, case: dict, noise: torch.Tensor) -> torch.Tensor:
    with torch.no_grad():
        return generate(
            model, enc, case["prompts"], height=case["height"], width=case["width"],
            steps=case["steps"], order=case["order"], cfg_scale=case["cfg_scale"],
            cfg_interval=tuple(case["cfg_interval"]), shift=case["shift"],
            negative_prompt=case["negative_prompt"], device="cpu", noise=noise,
            prediction=case["prediction"],
        )


def dump_controls(model: IrisDiT, enc: Qwen3VLTextEncoder, enc_f32: Qwen3VLTextEncoder) -> tuple[dict, list]:
    out = {}
    for case in CASES:
        noise = noise_for((case["height"], case["width"]), len(case["prompts"]))
        image = render(model, enc, case, noise)
        image_f32 = render(model, enc_f32, case, noise)
        out[f"{case['name']}/noise"] = noise
        out[f"{case['name']}/image"] = image
        out[f"{case['name']}/image_f32_tower"] = image_f32
        print(f"controls {case['name']}: image {tuple(image.shape)} mean={image.mean():.4f} "
              f"bf16-vs-fp32 tower max|Δ|={(image - image_f32).abs().max():.3e}")
    base_image = out["base/image"]
    for case in CASES[1:]:
        img = out[f"{case['name']}/image"]
        if img.shape == base_image.shape:
            print(f"  {case['name']} vs base: max|Δ|={(img - base_image).abs().max():.3e}")
    return out, CASES


def adapter_targets(model: IrisDiT) -> list[str]:
    linears = [n for n, m in model.named_modules() if isinstance(m, torch.nn.Linear)]
    picks = []
    for prefix in ("y_embedder.refiner.", "blocks.0.", "blocks.2.", "pixel_blocks.0."):
        name = next(n for n in linears if n.startswith(prefix) and n not in picks)
        picks.append(name)
    return picks


def smallest_divisor(n: int) -> int:
    return next(d for d in range(2, n + 1) if n % d == 0)


def dump_adapters(model: IrisDiT) -> None:
    g = torch.Generator().manual_seed(7)
    state = {k: v.clone() for k, v in model.state_dict().items()}
    targets = adapter_targets(model)
    lora, lokr, deltas = {}, {}, {}
    for path in targets:
        w = state[f"{path}.weight"]
        out_f, in_f = w.shape
        a = 0.05 * torch.randn(LORA_RANK, in_f, generator=g)
        b = 0.05 * torch.randn(out_f, LORA_RANK, generator=g)
        lora[f"transformer.{path}.lora_A.weight"] = a
        lora[f"transformer.{path}.lora_B.weight"] = b
        lora[f"transformer.{path}.alpha"] = torch.tensor([LORA_ALPHA])
        deltas[f"delta/lora/{path}"] = LORA_SCALE * (LORA_ALPHA / LORA_RANK) * (b @ a)
        # LoKr: out = a1·b1, in = c1·d1; w1 [a1, c1] full, w2 = w2_a [b1, r] @ w2_b [r, d1].
        a1, c1 = smallest_divisor(out_f), smallest_divisor(in_f)
        b1, d1 = out_f // a1, in_f // c1
        w1 = 0.3 * torch.randn(a1, c1, generator=g)
        w2_a = 0.3 * torch.randn(b1, LOKR_RANK, generator=g)
        w2_b = 0.3 * torch.randn(LOKR_RANK, d1, generator=g)
        lokr[f"{path}.lokr_w1"] = w1
        lokr[f"{path}.lokr_w2_a"] = w2_a
        lokr[f"{path}.lokr_w2_b"] = w2_b
        deltas[f"delta/lokr/{path}"] = (
            LOKR_SCALE * (LOKR_ALPHA / LOKR_RANK) * torch.kron(w1, w2_a @ w2_b)
        )
        assert deltas[f"delta/lokr/{path}"].shape == w.shape
    meta_lora = {**STAMPS, "networkType": "lora", "rank": str(LORA_RANK), "alpha": str(LORA_ALPHA)}
    meta_lokr = {**STAMPS, "networkType": "lokr", "rank": str(LOKR_RANK), "alpha": str(LOKR_ALPHA)}
    save_file({k: v.contiguous() for k, v in lora.items()}, str(FIXTURE_DIR / "iris_lora.safetensors"),
              metadata=meta_lora)
    save_file({k: v.contiguous() for k, v in lokr.items()}, str(FIXTURE_DIR / "iris_lokr.safetensors"),
              metadata=meta_lokr)
    save_file({k: v.contiguous() for k, v in lora.items()},
              str(FIXTURE_DIR / "iris_lora_depth_task.safetensors"),
              metadata={**meta_lora, "irisTask": "depth"})

    dit = load_file(str(FIXTURE_DIR / "iris_dit_golden.safetensors"))
    x, t, y, mask = dit["x"], dit["t"], dit["y"], dit["y_mask"].to(torch.int64)

    def forward_with(kinds: list[str]) -> torch.Tensor:
        merged = {k: v.clone() for k, v in state.items()}
        for kind in kinds:
            for path in targets:
                merged[f"{path}.weight"] += deltas[f"delta/{kind}/{path}"]
        m = IrisDiT(base_dump.TINY_DIT).eval()
        m.load_state_dict(merged, strict=True)
        with torch.no_grad():
            return m(x, t, y, y_mask=mask).x

    outs = {f"out/{name}": forward_with(kinds)
            for name, kinds in (("lora", ["lora"]), ("lokr", ["lokr"]), ("both", ["lora", "lokr"]))}
    torch.testing.assert_close(forward_with([]), dit["out"], rtol=0, atol=1e-6)
    for name, value in outs.items():
        print(f"adapter {name}: max|Δ vs base|={(value - dit['out']).abs().max():.3e}")
    save_safetensors(
        FIXTURE_DIR / "iris_adapter_golden.safetensors",
        {**deltas, **outs},
        {"upstream_commit": IRIS_COMMIT, "targets": json.dumps(targets),
         "lora_scale": LORA_SCALE, "lokr_scale": LOKR_SCALE},
    )


def main() -> None:
    model = load_snapshot()
    with tempfile.TemporaryDirectory() as null_dir:
        cfg = TextEncoderConfig(**{**vars(base_dump.TINY_TEXT), "null_embed_dir": null_dir})
        enc = Qwen3VLTextEncoder(cfg, device="cpu")
        # The committed snapshot is the one the committed e2e golden came from.
        e2e = load_file(str(FIXTURE_DIR / "iris_e2e_golden.safetensors"))
        with torch.no_grad():
            again = generate(model, enc, [SHORT], height=8, width=12, steps=6, order=2, cfg_scale=3.0,
                             cfg_interval=(0.0, 1.0), shift=4.0, negative_prompt="", device="cpu",
                             noise=e2e["noise"])
        torch.testing.assert_close(again, e2e["image"], rtol=0, atol=1e-6)
        # A separate null-embedding cache: upstream keys `null("")`'s on-disk memo by repo, window and
        # layers but not by dtype, so a shared directory would hand the fp32 tower the bf16 null.
        with tempfile.TemporaryDirectory() as null_dir_f32:
            cfg_f32 = TextEncoderConfig(
                **{**vars(cfg), "dtype": "float32", "null_embed_dir": null_dir_f32}
            )
            enc_f32 = Qwen3VLTextEncoder(cfg_f32, device="cpu")
            tensors, cases = dump_controls(model, enc, enc_f32)
        save_safetensors(
            FIXTURE_DIR / "iris_controls_golden.safetensors",
            tensors,
            {"upstream_commit": IRIS_COMMIT, "torch": torch.__version__, "cases": json.dumps(cases)},
        )
    dump_adapters(model)


if __name__ == "__main__":
    main()
