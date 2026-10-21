"""Drive the sc-25777 MiniMax-H3 Veda dense-vs-sparse A/B on the Windows CUDA host; dispatch only.

Each cell of `VEDA_AB_CELLS` (a JSON array of `VEDA_AB_*` env maps) renders twice, `VEDA_AB_VEDA=off`
then `on`, each in its own process of the prebuilt `candle-gen-minimax-h3` integration binary
(`veda_ab::veda_ab_render`). The physical GPU is claimed through the shared Veda CUDA guard (idle
check + host-global UUID lock) and re-verified idle before every render. A failed render is recorded
and the remaining renders still run; the exit status is non-zero if any render failed.

Placeholders expanded in cell values: `{PREDICTOR}` (the Veda predictor file), `{TURBO}` (the
turbo LoRA directory), `{CLAY}` (the claymation LoRA file) and `{REPO}` (the checkout root). A turbo
`VEDA_AB_RECIPE` adds its own LoRA at strength 1.0 unless `VEDA_AB_LORAS` already names that file,
which is what selecting a turbo variant does in the product.
"""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from scripts.release import veda_cuda_guard as guard  # noqa: E402

TEST = "veda_ab::veda_ab_render"
# The request contract shared with the MLX twin. `VEDA_AB_OUT` and `VEDA_AB_VEDA` are owned here.
CELL_KEYS = frozenset({
    "VEDA_AB_MODE", "VEDA_AB_FIRST_IMAGE", "VEDA_AB_LAST_IMAGE", "VEDA_AB_REF_IMAGES",
    "VEDA_AB_TIER", "VEDA_AB_LORAS", "VEDA_AB_RECIPE", "VEDA_AB_STEPS", "VEDA_AB_WIDTH",
    "VEDA_AB_HEIGHT", "VEDA_AB_FRAMES", "VEDA_AB_SEED", "VEDA_AB_PROMPT", "VEDA_AB_PREDICTOR",
})
MAX_CELLS = 8
TURBO_FILES = {
    "minimax_h3_turbo_8step": "minimax_h3_fl2v_turbo_8step_v1.0_bf16.safetensors",
    "minimax_h3_turbo_4step_768p": "minimax_h3_fl2v_turbo_4step_v1.0_768p_bf16.safetensors",
    "minimax_h3_turbo_4step_v01": "minimax_h3_fl2v_turbo_4step_v0.1.safetensors",
    "minimax_h3_ref2v_turbo_4step": "minimax_h3_ref2v_turbo_4step_v0.1_bf16.safetensors",
}
PREDICTOR_FILE = "minimax_h3_t2va_veda_8nfe_600step_preview_fp8.safetensors"
PREDICTOR_SHA256 = "2a8d8845c5342756a2781e8e69563940e4bb573c9a40ebb534915ff8fd76573a"
CLAY_FILE = "checkpoints/full_claymation_h3_v1_claymation_h3_v1.safetensors"
CLAY_SHA256 = "0fac46be5b3498124f1bb596e9f9050338ae8028425650d5524ed40fce50f2f8"


def sha256(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def compiled_binary(build_log: Path, target: Path) -> Path:
    """Cargo's exact `integration` test artifact from `--message-format=json`; never a glob."""
    candidates, finished = [], False
    for line in build_log.read_text(encoding="utf-8", errors="replace").splitlines():
        try:
            row = json.loads(line)
        except json.JSONDecodeError:
            continue  # vcvars diagnostics precede Cargo's JSON stream.
        if row.get("reason") == "build-finished":
            finished = row.get("success") is True
        if (row.get("reason") == "compiler-artifact"
                and row.get("package_id", "").find("candle-gen-minimax-h3") >= 0
                and row.get("target", {}).get("name") == "integration"
                and row.get("executable")):
            candidates.append(Path(row["executable"]).resolve())
    if not finished or len(candidates) != 1:
        raise RuntimeError("Cargo did not identify exactly one candle-gen-minimax-h3 integration binary")
    if not candidates[0].is_relative_to(target.resolve()) or not candidates[0].is_file():
        raise RuntimeError("integration binary is absent or outside the isolated Cargo target")
    return candidates[0]


def parse_cells(text: str) -> list[dict[str, str]]:
    cells = json.loads(text)
    if not isinstance(cells, list) or not 1 <= len(cells) <= MAX_CELLS:
        raise ValueError(f"cells must be a JSON array of 1..{MAX_CELLS} objects")
    parsed = []
    for index, cell in enumerate(cells):
        if not isinstance(cell, dict):
            raise ValueError(f"cell {index} is not an object")
        unknown = sorted(set(cell) - CELL_KEYS)
        if unknown:
            raise ValueError(f"cell {index}: keys outside the VEDA_AB contract: {unknown}")
        values = {}
        for key, value in cell.items():
            if isinstance(value, bool) or not isinstance(value, (str, int, float)):
                raise ValueError(f"cell {index}: {key} must be a string or number")
            values[key] = str(value)
        parsed.append(values)
    return parsed


def expand(cell: dict[str, str], tokens: dict[str, str]) -> dict[str, str]:
    out = {}
    for key, value in cell.items():
        for token, replacement in tokens.items():
            value = value.replace(token, replacement)
        out[key] = value
    recipe = out.get("VEDA_AB_RECIPE", "base")
    turbo = TURBO_FILES.get(recipe)
    if turbo is not None:
        lora = str(Path(tokens["{TURBO}"]) / turbo)
        existing = [e for e in out.get("VEDA_AB_LORAS", "").split(",") if e.strip()]
        if not any(Path(e.strip()).name.startswith(turbo) for e in existing):
            out["VEDA_AB_LORAS"] = ",".join([f"{lora}:1.0", *existing])
    tier = out.get("VEDA_AB_TIER", "q4")
    if tier in {"q4", "q8"}:
        out["VEDA_AB_DIT_DIR"] = os.environ[f"H3_{tier.upper()}_DIT_DIR"]
        out["VEDA_AB_TE_DIR"] = os.environ[f"H3_{tier.upper()}_TE_DIR"]
    out.setdefault("VEDA_AB_PREDICTOR", tokens["{PREDICTOR}"])
    return out


def cell_name(index: int, cell: dict[str, str]) -> str:
    parts = [cell.get("VEDA_AB_MODE", "t2va"), cell.get("VEDA_AB_RECIPE", "base"),
             cell.get("VEDA_AB_TIER", "q4")]
    return f"cell{index:02d}-" + "-".join(p.replace("_", "") for p in parts)


def verify_render(out: Path, result: dict) -> str | None:
    """Return a failure reason, or None when the render proved one passing test and a receipt."""
    stdout = (out / "render.stdout.log").read_text(encoding="utf-8", errors="replace")
    if result["timeout"]:
        return "timeout"
    if result["returncode"] != 0 or "test result: ok. 1 passed" not in stdout:
        return f"exit {result['returncode']} or not exactly one passing test"
    if not (out / "result.json").is_file():
        return "result.json missing"
    return None


def main() -> int:
    from scripts.media_campaign_windows import WindowsJob

    root = Path(os.environ["VEDA_AB_RESULTS_DIR"])
    root.mkdir(parents=True, exist_ok=True)
    render_seconds = int(os.environ.get("VEDA_AB_RENDER_TIMEOUT_MINUTES", "180")) * 60
    summary = dict(schema="sc-25777-veda-ab-cuda-v1", complete=False, status="starting",
                   runner=os.environ.get("RUNNER_NAME"), source_commit=os.environ.get("GITHUB_SHA"),
                   renders=[], started_unix=time.time())
    guard.persist_receipt(root, summary)
    lock = None
    failures = 0
    try:
        if os.name != "nt" or summary["runner"] not in guard.ALLOWED_RUNNERS:
            raise RuntimeError("the A/B requires an admitted Windows CUDA listener")
        binary = compiled_binary(Path(os.environ["VEDA_AB_BUILD_LOG"]),
                                 Path(os.environ["CARGO_TARGET_DIR"]))
        summary["binary"] = str(binary)
        summary["binary_sha256"] = sha256(binary)
        predictor = Path(os.environ["VEDA_AB_PREDICTOR_SNAPSHOT"]) / PREDICTOR_FILE
        clay = Path(os.environ["VEDA_AB_CLAY_SNAPSHOT"]) / CLAY_FILE
        for path, expected in ((predictor, PREDICTOR_SHA256), (clay, CLAY_SHA256)):
            if sha256(path) != expected:
                raise RuntimeError(f"{path} does not match its published SHA-256")
        tokens = {
            "{PREDICTOR}": str(predictor),
            "{TURBO}": os.environ["VEDA_AB_TURBO_SNAPSHOT"],
            "{CLAY}": str(clay),
            "{REPO}": str(Path.cwd()),
        }
        cells = [expand(c, tokens) for c in parse_cells(os.environ["VEDA_AB_CELLS"])]
        summary["cells"] = cells
        device, lock, baseline = guard.claim_device(root)
        summary["device"] = device
        os.environ["CUDA_VISIBLE_DEVICES"] = device["uuid"]
        summary["status"] = "device_admitted"
        guard.persist_receipt(root, summary)
        base_env = {k: v for k, v in os.environ.items() if not k.startswith("VEDA_AB_")}
        for index, cell in enumerate(cells):
            for arm in ("off", "on"):
                out = root / cell_name(index, cell) / f"veda-{arm}"
                out.mkdir(parents=True, exist_ok=True)
                guard.verify_idle(device, out, "pre-render", baseline)
                env = dict(base_env, **cell, VEDA_AB_OUT=str(out), VEDA_AB_VEDA=arm)
                argv = [str(binary), TEST, "--ignored", "--exact", "--nocapture"]
                saved = os.environ.copy()
                os.environ.clear()
                os.environ.update(env)
                try:
                    result = guard.execute_owned("render", argv, out,
                                                 time.monotonic() + render_seconds, WindowsJob)
                finally:
                    os.environ.clear()
                    os.environ.update(saved)
                reason = verify_render(out, result)
                result.update(cell=index, arm=arm, out=str(out), failure=reason)
                failures += reason is not None
                summary["renders"].append(result)
                summary["status"] = f"cell{index}_{arm}_done"
                guard.persist_receipt(root, summary)
                print(f"cell {index} veda {arm}: {reason or 'ok'} ({result['seconds']:.0f} s)",
                      flush=True)
        summary["status"] = "complete" if failures == 0 else f"{failures}_renders_failed"
        summary["complete"] = failures == 0
        return 0 if failures == 0 else 1
    except Exception as exc:
        summary["status"] = "refused_or_failed"
        summary["error"] = f"{type(exc).__name__}: {exc}"
        print(summary["error"], file=sys.stderr)
        return 2
    finally:
        summary["finished_unix"] = time.time()
        guard.persist_receipt(root, summary)
        if lock is not None:
            lock.close()


if __name__ == "__main__":
    sys.exit(main())
