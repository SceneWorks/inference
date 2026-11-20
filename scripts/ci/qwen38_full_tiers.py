#!/usr/bin/env python3
"""Convert and publish the fully quantized Qwen3.8-27B tiers (SW-7202, feature SW-F374).

Run only by the reusable `.github/workflows/qwen38-full-tiers.yml`, which the dispatch-only
`qwen38-full-tiers-*` profiles of `.github/workflows/real-weights.yml` call, one model and one
backend per job:

* `describer` — `huihui-ai/Huihui-Qwen3.8-27B-abliterated` (bf16) to
  `SceneWorks/qwen3.8-27b-abliterated-mlx` (MLX affine) or `SceneWorks/qwen3.8-27b-abliterated-candle`
  (Candle GGML blocks), folders `q8-full/` and `q4-full/`;
* `expander` — `MiniMaxAI/MiniMax-H3-Image-PE` (NVFP4, dequantized to a bf16 scratch intermediate
  first) to `SceneWorks/minimax-h3-image-pe`, folders `q8-full/` and `q4-full/` (MLX) or
  `candle-q8-full/` and `candle-q4-full/` (Candle).

The tiers are SW-7201's `prepare_snapshot` recipes `q8-full` (every language-model matrix 8-bit,
`lm_head`, `embed_tokens` and every Gated DeltaNet `in_proj_a`/`in_proj_b` included) and `q4-full`
(4-bit body; those matrices 8-bit). The vision tower and the MTP head are stored as in the plain
tiers. The tiers go to NEW folders beside the plain `q8/` / `q4/` (and `candle-q8/` / `candle-q4/`)
folders, which stay untouched at the repository head, so every install pinned to an earlier
revision keeps working and the plain folders stay downloadable until the SceneWorks manifest moves.

Each tier folder carries its own `SOURCE_REVISION.json` and `README.md` recording the source
repository and revision, the recipe, and the inference revision (`GITHUB_SHA`) the converter was
built from. Uploads use the runner's own Hugging Face login through `publish_hf_tiers.py`, which
verifies every published file against the local bytes and writes the publication report the
SceneWorks manifest pins are filled from.

Subcommands (each prints what it did; any failure exits non-zero):

    materialize --model M --hub DIR --env-name NAME --need-gb N
                                                      the pinned source into the hub; NAME=<path> to GITHUB_ENV
    headroom --path DIR --need-gb N                   refuse unless N GB + 25 GB stay free under DIR
    convert --model M --backend B --scratch DIR --bin-dir DIR
    publish --model M --backend B --scratch DIR --reports DIR
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
MANIFEST = ROOT / "release" / "real-weight-models.toml"
RECIPES = ("q8-full", "q4-full")
# Free space that must remain after every write, so the box never runs out of disk (nax runners
# have hit ENOSPC on Hub uploads).
RESERVE_GB = 25

MODELS = {
    "describer": {
        "source_key": "qwen38-27b-abliterated",
        "nvfp4": False,
        "license": "Apache-2.0 (as released by Qwen and kept by huihui-ai; see the repository's LICENSE)",
        "repos": {
            "mlx": "SceneWorks/qwen3.8-27b-abliterated-mlx",
            "candle": "SceneWorks/qwen3.8-27b-abliterated-candle",
        },
        "folders": {
            "mlx": {"q8-full": "q8-full", "q4-full": "q4-full"},
            "candle": {"q8-full": "q8-full", "q4-full": "q4-full"},
        },
    },
    "expander": {
        "source_key": "minimax-h3-image-pe",
        "nvfp4": True,
        "license": "MiniMax-H3-Image Community License Agreement (MiniMax Materials). The licensed "
                   "prompt files are never copied here; read them from the upstream repository.",
        "repos": {
            "mlx": "SceneWorks/minimax-h3-image-pe",
            "candle": "SceneWorks/minimax-h3-image-pe",
        },
        "folders": {
            "mlx": {"q8-full": "q8-full", "q4-full": "q4-full"},
            "candle": {"q8-full": "candle-q8-full", "q4-full": "candle-q4-full"},
        },
    },
}
FORMATS = {
    "mlx": "MLX group-wise affine quantized weights (group size 64), read by mlx-llm",
    "candle": "raw GGML blocks (Q8_0 / Q4_K body, Q8_0 head, embedding and decay inputs), read by candle-llm",
}
RECIPE_TEXT = {
    "q8-full": "every language-model matrix 8-bit, including `lm_head`, `embed_tokens` and every "
               "Gated DeltaNet `in_proj_a` / `in_proj_b`",
    "q4-full": "4-bit body; `lm_head`, `embed_tokens` and every Gated DeltaNet `in_proj_a` / "
               "`in_proj_b` 8-bit",
}


def source_model(key: str) -> dict:
    import tomllib

    matches = [m for m in tomllib.loads(MANIFEST.read_text(encoding="utf-8"))["models"] if m["key"] == key]
    if len(matches) != 1:
        raise SystemExit(f"expected one {key!r} model in {MANIFEST}, found {len(matches)}")
    return matches[0]


def snapshot_dir(hub: Path, model: dict) -> Path:
    return hub / ("models--" + model["repository"].replace("/", "--")) / "snapshots" / model["revision"]


def free_gb(path: Path) -> int:
    probe = path
    while not probe.exists():
        probe = probe.parent
    return shutil.disk_usage(probe).free // 10**9


def headroom(path: Path, need_gb: int) -> None:
    free = free_gb(path)
    print(f"{free} GB free under {path}; this stage writes {need_gb} GB and must leave {RESERVE_GB} GB")
    if free - need_gb < RESERVE_GB:
        raise SystemExit(
            f"::error::{free} GB free under {path}: the {need_gb} GB write would leave less than "
            f"{RESERVE_GB} GB. Free space on the volume (only this job's own scratch is ever removed) "
            "and dispatch again.")


def run(argv: list[str]) -> None:
    print("+ " + " ".join(argv), flush=True)
    subprocess.run(argv, check=True)


def append_env(name: str, value: str) -> None:
    target = os.environ.get("GITHUB_ENV")
    if target:
        with open(target, "a", encoding="utf-8") as handle:
            handle.write(f"{name}={value}\n")


def materialize(args) -> None:
    model = source_model(MODELS[args.model]["source_key"])
    hub = Path(args.hub)
    if not hub.is_absolute():
        raise SystemExit(f"--hub must be absolute, got {hub}")
    hub.mkdir(parents=True, exist_ok=True)
    snapshot = snapshot_dir(hub, model)
    if not snapshot.is_dir():
        # A resident snapshot only re-verifies; a fresh one writes its whole payload.
        headroom(hub, args.need_gb)
    for attempt in (1, 2, 3):
        result = subprocess.run([sys.executable, str(ROOT / "scripts/release/ensure_model_snapshot.py"),
                                 "--model", model["key"], "--snapshot", str(snapshot)])
        if result.returncode == 0:
            break
        if attempt == 3:
            raise SystemExit(f"{model['key']} materialization failed three times")
        print(f"::warning::{model['key']} materialization attempt {attempt} failed")
        time.sleep(30)
    append_env(args.env_name, str(snapshot))
    print(f"{args.env_name}={snapshot}")


def exe(bin_dir: Path, name: str) -> str:
    path = bin_dir / (name + (".exe" if os.name == "nt" else ""))
    if not path.is_file():
        raise SystemExit(f"converter {path} is not built")
    return str(path)


def provenance(args, folder: Path, recipe: str, source: dict) -> None:
    config = MODELS[args.model]
    sha = os.environ.get("GITHUB_SHA", "")
    record = {
        "source_repo": source["repository"],
        "source_revision": source["revision"],
        "source_format": "modelopt NVFP4, dequantized to bf16 on the CPU (modelopt_nvfp4_to_bf16)"
                         if config["nvfp4"] else "bf16",
        "producer": f"SceneWorks/inference {args.backend}-llm examples/prepare_snapshot",
        "producer_revision": sha,
        "recipe": recipe,
        "recipe_description": RECIPE_TEXT[recipe],
        "storage": FORMATS[args.backend],
        "vision_tower": "dense, as in the plain tiers",
        "mtp_head": "stored as in the plain tier of the same body width",
        "workflow_run": f"{os.environ.get('GITHUB_SERVER_URL', '')}/{os.environ.get('GITHUB_REPOSITORY', '')}"
                        f"/actions/runs/{os.environ.get('GITHUB_RUN_ID', '')}",
    }
    (folder / "SOURCE_REVISION.json").write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")
    (folder / "README.md").write_text(
        f"# `{folder.name}/` — fully quantized tier ({recipe})\n\n"
        f"Produced by SceneWorks from `{source['repository']}` at revision `{source['revision']}` "
        f"with `prepare_snapshot ... {recipe}` built from SceneWorks/inference `{sha}`.\n\n"
        f"- Recipe: {RECIPE_TEXT[recipe]}.\n"
        f"- Storage: {FORMATS[args.backend]}.\n"
        "- The vision tower stays dense; the MTP head is stored as in the plain tier.\n\n"
        "This folder is self-contained (config, tokenizer, chat template, weights). The plain tiers "
        "in this repository are unchanged. `SOURCE_REVISION.json` records the exact provenance.\n\n"
        f"License: {config['license']}\n",
        encoding="utf-8",
    )


def convert(args) -> None:
    config = MODELS[args.model]
    source = source_model(config["source_key"])
    env_name = "MINIMAX_PE_SNAPSHOT" if args.model == "expander" else "QWEN38_DESCRIBER_BF16_SNAPSHOT"
    src = Path(os.environ.get(env_name, ""))
    if not src.is_absolute() or not (src / "config.json").is_file():
        raise SystemExit(f"{env_name} must name the materialized source snapshot, got {src}")
    scratch = Path(args.scratch)
    bin_dir = Path(args.bin_dir)
    intermediate = None
    if config["nvfp4"]:
        intermediate = scratch / "bf16"
        shutil.rmtree(intermediate, ignore_errors=True)
        run([exe(bin_dir, "modelopt_nvfp4_to_bf16"), str(src), str(intermediate)])
        src = intermediate
    for recipe in RECIPES:
        out = scratch / config["folders"][args.backend][recipe]
        shutil.rmtree(out, ignore_errors=True)
        run([exe(bin_dir, "prepare_snapshot"), str(src), str(out), recipe])
        # Licensed upstream prompt files are never part of a tier (publish_hf_tiers also refuses them).
        shutil.rmtree(out / "prompt", ignore_errors=True)
        provenance(args, out, recipe, source)
        size = sum(p.stat().st_size for p in out.rglob("*") if p.is_file())
        print(f"{out.name}: {size} bytes in {sum(1 for p in out.rglob('*') if p.is_file())} files")
    if intermediate is not None:
        shutil.rmtree(intermediate)


def publish(args) -> None:
    config = MODELS[args.model]
    repo = config["repos"][args.backend]
    scratch = Path(args.scratch)
    reports = Path(args.reports)
    reports.mkdir(parents=True, exist_ok=True)
    report = reports / f"qwen38-full-tiers-{args.model}-{args.backend}.json"
    maps = []
    for recipe in RECIPES:
        folder = config["folders"][args.backend][recipe]
        maps += ["--map", f"{scratch / folder}={folder}"]
    env = dict(os.environ, HF_HUB_DISABLE_XET="1")
    argv = [sys.executable, str(ROOT / "scripts/ci/publish_hf_tiers.py"), "--repo", repo, *maps,
            "--message", f"SW-7202: {args.model} {args.backend} q8-full/q4-full tiers "
                         f"(inference {os.environ.get('GITHUB_SHA', '')})",
            "--report", str(report)]
    print("+ " + " ".join(argv), flush=True)
    subprocess.run(argv, check=True, env=env)
    published = json.loads(report.read_text(encoding="utf-8"))
    lines = [f"## {args.model} / {args.backend}: {repo}@{published['revision']}", ""]
    for prefix, size in published["prefixBytes"].items():
        files = [f["path"] for f in published["files"] if f["path"].startswith(prefix + "/")]
        lines.append(f"- `{prefix}/`: {size} bytes, {len(files)} files: {', '.join(files)}")
    text = "\n".join(lines) + "\n"
    print(text)
    summary = os.environ.get("GITHUB_STEP_SUMMARY")
    if summary:
        with open(summary, "a", encoding="utf-8") as handle:
            handle.write(text)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)
    m = sub.add_parser("materialize")
    m.add_argument("--model", choices=MODELS, required=True)
    m.add_argument("--hub", required=True)
    m.add_argument("--env-name", required=True, choices=["QWEN38_DESCRIBER_BF16_SNAPSHOT", "MINIMAX_PE_SNAPSHOT"])
    m.add_argument("--need-gb", type=int, required=True)
    h = sub.add_parser("headroom")
    h.add_argument("--path", required=True)
    h.add_argument("--need-gb", type=int, required=True)
    for name in ("convert", "publish"):
        p = sub.add_parser(name)
        p.add_argument("--model", choices=MODELS, required=True)
        p.add_argument("--backend", choices=["mlx", "candle"], required=True)
        p.add_argument("--scratch", required=True)
        if name == "convert":
            p.add_argument("--bin-dir", required=True)
        else:
            p.add_argument("--reports", required=True)
    args = parser.parse_args()
    if args.command == "materialize":
        materialize(args)
    elif args.command == "headroom":
        headroom(Path(args.path), args.need_gb)
    elif args.command == "convert":
        convert(args)
    else:
        publish(args)
    return 0


if __name__ == "__main__":
    sys.exit(main())
