#!/usr/bin/env python3
"""Dispatch-only, exact Rust fixture captures. Phone anchors remain campaign.py's job."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import re
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[2]
REGISTRY = tuple(json.loads(Path(__file__).with_name("ios-selected-captures.json").read_text(encoding="utf-8")))
FAMILIES = ("iris", "anima", "z-image", "sdxl", "sana", "pid", "qwen-image", "shared")
FAMILY_PACKAGES = {**{family: "mlx-gen-" + family for family in FAMILIES if family != "shared"},
                   "shared": "mlx-gen"}


def identity(row: dict) -> str:
    return row["package"] + "/" + row["fqn"]


def select(selection: str) -> tuple[dict, ...]:
    choices = ("all", *FAMILIES, *(identity(row) for row in REGISTRY))
    if selection not in choices:
        raise ValueError("selection must name a fixed family or registered package/FQN")
    rows = tuple(row for row in REGISTRY if selection == "all"
                 or row["package"] == FAMILY_PACKAGES.get(selection) or identity(row) == selection)
    if not rows or len({identity(row) for row in rows}) != len(rows):
        raise ValueError("empty or duplicate capture selection")
    return rows


def selection_from_binding(binding: str) -> str:
    value = json.loads(binding.strip() or "{}")
    if not isinstance(value, dict):
        raise ValueError("capture binding must be a JSON object")
    selection = value.get("selection", "iris")
    if not isinstance(selection, str):
        raise ValueError("provided capture selection must be a string")
    select(selection)
    return selection


def require_dispatch(env: dict[str, str]) -> None:
    for key, expected in (("GITHUB_EVENT_NAME", "workflow_dispatch"),
                          ("RUNNER_NAME", "nax-macos-2"), ("RUNNER_OS", "macOS"),
                          ("RUNNER_ARCH", "ARM64")):
        if env.get(key) != expected:
            raise ValueError(f"{key} must be {expected}; no local/shared-dev-Mac capture")


def command(row: dict, discover: bool) -> list[str]:
    # No command, target, feature, test pattern or manifest can come from dispatch input.
    if row not in REGISTRY or row["target"] != "lib":
        raise ValueError("unregistered capture target")
    return ["cargo", "test", "--locked", "--no-default-features", "-p", row["package"],
            "--lib", row["fqn"], "--", "--exact", "--ignored",
            *( ["--list"] if discover else ["--nocapture", "--test-threads=1"] )]


def verify_discovery(row: dict, text: str) -> None:
    discovered = re.findall(r"^(.+): test$", text, re.M)
    if discovered != [row["fqn"]]:
        raise ValueError(f"{identity(row)}: expected exactly one ignored test, got {discovered}")
    if not re.search(r"^1 test, 0 benchmarks$", text, re.M):
        raise ValueError("ignored discovery did not report exactly one test")


def verify_execution(row: dict, text: str) -> None:
    results = re.findall(r"^test result: ok\. (\d+) passed; (\d+) failed; (\d+) ignored;", text, re.M)
    # These captures all delegate to an exact CPU child. Both parent and child must execute one.
    if len(results) != 2 or any(result != ("1", "0", "0") for result in results):
        raise ValueError(f"{identity(row)}: expected one completed parent AND child: {results}")
    marker = "IOS_CAPTURE_BODY_COMPLETED " + row["fqn"]
    if text.splitlines().count(marker) != 1:
        raise ValueError(f"{identity(row)}: missing/duplicate actual capture-body completion")


def run(selection: str) -> int:
    require_dispatch(os.environ)
    rows = select(selection)
    completed = 0
    output = Path(os.environ["RUNNER_TEMP"]) / "ios-selected-captures"
    output.mkdir(parents=True, exist_ok=True)
    receipt = {"selected": len(rows), "completed": 0, "captures": []}
    try:
        for index, row in enumerate(rows):
            for discover in (True, False):
                proc = subprocess.run(command(row, discover), cwd=ROOT,
                                      stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
                                      encoding="utf-8")
                text = proc.stdout
                log = output / f"{index:03d}-{'discovery' if discover else 'body'}.log"
                log.write_text(text, encoding="utf-8")
                print(text, end="", flush=True)
                if proc.returncode:
                    return proc.returncode
                (verify_discovery if discover else verify_execution)(row, text)
            completed += 1
            receipt["completed"] = completed
            receipt["captures"].append(identity(row))
        if completed != len(rows):
            raise ValueError("selected/completed capture count mismatch")
        return 0
    finally:
        (output / "receipt.json").write_text(json.dumps(receipt, indent=2) + "\n", encoding="utf-8")
        print(f"IOS_CAPTURE_COUNTS selected={len(rows)} completed={completed}", flush=True)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--select", choices=("all", *FAMILIES, *(identity(row) for row in REGISTRY)))
    source.add_argument("--binding", help="capture-profile JSON; omitted selection defaults to iris")
    parser.add_argument("--list", action="store_true", help="list fixed bindings without running")
    args = parser.parse_args()
    try:
        selection = args.select if args.select is not None else selection_from_binding(args.binding)
        if args.list:
            print(json.dumps(select(selection), indent=2))
            return 0
        return run(selection)
    except (ValueError, OSError) as error:
        print(f"capture prerequisite/validation failure: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
