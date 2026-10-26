#!/usr/bin/env python3
"""Create and retire only the simulator owned by one manual SANA workflow attempt."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import re
import subprocess

UDID = re.compile(r"[0-9A-Fa-f]{8}(?:-[0-9A-Fa-f]{4}){3}-[0-9A-Fa-f]{12}\Z")


def simctl(*args: str) -> str:
    command = subprocess.run(("xcrun", "simctl", *args), check=True, capture_output=True,
                             text=True, encoding="utf-8", timeout=60)
    if len(command.stdout) > 1024**2 or len(command.stderr) > 1024**2:
        raise ValueError("simctl output exceeded 1 MiB")
    return command.stdout.strip()


def devices() -> list[tuple[str, dict]]:
    document = json.loads(simctl("list", "devices", "available", "--json"))
    return [(runtime, device) for runtime, rows in document["devices"].items() for device in rows]


def select_template(selector: str, output: Path, run_id: str, attempt: str, runner: str) -> dict:
    if not re.fullmatch(r"iPhone [A-Za-z0-9 ]{1,60}", selector):
        raise ValueError("exact iPhone simulator model name required")
    if output.exists() or not output.parent.is_dir():
        raise ValueError("fresh simulator template receipt path required")
    matches = []
    for runtime, item in devices():
        version = re.fullmatch(r"com\.apple\.CoreSimulator\.SimRuntime\.iOS-(\d+)-(\d+)", runtime)
        if item["name"] == selector and item["isAvailable"] and version:
            matches.append((tuple(map(int, version.groups())), runtime, item))
    if not matches:
        raise ValueError("requested iPhone simulator model has no available iOS runtime on this runner")
    _, runtime, item = max(matches, key=lambda row: row[0])
    receipt = {"template_udid": item["udid"], "template_name": item["name"],
               "runtime": runtime, "device_type": item["deviceTypeIdentifier"],
               "run_id": run_id, "attempt": attempt, "runner": runner}
    with output.open("x", encoding="utf-8") as stream:
        stream.write(json.dumps(receipt, sort_keys=True) + "\n")
    return receipt


def create(selected: Path, output: Path, run_id: str, attempt: str,
           github_env: Path, runner: str) -> dict:
    if output.exists() or not output.parent.is_dir():
        raise ValueError("fresh owned-simulator receipt path required")
    template = json.loads(selected.read_text())
    if (template.get("run_id") != run_id or template.get("attempt") != attempt
            or template.get("runner") != runner or not UDID.fullmatch(template.get("template_udid", ""))):
        raise ValueError("simulator selection is not bound to this runner and attempt")
    matches = [(runtime, item) for runtime, item in devices()
               if item["udid"].casefold() == template["template_udid"].casefold()
               and item["isAvailable"] and runtime == template["runtime"]
               and item["deviceTypeIdentifier"] == template["device_type"]]
    if len(matches) != 1:
        raise ValueError("selected iPhone simulator template changed before create")
    runtime, item = matches[0]
    device_type = item["deviceTypeIdentifier"]
    name = f"SceneWorksSana-{run_id}-{attempt}"
    owned = simctl("create", name, device_type, runtime)
    if not UDID.fullmatch(owned) or owned.casefold() == template["template_udid"].casefold():
        raise ValueError("simctl did not return a distinct owned simulator UUID")
    receipt = {"template_udid": template["template_udid"], "template_name": item["name"],
               "runtime": runtime, "device_type": device_type, "owned_udid": owned,
               "owned_name": name, "run_id": run_id, "attempt": attempt, "runner": runner}
    with output.open("x", encoding="utf-8") as stream:
        stream.write(json.dumps(receipt, sort_keys=True) + "\n")
    with github_env.open("a", encoding="utf-8") as stream:
        stream.write("SIMULATOR_UDID=" + owned + "\n")
    return receipt


def cleanup(receipt_path: Path, output: Path, run_id: str, attempt: str, runner: str) -> dict:
    if not receipt_path.is_file():
        result = {"status": "not-created"}
    else:
        receipt = json.loads(receipt_path.read_text())
        owned = receipt.get("owned_udid", "")
        name = f"SceneWorksSana-{run_id}-{attempt}"
        if (not UDID.fullmatch(owned) or receipt.get("run_id") != run_id
                or receipt.get("attempt") != attempt or receipt.get("owned_name") != name
                or receipt.get("runner") != runner):
            raise ValueError("owned simulator receipt does not match this workflow attempt")
        matches = [(runtime, item) for runtime, item in devices()
                   if item["udid"].casefold() == owned.casefold()]
        if len(matches) != 1 or matches[0][1]["name"] != name or matches[0][0] != receipt["runtime"]:
            raise ValueError("owned simulator UUID/name/runtime changed; refusing cleanup")
        if matches[0][1]["state"] != "Shutdown":
            simctl("shutdown", owned)
        simctl("delete", owned)
        result = {"status": "owned-simulator-deleted", "owned_udid": owned,
                  "owned_name": name, "runtime": receipt["runtime"]}
    output.write_text(json.dumps(result, sort_keys=True) + "\n")
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("phase", choices=("select", "create", "cleanup"))
    parser.add_argument("--receipt", required=True, type=Path)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--attempt", required=True)
    parser.add_argument("--selector")
    parser.add_argument("--selected", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    runner = os.environ.get("RUNNER_NAME", "")
    if (os.environ.get("GITHUB_EVENT_NAME") != "workflow_dispatch"
            or runner not in ("nax-macos", "nax-macos-2")
            or not args.run_id.isdecimal() or not args.attempt.isdecimal()):
        raise ValueError("simulator ownership is restricted to manual Mac dispatches")
    if args.phase == "select":
        if not args.selector:
            raise ValueError("select needs exact iPhone simulator model name")
        print(select_template(args.selector, args.receipt, args.run_id, args.attempt, runner))
    elif args.phase == "create":
        github_env = os.environ.get("GITHUB_ENV")
        if args.selected is None or not github_env:
            raise ValueError("create needs selected template and Actions environment file")
        print(create(args.selected, args.receipt, args.run_id, args.attempt, Path(github_env), runner))
    else:
        if args.output is None:
            raise ValueError("cleanup needs a result path")
        print(cleanup(args.receipt, args.output, args.run_id, args.attempt, runner))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
