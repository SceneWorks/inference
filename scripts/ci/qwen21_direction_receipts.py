"""Keep diagnostic completeness/provenance separate from terminal acceptance."""

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import re
import struct

from scripts.ci.qwen21_velocity_adapter import PROVENANCE, SOURCE

PROTOCOL = "09008e5a0ba6bc31dfb5f6f0ec95cb322741a7df3be6a9a09896896d29df709f"
STYLE_SHA = "165ef06aa4374084f5efec26756e1f01c1e41a8c0cec6f5057892d41e33ee8e0"
STYLE_PROVENANCE = {"trainingSource": "6cca130b55939e1223261a82b9eeaea876a9ba6b",
                    "trainingRun": 37127726908, "trainingSteps": 1000,
                    "donorSha256": STYLE_SHA, "donorBytes": 167849666}


def read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def validate_style(out, source, style_exit):
    if not re.fullmatch(r"[0-9a-f]{40}", source):
        raise ValueError("exact executing source required")
    style = out / "style-protocol"
    for filename in ("DIAGNOSTIC_ONLY.json", "style-direction.json"):
        receipt = read(style / filename)
        if (receipt.get("purpose") != "DIAGNOSTIC_ONLY" or receipt.get("acceptanceEvidence") is not False
                or receipt.get("retrain") is not False or receipt.get("sourceCandidate") != source
                or receipt.get("trainingProvenance") != STYLE_PROVENANCE or receipt.get("renderCount") != 6):
            raise ValueError("complete original style diagnostic provenance required")
    controls = read(style / "stage-controls.json")
    if (controls.get("kind") != "DIAGNOSTIC_ONLY" or controls.get("purpose") != "DIAGNOSTIC_ONLY" or controls.get("acceptanceEvidence") is not False
            or controls.get("sourceCandidate") != source or controls.get("renderCount") != 6
            or controls.get("donorSha256") != STYLE_SHA or controls.get("samplerJoined") is not True
            or controls.get("safetyChecksComplete") is not True or controls.get("foregroundRetired") is not True):
        raise ValueError("safety checked fully retired style stage required")
    run_id = os.environ.get("GITHUB_RUN_ID")
    if run_id is not None and (not re.fullmatch(r"[1-9][0-9]*", run_id) or controls.get("githubRunId") != run_id):
        raise ValueError("style controls must belong to the executing workflow run")
    if controls.get("sampleTrace") != "physical-allocator-samples.jsonl":
        raise ValueError("exact native physical sample trace required")
    trace_path = style / controls["sampleTrace"]
    if trace_path.resolve() != trace_path.absolute() or not trace_path.is_file():
        raise ValueError("physical trace must stay inside owned evidence")
    trace_bytes = trace_path.read_bytes()
    if hashlib.sha256(trace_bytes).hexdigest() != controls.get("sampleTraceSha256"):
        raise ValueError("physical trace hash mismatch")
    samples = [json.loads(line) for line in trace_bytes.splitlines() if line.strip()]
    if not samples or type(controls.get("sampleCount")) is not int or controls["sampleCount"] != len(samples):
        raise ValueError("complete physical sample count required")
    last = controls.get("startedUnixMillis")
    if type(last) is not int or last <= 0:
        raise ValueError("exact style start timestamp required")
    for sample in [*samples, controls.get("finalSafetySample", {})]:
        keys = ("unixMillis", "physFootprintBytes", "physicalCeilingBytes", "pressureLevel", "reclaimableBytes")
        if (any(type(sample.get(key)) is not int or sample[key] <= 0 for key in keys)
                or sample["pressureLevel"] != 1 or sample["physFootprintBytes"] > sample["physicalCeilingBytes"]
                or sample["unixMillis"] < last):
            raise ValueError("native pressure/available/physical/timestamp safety required")
        last = sample["unixMillis"]
    rows = receipt.get("renders", [])
    if len(rows) != 3 or [row.get("tier") for row in rows] != ["bf16", "q8", "q4"]:
        raise ValueError("three exact style tiers required")
    requests = []
    direction_failures = []
    for row in rows:
        request = row.get("request", {})
        if (row.get("mode") != "t2i" or row.get("adapter") != "mlx_t2i_1000_steps"
                or row.get("adapterSha256") != STYLE_SHA
                or row.get("requestProtocol") != "original-1000-step-style"
                or request.get("prompt") != "zxq style, a lighthouse on a rocky coast at dusk"
                or request.get("width") != 768 or request.get("height") != 768
                or request.get("steps") != 8 or request.get("seed") != 24163
                or request.get("conditioningCount") != 0 or request.get("count") != 1):
            raise ValueError("original fixed style request required")
        requests.append(request)
        for key in ("meanAbsDiff", "paletteDistanceBase", "paletteDistanceAdapted", "paletteDistanceGain"):
            value = row.get(key)
            if type(value) not in (int, float) or not math.isfinite(value):
                raise ValueError("complete finite style metric required")
        if row["meanAbsDiff"] < 2:
            direction_failures.append({"tier": row["tier"], "criterion": "movement", "actual": row["meanAbsDiff"], "floor": 2})
        if row["paletteDistanceGain"] < 1:
            direction_failures.append({"tier": row["tier"], "criterion": "palette_gain", "actual": row["paletteDistanceGain"], "floor": 1})
        for suffix in ("style_base", "style_mlx_t2i_1000_steps"):
            path = style / f"{row['tier']}_{suffix}.png"
            if path.is_symlink() or not path.is_file() or path.read_bytes()[:8] != b"\x89PNG\r\n\x1a\n":
                raise ValueError("six actual style PNG captures required")
    if any(request != requests[0] for request in requests[1:]):
        raise ValueError("style requests may not differ by tier")
    result = "direction_failed" if direction_failures else "passed"
    expected_exit = 101 if direction_failures else 0
    if (style_exit != expected_exit or controls.get("stageResult") != result
            or controls.get("directionFailures") != direction_failures):
        raise ValueError("only completed direction assertions may continue after failure")
    return {"kind": "DIAGNOSTIC_ONLY", "acceptanceEvidence": False, "accepted": False,
            "sourceCandidate": source, "styleExit": style_exit, "styleStageResult": result,
            "styleDirectionFailures": direction_failures, "styleRenderCount": 6}


def validate(out, source, manifest_path, style_exit):
    stage = validate_style(out, source, style_exit)

    velocity = out / "velocity-discriminator"
    receipt = read(velocity / "receipt.json")
    expected = {"kind": "DIAGNOSTIC_ONLY", "acceptanceEvidence": False, "accepted": False,
                "sourceCandidate": source, "sourceBase": SOURCE, "discriminatorProtocolSha256": PROTOCOL,
                "trainingProvenance": PROVENANCE, "forwardCount": 16, "stateCount": 4, "repeatCount": 2,
                "adapterStrength": 1, "sigma": 0.5, "arithmeticBoundVerdict": "UNPROVEN_RELAXED_NAX_PRECISION",
                "inputManifestSha256": hashlib.sha256(manifest_path.read_bytes()).hexdigest()}
    if (receipt.get("acceptanceEvidence") is not False or receipt.get("accepted") is not False
            or any(receipt.get(key) != value for key, value in expected.items())):
        raise ValueError("complete attributable diagnostic velocity receipt required")
    peak = receipt.get("cpuCachePeakBytes")
    if type(peak) is not int or not 0 <= peak <= 67108864:
        raise ValueError("bounded CPU cache receipt required")
    if len(receipt.get("vectors", [])) != 16 or len(receipt.get("states", [])) != 4:
        raise ValueError("all16 vectors and four states required")
    cells = set()
    for vector in receipt["vectors"]:
        state, repeat, adapted = vector.get("state"), vector.get("repeat"), vector.get("adapted")
        if type(state) is not int or state not in range(4) or type(repeat) is not int or repeat not in range(2) or type(adapted) is not bool:
            raise ValueError("exact state/repeat/base-or-adapted vector identity required")
        cell = (state, repeat, adapted)
        if cell in cells:
            raise ValueError("duplicate velocity cell")
        cells.add(cell)
        filename = f"velocities/state-{state}-repeat-{repeat}-{'adapted' if adapted else 'base'}.f32"
        if (vector.get("file") != filename or vector.get("dtype") != "Float32"
                or vector.get("shape") != [1, 2304, 64] or vector.get("elements") != 147456
                or vector.get("bytes") != 589824):
            raise ValueError("full fixed velocity shape/dtype/path required")
        path = velocity / filename
        if path.resolve() != path.absolute() or not path.is_file():
            raise ValueError("velocity file must stay inside owned evidence")
        raw = path.read_bytes()
        if (len(raw) != 589824 or hashlib.sha256(raw).hexdigest() != vector.get("sha256")
                or not all(math.isfinite(x[0]) for x in struct.iter_unpack("<f", raw))):
            raise ValueError("complete finite hash-pinned velocity bytes required")
    expected_states = [("denseBF16", "denseBF16"), ("Q4", "Q4"), ("denseBF16", "Q4"), ("Q4", "denseBF16")]
    for index, state in enumerate(receipt["states"]):
        if (state.get("state") != index or (state.get("conditioning"), state.get("dit")) != expected_states[index]
                or [pair.get("repeat") for pair in state.get("pairs", [])] != [0, 1]):
            raise ValueError("fixed four states and two score pairs required")
        for pair in state["pairs"]:
            for key in ("baseError", "adaptedError", "learnedGain", "projection", "deltaNorm2", "cpuIdentityResidual", "cpuIdentityBound"):
                value = pair.get(key)
                if type(value) not in (int, float) or not math.isfinite(value):
                    raise ValueError("complete finite full-vector score required")
    # Native test owns NAX observations and all admission/control checks. This validator never
    # converts negative/positive diagnostic gains into acceptance or an arithmetic PASS.
    velocity_exit = (out / "direction-velocity.exit-code").read_text(encoding="utf-8").strip()
    if velocity_exit != "0":
        raise ValueError("velocity test must complete without safety/control failure")
    return {**stage, "velocityExit": 0, "velocityForwardCount": 16,
            "arithmeticBoundVerdict": "UNPROVEN_RELAXED_NAX_PRECISION"}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--source", required=True)
    parser.add_argument("--velocity-manifest", type=Path)
    parser.add_argument("--style-only", action="store_true")
    parser.add_argument("--style-exit", type=int, required=True)
    args = parser.parse_args()
    if args.style_only:
        result = validate_style(args.out, args.source, args.style_exit)
    else:
        if args.velocity_manifest is None:
            parser.error("--velocity-manifest is required for the complete phase")
        result = validate(args.out, args.source, args.velocity_manifest, args.style_exit)
    (args.out / ("direction-style-stage.json" if args.style_only else "direction-protocol-summary.json")).write_text(
        json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    main()
