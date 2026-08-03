#!/usr/bin/env python3
"""Generate and verify the corrected sc-14542 sampler math oracle.

The artifact is intentionally small and weight-free. Existing committed P0 safetensors cover all
six real DiTs and Pingpong trajectories; this oracle isolates the three additional solver formulas,
typed per-example schedules, corrected partial strength, and eager Pingpong draw ordering.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from pathlib import Path

try:
    from sa3_reference import validate_upstream_checkout
except ModuleNotFoundError:
    from .sa3_reference import validate_upstream_checkout

ROOT = Path(__file__).resolve().parents[2]
STORY = "sc-14542"
UPSTREAM_COMMIT = "124e8a799f57a1f665495ecb72e547d0a62867f1"
SOURCES = {
    "stable_audio_3/inference/sampling.py":
        "76e96bc3a0c1fb346d13038f09a227ec2f9c7f2e56fd64623e8bbe6ff722bf55",
    "stable_audio_3/inference/distribution_shift.py":
        "18264eb8c705348a4b040cbd116c94b04b6235d5e52968ecf76658e375a98f13",
    "optimized/tflite/models/defs/tflite_pipeline.py":
        "4dd39154ab80dea4fee418116bea76ba976bf7f00325bae415c26d00fc55df12",
    "stable_audio_3/data/utils.py":
        "72740fc82eb9a2a54083ab9153769c63ab60ef29173d2d5f019a1dbfede757fe",
    "stable_audio_3/model.py":
        "8f3a18f1b8f02ffb1ce97f8e64dcb319eb2a6a864d8c405f89d40f6933f3a701",
    "stable_audio_3/models/diffusion.py":
        "b0eab27f7965b4a21a34bf009c932128217524c09cddbfd601bffe866e4b6732",
    "stable_audio_3/models/dit.py":
        "4b28dcb74937cdcab9ab5f745221178304f998f53aa60d2ee9e74b8fa76fd4fb",
}
DEFAULT_OUTPUT = ROOT / "docs/migration/sa3-sampler-reference"
P0_ROOT = ROOT / "docs/migration/sa3-reference"
P0_MANIFEST_SHA256 = "2b8180b8f42aaf4324bbe63edf071b9613b8257e16e271ca859ab3d20da6eb74"
ARTIFACT_SHA256 = "20b4047008f3819b79bebfebe5bcdc1d7b19e89c3e6f8394fed2d6b788bb27f6"
P0_KEYS = (
    "small-music",
    "small-sfx",
    "small-music-base",
    "small-sfx-base",
    "medium",
    "medium-base",
)


class InvalidReference(ValueError):
    pass


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def validate_upstream(upstream: Path) -> None:
    validate_upstream_checkout(
        upstream,
        UPSTREAM_COMMIT,
        error_type=InvalidReference,
        check_clean=False,
    )
    for relative, expected in SOURCES.items():
        if sha256(upstream / relative) != expected:
            raise InvalidReference(f"upstream source mismatch: {relative}")


def p0_reference() -> dict:
    path = P0_ROOT / "manifest.json"
    if sha256(path) != P0_MANIFEST_SHA256:
        raise InvalidReference("P0 manifest payload mismatch")
    manifest = json.loads(path.read_text(encoding="utf-8"))
    snapshots = {entry["key"]: entry for entry in manifest["snapshots"]}
    result = {
        "manifest": "docs/migration/sa3-reference/manifest.json",
        "manifestSha256": sha256(path),
        "cases": {},
    }
    for key in P0_KEYS:
        snapshot = snapshots[key]
        artifact_record = manifest["artifacts"][key]
        artifact_path = P0_ROOT / artifact_record["file"]
        if sha256(artifact_path) != artifact_record["sha256"]:
            raise InvalidReference(f"P0 artifact payload mismatch: {key}")
        result["cases"][key] = {
            "artifact": artifact_record["file"],
            "artifactSha256": artifact_record["sha256"],
            "snapshotRevision": snapshot["revision"],
            "modelConfigSha256": snapshot["modelConfigSha256"],
        }
    return result


def logsnr_schedule(steps: int, strength: float, length: int, rate: float) -> list[float]:
    start = -6.2 - rate * math.log2(max(length, 1) / 2000)
    result = []
    for index in range(steps + 1):
        t = 1.0 - index / steps
        if t <= 0:
            shifted = 0.0
        elif t >= 1:
            shifted = 1.0
        else:
            logsnr = 2.0 - t * (2.0 - start)
            shifted = 1.0 / (1.0 + math.exp(logsnr))
        result.append(shifted * strength)
    result[0] = strength
    result[-1] = 0.0
    return result


def field(x: float, t: float) -> float:
    return 0.25 * x + 0.5 * t - 0.125


def euler(x: float, schedule: list[float]) -> list[dict[str, float]]:
    trace = []
    for current, nxt in zip(schedule, schedule[1:]):
        velocity = field(x, current)
        trace.append({"x": x, "t": current, "denoised": x - current * velocity})
        x += (nxt - current) * velocity
    trace.append({"final": x})
    return trace


def rk4(x: float, schedule: list[float]) -> list[dict[str, float]]:
    trace = []
    for current, nxt in zip(schedule, schedule[1:]):
        dt = nxt - current
        k1 = field(x, current)
        trace.append({"x": x, "t": current, "denoised": x - current * k1})
        k2 = field(x + dt * k1 / 2, current + dt / 2)
        k3 = field(x + dt * k2 / 2, current + dt / 2)
        k4 = field(x + dt * k3, max(nxt, 1e-5))
        x += dt * (k1 + 2 * k2 + 2 * k3 + k4) / 6
    trace.append({"final": x})
    return trace


def dpmpp(x: float, schedule: list[float]) -> list[dict[str, float]]:
    trace = []
    old = None
    for index, (current, nxt) in enumerate(zip(schedule, schedule[1:])):
        velocity = field(x, current)
        clean = x - current * velocity
        trace.append({"x": x, "t": current, "denoised": clean})
        coefficient = (nxt - current) / (max(1 - nxt, 1e-10) * max(current, 1e-10))
        if old is None or nxt == 0:
            corrected = clean
        else:
            previous = schedule[index - 1]
            logsnr = lambda value: math.log(max(1 - value, 1e-10) / max(value, 1e-10))
            h = logsnr(nxt) - logsnr(current)
            h_last = logsnr(current) - logsnr(previous)
            ratio = h_last / h
            corrected = (1 + 1 / (2 * ratio)) * clean - old / (2 * ratio)
        x = nxt / max(current, 1e-10) * x - (1 - nxt) * coefficient * corrected
        old = clean
    trace.append({"final": x})
    return trace


def pingpong(x: float, schedule: list[float], draws: list[float]) -> list[dict[str, float]]:
    trace = []
    for index, (current, nxt) in enumerate(zip(schedule, schedule[1:])):
        velocity = field(x, current)
        clean = x - current * velocity
        trace.append({"x": x, "t": current, "denoised": clean, "draw": draws[index]})
        x = (1 - nxt) * clean + nxt * draws[index]
    trace.append({"final": x, "draws": len(draws)})
    return trace


def _tensor_trace(callback_states, final) -> list[dict]:
    result = []
    for state in callback_states:
        result.append({
            "x": state["x"].detach().cpu().float().reshape(-1).tolist(),
            "t": state["t"].detach().cpu().float().reshape(-1).tolist(),
            "denoised": state["denoised"].detach().cpu().float().reshape(-1).tolist(),
        })
    result.append({"final": final.detach().cpu().float().reshape(-1).tolist()})
    return result


def frozen_torch_trajectories(upstream: Path) -> dict:
    sys.path.insert(0, str(upstream))
    try:
        import torch
        from stable_audio_3.inference.sampling import (
            sample_discrete_euler,
            sample_flow_dpmpp,
            sample_rk4,
        )
    finally:
        sys.path.pop(0)
    schedule = torch.tensor(logsnr_schedule(4, 1.0, 16, 0.0), dtype=torch.float32)
    initial = torch.tensor([[[0.75]]], dtype=torch.float32)

    def model(x, t, **_kwargs):
        return 0.25 * x + 0.5 * t[:, None, None] - 0.125

    def run(function):
        states = []
        final = function(
            model,
            initial.clone(),
            schedule,
            callback=lambda state: states.append({
                key: value.detach().clone() if hasattr(value, "detach") else torch.as_tensor(value)
                for key, value in state.items()
                if key in {"x", "t", "denoised"}
            }),
            disable_tqdm=True,
        )
        return _tensor_trace(states, final)

    return {
        "euler": run(sample_discrete_euler),
        "dpmpp": run(sample_flow_dpmpp),
        "rk4FrozenScalar": run(sample_rk4),
        "runtime": {"python": sys.version.split()[0], "torch": torch.__version__},
    }


def artifact(upstream: Path) -> dict:
    shared = logsnr_schedule(4, 1.0, 16, 0.0)
    per_example = [
        logsnr_schedule(4, 1.0, 68, 1.0),
        logsnr_schedule(4, 1.0, 323, 1.0),
    ]
    partial = logsnr_schedule(8, 0.5, 16, 0.0)
    draws = [0.5, -1.25, 2.0, 99.0]
    return {
        "story": STORY,
        "upstreamCommit": UPSTREAM_COMMIT,
        "sourceSha256": SOURCES,
        "contract": {
            "field": "v(x,t)=0.25*x+0.5*t-0.125",
            "scheduleShape": ["shared", "per-example"],
            "partialStrengthOrder": "shift-normalized-then-scale",
            "rk4": "corrected-batched-with-scalar-frozen-formula",
            "dpmpp": "rectified-flow-not-VE",
            "pingpongTerminalDraw": True,
        },
        "schedules": {
            "shared": shared,
            "perExample": per_example,
            "partialStrength": partial,
        },
        "trajectories": {
            "eulerIndependent": euler(0.75, shared),
            "rk4": [rk4(0.75, row) for row in per_example],
            "dpmppIndependent": dpmpp(0.75, shared),
            "pingpong": pingpong(0.75, shared, draws),
            "frozenTorch": frozen_torch_trajectories(upstream),
        },
    }


def canonical(value: dict) -> bytes:
    return (json.dumps(value, sort_keys=True, separators=(",", ":")) + "\n").encode()


def generate(upstream: Path, output: Path) -> None:
    validate_upstream(upstream)
    output.mkdir(parents=True, exist_ok=True)
    payload = artifact(upstream)
    artifact_path = output / "sampler.json"
    artifact_path.write_bytes(canonical(payload))
    manifest = {
        "story": STORY,
        "upstreamCommit": UPSTREAM_COMMIT,
        "sourceSha256": SOURCES,
        "p0Reference": p0_reference(),
        "artifact": artifact_path.name,
        "artifactSha256": sha256(artifact_path),
    }
    (output / "manifest.json").write_bytes(canonical(manifest))


def verify(output: Path) -> None:
    manifest = json.loads((output / "manifest.json").read_text(encoding="utf-8"))
    if manifest.get("story") != STORY or manifest.get("upstreamCommit") != UPSTREAM_COMMIT:
        raise InvalidReference("manifest identity mismatch")
    if manifest.get("sourceSha256") != SOURCES:
        raise InvalidReference("source hash lock mismatch")
    if manifest.get("p0Reference") != p0_reference():
        raise InvalidReference("P0 manifest, artifact, snapshot, or config provenance mismatch")
    artifact_path = output / manifest.get("artifact", "")
    actual_artifact_sha256 = sha256(artifact_path)
    if (
        actual_artifact_sha256 != ARTIFACT_SHA256
        or manifest.get("artifactSha256") != ARTIFACT_SHA256
    ):
        raise InvalidReference("artifact payload mismatch")
    payload = json.loads(artifact_path.read_text(encoding="utf-8"))
    if (
        payload.get("story") != STORY
        or payload.get("upstreamCommit") != UPSTREAM_COMMIT
        or payload.get("sourceSha256") != SOURCES
        or payload.get("contract") != {
            "field": "v(x,t)=0.25*x+0.5*t-0.125",
            "scheduleShape": ["shared", "per-example"],
            "partialStrengthOrder": "shift-normalized-then-scale",
            "rk4": "corrected-batched-with-scalar-frozen-formula",
            "dpmpp": "rectified-flow-not-VE",
            "pingpongTerminalDraw": True,
        }
    ):
        raise InvalidReference("artifact identity, source provenance, or contract mismatch")
    frozen = payload.get("trajectories", {}).get("frozenTorch", {})
    if set(frozen) != {"euler", "dpmpp", "rk4FrozenScalar", "runtime"}:
        raise InvalidReference("frozen Torch solver trajectories are missing")
    schedules = payload["schedules"]
    rows = [schedules["shared"], schedules["partialStrength"], *schedules["perExample"]]
    if any(
        right > left
        for row in rows
        for left, right in zip(row, row[1:])
    ):
        raise InvalidReference("committed schedule is non-monotonic")
    if payload["trajectories"]["pingpong"][-1]["draws"] != 4:
        raise InvalidReference("terminal Pingpong draw is missing")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--upstream", type=Path)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--generate", action="store_true")
    args = parser.parse_args()
    try:
        if args.generate:
            if args.upstream is None:
                parser.error("--generate requires --upstream")
            generate(args.upstream, args.output)
        verify(args.output)
        print("Stable Audio 3 sampler reference verified")
    except (InvalidReference, OSError, ValueError, KeyError) as error:
        print(f"SA3 sampler reference error: {error}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
