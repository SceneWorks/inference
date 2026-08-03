#!/usr/bin/env python3
"""Generate/verify the real-weight sc-14542 CFG/APG sampler fixture."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from pathlib import Path

try:
    from sa3_reference import (
        InvalidReference,
        SPEC_BY_KEY,
        UPSTREAM_COMMIT,
        _localize_t5,
        _torch_modules,
        load_snapshot_lock,
        sha256_file,
        snapshot_revision,
    )
except ModuleNotFoundError:
    from .sa3_reference import (
        InvalidReference,
        SPEC_BY_KEY,
        UPSTREAM_COMMIT,
        _localize_t5,
        _torch_modules,
        load_snapshot_lock,
        sha256_file,
        snapshot_revision,
    )

ROOT = Path(__file__).resolve().parents[2]
STORY = "sc-14542"
DEFAULT_OUTPUT = ROOT / "docs/migration/sa3-sampler-reference"
SNAPSHOT_LOCK_PATH = ROOT / "docs/migration/sa3-reference/snapshot-files.json"
SNAPSHOT_LOCK_SHA256 = "58117a6dc85b3f7e84da88385a849a2d79c56534be06fef61e50fcd084758873"
ARTIFACT = "guidance.safetensors"
ARTIFACT_SHA256 = "2cd9225ee7a1e132eafd5f0dc0d12afb0e2a6e4879bb551f085feae4aca540f5"
P0_MANIFEST_SHA256 = "2b8180b8f42aaf4324bbe63edf071b9613b8257e16e271ca859ab3d20da6eb74"
CASES = ("small-music", "small-music-base")
P0_FILES = {
    "small-music": "small-music-reference.safetensors",
    "small-music-base": "small-music-base-reference.safetensors",
}
SOURCE_SHA256 = {
    "stable_audio_3/inference/sampling.py":
        "76e96bc3a0c1fb346d13038f09a227ec2f9c7f2e56fd64623e8bbe6ff722bf55",
    "stable_audio_3/data/utils.py":
        "72740fc82eb9a2a54083ab9153769c63ab60ef29173d2d5f019a1dbfede757fe",
    "stable_audio_3/model.py":
        "8f3a18f1b8f02ffb1ce97f8e64dcb319eb2a6a864d8c405f89d40f6933f3a701",
    "stable_audio_3/models/diffusion.py":
        "b0eab27f7965b4a21a34bf009c932128217524c09cddbfd601bffe866e4b6732",
    "stable_audio_3/models/dit.py":
        "4b28dcb74937cdcab9ab5f745221178304f998f53aa60d2ee9e74b8fa76fd4fb",
}
VARIANTS = {
    "vanilla": {"cfg_scale": 2.5, "apg_scale": 0.0},
    "apg": {"cfg_scale": 2.5, "apg_scale": 1.0},
    "blended_rescaled": {
        "cfg_scale": 2.5,
        "apg_scale": 0.5,
        "cfg_norm_threshold": 0.25,
        "scale_phi": 0.3,
    },
}
EXPECTED_RUNTIME = {
    "python": "3.12.13",
    "torch": "2.7.1",
    "torchaudio": "2.7.1",
    "transformers": "5.8.0",
}
EXPECTED_INPUTS = {
    "timestep": 0.5,
    "schedule": [0.5, 0.0],
    "paddingValid": 12,
    "paddingLength": 16,
    "negativeConditioning": "zero-cross-context",
    "variants": VARIANTS,
}
EXPECTED_OBJECTIVES = {
    "small-music": "rf_denoiser",
    "small-music-base": "rectified_flow",
}


def _validate_upstream(path: Path) -> None:
    commit = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=path,
        check=True,
        capture_output=True,
        text=True,
        encoding="utf-8",
    ).stdout.strip()
    if commit != UPSTREAM_COMMIT:
        raise InvalidReference("upstream checkout revision mismatch")
    for relative, expected in SOURCE_SHA256.items():
        if sha256_file(path / relative) != expected:
            raise InvalidReference(f"upstream source mismatch: {relative}")


def _records(path: Path) -> dict:
    data = path.read_bytes()
    length = int.from_bytes(data[:8], "little")
    header = json.loads(data[8:8 + length])
    start = 8 + length
    result = {}
    for key, record in header.items():
        if key == "__metadata__":
            continue
        left, right = record["data_offsets"]
        result[key] = {
            "dtype": record["dtype"],
            "shape": record["shape"],
            "sha256": hashlib.sha256(data[start + left:start + right]).hexdigest(),
        }
    return result


def generate(upstream: Path, snapshots: dict[str, Path], output: Path) -> None:
    _validate_upstream(upstream)
    torch, save_file, _sample_diffusion, loaders, versions = _torch_modules(upstream)
    _load_autoencoder, load_diffusion_cond = loaders
    if sha256_file(SNAPSHOT_LOCK_PATH) != SNAPSHOT_LOCK_SHA256:
        raise InvalidReference("snapshot file lock payload mismatch")
    snapshot_lock = load_snapshot_lock(SNAPSHOT_LOCK_PATH)["snapshots"]
    tensors = {}
    provenance = {}
    for key in CASES:
        snapshot = snapshots[key]
        spec = SPEC_BY_KEY[key]
        if snapshot_revision(snapshot) != spec.revision:
            raise InvalidReference(f"{key} snapshot revision mismatch")
        for name in ("model_config.json", "model.safetensors"):
            if sha256_file(snapshot / name) != snapshot_lock[key]["files"][name]["sha256"]:
                raise InvalidReference(f"{key} {name} payload mismatch")
        p0 = ROOT / "docs/migration/sa3-reference" / P0_FILES[key]
        if not p0.is_file():
            raise InvalidReference(f"{key} P0 artifact missing")
        config = json.loads((snapshot / "model_config.json").read_text(encoding="utf-8"))
        wrapper = load_diffusion_cond(
            _localize_t5(config, snapshot),
            str(snapshot / "model.safetensors"),
            device="cpu",
            model_half=False,
        )
        dit = wrapper.model.model.eval()
        from safetensors.torch import load_file
        fixture = load_file(str(p0), device="cpu")
        initial = fixture["sampler_initial_noise"].float()
        prompt = fixture["t5_projected_padded"].float()
        timestep = torch.tensor([0.5], dtype=torch.float32)
        seconds = torch.tensor([0.25], dtype=torch.float32)
        number = wrapper.conditioner.conditioners["seconds_total"]
        normalized = seconds.clamp(number.min_val, number.max_val)
        normalized = (normalized - number.min_val) / (number.max_val - number.min_val)
        number_features = number.embedder.embedding[0](normalized)
        number_embedding = number.embedder.embedding[1](number_features)
        raw_context = torch.cat([prompt, number_embedding.unsqueeze(1)], dim=1)
        local = torch.zeros((1, 257, 16), dtype=torch.float32)
        padding = torch.zeros((1, 16), dtype=torch.bool)
        padding[:, :12] = True
        with torch.inference_mode():
            for variant, kwargs in VARIANTS.items():
                velocity = dit(
                    initial,
                    timestep,
                    cross_attn_cond=raw_context,
                    global_embed=number_embedding,
                    local_add_cond=local,
                    padding_mask=padding,
                    **kwargs,
                ).float()
                tensors[f"{key}.{variant}.velocity"] = velocity
                tensors[f"{key}.{variant}.final"] = initial - 0.5 * velocity
        provenance[key] = {
            "repository": spec.repository,
            "revision": spec.revision,
            "modelConfigSha256": sha256_file(snapshot / "model_config.json"),
            "modelSha256": sha256_file(snapshot / "model.safetensors"),
            "p0": P0_FILES[key],
            "p0Sha256": sha256_file(p0),
            "objective": config["model"]["diffusion"]["diffusion_objective"],
        }
        del dit, wrapper

    output.mkdir(parents=True, exist_ok=True)
    artifact = output / ARTIFACT
    save_file(tensors, str(artifact))
    manifest = {
        "story": STORY,
        "upstreamCommit": UPSTREAM_COMMIT,
        "sourceSha256": SOURCE_SHA256,
        "runtime": versions,
        "inputs": EXPECTED_INPUTS,
        "snapshots": provenance,
        "artifact": ARTIFACT,
        "artifactSha256": sha256_file(artifact),
        "tensors": _records(artifact),
    }
    (output / "guidance-manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def verify(output: Path) -> None:
    manifest = json.loads(
        (output / "guidance-manifest.json").read_text(encoding="utf-8")
    )
    if manifest.get("story") != STORY or manifest.get("upstreamCommit") != UPSTREAM_COMMIT:
        raise InvalidReference("guidance identity mismatch")
    if manifest.get("sourceSha256") != SOURCE_SHA256:
        raise InvalidReference("guidance source provenance mismatch")
    if manifest.get("runtime") != EXPECTED_RUNTIME:
        raise InvalidReference("guidance runtime provenance mismatch")
    if manifest.get("inputs") != EXPECTED_INPUTS:
        raise InvalidReference("guidance input/control contract mismatch")
    snapshots = manifest.get("snapshots", {})
    if set(snapshots) != set(CASES):
        raise InvalidReference("guidance snapshot coverage mismatch")
    p0_manifest_path = ROOT / "docs/migration/sa3-reference/manifest.json"
    if sha256_file(p0_manifest_path) != P0_MANIFEST_SHA256:
        raise InvalidReference("P0 manifest payload mismatch")
    p0_manifest = json.loads(p0_manifest_path.read_text(encoding="utf-8"))
    p0_snapshots = {entry["key"]: entry for entry in p0_manifest["snapshots"]}
    if sha256_file(SNAPSHOT_LOCK_PATH) != SNAPSHOT_LOCK_SHA256:
        raise InvalidReference("snapshot file lock payload mismatch")
    snapshot_lock = load_snapshot_lock(SNAPSHOT_LOCK_PATH)["snapshots"]
    for key in CASES:
        record = snapshots[key]
        p0_artifact = p0_manifest["artifacts"][key]
        spec = SPEC_BY_KEY[key]
        locked_files = snapshot_lock[key]["files"]
        if (
            record.get("repository") != spec.repository
            or record.get("revision") != spec.revision
            or record.get("revision") != p0_snapshots[key]["revision"]
            or record.get("modelConfigSha256") != locked_files["model_config.json"]["sha256"]
            or record.get("modelConfigSha256") != p0_snapshots[key]["modelConfigSha256"]
            or record.get("modelSha256") != locked_files["model.safetensors"]["sha256"]
            or record.get("objective") != EXPECTED_OBJECTIVES[key]
            or record.get("p0") != p0_artifact["file"]
            or record.get("p0Sha256") != p0_artifact["sha256"]
        ):
            raise InvalidReference(f"{key} guidance snapshot/P0 provenance mismatch")
    artifact = output / manifest.get("artifact", "")
    if (
        sha256_file(artifact) != ARTIFACT_SHA256
        or manifest.get("artifactSha256") != ARTIFACT_SHA256
    ):
        raise InvalidReference("guidance artifact payload mismatch")
    records = _records(artifact)
    expected_keys = {
        f"{case}.{variant}.{output_name}"
        for case in CASES
        for variant in VARIANTS
        for output_name in ("velocity", "final")
    }
    if set(records) != expected_keys:
        raise InvalidReference("guidance tensor inventory mismatch")
    if any(
        record["dtype"] != "F32" or record["shape"] != [1, 256, 16]
        for record in records.values()
    ):
        raise InvalidReference("guidance tensor dtype/shape contract mismatch")
    if records != manifest.get("tensors"):
        raise InvalidReference("guidance tensor inventory/payload mismatch")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--upstream", type=Path)
    parser.add_argument("--small-music", type=Path)
    parser.add_argument("--small-music-base", type=Path)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--generate", action="store_true")
    args = parser.parse_args()
    try:
        if args.generate:
            if not args.upstream or not args.small_music or not args.small_music_base:
                parser.error("--generate requires --upstream and both snapshot paths")
            generate(
                args.upstream,
                {"small-music": args.small_music, "small-music-base": args.small_music_base},
                args.output,
            )
        verify(args.output)
        print("Stable Audio 3 sampler guidance reference verified")
    except (InvalidReference, OSError, ValueError, KeyError) as error:
        print(f"SA3 sampler guidance reference error: {error}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
