"""Strict failed-candidate input for one diagnostic; never grants acceptance."""

import argparse
import io
import json
import math
import os
from pathlib import Path
import re
import shutil
import struct
import subprocess
import tempfile
import urllib.request

from scripts.ci.qwen21_adapter_imports import materialize, sha256
from scripts.ci.qwen21_diagnostic_adapter import CAPTION

SOURCE = "d0775d4922f38e442d38485d412405997f9abc80"
RUN = 37249426665
DATASET = "44a19887ab4c11293069c5ae2b2a5022ed53eec4bdbbe423a75cc82136b09ba4"
ADAPTER_SHA = "1a8535148c65d266d2b583d88a4c0b056d0e8e55eca9c9fa05378d67b5fde781"
RECEIPT_SHA = "f7abbffe6aa4f1a0117e549524e5cfcc3cde2fc1214d1eec2ed3fe49a2567c74"
SCHEMA_SHA = "3f84a6ef09b0810da832f22e93f5ef9172390faf2aba3186beee903ea499ebe5"
PROVENANCE = {"sourceCandidate": SOURCE, "runId": RUN, "steps": 120, "datasetSha256": DATASET}
ENTRIES = [
    {"name": "mlx_failed_balanced_edit_lokr", "kind": "lokr", "assetId": 611294721,
     "assetName": "diagnostic_only_qwen21_edit_120step_1a853514.safetensors", "size": 6759417,
     "sha256": ADAPTER_SHA, "file": "qwen21_edit_lokr_120step_1a853.safetensors"},
    {"name": "training_receipt", "kind": "diagnostic_training_receipt", "assetId": 611294720,
     "assetName": "diagnostic_only_qwen21_edit_120step_1a853514_training.json", "size": 53379,
     "sha256": RECEIPT_SHA, "file": "qwen21_edit_lokr_120step_1a853_training.json"},
]
MANIFEST = {"kind": "DIAGNOSTIC_ONLY", "purpose": "DIAGNOSTIC_ONLY", "acceptanceEvidence": False,
            "repository": "SceneWorks/inference", "releaseId": 402708057,
            "trainingProvenance": PROVENANCE, "adapters": ENTRIES}


def validate_manifest(manifest):
    if json.dumps(manifest, sort_keys=True) != json.dumps(MANIFEST, sort_keys=True):
        raise ValueError("exact failed-candidate diagnostic provenance/assets required")


def read_adapter(path):
    data = path.read_bytes()
    if len(data) < 8:
        raise ValueError("truncated safetensors header")
    length = struct.unpack("<Q", data[:8])[0]
    if length > len(data) - 8:
        raise ValueError("invalid safetensors header length")
    def unique(pairs):
        result = {}
        for key, value in pairs:
            if key in result:
                raise ValueError("duplicate safetensors key")
            result[key] = value
        return result
    header = json.loads(data[8:8 + length], object_pairs_hook=unique)
    tensors = {k: v for k, v in header.items() if k != "__metadata__"}
    schema = {k: {"dtype": v.get("dtype"), "shape": v.get("shape")} for k, v in tensors.items()}
    import hashlib
    digest = hashlib.sha256(json.dumps(schema, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    if len(tensors) != 672 or digest != SCHEMA_SHA:
        raise ValueError("exact 224-module/672 F32 tensor schema required")
    ranges = []
    for value in tensors.values():
        shape, offsets = value["shape"], value.get("data_offsets", [])
        if (len(offsets) != 2 or any(type(x) is not int for x in offsets)
                or offsets[0] < 0 or offsets[1] > len(data) - 8 - length
                or offsets[1] - offsets[0] != math.prod(shape) * 4):
            raise ValueError("invalid F32 tensor bounds")
        ranges.append(offsets)
        payload = data[8 + length + offsets[0]:8 + length + offsets[1]]
        if not all(math.isfinite(x[0]) for x in struct.iter_unpack("<f", payload)):
            raise ValueError("nonfinite adapter factors")
    cursor = 0
    for start, end in sorted(ranges):
        if start != cursor:
            raise ValueError("overlap/gap in adapter storage")
        cursor = end
    if cursor != len(data) - 8 - length:
        raise ValueError("unaccounted adapter storage")
    return header.get("__metadata__", {})


def validate_training(receipt, metadata):
    training = receipt.get("training", {})
    losses = training.get("losses", [])
    samples = training.get("stepSamples", [])
    protocol = training.get("editProtocol", {})
    if (training.get("steps") != 120 or training.get("stepsRun") != 120
            or len(losses) != 120 or not all(type(x) in (int, float) and math.isfinite(x) for x in losses)
            or [x.get("step") for x in samples] != list(range(1, 121))
            or [x.get("loss") for x in samples] != losses
            or training.get("trainingStageTraceComplete") is not True
            or training.get("adapterSha256") != ADAPTER_SHA or training.get("adapterBytes") != 6759417
            or training.get("dataset", {}).get("sha256") != DATASET
            or protocol.get("dataset", {}).get("sha256") != DATASET
            or protocol.get("trainingReferenceCount") != 1 or protocol.get("evaluationReferenceCount") != 2
            or protocol.get("trainingCaption") != CAPTION
            or protocol.get("trainingTargetEdge") != 512 or protocol.get("trainingReferenceFittedEdge") != 1024
            or protocol.get("stepsRequested") != 120
            or protocol.get("evaluationKeySha256") != "da3be3ca711ec3d0e89df8f1e91bc662365fea188fd513d0108782b6c726189f"
            or protocol.get("trainingDataRecipe", {}).get("version") != "balanced64-v1"
            or protocol.get("trainingDataRecipe", {}).get("heldoutSource99UsedForTraining") is not False
            or training.get("rank") != 16 or training.get("learningRate") != struct.unpack("<f", struct.pack("<f", 0.0001))[0]
            or training.get("networkType") != "Lokr" or training.get("resolution") != 512):
        raise ValueError("complete finite original balanced64 120-step training receipt required")
    required = {"family": "qwen-image-2-1", "baseModel": "qwen_image_2_1", "trainingMode": "edit",
                "networkType": "lokr", "rank": "16", "alpha": "16",
                "license": "Qwen Research License Agreement (research/evaluation only)",
                "modelspec.license": "Qwen Research License Agreement (research/evaluation only)"}
    if any(metadata.get(k) != v for k, v in required.items()) or training.get("metadata") != metadata:
        raise ValueError("exact family/base/license/edit/LoKr metadata required")


def prepare(manifest, destination, source_sha, opener=urllib.request.urlopen):
    validate_manifest(manifest)
    if not re.fullmatch(r"[0-9a-f]{40}", source_sha):
        raise ValueError("exact executing source SHA required")
    destination = Path(destination).absolute()
    if destination.resolve() != destination:
        raise ValueError("diagnostic destination may not escape through symlinks")
    destination.parent.mkdir(parents=True, exist_ok=True)
    output = destination / "velocity-adapter-resolved.json"
    marker = destination.parent / "DIAGNOSTIC_ONLY.json"
    output.unlink(missing_ok=True)
    marker.unlink(missing_ok=True)

    def owned_draft(request):
        response = opener(request)
        if request.full_url.endswith("/releases/402708057"):
            with response:
                raw = response.read()
            release = json.loads(raw)
            if (release.get("id") != 402708057 or release.get("draft") is not True
                    or release.get("url") != "https://api.github.com/repos/SceneWorks/inference/releases/402708057"):
                raise ValueError("owned unpublished diagnostic draft required")
            return io.BytesIO(raw)
        return response

    with tempfile.TemporaryDirectory(prefix="velocity-stage-", dir=destination.parent) as stage:
        staging = Path(stage)
        materialize(manifest, staging, owned_draft)
        metadata = read_adapter(staging / ENTRIES[0]["file"])
        receipt = json.loads((staging / ENTRIES[1]["file"]).read_text(encoding="utf-8"))
        validate_training(receipt, metadata)
        destination.mkdir(parents=True, exist_ok=True)
        for entry in ENTRIES:
            target = destination / entry["file"]
            if target.is_symlink():
                raise ValueError("diagnostic asset destination is a symlink")
            shutil.copyfile(staging / entry["file"], target)
            if target.stat().st_size != entry["size"] or sha256(target) != entry["sha256"]:
                raise ValueError("installed diagnostic asset identity mismatch")
    resolved = {"kind": "DIAGNOSTIC_ONLY", "purpose": "DIAGNOSTIC_ONLY", "acceptanceEvidence": False,
                "sourceCandidate": source_sha, "directory": str(destination),
                "trainingProvenance": PROVENANCE, "adapters": [ENTRIES[0]],
                "trainingReceipt": {k: ENTRIES[1][k] for k in ("file", "size", "sha256")},
                "assetProvenance": {"repository": manifest["repository"], "releaseId": manifest["releaseId"],
                                    "draft": True, "assets": ENTRIES}}
    output.write_text(json.dumps(resolved, indent=2) + "\n", encoding="utf-8")
    marker.write_text(json.dumps(resolved, indent=2) + "\n", encoding="utf-8")
    return output


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--destination", type=Path, required=True)
    args = parser.parse_args()
    source = os.environ.get("GITHUB_SHA", "")
    head = subprocess.run(["git", "rev-parse", "HEAD"], check=True, capture_output=True, text=True, encoding="utf-8").stdout.strip()
    if source != head:
        raise ValueError("executing checkout must equal GITHUB_SHA")
    print(prepare(json.loads(args.manifest.read_text(encoding="utf-8")), args.destination, source))


if __name__ == "__main__":
    main()
