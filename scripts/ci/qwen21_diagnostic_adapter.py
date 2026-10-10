"""Verify the immutable failed-candidate donor; this never grants acceptance."""
import argparse
import json
import math
from pathlib import Path
import struct
import urllib.request

from scripts.ci.qwen21_adapter_imports import materialize, sha256

SOURCE = "b3ec3f0b8ee6880dc55e6433a35dc60116709752"
RUN = 37214050997
JOB = 111470848550
DATASET = "0d4927eedfcf33eb609278089c02c58f729a87f65d78170b0d7076f3a352073c"
ADAPTER_SHA = "c233129e9a64e384850331c9804b5b496d5208c08fe34aca4c680920fddb03d7"
RECEIPT_SHA = "fa5b6234870f6ac0784f893067524407178ee9ee1678c50bc6bd6713d008e333"
CAPTION = "zxq edit: invert each RGB colour channel of image 1 independently, then quantize each channel to the four numeric levels 0, 85, 170, 255; preserve the shapes and keep the result in colour"


def validate_provenance(manifest):
    expected = {"trainingSourceMain": SOURCE, "trainingRun": RUN, "trainingJob": JOB,
                "trainingRunConclusion": "failure retained; training complete and finite, learned/cross-mode render assertions failed",
                "steps": 120, "datasetSha256": DATASET, "trainingCaption": CAPTION}
    if (manifest.get("purpose") != "DIAGNOSTIC_ONLY"
            or manifest.get("acceptanceEvidence") is not False
            or manifest.get("trainingProvenance") != expected):
        raise ValueError("immutable original training provenance and diagnostic-only scope required")
    entries = {entry["name"]: entry for entry in manifest.get("adapters", [])}
    if set(entries) != {"cuda_lokr", "mlx_corrected_edit_lokr", "training_receipt"}:
        raise ValueError("exactly the two LoKr files and training receipt required")
    for name, expected_sha in [("cuda_lokr", "201bffa58dc8a1b61ec6399845109ebc3741220f9942f6caa0bfdc85695abf96"), ("mlx_corrected_edit_lokr", ADAPTER_SHA), ("training_receipt", RECEIPT_SHA)]:
        if entries[name].get("sha256") != expected_sha:
            raise ValueError("original donor byte identities are immutable")
    return entries


def validate_training(receipt, metadata):
    training = receipt.get("training", {})
    losses = training.get("losses", [])
    protocol = training.get("editProtocol", {})
    if (training.get("steps") != 120 or training.get("stepsRun") != 120
            or len(losses) != 120 or not all(isinstance(x, (int, float)) and math.isfinite(x) for x in losses)
            or [x.get("step") for x in training.get("stepSamples", [])] != list(range(1, 121))
            or training.get("trainingStageTraceComplete") is not True
            or training.get("adapterSha256") != ADAPTER_SHA
            or training.get("adapterBytes") != 6759417
            or training.get("dataset", {}).get("sha256") != DATASET
            or protocol.get("trainingReferenceCount") != 1
            or protocol.get("evaluationReferenceCount") != 2
            or protocol.get("trainingCaption") != CAPTION):
        raise ValueError("completed finite original 120-step receipt required")
    required = {"family": "qwen-image-2-1", "baseModel": "qwen_image_2_1",
                "trainingMode": "edit", "networkType": "lokr", "rank": "16", "alpha": "16",
                "license": "Qwen Research License Agreement (research/evaluation only)"}
    if any(metadata.get(k) != v for k, v in required.items()) or training.get("metadata") != metadata:
        raise ValueError("family, base, edit LoKr, rank and research license metadata must match")


def prepare(manifest, destination, opener=urllib.request.urlopen):
    entries = validate_provenance(manifest)
    resolved = materialize(manifest, destination, opener)
    receipt = json.loads((destination / entries["training_receipt"]["file"]).read_text(encoding="utf-8"))
    adapter = destination / entries["mlx_corrected_edit_lokr"]["file"]
    with adapter.open("rb") as stream:
        header_bytes = struct.unpack("<Q", stream.read(8))[0]
        if header_bytes > adapter.stat().st_size - 8:
            raise ValueError("invalid safetensors header length")
        metadata = json.loads(stream.read(header_bytes)).get("__metadata__", {})
    validate_training(receipt, metadata)
    marker = {"kind": "DIAGNOSTIC_ONLY", "acceptanceEvidence": False,
              "trainingProvenance": manifest["trainingProvenance"],
              "adapterSha256": sha256(adapter), "resolvedManifest": str(resolved)}
    (destination.parent / "DIAGNOSTIC_ONLY.json").write_text(json.dumps(marker, indent=2) + "\n", encoding="utf-8")
    return resolved


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--destination", type=Path, required=True)
    args = parser.parse_args()
    print(prepare(json.loads(args.manifest.read_text(encoding="utf-8")), args.destination))


if __name__ == "__main__":
    main()
