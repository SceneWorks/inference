#!/usr/bin/env python3
"""Transfer the immutable M3 CPU reference to an exact new engine source revision.

This runs on a hosted CPU runner. It neither loads models nor regenerates tensors.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import re
import stat
import zipfile

SOURCE_RUN_ID = 36748416547
SOURCE_ARTIFACT_ID = 11116254689
SOURCE_ENGINE_SHA = "4127a675fc8575555e029e01b7f6867488880a8f"
SOURCE_ZIP_SHA256 = "87b9e00ac3fc9e219b1deb9f6f8248c89835b68cfbcb4d333741aa415998213b"
REFERENCE_SHA256 = "c4073edb3c7abfbf5d50a9c5bfa67c73b9b9a8ba20115570ddb616d00d5b72d4"
SOURCE_METADATA_SHA256 = "7bd7dda64fb1ee638cdfeffab3de939e73be517bd1f76c54af3c1148a71489f6"
LICENSE_SHA256 = "08775dcfda5784cefaa501ded294f4ce3d42d632370cb871a4785439db465b9e"
FILES = {
    "vae_real_reference.safetensors": (18_593_152, REFERENCE_SHA256),
    "reference-provenance.json": (203, SOURCE_METADATA_SHA256),
    "NONCOMMERCIAL.txt": (404, LICENSE_SHA256),
}
RELAY_RUN_ID = 37138394196
RELAY_ARTIFACT_ID = 11279267367
RELAY_ENGINE_SHA = "825341ff8d0110ea448213485891b39d57806fa4"
RELAY_CONTROL_SHA = "29d4e30e9e281d45f04048560003d8e3b8c55790"
RELAY_ZIP_SHA256 = "c0a9ceef9276d12ddc8f80b27d671896811b04865e4a6d65b98aa4472851a472"
RELAY_METADATA_SHA256 = "09362aa98d0908760503a45a4ea60840ea1c42b621f209c6d8e0d3a3d7c2bb58"
RELAY_FILES = {
    "vae_real_reference.safetensors": (18_593_152, REFERENCE_SHA256),
    "reference-provenance.json": (828, RELAY_METADATA_SHA256),
    "NONCOMMERCIAL.txt": (404, LICENSE_SHA256),
}


def require(ok: bool, message: str) -> None:
    if not ok:
        raise ValueError(message)


def digest(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def transfer(source_zip: Path, run_json: Path, artifact_json: Path,
             engine_sha: str, control_sha: str, transfer_run_id: str,
             transfer_attempt: str, output: Path, source_mode: str = "direct") -> dict:
    require(re.fullmatch(r"[0-9a-f]{40}", engine_sha) is not None and
            re.fullmatch(r"[0-9a-f]{40}", control_sha) is not None,
            "new engine/control source must be exact lowercase SHAs")
    require(re.fullmatch(r"[1-9][0-9]*", transfer_run_id) is not None and
            transfer_attempt == "1",
            "hosted transfer run identity missing")
    require(not output.exists(), "refuse to overwrite a reference transfer")
    require(source_mode in ("direct", "relay"), "reference source mode is not reviewed")
    run = json.loads(run_json.read_text(encoding="utf-8"))
    artifact = json.loads(artifact_json.read_text(encoding="utf-8"))
    source_run = SOURCE_RUN_ID if source_mode == "direct" else RELAY_RUN_ID
    source_artifact = SOURCE_ARTIFACT_ID if source_mode == "direct" else RELAY_ARTIFACT_ID
    source_head = SOURCE_ENGINE_SHA if source_mode == "direct" else RELAY_CONTROL_SHA
    source_zip_sha = SOURCE_ZIP_SHA256 if source_mode == "direct" else RELAY_ZIP_SHA256
    files = FILES if source_mode == "direct" else RELAY_FILES
    require(run.get("id") == source_run and run.get("run_attempt") == 1 and
            run.get("head_sha") == source_head and run.get("conclusion") == "success" and
            run.get("event") == "workflow_dispatch" and
            run.get("name") == "YuE2 targeted stage-precision proof" and
            run.get("repository", {}).get("full_name") == "SceneWorks/inference" and
            run.get("repository", {}).get("id") == 1299380446,
            "CPU fixture source run identity or verdict changed")
    binding = artifact.get("workflow_run") or {}
    require(artifact.get("id") == source_artifact and
            artifact.get("name") == "yue2-precision-reference" and
            artifact.get("digest") == f"sha256:{source_zip_sha}" and
            artifact.get("expired") is False and
            binding.get("id") == source_run and
            binding.get("head_sha") == source_head and
            binding.get("repository_id") == run["repository"]["id"],
            "CPU fixture source artifact transport identity changed")
    require(source_zip.is_file() and digest(source_zip.read_bytes()) == source_zip_sha,
            "CPU fixture source ZIP digest changed")
    extracted: dict[str, bytes] = {}
    with zipfile.ZipFile(source_zip) as archive:
        members = archive.infolist()
        require(len(members) == len(files) and {member.filename for member in members} == set(files),
                "CPU fixture source archive inventory changed")
        for member in members:
            name = member.filename
            mode = member.external_attr >> 16
            require("/" not in name and "\\" not in name and not member.is_dir() and
                    (stat.S_IFMT(mode) in (0, stat.S_IFREG)),
                    "original CPU fixture archive has an unsafe member")
            expected_size, expected_hash = files[name]
            require(member.file_size == expected_size, f"original {name} size changed")
            data = archive.read(member)
            require(len(data) == expected_size and digest(data) == expected_hash,
                    f"original {name} bytes changed")
            extracted[name] = data
    previous = json.loads(extracted["reference-provenance.json"])
    original = {"engine_sha": SOURCE_ENGINE_SHA, "sha256": REFERENCE_SHA256,
                "runner": "nax-macos", "source": "pinned YuE2 VAE upstream CPU fixture"}
    if source_mode == "direct":
        require(previous == original, "original CPU fixture provenance changed")
    else:
        expected_relay = {"engine_sha": RELAY_ENGINE_SHA,
                          "control_sha": RELAY_CONTROL_SHA,
                          "sha256": REFERENCE_SHA256, "runner": "hosted-cpu-transfer",
                          "source": "immutable M3 YuE2 VAE upstream CPU fixture transfer",
                          "source_run_id": SOURCE_RUN_ID, "source_run_attempt": 1,
                          "source_engine_sha": SOURCE_ENGINE_SHA,
                          "source_artifact_id": SOURCE_ARTIFACT_ID,
                          "source_artifact_zip_sha256": SOURCE_ZIP_SHA256,
                          "source_provenance_sha256": SOURCE_METADATA_SHA256,
                          "noncommercial_sha256": LICENSE_SHA256,
                          "transfer_run_id": str(RELAY_RUN_ID), "transfer_run_attempt": "1"}
        require(previous == expected_relay,
                "relay provenance does not preserve the original CPU teacher chain")
    output.mkdir(parents=True)
    (output / "vae_real_reference.safetensors").write_bytes(extracted["vae_real_reference.safetensors"])
    (output / "NONCOMMERCIAL.txt").write_bytes(extracted["NONCOMMERCIAL.txt"])
    provenance = {"engine_sha": engine_sha, "control_sha": control_sha,
                  "sha256": REFERENCE_SHA256, "runner": "hosted-cpu-transfer",
                  "source": "immutable M3 YuE2 VAE upstream CPU fixture transfer",
                  "source_run_id": SOURCE_RUN_ID, "source_run_attempt": 1,
                  "source_engine_sha": SOURCE_ENGINE_SHA,
                  "source_artifact_id": SOURCE_ARTIFACT_ID,
                  "source_artifact_zip_sha256": SOURCE_ZIP_SHA256,
                  "source_provenance_sha256": SOURCE_METADATA_SHA256,
                  "noncommercial_sha256": LICENSE_SHA256,
                  "transfer_run_id": transfer_run_id,
                  "transfer_run_attempt": transfer_attempt}
    if source_mode == "relay":
        provenance.update({"reference_source_mode": "relay",
                           "relay_run_id": RELAY_RUN_ID,
                           "relay_artifact_id": RELAY_ARTIFACT_ID,
                           "relay_engine_sha": RELAY_ENGINE_SHA,
                           "relay_control_sha": RELAY_CONTROL_SHA,
                           "relay_artifact_zip_sha256": RELAY_ZIP_SHA256,
                           "relay_provenance_sha256": RELAY_METADATA_SHA256})
    (output / "reference-provenance.json").write_text(
        json.dumps(provenance, sort_keys=True, indent=2) + "\n", encoding="utf-8")
    return provenance


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("source-zip", "run-json", "artifact-json", "engine-sha", "control-sha",
                 "transfer-run-id", "transfer-attempt", "output"):
        parser.add_argument(f"--{name}", required=True, type=Path if name.endswith(("zip", "json")) or name == "output" else str)
    parser.add_argument("--source-mode", choices=("direct", "relay"), default="direct")
    args = parser.parse_args()
    print(json.dumps(transfer(args.source_zip, args.run_json, args.artifact_json,
                              args.engine_sha, args.control_sha, args.transfer_run_id,
                              args.transfer_attempt, args.output, args.source_mode), sort_keys=True))


if __name__ == "__main__":
    main()
