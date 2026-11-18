"""Publish the exact staged native q8/q4 tiers that passed this job's proof."""

from __future__ import annotations

import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

from sensenova_loop_assets import ROOT, SOURCE, digest, inventory, write

REPOS = {
    "mlx": "SceneWorks/sensenova-loop-9b-mlx",
    "candle-cuda": "SceneWorks/sensenova-loop-9b-candle",
}


def main() -> None:
    from huggingface_hub import HfApi

    env = os.environ
    if env.get("GITHUB_EVENT_NAME") != "workflow_dispatch":
        raise ValueError("publication requires an explicit dispatch")
    backend = env["MODEL_PROOF_BACKEND"]
    request = json.loads(env["MODEL_PROOF_REQUEST"])
    if request.get("sensenova_publish") is not True:
        raise ValueError("SenseNova publication was not selected")
    expected_sha = env["GITHUB_SHA"]
    evidence = Path(env["MODEL_PROOF_EVIDENCE"])
    native_report = Path(env["SENSENOVA_NATIVE_REPORT"])
    seal = json.loads((native_report / "sealed-inputs.json").read_bytes())
    if (seal["source_sha"], seal["backend"], seal["github_run_id"], seal["github_run_attempt"],
            seal["native_proof_receipt_sha256"]) != (
        expected_sha, backend, env["GITHUB_RUN_ID"], env["GITHUB_RUN_ATTEMPT"], digest(evidence / "receipt.json")
    ):
        raise ValueError("publication requires the exact sealed native proof from this dispatch")
    source = json.loads(SOURCE.read_bytes())
    native = json.loads((Path(env["SENSENOVA_NATIVE_REPORT"]) / "native-inputs.json").read_bytes())
    if (native["source"]["revision"], native["backend"], native["source_manifest_sha256"],
            native["oracle_sha256"]) != (
        expected_sha, backend, digest(SOURCE), digest(Path(env["SENSENOVA_LOOP_ORACLE"]))
    ):
        raise ValueError("staged input receipt does not bind the verified run")
    snapshots = {tier: Path(env[f"SENSENOVA_LOOP_{tier.upper()}_SNAPSHOT"])
                 for tier in ("q8", "q4")}
    output = Path(env["SENSENOVA_PUBLICATION_REPORT"])
    output.mkdir()
    staging = snapshots["q4"].parent / "publication"
    staging.mkdir()
    for tier, directory in snapshots.items():
        if inventory([item for item in directory.rglob("*") if item.is_file()], directory) != native["tiers"][tier]["files"]:
            raise ValueError(f"{tier} bytes changed after validation")

    metadata = staging / "provenance"
    metadata.mkdir()
    shutil.copyfile(ROOT / "LICENSE", metadata / "LICENSE")
    (metadata / "README.md").write_text(
        f"# SenseNova Loop-9B for SceneWorks ({backend})\n\n"
        f"Source model: [{source['repository']}](https://huggingface.co/{source['repository']}"
        f"/tree/{source['revision']}) by SenseNova, licensed Apache-2.0.\n\n"
        "The source link provides the official BF16 checkpoint. SceneWorks prepared the "
        "q8 and q4 directories from that pinned checkpoint with its native converter.\n\n"
        f"SceneWorks/inference source: `{expected_sha}`.\n",
        encoding="utf-8",
    )
    write(metadata / "source.json", source)
    write(metadata / "proof.json", {
        "inference_revision": expected_sha, "backend": backend,
        "source_manifest_sha256": native["source_manifest_sha256"],
        "oracle_sha256": native["oracle_sha256"], "native_proof_receipt_sha256": digest(evidence / "receipt.json"),
        "github_run_id": env["GITHUB_RUN_ID"], "github_run_attempt": env["GITHUB_RUN_ATTEMPT"],
    })
    argv = [sys.executable, str(ROOT / "scripts/ci/publish_hf_tiers.py"), "--repo", REPOS[backend],
            "--message", f"Publish verified SenseNova Loop-9B {backend} tiers ({expected_sha})",
            "--report", str(output / "publication.json")]
    argv.extend(("--map", f"{metadata}=provenance"))
    for tier, directory in snapshots.items():
        argv.extend(("--map", f"{directory}={tier}"))
    subprocess.run(argv, cwd=ROOT, check=True)
    report = json.loads((output / "publication.json").read_bytes())
    published = HfApi().model_info(REPOS[backend], revision=report["revision"], files_metadata=True, token=False)
    if published.sha != report["revision"] or published.private or published.gated:
        raise ValueError("published tier revision is not anonymously readable and ungated")
    write(output / "verified-publication.json", {
        "repository": REPOS[backend], "revision": published.sha, "inference_revision": expected_sha,
        "publication_receipt_sha256": digest(output / "publication.json"),
    })


if __name__ == "__main__":
    main()
