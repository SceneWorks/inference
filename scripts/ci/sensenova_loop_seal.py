"""Join successful native reader evidence to the exact same-run release and tier inputs."""

from __future__ import annotations

import json
import os
from pathlib import Path
import subprocess
import sys

from sensenova_loop_assets import ROOT, SOURCE, digest, inventory, write


def main() -> None:
    env = os.environ
    backend = env["MODEL_PROOF_BACKEND"]
    expected_sha = env["GITHUB_SHA"]
    evidence = Path(env["MODEL_PROOF_EVIDENCE"])
    verified = subprocess.check_output(
        [sys.executable, "-I", str(ROOT / "scripts/ci/model_inference_proofs.py"),
         "--verify-artifacts", "--backend", backend, "--request-env", "MODEL_PROOF_REQUEST",
         "--expected-sha", expected_sha, "--output", str(evidence)],
        text=True, cwd=ROOT,
    )
    if json.loads(verified)["accepted_case_execution"] is not True:
        raise ValueError("incomplete native proof")
    source = json.loads(SOURCE.read_bytes())
    report = Path(env["SENSENOVA_NATIVE_REPORT"])
    native = json.loads((report / "native-inputs.json").read_bytes())
    oracle = Path(env["SENSENOVA_LOOP_ORACLE"]).resolve(strict=True)
    oracle_sha = digest(oracle)
    if (native["source"]["revision"], native["backend"], native["source_manifest_sha256"],
            native["oracle_sha256"]) != (expected_sha, backend, digest(SOURCE), oracle_sha):
        raise ValueError("staged receipts differ from the proof source or oracle inputs")
    case = json.loads((evidence / "00000/result.json").read_bytes())
    consumed = {}
    for observation in case["actual_inputs"]:
        consumed.setdefault(observation["role"], set()).update(
            Path(item["absolute_lexical_path"]["display"]).resolve()
            for item in observation["items"] if "absolute_lexical_path" in item
        )
    if backend == "mlx":
        logs = [(evidence / f"00000/body.{stream}.log").read_text(encoding="utf-8")
                for stream in ("stdout", "stderr")]
        matches = [line.split(" ", 1)[1] for log in logs for line in log.splitlines()
                   if line.startswith("SENSENOVA_LOOP_ORACLE_SHA256 ")]
        if matches != [oracle_sha]:
            raise ValueError("MLX test did not attest to the exact oracle bytes")
    elif oracle not in consumed.get("reference", set()):
        raise ValueError("Candle test did not attest to the exact oracle path")

    tiers = {}
    for tier in ("bf16", "q8", "q4"):
        directory = Path(env[f"SENSENOVA_LOOP_{tier.upper()}_SNAPSHOT"])
        expectation = source["files"] if tier == "bf16" else native["tiers"][tier]["files"]
        files = [directory / entry["path"] for entry in expectation]
        if inventory(files, directory) != expectation:
            raise ValueError(f"{tier} bytes changed after native validation")
        weights = {path.resolve() for path in files if path.suffix == ".safetensors"}
        if not weights or not weights <= consumed.get("model-weights", set()):
            raise ValueError(f"proof did not witness the complete {tier} weight set")
        if (directory / "tokenizer.json").resolve() not in consumed.get("tokenizer", set()):
            raise ValueError(f"proof did not witness the {tier} tokenizer")
        tiers[tier] = {"weight_files": len(weights), "payload_bytes": sum(item["size"] for item in expectation)}
    write(report / "sealed-inputs.json", {
        "github_run_id": env["GITHUB_RUN_ID"], "github_run_attempt": env["GITHUB_RUN_ATTEMPT"],
        "source_sha": expected_sha, "hf_revision": source["revision"], "backend": backend,
        "source_manifest_sha256": digest(SOURCE), "oracle_sha256": oracle_sha,
        "native_proof_receipt_sha256": digest(evidence / "receipt.json"), "tiers": tiers,
        "claim": "Exact pre/post byte inventories and separately witnessed successful reader paths",
    })


if __name__ == "__main__":
    main()
