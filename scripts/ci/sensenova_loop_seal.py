"""Join successful native reader evidence to the exact same-run release and tier inputs."""

from __future__ import annotations

import json
import math
import os
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from scripts.ci.sensenova_loop_assets import SOURCE, digest, inventory, write

TIERS = ("bf16", "q8", "q4")


def observed(logs, marker):
    # Libtest writes a status prefix without a newline. Source witnesses must
    # emit their own newline first, so payloads cannot appear inside unrelated log lines.
    prefix = marker + " "
    return [line[len(prefix):] for log in logs for line in log.splitlines() if line.startswith(prefix)]


def verify_oracle_marker(logs, oracle_sha):
    if observed(logs, "SENSENOVA_LOOP_ORACLE_SHA256") != [oracle_sha]:
        raise ValueError("MLX test did not attest to the exact oracle bytes")


def verify_semantic_margins(oracle, logs):
    obligations = {}
    for row in oracle["cases"]:
        for index, step in enumerate(row["steps"]):
            if "semantic_score" in step:
                score = step["semantic_score"]
                logits = dict(zip(step["token_ids"], step["logits"], strict=True))
                positive = set(score["positive_token_ids"])
                negative = set(score["negative_token_ids"])
                if not positive or not negative or positive & negative or not (positive | negative) <= logits.keys():
                    raise ValueError("oracle semantic aliases must be distinct and present in the compared logits")
                margin = max(logits[token] for token in positive) - max(logits[token] for token in negative)
                if not math.isfinite(margin) or margin <= 0 or margin != score["source_positive_margin"]:
                    raise ValueError("oracle semantic score differs from source-bound logits")
                obligations[(row["id"], index)] = margin
    expected_counts = {"text": 2, "red": 1, "blue": 1, "quadrants": 1}
    counts = {case: sum(case == name for name, _ in obligations) for case in expected_counts}
    if counts != expected_counts or len(obligations) != sum(expected_counts.values()):
        raise ValueError("oracle is missing an independent semantic decision")
    margins = {}
    for payload in observed(logs, "SENSENOVA_LOOP_SEMANTIC"):
        record = json.loads(payload)
        if not isinstance(record, dict) or set(record) != {"tier", "case_id", "step", "margin"}:
            raise ValueError("invalid native semantic receipt")
        tier, case, step, margin = (record[key] for key in ("tier", "case_id", "step", "margin"))
        if (tier not in TIERS or not isinstance(case, str) or type(step) is not int
                or (case, step) not in obligations or (tier, case, step) in margins
                or type(margin) not in (int, float) or not math.isfinite(margin) or margin <= 0):
            raise ValueError("unexpected, duplicate or nonpositive native semantic decision")
        margins[tier, case, step] = margin
    if set(margins) != {(tier, case, step) for tier in TIERS for case, step in obligations}:
        raise ValueError("missing tier-specific semantic decision")
    return [{"tier": tier, "case_id": case, "step": step, "margin": margins[tier, case, step],
             "source_positive_margin": obligations[case, step]}
            for tier, case, step in sorted(margins)]



def verify_memory_endpoints(backend, logs):
    mlx = backend == "mlx"
    marker = "SENSENOVA_LOOP_MEMORY" if mlx else "SENSENOVA_LOOP_MEMORY_ENDPOINT"
    endpoints = {}
    for payload in observed(logs, marker):
        record = json.loads(payload)
        endpoint = record["product"] if mlx else record
        tier = endpoint["tier"]
        dimensions = endpoint["image"] if mlx else [endpoint["width"], endpoint["height"]]
        prompt = endpoint["prefill_tokens" if mlx else "prompt_tokens"]
        required = ("load_metal_peak_bytes", "request_metal_peak_bytes", "loaded_process_bytes")
        peaks = ([endpoint[key] for key in required] + [record["sampled_process_peak_bytes"]]
                 if mlx else [endpoint["after_request"]["pool_reserved_peak_bytes"],
                              endpoint["after_request"]["pool_used_peak_bytes"]])
        if (tier not in ("BF16", "Q4") or tier in endpoints or dimensions != [1536, 1024]
                or type(prompt) is not int or prompt <= 1536
                or type(endpoint["generated_tokens"]) is not int or endpoint["generated_tokens"] != 1
                or any(type(value) is not int or value <= 0 for value in peaks)):
            raise ValueError(f"missing, duplicate or incomplete measured {backend} memory endpoint")
        if not mlx and (endpoint["device"] != "candle-cuda" or not endpoint["memory_scope"]):
            raise ValueError("Candle memory endpoint has no actual device/scope")
        endpoints[tier] = record
    if set(endpoints) != {"BF16", "Q4"}:
        raise ValueError(f"{backend} proof lacks separate measured BF16 and q4 memory endpoints")
    return endpoints


def verify_native_card(native, request_policy):
    """The native conversions ran on the card their recorded selection chose (a card-free request)
    or on the named card (no selection), and every tier's claim held that card."""
    from scripts.ci.model_proof_resources import executed_card_policy, require_claimed_card
    if native["backend"] != "candle-cuda":
        if "cuda_card_selection" in native:
            raise ValueError("a non-CUDA native conversion records no card selection")
        return
    executed = executed_card_policy(request_policy, native.get("cuda_card_selection"))
    for tier in native["tiers"].values():
        require_claimed_card(tier["supervision"]["device_claim"], executed)


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
    verify_native_card(native, json.loads(env["MODEL_PROOF_REQUEST"])["policy"])
    case = json.loads((evidence / "00000/result.json").read_bytes())
    consumed = {}
    for observation in case["actual_inputs"]:
        consumed.setdefault(observation["role"], set()).update(
            Path(item["absolute_lexical_path"]["display"]).resolve()
            for item in observation["items"] if "absolute_lexical_path" in item
        )
    logs = [(evidence / f"00000/body.{stream}.log").read_text(encoding="utf-8")
            for stream in ("stdout", "stderr")]
    if backend == "mlx":
        verify_oracle_marker(logs, oracle_sha)
    elif oracle not in consumed.get("reference", set()):
        raise ValueError("Candle test did not attest to the exact oracle path")

    semantic = verify_semantic_margins(json.loads(oracle.read_bytes()), logs)
    memory = verify_memory_endpoints(backend, logs)
    tiers = {}
    for tier in TIERS:
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
        "semantic_margins": semantic, "memory_endpoints": memory,
        "claim": "Exact pre/post byte inventories and separately witnessed successful reader paths",
    })


if __name__ == "__main__":
    main()
