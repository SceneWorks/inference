#!/usr/bin/env python3
"""Recover the exact three- and five-reference GPU 0 measurements through RGB completion."""

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from scripts.ci import scail2_plural_fit as runner
from scripts.ci.scail2_recover_completed_job import replay_archive
from scripts.ci.scail_plural import contract as c
from scripts.ci.scail_plural.completed_job import OWNED_ZERO_CAPTURE


def replay_files(dest, policy):
    return derive_certificates(dest, policy, OWNED_ZERO_CAPTURE, [2, 3, 4, 5], [3, 5],
                               "scail2-quiet-gpu0-owned-zero-recovery-v1")


def derive_certificates(dest, policy, pin, counts, missing, schema, *, fresh_cuda=False):
    receipt = json.loads((dest / "receipt.json").read_bytes())
    contract = receipt["contract"]
    if not (
        c.digest_json(contract) == pin["plan_sha256"] and contract["head"] == pin["inference_revision"]
        and receipt["run_id"] == pin["run_id"] and receipt["run_attempt"] == pin["run_attempt"]
        and receipt["accepted"] is False and receipt["failure"] == "incomplete owned WDDM poll series"
        and not receipt.get("log_errors") and contract["counts"] == counts
        and set(receipt["counts"]) == {str(n) for n in counts}
        and all(receipt["counts"][str(n)]["accepted"] is (n not in missing) for n in counts)
        and c.digest_json(receipt["stage"]) == "8c64b413630a83219be5d9dc340da0614774a3a7ac98d1241de216b3c9827e0f"
    ):
        raise ValueError("unexpected CUDA continuation, stage or original count verdicts")
    certificates = {}
    for count in missing:
        record = receipt["counts"][str(count)]
        guard = json.loads((dest / f"count-{count}.supervisor.json").read_bytes())
        pool = json.loads((dest / f"count-{count}.cuda-pool.json").read_bytes())
        body = runner.proof.verify_execution(contract["case"], (dest / f"count-{count}.stdout.log").read_text())
        if record["body"] != body or record["telemetry_failure"] != "incomplete_wddm_samples":
            raise ValueError("native execution or original failure differs")
        args = (record["native"], guard, pool, receipt["adapter"], receipt["hardware"],
                "candle-cuda", count, policy, contract["head"])
        completed = (record["claim"], "owned-job-completion")
        original = runner.measured_outcome(*args, completed_job=completed, fresh_cuda=fresh_cuda)
        if original != {"accepted": False, "telemetry_failure": "incomplete_wddm_samples"}:
            raise ValueError("original missing-poll verdict did not reproduce")
        candidate = c.measured_row(*args, completed_job=completed, archived_owned_zero=True, fresh_cuda=fresh_cuda)
        if "terminal_rejected_read_ns" not in candidate.get("cuda_render_scope", {}):
            raise ValueError("no verified ordering between the final rejected reading and the owned zero")
        certificates[str(count)] = dict(schema=schema, capture=pin,
                                        references=count, candidate_row_pending_physical_review=candidate)
    return certificates


def recover(zip_path):
    request = dict(selection=c.SELECTION, continuation=c.CONTINUATION, policy=c.policy("candle-cuda", cuda_uuid=c.GPU0))
    return replay_archive(zip_path, OWNED_ZERO_CAPTURE, 309630, [2, 3, 4, 5], request, replay_files)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("archive", type=Path)
    print(json.dumps(recover(parser.parse_args().archive), sort_keys=True, separators=(",", ":")))
