#!/usr/bin/env python3
"""Independently replay the pinned six-reference Actions archive as a completed Windows Job."""

import argparse
import hashlib
import io
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import zipfile

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from scripts.ci import scail2_plural_fit as runner
from scripts.ci.scail_plural import contract as c


FAILURE = "rejected CUDA pool attempt without a verified owned zero"


def replay_files(dest, policy):
    receipt = json.loads((dest / "receipt.json").read_bytes())
    original = receipt["contract"]
    if (c.digest_json(original) != c.RECOVERED_SIX["plan_sha256"] or receipt["accepted"] is not False
            or set(receipt["counts"]) != {"6"} or receipt["failure"] != FAILURE):
        raise ValueError("archive is not the sealed terminal six-reference refusal")
    record = receipt["counts"]["6"]
    native = record["native"]
    claim = record["claim"]
    guard = json.loads((dest / "count-6.supervisor.json").read_bytes())
    pool = json.loads((dest / "count-6.cuda-pool.json").read_bytes())
    body = runner.proof.verify_execution(original["case"], (dest / "count-6.stdout.log").read_text())
    if record["accepted"] is not False or record["body"] != body or guard.get("owned_completion") is not None:
        raise ValueError("original native body or expected historical supervision changed")
    args = (native, guard, pool, receipt["adapter"], receipt["hardware"], "candle-cuda", 6, policy, original["head"])
    runner.require_release(claim, "candle-cuda", receipt["hardware"], policy)
    try:
        runner.measured_outcome(*args)
    except ValueError as exc:
        if str(exc) != FAILURE:
            raise
    else:
        raise ValueError("historical completed count did not reproduce its exact original refusal")
    # At this immutable source, run_guarded can return its successful RunResult
    # only after child.poll() is zero AND Job.members() is empty. The archive
    # has the actual successful return; no synthetic owned-zero sample is added.
    result = runner.measured_outcome(*args, completed_job=(claim, "archived-supervisor-return"))
    if not result["accepted"] or "cuda_completed_job" not in result["candidate_row_pending_physical_review"]:
        raise ValueError("completed Job could not independently qualify the source measurements")
    return dict(schema="scail2-quiet-gpu0-six-recovery-v1", capture=c.RECOVERED_SIX, references=6,
                candidate_row_pending_physical_review=result["candidate_row_pending_physical_review"])


def recover(zip_path):
    pin = c.RECOVERED_SIX
    with zip_path.open("rb") as archive:
        raw = archive.read(98831)
    if len(raw) != 98830 or hashlib.sha256(raw).hexdigest() != pin["archive_sha256"]:
        raise ValueError("not the authenticated GitHub Actions archive 14719969057")
    with tempfile.TemporaryDirectory() as temp:
        root = Path(temp).resolve()
        dest, checkout = root / "evidence", root / "original"
        dest.mkdir()
        files = {"receipt.json", "count-6.cuda-pool.json", "count-6.stdout.log", "count-6.stderr.log", "count-6.supervisor.json"}
        with zipfile.ZipFile(io.BytesIO(raw)) as archive:
            if sorted(archive.namelist()) != sorted(files):
                raise ValueError("unexpected source evidence files")
            for name in files:
                (dest / name).write_bytes(archive.read(name))
        if runner.tier1.sha(dest / "receipt.json") != pin["receipt_sha256"]:
            raise ValueError("original producer receipt seal mismatch")
        request = {"selection": c.SELECTION, "policy": c.policy("candle-cuda", cuda_uuid=c.GPU0)}
        # Run the actual archived verifier (including its exact-head check) in
        # a private, local metadata checkout. No network, weights or GPU calls.
        subprocess.run(["git", "clone", "--quiet", "--shared", "--no-checkout", str(ROOT), str(checkout)], check=True)
        subprocess.run(["git", "sparse-checkout", "set", "scripts"], cwd=checkout, check=True)
        subprocess.run(["git", "checkout", "--quiet", "--detach", pin["inference_revision"]], cwd=checkout, check=True)
        env = dict(os.environ, RUNNER_TEMP=str(root), GITHUB_RUN_ID=pin["run_id"], GITHUB_RUN_ATTEMPT=pin["run_attempt"],
                   VIDEO_TIER1_PLAN_SHA256=pin["plan_sha256"], SCAIL_PLURAL_RECEIPT_SHA256=pin["receipt_sha256"],
                   ARCHIVED_REQUEST=json.dumps(request))
        subprocess.run([sys.executable, "-I", "scripts/ci/scail2_plural_fit.py", "--verify-artifacts", "--backend",
                        "candle-cuda", "--expected-sha", pin["inference_revision"], "--request-env", "ARCHIVED_REQUEST",
                        "--output", str(dest)], cwd=checkout, env=env, check=True)
        policy = runner.tier1.read_policy(root, "policy.json", request["policy"])
        return replay_files(dest, policy)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("archive", type=Path)
    print(json.dumps(recover(parser.parse_args().archive), sort_keys=True, separators=(",", ":")))
