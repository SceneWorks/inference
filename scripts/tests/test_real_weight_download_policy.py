"""Every real-weight job that reads a Hub snapshot downloads it into the runner's hub cache (SW-F375).

A job that names a Hugging Face snapshot (a `models--*` path, a `*_SNAPSHOT` or `*MODELS_ROOT`
variable, or a direct Hub client call) must go through one of the shared materializers, which
download a missing pinned snapshot into the runner's Hugging Face hub cache and reuse a complete
one. Downloads anywhere else -- a per-run `local_dir`, the retired task-owned
`~/sceneworks-rw-weights/hub`, a user-profile `~/.cache` -- are rejected outright.
"""

from __future__ import annotations

import re
import sys
import unittest
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts" / "ci"))
import real_weights_workflow  # noqa: E402

# Text that means a job reads (or fetches) a Hugging Face Hub snapshot.
READS_HUB_SNAPSHOT = re.compile(r"models--|_SNAPSHOT\b|MODELS_ROOT|snapshot_download|hf_hub_download")

# The shared materializers: each downloads a missing pinned snapshot into the runner hub cache.
SHARED_MATERIALIZER = re.compile(
    r"ensure_model_snapshot"  # CLI, ensure_snapshot, download_to_runner_cache
    r"|prepare_\w+_proof_assets"  # model-proof setups built on ensure_snapshot
    r"|sensenova_loop_assets"
    r"|ios_prequant"
    r"|qwen38_bonsai_terminal\.py provision"
)

# Download locations that are never the runner's hub cache.
FORBIDDEN_LOCATIONS = (
    (re.compile(r"\blocal_dir\s*="), "downloads into a per-run local_dir"),
    (re.compile(r"sceneworks-rw-weights/hub"), "uses the retired task-owned hub"),
    (re.compile(r"~/\.cache/huggingface"), "uses the user-profile Hugging Face cache"),
    (re.compile(r"USERPROFILE[^\n]{0,60}huggingface", re.IGNORECASE), "uses the user-profile Hugging Face cache"),
)

DIRECT_HUB_CALL = re.compile(r"\b(snapshot_download|hf_hub_download)\(")

# Jobs that name a snapshot but load no model weights, each with the reason.
NO_WEIGHTS = {
    ("model-inference-proofs.yml", "inventory"): "reports which FLUX.1 inputs are present; loads no model",
}

WORKFLOWS = {
    "real-weights.yml": lambda: real_weights_workflow.inline_text(),
    "real-weights-yue.yml": lambda: (ROOT / ".github/workflows/real-weights-yue.yml").read_text(encoding="utf-8"),
    "model-inference-proofs.yml": lambda: (ROOT / ".github/workflows/model-inference-proofs.yml").read_text(
        encoding="utf-8"
    ),
}


def violations(workflow: str, jobs: dict) -> list[str]:
    found = []
    for name, job in jobs.items():
        body = yaml.safe_dump(job, width=10**6)
        if (
            READS_HUB_SNAPSHOT.search(body)
            and not SHARED_MATERIALIZER.search(body)
            and (workflow, name) not in NO_WEIGHTS
        ):
            found.append(f"{workflow}:{name} reads a Hub snapshot without a shared materializer")
        for pattern, reason in FORBIDDEN_LOCATIONS:
            if pattern.search(body):
                found.append(f"{workflow}:{name} {reason}")
        for call in DIRECT_HUB_CALL.finditer(body):
            arguments = body[call.end() : call.end() + 600].split("))", 1)[0]
            if "cache_dir" not in arguments:
                found.append(f"{workflow}:{name} calls {call.group(1)} without cache_dir")
    return found


class RealWeightDownloadPolicyTests(unittest.TestCase):
    def test_every_real_weight_workflow_downloads_only_into_the_runner_hub_cache(self) -> None:
        found = []
        for workflow, read in WORKFLOWS.items():
            found += violations(workflow, yaml.safe_load(read())["jobs"])
        self.assertEqual(found, [])

    def test_a_job_reading_a_snapshot_without_a_materializer_is_rejected(self) -> None:
        jobs = {"lane": {"env": {"X_SNAPSHOT": "${{ vars.X_SNAPSHOT }}"}, "steps": [{"run": "cargo test"}]}}
        self.assertEqual(
            violations("real-weights.yml", jobs),
            ["real-weights.yml:lane reads a Hub snapshot without a shared materializer"],
        )
        jobs["lane"]["steps"].insert(
            0, {"run": "python3.12 scripts/release/ensure_model_snapshot.py --model x --export X_SNAPSHOT"}
        )
        self.assertEqual(violations("real-weights.yml", jobs), [])

    def test_downloads_outside_the_runner_hub_cache_are_rejected(self) -> None:
        cases = {
            'snapshot_download(repo_id=r, local_dir=os.environ["X"])': "downloads into a per-run local_dir",
            'lane_hub="$HOME/sceneworks-rw-weights/hub"': "uses the retired task-owned hub",
            "X=~/.cache/huggingface/hub/models--a--b": "uses the user-profile Hugging Face cache",
        }
        for run, reason in cases.items():
            jobs = {"lane": {"steps": [{"run": "python3.12 scripts/release/ensure_model_snapshot.py\n" + run}]}}
            self.assertIn(f"real-weights.yml:lane {reason}", violations("real-weights.yml", jobs), run)

    def test_a_direct_hub_call_must_name_the_cache(self) -> None:
        run = "from ensure_model_snapshot import runner_hub_cache\nhf_hub_download(repo_id=r, filename=f)\n"
        self.assertEqual(
            violations("real-weights.yml", {"lane": {"steps": [{"run": run}]}}),
            ["real-weights.yml:lane calls hf_hub_download without cache_dir"],
        )
        run = run.replace("filename=f)", "filename=f, cache_dir=str(runner_hub_cache()))")
        self.assertEqual(violations("real-weights.yml", {"lane": {"steps": [{"run": run}]}}), [])


if __name__ == "__main__":
    unittest.main()
