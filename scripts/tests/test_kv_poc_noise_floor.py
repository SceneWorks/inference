"""The SC-20669 dense noise-floor phase (`nf`) of kv-poc-campaign.yml.

`nf` is a W1 phase that runs only when listed (`"phases": "nf"` runs it alone); its job runs
`phase.sh run nf` after the other W1 phases and uploads its evidence; phase.sh launches
`sc20671_kv_baseline noise-floor-parent` under the LLM policy; the job summary renders its
per-row and worst-case tables.
"""

import importlib.util
import os
import subprocess
import tempfile
import unittest
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[2]
KV_POC = ROOT / ".github" / "kv-poc"
WORKFLOW = ROOT / ".github" / "workflows" / "kv-poc-campaign.yml"


def run_config(phases: str) -> tuple[int, dict[str, str], str]:
    with tempfile.TemporaryDirectory() as tmp:
        output = Path(tmp) / "output"
        summary = Path(tmp) / "summary"
        env = {
            **os.environ,
            "EVENT_NAME": "workflow_dispatch",
            "DISPATCH_MODE": "w1",
            "DISPATCH_INFERENCE_REF": "a" * 40,
            "DISPATCH_SCENEWORKS_REF": "b" * 40,
            "DISPATCH_RUNNER_LABEL": "rw-krea",
            "DISPATCH_PHASES": phases,
            "DISPATCH_BASELINE_EVIDENCE_REF": "",
            "GITHUB_OUTPUT": str(output),
            "GITHUB_STEP_SUMMARY": str(summary),
        }
        done = subprocess.run(
            ["bash", str(KV_POC / "config.sh")],
            cwd=ROOT,
            env=env,
            capture_output=True,
            text=True,
            encoding="utf-8",
            check=False,
        )
        values = {}
        if output.exists():
            for line in output.read_text(encoding="utf-8").splitlines():
                key, _, value = line.partition("=")
                values[key] = value
        return done.returncode, values, done.stdout + done.stderr


def load_summarize():
    spec = importlib.util.spec_from_file_location("kv_poc_summarize", KV_POC / "summarize.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class NoiseFloorPhaseTests(unittest.TestCase):
    def test_nf_is_an_opt_in_w1_phase_in_fixed_order(self):
        code, values, log = run_config("nf")
        self.assertEqual(code, 0, log)
        self.assertEqual(values["phases"], ",nf,")
        code, values, log = run_config("")
        self.assertEqual(code, 0, log)
        self.assertEqual(values["phases"], ",a1,a3,a2,b,", "nf is never a default phase")
        code, values, log = run_config("nf,a2")
        self.assertEqual(code, 0, log)
        self.assertEqual(values["phases"], ",a2,nf,", "run order is fixed: nf runs last")
        code, _, log = run_config("noise")
        self.assertNotEqual(code, 0)
        self.assertIn("unknown phase 'noise'", log)

    def test_phase_script_launches_the_guarded_noise_floor_parent(self):
        phase = (KV_POC / "phase.sh").read_text(encoding="utf-8")
        self.assertIn('a1|b|nf|c|d|d-control) values="-"', phase)
        self.assertIn('nf) RESUME="$R/sc20669-noise-floor-resume"; OUT="$R/evidence/sc20669-noise-floor"', phase)
        self.assertIn('a1|a2|a3|nf) phase_policy="$F/policies/llm.json"', phase)
        launch = phase[phase.index("    nf)\n      run_cmd"):]
        launch = launch[: launch.index(";;")]
        for argument in (
            '"$F/sc20671_kv_baseline" noise-floor-parent',
            '--llama-snapshot "$LQ"',
            '--qwen-snapshot "$QQ"',
            '--prompt-file "$F/inputs/prompt.txt"',
            '--safety-policy "$F/policies/llm.json"',
            '--stop-file "$CTL/stop-requested"',
            '--resume-dir "$RESUME"',
            '--out "$OUT"',
        ):
            self.assertIn(argument, launch)

    def test_workflow_runs_collects_and_uploads_nf_after_every_other_w1_phase(self):
        workflow = yaml.safe_load(WORKFLOW.read_text(encoding="utf-8"))
        job = workflow["jobs"]["nf"]
        self.assertEqual(job["needs"], ["config", "build", "a1", "a3", "a2", "b"])
        self.assertIn("contains(needs.config.outputs.phases, ',nf,')", job["if"])
        for phase in ("a1", "a3", "a2", "b"):
            self.assertIn(f"needs.{phase}.outputs.stopped != 'true'", job["if"])
        runs = [step.get("run", "") for step in job["steps"]]
        self.assertIn("bash .github/kv-poc/phase.sh run nf", runs)
        self.assertIn("bash .github/kv-poc/phase.sh collect nf", runs)
        upload = next(step for step in job["steps"] if "upload-artifact" in step.get("uses", ""))
        self.assertEqual(upload["with"]["name"], "kv-poc-nf-${{ github.run_id }}-${{ github.run_attempt }}")
        self.assertEqual(upload["with"]["path"], "${{ runner.temp }}/kv-poc-artifact/nf/")
        self.assertEqual(upload["if"], "${{ always() }}")
        # Soft budget + the llm.json row deadline + 20 minutes, like every W1 job.
        self.assertEqual(job["env"]["KV_SOFT_BUDGET_MIN"], "120")
        self.assertEqual(job["timeout-minutes"], 120 + 300 + 20)

    def test_summary_renders_rows_and_the_worst_case(self):
        summarize = load_summarize()
        row = {
            "kind": summarize.NOISE_FLOOR_ROW_KIND,
            "coordinate": "llama-fit-boundary-single-chunked-cold",
            "controls": {
                "chunked-prefill": {"agreement": 0.998, "flipCount": 2, "perplexityDelta": 0.013},
                "one-shot-repeat": {"agreement": 1.0, "flipCount": 0, "perplexityDelta": 0.0},
            },
        }
        summary = {
            "kind": summarize.NOISE_FLOOR_SUMMARY_KIND,
            "rows": 1,
            "thresholds": {"greedyTokenAgreement": 0.999, "perplexityDelta": 0.01},
            "controls": {
                "chunked-prefill": {
                    "minAgreement": 0.998,
                    "minAgreementRow": row["coordinate"],
                    "maxAbsPerplexityDelta": 0.013,
                    "maxAbsPerplexityDeltaRow": row["coordinate"],
                    "greedyThresholdWithinDenseFloor": True,
                    "perplexityThresholdWithinDenseFloor": True,
                },
            },
        }
        markdown = summarize.noise_floor_markdown([row], summary)
        self.assertIn("| `llama-fit-boundary-single-chunked-cold` | chunked-prefill | 0.998 | 2 | 0.013 |", markdown)
        self.assertIn("the dense arm itself misses the greedy and perplexity threshold", markdown)
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp) / "evidence" / "sc20669-noise-floor"
            out.mkdir(parents=True)
            import json

            (out / "llama-fit-boundary-single-chunked-cold.json").write_text(json.dumps(row), encoding="utf-8")
            (out / "summary.json").write_text(json.dumps(summary), encoding="utf-8")
            from contextlib import redirect_stdout
            from io import StringIO

            buffer = StringIO()
            with redirect_stdout(buffer):
                summarize.report("nf", Path(tmp), [out])
            self.assertIn("Dense noise floor (SC-20669)", buffer.getvalue())
            self.assertIn("chunked-prefill", buffer.getvalue())


if __name__ == "__main__":
    unittest.main()
