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


def run_config(phases: str, **extra: str) -> tuple[int, dict[str, str], str]:
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
            **extra,
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
        self.assertEqual(values["phases"], ",a2,b,", "nf is never a default phase; a1/a3 neither (v3)")
        code, values, log = run_config("nf,a2")
        self.assertEqual(code, 0, log)
        self.assertEqual(values["phases"], ",a2,nf,", "run order is fixed: nf runs last")
        code, _, log = run_config("noise")
        self.assertNotEqual(code, 0)
        self.assertIn("unknown phase 'noise'", log)

    def test_a3_is_refused_and_a1_is_opt_in_at_schedule_v3(self):
        """sc-20688: an A1 campaign cannot complete at schedule v3 (llama8b-fit-boundary's bf16
        reference exceeds the child cap), so A3, which binds a published A1 at the same source
        closure, is refused before a self-hosted job queues; A1 alone still resolves."""
        for phases in ("a3", "a1,a3,a2"):
            code, _, log = run_config(phases)
            self.assertNotEqual(code, 0, phases)
            self.assertIn("phase a3 cannot run at SC-20671 schedule v3", log)
        code, values, log = run_config("a1")
        self.assertEqual(code, 0, log)
        self.assertEqual(values["phases"], ",a1,")

    def test_nf_only_coordinate_selects_one_scheduled_row(self):
        row = "llama-fit-boundary-single-chunked-cold"
        code, values, log = run_config("nf", DISPATCH_NF_ONLY_COORDINATE=row)
        self.assertEqual(code, 0, log)
        self.assertEqual(values["nf_only_coordinate"], row)
        code, values, log = run_config("nf")
        self.assertEqual(code, 0, log)
        self.assertEqual(values["nf_only_coordinate"], "")
        code, _, log = run_config("nf", DISPATCH_NF_ONLY_COORDINATE="llama-unknown")
        self.assertNotEqual(code, 0)
        self.assertIn("nf_only_coordinate must be empty or one scheduled SC-20671 coordinate", log)
        config = (KV_POC / "config.sh").read_text(encoding="utf-8")
        self.assertIn('nf_only_coordinate="$(read_key nf_only_coordinate "")"', config, "the run json carries it")
        workflow = yaml.safe_load(WORKFLOW.read_text(encoding="utf-8"))
        self.assertIn("nf_only_coordinate", workflow[True]["workflow_dispatch"]["inputs"])
        self.assertEqual(workflow["jobs"]["config"]["outputs"]["nf_only_coordinate"],
                         "${{ steps.resolve.outputs.nf_only_coordinate }}")
        resolve = next(step for step in workflow["jobs"]["config"]["steps"] if step.get("id") == "resolve")
        self.assertEqual(resolve["env"]["DISPATCH_NF_ONLY_COORDINATE"], "${{ inputs.nf_only_coordinate }}")
        self.assertEqual(workflow["jobs"]["nf"]["env"]["KV_NF_ONLY_COORDINATE"],
                         "${{ needs.config.outputs.nf_only_coordinate }}")

    def test_phase_script_launches_the_guarded_noise_floor_parent(self):
        phase = (KV_POC / "phase.sh").read_text(encoding="utf-8")
        self.assertIn('a1|b|nf|c|d|d-control) values="-"', phase)
        self.assertIn('RESUME="$R/sc20669-noise-floor$ONLY-resume"; OUT="$R/evidence/sc20669-noise-floor$ONLY"', phase)
        self.assertIn('a1|a2|a3|nf) phase_policy="$F/policies/llm.json"', phase)
        launch = phase[phase.rindex("    nf)\n", 0, phase.index("\"$F/sc20671_kv_baseline\" noise-floor-parent")):]
        launch = launch[: launch.index(";;")]
        for argument in (
            '"$F/sc20671_kv_baseline" noise-floor-parent "${LLM_ARGS[@]}"',
            '--stop-file "$CTL/stop-requested"',
            '--resume-dir "$RESUME"',
            '--out "$OUT"',
        ):
            self.assertIn(argument, launch)

    def test_nf_command_line_resolves_every_rows_pinned_snapshots(self):
        """Run 37003931977: the nf launch must name resolvable pinned snapshots for every row. It
        passes the A1/A2 snapshot arguments verbatim (the worker loads the 4-bit candidate; the
        bf16 reference flags ride along unused), resolved exactly like A1/A2's."""
        pins = [
            line.split("\t")
            for line in (KV_POC / "models.tsv").read_text(encoding="utf-8").splitlines()
            if line and not line.startswith("#")
        ]
        phase = (KV_POC / "phase.sh").read_text(encoding="utf-8")
        llm_args = next(line for line in phase.splitlines() if line.startswith("LLM_ARGS_TEXT="))
        launch = phase[phase.rindex("    nf)\n", 0, phase.index("\"$F/sc20671_kv_baseline\" noise-floor-parent")):]
        launch = launch[: launch.index(";;")]
        launch = "\n".join(line for line in launch.splitlines()[1:] if not line.strip().startswith("#"))
        with tempfile.TemporaryDirectory() as tmp:
            hub = Path(tmp) / "hub"
            for repo, revision, name, *_ in pins:
                file = hub / f"models--{repo.replace('/', '--')}" / "snapshots" / revision / name
                file.parent.mkdir(parents=True, exist_ok=True)
                file.write_bytes(b"")
            script = "\n".join([
                'source "$KV_POC/common.sh"',
                llm_args,
                "LLM_ARGS=($LLM_ARGS_TEXT)",
                'run_cmd() { printf "%s\\n" "$@"; }',
                'CTL=/ctl RESUME=/resume OUT=/out',
                launch,
            ])
            env = {
                **os.environ,
                "KV_POC": str(KV_POC),
                "INFERENCE_SHA": "a" * 40,
                "SCENEWORKS_SHA": "b" * 40,
                "KV_POC_ROOT": str(Path(tmp) / "root"),
                "KV_POC_HF_HUB": str(hub),
                "KV_EXPECTED_RUNNER": "",
            }
            row = "llama-fit-boundary-single-chunked-cold"
            only = subprocess.run(
                ["bash", "-c", script], env={**env, "KV_NF_ONLY_COORDINATE": row},
                capture_output=True, text=True, encoding="utf-8", check=False,
            )
            self.assertEqual(only.returncode, 0, only.stderr)
            only_argv = only.stdout.splitlines()
            self.assertEqual(only_argv[only_argv.index("--only-coordinate") + 1], row)
            done = subprocess.run(
                ["bash", "-c", script], env=env, capture_output=True, text=True,
                encoding="utf-8", check=False,
            )
            self.assertEqual(done.returncode, 0, done.stderr)
            argv = done.stdout.splitlines()
            self.assertNotIn("--only-coordinate", argv, "all sixteen rows by default")
            self.assertEqual(argv[1], "noise-floor-parent")
            flags = dict(zip(argv[2::2], argv[3::2]))
            expected = {
                "--llama-snapshot": "mlx-community/Llama-3.2-3B-Instruct-4bit",
                "--llama-fp32-reference-snapshot": "mlx-community/Llama-3.2-3B-Instruct-bf16",
                "--qwen-snapshot": "mlx-community/Qwen3-1.7B-4bit",
                "--qwen-fp32-reference-snapshot": "mlx-community/Qwen3-1.7B-bf16",
                "--llama8b-snapshot": "mlx-community/Llama-3.1-8B-Instruct-4bit",
                "--llama8b-fp32-reference-snapshot": "mlx-community/Meta-Llama-3.1-8B-Instruct-bf16",
                "--qwen8b-snapshot": "mlx-community/Qwen3-8B-4bit",
                "--qwen8b-fp32-reference-snapshot": "mlx-community/Qwen3-8B-bf16",
            }
            for flag, repo in expected.items():
                self.assertIn(flag, flags, f"nf launch lacks {flag}")
                snapshot = Path(flags[flag])
                self.assertTrue(snapshot.is_dir(), f"{flag} {snapshot} does not resolve")
                pinned = [name for pin_repo, _, name, *_ in pins if pin_repo == repo]
                self.assertTrue(pinned, repo)
                for name in pinned:
                    self.assertTrue((snapshot / name).is_file(), f"{flag} lacks pinned {name}")
            for flag in ("--prompt-file", "--safety-policy", "--stop-file", "--resume-dir", "--out"):
                self.assertIn(flag, flags)

    def test_w1_disk_precheck_needs_missing_bytes_plus_reserve_and_never_downloads_short(self):
        """build.sh's W1 stage refuses before any download when the hub volume's free space is
        under the missing pinned bytes (present right-size files skipped, wrong-size ones counted)
        plus the reserve, and logs free and needed bytes either way."""
        import contextlib
        import io
        from unittest import mock

        spec = importlib.util.spec_from_file_location("kv_poc_models", KV_POC / "models.py")
        models = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(models)
        gib = 1 << 30
        with tempfile.TemporaryDirectory() as tmp:
            hub = Path(tmp)
            pins_file = hub / "pins.tsv"
            pins_file.write_text(
                "\n".join([
                    "org/a\t" + "1" * 40 + "\tpresent.bin\t4\t" + "0" * 64,
                    "org/a\t" + "1" * 40 + "\twrong.bin\t" + str(3 * gib) + "\t" + "0" * 64,
                    "org/b\t" + "2" * 40 + "\tmissing.bin\t" + str(2 * gib) + "\t" + "0" * 64,
                ]) + "\n",
                encoding="utf-8",
            )
            snapshot = models.snapshot_dir(hub, "org/a", "1" * 40)
            snapshot.mkdir(parents=True)
            (snapshot / "present.bin").write_bytes(b"1234")
            (snapshot / "wrong.bin").write_bytes(b"x")
            pins = models.load_pins(pins_file)
            need = 5 * gib  # wrong.bin + missing.bin; present.bin is skipped

            def precheck(free):
                usage = mock.Mock(free=free)
                out = io.StringIO()
                with mock.patch.object(models.shutil, "disk_usage", return_value=usage), \
                        contextlib.redirect_stdout(out):
                    return models.disk_precheck(hub, pins, 20.0), out.getvalue()

            fits, log = precheck(need + 20 * gib)
            self.assertTrue(fits, log)
            self.assertIn(f"missing or wrong-size: 2 files, {need} bytes", log)
            self.assertIn(f"free on the hub volume ({hub}): {need + 20 * gib} bytes", log)
            fits, log = precheck(need + 20 * gib - 1)
            self.assertFalse(fits, log)
            self.assertIn("::error title=not enough disk for the pinned W1 snapshots::", log)
            self.assertIn("SHORTFALL", log)
            # main() refuses before importing huggingface_hub or downloading anything.
            argv = ["models.py", "--hub", str(hub), "--pins", str(pins_file), "--reserve-gib", "20"]
            usage = mock.Mock(free=need)
            with mock.patch.object(models.shutil, "disk_usage", return_value=usage), \
                    mock.patch.object(models.sys, "argv", argv), \
                    mock.patch.dict("sys.modules", {"huggingface_hub": None}), \
                    contextlib.redirect_stdout(io.StringIO()):
                self.assertEqual(models.main(), 1)
        build = (KV_POC / "build.sh").read_text(encoding="utf-8")
        self.assertIn('--reserve-gib "${KV_W1_DISK_RESERVE_GIB:-20}"', build)

    def test_sixteen_row_a1_a2_budgets_doubled_within_the_row_deadline(self):
        """Schedule v3 doubled the rows, so A1/A2 soft budgets doubled (330 -> 660 per
        invocation); each hard timeout stays budget x invocations + 300-min row deadline + 20."""
        jobs = yaml.safe_load(WORKFLOW.read_text(encoding="utf-8"))["jobs"]
        for job, invocations in (("a1", 1), ("a2", 2)):
            self.assertEqual(jobs[job]["env"]["KV_SOFT_BUDGET_MIN"], "660", job)
            self.assertEqual(jobs[job]["timeout-minutes"], invocations * 660 + 300 + 20, job)

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
        self.assertEqual(job["env"]["KV_SOFT_BUDGET_MIN"], "240")
        self.assertEqual(job["timeout-minutes"], 240 + 300 + 20)

    def test_models_tsv_mirrors_every_campaign_pin(self):
        """build.sh provisions exactly models.tsv on the Mac, and the campaign parent refuses any
        snapshot that lacks one of campaign.rs's pinned files, so the two must list the same
        (repository, revision, path, bytes, sha256) rows for every scheduled family."""
        import re

        source = (ROOT / "crates/llm/mlx-llm/src/campaign.rs").read_text(encoding="utf-8")
        files = {
            name: re.findall(r'path: "([^"]+)",\s*bytes: ([\d_]+),\s*sha256: "([0-9a-f]{64})"', body)
            for name, body in re.findall(
                r"const ([A-Z0-9_]+_FILES): &\[PinnedSnapshotFile\] = &\[(.*?)\n\];", source, re.S
            )
        }
        specs = re.findall(
            r'repository: "([^"]+)",\s*revision: "([0-9a-f]{40})",(?:.|\n)*?required_files: ([A-Z0-9_]+_FILES),',
            source,
        )
        expected = [
            (repo, revision, path, size.replace("_", ""), sha256)
            for repo, revision, files_name in specs
            for path, size, sha256 in files[files_name]
        ]
        pins = [
            tuple(line.split("\t"))
            for line in (KV_POC / "models.tsv").read_text(encoding="utf-8").splitlines()
            if line and not line.startswith("#")
        ]
        self.assertEqual(len({(repo, revision) for repo, revision, *_ in expected}), 8)
        # Ordered lists, not sets: a duplicated pin row (or a duplicated source entry) fails too.
        self.assertEqual(sorted(pins), sorted(expected))

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
        row["denseTiming"] = {"prefillMs": 91000.0, "firstTokenMs": 91500.0, "decodeTokensPerSecond": 30.5,
                              "decodeTokensPerSecondCoefficientOfVariation": 0.01}
        markdown = summarize.noise_floor_markdown([row], summary)
        self.assertIn("| `llama-fit-boundary-single-chunked-cold` | 91000.0 | 91500.0 | 30.5 | 0.01 |", markdown)
        self.assertIn("| `llama-fit-boundary-single-chunked-cold` | chunked-prefill | 0.998 | 2 | 0.013 |", markdown)
        self.assertIn("the dense arm itself misses the greedy and perplexity threshold", markdown)
        summary["controls"]["multi-turn-repeat"] = {
            "minAgreement": 0.996, "minAgreementRow": row["coordinate"],
            "maxAbsPerplexityDelta": 0.0, "maxAbsPerplexityDeltaRow": row["coordinate"],
            "greedyThresholdWithinDenseFloor": False, "perplexityThresholdWithinDenseFloor": False,
            "multiTurnThresholdWithinDenseFloor": True,
        }
        self.assertIn("the dense arm itself misses the multi-turn threshold",
                      summarize.noise_floor_markdown([row], summary))
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
