"""Fail-closed tests for the sc-24446 decode-speedups campaign runner
(`scripts/release/speculative_bench_campaign.py`)."""

from __future__ import annotations

import contextlib
import importlib.util
import io
import json
import os
import sys
import tempfile
import textwrap
import unittest
from pathlib import Path
from unittest import mock

SCRIPT = Path(__file__).resolve().parents[1] / "release" / "speculative_bench_campaign.py"
SPEC = importlib.util.spec_from_file_location("speculative_bench_campaign", SCRIPT)
assert SPEC and SPEC.loader
campaign = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(campaign)

REPOSITORY = Path(__file__).resolve().parents[2]
EPIC_SHA = "a" * 40
SNAPSHOTS = {
    "DECODE_BENCH_SNAPSHOT_QWEN38": "E:\\huggingface\\qwen38",
    "DECODE_BENCH_SNAPSHOT_BONSAI_MLX": "/models/bonsai-mlx",
}

# A stand-in for `cargo test …`: writes the document the real entry point would, shaped by FAKE_*,
# and records the environment it was given.
FAKE_CARGO = textwrap.dedent(
    """
    import json, os, sys
    out = os.environ["SPECULATIVE_BENCH_OUTPUT"]
    run = "baseline" if "speculative_bench_baseline" in sys.argv else "epic"
    with open(os.environ["FAKE_RECORD"], "a", encoding="utf-8") as record:
        record.write(json.dumps({"run": run, "argv": sys.argv[1:], "env": dict(os.environ)}) + "\\n")
    mode = os.environ.get("FAKE_MODE", "ok")
    if mode == "exit":
        sys.exit(101)
    sha = os.environ["SPECULATIVE_BENCH_GIT_SHA"]
    if mode == "wrong_sha":
        sha = "b" * 40
    changes = []
    if run == "baseline":
        changes = ["?? crates/llm/candle-llm/tests/speculative_bench_baseline.rs"]
    if mode == "dirty":
        changes = changes + [" M crates/llm/candle-llm/src/lib.rs"]
    thinking = os.environ["SPECULATIVE_BENCH_THINKING"]
    if mode == "wrong_thinking":
        thinking = "on"
    doc = {
        "schema": "sceneworks.decode-speedups.baseline/3",
        "thinking": {"setting": thinking},
        "provenance": {"git": {"sha": sha, "dirty": bool(changes), "changes": changes,
                               "source": "git"}},
        "rows": [{"prompt_id": "chat"}],
    }
    if mode != "no_document":
        with open(out, "x", encoding="utf-8") as sink:
            json.dump(doc, sink)
    print("test result: ok. 0 passed" if mode == "zero" else "test result: ok. 1 passed")
    """
)


def plan(matrix, environ=None):
    return campaign.plan_matrix(json.dumps(matrix), {**SNAPSHOTS, **(environ or {})})


class PlanTests(unittest.TestCase):
    def test_the_default_dispatch_matrix_plans_both_runs_on_the_qwen38_variable(self) -> None:
        workflow = (REPOSITORY / ".github" / "workflows" / "real-weights.yml").read_text(
            encoding="utf-8"
        )
        import yaml

        default = yaml.safe_load(workflow)[True]["workflow_dispatch"]["inputs"][
            "decode_bench_matrix"
        ]["default"]
        planned = campaign.plan_matrix(default, SNAPSHOTS)
        self.assertEqual(planned["pre_epic_sha"], campaign.PRE_EPIC_SHA)
        (row,) = planned["rows"]
        self.assertEqual(row["runs"], ["epic", "baseline"])
        self.assertEqual(
            row["env"],
            {
                "SPECULATIVE_BENCH_SNAPSHOT": SNAPSHOTS["DECODE_BENCH_SNAPSHOT_QWEN38"],
                "SPECULATIVE_BENCH_FORMAT": "bf16",
                "SPECULATIVE_BENCH_THINKING": "default",
            },
        )

    def test_every_knob_maps_onto_its_harness_variable(self) -> None:
        (row,) = plan(
            {
                "rows": [
                    {
                        "id": "bonsai-q4.graphs",
                        "snapshot": "bonsai-mlx",
                        "runs": ["baseline", "epic"],
                        "format": "q4",
                        "options": ["off", {"proposer": "mtp", "depth": 3}],
                        "sampling": {"temperature": 0.7},
                        "thinking": "low",
                        "new_tokens": 64,
                        "repeats": 5,
                        "warmup": False,
                        "prefix_cache_bytes": 0,
                        "model": "Bonsai@q4",
                        "switches": {"CANDLE_LLM_CUDA_GRAPHS": "1"},
                    }
                ]
            }
        )["rows"]
        self.assertEqual(row["runs"], ["epic", "baseline"], "epic always runs first")
        self.assertEqual(
            row["env"],
            {
                "SPECULATIVE_BENCH_SNAPSHOT": "/models/bonsai-mlx",
                "SPECULATIVE_BENCH_FORMAT": "q4",
                "SPECULATIVE_BENCH_OPTIONS": '["off",{"proposer":"mtp","depth":3}]',
                "SPECULATIVE_BENCH_SAMPLING": '{"temperature":0.7}',
                "SPECULATIVE_BENCH_THINKING": "low",
                "SPECULATIVE_BENCH_NEW_TOKENS": "64",
                "SPECULATIVE_BENCH_REPEATS": "5",
                "SPECULATIVE_BENCH_WARMUP": "0",
                "SPECULATIVE_BENCH_PREFIX_CACHE_BYTES": "0",
                "SPECULATIVE_BENCH_MODEL": "Bonsai@q4",
                "CANDLE_LLM_CUDA_GRAPHS": "1",
            },
        )
        (epic,) = plan(
            {
                "rows": [
                    {
                        "id": "draft",
                        "snapshot": "C:\\models\\target",
                        "runs": ["epic"],
                        "draft": "C:\\models\\draft",
                        "prefix_cache_bytes": "default",
                        "sampling": "greedy",
                    }
                ]
            }
        )["rows"]
        self.assertEqual(epic["env"]["SPECULATIVE_BENCH_DRAFT"], "C:\\models\\draft")
        self.assertEqual(epic["env"]["SPECULATIVE_BENCH_PREFIX_CACHE_BYTES"], "default")
        self.assertEqual(epic["env"]["SPECULATIVE_BENCH_SAMPLING"], '"greedy"')

    def test_the_switches_are_the_harness_recorded_set(self) -> None:
        source = (
            REPOSITORY / "crates/contracts/core-llm/core-llm-testkit/src/speculative.rs"
        ).read_text(encoding="utf-8")
        block = source.split("pub const BENCH_SWITCHES: [&str; 9] = [", 1)[1].split("];", 1)[0]
        self.assertEqual(
            tuple(line.strip().strip(",").strip('"') for line in block.strip().splitlines()),
            campaign.SWITCHES,
        )

    def test_every_malformed_matrix_is_refused_before_anything_runs(self) -> None:
        good = {"id": "r", "snapshot": "qwen38"}
        cases = {
            "not json": "{",
            "extra top key": {"rows": [good], "more": 1},
            "no rows": {"rows": []},
            "rows not a list": {"rows": good},
            "too many rows": {"rows": [{**good, "id": f"r{i}"} for i in range(65)]},
            "row not object": {"rows": ["r"]},
            "bad id": {"rows": [{**good, "id": "R 1"}]},
            "id path": {"rows": [{**good, "id": "../x"}]},
            "duplicate id": {"rows": [good, good]},
            "unknown key": {"rows": [{**good, "env": {}}]},
            "relative snapshot": {"rows": [{**good, "snapshot": "models/qwen"}]},
            "unknown alias": {"rows": [{**good, "snapshot": "qwen9"}]},
            "empty runs": {"rows": [{**good, "runs": []}]},
            "unknown run": {"rows": [{**good, "runs": ["head"]}]},
            "repeated run": {"rows": [{**good, "runs": ["epic", "epic"]}]},
            "bad format": {"rows": [{**good, "format": "fp8"}]},
            "bad thinking": {"rows": [{**good, "thinking": "high"}]},
            "empty options": {"rows": [{**good, "options": []}]},
            "options string": {"rows": [{**good, "options": "off"}]},
            "sampling string": {"rows": [{**good, "sampling": "hot"}]},
            "bool count": {"rows": [{**good, "repeats": True}]},
            "negative count": {"rows": [{**good, "new_tokens": -1}]},
            "string warmup": {"rows": [{**good, "warmup": "0"}]},
            "unknown switch": {"rows": [{**good, "switches": {"CANDLE_LLM_GRAPHS": "1"}}]},
            "non-string switch": {"rows": [{**good, "switches": {"CANDLE_LLM_CUDA_GRAPHS": 1}}]},
            "multi-line model": {"rows": [{**good, "model": "a\nb"}]},
            "relative draft": {"rows": [{**good, "runs": ["epic"], "draft": "draft"}]},
            "draft on baseline": {"rows": [{**good, "draft": "/models/draft"}]},
            "mtp head on baseline": {"rows": [{**good, "mtp_head": "/models/head"}]},
            "cache on baseline": {"rows": [{**good, "prefix_cache_bytes": 1024}]},
            "default cache on baseline": {"rows": [{**good, "prefix_cache_bytes": "default"}]},
        }
        for name, matrix in cases.items():
            with self.subTest(name):
                text = matrix if isinstance(matrix, str) else json.dumps(matrix)
                with self.assertRaises(campaign.MatrixError):
                    campaign.plan_matrix(text, SNAPSHOTS)
        with self.assertRaisesRegex(campaign.MatrixError, "DECODE_BENCH_SNAPSHOT_QWEN38"):
            campaign.plan_matrix(json.dumps({"rows": [good]}), {})

    def test_the_plan_command_reads_the_untrusted_matrix_only_from_the_environment(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "plan.json"
            environ = {**SNAPSHOTS, "DECODE_BENCH_MATRIX": '{"rows": [{"id": "r", "snapshot": "x"}]}'}
            with mock.patch.dict(os.environ, environ), contextlib.redirect_stderr(io.StringIO()):
                self.assertEqual(campaign.main(["plan", "--output", str(output)]), 1)
            self.assertFalse(output.exists())
            environ["DECODE_BENCH_MATRIX"] = '{"rows": [{"id": "r", "snapshot": "qwen38"}]}'
            with mock.patch.dict(os.environ, environ), contextlib.redirect_stdout(io.StringIO()):
                self.assertEqual(campaign.main(["plan", "--output", str(output)]), 0)
            self.assertEqual(json.loads(output.read_text(encoding="utf-8"))["rows"][0]["id"], "r")


class RunTests(unittest.TestCase):
    def setUp(self) -> None:
        self.directory = Path(self.enterContext(tempfile.TemporaryDirectory()))
        fake = self.directory / "fake_cargo.py"
        fake.write_text(FAKE_CARGO, encoding="utf-8")
        self.record = self.directory / "record.jsonl"
        commands = {run: [str(fake), *argv] for run, argv in campaign.COMMANDS.items()}
        self.enterContext(mock.patch.object(campaign, "COMMANDS", commands))
        heads = {"epic": EPIC_SHA, "pre-epic": campaign.PRE_EPIC_SHA}
        self.enterContext(
            mock.patch.object(campaign, "git_head", lambda root: heads[Path(root).name])
        )
        for name in ("epic", "pre-epic"):
            (self.directory / name).mkdir()

    def run_plan(self, rows, **fake_env):
        plan_path = self.directory / "plan.json"
        plan_path.unlink(missing_ok=True)
        plan_path.write_text(json.dumps(plan({"rows": rows})), encoding="utf-8")
        output = self.directory / "out"
        environ = {
            "FAKE_RECORD": str(self.record),
            "SPECULATIVE_BENCH_DRAFT": "/stale/draft",
            "CANDLE_LLM_FUSED_KERNELS": "0",
            **fake_env,
        }
        with mock.patch.dict(os.environ, environ), contextlib.redirect_stdout(
            io.StringIO()
        ), contextlib.redirect_stderr(io.StringIO()):
            code = campaign.main(
                [
                    "run",
                    "--plan",
                    str(plan_path),
                    "--epic-root",
                    str(self.directory / "epic"),
                    "--epic-sha",
                    EPIC_SHA,
                    "--pre-epic-root",
                    str(self.directory / "pre-epic"),
                    "--pre-epic-target-dir",
                    "T:\\cargo-target-pre-epic",
                    "--output",
                    str(output),
                    "--cargo",
                    sys.executable,
                ]
            )
        summary = json.loads((output / "summary.json").read_text(encoding="utf-8"))
        records = [
            json.loads(line) for line in self.record.read_text(encoding="utf-8").splitlines()
        ]
        return code, output, summary["runs"], records

    def test_each_row_runs_the_epic_entry_then_the_baseline_driver_in_isolation(self) -> None:
        code, output, runs, records = self.run_plan(
            [
                {"id": "a", "snapshot": "qwen38", "switches": {"CANDLE_LLM_CUDA_GRAPHS": "1"}},
                {"id": "b", "snapshot": "bonsai-mlx", "thinking": "off", "runs": ["epic"]},
            ]
        )
        self.assertEqual(code, 0, runs)
        self.assertEqual(
            [(r["id"], r["run"], r["status"]) for r in runs],
            [("a", "epic", "ok"), ("a", "baseline", "ok"), ("b", "epic", "ok")],
        )
        for name in ("a/epic.json", "a/baseline.json", "b/epic.json", "a/epic.log"):
            self.assertTrue((output / name).is_file(), name)
        self.assertEqual([r["run"] for r in records], ["epic", "baseline", "epic"])
        epic, baseline, _ = records
        self.assertIn("speculative_bench_writes_the_baseline_document", epic["argv"])
        self.assertIn("speculative_bench_baseline_writes_the_document", baseline["argv"])
        for argv in (epic["argv"], baseline["argv"]):
            self.assertEqual(argv[argv.index("--") + 1 :][1:], ["--exact", "--ignored", "--nocapture"])
        # Each run gets the row's knobs and nothing stale from the runner.
        for record in (epic, baseline):
            self.assertNotIn("SPECULATIVE_BENCH_DRAFT", record["env"])
            self.assertNotIn("CANDLE_LLM_FUSED_KERNELS", record["env"])
            self.assertEqual(record["env"]["CANDLE_LLM_CUDA_GRAPHS"], "1")
        self.assertEqual(epic["env"]["SPECULATIVE_BENCH_GIT_SHA"], EPIC_SHA)
        self.assertEqual(baseline["env"]["SPECULATIVE_BENCH_GIT_SHA"], campaign.PRE_EPIC_SHA)
        self.assertEqual(baseline["env"]["CARGO_TARGET_DIR"], "T:\\cargo-target-pre-epic")
        self.assertNotEqual(epic["env"].get("CARGO_TARGET_DIR"), "T:\\cargo-target-pre-epic")

    def test_a_run_that_does_not_count_fails_the_campaign_without_stopping_it(self) -> None:
        for mode in ("exit", "zero", "no_document", "wrong_sha", "wrong_thinking", "dirty"):
            with self.subTest(mode=mode):
                self.record.unlink(missing_ok=True)
                for path in sorted((self.directory / "out").rglob("*"), reverse=True):
                    path.rmdir() if path.is_dir() else path.unlink()
                code, _, runs, records = self.run_plan(
                    [{"id": "a", "snapshot": "qwen38"}, {"id": "b", "snapshot": "qwen38"}],
                    FAKE_MODE=mode,
                )
                self.assertEqual(code, 1)
                self.assertEqual(len(records), 4, "every row still ran")
                self.assertTrue(all(r["status"] == "failed" and r["error"] for r in runs), runs)

    def test_a_checkout_at_the_wrong_commit_is_refused_before_any_run(self) -> None:
        with mock.patch.object(campaign, "git_head", lambda root: "c" * 40):
            plan_path = self.directory / "plan.json"
            plan_path.write_text(
                json.dumps(plan({"rows": [{"id": "a", "snapshot": "qwen38"}]})), encoding="utf-8"
            )
            # With the record wired, a run that started would be recorded.
            environ = {"FAKE_RECORD": str(self.record)}
            with mock.patch.dict(os.environ, environ), contextlib.redirect_stdout(
                io.StringIO()
            ), contextlib.redirect_stderr(io.StringIO()):
                code = campaign.main(
                    [
                        "run",
                        "--plan",
                        str(plan_path),
                        "--epic-root",
                        str(self.directory / "epic"),
                        "--epic-sha",
                        EPIC_SHA,
                        "--pre-epic-root",
                        str(self.directory / "pre-epic"),
                        "--pre-epic-target-dir",
                        "T",
                        "--output",
                        str(self.directory / "out"),
                        "--cargo",
                        sys.executable,
                    ]
                )
        self.assertEqual(code, 1)
        self.assertFalse(self.record.exists())


if __name__ == "__main__":
    unittest.main()
