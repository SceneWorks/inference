"""Fail-closed tests for the sc-24446 decode-speedups campaign runner and comparator
(`scripts/release/speculative_bench_campaign.py`)."""

from __future__ import annotations

import contextlib
import copy
import importlib.util
import io
import json
import os
import shutil
import subprocess
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

# A stand-in for `cargo test …`: a `--no-run` build records itself; a run writes the document the
# real entry point would, shaped by FAKE_MODE, and records the environment it was given.
FAKE_CARGO = textwrap.dedent(
    """
    import json, os, sys
    run = "baseline" if any("speculative_bench_baseline" in a for a in sys.argv) else "epic"
    build = "--no-run" in sys.argv
    with open(os.environ["FAKE_RECORD"], "a", encoding="utf-8") as record:
        record.write(json.dumps({"run": run, "build": build, "argv": sys.argv[1:],
                                 "env": dict(os.environ)}) + "\\n")
    if build:
        sys.exit(int(os.environ.get("FAKE_BUILD_EXIT", "0")))
    out = os.environ["SPECULATIVE_BENCH_OUTPUT"]
    mode = os.environ.get("FAKE_MODE", "ok")
    if mode == "exit":
        sys.exit(101)
    stamp = os.environ.get("SPECULATIVE_BENCH_BUILD_GIT_SHA")
    dirty_stamp = {"0": False, "1": True}.get(os.environ.get("SPECULATIVE_BENCH_BUILD_GIT_DIRTY"))
    sha, source = stamp, "git"
    if stamp is None:
        sha, source = os.environ["SPECULATIVE_BENCH_GIT_SHA"], "SPECULATIVE_BENCH_GIT_SHA"
    if mode == "wrong_sha":
        sha = "b" * 40
    if mode == "stale_build":
        stamp = "b" * 40
    if mode == "wrong_dirty_stamp":
        dirty_stamp = not dirty_stamp
    changes = []
    if run == "baseline":
        changes = ["?? crates/llm/candle-llm/tests/speculative_bench_baseline.rs"]
    if mode == "dirty":
        changes = changes + [" M crates/llm/candle-llm/src/lib.rs"]
    thinking = os.environ["SPECULATIVE_BENCH_THINKING"]
    if mode == "wrong_thinking":
        thinking = "on"
    backend = os.environ["SPECULATIVE_BENCH_BACKEND"]
    if mode == "wrong_backend":
        backend = "candle"
    git = {"sha": sha, "dirty": bool(changes) if source == "git" else None,
           "changes": changes if source == "git" else [], "source": source,
           "build_sha": stamp, "build_dirty": dirty_stamp}
    doc = {
        "schema": "sceneworks.decode-speedups.baseline/4",
        "backend": backend,
        "thinking": {"setting": thinking},
        "provenance": {"git": git},
        "rows": [{"prompt_id": "chat"}],
    }
    if mode == "touch_snapshot":
        with open(os.path.join(os.environ["SPECULATIVE_BENCH_SNAPSHOT"], "config.json"), "a",
                  encoding="utf-8") as f:
            f.write(" ")
    if mode != "no_document":
        with open(out, "x", encoding="utf-8") as sink:
            json.dump(doc, sink)
    if mode == "zero":
        print("test result: ok. 0 passed; 0 failed; 0 ignored")
    elif mode == "two":
        print("test result: ok. 1 passed; 0 failed; 0 ignored")
        print("test result: ok. 1 passed; 0 failed; 0 ignored")
    else:
        print("test result: ok. 1 passed; 0 failed; 0 ignored; 0 measured; 9 filtered out")
    """
)

# A stand-in for the coordinator's memory guard: records that it wrapped the command, then runs it.
FAKE_GUARD = textwrap.dedent(
    """
    import json, os, subprocess, sys
    with open(os.environ["FAKE_RECORD"], "a", encoding="utf-8") as record:
        record.write(json.dumps({"guard": sys.argv[1:]}) + "\\n")
    sys.exit(subprocess.run(sys.argv[1:]).returncode)
    """
)


def plan(matrix, environ=None, lane="cuda"):
    return campaign.plan_matrix(json.dumps(matrix), {**SNAPSHOTS, **(environ or {})}, lane)


def write_snapshot(root: Path) -> Path:
    root.mkdir(parents=True, exist_ok=True)
    (root / "config.json").write_text('{"model_type": "llama"}', encoding="utf-8")
    (root / "model.safetensors").write_bytes(b"\0" * 64)
    return root


def git_repo(root: Path) -> None:
    root.mkdir(parents=True, exist_ok=True)
    for args in (
        ["init", "-q"],
        ["config", "user.email", "t@example.com"],
        ["config", "user.name", "t"],
        ["config", "commit.gpgsign", "false"],
    ):
        subprocess.run(["git", "-C", str(root), *args], check=True)


def commit(root: Path, message: str = "c") -> str:
    subprocess.run(["git", "-C", str(root), "add", "-A"], check=True)
    subprocess.run(["git", "-C", str(root), "commit", "-q", "-m", message], check=True)
    return campaign.git_head(root)


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
        self.assertEqual(planned["lane"], "cuda")
        (row,) = planned["rows"]
        self.assertEqual(row["runs"], ["epic", "baseline"])
        self.assertEqual(row["process_repeats"], 3, "noise is measured over three processes")
        self.assertEqual(
            row["env"],
            {
                "SPECULATIVE_BENCH_SNAPSHOT": SNAPSHOTS["DECODE_BENCH_SNAPSHOT_QWEN38"],
                "SPECULATIVE_BENCH_BACKEND": "candle-cuda",
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
                        "process_repeats": 5,
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
        self.assertEqual(row["runs"], ["epic", "baseline"])
        self.assertEqual(row["process_repeats"], 5)
        self.assertEqual(row["switches"], {"CANDLE_LLM_CUDA_GRAPHS": "1"})
        self.assertEqual(
            row["env"],
            {
                "SPECULATIVE_BENCH_SNAPSHOT": "/models/bonsai-mlx",
                "SPECULATIVE_BENCH_BACKEND": "candle-cuda",
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
        (mlx,) = plan(
            {"rows": [{"id": "m", "snapshot": "/m", "switches": {"MLX_LLM_PIPELINING": "0"}}]},
            lane="mlx",
        )["rows"]
        self.assertEqual(mlx["env"]["SPECULATIVE_BENCH_BACKEND"], "mlx")

    def test_the_switches_are_the_harness_recorded_set_split_by_lane(self) -> None:
        source = (
            REPOSITORY / "crates/contracts/core-llm/core-llm-testkit/src/speculative.rs"
        ).read_text(encoding="utf-8")
        block = source.split("pub const BENCH_SWITCHES: [&str; 9] = [", 1)[1].split("];", 1)[0]
        self.assertEqual(
            tuple(line.strip().strip(",").strip('"') for line in block.strip().splitlines()),
            campaign.SWITCHES,
        )
        self.assertEqual(campaign.LANES["cuda"].switches, campaign.CANDLE_SWITCHES)
        self.assertEqual(campaign.LANES["mlx"].switches, campaign.MLX_SWITCHES)

    def test_the_stamp_variables_and_schema_are_the_harness_ones(self) -> None:
        source = (
            REPOSITORY / "crates/contracts/core-llm/core-llm-testkit/src/speculative.rs"
        ).read_text(encoding="utf-8")
        for name in (
            campaign.BUILD_SHA_ENV,
            campaign.BUILD_DIRTY_ENV,
            campaign.GIT_SHA_ENV,
            campaign.ALLOW_SHA_OVERRIDE_ENV,
        ):
            self.assertIn(f'"{name}"', source)
        self.assertIn(f'option_env!("{campaign.BUILD_SHA_ENV}")', source)
        self.assertIn(f'pub const BENCH_SCHEMA: &str = "{campaign.SCHEMA}";', source)

    def test_the_optional_matrix_runner_is_accepted_for_both_macs_and_not_planned(self) -> None:
        good = {"id": "r", "snapshot": "qwen38"}
        base = campaign.plan_matrix(json.dumps({"rows": [good]}), SNAPSHOTS)
        for runner in ("rw-starvector", "rw-llm"):
            with self.subTest(runner):
                plan = campaign.plan_matrix(json.dumps({"rows": [good], "runner": runner}), SNAPSHOTS)
                self.assertEqual(plan, base)
        with self.assertRaisesRegex(campaign.MatrixError, "rw-starvector.*rw-llm"):
            campaign.plan_matrix(json.dumps({"rows": [good], "runner": "rw-mage"}), SNAPSHOTS)

    def test_the_plan_command_refuses_missing_runner_local_paths_but_not_hub_pins(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "snap").mkdir()
            (root / "head.safetensors").write_text("x", encoding="utf-8")
            snap, head = str(root / "snap"), str(root / "head.safetensors")
            missing = str(root / "absent")

            def check(row, environ=None):
                return campaign.plan_matrix(
                    json.dumps({"rows": [{"id": "r", **row}]}),
                    {**SNAPSHOTS, "DECODE_BENCH_SNAPSHOT_QWEN38": missing, **(environ or {})},
                    "mlx",
                    require_local=True,
                )

            check({"snapshot": snap, "runs": ["epic"], "draft": snap, "mtp_head": head})
            for row, field in (
                ({"snapshot": missing}, "snapshot"),
                ({"snapshot": snap, "runs": ["epic"], "draft": missing}, "draft"),
                ({"snapshot": snap, "runs": ["epic"], "mtp_head": missing}, "mtp_head"),
                ({"snapshot": snap, "runs": ["epic"], "draft": head}, "draft"),
                ({"snapshot": "qwen38"}, "snapshot"),  # the alias resolves to the absent $QWEN path
            ):
                with self.subTest(field=field, row=row):
                    with self.assertRaisesRegex(campaign.MatrixError, rf"row r: {field} .*not an existing"):
                        check(row)
            hub = {"snapshot": "hub:org/name@" + "a" * 40}
            planned = check(hub, {"MODEL_PROOF_HUB_ROOT": str(root / "hub")})
            self.assertEqual(len(planned["hub_snapshots"]), 1)
            # Without the check (the other tests, `local`) a missing path still plans.
            plan({"rows": [{"id": "r", "snapshot": missing}]})
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "plan.json"
            matrix = '{"rows": [{"id": "r", "snapshot": "qwen38"}]}'
            environ = {**SNAPSHOTS, "DECODE_BENCH_MATRIX": matrix}
            with mock.patch.dict(os.environ, environ), contextlib.redirect_stderr(io.StringIO()):
                self.assertEqual(campaign.main(["plan", "--output", str(output)]), 1)
            self.assertFalse(output.exists())

    def test_every_malformed_matrix_is_refused_before_anything_runs(self) -> None:
        good = {"id": "r", "snapshot": "qwen38"}
        cases = {
            "not json": "{",
            "extra top key": {"rows": [good], "more": 1},
            "unknown runner": {"rows": [good], "runner": "rw-mage"},
            "non-string runner": {"rows": [good], "runner": None},
            "runner without rows": {"runner": "rw-llm"},
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
            "zero processes": {"rows": [{**good, "process_repeats": 0}]},
            "bool processes": {"rows": [{**good, "process_repeats": True}]},
            "bad format": {"rows": [{**good, "format": "fp8"}]},
            "bad thinking": {"rows": [{**good, "thinking": "high"}]},
            "empty options": {"rows": [{**good, "options": []}]},
            "options string": {"rows": [{**good, "options": "off"}]},
            "sampling string": {"rows": [{**good, "sampling": "hot"}]},
            "bool count": {"rows": [{**good, "repeats": True}]},
            "negative count": {"rows": [{**good, "new_tokens": -1}]},
            "string warmup": {"rows": [{**good, "warmup": "0"}]},
            "unknown switch": {"rows": [{**good, "switches": {"CANDLE_LLM_GRAPHS": "1"}}]},
            "another lane's switch": {"rows": [{**good, "switches": {"MLX_LLM_PIPELINING": "1"}}]},
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
        with self.assertRaises(campaign.MatrixError):
            campaign.plan_matrix(json.dumps({"rows": [good]}), SNAPSHOTS, "metal")

    def test_the_plan_command_reads_the_untrusted_matrix_only_from_the_environment(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "plan.json"
            environ = {**SNAPSHOTS, "DECODE_BENCH_MATRIX": '{"rows": [{"id": "r", "snapshot": "x"}]}'}
            with mock.patch.dict(os.environ, environ), contextlib.redirect_stderr(io.StringIO()):
                self.assertEqual(campaign.main(["plan", "--output", str(output)]), 1)
            self.assertFalse(output.exists())
            environ["DECODE_BENCH_MATRIX"] = '{"rows": [{"id": "r", "snapshot": "qwen38"}]}'
            environ["DECODE_BENCH_SNAPSHOT_QWEN38"] = directory
            with mock.patch.dict(os.environ, environ), contextlib.redirect_stdout(io.StringIO()):
                self.assertEqual(
                    campaign.main(["plan", "--lane", "cuda", "--output", str(output)]), 0
                )
            planned = json.loads(output.read_text(encoding="utf-8"))
            self.assertEqual((planned["lane"], planned["rows"][0]["id"]), ("cuda", "r"))


class DriverTests(unittest.TestCase):
    SOURCE = (REPOSITORY / campaign.DRIVER_SOURCE).read_text(encoding="utf-8")

    @unittest.skipUnless(shutil.which("sed"), "needs sed")
    def test_the_mlx_copy_is_the_header_sed_and_names_only_mlx(self) -> None:
        script = (
            "/^\\/\\/ BEGIN candle backend/,/^\\/\\/ END candle backend$/d",
            "s/^\\/\\/mlx //",
        )
        self.assertIn(f"sed -e '{script[0]}' -e '{script[1]}'", self.SOURCE, "the header recipe")
        sed = subprocess.run(
            ["sed", "-e", script[0], "-e", script[1]],
            input=self.SOURCE,
            capture_output=True,
            text=True,
            encoding="utf-8",
            check=True,
        ).stdout
        mlx = campaign.baseline_driver_text(campaign.LANES["mlx"], self.SOURCE)
        self.assertEqual(mlx, sed)
        self.assertIn("\nuse mlx_llm::LlamaProvider as Provider;\n", mlx)
        self.assertNotIn("candle_llm::", mlx)
        self.assertFalse(any(line.startswith("//mlx ") for line in mlx.splitlines()))

    def test_the_candle_copy_is_verbatim_and_a_broken_block_is_refused(self) -> None:
        self.assertEqual(
            campaign.baseline_driver_text(campaign.LANES["cuda"], self.SOURCE), self.SOURCE
        )
        broken = self.SOURCE.replace("// END candle backend\n", "")
        with self.assertRaises(campaign.CampaignError):
            campaign.baseline_driver_text(campaign.LANES["mlx"], broken)


class CheckoutTests(unittest.TestCase):
    """The checkout checks and the worktree / driver set-up, on real git repositories."""

    DRIVER = (
        "// header\n// BEGIN candle backend\nuse candle;\n// END candle backend\n"
        "//mlx use mlx;\nfn x() {}\n"
    )

    def setUp(self) -> None:
        self.directory = Path(self.enterContext(tempfile.TemporaryDirectory()))
        self.epic = self.directory / "epic"
        git_repo(self.epic)
        (self.epic / campaign.DRIVER_SOURCE).parent.mkdir(parents=True)
        (self.epic / campaign.DRIVER_SOURCE).write_text(self.DRIVER, encoding="utf-8")
        self.epic_sha = commit(self.epic)
        self.pre = self.directory / "pre"
        git_repo(self.pre)
        main = self.pre / campaign.MLX_MAIN
        main.parent.mkdir(parents=True)
        main.write_text("mod common;\n", encoding="utf-8")
        (self.pre / "crates/llm/candle-llm/tests").mkdir(parents=True)
        (self.pre / "crates/llm/candle-llm/tests/keep.rs").write_text("", encoding="utf-8")
        self.pre_sha = commit(self.pre)

    def check(self, lane: str, run: str) -> None:
        root, sha = (self.epic, self.epic_sha) if run == "epic" else (self.pre, self.pre_sha)
        campaign.check_checkout(campaign.LANES[lane], run, root, sha, self.epic)

    def test_the_epic_checkout_must_be_the_commit_on_a_clean_tree(self) -> None:
        self.check("cuda", "epic")
        with self.assertRaisesRegex(campaign.CampaignError, "not"):
            campaign.check_checkout(campaign.LANES["cuda"], "epic", self.epic, "f" * 40, self.epic)
        (self.epic / "stray.txt").write_text("x", encoding="utf-8")
        with self.assertRaisesRegex(campaign.CampaignError, "tree changes"):
            self.check("cuda", "epic")

    def test_the_candle_baseline_differs_by_exactly_the_epic_driver(self) -> None:
        with self.assertRaisesRegex(campaign.CampaignError, "tree changes"):
            self.check("cuda", "baseline")
        campaign.apply_baseline_driver(campaign.LANES["cuda"], self.epic, self.pre)
        self.check("cuda", "baseline")
        driver = self.pre / campaign.LANES["cuda"].driver_path
        driver.write_text(self.DRIVER + "// edited\n", encoding="utf-8")
        with self.assertRaisesRegex(campaign.CampaignError, "not the epic checkout's driver"):
            self.check("cuda", "baseline")
        driver.write_text(self.DRIVER, encoding="utf-8")
        (self.pre / "stray.txt").write_text("x", encoding="utf-8")
        with self.assertRaisesRegex(campaign.CampaignError, "tree changes"):
            self.check("cuda", "baseline")

    def test_the_mlx_baseline_is_the_block_swap_plus_the_registration_applied_once(self) -> None:
        lane = campaign.LANES["mlx"]
        campaign.apply_baseline_driver(lane, self.epic, self.pre)
        campaign.apply_baseline_driver(lane, self.epic, self.pre)  # idempotent
        self.check("mlx", "baseline")
        self.assertEqual(
            (self.pre / lane.driver_path).read_text(encoding="utf-8"),
            "// header\nuse mlx;\nfn x() {}\n",
        )
        main = self.pre / campaign.MLX_MAIN
        self.assertEqual(
            main.read_text(encoding="utf-8"), "mod common;\n" + campaign.MLX_MODULE_REGISTRATION
        )
        main.write_text("mod common;\nmod other;\n", encoding="utf-8")
        with self.assertRaisesRegex(campaign.CampaignError, "registration"):
            self.check("mlx", "baseline")
        with self.assertRaisesRegex(campaign.CampaignError, "other than"):
            campaign.apply_baseline_driver(lane, self.epic, self.pre)

    def test_a_worktree_is_created_reused_at_its_commit_and_refused_at_another(self) -> None:
        first = self.epic_sha
        (self.epic / "more.txt").write_text("x", encoding="utf-8")
        second = commit(self.epic, "second")
        path = self.directory / "work" / "epic"
        campaign.ensure_worktree(self.epic, path, first)
        self.assertEqual(campaign.git_head(path), first)
        campaign.ensure_worktree(self.epic, path, first)
        with self.assertRaises(campaign.CampaignError):
            campaign.ensure_worktree(self.epic, path, second)

    def test_the_snapshot_identity_is_its_content_not_its_name(self) -> None:
        a = write_snapshot(self.directory / "a" / "model")
        b = write_snapshot(self.directory / "b" / "model")
        self.assertEqual(
            campaign.snapshot_identity(str(a))["sha256"],
            campaign.snapshot_identity(str(b))["sha256"],
        )
        for mutate in (
            lambda root: (root / "config.json").write_text('{"model_type": "qwen"}', encoding="utf-8"),
            lambda root: (root / "model.safetensors").write_bytes(b"\0" * 65),
            lambda root: (root / "model-2.safetensors").write_bytes(b""),
            lambda root: (root / "model.safetensors.index.json").write_text("{}", encoding="utf-8"),
        ):
            with self.subTest(mutate=mutate):
                shutil.rmtree(b)
                write_snapshot(b)
                mutate(b)
                self.assertNotEqual(
                    campaign.snapshot_identity(str(a))["sha256"],
                    campaign.snapshot_identity(str(b))["sha256"],
                )
        empty = self.directory / "empty"
        empty.mkdir()
        with self.assertRaises(campaign.CampaignError):
            campaign.snapshot_identity(str(empty))


def good_document(**overrides):
    document = {
        "schema": campaign.SCHEMA,
        "backend": "candle-cuda",
        "thinking": {"setting": "default"},
        "provenance": {
            "git": {
                "sha": EPIC_SHA,
                "dirty": False,
                "changes": [],
                "source": "git",
                "build_sha": EPIC_SHA,
                "build_dirty": False,
            }
        },
        "rows": [{"prompt_id": "chat"}],
    }
    document.update(overrides)
    return document


class CheckDocumentTests(unittest.TestCase):
    def check(self, document, **kwargs):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "epic-1.json"
            path.write_text(json.dumps(document), encoding="utf-8")
            options = {
                "expected_sha": EPIC_SHA,
                "changes": [],
                "thinking": "default",
                "backend": "candle-cuda",
                **kwargs,
            }
            return campaign.check_document(path, **options)

    def test_a_document_counts_only_as_the_run_it_claims_to_be(self) -> None:
        self.check(good_document())

        def git(**fields):
            document = good_document()
            document["provenance"]["git"].update(fields)
            return document

        cases = {
            "schema": good_document(schema="sceneworks.decode-speedups.baseline/3"),
            "backend": good_document(backend="candle"),
            "thinking": good_document(thinking={"setting": "off"}),
            "run-time sha": git(sha="b" * 40),
            "operator sha": git(source=campaign.GIT_SHA_ENV),
            "stale binary": git(build_sha="b" * 40),
            "unstamped binary": git(build_sha=None, build_dirty=None),
            "dirty stamp": git(build_dirty=True),
            "dirty tree": git(changes=[" M x.rs"], dirty=True),
            "dirty flag": git(dirty=True),
            "no rows": good_document(rows=[]),
        }
        for name, document in cases.items():
            with self.subTest(name), self.assertRaises(campaign.CampaignError):
                self.check(document)

    def test_the_operator_sha_counts_only_unstamped_and_under_the_override(self) -> None:
        operator = good_document()
        operator["provenance"]["git"].update(
            {"source": campaign.GIT_SHA_ENV, "build_sha": None, "build_dirty": None, "dirty": None}
        )
        self.check(operator, sha_override=True)
        with self.assertRaises(campaign.CampaignError):
            self.check(operator)
        with self.assertRaises(campaign.CampaignError):
            self.check(good_document(), sha_override=True)  # stamped under the override


class RunTests(unittest.TestCase):
    def setUp(self) -> None:
        self.directory = Path(self.enterContext(tempfile.TemporaryDirectory()))
        fake = self.directory / "fake_cargo.py"
        fake.write_text(FAKE_CARGO, encoding="utf-8")
        self.guard = self.directory / "fake_guard.py"
        self.guard.write_text(FAKE_GUARD, encoding="utf-8")
        self.record = self.directory / "record.jsonl"
        original = campaign.cargo_args
        self.enterContext(
            mock.patch.object(
                campaign,
                "cargo_args",
                lambda lane, run, no_run=False: [str(fake), *original(lane, run, no_run=no_run)],
            )
        )
        self.checked = []
        self.enterContext(
            mock.patch.object(
                campaign,
                "check_checkout",
                lambda lane, run, root, sha, epic_root: self.checked.append((run, sha)),
            )
        )
        self.snapshot = write_snapshot(self.directory / "snapshots" / "qwen38")
        for name in ("epic", "pre-epic"):
            (self.directory / name).mkdir()

    def run_plan(self, rows, *extra, **fake_env):
        self.record.unlink(missing_ok=True)
        self.checked.clear()
        output = self.directory / "out"
        if output.exists():
            shutil.rmtree(output)
        write_snapshot(self.snapshot)
        plan_path = self.directory / "plan.json"
        plan_path.write_text(
            json.dumps(plan({"rows": rows}, {"DECODE_BENCH_SNAPSHOT_QWEN38": str(self.snapshot)})),
            encoding="utf-8",
        )
        environ = {
            "FAKE_RECORD": str(self.record),
            "SPECULATIVE_BENCH_DRAFT": "/stale/draft",
            "SPECULATIVE_BENCH_GIT_SHA": "c" * 40,
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
                    *extra,
                ]
            )
        summary_path = output / "summary.json"
        summary = (
            json.loads(summary_path.read_text(encoding="utf-8")) if summary_path.exists() else None
        )
        records = (
            [json.loads(line) for line in self.record.read_text(encoding="utf-8").splitlines()]
            if self.record.exists()
            else []
        )
        return code, output, summary, records

    def test_both_checkouts_build_stamped_once_then_processes_alternate_per_row(self) -> None:
        code, output, summary, records = self.run_plan(
            [
                {
                    "id": "a",
                    "snapshot": "qwen38",
                    "process_repeats": 2,
                    "switches": {"CANDLE_LLM_CUDA_GRAPHS": "1"},
                },
                {
                    "id": "b",
                    "snapshot": "qwen38",
                    "thinking": "off",
                    "runs": ["epic"],
                    "process_repeats": 1,
                },
            ]
        )
        self.assertEqual(code, 0, summary)
        self.assertEqual(self.checked, [("epic", EPIC_SHA), ("baseline", campaign.PRE_EPIC_SHA)])
        self.assertEqual(
            [(r["run"], r["build"]) for r in records],
            [
                ("epic", True),
                ("baseline", True),
                ("epic", False),
                ("baseline", False),
                ("epic", False),
                ("baseline", False),
                ("epic", False),
            ],
        )
        for record in records:
            env = record["env"]
            # The compile-time stamp, the same for the build and every process of a run.
            self.assertEqual(
                (env["SPECULATIVE_BENCH_BUILD_GIT_SHA"], env["SPECULATIVE_BENCH_BUILD_GIT_DIRTY"]),
                (EPIC_SHA, "0") if record["run"] == "epic" else (campaign.PRE_EPIC_SHA, "1"),
            )
            self.assertNotIn("SPECULATIVE_BENCH_GIT_SHA", env)
            self.assertNotIn("SPECULATIVE_BENCH_ALLOW_SHA_OVERRIDE", env)
            self.assertNotIn("SPECULATIVE_BENCH_DRAFT", env)
            self.assertNotIn("CANDLE_LLM_FUSED_KERNELS", env)
            self.assertEqual(
                env.get("CARGO_TARGET_DIR") == "T:\\cargo-target-pre-epic",
                record["run"] == "baseline",
            )
            if record["build"]:
                self.assertEqual(record["argv"][-1], "--no-run")
            else:
                self.assertEqual(env["SPECULATIVE_BENCH_BACKEND"], "candle-cuda")
                self.assertEqual(
                    record["argv"][record["argv"].index("--") + 2 :],
                    ["--exact", "--ignored", "--nocapture"],
                )
        self.assertEqual(records[2]["env"]["CANDLE_LLM_CUDA_GRAPHS"], "1")
        self.assertNotIn("CANDLE_LLM_CUDA_GRAPHS", records[-1]["env"])
        rows = {row["id"]: row for row in summary["rows"]}
        self.assertEqual(
            [(p["run"], p["process"], p["status"]) for p in rows["a"]["processes"]],
            [("epic", 1, "ok"), ("baseline", 1, "ok"), ("epic", 2, "ok"), ("baseline", 2, "ok")],
        )
        identity = campaign.snapshot_identity(str(self.snapshot))["sha256"]
        self.assertEqual(rows["a"]["snapshot"]["identity"], identity)
        self.assertEqual(rows["a"]["processes"][0]["switches"], {"CANDLE_LLM_CUDA_GRAPHS": "1"})
        for name in ("a/epic-1.json", "a/baseline-2.json", "b/epic-1.json", "build-epic.log"):
            self.assertTrue((output / name).is_file(), name)
        self.assertEqual(
            summary["builds"], {"epic": {"status": "ok"}, "baseline": {"status": "ok"}}
        )

    def test_a_without_mtp_row_is_prepared_once_and_both_runs_measure_the_copy(self) -> None:
        source = write_hf_snapshot(self.directory / "huggingface")
        copy_path = campaign.without_mtp_path(str(source))
        row = {"snapshot": str(source), "options": ["off"], "without_mtp": True, "process_repeats": 1}
        code, _, summary, records = self.run_plan([{**row, "id": "a"}, {**row, "id": "b"}])
        self.assertEqual(code, 0, summary)
        runs = [r for r in records if not r["build"]]
        self.assertEqual([r["run"] for r in runs], ["epic", "baseline", "epic", "baseline"])
        self.assertEqual({r["env"]["SPECULATIVE_BENCH_SNAPSHOT"] for r in runs}, {copy_path})
        identity = campaign.snapshot_identity(copy_path)["sha256"]
        for entry in summary["rows"]:
            self.assertEqual(
                entry["snapshot"],
                {"path": copy_path, "identity": identity, "prepared_from": str(source)},
            )
        self.assertNotIn("mtp.fc.weight", read_safetensors(Path(copy_path) / "model-00002-of-00002.safetensors")[0])

    def test_the_guard_wraps_every_benchmark_process_and_no_build(self) -> None:
        code, _, _, records = self.run_plan(
            [{"id": "a", "snapshot": "qwen38", "process_repeats": 1}],
            "--guard",
            f"{sys.executable} {self.guard}",
        )
        self.assertEqual(code, 0)
        kinds = ["guard" if "guard" in r else ("build" if r["build"] else "run") for r in records]
        self.assertEqual(kinds, ["build", "build", "guard", "run", "guard", "run"])

    def test_a_process_that_does_not_count_fails_the_campaign_without_stopping_it(self) -> None:
        modes = (
            "exit",
            "zero",
            "two",
            "no_document",
            "wrong_sha",
            "stale_build",
            "wrong_dirty_stamp",
            "wrong_thinking",
            "wrong_backend",
            "dirty",
            "touch_snapshot",
        )
        for mode in modes:
            with self.subTest(mode=mode):
                code, _, summary, records = self.run_plan(
                    [
                        {"id": "a", "snapshot": "qwen38", "process_repeats": 1},
                        {"id": "b", "snapshot": "qwen38", "process_repeats": 1},
                    ],
                    FAKE_MODE=mode,
                )
                self.assertEqual(code, 1)
                self.assertEqual(len([r for r in records if not r["build"]]), 4, "every row ran")
                processes = [p for row in summary["rows"] for p in row["processes"]]
                self.assertTrue(
                    all(p["status"] == "failed" and p["error"] for p in processes), processes
                )

    def test_a_failed_build_stops_before_any_row(self) -> None:
        code, _, summary, records = self.run_plan(
            [{"id": "a", "snapshot": "qwen38"}], FAKE_BUILD_EXIT="101"
        )
        self.assertEqual(code, 1)
        self.assertEqual([r["build"] for r in records], [True])
        self.assertEqual(summary["builds"], {"epic": {"status": "failed"}})
        self.assertEqual(summary["rows"], [])

    def test_a_campaign_stopped_part_way_keeps_every_finished_process(self) -> None:
        """sc-24446: a job timeout kills the runner mid-campaign (run 36891326535 lost 32 finished
        rows that way). The summary is rewritten after every process, so what the stopped
        campaign leaves says which rows finished and which one it stopped during."""
        original = campaign.run_process
        calls = []

        def stop_on_the_third(**kwargs):
            calls.append(kwargs["name"])
            if len(calls) == 3:
                raise KeyboardInterrupt  # the job's timeout killing the runner
            return original(**kwargs)

        rows = [
            {"id": "a", "snapshot": "qwen38", "process_repeats": 1},
            {"id": "b", "snapshot": "qwen38", "process_repeats": 1},
        ]
        with mock.patch.object(campaign, "run_process", stop_on_the_third):
            with self.assertRaises(KeyboardInterrupt):
                self.run_plan(rows)
        output = self.directory / "out"
        summary = json.loads((output / "summary.json").read_text(encoding="utf-8"))
        self.assertFalse((output / "summary.json.tmp").exists())
        self.assertIs(summary["complete"], False)
        a, b = summary["rows"]
        self.assertIs(a["complete"], True)
        self.assertEqual(
            [(p["run"], p["status"]) for p in a["processes"]],
            [("epic", "ok"), ("baseline", "ok")],
        )
        self.assertIs(b["complete"], False)
        self.assertEqual(b["processes"], [])

        # A campaign that ran to its end says so, row by row.
        code, _, summary, _ = self.run_plan(rows)
        self.assertEqual(code, 0)
        self.assertIs(summary["complete"], True)
        self.assertTrue(all(row["complete"] for row in summary["rows"]))

    def test_the_operator_sha_is_taken_only_with_the_explicit_override(self) -> None:
        code, _, summary, records = self.run_plan(
            [{"id": "a", "snapshot": "qwen38", "process_repeats": 1}], "--allow-sha-override"
        )
        self.assertEqual(code, 0, summary)
        self.assertTrue(summary["sha_override"])
        for record in records:
            self.assertNotIn("SPECULATIVE_BENCH_BUILD_GIT_SHA", record["env"])
            self.assertEqual(record["env"]["SPECULATIVE_BENCH_ALLOW_SHA_OVERRIDE"], "1")
        self.assertEqual(
            [r["env"]["SPECULATIVE_BENCH_GIT_SHA"] for r in records if not r["build"]],
            [EPIC_SHA, campaign.PRE_EPIC_SHA],
        )

    def test_a_checkout_that_is_not_the_measured_tree_is_refused_before_any_build(self) -> None:
        def refuse(lane, run, root, sha, epic_root):
            raise campaign.CampaignError(f"{root} is not {sha}")

        with mock.patch.object(campaign, "check_checkout", refuse):
            code, _, summary, records = self.run_plan([{"id": "a", "snapshot": "qwen38"}])
        self.assertEqual(code, 1)
        self.assertEqual(records, [])
        self.assertIsNone(summary)


# ---------------------------------------------------------------------------------------- compare

PROMPTS = (("code_edit", "predictable"), ("chat", "open_ended"))
OPTIONS = ("off", {"proposer": "mtp", "depth": 3}, "auto")


def document(
    run, *, effective=None, switch_env=None, prefix_cache_bytes=0, value=None, epic_sha=EPIC_SHA,
    row_extra=None, **overrides,
):
    """A synthetic benchmark document; ``value(prompt, option, metric)`` gives each in-process
    mean, ``row_extra(prompt, option)`` extra row fields (e.g. the DecodeReport's ``proposer``);
    ``effective`` / ``switch_env`` the provenance's effective switch values / their variables."""
    value = value or (lambda prompt, option, metric: 50.0 if metric == "decode_tok_s" else 100.0)
    switches = {name: {"env": None, "effective": None} for name in campaign.SWITCHES}
    for name, state in (effective or {}).items():
        switches[name]["effective"] = state
    for name, env in (switch_env or {}).items():
        switches[name]["env"] = env
    rows = []
    for prompt, prompt_class in PROMPTS:
        for option in OPTIONS:
            key = campaign.option_key(option)
            rows.append(
                {
                    "prompt_id": prompt,
                    "class": prompt_class,
                    "requested": option,
                    "prompt_tokens": 40,
                    "decode_tok_s": {
                        "n": 3,
                        "mean": value(prompt, key, "decode_tok_s"),
                        "stddev": 0.5,
                    },
                    "ttft_ms": {"n": 3, "mean": value(prompt, key, "ttft_ms"), "stddev": 2.0},
                    **(row_extra(prompt, key) if row_extra else {}),
                }
            )
    return {
        "schema": campaign.SCHEMA,
        "model": "qwen38",
        "backend": "candle-cuda",
        "max_new_tokens": 256,
        "sampling": {"temperature": 0.0},
        "thinking": {"setting": "default"},
        "warmup": True,
        "repeats": 3,
        "options": list(OPTIONS),
        "load": {"prefix_cache_bytes": prefix_cache_bytes},
        "provenance": {
            "git": {"sha": epic_sha if run == "epic" else campaign.PRE_EPIC_SHA},
            "switches": switches,
            "env": {"CUDA_VISIBLE_DEVICES": "1", "CANDLE_LLM_DEVICE": None},
        },
        "rows": rows,
        **overrides,
    }


def write_campaign(root: Path, rows, lane="cuda", epic_sha=EPIC_SHA):
    """``rows``: ``{"id", "runs", "process_repeats", "switches", "env", "format", "snapshot",
    "document": fn(run, k)}``."""
    root.mkdir(parents=True)
    summary = {
        "schema": campaign.CAMPAIGN_SCHEMA,
        "document_schema": campaign.SCHEMA,
        "lane": lane,
        "backend": campaign.LANES[lane].backend,
        "epic_sha": epic_sha,
        "pre_epic_sha": campaign.PRE_EPIC_SHA,
        "sha_override": False,
        "builds": {"epic": {"status": "ok"}, "baseline": {"status": "ok"}},
        "rows": [],
    }
    for row in rows:
        (root / row["id"]).mkdir()
        entry = {
            "id": row["id"],
            "runs": row.get("runs", ["epic", "baseline"]),
            "process_repeats": row.get("process_repeats", 3),
            "switches": row.get("switches", {}),
            "format": row.get("format", "bf16"),
            "env": row.get("env", {}),
            "snapshot": {"path": row.get("snapshot", "/s"), "identity": "f" * 64},
            "processes": [],
        }
        for k in range(1, entry["process_repeats"] + 1):
            for run in entry["runs"]:
                name = f"{row['id']}/{run}-{k}.json"
                (root / name).write_text(json.dumps(row["document"](run, k)), encoding="utf-8")
                entry["processes"].append(
                    {
                        "run": run,
                        "process": k,
                        "document": name,
                        "status": "ok",
                        "error": None,
                        "snapshot_identity": "f" * 64,
                        "switches": entry["switches"],
                    }
                )
        summary["rows"].append(entry)
    (root / "summary.json").write_text(json.dumps(summary), encoding="utf-8")
    return root


def jitter(k):
    return (-1.0, 0.0, 1.0)[(k - 1) % 3]


LIVE_DEFAULTS = (REPOSITORY / "crates/contracts/core-llm/src/defaults.rs").read_text(
    encoding="utf-8"
)

# The defaults table as the sc-24446 campaign found it (the entries it decided still
# PROVISIONAL, CUDA graphs off): the compare logic is judged against this fixed table, not the
# live one, whose values and justifications the campaign's own decisions change.
CAMPAIGN_DEFAULTS = textwrap.dedent(
    """\
    pub const MLX: DecodeDefaults = DecodeDefaults {
        // justification: PROVISIONAL — sc-24446 campaign. Off until measured.
        speculative: Speculative::Off,
        // justification: proposer-intrinsic depths.
        recommended_depths: RECOMMENDED_DEPTHS,
        // justification: PROVISIONAL — sc-24446 campaign. On (E5), admission-clamped (E7).
        prefix_cache_bytes: PREFIX_CACHE_BYTES,
        // justification: n/a — MLX has no CUDA graphs.
        cuda_graphs: false,
        // justification: on (sc-24439) — token-identical to the unpipelined loop by construction.
        // Runtime switch `MLX_LLM_PIPELINING`.
        pipelining: true,
        // justification: on (sc-24439) — greedy is the device argmax. Runtime switch
        // `MLX_LLM_DEVICE_SAMPLER`.
        device_sampler: true,
    };
    pub const CANDLE_CUDA: DecodeDefaults = DecodeDefaults {
        // justification: PROVISIONAL — sc-24446 campaign. Off until measured.
        speculative: Speculative::Off,
        // justification: proposer-intrinsic depths.
        recommended_depths: RECOMMENDED_DEPTHS,
        // justification: PROVISIONAL — sc-24446 campaign. On (E5), admission-clamped (E7).
        prefix_cache_bytes: PREFIX_CACHE_BYTES,
        // justification: PROVISIONAL — sc-24446 campaign. Off until measured. Runtime switch
        // `CANDLE_LLM_CUDA_GRAPHS`.
        cuda_graphs: false,
        // justification: PROVISIONAL — sc-24446 campaign. On. Runtime switch
        // `CANDLE_LLM_DEVICE_POSITIONS`.
        device_positions: true,
        // justification: n/a — Candle's engine has no pipelined loop.
        pipelining: false,
    };
    pub const CANDLE_METAL: DecodeDefaults = DecodeDefaults {
        // justification: PROVISIONAL — sc-24446 campaign (not a campaign host; follows CUDA).
        speculative: Speculative::Off,
        // justification: PROVISIONAL — sc-24446 campaign. On (E5), admission-clamped (E7).
        prefix_cache_bytes: PREFIX_CACHE_BYTES,
        // justification: n/a — CUDA only.
        cuda_graphs: false,
    };
    pub const CANDLE_CPU: DecodeDefaults = DecodeDefaults {
        // justification: PROVISIONAL — sc-24446 campaign (not a campaign host; follows CUDA).
        speculative: Speculative::Off,
        // justification: PROVISIONAL — sc-24446 campaign. On (E5), admission-clamped.
        prefix_cache_bytes: PREFIX_CACHE_BYTES,
        // justification: n/a — CUDA only.
        cuda_graphs: false,
        // justification: PROVISIONAL — sc-24446 campaign. Off: no graph runner on the CPU.
        device_positions: false,
    };
    """
)


class CompareTests(unittest.TestCase):
    DEFAULTS = CAMPAIGN_DEFAULTS

    def setUp(self) -> None:
        self.directory = Path(self.enterContext(tempfile.TemporaryDirectory()))

    def campaign_with(self, epic_value=None, name="cuda-run", **row):
        def make(run, k):
            def value(prompt, option, metric):
                base = 50.0 if metric == "decode_tok_s" else 100.0
                if run == "epic" and epic_value:
                    base = epic_value(prompt, option, metric, base)
                return base + jitter(k)

            return document(run, value=value)

        return write_campaign(
            self.directory / name, [{"id": "qwen38-bf16", "document": make, **row}]
        )

    def compare(self, *directories, max_noise=campaign.DEFAULT_MAX_NOISE):
        return campaign.compare_campaigns(list(directories), self.DEFAULTS, max_noise)

    def test_equal_runs_pass_e6_and_auto_is_reported_not_judged(self) -> None:
        report = self.compare(self.campaign_with())
        self.assertEqual(
            {(e["class"], e["option"], e["verdict"]) for e in report["e6"]},
            {
                (prompt_class, option, "pass")
                for prompt_class in ("predictable", "open_ended")
                for option in ("off", campaign.option_key(OPTIONS[1]))
            },
        )
        self.assertNotIn("auto", {c["option"] for c in report["comparisons"]})
        self.assertEqual({r["option"] for r in report["not_comparable"]}, {"auto"})
        self.assertEqual(report["not_comparable"][0]["decode_tok_s"]["epic"]["n"], 3)
        comparison = report["comparisons"][0]
        self.assertEqual((comparison["epic"]["n"], comparison["baseline"]["n"]), (3, 3))
        self.assertEqual(comparison["epic"]["within_process_stddev"], 0.5)
        markdown = campaign.markdown_report(report)
        for heading in ("## E6", "## Not comparable", "## E5"):
            self.assertIn(heading, markdown)

    def test_a_regression_beyond_the_noise_band_is_a_regression_verdict(self) -> None:
        # The epic decodes the open-ended prompt 20% slower (noise ±1 tok/s).
        slower = self.campaign_with(
            lambda prompt, option, metric, base: base * 0.8
            if metric == "decode_tok_s" and prompt == "chat"
            else base
        )
        verdicts = {(e["class"], e["option"]): e for e in self.compare(slower)["e6"]}
        self.assertEqual(verdicts[("open_ended", "off")]["decode_tok_s"], "regression")
        self.assertEqual(verdicts[("open_ended", "off")]["verdict"], "regression")
        self.assertEqual(verdicts[("predictable", "off")]["verdict"], "pass")
        # TTFT is lower-is-better: a higher epic TTFT regresses, a lower one passes.
        later = self.campaign_with(
            lambda prompt, option, metric, base: base * 1.3 if metric == "ttft_ms" else base,
            name="later",
        )
        self.assertTrue(all(e["ttft_ms"] == "regression" for e in self.compare(later)["e6"]))
        sooner = self.campaign_with(
            lambda prompt, option, metric, base: base * 0.7 if metric == "ttft_ms" else base,
            name="sooner",
        )
        self.assertTrue(all(e["verdict"] == "pass" for e in self.compare(sooner)["e6"]))

    def test_the_noise_band_comes_from_process_repeats(self) -> None:
        verdict = campaign.verdict
        self.assertEqual(verdict([49, 50, 51], [50, 51, 52], True, 0.1)["verdict"], "pass")
        self.assertEqual(verdict([40, 41, 42], [50, 51, 52], True, 0.1)["verdict"], "regression")
        wide = verdict([30, 50, 70], [40, 55, 70], True, 0.1)
        self.assertEqual(wide["verdict"], "inconclusive", wide)
        self.assertEqual(verdict([40], [50], True, 0.1)["verdict"], "inconclusive")
        self.assertEqual(verdict([60, 61, 62], [50, 51, 52], True, 0.1)["verdict"], "pass")
        self.assertEqual(
            verdict([60, 61, 62], [50, 51, 52], False, 0.1)["verdict"], "regression"
        )
        self.assertAlmostEqual(campaign.t_critical(4.7), 2.776)
        self.assertAlmostEqual(campaign.t_critical(float("inf")), 1.960)

    def test_incomplete_or_mismatched_evidence_is_refused(self) -> None:
        def edit_document(directory, name, change):
            path = directory / "qwen38-bf16" / name
            doc = json.loads(path.read_text(encoding="utf-8"))
            change(doc)
            path.write_text(json.dumps(doc), encoding="utf-8")

        def edit_summary(directory, change):
            path = directory / "summary.json"
            summary = json.loads(path.read_text(encoding="utf-8"))
            change(summary)
            path.write_text(json.dumps(summary), encoding="utf-8")

        def process(summary, index):
            return summary["rows"][0]["processes"][index]

        def doc(name, change):
            return lambda d: edit_document(d, name, change)

        mutations = {
            "missing row": doc("baseline-2.json", lambda x: x["rows"].pop(1)),
            "duplicate row": doc("epic-1.json", lambda x: x["rows"].append(copy.deepcopy(x["rows"][0]))),
            "missing prompt": doc("baseline-1.json", lambda x: x.update(rows=x["rows"][:3])),
            "max_new_tokens": doc("baseline-3.json", lambda x: x.update(max_new_tokens=128)),
            "sampling": doc("baseline-1.json", lambda x: x.update(sampling={"temperature": 0.7})),
            "thinking": doc("epic-2.json", lambda x: x.update(thinking={"setting": "off"})),
            "repeats": doc("baseline-1.json", lambda x: x.update(repeats=5)),
            "warmup": doc("baseline-1.json", lambda x: x.update(warmup=False)),
            "model": doc("baseline-1.json", lambda x: x.update(model="qwen38-other")),
            "backend": doc("baseline-1.json", lambda x: x.update(backend="candle")),
            "options": doc("baseline-1.json", lambda x: x.update(options=["off", "auto"])),
            "device": doc(
                "baseline-1.json",
                lambda x: x["provenance"]["env"].update(CUDA_VISIBLE_DEVICES="0"),
            ),
            "switch env": doc(
                "epic-3.json",
                lambda x: x["provenance"]["switches"]["CANDLE_LLM_CUDA_GRAPHS"].update(env="1"),
            ),
            "schema": doc("epic-1.json", lambda x: x.update(schema="x")),
            "snapshot identity": lambda d: edit_summary(
                d, lambda s: process(s, 1).update(snapshot_identity="e" * 64)
            ),
            "row identity differs from every process's": lambda d: edit_summary(
                d, lambda s: s["rows"][0]["snapshot"].update(identity="e" * 64)
            ),
            "row identity unrecorded": lambda d: edit_summary(
                d, lambda s: s["rows"][0]["snapshot"].update(identity=None)
            ),
            "failed process": lambda d: edit_summary(
                d, lambda s: process(s, 2).update(status="failed", error="exited 101")
            ),
            "missing document": lambda d: (d / "qwen38-bf16" / "epic-2.json").unlink(),
            "short process count": lambda d: edit_summary(
                d, lambda s: s["rows"][0].update(process_repeats=4)
            ),
            "failed build": lambda d: edit_summary(
                d, lambda s: s["builds"].update(baseline={"status": "failed"})
            ),
            "pre-epic sha": lambda d: edit_summary(d, lambda s: s.update(pre_epic_sha="b" * 40)),
            "summary schema": lambda d: edit_summary(d, lambda s: s.update(schema="x")),
            "measured commit": doc(
                "epic-2.json", lambda x: x["provenance"]["git"].update(sha="b" * 40)
            ),
            "summary commit": lambda d: edit_summary(d, lambda s: s.update(epic_sha="b" * 40)),
            "no summary": lambda d: (d / "summary.json").unlink(),
        }
        for index, (name, mutate) in enumerate(mutations.items()):
            with self.subTest(name):
                directory = self.campaign_with(name=f"m{index}")
                self.compare(directory)  # the unmutated campaign compares
                mutate(directory)
                with self.assertRaises(campaign.Refused):
                    self.compare(directory)

    def test_a_row_the_campaign_stopped_during_is_set_aside_and_the_rest_compare(self) -> None:
        """sc-24446: a timed-out campaign's partial summary is usable — its finished rows are
        judged, the row it stopped during is named and excluded (never judged on a short
        process count); a campaign that finished no row is refused."""

        def make(run, k):
            def value(prompt, option, metric):
                return (50.0 if metric == "decode_tok_s" else 100.0) + jitter(k)

            return document(run, value=value)

        def partial(name, stopped):
            directory = write_campaign(
                self.directory / name,
                [{"id": row, "document": make} for row in ("finished", "stopped")],
            )
            path = directory / "summary.json"
            summary = json.loads(path.read_text(encoding="utf-8"))
            summary["complete"] = False
            for row in summary["rows"]:
                row["complete"] = row["id"] not in stopped
                if row["id"] in stopped:
                    row["processes"] = row["processes"][:2]
            path.write_text(json.dumps(summary), encoding="utf-8")
            return directory

        report = self.compare(partial("timed-out", {"stopped"}))
        self.assertEqual([r["row"] for r in report["rows"]], ["finished"])
        self.assertEqual(len(report["incomplete"]), 1)
        self.assertIn("timed-out/stopped", report["incomplete"][0])
        self.assertIn("2 of 6 processes", report["incomplete"][0])
        self.assertIs(report["campaigns"][0]["complete"], False)
        self.assertIn("## Incomplete", campaign.markdown_report(report))
        # A row marked finished with a short process count is still refused, not set aside.
        directory = partial("short", set())
        path = directory / "summary.json"
        summary = json.loads(path.read_text(encoding="utf-8"))
        summary["rows"][1]["processes"] = summary["rows"][1]["processes"][:2]
        path.write_text(json.dumps(summary), encoding="utf-8")
        with self.assertRaises(campaign.Refused):
            self.compare(directory)
        with self.assertRaises(campaign.Refused) as refused:
            self.compare(partial("nothing-finished", {"finished", "stopped"}))
        self.assertIn("no row ran to its end", str(refused.exception))

    def test_the_compare_command_writes_json_and_markdown_and_refuses_with_exit_1(self) -> None:
        directory = self.campaign_with()
        out_json, out_md = self.directory / "report.json", self.directory / "report.md"
        args = [
            "compare",
            "--campaign",
            str(directory),
            "--json",
            str(out_json),
            "--markdown",
            str(out_md),
        ]
        with contextlib.redirect_stderr(io.StringIO()):
            self.assertEqual(campaign.main(args), 0)
        report = json.loads(out_json.read_text(encoding="utf-8"))
        self.assertEqual(report["schema"], campaign.COMPARE_SCHEMA)
        self.assertIn("## E6", out_md.read_text(encoding="utf-8"))
        (directory / "qwen38-bf16" / "epic-1.json").unlink()
        out_json.unlink()
        out_md.unlink()
        with contextlib.redirect_stderr(io.StringIO()):
            self.assertEqual(campaign.main(args), 1)
        self.assertFalse(out_json.exists())

    def test_the_provisional_defaults_are_read_from_the_table(self) -> None:
        for table in (LIVE_DEFAULTS, self.DEFAULTS):
            self.assert_provisional_entries_are_read(table)
        # The live table parses: every row and its justified entries.
        self.assertTrue(campaign.default_entries(LIVE_DEFAULTS))

    def assert_provisional_entries_are_read(self, table: str) -> None:
        entries = campaign.provisional_defaults(table)
        provisional = sum(
            table.split(f"pub const {const}: DecodeDefaults", 1)[1]
            .split("\n};", 1)[0]
            .count("justification: PROVISIONAL")
            for const in campaign.DEFAULTS_BACKENDS
        )
        self.assertEqual(len(entries), provisional, entries)
        for entry in entries:
            self.assertIn(entry["backend"], campaign.DEFAULTS_BACKENDS.values())
        table = textwrap.dedent(
            """\
            pub const MLX: DecodeDefaults = DecodeDefaults {
                // justification: PROVISIONAL — sc-24446 campaign.
                speculative: Speculative::Off,
                // justification: measured.
                pipelining: true,
            };
            pub const CANDLE_CUDA: DecodeDefaults = DecodeDefaults {
                // justification: PROVISIONAL — sc-24446 campaign. Off until measured,
                // a second comment line.
                cuda_graphs: false,
                // justification: PROVISIONAL — sc-24446 campaign.
                device_positions: true,
                // justification: PROVISIONAL — something new.
                brand_new: true,
            };
            pub const CANDLE_METAL: DecodeDefaults = DecodeDefaults {
            };
            pub const CANDLE_CPU: DecodeDefaults = DecodeDefaults {
                // justification: PROVISIONAL.
                prefix_cache_bytes: PREFIX_CACHE_BYTES,
            };
            """
        )
        self.assertEqual(
            [(e["entry"], e["on"], e["kind"]) for e in campaign.provisional_defaults(table)],
            [
                ("MLX.speculative", False, "speculative"),
                ("CANDLE_CUDA.cuda_graphs", False, "switch"),
                ("CANDLE_CUDA.device_positions", True, "switch"),
                ("CANDLE_CUDA.brand_new", True, None),
                ("CANDLE_CPU.prefix_cache_bytes", True, "prefix_cache"),
            ],
        )

    def decisions(self, rows):
        directory = write_campaign(self.directory / f"d{len(list(self.directory.iterdir()))}", rows)
        return {d["entry"]: d for d in self.compare(directory)["defaults"]}

    def graphs_row(self, row_id, state, speed, **overrides):
        def make(run, k):
            return document(
                run,
                effective={"CANDLE_LLM_CUDA_GRAPHS": state},
                value=lambda p, o, m: (speed if m == "decode_tok_s" else 100.0) + jitter(k),
                **overrides,
            )

        return {
            "id": row_id,
            "runs": ["epic"],
            "switches": {"CANDLE_LLM_CUDA_GRAPHS": "1" if state else "0"},
            "document": make,
        }

    def test_a_switch_default_is_decided_from_its_on_off_pair(self) -> None:
        faster = self.decisions(
            [self.graphs_row("on", True, 60.0), self.graphs_row("off", False, 50.0)]
        )["CANDLE_CUDA.cuda_graphs"]
        self.assertEqual(
            (faster["outcome"], faster["recommended_on"], faster["flip"], faster["pairs"]),
            ("on", True, True, 1),
        )
        slower = self.decisions(
            [self.graphs_row("on", True, 40.0), self.graphs_row("off", False, 50.0)]
        )["CANDLE_CUDA.cuda_graphs"]
        self.assertEqual(
            (slower["outcome"], slower["recommended_on"], slower["flip"]), ("off", False, False)
        )
        self.assertIn("measured on-vs-off regression", slower["reason"])
        # Rows that differ in anything else (here the token budget) are not a pair.
        unpaired = self.decisions(
            [
                self.graphs_row("on", True, 60.0),
                self.graphs_row("off", False, 50.0, max_new_tokens=128),
            ]
        )
        self.assertEqual(unpaired["CANDLE_CUDA.cuda_graphs"]["outcome"], "unmeasured")
        self.assertEqual(unpaired["CANDLE_CUDA.device_positions"]["outcome"], "unmeasured")
        self.assertEqual(unpaired["MLX.speculative"]["outcome"], "unmeasured")

    def test_the_speculative_and_prefix_cache_defaults_are_decided_from_their_pairs(self) -> None:
        def speculative_row(auto_speed):
            def make(run, k):
                return document(
                    run,
                    value=lambda p, o, m: (
                        (auto_speed if o == "auto" else 50.0) if m == "decode_tok_s" else 100.0
                    )
                    + jitter(k),
                )

            return {"id": "spec", "runs": ["epic"], "document": make}

        faster = self.decisions([speculative_row(70.0)])["CANDLE_CUDA.speculative"]
        self.assertEqual((faster["outcome"], faster["flip"]), ("on", True))
        slower = self.decisions([speculative_row(30.0)])["CANDLE_CUDA.speculative"]
        self.assertEqual((slower["outcome"], slower["flip"]), ("off", False))

        def cache_row(row_id, budget, ttft):
            def make(run, k):
                return document(
                    run,
                    prefix_cache_bytes=budget,
                    value=lambda p, o, m: (50.0 if m == "decode_tok_s" else ttft) + jitter(k),
                )

            env = {"SPECULATIVE_BENCH_PREFIX_CACHE_BYTES": str(budget)}
            return {"id": row_id, "runs": ["epic"], "env": env, "document": make}

        cache = self.decisions(
            [cache_row("cache-on", 1 << 30, 60.0), cache_row("cache-off", 0, 100.0)]
        )["CANDLE_CUDA.prefix_cache_bytes"]
        self.assertEqual((cache["outcome"], cache["flip"]), ("on", False))

    # ---- sc-24446 compare fixes: supersession, E5 pairing, n=1, unpaired rows, labels

    def history(self):
        """A repository ``base -> older -> newer`` plus ``other`` branching from ``base``."""
        repo = self.directory / "repo"
        git_repo(repo)
        (repo / "f").write_text("0", encoding="utf-8")
        base = commit(repo)
        (repo / "f").write_text("1", encoding="utf-8")
        older = commit(repo)
        (repo / "f").write_text("2", encoding="utf-8")
        newer = commit(repo)
        subprocess.run(["git", "-C", str(repo), "checkout", "-q", "-b", "side", base], check=True)
        (repo / "g").write_text("x", encoding="utf-8")
        other = commit(repo)
        return repo, older, newer, other

    def sha_campaign(self, name, sha, rows):
        """``rows``: ``{row id: epic decode-rate factor}`` (the baseline decodes at 50 tok/s)."""

        def row(row_id, factor):
            def make(run, k):
                def value(prompt, option, metric):
                    base = 50.0 if metric == "decode_tok_s" else 100.0
                    if run == "epic" and metric == "decode_tok_s":
                        base *= factor
                    return base + jitter(k)

                return document(run, value=value, epic_sha=sha)

            return {"id": row_id, "document": make}

        return write_campaign(
            self.directory / name, [row(r, f) for r, f in rows.items()], epic_sha=sha
        )

    def test_a_row_re_measured_at_a_descendant_commit_supersedes_the_older_measurement(self) -> None:
        repo, older, newer, _ = self.history()
        stale = self.sha_campaign("c-old", older, {"qwen38-bf16": 0.7, "bonsai": 1.0})
        fresh = self.sha_campaign("c-new", newer, {"qwen38-bf16": 1.0})
        for order in ((stale, fresh), (fresh, stale)):
            report = campaign.compare_campaigns(list(order), self.DEFAULTS, repo=repo)
            self.assertEqual(
                [(s["row"], s["campaign"], s["epic_sha"], s["superseded_by"]) for s in report["superseded"]],
                [("qwen38-bf16", "c-old", older, f"c-new@{newer}")],
            )
            # The stale 30%-slower measurement is gone from every verdict; the row only the older
            # campaign measured stays, attributed to its commit.
            self.assertTrue(all(e["verdict"] == "pass" for e in report["e6"]), report["e6"])
            self.assertEqual(
                {(r["row"], r["campaign"], r["epic_sha"]) for r in report["rows"]},
                {("qwen38-bf16", "c-new", newer), ("bonsai", "c-old", older)},
            )
            self.assertFalse(any(c["campaign"] == "c-old" and c["row"] == "qwen38-bf16"
                                 for c in report["comparisons"]))
            markdown = campaign.markdown_report(report)
            self.assertIn("## Superseded", markdown)
            self.assertIn(f"`{older[:12]}`", markdown)

    def test_unordered_or_conflicting_re_measurements_are_refused(self) -> None:
        repo, older, newer, other = self.history()
        cases = {
            "neither is an ancestor": (newer, other, "no measurement descends"),
            "same commit, different data": (newer, newer, "with different data"),
            "unknown commit": (newer, "e" * 40, "cannot order"),
        }
        for index, (name, (first, second, message)) in enumerate(cases.items()):
            with self.subTest(name):
                a = self.sha_campaign(f"a{index}", first, {"qwen38-bf16": 1.0})
                b = self.sha_campaign(f"b{index}", second, {"qwen38-bf16": 0.9})
                with self.assertRaises(campaign.Refused) as refused:
                    campaign.compare_campaigns([a, b], self.DEFAULTS, repo=repo)
                self.assertIn(message, str(refused.exception))
        # A byte-identical copy at the same commit is a duplicate, not a conflict.
        a = self.sha_campaign("dup-a", newer, {"qwen38-bf16": 1.0})
        b = self.directory / "dup-b"
        shutil.copytree(a, b)
        report = campaign.compare_campaigns([a, b], self.DEFAULTS, repo=repo)
        self.assertEqual(
            [(s["campaign"], s["why"]) for s in report["superseded"]],
            [("dup-b", "duplicate at the same commit")],
        )

    def spec_row(self, row_id, auto_speed, *, process_repeats=3, proposer="mtp", runs=("epic",)):
        def make(run, k):
            return document(
                run,
                value=lambda p, o, m: (
                    (auto_speed if o == "auto" else 50.0) if m == "decode_tok_s" else 100.0
                )
                + jitter(k),
                row_extra=lambda p, o: {"proposer": (proposer if o == "auto" else "none")},
            )

        return {
            "id": row_id,
            "runs": list(runs),
            "process_repeats": process_repeats,
            "document": make,
        }

    def test_e5_speculative_is_judged_only_from_its_own_on_off_pairs(self) -> None:
        # An E6 row slower than the baseline is no speculative evidence: no option pair, no
        # regression, whatever the epic-vs-baseline verdict.
        def e6_row(run, k):
            def value(prompt, option, metric):
                slow = run == "epic" and metric == "decode_tok_s"
                return (25.0 if slow else 50.0 if metric == "decode_tok_s" else 100.0) + jitter(k)

            return document(run, value=value, options=["off"])

        e6_only = write_campaign(self.directory / "e6-only", [{"id": "cmp", "document": e6_row}])
        report = self.compare(e6_only)
        self.assertEqual({e["verdict"] for e in report["e6"]}, {"regression"})
        speculative = {d["entry"]: d for d in report["defaults"]}["CANDLE_CUDA.speculative"]
        self.assertEqual((speculative["outcome"], speculative["pairs"]), ("unmeasured", 0))
        # A regression cites the row and prompt it was measured on.
        slower = self.decisions([self.spec_row("spec", 30.0)])["CANDLE_CUDA.speculative"]
        self.assertEqual(slower["outcome"], "off")
        self.assertRegex(slower["reason"], r"/spec vs \S+/spec (chat|code_edit) `auto vs off`")
        # `auto` that resolved to plain decode (no proposer) is not applicable, never a regression.
        plain = self.decisions([self.spec_row("plain", 30.0, proposer="none")])[
            "CANDLE_CUDA.speculative"
        ]
        self.assertEqual({j["verdict"] for j in plain["judged"]}, {"not_applicable"})
        self.assertNotEqual(plain["outcome"], "off")
        # Unresolved pairs (here n=1) are not a regression: E5 keeps the default on.
        single = self.decisions([self.spec_row("single", 30.0, process_repeats=1)])[
            "CANDLE_CUDA.speculative"
        ]
        self.assertEqual(
            (single["outcome"], single["recommended_on"]), ("on (unresolved)", True)
        )
        self.assertEqual(
            {j["reason"] for j in single["judged"]}, {"n=1, no process-level band"}
        )
        self.assertIn("n<2, no process-level band", single["reason"])

    def test_n1_has_no_process_level_band(self) -> None:
        result = campaign.verdict([40], [50, 51, 52], True, 0.1)
        self.assertEqual(
            (result["verdict"], result["reason"]), ("inconclusive", "n=1, no process-level band")
        )

    def cache_row(self, row_id, budget, ttft, *, options=None, runs=("epic",)):
        def make(run, k):
            extra = {"options": options} if options else {}
            return document(
                run,
                prefix_cache_bytes=budget if run == "epic" else 0,
                value=lambda p, o, m: (50.0 if m == "decode_tok_s" else ttft) + jitter(k),
                **extra,
            )

        env = {"SPECULATIVE_BENCH_PREFIX_CACHE_BYTES": str(budget)} if budget else {}
        return {"id": row_id, "runs": list(runs), "env": env, "document": make}

    def test_prefix_cache_pairs_only_twin_rows_and_names_the_unpaired(self) -> None:
        twins = self.decisions(
            [self.cache_row("cache-on", 1 << 30, 140.0), self.cache_row("cache-off", 0, 100.0)]
        )["CANDLE_CUDA.prefix_cache_bytes"]
        self.assertEqual((twins["pairs"], twins["outcome"]), (1, "off"))
        # An option-list mismatch or a row interleaved with the pre-epic baseline is no twin: the
        # cache-on row is reported as unpaired instead of judged against a different regime.
        for name, off_row in {
            "options": self.cache_row("cache-off", 0, 100.0, options=["off", "auto"]),
            "schedule": self.cache_row("cache-off", 0, 100.0, runs=("epic", "baseline")),
        }.items():
            with self.subTest(name):
                decision = self.decisions([self.cache_row("cache-on", 1 << 30, 140.0), off_row])[
                    "CANDLE_CUDA.prefix_cache_bytes"
                ]
                self.assertEqual((decision["pairs"], decision["outcome"]), (0, "unmeasured"))
                self.assertEqual(len(decision["unpaired"]), 1)
                self.assertTrue(decision["unpaired"][0].endswith("/cache-on"))
                self.assertIn("unpaired, no twin row identical but for the prefix cache", decision["reason"])

    # ---- sc-24446 compare fixes: derived switches, prefix-cache miss / hit samples

    def graphs_twin(self, row_id, graphs, speed, stream_env=None):
        """A CUDA-graphs row as the harness records it: the stream switch's effective value
        follows the graph switch (own with graphs on, legacy off) unless ``stream_env`` sets it."""
        stream = stream_env or ("own" if graphs else "legacy")
        switches = {"CANDLE_LLM_CUDA_GRAPHS": "1" if graphs else "0"}
        if stream_env:
            switches["CANDLE_LLM_CUDA_STREAM"] = stream_env

        def make(run, k):
            return document(
                run,
                effective={"CANDLE_LLM_CUDA_GRAPHS": graphs, "CANDLE_LLM_CUDA_STREAM": stream},
                switch_env=switches,
                value=lambda p, o, m: (speed if m == "decode_tok_s" else 100.0) + jitter(k),
            )

        return {"id": row_id, "runs": ["epic"], "switches": switches, "document": make}

    def test_a_derived_switch_left_unset_does_not_break_the_graphs_twin(self) -> None:
        # The stream follows the graph switch (own vs legacy) with its variable unset: still twins.
        decision = self.decisions(
            [self.graphs_twin("on", True, 60.0), self.graphs_twin("off", False, 50.0)]
        )["CANDLE_CUDA.cuda_graphs"]
        self.assertEqual((decision["pairs"], decision["outcome"]), (1, "on"))
        self.assertEqual(decision["unpaired"], [])
        self.assertEqual(len(decision["judged"]), 2 * len(PROMPTS) * len(OPTIONS))
        # Set explicitly to the same stream on both rows, the stream is part of the key and equal.
        same = self.decisions(
            [
                self.graphs_twin("on", True, 60.0, stream_env="own"),
                self.graphs_twin("off", False, 50.0, stream_env="own"),
            ]
        )["CANDLE_CUDA.cuda_graphs"]
        self.assertEqual(same["pairs"], 1)
        # Set explicitly to different streams (or on one row only), the rows measured different
        # streams on purpose: not twins, both named unpaired.
        for name, off_row in {
            "different": self.graphs_twin("off", False, 50.0, stream_env="legacy"),
            "one side": self.graphs_twin("off", False, 50.0),
        }.items():
            with self.subTest(name):
                decision = self.decisions(
                    [self.graphs_twin("on", True, 60.0, stream_env="own"), off_row]
                )["CANDLE_CUDA.cuda_graphs"]
                self.assertEqual((decision["pairs"], decision["outcome"]), (0, "unmeasured"))
                self.assertEqual(len(decision["unpaired"]), 2)

    def cache_samples_row(self, row_id, budget, miss_ttft, hit_ttft, *, miss_hit_tokens=0):
        """A prefix-cache row whose per-sample TTFTs are a miss (sample 0, ``miss_hit_tokens``
        restored) then a hit (sample 1, the whole prompt but its last token restored); a
        cache-off row (``budget`` 0) runs two cold samples at ``miss_ttft``."""
        prompt_tokens = 40

        def make(run, k):
            def samples(prompt, option):
                cold = {"prefix_hit_tokens": 0, "ttft_ms": miss_ttft + jitter(k), "decode_tok_s": 50.0 + jitter(k)}
                if not budget:
                    return {"samples": [cold, dict(cold)]}
                miss = {**cold, "prefix_hit_tokens": miss_hit_tokens}
                hit = {
                    "prefix_hit_tokens": prompt_tokens - 1,
                    "ttft_ms": hit_ttft + jitter(k),
                    "decode_tok_s": 50.0 + jitter(k),
                }
                return {"samples": [miss, hit], "prompt_tokens": prompt_tokens}

            def mean(prompt, option, metric):
                values = [s[metric] for s in samples(prompt, option)["samples"]]
                return sum(values) / len(values)

            return document(run, prefix_cache_bytes=budget, value=mean, row_extra=samples)

        env = {"SPECULATIVE_BENCH_PREFIX_CACHE_BYTES": str(budget)} if budget else {}
        return {"id": row_id, "runs": ["epic"], "env": env, "document": make}

    def test_prefix_cache_judges_the_miss_and_hit_samples_separately(self) -> None:
        off_row = self.cache_samples_row("cache-off", 0, 100.0, 100.0)
        # A miss that costs 50 % more and a hit that saves 80 %: the mixed mean (105) would be
        # within noise of the cold 100, the miss path alone is a regression -> off.
        slower_miss = self.decisions(
            [self.cache_samples_row("cache-on", 1 << 30, 150.0, 20.0), off_row]
        )["CANDLE_CUDA.prefix_cache_bytes"]
        self.assertEqual((slower_miss["pairs"], slower_miss["outcome"]), (1, "off"))
        by_samples = {
            samples: {j["verdict"] for j in slower_miss["judged"] if j["samples"] == samples and j["metric"] == "ttft_ms"}
            for samples in ("miss", "hit")
        }
        self.assertEqual(by_samples, {"miss": {"regression"}, "hit": {"pass"}})
        self.assertIn("[miss samples] ttft_ms +50.0%", slower_miss["reason"])
        self.assertIn("hit samples: ", slower_miss["reason"])
        self.assertIn("ttft_ms Δ -80.0..-80.0%", slower_miss["reason"])
        # A miss as fast as no cache and a hit that saves 80 %: on, the hit's gain reported.
        free_miss = self.decisions(
            [self.cache_samples_row("cache-on", 1 << 30, 100.0, 20.0), off_row]
        )["CANDLE_CUDA.prefix_cache_bytes"]
        self.assertEqual(free_miss["outcome"], "on")
        cells = len(PROMPTS) * len(OPTIONS) * len(campaign.METRICS)
        self.assertIn(f"hit samples: {cells} judged, {cells} pass", free_miss["reason"])
        self.assertIn("ttft_ms Δ -80.0..-80.0%", free_miss["reason"])
        # A slower hit is a regression too.
        slower_hit = self.decisions(
            [self.cache_samples_row("cache-on", 1 << 30, 100.0, 150.0), off_row]
        )["CANDLE_CUDA.prefix_cache_bytes"]
        self.assertEqual(slower_hit["outcome"], "off")
        self.assertIn("[hit samples] ttft_ms +50.0%", slower_hit["reason"])
        # A first sample that restored only a few shared template tokens is still a miss.
        template = self.decisions(
            [self.cache_samples_row("cache-on", 1 << 30, 150.0, 20.0, miss_hit_tokens=3), off_row]
        )["CANDLE_CUDA.prefix_cache_bytes"]
        self.assertEqual(template["outcome"], "off")
        self.assertIn("[miss samples] ttft_ms +50.0%", template["reason"])
        # Documents without per-sample hits are judged on their in-process means, unsplit.
        legacy = self.decisions(
            [self.cache_row("cache-on", 1 << 30, 140.0), self.cache_row("cache-off", 0, 100.0)]
        )["CANDLE_CUDA.prefix_cache_bytes"]
        self.assertEqual({j["samples"] for j in legacy["judged"]}, {None})

    def test_a_justified_default_with_a_measured_pair_is_reported_as_informational(self) -> None:
        def pipe_row(row_id, state, ttft):
            def make(run, k):
                return document(
                    run,
                    backend="mlx",
                    effective={"MLX_LLM_PIPELINING": state},
                    value=lambda p, o, m: (50.0 if m == "decode_tok_s" else ttft) + jitter(k),
                )

            switches = {} if state else {"MLX_LLM_PIPELINING": "0"}
            return {"id": row_id, "runs": ["epic"], "switches": switches, "document": make}

        directory = write_campaign(
            self.directory / "pipe",
            [pipe_row("pipe-on", True, 130.0), pipe_row("pipe-off", False, 100.0)],
            lane="mlx",
        )
        decisions = {d["entry"]: d for d in self.compare(directory)["defaults"]}
        pipelining = decisions["MLX.pipelining"]
        self.assertTrue(pipelining["informational"])
        self.assertFalse(pipelining["provisional"])
        self.assertEqual((pipelining["pairs"], pipelining["outcome"]), (1, "off"))
        self.assertIn("sc-24439", pipelining["justification"])
        # A justified entry nothing measured is not listed.
        self.assertNotIn("MLX.device_sampler", decisions)
        self.assertIn("informational", campaign.markdown_report(self.compare(directory)))

    def test_family_labels_name_the_hugging_face_repository(self) -> None:
        def label(model, path, fmt="bf16"):
            row = {"snapshot": {"path": path}, "format": fmt}
            return campaign.family_label(
                campaign.Process("c", "r", "epic", 1, {"model": model}, {}, row)
            )

        hub = "/hub/models--prism-ml--Ternary-Bonsai-2-27B-mlx-2bit/snapshots/3f926b41"
        self.assertEqual(label("3f926b41", hub), "prism-ml/Ternary-Bonsai-2-27B-mlx-2bit")
        self.assertEqual(
            label("b968826d@q4", "/hub/models--Qwen--Qwen3-8B/snapshots/b968826d", "q4"),
            "Qwen/Qwen3-8B@q4",
        )
        self.assertEqual(
            label("enhancer@q4", "/hub/models--SceneWorks--ltx-2.5-mlx/snapshots/791e/enhancer", "q4"),
            "SceneWorks/ltx-2.5-mlx/enhancer@q4",
        )
        # An explicit label, or a snapshot outside the hub cache, is kept verbatim.
        self.assertEqual(label("my-label", hub), "my-label")
        self.assertEqual(label("Qwen3.8-27B-q4", "/prepared/Qwen3.8-27B-q4"), "Qwen3.8-27B-q4")
        directory = write_campaign(
            self.directory / "hub",
            [{"id": "bonsai", "snapshot": hub, "document": lambda run, k: document(run, model="3f926b41")}],
        )
        self.assertEqual(
            {e["model"] for e in self.compare(directory)["e6"]},
            {"prism-ml/Ternary-Bonsai-2-27B-mlx-2bit"},
        )

def write_safetensors(path: Path, tensors: dict[str, bytes], metadata=None) -> None:
    """A safetensors file of 1-D U8 tensors, stored in the given order."""
    header, offset, data = {}, 0, b""
    if metadata is not None:
        header["__metadata__"] = metadata
    for name, payload in tensors.items():
        header[name] = {"dtype": "U8", "shape": [len(payload)], "data_offsets": [offset, offset + len(payload)]}
        offset += len(payload)
        data += payload
    blob = json.dumps(header).encode("utf-8")
    path.write_bytes(len(blob).to_bytes(8, "little") + blob + data)


def read_safetensors(path: Path) -> tuple[dict[str, bytes], dict]:
    raw = path.read_bytes()
    length = int.from_bytes(raw[:8], "little")
    header = json.loads(raw[8 : 8 + length])
    metadata = header.pop("__metadata__", None)
    data = raw[8 + length :]
    tensors = {}
    for name, info in header.items():
        begin, end = info["data_offsets"]
        assert 0 <= begin <= end <= len(data), (name, info)
        tensors[name] = data[begin:end]
    return tensors, {"metadata": metadata, "aligned": (8 + length) % 8 == 0, "size": len(data)}


QWEN36_SHARD_1 = {
    "model.language_model.embed_tokens.weight": b"E" * 24,
    "model.language_model.layers.0.mlp.experts.gate_up_proj": b"G" * 40,
}
QWEN36_SHARD_2 = {
    "model.language_model.norm.weight": b"N" * 8,
    "mtp.fc.weight": b"F" * 16,
    "lm_head.weight": b"L" * 32,
    "mtp.layers.0.mlp.experts.down_proj": b"D" * 12,
}
QWEN36_CONFIG = {
    "architectures": ["Qwen3_5MoeForConditionalGeneration"],
    "model_type": "qwen3_5_moe",
    "text_config": {
        "model_type": "qwen3_5_moe_text",
        "num_experts": 256,
        "mtp_num_hidden_layers": 1,
        "mtp_use_dedicated_embeddings": False,
    },
}


def write_hf_snapshot(cache: Path, revision: str = "995ad96e") -> Path:
    """A Hugging Face cache snapshot of a Qwen3.6-shaped MoE: every file a symlink into `blobs`,
    the MTP head split across the last shard as the real checkpoint stores it."""
    repo = cache / "hub" / "models--Qwen--Qwen3.6-35B-A3B"
    blobs, snapshot = repo / "blobs", repo / "snapshots" / revision
    blobs.mkdir(parents=True)
    snapshot.mkdir(parents=True)
    staged = blobs / "staging"
    staged.mkdir()
    write_safetensors(staged / "1", QWEN36_SHARD_1)
    write_safetensors(staged / "2", QWEN36_SHARD_2, {"format": "pt"})
    weight_map = {k: "model-00001-of-00002.safetensors" for k in QWEN36_SHARD_1}
    weight_map.update({k: "model-00002-of-00002.safetensors" for k in QWEN36_SHARD_2})
    sizes = sum(len(v) for v in {**QWEN36_SHARD_1, **QWEN36_SHARD_2}.values())
    files = {
        "config.json": json.dumps(QWEN36_CONFIG).encode(),
        "tokenizer.json": b'{"model": {"vocab": {}}}',
        "chat_template.jinja": b"{{ messages }}",
        "model.safetensors.index.json": json.dumps(
            # A float, as the real Qwen3.6-35B-A3B index stores it.
            {"metadata": {"total_size": float(sizes)}, "weight_map": weight_map}
        ).encode(),
        "model-00001-of-00002.safetensors": (staged / "1").read_bytes(),
        "model-00002-of-00002.safetensors": (staged / "2").read_bytes(),
    }
    shutil.rmtree(staged)
    for i, (name, payload) in enumerate(files.items()):
        blob = blobs / f"blob{i}"
        blob.write_bytes(payload)
        (snapshot / name).symlink_to(Path("..") / ".." / "blobs" / blob.name)
    return snapshot


class WithoutMtpTests(unittest.TestCase):
    """sc-24446: the Candle lanes' MTP-free snapshot, so the pre-epic loader (which refuses
    Qwen3.6-35B-A3B's MoE MTP head) can measure that model's `off` baseline."""

    def setUp(self) -> None:
        self.directory = Path(self.enterContext(tempfile.TemporaryDirectory()))
        self.source = write_hf_snapshot(self.directory / "huggingface")
        self.destination = Path(campaign.without_mtp_path(str(self.source)))

    def test_the_copy_lives_beside_the_hub_in_the_cache_layout(self) -> None:
        self.assertEqual(
            self.destination,
            self.directory / "huggingface" / "prepared-without-mtp"
            / "models--Qwen--Qwen3.6-35B-A3B" / "snapshots" / "995ad96e",
        )
        self.assertEqual(
            campaign.without_mtp_path(
                "E:\\huggingface\\hub\\models--Qwen--Qwen3.6-35B-A3B\\snapshots\\995ad\\"
            ),
            "E:\\huggingface\\prepared-without-mtp\\models--Qwen--Qwen3.6-35B-A3B\\snapshots\\995ad",
        )
        self.assertIsNone(campaign.without_mtp_path("E:\\models\\qwen36"))
        self.assertIsNone(campaign.without_mtp_path("/models/models--a--b"))

    def test_every_mtp_tensor_is_dropped_and_everything_else_is_kept_byte_for_byte(self) -> None:
        self.assertTrue(campaign.prepare_without_mtp(str(self.source), str(self.destination)))
        out = self.destination
        one, one_info = read_safetensors(out / "model-00001-of-00002.safetensors")
        two, two_info = read_safetensors(out / "model-00002-of-00002.safetensors")
        self.assertEqual(one, QWEN36_SHARD_1)
        self.assertEqual(two, {k: v for k, v in QWEN36_SHARD_2.items() if not k.startswith("mtp.")})
        self.assertEqual((two_info["metadata"], two_info["aligned"]), ({"format": "pt"}, True))
        self.assertEqual(two_info["size"], 40)  # no dropped tensor's bytes left behind
        # The shard without a head tensor is the cache's own bytes, hard-linked, not copied.
        self.assertEqual(
            (out / "model-00001-of-00002.safetensors").stat().st_ino,
            (self.source / "model-00001-of-00002.safetensors").resolve().stat().st_ino,
        )
        config = json.loads((out / "config.json").read_text(encoding="utf-8"))
        expected = copy.deepcopy(QWEN36_CONFIG)
        expected["text_config"]["mtp_num_hidden_layers"] = 0
        self.assertEqual(config, expected)
        index = json.loads((out / "model.safetensors.index.json").read_text(encoding="utf-8"))
        self.assertEqual(
            index["weight_map"],
            {
                **{k: "model-00001-of-00002.safetensors" for k in QWEN36_SHARD_1},
                **{
                    k: "model-00002-of-00002.safetensors"
                    for k in QWEN36_SHARD_2
                    if not k.startswith("mtp.")
                },
            },
        )
        self.assertEqual(index["metadata"]["total_size"], 24.0 + 40 + 8 + 32)
        self.assertIsInstance(index["metadata"]["total_size"], float)
        for name in ("tokenizer.json", "chat_template.jinja"):
            self.assertEqual((out / name).read_bytes(), (self.source / name).read_bytes())
        self.assertFalse(out.with_name(out.name + ".partial").exists())
        self.assertNotEqual(
            campaign.snapshot_identity(str(out))["sha256"],
            campaign.snapshot_identity(str(self.source))["sha256"],
        )

    def test_a_flat_config_has_its_own_mtp_layers_zeroed(self) -> None:
        text = campaign._config_without_mtp('{"model_type": "qwen3_5_text", "mtp_num_hidden_layers": 1}')
        self.assertEqual(json.loads(text), {"model_type": "qwen3_5_text", "mtp_num_hidden_layers": 0})

    def test_preparing_again_reuses_the_copy_and_never_overwrites_another(self) -> None:
        self.assertTrue(campaign.prepare_without_mtp(str(self.source), str(self.destination)))
        shard = self.destination / "model-00002-of-00002.safetensors"
        before = shard.stat()
        self.assertFalse(campaign.prepare_without_mtp(str(self.source), str(self.destination)))
        self.assertEqual((shard.stat().st_ino, shard.stat().st_mtime_ns), (before.st_ino, before.st_mtime_ns))
        # Another source's content (or a hand-made directory) at the path is refused, untouched.
        (self.source / "tokenizer_config.json").write_text("{}", encoding="utf-8")
        with self.assertRaisesRegex(campaign.CampaignError, "not this preparer's"):
            campaign.prepare_without_mtp(str(self.source), str(self.destination))
        self.assertTrue(shard.exists())
        # A preparation stopped part-way is discarded and redone, never taken as finished.
        shutil.rmtree(self.destination)
        partial = self.destination.with_name(self.destination.name + ".partial")
        partial.mkdir()
        (partial / "model-00002-of-00002.safetensors").write_bytes(b"half")
        self.assertTrue(campaign.prepare_without_mtp(str(self.source), str(self.destination)))
        self.assertNotIn(
            "mtp.fc.weight", read_safetensors(self.destination / "model-00002-of-00002.safetensors")[0]
        )

    def test_the_plan_points_both_runs_at_the_copy_on_candle_lanes_only(self) -> None:
        row = {"id": "f5", "snapshot": str(self.source), "options": ["off"], "without_mtp": True}
        planned = plan({"rows": [row]})["rows"][0]
        self.assertEqual(planned["env"]["SPECULATIVE_BENCH_SNAPSHOT"], str(self.destination))
        self.assertEqual(planned["prepare"], {"without_mtp": str(self.source)})
        self.assertEqual(planned["runs"], ["epic", "baseline"])
        self.assertNotIn("prepare", plan({"rows": [{**row, "without_mtp": False}]})["rows"][0])
        for name, bad, lane in (
            ("mlx lane", row, "mlx"),
            ("not a cache snapshot", {**row, "snapshot": "/models/qwen36"}, "cuda"),
            ("not a boolean", {**row, "without_mtp": "yes"}, "cuda"),
        ):
            with self.subTest(name), self.assertRaises(campaign.MatrixError):
                plan({"rows": [bad]}, lane=lane)


if __name__ == "__main__":
    unittest.main()


# ------------------------------------------------------------------------------- SW-7200 paths

REV = "8e450d7faf38238b64580163a7d3dd7e44af759d"
TARGET_PIN = f"hub:SceneWorks/qwen3.8-27b-abliterated-mlx@{'7' * 40}/q4"
DSPINE_PIN = f"hub:SceneWorks/qwen3.8-27b-dspine-b16@{REV}"
TREE = {"proposer": "block_draft", "depth": "max", "verify": {"tree": {"max_rows": "max"}}}
LINEAR = {"proposer": "block_draft", "depth": "max", "verify": "linear"}
REC_MTP = {"proposer": "mtp", "depth": "recommended", "verify": "linear"}
REC_BLOCK = {"proposer": "block_draft", "depth": "recommended", "verify": "linear"}


class HubPinPlanTests(unittest.TestCase):
    """SW-7200: a row names its target and draft as pinned Hub snapshots; the plan binds them
    where the stage script puts them and lists each once for it."""

    def setUp(self) -> None:
        self.hub = Path(self.enterContext(tempfile.TemporaryDirectory())) / "hub"

    def plan(self, rows, **environ):
        return plan({"rows": rows}, {campaign.HUB_ROOT_ENV: str(self.hub), **environ}, "mlx")

    def test_pins_bind_under_the_hub_root_and_are_listed_once_for_the_stage_script(self) -> None:
        rows = [
            {"id": "q4-dspine", "snapshot": TARGET_PIN, "draft": DSPINE_PIN, "runs": ["epic"],
             "options": ["off", LINEAR, TREE]},
            {"id": "q4-plain", "snapshot": TARGET_PIN, "runs": ["epic"]},
        ]
        planned = self.plan(rows)
        root = Path(os.path.realpath(self.hub))
        target = root / "models--SceneWorks--qwen3.8-27b-abliterated-mlx" / "snapshots" / ("7" * 40) / "q4"
        draft = root / "models--SceneWorks--qwen3.8-27b-dspine-b16" / "snapshots" / REV
        env = planned["rows"][0]["env"]
        self.assertEqual(env["SPECULATIVE_BENCH_SNAPSHOT"], str(target))
        self.assertEqual(env["SPECULATIVE_BENCH_DRAFT"], str(draft))
        self.assertEqual(json.loads(env["SPECULATIVE_BENCH_OPTIONS"]), ["off", LINEAR, TREE])
        self.assertEqual(planned["asset_setup"], "hub-snapshots")
        self.assertEqual(
            planned["hub_snapshots"],
            {
                "DECODE_BENCH_HUB_1": {"repository": "SceneWorks/qwen3.8-27b-abliterated-mlx",
                                       "revision": "7" * 40, "subfolder": "q4"},
                "DECODE_BENCH_HUB_2": {"repository": "SceneWorks/qwen3.8-27b-dspine-b16",
                                       "revision": REV, "subfolder": ""},
            },
        )
        # The stage script binds each pin to exactly the path the plan gave the row.
        sys.path.insert(0, str(REPOSITORY))
        try:
            from scripts.ci import stage_pinned_hub_snapshots as stage
        finally:
            sys.path.remove(str(REPOSITORY))
        with contextlib.redirect_stdout(io.StringIO()):
            receipt = stage.stage(
                planned["hub_snapshots"],
                root,
                manifest=lambda repository, revision: (
                    None, None, {"config.json": {"size": 1}, "q4/config.json": {"size": 1}}
                ),
                fetch=lambda url, path: (None, None, None, False),
                verify=lambda repository, revision, base, files: {
                    "status": "ok", "manifest_sha256": "0", "files": []
                },
                free_bytes=lambda path: 1 << 50,
            )
        self.assertEqual(receipt["DECODE_BENCH_HUB_1"]["path"], str(target))
        self.assertEqual(receipt["DECODE_BENCH_HUB_2"]["path"], str(draft))
        # No pin, no staging keys.
        self.assertNotIn("hub_snapshots", plan({"rows": [{"id": "r", "snapshot": "qwen38"}]}))

    def test_a_malformed_pin_or_a_missing_hub_root_is_refused(self) -> None:
        for bad in (
            "hub:SceneWorks/qwen@main/q4",
            f"hub:SceneWorks@{REV}",
            f"hub:SceneWorks/qwen@{REV}/../q4",
            f"hub:SceneWorks/qwen@{REV}/q4/",
            f"hub:../qwen@{REV}",
        ):
            with self.subTest(pin=bad), self.assertRaises(campaign.MatrixError):
                self.plan([{"id": "r", "snapshot": bad, "runs": ["epic"]}])
        with self.assertRaisesRegex(campaign.MatrixError, campaign.HUB_ROOT_ENV):
            plan({"rows": [{"id": "r", "snapshot": TARGET_PIN}]}, {}, "mlx")
        with self.assertRaisesRegex(campaign.MatrixError, "draft"):
            self.plan([{"id": "r", "snapshot": TARGET_PIN, "draft": "hub:x", "runs": ["epic"]}])

    def test_path_options_are_checked_before_anything_builds(self) -> None:
        good = [
            "off", "auto", {"proposer": "mtp", "depth": 3}, {"mode": "auto"}, LINEAR, TREE,
            {"proposer": "prompt_lookup", "depth": 4, "verify": {"tree": {"max_rows": 8}}},
            REC_MTP, REC_BLOCK,
        ]
        self.plan([{"id": "r", "snapshot": TARGET_PIN, "runs": ["epic"], "options": good}])
        for bad in (
            "sometimes",
            7,
            {"proposer": "medusa", "depth": 3},
            {"proposer": "block_draft", "depth": 0},
            {"proposer": "block_draft", "depth": "most"},
            {"proposer": "block_draft", "depth": True},
            {"proposer": "block_draft", "depth": "max", "format": "dspine"},
            {"proposer": "block_draft", "depth": "max", "verify": "tree"},
            {"proposer": "block_draft", "depth": "max", "verify": {"tree": {}}},
            {"proposer": "block_draft", "depth": "max", "verify": {"tree": {"max_rows": 0}}},
            {"proposer": "block_draft", "depth": "max",
             "verify": {"tree": {"max_rows": "recommended"}}},
            {"proposer": "block_draft", "depth": "max", "verify": {"tree": {"max_rows": 4}, "x": 1}},
        ):
            with self.subTest(option=bad), self.assertRaises(campaign.MatrixError):
                self.plan([{"id": "r", "snapshot": TARGET_PIN, "runs": ["epic"], "options": [bad]}])


class EpicOnlyRunTests(unittest.TestCase):
    """SW-7200: a plan whose rows run only the epic needs no pre-epic checkout."""

    setUp = RunTests.setUp

    def run_epic_only(self, rows, *, pre_epic: bool):
        output = self.directory / "out"
        write_snapshot(self.snapshot)
        plan_path = self.directory / "plan.json"
        plan_path.write_text(
            json.dumps(plan({"rows": rows}, {"DECODE_BENCH_SNAPSHOT_QWEN38": str(self.snapshot)})),
            encoding="utf-8",
        )
        args = ["run", "--plan", str(plan_path), "--epic-root", str(self.directory / "epic"),
                "--epic-sha", EPIC_SHA, "--output", str(output), "--cargo", sys.executable]
        if pre_epic:
            args += ["--pre-epic-root", str(self.directory / "pre-epic")]
        environ = {"FAKE_RECORD": str(self.record)}
        with mock.patch.dict(os.environ, environ), contextlib.redirect_stdout(
            io.StringIO()
        ), contextlib.redirect_stderr(io.StringIO()):
            return campaign.main(args)

    def test_an_epic_only_plan_runs_without_the_pre_epic_checkout(self) -> None:
        code = self.run_epic_only(
            [{"id": "q", "snapshot": "qwen38", "runs": ["epic"], "process_repeats": 1}],
            pre_epic=False,
        )
        self.assertEqual(code, 0)
        self.assertEqual(self.checked, [("epic", EPIC_SHA)])
        code = self.run_epic_only(
            [{"id": "q", "snapshot": "qwen38", "process_repeats": 1}], pre_epic=False
        )
        self.assertEqual(code, 1)


def path_row(prompt, prompt_class, requested, *, tok_s, wall, proposer="none", draft=None,
             verify="linear", hits=(0, 0), accepted=None, draft_ms=None, sync=True, tree=None,
             fallbacks=()):
    samples = [
        {"ttft_ms": 100.0, "decode_tok_s": tok_s, "decode_ms": 256 / tok_s * 1e3,
         "wall_ms": wall, "prefix_hit_tokens": hit, "draft_ms": draft_ms,
         "memory_before_bytes": 20 << 30, "peak_memory_bytes": (21 << 30) + (1 << 20)}
        for hit in hits
    ]
    return {
        "prompt_id": prompt, "class": prompt_class, "requested": requested,
        "speculative": requested, "verify": verify, "proposer": proposer,
        "prompt_tokens": 100, "generated_tokens": 256,
        "verify_steps": 256 if accepted is None else round(256 / (1 + accepted)),
        "mean_accepted_length": accepted, "samples": samples, "draft": draft,
        "draft_time": None if draft_ms is None else {
            "calls": 50, "host_synchronized_calls": 50 if sync else 0,
            "total_ms": draft_ms, "per_call_ms": draft_ms / 50},
        "tree_verify": tree, "fallbacks": list(fallbacks),
    }


def paths_document(rows, *, draft_source="/hub/draft", refusal=None, model="qwen3.8-27b"):
    return {
        "schema": campaign.SCHEMA, "model": model, "backend": "mlx", "max_new_tokens": 256,
        "sampling": {"temperature": 0.0}, "thinking": {"setting": "off"}, "warmup": True,
        "repeats": 2, "options": [r["requested"] for r in rows],
        "load": {"prefix_cache_bytes": 0,
                 "draft": {"source": draft_source, "refusal": refusal}},
        "provenance": {"git": {"sha": EPIC_SHA}}, "rows": rows,
        "memory": {"source": "mlx_allocator", "peak": "backend_high_water"},
    }


class PathsReportTests(unittest.TestCase):
    """SW-7200: the paired paths report reads every path against its own process's `off` row."""

    def setUp(self) -> None:
        self.directory = Path(self.enterContext(tempfile.TemporaryDirectory()))

    def campaign(self):
        dspine = {"source": "/hub/dspine", "format": "dspine", "precision": "bf16"}
        dflash = {"source": "/hub/dflash2", "format": "dflash2", "precision": "bf16"}
        tree = {"accepted_per_tree_pass": 3.5, "rows_per_pass": 9.0}

        def a(run, k):
            return paths_document([
                row
                for prompt, cls, fast in (("code_edit", "predictable", 3.0), ("chat", "open_ended", 1.5))
                for row in (
                    path_row(prompt, cls, "off", tok_s=10.0, wall=26000.0),
                    path_row(prompt, cls, REC_MTP,
                             tok_s=15.0, wall=18000.0, proposer="mtp", accepted=1.0, draft_ms=500.0,
                             sync=False),
                    # The max-depth MTP row is reported, never used for `auto`'s inputs.
                    path_row(prompt, cls, {"proposer": "mtp", "depth": "max", "verify": "linear"},
                             tok_s=40.0, wall=7000.0, proposer="mtp", accepted=2.0),
                    path_row(prompt, cls, LINEAR, tok_s=10.0 * fast, wall=26000.0 / fast,
                             proposer="block_draft", draft=dspine, accepted=fast - 1, draft_ms=900.0),
                    path_row(prompt, cls, REC_BLOCK, tok_s=10.0 * fast, wall=26000.0 / fast,
                             proposer="block_draft", draft=dspine, accepted=fast - 1, draft_ms=900.0),
                    path_row(prompt, cls, TREE, tok_s=10.0 * fast * 1.1, wall=26000.0 / fast / 1.1,
                             proposer="block_draft", draft=dspine, verify={"tree": {"max_rows": 16}},
                             accepted=fast - 0.9, draft_ms=950.0, tree=tree),
                )
            ])

        def b(run, k):
            # A slower process: its own `off` is 8 tok/s, so its ratios stay paired.
            return paths_document([
                row
                for prompt, cls in (("code_edit", "predictable"), ("chat", "open_ended"))
                for row in (
                    path_row(prompt, cls, "off", tok_s=8.0, wall=32000.0),
                    path_row(prompt, cls, LINEAR, tok_s=8.0 * 1.2, wall=32000.0 / 1.2,
                             proposer="block_draft", draft=dflash, accepted=0.4, draft_ms=600.0),
                    path_row(prompt, cls, REC_BLOCK, tok_s=8.0 * 1.2, wall=32000.0 / 1.2,
                             proposer="block_draft", draft=dflash, accepted=0.4, draft_ms=600.0),
                )
            ], draft_source="/hub/dflash2")

        def probe(run, k):
            return paths_document(
                [path_row("code_edit", "predictable",
                          {"proposer": "draft_model", "depth": 4}, tok_s=10.0, wall=2000.0,
                          fallbacks=["speculative: `draft_model` is not available"])],
                refusal="draft model: its tokenizer vocabulary is not the target's",
            )

        def reused(run, k):
            return paths_document([
                path_row("chat", "open_ended", "off", tok_s=10.0, wall=6000.0, hits=(0, 90)),
                path_row("chat", "open_ended", LINEAR, tok_s=20.0, wall=3000.0, hits=(90, 90),
                         proposer="block_draft", draft=dspine, accepted=1.0, draft_ms=100.0),
            ])

        rows = [
            {"id": "q4-dspine", "runs": ["epic"], "process_repeats": 1, "document": a,
             "snapshot": "/hub/models--SceneWorks--qwen3.8-27b-abliterated-mlx/snapshots/r/q4"},
            {"id": "q4-dflash2", "runs": ["epic"], "process_repeats": 1, "document": b,
             "snapshot": "/hub/models--SceneWorks--qwen3.8-27b-abliterated-mlx/snapshots/r/q4"},
            {"id": "q4-draft-model", "runs": ["epic"], "process_repeats": 1, "document": probe,
             "snapshot": "/hub/models--SceneWorks--qwen3.8-27b-abliterated-mlx/snapshots/r/q4"},
            {"id": "q4-reused", "runs": ["epic"], "process_repeats": 1, "document": reused,
             "env": {"SPECULATIVE_BENCH_PREFIX_CACHE_BYTES": "default"},
             "snapshot": "/hub/models--SceneWorks--qwen3.8-27b-abliterated-mlx/snapshots/r/q4"},
        ]
        return write_campaign(self.directory / "mlx", rows, lane="mlx")

    def summary(self, report, path, context="cold", cache="0"):
        (match,) = [
            s for s in report["summary"]
            if s["path"] == path and s["context"] == context
            and s["group"]["prefix_cache_bytes"] == cache
        ]
        return match

    def test_every_path_is_measured_against_its_own_processs_off_row(self) -> None:
        report = campaign.paths_report([self.campaign()])
        self.assertEqual(report["refusals"], [])
        self.assertEqual(report["schema"], campaign.PATHS_SCHEMA)
        dspine = self.summary(report, "block_draft:dspine:linear")
        self.assertEqual(dspine["group"]["tier"], "q4")
        self.assertAlmostEqual(dspine["speedup_vs_off"]["decode_tok_s"]["predictable"], 3.0)
        self.assertAlmostEqual(dspine["speedup_vs_off"]["decode_tok_s"]["open_ended"], 1.5)
        self.assertAlmostEqual(dspine["speedup_vs_off"]["wall_ms"]["all"], math_sqrt(4.5))
        self.assertAlmostEqual(dspine["draft_ms_per_pass"], 18.0)
        self.assertTrue(dspine["draft_time_host_synchronized"])
        self.assertEqual(dspine["memory_added_bytes"], (1 << 30) + (1 << 20))
        tree = self.summary(report, "block_draft:dspine:tree")
        self.assertEqual(tree["accepted_per_tree_pass"], 3.5)
        # DFlash2's process ran its own `off` at 8 tok/s: its ratio is 1.2, not 9.6 / 10.
        dflash = self.summary(report, "block_draft:dflash2:linear")
        self.assertAlmostEqual(dflash["speedup_vs_off"]["decode_tok_s"]["all"], 1.2)
        # The MTP draft time was enqueue-only (drafts left on the device).
        self.assertFalse(self.summary(report, "mtp@recommended")["draft_time_host_synchronized"])
        self.assertAlmostEqual(self.summary(report, "mtp")["speedup_vs_off"]["decode_tok_s"]["all"], 4.0)
        # The draft-model path ran nothing, and says why.
        probe = self.summary(report, "draft_model")
        self.assertEqual(probe["ran"], ["none"])
        self.assertIn("tokenizer vocabulary", probe["draft_refusal"])
        # Reused-context samples are their own context.
        reused = self.summary(report, "block_draft:dspine:linear", "reused", "default")
        self.assertEqual(reused["prompts"], 1)
        self.assertAlmostEqual(reused["speedup_vs_off"]["decode_tok_s"]["all"], 2.0)

    def test_auto_inputs_compare_each_block_draft_with_mtp_through_off(self) -> None:
        report = campaign.paths_report([self.campaign()])
        inputs = {(i["path"], i["context"]): i for i in report["auto_inputs"]}
        # Only what `auto` would run: each block draft at its recommended depth, linear.
        self.assertEqual(
            sorted(inputs),
            [("block_draft:dflash2:linear@recommended", "cold"),
             ("block_draft:dspine:linear@recommended", "cold")],
        )
        dspine = inputs[("block_draft:dspine:linear@recommended", "cold")]["decode_tok_s"]
        # MTP at its recommended depth is 1.5x off in process A (the 4x max-depth row is not the
        # comparison); DSpine 3.0x / 1.5x: 2.0x / 1.0x over it.
        self.assertAlmostEqual(dspine["vs_mtp"], math_sqrt(2.0))
        self.assertAlmostEqual(dspine["vs_mtp_min"], 1.0)
        self.assertFalse(dspine["beats_mtp_on_every_prompt"])
        # DFlash2's process has no MTP row; its ratio meets MTP's through each process's `off`.
        dflash = inputs[("block_draft:dflash2:linear@recommended", "cold")]["decode_tok_s"]
        self.assertAlmostEqual(dflash["vs_mtp"], 0.8)
        self.assertFalse(dflash["beats_mtp_on_every_prompt"])

    def test_the_paths_command_writes_both_reports_and_fails_on_a_refusal(self) -> None:
        directory = self.campaign()
        out = self.directory / "report"
        with contextlib.redirect_stderr(io.StringIO()):
            code = campaign.main(["paths", "--campaign", str(directory), "--json",
                                  str(out.with_suffix(".json")), "--markdown",
                                  str(out.with_suffix(".md"))])
        self.assertEqual(code, 0)
        markdown = out.with_suffix(".md").read_text(encoding="utf-8")
        self.assertIn("| block_draft:dspine:tree | block_draft |", markdown)
        self.assertIn("## Block-draft inputs for `auto`", markdown)
        self.assertIn("enqueue-only", markdown)
        summary = json.loads((directory / "summary.json").read_text(encoding="utf-8"))
        summary["rows"][0]["processes"][0]["status"] = "failed"
        (directory / "summary.json").write_text(json.dumps(summary), encoding="utf-8")
        with contextlib.redirect_stderr(io.StringIO()):
            code = campaign.main(["paths", "--campaign", str(directory), "--json",
                                  str(self.directory / "again.json"), "--markdown",
                                  str(self.directory / "again.md")])
        self.assertEqual(code, 1)


def math_sqrt(value: float) -> float:
    import math

    return math.sqrt(value)


class TimeBudgetTests(unittest.TestCase):
    """SW-7200: on a loaned host the runner starts no process once its budget has passed, and
    never interrupts the one running."""

    setUp = RunTests.setUp

    def test_no_process_starts_after_the_budget_and_the_row_stays_incomplete(self) -> None:
        output = self.directory / "out"
        output.mkdir()
        rows = [{"id": f"q{i}", "snapshot": "qwen38", "runs": ["epic"], "process_repeats": 2}
                for i in (1, 2)]
        planned = plan({"rows": rows}, {"DECODE_BENCH_SNAPSHOT_QWEN38": str(self.snapshot)})
        ticks = iter([0.0, 10.0, 20.0, 400.0, 500.0])
        with mock.patch.dict(os.environ, {"FAKE_RECORD": str(self.record)}), \
                contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
            code = campaign.run_plan(
                planned, roots={"epic": self.directory / "epic"}, target_dirs={},
                epic_sha=EPIC_SHA, output=output, cargo=sys.executable, guard=[],
                sha_override=False, stop_starting_after=300.0, clock=lambda: next(ticks),
            )
        self.assertEqual(code, 1)
        summary = json.loads((output / "summary.json").read_text(encoding="utf-8"))
        self.assertIn("time budget", summary["stopped"])
        self.assertFalse(summary["complete"])
        first, second = summary["rows"]
        self.assertTrue(first["complete"])
        self.assertEqual(len(first["processes"]), 2)
        self.assertFalse(second["complete"])
        self.assertEqual(second["processes"], [])
        runs = [json.loads(line) for line in self.record.read_text(encoding="utf-8").splitlines()]
        self.assertEqual(sum(not r["build"] for r in runs), 2)


class GuardTests(unittest.TestCase):
    """SW-7200: every benchmark process runs under the external memory guard at its row's cap,
    and the guard's closing summary is kept beside the process."""

    setUp = RunTests.setUp

    def test_the_rows_cap_reaches_the_guard_and_its_summary_is_recorded(self) -> None:
        guard = self.directory / "summary_guard.py"
        guard.write_text(textwrap.dedent("""
            import subprocess, sys
            cap = sys.argv[1]
            code = subprocess.call(sys.argv[4:])
            print(f"GUARD_SUMMARY exit={code} root_lifetime_max_bytes=123"
                  f" max_process_lifetime_max_bytes=456 cap_gib={cap}", flush=True)
            sys.exit(code)
        """), encoding="utf-8")
        output = self.directory / "out"
        output.mkdir()
        rows = [{"id": "q4", "snapshot": "qwen38", "runs": ["epic"], "process_repeats": 1,
                 "memory_cap_gib": 40}]
        planned = plan({"rows": rows}, {"DECODE_BENCH_SNAPSHOT_QWEN38": str(self.snapshot)})
        self.assertEqual(planned["rows"][0]["memory_cap_gib"], 40)
        with mock.patch.dict(os.environ, {"FAKE_RECORD": str(self.record)}), \
                contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
            code = campaign.run_plan(
                planned, roots={"epic": self.directory / "epic"}, target_dirs={},
                epic_sha=EPIC_SHA, output=output, cargo=sys.executable,
                guard=[sys.executable, str(guard), "{cap_gib}", "60", "--"], sha_override=False,
            )
        summary = json.loads((output / "summary.json").read_text(encoding="utf-8"))
        self.assertEqual(code, 0, summary)
        self.assertEqual(
            summary["rows"][0]["processes"][0]["guard"],
            {"exit": 0, "root_lifetime_max_bytes": 123, "max_process_lifetime_max_bytes": 456,
             "cap_gib": 40},
        )
        # A guard that names a cap refuses a row without one.
        del planned["rows"][0]["memory_cap_gib"]
        out2 = self.directory / "out2"
        out2.mkdir()
        with mock.patch.dict(os.environ, {"FAKE_RECORD": str(self.record)}), \
                contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
            code = campaign.run_plan(
                planned, roots={"epic": self.directory / "epic"}, target_dirs={},
                epic_sha=EPIC_SHA, output=out2, cargo=sys.executable,
                guard=[sys.executable, str(guard), "{cap_gib}", "60", "--"], sha_override=False,
            )
        self.assertEqual(code, 1)
        process = json.loads((out2 / "summary.json").read_text(encoding="utf-8"))["rows"][0]["processes"][0]
        self.assertIn("memory_cap_gib", process["error"])
        for bad in (0, -1, True, "40", 2048):
            with self.subTest(cap=bad), self.assertRaises(campaign.MatrixError):
                plan({"rows": [{**rows[0], "memory_cap_gib": bad}]},
                     {"DECODE_BENCH_SNAPSHOT_QWEN38": str(self.snapshot)})


@unittest.skipUnless(sys.platform == "darwin", "the guard reads macOS process ledgers")
class MlxTreeGuardTests(unittest.TestCase):
    """SW-7200: the MLX campaign's external guard refuses without headroom and otherwise runs the
    command, passing its output through and closing with a parseable summary."""

    GUARD = REPOSITORY / "scripts" / "ci" / "mlx_tree_guard.py"

    def run_guard(self, *args):
        return subprocess.run([sys.executable, str(self.GUARD), *args], capture_output=True,
                              text=True, encoding="utf-8", check=False)

    def test_no_headroom_refuses_before_starting(self) -> None:
        result = self.run_guard("100000", "60", "--", sys.executable, "-c", "print('ran')")
        self.assertEqual(result.returncode, 3)
        self.assertNotIn("ran", result.stdout)
        self.assertIn("not started", result.stdout)

    def test_a_command_runs_and_the_summary_parses(self) -> None:
        result = self.run_guard("1", "60", "--", sys.executable, "-c", "print('ran')")
        if result.returncode == 3:
            self.skipTest("this host has under 11 GiB free")
        self.assertEqual(result.returncode, 0, result.stdout)
        self.assertIn("ran", result.stdout)
        log = Path(self.enterContext(tempfile.TemporaryDirectory())) / "p.log"
        log.write_text(result.stdout, encoding="utf-8")
        summary = campaign.guard_summary(log)
        self.assertEqual((summary["exit"], summary["cap_gib"]), (0, 1))
        self.assertGreater(summary["max_process_lifetime_max_bytes"], 0)


class CudaCardSelectionTests(unittest.TestCase):
    """SW-7200: the CUDA half binds its card by UUID and refuses when it is absent or ambiguous."""

    def setUp(self) -> None:
        path = REPOSITORY / "scripts" / "ci" / "decode_bench_select_cuda_card.py"
        spec = importlib.util.spec_from_file_location("decode_bench_select_cuda_card", path)
        self.select = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(self.select)
        self.directory = Path(self.enterContext(tempfile.TemporaryDirectory()))

    UUID = "GPU-e4b79931-7be6-f216-460a-f5405cfafffe"
    INVENTORY = (
        "0, GPU-11111111-2222-3333-4444-555555555555, 00000000:01:00.0, RTX PRO 6000, 97887, 1, 97886\n"
        f"1, {UUID}, 00000000:41:00.0, RTX PRO 6000, 97887, 3, 97884\n"
    )

    def test_the_card_is_bound_by_uuid_and_recorded(self) -> None:
        env_file = self.directory / "github_env"
        with mock.patch.object(self.select, "query", return_value=self.INVENTORY), \
                mock.patch.dict(os.environ, {"GITHUB_ENV": str(env_file)}), \
                contextlib.redirect_stdout(io.StringIO()):
            self.assertEqual(self.select.main([self.UUID, str(self.directory)]), 0)
        self.assertEqual(env_file.read_text(encoding="utf-8"), f"CUDA_VISIBLE_DEVICES={self.UUID}\n")
        card = json.loads((self.directory / "cuda-selected-device.json").read_text(encoding="utf-8"))
        self.assertEqual((card["index"], card["pci.bus_id"]), ("1", "00000000:41:00.0"))
        inventory = (self.directory / "cuda-device-inventory.csv").read_text(encoding="utf-8")
        self.assertTrue(inventory.startswith("index, uuid, pci.bus_id"))

    def test_an_absent_duplicated_or_empty_inventory_is_refused(self) -> None:
        other = "GPU-00000000-0000-0000-0000-000000000000"
        for inventory in ("", self.INVENTORY.replace(self.UUID, other),
                          self.INVENTORY + self.INVENTORY.splitlines()[1] + "\n", "1, broken\n"):
            env_file = self.directory / "github_env"
            env_file.unlink(missing_ok=True)
            with self.subTest(inventory=inventory), \
                    mock.patch.object(self.select, "query", return_value=inventory), \
                    mock.patch.dict(os.environ, {"GITHUB_ENV": str(env_file)}), \
                    contextlib.redirect_stderr(io.StringIO()):
                self.assertEqual(self.select.main([self.UUID, str(self.directory)]), 1)
                self.assertFalse(env_file.exists())
        with mock.patch.dict(os.environ, {"GITHUB_ENV": str(self.directory / "e")}), \
                contextlib.redirect_stderr(io.StringIO()):
            self.assertEqual(self.select.main(["GPU-1", str(self.directory)]), 1)
