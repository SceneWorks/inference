"""Source routing and proof protocol only: never invokes Cargo, devices, or model execution."""
from __future__ import annotations
import os
import io
from contextlib import redirect_stderr
import json
from pathlib import Path
import subprocess
import tempfile
import unittest
from unittest import mock
import yaml
from scripts.ci import moss_fixture_cpu as fixture
from scripts.ci import moss_fixture_cpu_profile as profile
from scripts.ci.moss_cuda_unit import UnitFailure

ROOT = Path(__file__).resolve().parents[2]
WORKFLOW = ROOT / ".github/workflows/ci.yml"
CPU_BRANCH = '''          elif [[ "$EVENT_NAME" == "workflow_dispatch" && "$DISPATCH_LANES" == "moss-cpu-fixture" ]]; then
            python3 scripts/ci/select_lanes.py --only workspace --github-output "$GITHUB_OUTPUT"
'''


class HostedCpuRoutingTests(unittest.TestCase):
    def workflow(self):
        return yaml.load(WORKFLOW.read_text(encoding="utf-8"), Loader=yaml.BaseLoader)

    def selector(self):
        return next(step["run"] for step in self.workflow()["jobs"]["changes"]["steps"]
                    if step.get("id") == "select")

    def fixture_repository(self, path):
        subprocess.run(["git", "init", "--initial-branch=main", str(path)], check=True, capture_output=True)
        script = path / "scripts/ci/select_lanes.py"
        script.parent.mkdir(parents=True)
        script.write_bytes((ROOT / "scripts/ci/select_lanes.py").read_bytes())
        subprocess.run(["git", "add", "--", "scripts/ci/select_lanes.py"], cwd=path, check=True, capture_output=True)
        self.fixture_commit(path, "baseline")
        return path

    def fixture_commit(self, path, message):
        subprocess.run(["git", "-c", "user.name=fixture", "-c", "user.email=fixture@example.invalid",
                        "commit", "-m", message], cwd=path, check=True, capture_output=True)

    def change_fixture(self, path):
        changed = path / "crates/media/mlx-gen/mlx-gen-moss-image/src/training.rs"
        changed.parent.mkdir(parents=True, exist_ok=True)
        previous = changed.read_text(encoding="utf-8") if changed.exists() else ""
        changed.write_text(previous + "// owned nonempty route fixture change\n", encoding="utf-8")
        subprocess.run(["git", "add", "--", str(changed.relative_to(path))], cwd=path, check=True, capture_output=True)
        self.fixture_commit(path, "nonempty MOSS change")

    def select(self, script, event, lanes, ref_type="branch", repository=None, base=None):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            output = root / "output"
            path_selected = event != "workflow_dispatch" and ref_type != "tag"
            if repository is None:
                repository = ROOT
                if path_selected:
                    repository = self.fixture_repository(root / "repository")
                    self.change_fixture(repository)
            if base is None:
                base = "HEAD^" if path_selected else "HEAD"
            env = dict(os.environ, EVENT_NAME=event, DISPATCH_LANES=lanes, REF_TYPE=ref_type,
                       GITHUB_OUTPUT=str(output), BASE_SHA=base, HEAD_SHA="HEAD")
            subprocess.run(["bash", "-e", "-c", script], cwd=repository, env=env, check=True, capture_output=True)
            return dict(line.split("=", 1) for line in output.read_text(encoding="utf-8").splitlines())

    def test_exact_manual_choice_and_only_two_fixed_hosted_rows(self):
        workflow = self.workflow()
        self.assertEqual(workflow["on"]["workflow_dispatch"]["inputs"]["lanes"]["options"],
                         ["all", "windows-cuda", "moss-cuda-unit", "moss-cpu-fixture"])
        self.assertEqual(set(workflow["on"]["workflow_dispatch"]["inputs"]), {"lanes"})
        job = workflow["jobs"]["moss-cpu-fixture"]
        self.assertEqual(job["if"], "github.event_name == 'workflow_dispatch' && inputs.lanes == 'moss-cpu-fixture'")
        self.assertEqual(job["strategy"]["matrix"], {"include": [
            {"backend": "candle", "runner": "ubuntu-latest"}, {"backend": "mlx", "runner": "macos-15"}]})
        self.assertEqual(job["runs-on"], "${{ matrix.runner }}")
        self.assertEqual(job["strategy"]["fail-fast"], "false")
        self.assertEqual(job["timeout-minutes"], "120")
        self.assertNotIn("continue-on-error", job)
        run = next(s for s in job["steps"] if s.get("name") == "Run actual CPU fixture and assertion mutations")
        self.assertEqual(run["run"], "python3 scripts/ci/moss_fixture_cpu_profile.py ${{ matrix.backend }}")
        self.assertNotIn("if", run)

    def test_cpu_profile_disables_every_cuda_selfhosted_and_broad_metal_lane(self):
        selected = self.select(self.selector(), "workflow_dispatch", "moss-cpu-fixture")
        self.assertEqual({name for name, value in selected.items() if value == "true"}, {"workspace"})
        for name in ("windows_cuda", "macos_metal", "moss_cuda_unit", "candle_cpu", "real_weights"):
            self.assertEqual(selected[name], "false")
        jobs = self.workflow()["jobs"]
        self.assertIn("workflow_dispatch", jobs["macos-nax"]["if"])
        self.assertIn("needs.changes.outputs.macos_metal == 'true'", jobs["macos-nax"]["if"])
        for name in ("windows-cuda-check", "windows-cuda"):
            self.assertIn("needs.changes.outputs.windows_cuda == 'true'", jobs[name]["if"])
        self.assertIn("inputs.lanes == 'moss-cuda-unit'", jobs["moss-cuda-unit"]["if"])

    def test_existing_manual_pr_push_tag_and_merge_routes_are_unchanged(self):
        source = WORKFLOW.read_text(encoding="utf-8")
        self.assertEqual(source.count(CPU_BRANCH), 1)
        # YAML removes common indentation from run blocks; remove the exact new branch only.
        addition = '\nelif [[ "$EVENT_NAME" == "workflow_dispatch" && "$DISPATCH_LANES" == "moss-cpu-fixture" ]]; then\n  python3 scripts/ci/select_lanes.py --only workspace --github-output "$GITHUB_OUTPUT"'
        current = self.selector()
        self.assertEqual(current.count(addition), 1)
        previous = current.replace(addition, "")
        for event, lanes, ref in (("workflow_dispatch", "all", "branch"),
                                  ("workflow_dispatch", "windows-cuda", "branch"),
                                  ("workflow_dispatch", "moss-cuda-unit", "branch"),
                                  ("pull_request", "", "branch"), ("push", "", "branch"),
                                  ("push", "", "tag"), ("merge_group", "", "branch")):
            with self.subTest(event=event, lanes=lanes, ref=ref):
                self.assertEqual(self.select(current, event, lanes, ref), self.select(previous, event, lanes, ref))

    def test_path_route_equivalence_works_in_actual_depth_one_detached_checkout(self):
        # Start at a true depth-one boundary, then create our own available parent/nonempty diff.
        # No historical repository SHA, network fetch, or broader CI checkout is needed.
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = self.fixture_repository(root / "source")
            self.change_fixture(source)
            checkout = root / "checkout"
            subprocess.run(["git", "clone", "--depth=1", source.as_uri(), str(checkout)], check=True, capture_output=True)
            subprocess.run(["git", "checkout", "--detach", "HEAD"], cwd=checkout, check=True, capture_output=True)
            self.change_fixture(checkout)
            shallow = subprocess.check_output(["git", "rev-parse", "--is-shallow-repository"], cwd=checkout, text=True, encoding="utf-8")
            self.assertEqual(shallow.strip(), "true")
            self.assertNotEqual(subprocess.run(["git", "rev-parse", "--verify", "HEAD^^"], cwd=checkout, capture_output=True).returncode, 0)
            delta = subprocess.check_output(["git", "diff", "--name-only", "HEAD^", "HEAD"], cwd=checkout, text=True, encoding="utf-8")
            self.assertEqual(delta.strip(), "crates/media/mlx-gen/mlx-gen-moss-image/src/training.rs")
            current = self.selector()
            addition = '\nelif [[ "$EVENT_NAME" == "workflow_dispatch" && "$DISPATCH_LANES" == "moss-cpu-fixture" ]]; then\n  python3 scripts/ci/select_lanes.py --only workspace --github-output "$GITHUB_OUTPUT"'
            previous = current.replace(addition, "")
            expected = {"workspace", "macos_metal", "real_weights"}
            for event in ("pull_request", "push", "merge_group"):
                selected = self.select(current, event, "", repository=checkout)
                self.assertEqual(selected, self.select(previous, event, "", repository=checkout))
                self.assertEqual({name for name, value in selected.items() if value == "true"}, expected)
            with self.assertRaises(subprocess.CalledProcessError):
                self.select(current, "pull_request", "", repository=checkout, base="HEAD^^")
            mutant = current.replace('--base "$BASE_SHA" --head "$HEAD_SHA"', '--only workspace')
            selected = self.select(mutant, "pull_request", "", repository=checkout)
            self.assertNotEqual({name for name, value in selected.items() if value == "true"}, expected)

    def test_requested_matrix_skip_or_failure_cannot_turn_aggregate_green(self):
        job = self.workflow()["jobs"]["gate"]
        self.assertIn("moss-cpu-fixture", job["needs"])
        gate = job["steps"][0]["run"]
        for result, success in (("success", True), ("skipped", False), ("failure", False), ("cancelled", False)):
            env = dict(os.environ, RESULTS="success skipped", REQUIRE_MOSS_CUDA_UNIT="false",
                       MOSS_CUDA_UNIT_RESULT="skipped", REQUIRE_MOSS_CPU_FIXTURE="true", MOSS_CPU_FIXTURE_RESULT=result)
            run = subprocess.run(["bash", "-c", gate], env=env, capture_output=True)
            self.assertEqual(run.returncode == 0, success)

    def test_literal_story_push_assumption_and_changed_trigger_refusal(self):
        proof = profile.push_route_proof()
        self.assertFalse(proof["matches_push_trigger"])
        self.assertEqual(proof["literal_branch"], "story/sc-26588-moss-recovery")
        self.assertEqual(self.workflow()["on"]["push"], {"branches": ["main"], "tags": ["runtime-*"]})
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            destination = root / ".github/workflows"
            destination.mkdir(parents=True)
            for name in ("ci.yml", "kv-poc-campaign.yml"):
                (destination / name).write_text((WORKFLOW.parent / name).read_text(encoding="utf-8"), encoding="utf-8")
            self.assertFalse(profile.push_route_proof(root)["matches_push_trigger"])
            path = destination / "ci.yml"
            path.write_text(path.read_text(encoding="utf-8").replace("\n    branches: [main]", "\n    branches: ['**']", 1), encoding="utf-8")
            with self.assertRaisesRegex(UnitFailure, "re-admit"):
                profile.push_route_proof(root)


class CpuReviewProtocolTests(unittest.TestCase):
    def red(self, name):
        return f"running 1 test\ntest {name} ... \nthread 'test' panicked at source.rs:1:1:\nassertion `left == right` failed\nFAILED\nfailures:\n    {name}\ntest result: FAILED. 0 passed; 1 failed; 0 ignored; 0 measured; 9 filtered out; finished in 0.1s\n"

    def test_only_actual_exact_assertion_failure_is_red(self):
        name = profile.PHASE_TEST
        output = self.red(name)
        profile.verify_red(output, name, "assertion `left == right` failed", 101)
        for code, changed in ((0, output), (-9, output), (101, output.replace("1 failed", "0 failed")),
                              (101, output.replace("running 1 test", "running 0 tests")),
                              (101, output.replace("0 ignored", "1 ignored")),
                              (101, output.replace(name, "other::test")),
                              (101, output.replace("assertion `left == right` failed", "unrelated error")),
                              (101, "error[E0001]: compile failure"), (101, output + output)):
            with self.subTest(code=code, output=changed), self.assertRaises(UnitFailure):
                profile.verify_red(changed, name, "assertion `left == right` failed", code)

    def test_mutations_have_unique_exact_live_source_anchors_and_observable_tests(self):
        for backend in fixture.PACKAGES:
            mutations = profile.mutations(backend)
            self.assertEqual(len(mutations), 7 if backend == "mlx" else 1)
            self.assertEqual(len({item["id"] for item in mutations}), len(mutations))
            for mutation in mutations:
                source = (ROOT / mutation["path"]).read_text(encoding="utf-8")
                self.assertEqual(source.count(mutation["old"]), 1, mutation["id"])
                self.assertNotEqual(mutation["old"], mutation["new"])
                self.assertTrue(mutation["test"].startswith(("training::", "training_memory::")))
                self.assertTrue(mutation["panic"])

    def test_compile_errors_are_rendered_and_stage_is_not_misreported_as_mutation(self):
        message = json.dumps({"reason": "compiler-message", "message": {
            "level": "error", "rendered": "error[E0603]: exact private import diagnostic"}})
        output = io.StringIO()
        with redirect_stderr(output):
            fixture.render_compile_errors(message)
        self.assertIn("error[E0603]: exact private import diagnostic", output.getvalue())
        for stage in ("baseline", "red", "green"):
            with tempfile.TemporaryDirectory() as directory, redirect_stderr(io.StringIO()):
                path = Path(directory) / f"{stage}.json"
                with mock.patch.object(profile.subprocess, "run", return_value=mock.Mock(returncode=101)):
                    with self.assertRaisesRegex(UnitFailure, f"{stage} compile failed; compile errors are NOT assertion RED"):
                        profile.compile_binary("candle", path)

    def test_contract_library_alias_still_requires_exact_package_and_artifact(self):
        with tempfile.TemporaryDirectory() as directory:
            target = Path(directory).resolve()
            binary = target / "gen-core-tests"
            binary.write_bytes(b"synthetic artifact protocol only")
            record = {"reason": "compiler-artifact", "package_id": "path+file:///repo/gen-core#sceneworks-gen-core@0.1.0",
                      "target": {"name": "gen_core", "kind": ["lib"]}, "profile": {"test": True}, "executable": str(binary)}
            self.assertEqual(fixture.select_binary(json.dumps(record), "sceneworks-gen-core", target, lib_name="gen_core"), binary)
            with self.assertRaises(UnitFailure):
                fixture.select_binary(json.dumps(record), "sceneworks-gen-core", target)
            record["package_id"] = "path+file:///repo/foreign#foreign@0.1.0"
            with self.assertRaises(UnitFailure):
                fixture.select_binary(json.dumps(record), "sceneworks-gen-core", target, lib_name="gen_core")

    def test_local_or_wrong_repository_cannot_start_profile(self):
        with mock.patch.dict(os.environ, {}, clear=True), mock.patch.object(profile.subprocess, "check_output") as run:
            with mock.patch.object(fixture, "git_head", return_value="a" * 40):
                with self.assertRaises(UnitFailure):
                    profile.venue("candle")
            run.assert_not_called()

    def test_source_stamp_binds_exact_tracked_delta_and_binary_digest_streams(self):
        with mock.patch.object(fixture, "git_head", return_value="a" * 40), mock.patch.object(
                fixture.subprocess, "check_output", side_effect=[b"", b"source mutation"]):
            first, second = fixture.source_stamp(), fixture.source_stamp()
        self.assertEqual(first["head"], second["head"])
        self.assertNotEqual(first["tracked_delta_sha256"], second["tracked_delta_sha256"])
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "binary"
            path.write_bytes(b"one")
            original = fixture.file_digest(path)
            path.write_bytes(b"two")
            self.assertNotEqual(fixture.file_digest(path), original)


if __name__ == "__main__":
    unittest.main()
