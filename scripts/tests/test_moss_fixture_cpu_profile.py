"""Source routing and proof protocol only: never invokes Cargo, devices, or model execution."""
from __future__ import annotations
import os
import copy
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
            self.assertEqual(len(mutations), 8 if backend == "mlx" else 1)
            self.assertEqual(len({item["id"] for item in mutations}), len(mutations))
            for mutation in mutations:
                source = (ROOT / mutation["path"]).read_text(encoding="utf-8")
                self.assertEqual(source.count(mutation["old"]), 1, mutation["id"])
                self.assertNotEqual(mutation["old"], mutation["new"])
                self.assertTrue(mutation["test"].startswith(("training::", "training_memory::", "test_support::")))
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


class DiagnosticBatchTests(unittest.TestCase):
    def source(self):
        return {"head": "a" * 40, "tracked_delta_sha256": "b" * 64}

    def demand(self):
        return {"cpu_reference": True, "token_bound_case_host_envelope_bytes": 1024,
                "cases": [{"tensor_peak_bytes": 173, "host_envelope_bytes": 1024} for _ in fixture.CASES]}

    def batch(self):
        source = self.source()
        rows = [{"test": name, "status": "passed", "owned_released": True, "source": source}
                for name in (fixture.TOKEN_TEST, *fixture.CASES)]
        return {"source": source, "final_source": source, "source_restored": True,
                "header_admitted": True, "fatal_error": None, "passed": True,
                "actual_demand": self.demand(), "cases": rows}

    def red(self, name):
        return (f"running 1 test\ntest {name} ... \nthread 'test' panicked at source.rs:1:1:\n"
                "fixed independent heldout objective did not decrease\nFAILED\n"
                f"test result: FAILED. 0 passed; 1 failed; 0 ignored; 0 measured; 9 filtered out; finished in 0.1s\n")

    def test_only_exact_assertions_aggregate_not_bare_admission_or_protocol_errors(self):
        output = self.red(profile.PHASE_TEST)
        fixture.verify_assertion_failure(output, profile.PHASE_TEST)
        for text in (output.replace("panicked at", "returned an error at"),
                     output.replace("fixed independent heldout", "Error: Unsupported admission; fixed independent heldout"),
                     output.replace("running 1 test", "running 0 tests"),
                     output.replace("0 ignored", "1 ignored"), output + output):
            with self.subTest(text=text), self.assertRaises(UnitFailure):
                fixture.verify_assertion_failure(text, profile.PHASE_TEST)

    def test_case_assertion_requires_released_ownership_and_unchanged_source(self):
        for mode in ("assertion", "ownership", "source", "deadline"):
            with self.subTest(mode=mode), tempfile.TemporaryDirectory() as directory, mock.patch("builtins.print"):
                root = Path(directory)
                binary, log = root / "binary", root / "run.log"
                binary.write_bytes(b"protocol artifact only")
                log.write_text(self.red(profile.PHASE_TEST), encoding="utf-8")
                owner = mock.Mock(peak=173)
                owner.child.returncode = 101
                owner.collect.return_value = {123: (1, 1, 173)} if mode == "ownership" else {}
                stamp = self.source()
                stamps = [stamp, {**stamp, "head": "c" * 40}] if mode == "source" else [stamp, stamp]
                failure = "exceeded the execution deadline" if mode == "deadline" else "test process exited with 101"
                with mock.patch.object(fixture, "CpuOwner", return_value=owner), mock.patch.object(
                        fixture, "source_stamp", side_effect=stamps), mock.patch.object(
                        fixture, "supervise", side_effect=UnitFailure(failure)):
                    if mode == "assertion":
                        with self.assertRaises(fixture.CaseAssertionFailure) as caught:
                            fixture.execute(binary, profile.PHASE_TEST, log)
                        self.assertTrue(caught.exception.result["owned_released"])
                        self.assertEqual(caught.exception.result["status"], "assertion_failed")
                    else:
                        with self.assertRaises(UnitFailure) as caught:
                            fixture.execute(binary, profile.PHASE_TEST, log)
                        self.assertNotIsInstance(caught.exception, fixture.CaseAssertionFailure)

    def test_receipts_bind_actual_request_context_and_hash_delta(self):
        head = self.source()["head"]
        receipt = {"request_identity": {"source_checkpoint": "synthetic-fixture:source",
                    "dataset_fingerprint": "d" * 64, "effective_config": json.dumps({"seed": 173, "steps": 6})},
                   "resolved_seed": 173, "evaluation_manifest": "e" * 64, "cpu_reference": True,
                   "factor_before": "f" * 64, "factor_after": "c" * 64}
        value = {"version": "moss-cpu-training-observations-v1", "source_head": head,
                 "evaluation": {"identity": "e" * 64, "width": 32, "height": 32, "noise_seed": 173,
                                "sigma": 0.375, "image_sha256": [0] * 32, "mask_sha256": None},
                 "receipts": [receipt], "parameter_hash_changed": [True]}
        with tempfile.TemporaryDirectory() as directory, mock.patch.dict(os.environ, {"GITHUB_SHA": head}):
            log = Path(directory) / "run.log"
            path = log.with_suffix(".receipts.json")
            initial_file=Path(directory)/"native-initial.safetensors"
            initial_file.write_bytes(b"protocol-only file identity, not a model")
            receipt["run_start_adapter"]={"path":str(initial_file),"sha256":fixture.file_digest(initial_file),
                "start_step":0,"start_update":0,"factor_tensors":1,"workspace_bytes":1024,"layout_sha256":"1"*64,
                "export_identity":[["source_checkpoint_identity",receipt["request_identity"]["source_checkpoint"]],
                    ["dataset_fingerprint",receipt["request_identity"]["dataset_fingerprint"]],
                    ["effective_training_controls",receipt["request_identity"]["effective_config"]]]}
            value["admitted_audit_workspace_bytes"]=64*1024*1024
            path.write_text(json.dumps(value), encoding="utf-8")
            self.assertEqual(fixture.read_observations(log, head, True)["runs"], 1)
            with_generation = copy.deepcopy(value)
            with_generation["generation"] = {
                "request": {"seed": 173}, "baseline_shape": [1, 1, 2, 2], "baseline_elements": 4,
                "trained_shape": [1, 1, 2, 2], "zero_strength_shape": [1, 1, 2, 2], "adapter_applied": True,
                "trained": {"elements": 4, "finite": True, "nonzero_elements_in_overlap": 1, "max_abs_delta": 0.125},
                "zero_strength": {"elements": 4, "finite": True, "nonzero_elements_in_overlap": 0, "max_abs_delta": 0.0},
                "initial_native_export":copy.deepcopy(receipt["run_start_adapter"]),
                "native_export_binding":{"layout_sha256":"1"*64,
                    "export_identity":copy.deepcopy(receipt["run_start_adapter"]["export_identity"]),
                    "initial_factor_state_sha256":"f"*64,"trained_factor_state_sha256":"c"*64},
                "route_matched_initial":{"shape":[1,1,2,2],"adapter_applied":True,
                    "initial_vs_no_adapter":{"elements":4,"finite":True,"nonzero_elements_in_overlap":1,"max_abs_delta":.125},
                    "trained_vs_initial":{"elements":4,"finite":True,"nonzero_elements_in_overlap":1,"max_abs_delta":.0625}},
            }
            path.write_text(json.dumps(with_generation), encoding="utf-8")
            self.assertTrue(fixture.read_observations(log, head, True)["generation_observed"])
            bad_generation = copy.deepcopy(with_generation)
            bad_generation["generation"]["request"]["seed"] = 999
            path.write_text(json.dumps(bad_generation), encoding="utf-8")
            with self.assertRaises(UnitFailure):
                fixture.read_observations(log, head, True)
            mutants = []
            for key, changed in (("source_head", "0" * 40), ("receipts", []), ("parameter_hash_changed", [False]), ("evaluation", {})):
                mutant = copy.deepcopy(value); mutant[key] = changed; mutants.append(mutant)
            for key, changed in (("resolved_seed", False), ("request_identity", None), ("evaluation_manifest", "missing")):
                mutant = copy.deepcopy(value); mutant["receipts"][0][key] = changed; mutants.append(mutant)
            mutant = copy.deepcopy(value)
            mutant["receipts"][0]["request_identity"]["effective_config"] = json.dumps({"seed": 0})
            mutants.append(mutant)
            for mutant in mutants:
                path.write_text(json.dumps(mutant), encoding="utf-8")
                with self.assertRaises(UnitFailure):
                    fixture.read_observations(log, head, True)
            path.unlink()
            with self.assertRaisesRegex(UnitFailure, "missing"):
                fixture.read_observations(log, head, True)

    def test_batch_continues_recorded_assertions_but_never_admission_source_or_cleanup_failure(self):
        batch = self.batch()
        profile.validate_batch(batch, 0, self.source())
        batch["cases"][1]["status"] = "assertion_failed"; batch["passed"] = False
        profile.validate_batch(batch, 1, self.source())
        mutants = []
        for key, value in (("fatal_error", "unknown ownership"), ("header_admitted", False),
                           ("source_restored", False), ("cases", [])):
            mutant = copy.deepcopy(batch); mutant[key] = value; mutants.append(mutant)
        mutant = copy.deepcopy(batch); mutant["cases"][0]["owned_released"] = False; mutants.append(mutant)
        mutant = copy.deepcopy(batch); mutant["cases"][0]["status"] = "skipped"; mutants.append(mutant)
        for mutant in mutants:
            with self.assertRaises(UnitFailure):
                profile.validate_batch(mutant, 1, self.source())
        with self.assertRaises(UnitFailure):
            profile.validate_batch(batch, 0, self.source())

    def test_independent_named_tests_continue_only_known_released_assertions(self):
        for fatal in (False, True):
            with self.subTest(fatal=fatal), tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                names = ["first", "second", "third"]
                def execute(_binary, name, log):
                    if name == "first":
                        if fatal:
                            raise UnitFailure("source readiness unknown")
                        result = {"test": name, "status": "assertion_failed"}
                        log.with_suffix(".result.json").write_text(json.dumps(result), encoding="utf-8")
                        raise fixture.CaseAssertionFailure(result)
                    log.with_suffix(".result.json").write_text(json.dumps({"test": name, "status": "passed"}), encoding="utf-8")
                    return "positive protocol"
                with mock.patch.object(fixture, "execute", side_effect=execute) as run:
                    if fatal:
                        with self.assertRaises(UnitFailure):
                            profile.independent_tests(root / "binary", names, root, "named")
                        self.assertEqual(run.call_count, 1)
                    else:
                        result = profile.independent_tests(root / "binary", names, root, "named")
                        self.assertEqual([row["status"] for row in result], ["assertion_failed", "passed", "passed"])
                        self.assertEqual(run.call_count, 3)

    def test_real_case_orchestrator_runs_all_assertion_cases_but_stops_unknown_failures(self):
        for mode in ("assertion", "fatal", "list_cleanup"):
            with self.subTest(mode=mode), tempfile.TemporaryDirectory() as directory, mock.patch("builtins.print"):
                root = Path(directory)
                binary = root / "binary"; binary.write_bytes(b"source protocol artifact")
                compiler = {"source": self.source(), "binary_sha256": fixture.file_digest(binary)}
                names = [fixture.DEMAND_TEST, fixture.TOKEN_TEST, *fixture.CASES]
                listing = "\n".join(name + ": test" for name in names)
                owner = mock.Mock()
                owner.collect.return_value = {123: (1, 1, 1)} if mode == "list_cleanup" else {}
                def execute(_binary, name, log):
                    if name == fixture.DEMAND_TEST:
                        return "MOSS_FIXTURE_DEMAND=" + json.dumps(self.demand())
                    status = "assertion_failed" if name == fixture.CASES[0] else "passed"
                    if status == "assertion_failed" and mode == "fatal":
                        raise UnitFailure("source/ownership readiness unknown")
                    result = {"test": name, "status": status, "source": self.source(), "owned_released": True}
                    log.with_suffix(".result.json").write_text(json.dumps(result), encoding="utf-8")
                    if status == "assertion_failed":
                        raise fixture.CaseAssertionFailure(result)
                    return "positive source protocol"
                with mock.patch.object(fixture, "source_stamp", return_value=self.source()), mock.patch.object(
                        fixture, "CpuOwner", return_value=owner), mock.patch.object(
                        fixture, "supervise", return_value=listing), mock.patch.object(fixture, "execute", side_effect=execute) as run:
                    if mode == "assertion":
                        result = fixture.run_cases(binary, "candle", root / "baseline.json", compiler)
                        self.assertFalse(result["passed"])
                        self.assertTrue(result["header_admitted"])
                        self.assertEqual([row["test"] for row in result["cases"]], names[1:])
                        self.assertEqual(run.call_count, len(names))
                    else:
                        with self.assertRaises(UnitFailure):
                            fixture.run_cases(binary, "candle", root / "baseline.json", compiler)
                        self.assertEqual(run.call_count, 0 if mode == "list_cleanup" else 3)
                        result = json.loads((root / "baseline.batch.json").read_text(encoding="utf-8"))
                        self.assertIsNotNone(result["fatal_error"])
                        self.assertFalse(result["passed"])

    def test_causal_controls_are_existing_unmodified_explicit_cpu_tests(self):
        candle = (ROOT / "crates/media/candle-gen/candle-gen-moss-image/src/tests/training_contract_tests.rs").read_text(encoding="utf-8")
        mlx = (ROOT / "crates/media/mlx-gen/mlx-gen-moss-image/src/transformer.rs").read_text(encoding="utf-8")
        for name in profile.CAUSAL_TESTS["candle"]:
            function = name.split("::")[-1]
            self.assertIn(f"fn {function}()", candle)
            body = candle.split(f"fn {function}()", 1)[1].split("\n#[test]", 1)[0]
            self.assertIn("native_graph_cases(Device::Cpu)", body)
        for name in profile.CAUSAL_TESTS["mlx"]:
            function = name.split("::")[-1]
            self.assertIn(f"fn {function}()", mlx)
            body = mlx.split(f"fn {function}()", 1)[1].split("\n    #[test]", 1)[0]
            self.assertIn("let _device = crate::cpu_test_device();", body)
        for name in profile.MASK_TESTS:
            self.assertIn("fn " + name.split("::")[-1] + "()", (ROOT / "crates/contracts/gen-core/src/train/subject_mask/tests.rs").read_text(encoding="utf-8"))

    def test_existing_fp32_witnesses_are_observed_before_unchanged_heldout_assertion(self):
        for backend in ("candle-gen/candle-gen-moss-image", "mlx-gen/mlx-gen-moss-image"):
            source = (ROOT / f"crates/media/{backend}/src/test_support.rs").read_text(encoding="utf-8")
            body = source.split("fn run(case: usize)", 1)[1].split("    #[test]", 1)[0]
            self.assertLess(body.index("probe.generation_observation"), body.index("assert_complete_case"))
            self.assertLess(body.index("let (disabled, disabled_shape"), body.index("probe.generation_observation"))
            self.assertIn("actual scale1 generator latents must change", body)
            self.assertIn("zero-strength adapter must not fake a generation delta", body)

    def test_real_run_start_export_and_streamed_controls_are_wired_without_fake_outputs(self):
        for backend in ("candle-gen/candle-gen-moss-image", "mlx-gen/mlx-gen-moss-image"):
            source=(ROOT/f"crates/media/{backend}/src/training.rs").read_text(encoding="utf-8")
            capture=source.split("if let Some(target) = probe.run_start_export_target()?",1)[1].split("let factors = audit::factors",1)[0]
            self.assertIn("Checkpoint",capture)
            self.assertIn("save_with_source(",capture)
            self.assertIn("probe.record_run_start_export",capture)
            self.assertLess(capture.index("save_with_source("),capture.index("record_run_start_export"))
            self.assertIn("probe.record_final_export(&path)",source)
        reader=(ROOT/"crates/contracts/gen-core-testkit/src/moss_training_probe/snapshot.rs").read_text(encoding="utf-8")
        for name in profile.SNAPSHOT_TESTS:
            self.assertIn("fn "+name.split("::")[-1]+"()",reader)
        self.assertIn("snapshot_reader_controls",profile.run.__code__.co_consts)

    def test_contract_compile_error_is_blocked_not_a_runtime_red(self):
        with tempfile.TemporaryDirectory() as directory, mock.patch("builtins.print"):
            root = Path(directory)
            result = mock.Mock(returncode=101, stdout=json.dumps({"reason": "build-finished", "success": False}))
            with mock.patch.object(fixture, "source_stamp", return_value=self.source()), mock.patch.object(
                    profile.subprocess, "run", return_value=result), mock.patch.object(fixture, "select_binary") as select:
                evidence = profile.contract_fraction_tests(root)
            self.assertFalse(evidence["passed"])
            self.assertEqual(evidence["compiler_exit"], 101)
            self.assertTrue(all(row["status"] == "blocked_compile" for row in evidence["tests"]))
            select.assert_not_called()

    def test_venue_or_source_failure_still_writes_truthful_failure_payload(self):
        for venue_error in (True, False):
            with self.subTest(venue_error=venue_error), tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                with mock.patch.object(profile, "venue", side_effect=UnitFailure("venue refused") if venue_error else None,
                                       return_value={"head": "a" * 40, "backend": "candle"}), mock.patch.object(
                        fixture, "source_stamp", return_value=self.source(),
                        side_effect=None if venue_error else subprocess.CalledProcessError(128, ["git", "diff"])), mock.patch.object(
                        profile, "compile_binary") as compile_:
                    with self.assertRaises((UnitFailure, subprocess.SubprocessError)):
                        profile.run("candle", root)
                evidence = json.loads((root / "proof.json").read_text(encoding="utf-8"))
                self.assertFalse(evidence["passed"])
                self.assertIsNone(evidence["source_restored"])
                self.assertIn("fatal_error", evidence)
                compile_.assert_not_called()

    def test_red_named_baseline_blocks_mutation_without_claiming_a_kill(self):
        source = self.source()
        batch = self.batch(); batch["cases"][1]["status"] = "assertion_failed"; batch["passed"] = False
        with tempfile.TemporaryDirectory() as directory, mock.patch("builtins.print"):
            root = Path(directory)
            binary = root / "binary"; binary.write_bytes(b"source protocol artifact")
            (root / "baseline.batch.json").write_text(json.dumps(batch), encoding="utf-8")
            with mock.patch.object(profile, "venue", return_value={"head": source["head"], "backend": "candle"}), mock.patch.object(
                    fixture, "source_stamp", return_value=source), mock.patch.object(
                    profile, "compile_binary", return_value=(binary, {"source": source})) as compile_, mock.patch.object(
                    profile.subprocess, "run", return_value=mock.Mock(returncode=1)), mock.patch.object(
                    profile, "contract_fraction_tests", return_value={"passed": True}), mock.patch.object(
                    profile, "snapshot_reader_tests", return_value={"passed": True}), mock.patch.object(
                    profile, "independent_tests", return_value=[{"test": profile.PHASE_TEST, "status": "assertion_failed"}]), mock.patch.object(
                    profile, "assertion_red") as red:
                evidence = profile.run("candle", root)
            self.assertFalse(evidence["passed"])
            self.assertEqual(compile_.call_count, 1)
            red.assert_not_called()
            self.assertEqual(evidence["mutations"][0]["status"], "blocked_baseline_red")
            self.assertFalse(evidence["mutations"][0]["runtime_assertion_red"])
            self.assertTrue(evidence["source_restored"])


if __name__ == "__main__":
    unittest.main()
