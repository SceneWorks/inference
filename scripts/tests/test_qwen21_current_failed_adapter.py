import copy
import datetime as dt
import hashlib
import json
import os
from pathlib import Path
import stat
import subprocess
import sys
import tempfile
import unittest
import zipfile

import yaml

from scripts.ci import qwen21_current_failed_adapter as current
from scripts.ci.real_weights_workflow import inline_text


class CurrentFailedAdapterTests(unittest.TestCase):
    def setUp(self):
        self.cfg = current.config()
        self.source_run = {
            "id": 37392084691, "run_attempt": 1,
            "head_sha": self.cfg["source"]["baseCommit"],
            "status": "completed", "conclusion": "failure", "event": "workflow_dispatch",
            "path": self.cfg["workflowPath"],
            "repository": {"full_name": self.cfg["repository"]},
            "head_commit": {"tree_id": self.cfg["source"]["baseTree"]},
        }
        self.source_job = {
            "id": self.cfg["source"]["jobId"], "run_id": self.cfg["source"]["runId"],
            "run_attempt": 1, "head_sha": self.cfg["source"]["baseCommit"],
            "name": self.cfg["source"]["jobName"], "status": "completed",
            "conclusion": "failure", "started_at": "2026-10-06T00:06:00Z",
            "completed_at": "2026-10-06T01:23:28Z",
            "runner_id": self.cfg["source"]["runnerId"],
            "runner_name": self.cfg["source"]["runnerName"],
            "labels": self.cfg["source"]["labels"],
        }

    def test_failed_source_run_job_and_artifact_are_exact(self):
        current.validate_source_run(self.source_run, copy.deepcopy(self.source_run),
                                    self.cfg["source"], self.cfg["repository"],
                                    self.cfg["workflowPath"])
        self.assertEqual(current.validate_source_job({"jobs": [self.source_job]},
                                                     self.cfg["source"])["id"],
                         self.cfg["source"]["jobId"])
        artifact = {"id": self.cfg["artifact"]["id"], "name": self.cfg["artifact"]["name"],
                    "size_in_bytes": self.cfg["artifact"]["bytes"],
                    "digest": "sha256:" + self.cfg["artifact"]["sha256"], "expired": False,
                    "created_at": "2026-10-06T01:23:24Z", "updated_at": "2026-10-06T01:23:24Z",
                    "expires_at": self.cfg["artifact"]["expiresAt"], "workflow_run": {
                        "id": self.cfg["source"]["runId"],
                        "head_sha": self.cfg["source"]["baseCommit"]}}
        current.validate_artifact(artifact, self.cfg, dt.datetime(2026, 10, 6,
                                                                  tzinfo=dt.timezone.utc))
        mutations = []
        for target, key, value in ((self.source_run, "head_sha", "0" * 40),
                                   (self.source_job, "id", 1),
                                   (self.source_job, "conclusion", "success"),
                                   (artifact, "id", 1),
                                   (artifact, "digest", "sha256:" + "0" * 64),
                                   (artifact, "expired", True)):
            mutant = copy.deepcopy(target)
            mutant[key] = value
            mutations.append((target, mutant))
        for original, mutant in mutations:
            with self.subTest(mutant=mutant), self.assertRaises(ValueError):
                if original is self.source_run:
                    current.validate_source_run(mutant, copy.deepcopy(mutant), self.cfg["source"],
                                                self.cfg["repository"], self.cfg["workflowPath"])
                elif original is self.source_job:
                    current.validate_source_job({"jobs": [mutant]}, self.cfg["source"])
                else:
                    current.validate_artifact(mutant, self.cfg,
                                              dt.datetime(2026, 10, 6, tzinfo=dt.timezone.utc))

    def live_fixture(self):
        context = {"repository": self.cfg["repository"], "jobKey": self.cfg["workflowJobKey"],
                   "sha": "a" * 40, "runId": "999", "runAttempt": "2"}
        run = {"id": 999, "run_attempt": 2, "head_sha": "a" * 40,
               "status": "in_progress", "conclusion": None, "event": "workflow_dispatch",
               "path": self.cfg["workflowPath"],
               "repository": {"full_name": self.cfg["repository"]}}
        host = self.cfg["liveHost"]
        job = {"id": 888, "run_id": 999, "run_attempt": 2, "head_sha": "a" * 40,
               "name": self.cfg["workflowJobName"], "status": "in_progress",
               "conclusion": None, "runner_id": host["runnerId"],
               "runner_name": host["runnerName"], "labels": host["labels"],
               "started_at": "2026-10-06T02:01:00Z", "completed_at": None}
        return context, run, job

    def test_live_execution_is_unique_in_progress_and_separate_from_failed_job(self):
        context, run, job = self.live_fixture()
        self.assertEqual(current.validate_live(run, {"jobs": [job]}, self.cfg, context)["id"], 888)
        mutations = []
        for key, value in (("runner_id", 1), ("runner_name", "nax-macos"),
                           ("status", "completed"), ("started_at", None),
                           ("id", self.cfg["source"]["jobId"])):
            mutant = copy.deepcopy(job); mutant[key] = value; mutations.append((context, run, [mutant]))
        mutations.append((context, run, [job, copy.deepcopy(job)]))
        historical = copy.deepcopy(context); historical["runId"] = str(self.cfg["source"]["runId"])
        historical_run = copy.deepcopy(run); historical_run["id"] = self.cfg["source"]["runId"]
        mutations.append((historical, historical_run, [job]))
        for candidate_context, candidate_run, jobs in mutations:
            with self.subTest(jobs=jobs), self.assertRaises(ValueError):
                current.validate_live(candidate_run, {"jobs": jobs}, self.cfg, candidate_context)

    def test_shallow_checkout_acquires_exact_base_without_moving_head_or_ref(self):
        def run(root, *args, check=True):
            return subprocess.run(["git", *args], cwd=root, check=check, text=True,
                                  encoding="utf-8",
                                  stdout=subprocess.PIPE, stderr=subprocess.PIPE)

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory).resolve()
            origin = root / "origin"
            run(root, "init", "-b", "main", str(origin))
            run(origin, "config", "user.name", "fixture")
            run(origin, "config", "user.email", "fixture@example.invalid")
            (origin / "base.txt").write_text("base\n", encoding="utf-8")
            run(origin, "add", "base.txt"); run(origin, "commit", "-m", "base")
            base = run(origin, "rev-parse", "HEAD").stdout.strip()
            base_tree = run(origin, "rev-parse", "HEAD^{tree}").stdout.strip()
            (origin / "head.txt").write_text("head\n", encoding="utf-8")
            run(origin, "add", "head.txt"); run(origin, "commit", "-m", "head")
            checkout = root / "checkout"
            run(root, "clone", "--depth=1", "--branch", "main", origin.as_uri(), str(checkout))
            head = run(checkout, "rev-parse", "HEAD").stdout.strip()
            ref = run(checkout, "symbolic-ref", "-q", "HEAD").stdout.strip()
            self.assertNotEqual(run(checkout, "cat-file", "-e", base + "^{commit}",
                                    check=False).returncode, 0)

            cfg = {"source": {"baseCommit": base, "baseTree": base_tree}}
            current.acquire_source_base(checkout, head, cfg)
            self.assertEqual(run(checkout, "rev-parse", "HEAD").stdout.strip(), head)
            self.assertEqual(run(checkout, "symbolic-ref", "-q", "HEAD").stdout.strip(), ref)
            self.assertEqual(run(checkout, "rev-parse", base + "^{tree}").stdout.strip(),
                             base_tree)

            mutant = copy.deepcopy(cfg)
            mutant["source"]["baseTree"] = "0" * 40
            with self.assertRaises(ValueError):
                current.acquire_source_base(checkout, head, mutant)

    def test_zip_checks_complete_digest_and_extracts_only_fixed_members(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            archive_path = root / "artifact.zip"
            payloads = {"adapters/a.bin": b"adapter", "receipt.json": b"receipt",
                        "support/unselected.png": b"not extracted"}
            with zipfile.ZipFile(archive_path, "w", zipfile.ZIP_DEFLATED) as archive:
                for name, payload in payloads.items():
                    archive.writestr(name, payload)
            cfg = {"artifact": {"bytes": archive_path.stat().st_size,
                                "sha256": current.sha256_file(archive_path)},
                   "members": [{"archivePath": "adapters/a.bin", "file": "a.bin",
                                "bytes": 7, "sha256": hashlib.sha256(b"adapter").hexdigest()}],
                   "validateOnlyMembers": [{"archivePath": "receipt.json", "bytes": 7,
                                            "sha256": hashlib.sha256(b"receipt").hexdigest()}]}
            stage = root / "stage"
            self.assertEqual(current.extract_selected(archive_path, stage, cfg)[0]["file"], "a.bin")
            self.assertEqual(sorted(path.name for path in stage.iterdir()), ["a.bin"])
            for mutate in (lambda row: row["artifact"].update({"sha256": "0" * 64}),
                           lambda row: row["members"][0].update({"sha256": "0" * 64}),
                           lambda row: row["members"][0].update({"bytes": 8})):
                mutant = copy.deepcopy(cfg); mutate(mutant)
                with self.subTest(mutant=mutant), self.assertRaises(ValueError):
                    current.extract_selected(archive_path, root / ("stage-" + os.urandom(3).hex()), mutant)

    def test_zip_rejects_traversal_duplicates_and_symlinks(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            cases = []
            traversal = root / "traversal.zip"
            with zipfile.ZipFile(traversal, "w") as archive:
                archive.writestr("../escape", b"x")
            cases.append(traversal)
            duplicate = root / "duplicate.zip"
            with zipfile.ZipFile(duplicate, "w") as archive:
                archive.writestr("A", b"x"); archive.writestr("a", b"y")
            cases.append(duplicate)
            symlink = root / "symlink.zip"
            with zipfile.ZipFile(symlink, "w") as archive:
                info = zipfile.ZipInfo("link")
                info.create_system = 3
                info.external_attr = (stat.S_IFLNK | 0o777) << 16
                archive.writestr(info, b"target")
            cases.append(symlink)
            for path in cases:
                with self.subTest(path=path.name):
                    with zipfile.ZipFile(path) as archive, self.assertRaises(ValueError):
                        current.safe_archive_members(archive)

    def test_manifest_matches_rust_handoff_schema_and_receipt_never_accepts(self):
        stage = Path(tempfile.gettempdir()).resolve() / "current-q4-stage"
        manifest = current.build_manifest(stage, self.cfg)
        self.assertEqual(manifest["directory"], str(stage))
        self.assertEqual(manifest["trainingProvenance"]["jobId"], 112039296411)
        self.assertEqual([row["file"] for row in manifest["adapters"]],
                         ["qwen21_edit_lokr.safetensors"])
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory).resolve()
            current.write_receipt(output, {"status": "PREPARING_INPUTS"})
            receipt = json.loads(current.receipt_path(output).read_text(encoding="utf-8"))
            self.assertEqual((receipt["trainingSteps"], receipt["renderCount"],
                              receipt["replayCount"]), (0, 0, 0))
            self.assertIs(receipt["accepted"], False)
            current.finish(type("Args", (), {"output": output, "selector_exit": 101})())
            self.assertEqual(json.loads(current.receipt_path(output).read_text(
                encoding="utf-8"))["status"],
                             "DIAGNOSTIC_FAILED")

    def test_cli_rejects_empty_and_relative_output_before_writing(self):
        repository = Path(__file__).resolve().parents[2]
        environment = os.environ.copy()
        environment["PYTHONPATH"] = str(repository)
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory).resolve()
            for raw in ("", "relative"):
                result = subprocess.run([
                    sys.executable, "-m", "scripts.ci.qwen21_current_failed_adapter",
                    "seal", "--output", raw,
                ], cwd=root, env=environment, text=True, encoding="utf-8",
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE)
                self.assertEqual(result.returncode, 1)
                self.assertIn("absolute evidence path required", result.stderr)
            self.assertFalse((root / "current-diagnostic").exists())

    def test_build_identity_and_selector_output_mutations_are_rejected(self):
        build = {"lockedMlxRsRevision": self.cfg["mlxBuild"]["mlxRsRevision"],
                 "expectedCoreTag": self.cfg["mlxBuild"]["coreTag"],
                 "actualStagedCoreTag": self.cfg["mlxBuild"]["coreTag"],
                 "buildManifest": {"fingerprint": self.cfg["mlxBuild"]["fingerprint"]},
                 "linkMode": "source_with_staged_tag",
                 "libTestExecutable": {"sha256": "a" * 64, "bytes": 1}}
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory).resolve()
            build_path = root / "build.json"
            build_path.write_text(json.dumps(build), encoding="utf-8")
            current.validate_build(build_path, self.cfg)
            for key, value in (("lockedMlxRsRevision", "0" * 40),
                               ("actualStagedCoreTag", "v0.31.1"),
                               ("linkMode", "prebuilt_verified_by_mlx_sys_source_fingerprint")):
                mutant = copy.deepcopy(build); mutant[key] = value
                build_path.write_text(json.dumps(mutant), encoding="utf-8")
                with self.subTest(key=key), self.assertRaises(ValueError):
                    current.validate_build(build_path, self.cfg)

            output = root / "success"
            selector = output / "current-q4-velocity-discriminator" / "receipt.json"
            selector.parent.mkdir(parents=True)
            good = {"kind": "DIAGNOSTIC_ONLY", "accepted": False,
                    "acceptanceEvidence": False, "trainingSteps": 0, "renderCount": 0}
            selector.write_text(json.dumps(good), encoding="utf-8")
            current.finish(type("Args", (), {"output": output, "selector_exit": 0})())
            self.assertEqual(current.read_json(current.receipt_path(output))["status"],
                             "DIAGNOSTIC_COMPLETED")
            for key, value in (("accepted", True), ("acceptanceEvidence", True),
                               ("trainingSteps", 1), ("renderCount", 1)):
                bad_output = root / ("bad-" + key)
                bad_selector = bad_output / "current-q4-velocity-discriminator" / "receipt.json"
                bad_selector.parent.mkdir(parents=True)
                mutant = copy.deepcopy(good); mutant[key] = value
                bad_selector.write_text(json.dumps(mutant), encoding="utf-8")
                with self.subTest(key=key), self.assertRaises(ValueError):
                    current.finish(type("Args", (), {"output": bad_output,
                                                      "selector_exit": 0})())

            unstarted = root / "unstarted"
            current.write_receipt(unstarted, {"status": "PREPARING_INPUTS"})
            current.seal(type("Args", (), {"output": unstarted})())
            self.assertEqual(current.read_json(current.receipt_path(unstarted))["refusal"],
                             "selector_not_started")

    @unittest.skipUnless(os.environ.get("QWEN21_CURRENT_ARTIFACT_ZIP"),
                         "set QWEN21_CURRENT_ARTIFACT_ZIP for the frozen 503,983,926-byte packet")
    def test_actual_frozen_artifact_packet(self):
        source = Path(os.environ["QWEN21_CURRENT_ARTIFACT_ZIP"])
        with tempfile.TemporaryDirectory() as directory:
            identities = current.extract_selected(source, Path(directory).resolve() / "stage", self.cfg)
            self.assertEqual(len(identities), 3)

    @unittest.skipUnless(os.environ.get("QWEN21_CURRENT_PACKET_ROOT"),
                         "set QWEN21_CURRENT_PACKET_ROOT for the frozen terminal packet")
    def test_actual_frozen_api_and_job_log_packet(self):
        root = Path(os.environ["QWEN21_CURRENT_PACKET_ROOT"])
        poll = root / "raw-final-poll" / "20261006T012526314Z"
        terminal = root / "terminal"
        run = current.read_json(poll / "run.json")
        attempt = current.read_json(poll / "attempt-1.json")
        current.validate_source_run(run, attempt, self.cfg["source"], self.cfg["repository"],
                                    self.cfg["workflowPath"])
        job = current.validate_source_job(current.read_json(poll / "jobs.json"),
                                          self.cfg["source"])
        self.assertEqual(job["runner_id"], 5281)
        artifact = current.read_json(terminal / "artifacts" / "11383988900-metadata.json")["artifact"]
        current.validate_artifact(artifact, self.cfg,
                                  dt.datetime(2026, 10, 6, tzinfo=dt.timezone.utc))
        current.validate_job_log(terminal / "logs" / "job-112039296411.log",
                                 self.cfg["source"], self.cfg["artifact"])


class CurrentDiagnosticWorkflowTests(unittest.TestCase):
    @staticmethod
    def embedded_python(command):
        return command.split("<<'PY'\n", 1)[1].rsplit("\nPY", 1)[0]

    def test_current_phase_is_opt_in_bounded_and_failure_preserving(self):
        workflow = yaml.safe_load(inline_text())
        inputs = workflow[True]["workflow_dispatch"]["inputs"]
        phase = inputs["qwen_image_2_1_lora_phase"]
        self.assertEqual(phase["default"], "full")
        self.assertIn("current-diagnostic", phase["options"])
        job = workflow["jobs"]["mlx-qwen-image-2-1"]
        self.assertEqual(job["timeout-minutes"], 300)
        self.assertEqual(job["permissions"], {"actions": "read", "contents": "read"})
        self.assertEqual(job["runs-on"], ["self-hosted", "macOS", "ARM64",
                                          "${{ inputs.qwen_image_2_1_lora_runner || 'rw-mage' }}"])
        self.assertNotIn("QWEN_IMAGE_2_1_RENDER_OUT", job["env"])
        self.assertEqual(workflow["concurrency"]["cancel-in-progress"], False)
        self.assertIn("inference-real-weights-physical-host", workflow["concurrency"]["group"])
        steps = {row.get("name"): row for row in job["steps"]}
        self.assertEqual(steps["Prove trained velocity survives adapter save and reload"]["if"],
                         "inputs.qwen_image_2_1_lora_phase != 'current-diagnostic'")
        self.assertEqual(steps["Bind live job and materialize the exact current failed adapter"]["if"],
                         "inputs.qwen_image_2_1_lora_phase == 'current-diagnostic'")
        self.assertEqual(steps["Keep the Qwen-Image 2.1 MLX evidence"]["if"],
                         "${{ always() && (inputs.qwen_image_2_1_lora_phase == "
                         "'current-diagnostic' || !cancelled()) }}")
        names = [row.get("name") for row in job["steps"]]
        fallback = steps["Initialize current diagnostic fallback before checkout"]
        self.assertLess(names.index(fallback["name"]),
                        next(index for index, row in enumerate(job["steps"])
                             if str(row.get("uses", "")).startswith("actions/checkout@")))
        self.assertNotIn("scripts/", fallback["run"])
        self.assertIn('$RUNNER_TEMP/qwen-image-2-1-mlx-evidence', fallback["run"])
        self.assertIn('QWEN_IMAGE_2_1_RENDER_OUT=$output', fallback["run"])
        seal = steps["Seal a current diagnostic refusal if the selector never started"]
        self.assertNotIn("scripts/", seal["run"])
        self.assertIn('$RUNNER_TEMP/qwen-image-2-1-mlx-evidence', seal["run"])
        run = steps["Run the Qwen-Image 2.1 LoRA/LoKr real-weight gates"]["run"]
        self.assertEqual(run.count(current.config()["selector"]), 1)
        self.assertIn('if [[ "$phase" == current-diagnostic ]]', run)
        self.assertNotIn("current-diagnostic ||", run)

    def test_checkout_independent_fallback_initializes_and_seals_absolute_output(self):
        workflow = yaml.safe_load(inline_text())
        steps = {row.get("name"): row for row in workflow["jobs"]["mlx-qwen-image-2-1"]["steps"]}
        initialize = self.embedded_python(
            steps["Initialize current diagnostic fallback before checkout"]["run"])
        seal = self.embedded_python(
            steps["Seal a current diagnostic refusal if the selector never started"]["run"])
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory).resolve() / "evidence"
            started = subprocess.run([sys.executable, "-", str(root)], input=initialize,
                                     text=True, encoding="utf-8", stdout=subprocess.PIPE,
                                     stderr=subprocess.PIPE)
            self.assertEqual(started.returncode, 0, started.stderr)
            receipt = root / "current-diagnostic" / "DIAGNOSTIC_ONLY.json"
            self.assertEqual(json.loads(receipt.read_text(encoding="utf-8"))["status"],
                             "EARLY_INITIALIZED")
            finished = subprocess.run([sys.executable, "-", str(root)], input=seal,
                                      text=True, encoding="utf-8", stdout=subprocess.PIPE,
                                      stderr=subprocess.PIPE)
            self.assertEqual(finished.returncode, 0, finished.stderr)
            row = json.loads(receipt.read_text(encoding="utf-8"))
            self.assertEqual((row["status"], row["refusal"]),
                             ("REFUSED", "selector_not_started"))

            relative = subprocess.run([sys.executable, "-", "relative"], input=seal,
                                      cwd=Path(directory), text=True, encoding="utf-8",
                                      stdout=subprocess.PIPE,
                                      stderr=subprocess.PIPE)
            self.assertNotEqual(relative.returncode, 0)
            self.assertFalse((Path(directory) / "relative").exists())


if __name__ == "__main__":
    unittest.main()
