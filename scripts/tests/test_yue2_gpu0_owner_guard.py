"""CPU mocks for the opt-in physical-owner guard; no GPU or GitHub calls."""
import base64
import copy
from datetime import datetime, timezone
import hashlib
import io
import json
import os
import re
from pathlib import Path
import subprocess
import sys
import tempfile
import threading
import time
import unittest
from unittest.mock import Mock, patch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts/ci"))
import yue2_gpu0_owner_guard as guard
from scripts.tests import test_yue2_precision_proof as routes


def run(run_id=guard.HOLDER_RUN, sha=guard.HOLDER_SHA, path=".github/workflows/real-weights.yml"):
    return {"id": run_id, "head_sha": sha, "run_attempt": 1, "event": "workflow_dispatch",
            "path": path, "repository": {"full_name": guard.REPO}, "status": "in_progress", "conclusion": None}


def jobs(job_id=guard.HOLDER_JOB, name=guard.HOLDER_NAME, runner=guard.HOLDER_RUNNER):
    selected = {"id": job_id, "name": name, "runner_name": runner, "status": "in_progress",
                "conclusion": None, "completed_at": None, "started_at": "2026-10-03T10:00:00Z"}
    return {"total_count": 55, "jobs": [selected] +
            [{"id": i, "status": "completed", "conclusion": "skipped"} for i in range(54)]}


def group(name=guard.OLD_GROUP, run_id=guard.HOLDER_RUN):
    return {"group_name": name, "total_count": 2,
            "group_members": [{"run_id": run_id, "status": "in_progress"},
                              {"run_id": 123, "status": "pending"}]}


def background():
    prefix = "pid_38212_luid_0x00000000_0x00020d46_"
    return {"identity": [38212, "signed-app.exe", "C:/signed-app.exe", "09/25/2026 07:02:03"],
            "signature": {"status": "Valid", "signerSubject": "reviewed", "signerThumbprint": "reviewed"},
            "luid": "luid_0x00000000_0x00020d46",
            "counters": {"engine": {prefix + f"phys_0_eng_{i}_engtype_compute": 0 for i in range(22)}}}


def background_sample():
    baseline = background()
    paths = [rf"\GPU Engine({key})\Utilization Percentage" for key in baseline["counters"]["engine"]]
    return {"process": dict(zip(("pid", "name", "executablePath", "creationDate"), baseline["identity"]),
                            signature=baseline["signature"]),
            "counter": {"counter": r"\GPU Engine(*)\Utilization Percentage",
                        "queryScope": "receipt-pid-gpu0", "enumeratedPaths": paths, "queryPaths": paths,
                        "samples": [{"instance": key, "cookedValue": 0, "status": "0"}
                                    for key in baseline["counters"]["engine"]]}}


class Owned:
    pid = 777
    code = None
    def poll(self): return self.code
    def wait(self, timeout):
        if self.code is not None: return self.code
        raise subprocess.TimeoutExpired("owned", timeout)


class OwnerGuardTests(unittest.TestCase):
    def test_routing_is_one_fixed_group_only_exact_cuda_owner_mode(self):
        for filename, selector in (("yue2-precision-proof.yml", "stage"),
                                   ("yue2-app-precision-profile.yml", "backend")):
            source = (ROOT / ".github/workflows" / filename).read_text(encoding="utf-8")
            settings = routes.PrecisionControlTests.concurrency_settings(source)
            self.assertEqual(settings["queue"], "max")
            self.assertEqual(settings["cancel-in-progress"], "false")
            for stage in ("fixture", "cuda", "metal", "cuda-diagnostic", "", "unknown"):
                for mode in ("shared-host", "owner-gpu0", "owner-gpu0-mac-anchor", "", "unknown"):
                    for receipt in (guard.RECEIPT, "37122359802", "37106146499", "", "arbitrary"):
                        for engine in (guard.ENGINE, "a" * 40):
                            opted = stage == "cuda" and mode in {"owner-gpu0", "owner-gpu0-mac-anchor"} and receipt == guard.RECEIPT and engine == guard.ENGINE
                            actual = routes.PrecisionControlTests.concurrency_group(settings["group"], stage, "101", mode, receipt, engine)
                            if opted:
                                expected = guard.GPU0_GROUP
                            elif selector == "stage" and stage == "fixture":
                                expected = "inference-yue2-precision-fixture-101"
                            elif (selector == "backend" and stage != "cuda") or stage == "metal":
                                expected = "yue2-app-precision-nax-macos-2"
                            else:
                                expected = guard.OLD_GROUP
                            self.assertEqual(actual, expected, (filename, stage, mode, receipt, engine))
                            if opted:
                                self.assertEqual(actual, routes.PrecisionControlTests.concurrency_group(settings["group"], stage, "102", mode, receipt, engine))
            cuda = source.split("  cuda:\n", 1)[1].split("\n  metal:", 1)[0]
            self.assertIn("      YUE2_CUDA_SCHEDULING_MODE: ${{ inputs.cuda_scheduling_mode }}", cuda)
            self.assertIn("GH_TOKEN: ${{ github.token }}", cuda)
            self.assertIn("default: shared-host", source)

    def test_exact_run_identity_mutants_refuse(self):
        guard.run_identity(run(), guard.HOLDER_RUN, guard.HOLDER_SHA, ".github/workflows/real-weights.yml")
        changes = {"id": 1, "head_sha": "b" * 40, "run_attempt": 2, "event": "push",
                   "path": "other.yml", "repository": {"full_name": "other/repo"},
                   "status": "completed", "conclusion": "success"}
        for key, value in changes.items():
            with self.subTest(key=key), self.assertRaises(RuntimeError):
                guard.run_identity({**run(), key: value}, guard.HOLDER_RUN, guard.HOLDER_SHA, ".github/workflows/real-weights.yml")

    def test_exact_sole_job_runner_attempt_inventory_mutants_refuse(self):
        guard.selected_job(jobs(), guard.HOLDER_JOB, guard.HOLDER_NAME, guard.HOLDER_RUNNER)
        for key, value in {"id": 1, "name": "other", "runner_name": "cuda-windows", "status": "queued",
                           "conclusion": "success", "completed_at": "2026-10-03T12:00:00Z", "started_at": None}.items():
            data = jobs(); data["jobs"][0][key] = value
            with self.subTest(key=key), self.assertRaises(RuntimeError):
                guard.selected_job(data, guard.HOLDER_JOB, guard.HOLDER_NAME, guard.HOLDER_RUNNER)
        for mutate in (lambda x: x.update(total_count=56),
                       lambda x: x["jobs"][1].update(conclusion=None),
                       lambda x: x["jobs"][1].update(status="queued")):
            data = jobs(); mutate(data)
            with self.assertRaises(RuntimeError):
                guard.selected_job(data, guard.HOLDER_JOB, guard.HOLDER_NAME, guard.HOLDER_RUNNER)

    def test_old_group_membership_mutants_refuse(self):
        guard.active_group(group(), guard.OLD_GROUP, guard.HOLDER_RUN)
        for mutate in (lambda x: x.update(group_name="new"), lambda x: x.update(total_count=1),
                       lambda x: x["group_members"][0].update(run_id=1),
                       lambda x: x["group_members"][0].update(status="pending"),
                       lambda x: x["group_members"][0].update(job_id=1),
                       lambda x: x["group_members"][1].update(status="in_progress")):
            data = group(); mutate(data)
            with self.assertRaises(RuntimeError): guard.active_group(data, guard.OLD_GROUP, guard.HOLDER_RUN)

    def test_typed_gpu0_only_owned_root_desktop_or_owned_descendants(self):
        header = "# gpu pid type sm mem enc dec jpg ofa command\n"
        good = header + "0 38212 C+G 0 0 0 0 0 0 desktop\n0 777 C 1 0 0 0 0 0 owned\n"
        with patch.object(guard.subprocess, "run", return_value=Mock(returncode=0, stdout=good)), patch.object(guard, "sample_background"):
            self.assertEqual(guard.gpu0_actors(777, background=background()), good)
            with self.assertRaises(RuntimeError): guard.gpu0_actors(None, background=background())
        for raw in (good.replace("777", "888"), good.replace(" C 1", " G 1"), good.replace("C+G", "G"),
                    good.replace("0 777", "1 777"), "", "0 777 C 1 0 0 0 owned\n", good + "0 777 C 1 0 0 0 0 0 owned\n"):
            with patch.object(guard.subprocess, "run", return_value=Mock(returncode=0, stdout=raw)), patch.object(guard, "sample_background"), self.assertRaises(RuntimeError):
                guard.gpu0_actors(777, background=background())
        with patch.object(guard.subprocess, "run", return_value=Mock(returncode=0, stdout=good.replace("777", "999"))), \
             patch.object(guard, "owned_descendants", return_value={777, 888, 999}), patch.object(guard, "sample_background"):
            guard.gpu0_actors(777, True, background=background())

    def test_background_supported_activity_and_unsupported_rows_never_bypass_windows_identity(self):
        good = "# gpu pid type sm mem enc dec jpg ofa command\n0 38212 C+G - - - - - - desktop\n0 777 C 80 50 0 0 0 0 owned\n"
        with patch.object(guard.subprocess, "run", return_value=Mock(returncode=0, stdout=good)), \
             patch.object(guard, "sample_background") as probe:
            guard.gpu0_actors(777, background=background())
            probe.assert_called_once()
        missing = "# gpu pid type sm mem enc dec jpg ofa command\n0 777 C 80 50 0 0 0 0 owned\n"
        with patch.object(guard.subprocess, "run", return_value=Mock(returncode=0, stdout=missing)), self.assertRaisesRegex(RuntimeError, "disappeared"):
            guard.gpu0_actors(777, background=background())
        for raw in (good.replace("C+G - -", "C+G 80 50"), good.replace("C+G - -", "C+G 0 50"),
                    good.replace("C+G - -", "C+G NaN 0"), good.replace(" jpg ofa", "")):
            with patch.object(guard.subprocess, "run", return_value=Mock(returncode=0, stdout=raw)), \
                 patch.object(guard, "sample_background") as probe, self.assertRaises(RuntimeError):
                guard.gpu0_actors(777, background=background())
            probe.assert_not_called()
        with patch.object(guard.subprocess, "run", return_value=Mock(returncode=0, stdout=good)), \
             patch.object(guard, "sample_background", side_effect=RuntimeError("live counter positive")), self.assertRaises(RuntimeError):
            guard.gpu0_actors(777, background=background())

    def test_live_background_counter_and_process_generation_mutations_refuse(self):
        baseline, value = background(), background_sample()
        guard.validate_background(value, baseline)
        mutations = [lambda x: x["process"].update(pid=1), lambda x: x["process"].update(name="different"),
                     lambda x: x["process"].update(creationDate="new start"),
                     lambda x: x["process"].update(executablePath="different"),
                     lambda x: x["process"]["signature"].update(status="NotSigned"),
                     lambda x: x["process"]["signature"].update(signerThumbprint="new"),
                     lambda x: x["counter"]["samples"].pop(),
                     lambda x: x["counter"]["samples"].append({"instance": "pid_38212_luid_0x00000000_0x00020d46_extra", "cookedValue": 0, "status": "0"}),
                     lambda x: x["counter"]["samples"].append(x["counter"]["samples"][0]),
                     lambda x: x["counter"].update(error="unsupported")]
        for number in (1, float("nan"), float("inf"), -1, True, None):
            mutations.append(lambda x, n=number: x["counter"]["samples"][0].update(cookedValue=n))
        mutations.extend([lambda x: x["counter"]["samples"][0].update(status="1"),
                          lambda x: x["counter"]["samples"][0].update(instance=x["counter"]["samples"][0]["instance"].replace("00020d46", "0001f8b5"))])
        for index, mutate in enumerate(mutations):
            changed = copy.deepcopy(value); mutate(changed)
            with self.subTest(index=index), self.assertRaises(RuntimeError):
                guard.validate_background(changed, baseline)
        # An unrelated LUID is not the physical GPU0 activity signal.
        foreign = copy.deepcopy(value)
        foreign["counter"]["samples"].append({"instance": "pid_38212_luid_0x00000000_0x0001f8b5_other", "cookedValue": 80, "status": "0"})
        guard.validate_background(foreign, baseline)

    def test_bounded_background_probe_uses_collector_schema_and_retains_failure_raw(self):
        with patch.object(guard.subprocess, "run", return_value=Mock(returncode=0, stdout=json.dumps(background_sample()), stderr="")) as probe:
            guard.sample_background(background())
            self.assertEqual(probe.call_args.kwargs["timeout"], guard.PHYSICAL_QUERY_TIMEOUT)
            script = probe.call_args.args[0][-1]
            self.assertIn("Get-Counter -ListSet 'GPU Engine' -ErrorAction Stop", script)
            self.assertIn("Get-Counter -Counter $probe.counter.queryPaths", script)
            self.assertNotIn("Get-Counter -Counter '\\GPU Engine(*)", script)
            self.assertLess(script.index("$selected.Count -ne"), script.index("Get-Counter -Counter"))
            pattern = script.split("$path -match '")[1].split("'")[0]
            for path in background_sample()["counter"]["queryPaths"]:
                self.assertIsNotNone(re.fullmatch(pattern, path, re.IGNORECASE))
            config = json.loads(base64.b64decode(script.split("FromBase64String('")[1].split("')")[0]))
            self.assertEqual(config, {"pid": 38212, "luid": background()["luid"],
                                      "instances": sorted(background()["counters"]["engine"])})
        with patch.object(guard.subprocess, "run", return_value=Mock(returncode=1, stdout="", stderr="missing counters")), self.assertRaises(RuntimeError):
            guard.sample_background(background())

    def test_current_catalog_exact_target_set_before_sampling(self):
        baseline = background()
        paths = background_sample()["counter"]["queryPaths"]
        other_pid = paths[0].replace("pid_38212_", "pid_777_")
        other_luid = paths[0].replace("00020d46", "0001f8b5")
        self.assertEqual(guard.background_query_paths(paths + [other_pid, other_luid], baseline),
                         sorted(paths, key=str.lower))
        variants = [paths[:-1], paths + [paths[0]],
                    paths + [paths[0].replace("eng_0_", "eng_99_")],
                    [path.replace("00020d46", "0001f8b5") for path in paths], [], None]
        for value in variants:
            with self.subTest(value=value), self.assertRaises(RuntimeError):
                guard.background_query_paths(value, baseline)

    def test_scoped_query_provenance_and_returned_samples_fail_closed(self):
        mutations = [lambda x: x["counter"]["enumeratedPaths"].pop(),
                     lambda x: x["counter"]["enumeratedPaths"].append(x["counter"]["enumeratedPaths"][0].replace("eng_0_", "eng_99_")),
                     lambda x: x["counter"]["queryPaths"].pop(),
                     lambda x: x["counter"].update(queryScope="global"),
                     lambda x: x["counter"]["samples"].append({"instance": "pid_777_other", "cookedValue": 0, "status": "0"}),
                     lambda x: x["counter"]["samples"][0].update(cookedValue=1),
                     lambda x: x["counter"]["samples"][0].update(status="1"),
                     lambda x: x["counter"]["samples"][0].update(cookedValue=float("nan"))]
        for index, mutate in enumerate(mutations):
            value = background_sample(); mutate(value)
            with self.subTest(index=index), patch.object(guard.subprocess, "run", return_value=Mock(returncode=0, stdout=json.dumps(value), stderr="")), self.assertRaises(RuntimeError):
                guard.sample_background(background())
        failed = background_sample()
        failed["counter"].update(error="CounterApiError", samples=[])
        records = []
        with patch.object(guard.subprocess, "run", return_value=Mock(returncode=1, stdout=json.dumps(failed), stderr="invalid selected counter")), self.assertRaises(RuntimeError):
            guard.sample_background(background(), records.append)
        self.assertEqual(json.loads(records[0]["raw"])["counter"]["enumeratedPaths"], failed["counter"]["enumeratedPaths"])
        self.assertEqual(records[0]["stderr"], "invalid selected counter")

    def test_cold_physical_queries_have_independent_bounded_budget(self):
        self.assertEqual(guard.API_TIMEOUT, 3)
        self.assertEqual(guard.PHYSICAL_QUERY_TIMEOUT, 15)
        self.assertEqual(guard.CYCLE_LIMIT_SECONDS, 55)
        def cold_query(argv, **kwargs):
            # A four-second cold start exceeded the original API-derived bound.
            if kwargs["timeout"] < 4:
                raise subprocess.TimeoutExpired(argv, kwargs["timeout"])
            self.assertEqual(kwargs["timeout"], 15)
            if "Get-AuthenticodeSignature" in argv[-1]:
                self.assertIn("-SampleInterval 1 -MaxSamples 1", argv[-1])
                output = background_sample()
            else:
                output = [{"ProcessId": 777, "ParentProcessId": 10,
                           "CreatedUtc": "2026-10-03T10:00:00Z"}]
            return Mock(returncode=0, stdout=json.dumps(output), stderr="")
        with patch.object(guard.subprocess, "run", side_effect=cold_query):
            guard.sample_background(background())
            self.assertEqual(guard.owned_descendants(777), {777})

    def test_physical_query_timeout_before_launch_never_creates_child(self):
        with tempfile.TemporaryDirectory() as directory:
            evidence = Path(directory)
            owner = guard.OwnerGuard(evidence, guard.ENGINE, "a" * 40, "app")
            with patch.dict(os.environ, {"EXPECTED_ENGINE_SHA": guard.ENGINE, "EXPECTED_CONTROL_SHA": "a" * 40}), \
                 patch.object(guard, "OwnerGuard", return_value=owner), \
                 patch.object(owner, "_preflight", side_effect=lambda: guard.sample_background(background())), \
                 patch.object(guard.subprocess, "run", side_effect=subprocess.TimeoutExpired("powershell", 15)), \
                 patch.object(guard.subprocess, "Popen") as launch:
                with self.assertRaises(subprocess.TimeoutExpired):
                    guard.guarded_command(["node", "unchanged-case"], evidence, {}, None, evidence, "case")
            launch.assert_not_called()
            self.assertIn("preflight_refusal", owner.path.read_text(encoding="utf-8"))

    def test_physical_query_timeout_during_child_refuses_and_reaps_own_tree(self):
        child = Owned()
        def query_or_reap(argv, **kwargs):
            if argv[0] == "powershell":
                self.assertEqual(kwargs["timeout"], 15)
                raise subprocess.TimeoutExpired(argv, 15)
            self.assertEqual(argv, ["taskkill", "/PID", "777", "/T", "/F"])
            child.code = -9
            return Mock(returncode=0)
        with tempfile.TemporaryDirectory() as directory:
            owner = guard.OwnerGuard(Path(directory), guard.ENGINE, "a" * 40)
            with patch.object(owner, "holder", side_effect=lambda *_: guard.sample_background(background())), \
                 patch.object(guard.subprocess, "run", side_effect=query_or_reap) as calls:
                owner.start(child)
                self.assertTrue(owner.failed.wait(1))
                code, timed_out, error = guard.wait(child, owner, 100)
                self.assertEqual(code, -9)
                self.assertFalse(timed_out)
                self.assertIn("15 seconds", error)
                with self.assertRaises(RuntimeError): owner.finish()
                self.assertEqual(sum(call.args[0][0] == "taskkill" for call in calls.call_args_list), 1)
            self.assertFalse(owner.thread.is_alive())
            self.assertFalse(owner.signals)

    def test_descendants_are_parent_chain_only_not_names(self):
        rows = [{"ProcessId": 777, "ParentProcessId": 10}, {"ProcessId": 888, "ParentProcessId": 777},
                {"ProcessId": 999, "ParentProcessId": 888}, {"ProcessId": 123, "ParentProcessId": 10}]
        for index, row in enumerate(rows):
            row["CreatedUtc"] = f"2026-10-03T10:00:0{index}Z"
        rows.append({"ProcessId": 0, "ParentProcessId": 0, "CreatedUtc": None})
        with patch.object(guard.subprocess, "run", return_value=Mock(returncode=0, stdout=json.dumps(rows))):
            self.assertEqual(guard.owned_descendants(777), {777, 888, 999})
            with self.assertRaises(RuntimeError): guard.owned_descendants(1000)
        old = copy.deepcopy(rows); old[1]["CreatedUtc"] = "2026-10-03T09:00:00Z"
        with patch.object(guard.subprocess, "run", return_value=Mock(returncode=0, stdout=json.dumps(old))):
            self.assertEqual(guard.owned_descendants(777), {777})
        missing = copy.deepcopy(rows); missing[1]["CreatedUtc"] = None
        with patch.object(guard.subprocess, "run", return_value=Mock(returncode=0, stdout=json.dumps(missing))), self.assertRaises(RuntimeError):
            guard.owned_descendants(777)
        for raw in ("null", "{}", json.dumps(rows + [rows[0]])):
            with patch.object(guard.subprocess, "run", return_value=Mock(returncode=0, stdout=raw)), self.assertRaises(RuntimeError):
                guard.owned_descendants(777)

    def test_holder_fault_and_network_failure_prevent_model_spawn(self):
        with tempfile.TemporaryDirectory() as directory:
            owner = guard.OwnerGuard(Path(directory), guard.ENGINE, "a" * 40)
            job = jobs()["jobs"][0]
            job.update(run_id=guard.HOLDER_RUN, run_attempt=1, head_sha=guard.HOLDER_SHA,
                       workflow_name="Real-weight validation")
            valid = [{"body": run()}, {"body": jobs()}, {"body": group()}, {"body": job}]
            with patch.object(guard, "api", side_effect=valid), patch.object(guard, "gpu0_actors", return_value="raw"):
                owner.holder()
            self.assertIn("holder_readback", owner.path.read_text(encoding="utf-8"))
            for failure in (RuntimeError("network unavailable"), TimeoutError("network deadline")):
                with patch.object(guard, "api", side_effect=failure), self.assertRaises(type(failure)):
                    owner.holder()
            owner.metadata_checked = None
            invalid = copy.deepcopy(valid); invalid[0]["body"]["status"] = "completed"
            with patch.object(guard, "api", side_effect=invalid), patch.object(guard, "gpu0_actors") as actors, self.assertRaises(RuntimeError):
                owner.holder()
            actors.assert_not_called()

    def test_watchdog_refusal_reaps_only_popen_tree(self):
        child = Owned(); owner = Mock(failed=threading.Event(), fault="holder ended", cycle_started=None)
        owner.failed.set()
        def taskkill(argv, **kwargs):
            self.assertEqual(argv, ["taskkill", "/PID", "777", "/T", "/F"])
            self.assertEqual(kwargs["timeout"], 15)
            child.code = -9
            return Mock(returncode=0)
        with patch.object(guard.subprocess, "run", side_effect=taskkill) as killer:
            code, timed_out, error = guard.wait(child, owner, 100)
        self.assertEqual(code, -9); self.assertFalse(timed_out)
        self.assertIn("holder ended", error); self.assertEqual(killer.call_count, 1)

    def test_wait_timeout_keyboardinterrupt_and_stale_cycle_reap(self):
        for situation in ("timeout", "interrupt", "stale"):
            child = Owned(); owner = Mock(failed=threading.Event(), fault=None, cycle_started=None)
            if situation == "stale": owner.cycle_started = time.monotonic() - 100
            if situation == "interrupt": child.wait = Mock(side_effect=KeyboardInterrupt("cancel"))
            def kill(argv, **kwargs): child.code = -9; return Mock(returncode=0)
            with patch.object(guard.subprocess, "run", side_effect=kill) as killer:
                code, timeout, error = guard.wait(child, owner, 0 if situation == "timeout" else 100)
            self.assertEqual(killer.call_count, 1); self.assertEqual(code, -9)
            self.assertEqual(timeout, situation == "timeout")
            if situation != "timeout": self.assertIsNotNone(error)

    def test_heartbeat_is_exact_and_full_metadata_is_bounded_not_omitted(self):
        heartbeat = jobs()["jobs"][0]
        heartbeat.update(run_id=guard.HOLDER_RUN, run_attempt=1, head_sha=guard.HOLDER_SHA,
                         workflow_name="Real-weight validation")
        def read(path):
            if path == f"actions/jobs/{guard.HOLDER_JOB}": body = heartbeat
            elif "attempts" in path: body = jobs()
            elif "concurrency_groups" in path: body = group()
            else: body = run()
            return {"body": body}
        with tempfile.TemporaryDirectory() as directory, patch.object(guard, "api", side_effect=read) as api, \
             patch.object(guard, "gpu0_actors", return_value="raw"):
            owner = guard.OwnerGuard(Path(directory), guard.ENGINE, "a" * 40)
            owner.holder(); self.assertEqual(api.call_count, 4)
            owner.holder(); self.assertEqual(api.call_count, 5)
            owner.metadata_checked -= 60
            owner.holder(); self.assertEqual(api.call_count, 9)
            for key, value in {"run_id": 1, "run_attempt": 2, "head_sha": "b" * 40,
                               "workflow_name": "other", "started_at": "other"}.items():
                wrong = {**heartbeat, key: value}
                with patch.object(guard, "api", return_value={"body": wrong}), self.subTest(key=key), self.assertRaises(RuntimeError):
                    owner.holder()

    def test_api_is_bounded_read_only_github_origin_no_redirect_token_log(self):
        response = Mock()
        response.__enter__ = Mock(return_value=response); response.__exit__ = Mock(return_value=False)
        response.headers = {"Date": datetime.now(timezone.utc).strftime("%a, %d %b %Y %H:%M:%S GMT")}
        response.read.return_value = b'{"id":1}'
        opener = Mock(open=Mock(return_value=response))
        with patch.dict(os.environ, {"GH_TOKEN": "fake-secret"}), patch.object(guard.urllib.request, "build_opener", return_value=opener):
            row = guard.api("actions/runs/1")
        request = opener.open.call_args.args[0]
        self.assertTrue(request.full_url.startswith("https://api.github.com/repos/SceneWorks/inference/"))
        self.assertEqual(opener.open.call_args.kwargs["timeout"], 3)
        self.assertNotIn("fake-secret", json.dumps(row))
        with self.assertRaises(RuntimeError): guard.NoRedirect().redirect_request(None, None, 302, "redirect", {}, "https://other.example")
        response.headers["Date"] = "Mon, 01 Jan 2001 00:00:00 GMT"
        with patch.dict(os.environ, {"GH_TOKEN": "fake"}), patch.object(guard.urllib.request, "build_opener", return_value=opener), self.assertRaises(RuntimeError):
            guard.api("actions/runs/1")

    def test_preflight_authenticates_frozen_source_and_rejects_receipt_or_source_drift(self):
        environ = {"GITHUB_RUN_ID": "8888", "GITHUB_RUN_ATTEMPT": "1", "GITHUB_JOB": "cuda",
                   "GITHUB_REPOSITORY": guard.REPO, "GITHUB_SHA": "a" * 40,
                   "GITHUB_WORKSPACE": "/workspace", "RUNNER_NAME": "cuda-windows",
                   "YUE2_IDLE_CONTEXT_RUN_ID": guard.RECEIPT, "CUDA_VISIBLE_DEVICES": "0",
                   "CUDA_DEVICE_ORDER": "PCI_BUS_ID"}
        # Portable synthetic bytes exercise the same immutable hash contract.
        synthetic = {path: path.encode() for path in guard.SOURCE_HASHES}
        hashes = {path: hashlib.sha256(data).hexdigest() for path, data in synthetic.items()}
        def read(path):
            if path == "actions/runs/8888": body = run(8888, "a" * 40, ".github/workflows/yue2-precision-proof.yml")
            elif path == "actions/runs/8888/attempts/1/jobs?per_page=100": body = jobs(999, "cuda", "cuda-windows")
            elif path == f"actions/concurrency_groups/{guard.GPU0_GROUP}": body = group(guard.GPU0_GROUP, 8888)
            elif path.startswith("contents/"):
                source_path = path[len("contents/"):].split("?ref=")[0]
                self.assertTrue(path.endswith("?ref=" + guard.HOLDER_SHA))
                body = {"path": source_path, "encoding": "base64",
                        "content": base64.b64encode(synthetic[source_path]).decode()}
            elif path == f"actions/jobs/{guard.HOLDER_JOB}":
                body = jobs()["jobs"][0]
                body.update(run_id=guard.HOLDER_RUN, run_attempt=1, head_sha=guard.HOLDER_SHA,
                            workflow_name="Real-weight validation")
            elif path == f"actions/runs/{guard.HOLDER_RUN}": body = run()
            elif "attempts/1/jobs" in path: body = jobs()
            else: body = group()
            return {"body": body}
        def git(argv, **kwargs):
            return Mock(stdout=(guard.ENGINE if argv[2].endswith("/engine") else "a" * 40)
                        if argv[-1] == "HEAD" else "")
        with tempfile.TemporaryDirectory() as directory, patch.dict(os.environ, environ), \
             patch.object(guard, "IS_WINDOWS", True), patch.object(guard, "SOURCE_HASHES", hashes), \
             patch.object(guard, "reviewed_background", return_value=background()), \
             patch.object(guard, "api", side_effect=read), patch.object(guard.subprocess, "run", side_effect=git), \
             patch.object(guard, "gpu0_actors", return_value="raw"):
            owner = guard.OwnerGuard(Path(directory), guard.ENGINE, "a" * 40)
            owner.preflight()
            self.assertEqual(owner.proof_job["id"], 999)
            for key, value in {"YUE2_IDLE_CONTEXT_RUN_ID": "37106146499", "CUDA_VISIBLE_DEVICES": "1",
                               "CUDA_DEVICE_ORDER": "FASTEST_FIRST", "GITHUB_RUN_ATTEMPT": "2",
                               "GITHUB_JOB": "metal", "GITHUB_REPOSITORY": "other/repo", "GITHUB_SHA": "b" * 40}.items():
                with patch.dict(os.environ, {key: value}), self.subTest(key=key), self.assertRaises(RuntimeError):
                    owner.preflight()
            with patch.object(guard, "SOURCE_HASHES", {**hashes, next(iter(hashes)): "0" * 64}), self.assertRaises(RuntimeError):
                owner.preflight()
            with patch.object(guard.subprocess, "run", return_value=Mock(stdout="dirty")), self.assertRaises(RuntimeError):
                owner.preflight()

    def test_mac_preflight_authenticates_both_foreign_workflow_revisions_before_launch(self):
        environ = {"GITHUB_RUN_ID": "8888", "GITHUB_RUN_ATTEMPT": "1", "GITHUB_JOB": "cuda",
                   "GITHUB_REPOSITORY": guard.REPO, "GITHUB_SHA": "a" * 40,
                   "GITHUB_WORKSPACE": "/workspace", "RUNNER_NAME": "cuda-windows",
                   "YUE2_IDLE_CONTEXT_RUN_ID": guard.RECEIPT, "CUDA_VISIBLE_DEVICES": "0",
                   "CUDA_DEVICE_ORDER": "PCI_BUS_ID"}
        bytes_by_sha = {guard.MAC_SHA: b"reviewed mac workflow",
                        guard.PENDING_SHA: b"reviewed pending workflow"}
        digest_by_sha = {sha: hashlib.sha256(data).hexdigest() for sha, data in bytes_by_sha.items()}
        def read(path):
            if path == "actions/runs/8888":
                body = run(8888, "a" * 40, ".github/workflows/yue2-precision-proof.yml")
            elif path == "actions/runs/8888/attempts/1/jobs?per_page=100":
                body = jobs(999, "cuda", "cuda-windows")
            elif path == f"actions/concurrency_groups/{guard.GPU0_GROUP}":
                body = group(guard.GPU0_GROUP, 8888)
            else:
                sha = path.split("?ref=")[1]
                body = {"path": ".github/workflows/real-weights.yml", "encoding": "base64",
                        "content": base64.b64encode(bytes_by_sha[sha]).decode()}
            return {"body": body}
        def git(argv, **kwargs):
            return Mock(stdout=(guard.ENGINE if argv[2].endswith("/engine") else "a" * 40)
                        if argv[-1] == "HEAD" else "")
        with tempfile.TemporaryDirectory() as directory, patch.dict(os.environ, environ), \
             patch.object(guard, "IS_WINDOWS", True), \
             patch.object(guard, "MAC_WORKFLOW_SHA256", digest_by_sha[guard.MAC_SHA]), \
             patch.object(guard, "PENDING_WORKFLOW_SHA256", digest_by_sha[guard.PENDING_SHA]), \
             patch.object(guard, "reviewed_background", return_value=background()), \
             patch.object(guard.subprocess, "run", side_effect=git), \
             patch.object(guard, "api", side_effect=read) as api:
            owner = guard.OwnerGuard(Path(directory), guard.ENGINE, "a" * 40,
                                     mode="owner-gpu0-mac-anchor")
            with patch.object(owner, "holder"):
                owner.preflight()
                self.assertEqual(api.call_count, 5)
                self.assertTrue(any(call.args[0].endswith("?ref=" + guard.MAC_SHA)
                                    for call in api.call_args_list))
                self.assertTrue(any(call.args[0].endswith("?ref=" + guard.PENDING_SHA)
                                    for call in api.call_args_list))
                with patch.object(guard, "PENDING_WORKFLOW_SHA256", "0" * 64), self.assertRaises(RuntimeError):
                    owner.preflight()

    def test_failed_preflight_never_spawns_app_and_token_is_not_in_owned_child(self):
        with tempfile.TemporaryDirectory() as directory, patch.dict(os.environ, {
            "EXPECTED_ENGINE_SHA": guard.ENGINE, "EXPECTED_CONTROL_SHA": "a" * 40,
            "YUE2_APP_PRECISION_JOB_STARTED_UTC_NS": str(time.time_ns())}):
            evidence = Path(directory)
            with patch.object(guard.OwnerGuard, "preflight", side_effect=RuntimeError("holder ended")), \
                 patch.object(guard.subprocess, "Popen") as launch, self.assertRaises(RuntimeError):
                guard.guarded_command(["node", "unchanged-case"], evidence, {}, None, evidence, "case")
            launch.assert_not_called()
            child = Owned(); child.code = 0
            with patch.object(guard.OwnerGuard, "preflight"), patch.object(guard.OwnerGuard, "arm"), patch.object(guard.OwnerGuard, "start"), \
                 patch.object(guard.OwnerGuard, "finish"), patch.object(guard.subprocess, "Popen", return_value=child) as launch:
                status = guard.guarded_command(["node", "unchanged-case"], evidence,
                                               {"GH_TOKEN": "secret", "GITHUB_TOKEN": "secret", "CUDA_VISIBLE_DEVICES": "0"},
                                               None, evidence, "case")
            self.assertEqual(status, 0)
            self.assertEqual(launch.call_args.args[0], ["node", "unchanged-case"])
            self.assertEqual(launch.call_args.kwargs["env"], {"CUDA_VISIBLE_DEVICES": "0"})
            self.assertIn("provisional-holder-chronology", (evidence / "gpu0-holder-chronology.jsonl").read_text(encoding="utf-8"))

    def test_app_mac_route_uses_forwarded_environment_and_refuses_before_child(self):
        with tempfile.TemporaryDirectory() as directory, patch.dict(os.environ, {
            "EXPECTED_ENGINE_SHA": guard.ENGINE, "EXPECTED_CONTROL_SHA": "a" * 40}):
            evidence = Path(directory)
            seen = []
            def refuse(owner):
                seen.append(owner.mode)
                raise RuntimeError("Mac barrier lost")
            with patch.object(guard.OwnerGuard, "preflight", refuse), \
                 patch.object(guard.subprocess, "Popen") as launch, \
                 self.assertRaisesRegex(RuntimeError, "Mac barrier lost"):
                guard.guarded_command(["node", "unchanged-case"], evidence,
                                      {"YUE2_CUDA_SCHEDULING_MODE": "owner-gpu0-mac-anchor"},
                                      None, evidence, "case")
            launch.assert_not_called()
            self.assertEqual(seen, ["owner-gpu0-mac-anchor"])

    def test_expired_app_job_tail_refuses_before_popen(self):
        with tempfile.TemporaryDirectory() as directory, patch.dict(os.environ, {
            "EXPECTED_ENGINE_SHA": guard.ENGINE, "EXPECTED_CONTROL_SHA": "a" * 40,
            "YUE2_APP_PRECISION_JOB_STARTED_UTC_NS": str(time.time_ns() - 480 * 60 * 1_000_000_000)}), \
             patch.object(guard.OwnerGuard, "preflight"), patch.object(guard.subprocess, "Popen") as launch, \
             self.assertRaisesRegex(RuntimeError, "cleanup/upload tail"):
            guard.guarded_command(["node"], Path(directory), {}, None, Path(directory), "case")
        launch.assert_not_called()

    def test_engine_owner_refusal_happens_before_any_popen(self):
        from scripts.tests import test_yue2_precision_proof as existing
        control = existing.CONTROL
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory); reference = root / "reference"; reference.mkdir()
            (reference / "vae_real_reference.safetensors").write_bytes(b"fixture")
            binary = root / "binary"; binary.write_bytes(b"binary")
            args = Mock(evidence=root / "evidence", reference=reference, binary=binary,
                        quant_smoke_binary=binary, vae_smoke_binary=binary,
                        work_dir=root / "listening", engine_sha=guard.ENGINE, control_sha="a" * 40,
                        app_sha="", backend="cuda", cuda_scheduling_mode="owner-gpu0")
            with patch.dict(os.environ, {"CUDA_VISIBLE_DEVICES": "0", "RUNNER_NAME": "cuda-windows",
                                         "YUE2_PRECISION_JOB_STARTED_UTC_NS": str(time.time_ns())}), \
                 patch.object(control, "sha256", return_value=control.REFERENCE_SHA256), \
                 patch.object(control, "verify_revisions"), patch.object(control.subprocess, "run", return_value=Mock(stdout="")), \
                 patch.object(existing.IDLE, "require_remaining_window", return_value=({}, root)), \
                 patch.object(control, "retain_reviewed_baseline", return_value=[]), \
                 patch.object(control, "retain_cuda_physical_evidence", return_value=[]), \
                 patch.object(control, "verify_binary_identity", return_value={}), \
                 patch.object(control, "cuda_physical_census", return_value=("raw", [])), \
                 patch.object(guard.OwnerGuard, "preflight", side_effect=RuntimeError("exact holder lost")), \
                 patch.object(control.subprocess, "Popen") as launch, self.assertRaisesRegex(RuntimeError, "exact holder lost"):
                control.execute(args)
            launch.assert_not_called()

    def test_signals_are_armed_before_popen_and_flag_instead_of_interrupting_handle_creation(self):
        handlers = {}
        with tempfile.TemporaryDirectory() as directory, \
             patch.object(guard.signal, "signal", side_effect=lambda number, handler: handlers.update({number: handler})), \
             patch.object(guard.signal, "getsignal", return_value="original"):
            owner = guard.OwnerGuard(Path(directory), guard.ENGINE, "a" * 40)
            owner.arm()
            handlers[guard.signal.SIGTERM](guard.signal.SIGTERM, None)
            self.assertTrue(owner.failed.is_set())
            child = Owned()
            def kill(argv, **kwargs): child.code = -9; return Mock(returncode=0)
            with patch.object(guard.subprocess, "run", side_effect=kill) as killer:
                guard.wait(child, owner, 100)
            self.assertEqual(killer.call_count, 1)
            with self.assertRaises(RuntimeError): owner.finish()
            self.assertEqual(owner.signals, {})
            self.assertTrue(all(value == "original" for value in handlers.values()))

    def test_watchdog_network_fault_sets_failure_and_preserves_refusal(self):
        with tempfile.TemporaryDirectory() as directory:
            owner = guard.OwnerGuard(Path(directory), guard.ENGINE, "a" * 40)
            with patch.object(owner, "holder", side_effect=TimeoutError("network timeout")):
                owner.start(Owned())
                self.assertTrue(owner.failed.wait(1))
                with self.assertRaises(RuntimeError): owner.finish()
            self.assertIn("network timeout", owner.path.read_text(encoding="utf-8"))
            self.assertEqual(owner.signals, {})

    def test_provisional_summary_cannot_claim_final_acceptance(self):
        with tempfile.TemporaryDirectory() as directory:
            owner = guard.OwnerGuard(Path(directory), guard.ENGINE, "a" * 40)
            summary = owner.summary()
            self.assertEqual(summary["acceptance"], "provisional-holder-chronology")
            self.assertIn("completed_at strictly after proof job completed_at", summary["final_acceptance_requires"])
            self.assertEqual(summary["maximum_detection_seconds"], 65)
            self.assertEqual(summary["maximum_group_detection_seconds"], 125)
            self.assertEqual(summary["steady_requests_per_hour_upper_bound"], 540)
            mac = guard.OwnerGuard(Path(directory), guard.ENGINE, "a" * 40,
                                   mode="owner-gpu0-mac-anchor").summary()
            self.assertEqual(mac["acceptance"], "provisional-mac-anchor-chronology")
            self.assertEqual(mac["pending_run_id"], guard.PENDING_RUN)
            self.assertIn("strictly after the entire owned proof job", mac["final_acceptance_requires"])

    def test_mac_anchor_exact_barrier_and_pending_mutations_refuse(self):
        pending = run(guard.PENDING_RUN, guard.PENDING_SHA)
        pending.update(created_at=guard.PENDING_CREATED, status="pending")
        group = {"group_name": guard.OLD_GROUP, "total_count": 2, "group_members": [
            {"run_id": guard.MAC_RUN, "status": "in_progress"},
            {"run_id": guard.PENDING_RUN, "status": "pending"}]}
        guard.pending_identity(pending)
        guard.mac_group_barrier(group)
        for key, value in {"id": 1, "head_sha": "0" * 40, "run_attempt": 2,
                           "created_at": "other", "status": "queued",
                           "conclusion": "success", "path": "other.yml"}.items():
            changed = {**pending, key: value}
            with self.subTest(pending=key), self.assertRaises(RuntimeError):
                guard.pending_identity(changed)
        for mutate in (lambda x: x.update(total_count=3),
                       lambda x: x["group_members"][0].update(status="completed"),
                       lambda x: x["group_members"][1].update(status="in_progress"),
                       lambda x: x["group_members"][1].update(run_id=99),
                       lambda x: x["group_members"].append({"run_id": 99, "status": "pending"})):
            changed = copy.deepcopy(group); mutate(changed)
            with self.assertRaises(RuntimeError):
                guard.mac_group_barrier(changed)

    def test_mac_anchor_heartbeat_inventory_and_network_refuse_before_gpu_query(self):
        mac = run(guard.MAC_RUN, guard.MAC_SHA)
        mac["created_at"] = guard.MAC_CREATED
        pending = run(guard.PENDING_RUN, guard.PENDING_SHA)
        pending.update(created_at=guard.PENDING_CREATED, status="pending")
        inventory = jobs(guard.MAC_JOB, guard.MAC_NAME, guard.MAC_RUNNER)
        selected = inventory["jobs"][0]
        selected.update(runner_id=guard.MAC_RUNNER_ID, started_at=guard.MAC_STARTED,
                        run_id=guard.MAC_RUN, run_attempt=1, head_sha=guard.MAC_SHA,
                        workflow_name="Real-weight validation")
        group = {"group_name": guard.OLD_GROUP, "total_count": 2, "group_members": [
            {"run_id": guard.MAC_RUN, "status": "in_progress"},
            {"run_id": guard.PENDING_RUN, "status": "pending"}]}
        responses = [mac, inventory, pending, {"total_count": 0, "jobs": []}, group, selected]
        with tempfile.TemporaryDirectory() as directory:
            owner = guard.OwnerGuard(Path(directory), guard.ENGINE, "a" * 40,
                                     mode="owner-gpu0-mac-anchor")
            with patch.object(guard, "api", side_effect=[{"body": value} for value in responses]) as api, \
                 patch.object(guard, "gpu0_actors", return_value="typed GPU0"):
                owner.holder()
                self.assertEqual(api.call_count, 6)
            with patch.object(guard, "api", return_value={"body": selected}) as api, \
                 patch.object(guard, "gpu0_actors", return_value="typed GPU0"):
                owner.holder()
                api.assert_called_once_with(f"actions/jobs/{guard.MAC_JOB}")
            for index, mutation in ((0, lambda x: x.update(status="completed")),
                                    (1, lambda x: x["jobs"][1].update(conclusion=None)),
                                    (2, lambda x: x.update(status="in_progress")),
                                    (3, lambda x: x.update(total_count=1)),
                                    (4, lambda x: x["group_members"][1].update(run_id=99)),
                                    (5, lambda x: x.update(runner_id=99))):
                bad = copy.deepcopy(responses); mutation(bad[index])
                owner.metadata_checked = None
                with self.subTest(index=index), patch.object(guard, "api", side_effect=[{"body": value} for value in bad]), \
                     patch.object(guard, "gpu0_actors") as physical, self.assertRaises(RuntimeError):
                    owner.holder()
                physical.assert_not_called()
            owner.metadata_checked = None
            with patch.object(guard, "api", side_effect=TimeoutError("API fault")), \
                 patch.object(guard, "gpu0_actors") as physical, self.assertRaises(TimeoutError):
                owner.holder()
            physical.assert_not_called()


if __name__ == "__main__": unittest.main()
