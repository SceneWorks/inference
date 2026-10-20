"""CPU-only campaign tests. Every subprocess is globally trapped: mutations cannot launch tools."""
import contextlib
import importlib.util
import io
import json
import os
from pathlib import Path
import signal
import subprocess
import tempfile
import argparse
import unittest
from unittest import mock


def argparse_error():
    return argparse.ArgumentTypeError

MODULE = Path(__file__).resolve().parents[1] / "ios/campaign.py"
spec = importlib.util.spec_from_file_location("ios_campaign", MODULE)
campaign = importlib.util.module_from_spec(spec)
spec.loader.exec_module(campaign)


class CampaignTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        # The anchor listing: per tier x mode a resident (rung 0) and an optimized (rung 4, cheapest
        # selection, 3 repeats) anchor at 512 with 1 step; plus the two validation cases.
        self.entries = []
        for tier in campaign.TIERS:
            for mode in ("text", "reference"):
                self.entries.append({"case": f"fake/fake/{tier}/512/{mode}/0/base/1", "purpose": "resident", "repeats": 1})
                self.entries.append({"case": f"fake/fake/{tier}/512/{mode}/4/s_t256/1", "purpose": "optimized", "repeats": 2})
        self.entries.append({"case": "fake/fake/q4/512/text/4/s_t256/0", "purpose": "validation_steps", "repeats": 1})
        self.entries.append({"case": "fake/fake/q4/1024/text/4/s_t256/1", "purpose": "validation_geometry", "repeats": 1})
        self.entries.append({"case": "fake/fake/q4/1024/control/4/s_t256/1", "purpose": "validation_mode", "repeats": 1})
        self.keys = [e["case"] for e in self.entries]
        self.catalog = self.root / "catalog.json"
        self.catalog.write_text(json.dumps({"schema": 2, "cases": self.entries}), encoding="utf-8")
        self.guard = mock.patch.object(campaign.subprocess, "Popen", side_effect=AssertionError("TEST MUST NOT START ANY REAL PROCESS"))
        self.guard.start()
        self.addCleanup(self.guard.stop)
        self.addCleanup(mock.patch.stopall)
        self.sleeper = mock.patch.object(campaign.time, "sleep")
        self.sleeper.start()
        mock.patch.object(campaign.uuid, "uuid4", return_value=mock.Mock(hex="fixture-run")).start()

    def main(self, *args):
        with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
            return campaign.main([*args, "--catalog", str(self.catalog)])

    def success(self, key=None):
        row = campaign.failure(key or self.keys[0], "render")
        row.update(status="successful", production_success=True, elapsed_seconds=1.25,
                   mlx_active_peak_bytes=10, mlx_cache_peak_bytes=20, mlx_active_cache_peak_bytes=30,
                   mlx_peak_bytes=15, phys_footprint_peak_bytes=123,
                   kernel_lifetime_phys_footprint_peak_bytes=150, min_headroom_mib=456,
                   diagnostic=False, build_sha="test-sha", expected_render_count=1)
        return row

    def fake_session(self, row, entitlement="YES"):
        row.setdefault("run_id", "fixture-run")
        args = campaign.parser().parse_args(["run", "--device", "fixture-device", "--timeout-seconds", "77", "--output", str(self.root / "run.jsonl")])
        session = object.__new__(campaign.DeviceSession)
        session.args, session.bundle, session.blocked, session.sha, session.env = args, "fixture.bundle", False, "test-sha", {}
        session.exited = mock.Mock(return_value=True)
        session.device = mock.Mock(return_value=subprocess.CompletedProcess([], 0, "", ""))
        def copy(remote, local):
            if remote == "smoke-report.txt":
                local.write_text("CAMPAIGN_JSON " + json.dumps(row), encoding="utf-8")
            elif remote == "launch-memory.txt":
                local.write_text("launch: increased-memory-limit = " + entitlement, encoding="utf-8")
            elif remote == "renders.txt":
                local.write_text("campaign-" + campaign.artifact_stem(row["case"]) + "-0.png\n", encoding="utf-8")
            else:
                local.write_bytes(b"fixture PNG")
            return subprocess.CompletedProcess([], 0, "", "")
        session.copy = mock.Mock(side_effect=copy)
        return session

    def test_public_cli_listing_is_pure_and_rung_zero_is_selected(self):
        self.assertEqual(self.main("list", "--tier", "q8", "--mode", "reference", "--rung", "4"), 0)
        args = campaign.parser().parse_args(["list", "--rung", "0"])
        self.assertFalse(campaign.selected(args, "fake/fake/q4/512/text/1/base/1"))
        self.assertTrue(campaign.selected(args, "fake/fake/q4/512/text/0/base/1"))
        args = campaign.parser().parse_args(["list", "--selection", "t512"])
        self.assertTrue(campaign.selected(args, "fake/fake/q4/512/text/2/t512/1"))
        self.assertFalse(campaign.selected(args, "fake/fake/q4/512/text/2/t768/1"))
        args = campaign.parser().parse_args(["list", "--purpose", "optimized"])
        self.assertEqual([e["purpose"] for e in self.entries if campaign.selected(args, e)], ["optimized"] * 6)
        args = campaign.parser().parse_args(["list", "--steps", "0"])
        self.assertEqual([e["case"] for e in self.entries if campaign.selected(args, e)], ["fake/fake/q4/512/text/4/s_t256/0"])

    def test_the_listing_is_the_anchor_shape_and_malformed_listings_are_refused(self):
        args = campaign.parser().parse_args(["list", "--catalog", str(self.catalog)])
        self.assertEqual(campaign.catalog(args)["cases"], self.entries)
        bad = {
            "schema 1 grid": {"schema": 1, "cases": self.keys},
            "bare key": {"schema": 2, "cases": [self.keys[0]]},
            "unknown purpose": {"schema": 2, "cases": [dict(self.entries[0], purpose="ladder")]},
            "too few optimized repeats": {"schema": 2, "cases": [dict(self.entries[1], repeats=1)]},
            "zero repeats": {"schema": 2, "cases": [dict(self.entries[0], repeats=0)]},
            "seven-field key": {"schema": 2, "cases": [dict(self.entries[0], case="fake/fake/q4/512/text/0/base")]},
            "duplicate": {"schema": 2, "cases": [self.entries[0], self.entries[0]]},
            "extra field": {"schema": 2, "cases": [dict(self.entries[0], edge=512)]},
            "empty": {"schema": 2, "cases": []},
        }
        for name, listing in bad.items():
            with self.subTest(name=name):
                path = self.root / (name.replace(" ", "-") + ".json")
                path.write_text(json.dumps(listing), encoding="utf-8")
                with self.assertRaises((ValueError, argparse_error())):
                    campaign.catalog(campaign.parser().parse_args(["list", "--catalog", str(path)]))
        # No 20,000-case budget: the listing is finite by construction and --max-cases optional.
        self.assertIsNone(campaign.parser().parse_args(["run"]).max_cases)
        self.assertEqual(campaign.parser().parse_args(["run", "--max-cases", "50000"]).max_cases, 50000)

    def test_invalid_finite_budgets_and_cases(self):
        for options in (("--case", "p/../q4/512/text/0/base/1"), ("--case", "p/p/q4/512/text/0/base"), ("--case", "p/p/q4/512/text/0/t5.1/1"), ("--case", "p/p/q4/512/text/0/base/4294967296"), ("--timeout-seconds", "0"), ("--timeout-seconds", "20001"), ("--max-cases", "0"), ("--tier", "q5"), ("--edge", "999999"), ("--rung", "5"), ("--steps", "4294967296"), ("--steps", "-1"), ("--steps", "not-an-integer"), ("--purpose", "ladder")):
            with self.subTest(options=options), contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
                campaign.parser().parse_args(["run", *options])
        with self.assertRaises(SystemExit):
            self.main("run", "--device", "fake", "--output", str(self.root / "not-written"))
        self.assertFalse((self.root / "not-written").exists())

    def test_capture_step_transport_preserves_u32_domain_without_enumerated_choices(self):
        for steps in (0, 1, 101, (1 << 32) - 1):
            key = f"p/p/q4/512/text/0/base/{steps}"
            try:
                with contextlib.redirect_stderr(io.StringIO()):
                    parsed = campaign.parser().parse_args(["list", "--steps", str(steps), "--case", key])
            except SystemExit as error:
                self.fail(f"legal wire/default step count {steps} was refused by scalar parser: exit={error.code}")
            self.assertEqual(parsed.steps, steps)
            self.assertEqual(parsed.case, key)
        action = next(action for action in campaign.parser()._actions if action.dest == "steps")
        self.assertIsNone(action.choices, "argparse must not enumerate a step-sized domain even on invalid input")
        # Zero remains the existing explicit default sentinel, not a request for zero solver steps.
        self.assertEqual(campaign.step_count("0"), 0)
        row = self.capture_row(self.keys[1], "diagnostic_success")
        for invalid_count in (0, True, -1, 1 << 32):
            row["steps"] = invalid_count
            self.assertIsNone(campaign.capture_of(row), "recorded actual solver steps must be nonzero u32, never the CLI default sentinel")
        row["steps"] = (1 << 32) - 1
        self.assertIsNotNone(campaign.capture_of(row))  # Schema only; does not execute a large run.
        for invalid in ("-1", "4294967296", "not-an-integer"):
            with self.assertRaises(argparse.ArgumentTypeError):
                campaign.step_count(invalid)

    def test_long_recipe_case_preserves_wire_identity_with_digest_artifact_paths(self):
        key = "sdxl/sdxl/q4/1024/text/4/base/3/sdxl1-" + "ab" * 1024
        self.assertEqual(campaign.case_key(key), key)
        self.assertEqual(campaign.parser().parse_args(["run", "--case", key]).case, key)
        stem = campaign.artifact_stem(key)
        self.assertEqual(stem, "recipe-" + campaign.hashlib.sha256(key.encode("utf-8")).hexdigest())
        self.assertLess(len(stem), 128)
        self.assertNotEqual(stem, campaign.artifact_stem(key + "ab"))
        legacy = self.keys[0]
        self.assertEqual(campaign.artifact_stem(legacy), legacy.replace("/", "-"))
        session = self.fake_session(self.success(key))
        result = session.run_case(key)
        self.assertEqual(result["case"], key)
        self.assertEqual(result["status"], "successful", result)
        files = list((self.root / "run.artifacts" / stem).glob("*.png"))
        self.assertEqual(len(files), 1)
        self.assertLess(len(files[0].name), 255)
        for invalid in (key + "/extra", key + "/", key + "."):
            with self.assertRaises(argparse_error()):
                campaign.case_key(invalid)

    def test_step_validation_never_pairs_different_sampling_recipes(self):
        short_key = "sdxl/sdxl/q4/512/text/4/base/3/sdxl1-aa"
        full_key = "sdxl/sdxl/q4/512/text/4/base/0/sdxl1-bb"
        short = dict(self.capture_row(short_key, "diagnostic_success"), purpose="optimized")
        full = dict(self.capture_row(full_key, "diagnostic_success"), purpose="validation_steps")
        report = campaign.steps_validation([short, full])
        self.assertEqual(report[0]["problem"], "no completed optimized anchor of the same case")
        full["case"] = full_key.replace("sdxl1-bb", "sdxl1-aa")
        matched = campaign.steps_validation([short, full])
        self.assertIn("ratio", matched[0])
        self.assertNotIn("problem", matched[0])

    def test_install_once_budget_and_all_explicit_status_rows(self):
        output = self.root / "evidence.jsonl"
        with mock.patch.object(campaign, "DeviceSession") as cls, mock.patch.object(campaign, "invoke", side_effect=lambda session, key, **kw: dict(campaign.failure(key, "no capture"), status="admitted_refused", refusal_kind="no_evidence")) as invoke:
            cls.return_value.blocked = False
            self.assertEqual(self.main("run", "--all-cases", "--device", "fake", "--output", str(output), "--max-cases", "2"), 2)
            cls.return_value.install.assert_called_once()
            self.assertEqual(invoke.call_count, 2)
        rows = [json.loads(line) for line in output.read_text(encoding="utf-8").splitlines()]
        self.assertEqual([r["case"] for r in rows], self.keys)
        self.assertTrue(all(r["status"] == "skipped" for r in rows[2:]))
        self.assertTrue(all(r["production_success"] is False for r in rows))
        with self.assertRaises(FileExistsError):
            self.main("run", "--all-cases", "--device", "fake", "--output", str(output))

    def test_next_edge_rewrite_and_probes_count_toward_max_cases(self):
        key = self.keys[0]
        for budget, expected_calls in ((1, 1), (2, 2)):
            output = self.root / f"next-{budget}.jsonl"
            with mock.patch.object(campaign, "DeviceSession") as cls, mock.patch.object(campaign, "invoke", side_effect=lambda session, current, **kw: self.success(current)) as run:
                cls.return_value.blocked = False
                self.main("run", "--case", key, "--next-edge", "768", "--device", "fake", "--output", str(output), "--max-cases", str(budget))
                self.assertEqual(run.call_count, expected_calls)
                if budget == 2:
                    self.assertEqual(run.call_args.args[1], key.replace("/512/", "/768/"))
                    self.assertEqual(run.call_args.kwargs, {"refusal_only": True})
            rows = [json.loads(line) for line in output.read_text(encoding="utf-8").splitlines()]
            self.assertEqual(len(rows), 2)
            if budget == 1:
                self.assertEqual(rows[-1]["status"], "skipped")

    def test_probe_consumes_budget_before_the_next_base_case(self):
        output = self.root / "probe-budget.jsonl"
        with mock.patch.object(campaign, "DeviceSession") as cls, mock.patch.object(campaign, "invoke", side_effect=lambda session, current, **kw: self.success(current)) as run:
            cls.return_value.blocked = False
            self.main("run", "--provider", "fake", "--tier", "q4", "--edge", "512", "--mode", "text", "--next-edge", "768", "--max-cases", "2", "--device", "fake", "--output", str(output))
            self.assertEqual(run.call_count, 2)
        rows = [json.loads(line) for line in output.read_text(encoding="utf-8").splitlines()]
        self.assertEqual(rows[2]["status"], "skipped")

    def test_next_edge_cannot_use_diagnostics(self):
        with self.assertRaises(SystemExit):
            self.main("run", "--case", self.keys[0], "--next-edge", "768", "--diagnostic", "--device", "fake", "--output", str(self.root / "bad.jsonl"))

    def test_next_edge_must_exceed_selection(self):
        for edge in (256, 512):
            with self.assertRaises(SystemExit):
                self.main("run", "--case", self.keys[0], "--next-edge", str(edge), "--device", "fake", "--output", str(self.root / "bad.jsonl"))

    def test_refusal_only_requires_capacity_not_missing_evidence(self):
        key = self.keys[0]
        for kind in ("no_evidence", "mode", "strategy", "capacity"):
            row = dict(campaign.failure(key, "no capture"), status="admitted_refused", refusal_kind=kind, build_sha="test-sha", refusal_only=True, artifact_identity={"layout_sha256": "a" * 64}, evidence_revision="fixture-capture", selected_parameters={"decode_tile_edge": 192})
            session = self.fake_session(row)
            out = session.run_case(key, refusal_only=True)
            self.assertEqual(out["status"], "admitted_refused" if kind == "capacity" else "diagnostic_failure")
            # Separate temp directory for the next fixture, never overwrite artifacts.
            session.args.output = self.root / (kind + ".jsonl")
            for artifact in (self.root / "run.artifacts").glob("*"):
                # No removal needed: subsequent session output gets a new parent.
                self.assertTrue(artifact.is_dir())
            self.root = self.root / kind
            self.root.mkdir()

    def test_zero_footprint_rejects_otherwise_valid_success(self):
        row = self.success()
        self.assertEqual(campaign.parse_report("CAMPAIGN_JSON " + json.dumps(row), self.keys[0])["status"], "successful")
        row["phys_footprint_peak_bytes"] = 0
        self.assertEqual(campaign.parse_report("CAMPAIGN_JSON " + json.dumps(row), self.keys[0])["status"], "diagnostic_failure")

    def test_diagnostic_success_requires_explicit_diagnostic_lane(self):
        row = dict(self.success(), status="diagnostic_success", production_success=False, diagnostic=False)
        self.assertEqual(campaign.parse_report("CAMPAIGN_JSON " + json.dumps(row), self.keys[0])["status"], "diagnostic_failure")
        row["diagnostic"] = True
        self.assertEqual(campaign.parse_report("CAMPAIGN_JSON " + json.dumps(row), self.keys[0])["status"], "diagnostic_success")

    def test_typed_status_and_finite_metric_errors_never_abort(self):
        for status in ([], {}, None, "unknown"):
            row = dict(self.success(), status=status)
            self.assertEqual(campaign.parse_report("CAMPAIGN_JSON " + json.dumps(row), self.keys[0])["status"], "diagnostic_failure")
        for value in (float("nan"), -1, None, True):
            row = dict(self.success(), min_headroom_mib=value)
            self.assertEqual(campaign.parse_report("CAMPAIGN_JSON " + json.dumps(row), self.keys[0])["status"], "diagnostic_failure")
        self.assertEqual(campaign.parse_report("CAMPAIGN_JSON []", self.keys[0])["status"], "diagnostic_failure")

    def test_legacy_capture_and_unadmitted_flags_are_stripped(self):
        with mock.patch.dict(os.environ, {"IOS_SMOKE_SANA_CAPTURE": "1", "IOS_SMOKE_UNADMITTED": "1", "IOS_CAMPAIGN_DIAGNOSTIC": "1"}):
            env = campaign.clean_environment()
        self.assertNotIn("IOS_SMOKE_SANA_CAPTURE", env)
        self.assertNotIn("IOS_SMOKE_UNADMITTED", env)
        self.assertNotIn("IOS_CAMPAIGN_DIAGNOSTIC", env)

    def test_host_timeout_owns_process_group(self):
        child = mock.Mock(pid=12345)
        child.communicate.side_effect = [subprocess.TimeoutExpired("fixture", 1), subprocess.TimeoutExpired("fixture", 3), ("", "")]
        with mock.patch.object(campaign.subprocess, "Popen", return_value=child) as popen, mock.patch.object(campaign.os, "killpg") as kill, self.assertRaises(subprocess.TimeoutExpired):
            campaign.command(["fixture-only"], 1)
        self.assertTrue(popen.call_args.kwargs["start_new_session"])
        self.assertEqual(kill.call_args_list, [mock.call(12345, signal.SIGTERM), mock.call(12345, signal.SIGKILL)])

    def test_device_deadline_exit_code_entitlement_and_artifact_copy(self):
        session = self.fake_session(self.success())
        result = session.run_case(self.keys[0])
        self.assertEqual(result["status"], "successful")
        self.assertEqual(result["launcher_exit_code"], 0)
        self.assertTrue(result["device_exited"])
        launched = session.device.call_args.args[0]
        self.assertNotIn("--terminate-existing", launched)
        variables = json.loads(launched[launched.index("--environment-variables") + 1])
        self.assertEqual(variables["IOS_CAMPAIGN_TIMEOUT_SECONDS"], "77")
        self.assertTrue(any(call.args[0] == "campaign-" + self.keys[0].replace("/", "-") + "-0.png" for call in session.copy.call_args_list))
        session = self.fake_session(self.success(), entitlement="NO")
        session.args.output = self.root / "wrong-entitlement.jsonl"
        self.assertEqual(session.run_case(self.keys[0])["status"], "diagnostic_failure")

    def test_stale_launch_identity_cannot_pass_even_with_matching_case_and_sha(self):
        session = self.fake_session(dict(self.success(), run_id="previous-launch"))
        row = session.run_case(self.keys[0])
        self.assertEqual(row["status"], "diagnostic_failure")
        self.assertIn("different case launch", row["reason"])

    def test_live_device_process_blocks_new_case_without_killing(self):
        session = self.fake_session(self.success())
        session.exited.return_value = False
        row = session.run_case(self.keys[0])
        self.assertEqual(row["status"], "diagnostic_failure")
        self.assertFalse(row["device_exited"])
        self.assertTrue(session.blocked)
        session.device.assert_not_called()

    def test_install_is_separate_from_case_launch(self):
        session = self.fake_session(self.success())
        with mock.patch.object(campaign, "command", return_value=subprocess.CompletedProcess([], 0, "", "")) as run:
            session.install()
        self.assertIn("--install-only", run.call_args.args[0])
        self.assertNotIn("--campaign-case", run.call_args.args[0])

    def test_table_has_all_metrics_and_typed_capacity_probes(self):
        row = dict(self.success(), status="diagnostic_success", production_success=False, diagnostic=True)
        table = campaign.render([row])
        self.assertIn("Active MiB | Cache MiB | Combined MiB", table)
        row.update(status="admitted_refused", refusal_only=True, refusal_kind="no_evidence")
        self.assertIn("Capacity-refused next-up probes: []", campaign.render([row]))
        row.update(refusal_kind="capacity", artifact_identity={"layout_sha256": "a" * 64}, evidence_revision="fixture-capture", selected_parameters={"decode_tile_edge": 192})
        self.assertIn(self.keys[0], campaign.render([row]).split("Capacity-refused next-up probes:")[-1])
        row["evidence_revision"] = "no-device-capture"
        self.assertIn("Capacity-refused next-up probes: []", campaign.render([row]))

    def capture_row(self, key, status):
        row = dict(self.success(key), status=status, production_success=False, diagnostic=True, evidence_class="capture",
                   evidence_key={"machine": "iPhone17,2", "provider": "fake", "execution": key.split("/")[5],
                                 "reference_count": 0, "mask_count": 0, "control_count": 0}, run_id="fixture-run",
                   baseline_phys_footprint_bytes=40, steps=int(key.split("/")[7]) or 16,  # Synthetic resolved default, not the CLI0 sentinel.
                   anchor_selection={"rung": int(key.split("/")[5])},
                   phase_peaks={"denoise": {"peak_bytes": 120, "exact": True}, "decode": {"peak_bytes": 150, "exact": False}})
        # Synthetic ABI/transport fixture only, not device evidence or a physical transfer proof.
        def snapshot(current, highwater):
            return {"method": "task_vm_info", "query_status": 0, "requested_words": 93, "returned_words": 93,
                    "current_required_words": 38, "peak_required_words": 84,
                    "current_bytes": current, "lifetime_peak_bytes": highwater}
        row.update(before_request_snapshot=snapshot(40, 40), before_model_load_snapshot=snapshot(60, 60),
                   ending_snapshot=snapshot(100, 150), observed_current_peak_bytes=150,
                   observed_model_current_peak_bytes=150, valid_kernel_queries=3, valid_model_kernel_queries=2,
                   sampling_coverage=self.observed_fixture(), quantum_query_gaps=0,
                   quantum_coverage={"checks": [{"phase": "denoise", "label": "synthetic_kernel", "context": [0], "count": 1}],
                                     "total_checks": 1, "dropped_checks": 0, "truncated": False})
        if status == "capture_refused":
            row.update(refusal_kind="capacity", refused_quantum={"label": "SANA block", "scratch_bytes": 9, "reserve_bytes": 1, "available_bytes": 3})
        return row

    def test_capture_runs_exactly_the_listed_anchor_cases_with_their_repeats(self):
        calls = []
        def fake(session, key, **kw):
            calls.append(key)
            # Resident anchors refuse before allocation (their evidence); optimized complete.
            status = "capture_refused" if key.split("/")[5] == "0" else "diagnostic_success"
            row = self.capture_row(key, status)
            row["run_id"] = f"run-{len(calls)}"
            return row
        output = self.root / "capture.jsonl"
        with mock.patch.object(campaign, "DeviceSession") as cls, mock.patch.object(campaign, "invoke", side_effect=fake):
            cls.return_value.blocked = False
            code = self.main("capture", "--all-cases", "--device", "fake", "--output", str(output))
        self.assertEqual(code, 0)
        expected = [e["case"] for e in self.entries if e["purpose"] not in campaign.ADMITTED_VALIDATIONS for _ in range(e["repeats"])]
        self.assertEqual(calls, expected, "each capture case its listed repeats, in listing order; no ladder, no admitted validation")
        rows = [json.loads(line) for line in output.read_text(encoding="utf-8").splitlines()]
        self.assertTrue(all(r["diagnostic"] is True and r["production_success"] is False for r in rows))
        self.assertEqual([r["repeat"] for r in rows if r["case"] == self.keys[1]], [0, 1])
        table = campaign.render(rows)
        self.assertIn("refused before construction: SANA block", table)
        self.assertIn('"completed_demand_mib": [0.0, 0.0]', table)

    def test_an_optimized_anchor_that_does_not_complete_is_a_defect_and_is_not_repeated(self):
        calls = []
        def fake(session, key, **kw):
            calls.append(key)
            return self.capture_row(key, "capture_refused")
        output = self.root / "defect.jsonl"
        with mock.patch.object(campaign, "DeviceSession") as cls, mock.patch.object(campaign, "invoke", side_effect=fake):
            cls.return_value.blocked = False
            code = self.main("capture", "--purpose", "optimized", "--tier", "q4", "--mode", "text", "--device", "fake", "--output", str(output))
        self.assertEqual(code, 2)
        self.assertEqual(calls, ["fake/fake/q4/512/text/4/s_t256/1"], "a refused case is not repeated")
        row = json.loads(output.read_text(encoding="utf-8"))
        self.assertIn("must complete", row["anchor_defect"])
        self.assertIn("DEFECT", campaign.render([row]))
        for options in (("--case", self.keys[0]), ("--next-edge", "1024")):
            with self.subTest(options=options), self.assertRaises(SystemExit):
                self.main("capture", "--provider", "fake", "--device", "fake", "--output", str(self.root / "x.jsonl"), *options)

    def test_capture_refusal_rows_need_the_typed_quantum_record(self):
        key = self.keys[0]
        good = self.capture_row(key, "capture_refused")
        self.assertEqual(campaign.parse_report("CAMPAIGN_JSON " + json.dumps(good), key)["status"], "capture_refused")
        for broken in (dict(good, refused_quantum=None), dict(good, refusal_kind="no_evidence"), dict(good, diagnostic=False), dict(good, evidence_class="admitted")):
            self.assertEqual(campaign.parse_report("CAMPAIGN_JSON " + json.dumps(broken), key)["status"], "diagnostic_failure")
        artifact = dict(self.success(key), status="admitted_refused", production_success=False, refusal_kind="artifact")
        self.assertEqual(campaign.parse_report("CAMPAIGN_JSON " + json.dumps(artifact), key)["status"], "admitted_refused")

    @staticmethod
    def observed_fixture():
        # Schema fixture ONLY; it is not a native execution/physical-demand proof.
        return {"marks": [{"provider": "curated", "kind": "solver_primary", "fields": {"step": 0, "state_dtype": 3}},
                          {"provider": "curated", "kind": "solve_complete", "fields": {"evaluations": 3}},
                          {"provider": "ios_capture", "kind": "conditioning-roles", "fields": {"reference_count": 0, "mask_count": 0, "control_count": 0}}],
                "counts": {"curated": {"solver_primary": 1, "solve_complete": 1}, "ios_capture": {"conditioning-roles": 1}},
                "total_marks": 3, "dropped_marks": 0, "malformed_marks": 0, "truncated": False}

    def test_anchor_import_preserves_complete_sampling_recipe_and_execution_identity(self):
        recipe = {"version": 1, "sampler": "euler", "scheduler": "normal", "steps": 3,
                  "guidance": 7.0, "guidance_method": "cfg", "cfg_batching": "batched",
                  "strength": None, "reference_strength": None, "use_pid": False, "pid_capture_sigma": None}
        token = "sdxl1-" + json.dumps(recipe, sort_keys=True, separators=(",", ":")).encode("utf-8").hex()
        key = "sdxl/sdxl/q4/512/text/4/base/3/" + token
        rows = []
        for run in ("a", "b"):
            row = dict(self.capture_row(key, "diagnostic_success"), purpose="optimized", run_id=run)
            row["evidence_key"].update(provider="sdxl", sampling_recipe=token, execution="Sequential|DeferredMaterialization|recipe=" + token)
            row["sampling_recipe"] = recipe
            row.update(sampling_coverage=self.observed_fixture(), sampling_metadata_required=True, sampling_metadata_missing=False)
            rows.append(row)
        anchors, problems = campaign.anchor_rows(rows)
        self.assertEqual(problems, [])
        self.assertEqual(len(anchors), 1)
        self.assertEqual(anchors[0]["key"]["sampling_recipe"], token)
        self.assertEqual(anchors[0]["key"]["execution"], rows[0]["evidence_key"]["execution"])
        decoded = json.loads(bytes.fromhex(anchors[0]["key"]["sampling_recipe"].removeprefix("sdxl1-")))
        self.assertEqual(decoded, recipe)
        self.assertEqual([c["run_id"] for c in anchors[0]["outcome"]["completed"]["captures"]], ["a", "b"])

    def test_capture_preserves_both_real_baseline_ledgers_and_refuses_short_forged_queries(self):
        row = self.capture_row(self.keys[1], "diagnostic_success")
        capture = campaign.capture_of(row)
        self.assertEqual(capture["before_request_snapshot"]["current_bytes"], 40)
        self.assertEqual(capture["before_model_load_snapshot"]["current_bytes"], 60)
        for boundary in ("before_request_snapshot", "before_model_load_snapshot", "ending_snapshot"):
            for field, bad in (("query_status", 1), ("returned_words", 83), ("returned_words", 94),
                               ("current_required_words", 1), ("peak_required_words", 1), ("current_bytes", False)):
                broken = json.loads(json.dumps(row))
                broken[boundary][field] = bad
                self.assertIsNone(campaign.capture_of(broken), (boundary, field, bad))
            missing = json.loads(json.dumps(row))
            del missing[boundary]
            self.assertIsNone(campaign.capture_of(missing), boundary)
        mismatched_end = json.loads(json.dumps(row))
        mismatched_end["ending_snapshot"]["lifetime_peak_bytes"] = 160
        self.assertIsNone(campaign.capture_of(mismatched_end), "raw ending kernel peak must agree with imported upper")
        masked = json.loads(json.dumps(row))
        masked["kernel_lifetime_phys_footprint_peak_bytes"] = 200
        for boundary in ("before_request_snapshot", "before_model_load_snapshot", "ending_snapshot"):
            masked[boundary]["lifetime_peak_bytes"] = 200
        masked_capture = campaign.capture_of(masked)
        self.assertIsNotNone(masked_capture)  # Upper retained, never replaced by sampled current150.
        self.assertEqual(masked_capture["footprint_bytes"], 200)
        self.assertEqual(masked_capture["observed_model_current_peak_bytes"], 150)

    def test_quantum_coverage_preserves_actual_context_and_refuses_dangling_or_truncated_transport(self):
        row = self.capture_row(self.keys[1], "diagnostic_success")
        self.assertEqual(campaign.capture_of(row)["quantum_coverage"], row["quantum_coverage"])
        mutations = [lambda q: q["checks"][0].update(context=[99]),
                     lambda q: q["checks"][0].update(count=0),
                     lambda q: q.update(total_checks=2),
                     lambda q: q.update(truncated=True, dropped_checks=1, total_checks=2)]
        for mutate in mutations:
            broken = json.loads(json.dumps(row))
            mutate(broken["quantum_coverage"])
            self.assertIsNone(campaign.capture_of(broken))
        row["quantum_query_gaps"] = 1
        self.assertEqual(campaign.capture_of(row)["quantum_query_gaps"], 1)  # Valid whole-run upper, not phase/class authority.

    def test_role_distinct_receipts_do_not_deduplicate_and_role_aliases_keep_repeat_groups(self):
        # Transport grouping is exact-key only; Python must not invent its own additivity law.
        rows = []
        for references in (0, 1):
            for run, kind in (("a", "pose"), ("b", "canny")):
                mode = ("reference_" if references else "control_") + kind
                key = f"fake/fake/q4/512/{mode}/4/s_t256/1"
                row = dict(self.capture_row(key, "diagnostic_success"), purpose="optimized",
                           run_id=f"{references}-{run}")
                # Existing wire reference_count is aggregate: Control + optional true Reference.
                roles = {"reference_count": references + 1, "mask_count": 0, "control_count": 1}
                row["evidence_key"].update(mode="control", **roles)
                row["sampling_coverage"]["marks"][2]["fields"].update(roles)
                rows.append(row)
        anchors, problems = campaign.anchor_rows(rows)
        self.assertEqual(problems, [])
        self.assertEqual(len(anchors), 2)
        self.assertEqual({anchor["key"]["reference_count"] for anchor in anchors}, {1, 2})
        for anchor in anchors:
            self.assertEqual(anchor["key"]["control_count"], 1)
            captures = anchor["outcome"]["completed"]["captures"]
            self.assertEqual(len(captures), 2)
            self.assertEqual({capture["run_id"] for capture in captures},
                             {f"{anchor['key']['reference_count'] - 1}-a", f"{anchor['key']['reference_count'] - 1}-b"})

    def test_actual_conditioning_roles_are_required_and_match_prepared_carrier_witness(self):
        row = self.capture_row(self.keys[1], "diagnostic_success")
        row["evidence_key"]["sampling_recipe"] = "synthetic-schema-fixture"
        row.update(sampling_coverage=self.observed_fixture(), sampling_metadata_required=True, sampling_metadata_missing=False)
        self.assertIsNotNone(campaign.capture_of(row))
        for field in ("reference_count", "mask_count", "control_count"):
            for bad in (None, False, True, -1, 1 << 32):
                broken = json.loads(json.dumps(row))
                broken["evidence_key"][field] = bad
                self.assertIsNone(campaign.capture_of(broken), (field, bad))
            missing = json.loads(json.dumps(row))
            del missing["evidence_key"][field]
            self.assertIsNone(campaign.capture_of(missing), field)
            forged = json.loads(json.dumps(row))
            forged["sampling_coverage"]["marks"][2]["fields"][field] = 1
            self.assertIsNone(campaign.capture_of(forged), field)

    def test_observed_metadata_is_bounded_preserved_and_quantum_context_is_validated(self):
        row = self.capture_row(self.keys[1], "diagnostic_success")
        row["sampling_coverage"] = self.observed_fixture()
        row["quanta"] = {"q": {"priced_scratch_bytes": 100, "observed_bytes": 80, "exact": True, "preceding_sampling_mark": 0}}
        captured = campaign.capture_of(row)
        self.assertEqual(captured["sampling_coverage"], row["sampling_coverage"])
        self.assertEqual(captured["quanta"]["q"]["preceding_sampling_mark"], 0)
        def oversized(c):
            copies = campaign.SAMPLING_MARKS_MAX // len(c["marks"]) + 1
            c["marks"] *= copies
            c["counts"] = {"curated": {"solver_primary": copies, "solve_complete": copies}, "ios_capture": {"conditioning-roles": copies}}
            c["total_marks"] = len(c["marks"])
        for mutate in (
            lambda c: c["counts"]["curated"].update(solver_primary=2),
            lambda c: c.update(total_marks=c["total_marks"] + 1),
            lambda c: c.update(dropped_marks=1, total_marks=3, truncated=True),
            lambda c: c["marks"][0]["fields"].update(step=True),
            lambda c: c["marks"][0].update(provider="x" * (campaign.SAMPLING_NAME_BYTES_MAX + 1)),
            oversized,
            lambda c: c["marks"][0].update(fields={f"f{i}": i for i in range(campaign.SAMPLING_FIELDS_MAX + 1)}),
        ):
            broken = json.loads(json.dumps(row))
            mutate(broken["sampling_coverage"])
            self.assertTrue(campaign.capture_of(broken) is None, "invalid bounded observation data must be rejected")
        broken = json.loads(json.dumps(row))
        broken["quanta"]["q"]["preceding_sampling_mark"] = 99
        self.assertIsNone(campaign.capture_of(broken))
        required = json.loads(json.dumps(row))
        required["evidence_key"]["sampling_recipe"] = "fixture-token-not-a-class-claim"
        required.update(sampling_metadata_required=True, sampling_metadata_missing=False)
        self.assertIsNotNone(campaign.capture_of(required))
        for missing in (None, {"marks": [], "counts": {}, "total_marks": 0, "dropped_marks": 0, "malformed_marks": 0, "truncated": False}):
            broken = json.loads(json.dumps(required))
            broken["quanta"] = {}
            broken["sampling_coverage"] = missing
            self.assertIsNone(campaign.capture_of(broken), "declared recipe/progress alone is not observed metadata")

    def test_anchor_rows_come_only_from_completed_captures_and_resident_refusals(self):
        opt, res = self.keys[1], self.keys[0]
        def completed(run, **kw):
            return dict(self.capture_row(opt, "diagnostic_success"), purpose="optimized", run_id=run, **kw)
        rows = [completed("a"), completed("b"), completed("c", kernel_lifetime_phys_footprint_peak_bytes=170),
                dict(self.capture_row(res, "capture_refused"), purpose="resident"),
                # Not captures: no telemetry / no baseline below the peak / not a capture.
                completed("d", kernel_lifetime_phys_footprint_peak_bytes=None),
                completed("e", baseline_phys_footprint_bytes=150)]
        # The valid changed ledger peak is recorded by the same ending kernel query.
        rows[2]["ending_snapshot"]["lifetime_peak_bytes"] = 170
        anchors, problems = campaign.anchor_rows(rows)
        self.assertEqual(problems, [])
        # A non-capture row under another key is reported, never turned into an anchor.
        junk, junk_problems = campaign.anchor_rows([dict(self.success(opt), purpose="optimized", evidence_key={"machine": "x"})])
        self.assertEqual(junk, [])
        self.assertEqual(len(junk_problems), 1)
        kinds = {a["anchor_kind"]: a for a in anchors}
        self.assertEqual(set(kinds), {"optimized", "resident"})
        optimized = kinds["optimized"]
        # The serde shape of `evidence::AnchorRow` (deny_unknown_fields on the Rust side).
        self.assertEqual(set(optimized), {"key", "anchor_kind", "selection", "outcome", "evidence_class"})
        captures = optimized["outcome"]["completed"]["captures"]
        self.assertEqual([c["run_id"] for c in captures], ["a", "b", "c"])
        self.assertEqual(set(captures[0]), {"run_id", "build_sha", "steps", "baseline_footprint_bytes", "footprint_bytes", "phase_peaks", "sampling_coverage", "quanta", "mlx_peak_bytes", "min_headroom_mib", "elapsed_seconds",
                         "before_request_snapshot", "before_model_load_snapshot", "ending_snapshot", "observed_current_peak_bytes", "observed_model_current_peak_bytes",
                         "valid_kernel_queries", "valid_model_kernel_queries", "quantum_coverage", "quantum_query_gaps"})
        self.assertEqual((captures[2]["footprint_bytes"], captures[2]["baseline_footprint_bytes"]), (170, 40))
        self.assertEqual(captures[0]["phase_peaks"], {"denoise": {"peak_bytes": 120, "exact": True}, "decode": {"peak_bytes": 150, "exact": False}})
        self.assertEqual(captures[0]["steps"], 1)
        self.assertEqual(optimized["evidence_class"], "device_capture")
        refused = kinds["resident"]["outcome"]["refused"]
        self.assertEqual(refused["refused_quantum"], {"label": "SANA block", "scratch_bytes": 9, "reserve_bytes": 1, "available_bytes": 3})
        # A phase peak outside [baseline, lifetime peak] is dropped, never kept.
        odd = completed("z", phase_peaks={"decode": {"peak_bytes": 10, "exact": True}, "denoise": {"peak_bytes": 999, "exact": True}})
        self.assertEqual(campaign.capture_of(odd)["phase_peaks"], {})
        # Fewer than three completed optimized repeats: no margin basis, a problem, no row.
        anchors, problems = campaign.anchor_rows(rows[:1])
        self.assertEqual(anchors, [])
        self.assertIn("2 needed", problems[0])
        # An optimized anchor that only refused is a DEFECT problem, never refusal evidence.
        anchors, problems = campaign.anchor_rows([dict(self.capture_row(opt, "capture_refused"), purpose="optimized")])
        self.assertEqual(anchors, [])
        self.assertIn("DEFECT", problems[0])
        source = self.root / "capture.jsonl"
        source.write_text("\n".join(json.dumps(r) for r in rows), encoding="utf-8")
        table = Path(campaign.ROOT / campaign.EVIDENCE_TABLE)
        before = table.read_bytes()
        with contextlib.redirect_stdout(io.StringIO()) as printed:
            self.assertEqual(campaign.main(["evidence", "--input", str(source)]), 0)
        self.assertIn("never changes admission", printed.getvalue())
        self.assertIn('"schema": 3', printed.getvalue())
        self.assertEqual(table.read_bytes(), before, "evidence output is printed, never written")

    def test_the_per_formula_device_calibration_table(self):
        mib = 1 << 20
        def row(run, quanta):
            value = dict(self.capture_row(self.keys[1], "diagnostic_success"), run_id=run, kernel_lifetime_phys_footprint_peak_bytes=4000 * mib, baseline_phys_footprint_bytes=100 * mib, quanta=quanta)
            for name in ("before_request_snapshot", "before_model_load_snapshot"):
                value[name].update(current_bytes=100 * mib, lifetime_peak_bytes=100 * mib)
            value["ending_snapshot"].update(current_bytes=150 * mib, lifetime_peak_bytes=4000 * mib)
            value.update(observed_current_peak_bytes=4000 * mib, observed_model_current_peak_bytes=4000 * mib)
            return value
        q = lambda p, o, e: {"priced_scratch_bytes": p * mib, "observed_bytes": o * mib, "exact": e}
        rows = [row("a", {"tile": q(400, 200, True), "block": q(100, 95, True)}),
                row("b", {"tile": q(400, 260, True), "block": q(100, 300, False)}),
                row("c", {"tile": q(400, 999, False), "bad": {"priced_scratch_bytes": True, "observed_bytes": 1, "exact": True}})]
        table = campaign.calibration_table(rows)
        self.assertEqual(set(table), {"tile", "block"}, "malformed observations are dropped")
        self.assertEqual((table["tile"]["priced_scratch_bytes"], table["tile"]["observed_bytes"]), (400 * mib, 260 * mib))
        self.assertEqual((table["tile"]["exact_captures"], table["tile"]["masked_captures"]), (2, 1))
        self.assertTrue(table["tile"]["over_priced_beyond_margin"], "400 priced vs 260 observed > 32 MiB")
        self.assertFalse(table["block"]["over_priced_beyond_margin"], "100 vs 95 is within the margin")
        self.assertEqual(table["tile"]["priced_over_observed"], round(400 / 260, 4))

    def test_steps_and_geometry_validation_reports(self):
        opt = self.keys[1].replace("/text/", "/text/")
        short = [dict(self.capture_row("fake/fake/q4/512/text/4/s_t256/1", "diagnostic_success"), purpose="optimized", run_id=r) for r in "abc"]
        full = dict(self.capture_row("fake/fake/q4/512/text/4/s_t256/0", "diagnostic_success"), purpose="validation_steps",
                    kernel_lifetime_phys_footprint_peak_bytes=160, phase_peaks={"denoise": {"peak_bytes": 130, "exact": True}})
        full["ending_snapshot"]["lifetime_peak_bytes"] = 160
        full["steps"] = 16  # Synthetic provider's actually resolved default; case0 remains the sentinel.
        report = campaign.steps_validation(short + [full])
        self.assertEqual(len(report), 1)
        self.assertEqual((report[0]["full_steps_demand_bytes"], report[0]["short_steps_demand_bytes"]), (120, 110))
        self.assertEqual(report[0]["ratio"], round(120 / 110, 4))
        self.assertEqual(report[0]["phase_ratios"], {"denoise": round(90 / 80, 4)})
        self.assertEqual(campaign.steps_validation([full])[0]["problem"], "no completed optimized anchor of the same case")
        admitted = dict(self.success("fake/fake/q4/1024/text/4/s_t256/1"), purpose="validation_geometry",
                        baseline_phys_footprint_bytes=40, admitted={"demand_bytes": 100, "margin_bytes": 5, "basis": "Derived { from: Optimized }"})
        geometry = campaign.geometry_validation([admitted])
        self.assertEqual(geometry[0]["measured_demand_bytes"], 110)
        self.assertEqual(geometry[0]["relative_error"], round(-10 / 110, 4))
        self.assertEqual(geometry[0]["basis"], "Derived { from: Optimized }")
        output = self.root / "validate.jsonl"
        with mock.patch.object(campaign, "DeviceSession") as cls, mock.patch.object(campaign, "invoke", return_value=admitted) as run:
            cls.return_value.blocked = False
            self.assertEqual(self.main("validate", "--all-cases", "--device", "fake", "--output", str(output)), 0)
            self.assertEqual([c.args[1] for c in run.call_args_list], ["fake/fake/q4/1024/text/4/s_t256/1", "fake/fake/q4/1024/control/4/s_t256/1"])
            self.assertFalse(cls.call_args.args[0].diagnostic, "the geometry validation runs ADMITTED")

    def primitive_row(self):
        reading = {"available_bytes": 6000 << 20, "footprint_bytes": 40 << 20, "kernel_lifetime_peak_bytes": 60 << 20,
                   "mlx_active_bytes": 0, "mlx_cache_bytes": 0, "mlx_peak_bytes": 0}
        row = campaign.failure(campaign.PRIMITIVE_KEY, "primitive characterized")
        row.update(status="diagnostic_success", production_success=False, diagnostic=True, cap_validated=False,
                   evidence_class=campaign.PRIMITIVE_CLASS, model_admission_evidence=False, build_sha="test-sha",
                   run_id="fixture-run", nonce="fixture-run", entitlement="YES",
                   allocation_attempted=True, refused_before_allocation=False,
                   primitive={"required_headroom_bytes": 974_000_000, "scratch_floor_bytes": 437_000_000, "engineering_reserve_bytes": 512 << 20},
                   before=dict(reading), resident=dict(reading, mlx_active_bytes=48 << 20),
                   after=dict(reading, kernel_lifetime_peak_bytes=200 << 20, mlx_peak_bytes=144 << 20),
                   quantum_peak_over_baseline_bytes=160 << 20, unaccounted_by_mlx_peak_bytes=16 << 20)
        return row

    def test_primitive_rows_are_characterization_only_and_fully_measured(self):
        row = self.primitive_row()
        self.assertEqual(campaign.validate_primitive(row, "fixture-run")["status"], "diagnostic_success")
        bad = {
            "class": dict(row, evidence_class="model"),
            "admission": dict(row, model_admission_evidence=True),
            "production": dict(row, production_success=True),
            "cap": dict(row, cap_validated=True),
            "nonce": dict(row, nonce="other-launch"),
            "missing": dict(row, after={k: v for k, v in row["after"].items() if k != "footprint_bytes"}),
            "zero": dict(row, resident=dict(row["resident"], footprint_bytes=0)),
            "bool": dict(row, before=dict(row["before"], mlx_peak_bytes=True)),
            "below": dict(row, before=dict(row["before"], available_bytes=974_000_000 - 1)),
            "requirement": dict(row, primitive={}),
            "not-attempted": dict(row, allocation_attempted=False),
            "refused": dict(row, refused_before_allocation=True),
            "entitlement": dict(row, entitlement=None),
        }
        for name, candidate in bad.items():
            with self.subTest(name=name):
                out = campaign.validate_primitive(candidate, "fixture-run")
                self.assertEqual(out["status"], "diagnostic_failure")
                self.assertIn("primitive row rejected", out["reason"])
        # A device-side refusal stays a failure with its own reason, never upgraded.
        refused = dict(row, status="diagnostic_failure", reason="refused before construction")
        self.assertEqual(campaign.validate_primitive(refused, "fixture-run")["reason"], "refused before construction")

    def test_primitive_action_reuses_device_session_without_catalog_and_forces_diagnostic(self):
        output = self.root / "primitive.jsonl"
        seen = {}
        def fake_invoke(session, key, **kw):
            seen.update(key=key, kw=kw)
            return self.primitive_row()
        with mock.patch.object(campaign, "DeviceSession") as cls, mock.patch.object(campaign, "invoke", side_effect=fake_invoke):
            cls.return_value.run_id = "fixture-run"
            with contextlib.redirect_stdout(io.StringIO()) as printed:
                # No --catalog: consulting the catalog would start cargo and trip the Popen trap.
                self.assertEqual(campaign.main(["primitive", "--device", "fake", "--output", str(output)]), 0)
            cls.return_value.install.assert_called_once()
            self.assertTrue(cls.call_args.args[0].diagnostic)
        self.assertEqual(seen, {"key": campaign.PRIMITIVE_KEY, "kw": {}})
        self.assertIn("not admission evidence", printed.getvalue())
        rows = [json.loads(line) for line in output.read_text(encoding="utf-8").splitlines()]
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["evidence_class"], campaign.PRIMITIVE_CLASS)
        with self.assertRaises(FileExistsError):
            campaign.main(["primitive", "--device", "fake", "--output", str(output)])
        for extra in (("--case", self.keys[0]), ("--all-cases",), ("--tier", "q4")):
            with self.subTest(extra=extra), contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
                campaign.main(["primitive", "--device", "fake", "--output", str(self.root / "x.jsonl"), *extra])

    def test_primitive_nonce_is_checked_against_the_host_launch_not_the_row(self):
        # Row is self-consistent (run_id == nonce) but belongs to a different launch.
        stale = dict(self.primitive_row(), run_id="previous-launch", nonce="previous-launch")
        output = self.root / "stale.jsonl"
        with mock.patch.object(campaign, "DeviceSession") as cls, mock.patch.object(campaign, "invoke", return_value=stale):
            cls.return_value.run_id = "fixture-run"
            with contextlib.redirect_stdout(io.StringIO()):
                self.assertEqual(campaign.main(["primitive", "--device", "fake", "--output", str(output)]), 2)
        row = json.loads(output.read_text(encoding="utf-8"))
        self.assertEqual(row["status"], "diagnostic_failure")
        self.assertIn("nonce", row["reason"])

    def test_primitive_through_the_real_session_launch_path(self):
        session = self.fake_session(self.primitive_row())
        session.args.diagnostic = True
        row = session.run_case(campaign.PRIMITIVE_KEY)
        self.assertEqual(row["status"], "diagnostic_success", row.get("reason"))
        launched = session.device.call_args.args[0]
        variables = json.loads(launched[launched.index("--environment-variables") + 1])
        self.assertEqual((variables["IOS_CAMPAIGN_CASE"], variables["IOS_CAMPAIGN_DIAGNOSTIC"]), (campaign.PRIMITIVE_KEY, "1"))
        self.assertEqual(session.run_id, "fixture-run")
        self.assertEqual(campaign.validate_primitive(row, session.run_id)["status"], "diagnostic_success")


if __name__ == "__main__":
    unittest.main()
