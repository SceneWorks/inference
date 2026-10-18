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
import unittest
from unittest import mock

MODULE = Path(__file__).resolve().parents[1] / "ios/campaign.py"
spec = importlib.util.spec_from_file_location("ios_campaign", MODULE)
campaign = importlib.util.module_from_spec(spec)
spec.loader.exec_module(campaign)


class CampaignTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.keys = [f"fake/fake/{tier}/{edge}/{mode}/{rung}" for tier in campaign.TIERS for edge in (512, 768, 1024) for mode in ("text", "reference", "restore", "depth") for rung in range(5)]
        self.catalog = self.root / "catalog.json"
        self.catalog.write_text(json.dumps({"schema": 1, "cases": self.keys}), encoding="utf-8")
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
                local.write_text("campaign-" + row["case"].replace("/", "-") + "-0.png\n", encoding="utf-8")
            else:
                local.write_bytes(b"fixture PNG")
            return subprocess.CompletedProcess([], 0, "", "")
        session.copy = mock.Mock(side_effect=copy)
        return session

    def test_public_cli_listing_is_pure_and_rung_zero_is_selected(self):
        self.assertEqual(self.main("list", "--tier", "q8", "--edge", "768", "--mode", "restore", "--rung", "4"), 0)
        args = campaign.parser().parse_args(["list", "--rung", "0"])
        self.assertFalse(campaign.selected(args, "fake/fake/q4/512/text/1"))
        self.assertTrue(campaign.selected(args, "fake/fake/q4/512/text/0"))

    def test_invalid_finite_budgets_and_cases(self):
        for options in (("--case", "p/../q4/512/text/0"), ("--timeout-seconds", "0"), ("--max-cases", "20001"), ("--tier", "q5"), ("--edge", "999999"), ("--rung", "5")):
            with self.subTest(options=options), contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
                campaign.parser().parse_args(["run", *options])
        with self.assertRaises(SystemExit):
            self.main("run", "--device", "fake", "--output", str(self.root / "not-written"))
        self.assertFalse((self.root / "not-written").exists())

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

    def test_table_has_all_metrics_and_no_diagnostic_largest_edge(self):
        row = dict(self.success(), status="diagnostic_success", production_success=False, diagnostic=True)
        table = campaign.render([row])
        self.assertIn("diagnostics excluded): {}", table)
        self.assertIn("Active MiB | Cache MiB | Combined MiB", table)
        row.update(status="admitted_refused", refusal_only=True, refusal_kind="no_evidence")
        self.assertIn("Capacity-refused next-up probes: []", campaign.render([row]))
        row.update(refusal_kind="capacity", artifact_identity={"layout_sha256": "a" * 64}, evidence_revision="fixture-capture", selected_parameters={"decode_tile_edge": 192})
        self.assertIn(self.keys[0], campaign.render([row]).split("Capacity-refused next-up probes:")[-1])
        row["evidence_revision"] = "no-device-capture"
        self.assertIn("Capacity-refused next-up probes: []", campaign.render([row]))

    def capture_row(self, key, status):
        row = dict(self.success(key), status=status, production_success=False, diagnostic=True, evidence_class="capture",
                   evidence_key={"machine": "iPhone17,2", "provider": "fake"}, run_id="fixture-run",
                   baseline_phys_footprint_bytes=40)
        if status == "capture_refused":
            row.update(refusal_kind="capacity", refused_quantum={"label": "SANA block", "scratch_bytes": 9, "reserve_bytes": 1, "available_bytes": 3})
        return row

    def test_capture_walks_each_ladder_upward_and_stops_at_the_first_refusal(self):
        outcome = {512: "diagnostic_success", 768: "capture_refused", 1024: "diagnostic_success"}
        calls = []
        def fake(session, key, **kw):
            calls.append(key)
            return self.capture_row(key, outcome[int(key.split("/")[3])])
        output = self.root / "capture.jsonl"
        with mock.patch.object(campaign, "DeviceSession") as cls, mock.patch.object(campaign, "invoke", side_effect=fake):
            cls.return_value.blocked = False
            code = self.main("capture", "--provider", "fake", "--tier", "q4", "--mode", "text", "--device", "fake", "--output", str(output))
        self.assertEqual(code, 0)
        ladders = {}
        for key in calls:
            fields = key.split("/")
            ladders.setdefault(fields[5], []).append(int(fields[3]))
        self.assertEqual(set(ladders), {"0", "1", "2", "3", "4"})
        for edges in ladders.values():
            self.assertEqual(edges, [512, 768], "never past the refused size, never a retry")
        rows = [json.loads(line) for line in output.read_text(encoding="utf-8").splitlines()]
        self.assertTrue(all(r["diagnostic"] is True and r["production_success"] is False for r in rows))
        table = campaign.render(rows)
        self.assertIn('"largest_completed_edge": 512', table)
        self.assertIn('"refused_before_jetsam_edge": 768', table)
        self.assertIn("refused before construction: SANA block", table)
        self.assertIn("diagnostics excluded): {}", table)

    def test_capture_rejects_edge_selectors_and_failure_stops_the_ladder(self):
        for options in (("--edge", "512"), ("--case", self.keys[0]), ("--next-edge", "1024")):
            with self.subTest(options=options), self.assertRaises(SystemExit):
                self.main("capture", "--provider", "fake", "--device", "fake", "--output", str(self.root / "x.jsonl"), *options)
        calls = []
        def fake(session, key, **kw):
            calls.append(key)
            return campaign.failure(key, "load failed")
        with mock.patch.object(campaign, "DeviceSession") as cls, mock.patch.object(campaign, "invoke", side_effect=fake):
            cls.return_value.blocked = False
            code = self.main("capture", "--provider", "fake", "--tier", "q4", "--mode", "text", "--rung", "0", "--device", "fake", "--output", str(self.root / "fail.jsonl"))
        self.assertEqual(code, 2)
        self.assertEqual(len(calls), 1)

    def test_capture_refusal_rows_need_the_typed_quantum_record(self):
        key = self.keys[0]
        good = self.capture_row(key, "capture_refused")
        self.assertEqual(campaign.parse_report("CAMPAIGN_JSON " + json.dumps(good), key)["status"], "capture_refused")
        for broken in (dict(good, refused_quantum=None), dict(good, refusal_kind="no_evidence"), dict(good, diagnostic=False), dict(good, evidence_class="admitted")):
            self.assertEqual(campaign.parse_report("CAMPAIGN_JSON " + json.dumps(broken), key)["status"], "diagnostic_failure")
        artifact = dict(self.success(key), status="admitted_refused", production_success=False, refusal_kind="artifact")
        self.assertEqual(campaign.parse_report("CAMPAIGN_JSON " + json.dumps(artifact), key)["status"], "admitted_refused")

    def test_evidence_rows_come_only_from_completed_captures_with_telemetry(self):
        key = self.keys[0]
        rows = [self.capture_row(key, "diagnostic_success"), self.capture_row(key, "capture_refused"),
                dict(self.capture_row(key, "diagnostic_success"), kernel_lifetime_phys_footprint_peak_bytes=None),
                dict(self.capture_row(key, "diagnostic_success"), baseline_phys_footprint_bytes=None),
                dict(self.capture_row(key, "diagnostic_success"), baseline_phys_footprint_bytes=150),
                dict(self.success(key), evidence_key={"machine": "x"})]
        out = campaign.evidence_rows(rows)
        self.assertEqual(len(out), 1, "no row without a capture baseline below the peak")
        self.assertEqual(set(out[0]), {"key", "footprint_bytes", "baseline_footprint_bytes", "mlx_peak_bytes", "min_headroom_mib", "elapsed_seconds", "evidence_class", "run_id", "build_sha"})
        self.assertEqual((out[0]["footprint_bytes"], out[0]["baseline_footprint_bytes"]), (150, 40))
        self.assertEqual(out[0]["evidence_class"], "device_capture")
        source = self.root / "capture.jsonl"
        source.write_text("\n".join(json.dumps(r) for r in rows), encoding="utf-8")
        table = Path(campaign.ROOT / campaign.EVIDENCE_TABLE)
        before = table.read_bytes()
        with contextlib.redirect_stdout(io.StringIO()) as printed:
            self.assertEqual(campaign.main(["evidence", "--input", str(source)]), 0)
        self.assertIn("never changes admission", printed.getvalue())
        self.assertEqual(table.read_bytes(), before, "evidence output is printed, never written")

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
