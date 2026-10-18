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
                local.write_text("campaign-" + row["case"].replace("/", "-") + "-0.png\n", encoding="utf-8")
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
        for options in (("--case", "p/../q4/512/text/0/base/1"), ("--case", "p/p/q4/512/text/0/base"), ("--case", "p/p/q4/512/text/0/t5.1/1"), ("--case", "p/p/q4/512/text/0/base/101"), ("--timeout-seconds", "0"), ("--timeout-seconds", "20001"), ("--max-cases", "0"), ("--tier", "q5"), ("--edge", "999999"), ("--rung", "5"), ("--steps", "101"), ("--purpose", "ladder")):
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
                   evidence_key={"machine": "iPhone17,2", "provider": "fake", "execution": key.split("/")[5]}, run_id="fixture-run",
                   baseline_phys_footprint_bytes=40, steps=int(key.split("/")[7]),
                   anchor_selection={"rung": int(key.split("/")[5])},
                   phase_peaks={"denoise": {"peak_bytes": 120, "exact": True}, "decode": {"peak_bytes": 150, "exact": False}})
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

    def test_anchor_rows_come_only_from_completed_captures_and_resident_refusals(self):
        opt, res = self.keys[1], self.keys[0]
        def completed(run, **kw):
            return dict(self.capture_row(opt, "diagnostic_success"), purpose="optimized", run_id=run, **kw)
        rows = [completed("a"), completed("b"), completed("c", kernel_lifetime_phys_footprint_peak_bytes=170),
                dict(self.capture_row(res, "capture_refused"), purpose="resident"),
                # Not captures: no telemetry / no baseline below the peak / not a capture.
                completed("d", kernel_lifetime_phys_footprint_peak_bytes=None),
                completed("e", baseline_phys_footprint_bytes=150)]
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
        self.assertEqual(set(captures[0]), {"run_id", "build_sha", "steps", "baseline_footprint_bytes", "footprint_bytes", "phase_peaks", "quanta", "mlx_peak_bytes", "min_headroom_mib", "elapsed_seconds"})
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
            return dict(self.capture_row(self.keys[1], "diagnostic_success"), run_id=run, kernel_lifetime_phys_footprint_peak_bytes=4000 * mib, baseline_phys_footprint_bytes=100 * mib, quanta=quanta)
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
