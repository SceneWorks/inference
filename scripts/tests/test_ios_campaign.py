"""CPU-only campaign CLI tests; no cargo, devicectl, simulator, weights or accelerator."""
import contextlib
import importlib.util
import io
import json
from pathlib import Path
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

    def main(self, *args):
        with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
            return campaign.main([*args, "--catalog", str(self.catalog)])

    def test_public_cli_lists_without_invoking_device_or_compiler(self):
        with mock.patch.object(campaign.subprocess, "run", side_effect=AssertionError("must be pure")):
            self.assertEqual(self.main("list", "--tier", "q8", "--edge", "768", "--mode", "restore", "--rung", "4"), 0)
        args = campaign.parser().parse_args(["list", "--rung", "0"])
        self.assertFalse(campaign.selected(args, "fake/fake/q4/512/text/1"))
        self.assertTrue(campaign.selected(args, "fake/fake/q4/512/text/0"))

    def test_public_cli_rejects_invalid_and_unbounded_inputs(self):
        for options in (("--case", "p/../q4/512/text/0"), ("--timeout-seconds", "0"), ("--max-cases", "20001"), ("--tier", "q5"), ("--edge", "999999"), ("--rung", "5")):
            with self.subTest(options=options), contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
                campaign.parser().parse_args(["run", *options])
        with self.assertRaises(SystemExit):
            self.main("run", "--device", "fake", "--output", str(self.root / "not-written"))
        self.assertFalse((self.root / "not-written").exists())

    def test_default_admission_refusal_is_not_a_pass_and_has_all_status_rows(self):
        output = self.root / "evidence.jsonl"
        refusal = lambda args, key: dict(campaign.failure(key, "no device footprint"), status="admitted_refused")
        with mock.patch.object(campaign, "invoke", side_effect=refusal) as invoke:
            rc = self.main("run", "--all-cases", "--device", "fake", "--output", str(output), "--max-cases", "2")
        self.assertEqual(rc, 2)
        self.assertEqual(invoke.call_count, 2)
        rows = [json.loads(line) for line in output.read_text(encoding="utf-8").splitlines()]
        self.assertEqual([r["case"] for r in rows], self.keys)
        self.assertEqual(rows[0]["status"], "admitted_refused")
        self.assertTrue(all(r["status"] == "skipped" for r in rows[2:]))
        self.assertTrue(all(r["production_success"] is False for r in rows))
        with self.assertRaises(FileExistsError):
            self.main("run", "--all-cases", "--device", "fake", "--output", str(output))

    def test_report_identity_and_diagnostics_fail_closed(self):
        key = self.keys[0]
        self.assertEqual(campaign.parse_report("SMOKE: PASS\n", key)["status"], "diagnostic_failure")
        for status in campaign.STATUSES:
            row = dict(campaign.failure(key, "fixture"), status=status, production_success=status == "successful", elapsed_seconds=1.25, mlx_active_peak_bytes=10, mlx_cache_peak_bytes=20, mlx_active_cache_peak_bytes=30, mlx_peak_bytes=15, phys_footprint_peak_bytes=123, min_headroom_mib=456)
            encoded = "CAMPAIGN_JSON " + json.dumps(row)
            self.assertEqual(campaign.parse_report(encoded, key)["status"], status)
            self.assertEqual(campaign.parse_report(encoded, "wrong/wrong/q4/512/text/0")["status"], "diagnostic_failure")
        row.update(status="successful", production_success=True, min_headroom_mib=float("nan"))
        self.assertEqual(campaign.parse_report("CAMPAIGN_JSON " + json.dumps(row), key)["status"], "diagnostic_failure")
        row.update(status="diagnostic_success", production_success=True)
        self.assertEqual(campaign.parse_report("CAMPAIGN_JSON " + json.dumps(row), key)["status"], "diagnostic_failure")
        row.update(status="successful", diagnostic=True)
        self.assertEqual(campaign.parse_report("CAMPAIGN_JSON " + json.dumps(row), key)["status"], "diagnostic_failure")

    def test_one_cold_process_per_case_drops_legacy_bypasses(self):
        key = self.keys[0]
        args = campaign.parser().parse_args(["run", "--device", "explicit-device", "--case", key])
        report = dict(campaign.failure(key, "no capture"), status="admitted_refused")
        with mock.patch.dict(campaign.os.environ, {"IOS_SMOKE_UNADMITTED": "1", "IOS_SMOKE_SANA_CAPTURE": "1"}), mock.patch.object(campaign.subprocess, "run", return_value=subprocess.CompletedProcess([], 1, "CAMPAIGN_JSON " + json.dumps(report))) as run:
            row = campaign.invoke(args, key, refusal_only=True)
        self.assertEqual(row["status"], "admitted_refused")
        self.assertIn("--campaign-case", run.call_args.args[0])
        self.assertEqual(run.call_args.kwargs["env"]["IOS_CAMPAIGN_REFUSAL_ONLY"], "1")
        self.assertNotIn("IOS_SMOKE_UNADMITTED", run.call_args.kwargs["env"])
        self.assertEqual(run.call_args.kwargs["timeout"], 1800)

    def test_next_up_is_refusal_only_and_a_kill_is_not_refusal(self):
        key = self.keys[0]
        output = self.root / "next-up.jsonl"
        def invoke(args, current, refusal_only=False):
            if refusal_only:
                return dict(campaign.failure(current, "no capture"), status="admitted_refused")
            return dict(campaign.failure(current, "render"), status="successful", production_success=True)
        with mock.patch.object(campaign, "invoke", side_effect=invoke) as run:
            self.main("run", "--case", key, "--next-edge", "768", "--device", "fake", "--output", str(output))
        self.assertEqual(run.call_count, 2)
        self.assertEqual(run.call_args.kwargs, {"refusal_only": True})
        args = campaign.parser().parse_args(["run", "--device", "fake", "--case", key])
        with mock.patch.object(campaign.subprocess, "run", side_effect=subprocess.TimeoutExpired("fake", 1)):
            self.assertEqual(campaign.invoke(args, key, True)["status"], "diagnostic_failure")

    def test_table_excludes_diagnostics_from_largest_completed(self):
        row = dict(campaign.failure(self.keys[0], "render"), status="diagnostic_success")
        table = campaign.render([row])
        self.assertIn("diagnostics excluded): {}", table)
        self.assertIn("unavailable", table)
        row.update(status="successful", production_success=True)
        self.assertIn('"fake/fake/q4/text/0": 512', campaign.render([row]))

    def test_push_campaign_validation_precedes_any_device_call(self):
        script = MODULE.parent / "push_model.sh"
        for flags in (("--campaign-route", "../escape", "--tier", "q4"), ("--campaign-route", "iris_3b", "--tier", "q5")):
            result = subprocess.run(["bash", str(script), str(self.root), *flags], capture_output=True, text=True, encoding="utf-8", check=False)
            self.assertEqual(result.returncode, 2)
            self.assertNotIn("device ", result.stdout)


if __name__ == "__main__":
    unittest.main()
