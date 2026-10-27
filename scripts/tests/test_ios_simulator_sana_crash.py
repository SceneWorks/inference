"""CI_IMPACT: pure bounded crash-report matching; no simulator, model, or remote runner."""
from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import tempfile
import unittest

from scripts.ci.ios_simulator_sana_crash import recover, utc


RUN = "40576563280"
ATTEMPT = "1"
UDID = "AA0EDD4A-DE57-4E70-B22A-F4D9597C42D7"
BUNDLE = f"com.michaeltrefry.SceneWorksSmokeSim{RUN}a{ATTEMPT}"
START = utc("2026-10-27T20:22:00Z")
END = utc("2026-10-27T20:23:00Z")


def ips(*, bundle: str = BUNDLE, udid: str = UDID,
        captured: str = "2026-10-27 16:22:20.1234 -0400") -> bytes:
    header = {"app_name": "SceneWorksSmoke", "bundleID": bundle,
              "timestamp": captured}
    body = {"procName": "SceneWorksSmoke", "pid": 1234, "captureTime": captured,
            "bundleInfo": {"CFBundleIdentifier": bundle},
            "procPath": f"/Users/runner/Library/Developer/CoreSimulator/Devices/{udid}/data/Containers/Bundle/Application/X/SceneWorksSmoke.app/SceneWorksSmoke",
            "termination": {"namespace": "SIGNAL", "code": 9}}
    return (json.dumps(header) + "\n" + json.dumps(body)).encode()


class SimulatorCrashRecoveryTests(unittest.TestCase):
    def test_one_exact_prior_app_crash_is_retained_as_diagnostic_only(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            reports = root / "reports"
            reports.mkdir()
            data = ips()
            (reports / "SceneWorksSmoke-2026-10-27.ips").write_bytes(data)
            for suffix, other in (("other-bundle", ips(bundle="com.example.Other")),
                                  ("other-device", ips(udid="11111111-1111-1111-1111-111111111111")),
                                  ("other-time", ips(captured="2026-10-27 16:30:00.0000 -0400"))):
                (reports / f"SceneWorksSmoke-{suffix}.ips").write_bytes(other)
            output = root / "result"
            self.assertTrue(recover(reports, output, RUN, ATTEMPT, UDID, START, END))
            receipt = json.loads((output / "recovery.json").read_text(encoding="utf-8"))
            self.assertFalse(receipt["accepted"])
            self.assertEqual(receipt["matching_crash_count"], 1)
            self.assertEqual(receipt["crash"]["termination"]["code"], 9)
            self.assertEqual(receipt["crash_sha256"], hashlib.sha256(data).hexdigest())
            self.assertEqual((output / "matching-crash.ips").read_bytes(), data)

    def test_ambiguous_or_oversized_reports_cannot_be_certified_or_copied(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            reports = root / "reports"
            reports.mkdir()
            (reports / "SceneWorksSmoke-one.ips").write_bytes(ips())
            (reports / "SceneWorksSmoke-two.ips").write_bytes(ips())
            output = root / "ambiguous"
            self.assertFalse(recover(reports, output, RUN, ATTEMPT, UDID, START, END))
            self.assertFalse((output / "matching-crash.ips").exists())
            self.assertEqual(json.loads((output / "recovery.json").read_text(encoding="utf-8"))[
                "status"], "ambiguous-matches")
            for path in reports.iterdir():
                path.unlink()
            (reports / "SceneWorksSmoke-large.ips").write_bytes(b"x" * (4 * 1024 * 1024 + 1))
            oversized = root / "oversized"
            self.assertFalse(recover(reports, oversized, RUN, ATTEMPT, UDID, START, END))
            self.assertFalse((oversized / "matching-crash.ips").exists())

    def test_recovery_requires_exact_identity_and_narrow_utc_window(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for run, attempt, udid, end in (("0", ATTEMPT, UDID, END),
                                            (RUN, "٠١", UDID, END),
                                            (RUN, ATTEMPT, "not-a-udid", END),
                                            (RUN, ATTEMPT, UDID, datetime(2026, 10, 27, 20, 29,
                                                                         tzinfo=timezone.utc))):
                with self.assertRaises(ValueError):
                    recover(root, root / "out", run, attempt, udid, START, end)
            self.assertFalse((root / "out").exists())


if __name__ == "__main__":
    unittest.main()
