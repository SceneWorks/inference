"""CPU-only regressions for the one-shot remote app Metal proof controller."""

from __future__ import annotations

import hashlib
import importlib.util
import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[2]
spec = importlib.util.spec_from_file_location("yue2_app_metal", ROOT / "scripts/ci/yue2_app_metal.py")
app = importlib.util.module_from_spec(spec)
assert spec.loader is not None
spec.loader.exec_module(app)
sys.path.insert(0, str(ROOT / "scripts/ci"))
import yue2_app_metal_resume as resume


def chain(events: list[dict]) -> str:
    prior = "0" * 64
    lines = []
    for sequence, payload in enumerate(events, 1):
        item = {**payload, "eventSequence": sequence, "previousEventHash": prior}
        prior = hashlib.sha256(json.dumps(item, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
        lines.append(json.dumps({**item, "eventHash": prior}))
    return "\n".join(lines) + "\n"


class AppMetalProofTests(unittest.TestCase):
    def test_resume_binds_original_run_and_refuses_started_profile(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            tag = resume.ORIGINAL_RUN
            state = root / f"yue2-app-metal-state-{tag}"
            out = root / f"yue2-app-metal-out-{tag}"
            target = root / f"yue2-app-metal-target-{tag}"
            (state / "app-data").mkdir(parents=True)
            (state / "hf-home").mkdir()
            (target / "release").mkdir(parents=True)
            (target / "release" / "sceneworks-rust-api").write_text("binary")
            evidence = out / "acceptance" / "evidence"
            evidence.mkdir(parents=True)
            summary = evidence / "summary.json"
            summary.write_text("original summary")
            chain = out / "watchdog-acceptance.jsonl"
            chain.write_text("original chain")
            (out / "identity.json").write_text(json.dumps({
                "app_sha": app.APP_SHA, "engine_sha": app.ENGINE_SHA,
                "runner": "nax-macos-2", "state": str(state), "out": str(out),
            }))
            preflight = root / "preflight.json"
            preflight.write_text(json.dumps({"runner": "nax-macos-2", "admitted": True}))
            with patch.object(resume, "SUMMARY_SHA256", resume.file_sha256(summary)), \
                 patch.object(resume, "CHAIN_SHA256", resume.file_sha256(chain)):
                self.assertEqual(resume.verify_original(state, out, target, preflight)["chain"], chain)
                (out / "profile").mkdir()
                with self.assertRaisesRegex(ValueError, "already started"):
                    resume.verify_original(state, out, target, preflight)
                (out / "profile").rmdir()
                summary.write_text("different run")
                with self.assertRaisesRegex(ValueError, "differs from published"):
                    resume.verify_original(state, out, target, preflight)
                summary.write_text("original summary")
                with self.assertRaisesRegex(ValueError, "same original run"):
                    resume.verify_original(state, out, root / "other-target", preflight)

    def test_staged_ffmpeg_is_explicit_and_probed_without_path_lookup(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            binary = Path(temp) / "ffmpeg"
            binary.write_text("#!/bin/sh\nprintf 'ffmpeg version staged-test\\n'\n")
            binary.chmod(0o755)
            self.assertEqual(app.probe_ffmpeg(str(binary)), (str(binary), "ffmpeg version staged-test"))
            with self.assertRaisesRegex(ValueError, "YUE2_FFMPEG_BIN"):
                app.probe_ffmpeg(None)
            with self.assertRaisesRegex(ValueError, "absolute executable"):
                app.probe_ffmpeg("ffmpeg")
            binary.write_text("#!/bin/sh\nprintf 'wrong tool\\n'\n")
            with self.assertRaisesRegex(ValueError, "version probe"):
                app.probe_ffmpeg(str(binary))

    def test_guard_ceiling_scales_with_host_and_caps_at_incident(self) -> None:
        self.assertEqual(app.guard_ceiling(137_438_953_472), 94_822_600_832)
        self.assertEqual(app.guard_ceiling(64 << 30), (62 << 30))

    def test_acceptance_requires_correct_skip_and_complete_hash_chain(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            summary = root / "summary.json"
            events = root / "watchdog.jsonl"
            data = {
                "verdict": "incomplete", "fatal": None,
                "counts": {"passed": 26, "failed": 0, "blocked": 0, "skipped": 1},
                "missing": [], "cases": [{"status": "passed"}] * 26 + [{"status": "skipped", "caseId": "worker-kill-resume", "reason": "signal-kill of a Metal process is unsafe"}],
                "identity": {"sceneworksRevision": app.APP_SHA, "inferencePin": app.ENGINE_SHA, "dirty": False},
                "platform": "metal",
            }
            summary.write_text(json.dumps(data))
            digest = "a" * 64
            completed = [
                {"event": "started"},
                {"event": "sample", "phase": "runtime"},
                {"event": "child_completion_requested", "evidenceSha256": digest},
                {"event": "sample", "phase": "completion_before_release"},
                {"event": "child_completion_measured", "evidenceSha256": digest, "rootIdentity": {"pid": 1}},
                {"event": "child_completed"},
            ]
            events.write_text(chain(completed))
            self.assertEqual(app.verify_acceptance(summary, events, 1, digest)["watchdog_exit"], 1)
            with self.assertRaisesRegex(ValueError, "exited"):
                app.verify_acceptance(summary, events, 97, digest)
            data["counts"]["failed"] = 1
            summary.write_text(json.dumps(data))
            with self.assertRaisesRegex(ValueError, "case verdict"):
                app.verify_acceptance(summary, events, 1, digest)
            data["counts"]["failed"] = 0
            summary.write_text(json.dumps(data))
            events.write_text(chain([{"event": "started"}, {"event": "sample"}, {"event": "child_completion_requested"}, {"event": "hard_stop"}]))
            with self.assertRaisesRegex(ValueError, "hard-stopped"):
                app.verify_acceptance(summary, events, 1, digest)
            completed[3]["phase"] = "runtime"
            events.write_text(chain(completed))
            with self.assertRaisesRegex(ValueError, "final group/host sample"):
                app.verify_acceptance(summary, events, 1, digest)
            completed[3]["phase"] = "completion_before_release"
            completed[4]["evidenceSha256"] = "b" * 64
            events.write_text(chain(completed))
            with self.assertRaisesRegex(ValueError, "completion digest"):
                app.verify_acceptance(summary, events, 1, digest)

    def test_profile_requires_load_sample_and_artifact_excludes_audio(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            profile = root / "record.json"
            data = {"caseId": app.CASE, "backend": "metal", "outcome": {"status": "completed"},
                    "measured": {"peakBytes": 100, "stages": {name: {"samples": 1, "peakBytes": 100} for name in app.STAGES}}}
            profile.write_text(json.dumps(data))
            self.assertEqual(app.verify_profile(profile)["stage_samples"]["load"], 1)
            data["measured"]["stages"]["load"]["samples"] = 0
            profile.write_text(json.dumps(data))
            with self.assertRaisesRegex(ValueError, "missing a measured stage"):
                app.verify_profile(profile)
            out = root / "out"
            evidence = root / "evidence"
            (out / "acceptance" / "evidence").mkdir(parents=True)
            (out / "acceptance" / "evidence" / "summary.json").write_text("{}")
            (out / "acceptance" / "evidence" / "song.wav").write_bytes(b"audio")
            evidence.mkdir()
            app.copy_receipts(out, evidence)
            self.assertTrue((evidence / "acceptance" / "summary.json").is_file())
            self.assertFalse((evidence / "acceptance" / "song.wav").exists())


if __name__ == "__main__":
    unittest.main()
