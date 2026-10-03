"""CPU-only parent artifact and stage-2 selector guard regressions."""
import hashlib
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch
import urllib.request
import zipfile

CI = Path(__file__).resolve().parents[1] / "ci"
sys.path.insert(0, str(CI))
import yue2_native_convt_math_parent as parent
import yue2_bf16_tile_diagnostic as diagnostic


def sha(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


class NativeMathParent(unittest.TestCase):
    def test_https_redirect_credentials_stay_on_original_origin_only(self):
        handler = parent.ArtifactRedirects()
        request = urllib.request.Request(
            "https://api.github.com/repos/SceneWorks/inference/actions/artifacts/11259450025/zip",
            headers={"Authorization": "Bearer SYNTHETIC_ONLY", "Accept": "application/zip"})
        same = handler.redirect_request(request, None, 302, "Found", {},
                                        "https://API.GITHUB.COM:443/next")
        self.assertEqual(same.get_header("Authorization"), "Bearer SYNTHETIC_ONLY")
        cross = handler.redirect_request(request, None, 302, "Found", {},
                                         "https://artifact-host.example/signed.zip")
        self.assertIsNone(cross.get_header("Authorization"))
        self.assertEqual(cross.get_header("Accept"), "application/zip")
        chained = handler.redirect_request(cross, None, 302, "Found", {},
                                           "https://artifact-host.example/another-signed.zip")
        self.assertIsNone(chained.get_header("Authorization"))
        back = handler.redirect_request(chained, None, 302, "Found", {},
                                        "https://api.github.com/back")
        self.assertIsNone(back.get_header("Authorization"))

    def test_redirect_refuses_http_downgrade_and_userinfo(self):
        handler = parent.ArtifactRedirects()
        request = urllib.request.Request("https://api.github.com/fixture",
                                         headers={"Authorization": "Bearer SYNTHETIC_ONLY"})
        for destination in ("http://artifact-host.example/zip",
                            "https://user:pass@artifact-host.example/zip",
                            "//artifact-host.example/zip"):
            with self.subTest(destination=destination), self.assertRaisesRegex(
                    ValueError, "HTTPS URL without userinfo"):
                handler.redirect_request(request, None, 302, "Found", {}, destination)
        insecure_source = urllib.request.Request("http://api.github.com/fixture",
                                                 headers={"Authorization": "Bearer SYNTHETIC_ONLY"})
        with self.assertRaisesRegex(ValueError, "HTTPS URL without userinfo"):
            handler.redirect_request(insecure_source, None, 302, "Found", {},
                                     "https://artifact-host.example/zip")

    def test_native_math_raw_shape_hash_and_parent_geometry_mutations(self):
        with tempfile.TemporaryDirectory() as directory:
            data = Path(directory)
            raw = b"\x00\x3f\x80\x3f"
            (data / "a.bf16le").write_bytes(raw)
            row = {"file": "a.bf16le", "sha256": sha(raw), "bytes": 4,
                   "shape": [1, 1, 2], "dtype": "BF16", "layout": "bct_bf16le", "origin": 0}
            diagnostic.native_math_array(data, row, set(), row)
            for key, value in (("shape", [1, 1, 3]), ("bytes", 2),
                               ("layout", "blck_bf16le"), ("origin", 96)):
                with self.subTest(key=key):
                    changed = dict(row)
                    changed[key] = value
                    with self.assertRaises(RuntimeError):
                        diagnostic.native_math_array(data, changed, set(), row)

    def test_same_handle_mode_sequence_and_restore_mutants(self):
        actions = ["before_default_sync", "before_default_get", "after_default_sync", "after_default_get",
                   "before_flag_sync", "before_flag_get", "set_disallow", "read_disallow",
                   "before_restore_sync", "restore_default", "after_restore_sync", "read_restored"]
        modes = [None, 0, None, 0, None, 0, 16, 16, None, 0, None, 0]
        original = [{"action": action, "rawMode": mode,
                     "status": "success" if action.endswith("sync") else "CUBLAS_STATUS_SUCCESS",
                     "handleAddress": "handle1", "streamObjectAddress": "stream1", "thread": "thread1",
                     "deviceOrdinal": 0, "bf16ReducedPrecisionAtomic": False}
                    for action, mode in zip(actions, modes)]
        diagnostic.verify_native_math_events(original, "handle1", "stream1", "thread1")
        for index, key, value in ((7, "rawMode", 0), (11, "rawMode", 16),
                                  (6, "handleAddress", "handle2"),
                                  (8, "streamObjectAddress", "stream2"),
                                  (5, "bf16ReducedPrecisionAtomic", True),
                                  (10, "status", "failure")):
            with self.subTest(index=index, key=key):
                changed = [dict(event) for event in original]
                changed[index][key] = value
                with self.assertRaisesRegex(RuntimeError, "same-handle"):
                    diagnostic.verify_native_math_events(changed, "handle1", "stream1", "thread1")

    def test_exact_parent_zip_and_extracted_raw_mutations(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            archive = root / "parent.zip"
            destination = root / "extracted"
            anchor_path = root / "anchor.json"
            audit_path = root / "audit.json"
            report = {"schemaVersion": 5, "selector": "native_convt_columns", "engineSha": "a" * 40,
                      "waveformParity": True, "earliestBf16Stage": 2,
                      "nativeColumns": {"status": "collected"},
                      "historicalStage2": {"matchesHistorical": True}}
            report_bytes = json.dumps(report).encode()
            raw = b"\x00\x3f\x80\x3f"
            with zipfile.ZipFile(archive, "w") as output:
                output.writestr("data/report.json", report_bytes)
                output.writestr("data/one.bf16le", raw)
            audit = {"diagnosticCollectionVerified": True, "rawCheckpointsPerDtype": 198,
                     "earliestBf16Stage": 2,"waveform": {"originalWaveformHashParity": True},
                     "historicalStage2": {"currentRecordsByteIdentical": True}}
            audit_path.write_text(json.dumps(audit), encoding="utf-8")
            anchor = {"metricsZipSha256": parent.digest(archive), "reportSha256": sha(report_bytes),
                      "independentAuditSha256": parent.digest(audit_path), "engineSha": "a" * 40,
                      "records": {"one": {"file": "one.bf16le", "bytes": len(raw),
                                          "sha256": sha(raw)}}, "runId": 1, "runAttempt": 1,
                      "controlSha": "b" * 40, "metricsArtifactId": 2,
                      "sourceZipSha256": "c" * 64}
            anchor_path.write_text(json.dumps(anchor), encoding="utf-8")
            with patch.object(parent, "ANCHOR", anchor_path), patch.object(parent, "AUDIT", audit_path):
                self.assertEqual(parent.verify_parent_zip(archive, destination), anchor)
                (destination / "parent-proof.json").write_text(json.dumps({key: anchor[key] for key in
                    ("runId", "runAttempt", "controlSha", "engineSha", "metricsArtifactId",
                     "metricsZipSha256", "reportSha256", "sourceZipSha256",
                     "independentAuditSha256")}), encoding="utf-8")
                self.assertEqual(parent.verify_extracted_parent(destination), anchor)
                (destination / "data/one.bf16le").write_bytes(b"\0" * len(raw))
                with self.assertRaisesRegex(ValueError, "raw bytes changed"):
                    parent.verify_extracted_parent(destination)
                (destination / "data/one.bf16le").write_bytes(raw)
                anchor["metricsZipSha256"] = "0" * 64
                anchor_path.write_text(json.dumps(anchor), encoding="utf-8")
                with self.assertRaisesRegex(ValueError, "ZIP changed"):
                    parent.verify_parent_zip(archive, root / "other")

    def test_workflow_authenticates_parent_before_gpu_child_without_new_whole_decoder(self):
        workflow = (CI.parent.parent / ".github/workflows/yue2-bf16-tile-diagnostic.yml").read_text(
            encoding="utf-8")
        self.assertLess(workflow.index("Authenticate and extract only immutable parent stage-2 inputs"),
                        workflow.index("Build the external M3 VAE diagnostic harness"))
        self.assertLess(workflow.index("Build the external M3 VAE diagnostic harness"),
                        workflow.index("Run only the bounded same-input VAE diagnostic"))
        self.assertIn("native_convt_math_mode", workflow)
        source = (CI / "yue2_bf16_tile_diagnostic/src/native_math.rs").read_text(encoding="utf-8")
        self.assertIn("trace_replay_stage(2, &input", source)
        self.assertNotIn("trace_decode_full", source)
        self.assertNotIn("trace_decode_tiled", source)


if __name__ == "__main__":
    unittest.main()
