"""Weights-free comparison protocol checks."""

import hashlib
import json
from pathlib import Path
import struct
import tempfile
import unittest
from unittest.mock import MagicMock, patch

from scripts.ci import sga_adapter_comparison as compare
from scripts.ci import wan_adapter_tier1_fixture as wan


REQUEST = {"selection": "sga-adapter-tier1", "profile": wan.PROFILE,
           "runner": "nax", "download_missing": True}


class ProtocolTests(unittest.TestCase):
    def test_only_source_owned_profile_and_secondary_runner(self):
        self.assertIs(compare.request_value("mlx", REQUEST), wan)
        for backend, supplied in (
            ("candle-cuda", REQUEST), ("mlx", dict(REQUEST, profile="../wanted.py")),
            ("mlx", dict(REQUEST, profile=[])), ("mlx", dict(REQUEST, runner="nax-primary")),
            ("mlx", dict(REQUEST, download_missing=1)), ("mlx", dict(REQUEST, policy={})),
            ("mlx", dict(REQUEST, download_missing=False)),
        ):
            with self.subTest(supplied=supplied), self.assertRaises(ValueError):
                compare.request_value(backend, supplied)

    def test_attempt_path_identity_is_numeric(self):
        self.assertEqual(compare.attempt_identity({"GITHUB_RUN_ID": "17", "GITHUB_RUN_ATTEMPT": "3"}),
                         {"run_id": "17", "run_attempt": "3"})
        with self.assertRaises(ValueError):
            compare.attempt_identity({"GITHUB_RUN_ID": "../17", "GITHUB_RUN_ATTEMPT": "3"})
        with self.assertRaises(ValueError):
            json.loads('{"profile":"a","profile":"b"}', object_pairs_hook=wan.unique_object)

    def test_run_must_complete_within_unified_memory_guard(self):
        result = {"pid": 17, "returncode": 0, "elapsed_seconds": 1, "peak_host_bytes": 1024,
                  "peak_gpu_bytes": None}
        compare.validate_guard(result)
        for changes in ({"returncode": 1}, {"elapsed_seconds": float("nan")}, {"elapsed_seconds": 301},
                        {"peak_host_bytes": 17 * compare.GIB}, {"peak_gpu_bytes": 1024}):
            with self.assertRaises(ValueError):
                compare.validate_guard(dict(result, **changes))

    def test_evidence_must_distinguish_bare_and_both_experts(self):
        marker = "SGA_ADAPTER_TIER1_BODY " + wan.PROFILE
        row = {"device": {"kind": "mlx-metal", "index": 0}, "shape": wan.SHAPE, "order": wan.ORDER,
               "applied": {"high": [1, 1, 1], "low": [1, 1, 1]},
               "velocity_bits": {key: [index] * 768 for index, key in enumerate(("bare", "low", "high"))}}
        with tempfile.TemporaryDirectory() as directory:
            result = Path(directory) / "result.json"
            result.write_text(json.dumps(row), encoding="utf-8")
            self.assertEqual(wan.observe(f"setup\n{marker}\n", result), row)
            for log, data in (
                (marker + "\n" + marker, row), (marker, dict(row, device="mlx-cpu")),
                (marker, dict(row, order={"high": [0, 1, 3], "low": [1, 2, 3]})),
                (marker, dict(row, velocity_bits=dict(row["velocity_bits"], low=[1] * 767))),
                (marker, dict(row, velocity_bits=dict(row["velocity_bits"], low=[0x7F800000] * 768))),
                (marker, dict(row, velocity_bits=dict(row["velocity_bits"], low=[0] * 768))),
            ):
                result.write_text(json.dumps(data), encoding="utf-8")
                with self.assertRaises(ValueError):
                    wan.observe(log, result)


class HeaderTests(unittest.TestCase):
    def test_only_https_hub_origins_allow_header_redirects(self):
        wan.validate_hub_url("https://cas-bridge.xethub.hf.co/file")
        for url in ("http://huggingface.co/file", "https://huggingface.co.invalid/file",
                    "https://user@huggingface.co/file", "https://huggingface.co:8000/file"):
            with self.subTest(url=url), self.assertRaises(ValueError):
                wan.validate_hub_url(url)

    def test_header_request_checks_length_digest_and_agreement(self):
        good = b"header"
        digest = hashlib.sha256(good).hexdigest()
        pin = {file: digest for file in wan.HEADERS}
        with patch.object(wan, "HEADERS", pin), patch.object(wan, "retained_geometry", return_value={"w": [4, 4]}):
            def transport(url, start, end):
                self.assertTrue(url.startswith("https://huggingface.co/SceneWorks/"))
                self.assertIn(wan.REVISION, url)
                if start == 0:
                    self.assertEqual(end, 7)
                    return struct.pack("<Q", len(good))
                self.assertEqual((start, end), (8, 7 + len(good)))
                return good

            self.assertEqual(wan.fetch_geometry(transport), {"w": [4, 4]})
            with self.assertRaisesRegex(ValueError, "bounded"):
                wan.fetch_geometry(lambda *_: struct.pack("<Q", wan.MAX_HEADER_BYTES + 1))
            with self.assertRaisesRegex(ValueError, "snapshot pin"):
                wan.fetch_geometry(lambda _, start, end: transport(_, start, end) if start == 0 else b"wrong!")
            with patch.object(wan, "retained_geometry", side_effect=[{"w": [4, 4]}, {"w": [8, 4]}]):
                with self.assertRaisesRegex(ValueError, "disagree"):
                    wan.fetch_geometry(transport)

    def test_unbounded_or_wrong_range_does_not_read_remote_body(self):
        url = "https://huggingface.co/asset"
        with patch.object(wan, "build_opener") as build:
            response = MagicMock()
            build.return_value.open.return_value.__enter__.return_value = response
            for status, headers in ((200, {}), (206, {"Content-Range": "bytes 1-8/100"}),
                                    (206, {"Content-Range": "bytes 0-7/7"})):
                response.status, response.headers = status, headers
                with self.assertRaisesRegex(ValueError, "exact bounded"):
                    wan.fetch_range(url, 0, 7)
            response.read.assert_not_called()
            response.headers = {"Content-Range": "bytes 0-7/100"}
            response.read.return_value = b"12345678"
            self.assertEqual(wan.fetch_range(url, 0, 7), b"12345678")
            response.read.assert_called_once_with(9)

    def test_staging_cannot_overwrite_an_existing_file_or_accept_a_link(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            item = root / "transformer.safetensors"
            item.write_bytes(b"existing")
            with self.assertRaises(FileExistsError):
                wan.fixture.write_component(item, {"w": [2, 2]})
            self.assertEqual(item.read_bytes(), b"existing")
            for other in sorted(wan.FIXTURE_FILES - {item.name}):
                (root / other).symlink_to(item)
            with self.assertRaises(ValueError):
                wan.identify(root)


if __name__ == "__main__":
    unittest.main()
