"""No-model checks for the manual iOS CPU conversion selector."""

import importlib.util
import json
from pathlib import Path
import struct
import tempfile
import unittest
import sys

SCRIPT = Path(__file__).resolve().parents[1] / "ci" / "ios_prequant.py"
SPEC = importlib.util.spec_from_file_location("ios_prequant", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


class IosPrequantTests(unittest.TestCase):
    def test_selector_requires_exact_frozen_source_and_one_known_route(self):
        sha = "a" * 40
        for route in MODULE.ROUTES:
            self.assertEqual(MODULE.parse_binding(json.dumps({"source_sha": sha, "route": route}))["route"], route)
        for bad in ({"source_sha": sha, "route": "all"},
                    {"source_sha": sha, "route": "iris-generation-q2"},
                    {"source_sha": sha, "route": "iris-generation-q4", "argv": "rm -rf /"},
                    {"source_sha": "main", "route": "iris-generation-q4"}):
            with self.subTest(bad=bad), self.assertRaises(ValueError):
                MODULE.parse_binding(json.dumps(bad))

    def test_source_and_command_are_pinned_and_typed(self):
        cache = Path("/cache/hub")
        for route in MODULE.ROUTES:
            name, revision, source = MODULE.source_for(route, cache)
            self.assertEqual(source.parts[0:3], ("/", "cache", "hub"))
            self.assertIn("@", revision)
            self.assertIn(revision.split("@", 1)[1], source.parts)
            if name == "anima-turbo":
                self.assertEqual(MODULE.converter_arguments(name, source, Path("/out") / route, None, 1024),
                                 [str(source), str(Path("/out") / route)])
                continue
            bits = int(route.rsplit("-q", 1)[1])
            args = MODULE.converter_arguments(name, source, Path("/out") / route, bits, 1024)
            self.assertEqual(args[-3:], [str(Path("/out") / route), str(bits), "1024"])
            self.assertEqual(len(args), 5 if name.startswith("iris-") else 4)

    def test_header_budget_is_derived_without_payload_read(self):
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "model.safetensors"
            header = json.dumps({"weight": {"dtype": "F16", "shape": [2, 2],
                                             "data_offsets": [0, 8]}}).encode()
            path.write_bytes(struct.pack("<Q", len(header)) + header + b"12345678")
            rows = MODULE.header_rows(path)
            self.assertEqual(rows[0]["bytes"], 8)
            self.assertEqual(MODULE.tensor_budget_mib("iris-generation", rows),
                             (1, 40, 8 + 40 + 8 * MODULE.GIB))
            path.write_bytes(struct.pack("<Q", len(header)) + header + b"1234")
            with self.assertRaises(ValueError):
                MODULE.header_rows(path)

    def test_anima_budget_prices_existing_two_tier_converter(self):
        headers = [{"file": "/source/diffusion_models/anima-turbo-v1.0.safetensors", "bytes": 100},
                   {"file": "/source/text_encoders/qwen_3_06b_base.safetensors", "bytes": 20},
                   {"file": "/source/vae/qwen_image_vae.safetensors", "bytes": 10}]
        mib, staging, work = MODULE.tensor_budget_mib("anima-turbo", headers)
        self.assertEqual((mib, staging, work), (1, 400, 130 + 400 + 8 * MODULE.GIB))

    def test_outputs_are_unique_per_run_and_build_reused_only_for_source(self):
        binding = MODULE.parse_binding(json.dumps({"source_sha": "a" * 40,
                                                   "route": "iris-generation-q4"}))
        with tempfile.TemporaryDirectory() as temp:
            cache = Path(temp) / "huggingface/hub"
            cache.mkdir(parents=True)
            root, output, target = MODULE.output_paths(cache, "123", "1", binding, "prebuilt")
            self.assertEqual(output, root / "iris-generation-q4")
            self.assertTrue(root.is_relative_to(Path(temp).resolve()))
            rerun, _, same_target = MODULE.output_paths(cache, "123", "2", binding, "prebuilt")
            self.assertNotEqual(root, rerun)
            self.assertEqual(target, same_target)
            _, _, source_target = MODULE.output_paths(cache, "123", "2", binding, "source")
            self.assertNotEqual(target, source_target)
            with self.assertRaises(ValueError):
                MODULE.output_paths(cache, "invalid", "1", binding, "prebuilt")

    def test_live_memory_admission_reserves_host_headroom(self):
        pressure = "The system has 137438953472 bytes\nSystem-wide memory free percentage: 80%"
        vm_stat = 'Mach Virtual Memory Statistics: (page size of 16384 bytes)\nPages free: 1500000.\nPages inactive: 1500000.\nPages speculative: 1000000.\n'
        observed = MODULE.parse_host_memory(pressure, vm_stat)
        self.assertEqual(observed["available_bytes"], 4000000 * 16384)
        original = MODULE.host_memory
        try:
            MODULE.host_memory = lambda: observed
            cap, reserve, _ = MODULE.admit_memory(32 * MODULE.GIB, 8 * MODULE.GIB)
            self.assertEqual(cap, 32 * MODULE.GIB)
            self.assertEqual(reserve, 16 * MODULE.GIB)
            with self.assertRaisesRegex(ValueError, "admission refused"):
                MODULE.admit_memory(50 * MODULE.GIB, 50 * MODULE.GIB)
        finally:
            MODULE.host_memory = original

    def test_pinned_hub_manifest_verifies_bytes_not_snapshot_name(self):
        with tempfile.TemporaryDirectory() as temp:
            source = Path(temp)
            file = source / "model.safetensors"
            file.write_bytes(b"exact model bytes")
            owner, repo, revision, _ = MODULE.PINS["iris-generation"]
            sibling = {"rfilename": "model.safetensors", "size": file.stat().st_size,
                       "lfs": {"sha256": MODULE.sha(file), "size": file.stat().st_size}}
            document = {"id": f"{owner}/{repo}", "sha": revision, "private": False,
                        "gated": False, "siblings": [sibling]}
            class Response:
                def __enter__(self): return self
                def __exit__(self, *_): pass
                def read(self, size): return json.dumps(document).encode()
            original = MODULE.urllib.request.urlopen
            try:
                MODULE.urllib.request.urlopen = lambda *_, **__: Response()
                result = MODULE.verify_hub_source("iris-generation", source, [file])
                self.assertEqual(result["status"], "hub_content_verified")
                file.write_bytes(b"different bytes")
                with self.assertRaisesRegex(ValueError, "size/path mismatch|content mismatch"):
                    MODULE.verify_hub_source("iris-generation", source, [file])
            finally:
                MODULE.urllib.request.urlopen = original

    def test_git_blob_sidecar_and_bounded_log_guard(self):
        with tempfile.TemporaryDirectory() as temp:
            source = Path(temp)
            file = source / "config.json"
            file.write_bytes(b'{}')
            owner, repo, revision, _ = MODULE.PINS["iris-generation"]
            document = {"id": f"{owner}/{repo}", "sha": revision, "private": False,
                        "gated": False, "siblings": [{"rfilename": file.name,
                        "size": 2, "blobId": MODULE.git_blob_sha1(file)}]}
            class Response:
                def __enter__(self): return self
                def __exit__(self, *_): pass
                def read(self, size): return json.dumps(document).encode()
            old_open = MODULE.urllib.request.urlopen
            old_host = MODULE.host_memory
            old_limit = MODULE.MAX_LOG_BYTES
            old_floor = MODULE.DISK_FLOOR_BYTES
            try:
                MODULE.urllib.request.urlopen = lambda *_, **__: Response()
                self.assertEqual(MODULE.verify_hub_source("iris-generation", source, [file])["files"][0]["kind"],
                                 "git_blob_sha1")
                MODULE.host_memory = lambda: {"available_bytes": 100 * MODULE.GIB}
                MODULE.MAX_LOG_BYTES = 1024
                MODULE.DISK_FLOOR_BYTES = 0
                log = source / "bounded.log"
                with self.assertRaisesRegex(RuntimeError, "log byte ceiling exceeded"):
                    MODULE.run_guarded([sys.executable, "-c", "import time; print('x' * 2048, flush=True); time.sleep(1)"],
                                       None, log, 5, MODULE.GIB, MODULE.GIB)
                self.assertLess(log.stat().st_size, 4096)
            finally:
                MODULE.urllib.request.urlopen = old_open
                MODULE.host_memory = old_host
                MODULE.MAX_LOG_BYTES = old_limit
                MODULE.DISK_FLOOR_BYTES = old_floor


if __name__ == "__main__":
    unittest.main()
