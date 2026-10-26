"""No-model checks for the manual iOS CPU conversion selector."""

import importlib.util
import json
from pathlib import Path
import struct
import tempfile
import unittest

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
            self.assertEqual(MODULE.tensor_budget_mib("iris-generation", rows), (1, 40))
            path.write_bytes(struct.pack("<Q", len(header)) + header + b"1234")
            with self.assertRaises(ValueError):
                MODULE.header_rows(path)

    def test_outputs_are_unique_per_run_and_build_reused_only_for_source(self):
        binding = MODULE.parse_binding(json.dumps({"source_sha": "a" * 40,
                                                   "route": "iris-generation-q4"}))
        with tempfile.TemporaryDirectory() as temp:
            cache = Path(temp) / "huggingface/hub"
            cache.mkdir(parents=True)
            root, output, target = MODULE.output_paths(cache, "123", "1", binding)
            self.assertEqual(output, root / "iris-generation-q4")
            self.assertTrue(root.is_relative_to(Path(temp).resolve()))
            rerun, _, same_target = MODULE.output_paths(cache, "123", "2", binding)
            self.assertNotEqual(root, rerun)
            self.assertEqual(target, same_target)
            with self.assertRaises(ValueError):
                MODULE.output_paths(cache, "invalid", "1", binding)


if __name__ == "__main__":
    unittest.main()
