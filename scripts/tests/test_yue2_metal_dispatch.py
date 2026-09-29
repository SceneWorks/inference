"""CPU-only checks for the isolated YuE2 Metal dispatch guards."""

from __future__ import annotations

import importlib.util
import json
import struct
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def load(name: str, file: Path):
    spec = importlib.util.spec_from_file_location(name, file)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


preflight = load("yue2_metal_preflight", ROOT / "scripts/ci/yue2_metal_preflight.py")
reference = load("yue2_quality_reference", ROOT / "scripts/ci/yue2_quality_reference.py")


class MetalPreflightTests(unittest.TestCase):
    def test_busy_process_and_resource_refusals(self) -> None:
        vm = (
            "Mach Virtual Memory Statistics: (page size of 16384 bytes)\n"
            "Pages free: 1000000.\nPages inactive: 1000000.\n"
            "Pages speculative: 1000000.\n"
        )
        self.assertEqual(preflight.available_memory(vm), 49_152_000_000)
        processes = """100 1 /bin/zsh zsh
101 100 /usr/bin/cargo cargo test --release
102 1 /opt/actions/Runner.Worker Runner.Worker
103 1 /opt/actions/Runner.Worker Runner.Worker
"""
        busy = preflight.competing_processes(processes, 100)
        self.assertEqual(len(busy), 2)
        self.assertEqual(
            preflight.assess("nax-macos-2", 64 << 30, 40 << 30, 40 << 30, []), []
        )
        self.assertEqual(
            len(preflight.assess("wrong-host", 16 << 30, 8 << 30, 1 << 30, busy)), 4
        )

    def test_incomplete_vm_stat_fails_closed(self) -> None:
        with self.assertRaises(ValueError):
            preflight.available_memory("Pages free: 123.\n")

    def test_controller_argument_is_not_running_api_but_real_worker_is_busy(self) -> None:
        controller = (
            "79363 79100 /opt/homebrew/Ce /opt/homebrew/opt/python@3.12/bin/python3.12 "
            "scripts/ci/yue2_app_metal.py --app app --engine engine "
            "--api-bin /Users/MTrefry/actions-runner-nax/_work/inference/"
            "yue2-app-metal-target-36565987342-1/release/sceneworks-rust-api"
        )
        self.assertEqual(preflight.competing_processes(controller, 99999), [])
        running = controller + "\n81000 79363 /opt/homebrew/Ce /tmp/target/release/sceneworks-rust-api --port 8123\n"
        self.assertEqual(
            preflight.competing_processes(running, 99999),
            ["pid=81000 ppid=79363 executable=sceneworks-rust-api comm=/opt/homebrew/Ce"],
        )
        self.assertEqual(
            len(preflight.competing_processes("81001 79363 /tmp/sceneworks-worker /tmp/sceneworks-worker --job 1", 99999)), 1
        )


class ReferenceTests(unittest.TestCase):
    def test_reference_shape_dtype_and_missing_key(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            fixture = root / "fixture.json"
            fixture.write_text(
                json.dumps({"modes": {"full": {"abc": {"emitted": [1, 2]}, "semantic": None}}})
            )
            expected = reference.expected_shapes(json.loads(fixture.read_text()))
            offset = 0
            header = {}
            for key, shape in expected.items():
                length = 4
                for dimension in shape:
                    length *= dimension
                header[key] = {
                    "dtype": "F32",
                    "shape": shape,
                    "data_offsets": [offset, offset + length],
                }
                offset += length
            path = root / "reference.safetensors"

            def write() -> None:
                raw = json.dumps(header).encode()
                with path.open("wb") as stream:
                    stream.write(struct.pack("<Q", len(raw)))
                    stream.write(raw)
                    stream.truncate(8 + len(raw) + offset)

            write()
            self.assertEqual(reference.verify(path, fixture)["required_tensors"], len(expected))
            header["ar/full/abc"]["dtype"] = "BF16"
            write()
            with self.assertRaisesRegex(ValueError, "incorrect dtype or shape"):
                reference.verify(path, fixture)
            del header["ar/full/abc"]
            write()
            with self.assertRaisesRegex(ValueError, "missing current-M2 fixture tensors"):
                reference.verify(path, fixture)


if __name__ == "__main__":
    unittest.main()
