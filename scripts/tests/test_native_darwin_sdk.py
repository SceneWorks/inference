"""Portable SDK/linker selection tests; compiler and xcrun processes are mocked."""

import hashlib
from pathlib import Path
import subprocess
import tempfile
import unittest
from unittest.mock import patch

from scripts.ci import native_darwin_sdk as sdk


class NativeDarwinSdkTests(unittest.TestCase):
    def environment(self):
        return {"CARGO_ENCODED_RUSTFLAGS": "-C\x1flinker-flavor=ld64.lld\x1f-C\x1flinker=/selected/rust-lld",
                "MACOSX_DEPLOYMENT_TARGET": "26.2"}

    def test_candidates_prefer_selected_then_versions_and_deduplicate_aliases(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            selected = root / "MacOSX27.0.sdk"
            alternate = root / "MacOSX26.5.sdk"
            selected.mkdir()
            alternate.mkdir()
            (root / "MacOSX.sdk").symlink_to(selected, target_is_directory=True)
            with patch.object(sdk, "sdk_directories", return_value=[root]), \
                    patch.object(sdk.subprocess, "run", return_value=subprocess.CompletedProcess(
                        [], 0, stdout=str(selected) + "\n")) as command:
                self.assertEqual(sdk.sdk_candidates({}, root=root), [selected.resolve(), alternate.resolve()])
            self.assertEqual(command.call_count, 1)

    def test_bad_tapi_sdk_falls_back_only_after_real_macro_compile_and_load_pass(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            bad, good = root / "incompatible.sdk", root / "compatible.sdk"
            bad.mkdir()
            (good / "usr/lib").mkdir(parents=True)
            (good / "SDKSettings.json").write_text('{}', encoding="utf-8")
            (good / "usr/lib/libSystem.tbd").write_bytes(b"SDK TAPI fixture")
            invocations = []
            def command(argv, **kwargs):
                invocations.append(argv)
                self.assertIn("linker-flavor=ld64.lld", argv)
                self.assertEqual(kwargs['env']['MACOSX_DEPLOYMENT_TARGET'], "26.2")
                if kwargs['env']['SDKROOT'] == str(bad):
                    return subprocess.CompletedProcess(argv, 1, stdout="", stderr="TAPI: unknown architecture arm64e.x1")
                if "proc-macro" in argv:
                    source = Path(argv[1]).read_text(encoding="utf-8")
                    self.assertIn("#[proc_macro]", source)
                else:
                    self.assertIn("--emit=metadata", argv)
                    self.assertIn("--extern", argv)
                    self.assertIn("sdk_link_probe::marker!()", Path(argv[1]).read_text(encoding="utf-8"))
                return subprocess.CompletedProcess(argv, 0, stdout="", stderr="")
            env = self.environment()
            with patch.object(sdk, "sdk_candidates", return_value=[bad, good]), \
                    patch.object(sdk.subprocess, "run", side_effect=command):
                selected, receipt = sdk.select_sdk(["/selected/rustc"], env, root=root)
            self.assertEqual(selected, str(good))
            self.assertEqual(len(invocations), 3)
            self.assertEqual([attempt['passed'] for attempt in receipt['attempts']], [False, True])
            self.assertIn("arm64e.x1", receipt['attempts'][0]['commands'][0]['stderr_tail'])
            self.assertEqual(receipt['selected_sdk_file_sha256']['usr/lib/libSystem.tbd'],
                             hashlib.sha256(b"SDK TAPI fixture").hexdigest())
            self.assertNotIn("SDKROOT", env)  # Only controller commits a successful choice.

    def test_explicit_sdk_failure_does_not_try_alternatives(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            explicit = root / "explicit.sdk"
            explicit.mkdir()
            env = dict(self.environment(), SDKROOT=str(explicit))
            with patch.object(sdk, "sdk_directories") as alternatives, \
                    patch.object(sdk.subprocess, "run", return_value=subprocess.CompletedProcess(
                        [], 1, stdout="", stderr="unknown architecture arm64e.x1")) as command:
                with self.assertRaisesRegex(ValueError, "explicit SDKROOT.*arm64e.x1"):
                    sdk.select_sdk(["rustc"], env, root=root)
            alternatives.assert_not_called()
            self.assertEqual(command.call_count, 1)
            self.assertEqual(env['SDKROOT'], str(explicit))

    def test_dylib_link_success_without_load_success_is_not_compatible(self):
        commands = [subprocess.CompletedProcess([], 0, stdout="", stderr=""),
                    subprocess.CompletedProcess([], 1, stdout="", stderr="mis-aligned LINKEDIT string pool")]
        with patch.object(sdk.subprocess, "run", side_effect=commands):
            result = sdk.probe_sdk(["rustc"], self.environment(), root=Path("."), sdk=Path("/SDK"))
        self.assertFalse(result['passed'])
        self.assertEqual(result['commands'][-1]['stage'], "load-and-expand-proc-macro")
        self.assertIn("LINKEDIT", result['commands'][-1]['stderr_tail'])

    def test_timeout_and_invalid_explicit_directory_refuse(self):
        with patch.object(sdk.subprocess, "run", side_effect=subprocess.TimeoutExpired("rustc", 30)):
            result = sdk.probe_sdk(["rustc"], self.environment(), root=Path("."), sdk=Path("/SDK"))
        self.assertFalse(result['passed'])
        self.assertEqual(result['commands'][0]['timeout_seconds'], 30)
        with patch.object(sdk.subprocess, "run") as command:
            with self.assertRaisesRegex(ValueError, "explicit SDKROOT"):
                sdk.sdk_candidates({"SDKROOT": "relative/sdk"}, root=Path("."))
            command.assert_not_called()


if __name__ == "__main__":
    unittest.main()
