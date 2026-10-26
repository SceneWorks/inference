"""Portable startup fixtures; no compiler, Cargo, SDK command or model is run."""

import hashlib
from pathlib import Path
import subprocess
import tempfile
import unittest
from unittest.mock import patch

from scripts.ci import model_inference_proofs as proof


class DarwinLinkerTests(unittest.TestCase):
    def fixture(self, root):
        sysroot = root / "custom rustup home/toolchains/selected"
        linker = sysroot / "lib/rustlib/aarch64-apple-darwin/bin/rust-lld"
        linker.parent.mkdir(parents=True)
        linker.write_bytes(b"fixture selected linker; never executed")
        sdk = root / "selected SDK/MacOSX.sdk"
        sdk.mkdir(parents=True)
        return sysroot, linker, sdk

    def inspector(self, sysroot, sdk):
        def inspect(argv, **kwargs):
            self.assertTrue(kwargs["check"])
            if argv == ["xcrun", "--sdk", "macosx", "--show-sdk-path"]:
                output = str(sdk)
            elif argv[-2:] == ["--print", "sysroot"]:
                output = str(sysroot)
            elif argv[-1] == "-vV":
                output = "rustc 1.96.0\nhost: aarch64-apple-darwin"
            else:
                self.fail(f"unexpected startup command: {argv}")
            return subprocess.CompletedProcess(argv, 0, stdout=output + "\n")
        return inspect

    def test_selected_toolchain_linker_preserves_native_configuration_and_plain_flags(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            sysroot, linker, sdk = self.fixture(root)
            env = {"SDKROOT": str(sdk), "MACOSX_DEPLOYMENT_TARGET": "26.2",
                   "RUSTFLAGS": "-C debuginfo=0 -C linker=old-linker -C linker-flavor=darwin",
                   "PMETAL_MLX_PREBUILT_DIR": "/verified/native",
                   "PMETAL_METALLIB_PATH": "/verified/native/mlx.metallib"}
            original = env.copy()
            with patch.object(proof.subprocess, "run", side_effect=self.inspector(sysroot, sdk)) as run:
                receipt = proof.configure_native_darwin_build(
                    env, ["/selected/rustc"], root=root, system="Darwin")
            self.assertEqual(run.call_count, 2)  # Explicit SDK is retained.
            self.assertEqual(env["CARGO_ENCODED_RUSTFLAGS"].split("\x1f"), [
                "-C", "debuginfo=0", "-C", "linker=old-linker", "-C", "linker-flavor=darwin",
                "-C", "linker-flavor=ld64.lld", "-C", "linker=" + str(linker)])
            self.assertEqual(receipt["linker_sha256"], hashlib.sha256(linker.read_bytes()).hexdigest())
            for name, value in original.items():
                self.assertEqual(env[name], value)
            for profile in ("DEV", "TEST"):
                self.assertEqual(env[f"CARGO_PROFILE_{profile}_STRIP"], "none")
                self.assertEqual(env[f"CARGO_PROFILE_{profile}_BUILD_OVERRIDE_STRIP"], "none")

    def test_encoded_flags_take_precedence_and_missing_sdk_uses_selected_xcrun(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            sysroot, linker, sdk = self.fixture(root)
            env = {"RUSTFLAGS": "-C ignored-plain", "CARGO_ENCODED_RUSTFLAGS": "-C\x1flink-arg=has a space"}
            with patch.object(proof.subprocess, "run", side_effect=self.inspector(sysroot, sdk)) as run:
                receipt = proof.configure_native_darwin_build(
                    env, ["/selected/rustc"], root=root, system="Darwin")
            self.assertEqual(run.call_count, 3)
            self.assertEqual(receipt["rustflags"][:2], ["-C", "link-arg=has a space"])
            self.assertNotIn("ignored-plain", env["CARGO_ENCODED_RUSTFLAGS"])
            self.assertEqual(env["SDKROOT"], str(sdk))
            self.assertEqual(receipt["linker"], str(linker))

    def test_target_flags_are_retained_only_when_no_global_flags_are_present(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            sysroot, _, sdk = self.fixture(root)
            for inherited, expected in (({}, ["--cfg", "target_flag"]),
                                        ({"RUSTFLAGS": ""}, []),
                                        ({"CARGO_ENCODED_RUSTFLAGS": ""}, [])):
                env = {"SDKROOT": str(sdk), "CARGO_TARGET_AARCH64_APPLE_DARWIN_RUSTFLAGS": "--cfg target_flag", **inherited}
                with patch.object(proof.subprocess, "run", side_effect=self.inspector(sysroot, sdk)):
                    receipt = proof.configure_native_darwin_build(
                        env, ["rustc"], root=root, system="Darwin")
                self.assertEqual(receipt["rustflags"][:-4], expected)

    def test_missing_bundled_linker_refuses_before_discovery_or_environment_mutation(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            sysroot, linker, sdk = self.fixture(root)
            linker.unlink()
            env = {"SDKROOT": str(sdk)}
            with patch.object(proof.subprocess, "run", side_effect=self.inspector(sysroot, sdk)) as run:
                with self.assertRaisesRegex(ValueError, "lacks.*rust-lld"):
                    proof.configure_native_darwin_build(env, ["rustc"], root=root, system="Darwin")
            self.assertEqual(run.call_count, 2)
            self.assertEqual(env, {"SDKROOT": str(sdk)})

    def test_cross_target_refuses_and_other_platforms_are_unchanged(self):
        env = {"CARGO_BUILD_TARGET": "aarch64-apple-ios", "RUSTFLAGS": "old"}
        with patch.object(proof.subprocess, "run", return_value=subprocess.CompletedProcess(
                [], 0, stdout="host: aarch64-apple-darwin\n")) as run:
            with self.assertRaisesRegex(ValueError, "cross target"):
                proof.configure_native_darwin_build(env, ["rustc"], root=Path("."), system="Darwin")
            self.assertEqual(run.call_count, 1)
        for system in ("Windows", "Linux"):
            original = env.copy()
            with patch.object(proof.subprocess, "run") as run:
                self.assertEqual(proof.configure_native_darwin_build(
                    env, ["rustc"], root=Path("."), system=system), {"mechanism": "not applicable"})
                run.assert_not_called()
            self.assertEqual(env, original)


if __name__ == "__main__":
    unittest.main()
