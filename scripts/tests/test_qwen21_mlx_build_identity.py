"""Build identity comes from Cargo's actual linker paths, with fail-closed source checks."""

import json
from pathlib import Path
import tempfile
import unittest

from scripts.ci.qwen21_mlx_build_identity import collect_identity, collect_lib_test_identity


class BuildIdentityTests(unittest.TestCase):
    def test_lib_test_identity_rejects_integration_release_library_missing_and_duplicates(self):
        with tempfile.TemporaryDirectory() as directory:
            binary = Path(directory) / "lib-test"
            binary.write_bytes(b"actual cfg(test) executable")
            record = {"reason": "compiler-artifact", "target": {
                "name": "mlx_gen_qwen_image_2_1", "kind": ["lib"]},
                "profile": {"test": True}, "executable": str(binary)}
            identity = collect_lib_test_identity([record])
            self.assertEqual(identity["bytes"], 27)
            self.assertEqual(identity["path"], str(binary.resolve()))
            for records in ([], [record, record],
                            [{**record, "target": {"name": "integration", "kind": ["test"]}}],
                            [{**record, "profile": {"test": False}}],
                            [{**record, "executable": None}]):
                with self.subTest(records=records), self.assertRaises(ValueError):
                    collect_lib_test_identity(records)

    def fixture(self, root):
        libraries = root / "build" / "lib"
        libraries.mkdir(parents=True)
        for name in ("libmlx.a", "libmlxc.a", "mlx.metallib"):
            (libraries / name).write_bytes(name.encode())
        (libraries / "pmetal-mlx-prebuilt.txt").write_text("fingerprint=actual-source\ntarget=aarch64-apple-darwin\n", encoding="utf-8")
        staged = root / "mlx-c-staged"
        staged.mkdir()
        (staged / "CMakeLists.txt").write_text("FetchContent_Declare(mlx GIT_TAG v0.32.0)", encoding="utf-8")
        revision = "a" * 40
        messages = [{"reason": "build-script-executed",
                     "package_id": f"git+https://example.test?rev={revision}#pmetal-mlx-sys@0.2.4",
                     "out_dir": str(root), "linked_paths": ["native=" + str(libraries)]}]
        lock = {"package": [{"name": "pmetal-mlx-sys", "source": "git+test#" + revision}]}
        return messages, lock, libraries, staged

    def test_captures_actual_archives_manifest_source_and_metallib(self):
        with tempfile.TemporaryDirectory() as directory:
            messages, lock, libraries, staged = self.fixture(Path(directory))
            identity = collect_identity(messages, lock)
            self.assertEqual(identity["actualStagedCoreTag"], "v0.32.0")
            self.assertEqual(identity["buildManifest"]["fingerprint"], "actual-source")
            self.assertEqual(identity["archives"][0]["path"], str((libraries / "libmlx.a").resolve()))
            self.assertEqual(len(identity["archives"][0]["sha256"]), 64)
            old = identity["archives"][0]["sha256"]
            (libraries / "libmlx.a").write_bytes(b"different archive")
            self.assertNotEqual(collect_identity(messages, lock)["archives"][0]["sha256"], old)

    def test_wrong_locked_revision_missing_archive_and_stale_tag_are_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            messages, lock, libraries, staged = self.fixture(Path(directory))
            wrong = json.loads(json.dumps(messages))
            wrong[0]["package_id"] = wrong[0]["package_id"].replace("a" * 40, "b" * 40)
            with self.assertRaisesRegex(ValueError, "locked mlx-sys"):
                collect_identity(wrong, lock)
            (staged / "CMakeLists.txt").write_text("GIT_TAG v0.31.1", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "v0.32.0"):
                collect_identity(messages, lock)
            (libraries / "libmlx.a").unlink()
            with self.assertRaisesRegex(ValueError, "library directory"):
                collect_identity(messages, lock)

    def test_prebuilt_preserves_actual_source_fingerprint_without_inventing_staged_tag(self):
        with tempfile.TemporaryDirectory() as directory:
            messages, lock, _, staged = self.fixture(Path(directory))
            (staged / "CMakeLists.txt").unlink()
            identity = collect_identity(messages, lock)
            self.assertIsNone(identity["actualStagedCoreTag"])
            self.assertEqual(identity["linkMode"], "prebuilt_verified_by_mlx_sys_source_fingerprint")
