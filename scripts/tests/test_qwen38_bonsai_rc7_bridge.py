"""Mutation checks for the one-off RC7 accelerator evidence bridge."""

import copy
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import tomllib
import unittest
from unittest import mock

from scripts.release import qwen38_bonsai_rc7_bridge as bridge


class Rc7BridgeTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.old_matrix = json.loads(bridge.git_file(bridge.OBSERVED_SHA, "release/qwen38-bonsai-matrix.json"))
        cls.new_matrix = json.loads((bridge.ROOT / "release/qwen38-bonsai-matrix.json").read_text())
        cls.old_manifest = tomllib.loads(bridge.git_file(bridge.OBSERVED_SHA, "release/real-weight-models.toml").decode())
        cls.new_manifest = tomllib.loads((bridge.ROOT / "release/real-weight-models.toml").read_text())

    def test_retained_accelerator_contract_and_pins_are_exact(self) -> None:
        bridge.validate_matrix(self.old_matrix, self.new_matrix)
        bridge.validate_manifest(self.old_manifest, self.new_manifest)

    def test_case_or_cell_changes_fail(self) -> None:
        changed = copy.deepcopy(self.new_matrix)
        changed["groups"]["matched"]["case_ids"].remove("code")
        with self.assertRaisesRegex(ValueError, "cases, groups"):
            bridge.validate_matrix(self.old_matrix, changed)
        changed = copy.deepcopy(self.new_matrix)
        changed["cells"].pop()
        with self.assertRaisesRegex(ValueError, "cell set"):
            bridge.validate_matrix(self.old_matrix, changed)

    def test_pin_or_supported_profile_changes_fail(self) -> None:
        changed = copy.deepcopy(self.new_manifest)
        next(model for model in changed["models"] if model["key"] == "bonsai-gguf")["revision"] = "0" * 40
        with self.assertRaisesRegex(ValueError, "model revisions"):
            bridge.validate_manifest(self.old_manifest, changed)
        changed = copy.deepcopy(self.new_manifest)
        next(model for model in changed["models"] if model["key"] == "bonsai-qwen38-parent")["supported_execution_profiles"].append("candle-dense-cpu")
        with self.assertRaisesRegex(ValueError, "supported model"):
            bridge.validate_manifest(self.old_manifest, changed)

    def test_artifact_identity_and_archive_bytes_are_bound(self) -> None:
        role = "mlx"
        expected = bridge.ARTIFACTS[role]
        item = {
            "id": expected["id"], "digest": expected["digest"], "expired": False,
            "name": f"qwen38-bonsai-mlx-{bridge.OBSERVED_SHA}-{bridge.RUN_ID}-1",
            "workflow_run": {"id": bridge.RUN_ID, "head_sha": bridge.OBSERVED_SHA},
        }
        bridge.validate_artifact_metadata(role, item)
        for key, value in (("id", 9), ("digest", "sha256:" + "0" * 64), ("expired", True)):
            changed = copy.deepcopy(item)
            changed[key] = value
            with self.assertRaisesRegex(ValueError, "identity"):
                bridge.validate_artifact_metadata(role, changed)
        changed = copy.deepcopy(item)
        changed["workflow_run"]["head_sha"] = bridge.SCOPED_SHA
        with self.assertRaisesRegex(ValueError, "identity"):
            bridge.validate_artifact_metadata(role, changed)
        with TemporaryDirectory() as directory:
            fake = Path(directory) / "fake.zip"
            fake.write_bytes(b"not the RC7 artifact")
            with self.assertRaisesRegex(ValueError, "archive bytes"):
                bridge.extract_verified_archive(role, fake, Path(directory) / "extract")

    def test_candidate_scope_rejects_supported_path_and_wrong_version(self) -> None:
        head = "c" * 40
        def fake_git(*args):
            if args == ("rev-parse", "HEAD"):
                return head
            if args == ("merge-base", bridge.OBSERVED_SHA, bridge.SCOPED_SHA):
                return bridge.OBSERVED_SHA
            if args == ("merge-base", bridge.SCOPED_SHA, head):
                return bridge.SCOPED_SHA
            if args == ("rev-parse", f"{head}:crates/llm/candle-llm/src/provider.rs"):
                return "e3db6e703fc7408975eaa149199fb01f42621e4c"
            raise AssertionError(args)
        with mock.patch.object(bridge, "git", side_effect=fake_git), mock.patch.object(
            bridge, "changed_paths", side_effect=[bridge.SCOPED_CHANGED_PATHS, {"release/VERSION"}]
        ):
            bridge.validate_source_scope(head=head, clean=True, version="runtime-2026.09.0-rc.8\n")
        with mock.patch.object(bridge, "git", side_effect=fake_git), mock.patch.object(
            bridge, "changed_paths", side_effect=[bridge.SCOPED_CHANGED_PATHS, {"crates/llm/mlx-llm/src/models/qwen35.rs"}]
        ):
            with self.assertRaisesRegex(ValueError, "supported runtime"):
                bridge.validate_source_scope(head=head, clean=True, version="runtime-2026.09.0-rc.8\n")
        with mock.patch.object(bridge, "git", side_effect=fake_git), mock.patch.object(
            bridge, "changed_paths", side_effect=[bridge.SCOPED_CHANGED_PATHS, {"release/VERSION"}]
        ):
            with self.assertRaisesRegex(ValueError, "RC8"):
                bridge.validate_source_scope(head=head, clean=True, version="runtime-2026.09.0-rc.9\n")


if __name__ == "__main__":
    unittest.main()
