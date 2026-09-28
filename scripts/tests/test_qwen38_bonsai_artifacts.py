"""Regression checks for immutable same-run Qwen campaign partitions."""

import argparse
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

from scripts.release import qwen38_bonsai_artifacts as artifacts
from scripts.release import qwen38_bonsai_terminal as terminal
from scripts.release import qwen38_bonsai_assets as assets


SHA = "a" * 40
RUN = "35589659377"
HOST = "frozen-windows-host"
GPU = "GPU-b1a31911-c7b4-2901-3d8b-9a62e228bfc0"


def publisher(root: Path, hostname: str = HOST) -> None:
    terminal.write_new(root / "snapshot-metadata.json", {
        "all_metadata_qualified": True, "publisher_verified_all": True,
        "runtime_sha": SHA, "publisher_closure_sha256": terminal.sha256(assets.DEFAULT_CLOSURE),
        "hostname": hostname,
    })
    terminal.write_new(root / "provision-report.json", {
        "complete": True, "runtime_sha": SHA,
        "publisher_closure_sha256": terminal.sha256(assets.DEFAULT_CLOSURE),
        "model_execution_performed": False,
    })


def artifact(role: str, attempt: int, identity: int, sha: str = SHA) -> dict:
    return {
        "name": f"qwen38-bonsai-{role}-{sha}-{RUN}-{attempt}",
        "id": identity, "expired": False, "digest": "sha256:" + "b" * 64,
        "workflow_run": {"id": int(RUN), "head_sha": sha},
    }


class QwenPartitionArtifactTests(unittest.TestCase):
    def test_selects_exact_accelerator_partitions_and_newest_retry(self) -> None:
        items = [artifact(role, 1, index + 10) for index, role in enumerate(artifacts.ROLES)]
        items.append(artifact("mlx", 2, 99))
        selected = artifacts.choose_artifacts(items, roles=artifacts.ROLES, runtime_sha=SHA, run_id=RUN)
        by_role = {entry["role"]: entry for entry in selected}
        self.assertEqual(set(by_role), {"mlx", "cuda"})
        self.assertEqual(by_role["mlx"]["artifact_id"], 99)
        self.assertEqual(by_role["cuda"]["artifact_id"], 11)
        self.assertEqual(by_role["mlx"]["run_attempt"], 2)

    def test_newest_invalid_artifact_fails_instead_of_falling_back(self) -> None:
        old = artifact("mlx", 1, 1)
        latest = artifact("mlx", 2, 2)
        latest["workflow_run"]["head_sha"] = "c" * 40
        with self.assertRaisesRegex(ValueError, "source identity"):
            artifacts.choose_artifacts([old, latest], roles=("mlx",), runtime_sha=SHA, run_id=RUN)
        latest["workflow_run"]["head_sha"] = SHA
        latest["digest"] = None
        with self.assertRaisesRegex(ValueError, "source identity"):
            artifacts.choose_artifacts([old, latest], roles=("mlx",), runtime_sha=SHA, run_id=RUN)

    def test_cuda_staging_preserves_all_accelerator_cells(self) -> None:
        with TemporaryDirectory() as directory:
            source = Path(directory) / "working"
            output = Path(directory) / "cuda"
            source.mkdir()
            publisher(source)
            for name in ("hardware-before.json", "gpu-reservation.json"):
                (source / name).write_text("{}\n", encoding="utf-8")
            for cell in artifacts.CUDA_CELLS:
                (source / f"{cell}-preflight.json").write_text("{}\n", encoding="utf-8")
                row = source / cell
                row.mkdir()
                (row / "seal.json").write_text("{}\n", encoding="utf-8")
            self.assertEqual(artifacts.stage_cuda(argparse.Namespace(source=source, output=output, runtime_sha=SHA)), 0)
            for cell in artifacts.CUDA_CELLS:
                self.assertTrue((output / f"{cell}-preflight.json").is_file())
                self.assertTrue((output / cell / "seal.json").is_file())
            self.assertEqual(len(list(output.glob("*-preflight.json"))), 8)
            (source / "gpu-reservation.json").unlink()
            with self.assertRaisesRegex(ValueError, "CUDA evidence lacks gpu-reservation.json"):
                artifacts.stage_cuda(argparse.Namespace(source=source, output=Path(directory) / "missing-reservation", runtime_sha=SHA))

    def test_physical_host_identity_requires_gpu0_uuid(self) -> None:
        record = {"host": {"hostname": HOST, "system": "Windows", "machine": "AMD64", "physical_memory_bytes": 256},
                  "gpus": [{"index": 0, "uuid": GPU}]}
        self.assertEqual(terminal.physical_windows_host_identity(record)[-1], GPU)
        record["gpus"] = []
        with self.assertRaisesRegex(ValueError, "physical Windows host identity"):
            terminal.physical_windows_host_identity(record)


if __name__ == "__main__":
    unittest.main()
