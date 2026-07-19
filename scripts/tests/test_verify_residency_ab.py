import tempfile
import unittest
from pathlib import Path

from scripts.release.verify_residency_ab import read_peak, verify


class VerifyResidencyAbTests(unittest.TestCase):
    def write_log(self, root: Path, name: str, mode: str, peak: int) -> Path:
        path = root / name
        path.write_text(
            f"test output\nSEQ_AB model=flux1_dev mode={mode} gpu=NVIDIA peak_mib={peak} | sample\n",
            encoding="utf-8",
        )
        return path

    def test_accepts_material_reduction(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            resident = self.write_log(root, "resident.log", "resident", 24000)
            sequential = self.write_log(root, "sequential.log", "spec-sequential", 15000)
            self.assertEqual(verify(resident, sequential, 512), (24000, 15000))

    def test_rejects_wrong_mode_and_insufficient_reduction(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            resident = self.write_log(root, "resident.log", "resident", 16000)
            wrong = self.write_log(root, "wrong.log", "resident", 14000)
            with self.assertRaisesRegex(RuntimeError, "expected mode=spec-sequential"):
                read_peak(wrong, "spec-sequential")
            sequential = self.write_log(root, "sequential.log", "spec-sequential", 15700)
            with self.assertRaisesRegex(RuntimeError, "required at least 512"):
                verify(resident, sequential, 512)

    def test_rejects_missing_or_ambiguous_results(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "log"
            path.write_text("no result\n", encoding="utf-8")
            with self.assertRaisesRegex(RuntimeError, "found 0"):
                read_peak(path, "resident")
            path.write_text(
                "SEQ_AB mode=resident peak_mib=1\nSEQ_AB mode=resident peak_mib=2\n",
                encoding="utf-8",
            )
            with self.assertRaisesRegex(RuntimeError, "found 2"):
                read_peak(path, "resident")


if __name__ == "__main__":
    unittest.main()
