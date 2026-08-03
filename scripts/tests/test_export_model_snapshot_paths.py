import tempfile
import unittest
from pathlib import Path
from unittest import mock

from scripts.release import export_model_snapshot_paths as EXPORT

SCRIPT = Path(__file__).resolve().parents[1] / "release" / "export_model_snapshot_paths.py"


class ExportModelSnapshotPathsTests(unittest.TestCase):
    def manifest(self, directory: Path, revision: str = "a" * 40) -> Path:
        path = directory / "models.toml"
        path.write_text(
            "[[models]]\n"
            'key = "sa3-test"\n'
            'repository = "example/test"\n'
            f'revision = "{revision}"\n'
            'environment = ["SA3_TEST_SNAPSHOT"]\n'
            'expected_files = ["model.safetensors"]\n',
            encoding="utf-8",
        )
        return path

    def test_formats_macos_and_windows_runner_roots(self) -> None:
        model = {"key": "sa3-test", "revision": "a" * 40}
        self.assertEqual(
            EXPORT.snapshot_path("/Users/runner/work/_temp", model),
            "/Users/runner/work/_temp/model-snapshots/sa3-test/" + "a" * 40,
        )
        self.assertEqual(
            EXPORT.snapshot_path(r"D:\actions\_temp", model),
            "D:\\actions\\_temp\\model-snapshots\\sa3-test\\" + "a" * 40,
        )

    def test_manifest_revision_changes_the_exported_path(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            directory = Path(temp)
            first = EXPORT.environment_assignments(
                self.manifest(directory, "a" * 40), ["sa3-test"], "/tmp/runner"
            )
            second = EXPORT.environment_assignments(
                self.manifest(directory, "b" * 40), ["sa3-test"], "/tmp/runner"
            )
        self.assertNotEqual(first, second)
        self.assertTrue(first[0].endswith("/" + "a" * 40))
        self.assertTrue(second[0].endswith("/" + "b" * 40))

    def test_invalid_model_key_fails_closed(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            manifest = self.manifest(Path(temp))
            with self.assertRaisesRegex(RuntimeError, "expected one model policy"):
                EXPORT.environment_assignments(manifest, ["missing"], "/tmp/runner")

    def test_main_appends_assignments_to_github_env(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            directory = Path(temp)
            manifest = self.manifest(directory)
            github_env = directory / "github-env"
            argv = [
                str(SCRIPT),
                "--model",
                "sa3-test",
                "--manifest",
                str(manifest),
                "--runner-temp",
                "/runner/temp",
                "--github-env",
                str(github_env),
            ]
            with mock.patch("sys.argv", argv):
                self.assertEqual(EXPORT.main(), 0)
            self.assertEqual(
                github_env.read_text(encoding="utf-8"),
                "SA3_TEST_SNAPSHOT=/runner/temp/model-snapshots/sa3-test/"
                + "a" * 40
                + "\n",
            )


if __name__ == "__main__":
    unittest.main()
