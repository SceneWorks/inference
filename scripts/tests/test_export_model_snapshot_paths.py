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

    def test_formats_macos_and_windows_hub_cache_layouts(self) -> None:
        # The snapshot lives in the runner's Hugging Face hub cache layout, never RUNNER_TEMP, so a
        # complete snapshot is reused by every later run instead of being fetched again.
        model = {"key": "sa3-test", "repository": "example/test", "revision": "a" * 40}
        self.assertEqual(
            EXPORT.snapshot_path("/Volumes/Models/huggingface/hub", model),
            "/Volumes/Models/huggingface/hub/models--example--test/snapshots/" + "a" * 40,
        )
        self.assertEqual(
            EXPORT.snapshot_path(r"E:\HuggingFace\hub", model),
            "E:\\HuggingFace\\hub\\models--example--test\\snapshots\\" + "a" * 40,
        )

    def test_default_hub_cache_follows_the_runner_environment(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            directory = Path(temp)
            manifest = self.manifest(directory)
            github_env = directory / "github-env"
            argv = [str(SCRIPT), "--model", "sa3-test", "--manifest", str(manifest),
                    "--github-env", str(github_env)]
            hub = directory / "hf-home" / "hub"
            with mock.patch.dict("os.environ", {"HF_HUB_CACHE": str(hub)}), \
                    mock.patch("sys.argv", argv):
                self.assertEqual(EXPORT.main(), 0)
            self.assertEqual(
                github_env.read_text(encoding="utf-8"),
                f"SA3_TEST_SNAPSHOT={hub}/models--example--test/snapshots/{'a' * 40}\n",
            )

    def test_manifest_revision_changes_the_exported_path(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            directory = Path(temp)
            first = EXPORT.environment_assignments(
                self.manifest(directory, "a" * 40), ["sa3-test"], "/tmp/hub"
            )
            second = EXPORT.environment_assignments(
                self.manifest(directory, "b" * 40), ["sa3-test"], "/tmp/hub"
            )
        self.assertNotEqual(first, second)
        self.assertTrue(first[0].endswith("/" + "a" * 40))
        self.assertTrue(second[0].endswith("/" + "b" * 40))

    def test_invalid_model_key_fails_closed(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            manifest = self.manifest(Path(temp))
            with self.assertRaisesRegex(RuntimeError, "expected one model policy"):
                EXPORT.environment_assignments(manifest, ["missing"], "/tmp/hub")

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
                "--hub-cache",
                "/runner/hub",
                "--github-env",
                str(github_env),
            ]
            with mock.patch("sys.argv", argv):
                self.assertEqual(EXPORT.main(), 0)
            self.assertEqual(
                github_env.read_text(encoding="utf-8"),
                "SA3_TEST_SNAPSHOT=/runner/hub/models--example--test/snapshots/"
                + "a" * 40
                + "\n",
            )


if __name__ == "__main__":
    unittest.main()
