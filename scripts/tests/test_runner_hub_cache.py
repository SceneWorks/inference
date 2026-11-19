import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

SCRIPT = Path(__file__).resolve().parents[1] / "release" / "runner_hub_cache.py"


class RunnerHubCacheScriptTests(unittest.TestCase):
    def test_prints_the_cache_the_runner_environment_names(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            environment = {
                name: value
                for name, value in os.environ.items()
                if name not in ("HF_HUB_CACHE", "HUGGINGFACE_HUB_CACHE", "HF_HOME")
            }
            environment["HF_HOME"] = temp
            result = subprocess.run(
                [sys.executable, str(SCRIPT)],
                env=environment,
                capture_output=True,
                text=True,
                encoding="utf-8",
                check=True,
            )
        self.assertEqual(result.stdout, f"{Path(os.path.abspath(temp)) / 'hub'}\n")


if __name__ == "__main__":
    unittest.main()
