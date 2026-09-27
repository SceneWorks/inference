"""Execute the real-weights dispatch steps with isolated side effects (sc-24253)."""

import os
from pathlib import Path
import subprocess
import tempfile
import unittest

import yaml


WORKFLOW = Path(__file__).resolve().parents[2] / ".github/workflows/real-weights.yml"
REVISION = "a" * 40


def workflow_step(job_name, step_name):
    workflow = yaml.safe_load(WORKFLOW.read_text(encoding="utf-8"))
    steps = workflow["jobs"][job_name]["steps"]
    return next(step for step in steps if step.get("name") == step_name)


def workflow_steps(job_name):
    workflow = yaml.safe_load(WORKFLOW.read_text(encoding="utf-8"))
    return workflow["jobs"][job_name]["steps"]


class RealWeightsDispatchInputTests(unittest.TestCase):
    def run_step(self, step, *, environment):
        return subprocess.run(
            ["bash", "--noprofile", "--norc", "-e", "-o", "pipefail", "-c", step["run"]],
            env={**os.environ, **environment},
            text=True,
            capture_output=True,
            check=False,
        )

    def test_confirmation_steps_execute_the_exact_script_with_untrusted_input(self):
        for job in ("mlx-chroma-auxiliary", "mlx-chroma-publish"):
            with self.subTest(job=job), tempfile.TemporaryDirectory() as directory:
                step = workflow_step(job, "Validate publication authorization")
                self.assertEqual(workflow_steps(job)[0], step)
                self.assertEqual(
                    step["env"]["CHROMA_PUBLISH_CONFIRMATION"],
                    "${{ inputs.chroma_publish_confirmation }}",
                )
                marker = Path(directory) / "injected"
                for value in (
                    f"$(touch {marker})",
                    f'"; touch {marker}; : "',
                    "--upload-pack=malicious",
                    "sc-16462 extra",
                    "",
                ):
                    with self.subTest(value=value):
                        result = self.run_step(
                            step, environment={"CHROMA_PUBLISH_CONFIRMATION": value}
                        )
                        self.assertNotEqual(result.returncode, 0)
                        self.assertFalse(marker.exists())
                result = self.run_step(
                    step, environment={"CHROMA_PUBLISH_CONFIRMATION": "sc-16462"}
                )
                self.assertEqual(result.returncode, 0, result.stderr)

    def test_revision_rejected_before_any_git_side_effect_and_valid_sha_is_preserved(self):
        step = workflow_step(
            "mlx-memory-evidence-v1",
            "Resolve and bind exact SceneWorks provenance outside the inference worktree",
        )
        self.assertEqual(
            step["env"]["REQUESTED_SCENEWORKS_REVISION"],
            "${{ inputs.sceneworks_revision }}",
        )
        preflight = workflow_steps("mlx-memory-evidence-v1")[0]
        self.assertEqual(preflight["name"], "Validate SceneWorks revision input")
        self.assertEqual(
            preflight["env"],
            {"REQUESTED_SCENEWORKS_REVISION": "${{ inputs.sceneworks_revision }}"},
        )
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            marker = root / "injected"
            git_log = root / "git.log"
            github_env = root / "github-env"
            fake_git = root / "git"
            fake_git.write_text(
                '#!/bin/bash\n'
                'printf "%s\\n" "$*" >> "$GIT_LOG"\n'
                'if [[ "$*" == *"rev-parse HEAD" ]]; then printf "%s\\n" "$RESOLVED_REVISION"; fi\n',
                encoding="utf-8",
            )
            fake_git.chmod(0o755)
            base_env = {
                "PATH": f"{root}:{os.environ['PATH']}",
                "GIT_LOG": str(git_log),
                "GITHUB_ENV": str(github_env),
                "SCENEWORKS_PROVENANCE_ROOT": str(root / "provenance"),
                "RESOLVED_REVISION": REVISION,
            }
            for value in (
                "",
                "--upload-pack=malicious",
                f"$(touch {marker})",
                f'"; touch {marker}; : "',
                "A" * 40,
                REVISION + "extra",
            ):
                with self.subTest(value=value):
                    preflight_result = self.run_step(
                        preflight,
                        environment={**base_env, "REQUESTED_SCENEWORKS_REVISION": value},
                    )
                    self.assertNotEqual(preflight_result.returncode, 0)
                    result = self.run_step(
                        step, environment={**base_env, "REQUESTED_SCENEWORKS_REVISION": value}
                    )
                    self.assertNotEqual(result.returncode, 0)
                    self.assertFalse(git_log.exists())
                    self.assertFalse(github_env.exists())
                    self.assertFalse(marker.exists())

            preflight_result = self.run_step(
                preflight,
                environment={**base_env, "REQUESTED_SCENEWORKS_REVISION": REVISION},
            )
            self.assertEqual(preflight_result.returncode, 0, preflight_result.stderr)
            result = self.run_step(
                step, environment={**base_env, "REQUESTED_SCENEWORKS_REVISION": REVISION}
            )
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertIn(f"fetch --depth=1 -- origin {REVISION}", git_log.read_text())
            self.assertEqual(github_env.read_text(), f"SCENEWORKS_REVISION={REVISION}\n")

            git_log.unlink()
            github_env.unlink()
            result = self.run_step(
                step,
                environment={
                    **base_env,
                    "REQUESTED_SCENEWORKS_REVISION": REVISION,
                    "RESOLVED_REVISION": "b" * 40,
                },
            )
            self.assertNotEqual(result.returncode, 0)
            self.assertFalse(github_env.exists())


if __name__ == "__main__":
    unittest.main()
