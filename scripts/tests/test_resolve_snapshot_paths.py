import re
import unittest
from pathlib import Path

import yaml

from scripts.release.resolve_snapshot_paths import resolve


WORKFLOW = Path(__file__).resolve().parents[2] / ".github/workflows/real-weights.yml"
RESOLVE_STEP = "Resolve runner-local snapshot paths"


class ResolveTests(unittest.TestCase):
    def test_leading_tilde_resolves_against_the_runner_home(self) -> None:
        self.assertEqual(
            resolve("~/.cache/huggingface/hub/models--x/snapshots/y", "/Users/MTrefry"),
            "/Users/MTrefry/.cache/huggingface/hub/models--x/snapshots/y",
        )

    def test_a_path_containing_spaces_survives(self) -> None:
        self.assertEqual(
            resolve("~/Library/Application Support/SceneWorks/oracles/x", "/Users/mt"),
            "/Users/mt/Library/Application Support/SceneWorks/oracles/x",
        )

    def test_absolute_posix_values_pass_through_untouched(self) -> None:
        """This is what lets each variable be flipped to `~/` on its own, with no cutover."""
        value = "/Users/michael/.cache/huggingface/hub/models--x/snapshots/y"
        self.assertEqual(resolve(value, "/Users/MTrefry"), value)

    def test_absolute_windows_values_pass_through_untouched(self) -> None:
        """The CUDA pool's `E:\\huggingface\\hub` values must never be rewritten."""
        value = r"E:\huggingface\hub\models--x\snapshots\y"
        self.assertEqual(resolve(value, "/Users/MTrefry"), value)

    def test_a_bare_tilde_and_an_interior_tilde_are_distinguished(self) -> None:
        self.assertEqual(resolve("~", "/Users/mt"), "/Users/mt")
        # Only a LEADING `~/` is a home reference; `~` inside a path is a literal character.
        self.assertEqual(resolve("/srv/we~ird/x", "/Users/mt"), "/srv/we~ird/x")


class WorkflowWiringTests(unittest.TestCase):
    """Every macOS job that reads a snapshot variable must resolve it before consuming it.

    A job that reads one of these variables without the step is not a loud failure — on the
    box whose home matches the variable it passes, and it only breaks once the OTHER macOS
    runner claims that job. That is exactly the kind of gap a scheduled weekly lane hides,
    so it is asserted here rather than left to review.
    """

    def setUp(self) -> None:
        self.workflow = yaml.safe_load(WORKFLOW.read_text(encoding="utf-8"))
        self.raw = WORKFLOW.read_text(encoding="utf-8")

    def macos_jobs_reading_variables(self) -> dict[str, set[str]]:
        found: dict[str, set[str]] = {}
        for name, job in self.workflow["jobs"].items():
            runs_on = job.get("runs-on") or []
            if "macOS" not in runs_on:
                continue
            names = set(re.findall(r"vars\.([A-Z0-9_]+)", yaml.dump(job)))
            if names:
                found[name] = names
        return found

    def test_every_macos_job_reading_variables_resolves_them(self) -> None:
        for name, variables in self.macos_jobs_reading_variables().items():
            with self.subTest(job=name):
                steps = self.workflow["jobs"][name]["steps"]
                resolvers = [s for s in steps if s.get("name") == RESOLVE_STEP]
                self.assertEqual(
                    len(resolvers), 1, f"{name} must carry exactly one {RESOLVE_STEP!r} step"
                )
                resolved = set(re.findall(r"[A-Z0-9_]{4,}", resolvers[0]["run"]))
                missing = variables - resolved
                self.assertFalse(
                    missing, f"{name} reads {sorted(missing)} without resolving them"
                )

    def test_the_resolve_step_precedes_every_consumer(self) -> None:
        for name in self.macos_jobs_reading_variables():
            with self.subTest(job=name):
                steps = self.workflow["jobs"][name]["steps"]
                index = next(
                    i for i, s in enumerate(steps) if s.get("name") == RESOLVE_STEP
                )
                for step in steps[:index]:
                    body = yaml.dump(step)
                    self.assertNotIn(
                        "SNAPSHOT",
                        body.upper().replace("RESOLVE RUNNER-LOCAL SNAPSHOT PATHS", ""),
                        f"{name} consumes a snapshot path before resolving it",
                    )

    def test_windows_jobs_are_left_alone(self) -> None:
        """The CUDA pool's values are absolute Windows paths; resolving them is a no-op."""
        for name, job in self.workflow["jobs"].items():
            runs_on = job.get("runs-on") or []
            if "windows" not in runs_on:
                continue
            with self.subTest(job=name):
                names = [s.get("name") for s in job.get("steps", [])]
                self.assertNotIn(RESOLVE_STEP, names)


if __name__ == "__main__":
    unittest.main()
