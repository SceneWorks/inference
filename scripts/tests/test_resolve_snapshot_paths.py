import re
import unittest
from pathlib import Path

import yaml

from scripts.release.resolve_snapshot_paths import resolve


WORKFLOW = Path(__file__).resolve().parents[2] / ".github/workflows/real-weights.yml"
RESOLVE_STEP = "Resolve runner-local snapshot paths"
REPORT_STEP = "Report runner disk headroom"
REPORT_SCRIPT = "scripts/ci/report_runner_disk_headroom.sh"
REPORT_SCRIPT_PATH = Path(__file__).resolve().parents[2] / REPORT_SCRIPT


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
        """Map each macOS job to the ENV VAR NAMES it defines from a `vars.` reference.

        Keying on the env var name rather than the repository variable name is the whole
        correctness of this check. `mlx-request-memory-scope` defines
        `MAGE_REQUEST_SCOPE_SNAPSHOT: ${{ vars.MAGE_SNAPSHOT }}` — the names differ. The
        resolver rewrites ENVIRONMENT variables, so handing it `MAGE_SNAPSHOT` there resolves
        nothing at all and the job receives a literal `~/...` path. An earlier version of this
        test compared variable names against variable names and passed while that exact bug
        was live, which is why it now derives the expected set from `env` keys.
        """
        found: dict[str, set[str]] = {}
        for name, job in self.workflow["jobs"].items():
            runs_on = job.get("runs-on") or []
            if "macOS" not in runs_on:
                continue
            names = {
                key
                for key, value in (job.get("env") or {}).items()
                if re.search(r"vars\.[A-Z0-9_]+", str(value))
            }
            if names:
                found[name] = names
        return found

    def test_no_step_reads_a_variable_directly(self) -> None:
        """A `${{ vars.X }}` inside a step is expanded by Actions, not read from the
        environment, so the resolve step cannot reach it — it would keep the literal `~/`."""
        for name, job in self.workflow["jobs"].items():
            if "macOS" not in (job.get("runs-on") or []):
                continue
            for index, step in enumerate(job.get("steps", [])):
                with self.subTest(job=name, step=index):
                    self.assertNotRegex(
                        yaml.dump(step),
                        r"vars\.[A-Z0-9_]+",
                        f"{name} step {index} reads a variable directly, bypassing the resolver",
                    )

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


def headroom_wiring_errors(workflow: dict) -> list[str]:
    """Return every way the headroom report has drifted from the resolve step it shadows.

    Kept as a free function so the mutation twin below can feed it a doctored workflow and
    prove the gate actually discriminates, rather than passing because it checks nothing.
    """
    errors: list[str] = []
    for name, job in workflow["jobs"].items():
        steps = job.get("steps", [])
        names = [step.get("name") for step in steps]
        if RESOLVE_STEP not in names:
            # No `~/`-relative snapshot to cost, so no report is owed. This is what excuses
            # the `rw-chroma` and `rw-sa3` lanes: they materialize into `RUNNER_TEMP` and
            # never touch the shared Hugging Face cache the report exists to account for.
            if REPORT_STEP in names:
                errors.append(f"{name} reports headroom without resolving any snapshot path")
            continue
        if names.count(REPORT_STEP) != 1:
            errors.append(f"{name} must carry exactly one {REPORT_STEP!r} step")
            continue

        resolve_at = names.index(RESOLVE_STEP)
        report_at = names.index(REPORT_STEP)
        report = steps[report_at]

        # Ordering: the report reads RESOLVED absolute paths, and it has to be a record of
        # what the box held BEFORE the transfer, so it sits strictly between the two.
        if not resolve_at < report_at:
            errors.append(f"{name} reports headroom before resolving the paths it reports")
        materialize = [
            i for i, step in enumerate(names) if str(step).startswith("Materialize")
        ]
        if materialize and not report_at < materialize[0]:
            errors.append(f"{name} reports headroom after the transfer it is meant to precede")

        # The two lists must name the same variables: a lane that resolves a snapshot the
        # report omits silently stops accounting for it, which is exactly the drift that
        # left `mlx-media`'s cache root and three Mage snapshots unreported when this step
        # was first written.
        resolved = steps[resolve_at]["run"].split()[2:]
        reported = report["run"].split()[1:]
        if resolved != reported:
            errors.append(
                f"{name} resolves {resolved} but reports {reported}"
            )

        if report["run"].split()[0] != REPORT_SCRIPT:
            errors.append(f"{name} does not call {REPORT_SCRIPT}")
        # Report-only means report-only: this step must never be able to red the lane.
        if report.get("continue-on-error") is not True:
            errors.append(f"{name} lets a report-only step fail the job")
    return errors


class ReportHeadroomWiringTests(unittest.TestCase):
    """Every lane that resolves a snapshot path must also record what holding it costs.

    Neither box is reachable from the other and `gh api /orgs/.../actions/runners` needs
    `admin:org`, so a run log is the only measurement of either Mac's disk anyone gets --
    and `release/real-weight-models.toml` records `size_class`, never bytes, so the cost is
    not derivable from the repo. A lane that resolves without reporting is silently exempt
    from that accounting, which is a gap only a scheduled weekly lane would hide.
    """

    def setUp(self) -> None:
        self.workflow = yaml.safe_load(WORKFLOW.read_text(encoding="utf-8"))

    def test_the_report_script_is_committed_executable(self) -> None:
        """`continue-on-error` keeps a lost exec bit from redding a lane, but the step would
        still record nothing, so the bit itself is asserted rather than assumed."""
        self.assertTrue(REPORT_SCRIPT_PATH.is_file(), f"{REPORT_SCRIPT} is missing")
        self.assertTrue(
            REPORT_SCRIPT_PATH.stat().st_mode & 0o111,
            f"{REPORT_SCRIPT} is committed without an executable bit",
        )

    def test_every_resolving_lane_reports_what_it_costs(self) -> None:
        self.assertEqual(headroom_wiring_errors(self.workflow), [])

    def test_headroom_wiring_discriminates_mutations(self) -> None:
        """The positive case alone cannot tell "the wiring is right" from "the gate is inert"."""
        import copy

        def doctor(mutate) -> dict:
            workflow = copy.deepcopy(self.workflow)
            steps = workflow["jobs"]["candle-audio-chatterbox"]["steps"]
            index = [s.get("name") for s in steps].index(REPORT_STEP)
            mutate(workflow, steps, index)
            return workflow

        def drop_step(_workflow, steps, index):
            del steps[index]

        def drop_a_variable(_workflow, steps, index):
            steps[index]["run"] = " ".join(steps[index]["run"].split()[:-1])

        def allow_failure(_workflow, steps, index):
            steps[index]["continue-on-error"] = False

        def move_after_materialize(_workflow, steps, index):
            steps.append(steps.pop(index))

        def call_something_else(_workflow, steps, index):
            steps[index]["run"] = "true " + " ".join(steps[index]["run"].split()[1:])

        def report_without_resolving(workflow, _steps, _index):
            steps = workflow["jobs"]["same-l-metal"]["steps"]
            steps.insert(0, {"name": REPORT_STEP, "run": f"{REPORT_SCRIPT} X"})

        mutations = {
            "report step deleted": drop_step,
            "a resolved variable goes unreported": drop_a_variable,
            "report-only step can red the lane": allow_failure,
            "report moved after the transfer": move_after_materialize,
            "report no longer calls the script": call_something_else,
            "a non-resolving lane reports anyway": report_without_resolving,
        }
        for mutation, mutate in mutations.items():
            with self.subTest(mutation=mutation):
                self.assertNotEqual(
                    headroom_wiring_errors(doctor(mutate)),
                    [],
                    f"the gate accepted a workflow where {mutation}",
                )


class WeightSetLabelTests(unittest.TestCase):
    """Every macOS job must select exactly one `rw-*` weight-set label.

    A job left on the old shared `real-weights` label is the failure this guards: no macOS runner
    carries that label any more, so the job does not fail — it QUEUES, silently, until someone
    cancels the run. On a weekly scheduled lane that is a week of missing coverage that still
    looks green at a glance, so it is asserted rather than reviewed.
    """

    KNOWN = {
        "rw-mage",
        "rw-sa3",
        "rw-krea",
        "rw-audio",
        "rw-llm",
        "rw-chroma",
        "rw-starvector",
    }

    def setUp(self) -> None:
        self.workflow = yaml.safe_load(WORKFLOW.read_text(encoding="utf-8"))

    def test_every_macos_job_selects_exactly_one_known_weight_set(self) -> None:
        macos = {
            n: j for n, j in self.workflow["jobs"].items() if "macOS" in (j.get("runs-on") or [])
        }
        self.assertTrue(macos, "found no macOS jobs — the label scheme moved")
        for name, job in macos.items():
            with self.subTest(job=name):
                selected = self.KNOWN.intersection(job["runs-on"])
                self.assertEqual(
                    len(selected),
                    1,
                    f"{name} selects {sorted(selected) or 'no'} weight-set label",
                )
                self.assertNotIn(
                    "real-weights",
                    job["runs-on"],
                    f"{name} still carries the retired shared macOS label; it would queue forever",
                )

    def test_every_declared_label_is_documented_with_its_host(self) -> None:
        """The header table is the only record of which box stores which set — keep it honest."""
        header = WORKFLOW.read_text(encoding="utf-8").split("\non:", 1)[0]
        used = {
            label
            for job in self.workflow["jobs"].values()
            for label in (job.get("runs-on") or [])
            if label.startswith("rw-")
        }
        for label in used:
            with self.subTest(label=label):
                self.assertIn(label, header, f"{label} is used but absent from the header table")

    def test_the_cuda_pool_keeps_its_shared_label(self) -> None:
        """Both Windows boxes share `real-weights` and already load-balance; do not split them."""
        windows = [j for j in self.workflow["jobs"].values() if "windows" in (j.get("runs-on") or [])]
        self.assertTrue(windows)
        for job in windows:
            self.assertIn("real-weights", job["runs-on"])


if __name__ == "__main__":
    unittest.main()
