"""Enforce that `candle-gen*` integration targets execute in an automatic CI lane.

sc-19447. Every `candle-gen*` integration target used to compile in three lanes and execute in
none of them: `candle-cpu-test` ran `--lib` only, `candle-cpu-lint` reached them through
all-target Clippy without running anything, `windows-cuda-check` built them `--no-run`, and the
only `--tests` runner for the family — `windows-cuda` — is `workflow_dispatch`-only. A whole
epic's candle-side parity evidence was green because it was never evaluated.

The fix is a glob (`cargo test --lib --tests -p 'candle-gen*'`) rather than a curated
`--test` list, so a newly added target is selected the moment it exists. This check pins the two
properties that make that true, because both are one careless edit away from silently reverting:

  * the step still passes `--tests` (dropping it restores the original defect verbatim), and
  * its package selectors still cover every `candle-gen*` crate that carries at least one
    non-`#[ignore]`d integration case (narrowing the glob to a hand-listed set restores the drift
    the audio side already suffered — see `test_sa3_ci_target_coverage.py`).

Scope caveat, recorded so a green run is not over-read: this policies the **weight-free** half.
Cases that need real weights are `#[ignore]`d or sit behind a file-level
`#![cfg(feature = "cuda")]`; they are selected by the manual `windows-cuda` dispatch and by
`real-weights.yml`, and many of them are named by no lane at all. Wiring those is a separate
question that sc-19447 records rather than solves.
"""

import fnmatch
import re
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
WORKFLOW = ROOT / ".github" / "workflows" / "ci.yml"
CANDLE_GEN = ROOT / "crates" / "media" / "candle-gen"
STEP_NAME = "Test Candle gen packages and integration targets"

# The story that made this concrete: its integration targets are the epic's entire candle-side
# evidence surface, so it is asserted by name as well as by the general rule.
EPIC_CRATE = "candle-gen-minimax-h3"


def step_run_block(workflow: str, step_name: str = STEP_NAME) -> str:
    """Return the `run:` scalar of the named step.

    Only the `run:` block is returned: the step carries a long `#` comment block that names
    `--tests` and `candle-gen*` in prose, and matching those would let the real command drift
    while this check stayed green. Prose is not wiring.
    """
    lines = workflow.splitlines()
    marker = f"- name: {step_name}"
    start = next((i for i, line in enumerate(lines) if line.strip() == marker), None)
    if start is None:
        raise AssertionError(f"missing ci.yml step: {step_name}")

    indent = len(lines[start]) - len(lines[start].lstrip())
    body: list[str] = []
    for line in lines[start + 1 :]:
        if line.strip() and (len(line) - len(line.lstrip())) <= indent:
            break
        body.append(line)

    run_at = next((i for i, line in enumerate(body) if line.strip().startswith("run:")), None)
    if run_at is None:
        raise AssertionError(f"ci.yml step has no run: block: {step_name}")
    return "\n".join(body[run_at:])


def selected_packages(run_block: str) -> list[str]:
    """The `-p <spec>` selectors in a run block, quotes stripped. Globs are kept as globs."""
    return [spec.strip("'\"") for spec in re.findall(r"-p\s+('[^']+'|\"[^\"]+\"|\S+)", run_block)]


def crates_with_weight_free_integration_cases() -> dict[str, list[str]]:
    """Map each `candle-gen*` crate to the integration targets this lane can actually execute.

    A target is counted when it has at least one `#[test]` whose contiguous attribute run carries
    no `#[ignore]`, AND it is not switched off wholesale by a file-level `#![cfg(...)]` naming a
    GPU backend feature — a cuda-gated file compiles to an empty binary on a CPU runner, so
    requiring the lane to "cover" it would assert nothing.
    """
    gated = re.compile(r'#!\[cfg\(.*(?:"cuda"|"metal").*\)\]')
    found: dict[str, list[str]] = {}
    for crate in sorted(p for p in CANDLE_GEN.iterdir() if (p / "tests").is_dir()):
        for source in sorted((crate / "tests").glob("*.rs")):
            text = source.read_text(encoding="utf-8")
            if gated.search(text):
                continue
            live = 0
            run: list[str] = []
            for raw in text.splitlines():
                line = raw.strip()
                if line.startswith("#["):
                    run.append(line)
                    continue
                if run:
                    is_case = any(
                        a.startswith("#[test]") or a.startswith("#[tokio::test") for a in run
                    )
                    if is_case and not any(a.startswith("#[ignore") for a in run):
                        live += 1
                    run = []
            if live:
                found.setdefault(crate.name, []).append(source.stem)
    return found


def uncovered_crates(workflow: str) -> list[str]:
    """Crates carrying weight-free integration cases that the step does not select."""
    run_block = step_run_block(workflow)
    if "--tests" not in run_block:
        return sorted(crates_with_weight_free_integration_cases())
    specs = selected_packages(run_block)
    return sorted(
        crate
        for crate in crates_with_weight_free_integration_cases()
        if not any(fnmatch.fnmatchcase(crate, spec) for spec in specs)
    )


class CandleGenCiTargetCoverageTests(unittest.TestCase):
    def setUp(self) -> None:
        self.workflow = WORKFLOW.read_text(encoding="utf-8")

    def test_parsers_find_something(self) -> None:
        """A parse that silently yields nothing would make the real checks vacuously green."""
        self.assertTrue(CANDLE_GEN.is_dir(), f"missing candle-gen tree: {CANDLE_GEN}")
        crates = crates_with_weight_free_integration_cases()
        self.assertTrue(crates, f"parsed no weight-free candle-gen integration targets under {CANDLE_GEN}")
        self.assertIn(
            EPIC_CRATE,
            crates,
            f"{EPIC_CRATE} parsed with no weight-free integration case; the attribute parser is broken",
        )
        self.assertTrue(
            selected_packages(step_run_block(self.workflow)),
            f"parsed no -p selectors from the {STEP_NAME!r} step",
        )

    def test_the_step_actually_runs_integration_targets(self) -> None:
        run_block = step_run_block(self.workflow)
        self.assertIn(
            "--tests",
            run_block,
            f"the {STEP_NAME!r} step no longer passes --tests, so every candle-gen integration "
            "target compiles in lint lanes and executes nowhere automatic again (sc-19447).",
        )
        self.assertNotRegex(
            run_block,
            r"--no-run\b",
            f"the {STEP_NAME!r} step must EXECUTE its test binaries; --no-run reproduces the "
            "windows-cuda-check hole this step exists to close.",
        )

    def test_every_crate_with_weight_free_cases_is_selected(self) -> None:
        missing = uncovered_crates(self.workflow)
        self.assertEqual(
            missing,
            [],
            "candle-gen crates carry weight-free integration cases that no automatic lane runs: "
            + ", ".join(missing)
            + f". Widen the -p selectors on the {STEP_NAME!r} step in .github/workflows/ci.yml.",
        )

    def test_the_epic_crate_is_selected_by_name_or_glob(self) -> None:
        """AC5 of sc-19447, stated directly rather than only as a consequence of the general rule."""
        specs = selected_packages(step_run_block(self.workflow))
        self.assertTrue(
            any(fnmatch.fnmatchcase(EPIC_CRATE, spec) for spec in specs),
            f"{EPIC_CRATE} is selected by no -p spec in the {STEP_NAME!r} step (specs: {specs}).",
        )

    def test_the_coverage_check_discriminates_mutations(self) -> None:
        """Each mutation is applied ALONE — a batch would let one survivor hide behind another."""
        mutations = {
            "--tests dropped": self.workflow.replace(
                "cargo test --locked --lib --tests -j 1\n          -p 'candle-gen*'",
                "cargo test --locked --lib -j 1\n          -p 'candle-gen*'",
            ),
            "glob narrowed to one crate": self.workflow.replace(
                "cargo test --locked --lib --tests -j 1\n          -p 'candle-gen*'",
                "cargo test --locked --lib --tests -j 1\n          -p candle-gen-minimax-h3",
            ),
            "glob narrowed away from the epic crate": self.workflow.replace(
                "cargo test --locked --lib --tests -j 1\n          -p 'candle-gen*'",
                "cargo test --locked --lib --tests -j 1\n          -p 'candle-gen-s*'",
            ),
        }
        for label, mutant in mutations.items():
            with self.subTest(mutation=label):
                self.assertNotEqual(
                    mutant, self.workflow, f"mutation {label!r} matched nothing; the check is stale"
                )
                self.assertNotEqual(
                    uncovered_crates(mutant),
                    [],
                    f"mutation {label!r} left the coverage check green",
                )

    def test_the_step_parser_rejects_a_renamed_or_missing_step(self) -> None:
        with self.assertRaises(AssertionError):
            step_run_block(self.workflow, "Test Candle gen packages that do not exist")
