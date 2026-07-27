"""Enforce the Stable Audio 3 weight-free CI coverage invariant.

`ci.yml` runs the SA3 crate's integration tests by naming each target explicitly
(`--test provider --test dtype_policy ...`) rather than widening to `--tests`, because
`--tests` also links the real-weight-only binaries for zero executed cases. The cost of
that choice is that a *new* integration target, or a new non-`#[ignore]`d case added to an
unnamed one, runs in no lane at all and nobody notices. Three consecutive review cycles
across two PRs found exactly that drift, each time against a hand-maintained comment.

This check makes the comment executable: every target under the SA3 crate's `tests/` with
at least one non-`#[ignore]`d case must appear as a `--test` flag in the weight-free step.

Scope caveat: this policies the **weight-free** half only. The `#[ignore]`d real-weight
half is selected by `real-weights.yml`, where `real_snapshots` and `text_oracle` are still
named by no lane, and as of sc-14546 `dit_oracle` is named only for the one case that
story's acceptance depends on -- `real_weights_detect_conditioning_mutations_and_exercise_cfg_apg`,
in `sa3-base-identity-{metal,cuda}` -- while its other real-weight cases run nowhere.
Wiring the remainder is sc-15235's job; a green run of this check is not evidence of full
SA3 coverage.
"""

import re
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
WORKFLOW = ROOT / ".github" / "workflows" / "ci.yml"
SA3_TESTS = ROOT / "crates" / "audio" / "candle-audio-stable-audio-3" / "tests"
STEP_NAME = "Test Stable Audio 3 weight-free quality gates"


def live_case_counts() -> dict[str, int]:
    """Map each SA3 integration target to its number of non-`#[ignore]`d `#[test]` cases.

    Rust attributes attach to the item that follows them, so a case is described by the
    contiguous run of attribute lines above its `fn`. A run containing both `#[test]` and
    `#[ignore]` is a real-weight case regardless of the order the two are written in.
    """
    counts: dict[str, int] = {}
    for source in sorted(SA3_TESTS.glob("*.rs")):
        live = 0
        run: list[str] = []
        for raw in source.read_text(encoding="utf-8").splitlines():
            line = raw.strip()
            if line.startswith("#["):
                run.append(line)
                continue
            if run:
                if any(a == "#[test]" for a in run) and not any(
                    a.startswith("#[ignore") for a in run
                ):
                    live += 1
                run = []
        counts[source.stem] = live
    return counts


def named_targets() -> set[str]:
    """Extract the `--test <name>` flags from the weight-free step's `run:` block.

    Only the `run:` scalar is scanned: the step carries a long `#` comment block that also
    mentions target names, and matching those would let the real command drift silently.
    """
    lines = WORKFLOW.read_text(encoding="utf-8").splitlines()
    marker = f"- name: {STEP_NAME}"
    start = next((i for i, line in enumerate(lines) if line.strip() == marker), None)
    if start is None:
        raise AssertionError(f"missing ci.yml step: {STEP_NAME}")

    indent = len(lines[start]) - len(lines[start].lstrip())
    body: list[str] = []
    for line in lines[start + 1 :]:
        if line.strip() and (len(line) - len(line.lstrip())) <= indent:
            break
        body.append(line)

    run_at = next((i for i, line in enumerate(body) if line.strip().startswith("run:")), None)
    if run_at is None:
        raise AssertionError(f"ci.yml step has no run: block: {STEP_NAME}")
    return set(re.findall(r"--test\s+([A-Za-z0-9_]+)", "\n".join(body[run_at:])))


class Sa3CiTargetCoverageTests(unittest.TestCase):
    def test_parsers_find_something(self) -> None:
        """A parse that silently yields nothing would make the real check vacuously green."""
        counts = live_case_counts()
        self.assertTrue(SA3_TESTS.is_dir(), f"missing SA3 test directory: {SA3_TESTS}")
        self.assertTrue(counts, f"parsed no SA3 integration targets from {SA3_TESTS}")
        self.assertTrue(
            any(count > 0 for count in counts.values()),
            "parsed no non-#[ignore]d SA3 cases; the attribute parser is broken",
        )
        self.assertTrue(named_targets(), f"parsed no --test flags from the {STEP_NAME!r} step")

    def test_every_weight_free_target_runs_on_pull_requests(self) -> None:
        named = named_targets()
        missing = sorted(
            f"{target} ({count} non-#[ignore]d case{'s' if count != 1 else ''})"
            for target, count in live_case_counts().items()
            if count > 0 and target not in named
        )
        self.assertEqual(
            missing,
            [],
            "SA3 integration targets carry weight-free cases that run in no CI lane: "
            + ", ".join(missing)
            + ". Add `--test <target>` to the "
            + f"{STEP_NAME!r} step in .github/workflows/ci.yml (that step names each target "
            "explicitly, so an unnamed target runs nowhere), or mark the new cases "
            "`#[ignore]` if they need real weights.",
        )
