import contextlib
import importlib.util
import io
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock


ROOT = Path(__file__).resolve().parents[2]


def load_gate_module():
    spec = importlib.util.spec_from_file_location(
        "check_review_findings", ROOT / "scripts" / "check-review-findings.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class ReviewFindingIdTests(unittest.TestCase):
    def setUp(self) -> None:
        self.gate = load_gate_module()
        self.root = Path(self.enterContext(tempfile.TemporaryDirectory()))
        for allocation in self.gate.LEGACY_ALLOCATIONS:
            self.write_review(
                allocation.document, range(allocation.start, allocation.end + 1)
            )
        self.write_registry(list(self.gate.LEGACY_ALLOCATIONS))

    def write_review(self, relative: Path | str, numbers) -> None:
        path = self.root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            "# Review\n\n"
            + "\n".join(f"#### [F-{number:03d}] Finding {number}" for number in numbers)
            + "\n",
            encoding="utf-8",
        )

    def write_registry(self, allocations) -> str:
        text = "# kind<TAB>start<TAB>end<TAB>review document\n" + "".join(
            f"{allocation.kind}\t{allocation.start}\t{allocation.end}\t"
            f"{allocation.document.as_posix()}\n"
            for allocation in allocations
        )
        path = self.root / self.gate.REGISTRY
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text, encoding="utf-8")
        return text

    def register(self, start: int, end: int, document: str):
        allocations, errors = self.gate.parse_registry(
            (self.root / self.gate.REGISTRY).read_text(encoding="utf-8")
        )
        self.assertEqual(errors, [])
        allocations.append(self.gate.Allocation("review", start, end, Path(document)))
        return self.write_registry(allocations)

    def run_main(self, base: str) -> tuple[int, str, str]:
        stdout = io.StringIO()
        stderr = io.StringIO()
        with (
            mock.patch.object(self.gate, "ROOT", self.root),
            mock.patch.object(sys, "argv", ["check-review-findings.py", "--base", base]),
            contextlib.redirect_stdout(stdout),
            contextlib.redirect_stderr(stderr),
        ):
            result = self.gate.main()
        return result, stdout.getvalue(), stderr.getvalue()

    def test_the_live_repository_allocation_is_valid(self) -> None:
        self.assertEqual(self.gate.validate(ROOT), [])

    def test_next_contiguous_global_allocations_pass(self) -> None:
        self.write_review("docs/CODE_REVIEW_2026-08-01.md", [182, 183])
        self.register(182, 183, "docs/CODE_REVIEW_2026-08-01.md")
        self.assertEqual(self.gate.validate(self.root), [])

    def test_parallel_reviews_cannot_allocate_the_same_id(self) -> None:
        self.write_review("docs/CODE_REVIEW_2026-08-01.md", [182])
        self.write_review("crates/media/CODE_REVIEW_2026-08-02.md", [182])
        self.register(182, 182, "docs/CODE_REVIEW_2026-08-01.md")
        self.register(182, 182, "crates/media/CODE_REVIEW_2026-08-02.md")
        errors = self.gate.validate(self.root)
        self.assertTrue(any("expected F-183" in error for error in errors), errors)

    def test_new_review_cannot_restart_the_legacy_sequence(self) -> None:
        self.write_review("docs/CODE_REVIEW_2026-08-01.md", [1])
        self.register(1, 1, "docs/CODE_REVIEW_2026-08-01.md")
        errors = self.gate.validate(self.root)
        self.assertTrue(any("expected F-182" in error for error in errors), errors)

    def test_new_sequence_cannot_skip_an_id(self) -> None:
        self.write_review("docs/CODE_REVIEW_2026-08-01.md", [183])
        self.register(183, 183, "docs/CODE_REVIEW_2026-08-01.md")
        errors = self.gate.validate(self.root)
        self.assertTrue(any("expected F-182" in error for error in errors), errors)

    def test_parser_supports_more_than_three_digits(self) -> None:
        text = "#### [F-999] Last three-digit id\n#### [F-1000] First four-digit id\n"
        self.assertEqual(self.gate.finding_ids(text), [999, 1000])

    def test_legacy_allocation_is_frozen(self) -> None:
        allocation = self.gate.LEGACY_ALLOCATIONS[0]
        self.write_review(allocation.document, [1])
        errors = self.gate.validate(self.root)
        self.assertTrue(any("allocation mismatch" in error for error in errors), errors)

    def test_deleting_the_highest_heading_does_not_lower_the_high_water(self) -> None:
        document = "docs/CODE_REVIEW_2026-08-01.md"
        self.write_review(document, [182, 183])
        self.register(182, 183, document)
        self.write_review(document, [182])
        errors = self.gate.validate(self.root)
        self.assertTrue(any("registry requires exactly F-182..F-183" in error for error in errors), errors)

    def test_deleting_all_new_allocations_is_rejected_against_the_base(self) -> None:
        document = "docs/CODE_REVIEW_2026-08-01.md"
        self.write_review(document, [182])
        base_registry = self.register(182, 182, document)
        (self.root / document).unlink()
        self.write_registry(list(self.gate.LEGACY_ALLOCATIONS))
        errors = self.gate.validate(self.root, base_registry_text=base_registry)
        self.assertTrue(any("append-only" in error for error in errors), errors)

    def test_missing_base_warns_and_validates_the_current_registry(self) -> None:
        # --verify --quiet intentionally leaves stderr empty for a missing object.
        unresolved = subprocess.CompletedProcess(
            ["git", "rev-parse", "--verify", "--quiet"],
            1,
            stdout="",
            stderr="",
        )
        with mock.patch.object(self.gate.subprocess, "run", return_value=unresolved) as run:
            result, stdout, stderr = self.run_main("force-pushed-base")

        self.assertEqual(result, 0)
        self.assertIn("review finding ids: OK", stdout)
        self.assertIn("warning: unable to resolve base revision 'force-pushed-base'", stderr)
        self.assertIn("validating the current tree", stderr)
        self.assertEqual(run.call_count, 1)
        self.assertEqual(
            run.call_args.args[0][:4], ["git", "rev-parse", "--verify", "--quiet"]
        )

    def test_missing_base_does_not_suppress_current_registry_errors(self) -> None:
        (self.root / self.gate.LEGACY_ALLOCATIONS[0].document).unlink()
        unresolved = subprocess.CompletedProcess(
            ["git", "rev-parse", "--verify", "--quiet"],
            1,
            stdout="",
            stderr="",
        )
        with mock.patch.object(self.gate.subprocess, "run", return_value=unresolved):
            result, stdout, stderr = self.run_main("garbage-collected-base")

        self.assertEqual(result, 1)
        self.assertEqual(stdout, "")
        self.assertIn("warning: unable to resolve base revision", stderr)
        self.assertIn("registered review document is missing", stderr)

    def test_reachable_base_still_enforces_append_only_history(self) -> None:
        document = "docs/CODE_REVIEW_2026-08-01.md"
        self.write_review(document, [182])
        base_registry = self.register(182, 182, document)
        (self.root / document).unlink()
        self.write_registry(list(self.gate.LEGACY_ALLOCATIONS))
        resolved = subprocess.CompletedProcess(["git", "rev-parse"], 0)
        shown = subprocess.CompletedProcess(["git", "show"], 0, stdout=base_registry)
        with mock.patch.object(
            self.gate.subprocess, "run", side_effect=[resolved, shown]
        ) as run:
            result, stdout, stderr = self.run_main("reachable-base")

        self.assertEqual(result, 1)
        self.assertEqual(stdout, "")
        self.assertNotIn("warning:", stderr)
        self.assertIn("allocation registry is append-only", stderr)
        self.assertEqual(run.call_count, 2)
        self.assertEqual(run.call_args_list[1].args[0][:2], ["git", "show"])

    def test_reachable_base_registry_read_failure_does_not_fall_back(self) -> None:
        resolved = subprocess.CompletedProcess(["git", "rev-parse"], 0)
        show_failure = subprocess.CalledProcessError(128, ["git", "show"])
        with mock.patch.object(
            self.gate.subprocess, "run", side_effect=[resolved, show_failure]
        ):
            result, stdout, stderr = self.run_main("reachable-base")

        self.assertEqual(result, 1)
        self.assertEqual(stdout, "")
        self.assertNotIn("warning:", stderr)
        self.assertIn("error: unable to inspect base revision", stderr)

    def test_base_resolution_usage_failure_does_not_fall_back(self) -> None:
        usage_failure = subprocess.CompletedProcess(
            ["git", "rev-parse"],
            129,
            stdout="",
            stderr="usage: git rev-parse [<options>] <args>...",
        )
        with mock.patch.object(
            self.gate.subprocess, "run", return_value=usage_failure
        ):
            result, stdout, stderr = self.run_main("requested-base")

        self.assertEqual(result, 1)
        self.assertEqual(stdout, "")
        self.assertNotIn("warning:", stderr)
        self.assertIn("error: unable to inspect base revision", stderr)
        self.assertIn("exit status 129", stderr)

    def test_base_resolution_subprocess_error_does_not_fall_back(self) -> None:
        with mock.patch.object(
            self.gate.subprocess, "run", side_effect=OSError("git is unavailable")
        ):
            result, stdout, stderr = self.run_main("requested-base")

        self.assertEqual(result, 1)
        self.assertEqual(stdout, "")
        self.assertNotIn("warning:", stderr)
        self.assertIn("error: unable to inspect base revision", stderr)
        self.assertIn("git is unavailable", stderr)


if __name__ == "__main__":
    unittest.main()
