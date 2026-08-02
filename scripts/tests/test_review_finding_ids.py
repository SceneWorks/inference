import importlib.util
import tempfile
import unittest
from pathlib import Path


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


if __name__ == "__main__":
    unittest.main()
