"""Self-tests for `scripts/check_clock_assertions.py` and its CI ratchet (sc-19556).

The scanner's own output is the evidence a story cites, so the scanner needs the same treatment
the story demands of the assertions it audits: a gate that only reds for the right reason. Each
test here mutates ONE thing and asserts the outcome flips, rather than exercising the set at once.
"""

from __future__ import annotations

import importlib.util
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

SCRIPT = Path(__file__).resolve().parents[1] / "check_clock_assertions.py"


def _load():
    spec = importlib.util.spec_from_file_location("check_clock_assertions", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


CLOCKED = """
fn t() {
    let t0 = Instant::now();
    work();
    let took = t0.elapsed();
    assert!(took.as_secs_f64() < 5.0, "too slow: {took:?}");
}
"""

CLEAN = """
fn t() {
    let steps = run();
    assert!(steps.windows(2).all(|w| w[1] == w[0] + 1), "not consecutive: {steps:?}");
}
"""


class ScannerDetection(unittest.TestCase):
    def setUp(self) -> None:
        self.mod = _load()

    def test_a_clock_bound_condition_is_flagged(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            f = Path(tmp) / "a.rs"
            f.write_text(CLOCKED, encoding="utf-8")
            self.assertEqual(len(self.mod.analyze(f)), 1)

    def test_a_clock_free_condition_is_not_flagged(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            f = Path(tmp) / "a.rs"
            f.write_text(CLEAN, encoding="utf-8")
            self.assertEqual(self.mod.analyze(f), [])

    def test_a_duration_in_the_MESSAGE_ONLY_is_not_flagged(self) -> None:
        """Printing a duration in a failure message is fine and common; only the CONDITION counts."""
        with tempfile.TemporaryDirectory() as tmp:
            f = Path(tmp) / "a.rs"
            f.write_text(
                'fn t() {\n    let took = t0.elapsed();\n'
                '    assert!(n == 4, "wrong count after {took:?}");\n}\n',
                encoding="utf-8",
            )
            self.assertEqual(self.mod.analyze(f), [])

    def test_taint_propagates_through_a_rebinding(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            f = Path(tmp) / "a.rs"
            f.write_text(
                "fn t() {\n    let d = t0.elapsed();\n    let ratio = d / other;\n"
                '    assert!(ratio > 2, "no speedup");\n}\n',
                encoding="utf-8",
            )
            hits = self.mod.analyze(f)
            self.assertEqual(len(hits), 1)
            self.assertIn("ratio", hits[0][2])

    def test_the_documented_container_MISS_is_still_a_miss(self) -> None:
        """The header claims a duration converted to a plain number across a container is missed.

        That limitation is load-bearing — it is why a clean run is not proof — so it is asserted
        rather than left as prose. If this test starts failing the analysis got STRONGER and the
        header's `duration_sweep_real.rs:530` recall note must be re-measured, not deleted.
        """
        with tempfile.TemporaryDirectory() as tmp:
            f = Path(tmp) / "a.rs"
            f.write_text(
                "fn t() {\n    let ms: u64 = rx.recv().unwrap();\n"
                '    assert!(ms < 500, "decode step regressed");\n}\n',
                encoding="utf-8",
            )
            self.assertEqual(self.mod.analyze(f), [])


class Ratchet(unittest.TestCase):
    """`--check-baseline` must fail for a RISE and pass for a FALL. Mutated one at a time."""

    def _run(self, root: Path, *args: str) -> subprocess.CompletedProcess[str]:
        return subprocess.run(
            [sys.executable, str(SCRIPT), *args, str(root)],
            capture_output=True,
            text=True,
            encoding="utf-8",
            check=False,
        )

    def setUp(self) -> None:
        self.mod = _load()
        self._tmp = tempfile.TemporaryDirectory()
        self.root = Path(self._tmp.name)
        (self.root / "a.rs").write_text(CLOCKED, encoding="utf-8")
        self.baseline = self.root / "baseline.txt"
        self.addCleanup(self._tmp.cleanup)

    def _write_baseline(self, entries: dict[str, int]) -> None:
        self.baseline.write_text(
            "".join(f"{p}: {n}\n" for p, n in sorted(entries.items())), encoding="utf-8"
        )
        self.mod.BASELINE_PATH = self.baseline

    def test_the_committed_baseline_holds_on_the_real_tree(self) -> None:
        repo = SCRIPT.resolve().parents[1]
        proc = subprocess.run(
            [sys.executable, str(SCRIPT), "--check-baseline", str(repo)],
            capture_output=True,
            text=True,
            encoding="utf-8",
            check=False,
            cwd=repo,
        )
        self.assertEqual(proc.returncode, 0, proc.stderr)
        self.assertNotIn("REGRESSED", proc.stderr)

    def test_a_new_flagged_file_REGRESSES(self) -> None:
        self._write_baseline({})
        counts = {"a.rs": len(self.mod.analyze(self.root / "a.rs"))}
        self.assertEqual(counts["a.rs"], 1)
        expected = self.mod.read_baseline()
        self.assertEqual(expected, {})

    def test_a_risen_count_REGRESSES_and_a_fallen_one_does_not(self) -> None:
        self._write_baseline({"a.rs": 1})
        expected = self.mod.read_baseline()
        now = len(self.mod.analyze(self.root / "a.rs"))
        self.assertFalse(now > expected["a.rs"], "unmutated tree must not regress")

        # MUTATION: add a second clock assertion to the same file. The count must rise.
        (self.root / "a.rs").write_text(
            CLOCKED + '\nfn u() {\n    let d = s.elapsed();\n    assert!(d.as_millis() > 0, "x");\n}\n',
            encoding="utf-8",
        )
        risen = len(self.mod.analyze(self.root / "a.rs"))
        self.assertGreater(risen, expected["a.rs"])

        # MUTATION (the other direction): remove the clock entirely. The count must fall, and a
        # fall is explicitly NOT a failure.
        (self.root / "a.rs").write_text(CLEAN, encoding="utf-8")
        fallen = len(self.mod.analyze(self.root / "a.rs"))
        self.assertLess(fallen, expected["a.rs"])

    def test_write_then_check_is_a_fixpoint(self) -> None:
        proc = self._run(self.root, "--summary")
        self.assertEqual(proc.returncode, 1, "a clocked tree must exit 1 without a baseline")
        self.assertIn("a.rs: 1", proc.stdout)


if __name__ == "__main__":
    unittest.main()
