"""Gate for `scripts/ci/s18_memory_preflight.py`.

The preflight exists because sc-17324 took `nax-macos-2` down twice with row F. Its whole value is
that it OVER-predicts: a guard that under-predicts is worse than no guard, because it hands back a
green light for the row that kills the runner and destroys every cell measured so far with it. So
the load-bearing assertion here is per-row accuracy against the real measured peaks, with the error
required to be on the safe side.
"""

from __future__ import annotations

import importlib.util
import unittest
from pathlib import Path

MODULE_PATH = Path(__file__).resolve().parents[1] / "ci" / "s18_memory_preflight.py"
_spec = importlib.util.spec_from_file_location("s18_memory_preflight", MODULE_PATH)
preflight = importlib.util.module_from_spec(_spec)
assert _spec.loader is not None
_spec.loader.exec_module(preflight)

# MLX active peaks measured on nax-macos in run 30787887176 at 832x480, 45 latent frames, q4.
# These are observations, not expectations — do not "fix" the model by editing them.
MEASURED_832 = {
    "A": 17.43,
    "B": 18.94,
    "C": 21.56,
    "D": 29.41,
    "F": 49.14,
    "E": 63.32,
    "Z": 13.74,
}


class S18MemoryPreflightTests(unittest.TestCase):
    def test_predictions_never_fall_below_the_measured_peak(self) -> None:
        """Under-prediction is the failure mode that crashes a box. Over-prediction only costs a row."""
        for row, measured in MEASURED_832.items():
            predicted = preflight.predicted_peak_gib(row, 832, 480, 45)
            self.assertGreaterEqual(
                predicted,
                measured,
                f"row {row}: predicted {predicted:.2f} GiB is BELOW the measured {measured:.2f} GiB — "
                "the guard would wave through a row that does not fit",
            )

    def test_predictions_are_tight_enough_to_be_useful(self) -> None:
        """A guard that over-predicts wildly refuses rows that would have been fine.

        Rows A-F are held to 10%. Row Z is exempt: its clip is six latent frames total, so its
        activations and decode are far smaller than the window-derived estimate implies, and the
        model deliberately does not special-case that — it is the cheapest row either way.
        """
        for row in "ABCDFE":
            predicted = preflight.predicted_peak_gib(row, 832, 480, 45)
            measured = MEASURED_832[row]
            self.assertLessEqual(
                (predicted - measured) / measured,
                0.10,
                f"row {row}: predicted {predicted:.2f} GiB is >10% above measured {measured:.2f}",
            )

    def test_the_sink_anchor_is_counted(self) -> None:
        """Rows B and C pin first-chunk KV permanently, on top of the read window.

        Omitting it under-predicted row C by 17.6% — caught only because the prediction was checked
        against every measured row rather than the interesting ones.
        """
        a = preflight.window_tokens("A", 832, 480, 45)
        self.assertEqual(preflight.window_tokens("B", 832, 480, 45), a + 1560)
        self.assertEqual(preflight.window_tokens("C", 832, 480, 45), a + 3 * 1560)

    def test_geometry_drives_the_estimate(self) -> None:
        """640x384 is the documented fallback bucket for the rows that do not fit at 832x480."""
        self.assertEqual(preflight.tokens_per_latent_frame(832, 480), 1560)
        self.assertEqual(preflight.tokens_per_latent_frame(640, 384), 960)
        # Recorded 640x384 row F peak: 36,210,546,832 bytes = 33.72 GiB.
        predicted = preflight.predicted_peak_gib("F", 640, 384, 45)
        self.assertGreaterEqual(predicted, 33.72)
        self.assertLessEqual(predicted, 33.72 * 1.10)

    def test_row_e_spans_the_whole_clip_and_row_z_is_capped_by_its_own(self) -> None:
        # Row E is the global window: no eviction, so its KV is the entire clip.
        self.assertEqual(preflight.window_tokens("E", 832, 480, 45), 45 * 1560)
        # Row Z's clip is 6 latent frames, so its 6-frame window cannot exceed it.
        self.assertEqual(preflight.window_tokens("Z", 832, 480, 45), 6 * 1560)

    def test_the_budget_reproduces_what_each_HOST_actually_did(self) -> None:
        """Anchored on four real outcomes across two boxes, not on a round number.

        This is why the budget subtracts a fixed overhead rather than taking a fraction of RAM: a
        fraction cannot be right on both hosts at once. At 40% of RAM the guard refuses row E on
        the 128 GiB dev Mac, where run 30787887176 ran it to completion at 63.32 GiB.
        """

        def budget(ram: float) -> float:
            return ram - preflight.NON_MLX_OVERHEAD_GIB - preflight.SAFETY_MARGIN_GIB

        peak = lambda row, w=832, h=480: preflight.predicted_peak_gib(row, w, h, 45)

        # nax-macos-2, ~101 GiB: row D completed, row F took the runner down TWICE.
        self.assertLess(peak("D"), budget(101.0))
        self.assertGreater(peak("F"), budget(101.0), "row F must be refused on the box it killed")

        # nax-macos, 128 GiB: run 30787887176 completed all seven rows, so the bounded ones must
        # not be refused there.
        for row in "ABCDFZ":
            self.assertLess(
                peak(row), budget(128.0), f"row {row} completed on the 128 GiB host; do not refuse it"
            )

        # Row E is the DELIBERATE exception, and it is a false refusal by design. It completed on
        # that host at 63.32 GiB against a 63.0 GiB budget — i.e. it fit with 0.3 GiB to spare, and
        # only because the model over-predicts it by 8.7% does the guard say no. Keeping the refusal
        # is the right trade: E is the global-window row, the verdict rule refuses to use it as
        # attribution evidence at all, and sc-15571 records it driving a host into enough swap to
        # fill the boot volume even at the smaller bucket. An operator who genuinely wants it can
        # pass --margin-gib 0. Do not "fix" this by shrinking the margin: the margin is what covers
        # a two-sample overhead estimate.
        self.assertGreater(peak("E"), budget(128.0))
        self.assertLess(peak("E"), budget(128.0) + preflight.SAFETY_MARGIN_GIB + 2.0)

        # ...and the fallback bucket is what makes row F reachable on the smaller box at all.
        self.assertLess(peak("F", 640, 384), budget(101.0))

    def test_exit_codes(self) -> None:
        import contextlib
        import io

        def run(argv: list[str]) -> int:
            import sys

            old = sys.argv
            sys.argv = ["s18_memory_preflight.py", *argv]
            try:
                with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(
                    io.StringIO()
                ):
                    return preflight.main()
            finally:
                sys.argv = old

        self.assertEqual(run(["--rows", "ABCDZ", "--ram-gib", "101"]), 0)
        self.assertEqual(run(["--rows", "F", "--ram-gib", "101"]), 1, "row F must be refused")
        self.assertEqual(
            run(["--rows", "F", "--ram-gib", "101", "--width", "640", "--height", "384"]),
            0,
            "row F must be allowed at the fallback bucket",
        )
        self.assertEqual(run(["--rows", "AQ", "--ram-gib", "101"]), 1, "unknown row is an error")
        # An unmeasurable host must not block the lane.
        self.assertEqual(run(["--rows", "F", "--ram-gib", "0"]), 0)


if __name__ == "__main__":
    unittest.main()
