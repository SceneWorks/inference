"""Gate for `scripts/ci/iris_memory_probe_row.py` and the footprint guard it reads (sc-25690).

The probe jobs are report-only, so the classification IS the deliverable: a row that says MEASURED
for an allocator ceiling, a saturated card, or a run whose instrument printed nothing is a wrong
number with a trustworthy label. Each case below is one of those mislabels.
"""

from __future__ import annotations

import importlib.util
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

CI_DIR = Path(__file__).resolve().parents[1] / "ci"
MODULE_PATH = CI_DIR / "iris_memory_probe_row.py"
GUARD_PATH = CI_DIR / "iris_restore_footprint_guard.py"
_spec = importlib.util.spec_from_file_location("iris_memory_probe_row", MODULE_PATH)
row = importlib.util.module_from_spec(_spec)
assert _spec.loader is not None
_spec.loader.exec_module(row)

OK = "test result: ok. 1 passed; 0 failed; 0 ignored"
P = "[[IRIS_MEMORY_PROBE]]"


def mlx_log(
    *,
    footprint: float,
    lifetime: float | None = None,
    active: float | None = 70.0,
    cache_limit: float = 130.57,
    gc_limit: float = 109.68,
    passed: bool = True,
    stop: str | None = None,
    step_active: float | None = None,
) -> str:
    lines = [
        f"{P} mlx_limits cache_mode={'none' if cache_limit == 0 else 'default'}"
        f" cache_limit_gb={cache_limit:.2f} gc_limit_gb={gc_limit:.2f} memory_limit_gb=130.57"
        " recommended_working_set_gb=115.45"
    ]
    if step_active is not None:
        lines.append(
            f"{P} step 2/4 loss 1.0000 (3.0s) active_peak_gb={step_active:.2f}"
            " active_gb=60.00 cache_gb=40.00"
        )
    if active is not None:
        lines.append(f"{P} mlx_memory active_peak_gb={active:.2f} cache_end_gb=30.00")
    if passed:
        lines.append(OK)
    if stop:
        lines.append(f"[guard] SIGTERM: {stop}")
    lifetime = footprint if lifetime is None else lifetime
    lines.append(
        f"[guard] exit={0 if passed else -15} peak_phys_footprint={footprint:.2f} GB"
        f" lifetime_max_phys_footprint={lifetime:.2f} GB wall=40.0s"
    )
    return "\n".join(lines) + "\n"


def cuda_log(
    *,
    used_high: float | None = 60.0,
    reserved_high: float = 70.0,
    device_peak: float = 70.5,
    baseline: float = 0.3,
    passed: bool = True,
    tail: str = "",
) -> str:
    lines = [f"{P} cuda_baseline baseline_gb={baseline:.2f}"]
    if used_high is not None:
        lines.append(
            f"{P} cuda_memory used_high_gb={used_high:.2f} reserved_high_gb={reserved_high:.2f}"
            f" device_overall_peak_gb={device_peak:.2f} baseline_gb={baseline:.2f}"
        )
    if tail:
        lines.append(tail)
    if passed:
        lines.append(OK)
    return "\n".join(lines) + "\n"


# The CUDA box: two 97887 MiB cards (102.64 GB); card 0 held by another tenant.
SNAPSHOT = "1, 97887, 840, 96599\n"
OOM = "DriverError(CUDA_ERROR_OUT_OF_MEMORY, \"out of memory\")"


class MlxRows(unittest.TestCase):
    def test_a_footprint_below_the_gc_limit_is_measured(self) -> None:
        cls, text = row.mlx_row("g", mlx_log(footprint=101.52, active=83.1), 0)
        self.assertEqual(cls, row.MEASURED)
        self.assertIn("active_peak_gb=83.10", text)
        self.assertIn("footprint_peak_gb=101.52", text)
        self.assertIn("gc_limit_gb=109.68", text)

    def test_a_footprint_at_the_gc_limit_is_cache_saturated(self) -> None:
        # The two rows the first run published as peaks: 109.96 and 110.08 against a 109.68 limit.
        for footprint in (109.96, 110.08, 108.70):
            with self.subTest(footprint=footprint):
                cls, _ = row.mlx_row("g", mlx_log(footprint=footprint), 0)
                self.assertEqual(cls, row.CACHE_SATURATED)
        cls, _ = row.mlx_row("g", mlx_log(footprint=108.60), 0)
        self.assertEqual(cls, row.MEASURED)

    def test_the_kernel_lifetime_maximum_outranks_the_polled_peak(self) -> None:
        # A transient between two polls: only the kernel's own high-water saw the ceiling.
        cls, text = row.mlx_row("g", mlx_log(footprint=95.0, lifetime=110.0), 0)
        self.assertEqual(cls, row.CACHE_SATURATED)
        self.assertIn("footprint_peak_gb=110.00 footprint_polled_gb=95.00", text)

    def test_with_the_cache_disabled_the_footprint_is_the_shape(self) -> None:
        cls, _ = row.mlx_row("g", mlx_log(footprint=110.0, cache_limit=0.0), 0)
        self.assertEqual(cls, row.MEASURED)

    def test_a_guard_stop_is_a_lower_bound_carrying_the_last_step(self) -> None:
        log = mlx_log(
            footprint=116.75,
            active=None,
            passed=False,
            stop="footprint 116.8 GB > cap 115 GB",
            step_active=91.25,
        )
        cls, text = row.mlx_row("r", log, 241)
        self.assertEqual(cls, row.GUARD_STOP)
        self.assertIn("active_peak_gb=>=91.25", text)

    def test_a_wall_stop_or_a_crash_is_failed(self) -> None:
        wall = mlx_log(footprint=80.0, active=None, passed=False, stop="wall 601s > 600s")
        self.assertEqual(row.mlx_row("g", wall, 241)[0], row.FAILED)
        crash = mlx_log(footprint=80.0, passed=False)
        self.assertEqual(row.mlx_row("g", crash, 101)[0], row.FAILED)

    def test_a_pass_without_counters_or_without_a_guard_line_is_failed(self) -> None:
        self.assertEqual(row.mlx_row("g", mlx_log(footprint=80.0, active=None), 0)[0], row.FAILED)
        self.assertEqual(row.mlx_row("g", f"{OK}\n", 0)[0], row.FAILED)


class CudaRows(unittest.TestCase):
    def test_a_run_with_room_on_the_card_is_measured_in_gb(self) -> None:
        cls, text = row.cuda_row("d", cuda_log(), 0, SNAPSHOT, 1)
        self.assertEqual(cls, row.MEASURED)
        self.assertIn("used_high_gb=60.00 reserved_high_gb=70.00 device_peak_gb=70.50", text)
        # 97887 / 840 / 96599 MiB, stated in the same unit as the peaks.
        self.assertIn("card_total_gb=102.64", text)
        self.assertIn("card_used_before_gb=0.88", text)
        self.assertIn("card_free_before_gb=101.29", text)
        self.assertIn("sysmem_fallback=not-needed", text)
        self.assertNotIn("MiB", text)

    def test_the_row_carries_every_step_in_order(self) -> None:
        steps = "\n".join(
            f"{P} cuda_step step={i} total=3 loss={loss} seconds={s} used_high_gb={gb}"
            for i, (gb, s, loss) in enumerate(
                [("80.10", "17.4", "0.013100"), ("82.41", "3.6", "0.031900"), ("82.40", "3.5", "0.020000")],
                start=1,
            )
        )
        _, text = row.cuda_row("g", cuda_log(tail=steps), 0, SNAPSHOT, 1)
        self.assertIn(
            " steps=3 step_peaks_gb=80.10/82.41/82.40 step_s=17.4/3.6/3.5"
            " step_loss=0.013100/0.031900/0.020000",
            text,
        )
        # a run that printed no step carries no trail
        self.assertNotIn("steps=", row.cuda_row("g", cuda_log(), 0, SNAPSHOT, 1)[1])

    def test_a_saturated_card_is_a_lower_bound(self) -> None:
        # The first run's restoration 512² row: 101.8 + 0.3 GB of a 102.64 GB card.
        log = cuda_log(device_peak=101.8, baseline=0.3, reserved_high=101.0)
        self.assertEqual(row.cuda_row("r", log, 0, SNAPSHOT, 1)[0], row.AT_CARD_LIMIT)
        # 98% of 102.64 GB is 100.59 GB.
        at = cuda_log(device_peak=100.30, baseline=0.3, reserved_high=90.0)
        self.assertEqual(row.cuda_row("r", at, 0, SNAPSHOT, 1)[0], row.AT_CARD_LIMIT)
        under = cuda_log(device_peak=100.20, baseline=0.3, reserved_high=90.0)
        self.assertEqual(row.cuda_row("r", under, 0, SNAPSHOT, 1)[0], row.MEASURED)

    def test_a_pool_larger_than_the_card_records_the_sysmem_spill(self) -> None:
        log = cuda_log(used_high=110.0, reserved_high=118.0, device_peak=102.0, baseline=0.3)
        cls, text = row.cuda_row("r", log, 0, SNAPSHOT, 1)
        self.assertEqual(cls, row.AT_CARD_LIMIT)
        self.assertIn("sysmem_spill_gb=15.66 sysmem_fallback=engaged", text)

    def test_an_out_of_memory_run_exceeds_the_card_and_keeps_its_counters(self) -> None:
        log = cuda_log(used_high=95.0, reserved_high=99.0, device_peak=101.9, passed=False, tail=OOM)
        cls, text = row.cuda_row("g", log, 101, SNAPSHOT, 1)
        self.assertEqual(cls, row.EXCEEDS_CARD)
        self.assertIn("used_high_gb=95.00", text)
        self.assertIn("card_free_before_gb=101.29", text)
        self.assertIn("sysmem_fallback=not-absorbing", text)
        self.assertTrue(text.endswith("card_free_before_gb=101.29"), text)
        bare = cuda_log(used_high=None, passed=False, tail=OOM)
        self.assertEqual(row.cuda_row("g", bare, 101, SNAPSHOT, 1)[0], row.EXCEEDS_CARD)

    def test_any_other_failure_is_failed_not_dropped(self) -> None:
        # The defect this replaces: a non-OOM failure produced no row and a green job.
        log = cuda_log(used_high=None, passed=False, tail="sampled GPU was not idle (baseline 5.0 GB")
        cls, text = row.cuda_row("g", log, 101, SNAPSHOT, 1)
        self.assertEqual(cls, row.FAILED)
        self.assertIn("rc=101", text)

    def test_a_pass_without_counters_is_failed(self) -> None:
        log = cuda_log(used_high=None)
        self.assertEqual(row.cuda_row("g", log, 0, SNAPSHOT, 1)[0], row.FAILED)

    def test_the_snapshot_must_name_the_selected_card(self) -> None:
        with self.assertRaises(ValueError):
            row.cuda_row("g", cuda_log(), 0, "0, 97887, 95262, 2177\n", 1)

    def test_the_card_line_states_every_card_in_gb(self) -> None:
        inventory = (
            "index, name, memory.total [MiB], memory.used [MiB], memory.free [MiB]\n"
            "0, NVIDIA RTX PRO 6000, 97887 MiB, 95262 MiB, 2177 MiB\n"
            "1, NVIDIA RTX PRO 6000, 97887 MiB, 840 MiB, 96599 MiB\n"
        )
        line = row.card_line(inventory, 1)
        self.assertIn("selected card 1 (NVIDIA RTX PRO 6000): total 102.64 GB, used 0.88 GB", line)
        self.assertIn("other card 0 (NVIDIA RTX PRO 6000): total 102.64 GB, used 99.89 GB", line)
        self.assertNotIn("MiB", line)
        self.assertTrue(line.isascii())
        with self.assertRaises(ValueError):
            row.card_line(inventory, 2)


class Summary(unittest.TestCase):
    def test_a_probe_without_exactly_one_row_is_reported(self) -> None:
        summary = (
            f"{row.CUDA_TAG} cards before the probes: selected card 1\n"
            f"{row.CUDA_TAG} generation-1024-ac-full MEASURED used_high_gb=1.00\n"
            f"{row.CUDA_TAG} depth-512-ac-none FAILED rc=101\n"
            f"{row.CUDA_TAG} depth-512-ac-none FAILED rc=101\n"
        )
        names = ["generation-1024-ac-full", "generation-1024-ac-none", "depth-512-ac-none"]
        self.assertEqual(
            row.missing_rows(summary, row.CUDA_TAG, names),
            [
                "generation-1024-ac-none: 0 rows (expected exactly 1)",
                "depth-512-ac-none: 2 rows (expected exactly 1)",
            ],
        )
        self.assertEqual(row.missing_rows(summary, row.CUDA_TAG, names[:1]), [])

    def test_the_command_appends_the_row_and_exits_one_only_for_failed(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            summary = root / "summary.txt"
            snapshot = root / "card.csv"
            snapshot.write_text(SNAPSHOT, encoding="utf-8")
            good = root / "good.log"
            good.write_text(cuda_log(), encoding="utf-8")
            bad = root / "bad.log"
            bad.write_text(cuda_log(used_high=None, passed=False, tail="panicked"), encoding="utf-8")

            def run(name: str, log: Path, rc: int) -> int:
                return subprocess.run(
                    [
                        sys.executable, str(MODULE_PATH), "cuda", "--name", name, "--log", str(log),
                        "--rc", str(rc), "--card-snapshot", str(snapshot), "--card", "1",
                        "--summary", str(summary),
                    ],
                    capture_output=True,
                    check=False,
                ).returncode

            self.assertEqual(run("a", good, 0), 0)
            self.assertEqual(run("b", bad, 101), 1)
            rows = summary.read_text(encoding="utf-8").splitlines()
            self.assertEqual(len(rows), 2)
            self.assertTrue(rows[0].startswith(f"{row.CUDA_TAG} a MEASURED "))
            self.assertTrue(rows[1].startswith(f"{row.CUDA_TAG} b FAILED rc=101"))
            expect = [sys.executable, str(MODULE_PATH), "expect", "--backend", "cuda",
                      "--summary", str(summary)]
            self.assertEqual(subprocess.run(expect + ["a", "b"], check=False).returncode, 0)
            missing = subprocess.run(expect + ["a", "b", "c"], capture_output=True, check=False)
            self.assertEqual(missing.returncode, 1)
            self.assertIn(b"c: 0 rows", missing.stdout)


@unittest.skipUnless(sys.platform == "darwin", "the guard reads macOS proc_pid_rusage")
class FootprintGuard(unittest.TestCase):
    def guard(self, cap: str, wall: str, child: str, grace: str = "60") -> str:
        result = subprocess.run(
            [sys.executable, str(GUARD_PATH), cap, wall, sys.executable, "-c", child],
            capture_output=True,
            text=True,
            encoding="utf-8",
            check=False,
            env={**os.environ, "IRIS_GUARD_TERM_GRACE_S": grace},
        )
        return result.stdout

    def test_the_kernel_lifetime_maximum_sees_a_transient_the_poll_misses(self) -> None:
        # ~400 MB of anonymous pages written and unmapped (so they really leave the footprint) well
        # inside one 0.25 s poll interval, then idle: a poll sees the transient only by luck.
        child = (
            "import mmap, time\n"
            "m = mmap.mmap(-1, 400 << 20)\n"
            "chunk = b'x' * (1 << 20)\n"
            "for _ in range(400):\n"
            "    m.write(chunk)\n"
            "m.close()\n"
            "time.sleep(1.5)\n"
        )
        out = self.guard("100", "60", child)
        match = row._GUARD.search(out)
        self.assertIsNotNone(match, out)
        assert match is not None
        self.assertEqual(match["exit"], "0")
        # The 400 MB, and nothing like a timestamp or a counter read from a neighbouring word.
        self.assertGreaterEqual(float(match["lifetime"]), 0.4)
        self.assertLess(float(match["lifetime"]), 2.0)
        self.assertGreaterEqual(float(match["lifetime"]), float(match["polled"]))

    def test_a_child_that_ignores_sigterm_is_killed(self) -> None:
        child = (
            "import signal, time\n"
            "signal.signal(signal.SIGTERM, signal.SIG_IGN)\n"
            "time.sleep(600)\n"
        )
        out = self.guard("100", "0.5", child, grace="1")
        self.assertIn("[guard] SIGTERM: wall", out)
        self.assertIn("[guard] SIGKILL: child ignored SIGTERM", out)
        self.assertIn("[guard] exit=-9 ", out)


if __name__ == "__main__":
    unittest.main()
