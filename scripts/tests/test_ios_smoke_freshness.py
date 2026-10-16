"""Exercise the launch boundary and actual report freshness predicate without a device."""
import os
from pathlib import Path
import subprocess
import tempfile
import unittest


SCRIPT = Path(__file__).resolve().parents[1] / "ios" / "run_smoke.sh"


class IosSmokeFreshnessTests(unittest.TestCase):
    def test_launch_timestamp_is_available_inside_both_launch_invocations(self):
        source = SCRIPT.read_text(encoding="utf-8")
        # Include all real timestamp assignments around launch, including a post-launch one if
        # regressed. The fake device writes its report immediately at launch, before returning.
        start = source.index('say "launching (unlock')
        before = source[:start].rfind("LAUNCHED_AT=$(date +%s)")
        start = before if before != -1 else start
        end = source.index('for _ in $(seq 1 "$POLL_TRIES")', start)
        block = source[start:end].replace('REPORT=/tmp/ios-smoke-report.txt', 'REPORT="$POLL_REPORT"')
        for environment in ["{}", '{"IOS_SMOKE_ONLY":"sana"}']:
            with self.subTest(environment=environment), tempfile.TemporaryDirectory() as temp:
                setup = r'''
set -eu
say() { :; }
date() { echo 100; }
sleep() { :; }
xcrun() {
  test "${LAUNCHED_AT:-unset}" = 100 || return 72
  printf 'fresh report' > "$FAKE_REPORT"
}
DEVICE=fake
SMOKE_BUNDLE_ID=fake
CARGO_FEATURES=
POLL_TRIES=1
'''
                result = subprocess.run(
                    ["bash", "-c", setup + block + '\ntest -f "$FAKE_REPORT"'],
                    env={**os.environ, "LAUNCH_ENV": environment, "FAKE_REPORT": str(Path(temp) / "marker"), "POLL_REPORT": str(Path(temp) / "report")},
                    capture_output=True, text=True, encoding="utf-8",
                )
                self.assertEqual(result.returncode, 0, result.stderr)

    def test_actual_report_guard_accepts_fresh_and_equal_but_rejects_stale(self):
        source = SCRIPT.read_text(encoding="utf-8")
        guard = next(line.strip() for line in source.splitlines()
                     if 'stat -f%m "$REPORT"' in line and "&& break" in line)
        # Run the exact shell condition, substituting only the loop action with an observable code.
        guard = guard.replace("&& break", "&& exit 0\nexit 9")
        for mtime, expected in [(99, 9), (100, 0), (101, 0)]:
            with self.subTest(mtime=mtime):
                result = subprocess.run(["bash", "-c", 'stat() { echo "$MTIME"; };\n' + guard],
                    env={**os.environ, "REPORT": "fake", "MTIME": str(mtime), "LAUNCHED_AT": "100"},
                    capture_output=True, text=True, encoding="utf-8")
                self.assertEqual(result.returncode, expected, result.stderr)


if __name__ == "__main__":
    unittest.main()
