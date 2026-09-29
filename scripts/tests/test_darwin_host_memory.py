"""The macOS host-memory admission metric: one definition, three implementations, one fixture.

The Rust campaign supervisor, the Python media supervisor and the kv-poc workflow precheck must
compute the same `availableBytes` from the same vm_stat text, and all must fail closed on a
missing, duplicated or malformed counter. The fixture is shared with the Rust tests
(crates/llm/mlx-llm/src/campaign_supervisor.rs). No model or GPU is started.
"""

import json
import os
import platform
import shutil
import subprocess
import tempfile
import unittest
from pathlib import Path

from scripts import media_campaign_supervisor as safety

ROOT = Path(__file__).resolve().parents[2]
CASES = json.loads((ROOT / "crates/llm/mlx-llm/testdata/darwin-host-memory-cases.json").read_text(encoding="utf-8"))["cases"]
COMMON = ROOT / ".github/kv-poc/common.sh"
COUNTERS = ("freePages", "speculativePages", "purgeablePages", "inactivePages", "fileBackedPages",
            "anonymousPages", "throttledPages")


def reclaimable(pages: dict[str, int]) -> int:
    return min(max(0, pages["inactivePages"] - pages["purgeablePages"]),
               max(0, pages["fileBackedPages"] - pages["speculativePages"]),
               max(0, pages["inactivePages"] + pages["throttledPages"] - pages["anonymousPages"]))


# Plausible wrong definitions. Each must disagree with the fixture on at least one case.
MUTANTS = {
    "drop the anonymous bound (v1)": lambda p: min(max(0, p["inactivePages"] - p["purgeablePages"]),
                                                   max(0, p["fileBackedPages"] - p["speculativePages"])),
    "drop the min (credit all file-backed)": lambda p: max(0, p["fileBackedPages"] - p["speculativePages"]),
    "count all inactive": lambda p: p["inactivePages"],
    "drop file-backed (inactive minus purgeable)": lambda p: max(0, p["inactivePages"] - p["purgeablePages"]),
    "purgeable counted twice": lambda p: min(p["inactivePages"], max(0, p["fileBackedPages"] - p["speculativePages"]),
                                             max(0, p["inactivePages"] + p["throttledPages"] - p["anonymousPages"])),
    "throttled ignored": lambda p: min(max(0, p["inactivePages"] - p["purgeablePages"]),
                                       max(0, p["fileBackedPages"] - p["speculativePages"]),
                                       max(0, p["inactivePages"] - p["anonymousPages"])),
    "no file cache (free + speculative + purgeable)": lambda p: 0,
}


def page_counter(text: str, key: str) -> int:
    return int(next(line for line in text.splitlines() if line.startswith(key + ":")).split(":")[1].strip(" ."))


def shell_measure(text: str) -> subprocess.CompletedProcess:
    env = {**os.environ, "INFERENCE_SHA": "x", "SCENEWORKS_SHA": "y"}
    return subprocess.run(
        ["bash", "-c", f'source "{COMMON}"; host_memory_from_vm_stat'],
        input=text, capture_output=True, text=True, encoding="utf-8", env=env, check=False,
    )


class DarwinHostMemoryTests(unittest.TestCase):
    def test_fixture_covers_the_required_shapes(self):
        names = {case["name"] for case in CASES}
        for required in ("heavy-file-cache", "heavy-anonymous-inactive", "active-mapped-file",
                         "missing-pages-purgeable", "missing-file-backed-pages", "missing-pages-inactive"):
            self.assertIn(required, names)
        self.assertTrue(any(case["expect"] is None for case in CASES))

    def test_python_matches_the_shared_fixture_and_fails_closed(self):
        for case in CASES:
            with self.subTest(case["name"]):
                if case["expect"] is None:
                    with self.assertRaisesRegex(safety.SupervisionError, "probe-failure"):
                        safety.darwin_available(case["vmStat"])
                else:
                    self.assertEqual(safety.darwin_available(case["vmStat"]), case["expect"])
                    self.assertEqual(safety.validate_host_memory(case["expect"]), case["expect"])

    @unittest.skipUnless(shutil.which("bash") and shutil.which("awk"), "needs bash and awk")
    def test_workflow_precheck_matches_the_shared_fixture_and_fails_closed(self):
        for case in CASES:
            with self.subTest(case["name"]):
                result = shell_measure(case["vmStat"])
                if case["expect"] is None:
                    self.assertNotEqual(result.returncode, 0)
                    self.assertEqual(result.stdout, "")
                    continue
                self.assertEqual(result.returncode, 0, result.stderr)
                expect = case["expect"]
                self.assertEqual(
                    [int(value) for value in result.stdout.split()],
                    [expect["availableBytes"], expect["pageSizeBytes"],
                     *(expect[key] for key in COUNTERS), expect["reclaimableFilePages"]],
                )

    def test_fixture_discriminates_every_plausible_wrong_definition(self):
        valid = [case["expect"] for case in CASES if case["expect"] is not None]
        for case in valid:
            self.assertEqual(reclaimable(case), case["reclaimableFilePages"])
        for name, mutant in MUTANTS.items():
            with self.subTest(name):
                self.assertTrue(any(mutant(case) != case["reclaimableFilePages"] for case in valid))

    def test_every_case_obeys_the_vm_stat_page_identity(self):
        # File-backed + Anonymous = active + inactive + speculative + throttled: the identity the
        # anonymous bound rests on. It holds exactly on both real snapshots from this host.
        for case in CASES:
            if case["expect"] is None:
                continue
            with self.subTest(case["name"]):
                text = case["vmStat"]
                self.assertEqual(
                    page_counter(text, "File-backed pages") + page_counter(text, "Anonymous pages"),
                    sum(page_counter(text, key) for key in
                        ("Pages active", "Pages inactive", "Pages speculative", "Pages throttled")))

    def test_inactive_anonymous_with_active_file_cache_is_not_credited(self):
        case = next(case for case in CASES if case["name"] == "inactive-anonymous-active-file")["expect"]
        self.assertEqual(case["reclaimableFilePages"], 0)
        self.assertEqual(MUTANTS["drop the anonymous bound (v1)"](case), 40000)

    def test_recorded_components_must_recompute(self):
        expect = next(case for case in CASES if case["name"] == "heavy-anonymous-inactive")["expect"]
        for field, value in (("availableBytes", expect["availableBytes"] + 16384),
                             ("reclaimableFilePages", expect["inactivePages"]),
                             ("metric", "darwin-vm-stat-available-v1"), ("pageSizeBytes", 1000)):
            with self.subTest(field):
                with self.assertRaisesRegex(safety.SupervisionError, "invalid-admission"):
                    safety.validate_host_memory({**expect, field: value})
        with self.assertRaisesRegex(safety.SupervisionError, "invalid-admission"):
            safety.validate_host_memory({key: value for key, value in expect.items() if key != "inactivePages"})

    def test_admission_records_components_on_darwin_only(self):
        with tempfile.TemporaryDirectory() as directory:
            def policy(backend, extra=None):
                path = Path(directory) / f"{backend}.json"
                path.write_text(json.dumps({
                    "schemaVersion": 1, "backend": backend, "deadlineSeconds": 1, "pollMillis": 10,
                    "termGraceMillis": 50, "hostFreeReserveBytes": 100, "childFootprintCapBytes": 10**6,
                    "stdoutCapBytes": 4096, "stderrCapBytes": 4096, "eventCapBytes": 10**6, **(extra or {}),
                }), encoding="utf-8")
                return safety.load_policy(path)
            darwin = policy("darwin-mlx")
            admission = safety.runtime_guarded_admission(darwin)
            with self.assertRaisesRegex(safety.SupervisionError, "invalid-admission"):
                safety.validate_admission(admission, policy_sha256=darwin.sha256)
            admission["hostMemoryComponents"] = CASES[0]["expect"]
            safety.validate_admission(admission, policy_sha256=darwin.sha256)
            cuda = policy("linux-cuda", {"cudaDeviceUuid": "GPU-00000000-0000-0000-0000-000000000000",
                                         "gpuFreeReserveBytes": 1, "childGpuCapBytes": 1})
            foreign = {**safety.runtime_guarded_admission(cuda), "hostMemoryComponents": CASES[0]["expect"]}
            with self.assertRaisesRegex(safety.SupervisionError, "invalid-admission"):
                safety.validate_admission(foreign, policy_sha256=cuda.sha256)

    @unittest.skipUnless(platform.system() == "Darwin", "reads this Mac's vm_stat")
    def test_live_vm_stat_parses_identically_in_python_and_shell(self):
        text = subprocess.run(["/usr/bin/vm_stat"], capture_output=True, text=True, encoding="ascii",
                              check=True).stdout
        host = safety.validate_host_memory(safety.darwin_available(text))
        shell = [int(value) for value in shell_measure(text).stdout.split()]
        self.assertEqual(shell[0], host["availableBytes"])
        self.assertGreaterEqual(host["availableBytes"],
                                (host["freePages"] + host["speculativePages"]) * host["pageSizeBytes"])


if __name__ == "__main__":
    unittest.main()
