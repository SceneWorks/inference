import importlib.util
import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest import mock

import yaml

ROOT = Path(__file__).resolve().parents[2]
HELPER = ROOT / "scripts/ci/mac2_proc_macro_strip.py"
WORKFLOW = ROOT / ".github/workflows/real-weights.yml"
LOCK = '''version = 4
[[package]]
name = "castaway"
version = "0.2.4"
source = "registry+https://github.com/rust-lang/crates.io-index"
checksum = "dec551ab6e7578819132c713a93c022a05d60159dc86e7a7050223577484c55a"
[[package]]
name = "rustversion"
version = "1.0.23"
source = "registry+https://github.com/rust-lang/crates.io-index"
checksum = "cf54715a573b99ac80df0bc206da022bcd442c974952c7b9720069370852e21f"
'''

def load_helper():
    spec = importlib.util.spec_from_file_location("mac2_proc_macro_strip", HELPER)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module

class Mac2ProcMacroStripTests(unittest.TestCase):
    def test_workflow_is_one_manual_cpu_job_and_hash_bound(self):
        workflow = yaml.safe_load(WORKFLOW.read_text(encoding="utf-8"))
        self.assertEqual(set(workflow[True]), {"workflow_dispatch"})
        self.assertEqual(set(workflow["jobs"]), {"mac2-proc-macro-strip"})
        self.assertEqual(workflow[True]["workflow_dispatch"]["inputs"]["profile"]["options"],
                         ["mac2-proc-macro-strip"])
        job = workflow["jobs"]["mac2-proc-macro-strip"]
        self.assertEqual(job["if"], "github.event_name == 'workflow_dispatch' && inputs.profile == 'mac2-proc-macro-strip'")
        self.assertEqual(job["runs-on"], ["self-hosted", "macOS", "ARM64", "rw-mage"])
        self.assertEqual(job["timeout-minutes"], 10)
        self.assertEqual(job["permissions"], {"contents": "read"})
        rendered = json.dumps(job)
        combined = rendered + HELPER.read_text(encoding="utf-8")
        for required in ("68b925baa5746281d47af9342cdcc28bc54d2ad0", "--offline",
                         "GITHUB_RUN_ATTEMPT", "CARGO_PROFILE_RELEASE_STRIP", "stroffMod8",
                         "ctypes.CDLL", "if-no-files-found", "always()"):
            self.assertIn(required, combined)
        for forbidden in ("actions/checkout", "pip install", "cargo clean", "rm -", "mlx-gen", "Metal"):
            self.assertNotIn(forbidden, rendered)

    def test_two_arms_change_only_strip_and_capture_failure_and_success(self):
        helper = load_helper()
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory).resolve()
            lock = root / "source.lock"; lock.write_text(LOCK, encoding="utf-8")
            config = root / "config.toml"; config.write_text("[env]\n", encoding="utf-8")
            checks = []
            def fake_capture(command, *, env=None, timeout=30):
                if command[:2] == ["cargo", "generate-lockfile"]:
                    Path(command[command.index("--manifest-path") + 1]).with_name("Cargo.lock").write_text(LOCK, encoding="utf-8")
                if command[:2] == ["cargo", "check"]:
                    checks.append({key: env.get(key) for key in ("CARGO_TARGET_DIR", "RUSTC_WRAPPER",
                                  "RUSTC_WORKSPACE_WRAPPER", "CARGO_PROFILE_RELEASE_STRIP", "UNCHANGED")})
                    deps = Path(env["CARGO_TARGET_DIR"]) / "release/deps"; deps.mkdir(parents=True)
                    (deps / "librustversion-test.dylib").write_bytes(b"proc-macro")
                    ok = env.get("CARGO_PROFILE_RELEASE_STRIP") == "none"
                    return {"argv": command, "exitCode": 0 if ok else 1, "stdout": "{}\n",
                            "stderr": "" if ok else "E0463\n", "timedOut": False}
                if command[:2] == ["otool", "-l"]:
                    ok = "no-strip" in command[-1]; offset = 80 if ok else 84
                    return {"argv": command, "exitCode": 0, "stdout": f" stroff {offset}\n strsize 12\n minos 26.2\n sdk 27.0\n", "stderr": "", "timedOut": False}
                if len(command) > 2 and command[1] == "-c" and "ctypes.CDLL" in command[2]:
                    ok = "no-strip" in command[-1]
                    return {"argv": command, "exitCode": 0 if ok else 1, "stdout": "dlopen-ok\n" if ok else "",
                            "stderr": "" if ok else "mis-aligned LINKEDIT\n", "timedOut": False}
                return {"argv": command, "exitCode": 0, "stdout": "ok\n", "stderr": "", "timedOut": False}
            env = {"RUNNER_TEMP": str(root), "GITHUB_RUN_ID": "123", "GITHUB_RUN_ATTEMPT": "2",
                   "PROC_MACRO_SOURCE_LOCK": str(lock), "PROC_MACRO_SOURCE_CONFIG": str(config),
                   "RUSTC_WRAPPER": "must-clear", "CARGO_PROFILE_RELEASE_STRIP": "symbols", "UNCHANGED": "same"}
            with mock.patch.dict(os.environ, env, clear=True), mock.patch.object(helper, "capture", fake_capture), \
                 mock.patch.object(helper.sys, "argv", [str(HELPER), str(root / "out")]):
                self.assertEqual(helper.main(), 0)
            summary = json.loads((root / "out/summary.json").read_text(encoding="utf-8"))
            self.assertTrue(summary["captureComplete"])
            self.assertEqual(summary["compilationAcceptance"], {"baseline": False, "no-strip": True})
            self.assertTrue(summary["causalAcceptance"])
            self.assertIsNone(checks[0]["CARGO_PROFILE_RELEASE_STRIP"])
            self.assertEqual(checks[1]["CARGO_PROFILE_RELEASE_STRIP"], "none")
            self.assertEqual({r["UNCHANGED"] for r in checks}, {"same"})
            self.assertEqual({r["RUSTC_WRAPPER"] for r in checks}, {""})
            self.assertEqual({r["RUSTC_WORKSPACE_WRAPPER"] for r in checks}, {""})

    def test_parser_and_identity_mutants_fail_closed(self):
        helper = load_helper()
        self.assertEqual(helper.macho_fields(" stroff 101\n strsize 9\n minos 26.2\n sdk 27.0\n"),
                         {"stroff": 101, "strsize": 9, "minos": "26.2", "sdk": "27.0", "stroffMod8": 5})
        source = HELPER.read_text(encoding="utf-8")
        self.assertIn('arm_env["CARGO_PROFILE_RELEASE_STRIP"] = strip_value', source)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "Cargo.lock"
            path.write_text(LOCK.replace("1.0.23", "1.0.22"), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "rustversion identity changed"):
                helper.validate_lock(path)

if __name__ == "__main__":
    unittest.main()
