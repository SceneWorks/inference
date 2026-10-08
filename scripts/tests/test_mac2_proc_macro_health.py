import importlib.util
import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest import mock

import yaml


ROOT = Path(__file__).resolve().parents[2]
HELPER = ROOT / "scripts/ci/mac2_proc_macro_health.py"
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
    spec = importlib.util.spec_from_file_location("mac2_proc_macro_health", HELPER)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class Mac2ProcMacroHealthTests(unittest.TestCase):
    def test_workflow_is_dispatch_only_cpu_bounded_and_hash_bound(self):
        workflow = yaml.safe_load(WORKFLOW.read_text(encoding="utf-8"))
        self.assertEqual(set(workflow[True]), {"workflow_dispatch"})
        self.assertEqual(set(workflow["jobs"]), {"mac2-proc-macro-health"})
        self.assertEqual(workflow[True]["workflow_dispatch"]["inputs"]["profile"]["options"],
                         ["mac2-proc-macro-health"])
        self.assertIn("mac2-proc-macro-health",
                      workflow[True]["workflow_dispatch"]["inputs"]["profile"]["options"])
        job = workflow["jobs"]["mac2-proc-macro-health"]
        self.assertEqual(job["if"], "github.event_name == 'workflow_dispatch' && inputs.profile == 'mac2-proc-macro-health'")
        self.assertEqual(job["runs-on"], ["self-hosted", "macOS", "ARM64", "rw-mage"])
        self.assertEqual(job["timeout-minutes"], 10)
        self.assertEqual(job["permissions"], {"contents": "read"})
        rendered = json.dumps(job)
        combined = rendered + HELPER.read_text(encoding="utf-8")
        for required in ("68b925baa5746281d47af9342cdcc28bc54d2ad0", "--offline", "GITHUB_RUN_ATTEMPT",
                         "mac2-proc-macro-health.py", "if-no-files-found", "always()"):
            self.assertIn(required, combined)
        for forbidden in ("actions/checkout", "pip install", "cargo clean", "rm -", "mlx-gen", "Metal"):
            self.assertNotIn(forbidden, rendered)

    def test_helper_captures_both_arms_even_when_compilation_fails(self):
        helper = load_helper()
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory).resolve()
            source_lock = root / "source.lock"; source_lock.write_text(LOCK, encoding="utf-8")
            source_config = root / "config.toml"; source_config.write_text("[env]\n", encoding="utf-8")
            calls = []

            def fake_capture(command, *, env=None, timeout=30):
                calls.append((command, None if env is None else {key: env.get(key) for key in
                              ("CARGO_TARGET_DIR", "RUSTC_WRAPPER", "RUSTC_WORKSPACE_WRAPPER")}))
                if command[:2] == ["cargo", "generate-lockfile"]:
                    manifest = Path(command[command.index("--manifest-path") + 1])
                    manifest.with_name("Cargo.lock").write_text(LOCK, encoding="utf-8")
                if command[:2] == ["cargo", "check"]:
                    target = Path(env["CARGO_TARGET_DIR"]) / "release/deps"
                    target.mkdir(parents=True)
                    (target / "librustversion-test.dylib").write_bytes(b"proc-macro")
                    return {"argv": command, "exitCode": 1, "stdout": "{}\n",
                            "stderr": "E0463\n", "timedOut": False}
                return {"argv": command, "exitCode": 0, "stdout": "ok\n", "stderr": "", "timedOut": False}

            environment = {"RUNNER_TEMP": str(root), "GITHUB_RUN_ID": "123", "GITHUB_RUN_ATTEMPT": "2",
                           "PROC_MACRO_SOURCE_LOCK": str(source_lock), "PROC_MACRO_SOURCE_CONFIG": str(source_config),
                           "RUSTC_WRAPPER": "sccache"}
            with mock.patch.dict(os.environ, environment, clear=True), mock.patch.object(helper, "capture", fake_capture), \
                    mock.patch.object(helper.sys, "argv", [str(HELPER), str(root / "out")]):
                self.assertEqual(helper.main(), 0)
            summary = json.loads((root / "out/summary.json").read_text(encoding="utf-8"))
            self.assertTrue(summary["captureComplete"])
            self.assertEqual(summary["compilationAcceptance"], {"direct": False, "inherited": False})
            self.assertEqual([row["name"] for row in summary["arms"]], ["inherited", "direct"])
            checks = [row for row in calls if row[0][:2] == ["cargo", "check"]]
            self.assertEqual(checks[0][1]["RUSTC_WRAPPER"], "sccache")
            self.assertEqual(checks[1][1]["RUSTC_WRAPPER"], "")
            self.assertEqual(checks[1][1]["RUSTC_WORKSPACE_WRAPPER"], "")
            self.assertTrue(all("--locked" in row[0] and "--offline" in row[0] and "-vv" in row[0]
                                for row in checks))

    def test_lock_identity_and_new_output_are_fail_closed(self):
        helper = load_helper()
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "Cargo.lock"
            path.write_text(LOCK.replace("1.0.23", "1.0.22"), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "rustversion identity changed"):
                helper.validate_lock(path)


if __name__ == "__main__":
    unittest.main()
