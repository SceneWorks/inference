"""Ensure the hosted dependency-test runner cannot report success on empty/ignored rows."""
import importlib.util
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def load_runner():
    spec = importlib.util.spec_from_file_location("clip_runner", ROOT / "scripts/ci/run_clip_regex_certificate_tests.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class CertificateRunnerTests(unittest.TestCase):
    def test_accepts_exactly_one_actual_passed_row(self):
        load_runner().require_one_passed("test result: ok. 1 passed; 0 failed; 0 ignored; 0 measured; 5 filtered out;", "fixture")

    def test_rejects_empty_filtered_ignored_failed_and_duplicate_rows(self):
        runner = load_runner()
        passed = "test result: ok. 1 passed; 0 failed; 0 ignored;"
        for output in ["", "test result: ok. 0 passed; 0 failed; 0 ignored;", "test result: ok. 0 passed; 0 failed; 1 ignored;", "test result: FAILED. 0 passed; 1 failed; 0 ignored;", passed + "\n" + passed]:
            with self.subTest(output=output), self.assertRaises(RuntimeError):
                runner.require_one_passed(output, "fixture")


class RustcArtifactBridgeTests(unittest.TestCase):
    def load_bridge(self):
        spec = importlib.util.spec_from_file_location("clip_rustc", ROOT / "scripts/ci/clip_regex_test_rustc.py")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module

    def test_only_primary_regex_test_uses_exact_same_hash_rlib(self):
        import tempfile
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            rlib = root / "libregex_automata-hash123.rlib"
            rlib.write_bytes(b"fixture, never executed")
            args = ["--crate-name", "regex", "--test", "--extern", "regex_automata=" + str(rlib.with_suffix(".rmeta"))]
            actual = self.load_bridge().linkable_test_args(args)
            self.assertEqual(actual[:-1], args[:-1])
            self.assertEqual(actual[-1], "regex_automata=" + str(rlib))
            self.assertTrue(args[-1].endswith(".rmeta"), "input argv is not mutated")

    def test_missing_exact_rlib_fails_without_other_hash_fallback(self):
        import tempfile
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "libregex_syntax-other.rlib").write_bytes(b"fixture")
            args = ["--crate-name", "regex", "--test", "--extern", "regex_syntax=" + str(root / "libregex_syntax-required.rmeta")]
            with self.assertRaises(RuntimeError):
                self.load_bridge().linkable_test_args(args)

    def test_other_compiler_commands_and_existing_rlibs_are_unchanged(self):
        bridge = self.load_bridge()
        for args in [["-vV"], ["--crate-name", "regex", "--extern", "x=missing.rmeta"], ["--crate-name", "other", "--test", "--extern", "x=missing.rmeta"], ["--crate-name", "regex", "--test", "--extern", "x=existing.rlib"]]:
            with self.subTest(args=args):
                self.assertEqual(bridge.linkable_test_args(args), args)


class RustcTestArtifactPlanTests(unittest.TestCase):
    def load_bridge(self):
        spec = importlib.util.spec_from_file_location("clip_rustc_plan", ROOT / "scripts/ci/clip_regex_test_rustc.py")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module

    def test_explicit_test_output_is_bin_link_only_without_cargo_suffix(self):
        args = ["--crate-name", "regex", "--crate-type", "lib", "--emit=dep-info,metadata,link", "-C", "extra-filename=-cargo", "-C", "metadata=keep", "--out-dir", "/cargo/out", "--test", "-o", "/test/exact"]
        self.assertEqual(self.load_bridge().linkable_test_args(args), ["--crate-name", "regex", "--crate-type", "bin", "--emit=link", "-C", "metadata=keep", "--test", "-o", "/test/exact"])
        self.assertIn("extra-filename=-cargo", args, "input arguments remain unchanged")

    def test_inline_output_flags_are_normalized_only_for_explicit_primary_test(self):
        bridge = self.load_bridge()
        args = ["--crate-name=regex", "--crate-type=rlib", "--emit", "dep-info,metadata,link", "-Cextra-filename=-cargo", "-Cmetadata=keep", "--out-dir=/cargo/out", "--test", "-o", "/test/exact"]
        self.assertEqual(bridge.linkable_test_args(args), ["--crate-name=regex", "--crate-type=bin", "--emit", "link", "-Cmetadata=keep", "--test", "-o", "/test/exact"])
        without_output = args[:-2]
        self.assertEqual(bridge.linkable_test_args(without_output), without_output)
