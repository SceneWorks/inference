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
