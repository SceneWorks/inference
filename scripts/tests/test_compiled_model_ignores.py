"""Compiler-truth policy tests use fabricated metadata and mocked listing, never Cargo/NN."""
import importlib.util
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

SOURCE = Path(__file__).resolve().parents[1] / 'ci/check_compiled_model_ignores.py'
spec = importlib.util.spec_from_file_location('compiled_ignore_test', SOURCE)
check = importlib.util.module_from_spec(spec)
spec.loader.exec_module(check)


class CompiledIgnores(unittest.TestCase):
    def test_a_listed_registered_model_must_also_be_in_ignored_listing(self):
        rows = [{'fqn': 'tests::model'}]
        with self.assertRaisesRegex(ValueError, 'NOT ignored'):
            check.verify_lists(rows, 'tests::model: test\n', '0 tests, 0 benchmarks\n')
        result = check.verify_lists(rows, 'tests::model: test\nordinary: test\n', 'tests::model: test\n')
        self.assertEqual(result['listed_registered_ignored'], 1)
        self.assertEqual(result['model_bodies_executed'], 0)

    def test_cfg_absence_is_not_a_runtime_pass(self):
        result = check.verify_lists([{'fqn': 'tests::cfg_absent'}], '', '')
        self.assertEqual(result['listed_registered_ignored'], 0)
        self.assertEqual(result['cfg_absent_not_runtime_proof'], 1)
        self.assertEqual(result['model_bodies_executed'], 0)
        with self.assertRaises(ValueError):
            check.verify_lists([], '', 'unexpected: test\n')

    def test_exact_package_and_staticlib_target_metadata_select_only_libtest_listings(self):
        with tempfile.TemporaryDirectory() as temporary:
            binary = Path(temporary) / 'test-binary'; binary.write_bytes(b'not executed')
            target = {'name': 'ios_smoke', 'kind': ['staticlib', 'rlib']}
            artifact = {'package_id': 'exact-smoke-id', 'target': target, 'executable': str(binary)}
            metadata = {'packages': [{'id': 'exact-smoke-id', 'name': 'ios-smoke'}]}
            rows = [{'package': 'ios-smoke', 'target': 'ios_smoke', 'target_kind': 'lib', 'fqn': 'tests::model'}]
            with patch.object(check.subprocess, 'run', return_value=subprocess.CompletedProcess([], 0, 'tests::model: test\n', '')) as run:
                reports = check.check_artifacts([artifact], rows, metadata)
                self.assertEqual(run.call_count, 2)
                self.assertEqual(run.call_args_list[0].args[0], [str(binary), '--list'])
                self.assertEqual(run.call_args_list[1].args[0], [str(binary), '--list', '--ignored'])
                self.assertEqual(reports[0]['cargo_target_kind'], ['staticlib', 'rlib'])

    def test_unknown_custom_harness_is_never_launched_with_a_guessed_list_argument(self):
        artifact = {'package_id': 'id', 'target': {'name': 'custom', 'kind': ['bench']}, 'executable': '/not/executed'}
        with patch.object(check.subprocess, 'run') as run:
            self.assertEqual(check.check_artifacts([artifact], [], {'packages': [{'id': 'id', 'name': 'raw-kernels'}]}), [])
            run.assert_not_called()

    def test_existing_collection_wrapper_is_preserved(self):
        self.assertEqual(check.cargo_boundary(['cargo', 'test', '--lib']), 0)
        self.assertEqual(check.cargo_boundary(['python3', 'scripts/ci/run_ci_collection.py', '--', 'cargo', 'test', '--lib']), 3)
        with self.assertRaises(ValueError):
            check.cargo_boundary(['sh', '-c', 'arbitrary'])


if __name__ == '__main__':
    unittest.main()
