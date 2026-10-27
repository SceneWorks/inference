"""CI_IMPACT: <1 s offline Python/fake-shell checks; no tool download, simulator or model."""
import hashlib
import io
import json
import os
from pathlib import Path
import subprocess
import tempfile
import unittest
from unittest.mock import patch
import zipfile

import yaml

from scripts.ci import provision_simulator_xcodegen as tool

ROOT = Path(__file__).resolve().parents[2]


def archive(extra=None):
    value = io.BytesIO()
    with zipfile.ZipFile(value, 'w') as bundle:
        bundle.writestr(tool.BUNDLE + '/info.json', json.dumps({'artifacts': {'xcodegen': {
            'version': tool.VERSION, 'type': 'executable', 'variants': [{
                'path': 'xcodegen-2.46.0-macosx/bin/xcodegen',
                'supportedTriples': ['x86_64-apple-macosx', 'arm64-apple-macosx']}]}}}))
        bundle.writestr(tool.BINARY, b'fixture executable, never executed')
        bundle.writestr(str(Path(tool.BINARY).parent / 'XcodeGen_XcodeGenKit.bundle/preset.yml'), 'fixture')
        if extra:
            bundle.writestr(*extra)
    return value.getvalue()


class SimulatorProjectToolTests(unittest.TestCase):
    def test_verified_archive_keeps_resources_and_rejects_tampering_or_reuse(self):
        data = archive()
        with tempfile.TemporaryDirectory() as directory, \
             patch.object(tool, 'ARCHIVE_BYTES', len(data)), \
             patch.object(tool, 'SHA256', hashlib.sha256(data).hexdigest()):
            destination = Path(directory) / 'tool'
            binary = tool.install(data, destination)
            self.assertTrue(binary.stat().st_mode & 0o111)
            self.assertTrue((binary.parent / 'XcodeGen_XcodeGenKit.bundle/preset.yml').is_file())
            with self.assertRaisesRegex(ValueError, 'fresh job-local'):
                tool.install(data, destination)
            with self.assertRaisesRegex(ValueError, 'SHA-256 mismatch'):
                tool.install(data[:-1] + bytes([data[-1] ^ 1]), Path(directory) / 'tampered')
            with self.assertRaisesRegex(ValueError, 'SHA-256 mismatch'):
                tool.install(data + b'extra', Path(directory) / 'oversized')

    def test_archive_cannot_write_outside_its_job_directory(self):
        for name in ('../escape', '/absolute', tool.BUNDLE + '/../escape'):
            data = archive((name, b'bad'))
            with self.subTest(name=name), tempfile.TemporaryDirectory() as directory, \
                 patch.object(tool, 'ARCHIVE_BYTES', len(data)), \
                 patch.object(tool, 'SHA256', hashlib.sha256(data).hexdigest()):
                with self.assertRaisesRegex(ValueError, 'archive member'):
                    tool.install(data, Path(directory) / 'tool')
                self.assertEqual(list(Path(directory).iterdir()), [])

    def test_version_probe_rejects_wrong_executable(self):
        with patch.object(tool.subprocess, 'run', return_value=subprocess.CompletedProcess([], 0, 'Version: 9.9.9\n', '')):
            with self.assertRaisesRegex(ValueError, 'unexpected version'):
                tool.probe(Path('/not-executed'))

    def test_prerequisite_follows_checkout_and_preserves_both_unsigned_runners(self):
        workflow = yaml.safe_load((ROOT / '.github/workflows/ios-simulator-sana.yml').read_text(encoding='utf-8'))
        job = workflow['jobs']['image']
        self.assertEqual(job['runs-on'], ['self-hosted', 'macOS', 'ARM64', 'nax'])
        steps = job['steps']
        checkout = next(i for i, step in enumerate(steps) if step.get('id') == 'checkout')
        prerequisite = next(i for i, step in enumerate(steps) if 'provision_simulator_xcodegen.py' in step.get('run', ''))
        self.assertLess(checkout, prerequisite)
        before = '\n'.join(step.get('run', '') for step in steps[:checkout])
        self.assertNotIn('command -v', before)
        self.assertIn('nax-macos-2', before)
        self.assertIn('nax-macos', before)
        self.assertIn('command -v python3.12', steps[prerequisite]['run'])
        self.assertNotIn('brew install', yaml.safe_dump(workflow))
        create = next(i for i, step in enumerate(steps) if 'ios_simulator_sana_device.py create' in step.get('run', ''))
        self.assertLess(prerequisite, create)
        execute = next(i for i, step in enumerate(steps) if 'ios_simulator_sana.py' in step.get('run', ''))
        self.assertLess(create, execute)
        self.assertIn('--owned-receipt "$RUNNER_TEMP/ios-simulator-sana/owned-simulator.json"',
                      steps[execute]['run'])

    def test_precheckout_failure_emits_cleanup_receipt_without_python_or_simctl(self):
        workflow = yaml.safe_load((ROOT / '.github/workflows/ios-simulator-sana.yml').read_text(encoding='utf-8'))
        cleanup = next(step for step in workflow['jobs']['image']['steps'] if step.get('name', '').startswith('Retire only'))
        self.assertEqual(cleanup['if'], 'always()')
        with tempfile.TemporaryDirectory() as directory:
            env = dict(os.environ, RUNNER_TEMP=directory, CHECKOUT_OUTCOME='skipped')
            # No checkout exists in cwd. Neither Python nor simctl may be invoked.
            result = subprocess.run(['/bin/bash', '-eo', 'pipefail', '-c', cleanup['run']],
                                    cwd=directory, env=env, capture_output=True, text=True, encoding='utf-8')
            self.assertEqual(result.returncode, 0, result.stderr)
            output = Path(directory) / 'ios-simulator-sana/simulator-cleanup.json'
            receipt = json.loads(output.read_text(encoding='utf-8'))
            self.assertEqual(receipt['status'], 'no-owned-simulator-receipt')
            self.assertFalse(receipt['resource_cleanup_performed'])
            self.assertEqual(receipt['model_validation'], 'not-established')
            (output.parent / 'owned-simulator.json').write_text('{}', encoding='utf-8')
            result = subprocess.run(['/bin/bash', '-eo', 'pipefail', '-c', cleanup['run']],
                                    cwd=directory, env=env, capture_output=True, text=True, encoding='utf-8')
            self.assertNotEqual(result.returncode, 0)
            self.assertEqual(json.loads(output.read_text(encoding='utf-8'))['status'], 'cleanup-refused')


if __name__ == '__main__':
    unittest.main()
