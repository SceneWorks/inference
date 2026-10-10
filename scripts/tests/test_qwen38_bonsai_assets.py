"""Publisher closure, resumable cache materialization, and phase-boundary fixtures."""
from __future__ import annotations

import argparse
import hashlib
import json
import tempfile
import types
import unittest
from pathlib import Path
from unittest import mock

from scripts.release import qwen38_bonsai_assets as assets
from scripts.release import qwen38_bonsai_terminal as terminal
from scripts.tests import test_qwen38_bonsai_terminal as fixtures


class AssetTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.payload = b'publisher fixture'
        self.expected = {'path': 'model.bin', 'bytes': len(self.payload),
                         'sha256': hashlib.sha256(self.payload).hexdigest(), 'lfs_sha256': None,
                         'git_blob_sha1': hashlib.sha1(f'blob {len(self.payload)}\0'.encode() + self.payload).hexdigest()}
        self.model = {'key': 'fixture', 'repository': 'example/fixture', 'revision': 'c' * 40}
        self.snapshot = self.root / 'hub/models--example--fixture/snapshots' / self.model['revision']

    def tearDown(self):
        self.temp.cleanup()

    def download(self, **kwargs):
        self.assertEqual(kwargs['repo_id'], self.model['repository'])
        self.assertEqual(kwargs['revision'], self.model['revision'])
        self.assertIs(kwargs['token'], False)
        path = self.snapshot / kwargs['filename']
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(self.payload)
        return str(path)

    def test_missing_download_then_cached_reuse_verifies_publisher_bytes(self):
        fetch = mock.Mock(side_effect=self.download)
        rows = assets.provision_snapshot(self.model, self.snapshot, [self.expected], fetch)
        self.assertEqual(fetch.call_count, 1)
        self.assertTrue(rows[0]['publisher_identity_verified'])
        fetch.reset_mock()
        self.assertEqual(assets.provision_snapshot(self.model, self.snapshot, [self.expected], fetch), rows)
        fetch.assert_not_called()
        # Same-length corruption must neither be accepted nor evicted/replaced.
        path = self.snapshot / self.expected['path']
        path.write_bytes(b'x' * len(self.payload))
        with self.assertRaisesRegex(ValueError, 'publisher SHA256 mismatch'):
            assets.provision_snapshot(self.model, self.snapshot, [self.expected], fetch)
        self.assertEqual(path.read_bytes(), b'x' * len(self.payload))
        fetch.assert_not_called()

    def test_git_publisher_identity_and_download_location_cannot_drift(self):
        wrong = {**self.expected, 'git_blob_sha1': 'a' * 40}
        with self.assertRaisesRegex(ValueError, 'Git blob mismatch'):
            assets.provision_snapshot(self.model, self.snapshot, [wrong], self.download)
        with self.assertRaisesRegex(ValueError, 'exact publisher'):
            assets.provision_snapshot(self.model, self.root / 'wrong', [self.expected], mock.Mock())

    def test_frozen_gguf_download_scope_and_admitted_variants(self):
        model = terminal.load_model(Path('release/real-weight-models.toml'), 'bonsai-gguf')
        frozen = assets.frozen_model(model)
        self.assertEqual(model['download_files'], [item['path'] for item in frozen['files']])
        self.assertEqual(len([p for p in model['download_files'] if p.endswith('.gguf')]), 4)
        self.assertFalse(any('F16.gguf' in p and 'mmproj' not in p for p in model['download_files']))
        selected = assets.selected_files(frozen, [{'language_variant': 'ptq1', 'vision_variant': 'q8'}])
        gguf = {item['path'] for item in selected if item['path'].endswith('.gguf')}
        self.assertEqual(gguf, {'Ternary-Bonsai-2-27B-PTQ1_0.gguf', 'Ternary-Bonsai-2-27B-mmproj-Q8_0.gguf'})
        for key in ('bonsai-qwen38-parent', 'bonsai-mlx-2bit', 'bonsai-gguf', 'bonsai-qwen3vl-baseline'):
            assets.frozen_model(terminal.load_model(Path('release/real-weight-models.toml'), key))

    def test_native_inventory_must_match_the_frozen_publisher_closure(self):
        closure = self.root / 'closure.json'
        closure.write_text(json.dumps({'schema_version': 1, 'models': {'fixture': {
            **self.model, 'files': [self.expected]}}}), encoding='utf-8')
        inventory = {'files': [{'path': 'model.bin', 'size': len(self.payload), 'sha256': self.expected['sha256']}]}
        assets.verify_inventory(self.model, inventory, closure)
        inventory['files'][0]['sha256'] = 'a' * 64
        with self.assertRaisesRegex(ValueError, 'differs from frozen publisher identity'):
            assets.verify_inventory(self.model, inventory, closure)

    def provision_fixture(self, admitted=True, current_available=2000):
        # Reuse the same independently validated preflight builder as the sealed evidence tests.
        builder = fixtures.TerminalEvidenceTests()
        builder.root = self.root
        builder.runtime_sha = 'b' * 40
        manifest = builder.write_manifest([('fixture', self.model['revision'])])
        with manifest.open('a', encoding='utf-8') as handle:
            handle.write('repository="example/fixture"\nenvironment=["FIXTURE_SNAPSHOT"]\nexpected_files=["model.bin"]\n')
        builder.make_preflight('one', model_key='fixture', revision=self.model['revision'],
                               load_profile='mlx-unified', admitted=admitted)
        matrix = self.root / 'matrix.json'
        matrix.write_text(json.dumps({'schema_version': 1, 'suite': terminal.SUITE, 'cells': [
            {'id': 'one', 'backend': 'mlx', 'device': 'unified', 'model_key': 'fixture', 'load_profile': 'mlx-unified'}]}), encoding='utf-8')
        closure = self.root / 'closure.json'
        closure.write_text(json.dumps({'schema_version': 1, 'models': {'fixture': {
            **self.model, 'files': [self.expected]}}}), encoding='utf-8')
        args = argparse.Namespace(platform='macos', runtime_sha=builder.runtime_sha,
                                  evidence_root=self.root, matrix=matrix, manifest=manifest, closure=closure)
        fetch = mock.Mock(side_effect=self.download)
        with mock.patch.dict('os.environ', {'FIXTURE_SNAPSHOT': str(self.snapshot)}), \
             mock.patch.dict('sys.modules', {'huggingface_hub': types.SimpleNamespace(hf_hub_download=fetch)}), \
             mock.patch.object(terminal, 'source_identity', return_value={}), \
             mock.patch.object(terminal, 'physical_memory', return_value=(2000, current_available, None)):
            terminal.qualify_snapshots(argparse.Namespace(platform='macos', manifest=manifest,
                binding=['FIXTURE_SNAPSHOT=fixture'], output=self.root / 'snapshot-metadata-before.json'))
            result = terminal.provision_assets(args)
        return result, fetch

    def test_cold_cache_refreshes_metadata_after_publisher_verification(self):
        result, fetch = self.provision_fixture()
        self.assertEqual(result, 0)
        self.assertEqual(fetch.call_count, 1)
        before = json.loads((self.root / 'snapshot-metadata-before.json').read_text(encoding="utf-8"))
        after = json.loads((self.root / 'snapshot-metadata.json').read_text(encoding="utf-8"))
        self.assertFalse(before['all_metadata_qualified'])
        self.assertTrue(after['all_metadata_qualified'])
        self.assertTrue(after['publisher_verified_all'])
        self.assertEqual(after['runtime_sha'], 'b' * 40)
        self.assertEqual(after['publisher_closure_sha256'], terminal.sha256(self.root / 'closure.json'))
        self.assertFalse(json.loads((self.root / 'provision-report.json').read_text(encoding="utf-8"))['model_execution_performed'])

    def test_aggregate_disk_rejection_precedes_any_transfer(self):
        with mock.patch.object(terminal.shutil, 'disk_usage', return_value=types.SimpleNamespace(free=0)), \
             mock.patch.object(assets, 'provision_snapshot') as provision:
            with self.assertRaisesRegex(ValueError, 'insufficient disk capacity'):
                self.provision_fixture()
        provision.assert_not_called()
        receipt = json.loads((self.root / 'disk-admission.json').read_text(encoding='utf-8'))
        self.assertFalse(receipt['is_measured_peak'])
        self.assertFalse(receipt['filesystems'][0]['admitted'])
        self.assertEqual(receipt['filesystems'][0]['required_available_bytes'], 2 * len(self.payload))

    def test_rejected_preflight_never_downloads(self):
        result, fetch = self.provision_fixture(admitted=False)
        self.assertEqual(result, 1)
        fetch.assert_not_called()

    def test_capacity_lost_since_preflight_never_downloads(self):
        result, fetch = self.provision_fixture(current_available=0)
        self.assertEqual(result, 1)
        fetch.assert_not_called()

    def test_mlx_admission_uses_native_header_upper_bounds(self):
        manifest = Path('release/real-weight-models.toml')
        for key, expected in [('bonsai-qwen38-parent', 55572906808),
                              ('bonsai-mlx-2bit', 9514891750),
                              ('bonsai-qwen3vl-baseline', 17542650296)]:
            model = terminal.load_model(manifest, key)
            sizes = terminal.pinned_admission_sizes(model)
            self.assertEqual(terminal.host_load_bound(model, 'mlx-unified', sizes, None, None), expected)
            del model['admission_mlx_load_upper_bound_bytes']
            with self.assertRaisesRegex(ValueError, 'missing pinned MLX'):
                terminal.host_load_bound(model, 'mlx-unified', sizes, None, None)
        model = terminal.load_model(manifest, 'bonsai-gguf')
        for language, language_bound in [('pq2', 17061529952), ('ptq1', 15802009952)]:
            for vision, vision_bound in [('bf16', 2868438080), ('q8', 2566539200)]:
                sizes = terminal.pinned_admission_sizes(model, language, vision)
                self.assertEqual(terminal.host_load_bound(model, 'mlx-unified', sizes, language, vision),
                                 language_bound + vision_bound)

    def test_qualification_phases_are_mutually_exclusive(self):
        for preflight, provision in [('true', 'false'), ('false', 'true'), ('false', 'false')]:
            self.assertEqual(terminal.validate_phase(argparse.Namespace(preflight_only=preflight, provision_only=provision)), 0)
        with self.assertRaisesRegex(ValueError, 'mutually exclusive'):
            terminal.validate_phase(argparse.Namespace(preflight_only='true', provision_only='true'))
