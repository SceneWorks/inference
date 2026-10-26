"""CI_IMPACT: ~0.1 s pure Python; no simulator, Metal, model, or download.

Covers the mandatory dispatch boundary and independent PNG-content witness for sc-16780.
"""
from __future__ import annotations

import json
import subprocess
from pathlib import Path
import struct
import sys
import tempfile
import unittest
from unittest.mock import patch
import zlib

from scripts.ci import ios_simulator_sana as sim
from scripts.ci import ios_simulator_sana_device as device
from scripts.ci.ios_simulator_sana_inventory import (REPOSITORY, REVISION, REQUIRED,
                                                    authenticate_source_bf16, materialize_source_bf16,
                                                    verify_copied_bf16)


def png(row: bytes) -> bytes:
    def chunk(kind: bytes, payload: bytes) -> bytes:
        return (struct.pack('>I', len(payload)) + kind + payload
                + struct.pack('>I', zlib.crc32(kind + payload)))
    header = struct.pack('>IIBBBBB', 256, 256, 8, 2, 0, 0, 0)
    rows = b''.join(b'\x00' + row for _ in range(256))
    return sim.PNG + chunk(b'IHDR', header) + chunk(b'IDAT', zlib.compress(rows)) + chunk(b'IEND', b'')


class SimulatorSanaTests(unittest.TestCase):
    def test_inventory_entrypoint_imports_under_workflow_isolation(self):
        script = str(sim.ROOT / 'scripts/ci/ios_simulator_sana_inventory.py')
        with tempfile.TemporaryDirectory() as directory:
            snapshot = (Path(directory) / 'models--SceneWorks--Sana_1600M_1024px_mlx'
                        / 'snapshots' / REVISION)
            for name in REQUIRED:
                file = snapshot / 'bf16' / name
                file.parent.mkdir(parents=True, exist_ok=True)
                file.write_bytes(b'fixture')
            receipt = Path(directory) / 'materialized.json'
            result = subprocess.run([sys.executable, '-I', script, '--materialize-bf16',
                                     str(snapshot), '--output', str(receipt)],
                                    capture_output=True, text=True, encoding="utf-8")
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertEqual(json.loads(receipt.read_text(encoding="utf-8"))['missing_before_fetch'], [])

    def test_bf16_materialization_uses_exact_pinned_cache_projection(self):
        with tempfile.TemporaryDirectory() as directory:
            snapshot = (Path(directory) / 'models--SceneWorks--Sana_1600M_1024px_mlx'
                        / 'snapshots' / REVISION)
            calls = []

            def download(**kwargs):
                calls.append(kwargs)
                for name in kwargs['allow_patterns']:
                    file = snapshot / name
                    file.parent.mkdir(parents=True, exist_ok=True)
                    file.write_bytes(b'fixture')
                return str(snapshot)

            result = materialize_source_bf16(snapshot, download)
            self.assertEqual(result['missing_before_fetch'], calls[0]['allow_patterns'])
            self.assertEqual(calls[0]['revision'], REVISION)
            self.assertEqual(calls[0]['repo_id'], REPOSITORY)
            self.assertEqual(calls[0]['cache_dir'], directory)
            materialize_source_bf16(snapshot, download)
            self.assertEqual(len(calls), 1)
            with self.assertRaisesRegex(ValueError, 'canonical Hub cache path'):
                materialize_source_bf16(Path(directory) / 'other' / REVISION, download)

    def test_runner_local_template_selection_and_owned_cleanup_identity(self):
        model = 'iPhone 17 Pro Max'
        template = {'name': model, 'udid': '11111111-1111-1111-1111-111111111111',
                    'isAvailable': True, 'state': 'Shutdown',
                    'deviceTypeIdentifier': 'com.apple.CoreSimulator.SimDeviceType.iPhone-17-Pro-Max'}
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            selection = root / 'template.json'
            with patch.object(device, 'devices', return_value=[
                ('com.apple.CoreSimulator.SimRuntime.iOS-26-5', template),
                ('com.apple.CoreSimulator.SimRuntime.iOS-26-6', template)]):
                chosen = device.select_template(model, selection, '123', '1', 'nax-macos-2')
            self.assertEqual(chosen['runtime'], 'com.apple.CoreSimulator.SimRuntime.iOS-26-6')
            owned = root / 'owned.json'
            with patch.object(device, 'devices', return_value=[(chosen['runtime'], template)]), \
                 patch.object(device, 'simctl', return_value='22222222-2222-2222-2222-222222222222'):
                receipt = device.create(selection, owned, '123', '1', root / 'env', 'nax-macos-2')
            self.assertEqual(receipt['template_udid'], template['udid'])
            self.assertEqual((root / 'env').read_text(encoding="utf-8"), 'SIMULATOR_UDID=' + receipt['owned_udid'] + '\n')
            with self.assertRaisesRegex(ValueError, 'does not match this workflow attempt'):
                device.cleanup(owned, root / 'cleanup.json', '123', '1', 'nax-macos')

    def test_bf16_source_authenticates_exact_pinned_hub_files(self):
        snapshot = Path('/some/cache/snapshots') / REVISION
        source = {'model': 'sana-1600m-mlx', 'repository': REPOSITORY, 'revision': REVISION}
        with patch('scripts.ci.ios_simulator_sana_inventory.verify_pinned_hub_files',
                   return_value={'status': 'hub_content_verified'}) as verify:
            self.assertEqual(authenticate_source_bf16(source, snapshot)['status'], 'hub_content_verified')
            verify.assert_called_once_with(REPOSITORY, REVISION, snapshot,
                                           tuple('bf16/' + name for name in sorted(REQUIRED)))
        with self.assertRaisesRegex(ValueError, 'pinned canonical'):
            authenticate_source_bf16({**source, 'revision': '0' * 40}, snapshot)

    def test_canonical_inventory_binds_exact_copied_bf16_tree(self):
        with tempfile.TemporaryDirectory() as directory:
            copied = Path(directory) / 'sana'
            copied.mkdir()
            files = []
            for name in sorted(REQUIRED):
                path = copied / name
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_bytes(name.encode())
                files.append({'path': 'bf16/' + name, 'size': path.stat().st_size,
                              'sha256': sim.sha256_file(path)})
            source = {'model': 'sana-1600m-mlx', 'revision': 'a' * 40,
                      'inventory_sha256': 'b' * 64, 'files': files}
            receipt = verify_copied_bf16(source, copied)
            self.assertEqual(receipt['source_inventory_sha256'], 'b' * 64)
            self.assertEqual(receipt['tier'], 'bf16')
            (copied / sorted(REQUIRED)[0]).write_bytes(b'changed')
            with self.assertRaisesRegex(ValueError, 'differs'):
                verify_copied_bf16(source, copied)

    def test_install_only_cannot_be_downgraded_by_later_media_flag(self):
        script = sim.ROOT / 'scripts/ios/run_smoke.sh'
        result = subprocess.run(['bash', str(script), '--simulator-functional', '--install-only',
                                 '--simulator', 'not-used', '--media'], capture_output=True, text=True, encoding="utf-8")
        self.assertEqual(result.returncode, 2)
        self.assertIn('cannot combine with other model modes', result.stderr)

    def test_dispatch_only_exact_runner(self):
        good = dict(GITHUB_EVENT_NAME='workflow_dispatch', RUNNER_NAME='nax-macos',
                    RUNNER_OS='macOS', RUNNER_ARCH='ARM64')
        sim.require_dispatch(good)
        sim.require_dispatch({**good, 'RUNNER_NAME': 'nax-macos-2'})
        for key, wrong in [('GITHUB_EVENT_NAME', 'push'), ('RUNNER_NAME', 'nax-macos-3'),
                           ('RUNNER_OS', 'Linux'), ('RUNNER_ARCH', 'X64')]:
            with self.subTest(key=key), self.assertRaises(ValueError):
                sim.require_dispatch({**good, key: wrong})

    def test_png_pixels_requires_nonconstant_decoded_rgb(self):
        with tempfile.TemporaryDirectory() as directory:
            file = Path(directory) / 'render.png'
            gradient = b''.join(bytes((i, i // 2, 255 - i)) for i in range(256))
            file.write_bytes(png(gradient))
            self.assertEqual(sim.png_pixels(file), (256, 256, 0, 255, 255))
            file.write_bytes(png(bytes((0, 128, 255)) * 256))
            with self.assertRaisesRegex(ValueError, 'spatially near-constant'):
                sim.png_pixels(file)
            file.write_bytes(png(bytes((4, 4, 4)) * 256))
            with self.assertRaisesRegex(ValueError, 'near-constant'):
                sim.png_pixels(file)
            bad = bytearray(png(gradient))
            bad[32] ^= 1
            file.write_bytes(bad)
            with self.assertRaisesRegex(ValueError, 'CRC'):
                sim.png_pixels(file)

    def test_reviewed_policy_cannot_lose_host_reserve_or_expand_cap(self):
        data = json.loads((sim.ROOT / 'scripts/ci/ios-simulator-sana-policy.json').read_text(encoding="utf-8"))
        with tempfile.TemporaryDirectory() as directory:
            file = Path(directory) / 'policy.json'
            file.write_text(json.dumps(data), encoding="utf-8")
            sim.policy_for_simulator(file)
            for field, value in [('hostFreeReserveBytes', 1),
                                 ('childFootprintCapBytes', 13 * 1024**3),
                                 ('deadlineSeconds', 7201)]:
                with self.subTest(field=field):
                    file.write_text(json.dumps({**data, field: value}), encoding="utf-8")
                    with self.assertRaises(ValueError):
                        sim.policy_for_simulator(file)


if __name__ == '__main__':
    unittest.main()
