"""CI_IMPACT: ~0.1 s pure Python and a mocked shell command; no simulator, Metal, model, or download.

Covers the mandatory dispatch boundary and independent PNG-content witness for sc-16780.
"""
from __future__ import annotations

import json
import os
import plistlib
import subprocess
from argparse import Namespace
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
    def test_missing_png_retains_fresh_report_before_owned_app_termination(self):
        source, nonce = 'b' * 40, 'a' * 32
        simulator = '22222222-2222-2222-2222-222222222222'
        bundle = 'com.michaeltrefry.SceneWorksSmokeSim123a1'
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            app, data, output = root / 'app', root / 'data', root / 'result'
            app.mkdir()
            docs = data / 'Documents'
            (docs / 'sana').mkdir(parents=True)
            (app / 'Info.plist').write_bytes(plistlib.dumps({
                'CFBundleIdentifier': bundle, 'CFBundleExecutable': 'Smoke'}))
            (app / 'Smoke').write_bytes(b'executable')
            (app / 'mlx.metallib').write_bytes(b'metallib')
            inventory = {'model': 'sana-1600m-mlx', 'inventory_sha256': 'c' * 64,
                         'revision': 'd' * 40, 'files': []}
            copied = {'source_inventory_sha256': inventory['inventory_sha256'],
                      'revision': inventory['revision'], 'tier': 'bf16', 'files': [],
                      'tier_inventory_sha256': sim.content_digest([])}
            for name, value in [('inventory.json', inventory), ('copied.json', copied)]:
                (root / name).write_text(json.dumps(value), encoding='utf-8')
            args = Namespace(simulator=simulator, bundle_id=bundle, source_sha=source,
                             inventory=root / 'inventory.json', copied_inventory=root / 'copied.json',
                             owned_receipt=root / 'owned.json',
                             policy=sim.ROOT / 'scripts/ci/ios-simulator-sana-policy.json', output=output)
            calls = []

            def command(*parts, timeout=20, env=None):
                if parts[:2] == ('git', 'rev-parse'):
                    return source
                if parts[:4] == ('xcrun', 'simctl', 'get_app_container', simulator):
                    return str(app if parts[-1] == 'app' else data)
                if parts[:3] == ('xcrun', 'simctl', 'launch'):
                    assert env is not None
                    self.assertEqual(env['SIMCTL_CHILD_IOS_CAMPAIGN_BUILD_SHA'], source)
                    (docs / 'smoke-report.txt').write_text(
                        f'SMOKE: PASS\nSIMULATOR_NONCE {env["SIMCTL_CHILD_IOS_SIMULATOR_SANA_NONCE"]}\n'
                        f'SOURCE_SHA {source}\n  [ok] SANA image generation (E5)\n'
                        'SIMULATOR FUNCTIONAL ONLY; no phone admission, memory fit, or performance claim',
                        encoding='utf-8')
                    (docs / 'sana-progress.txt').write_text('no candidate: unserved tile128\n', encoding='utf-8')
                    return bundle + ': 42'
                if parts[:3] == ('xcrun', 'simctl', 'terminate'):
                    self.assertTrue((output / 'failure-report.txt').is_file())
                    calls.append('terminated after diagnostic copy')
                    return ''
                self.fail(f'unexpected command: {parts}')

            with patch.dict(os.environ, {'RUNNER_NAME': 'nax-macos-2'}), \
                 patch.object(sim, 'require_dispatch'), patch.object(sim, 'command', side_effect=command), \
                 patch.object(sim, 'owned_booted_device', return_value=('iOS-26-5', {'name': 'SceneWorksSana-123-1'})), \
                 patch.object(sim.safety, 'SystemProbe', return_value=object()), \
                 patch.object(sim.safety, 'read_host', return_value=(32 * 1024**3, {})), \
                 patch.object(sim.safety, 'proc_pid_rusage_footprint', return_value=(100, 100)), \
                 patch.object(sim, 'app_identity', return_value=(str((app / 'Smoke').resolve()), 1)), \
                 patch.object(sim.os, 'urandom', return_value=bytes.fromhex(nonce)):
                with self.assertRaisesRegex(ValueError, 'no exact PNG'):
                    sim.run(args)
            self.assertEqual(calls, ['terminated after diagnostic copy'])
            self.assertFalse((docs / ('simulator-go-' + nonce)).exists())
            self.assertFalse((output / 'result.json').exists())
            self.assertFalse(json.loads((output / 'failure-diagnostics.json').read_text(encoding='utf-8'))['accepted'])

    def test_failed_simulator_report_and_progress_are_retained_only_as_bounded_diagnostics(self):
        policy = sim.policy_for_simulator(sim.ROOT / 'scripts/ci/ios-simulator-sana-policy.json')
        nonce, source = 'a' * 32, 'b' * 40
        with tempfile.TemporaryDirectory() as directory:
            docs, output = Path(directory) / 'Documents', Path(directory) / 'result'
            docs.mkdir()
            output.mkdir()
            report = f'SMOKE: PASS\nSIMULATOR_NONCE {nonce}\nSOURCE_SHA {source}\n  [ok] SANA image generation (E5)\n'
            (docs / 'smoke-report.txt').write_text(report, encoding='utf-8')
            (docs / 'sana-progress.txt').write_text('no candidate: unserved tile128\n', encoding='utf-8')
            sim.retain_failure_diagnostics(docs, output, policy, nonce, source)
            receipt = json.loads((output / 'failure-diagnostics.json').read_text(encoding='utf-8'))
            self.assertFalse(receipt['accepted'])
            self.assertIn('diagnostic-only', receipt['claim'])
            self.assertTrue(receipt['files']['smoke-report.txt']['nonce_seen'])
            self.assertTrue(receipt['files']['smoke-report.txt']['source_seen'])
            self.assertEqual((output / 'failure-report.txt').read_text(encoding='utf-8'), report)
            self.assertIn('unserved tile128', (output / 'failure-progress.txt').read_text(encoding='utf-8'))
            (docs / 'smoke-report.txt').unlink()
            (docs / 'smoke-report.txt').symlink_to(docs / 'sana-progress.txt')
            (docs / 'sana-progress.txt').write_bytes(b'x' * (policy.stderr_cap_bytes + 1))
            second = Path(directory) / 'second'
            second.mkdir()
            sim.retain_failure_diagnostics(docs, second, policy, nonce, source)
            self.assertFalse((second / 'failure-report.txt').exists())
            self.assertFalse((second / 'failure-progress.txt').exists())

    def test_controller_accepts_only_this_attempts_booted_owned_iphone(self):
        owned = '22222222-2222-2222-2222-222222222222'
        template = '11111111-1111-1111-1111-111111111111'
        runtime = 'com.apple.CoreSimulator.SimRuntime.iOS-26-5'
        device_type = 'com.apple.CoreSimulator.SimDeviceType.iPhone-17-Pro-Max'
        receipt = {'owned_udid': owned, 'owned_name': 'SceneWorksSana-123-1',
                   'template_udid': template, 'template_name': 'iPhone 17 Pro Max',
                   'run_id': '123', 'attempt': '1', 'runner': 'nax-macos-2',
                   'runtime': runtime, 'device_type': device_type}
        booted = {'udid': owned, 'name': receipt['owned_name'], 'state': 'Booted',
                  'isAvailable': True, 'deviceTypeIdentifier': device_type}
        env = {'GITHUB_RUN_ID': '123', 'GITHUB_RUN_ATTEMPT': '1', 'RUNNER_NAME': 'nax-macos-2',
               'SIMULATOR_SELECTOR': 'iPhone 17 Pro Max'}
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'owned.json'
            path.write_text(json.dumps(receipt), encoding='utf-8')
            with patch.object(sim, 'command', return_value=json.dumps({'devices': {runtime: [booted]}})):
                self.assertEqual(sim.owned_booted_device(owned, path, env), (runtime, booted))
                # The old "iPhone" display-name selector rejected this real owned device.
                with self.assertRaisesRegex(ValueError, 'receipt does not match'):
                    sim.owned_booted_device(template, path, env)
                with self.assertRaisesRegex(ValueError, 'receipt does not match'):
                    sim.owned_booted_device(owned, path, {**env, 'GITHUB_RUN_ATTEMPT': '2'})
                with self.assertRaisesRegex(ValueError, 'receipt does not match'):
                    sim.owned_booted_device(owned, path, {**env, 'RUNNER_NAME': 'nax-macos'})
                with self.assertRaisesRegex(ValueError, 'receipt does not match'):
                    sim.owned_booted_device(owned, path, {**env, 'SIMULATOR_SELECTOR': 'iPhone 17'})
            for altered in ({**booted, 'name': 'iPhone 17 Pro Max'},
                            {**booted, 'deviceTypeIdentifier': 'other'},
                            {**booted, 'state': 'Shutdown'},
                            {**booted, 'isAvailable': False}):
                with self.subTest(altered=altered), patch.object(sim, 'command', return_value=json.dumps({'devices': {runtime: [altered]}})):
                    with self.assertRaisesRegex(ValueError, 'not uniquely booted'):
                        sim.owned_booted_device(owned, path, env)
            with patch.object(sim, 'command', return_value=json.dumps({'devices': {runtime: [booted, booted]}})):
                with self.assertRaisesRegex(ValueError, 'not uniquely booted'):
                    sim.owned_booted_device(owned, path, env)
            with patch.object(sim, 'command', return_value=json.dumps({'devices': {'other': [booted]}})):
                with self.assertRaisesRegex(ValueError, 'not uniquely booted'):
                    sim.owned_booted_device(owned, path, env)

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

    def test_simulator_app_build_uses_generic_arm64_destination_then_exact_install(self):
        source = (sim.ROOT / 'scripts/ios/run_smoke.sh').read_text(encoding='utf-8')
        start = source.index('  xcodebuild \\', source.index('say "generating and building the app (iphonesimulator)"'))
        end = source.index('    build 2>&1 | tail -20', start) + len('    build 2>&1 | tail -20')
        with tempfile.TemporaryDirectory() as directory:
            args_file = Path(directory) / 'xcodebuild-args.txt'
            script = 'set -euo pipefail\nxcodebuild() { printf "%s\\n" "$@" > "$ARGS_FILE"; }\n' + source[start:end]
            result = subprocess.run(['bash', '-c', script], capture_output=True, text=True, encoding='utf-8',
                                    env={**os.environ, 'CONFIG': 'Release', 'DERIVED': directory,
                                         'ARGS_FILE': str(args_file), 'SIMULATOR': '11111111-1111-1111-1111-111111111111'})
            self.assertEqual(result.returncode, 0, result.stderr)
            args = args_file.read_text(encoding='utf-8').splitlines()
        self.assertEqual(args[args.index('-destination') + 1], 'generic/platform=iOS Simulator')
        self.assertEqual(args[args.index('-sdk') + 1], 'iphonesimulator')
        self.assertIn('ARCHS=arm64', args)
        self.assertNotIn('id=11111111-1111-1111-1111-111111111111', args)
        self.assertIn('xcrun simctl install "$SIMULATOR" "$APP"', source)

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
