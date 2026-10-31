"""Mock collector integration: no Cargo, model, weights, GPU or real resource probes."""
import contextlib
import copy
import importlib.util
import json
import os
from pathlib import Path
import sys
import subprocess
import tempfile
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from scripts import media_campaign_supervisor as safety
from scripts.ci import model_proof_resources as resources


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


proof = load('collector_test_proof', ROOT / 'scripts/ci/model_inference_proofs.py')
policy = load('collector_test_policy', ROOT / 'scripts/check_model_inference_policy.py')


def request(backend='darwin-mlx'):
    return {'selection': 'mlx-gen-test', 'policy': {
        'schemaVersion': 1, 'backend': backend, 'deadlineSeconds': 10,
        'pollMillis': 10, 'termGraceMillis': 50,
        'hostFreeReserveBytes': 100, 'childFootprintCapBytes': 1000000,
        'stdoutCapBytes': 1000000, 'stderrCapBytes': 1000000, 'eventCapBytes': 1000000}}


def case():
    return {'package': 'mlx-gen-test', 'target': 'mlx_gen_test', 'target_kind': 'lib',
            'fqn': 'tests::mock', 'source': 'case.rs', 'execution_role': 'direct',
            'backend': 'cpu', 'fixture_identity': 'protocol mock; NO NN',
            'configuration_source': 'case.rs::mock', 'output_contract': 'mock logs only',
            'input_contract': {'fixture_kind': 'source-defined', 'source_identity': 'case.rs::mock', 'requirements': []}}


def external_case(grade='reader-binding'):
    row = case()
    row['input_contract'] = {'fixture_kind': 'external', 'source_identity': 'case.rs::mock/actual-read',
        'requirements': [{'slot': 'model-weights', 'role': 'model-weights', 'readers': ['mlx-gen::Weights::from_file'],
                          'minimum_collections': 1, 'minimum_items': 1, 'required_grade': grade, 'scope': 'case'}]}
    path = {'display': '/MOCK-ONLY/actual-input', 'os_encoded_hex': b'/MOCK-ONLY/actual-input'.hex()}
    item = {'literal_path': path, 'absolute_lexical_path': path, 'os_encoding': 'rust-osstr-encoded',
            'provenance': {'kind': 'successful-reader-binding', 'content_integrity': 'not_attested'}}
    packet = {'schema_version': 1, 'reader': 'mlx-gen::Weights::from_file', 'role': 'model-weights',
              'complete': True, 'items': [item]}
    return row, 'MODEL_INFERENCE_INPUT_USED ' + row['fqn'] + ' ' + json.dumps(packet) + '\n'


def adapter_spec():
    return {'kind': 'lora', 'rank': 2, 'dtype': 'F32', 'per_target_alpha': True,
            'max_bytes': 4096, 'targets': {'blocks.0.to_q': [3, 4]}}


def flux3_adapter_spec():
    return {**adapter_spec(), 'metadata_format': 'flux3', 'alpha': 2.0, 'per_target_alpha': False}


def adapter_bytes(mutate=None, *, flux3=False):
    # Format-only fixture: zero bytes, no tensor library or learned execution.
    header = {'__metadata__': {'networkType': 'lora', 'rank': '2', 'alpha': '2'}}
    factors = [('lora_A.weight', [2, 4]), ('lora_B.weight', [3, 2])]
    if flux3:
        # Native export_adapter metadata, without product changes or invented aliases.
        header['__metadata__'] = {
            'base_model': 'MOCK model identity; not attested', 'base_revision': 'MOCK revision',
            'network_type': 'lora', 'rank': '2', 'lora_alpha': '2', 'scaling_convention': 'alpha/rank',
            'training_identity': '{}', 'lora_adapter_metadata': json.dumps({
                'peft_type': 'LORA', 'r': 2, 'lora_alpha': 2.0, 'use_rslora': False,
                'use_dora': False, 'rank_pattern': {}, 'alpha_pattern': {}})}
    else:
        factors.append(('alpha', [1]))
    end = 0
    for name, shape in factors:
        count = 1
        for dim in shape:
            count *= dim
        header['blocks.0.to_q.' + name] = {'dtype': 'F32', 'shape': shape, 'data_offsets': [end, end + 4*count]}
        end += 4*count
    if mutate:
        mutate(header)
    raw = json.dumps(header, separators=(',', ':')).encode()
    raw += b' ' * (-len(raw) % 8)
    return len(raw).to_bytes(8, 'little') + raw + bytes(end)


class Collector(unittest.TestCase):
    def test_native_flux3_metadata_requires_explicit_typed_source_contract(self):
        proof.validate_generated_lora_bytes(adapter_bytes(flux3=True), flux3_adapter_spec())
        for changes in ({'metadata_format': 'unknown'}, {'per_target_alpha': True},
                        {'alpha': None}, {'alpha': True}, {'alpha': float('inf')}, {'alpha': 0}):
            with self.subTest(changes=changes), self.assertRaises(ValueError):
                proof.validate_generated_adapter_spec({**flux3_adapter_spec(), **changes})
        for spec, raw in ((adapter_spec(), adapter_bytes(flux3=True)),
                          (flux3_adapter_spec(), adapter_bytes())):
            with self.assertRaises(ValueError):
                proof.validate_generated_lora_bytes(raw, spec)

    def test_native_flux3_rejects_conflicting_metadata_and_wrong_factors(self):
        def peft_change(header, **changes):
            metadata = header['__metadata__']
            peft = json.loads(metadata['lora_adapter_metadata'])
            metadata['lora_adapter_metadata'] = json.dumps({**peft, **changes})
        mutations = [
            lambda h: h['__metadata__'].pop('rank'),
            lambda h: h['__metadata__'].pop('lora_adapter_metadata'),
            lambda h: h['__metadata__'].update(network_type='checkpoint'),
            lambda h: h['__metadata__'].update(networkType='lora'),
            lambda h: h['__metadata__'].update(alpha='2'),
            lambda h: h['__metadata__'].update(lora_alpha='nan'),
            lambda h: h['__metadata__'].update(lora_alpha='3'),
            lambda h: h['__metadata__'].update(rank='3'),
            lambda h: h['__metadata__'].update(scaling_convention='alpha/sqrt(rank)'),
            lambda h: h['__metadata__'].update(lora_adapter_metadata='{"r":2,"r":2}'),
            lambda h: peft_change(h, lora_alpha=3),
            lambda h: peft_change(h, use_dora=True),
            lambda h: peft_change(h, rank_pattern={'blocks.0.to_q': 3}),
            lambda h: h.update({'model.weight': h.pop('blocks.0.to_q.lora_A.weight')}),
            lambda h: h['blocks.0.to_q.lora_A.weight'].update(shape=[4, 2]),
            lambda h: h['blocks.0.to_q.lora_B.weight'].update(data_offsets=[0, 24]),
        ]
        for index, mutate in enumerate(mutations):
            with self.subTest(index=index), self.assertRaises(ValueError):
                proof.validate_generated_lora_bytes(adapter_bytes(mutate, flux3=True), flux3_adapter_spec())

    def test_native_flux3_survives_collector_without_relaxing_generic_safetensors_ban(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary).resolve()
            row = case()
            name = 'trained/adapter.safetensors'
            row['output_artifacts'] = [name]
            row['generated_adapter_artifacts'] = {name: flux3_adapter_spec()}
            raw = adapter_bytes(flux3=True)
            status, _, verified, _, output = self._mock_run(root, row_override=row, generated_outputs={name: raw})
            self.assertEqual(status, 0)
            self.assertTrue(verified['accepted_model_run'])  # MOCK ONLY.
            artifact = output / '00000/generated' / name
            self.assertEqual(artifact.read_bytes(), raw)
            with self.assertRaisesRegex(ValueError, 'checkpoint/model payload'):
                proof.publish_file(artifact, output / 'undeclared.safetensors', output, 4096)

    def test_generated_adapter_requires_exact_declared_output_contract(self):
        row = case()
        row['output_artifacts'] = ['trained/adapter.safetensors']
        row['generated_adapter_artifacts'] = {'trained/adapter.safetensors': adapter_spec()}
        self.assertEqual(proof.validate_generated_adapter_artifacts(row), row['generated_adapter_artifacts'])
        for path in ('unlisted.safetensors', '../input.safetensors', '/input.safetensors', 'C:\\input.safetensors'):
            bad = copy.deepcopy(row)
            bad['generated_adapter_artifacts'] = {path: adapter_spec()}
            with self.subTest(path=path), self.assertRaises(ValueError):
                proof.validate_generated_adapter_artifacts(bad)
        for key, value in [('kind', 'checkpoint'), ('rank', 4096), ('dtype', 'I64'), ('max_bytes', 0), ('targets', {})]:
            bad = adapter_spec(); bad[key] = value
            with self.subTest(key=key), self.assertRaises(ValueError):
                proof.validate_generated_adapter_spec(bad)

    def test_generated_adapter_header_rejects_model_keys_shape_dtype_rank_and_payload(self):
        proof.validate_generated_lora_bytes(adapter_bytes(), adapter_spec())
        mutations = [
            lambda h: h.update({'model.weight': h.pop('blocks.0.to_q.lora_A.weight')}),
            lambda h: h.pop('blocks.0.to_q.lora_B.weight'),
            lambda h: h['blocks.0.to_q.lora_A.weight'].update(shape=[4, 2]),
            lambda h: h['blocks.0.to_q.lora_A.weight'].update(dtype='BF16'),
            lambda h: h['blocks.0.to_q.lora_A.weight'].update(data_offsets=[0, 28]),
            lambda h: h['blocks.0.to_q.lora_B.weight'].update(data_offsets=[0, 24]),
            lambda h: h['__metadata__'].update(rank='3'),
            lambda h: h['__metadata__'].update(alpha='nan'),
        ]
        for index, mutate in enumerate(mutations):
            with self.subTest(index=index), self.assertRaises(ValueError):
                proof.validate_generated_lora_bytes(adapter_bytes(mutate), adapter_spec())
        for raw in (adapter_bytes()[:-1], adapter_bytes()+b'extra', b'\xff'*8+b'{}', b'x'*4097):
            with self.assertRaises(ValueError):
                proof.validate_generated_lora_bytes(raw, adapter_spec())
        valid = adapter_bytes()
        size = int.from_bytes(valid[:8], 'little')
        utf16_header = valid[8:8+size].decode('utf-8').encode('utf-16')
        with self.assertRaises(ValueError):
            proof.validate_generated_lora_bytes(
                len(utf16_header).to_bytes(8, 'little') + utf16_header + valid[8+size:], adapter_spec())

    def test_adapter_publisher_preserves_checkpoint_input_link_and_size_bans(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary).resolve()
            generated = root / 'generated'; generated.mkdir()
            published = root / 'published'; published.mkdir()
            source = generated / 'adapter.safetensors'; source.write_bytes(adapter_bytes())
            destination = published / 'adapter.safetensors'
            # Even a well-formed LoRA is forbidden without explicit source permission.
            with self.assertRaises(ValueError):
                proof.publish_file(source, destination, generated, 4096, destination_root=published)
            result = proof.publish_file(source, destination, generated, 4096, destination_root=published, adapter=adapter_spec())
            self.assertEqual(result['bytes'], len(adapter_bytes()))
            self.assertEqual(destination.read_bytes(), source.read_bytes())
            external = root / 'existing-input.safetensors'; external.write_bytes(adapter_bytes())
            with self.assertRaises(ValueError):
                proof.publish_file(external, destination, generated, 4096, destination_root=published, adapter=adapter_spec())
            source.unlink(); os.link(external, source)
            with self.assertRaises(ValueError):
                proof.publish_file(source, destination, generated, 4096, destination_root=published, adapter=adapter_spec())
            source.unlink(); source.symlink_to(external)
            with self.assertRaises(ValueError):
                proof.publish_file(source, destination, generated, 4096, destination_root=published, adapter=adapter_spec())
            source.unlink(); source.write_bytes(adapter_bytes())
            with self.assertRaises(ValueError):
                proof.publish_file(source, destination, generated, 32, destination_root=published, adapter=adapter_spec())
            source.write_bytes(adapter_bytes(lambda h: h.update({'model.weight': h.pop('blocks.0.to_q.lora_A.weight')})))
            with self.assertRaises(ValueError):
                proof.publish_file(source, destination, generated, 4096, destination_root=published, adapter=adapter_spec())
    def test_global_cuda_card_bound_is_labeled_without_attributed_peak(self):
        # CI_IMPACT: pure result-label check; adds no model, Cargo, GPU or process work.
        bounded = safety.RunResult(1, 0, 0, None, 0, 0, 0.1,
                                   ({'phase': 'cuda-global-used-upper-bound-sample',
                                     'globalUsedUpperBoundBytes': 1024},), {})
        self.assertEqual(proof.gpu_peak_observation(bounded),
                         'global-used-upper-bound-sampled/not-child-attributed')
        self.assertEqual(proof.gpu_peak_observation(safety.RunResult(1, 0, 0, 1024, 0, 0, 0.1, (), {})),
                         'sampled')
        multi = safety.MultiCudaRunResult(1, 0, 0, None, 0, 0, 0.1, (), {},
                                         ({'globalUsedUpperBoundSampleCount': 1},))
        self.assertEqual(proof.gpu_peak_observation(multi),
                         'global-used-upper-bound-sampled/not-child-attributed')

    @unittest.skipIf(os.name == 'nt', 'Unix proxy contract; Windows pinned binaries have separate tests')
    def test_toolchain_uses_declaration_and_verified_proxies_for_actual_commands(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary).resolve()
            (root / 'rust-toolchain.toml').write_text('[toolchain]\nchannel = "repository-channel"\n', encoding='utf-8')
            bin_dir = root / 'cargo-home/bin'
            bin_dir.mkdir(parents=True)
            suffix = '.exe' if os.name == 'nt' else ''
            for tool in ('cargo', 'rustc', 'rustup'):
                path = bin_dir / (tool + suffix)
                path.write_text('MOCK; never executed', encoding='utf-8')
                path.chmod(0o755)
            env = {'CARGO_HOME': str(bin_dir.parent), 'PATH': '/shadow', 'RUSTUP_TOOLCHAIN': 'repository-channel'}
            commands = []
            def inspect(argv, **kwargs):
                commands.append(argv)
                self.assertEqual(kwargs['env']['RUSTUP_TOOLCHAIN'], 'repository-channel')
                self.assertEqual(kwargs['cwd'], root)
                self.assertTrue(kwargs['check'])
                if 'which' in argv:
                    self.assertEqual(argv[1:4], ['which', '--toolchain', 'repository-channel'])
                    output = str(root / 'installed/bin' / argv[-1])
                else:
                    self.assertEqual(kwargs['env']['RUSTC'], str(bin_dir / ('rustc' + suffix)))
                    output = 'MOCK compiler version; not actually run'
                return subprocess.CompletedProcess(argv, 0, stdout=output + '\n')
            with patch.object(proof.subprocess, 'run', side_effect=inspect), \
                    patch.object(proof, 'configure_native_darwin_build',
                                 return_value={'mechanism': 'MOCK linker; not executed'}) as native:
                value = proof.verified_toolchain(env, root=root)
            native.assert_called_once_with(env, [str(bin_dir / ('rustc' + suffix))],
                                           root=root, system=proof.platform.system())
            self.assertEqual(value['native_linker']['mechanism'], 'MOCK linker; not executed')
            self.assertEqual(value['channel'], 'repository-channel')
            self.assertEqual(value['inherited_rustup_toolchain'], 'repository-channel')
            self.assertEqual(env['PATH'].split(os.pathsep)[0], str(bin_dir))
            self.assertEqual(len(commands), 4)
            self.assertEqual(commands[-2][0], str(bin_dir / ('cargo' + suffix)))
            self.assertEqual(commands[-1][0], str(bin_dir / ('rustc' + suffix)))
            self.assertEqual(value['rustc_version'], 'MOCK compiler version; not actually run')
            env['RUSTC'] = '/unapproved/compiler'
            with patch.object(proof.subprocess, 'run', side_effect=inspect):
                with self.assertRaisesRegex(ValueError, 'RUSTC differs'):
                    proof.verified_toolchain(env, root=root)

    def test_toolchain_rejects_unexpected_override_before_any_compiler_or_download(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary).resolve()
            (root / 'rust-toolchain.toml').write_text('[toolchain]\nchannel = "repository-channel"\n', encoding='utf-8')
            with patch.object(proof.subprocess, 'run') as command:
                with self.assertRaisesRegex(ValueError, 'RUSTUP_TOOLCHAIN differs'):
                    proof.verified_toolchain({'RUSTUP_TOOLCHAIN': 'host-override'}, root=root)
                command.assert_not_called()

    def test_windows_toolchain_uses_real_pinned_binaries_without_listener_cache_proxies(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary).resolve()
            rustup_home = root / 'rustup'
            bin_dir = rustup_home / 'toolchains/repository-channel-x86_64-pc-windows-msvc/bin'
            bin_dir.mkdir(parents=True)
            for tool in ('cargo', 'rustc'):
                binary = bin_dir / f'{tool}.exe'
                binary.write_text('MOCK; never executed', encoding='utf-8')
                binary.chmod(0o755)
            env = {'USERPROFILE': str(root), 'RUSTUP_HOME': str(rustup_home),
                   'CARGO_HOME': str(root / 'listener-cache-without-bin'),
                   'MODEL_PROOF_RUST_BIN': str(bin_dir), 'PATH': '/shadow'}
            declaration = b'[toolchain]\nchannel = "repository-channel"\n'

            def inspect(argv, **kwargs):
                self.assertEqual(kwargs['env']['PATH'].split(os.pathsep)[0], str(bin_dir))
                self.assertEqual(kwargs['env']['RUSTC'], str(bin_dir / 'rustc.exe'))
                output = ('cargo repository-channel (mock)\n' if argv[0].endswith('cargo.exe')
                          else 'rustc repository-channel (mock)\nrelease: repository-channel\n')
                return subprocess.CompletedProcess(argv, 0, stdout=output)

            with patch.object(proof.shutil, 'which', side_effect=lambda name, path: str(bin_dir / f'{name}.exe')), \
                 patch.object(proof.subprocess, 'run', side_effect=inspect):
                value = proof.verified_windows_toolchain(env, root, declaration, 'repository-channel', None)
                self.assertEqual(value['binaries']['cargo'], str(bin_dir / 'cargo.exe'))
                self.assertEqual(value['channel'], 'repository-channel')
                self.assertEqual(env['CARGO_HOME'], str(root / 'listener-cache-without-bin'))
                env['MODEL_PROOF_RUST_BIN'] = str(root / 'wrong-toolchain/bin')
                with self.assertRaisesRegex(ValueError, 'differs from the pinned'):
                    proof.verified_windows_toolchain(env, root, declaration, 'repository-channel', None)
                env['MODEL_PROOF_RUST_BIN'] = str(bin_dir)
                (bin_dir / 'rustc.exe').unlink()
                with self.assertRaisesRegex(ValueError, 'binaries are missing'):
                    proof.verified_windows_toolchain(env, root, declaration, 'repository-channel', None)

    def test_windows_toolchain_rejects_reported_version_drift(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary).resolve()
            rustup_home = root / 'rustup'
            bin_dir = rustup_home / 'toolchains/1.96.0-x86_64-pc-windows-msvc/bin'
            bin_dir.mkdir(parents=True)
            for tool in ('cargo', 'rustc'):
                binary = bin_dir / f'{tool}.exe'
                binary.write_text('MOCK; never executed', encoding='utf-8')
                binary.chmod(0o755)
            env = {'USERPROFILE': str(root), 'RUSTUP_HOME': str(rustup_home),
                   'CARGO_HOME': str(root / 'listener-cache'), 'MODEL_PROOF_RUST_BIN': str(bin_dir),
                   'PATH': '/shadow'}
            def inspect(argv, **kwargs):
                output = ('cargo 1.97.0 (mock)\n' if argv[0].endswith('cargo.exe')
                          else 'rustc 1.96.0 (mock)\nrelease: 1.96.0\n')
                return subprocess.CompletedProcess(argv, 0, stdout=output)
            with patch.object(proof.shutil, 'which', side_effect=lambda name, path: str(bin_dir / f'{name}.exe')), \
                 patch.object(proof.subprocess, 'run', side_effect=inspect):
                with self.assertRaisesRegex(ValueError, 'version differs'):
                    proof.verified_windows_toolchain(env, root, b'pin', '1.96.0', None)

    def test_bridge_accepts_only_fixed_backend_and_request_envelope(self):
        raw = json.dumps({'backend': 'mlx', 'request': request()})
        self.assertEqual(proof.binding(raw), ('mlx', request()))
        for invalid in ('', '{}', 'null', '{"backend":"anything","request":{}}',
                        json.dumps({'backend': 'mlx', 'request': request(), 'command': 'false'})):
            with self.assertRaises((ValueError, TypeError)):
                proof.binding(invalid)

    def test_rustdoc_model_seams_are_not_ordinary_even_in_should_panic(self):
        for tag in ('rust', 'rust,should_panic', ''):
            source = '/// ```' + tag + '\n/// model.forward(&x)?;\n/// ```\npub struct X;'
            self.assertTrue(policy.model_doc_errors(source, 'case.rs'))
        for tag in ('rust,no_run', 'no_run', 'rust,ignore', 'compile_fail', 'text', 'sh'):
            source = '/// ```' + tag + '\n/// model.forward(&x)?;\n/// ```\npub struct X;'
            self.assertEqual(policy.model_doc_errors(source, 'case.rs'), [])
        self.assertEqual(policy.model_doc_errors('/// ```rust\n/// a.matmul(&b)?;\n/// ```', 'raw.rs'), [])
        self.assertEqual(policy.model_doc_errors('///  1. Explanation\n///     `loss.backward()` is a description.', 'prose.rs'), [])

    def test_existing_ignore_reason_is_preserved_but_marker_requires_registry(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary).resolve()
            path = root / 'crates/case.rs'
            path.parent.mkdir()
            path.write_text('#[test]\n#[ignore = "original prerequisite reason"]\nfn mock() { println!("MODEL_INFERENCE_BODY_COMPLETED tests::mock"); }', encoding="utf-8")
            row = case(); row['source'] = 'crates/case.rs'
            self.assertEqual(policy.registry_errors(root, [row]), [])
            self.assertTrue(policy.registry_errors(root, []))

    def test_destination_ancestor_and_hardlink_cannot_overwrite_other_files(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary).resolve()
            source = root / 'source.png'; source.write_bytes(b'output')
            outside = root / 'outside'; outside.mkdir()
            destination = root / 'dest'; destination.mkdir()
            (destination / 'link').symlink_to(outside, target_is_directory=True)
            with self.assertRaises(ValueError):
                proof.publish_file(source, destination / 'link/out.png', root, 100)
            victim = outside / 'private.png'; victim.write_bytes(b'private')
            os.link(victim, destination / 'hard.png')
            with self.assertRaises(ValueError):
                proof.publish_file(source, destination / 'hard.png', root, 100)
            self.assertEqual(victim.read_bytes(), b'private')

    def _mock_run(self, root, *, fail=False, missing=False, preseal=None, row_override=None, input_body='', observer_events=True, generated_outputs=None, backend='mlx', supervisor_module=None):
        row = case() if row_override is None else row_override
        if missing:
            row['required_environment'] = ['MODEL_PROOF_TEST_PREREQUISITE_ABSENT']
        source = root / row['source']
        source.parent.mkdir(parents=True, exist_ok=True)
        source.write_text('// test fixture; no actual Rust execution', encoding='utf-8')
        mock_request = request('windows-cuda' if backend == 'candle-cuda' else 'darwin-mlx')
        if backend == 'candle-cuda':
            mock_request['policy'].update(cudaDeviceUuid=row['required_cuda_device_uuid'], gpuFreeReserveBytes=100, childGpuCapBytes=1000000)
        source_safety = supervisor_module or safety
        def toolchain(env, **_kwargs):
            if backend == 'mlx':
                self.assertEqual(env['MACOSX_DEPLOYMENT_TARGET'], row.get('deployment_target', '15.0'))
            env['CARGO_ENCODED_RUSTFLAGS'] = 'MOCK_NATIVE_LINKER_FLAGS'
            return {'toolchain': 'MOCK; no compiler invoked'}
        def preparation(_row, env):
            if backend == 'mlx':
                self.assertEqual(env['MACOSX_DEPLOYMENT_TARGET'], row.get('deployment_target', '15.0'))
            self.assertEqual(env['CARGO_ENCODED_RUSTFLAGS'], 'MOCK_NATIVE_LINKER_FLAGS')
            return {'prebuilt': 'MOCK; not executed'}
        def guarded(argv, **kwargs):
            self.assertEqual(kwargs['env']['CARGO_ENCODED_RUSTFLAGS'], 'MOCK_NATIVE_LINKER_FLAGS')
            discovery = '--list' in argv
            self.assertEqual(kwargs['reap_metadata_descendants'], discovery and backend == 'candle-cuda')
            if not discovery:
                for name, payload in (generated_outputs or {}).items():
                    generated = Path(kwargs['env']['MODEL_PROOF_OUTPUT_DIR']) / name
                    generated.parent.mkdir(parents=True, exist_ok=True)
                    generated.write_bytes(payload)
            text = (row['fqn'] + ': test\n\n1 test, 0 benchmarks\n') if discovery else (
                'test result: ok. 1 passed; 0 failed; 0 ignored;\n'
                'MODEL_INFERENCE_BODY_COMPLETED ' + row['fqn'] + '\n'
                'MODEL_INFERENCE_DEVICE ' + row['fqn'] + (' candle-cuda\n' if backend == 'candle-cuda' else ' mlx-cpu\n') + input_body)
            if not discovery and getattr(fail, 'owned_completion', {}).get('scope') == 'supervisor-no-child-created':
                text = ''
            if discovery and row.get('cargo_target_metadata') is not None:
                text += json.dumps({'reason':'compiler-artifact','profile':{'test':True},'executable':'MOCK-never-executed.exe',
                    'target':row['cargo_target_metadata'],'package_id':'MOCK only','features':[]})+'\n'
            kwargs['stdout_path'].write_text(text, encoding="utf-8")
            kwargs['stderr_path'].write_text('', encoding="utf-8")
            if observer_events == 'broken-symlink':
                kwargs['event_path'].symlink_to(root / 'missing-observer-events')
            elif observer_events:
                kwargs['event_path'].write_text('{"mock":true}\n', encoding="utf-8")
            completion = {'scope':'supervisor-owned-windows-job','root_pid':42,
                'root_returncode':0,'waited_returncode':0,'job_member_pids':[],'owned_pids':[]}
            if isinstance(fail, safety.SupervisionError) and not discovery:
                if backend == 'candle-cuda' and not hasattr(fail, 'owned_completion'):
                    fail.owned_completion = {**completion, 'root_returncode':1,'waited_returncode':1}
                raise fail
            if backend == 'candle-cuda' and not discovery:
                self.assertTrue(kwargs['retain_owned_completion'])
                return source_safety.OwnedCudaRunResult(42, 0, 1024, None, 1000000000, None, .01, (), {'mode':'mock','originalClaimDeadlineMonotonic':kwargs['claim_deadline_monotonic']}, completion)
            return safety.RunResult(42, 1 if fail and not discovery else 0, 1024, None,
                                    1000000000, None, 0.01, (), {'mode': 'mock'})
        @contextlib.contextmanager
        def incomplete_controller(*_args, **_kwargs):
            if preseal == 'claim':
                raise resources.CudaClaimRejected({'requested_uuid': 'GPU-12345678-1234-1234-1234-123456789abc',
                                                   'predicates': {'unique_uuid': True, 'selected_process_types_allowed': False}})
            yield {'mechanism': 'MOCK controller', 'post_execution_verified': False}
            if preseal == 'cancel':
                raise KeyboardInterrupt('MOCK cancellation before physical seal')
            if preseal == 'release':
                raise resources.CudaReleaseRejected('MOCK release refusal',
                                                    {'status': 'retained-existing-samples', 'sample_count': 1})
            raise ValueError('MOCK controller post-idle failure')
        @contextlib.contextmanager
        def shared_controller(*_args, **_kwargs):
            physical = {'activity_profile': resources.SHARED_CUDA_HEADROOM_ACTIVITY,
                'device_uuid':row['required_cuda_device_uuid'],'physical_index':1,
                'post_execution_verified':False,'lease_released':False,
                'claim_deadline_monotonic':100000000000.0}
            try:
                yield physical
            finally:
                if not resources.shared_owned_completion(physical):
                    raise resources.CudaReleaseRejected('MOCK unproven owned completion', {})
                physical['lease_released'] = True
                physical['post_execution_verified'] = True
        claim_patch = (patch.object(resources, 'execution_claim', side_effect=shared_controller) if backend == 'candle-cuda'
                       else patch.object(resources, 'execution_claim', side_effect=incomplete_controller) if preseal
                       else contextlib.nullcontext())
        expected = '1' * 40
        env = {'RUNNER_TEMP': str(root), 'RUNNER_OS': 'Windows' if backend == 'candle-cuda' else 'macOS', 'RUNNER_NAME': 'mock-hosted',
               'GITHUB_EVENT_NAME': 'workflow_dispatch', 'MODEL_PROOF_RUNNER_ENVIRONMENT': 'github-hosted'}
        with patch.dict(os.environ, env), patch.object(proof, 'ROOT', root), \
                patch.object(proof, 'repository_identity', return_value={'revision': expected, 'tree': '2' * 40}), \
                patch.object(proof, 'verified_toolchain', side_effect=toolchain), \
                patch.object(proof, 'prepare_build', side_effect=preparation), \
                patch.object(safety, 'run_guarded', side_effect=guarded) as guard, claim_patch:
            output = root / 'evidence'
            try:
                status = proof.run((row,), backend, mock_request, expected, output)
            except KeyboardInterrupt:
                status = 130
            verified = proof.verify_artifacts(output, root, mock_request, expected, (row,))
            receipt = json.loads((output / 'receipt.json').read_text(encoding="utf-8"))
            return status, receipt, verified, guard.call_count, output

    def test_shared_owned_result_cross_module_identity_and_failed_closed_body(self):
        # The real collector loader imports canonical scripts.media_campaign_supervisor;
        # a distinct loaded class must survive dataclass/asdict and release value validation.
        alternate = load('collector_alternate_supervisor', ROOT / 'scripts/media_campaign_supervisor.py')
        self.assertIsNot(alternate.OwnedCudaRunResult, safety.OwnedCudaRunResult)
        fqn = 'train::projection::tests::cuda_bf16_packed_qkvg_preserves_outputs_and_every_adapter_gradient'
        source = next(row for row in proof.load_registry() if row['fqn'] == fqn)
        row = proof.configured_case(source, 'fixed_cuda')
        for failure in (False, safety.SupervisionError('child-exit', 'MOCK closed root returned1')):
            with self.subTest(failure=failure), tempfile.TemporaryDirectory() as temporary:
                status, receipt, verified, _, _ = self._mock_run(Path(temporary).resolve(),
                    row_override=row, backend='candle-cuda', supervisor_module=alternate, fail=failure)
                physical = receipt['cases'][0]['physical_execution']
                self.assertTrue(physical['post_execution_verified'])
                self.assertTrue(physical['lease_released'])
                self.assertEqual(physical['owned_execution_completion']['scope'], 'supervisor-owned-windows-job')
                self.assertEqual(status, 1 if failure else 0)
                self.assertEqual(receipt['accepted'], not bool(failure))
                if not failure:
                    self.assertTrue(verified['accepted_model_run'])
                else:
                    self.assertIn('child-exit', receipt['cases'][0]['error'])

    def test_paired_precreation_zero_is_joined_to_actual_source_observation_not_launch_attempt_flag(self):
        source = next(row for row in proof.load_registry() if row.get('fqn') == 'trainer_e2e::krea_bounded_acceleration_profile')
        row = proof.configured_case(source, 'fixed_cuda_paired_comparison')
        for none_created in (True, False):
            error = safety.SupervisionError('preflight-memory' if none_created else 'child-exit','MOCK refusal')
            if none_created:
                error.owned_completion = {'scope':'supervisor-no-child-created','creation_attempted':False}
            with self.subTest(none_created=none_created), tempfile.TemporaryDirectory() as temporary, \
                    patch.dict(os.environ, {'KREA_RAW_DIR':'MOCK path; no files opened'}):
                status,receipt,_,_,_=self._mock_run(Path(temporary).resolve(),row_override=row,
                    backend='candle-cuda',fail=error)
                case=receipt['cases'][0]
                self.assertEqual(status,1)
                self.assertFalse(case['accepted'])
                self.assertTrue(case['krea_paired_supervisor_invocation_attempted'])
                self.assertNotIn('krea_paired_body_launched',case)
                self.assertEqual(case['krea_paired_update_accounting']['conservative_new_charge'],0 if none_created else 12)

    def test_isolated_cli_import_preserves_owned_completion_value_protocol(self):
        # Same isolated Python import setup as the workflow's -I entry point; mocks only.
        script = ("import sys,unittest; sys.path.insert(0,sys.argv[1]); "
                  "suite=unittest.defaultTestLoader.loadTestsFromName("
                  "'scripts.tests.test_model_proof_collector.Collector.test_shared_owned_result_cross_module_identity_and_failed_closed_body'); "
                  "result=unittest.TextTestRunner(verbosity=0).run(suite); raise SystemExit(not result.wasSuccessful())")
        result = subprocess.run([sys.executable, '-I', '-c', script, str(ROOT)],
                                capture_output=True, text=True, encoding='utf-8', timeout=60)
        self.assertEqual(result.returncode, 0, result.stderr)

    def test_full_mock_collector_records_body_device_events_and_safe_upload(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary).resolve()
            status, receipt, verified, calls, output = self._mock_run(root)
            self.assertEqual(status, 0)
            self.assertEqual(calls, 2)
            self.assertEqual(receipt['completed'], 1)
            self.assertTrue(receipt['accepted'])
            self.assertEqual(receipt['cases'][0]['actual_devices'], ['mlx-cpu'])
            self.assertNotIn('cuda_claim_diagnostic', receipt['cases'][0])
            self.assertTrue(verified['accepted_model_run'])  # MOCK ONLY, temp tree removed.
            self.assertTrue((output / '00000/body.events.jsonl').is_file())
            self.assertFalse(any(path.suffix == '.safetensors' for path in output.rglob('*')))

    def test_generated_adapter_survives_existing_collector_and_reverification(self):
        with tempfile.TemporaryDirectory() as temporary:
            row = case()
            name = 'trained/adapter.safetensors'
            row['output_artifacts'] = [name]
            row['generated_adapter_artifacts'] = {name: adapter_spec()}
            status, receipt, verified, _, output = self._mock_run(
                Path(temporary).resolve(), row_override=row, generated_outputs={name: adapter_bytes()})
            self.assertEqual(status, 0)
            self.assertTrue(verified['accepted_model_run'])  # MOCK ONLY.
            self.assertEqual((output / '00000/generated' / name).read_bytes(), adapter_bytes())
            self.assertTrue(any(item['path'].endswith(name) for item in receipt['cases'][0]['outputs']))

    def test_adapter_reverification_rejects_removed_output_and_rehashed_model_payload(self):
        for remove in (False, True):
            with self.subTest(remove=remove), tempfile.TemporaryDirectory() as temporary:
                root = Path(temporary).resolve()
                row = case()
                name = 'trained/adapter.safetensors'
                row['output_artifacts'] = [name]
                row['generated_adapter_artifacts'] = {name: adapter_spec()}
                _, receipt, _, _, output = self._mock_run(
                    root, row_override=row, generated_outputs={name: adapter_bytes()})
                result = receipt['cases'][0]
                artifact = output / '00000/generated' / name
                record = next(item for item in result['outputs'] if item['path'].endswith(name))
                if remove:
                    artifact.unlink()
                    result['outputs'].remove(record)
                else:
                    raw = adapter_bytes(lambda h: h.update({'model.weight': h.pop('blocks.0.to_q.lora_A.weight')}))
                    artifact.write_bytes(raw)
                    record.update(bytes=len(raw), sha256=proof.hashlib.sha256(raw).hexdigest())
                (output / 'receipt.json').write_text(json.dumps(receipt), encoding='utf-8')
                (output / '00000/result.json').write_text(json.dumps(result), encoding='utf-8')
                with self.assertRaisesRegex(ValueError, 'required generated output|unapproved/model tensor'):
                    proof.verify_artifacts(output, root, request(), '1' * 40, (row,))

    def test_success_without_optional_observer_events_keeps_actual_supervision(self):
        # CI_IMPACT: pure mocked controller regression for the Windows no-event
        # discovery/body path; no Cargo, model, weight, GPU, or process execution.
        with tempfile.TemporaryDirectory() as temporary:
            status, receipt, verified, calls, output = self._mock_run(
                Path(temporary).resolve(), observer_events=False)
            self.assertEqual(status, 0)
            self.assertEqual(calls, 2)
            self.assertTrue(verified['accepted_model_run'])  # MOCK ONLY.
            case_result = receipt['cases'][0]
            self.assertTrue(case_result['accepted'])
            self.assertEqual(len(case_result['commands']), 2)
            self.assertEqual(case_result['actual_devices'], ['mlx-cpu'])
            self.assertFalse(any(item['path'].endswith('.events.jsonl') for item in case_result['outputs']))
            self.assertFalse(list(output.rglob('*.events.jsonl')))

    def test_broken_observer_event_link_cannot_pass_as_absent(self):
        with tempfile.TemporaryDirectory() as temporary:
            status, receipt, verified, calls, _ = self._mock_run(
                Path(temporary).resolve(), observer_events='broken-symlink')
            self.assertEqual(status, 1)
            self.assertEqual(calls, 1)
            self.assertFalse(receipt['cases'][0]['accepted'])
            self.assertFalse(verified['accepted_model_run'])

    def test_registered_mlx_deployment_precedes_toolchain_sdk_preflight(self):
        with tempfile.TemporaryDirectory() as temporary:
            row = case()
            row['deployment_target'] = '26.2'
            status, _receipt, _verified, calls, _output = self._mock_run(
                Path(temporary).resolve(), row_override=row)
            self.assertEqual(status, 0)
            self.assertEqual(calls, 2)

    def test_device_memory_failure_sample_survives_collector(self):
        error = safety.SupervisionError('device-memory', 'MOCK device predicate')
        error.pid = 42
        error.watchdog_device_memory = {
            'cudaDeviceUuid': 'GPU-12345678-1234-1234-1234-123456789abc',
            'gpuUsageBasis': 'global-used-upper-bound', 'globalUsedUpperBoundBytes': 1001,
            'ownedGpuBytes': None, 'gpuFreeBytes': 4000, 'childGpuCapBytes': 1000,
            'gpuFreeReserveBytes': 100, 'capExceeded': True, 'reserveBreached': False,
            'sampledGlobalUpperBoundPeakBytes': 1001,
        }
        with tempfile.TemporaryDirectory() as temporary:
            status, receipt, verified, calls, output = self._mock_run(Path(temporary).resolve(), fail=error)
            self.assertEqual(status, 1)
            self.assertEqual(calls, 2)
            self.assertEqual(receipt['completed'], 0)
            self.assertFalse(receipt['accepted'])
            self.assertFalse(verified['accepted_model_run'])
            result = json.loads((output / '00000/result.json').read_text(encoding='utf-8'))
            self.assertEqual(result['supervision_failure']['watchdog_device_memory'], error.watchdog_device_memory)
            self.assertEqual(receipt['cases'][0]['supervision_failure'], result['supervision_failure'])

    def test_failed_child_keeps_failure_receipt_and_known_logs_not_acceptance(self):
        with tempfile.TemporaryDirectory() as temporary:
            status, receipt, verified, calls, output = self._mock_run(Path(temporary).resolve(), fail=True)
            self.assertEqual(status, 1)
            self.assertFalse(receipt['accepted'])
            self.assertEqual(receipt['completed'], 0)
            self.assertFalse(verified['accepted_model_run'])
            paths = [item['path'] for item in receipt['cases'][0]['outputs']]
            self.assertEqual(len(paths), len(set(paths)))
            self.assertTrue((output / '00000/body.stderr.log').is_file())

    def test_missing_prerequisite_never_launches_or_reports_a_completed_body(self):
        with tempfile.TemporaryDirectory() as temporary:
            status, receipt, verified, calls, _ = self._mock_run(Path(temporary).resolve(), missing=True)
            self.assertEqual(status, 1)
            self.assertEqual(calls, 0)
            self.assertEqual(receipt['completed'], 0)
            self.assertFalse(verified['accepted_model_run'])

    def test_controller_failure_after_body_never_publishes_preliminary_acceptance(self):
        for failure in ('controller', 'cancel'):
            with self.subTest(failure=failure), tempfile.TemporaryDirectory() as temporary:
                status, receipt, verified, calls, _ = self._mock_run(Path(temporary).resolve(), preseal=failure)
                self.assertNotEqual(status, 0)
                self.assertEqual(calls, 2)
                self.assertFalse(receipt['accepted'])
                self.assertEqual(receipt['completed'], 0)
                case = receipt['cases'][0]
                self.assertTrue(case['body_proof_accepted'])
                self.assertTrue(case['resource_verification_pending'])
                self.assertFalse(case['accepted'])
                self.assertFalse(verified['accepted_model_run'])
                self.assertEqual(len(case['commands']), 2)

    def test_prebody_claim_diagnostic_survives_existing_failed_artifact_without_acceptance(self):
        # CI_IMPACT: collector mock only; no model, CUDA, Cargo, or process execution.
        with tempfile.TemporaryDirectory() as temporary:
            status, receipt, verified, calls, output = self._mock_run(Path(temporary).resolve(), preseal='claim')
            self.assertEqual(status, 1)
            self.assertEqual(calls, 1)  # discovery only; body never starts
            self.assertFalse(receipt['accepted'])
            self.assertEqual(receipt['completed'], 0)
            case = receipt['cases'][0]
            self.assertFalse(case['accepted'])
            self.assertFalse(case['body_proof_accepted'])
            self.assertTrue(case['resource_verification_pending'])
            self.assertEqual(case['cuda_claim_diagnostic']['predicates']['selected_process_types_allowed'], False)
            self.assertEqual(json.loads((output / '00000/result.json').read_text(encoding='utf-8')), case)
            self.assertFalse(verified['accepted_model_run'])
            self.assertFalse((output / '00000/body.stdout.log').exists())

    def test_postbody_release_diagnostic_retains_failure_without_resource_seal(self):
        # CI_IMPACT: mocked resource controller only; no model, CUDA, Cargo or process execution.
        with tempfile.TemporaryDirectory() as temporary:
            status, receipt, verified, calls, output = self._mock_run(Path(temporary).resolve(), preseal='release')
            self.assertEqual(status, 1)
            self.assertEqual(calls, 2)
            self.assertFalse(receipt['accepted'])
            self.assertEqual(receipt['completed'], 0)
            case = receipt['cases'][0]
            self.assertTrue(case['body_proof_accepted'])
            self.assertTrue(case['resource_verification_pending'])
            self.assertFalse(case['accepted'])
            self.assertFalse(case['physical_execution']['post_execution_verified'])
            self.assertEqual(case['cuda_release_diagnostic']['status'], 'retained-existing-samples')
            self.assertEqual(json.loads((output / '00000/result.json').read_text(encoding='utf-8')), case)
            self.assertFalse(verified['accepted_model_run'])

    def test_final_upload_rejects_unknown_files_changed_results_and_wrong_source(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary).resolve()
            _, _, _, _, output = self._mock_run(root)
            unknown = output / 'checkpoint.safetensors'; unknown.write_bytes(b'forbidden')
            with self.assertRaises(ValueError):
                proof.verify_artifacts(output, root, request(), '1' * 40, (case(),))
            unknown.unlink()  # own temporary test scratch only
            result = output / '00000/result.json'; result.write_text('{}', encoding="utf-8")
            with self.assertRaises(ValueError):
                proof.verify_artifacts(output, root, request(), '1' * 40, (case(),))
            with self.assertRaises(ValueError):
                proof.verify_artifacts(output, root, request(), '3' * 40, (case(),))

    def test_weak_actual_reader_grade_survives_physical_seal_artifact_validation_and_summary(self):
        with tempfile.TemporaryDirectory() as temporary:
            row, text = external_case()
            status, receipt, verified, calls, output = self._mock_run(Path(temporary).resolve(), row_override=row, input_body=text)
            self.assertEqual(status, 0)
            self.assertEqual(calls, 2)
            for record in (receipt, receipt['cases'][0], verified):
                self.assertEqual(record['acceptance_scope'], 'source-bound-case-execution')
                self.assertEqual(record['input_grade'], 'successful-reader-binding')
                self.assertFalse(record['checkpoint_byte_integrity_attested'])
                self.assertFalse(record['verified_named_revision'])
                self.assertFalse(record['terminal_model_evidence'])
            self.assertTrue(receipt['cases'][0]['physical_execution']['post_execution_verified'])
            summary = (output / 'summary.md').read_text(encoding="utf-8")
            self.assertIn('successful-reader-binding', summary)
            self.assertIn('Checkpoint-byte integrity attested: False', summary)

    def test_resource_seal_cannot_upgrade_weak_case_json_or_aggregate_input_grade(self):
        for mutate_case in (False, True):
            with self.subTest(case=mutate_case), tempfile.TemporaryDirectory() as temporary:
                root = Path(temporary).resolve()
                row, text = external_case()
                _, receipt, _, _, output = self._mock_run(root, row_override=row, input_body=text)
                target = receipt['cases'][0] if mutate_case else receipt
                target.update(acceptance_scope='content-bound-case-execution', input_grade='validated-content',
                              checkpoint_byte_integrity_attested=True, terminal_model_evidence=True)
                (output / 'receipt.json').write_text(json.dumps(receipt), encoding="utf-8")
                if mutate_case:
                    (output / '00000/result.json').write_text(json.dumps(target), encoding="utf-8")
                with self.assertRaisesRegex(ValueError, 'input.*(grade|scope)|aggregate physical'):
                    proof.verify_artifacts(output, root, request(), '1' * 40, (row,))

    def test_missing_input_contract_refuses_before_any_command_and_weak_cannot_satisfy_strong_case(self):
        with tempfile.TemporaryDirectory() as temporary:
            row = case(); row.pop('input_contract')
            status, receipt, _, calls, _ = self._mock_run(Path(temporary).resolve(), row_override=row)
            self.assertNotEqual(status, 0)
            self.assertEqual(calls, 0)
            self.assertFalse(receipt['accepted'])
        with tempfile.TemporaryDirectory() as temporary:
            row, text = external_case('content')
            status, receipt, verified, calls, _ = self._mock_run(Path(temporary).resolve(), row_override=row, input_body=text)
            self.assertNotEqual(status, 0)
            self.assertEqual(calls, 2)
            self.assertFalse(receipt['accepted'])
            self.assertFalse(verified['checkpoint_byte_integrity_attested'])
            self.assertFalse(receipt['cases'][0]['body_proof_accepted'])

    def test_indexed_children_require_every_exact_body_and_actual_child_device(self):
        row = case()
        row.update(execution_role='indexed_children', expected_child_ids=['first', 'second'])
        text = 'test result: ok. 1 passed; 0 failed; 0 ignored;\n' * 3
        for child in row['expected_child_ids']:
            text += f"MODEL_INFERENCE_BODY_COMPLETED {row['fqn']}#child={child}\n"
            text += f"MODEL_INFERENCE_DEVICE {row['fqn']}#child={child} mlx-cpu\n"
        text += 'MODEL_INFERENCE_PARENT_COMPLETED ' + row['fqn'] + '\n'
        result = proof.verify_execution(row, text)
        self.assertEqual(result['body_completed'], 2)
        for invalid in (text.replace('#child=second', '#child=first'),
                        text.replace('MODEL_INFERENCE_DEVICE ' + row['fqn'] + '#child=second mlx-cpu', ''),
                        text + 'MODEL_INFERENCE_BODY_COMPLETED ' + row['fqn'] + '\n',
                        text.replace('MODEL_INFERENCE_PARENT_COMPLETED', 'MISSING_PARENT')):
            with self.assertRaises(ValueError):
                proof.verify_execution(row, invalid)

    def test_explicit_missing_required_work_rejects_even_successful_body_and_device(self):
        row = case()
        text = ('test result: ok. 1 passed; 0 failed; 0 ignored;\n'
                'MODEL_INFERENCE_BODY_COMPLETED ' + row['fqn'] + '\n'
                'MODEL_INFERENCE_DEVICE ' + row['fqn'] + ' mlx-cpu\n')
        with self.assertRaises(ValueError):
            proof.verify_execution(row, text + 'MODEL_INFERENCE_PREREQUISITE_MISSING ' + row['fqn'] + ' missing fixture tier\n')

    def test_manual_discovery_records_actual_compiler_target_not_a_package_name_guess(self):
        row = case()
        row['cargo_target_metadata'] = {'name': row['target'], 'kind': ['staticlib', 'rlib'],
                                        'crate_types': ['staticlib', 'rlib']}
        artifact = {'reason': 'compiler-artifact', 'profile': {'test': True}, 'executable': 'MOCK-only',
                    'package_id': 'mock-id', 'target': dict(row['cargo_target_metadata']), 'features': []}
        listing = row['fqn'] + ': test\n\n1 test, 0 benchmarks\n'
        report = proof.verify_discovery(row, json.dumps(artifact) + '\n' + listing)
        self.assertEqual(report['target']['kind'], ['staticlib', 'rlib'])
        self.assertEqual(report['model_bodies_executed'], 0)
        with self.assertRaises(ValueError):
            proof.verify_discovery(row, listing)
        artifact['target']['name'] = row['package']
        with self.assertRaises(ValueError):
            proof.verify_discovery(row, json.dumps(artifact) + '\n' + listing)

    def test_cpu_metadata_policy_strips_cuda_only_fields(self):
        value = request('windows-cuda')['policy']
        value.update(cudaDeviceUuid='GPU-00000000-0000-0000-0000-000000000000',
                     gpuFreeReserveBytes=1, childGpuCapBytes=1)
        metadata = resources.metadata_policy_payload(value)
        self.assertEqual(metadata['backend'], 'windows-cpu')
        self.assertNotIn('cudaDeviceUuid', metadata)
        self.assertNotIn('childGpuCapBytes', metadata)
        self.assertEqual(value['backend'], 'windows-cuda')


if __name__ == '__main__':
    unittest.main()
