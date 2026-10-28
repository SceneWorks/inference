"""Fixed SD35 recipe protocol scaffolding; synthetic fixtures only, all execution mocked."""
import copy
import json
import os
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from scripts import media_campaign_supervisor as safety
from scripts.ci import model_inference_proofs as proof


def configured_row():
    # These are protocol fixtures, NOT declarations of the real SD35 branch/default facts.
    controls = sorted(proof.SD35_CONFIGURATION_CONTROLS)
    def recipe(cuda, skip_dit, devices, scope, work):
        return {'source_identity': 'synthetic-fixture-only::' + scope,
                'environment': {name: ('1' if name == 'SD35_PARITY_CUDA' and cuda
                                      or name == 'SD35_PARITY_SKIP_DIT' and skip_dit else None) for name in controls},
                'backend': 'candle-cuda' if 'candle-cuda' in devices else 'candle-cpu',
                'expected_devices': devices, 'scope': scope, 'required_work': work,
                'cargo_features': ['model-inference-proof', *(['cuda'] if cuda else [])]}
    return {'package': 'candle-gen-sd3', 'target': 'integration', 'target_kind': 'test',
            'fqn': 'component_parity::sd35_large_component_parity_vs_diffusers', 'source': proof.SD35_CONFIGURATION_SOURCE,
            'execution_role': 'direct', 'backend': 'candle-cpu', 'expected_devices': ['candle-cpu'],
            'cargo_features': ['model-inference-proof'], 'fixture_identity': 'NO NN; mock protocol fixture',
            'configuration_source': 'synthetic-fixture-only', 'output_contract': 'mock logs',
            'input_contract': {'fixture_kind': 'source-defined', 'source_identity': 'synthetic-fixture-only', 'requirements': []},
            'configuration_controls': controls, 'default_configuration': 'source-default',
            'configurations': {'source-default': recipe(False, False, ['candle-cpu'], 'full-cpu', ['encoders', 'dit']),
                               'encoder-only': recipe(False, True, ['candle-cpu'], 'encoder-only', ['encoders']),
                               'full-cuda': recipe(True, False, ['candle-cpu', 'candle-cuda'], 'full', ['encoders', 'dit'])}}


def request():
    return {'selection': 'candle-gen-sd3', 'policy': {'schemaVersion': 1, 'backend': 'linux-cpu',
            'deadlineSeconds': 10, 'pollMillis': 10, 'termGraceMillis': 50,
            'hostFreeReserveBytes': 100, 'childFootprintCapBytes': 1000000,
            'stdoutCapBytes': 1000000, 'stderrCapBytes': 1000000, 'eventCapBytes': 1000000}}


class FixedConfigurations(unittest.TestCase):
    def test_legacy_identity_row_and_environment_are_unchanged(self):
        row = {'package': 'candle-llm', 'target': 'candle_llm', 'fqn': 'tests::fixture', 'backend': 'candle-cpu'}
        env = {'SD35_PARITY_CUDA': '1', 'OTHER_CONTROL': 'original'}
        original = dict(env)
        self.assertIs(proof.configured_case(row), row)
        proof.apply_configuration(row, env)
        self.assertEqual(env, original)
        self.assertEqual(proof.select((row,), 'all', 'candle-cpu'), (row,))
        with self.assertRaises(ValueError):
            proof.select((row,), 'all', 'candle-cpu', 'undeclared')

    def test_default_explicit_unsets_override_inherited_flags_without_optional_work(self):
        row = configured_row()
        effective = proof.select((row,), 'all', 'candle-cpu')[0]
        env = {name: '1' for name in row['configuration_controls']}
        env['MODEL_PATH'] = '/operator/asset-path'
        proof.apply_configuration(effective, env)
        self.assertNotIn('SD35_PARITY_CUDA', env)
        self.assertNotIn('SD35_PARITY_SKIP_ENCODERS', env)
        self.assertNotIn('SD35_PARITY_SKIP_DIT', env)
        self.assertEqual(env['MODEL_PATH'], '/operator/asset-path')
        self.assertEqual(effective['selected_configuration']['required_work'], ['encoders', 'dit'])
        self.assertEqual(effective['selected_configuration']['scope'], 'full-cpu')
        recorded = proof.configuration_bindings(effective, env)
        self.assertEqual(recorded['SD35_PARITY_CUDA'], {'present': False, 'value': None})
        self.assertEqual(proof.identity(effective), proof.identity(row))

    def test_named_configuration_binds_exact_lane_features_and_mixed_device_set(self):
        row = configured_row()
        effective = proof.select((row,), proof.identity(row), 'candle-cuda', 'full-cuda')[0]
        self.assertEqual(effective['expected_devices'], ['candle-cpu', 'candle-cuda'])
        self.assertEqual(effective['cargo_features'], ['model-inference-proof', 'cuda'])
        self.assertEqual(proof.identity(effective), proof.identity(row))
        for name, lane in (('unknown', 'candle-cpu'), ('full-cuda', 'candle-cpu'), ('encoder-only', 'candle-cuda')):
            with self.subTest(configuration=name, backend=lane), self.assertRaises(ValueError):
                proof.select((row,), 'all', lane, name)

    def test_unresolved_or_tampered_runtime_recipe_cannot_set_environment_authority(self):
        base = configured_row()
        env = {'MODEL_PATH': 'original'}
        with self.assertRaises(ValueError):
            proof.apply_configuration(base, env)
        effective = proof.select((base,), 'all', 'candle-cpu')[0]
        effective['selected_configuration'] = copy.deepcopy(effective['selected_configuration'])
        effective['selected_configuration']['environment']['MODEL_INFERENCE_CASE_FQN'] = 'spoof'
        with self.assertRaises(ValueError):
            proof.apply_configuration(effective, env)
        self.assertEqual(env, {'MODEL_PATH': 'original'})

    def test_request_accepts_only_fixed_name_not_environment_or_argv(self):
        value = request()
        value['configuration'] = 'encoder-only'
        self.assertEqual(proof.request(json.dumps(value), 'candle-cpu'), value)
        for invalid in ({'SD35_PARITY_CUDA': '1'}, ['encoder-only'], 'encoder-only --ignored', None):
            value['configuration'] = invalid
            with self.assertRaises(ValueError):
                proof.request(json.dumps(value), 'candle-cpu')
        value = request(); value['environment'] = {'SD35_PARITY_CUDA': '1'}
        with self.assertRaises(ValueError):
            proof.request(json.dumps(value), 'candle-cpu')

    def test_reserved_asset_child_authority_or_unknown_controls_are_rejected(self):
        for name in ('CUDA_VISIBLE_DEVICES', 'MODEL_INFERENCE_CASE_FQN', 'HF_TOKEN',
                     'SD35_PARITY_REF', 'IOS_CAPTURE_IN_CHILD', 'ALLOW_GPU', 'UNDECLARED'):
            row = configured_row()
            row['configuration_controls'].append(name)
            for recipe in row['configurations'].values():
                recipe['environment'][name] = '1'
            with self.subTest(control=name), self.assertRaises(ValueError):
                proof.validate_configurations(row)

    def test_missing_default_flags_work_devices_features_or_invalid_skip_combination_rejected(self):
        mutations = [lambda row: row.pop('default_configuration'),
                     lambda row: row['configurations']['encoder-only']['environment'].pop('SD35_PARITY_CUDA'),
                     lambda row: row['configurations']['encoder-only'].update(required_work=[]),
                     lambda row: row['configurations']['full-cuda'].update(backend='candle-cpu'),
                     lambda row: row['configurations']['full-cuda'].update(cargo_features=['model-inference-proof']),
                     lambda row: row['configurations']['encoder-only']['environment'].update(SD35_PARITY_SKIP_ENCODERS='1'),
                     lambda row: row['configurations']['encoder-only']['environment'].update(SD35_PARITY_CUDA='0'),
                     lambda row: row['configurations']['full-cuda']['environment'].update(SD35_PARITY_SKIP_DIT='1'),
                     lambda row: row['configurations']['encoder-only']['cargo_features'].append('cuda'),
                     lambda row: row['configurations']['source-default']['environment'].update(SD35_PARITY_SKIP_DIT='1'),
                     lambda row: row.update(package='unrelated-provider'),
                     lambda row: row.update(backend='candle-cuda')]
        for mutate in mutations:
            row = configured_row(); mutate(row)
            with self.subTest(mutation=mutate), self.assertRaises(ValueError):
                proof.validate_configurations(row)

    def test_cpu_source_default_may_be_full_work_without_forcing_cuda_build(self):
        row = configured_row()
        full = copy.deepcopy(row['configurations']['encoder-only'])
        full['environment'] = {name: None for name in row['configuration_controls']}
        full.update(scope='full-component-parity F32/CPU', required_work=['F-002/encoders', 'F-001/dit_velocity'])
        row['configurations']['source-default'] = full
        row['default_configuration'] = 'source-default'
        effective = proof.select((row,), 'all', 'candle-cpu')[0]
        self.assertEqual(effective['selected_configuration']['required_work'], ['F-002/encoders', 'F-001/dit_velocity'])
        self.assertNotIn('cuda', effective['cargo_features'])
        self.assertTrue(all(value is None for value in effective['selected_configuration']['environment'].values()))

    def test_source_ab_case_controls_only_cuda_and_does_not_reinterpret_skip_flags(self):
        row = configured_row()
        row['configuration_controls'] = ['SD35_PARITY_CUDA']
        row['fqn'] = 'component_parity::sd35_bigg_pad_token_ab'
        row['configurations']['source-default'].update(scope='encoder-conditioning-A/B only', required_work=['encode_fixed', 'encode_prefix'])
        for recipe in row['configurations'].values():
            recipe['environment'] = {'SD35_PARITY_CUDA': recipe['environment']['SD35_PARITY_CUDA']}
            recipe['cargo_features'].append('sc9581-ab')
        row['cargo_features'].append('sc9581-ab')
        effective = proof.select((row,), 'all', 'candle-cpu')[0]
        env = {'SD35_PARITY_CUDA': '1', 'SD35_PARITY_SKIP_ENCODERS': '1', 'SD35_PARITY_SKIP_DIT': '1'}
        proof.apply_configuration(effective, env)
        self.assertNotIn('SD35_PARITY_CUDA', env)
        self.assertEqual(env['SD35_PARITY_SKIP_ENCODERS'], '1')
        self.assertEqual(env['SD35_PARITY_SKIP_DIT'], '1')
        self.assertNotIn('cuda', effective['cargo_features'])
        self.assertIn('sc9581-ab', effective['cargo_features'])

    def test_features_are_validated_against_actual_source_manifest(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            manifest = root / 'crates/media/candle-gen/candle-gen-sd3/Cargo.toml'
            manifest.parent.mkdir(parents=True)
            manifest.write_text('[package]\nname = "candle-gen-sd3"\n[features]\nmodel-inference-proof = []\ncuda = []\n', encoding="utf-8")
            row = configured_row()
            proof.validate_configurations(row, root=root)
            row['configurations']['full-cuda']['cargo_features'].append('absent-feature')
            with self.assertRaisesRegex(ValueError, 'absent from actual source manifest'):
                proof.validate_configurations(row, root=root)

    def test_registry_rejects_duplicate_canonical_id_for_configured_case(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            row = configured_row()
            row['cargo_target_metadata'] = {'name': 'integration', 'kind': ['test'], 'crate_types': ['bin']}
            source = root / row['source']; source.parent.mkdir(parents=True); source.write_text('// mock source', encoding="utf-8")
            manifest = root / 'crates/media/candle-gen/candle-gen-sd3/Cargo.toml'
            manifest.write_text('[package]\nname = "candle-gen-sd3"\n[features]\nmodel-inference-proof = []\ncuda = []\n', encoding="utf-8")
            duplicate = {key: value for key, value in row.items() if key not in ('configurations', 'default_configuration', 'configuration_controls')}
            duplicate.update(backend='candle-cuda', expected_devices=['candle-cuda'], cargo_features=['model-inference-proof', 'cuda'])
            registry = root / 'registry.json'; registry.write_text(json.dumps([row, duplicate]), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, 'exactly one canonical binding'):
                proof.load_registry(registry, root=root)

    def _collect(self, root, missing=False, configuration=None):
        base = configured_row()
        value = request()
        if configuration is not None:
            value['configuration'] = configuration
        row = proof.select((base,), 'all', 'candle-cpu', configuration)[0]
        source = root / row['source']; source.parent.mkdir(parents=True); source.write_text('// mock only', encoding="utf-8")
        calls = []
        def guarded(argv, **kwargs):
            calls.append(kwargs['env'].copy())
            discovery = '--list' in argv
            text = (row['fqn'] + ': test\n\n1 test, 0 benchmarks\n') if discovery else (
                'test result: ok. 1 passed; 0 failed; 0 ignored;\n'
                'MODEL_INFERENCE_BODY_COMPLETED ' + row['fqn'] + '\n'
                'MODEL_INFERENCE_DEVICE ' + row['fqn'] + ' candle-cpu\n')
            if missing and not discovery:
                text += 'MODEL_INFERENCE_PREREQUISITE_MISSING ' + row['fqn'] + ' required encoder fixture absent\n'
            kwargs['stdout_path'].write_text(text, encoding="utf-8"); kwargs['stderr_path'].write_text('', encoding="utf-8")
            kwargs['event_path'].write_text('{"mock":true}\n', encoding="utf-8")
            return safety.RunResult(42, 0, 1024, None, 1000000000, None, 0.01, (), {'mode': 'mock'})
        env = {'RUNNER_TEMP': str(root), 'RUNNER_OS': 'Linux', 'RUNNER_NAME': 'mock-hosted',
               'GITHUB_EVENT_NAME': 'workflow_dispatch', 'MODEL_PROOF_RUNNER_ENVIRONMENT': 'github-hosted',
               'SD35_PARITY_CUDA': '1', 'SD35_PARITY_SKIP_ENCODERS': '1'}
        expected = '1' * 40
        output = root / 'evidence'
        with patch.dict(os.environ, env), patch.object(proof, 'ROOT', root), \
                patch.object(proof, 'repository_identity', return_value={'revision': expected, 'tree': '2' * 40}), \
                patch.object(proof, 'verified_toolchain', return_value={'MOCK': 'no compiler invoked'}), \
                patch.object(proof, 'prepare_build', return_value={'MOCK': 'no build invoked'}), \
                patch.object(safety, 'run_guarded', side_effect=guarded):
            status = proof.run((row,), 'candle-cpu', value, expected, output)
        receipt = json.loads((output / 'receipt.json').read_text(encoding="utf-8"))
        verified = proof.verify_artifacts(output, root, value, expected, (row,))
        return status, receipt, verified, calls, output, row

    def test_mock_collector_records_effective_unsets_scope_and_rejects_tampered_config(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary).resolve()
            status, receipt, verified, calls, output, row = self._collect(root, configuration='encoder-only')
            self.assertEqual(status, 0)
            self.assertTrue(receipt['accepted'])
            self.assertEqual(receipt['cases'][0]['declared']['selected_configuration']['scope'], 'encoder-only')
            self.assertTrue(all('SD35_PARITY_CUDA' not in env and 'SD35_PARITY_SKIP_ENCODERS' not in env for env in calls))
            case = receipt['cases'][0]
            case['configuration_environment']['SD35_PARITY_CUDA'] = {'present': True, 'value': '1'}
            (output / 'receipt.json').write_text(json.dumps(receipt), encoding="utf-8")
            (output / '00000/result.json').write_text(json.dumps(case), encoding="utf-8")
            value = request(); value['configuration'] = 'encoder-only'
            with self.assertRaisesRegex(ValueError, 'effective source control differs'):
                proof.verify_artifacts(output, root, value, '1' * 40, (row,))

    def test_named_configuration_is_bound_in_receipt_and_scope_is_visible_in_summary(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary).resolve()
            status, receipt, _, _, output, row = self._collect(root, configuration='encoder-only')
            self.assertEqual(status, 0)
            self.assertEqual(receipt['request_configuration'], 'encoder-only')
            summary = (output / 'summary.md').read_text(encoding="utf-8")
            self.assertIn('encoder-only', summary)
            self.assertIn('Required source work: encoders', summary)
            receipt['request_configuration'] = 'full-cuda'
            (output / 'receipt.json').write_text(json.dumps(receipt), encoding="utf-8")
            value = request(); value['configuration'] = 'encoder-only'
            with self.assertRaisesRegex(ValueError, 'approved fixed configuration selection'):
                proof.verify_artifacts(output, root, value, '1' * 40, (row,))

    def test_mock_missing_required_work_rejects_even_successful_body_and_device(self):
        with tempfile.TemporaryDirectory() as temporary:
            status, receipt, _, _, _, _ = self._collect(Path(temporary).resolve(), missing=True)
            self.assertNotEqual(status, 0)
            self.assertFalse(receipt['accepted'])
            self.assertFalse(receipt['cases'][0]['accepted'])


if __name__ == '__main__':
    unittest.main()
