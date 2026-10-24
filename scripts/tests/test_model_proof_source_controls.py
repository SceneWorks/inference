"""Fixed source-declared NN controls; no operator environment authority or NN execution."""
import json
from pathlib import Path
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from scripts.ci import model_inference_proofs as proof


def row():
    source = 'crates/llm/candle-llm/src/decode/graph.rs'
    def recipe(environment, features):
        return {'source_identity': source + '::cuda_tests/source-fixed-branch',
                'environment': environment, 'backend': 'candle-cuda', 'expected_devices': ['candle-cuda'],
                'cargo_features': features, 'scope': 'source-preserved whole NN case and assertions',
                'required_work': ['learned generation and original graph assertion']}
    return {'package': 'candle-llm', 'target': 'candle_llm', 'target_kind': 'lib',
            'fqn': 'decode::graph::cuda_tests::a_flash_attn_build_is_refused_by_name', 'source': source,
            'execution_role': 'direct', 'backend': None, 'expected_devices': [], 'cargo_features': [],
            'configuration_kind': 'source-controls', 'configuration_controls': ['CANDLE_LLM_CUDA_STREAM', 'CANDLE_LLM_DEVICE'],
            'default_configuration': None,
            'configurations': {'own-stream': recipe({'CANDLE_LLM_CUDA_STREAM': 'own', 'CANDLE_LLM_DEVICE': None}, ['model-inference-proof', 'cuda']),
                               'flash-refusal': recipe({'CANDLE_LLM_CUDA_STREAM': None, 'CANDLE_LLM_DEVICE': None}, ['model-inference-proof', 'cuda', 'flash-attn'])}}


class SourceControls(unittest.TestCase):
    def test_exact_source_recipes_select_own_or_unset_without_forcing_optional_flash(self):
        case = row()
        for name in case['configurations']:
            actual = proof.select((case,), proof.identity(case), 'candle-cuda', name)[0]
            environment = {'CANDLE_LLM_CUDA_STREAM': 'inherited', 'CANDLE_LLM_DEVICE': 'cpu', 'ASSET_PATH': 'operatorasset'}
            proof.apply_configuration(actual, environment)
            self.assertNotIn('CANDLE_LLM_DEVICE', environment)
            self.assertEqual(environment.get('CANDLE_LLM_CUDA_STREAM'), 'own' if name == 'own-stream' else None)
            self.assertEqual('flash-attn' in actual['cargo_features'], name == 'flash-refusal')
            self.assertEqual(environment['ASSET_PATH'], 'operatorasset')
            self.assertEqual(proof.identity(actual), proof.identity(case))
            recorded = proof.configuration_bindings(actual, environment)
            self.assertEqual(recorded['CANDLE_LLM_DEVICE'], {'present': False, 'value': None})
        with self.assertRaisesRegex(ValueError, 'explicit configuration required'):
            proof.select((case,), 'all', 'candle-cuda')

    def test_presence_controls_preserve_empty_and_zero_strings_not_boolean_coercion(self):
        case = row()
        case.update(configuration_controls=['SA3_TEST_METAL', 'SA3_TEST_CUDA'])
        case['configurations'] = {'present-metal': {'source_identity': case['source'] + '::presence-helper',
            'environment': {'SA3_TEST_METAL': '', 'SA3_TEST_CUDA': '0'}, 'backend': 'candle-metal',
            'expected_devices': ['candle-metal'], 'cargo_features': ['model-inference-proof', 'metal'],
            'scope': 'source Metal-first presence semantics', 'required_work': ['actual learned oracle output']}}
        actual = proof.select((case,), 'all', 'candle-metal', 'present-metal')[0]
        environment = {}
        proof.apply_configuration(actual, environment)
        self.assertEqual(environment, {'SA3_TEST_METAL': '', 'SA3_TEST_CUDA': '0'})
        self.assertEqual(proof.configuration_bindings(actual, environment)['SA3_TEST_METAL'], {'present': True, 'value': ''})

    def test_data_driven_rung_variants_do_not_create_a_runtime_grid_or_force_optional_work(self):
        case = row()
        case.update(package='candle-gen-flux2', target='integration', target_kind='test',
                    fqn='source_fixture::one_rung', source='crates/media/candle-gen/candle-gen-flux2/tests/source_fixture.rs',
                    configuration_controls=['FLUX2_MEMORY_RUNG'], default_configuration='resident', backend='candle-cuda',
                    expected_devices=['candle-cuda'], cargo_features=['model-inference-proof', 'cuda'])
        def rung(value, devices, scope):
            return {'source_identity': case['source'] + '::one_rung/source-branch', 'environment': {'FLUX2_MEMORY_RUNG': value},
                    'backend': 'candle-cuda', 'expected_devices': devices, 'cargo_features': ['model-inference-proof', 'cuda'],
                    'scope': scope, 'required_work': ['exact selected source scope only']}
        case['configurations'] = {'resident': rung(None, ['candle-cuda'], 'resident GPU only'),
                                  'cpu-leg-fixture': rung('source-fixture-value', ['candle-cpu', 'candle-cuda'], 'source-permitted CPU leg plus GPU')}
        default = proof.select((case,), 'all', 'candle-cuda')[0]
        self.assertEqual(default['selected_configuration']['name'], 'resident')
        self.assertEqual(default['expected_devices'], ['candle-cuda'])
        selected = proof.select((case,), 'all', 'candle-cuda', 'cpu-leg-fixture')
        self.assertEqual(len(selected), 1)
        self.assertEqual(selected[0]['expected_devices'], ['candle-cpu', 'candle-cuda'])
        with self.assertRaises(ValueError):
            proof.select((case,), 'all', 'candle-cuda', 'all-rungs')

    def test_reserved_process_child_permission_credentials_visibility_and_assets_denied(self):
        for name in ('CUDA_VISIBLE_DEVICES', 'CUDA_DEVICE_ORDER', 'MODEL_INFERENCE_CASE_FQN', 'MODEL_PROOF_OUTPUT_DIR',
                     'SCENEWORKS_METAL_EXECUTION_LOCK_FD', 'IRIS_DIST_CHILD', 'IOS_CAPTURE_IN_CHILD', 'ALLOW_GPU',
                     'SCENEWORKS_ALLOW_HOST_METAL', 'MODEL_PERMIT', 'HF_TOKEN', 'API_PASSWORD', 'PATH', 'PYTHONPATH',
                     'RUSTFLAGS', 'CARGO_HOME', 'LD_PRELOAD', 'GITHUB_TOKEN', 'MODEL_WEIGHTS', 'EXTERNAL_MODEL_PATH'):
            case = row(); case['configuration_controls'] = [name]
            for recipe in case['configurations'].values():
                recipe['environment'] = {name: '1'}
            with self.subTest(name=name), self.assertRaises(ValueError):
                proof.validate_configurations(case)
        case = row(); case['external_asset_environment'] = ['CANDLE_LLM_DEVICE']
        with self.assertRaises(ValueError):
            proof.validate_configurations(case)

    def test_required_source_selector_is_not_misclassified_as_an_external_asset(self):
        case = row(); case['required_environment'] = ['CANDLE_LLM_DEVICE']
        case['configurations']['own-stream']['environment']['CANDLE_LLM_DEVICE'] = 'cuda'
        actual = proof.select((case,), 'all', 'candle-cuda', 'own-stream')[0]
        environment = {}
        proof.apply_configuration(actual, environment)
        self.assertEqual(environment['CANDLE_LLM_DEVICE'], 'cuda')
        self.assertTrue(all(environment.get(name) for name in actual['required_environment']))

    def test_shared_source_uses_exact_selected_package_manifest_not_filename_guess(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary).resolve()
            case = row()
            case['source'] = 'crates/contracts/core-llm/core-llm-testkit/baseline/speculative_bench_baseline.rs'
            for recipe in case['configurations'].values():
                recipe['source_identity'] = case['source'] + '::shared-test/fixed-recipe'
            manifest = root / 'crates/llm/candle-llm/Cargo.toml'
            manifest.parent.mkdir(parents=True)
            manifest.write_text('[package]\nname = "candle-llm"\n[features]\nmodel-inference-proof = []\ncuda = []\nflash-attn = []\n', encoding="utf-8")
            with self.assertRaisesRegex(ValueError, 'no actual matching'):
                proof.validate_configurations(case, root=root)
            case['configuration_manifest'] = 'crates/llm/candle-llm/Cargo.toml'
            proof.validate_configurations(case, root=root)
            for invalid in ('../Cargo.toml', str(manifest), 'crates/llm/candle-llm/other.toml'):
                case['configuration_manifest'] = invalid
                with self.subTest(manifest=invalid), self.assertRaises(ValueError):
                    proof.validate_configurations(case, root=root)
            case['configuration_manifest'] = 'crates/llm/candle-llm/Cargo.toml'
            manifest.write_text('[package]\nname = "wrong-package"\n[features]\nmodel-inference-proof = []\ncuda = []\nflash-attn = []\n', encoding="utf-8")
            with self.assertRaisesRegex(ValueError, 'not the declared selected Cargo package'):
                proof.validate_configurations(case, root=root)

    def test_fixed_values_unknown_controls_source_identity_or_default_cannot_be_injected(self):
        mutations = [lambda case: case['configurations']['own-stream']['environment'].update(CANDLE_LLM_DEVICE={'arbitrary': 'cuda'}),
                     lambda case: case['configurations']['own-stream']['environment'].update(CANDLE_LLM_DEVICE=True),
                     lambda case: case['configurations']['own-stream']['environment'].update(CANDLE_LLM_DEVICE='cpu\nALLOW_GPU=1'),
                     lambda case: case['configurations']['own-stream']['environment'].update(UNKNOWN='1'),
                     lambda case: case['configurations']['own-stream'].update(source_identity='different/source.rs::helper'),
                     lambda case: case.update(default_configuration='own-stream', backend='candle-cuda', expected_devices=['candle-cuda'], cargo_features=['model-inference-proof', 'cuda'])]
        for mutate in mutations:
            case = row(); mutate(case)
            with self.subTest(mutation=mutate), self.assertRaises(ValueError):
                proof.validate_configurations(case)
        request = {'selection': 'all', 'policy': {'backend': 'windows-cuda'}, 'configuration': 'own-stream',
                   'environment': {'CANDLE_LLM_DEVICE': 'cuda'}}
        with self.assertRaises(ValueError):
            proof.request(json.dumps(request), 'candle-cuda')

    def test_logical_device_selector_does_not_replace_actual_nn_evidence_or_visibility_authority(self):
        case = row()
        actual = proof.select((case,), 'all', 'candle-cuda', 'own-stream')[0]
        text = ('test result: ok. 1 passed; 0 failed; 0 ignored;\n'
                'MODEL_INFERENCE_BODY_COMPLETED ' + case['fqn'] + '\n'
                'MODEL_INFERENCE_DEVICE ' + case['fqn'] + ' candle-cpu\n')
        with self.assertRaises(ValueError):
            proof.verify_execution(actual, text)
        environment = {'CUDA_VISIBLE_DEVICES': 'approved-controller-uuid', 'CUDA_DEVICE_ORDER': 'controller-order'}
        proof.apply_configuration(actual, environment)
        self.assertEqual(environment['CUDA_VISIBLE_DEVICES'], 'approved-controller-uuid')
        self.assertEqual(environment['CUDA_DEVICE_ORDER'], 'controller-order')


if __name__ == '__main__':
    unittest.main()
