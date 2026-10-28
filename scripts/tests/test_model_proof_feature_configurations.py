"""Feature-only source recipes and compile-host/actual-NN-device separation; no native work."""
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from scripts.ci import model_inference_proofs as proof
from scripts.ci import check_compiled_model_ignores as compiled


def row():
    source = 'crates/media/candle-gen/candle-gen/tests/feature_fixture.rs'
    def recipe(backend, devices, feature=None, route=None):
        value = {'source_identity': source + '::fixture/exact-source-branch', 'environment': {},
                 'backend': backend, 'expected_devices': devices, 'scope': 'source fixture whole case',
                 'required_work': ['learned target and source assertions'],
                 'cargo_features': ['model-inference-proof', *([feature] if feature else [])]}
        if route:
            value['execution_lane'] = route
        return value
    return {'package': 'candle-gen', 'target': 'integration', 'target_kind': 'test', 'fqn': 'feature_fixture::fixture',
            'source': source, 'execution_role': 'direct', 'backend': None, 'expected_devices': [], 'cargo_features': [],
            'cargo_target_metadata': {'name': 'integration', 'kind': ['test'], 'crate_types': ['bin']},
            'fixture_identity': 'metadata protocol fixture; no model exists or runs', 'configuration_source': source + '::fixture',
            'output_contract': 'strict protocol text only', 'configuration_kind': 'feature-only',
            'configuration_controls': [], 'default_configuration': None,
            'configurations': {'cpu': recipe('candle-cpu', ['candle-cpu']),
                               'cuda': recipe('candle-cuda', ['candle-cpu', 'candle-cuda'], 'cuda'),
                               'metal': recipe('candle-metal', ['candle-metal'], 'metal'),
                               'cpu-metal-build': recipe('candle-cpu', ['candle-cpu'], 'metal', 'candle-metal'),
                               'cpu-cuda-build': recipe('candle-cpu', ['candle-cpu'], 'cuda', 'candle-cuda')}}


class FeatureConfigurations(unittest.TestCase):
    def test_no_nn_source_default_refuses_exact_all_and_package_selection_without_omission(self):
        case = row()
        proof.validate_configurations(case)
        for selection in ('all', case['package'], proof.identity(case)):
            with self.subTest(selection=selection), self.assertRaisesRegex(ValueError, 'explicit configuration required'):
                proof.select((case,), selection, 'candle-metal')
        with self.assertRaisesRegex(ValueError, 'explicit configuration required'):
            proof.lane(case)
        for recipe in case['configurations']:
            route = proof.configuration_route(case, case['configurations'][recipe])
            actual = proof.select((case,), proof.identity(case), route, recipe)[0]
            self.assertEqual(proof.identity(actual), proof.identity(case))
        with self.assertRaises(ValueError):
            proof.select((case,), 'all', 'candle-cpu', 'unknown')

    def test_cpu_source_default_remains_real_cpu_without_cuda_compile_or_implicit_alternatives(self):
        case = row(); default = case['configurations']['cpu']
        case.update(default_configuration='cpu', backend=default['backend'], expected_devices=default['expected_devices'], cargo_features=default['cargo_features'])
        actual = proof.select((case,), 'all', 'candle-cpu')[0]
        self.assertEqual(actual['expected_devices'], ['candle-cpu'])
        self.assertNotIn('cuda', actual['cargo_features'])
        environment = {'CANDLE_LLM_DEVICE': 'operator-value', 'MODEL_PATH': 'asset'}
        proof.apply_configuration(actual, environment)
        self.assertEqual(environment, {'CANDLE_LLM_DEVICE': 'operator-value', 'MODEL_PATH': 'asset'})

    def test_gpu_compile_gate_with_cpu_nn_requires_source_bound_execution_lane_and_never_gpu_evidence(self):
        case = row()
        for name, route in (('cpu-metal-build', 'candle-metal'), ('cpu-cuda-build', 'candle-cuda')):
            actual = proof.select((case,), 'all', route, name)[0]
            self.assertEqual(actual['backend'], 'candle-cpu')
            self.assertEqual(actual['expected_devices'], ['candle-cpu'])
            self.assertEqual(proof.lane(actual), route)
            environment = {'ORIGINAL': 'kept'}
            proof.apply_configuration(actual, environment)
            self.assertEqual(environment, {'ORIGINAL': 'kept'})
            text = ('test result: ok. 1 passed; 0 failed; 0 ignored;\n'
                    'MODEL_INFERENCE_BODY_COMPLETED ' + case['fqn'] + '\n'
                    'MODEL_INFERENCE_DEVICE ' + case['fqn'] + ' candle-cpu\n')
            self.assertEqual(proof.verify_execution(actual, text)['actual_devices'], ['candle-cpu'])
            with self.assertRaises(ValueError):
                proof.verify_execution(actual, text.replace(' candle-cpu\n', ' ' + route + '\n'))
            with self.assertRaises(ValueError):
                proof.select((case,), 'all', 'candle-cpu', name)

    def test_impossible_features_lanes_device_sets_and_environment_controls_rejected(self):
        mutations = [lambda case: case['configuration_controls'].append('ALLOW_GPU'),
                     lambda case: case['configurations']['cpu']['environment'].update(CANDLE_LLM_DEVICE='cuda:1'),
                     lambda case: case['configurations']['cpu-metal-build'].pop('execution_lane'),
                     lambda case: case['configurations']['cpu-metal-build'].update(execution_lane='candle-cuda'),
                     lambda case: case['configurations']['metal']['cargo_features'].append('cuda'),
                     lambda case: case['configurations']['cuda']['expected_devices'].append('candle-metal'),
                     lambda case: case['configurations']['cpu']['expected_devices'].append('candle-cuda'),
                     lambda case: case['configurations']['cpu'].update(source_identity='unbound_source'),
                     lambda case: case.update(backend='candle-cpu'),
                     lambda case: case.update(cargo_features=['cuda'])]
        for mutate in mutations:
            case = row(); mutate(case)
            with self.subTest(mutation=mutate), self.assertRaises(ValueError):
                proof.validate_configurations(case)

    def test_actual_manifest_features_all_recipes_and_runtime_override_rejection(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary).resolve()
            case = row()
            source = root / case['source']; source.parent.mkdir(parents=True); source.write_text('// source fixture; not compiled', encoding="utf-8")
            manifest = source.parent.parent / 'Cargo.toml'
            manifest.write_text('[package]\nname = "candle-gen"\n[features]\nmodel-inference-proof = []\ncuda = []\nmetal = []\n', encoding="utf-8")
            registry = root / 'registry.json'; registry.write_text(json.dumps([case]), encoding="utf-8")
            self.assertEqual(proof.load_registry(registry, root=root), (case,))
            case['configurations']['metal']['cargo_features'].append('absent-feature')
            registry.write_text(json.dumps([case]), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, 'absent from actual source manifest'):
                proof.load_registry(registry, root=root)
            case = row(); case['execution_lane'] = 'candle-metal'
            registry.write_text(json.dumps([case]), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, 'never runtime-selected'):
                proof.load_registry(registry, root=root)
        legacy = {'package': 'candle-gen', 'backend': 'candle-cpu', 'execution_lane': 'candle-metal'}
        with self.assertRaises(ValueError):
            proof.lane(legacy)

    def test_no_default_case_still_participates_in_compiler_ignore_guard(self):
        case = row()
        listing = case['fqn'] + ': test\n'
        self.assertEqual(compiled.verify_lists([case], listing, listing)['listed_registered_ignored'], 1)
        with self.assertRaises(ValueError):
            compiled.verify_lists([case], listing, '')
        self.assertEqual(compiled.verify_lists([case], '', '')['cfg_absent_not_runtime_proof'], 1)

    def test_request_cannot_override_execution_lane_or_add_feature_or_environment_authority(self):
        for field, value in (('execution_lane', 'candle-metal'), ('environment', {'ALLOW_GPU': '1'}), ('cargo_features', ['cuda'])):
            request = {'selection': 'all', 'policy': {'backend': 'linux-cpu'}, field: value}
            with self.subTest(field=field), self.assertRaises(ValueError):
                proof.request(json.dumps(request), 'candle-cpu')


if __name__ == '__main__':
    unittest.main()
