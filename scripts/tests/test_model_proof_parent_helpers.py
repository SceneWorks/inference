"""Parent-only Iris metadata/source tests; no subprocess, training, device, or model execution."""
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from scripts.ci import model_inference_proofs as proof
from scripts.ci import check_compiled_model_ignores as compiled
from scripts.ci.model_proof_parent_helpers import IRIS_HELPER, validate_parent_helpers
from scripts import check_model_inference_policy as policy


def rows():
    base = {**IRIS_HELPER, 'cargo_target_metadata': {'name': 'integration', 'kind': ['test'], 'crate_types': ['bin']},
            'execution_role': 'direct', 'backend': 'candle-cpu', 'cargo_features': ['model-inference-proof'],
            'fixture_identity': 'source/metadata fixture; NO NN', 'configuration_source': 'source-fixture',
            'output_contract': 'parent case validation only; libtest success is not trainer success'}
    parents = []
    for name, count in (('ddp_runs_reproduce_bit_for_bit', 5), ('a_rank_that_exits_mid_run_fails_the_survivor_typed', 2)):
        parents.append({**base, 'fqn': 'train_distributed::' + name, 'execution_role': 'nested_rank_direct',
                        'expected_successful_summaries': count})
    helper = {**base, 'parent_only_helper': True, 'parent_bindings': [proof.identity(parent) for parent in parents]}
    return [helper, *parents]


def source_fixture(bindings):
    source = '''
const CHILD_ENV: &str = "IRIS_DIST_CHILD";
#[test]
#[ignore = "the rank-process entry of train_distributed (driven by the scenarios)"]
fn rank_entry() {
    let Ok(raw) = std::env::var(CHILD_ENV) else { return; };
    let j = parse(&raw);
    let result = run_job(&j);
    write(result);
}
fn spawn(mut j: Json, result: &Path) -> Child {
    Command::new(std::env::current_exe().unwrap())
        .args(["--exact", "train_distributed::rank_entry", "--ignored", "--nocapture", "--test-threads", "1"])
        .env(CHILD_ENV, j.to_string()).spawn().unwrap()
}
'''
    for parent in bindings[1:]:
        source += ('#[test]\n#[ignore = "model inference; explicit model-proof workflow"]\nfn '
                   + parent['fqn'].split('::')[-1] + '() {\n'
                   + 'println!("MODEL_INFERENCE_BODY_COMPLETED ' + parent['fqn'] + '");\n'
                   + 'println!("MODEL_INFERENCE_PARENT_COMPLETED ' + parent['fqn'] + '");\n}\n')
    return source


class ParentOnlyHelpers(unittest.TestCase):
    def test_source_bound_relation_is_not_a_standalone_proof_or_new_runtime_role(self):
        bindings = rows()
        relation = validate_parent_helpers(bindings)
        self.assertEqual(relation[proof.identity(bindings[0])], tuple(bindings[0]['parent_bindings']))
        self.assertEqual(set(proof.ROLES), {'direct', 'cpu_child', 'nested_rank_cpu', 'nested_rank_direct', 'indexed_children'})
        self.assertEqual(proof.select(tuple(bindings), 'all', 'candle-cpu'), tuple(bindings[1:]))
        self.assertEqual(proof.select(tuple(bindings), 'candle-gen-iris', 'candle-cpu'), tuple(bindings[1:]))
        for selection in (proof.identity(bindings[0]), 'candle-gen-iris/train_distributed::rank_entry'):
            with self.assertRaisesRegex(ValueError, 'not a standalone proof'):
                proof.select(tuple(bindings), selection, 'candle-cpu')
        with patch.object(proof, 'require_host') as host:
            with self.assertRaises(ValueError):
                proof.run((bindings[0],), 'candle-cpu', {}, '1' * 40, Path('/never-created'))
            host.assert_not_called()
        text = ('test result: ok. 1 passed; 0 failed; 0 ignored;\n'
                'MODEL_INFERENCE_BODY_COMPLETED train_distributed::rank_entry\n'
                'MODEL_INFERENCE_DEVICE train_distributed::rank_entry candle-cpu\n')
        with self.assertRaisesRegex(ValueError, 'cannot claim standalone'):
            proof.verify_execution(bindings[0], text)

    def test_compiler_ignore_guard_still_checks_nonselectable_nn_helper(self):
        bindings = rows()
        listing = IRIS_HELPER['fqn'] + ': test\n'
        result = compiled.verify_lists(bindings, listing, listing)
        self.assertEqual(result['listed_registered_ignored'], 1)
        self.assertEqual(result['model_bodies_executed'], 0)
        with self.assertRaisesRegex(ValueError, 'NOT ignored'):
            compiled.verify_lists(bindings, listing, '')

    def test_missing_extra_duplicate_or_wrong_target_parent_links_are_rejected(self):
        mutations = [lambda bindings: bindings[0]['parent_bindings'].pop(),
                     lambda bindings: bindings[0]['parent_bindings'].append('candle-gen-iris/integration/not_a_parent'),
                     lambda bindings: bindings[0]['parent_bindings'].append(bindings[0]['parent_bindings'][0]),
                     lambda bindings: bindings[1].update(execution_role='direct'),
                     lambda bindings: bindings[1].update(target='other'),
                     lambda bindings: bindings[1].update(cargo_target_metadata={'name': 'other', 'kind': ['test'], 'crate_types': ['bin']}),
                     lambda bindings: bindings[1].update(expected_successful_summaries=0)]
        for mutate in mutations:
            bindings = rows(); mutate(bindings)
            with self.subTest(mutation=mutate), self.assertRaises(ValueError):
                validate_parent_helpers(bindings)

    def test_no_other_source_function_or_package_gets_a_body_exemption(self):
        for key, value in (('package', 'candle-gen-other'), ('source', 'other.rs'), ('fqn', 'other::rank_entry'),
                           ('source_function', 'other'), ('parent_only_helper', 1), ('execution_role', 'indexed_children'),
                           ('body_completion_marker', 'fake'), ('expected_successful_summaries', 1)):
            bindings = rows(); bindings[0][key] = value
            with self.subTest(field=key), self.assertRaises(ValueError):
                validate_parent_helpers(bindings)

    def test_guard_requires_original_ignore_real_child_guard_sameexe_argv_and_no_fake_body(self):
        bindings = rows()
        source = source_fixture(bindings)
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            path = root / IRIS_HELPER['source']; path.parent.mkdir(parents=True)
            path.write_text(source, encoding="utf-8")
            self.assertEqual(policy.registry_errors(root, bindings), [])
            variants = [source.replace('IRIS_DIST_CHILD', 'SOME_OTHER_CHILD'),
                        source.replace('let Ok(raw) = std::env::var(CHILD_ENV) else { return; };', 'let raw = String::new();'),
                        source.replace('run_job(&j)', 'metadata_only(&j)'),
                        source.replace('std::env::current_exe().unwrap()', 'another_program()'),
                        source.replace('"--ignored", ', ''),
                        source.replace('"--exact", ', ''),
                        source.replace('.env(CHILD_ENV, j.to_string())', '.env("unrelated", j.to_string())'),
                        source.replace('#[ignore = "the rank-process entry of train_distributed (driven by the scenarios)"]', ''),
                        source.replace('let result = run_job(&j);', 'let result = run_job(&j); println!("MODEL_INFERENCE_BODY_COMPLETED train_distributed::rank_entry");')]
            for variant in variants:
                path.write_text(variant, encoding="utf-8")
                with self.subTest(source=variant[:100]):
                    self.assertTrue(policy.registry_errors(root, bindings))

    def test_existing_nested_case_counts_all_libtest_success_without_claiming_training_success(self):
        row = rows()[1]
        # Synthetic analogue of CUDA6152: two pre-NN typed-error normal returns,
        # two NN normal returns, plus parent; only case-level BODY/device is asserted.
        row['expected_successful_summaries'] = 5
        text = ('test result: ok. 1 passed; 0 failed; 0 ignored;\n' * 5
                + 'MODEL_INFERENCE_DEVICE ' + row['fqn'] + ' candle-cpu\n'
                + 'MODEL_INFERENCE_BODY_COMPLETED ' + row['fqn'] + '\n'
                + 'MODEL_INFERENCE_PARENT_COMPLETED ' + row['fqn'] + '\n')
        result = proof.verify_execution(row, text)
        self.assertEqual(result['successful_libtest_summaries'], 5)
        self.assertEqual(result['body_completed'], 1)
        for invalid in (text.replace('test result: ok. 1 passed; 0 failed; 0 ignored;\n', '', 1),
                        text + 'test result: FAILED. 0 passed; 1 failed; 0 ignored;\n',
                        text + 'MODEL_INFERENCE_PREREQUISITE_MISSING ' + row['fqn'] + ' CPU guard skipped CUDA work\n',
                        text.replace('MODEL_INFERENCE_PARENT_COMPLETED', 'NO_PARENT')):
            with self.assertRaises(ValueError):
                proof.verify_execution(row, invalid)

    def test_original_source_helper_seam_matches_guard_without_using_on_cpu_as_evidence(self):
        path = ROOT / IRIS_HELPER['source']
        source = path.read_text(encoding="utf-8")
        candidates = [item for item in policy.functions(source) if item['name'] == 'rank_entry']
        self.assertEqual(policy.parent_helper_source_errors(source, candidates), [])
        self.assertNotIn('on_cpu', str(validate_parent_helpers(rows())))


if __name__ == '__main__':
    unittest.main()
