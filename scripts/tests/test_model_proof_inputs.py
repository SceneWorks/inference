"""Actual-input receipt protocol tests; JSON/source fixtures only, never open/hash model inputs."""
import copy
import json
from pathlib import Path
import sys
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from scripts.ci import model_proof_inputs as inputs

READER = 'MOCK::actual_consumption_reader'
DIGEST = 'a' * 64
OTHER = 'b' * 64


def row(kind='external', grade='reader-binding'):
    return {'package': 'candle-gen', 'fqn': 'tests::actual_case', 'execution_role': 'direct',
            'input_contract': {'fixture_kind': kind, 'source_identity': 'source.rs::actual_case',
                'requirements': [] if kind in ('source-defined', 'unresolved') else [
                    {'slot': 'actual-weight-collection', 'role': 'model-weights', 'readers': [READER],
                     'minimum_collections': 1, 'minimum_items': 1, 'required_grade': grade, 'scope': 'case'}]}}


def item(path='/actual/selected.safetensors', kind='successful-reader-binding', digest=DIGEST):
    provenance = {'kind': kind, 'content_integrity': inputs.INTEGRITY[kind]}
    if kind != 'successful-reader-binding':
        provenance['sha256'] = digest
    if kind == 'validated-content':
        provenance.update(verifier='existing-finalized-seal', source_binding='actual retained pre/post path checks; no opened-handle/ABA claim')
    identity = {'display': path, 'os_encoded_hex': path.encode().hex()}
    return {'literal_path': identity, 'absolute_lexical_path': dict(identity), 'os_encoding': 'rust-osstr-encoded', 'provenance': provenance}


def event(value, scope='tests::actual_case', reader=READER, role='model-weights', complete=True):
    packet = {'schema_version': 1, 'reader': reader, 'role': role, 'complete': complete,
              'items': value if isinstance(value, list) else [value]}
    return 'MODEL_INFERENCE_INPUT_USED ' + scope + ' ' + json.dumps(packet, separators=(',', ':')) + '\n'


class InputReceipts(unittest.TestCase):
    def setUp(self):
        # A source-contract fixture, not a claim that any real mmap API hashes consumed bytes.
        roles = patch.dict(inputs.READERS, {READER: 'model-weights'})
        grades = patch.dict(inputs.READER_GRADES, {READER: frozenset(('reader-binding', 'content', 'validated-content'))})
        roles.start(); grades.start()
        self.addCleanup(roles.stop); self.addCleanup(grades.stop)

    def test_flux3_bounded_requires_base_tensors_text_config_and_tokenizer(self):
        registry = json.loads((ROOT / 'scripts/ci/model-inference-proofs.json').read_text(encoding='utf-8'))
        case = next(row for row in registry if row['fqn'] == 'real_weights::flux3_bounded_acceleration_acceptance')
        requirements = case['input_contract']['requirements']
        self.assertEqual({r['slot'] for r in requirements},
                         {'flux3-source-tensors', 'flux3-text-config', 'flux3-tokenizer'})
        events = [event(item('/MOCK-ONLY/' + r['slot']), scope=case['fqn'],
                        reader=r['readers'][0], role=r['role']) for r in requirements]
        self.assertEqual(inputs.assess_inputs(case, ''.join(events))['input_grade'], 'successful-reader-binding')
        for omitted in range(len(events)):
            with self.subTest(omitted=requirements[omitted]['slot']), self.assertRaises(ValueError):
                inputs.assess_inputs(case, ''.join(e for i, e in enumerate(events) if i != omitted))
        # A generated-adapter reload receipt cannot substitute for base source weights.
        reload = event(item('/MOCK-ONLY/adapter.safetensors'), scope=case['fqn'],
                       reader='candle-core::safetensors::load', role='tensor-input')
        with self.assertRaises(ValueError):
            inputs.assess_inputs(case, reload + ''.join(events[1:]))

    def test_source_defined_fixture_remains_source_bound_without_checkpoint_claim(self):
        result = inputs.assess_inputs(row('source-defined'), '')
        self.assertEqual(result['input_grade'], 'source-defined')
        self.assertEqual(result['acceptance_scope'], 'source-bound-case-execution')
        self.assertFalse(result['checkpoint_byte_integrity_attested'])
        self.assertFalse(result['terminal_model_evidence'])

    def test_unknown_or_unresolved_external_contract_never_accepts_source_descriptor_or_any_config(self):
        with self.assertRaises(ValueError):
            inputs.assess_inputs({'fqn': 'tests::actual_case', 'execution_role': 'direct', 'fixture_identity': 'catalogue hash only'}, event(item()))
        with self.assertRaises(ValueError):
            inputs.assess_inputs(row('unresolved'), event(item()))
        case = row(); case['input_contract']['requirements'] = []
        with self.assertRaises(ValueError):
            inputs.assess_inputs(case, event(item()))
        with self.assertRaises(ValueError):
            inputs.assess_inputs(row(), event(item(), reader='gen-core::LoadSpec::read_file_unchanged_if_prepared', role='artifact-input'))

    def test_successful_reader_binding_is_actual_case_evidence_only_not_content_or_named_revision(self):
        result = inputs.assess_inputs(row(), event(item()))
        self.assertEqual(result['input_grade'], 'successful-reader-binding')
        self.assertEqual(result['acceptance_scope'], 'source-bound-case-execution')
        self.assertFalse(result['required_input_content_attested'])
        self.assertFalse(result['checkpoint_byte_integrity_attested'])
        self.assertFalse(result['verified_named_revision'])
        self.assertFalse(result['terminal_model_evidence'])
        self.assertIn('checkpoint-byte-integrity-not-attested', result['input_claim_limits'])
        self.assertEqual(result['actual_inputs'][0]['items'][0]['absolute_lexical_path']['display'], '/actual/selected.safetensors')

    def test_weak_receipt_cannot_fulfil_existing_strong_slot_and_pins_cannot_downgrade(self):
        for grade in ('content', 'validated-content'):
            with self.subTest(grade=grade), self.assertRaises(ValueError):
                inputs.assess_inputs(row(grade=grade), event(item()))
        case = row(); case['input_contract']['requirements'][0]['expected_sha256'] = [DIGEST]
        with self.assertRaises(ValueError):
            inputs.validate_contract(case)

    def test_same_owned_read_buffer_and_existing_finalized_receipt_keep_distinct_grades(self):
        result = inputs.assess_inputs(row(grade='content'), event(item(kind='same-read-buffer')))
        self.assertEqual(result['input_grade'], 'same-read-buffer')
        self.assertTrue(result['required_input_content_attested'])
        self.assertFalse(result['checkpoint_byte_integrity_attested'], 'an observed digest alone is not a pinned named checkpoint')
        case = row(grade='validated-content'); case['input_contract']['requirements'][0]['expected_sha256'] = [DIGEST]
        strong = inputs.assess_inputs(case, event(item(kind='validated-content')))
        self.assertEqual(strong['input_grade'], 'validated-content')
        self.assertTrue(strong['checkpoint_byte_integrity_attested'])
        self.assertFalse(strong['verified_named_revision'])
        with self.assertRaises(ValueError):
            inputs.assess_inputs(case, event(item(kind='validated-content', digest=OTHER)))
        with self.assertRaises(ValueError):
            inputs.assess_inputs(case, event(item(kind='validated-content', digest='0' * 64)))

    def test_config_unknown_reader_wrong_role_and_point_in_time_verification_are_not_weights_use(self):
        for reader, role in (('unknown::reader', 'model-weights'), (READER, 'configuration'),
                             ('candle-audio-yue2::snapshot::verify_component', 'artifact-input')):
            with self.subTest(reader=reader), self.assertRaises(ValueError):
                inputs.assess_inputs(row(), event(item(), reader=reader, role=role))
        case = row(); case['input_contract']['reader_roles'] = {'actual::config_reader': 'configuration'}
        with self.assertRaises(ValueError):
            inputs.assess_inputs(case, event(item(kind='same-read-buffer'), reader='actual::config_reader', role='configuration'))

    def test_actual_reader_catalogue_forbids_fake_mmap_content_or_unknown_declared_reader(self):
        reader = 'candle-core::safetensors::MmapedSafetensors::new'
        self.assertEqual(inputs.READERS[reader], 'tensor-input')
        case = row(); requirement = case['input_contract']['requirements'][0]
        requirement.update(role='tensor-input', readers=[reader])
        with self.assertRaisesRegex(ValueError, 'exceeds actual source reader'):
            inputs.assess_inputs(case, event(item(kind='same-read-buffer'), reader=reader, role='tensor-input'))
        case = row(); case['input_contract']['reader_roles'] = {'not-implemented::reader': 'model-weights'}
        with self.assertRaises(ValueError):
            inputs.validate_contract(case)

    def test_missing_empty_partial_duplicate_collection_or_wrong_case_never_completes_required_inputs(self):
        for text in ('', event([]), event(item(), complete=False), event(item(), scope='tests::other_case'),
                     event([item(), item()])):
            with self.subTest(text=text[:70]), self.assertRaises(ValueError):
                inputs.assess_inputs(row(), text)
        case = row(); case['input_contract']['requirements'][0]['minimum_items'] = 2
        with self.assertRaises(ValueError):
            inputs.assess_inputs(case, event(item()))
        result = inputs.assess_inputs(case, event([item('/actual/shard0'), item('/actual/shard1')]))
        self.assertEqual(len(result['actual_inputs'][0]['items']), 2)

    def test_indexed_inputs_bind_each_actual_child_and_root_cannot_replace_child(self):
        case = row(); case.update(execution_role='indexed_children', expected_child_ids=['0', '1'])
        case['input_contract']['requirements'][0]['scope'] = 'each-child'
        valid = event(item(), scope=case['fqn'] + '#child=0') + event(item(), scope=case['fqn'] + '#child=1')
        result = inputs.assess_inputs(case, valid)
        self.assertEqual({slot['scope'] for slot in result['required_input_slots']}, {case['fqn'] + '#child=0', case['fqn'] + '#child=1'})
        for invalid in (event(item()), event(item(), scope=case['fqn'] + '#child=0'), valid.replace('#child=1', '#child=unexpected')):
            with self.assertRaises(ValueError):
                inputs.assess_inputs(case, invalid)

    def test_no_path_mtime_catalogue_hash_as_content_and_memory_has_no_invented_path(self):
        bad = item(); bad['provenance']['sha256'] = DIGEST
        with self.assertRaises(ValueError):
            inputs.assess_inputs(row(), event(bad))
        bad = item(); bad['provenance']['content_hash'] = DIGEST
        with self.assertRaises(ValueError):
            inputs.assess_inputs(row(), event(bad))
        bad = item(); bad['provenance']['content_integrity'] = 'verified-from-path-mtime'
        with self.assertRaises(ValueError):
            inputs.assess_inputs(row(), event(bad))
        memory = {'origin': 'memory-buffer', 'provenance': {'kind': 'same-read-buffer', 'content_integrity': 'same-buffer-sha256', 'sha256': DIGEST}}
        self.assertEqual(inputs.assess_inputs(row(grade='content'), event(memory))['input_grade'], 'same-read-buffer')
        memory['literal_path'] = {'display': 'invented', 'os_encoded_hex': '00'}
        with self.assertRaises(ValueError):
            inputs.assess_inputs(row(grade='content'), event(memory))

    def test_mixed_source_default_and_consumed_override_use_actual_origin_not_env_or_path(self):
        case = row('mixed'); case['input_contract']['source_defined_default'] = True
        case['input_contract']['requirements'][0]['when'] = 'external-selected'
        empty_default = inputs.assess_inputs(case, '')
        self.assertEqual(empty_default['input_grade'], 'source-defined')
        self.assertEqual(empty_default['required_input_slots'], [])
        default = item(); default['source_origin'] = 'source-defined-default'
        self.assertEqual(inputs.assess_inputs(case, event(default))['input_grade'], 'source-defined')
        override = item(); override['source_origin'] = 'external-selected'
        result = inputs.assess_inputs(case, event(override))
        self.assertEqual(result['input_grade'], 'successful-reader-binding')
        self.assertEqual(result['input_source_branches'][0]['source_origin'], 'external-selected')
        self.assertEqual(len(result['required_input_slots']), 1)
        with self.assertRaises(ValueError):
            inputs.assess_inputs(case, event(item()))  # Same reader cannot infer default from a path.
        bad = item(); bad['source_origin'] = 'cache-default'
        with self.assertRaises(ValueError):
            inputs.assess_inputs(case, event(bad))

    def test_every_consumed_override_is_validated_and_undeclared_external_branch_rejects(self):
        case = row('mixed', grade='content'); case['input_contract']['source_defined_default'] = True
        case['input_contract']['requirements'][0]['when'] = 'external-selected'
        good = item('/actual/A', kind='same-read-buffer'); good['source_origin'] = 'external-selected'
        weak = item('/actual/B'); weak['source_origin'] = 'external-selected'
        with self.assertRaises(ValueError):
            inputs.assess_inputs(case, event(good) + event(weak))
        case['input_contract']['reader_roles'] = {'actual::other_weights': 'model-weights'}
        with self.assertRaises(ValueError):
            inputs.assess_inputs(case, event(good) + event(good, reader='actual::other_weights'))
        bad = row('mixed'); bad['input_contract']['requirements'][0]['when'] = 'external-selected'
        with self.assertRaises(ValueError):
            inputs.validate_contract(bad)

    def test_exact_matching_cannot_reuse_one_collection_for_two_slots_or_greedily_lose_pinned_match(self):
        case = row(grade='content')
        second = copy.deepcopy(case['input_contract']['requirements'][0]); second.update(slot='pinned-second', expected_sha256=[DIGEST])
        case['input_contract']['requirements'].append(second)
        with self.assertRaises(ValueError):
            inputs.assess_inputs(case, event(item(kind='same-read-buffer')))
        text = event(item('/actual/A', kind='same-read-buffer')) + event(item('/actual/B', kind='same-read-buffer', digest=OTHER))
        result = inputs.assess_inputs(case, text)
        self.assertEqual(len(result['required_input_slots']), 2)
        self.assertEqual(result['required_input_slots'][1]['receipt_indices'], [0])


if __name__ == '__main__':
    unittest.main()
